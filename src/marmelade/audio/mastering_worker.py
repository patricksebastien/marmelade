"""``MasteringRunnable`` QRunnable — Phase 7 mastered-WAV render-and-cache worker.

Line-for-line structural peer of
:class:`marmelade.audio.heatmap_worker.EnergyHeatmapRunnable` (D-18).
Reuses :class:`marmelade.concurrency.worker.WorkerSignals` VERBATIM
(D-16 — no subclass into a mastering-specific signals object).

Pipeline (see :meth:`run`):
    1. ``soundfile.read`` the source proxy WAV (already 44.1 kHz stereo
       float32 from Phase 2.1).
    2. Build :class:`MasteringChain` with the per-keeper config, inject
       ``_cancel_check`` and the throttled stage-progress callback.
    3. ``chain.process(audio, sr)`` → float32 numpy output.
    4. Atomic write: ``soundfile.write(<dst>.tmp, ..., subtype="PCM_24")``
       then ``os.replace`` into ``dst_cache_path``.
    5. Emit ``finished(str(dst_cache_path))``.

Terminal-signal contract: exactly one of ``finished`` / ``cancelled`` /
``error`` fires per call. ``BuildCancelled`` is caught BEFORE the broad
``Exception`` fallthrough (it derives from ``RuntimeError`` and would
otherwise be swallowed).

Throttle: progress emits at-most-once-per-100 ms, except the explicit
0 / 5 / 100 bookends and stage-boundary emits from the chain (which
the throttle gate also enforces).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QRunnable, Slot

from marmelade.audio.mastering.chain import MasteringChain
from marmelade.audio.peak_builder import BuildCancelled
from marmelade.concurrency.worker import WorkerSignals

logger = logging.getLogger(__name__)


class MasteringRunnable(QRunnable):
    """Render a per-keeper mastered WAV to the mastered-cache atomically.

    Constructor args:
        src_proxy_path: Source audio path (the Phase 2.1 canonical
            44.1 kHz stereo float32 proxy WAV).
        dst_cache_path: Destination path under the mastered cache —
            resolve via :func:`marmelade.audio.mastering_cache.mastered_cache_path`.
        keeper_id: 32-hex keeper UUID. Stored on the runnable for
            UI-side lookup (the worker itself does not consult it; the
            keeper_id has already been baked into the dst_cache_path).
        mastering_cfg: The per-keeper mastering config dict (the
            sidecar snapshot — D-04). Passed verbatim to
            :class:`MasteringChain`.
        start_frame: Optional keyword-only — first source-proxy frame
            to master (inclusive). When ``None`` (default), masters
            from the start of the source.
        end_frame: Optional keyword-only — one past the last
            source-proxy frame to master (exclusive). When ``None``
            (default), masters through the end of the source.

    Region kwargs (Plan 07-08):
        When both ``start_frame`` and ``end_frame`` are ``None``
        (default), the runnable masters the ENTIRE source — fully
        backward-compatible with pre-Plan-07-08 callers.

        When BOTH are set, the runnable masters ONLY
        ``audio[:, start_frame:end_frame]`` (in source-proxy frame
        coordinates). Slicing happens AFTER ``sf.read`` and BEFORE the
        chain executes, so the chain still sees the standard
        ``(channels, samples)`` float32 layout.

        Both must be either omitted or specified together — passing
        only one raises ``ValueError`` at ``run()`` time (surfaced via
        the ``error`` signal). Inverted or out-of-source bounds also
        raise ``ValueError`` → ``error`` signal.

    Lifecycle (IDENTICAL to
    :class:`marmelade.audio.heatmap_worker.EnergyHeatmapRunnable`):
        1. Caller constructs + connects ``self.signals.*`` to GUI slots.
        2. Caller submits via ``QThreadPool.globalInstance().start(runnable)``.
        3. :meth:`run` executes on a pool thread, polling
           :attr:`_cancel_event` between chain stages.
        4. Exactly one terminal signal fires: ``finished`` / ``cancelled``
           / ``error``.
    """

    def __init__(
        self,
        src_proxy_path: str | os.PathLike,
        dst_cache_path: str | os.PathLike,
        keeper_id: str,
        mastering_cfg: dict,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> None:
        super().__init__()
        # CR-02 — Python refcount owns lifetime, not QThreadPool's
        # autoDelete. Same rationale as EnergyHeatmapRunnable (the
        # MainWindow dict reference keeps the WorkerSignals QObject
        # valid for late ``_disconnect_*_tokens`` calls).
        self.setAutoDelete(False)

        self._src_proxy_path: Path = Path(src_proxy_path)
        self._dst_cache_path: Path = Path(dst_cache_path)
        self.keeper_id: str = keeper_id
        self._mastering_cfg: dict = mastering_cfg

        # Plan 07-08 — keyword-only region kwargs. Validation deferred
        # to run() so a bad-bounds construction surfaces as a normal
        # worker error (via the existing broad-except in run()), not as
        # a raise in __init__ that would propagate on the GUI thread.
        self._start_frame: int | None = start_frame
        self._end_frame: int | None = end_frame

        # REUSE VERBATIM — DO NOT subclass per D-16. The 4-signal
        # contract (progress / finished / error / cancelled) is the
        # cross-worker invariant — identical class identity across
        # every Marmelade QRunnable. Test
        # ``test_runnable_uses_workersignals_verbatim`` pins this.
        self.signals: WorkerSignals = WorkerSignals()
        self._cancel_event: threading.Event = threading.Event()

        # Throttle bookkeeping for stage-progress emits (one per 100 ms
        # plus the 0 / 100 bookends; mirrors EnergyHeatmapRunnable's
        # last-pct discipline).
        self._last_progress_emit_ts: float = 0.0
        self._last_progress_pct: int = -1

    # ----- public API mirroring EnergyHeatmapRunnable -----

    def cancel(self) -> None:
        """Request cooperative cancellation. Idempotent.

        The chain orchestrator polls the underlying
        :class:`threading.Event` between stages (RESEARCH §Pitfall 5 —
        pedalboard chains are not interruptible mid-call). Worst-case
        latency is one stage's pedalboard pass.
        """
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        """Bridge :meth:`MasteringChain.process`'s cancel check to the event."""
        return self._cancel_event.is_set()

    # ----- internal helpers -----

    def _emit_progress_throttled(self, pct: int) -> None:
        """Forward ``pct`` to ``self.signals.progress`` with rate limiting.

        Rule: always emit ``0`` and ``100``; otherwise drop emits that
        come within 100 ms of the previous emit. Mirrors the
        EnergyHeatmapRunnable progress-throttle pattern.
        """
        pct = int(pct)
        if pct == self._last_progress_pct:
            return
        now = time.monotonic()
        is_bookend = pct in (0, 100)
        if not is_bookend and (now - self._last_progress_emit_ts) < 0.1:
            return
        self._last_progress_emit_ts = now
        self._last_progress_pct = pct
        self.signals.progress.emit(pct)

    # ----- worker entry point -----

    @Slot()
    def run(self) -> None:  # noqa: C901 — flat try/except per heatmap_worker.py
        """Worker entry point. Exactly one terminal signal fires per call.

        Order of try/except matters: :class:`BuildCancelled` derives
        from :class:`RuntimeError`, so we catch it BEFORE the broad
        ``except Exception`` fallthrough — cancellation is not an
        error.
        """
        tmp = Path(str(self._dst_cache_path) + ".tmp")
        try:
            # Early cancel check before any I/O.
            if self._is_cancelled():
                raise BuildCancelled()

            # (1) Read the source proxy. soundfile returns
            # (samples, channels); pedalboard wants (channels, samples).
            audio, sr = sf.read(
                str(self._src_proxy_path), dtype="float32", always_2d=True
            )
            audio = np.ascontiguousarray(audio.T)

            # Plan 07-08 — region slice. The audio array is currently
            # (channels, total_frames); slicing happens on axis=1 so
            # the channel axis is preserved. Slice BEFORE chain.process
            # so the chain sees only the region the keeper covers and
            # the cache file's frame count matches the keeper region.
            total_frames = int(audio.shape[1])
            sf_partial = (self._start_frame is None) ^ (
                self._end_frame is None
            )
            if sf_partial:
                # Exactly one of the two is set — contract violation.
                raise ValueError(
                    "MasteringRunnable: start_frame and end_frame must "
                    "be set together (or both omitted)."
                )
            if self._start_frame is not None and self._end_frame is not None:
                start = int(self._start_frame)
                end = int(self._end_frame)
                if not (0 <= start < end <= total_frames):
                    raise ValueError(
                        f"MasteringRunnable: invalid region "
                        f"[{start},{end}) for source of "
                        f"{total_frames} frames."
                    )
                audio = np.ascontiguousarray(audio[:, start:end])

            # Bookend emit: source loaded.
            self._emit_progress_throttled(5)

            if self._is_cancelled():
                raise BuildCancelled()

            # (2) Build chain + inject cancel + stage-progress callback.
            chain = MasteringChain(self._mastering_cfg)
            chain._cancel_check = self._is_cancelled  # type: ignore[attr-defined]
            chain._stage_progress_cb = self._emit_progress_throttled  # type: ignore[attr-defined]

            # (3) Render.
            out = chain.process(audio, sr)

            # (4) Atomic write — write to <dst>.tmp, then os.replace.
            #     soundfile expects (samples, channels) — transpose back.
            tmp.parent.mkdir(parents=True, exist_ok=True)
            # Defensive shape coercion (out should be (channels, samples) float32).
            if out.ndim == 2:
                samples_first = out.T
            else:
                samples_first = out
            # NOTE: pass ``format="WAV"`` explicitly — soundfile cannot
            # infer the format from the ``.wav.tmp`` suffix, so without
            # this kwarg the write would fail with "No format specified
            # and unable to get format from file extension" (caught by
            # the broad except and surfaced as a worker error). The
            # final ``os.replace`` puts the file at the correct
            # ``.wav`` name, but the write itself happens against the
            # ``.tmp`` sibling so the kwarg must be explicit.
            sf.write(str(tmp), samples_first, sr, subtype="PCM_24", format="WAV")
            os.replace(str(tmp), str(self._dst_cache_path))

            # Bookend emit: rendered + cached.
            # (Force the 100% emit past the throttle gate.)
            self._last_progress_pct = -1
            self._emit_progress_throttled(100)
            self.signals.finished.emit(str(self._dst_cache_path))
        except BuildCancelled:
            # Not an error — clean up the partial .tmp.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            self.signals.cancelled.emit()
        except Exception as e:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            logger.exception(
                "MasteringRunnable failed: keeper_id=%s src=%s dst=%s "
                "start_frame=%s end_frame=%s",
                self.keeper_id,
                self._src_proxy_path,
                self._dst_cache_path,
                self._start_frame,
                self._end_frame,
            )
            msg = str(e) if str(e) else type(e).__name__
            self.signals.error.emit(msg)


__all__ = ["MasteringRunnable"]
