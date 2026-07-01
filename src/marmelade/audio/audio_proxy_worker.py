"""QRunnable wrapper around :func:`audio_proxy_builder.build_audio_proxy` (Plan 02.1-03 — D-16).

Structural 1:1 mirror of :class:`marmelade.audio.peak_builder_worker.PeakBuilderRunnable`
and :class:`marmelade.audio.heatmap_worker.EnergyHeatmapRunnable`. This is the
bridge between the Qt-free audio backbone (:mod:`marmelade.audio.audio_proxy_builder`)
and the GUI tier — the runnable runs on a worker thread via
:class:`PySide6.QtCore.QThreadPool` and reports back exclusively through
:class:`marmelade.concurrency.worker.WorkerSignals`. Every call here goes:

    worker thread → ``signals.<x>.emit`` → Qt's queued signal delivery
        → GUI thread slot → widget mutation

Pitfall #4 invariant: :meth:`AudioProxyRunnable.run` makes ZERO calls to any
QtWidgets / QtGui / PyQtGraph method, and zero attribute accesses on widget
instances. All UI changes happen on the GUI thread via signal slots.

D-16 invariant: ``WorkerSignals`` is REUSED VERBATIM — DO NOT subclass into
an audio-proxy-specific signals object. The 4-signal contract (``progress`` /
``finished`` / ``error`` / ``cancelled``) is the cross-worker invariant that
Plan 02.1-04's MainWindow integration depends on (it disconnects via the
same ``_mw_*_conn_*`` token shape it already uses for peak/heatmap workers).

D-17 invariant: cancellation uses the shared
:class:`marmelade.audio.peak_builder.BuildCancelled` exception class — the
same one raised by ``peak_builder.build_proxy`` and re-exported through
``audio_proxy_builder``. There is NO ``AudioProxyCancelled``. The exception
is caught BEFORE the broad ``except Exception`` fallthrough because
``BuildCancelled`` is a ``RuntimeError`` and would otherwise be swallowed by
the error branch (cancellation is not an error).

CR-02 invariant: ``setAutoDelete(False)`` opts out of QRunnable's default
auto-deletion-after-run. MainWindow stores the runnable on
``self._current_proxy_runnable`` and later dereferences ``runnable.signals``
via stored ``_mw_proxy_conn_*`` connection tokens during the cancel-restart
preamble; the C++ runnable must outlive ``run()`` so that access remains
safe. See :class:`PeakBuilderRunnable` lines 60-67 for the full rationale.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QRunnable, Slot

from marmelade.audio.audio_proxy_builder import build_audio_proxy
from marmelade.audio.peak_builder import BuildCancelled  # shared cancellation type (D-17)
from marmelade.concurrency.worker import WorkerSignals  # REUSE VERBATIM — DO NOT subclass (D-16)


__all__ = ["AudioProxyRunnable", "BuildCancelled"]


class AudioProxyRunnable(QRunnable):
    """Wrap :func:`build_audio_proxy` for execution on :class:`QThreadPool`.

    Lifecycle (IDENTICAL to :class:`PeakBuilderRunnable` / :class:`EnergyHeatmapRunnable`):
        1. Caller constructs with ``(src_path, dst_path)``.
        2. Caller connects ``self.signals.progress``, ``self.signals.finished``,
           ``self.signals.error``, and ``self.signals.cancelled`` to GUI slots
           (storing the returned ``QMetaObject.Connection`` tokens for targeted
           disconnect during cancel-restart — see Plan 02.1-04).
        3. Caller submits via ``QThreadPool.globalInstance().start(runnable)``.
        4. :meth:`run` executes on a pool thread, polling :attr:`_cancel_event`
           between blocks via :meth:`_is_cancelled` and reporting progress via
           :meth:`_on_progress`.
        5. Exactly one terminal signal fires: ``finished`` / ``cancelled`` /
           ``error``.

    Pitfall #4 invariant: :meth:`run` makes ZERO calls to any QtWidgets /
    QtGui / PyQtGraph method, and zero attribute accesses on widget
    instances. All UI changes happen on the GUI thread via signal slots.
    """

    def __init__(
        self,
        src_path: str | os.PathLike,
        dst_path: str | os.PathLike,
    ) -> None:
        super().__init__()
        # CR-02 — Python refcount owns the lifetime, not QThreadPool's
        # autoDelete-after-run. With autoDelete=True (the QRunnable default)
        # the C++ QRunnable is deleted as soon as run() returns; the Python
        # wrapper survives, but accessing the runnable's QObject children
        # (notably ``self.signals``, a WorkerSignals QObject) after that
        # deletion is a use-after-free in C++ terms even if sip currently
        # papers over it. MainWindow stores the runnable in
        # ``self._current_proxy_runnable`` and later calls a disconnect
        # path that dereferences ``runnable.signals`` — that needs the C++
        # runnable to still be alive. autoDelete=False lets us own the
        # lifetime via the MainWindow attribute reference; the runnable is
        # freed when we drop that reference and let Python GC it.
        # MIRRORS peak_builder_worker.py lines 60-67 EXACTLY.
        self.setAutoDelete(False)
        self.src_path: Path = Path(src_path)
        self.dst_path: Path = Path(dst_path)
        # REUSE VERBATIM — DO NOT subclass per D-16. The 4-signal contract
        # (progress / finished / error / cancelled) is the cross-worker
        # invariant.
        self.signals: WorkerSignals = WorkerSignals()
        self._cancel_event: threading.Event = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation. Idempotent — calling twice is a no-op.

        The worker polls the underlying :class:`threading.Event` between every
        source block, so worst-case latency is one block (≈ 3 s at
        ``BLOCK_SAMPLES=131_072`` and 44.1 kHz). :func:`build_audio_proxy`
        deletes any partial ``<dst>.tmp`` sibling before re-raising
        :class:`BuildCancelled`; the :meth:`run` exception handler below
        defensively unlinks any remaining ``.tmp`` a second time before
        emitting ``cancelled``.
        """
        self._cancel_event.set()

    def _on_progress(self, pct: int) -> None:
        """Bridge :func:`build_audio_proxy`'s ``progress_cb`` to the Qt signal.

        Runs on the worker thread; the queued signal delivery hops the
        ``int`` payload over to the GUI thread.
        """
        self.signals.progress.emit(int(pct))

    def _is_cancelled(self) -> bool:
        """Bridge :func:`build_audio_proxy`'s ``cancel_check`` to the event."""
        return self._cancel_event.is_set()

    @Slot()
    def run(self) -> None:
        """Worker entry point. Exactly one terminal signal fires per call.

        Order of try/except matters: :class:`BuildCancelled` derives from
        :class:`RuntimeError` (and ultimately :class:`Exception`), so we
        catch it BEFORE the broad ``except Exception`` fallthrough —
        cancellation is not an error.

        Defensive ``.tmp`` cleanup on both the ``BuildCancelled`` and
        ``Exception`` branches mirrors :mod:`heatmap_worker` (lines
        156-160 / 162-168): the builder itself already removes its
        ``<dst>.tmp`` on cancel (D-11 / D-17), but a best-effort second
        unlink here keeps the cache directory clean in the unlikely event
        that cancel raced past the builder's own ``except`` block.
        """
        try:
            build_audio_proxy(
                self.src_path,
                self.dst_path,
                progress_cb=self._on_progress,
                cancel_check=self._is_cancelled,
            )
            # signals.finished payload contract: the proxy WAV path (str).
            # MainWindow's _on_audio_proxy_finished consumes the string to
            # prime playback. Mirrors heatmap_worker.py line 147
            # (emit ``str(self.dst_path)``).
            self.signals.finished.emit(str(self.dst_path))
        except BuildCancelled:
            # Not an error — the user opened another file or quit. The
            # builder already removed the .tmp; defensive double-cleanup
            # mirrors heatmap_worker.py lines 156-160.
            tmp = Path(str(self.dst_path) + ".tmp")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            self.signals.cancelled.emit()
        except Exception as e:
            # Best-effort cleanup of any partial .tmp before reporting the
            # error to the UI. UI-SPEC §Copywriting "Couldn't open file"
            # dialog body uses this string verbatim — prefer str(e) (one
            # human-readable line) over repr(e) (which adds the class name
            # as noise). Fall back to type name when str(e) is empty.
            tmp = Path(str(self.dst_path) + ".tmp")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            msg = str(e) if str(e) else type(e).__name__
            self.signals.error.emit(msg)
