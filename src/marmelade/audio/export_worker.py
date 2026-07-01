"""QRunnable wrapper around :func:`export_builder.export_region` (Plan 03-04b — EXP-01).

Structural 1:1 mirror of
:class:`marmelade.audio.audio_proxy_worker.AudioProxyRunnable`. The
runnable runs on a worker thread via :class:`QThreadPool` and reports
back exclusively through
:class:`marmelade.concurrency.worker.WorkerSignals`.

Pitfall #4 invariant: :meth:`ExportRunnable.run` makes ZERO calls to any
QtWidgets / QtGui / PyQtGraph method. All cross-thread communication is
via signals.

D-16 invariant: ``WorkerSignals`` is REUSED VERBATIM — DO NOT subclass.

D-17 invariant: cancellation uses the shared
:class:`marmelade.audio.peak_builder.BuildCancelled` exception class.

CR-02 invariant: ``setAutoDelete(False)`` opts out of QRunnable's default
auto-deletion-after-run; MainWindow stores the runnable on
``self._current_export_runnable`` and later dereferences
``runnable.signals`` via stored ``_mw_export_conn_*`` tokens.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QRunnable, Slot

from marmelade.audio.export_builder import export_region
from marmelade.audio.peak_builder import BuildCancelled  # shared (D-17)
from marmelade.concurrency.worker import WorkerSignals  # REUSE VERBATIM (D-16)


__all__ = ["ExportRunnable", "BuildCancelled"]


class ExportRunnable(QRunnable):
    """Wrap :func:`export_region` for execution on :class:`QThreadPool`.

    Lifecycle (IDENTICAL to
    :class:`marmelade.audio.audio_proxy_worker.AudioProxyRunnable`):

        1. Caller constructs with
           ``(proxy_path, dst_path, start_frame, end_frame, fade_frames,
           fmt, sample_rate)``.
        2. Caller connects ``self.signals.{progress,finished,error,
           cancelled}`` to GUI slots (storing the
           :class:`QMetaObject.Connection` tokens for targeted disconnect
           during cancel-restart).
        3. Caller submits via ``QThreadPool.globalInstance().start(runnable)``.
        4. :meth:`run` executes on a pool thread, polling
           :attr:`_cancel_event` between blocks via :meth:`_is_cancelled`
           and reporting progress via :meth:`_on_progress`.
        5. Exactly ONE terminal signal fires: ``finished`` / ``cancelled`` /
           ``error``.
    """

    def __init__(
        self,
        proxy_path: str | os.PathLike,
        dst_path: str | os.PathLike,
        start_frame: int,
        end_frame: int,
        fade_frames: int,
        fmt: str,
        sample_rate: int,
        *,
        source_path: str | os.PathLike | None = None,
    ) -> None:
        super().__init__()
        # CR-02 — Python refcount owns the lifetime.
        self.setAutoDelete(False)
        self.proxy_path: Path = Path(proxy_path)
        self.dst_path: Path = Path(dst_path)
        self.start_frame: int = int(start_frame)
        self.end_frame: int = int(end_frame)
        self.fade_frames: int = int(fade_frames)
        self.fmt: str = str(fmt)
        self.sample_rate: int = int(sample_rate)
        # Phase 7 Plan 07-06 D-20 — keyword-only source_path override.
        # When provided, ExportRunnable forwards it to ``export_region``
        # which then reads audio from this path instead of ``proxy_path``.
        # Phase 3 callers (no source_path) get identical behavior.
        self.source_path: Path | None = (
            Path(source_path) if source_path is not None else None
        )
        # quick-260621-gfq — export-time normalize params removed. Normalize is
        # now the mastering chain's final stage; raw export never normalizes.
        # REUSE VERBATIM — DO NOT subclass per D-16.
        self.signals: WorkerSignals = WorkerSignals()
        self._cancel_event: threading.Event = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation. Idempotent."""
        self._cancel_event.set()

    def _on_progress(self, pct: int) -> None:
        self.signals.progress.emit(int(pct))

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    @Slot()
    def run(self) -> None:
        """Exactly one terminal signal fires per call.

        ``BuildCancelled`` derives from ``RuntimeError`` (and thus
        ``Exception``); the cancellation branch MUST come before the
        broad ``except Exception`` so cancel is never reported as an
        error.
        """
        try:
            export_region(
                proxy_path=self.proxy_path,
                dst_path=self.dst_path,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                fade_frames=self.fade_frames,
                fmt=self.fmt,
                sample_rate=self.sample_rate,
                progress_cb=self._on_progress,
                cancel_check=self._is_cancelled,
                source_path=self.source_path,
            )
            self.signals.finished.emit(str(self.dst_path))
        except BuildCancelled:
            # Defensive second-pass .tmp unlink — export_builder already
            # removed it on cancel, but we mirror the audio_proxy_worker
            # discipline so a race past the builder's except block can't
            # leave debris. The export_builder uses ``<stem>.tmp<ext>``
            # so the codec dispatch by extension still lands on the right
            # writer; mirror that naming here.
            tmp = self.dst_path.with_name(
                self.dst_path.stem + ".tmp" + self.dst_path.suffix
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            self.signals.cancelled.emit()
        except Exception as e:
            tmp = self.dst_path.with_name(
                self.dst_path.stem + ".tmp" + self.dst_path.suffix
            )
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            msg = str(e) if str(e) else type(e).__name__
            self.signals.error.emit(msg)
