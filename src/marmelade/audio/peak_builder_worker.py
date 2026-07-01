"""QRunnable wrapper around :func:`peak_builder.build_proxy`.

This is the bridge between the Qt-free audio backbone (Plan 02) and the GUI
tier (Plan 03). The :class:`PeakBuilderRunnable` runs on a worker thread via
:class:`PySide6.QtCore.QThreadPool` and reports back exclusively through
:class:`marmelade.concurrency.worker.WorkerSignals` — RESEARCH Pitfall
#4 forbids touching PyQtGraph (or any other GUI widget) from
:meth:`run`. Every call here goes:

    worker thread → ``signals.<x>.emit`` → Qt's queued signal delivery
        → GUI thread slot → widget mutation

The runnable is otherwise a thin shell: cancellation is a
:class:`threading.Event`; progress is a percent-int callback wired straight
into :attr:`WorkerSignals.progress`; success/error/cancel each emit exactly
one of the three terminal signals.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QRunnable, Slot

from marmelade.audio.peak_builder import BuildCancelled, build_proxy
from marmelade.audio.proxy_cache import DEFAULT_SAMPLES_PER_PIXEL
from marmelade.concurrency.worker import WorkerSignals


class PeakBuilderRunnable(QRunnable):
    """Wrap :func:`build_proxy` for execution on :class:`QThreadPool`.

    Lifecycle:
        1. Caller constructs with ``(src_path, dst_path, samples_per_pixel)``.
        2. Caller connects ``self.signals.progress``,
           ``self.signals.finished``, ``self.signals.error``, and
           ``self.signals.cancelled`` to GUI slots.
        3. Caller submits via
           ``QThreadPool.globalInstance().start(runnable)``.
        4. :meth:`run` executes on a pool thread, polling
           :attr:`_cancel_event` between blocks via :meth:`_is_cancelled`
           and reporting progress via :meth:`_on_progress`.
        5. Exactly one terminal signal fires: ``finished`` /
           ``cancelled`` / ``error``.

    Pitfall #4 invariant: :meth:`run` makes ZERO calls to any QtWidgets /
    QtGui method, and zero attribute accesses on widget instances. All
    UI changes happen on the GUI thread via signal slots.
    """

    def __init__(
        self,
        src_path: str | os.PathLike,
        dst_path: str | os.PathLike,
        samples_per_pixel: int = DEFAULT_SAMPLES_PER_PIXEL,
    ) -> None:
        super().__init__()
        # CR-02 (Phase 2 review carry-over) — Python refcount owns the
        # lifetime, not QThreadPool's autoDelete-after-run. MainWindow
        # stores the runnable in `self._current_runnable` and later
        # dereferences `runnable.signals` via the stored connection
        # tokens; the C++ runnable must outlive run() so that access
        # remains safe. See EnergyHeatmapRunnable.__init__ for the full
        # rationale.
        self.setAutoDelete(False)
        self.src_path: Path = Path(src_path)
        self.dst_path: Path = Path(dst_path)
        self.samples_per_pixel: int = int(samples_per_pixel)
        self.signals: WorkerSignals = WorkerSignals()
        self._cancel_event: threading.Event = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation. Idempotent — calling twice is a no-op.

        The worker polls the underlying :class:`threading.Event` between
        every source block, so worst-case latency is one block (≈ 3 s at
        ``BLOCK_SAMPLES=131_072`` and 44.1 kHz). :func:`build_proxy`
        deletes any partial ``.tmp`` sibling before re-raising
        :class:`BuildCancelled`.
        """
        self._cancel_event.set()

    def _on_progress(self, pct: int) -> None:
        """Bridge :func:`build_proxy`'s ``progress_cb`` to the Qt signal.

        Runs on the worker thread; the queued signal delivery hops the
        ``int`` payload over to the GUI thread.
        """
        self.signals.progress.emit(int(pct))

    def _is_cancelled(self) -> bool:
        """Bridge :func:`build_proxy`'s ``cancel_check`` to the event."""
        return self._cancel_event.is_set()

    @Slot()
    def run(self) -> None:
        """Worker entry point. Exactly one terminal signal fires per call.

        Order of try/except matters: :class:`BuildCancelled` derives from
        :class:`RuntimeError` (and ultimately :class:`Exception`), so we
        catch it BEFORE the broad ``except Exception`` fallthrough —
        cancellation is not an error.
        """
        try:
            build_proxy(
                self.src_path,
                self.dst_path,
                self.samples_per_pixel,
                progress_cb=self._on_progress,
                cancel_check=self._is_cancelled,
            )
            self.signals.finished.emit(self.dst_path)
        except BuildCancelled:
            # Not an error — the user pressed "Stop building proxy" and
            # build_proxy already cleaned up its <dst>.tmp sibling.
            self.signals.cancelled.emit()
        except Exception as e:
            # UI-SPEC §Copywriting "Couldn't open file" dialog body uses
            # this string verbatim. Prefer str(e) (one human-readable line)
            # over repr(e) (which adds the class name as noise).
            msg = str(e) if str(e) else type(e).__name__
            self.signals.error.emit(msg)
