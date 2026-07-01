"""QRunnable wrapper around :func:`spectral_builder.build_spectral_proxy` (Plan 11-05 — R-3).

Structural 1:1 mirror of :class:`marmelade.audio.audio_proxy_worker.AudioProxyRunnable`
(and, transitively, :class:`marmelade.audio.peak_builder_worker.PeakBuilderRunnable` /
:class:`marmelade.audio.heatmap_worker.EnergyHeatmapRunnable`). This is the bridge
between the Qt-free spectral backbone (:mod:`marmelade.audio.spectral_builder`) and
the GUI tier — the runnable runs on a worker thread via
:class:`PySide6.QtCore.QThreadPool` and reports back exclusively through
:class:`marmelade.concurrency.worker.WorkerSignals`. Every call here goes:

    worker thread → ``signals.<x>.emit`` → Qt's queued signal delivery
        → GUI thread slot → widget mutation

Pitfall #4 invariant: :meth:`SpectralProxyRunnable.run` makes ZERO calls to any
QtWidgets / QtGui / PyQtGraph method, and zero attribute accesses on widget
instances. All UI changes happen on the GUI thread via signal slots.

D-16 invariant: ``WorkerSignals`` is REUSED VERBATIM — DO NOT subclass into a
spectral-specific signals object. The 4-signal contract (``progress`` /
``finished`` / ``error`` / ``cancelled``) is the cross-worker invariant that
MainWindow's integration (plan 11-07) depends on (it disconnects via the same
``_mw_*_conn_*`` token shape it already uses for peak/heatmap/audio-proxy
workers).

Cancellation uses the shared
:class:`marmelade.audio.peak_builder.BuildCancelled` exception class — the same
one raised by ``spectral_builder.build_spectral_proxy``. There is NO
``SpectralCancelled``. The exception is caught BEFORE the broad
``except Exception`` fallthrough because ``BuildCancelled`` is a
``RuntimeError`` and would otherwise be swallowed by the error branch
(cancellation is not an error).

CR-02 invariant: ``setAutoDelete(False)`` opts out of QRunnable's default
auto-deletion-after-run. MainWindow stores the runnable and later dereferences
``runnable.signals`` via stored connection tokens during the cancel-restart
preamble; the C++ runnable must outlive ``run()`` so that access remains safe.
See :class:`AudioProxyRunnable` for the full rationale.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QRunnable, Slot

from marmelade.audio.peak_builder import BuildCancelled  # shared cancellation type
from marmelade.audio.spectral_builder import build_spectral_proxy
from marmelade.concurrency.worker import WorkerSignals  # REUSE VERBATIM — DO NOT subclass (D-16)


__all__ = ["SpectralProxyRunnable", "BuildCancelled"]


class SpectralProxyRunnable(QRunnable):
    """Wrap :func:`build_spectral_proxy` for execution on :class:`QThreadPool`.

    Lifecycle (IDENTICAL to :class:`AudioProxyRunnable`):
        1. Caller constructs with ``(src_path, cache_root)``.
        2. Caller connects ``self.signals.progress``, ``self.signals.finished``,
           ``self.signals.error``, and ``self.signals.cancelled`` to GUI slots
           (storing the returned ``QMetaObject.Connection`` tokens for targeted
           disconnect during cancel-restart).
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
        cache_root: str | os.PathLike,
    ) -> None:
        super().__init__()
        # CR-02 — Python refcount owns the lifetime, not QThreadPool's
        # autoDelete-after-run. With autoDelete=True (the QRunnable default)
        # the C++ QRunnable is deleted as soon as run() returns; accessing the
        # runnable's QObject children (notably ``self.signals``, a
        # WorkerSignals QObject) after that deletion is a use-after-free in
        # C++ terms. MainWindow stores the runnable and later dereferences
        # ``runnable.signals`` during the cancel preamble — that needs the C++
        # runnable to still be alive. autoDelete=False lets us own the
        # lifetime via the stored reference. MIRRORS audio_proxy_worker.py
        # lines 82-95 EXACTLY.
        self.setAutoDelete(False)
        self.src_path: Path = Path(src_path)
        self.cache_root: Path = Path(cache_root)
        # REUSE VERBATIM — DO NOT subclass per D-16. The 4-signal contract
        # (progress / finished / error / cancelled) is the cross-worker
        # invariant.
        self.signals: WorkerSignals = WorkerSignals()
        self._cancel_event: threading.Event = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation. Idempotent — calling twice is a no-op.

        The worker polls the underlying :class:`threading.Event` at the top of
        every source block, so worst-case latency is one block.
        :func:`build_spectral_proxy` removes any partial ``*.dat.tmp`` siblings
        before re-raising :class:`BuildCancelled` (T-11-03), so a cancelled
        build leaves no partial ``.dat``.
        """
        self._cancel_event.set()

    def _on_progress(self, pct: int) -> None:
        """Bridge :func:`build_spectral_proxy`'s ``progress_cb`` to the Qt signal.

        Runs on the worker thread; the queued signal delivery hops the
        ``int`` payload over to the GUI thread.
        """
        self.signals.progress.emit(int(pct))

    def _is_cancelled(self) -> bool:
        """Bridge :func:`build_spectral_proxy`'s ``cancel_check`` to the event."""
        return self._cancel_event.is_set()

    @Slot()
    def run(self) -> None:
        """Worker entry point. Exactly one terminal signal fires per call.

        Order of try/except matters: :class:`BuildCancelled` derives from
        :class:`RuntimeError` (and ultimately :class:`Exception`), so we catch
        it BEFORE the broad ``except Exception`` fallthrough — cancellation is
        not an error.

        The builder itself already removes any partial ``*.dat.tmp`` on cancel
        (T-11-03); no extra cleanup is needed here because the spectral write
        path is atomic and self-cleaning.
        """
        try:
            build_spectral_proxy(
                self.src_path,
                self.cache_root,
                progress_cb=self._on_progress,
                cancel_check=self._is_cancelled,
            )
            # signals.finished payload contract: the cache root (str).
            # MainWindow locates the freshly-written spectral proxy under
            # ``<cache_root>/spectra/<key>/`` from this string. Mirrors
            # audio_proxy_worker.py line 156 (emit ``str(...)``).
            self.signals.finished.emit(str(self.cache_root))
        except BuildCancelled:
            # Not an error — the user opened another file or quit. The builder
            # already removed any partial ``*.dat.tmp`` (T-11-03).
            self.signals.cancelled.emit()
        except Exception as e:
            # UI-SPEC §Copywriting "Couldn't open file" dialog body uses this
            # string verbatim — prefer str(e) (one human-readable line) over
            # repr(e). Fall back to the type name when str(e) is empty.
            msg = str(e) if str(e) else type(e).__name__
            self.signals.error.emit(msg)
