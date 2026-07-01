"""Generic worker-signal contract reused by every background QRunnable.

RESEARCH §Pattern 2 defines this exact quartet — :attr:`progress` (0..100),
:attr:`finished` (payload), :attr:`error` (one-line user-visible string),
:attr:`cancelled` (no payload). All cross-thread communication between a
worker and the GUI tier uses these signals, NEVER direct widget mutation
(RESEARCH Pitfall #4 — PyQtGraph items are not thread-safe).

PySide6 spelling: we import :class:`Signal` from :mod:`PySide6.QtCore`, NOT
``pyqtSignal`` (which is the PyQt5/PyQt6 spelling). The two are not
interchangeable.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class WorkerSignals(QObject):
    """The four cross-thread signals every Marmelade worker emits.

    Attributes:
        progress: Integer percent complete in ``[0, 100]``. The producer is
            expected to fire monotonically and at most ``101`` times per job
            (see :func:`marmelade.audio.peak_builder.build_proxy`'s
            strictly-increasing-percent contract — keeps the Qt signal queue
            from flooding the GUI thread).
        finished: Emitted exactly once on successful completion, carrying a
            job-specific payload. For
            :class:`marmelade.audio.peak_builder_worker.PeakBuilderRunnable`
            the payload is the destination :class:`pathlib.Path` of the
            freshly-written proxy.
        error: Emitted exactly once on uncaught exception, carrying a short,
            user-visible message (UI-SPEC §Copywriting > "Couldn't open file"
            dialog body uses this string verbatim).
        cancelled: Emitted exactly once when the worker observes its cancel
            signal and tears down cleanly. No payload — the GUI just hides
            the progress overlay and returns to empty state.

    A given worker emits EXACTLY ONE of ``finished`` / ``error`` /
    ``cancelled`` in its lifetime; ``progress`` fires zero or more times
    before that terminal signal.
    """

    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()
