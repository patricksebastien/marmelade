"""Cross-thread signalling primitives shared by all background workers.

This package houses the generic worker-signal contract that every long-running
job in Marmelade uses to communicate progress back to the GUI thread. The
package is intentionally Qt-light: it only depends on ``PySide6.QtCore`` (no
QtWidgets / QtGui), so it can be imported from both the GUI tier and worker
tier without dragging widget machinery onto the worker thread.

Phase 2+ heatmap runnables reuse :class:`WorkerSignals` directly — the same
``progress / finished / error / cancelled`` quartet is the canonical shape.
"""

from __future__ import annotations

from marmelade.concurrency.worker import WorkerSignals

__all__ = ["WorkerSignals"]
