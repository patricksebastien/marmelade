"""Wave 0 RED stub — :class:`MasteringRunnable` reuses ``WorkerSignals`` verbatim.

D-16 invariant: every Marmelade QRunnable uses the same
:class:`marmelade.concurrency.worker.WorkerSignals` class — no
subclassing into a mastering-specific signals object.

Phase 7 — Plan 01 Wave 0 (07-01-PLAN.md Task 1).
"""

from __future__ import annotations

from pathlib import Path


def test_runnable_uses_workersignals_verbatim(tmp_path: Path):
    """``MasteringRunnable.signals`` is a :class:`WorkerSignals` (not subclass)."""
    from marmelade.audio.mastering_worker import MasteringRunnable
    from marmelade.concurrency.worker import WorkerSignals

    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    keeper_id = "0" * 32  # 32-hex (uuid4().hex shape)
    cfg = {
        "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0},
    }
    runnable = MasteringRunnable(src, dst, keeper_id=keeper_id, mastering_cfg=cfg)

    assert isinstance(runnable.signals, WorkerSignals)
    # No subclass — type identity check (catches the "I made a
    # MasteringSignals(WorkerSignals)" anti-pattern).
    assert type(runnable.signals) is WorkerSignals
