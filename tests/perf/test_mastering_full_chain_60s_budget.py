"""Phase 7 Plan 07-06 — perf gate for the FULL mastering chain.

Budget: full chain (HP + LP + EQ + Compressor +
Limiter + Matchering with a synthetic reference) must render a
60-second 44.1 kHz stereo float32 keeper in < 60 s on CPU.

This is the "everything turned on" budget. Matchering is the dominant
cost and is uncancellable mid-call (RESEARCH §Pitfall 5) — if THIS
gate trips, the user experiences mastering cancellation latency
that exceeds the UI-SPEC tooltip's "~30 s for Matchering" promise.

Skippable via ``MARMELADE_PERF_SKIP=1``. Also marked ``slow`` so
``pytest -m 'not slow'`` skips it in the fast development loop.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from marmelade.audio.mastering.chain import MasteringChain
from marmelade.paths import matchering_reference_dir


SR = 48000  # quick-260615-f77: canonical mastering rate (reverses D-04)


@pytest.mark.slow
@pytest.mark.perf
def test_full_chain_under_60s(tmp_path: Path) -> None:
    """Full chain + Matchering on 60 s stereo float32 < 60 s wall time."""
    if os.environ.get("MARMELADE_PERF_SKIP") == "1":
        pytest.skip("MARMELADE_PERF_SKIP=1 — perf gate skipped")

    rng = np.random.RandomState(0)
    n = 60 * SR
    audio = (rng.randn(2, n) * 0.3).astype(np.float32)

    # Write the matchering reference INTO the library dir so the
    # T-7-01 containment check accepts it without an is_one_off flag.
    ref_dir = matchering_reference_dir()
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / "perf_ref.wav"
    ref_audio = (rng.randn(60 * SR, 2) * 0.3).astype(np.float32)
    sf.write(str(ref_path), ref_audio, SR, subtype="FLOAT", format="WAV")

    cfg = {
        "highpass": {"enabled": True, "cutoff_hz": 30.0},
        "lowpass": {"enabled": True, "cutoff_hz": 18000.0},
        "eq": {
            "enabled": True,
            "low_db": 1.5,
            "mid_db": 0.0,
            "high_db": 1.5,
        },
        "compressor": {
            "enabled": True,
            "threshold_db": -18.0,
            "ratio": 2.0,
            "attack_ms": 30.0,
            "release_ms": 200.0,
        },
        "limiter": {
            "enabled": True,
            "ceiling_dbtp": -1.0,
            "release_ms": 100.0,
        },
        "matchering": {
            "enabled": True,
            "reference_path": str(ref_path),
            "is_one_off": False,
        },
    }
    chain = MasteringChain(cfg)

    t0 = time.perf_counter()
    out = chain.process(audio, SR)
    elapsed = time.perf_counter() - t0

    assert out.shape == audio.shape

    assert elapsed < 60.0, (
        f"Full chain (HP+LP+eq+compressor+limiter+matchering) took "
        f"{elapsed:.2f}s on 60 s of audio; budget is 60.0 s. The "
        f"~30 s cancel-latency tooltip promise is broken at this rate."
    )
