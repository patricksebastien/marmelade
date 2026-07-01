"""Phase 7 Plan 07-06 — perf gate for the default mastering chain.

Budget: default chain = Limiter ONLY (per ``_SESSION_DEFAULTS``) must
render a 60-second 44.1 kHz stereo float32 keeper in < 2 s on CPU.

This is the "minimum viable mastering" budget — every keeper opts into
this chain by default (D-04 — keepers snapshot the session at creation,
and the session defaults to limiter-enabled). If THIS gate trips, the
"Master All" loop is too slow for the MVP claim.

Skippable via ``MARMELADE_PERF_SKIP=1`` so CI on exotic CPUs can
opt out without breaking the suite.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from marmelade.audio.mastering.chain import MasteringChain


SR = 48000  # quick-260615-f77: canonical mastering rate (reverses D-04)


@pytest.mark.perf
def test_default_chain_under_2s() -> None:
    """Limiter-only chain on 60 s stereo float32 < 2.0 s wall time."""
    if os.environ.get("MARMELADE_PERF_SKIP") == "1":
        pytest.skip("MARMELADE_PERF_SKIP=1 — perf gate skipped")

    rng = np.random.RandomState(0)
    # 60 s of pink-ish noise scaled to drive the limiter.
    n = 60 * SR
    audio = (rng.randn(2, n) * 0.5).astype(np.float32)

    cfg = {
        "limiter": {
            "enabled": True,
            "ceiling_dbtp": -1.0,
            "release_ms": 100.0,
        }
    }
    chain = MasteringChain(cfg)

    t0 = time.perf_counter()
    out = chain.process(audio, SR)
    elapsed = time.perf_counter() - t0

    # Sanity — the output has the same shape as the input.
    assert out.shape == audio.shape

    assert elapsed < 2.0, (
        f"Default (Limiter-only) chain took {elapsed:.2f}s on 60 s of "
        f"audio; budget is 2.0 s. Linear extrapolation to N keepers "
        f"makes Phase A Master All visibly slow."
    )
