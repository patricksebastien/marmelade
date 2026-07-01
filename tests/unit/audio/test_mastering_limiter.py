"""Wave 0 RED stub — :class:`LimiterStage` ceiling enforcement.

Pinned invariant: feeding a hot synthetic stereo signal through the
:func:`build_limiter_subchain` two-plugin chain produces output whose
sample peak is at or below the configured ceiling (within a small
tolerance for the ISP-vs-sample-peak gap; RESEARCH §Pitfall 1).

Phase 7 — Plan 01 Wave 0 (07-01-PLAN.md Task 1).
"""

from __future__ import annotations

import math

import numpy as np
import pedalboard
import pytest


def test_ceiling_enforced():
    """Hot ones signal through Limiter+Gain chain stays below -1 dBTP.

    The two-plugin sub-chain (Limiter at ``-2 dBFS`` + Gain at ``-2 dB``)
    yields a sample-peak ceiling of ``-2 dBFS`` after the gain — provably
    quieter than the ``-1 dBTP`` target (with ~1 dB ISP headroom).
    """
    from marmelade.audio.mastering.stages.limiter import build_limiter_subchain

    sr = 44100
    # 0.5 s hot stereo signal at 0.99 — louder than the limiter's threshold.
    audio = (np.ones((2, sr // 2), dtype=np.float32) * 0.99).astype(np.float32)

    plugins = build_limiter_subchain({"ceiling_dbtp": -1.0, "release_ms": 100.0})
    pb = pedalboard.Pedalboard(plugins)
    out = pb(audio, sr)

    sample_peak = float(np.max(np.abs(out)))
    # Target ceiling -1 dBTP; sample-peak ceiling is -2 dBFS plus tiny
    # numerical wobble. Allow 5% linear headroom (the test's 5% bound from
    # the plan accounts for the ISP-vs-sample-peak gap that the later
    # ``run_isp_verification`` pass closes; here we just confirm the
    # sub-chain enforces the sample-peak ceiling robustly).
    ceiling_dbtp = -1.0
    sample_peak_ceiling_lin = 10 ** (ceiling_dbtp / 20.0)
    assert sample_peak <= sample_peak_ceiling_lin * 1.05, (
        f"sample_peak={sample_peak} ({20*math.log10(max(sample_peak, 1e-12)):.2f} dBFS) "
        f"exceeds ceiling {ceiling_dbtp} dBTP (linear {sample_peak_ceiling_lin})"
    )
