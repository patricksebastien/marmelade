"""Functional proof for the unified 3-band EQ stage (quick-260623-k7t).

Replaces the two separate lowshelf + highshelf mastering stages. These
tests prove:

* ``eq`` sits exactly once in ``_STAGE_ORDER`` / ``_PRE_LIMITER_STAGES``,
  positioned after ``lowpass`` and before ``compressor`` (and the legacy
  shelf names are gone).
* ``_STAGE_FACTORY["eq"]`` is DSP-identical to ``EqStage.build_plugin()``
  for matching overrides (single source of truth for freqs/Q).
* Each of the three bands (low/mid/high) measurably raises its target
  frequency band's energy when boosted +12 dB through the real
  pre-limiter DSP path.
"""

from __future__ import annotations

import numpy as np
import pytest

from marmelade.audio.mastering.chain import (
    _PRE_LIMITER_STAGES,
    _STAGE_FACTORY,
    _STAGE_ORDER,
    run_pre_limiter_stages,
)
from marmelade.audio.mastering.stages.eq import EqStage

SR = 48000


def test_eq_in_order_single_stage() -> None:
    """eq appears once, after lowpass before compressor; shelves gone."""
    for tup in (_STAGE_ORDER, _PRE_LIMITER_STAGES):
        assert tup.count("eq") == 1
        assert "lowshelf" not in tup
        assert "highshelf" not in tup
        assert tup.index("lowpass") < tup.index("eq") < tup.index("compressor")


def test_eq_factory_matches_stage() -> None:
    """Factory-built eq plugin == EqStage.build_plugin() on a fixed signal."""
    overrides = {"low_db": 3.0, "mid_db": -2.0, "high_db": 4.0}

    factory_plugin = _STAGE_FACTORY["eq"](overrides)

    stage = EqStage()
    stage._param_overrides = dict(overrides)
    stage_plugin = stage.build_plugin()

    rng = np.random.default_rng(1234)
    noise = rng.standard_normal((2, SR)).astype(np.float32) * 0.1

    out_factory = factory_plugin(noise.copy(), SR)
    out_stage = stage_plugin(noise.copy(), SR)

    assert np.allclose(out_factory, out_stage, atol=1e-6)


def _band_energy(signal: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Sum of rFFT magnitude over [lo, hi) Hz across both channels."""
    total = 0.0
    for ch in signal:
        spectrum = np.abs(np.fft.rfft(ch))
        freqs = np.fft.rfftfreq(ch.shape[-1], d=1.0 / sr)
        mask = (freqs >= lo) & (freqs < hi)
        total += float(spectrum[mask].sum())
    return total


@pytest.mark.parametrize(
    "band_key,boost_param,lo,hi",
    [
        ("low", {"low_db": 12.0}, 0.0, 200.0),
        ("mid", {"mid_db": 12.0}, 700.0, 1400.0),
        ("high", {"high_db": 12.0}, 5000.0, 24000.0),
    ],
)
def test_eq_bands_apply(band_key: str, boost_param: dict, lo: float, hi: float) -> None:
    """Boosting one band +12 dB raises its target band energy vs flat."""
    rng = np.random.default_rng(2026)
    noise = (rng.standard_normal((2, SR)).astype(np.float32)) * 0.1

    flat_cfg = {"eq": {"enabled": True, "low_db": 0.0, "mid_db": 0.0, "high_db": 0.0}}
    boosted = {"enabled": True, "low_db": 0.0, "mid_db": 0.0, "high_db": 0.0}
    boosted.update(boost_param)
    boost_cfg = {"eq": boosted}

    out_flat = run_pre_limiter_stages(noise.copy(), SR, flat_cfg)
    out_boost = run_pre_limiter_stages(noise.copy(), SR, boost_cfg)

    e_flat = _band_energy(out_flat, SR, lo, hi)
    e_boost = _band_energy(out_boost, SR, lo, hi)

    assert e_boost > e_flat * 1.2, (
        f"{band_key} band not boosted: flat={e_flat:.1f} boost={e_boost:.1f}"
    )
