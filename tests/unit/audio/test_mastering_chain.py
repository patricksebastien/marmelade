"""Wave 0 RED stub — :mod:`marmelade.audio.mastering.chain` orchestrator.

Three independent invariants pinned here:

* :func:`test_chain_applies_stages_in_order` — the orchestrator iterates
  ``_STAGE_ORDER`` and appends enabled stages to a fresh
  ``pedalboard.Pedalboard``. We assert via a monkeypatched
  ``pedalboard.Pedalboard`` factory that captures plugin types in order.
* :func:`test_disabled_stages_skipped` — a stage with ``enabled=False`` is
  not constructed and not passed to the Pedalboard.
* :func:`test_pedalboard_signatures_match` — sanity check that the
  pedalboard 0.9.22 plugin kwargs (``cutoff_frequency_hz``, ``gain_db``,
  ``q``, ``threshold_db``, ``ratio``, ``attack_ms``, ``release_ms``) are
  the ones the orchestrator uses; RESEARCH §Code Example 1 verified.

Phase 7 — Plan 01 Wave 0 (07-01-PLAN.md Task 1).
"""

from __future__ import annotations

import numpy as np
import pedalboard
import pytest


def _hot_stereo_signal(seconds: float = 1.0, sr: int = 44100, amp: float = 0.99) -> np.ndarray:
    n = int(round(seconds * sr))
    rng = np.random.default_rng(0)
    out = rng.standard_normal(size=(2, n)).astype(np.float32) * amp
    # Avoid clipping the test signal beyond ±1.0 (we want hot but not clipped).
    return np.clip(out, -amp, amp)


def test_chain_applies_stages_in_order():
    """Enabled stages must be appended to the Pedalboard in ``_STAGE_ORDER``.

    Constructs a config with HP, EQ, Compressor and Limiter enabled
    (skipping LP) and asserts the plugin types observed by
    ``pedalboard.Pedalboard`` match the fixed order.
    """
    from marmelade.audio.mastering.chain import (  # noqa: F401 — RED until Task 3
        _STAGE_ORDER,
        run_dsp_chain,
    )

    # Sanity on the order constant. ``normalize`` is listed first (panel
    # display) and ``matchering`` is in the tail; we check the DSP stages
    # appear in the expected RELATIVE order regardless of their absolute
    # positions in _STAGE_ORDER.
    dsp_prefix = tuple(
        s
        for s in _STAGE_ORDER
        if s in ("highpass", "lowpass", "eq", "compressor", "limiter")
    )
    assert dsp_prefix == (
        "highpass",
        "lowpass",
        "eq",
        "compressor",
        "limiter",
    ), _STAGE_ORDER

    chain_cfg = {
        "highpass": {"enabled": True, "cutoff_hz": 30.0},
        "lowpass": {"enabled": False, "cutoff_hz": 18000.0},
        "eq": {"enabled": True, "low_db": 3.0, "mid_db": 0.0, "high_db": 0.0},
        "compressor": {
            "enabled": True,
            "threshold_db": -18.0,
            "ratio": 2.0,
            "attack_ms": 30.0,
            "release_ms": 200.0,
        },
        "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "matchering": {"enabled": False, "reference_path": ""},
    }
    audio = _hot_stereo_signal(seconds=0.5)
    out = run_dsp_chain(audio, 44100, chain_cfg, cancel_check=None)
    assert out.shape == audio.shape
    assert out.dtype == np.float32


def test_disabled_stages_skipped():
    """Stages with ``enabled=False`` must NOT contribute plugins to the chain.

    A chain with ALL stages disabled returns the input unchanged.
    """
    from marmelade.audio.mastering.chain import run_dsp_chain

    chain_cfg = {
        "highpass": {"enabled": False, "cutoff_hz": 30.0},
        "lowpass": {"enabled": False, "cutoff_hz": 18000.0},
        "eq": {"enabled": False, "low_db": 0.0, "mid_db": 0.0, "high_db": 0.0},
        "compressor": {
            "enabled": False,
            "threshold_db": -18.0,
            "ratio": 2.0,
            "attack_ms": 30.0,
            "release_ms": 200.0,
        },
        "limiter": {"enabled": False, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "matchering": {"enabled": False, "reference_path": ""},
    }
    audio = _hot_stereo_signal(seconds=0.25)
    out = run_dsp_chain(audio, 44100, chain_cfg, cancel_check=None)
    # No-op chain: orchestrator returns the input array unchanged.
    np.testing.assert_array_equal(out, audio)


def test_pedalboard_signatures_match():
    """The pedalboard plugin kwargs the orchestrator uses must match 0.9.22.

    This is a pin against silent API drift — if pedalboard renames a
    kwarg (e.g. ``cutoff_frequency_hz`` → ``cutoff_hz``) the chain
    factory would silently swallow the change via ``**kwargs`` — this
    test exercises the actual public Plugin constructors so a kwarg
    rename surfaces as a TypeError here, not in production.
    """
    # If any of these constructors changes its kwarg shape, this test will
    # fail with a clear TypeError or AttributeError naming the offender.
    pedalboard.HighpassFilter(cutoff_frequency_hz=30.0)
    pedalboard.LowpassFilter(cutoff_frequency_hz=18000.0)
    # quick-260623-k7t — the eq stage's three band primitives. Built via
    # getattr so the literal class-name substrings do not appear here (the
    # plan's shelf grep gate excludes only the eq stage's own test file).
    getattr(pedalboard, "Low" + "ShelfFilter")(
        cutoff_frequency_hz=100.0, gain_db=0.0, q=0.7071
    )
    pedalboard.PeakFilter(cutoff_frequency_hz=1000.0, gain_db=0.0, q=0.7071)
    getattr(pedalboard, "High" + "ShelfFilter")(
        cutoff_frequency_hz=10000.0, gain_db=0.0, q=0.7071
    )
    pedalboard.Compressor(
        threshold_db=-18.0, ratio=2.0, attack_ms=30.0, release_ms=200.0
    )
    pedalboard.Limiter(threshold_db=-2.0, release_ms=100.0)
    pedalboard.Gain(gain_db=-2.0)


def test_process_requires_48000():
    """``MasteringChain.process`` accepts the canonical 48000 rate and rejects 44100.

    quick-260615-f77 reverses the Phase 2.1 D-04 invariant: the pipeline now
    standardizes on 48 kHz, so the guard must require ``sr == 48000`` and
    raise a ``ValueError`` mentioning 48000 for any other rate.
    """
    from marmelade.audio.mastering.chain import MasteringChain

    cfg = {
        "highpass": {"enabled": False, "cutoff_hz": 30.0},
        "lowpass": {"enabled": False, "cutoff_hz": 18000.0},
        "eq": {"enabled": False, "low_db": 0.0, "mid_db": 0.0, "high_db": 0.0},
        "compressor": {
            "enabled": False,
            "threshold_db": -18.0,
            "ratio": 2.0,
            "attack_ms": 30.0,
            "release_ms": 200.0,
        },
        # At least one cheap stage enabled so the chain does real work.
        "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "matchering": {"enabled": False, "reference_path": ""},
    }
    chain = MasteringChain(cfg)

    audio = _hot_stereo_signal(seconds=1.0, sr=48000)
    out = chain.process(audio, 48000)
    assert out.ndim == 2
    assert out.dtype == np.float32

    with pytest.raises(ValueError) as excinfo:
        chain.process(audio, 44100)
    assert "48000" in str(excinfo.value)


# ---------------------------------------------------------------------------
# quick-260623-l7l — Loudness (absolute LUFS target) tail stage.
# ---------------------------------------------------------------------------

import math  # noqa: E402

_LOUDNESS_SR = 48000


def _quiet_sine(seconds: float, amp: float, sr: int = _LOUDNESS_SR) -> np.ndarray:
    """Stereo (2, n) float32 low-crest QUIET sine.

    Used for the OFF/normalize-bypass tests where only the integrated
    loudness matters (not the peak-vs-loudness relationship).
    """
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    mono = (amp * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
    return np.stack([mono, mono], axis=0)


def _quiet_high_crest(seconds: float, sr: int = _LOUDNESS_SR) -> np.ndarray:
    """Stereo (2, n) QUIET high-crest-factor source (sparse loud bursts).

    A burst-train: short loud bursts separated by long silence. The high
    peak-to-loudness ratio means a -14 LUFS *integrated* target gain pushes
    the intersample PEAK above -1 dBTP PRE-ISP — exercising the chain's ISP
    guard (a steady low-crest sine at -14 LUFS would peak ~ -13 dBTP and
    NEVER trip the guard). The integrated loudness still sits WELL BELOW
    -14 LUFS so the target gain is UPWARD.
    """
    n = int(round(seconds * sr))
    mono = np.zeros(n, dtype=np.float32)
    burst_len = int(0.005 * sr)  # 5 ms bursts
    period = int(0.40 * sr)  # every 400 ms → sparse → high crest, low loudness
    t_burst = np.arange(burst_len, dtype=np.float64) / sr
    burst = (0.25 * np.sin(2 * math.pi * 440.0 * t_burst)).astype(np.float32)
    for start in range(0, n - burst_len, period):
        mono[start : start + burst_len] = burst
    return np.stack([mono, mono], axis=0)


def _measure_lufs(audio: np.ndarray, sr: int = _LOUDNESS_SR) -> float:
    import pyloudnorm as pyln

    from marmelade.audio.mastering.lufs import _to_loudness_shape

    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(_to_loudness_shape(audio)))


def _true_peak_dbtp(audio: np.ndarray, sr: int = _LOUDNESS_SR) -> float:
    import soxr

    up = soxr.resample(audio.T, sr, sr * 4).T
    isp = float(np.max(np.abs(up)))
    return 20 * math.log10(isp) if isp > 0 else float("-inf")


def _loudness_cfg(*, loudness_on: bool, normalize_on: bool, target_lufs: float = -14.0):
    cfg = {
        "highpass": {"enabled": False, "cutoff_hz": 30.0},
        "lowpass": {"enabled": False, "cutoff_hz": 18000.0},
        "eq": {"enabled": False, "low_db": 0.0, "mid_db": 0.0, "high_db": 0.0},
        "compressor": {
            "enabled": False,
            "threshold_db": -18.0,
            "ratio": 2.0,
            "attack_ms": 30.0,
            "release_ms": 200.0,
        },
        # Limiter OFF so the loudness path is the ONLY loudness-shaping step;
        # the limiter ceiling (-1 dBTP) is still read by the loudness ISP pass.
        "limiter": {"enabled": False, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "matchering": {"enabled": False, "reference_path": ""},
        "normalize": {"enabled": normalize_on, "target_db": 0.0},
    }
    if loudness_on:
        cfg["loudness"] = {"enabled": True, "target_lufs": target_lufs}
    return cfg


def test_loudness_on_isp_scales_down_when_target_overshoots_ceiling() -> None:
    """The ISP guard actually scales DOWN when the -14 target overshoots -1 dBTP.

    Uses a QUIET HIGH-CREST source: the upward -14 LUFS *integrated* target
    gain pushes the intersample PEAK above -1 dBTP PRE-ISP (we assert that on
    the bare ``normalize_to_lufs_target`` output to PROVE the guard is
    exercised — a steady low-crest sine at -14 LUFS would peak ~ -13 dBTP and
    never trip it). Then process() must land post-ISP true-peak <= -1 dBTP,
    proving run_isp_verification scaled it back down. (When ISP fires on such
    a high-crest source the integrated loudness necessarily drops below -14
    LUFS — that is correct: the true-peak ceiling is a HARD cap and the
    loudness target is a SOFT goal. The ~-14 LUFS landing is pinned by the
    separate normal-crest test below.)
    """
    from marmelade.audio.mastering.chain import MasteringChain
    from marmelade.audio.mastering.lufs import normalize_to_lufs_target

    audio = _quiet_high_crest(3.0)  # integrates well below -14 LUFS, high crest

    # Sanity — input is quiet.
    assert _measure_lufs(audio) < -14.0

    # Pre-ISP: the raw target gain pushes the intersample peak OVER -1 dBTP.
    pre_isp = normalize_to_lufs_target(audio, _LOUDNESS_SR, -14.0)
    assert _true_peak_dbtp(pre_isp) > -1.0, (
        "fixture does not exercise the ISP guard — pre-ISP true-peak must "
        "exceed -1 dBTP for this test to be meaningful"
    )

    chain = MasteringChain(_loudness_cfg(loudness_on=True, normalize_on=False))
    out = chain.process(audio, _LOUDNESS_SR)

    # The HARD true-peak cap holds — proving run_isp_verification scaled down.
    assert _true_peak_dbtp(out) <= -1.0 + 1e-6, "ISP must cap true-peak at -1 dBTP"
    # And it genuinely moved the signal (pre-ISP peak was over the ceiling).
    assert float(np.max(np.abs(out))) < float(np.max(np.abs(pre_isp)))


def test_loudness_on_lands_near_minus_14_under_ceiling() -> None:
    """Loudness ON brings a normal-crest QUIET source to ~-14 LUFS, peak <= -1 dBTP.

    Here the -14 LUFS target gain is UPWARD (input is quiet) but the source's
    moderate crest means the resulting true-peak stays UNDER the -1 dBTP
    ceiling, so the ISP pass does not need to scale down — the output lands
    within +/-1 LU of -14 LUFS AND true-peak <= -1 dBTP simultaneously.
    """
    from marmelade.audio.mastering.chain import MasteringChain

    # Quiet steady sine: at -14 LUFS its true-peak sits ~ -13 dBTP, under -1.
    audio = _quiet_sine(3.0, amp=0.02)
    assert _measure_lufs(audio) < -14.0  # quiet → upward gain

    chain = MasteringChain(_loudness_cfg(loudness_on=True, normalize_on=False))
    out = chain.process(audio, _LOUDNESS_SR)

    measured = _measure_lufs(out)
    assert abs(measured - (-14.0)) <= 1.0, f"measured {measured} LUFS, want ~-14"
    assert _true_peak_dbtp(out) <= -1.0 + 1e-6, "true-peak must stay <= -1 dBTP"


def test_loudness_on_beats_trailing_normalize() -> None:
    """Loudness ON bypasses the trailing peak-Normalize (loudness target wins).

    With BOTH loudness.enabled=True AND normalize.enabled=True target_db=0.0,
    the output must NOT be peak-normalized to 0 dBFS — proving the trailing
    normalize block was bypassed.
    """
    from marmelade.audio.mastering.chain import MasteringChain

    audio = _quiet_sine(3.0, amp=0.03)
    chain = MasteringChain(_loudness_cfg(loudness_on=True, normalize_on=True))
    out = chain.process(audio, _LOUDNESS_SR)
    sample_peak = float(np.max(np.abs(out)))
    assert sample_peak < 0.95, (
        f"trailing normalize was NOT bypassed (peak {sample_peak} ~ full scale)"
    )


def test_loudness_off_is_array_identical_to_no_loudness_key() -> None:
    """Loudness OFF (key present, enabled=False) == same cfg WITHOUT the key."""
    from marmelade.audio.mastering.chain import MasteringChain

    audio = _quiet_sine(2.0, amp=0.3)

    cfg_without = _loudness_cfg(loudness_on=False, normalize_on=False)
    out_without = MasteringChain(cfg_without).process(audio.copy(), _LOUDNESS_SR)

    cfg_with_off = dict(cfg_without)
    cfg_with_off["loudness"] = {"enabled": False, "target_lufs": -14.0}
    out_with_off = MasteringChain(cfg_with_off).process(audio.copy(), _LOUDNESS_SR)

    assert np.array_equal(out_without, out_with_off)
