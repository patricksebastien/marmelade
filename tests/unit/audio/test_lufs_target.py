"""quick-260623-l7l Task 1 — :func:`normalize_to_lufs_target` DSP core.

Absolute integrated-loudness target (BS.1770) for the per-keeper mastering
chain's new "Loudness" stage. The helper is single-purpose:

* It re-measures the input's integrated loudness via ``pyln.Meter`` and
  applies a static gain so the output lands at ``target_lufs``.
* The target is a SOFT goal applied IN FULL (no upward clamp) — the chain
  caller runs :func:`run_isp_verification` next to enforce the true-peak
  ceiling, so this helper performs NO peak protection itself.
* Short clips (``<= int(0.4 * sr) + 1`` samples) return UNCHANGED — pyloudnorm
  raises on input shorter than its 400 ms block size.
* Silent / near-silent input (non-finite measured loudness) returns UNCHANGED.
"""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln

from marmelade.audio.mastering.lufs import (
    _to_loudness_shape,
    normalize_to_lufs_target,
)


SR = 48000


def _sine(seconds: float, amp: float, sr: int = SR, freq: float = 440.0) -> np.ndarray:
    """Stereo (2, n) float32 sine at ``amp`` linear level."""
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    mono = (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono], axis=0)


def _measure_lufs(audio: np.ndarray, sr: int = SR) -> float:
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(_to_loudness_shape(audio)))


def test_lands_within_1_lu_of_minus_14() -> None:
    """A steady ~3 s sine is brought to within +/-1 LU of -14 LUFS."""
    audio = _sine(3.0, amp=0.2)
    out = normalize_to_lufs_target(audio, SR, -14.0)
    measured = _measure_lufs(out)
    assert math.isfinite(measured)
    assert abs(measured - (-14.0)) <= 1.0, f"measured {measured} LUFS, want ~-14"


def test_quiet_signal_gets_louder() -> None:
    """A quiet source integrating well below -14 LUFS gains UPWARD."""
    audio = _sine(3.0, amp=0.01)  # very quiet → integrates far below -14
    loudness_in = _measure_lufs(audio)
    out = normalize_to_lufs_target(audio, SR, -14.0)
    loudness_out = _measure_lufs(out)
    assert loudness_in < -14.0, f"fixture not quiet enough: {loudness_in} LUFS"
    assert loudness_out > loudness_in, "target gain must be upward for a quiet input"
    # Output peak must EXCEED the input peak (gain applied in full, no clamp).
    assert float(np.max(np.abs(out))) > float(np.max(np.abs(audio)))


def test_short_clip_returns_identity() -> None:
    """A clip <= int(0.4*sr)+1 samples returns the SAME array, no exception."""
    n = int(0.4 * SR) + 1  # exactly the guard boundary (<= returns unchanged)
    short = (0.2 * np.ones((2, n), dtype=np.float32))
    out = normalize_to_lufs_target(short, SR, -14.0)
    assert np.array_equal(out, short)


def test_silence_returns_unchanged() -> None:
    """All-zeros (non-finite measured loudness) returns unchanged, no inf gain."""
    silence = np.zeros((2, 3 * SR), dtype=np.float32)
    out = normalize_to_lufs_target(silence, SR, -14.0)
    assert np.array_equal(out, silence)


def test_dtype_preserved() -> None:
    """Output dtype matches the float32 input."""
    audio = _sine(3.0, amp=0.2)
    out = normalize_to_lufs_target(audio, SR, -14.0)
    assert out.dtype == np.float32
