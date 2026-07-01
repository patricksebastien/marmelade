"""Wave 0 RED stub — :func:`apply_lufs_makeup` LUFS round-trip + short-clip guard.

Pinned invariants:
* :func:`test_lufs_roundtrip` — feeding a 1-second pink-noise clip
  through a small attenuation and then ``apply_lufs_makeup`` recovers
  ILUFS within ±0.5 dB.
* :func:`test_short_clip_no_lufs_error` — a 300 ms clip (shorter than
  pyloudnorm's 400 ms block size) does NOT raise; the helper returns
  ``audio_post`` unchanged (RESEARCH §Pitfall 6).

Phase 7 — Plan 01 Wave 0 (07-01-PLAN.md Task 1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _pink_noise_stereo(sr: int, seconds: float, seed: int = 0) -> np.ndarray:
    """Return a ``(2, N)`` float32 stereo pink-noise signal at moderate level."""
    rng = np.random.default_rng(seed)
    n = int(round(sr * seconds))
    # 1/f noise via cumulative-sum filter — cheap, deterministic, sufficient.
    white = rng.standard_normal(size=(2, n)).astype(np.float32)
    # Simple pink-ish filter: cumulative sum (1/f-ish) then normalize.
    pink = np.cumsum(white, axis=1).astype(np.float32)
    peak = float(np.max(np.abs(pink))) or 1.0
    return (pink / peak * 0.3).astype(np.float32)  # ~ -10 dBFS peak


def test_lufs_roundtrip():
    """1-second pink-noise clip — ILUFS round-trip within ±0.5 dB."""
    import pyloudnorm as pyln

    from marmelade.audio.mastering.lufs import apply_lufs_makeup

    sr = 44100
    audio_pre = _pink_noise_stereo(sr, seconds=1.0)
    # Apply a known attenuation to make the post quieter than the pre.
    attenuation_db = -6.0
    audio_post = (audio_pre * (10 ** (attenuation_db / 20.0))).astype(np.float32)

    out = apply_lufs_makeup(audio_pre, audio_post, sr, ceiling_dbtp=-1.0)
    assert out.dtype == np.float32
    assert out.shape == audio_post.shape

    meter = pyln.Meter(sr)
    ilufs_in = meter.integrated_loudness(audio_pre.T)
    ilufs_out = meter.integrated_loudness(out.T)
    # Both must be finite for the assertion to be meaningful (pink-noise
    # at -10 dBFS easily exceeds the LUFS gating floor at 1 second).
    assert math.isfinite(ilufs_in)
    assert math.isfinite(ilufs_out)
    # The helper clamps makeup to available headroom — for a moderate-peak
    # pink-noise signal at -10 dBFS the ceiling has plenty of headroom and
    # the clamp does not bite, so the round-trip should be tight.
    assert abs(ilufs_in - ilufs_out) <= 0.5, (
        f"ilufs_in={ilufs_in:.2f}, ilufs_out={ilufs_out:.2f}, "
        f"delta={ilufs_in - ilufs_out:.2f} dB"
    )


def test_short_clip_no_lufs_error():
    """300 ms clip — :func:`apply_lufs_makeup` returns ``audio_post`` unchanged.

    pyloudnorm's default block size is 400 ms — a 300 ms input would
    raise ``ValueError("Audio must have length greater than the block
    size")`` if we forwarded it. The short-clip guard returns
    ``audio_post`` unchanged.
    """
    from marmelade.audio.mastering.lufs import apply_lufs_makeup

    sr = 44100
    n = int(0.3 * sr)  # 300 ms — strictly below the 400 ms block size.
    audio_pre = _pink_noise_stereo(sr, seconds=0.3)[:, :n]
    audio_post = (audio_pre * 0.5).astype(np.float32)

    out = apply_lufs_makeup(audio_pre, audio_post, sr, ceiling_dbtp=-1.0)
    # Must NOT raise. Must return ``audio_post`` unchanged.
    np.testing.assert_array_equal(out, audio_post)
