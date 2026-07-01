"""LUFS makeup-gain helper + intersample-peak (ISP) verification.

RESEARCH §Pattern 5 verbatim (07-RESEARCH.md lines 732-790):

* :func:`apply_lufs_makeup` — restore pre-limit ILUFS via a static
  gain pass, clamped to the headroom available so the makeup gain
  cannot push the post-limiter peaks above the ceiling.
* :func:`run_isp_verification` — 4× soxr resample + intersample-peak
  measurement; apply a fallback gain if the chain output exceeds the
  ceiling at the upsampled rate.

Pitfall 6 — pyloudnorm requires audio length > ``block_size``
(400 ms default at 44.1 kHz). The short-clip guard returns
``audio_post`` unchanged for clips ≤ 400 ms.

N-3 invariant: no PySide6 / QtWidgets / QtGui imports.
"""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln
import soxr


def _to_loudness_shape(audio: np.ndarray) -> np.ndarray:
    """Coerce ``audio`` to ``(num_samples, num_channels)`` for pyloudnorm.

    Pedalboard uses ``(channels, samples)`` but pyloudnorm wants the
    samples axis FIRST. A 1-D mono input is returned unchanged.
    """
    if audio.ndim == 2:
        return audio.T
    return audio


def apply_lufs_makeup(
    audio_pre: np.ndarray,
    audio_post: np.ndarray,
    sr: int,
    ceiling_dbtp: float,
) -> np.ndarray:
    """Restore pre-limit ILUFS via a static gain pass, clamped to headroom.

    Args:
        audio_pre: Pre-limiter audio in pedalboard shape
            ``(num_channels, num_samples)`` or 1-D mono.
        audio_post: Post-limiter audio (same shape).
        sr: Sample rate in Hz.
        ceiling_dbtp: True-peak ceiling target in dBTP. The makeup gain is
            clamped so the output sample peak does not exceed
            ``ceiling - 1`` dBFS (RESEARCH §Pattern 5 lines 762-771).

    Returns:
        Audio with the same shape as ``audio_post``. For clips shorter
        than ``int(0.4 * sr) + 1`` samples the helper returns
        ``audio_post`` unchanged (Pitfall 6 — pyloudnorm short-clip
        guard). For non-finite ILUFS (silent or near-silent input) the
        helper also returns ``audio_post`` unchanged.
    """
    # Short-clip guard — pyloudnorm raises on input shorter than its
    # block size; we never invoke it on clips that short.
    min_samples = int(0.4 * sr) + 1
    if audio_pre.shape[-1] <= min_samples:
        return audio_post

    meter = pyln.Meter(sr)
    pre = _to_loudness_shape(audio_pre)
    post = _to_loudness_shape(audio_post)
    ilufs_in = meter.integrated_loudness(pre)
    ilufs_out = meter.integrated_loudness(post)

    # Either can be -inf for silence — guard.
    if not (math.isfinite(ilufs_in) and math.isfinite(ilufs_out)):
        return audio_post

    makeup_db = ilufs_in - ilufs_out

    # Clamp so makeup gain cannot push sample peaks above the ceiling
    # (RESEARCH §Pattern 5 lines 762-771).
    sample_peak_post = float(np.max(np.abs(audio_post)))
    if sample_peak_post <= 0:
        return audio_post
    # WR-04 (Phase 7 review) — short-circuit denormal / sub-noise-floor
    # peaks. Without this, sample_peak_post in the 1e-20 range pushes
    # sample_peak_post_dbfs to ~ -400 dBFS, headroom_db to +399, and
    # the makeup_db clamp degrades to "anything goes" which would
    # produce a 5x+ amplification on near-silent signals. The
    # downstream ISP-verification pass would still catch this, but
    # the docstring's "makeup gain cannot push sample peaks above
    # the ceiling" invariant relies on this clamp being tight. 1e-6
    # corresponds to -120 dBFS — below any meaningful music signal.
    if sample_peak_post < 1e-6:
        return audio_post
    sample_peak_post_dbfs = 20 * math.log10(sample_peak_post)
    # Reserve 1 dB of ISP headroom inside the makeup-gain clamp — same
    # invariant the limiter sub-chain uses (see build_limiter_subchain).
    headroom_db = (ceiling_dbtp - 1.0) - sample_peak_post_dbfs
    # Available upward room is the positive of -headroom_db when the
    # post is BELOW the budget; if it's ABOVE the budget, room is 0
    # (we cannot make it louder).
    max_makeup = max(0.0, headroom_db)
    # Also clamp downward at -12 dB so a wild ILUFS difference cannot
    # eat the signal.
    makeup_db = max(-12.0, min(makeup_db, max_makeup))
    return (audio_post * (10 ** (makeup_db / 20.0))).astype(audio_post.dtype, copy=False)


def normalize_to_lufs_target(
    audio: np.ndarray, sr: int, target_lufs: float
) -> np.ndarray:
    """Apply a static gain so ``audio`` lands at ``target_lufs`` integrated LUFS.

    quick-260623-l7l — the absolute loudness-target counterpart to the
    RELATIVE :func:`apply_lufs_makeup`. Used by the new virtual
    :class:`marmelade.audio.mastering.stages.loudness.LoudnessStage` to hit
    a streaming-delivery loudness (e.g. -14 LUFS for Spotify/Apple).

    Single-purpose by design — the target is a SOFT goal applied IN FULL
    (no upward clamp here). The chain caller runs
    :func:`run_isp_verification` immediately after to enforce the true-peak
    ceiling, so this helper performs NO peak protection itself. If the
    upward target gain overshoots the ceiling, the caller's ISP pass scales
    it back DOWN.

    Args:
        audio: Audio array — pedalboard shape ``(num_channels, num_samples)``
            or 1-D mono.
        sr: Sample rate in Hz.
        target_lufs: Absolute integrated-loudness target (BS.1770) in LUFS.

    Returns:
        Audio with the same shape/dtype as the input, gained toward
        ``target_lufs``. For clips shorter than ``int(0.4 * sr) + 1``
        samples the helper returns ``audio`` unchanged (Pitfall 6 —
        pyloudnorm short-clip guard). For non-finite measured loudness
        (silent / near-silent input) the helper also returns ``audio``
        unchanged so no ``inf``/``NaN`` gain can be produced.
    """
    # Short-clip guard — pyloudnorm raises on input shorter than its
    # block size; mirror apply_lufs_makeup exactly.
    min_samples = int(0.4 * sr) + 1
    if audio.shape[-1] <= min_samples:
        return audio

    meter = pyln.Meter(sr)
    current = meter.integrated_loudness(_to_loudness_shape(audio))
    # Silence / near-silence measures -inf; bail so no inf/NaN gain arises.
    if not math.isfinite(current):
        return audio

    gain_db = target_lufs - current
    return (audio * (10 ** (gain_db / 20.0))).astype(audio.dtype, copy=False)


def run_isp_verification(
    audio: np.ndarray, sr: int, ceiling_dbtp: float
) -> np.ndarray:
    """4× soxr resample + intersample-peak measurement; apply fallback gain.

    Args:
        audio: Audio array — pedalboard shape ``(num_channels, num_samples)``
            or 1-D mono.
        sr: Sample rate in Hz.
        ceiling_dbtp: True-peak ceiling target in dBTP.

    Returns:
        ``audio`` unchanged if the upsampled intersample peak is at or
        below the ceiling; otherwise scaled by the linear correction
        factor so the upsampled peak meets the ceiling.

    Cheap — one resample per keeper (sub-second for typical clip
    lengths) and provides a provable -X dBTP guarantee for
    HUMAN-UAT verification with ffprobe / loudness-scanner.
    """
    if audio.ndim == 2:
        # pedalboard shape (channels, samples) → soxr expects
        # (samples, channels). Transpose, resample, transpose back.
        up = soxr.resample(audio.T, sr, sr * 4).T
    else:
        up = soxr.resample(audio, sr, sr * 4)

    isp_lin = float(np.max(np.abs(up)))
    if isp_lin <= 0:
        return audio
    isp_dbtp = 20 * math.log10(isp_lin)
    if isp_dbtp <= ceiling_dbtp:
        return audio

    correction_db = ceiling_dbtp - isp_dbtp  # negative
    return (audio * (10 ** (correction_db / 20.0))).astype(audio.dtype, copy=False)
