"""In-RAM DC-removal + peak-to-dB normalization (quick-260620-ht4 / -gfq).

Pure NumPy + stdlib. Like the rest of ``audio/`` this module has NO toolkit
imports (N-3 invariant).

What it does:
    DC-remove + peak-scale an in-RAM ``(channels, n)`` array: recenter each
    channel on 0.0, then scale so the absolute peak equals a configurable dB
    target. Used as the FINAL stage of the mastering chain
    (:class:`marmelade.audio.mastering.chain.MasteringChain`) per
    quick-260621-gfq — and by the in-place WYSIWYG waveform re-render (which
    shares the :func:`_compute_scale` core).

History: quick-260621-gfq removed the whole-file streaming ``normalize_audio``
function and its ``NormalizeRunnable`` worker (the old toolbar Normalize
action). Normalize now lives strictly inside the mastering chain; only the
pure in-RAM :func:`normalize_array` + :func:`_compute_scale` survive.

Scaling:
    ``target_linear = 10 ** (target_db / 20)``. ``scale = target_linear /
    max(post_dc_peak, eps)``, but clamped to ``1.0`` for a silent / all-zero
    source so there is no divide-by-zero and the noise floor is never amplified
    to full scale.
"""

from __future__ import annotations

import numpy as np

# Floor for the peak divisor so a silent source never divides by zero.
_PEAK_EPS = 1e-9


def _compute_scale(post_dc_peak: float, target_db: float) -> float:
    """Single source of the dB→linear + eps-clamp scaling math (NORM-02).

    Returns the multiplicative scale that maps ``post_dc_peak`` onto the
    linear amplitude for ``target_db``. Clamps to ``1.0`` when the post-DC
    peak is at or below :data:`_PEAK_EPS` so a silent / all-zero (or
    pure-DC) source never divides by zero and never amplifies its noise
    floor to full scale.

    :func:`normalize_array` (the mastering-chain final stage) and the
    in-place WYSIWYG waveform re-render both call this so the formula lives
    in exactly ONE place.
    """
    if post_dc_peak <= _PEAK_EPS:
        return 1.0
    target_linear = float(10.0 ** (target_db / 20.0))
    return target_linear / post_dc_peak


def normalize_array(
    samples: np.ndarray,
    target_db: float = 0.0,
) -> np.ndarray:
    """DC-remove + peak-to-``target_db`` an in-RAM ``(channels, n)`` array (NORM-02).

    Pure in-RAM DC-removal + peak-to-dB using the shared
    :func:`_compute_scale` core so the math has a single home. Used as the
    FINAL stage of the mastering chain (quick-260621-gfq), where the short
    keeper region is held whole in RAM.

    Args:
        samples: ``(channels, n)`` float array. A 1-D ``(n,)`` input is
            treated as a single channel via ``np.atleast_2d`` and returns
            shape ``(1, n)``.
        target_db: Desired peak amplitude in dBFS (``<= 0``). Default ``0.0``
            → linear 1.0 (full scale).

    Returns:
        A NEW ``(channels, n)`` ``float32`` array — the input is never
        mutated in place. Each channel's mean is ~0.0 (DC removed) and the
        global absolute peak equals ``10 ** (target_db / 20)`` for a
        non-silent input. A silent / all-zero / pure-DC input (post-DC peak
        ``<= _PEAK_EPS``) returns the DC-removed array unscaled (scale 1.0).
    """
    arr = np.atleast_2d(np.asarray(samples)).astype(np.float32, copy=False)
    # Per-channel mean column (the DC offset). Mean accumulated in float64
    # for precision.
    mean_col = arr.mean(axis=1, dtype=np.float64).astype(np.float32)
    mean_col = mean_col.reshape(arr.shape[0], 1)
    centered = arr - mean_col
    post_dc_peak = float(np.abs(centered).max()) if centered.size else 0.0
    scale = _compute_scale(post_dc_peak, target_db)
    return (centered * np.float32(scale)).astype(np.float32, copy=False)


__all__ = ["normalize_array"]
