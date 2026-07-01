"""RED scaffold — R-4 spectral viewport-density MAX-pool aggregation.

Phase 11 Wave 0 (plan 11-01). The waveform viewer paints at VIEWPORT density,
not native proxy density (USER FEEDBACK 2026-05-13 — coarse/vague render). For
the spectrogram lane this means: a stored mel with ``n_frames`` far larger than
``MAX_RENDER_SPECTRAL_COLS`` must be pooled DOWN to at most
``MAX_RENDER_SPECTRAL_COLS`` columns before painting — and the pool must be a
MAX-pool, so a single transient-hot column survives the downsample (averaging
would smear it away and hide a brief musical event).

PINs the not-yet-existing ``MAX_RENDER_SPECTRAL_COLS`` constant and the
viewport-aggregate helper on :mod:`marmelade.ui.waveform_view`. RED until the
spectral render path lands.
"""

from __future__ import annotations

import numpy as np
import pytest


def _max_cols() -> int:
    from marmelade.ui.waveform_view import MAX_RENDER_SPECTRAL_COLS

    return int(MAX_RENDER_SPECTRAL_COLS)


def _aggregate_fn():
    """Resolve the viewport-aggregate helper (name pinned here).

    The helper takes a ``(n_mels, n_frames)`` mel array and a max column count
    and returns a ``(n_mels, <=max_cols)`` pooled array. Tries the module-level
    function first, then a WaveformView static/instance method, so the impl wave
    has flexibility in placement without changing this test's intent.
    """
    from marmelade.ui import waveform_view as wv

    for name in (
        "aggregate_spectral_columns",
        "_aggregate_spectral_columns",
        "max_pool_spectral_columns",
    ):
        fn = getattr(wv, name, None)
        if fn is not None:
            return fn
    raise ImportError(
        "no spectral viewport-aggregate helper found on waveform_view "
        "(expected aggregate_spectral_columns / max_pool_spectral_columns)"
    )


def test_max_cols_constant_is_reasonable() -> None:
    """MAX_RENDER_SPECTRAL_COLS exists and is a sane viewport bound."""
    n = _max_cols()
    assert 100 <= n <= 100_000


def test_aggregate_caps_column_count() -> None:
    """R-4: feeding more frames than the cap yields <= cap columns."""
    max_cols = _max_cols()
    aggregate = _aggregate_fn()

    n_mels = 64
    n_frames = max_cols * 4 + 7  # well over the cap, non-divisible remainder
    mel = np.random.default_rng(0).random((n_mels, n_frames)).astype(np.float32)

    out = np.asarray(aggregate(mel, max_cols))
    assert out.shape[0] == n_mels, "mel rows (frequency bins) must be preserved"
    assert out.shape[1] <= max_cols, (
        f"pooled to {out.shape[1]} cols, exceeds MAX_RENDER_SPECTRAL_COLS={max_cols}"
    )


def test_aggregate_is_max_pool_not_average() -> None:
    """R-4: a single hot column survives the pool (MAX-pool, not mean-pool).

    Build a near-zero mel with ONE very hot column. A mean-pool would dilute the
    hot column's energy by the pool factor; a max-pool preserves a hot output
    column. We assert the brightest output column retains most of the hot input
    column's magnitude.
    """
    max_cols = _max_cols()
    aggregate = _aggregate_fn()

    n_mels = 32
    n_frames = max_cols * 10  # 10x pool factor — averaging would crush a spike
    mel = np.full((n_mels, n_frames), 0.01, dtype=np.float32)
    hot_value = 1.0
    mel[:, n_frames // 2] = hot_value  # one hot column

    out = np.asarray(aggregate(mel, max_cols), dtype=np.float64)
    brightest = float(out.max())
    # MAX-pool keeps ~hot_value; mean-pool over a 10-wide window would give
    # ~ (hot_value + 9*0.01)/10 ≈ 0.109. Assert clearly above the mean-pool value.
    assert brightest > 0.5 * hot_value, (
        f"brightest pooled column {brightest:.3g} — looks averaged, not max-pooled "
        "(R-4 MAX-pool requirement)"
    )
