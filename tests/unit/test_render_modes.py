"""Unit tests for ``marmelade.audio.render_modes`` (quick-260627-gb7).

Covers the pure-numpy render-mode registry that transforms an aggregated
``(N, 2)`` int16 min/max proxy into a flat saw-wave y array + a y-range
tuple, for the three Tier-1 waveform view modes.

Covered cases:
- Registry shape: RenderMode has the 3 Tier-1 amplitude modes (CLASSIC first)
  plus the 3 Tier-2 spectral modes added in Phase 11; transform_for returns a
  callable for each Tier-1 mode and raises KeyError for the spectral modes
  (which route to the *_colors helpers); MODE_LABELS has a label per mode.
- N-3 toolkit-free invariant is exercised structurally by importing the module
  (no Qt import) — the verify step greps the source separately.
- CLASSIC identity: y_flat equals pairs.reshape(-1) elementwise (cast aside);
  y_range == (-32768.0, 32767.0); length == 2*N.
- DB zero-floor finiteness: an all-zero pairs array yields finite y (no
  inf/nan — the log(0) magnitude floor works); y_range == (-32768.0, 32767.0).
- DB full-scale mapping: a +full-scale max sample maps near +32767.
- DB sign preservation: a negative min maps to a negative plotted value.
- DB length == 2*N.
- ENERGY rectified envelope: env per bin == max(|min|, |max|).
- ENERGY non-negativity: every plotted y value is >= 0; y_range == (0.0, 32767.0).
- ENERGY length == 2*N.
- Empty-array case: every mode returns an empty float32 array + its y_range
  (a cleared plot must not crash).
"""

from __future__ import annotations

import numpy as np
import pytest

from marmelade.audio.render_modes import (
    DB_FLOOR,
    INT16_FULL_SCALE,
    MODE_LABELS,
    RenderMode,
    transform_for,
)


def _pairs(rows: list[tuple[int, int]]) -> np.ndarray:
    """Build a hand-specified (N, 2) int16 min/max array."""
    return np.array(rows, dtype=np.int16)


# --------------------------------------------------------------- registry shape
# The 3 Tier-1 amplitude modes are registry-backed (min/max -> saw-wave). The 3
# Tier-2 spectral modes (Phase 11) consume spectral arrays, not min/max pairs,
# so they are intentionally NOT in _REGISTRY; their color math lives in the
# rgb_band_colors / centroid_tint_colors helpers (tested in tests/integration).
TIER1_MODES = (RenderMode.CLASSIC, RenderMode.DB, RenderMode.ENERGY)


def test_render_mode_classic_is_first_default() -> None:
    members = list(RenderMode)
    assert members[0] is RenderMode.CLASSIC, "CLASSIC must be the first/default member"


def test_render_mode_includes_tier1_and_tier2_members() -> None:
    names = {m.name for m in RenderMode}
    # Tier-1 amplitude modes (gb7).
    assert {"CLASSIC", "DB", "ENERGY"} <= names
    # Tier-2 spectral modes (Phase 11) — auto-populate the dropdown.
    assert {"SPECTROGRAM", "CENTROID", "RGB_BAND"} <= names


def test_transform_for_returns_callable_per_tier1_mode() -> None:
    for mode in TIER1_MODES:
        fn = transform_for(mode)
        assert callable(fn)


def test_transform_for_raises_for_spectral_modes() -> None:
    # Spectral modes are not amplitude transforms; transform_for must reject them
    # (the view routes them to the *_colors helpers instead).
    for mode in (RenderMode.SPECTROGRAM, RenderMode.CENTROID, RenderMode.RGB_BAND):
        with pytest.raises(KeyError):
            transform_for(mode)


def test_mode_labels_cover_every_mode() -> None:
    for mode in RenderMode:
        assert mode in MODE_LABELS
        assert isinstance(MODE_LABELS[mode], str) and MODE_LABELS[mode]


# ------------------------------------------------------------------- CLASSIC
def test_classic_is_identity_passthrough() -> None:
    pairs = _pairs([(-100, 200), (-32768, 32767), (0, 0), (5, 7)])
    y_flat, y_range = transform_for(RenderMode.CLASSIC)(pairs)
    expected = pairs.reshape(-1).astype(np.float32)
    np.testing.assert_array_equal(y_flat, expected)
    assert y_range == (-32768.0, 32767.0)


def test_classic_length_is_2n() -> None:
    pairs = _pairs([(-1, 1)] * 10)
    y_flat, _ = transform_for(RenderMode.CLASSIC)(pairs)
    assert y_flat.shape[0] == 2 * pairs.shape[0]


# ------------------------------------------------------------------------ DB
def test_db_all_zero_is_finite_no_log0_crash() -> None:
    pairs = np.zeros((8, 2), dtype=np.int16)
    y_flat, y_range = transform_for(RenderMode.DB)(pairs)
    assert np.all(np.isfinite(y_flat)), "log(0) floor must keep all y finite"
    assert y_range == (-32768.0, 32767.0)


def test_db_full_scale_maps_near_plus_full_scale() -> None:
    # A +full-scale max sample should map to ~ +INT16_FULL_SCALE (0 dB -> top).
    pairs = _pairs([(0, 32767)])
    y_flat, _ = transform_for(RenderMode.DB)(pairs)
    # y_flat layout is [v0_lo, v0_hi]; the hi (max=32767) is the full-scale one.
    assert y_flat[1] == pytest.approx(INT16_FULL_SCALE, rel=0.02)


def test_db_preserves_sign() -> None:
    pairs = _pairs([(-32767, 32767)])
    y_flat, _ = transform_for(RenderMode.DB)(pairs)
    assert y_flat[0] < 0.0, "negative min must map to negative plotted value"
    assert y_flat[1] > 0.0, "positive max must map to positive plotted value"


def test_db_quiet_floor_maps_near_zero() -> None:
    # A sample at the DB_FLOOR magnitude should map to ~0 (lifted baseline).
    mag = 10 ** (DB_FLOOR / 20.0)
    quiet = int(round(mag * INT16_FULL_SCALE))
    pairs = _pairs([(0, quiet)])
    y_flat, _ = transform_for(RenderMode.DB)(pairs)
    assert abs(y_flat[1]) < INT16_FULL_SCALE * 0.1


def test_db_length_is_2n() -> None:
    pairs = _pairs([(-3, 5)] * 7)
    y_flat, _ = transform_for(RenderMode.DB)(pairs)
    assert y_flat.shape[0] == 2 * pairs.shape[0]


# -------------------------------------------------------------------- ENERGY
def test_energy_env_is_max_abs_per_bin() -> None:
    pairs = _pairs([(-100, 50), (-10, 200), (-32768, 0), (0, 0)])
    y_flat, _ = transform_for(RenderMode.ENERGY)(pairs)
    # Each bin emits two interleaved points; the non-zero one is the envelope.
    expected_env = np.maximum(
        np.abs(pairs[:, 0].astype(np.int64)), np.abs(pairs[:, 1].astype(np.int64))
    )
    got_env = y_flat.reshape(-1, 2).max(axis=1)
    np.testing.assert_array_equal(got_env, expected_env.astype(np.float32))


def test_energy_all_non_negative() -> None:
    pairs = _pairs([(-32768, 32767), (-5, 3), (0, -7)])
    y_flat, y_range = transform_for(RenderMode.ENERGY)(pairs)
    assert np.all(y_flat >= 0.0)
    assert y_range == (0.0, 32767.0)


def test_energy_length_is_2n() -> None:
    pairs = _pairs([(-1, 1)] * 12)
    y_flat, _ = transform_for(RenderMode.ENERGY)(pairs)
    assert y_flat.shape[0] == 2 * pairs.shape[0]


# ------------------------------------------------------------- empty-array case
@pytest.mark.parametrize("mode", TIER1_MODES)
def test_empty_pairs_returns_empty_float32_and_y_range(mode: RenderMode) -> None:
    empty = np.zeros((0, 2), dtype=np.int16)
    y_flat, y_range = transform_for(mode)(empty)
    assert y_flat.shape[0] == 0
    assert y_flat.dtype == np.float32
    assert isinstance(y_range, tuple) and len(y_range) == 2
