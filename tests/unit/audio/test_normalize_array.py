"""quick-260620-mgu Task 1 / quick-260621-gfq — normalize_array() in-RAM helper.

NORM-02 — the DC-removal + peak-to-dB formula lives in exactly ONE place
(:func:`marmelade.audio.normalize.normalize_array`). quick-260621-gfq made
``normalize_array`` the mastering chain's FINAL stage and removed the
whole-file streaming ``normalize_audio`` (and its NormalizeRunnable worker), so
this module now pins only the pure in-RAM helper.
"""

from __future__ import annotations

import numpy as np

from marmelade.audio.normalize import normalize_array


def test_returns_new_float32_array_same_shape() -> None:
    """(channels, n) input → NEW (channels, n) float32 array (input untouched)."""
    samples = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    original = samples.copy()
    out = normalize_array(samples, -6.0)
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    # Input must not be mutated in place.
    np.testing.assert_array_equal(samples, original)


def test_dc_removed_per_channel() -> None:
    """Each channel's output mean is ~0.0 (within 1e-6)."""
    n = 1000
    t = np.linspace(0, 1, n, endpoint=False)
    ch0 = np.sin(2 * np.pi * 5 * t) + 0.3  # DC offset 0.3
    ch1 = np.sin(2 * np.pi * 7 * t) - 0.2  # DC offset -0.2
    samples = np.stack([ch0, ch1]).astype(np.float32)
    out = normalize_array(samples, -6.0)
    assert abs(float(out[0].mean())) < 1e-6
    assert abs(float(out[1].mean())) < 1e-6


def test_peak_hits_target_db() -> None:
    """Output absolute peak == 10**(target_db/20) within 1e-5 for non-silent input."""
    n = 1000
    t = np.linspace(0, 1, n, endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 5 * t)).reshape(1, n).astype(np.float32)
    out = normalize_array(samples, -6.0)
    expected_peak = 10.0 ** (-6.0 / 20.0)
    assert abs(float(np.abs(out).max()) - expected_peak) < 1e-5
    # Sanity on the documented numeric value.
    assert abs(expected_peak - 0.50119) < 1e-4


def test_silent_input_no_amplification() -> None:
    """All-zero input → scale 1.0; output stays all-zero (no noise-floor amp)."""
    samples = np.zeros((2, 500), dtype=np.float32)
    out = normalize_array(samples, -6.0)
    assert out.shape == (2, 500)
    assert float(np.abs(out).max()) == 0.0


def test_dc_only_input_no_amplification() -> None:
    """A pure-DC (constant) input has post-DC peak ~0 → no divide-by-zero blowup."""
    samples = np.full((1, 500), 0.5, dtype=np.float32)
    out = normalize_array(samples, -6.0)
    # DC removed → all ~0, scale clamped to 1.0, no NaN/inf.
    assert np.all(np.isfinite(out))
    assert float(np.abs(out).max()) < 1e-6


def test_1d_input_treated_as_single_channel() -> None:
    """A 1-D (n,) input returns shape (1, n)."""
    samples = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    out = normalize_array(samples, -6.0)
    assert out.shape == (1, 4)


def test_default_target_is_zero_db() -> None:
    """quick-260621-gfq — the default target is now 0 dB (full-scale peak)."""
    n = 1000
    t = np.linspace(0, 1, n, endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 5 * t)).reshape(1, n).astype(np.float32)
    out = normalize_array(samples)  # no explicit target
    assert abs(float(np.abs(out).max()) - 1.0) < 1e-5
