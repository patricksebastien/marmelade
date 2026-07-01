"""RED scaffold — R-6 RGB-band render.

Phase 11 Wave 0 (plan 11-01). PINs the not-yet-existing RGB-band render:

    * ``marmelade.audio.render_modes.RenderMode.RGB_BAND``
    * ``marmelade.audio.render_modes.rgb_band_colors(low, mid, high)`` — maps
      per-column low/mid/high band energies to a per-column RGB color where R
      tracks low (bass), G tracks mid, B tracks high (treble). A
      bass-dominant column reads red-ish; a full-band column reads white-ish; a
      mid-dominant column reads green-ish.

Fixture mirrors make_sine(80) (bass) -> bass+mid+treble mix (full) ->
make_sine(1000) (mid): we assert the per-segment dominant channel. RED until
the mapping lands.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rgb_band_colors():
    from marmelade.audio.render_modes import rgb_band_colors

    return rgb_band_colors


def _norm(colors: np.ndarray) -> np.ndarray:
    """Return float RGB in [0,1] from a uint8/float (..., 3|4) color array."""
    c = np.asarray(colors, dtype=np.float64)[..., :3]
    if c.max() > 1.0:
        c = c / 255.0
    return c


def test_rgb_band_exists_and_shapes() -> None:
    """R-6: rgb_band_colors(low, mid, high) returns N RGB(A) colors."""
    rgb_band_colors = _rgb_band_colors()
    n = 30
    low = np.linspace(0, 1, n, dtype=np.float32)
    mid = np.linspace(1, 0, n, dtype=np.float32)
    high = np.full(n, 0.5, dtype=np.float32)
    colors = np.asarray(rgb_band_colors(low, mid, high))
    assert colors.shape[0] == n
    assert colors.shape[-1] in (3, 4)


def test_rgb_band_segment_dominance() -> None:
    """R-6: bass segment -> red dominant, full -> white-ish, mid -> green dominant."""
    rgb_band_colors = _rgb_band_colors()

    n = 20
    # Bass-only segment: low hot, mid/high cold.
    low = np.concatenate([np.ones(n), np.ones(n), np.zeros(n)]).astype(np.float32)
    mid = np.concatenate([np.zeros(n), np.ones(n), np.ones(n)]).astype(np.float32)
    high = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n)]).astype(np.float32)

    colors = _norm(rgb_band_colors(low, mid, high))
    r, g, b = colors[..., 0], colors[..., 1], colors[..., 2]

    # Segment 1 (bass): red channel dominant.
    assert r[:n].mean() > g[:n].mean() and r[:n].mean() > b[:n].mean(), (
        "bass-only segment must read red-dominant (R-6)"
    )
    # Segment 2 (full band): all three channels high (white-ish).
    full = colors[n : 2 * n]
    assert full.min() > 0.4, "full-band segment must read white-ish (all channels high)"
    # Segment 3 (mid): green channel dominant.
    assert g[2 * n :].mean() > r[2 * n :].mean() and g[2 * n :].mean() > b[2 * n :].mean(), (
        "mid-only segment must read green-dominant (R-6)"
    )


def test_rgb_band_mode_is_registered() -> None:
    """R-6: RenderMode.RGB_BAND exists in the render-mode registry."""
    from marmelade.audio.render_modes import RenderMode

    assert hasattr(RenderMode, "RGB_BAND"), "RenderMode.RGB_BAND must be registered"
