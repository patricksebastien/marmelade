"""RED scaffold — R-5 spectral-centroid tint.

Phase 11 Wave 0 (plan 11-01). PINs the not-yet-existing centroid-tint render:

    * ``marmelade.audio.render_modes.RenderMode.CENTROID``
    * ``marmelade.audio.render_modes.centroid_tint_colors(centroid)`` — maps a
      per-column spectral centroid (normalised) to a per-column tint color via a
      perceptual LUT (Magma). A brighter / higher-index color for higher
      centroid (more treble).

The fixture is a bass->treble centroid ramp (low centroid then high centroid),
mirroring make_sine(80)+make_sine(6000); we assert the treble segment's tint is
a higher LUT index (brighter) than the bass segment's. RED until the LUT lands.
"""

from __future__ import annotations

import numpy as np
import pytest


def _centroid_tint_colors():
    from marmelade.audio.render_modes import centroid_tint_colors

    return centroid_tint_colors


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of an (..., 3) or (..., 4) uint8/float color array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def test_centroid_tint_exists_and_shapes() -> None:
    """R-5: centroid_tint_colors maps N centroids to N RGB(A) colors."""
    centroid_tint_colors = _centroid_tint_colors()
    centroid = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    colors = np.asarray(centroid_tint_colors(centroid))
    assert colors.shape[0] == centroid.shape[0]
    assert colors.shape[-1] in (3, 4), "expected RGB or RGBA per column"


def test_centroid_tint_brightens_on_treble() -> None:
    """R-5: a treble (high-centroid) segment tints brighter than a bass segment."""
    centroid_tint_colors = _centroid_tint_colors()

    n = 40
    bass = np.full(n, 0.1, dtype=np.float32)    # low centroid (≈80 Hz energy)
    treble = np.full(n, 0.9, dtype=np.float32)  # high centroid (≈6 kHz energy)
    centroid = np.concatenate([bass, treble])

    colors = np.asarray(centroid_tint_colors(centroid))
    luma = _luminance(colors)
    bass_luma = float(luma[:n].mean())
    treble_luma = float(luma[n:].mean())
    assert treble_luma > bass_luma, (
        f"treble tint luma {treble_luma:.1f} not brighter than bass {bass_luma:.1f} "
        "(R-5 centroid-tint must shift color toward the high end of the LUT)"
    )


def test_centroid_mode_is_registered() -> None:
    """R-5: RenderMode.CENTROID exists in the render-mode registry."""
    from marmelade.audio.render_modes import RenderMode

    assert hasattr(RenderMode, "CENTROID"), "RenderMode.CENTROID must be registered"
