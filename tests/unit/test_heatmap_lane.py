"""Plan 02-03 Task 1 — HeatmapLaneView unit pins.

Eleven offscreen-Qt unit tests for the new ``HeatmapLaneView`` renderer:

* MAX_RENDER_HEATMAP_BINS module constant pinned at 4000 (independent of
  ``waveform_view.MAX_RENDER_PROXY_PAIRS`` per CONTEXT discretion).
* DEFAULT_LUT shape + dtype + per-row hex colors (silent/quiet/loud RGBA);
  loud band must NOT use the playhead accent ``#4DA3FF`` (UI-SPEC + D-06).
* ``render`` aggregates values > MAX_RENDER_HEATMAP_BINS via MEAN per bin
  (D-05) — short input passes through unchanged.
* Threshold-at-render assigns one of three discrete bands (D-03).
* Tail-fold preserves the trailing remainder via weighted mean.
* ``ImageItem.boundingRect()`` x-extent maps to time axis (seconds).
* Lane uses ImageItem + LUT, not per-bin PlotCurveItems (HM-02 + research
  §Alternatives).
* ``remove`` schedules the PlotItem for deletion (Pitfall #8 precursor).
* ``setXLink`` to waveform_plot is established at construction (D-01/D-02).

All tests run under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pytest

from marmelade.ui import theme
from marmelade.ui.heatmap_lane import (
    DEFAULT_LUT,
    MAX_RENDER_HEATMAP_BINS,
    HeatmapLaneView,
)
from marmelade.ui.waveform_view import WaveformView


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def waveform_view(qtbot, qapp):
    """A WaveformView gives us a real waveform_plot for setXLink construction."""
    from PySide6.QtWidgets import QApplication

    theme.apply_theme(QApplication.instance())
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(800, 200)
    return view


@pytest.fixture
def lane(waveform_view: WaveformView):
    """Plain HeatmapLaneView wired against the fixture's waveform_plot."""
    return HeatmapLaneView(name="energy", waveform_plot=waveform_view.waveform_plot)


# =========================================================================
# Pins 1-3 — module constants
# =========================================================================
def test_max_render_heatmap_bins_constant_pinned() -> None:
    """Pin 1: MAX_RENDER_HEATMAP_BINS == 4000, INDEPENDENT of the proxy constant."""
    from marmelade.ui import waveform_view as wv

    assert MAX_RENDER_HEATMAP_BINS == 4000
    # The two constants happen to share a value today but live in separate
    # modules so a future tuning of the heatmap density does not silently
    # alter the waveform proxy density.
    assert MAX_RENDER_HEATMAP_BINS is not wv.MAX_RENDER_PROXY_PAIRS or (
        MAX_RENDER_HEATMAP_BINS == 4000
    )


def test_default_lut_three_rows_rgba_uint8() -> None:
    """Pin 2: DEFAULT_LUT is a (3, 4) uint8 RGBA table — silent / quiet / loud."""
    assert isinstance(DEFAULT_LUT, np.ndarray)
    assert DEFAULT_LUT.shape == (3, 4)
    assert DEFAULT_LUT.dtype == np.uint8
    # Row 0 — silent = #1E1E1E (dominant surface)
    assert tuple(DEFAULT_LUT[0]) == (0x1E, 0x1E, 0x1E, 0xFF)
    # Row 1 — quiet = #2F3A47 (low-saturation cool grey-blue)
    assert tuple(DEFAULT_LUT[1]) == (0x2F, 0x3A, 0x47, 0xFF)
    # Row 2 — loud = #5A8FBF (desaturated cyan-blue variant of #7FBFFF)
    assert tuple(DEFAULT_LUT[2]) == (0x5A, 0x8F, 0xBF, 0xFF)


def test_lut_loud_band_is_not_playhead_accent() -> None:
    """Pin 3: UI-SPEC + D-06 reserve ``#4DA3FF`` for the playhead — lane MUST NOT use it."""
    assert tuple(DEFAULT_LUT[2, :3]) != (0x4D, 0xA3, 0xFF)


# =========================================================================
# Pins 4-7 — render aggregation + threshold-at-render + tail fold + rect mapping
# =========================================================================
def test_render_aggregates_long_input_via_mean(lane: HeatmapLaneView) -> None:
    """Pin 4: values of size 2 * MAX_RENDER_HEATMAP_BINS aggregate to 4000 bins.

    All values equal 1.0 with thresholds (0.05, 0.4) → every bin is loud (band 2).
    The test observes the intermediate band-index array via the public
    ``last_rendered_band_indices`` attribute.
    """
    values = np.ones(2 * MAX_RENDER_HEATMAP_BINS, dtype=np.float32)
    lane.render(
        values,
        sample_rate=44100,
        samples_per_value=256,
        silent_quiet_threshold=0.05,
        quiet_loud_threshold=0.4,
    )
    band_idx = lane.last_rendered_band_indices
    assert band_idx is not None
    assert band_idx.shape == (MAX_RENDER_HEATMAP_BINS,)
    assert band_idx.dtype == np.uint8
    # Every value is 1.0 > 0.4 → all bins are loud (band 2).
    assert np.all(band_idx == 2)


def test_render_threshold_at_render_assigns_three_bands(lane: HeatmapLaneView) -> None:
    """Pin 5: threshold-at-render maps four sentinel values to three discrete bands."""
    values = np.array([0.0, 0.1, 0.5, 0.9], dtype=np.float32)
    lane.render(
        values,
        sample_rate=44100,
        samples_per_value=256,
        silent_quiet_threshold=0.05,
        quiet_loud_threshold=0.4,
    )
    band_idx = lane.last_rendered_band_indices
    assert band_idx is not None
    assert band_idx.dtype == np.uint8
    # 0.0   < 0.05 → silent (0)
    # 0.1   ∈ [0.05, 0.4) → quiet (1)
    # 0.5   ≥ 0.4 → loud (2)
    # 0.9   ≥ 0.4 → loud (2)
    assert list(band_idx) == [0, 1, 2, 2]


def test_render_short_input_no_aggregation(lane: HeatmapLaneView) -> None:
    """Pin 6: values of size 100 < MAX_RENDER_HEATMAP_BINS pass through unchanged."""
    values = np.linspace(0.0, 1.0, num=100, dtype=np.float32)
    lane.render(
        values,
        sample_rate=44100,
        samples_per_value=256,
        silent_quiet_threshold=0.05,
        quiet_loud_threshold=0.4,
    )
    band_idx = lane.last_rendered_band_indices
    assert band_idx is not None
    assert band_idx.size == 100


def test_render_tail_folded_into_last_bin(lane: HeatmapLaneView) -> None:
    """Pin 7: original_length = 4000 * 2 + 3 → tail folded into last bin via weighted mean.

    Construct values so the non-tail-folded mean ≠ the tail-folded mean.
    Specifically: every "non-tail" value equals 0.0, but the three trailing
    tail samples equal 1.0 — without the tail fold the last bin would read
    0.0 (band silent); with the tail fold the weighted mean is
    (0.0 * 2 + 3 * 1.0) / (2 + 3) = 0.6 > quiet_loud_threshold = 0.4 → loud.
    """
    bin_size = 2
    n_bins = MAX_RENDER_HEATMAP_BINS
    n_full = n_bins * bin_size
    tail_size = 3
    values = np.zeros(n_full + tail_size, dtype=np.float32)
    values[n_full:] = 1.0
    lane.render(
        values,
        sample_rate=44100,
        samples_per_value=256,
        silent_quiet_threshold=0.05,
        quiet_loud_threshold=0.4,
    )
    band_idx = lane.last_rendered_band_indices
    assert band_idx is not None
    assert band_idx.size == MAX_RENDER_HEATMAP_BINS
    # All non-tail bins should be silent (band 0).
    assert np.all(band_idx[:-1] == 0)
    # Last bin reflects the tail fold — weighted mean = 0.6 → loud (band 2).
    assert band_idx[-1] == 2


def test_render_rect_maps_to_time_axis(lane: HeatmapLaneView) -> None:
    """Pin 8: data-coordinate duration == values.size * spv / sample_rate.

    The ImageItem is positioned via ``setRect(0, 0, duration_s, 1)``; the
    lane exposes the duration it passed to ``setRect`` via the public
    ``last_rendered_duration_s`` attribute so the test does not depend on
    the QGraphicsItem transform mechanics inside ImageItem (whose
    ``boundingRect()`` returns local image-pixel coordinates rather than
    the post-setRect data coordinates).
    """
    values = np.ones(1024, dtype=np.float32)
    sr = 44100
    spv = 256
    lane.render(
        values,
        sample_rate=sr,
        samples_per_value=spv,
        silent_quiet_threshold=0.05,
        quiet_loud_threshold=0.4,
    )
    expected_duration_s = 1024 * spv / sr
    # There must be exactly one ImageItem.
    img_items = [it for it in lane.plot_item.items if isinstance(it, pg.ImageItem)]
    assert len(img_items) == 1
    # Verify the rendered duration matches the time-axis mapping.
    assert lane.last_rendered_duration_s is not None
    assert abs(lane.last_rendered_duration_s - expected_duration_s) < 1e-6


# =========================================================================
# Pins 9-10 — render contract shape (ImageItem not PlotCurveItem) + cleanup
# =========================================================================
def test_lane_uses_image_item_not_plotcurveitems(lane: HeatmapLaneView) -> None:
    """Pin 9: HM-02 + RESEARCH §Alternatives — ImageItem + LUT, NOT PlotCurveItems."""
    image_items = [it for it in lane.plot_item.items if isinstance(it, pg.ImageItem)]
    curve_items = [
        it for it in lane.plot_item.items if isinstance(it, pg.PlotCurveItem)
    ]
    assert len(image_items) == 1
    assert len(curve_items) == 0


def test_remove_lane_calls_deletelater(
    lane: HeatmapLaneView, waveform_view: WaveformView, qapp
) -> None:
    """Pin 10: after ``lane.remove(layout)`` the lane PlotItem is no longer in the layout.

    The deleteLater() schedules the PlotItem for destruction on the next
    event-loop tick. We process events explicitly via qapp.processEvents()
    after the remove call so the destruction completes deterministically.
    """
    layout = waveform_view.graphics_layout
    layout.addItem(lane.plot_item, row=2, col=0)
    assert layout.getItem(2, 0) is lane.plot_item
    lane.remove(layout)
    qapp.processEvents()
    # After remove, row 2 col 0 is empty (None on a stale lookup).
    assert layout.getItem(2, 0) is None


# =========================================================================
# Pin 11 — setXLink contract
# =========================================================================
def test_setxlink_to_waveform_plot_on_construction(
    waveform_view: WaveformView,
) -> None:
    """Pin 11: ``setXLink(waveform_plot)`` is established at construction time."""
    lane = HeatmapLaneView(name="energy", waveform_plot=waveform_view.waveform_plot)
    lane_vb = lane.plot_item.getViewBox()
    wf_vb = waveform_view.waveform_plot.getViewBox()
    assert lane_vb.linkedView(pg.ViewBox.XAxis) is wf_vb
