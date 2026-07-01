"""Integration: WaveformView view-mode selector + cached re-render (quick-260627-gb7).

Asserts:

* The view-mode QComboBox is OWNED by WaveformView (attribute + wiring) but is
  NOT placed inside the view's own layout — MainWindow reparents it onto the top
  toolbar. It lists Classic / dB / Energy in RenderMode order with Classic the
  default current index.
* Each mode renders a proxy without raising and leaves exactly one PlotDataItem
  with non-empty data of length 2*N (render-smoke).
* Classic output is byte-identical to today's int16 saw-wave
  (getData()[1] == pairs.reshape(-1) elementwise).
* A mode change re-renders from the CACHED proxy args (no audio reload): the
  PlotDataItem y changes and the Energy envelope is all >= 0.
* Heatmap lane / regions overlay / playhead are undisturbed across a mode
  switch (lane occupies row 1; playhead is not None; waveform item present).
* set_region_normalize is a no-op outside Classic (defensive guard).

Tests run under QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QComboBox

from marmelade.audio.render_modes import MODE_LABELS, RenderMode
from marmelade.ui import theme
from marmelade.ui.heatmap_lane import HeatmapLaneView
from marmelade.ui.regions_overlay import RegionsOverlay
from marmelade.ui.waveform_view import WaveformView


@pytest.fixture
def waveform_view(qtbot, qapp):
    """Build a WaveformView with the theme applied."""
    theme.apply_theme(QApplication.instance())
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(800, 200)
    return view


def _make_int16_pairs(n: int = 4000) -> np.ndarray:
    """Return a sine-shaped (n, 2) int16 array (mirrors test_waveform_view)."""
    t = np.arange(n) * (2 * np.pi / 64)
    base = (np.sin(t) * 16000).astype(np.int16)
    pairs = np.empty((n, 2), dtype=np.int16)
    pairs[:, 0] = -base
    pairs[:, 1] = base
    return pairs


def _waveform_item(view: WaveformView):
    return view.plot_widget.plotItem.listDataItems()[0]


# ---------------------------------------------------------------- combo wiring
def test_combo_owned_by_view_but_not_in_its_layout(waveform_view: WaveformView) -> None:
    combo = waveform_view.render_mode_combo
    assert isinstance(combo, QComboBox)
    # The combo is now RELOCATED to the top toolbar (MainWindow reparents it).
    # WaveformView still OWNS the attribute + wiring, but the combo is NOT
    # placed inside the view's own layout — so on a standalone WaveformView it
    # has no parent (MainWindow gives it one via toolbar.addWidget). See
    # test_main_window_skeleton for the toolbar-placement assertion.
    assert not waveform_view.isAncestorOf(combo)
    # Items are the labels in RenderMode order; Classic is current.
    expected = [MODE_LABELS[m] for m in RenderMode]
    assert [combo.itemText(i) for i in range(combo.count())] == expected
    assert combo.currentIndex() == list(RenderMode).index(RenderMode.CLASSIC)


# ------------------------------------------------------------- per-mode smoke
@pytest.mark.parametrize("mode", list(RenderMode))
def test_each_mode_renders_without_error(
    waveform_view: WaveformView, mode: RenderMode
) -> None:
    pairs = _make_int16_pairs(4000)
    waveform_view.render_mode_combo.setCurrentIndex(list(RenderMode).index(mode))
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    items = waveform_view.plot_widget.plotItem.listDataItems()
    assert len(items) == 1
    x, y = items[0].getData()
    assert x is not None and len(x) > 0
    assert y is not None and len(y) == 2 * pairs.shape[0]


# ------------------------------------------------------- Classic unchanged
def test_classic_output_byte_identical(waveform_view: WaveformView) -> None:
    # The data HANDED to setData in Classic mode must be value-identical to the
    # pre-change int16 saw-wave (pairs.reshape(-1)). We read the authoritative
    # ``_rendered_y`` stash rather than getData(): PyQtGraph's
    # setDownsampling(auto=True, method='peak') makes getData() return the
    # peak-DECIMATED display view (which can re-order min/max within a pixel
    # column), so getData() is not a byte-identity oracle — _rendered_y is.
    pairs = _make_int16_pairs(4000)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    rendered = waveform_view._rendered_y
    assert rendered is not None
    np.testing.assert_array_equal(rendered, pairs.reshape(-1).astype(rendered.dtype))
    # y-range is the full int16 span in Classic.
    y_min, y_max = waveform_view.waveform_plot.viewRange()[1]
    assert y_min == pytest.approx(-32768.0, abs=1.0)
    assert y_max == pytest.approx(32767.0, abs=1.0)


# ------------------------------------------------- mode change re-renders cache
def test_mode_change_rerenders_from_cache_no_reload(
    waveform_view: WaveformView,
) -> None:
    pairs = _make_int16_pairs(4000)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    _x, classic_y = _waveform_item(waveform_view).getData()
    classic_y = np.array(classic_y, copy=True)

    # Switch to Energy WITHOUT calling render_proxy again — the combo signal
    # must re-run the transform from the cached proxy args.
    energy_idx = list(RenderMode).index(RenderMode.ENERGY)
    waveform_view.render_mode_combo.setCurrentIndex(energy_idx)

    _x2, energy_y = _waveform_item(waveform_view).getData()
    assert not np.array_equal(energy_y, classic_y), "mode switch must change y"
    # Authoritative (un-downsampled) stash: the Energy envelope is single-sided.
    rendered = waveform_view._rendered_y
    assert rendered is not None
    assert np.all(rendered >= 0.0), "Energy envelope must be single-sided non-negative"
    assert len(rendered) == 2 * pairs.shape[0]


# ----------------------------------------- heatmap / overlay / playhead intact
def test_heatmap_overlay_playhead_intact_across_switch(
    waveform_view: WaveformView,
) -> None:
    pairs = _make_int16_pairs(4000)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=441)
    QApplication.processEvents()

    # Attach a heatmap lane (occupies row 1) + a regions overlay.
    lane = HeatmapLaneView("energy", waveform_view.waveform_plot)
    waveform_view.add_heatmap_lane("energy", lane)
    overlay = RegionsOverlay(
        plot_item=waveform_view.waveform_plot,
        duration_s_provider=lambda: waveform_view._duration_s,
    )
    waveform_view.set_regions_overlay(overlay)

    assert waveform_view.graphics_layout.getItem(1, 0) is lane.plot_item
    assert waveform_view.playhead is not None

    # Switch mode — lane / overlay / playhead must be undisturbed.
    waveform_view.render_mode_combo.setCurrentIndex(
        list(RenderMode).index(RenderMode.DB)
    )

    assert waveform_view.graphics_layout.getItem(1, 0) is lane.plot_item
    assert waveform_view.playhead is not None
    assert len(waveform_view.plot_widget.plotItem.listDataItems()) == 1


# ------------------------------------- set_region_normalize Classic-only guard
def test_region_normalize_no_op_outside_classic(
    waveform_view: WaveformView,
) -> None:
    pairs = _make_int16_pairs(4000)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=441)

    # Switch to Energy, snapshot the rendered y, then attempt a normalize.
    waveform_view.render_mode_combo.setCurrentIndex(
        list(RenderMode).index(RenderMode.ENERGY)
    )
    _x, before_y = _waveform_item(waveform_view).getData()
    before_y = np.array(before_y, copy=True)

    waveform_view.set_region_normalize(
        start_s=1.0, end_s=5.0, enabled=True, target_db=0.0
    )

    _x2, after_y = _waveform_item(waveform_view).getData()
    np.testing.assert_array_equal(after_y, before_y)


# ===========================================================================
# Phase 11 (plan 11-01) — R-7 spectral-mode coexistence RED scaffold.
#
# These EXTEND the gb7 suite (the tests above remain unchanged and GREEN).
# They PIN that Classic stays the default + byte-identical when the spectral
# modes are added, that overlays + heatmap lanes + playhead keep working with
# Spectrogram selected, and that set_region_normalize is a no-op in Spectrogram
# mode. RED until the spectral render modes land (plans 11-02..11-04).
# ===========================================================================


def _silence_low_high_mel(n_mels: int = 48, n_cols: int = 120) -> np.ndarray:
    """A simple row-major mel: bottom rows hot in the first half, top in second."""
    mel = np.full((n_mels, n_cols), 0.02, dtype=np.float32)
    mel[: n_mels // 4, : n_cols // 2] = 1.0  # low freq, first half
    mel[3 * n_mels // 4 :, n_cols // 2 :] = 1.0  # high freq, second half
    return mel


def test_classic_is_default_and_byte_identical_with_spectral_modes(
    waveform_view: WaveformView,
) -> None:
    """R-7: Classic remains the default mode and its output is byte-identical.

    Even after the spectral modes are registered, the default current index must
    still be Classic and the rendered y must equal pairs.reshape(-1) — the gb7
    invariant must survive the registry growing.
    """
    combo = waveform_view.render_mode_combo
    assert combo.currentIndex() == list(RenderMode).index(RenderMode.CLASSIC)

    pairs = _make_int16_pairs(4000)
    waveform_view.render_proxy(pairs, sample_rate=48000, samples_per_pixel=256)
    rendered = waveform_view._rendered_y
    assert rendered is not None
    np.testing.assert_array_equal(rendered, pairs.reshape(-1).astype(rendered.dtype))


def test_overlays_and_lane_intact_with_spectrogram(
    waveform_view: WaveformView,
) -> None:
    """R-7: regions overlay + playhead + >=1 heatmap lane survive Spectrogram mode."""
    pairs = _make_int16_pairs(4000)
    waveform_view.render_proxy(pairs, sample_rate=48000, samples_per_pixel=441)
    QApplication.processEvents()

    lane = HeatmapLaneView("energy", waveform_view.waveform_plot)
    waveform_view.add_heatmap_lane("energy", lane)
    overlay = RegionsOverlay(
        plot_item=waveform_view.waveform_plot,
        duration_s_provider=lambda: waveform_view._duration_s,
    )
    waveform_view.set_regions_overlay(overlay)

    # Seed spectral data and switch to Spectrogram.
    waveform_view.set_spectral_data(_silence_low_high_mel(), None, None, None)
    waveform_view.render_mode_combo.setCurrentIndex(
        list(RenderMode).index(RenderMode.SPECTROGRAM)
    )
    QApplication.processEvents()

    assert waveform_view.graphics_layout.getItem(1, 0) is lane.plot_item
    assert waveform_view.playhead is not None


def test_set_region_normalize_no_op_in_spectrogram(
    waveform_view: WaveformView,
) -> None:
    """R-7: set_region_normalize is a no-op in Spectrogram mode (Classic-only)."""
    pairs = _make_int16_pairs(4000)
    waveform_view.render_proxy(pairs, sample_rate=48000, samples_per_pixel=441)

    waveform_view.set_spectral_data(_silence_low_high_mel(), None, None, None)
    waveform_view.render_mode_combo.setCurrentIndex(
        list(RenderMode).index(RenderMode.SPECTROGRAM)
    )
    QApplication.processEvents()

    before = np.asarray(waveform_view._rendered_spectral_image, dtype=np.float64).copy()
    waveform_view.set_region_normalize(start_s=1.0, end_s=5.0, enabled=True, target_db=0.0)
    after = np.asarray(waveform_view._rendered_spectral_image, dtype=np.float64)
    np.testing.assert_array_equal(after, before)
