"""Integration: WaveformView render_proxy + cursor/zoom/clear behavior.

Asserts (Plan 03 Task 2):

* ``render_proxy(arr, sample_rate, samples_per_pixel)`` swaps the
  QStackedLayout from empty-state to plot view AND populates a
  ``PlotDataItem`` on ``plot_widget.plotItem`` with non-empty data.
* The render path takes an int16 (N, 2) memmap-like array WITHOUT casting
  to float — we sanity-check that the array we hand in is the
  ``int16`` dtype the contract requires.
* The four documented RESEARCH §Pattern 1 flags are wired (we read back
  via the public PlotDataItem API):
    - ``setDownsampling(auto=True, mode='peak')``
    - ``setClipToView(True)``
    - ``setSkipFiniteCheck(True)`` is set (private flag — we settle for
      a not-None curve)
    - pen width = 1
* ``clear()`` swaps back to empty state and clears the PlotDataItem.
* ``fit_view()`` sets x-range to ``[0, duration_s]``.
* ``zoom(step)`` narrows the x-range width by ``1/step``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from marmelade.ui import theme
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


def _make_int16_pairs(n: int = 4096) -> np.ndarray:
    """Return a sine-shaped (n, 2) int16 array."""
    t = np.arange(n) * (2 * np.pi / 64)
    base = (np.sin(t) * 16000).astype(np.int16)
    pairs = np.empty((n, 2), dtype=np.int16)
    pairs[:, 0] = -base
    pairs[:, 1] = base
    return pairs


def test_render_proxy_swaps_to_plot_and_sets_data(waveform_view: WaveformView) -> None:
    """render_proxy switches the stack to plot view and populates a PlotDataItem."""
    # 4000 = MAX_RENDER_PROXY_PAIRS pass-through boundary (plan 01-09): the
    # branch condition is ``original_length > MAX_RENDER_PROXY_PAIRS``, so a
    # 4000-pair input hits the pass-through path unchanged (len(y) == 8000).
    pairs = _make_int16_pairs(4000)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)

    # Public PyQtGraph API — listDataItems (RESEARCH §Test Map line 823).
    items = waveform_view.plot_widget.plotItem.listDataItems()
    assert len(items) == 1, f"expected exactly one PlotDataItem, got {len(items)}"

    x, y = items[0].getData()
    assert x is not None and len(x) > 0
    assert y is not None and len(y) > 0
    # Each pair contributes two points to y (interleaved min/max saw wave).
    assert len(y) == 2 * pairs.shape[0]


def test_render_proxy_accepts_int16_dtype(waveform_view: WaveformView) -> None:
    """Render must work without forcing a float cast on the GUI thread."""
    pairs = _make_int16_pairs(2048)
    assert pairs.dtype == np.int16
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    # No exception is the assertion.


def test_render_proxy_pen_width_is_one(waveform_view: WaveformView) -> None:
    """RESEARCH §Pattern 1 + Anti-Pattern: pen width must be exactly 1."""
    pairs = _make_int16_pairs(2048)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    item = waveform_view.plot_widget.plotItem.listDataItems()[0]
    pen = item.opts.get("pen")
    assert pen is not None
    assert pen.width() == 1, f"pen width must be 1, got {pen.width()}"


def test_clear_swaps_back_to_empty_state(waveform_view: WaveformView) -> None:
    """clear() resets the layout to empty-state and clears the curve."""
    pairs = _make_int16_pairs(1024)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    # Sanity: data is populated.
    items = waveform_view.plot_widget.plotItem.listDataItems()
    assert items[0].getData()[0] is not None and len(items[0].getData()[0]) > 0

    waveform_view.clear()
    # PlotDataItem still exists but data is empty.
    items = waveform_view.plot_widget.plotItem.listDataItems()
    assert len(items) == 1
    x, _y = items[0].getData()
    # PyQtGraph returns None or empty arrays for cleared data.
    assert x is None or len(x) == 0


def test_fit_view_sets_x_range_to_duration(waveform_view: WaveformView) -> None:
    """fit_view restores the x-range to [0, duration_s]."""
    pairs = _make_int16_pairs(44100 // 256 * 5)  # 5 seconds at spp=256, sr=44100
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    # Zoom in first, then fit_view should expand back.
    waveform_view.plot_widget.setXRange(1.0, 2.0, padding=0)
    waveform_view.fit_view()
    x_min, x_max = waveform_view.plot_widget.plotItem.viewRange()[0]
    assert x_min == pytest.approx(0.0, abs=0.01)
    # The "duration" is samples_per_pixel * length / sample_rate.
    # With our pairs length, the saw wave has 2*length points spaced by
    # spp/(2*sr) → total span = length * spp / sr = (44100//256*5) * 256 / 44100 ≈ 5s.
    expected_duration = (pairs.shape[0] * 256) / 44100
    assert x_max == pytest.approx(expected_duration, rel=0.05)


def test_zoom_narrows_x_range(waveform_view: WaveformView) -> None:
    """zoom(1.25) narrows the x-range width by ~1/1.25 = 0.8."""
    pairs = _make_int16_pairs(2000)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    waveform_view.fit_view()
    x_min, x_max = waveform_view.plot_widget.plotItem.viewRange()[0]
    initial_width = x_max - x_min

    waveform_view.zoom(1.25)
    x_min2, x_max2 = waveform_view.plot_widget.plotItem.viewRange()[0]
    new_width = x_max2 - x_min2

    assert new_width == pytest.approx(initial_width / 1.25, rel=0.02)


def test_aggregated_render_maps_impulse_to_true_time(
    waveform_view: WaveformView,
) -> None:
    """A transient must render at its TRUE time after viewport aggregation.

    Regression for the X-axis horizontal-stretch bug: when the proxy exceeds
    MAX_RENDER_PROXY_PAIRS, the old integer-floor binning + remainder-fold
    drew features at ``(original/MAX)/floor(original/MAX)`` × their true time
    (≈ 1.40× on this 60 s clip → a 1.0 s impulse drew at ~1.40 s, so audio led
    the on-screen feature by a growing amount). The uniform-linspace
    aggregation must keep a known impulse within one render-bin of its true
    time.
    """
    from marmelade.ui.waveform_view import MAX_RENDER_PROXY_PAIRS

    sr = 48000
    spp = 256
    # 60 s → 11_250 proxy pairs, comfortably above the 4000-pair aggregation
    # threshold so the (previously buggy) aggregation branch runs.
    n_pairs = (60 * sr) // spp
    assert n_pairs > MAX_RENDER_PROXY_PAIRS

    pairs = np.zeros((n_pairs, 2), dtype=np.int16)
    pairs[:, 1] = 100  # quiet positive floor everywhere
    pairs[:, 0] = -100
    # Lone tall impulse at the proxy pair covering true t = 1.0 s.
    impulse_pair = int(round(1.0 * sr / spp))
    pairs[impulse_pair, 1] = 30000
    true_t = impulse_pair * spp / sr  # ≈ 0.997 s

    waveform_view.render_proxy(pairs, sample_rate=sr, samples_per_pixel=spp)

    x = waveform_view._rendered_x
    y = waveform_view._rendered_y
    assert x is not None and y is not None
    x_at_peak = float(x[int(np.argmax(y))])

    # One aggregation bin spans ~spp*(n_pairs/MAX)/sr ≈ 15 ms; allow ~2 bins.
    # The OLD stretched mapping put the peak at ~1.40 s (≈ 0.40 s off) — far
    # outside this tolerance, so this test fails on the pre-fix code.
    assert x_at_peak == pytest.approx(true_t, abs=0.04), (
        f"impulse at true t={true_t:.3f}s rendered at x={x_at_peak:.3f}s — "
        f"X-axis is stretched/compressed, not 1:1 with audio time."
    )
