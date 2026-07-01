"""Plan 02-01 Task 2 — GraphicsLayoutWidget refactor + reserved Energy lane stub.

Ten integration pins. Five prove the new container (graphics_layout exists,
waveform_plot at row 0, reserved lane at row 1 hidden, setXLink wired,
bi-directional x-range sync). Five prove zero Phase 1 regression (four-flag
PlotDataItem contract, plot_widget shim, float32 x-array, centerline placement,
mouse-down-x tracking, end-to-end open→render).

Run under offscreen Qt; uses the existing make_sine fixture for the end-to-end
gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog

from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from marmelade.ui.waveform_view import (
    MAX_RENDER_PROXY_PAIRS,
    SEEK_THRESHOLD_PX,
    WaveformView,
)
from tests.fixtures.synthesize import make_sine


# ----------------------------------------------------------------- fixtures
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
    """Return a sine-shaped (n, 2) int16 array (same helper as test_waveform_view)."""
    t = np.arange(n) * (2 * np.pi / 64)
    base = (np.sin(t) * 16000).astype(np.int16)
    pairs = np.empty((n, 2), dtype=np.int16)
    pairs[:, 0] = -base
    pairs[:, 1] = base
    return pairs


# =========================================================================
# Container refactor — pins 1-3 + 7-8
# =========================================================================
def test_graphics_layout_widget_exists(waveform_view: WaveformView) -> None:
    """Pin 1: WaveformView owns a pg.GraphicsLayoutWidget as `graphics_layout`."""
    assert hasattr(waveform_view, "graphics_layout")
    assert isinstance(waveform_view.graphics_layout, pg.GraphicsLayoutWidget)


def test_waveform_plot_at_row_0(waveform_view: WaveformView) -> None:
    """Pin 2: the waveform PlotItem lives at row 0 col 0 of the layout."""
    assert hasattr(waveform_view, "waveform_plot")
    assert isinstance(waveform_view.waveform_plot, pg.PlotItem)
    # GraphicsLayout.getItem(row, col) — same idiom as PATTERNS.md skeleton.
    assert waveform_view.graphics_layout.getItem(0, 0) is waveform_view.waveform_plot


def test_reserved_energy_lane_at_row_1_hidden(waveform_view: WaveformView) -> None:
    """Pin 3: reserved Energy lane is a PlotItem at row 1, hidden (maxH=0), x-linked."""
    lane = waveform_view.graphics_layout.getItem(1, 0)
    assert isinstance(lane, pg.PlotItem), (
        f"expected pg.PlotItem at row 1; got {type(lane).__name__}"
    )
    assert lane.maximumHeight() == 0, (
        f"reserved lane must be invisible in Plan 02-01 (height=0); "
        f"got maximumHeight={lane.maximumHeight()}"
    )
    # setXLink wires the lane's x-axis to the waveform's viewbox.
    linked = lane.getViewBox().linkedView(pg.ViewBox.XAxis)
    assert linked is waveform_view.waveform_plot.getViewBox(), (
        "reserved lane's x-axis must be setXLinked to the waveform PlotItem's ViewBox"
    )


def test_centerline_in_waveform_plot(waveform_view: WaveformView) -> None:
    """Pin 7: the zero-amplitude centerline (Phase 1) lives on the waveform plot
    (NOT on the reserved energy lane — that would put it in the wrong row)."""
    # Walk waveform_plot's items; an InfiniteLine with angle 0 (horizontal) is
    # the centerline.
    wf_items = waveform_view.waveform_plot.items
    has_centerline = any(
        isinstance(it, pg.InfiniteLine) and it.angle == 0 for it in wf_items
    )
    assert has_centerline, "centerline (horizontal InfiniteLine) missing from waveform_plot"

    # And the reserved lane has no InfiniteLine.
    lane = waveform_view.graphics_layout.getItem(1, 0)
    lane_lines = [it for it in lane.items if isinstance(it, pg.InfiniteLine)]
    assert lane_lines == [], (
        f"reserved energy lane must NOT contain InfiniteLine items; got {lane_lines}"
    )


def test_setxlink_propagates_xrange(waveform_view: WaveformView, qtbot) -> None:
    """Pin 8: setting waveform_plot's x-range propagates to the reserved lane via setXLink."""
    # Need data so setXRange has a defined coordinate system. Use a small fixture.
    pairs = _make_int16_pairs(2000)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)

    waveform_view.waveform_plot.setXRange(0.0, 5.0, padding=0)
    qtbot.wait(20)
    lane = waveform_view.graphics_layout.getItem(1, 0)
    lane_x = lane.getViewBox().viewRange()[0]
    assert lane_x[0] == pytest.approx(0.0, abs=0.05)
    assert lane_x[1] == pytest.approx(5.0, abs=0.05)


# =========================================================================
# Phase 1 invariants — pins 4 + 5 + 6
# =========================================================================
def test_four_flag_contract_preserved(waveform_view: WaveformView) -> None:
    """Pin 4: the Phase 1 four-flag PlotDataItem contract is intact on the waveform.

    Reads back through the public PlotDataItem.opts dict — same approach as
    Phase 1's test_waveform_view.py for pen width. The four flags:
        - autoDownsample = True
        - autoDownsampleMethod = 'peak'
        - clipToView = True
        - skipFiniteCheck = True
        - pen color = '#7FBFFF'
    """
    items = waveform_view.waveform_plot.listDataItems()
    assert len(items) == 1, f"expected one PlotDataItem on waveform_plot, got {len(items)}"
    pdi = items[0]
    assert pdi.opts.get("autoDownsample") is True
    assert pdi.opts.get("autoDownsampleMethod") == "peak"
    assert pdi.opts.get("clipToView") is True
    assert pdi.opts.get("skipFiniteCheck") is True
    pen = pdi.opts.get("pen")
    assert pen is not None
    assert pen.color().name().lower() == "#7fbfff", (
        f"pen color must be #7FBFFF; got {pen.color().name()}"
    )


def test_plot_widget_shim_still_works(waveform_view: WaveformView) -> None:
    """Pin 5: `plot_widget` shim exposes .plotItem, .viewport(), still
    works as the Phase 1 idiom."""
    # Populate the curve so listDataItems is non-empty.
    pairs = _make_int16_pairs(1024)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)

    pw = waveform_view.plot_widget
    # .plotItem → the waveform PlotItem.
    assert pw.plotItem is waveform_view.waveform_plot
    items = pw.plotItem.listDataItems()
    assert len(items) == 1
    # .viewport() returns a usable QWidget (event filter installable).
    vp = pw.viewport()
    assert vp is not None
    # Sanity — viewport is the same object the eventFilter saw at __init__ time.
    # We don't have direct access; we verify the type is reasonable.
    from PySide6.QtWidgets import QWidget as _QWidget

    assert isinstance(vp, _QWidget)


def test_render_proxy_x_array_still_float32(waveform_view: WaveformView) -> None:
    """Pin 6: CR-01 regression gate — render_proxy x-array dtype is float32."""
    pairs = _make_int16_pairs(2048)
    waveform_view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=256)
    items = waveform_view.waveform_plot.listDataItems()
    assert len(items) == 1
    x, _y = items[0].getData()
    assert x.dtype == np.float32, f"x-array dtype must be float32 (CR-01); got {x.dtype}"


# =========================================================================
# Mouse-down-x tracking — pin 9
# =========================================================================
def test_mouse_down_x_tracked(waveform_view: WaveformView, qtbot) -> None:
    """Pin 9: pressing the left mouse button sets _mouse_down_x_px to an int;
    releasing clears it. NO signal is emitted in Plan 02-01 — Plan 02-04 wires
    seek_requested."""
    # Initial state.
    assert waveform_view._mouse_down_x_px is None
    # The constant must be importable + correctly valued.
    assert SEEK_THRESHOLD_PX == 4
    assert MAX_RENDER_PROXY_PAIRS == 4000

    # Synthesise a real QMouseEvent at (50, 30) inside the viewport. Synthetic
    # presses dispatched via QApplication.sendEvent reach our installed
    # eventFilter.
    pw = waveform_view.plot_widget
    viewport = pw.viewport()
    press_pos = QPointF(50.0, 30.0)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        press_pos,
        viewport.mapToGlobal(QPoint(50, 30)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, press)
    assert isinstance(waveform_view._mouse_down_x_px, int)
    assert waveform_view._mouse_down_x_px == 50

    release_pos = QPointF(52.0, 30.0)
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        release_pos,
        viewport.mapToGlobal(QPoint(52, 30)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, release)
    assert waveform_view._mouse_down_x_px is None


# =========================================================================
# End-to-end open→render — pin 10
# =========================================================================
def test_open_to_render_still_works(
    qtbot,
    qapp,
    tmp_cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin 10: opening a small WAV through MainWindow still renders into
    waveform_plot. The reserved Energy lane stays height-0 (no data wired yet
    — that lands in Plan 02-02)."""
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    # 2-second sine fixture — fastest end-to-end open path.
    src = tmp_path / "e2e.wav"
    make_sine(src, freq_hz=440.0, amp=0.5, duration_s=2.0, sample_rate=44100, channels=1)

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(src), "Audio files (*.wav *.flac *.mp3)")),
    )

    with qtbot.waitSignal(window.render_complete, timeout=10000, raising=True):
        window._action_open_file()

    # Curve is on the waveform PlotItem.
    items = window._waveform_view.plot_widget.plotItem.listDataItems()
    assert len(items) == 1
    x, y = items[0].getData()
    assert x is not None and len(x) > 0
    assert y is not None and len(y) > 0

    # Reserved Energy lane stays hidden in Plan 02-01.
    lane = window._waveform_view.graphics_layout.getItem(1, 0)
    assert lane.maximumHeight() == 0, (
        f"reserved lane must remain height=0 in Plan 02-01; got {lane.maximumHeight()}"
    )
