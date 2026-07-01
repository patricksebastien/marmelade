"""Plan 03-01 Task 2 — Shift+drag region-create gesture (REG-01 thin).

Five pins covering the WaveformView's new Shift+drag branch:

* Shift+drag from x=200 to x=400 creates a single region with a positive
  duration in data-space seconds.
* The drag emits ``RegionsOverlay.regions_changed`` exactly once.
* A plain drag (no Shift) creates zero regions and does not break the
  existing pan + click-to-seek invariants (Phase 2 regression gate).
* A Shift+click ≤ SEEK_THRESHOLD_PX delta does NOT create a region —
  the ``min_width_sec`` gate in commit_draft handles the dead zone.
* New regions get UUID4-hex region ids (32 lowercase hex chars).

Tests run under ``QT_QPA_PLATFORM=offscreen``; synthetic QMouseEvents are
dispatched via QApplication.sendEvent so the eventFilter sees them with
zero queueing — same helper shape as ``test_click_to_seek.py``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from marmelade.ui import theme
from marmelade.ui.regions_overlay import RegionsOverlay
from marmelade.ui.waveform_view import SEEK_THRESHOLD_PX, WaveformView


@pytest.fixture
def waveform_view_with_overlay(qtbot, qapp):
    """Build a WaveformView, render a ~40s synthetic waveform, attach a RegionsOverlay."""
    theme.apply_theme(QApplication.instance())
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(800, 200)
    # Render a known-duration waveform (4000 pairs, sr=44100, spp=441 ≈ 40 s).
    n = 4000
    t = np.arange(n) * (2 * np.pi / 64)
    base = (np.sin(t) * 16000).astype(np.int16)
    pairs = np.empty((n, 2), dtype=np.int16)
    pairs[:, 0] = -base
    pairs[:, 1] = base
    view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=441)
    QApplication.processEvents()
    # Attach the regions overlay. Duration is read lazily via the provider.
    overlay = RegionsOverlay(
        plot_item=view.waveform_plot,
        duration_s_provider=lambda: view._duration_s,
    )
    view.set_regions_overlay(overlay)
    return view, overlay


def _send_press(
    view: WaveformView,
    x: int,
    modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    y: int = 30,
) -> None:
    """Dispatch a synthetic left-button MouseButtonPress to the viewport."""
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        viewport.mapToGlobal(QPoint(x, y)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        modifier,
    )
    QApplication.sendEvent(viewport, press)


def _send_move(
    view: WaveformView,
    x: int,
    modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    y: int = 30,
) -> None:
    """Dispatch a synthetic MouseMove to the viewport while LeftButton is held."""
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        pos,
        viewport.mapToGlobal(QPoint(x, y)).toPointF(),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        modifier,
    )
    QApplication.sendEvent(viewport, move)


def _send_release(
    view: WaveformView,
    x: int,
    modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    y: int = 30,
) -> None:
    """Dispatch a synthetic left-button MouseButtonRelease to the viewport."""
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        pos,
        viewport.mapToGlobal(QPoint(x, y)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        modifier,
    )
    QApplication.sendEvent(viewport, release)


def test_shift_drag_creates_region(waveform_view_with_overlay, qtbot) -> None:
    """Shift+drag from x=200 to x=400 creates one region with positive duration."""
    view, overlay = waveform_view_with_overlay
    _send_press(view, x=200, modifier=Qt.KeyboardModifier.ShiftModifier)
    _send_move(view, x=300, modifier=Qt.KeyboardModifier.ShiftModifier)
    _send_release(view, x=400, modifier=Qt.KeyboardModifier.ShiftModifier)
    assert len(overlay._regions) == 1
    (region,) = overlay._regions.values()
    start_s, end_s = region.getRegion()
    assert start_s < end_s
    # Region falls inside the source's duration range.
    assert 0.0 <= start_s < view._duration_s
    assert end_s <= view._duration_s + 0.5  # tolerance for viewport→data mapping


def test_shift_drag_emits_regions_changed(waveform_view_with_overlay, qtbot) -> None:
    """The drag emits ``regions_changed`` exactly once on mouse-release."""
    view, overlay = waveform_view_with_overlay
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        _send_press(view, x=200, modifier=Qt.KeyboardModifier.ShiftModifier)
        _send_move(view, x=300, modifier=Qt.KeyboardModifier.ShiftModifier)
        _send_release(view, x=400, modifier=Qt.KeyboardModifier.ShiftModifier)


def test_plain_drag_does_not_create_region(waveform_view_with_overlay, qtbot) -> None:
    """A plain drag (no Shift) creates zero regions — falls through to pan."""
    view, overlay = waveform_view_with_overlay
    _send_press(view, x=200)  # no modifier
    _send_move(view, x=300)
    _send_release(view, x=400)
    assert len(overlay._regions) == 0


def test_plain_short_click_still_seeks(waveform_view_with_overlay, qtbot) -> None:
    """Click-to-seek invariant: a press/release ≤ 4 px (no Shift) still emits seek_requested."""
    view, _overlay = waveform_view_with_overlay
    with qtbot.waitSignal(view.seek_requested, timeout=1000):
        _send_press(view, x=200)
        _send_release(view, x=201)


def test_click_under_threshold_with_shift_does_not_create_region(
    waveform_view_with_overlay, qtbot
) -> None:
    """Shift+press at x=200, Shift+release at x=200 (zero data-x delta) — no region.

    The ``min_width_sec`` gate in :meth:`RegionsOverlay.commit_draft` rejects
    drafts whose data-space width is below ~0.001 s, so a Shift+click in
    place (zero pixel delta) does NOT create a region.
    """
    view, overlay = waveform_view_with_overlay
    _send_press(view, x=200, modifier=Qt.KeyboardModifier.ShiftModifier)
    _send_release(view, x=200, modifier=Qt.KeyboardModifier.ShiftModifier)
    assert len(overlay._regions) == 0


def test_commit_draft_returns_region_with_uuid_id(
    waveform_view_with_overlay, qtbot
) -> None:
    """Newly committed regions get a 32-char lowercase hex UUID4 id."""
    view, overlay = waveform_view_with_overlay
    _send_press(view, x=200, modifier=Qt.KeyboardModifier.ShiftModifier)
    _send_move(view, x=300, modifier=Qt.KeyboardModifier.ShiftModifier)
    _send_release(view, x=400, modifier=Qt.KeyboardModifier.ShiftModifier)
    assert len(overlay._regions) == 1
    region_id = next(iter(overlay._regions.keys()))
    assert re.fullmatch(r"[0-9a-f]{32}", region_id), (
        f"region id {region_id!r} is not a UUID4 hex"
    )


def test_set_regions_populates_overlay(waveform_view_with_overlay, qtbot) -> None:
    """``set_regions`` clears + rebuilds the overlay's LinearRegionItem list."""
    from marmelade.audio.sidecar_cache import Region

    view, overlay = waveform_view_with_overlay
    regions = [
        Region(id="aaaa", start_sec=1.0, end_sec=2.0, state="untouched"),
        Region(id="bbbb", start_sec=10.0, end_sec=12.0, state="untouched"),
    ]
    overlay.set_regions(regions)
    assert len(overlay._regions) == 2
    assert "aaaa" in overlay._regions
    assert "bbbb" in overlay._regions
    # Re-setting clears the previous list.
    overlay.set_regions([Region(id="cccc", start_sec=5.0, end_sec=6.0)])
    assert len(overlay._regions) == 1
    assert "cccc" in overlay._regions
    assert "aaaa" not in overlay._regions


def test_regions_data_round_trips(waveform_view_with_overlay, qtbot) -> None:
    """``regions_data`` returns a fresh list[Region] matching current overlay state."""
    from marmelade.audio.sidecar_cache import Region

    view, overlay = waveform_view_with_overlay
    initial = [
        Region(id="aaaa", start_sec=1.0, end_sec=2.0, state="untouched"),
        Region(id="bbbb", start_sec=10.0, end_sec=12.0, state="untouched"),
    ]
    overlay.set_regions(initial)
    out = overlay.regions_data()
    assert len(out) == 2
    ids = {r.id for r in out}
    assert ids == {"aaaa", "bbbb"}
    for r in out:
        assert r.start_sec < r.end_sec


def test_resize_only_region_body_drag_ignored(waveform_view_with_overlay) -> None:
    """``ResizeOnlyRegion.mouseDragEvent`` calls ev.ignore() (body-drag disabled)."""
    from marmelade.audio.sidecar_cache import Region
    from marmelade.ui.regions_overlay import ResizeOnlyRegion

    view, overlay = waveform_view_with_overlay
    overlay.set_regions(
        [Region(id="aaaa", start_sec=1.0, end_sec=2.0, state="untouched")]
    )
    region = overlay._regions["aaaa"]
    assert isinstance(region, ResizeOnlyRegion)

    # Synthesize a minimal mouse-drag event-like object and ensure ev.ignore() is called.
    class _FakeEvent:
        def __init__(self) -> None:
            self.ignored = False

        def ignore(self) -> None:
            self.ignored = True

    fake = _FakeEvent()
    region.mouseDragEvent(fake)
    assert fake.ignored, "ResizeOnlyRegion.mouseDragEvent must call ev.ignore()"


# ============================================================================
# Plan 03-02 Task 3 — toolbar Region Select mode + middle-mouse pan extension
# ============================================================================
def test_toolbar_mode_off_plain_drag_does_not_create_region(
    waveform_view_with_overlay, qtbot
) -> None:
    """With Region Select mode OFF, a plain Left+drag does the pan (no region)."""
    view, overlay = waveform_view_with_overlay
    view.set_region_select_mode(False)
    _send_press(view, x=200)  # no modifier, no shift
    _send_move(view, x=300)
    _send_release(view, x=400)
    assert len(overlay._regions) == 0


def test_toolbar_mode_on_plain_drag_creates_region(
    waveform_view_with_overlay, qtbot
) -> None:
    """With Region Select mode ON, a plain Left+drag creates a region."""
    view, overlay = waveform_view_with_overlay
    view.set_region_select_mode(True)
    _send_press(view, x=200)  # no modifier — but mode is ON
    _send_move(view, x=300)
    _send_release(view, x=400)
    assert len(overlay._regions) == 1


def test_set_region_select_mode_swaps_cursor(waveform_view_with_overlay, qtbot) -> None:
    """``set_region_select_mode`` swaps the viewport cursor to/from CrossCursor."""
    view, _overlay = waveform_view_with_overlay
    viewport = view.graphics_layout.viewport()
    view.set_region_select_mode(True)
    assert viewport.cursor().shape() == Qt.CursorShape.CrossCursor
    view.set_region_select_mode(False)
    # OpenHandCursor is the idle state per UI-SPEC §Pan.
    assert viewport.cursor().shape() == Qt.CursorShape.OpenHandCursor


def test_middle_button_drag_does_not_create_region(
    waveform_view_with_overlay, qtbot
) -> None:
    """Middle-button drag never creates a region regardless of mode."""
    view, overlay = waveform_view_with_overlay
    view.set_region_select_mode(True)  # even with mode ON

    viewport = view.graphics_layout.viewport()

    def _send_button(
        type_: QEvent.Type, x: int, button: Qt.MouseButton, buttons: Qt.MouseButton
    ) -> None:
        pos = QPointF(float(x), 30.0)
        ev = QMouseEvent(
            type_,
            pos,
            viewport.mapToGlobal(QPoint(x, 30)).toPointF(),
            button,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, ev)

    _send_button(
        QEvent.Type.MouseButtonPress,
        x=200,
        button=Qt.MouseButton.MiddleButton,
        buttons=Qt.MouseButton.MiddleButton,
    )
    _send_button(
        QEvent.Type.MouseMove,
        x=300,
        button=Qt.MouseButton.NoButton,
        buttons=Qt.MouseButton.MiddleButton,
    )
    _send_button(
        QEvent.Type.MouseButtonRelease,
        x=400,
        button=Qt.MouseButton.MiddleButton,
        buttons=Qt.MouseButton.NoButton,
    )
    assert len(overlay._regions) == 0


def test_middle_button_drag_pans_view(waveform_view_with_overlay, qtbot) -> None:
    """Middle+drag pans the waveform ViewBox in X (range shifts)."""
    view, _overlay = waveform_view_with_overlay
    vb = view.waveform_plot.getViewBox()
    # Establish a known starting range.
    vb.setXRange(5.0, 15.0, padding=0)
    QApplication.processEvents()
    (x_min_before, x_max_before), _ = vb.viewRange()

    viewport = view.graphics_layout.viewport()

    def _send_button(
        type_: QEvent.Type, x: int, button: Qt.MouseButton, buttons: Qt.MouseButton
    ) -> None:
        pos = QPointF(float(x), 30.0)
        ev = QMouseEvent(
            type_,
            pos,
            viewport.mapToGlobal(QPoint(x, 30)).toPointF(),
            button,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, ev)

    _send_button(
        QEvent.Type.MouseButtonPress,
        x=200,
        button=Qt.MouseButton.MiddleButton,
        buttons=Qt.MouseButton.MiddleButton,
    )
    _send_button(
        QEvent.Type.MouseMove,
        x=300,
        button=Qt.MouseButton.NoButton,
        buttons=Qt.MouseButton.MiddleButton,
    )
    _send_button(
        QEvent.Type.MouseButtonRelease,
        x=400,
        button=Qt.MouseButton.MiddleButton,
        buttons=Qt.MouseButton.NoButton,
    )
    QApplication.processEvents()
    (x_min_after, x_max_after), _ = vb.viewRange()
    # The range was shifted by middle-mouse pan — exact magnitude depends
    # on pixel-to-data mapping; just assert non-trivial change.
    assert abs(x_min_after - x_min_before) > 0.01, (
        f"middle-drag should pan x-range; before=({x_min_before:.3f},"
        f"{x_max_before:.3f}) after=({x_min_after:.3f},{x_max_after:.3f})"
    )


def test_shift_drag_works_even_with_mode_off(waveform_view_with_overlay, qtbot) -> None:
    """Combo gesture — Shift+drag always creates regions, regardless of mode."""
    view, overlay = waveform_view_with_overlay
    view.set_region_select_mode(False)  # mode OFF
    _send_press(view, x=200, modifier=Qt.KeyboardModifier.ShiftModifier)
    _send_move(view, x=300, modifier=Qt.KeyboardModifier.ShiftModifier)
    _send_release(view, x=400, modifier=Qt.KeyboardModifier.ShiftModifier)
    assert len(overlay._regions) == 1
