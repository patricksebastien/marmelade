"""Plan 03-02 Task 1 — edge-resize behavior + body-drag suppression (REG-01).

Pins covering the survived Plan 01 contract under Plan 02 styling:

* ``setRegion((lo, hi))`` mutates the region and the ``regions_changed``
  signal fires (sigRegionChangeFinished is wired to ``_on_region_changed``).
* ``regions_data`` reflects the new range after a setRegion.
* ``ResizeOnlyRegion.mouseDragEvent`` ignores the event (body-drag stays
  suppressed regardless of state styling).
* Per-state styling does NOT break the underlying ``setRegion`` /
  ``getRegion`` round-trip.

These are integration pins — they protect against Plan 02 regressions on
the Plan 01 thin-slice gestures.
"""

from __future__ import annotations

import pytest

import pyqtgraph as pg

from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui.regions_overlay import RegionsOverlay, ResizeOnlyRegion


@pytest.fixture
def overlay_with_region(qtbot, qapp):
    """See note in test_regions_overlay_hover_target_delete.py — the
    GraphicsLayoutWidget MUST be registered with qtbot."""
    glw = pg.GraphicsLayoutWidget()
    qtbot.addWidget(glw)
    plot = glw.addPlot()
    overlay = RegionsOverlay(
        plot_item=plot,
        duration_s_provider=lambda: 100.0,
    )
    overlay.set_regions(
        [Region(id="rrr1", start_sec=10.0, end_sec=20.0, state="untouched")]
    )
    yield overlay, overlay._regions["rrr1"]
    # Teardown — clear overlay before GLW cleanup (libshiboken
    # InfiniteLine/ViewBox already-deleted race; see note in
    # test_regions_overlay_context_menu.py).
    try:
        overlay.clear()
    except Exception:
        pass


def test_edge_drag_via_setRegion_changes_range(overlay_with_region, qtbot) -> None:
    """Programmatic setRegion (proxy for an edge-drag) updates the range."""
    _overlay, region = overlay_with_region
    region.setRegion((10.0, 25.0))
    start_s, end_s = region.getRegion()
    assert start_s == pytest.approx(10.0)
    assert end_s == pytest.approx(25.0)


def test_setRegion_emits_regions_changed(overlay_with_region, qtbot) -> None:
    """An edge-drag finish should fire ``regions_changed`` via the wired ``sigRegionChangeFinished``."""
    overlay, region = overlay_with_region
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        region.setRegion((10.0, 25.0))


def test_body_drag_event_is_ignored(overlay_with_region) -> None:
    """``ResizeOnlyRegion.mouseDragEvent`` must call ev.ignore() (body-drag suppressed)."""
    _overlay, region = overlay_with_region

    class _FakeEvent:
        def __init__(self) -> None:
            self.ignored = False

        def ignore(self) -> None:
            self.ignored = True

    fake = _FakeEvent()
    region.mouseDragEvent(fake)
    assert fake.ignored
    # Range unchanged (no movement applied).
    start_s, end_s = region.getRegion()
    assert start_s == pytest.approx(10.0)
    assert end_s == pytest.approx(20.0)


def test_setRegion_round_trips_through_regions_data(overlay_with_region) -> None:
    """``regions_data`` reflects the post-edge-drag start/end."""
    overlay, region = overlay_with_region
    region.setRegion((5.0, 30.0))
    out = overlay.regions_data()
    assert len(out) == 1
    assert out[0].start_sec == pytest.approx(5.0)
    assert out[0].end_sec == pytest.approx(30.0)


def test_resize_only_region_is_subclass_of_linear_region(overlay_with_region) -> None:
    _overlay, region = overlay_with_region
    assert isinstance(region, ResizeOnlyRegion)
    assert isinstance(region, pg.LinearRegionItem)


def test_keeper_state_preserves_resize_behavior(overlay_with_region, qtbot) -> None:
    """Marking a region Keeper does NOT lock its resize — edge-drag still works."""
    overlay, region = overlay_with_region
    overlay.set_state(region.region_id, "keeper")
    region.setRegion((10.0, 30.0))
    start_s, end_s = region.getRegion()
    assert end_s == pytest.approx(30.0)


def test_swap_mode_sort_preserves_lo_le_hi(overlay_with_region) -> None:
    """PyQtGraph swapMode='sort' guarantees getRegion() returns (lo, hi) sorted."""
    overlay, region = overlay_with_region
    # Set with hi < lo — sort mode reorders.
    region.setRegion((30.0, 5.0))
    start_s, end_s = region.getRegion()
    assert start_s <= end_s
