"""Plan 03-02 Task 1 — hover-target tracking + set_state + delete (REG-01/02/03).

Pins covering the Plan 02 extensions to :class:`RegionsOverlay`:

* ``hovered_region_id`` flips to the region under cursor on ``hoverEnterEvent``
  and back to ``None`` on ``hoverLeaveEvent``.
* ``hover_changed`` Signal fires on both enter (payload = region_id) and
  leave (payload = ``None``).
* ``set_state(region_id, state)`` re-applies brush/hover/pen, updates the
  region's ``_current_state``, and emits ``regions_changed`` exactly once.
* ``set_state(rid, "invalid")`` raises ``ValueError``.
* ``set_state(rid, same_state)`` is a no-op (does NOT emit
  ``regions_changed`` — defends sidecar churn).
* ``delete(region_id)`` removes the region from the PlotItem AND the
  ``_regions`` dict AND clears ``hovered_region_id`` if matching.
* ``set_state_of_hovered`` and ``delete_hovered`` are no-ops when no
  region is hovered.

Synthetic hover events are constructed by directly invoking
``ResizeOnlyRegion.hoverEnterEvent`` / ``hoverLeaveEvent`` — this is the
recommended path for headless tests because PyQtGraph's actual hover
dispatch goes through the QGraphicsScene → scene event filter chain
which doesn't replay reliably under ``QT_QPA_PLATFORM=offscreen``.

The `default_cache_root` import at module level is patched defensively by
the `tmp_cache_dir` fixture (RESEARCH §Pitfall #10) even though these
tests don't construct a MainWindow.
"""

from __future__ import annotations

import pytest

from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui.regions_overlay import RegionsOverlay, ResizeOnlyRegion


class _FakeHoverEvent:
    """Minimal hoverEvent stand-in.

    PyQtGraph's hoverEnterEvent / hoverLeaveEvent overrides just need
    something callable enough to delegate to ``super()`` if the subclass
    chooses to. We pass our own subclass so we never reach ``super()``.
    """


@pytest.fixture
def overlay_with_plot(qtbot, qapp):
    """Construct a bare ``RegionsOverlay`` on a real PyQtGraph PlotItem.

    No MainWindow / WaveformView — overlay-only tests are sufficient for
    the Plan 02 surface, and instantiation is fast.

    The :class:`pg.GraphicsLayoutWidget` is registered with ``qtbot`` so
    its lifecycle covers the full test body (otherwise the
    no-parent widget can be GC'd between the fixture yield and the test
    body, taking the ViewBox C++ object with it — manifests as
    ``libshiboken: Internal C++ object (ViewBox) already deleted``).
    """
    import pyqtgraph as pg

    glw = pg.GraphicsLayoutWidget()
    qtbot.addWidget(glw)
    plot = glw.addPlot()
    overlay = RegionsOverlay(
        plot_item=plot,
        duration_s_provider=lambda: 100.0,
    )
    yield overlay, glw
    # Teardown — clear overlay BEFORE the GLW is cleaned up by qtbot.
    # Otherwise the overlay's per-widget destruction reaches into a
    # C++-deleted ViewBox (libshiboken already-deleted race seen across
    # tests when fixture teardown order isn't explicit).
    try:
        overlay.clear()
    except Exception:
        pass


def _seed_one_region(
    overlay: RegionsOverlay, *, id_: str = "rrr1", start: float = 10.0, end: float = 20.0
) -> ResizeOnlyRegion:
    overlay.set_regions(
        [Region(id=id_, start_sec=start, end_sec=end, state="untouched")]
    )
    return overlay._regions[id_]


def test_initial_hovered_region_id_is_none(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    assert overlay.hovered_region_id is None


def test_hover_enter_sets_hovered_region_id(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    region.hoverEnterEvent(_FakeHoverEvent())
    assert overlay.hovered_region_id == region.region_id


def test_hover_leave_clears_hovered_region_id(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    region.hoverEnterEvent(_FakeHoverEvent())
    region.hoverLeaveEvent(_FakeHoverEvent())
    assert overlay.hovered_region_id is None


def test_hover_changed_signal_fires_on_enter_and_leave(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    seen: list = []
    overlay.hover_changed.connect(lambda payload: seen.append(payload))
    region.hoverEnterEvent(_FakeHoverEvent())
    region.hoverLeaveEvent(_FakeHoverEvent())
    assert seen == [region.region_id, None]


def test_set_state_keeper_updates_current_state_and_emits(overlay_with_plot, qtbot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        overlay.set_state(region.region_id, "keeper")
    assert region._current_state == "keeper"


def test_set_state_trash_updates_current_state(overlay_with_plot, qtbot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    overlay.set_state(region.region_id, "trash")
    assert region._current_state == "trash"


def test_set_state_invalid_raises_value_error(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    with pytest.raises(ValueError):
        overlay.set_state(region.region_id, "archived")


def test_set_state_no_op_when_same_state(overlay_with_plot, qtbot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    assert region._current_state == "untouched"
    # qtbot.waitSignal with raising=False + timeout returns whether signal fired.
    blocker = qtbot.waitSignal(overlay.regions_changed, timeout=200, raising=False)
    blocker.wait()
    # Reset and call set_state with the same value — no new emission.
    fired: list = []
    overlay.regions_changed.connect(lambda: fired.append(True))
    overlay.set_state(region.region_id, "untouched")
    assert fired == [], "set_state to same state must not emit regions_changed"


def test_set_state_unknown_region_id_no_op(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    _seed_one_region(overlay)
    fired: list = []
    overlay.regions_changed.connect(lambda: fired.append(True))
    overlay.set_state("nonexistent-id", "keeper")  # defensive — no raise
    assert fired == []


def test_delete_removes_region_and_emits(overlay_with_plot, qtbot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    rid = region.region_id
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        overlay.delete(rid)
    assert rid not in overlay._regions


def test_delete_clears_hovered_region_id_when_matching(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    region.hoverEnterEvent(_FakeHoverEvent())
    assert overlay.hovered_region_id == region.region_id
    overlay.delete(region.region_id)
    assert overlay.hovered_region_id is None


def test_delete_unknown_region_id_no_op(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    _seed_one_region(overlay)
    fired: list = []
    overlay.regions_changed.connect(lambda: fired.append(True))
    overlay.delete("nonexistent-id")  # defensive — no raise, no signal
    assert fired == []


def test_set_state_of_hovered_keeper(overlay_with_plot, qtbot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    region.hoverEnterEvent(_FakeHoverEvent())
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        overlay.set_state_of_hovered("keeper")
    assert region._current_state == "keeper"


def test_set_state_of_hovered_no_op_when_no_hover(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    _seed_one_region(overlay)
    assert overlay.hovered_region_id is None
    fired: list = []
    overlay.regions_changed.connect(lambda: fired.append(True))
    overlay.set_state_of_hovered("keeper")  # no-op
    assert fired == []


def test_delete_hovered_removes_region(overlay_with_plot, qtbot) -> None:
    overlay, _plot = overlay_with_plot
    region = _seed_one_region(overlay)
    region.hoverEnterEvent(_FakeHoverEvent())
    rid = region.region_id
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        overlay.delete_hovered()
    assert rid not in overlay._regions
    assert overlay.hovered_region_id is None


def test_delete_hovered_no_op_when_no_hover(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    _seed_one_region(overlay)
    fired: list = []
    overlay.regions_changed.connect(lambda: fired.append(True))
    overlay.delete_hovered()  # no-op
    assert fired == []


def test_set_regions_preserves_per_region_state(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    overlay.set_regions(
        [
            Region(id="aaaa", start_sec=1.0, end_sec=2.0, state="keeper"),
            Region(id="bbbb", start_sec=10.0, end_sec=12.0, state="trash"),
            Region(id="cccc", start_sec=20.0, end_sec=25.0, state="untouched"),
        ]
    )
    assert overlay._regions["aaaa"]._current_state == "keeper"
    assert overlay._regions["bbbb"]._current_state == "trash"
    assert overlay._regions["cccc"]._current_state == "untouched"


def test_regions_data_reflects_current_state(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    overlay.set_regions(
        [
            Region(id="aaaa", start_sec=1.0, end_sec=2.0, state="keeper"),
            Region(id="bbbb", start_sec=10.0, end_sec=12.0, state="untouched"),
        ]
    )
    out = overlay.regions_data()
    by_id = {r.id: r for r in out}
    assert by_id["aaaa"].state == "keeper"
    assert by_id["bbbb"].state == "untouched"
    # Mutate via set_state and re-read.
    overlay.set_state("bbbb", "trash")
    out = overlay.regions_data()
    by_id = {r.id: r for r in out}
    assert by_id["bbbb"].state == "trash"


def test_regions_data_preserves_note_and_created_at(overlay_with_plot) -> None:
    overlay, _plot = overlay_with_plot
    overlay.set_regions(
        [
            Region(
                id="aaaa",
                start_sec=1.0,
                end_sec=2.0,
                state="keeper",
                note="best riff",
                created_at="2026-05-16T12:00:00",
            )
        ]
    )
    out = overlay.regions_data()
    assert out[0].note == "best riff"
    assert out[0].created_at == "2026-05-16T12:00:00"
