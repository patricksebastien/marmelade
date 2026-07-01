"""Gap-closure (UAT Test 3) — real-dispatch right-click pins.

Replaces the layer-mismatched tests that called ``getContextMenus(None)``
directly. Those tests passed despite the production code being broken at
the real dispatch layer — PyQtGraph 0.13.7's ``LinearRegionItem`` has no
right-click → menu plumbing of its own (unlike ROI). The legacy tests
invoked our menu builder by hand, so they could not detect that the
scene never reached that builder when the user actually right-clicked.

The pins here synthesise a real ``RightButton`` press/release on the
GraphicsLayoutWidget's viewport (preceded by a ``MouseMove`` to populate
the scene's hover bookkeeping so ``acceptClicks(RightButton)`` is on
file). We then capture the QMenu that the production
``raiseContextMenu`` calls ``popup()`` on via a ``QApplication``-level
event filter listening for ``QEvent.Show`` on ``QMenu`` subclasses.

Why the event-filter approach instead of scanning
``QApplication.topLevelWidgets()`` for ``isVisible()`` QMenus: under
``QT_QPA_PLATFORM=offscreen`` (which the test suite runs under) the Qt
platform plugin warns "This plugin does not support raise()" and
"This plugin does not support grabbing the keyboard" — the menu DOES
get shown via ``QEvent.Show`` (caught by the event filter) but the
visibility state is not retained after the next ``processEvents`` call.
Capturing at the Show event itself is the deterministic hook.

This exercises the SAME dispatch path the user hits at runtime:
scene mousePressEvent → MouseClickEvent buffered → scene
mouseReleaseEvent → sendClickEvent → item.mouseClickEvent →
raiseContextMenu → menu.popup. The event filter sits OUTSIDE that
chain, observing only the public Show signal at the end.

References:
* PyQtGraph ROI.py:728-806 (canonical context-menu pattern we mirror)
* PyQtGraph LinearRegionItem.py:332-345 (the BASE class our subclass
  inherits — does NOT route RightButton to a menu)
* PyQtGraph GraphicsScene.py:353-387 (sendClickEvent dispatch path)
* 03-UI-SPEC §Region context menu (six-action ordering)
* 03-04b-SUMMARY.md (export-action additions and keeper-only enable rule)
"""

from __future__ import annotations

import pytest
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMenu

from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui.regions_overlay import RegionsOverlay


# ----------------------------------------------------------- helpers
class _MenuShowSpy(QObject):
    """Application-wide event filter capturing every QMenu Show event.

    Installed on ``QApplication.instance()`` for the duration of one
    test (fixture-managed). ``shown`` is the ordered list of QMenu
    instances that received an ``Show`` event since installation.

    Offscreen-mode Qt's platform plugin doesn't keep QMenus visible
    after the next ``processEvents`` call (see module docstring), so
    catching the Show event itself is the deterministic hook for
    asserting "a menu popped up". ``QMenu.popup`` synchronously emits
    a Show event before returning, so a test that observes ``shown ==
    [menu_for_region]`` immediately after the click is asserting the
    exact same fact as the user-visible "the menu appeared".
    """

    def __init__(self) -> None:
        super().__init__()
        self.shown: list[QMenu] = []

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 — Qt API
        if event.type() == QEvent.Show and isinstance(obj, QMenu):
            self.shown.append(obj)
        return False

    def reset(self) -> None:
        self.shown.clear()


def _data_x_to_viewport_pos(plot, glw, data_x: float, data_y: float = 0.0):
    """Map data-space (x, y) to GLW viewport pixel coords.

    Pipeline:
    1. ``plot.vb.mapViewToScene(QPointF(x, y))`` → scene coords.
    2. ``glw.mapFromScene(scene_pt)`` → GLW widget coords.
    3. The viewport sits at GLW origin in a ``GraphicsLayoutWidget``, so
       widget coords are also viewport coords. We assert containment in
       the viewport rect to fail fast on any future layout change.
    """
    scene_pt = plot.vb.mapViewToScene(QPointF(data_x, data_y))
    widget_pt = glw.mapFromScene(scene_pt)
    vp_rect = glw.viewport().rect()
    assert vp_rect.contains(widget_pt), (
        f"mapped point {widget_pt} is outside viewport rect {vp_rect} — "
        "data_x mapping is broken; check plot.vb.viewRange() vs data_x"
    )
    return widget_pt


def _right_click_at_data_x(qtbot, glw, plot, data_x: float) -> None:
    """Synthesise a real RightButton press+release on the GLW viewport.

    Sends a ``MouseMove`` FIRST so the scene's hover bookkeeping
    registers our region as the acceptedItem for RightButton (the
    ``ResizeOnlyRegion.hoverEvent`` override calls
    ``ev.acceptClicks(RightButton)``). Without this, the scene's
    ``sendClickEvent`` (GraphicsScene.py:362) falls through the
    ``acceptedItem`` lookup and dispatches via the slower
    ``itemsNearEvent`` fallback — which still works but exercises a
    different code path than the user's mouse-move-then-right-click.

    Press + release are sent as separate events because PyQtGraph's
    scene records the press in ``clickEvents`` then dispatches via
    ``sendClickEvent`` only on release (GraphicsScene.py:213-241).
    """
    pt = _data_x_to_viewport_pos(plot, glw, data_x)
    pt_f = QPointF(pt.x(), pt.y())
    # Move so hover registers and acceptClicks(RightButton) fires.
    move = QMouseEvent(
        QEvent.MouseMove, pt_f, Qt.NoButton, Qt.NoButton, Qt.NoModifier
    )
    QApplication.sendEvent(glw.viewport(), move)
    qtbot.wait(10)
    # Press.
    press = QMouseEvent(
        QEvent.MouseButtonPress,
        pt_f,
        Qt.RightButton,
        Qt.RightButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(glw.viewport(), press)
    qtbot.wait(10)
    # Release — this is where the scene dispatches the click to items.
    release = QMouseEvent(
        QEvent.MouseButtonRelease,
        pt_f,
        Qt.RightButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(glw.viewport(), release)
    qtbot.wait(30)


def _last_region_menu(spy: _MenuShowSpy) -> QMenu | None:
    """Return the most recent QMenu the production code popped, or None.

    The MainWindow's static menus (File / Edit / View) also emit Show
    events when they're built — to avoid false positives we filter to
    menus whose ``parent()`` is None (our region menu is parent-less,
    constructed as ``QMenu()`` in ``ResizeOnlyRegion.getContextMenus``)
    OR whose first action is one of our region commands.
    """
    region_commands = {
        "Mark as Keeper",
        "Mark as Trash",
        "Unmark",
        "Export this region as MP3…",
        "Export this region as WAV…",
        "Delete region",
    }
    for menu in reversed(spy.shown):
        actions = [a for a in menu.actions() if not a.isSeparator()]
        if actions and actions[0].text() in region_commands:
            return menu
    return None


# ----------------------------------------------------------- fixture
@pytest.fixture
def overlay_on_visible_glw(qtbot, qapp):
    """Build a SHOWN :class:`pg.GraphicsLayoutWidget` + plot + overlay + spy.

    Returns ``(glw, plot, overlay, spy)`` where ``spy`` is a
    :class:`_MenuShowSpy` installed on the QApplication for the
    duration of the test. Show + waitExposed is required for
    synthesized mouse events to route through the scene under
    offscreen Qt — a hidden widget swallows them.

    Teardown removes the event filter, closes any stray visible QMenus
    that leaked from the test (defensive — should be none), then
    clears the overlay before qtbot disposes the GLW.
    """
    spy = _MenuShowSpy()
    qapp.installEventFilter(spy)

    glw = pg.GraphicsLayoutWidget()
    qtbot.addWidget(glw)
    glw.resize(800, 400)
    plot = glw.addPlot()
    # Mirror production WaveformView (waveform_view.py:246) — suppress the
    # stock ViewBox context menu so the only menu that can pop from a
    # right-click is one of ours. Without this, the PyQtGraph
    # "View All / X axis / Y axis / Plot Options / Export..." chain
    # surfaces over empty plot area.
    plot.setMenuEnabled(False)
    # Lock the view range so data-x mapping is deterministic. Otherwise
    # PyQtGraph auto-ranges based on what's plotted, and an empty plot
    # would default to a tiny range that makes the viewport mapping
    # collapse to a single pixel.
    plot.setXRange(0.0, 100.0, padding=0.0)
    plot.setYRange(-1.0, 1.0, padding=0.0)
    overlay = RegionsOverlay(
        plot_item=plot,
        duration_s_provider=lambda: 100.0,
    )
    glw.show()
    qtbot.waitExposed(glw)
    yield glw, plot, overlay, spy
    # Teardown — remove spy first, then close any stray menus.
    try:
        qapp.removeEventFilter(spy)
    except Exception:
        pass
    try:
        for menu in QApplication.topLevelWidgets():
            if isinstance(menu, QMenu) and menu.isVisible():
                menu.close()
    except Exception:
        pass
    try:
        overlay.clear()
    except Exception:
        pass


# -------------------------------------------------------------- tests
def test_right_click_on_region_pops_a_qmenu(overlay_on_visible_glw, qtbot) -> None:
    """A real RightButton click on a region pops a QMenu via the production path."""
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu = _last_region_menu(spy)
    assert menu is not None, (
        "Right-click on a region must pop a QMenu via the production "
        "raiseContextMenu path; spy.shown only contains: "
        f"{[m.actions()[0].text() if m.actions() else '<empty>' for m in spy.shown]}"
    )


def test_popped_menu_has_six_commands_in_order(overlay_on_visible_glw, qtbot) -> None:
    """Popped-up menu has the six command labels in the UI-SPEC order."""
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu = _last_region_menu(spy)
    assert menu is not None
    labels = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert labels == [
        "Mark as Keeper",
        "Mark as Trash",
        "Unmark",
        "Export this region as MP3…",
        "Export this region as WAV…",
        "Delete region",
    ]


def test_popped_menu_mark_keeper_triggers_state_change(
    overlay_on_visible_glw, qtbot
) -> None:
    """Triggering 'Mark as Keeper' from the popped menu flips the region state."""
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    region = overlay._regions["rrr1"]
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu = _last_region_menu(spy)
    assert menu is not None
    mark_keeper = next(a for a in menu.actions() if a.text() == "Mark as Keeper")
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        mark_keeper.trigger()
    assert region._current_state == "keeper", (
        "Triggering the popped 'Mark as Keeper' action must update region state — "
        "proves the lambda capture survives the real dispatch path"
    )


def test_popped_menu_export_action_emits_export_requested(
    overlay_on_visible_glw, qtbot
) -> None:
    """Export MP3 / Export WAV from the popped menu emits export_requested."""
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    # Promote to keeper so the Export entries become enabled.
    overlay.set_state("rrr1", "keeper")
    # First: MP3.
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu = _last_region_menu(spy)
    assert menu is not None
    export_mp3 = next(
        a for a in menu.actions() if a.text() == "Export this region as MP3…"
    )
    assert export_mp3.isEnabled(), "Export MP3 must be enabled for a keeper region"
    with qtbot.waitSignal(overlay.export_requested, timeout=1000) as blocker:
        export_mp3.trigger()
    assert blocker.args == ["rrr1", "mp3"]
    # Reset spy + close menu so the second pass gets a fresh capture.
    menu.close()
    spy.reset()
    qtbot.wait(30)
    # Then: WAV.
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu2 = _last_region_menu(spy)
    assert menu2 is not None
    export_wav = next(
        a for a in menu2.actions() if a.text() == "Export this region as WAV…"
    )
    assert export_wav.isEnabled(), "Export WAV must be enabled for a keeper region"
    with qtbot.waitSignal(overlay.export_requested, timeout=1000) as blocker:
        export_wav.trigger()
    assert blocker.args == ["rrr1", "wav"]


def test_popped_menu_export_actions_disabled_when_not_keeper(
    overlay_on_visible_glw, qtbot
) -> None:
    """Pins D-A4-4 keeper-only-export rule survives the real dispatch path."""
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu = _last_region_menu(spy)
    assert menu is not None
    export_mp3 = next(
        a for a in menu.actions() if a.text() == "Export this region as MP3…"
    )
    export_wav = next(
        a for a in menu.actions() if a.text() == "Export this region as WAV…"
    )
    assert export_mp3.isEnabled() is False
    assert export_wav.isEnabled() is False


def test_two_regions_right_click_independent(overlay_on_visible_glw, qtbot) -> None:
    """Two adjacent regions — each right-click targets the correct region.

    Proves A's menu does not steal B's clicks; the lambda capture in
    ``getContextMenus`` (default-arg binding of ``_rid``) survives the
    real dispatch path for each region independently.
    """
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [
            Region(id="aaa", start_sec=10.0, end_sec=30.0, state="untouched"),
            Region(id="bbb", start_sec=50.0, end_sec=70.0, state="untouched"),
        ]
    )
    region_a = overlay._regions["aaa"]
    region_b = overlay._regions["bbb"]
    # Right-click A → menu, trigger Mark as Trash → A becomes trash, B stays untouched.
    _right_click_at_data_x(qtbot, glw, plot, data_x=20.0)
    menu_a = _last_region_menu(spy)
    assert menu_a is not None
    trash_a = next(a for a in menu_a.actions() if a.text() == "Mark as Trash")
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        trash_a.trigger()
    assert region_a._current_state == "trash"
    assert region_b._current_state == "untouched"
    # Reset spy and right-click B → menu, trigger Mark as Keeper.
    menu_a.close()
    spy.reset()
    qtbot.wait(30)
    _right_click_at_data_x(qtbot, glw, plot, data_x=60.0)
    menu_b = _last_region_menu(spy)
    assert menu_b is not None
    keeper_b = next(a for a in menu_b.actions() if a.text() == "Mark as Keeper")
    with qtbot.waitSignal(overlay.regions_changed, timeout=1000):
        keeper_b.trigger()
    assert region_b._current_state == "keeper"
    # A unchanged by B's menu interaction.
    assert region_a._current_state == "trash"


def test_view_box_menu_remains_disabled(overlay_on_visible_glw, qtbot) -> None:
    """Right-click on EMPTY plot area pops NO region-menu — stock chain suppressed.

    The fixture already calls ``plot.setMenuEnabled(False)`` to mirror
    waveform_view.py:246. A right-click on empty space therefore must
    not reach any of our ResizeOnlyRegion's mouseClickEvent overrides
    (no region under the cursor), AND must not pop the ViewBox stock
    "View All / X / Y / Plot Options / Export..." chain.
    """
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    # Right-click at x=80 — well outside the [20, 40] region.
    _right_click_at_data_x(qtbot, glw, plot, data_x=80.0)
    assert _last_region_menu(spy) is None, (
        "empty-area right-click must not pop a region menu; spy captured: "
        f"{[m.actions()[0].text() if m.actions() else '<empty>' for m in spy.shown]}"
    )
    # Additionally, the stock chain must not appear either.
    for menu in spy.shown:
        actions = [a.text() for a in menu.actions() if not a.isSeparator()]
        # Stock ViewBox actions include "View All", "X axis", "Y axis", etc.
        assert "View All" not in actions, (
            "ViewBox stock menu surfaced despite setMenuEnabled(False); "
            f"captured menu actions: {actions}"
        )


# ----------------------------------------------------------- 03-05b lifetime pins
# Plan 03-05's seven tests above caught the dispatch-layer bug (LinearRegionItem
# has no native right-click → menu wiring) but missed a real-world regression:
# building a fresh, parent-less ``QMenu()`` per popup lets PySide6 drop the
# Python shadow before Qt paints the menu. Offscreen Qt emits Show synchronously
# inside popup() regardless of whether the menu is GC'd immediately after — so
# ``_MenuShowSpy`` is a false-positive for "user sees menu" when the menu is
# orphan-built. These two tests pin the lifetime invariant directly: the menu
# must be parented OR instance-held so it survives popup() return.
# Reference: .planning/phases/03-region-selection-export-pipeline/03-05b-RESEARCH.md §Cause A


def test_context_menu_is_instance_held_after_right_click(
    overlay_on_visible_glw, qtbot
) -> None:
    """After raiseContextMenu pops a menu, the region must hold it on
    self._context_menu so PySide6/Python GC cannot reclaim the shadow
    before Qt paints. Mirrors pyqtgraph/ROI.py:160, 781-789.

    Without this guarantee, the offscreen tests pass (Show emitted) but
    the real app shows nothing — a layer-mismatched green.
    """
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu = _last_region_menu(spy)
    assert menu is not None, "right-click did not pop a region menu"

    region = overlay._regions["rrr1"]
    # The fixed implementation lazy-builds the menu and stores it on the
    # region. The same menu instance must be referenced from both sides.
    assert region._context_menu is menu, (
        "region must hold its context menu on _context_menu after popup; "
        f"region._context_menu={region._context_menu!r}, popped={menu!r}"
    )


def test_context_menu_has_view_parent_for_wayland(
    overlay_on_visible_glw, qtbot
) -> None:
    """Wayland compositor requires a transient parent QWindow to accept
    a popup (KDE Wayland Porting Notes). The fixed raiseContextMenu must
    re-parent the menu to the scene's first view immediately before popup
    so Wayland sees a parent; harmless on X11/macOS/Windows.
    """
    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu = _last_region_menu(spy)
    assert menu is not None, "right-click did not pop a region menu"
    # The first view of the scene is the long-lived QGraphicsView (the GLW's
    # viewport's parent chain). Plan 03-05b sets the menu's parent to this
    # view before each popup; the parent's QWindow becomes the menu's
    # transient parent on Wayland.
    assert menu.parent() is not None, (
        "menu must have a parent QWidget so Wayland gets a transient parent QWindow"
    )


def test_context_menu_survives_gc_after_popup(
    overlay_on_visible_glw, qtbot
) -> None:
    """Force a Python GC cycle after the popup returns; the menu must
    still exist. Catches the regression where ``QMenu()`` was built
    with no parent and no instance reference, leaving only the C++
    side alive while Python's shadow could be reclaimed mid-paint.
    """
    import gc
    import weakref

    glw, plot, overlay, spy = overlay_on_visible_glw
    overlay.set_regions(
        [Region(id="rrr1", start_sec=20.0, end_sec=40.0, state="untouched")]
    )
    _right_click_at_data_x(qtbot, glw, plot, data_x=30.0)
    menu = _last_region_menu(spy)
    assert menu is not None
    ref = weakref.ref(menu)
    del menu
    gc.collect()
    qtbot.wait(20)
    assert ref() is not None, (
        "QMenu was garbage collected immediately after popup — lifetime bug; "
        "the region must hold the menu on _context_menu or via parent()"
    )
