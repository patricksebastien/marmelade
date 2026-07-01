"""Gap-closure (UAT Test 7 / Plan 03-07) — Edit menu Export submenu pins.

Independent fallback entry point for Export so SC-4 survives a right-click
regression. The Edit menu gains a new "Export hovered region as ▸" submenu
with two child QActions ("MP3…" and "WAV…"). The submenu's enable/disable
state mirrors the existing K/T/U/Delete hover-targeting state machine
(:meth:`MainWindow._on_overlay_hover_changed`) but with the extra
keeper-only-export gate from CONTEXT D-A4-4 LOCKED dual-format.

Pins:

1. ``test_edit_export_submenu_exists_with_two_actions`` — submenu titled
   "Export hovered region as" exists in the Edit menu with two child
   QActions in the order MP3…, WAV….
2. ``test_edit_export_submenu_disabled_when_no_hover`` — default state.
3. ``test_edit_export_submenu_disabled_on_hover_of_non_keeper`` — D-A4-4
   keeper-only-export rule (untouched + trash both disabled).
4. ``test_edit_export_submenu_enabled_on_hover_of_keeper`` — enabled on
   hover-enter of a Keeper, disabled on hover-leave.
5. ``test_edit_export_submenu_mp3_action_triggers_on_export_region_requested``
6. ``test_edit_export_submenu_wav_action_triggers_on_export_region_requested``
7. ``test_edit_export_submenu_state_changes_when_region_state_mutates`` —
   the enable/disable computation must consult the CURRENT region state
   AND react to state mutations on the hovered region
   (regions_changed subscription, not just hover_changed).

Test contract (B-4 — production refresh path): regions are injected into
BOTH the overlay and the Keepers dock via
``window._regions_overlay.set_regions(...)`` followed by
``window._on_regions_changed(window._regions_overlay.get_regions())``,
because ``set_regions`` does NOT emit ``regions_changed`` (it is a
load-time call, not a user mutation — see regions_overlay.py:438-439).
This mimics the same refresh path that the production ``_open_file``
flow uses.

Test contract (W-8 — clear QLineEdit focus before triggering Export):
``_on_export_hovered_region`` bails if ``QApplication.focusWidget()`` is
a QLineEdit (defense-in-depth so the user typing a note can't
accidentally fire Export). The Keepers dock note QLineEdit can hold
focus from fixture setup under offscreen Qt — tests 5/6/7 call
``_clear_lineedit_focus(window)`` to drop that focus before triggering.

All tests run under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu

from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


# =========================================================================
# Module-level helpers
# =========================================================================
def _find_export_submenu(window: MainWindow) -> QMenu | None:
    """Return the "Export hovered region as" submenu via the MainWindow attribute.

    The submenu is exposed as ``MainWindow._menu_export_hovered`` after
    Task 2 GREEN lands. Reading the attribute directly (rather than
    walking ``menuBar().actions() -> sub_action.menu()``) avoids a
    libshiboken lifetime issue where the C++ QMenu reference returned by
    ``sub_action.menu()`` can be reaped before the Python wrapper is used
    in offscreen-Qt + qtbot fixture combinations.

    Returns None if the attribute is missing (the desired RED failure
    shape for Test 1 — submenu does not exist before Task 2 GREEN).
    """
    return getattr(window, "_menu_export_hovered", None)


def _find_action_in_menu(menu: QMenu, text: str) -> QAction | None:
    """Return the first non-separator QAction whose text matches ``text``."""
    for act in menu.actions():
        if act.isSeparator():
            continue
        if act.text() == text:
            return act
    return None


def _clear_lineedit_focus(window: MainWindow) -> None:
    """W-8: drop any QLineEdit focus before triggering Edit-menu Export actions.

    :meth:`MainWindow._on_export_hovered_region` bails if
    ``QApplication.focusWidget()`` is a QLineEdit (defense-in-depth —
    mirrors :meth:`MainWindow._mark_hovered_region` Pitfall #5 gate). In
    offscreen-Qt, the Keepers dock note QLineEdit can hold focus from
    fixture setup, silently causing Export-trigger tests to skip the
    emit branch.
    """
    QApplication.setActiveWindow(window)
    window.setFocus()
    QApplication.processEvents()


def _inject_regions(window: MainWindow, regions: list[Region]) -> None:
    """B-4: populate BOTH the overlay AND the Keepers dock via the production refresh path.

    :meth:`RegionsOverlay.set_regions` does NOT emit ``regions_changed``
    (regions_overlay.py:438-439 — "this is a load-time call from
    MainWindow._open_file, not a user mutation"). MainWindow's
    ``_on_regions_changed`` slot is what drives ``KeepersSidebar.add_row``.
    We mimic the production refresh path by calling both:

    1. ``set_regions(...)`` populates the overlay.
    2. ``_on_regions_changed(...)`` explicitly refreshes the Keepers dock.
    """
    window._regions_overlay.set_regions(regions)
    window._on_regions_changed()


# =========================================================================
# Fixture: a primed MainWindow with three regions (keeper + untouched + trash)
# =========================================================================
@pytest.fixture
def primed_window(qtbot, qapp, tmp_cache_dir):
    """MainWindow + 3 regions (one keeper + one untouched + one trash) injected.

    The window is NOT given an audio file — Export-pipeline downstream
    bookkeeping (``_current_playback_path`` etc.) is not exercised by
    these tests. The Edit-menu submenu state machine + trigger wiring
    are the only contracts under test.
    """
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    keeper = Region(
        id="keep",
        start_sec=10.0,
        end_sec=15.0,
        state="keeper",
        created_at="2026-05-16T00:00:00",
        note="",
    )
    untouched = Region(
        id="unt",
        start_sec=20.0,
        end_sec=25.0,
        state="untouched",
        created_at="2026-05-16T00:00:00",
        note="",
    )
    trash = Region(
        id="tr",
        start_sec=30.0,
        end_sec=35.0,
        state="trash",
        created_at="2026-05-16T00:00:00",
        note="",
    )
    _inject_regions(window, [keeper, untouched, trash])
    return window


# =========================================================================
# Pin 1 — submenu exists with two child actions
# =========================================================================
def test_edit_export_submenu_exists_with_two_actions(primed_window) -> None:
    """Edit menu has an "Export hovered region as" submenu with MP3 + WAV children."""
    submenu = _find_export_submenu(primed_window)
    assert submenu is not None, "Edit menu has no Export submenu"
    assert submenu.title() == "Export hovered region as", (
        f"Submenu title mismatch: {submenu.title()!r}"
    )

    # Verify the submenu is actually attached to the Edit menu (the
    # attribute alone does not prove it was wired in via addMenu).
    edit_menu = None
    for menu_action in primed_window.menuBar().actions():
        if menu_action.text() == "Edit":
            edit_menu = menu_action.menu()
            break
    assert edit_menu is not None, "Edit menu does not exist"
    # The submenu's menuAction() must appear in the Edit menu's actions
    # list — this proves submenu was wired in via edit_menu.addMenu(...).
    submenu_action = submenu.menuAction()
    edit_actions = list(edit_menu.actions())
    assert submenu_action in edit_actions, (
        "Export submenu is not attached to the Edit menu"
    )

    # Walk the submenu's actions and filter out separators.
    actions = [a for a in submenu.actions() if not a.isSeparator()]
    assert len(actions) == 2, (
        f"Expected exactly 2 child actions in submenu; got {len(actions)}: "
        f"{[a.text() for a in actions]}"
    )
    assert actions[0].text() == "MP3…", f"First action should be 'MP3…', got: {actions[0].text()!r}"
    assert actions[1].text() == "WAV…", f"Second action should be 'WAV…', got: {actions[1].text()!r}"


# =========================================================================
# Pin 2 — submenu disabled when no hover
# =========================================================================
def test_edit_export_submenu_disabled_when_no_hover(primed_window) -> None:
    """Default state — no region is hovered, Export is unreachable from Edit menu."""
    submenu = _find_export_submenu(primed_window)
    assert submenu is not None, "Edit menu has no Export submenu"

    mp3 = _find_action_in_menu(submenu, "MP3…")
    wav = _find_action_in_menu(submenu, "WAV…")
    assert mp3 is not None and wav is not None

    # The fixture injected regions but no hover has fired — hovered_region_id is None.
    assert primed_window._regions_overlay.hovered_region_id is None
    assert mp3.isEnabled() is False
    assert wav.isEnabled() is False


# =========================================================================
# Pin 3 — submenu disabled on hover of non-keeper (D-A4-4 keeper-only-export)
# =========================================================================
def test_edit_export_submenu_disabled_on_hover_of_non_keeper(primed_window) -> None:
    """Hovering an untouched or trash region must NOT enable Export."""
    submenu = _find_export_submenu(primed_window)
    assert submenu is not None
    mp3 = _find_action_in_menu(submenu, "MP3…")
    wav = _find_action_in_menu(submenu, "WAV…")
    assert mp3 is not None and wav is not None

    # Simulate hover-enter on the untouched region.
    primed_window._regions_overlay.hovered_region_id = "unt"
    primed_window._regions_overlay.hover_changed.emit("unt")
    assert mp3.isEnabled() is False, "Untouched region must NOT enable MP3 export"
    assert wav.isEnabled() is False, "Untouched region must NOT enable WAV export"

    # Simulate hover-enter on the trash region.
    primed_window._regions_overlay.hovered_region_id = "tr"
    primed_window._regions_overlay.hover_changed.emit("tr")
    assert mp3.isEnabled() is False, "Trash region must NOT enable MP3 export"
    assert wav.isEnabled() is False, "Trash region must NOT enable WAV export"


# =========================================================================
# Pin 4 — submenu enabled on hover of keeper (then disabled on hover-leave)
# =========================================================================
def test_edit_export_submenu_enabled_on_hover_of_keeper(primed_window) -> None:
    """Hover-enter on a keeper enables MP3 + WAV; hover-leave disables both."""
    submenu = _find_export_submenu(primed_window)
    assert submenu is not None
    mp3 = _find_action_in_menu(submenu, "MP3…")
    wav = _find_action_in_menu(submenu, "WAV…")
    assert mp3 is not None and wav is not None

    # Simulate hover-enter on the keeper region.
    primed_window._regions_overlay.hovered_region_id = "keep"
    primed_window._regions_overlay.hover_changed.emit("keep")
    assert mp3.isEnabled() is True, "Keeper hover must enable MP3 export"
    assert wav.isEnabled() is True, "Keeper hover must enable WAV export"

    # Simulate hover-leave.
    primed_window._regions_overlay.hovered_region_id = None
    primed_window._regions_overlay.hover_changed.emit(None)
    assert mp3.isEnabled() is False, "Hover-leave must disable MP3 export"
    assert wav.isEnabled() is False, "Hover-leave must disable WAV export"


# =========================================================================
# Pin 5 — MP3 action triggers _on_export_region_requested with ("keep", "mp3")
# =========================================================================
def test_edit_export_submenu_mp3_action_triggers_on_export_region_requested(
    primed_window,
) -> None:
    """MP3 submenu trigger delegates to _on_export_region_requested(rid, "mp3")."""
    _clear_lineedit_focus(primed_window)  # W-8
    submenu = _find_export_submenu(primed_window)
    assert submenu is not None
    mp3 = _find_action_in_menu(submenu, "MP3…")
    assert mp3 is not None

    # Monkey-patch the export slot — we're testing wiring, not the pipeline.
    primed_window._on_export_region_requested = MagicMock()

    # Simulate hover on the keeper region.
    primed_window._regions_overlay.hovered_region_id = "keep"
    primed_window._regions_overlay.hover_changed.emit("keep")

    mp3.trigger()
    primed_window._on_export_region_requested.assert_called_once_with("keep", "mp3")


# =========================================================================
# Pin 6 — WAV action triggers _on_export_region_requested with ("keep", "wav")
# =========================================================================
def test_edit_export_submenu_wav_action_triggers_on_export_region_requested(
    primed_window,
) -> None:
    """WAV submenu trigger delegates to _on_export_region_requested(rid, "wav")."""
    _clear_lineedit_focus(primed_window)  # W-8
    submenu = _find_export_submenu(primed_window)
    assert submenu is not None
    wav = _find_action_in_menu(submenu, "WAV…")
    assert wav is not None

    primed_window._on_export_region_requested = MagicMock()

    primed_window._regions_overlay.hovered_region_id = "keep"
    primed_window._regions_overlay.hover_changed.emit("keep")

    wav.trigger()
    primed_window._on_export_region_requested.assert_called_once_with("keep", "wav")


# =========================================================================
# Pin 7 — state machine reacts to mid-hover state mutation
# =========================================================================
def test_edit_export_submenu_state_changes_when_region_state_mutates(
    primed_window,
) -> None:
    """Mutating the hovered Keeper's state to Trash must DISABLE Export.

    Pins a subtle wiring requirement: the enable/disable computation
    must (a) consult the CURRENT region state at the moment of evaluation
    AND (b) react to ``regions_changed`` even when ``hover_changed``
    does not re-fire. Implementation must subscribe to
    ``regions_overlay.regions_changed`` in addition to ``hover_changed``.
    """
    submenu = _find_export_submenu(primed_window)
    assert submenu is not None
    mp3 = _find_action_in_menu(submenu, "MP3…")
    wav = _find_action_in_menu(submenu, "WAV…")
    assert mp3 is not None and wav is not None

    # Hover on the keeper region — both actions enabled.
    primed_window._regions_overlay.hovered_region_id = "keep"
    primed_window._regions_overlay.hover_changed.emit("keep")
    assert mp3.isEnabled() is True
    assert wav.isEnabled() is True

    # Demote the still-hovered Keeper to Trash. This fires
    # regions_changed but NOT hover_changed. The Edit-menu state
    # machine must re-evaluate and disable both actions.
    primed_window._regions_overlay.set_state("keep", "trash")
    assert mp3.isEnabled() is False, "Mid-hover demotion to Trash must DISABLE MP3 export"
    assert wav.isEnabled() is False, "Mid-hover demotion to Trash must DISABLE WAV export"
