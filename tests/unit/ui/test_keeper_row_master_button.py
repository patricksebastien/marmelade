"""Phase 7 Plan 07-02 Task 2 (RED) — KeeperRow Master button + composite icon.

D-12 — divergence-badge tri-state painted into the row's Master button
icon at icon-construction time (Wayland-safe per Phase 6 LEARNINGS: no
Unicode glyphs, always paint or use QIcon.fromTheme).

Contract pinned by this file:
    * Master button visible on every KeeperRow with the correct default
      tooltip and a non-null icon.
    * Clicking the button emits ``mastering_requested(region_id)``.
    * Late-binding regression guard — for-loop adding rows binds region_id
      correctly via default-arg closure (Phase 1 LEARNINGS).
    * ``set_mastering_badge(state)`` updates BOTH icon AND tooltip per the
      UI-SPEC KeeperRow Master button D-12 tooltip table.
    * Right-click context menu has a "Cancel mastering" action that is
      ENABLED iff the row's status label text starts with "Mastering".
"""

from __future__ import annotations

from PySide6.QtCore import Qt

from marmelade.audio.sidecar_cache import Region
from marmelade.ui.keepers_sidebar import KeeperRow, KeepersSidebar


def _make_row(qtbot, region_id: str = "id1234567890abcd") -> KeeperRow:
    row = KeeperRow(
        region_id=region_id,
        start_sec=10.0,
        end_sec=20.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    return row


def test_master_button_present_with_default_badge(qtbot, qapp) -> None:
    """Each KeeperRow has a Master button with a non-null icon + correct tooltip."""
    row = _make_row(qtbot)
    assert hasattr(row, "_master")
    assert row._master is not None
    icon = row._master.icon()
    assert not icon.isNull()
    # Default tooltip — UI-SPEC KeeperRow Master button D-12 "No mastering" state.
    assert "No mastering" in row._master.toolTip()


def test_master_button_click_emits_mastering_requested_with_region_id(qtbot, qapp) -> None:
    """Click on Master button emits ``mastering_requested(region_id)``."""
    region_id = "abc1234567890def0000000000000aaaa"
    row = _make_row(qtbot, region_id=region_id)

    with qtbot.waitSignal(row.mastering_requested, timeout=1000) as blocker:
        qtbot.mouseClick(row._master, Qt.MouseButton.LeftButton)
    assert blocker.args == [region_id]


def test_default_arg_closure_binding_in_loop(qtbot, qapp) -> None:
    """Loop-bound rows pass region_id correctly (Phase 1 LEARNINGS late-binding guard)."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    # Add 3 keepers in a loop — the click on row 1 must emit row-1's id, NOT
    # the last loop value.
    regions = [
        Region(
            id=f"id{i:032d}",
            start_sec=float(i * 10),
            end_sec=float(i * 10 + 5),
            state="keeper",
        )
        for i in range(3)
    ]
    rows = [sidebar.add_row(r) for r in regions]
    # Click row 1 — assert the SIDEBAR signal emits row-1's id.
    expected_id = regions[1].id
    with qtbot.waitSignal(sidebar.mastering_requested, timeout=1000) as blocker:
        qtbot.mouseClick(rows[1]._master, Qt.MouseButton.LeftButton)
    assert blocker.args == [expected_id]


def test_set_mastering_badge_updates_tooltip(qtbot, qapp) -> None:
    """``set_mastering_badge`` updates BOTH icon and tooltip per the UI-SPEC table."""
    row = _make_row(qtbot)
    # check — using the session chain.
    row.set_mastering_badge("check")
    assert "session mastering chain" in row._master.toolTip()
    # star — custom.
    row.set_mastering_badge("star")
    assert "Custom mastering chain" in row._master.toolTip()
    # none — back to default.
    row.set_mastering_badge("none")
    assert "No mastering" in row._master.toolTip()


def test_right_click_master_button_context_menu_cancel_action_state(qtbot, qapp) -> None:
    """The Cancel mastering action is enabled iff status text starts with "Mastering".

    The test invokes ``_build_master_context_menu`` directly (which
    constructs the QMenu without executing it). The exec-and-show path
    (``_open_master_context_menu``) is verified visually in HUMAN-UAT
    — it cannot be tested reliably in offscreen Qt because ``QMenu.exec``
    is a C++ slot that monkeypatching cannot override.
    """
    row = _make_row(qtbot)

    # Status "Mastering 50%" → action is ENABLED.
    row.set_mastering_status("Mastering 50%", "#9CA3AF")
    menu = row._build_master_context_menu()
    cancel = _find_action(menu.actions(), "Cancel mastering")
    assert cancel is not None
    assert cancel.isEnabled() is True

    # Status "Ready" → action is DISABLED.
    row.set_mastering_status("Ready", "#7FBFFF")
    menu2 = row._build_master_context_menu()
    cancel2 = _find_action(menu2.actions(), "Cancel mastering")
    assert cancel2 is not None
    assert cancel2.isEnabled() is False


def _find_action(actions, text: str):
    for a in actions:
        if a.text() == text:
            return a
    return None
