"""Phase 7 Plan 07-10e Task 8 (RED) — KeeperRow Play button.

User request: "also add a play button before time code, switchin to pause
when playing".

Contract pinned by this file:
    * KeeperRow has a ``_play`` QPushButton sized 24×24 with a play icon.
    * The button sits BEFORE the time label in the layout (visual position
      0; time_label is position 1).
    * Clicking the play button emits ``play_requested(region_id)`` — the
      same signal the existing double-click path uses, so MainWindow
      routing already exists.
    * ``KeeperRow.set_active_mode(mode)`` highlights exactly one of the
      three play buttons (start / middle / end) via a QSS accent and
      updates the public ``active_mode`` attribute (quick-260622-tit).
      There is no pause glyph — clicking always fires the action; the
      highlight (not an icon swap) conveys which button is active.
"""

from __future__ import annotations

from PySide6.QtCore import Qt

from marmelade.audio.sidecar_cache import Region
from marmelade.ui.keepers_sidebar import KeeperRow, KeepersSidebar


def _make_row(qtbot, region_id: str = "rowid_a1b2c3d4e5f60011") -> KeeperRow:
    row = KeeperRow(
        region_id=region_id,
        start_sec=10.0,
        end_sec=20.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    return row


def test_play_button_present_with_play_icon(qtbot, qapp) -> None:
    """Each KeeperRow has a Play button with a non-null icon and 24×24 size."""
    row = _make_row(qtbot)
    assert hasattr(row, "_play"), "KeeperRow must expose a _play button"
    assert row._play is not None
    icon = row._play.icon()
    assert not icon.isNull(), "Play button must have a non-null icon"
    assert row._play.size().width() == 24
    assert row._play.size().height() == 24


def test_play_button_positioned_before_time_label(qtbot, qapp) -> None:
    """Play button must appear before _time_label in the row's layout."""
    row = _make_row(qtbot)
    layout = row.layout()
    assert layout is not None
    play_index = None
    time_index = None
    for i in range(layout.count()):
        item = layout.itemAt(i)
        w = item.widget() if item is not None else None
        if w is row._play:
            play_index = i
        elif w is row._time_label:
            time_index = i
    assert play_index is not None, "_play not found in layout"
    assert time_index is not None, "_time_label not found in layout"
    assert play_index < time_index, (
        f"Play button must be before time label. Got play@{play_index} "
        f"time@{time_index}."
    )


def test_play_button_click_emits_play_requested(qtbot, qapp) -> None:
    """Single click on the play button emits play_requested(region_id)."""
    region_id = "rowid_a1b2c3d4e5f60011"
    row = _make_row(qtbot, region_id=region_id)
    payloads: list[str] = []
    row.play_requested.connect(lambda rid: payloads.append(rid))
    qtbot.mouseClick(row._play, Qt.MouseButton.LeftButton)
    assert payloads == [region_id], (
        f"Expected play_requested.emit({region_id!r}); got {payloads!r}."
    )


def test_set_active_mode_highlights_only_matching_button(qtbot, qapp) -> None:
    """quick-260622-tit — set_active_mode highlights exactly one button.

    Replaces the old play↔pause glyph swap. Active state is conveyed
    purely by a QSS highlight on the just-clicked button; the other two
    are un-highlighted (empty styleSheet). The start/Play button icon
    NEVER swaps to a pause glyph.
    """
    row = _make_row(qtbot)
    # Default: nothing active, no highlights.
    assert hasattr(row, "active_mode"), (
        "KeeperRow must expose an active_mode attribute"
    )
    assert row.active_mode is None
    assert row._play.styleSheet() == ""
    assert row._play_middle.styleSheet() == ""
    assert row._play_end.styleSheet() == ""

    # The start button must always show the mirror icon (never a pause glyph).
    play_icon_before = row._play.icon().pixmap(24, 24).toImage()

    row.set_active_mode("start")
    assert row.active_mode == "start"
    assert row._play.styleSheet() != ""
    assert row._play_middle.styleSheet() == ""
    assert row._play_end.styleSheet() == ""
    # Icon unchanged — only the QSS highlight conveys active state.
    assert row._play.icon().pixmap(24, 24).toImage() == play_icon_before, (
        "The Play button icon must NOT swap when set_active_mode('start') "
        "— active state is the QSS highlight only."
    )


def _layout_index(row, widget) -> int | None:
    layout = row.layout()
    for i in range(layout.count()):
        item = layout.itemAt(i)
        w = item.widget() if item is not None else None
        if w is widget:
            return i
    return None


def test_play_middle_button_present(qtbot, qapp) -> None:
    """quick-260622-sr8 — KeeperRow exposes a _play_middle button (24x24)."""
    row = _make_row(qtbot)
    assert hasattr(row, "_play_middle")
    assert not row._play_middle.icon().isNull()
    assert row._play_middle.size().width() == 24
    assert row._play_middle.size().height() == 24


def test_play_end_button_present(qtbot, qapp) -> None:
    """quick-260622-sr8 — KeeperRow exposes a _play_end button (24x24)."""
    row = _make_row(qtbot)
    assert hasattr(row, "_play_end")
    assert not row._play_end.icon().isNull()
    assert row._play_end.size().width() == 24
    assert row._play_end.size().height() == 24


def test_new_buttons_positioned_after_play_before_time_label(qtbot, qapp) -> None:
    """Order must be: play, middle, end, time_label."""
    row = _make_row(qtbot)
    play_i = _layout_index(row, row._play)
    middle_i = _layout_index(row, row._play_middle)
    end_i = _layout_index(row, row._play_end)
    time_i = _layout_index(row, row._time_label)
    assert None not in (play_i, middle_i, end_i, time_i)
    assert play_i < middle_i < end_i < time_i, (
        f"Expected play<middle<end<time; got play@{play_i} middle@{middle_i} "
        f"end@{end_i} time@{time_i}."
    )


def test_play_middle_click_emits_play_middle_requested(qtbot, qapp) -> None:
    region_id = "rowid_a1b2c3d4e5f60011"
    row = _make_row(qtbot, region_id=region_id)
    payloads: list[str] = []
    row.play_middle_requested.connect(lambda rid: payloads.append(rid))
    qtbot.mouseClick(row._play_middle, Qt.MouseButton.LeftButton)
    assert payloads == [region_id]


def test_play_end_click_emits_play_end_requested(qtbot, qapp) -> None:
    region_id = "rowid_a1b2c3d4e5f60011"
    row = _make_row(qtbot, region_id=region_id)
    payloads: list[str] = []
    row.play_end_requested.connect(lambda rid: payloads.append(rid))
    qtbot.mouseClick(row._play_end, Qt.MouseButton.LeftButton)
    assert payloads == [region_id]


def test_sidebar_forwards_middle_and_end_signals(qtbot, qapp) -> None:
    """KeepersSidebar aggregates row middle/end signals (signal-to-signal)."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    region = Region(
        id="id" + "0" * 30 + "11",
        start_sec=10.0,
        end_sec=40.0,
        state="keeper",
    )
    row = sidebar.add_row(region)
    mid: list[str] = []
    end: list[str] = []
    sidebar.play_middle_requested.connect(lambda rid: mid.append(rid))
    sidebar.play_end_requested.connect(lambda rid: end.append(rid))
    qtbot.mouseClick(row._play_middle, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(row._play_end, Qt.MouseButton.LeftButton)
    assert mid == [region.id]
    assert end == [region.id]


def test_set_active_mode_switches_and_clears(qtbot, qapp) -> None:
    """quick-260622-tit — switching modes moves the highlight; None clears all.

    set_active_mode("end") highlights only _play_end; set_active_mode(None)
    clears all three styleSheets.
    """
    row = _make_row(qtbot)

    row.set_active_mode("end")
    assert row.active_mode == "end"
    assert row._play.styleSheet() == ""
    assert row._play_middle.styleSheet() == ""
    assert row._play_end.styleSheet() != ""

    # Switching to middle moves the highlight off end.
    row.set_active_mode("middle")
    assert row.active_mode == "middle"
    assert row._play.styleSheet() == ""
    assert row._play_middle.styleSheet() != ""
    assert row._play_end.styleSheet() == ""

    # None clears every highlight.
    row.set_active_mode(None)
    assert row.active_mode is None
    assert row._play.styleSheet() == ""
    assert row._play_middle.styleSheet() == ""
    assert row._play_end.styleSheet() == ""
