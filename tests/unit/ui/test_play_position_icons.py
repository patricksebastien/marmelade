"""quick-260622-sr8 Task 2 — the three KeeperRow play-position buttons
(Play / middle / end) must carry mutually distinct icon pixmaps so a user
can tell them apart at a glance.
"""

from __future__ import annotations

from marmelade.ui.keepers_sidebar import KeeperRow


def _make_row(qtbot) -> KeeperRow:
    row = KeeperRow(
        region_id="rowid_a1b2c3d4e5f60011",
        start_sec=10.0,
        end_sec=40.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    return row


def test_play_middle_end_icons_mutually_distinct(qtbot, qapp) -> None:
    row = _make_row(qtbot)
    play = row._play.icon().pixmap(24, 24).toImage()
    middle = row._play_middle.icon().pixmap(24, 24).toImage()
    end = row._play_end.icon().pixmap(24, 24).toImage()
    assert play != middle, "Play and middle icons must differ"
    assert play != end, "Play and end icons must differ"
    assert middle != end, "Middle and end icons must differ"
