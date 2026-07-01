"""Plan 03-02 Task 2 — KeepersSidebar + KeeperRow widget contract pins.

Twelve pins covering the Keepers dock widget contract:

quick-260622-upg trimmed the row: the warm-amber KEEPER/TRASH state badge
(``_badge``) and the inline "Add a note…" note input (``_note``) were
removed, the ``note_changed`` signal path was deleted end-to-end, and the
Norm toggle now sits immediately before the gear (Master) button. The note
DATA model + sidecar schema are PRESERVED (no migration); only the row's
editing UI + its live signal path are gone.

Group A — KeeperRow:
    1. Composition (time label + Norm + gear + Delete button; badge/note gone).
    2. Row stores region_id for signal routing.
    3. Time label format: f"HH:MM:SS – HH:MM:SS  (M:SS)".
    4. set_state is a documented no-op (no badge to update).
    5. set_range updates time label.
    6. Norm toggle precedes the gear in the layout.
    7. mouseDoubleClickEvent emits play_requested(region_id).
    8. Delete button emits delete_requested(region_id).

Group B — KeepersSidebar:
    9. Empty sidebar shows "No Keepers yet" heading; _stack page 0.
   10. add_row flips to page 1; keeper_count == len(_rows).
   11. Rows sorted chronologically by start_sec ascending.
   12. update_row_state to non-keeper removes the row from the sidebar.
   13. remove_row(rid) → if list empty, flip back to page 0.
   14. clear() resets to empty state.
   15. Signal forwarding: jump_requested / play_requested /
       delete_requested all aggregate from rows to sidebar.

All tests run under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui.keepers_sidebar import KeeperRow, KeepersSidebar


# =========================================================================
# Group A — KeeperRow
# =========================================================================
def test_row_has_surviving_widgets_and_no_badge_or_note(qtbot, qapp) -> None:
    row = KeeperRow(
        region_id="abc123",
        start_sec=10.0,
        end_sec=20.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    # Surviving semi-private attrs exposed for state access + tests.
    assert hasattr(row, "_time_label")
    assert hasattr(row, "_master")
    assert hasattr(row, "_delete")
    # quick-260622-upg — the state badge + note input were removed.
    assert not hasattr(row, "_badge")
    assert not hasattr(row, "_note")
    # quick-260625 — the per-row "Norm" button was removed; per-keeper
    # normalize is configured through the Master (mastering) dialog.
    assert not hasattr(row, "_normalize")


def test_row_stores_region_id(qtbot, qapp) -> None:
    row = KeeperRow(
        region_id="abc123",
        start_sec=10.0,
        end_sec=20.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    assert row._region_id == "abc123"


def test_row_time_label_format(qtbot, qapp) -> None:
    """Region from 872s to 1087s = 14:32 to 18:07, duration 3:35."""
    row = KeeperRow(
        region_id="rid",
        start_sec=872.0,
        end_sec=1087.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    text = row._time_label.text()
    assert "00:14:32" in text
    assert "00:18:07" in text
    assert "(3:35)" in text


def test_row_time_label_short_duration(qtbot, qapp) -> None:
    """Sub-minute duration uses 0:NN format."""
    row = KeeperRow(
        region_id="rid",
        start_sec=10.0,
        end_sec=22.0,  # 0:12 duration
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    text = row._time_label.text()
    assert "(0:12)" in text


def test_row_set_state_is_noop_no_badge(qtbot, qapp) -> None:
    """quick-260622-upg — set_state is a documented no-op (badge removed)."""
    row = KeeperRow(
        region_id="rid",
        start_sec=0.0,
        end_sec=1.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    # Does not raise and creates no badge attribute.
    row.set_state("trash")
    assert not hasattr(row, "_badge")


def test_row_set_playing_toggles_row_tint(qtbot, qapp) -> None:
    """quick-260625 — set_playing applies/clears the whole-row 'now playing' tint."""
    from marmelade.ui.keepers_sidebar import _PLAYING_ROW_QSS

    row = KeeperRow(
        region_id="rid",
        start_sec=0.0,
        end_sec=1.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    # The row opts into stylesheet backgrounds + carries the scoping objectName.
    assert row.objectName() == "KeeperRow"
    assert row.styleSheet() == ""

    row.set_playing(True)
    assert row.styleSheet() == _PLAYING_ROW_QSS
    assert row._is_playing_highlight is True

    row.set_playing(False)
    assert row.styleSheet() == ""
    assert row._is_playing_highlight is False


def test_row_set_range_updates_time_label(qtbot, qapp) -> None:
    row = KeeperRow(
        region_id="rid",
        start_sec=10.0,
        end_sec=22.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    row.set_range(0.0, 60.0)
    text = row._time_label.text()
    assert "00:00:00" in text
    assert "00:01:00" in text
    assert "(1:00)" in text


def test_row_delete_button_emits_delete_requested(qtbot, qapp) -> None:
    row = KeeperRow(
        region_id="rid42",
        start_sec=0.0,
        end_sec=1.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    seen: list = []
    row.delete_requested.connect(lambda rid: seen.append(rid))
    qtbot.mouseClick(row._delete, Qt.MouseButton.LeftButton)
    assert seen == ["rid42"]


def test_row_delete_button_label_is_word_not_glyph(qtbot, qapp) -> None:
    """UI-SPEC §Copywriting locks 'Delete' text, NOT '×' or icon."""
    row = KeeperRow(
        region_id="rid",
        start_sec=0.0,
        end_sec=1.0,
        state="keeper",
        note="",
    )
    qtbot.add_widget(row)
    assert row._delete.text() == "Delete"


# =========================================================================
# Group B — KeepersSidebar
# =========================================================================
def test_empty_sidebar_shows_no_keepers_yet(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    assert sidebar._stack.currentIndex() == 0
    # The empty page contains a heading widget with the locked copy.
    # Walk children to find the QLabel containing "No Keepers yet".
    from PySide6.QtWidgets import QLabel

    empty_page = sidebar._stack.widget(0)
    headings = [
        c
        for c in empty_page.findChildren(QLabel)
        if "No Keepers yet" in c.text()
    ]
    assert len(headings) >= 1


def test_add_row_flips_to_page_one_and_increments_count(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    region = Region(
        id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note=""
    )
    sidebar.add_row(region)
    assert sidebar._stack.currentIndex() == 1
    assert sidebar.keeper_count() == 1
    assert "abc" in sidebar._rows


def test_remove_row_back_to_empty_state(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    region = Region(
        id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note=""
    )
    sidebar.add_row(region)
    sidebar.remove_row("abc")
    assert sidebar._stack.currentIndex() == 0
    assert sidebar.keeper_count() == 0


def test_rows_inserted_chronologically(qtbot, qapp) -> None:
    """Add region at 30s, then region at 10s — sidebar lists them in order."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    sidebar.add_row(
        Region(id="late", start_sec=30.0, end_sec=35.0, state="keeper", note="")
    )
    sidebar.add_row(
        Region(id="early", start_sec=10.0, end_sec=15.0, state="keeper", note="")
    )
    # _row_start_sec is tracked on each row instance for sort comparisons.
    early_row = sidebar._rows["early"]
    late_row = sidebar._rows["late"]
    assert early_row._row_start_sec == 10.0
    assert late_row._row_start_sec == 30.0
    # Walking the rows layout in order gives 'early' before 'late'.
    rows_in_layout: list = []
    for i in range(sidebar._rows_layout.count()):
        w = sidebar._rows_layout.itemAt(i).widget()
        if isinstance(w, KeeperRow):
            rows_in_layout.append(w._region_id)
    assert rows_in_layout.index("early") < rows_in_layout.index("late")


def test_update_row_state_to_trash_removes_row(qtbot, qapp) -> None:
    """Keepers panel shows ONLY Keepers — switching to trash removes the row."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    sidebar.add_row(
        Region(id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note="")
    )
    assert sidebar.keeper_count() == 1
    sidebar.update_row_state("abc", "trash")
    assert sidebar.keeper_count() == 0


def test_update_row_state_keeper_to_keeper_no_op(qtbot, qapp) -> None:
    """A keeper-to-keeper transition leaves the row alone."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    sidebar.add_row(
        Region(id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note="")
    )
    sidebar.update_row_state("abc", "keeper")
    assert sidebar.keeper_count() == 1
    # quick-260622-upg — no badge to inspect; the row simply stays present.
    assert "abc" in sidebar._rows


def test_update_row_range_updates_time_label(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    sidebar.add_row(
        Region(id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note="")
    )
    sidebar.update_row_range("abc", 0.0, 60.0)
    row = sidebar._rows["abc"]
    text = row._time_label.text()
    assert "00:00:00" in text
    assert "00:01:00" in text


def test_clear_resets_state(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    for rid, start in [("a", 1.0), ("b", 5.0), ("c", 10.0)]:
        sidebar.add_row(
            Region(
                id=rid, start_sec=start, end_sec=start + 1.0, state="keeper", note=""
            )
        )
    assert sidebar.keeper_count() == 3
    sidebar.clear()
    assert sidebar.keeper_count() == 0
    assert sidebar._stack.currentIndex() == 0


def test_sidebar_forwards_jump_requested(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    sidebar.add_row(
        Region(id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note="")
    )
    seen: list = []
    sidebar.jump_requested.connect(lambda rid: seen.append(rid))
    # Emit on the row directly (no synthesized click — covered in
    # the row-level test).
    sidebar._rows["abc"].jump_requested.emit("abc")
    assert seen == ["abc"]


def test_sidebar_forwards_play_requested(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    sidebar.add_row(
        Region(id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note="")
    )
    seen: list = []
    sidebar.play_requested.connect(lambda rid: seen.append(rid))
    sidebar._rows["abc"].play_requested.emit("abc")
    assert seen == ["abc"]


def test_sidebar_forwards_delete_requested(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    sidebar.add_row(
        Region(id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note="")
    )
    seen: list = []
    sidebar.delete_requested.connect(lambda rid: seen.append(rid))
    sidebar._rows["abc"].delete_requested.emit("abc")
    assert seen == ["abc"]


def test_dock_title_callback_fires_after_add_row(qtbot, qapp) -> None:
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    seen: list = []
    sidebar.set_dock_title_callback(lambda n: seen.append(n))
    sidebar.add_row(
        Region(id="abc", start_sec=10.0, end_sec=20.0, state="keeper", note="")
    )
    sidebar.add_row(
        Region(id="def", start_sec=30.0, end_sec=40.0, state="keeper", note="")
    )
    sidebar.remove_row("abc")
    # Callback fires after every add and remove.
    assert seen == [1, 2, 1]
