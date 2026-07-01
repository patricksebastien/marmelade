"""Phase 8 Plan 08-05 Task 2 — drag-handle + drag/drop reorder + QSettings persistence.

Pins D-05 (drag-and-drop reorder via drag-handle BEFORE the Play
button) + D-29 (Wayland-safe icon ≥100 non-bg pixels) + Shared Pattern
4 (explicit ``QSettings("Marmelade", "Marmelade")`` org/app
pair, no bare ``QSettings()``) + RESEARCH Pattern 4 gotcha
(``setAcceptDrops(True)`` mandatory on the receiver; drag source MUST
be the row, not the handle).

The QListWidget(InternalMove) escape hatch (RESEARCH Pattern 5) lives
in Plan 08-05 Task 3 — BundleDialog's ``use_fallback_reorder=True``
flag. This test file covers the PRIMARY in-sidebar drag-handle path
ONLY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QDropEvent, QImage

from marmelade.audio.sidecar_cache import Region
from marmelade.ui.keepers_sidebar import KeeperRow, KeepersSidebar


# Region UUID fixtures.
RID_A = "0123456789abcdef0123456789abcde0"
RID_B = "0123456789abcdef0123456789abcde1"
RID_C = "0123456789abcdef0123456789abcde2"


def _make_region(rid: str, start_sec: float) -> Region:
    return Region(
        id=rid,
        start_sec=float(start_sec),
        end_sec=float(start_sec + 10.0),
        state="keeper",
        note="",
    )


def _make_row(qtbot, rid: str = RID_A) -> KeeperRow:
    row = KeeperRow(
        region_id=rid,
        start_sec=10.0,
        end_sec=20.0,
        state="keeper",
        note="",
    )
    qtbot.addWidget(row)
    return row


# ---------------------------------------------------------------------------
# Test 1 — drag-handle at layout index 0 (FIRST visible widget per D-05)
# ---------------------------------------------------------------------------


def test_drag_handle_present_at_index_0(qtbot) -> None:
    """_drag_handle is the first widget on the row, BEFORE _play (D-05)."""
    row = _make_row(qtbot)
    layout = row.layout()

    idx_handle = layout.indexOf(row._drag_handle)
    idx_play = layout.indexOf(row._play)

    assert idx_handle == 0, (
        f"drag-handle must be the first widget on the row; got index {idx_handle}"
    )
    assert idx_handle < idx_play, (
        f"drag-handle must come BEFORE play (D-05); handle={idx_handle}, play={idx_play}"
    )


# ---------------------------------------------------------------------------
# Test 2 — drag-handle icon ≥100 non-transparent pixels (Wayland safety)
# ---------------------------------------------------------------------------


def test_drag_handle_icon_renders_non_background_pixels(qtbot) -> None:
    """_drag_handle_icon() rendered to 24x24 has ≥100 non-transparent pixels.

    Regression pin per Phase 7 LEARNINGS Surprise #9 (Wayland-safe icon
    contract) and 08-PATTERNS.md Shared Pattern 1. Dotted-grip glyph
    (six 2-px-radius circles) sums to ≈ pi * 4 * 6 ≈ 75 pixels of
    nominal area but anti-aliasing inflates that comfortably above 100.
    """
    from marmelade.ui.icons import _drag_handle_icon

    icon = _drag_handle_icon()
    pix = icon.pixmap(24, 24)
    assert not pix.isNull()
    img = pix.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    count = 0
    for y in range(img.height()):
        for x in range(img.width()):
            argb = img.pixel(x, y)
            alpha = (argb >> 24) & 0xFF
            if alpha > 0:
                count += 1
    assert count >= 100, (
        f"drag-handle icon has only {count} non-transparent pixels; need ≥100 "
        "(Wayland safety — D-29 + Phase 7 LEARNINGS)"
    )


# ---------------------------------------------------------------------------
# Test 3 — sidebar acceptDrops True (RESEARCH Pattern 4 gotcha)
# ---------------------------------------------------------------------------


def test_setAcceptDrops_True_on_sidebar(qtbot) -> None:
    """KeepersSidebar.setAcceptDrops(True) is set in __init__.

    Per RESEARCH Pattern 4 lines 553-557 — without this the dropEvent
    never fires no matter what the drag source does.
    """
    sidebar = KeepersSidebar()
    qtbot.addWidget(sidebar)
    assert sidebar.acceptDrops() is True, (
        "KeepersSidebar must call setAcceptDrops(True) — RESEARCH Pattern 4 gotcha"
    )


# ---------------------------------------------------------------------------
# Test 4 — synthetic drop reorders rows + emits order_changed
# ---------------------------------------------------------------------------


def test_drag_reorder_via_synthetic_drop_emits_order_changed(qtbot) -> None:
    """Synthesise a QDropEvent payload — order changes + signal fires.

    Builds a sidebar with 3 keepers (chronological order: A, B, C).
    Constructs a QDropEvent whose mimeData carries C's region_id and
    whose position lands above the first row → expected final order
    ``[C, A, B]``.
    """
    sidebar = KeepersSidebar()
    qtbot.addWidget(sidebar)

    # add_row sorts chronologically by start_sec; pass A=10, B=20, C=30
    # so the initial visual order is [A, B, C].
    sidebar.add_row(_make_region(RID_A, 10.0))
    sidebar.add_row(_make_region(RID_B, 20.0))
    sidebar.add_row(_make_region(RID_C, 30.0))

    assert sidebar.current_order() == [RID_A, RID_B, RID_C]

    received: list[list[str]] = []
    sidebar.order_changed.connect(received.append)

    # Build a fake QDropEvent. PySide6's QDropEvent takes positional
    # args: (pos, possibleActions, data, buttons, modifiers).
    mime = QMimeData()
    mime.setText(RID_C)
    drop_pos = QPointF(10.0, 0.0)  # above the first row's center → insert at 0
    ev = QDropEvent(
        drop_pos,
        Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    sidebar.dropEvent(ev)

    assert sidebar.current_order() == [RID_C, RID_A, RID_B], (
        f"expected [C, A, B] after dropping C at top; got {sidebar.current_order()!r}"
    )
    assert received == [[RID_C, RID_A, RID_B]], (
        f"order_changed should fire once with the new order; got {received!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — QSettings persistence round-trip (Shared Pattern 4 explicit org/app)
# ---------------------------------------------------------------------------


def test_qsettings_persistence_roundtrip(qtbot, tmp_path: Path) -> None:
    """After a reorder, QSettings carries the new order; restore_order restores it.

    Uses the same QSettings("Marmelade", "Marmelade") org/app pair
    that the implementation uses (Shared Pattern 4 — bare QSettings()
    is forbidden).
    """
    # Isolate test from real QSettings by using IniFormat + a tmp file.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    settings_path = tmp_path / "settings.ini"
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )

    sidecar_path = "/tmp/test-source.wav.sidecar.json"

    # --- Build a sidebar with 3 keepers, reorder, persist.
    sidebar1 = KeepersSidebar()
    qtbot.addWidget(sidebar1)
    sidebar1.set_sidecar_path(sidecar_path)
    sidebar1.add_row(_make_region(RID_A, 10.0))
    sidebar1.add_row(_make_region(RID_B, 20.0))
    sidebar1.add_row(_make_region(RID_C, 30.0))

    # Reorder: drop C at top → [C, A, B].
    mime = QMimeData()
    mime.setText(RID_C)
    ev = QDropEvent(
        QPointF(10.0, 0.0),
        Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    sidebar1.dropEvent(ev)
    assert sidebar1.current_order() == [RID_C, RID_A, RID_B]

    # --- Build a fresh sidebar, restore — should reorder to [C, A, B].
    sidebar2 = KeepersSidebar()
    qtbot.addWidget(sidebar2)
    sidebar2.add_row(_make_region(RID_A, 10.0))
    sidebar2.add_row(_make_region(RID_B, 20.0))
    sidebar2.add_row(_make_region(RID_C, 30.0))

    assert sidebar2.current_order() == [RID_A, RID_B, RID_C]
    sidebar2.set_sidecar_path(sidecar_path)
    # set_sidecar_path SHOULD trigger restore.
    assert sidebar2.current_order() == [RID_C, RID_A, RID_B], (
        f"restore should reorder to [C, A, B]; got {sidebar2.current_order()!r}"
    )


# ---------------------------------------------------------------------------
# Test 6 — drag to self is a no-op
# ---------------------------------------------------------------------------


def test_drag_to_self_is_noop(qtbot) -> None:
    """Dropping a row at its own position leaves the order unchanged."""
    sidebar = KeepersSidebar()
    qtbot.addWidget(sidebar)
    sidebar.add_row(_make_region(RID_A, 10.0))
    sidebar.add_row(_make_region(RID_B, 20.0))
    sidebar.add_row(_make_region(RID_C, 30.0))

    received: list[list[str]] = []
    sidebar.order_changed.connect(received.append)

    # Drop A at its own y-position. The dropEvent computes the target
    # index from y; landing at A's own center should not move A.
    a_row = sidebar.find_row(RID_A)
    assert a_row is not None
    drop_y = float(a_row.geometry().center().y())

    mime = QMimeData()
    mime.setText(RID_A)
    ev = QDropEvent(
        QPointF(10.0, drop_y),
        Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    sidebar.dropEvent(ev)

    assert sidebar.current_order() == [RID_A, RID_B, RID_C], (
        "drag-to-self must not change order"
    )
    # Signal should NOT fire (or, defensively, may fire with the unchanged
    # order — either is acceptable per behavior spec). The contract pins
    # "order unchanged"; emitting no signal is the cleaner outcome.
    assert received == [] or received == [[RID_A, RID_B, RID_C]], (
        f"unexpected order_changed payload on self-drop: {received!r}"
    )


# ---------------------------------------------------------------------------
# Test 7 — explicit org/app pair (no bare QSettings)
# ---------------------------------------------------------------------------


def test_explicit_org_app_pair_no_bare_qsettings() -> None:
    """grep — no bare QSettings() instantiations in keepers_sidebar.py.

    Shared Pattern 4 — Phase 7 LEARNINGS pinned that bare QSettings()
    silently picks up the QApplication's org/app at instantiation time,
    which leaks state into the user's real settings if MainWindow forgot
    to set QApplication.setOrganizationName / setApplicationName. The
    plan + the Plan 08-04 grep gate require explicit ``QSettings(
    "Marmelade", "Marmelade")`` on every instantiation.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    src = repo_root / "src" / "marmelade" / "ui" / "keepers_sidebar.py"
    text = src.read_text()

    # Count bare QSettings() calls (LINE-LEVEL — skip docstrings/comments
    # that mention the literal as warning text). A bare-instantiation line
    # in real code has either an ``=`` (assignment) or an attribute access
    # (``.value(...)``) on the same line; a docstring line starts with a
    # backtick or quote and has no operator.
    import re

    bare_call_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        # Skip comments and docstring-ish lines (rough heuristic — a true
        # bare instantiation in code always sits on a line with either
        # ``=`` or a method call attribute).
        if stripped.startswith("#"):
            continue
        if "``QSettings()``" in line or '"QSettings()"' in line:
            continue  # documentation reference, not a call
        if re.search(r"QSettings\(\s*\)", line):
            # Real code usage — flag it.
            bare_call_lines.append(line)
    assert len(bare_call_lines) == 0, (
        f"bare QSettings() forbidden — Shared Pattern 4. Found: {bare_call_lines!r}"
    )

    # Count explicit Marmelade/Marmelade pairs — should be ≥1.
    explicit = re.findall(
        r'QSettings\(\s*"Marmelade"\s*,\s*"Marmelade"\s*\)', text
    )
    assert len(explicit) >= 1, (
        "must use explicit QSettings(\"Marmelade\", \"Marmelade\") — "
        "Shared Pattern 4"
    )
