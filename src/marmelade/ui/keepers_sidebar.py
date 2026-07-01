"""Keepers QDockWidget body (Plan 03-02 — UI-07).

Right-side dock listing every Keeper-state region with per-row time +
Delete button. Structural 1:1 mirror of
:mod:`marmelade.ui.layers_sidebar` shape: an aggregating
:class:`KeepersSidebar` widget owning a :class:`KeeperRow` per region,
with class-level signals forwarded from rows to sidebar.

The sidebar contract — typed signals out, public methods in:

    KeepersSidebar.jump_requested = Signal(str)
        Forwarded from each row's left-click on the row body.
        MainWindow connects this to ``PlaybackEngine.seek``.
    KeepersSidebar.play_requested = Signal(str)
        Forwarded from each row's double-click. MainWindow connects
        this to ``PlaybackEngine.play(path, start_seconds=...)``.
    KeepersSidebar.delete_requested = Signal(str)
        Forwarded from each row's Delete button click. MainWindow
        connects this to ``RegionsOverlay.delete(rid)``.

    KeepersSidebar.add_row(region) -> KeeperRow
        Insert in chronological order (sorted by start_sec asc).
        Flips _stack to page 1 (the populated state).
    KeepersSidebar.remove_row(region_id)
        Remove a row. Flips _stack back to page 0 when empty.
    KeepersSidebar.clear()
        Drop every row + reset to empty state.
    KeepersSidebar.update_row_state(region_id, state)
        If state != "keeper", REMOVE the row from the panel
        (Keepers-only invariant per UI-SPEC §Layout Architecture).
        Otherwise a no-op (the visible state badge was removed in
        quick-260622-upg — rows track no visible state anymore).
    KeepersSidebar.update_row_range(region_id, start_sec, end_sec)
        Update the time-block label + re-sort the row chronologically.
    KeepersSidebar.keeper_count() -> int
        Total rows currently shown.
    KeepersSidebar.set_dock_title_callback(cb)
        Register a callback invoked with the current keeper_count
        after every add/remove. MainWindow uses this to keep the
        dock title ``Keepers (N)`` live.

Closure-binding discipline (Phase 1 LEARNINGS §"Late-binding closure
capture would break the generation-token guard"): every signal connect
inside :class:`KeeperRow` uses the default-arg trick
(``lambda rid=region_id: ...``) so a for-loop adding multiple rows
does not late-bind the loop variable.

UI-SPEC tokens — locked here for source-grep gates:

* Time block: ``color: #E6E6E6; font-family: monospace; font-size: 8pt;``

The warm-amber KEEPER/TRASH state badge + the "Add a note…" inline note
input were removed in quick-260622-upg to declutter the row. The note
DATA model + sidecar schema are PRESERVED (no migration) — only the
row's editing UI + its live note-edit signal path are gone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QMimeData, QPoint, QSettings, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QDrag,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from marmelade.audio.mastering_cache import is_mastered_cache_fresh
from marmelade.audio.sidecar_cache import Region
from marmelade.ui.icons import (
    _drag_handle_icon,
    _master_icon_with_badge,
    _play_end_icon,
    _play_middle_icon,
    _play_start_icon,
    _share_icon,
)


# quick-260622-tit — QSS accent applied to the active keeper play button.
# A subtle muted highlight consistent with the dark chrome (faint blue
# border + slightly raised grey fill). Wayland-safe (no Unicode, no
# geometry change — button stays 24x24 / 20x20 icon / NoFocus). Cleared
# to "" on the two inactive buttons.
_ACTIVE_PLAY_BUTTON_QSS = (
    "background-color: #3A3D44; border: 1px solid #5A8DBE; border-radius: 4px;"
)

# quick-260625 — "now playing" whole-row tint applied to the KeeperRow whose
# region the playhead is currently inside (see MainWindow._set_playing_row_
# highlight). Scoped to the row's objectName so it does NOT bleed onto the
# row's child widgets (the bare ``QWidget { … }`` form would). The row sets
# Qt.WA_StyledBackground so a QWidget subclass actually paints the stylesheet
# background. Cleared to "" when the playhead leaves the region.
_PLAYING_ROW_QSS = "#KeeperRow { background-color: #2F3A47; }"


def _fmt_hhmmss(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` (24h, no rollover). Always 8 chars."""
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    """Format duration as ``0:NN`` for <60 s, ``M:SS`` for ≥60 s.

    Matches UI-SPEC §Copywriting Keepers row time block parenthetical.
    """
    total = max(0, int(seconds))
    if total < 60:
        return f"0:{total:02d}"
    m = total // 60
    s = total % 60
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Phase 8 Plan 08-05 Task 2 — drag-handle button + QSettings helpers
# ---------------------------------------------------------------------------


def _sidecar_namespace_key(sidecar_path: str) -> str:
    """SHA1-derived QSettings sub-key for per-sidecar bundle-order storage.

    RESEARCH §Open Question 4 — the bundle-order persistence is scoped
    per audio source so opening a different file does not inherit the
    previous file's keeper order. SHA1 of the sidecar path collapses
    path-separator/encoding variance into a stable hex namespace.
    """
    return hashlib.sha1(sidecar_path.encode("utf-8")).hexdigest()


class _DragHandleButton(QPushButton):
    """A QPushButton that starts a QDrag whose SOURCE is the parent KeeperRow.

    RESEARCH Pattern 4 gotcha (lines 533-544): the drag source MUST be
    the row, NOT the handle — Qt computes drag indicator + drop target
    from the source widget's geometry. Hooking ``QDrag(row)`` here, the
    handle just provides the mouse-input affordance.

    The drag payload is the row's ``region_id`` carried via
    :meth:`QMimeData.setText` (custom MIME would also work; text/plain
    is fine for the in-app reorder use case — Threat T-08-05-05 accepts
    that another Qt app could theoretically read the opaque UUID).
    """

    def __init__(self, owner_row: "KeeperRow", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._owner_row = owner_row

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if e.buttons() & Qt.MouseButton.LeftButton:
            drag = QDrag(self._owner_row)  # SOURCE = row, NOT handle
            mime = QMimeData()
            mime.setText(self._owner_row._region_id)
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.MoveAction)
            return
        super().mouseMoveEvent(e)


class KeeperRow(QWidget):
    """One row inside the Keepers dock's populated state.

    Composition (single QHBoxLayout, left-to-right):
        [time QLabel] [Norm] [gear] [share] [stretch] [Delete]

    Signals:
        jump_requested(region_id: str): emitted on a left-mouse press
            that lands on the row body (not on the QLineEdit / Delete
            button — those consume their own clicks). MainWindow connects
            to ``PlaybackEngine.seek``.
        play_requested(region_id: str): emitted on a left double-click —
            MainWindow connects to ``PlaybackEngine.play(path,
            start_seconds=region.start_sec)``.
        delete_requested(region_id: str): emitted on the Delete button click.

    Attributes:
        _region_id: stable id captured at construction (default-arg
            closure binding on every signal connect uses this).
        _row_start_sec: cached for chronological-insertion sort key in
            :meth:`KeepersSidebar.add_row` and re-sort in
            :meth:`KeepersSidebar.update_row_range`.
    """

    jump_requested = Signal(str)
    play_requested = Signal(str)
    # quick-260622-sr8 — Play-from-middle / Play-from-end per-row buttons.
    # Payload is region_id; KeepersSidebar aggregates these into its own
    # play_middle_requested / play_end_requested signals, which MainWindow
    # connects to _on_keeper_play(rid, start_mode="middle"/"end").
    play_middle_requested = Signal(str)
    play_end_requested = Signal(str)
    delete_requested = Signal(str)
    # Phase 7 Plan 07-02 Task 2 — per-row Master button. Payload is
    # region_id; opens the MasteringDialog for that keeper.
    mastering_requested = Signal(str)
    # Phase 7 Plan 07-02 Task 2 — right-click context menu "Cancel
    # mastering" action. Payload is region_id; MainWindow handler
    # cancels any in-flight MasteringRunnable for this keeper.
    cancel_mastering_requested = Signal(str)
    # Phase 8 Plan 08-04 Task 3 — per-row Share-to-YouTube button.
    # Payload is region_id. ALWAYS enabled at construction per
    # revision iter 1 B1 (R-05 + D-02 fallback contract owns the
    # mastered-cache-vs-source-proxy routing in MainWindow's slot,
    # not in this widget). KeepersSidebar.add_row connects this to
    # its own share_requested signal which MainWindow's
    # _on_share_requested slot subscribes to.
    share_requested = Signal(str)
    # quick-260620-mgu NORM-04 — per-row Normalize toggle. Payload is
    # (region_id, enabled). The per-row "Norm" button that emitted this was
    # removed (quick-260625); per-keeper normalize is now configured through
    # the Master (mastering) dialog. The signal + MainWindow's
    # _on_keeper_normalize_changed slot are retained (dormant) so the
    # persistence + waveform-rerender path stays available.
    normalize_changed = Signal(str, bool)

    def __init__(
        self,
        region_id: str,
        start_sec: float,
        end_sec: float,
        # quick-260622-upg — `state` + `note` are accepted-but-unused. The
        # state badge + note QLineEdit were removed from the row, but these
        # params are retained for call-site / add_row + existing-test
        # compatibility (no signature change). The note DATA model + sidecar
        # schema are PRESERVED elsewhere; only the row's editing UI is gone.
        state: str,
        note: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._region_id = region_id
        self._row_start_sec = float(start_sec)
        # quick-260625 — enable the "now playing" whole-row tint. The
        # objectName scopes _PLAYING_ROW_QSS to this row; WA_StyledBackground
        # makes a QWidget subclass honor a stylesheet background-color.
        self.setObjectName("KeeperRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Tracks the current row-tint state so set_playing can no-op when
        # unchanged — avoids a style recompute on every 30 Hz playhead tick.
        self._is_playing_highlight: bool = False

        layout = QHBoxLayout(self)
        # UI-SPEC §Spacing Scale — row internal padding 8/8/16/8 T/R/B/L
        # (PySide6 QHBoxLayout setContentsMargins order is L,T,R,B).
        layout.setContentsMargins(16, 8, 8, 8)
        layout.setSpacing(8)

        # ---- Phase 8 Plan 08-05 Task 2 — drag-handle (BEFORE Play, D-05) ----
        # Drag-and-drop reorder affordance for the bundle Share path.
        # The handle itself is a thin QPushButton subclass that starts
        # a QDrag whose SOURCE is THIS KeeperRow (not the handle) —
        # RESEARCH Pattern 4 gotcha (lines 533-544) requires this so the
        # drop event resolves to the row's geometry rather than the
        # handle's. The cursor changes to an open-hand on hover so the
        # affordance is discoverable without a tooltip read.
        #
        # Icon: QPainter-drawn dotted-grip (≥100 non-bg pixels regression-
        # pinned per Phase 7 LEARNINGS Surprise #9 + D-29 Wayland-safe
        # icon discipline). The icon lives in ``marmelade.ui.icons``
        # alongside the share + master glyphs so all per-row icons share
        # one Wayland-safety regime.
        self._drag_handle = _DragHandleButton(self)
        self._drag_handle.setFixedSize(24, 24)
        self._drag_handle.setIconSize(QSize(20, 20))
        self._drag_handle.setIcon(_drag_handle_icon())
        self._drag_handle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_handle.setAccessibleName(
            "Drag handle — reorder this keeper for bundle export"
        )
        self._drag_handle.setToolTip(
            "Drag to reorder keepers for bundle export"
        )
        # Flat styling so the handle reads as an affordance rather than
        # a clickable button (the visual cue is the grip glyph + cursor,
        # not a 3-D button border).
        self._drag_handle.setFlat(True)
        layout.addWidget(self._drag_handle, 0)

        # ---- Phase 7 Plan 07-10e — per-row Play / Pause button ----
        # User request: play button before the time code, switching to a
        # play icon that mirrors the End-button glyph (quick-260622-tit):
        # a right-pointing triangle rolling out of a left start-bar (|▶).
        # No pause behavior — clicking always fires the action; which
        # button is active is conveyed by a QSS highlight via
        # ``set_active_mode`` (no icon swap). Wayland-safe painted glyph
        # (Phase 6 LEARNINGS: no Unicode glyphs).
        #
        # ``play_requested`` is the same signal the existing double-click
        # path already emits — MainWindow's _on_keeper_play handler is
        # the single sink for both gestures. ``active_mode`` reflects
        # which of the three buttons MainWindow has flagged active for
        # this keeper (None when nothing is playing this keeper).
        self.active_mode: Optional[str] = None
        self._play = QPushButton()
        self._play.setFixedSize(24, 24)
        self._play.setIconSize(QSize(20, 20))
        self._play.setIcon(_play_start_icon())
        self._play.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play.setAccessibleName(
            f"Play keeper {_fmt_hhmmss(start_sec)} to {_fmt_hhmmss(end_sec)}"
        )
        self._play.setToolTip(
            "Play this keeper from the start "
            "(mastered if Ready, source otherwise)."
        )
        self._play.clicked.connect(
            lambda _checked=False, rid=region_id: self.play_requested.emit(rid)
        )
        layout.addWidget(self._play, 0)

        # ---- Play-from-middle / Play-from-end buttons (quick-260622-sr8) ----
        # Mental model: Play=start / middle=middle / end=ending. Both mirror
        # the _play button setup verbatim (24x24, 20x20 iconSize, NoFocus) and
        # route through the existing _on_keeper_play path via mode-injecting
        # sidebar signals — no new audio code path.
        self._play_middle = QPushButton()
        self._play_middle.setFixedSize(24, 24)
        self._play_middle.setIconSize(QSize(20, 20))
        self._play_middle.setIcon(_play_middle_icon())
        self._play_middle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_middle.setAccessibleName(
            f"Play keeper {_fmt_hhmmss(start_sec)} to {_fmt_hhmmss(end_sec)} "
            "from the middle"
        )
        self._play_middle.setToolTip(
            "Play from the middle of this keeper (no fade-in)"
        )
        self._play_middle.clicked.connect(
            lambda _checked=False, rid=region_id: self.play_middle_requested.emit(
                rid
            )
        )
        layout.addWidget(self._play_middle, 0)

        self._play_end = QPushButton()
        self._play_end.setFixedSize(24, 24)
        self._play_end.setIconSize(QSize(20, 20))
        self._play_end.setIcon(_play_end_icon())
        self._play_end.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_end.setAccessibleName(
            f"Play last 5 seconds of keeper {_fmt_hhmmss(start_sec)} to "
            f"{_fmt_hhmmss(end_sec)}"
        )
        self._play_end.setToolTip(
            "Play the ending — starts 5 s before the keeper end"
        )
        self._play_end.clicked.connect(
            lambda _checked=False, rid=region_id: self.play_end_requested.emit(
                rid
            )
        )
        layout.addWidget(self._play_end, 0)

        # ---- Time block ----
        time_text = (
            f"{_fmt_hhmmss(start_sec)} – {_fmt_hhmmss(end_sec)}  "
            f"({_fmt_duration(end_sec - start_sec)})"
        )
        self._time_label = QLabel(time_text)
        self._time_label.setStyleSheet(
            "color: #E6E6E6; font-family: monospace; font-size: 8pt;"
        )
        layout.addWidget(self._time_label, 0)

        # ---- Phase 7 Plan 07-02 Task 2 — Master button + status label ----
        # The Master button opens a per-keeper MasteringDialog (Task 3
        # wires this). Composite icon: gear + divergence badge in the
        # bottom-right corner. Wayland-safe — no Unicode glyphs (Phase 6
        # LEARNINGS — QFontMetrics.inFont lies at small sizes on Ubuntu).
        #
        # Default-arg closure binding (Phase 1 LEARNINGS) — capture
        # ``region_id`` by value at connect time so a for-loop adding
        # multiple rows in MainWindow does NOT late-bind to the last id.
        start_label = _fmt_hhmmss(start_sec)
        end_label = _fmt_hhmmss(end_sec)
        self._master = QPushButton()
        self._master.setFixedSize(24, 24)
        self._master.setIconSize(QSize(20, 20))
        self._master.setIcon(_master_icon_with_badge("none"))
        self._master.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Phase 7 Plan 07-03 — track the badge state so tests + the
        # session-chain-changed slot can inspect it without re-running
        # the config_hash comparison.
        self._badge_state: str = "none"
        self._master.setAccessibleName(
            f"Mastering for keeper {start_label} to {end_label}"
        )
        # UI-SPEC KeeperRow Master button D-12 — tooltip per badge state.
        self._master.setToolTip(
            "No mastering — exports use the source audio. Click to configure."
        )
        self._master.clicked.connect(
            lambda _checked=False, rid=region_id: self.mastering_requested.emit(rid)
        )
        # Right-click context menu: "Cancel mastering" action.
        self._master.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._master.customContextMenuRequested.connect(
            lambda pt, rid=region_id: self._open_master_context_menu(pt, rid)
        )
        layout.addWidget(self._master, 0)

        # ---- Phase 8 Plan 08-04 Task 3 — per-row Share-to-YouTube button ----
        # D-18 + D-29 — 24x24 QPushButton AFTER the Master gear,
        # BEFORE the master_status label. Wayland-safe QPainter-drawn
        # right-arrow icon (no Unicode glyph; ≥100 non-bg pixels pinned
        # by tests).
        #
        # ALWAYS enabled at construction per revision iter 1 B1: the
        # R-05 + D-02 fallback contract owns the mastered-cache-vs-
        # source-proxy routing in MainWindow's _on_share_requested
        # slot, so this widget does not need a per-row enable gate. The
        # tooltip is a single string (no enabled/disabled split).
        #
        # Default-arg closure binding (Phase 1 LEARNINGS) — capture
        # region_id by value at connect time so a for-loop adding
        # multiple rows in MainWindow does NOT late-bind to the last id.
        self._share = QPushButton()
        self._share.setFixedSize(24, 24)
        self._share.setIconSize(QSize(20, 20))
        self._share.setIcon(_share_icon())
        self._share.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._share.setAccessibleName(
            f"Share keeper {start_label} to {end_label} to YouTube"
        )
        self._share.setToolTip(
            "Share to YouTube — uploads as MP4 with abstract image"
        )
        self._share.clicked.connect(
            lambda _checked=False, rid=region_id: self.share_requested.emit(rid)
        )
        layout.addWidget(self._share, 0)

        # Master-status QLabel — Caption 8 pt semibold per UI-SPEC
        # Per-KeeperRow Master All status badge. Empty initial text.
        self._master_status = QLabel("")
        self._master_status.setStyleSheet(
            "color: #9CA3AF; font-size: 8pt; font-weight: 600;"
        )
        layout.addWidget(self._master_status, 0)

        # ---- Stretch (quick-260622-upg) ----
        # The "Add a note…" QLineEdit used to grow here with stretch factor 1,
        # keeping the right-hand controls right-aligned. The note input was
        # removed; this stretch preserves that right-alignment so the Delete
        # button stays flush-right and the left controls stay flush-left.
        #
        # Per-keeper MP3/WAV export buttons (Plan 03-07) were removed
        # (quick-260625): per-keeper export now lives only in the right-click
        # context menu + Edit menu; clip export is driven by the
        # "Export All Keepers" batch button at the dock bottom.
        layout.addStretch(1)

        # ---- Delete button ----
        # UI-SPEC §Copywriting — LITERAL "Delete" (not "×", not icon).
        self._delete = QPushButton("Delete")
        self._delete.clicked.connect(
            lambda _checked=False, rid=region_id: self.delete_requested.emit(rid)
        )
        layout.addWidget(self._delete, 0)

    # ----------------------------------------------------------- public API
    def set_state(self, state: str) -> None:
        """No-op (quick-260622-upg — the state badge UI was removed).

        Retained as a stable public method so its only caller,
        :meth:`KeepersSidebar.update_row_state`, can keep calling it
        unchanged. The visible KEEPER/TRASH badge no longer exists, so a
        row tracks no visible state; non-keeper transitions still REMOVE
        the row from the panel (handled in ``update_row_state`` before
        this method is reached).
        """
        return None

    def set_range(self, start_sec: float, end_sec: float) -> None:
        """Refresh the time-block label after an edge-resize."""
        self._row_start_sec = float(start_sec)
        text = (
            f"{_fmt_hhmmss(start_sec)} – {_fmt_hhmmss(end_sec)}  "
            f"({_fmt_duration(end_sec - start_sec)})"
        )
        self._time_label.setText(text)

    # ----------------------------------- Play button highlight API
    def set_active_mode(self, mode: Optional[str]) -> None:
        """Highlight exactly one play button for this keeper (quick-260622-tit).

        Replaces the old play↔pause glyph swap. ``mode`` is one of
        ``"start"`` / ``"middle"`` / ``"end"`` (or ``None`` / an unknown
        value to clear). The matching button gets the
        :data:`_ACTIVE_PLAY_BUTTON_QSS` accent; the other two are reset
        to an empty stylesheet. The Play button icon NEVER swaps to a
        pause glyph — there is no pause behavior; clicking always fires
        the playback action, and the highlight is the only active-state
        affordance. Button size / icon size / focus policy are unchanged.

        MainWindow calls this when the engine starts / stops playing audio
        sourced from THIS keeper (cache or source), passing the mode that
        was last clicked, and calls ``set_active_mode(None)`` on every
        other row.
        """
        self.active_mode = mode
        self._play.setStyleSheet(
            _ACTIVE_PLAY_BUTTON_QSS if mode == "start" else ""
        )
        self._play_middle.setStyleSheet(
            _ACTIVE_PLAY_BUTTON_QSS if mode == "middle" else ""
        )
        self._play_end.setStyleSheet(
            _ACTIVE_PLAY_BUTTON_QSS if mode == "end" else ""
        )

    # ----------------------------------- "Now playing" row tint API
    def set_playing(self, on: bool) -> None:
        """Tint the WHOLE row when the playhead is inside this keeper.

        quick-260625 — independent of :meth:`set_active_mode` (which accents
        the play BUTTON you clicked). MainWindow drives this from the 30 Hz
        playhead tick so the tint follows the playhead into whatever keeper
        region it is currently passing through. No-ops when the state is
        unchanged so a tick that doesn't move between keepers costs nothing.
        """
        on = bool(on)
        if on == self._is_playing_highlight:
            return
        self._is_playing_highlight = on
        self.setStyleSheet(_PLAYING_ROW_QSS if on else "")

    # --------------------------------------- Phase 7 Master button API
    def set_mastering_badge(self, state: str) -> None:
        """Update the Master button's composite icon AND its tooltip.

        Phase 7 Plan 07-02 Task 2 — D-12 divergence badge tri-state.
        UI-SPEC KeeperRow Master button D-12 tooltip table:

            ``"none"`` — "No mastering — exports use the source audio.
                Click to configure."
            ``"check"`` — "Using the session mastering chain. Click to
                view or override."
            ``"star"`` — "Custom mastering chain (differs from session
                default). Click to view or edit."
        """
        self._master.setIcon(_master_icon_with_badge(state))  # type: ignore[arg-type]
        if state == "none":
            tooltip = (
                "No mastering — exports use the source audio. "
                "Click to configure."
            )
        elif state == "check":
            tooltip = (
                "Using the session mastering chain. "
                "Click to view or override."
            )
        elif state == "star":
            tooltip = (
                "Custom mastering chain (differs from session default). "
                "Click to view or edit."
            )
        else:
            # Defensive — unknown state keeps the existing tooltip.
            return
        self._master.setToolTip(tooltip)
        # Phase 7 Plan 07-03 — record the latest state so tests + the
        # session-chain-changed slot can introspect without recomputing.
        self._badge_state = state

    def set_mastering_status(self, text: str, color: str) -> None:
        """Update the master-status QLabel text + color.

        Phase 7 Plan 07-02 Task 2 — UI-SPEC Per-KeeperRow Master All
        status badge colors. Used by MainWindow as the MasteringRunnable
        progresses (``"Mastering N%"`` / ``"Ready"`` / ``"Failed"``).
        """
        self._master_status.setText(text)
        self._master_status.setStyleSheet(
            f"color: {color}; font-size: 8pt; font-weight: 600;"
        )

    def _build_master_context_menu(self, region_id: str | None = None) -> QMenu:
        """Build (but do NOT exec) the right-click context menu for Master.

        Phase 7 Plan 07-02 Task 2 — factored out so tests can inspect
        the menu's action enabled-state without spinning a modal
        ``QMenu.exec`` (which blocks the event loop in offscreen Qt and
        cannot be reliably monkeypatched — ``QMenu.exec`` is a C++ slot).

        Args:
            region_id: Captured by the caller's lambda so we can emit
                ``cancel_mastering_requested(region_id)`` from the
                action's trigger. Defaults to ``self._region_id`` so
                direct test callers can omit it.

        Returns:
            A :class:`QMenu` with the "Cancel mastering" QAction. The
            action's enabled-state already reflects whether a
            MasteringRunnable is in flight (status text starts with
            "Mastering").
        """
        if region_id is None:
            region_id = self._region_id
        menu = QMenu(self)
        cancel = QAction("Cancel mastering", menu)
        cancel.setEnabled(self._master_status.text().startswith("Mastering"))
        cancel.triggered.connect(
            lambda _checked=False, rid=region_id: self.cancel_mastering_requested.emit(rid)
        )
        menu.addAction(cancel)
        return menu

    def _open_master_context_menu(self, pt, region_id: str | None = None) -> None:
        """Build + show the right-click context menu for the Master button.

        Phase 7 Plan 07-02 Task 2 — right-click on the Master button
        opens a QMenu with a "Cancel mastering" action. The action is
        ENABLED iff the row's status label text starts with "Mastering"
        (i.e. a MasteringRunnable is in flight).

        Args:
            pt: ``QPoint`` in widget-local coordinates of the right-click.
            region_id: Captured by the lambda that invokes this method
                so we can emit ``cancel_mastering_requested(region_id)``
                without re-deriving it. Defaults to ``self._region_id``
                so direct test callers can omit it.
        """
        menu = self._build_master_context_menu(region_id)
        menu.exec(self._master.mapToGlobal(pt))

    # ----------------------------------------------- mouse event overrides
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Emit ``jump_requested`` on left-press over the row body.

        Per the Plan 02 contract (corrected per checker W-5): child
        widgets receive mouse events FIRST. QLineEdit + QPushButton
        accept left-clicks by default and stop propagation — their
        mouse press is consumed inside the child and never bubbles to
        the parent KeeperRow. ``mousePressEvent`` therefore fires ONLY
        for clicks falling outside child widgets (e.g., the time block
        QLabel area or the row's padding). Locked by
        ``test_sidebar_forwards_jump_requested`` (row-level emit) +
        the integration-time MainWindow test that
        ``qtbot.mouseClick(row._note, ...)`` does NOT fire jump.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.jump_requested.emit(self._region_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Emit ``play_requested`` on a left double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.play_requested.emit(self._region_id)
        super().mouseDoubleClickEvent(event)


class KeepersSidebar(QWidget):
    """The body widget of the Keepers QDockWidget.

    Mirrors the LayersSidebar shape:
        * Outer ``QVBoxLayout`` with zero margins (the dock's own chrome
          provides the visual separation).
        * ``QStackedWidget`` with two pages:
            * Page 0 — empty state ("No Keepers yet" heading + helper
              copy per UI-SPEC §Empty states).
            * Page 1 — populated state (QScrollArea wrapping a vertical
              row container).

    Signals:
        jump_requested(region_id): aggregated from each row.
        play_requested(region_id): aggregated from each row.
        delete_requested(region_id): aggregated from each row.
        selection_changed(region_id): Phase 7 Plan 07-04 — emitted
            on a left-click of the row body. MainWindow uses this to
            track the currently-selected Keeper for the A/B toolbar
            widget's enable-state logic. Payload mirrors
            ``jump_requested`` (same row click serves both purposes).
    """

    jump_requested = Signal(str)
    play_requested = Signal(str)
    # quick-260622-sr8 — aggregated per-row Play-from-middle / Play-from-end
    # signals. MainWindow connects these to _on_keeper_play with the
    # start_mode injected ("middle" / "end").
    play_middle_requested = Signal(str)
    play_end_requested = Signal(str)
    delete_requested = Signal(str)
    # Phase 7 Plan 07-02 Task 2 — aggregated per-row Master button +
    # right-click "Cancel mastering" signals. MainWindow connects to
    # _on_keeper_mastering_requested + _on_keeper_mastering_cancel_requested.
    mastering_requested = Signal(str)
    cancel_mastering_requested = Signal(str)
    # Phase 7 Plan 07-04 Task 2 — aggregated selection signal for the
    # A/B preview tracker. Emitted on a left-click of any row body.
    # Payload is the clicked row's region_id. MainWindow tracks the
    # most-recently-clicked Keeper for the A/B toolbar widget's
    # enable-state logic (UI-SPEC §"A/B Preview Toolbar Toggle"
    # line 550).
    selection_changed = Signal(str)
    # Quick-260615-l4y — two persistent top-of-sidebar buttons: a Master
    # button (_batch_button) and an Export button (_export_button).
    # Neither morphs into the other's role.
    #
    # Master button (two states, set via :meth:`set_batch_state`):
    #   idle    → master_all_requested        (re-master every keeper)
    #   running → mastering_cancel_requested   (Cancel mastering)
    #
    # Export button (freshness-gated, no state machine):
    #   click   → export_all_requested         (start the batch export flow)
    #
    # MainWindow connects each to its orchestration slots (Task 2).
    master_all_requested = Signal()
    mastering_cancel_requested = Signal()
    export_all_requested = Signal()
    # Phase 8 Plan 08-04 Task 3 — aggregated per-row Share button signal.
    # Payload is region_id. MainWindow connects this to
    # _on_share_requested which opens the UploadDialog modal.
    share_requested = Signal(str)
    # quick-260620-mgu NORM-04 — aggregated per-row Normalize toggle signal.
    # Payload is (region_id, enabled). MainWindow connects this to
    # _on_keeper_normalize_changed which persists the field to the sidecar.
    normalize_changed = Signal(str, bool)
    # Phase 8 Plan 08-05 Task 2 — emitted when drag-and-drop reorders
    # rows (or when QSettings restore loads a non-default order). Payload
    # is the FULL new region_id list in current visual order. The
    # BundleDialog subscribes to this so its in-dialog "current order"
    # widget stays live; tests pin the reorder via this signal.
    order_changed = Signal(list)
    # Phase 8 Plan 08-05 Task 3 — top-of-sidebar bundle Share button
    # signal. Emitted on click; MainWindow connects this to
    # _on_bundle_share_requested which opens the BundleDialog modal.
    # The button is DISABLED when ANY keeper is unmastered (D-02 gate),
    # which is enforced by _refresh_bundle_button_enabled.
    bundle_share_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Phase 8 Plan 08-05 Task 2 — drag-and-drop reorder receiver.
        # MANDATORY per RESEARCH Pattern 4 gotcha lines 553-557 — without
        # this the dropEvent never fires no matter what the drag source
        # configures. Set BEFORE any child widget is constructed so the
        # rest of __init__ does not race against a half-initialised drop
        # surface.
        self.setAcceptDrops(True)

        # Phase 8 Plan 08-05 Task 2 — sidecar path scope for QSettings
        # bundle-order persistence. None means "no sidecar yet" — no
        # persistence happens until MainWindow calls set_sidecar_path
        # after opening a file. Per-sidecar isolation prevents bundle
        # order from leaking across files.
        self._sidecar_path: Optional[str] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # UI-SPEC §Layout Architecture — Keepers dock minimum width.
        # Plan 03-07 / W-7 — raised from 280 px to 340 px to fit the
        # per-row MP3 + WAV QToolButtons (~36 px each = 72 px new widget
        # width) without collapsing the note QLineEdit below readable
        # width. UI-SPEC §Layout Architecture documents the new min.
        self.setMinimumWidth(340)

        # Quick-260615-l4y — top-of-sidebar MASTER button. Inserted ABOVE
        # the QStackedWidget so it remains visible in BOTH the empty and
        # the populated state of the sidebar. Two states (idle/running)
        # driven by :meth:`set_batch_state`; enabled whenever ≥1 keeper
        # exists so the user can re-master at any time. Disabled in the
        # empty state.
        self._batch_button = QPushButton("Master All Keepers", self)
        self._batch_button.setMinimumHeight(32)
        self._batch_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._batch_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._batch_button.clicked.connect(self._on_batch_button_clicked)
        # Initial state: idle, no keepers → disabled with disabled tooltip.
        self._batch_state: str = "idle"
        # True while a batch master is in-flight (set by the "running"
        # state). The Export gate keeps the button disabled while this
        # is set, regardless of cache freshness.
        self._mastering_in_flight: bool = False
        outer.addWidget(self._batch_button)

        # Quick-260615-l4y — top-of-sidebar EXPORT button. A persistent,
        # freshness-gated sibling of the Master button (it never morphs
        # into Master). Disabled until every keeper has a fresh mastered
        # cache (the SAME _mastered_cache_fresh_probe gate the bundle
        # button uses) and while mastering is in-flight. Click starts the
        # batch export flow (output folder + format modal, then export).
        self._export_button = QPushButton("Export All Keepers", self)
        self._export_button.setMinimumHeight(32)
        self._export_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._export_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._export_button.setEnabled(False)
        self._export_button.setToolTip(self._EXPORT_DISABLED_NO_KEEPERS_TT)
        self._export_button.clicked.connect(self.export_all_requested.emit)
        outer.addWidget(self._export_button)

        # Phase 8 Plan 08-05 Task 3 — top-of-sidebar bundle Share button
        # (D-19 same gravitational center as Master & Export All). Click
        # emits ``bundle_share_requested``; MainWindow opens the
        # BundleDialog modal. Disabled at construction (no keepers yet);
        # _refresh_bundle_button_enabled toggles enable based on the
        # mastered-cache-fresh probe for every keeper (D-02 — bundle
        # requires every keeper to have a fresh cache).
        self._bundle_button = QPushButton(_share_icon(), "Share All Keepers", self)
        self._bundle_button.setMinimumHeight(32)
        self._bundle_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._bundle_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._bundle_button.setEnabled(False)
        self._bundle_button.setToolTip(
            "Master all keepers first — bundle requires every keeper "
            "to have a fresh cache."
        )
        self._bundle_button.clicked.connect(self.bundle_share_requested.emit)
        # MainWindow installs a callable here so the sidebar can probe
        # mastered-cache freshness for every keeper without importing
        # paths / config_hash machinery itself (keeps the audio-tier
        # imports out of the UI tier's sidebar widget). Signature:
        # ``Callable[[str], bool]`` mapping region_id → is-fresh.
        self._mastered_cache_fresh_probe: Optional[Callable[[str], bool]] = None
        outer.addWidget(self._bundle_button)

        self._stack = QStackedWidget()

        # ---- Page 0 — empty state ----
        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        empty_layout.setSpacing(8)

        heading = QLabel("No Keepers yet")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet(
            "font-size: 11pt; font-weight: 600; color: #E6E6E6;"
        )

        body = QLabel(
            "Drag on the waveform to mark a region, "
            "then press K (or right-click → Mark as Keeper)."
        )
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)
        body.setStyleSheet("font-size: 9pt; color: #9CA3AF;")

        empty_layout.addStretch(1)
        empty_layout.addWidget(heading)
        empty_layout.addWidget(body)
        empty_layout.addStretch(1)

        # ---- Page 1 — populated state (scroll area + row container) ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        rows_container = QWidget()
        self._rows_layout = QVBoxLayout(rows_container)
        # UI-SPEC §Spacing Scale — row vertical gap 4 px.
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        # Bottom stretch so rows pack to the top of the scroll area.
        self._rows_layout.addStretch(1)
        scroll.setWidget(rows_container)

        self._stack.addWidget(empty_page)  # index 0
        self._stack.addWidget(scroll)  # index 1
        self._stack.setCurrentIndex(0)
        outer.addWidget(self._stack)

        # Row registry keyed by region_id for O(1) lookup.
        self._rows: dict[str, KeeperRow] = {}
        # Dock title callback (registered by MainWindow via
        # :meth:`set_dock_title_callback`). Invoked with current count
        # after every add/remove.
        self._dock_title_callback: Optional[Callable[[int], None]] = None

        # Phase 7 Plan 07-06 Task 1 — finalize the initial batch button
        # state once the row registry exists (the helper consults it).
        self._refresh_batch_button_idle_enabled()
        # Phase 8 Plan 08-05 Task 3 — finalize bundle button state too.
        self._refresh_bundle_button_enabled()
        # Quick-260615-l4y — finalize Export button state too.
        self._refresh_export_button_enabled()

    # ---------------------- Quick-260615-l4y — Master button API
    _BATCH_IDLE_LABEL = "Master All Keepers"
    _BATCH_RUNNING_LABEL = "Cancel mastering"
    _BATCH_DISABLED_TOOLTIP = (
        "Create at least one Keeper to enable batch mastering."
    )
    _BATCH_ENABLED_TOOLTIP = (
        "Master every Keeper so they're ready to audition and export."
    )
    _BATCH_RUNNING_TOOLTIP = (
        # RESEARCH §Pitfall 5 — Matchering is uncancellable mid-call;
        # surface the worst-case latency so the user doesn't think the
        # button is broken.
        "Cancel mastering — current keeper finishes its current "
        "stage first (up to ~30 s for Matchering)."
    )

    # ---------------------- Quick-260615-l4y — Export button API
    _EXPORT_LABEL = "Export All Keepers"
    _EXPORT_DISABLED_NO_KEEPERS_TT = (
        "Create at least one Keeper to enable batch export."
    )
    _EXPORT_DISABLED_UNMASTERED_TT = (
        "Master all keepers first — export requires every keeper to "
        "have a fresh mastered cache."
    )
    _EXPORT_ENABLED_TT = (
        "Export every mastered Keeper — choose an output folder and "
        "format, then export sequentially."
    )

    # --------------- Phase 8 Plan 08-05 Task 3 — bundle button API
    _BUNDLE_DISABLED_NO_KEEPERS_TT = (
        "Create at least one Keeper to enable bundle sharing."
    )
    _BUNDLE_DISABLED_UNMASTERED_TT = (
        "Master all keepers first — bundle requires every keeper "
        "to have a fresh cache."
    )
    _BUNDLE_ENABLED_TT = (
        "Share All Keepers — concatenates every mastered keeper "
        "into a single YouTube video."
    )

    def set_mastered_cache_fresh_probe(
        self, probe: Optional[Callable[[str], bool]]
    ) -> None:
        """Install the freshness-probe callback used by the bundle gate.

        ``probe(region_id) -> bool`` returns True iff the keeper has a
        fresh mastered cache on disk. MainWindow installs this with a
        closure that captures the current source's cache_key and
        mastering chain config so the sidebar does not need to import
        the paths/config_hash machinery itself.

        Passing None clears the probe (bundle + export buttons stay
        disabled). Calling this triggers an immediate refresh of BOTH
        the bundle button and the Export button so their state reflects
        the new probe.
        """
        self._mastered_cache_fresh_probe = probe
        self._refresh_bundle_button_enabled()
        self._refresh_export_button_enabled()

    def refresh_export_button(self) -> None:
        """Public hook to re-evaluate the Export button's enabled state.

        Called by MainWindow whenever a keeper's mastered cache may have
        changed on disk (e.g. after ``mastering_complete`` fires, or when
        a batch master finishes). Mirrors :meth:`refresh_bundle_button`.
        """
        self._refresh_export_button_enabled()

    def _refresh_export_button_enabled(self) -> None:
        """Recompute the Export button enable + tooltip.

        Enable rule: enabled iff (a) at least one keeper exists, (b) a
        batch master is NOT in-flight, (c) a freshness probe is installed,
        and (d) every keeper has a fresh mastered cache. Reuses the SAME
        ``_mastered_cache_fresh_probe`` the bundle button uses. Disabled
        cases get an explanatory tooltip.
        """
        if not self._rows:
            self._export_button.setEnabled(False)
            self._export_button.setToolTip(
                self._EXPORT_DISABLED_NO_KEEPERS_TT
            )
            return
        if self._mastering_in_flight:
            self._export_button.setEnabled(False)
            self._export_button.setToolTip(
                self._EXPORT_DISABLED_UNMASTERED_TT
            )
            return
        probe = self._mastered_cache_fresh_probe
        if probe is None:
            self._export_button.setEnabled(False)
            self._export_button.setToolTip(
                self._EXPORT_DISABLED_UNMASTERED_TT
            )
            return
        try:
            all_fresh = all(probe(rid) for rid in self._rows.keys())
        except Exception:
            all_fresh = False
        if all_fresh:
            self._export_button.setEnabled(True)
            self._export_button.setToolTip(self._EXPORT_ENABLED_TT)
        else:
            self._export_button.setEnabled(False)
            self._export_button.setToolTip(
                self._EXPORT_DISABLED_UNMASTERED_TT
            )

    def refresh_bundle_button(self) -> None:
        """Public hook to re-evaluate the bundle button's enabled state.

        Called by MainWindow whenever a keeper's mastered cache may have
        changed on disk (e.g., after ``mastering_complete`` fires). The
        sidebar can't know cache freshness itself — the probe installed
        via :meth:`set_mastered_cache_fresh_probe` reads disk — so the
        owner must poke us when external state changes.
        """
        self._refresh_bundle_button_enabled()

    def _refresh_bundle_button_enabled(self) -> None:
        """Recompute bundle button enable + tooltip based on mastered-cache freshness.

        Enable rule (D-02): the button is enabled iff (a) at least one
        keeper exists AND (b) every keeper has a fresh mastered cache.
        Disabled cases get an explanatory tooltip.

        Called from the same call sites that already call
        :meth:`_refresh_batch_button_idle_enabled` (add_row / remove_row
        / clear). Also runs whenever
        :meth:`set_mastered_cache_fresh_probe` is called so MainWindow
        can refresh after a mastering completes.
        """
        if not self._rows:
            self._bundle_button.setEnabled(False)
            self._bundle_button.setToolTip(self._BUNDLE_DISABLED_NO_KEEPERS_TT)
            return
        probe = self._mastered_cache_fresh_probe
        if probe is None:
            # No probe installed → assume not-mastered (defensive
            # default in test contexts without MainWindow wiring).
            self._bundle_button.setEnabled(False)
            self._bundle_button.setToolTip(self._BUNDLE_DISABLED_UNMASTERED_TT)
            return
        try:
            all_fresh = all(probe(rid) for rid in self._rows.keys())
        except Exception:
            all_fresh = False
        if all_fresh:
            self._bundle_button.setEnabled(True)
            self._bundle_button.setToolTip(self._BUNDLE_ENABLED_TT)
        else:
            self._bundle_button.setEnabled(False)
            self._bundle_button.setToolTip(self._BUNDLE_DISABLED_UNMASTERED_TT)

    def _refresh_batch_button_idle_enabled(self) -> None:
        """Update the idle-state Master enabled flag + tooltip by row count.

        Called from :meth:`add_row` / :meth:`remove_row` / :meth:`clear`
        AND from :meth:`set_batch_state` when transitioning back to idle.
        No-op while a batch master is running (that state forces the
        button enabled with the cancel label in :meth:`set_batch_state`).
        """
        if self._batch_state != "idle":
            return
        if self._rows:
            self._batch_button.setEnabled(True)
            self._batch_button.setToolTip(self._BATCH_ENABLED_TOOLTIP)
        else:
            self._batch_button.setEnabled(False)
            self._batch_button.setToolTip(self._BATCH_DISABLED_TOOLTIP)

    def set_batch_state(self, state: str, ok_count: int = 0) -> None:
        """Drive the Master button's two-state label machine.

        Quick-260615-l4y — the morphing one-button machine was split into
        a persistent Master button and a persistent Export button, so the
        Master button now has only two states:

            ``"idle"``    → ``"Master All Keepers"`` (disabled iff no
                            keepers). Re-evaluates the Export gate.
            ``"running"`` → ``"Cancel mastering"`` (always enabled). Marks
                            mastering in-flight, which forces the Export
                            button disabled until the run finishes/cancels.

        Args:
            state: Target state name (``"idle"`` or ``"running"``).
            ok_count: Accepted for backward-compatibility with callers
                that still pass it; ignored (the dynamic count moved off
                the label per the two-button redesign).

        Raises:
            ValueError: ``state`` not in the two allowed values.
        """
        if state not in ("idle", "running"):
            raise ValueError(f"Unknown batch state: {state!r}")
        self._batch_state = state
        if state == "idle":
            self._mastering_in_flight = False
            self._batch_button.setText(self._BATCH_IDLE_LABEL)
            self._refresh_batch_button_idle_enabled()
        else:  # running
            self._mastering_in_flight = True
            self._batch_button.setText(self._BATCH_RUNNING_LABEL)
            self._batch_button.setEnabled(True)
            self._batch_button.setToolTip(self._BATCH_RUNNING_TOOLTIP)
        # Either transition affects the Export gate (running → disable;
        # idle → re-probe freshness).
        self._refresh_export_button_enabled()

    def _on_batch_button_clicked(self) -> None:
        """Route the Master click to the right signal based on state."""
        if self._batch_state == "idle":
            self.master_all_requested.emit()
        elif self._batch_state == "running":
            self.mastering_cancel_requested.emit()

    # ------------------------------------------------------------- internal
    def _emit_title_callback(self) -> None:
        if self._dock_title_callback is not None:
            try:
                self._dock_title_callback(self.keeper_count())
            except Exception:
                # Never let a misbehaving callback break a sidebar mutation.
                pass

    def _insert_row_chronologically(self, row: KeeperRow) -> None:
        """Insert ``row`` into ``_rows_layout`` at the right chronological index.

        Walks the existing rows in layout order — at the first existing
        row whose ``_row_start_sec`` is greater than the new row's, insert
        the new row BEFORE it. Otherwise append at the end (before the
        trailing stretch added in __init__).
        """
        insert_index = self._rows_layout.count() - 1  # before trailing stretch
        for i in range(self._rows_layout.count()):
            item = self._rows_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, KeeperRow):
                if w._row_start_sec > row._row_start_sec:
                    insert_index = i
                    break
        self._rows_layout.insertWidget(insert_index, row)

    # ----------------------------------------------------------- public API
    def add_row(self, region: Region) -> KeeperRow:
        """Insert a new KeeperRow chronologically and flip to populated state.

        Returns the constructed row so the MainWindow can directly drive
        its state if needed (analogous to LayersSidebar.add_dsp_row).
        """
        row = KeeperRow(
            region_id=region.id,
            start_sec=region.start_sec,
            end_sec=region.end_sec,
            state=region.state,
            note=region.note,
        )
        # Signal-to-signal forward — aggregates the per-row payload to
        # the sidebar-level signal automatically. Same pattern as
        # LayersSidebar (PySide6 Signal.connect(other_signal) overload).
        row.jump_requested.connect(self.jump_requested)
        row.play_requested.connect(self.play_requested)
        # quick-260622-sr8 — forward the two new play-position signals.
        row.play_middle_requested.connect(self.play_middle_requested)
        row.play_end_requested.connect(self.play_end_requested)
        row.delete_requested.connect(self.delete_requested)
        # Phase 7 Plan 07-02 Task 2 — forward Master button + Cancel
        # mastering signals. Signal-to-signal aggregation; the per-row
        # default-arg closure already captured ``region_id`` so no
        # additional binding is needed here.
        row.mastering_requested.connect(self.mastering_requested)
        row.cancel_mastering_requested.connect(self.cancel_mastering_requested)
        # Phase 8 Plan 08-04 Task 3 — forward per-row Share button click
        # to the sidebar-level signal. Signal-to-signal aggregation; the
        # per-row default-arg closure already captured region_id so no
        # additional binding is needed here.
        row.share_requested.connect(self.share_requested)
        # quick-260620-mgu NORM-04 — forward the per-row Normalize toggle to
        # the sidebar-level signal. Signal-to-signal aggregation; the per-row
        # closure already captured region_id.
        row.normalize_changed.connect(self.normalize_changed)
        # Phase 7 Plan 07-04 Task 2 — forward row's jump_requested to
        # selection_changed so MainWindow can track the most-recently-
        # clicked Keeper for the A/B toolbar widget. A left-click on
        # the row body is the selection semantics chosen by the plan
        # (UI-SPEC §"A/B Preview Toolbar Toggle" line 550). The two
        # signals carry the same payload (region_id) — a single click
        # both seeks the playhead AND selects the keeper for A/B.
        row.jump_requested.connect(self.selection_changed)
        self._insert_row_chronologically(row)
        self._rows[region.id] = row
        self._stack.setCurrentIndex(1)
        # Phase 7 Plan 07-06 — refresh batch-button enable state.
        self._refresh_batch_button_idle_enabled()
        # Phase 8 Plan 08-05 — refresh bundle-button enable state.
        self._refresh_bundle_button_enabled()
        # Quick-260615-l4y — refresh export-button enable state.
        self._refresh_export_button_enabled()
        self._emit_title_callback()
        return row

    def remove_row(self, region_id: str) -> None:
        """Drop the row + flip back to empty state if the panel is now empty."""
        row = self._rows.pop(region_id, None)
        if row is None:
            return
        self._rows_layout.removeWidget(row)
        try:
            row.deleteLater()
        except RuntimeError:
            pass
        if not self._rows:
            self._stack.setCurrentIndex(0)
        # Phase 7 Plan 07-06 — refresh batch-button enable state.
        self._refresh_batch_button_idle_enabled()
        # Phase 8 Plan 08-05 — refresh bundle-button enable state.
        self._refresh_bundle_button_enabled()
        # Quick-260615-l4y — refresh export-button enable state.
        self._refresh_export_button_enabled()
        self._emit_title_callback()

    def clear(self) -> None:
        """Drop every row + reset to empty state."""
        for region_id, row in list(self._rows.items()):
            self._rows_layout.removeWidget(row)
            try:
                row.deleteLater()
            except RuntimeError:
                pass
        self._rows.clear()
        self._stack.setCurrentIndex(0)
        # Phase 7 Plan 07-06 — refresh batch-button enable state.
        self._refresh_batch_button_idle_enabled()
        # Phase 8 Plan 08-05 — refresh bundle-button enable state.
        self._refresh_bundle_button_enabled()
        # Quick-260615-l4y — refresh export-button enable state.
        self._refresh_export_button_enabled()
        self._emit_title_callback()

    def update_row_state(self, region_id: str, state: str) -> None:
        """Update the row's badge OR remove it if the new state is non-keeper.

        UI-SPEC §Layout Architecture: the Keepers panel shows ONLY
        Keeper-state regions. If the overlay mutates a region to trash
        or untouched, the row must disappear from the panel.
        """
        row = self._rows.get(region_id)
        if row is None:
            return
        if state != "keeper":
            self.remove_row(region_id)
            return
        row.set_state(state)

    def update_row_range(self, region_id: str, start_sec: float, end_sec: float) -> None:
        """Refresh the row's time-block label + re-sort it chronologically."""
        row = self._rows.get(region_id)
        if row is None:
            return
        row.set_range(start_sec, end_sec)
        # Re-sort: remove from current position and re-insert at the new
        # chronological index. Cleanup-safe because removeWidget +
        # insertWidget without deleteLater keeps the Python ref alive.
        self._rows_layout.removeWidget(row)
        self._insert_row_chronologically(row)

    def keeper_count(self) -> int:
        """Total rows currently shown — equals number of Keeper regions."""
        return len(self._rows)

    def find_row(self, region_id: str) -> Optional[KeeperRow]:
        """Return the :class:`KeeperRow` for ``region_id``, or None if absent.

        Phase 7 Plan 07-02 Task 2 — lookup helper used by MainWindow to
        drive ``set_mastering_badge`` / ``set_mastering_status`` per
        keeper without the caller needing direct access to ``_rows``.
        """
        return self._rows.get(region_id)

    def set_dock_title_callback(
        self, cb: Optional[Callable[[int], None]]
    ) -> None:
        """Register (or clear) a callback fired with the count after every mutation.

        MainWindow uses this to keep the dock title ``Keepers (N)`` live.
        Passing ``None`` clears the callback.
        """
        self._dock_title_callback = cb

    # ----------------------------- Phase 8 Plan 08-05 Task 2 — drag reorder

    def current_order(self) -> list[str]:
        """Return the live region_id list in current visual (drag-reordered) order.

        Walks ``_rows_layout`` in index order and emits the ``_region_id``
        of every :class:`KeeperRow` widget. The trailing layout stretch
        is skipped (it has no widget).

        Used by:
            * :class:`marmelade.ui.bundle_dialog.BundleDialog` to read
              the user-arranged order at dialog-open time.
            * The reorder + restore paths to compute the new ordering
              vector for :attr:`order_changed`.
        """
        order: list[str] = []
        for i in range(self._rows_layout.count()):
            item = self._rows_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, KeeperRow):
                order.append(w._region_id)
        return order

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802 (Qt override)
        """Accept drags carrying a text/plain payload (the region_id).

        Per RESEARCH Pattern 4 lines 553-557 — the receiver MUST call
        ``acceptProposedAction`` for the drop to be eligible.
        """
        if e.mimeData().hasText():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e: QDragMoveEvent) -> None:  # noqa: N802 (Qt override)
        """Accept move events so the drop indicator tracks the cursor."""
        if e.mimeData().hasText():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802 (Qt override)
        """Reorder rows based on drop position.

        Algorithm:
            1. Resolve the dragged ``region_id`` from the mime text.
            2. Find the source row in ``_rows_layout``.
            3. Compute the drop target index by walking the rows and
               comparing each row's geometry center's y against the drop
               point's y — the first row whose center.y is greater than
               drop.y is the insert-before target; if none, append.
            4. If the resolved insert index equals the source's current
               index (drag-to-self), bail out without emitting
               ``order_changed``.
            5. Remove + re-insert the source widget; persist to
               QSettings; emit ``order_changed``.
        """
        if not e.mimeData().hasText():
            e.ignore()
            return
        src_id = e.mimeData().text()
        src_row = self._rows.get(src_id)
        if src_row is None:
            e.ignore()
            return

        # Walk the layout to find the source's current index AND the
        # drop target index.
        rows_in_layout: list[tuple[int, KeeperRow]] = []
        for i in range(self._rows_layout.count()):
            item = self._rows_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, KeeperRow):
                rows_in_layout.append((i, w))

        src_idx_in_layout = -1
        for layout_idx, w in rows_in_layout:
            if w is src_row:
                src_idx_in_layout = layout_idx
                break
        if src_idx_in_layout < 0:
            e.ignore()
            return

        # Compute the drop target index. The drop point's y is in the
        # sidebar's own coordinate space; each row's geometry is in the
        # rows_container coordinate space — but for the in-sidebar drag
        # we use the row widgets' own y in their parent scroll area,
        # which is the same coordinate system the layout walks. We
        # compare against the rows in _rows_layout (skipping the
        # trailing stretch).
        drop_y = float(e.position().y())
        # Map the drop point from sidebar coords down to rows_container
        # coords. The trailing stretch widget belongs to the rows
        # container; walking layout items directly is cheaper than
        # crossing coordinate spaces.
        #
        # For each row in _rows_layout, get its geometry.center().y in
        # its parent (rows_container) and find the first row whose
        # center.y > drop_y. That's where we insert BEFORE.
        target_layout_idx = self._rows_layout.count() - 1  # before trailing stretch
        for layout_idx, w in rows_in_layout:
            # We need to map the drop y from the sidebar's own coords to
            # the rows_container's coords. The simplest path is to use
            # the row's mapTo(self, geometry-center) and compare in
            # sidebar coordinates.
            try:
                center_global = w.mapTo(self, QPoint(0, w.height() // 2))
                center_y = float(center_global.y())
            except Exception:
                center_y = float(w.geometry().center().y())
            if center_y > drop_y:
                target_layout_idx = layout_idx
                break

        # Drag-to-self: the resolved target matches the source's current
        # position (either exactly OR target = source + 1, which would
        # also be a no-op after removeWidget + insertWidget at the same
        # spot).
        if target_layout_idx == src_idx_in_layout or target_layout_idx == src_idx_in_layout + 1:
            e.acceptProposedAction()
            return

        # Remove the source row from the layout (keeping the Python ref
        # alive) and re-insert at the new index. Note: removeWidget
        # shifts subsequent indices down by one, so if the target was
        # AFTER the source, decrement by one to land at the intended
        # visual position.
        self._rows_layout.removeWidget(src_row)
        if target_layout_idx > src_idx_in_layout:
            target_layout_idx -= 1
        self._rows_layout.insertWidget(target_layout_idx, src_row)
        e.acceptProposedAction()

        # Persist the new order to QSettings (per-sidecar namespace) and
        # emit order_changed so subscribers can react (BundleDialog).
        new_order = self.current_order()
        self._persist_order_to_qsettings(new_order)
        self.order_changed.emit(new_order)

    def set_sidecar_path(self, sidecar_path: Optional[str]) -> None:
        """Bind the bundle-order persistence namespace to ``sidecar_path``.

        Called by MainWindow on every file-open (and on close with
        ``None``). When ``sidecar_path`` is not None and a saved bundle
        order exists for the sidecar's SHA1 namespace, the rows are
        reordered to match it AND ``order_changed`` fires once with the
        restored order. When no saved order exists (or restore would
        produce the current order), the rows stay in chronological
        order (the add_row default).
        """
        self._sidecar_path = sidecar_path
        if sidecar_path is None:
            return
        saved = self._load_order_from_qsettings()
        if not saved:
            return
        # Reorder rows to match the saved order. Only reorder IDs we
        # currently have (a saved order may include keepers that the
        # user has since deleted; ignore those).
        current = set(self.current_order())
        target_order = [rid for rid in saved if rid in current]
        # Append any current rows that weren't in the saved order at
        # the tail (defensive — new keepers added since last save).
        for rid in self.current_order():
            if rid not in target_order:
                target_order.append(rid)
        if target_order == self.current_order():
            return  # already in the saved order — no-op
        # Re-stack the rows in target order.
        for rid in target_order:
            row = self._rows.get(rid)
            if row is None:
                continue
            self._rows_layout.removeWidget(row)
            # Insert before the trailing stretch (which is at the last
            # index after each removeWidget).
            insert_idx = self._rows_layout.count() - 1
            self._rows_layout.insertWidget(insert_idx, row)
        self.order_changed.emit(self.current_order())

    def _persist_order_to_qsettings(self, order: list[str]) -> None:
        """Write ``order`` as JSON under ``youtube/bundle_order/<sidecar-sha1>``.

        No-op when no sidecar path is bound. Uses explicit
        ``QSettings("Marmelade", "Marmelade")`` per Shared Pattern
        4 — bare ``QSettings()`` is forbidden.
        """
        if self._sidecar_path is None:
            return
        s = QSettings("Marmelade", "Marmelade")
        key = f"youtube/bundle_order/{_sidecar_namespace_key(self._sidecar_path)}"
        s.setValue(key, json.dumps(list(order)))
        s.sync()

    def _load_order_from_qsettings(self) -> list[str]:
        """Read the saved bundle order for the current sidecar.

        Returns an empty list when nothing is saved OR the value cannot
        be parsed as JSON (defensive — a corrupted settings file should
        not crash the sidebar).
        """
        if self._sidecar_path is None:
            return []
        s = QSettings("Marmelade", "Marmelade")
        key = f"youtube/bundle_order/{_sidecar_namespace_key(self._sidecar_path)}"
        raw = s.value(key, "")
        if not raw:
            return []
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass
        return []
