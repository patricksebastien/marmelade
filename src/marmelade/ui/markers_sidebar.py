"""Markers QDockWidget body (quick-260701-jc5 — MARK-02).

A side panel listing every marker with a monospace time label, an editable
text label, and a Delete button, plus a persistent "Add marker" [+] button
above the row list. Structural sibling of
:mod:`marmelade.ui.keepers_sidebar` (KeepersSidebar / KeeperRow): an
aggregating :class:`MarkersSidebar` widget owning a :class:`MarkerRow` per
marker, with per-row signals forwarded to sidebar-level signals.

The sidebar contract — typed signals out, public methods in:

    MarkersSidebar.add_requested = Signal()
        The [+] button. MainWindow connects this to ``_action_add_marker``
        (same live-playhead position source as the "m" shortcut).
    MarkersSidebar.jump_requested = Signal(str)
        Forwarded from each row body left-click. MainWindow connects this to
        seek + play (mirrors _on_keeper_play).
    MarkersSidebar.delete_requested = Signal(str)
        Forwarded from each row's Delete button. MainWindow removes the row,
        overlay line, and sidecar entry.
    MarkersSidebar.label_edited = Signal(str, str)
        (marker_id, new_label) — forwarded from each row's QLineEdit
        ``editingFinished``. MainWindow persists + updates the overlay label.

    MarkersSidebar.add_row(marker) -> MarkerRow
        Chronological insert (sorted by time_sec asc). Flips to populated.
    MarkersSidebar.remove_row(marker_id)
        Remove a row. Flips back to empty when the panel is emptied.
    MarkersSidebar.clear()
        Drop every row + reset to empty state.
    MarkersSidebar.set_add_enabled(bool)
        Enable/disable the [+] button (a file must be open to add markers).
    MarkersSidebar.marker_count() -> int
        Total rows currently shown.
    MarkersSidebar.set_dock_title_callback(cb)
        Register a callback invoked with the current count after every
        add/remove so MainWindow can keep the dock title "Markers (N)" live.

Closure-binding discipline (Phase 1 LEARNINGS): every per-row signal connect
uses the default-arg trick (``lambda mid=marker_id: ...``) so a for-loop
adding multiple rows does not late-bind the loop variable.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from marmelade.audio.sidecar_cache import Marker, _MAX_NOTE_LEN
from marmelade.ui.icons import _play_start_icon


def _fmt_hhmmss(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` (24h, no rollover). Always 8 chars.

    Mirrors :func:`marmelade.ui.keepers_sidebar._fmt_hhmmss` so marker times
    read identically to keeper times.
    """
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class MarkerRow(QWidget):
    """One row inside the Markers dock's populated state.

    Composition (single QHBoxLayout, left-to-right):
        [Play QToolButton] [time QLabel] [label QLineEdit (editable)]
        [stretch] [Delete QPushButton]

    Signals:
        jump_requested(marker_id): emitted on the Play button click AND on a
            left-mouse press landing on the row body (not the QLineEdit /
            Play / Delete buttons — those consume their own clicks).
            MainWindow connects to seek + play (playback starts at the marker).
        delete_requested(marker_id): emitted on the Delete button click.
        label_edited(marker_id, new_label): emitted on the QLineEdit's
            ``editingFinished``. MainWindow persists + updates the overlay.

    Attributes:
        _marker_id: stable id captured at construction (default-arg closure
            binding on every signal connect uses this).
        _row_time_sec: cached for chronological-insertion sort key.
    """

    jump_requested = Signal(str)
    delete_requested = Signal(str)
    label_edited = Signal(str, str)

    def __init__(
        self,
        marker_id: str,
        time_sec: float,
        label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._marker_id = marker_id
        self._row_time_sec = float(time_sec)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 8, 8)
        layout.setSpacing(8)

        # ---- Play button (before the timecode, mirrors KeeperRow._play) ----
        # Same 24x24 button / 20x20 painted start-glyph as the Keepers row.
        # Clicking it reuses the row's ``jump_requested`` signal, whose
        # MainWindow sink (_on_marker_jump) already seeks + starts playback
        # at the marker — so "play at this position" needs no new signal.
        self._play = QPushButton()
        self._play.setFixedSize(24, 24)
        self._play.setIconSize(QSize(20, 20))
        self._play.setIcon(_play_start_icon())
        self._play.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play.setToolTip("Play from this marker position.")
        self._play.setAccessibleName(
            f"Play from marker at {_fmt_hhmmss(time_sec)}"
        )
        self._play.clicked.connect(
            lambda _checked=False, mid=marker_id: self.jump_requested.emit(mid)
        )
        layout.addWidget(self._play, 0)

        # ---- Time block ----
        self._time_label = QLabel(_fmt_hhmmss(time_sec))
        self._time_label.setStyleSheet(
            "color: #E6E6E6; font-family: monospace; font-size: 8pt;"
        )
        layout.addWidget(self._time_label, 0)

        # ---- Editable label ----
        self._label_edit = QLineEdit(label)
        self._label_edit.setPlaceholderText("Label…")
        # Cap to the sidecar's marker-label length so the UI can never
        # produce a value the validator would quarantine (T-jc5-03).
        self._label_edit.setMaxLength(_MAX_NOTE_LEN)
        self._label_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._label_edit.editingFinished.connect(
            lambda mid=marker_id: self.label_edited.emit(
                mid, self._label_edit.text()
            )
        )
        layout.addWidget(self._label_edit, 1)

        layout.addStretch(0)

        # ---- Delete button ----
        self._delete = QPushButton("Delete")
        self._delete.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._delete.clicked.connect(
            lambda _checked=False, mid=marker_id: self.delete_requested.emit(mid)
        )
        layout.addWidget(self._delete, 0)

    # ----------------------------------------------------------- public API
    def set_time(self, time_sec: float) -> None:
        """Refresh the time-block label + cached sort key."""
        self._row_time_sec = float(time_sec)
        self._time_label.setText(_fmt_hhmmss(time_sec))

    def label_text(self) -> str:
        """Return the current QLineEdit label text."""
        return self._label_edit.text()

    def set_label(self, label: str) -> None:
        """Set the QLineEdit text without emitting label_edited."""
        self._label_edit.setText(label)

    def focus_label(self) -> None:
        """Put the keyboard focus in the label field, ready for typing.

        Called by MainWindow right after a marker is created (via the "m"
        key OR the [+] button) so the user can type the label immediately.
        Selects any existing text so typing replaces it. Only invoked on
        creation — existing rows are never auto-focused.
        """
        self._label_edit.setFocus(Qt.FocusReason.OtherFocusReason)
        self._label_edit.selectAll()

    # ----------------------------------------------- mouse event overrides
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Emit ``jump_requested`` on a left-press over the row body.

        Child widgets (QLineEdit / Delete button) receive mouse events first
        and consume their own clicks, so this fires only for clicks landing
        on the time label area or the row padding — NOT while typing in the
        label field. Mirrors KeeperRow's mousePressEvent contract.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.jump_requested.emit(self._marker_id)
        super().mousePressEvent(event)


class MarkersSidebar(QWidget):
    """The body widget of the Markers QDockWidget.

    Mirrors the KeepersSidebar shape:
        * Outer ``QVBoxLayout`` (zero margins — the dock chrome separates).
        * A persistent "Add marker" [+] button ABOVE the stack (visible in
          BOTH empty and populated states; enabled only when a file is open).
        * ``QStackedWidget`` with two pages:
            * Page 0 — empty state ("No markers yet" heading + helper copy).
            * Page 1 — populated state (QScrollArea wrapping a row container).

    Signals:
        add_requested(): the [+] button.
        jump_requested(marker_id): aggregated from each row.
        delete_requested(marker_id): aggregated from each row.
        label_edited(marker_id, new_label): aggregated from each row.
    """

    add_requested = Signal()
    jump_requested = Signal(str)
    delete_requested = Signal(str)
    label_edited = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setMinimumWidth(240)

        # ---- Persistent "Add marker" [+] button (above the stack) ----
        self._add_button = QPushButton("+ Add marker", self)
        self._add_button.setMinimumHeight(32)
        self._add_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._add_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._add_button.setToolTip(
            "Drop a marker at the current playhead position "
            "(or press M)."
        )
        # Disabled until a file is open — MainWindow drives set_add_enabled.
        self._add_button.setEnabled(False)
        self._add_button.clicked.connect(self.add_requested.emit)
        outer.addWidget(self._add_button)

        self._stack = QStackedWidget()

        # ---- Page 0 — empty state ----
        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        empty_layout.setSpacing(8)

        heading = QLabel("No markers yet")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet(
            "font-size: 11pt; font-weight: 600; color: #E6E6E6;"
        )

        body = QLabel(
            "Press M (or click + Add marker) while playing to drop a "
            "marker at the current playhead position."
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
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._rows_layout.addStretch(1)  # rows pack to the top
        scroll.setWidget(rows_container)

        self._stack.addWidget(empty_page)  # index 0
        self._stack.addWidget(scroll)  # index 1
        self._stack.setCurrentIndex(0)
        outer.addWidget(self._stack)

        # Row registry keyed by marker_id for O(1) lookup.
        self._rows: dict[str, MarkerRow] = {}
        self._dock_title_callback: Optional[Callable[[int], None]] = None

    # ------------------------------------------------------------- internal
    def _emit_title_callback(self) -> None:
        if self._dock_title_callback is not None:
            try:
                self._dock_title_callback(self.marker_count())
            except Exception:
                pass

    def _insert_row_chronologically(self, row: MarkerRow) -> None:
        """Insert ``row`` at the right chronological index by ``_row_time_sec``.

        Walks existing rows in layout order; inserts before the first row
        whose time is greater, else appends before the trailing stretch.
        """
        insert_index = self._rows_layout.count() - 1  # before trailing stretch
        for i in range(self._rows_layout.count()):
            item = self._rows_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, MarkerRow):
                if w._row_time_sec > row._row_time_sec:
                    insert_index = i
                    break
        self._rows_layout.insertWidget(insert_index, row)

    # ----------------------------------------------------------- public API
    def add_row(self, marker: Marker) -> MarkerRow:
        """Insert a new MarkerRow chronologically and flip to populated state.

        Returns the constructed row so MainWindow can drive it directly.
        """
        row = MarkerRow(
            marker_id=marker.id,
            time_sec=marker.time_sec,
            label=marker.label,
        )
        # Signal-to-signal forwarding (per-row closures already captured the
        # marker_id, so no additional binding is needed here).
        row.jump_requested.connect(self.jump_requested)
        row.delete_requested.connect(self.delete_requested)
        row.label_edited.connect(self.label_edited)

        self._insert_row_chronologically(row)
        self._rows[marker.id] = row
        self._stack.setCurrentIndex(1)
        self._emit_title_callback()
        return row

    def remove_row(self, marker_id: str) -> None:
        """Drop the row + flip back to empty state if the panel is now empty."""
        row = self._rows.pop(marker_id, None)
        if row is None:
            return
        self._rows_layout.removeWidget(row)
        try:
            row.deleteLater()
        except RuntimeError:
            pass
        if not self._rows:
            self._stack.setCurrentIndex(0)
        self._emit_title_callback()

    def clear(self) -> None:
        """Drop every row + reset to empty state."""
        for marker_id, row in list(self._rows.items()):
            self._rows_layout.removeWidget(row)
            try:
                row.deleteLater()
            except RuntimeError:
                pass
        self._rows.clear()
        self._stack.setCurrentIndex(0)
        self._emit_title_callback()

    def update_row_label(self, marker_id: str, label: str) -> None:
        """Set a row's QLineEdit text without emitting label_edited."""
        row = self._rows.get(marker_id)
        if row is None:
            return
        row.set_label(label)

    def find_row(self, marker_id: str) -> Optional[MarkerRow]:
        """Return the :class:`MarkerRow` for ``marker_id`` or None."""
        return self._rows.get(marker_id)

    def marker_count(self) -> int:
        """Total rows currently shown."""
        return len(self._rows)

    def set_add_enabled(self, enabled: bool) -> None:
        """Enable/disable the [+] Add-marker button (file-open gate)."""
        self._add_button.setEnabled(bool(enabled))

    def set_dock_title_callback(
        self, cb: Optional[Callable[[int], None]]
    ) -> None:
        """Register (or clear) a callback fired with the count after every
        mutation. MainWindow keeps the dock title ``Markers (N)`` live."""
        self._dock_title_callback = cb
