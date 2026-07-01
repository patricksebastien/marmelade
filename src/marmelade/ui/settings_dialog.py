"""Marmelade Preferences dialog (Phase 8 Plan 08-02 — D-10).

Introduces the Preferences surface for v1: a modal QDialog with a single
``"YouTube"`` QGroupBox showing the current OAuth connection state plus a
Connect/Disconnect button. Forward-compatible — future phases can add
additional QGroupBoxes (e.g., bundle defaults, export format defaults)
without touching this skeleton.

D-10 (Settings panel for connection management) — the single discoverable
place for YouTube connection management. Accessed via ``View → Preferences…``
in the main window menu bar.

Signal/slot wiring map:

    SettingsDialog button click
        → SettingsDialog.youtube_{connect,disconnect}_requested.emit()
        → MainWindow._on_youtube_{connect,disconnect}() slot
        → marmelade.youtube.oauth.{first_time_connect,disconnect}()
        → MainWindow updates dialog state via dlg.update_connection_state()

State is owned by MainWindow (which reads
:func:`marmelade.youtube.oauth.is_connected` + :func:`channel_info` at
dialog-open time and pushes state into the dialog via the constructor
+ :meth:`update_connection_state`). The dialog itself does not import
anything from :mod:`marmelade.youtube` — keeps the UI test surface
clean of network mocks.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# Object names — set on the widgets so tests + accessibility tools can
# locate them deterministically (and so future stylesheets can target them).
_YT_BOX_NAME = "youtube_box"
_YT_LABEL_NAME = "youtube_status_label"
_YT_BTN_NAME = "youtube_action_button"
_PLAYBACK_BOX_NAME = "playback_box"
_PLAYHEAD_OFFSET_SPIN_NAME = "playhead_offset_spinbox"

# Default playhead visual offset (seconds) used when the caller does not pass
# one. MainWindow owns the persisted value (QSettings); this is only the
# standalone-construction fallback.
_DEFAULT_PLAYHEAD_OFFSET_SEC = 0.15


class SettingsDialog(QDialog):
    """Modal Preferences dialog hosting per-feature configuration panels.

    Args:
        is_connected: ``True`` when :func:`marmelade.youtube.oauth.is_connected`
            returned True at dialog-open time. Drives the initial label +
            button rendering.
        channel_name: Authenticated YouTube channel display name. Required
            when ``is_connected=True`` (the label reads "Connected as
            <channel_name>"); ignored otherwise.
        parent: Optional parent widget (typically :class:`MainWindow`).

    Signals:
        youtube_connect_requested: Emitted when the user clicks the
            ``"Connect YouTube"`` button (disconnected state). No payload.
        youtube_disconnect_requested: Emitted when the user clicks the
            ``"Disconnect"`` button (connected state). No payload.
    """

    youtube_connect_requested = Signal()
    youtube_disconnect_requested = Signal()
    # Emitted live as the user adjusts the playhead visual offset spinbox.
    # Payload = the new offset in seconds. MainWindow applies it to the
    # playhead immediately and persists it to QSettings.
    playhead_offset_changed = Signal(float)

    def __init__(
        self,
        is_connected: bool,
        channel_name: str | None = None,
        playhead_offset_sec: float = _DEFAULT_PLAYHEAD_OFFSET_SEC,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.resize(480, 360)

        # Outer layout — mirrors mastering_dialog.py:168-216 shape
        # (Section header / per-feature group boxes / bottom button row).
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(8)

        # ------------------------------------------------------------------
        # YouTube section (D-10) — the only group in v1; forward-compatible.
        # ------------------------------------------------------------------
        self._yt_box = QGroupBox("YouTube")
        self._yt_box.setObjectName(_YT_BOX_NAME)
        yt_layout = QHBoxLayout(self._yt_box)
        yt_layout.setContentsMargins(12, 12, 12, 12)
        yt_layout.setSpacing(8)

        # Status label + action button. Initial state derived from the
        # constructor args; the rebuild lives in :meth:`update_connection_state`.
        self._status_label = QLabel()
        self._status_label.setObjectName(_YT_LABEL_NAME)
        self._action_btn = QPushButton()
        self._action_btn.setObjectName(_YT_BTN_NAME)
        # Track current click-handler state so update_connection_state can
        # disconnect without triggering the PySide6 "Failed to disconnect"
        # RuntimeWarning on the first render (no prior connection).
        self._current_action: str = "none"  # "connect" | "disconnect" | "none"

        yt_layout.addWidget(self._status_label, 1)
        yt_layout.addWidget(self._action_btn, 0)

        outer.addWidget(self._yt_box)

        # ------------------------------------------------------------------
        # Playback section — playhead visual sync trim. The drawn playhead
        # can lag the sound by the GUI/render pipeline delay; this draws it
        # that many seconds AHEAD so the line sits on the waveform feature
        # you are hearing. POSITIVE = playhead further ahead (raise it if the
        # sound still leads the playhead). Cosmetic only — never affects
        # audio, seek, or export.
        # ------------------------------------------------------------------
        self._playback_box = QGroupBox("Playback")
        self._playback_box.setObjectName(_PLAYBACK_BOX_NAME)
        pb_layout = QVBoxLayout(self._playback_box)
        pb_layout.setContentsMargins(12, 12, 12, 12)
        pb_layout.setSpacing(6)

        offset_row = QHBoxLayout()
        offset_row.setSpacing(8)
        offset_label = QLabel("Playhead visual offset")
        self._playhead_offset_spin = QDoubleSpinBox()
        self._playhead_offset_spin.setObjectName(_PLAYHEAD_OFFSET_SPIN_NAME)
        self._playhead_offset_spin.setRange(-2.0, 2.0)
        self._playhead_offset_spin.setSingleStep(0.01)
        self._playhead_offset_spin.setDecimals(2)
        self._playhead_offset_spin.setSuffix(" s")
        self._playhead_offset_spin.setValue(float(playhead_offset_sec))
        self._playhead_offset_spin.setToolTip(
            "Draw the playhead this many seconds ahead of the audio to cancel "
            "display lag. Increase if the sound still plays before the "
            "playhead reaches it; decrease if the playhead now leads the sound."
        )
        self._playhead_offset_spin.valueChanged.connect(
            self.playhead_offset_changed.emit
        )
        offset_row.addWidget(offset_label, 1)
        offset_row.addWidget(self._playhead_offset_spin, 0)
        pb_layout.addLayout(offset_row)

        offset_hint = QLabel(
            "Raise if the sound plays before the playhead; lower if the "
            "playhead leads the sound."
        )
        offset_hint.setWordWrap(True)
        offset_hint.setStyleSheet("color: #9CA3AF; font-size: 9pt;")
        pb_layout.addWidget(offset_hint)

        outer.addWidget(self._playback_box)
        outer.addStretch(1)

        # Bottom button row — single Close button (mirrors mastering_dialog.py
        # pattern but with Close-only since this dialog has no Apply concept
        # — every state change is committed synchronously through the
        # connect/disconnect slots).
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        # Standard buttons' default trigger maps to ``rejected`` for Close —
        # the explicit connect above is defense in depth in case the QStyle
        # changes the standard-button role mapping.
        close_btn = button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.reject)
        outer.addLayout(self._wrap_in_hbox(button_box))

        # Initial render.
        self.update_connection_state(is_connected, channel_name)

    @staticmethod
    def _wrap_in_hbox(widget: QWidget | QDialogButtonBox) -> QHBoxLayout:
        """Right-align the bottom button row inside an HBox + stretch."""
        h = QHBoxLayout()
        h.setContentsMargins(0, 8, 0, 0)
        h.addStretch(1)
        h.addWidget(widget)
        return h

    # ------------------------------------------------------------------ API

    def update_connection_state(
        self, is_connected: bool, channel_name: str | None
    ) -> None:
        """Rebuild the label text + button text + button click handler.

        Called by MainWindow after a successful Connect/Disconnect. The
        button is the SAME widget across both states — only its text and
        ``clicked`` connection target change. We disconnect any previous
        slot before reconnecting so a Connect → Disconnect → Connect cycle
        doesn't accidentally fire BOTH signals on a single click.

        Args:
            is_connected: New connection state.
            channel_name: New channel display name (required when
                ``is_connected=True``; ignored otherwise).
        """
        # Tear down any previous click connection so a Connect → Disconnect
        # cycle doesn't accidentally fire BOTH signals on a single click.
        # We disconnect the specific previous slot only when one was
        # connected — avoids the PySide6 "Failed to disconnect" RuntimeWarning
        # on the first render.
        if self._current_action == "connect":
            self._action_btn.clicked.disconnect(
                self.youtube_connect_requested.emit
            )
        elif self._current_action == "disconnect":
            self._action_btn.clicked.disconnect(
                self.youtube_disconnect_requested.emit
            )

        if is_connected:
            self._status_label.setText(f"Connected as {channel_name or ''}".rstrip())
            self._action_btn.setText("Disconnect")
            self._action_btn.clicked.connect(self.youtube_disconnect_requested.emit)
            self._current_action = "disconnect"
            # Accessibility: tooltip explains the consequence.
            self._action_btn.setToolTip(
                "Disconnect this YouTube account. Local credentials will "
                "be cleared and the access token revoked at Google."
            )
        else:
            self._status_label.setText("Not connected")
            self._action_btn.setText("Connect YouTube")
            self._action_btn.clicked.connect(self.youtube_connect_requested.emit)
            self._current_action = "connect"
            self._action_btn.setToolTip(
                "Open the Marmelade consent screen in your system browser "
                "to authorize uploads to your YouTube channel."
            )
