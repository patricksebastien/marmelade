"""Phase 8 Plan 08-02 Task 2 (GREEN) — SettingsDialog YouTube section.

Pins the contract published by :class:`marmelade.ui.settings_dialog.SettingsDialog`:

* Disconnected state shows ``"Not connected"`` + a ``"Connect YouTube"``
  button.
* Connected state shows ``"Connected as <channel_name>"`` + a
  ``"Disconnect"`` button.
* Clicking the action button in disconnected state emits
  ``youtube_connect_requested`` exactly once (no payload).
* Clicking the action button in connected state emits
  ``youtube_disconnect_requested`` exactly once (no payload).
* :meth:`update_connection_state` swaps the label + button text + the
  signal the button emits when clicked.

D-10 (Settings panel for connection management) — single discoverable
place for YouTube connection management.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QPushButton

from marmelade.ui.settings_dialog import SettingsDialog


# ---------------------------------------------------------------------------
# Test 1 — disconnected state renders correctly.
# ---------------------------------------------------------------------------


def test_dialog_renders_disconnected(qtbot, qapp) -> None:
    """SettingsDialog(is_connected=False) shows 'Not connected' + Connect button."""
    dlg = SettingsDialog(is_connected=False, channel_name=None)
    qtbot.add_widget(dlg)

    yt_box = dlg.findChild(QGroupBox, "youtube_box")
    assert yt_box is not None, "YouTube QGroupBox must be present"
    assert yt_box.title() == "YouTube"

    label = dlg.findChild(QLabel, "youtube_status_label")
    assert label is not None
    assert label.text() == "Not connected"

    btn = dlg.findChild(QPushButton, "youtube_action_button")
    assert btn is not None
    assert btn.text() == "Connect YouTube"


# ---------------------------------------------------------------------------
# Test 2 — connected state renders correctly.
# ---------------------------------------------------------------------------


def test_dialog_renders_connected(qtbot, qapp) -> None:
    """SettingsDialog(is_connected=True, channel_name=...) shows Disconnect."""
    dlg = SettingsDialog(is_connected=True, channel_name="Patrick's Channel")
    qtbot.add_widget(dlg)

    label = dlg.findChild(QLabel, "youtube_status_label")
    assert label is not None
    assert "Connected as Patrick's Channel" in label.text()

    btn = dlg.findChild(QPushButton, "youtube_action_button")
    assert btn is not None
    assert btn.text() == "Disconnect"


# ---------------------------------------------------------------------------
# Test 3 — connect button emits the connect signal.
# ---------------------------------------------------------------------------


def test_connect_button_emits_signal(qtbot, qapp) -> None:
    """Clicking the action button in disconnected state emits youtube_connect_requested."""
    dlg = SettingsDialog(is_connected=False, channel_name=None)
    qtbot.add_widget(dlg)
    btn = dlg.findChild(QPushButton, "youtube_action_button")
    assert btn is not None
    with qtbot.waitSignal(dlg.youtube_connect_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)


# ---------------------------------------------------------------------------
# Test 4 — disconnect button emits the disconnect signal.
# ---------------------------------------------------------------------------


def test_disconnect_button_emits_signal(qtbot, qapp) -> None:
    """Clicking the action button in connected state emits youtube_disconnect_requested."""
    dlg = SettingsDialog(is_connected=True, channel_name="X")
    qtbot.add_widget(dlg)
    btn = dlg.findChild(QPushButton, "youtube_action_button")
    assert btn is not None
    with qtbot.waitSignal(dlg.youtube_disconnect_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)


# ---------------------------------------------------------------------------
# Test 5 — update_connection_state swaps label + button text + signal.
# ---------------------------------------------------------------------------


def test_update_connection_state_swaps_button(qtbot, qapp) -> None:
    """update_connection_state(True, name) flips label + button + emit target."""
    dlg = SettingsDialog(is_connected=False, channel_name=None)
    qtbot.add_widget(dlg)
    label = dlg.findChild(QLabel, "youtube_status_label")
    btn = dlg.findChild(QPushButton, "youtube_action_button")
    assert label is not None and btn is not None

    # Pre-condition.
    assert label.text() == "Not connected"
    assert btn.text() == "Connect YouTube"

    dlg.update_connection_state(True, "My Channel")

    # Label + button text updated in place.
    label = dlg.findChild(QLabel, "youtube_status_label")
    btn = dlg.findChild(QPushButton, "youtube_action_button")
    assert "Connected as My Channel" in label.text()
    assert btn.text() == "Disconnect"

    # Clicking the same button now emits the disconnect signal, NOT connect.
    connect_received: list[bool] = []
    dlg.youtube_connect_requested.connect(lambda: connect_received.append(True))

    with qtbot.waitSignal(dlg.youtube_disconnect_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)

    assert connect_received == [], (
        "youtube_connect_requested must NOT fire after update_connection_state(True, ...)"
    )

    # And back to disconnected for symmetry.
    dlg.update_connection_state(False, None)
    label = dlg.findChild(QLabel, "youtube_status_label")
    btn = dlg.findChild(QPushButton, "youtube_action_button")
    assert label.text() == "Not connected"
    assert btn.text() == "Connect YouTube"


# ---------------------------------------------------------------------------
# quick-260625 — Playback section: playhead visual offset control.
# ---------------------------------------------------------------------------


def test_dialog_has_playback_offset_spinbox(qtbot, qapp) -> None:
    """SettingsDialog exposes a Playback group with the offset spinbox seeded."""
    from PySide6.QtWidgets import QDoubleSpinBox

    dlg = SettingsDialog(
        is_connected=False, channel_name=None, playhead_offset_sec=0.12
    )
    qtbot.add_widget(dlg)

    box = dlg.findChild(QGroupBox, "playback_box")
    assert box is not None and box.title() == "Playback"

    spin = dlg.findChild(QDoubleSpinBox, "playhead_offset_spinbox")
    assert spin is not None
    assert spin.value() == pytest.approx(0.12)
    assert spin.suffix() == " s"


def test_offset_spinbox_emits_playhead_offset_changed(qtbot, qapp) -> None:
    """Changing the spinbox emits playhead_offset_changed with the new value."""
    from PySide6.QtWidgets import QDoubleSpinBox

    dlg = SettingsDialog(
        is_connected=False, channel_name=None, playhead_offset_sec=0.15
    )
    qtbot.add_widget(dlg)
    spin = dlg.findChild(QDoubleSpinBox, "playhead_offset_spinbox")

    with qtbot.waitSignal(dlg.playhead_offset_changed, timeout=1000) as blocker:
        spin.setValue(0.20)
    assert blocker.args[0] == pytest.approx(0.20)
