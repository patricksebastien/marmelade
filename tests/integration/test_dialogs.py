"""Integration: dialog helper copy matches UI-SPEC verbatim.

We don't actually .exec() the QMessageBox in these tests (modal blocks the
event loop). Instead we patch ``QMessageBox.exec`` to a no-op-returning
stub so the helper builds a real QMessageBox we can inspect.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from marmelade.ui import dialogs, theme


@pytest.fixture(autouse=True)
def _theme(qapp):
    theme.apply_theme(QApplication.instance())


@pytest.fixture(autouse=True)
def _no_modal(monkeypatch: pytest.MonkeyPatch):
    """Stop QMessageBox.exec from blocking — capture the constructed box."""
    boxes: list[QMessageBox] = []
    original_exec = QMessageBox.exec

    def fake_exec(self: QMessageBox) -> int:
        boxes.append(self)
        # Simulate the user clicking the default button.
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    yield boxes
    QMessageBox.exec = original_exec


def test_show_unsupported_format_copy(qtbot, _no_modal: list[QMessageBox]) -> None:
    dialogs.show_unsupported_format(None, "weird.txt")
    assert len(_no_modal) == 1
    box = _no_modal[0]
    assert box.windowTitle() == "Couldn't open file"
    assert "weird.txt" in box.text()
    assert "isn't a supported audio format" in box.text()
    assert "Marmelade opens WAV, FLAC, and MP3 files." in box.text()
    button_texts = [b.text() for b in box.buttons()]
    assert "Choose a different file" in button_texts


def test_show_corrupt_file_copy(qtbot, _no_modal: list[QMessageBox]) -> None:
    dialogs.show_corrupt_file(None, "broken.wav", "pedalboard: bad header")
    assert len(_no_modal) == 1
    box = _no_modal[0]
    assert box.windowTitle() == "Couldn't open file"
    assert "broken.wav" in box.text()
    assert "couldn't be read" in box.text()
    assert "may be corrupt or in use by another program" in box.text()
    assert "Underlying error: pedalboard: bad header" in box.text()
    button_texts = [b.text() for b in box.buttons()]
    assert "Try again" in button_texts
    assert "Close" in button_texts


def test_show_too_long_copy(qtbot, _no_modal: list[QMessageBox]) -> None:
    dialogs.show_too_long(None, "marathon.wav", "9:14:22")
    assert len(_no_modal) == 1
    box = _no_modal[0]
    assert box.windowTitle() == "File is longer than supported"
    assert "marathon.wav" in box.text()
    assert "9:14:22" in box.text()
    assert "Marmelade supports files up to 8 hours in this version." in box.text()
    button_texts = [b.text() for b in box.buttons()]
    assert "Choose a different file" in button_texts
