"""Three UI-SPEC error/info dialog helpers — verbatim copy.

UI-SPEC §Copywriting > Error states:
    * Unsupported format → "Couldn't open file" / "{filename} isn't a supported
      audio format. Marmelade opens WAV, FLAC, and MP3 files." /
      "Choose a different file"
    * Corrupt file → "Couldn't open file" / "{filename} couldn't be read. The
      file may be corrupt or in use by another program. Underlying error:
      {exception_message}" / "Try again" + "Close"
    * Too long → "File is longer than supported" / "{filename} is {duration_str}
      long. Marmelade supports files up to 8 hours in this version." /
      "Choose a different file"

All three are modal ``QMessageBox`` with the warning icon. Functions return
the user's button choice (a ``QMessageBox.StandardButton``) so the caller can
branch on Retry / Close for the corrupt-file case.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget


def show_unsupported_format(
    parent: QWidget | None,
    filename: str,
) -> QMessageBox.StandardButton:
    """UI-SPEC: unsupported audio format → warning modal."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Couldn't open file")
    box.setText(
        f"{filename} isn't a supported audio format. "
        "Marmelade opens WAV, FLAC, and MP3 files."
    )
    choose = box.addButton(
        "Choose a different file", QMessageBox.ButtonRole.AcceptRole
    )
    box.setDefaultButton(choose)
    box.exec()
    return QMessageBox.StandardButton.Ok


def show_corrupt_file(
    parent: QWidget | None,
    filename: str,
    exception_message: str,
) -> QMessageBox.StandardButton:
    """UI-SPEC: corrupt / unreadable file → warning modal with Retry / Close.

    Returns ``QMessageBox.StandardButton.Retry`` if the user clicked "Try
    again", ``QMessageBox.StandardButton.Close`` otherwise.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Couldn't open file")
    box.setText(
        f"{filename} couldn't be read. "
        "The file may be corrupt or in use by another program. "
        f"Underlying error: {exception_message}"
    )
    retry = box.addButton("Try again", QMessageBox.ButtonRole.AcceptRole)
    close = box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(retry)
    box.exec()
    clicked = box.clickedButton()
    if clicked is retry:
        return QMessageBox.StandardButton.Retry
    return QMessageBox.StandardButton.Close


def show_too_long(
    parent: QWidget | None,
    filename: str,
    duration_str: str,
) -> QMessageBox.StandardButton:
    """UI-SPEC: file longer than 8 hours → warning modal."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("File is longer than supported")
    box.setText(
        f"{filename} is {duration_str} long. "
        "Marmelade supports files up to 8 hours in this version."
    )
    choose = box.addButton(
        "Choose a different file", QMessageBox.ButtonRole.AcceptRole
    )
    box.setDefaultButton(choose)
    box.exec()
    return QMessageBox.StandardButton.Ok
