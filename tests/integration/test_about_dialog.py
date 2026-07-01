"""quick-260626-pbl — headless integration tests for AboutDialog.

Constructs the dialog under pytest-qt's offscreen ``qapp`` (forced via
``tests/conftest.py`` ``qapp_args``) and asserts on the constructed widget
state only — ``.exec()`` is never called (it would block headless).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QTextBrowser

import marmelade
from marmelade.ui.about_dialog import AboutDialog


def _all_text(dlg: AboutDialog) -> str:
    """Combine visible text from every child QLabel + QTextBrowser."""
    parts: list[str] = []
    for label in dlg.findChildren(QLabel):
        parts.append(label.text())
    for browser in dlg.findChildren(QTextBrowser):
        parts.append(browser.toPlainText())
    return "\n".join(parts)


def test_constructs_ok(qtbot, qapp) -> None:
    dlg = AboutDialog()
    qtbot.add_widget(dlg)
    assert isinstance(dlg, QDialog)
    assert dlg.isModal() is True
    assert dlg.windowTitle() == "About Marmelade"


def test_contains_expected_text(qtbot, qapp) -> None:
    dlg = AboutDialog()
    qtbot.add_widget(dlg)
    text = _all_text(dlg)
    assert "Patrick Sébastien Coulombe" in text
    assert "workinprogress.ca" in text
    assert "Marmelade" in text
    assert f"Version {marmelade.__version__}" in text
    assert "0.1.0" in text
    assert "pedalboard" in text
    assert "PySide6" in text


def test_website_link_opens_external(qtbot, qapp) -> None:
    dlg = AboutDialog()
    qtbot.add_widget(dlg)
    link_labels = [
        lbl
        for lbl in dlg.findChildren(QLabel)
        if "workinprogress.ca" in lbl.text()
    ]
    assert link_labels, "expected a QLabel containing the website link"
    assert any(lbl.openExternalLinks() for lbl in link_labels)


def test_resource_pixmap_non_null(qtbot, qapp) -> None:
    path = Path(
        str(files("marmelade.resources").joinpath("baldapprouved.png"))
    )
    assert path.exists()
    assert QPixmap(str(path)).isNull() is False
