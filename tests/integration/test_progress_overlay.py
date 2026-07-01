"""Integration: ProgressOverlay layout + UI-SPEC copy.

Asserts (Plan 03 Task 2):

* Heading reads "Preparing waveform" (UI-SPEC §Copywriting > Loading).
* Cancel button reads "Stop building proxy" (UI-SPEC §Copywriting > Loading).
* ``set_progress(int)`` advances the progress bar.
* ``set_body(str)`` updates the body label text.
* ``resize_to_parent()`` keeps the overlay covering its parent widget.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from marmelade.ui import theme
from marmelade.ui.progress_overlay import ProgressOverlay


@pytest.fixture
def parent_widget(qtbot, qapp):
    theme.apply_theme(QApplication.instance())
    parent = QWidget()
    parent.resize(640, 480)
    qtbot.addWidget(parent)
    parent.show()
    return parent


def test_overlay_heading_is_preparing_waveform(parent_widget: QWidget) -> None:
    overlay = ProgressOverlay(parent_widget)
    # Heading text exists somewhere among the labels.
    from PySide6.QtWidgets import QLabel

    labels = overlay.findChildren(QLabel)
    texts = [lbl.text() for lbl in labels]
    assert any("Preparing waveform" in t for t in texts), texts


def test_overlay_cancel_button_text(parent_widget: QWidget) -> None:
    overlay = ProgressOverlay(parent_widget)
    assert overlay.cancel_button.text() == "Stop building proxy"


def test_overlay_set_progress_updates_bar(parent_widget: QWidget) -> None:
    overlay = ProgressOverlay(parent_widget)
    overlay.set_progress(42)
    # Find the progress bar.
    from PySide6.QtWidgets import QProgressBar

    bars = overlay.findChildren(QProgressBar)
    assert len(bars) >= 1
    assert bars[0].value() == 42


def test_overlay_set_body_updates_label(parent_widget: QWidget) -> None:
    overlay = ProgressOverlay(parent_widget)
    msg = "test.wav · 0:30 · first open — building a downsampled proxy."
    overlay.set_body(msg)
    from PySide6.QtWidgets import QLabel

    labels = overlay.findChildren(QLabel)
    texts = [lbl.text() for lbl in labels]
    assert any(msg in t for t in texts), texts


def test_overlay_resize_to_parent_matches_parent_geometry(
    parent_widget: QWidget,
) -> None:
    overlay = ProgressOverlay(parent_widget)
    parent_widget.resize(800, 600)
    overlay.resize_to_parent()
    assert overlay.size().width() == 800
    assert overlay.size().height() == 600
