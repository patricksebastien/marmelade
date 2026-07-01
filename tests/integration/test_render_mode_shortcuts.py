"""Integration: number-key render-mode shortcuts (quick-260630-dqd).

Asserts THROUGH the live MainWindow seam (not an isolated helper — the
unit-tested-but-unwired history mandates a live-seam check):

* Pressing number key N selects the Nth registered render mode AND the
  "View:" combo's current index moves to match (combo is the single source
  of truth; the keystroke re-renders via the existing combo signal path).
* The number of bound shortcuts equals ``len(list(RenderMode))`` and their
  keys are "1".."N" in registry order — proving the bindings are
  registry-driven, not a hardcoded list.
* A number key past the last mode (out-of-range index) is a no-op.
* A focused QLineEdit suppresses the switch (the user is typing a note).
* The real ``QShortcut.activated`` signal — not just the handler — is wired.

Tests run under QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QLineEdit

from marmelade.audio.render_modes import RenderMode
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


@pytest.fixture
def main_window(qtbot, qapp):
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


def test_number_key_selects_corresponding_render_mode(main_window):
    """Key N selects the Nth mode through the live seam, combo synced."""
    view = main_window._waveform_view
    for i, mode in enumerate(RenderMode):
        main_window._on_render_mode_shortcut(i)
        assert view._render_mode is mode, (i, view._render_mode, mode)
        assert view.render_mode_combo.currentIndex() == i, (
            i,
            view.render_mode_combo.currentIndex(),
        )


def test_shortcut_count_matches_registry(main_window):
    """Bindings are registry-driven: one per mode, keys "1".."N" in order."""
    shortcuts = main_window._render_mode_shortcuts
    assert len(shortcuts) == len(list(RenderMode))
    for i, sc in enumerate(shortcuts):
        assert sc.key().toString() == str(i + 1), (i, sc.key().toString())


def test_out_of_range_number_key_is_noop(main_window):
    """A number key past the last mode leaves the active mode unchanged."""
    view = main_window._waveform_view
    main_window._on_render_mode_shortcut(0)  # Classic
    assert view._render_mode is RenderMode.CLASSIC
    before_idx = view.render_mode_combo.currentIndex()

    # One past the last registered mode — no IndexError, no mode change.
    main_window._on_render_mode_shortcut(len(list(RenderMode)))

    assert view._render_mode is RenderMode.CLASSIC
    assert view.render_mode_combo.currentIndex() == before_idx


def test_shortcut_bypassed_while_lineedit_focused(main_window, monkeypatch):
    """A focused QLineEdit suppresses the mode switch (user typing a note)."""
    view = main_window._waveform_view
    # Start on a non-default mode so an accidental reset would be detectable.
    main_window._on_render_mode_shortcut(1)  # DB
    assert view._render_mode is RenderMode.DB
    before_idx = view.render_mode_combo.currentIndex()

    line_edit = QLineEdit()
    line_edit.setFocus()
    QApplication.processEvents()

    # Offscreen focus is unreliable; if the platform did not grant focus to the
    # QLineEdit, force the bail condition deterministically by monkeypatching
    # QApplication.focusWidget so the test still exercises the no-op contract.
    if QApplication.focusWidget() is not line_edit:
        monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: line_edit))

    main_window._on_render_mode_shortcut(0)  # would select Classic if not bailed

    assert view._render_mode is RenderMode.DB
    assert view.render_mode_combo.currentIndex() == before_idx


def test_shortcut_signal_is_wired(main_window):
    """The real QShortcut.activated signal drives the switch (wiring proof)."""
    view = main_window._waveform_view
    main_window._on_render_mode_shortcut(0)  # Classic baseline
    assert view._render_mode is RenderMode.CLASSIC

    # Emit the second shortcut's activated signal (key "2" -> DB).
    main_window._render_mode_shortcuts[1].activated.emit()

    assert view._render_mode is RenderMode.DB
    assert view.render_mode_combo.currentIndex() == 1
