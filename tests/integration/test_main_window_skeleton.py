"""Integration: MainWindow chrome matches UI-SPEC exactly.

Asserts:

* Window title is "Marmelade".
* The toolbar exists and has 4 QActions (Open, Zoom Fit/In/Out).
* The menu bar exposes File, View, Help.
* The LEFT dock is the Mastering panel; Keepers is standalone on the RIGHT.
* The central widget is a WaveformView with `open_button: QPushButton`
  text "Open audio file".
* MainWindow.file_open_requested is a Signal (asserted via emit/connect
  round-trip — Qt's bound Signal is not the same type as the class-level
  Signal descriptor, so we test the behavior rather than the type).
* Clicking the empty-state Open button while QFileDialog is monkey-patched
  to return ("", "") (cancel-equivalent) does NOT emit file_open_requested.

Phase 7 Plan 07-05 — additionally pins that MainWindow auto-creates the
Matchering reference library directory on first launch (D-12 — silent,
``mkdir(parents=True, exist_ok=True)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QToolBar,
)

import marmelade.paths as paths_module
import marmelade.ui.main_window as main_window_module
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from marmelade.ui.waveform_view import WaveformView


@pytest.fixture
def main_window(qtbot, qapp):
    """Build a MainWindow with the theme applied. Returns the window instance."""
    # apply_theme is idempotent enough to call once per test; we want to be
    # sure setConfigOption flags are set even in isolation.
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


def test_window_title(main_window: MainWindow) -> None:
    assert main_window.windowTitle() == "Marmelade"


def test_toolbar_has_six_actions(main_window: MainWindow) -> None:
    """Toolbar action sequence:

    Open → Zoom Fit → Zoom In → Zoom Out → Follow Playhead → Region
    Select → "View:" label → render-mode combo → A/B preview toggle
    widget → fixed gap spacer → playback timestamp QLabel.

    Function name preserved for historical traceability. The "View:"
    render-mode selector was relocated from inside WaveformView onto the
    toolbar (between Region-select and A/B), adding two QWidgetActions.
    Total action count is now 11 (7 named + View-label + combo +
    gap-spacer + timestamp label).
    """
    toolbar = main_window.findChild(QToolBar)
    assert toolbar is not None, "Toolbar missing"
    assert len(toolbar.actions()) == 11, [a.text() for a in toolbar.actions()]


def test_toolbar_action_labels(main_window: MainWindow) -> None:
    """Toolbar action text matches UI-SPEC §Copywriting tooltips.

    The trailing empty labels are QWidgetActions wrapping (View: label,
    render-mode combo, A/B toggle, gap spacer, timestamp QLabel) — a
    QWidgetAction's text() is always "". The render-mode combo sits
    between Region-select and the A/B toggle.
    """
    toolbar = main_window.findChild(QToolBar)
    labels = [a.text() for a in toolbar.actions()]
    assert labels == [
        "Open audio file",
        "Fit waveform to view",
        "Zoom in",
        "Zoom out",
        "Follow playhead",
        "Region select mode",
        "",  # "View:" QLabel
        "",  # render-mode QComboBox (relocated from WaveformView)
        "",  # ABToggleWidget
        "",  # gap spacer (24px fixed)
        "",  # timestamp QLabel
    ], labels


def test_render_mode_combo_on_toolbar_between_region_select_and_ab(
    main_window: MainWindow,
) -> None:
    """The "View:" render-mode combo is placed on the toolbar, between the
    Region-select action and the A/B preview toggle (user request).

    It remains the SAME object WaveformView owns as ``render_mode_combo`` —
    MainWindow only reparents it onto the toolbar via addWidget.
    """
    toolbar = main_window.findChild(QToolBar)
    combo = main_window._waveform_view.render_mode_combo
    assert isinstance(combo, QComboBox)

    actions = toolbar.actions()
    combo_idx = ab_idx = region_idx = None
    for i, a in enumerate(actions):
        w = toolbar.widgetForAction(a)
        if w is combo:
            combo_idx = i
        elif w is main_window._ab_toggle:
            ab_idx = i
        if a is main_window._tb_region_select:
            region_idx = i

    assert combo_idx is not None, "render-mode combo is not on the toolbar"
    assert region_idx is not None and ab_idx is not None
    # Strictly between Region-select and A/B preview.
    assert region_idx < combo_idx < ab_idx


def test_menu_bar_top_level(main_window: MainWindow) -> None:
    """Plan 03-02 inserts an Edit menu between File and View (Qt convention)."""
    mb = main_window.menuBar()
    top_level = [a.text() for a in mb.actions()]
    assert top_level == ["File", "Edit", "View", "Help"], top_level


def test_left_dock_is_mastering(main_window: MainWindow) -> None:
    """quick-260621-dt4 — the Heatmaps panel was retired. The Mastering
    dock is now the LEFT dock; Keepers is standalone on the RIGHT with no
    tab sibling.
    """
    mastering = main_window._mastering_dock
    assert mastering is not None, "Mastering dock missing"
    assert mastering.windowTitle() == "Mastering", mastering.windowTitle()
    assert (
        main_window.dockWidgetArea(mastering)
        == Qt.DockWidgetArea.LeftDockWidgetArea
    )

    keepers = main_window._dock_keepers
    assert (
        main_window.dockWidgetArea(keepers)
        == Qt.DockWidgetArea.RightDockWidgetArea
    )
    assert main_window.tabifiedDockWidgets(keepers) == [], (
        "Keepers must be standalone (no tabify sibling)"
    )


def test_central_widget_is_waveform_view(main_window: MainWindow) -> None:
    """The central widget is the plain WaveformView (pre-Phase-9 restore).

    The WaveformView fills the central area and owns the empty-state Open
    button + click-to-seek + region-select machinery. It is the same object
    exposed as ``_waveform_view``.
    """
    central = main_window.centralWidget()
    assert isinstance(central, WaveformView), type(central)
    assert central is main_window._waveform_view
    wave = main_window._waveform_view
    assert isinstance(wave.open_button, QPushButton)
    assert wave.open_button.text() == "Open audio file"


def test_file_open_requested_is_a_signal(
    main_window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-trip emit/connect proves file_open_requested is a real Signal.

    Plan 03 wires file_open_requested → _open_file, which would probe the
    fake path "/tmp/probe.wav" and pop a modal QMessageBox on failure
    (blocking the test). Patch QMessageBox.exec to a no-op so the emit
    round-trip remains the assertion under test.
    """
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    received: list[str] = []
    main_window.file_open_requested.connect(received.append)
    main_window.file_open_requested.emit("/tmp/probe.wav")
    assert received == ["/tmp/probe.wav"]


def test_empty_state_button_cancel_is_silent(
    main_window: MainWindow, qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clicking the empty-state Open button → cancel dialog → no signal fired."""
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: ("", "")),
    )
    central = main_window._waveform_view
    assert isinstance(central, WaveformView)

    with qtbot.waitSignal(
        main_window.file_open_requested, timeout=500, raising=False
    ) as blocker:
        QTest.mouseClick(central.open_button, Qt.MouseButton.LeftButton)

    # Cancel must be silent — no emission.
    assert blocker.signal_triggered is False, "file_open_requested fired on cancel"


def test_empty_state_button_path_emits_signal(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Picking a path via the empty-state button emits file_open_requested.

    Plan 03 wires ``file_open_requested`` → ``_open_file``, which probes the
    file. The fake path doesn't exist, so the probe would raise
    FileNotFoundError and a modal ``QMessageBox`` would block the test
    event loop. We patch ``QMessageBox.exec`` to a no-op so the open flow
    can run end-to-end without UI interaction; the assertion remains the
    same — the signal must have fired with ``fake_path``.
    """
    fake_path = str(tmp_path / "fake.wav")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (fake_path, "Audio files (*.wav *.flac *.mp3)")),
    )
    # Plan 03: any dialog raised by the open flow must not block the test.
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)

    central = main_window._waveform_view
    assert isinstance(central, WaveformView)

    with qtbot.waitSignal(
        main_window.file_open_requested, timeout=2000, raising=True
    ) as blocker:
        QTest.mouseClick(central.open_button, Qt.MouseButton.LeftButton)

    assert blocker.args == [fake_path], blocker.args


# ---------------------------------------------------------------------------
# Phase 7 Plan 07-05 — Matchering reference dir auto-create.
# ---------------------------------------------------------------------------


def test_main_window_creates_matchering_reference_dir_on_init(
    qtbot, qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MainWindow.__init__ auto-creates ``matchering_reference_dir()``.

    D-12 (CONTEXT.md domain item 12) — the references library lives at
    ``~/Music/Marmelade/References/``. We don't want to require the
    user to ``mkdir`` it manually; MainWindow does it silently at
    startup. The call is wrapped in ``try/except OSError`` so a
    read-only ``$HOME`` cannot crash app startup (defensive).

    Test strategy: monkeypatch :func:`matchering_reference_dir` (both
    in ``marmelade.paths`` AND in ``marmelade.ui.main_window`` —
    the latter is needed because the MainWindow module imports the
    function by name) to return a tmp directory that does NOT exist
    pre-construction. Construct MainWindow; assert the dir exists
    post-construction.
    """
    ref_dir = tmp_path / "fake_home" / "Music" / "Marmelade" / "References"
    assert not ref_dir.exists(), "ref_dir must NOT exist pre-init"

    monkeypatch.setattr(paths_module, "matchering_reference_dir", lambda: ref_dir)
    monkeypatch.setattr(
        main_window_module,
        "matchering_reference_dir",
        lambda: ref_dir,
        raising=False,
    )

    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    try:
        assert ref_dir.exists(), (
            f"MainWindow.__init__ failed to create the Matchering reference "
            f"dir at {ref_dir} (D-12 — Plan 07-05)"
        )
        assert ref_dir.is_dir(), f"ref_dir is not a directory: {ref_dir}"
    finally:
        window.close()
