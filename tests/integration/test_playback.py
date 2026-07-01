"""Plan 02-05 Task 2 — MainWindow playback wiring integration tests.

Seventeen pins covering:

* Toolbar Follow-Playhead action (5th, after Open/Fit/In/Out; checkable;
  default ON).
* Spacebar QShortcut at Qt.ShortcutContext.ApplicationShortcut (D-14b).
* Spacebar starts playback when a file is loaded; pauses when already
  playing; no-op when no file.
* 30 Hz QTimer interval (33 ms); updates the playhead InfiniteLine value
  on tick; stops when playback ends.
* Follow-Playhead OFF → view does not auto-pan; ON → view auto-pages.
* Click-to-seek calls engine.seek with the right value.
* Per-lane playhead instances (PyQtGraph QGraphicsItem can only belong to
  one scene — W1 constraint).
* Opening a new file resets the playhead to 0 and stops playback.
* Audio-unavailable graceful degradation (toolbar disabled, no crash).
* PlaybackError from engine.play() caught and surfaced via QMessageBox.

sounddevice is mocked at the engine API boundary (patches
``marmelade.audio.playback.sd.OutputStream``) so the tests don't depend
on a real audio device; we drive the engine's _is_playing flag + position
counter manually where needed.

Tests run under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from marmelade.audio.playback import PlaybackError
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def main_window(qtbot, qapp, tmp_cache_dir: Path):
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    p = tmp_path / "playback_fixture.wav"
    make_sine(p, freq_hz=1000.0, amp=0.5, duration_s=2.0, sample_rate=44100, channels=1)
    return p


@pytest.fixture
def sine_wav_b(tmp_path: Path) -> Path:
    p = tmp_path / "playback_fixture_b.wav"
    make_sine(p, freq_hz=2000.0, amp=0.5, duration_s=2.0, sample_rate=44100, channels=1)
    return p


@pytest.fixture
def mocked_sounddevice(monkeypatch):
    """Patch sd.OutputStream + force _SOUNDDEVICE_AVAILABLE True.

    Returns (ctor_mock, stream_mock) so individual tests can assert on the
    stream's start/stop/close call counts.
    """
    fake_stream = MagicMock()
    fake_ctor = MagicMock(return_value=fake_stream)
    monkeypatch.setattr("marmelade.audio.playback.sd.OutputStream", fake_ctor)
    monkeypatch.setattr("marmelade.audio.playback._SOUNDDEVICE_AVAILABLE", True)
    return fake_ctor, fake_stream


def _open(main_window: MainWindow, monkeypatch, path: Path) -> None:
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(path), "")),
    )
    main_window._action_open_file()


# =========================================================================
# Test 7 — Toolbar has Follow Playhead action at position 5
# =========================================================================
def test_toolbar_has_follow_playhead_action_5th(main_window: MainWindow) -> None:
    """Toolbar order: Open, Zoom Fit, Zoom In, Zoom Out, Follow Playhead, …

    The 5th action (Follow Playhead) is checkable AND checked by default
    (CONTEXT discretion — DAW convention). Plan 03-02 added a 6th action
    (Region Select) after Follow Playhead; Phase 7 Plan 07-04 inserts
    the A/B toolbar toggle widget as the 7th action. This test still
    pins the 5th slot. quick-260621-gfq removed the trailing Normalize
    spinbox + Normalize action; with the spacer gap + time-label widgets
    the total action count is now 9.
    """
    actions = main_window._toolbar.actions()
    assert len(actions) == 9, f"expected 9 toolbar actions; got {len(actions)}"
    assert actions[0] is main_window._tb_open
    assert actions[1] is main_window._tb_zoom_fit
    assert actions[2] is main_window._tb_zoom_in
    assert actions[3] is main_window._tb_zoom_out
    assert actions[4] is main_window._tb_follow_playhead
    assert main_window._tb_follow_playhead.isCheckable()
    # quick-260629 — Follow playhead now defaults OFF (was ON).
    assert not main_window._tb_follow_playhead.isChecked()


# =========================================================================
# Test 8 — Follow-Playhead action toggles state on click
# =========================================================================
def test_follow_playhead_action_toggles_state(main_window: MainWindow) -> None:
    """Programmatically trigger the action; isChecked() flips."""
    initial = main_window._tb_follow_playhead.isChecked()
    main_window._tb_follow_playhead.trigger()
    assert main_window._tb_follow_playhead.isChecked() != initial
    main_window._tb_follow_playhead.trigger()
    assert main_window._tb_follow_playhead.isChecked() == initial


# =========================================================================
# Test 9 — Spacebar QShortcut context is ApplicationShortcut (locked D-14b)
# =========================================================================
def test_spacebar_shortcut_is_application_context(main_window: MainWindow) -> None:
    """The shortcut MUST use Qt.ShortcutContext.ApplicationShortcut per locked D-14b.

    Phase 2 has no text-input widgets so ApplicationShortcut and WindowShortcut
    behave identically; the literal is locked for Phase 3+ when a text widget
    appears and the override hook is added.
    """
    ctx = main_window._shortcut_play_pause.context()
    assert ctx == Qt.ShortcutContext.ApplicationShortcut


# =========================================================================
# Test 10 — Spacebar starts playback when a file is loaded
# =========================================================================
def test_spacebar_starts_playback_when_file_loaded(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """Open a file, activate the spacebar shortcut, assert engine.play was called."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    # Spy on engine.play.
    monkeypatch.setattr(main_window._playback_engine, "play", MagicMock())
    main_window._shortcut_play_pause.activated.emit()
    assert main_window._playback_engine.play.called
    call = main_window._playback_engine.play.call_args
    # First positional arg is the path string.
    assert call.args[0] == str(sine_wav)
    # start_seconds defaults to 0.0 (playhead starts at 0).
    assert call.kwargs.get("start_seconds", 0.0) == pytest.approx(0.0)


# =========================================================================
# Test 11 — Spacebar pauses when currently playing
# =========================================================================
def test_spacebar_pauses_when_playing(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """Set engine to is_playing=True manually; activate spacebar; assert pause called."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    # Force the engine into a "playing" state without actually calling play().
    main_window._playback_engine._is_playing = True
    monkeypatch.setattr(main_window._playback_engine, "pause", MagicMock())
    monkeypatch.setattr(main_window._playback_engine, "play", MagicMock())
    main_window._shortcut_play_pause.activated.emit()
    assert main_window._playback_engine.pause.called
    assert not main_window._playback_engine.play.called


# =========================================================================
# Test 12 — Spacebar with no file is a graceful no-op
# =========================================================================
def test_spacebar_with_no_file_does_nothing(
    main_window: MainWindow, mocked_sounddevice, monkeypatch
) -> None:
    """No file open; activating the spacebar must NOT call engine.play()."""
    monkeypatch.setattr(main_window._playback_engine, "play", MagicMock())
    main_window._shortcut_play_pause.activated.emit()
    assert not main_window._playback_engine.play.called


# =========================================================================
# Test 13 — Playback start starts the 30 Hz QTimer
# =========================================================================
def test_playback_starts_30hz_qtimer(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """After _action_toggle_playback() with a file open, QTimer is active at 33 ms."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    main_window._action_toggle_playback()
    assert main_window._playback_timer.isActive() is True
    assert main_window._playback_timer.interval() == 33


# =========================================================================
# Test 14 — QTimer updates the waveform playhead position
# =========================================================================
def test_qtimer_updates_playhead_value(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """Mock engine.position_seconds to return ascending values; assert playhead follows.

    Drive _on_playback_tick directly with the engine's is_playing flag forced
    True. The waveform's playhead InfiniteLine.value() should reflect the
    latest mocked position.
    """
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    main_window._playback_engine._is_playing = True
    values = iter([0.1, 0.2, 0.3, 0.4, 0.5])
    monkeypatch.setattr(
        type(main_window._playback_engine),
        "position_seconds",
        property(lambda self: next(values)),
    )
    for _ in range(5):
        main_window._on_playback_tick()
    # The last mocked value is what InfiniteLine.value() should report.
    assert main_window._waveform_view.playhead.value() == pytest.approx(0.5)


# =========================================================================
# Test 15 — QTimer stops when playback ends
# =========================================================================
def test_qtimer_stops_when_playback_ends(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """When engine.is_playing flips False, the QTimer is stopped."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    main_window._action_toggle_playback()
    assert main_window._playback_timer.isActive()
    # Force engine state to "not playing".
    main_window._playback_engine._is_playing = False
    main_window._on_playback_tick()
    assert main_window._playback_timer.isActive() is False


# =========================================================================
# Test 16 — Follow-Playhead OFF does not pan view
# =========================================================================
def test_follow_playhead_off_does_not_pan_view(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """With Follow-Playhead unchecked, advancing position past visible range
    must NOT change the view's x-range."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    main_window._tb_follow_playhead.setChecked(False)
    vb = main_window._waveform_view.waveform_plot.getViewBox()
    # Capture range, then zoom in so the visible range is < duration.
    main_window._waveform_view.waveform_plot.setXRange(0.0, 0.5, padding=0)
    QApplication.processEvents()
    before_range = vb.viewRange()[0]
    # Position past the visible range.
    monkeypatch.setattr(
        type(main_window._playback_engine),
        "position_seconds",
        property(lambda self: 1.5),
    )
    main_window._playback_engine._is_playing = True
    main_window._on_playback_tick()
    after_range = vb.viewRange()[0]
    assert before_range == after_range, (
        f"Follow-Playhead OFF: view must not pan; before={before_range}, after={after_range}"
    )


# =========================================================================
# Test 17 — Follow-Playhead ON pans view via page-flip-on-edge
# =========================================================================
def test_follow_playhead_on_pans_view_via_page_flip(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """With Follow-Playhead checked, advancing past visible range pages the view."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    main_window._tb_follow_playhead.setChecked(True)
    vb = main_window._waveform_view.waveform_plot.getViewBox()
    main_window._waveform_view.waveform_plot.setXRange(0.0, 0.5, padding=0)
    QApplication.processEvents()
    before_range = vb.viewRange()[0]
    # Position WELL past the visible range.
    monkeypatch.setattr(
        type(main_window._playback_engine),
        "position_seconds",
        property(lambda self: 1.5),
    )
    main_window._playback_engine._is_playing = True
    main_window._on_playback_tick()
    after_range = vb.viewRange()[0]
    assert after_range != before_range, (
        f"Follow-Playhead ON: view must page; before={before_range}, after={after_range}"
    )
    # Playhead position should now be inside the new range.
    assert after_range[0] <= 1.5 <= after_range[1]


# =========================================================================
# Test 18 — Click-to-seek calls engine.seek with the right value
# =========================================================================
def test_click_to_seek_calls_engine_seek(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """A click on the waveform within 4 px routes through MainWindow._on_seek_requested
    and calls engine.seek(seconds)."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    seek_mock = MagicMock()
    monkeypatch.setattr(main_window._playback_engine, "seek", seek_mock)
    # Synthesize a click within the threshold.
    viewport = main_window._waveform_view.graphics_layout.viewport()
    press_pos = QPointF(200.0, 30.0)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        press_pos,
        viewport.mapToGlobal(QPoint(200, 30)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, press)
    release_pos = QPointF(201.0, 30.0)
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        release_pos,
        viewport.mapToGlobal(QPoint(201, 30)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, release)
    QApplication.processEvents()
    assert seek_mock.called
    seek_arg = seek_mock.call_args.args[0]
    assert isinstance(seek_arg, float)
    assert seek_arg >= 0.0


# =========================================================================
# Test 20 — Playhead is not draggable (D-14: click-to-seek, not drag)
# =========================================================================
def test_playhead_movable_is_false(main_window: MainWindow) -> None:
    """The InfiniteLine.movable attribute is False — D-14."""
    ph = main_window._waveform_view.playhead
    assert ph.movable is False


# =========================================================================
# Test 21 — Opening a new file resets the playhead to 0
# =========================================================================
def test_open_new_file_resets_playhead_to_0(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    sine_wav_b: Path,
    mocked_sounddevice,
) -> None:
    """After playing → moving playhead → opening a different file, playhead is at 0
    and the engine is stopped."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    # Move the playhead to 5 s manually.
    main_window._waveform_view.playhead.setValue(5.0)
    assert main_window._waveform_view.playhead.value() == pytest.approx(5.0)
    # Spy on engine.stop so we know it was called by the cancel preamble.
    stop_mock = MagicMock()
    monkeypatch.setattr(main_window._playback_engine, "stop", stop_mock)
    # Open the second file.
    _open(main_window, monkeypatch, sine_wav_b)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()
    assert main_window._waveform_view.playhead.value() == pytest.approx(0.0)
    assert stop_mock.called


# =========================================================================
# Test 22 — Audio backend unavailable degrades gracefully
# =========================================================================
def test_audio_unavailable_disables_toolbar_action_gracefully(
    qtbot, qapp, tmp_cache_dir: Path, monkeypatch, sine_wav: Path
) -> None:
    """Patch _SOUNDDEVICE_AVAILABLE=False at the module level BEFORE constructing
    MainWindow; assert the app constructs without crashing AND the playback
    toolbar action / spacebar shortcut are disabled AND the status bar shows
    the unavailable message.
    """
    monkeypatch.setattr("marmelade.audio.playback._SOUNDDEVICE_AVAILABLE", False)
    monkeypatch.setattr(
        "marmelade.audio.playback._SOUNDDEVICE_IMPORT_ERROR",
        OSError("simulated libportaudio2 missing"),
    )
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    # Toolbar action disabled.
    assert window._tb_follow_playhead.isEnabled() is False
    # Spacebar shortcut disabled.
    assert window._shortcut_play_pause.isEnabled() is False
    # Status bar shows a message (any non-empty message acknowledging the issue).
    status_msg = window.statusBar().currentMessage()
    assert "audio" in status_msg.lower() or "playback" in status_msg.lower()
    # Sanity: the rest of the app still works — open a file should still render.
    _open(window, monkeypatch, sine_wav)
    qtbot.waitSignal(window.render_complete, timeout=15000).wait()


# =========================================================================
# Test 23 — engine.play() raising PlaybackError is caught and surfaced
# =========================================================================
def test_engine_play_error_caught_and_surfaced(
    main_window: MainWindow,
    qtbot,
    monkeypatch,
    sine_wav: Path,
    mocked_sounddevice,
) -> None:
    """If engine.play raises PlaybackError, the app catches it and shows a
    QMessageBox warning (we patch QMessageBox.warning to capture the call)."""
    _open(main_window, monkeypatch, sine_wav)
    qtbot.waitSignal(main_window.render_complete, timeout=15000).wait()

    def _raise(*args, **kwargs):
        raise PlaybackError("simulated playback init failure")

    monkeypatch.setattr(main_window._playback_engine, "play", _raise)
    warning_mock = MagicMock()
    monkeypatch.setattr(
        "marmelade.ui.main_window.QMessageBox.warning",
        warning_mock,
    )
    # Trigger play via the spacebar shortcut.
    main_window._shortcut_play_pause.activated.emit()
    # QMessageBox.warning should have been called (the app must not have crashed).
    assert warning_mock.called
