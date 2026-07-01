"""Plan 02-05 Task 2 — Click-to-seek event-filter integration tests.

Six pins covering the WaveformView's click-vs-drag disambiguation at the
SEEK_THRESHOLD_PX=4 pixel boundary:

* A press/release pair within 4 px emits ``seek_requested(float)`` carrying
  the data-space x coordinate in seconds.
* Exactly 4 px (the threshold) is inclusive — still emits.
* > 4 px is a drag and does NOT emit (PyQtGraph's ViewBox pan owns it).
* Far drags don't emit (defensive against off-by-one).
* The signal payload is in seconds (not pixels), mapped via
  ``ViewBox.mapSceneToView``.
* A release WITHOUT a prior press is ignored (defensive against stale state).

Tests run under ``QT_QPA_PLATFORM=offscreen``; QMouseEvents are dispatched
synchronously via QApplication.sendEvent so the eventFilter sees them with
zero queueing.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QHBoxLayout, QWidget

from marmelade.ui import theme
from marmelade.ui.waveform_view import SEEK_THRESHOLD_PX, WaveformView


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def waveform_view(qtbot, qapp):
    """Build a WaveformView with theme applied + a 10 s synthetic render.

    The waveform is rendered with 4000 int16 pairs (the pass-through boundary)
    so ``_duration_s == 10.0`` and the data-space x maps to [0, 10] seconds.
    The viewport is resized to 800x200 so x-pixel coordinates map predictably.
    """
    theme.apply_theme(QApplication.instance())
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(800, 200)
    # Render a known-duration waveform so mapSceneToView returns predictable seconds.
    # duration_s = (n * spp) / sr. For 4000 pairs, spp=1102, sr=44100:
    # duration ≈ 99.9 s. Use a simpler shape: 4000 pairs with spp s.t. duration = 10s.
    # length=4000, sr=44100 → spp = 10s * sr / length = 110.25 → use spp=110 → ~9.98s.
    n = 4000
    t = np.arange(n) * (2 * np.pi / 64)
    base = (np.sin(t) * 16000).astype(np.int16)
    pairs = np.empty((n, 2), dtype=np.int16)
    pairs[:, 0] = -base
    pairs[:, 1] = base
    # sr=44100, spp=110.25 not allowed (int) — use 441 to get 4000*441/44100 = 40s.
    # That's fine — we only need a known finite duration.
    view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=441)
    # Process events so the viewport reflects the rendered range.
    QApplication.processEvents()
    return view


def _send_press(view: WaveformView, x: int, y: int = 30) -> None:
    """Dispatch a synthetic left-button MouseButtonPress to the viewport."""
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        viewport.mapToGlobal(QPoint(x, y)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, press)


def _send_release(view: WaveformView, x: int, y: int = 30) -> None:
    """Dispatch a synthetic left-button MouseButtonRelease to the viewport."""
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        pos,
        viewport.mapToGlobal(QPoint(x, y)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, release)


# =========================================================================
# Test 1 — Click under the 4 px threshold emits seek_requested
# =========================================================================
def test_click_under_threshold_emits_seek(waveform_view: WaveformView, qtbot) -> None:
    """A 2 px press/release (delta ≤ 4) emits seek_requested with a positive seconds value."""
    with qtbot.waitSignal(waveform_view.seek_requested, timeout=1000) as blocker:
        _send_press(waveform_view, x=200)
        _send_release(waveform_view, x=202)
    # Payload is a float — the data-space x in seconds. Should be positive
    # (we're well inside the viewport).
    assert isinstance(blocker.args[0], float)
    assert blocker.args[0] >= 0.0


# =========================================================================
# Test 2 — Click at exactly the threshold (4 px) still emits
# =========================================================================
def test_click_at_exactly_4_px_emits_seek(waveform_view: WaveformView, qtbot) -> None:
    """Press at x=200, release at x=204 (delta=4 == SEEK_THRESHOLD_PX). Inclusive."""
    assert SEEK_THRESHOLD_PX == 4
    with qtbot.waitSignal(waveform_view.seek_requested, timeout=1000):
        _send_press(waveform_view, x=200)
        _send_release(waveform_view, x=204)


# =========================================================================
# Test 3 — Drag over threshold does NOT emit
# =========================================================================
def test_drag_over_threshold_does_not_seek(
    waveform_view: WaveformView, qtbot
) -> None:
    """5 px delta > SEEK_THRESHOLD_PX → no seek_requested fires."""
    # assertNotEmitted via qtbot.waitSignal(..., raising=False).wait(short timeout).
    with qtbot.assertNotEmitted(waveform_view.seek_requested, wait=200):
        _send_press(waveform_view, x=200)
        _send_release(waveform_view, x=205)


# =========================================================================
# Test 4 — Far drag does NOT emit
# =========================================================================
def test_drag_far_does_not_seek(waveform_view: WaveformView, qtbot) -> None:
    """200 px drag clearly belongs to PyQtGraph's pan handler; no seek fires."""
    with qtbot.assertNotEmitted(waveform_view.seek_requested, wait=200):
        _send_press(waveform_view, x=200)
        _send_release(waveform_view, x=400)


# =========================================================================
# Test 5 — Signal payload is in seconds (data-space mapping)
# =========================================================================
def test_seek_signal_payload_in_seconds(waveform_view: WaveformView, qtbot) -> None:
    """Click at the viewport's horizontal center → emitted seconds ≈ duration/2.

    The fixture renders a ~40 s waveform (length=4000, spp=441, sr=44100).
    Clicking at the viewport center should emit ~20 s ± a generous tolerance
    for the viewport/data mapping precision.
    """
    duration_s = waveform_view._duration_s
    assert duration_s > 0.0
    viewport = waveform_view.graphics_layout.viewport()
    # Use the actual viewport width to find the visual center.
    center_x = viewport.width() // 2
    with qtbot.waitSignal(waveform_view.seek_requested, timeout=1000) as blocker:
        _send_press(waveform_view, x=center_x)
        _send_release(waveform_view, x=center_x + 1)  # 1 px delta → emits
    emitted_seconds = blocker.args[0]
    # Wide tolerance — the exact mapping depends on the left-axis width / viewport
    # margins, but ~half the duration is the expected ballpark.
    assert 0.0 <= emitted_seconds <= duration_s, (
        f"emitted seconds {emitted_seconds} out of bounds [0, {duration_s}]"
    )
    # Should be roughly at the middle (within ±25% of duration).
    assert abs(emitted_seconds - duration_s / 2) < duration_s * 0.25


# =========================================================================
# Test 6 — Release without prior press is ignored (no signal)
# =========================================================================
def test_release_without_press_is_ignored(
    waveform_view: WaveformView, qtbot
) -> None:
    """Defensive: a release event when _mouse_down_x_px is None must not emit.

    Manually clear the press-coord bookkeeping (simulating a state where the
    press was suppressed by another handler) and dispatch a bare release —
    seek_requested must NOT fire because we have no reference point for the
    click-vs-drag delta.
    """
    waveform_view._mouse_down_x_px = None
    with qtbot.assertNotEmitted(waveform_view.seek_requested, wait=200):
        _send_release(waveform_view, x=200)


# =========================================================================
# Test 7 — Click-to-seek under chrome offset (regression pin)
# =========================================================================
# Pins the viewport→scene→data mapping when the WaveformView is embedded
# inside a parent with a horizontal offset (200 px left spacer simulating
# the LayersSidebar dock). Guards against reintroducing
# QMouseEvent.scenePosition() (window-coord) in place of
# graphics_layout.mapToScene (scene-coord).


def _send_press_with_scene_pos(
    view: WaveformView, x: int, scene_pos: QPointF, y: int = 30
) -> None:
    """Dispatch a press event with an explicit ``scenePos`` (7-arg ctor).

    The shared ``_send_press`` uses the 5-arg ``QMouseEvent`` ctor, which sets
    ``scenePosition() == position()``. In real OS delivery via the widget
    tree, Qt fills ``scenePosition()`` with **window-relative** coords — so
    to faithfully reproduce the production bug under offscreen tests we must
    supply that scene position explicitly. Without this, the buggy code path
    (``mouse_event.scenePosition()``) reads the viewport-local value and the
    regression test cannot distinguish the two implementations.
    """
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    global_pos = viewport.mapToGlobal(QPoint(x, y)).toPointF()
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        scene_pos,
        global_pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, press)


def _send_release_with_scene_pos(
    view: WaveformView, x: int, scene_pos: QPointF, y: int = 30
) -> None:
    """Release-event sibling of ``_send_press_with_scene_pos`` (see docstring there)."""
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    global_pos = viewport.mapToGlobal(QPoint(x, y)).toPointF()
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        pos,
        scene_pos,
        global_pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(viewport, release)


def test_click_to_seek_under_chrome_offset(qtbot, qapp) -> None:
    """Click at viewport center → emitted seconds ≈ duration/2 even with chrome.

    Reproduces the production widget hierarchy: a parent QWidget with a
    QHBoxLayout containing a 200 px fixed-width left spacer (the LayersSidebar
    stand-in) and the WaveformView on the right. The chrome offset must NOT
    leak into the emitted seconds — that's the whole point of the
    ``graphics_layout.mapToScene`` mapping path.

    Mouse events are constructed with an explicit ``scenePos`` set to the
    **window-relative** position (what Qt would supply on real OS delivery
    via ``viewport.eventFilter``). With the pre-fix code
    (``mouse_event.scenePosition()`` on the release path), that window-local
    coordinate is fed straight into ``ViewBox.mapSceneToView``, producing a
    seconds value shifted by ~the chrome offset in scene units — empirically
    ~24% of duration off-center on this 1000-px-wide / 200-px-spacer setup
    (vs. ~3% intrinsic offset under the fix, from pyqtgraph's left-axis
    chrome). The ±5% tolerance cleanly separates the two regimes.
    """
    theme.apply_theme(QApplication.instance())

    # Build the parent hierarchy inline so the chrome offset exists at the
    # Qt level when events are dispatched. We deliberately do NOT reuse the
    # ``waveform_view`` fixture — it builds a standalone widget with no chrome.
    parent = QWidget()
    layout = QHBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    spacer = QWidget()
    spacer.setFixedWidth(200)  # simulate LayersSidebar dock width
    spacer.setMinimumHeight(200)
    layout.addWidget(spacer)

    view = WaveformView()
    layout.addWidget(view)

    qtbot.addWidget(parent)
    parent.resize(1000, 200)
    parent.show()
    QApplication.processEvents()

    # Render the same synthetic proxy shape as the shared fixture
    # (length=4000 int16 pairs, sr=44100, spp=441 → _duration_s ≈ 40.0).
    n = 4000
    t = np.arange(n) * (2 * np.pi / 64)
    base = (np.sin(t) * 16000).astype(np.int16)
    pairs = np.empty((n, 2), dtype=np.int16)
    pairs[:, 0] = -base
    pairs[:, 1] = base
    view.render_proxy(pairs, sample_rate=44100, samples_per_pixel=441)
    QApplication.processEvents()

    duration_s = view._duration_s
    assert duration_s > 0.0, "render_proxy did not set _duration_s"

    # Read the ACTUAL rendered viewport width — depends on layout, parent
    # width, and chrome offset. center_x is in viewport-local coordinates.
    viewport = view.graphics_layout.viewport()
    center_x = viewport.width() // 2
    click_y = 30

    # Faithful Qt-delivery simulation: scenePosition is the click's position
    # relative to the **top-level window**, NOT the viewport. With the 200 px
    # spacer, viewport-local (center_x, y) maps to window-local (center_x +
    # 200, y) — this is what the buggy code path would consume.
    top = view.window()
    window_pt_press = viewport.mapTo(top, QPoint(center_x, click_y))
    window_pt_release = viewport.mapTo(top, QPoint(center_x + 1, click_y))
    scene_pos_press = QPointF(float(window_pt_press.x()), float(window_pt_press.y()))
    scene_pos_release = QPointF(
        float(window_pt_release.x()), float(window_pt_release.y())
    )

    with qtbot.waitSignal(view.seek_requested, timeout=1000) as blocker:
        _send_press_with_scene_pos(view, x=center_x, scene_pos=scene_pos_press)
        _send_release_with_scene_pos(
            view, x=center_x + 1, scene_pos=scene_pos_release
        )  # 1 px delta → emits

    emitted_seconds = blocker.args[0]
    assert isinstance(emitted_seconds, float)

    expected = duration_s / 2.0
    # ±5% of duration. Tight enough to fail under the bug (where the emitted
    # seconds are off by ~24%), loose enough to accommodate the intrinsic
    # ~3% offset from pyqtgraph's left-axis chrome eating viewport pixels.
    tolerance = 0.05 * duration_s
    assert abs(emitted_seconds - expected) <= tolerance, (
        f"emitted {emitted_seconds:.3f}s, expected ~{expected:.3f}s "
        f"±{tolerance:.3f}s (chrome offset leaked through?)"
    )


# =========================================================================
# Test 8 — Bug #1 regression: click-before-spacebar updates engine position
# =========================================================================
# Pins the end-to-end behaviour the user is promised by D-14 (click-to-seek)
# + D-15 (lazy compute) acting together: after opening a file, a pre-play
# click on the waveform sets ``engine.position_seconds`` to the seek target
# (NOT zero) so the next ``play(start_seconds=engine.position_seconds)``
# starts from the clicked position. Before the fix (``prime()`` not wired
# into ``_open_file``), ``_sample_rate`` stayed 0 until first play, so
# ``seek()`` silently zeroed its target and spacebar started from 0.


@pytest.fixture
def main_window_for_seek(qtbot, qapp, tmp_cache_dir):
    """Minimal MainWindow for the bug #1 regression test.

    Local to this test module — we deliberately do NOT refactor the shared
    ``waveform_view`` fixture above. ``tmp_cache_dir`` keeps proxy cache
    writes inside the per-test tmp tree so a stray cache file from a real
    user session never causes a false-positive HIT.
    """
    from marmelade.ui import theme
    from marmelade.ui.main_window import MainWindow

    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.resize(1280, 800)
    QApplication.processEvents()
    return window


def test_first_click_then_spacebar_plays_from_seek_position(
    main_window_for_seek, qtbot, tmp_path
) -> None:
    """Bug #1 regression — click BEFORE spacebar leaves engine.position_seconds
    at the seek target, not at zero.

    Flow:
        1. Open a small synthetic WAV via _open_file (bypasses QFileDialog).
        2. Wait for render_complete.
        3. Verify prime() fired (engine._sample_rate > 0).
        4. Click at the viewport horizontal center.
        5. Assert main_window._playback_engine.position_seconds ≈ duration/2
           (and critically: NOT 0.0).

    We do NOT press spacebar — the test pins the engine's authoritative
    position state which is what ``play(start_seconds=engine.position_seconds)``
    would consume. The seek path is the same one D-14 wires; spacebar is
    not in the loop.
    """
    import soundfile as sf

    from marmelade.ui.waveform_view import WaveformView  # noqa: F401

    main_window = main_window_for_seek

    # (1) Synthesize a 4 s stereo low-amplitude sine WAV — long enough to
    # render past the proxy pass-through boundary but fast to build.
    sr = 44100
    duration_s = 4.0
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    audio = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.stack([audio, audio], axis=1)  # (frames, channels)
    wav_path = tmp_path / "seek_fixture.wav"
    sf.write(str(wav_path), stereo, sr)

    # (2) Open + wait for render_complete to fire.
    # If the CI environment has libportaudio2 missing, engine._sd_available
    # is False and prime() short-circuits without populating _sample_rate.
    # In that case force _sd_available True BEFORE _open_file so the priming
    # call inside _open_file does its work; prime() itself does not touch
    # sounddevice (only pedalboard AudioFile).
    main_window._playback_engine._sd_available = True

    with qtbot.waitSignal(main_window.render_complete, timeout=15000):
        main_window._open_file(str(wav_path))

    # (3) Priming fired: engine has a non-zero sample_rate.
    assert main_window._playback_engine._sample_rate > 0, (
        "prime() should have populated _sample_rate at file-open time"
    )

    # (4) Simulate a click at the viewport horizontal center. Mirror the
    # click-helper pattern used by the chrome-offset regression test
    # (test_click_to_seek_under_chrome_offset) — supply an explicit
    # window-relative scenePos so the real ``mapToScene`` path is exercised
    # under MainWindow's actual chrome (left dock, toolbar, status bar).
    view = main_window._waveform_view
    viewport = view.graphics_layout.viewport()
    center_x = viewport.width() // 2
    click_y = viewport.height() // 2

    top = view.window()
    window_pt_press = viewport.mapTo(top, QPoint(center_x, click_y))
    window_pt_release = viewport.mapTo(top, QPoint(center_x + 1, click_y))
    scene_pos_press = QPointF(
        float(window_pt_press.x()), float(window_pt_press.y())
    )
    scene_pos_release = QPointF(
        float(window_pt_release.x()), float(window_pt_release.y())
    )
    _send_press_with_scene_pos(view, x=center_x, scene_pos=scene_pos_press, y=click_y)
    _send_release_with_scene_pos(
        view, x=center_x + 1, scene_pos=scene_pos_release, y=click_y
    )

    # Flush the eventFilter → seek_requested → _on_seek_requested →
    # engine.seek() pipeline.
    QApplication.processEvents()

    # (5) The crux: engine's authoritative position now reflects the click.
    # Under the bug, this is 0.0 because seek() silently zeros when
    # _sample_rate == 0. Under the fix, prime() set _sample_rate to 44100
    # so seek(t) lands at frame int(t * 44100) and position_seconds == t.
    pos = main_window._playback_engine.position_seconds
    expected = duration_s / 2.0
    # Wide-ish tolerance (±25% of duration = ±1 s) to absorb viewport-edge
    # insets, chrome offsets, and the int(t*sr) rounding without hiding the
    # bug — the bug would leave pos == 0.0, well outside ±1 s of 2.0 s.
    assert pos != 0.0, (
        "engine.position_seconds is 0.0 after pre-play click — "
        "prime() did not fire OR seek() saw _sample_rate==0 (bug #1)"
    )
    assert abs(pos - expected) <= 1.0, (
        f"engine.position_seconds={pos:.3f}s should be ~{expected:.1f}s "
        f"(±1 s) after a viewport-center click"
    )
