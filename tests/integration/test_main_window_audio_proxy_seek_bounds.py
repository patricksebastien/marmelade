"""Phase 2.1 HUMAN-UAT bug #2 — defensive seek-bounds clamp.

Even after the RF64 format swap (see ``test_audio_proxy_builder_rf64``),
MainWindow should clamp click-to-seek targets to the engine's known
duration. The original crash:

    ValueError: Cannot seek to position 740226612 frames, which is beyond
    end of file (536870911 frames) by -203355701 frames.

happened because the click coordinate translated to a frame past what the
truncated proxy could seek to. The structural fix (RF64) eliminates the
truncation. The clamp is belt-and-suspenders: any future divergence
between waveform-frame-count and engine-frame-count (rendering artifact,
mid-build file open, manual cache tampering) must not crash the playback
path.

This module pins:
  1. ``_on_seek_requested`` with a target past duration clamps to
     ``duration - 0.01``.
  2. ``_action_toggle_playback`` with engine.position_seconds past
     duration calls engine.play() with a clamped start.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from marmelade.paths import default_cache_root  # noqa: F401 — conftest patches at module load
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_seek_past_duration_is_clamped(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """Clicking past duration clamps to ``duration - 0.01`` instead of crashing.

    Setup: open a 2-second WAV (engine duration = 2.0). Request a seek to
    100 seconds. The seek handler must clamp to ~1.99 — both the visual
    playhead and the engine's start_frame.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "clamp_seek.wav"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=2.0,
        sample_rate=44100,
        channels=1,
        fmt="wav",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    monkeypatch.setattr(
        type(window._playback_engine),
        "is_available",
        property(lambda self: True),
    )

    seeks: list[float] = []

    def fake_seek(target_seconds: float) -> None:
        seeks.append(float(target_seconds))

    monkeypatch.setattr(window._playback_engine, "seek", fake_seek)

    window._open_file(src)
    qapp.processEvents()

    # Engine should be primed with duration ≈ 2.0.
    assert window._playback_engine.duration_seconds == 2.0, (
        f"engine duration should be 2.0 after WAV-skip prime; got "
        f"{window._playback_engine.duration_seconds}"
    )

    # User clicks at the right edge of a wildly mis-scaled waveform —
    # request 100 seconds.
    window._on_seek_requested(100.0)

    # The clamp ran: visual playhead should be at ≤ duration - 0.01.
    # Read the first lane playhead value as the canonical visual position.
    if window._lane_playheads:
        first_line = next(iter(window._lane_playheads.values()))
        playhead_pos = float(first_line.value())
        assert playhead_pos <= 2.0 - 0.01 + 1e-6, (
            f"visual playhead should be clamped to duration-0.01; got "
            f"{playhead_pos}"
        )

    # engine.seek should also see the clamped value, not 100.0.
    assert len(seeks) == 1, f"engine.seek expected once; got {len(seeks)}"
    assert seeks[0] <= 2.0 - 0.01 + 1e-6, (
        f"engine.seek should receive clamped target; got {seeks[0]}"
    )


def test_toggle_playback_clamps_start_to_engine_duration(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """play() is called with a start past EOF → clamped, no ValueError.

    Mirrors the original HUMAN-UAT bug #2 flow exactly:
      1. Open file.
      2. Visually click past engine duration (engine.position_seconds
         pushed past EOF by some artifact).
      3. Press play.
    Without the clamp, play() raises ValueError from af.seek. With the
    clamp, engine.play receives a start <= duration - 0.01.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "toggle_play_clamp.wav"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=2.0,
        sample_rate=44100,
        channels=1,
        fmt="wav",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    monkeypatch.setattr(
        type(window._playback_engine),
        "is_available",
        property(lambda self: True),
    )
    # Pretend the engine got a stale position past EOF (the scenario
    # that caused HUMAN-UAT bug #2). Real-world the engine's
    # position_seconds came from a stored _start_frame that was set by
    # an unclamped seek; we simulate by mocking position_seconds directly.
    monkeypatch.setattr(
        type(window._playback_engine),
        "is_playing",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 100.0),  # 50× past EOF
    )

    captured: dict[str, float] = {}

    def fake_play(path: str, start_seconds: float = 0.0) -> None:
        captured["start_seconds"] = float(start_seconds)

    monkeypatch.setattr(window._playback_engine, "play", fake_play)

    window._open_file(src)
    qapp.processEvents()

    # Engine duration = 2.0; position_seconds (mocked) = 100.0.
    # Press play — clamp must engage.
    window._action_toggle_playback()

    assert "start_seconds" in captured, "engine.play() was not called"
    assert captured["start_seconds"] <= 2.0 - 0.01 + 1e-6, (
        f"_action_toggle_playback must clamp start to "
        f"duration-0.01 (=1.99) when position is past EOF; got "
        f"{captured['start_seconds']}. Phase 2.1 HUMAN-UAT bug #2."
    )
    assert captured["start_seconds"] >= 0.0
