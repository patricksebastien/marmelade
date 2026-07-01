"""Phase 2.1 HUMAN-UAT bug #1 — playback opens the proxy WAV, not the source MP3.

Regression pin for the bug surfaced during /gsd-verify-work 2.1:

  > "proxy works, created and then we see the cache size, good! BUT now
  >  clicking anywhere in the waveform takes all my cpu and finally after
  >  a few seconds play like if we are not using the wav"

Root cause: ``MainWindow._action_toggle_playback`` passed
``self._current_path`` (the source MP3) to ``engine.play()``, so the
pedalboard ``AudioFile`` open + seek hit the source MP3's O(n) seek path
every time the user pressed space — defeating the entire phase goal (SC-4).

Fix: ``MainWindow`` now tracks ``_current_playback_path`` separately from
``_current_path``. The playback path is set to:
  * the source path for native WAV opens (D-05)
  * the proxy path on cache HIT (D-13)
  * the proxy path inside ``_on_audio_proxy_finished`` (MISS completion)
and cleared on close / cancel.

This pin asserts that ``engine.play()`` is called with the **proxy path**,
not the source MP3 path, after the proxy build completes.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from marmelade.audio import audio_proxy_cache
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patches at module load
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_play_after_proxy_build_opens_proxy_path_not_source(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """After audio_proxy_complete fires, engine.play() opens the proxy WAV.

    Pinning the Phase 2.1 SC-4 contract end-to-end:
      1. Open a synthetic MP3 source (triggers the MISS → worker build path).
      2. Wait for ``audio_proxy_complete``.
      3. Stub ``engine.play`` to capture the path argument.
      4. Trigger play via ``_action_toggle_playback``.
      5. Assert the captured path equals the proxy WAV under the cache root,
         NOT the source MP3 path.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "playback_uses_proxy.mp3"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=2.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    # Force the engine into the "available" state so toggle_playback runs the
    # play() branch rather than the early-return for missing libportaudio2.
    monkeypatch.setattr(
        type(window._playback_engine),
        "is_available",
        property(lambda self: True),
    )

    # Track the path engine.play() opens — the load-bearing assertion of
    # this regression test. Bypass the real audio backend by stubbing play()
    # entirely; we only care which path was handed in.
    captured: dict[str, object] = {}

    def fake_play(path: str, start_seconds: float = 0.0) -> None:
        captured["path"] = path
        captured["start_seconds"] = start_seconds

    monkeypatch.setattr(window._playback_engine, "play", fake_play)
    # Stub prime() so the conftest tmp_cache_dir-rooted proxy file does
    # not need to be a real WAV decodable by pedalboard.
    monkeypatch.setattr(
        window._playback_engine, "prime", lambda path: None
    )
    monkeypatch.setattr(
        type(window._playback_engine),
        "is_playing",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 0.0),
    )

    # Open the source — kicks off the proxy build.
    with qtbot.waitSignal(window.audio_proxy_complete, timeout=10_000):
        window._open_file(src)

    # The phase contract: the playback path is now the proxy WAV.
    expected_proxy = audio_proxy_cache.audio_proxy_path(
        default_cache_root(), audio_proxy_cache.cache_key(src)
    )
    assert window._current_playback_path == expected_proxy, (
        f"playback path should be the proxy WAV; got "
        f"{window._current_playback_path}"
    )
    assert window._current_path == src, (
        f"source path should still be the MP3 (drives waveform + heatmap "
        f"reads); got {window._current_path}"
    )

    # Trigger play via the toolbar action (mirrors what the spacebar +
    # play button do — both route through _action_toggle_playback).
    window._action_toggle_playback()

    # The load-bearing assertion: play() was handed the PROXY path, not
    # the source MP3. If this fails, pedalboard would re-open the MP3
    # and pay the O(n) seek cost on every spacebar press (SC-4 broken).
    assert "path" in captured, "engine.play() was not called"
    assert Path(captured["path"]) == expected_proxy, (
        f"engine.play() must open the proxy WAV, not the source MP3.\n"
        f"  expected: {expected_proxy}\n"
        f"  got:      {captured['path']}\n"
        f"This is the Phase 2.1 SC-4 regression — see "
        f"02.1-HUMAN-UAT.md issue #1."
    )


def test_play_on_wav_source_opens_source_path(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """WAV-skip path (D-05): playback uses the source directly, no proxy.

    Native WAV files are already O(1) seek; D-05 explicitly skips the
    proxy build for them. The playback path should therefore equal the
    source WAV path — not be None, and not be redirected through a
    (nonexistent) proxy file.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "wav_skip.wav"
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
    captured: dict[str, object] = {}

    def fake_play(path: str, start_seconds: float = 0.0) -> None:
        captured["path"] = path

    monkeypatch.setattr(window._playback_engine, "play", fake_play)
    monkeypatch.setattr(window._playback_engine, "prime", lambda path: None)
    monkeypatch.setattr(
        type(window._playback_engine),
        "is_playing",
        property(lambda self: False),
    )
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 0.0),
    )

    window._open_file(src)
    # WAV path is synchronous — no signal to wait on.
    qapp.processEvents()

    assert window._current_playback_path == src, (
        f"WAV-skip path should set _current_playback_path to the source; "
        f"got {window._current_playback_path}"
    )

    window._action_toggle_playback()

    assert "path" in captured, "engine.play() was not called"
    assert Path(captured["path"]) == src, (
        f"WAV source should play directly (D-05). got {captured['path']}"
    )


def test_play_during_proxy_build_is_no_op(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """While the proxy is mid-build, _action_toggle_playback bails out.

    Spacebar / play are disabled at the shortcut level (SC-3), but
    _action_toggle_playback also defends with a None-check on
    ``_current_playback_path`` — without that, a fast-clicker who taps the
    toolbar between MISS-spawn and proxy-complete could still trigger
    play() on a None playback path (AttributeError) or worse, on the
    source MP3 (regression).
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "playback_during_build.mp3"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=2.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
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
    captured: dict[str, object] = {}

    def fake_play(path: str, start_seconds: float = 0.0) -> None:
        captured["path"] = path

    monkeypatch.setattr(window._playback_engine, "play", fake_play)
    monkeypatch.setattr(window._playback_engine, "prime", lambda path: None)
    monkeypatch.setattr(
        type(window._playback_engine),
        "is_playing",
        property(lambda self: False),
    )

    # Open the file synchronously, but cancel the worker IMMEDIATELY (before
    # qapp.processEvents pumps the worker's signals through). After cancel,
    # _current_playback_path should be None (cleared by cancel preamble on
    # the next open OR by close). Easiest way to land in the "build pending"
    # window is to inspect state right after _open_file returns.
    window._open_file(src)

    # MISS path was taken: worker spawned, playback path NOT yet set.
    assert window._current_proxy_runnable is not None, (
        "MISS path should have spawned a worker"
    )
    assert window._current_playback_path is None, (
        "Playback path must remain None until proxy completion — "
        "otherwise mid-build play() would hit the source MP3 (regression)"
    )

    # User taps play during build — must be a no-op.
    window._action_toggle_playback()
    assert "path" not in captured, (
        "_action_toggle_playback must not call engine.play() while "
        "_current_playback_path is None"
    )

    # Clean up — let the worker finish so the closeEvent timeout doesn't
    # need to handle it.
    with qtbot.waitSignal(window.audio_proxy_complete, timeout=10_000):
        pass
