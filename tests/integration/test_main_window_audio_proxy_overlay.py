"""Phase 2.1 HUMAN-UAT request #3 (final form) — inline audio-proxy banner.

User pivoted from "full modal overlay" to "inline progress over disabled
waveform" after discovering the full-screen ProgressOverlay reports
``isVisible()=True`` but does NOT actually paint over the WaveformView's
``pg.GraphicsLayoutWidget`` on Linux+Qt compositor stacks. The new design
uses :class:`AudioProxyProgressBanner` — a compact QFrame pinned at
top-center of the WaveformView — and gates waveform clicks via
``MainWindow._on_seek_requested``.

This module pins:
  1. Banner visible during audio-proxy build.
  2. Banner body text identifies the audio proxy build.
  3. Banner heading is "Preparing audio proxy".
  4. Banner hidden after audio_proxy_complete fires.
  5. Banner NOT shown on WAV-skip path.
  6. Banner NOT shown on audio cache HIT.
  7. Click-to-seek is gated during the build (no playhead movement,
     no engine.seek call).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from marmelade.audio import audio_proxy_builder
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patches at module load
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def _stall_iter_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slow the audio-proxy builder so 'mid-build' assertions are observable."""
    real_iter_blocks = audio_proxy_builder.iter_blocks

    def slow_iter_blocks(*args, **kwargs):
        for item in real_iter_blocks(*args, **kwargs):
            time.sleep(0.05)
            yield item

    monkeypatch.setattr(
        audio_proxy_builder, "iter_blocks", slow_iter_blocks, raising=True
    )


def test_banner_visible_during_audio_proxy_build(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio proxy MISS spawn → banner visible, flag set."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "banner_visible.mp3"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=3.0,
        sample_rate=44100, channels=1, fmt="mp3",
    )
    _stall_iter_blocks(monkeypatch)

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    window._open_file(src)
    qapp.processEvents()
    with qtbot.waitSignal(
        window._current_proxy_runnable.signals.progress, timeout=5_000
    ):
        pass
    qapp.processEvents()

    assert window._audio_proxy_overlay_active is True, (
        "audio proxy overlay flag must be set after MISS spawn"
    )
    assert window._audio_proxy_banner.isVisible() is True, (
        "AudioProxyProgressBanner must be visible during audio proxy build "
        "(HUMAN-UAT #3 final form)"
    )

    window._current_proxy_runnable.cancel()
    qapp.processEvents()


def test_banner_heading_is_preparing_audio_proxy(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Banner heading reads 'Preparing audio proxy' during the build."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "banner_heading.mp3"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=3.0,
        sample_rate=44100, channels=1, fmt="mp3",
    )
    _stall_iter_blocks(monkeypatch)

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    window._open_file(src)
    qapp.processEvents()
    with qtbot.waitSignal(
        window._current_proxy_runnable.signals.progress, timeout=5_000
    ):
        pass
    qapp.processEvents()

    from PySide6.QtWidgets import QLabel

    headings = [
        label.text()
        for label in window._audio_proxy_banner.findChildren(QLabel)
        if "preparing" in label.text().lower()
    ]
    assert any("audio proxy" in h.lower() for h in headings), (
        f"banner heading must read 'Preparing audio proxy'; got: {headings}"
    )

    window._current_proxy_runnable.cancel()
    qapp.processEvents()


def test_banner_body_identifies_audio_proxy_build(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Banner body line shows filename + duration + lock note."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "banner_body.mp3"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=3.0,
        sample_rate=44100, channels=1, fmt="mp3",
    )
    _stall_iter_blocks(monkeypatch)

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    window._open_file(src)
    qapp.processEvents()
    with qtbot.waitSignal(
        window._current_proxy_runnable.signals.progress, timeout=5_000
    ):
        pass
    qapp.processEvents()

    from PySide6.QtWidgets import QLabel

    body_candidates = [
        label.text()
        for label in window._audio_proxy_banner.findChildren(QLabel)
        if "click-to-play locked" in label.text()
        or "banner_body.mp3" in label.text()
    ]
    assert body_candidates, (
        f"banner body must mention the filename or the lock note; got "
        f"labels: {[l.text() for l in window._audio_proxy_banner.findChildren(QLabel)]}"
    )

    window._current_proxy_runnable.cancel()
    qapp.processEvents()


def test_banner_hidden_after_audio_proxy_finished(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
) -> None:
    """After audio_proxy_complete fires, banner is hidden and flag cleared."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "banner_finishes.mp3"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=2.0,
        sample_rate=44100, channels=1, fmt="mp3",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    with qtbot.waitSignal(window.audio_proxy_complete, timeout=10_000):
        window._open_file(src)
    qapp.processEvents()

    assert window._audio_proxy_overlay_active is False, (
        "audio proxy overlay flag must clear after build completion"
    )
    assert window._audio_proxy_banner.isVisible() is False, (
        "banner must be hidden after audio_proxy_complete"
    )


def test_banner_not_shown_on_wav_skip(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
) -> None:
    """WAV-skip path: no audio proxy build → no banner shown."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "banner_wav_skip.wav"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=2.0,
        sample_rate=44100, channels=1, fmt="wav",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    window._open_file(src)
    with qtbot.waitSignal(window.render_complete, timeout=10_000):
        pass
    qapp.processEvents()

    assert window._audio_proxy_overlay_active is False
    assert window._audio_proxy_banner.isVisible() is False, (
        "banner must not appear for WAV-skip path"
    )


def test_banner_not_shown_on_audio_cache_hit(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
) -> None:
    """Audio cache HIT: no build → no banner."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "banner_cache_hit.mp3"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=2.0,
        sample_rate=44100, channels=1, fmt="mp3",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    # First open builds the proxy.
    with qtbot.waitSignal(window.audio_proxy_complete, timeout=10_000):
        window._open_file(src)
    qapp.processEvents()

    # Second open hits the cache.
    window._open_file(src)
    qapp.processEvents()

    assert window._audio_proxy_overlay_active is False
    assert window._current_proxy_runnable is None
    assert window._audio_proxy_banner.isVisible() is False, (
        "banner must not appear on audio cache HIT"
    )


def test_banner_stays_visible_in_unavailable_state_after_cancel(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After user cancels build, banner stays up in NOT-BUILT mode.

    Without this, the user is stranded — playback unavailable, waveform
    clicks gated, no UI to retry. The banner persists with a "Build
    proxy" button that re-spawns the worker.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "banner_cancel.mp3"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=3.0,
        sample_rate=44100, channels=1, fmt="mp3",
    )
    _stall_iter_blocks(monkeypatch)

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    window._open_file(src)
    qapp.processEvents()
    with qtbot.waitSignal(
        window._current_proxy_runnable.signals.progress, timeout=5_000
    ):
        pass
    qapp.processEvents()

    assert window._audio_proxy_banner.isVisible() is True

    # User clicks "Stop building proxy".
    with qtbot.waitSignal(
        window._current_proxy_runnable.signals.cancelled, timeout=5_000
    ):
        window._audio_proxy_banner.cancel_button.click()
    qapp.processEvents()

    # Banner stays visible in unavailable mode.
    assert window._audio_proxy_overlay_active is True, (
        "banner flag must stay True after cancel — user needs a retry surface"
    )
    assert window._audio_proxy_banner.isVisible() is True, (
        "banner must remain visible in unavailable mode after cancel"
    )
    assert window._audio_proxy_banner.cancel_button.text() == "Build proxy", (
        f"button label must switch to 'Build proxy' after cancel; got "
        f"{window._audio_proxy_banner.cancel_button.text()!r}"
    )
    # Retry args must be stashed so the Build button can spawn again.
    assert window._audio_proxy_retry_args is not None, (
        "retry args must persist so the Build button can re-spawn"
    )


def test_build_button_respawns_audio_proxy_worker(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clicking 'Build proxy' on the unavailable banner spawns a new worker.

    Round-trip: spawn → cancel → unavailable banner → click Build →
    new worker spawned → banner switches back to BUILDING mode.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "banner_retry.mp3"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=3.0,
        sample_rate=44100, channels=1, fmt="mp3",
    )
    _stall_iter_blocks(monkeypatch)

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    # First build: spawn + cancel.
    window._open_file(src)
    qapp.processEvents()
    with qtbot.waitSignal(
        window._current_proxy_runnable.signals.progress, timeout=5_000
    ):
        pass
    first_runnable = window._current_proxy_runnable
    with qtbot.waitSignal(
        window._current_proxy_runnable.signals.cancelled, timeout=5_000
    ):
        window._audio_proxy_banner.cancel_button.click()
    qapp.processEvents()

    assert window._audio_proxy_banner.cancel_button.text() == "Build proxy"

    # Click Build → new worker spawned, banner switches to BUILDING mode.
    window._audio_proxy_banner.cancel_button.click()
    qapp.processEvents()

    assert window._current_proxy_runnable is not None, (
        "Build button must spawn a new AudioProxyRunnable"
    )
    assert window._current_proxy_runnable is not first_runnable, (
        "must be a NEW runnable (the cancelled one is dead)"
    )
    assert window._audio_proxy_banner.cancel_button.text() == "Stop building proxy", (
        "banner must switch back to BUILDING mode (button reads 'Stop building proxy')"
    )

    window._current_proxy_runnable.cancel()
    qapp.processEvents()


def test_click_to_seek_is_gated_during_build(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While the banner is up, _on_seek_requested is a no-op.

    Without this gate the visual playhead would still jump on click and
    the user would think they CAN seek mid-build. The whole handler
    early-returns so neither the playhead lines nor engine.seek fire.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "click_gated.mp3"
    make_sine(
        src, freq_hz=1000.0, amp=0.5, duration_s=3.0,
        sample_rate=44100, channels=1, fmt="mp3",
    )
    _stall_iter_blocks(monkeypatch)

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1200, 700)
    window.show()
    qtbot.waitExposed(window)

    monkeypatch.setattr(
        type(window._playback_engine),
        "is_available",
        property(lambda self: True),
    )
    seeks: list[float] = []
    monkeypatch.setattr(
        window._playback_engine, "seek", lambda s: seeks.append(float(s))
    )

    window._open_file(src)
    qapp.processEvents()
    with qtbot.waitSignal(
        window._current_proxy_runnable.signals.progress, timeout=5_000
    ):
        pass
    qapp.processEvents()

    assert window._audio_proxy_overlay_active is True

    # Simulate a click — _on_seek_requested(seconds=1.5).
    window._on_seek_requested(1.5)
    qapp.processEvents()

    assert len(seeks) == 0, (
        f"engine.seek must NOT be called during audio-proxy build; "
        f"got calls: {seeks}"
    )

    window._current_proxy_runnable.cancel()
    qapp.processEvents()
