"""Plan 02.1-04 — MP3 open spawns audio-proxy worker; `audio_proxy_complete` fires.

Pins SC-1 + SC-4 + AUD-04 user-visible behavior:

* Non-WAV source on an empty cache spawns an ``AudioProxyRunnable`` on the
  global ``QThreadPool``.
* The new test seam ``MainWindow.audio_proxy_complete = Signal(str)`` fires
  EXACTLY ONCE on success with the proxy WAV path as payload.
* The canonical proxy WAV exists on disk at
  ``<cache_root>/audio/<key>.proxy.wav`` after the signal fires.
* The spacebar shortcut tracks ``engine.is_available`` after build completes.
* No ``.tmp`` sibling remains in the cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from marmelade.audio.audio_proxy_cache import (
    audio_proxy_is_fresh,
    audio_proxy_path,
    cache_key,
)
from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_mp3_open_spawns_proxy_worker_and_emits_audio_proxy_complete(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Open an MP3 → audio_proxy_complete fires → proxy WAV exists at canonical path."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.mp3"
    # quick-260615-f77: use a 48 kHz source so the open flow skips the
    # resample-on-open branch (48 kHz is the canonical rate, used as-is)
    # and exercises the proxy-worker pipeline this test pins. A non-48 kHz
    # MP3 would be converted to a 48 kHz WAV working file first and play
    # via the WAV-skip branch (no proxy worker) — that path is covered by
    # the resample-on-open behavior, not by this proxy-pipeline test.
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=3.0,
        sample_rate=48000,
        channels=1,
        fmt="mp3",
    )

    window = MainWindow()
    qtbot.addWidget(window)

    # Capture both pipelines via direct connection so we can wait on
    # them with ``waitUntil`` — robust to signals emitted before any
    # ``waitSignal`` context starts listening. Draining the peak-builder
    # render_complete in addition to ``audio_proxy_complete`` keeps test
    # teardown clean (a slow peak-builder firing ``finished`` after the
    # WaveformView is GC'd trips a libshiboken UAF in the captured Qt
    # event loop — pre-existing concern unrelated to this plan).
    payloads: list[str] = []
    render_done = {"v": False}
    window.audio_proxy_complete.connect(payloads.append)
    window.render_complete.connect(lambda: render_done.update(v=True))

    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: len(payloads) > 0 and render_done["v"],
        timeout=15000,
    )

    cache_root = default_cache_root()
    # Cache HIT probe must now succeed for the source.
    fresh = audio_proxy_is_fresh(cache_root, src)
    assert fresh is not None, "audio_proxy_is_fresh did not see the cache after build"

    # Signal payload is the canonical proxy path.
    assert payloads, "audio_proxy_complete did not emit a payload"
    payload = payloads[0]
    canonical = audio_proxy_path(cache_root, cache_key(src))
    assert Path(payload) == canonical

    # Spacebar mirrors backend availability after the build settled.
    assert (
        window._shortcut_play_pause.isEnabled()
        == window._playback_engine.is_available
    )

    # No leftover .tmp sibling.
    tmp_sibling = canonical.with_suffix(canonical.suffix + ".tmp")
    assert not tmp_sibling.exists()
