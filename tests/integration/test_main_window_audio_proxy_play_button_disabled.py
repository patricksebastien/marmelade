"""Plan 02.1-04 — SC-3: spacebar shortcut disabled during proxy build.

Pins the Option 1 UX from CONTEXT:

* Immediately after ``_open_file`` returns on the MISS path,
  ``_shortcut_play_pause.isEnabled()`` is False (worker is building; user
  can't trigger playback because pedalboard would seek O(n) on the source
  MP3, which is exactly the latency bug 2.1 fixes).
* After ``audio_proxy_complete`` arrives, the shortcut's enable-state
  tracks ``engine.is_available`` (the engine is now primed with the
  proxy path; constant-time seek is available).

Implementation detail: same monkeypatch on
``audio_proxy_builder.iter_blocks`` as the cancel-restart test — synthetic
fixtures decode too quickly for the "in-flight" assertion to be reliable
without per-block stalling.
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


def test_spacebar_disabled_during_build_then_re_enabled_on_finished(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spacebar shortcut: enabled→disabled on MISS open; tracks backend on finish."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "play_button.mp3"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=3.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
    )

    # Stall each builder block so the "in-flight" state is observable.
    real_iter_blocks = audio_proxy_builder.iter_blocks

    def slow_iter_blocks(*args, **kwargs):
        for item in real_iter_blocks(*args, **kwargs):
            time.sleep(0.05)
            yield item

    monkeypatch.setattr(
        audio_proxy_builder, "iter_blocks", slow_iter_blocks
    )

    window = MainWindow()
    qtbot.addWidget(window)

    # Sanity: read pre-open state (depends on backend availability — we
    # don't assert, we just verify it doesn't crash).
    _ = window._shortcut_play_pause.isEnabled()

    # Track both terminal pipelines via direct connection so a fired-
    # before-wait emission is still observed.
    proxy_done = {"v": False}
    render_done = {"v": False}
    window.audio_proxy_complete.connect(lambda _p: proxy_done.update(v=True))
    window.render_complete.connect(lambda: render_done.update(v=True))

    # Kick off the open WITHOUT waiting for completion so we can observe
    # the in-flight state.
    window._open_file(str(src))
    qtbot.wait(20)  # let the worker enqueue (slow_iter_blocks keeps it alive)

    # During the build, spacebar MUST be disabled (Option 1 UX).
    assert window._shortcut_play_pause.isEnabled() is False
    assert window._current_proxy_runnable is not None

    # Wait for BOTH pipelines so teardown sees a clean event loop.
    qtbot.waitUntil(
        lambda: proxy_done["v"] and render_done["v"],
        timeout=20000,
    )

    # Post-build: shortcut tracks backend availability.
    assert (
        window._shortcut_play_pause.isEnabled()
        == window._playback_engine.is_available
    )
