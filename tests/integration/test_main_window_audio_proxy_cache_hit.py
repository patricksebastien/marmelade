"""Plan 02.1-04 — D-13 / SC-5: cache HIT skips the worker.

Pins:

* First open of an MP3 source primes the cache (waits for
  ``audio_proxy_complete``).
* Second open of the SAME source on the (now-warm) cache:
    - does NOT emit ``audio_proxy_complete`` again (HIT path skips the
      worker entirely; the success-only seam fires only on builds).
    - ``window._current_proxy_runnable`` is None after the second open.
    - ``window._status_proxy_progress`` stays hidden — no "Preparing audio
      proxy" text shown for the cached open.
    - Spacebar shortcut tracks ``engine.is_available`` synchronously.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from marmelade.paths import default_cache_root  # noqa: F401 — conftest patches at module load
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_cache_hit_open_skips_worker_and_progress_ui(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Second open of an MP3 with a warm cache is synchronous (no worker, no UI)."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "cache_hit.mp3"
    # quick-260615-f77: 48 kHz source skips the resample-on-open branch
    # (canonical rate, used as-is) so this test exercises the MP3
    # proxy/cache-HIT pipeline it pins, unchanged by the new flow.
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

    # Track both terminal pipelines so the test reaches a clean state
    # before teardown (see mp3_open test rationale).
    proxy_done = {"v": False}
    render_done = {"v": False}
    window.audio_proxy_complete.connect(lambda _p: proxy_done.update(v=True))
    window.render_complete.connect(lambda: render_done.update(v=True))

    # First open — MISS path; wait for the proxy build AND the parallel
    # peak-builder render.
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: proxy_done["v"] and render_done["v"],
        timeout=15000,
    )

    # Now the cache is warm. Second open MUST be synchronous — no worker,
    # no progress text, no audio_proxy_complete emission. Reset the
    # render flag because the second open will fire render_complete again
    # (peak-builder cache HIT path), then drain it.
    render_done["v"] = False
    window._open_file(str(src))
    qtbot.waitUntil(lambda: render_done["v"], timeout=15000)

    assert window._current_proxy_runnable is None
    assert window._status_proxy_progress.isHidden() is True
    assert (
        window._shortcut_play_pause.isEnabled()
        == window._playback_engine.is_available
    )

    # Wait briefly — no audio_proxy_complete must fire on the second open.
    with qtbot.waitSignal(
        window.audio_proxy_complete, timeout=500, raising=False
    ) as blocker:
        pass
    assert blocker.signal_triggered is False
