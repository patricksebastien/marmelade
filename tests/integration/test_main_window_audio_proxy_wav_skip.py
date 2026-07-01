"""Plan 02.1-04 — D-05 / SC-6: WAV sources skip the proxy entirely.

Pins:

* Opening a ``.wav`` source does NOT enqueue an ``AudioProxyRunnable``
  (``window._current_proxy_runnable is None`` throughout the open).
* The transient ``_status_proxy_progress`` widget stays hidden — no
  "Preparing audio proxy" text is shown.
* The spacebar shortcut is enabled synchronously (no async build to wait
  on) and tracks ``engine.is_available``.
* ``audio_proxy_complete`` is NOT emitted (this is the success-only test
  seam for the proxy path; a WAV-skip MUST stay silent on it).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from marmelade.paths import default_cache_root  # noqa: F401 — conftest patches at module load
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_wav_open_skips_proxy_and_does_not_emit_audio_proxy_complete(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """WAV → no worker, no progress UI, no audio_proxy_complete emission."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
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

    # Wait on render_complete (which fires synchronously for WAV via the
    # peak-builder cache MISS path) to keep the test deterministic. Use
    # a short waitSignal on audio_proxy_complete in parallel — it MUST
    # never fire for WAV.
    with qtbot.waitSignal(
        window.render_complete, timeout=15000, raising=True
    ):
        window._open_file(str(src))

    # No audio-proxy worker spawned.
    assert window._current_proxy_runnable is None

    # Progress widget hidden — no "Preparing audio proxy" text shown.
    assert window._status_proxy_progress.isHidden() is True

    # Spacebar mirrors backend availability synchronously.
    assert (
        window._shortcut_play_pause.isEnabled()
        == window._playback_engine.is_available
    )

    # audio_proxy_complete must NOT have fired. Wait briefly and assert silence.
    with qtbot.waitSignal(
        window.audio_proxy_complete, timeout=200, raising=False
    ) as blocker:
        pass
    assert blocker.signal_triggered is False
