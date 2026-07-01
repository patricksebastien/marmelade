"""Plan 02.1-04 — D-10 / SC-7: concurrent-open cancel-restart.

Open A mid-build, then open B → A is cancelled, A's ``.tmp`` is cleaned,
B proceeds and emits ``audio_proxy_complete``. No stale signal contamination.

* ``_current_proxy_runnable`` is non-None during A's build.
* After B completes, ``audio_proxy_complete`` payload is B's proxy path
  (NOT A's).
* No ``.tmp`` files remain under ``<cache_root>/audio/`` (A's tmp was
  cleaned by the cancel preamble).
* ``audio_proxy_is_fresh(cache_root, A)`` is None (A was cancelled before
  the atomic rename).
* ``audio_proxy_is_fresh(cache_root, B)`` returns B's proxy path.

Implementation detail: synthetic-sine MP3 fixtures decode so fast that a
10-second source can complete the build in < 50 ms on modern hardware
(pedalboard's JUCE decoder + soundfile's float32 WAV writer release the
GIL the whole time). To keep the test deterministic on every machine we
monkeypatch ``audio_proxy_builder.iter_blocks`` to ``time.sleep(0.05)``
between blocks — enough for A's worker to still be live when we open B.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from marmelade.audio import audio_proxy_builder
from marmelade.audio.audio_proxy_cache import (
    audio_proxy_is_fresh,
    audio_proxy_path,
    cache_key,
)
from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_open_b_mid_build_cancels_a_and_completes_b(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-build re-open cancels A, removes A's .tmp, B completes cleanly."""
    theme.apply_theme(QApplication.instance())
    src_a = tmp_path / "race_a.mp3"
    src_b = tmp_path / "race_b.mp3"
    make_sine(
        src_a,
        freq_hz=500.0,
        amp=0.5,
        duration_s=5.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
    )
    make_sine(
        src_b,
        freq_hz=2000.0,
        amp=0.5,
        duration_s=3.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
    )

    # Slow the builder's block stream so A is reliably still in flight
    # when we open B. ``iter_blocks`` is the inner-loop iterator used by
    # ``build_audio_proxy``; wrapping each yield with a small sleep
    # bounds the total build time to (n_blocks * 50 ms) which is ample.
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

    # Capture audio_proxy_complete payloads + track both terminal
    # pipelines for the deterministic drain at the end.
    payloads: list[str] = []
    render_done = {"v": False}
    window.audio_proxy_complete.connect(payloads.append)
    window.render_complete.connect(lambda: render_done.update(v=True))

    # Kick off A but do NOT wait for completion.
    window._open_file(str(src_a))
    qtbot.wait(20)  # let the worker enqueue (but not finish — slow_iter_blocks)

    # A's worker should be live now.
    assert window._current_proxy_runnable is not None

    # Open B — this triggers the cancel preamble and starts B's build.
    # Reset render_done flag because the cancel preamble will re-emit
    # render_complete for B's render pass.
    render_done["v"] = False
    pre_payloads_len = len(payloads)
    window._open_file(str(src_b))
    qtbot.waitUntil(
        lambda: len(payloads) > pre_payloads_len and render_done["v"],
        timeout=20000,
    )

    cache_root = default_cache_root()
    canonical_b = audio_proxy_path(cache_root, cache_key(src_b))

    # Payload (last one) is B's proxy, not A's.
    assert payloads, "audio_proxy_complete did not emit a payload"
    payload = Path(payloads[-1])
    assert payload == canonical_b

    # Cache freshness: A absent, B present.
    assert audio_proxy_is_fresh(cache_root, src_a) is None
    assert audio_proxy_is_fresh(cache_root, src_b) is not None

    # No leftover .tmp anywhere in the audio cache directory.
    audio_dir = cache_root / "audio"
    leftover_tmps = list(audio_dir.glob("*.tmp"))
    assert leftover_tmps == [], f"unexpected leftover .tmp files: {leftover_tmps}"
