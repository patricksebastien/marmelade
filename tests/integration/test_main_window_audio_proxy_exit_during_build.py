"""Plan 02.1-05 Task 1 — D-12: app-exit cooperative cancel during proxy build.

Pins AUD-04 / D-12 / SC-7 (extended to the exit path):

* Open a non-WAV (MP3) source so the proxy worker is mid-build at close time.
* Trigger ``window.close()`` — ``MainWindow.closeEvent`` runs.
* Within the 800 ms cooperative-cancel deadline (D-12), the in-flight worker
  cancels, deletes its ``.proxy.tmp`` file, and the closeEvent returns.
* After close: NO ``.proxy.tmp`` remains under ``<cache_root>/audio/``.
* After close: ``audio_proxy_is_fresh(cache_root, src)`` returns ``None`` —
  the worker was cancelled before the atomic ``.tmp → .wav`` rename.
* After close: ``window._current_proxy_runnable`` is ``None`` (cleared by
  the closeEvent latch, or by the slot if the cancelled signal fired
  before the deadline expired).
* Wall-clock budget for the whole flow < 2 seconds (800 ms cancel
  deadline + cleanup overhead; the test must NOT hang).

Implementation detail: synthetic-sine MP3 fixtures decode so fast that a
10 s source would race to completion before the close fires. To keep the
test deterministic we monkeypatch ``audio_proxy_builder.iter_blocks`` with
a per-block ``time.sleep(0.05)`` shim — same idiom as
``test_main_window_audio_proxy_cancel_restart.py``. This guarantees A's
worker is reliably in flight when ``window.close()`` is called.

Why this test exists (Plan 04 SUMMARY context):
    Plan 04 ships the integration surface but DELIBERATELY defers the
    closeEvent. As documented in Plan 04 SUMMARY § "Deferred Issues",
    without a closeEvent override the peak-builder / audio-proxy
    workers can outlive their MainWindow GC, causing intermittent
    libshiboken UAFs on torn-down WaveformView. The structural fix
    is this plan's ``closeEvent`` — once it lands the UAF disappears
    because in-flight workers are cancelled cooperatively before the
    QMainWindow tear-down.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from marmelade.audio import audio_proxy_builder
from marmelade.audio.audio_proxy_cache import audio_proxy_is_fresh
from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_close_window_during_build_cancels_worker_and_removes_tmp(
    qtbot,
    qapp,
    tmp_cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``window.close()`` mid-build → cooperative cancel within 800 ms; no `.tmp` leftover."""
    theme.apply_theme(QApplication.instance())

    src = tmp_path / "long.mp3"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=10.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
    )

    # Slow the builder's block stream so the worker is reliably still
    # in flight when window.close() fires. ``iter_blocks`` is the inner
    # iterator used by ``build_audio_proxy``; wrapping each yield with a
    # 50 ms sleep bounds the total build time to (n_blocks * 50 ms),
    # which is far longer than the 800 ms cancel deadline.
    real_iter_blocks = audio_proxy_builder.iter_blocks

    def slow_iter_blocks(*args, **kwargs):
        for item in real_iter_blocks(*args, **kwargs):
            time.sleep(0.05)
            yield item

    monkeypatch.setattr(audio_proxy_builder, "iter_blocks", slow_iter_blocks)

    window = MainWindow()
    qtbot.addWidget(window)

    # Kick off the build but do NOT wait for completion.
    window._open_file(str(src))
    qtbot.wait(50)  # let the worker enqueue (slow_iter_blocks keeps it live)

    # Sanity — the worker is reliably mid-build now.
    assert window._current_proxy_runnable is not None, (
        "expected the proxy worker to be live; slow_iter_blocks did not engage"
    )

    cache_root = default_cache_root()
    audio_dir = cache_root / "audio"

    # Trigger the close. ``window.close()`` synchronously dispatches
    # ``closeEvent`` on the GUI thread; the override drains its own
    # event loop until the cancel deadline or terminal signal fires.
    t0 = time.perf_counter()
    window.close()
    close_elapsed_s = time.perf_counter() - t0

    # Wall-clock budget — 800 ms cancel deadline + ~200 ms overhead.
    # Hard ceiling at 2 s so a hang is loud, not silent.
    assert close_elapsed_s < 2.0, (
        f"closeEvent took {close_elapsed_s:.3f} s — must complete within 2 s "
        f"(800 ms cancel deadline + cleanup overhead)"
    )

    # The closeEvent latch clears the reference whether the worker
    # terminal signal fired in time OR the deadline expired.
    assert window._current_proxy_runnable is None, (
        "closeEvent did not clear _current_proxy_runnable after the cancel "
        "latch — the slot-prologue guard rests on this invariant"
    )

    # Drain a few extra ms — the worker thread may still be unwinding
    # for a tick after ``closeEvent`` returned. The builder unlinks
    # ``.tmp`` BEFORE re-raising BuildCancelled, then the worker thread
    # exits. ``qtbot.waitUntil`` lets the worker finish unwinding so
    # the ``.tmp`` cleanup is observable.
    qtbot.waitUntil(
        lambda: (not audio_dir.exists())
        or not any(p.suffix == ".tmp" for p in audio_dir.glob("*")),
        timeout=2500,
    )

    # No leftover ``.tmp`` anywhere in the audio cache directory.
    leftover_tmps = (
        list(audio_dir.glob("*.tmp")) if audio_dir.exists() else []
    )
    assert leftover_tmps == [], (
        f"unexpected leftover .tmp files after closeEvent: {leftover_tmps}"
    )

    # The cancellation completed before the atomic ``.tmp → .wav`` rename,
    # so no canonical proxy exists for the source.
    assert audio_proxy_is_fresh(cache_root, src) is None, (
        "audio_proxy_is_fresh saw a complete proxy after a mid-build "
        "close — atomic-rename invariant broken"
    )
