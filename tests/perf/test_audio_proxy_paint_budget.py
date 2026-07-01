"""Plan 02.1-05 Task 2 — D-15 perf gate: proxy build does not block UI paints.

Mirrors ``tests/perf/test_render_frame_budget.py::test_paintevent_latency``
structurally (PATTERNS.md §14). Pins RESEARCH §Open-Q-3:

* Synthesise a 30 s MP3 source.
* Construct MainWindow, ``qtbot.addWidget(window)``, ``window.show()``.
* Instrument ``paintEvent`` on the central waveform plot widget — record
  per-call duration in ms.
* Trigger ``_open_file(mp3_src)`` — this enqueues the proxy worker on
  QThreadPool. The worker decodes via ``pedalboard.AudioFile.read()``
  (C++ — GIL released) and writes via ``soundfile.SoundFile.write()``
  (C via CFFI — GIL released), so the GUI thread should remain free
  to dispatch paint events.
* While the worker runs, drive ``setXRange`` deltas + ``qtbot.wait(5)``
  bursts (10 iterations, same pump shape as the analog) to force
  paintEvent dispatches under offscreen.
* Wait for ``audio_proxy_complete`` (60 s ceiling — generous, leaves
  room for slow CI hardware on a 30 s MP3).
* Assert at least 10 recorded paint durations (hard-fail with
  "Xvfb required" diagnostic if zero — silent ``ZeroDivisionError``
  is forbidden).
* Assert ``mean_ms < 50.0``.

RESEARCH §Open-Q-3 explicitly allows raising the inline ceiling to 100 ms
if CI flakes (the gate is a regression detector, not a perfectionist
target). I/O throttling is deferred to a 2.1.1 polish phase — adding
``QThread.usleep`` here would mask the regression.

Always-runs gate (no ``@skipif(_OFFSCREEN)`` — paintEvent fires under
offscreen once ``window.show()`` per Phase 1 N-2 pattern; same discipline
as ``test_paintevent_latency`` in the analog).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from pytestqt.exceptions import TimeoutError as QtBotTimeoutError
from PySide6.QtWidgets import QApplication

from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_audio_proxy_build_paintevent_under_50ms(
    qtbot,
    qapp,
    tmp_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """D-15 — mean paintEvent latency < 50 ms during an active proxy build.

    RESEARCH §Open-Q-3: 50 ms ceiling on a synthetic 30 s MP3 (~1.5x
    headroom on the 33 ms / 30 fps paint budget; allowance for occasional
    GIL contention during signal emission across the QThreadPool boundary).
    The worker decodes via pedalboard (C++) and writes via soundfile (C
    via CFFI) — both release the GIL so paintEvent dispatch on the GUI
    thread should remain unblocked.

    If the gate fails on CI at 60-80 ms: raise the inline ceiling to
    100 ms with a comment per RESEARCH §Open-Q-3. Do NOT add I/O
    throttling — that path is deferred to a 2.1.1 polish phase if
    HUMAN-UAT surfaces real-world UI freezes on slow disks.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.mp3"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=30.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
    )

    window = MainWindow()
    qtbot.addWidget(window)

    # N-2: MUST show() before paintEvent dispatch under offscreen.
    window.show()

    # N-2: best-effort exposure — offscreen may not raise the exposed
    # event reliably per Pitfall #6.
    try:
        with qtbot.waitExposed(window, timeout=2000):
            pass
    except QtBotTimeoutError:
        # Acceptable under offscreen — the paint-count precondition
        # guard below catches zero-paint cases honestly.
        pass

    # Instrument paintEvent — wrap the bound method on the PlotWidget
    # so we time every dispatch. Same shape as the analog
    # (test_render_frame_budget.py:312-323) — instrumenting the
    # central plot_widget rather than any heatmap lane keeps the
    # measurement focused on the waveform's hot path.
    plot_widget = window._waveform_view.plot_widget
    original_paint = plot_widget.paintEvent
    durations: list[float] = []

    def wrapped_paint(ev):
        t0 = time.perf_counter()
        original_paint(ev)
        durations.append((time.perf_counter() - t0) * 1000.0)

    plot_widget.paintEvent = wrapped_paint  # type: ignore[method-assign]

    # Kick off the build, then drive paints during the worker's lifetime.
    # ``audio_proxy_complete`` fires success-only AFTER prime() and the
    # cache-size footer refresh; waitSignal here is a clean
    # synchronisation point — same discipline as the integration tests.
    with qtbot.waitSignal(
        window.audio_proxy_complete, timeout=60000, raising=True
    ):
        window._open_file(str(src))
        # Force paints during the build — mirror the pump shape from
        # test_render_frame_budget.py:329-338.
        vb = plot_widget.plotItem.getViewBox()
        (x0, x1), _y = vb.viewRange()
        dx = max(0.001, (x1 - x0) * 0.01)
        for i in range(10):
            vb.setXRange(
                x0 + dx * (i + 1), x1 + dx * (i + 1), padding=0
            )
            plot_widget.update()
            qtbot.wait(5)

    # N-2: drain any queued paint events after the last delta.
    qtbot.wait(50)

    # N-2: HARD-FAIL if too few paints fired. Silent ZeroDivisionError
    # or vacuous pass is forbidden — same discipline as the analog.
    if len(durations) < 10:
        pytest.fail(
            f"paintEvent never fired under offscreen during proxy build — "
            f"Xvfb required (recorded {len(durations)} paints, expected >= 10)"
        )

    mean_ms = sum(durations) / len(durations)
    max_ms = max(durations)
    # Diagnostic — visible in pytest -s; the assertion error below
    # surfaces the same numbers if the gate trips.
    print(
        f"\n[D-15 perf gate] paintEvent during proxy build: "
        f"mean={mean_ms:.2f} ms, max={max_ms:.2f} ms, "
        f"samples={len(durations)}"
    )
    assert mean_ms < 50.0, (
        f"Mean paintEvent latency during proxy build = {mean_ms:.2f} ms "
        f"(max = {max_ms:.2f} ms, samples = {len(durations)}) — "
        f"D-15 budget is < 50 ms. If CI flakes at 60-80 ms, raise "
        f"the inline ceiling to 100 ms per RESEARCH §Open-Q-3."
    )
