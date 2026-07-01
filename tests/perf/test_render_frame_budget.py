"""Plan 01-04 Task 2 — perf gates for the UI-SPEC §Performance contract.

Four tests replace the Plan-01 placeholder body:

1. ``test_reopen_under_500ms`` (ALWAYS runs, including under offscreen) —
   verifies the cache-HIT branch (load_proxy + render_proxy + signal
   emission) completes in < 500 ms wall-clock. Pre-warms the cache via
   :func:`peak_builder.build_proxy` so the second open through
   :meth:`MainWindow._open_file` hits the cache HIT branch.

2. ``test_pan_fps_at_least_30`` (SKIPPED under offscreen per RESEARCH
   Pitfall #6) — verifies ≥ 30 fps for 30 sequential ``setXRange`` pan
   deltas of 1% view width on a 30-second fixture. Pan-only — does not
   call ``zoom(...)``.

3. ``test_zoom_fps_at_least_30`` (SKIPPED under offscreen per RESEARCH
   Pitfall #6) — verifies ≥ 30 fps for 30 alternating ``zoom(1.05)`` /
   ``zoom(1/1.05)`` round-trips. Zoom-only — does not call
   ``setXRange``.

4. ``test_paintevent_latency`` (ALWAYS runs, including under offscreen —
   W-1: closes the CI gap left by ``skipif(offscreen)`` on the pan/zoom
   tests) — verifies mean ``paintEvent`` duration < 33 ms across 10 pan
   deltas. **N-2 fix:** calls ``window.show()`` + ``qtbot.waitExposed``
   (best-effort under offscreen per Pitfall #6) so paintEvent dispatch
   actually fires, then asserts at least ten recorded paints before
   computing the mean — a silent ``ZeroDivisionError`` or vacuous pass
   is forbidden (``pytest.fail`` with explicit "Xvfb required"
   message if zero paints recorded).

W-5 / N-3: ``default_cache_root`` is imported from ``marmelade.paths``
NOT from ``marmelade.audio.proxy_cache`` (the helper was relocated
there in Plan 02 to keep the audio package Qt-free). The
``tmp_cache_dir`` conftest fixture redirects the Qt writable-location
helper via ``setTestModeEnabled(True)`` so we never touch the user's
real cache.

B-2 / W-6: tests wait on ``MainWindow.render_complete`` (no
``_for_test`` suffix). On cache HIT the signal fires synchronously
inside ``_open_file`` so ``waitSignal`` exits on the same tick.

The 500 ms threshold and 33 ms paintEvent budget come from
UI-SPEC §Performance contract. The ≥ 30 fps pan/zoom budgets too.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from pytestqt.exceptions import TimeoutError as QtBotTimeoutError
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from marmelade.audio import peak_builder, proxy_cache
# N-3: import default_cache_root from marmelade.paths, NOT from proxy_cache.
from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from marmelade.ui.waveform_view import WaveformView
from tests.fixtures.synthesize import make_sine


_OFFSCREEN = os.environ.get("QT_QPA_PLATFORM") == "offscreen"


def _prewarm_cache(src: Path) -> Path:
    """Pre-warm the proxy cache for ``src`` and return the proxy path.

    Calls :func:`peak_builder.build_proxy` synchronously (no Qt worker
    — we are exercising the cache-HIT path the second time around).
    Derives the destination via ``proxy_cache.proxy_path(default_cache_root(),
    proxy_cache.cache_key(src))`` per W-5 / N-3 — DO NOT inline the Qt
    writable-location helper here; always go through
    :func:`marmelade.paths.default_cache_root`.
    """
    key = proxy_cache.cache_key(src)
    proxy_p = proxy_cache.proxy_path(default_cache_root(), key)
    proxy_p.parent.mkdir(parents=True, exist_ok=True)
    peak_builder.build_proxy(src, proxy_p, samples_per_pixel=256)
    return proxy_p


# ===================================================================== 500 ms


def test_reopen_under_500ms(
    qtbot,
    qapp,
    tmp_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """Cache HIT path (load_proxy → render_proxy → render_complete) < 500 ms.

    UI-SPEC §Performance contract: "Initial render of an 8-hour file
    from a cached proxy: under 500 ms wall-clock from QFileDialog
    accept (cache HIT)."

    We pre-warm the cache on a 30-second sine fixture (the threshold
    is dominated by header probe + memmap + setData + signal-emit,
    NOT by file length — a 30 s fixture is sufficient to exercise
    every step of the cache-HIT branch). Runs under offscreen because
    cache HIT does not require visible rendering.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=30.0,
        sample_rate=44100,
        channels=1,
    )

    # Pre-warm the cache via the same helper Plan 01-03's _open_file uses.
    _prewarm_cache(src)

    window = MainWindow()
    qtbot.addWidget(window)

    # Measure the cache-HIT path: probe → cache_key → proxy_path →
    # load_proxy → render_proxy → render_complete.emit (synchronous on HIT
    # per Plan 01-03 B-2). waitSignal exits on the same tick.
    # B-2: raising=True is INTENTIONAL — if first paint never fires
    # within 2 s the test must hard-fail loudly rather than silently no-op.
    with qtbot.waitSignal(window.render_complete, timeout=2000, raising=True):
        t0 = time.perf_counter()
        window._open_file(str(src))
    t1 = time.perf_counter()

    elapsed = t1 - t0
    assert elapsed < 0.5, (
        f"Cache HIT reopen took {elapsed * 1000:.1f} ms — "
        f"UI-SPEC §Performance budget is 500 ms"
    )


# ============================================================== pan-FPS (skip)


@pytest.mark.perf
@pytest.mark.skipif(_OFFSCREEN, reason="requires a real display; offscreen per RESEARCH Pitfall #6")
def test_pan_fps_at_least_30(
    qtbot,
    qapp,
    tmp_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """Pan ≥ 30 fps on a 30-s fixture (5M-pair equivalent for 8-h budget).

    UI-SPEC §Performance contract: "Pan and zoom remain at >= 30 fps".

    Drives 30 sequential ``setXRange`` pan deltas of 1% of view width
    (mouse-drag granularity) and asserts ``1 / mean_frame_time >= 30``.
    Pan-only — does NOT call ``zoom(...)``. SKIPPED under
    ``QT_QPA_PLATFORM=offscreen`` per RESEARCH Pitfall #6; expected to
    run locally before phase sign-off.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=30.0, sample_rate=44100, channels=1)
    _prewarm_cache(src)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    with qtbot.waitSignal(window.render_complete, timeout=2000, raising=True):
        window._open_file(str(src))

    vb = window._waveform_view.plot_widget.plotItem.getViewBox()
    (x0, x1), _y = vb.viewRange()
    dx = (x1 - x0) * 0.01  # 1% pan step matches mouse-drag granularity

    durations = []
    for i in range(30):
        t0 = time.perf_counter()
        vb.setXRange(x0 + dx, x1 + dx, padding=0)
        qtbot.wait(0)
        t1 = time.perf_counter()
        x0, x1 = x0 + dx, x1 + dx
        if i > 0:  # skip warm-up
            durations.append(t1 - t0)

    mean_frame_time = sum(durations) / len(durations)
    fps = 1.0 / mean_frame_time if mean_frame_time > 0 else float("inf")
    assert fps >= 30.0, (
        f"Pan FPS = {fps:.1f} (mean frame = {mean_frame_time * 1000:.2f} ms) — "
        f"UI-SPEC §Performance budget is >= 30 fps"
    )


# ============================================================= zoom-FPS (skip)


@pytest.mark.perf
@pytest.mark.skipif(_OFFSCREEN, reason="requires a real display; offscreen per RESEARCH Pitfall #6")
def test_zoom_fps_at_least_30(
    qtbot,
    qapp,
    tmp_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """Zoom ≥ 30 fps via 30 alternating zoom(1.05) / zoom(1/1.05) round-trips.

    UI-SPEC §Performance contract: "Pan and zoom remain at >= 30 fps".

    Zoom-only — does NOT call ``setXRange``. SKIPPED under
    ``QT_QPA_PLATFORM=offscreen`` per RESEARCH Pitfall #6.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=30.0, sample_rate=44100, channels=1)
    _prewarm_cache(src)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    with qtbot.waitSignal(window.render_complete, timeout=2000, raising=True):
        window._open_file(str(src))

    wf = window._waveform_view
    durations = []
    for i in range(30):
        t0 = time.perf_counter()
        wf.zoom(1.05)
        wf.zoom(1.0 / 1.05)  # round-trip — keeps the view from drifting
        qtbot.wait(0)
        t1 = time.perf_counter()
        if i > 0:  # skip warm-up
            durations.append(t1 - t0)

    mean_frame_time = sum(durations) / len(durations)
    fps = 1.0 / mean_frame_time if mean_frame_time > 0 else float("inf")
    assert fps >= 30.0, (
        f"Zoom FPS = {fps:.1f} (mean frame = {mean_frame_time * 1000:.2f} ms) — "
        f"UI-SPEC §Performance budget is >= 30 fps"
    )


# ============================================================ paintEvent < 33ms


def test_paintevent_latency(
    qtbot,
    qapp,
    tmp_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """Mean paintEvent duration < 33 ms (the 30-fps budget) across 10 pans.

    Always runs — including under ``QT_QPA_PLATFORM=offscreen``. This is
    the W-1 / N-2 offscreen-runnable gate that the pan/zoom skip-tests
    cannot provide.

    N-2 fix (this revision):
        * ``window.show()`` is REQUIRED — under the offscreen platform
          plugin, Qt only dispatches ``paintEvent`` to widgets that
          have been ``show()``-n. The previous plan revision omitted
          this call, which would have made the monkey-patched wrapper
          record zero durations and either raise ``ZeroDivisionError``
          or pass vacuously.
        * ``qtbot.waitExposed(window, timeout=2000)`` — best-effort
          exposure under offscreen per Pitfall #6; wrapped in
          try/except because offscreen exposure is non-guaranteed.
        * ``len(durations) >= 10`` precondition asserted BEFORE dividing
          — if zero (or too few) paints fired, hard-fail with
          ``pytest.fail("paintEvent never fired under offscreen — Xvfb
          required ...")`` so the gate is honest. A silent
          ``ZeroDivisionError`` or vacuous pass is forbidden.

    Why this is meaningful headless:
        PyQtGraph's ``PlotWidget`` inherits from ``QGraphicsView``,
        which DOES dispatch ``paintEvent`` under
        ``QT_QPA_PLATFORM=offscreen`` once the widget is shown/exposed
        — the offscreen platform plugin runs the paint pipeline, it
        just doesn't blit pixels to a screen. So the metric measures
        the rendering hot path's cost (downsampling + clipping +
        pen-stroking) without requiring a real display.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=30.0, sample_rate=44100, channels=1)
    _prewarm_cache(src)

    window = MainWindow()
    qtbot.addWidget(window)

    # N-2: MUST show() before paintEvent dispatch under offscreen.
    window.show()

    # N-2: best-effort exposure — offscreen may not raise the exposed
    # event reliably per Pitfall #6, but we should give it a chance
    # before driving paints.
    try:
        with qtbot.waitExposed(window, timeout=2000):
            pass
    except QtBotTimeoutError:
        # Acceptable under offscreen — we proceed and let the
        # paint-count precondition guard catch zero-paint cases.
        pass

    # Wait for the render before instrumenting paints so the
    # PlotDataItem has data.
    with qtbot.waitSignal(window.render_complete, timeout=2000, raising=True):
        window._open_file(str(src))

    # Instrument paintEvent — wrap the bound method on the PlotWidget
    # instance so we time every dispatch.
    plot_widget = window._waveform_view.plot_widget
    original_paint = plot_widget.paintEvent
    durations: list[float] = []

    def wrapped_paint(ev):
        t0 = time.perf_counter()
        original_paint(ev)
        durations.append((time.perf_counter() - t0) * 1000.0)

    plot_widget.paintEvent = wrapped_paint  # type: ignore[method-assign]

    # Drive 10 setXRange deltas — same 1% pan step as the pan FPS test.
    # Under offscreen Qt batches paint events aggressively, so each
    # delta calls update() + waits a few ms so paintEvent dispatches
    # individually for measurement.
    vb = plot_widget.plotItem.getViewBox()
    (x0, x1), _y = vb.viewRange()
    dx = (x1 - x0) * 0.01
    for i in range(10):
        vb.setXRange(x0 + dx * (i + 1), x1 + dx * (i + 1), padding=0)
        plot_widget.update()  # request a paint
        qtbot.wait(5)         # let the event loop dispatch it

    # N-2: drain any queued paint events after the last delta.
    qtbot.wait(50)

    # N-2: HARD-FAIL if too few paints fired. A silent ZeroDivisionError
    # or a vacuous pass is forbidden by the plan.
    if len(durations) < 10:
        pytest.fail(
            f"paintEvent never fired under offscreen — Xvfb required "
            f"(recorded {len(durations)} paints, expected >= 10)"
        )

    mean_ms = sum(durations) / len(durations)
    assert mean_ms < 33.0, (
        f"Mean paintEvent latency = {mean_ms:.2f} ms — "
        f"UI-SPEC §Performance budget is < 33 ms (30 fps)"
    )


# ===================================================== CR-01 memory regression


def test_render_proxy_x_array_memory_bound(qtbot, qapp) -> None:
    """CR-01 regression gate — render_proxy x-array peak < 64 MiB on 5M pairs.

    Plan 01-05 closure for REVIEW.md CR-01. Pre-fix path used
    ``np.arange(n_points, dtype=np.float64) * (spp / (2 * sr))`` which
    materialised TWO float64 arrays of 10M entries (the arange + the
    scaled result), peaking at ≈ 85.83 MiB on the 5M-pair calibration
    fixture (measured under offscreen on the calibration step of this
    plan). Post-fix path uses
    ``np.arange(n_points, dtype=np.float32) * np.float32(spp / (2 * sr))``
    which peaks at ≈ 47.69 MiB (one float32 arange materialised plus the
    scaled float32 result before the intermediate is freed).

    Threshold rationale:
        OLD peak ≈ 85.83 MiB ; NEW peak ≈ 47.69 MiB. Threshold set to
        64 MiB — sits between OLD (fails: 85>64) and NEW (passes: 47<64,
        ~25% headroom on the new value). The OLD float64 implementation
        would FAIL this test loudly with a clear miss; the NEW float32
        implementation passes with margin. PyQtGraph's setData may copy
        the x array internally; 64 MiB leaves enough room without false
        positives on the calibrated NEW peak.

    Always runs (including under QT_QPA_PLATFORM=offscreen) — no disk I/O,
    no MainWindow, < 1 s wall-clock. Use ``WaveformView`` directly so the
    measurement isolates the render path from the open-file flow (CR-01
    is about the render path only; T-05-03 — baseline noise floor is
    minimised by avoiding MainWindow construction).
    """
    theme.apply_theme(QApplication.instance())

    # 5_000_000 pairs ≈ 29 000 s at sr=44100/spp=256 — well past the
    # "tens of MB" GUI-tier ceiling (RESEARCH §Architectural
    # Responsibility Map). No disk I/O; constructed in-memory.
    proxy_arr = np.zeros((5_000_000, 2), dtype=np.int16)

    wf = WaveformView()
    qtbot.addWidget(wf)

    tracemalloc.start()
    baseline_curr, _ = tracemalloc.get_traced_memory()
    wf.render_proxy(proxy_arr, sample_rate=44100, samples_per_pixel=256)
    _curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    delta_bytes = peak - baseline_curr
    delta_mib = delta_bytes / (1024 * 1024)
    threshold_mib = 64.0
    assert delta_bytes < int(threshold_mib * 1024 * 1024), (
        f"render_proxy x-array peak = {delta_mib:.2f} MiB exceeds "
        f"{threshold_mib:.0f} MiB ceiling. Pre-fix (float64) peak was "
        f"≈ 85.83 MiB; post-fix (float32) peak should be ≈ 47.69 MiB. "
        f"This test guards REVIEW.md CR-01 — a float64 regression in "
        f"render_proxy will fail here before reaching merge."
    )


# ===================================================== HUMAN-UAT item 4 (slow)


@pytest.mark.slow
def test_render_proxy_long_file_memory_under_200mib(
    qtbot,
    qapp,
    tmp_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """HUMAN-UAT item 4 closure (automated half) — _open_file peak < 200 MiB on a 1h fixture.

    Synth a 1-hour mono 1 kHz sine WAV (~317 MB on disk), pre-warm the
    proxy cache, then time ``MainWindow._open_file`` end-to-end under
    tracemalloc. Asserts the per-call peak across the FULL cache-HIT
    path (probe → load_proxy → render_proxy → render_complete) stays
    under 200 MiB.

    Why 200 MiB:
        UI-SPEC §Performance contract tolerates 1 GiB. The 1h fixture at
        sr=44100 / spp=256 builds ≈ 620 000 proxy pairs; render_proxy
        materialises an x array of 1.24 M float32 entries ≈ 4.96 MiB.
        The 200 MiB ceiling captures the full _open_file flow including
        build_proxy peak (numpy reshape on 131_072-sample blocks during
        the cache MISS pre-warm). 8h extrapolation: ≈ 4x → ≈ 800 MiB
        peak in the worst case, still under UI-SPEC's 1 GiB tolerance.

    The 8h fixture (~2.5 GB on disk) is impractical in CI; this 1h gate
    is the documented practical compromise (HUMAN-UAT item 4 — automated
    half). The 8h end-user UAT remains a human_needed item but the cliff
    is now extrapolation, not unmeasured behaviour. CR-01 scaling
    extrapolation also lands here.

    Slow-marked: runs under ``pytest -m slow`` (~ 1-3 min wall-clock for
    fixture synthesis + proxy build + render). Defensive fixture-size
    check guards against vacuous passes from a misconfigured make_sine
    yielding zero blocks.

    T-05-03: tracemalloc baseline snapshot taken IMMEDIATELY before the
    timed call so framework allocations do not inflate the measurement.
    """
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture_1h.wav"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=3600.0,
        sample_rate=44100,
        channels=1,
    )

    # Defensive: if make_sine silently produces an empty/short fixture
    # the memory measurement is meaningless. Expected: 3600 * 44100 * 2
    # = 317_520_000 bytes (~303 MiB) for 1h mono int16 PCM WAV.
    assert src.stat().st_size > 300 * 1024 * 1024, (
        f"1h fixture too small: {src.stat().st_size} bytes — make_sine "
        f"may have produced zero or insufficient blocks"
    )

    # Pre-warm the proxy via the existing helper — keeps the timed
    # _open_file call on the cache-HIT branch.
    _prewarm_cache(src)

    window = MainWindow()
    qtbot.addWidget(window)

    # depth=25 captures numpy internal allocations (default depth=1
    # truncates them and underestimates peak).
    tracemalloc.start(25)
    baseline_curr, _ = tracemalloc.get_traced_memory()
    with qtbot.waitSignal(window.render_complete, timeout=10000, raising=True):
        window._open_file(str(src))
    _curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    delta_bytes = peak - baseline_curr
    delta_mib = delta_bytes / (1024 * 1024)
    threshold_mib = 200.0
    assert delta_bytes < int(threshold_mib * 1024 * 1024), (
        f"_open_file peak = {delta_mib:.2f} MiB exceeds {threshold_mib:.0f} "
        f"MiB ceiling on a 1h fixture. UI-SPEC §Performance tolerates "
        f"1 GiB; this 200 MiB gate is the documented HUMAN-UAT item 4 + "
        f"CR-01 scaling extrapolation ceiling. 8h projection at ~4x → "
        f"≈ {delta_mib * 4:.0f} MiB."
    )


# ===================================================== plan 01-09 CPU-cost gate


def test_setxrange_cpu_under_5ms_on_5m_pairs(qtbot, qapp) -> None:
    """USER FEEDBACK 2026-05-13 regression gate — mean ``setXRange`` wall-clock < 5 ms on 5M pairs.

    Plan 01-09 closure for the user's "zooming is eating all my cpu" complaint.
    The structural cause (pre-fix): PyQtGraph's
    ``setDownsampling(auto=True, method='peak')`` still pages through every
    proxy point on every paint, and ``setXRange`` (the API the mouse-wheel
    zoom dispatches) triggers a repaint. For an 8h file at sr=44100 / spp=256
    the proxy contains ≈ 5.4M min/max pairs → ≈ 10.8M plot points after
    saw-wave doubling. The structural fix: ``WaveformView.render_proxy`` now
    pre-aggregates the proxy to at most MAX_RENDER_PROXY_PAIRS = 4000 bins
    BEFORE handing the array to PyQtGraph, so each paint sees at most 8000
    points regardless of file duration.

    This test is the always-on CPU-cost regression gate that proves the fix
    lands. It deliberately constructs a 5_000_000-pair synthetic in-memory
    int16 proxy (5_000_000 >> 4000) to force the aggregation branch, then
    times 10 sequential ``setXRange`` deltas via ``time.perf_counter_ns()``
    and asserts the mean per-call wall-clock stays under 5 ms. The 5 ms
    ceiling rationale: post-fix wall-clock is typically < 1 ms (4000-bin
    PlotDataItem is trivial to paint); pre-fix at this density is typically
    > 50 ms per setXRange (PyQtGraph pages through 10M points each paint).
    5 ms gives 5× headroom over the post-fix expectation while still
    rejecting any future regression that removes the aggregation branch.

    Always runs (including under QT_QPA_PLATFORM=offscreen) — no disk I/O,
    no top-level window construction, < 2 s wall-clock. Uses
    ``WaveformView`` directly so the measurement isolates the render →
    PlotDataItem → ViewBox.setXRange chain from the open-file flow.

    Calibration protocol: if this test fails with mean between 5 and 10 ms
    on CI hardware, raise the ceiling to 10 ms inline and document why
    (slow hardware; the pre-fix > 50 ms gate still holds). If mean > 10 ms,
    the aggregation branch did NOT land correctly — investigate Task 1's
    implementation; do NOT raise the ceiling further.
    """
    theme.apply_theme(QApplication.instance())

    # 5_000_000 pairs > MAX_RENDER_PROXY_PAIRS (4000) — forces the aggregation
    # branch in render_proxy. Content is irrelevant (zeros are fine); only
    # the SHAPE matters. ≈ 20 MB in RAM; no disk I/O.
    proxy_arr = np.zeros((5_000_000, 2), dtype=np.int16)

    wf = WaveformView()
    qtbot.addWidget(wf)
    # ``show()`` matches the offscreen paint-dispatch pattern from
    # ``test_paintevent_latency`` — the offscreen platform plugin still runs
    # the paint pipeline once the widget is shown.
    wf.show()

    wf.render_proxy(proxy_arr, sample_rate=44100, samples_per_pixel=256)

    vb = wf.plot_widget.plotItem.getViewBox()
    (x0, x1), _y = vb.viewRange()
    dx = (x1 - x0) * 0.01  # 1% pan step matches mouse-drag granularity

    durations_ns: list[int] = []
    for i in range(10):
        t0 = time.perf_counter_ns()
        vb.setXRange(x0 + dx * (i + 1), x1 + dx * (i + 1), padding=0)
        qtbot.wait(0)  # let the redraw dispatch
        t1 = time.perf_counter_ns()
        durations_ns.append(t1 - t0)

    mean_ms = (sum(durations_ns) / len(durations_ns)) / 1_000_000.0
    max_ms = max(durations_ns) / 1_000_000.0
    assert mean_ms < 5.0, (
        f"USER FEEDBACK 2026-05-13 / plan 01-09 regression gate: "
        f"mean setXRange wall-clock = {mean_ms:.3f} ms (max = {max_ms:.3f} ms) "
        f"on 5_000_000-pair proxy exceeds 5.0 ms ceiling. "
        f"Pre-fix at this density: PyQtGraph pages through ~10M points per "
        f"paint = wall-clock typically > 50 ms per setXRange (the user's "
        f"'zooming is eating all my cpu' complaint). Post-fix at 4000-bin "
        f"density: typical wall-clock < 1 ms. The 5 ms ceiling has 5x "
        f"headroom over the post-fix expectation. If this fails, the "
        f"aggregation branch in WaveformView.render_proxy likely regressed."
    )
