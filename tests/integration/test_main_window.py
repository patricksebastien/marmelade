"""Integration: MainWindow double-open cancels first build then swaps in new (N-5).

This test deliberately uses a 5-minute fixture so the first build is in flight
for several seconds — long enough that a SECOND open call (with a different
file) lands while the first runnable is still running. The N-5 distinct-
instance assertion (``runnable_b is not runnable_a``) proves the cancel-and-
restart cycle actually executed, even on fast hardware where a tiny fixture
would let the first build finish vacuously before the second open arrives.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from marmelade.audio import peak_builder, proxy_cache
from marmelade.audio.peak_builder_worker import PeakBuilderRunnable
from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


@pytest.fixture
def main_window(qtbot, qapp, tmp_cache_dir: Path):
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


@pytest.mark.slow
def test_double_open_cancels_first_then_starts_second(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """N-5: 5-min fixture A, then 30-s fixture B mid-build — distinct runnables."""
    # 5-minute fixture A — long enough that its build stays in flight while
    # the test wires up the second open call. Pinned per N-5 option 1.
    file_a = tmp_path / "long_a.wav"
    make_sine(
        file_a,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=300,  # 5 minutes — N-5 keeps the build window open
        sample_rate=44100,
        channels=1,
    )
    file_b = tmp_path / "short_b.wav"
    make_sine(
        file_b,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=30,
        sample_rate=44100,
        channels=1,
    )

    # First open: cache MISS for file A.
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(file_a), "")),
    )
    main_window._action_open_file()

    # Capture the in-flight runnable.
    runnable_a = main_window._current_runnable
    assert runnable_a is not None, "Expected a runnable to be in flight"
    assert isinstance(runnable_a, PeakBuilderRunnable)

    # Pre-register a render_complete latch BEFORE the second open. Post plan
    # 01-10 the cancel preamble no longer drains events inside `_open_file`;
    # B's worker for this 30-s fixture is fast enough that `render_complete`
    # can fire while qtbot's `waitSignal(cancelled).__exit__` is still
    # pumping events. A latch connected here captures the emission whenever
    # it lands, so a subsequent `waitUntil(latch.fired)` can't lose the
    # signal to a race.
    render_complete_fired = {"value": False}

    def _on_render_complete() -> None:
        render_complete_fired["value"] = True

    main_window.render_complete.connect(_on_render_complete)

    # Set up the cancelled-signal watcher BEFORE calling _action_open_file
    # for B. We capture `runnable_b` synchronously right after
    # `_action_open_file()` returns (still inside the with-block, before
    # __exit__'s drain lets B's quick worker finish and clear
    # `_current_runnable` via `_on_proxy_ready`).
    with qtbot.waitSignal(
        runnable_a.signals.cancelled, timeout=15000, raising=True
    ):
        # Second open: cache MISS for file B. Patch the dialog to return B.
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **kw: (str(file_b), "")),
        )
        # _open_file synchronously: cancels runnable_a, resets state, then
        # creates a fresh runnable for B and stores it on `_current_runnable`.
        main_window._action_open_file()

        # Confirm a distinct runnable replaced it BEFORE __exit__ drains
        # (B's build is quick on this 30-s fixture and would otherwise
        # complete and null out `_current_runnable` while we wait for
        # A's cancelled).
        runnable_b = main_window._current_runnable
        assert runnable_b is not None
        assert runnable_b is not runnable_a, (
            "Expected a fresh PeakBuilderRunnable for file B; got the same "
            "instance (cancel-and-restart did not actually happen — N-5 "
            "vacuous-pass guard)."
        )

    # File B finishes rendering within a generous timeout. Use the latch
    # so we don't race against an emission that may have already happened
    # during the cancel-drain above.
    qtbot.waitUntil(lambda: render_complete_fired["value"], timeout=60000)

    # Plan 01-09 caps the rendered point count at MAX_RENDER_PROXY_PAIRS,
    # so file-size no longer discriminates A from B at the y-array length.
    # Use _current_path as the direct semantic discriminator instead.
    assert main_window._current_path == file_b, (
        f"expected B loaded; got {main_window._current_path}"
    )
    items = main_window._waveform_view.plot_widget.plotItem.listDataItems()
    assert len(items) == 1
    _x, y = items[0].getData()
    from marmelade.ui.waveform_view import MAX_RENDER_PROXY_PAIRS
    expected_pairs = min(30 * 44100 // 256, MAX_RENDER_PROXY_PAIRS)
    actual_pairs = len(y) // 2
    assert abs(actual_pairs - expected_pairs) <= 2, (
        f"expected ~{expected_pairs} pairs (post-aggregation cap), "
        f"got {actual_pairs}"
    )


@pytest.mark.slow
def test_stale_finished_from_cancelled_worker_does_not_render_wrong_file(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CR-02 + CR-04 closure proof.

    A's `finished` signal arriving LATE (after we have moved on to B) must
    NOT overwrite B's rendered proxy. Two mechanisms close the gap:

    * CR-02: `_open_file`'s cancel-restart preamble disconnects our own
      finished/error/cancelled slot wirings from A's runnable BEFORE
      cancelling — so a stale terminal signal from A never reaches our
      slot at all.
    * CR-04: even if the disconnect missed (defence in depth), the
      generation-token + runnable-identity double-guard at the slot
      entry would drop the call silently.

    Scenario: A is a cache-MISS 5-min fixture (in-flight); B is a
    cache-HIT pre-warmed 30-s fixture. After B renders, we manually emit
    A's finished signal (mimicking a late delivery from the cancelled
    worker) and assert B's waveform stays put.
    """
    # 5-minute fixture A — long enough that A's build is still in flight
    # while we wire up the second open.
    file_a = tmp_path / "long_a.wav"
    make_sine(
        file_a,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=300,
        sample_rate=44100,
        channels=1,
    )
    # 30-second fixture B — short enough that it builds and renders
    # quickly, AND we pre-warm its cache so opening B is a cache HIT
    # (synchronous; no worker spawned for B).
    file_b = tmp_path / "short_b.wav"
    make_sine(
        file_b,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=30,
        sample_rate=44100,
        channels=1,
    )

    # Pre-warm B's cache so its open is a HIT (synchronous load_proxy).
    key_b = proxy_cache.cache_key(file_b)
    proxy_p_b = proxy_cache.proxy_path(default_cache_root(), key_b)
    proxy_p_b.parent.mkdir(parents=True, exist_ok=True)
    peak_builder.build_proxy(file_b, proxy_p_b, samples_per_pixel=256)

    # First open: cache MISS for A — runnable_a goes in flight.
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(file_a), "")),
    )
    main_window._action_open_file()

    runnable_a = main_window._current_runnable
    assert runnable_a is not None, "Expected a runnable to be in flight"
    assert isinstance(runnable_a, PeakBuilderRunnable)
    gen_a = main_window._open_generation
    assert gen_a >= 1, "generation token must have been incremented"

    # Second open: cache HIT for B. waitSignal arms the render_complete
    # watcher BEFORE the second click so the synchronous HIT-path emit
    # is captured.
    with qtbot.waitSignal(
        main_window.render_complete, timeout=15000, raising=True
    ):
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **kw: (str(file_b), "")),
        )
        main_window._action_open_file()

    # The CR-04 increment must have fired exactly once for B's open.
    assert main_window._open_generation == gen_a + 1, (
        f"expected generation to advance from {gen_a} to {gen_a + 1}; "
        f"got {main_window._open_generation}"
    )

    # B's waveform is on the view. Capture its length BEFORE the stale
    # emit so we can prove it didn't change.
    # Plan 01-09 caps the rendered point count at MAX_RENDER_PROXY_PAIRS,
    # so the discriminator is _current_path + the post-aggregation length
    # (not the raw proxy pair count).
    assert main_window._current_path == file_b, (
        f"expected B loaded; got {main_window._current_path}"
    )
    items = main_window._waveform_view.plot_widget.plotItem.listDataItems()
    assert len(items) == 1, "Exactly one PlotDataItem after B's HIT-path render"
    _x_b_before, y_b_before = items[0].getData()
    len_b = len(y_b_before)
    from marmelade.ui.waveform_view import MAX_RENDER_PROXY_PAIRS
    expected_pairs_b = min(30 * 44100 // 256, MAX_RENDER_PROXY_PAIRS)
    expected_y_b = expected_pairs_b * 2
    assert abs(len_b - expected_y_b) <= 4, (
        f"B should render ~{expected_y_b} y points (post-aggregation cap); "
        f"got {len_b}"
    )

    # CR-02 + CR-04 closure proof — manually fire a stale finished from
    # A's runnable, simulating a late delivery from the cancelled worker.
    # Since CR-02 disconnected our slot wiring from runnable_a's terminal
    # signals BEFORE cancelling it, the emit is a no-op at the slot level.
    # Even if the disconnect somehow missed, CR-04's generation guard at
    # the slot entry would drop the call silently. Either way, B's
    # waveform must not be overwritten with A's much larger 5-min payload.
    runnable_a.signals.finished.emit(runnable_a.dst_path)

    # Give Qt a chance to dispatch the (potentially-late-delivered) signal.
    qtbot.wait(100)

    # Re-read the waveform y array; if A's stale finished had been
    # processed, the array would have grown to ≈ 100 800 from A's
    # 5-min × 44100 // 256 × 2.
    items_after = main_window._waveform_view.plot_widget.plotItem.listDataItems()
    assert len(items_after) == 1, "Plot must still have exactly one PlotDataItem"
    _x_b_after, y_b_after = items_after[0].getData()
    assert len(y_b_after) == len_b, (
        f"CR-02 + CR-04 failure — B's waveform was overwritten by A's "
        f"stale finished signal: before={len_b}, after={len(y_b_after)}"
    )


@pytest.mark.slow
def test_reentrant_open_during_cancel_drain_is_dropped(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CR-05 closure proof — `_open_in_progress` gate drops re-entrant clicks.

    The simplest surgical proof of CR-05: when `_open_in_progress` is
    already True (mimicking the in-flight state during a cancel-and-drain
    spinloop), a fresh `_action_open_file()` call returns IMMEDIATELY
    without invoking QFileDialog. We assert that the dialog patch is not
    reached. The positive-control half (clearing the flag and seeing the
    dialog called) confirms the test harness works.

    A full timer-simulated double-click test under offscreen is non-
    deterministic (timer dispatch ordering vs. event-loop drain races);
    that variant is deferred to a follow-up. The flag-only assertion here
    directly proves the gate mechanism at the entry of `_action_open_file`.

    Decorated `@pytest.mark.slow` only because the `main_window` fixture
    is shared with the existing N-5 5-minute test that lives in the slow
    suite; this particular case finishes in milliseconds.
    """
    # Sentinel that counts how many times the QFileDialog patch was reached.
    dialog_calls = {"count": 0}

    def _track_dialog(*args, **kwargs):
        dialog_calls["count"] += 1
        # Return a no-op cancel — we only care that we got here.
        return ("", "")

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(_track_dialog),
    )

    # ---- CR-05 negative case: gate set → click is silently dropped ----
    main_window._open_in_progress = True
    main_window._action_open_file()
    assert dialog_calls["count"] == 0, (
        "CR-05 failure — re-entrant _action_open_file did NOT honour the "
        "_open_in_progress gate; the QFileDialog patch was reached "
        f"{dialog_calls['count']} time(s) when it should have been 0."
    )

    # ---- Positive control: gate cleared → dialog reached normally ----
    main_window._open_in_progress = False
    main_window._action_open_file()
    assert dialog_calls["count"] == 1, (
        f"Positive-control failure — expected 1 dialog call after "
        f"clearing the gate; got {dialog_calls['count']}. This means the "
        "test harness itself is wrong, not the CR-05 fix."
    )


@pytest.mark.slow
def test_miss_then_hit_clears_overlay_and_current_runnable(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REVIEW-GAPS CR-01 + CR-02 closure proof.

    After a MISS-then-HIT double-open completes:
    - `_overlay` is hidden (CR-01 was: overlay from A still visible over B)
    - `_current_runnable is None` (CR-02 was: stale A runnable still bound)

    Pre plan 01-10 the cache-HIT branch never called `_overlay.hide()` and
    never reset `_current_runnable`; the CR-02 disconnect + CR-04 generation
    guard combined to drop the cancelled-slot side effects that USED to do
    those resets, so the overlay sat on top of B's freshly-rendered waveform
    indefinitely. The unified cancel-preamble fix synchronously resets both.
    """
    # 5-minute fixture A — cache MISS, long enough that the worker is in
    # flight when we trigger the second open.
    file_a = tmp_path / "miss_hit_a.wav"
    make_sine(
        file_a,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=300,
        sample_rate=44100,
        channels=1,
    )
    # 30-second fixture B — pre-warmed so its open is a synchronous cache HIT.
    file_b = tmp_path / "miss_hit_b.wav"
    make_sine(
        file_b,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=30,
        sample_rate=44100,
        channels=1,
    )

    # Pre-build B's proxy so it's a cache HIT when opened.
    cache_root = default_cache_root()
    key_b = proxy_cache.cache_key(file_b)
    proxy_p_b = proxy_cache.proxy_path(cache_root, key_b)
    proxy_p_b.parent.mkdir(parents=True, exist_ok=True)
    peak_builder.build_proxy(file_b, proxy_p_b, samples_per_pixel=256)
    assert proxy_p_b.exists()

    # Open A (MISS, will start a long worker).
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(file_a), "")),
    )
    main_window._action_open_file()

    # Wait for A's overlay to appear (proves the MISS worker actually
    # started — the overlay is shown after cache lookup, on the MISS branch).
    qtbot.waitUntil(lambda: main_window._overlay.isVisible(), timeout=2000)
    runnable_a = main_window._current_runnable
    assert runnable_a is not None, "Expected A's runnable to be in flight"

    # Open B (HIT) — this triggers the cancel-preamble for A and then runs
    # the synchronous load_proxy + render path.
    with qtbot.waitSignal(
        main_window.render_complete, timeout=10000, raising=True
    ):
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **kw: (str(file_b), "")),
        )
        main_window._action_open_file()

    # CR-01: overlay must be hidden after B's HIT render completes.
    assert not main_window._overlay.isVisible(), (
        "REVIEW-GAPS CR-01 regression — overlay still visible after "
        "MISS→HIT double-open completed"
    )

    # CR-02: stale runnable must be cleared.
    assert main_window._current_runnable is None, (
        "REVIEW-GAPS CR-02 regression — stale _current_runnable not cleared "
        "after MISS→HIT; got "
        f"{type(main_window._current_runnable).__name__}"
    )


@pytest.mark.slow
def test_third_open_after_miss_hit_is_fast(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REVIEW-GAPS WR-01 closure proof.

    After MISS→HIT, the third open (also a HIT) must complete quickly —
    well under the pre-fix 2-second drain cost. Pre plan 01-10 the
    cancel-preamble for the second open paid a 2-second drain timeout
    because `_current_runnable` was never cleared on the HIT branch, so
    the third open re-entered the preamble against a dead worker and
    spun the QEventLoop to its 2 s safety bound.
    """
    import time

    file_a = tmp_path / "wr01_a.wav"
    make_sine(
        file_a,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=300,
        sample_rate=44100,
        channels=1,
    )
    file_b = tmp_path / "wr01_b.wav"
    make_sine(
        file_b,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=30,
        sample_rate=44100,
        channels=1,
    )
    file_c = tmp_path / "wr01_c.wav"
    make_sine(
        file_c,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=20,
        sample_rate=44100,
        channels=1,
    )

    cache_root = default_cache_root()
    for f in (file_b, file_c):
        key = proxy_cache.cache_key(f)
        pp = proxy_cache.proxy_path(cache_root, key)
        pp.parent.mkdir(parents=True, exist_ok=True)
        peak_builder.build_proxy(f, pp, samples_per_pixel=256)

    # 1. Open A (MISS).
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(file_a), "")),
    )
    main_window._action_open_file()
    qtbot.waitUntil(lambda: main_window._overlay.isVisible(), timeout=2000)

    # 2. Open B (HIT, cancels A).
    with qtbot.waitSignal(
        main_window.render_complete, timeout=10000, raising=True
    ):
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **kw: (str(file_b), "")),
        )
        main_window._action_open_file()

    # 3. Open C — measure wall-clock. Must NOT pay the 2 s drain cost.
    #    Pre-fix this was ~2000 ms; post-fix it should be <500 ms even on
    #    slow CI hardware.
    t0 = time.monotonic()
    with qtbot.waitSignal(
        main_window.render_complete, timeout=5000, raising=True
    ):
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **kw: (str(file_c), "")),
        )
        main_window._action_open_file()
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < 500, (
        "REVIEW-GAPS WR-01 regression — third open took "
        f"{elapsed_ms:.0f} ms; expected < 500 ms (no 2 s drain). "
        "The drain spinloop must remain removed."
    )
