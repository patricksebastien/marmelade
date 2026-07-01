"""Quick-260615-l4y — two-button Master / Export batch flow.

End-to-end MAS-05 pins for the split-button redesign:

* Master All — sequential MasteringRunnable loop on the global
  QThreadPool. The Master button reads "Cancel mastering" while the loop
  runs and reverts to "Master All Keepers" (idle) when it finishes.
* Export gate — once every keeper has a fresh mastered cache the
  persistent Export button enables (same freshness probe the bundle
  button uses).
* Export All — passes the mastered cache path to
  ``export_region(source_path=...)`` for mastered keepers; uses the
  source proxy otherwise. The batch loop advances across all keepers via
  the ``_export_all_in_flight`` sentinel.
* Cancel-restart during the master loop — 3-layer pattern (D-08 + D-18):
  gen bump, runnable cancel, queue cleared, Master reverts to idle.

For speed: tests use TINY mastering configs (default Limiter only,
no Matchering) on TINY keeper regions (~0.5 s each) so the actual
mastering pass completes in <100 ms per keeper. The orchestration is
what's under test — DSP correctness is pinned elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QSettings, QThreadPool
from PySide6.QtWidgets import QApplication, QDialog

from marmelade.audio import sidecar_cache
from marmelade.audio.mastering.chain import (
    config_hash,
    load_session_chain_snapshot,
)
from marmelade.audio.mastering_cache import (
    is_mastered_cache_fresh,
    mastered_cache_path,
)
from marmelade.audio.proxy_cache import cache_key as proxy_cache_key
from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


SR = 44100


def _make_source_wav(path: Path, seconds: float = 3.0) -> Path:
    """Tiny 44.1 kHz stereo float32 WAV used as the open file."""
    n = int(seconds * SR)
    # Loud-ish sine — gives the limiter something to do.
    t = np.arange(n, dtype=np.float64) / SR
    mono = (0.9 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    audio = np.stack([mono, mono], axis=1)
    sf.write(str(path), audio, SR, subtype="FLOAT", format="WAV")
    return path


def _seed_keepers_with_default_session(
    window: MainWindow, keeper_specs: list[tuple[float, float, bool]]
) -> list[Region]:
    """Inject regions with state=keeper + optional mastering snapshot.

    Args:
        keeper_specs: list of (start_sec, end_sec, has_mastering) tuples.

    Returns:
        The list of Regions that were injected.
    """
    snapshot = load_session_chain_snapshot()
    regions: list[Region] = []
    # Build 32-hex keeper_ids — mastered_cache_path validates these
    # against ^[0-9a-f]{32}$ per T-7-02. The shape mirrors uuid4().hex.
    HEX = "0123456789abcdef"
    for i, (start, end, has_mastering) in enumerate(keeper_specs):
        # Distinguishable hex IDs.
        prefix = HEX[i % 16] * 16
        suffix = HEX[(i + 1) % 16] * 16
        regions.append(
            Region(
                id=prefix + suffix,
                start_sec=start,
                end_sec=end,
                state="keeper",
                note="",
                mastering=snapshot if has_mastering else None,
            )
        )
    window._regions_overlay.set_regions(regions)
    window._on_regions_changed()
    return regions


def _open_window_with_source(
    qtbot, qapp, tmp_path: Path, *, seconds: float = 6.0
) -> tuple[MainWindow, Path]:
    """Construct MainWindow, open a synthetic source, return (window, src).

    Plan 07-08 — default bumped from 3.0 s → 6.0 s so a 0.5-s keeper
    region is discriminating: a buggy build that masters the full
    source produces a 6-s cache, the fix produces a 0.5-s cache.
    """
    theme.apply_theme(QApplication.instance())
    src = _make_source_wav(tmp_path / "src.wav", seconds=seconds)
    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None
        and window._current_playback_path is not None,
        timeout=15000,
    )
    return window, src


# =========================================================================
# Pin 1 — Disabled state with zero keepers
# =========================================================================
def test_batch_button_disabled_when_no_keepers(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """MainWindow opens a file but no keepers exist — batch button disabled."""
    window, _ = _open_window_with_source(qtbot, qapp, tmp_path)
    assert not window._keepers_sidebar._batch_button.isEnabled()


# =========================================================================
# Pin 2 — Idle click starts mastering immediately (no modal)
# =========================================================================
def test_batch_button_click_starts_mastering_without_modal(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """Click on idle "Master All Keepers" → mastering kicks off, no QDialog opens.

    The output-folder / format modal only appears when the user clicks
    the separate Export button (see ``_on_export_all_requested``). The
    mastering pass itself runs friction-free.
    """
    window, _ = _open_window_with_source(qtbot, qapp, tmp_path)
    _seed_keepers_with_default_session(
        window, [(0.1, 0.6, True), (1.0, 1.5, True)]
    )

    invocations: list[QDialog] = []

    original_exec = QDialog.exec

    def fake_exec(self: QDialog) -> int:
        invocations.append(self)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    try:
        window._keepers_sidebar._batch_button.click()
    finally:
        monkeypatch.setattr(QDialog, "exec", original_exec)

    # No modal opened on the mastering click.
    assert invocations == []
    # Master button transitioned into the running (cancel) state.
    assert window._keepers_sidebar._batch_state == "running"


# =========================================================================
# Pin 3 — Phase A sequential render (the core happy path)
# =========================================================================
def test_phase_a_renders_all_keepers_sequentially(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """3 keepers, default-Limiter chain → 3 mastered cache files appear sequentially.

    Each MasteringRunnable emits a ``mastering_complete(kid)`` signal
    on the MainWindow. We collect 3 emissions, assert the cache file
    exists for each, AND (Plan 07-08) assert the cache frame count
    matches the keeper region — NOT the full source.
    """
    window, src = _open_window_with_source(qtbot, qapp, tmp_path)
    # Plan 07-08 — keepers at non-trivial offsets in [0.5, 4.5] of a
    # 6-s source. Each region is 0.5 s long → cache should be ~22050
    # frames at 44.1 kHz, NOT 264600 (6 * SR).
    regions = _seed_keepers_with_default_session(
        window,
        [
            (0.5, 1.0, True),
            (2.0, 2.5, True),
            (4.0, 4.5, True),
        ],
    )

    emitted_kids: list[str] = []
    window.mastering_complete.connect(emitted_kids.append)

    # Drive Phase A entry — skip the confirmation dialog by calling the
    # post-confirmation kickoff directly. This is the production seam
    # the dialog's accept path calls into.
    window._kickoff_master_all(
        target_dir=tmp_path / "out", fmt="wav"
    )

    qtbot.waitUntil(lambda: len(emitted_kids) == 3, timeout=60000)
    assert set(emitted_kids) == {r.id for r in regions}

    # Every cache file exists.
    src_key = proxy_cache_key(src)
    for r in regions:
        chash = config_hash(r.mastering)
        cache_p = mastered_cache_path(
            default_cache_root(), src_key, r.id, chash
        )
        assert is_mastered_cache_fresh(cache_p), (
            f"Cache file missing for keeper {r.id}: {cache_p}"
        )

    # Plan 07-08 — per-region frame-count pin. Discriminating: source
    # is 6 s, each keeper is 0.5 s. A buggy build masters the full
    # source → cache frames ≈ 264600. The fix masters only the keeper
    # region → cache frames ≈ 22050 (±2048 for limiter lookahead).
    for r in regions:
        chash = config_hash(r.mastering)
        cache_p = mastered_cache_path(
            default_cache_root(), src_key, r.id, chash
        )
        actual = sf.info(str(cache_p)).frames
        expected = int((r.end_sec - r.start_sec) * SR)
        assert abs(actual - expected) <= 2048, (
            f"Cache for keeper {r.id} has {actual} frames, expected "
            f"{expected} (±2048 for limiter lookahead). Source is 6 s; "
            f"if actual ≈ 264600 the region-bound fix did not land — "
            f"see Plan 07-08."
        )


# =========================================================================
# Pin 4 — Master complete reverts Master to idle AND opens the Export gate
# =========================================================================
def test_master_complete_reverts_to_idle_and_enables_export(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """After all keepers Ready — Master button reads "Master All Keepers"
    (idle) AND the Export button enables (freshness gate opened)."""
    window, _ = _open_window_with_source(qtbot, qapp, tmp_path)
    regions = _seed_keepers_with_default_session(
        window, [(0.05, 0.55, True), (1.0, 1.5, True)]
    )
    emitted: list[str] = []
    window.mastering_complete.connect(emitted.append)
    window._kickoff_master_all(target_dir=tmp_path / "out", fmt="wav")
    qtbot.waitUntil(lambda: len(emitted) == 2, timeout=60000)
    # Drain queued slots (the QTimer.singleShot to _on_master_all_complete).
    qtbot.waitUntil(
        lambda: window._keepers_sidebar._batch_state == "idle",
        timeout=10000,
    )
    QApplication.processEvents()
    assert (
        window._keepers_sidebar._batch_button.text() == "Master All Keepers"
    )
    # The freshness probe (installed by MainWindow) now sees fresh caches
    # for every keeper → the persistent Export button is enabled.
    assert window._keepers_sidebar._export_button.isEnabled(), (
        "Export button must enable once every keeper has a fresh "
        "mastered cache"
    )


# =========================================================================
# Pin 5 — Export uses source_path for mastered keepers, source proxy otherwise
# =========================================================================
def test_export_all_uses_mastered_cache_via_source_path(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """Mixed keepers (mastering set vs None) — export_region receives the right source_path."""
    window, src = _open_window_with_source(qtbot, qapp, tmp_path)
    regions = _seed_keepers_with_default_session(
        window,
        [
            (0.05, 0.55, True),  # mastered → expect source_path = cache
            (1.0, 1.5, False),  # no mastering → expect source_path = None
        ],
    )

    emitted: list[str] = []
    window.mastering_complete.connect(emitted.append)
    target_dir = tmp_path / "out"
    window._kickoff_master_all(target_dir=target_dir, fmt="wav")
    qtbot.waitUntil(lambda: len(emitted) >= 1, timeout=60000)
    QApplication.processEvents()

    # Now invoke Phase C — intercept the export_region call.
    captured_calls: list[dict[str, Any]] = []
    import marmelade.ui.main_window as mw

    real_spawn = mw.MainWindow._spawn_export_worker

    def capture_spawn(self, **kwargs):
        captured_calls.append(kwargs)
        # Do not actually run the export — invoke the finished slot
        # directly so the sequential loop can progress.
        # Emit through the public signal mechanism: simulate immediate
        # success by calling our finished slot.
        self.export_complete.emit(str(kwargs["dst_path"]))

    monkeypatch.setattr(
        mw.MainWindow, "_spawn_export_worker", capture_spawn
    )

    # Connect a counter that increments on export_complete so we can
    # wait for Phase C to fan out all keepers.
    finished_paths: list[str] = []
    window.export_complete.connect(finished_paths.append)

    window._on_export_all_requested()
    qtbot.waitUntil(lambda: len(captured_calls) == 2, timeout=10000)

    # The mastered keeper's spawn must have source_path set.
    mastered = next(
        c
        for c in captured_calls
        if Path(c["dst_path"]).stem.startswith(
            # Whatever the naming_resolver chose, both keepers will have
            # a similar shape — we identify by which one has source_path.
            ""
        )
        and c.get("source_path") is not None
    )
    unmastered = next(
        c
        for c in captured_calls
        if c.get("source_path") is None
    )
    assert mastered is not None
    assert unmastered is not None

    # The mastered-keeper spawn's source_path must point at the
    # mastered cache file for that keeper.
    src_key = proxy_cache_key(src)
    mastered_region = regions[0]
    chash = config_hash(mastered_region.mastering)
    expected_cache = mastered_cache_path(
        default_cache_root(), src_key, mastered_region.id, chash
    )
    assert Path(mastered["source_path"]) == expected_cache


# =========================================================================
# Pin 6 — Cancel during Phase A
# =========================================================================
def test_cancel_during_master_clears_queue_and_returns_to_idle(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Master loop in flight, click Cancel → Master button reverts to idle.

    We seed a queue of 3 keepers and call cancel after the first
    mastering_complete fires; the remaining 2 must be skipped.
    """
    window, _ = _open_window_with_source(qtbot, qapp, tmp_path)
    regions = _seed_keepers_with_default_session(
        window,
        [
            (0.05, 0.55, True),
            (1.0, 1.5, True),
            (2.0, 2.5, True),
        ],
    )

    completed: list[str] = []
    window.mastering_complete.connect(completed.append)

    window._kickoff_master_all(target_dir=tmp_path / "out", fmt="wav")
    # Wait for the first keeper to finish.
    qtbot.waitUntil(lambda: len(completed) >= 1, timeout=30000)

    # Click cancel — orchestrator should clear the queue and revert the
    # Master button to idle on the next event loop.
    window._on_master_all_cancel_requested()
    QThreadPool.globalInstance().waitForDone(60000)
    QApplication.processEvents()

    # Quick-260615-l4y — cancel always reverts the Master button to idle
    # (the morphing phase_b "Export N" state no longer exists).
    assert window._keepers_sidebar._batch_state == "idle", (
        f"After cancel, Master button must be idle "
        f"(got {window._keepers_sidebar._batch_state!r})"
    )
    # Fewer than 3 keepers completed (cancel arrived during the queue).
    assert len(completed) < 3, (
        f"Cancel must skip not-yet-started keepers "
        f"(got {len(completed)} completed, expected < 3)"
    )


# =========================================================================
# Pin 7 — Failure in one keeper does not halt the others (Phase A continues)
# =========================================================================
def test_failure_in_one_keeper_does_not_halt_others(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """One keeper's mastering chain raises; the other two still complete Phase A."""
    window, _ = _open_window_with_source(qtbot, qapp, tmp_path)
    regions = _seed_keepers_with_default_session(
        window,
        [
            (0.05, 0.55, True),
            (1.0, 1.5, True),
            (2.0, 2.5, True),
        ],
    )

    # Monkeypatch MasteringChain.process so it raises for the MIDDLE keeper
    # only. We identify "middle" via call count — the orchestrator runs
    # the queue in chronological order.
    import marmelade.audio.mastering.chain as chain_mod

    real_process = chain_mod.MasteringChain.process
    call_count = {"n": 0}

    def maybe_fail(self, audio, sr):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("synthetic mastering failure")
        return real_process(self, audio, sr)

    monkeypatch.setattr(chain_mod.MasteringChain, "process", maybe_fail)

    completed: list[str] = []
    window.mastering_complete.connect(completed.append)

    window._kickoff_master_all(target_dir=tmp_path / "out", fmt="wav")
    # The master loop finishes when the Master button reverts to idle
    # (all keepers processed — success + failure).
    qtbot.waitUntil(
        lambda: window._keepers_sidebar._batch_state == "idle",
        timeout=60000,
    )
    QApplication.processEvents()

    # Exactly 2 keepers reported success; the failed one did NOT emit
    # mastering_complete.
    assert len(completed) == 2, (
        f"Expected 2 successful mastering_complete emissions, "
        f"got {len(completed)} (call_count = {call_count['n']})"
    )


# =========================================================================
# Plan 07-08 Task 2 Part B-0 — legacy single-keeper spawn passes the
# AUDIO source path (_current_playback_path), NOT the peak-builder's
# peaks.dat binary (_current_proxy_p). This was a pre-existing bug from
# Plan 07-02 that Plan 07-06's source-path unification missed at the
# legacy site.
# =========================================================================


def _divergent_cfg() -> dict:
    """Limiter-only config with a non-default ceiling — produces a different
    config_hash from ``load_session_chain_snapshot()`` so Apply triggers a
    real render (needs_render = True instead of the hash-match short-circuit).
    """
    return {
        "highpass": {"enabled": False, "cutoff_hz": 30.0},
        "lowpass": {"enabled": False, "cutoff_hz": 18000.0},
        "eq": {"enabled": False, "low_db": 0.0, "mid_db": 0.0, "high_db": 0.0},
        "compressor": {
            "enabled": False,
            "threshold_db": -18.0,
            "ratio": 2.0,
            "attack_ms": 30.0,
            "release_ms": 200.0,
        },
        # Divergent: ceiling -2.0 (default is -1.0).
        "limiter": {
            "enabled": True, "ceiling_dbtp": -2.0, "release_ms": 100.0
        },
        "matchering": {"enabled": False, "reference_path": ""},
    }


def test_legacy_single_keeper_master_button_uses_audio_path_not_peaks_dat(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """Legacy spawn passes the audio WAV path to MasteringRunnable, not peaks.dat.

    Pre-fix: ``_on_mastering_config_applied`` constructs
    ``MasteringRunnable(self._current_proxy_p, ...)`` where
    ``_current_proxy_p`` is the peak-builder's ``peaks.dat`` binary
    (NOT audio). This test FAILS on HEAD because the spy captures the
    peaks.dat path.

    Post-fix: legacy spawn matches the Phase A site — passes
    ``_current_playback_path`` (the audio source WAV).
    """
    window, src = _open_window_with_source(qtbot, qapp, tmp_path)
    # Seed ONE keeper at (2.0, 3.5) — has_mastering=True. The legacy
    # auto-snapshot at _on_regions_changed gives every keeper the
    # default session chain. We then Apply a DIVERGENT config so
    # needs_render = True (previous_hash != new_hash) and the spawn
    # branch is reached.
    regions = _seed_keepers_with_default_session(
        window, [(2.0, 3.5, True)]
    )
    keeper = regions[0]

    # Spy on MasteringRunnable construction. We patch the symbol that
    # main_window.py uses (the from-import binding) so the slot picks
    # up the spy.
    import marmelade.ui.main_window as mw

    real_runnable_cls = mw.MasteringRunnable
    captured_src_args: list[str] = []

    class SpyRunnable(real_runnable_cls):
        def __init__(self, src_proxy_path, *args, **kwargs):
            captured_src_args.append(str(src_proxy_path))
            super().__init__(src_proxy_path, *args, **kwargs)

    monkeypatch.setattr(mw, "MasteringRunnable", SpyRunnable)

    # Drive the legacy slot: Apply a divergent mastering config.
    divergent = _divergent_cfg()
    window._on_mastering_config_applied(keeper.id, divergent)

    # Construction is synchronous within the slot — drain pending
    # events for good measure.
    QApplication.processEvents()

    assert captured_src_args, (
        "MasteringRunnable was NOT constructed by the legacy slot. "
        "Either the slot's no-source guard rejected the call, or the "
        "needs_render branch returned early because the cache HIT "
        "fired."
    )
    captured = captured_src_args[0]
    expected_audio = str(window._current_playback_path)
    forbidden_peaks = str(window._current_proxy_p)

    assert captured == expected_audio, (
        f"Legacy spawn passed src_proxy_path={captured!r} but expected "
        f"{expected_audio!r} (_current_playback_path — the AUDIO WAV). "
        f"Plan 07-06's source-path unification missed this site."
    )
    assert captured != forbidden_peaks, (
        f"Legacy spawn passed _current_proxy_p={forbidden_peaks!r} "
        f"which is the peak-builder's peaks.dat binary, NOT audio. "
        f"This has been producing garbage masters since Plan 07-02."
    )


# =========================================================================
# Plan 07-08 Task 2 Part B — legacy single-keeper spawn forwards the
# keeper region as start_frame / end_frame kwargs.
# =========================================================================


def test_legacy_single_keeper_master_button_renders_region_bounded(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Legacy Master button on a 6-s source with one keeper at (2.0, 3.5)
    produces a mastered cache with frames ≈ 1.5*SR (±2048), NOT 6*SR.
    """
    window, src = _open_window_with_source(qtbot, qapp, tmp_path)
    # Seed ONE keeper at (2.0, 3.5) — has_mastering=True (auto-snapshot
    # gives the default session chain). Apply a DIVERGENT config so
    # needs_render = True and the spawn branch is reached.
    regions = _seed_keepers_with_default_session(
        window, [(2.0, 3.5, True)]
    )
    keeper = regions[0]

    divergent = _divergent_cfg()

    # The slot emits mastering_complete on the MainWindow once the
    # runnable finishes. We wait on that signal.
    with qtbot.waitSignal(window.mastering_complete, timeout=30000):
        window._on_mastering_config_applied(keeper.id, divergent)

    src_key = proxy_cache_key(src)
    chash = config_hash(divergent)
    cache_p = mastered_cache_path(
        default_cache_root(), src_key, keeper.id, chash
    )
    assert cache_p.exists(), f"Cache file missing: {cache_p}"

    info = sf.info(str(cache_p))
    expected = int((keeper.end_sec - keeper.start_sec) * SR)
    assert abs(info.frames - expected) <= 2048, (
        f"Legacy-spawn cache for keeper {keeper.id} has {info.frames} "
        f"frames, expected {expected} (±2048 for limiter lookahead). "
        f"Source is 6 s; if actual ≈ {6*SR} the legacy spawn did NOT "
        f"forward keeper region frames — see Plan 07-08 Part B."
    )
