"""Phase 7 Plan 07-04 Task 2 (RED) — A/B switch behavior + fail-closed.

Pins the live A/B switch via :meth:`PlaybackEngine.play(path, start_seconds=)`:

* Press B during playback → ``play(mastered_cache_path, start_seconds≈pos)``.
* Press A from B → ``play(proxy_path, start_seconds≈pos)``.
* Position preserved within one playback block (~50 ms tolerance).
* Fail-closed (T-7-05) — if the mastered cache file is missing/corrupt
  at switch time, revert toggle to A, show status-bar toast, NO
  ``play(...)`` call.
* STATE-KEY no-op — pressing A while already on A does NOT call play().
* Modal dialog suppression — A/B shortcuts inactive while MasteringDialog
  is shown modally.

PlaybackEngine.sounddevice is mocked at the boundary (same pattern as
``test_playback.py``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from marmelade.audio.mastering.chain import (
    _SESSION_DEFAULTS,
    config_hash,
)
from marmelade.audio.mastering_cache import mastered_cache_path
from marmelade.audio.proxy_cache import cache_key
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


_TOAST_CACHE_MISSING = (
    "Mastered preview unavailable — cache is missing. "
    "Re-master this keeper."
)


# ----------------------------------------------------------------- fixtures
def _make_proxy_wav(tmp_path: Path, seconds: float = 2.0) -> Path:
    sr = 44100
    n = int(seconds * sr)
    audio = (np.random.RandomState(0).randn(n, 2) * 0.05).astype("float32")
    p = tmp_path / "proxy.wav"
    sf.write(str(p), audio, sr, subtype="FLOAT", format="WAV")
    return p


def _write_fake_mastered_cache(
    cache_root: Path,
    src_key: str,
    keeper_id: str,
    cfg: dict,
) -> Path:
    chash = config_hash(cfg)
    dst = mastered_cache_path(cache_root, src_key, keeper_id, chash)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sr = 44100
    sf.write(str(dst), np.zeros((sr, 2), dtype="float32"), sr,
             subtype="PCM_24", format="WAV")
    return dst


@pytest.fixture
def mocked_sounddevice(monkeypatch):
    fake_stream = MagicMock()
    fake_ctor = MagicMock(return_value=fake_stream)
    monkeypatch.setattr(
        "marmelade.audio.playback.sd.OutputStream", fake_ctor
    )
    monkeypatch.setattr(
        "marmelade.audio.playback._SOUNDDEVICE_AVAILABLE", True
    )
    return fake_ctor, fake_stream


@pytest.fixture
def primed_window_with_keeper(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
):
    """Open a file, create one keeper with default mastering + fresh cache."""
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None
        and window._current_proxy_p is not None,
        timeout=15000,
    )
    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None
    overlay.set_state(region.id, "keeper")
    cfg = copy.deepcopy(_SESSION_DEFAULTS)
    overlay.set_mastering(region.id, cfg)

    # Drop a placeholder mastered cache in place so the toggle enables.
    src_key = cache_key(window._current_path)
    cache_path = _write_fake_mastered_cache(
        default_cache_root(), src_key, region.id, cfg
    )

    # Select the keeper.
    window._on_keeper_selection_changed(region.id)
    assert window._ab_toggle.isEnabled() is True

    # quick-260629 — selecting a keeper with a fresh mastered cache now
    # auto-switches the toggle to B. The swap-mechanics tests below deliberately
    # exercise transitions starting from A, so reset to a known A baseline here;
    # the auto-switch-on-selection behaviour itself is covered by
    # test_select_keeper_with_cache_auto_switches_to_b / _without_cache_*.
    window._ab_toggle.set_state("A")
    assert window._ab_toggle.state == "A"

    return window, region, cache_path


# =========================================================================
# quick-260629 — selecting a keeper auto-switches the A/B preview
# =========================================================================
def test_select_keeper_with_cache_auto_switches_to_b(
    primed_window_with_keeper, qtbot, mocked_sounddevice
) -> None:
    """Selecting a keeper with a fresh mastered cache auto-switches to B."""
    window, region, cache_path = primed_window_with_keeper
    # Fixture reset the toggle to A; re-selecting the mastered keeper must
    # auto-switch the preview to mastered (B).
    assert window._ab_toggle.state == "A"
    window._on_keeper_selection_changed(region.id)
    assert window._ab_toggle.is_enabled is True
    assert window._ab_toggle.state == "B", (
        "selecting a keeper with a fresh mastered cache should auto-switch "
        f"the A/B preview to B; got {window._ab_toggle.state!r}"
    )


def test_select_keeper_without_mastering_reverts_to_a(
    primed_window_with_keeper, qtbot, mocked_sounddevice
) -> None:
    """Selecting a keeper that has NO mastering reverts the preview to A."""
    window, region, cache_path = primed_window_with_keeper
    # First land on B via the mastered keeper.
    window._on_keeper_selection_changed(region.id)
    assert window._ab_toggle.state == "B"

    # Add a second keeper with NO mastering; selecting it reverts to A and
    # disables the toggle.
    overlay = window._regions_overlay
    overlay.start_draft(0.6)
    overlay.update_draft(0.9)
    r2 = overlay.commit_draft(0.9)
    assert r2 is not None
    overlay.set_state(r2.id, "keeper")
    overlay.set_mastering(r2.id, None)

    window._on_keeper_selection_changed(r2.id)
    assert window._ab_toggle.is_enabled is False
    assert window._ab_toggle.state == "A", (
        "selecting a keeper without mastering should revert the A/B preview "
        f"to A; got {window._ab_toggle.state!r}"
    )


# =========================================================================
# Test 1 — Press B during playback swaps source to mastered cache
# =========================================================================
def test_b_press_switches_source_to_mastered_cache(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """B-press calls ``engine.play(mastered_cache_path, start_seconds=translated)``.

    Plan 07-10: cache files are keeper-bounded (Plan 07-08), so the source
    timeline position must be translated to the cache timeline by subtracting
    ``keeper.start_sec``. Fixture keeper is (0.1, 0.5); source position is
    0.5 → cache position should be 0.4 (0.5 - 0.1).
    """
    window, region, cache_path = primed_window_with_keeper

    # Force engine into a playing state with a known position.
    window._playback_engine._is_playing = True
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 0.5),
    )
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    # Press B via the widget (equivalent to clicking or B shortcut).
    window._ab_toggle.set_state("B")

    assert play_spy.called, "Press-B during playback must call engine.play"
    call = play_spy.call_args
    # First positional arg is the path string. It should match the
    # mastered cache path.
    assert call.args[0] == str(cache_path)
    # start_seconds is the SOURCE position translated to CACHE timeline.
    # Source 0.5 - keeper.start_sec 0.1 = cache 0.4 s.
    actual_start = call.kwargs.get("start_seconds", 0.0)
    assert actual_start == pytest.approx(0.4, abs=0.05), (
        f"B-press must translate source-timeline position to cache timeline "
        f"by subtracting keeper.start_sec (0.1). Got start_seconds={actual_start!r}, "
        f"expected ≈0.4. If this is ≈0.5, the code is passing the raw source "
        f"position to engine.play(cache) — engine.seek will fail with ValueError "
        f"when cache_duration < position (the user-reported 07-10 bug)."
    )


# =========================================================================
# Test 2 — Press A from state B switches back to source
# =========================================================================
def test_a_press_switches_back_to_source(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """From B-active, press A → engine.play(source_proxy_path, start_seconds≈pos)."""
    window, region, cache_path = primed_window_with_keeper

    window._playback_engine._is_playing = True
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 0.5),
    )
    # First press B to enter state B.
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)
    window._ab_toggle.set_state("B")
    assert play_spy.called
    play_spy.reset_mock()

    # Now press A — should play the AUDIO source proxy (_current_playback_path),
    # NOT the peak-builder binary (_current_proxy_p which is peaks.dat).
    # Same source-path rule Plan 07-08 enforced for mastering spawns:
    # _current_playback_path is the decodable audio file; _current_proxy_p
    # is the visual peaks.dat binary and is NOT a valid playback target.
    window._ab_toggle.set_state("A")
    assert play_spy.called
    call = play_spy.call_args
    assert call.args[0] == str(window._current_playback_path), (
        f"A-press must play _current_playback_path (audio), got "
        f"{call.args[0]!r}. If this matches _current_proxy_p "
        f"({str(window._current_proxy_p)!r}), the engine is being asked to "
        f"play the peaks.dat binary — silent failure or PlaybackError."
    )
    # Defense: explicitly assert peaks.dat is NOT the target.
    assert call.args[0] != str(window._current_proxy_p), (
        "A-press target equals peaks.dat — peaks.dat is a peak-builder "
        "binary, not audio. Audio swap will silently fail."
    )


# =========================================================================
# Test 3 — Position preserved across the switch (within one block)
# =========================================================================
def test_position_preserved_within_one_block(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """B-press at source pos=0.5s → engine.play(cache, start_seconds=0.4±50ms).

    Plan 07-10: position is translated source→cache by subtracting
    keeper.start_sec (0.1 in fixture). The 50 ms tolerance is one
    PlaybackEngine block at 44.1 kHz (~46 ms).
    """
    window, region, cache_path = primed_window_with_keeper
    window._playback_engine._is_playing = True
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 0.5),
    )
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    window._ab_toggle.set_state("B")
    call = play_spy.call_args
    start = call.kwargs.get("start_seconds", call.args[1] if len(call.args) > 1 else None)
    # Plan 07-10: cache-timeline = source-timeline (0.5) - keeper.start_sec (0.1) = 0.4
    assert abs(start - 0.4) < 0.05


# =========================================================================
# Test 3b (Plan 07-10) — clamp when source position is OUTSIDE keeper region
# =========================================================================
def test_b_press_clamps_when_source_position_outside_keeper(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """User-reported 07-10 symptom: pressing B with playhead far past keeper.

    Stack-trace evidence: ``af.seek(start_frame)`` raised
    ``ValueError: Cannot seek to position 20391872 frames, which is beyond
    end of file (2717371 frames) by -17674501 frames``. Caused by passing
    raw source-timeline position to engine.play(cache) when cache is
    keeper-bounded (Plan 07-08).

    Contract: when source position is past ``keeper.end_sec``, the cache
    start position must be clamped to the keeper duration (NOT seek past EOF).
    Below ``keeper.start_sec`` clamps to 0.
    """
    window, region, cache_path = primed_window_with_keeper
    # Keeper region is (0.1, 0.5) per fixture; source position far past end.
    window._playback_engine._is_playing = True
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 100.0),  # 100 s into source, ~10 minutes past keeper.end
    )
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    window._ab_toggle.set_state("B")
    call = play_spy.call_args
    start = call.kwargs.get("start_seconds", 0.0)
    keeper_duration = 0.5 - 0.1  # 0.4 s
    assert 0.0 <= start <= keeper_duration + 0.05, (
        f"Source pos=100s far past keeper region (0.1-0.5s) must clamp to "
        f"[0, keeper_duration={keeper_duration}], got start_seconds={start!r}. "
        f"Unclamped raw source position would trip engine.seek ValueError "
        f"(the user-reported 07-10 symptom)."
    )


# =========================================================================
# Test 3c (Plan 07-10c) — visual playhead stays anchored to source timeline
#                          when B-state cache plays from cache timeline
# =========================================================================
def test_playback_tick_translates_cache_position_to_source_for_playhead(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """In B-state, playhead displays source-timeline (not cache-timeline).

    Plan 07-08 made cache files keeper-bounded. When the engine plays the
    cache, ``engine.position_seconds`` reports the position relative to the
    cache file (0..keeper_duration). The visual playhead InfiniteLine is
    placed on the source-timeline waveform plot — so without translation it
    jumps to 0..keeper_duration on the source waveform, which is NOT where
    the audio is sounding in the source.

    Contract: when toggle state is "B" and a keeper is selected,
    _on_playback_tick must translate by adding keeper.start_sec so the
    playhead displays at source-time (keeper.start_sec + cache_position).

    User report: "playhead when pressing B start from 0 + current playtime
    instead of segment beginning + current playtime".
    """
    window, region, cache_path = primed_window_with_keeper
    # Keeper region per fixture: (0.1, 0.5). cache plays at offset 0.2 →
    # source-timeline display should be 0.1 + 0.2 = 0.3.
    window._playback_engine._is_playing = True
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 0.2),  # 0.2 s into cache (offset within keeper-bounded cache)
    )

    # Put the toggle in B-state (cache is playing).
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)
    window._ab_toggle.set_state("B")
    assert window._ab_toggle.state == "B"

    # Drive the 30 Hz tick.
    window._on_playback_tick()

    waveform_playhead = window._lane_playheads["waveform"]
    assert waveform_playhead.value() == pytest.approx(0.3, abs=0.01), (
        f"In B-state, visual playhead must display source-time = "
        f"keeper.start_sec (0.1) + cache_position (0.2) = 0.3. "
        f"Got playhead={waveform_playhead.value()!r}. If this equals 0.2 "
        f"the tick is passing raw cache-timeline to the source-timeline "
        f"InfiniteLine — user-reported 07-10c symptom."
    )


def test_playback_tick_no_translation_in_A_state(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """Sanity guard — A-state must NOT translate (engine reports source-time)."""
    window, region, cache_path = primed_window_with_keeper
    window._playback_engine._is_playing = True
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 0.3),  # 0.3 s in source timeline
    )

    # Toggle stays in default A-state — no translation expected.
    assert window._ab_toggle.state == "A"
    window._on_playback_tick()

    waveform_playhead = window._lane_playheads["waveform"]
    assert waveform_playhead.value() == pytest.approx(0.3, abs=0.01), (
        f"In A-state, playhead is raw engine.position_seconds. Got "
        f"{waveform_playhead.value()!r}, expected 0.3."
    )


# =========================================================================
# Test 3d (Plan 07-10d) — _on_seek_requested translates source→cache in B
# =========================================================================
def test_smart_click_inside_keeper_switches_to_B_and_plays_cache(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """Plan 07-10e: waveform click inside a keeper with fresh cache → B mode.

    Smart-click contract:
      * Hit-test against keeper regions with fresh mastered cache.
      * Click inside such a keeper → toggle becomes B, engine plays the
        cache file at cache_offset = source_click - keeper.start_sec.
      * The selected keeper updates to the hit keeper.
    """
    window, region, cache_path = primed_window_with_keeper
    # Start in A (default).
    assert window._ab_toggle.state == "A"

    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    # Click at source 0.3, inside keeper (0.1, 0.5).
    window._on_seek_requested(0.3)

    # Toggle should have switched to B.
    assert window._ab_toggle.state == "B", (
        f"Click inside keeper region must switch toggle to B. "
        f"Got state={window._ab_toggle.state!r}."
    )
    # Engine must be asked to play the cache file at translated offset.
    assert play_spy.called, "Smart click must call engine.play to swap file"
    call = play_spy.call_args
    assert call.args[0] == str(cache_path), (
        f"Smart click inside keeper must engine.play(cache_path), got "
        f"{call.args[0]!r}."
    )
    actual_start = call.kwargs.get("start_seconds", 0.0)
    assert actual_start == pytest.approx(0.2, abs=0.01), (
        f"cache_offset = source_click (0.3) - keeper.start_sec (0.1) = 0.2. "
        f"Got start_seconds={actual_start!r}."
    )

    # Visual playhead displays source-time of the click.
    waveform_playhead = window._lane_playheads["waveform"]
    assert waveform_playhead.value() == pytest.approx(0.3, abs=0.01), (
        f"Visual playhead must show source-time of the click (0.3). Got "
        f"{waveform_playhead.value()!r}."
    )


def test_smart_click_outside_keeper_in_B_state_switches_back_to_A(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """Plan 07-10e: in B mode, click outside any keeper → switch back to A.

    User report: "if outside a keeper switch to A (original) but when
    clicking inside a keeper applying right away B mode".
    """
    window, region, cache_path = primed_window_with_keeper
    # Force toggle to B first.
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)
    window._ab_toggle.set_state("B")
    assert window._ab_toggle.state == "B"
    # Simulate the engine actually swapping to cache after B-press
    # (play is mocked so the real engine didn't reload). The smart-click
    # branch reads engine._current_path to decide same_file vs swap.
    monkeypatch.setattr(window._playback_engine, "_current_path", cache_path)
    play_spy.reset_mock()

    # Click at source 1.5 — outside keeper (0.1, 0.5) but inside source proxy (2 s).
    window._on_seek_requested(1.5)

    # Toggle should revert to A.
    assert window._ab_toggle.state == "A", (
        f"Click outside keeper region in B-mode must revert toggle to A. "
        f"Got state={window._ab_toggle.state!r}."
    )
    # Engine must play the SOURCE proxy at the clicked source-time.
    assert play_spy.called, "Smart click in B with outside-click must call engine.play"
    call = play_spy.call_args
    assert call.args[0] == str(window._current_playback_path), (
        f"Click outside keeper must engine.play(source_proxy). Got "
        f"{call.args[0]!r}; source_proxy is {str(window._current_playback_path)!r}."
    )
    actual_start = call.kwargs.get("start_seconds", 0.0)
    # The exact value may be clamped by engine duration; just verify it
    # is at least the requested source-time (1.5), not the cache-clamped
    # value (≤ keeper_duration=0.4) the legacy code would have produced.
    assert actual_start >= 1.0, (
        f"Click outside keeper at source 1.5 must seek source ≈1.5, not "
        f"clamp to cache-duration. Got start_seconds={actual_start!r}."
    )


def test_smart_click_in_A_state_inside_keeper_switches_to_B(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """Plan 07-10e: in A mode, click inside keeper → switch to B."""
    window, region, cache_path = primed_window_with_keeper
    assert window._ab_toggle.state == "A"

    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    # Click at source 0.4, inside keeper (0.1, 0.5).
    window._on_seek_requested(0.4)

    assert window._ab_toggle.state == "B"
    assert play_spy.called
    call = play_spy.call_args
    assert call.args[0] == str(cache_path)
    actual_start = call.kwargs.get("start_seconds", 0.0)
    # cache_offset = 0.4 - 0.1 = 0.3
    assert actual_start == pytest.approx(0.3, abs=0.01)


def test_waveform_click_clears_all_keeper_row_highlights(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """quick-260622-tit: waveform click clears every KeeperRow highlight.

    User request: "when clicking in the soundwave let's remove any active
    states to all keepers".

    Per-row active highlights ONLY reflect a row's own Play-button click.
    A waveform-click swap (which may change the engine's audio source) is
    NOT a per-row gesture — every row's highlight should clear so the
    affordance isn't lying about which row is in control.
    """
    window, region, cache_path = primed_window_with_keeper
    # Simulate: user previously clicked Play on this keeper, so the row
    # is marked active in "start" mode.
    window._currently_playing_keeper_id = region.id
    window._currently_playing_mode = "start"
    sidebar_row = window._keepers_sidebar._rows[region.id]
    sidebar_row.set_active_mode("start")
    assert sidebar_row.active_mode == "start"

    # Mock engine.play so the click doesn't actually try to load files.
    monkeypatch.setattr(window._playback_engine, "play", MagicMock())

    # User clicks the waveform — anywhere.
    window._on_seek_requested(0.3)

    # Every keeper row's highlight should clear, and the orchestrator
    # state should forget the previously-playing keeper + mode.
    assert window._currently_playing_keeper_id is None, (
        "Waveform click must clear _currently_playing_keeper_id."
    )
    assert window._currently_playing_mode is None, (
        "Waveform click must clear _currently_playing_mode."
    )
    assert sidebar_row.active_mode is None, (
        "Waveform click must clear the previously-active row's highlight."
    )


def test_keeper_play_middle_then_end_highlights_only_end(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """quick-260622-tit: clicking middle then end leaves only end highlighted.

    Clicking any keeper button fires its action immediately and moves the
    highlight to that button; the previous button clears. Re-clicking the
    SAME button keeps it active — it never pauses (no pause() call, state
    stays on that keeper+mode).
    """
    window, region, cache_path = primed_window_with_keeper
    sidebar_row = window._keepers_sidebar._rows[region.id]

    # Engine reports playing so _refresh_keeper_row_play_icons highlights.
    window._playback_engine._is_playing = True
    play_spy = MagicMock()
    pause_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)
    monkeypatch.setattr(window._playback_engine, "pause", pause_spy)

    # Click middle, then end.
    window._on_keeper_play(region.id, "middle")
    assert window._currently_playing_mode == "middle"
    window._on_keeper_play(region.id, "end")

    assert window._currently_playing_keeper_id == region.id
    assert window._currently_playing_mode == "end"
    assert sidebar_row.active_mode == "end", (
        "Middle-then-end must leave only the end button highlighted."
    )
    # Other rows (none here beyond this single keeper) are un-highlighted;
    # assert the per-row API holds: only one mode active at a time.
    assert sidebar_row._play.styleSheet() == ""
    assert sidebar_row._play_middle.styleSheet() == ""
    assert sidebar_row._play_end.styleSheet() != ""

    # Re-click the SAME (end) button — must NOT pause; stays active on end.
    window._on_keeper_play(region.id, "end")
    assert window._currently_playing_keeper_id == region.id
    assert window._currently_playing_mode == "end"
    assert sidebar_row.active_mode == "end"
    assert not pause_spy.called, (
        "Re-clicking the active button must never pause the engine "
        "(quick-260622-tit removed the pause-toggle)."
    )


def test_space_resume_in_B_state_plays_cache_not_source(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """Plan 07-10e: Space-to-resume after pausing in B-mode keeps the cache.

    User report: "pass on export segment, but keyboard shortcut a and b
    do nothing" — followed by terminal trace showing engine playing
    source from 0 after a click+space sequence. The bug: _action_toggle_
    playback always used self._current_playback_path (source proxy) on
    resume, blowing away the B-mode cache that was just loaded.

    Contract: when toggle is in B-state with a selected keeper and
    fresh cache, Space-resume MUST call engine.play with the cache path
    — not the source proxy.
    """
    window, region, cache_path = primed_window_with_keeper

    # Put engine in B-state — simulate _on_ab_state_changed having
    # already swapped to cache (we mock play so the actual swap is a
    # spy call; we still need the engine to report cache as current).
    monkeypatch.setattr(
        window._playback_engine, "_current_path", cache_path
    )
    window._ab_toggle.set_state("B")
    assert window._ab_toggle.state == "B"
    # Engine is paused (was_playing=False after a hypothetical pause).
    window._playback_engine._is_playing = False

    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    # User presses Space → resume.
    window._action_toggle_playback()

    assert play_spy.called, "Space-resume must call engine.play"
    call = play_spy.call_args
    assert call.args[0] == str(cache_path), (
        f"In B-state, Space-resume must play the CACHE path, got "
        f"{call.args[0]!r}. Expected {str(cache_path)!r}. The legacy "
        f"code passed self._current_playback_path (source proxy), which "
        f"silently reverted the user's B-mode preview."
    )


# =========================================================================
# Test 4 — Fail-closed when cache is missing/corrupt (T-7-05)
# =========================================================================
def test_fail_closed_when_cache_missing_or_corrupt(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """Cache file deleted after enable-check → press B fails closed.

    Expected behavior:
        * engine.play NOT called.
        * Toggle reverts to A.
        * Status bar shows the destructive toast.
    """
    window, region, cache_path = primed_window_with_keeper

    # Remove the cache AFTER the enable-state check passed so the
    # toggle is still enabled when the user presses B.
    cache_path.unlink()
    assert not cache_path.exists()

    window._playback_engine._is_playing = True
    monkeypatch.setattr(
        type(window._playback_engine),
        "position_seconds",
        property(lambda self: 0.5),
    )
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    window._ab_toggle.set_state("B")

    assert not play_spy.called, (
        "Cache-missing path must NOT call engine.play (fail-closed)"
    )
    # Toggle reverted back to A.
    assert window._ab_toggle.state == "A"
    # Status bar shows the destructive toast.
    msg = window.statusBar().currentMessage()
    assert msg == _TOAST_CACHE_MISSING


# =========================================================================
# Test 5 — STATE-KEY no-op in MainWindow (start in A, press A again)
# =========================================================================
def test_state_key_no_op_in_main_window(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """Already in A, press A again → engine.play NOT called."""
    window, region, cache_path = primed_window_with_keeper
    window._playback_engine._is_playing = True
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    assert window._ab_toggle.state == "A"
    window._ab_toggle.set_state("A")  # no-op
    assert not play_spy.called


# =========================================================================
# Test 6 — Modal dialog suppresses A/B shortcut activation
# =========================================================================
def test_modal_dialog_suppresses_shortcuts(
    primed_window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """While a modal dialog is open, the A/B shortcut handler must NOT swap.

    Qt's modal-dialog semantics suppress ``Qt.ShortcutContext.
    ApplicationShortcut`` activation automatically. We exercise the
    handler directly while pretending a modal dialog is up: the handler
    must check ``QApplication.activeModalWidget()`` and bail.
    """
    from PySide6.QtWidgets import QDialog

    window, region, cache_path = primed_window_with_keeper
    window._playback_engine._is_playing = True
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    # Create + show a modal QDialog so Qt's modal-active state is True.
    dlg = QDialog(window)
    dlg.setModal(True)
    dlg.show()
    try:
        QApplication.processEvents()
        assert QApplication.activeModalWidget() is dlg
        # Now drive the shortcut handler — should bail without play().
        window._on_ab_shortcut_pressed("B")
        assert not play_spy.called, (
            "A/B shortcut must not swap playback while a modal dialog is open"
        )
    finally:
        dlg.close()
        QApplication.processEvents()
