"""quick-260622-vwr — keeper preview applies the WYSIWYG normalize affine.

Pins :meth:`MainWindow._on_keeper_play` call-site wiring:

* A-mode (no fresh mastered cache) + Normalize ON -> compute the affine over
  the FULL keeper span and pass the non-default (normalize_dc, normalize_scale)
  to ``engine.play(...)`` so the audition matches the normalized waveform.
* A-mode + Normalize OFF -> pass the DEFAULT affine (0.0 / 1.0); the pre-pass
  result is never applied.
* B-mode (fresh mastered cache) -> pass the DEFAULT affine; the cache already
  baked normalize in as the mastering chain's final stage (no double-normalize).

PlaybackEngine.sounddevice is mocked at the boundary (same pattern as
test_ab_switch.py); the engine's ``play`` and
``compute_segment_normalize_params`` are spied/stubbed so we assert the exact
arguments the call site passes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

import marmelade.ui.main_window as main_window_mod
from marmelade.audio.mastering.chain import _SESSION_DEFAULTS, config_hash
from marmelade.audio.mastering_cache import mastered_cache_path
from marmelade.audio.proxy_cache import cache_key
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


def default_cache_root():
    """Resolve the cache root via the main_window binding the conftest patches.

    The ``tmp_cache_dir`` fixture monkeypatches
    ``marmelade.ui.main_window.default_cache_root`` (the production caller)
    but NOT this test module's own binding. Routing through the module ensures
    we write the fake cache to the SAME per-test directory
    :meth:`MainWindow._on_keeper_play` reads.
    """
    return main_window_mod.default_cache_root()


# ----------------------------------------------------------------- fixtures
def _make_proxy_wav(tmp_path: Path, seconds: float = 2.0) -> Path:
    sr = 44100
    n = int(seconds * sr)
    audio = (np.random.RandomState(0).randn(n, 2) * 0.05).astype("float32")
    p = tmp_path / "proxy.wav"
    sf.write(str(p), audio, sr, subtype="FLOAT", format="WAV")
    return p


def _write_fake_mastered_cache(
    cache_root: Path, src_key: str, keeper_id: str, cfg: dict
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
def window_with_keeper(qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path):
    """Open a file + one keeper. No mastered cache written -> A-mode path."""
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
    return window, region


# =========================================================================
# Test 1 — A-mode + Normalize ON passes the computed (non-default) affine
# =========================================================================
def test_amode_normalize_enabled_passes_affine(
    window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """A-mode + Normalize ON: compute affine over FULL span, pass it to play().

    No fresh mastered cache -> A-mode. get_normalize -> (True, -3.0). The call
    site must call compute_segment_normalize_params over the FULL keeper span
    (start_sec, end_sec, -3.0) and pass the returned (non-default) values as
    normalize_dc / normalize_scale to the A-mode play().
    """
    window, region = window_with_keeper
    start_sec, end_sec = region.start_sec, region.end_sec

    # Normalize ON at -3 dB.
    monkeypatch.setattr(
        window._regions_overlay, "get_normalize",
        lambda rid: (True, -3.0),
    )
    # Pre-pass returns a known non-default affine.
    compute_spy = MagicMock(return_value=(0.123, 1.75))
    monkeypatch.setattr(
        window._playback_engine,
        "compute_segment_normalize_params",
        compute_spy,
    )
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    window._on_keeper_play(region.id, start_mode="start")

    assert compute_spy.called, "Normalize ON must compute the segment affine"
    cargs = compute_spy.call_args.args
    # (path, start_sec, end_sec, target_db) over the FULL keeper span.
    assert cargs[1] == pytest.approx(float(start_sec), abs=1e-6)
    assert cargs[2] == pytest.approx(float(end_sec), abs=1e-6)
    assert cargs[3] == pytest.approx(-3.0)

    assert play_spy.called, "A-mode must call engine.play"
    pk = play_spy.call_args.kwargs
    assert pk.get("normalize_dc") == pytest.approx(0.123)
    assert pk.get("normalize_scale") == pytest.approx(1.75)


# =========================================================================
# Test 2 — A-mode + Normalize OFF passes the default affine
# =========================================================================
def test_amode_normalize_disabled_passes_defaults(
    window_with_keeper, qtbot, monkeypatch, mocked_sounddevice
) -> None:
    """A-mode + Normalize OFF: play() gets the identity affine (0.0 / 1.0).

    compute_segment_normalize_params must NOT be applied (its result ignored
    even if called) — the play() affine is the identity default.
    """
    window, region = window_with_keeper

    monkeypatch.setattr(
        window._regions_overlay, "get_normalize",
        lambda rid: (False, 0.0),
    )
    # If the pre-pass were applied, this non-default value would leak into play.
    compute_spy = MagicMock(return_value=(0.9, 9.9))
    monkeypatch.setattr(
        window._playback_engine,
        "compute_segment_normalize_params",
        compute_spy,
    )
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    window._on_keeper_play(region.id, start_mode="start")

    assert play_spy.called, "A-mode must call engine.play"
    pk = play_spy.call_args.kwargs
    assert pk.get("normalize_dc", 0.0) == pytest.approx(0.0)
    assert pk.get("normalize_scale", 1.0) == pytest.approx(1.0)


# =========================================================================
# Test 3 — B-mode (fresh cache) does NOT re-normalize
# =========================================================================
def test_bmode_cache_does_not_renormalize(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch,
    mocked_sounddevice,
) -> None:
    """B-mode (fresh mastered cache): play() gets the default affine.

    The mastered cache already baked normalize in as the chain's final stage,
    so the B-mode play() must NOT receive a non-default affine (no
    double-normalize) — even with Normalize ON.
    """
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

    # Drop a fresh mastered cache so the B-mode branch fires.
    src_key = cache_key(window._current_path)
    _write_fake_mastered_cache(default_cache_root(), src_key, region.id, cfg)

    # Normalize ON to prove B-mode ignores it (would be a double-normalize).
    monkeypatch.setattr(
        window._regions_overlay, "get_normalize",
        lambda rid: (True, -3.0),
    )
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    window._on_keeper_play(region.id, start_mode="start")

    assert play_spy.called, "B-mode must call engine.play with the cache"
    pk = play_spy.call_args.kwargs
    # Default affine (or no normalize kwargs at all).
    assert pk.get("normalize_dc", 0.0) == pytest.approx(0.0)
    assert pk.get("normalize_scale", 1.0) == pytest.approx(1.0)


# =========================================================================
# quick-260625 — "now playing" row tint follows the playhead
# =========================================================================
def test_set_playing_row_highlight_follows_playhead(window_with_keeper, qtbot) -> None:
    """_set_playing_row_highlight lights the keeper row the playhead is inside.

    Follows the playhead between regions and clears every row in inter-region
    gaps when the engine is not playing.
    """
    window, region = window_with_keeper
    overlay = window._regions_overlay
    # Add a second keeper region further along the timeline.
    overlay.start_draft(1.0)
    overlay.update_draft(1.5)
    region2 = overlay.commit_draft(1.5)
    assert region2 is not None
    overlay.set_state(region2.id, "keeper")

    rows = window._keepers_sidebar._rows
    assert region.id in rows and region2.id in rows

    # Playhead inside region 1 ([0.1, 0.5]) -> row 1 lit, row 2 dark.
    window._set_playing_row_highlight(0.3)
    assert rows[region.id]._is_playing_highlight is True
    assert rows[region2.id]._is_playing_highlight is False

    # Playhead moves into region 2 ([1.0, 1.5]) -> tint follows.
    window._set_playing_row_highlight(1.2)
    assert rows[region.id]._is_playing_highlight is False
    assert rows[region2.id]._is_playing_highlight is True

    # Playhead in a gap with the engine stopped -> all rows dark.
    window._set_playing_row_highlight(0.8)
    assert rows[region.id]._is_playing_highlight is False
    assert rows[region2.id]._is_playing_highlight is False


# =========================================================================
# quick-260625 — playhead visual offset: live-apply + QSettings persistence
# =========================================================================
def test_playhead_offset_changed_applies_and_persists(window_with_keeper) -> None:
    """_on_playhead_offset_changed updates the live value AND writes QSettings."""
    from PySide6.QtCore import QSettings
    from marmelade.ui.main_window import _PLAYHEAD_OFFSET_SETTINGS_KEY

    window, _region = window_with_keeper
    window._on_playhead_offset_changed(0.22)

    # Live value applied to the running window.
    assert window._playhead_visual_offset_sec == pytest.approx(0.22)
    # Persisted to QSettings (isolated to the per-test org/app namespace).
    stored = float(QSettings().value(_PLAYHEAD_OFFSET_SETTINGS_KEY))
    assert stored == pytest.approx(0.22)


# =========================================================================
# quick-260625 — VST3 renders route to the GUI thread (JUCE deadlock fix)
# =========================================================================
def test_dispatch_mastering_routes_vst3_to_gui_thread(
    window_with_keeper, monkeypatch
) -> None:
    """A vst3-enabled chain renders on the GUI thread (singleShot), not the pool.

    A VST3/JUCE plugin pins its message manager to the GUI thread; a worker
    render then deadlocks ("stuck at 5%"). So _dispatch_mastering_runnable must
    route vst3-enabled keepers through QTimer.singleShot (GUI thread) and leave
    every other keeper on the QThreadPool worker path.
    """
    import marmelade.ui.main_window as mw

    window, _region = window_with_keeper

    pool_calls: list = []
    timer_calls: list = []

    # Patch only the singleton's start() (leaving globalInstance intact so the
    # window's closeEvent can still waitForDone/clear at teardown).
    pool = mw.QThreadPool.globalInstance()
    monkeypatch.setattr(pool, "start", lambda runnable: pool_calls.append(runnable))
    monkeypatch.setattr(
        mw.QTimer, "singleShot", staticmethod(lambda ms, fn: timer_calls.append(fn))
    )

    class _FakeRunnable:
        def __init__(self, cfg):
            self._mastering_cfg = cfg

        def run(self):  # pragma: no cover — never actually invoked here
            pass

    # VST3 enabled -> GUI thread (singleShot), NOT the worker pool.
    r_vst = _FakeRunnable({"vst3": {"enabled": True}})
    window._dispatch_mastering_runnable(r_vst)
    assert len(timer_calls) == 1
    assert pool_calls == []

    # VST3 disabled -> worker pool (responsive path preserved).
    r_plain = _FakeRunnable({"vst3": {"enabled": False}})
    window._dispatch_mastering_runnable(r_plain)
    assert pool_calls == [r_plain]
    assert len(timer_calls) == 1  # unchanged
