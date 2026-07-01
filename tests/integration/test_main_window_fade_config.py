"""quick-260626-o9y — fade is config-driven at the 4 main_window fade sites.

Replaces the old forced ``fade_sec = min(2.0, region_dur/2.0)`` with values
read from the keeper's mastering config via ``fade_params``. Pins:

* PREVIEW (``_on_keeper_play``): fade enabled/duration honored; disabled → 0;
  legacy mastering=None → default (True, 2.0); middle/end suppress the fade-IN
  while keeping the fade-OUT.
* SINGLE export (``_on_export_region_requested``): fade_frames computed from
  the config (honored / clamped / disabled).

The PlaybackEngine sounddevice boundary is mocked; ``engine.play`` and
``_spawn_export_worker`` are spied so we assert the exact fade values the call
sites pass (mirrors test_keeper_play_normalize.py).
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from marmelade.audio.mastering.chain import _SESSION_DEFAULTS
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


# ----------------------------------------------------------------- fixtures
def _make_proxy_wav(tmp_path: Path, seconds: float = 2.0) -> Path:
    sr = 44100
    n = int(seconds * sr)
    audio = (np.random.RandomState(0).randn(n, 2) * 0.05).astype("float32")
    p = tmp_path / "proxy.wav"
    sf.write(str(p), audio, sr, subtype="FLOAT", format="WAV")
    return p


@pytest.fixture
def mocked_sounddevice(monkeypatch):
    fake_stream = MagicMock()
    fake_ctor = MagicMock(return_value=fake_stream)
    monkeypatch.setattr("marmelade.audio.playback.sd.OutputStream", fake_ctor)
    monkeypatch.setattr("marmelade.audio.playback._SOUNDDEVICE_AVAILABLE", True)
    return fake_ctor, fake_stream


@pytest.fixture
def window_with_keeper(qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path):
    """Open a file + one keeper spanning [0.1, 0.5] (region_dur = 0.4s)."""
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


def _set_fade(window, region_id, *, enabled: bool, duration_sec: float):
    """Override the keeper's fade config via get_mastering."""
    cfg = copy.deepcopy(_SESSION_DEFAULTS)
    cfg["fade"] = {"enabled": enabled, "duration_sec": duration_sec}
    window._regions_overlay.set_mastering(region_id, cfg)


# =========================================================================
# PREVIEW — fade enabled honored + clamped
# =========================================================================
def test_preview_fade_enabled_honored_and_clamped(
    window_with_keeper, monkeypatch, mocked_sounddevice
):
    window, region = window_with_keeper
    region_dur = float(region.end_sec) - float(region.start_sec)  # 0.4

    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    # duration 0.1 < region_dur/2 (0.2) -> honored verbatim.
    _set_fade(window, region.id, enabled=True, duration_sec=0.1)
    window._on_keeper_play(region.id, start_mode="start")
    assert play_spy.called
    pk = play_spy.call_args.kwargs
    assert pk["fade_in_seconds"] == pytest.approx(0.1)
    assert pk["fade_out_seconds"] == pytest.approx(0.1)

    # duration 9.0 > region_dur/2 -> clamped to region_dur/2 (0.2).
    play_spy.reset_mock()
    _set_fade(window, region.id, enabled=True, duration_sec=9.0)
    window._on_keeper_play(region.id, start_mode="start")
    pk = play_spy.call_args.kwargs
    assert pk["fade_in_seconds"] == pytest.approx(region_dur / 2.0)
    assert pk["fade_out_seconds"] == pytest.approx(region_dur / 2.0)


# =========================================================================
# PREVIEW — fade disabled -> both ends 0
# =========================================================================
def test_preview_fade_disabled_zero(
    window_with_keeper, monkeypatch, mocked_sounddevice
):
    window, region = window_with_keeper
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    _set_fade(window, region.id, enabled=False, duration_sec=5.0)
    window._on_keeper_play(region.id, start_mode="start")
    pk = play_spy.call_args.kwargs
    assert pk["fade_in_seconds"] == pytest.approx(0.0)
    assert pk["fade_out_seconds"] == pytest.approx(0.0)


# =========================================================================
# PREVIEW — legacy mastering=None -> default (True, 2.0)
# =========================================================================
def test_preview_legacy_none_uses_default(
    window_with_keeper, monkeypatch, mocked_sounddevice
):
    window, region = window_with_keeper
    region_dur = float(region.end_sec) - float(region.start_sec)  # 0.4
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)
    monkeypatch.setattr(
        window._regions_overlay, "get_mastering", lambda rid: None
    )

    window._on_keeper_play(region.id, start_mode="start")
    pk = play_spy.call_args.kwargs
    # default 2.0 clamps to region_dur/2 (0.2).
    assert pk["fade_in_seconds"] == pytest.approx(region_dur / 2.0)
    assert pk["fade_out_seconds"] == pytest.approx(region_dur / 2.0)


# =========================================================================
# PREVIEW — middle/end suppress the fade-IN, keep fade-OUT
# =========================================================================
@pytest.mark.parametrize("mode", ["middle", "end"])
def test_preview_middle_end_suppress_fade_in(
    window_with_keeper, monkeypatch, mocked_sounddevice, mode
):
    window, region = window_with_keeper
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    _set_fade(window, region.id, enabled=True, duration_sec=0.1)
    window._on_keeper_play(region.id, start_mode=mode)
    pk = play_spy.call_args.kwargs
    # fade-IN suppressed for middle/end; fade-OUT preserved.
    assert pk["fade_in_seconds"] == pytest.approx(0.0)
    assert pk["fade_out_seconds"] == pytest.approx(0.1)


# =========================================================================
# SINGLE export — fade_frames config-driven
# =========================================================================
def test_single_export_fade_frames_config_driven(
    window_with_keeper, monkeypatch, tmp_path
):
    window, region = window_with_keeper
    region_dur = float(region.end_sec) - float(region.start_sec)  # 0.4
    sr = window._playback_engine.sample_rate
    sr = sr if sr > 0 else 44100

    out_dir = tmp_path / "exports"
    out_dir.mkdir()
    monkeypatch.setattr(window, "_resolve_export_dir", lambda: out_dir)

    spawn_spy = MagicMock()
    monkeypatch.setattr(window, "_spawn_export_worker", spawn_spy)

    # Enabled @ 0.1s -> fade_frames = int(0.1 * sr).
    _set_fade(window, region.id, enabled=True, duration_sec=0.1)
    window._on_export_region_requested(region.id, "wav")
    assert spawn_spy.called
    assert spawn_spy.call_args.kwargs["fade_frames"] == int(0.1 * sr)

    # Disabled -> fade_frames == 0.
    spawn_spy.reset_mock()
    _set_fade(window, region.id, enabled=False, duration_sec=5.0)
    window._on_export_region_requested(region.id, "wav")
    assert spawn_spy.call_args.kwargs["fade_frames"] == 0

    # Large duration -> clamped to region_dur/2.
    spawn_spy.reset_mock()
    _set_fade(window, region.id, enabled=True, duration_sec=9.0)
    window._on_export_region_requested(region.id, "wav")
    assert spawn_spy.call_args.kwargs["fade_frames"] == int(
        (region_dur / 2.0) * sr
    )
