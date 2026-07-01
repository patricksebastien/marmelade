"""Phase 07.1 Plan 02 Task 2 — end-to-end proof the ending-FX tail grows the cache.

Drives the REAL :class:`MasteringRunnable` (no worker edits) over a 48 kHz
stereo WAV region and asserts the written mastered-cache WAV:

* is LONGER than the region by ~``tail_sec`` (the longer ``chain.process``
  output is written VERBATIM — 07.1-CONTEXT "MasteringRunnable writes the
  longer output to the cache"), and
* ends in TRUE silence (the safety fade survives the cache round-trip).

A SECOND run with ending_fx DISABLED writes a region-length cache (no tail),
proving the off path does not grow the cache. The two runs also resolve to
DIFFERENT ``mastered_cache_path`` filenames (distinct ``config_hash``) — the
cache-invalidation seam comes free.

The runnable is driven by calling ``.run()`` directly on the current thread
(other mastering tests do the same) and ``mastered_cache_path`` is given an
explicit per-test cache root, so no ``default_cache_root`` monkeypatch or
QThreadPool is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from marmelade.audio.mastering.chain import config_hash
from marmelade.audio.mastering_cache import mastered_cache_path
from marmelade.audio.mastering_worker import MasteringRunnable

SR = 48000  # quick-260615-f77 canonical mastering rate.
_VALID_KEY = "0123456789abcdef"
_VALID_KEEPER_ID = "a" * 32
TAIL_SEC = 4.0


def _stereo_sine_wav(path: Path, seconds: float = 3.0, sr: int = SR) -> int:
    """Write a ``(samples, channels)`` float32 WAV; return its frame count."""
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float32) / sr
    mono = (0.4 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    samples_first = np.stack([mono, mono], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples_first, sr, subtype="FLOAT", format="WAV")
    return n


def _base_cfg() -> dict:
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
        "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "matchering": {"enabled": False, "reference_path": ""},
        "loudness": {"enabled": False, "target_lufs": -14.0},
        "normalize": {"enabled": False, "target_db": 0.0},
    }


def _enabled_cfg() -> dict:
    cfg = _base_cfg()
    cfg["ending_fx"] = {
        "enabled": True,
        "effect_type": "hall_wash",
        "tail_sec": TAIL_SEC,
        "wet": 1.0,
        "primary": 0.5,
    }
    return cfg


def _disabled_cfg() -> dict:
    cfg = _base_cfg()
    cfg["ending_fx"] = {
        "enabled": False,
        "effect_type": "hall_wash",
        "tail_sec": TAIL_SEC,
        "wet": 1.0,
        "primary": 0.5,
    }
    return cfg


def _run(src: Path, dst: Path, cfg: dict) -> None:
    """Run a MasteringRunnable on the current thread and assert it finished."""
    finished: list[str] = []
    errors: list[str] = []
    runnable = MasteringRunnable(
        src, dst, keeper_id=_VALID_KEEPER_ID, mastering_cfg=cfg
    )
    runnable.signals.finished.connect(lambda p: finished.append(str(p)))
    runnable.signals.error.connect(lambda m: errors.append(str(m)))
    runnable.run()
    assert not errors, f"runnable errored: {errors}"
    assert finished == [str(dst)], f"unexpected terminal: finished={finished}"


@pytest.mark.slow
def test_ending_fx_render_grows_cache(qapp, tmp_path: Path) -> None:
    """Enabled render → cache WAV longer than region by ~tail_sec, ending silent."""
    src = tmp_path / "src.wav"
    region_frames = _stereo_sine_wav(src, seconds=3.0)

    cfg = _enabled_cfg()
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, config_hash(cfg))
    _run(src, dst, cfg)

    assert dst.exists() and dst.stat().st_size > 0
    info = sf.info(str(dst))
    assert info.subtype == "PCM_24"

    cache_audio, cache_sr = sf.read(str(dst), dtype="float32", always_2d=True)
    assert cache_sr == SR
    cache_frames = cache_audio.shape[0]
    min_growth = int(TAIL_SEC * SR * 0.9)
    assert cache_frames >= region_frames + min_growth, (
        f"cache did not grow: {cache_frames} frames vs region {region_frames} "
        f"(expected >= {region_frames + min_growth})"
    )

    last = float(np.max(np.abs(cache_audio[-240:, :])))
    assert last < 1e-3, f"cache tail not silent: max abs over final 240 frames = {last}"


@pytest.mark.slow
def test_ending_fx_disabled_render_does_not_grow_cache(qapp, tmp_path: Path) -> None:
    """Disabled render → cache frame count ~ region frame count (no tail)."""
    src = tmp_path / "src.wav"
    region_frames = _stereo_sine_wav(src, seconds=3.0)

    cfg = _disabled_cfg()
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, config_hash(cfg))
    _run(src, dst, cfg)

    cache_audio, _ = sf.read(str(dst), dtype="float32", always_2d=True)
    assert cache_audio.shape[0] == region_frames, (
        f"disabled path must not grow the cache: {cache_audio.shape[0]} "
        f"vs region {region_frames}"
    )


@pytest.mark.slow
def test_enabled_and_disabled_resolve_distinct_cache_paths(tmp_path: Path) -> None:
    """The cache-invalidation seam: enabled vs disabled → different filenames."""
    enabled_hash = config_hash(_enabled_cfg())
    disabled_hash = config_hash(_disabled_cfg())
    assert enabled_hash != disabled_hash

    p_enabled = mastered_cache_path(
        tmp_path, _VALID_KEY, _VALID_KEEPER_ID, enabled_hash
    )
    p_disabled = mastered_cache_path(
        tmp_path, _VALID_KEY, _VALID_KEEPER_ID, disabled_hash
    )
    assert p_enabled != p_disabled
