"""End-to-end vertical-slice proof — :class:`MasteringRunnable` writes
a 24-bit PCM WAV whose sample peak is at or below the -1 dBTP target.

This is the Plan 07-01 synthesis test: hand-built mastering config
dict → :class:`MasteringRunnable` driven through ``QThreadPool`` →
real :class:`MasteringChain` rendering → atomic 24-bit WAV write at
:func:`mastered_cache_path`.

Phase 7 — Plan 01 Task 4 (07-01-PLAN.md).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QThreadPool

from marmelade.audio.mastering_cache import mastered_cache_path
from marmelade.audio.mastering_worker import MasteringRunnable
from marmelade.concurrency.worker import WorkerSignals


SR = 48000  # quick-260615-f77: canonical mastering rate (reverses D-04)
_VALID_KEY = "0123456789abcdef"
_VALID_KEEPER_ID = "a" * 32
_VALID_CONFIG_HASH = "0123456789ab"


def _hot_pink_noise_stereo(seconds: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Return a ``(samples, channels)`` hot stereo float32 signal at ~ -3 dBFS peak."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sr))
    white = rng.standard_normal(size=(n, 2)).astype(np.float32)
    pink = np.cumsum(white, axis=0).astype(np.float32)
    peak = float(np.max(np.abs(pink))) or 1.0
    # Normalize and push to ~ -3 dBFS peak.
    target_lin = 10 ** (-3.0 / 20.0)  # ~0.707
    return (pink / peak * target_lin).astype(np.float32)


def _write_proxy_wav(path: Path, audio_samples_first: np.ndarray, sr: int = SR) -> None:
    """Write a stereo float32 WAV in the Phase 2.1 proxy shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio_samples_first, sr, subtype="FLOAT", format="WAV")


@pytest.fixture
def src_proxy_wav(tmp_path: Path) -> Path:
    """Build a 5-second hot pink-noise stereo proxy WAV at ``tmp_path/src.wav``."""
    audio = _hot_pink_noise_stereo(seconds=5.0)
    src = tmp_path / "src.wav"
    _write_proxy_wav(src, audio)
    return src


def _default_limiter_only_cfg() -> dict:
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
    }


def test_mastering_runnable_renders_pcm24_below_minus1_dbtp(
    qtbot, qapp, tmp_path: Path, src_proxy_wav: Path
):
    """End-to-end proof: render hot pink-noise through the default chain.

    Default chain = Limiter ON, ceiling -1 dBTP. The resulting WAV
    must be PCM_24 with a sample peak at or below -1 dBFS (with a
    small tolerance for the ISP-vs-sample-peak gap).
    """
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, _VALID_CONFIG_HASH)
    runnable = MasteringRunnable(
        src_proxy_wav,
        dst,
        keeper_id=_VALID_KEEPER_ID,
        mastering_cfg=_default_limiter_only_cfg(),
    )
    assert isinstance(runnable.signals, WorkerSignals)
    assert type(runnable.signals) is WorkerSignals  # no subclass — D-16

    finished_payload: list[str] = []
    runnable.signals.finished.connect(lambda p: finished_payload.append(str(p)))

    with qtbot.waitSignal(runnable.signals.finished, timeout=30000):
        QThreadPool.globalInstance().start(runnable)

    # Terminal signal arrived; the runnable must NOT also have emitted
    # error / cancelled (single-terminal-signal contract).
    assert finished_payload, "finished did not emit"
    assert Path(finished_payload[0]) == dst

    # File assertions.
    assert dst.exists()
    assert dst.stat().st_size > 0

    info = sf.info(str(dst))
    assert info.subtype == "PCM_24", f"unexpected WAV subtype: {info.subtype}"

    # Read back and verify the sample-peak ceiling.
    read_audio, read_sr = sf.read(str(dst), dtype="float32", always_2d=True)
    assert read_sr == SR
    sample_peak = float(np.max(np.abs(read_audio)))
    sample_peak_dbfs = 20 * math.log10(max(sample_peak, 1e-12))
    # -1 dBTP target with ~1 dB ISP headroom from the limiter sub-chain;
    # allow 0.1 dB linear tolerance.
    assert sample_peak <= 10 ** (-1.0 / 20.0) + 1e-3, (
        f"sample_peak={sample_peak} ({sample_peak_dbfs:.2f} dBFS) "
        f"exceeds -1 dBFS"
    )


def test_mastering_runnable_cancel_before_run_leaves_no_tmp(
    qtbot, qapp, tmp_path: Path, src_proxy_wav: Path
):
    """Forcing cancel before ``run`` starts: no ``.tmp`` survives, ``cancelled`` fires."""
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, _VALID_CONFIG_HASH)
    runnable = MasteringRunnable(
        src_proxy_wav,
        dst,
        keeper_id=_VALID_KEEPER_ID,
        mastering_cfg=_default_limiter_only_cfg(),
    )
    runnable.cancel()  # set the cancel event before submit.

    with qtbot.waitSignal(runnable.signals.cancelled, timeout=10000):
        QThreadPool.globalInstance().start(runnable)

    # Atomic-write invariant: no leftover .tmp on disk.
    tmp = Path(str(dst) + ".tmp")
    assert not tmp.exists()
    # The cancel before run() also means the final cache file should
    # NOT exist (we cancelled before writing it).
    assert not dst.exists()


# =========================================================================
# Plan 07-08 Task 1 — keyword-only region kwargs (start_frame / end_frame).
# =========================================================================


def _discriminating_source_stereo(
    seconds: float, sr: int = SR, *, silent_first_half: bool = True
) -> np.ndarray:
    """Discriminating stereo source: silence in [0, seconds/2) + tone in [seconds/2, seconds).

    Used by ``test_mastering_runnable_slices_region_of_source`` to prove
    the region slice actually selected the silent first half (peak ~0)
    versus the full source which would include the loud tone.
    """
    n = int(round(seconds * sr))
    half = n // 2
    audio = np.zeros((n, 2), dtype=np.float32)
    if silent_first_half:
        # Tone in second half: 440 Hz at amp ~ 0.5 (well above the 0.05
        # peak threshold the slice-pin test asserts against).
        t = np.arange(n - half, dtype=np.float64) / sr
        tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        audio[half:, 0] = tone
        audio[half:, 1] = tone
    else:
        t = np.arange(half, dtype=np.float64) / sr
        tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        audio[:half, 0] = tone
        audio[:half, 1] = tone
    return audio


@pytest.fixture
def src_proxy_wav_6s_discriminating(tmp_path: Path) -> Path:
    """6-second source: silence in [0, 3) + 440 Hz tone in [3, 6).

    The slice-region pin selects [2, 3) — entirely inside the silent
    half — so the rendered output's peak amplitude must be < 0.05 if
    the slice landed correctly. A buggy build that masters the full
    source (or the wrong slice) would render the tone and the peak
    would be much higher.
    """
    audio = _discriminating_source_stereo(seconds=6.0, silent_first_half=True)
    src = tmp_path / "src_6s_discriminating.wav"
    _write_proxy_wav(src, audio)
    return src


def test_mastering_runnable_slices_region_of_source(
    qtbot, qapp, tmp_path: Path, src_proxy_wav_6s_discriminating: Path
):
    """Region kwargs (start_frame, end_frame) slice the source proxy before mastering.

    Pre-fix: MasteringRunnable has no region kwargs → TypeError at
    construction (or, if a future regression silently drops the kwargs,
    masters the full 6-s source and the output peak includes the loud
    tone — fails the < 0.05 peak check).

    Post-fix: cache contains ONLY [2 s, 3 s) of the source, which is
    silent in this discriminating fixture. Cache frames ≈ 1 * sr (within
    ±2048 for limiter lookahead) AND peak < 0.05.
    """
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, _VALID_CONFIG_HASH)
    runnable = MasteringRunnable(
        src_proxy_wav_6s_discriminating,
        dst,
        keeper_id=_VALID_KEEPER_ID,
        mastering_cfg=_default_limiter_only_cfg(),
        start_frame=int(2 * SR),
        end_frame=int(3 * SR),
    )

    finished_payload: list[str] = []
    runnable.signals.finished.connect(lambda p: finished_payload.append(str(p)))

    with qtbot.waitSignal(runnable.signals.finished, timeout=30000):
        QThreadPool.globalInstance().start(runnable)

    assert finished_payload, "finished did not emit"
    assert Path(finished_payload[0]) == dst
    assert dst.exists()

    info = sf.info(str(dst))
    # 1 second of audio at SR, ±2048 frames for limiter lookahead /
    # block-size rounding.
    expected = SR
    assert (expected - 2048) <= info.frames <= (expected + 2048), (
        f"Cache frames={info.frames}, expected ≈ {expected} (±2048). "
        f"If actual ≈ 6*SR (={6*SR}), the slice did not happen — "
        f"the runnable mastered the full source instead of the region."
    )

    # Discriminating audio assertion: the slice region [2 s, 3 s) is
    # entirely inside the silent half — the output peak should be very
    # close to 0. A buggy build that masters the full source would
    # render the 440 Hz tone and produce a peak ~ 0.5.
    audio_out, sr_out = sf.read(str(dst), dtype="float32", always_2d=True)
    assert sr_out == SR
    peak = float(np.max(np.abs(audio_out)))
    assert peak < 0.05, (
        f"Output peak={peak:.4f} — region was NOT the silent slice. "
        f"This means the runnable mastered a different region than "
        f"start_frame={int(2*SR)}, end_frame={int(3*SR)}."
    )


def test_mastering_runnable_invalid_region_emits_error(
    qtbot, qapp, tmp_path: Path, src_proxy_wav_6s_discriminating: Path
):
    """Inverted bounds (start >= end) raise ValueError → ``error`` signal."""
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, _VALID_CONFIG_HASH)
    runnable = MasteringRunnable(
        src_proxy_wav_6s_discriminating,
        dst,
        keeper_id=_VALID_KEEPER_ID,
        mastering_cfg=_default_limiter_only_cfg(),
        start_frame=int(5 * SR),
        end_frame=int(2 * SR),  # inverted
    )

    error_msgs: list[str] = []
    finished_payloads: list[str] = []
    runnable.signals.error.connect(lambda m: error_msgs.append(str(m)))
    runnable.signals.finished.connect(
        lambda p: finished_payloads.append(str(p))
    )

    with qtbot.waitSignal(runnable.signals.error, timeout=15000):
        QThreadPool.globalInstance().start(runnable)

    assert error_msgs, "error did not emit"
    assert not finished_payloads, "finished should NOT have emitted"
    assert not dst.exists()
    tmp = Path(str(dst) + ".tmp")
    assert not tmp.exists()


def test_mastering_runnable_partial_region_kwargs_raises(
    qtbot, qapp, tmp_path: Path, src_proxy_wav_6s_discriminating: Path
):
    """Only one of (start_frame, end_frame) set → ``error`` signal."""
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, _VALID_CONFIG_HASH)
    runnable = MasteringRunnable(
        src_proxy_wav_6s_discriminating,
        dst,
        keeper_id=_VALID_KEEPER_ID,
        mastering_cfg=_default_limiter_only_cfg(),
        start_frame=int(1 * SR),
        end_frame=None,  # missing partner
    )

    error_msgs: list[str] = []
    runnable.signals.error.connect(lambda m: error_msgs.append(str(m)))

    with qtbot.waitSignal(runnable.signals.error, timeout=15000):
        QThreadPool.globalInstance().start(runnable)

    assert error_msgs, "error did not emit"
    # Message should hint at the partial-kwargs contract.
    lower = error_msgs[0].lower()
    assert "together" in lower or "start_frame" in lower or "end_frame" in lower


def test_mastering_runnable_out_of_bounds_region_emits_error(
    qtbot, qapp, tmp_path: Path, src_proxy_wav_6s_discriminating: Path
):
    """Bounds outside the source (start_frame > total) → ``error`` signal."""
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, _VALID_CONFIG_HASH)
    runnable = MasteringRunnable(
        src_proxy_wav_6s_discriminating,
        dst,
        keeper_id=_VALID_KEEPER_ID,
        mastering_cfg=_default_limiter_only_cfg(),
        start_frame=int(10 * SR),  # 10 s into a 6 s source
        end_frame=int(11 * SR),
    )

    error_msgs: list[str] = []
    runnable.signals.error.connect(lambda m: error_msgs.append(str(m)))

    with qtbot.waitSignal(runnable.signals.error, timeout=15000):
        QThreadPool.globalInstance().start(runnable)

    assert error_msgs, "error did not emit"
    assert not dst.exists()


def test_mastering_runnable_no_region_kwargs_masters_full_source(
    qtbot, qapp, tmp_path: Path, src_proxy_wav_6s_discriminating: Path
):
    """Back-compat: no region kwargs → full source mastered (existing behavior)."""
    dst = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, _VALID_CONFIG_HASH)
    runnable = MasteringRunnable(
        src_proxy_wav_6s_discriminating,
        dst,
        keeper_id=_VALID_KEEPER_ID,
        mastering_cfg=_default_limiter_only_cfg(),
    )

    with qtbot.waitSignal(runnable.signals.finished, timeout=30000):
        QThreadPool.globalInstance().start(runnable)

    assert dst.exists()
    info = sf.info(str(dst))
    expected = 6 * SR
    assert (expected - 2048) <= info.frames <= (expected + 2048), (
        f"Back-compat broken: no region kwargs should master the full "
        f"6-s source (≈ {expected} frames ±2048), got {info.frames}."
    )
