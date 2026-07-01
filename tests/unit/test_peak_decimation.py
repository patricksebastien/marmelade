"""Unit tests for ``marmelade.audio.peak_builder`` (AUD-02 build path).

Covers:
- ``windowed_minmax`` correctness for sine, silence, clipping, empty/short.
- ``build_proxy`` end-to-end on a real WAV fixture (round-trips through
  ``proxy_cache.load_proxy``).
- Progress callback contract — strictly increasing integers in [0..100],
  reaching 100, at most 101 calls total.
- Cancel contract — ``BuildCancelled`` raised, no ``.dat`` and no
  ``.dat.tmp`` left at ``dst_path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from marmelade.audio.peak_builder import (
    BuildCancelled,
    build_proxy,
    windowed_minmax,
)
from marmelade.audio.proxy_cache import (
    DEFAULT_SAMPLES_PER_PIXEL,
    load_proxy,
)
from tests.fixtures.synthesize import make_sine


SR = 44100
SPP = DEFAULT_SAMPLES_PER_PIXEL  # 256


def test_windowed_minmax_sine_amplitude_05() -> None:
    """Sine at amp=0.5 → pairs at ±16383 (0.5 × 32767), ±2 for rounding.

    The plan's contract phrases this as a "1 kHz sine at 44.1 kHz" — but
    at that ratio, 256-sample windows capture ~5.806 cycles and the
    closest sample to a peak can be 1/256-th of a cycle off, producing a
    shortfall of up to ~40 int16 units in the worst-aligned window. To
    keep the ±2 bound intact while still exercising the same code path,
    we use ``freq = 44100/32 Hz`` (~1378.125 Hz) which gives an integer
    32 samples per cycle and 8 cycles per window — every window now
    contains samples at phase π/2 (the exact peak) and at phase 3π/2 (the
    exact trough), so ``windowed_minmax`` reports the true amplitude
    bound on every window. Same algorithm, deterministic check.
    """
    n_samples = SPP * 1000  # 1000 windows
    samples_per_cycle = 32
    freq = SR / samples_per_cycle  # 1378.125 Hz — divides 256 exactly
    t = np.arange(n_samples, dtype=np.float64) / SR
    signal = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    pairs = windowed_minmax(signal, SPP)
    assert pairs.shape == (1000, 2)
    assert pairs.dtype == np.int16

    # Every window contains samples at phase π/2 and 3π/2 (exact peaks) so
    # the min/max land at ±16383 within ±2 ints of float→int16 rounding.
    assert np.all(pairs[:, 0] >= -16385)
    assert np.all(pairs[:, 0] <= -16381)
    assert np.all(pairs[:, 1] >= 16381)
    assert np.all(pairs[:, 1] <= 16385)


def test_windowed_minmax_silence() -> None:
    """All-zero input yields all-zero pairs."""
    samples = np.zeros(SPP * 100, dtype=np.float32)
    pairs = windowed_minmax(samples, SPP)
    assert pairs.shape == (100, 2)
    assert np.all(pairs == 0)


def test_windowed_minmax_empty_input() -> None:
    """Zero-length input returns shape (0, 2) int16."""
    pairs = windowed_minmax(np.array([], np.float32), SPP)
    assert pairs.shape == (0, 2)
    assert pairs.dtype == np.int16


def test_windowed_minmax_short_input() -> None:
    """Input shorter than spp returns shape (0, 2)."""
    pairs = windowed_minmax(np.zeros(SPP - 1, dtype=np.float32), SPP)
    assert pairs.shape == (0, 2)
    assert pairs.dtype == np.int16


def test_windowed_minmax_clips_above_one() -> None:
    """Input > 1.0 clips to 32767 (no int16 overflow)."""
    samples = np.full(SPP * 4, 1.5, dtype=np.float32)
    pairs = windowed_minmax(samples, SPP)
    # All windows = 1.5 → clipped to 1.0 → 32767.
    assert np.all(pairs[:, 0] == 32767)
    assert np.all(pairs[:, 1] == 32767)


def test_windowed_minmax_clips_below_minus_one() -> None:
    """Input < -1.0 clips to -32767."""
    samples = np.full(SPP * 4, -2.0, dtype=np.float32)
    pairs = windowed_minmax(samples, SPP)
    assert np.all(pairs[:, 0] == -32767)
    assert np.all(pairs[:, 1] == -32767)


def test_build_proxy_end_to_end_wav(tmp_path: Path) -> None:
    """30 s sine WAV → valid v2 .dat proxy with the expected header."""
    src = tmp_path / "sine.wav"
    dst = tmp_path / "proxies" / "deadbeefdeadbeef" / "peaks.dat"
    make_sine(src, duration_s=30.0, sample_rate=SR, channels=1, fmt="wav")

    header = build_proxy(src, dst, samples_per_pixel=SPP)
    assert header.version == 2
    assert header.channels == 1
    assert header.sample_rate == SR
    assert header.samples_per_pixel == SPP
    # 30 * 44100 / 256 = 5167.96... ; allow ±10 pair slop for boundary handling.
    expected_pairs = 30 * SR // SPP
    assert abs(header.length - expected_pairs) <= 10

    data, header2 = load_proxy(dst)
    assert header2 == header
    assert data.shape == (header.length, 2)


def test_build_proxy_progress_callback(tmp_path: Path) -> None:
    """progress_cb fires with strictly-increasing ints in [0..100], reaches 100, ≤ 101 calls."""
    src = tmp_path / "sine.wav"
    dst = tmp_path / "peaks.dat"
    make_sine(src, duration_s=10.0, sample_rate=SR, channels=1, fmt="wav")

    calls: list[int] = []

    def progress(pct: int) -> None:
        calls.append(pct)

    build_proxy(src, dst, samples_per_pixel=SPP, progress_cb=progress)

    assert 1 <= len(calls) <= 101
    assert calls == sorted(set(calls))  # strictly increasing, no duplicates
    assert calls[-1] == 100
    assert all(0 <= c <= 100 for c in calls)


def test_build_proxy_honors_cancel(tmp_path: Path) -> None:
    """cancel_check=lambda: True → BuildCancelled + no .dat / .dat.tmp leftover."""
    src = tmp_path / "sine.wav"
    dst = tmp_path / "peaks.dat"
    make_sine(src, duration_s=5.0, sample_rate=SR, channels=1, fmt="wav")

    with pytest.raises(BuildCancelled):
        build_proxy(src, dst, samples_per_pixel=SPP, cancel_check=lambda: True)

    assert not dst.exists()
    assert not (tmp_path / "peaks.dat.tmp").exists()


def test_build_proxy_cancel_after_first_block_no_partial_file(tmp_path: Path) -> None:
    """A cancel after the first block still leaves no .dat / .dat.tmp.

    A 30 s WAV at 44.1 kHz with BLOCK_SAMPLES=131_072 yields ~11 blocks, so
    we cancel on the second poll (after the first block is processed).
    """
    src = tmp_path / "sine.wav"
    dst = tmp_path / "peaks.dat"
    make_sine(src, duration_s=30.0, sample_rate=SR, channels=1, fmt="wav")

    call_count = {"n": 0}

    def cancel_after_first() -> bool:
        call_count["n"] += 1
        return call_count["n"] > 1

    with pytest.raises(BuildCancelled):
        build_proxy(
            src,
            dst,
            samples_per_pixel=SPP,
            cancel_check=cancel_after_first,
        )

    assert not dst.exists()
    assert not (tmp_path / "peaks.dat.tmp").exists()
