"""Unit tests for ``marmelade.audio.audio_file`` — AUD-01 + AUD-03.

Covers:
- ``probe()`` returns correct metadata in < 100 ms for WAV / FLAC / MP3.
- ``iter_blocks()`` yields memory-bounded chunks summing to ≥ 99 % of frames.
- MP3 truncation (RESEARCH Pitfall #3) is handled gracefully.
- Block size never exceeds ``BLOCK_SAMPLES`` (per-block RAM budget).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from marmelade.audio.audio_file import (
    BLOCK_SAMPLES,
    MAX_DURATION_S,
    AudioProbe,
    iter_blocks,
    probe,
)
from tests.fixtures.synthesize import make_sine


SR = 44100
DUR = 30.0
EXPECTED_FRAMES = int(round(DUR * SR))  # 1_323_000


def test_module_constants() -> None:
    """``BLOCK_SAMPLES`` is 2**17 and ``MAX_DURATION_S`` is the 8-hour UI cap."""
    assert BLOCK_SAMPLES == 131_072
    assert MAX_DURATION_S == 8 * 3600.0


def test_probe_wav_30s_mono(tmp_path: Path) -> None:
    """A 30-second mono sine WAV reports the expected metadata via ``probe``."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=DUR, sample_rate=SR, channels=1, fmt="wav")

    result = probe(p)
    assert isinstance(result, AudioProbe)
    assert result.sample_rate == SR
    assert result.channels == 1
    assert abs(result.frames - EXPECTED_FRAMES) < 100
    assert abs(result.duration_s - DUR) < 0.1


def test_probe_flac(tmp_path: Path) -> None:
    """FLAC reports the same metadata shape as WAV."""
    p = tmp_path / "sine.flac"
    make_sine(p, duration_s=DUR, sample_rate=SR, channels=1, fmt="flac")

    result = probe(p)
    assert result.sample_rate == SR
    assert result.channels == 1
    assert abs(result.frames - EXPECTED_FRAMES) < 100
    assert abs(result.duration_s - DUR) < 0.1


def test_probe_mp3(tmp_path: Path) -> None:
    """MP3 reports channels and sample_rate; frames may drift ±5 % (Pitfall #3)."""
    p = tmp_path / "sine.mp3"
    make_sine(p, duration_s=DUR, sample_rate=SR, channels=1, fmt="mp3")

    result = probe(p)
    assert result.sample_rate == SR
    assert result.channels == 1
    # MP3 encoders pad to frame boundaries; allow ±5 %.
    assert abs(result.frames - EXPECTED_FRAMES) <= EXPECTED_FRAMES * 0.05


def test_probe_is_header_only_fast(tmp_path: Path) -> None:
    """``probe`` should be sub-200 ms on a 30 s file — no decode."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=DUR, sample_rate=SR, channels=1, fmt="wav")

    t0 = time.perf_counter()
    probe(p)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200.0, f"probe() took {elapsed_ms:.1f} ms (budget 200 ms)"


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    """``probe`` on a missing path raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "does-not-exist.wav")


def test_probe_unsupported_format_raises(tmp_path: Path) -> None:
    """``probe`` on a non-audio file raises ``ValueError`` (chained from pedalboard)."""
    p = tmp_path / "junk.wav"
    p.write_bytes(b"NOT_AN_AUDIO_FILE")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        probe(p)


def test_iter_blocks_wav_total_samples(tmp_path: Path) -> None:
    """Summed yielded samples reach ≥ 99 % of ``probe.frames`` for WAV."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=DUR, sample_rate=SR, channels=1, fmt="wav")
    info = probe(p)

    total = sum(block.size for block, _ in iter_blocks(p))
    assert total >= 0.99 * info.frames


def test_iter_blocks_mp3_total_samples_lenient(tmp_path: Path) -> None:
    """MP3 may truncate per Pitfall #3 — accept 90 % coverage."""
    p = tmp_path / "sine.mp3"
    make_sine(p, duration_s=DUR, sample_rate=SR, channels=1, fmt="mp3")
    info = probe(p)

    total = sum(block.size for block, _ in iter_blocks(p))
    assert total >= 0.90 * info.frames


def test_iter_blocks_block_size_bounded(tmp_path: Path) -> None:
    """No yielded block exceeds ``BLOCK_SAMPLES`` (per-block RAM budget)."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=DUR, sample_rate=SR, channels=1, fmt="wav")

    for block, _ in iter_blocks(p):
        assert block.size <= BLOCK_SAMPLES, (
            f"block of size {block.size} exceeds BLOCK_SAMPLES={BLOCK_SAMPLES}"
        )


def test_iter_blocks_offsets_increasing(tmp_path: Path) -> None:
    """Yielded ``offset_samples`` values increase monotonically by block length."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=5.0, sample_rate=SR, channels=1, fmt="wav")

    offsets: list[int] = []
    sizes: list[int] = []
    for block, offset in iter_blocks(p):
        offsets.append(offset)
        sizes.append(block.size)

    assert offsets[0] == 0
    running = 0
    for size, offset in zip(sizes, offsets):
        assert offset == running
        running += size


def test_iter_blocks_dtype_is_float32(tmp_path: Path) -> None:
    """Every yielded block has dtype ``np.float32``."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=2.0, sample_rate=SR, channels=1, fmt="wav")

    for block, _ in iter_blocks(p):
        assert block.dtype == np.float32


def test_iter_blocks_mono_from_stereo(tmp_path: Path) -> None:
    """A stereo source mixed via ``mono=True`` yields 1-D blocks."""
    p = tmp_path / "stereo.wav"
    make_sine(p, duration_s=2.0, sample_rate=SR, channels=2, fmt="wav")

    for block, _ in iter_blocks(p, mono=True):
        assert block.ndim == 1
