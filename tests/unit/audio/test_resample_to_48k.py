"""quick-260615-f77 — streaming resample-to-48 kHz working-file builder.

Pins the block-based RF64 streaming resampler:
  * output sample rate is exactly 48000,
  * output is RF64 FLOAT stereo,
  * output frame count ~= in_frames * 48000/44100 (within one block),
  * cancel mid-stream removes the partial .tmp and raises BuildCancelled
    leaving no output file behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from marmelade.audio.audio_file import BLOCK_SAMPLES, probe
from marmelade.audio.resample_to_48k import (
    CANONICAL_SAMPLE_RATE,
    BuildCancelled,
    resample_to_48k,
)
from tests.fixtures.synthesize import make_white_noise


def _make_44100_stereo(path: Path, duration_s: float = 0.5) -> Path:
    return make_white_noise(
        path,
        duration_s=duration_s,
        sample_rate=44100,
        channels=2,
        fmt="wav",
        seed=0,
    )


def test_output_rate_is_48000(tmp_path: Path) -> None:
    src = _make_44100_stereo(tmp_path / "in.wav")
    dst = tmp_path / "out.wav"
    resample_to_48k(src, dst)
    assert sf.info(str(dst)).samplerate == CANONICAL_SAMPLE_RATE == 48000


def test_output_format_is_rf64(tmp_path: Path) -> None:
    src = _make_44100_stereo(tmp_path / "in.wav")
    dst = tmp_path / "out.wav"
    resample_to_48k(src, dst)
    info = sf.info(str(dst))
    assert "RF64" in info.format, f"got format={info.format}"
    assert info.subtype == "FLOAT", f"got subtype={info.subtype}"
    assert info.channels == 2, f"got channels={info.channels}"


def test_frame_count_ratio(tmp_path: Path) -> None:
    src = _make_44100_stereo(tmp_path / "in.wav", duration_s=1.0)
    dst = tmp_path / "out.wav"
    resample_to_48k(src, dst)

    in_frames = probe(src).frames
    out_frames = sf.info(str(dst)).frames
    expected = in_frames * CANONICAL_SAMPLE_RATE / 44100
    # Within one block of tolerance (block-boundary resampler latency).
    assert abs(out_frames - expected) <= BLOCK_SAMPLES, (
        f"out_frames={out_frames}, expected~{expected:.0f}"
    )


def test_cancel_removes_partial(tmp_path: Path) -> None:
    # Multi-block source so a cancel after the first block fires mid-stream.
    duration_s = (BLOCK_SAMPLES * 3) / 44100
    src = _make_44100_stereo(tmp_path / "in.wav", duration_s=duration_s)
    dst = tmp_path / "out.wav"
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    calls = {"n": 0}

    def cancel_check() -> bool:
        # Allow the first block, cancel before the second.
        cancel = calls["n"] >= 1
        calls["n"] += 1
        return cancel

    with pytest.raises(BuildCancelled):
        resample_to_48k(src, dst, cancel_check=cancel_check)

    assert not dst.exists(), "no output file should remain after cancel"
    assert not tmp.exists(), "partial .tmp must be cleaned up on cancel"
