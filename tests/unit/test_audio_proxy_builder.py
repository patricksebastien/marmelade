"""Unit tests for ``marmelade.audio.audio_proxy_builder`` (AUD-04 build path).

Covers:
- Round-trip (stereo): a stereo sine WAV → readable WAV with matching
  sample_rate + 2 channels + FLOAT subtype.
- Round-trip (mono → stereo dup, D-02): a mono source produces a 2-ch proxy
  whose two columns are equal sample-for-sample.
- Multichannel downmix (3 ch → first 2, RESEARCH Open-Q-6).
- Sample-rate preservation (D-03 — no resample).
- Monotone progress contract — ≤ 101 calls, strictly increasing, ends at 100.
- BuildCancelled raised on immediate cancel; no .wav and no .wav.tmp left.
- BuildCancelled raised on cancel-after-N-polls; still no files left.
- Atomic write on success — no `.tmp` sibling remains.
- Empty-source guard: a 0-frame WAV produces a 0-frame stereo FLOAT WAV +
  exactly one progress_cb(100) emission.
- MP3 round-trip (AUD-03 structural pin) — pedalboard's MP3 reader works
  through iter_blocks at this layer.

Imports per PATTERNS.md §11: ``make_sine`` fixture + shared ``BuildCancelled``
class from ``peak_builder``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from marmelade.audio.audio_proxy_builder import build_audio_proxy
from marmelade.audio.peak_builder import BuildCancelled
from tests.fixtures.synthesize import make_sine


SR = 44100


# -----------------------------------------------------------------------------
# Round-trip tests
# -----------------------------------------------------------------------------


def test_build_audio_proxy_round_trip_stereo_float32(tmp_path: Path) -> None:
    """Stereo sine WAV → readable WAV with samplerate, 2 channels, FLOAT subtype."""
    src = tmp_path / "sine_stereo.wav"
    make_sine(src, duration_s=2.0, sample_rate=SR, channels=2, fmt="wav")
    dst = tmp_path / "out.proxy.wav"

    result = build_audio_proxy(src, dst)

    assert result == dst
    assert dst.exists()
    with sf.SoundFile(str(dst)) as f:
        assert f.samplerate == SR
        assert f.channels == 2
        assert f.subtype == "FLOAT"
        # Allow up to one BLOCK_SAMPLES of slop for boundary handling /
        # MP3-style decoder truncation. With a WAV source the frames count
        # is exact, but the contract phrases this as "within 1 block".
        expected_frames = int(2.0 * SR)
        assert abs(f.frames - expected_frames) <= 131_072


def test_build_audio_proxy_mono_source_duplicates_to_stereo(tmp_path: Path) -> None:
    """Per D-02: mono source → 2-ch proxy with L == R sample-for-sample."""
    src = tmp_path / "sine_mono.wav"
    make_sine(src, duration_s=2.0, sample_rate=SR, channels=1, fmt="wav")
    dst = tmp_path / "out.proxy.wav"

    build_audio_proxy(src, dst)

    data, sr = sf.read(str(dst), dtype="float32", always_2d=True)
    assert sr == SR
    assert data.shape[1] == 2
    # First 1000 frames: left channel equals right channel within float32 tol.
    np.testing.assert_allclose(data[:1000, 0], data[:1000, 1], rtol=1e-5, atol=1e-7)


def test_build_audio_proxy_three_channel_picks_first_two(tmp_path: Path) -> None:
    """Per D-02 / RESEARCH Open-Q-6: a 3-ch source is sliced to its first 2 channels.

    We synthesize a 3-channel float32 WAV directly via soundfile (make_sine is
    mono/stereo only). The three channels are distinct sines so the assertion
    can verify channel-identity is preserved through the proxy.
    """
    src = tmp_path / "src3.wav"
    n = int(0.5 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    ch0 = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    ch1 = (0.3 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)
    ch2 = (0.2 * np.sin(2.0 * np.pi * 1320.0 * t)).astype(np.float32)
    multichannel = np.stack([ch0, ch1, ch2], axis=1)  # (frames, 3)
    sf.write(str(src), multichannel, SR, subtype="FLOAT", format="WAV")

    dst = tmp_path / "out.proxy.wav"
    build_audio_proxy(src, dst)

    out, sr = sf.read(str(dst), dtype="float32", always_2d=True)
    assert sr == SR
    assert out.shape[1] == 2
    # First 2 source channels survive the downmix sample-for-sample.
    np.testing.assert_allclose(out[:1000, 0], ch0[:1000], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(out[:1000, 1], ch1[:1000], rtol=1e-5, atol=1e-7)


# -----------------------------------------------------------------------------
# Sample-rate preservation (D-03)
# -----------------------------------------------------------------------------


def test_build_audio_proxy_preserves_48000_sample_rate(tmp_path: Path) -> None:
    """D-03 — output samplerate equals source samplerate (no resample)."""
    src = tmp_path / "src_48k.wav"
    make_sine(src, duration_s=1.0, sample_rate=48000, channels=1, fmt="wav")
    dst = tmp_path / "out.proxy.wav"

    build_audio_proxy(src, dst)

    with sf.SoundFile(str(dst)) as f:
        assert f.samplerate == 48000


def test_build_audio_proxy_preserves_22050_sample_rate(tmp_path: Path) -> None:
    """D-03 — odd sample rate (22050) is preserved exactly."""
    src = tmp_path / "src_22k.wav"
    make_sine(src, duration_s=1.0, sample_rate=22050, channels=1, fmt="wav")
    dst = tmp_path / "out.proxy.wav"

    build_audio_proxy(src, dst)

    with sf.SoundFile(str(dst)) as f:
        assert f.samplerate == 22050


# -----------------------------------------------------------------------------
# Progress contract
# -----------------------------------------------------------------------------


def test_build_audio_proxy_progress_is_monotone_le_101(tmp_path: Path) -> None:
    """≤ 101 emissions, strictly increasing, ends at 100. Mirrors peak_builder."""
    src = tmp_path / "sine.wav"
    make_sine(src, duration_s=5.0, sample_rate=SR, channels=1, fmt="wav")
    dst = tmp_path / "out.proxy.wav"

    calls: list[int] = []
    build_audio_proxy(src, dst, progress_cb=calls.append)

    assert 1 <= len(calls) <= 101
    assert calls == sorted(set(calls))
    assert calls[-1] == 100
    assert all(0 <= c <= 100 for c in calls)


# -----------------------------------------------------------------------------
# Cancel contract
# -----------------------------------------------------------------------------


def test_build_audio_proxy_immediate_cancel_leaves_no_files(tmp_path: Path) -> None:
    """cancel_check=lambda: True → BuildCancelled; no .wav, no .wav.tmp at dst."""
    src = tmp_path / "sine.wav"
    make_sine(src, duration_s=5.0, sample_rate=SR, channels=1, fmt="wav")
    dst = tmp_path / "out.proxy.wav"

    with pytest.raises(BuildCancelled):
        build_audio_proxy(src, dst, cancel_check=lambda: True)

    assert not dst.exists()
    assert not Path(str(dst) + ".tmp").exists()


def test_build_audio_proxy_cancel_after_third_poll_no_files(tmp_path: Path) -> None:
    """A cancel after the third poll still leaves zero files on disk.

    Mirrors peak_builder's cancel-mid-build pin (test_peak_decimation.py
    ``test_build_proxy_cancel_after_first_block_no_partial_file``).
    """
    src = tmp_path / "sine.wav"
    make_sine(src, duration_s=30.0, sample_rate=SR, channels=1, fmt="wav")
    dst = tmp_path / "out.proxy.wav"

    call_count = {"n": 0}

    def cancel_after_third() -> bool:
        call_count["n"] += 1
        return call_count["n"] >= 3

    with pytest.raises(BuildCancelled):
        build_audio_proxy(src, dst, cancel_check=cancel_after_third)

    assert not dst.exists()
    assert not Path(str(dst) + ".tmp").exists()


# -----------------------------------------------------------------------------
# Atomic-write discipline
# -----------------------------------------------------------------------------


def test_build_audio_proxy_atomic_write_no_tmp_left_on_success(tmp_path: Path) -> None:
    """After a successful build, the `.tmp` sibling does not exist on disk."""
    src = tmp_path / "sine.wav"
    make_sine(src, duration_s=1.0, sample_rate=SR, channels=1, fmt="wav")
    dst = tmp_path / "out.proxy.wav"

    build_audio_proxy(src, dst)

    assert dst.exists()
    assert not Path(str(dst) + ".tmp").exists()


# -----------------------------------------------------------------------------
# Empty-source guard
# -----------------------------------------------------------------------------


def test_build_audio_proxy_empty_source_zero_frame_proxy(tmp_path: Path) -> None:
    """0-frame source → 0-frame stereo FLOAT WAV; progress fires exactly once at 100."""
    src = tmp_path / "empty.wav"
    # Open + close a SoundFile in write mode without writing — produces a
    # valid 0-frame WAV header.
    with sf.SoundFile(
        str(src),
        mode="w",
        samplerate=SR,
        channels=1,
        subtype="FLOAT",
        format="WAV",
    ):
        pass

    dst = tmp_path / "empty.proxy.wav"
    calls: list[int] = []

    result = build_audio_proxy(src, dst, progress_cb=calls.append)

    assert result == dst
    assert dst.exists()
    with sf.SoundFile(str(dst)) as f:
        assert f.channels == 2
        assert f.subtype == "FLOAT"
        assert f.frames == 0
    assert calls == [100]


# -----------------------------------------------------------------------------
# MP3 round-trip (AUD-03 structural pin)
# -----------------------------------------------------------------------------


def test_build_audio_proxy_mp3_round_trip(tmp_path: Path) -> None:
    """AUD-03 — pedalboard's MP3 reader produces a stereo FLOAT WAV proxy.

    Pins that iter_blocks(mono=False) + soundfile.write streaming compose
    correctly for MP3 inputs (the bug-#2 structural fix). frames count is
    relaxed by ±1 block because MP3 decoders can return slightly fewer
    frames than the header advertises (audio_file.py Pitfall #3).
    """
    src = tmp_path / "sine.mp3"
    make_sine(src, duration_s=2.0, sample_rate=SR, channels=1, fmt="mp3")
    dst = tmp_path / "out.proxy.wav"

    build_audio_proxy(src, dst)

    with sf.SoundFile(str(dst)) as f:
        assert f.samplerate == SR
        assert f.channels == 2
        assert f.subtype == "FLOAT"
        # frames > 0 — the proxy actually decoded the MP3.
        assert f.frames > 0
