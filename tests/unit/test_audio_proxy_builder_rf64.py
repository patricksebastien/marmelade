"""Phase 2.1 HUMAN-UAT bug #2 — proxy uses RF64, not standard WAV.

The phase goal targets 8 h source files. Standard WAV format uses a 32-bit
RIFF data-chunk size field, so float32-stereo content caps at:

    2^32 bytes / (2 channels × 4 bytes/frame) = 536_870_912 frames
    536_870_912 frames / 44_100 sps ≈ 12_174 s ≈ 3.38 h

That's well under the phase's 8 h target. The crash surfaced during
/gsd-verify-work 2.1 on a real source longer than 3.4 h:

    ValueError: Cannot seek to position 740226612 frames, which is beyond
    end of file (536870911 frames) by -203355701 frames.

(536_870_911 = 2^32 / 8 - 1 — the exact frame boundary.)

Fix: build proxies in RF64 instead of WAV. RF64 is the WAV-compatible
64-bit RIFF extension — same .wav extension, readable by JUCE/pedalboard
and libsndfile-based readers — supports >4 GiB cleanly.

This module pins:
  1. The builder writes RF64 (libsndfile format detection on read).
  2. Default 30-second synthetic sources still produce playable, seekable
     files (no regression on small inputs).
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from marmelade.audio.audio_proxy_builder import build_audio_proxy
from tests.fixtures.synthesize import make_sine


def test_proxy_format_is_rf64(tmp_path: Path) -> None:
    """libsndfile reports the proxy file as RF64, not WAV.

    Load-bearing assertion: if this test fails, the proxy is in standard
    WAV format and will silently truncate at 4 GiB on long sources — same
    failure mode that surfaced as HUMAN-UAT bug #2.
    """
    src = tmp_path / "in.wav"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=1.0,
        sample_rate=44100,
        channels=1,
        fmt="wav",
    )

    dst = tmp_path / "proxy.wav"
    build_audio_proxy(src, dst)

    info = sf.info(str(dst))
    # libsndfile reports the format as a short string ('RF64' or 'WAV',
    # etc). On RF64 the major format is RF64; subtype stays FLOAT.
    assert info.format == "RF64", (
        f"proxy must be RF64 to support >4 GiB content "
        f"(8 h phase goal); got format={info.format}. "
        f"See 02.1-HUMAN-UAT.md issue #2 for the original crash signature."
    )
    assert info.subtype == "FLOAT", (
        f"proxy subtype must be float32 (D-01); got {info.subtype}"
    )
    assert info.channels == 2, (
        f"proxy channels must be 2 (D-02 stereo); got {info.channels}"
    )
    assert info.samplerate == 44100, (
        f"proxy samplerate must match source (D-03); got {info.samplerate}"
    )


def test_proxy_round_trip_seek_to_near_end(tmp_path: Path) -> None:
    """End-to-end: build a proxy, open it via soundfile, seek to near end.

    Mirrors the playback engine's `af.seek(start_frame)` call that crashed
    in the original HUMAN-UAT — but on a small synthetic source so the
    test runs quickly. The seek must NOT raise, and the position
    afterward must equal the requested frame.
    """
    src = tmp_path / "in.wav"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=2.0,
        sample_rate=44100,
        channels=1,
        fmt="wav",
    )

    dst = tmp_path / "proxy.wav"
    build_audio_proxy(src, dst)

    with sf.SoundFile(str(dst), mode="r") as f:
        # Seek to within 100 frames of the end.
        target = f.frames - 100
        result = f.seek(target)
        assert result == target, (
            f"soundfile.seek must succeed near EOF; "
            f"requested {target} but got {result}"
        )
        # Read the last 100 frames — should yield exactly 100 frames.
        remainder = f.read()
        assert remainder.shape[0] == 100, (
            f"reading after seek-to-near-end yielded "
            f"{remainder.shape[0]} frames; expected 100"
        )
