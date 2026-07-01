"""Phase 8 Plan 08-04 — ffmpeg subprocess MP4 builder (YT-03).

Plan 08-04 Task 1 (TDD GREEN — Wave 0 skip marker removed). The 2 test
names below pin the canonical contract for
:mod:`marmelade.youtube.video_builder`.

D-12 ffmpeg args: ``-loop 1 -framerate 2 -tune stillimage -c:v libx264
-c:a aac -b:a 192k -pix_fmt yuv420p -shortest``. Bundled binary via
``imageio_ffmpeg.get_ffmpeg_exe()``.
"""

from __future__ import annotations

import subprocess
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from PIL import Image


def _make_silent_wav(path) -> None:
    """Write a 1-second silent stereo WAV at 44.1 kHz."""
    data = np.zeros((44100, 2), dtype=np.float32)
    sf.write(str(path), data, 44100, subtype="PCM_24")


def _make_jpeg(path, color=(64, 96, 192)) -> None:
    """Write a 1280x720 JPEG at `path` with the given color fill."""
    img = Image.new("RGB", (1280, 720), color=color)
    img.save(str(path), "JPEG", quality=90)


def test_ffmpeg_subprocess_builds_mp4(tmp_path) -> None:
    """build_video(image, audio, out_mp4) invokes bundled ffmpeg + produces an MP4.

    Verifies the output file exists, has non-trivial size, and ffprobe
    can read it back (the .tmp file is cleaned up by the atomic rename).
    """
    from marmelade.youtube import video_builder as vb

    wav = tmp_path / "in.wav"
    jpg = tmp_path / "in.jpg"
    out = tmp_path / "out.mp4"
    _make_silent_wav(wav)
    _make_jpeg(jpg)

    vb.build_video(jpg, wav, out)

    assert out.exists()
    assert out.stat().st_size > 1000  # MP4 container + libx264 + aac > 1 KiB

    # No stray .tmp.mp4 sibling.
    tmp_sibling = out.with_name(out.stem + ".tmp" + out.suffix)
    assert not tmp_sibling.exists()

    # ffprobe via the same bundled ffmpeg binary — null muxer confirms
    # the encoded streams parse cleanly.
    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg = get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(out), "-f", "null", "-"],
        capture_output=True,
    )
    assert probe.returncode == 0, f"ffmpeg null-mux failed: {probe.stderr.decode()[-500:]}"


def test_ffmpeg_failure_surfaces_returncode(tmp_path) -> None:
    """Non-zero ffmpeg exit raises RuntimeError carrying returncode + stderr tail.

    A nonexistent image path forces ffmpeg to fail. The .tmp sibling
    must be cleaned up on failure (atomic write contract).
    """
    from marmelade.youtube import video_builder as vb

    wav = tmp_path / "in.wav"
    out = tmp_path / "out.mp4"
    _make_silent_wav(wav)

    missing_jpg = tmp_path / "does_not_exist.jpg"

    with pytest.raises(RuntimeError) as exc_info:
        vb.build_video(missing_jpg, wav, out)

    msg = str(exc_info.value).lower()
    assert "ffmpeg" in msg
    # Either "failed" (non-zero exit) or "timed out" — both surface a
    # RuntimeError. The missing-file path produces a non-zero exit.
    assert "failed" in msg

    # .tmp sibling must NOT exist after the raise.
    tmp_sibling = out.with_name(out.stem + ".tmp" + out.suffix)
    assert not tmp_sibling.exists()
    # And the canonical output also does NOT exist (atomic rename never ran).
    assert not out.exists()
