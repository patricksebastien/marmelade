"""Phase 8 Plan 08-05 Task 1 — bundle_builder.build_bundle unit tests.

Pins the multi-source WAV concat + silent spacer → 320 kbps CBR MP3
contract (D-04 + D-06):

* Concatenate N keepers in user-arranged order.
* Insert a user-configurable silent spacer between consecutive keepers
  (NOT after the last). Default 2.0 s; range 0..10 s (range validation
  lives in the dialog, not here — the function accepts any non-negative
  float so unit tests can exercise small values).
* 320 kbps CBR MP3 via pedalboard ``AudioFile`` writer (mirrors
  :func:`marmelade.audio.export_builder._stream_blocks_mp3` lines
  213-258 verbatim with ``quality=320``).
* Atomic write — ``<dst>.tmp.<ext>`` then ``os.replace``.
* Cancel via ``cancel_check()`` callback — partial ``.tmp`` removed.
* Sample-rate mismatch raises ``ValueError`` BEFORE any write.
* N-3 invariant — zero PySide6 imports in ``bundle_builder.py``.

The tests use 44100 Hz stereo float32 WAV fixtures so the sample-rate
path matches the production mastered cache shape exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from marmelade.audio.bundle_builder import build_bundle
from marmelade.audio.peak_builder import BuildCancelled


# ---------------------------------------------------------------------------
# Fixture helpers — module-scoped because they are pure-function file writes.
# ---------------------------------------------------------------------------


def _write_sine_wav(
    path: Path,
    *,
    duration_sec: float,
    sample_rate: int = 44100,
    freq_hz: float = 440.0,
    amplitude: float = 0.5,
) -> None:
    """Write a stereo float32 sine WAV at ``path``.

    Mirrors the shape of the mastered-cache WAV (44100 Hz, 2 channels,
    float32). Both channels carry the same sine so the mix-down is
    deterministic across the test suite.
    """
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    sig = (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    stereo = np.stack([sig, sig], axis=1)  # shape (frames, 2) — soundfile layout
    sf.write(str(path), stereo, sample_rate, subtype="FLOAT", format="WAV")


def _read_mp3_total_frames(path: Path) -> tuple[int, int]:
    """Return ``(frames, sample_rate)`` for an MP3 by streaming via pedalboard.

    soundfile cannot read MP3 in many distributions; pedalboard's
    AudioFile decode path is the same one the writer uses, so the
    round-trip stays in-house.
    """
    from pedalboard.io import AudioFile

    with AudioFile(str(path), "r") as f:
        sr = int(f.samplerate)
        total = int(f.frames)
    return total, sr


# ---------------------------------------------------------------------------
# Test 1 — concat with non-zero spacer
# ---------------------------------------------------------------------------


def test_concat_two_wavs_with_spacer(tmp_path: Path) -> None:
    """build_bundle([wav_a, wav_b], spacer_sec=2.0) — total duration ~= 4 s.

    1 s + 2 s silent gap + 1 s = 4 s of audio. Allow a generous +-0.25 s
    tolerance because LAME-style MP3 encoders pad the start/end with
    short silence frames (typically a few hundred ms) that round-trip
    via pedalboard.
    """
    sr = 44100
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = tmp_path / "bundle.mp3"

    _write_sine_wav(a, duration_sec=1.0, sample_rate=sr, freq_hz=440.0)
    _write_sine_wav(b, duration_sec=1.0, sample_rate=sr, freq_hz=880.0)

    build_bundle([a, b], spacer_sec=2.0, out_mp3_path=out, sample_rate=sr)

    assert out.exists(), "bundle MP3 not written"
    frames, out_sr = _read_mp3_total_frames(out)
    assert out_sr == sr
    expected_frames = int(4.0 * sr)  # 1 + 2 + 1
    # +- 0.25 s padding tolerance (MP3 codec adds ~50-150 ms of pad).
    tol_frames = int(0.25 * sr)
    assert abs(frames - expected_frames) <= tol_frames, (
        f"expected ~{expected_frames} frames (+-{tol_frames}); got {frames}"
    )


# ---------------------------------------------------------------------------
# Test 2 — spacer = 0 means back-to-back
# ---------------------------------------------------------------------------


def test_spacer_zero_means_no_gap(tmp_path: Path) -> None:
    """spacer_sec=0.0 — total duration ~= sum(individual durations).

    Two 1 s clips back-to-back → ~2 s output.
    """
    sr = 44100
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = tmp_path / "bundle.mp3"

    _write_sine_wav(a, duration_sec=1.0, sample_rate=sr)
    _write_sine_wav(b, duration_sec=1.0, sample_rate=sr)

    build_bundle([a, b], spacer_sec=0.0, out_mp3_path=out, sample_rate=sr)

    frames, _ = _read_mp3_total_frames(out)
    expected = int(2.0 * sr)
    tol = int(0.25 * sr)
    assert abs(frames - expected) <= tol, (
        f"expected ~{expected} frames (+-{tol}); got {frames}"
    )


# ---------------------------------------------------------------------------
# Test 3 — sample-rate mismatch raises ValueError before any write
# ---------------------------------------------------------------------------


def test_sample_rate_mismatch_raises(tmp_path: Path) -> None:
    """Mismatched source sample rates raise ValueError BEFORE any write."""
    a = tmp_path / "a44100.wav"
    b = tmp_path / "b22050.wav"
    out = tmp_path / "bundle.mp3"

    _write_sine_wav(a, duration_sec=0.5, sample_rate=44100)
    _write_sine_wav(b, duration_sec=0.5, sample_rate=22050)

    with pytest.raises(ValueError, match="Sample rate mismatch"):
        build_bundle([a, b], spacer_sec=1.0, out_mp3_path=out, sample_rate=44100)

    # Critical: NO half-written .tmp file should exist on disk.
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    assert not tmp.exists(), "tmp file should not exist after fail-fast ValueError"
    assert not out.exists(), "output file should not exist after fail-fast ValueError"


# ---------------------------------------------------------------------------
# Test 4 — cancel mid-build drops the partial .tmp file
# ---------------------------------------------------------------------------


def test_cancel_check_drops_partial_tmp(tmp_path: Path) -> None:
    """cancel_check returning True mid-concat raises BuildCancelled + removes .tmp."""
    sr = 44100
    paths = []
    for i in range(3):
        p = tmp_path / f"k{i}.wav"
        _write_sine_wav(p, duration_sec=0.5, sample_rate=sr)
        paths.append(p)
    out = tmp_path / "bundle.mp3"

    # State machine: cancel after the first keeper write completes.
    call_count = {"n": 0}

    def cancel_check() -> bool:
        call_count["n"] += 1
        # Allow first invocation (called BEFORE first write) to pass,
        # but trip on subsequent polls (BEFORE second write).
        return call_count["n"] > 1

    with pytest.raises(BuildCancelled):
        build_bundle(
            paths,
            spacer_sec=0.5,
            out_mp3_path=out,
            sample_rate=sr,
            cancel_check=cancel_check,
        )

    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    assert not tmp.exists(), "tmp file should be removed on cancel"
    assert not out.exists(), "final file should not exist on cancel"


# ---------------------------------------------------------------------------
# Test 5 — progress callback emits monotone non-decreasing values ending at 100
# ---------------------------------------------------------------------------


def test_progress_callback_monotone(tmp_path: Path) -> None:
    """progress_cb receives monotone non-decreasing values ending at 100."""
    sr = 44100
    paths = []
    for i in range(3):
        p = tmp_path / f"k{i}.wav"
        _write_sine_wav(p, duration_sec=0.25, sample_rate=sr)
        paths.append(p)
    out = tmp_path / "bundle.mp3"

    progress: list[int] = []
    build_bundle(
        paths,
        spacer_sec=0.0,
        out_mp3_path=out,
        sample_rate=sr,
        progress_cb=progress.append,
    )

    assert progress, "progress callback never fired"
    # Monotone non-decreasing.
    for i in range(1, len(progress)):
        assert progress[i] >= progress[i - 1], (
            f"progress non-monotone at index {i}: {progress!r}"
        )
    # Ends at 100.
    assert progress[-1] == 100, (
        f"progress should end at 100; got {progress[-1]} (full list: {progress!r})"
    )
    # Exactly 3 calls — one per keeper (33/66/100 shape).
    assert len(progress) == 3, (
        f"expected one progress call per keeper (3 total); got {progress!r}"
    )


# ---------------------------------------------------------------------------
# Test 6 — atomic write: os.replace failure removes .tmp and leaves no .mp3
# ---------------------------------------------------------------------------


def test_writes_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """os.replace failure removes the .tmp file + final .mp3 does NOT exist."""
    sr = 44100
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = tmp_path / "bundle.mp3"

    _write_sine_wav(a, duration_sec=0.25, sample_rate=sr)
    _write_sine_wav(b, duration_sec=0.25, sample_rate=sr)

    # Patch os.replace inside the bundle_builder module so the writer
    # succeeds but the atomic rename trips.
    import marmelade.audio.bundle_builder as bb

    def _boom(_src, _dst):
        raise OSError("rename refused")

    monkeypatch.setattr(bb.os, "replace", _boom)

    with pytest.raises(OSError, match="rename refused"):
        build_bundle([a, b], spacer_sec=0.5, out_mp3_path=out, sample_rate=sr)

    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    assert not tmp.exists(), "tmp must be removed when os.replace fails"
    assert not out.exists(), "final mp3 must not exist when os.replace fails"


# ---------------------------------------------------------------------------
# Test 7 — N-3 invariant: bundle_builder.py imports zero PySide6 modules
# ---------------------------------------------------------------------------


def test_no_pyside6_imports() -> None:
    """N-3 (D-27): marmelade.audio.bundle_builder has zero PySide6 imports."""
    here = Path(__file__).resolve()
    # tests/unit/audio/test_bundle_builder.py → repo root.
    repo_root = here.parents[3]
    src = repo_root / "src" / "marmelade" / "audio" / "bundle_builder.py"
    text = src.read_text()
    assert "from PySide6" not in text, (
        "bundle_builder.py must not import PySide6 (N-3 / D-27)"
    )
    assert "import PySide6" not in text, (
        "bundle_builder.py must not import PySide6 (N-3 / D-27)"
    )


# ---------------------------------------------------------------------------
# Test 8 — 320 kbps CBR pin (RESEARCH A1 verification, export_builder parity)
# ---------------------------------------------------------------------------


def test_uses_quality_320() -> None:
    """Source contains literal ``quality=320`` (RESEARCH A1 + export_builder parity)."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    src = repo_root / "src" / "marmelade" / "audio" / "bundle_builder.py"
    text = src.read_text()
    assert "quality=320" in text, (
        "bundle_builder.py must pin quality=320 (320 kbps CBR) — "
        "RESEARCH A1 verification + parity with export_builder._stream_blocks_mp3"
    )
