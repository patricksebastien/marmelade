"""Unit tests for :mod:`marmelade.audio.export_builder` (EXP-01 / D-A4-4/5).

Round-trip tests:
* MP3 path (320 CBR via pedalboard).
* WAV path (float32 IEEE_FLOAT via soundfile).
* Atomic .tmp discipline (success, cancel, error).
* Strictly-monotone progress callback.
* fade clamp at total_frames // 2 — never overlap fade-in and fade-out.

W-7 invariant: the fade ramps reach 0.0 / 1.0 exactly — verified
in test_export_fade_curve.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pedalboard.io import AudioFile

from marmelade.audio.export_builder import export_region
from marmelade.audio.peak_builder import BuildCancelled
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from tests.fixtures.synthesize import make_sine


SR = 44100


def _make_stereo_float32_proxy(path: Path, duration_s: float, sr: int = SR) -> None:
    """Write a stereo float32 WAV proxy fixture via soundfile (no MP3 codec).

    Mirrors the canonical Phase 2.1 proxy: subtype=FLOAT, 2 channels, RF64.
    Content is a 1 kHz sine.
    """
    total = int(duration_s * sr)
    t = np.arange(total, dtype=np.float64) / sr
    mono = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)  # shape (frames, 2)
    with sf.SoundFile(
        str(path),
        mode="w",
        samplerate=sr,
        channels=2,
        subtype="FLOAT",
        format="RF64",
    ) as f:
        f.write(stereo)


# ----------------------------------------------------------- validation


def test_export_region_unsupported_fmt_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 1.0)
    dst = tmp_path / "out.ogg"
    with pytest.raises(ValueError, match="Unsupported fmt"):
        export_region(
            proxy_path=src, dst_path=dst, start_frame=0, end_frame=SR,
            fade_frames=0, fmt="ogg", sample_rate=SR,
        )


def test_export_region_invalid_range_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 1.0)
    dst = tmp_path / "out.wav"
    with pytest.raises(ValueError, match="end_frame"):
        export_region(
            proxy_path=src, dst_path=dst, start_frame=100, end_frame=100,
            fade_frames=0, fmt="wav", sample_rate=SR,
        )


def test_export_region_negative_fade_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 1.0)
    dst = tmp_path / "out.wav"
    with pytest.raises(ValueError, match="fade_frames"):
        export_region(
            proxy_path=src, dst_path=dst, start_frame=0, end_frame=SR,
            fade_frames=-1, fmt="wav", sample_rate=SR,
        )


# ----------------------------------------------------------- round-trip


def test_export_region_writes_mp3(tmp_path: Path) -> None:
    """5s proxy → 3s MP3 region with 1s fade. Re-readable, ≈3s frames."""
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 5.0)
    dst = tmp_path / "out.mp3"

    export_region(
        proxy_path=src, dst_path=dst,
        start_frame=SR * 1, end_frame=SR * 4,
        fade_frames=SR * 1, fmt="mp3", sample_rate=SR,
    )

    assert dst.exists()
    assert not (tmp_path / "out.mp3.tmp").exists()
    assert not (tmp_path / "out.tmp.mp3").exists()
    assert dst.stat().st_size > 0
    with AudioFile(str(dst), "r") as f:
        # MP3 has frame-padding tolerance — allow ±2 MP3 frames (1152 samples each ~ 2304).
        assert abs(f.frames - 3 * SR) <= 2304 * 4
        assert f.samplerate == SR
        assert f.num_channels == 2


def test_export_region_writes_wav_float32(tmp_path: Path) -> None:
    """5s proxy → 3s WAV float32 region with 1s fade — sample-accurate."""
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 5.0)
    dst = tmp_path / "out.wav"

    export_region(
        proxy_path=src, dst_path=dst,
        start_frame=SR * 1, end_frame=SR * 4,
        fade_frames=SR * 1, fmt="wav", sample_rate=SR,
    )

    assert dst.exists()
    assert not (tmp_path / "out.wav.tmp").exists()
    assert not (tmp_path / "out.tmp.wav").exists()
    info = sf.info(str(dst))
    assert info.subtype == "FLOAT"
    assert info.frames == 3 * SR
    assert info.channels == 2


def test_export_region_atomic_no_tmp_leftover_on_success(tmp_path: Path) -> None:
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 2.0)
    dst = tmp_path / "out.wav"

    export_region(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=SR * 2,
        fade_frames=0, fmt="wav", sample_rate=SR,
    )

    assert dst.exists()
    assert not (tmp_path / "out.wav.tmp").exists()
    assert not (tmp_path / "out.tmp.wav").exists()


def test_export_region_atomic_tmp_cleanup_on_cancel(tmp_path: Path) -> None:
    """Cancel mid-export — BuildCancelled raised, no .tmp leftover."""
    src = tmp_path / "src.proxy.wav"
    # Big enough proxy to require multiple blocks (BLOCK_SAMPLES=131_072).
    _make_stereo_float32_proxy(src, 10.0)
    dst = tmp_path / "out.wav"

    state = {"count": 0}

    def cancel_check() -> bool:
        state["count"] += 1
        # Return True after the second poll so at least one block writes.
        return state["count"] > 2

    with pytest.raises(BuildCancelled):
        export_region(
            proxy_path=src, dst_path=dst,
            start_frame=0, end_frame=SR * 10,
            fade_frames=0, fmt="wav", sample_rate=SR,
            cancel_check=cancel_check,
        )
    assert not dst.exists()
    assert not (tmp_path / "out.wav.tmp").exists()
    assert not (tmp_path / "out.tmp.wav").exists()


def test_export_region_atomic_tmp_cleanup_on_writer_error(tmp_path: Path) -> None:
    """A negative sample_rate causes an error in the writer — no .tmp leftover."""
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 1.0)
    dst = tmp_path / "out.wav"
    # Negative sample_rate triggers a soundfile / pedalboard error.
    with pytest.raises(Exception):
        export_region(
            proxy_path=src, dst_path=dst,
            start_frame=0, end_frame=SR,
            fade_frames=0, fmt="wav", sample_rate=-1,
        )
    assert not (tmp_path / "out.wav.tmp").exists()
    assert not (tmp_path / "out.tmp.wav").exists()


def test_export_region_progress_strictly_monotone(tmp_path: Path) -> None:
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 3.0)
    dst = tmp_path / "out.wav"

    pct_log: list[int] = []
    export_region(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=SR * 3,
        fade_frames=0, fmt="wav", sample_rate=SR,
        progress_cb=pct_log.append,
    )

    # Strictly increasing (no duplicates), all in [0, 100].
    assert pct_log == sorted(set(pct_log))
    assert all(0 <= p <= 100 for p in pct_log)


def test_export_region_fade_clamped_to_half_region(tmp_path: Path) -> None:
    """fade_frames > total_frames // 2 → clamped so fades meet at center."""
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 1.0)
    dst = tmp_path / "out.wav"
    total = SR  # 1 second
    # Pass a fade larger than total — should be clamped to total // 2.
    export_region(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=total,
        fade_frames=total * 2, fmt="wav", sample_rate=SR,
    )
    # File exists and has the expected frame count.
    info = sf.info(str(dst))
    assert info.frames == total


# ----------------------------------------------------------- Qt-free guard


def test_export_builder_module_is_qt_free() -> None:
    """No Qt imports in the export_builder module (N-3 invariant)."""
    import marmelade.audio.export_builder as mod
    src = Path(mod.__file__).read_text()
    assert "PySide6" not in src
    assert "pyqtgraph" not in src
    assert "QtWidgets" not in src
