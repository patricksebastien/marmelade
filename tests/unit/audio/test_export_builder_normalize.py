"""quick-260621-gfq Task 3 — export-time normalize REMOVED from export_region.

quick-260620-mgu added keyword-only ``normalize_enabled`` /
``normalize_target_db`` to ``export_region`` (an export-time DC-remove +
peak-scale). quick-260621-gfq REMOVES that path entirely: normalize now lives
strictly inside the mastering chain (its final stage), and the
honest-end-to-end normalized bytes come from the mastered-cache export path.
Raw export streams the source verbatim.

This module pins: (1) the normalize kwargs no longer exist, and (2) raw
export of a DC-offset source is byte-identical to the pre-mgu behavior
(DC offset preserved, source streamed verbatim).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import soundfile as sf

from marmelade.audio.export_builder import export_region


SR = 44100


def _make_offset_stereo_proxy(
    path: Path, duration_s: float, offset: float, sr: int = SR
) -> None:
    """Stereo float32 WAV: 1 kHz sine + a constant DC ``offset``."""
    total = int(duration_s * sr)
    t = np.arange(total, dtype=np.float64) / sr
    mono = (0.3 * np.sin(2.0 * np.pi * 1000.0 * t) + offset).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    with sf.SoundFile(
        str(path), mode="w", samplerate=sr, channels=2,
        subtype="FLOAT", format="RF64",
    ) as f:
        f.write(stereo)


def test_export_region_has_no_normalize_kwargs() -> None:
    """export_region's signature no longer carries the normalize params."""
    sig = inspect.signature(export_region)
    assert "normalize_enabled" not in sig.parameters
    assert "normalize_target_db" not in sig.parameters


def test_raw_export_retains_dc_offset(tmp_path: Path) -> None:
    """Raw export streams the source verbatim — its DC offset is preserved."""
    src = tmp_path / "src.proxy.wav"
    _make_offset_stereo_proxy(src, 1.0, offset=0.2)
    dst = tmp_path / "raw.wav"
    export_region(
        proxy_path=src, dst_path=dst, start_frame=0, end_frame=SR,
        fade_frames=0, fmt="wav", sample_rate=SR,
    )
    data, _ = sf.read(str(dst), dtype="float32")
    # DC offset 0.2 should still be present (no normalize ran).
    assert abs(float(data.mean()) - 0.2) < 1e-3


def test_raw_export_byte_identity_is_deterministic(tmp_path: Path) -> None:
    """Two raw exports of the same source produce byte-identical files."""
    src = tmp_path / "src.proxy.wav"
    _make_offset_stereo_proxy(src, 1.0, offset=0.2)
    out_a = tmp_path / "a.wav"
    out_b = tmp_path / "b.wav"
    for dst in (out_a, out_b):
        export_region(
            proxy_path=src, dst_path=dst, start_frame=0, end_frame=SR,
            fade_frames=0, fmt="wav", sample_rate=SR,
        )
    assert out_a.read_bytes() == out_b.read_bytes()


def test_export_builder_has_no_in_ram_source(tmp_path: Path) -> None:
    """The mgu-only _InRamSource adapter is gone (normalize-only helper)."""
    import marmelade.audio.export_builder as eb

    assert not hasattr(eb, "_InRamSource")
