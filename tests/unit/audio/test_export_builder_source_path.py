"""Phase 7 Plan 07-06 Task 1 (RED) — D-20 ``source_path`` override.

Pins the additive keyword-only ``source_path`` argument on
:func:`marmelade.audio.export_builder.export_region`:

* When ``source_path is None`` (default) — existing Phase 3 behavior:
  the audio source is ``proxy_path``.
* When ``source_path`` is provided — it replaces ``proxy_path`` as the
  audio source. Start/end frames are interpreted against ``source_path``'s
  frame timeline.

The override is the load-bearing piece of D-20 — Phase 7 Plan 07-06
Phase C export pipeline passes the per-keeper mastered cache WAV via
this kwarg so the exported MP3/WAV carries the mastered audio rather
than the source proxy. All other ``export_region`` behavior (fade,
format, atomic-write) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from marmelade.audio.export_builder import export_region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target


SR = 44100


def _write_stereo_float32_wav(
    path: Path, duration_s: float, fill_value: float
) -> None:
    """Write a 2-channel float32 WAV of duration_s filled with ``fill_value``."""
    total = int(duration_s * SR)
    audio = np.full((total, 2), fill_value, dtype=np.float32)
    with sf.SoundFile(
        str(path),
        mode="w",
        samplerate=SR,
        channels=2,
        subtype="FLOAT",
        format="RF64",
    ) as f:
        f.write(audio)


def test_export_region_source_path_override_uses_alternate_source(
    tmp_path: Path,
) -> None:
    """``source_path=other`` makes export read from ``other``, not ``proxy_path``.

    Two distinct fixtures with different fill values: A is all 0.0,
    B is all 0.5. We pass A as ``proxy_path`` AND B as ``source_path``;
    the output must contain B's audio (mean ≈ 0.5), not A's.
    """
    proxy = tmp_path / "proxy_a.wav"  # all zeros
    alt = tmp_path / "alt_b.wav"  # all 0.5
    out = tmp_path / "out.wav"
    _write_stereo_float32_wav(proxy, 2.0, 0.0)
    _write_stereo_float32_wav(alt, 2.0, 0.5)

    export_region(
        proxy_path=proxy,
        dst_path=out,
        start_frame=0,
        end_frame=SR,  # 1 second
        fade_frames=0,  # zero fade so the output is the raw fill_value
        fmt="wav",
        sample_rate=SR,
        source_path=alt,
    )

    # Read the output and assert it carries the ALT fill, not the proxy.
    audio, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == SR
    # The output reads from ``alt`` (filled with 0.5) — the mean of the
    # middle frames (skipping potential boundary noise) must be ≈ 0.5,
    # NOT 0.0 (which is what ``proxy`` is filled with).
    middle = audio[1000:-1000]
    assert middle.size > 0
    assert abs(middle.mean() - 0.5) < 0.05, (
        f"Output mean {middle.mean():.4f} suggests proxy was used "
        f"instead of source_path override"
    )


def test_export_region_source_path_default_is_proxy_path_compat(
    tmp_path: Path,
) -> None:
    """Omitting ``source_path`` preserves Phase 3 behavior (uses proxy_path)."""
    proxy = tmp_path / "proxy.wav"  # all 0.5
    out = tmp_path / "out.wav"
    _write_stereo_float32_wav(proxy, 2.0, 0.5)

    export_region(
        proxy_path=proxy,
        dst_path=out,
        start_frame=0,
        end_frame=SR,
        fade_frames=0,
        fmt="wav",
        sample_rate=SR,
    )

    audio, _ = sf.read(str(out), dtype="float32", always_2d=True)
    middle = audio[1000:-1000]
    assert middle.size > 0
    assert abs(middle.mean() - 0.5) < 0.05, (
        "Default (no source_path) must read from proxy_path"
    )


def test_export_region_source_path_is_keyword_only(tmp_path: Path) -> None:
    """``source_path`` is keyword-only — positional passing must TypeError."""
    proxy = tmp_path / "proxy.wav"
    out = tmp_path / "out.wav"
    _write_stereo_float32_wav(proxy, 2.0, 0.5)

    import pytest

    # Try to pass source_path positionally as the 9th positional arg —
    # which would land in cancel_check (the last current positional).
    # The contract is keyword-only, so any positional call past the
    # existing signature should fail with TypeError.
    with pytest.raises(TypeError):
        export_region(
            proxy,
            out,
            0,
            SR,
            0,
            "wav",
            SR,
            None,  # progress_cb
            None,  # cancel_check
            proxy,  # source_path — must NOT be accepted positionally
        )
