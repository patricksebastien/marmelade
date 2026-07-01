"""Header probe + block-based audio reader (AUD-01 + AUD-03).

This module is the Qt-free entry point into the audio backbone. It uses
``pedalboard.io.AudioFile`` (NOT ``soundfile`` — RESEARCH Pitfall #2) for all
primary reads because pedalboard ships a statically-bundled JUCE that handles
WAV / FLAC / MP3 on every platform without external system deps.

Memory contract (CLAUDE.md): never load a full multi-hour file into RAM.
``probe()`` reads only the header (sub-100 ms even on an 8-hour WAV) and
``iter_blocks()`` yields 131_072-sample chunks (≤ 512 KiB per block at mono
float32). Plan 02-03's ``peak_builder.build_proxy`` consumes ``iter_blocks``;
Plan 03 wraps the whole thing in a ``QRunnable`` worker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from pedalboard.io import AudioFile

# Per-block sample budget. 131_072 float32 samples = 512 KiB per block (mono).
# Even with 8 channels of float32, a single block stays under 4 MiB.
BLOCK_SAMPLES = 131_072  # 2**17

# Maximum source duration the UI accepts before showing the
# "File is longer than supported" dialog (UI-SPEC §Copywriting > Error states).
# 8 hours, matching CLAUDE.md's longest-recording target.
MAX_DURATION_S = 8 * 3600.0


@dataclass(frozen=True)
class AudioProbe:
    """Header-only metadata returned by :func:`probe`.

    Attributes:
        sample_rate: Source sample rate in Hz.
        frames: Total number of samples per channel as reported by the header.
        channels: Number of channels in the source file.
        duration_s: ``frames / sample_rate`` in seconds.
    """

    sample_rate: int
    frames: int
    channels: int
    duration_s: float


def probe(path: str | os.PathLike) -> AudioProbe:
    """Read only the header of ``path`` and return an :class:`AudioProbe`.

    Does not decode any audio samples. Wall-clock budget: < 100 ms for an
    8-hour WAV.

    Raises:
        FileNotFoundError: when ``path`` does not exist.
        ValueError: when ``path`` exists but is not a pedalboard-readable
            audio file (the original pedalboard exception is chained).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    try:
        with AudioFile(str(p), "r") as f:
            sample_rate = int(f.samplerate)
            frames = int(f.frames)
            channels = int(f.num_channels)
    except Exception as e:  # pedalboard raises RuntimeError / ValueError variants
        raise ValueError(f"Unsupported audio format: {p}") from e

    duration_s = frames / sample_rate if sample_rate > 0 else 0.0
    return AudioProbe(
        sample_rate=sample_rate,
        frames=frames,
        channels=channels,
        duration_s=duration_s,
    )


def iter_blocks(
    path: str | os.PathLike,
    block_samples: int = BLOCK_SAMPLES,
    mono: bool = True,
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield ``(samples_f32, offset_samples)`` blocks of ``path``.

    pedalboard returns chunks of shape ``(channels, n)`` and dtype ``float32``.
    With ``mono=True`` and a multi-channel source, channels are mixed via
    ``.mean(axis=0)`` per block so the yielded array is 1-D shape ``(n,)``.
    With ``mono=False``, the raw 2-D shape is yielded.

    The total number of yielded samples will equal
    ``probe(path).frames`` for WAV / FLAC, but may be slightly less for MP3
    because pedalboard's JUCE decoder can return ``size == 0`` before reaching
    the header-declared ``frames`` (RESEARCH Pitfall #3). We handle that
    truncation by breaking the loop cleanly — never raising.

    No yielded block exceeds ``block_samples`` along its leading axis (so the
    per-block RAM budget is bounded — 131_072 float32 samples = 512 KiB at
    mono).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    with AudioFile(str(p), "r") as f:
        total = int(f.frames)
        offset = 0
        while offset < total:
            remaining = total - offset
            n = min(block_samples, remaining)
            chunk = f.read(n)  # shape: (channels, n_actual), dtype float32
            if chunk.size == 0:
                # MP3 truncation safety (Pitfall #3) — stop cleanly.
                break
            n_actual = chunk.shape[1]

            if mono:
                if chunk.shape[0] > 1:
                    out = chunk.mean(axis=0).astype(np.float32, copy=False)
                else:
                    out = chunk[0].astype(np.float32, copy=False)
            else:
                out = chunk.astype(np.float32, copy=False)

            yield out, offset
            offset += n_actual
