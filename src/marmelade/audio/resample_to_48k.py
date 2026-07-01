"""Streaming resample-to-48 kHz working-file builder.

quick-260615-f77: the audio pipeline standardizes on a 48 kHz canonical
sample rate (reverses Phase 2.1 D-04). 48 kHz sources are used as-is; a
non-48 kHz source (e.g. 44.1 kHz) is converted to a 48 kHz working file
on open via this module so its keepers master successfully.

Pure NumPy + stdlib + ``soundfile`` + ``soxr``. Like the rest of
``audio/``, this module has NO toolkit imports (N-3 invariant).

Algorithm:
    Iterate the source via :func:`audio_file.iter_blocks` (``mono=False``)
    so each yielded block has shape ``(channels, n)`` float32. Per block:
    downmix to canonical stereo ``(n, 2)`` via the shared
    :func:`audio_proxy_builder._downmix_to_stereo` (imported to avoid
    drift), then resample the stereo block to 48 kHz with soxr in the
    established ``(samples, channels)`` orientation, and append-write the
    resampled frames into an open RF64 FLOAT stereo ``soundfile.SoundFile``.
    On success, atomically rename the in-flight ``<dst>.tmp`` to ``<dst>``
    via ``os.replace``.

    A :class:`soxr.ResampleStream` is used when available (clean
    block-boundary handling); otherwise per-block :func:`soxr.resample`
    is used. Per-block soxr is acceptable here — this is an analysis
    viewer, not a sample-accurate editor.

Progress contract:
    ``progress_cb(pct)`` fires only when ``int(100 * decoded / total)``
    *strictly increases* (≤101 emissions, monotonic). A final
    ``progress_cb(100)`` always fires on success.

Cancellation contract:
    ``cancel_check()`` is polled BEFORE doing any work for each block. On
    ``True`` the partial ``<dst>.tmp`` is removed and the shared
    :class:`marmelade.audio.peak_builder.BuildCancelled` is re-raised
    BEFORE the exception propagates.

Memory contract (CLAUDE.md / AUD-01):
    Peak RAM is one block. The full 8 h file is NEVER materialised. RF64
    is mandatory — an 8 h 48 kHz stereo float WAV exceeds the 4 GiB
    standard-WAV data-chunk cap (Phase 2.1 HUMAN-UAT bug #2).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import soxr

from marmelade.audio.audio_file import BLOCK_SAMPLES, iter_blocks, probe

# REUSE VERBATIM — the shared downmix keeps stereo normalization identical
# to the proxy builder (no drift).
from marmelade.audio.audio_proxy_builder import _downmix_to_stereo

# REUSE VERBATIM from peak_builder — ONE cross-worker cancellation class.
from marmelade.audio.peak_builder import BuildCancelled  # re-export


__all__ = ["resample_to_48k", "CANONICAL_SAMPLE_RATE", "BuildCancelled"]

CANONICAL_SAMPLE_RATE = 48000


def resample_to_48k(
    src_path: str | os.PathLike,
    dst_path: str | os.PathLike,
    *,
    progress_cb: Callable[[int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    """Stream-resample ``src_path`` to a 48 kHz RF64 FLOAT stereo WAV.

    Args:
        src_path: Any pedalboard-readable source (WAV/FLAC/MP3/OGG/M4A/AIFF).
        dst_path: Destination path for the 48 kHz working file. The parent
            directory must exist — callers own the cache-tree layout.
        progress_cb: Optional ``Callable[[int], None]``. Fired only on
            strictly-increasing integer percentages 0..100. A final
            ``progress_cb(100)`` always fires on success.
        cancel_check: Optional ``Callable[[], bool]``. Polled BEFORE doing
            any work for each block. Returning ``True`` raises
            :class:`BuildCancelled`.

    Returns:
        The destination :class:`pathlib.Path`.

    Raises:
        BuildCancelled: when ``cancel_check()`` returns True at any
            between-block poll. The partial ``<dst_path>.tmp`` (if any) is
            removed BEFORE the exception propagates.
    """
    src_p = Path(src_path)
    dst_p = Path(dst_path)
    tmp_p = dst_p.with_suffix(dst_p.suffix + ".tmp")

    info = probe(src_p)
    in_rate = int(info.sample_rate)
    total_frames = int(info.frames)

    if total_frames <= 0:
        # Empty source — emit a zero-frame valid 48 kHz RF64 file.
        with sf.SoundFile(
            str(tmp_p),
            mode="w",
            samplerate=CANONICAL_SAMPLE_RATE,
            channels=2,
            subtype="FLOAT",
            format="RF64",
        ):
            pass
        os.replace(str(tmp_p), str(dst_p))
        if progress_cb is not None:
            progress_cb(100)
        return dst_p

    # Prefer the streaming resampler for clean block-boundary handling.
    stream = None
    if hasattr(soxr, "ResampleStream"):
        stream = soxr.ResampleStream(
            float(in_rate), float(CANONICAL_SAMPLE_RATE), 2, dtype="float32"
        )

    last_pct = -1
    decoded = 0

    try:
        # RF64 (NOT "WAV") is mandatory — an 8 h 48 kHz stereo float WAV
        # exceeds the 4 GiB standard-WAV data-chunk cap (HUMAN-UAT bug #2).
        with sf.SoundFile(
            str(tmp_p),
            mode="w",
            samplerate=CANONICAL_SAMPLE_RATE,
            channels=2,
            subtype="FLOAT",
            format="RF64",
        ) as sf_out:
            # Stream block-by-block — NEVER materialise the full file
            # (CLAUDE.md memory contract; an 8 h file would be ~11 GB).
            # The last block is detected from the running frame count so
            # `soxr.ResampleStream.resample_chunk(..., last=True)` can flush
            # its tail without pre-counting blocks.
            for block, _offset in iter_blocks(
                src_p, block_samples=BLOCK_SAMPLES, mono=False
            ):
                # Poll cancel BEFORE doing any work for this block.
                if cancel_check is not None and cancel_check():
                    raise BuildCancelled()

                stereo_fr = _downmix_to_stereo(block)  # (n, 2) float32
                is_last = (decoded + int(block.shape[1])) >= total_frames

                if stream is not None:
                    out = stream.resample_chunk(stereo_fr, last=is_last)
                else:
                    out = soxr.resample(
                        stereo_fr, float(in_rate), float(CANONICAL_SAMPLE_RATE)
                    )
                out = np.ascontiguousarray(out, dtype=np.float32)
                if out.ndim == 1:
                    out = out.reshape(-1, 1)
                if out.shape[0] > 0:
                    sf_out.write(out)

                decoded += int(block.shape[1])
                pct = int(100 * decoded / total_frames)
                if pct > last_pct:
                    last_pct = pct
                    if progress_cb is not None:
                        progress_cb(pct)

        if progress_cb is not None and last_pct < 100:
            progress_cb(100)

        # Atomic rename OUTSIDE the SoundFile `with` so libsndfile has
        # finalised the RF64 header before any consumer reads the file.
        os.replace(str(tmp_p), str(dst_p))
    except BaseException:
        # Best-effort .tmp cleanup BEFORE the exception propagates (covers
        # BuildCancelled and any decode/resample/write error).
        try:
            os.remove(str(tmp_p))
        except FileNotFoundError:
            pass
        raise

    return dst_p
