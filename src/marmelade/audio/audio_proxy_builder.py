"""Audio proxy builder: source audio → canonical float32 stereo WAV cache.

Pure NumPy + stdlib + ``soundfile``. Like the rest of ``audio/``, this module
has NO toolkit imports (N-3 invariant). Plan 02.1-03's worker wraps
:func:`build_audio_proxy` in a ``QRunnable`` and bridges its progress/cancel
hooks to Qt signals via a thin shell.

Algorithm:
    Iterate the source via :func:`audio_file.iter_blocks` (``mono=False``)
    so each yielded block has shape ``(channels, n_frames)`` float32. Per
    block: downmix to canonical stereo (mono → duplicate L=R, 2ch →
    pass-through, >2ch → pick first 2 per D-02 / RESEARCH Open-Q-6),
    transpose to ``(n_frames, 2)`` for soundfile, and append-write into
    an open ``soundfile.SoundFile(mode='w', subtype='FLOAT', format='WAV')``.
    The output preserves the source sample rate exactly (D-03 — no resample).
    On success, atomically rename the in-flight ``<dst>.tmp`` to ``<dst>``
    via ``os.replace`` (D-18). The standard WAV header (managed by
    libsndfile) is sufficient — the cache_key in the filename carries
    source-freshness invalidation, so no BBC-v2-style sidecar is needed.

Progress contract (matters for Plan 02.1-03's QSignal-throttling):
    ``progress_cb(pct)`` fires only when ``int(100 * decoded / total)``
    *strictly increases* — so at most 101 calls total (0..100 inclusive),
    monotonic, no duplicates. A final ``progress_cb(100)`` is fired
    whenever the last block did not bump the integer percentage, so
    consumers always observe a terminal 100 on success.

Cancellation contract:
    ``cancel_check()`` is polled BEFORE doing any work for each yielded
    block. On ``True``, the partial ``<dst>.tmp`` file is deleted and the
    shared :class:`marmelade.audio.peak_builder.BuildCancelled` is
    re-raised. The caller (Plan 02.1-03's worker) catches
    :class:`BuildCancelled` and bridges it to ``signals.cancelled``. The
    cleanup happens BEFORE the exception propagates (D-11 / D-17) so
    cancel-restart never leaves debris under the canonical ``.wav`` name.

Memory contract (CLAUDE.md / AUD-01):
    Peak RAM is one block (``BLOCK_SAMPLES = 131_072`` float32 samples per
    channel ≈ 512 KiB × channels). The full 8h file is never materialised
    in memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from marmelade.audio.audio_file import BLOCK_SAMPLES, iter_blocks, probe

# REUSE VERBATIM from peak_builder per D-16 / D-17 — the cross-worker
# cancellation contract uses ONE exception class so heatmap, peak-builder
# and audio-proxy workers share `except BuildCancelled` machinery in their
# QRunnable shells. DO NOT define AudioProxyCancelled.
from marmelade.audio.peak_builder import BuildCancelled  # re-export


__all__ = ["build_audio_proxy", "_downmix_to_stereo", "BuildCancelled"]


def _downmix_to_stereo(block: np.ndarray) -> np.ndarray:
    """Reduce a ``(channels, n_frames)`` block to ``(n_frames, 2)`` float32.

    Per D-02 / RESEARCH Open-Q-6:
      * channels == 1 → duplicate mono into both stereo channels
      * channels == 2 → pass-through (transposed for soundfile's (frames, ch) shape)
      * channels  > 2 → pick the first 2 (predictable; multichannel jam
        recordings >2ch are vanishingly rare per CONTEXT discretion note)

    Input shape from ``iter_blocks(mono=False)`` is ``(channels, n)`` float32;
    ``soundfile.SoundFile.write`` wants ``(frames, channels)`` so we
    transpose at the end. The output is always contiguous float32.
    """
    n_ch = block.shape[0]
    if n_ch == 1:
        out = np.empty((block.shape[1], 2), dtype=np.float32)
        out[:, 0] = block[0]
        out[:, 1] = block[0]
        return out
    if n_ch >= 2:
        return np.ascontiguousarray(block[:2].T.astype(np.float32, copy=False))
    # Defensive — iter_blocks never yields 0-channel data, but pin the shape.
    return np.zeros((block.shape[1], 2), dtype=np.float32)


def build_audio_proxy(
    src_path: str | os.PathLike,
    dst_path: str | os.PathLike,
    progress_cb: Callable[[int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    """Transcode ``src_path`` to a canonical float32-stereo WAV at ``dst_path``.

    The output is a RIFF/WAVE file with ``audio_format=3`` (IEEE_FLOAT),
    32 bits per sample, exactly 2 channels (D-01), and the source sample
    rate preserved exactly (D-03 — no resample).

    Args:
        src_path: Any pedalboard-readable source (WAV/FLAC/MP3/OGG/M4A/AIFF).
        dst_path: Destination path for the proxy. The parent directory must
            exist — :class:`build_audio_proxy` does NOT create it (callers
            own the cache-tree layout).
        progress_cb: Optional ``Callable[[int], None]``. Fired only on
            strictly-increasing integer percentages 0..100. At most 101
            emissions. A final ``progress_cb(100)`` always fires on success.
        cancel_check: Optional ``Callable[[], bool]``. Polled BEFORE doing
            any work for each block. Returning ``True`` raises
            :class:`BuildCancelled`.

    Returns:
        The destination :class:`pathlib.Path`.

    Raises:
        BuildCancelled: when ``cancel_check()`` returns True at any
            between-block poll. The partial ``<dst_path>.tmp`` (if any) is
            removed BEFORE the exception propagates (D-11 / D-17).
    """
    src_p = Path(src_path)
    dst_p = Path(dst_path)
    # Same .tmp shape as proxy_cache + peak_builder — atomic-rename target.
    tmp_p = dst_p.with_suffix(dst_p.suffix + ".tmp")

    info = probe(src_p)
    total_frames = info.frames
    if total_frames <= 0:
        # Empty source — emit a zero-frame valid WAV and return. The `with`
        # block opens and closes the SoundFile (writing only the RIFF/WAVE
        # header + an empty data chunk) atomically into the .tmp; then we
        # rename. progress_cb fires exactly once at 100 (success terminal).
        with sf.SoundFile(
            str(tmp_p),
            mode="w",
            samplerate=info.sample_rate,
            channels=2,
            subtype="FLOAT",
            format="RF64",
        ):
            pass
        os.replace(str(tmp_p), str(dst_p))
        if progress_cb is not None:
            progress_cb(100)
        return dst_p

    last_pct = -1
    decoded = 0

    try:
        # soundfile streaming-write contract verified in RESEARCH §"soundfile.SoundFile":
        # multiple .write(block) calls flush header+data; an unclosed file is
        # readable; the data-chunk size is finalised on close. The `with` block
        # guarantees close (and FD cleanup) even on exception unwind — see
        # threat model T-02.1-09. Atomic-write discipline mirrors
        # proxy_cache.write_proxy: tmp sibling + os.replace on success.
        #
        # NB: format="RF64", NOT "WAV". Phase 2.1 HUMAN-UAT bug #2 — the
        # phase goal targets 8 h sources; standard WAV uses a 32-bit RIFF
        # data-chunk size, so float32 stereo caps at ~4 GiB ≈ 3.4 h at
        # 44.1 kHz. RF64 is the WAV-compatible 64-bit extension (still
        # `.wav` extension, read by JUCE/pedalboard + libsndfile). Original
        # crash signature: `af.seek` raised
        # `ValueError: Cannot seek to position N frames, which is beyond
        # end of file (536870911 frames)` — 536870912 == 2^32 / 8 ==
        # standard-WAV float32-stereo frame cap.
        with sf.SoundFile(
            str(tmp_p),
            mode="w",
            samplerate=info.sample_rate,
            channels=2,
            subtype="FLOAT",
            format="RF64",
        ) as sf_out:
            for block, _offset in iter_blocks(
                src_p, block_samples=BLOCK_SAMPLES, mono=False
            ):
                # Poll cancel between blocks BEFORE doing any work for this
                # block. MIRRORS peak_builder.py lines 147-148 EXACTLY.
                if cancel_check is not None and cancel_check():
                    raise BuildCancelled()

                stereo_fr = _downmix_to_stereo(block)
                sf_out.write(stereo_fr)

                decoded += int(block.shape[1])
                pct = int(100 * decoded / total_frames)
                # Strictly-monotone progress contract — mirrors peak_builder
                # lines 165-170 EXACTLY (≤101 emissions; no duplicates).
                if pct > last_pct and progress_cb is not None:
                    progress_cb(pct)
                    last_pct = pct
                elif pct > last_pct:
                    last_pct = pct

        # Ensure the final 100% fires even when the last block did not bump
        # the integer percentage. MIRRORS peak_builder.py lines 184-186.
        # MUST happen AFTER the `with sf.SoundFile(...)` block so libsndfile
        # has flushed + closed the file before any consumer reads it.
        if progress_cb is not None and last_pct < 100:
            progress_cb(100)

        # Atomic rename — must happen OUTSIDE the SoundFile `with` block so
        # libsndfile has finalised the RIFF chunk-size header. See threat
        # model T-02.1-08 (atomic visibility, not corruption prevention).
        os.replace(str(tmp_p), str(dst_p))
    except BuildCancelled:
        # D-11 / D-17 — best-effort .tmp cleanup BEFORE the exception
        # propagates. The SoundFile `with` block has already flushed +
        # closed the FD by the time we land here (context-manager exit
        # runs during exception unwind), so the unlink is safe.
        # MIRRORS peak_builder.py lines 199-207 EXACTLY.
        try:
            os.remove(str(tmp_p))
        except FileNotFoundError:
            pass
        raise

    return dst_p
