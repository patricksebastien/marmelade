"""Peak-pyramid builder: source audio → BBC-audiowaveform v2 ``.dat`` proxy.

Pure NumPy + stdlib. Like the rest of ``audio/``, this module has NO
toolkit imports (N-3 invariant). Plan 03's worker wraps :func:`build_proxy`
in a ``QRunnable`` and connects its progress/cancel hooks to Qt signals
via a thin shell.

Algorithm:
    Iterate the source via :func:`audio_file.iter_blocks` (≤ 131_072
    samples per yielded block, mono mix-down). Maintain a leftover-samples
    buffer so windows never straddle block boundaries. For each
    ``samples_per_pixel``-sized window, take ``min``/``max``, clip to ±1.0
    (RESEARCH Open Q #2 — inter-sample MP3 peaks can overshoot), scale by
    32767, and cast to ``int16``. After all blocks, pad any remainder to a
    full window with the last sample and emit one final pair. Persist via
    :func:`proxy_cache.write_proxy`.

Progress contract (matters for Plan 03's QSignal-throttling):
    ``progress_cb(pct)`` fires only when ``int(100 * decoded / frames)``
    *strictly increases* — so at most 101 calls total (0..100 inclusive),
    monotonic, no duplicates.

Cancellation contract:
    ``cancel_check()`` is polled between blocks. On ``True``, any partial
    ``<dst>.tmp`` file is deleted and :class:`BuildCancelled` is raised.
    The caller (Plan 03) catches ``BuildCancelled`` and silently aborts
    the build worker.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

from marmelade.audio.audio_file import BLOCK_SAMPLES, iter_blocks, probe
from marmelade.audio.proxy_cache import (
    DEFAULT_SAMPLES_PER_PIXEL,
    ProxyHeader,
    load_proxy,
    write_proxy,
)


class BuildCancelled(RuntimeError):
    """Raised by :func:`build_proxy` when ``cancel_check()`` returns True.

    The partial ``<dst_path>.tmp`` file is removed before the exception
    propagates so no half-written proxy is left in the cache.
    """


def windowed_minmax(samples_f32: np.ndarray, spp: int) -> np.ndarray:
    """Return ``(N, 2) int16`` min/max pairs for ``samples_f32`` reshaped to ``spp``-wide windows.

    Behaviour:
        * Input shorter than ``spp`` → shape ``(0, 2)`` int16.
        * Leading ``n_full * spp`` samples are reshaped to ``(n_full, spp)``
          for vectorised min/max along axis 1.
        * Values are clipped to ``[-1.0, 1.0]`` BEFORE the int16 scale (RESEARCH
          Open Q #2 — inter-sample MP3 peaks otherwise overflow int16).
        * Float samples are scaled by 32767 and cast to ``int16``.

    Trailing samples (less than ``spp``) are intentionally NOT emitted by this
    helper — :func:`build_proxy` handles end-of-stream padding so the
    function stays pure and stateless.
    """
    if samples_f32.size < spp:
        return np.zeros((0, 2), dtype=np.int16)

    n_full = samples_f32.size // spp
    view = samples_f32[: n_full * spp].reshape(n_full, spp)

    mins = view.min(axis=1)
    maxs = view.max(axis=1)

    # Inter-sample MP3 peaks can overshoot ±1.0; clip BEFORE int16 scale
    # so we never overflow (Open Q #2 — see RESEARCH).
    mins = np.clip(mins, -1.0, 1.0)
    maxs = np.clip(maxs, -1.0, 1.0)

    out = np.empty((n_full, 2), dtype=np.int16)
    out[:, 0] = (mins * 32767.0).astype(np.int16)
    out[:, 1] = (maxs * 32767.0).astype(np.int16)
    return out


def _pair_from_short_window(window: np.ndarray) -> np.ndarray:
    """Emit a single (1, 2) int16 pair from a short trailing window."""
    if window.size == 0:
        return np.zeros((0, 2), dtype=np.int16)
    lo = float(np.clip(window.min(), -1.0, 1.0))
    hi = float(np.clip(window.max(), -1.0, 1.0))
    pair = np.empty((1, 2), dtype=np.int16)
    pair[0, 0] = np.int16(lo * 32767.0)
    pair[0, 1] = np.int16(hi * 32767.0)
    return pair


def build_proxy(
    src_path: str | os.PathLike,
    dst_path: str | os.PathLike,
    samples_per_pixel: int = DEFAULT_SAMPLES_PER_PIXEL,
    progress_cb: Callable[[int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> ProxyHeader:
    """Build a BBC-v2 ``.dat`` proxy from ``src_path`` to ``dst_path``.

    Returns:
        The :class:`ProxyHeader` parsed back from the freshly-written
        ``.dat`` (so the caller has an authoritative view of dimensions).

    Raises:
        BuildCancelled: when ``cancel_check()`` returns True at any
            between-block poll. The partial ``<dst_path>.tmp`` (if any) is
            removed before the exception propagates.
    """
    src_p = Path(src_path)
    dst_p = Path(dst_path)
    tmp_p = dst_p.with_suffix(dst_p.suffix + ".tmp")

    info = probe(src_p)
    total_frames = info.frames
    if total_frames <= 0:
        # Empty source — write an empty proxy and return.
        write_proxy(
            dst_p,
            sample_rate=info.sample_rate,
            samples_per_pixel=samples_per_pixel,
            pairs_int16=np.zeros((0, 2), dtype=np.int16),
        )
        _, header = load_proxy(dst_p)
        return header

    last_pct = -1
    buffer = np.empty(0, dtype=np.float32)
    pair_chunks: list[np.ndarray] = []
    decoded = 0

    try:
        for block, _offset in iter_blocks(
            src_p, block_samples=BLOCK_SAMPLES, mono=True
        ):
            # Poll cancel between blocks BEFORE doing any work for this block.
            if cancel_check is not None and cancel_check():
                raise BuildCancelled()

            # Accumulate into the leftover buffer so windows never straddle.
            if buffer.size == 0:
                buffer = np.ascontiguousarray(block, dtype=np.float32)
            else:
                buffer = np.concatenate([buffer, block])

            n_full = buffer.size // samples_per_pixel
            if n_full > 0:
                consumed = n_full * samples_per_pixel
                pair_chunks.append(
                    windowed_minmax(buffer[:consumed], samples_per_pixel)
                )
                buffer = buffer[consumed:].copy()  # keep remainder

            decoded += int(block.size)
            pct = int(100 * decoded / total_frames)
            if pct > last_pct and progress_cb is not None:
                progress_cb(pct)
                last_pct = pct
            elif pct > last_pct:
                last_pct = pct

        # Final remainder: pad to a full window so we don't drop the tail.
        if buffer.size > 0:
            padded = np.empty(samples_per_pixel, dtype=np.float32)
            padded[: buffer.size] = buffer
            # Pad with the last sample so silence isn't injected.
            padded[buffer.size :] = buffer[-1] if buffer.size else 0.0
            pair_chunks.append(
                windowed_minmax(padded, samples_per_pixel)
            )

        # Ensure the final 100% progress notification fires even when the
        # last block did not bump the integer percentage.
        if progress_cb is not None and last_pct < 100:
            progress_cb(100)
            last_pct = 100

        if pair_chunks:
            pairs = np.concatenate(pair_chunks, axis=0)
        else:
            pairs = np.zeros((0, 2), dtype=np.int16)

        write_proxy(
            dst_p,
            sample_rate=info.sample_rate,
            samples_per_pixel=samples_per_pixel,
            pairs_int16=pairs,
        )
    except BuildCancelled:
        # Best-effort cleanup of the .tmp sibling left by a partial
        # write_proxy. Note: write_proxy is atomic, but if we cancelled
        # BEFORE write_proxy then no tmp exists; either way, try-and-ignore.
        try:
            os.remove(str(tmp_p))
        except FileNotFoundError:
            pass
        raise

    _, header = load_proxy(dst_p)
    return header
