"""Region export builder: proxy WAV → MP3 (320 CBR) or WAV (float32) clip.

Pure NumPy + stdlib + soundfile + pedalboard. Like the rest of ``audio/``,
this module has NO toolkit imports (N-3 invariant). The matching
:mod:`marmelade.audio.export_worker` wraps :func:`export_region` in a
QRunnable and bridges its progress/cancel hooks to Qt signals via a thin
shell.

Algorithm (Phase 3 EXP-01 / D-A4-4):
    Open the source proxy WAV via :class:`pedalboard.io.AudioFile`, seek to
    the region's start frame, read in :data:`BLOCK_SAMPLES`-frame chunks
    until the end frame, apply auto-scaled linear fade ramps at head + tail
    (D-A4-5: per-side duration = ``min(2.0, region_duration / 2.0)`` seconds,
    linear), and write block-by-block to ``<dst>.tmp`` via either:
        * :class:`pedalboard.io.AudioFile` (MP3, ``quality=320``) — 320 kbps
          CBR via pedalboard's bundled JUCE codec.
        * :class:`soundfile.SoundFile` (WAV, ``subtype='FLOAT'``,
          ``format='RF64'``) — float32 IEEE_FLOAT; mirrors Phase 2.1
          :mod:`marmelade.audio.audio_proxy_builder` verbatim.

    On success, atomically rename to the final path via :func:`os.replace`.

Fade ramp shape (W-7):
    ``np.linspace(0.0, 1.0, fade_n, endpoint=True, dtype=np.float32)``
    so the FIRST fade-in sample is exactly 0.0 AND the LAST is exactly 1.0.
    Fade-out mirror: ``np.linspace(1.0, 0.0, fade_n, endpoint=True,
    dtype=np.float32)``.

Memory contract (CLAUDE.md):
    Peak RAM is one block (~1 MiB for stereo float32 at
    :data:`BLOCK_SAMPLES` frames). The full region is NEVER materialised in
    memory whole. Verified by
    :mod:`tests.unit.audio.test_export_builder`.

Cancellation contract (mirrors
:mod:`marmelade.audio.audio_proxy_builder`):
    ``cancel_check()`` is polled before each block. On True, the partial
    ``<dst>.tmp`` is removed and :class:`BuildCancelled` is re-raised.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from pedalboard.io import AudioFile as PedalboardAudioFile

# REUSE VERBATIM from peak_builder per D-17 — the cross-worker cancellation
# contract uses ONE exception class.
from marmelade.audio.peak_builder import BuildCancelled  # noqa: F401 — re-export


__all__ = [
    "export_region",
    "BuildCancelled",
    "BLOCK_SAMPLES",
    "_apply_fade_pedalboard_layout",
    "_apply_fade_soundfile_layout",
]

# quick-260621-gfq — the export-time normalize path (and its mgu-only
# ``_InRamSource`` adapter) was REMOVED. Normalize now lives strictly inside
# the mastering chain (its final stage); raw export streams the source
# verbatim. The honest normalized bytes come from the mastered-cache export
# path. ``normalize_array`` stays in marmelade.audio.normalize for the
# chain stage to use.


# Block size mirrors :mod:`marmelade.audio.audio_proxy_builder.BLOCK_SAMPLES`
# (~3 s @ 44.1 kHz). 131072 frames × 2 channels × float32 = ~1 MiB peak per
# block — well within the CLAUDE.md memory contract.
BLOCK_SAMPLES = 131_072

# Allowed format set — matches the resolver's allow-list. D-A4-4 LOCKED
# dual-format: 320 kbps CBR MP3 and float32 WAV are first-class.
_SUPPORTED_FMTS = frozenset({"mp3", "wav"})


def export_region(
    proxy_path: str | os.PathLike,
    dst_path: str | os.PathLike,
    start_frame: int,
    end_frame: int,
    fade_frames: int,
    fmt: str,
    sample_rate: int,
    progress_cb: Callable[[int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    *,
    source_path: os.PathLike | str | None = None,
) -> None:
    """Export ``[start_frame, end_frame)`` from ``proxy_path`` (or ``source_path``) to ``dst_path``.

    Args:
        proxy_path: Phase 2.1 audio proxy ``.proxy.wav`` path (float32 stereo).
            Used as the audio source when ``source_path`` is None
            (existing Phase 3 behavior).
        dst_path: Final output path. The file extension is informational —
            the writer is selected via ``fmt``.
        start_frame, end_frame: Frame indices into the actual audio source
            (``source_path`` when provided, else ``proxy_path``).
        fade_frames: Fade-in AND fade-out length (each, in frames). Caller
            computes as ``int(min(2.0, region_dur / 2.0) * sr)``. Clamped
            internally at ``total_frames // 2`` so fade-in and fade-out
            never overlap.
        fmt: ``"mp3"`` or ``"wav"``.
        sample_rate: Proxy's sample rate (passed verbatim to the writer).
        progress_cb: 0..100 strictly-monotone integer-percent callback.
        cancel_check: Zero-arg callable returning bool; polled before each
            block.
        source_path: Phase 7 D-20 — optional keyword-only override for the
            audio source. When provided, the function reads audio from
            ``source_path`` instead of ``proxy_path``. ``start_frame`` /
            ``end_frame`` are interpreted against ``source_path``'s frame
            timeline (callers using the mastered-cache override pass
            ``start_frame=0, end_frame=mastered_cache_total_frames`` because
            the mastered cache holds exactly the keeper-region duration).
            Existing Phase 3 callers that omit this argument receive
            identical behavior to before. ADDITIVE — does NOT alter the
            fade-in/fade-out shape, MP3/WAV format dispatch, atomic-write
            pattern, or auto-naming.

    Writes to ``<dst_path>.tmp`` first, then atomically renames on success.

    Raises:
        BuildCancelled: ``cancel_check()`` returned True.
        ValueError: ``fmt`` not supported or ``end_frame <= start_frame``.
        OSError / sf.SoundFileError / pedalboard exceptions: propagated as-is.
    """
    if fmt not in _SUPPORTED_FMTS:
        raise ValueError(f"Unsupported fmt: {fmt!r}")
    if end_frame <= start_frame:
        raise ValueError(
            f"end_frame ({end_frame}) must be > start_frame ({start_frame})"
        )
    if fade_frames < 0:
        raise ValueError(f"fade_frames must be >= 0, got {fade_frames}")
    total_frames = int(end_frame - start_frame)
    # Cap fade_frames at total_frames // 2 so fade-in and fade-out never
    # overlap. CR-01 — when total_frames == 2 * fade_frames (i.e. the
    # region duration is exactly twice the fade duration; happens for any
    # region <= 4 s under the caller's fade_sec = min(2.0, dur/2.0)
    # auto-scale), the fade-in window [0, fade_frames) and the fade-out
    # window [fade_frames, 2*fade_frames) are ADJACENT — no overlap, no
    # full-amplitude plateau in the middle. The W-7 endpoint=True
    # invariant still pins the boundary samples (fade-in's last sample
    # reaches 1.0, fade-out's first sample reaches 1.0, both at adjacent
    # indices), but no sample in the body sits at a sustained 1.0.
    # Result: a triangle-window export for regions <= 4 s. This is
    # perceptually fine for jam clips (the listener hears a smooth
    # crescendo+decrescendo with the peak sample at full amplitude), so
    # we accept the behavior as documented rather than reduce fade_frames
    # further to carve out a plateau.
    fade_frames = min(int(fade_frames), total_frames // 2)

    # Phase 7 D-20 — when ``source_path`` is provided, it REPLACES
    # ``proxy_path`` as the actual audio source. Phase 7 Plan 07-06 Phase C
    # uses this to pipe a per-keeper mastered cache WAV through the export
    # pipeline (fade + format + atomic write) instead of the source proxy.
    # Existing Phase 3 callers (no ``source_path`` kwarg) get the original
    # ``proxy_path`` behavior verbatim.
    src_p = Path(source_path) if source_path is not None else Path(proxy_path)
    dst_p = Path(dst_path)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    # Pedalboard's AudioFile detects the container by file extension, so the
    # in-flight path must end in ``.mp3`` / ``.wav`` for the codec dispatch
    # to land on the right writer. We append ``.tmp`` IN FRONT OF the
    # extension (``out.mp3`` → ``out.tmp.mp3``) so pedalboard still sees
    # ``.mp3``; ``os.replace`` at the end renames to the canonical name.
    tmp_p = dst_p.with_name(dst_p.stem + ".tmp" + dst_p.suffix)

    try:
        with PedalboardAudioFile(str(src_p), "r") as src:
            src.seek(int(start_frame))
            n_channels = int(src.num_channels)

            # quick-260621-gfq — the export-time normalize branch was removed.
            # Raw export streams the source verbatim (the pre-mgu behavior);
            # normalize is now the mastering chain's final stage and the
            # mastered-cache export path carries the normalized bytes.
            stream_src = src

            if fmt == "mp3":
                _stream_blocks_mp3(
                    src=stream_src,
                    tmp_p=tmp_p,
                    total_frames=total_frames,
                    fade_frames=fade_frames,
                    n_channels=n_channels,
                    sample_rate=int(sample_rate),
                    progress_cb=progress_cb,
                    cancel_check=cancel_check,
                )
            else:  # wav
                _stream_blocks_wav(
                    src=stream_src,
                    tmp_p=tmp_p,
                    total_frames=total_frames,
                    fade_frames=fade_frames,
                    n_channels=n_channels,
                    sample_rate=int(sample_rate),
                    progress_cb=progress_cb,
                    cancel_check=cancel_check,
                )
        # Successful close — atomic rename.
        os.replace(str(tmp_p), str(dst_p))
    except BuildCancelled:
        # D-17 — best-effort .tmp cleanup BEFORE re-raising.
        try:
            os.remove(str(tmp_p))
        except FileNotFoundError:
            pass
        raise
    except BaseException:
        # Other failure (OSError, codec error, etc.) — also cleanup .tmp.
        try:
            os.remove(str(tmp_p))
        except FileNotFoundError:
            pass
        raise


def _stream_blocks_mp3(
    src,
    tmp_p: Path,
    total_frames: int,
    fade_frames: int,
    n_channels: int,
    sample_rate: int,
    progress_cb: Callable[[int], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> None:
    """Block-streaming write to a pedalboard ``AudioFile`` (MP3 path).

    Pedalboard read returns shape ``(n_channels, n_frames)``; pedalboard
    write expects the same shape — no transpose.
    """
    with PedalboardAudioFile(
        str(tmp_p), "w",
        samplerate=int(sample_rate),
        num_channels=n_channels,
        quality=320,
    ) as dst:
        frames_written = 0
        last_pct = -1
        while frames_written < total_frames:
            if cancel_check is not None and cancel_check():
                raise BuildCancelled()
            n_to_read = min(BLOCK_SAMPLES, total_frames - frames_written)
            block = src.read(n_to_read)
            # block.shape: (n_channels, actual_n)
            actual_n = (
                block.shape[1] if block.ndim == 2 else block.shape[0]
            )
            if actual_n == 0:
                break
            # Ensure float32 + writable for in-place fade arithmetic.
            block = np.ascontiguousarray(block, dtype=np.float32)
            _apply_fade_pedalboard_layout(
                block, frames_written, total_frames, fade_frames
            )
            dst.write(block)
            frames_written += actual_n
            if progress_cb is not None:
                pct = int(100 * frames_written / max(total_frames, 1))
                if pct > last_pct:
                    progress_cb(pct)
                    last_pct = pct


def _stream_blocks_wav(
    src,
    tmp_p: Path,
    total_frames: int,
    fade_frames: int,
    n_channels: int,
    sample_rate: int,
    progress_cb: Callable[[int], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> None:
    """Block-streaming write to ``soundfile.SoundFile`` (WAV float32 path).

    Mirrors :mod:`marmelade.audio.audio_proxy_builder` verbatim:
    ``subtype='FLOAT'``, ``format='RF64'`` for >4 GiB support. Pedalboard
    read returns ``(n_channels, n_frames)``; soundfile write expects
    ``(n_frames, n_channels)`` — transpose per block.
    """
    with sf.SoundFile(
        str(tmp_p),
        mode="w",
        samplerate=int(sample_rate),
        channels=n_channels,
        subtype="FLOAT",
        format="RF64",
    ) as dst:
        frames_written = 0
        last_pct = -1
        while frames_written < total_frames:
            if cancel_check is not None and cancel_check():
                raise BuildCancelled()
            n_to_read = min(BLOCK_SAMPLES, total_frames - frames_written)
            block = src.read(n_to_read)
            actual_n = (
                block.shape[1] if block.ndim == 2 else block.shape[0]
            )
            if actual_n == 0:
                break
            # Transpose pedalboard (nchan, frames) → soundfile (frames, nchan)
            # and ensure writable contiguous float32.
            block_sf = np.ascontiguousarray(block.T, dtype=np.float32)
            _apply_fade_soundfile_layout(
                block_sf, frames_written, total_frames, fade_frames
            )
            dst.write(block_sf)
            frames_written += actual_n
            if progress_cb is not None:
                pct = int(100 * frames_written / max(total_frames, 1))
                if pct > last_pct:
                    progress_cb(pct)
                    last_pct = pct


def _apply_fade_pedalboard_layout(
    block: np.ndarray,            # shape (n_channels, n_frames)
    block_start_frame: int,
    region_total_frames: int,
    fade_frames: int,
) -> None:
    """Apply linear fade-in [0, fade_frames) and fade-out [total-fade, total) in place.

    Block layout: ``(n_channels, n_frames)`` — pedalboard's native shape.

    W-7 invariant: ramps use ``np.linspace(..., endpoint=True)``. The very
    first fade-in sample is exactly 0.0, the very last fade-in sample is
    exactly 1.0, the very first fade-out sample is exactly 1.0, the very
    last fade-out sample is exactly 0.0.
    """
    if fade_frames <= 0:
        return
    n = block.shape[1]
    block_end_frame = block_start_frame + n
    # ----- fade-in window [0, fade_frames) -----
    if block_start_frame < fade_frames:
        lo = 0
        hi = min(n, fade_frames - block_start_frame)
        global_start_idx = block_start_frame
        global_end_idx = block_start_frame + hi
        if fade_frames > 1:
            ramp = np.linspace(
                global_start_idx / (fade_frames - 1),
                (global_end_idx - 1) / (fade_frames - 1),
                hi,
                dtype=np.float32,
                endpoint=True,
            )
        else:
            # Degenerate single-sample fade — guard against divide-by-zero.
            # CR-02 — that single sample is BOTH the first AND the last
            # sample of the fade window, so under the W-7 endpoint=True
            # invariant it must be 0.0 (the boundary), not 1.0. The
            # caller's fade_sec auto-scale (min(2.0, dur/2.0)) never
            # produces fade_frames=1 in practice (it would require a
            # region of ~45 µs which fails the end > start check
            # earlier), so this branch is dead code in production —
            # but the semantic must still be correct.
            ramp = np.zeros(hi, dtype=np.float32)
        block[:, lo:hi] *= ramp[None, :]
    # ----- fade-out window [total-fade, total) -----
    fade_out_start = region_total_frames - fade_frames
    if block_end_frame > fade_out_start:
        lo = max(0, fade_out_start - block_start_frame)
        hi = n
        j_start = (block_start_frame + lo) - fade_out_start
        j_end = (block_start_frame + hi) - fade_out_start
        if fade_frames > 1:
            ramp = np.linspace(
                1.0 - j_start / (fade_frames - 1),
                1.0 - (j_end - 1) / (fade_frames - 1),
                hi - lo,
                dtype=np.float32,
                endpoint=True,
            )
        else:
            # CR-02 — single-sample fade-out: same boundary semantic as
            # the fade-in (the lone sample is both the first AND the
            # last of the fade-out window; boundary value is 0.0).
            ramp = np.zeros(hi - lo, dtype=np.float32)
        block[:, lo:hi] *= ramp[None, :]


def _apply_fade_soundfile_layout(
    block: np.ndarray,            # shape (n_frames, n_channels)
    block_start_frame: int,
    region_total_frames: int,
    fade_frames: int,
) -> None:
    """Apply linear fade-in + fade-out to a soundfile-layout block in place.

    Same W-7 endpoint=True invariant as the pedalboard helper.
    """
    if fade_frames <= 0:
        return
    n = block.shape[0]
    block_end_frame = block_start_frame + n
    if block_start_frame < fade_frames:
        lo = 0
        hi = min(n, fade_frames - block_start_frame)
        global_start_idx = block_start_frame
        global_end_idx = block_start_frame + hi
        if fade_frames > 1:
            ramp = np.linspace(
                global_start_idx / (fade_frames - 1),
                (global_end_idx - 1) / (fade_frames - 1),
                hi,
                dtype=np.float32,
                endpoint=True,
            )
        else:
            # CR-02 — single-sample fade boundary semantic. See the
            # pedalboard helper for the full reasoning; here the lone
            # sample is the boundary of the fade window and must be 0.0.
            ramp = np.zeros(hi, dtype=np.float32)
        block[lo:hi, :] *= ramp[:, None]
    fade_out_start = region_total_frames - fade_frames
    if block_end_frame > fade_out_start:
        lo = max(0, fade_out_start - block_start_frame)
        hi = n
        j_start = (block_start_frame + lo) - fade_out_start
        j_end = (block_start_frame + hi) - fade_out_start
        if fade_frames > 1:
            ramp = np.linspace(
                1.0 - j_start / (fade_frames - 1),
                1.0 - (j_end - 1) / (fade_frames - 1),
                hi - lo,
                dtype=np.float32,
                endpoint=True,
            )
        else:
            # CR-02 — single-sample fade-out boundary semantic.
            ramp = np.zeros(hi - lo, dtype=np.float32)
        block[lo:hi, :] *= ramp[:, None]
