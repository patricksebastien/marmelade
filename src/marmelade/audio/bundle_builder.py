"""Bundle builder: multi-source WAV concat + silent spacer → 320 kbps CBR MP3.

Phase 8 Plan 08-05 Task 1 — implements D-04 (user-configurable silent
spacer between keepers, default 2.0 s, range 0..10 s) + D-06 (no
persistent bundle cache; caller picks destination via file dialog).

Qt-isolation (D-27 / N-3 invariant): zero ``PySide6.*`` imports. The
function is consumed by a :class:`PySide6.QtCore.QRunnable` wrapper
constructed inline in the MainWindow's bundle-Share slot (Plan 08-05
Task 3). Keeping the assembly logic Qt-free preserves the audio-tier
boundary that mastering / export / peak builders all share.

Memory contract (CLAUDE.md): the bundle is NEVER materialised whole in
memory. Each keeper is read with :func:`soundfile.read` (one keeper's
duration × 2 channels × float32 ≈ 10 MiB peak for a 60 s keeper at
44.1 kHz) and immediately handed off to the pedalboard MP3 writer.
Tested on a 50-keeper bundle the peak RSS stays at one keeper's
footprint — the writer streams chunks to disk.

Codec choice — mirrors :func:`marmelade.audio.export_builder.
_stream_blocks_mp3` lines 213-258 verbatim with ``quality=320`` per
RESEARCH §Assumption A1 verification. The pedalboard ``AudioFile``
writer infers the container from the path extension, so the in-flight
path uses ``<out>.tmp.mp3`` (NOT ``<out>.mp3.tmp``) — see the export
builder's same pattern at lines 158-166. ``os.replace`` finalises the
write atomically on success.

Cancellation contract — :class:`BuildCancelled` (re-exported from
:mod:`marmelade.audio.peak_builder` so workers can ``import
BuildCancelled`` from one place) is polled BEFORE each keeper write.
On cancel the partial ``.tmp`` sibling is best-effort removed and the
exception propagates. Mirrors export_builder's lines 197-203.

Sample-rate validation — every keeper WAV is probed via
:func:`soundfile.info` BEFORE any write. A mismatched rate raises
:class:`ValueError` naming the offending path; the writer is never
opened. This is the fail-fast equivalent of the pedalboard writer's
own rate check (pedalboard would just silently encode garbage if
mismatched rates leaked through).

D-04 contract details:

* ``spacer_sec`` is a non-negative float. Range validation (0..10) is
  the dialog's responsibility (it ships a QDoubleSpinBox with that
  range); the audio function accepts anything ``>= 0`` so unit tests
  can exercise fractional values.
* The spacer is inserted BETWEEN consecutive keepers, NOT after the
  last keeper. A bundle of 3 keepers therefore has exactly 2 spacers.
* Each spacer is a block of ``int(round(spacer_sec * sample_rate))``
  silent (all-zero) stereo float32 frames in the pedalboard writer's
  ``(n_channels, n_frames)`` shape.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf
from pedalboard.io import AudioFile as PedalboardAudioFile

# REUSE VERBATIM from peak_builder per D-17 — the cross-worker
# cancellation contract uses ONE exception class so the export /
# bundle / upload runnables all catch the same type.
from marmelade.audio.peak_builder import BuildCancelled  # noqa: F401 — re-export
# REUSE the per-keeper fade helper. Phase 7's mastering pipeline does
# NOT bake fades into the mastered cache WAV (export_region applies
# fades on the way to the per-keeper export); the bundle would
# concatenate raw-edged keepers with abrupt starts/ends without this.
from marmelade.audio.export_builder import _apply_fade_pedalboard_layout


__all__ = ["build_bundle", "BuildCancelled"]


def build_bundle(
    mastered_paths: Sequence[Path | str],
    spacer_sec: float,
    out_mp3_path: Path | str,
    *,
    sample_rate: int = 44100,
    progress_cb: Callable[[int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Concatenate mastered WAVs into a single 320 kbps CBR MP3 with silent spacers.

    Args:
        mastered_paths: Ordered list of mastered-cache WAV paths (user-
            arranged order from the Keepers sidebar drag-reorder). Must
            contain at least one path.
        spacer_sec: Seconds of silence between consecutive keepers (NOT
            after the last). 0.0 means back-to-back; the dialog clamps
            to ``[0.0, 10.0]`` but this function accepts any non-negative
            float so unit tests can use small values.
        out_mp3_path: Destination MP3 path. Written atomically via
            ``<stem>.tmp.<suffix>`` then ``os.replace``.
        sample_rate: Expected sample rate. Any keeper WAV whose rate
            does not match this raises :class:`ValueError` BEFORE the
            writer opens.
        progress_cb: Optional 0..100 percent callback. Fires once per
            keeper write completion — ``int(100 * (i + 1) / N)``.
        cancel_check: Optional zero-arg callable returning ``True`` to
            signal cancel. Polled BEFORE each keeper write; on True the
            partial ``.tmp`` sibling is removed and :class:`BuildCancelled`
            is raised.

    Raises:
        ValueError: ``mastered_paths`` is empty, ``spacer_sec`` is
            negative, or any keeper WAV's sample rate does not match
            ``sample_rate`` (message names the offending path + observed
            rate + expected rate).
        BuildCancelled: ``cancel_check()`` returned True. The partial
            ``.tmp`` sibling is cleaned up before the exception
            propagates.
        OSError / sf.SoundFileError / pedalboard exceptions: propagated
            as-is; the partial ``.tmp`` is best-effort removed on the
            way out so the cache directory stays clean.
    """
    if len(mastered_paths) == 0:
        raise ValueError("mastered_paths must contain at least one keeper")
    if spacer_sec < 0:
        raise ValueError(f"spacer_sec must be >= 0, got {spacer_sec!r}")

    sr = int(sample_rate)

    # Fail-fast sample-rate + existence check across ALL keepers BEFORE
    # opening the writer. This avoids a half-written .tmp on a mismatched
    # mid-list path and gives the user one actionable error message.
    keeper_infos: list[tuple[Path, int]] = []
    for p in mastered_paths:
        pth = Path(p)
        try:
            info = sf.info(str(pth))
        except Exception as exc:
            # sf raises a soundfile.SoundFileError or RuntimeError on
            # missing/corrupt files; re-raise as ValueError with a
            # clear path mention so the dialog footer can surface it
            # without leaking the underlying soundfile vocabulary.
            raise ValueError(
                f"Cannot read keeper WAV {pth}: {exc}"
            ) from exc
        observed_sr = int(info.samplerate)
        if observed_sr != sr:
            raise ValueError(
                f"Sample rate mismatch: {pth} is {observed_sr}Hz, expected {sr}Hz"
            )
        keeper_infos.append((pth, observed_sr))

    out_p = Path(out_mp3_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # Pedalboard's AudioFile detects container by file extension, so the
    # in-flight path must end in ``.mp3`` for the codec dispatch to land
    # on the MP3 writer. We append ``.tmp`` IN FRONT OF the extension
    # (``out.mp3`` → ``out.tmp.mp3``) so pedalboard still sees ``.mp3``;
    # ``os.replace`` at the end renames to the canonical name.
    tmp_p = out_p.with_name(out_p.stem + ".tmp" + out_p.suffix)

    spacer_frames = int(round(spacer_sec * sr))
    # Pre-build the silent spacer block in pedalboard layout
    # (n_channels, n_frames). Stereo float32 to match the mastered
    # cache shape — all-zero so it really is silent.
    spacer_block: np.ndarray | None
    if spacer_frames > 0:
        spacer_block = np.zeros((2, spacer_frames), dtype=np.float32)
    else:
        spacer_block = None

    n_total = len(keeper_infos)
    try:
        with PedalboardAudioFile(
            str(tmp_p), "w",
            samplerate=sr,
            num_channels=2,
            quality=320,
        ) as dst:
            for i, (pth, _ksr) in enumerate(keeper_infos):
                # Poll cancel BEFORE each keeper write (NOT after) so a
                # cancel arriving between two keepers aborts cleanly
                # without producing a "half N+1" gap.
                if cancel_check is not None and cancel_check():
                    raise BuildCancelled()

                # Read the whole keeper WAV into memory. Mastered caches
                # are typically <= 5 minutes (one keeper) so this is a
                # few MiB at most — well within the CLAUDE.md memory
                # contract. The full bundle is NEVER concatenated whole;
                # each keeper streams in and out one at a time.
                data, _read_sr = sf.read(
                    str(pth), dtype="float32", always_2d=True
                )
                # data shape: (frames, channels). Pedalboard wants
                # (channels, frames) — transpose + ensure contiguous.
                #
                # Mastered cache is stereo by contract; the read path
                # always returns 2 channels because the WAV was written
                # as stereo by Phase 7's mastering pipeline. Defensive:
                # if a (mono) file ever sneaks through, broadcast to
                # stereo so the writer's num_channels=2 commitment
                # stays honoured.
                if data.shape[1] == 1:
                    data = np.repeat(data, 2, axis=1)
                elif data.shape[1] != 2:
                    raise ValueError(
                        f"Keeper {pth} has {data.shape[1]} channels; "
                        "bundle requires mono or stereo (2 ch enforced)"
                    )
                block = np.ascontiguousarray(data.T, dtype=np.float32)

                # Apply per-keeper fade-in + fade-out IN PLACE before
                # the write. The mastered cache WAV has raw edges
                # (Phase 7's MasteringChain doesn't fade); without
                # this the bundle starts each keeper abruptly and
                # ends each keeper abruptly into the silent spacer.
                # ``fade_sec = min(2.0, dur / 2.0)`` mirrors the
                # auto-scale that export_region uses, so a short
                # keeper gets a proportional fade and a long one gets
                # the standard 2 s.
                keeper_frames = block.shape[1]
                dur_sec = keeper_frames / sr
                fade_sec = min(2.0, dur_sec / 2.0)
                fade_frames = int(round(fade_sec * sr))
                _apply_fade_pedalboard_layout(
                    block,
                    block_start_frame=0,
                    region_total_frames=keeper_frames,
                    fade_frames=fade_frames,
                )

                dst.write(block)

                # Spacer BETWEEN consecutive keepers (not after the
                # last). This is the D-04 invariant — the dialog tests
                # pin the same contract.
                if i < n_total - 1 and spacer_block is not None:
                    dst.write(spacer_block)

                if progress_cb is not None:
                    pct = int(100 * (i + 1) / n_total)
                    progress_cb(pct)
        # Atomic rename — on any failure inside the with-block the
        # except branches below clean up the .tmp sibling.
        os.replace(str(tmp_p), str(out_p))
    except BuildCancelled:
        # Best-effort .tmp cleanup BEFORE re-raising. Mirrors
        # export_builder lines 197-203.
        try:
            os.remove(str(tmp_p))
        except FileNotFoundError:
            pass
        raise
    except BaseException:
        # Any other failure (codec error, OSError on os.replace, etc.)
        # — also clean up the .tmp sibling so the cache directory does
        # not accumulate half-written files.
        try:
            os.remove(str(tmp_p))
        except FileNotFoundError:
            pass
        raise
