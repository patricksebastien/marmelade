"""Video assembly — image + audio → MP4 via ffmpeg subprocess (Phase 8 YT-03).

Qt-free per D-27 (N-3 invariant): zero ``PySide6.*`` imports. The single
public function :func:`build_video` is consumed by
:class:`marmelade.youtube.upload_runnable.YouTubeUploadRunnable` as
the synchronous prelude to the resumable YouTube upload loop.

Implements:

* **D-11** — ffmpeg subprocess via :mod:`subprocess`. The binary is
  discovered via :func:`imageio_ffmpeg.get_ffmpeg_exe` which returns
  the bundled static binary (~76 MiB extracted, ffmpeg 7.0.2) shipped
  in the wheel; falls back to system ``ffmpeg`` on PATH when the
  bundled binary is unavailable for any reason.
* **D-12** — H.264 + AAC MP4 with exactly the arg set
  ``-loop 1 -framerate 2 -tune stillimage -c:v libx264 -c:a aac
  -b:a 192k -pix_fmt yuv420p -shortest``. The ``yuv420p`` pixel format
  is required for QuickTime + iOS playback (without it, mobile users
  download a file their default player refuses to open).
* **D-13** — ``-loop 1`` on the image input keeps the encoded video
  showing the static frame for the audio's full duration; ``-shortest``
  cuts the video stream when the audio stream ends.
* **T-08-04-06 mitigation** (revision iter 1 N2) — ``subprocess.run``
  is invoked with ``timeout=600`` (10 minutes). On
  :class:`subprocess.TimeoutExpired` the child process is killed, the
  in-flight ``.tmp`` sibling is removed, and a :class:`RuntimeError`
  is raised with an actionable message. Bounded-duration safety net so
  a legitimately-stuck encode surfaces as a clean error in the dialog
  footer rather than a hung modal.

Atomic write contract — mirrors
:func:`marmelade.audio.export_builder.export_region` lines 158-210:
write to ``<out>.tmp.<ext>`` first so pedalboard/ffmpeg's codec
dispatch still sees the canonical extension, then ``os.replace`` to the
final path on success. On any failure (cancel, non-zero exit, timeout,
generic exception) the ``.tmp`` sibling is best-effort removed so the
cache directory stays clean.

Cancellation contract — :class:`BuildCancelled` (re-exported from
:mod:`marmelade.audio.peak_builder` so workers can import a single
exception class) is polled exactly ONCE before launching the subprocess.
Once ffmpeg starts it runs to completion (or its own timeout) — this is
the v1 acceptance per RESEARCH §Pitfall 6 (encode is bounded by the
audio duration; ``subprocess.Popen`` + kill mid-stream is a Plan 08-06
follow-up if user feedback demands it).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

import soundfile as sf
from imageio_ffmpeg import get_ffmpeg_exe

# REUSE VERBATIM from peak_builder per D-17 — the cross-worker
# cancellation contract uses ONE exception class.
from marmelade.audio.peak_builder import BuildCancelled  # noqa: F401 — re-export


__all__ = ["build_video", "BuildCancelled"]


# T-08-04-06 mitigation: bound the subprocess so a stuck encode surfaces
# as an actionable RuntimeError rather than a hung dialog. 600 s (10 min)
# is a deliberately generous cap — typical keeper encodes finish in 10-30
# seconds at 2 fps still-image. A multi-hour bundle MP3 at 192 kbps mux
# is well under 60 s.
_FFMPEG_TIMEOUT_SEC: int = 600

# Tail padding (seconds) appended AFTER the audio ends so a fade-out
# has room to breathe before the video cuts. Without this the muxer
# stops the instant the audio sample data ends — fades that taper into
# the last ~50 ms feel chopped and AAC encoder framing can clip the
# very tail. 1.5 s feels natural (matches a YouTube end-of-video pause
# before the "Up Next" overlay) without padding so much that the
# trailing silence becomes awkward. The audio stream is extended with
# silence via ``-af apad=pad_dur=...`` so the listener hears the fade
# end → silence, not an abrupt video cut.
_TAIL_PAD_SEC: float = 1.5

# ffmpeg D-12 arg list pieces — kept module-level for grep-pin auditing.
# The full argv is assembled inside :func:`build_video` because the
# input/output paths interleave around these flags.
_FFMPEG_VIDEO_FLAGS: list[str] = [
    "-c:v", "libx264", "-tune", "stillimage",
]
_FFMPEG_AUDIO_FLAGS: list[str] = [
    "-c:a", "aac", "-b:a", "192k",
]
_FFMPEG_CONTAINER_FLAGS: list[str] = [
    "-pix_fmt", "yuv420p", "-shortest",
]


def build_video(
    image_path: str | os.PathLike,
    audio_path: str | os.PathLike,
    out_mp4_path: str | os.PathLike,
    *,
    progress_cb: Callable[[int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    tail_pad_sec: float | None = None,
) -> None:
    """Mux a static image with an audio file into an MP4 via ffmpeg.

    Args:
        image_path: JPEG (or any ffmpeg-readable image) used as the
            looped video frame. Typically the Picsum-fetched thumbnail
            or its Pillow fallback (both 1280x720 per R-02).
        audio_path: Audio file (WAV, MP3, FLAC — anything ffmpeg can
            decode) used as the audio stream. The encoded MP4's
            duration equals this audio's duration (``-shortest``).
        out_mp4_path: Destination MP4 path. Written atomically via a
            ``.tmp`` sibling + ``os.replace``. The extension MUST be
            ``.mp4``; ffmpeg's container dispatch is extension-driven.
        progress_cb: Optional zero-or-100 percent bookend callback. v1
            emits ``0`` before launching ffmpeg and ``100`` on success
            (RESEARCH §Pitfall 6 notes the encode is bounded; stderr
            parsing for fractional progress is a Plan 08-06 follow-up
            if user feedback demands it).
        cancel_check: Optional zero-arg callable returning ``True`` to
            signal cancellation. Polled exactly ONCE before launching
            ffmpeg — once the subprocess starts, ffmpeg runs to
            completion (or its own timeout).

    Raises:
        BuildCancelled: ``cancel_check()`` returned ``True`` before
            launch. The ``.tmp`` sibling is removed if it existed.
        RuntimeError: ffmpeg exited non-zero OR exceeded the
            :data:`_FFMPEG_TIMEOUT_SEC` budget. The error message
            includes ``"ffmpeg failed"`` (non-zero exit) or
            ``"ffmpeg timed out"`` (timeout), followed by the last 500
            characters of stderr (when available). The ``.tmp`` sibling
            is removed on either failure path.
    """
    out_p = Path(out_mp4_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # Append ``.tmp`` BEFORE the suffix so ffmpeg's container dispatch
    # still sees ``.mp4`` (foo.mp4 → foo.tmp.mp4). os.replace at the end
    # renames to the canonical name. Mirrors the export_builder atomic
    # write pattern at lines 158-166.
    tmp_p = out_p.with_name(out_p.stem + ".tmp" + out_p.suffix)

    # Pre-launch cancel check — cheapest possible early-out.
    if cancel_check is not None and cancel_check():
        # No .tmp yet; nothing to clean up.
        raise BuildCancelled()

    if progress_cb is not None:
        progress_cb(0)

    # Probe the audio duration so we can pass an explicit ``-t`` clamp
    # to ffmpeg. ``-shortest`` alone is unreliable with ``-loop 1
    # -framerate 2`` + libx264 (the encoder flushes to the next GOP
    # boundary, which at 2 fps with the default keyint=250 can extend
    # the video well past the audio end — the symptom in HUMAN-UAT #2
    # was a 2:13 audio producing a 2:51 video). ``-t <duration>`` is a
    # hard clamp at the muxer level and gives a bit-exact output
    # duration matching the audio. We keep ``-shortest`` as
    # belt-and-braces.
    try:
        audio_duration_sec = sf.info(str(audio_path)).duration
    except Exception:
        # If we can't probe (e.g., a format soundfile doesn't recognise),
        # fall back to letting ffmpeg's ``-shortest`` do the cutting on
        # its own. The original "video may extend past audio" symptom
        # then re-emerges, but at least the encode still completes.
        audio_duration_sec = None

    ffmpeg = get_ffmpeg_exe()
    args: list[str] = [
        ffmpeg,
        "-y",                            # overwrite output (non-interactive)
        "-loop", "1",                    # loop the still image (D-13)
        "-framerate", "2",               # 2 fps for the looped image
        "-i", str(image_path),           # video input
        "-i", str(audio_path),           # audio input
    ]
    # Resolve the effective tail pad. ``None`` means "use the module
    # default" (per-keeper Share path — audio ends abruptly so the
    # 1.5 s breathing room is needed). Bundle path passes ``0`` because
    # the bundle MP3 already ends in a real per-keeper fade-out and an
    # extra second of silent video tail would just feel like dead air.
    effective_tail = _TAIL_PAD_SEC if tail_pad_sec is None else float(tail_pad_sec)
    if effective_tail < 0:
        effective_tail = 0.0

    if audio_duration_sec is not None and effective_tail > 0:
        # Pad the audio stream with ``effective_tail`` of silence so the
        # mastered fade-out plays out before the muxer stops. Without
        # ``apad`` the audio simply ends at the last sample of the WAV
        # and AAC framing can clip the very tail of the fade.
        args += ["-af", f"apad=pad_dur={effective_tail:.3f}"]
    args += [
        *_FFMPEG_VIDEO_FLAGS,            # H.264 still-image preset
        *_FFMPEG_AUDIO_FLAGS,            # AAC 192k
        *_FFMPEG_CONTAINER_FLAGS,        # yuv420p + -shortest
    ]
    if audio_duration_sec is not None:
        # Hard duration clamp at the muxer: audio_duration + tail_pad.
        # Must come before the output path. 3 decimal places keeps
        # sub-second precision without overrunning the parser.
        total_dur = audio_duration_sec + effective_tail
        args += ["-t", f"{total_dur:.3f}"]
    args.append(str(tmp_p))              # output path

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        # T-08-04-06 mitigation: kill the child process (subprocess.run
        # already kills it before raising) and clean up the .tmp sibling.
        _best_effort_unlink(tmp_p)
        raise RuntimeError(
            f"ffmpeg timed out after {_FFMPEG_TIMEOUT_SEC}s — "
            "video assembly aborted (file too large or system overloaded)."
        ) from None
    except BaseException:
        # Any other exception (KeyboardInterrupt, OSError on missing
        # ffmpeg binary, etc.) — clean up the .tmp sibling, re-raise.
        _best_effort_unlink(tmp_p)
        raise

    if proc.returncode != 0:
        _best_effort_unlink(tmp_p)
        try:
            stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-500:]
        except Exception:
            stderr_tail = "<could not decode stderr>"
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}): {stderr_tail}"
        )

    # Success — atomic rename.
    try:
        os.replace(str(tmp_p), str(out_p))
    except OSError:
        _best_effort_unlink(tmp_p)
        raise

    if progress_cb is not None:
        progress_cb(100)


def _best_effort_unlink(path: Path) -> None:
    """Remove ``path`` if it exists; silently no-op otherwise."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # Permission denied or similar — surface no exception. The
        # caller is already handling some other failure; double-faulting
        # on cleanup would mask the original error.
        pass
