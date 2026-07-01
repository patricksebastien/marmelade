"""sounddevice-driven playback engine — Qt-free Qt-bridge module.

The :class:`PlaybackEngine` streams audio from disk via
:class:`pedalboard.io.AudioFile` in ``BLOCKSIZE``-frame (2048-sample) chunks
through a bounded :class:`queue.Queue` (``BUFFERSIZE=20``) into a
:class:`sounddevice.OutputStream`. The producer thread reads + mixes-down
to mono float32 + enqueues; the PortAudio callback consumes the queue +
advances a ``_frames_played`` counter under a :class:`threading.Lock`.

CLAUDE.md memory contract: the full file is NEVER loaded into RAM. An 8-hour
mono float32 file at 44.1 kHz would be ≈ 5.1 GiB; the bounded queue caps
in-flight numpy data at ``BUFFERSIZE * BLOCKSIZE * channels * 4 B`` ≈ 160 KiB.
This is the same block-streaming discipline as ``audio_file.iter_blocks`` /
``peak_builder.build_proxy`` — the engine just routes blocks to sounddevice
instead of accumulating min/max peaks.

Qt-free policy (Plan 02-05):
    This module has NO ``import PySide6`` or ``import PyQt6``. The PortAudio
    callback writes only to atomic primitives (one int + threading.Lock +
    threading.Event); it NEVER touches a Qt widget (RESEARCH §Pitfall #2 —
    thread-affinity violation crashes randomly). The Qt-bridge (QTimer poll
    of ``position_seconds`` + signals + QShortcut wiring) lives entirely in
    :class:`marmelade.ui.main_window.MainWindow`. This makes the engine
    headlessly unit-testable AND keeps the audio-thread/GUI-thread boundary
    crisp.

Graceful degradation on missing libportaudio2 (Linux):
    The top-level ``import sounddevice`` is wrapped in try/except so dlopen()
    failure (e.g. ``libportaudio2`` not installed) does NOT crash the app.
    ``_SOUNDDEVICE_AVAILABLE`` is captured into ``self._sd_available`` at
    construction time so MainWindow can disable the playback toolbar action
    + spacebar shortcut while keeping the waveform + heatmap pipeline
    functional. ``engine.play(...)`` raises :class:`PlaybackError` when the
    backend is unavailable so callers can surface a user-visible message.

Common pitfalls handled here:
    * Pitfall #2 — callback never touches Qt; only writes to primitives.
    * Pitfall #3 — ``output_underflow`` fills silence and returns; we do NOT
      raise ``sd.CallbackAbort`` (which would terminate the stream). For a
      long-file analysis tool a brief dropout > stream death under heatmap
      compute spike.
    * Pitfall #4 — libportaudio2 missing on Linux degrades gracefully (top-
      level try/except + per-instance flag).
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# Graceful degradation on missing libportaudio2 (Linux) — Pitfall #4. The
# dlopen failure inside the sounddevice package surfaces as OSError; on a
# system without sounddevice installed at all we'd see ImportError. Catch
# both and flip the module-level flag so MainWindow can degrade gracefully.
try:
    import sounddevice as sd

    _SOUNDDEVICE_AVAILABLE = True
    _SOUNDDEVICE_IMPORT_ERROR: Optional[BaseException] = None
except (OSError, ImportError) as e:  # pragma: no cover — backend-availability gate
    sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_AVAILABLE = False
    _SOUNDDEVICE_IMPORT_ERROR = e

from pedalboard.io import AudioFile

# quick-260622-vwr — REUSE the single-home dB→linear + eps-clamp scaling math
# for the keeper-segment normalize pre-pass. The engine (not normalize.py)
# owns AudioFile + _mix_to_mono, so the streaming pre-pass lives here while the
# affine math stays in normalize.py (N-3 invariant: no toolkit/file imports in
# normalize.py).
from marmelade.audio.normalize import _compute_scale


# ============================================================================
# Module-level constants — verified by RESEARCH §Pattern 6 (play_long_file.py
# canonical defaults; cross-platform safe).
# ============================================================================

# ~46 ms @ 44.1 kHz. Smaller blocks reduce playback latency but increase
# underflow risk under CPU contention. play_long_file.py default; tested
# across Linux/macOS/Windows.
BLOCKSIZE = 2048

# Number of queue slots — ~928 ms playback buffer at 2048 frames each.
# Larger buffer = safer under CPU spikes (heatmap compute) at the cost of
# slower seek response (each seek drains + refills this many blocks).
BUFFERSIZE = 20


class PlaybackError(RuntimeError):
    """Raised when the audio backend is unavailable or playback fails to start.

    The caller (MainWindow) catches this in ``_action_toggle_playback`` and
    surfaces a QMessageBox warning. Construction-time backend availability
    is checked via ``engine.is_available`` so the UI can disable the toolbar
    action proactively rather than waiting for the exception.
    """


class PlaybackEngine:
    """sounddevice OutputStream + bounded queue producer + atomic position counter.

    Lifecycle:
        1. Construct (cheap; captures backend-availability flag).
        2. ``engine.play(path, start_seconds=0.0)`` — opens AudioFile, seeks
           to start, prebuffers BUFFERSIZE blocks, constructs OutputStream,
           starts a producer thread, calls stream.start().
        3. The producer thread reads BLOCKSIZE-frame chunks + mixes-down to
           mono float32 + enqueues until EOF or cancel.
        4. The PortAudio callback consumes the queue + advances
           ``_frames_played`` under ``_lock``.
        5. ``engine.pause()`` — sets the stop event + stream.stop().
        6. ``engine.seek(target_seconds)`` — stop() + new play() if was
           playing; otherwise just update position state.
        7. ``engine.stop()`` — full cleanup (stop + close + drain queue).

    Qt-bridge contract:
        ``position_seconds`` is the GUI thread's poll point. MainWindow's
        30 Hz QTimer calls it from the GUI thread; the lock serialises with
        the callback's update from the audio thread. No widget reference
        crosses the boundary.
    """

    def __init__(self) -> None:
        # Bounded queue — drops underflow margin to BUFFERSIZE * blocksize / sr.
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=BUFFERSIZE)
        self._stream: Optional["sd.OutputStream"] = None
        self._producer: Optional[threading.Thread] = None
        # Producer-thread cancel signal. Cleared on play(), set on pause/stop/seek.
        self._stop_event: threading.Event = threading.Event()
        # Set by sounddevice's finished_callback. Used by callers that want to
        # wait for clean stream shutdown.
        self._finished_event: threading.Event = threading.Event()
        # Serialises every read AND write of _frames_played / _is_playing /
        # _start_frame. Held only briefly (small int update) so contention
        # with the GUI thread poll is negligible.
        self._lock: threading.Lock = threading.Lock()
        # Position counter — frames played by the callback since the current
        # play() call. Total playback position = _start_frame + _frames_played.
        self._frames_played: int = 0
        # Starting frame of the current play() segment. After seek(), reset
        # to the new frame; after fresh play(), set to start_seconds * sr.
        self._start_frame: int = 0
        # Source sample rate, captured from AudioFile.samplerate.
        self._sample_rate: int = 0
        # Source total frames, captured from AudioFile.frames.
        self._duration_frames: int = 0
        # Engine state — flips True in play(), False on stop/pause/EOF.
        self._is_playing: bool = False
        # The currently-loaded file path; cleared on stop. seek() uses this
        # to restart playback after stopping the old stream.
        self._current_path: Optional[Path] = None
        # Per-instance backend-availability flag — captured at construction
        # time so monkeypatch-based tests work deterministically (W2).
        self._sd_available: bool = _SOUNDDEVICE_AVAILABLE
        # Plan 03-03 — Trash playback skip (D-A2-3).
        # Sorted, non-overlapping list of (start_sec, end_sec) tuples.
        # Keeper-punch-through subtraction is done by the GUI tier
        # (regions_overlay.trash_minus_keepers) before pushing here, but
        # set_skip_ranges defensively re-sorts + filters invalid ranges.
        # Read in _callback under _lock so a GUI-thread update doesn't
        # tear the audio-thread read.
        self._skip_ranges: list[tuple[float, float]] = []
        # 03-06b — explicit EOF signal so the callback can distinguish
        # "queue empty because producer hit EOF" (→ CallbackAbort, real
        # end-of-stream) from "queue empty because we just drained it
        # for a Trash skip and the producer is mid-seek" (→ silence
        # until refill). Set by _produce when it breaks out of its loop
        # at EOF, cleared by play().
        self._producer_eof: threading.Event = threading.Event()
        # 03-06c — GUI-driven Trash skip. The audio-thread producer
        # coordination (above two fields) was fundamentally racy in the
        # real audio thread + ALSA stack. Replaced with: callback
        # silences + sets _pending_skip_to_sec; GUI thread polls
        # consume_pending_skip() and calls seek(value), which restarts
        # the stream cleanly at trash_end + 50 ms — same code path as
        # click-to-seek (well tested). Cleared by play() and by
        # consume_pending_skip().
        self._pending_skip_to_sec: float | None = None
        # quick-260621-iuc — keeper-segment playback window + fade envelope.
        # These are set in play() from the new end_seconds/fade_seconds
        # kwargs and consumed ONLY by the producer-side _read_segment_block
        # helper (NEVER the real-time _callback). Defaults make a fresh /
        # never-played engine and the backward-compat play(path) path a
        # no-op (no truncation, no gain):
        #   _segment_start_frame / _segment_end_frame bound the readable
        #     source window in ABSOLUTE source frames; end defaults to the
        #     full file (set to _duration_frames in play()).
        #   _segment_fade_in_frames / _segment_fade_out_frames are the
        #     INDEPENDENT fade-in (segment START) and fade-out (segment END)
        #     ramp lengths in source frames. quick-260622-ud0 split the old
        #     single symmetric _segment_fade_frames into these two so the
        #     keeper auditions can drop ONLY the fade-in (middle/end modes)
        #     while keeping the fade-out. 0 in BOTH means "apply no fade"
        #     (backward-compat: byte-identical passthrough). A symmetric
        #     fade_seconds=X caller resolves both to the same capped value, so
        #     it stays byte-identical to the pre-split behavior.
        #   _segment_read_cursor is the producer-owned absolute source frame
        #     index of the NEXT frame to be read; both the synchronous
        #     prebuffer loop and the _produce thread advance it (they never
        #     run concurrently — prebuffer completes before the producer
        #     thread is spawned — so no lock is needed).
        self._segment_start_frame: int = 0
        self._segment_end_frame: int = 0
        self._segment_fade_in_frames: int = 0
        self._segment_fade_out_frames: int = 0
        self._segment_read_cursor: int = 0
        # quick-260622-vwr — optional per-segment WYSIWYG normalize affine
        # applied to each mono block as ``(mono - dc) * scale`` BEFORE the
        # fade. Defaults (0.0 / 1.0) are a byte-identical no-op so every
        # existing caller is unchanged. A-mode keeper preview sets these so
        # the audition matches the normalized waveform display; B-mode leaves
        # them at the defaults (the mastered cache already baked normalize in).
        self._segment_norm_dc: float = 0.0
        self._segment_norm_scale: float = 1.0
        # quick — A/V sync. PortAudio's output latency: the gap between the
        # callback handing a block to the output buffer (where we advance
        # _frames_played) and that block actually reaching the DAC / speaker.
        # Captured from the live stream at play() time. position_seconds
        # subtracts it so the visual playhead tracks the AUDIBLE sample, not
        # the frame merely queued for output (otherwise the playhead runs
        # ``_output_latency_sec`` AHEAD of the sound — the desync where a
        # sharp transient is heard before the playhead reaches it). 0.0 when
        # no stream is open.
        self._output_latency_sec: float = 0.0

    # ----------------------------------------------------- file-open priming
    def prime(self, path: str) -> None:
        """Open ``path`` briefly to capture ``sample_rate`` and ``duration_frames``.

        Bug #1 fix — after D-15 made playback lazy (file open no longer
        triggers play()), ``_sample_rate`` stayed 0 until the first play()
        call. A user who clicked to seek BEFORE pressing spacebar saw their
        seek silently zeroed because ``seek()`` falls back to
        ``_start_frame = 0`` whenever ``sample_rate`` is 0. ``prime()`` runs
        at file-open time to populate ``_sample_rate`` + ``_duration_frames``
        so a pre-play seek lands on the right frame.

        Idempotent: priming the same file twice is a no-op-equivalent
        (overwrites with the same values). Silent on failure: any exception
        opening the file is swallowed so the file-open UI flow never breaks
        because of a priming failure. Called by
        :meth:`marmelade.ui.main_window.MainWindow._open_file` after the
        playback-stop block.
        """
        if not self._sd_available:
            return
        try:
            af = AudioFile(str(path), "r")
        except Exception:
            # Any open failure (corrupt file, unsupported format, missing
            # file). Stay silent — don't mutate engine state.
            return
        try:
            new_sr = int(af.samplerate)
            new_frames = int(af.frames)
        finally:
            try:
                af.close()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
        with self._lock:
            self._sample_rate = new_sr
            self._duration_frames = new_frames
            self._current_path = Path(path)
            self._start_frame = 0
            self._frames_played = 0

    # ------------------------------------------------------------- properties
    @property
    def position_seconds(self) -> float:
        """Current playback position in seconds (start_frame + frames_played) / sr.

        Read under ``_lock`` so a concurrent callback update doesn't tear
        the composite read. Returns 0.0 before first play() (sample_rate=0).

        A/V sync: ``_frames_played`` counts frames HANDED to PortAudio's
        output buffer, which become audible ``_output_latency_sec`` later.
        We subtract that latency so the reported position matches what the
        user is hearing NOW (not the frame already queued for output). The
        audible-frame offset is clamped at 0 so the playhead sits at the
        segment start until the first sample actually reaches the speaker.
        """
        with self._lock:
            if self._sample_rate <= 0:
                return 0.0
            latency_frames = int(self._output_latency_sec * self._sample_rate)
            audible_frames = self._frames_played - latency_frames
            if audible_frames < 0:
                audible_frames = 0
            return (self._start_frame + audible_frames) / float(self._sample_rate)

    @property
    def duration_seconds(self) -> float:
        """Source duration in seconds, populated by prime() / play().

        Returns 0.0 before the engine has been primed (sample_rate=0).
        MainWindow uses this to clamp click-to-seek targets to
        ``duration_seconds - epsilon`` before calling :meth:`play` —
        without that clamp, a click near the right edge of the waveform
        on a source whose engine target is shorter (e.g. proxy < source
        due to a build truncation, or a render artifact) raises
        ``ValueError: Cannot seek to position N frames, which is beyond
        end of file``. Phase 2.1 HUMAN-UAT bug #2.
        """
        with self._lock:
            if self._sample_rate <= 0:
                return 0.0
            return self._duration_frames / float(self._sample_rate)

    @property
    def is_playing(self) -> bool:
        """True between play() and pause/stop/EOF. Read under lock."""
        with self._lock:
            return self._is_playing

    @property
    def is_available(self) -> bool:
        """True iff sounddevice imported cleanly at module load.

        MainWindow consults this to decide whether to enable the playback
        toolbar action + spacebar shortcut. Captured at engine construction
        time from the module-level flag so monkeypatch-based tests work
        (W2 — read the instance attribute, not the module-level flag).
        """
        return self._sd_available

    @property
    def sample_rate(self) -> int:
        """Current source sample rate in Hz, or 0 before prime()/play().

        WR-04 — public accessor so MainWindow (and other consumers) do not
        have to reach into ``engine._sample_rate``. Read under ``_lock``
        so a concurrent ``play()`` or ``prime()`` update does not produce
        a torn read between the position check and the multiplication
        against region bounds in the export path.
        """
        with self._lock:
            return self._sample_rate

    # ---------------------------------------------- Plan 03-03 — Trash skip
    def set_skip_ranges(self, ranges: list[tuple[float, float]]) -> None:
        """Set Trash playback-skip ranges (Plan 03-03 / D-A2-3).

        Each tuple is ``(start_sec, end_sec)`` — half-open
        ``[start, end)``. The GUI tier is expected to pass sorted,
        non-overlapping ranges via
        :meth:`marmelade.ui.regions_overlay.RegionsOverlay.trash_minus_keepers`
        (which already performs Keeper-punch-through subtraction), but
        we defensively re-sort + filter ``end <= start`` so the engine
        does not trust untrusted input (T-03-03-04).

        Hard-jump v1 — when the playhead enters a skip range during
        playback, the audio-thread callback advances ``_frames_played``
        to put the position at ``end_sec``, fills the current callback
        with silence, and signals the producer thread to re-seek the
        AudioFile to the new frame. No crossfade in v1 (escalation
        path documented in 03-CONTEXT D-A2-3).

        Thread-safety: ranges are stored under ``self._lock`` so the
        audio-thread read in :meth:`_callback` never tears.
        """
        cleaned = sorted(
            (float(a), float(b)) for a, b in ranges if float(b) > float(a)
        )
        with self._lock:
            self._skip_ranges = cleaned

    def consume_pending_skip(self) -> float | None:
        """GUI-thread poll point for the Trash-skip → seek redirect (03-06c).

        Returns the target seconds the engine wants to seek to and
        atomically clears the flag, OR None if no skip is pending. The
        GUI's playhead-poll QTimer calls this on each tick (30 Hz) and,
        if non-None, calls :meth:`seek` to restart the stream past the
        Trash range. Restart goes through the same well-tested code
        path as user click-to-seek — no audio-thread + producer-thread
        coordination dance.
        """
        with self._lock:
            v = self._pending_skip_to_sec
            self._pending_skip_to_sec = None
            return v

    # ------------------------------------------------------------- lifecycle
    def play(
        self,
        path: str,
        start_seconds: float = 0.0,
        end_seconds: Optional[float] = None,
        fade_seconds: float = 0.0,
        fade_in_seconds: Optional[float] = None,
        fade_out_seconds: Optional[float] = None,
        normalize_dc: float = 0.0,
        normalize_scale: float = 1.0,
    ) -> None:
        """Open ``path``, seek to ``start_seconds``, prebuffer, and start the stream.

        If the engine is already playing, stop() the old stream first so
        play(seek_pos) acts as a clean seek-and-resume. Raises PlaybackError
        if the audio backend is unavailable (libportaudio2 missing).

        Args:
            path: Absolute path to a pedalboard-readable audio file.
            start_seconds: Position to start playback from (default 0.0).
            end_seconds: Optional segment end in source seconds (default None).
                When set, reads STOP at ``int(end_seconds * sr)`` — the
                producer truncates the final block and signals
                ``_producer_eof`` so the existing CallbackStop/CallbackAbort
                path flips ``_is_playing`` False at the segment end (no new
                GUI stop logic needed). ``None`` plays to the file's natural
                EOF (backward-compat full-file playback).
            fade_seconds: Symmetric fallback linear fade duration in seconds.
                When ``fade_in_seconds`` / ``fade_out_seconds`` are left as
                ``None`` (the default), BOTH the segment START (fade-in 0→1)
                and END (fade-out 1→0) use this single value — byte-identical
                to the pre-split behavior every existing caller relies on.
                ``0.0`` (the default) applies NO gain.
            fade_in_seconds: Optional independent fade-IN duration (segment
                START, 0→1). ``None`` falls back to ``fade_seconds``; pass
                ``0.0`` to suppress the fade-in while keeping a fade-out
                (quick-260622-ud0 keeper middle/end auditions).
            fade_out_seconds: Optional independent fade-OUT duration (segment
                END, 1→0). ``None`` falls back to ``fade_seconds``.

            Per-end resolution rule: ``eff_in = fade_in_seconds if not None
            else fade_seconds`` and ``eff_out = fade_out_seconds if not None
            else fade_seconds``. Each ramp matches ``export_builder``'s
            ``endpoint=True`` shape and is independently capped at
            ``segment_len // 2`` so the two windows never overlap.

            normalize_dc / normalize_scale: optional per-segment WYSIWYG
                normalize affine applied to every mono block as
                ``(mono - normalize_dc) * normalize_scale`` — the SAME
                DC-remove + peak-to-target the normalized waveform render
                applies. Defaults (``0.0`` / ``1.0``) are a byte-identical
                no-op so every existing caller is unchanged. The affine is
                applied BEFORE the fade so the fade envelope still brings the
                segment ends to exactly 0.0 (it multiplies the normalized
                signal). A-mode keeper preview (quick-260622-vwr) sets these
                via :meth:`compute_segment_normalize_params`; B-mode leaves
                them at the defaults because the mastered cache already has
                normalize baked in as the chain's final stage.

        quick-260621-iuc — keeper-segment audition:
            B-mode (mastered cache) MUST pass ``fade_seconds`` (not 0.0): the
            mastering chain bakes NO fades (fades live only in
            ``export_builder``), so the cache is UN-faded and the fade is
            applied HERE. There is NO double-fade risk.

            The fade-in lands in the FIRST blocks, which are read by the
            synchronous prebuffer loop below — that loop uses the SAME
            ``_read_segment_block`` helper as the producer, so the fade-in is
            never silently lost.

        Limitation: ``seek()`` restarts via ``play(path, start)`` only — it
        passes no ``end_seconds``/``fade_seconds``. A user seek inside or past
        a keeper therefore falls back to full-file behavior (no segment stop,
        no fades). Acceptable for v1 (seek is a deliberate "leave the keeper"
        gesture).
        """
        if not self._sd_available:
            raise PlaybackError(
                f"audio backend unavailable: {_SOUNDDEVICE_IMPORT_ERROR}"
            )

        # Stop any previous playback cleanly. play() on a fresh engine is a
        # no-op cleanup; on an already-playing engine this is the seek-and-
        # restart fast path.
        if self._is_playing or self._stream is not None:
            self.stop()

        # Open the file. NB: we hold the AudioFile open across the producer's
        # lifetime; it's closed in the producer's finally clause OR in stop().
        af = AudioFile(str(path), "r")
        self._current_path = Path(path)
        self._sample_rate = int(af.samplerate)
        self._duration_frames = int(af.frames)

        start_frame = int(max(0.0, start_seconds) * self._sample_rate)
        af.seek(start_frame)

        # quick-260621-iuc — compute the segment window + fade envelope in
        # ABSOLUTE source frames. end_seconds=None => play to natural EOF
        # (backward-compat). Clamp the end into [start+1, duration] so a
        # bogus/short end can never produce a zero- or negative-length
        # segment.
        sr = self._sample_rate
        if end_seconds is not None:
            segment_end_frame = int(end_seconds * sr)
        else:
            segment_end_frame = self._duration_frames
        segment_end_frame = max(
            start_frame + 1, min(segment_end_frame, self._duration_frames)
        )
        segment_len = segment_end_frame - start_frame
        # quick-260622-ud0 — resolve independent in/out fades. None falls back
        # to the symmetric fade_seconds so every existing fade_seconds-only
        # caller stays byte-identical (both ends equal). Each window is capped
        # independently at segment_len // 2: even when BOTH are maxed they meet
        # but never overlap. 0 in BOTH => no gain (backward-compat).
        eff_in = fade_in_seconds if fade_in_seconds is not None else fade_seconds
        eff_out = fade_out_seconds if fade_out_seconds is not None else fade_seconds
        fade_in_frames = int(max(0.0, eff_in) * sr)
        fade_out_frames = int(max(0.0, eff_out) * sr)
        fade_in_frames = min(fade_in_frames, segment_len // 2)
        fade_out_frames = min(fade_out_frames, segment_len // 2)
        self._segment_start_frame = start_frame
        self._segment_end_frame = segment_end_frame
        self._segment_fade_in_frames = fade_in_frames
        self._segment_fade_out_frames = fade_out_frames
        # quick-260622-vwr — store the WYSIWYG normalize affine. Defaults
        # (0.0 / 1.0) keep the _read_segment_block path byte-identical.
        self._segment_norm_dc = float(normalize_dc)
        self._segment_norm_scale = float(normalize_scale)
        # Producer-owned absolute source frame cursor. Initialised BEFORE the
        # prebuffer loop so the first block's fade-in math is correct. Advanced
        # by the PRE-truncation frames read (so the fade envelope stays aligned
        # to true source frames even when the final block is truncated).
        self._segment_read_cursor = start_frame

        with self._lock:
            self._start_frame = start_frame
            self._frames_played = 0
            self._is_playing = True

        # Reset thread-signal primitives for the new playback.
        self._stop_event.clear()
        self._finished_event.clear()
        self._producer_eof.clear()  # 03-06b

        # Drain any stale queue items left over from a prior playback.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Synchronous prebuffer — fill the queue before starting the stream so
        # the callback's first call doesn't underflow immediately. The producer
        # thread will continue refilling from the same AudioFile.
        #
        # quick-260621-iuc — read via _read_segment_block so the segment-end
        # truncation AND the fade-in (which lands in these FIRST blocks) are
        # applied in the prebuffer path, not only the producer thread. A
        # zero-length return means EOF (natural file EOF or segment-end
        # truncation collapsed the block) — break, same as before.
        for _ in range(BUFFERSIZE):
            mono = self._read_segment_block(af)
            if mono.shape[0] == 0:
                break  # EOF before we could fill the buffer
            try:
                self._queue.put_nowait(mono)
            except queue.Full:
                break

        # Construct the stream. The callback runs on PortAudio's real-time
        # audio thread — Pitfall #2 mandates no Qt widget access here. We
        # also wire the finished_callback to _finished_event.set so callers
        # can wait on clean stream shutdown.
        # quick — ``latency="low"`` (was "high") shrinks PortAudio's output
        # buffer so the audible sound stays tight to the visual playhead. The
        # producer queue (BUFFERSIZE blocks) still absorbs disk-read / CPU
        # stalls independently of the device latency, and underflow degrades
        # gracefully (Pitfall #3 — fill silence, never abort), so the lower
        # latency does not risk stream death.
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            blocksize=BLOCKSIZE,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self._callback,
            finished_callback=self._finished_event.set,
        )
        # Capture the device output latency PortAudio actually granted so
        # position_seconds can subtract it (A/V sync — see the property).
        # Only trust a real number: a real OutputStream reports a float here,
        # while a mocked stream (tests) reports a non-numeric stand-in we must
        # ignore (treat as no latency) rather than coerce.
        lat = getattr(self._stream, "latency", 0.0)
        self._output_latency_sec = (
            float(lat) if isinstance(lat, (int, float)) else 0.0
        )
        self._stream.start()

        # Spawn the producer. daemon=True so the OS terminates it on app
        # exit (no Hang On Quit). The producer is the only consumer of the
        # AudioFile after this point.
        self._producer = threading.Thread(
            target=self._produce,
            args=(af,),
            name="Marmelade-PlaybackProducer",
            daemon=True,
        )
        self._producer.start()

    def pause(self) -> None:
        """Stop the stream cleanly and clear is_playing.

        pause() vs stop(): pause keeps the engine state ready for a quick
        resume via play(start=position_seconds); stop() is the full cleanup
        (closes the stream + clears the current path).

        BL-01 fix — pause() now mirrors stop() unconditionally: the
        ``_stop_event`` is ALWAYS set, the stream is fully closed (not just
        stopped), and the queue is drained. The only difference from stop()
        is that ``_current_path`` is preserved so a subsequent
        ``play(self._current_path, start=position_seconds)`` resume works.
        """
        # Always set the cancel signal — mirrors stop() so the producer
        # thread can see it regardless of stream state. Without this, a
        # pause() called before any play() (defensive UI wiring, stale
        # spacebar press, manual test scenario) would leave _stop_event
        # unset and _is_playing stale.
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
            self._stream = None
        # Drain the queue so a subsequent play() starts clean (mirrors
        # stop() without clearing _current_path — pause keeps _current_path
        # so resume can use it).
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._is_playing = False
            # No live stream → no output latency to compensate. A paused
            # engine reports its exact last position so resume picks it up.
            self._output_latency_sec = 0.0

    def stop(self) -> None:
        """Full cleanup: stop event + stream.stop() + close() + drain queue."""
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
            self._stream = None
        # Drain any leftover queue items so the next play() starts clean.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._is_playing = False
            # No live stream → no output latency to compensate.
            self._output_latency_sec = 0.0
            # Plan 03-06c — clear pending Trash skip-to-seek so stale
            # values cannot carry across a stop->play cycle. Running on
            # the GUI thread (not the audio thread), so no lock-nesting
            # risk.
            self._pending_skip_to_sec = None

    def seek(self, target_seconds: float) -> None:
        """Seek to ``target_seconds``. Resumes playback iff it was already playing.

        Implementation: stop() the current stream + drain the queue, then
        if we were playing, restart via play() at the new position. If we
        were paused, just update _start_frame so position_seconds reflects
        the new cursor (next play() call will pick it up).
        """
        was_playing = self.is_playing
        path = self._current_path
        sample_rate = self._sample_rate

        # Stop the old stream + drain the queue.
        self.stop()

        # Reset position state for the new segment.
        with self._lock:
            self._start_frame = (
                int(max(0.0, target_seconds) * sample_rate) if sample_rate else 0
            )
            self._frames_played = 0
            self._sample_rate = sample_rate  # stop() didn't touch this; preserve

        if was_playing and path is not None:
            self.play(str(path), start_seconds=float(target_seconds))

    # ------------------------------------------------ normalize pre-pass
    def compute_segment_normalize_params(
        self,
        path: str,
        start_seconds: float,
        end_seconds: float,
        target_db: float,
    ) -> tuple[float, float]:
        """Stream a keeper segment and return its WYSIWYG ``(dc, scale)`` affine.

        quick-260622-vwr — A PURE pre-pass READ: opens ``path``, streams the
        ``[start, end)`` keeper window block-by-block (one ``BLOCKSIZE`` block
        at a time — CLAUDE.md no-full-file-load contract), and derives the
        SAME DC-remove + peak-to-target affine the normalized waveform render
        applies. Never starts the stream / touches sounddevice, so it is safe
        to call in CI without libportaudio2.

        Mixes to mono via :meth:`_mix_to_mono` (the SAME domain :meth:`play`
        normalizes in) and reuses :func:`marmelade.audio.normalize._compute_scale`
        for the dB→linear + eps-clamp math (single home; N-3 keeps that
        function toolkit/file-import-free).

        Args:
            path: Absolute path to a pedalboard-readable audio file (the proxy
                A-mode plays — so the affine domain matches the audition).
            start_seconds: Keeper segment start in source seconds.
            end_seconds: Keeper segment end in source seconds.
            target_db: Desired peak amplitude in dBFS (``<= 0``).

        Returns:
            ``(dc, scale)`` where ``dc`` is the segment mean (the DC offset)
            and ``scale`` maps the centered peak onto ``target_db``. A silent /
            pure-DC segment returns ``(mean, 1.0)`` (no noise-floor blow-up);
            a fully-zero or zero-length / degenerate segment returns
            ``(0.0, 1.0)``.
        """
        af = AudioFile(str(path), "r")
        try:
            sr = int(af.samplerate)
            start_frame = int(max(0.0, start_seconds) * sr)
            end_frame = int(end_seconds * sr)
            end_frame = min(end_frame, int(af.frames))
            if end_frame <= start_frame:
                return (0.0, 1.0)
            af.seek(start_frame)

            total = 0.0  # sum of mono samples (float64)
            count = 0
            gmin = float("inf")
            gmax = float("-inf")
            cursor = start_frame
            while cursor < end_frame:
                data = af.read(BLOCKSIZE)
                frames_read = int(data.shape[1])
                if frames_read == 0:
                    break  # natural EOF before end_frame
                # End-truncate the final block so we never read past end_frame.
                if cursor + frames_read > end_frame:
                    data = data[:, : end_frame - cursor]
                mono = self._mix_to_mono(data)[:, 0].astype(np.float64, copy=False)
                if mono.size:
                    total += float(mono.sum())
                    count += int(mono.size)
                    gmin = min(gmin, float(mono.min()))
                    gmax = max(gmax, float(mono.max()))
                cursor += frames_read
        finally:
            af.close()

        if count == 0:
            return (0.0, 1.0)
        mean = total / count
        # Centered peak: mean ∈ [gmin, gmax], so the post-DC peak is the
        # larger absolute distance from the mean to either extreme.
        post_dc_peak = max(gmax - mean, mean - gmin)
        scale = _compute_scale(post_dc_peak, target_db)
        return (float(mean), float(scale))

    # ------------------------------------------------------------ internals
    def _mix_to_mono(self, chunk: np.ndarray) -> np.ndarray:
        """Convert a (channels, n) float32 chunk to a (n, 1) mono float32 column.

        Multi-channel sources are averaged across channels. The output shape
        matches sounddevice's per-block expectation: ``(frames, channels)``
        with channels=1 since we constructed OutputStream(channels=1).
        """
        if chunk.shape[0] > 1:
            mono = chunk.mean(axis=0)
        else:
            mono = chunk[0]
        return mono.astype(np.float32, copy=False).reshape(-1, 1)

    def _read_segment_block(self, af) -> np.ndarray:
        """Read one block, apply segment-end truncation + fade gain, return mono.

        quick-260621-iuc — the SHARED block reader used by BOTH the synchronous
        prebuffer loop in :meth:`play` and the :meth:`_produce` producer thread.
        Centralising the truncation + fade math here guarantees the fade-in
        (which lands in the first prebuffered blocks) is never lost and that
        the producer and prebuffer paths can never drift.

        Returns a ``(n, 1)`` mono float32 column, or an empty ``(0, 1)`` array
        to signal EOF (natural file EOF OR the segment-end truncation collapsed
        this block to zero frames). Callers treat the empty return exactly like
        a real ``af.read`` EOF.

        Steps:
            1. ``af.read(BLOCKSIZE)`` — raw ``(channels, n)`` source frames.
            2. Locate this block's ABSOLUTE source frame span from
               ``_segment_read_cursor``.
            3. END-TRUNCATE: keep only frames strictly before
               ``_segment_end_frame``; if that leaves 0 frames, return empty.
            4. Mix to mono via :meth:`_mix_to_mono` (UNCHANGED body).
            5. Apply the per-frame linear fade envelope (skipped entirely when
               BOTH ``_segment_fade_in_frames`` and
               ``_segment_fade_out_frames`` are 0 — the backward-compat path).
            6. Advance ``_segment_read_cursor`` by the PRE-truncation frames
               read (keeps the fade envelope aligned to true source frames even
               on the truncated final block — an off-by-one here would drift
               the boundary fade).

        CLAUDE.md memory contract preserved: one BLOCKSIZE block at a time; no
        full-file load.
        """
        data = af.read(BLOCKSIZE)
        frames_read = int(data.shape[1])
        if frames_read == 0:
            # Natural file EOF. Match the (channels, 0) -> (0, 1) mono shape.
            return np.zeros((0, 1), dtype=np.float32)

        block_start = self._segment_read_cursor
        # END-TRUNCATE to the segment window. The last kept frame is at
        # _segment_end_frame - 1 (half-open [start, end)).
        block_end = block_start + frames_read
        kept = frames_read
        if block_end > self._segment_end_frame:
            kept = self._segment_end_frame - block_start
            if kept <= 0:
                # Cursor already at/past the segment end — collapse to EOF.
                # Advance the cursor by the full pre-truncation read so a
                # follow-up call stays past the end (defensive; producer will
                # break on this empty return anyway).
                self._segment_read_cursor += frames_read
                return np.zeros((0, 1), dtype=np.float32)
            data = data[:, :kept]

        mono = self._mix_to_mono(data)

        # quick-260622-vwr — WYSIWYG normalize affine, applied BEFORE the fade.
        # `(mono - dc) * scale` is the SAME DC-remove + peak-to-target the
        # normalized waveform render applies. Guarded so the default path
        # (dc==0.0, scale==1.0) stays byte-identical. Order matters: normalize
        # FIRST, then fade — the fade envelope must still bring the segment
        # ends to exactly 0.0 by multiplying the already-normalized signal.
        if self._segment_norm_dc != 0.0 or self._segment_norm_scale != 1.0:
            mono = (
                (mono - self._segment_norm_dc) * self._segment_norm_scale
            ).astype(np.float32, copy=False)

        # Per-block linear fade envelope, mirroring export_builder's
        # endpoint=True linspace + global-index ramp math. The in (segment
        # START) and out (segment END) windows are independent — fire the
        # apply whenever EITHER has frames. Skipped when no fade requested
        # (backward-compat: byte-identical mono passthrough).
        if self._segment_fade_in_frames > 0 or self._segment_fade_out_frames > 0:
            self._apply_segment_fade(mono, block_start)

        # Advance by the PRE-truncation frames read (NOT `kept`).
        self._segment_read_cursor += frames_read
        return mono

    def _apply_segment_fade(self, mono: np.ndarray, block_start: int) -> None:
        """Apply the in/out linear fade to ``mono`` in place for its frame span.

        ``mono`` is ``(n, 1)`` and represents absolute source frames
        ``[block_start, block_start + n)`` AFTER end-truncation. The two fade
        windows are INDEPENDENT (quick-260622-ud0): the fade-in window is
        ``[seg_start, seg_start + fade_in)`` (length
        ``_segment_fade_in_frames``) and the fade-out window is
        ``[seg_end - fade_out, seg_end)`` (length ``_segment_fade_out_frames``).
        Each is guarded by its own ``> 0`` check so a zero-length window applies
        no gain — that is how the keeper middle/end auditions drop ONLY the
        fade-in. Ramp math mirrors ``export_builder._apply_fade_*`` (W-7
        invariant): ``endpoint=True`` linspace over the global index so the
        first fade-in sample is exactly 0.0, the last fade-out sample exactly
        0.0, and a fade spanning multiple blocks stitches together seamlessly.
        """
        fade_in = self._segment_fade_in_frames
        fade_out = self._segment_fade_out_frames
        seg_start = self._segment_start_frame
        seg_end = self._segment_end_frame
        n = mono.shape[0]
        block_end = block_start + n
        # Local frame indices are relative to block_start; global frame index
        # of local i is (block_start + i). Express windows relative to the
        # SEGMENT start/end (export_builder uses region-relative indices).
        rel_start = block_start - seg_start  # block's first frame, seg-relative

        # ----- fade-in window [seg_start, seg_start + fade_in) -----
        if fade_in > 0 and rel_start < fade_in:
            lo = 0
            hi = min(n, fade_in - rel_start)
            g_start = rel_start + lo
            g_end = rel_start + hi  # exclusive
            if fade_in > 1:
                ramp = np.linspace(
                    g_start / (fade_in - 1),
                    (g_end - 1) / (fade_in - 1),
                    hi - lo,
                    dtype=np.float32,
                    endpoint=True,
                )
            else:
                # Degenerate single-sample fade — boundary value is 0.0
                # (matches export_builder CR-02). Not reachable via the
                # GUI auto-scale, but keep the semantic correct.
                ramp = np.zeros(hi - lo, dtype=np.float32)
            mono[lo:hi, 0] *= ramp

        # ----- fade-out window [seg_end - fade_out, seg_end) -----
        if fade_out > 0:
            fade_out_start = seg_end - fade_out  # absolute source frame
            if block_end > fade_out_start:
                lo = max(0, fade_out_start - block_start)
                hi = n
                j_start = (block_start + lo) - fade_out_start
                j_end = (block_start + hi) - fade_out_start  # exclusive
                if fade_out > 1:
                    ramp = np.linspace(
                        1.0 - j_start / (fade_out - 1),
                        1.0 - (j_end - 1) / (fade_out - 1),
                        hi - lo,
                        dtype=np.float32,
                        endpoint=True,
                    )
                else:
                    ramp = np.zeros(hi - lo, dtype=np.float32)
                mono[lo:hi, 0] *= ramp

    def _produce(self, af) -> None:
        """Producer thread — read BLOCKSIZE-frame chunks from ``af`` into the queue.

        Polls ``_stop_event`` between reads; exits cleanly on stop or EOF.
        Plan 03-06c — Trash skips are now handled by the GUI thread via
        seek() (full stream restart), not by an in-producer re-seek.

        quick-260621-iuc — reads via :meth:`_read_segment_block` (shared with
        the prebuffer loop) so segment-end truncation + the fade envelope are
        applied identically. A zero-length return is EOF (natural file EOF OR
        the segment-end window) — set ``_producer_eof`` and break, exactly like
        the prior inline ``data.shape[1] == 0`` EOF branch.
        """
        try:
            while not self._stop_event.is_set():
                mono = self._read_segment_block(af)
                if mono.shape[0] == 0:
                    # 03-06b — signal EOF so the callback can distinguish
                    # "queue empty because we drained for a skip" from
                    # "queue empty because the file is done". quick-260621-iuc:
                    # the segment-end truncation collapsing to 0 frames is
                    # treated identically to natural EOF, so the existing
                    # CallbackStop/CallbackAbort path flips is_playing False
                    # at the keeper segment end.
                    self._producer_eof.set()
                    break  # EOF (natural or segment-end)
                try:
                    # Use a timeout so a stopped callback (queue never drains)
                    # doesn't block forever. Worst-case wait = one full buffer.
                    timeout = (BLOCKSIZE * BUFFERSIZE) / max(self._sample_rate, 1)
                    self._queue.put(mono, timeout=timeout)
                except queue.Full:
                    return  # callback died; stop producing
        finally:
            try:
                af.close()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass

    def _callback(self, outdata, frames, time, status) -> None:
        """PortAudio callback — runs on the audio thread.

        Pitfall #2: NEVER touch a Qt widget here. Pitfall #3: on underflow,
        fill silence and return (do NOT raise CallbackAbort which would
        terminate the stream).

        Plan 03-03 — Trash playback skip (D-A2-3) lives at the TOP of this
        function, BEFORE the underflow check and BEFORE the existing two
        ``with self._lock:`` blocks at the tail (the ``len(data) < len(outdata)``
        final-chunk path and the normal-frame ``_frames_played`` update).
        Inserting it at the top guarantees no nested-lock deadlock under
        stdlib non-reentrant :class:`threading.Lock` — the audio thread
        acquires the lock here, releases it before any later acquisition.
        """
        # Plan 03-06c — Trash playback skip via GUI-driven seek
        # (D-A2-3 / UAT Test 4). Read _skip_ranges + position state under
        # _lock so a GUI-thread update via set_skip_ranges doesn't tear
        # the audio-thread read. When the playhead is inside any skip
        # range, advance _frames_played past the Trash, set
        # _pending_skip_to_sec to (end_sec + 50 ms), fill silence, and
        # return. The GUI thread's playhead-poll QTimer calls
        # consume_pending_skip and then seek() — same code path as
        # click-to-seek, which restarts the stream cleanly. While the
        # skip is pending (between detection and the GUI's seek), the
        # callback silences so no pre-skip stale audio reaches outdata.
        with self._lock:
            # 03-06c — Trash skip via GUI-driven seek. If a skip is
            # already pending (waiting for GUI's poll tick to seek), just
            # silence — the next callback (or the seek-restart that
            # replaces this stream) will resume audio.
            if self._pending_skip_to_sec is not None:
                outdata.fill(0)
                return
            if self._skip_ranges and self._sample_rate > 0:
                pos_frames = self._start_frame + self._frames_played
                pos_sec = pos_frames / float(self._sample_rate)
                for start_sec, end_sec in self._skip_ranges:
                    if start_sec <= pos_sec < end_sec:
                        # 03-06c — advance position past the Trash and
                        # signal the GUI thread to seek (50 ms buffer
                        # past trash_end so the seek doesn't land back
                        # inside the half-open [start, end) range due
                        # to float rounding). Audio-thread silences
                        # until the GUI processes the seek (~33 ms
                        # average at 30 Hz playhead-poll). Reuses the
                        # same code path as click-to-seek — no fragile
                        # producer-thread coordination.
                        skip_to_sec = end_sec + 0.05
                        target_frame = int(skip_to_sec * self._sample_rate)
                        skip_frames = target_frame - pos_frames
                        if skip_frames > 0:
                            self._frames_played += skip_frames
                        self._pending_skip_to_sec = skip_to_sec
                        outdata.fill(0)
                        return

        # 03-06b — try to consume queue FIRST, then fall back to silence
        # on empty. The old order (underflow check → silence → return,
        # before the queue consume) silenced even when post-skip audio
        # was available: after a Trash drain the device reported
        # underflow (we just fed silence), and the next several callbacks
        # silenced even though the producer had refilled. Consuming first
        # lets fresh audio reach outdata as soon as the producer puts
        # the first post-seek block.
        try:
            data = self._queue.get_nowait()
        except queue.Empty:
            # 03-06b — only abort the stream on REAL EOF (producer set
            # _producer_eof). Otherwise the empty queue is transient:
            # either the producer is mid-seek after a Trash skip (we
            # just drained at detection) or it's catching up after a
            # generic stall. Silence + return so the next callback can
            # consume the producer's first post-skip put. Without this
            # gate, the drain-at-detection fix would race a CallbackAbort
            # before the producer's first put landed, killing the stream.
            outdata.fill(0)
            if self._producer_eof.is_set():
                raise sd.CallbackAbort
            return

        if len(data) < len(outdata):
            # Final-chunk-of-file: write what we have, zero-pad remainder,
            # flip is_playing False, raise CallbackStop so sounddevice drains
            # cleanly.
            outdata[:len(data)] = data
            outdata[len(data):].fill(0)
            with self._lock:
                self._frames_played += int(len(data))
                self._is_playing = False
            raise sd.CallbackStop

        outdata[:] = data
        with self._lock:
            self._frames_played += int(frames)
