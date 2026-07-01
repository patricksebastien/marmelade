"""Plan 02-05 Task 1 — PlaybackEngine unit tests.

18 pins covering the sounddevice OutputStream + bounded queue producer +
threading.Lock-protected position counter architecture. sounddevice is
mocked at the engine API boundary (we patch
``marmelade.audio.playback.sd.OutputStream`` and
``marmelade.audio.playback.AudioFile``) so the tests run deterministically
under CI without libportaudio2.

Three categories:

1. **Module-level constants** (Test 1) — pin BLOCKSIZE / BUFFERSIZE so the
   trade-offs documented in RESEARCH §Pattern 6 stay locked.
2. **Callback semantics** (Tests 3-7, 14, 16) — invoke ``engine._callback``
   directly with synthetic numpy ``out`` buffers + constructed status flags.
   This is the load-bearing path of the PortAudio thread and we must NOT
   touch a real OutputStream to test it.
3. **Lifecycle** (Tests 2, 8-13, 15, 17, 18) — play / pause / stop / seek
   semantics + thread-safe position counter + graceful degradation on missing
   libportaudio2.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from marmelade.audio.playback import (
    BLOCKSIZE,
    BUFFERSIZE,
    PlaybackEngine,
    PlaybackError,
)


# =========================================================================
# Helpers
# =========================================================================
class _FakeStatus:
    """Minimal stand-in for sounddevice.CallbackFlags.

    Only exposes ``output_underflow`` — the only flag the callback inspects.
    """

    def __init__(self, output_underflow: bool = False) -> None:
        self.output_underflow = output_underflow


class _FakeTime:
    """Minimal stand-in for sounddevice's time argument (unused by the callback)."""

    outputBufferDacTime = 0.0  # noqa: N815 — matches PortAudio attribute name
    currentTime = 0.0  # noqa: N815
    inputBufferAdcTime = 0.0  # noqa: N815


def _make_fake_audiofile(samplerate: int = 44100, frames: int = 44100 * 4,
                        channels: int = 1, fill: str = "const",
                        const_value: float = 0.1) -> MagicMock:
    """Return a MagicMock that quacks like a pedalboard.io.AudioFile.

    Read returns shape ``(channels, n)`` float32 chunks until exhausted.

    quick-260621-iuc — ``fill`` selects the synthetic source content so fade
    gains are observable:

    * ``"const"`` (default, preserves every existing test): every frame equals
      ``const_value`` (0.1) so a scaled block is visibly < the source.
    * ``"ones"``: every frame == 1.0 so the applied gain equals the raw fade
      ramp value (easy endpoint comparison against ``np.linspace``).
    * ``"ramp"``: frame ``i`` (ABSOLUTE source frame, tracked across reads) ==
      ``i`` so the seek-aware read returns a known per-frame value.
    """
    af = MagicMock()
    af.samplerate = samplerate
    af.frames = frames
    af.num_channels = channels
    state = {"pos": 0}

    def _read(n: int) -> np.ndarray:
        remaining = frames - state["pos"]
        if remaining <= 0:
            return np.zeros((channels, 0), dtype=np.float32)
        n_actual = min(n, remaining)
        start = state["pos"]
        if fill == "ones":
            out = np.ones((channels, n_actual), dtype=np.float32)
        elif fill == "ramp":
            idx = np.arange(start, start + n_actual, dtype=np.float32)
            out = np.tile(idx, (channels, 1))
        else:  # "const"
            out = np.full((channels, n_actual), const_value, dtype=np.float32)
        state["pos"] += n_actual
        return out

    def _seek(frame_index: int) -> None:
        state["pos"] = int(frame_index)

    af.read.side_effect = _read
    af.seek.side_effect = _seek
    af.close.return_value = None
    return af


def _drain_mono_blocks(engine, af, max_blocks: int = 10_000) -> list[np.ndarray]:
    """Synchronously pull every mono block the engine would produce for ``af``.

    quick-260621-iuc — drives ``engine._read_segment_block(af)`` directly (the
    SAME helper the prebuffer loop + producer use) until it returns EOF. This
    deterministically exercises the truncation + fade math without spinning up
    the producer thread or a real OutputStream. The engine's segment-window
    attributes (_segment_*) must already be set by a prior ``play()`` call.
    """
    blocks: list[np.ndarray] = []
    for _ in range(max_blocks):
        mono = engine._read_segment_block(af)
        if mono.shape[0] == 0:
            break
        blocks.append(mono)
    return blocks


# =========================================================================
# Test 1 — Module-level constants pinned
# =========================================================================
def test_constants_pinned() -> None:
    """BLOCKSIZE=2048 and BUFFERSIZE=20 are the verified play_long_file.py defaults.

    Changing these would alter the latency / underflow-margin trade-off documented
    in RESEARCH §Pattern 6. Locked literals.
    """
    assert BLOCKSIZE == 2048
    assert BUFFERSIZE == 20


# =========================================================================
# Test 2 — Fresh engine initial state
# =========================================================================
def test_initial_state() -> None:
    """A fresh ``PlaybackEngine()`` has is_playing=False and position_seconds=0.0."""
    engine = PlaybackEngine()
    assert engine.is_playing is False
    assert engine.position_seconds == 0.0


# =========================================================================
# Test 3 — Callback underflow path: fill silence, do NOT abort
# =========================================================================
def test_callback_underflow_fills_silence_does_not_abort() -> None:
    """RESEARCH §Pitfall #3 — output_underflow must fill silence and continue.

    The play_long_file.py canonical example raises ``sd.CallbackAbort`` on
    underflow which terminates the stream. Marmelade fills silence so a
    brief dropout under CPU contention (heatmap compute spike) doesn't kill
    playback.
    """
    engine = PlaybackEngine()
    engine._sample_rate = 44100
    out = np.full((BLOCKSIZE, 1), 0.7, dtype=np.float32)  # garbage prefill
    status = _FakeStatus(output_underflow=True)
    # No exception — engine swallows underflow.
    engine._callback(out, BLOCKSIZE, _FakeTime(), status)
    # outdata zero-filled.
    assert np.all(out == 0.0)
    # Counter NOT incremented (no real frames played).
    assert engine._frames_played == 0


# =========================================================================
# Test 4 — Callback empty queue raises CallbackAbort
# =========================================================================
def test_callback_empty_queue_raises_callback_abort_only_at_eof() -> None:
    """Empty queue + producer EOF flag set → CallbackAbort (real end-of-stream).

    Plan 03-06b — the callback no longer aborts on every empty queue, because
    Trash-skip detection drains the queue intentionally and the producer
    needs one scheduler slice to refill it. Aborting on the transient empty
    window would kill the stream mid-skip. The producer sets _producer_eof
    when it breaks out of its read loop at real EOF; the callback consults
    that flag to distinguish "real EOF" (→ abort) from "drained for skip"
    (→ silence, wait for producer's first post-seek put).
    """
    import sounddevice as sd

    engine = PlaybackEngine()
    engine._sample_rate = 44100
    engine._producer_eof.set()  # signal real EOF
    out = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
    status = _FakeStatus(output_underflow=False)
    with pytest.raises(sd.CallbackAbort):
        engine._callback(out, BLOCKSIZE, _FakeTime(), status)


def test_callback_empty_queue_silences_when_no_eof() -> None:
    """Empty queue WITHOUT producer EOF → silence + return (transient).

    Plan 03-06b — protects the drain-at-detection Trash skip flow. Without
    this gate the very next callback after a skip-drain would raise
    CallbackAbort on the empty queue (producer hasn't refilled yet),
    killing the stream and leaving the user with permanent silence past
    the Trash boundary.
    """
    engine = PlaybackEngine()
    engine._sample_rate = 44100
    assert not engine._producer_eof.is_set()
    out = np.full((BLOCKSIZE, 1), 0.9, dtype=np.float32)
    status = _FakeStatus(output_underflow=False)
    # Must NOT raise; must silence outdata.
    engine._callback(out, BLOCKSIZE, _FakeTime(), status)
    assert np.all(out == 0.0), "callback must silence outdata when queue empty + no EOF"


# =========================================================================
# Test 5 — Callback short data raises CallbackStop, zero-pads remainder
# =========================================================================
def test_callback_short_data_raises_callback_stop() -> None:
    """Final-chunk-of-file semantics: short data → fill prefix + zero-pad + CallbackStop.

    The producer's last chunk before EOF is ≤ BLOCKSIZE. The callback writes
    it, zero-fills the remainder, and raises ``sd.CallbackStop`` so sounddevice
    knows to drain and terminate cleanly.
    """
    import sounddevice as sd

    engine = PlaybackEngine()
    engine._sample_rate = 44100
    engine._is_playing = True
    # Pre-queue a 1024-sample chunk (half a block).
    short = np.full((1024, 1), 0.5, dtype=np.float32)
    engine._queue.put_nowait(short)
    out = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
    status = _FakeStatus(output_underflow=False)
    with pytest.raises(sd.CallbackStop):
        engine._callback(out, BLOCKSIZE, _FakeTime(), status)
    # Prefix filled with the queued data.
    assert np.allclose(out[:1024], short)
    # Suffix zero-padded.
    assert np.all(out[1024:] == 0.0)
    # is_playing flipped to False because we hit the final chunk.
    assert engine.is_playing is False


# =========================================================================
# Test 6 — Callback writes _frames_played under lock; position advances
# =========================================================================
def test_callback_writes_frames_played_under_lock() -> None:
    """A full-block callback increments _frames_played by BLOCKSIZE."""
    engine = PlaybackEngine()
    engine._sample_rate = 44100
    block = np.full((BLOCKSIZE, 1), 0.1, dtype=np.float32)
    engine._queue.put_nowait(block)
    out = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
    status = _FakeStatus(output_underflow=False)
    engine._callback(out, BLOCKSIZE, _FakeTime(), status)
    assert engine._frames_played == BLOCKSIZE
    assert engine.position_seconds == pytest.approx(BLOCKSIZE / 44100.0)


# =========================================================================
# Test 7 — Concurrent reader/writer on _frames_played: no torn read
# =========================================================================
def test_position_seconds_reads_under_lock() -> None:
    """Two threads racing increment + read; no exception, no torn read.

    threading.Lock on every read AND write of _frames_played is the load-
    bearing invariant. On a 64-bit Python build the int read is naturally
    atomic, but the lock also serialises the start_frame addition and the
    division by sample_rate so we never observe a stale composite value.
    """
    engine = PlaybackEngine()
    engine._sample_rate = 44100
    n_iters = 1000

    def writer() -> None:
        for _ in range(n_iters):
            with engine._lock:
                engine._frames_played += 1

    def reader() -> None:
        for _ in range(n_iters):
            _ = engine.position_seconds  # property reads under lock

    t_write = threading.Thread(target=writer)
    t_read = threading.Thread(target=reader)
    t_write.start()
    t_read.start()
    t_write.join(timeout=5.0)
    t_read.join(timeout=5.0)
    assert not t_write.is_alive()
    assert not t_read.is_alive()
    assert engine._frames_played == n_iters


# =========================================================================
# Test 8 — engine.play() initializes state and constructs OutputStream
# =========================================================================
def test_play_initializes_state(tmp_path) -> None:
    """play() wires the stream with the right kwargs and sets _is_playing.

    Mocks sd.OutputStream + AudioFile. Verifies kwargs match the
    RESEARCH §Pattern 6 contract (samplerate, blocksize=2048, channels=1,
    dtype='float32', latency='low').
    """
    fake_af = _make_fake_audiofile(samplerate=44100, frames=44100 * 4)
    fake_stream = MagicMock()

    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream",
               return_value=fake_stream) as mock_ctor:
        engine = PlaybackEngine()
        engine.play(str(tmp_path / "fake.wav"), start_seconds=2.5)
        # is_playing flipped True.
        assert engine.is_playing is True
        # start_frame computed from sample_rate.
        assert engine._start_frame == int(2.5 * 44100)
        # OutputStream constructor was called with the documented contract.
        assert mock_ctor.called
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs["samplerate"] == 44100
        assert kwargs["blocksize"] == BLOCKSIZE
        assert kwargs["channels"] == 1
        assert kwargs["dtype"] == "float32"
        assert kwargs["latency"] == "low"
        assert callable(kwargs["callback"])
        # stream.start() was called.
        assert fake_stream.start.called
        # Producer thread was spawned.
        assert engine._producer is not None
        assert engine._producer.is_alive() or engine._producer.daemon
    # Tear down cleanly so the producer doesn't outlive the test.
    engine.stop()


# =========================================================================
# Test 9 — play() prebuffers BUFFERSIZE blocks
# =========================================================================
def test_play_prebuffers_BUFFERSIZE_blocks(tmp_path) -> None:
    """After play(), the queue has roughly BUFFERSIZE items pre-loaded.

    The prebuffer loop runs synchronously inside play() so the stream's first
    callback can pull without underflow. We allow small tolerance for the
    producer thread that ALSO starts adding chunks immediately after.
    """
    fake_af = _make_fake_audiofile(samplerate=44100, frames=44100 * 60)  # 60 s — plenty
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        engine.play(str(tmp_path / "fake.wav"))
        # Sleep briefly to let the producer thread settle, but we mainly care
        # the prebuffer is AT LEAST close to BUFFERSIZE.
        time.sleep(0.05)
        # Queue should be at least half full from the synchronous prebuffer.
        assert engine._queue.qsize() >= BUFFERSIZE // 2
        engine.stop()


# =========================================================================
# Test 10 — pause() stops stream and clears is_playing
# =========================================================================
def test_pause_stops_stream_and_clears_is_playing(tmp_path) -> None:
    """pause() flips is_playing False, sets stop_event, and calls stream.stop()."""
    fake_af = _make_fake_audiofile()
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        engine.play(str(tmp_path / "fake.wav"))
        engine.pause()
        assert fake_stream.stop.called
        assert engine._stop_event.is_set() is True
        assert engine.is_playing is False
        engine.stop()


# =========================================================================
# Test 11 — stop() closes stream, drains queue, resets _stream to None
# =========================================================================
def test_stop_closes_stream_drains_queue(tmp_path) -> None:
    """stop() is the full cleanup: stream.stop() + close(), queue drained, _stream=None."""
    fake_af = _make_fake_audiofile()
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        engine.play(str(tmp_path / "fake.wav"))
        # Stuff some extra items so we can verify drain.
        for _ in range(5):
            try:
                engine._queue.put_nowait(np.zeros((BLOCKSIZE, 1), dtype=np.float32))
            except queue.Full:
                break
        engine.stop()
        assert fake_stream.stop.called
        assert fake_stream.close.called
        assert engine._queue.empty()
        assert engine._stream is None
        assert engine.is_playing is False


# =========================================================================
# Test 12 — seek() restarts the stream at the new frame
# =========================================================================
def test_seek_pauses_then_restarts(tmp_path) -> None:
    """seek() while playing stops the old stream and constructs a new one.

    After seek, _start_frame reflects the new position and _frames_played
    is reset to 0 (the new segment starts from scratch from the callback's
    POV; total position = start_frame + frames_played).
    """
    fake_af = _make_fake_audiofile(samplerate=44100, frames=44100 * 60)
    fake_stream_1 = MagicMock()
    fake_stream_2 = MagicMock()
    stream_ctor = MagicMock(side_effect=[fake_stream_1, fake_stream_2])
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", new=stream_ctor):
        engine = PlaybackEngine()
        engine.play(str(tmp_path / "fake.wav"))
        engine.seek(5.0)
        # Old stream stopped + closed.
        assert fake_stream_1.stop.called
        assert fake_stream_1.close.called
        # start_frame reset for the new segment.
        assert engine._start_frame == int(5.0 * 44100)
        assert engine._frames_played == 0
        engine.stop()


# =========================================================================
# Test 13 — seek() while paused does NOT auto-resume playback
# =========================================================================
def test_seek_while_paused_does_not_start_playback(tmp_path) -> None:
    """If the engine was paused, seek() updates position but does not auto-play.

    Implementation choice per RESEARCH §Pattern 6 sketch — seek only resumes
    if it was already playing. This matches DAW behavior (seek-while-paused
    moves the cursor; press play to start).
    """
    fake_af = _make_fake_audiofile(samplerate=44100, frames=44100 * 60)
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        engine.play(str(tmp_path / "fake.wav"))
        engine.pause()
        engine.seek(5.0)
        assert engine.is_playing is False


# =========================================================================
# Test 14 — Sequential callback invocations advance position correctly
# =========================================================================
def test_position_seconds_during_callback_progression() -> None:
    """Five sequential _callback invocations on 2048-frame blocks at 44.1 kHz
    yield position_seconds = 5 * 2048 / 44100 ≈ 0.232 s.

    Pins the integer-math composite of (start_frame + frames_played) / sample_rate
    that the QTimer-driven playhead reads.
    """
    engine = PlaybackEngine()
    engine._sample_rate = 44100
    for _ in range(10):
        engine._queue.put_nowait(np.full((BLOCKSIZE, 1), 0.1, dtype=np.float32))
    out = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
    status = _FakeStatus(output_underflow=False)
    for _ in range(5):
        engine._callback(out, BLOCKSIZE, _FakeTime(), status)
    expected = (5 * BLOCKSIZE) / 44100.0
    assert engine.position_seconds == pytest.approx(expected)


# =========================================================================
# Test 15 — playback.py has no Qt imports (Qt-free policy)
# =========================================================================
def test_callback_never_imports_qt() -> None:
    """playback.py imports zero Qt symbols.

    The PortAudio callback runs on the audio thread; Qt-bridging happens in
    MainWindow via the position_seconds property poll. Keeping playback.py
    Qt-free isolates the audio thread from any accidental widget reference.
    Tests N-3-style structural invariant — the module's executable code (NOT
    including docstrings) must contain zero ``import PySide6`` / ``import PyQt6``
    lines. We inspect the AST so the docstring's prose about the policy doesn't
    false-match.
    """
    import ast
    import marmelade.audio.playback as playback_module
    import inspect

    source = inspect.getsource(playback_module)
    tree = ast.parse(source)
    forbidden_modules = {"PySide6", "PyQt6", "PyQt5"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in forbidden_modules, (
                    f"playback.py imports forbidden Qt module {alias.name!r}"
                )
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden_modules, (
                f"playback.py imports forbidden Qt module {node.module!r}"
            )


# =========================================================================
# Test 16 — _callback closure references no widget/signal
# =========================================================================
def test_callback_no_widget_references_in_closure() -> None:
    """The callback's closure only captures ``self`` (the engine).

    Pitfall #4 invariant: the PortAudio thread NEVER touches a Qt widget.
    Since the callback is a bound method, the closure naturally contains
    only ``self`` — the engine's atomic primitives. We assert this by
    inspecting the method's code object for any name that looks like a
    Qt symbol.
    """
    engine = PlaybackEngine()
    # Bound method's underlying function code object.
    code = engine._callback.__func__.__code__
    # The callback's referenced names (constants + global lookups).
    forbidden = ("PySide6", "PyQt6", "QWidget", "Signal", "QObject", "setValue", "setText")
    all_names = code.co_names + code.co_varnames + code.co_freevars
    for name in all_names:
        for bad in forbidden:
            assert bad not in name, (
                f"_callback references forbidden Qt symbol '{name}' "
                f"(matches '{bad}'); Pitfall #4 invariant violated"
            )


# =========================================================================
# Test 17 — finished_callback sets _finished_event
# =========================================================================
def test_finished_callback_sets_finished_event(tmp_path) -> None:
    """When the stream signals finished, _finished_event is set.

    sounddevice's OutputStream.finished_callback fires after stream.stop()
    drains its pending buffers. We wire it to engine._finished_event.set so
    callers can wait on the event for clean shutdown semantics.
    """
    fake_af = _make_fake_audiofile()

    # We capture the finished_callback argument so we can invoke it ourselves.
    captured = {}

    def _ctor(*args, **kwargs):
        captured["finished_callback"] = kwargs.get("finished_callback")
        return MagicMock()

    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", side_effect=_ctor):
        engine = PlaybackEngine()
        engine.play(str(tmp_path / "fake.wav"))
        # Simulate sounddevice firing the finished_callback.
        assert captured["finished_callback"] is not None
        captured["finished_callback"]()
        assert engine._finished_event.is_set()
        engine.stop()


# =========================================================================
# Test 18 — play() raises PlaybackError when sounddevice unavailable
# =========================================================================
def test_playback_error_on_import_failure(monkeypatch, tmp_path) -> None:
    """Simulate libportaudio2-missing: engine constructs OK but play() raises.

    The module-level try/except around ``import sounddevice`` sets
    ``_SOUNDDEVICE_AVAILABLE = False`` on dlopen failure. The engine instance
    captures this flag into ``self._sd_available`` AT CONSTRUCTION TIME (W2 —
    so monkeypatch must apply BEFORE constructing the engine). Construction
    succeeds; play() raises PlaybackError so MainWindow can gracefully
    disable the toolbar action.
    """
    monkeypatch.setattr("marmelade.audio.playback._SOUNDDEVICE_AVAILABLE", False)
    monkeypatch.setattr(
        "marmelade.audio.playback._SOUNDDEVICE_IMPORT_ERROR",
        OSError("libportaudio2 not found"),
    )
    # Construct AFTER monkeypatch so instance captures patched values.
    engine = PlaybackEngine()
    assert engine._sd_available is False  # Sanity-check.
    assert engine.is_available is False
    with pytest.raises(PlaybackError):
        engine.play(str(tmp_path / "fake.wav"))


# =========================================================================
# Test 19 — Bug #1 regression: prime() + seek() → non-zero position_seconds
# =========================================================================
def test_prime_sets_sample_rate_and_duration(tmp_path) -> None:
    """Bug #1 regression pin — prime() populates ``_sample_rate`` at file-open
    time so a pre-play seek lands on a non-zero frame.

    Before the fix, ``_sample_rate`` stayed 0 until the first ``play()`` call;
    ``seek()`` then silently zeroed its target via ``int(t * sr) if sr else 0``.
    With ``prime()`` wired into ``MainWindow._open_file``, the click-to-seek
    interaction works BEFORE the user presses spacebar.

    This test exercises the real prime() → seek() → position_seconds chain
    against a tiny synthetic WAV; no sounddevice / OutputStream mocking is
    needed because prime() only opens a pedalboard ``AudioFile`` (which works
    in CI without libportaudio2).
    """
    # 1 s, 44100 Hz, stereo float32 silence — small + portable.
    sr = 44100
    wav_path = tmp_path / "primer.wav"
    audio = np.zeros((2, sr), dtype=np.float32)  # (channels, frames)
    # soundfile expects (frames, channels), so transpose.
    sf.write(str(wav_path), audio.T, sr)

    engine = PlaybackEngine()
    # If the CI environment short-circuits prime() because libportaudio2 is
    # missing, force _sd_available True so the test exercises the real path.
    # ``prime()`` itself does NOT touch sounddevice (only pedalboard's
    # AudioFile), so flipping this flag is safe.
    engine._sd_available = True

    # Baseline: fresh engine reports 0.0 because _sample_rate==0.
    assert engine.position_seconds == 0.0

    engine.prime(str(wav_path))

    # prime() does NOT start playback.
    assert not engine.is_playing
    # No seek yet — engine is at frame 0 of the primed file. position_seconds
    # is still 0.0, but for a DIFFERENT reason: _sample_rate is now set, but
    # _start_frame and _frames_played are both 0.
    assert engine.position_seconds == 0.0

    # The crux of the bug: BEFORE the fix, seek() saw _sample_rate==0 and
    # silently zeroed _start_frame. AFTER the fix, seek(0.5) with sr=44100
    # sets _start_frame = 22050 → position_seconds == 0.5.
    engine.seek(0.5)
    assert abs(engine.position_seconds - 0.5) < 0.01, (
        f"seek(0.5) should land at 0.5 s with primed engine, got "
        f"{engine.position_seconds}"
    )


# =========================================================================
# quick-260621-iuc — keeper-segment stop-at-end + fade pins
# =========================================================================
#
# These exercise the SHARED _read_segment_block helper (the same path the
# synchronous prebuffer loop AND the producer thread use). To get a clean,
# deterministic full-segment drain we play() with the mocked stream (which
# sets the _segment_* window attributes correctly), then RESET the AudioFile
# position + the producer-owned cursor back to the segment start and drain via
# _read_segment_block directly. This avoids racing the real producer thread
# while still testing the exact code that runs in production.


def _play_then_reset_for_drain(engine, fake_af, path, **play_kwargs):
    """play() to set _segment_* window, then rewind af + cursor to seg start."""
    engine.play(path, **play_kwargs)
    # play() spawned a producer + consumed the prebuffer; stop it so it can't
    # race our manual drain, then rewind to the segment start.
    engine._stop_event.set()
    if engine._producer is not None:
        engine._producer.join(timeout=2.0)
    fake_af.seek(engine._segment_start_frame)
    engine._segment_read_cursor = engine._segment_start_frame
    engine._stop_event.clear()


def test_play_backward_compat_full_file(tmp_path) -> None:
    """play(path) with no end/fade reads to EOF and applies NO gain.

    Pins byte-identical behavior: total mono frames == file frame count (no
    truncation) and the first block equals the raw source const (not scaled).
    """
    frames = BLOCKSIZE * 5 + 137  # non-block-aligned tail
    fake_af = _make_fake_audiofile(samplerate=44100, frames=frames, const_value=0.1)
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        _play_then_reset_for_drain(engine, fake_af, str(tmp_path / "f.wav"))
        # No end/fade requested.
        assert engine._segment_fade_in_frames == 0
        assert engine._segment_fade_out_frames == 0
        assert engine._segment_end_frame == frames
        blocks = _drain_mono_blocks(engine, fake_af)
        total = sum(b.shape[0] for b in blocks)
        assert total == frames, "full-file playback must read every source frame"
        # No gain applied — first block is the raw const.
        assert np.allclose(blocks[0], 0.1), "backward-compat path must apply no gain"
        engine.stop()


def test_play_stops_at_end_seconds(tmp_path) -> None:
    """end_seconds truncates reads + sets _producer_eof; final block → CallbackStop.

    The producer reads NO source frame past int(end*sr); the truncated final
    block drives the existing CallbackStop path and is_playing flips False.
    """
    import sounddevice as sd

    sr = 44100
    frames = sr * 10
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, const_value=0.1)
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        start_s, end_s = 1.0, 3.0
        _play_then_reset_for_drain(
            engine, fake_af, str(tmp_path / "f.wav"),
            start_seconds=start_s, end_seconds=end_s, fade_seconds=0.0,
        )
        s_frame = int(start_s * sr)
        e_frame = int(end_s * sr)
        blocks = _drain_mono_blocks(engine, fake_af)
        total = sum(b.shape[0] for b in blocks)
        assert total == e_frame - s_frame, (
            "producer must read exactly the segment window, no frame past end"
        )
        # The cursor never advanced past the segment end (last read may be the
        # one that crossed; truncation kept frames < end).
        assert engine._segment_end_frame == e_frame

        # Run the EXACT producer loop (_produce) against a rewound af and assert
        # it sets _producer_eof when the segment-end truncation collapses a
        # read to zero frames. We drain the bounded queue concurrently from
        # this thread by consuming as the producer fills (the real callback
        # plays the role of consumer). To keep it single-threaded + simple we
        # patch the queue to an unbounded one for THIS phase so _produce runs
        # to the segment-end EOF without blocking on queue.Full.
        fake_af.seek(s_frame)
        engine._segment_read_cursor = s_frame
        engine._producer_eof.clear()
        engine._stop_event.clear()
        engine._queue = queue.Queue()  # unbounded for the loop-to-EOF check
        engine._produce(fake_af)  # runs to segment-end EOF, sets _producer_eof
        assert engine._producer_eof.is_set(), "segment-end must set _producer_eof"

        # The last enqueued block is the truncated final block (< BLOCKSIZE).
        # Drive full blocks through _callback until that short block triggers
        # the existing CallbackStop path and is_playing flips False.
        engine._is_playing = True
        out = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
        status = _FakeStatus(output_underflow=False)
        stopped = False
        for _ in range(10_000):
            try:
                engine._callback(out, BLOCKSIZE, _FakeTime(), status)
            except (sd.CallbackStop, sd.CallbackAbort):
                stopped = True
                break
        assert stopped, "final truncated/empty block must raise CallbackStop/Abort"
        assert engine.is_playing is False, "is_playing flips False at segment end"
        engine.stop()


def test_fade_shape_matches_export(tmp_path) -> None:
    """Fade-in 0→1 then fade-out →0 with export's endpoint=True linspace shape.

    Source == 1.0 everywhere so the produced sample IS the fade gain. Assert
    first fade_frames ramp up from ~0, last fade_frames ramp down to ~0, mid
    samples at full amplitude, and endpoints match np.linspace(endpoint=True).
    """
    sr = 44100
    # Segment ~0.5 s, fade 0.1 s (well under segment_len // 2 so no cap).
    frames = sr * 2
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, fill="ones")
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        start_s, end_s, fade_s = 0.0, 0.5, 0.1
        _play_then_reset_for_drain(
            engine, fake_af, str(tmp_path / "f.wav"),
            start_seconds=start_s, end_seconds=end_s, fade_seconds=fade_s,
        )
        fade_n = engine._segment_fade_in_frames
        assert fade_n == int(fade_s * sr), "no cap expected for this segment"
        blocks = _drain_mono_blocks(engine, fake_af)
        env = np.concatenate([b[:, 0] for b in blocks])
        seg_len = int(end_s * sr) - int(start_s * sr)
        assert env.shape[0] == seg_len

        # Fade-in: first sample ~0.0, monotone non-decreasing, last fade sample 1.0.
        fade_in = env[:fade_n]
        assert fade_in[0] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(fade_in) >= -1e-7), "fade-in must be non-decreasing"
        ref_in = np.linspace(0.0, 1.0, fade_n, endpoint=True, dtype=np.float32)
        assert np.allclose(fade_in, ref_in, atol=1e-5), "fade-in shape must match export"

        # Mid region at full amplitude.
        mid = env[fade_n:seg_len - fade_n]
        assert np.allclose(mid, 1.0, atol=1e-6), "non-fade region must be full amplitude"

        # Fade-out: last sample ~0.0, monotone non-increasing, matches linspace.
        fade_out = env[seg_len - fade_n:]
        assert fade_out[-1] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(fade_out) <= 1e-7), "fade-out must be non-increasing"
        ref_out = np.linspace(1.0, 0.0, fade_n, endpoint=True, dtype=np.float32)
        assert np.allclose(fade_out, ref_out, atol=1e-5), "fade-out shape must match export"
        engine.stop()


def test_asymmetric_fade_in_suppressed_out_kept(tmp_path) -> None:
    """fade_in_seconds=0 drops the fade-IN while fade_out_seconds keeps the fade-OUT.

    quick-260622-ud0 — the keeper middle/end auditions request NO fade-in but a
    fade-out. Source == 1.0 so the produced sample IS the gain: the first kept
    frame must be full amplitude (1.0, no ramp-up) and the tail must match the
    export endpoint=True linspace down to 0.0.
    """
    sr = 44100
    frames = sr * 2
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, fill="ones")
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        start_s, end_s, fade_out_s = 0.0, 0.5, 0.1
        _play_then_reset_for_drain(
            engine, fake_af, str(tmp_path / "f.wav"),
            start_seconds=start_s, end_seconds=end_s,
            fade_in_seconds=0.0, fade_out_seconds=fade_out_s,
        )
        assert engine._segment_fade_in_frames == 0, "fade-in must be suppressed"
        fade_out_n = engine._segment_fade_out_frames
        assert fade_out_n == int(fade_out_s * sr), "fade-out frames must be kept"

        blocks = _drain_mono_blocks(engine, fake_af)
        env = np.concatenate([b[:, 0] for b in blocks])
        seg_len = int(end_s * sr) - int(start_s * sr)
        assert env.shape[0] == seg_len

        # NO fade-in: the very first kept frame is at full amplitude.
        assert env[0] == pytest.approx(1.0, abs=1e-6), "no fade-in: first frame full gain"
        # Region before the fade-out is full amplitude.
        mid = env[:seg_len - fade_out_n]
        assert np.allclose(mid, 1.0, atol=1e-6), "pre-fade-out region must be full amplitude"
        # Fade-out tail matches export linspace down to exactly 0.0.
        fade_out = env[seg_len - fade_out_n:]
        assert fade_out[-1] == pytest.approx(0.0, abs=1e-6)
        assert np.all(np.diff(fade_out) <= 1e-7), "fade-out must be non-increasing"
        ref_out = np.linspace(1.0, 0.0, fade_out_n, endpoint=True, dtype=np.float32)
        assert np.allclose(fade_out, ref_out, atol=1e-5), "fade-out shape must match export"
        engine.stop()


def test_fade_cap_short_segment(tmp_path) -> None:
    """Segment <= 2*fade engages the segment_len // 2 cap; windows do not overlap.

    Effective fade_frames == segment_len // 2; fade-in window + fade-out window
    are disjoint (no sample scaled by BOTH ramps).
    """
    sr = 44100
    frames = sr * 5
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, fill="ones")
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        # 0.6 s segment, request 2 s fade → cap to segment_len // 2 (0.3 s).
        start_s, end_s, fade_s = 0.0, 0.6, 2.0
        _play_then_reset_for_drain(
            engine, fake_af, str(tmp_path / "f.wav"),
            start_seconds=start_s, end_seconds=end_s, fade_seconds=fade_s,
        )
        seg_len = int(end_s * sr) - int(start_s * sr)
        assert engine._segment_fade_in_frames == seg_len // 2, "cap must be segment_len // 2"
        assert engine._segment_fade_out_frames == seg_len // 2, "fade-out also capped"
        fade_n = engine._segment_fade_in_frames

        blocks = _drain_mono_blocks(engine, fake_af)
        env = np.concatenate([b[:, 0] for b in blocks])
        assert env.shape[0] == seg_len
        # Windows are exactly adjacent (or meet) — fade-in [0, fade_n),
        # fade-out [seg_len - fade_n, seg_len). With the cap, fade_n ==
        # seg_len // 2 so for even seg_len they tile perfectly; for odd there
        # is a single full-amplitude sample between them. Assert no sample is
        # the PRODUCT of both ramps (i.e. each sample obeys exactly one ramp).
        fade_out_start = seg_len - fade_n
        assert fade_out_start >= fade_n, "fade windows must not overlap under the cap"
        # Endpoints still hit the boundary values.
        assert env[0] == pytest.approx(0.0, abs=1e-6)
        assert env[-1] == pytest.approx(0.0, abs=1e-6)
        engine.stop()


def test_fade_in_present_in_prebuffer(tmp_path) -> None:
    """The fade-in ramp is applied in the SYNCHRONOUS prebuffer loop.

    Bug-class the design calls out: the first blocks (carrying the fade-in)
    are read by play()'s prebuffer loop, NOT the producer. Inspect ONLY the
    items the prebuffer loop enqueued and assert the 0→1 ramp is present.
    """
    sr = 44100
    # Segment short enough that the whole fade-in lands inside the first few
    # prebuffered blocks (fade_n < BLOCKSIZE so it fits in block 0).
    frames = sr * 2
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, fill="ones")
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        # Stop the producer thread immediately after play() so it cannot add
        # to the queue — what remains is EXACTLY the prebuffered items.
        engine.play(
            str(tmp_path / "f.wav"),
            start_seconds=0.0, end_seconds=0.5, fade_seconds=0.01,
        )
        engine._stop_event.set()
        if engine._producer is not None:
            engine._producer.join(timeout=2.0)
        fade_n = engine._segment_fade_in_frames
        assert 0 < fade_n < BLOCKSIZE, "fade-in must fit inside the first prebuffered block"

        # Drain the prebuffered queue (do NOT call _read_segment_block again).
        prebuffered = []
        while True:
            try:
                prebuffered.append(engine._queue.get_nowait())
            except queue.Empty:
                break
        assert prebuffered, "prebuffer must have enqueued at least one block"
        first = prebuffered[0][:, 0]
        # The fade-in lives at the head of the FIRST prebuffered block.
        assert first[0] == pytest.approx(0.0, abs=1e-6), (
            "fade-in must be applied in the prebuffer path, not only the producer"
        )
        ref_in = np.linspace(0.0, 1.0, fade_n, endpoint=True, dtype=np.float32)
        assert np.allclose(first[:fade_n], ref_in, atol=1e-5)
        # Past the fade-in the source (ones) is at full amplitude.
        assert first[fade_n] == pytest.approx(1.0, abs=1e-6)
        engine.stop()


def test_position_seconds_true_source_time_under_segment(tmp_path) -> None:
    """With start+end set, position_seconds stays (start_frame + frames_played)/sr.

    NOT reset to segment-relative zero — the waveform playhead must track true
    source time so it lands on the keeper's source position.
    """
    sr = 44100
    frames = sr * 10
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, const_value=0.1)
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        start_s, end_s = 2.0, 6.0
        engine.play(
            str(tmp_path / "f.wav"),
            start_seconds=start_s, end_seconds=end_s, fade_seconds=0.0,
        )
        # Stop producer so it can't drain/refill underneath us.
        engine._stop_event.set()
        if engine._producer is not None:
            engine._producer.join(timeout=2.0)
        start_frame = int(start_s * sr)
        assert engine._start_frame == start_frame

        # Drive a few full callback blocks from prebuffered data.
        out = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
        status = _FakeStatus(output_underflow=False)
        n_cb = 3
        for _ in range(n_cb):
            engine._callback(out, BLOCKSIZE, _FakeTime(), status)
        expected = (start_frame + n_cb * BLOCKSIZE) / float(sr)
        assert engine.position_seconds == pytest.approx(expected), (
            "position must be true source time, not segment-relative"
        )
        engine.stop()


def test_position_seconds_subtracts_output_latency(tmp_path) -> None:
    """position_seconds compensates for the stream's output latency (A/V sync).

    The callback advances _frames_played when a block is HANDED to PortAudio's
    output buffer, but that audio is audible _output_latency_sec later. The
    reported position must subtract the latency so the visual playhead tracks
    what is heard NOW, not the frame merely queued for output.
    """
    sr = 44100
    frames = sr * 10
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, const_value=0.1)
    fake_stream = MagicMock()
    fake_stream.latency = 0.05  # 50 ms granted output latency → 2205 frames
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        engine.play(str(tmp_path / "f.wav"))
        engine._stop_event.set()
        if engine._producer is not None:
            engine._producer.join(timeout=2.0)
        assert engine._output_latency_sec == pytest.approx(0.05)

        out = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
        status = _FakeStatus(output_underflow=False)
        n_cb = 3
        for _ in range(n_cb):
            engine._callback(out, BLOCKSIZE, _FakeTime(), status)

        latency_frames = int(0.05 * sr)
        audible = n_cb * BLOCKSIZE - latency_frames
        expected = audible / float(sr)
        assert engine.position_seconds == pytest.approx(expected)

        # Pause clears the compensation so resume reads the exact last position.
        engine.pause()
        assert engine._output_latency_sec == 0.0
        engine.stop()


def test_position_seconds_clamps_to_start_before_audio_is_heard(tmp_path) -> None:
    """While played frames < latency frames, position stays at the segment start."""
    sr = 44100
    frames = sr * 10
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, const_value=0.1)
    fake_stream = MagicMock()
    fake_stream.latency = 0.5  # huge latency → 22050 frames, exceeds 1 block
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        start_s = 2.0
        engine.play(str(tmp_path / "f.wav"), start_seconds=start_s)
        engine._stop_event.set()
        if engine._producer is not None:
            engine._producer.join(timeout=2.0)

        out = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
        status = _FakeStatus(output_underflow=False)
        engine._callback(out, BLOCKSIZE, _FakeTime(), status)  # 2048 < 22050
        # Audible offset clamps to 0 → playhead pinned at the segment start.
        assert engine.position_seconds == pytest.approx(start_s)
        engine.stop()


# =========================================================================
# quick-260622-vwr — per-segment normalize affine (WYSIWYG A-mode preview)
# =========================================================================
#
# A-mode keeper preview plays the raw source proxy windowed to the keeper,
# but the normalized waveform display applies a DC-remove + peak-to-target
# affine. These pins make play() grow an optional per-segment affine and add
# a pure pre-pass (compute_segment_normalize_params) that streams the keeper
# span block-by-block to derive the SAME (dc, scale) the WYSIWYG render uses.


def _make_dc_sine_wav(tmp_path, sr=44100, seconds=1.0, dc=0.3, amp=0.5):
    """Write a (1-channel) WAV with a known DC offset + sine.

    The signal is ``dc + amp * sin(...)`` so the segment mean ≈ dc and the
    DC-removed (centered) absolute peak ≈ amp. Returns (path, dc, amp).
    """
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float64) / sr
    sig = (dc + amp * np.sin(2.0 * np.pi * 5.0 * t)).astype(np.float32)
    wav_path = tmp_path / "dc_sine.wav"
    # soundfile expects (frames, channels); single channel -> (n, 1) or (n,).
    sf.write(str(wav_path), sig, sr)
    return wav_path, dc, amp


def test_compute_segment_normalize_params_dc_and_peak(tmp_path) -> None:
    """Pre-pass returns the segment mean (DC) and the peak-to-target scale.

    A 1 s WAV with DC≈0.3 + a ±0.5 sine: over the full span the mean ≈ 0.3
    and the centered peak ≈ 0.5, so scale ≈ 10**(-3/20) / 0.5. The returned
    mean must be the SEGMENT mean (sub-window), not the global file mean.
    """
    sr = 44100
    wav_path, dc, amp = _make_dc_sine_wav(tmp_path, sr=sr, seconds=1.0,
                                          dc=0.3, amp=0.5)
    engine = PlaybackEngine()
    engine._sd_available = True  # pre-pass never touches sounddevice

    target_db = -3.0
    dc_out, scale_out = engine.compute_segment_normalize_params(
        str(wav_path), 0.0, 1.0, target_db
    )
    assert dc_out == pytest.approx(dc, abs=2e-3), (
        f"segment mean (DC) should be ≈{dc}, got {dc_out}"
    )
    target_linear = 10.0 ** (target_db / 20.0)
    expected_scale = target_linear / amp
    assert scale_out == pytest.approx(expected_scale, rel=2e-2), (
        f"scale should map centered peak {amp} to {target_linear}"
    )

    # Sub-window: mean over [start, end) must use ONLY that window. Compare
    # against the direct numpy mean of the same window.
    raw, _ = sf.read(str(wav_path), dtype="float32")
    start_s, end_s = 0.25, 0.75
    sf_start, sf_end = int(start_s * sr), int(end_s * sr)
    window_mean = float(np.mean(raw[sf_start:sf_end]))
    dc_win, _ = engine.compute_segment_normalize_params(
        str(wav_path), start_s, end_s, target_db
    )
    assert dc_win == pytest.approx(window_mean, abs=2e-3), (
        "pre-pass must use the segment window mean, not the global file mean"
    )


def test_compute_segment_normalize_params_silent_returns_identity(tmp_path) -> None:
    """Silent / pure-DC / degenerate segments return scale exactly 1.0.

    * A fully-zero segment -> (0.0, 1.0): no noise-floor blow-up.
    * A zero-length (count==0) window -> (0.0, 1.0).
    """
    sr = 44100
    # Fully-zero 0.5 s WAV.
    wav_path = tmp_path / "silent.wav"
    sf.write(str(wav_path), np.zeros(int(0.5 * sr), dtype=np.float32), sr)
    engine = PlaybackEngine()
    engine._sd_available = True

    dc_out, scale_out = engine.compute_segment_normalize_params(
        str(wav_path), 0.0, 0.5, -3.0
    )
    assert dc_out == pytest.approx(0.0, abs=1e-9)
    assert scale_out == 1.0, "silent segment must not amplify noise floor"

    # Degenerate / zero-length window (start == end -> count 0).
    dc_z, scale_z = engine.compute_segment_normalize_params(
        str(wav_path), 0.25, 0.25, -3.0
    )
    assert dc_z == 0.0
    assert scale_z == 1.0


def test_play_affine_default_is_noop(tmp_path) -> None:
    """play() with no normalize params is byte-identical to backward-compat.

    Mirrors test_play_backward_compat_full_file: first block equals the raw
    source const (no scaling), proving the default affine path is a no-op.
    """
    frames = BLOCKSIZE * 5 + 137
    fake_af = _make_fake_audiofile(samplerate=44100, frames=frames, const_value=0.1)
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        _play_then_reset_for_drain(engine, fake_af, str(tmp_path / "f.wav"))
        # Default affine is identity.
        assert engine._segment_norm_dc == 0.0
        assert engine._segment_norm_scale == 1.0
        blocks = _drain_mono_blocks(engine, fake_af)
        assert np.allclose(blocks[0], 0.1), "default affine path must apply no gain"
        engine.stop()


def test_play_affine_dc_removed_and_peak_scaled(tmp_path) -> None:
    """A non-default affine removes DC and scales the source per (dc, scale).

    Source const 0.3; pass normalize_dc=0.3, normalize_scale=2.0 -> every
    produced sample is (0.3 - 0.3) * 2.0 == 0.0. With const 0.5 and dc=0.3,
    scale=2.0 -> (0.5-0.3)*2.0 == 0.4. We use a const source so the per-block
    affine output is exactly predictable.
    """
    frames = BLOCKSIZE * 4
    fake_af = _make_fake_audiofile(samplerate=44100, frames=frames, const_value=0.5)
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        _play_then_reset_for_drain(
            engine, fake_af, str(tmp_path / "f.wav"),
            normalize_dc=0.3, normalize_scale=2.0,
        )
        assert engine._segment_norm_dc == pytest.approx(0.3)
        assert engine._segment_norm_scale == pytest.approx(2.0)
        blocks = _drain_mono_blocks(engine, fake_af)
        env = np.concatenate([b[:, 0] for b in blocks])
        expected = (0.5 - 0.3) * 2.0
        assert np.allclose(env, expected, atol=1e-6), (
            "affine must apply (mono - dc) * scale per block"
        )
        # float32 preserved.
        assert blocks[0].dtype == np.float32
        engine.stop()


def test_play_affine_then_fade_ends_hit_zero(tmp_path) -> None:
    """Affine applied BEFORE fade: the fade-out tail still reaches exactly 0.0.

    Source const 0.5 + a non-default affine + a fade-out. The fade envelope
    multiplies the already-normalized signal, so the last fade-out sample is
    still exactly 0.0.
    """
    sr = 44100
    frames = sr * 2
    fake_af = _make_fake_audiofile(samplerate=sr, frames=frames, const_value=0.5)
    fake_stream = MagicMock()
    with patch("marmelade.audio.playback.AudioFile", return_value=fake_af), \
         patch("marmelade.audio.playback.sd.OutputStream", return_value=fake_stream):
        engine = PlaybackEngine()
        start_s, end_s, fade_s = 0.0, 0.5, 0.1
        _play_then_reset_for_drain(
            engine, fake_af, str(tmp_path / "f.wav"),
            start_seconds=start_s, end_seconds=end_s, fade_seconds=fade_s,
            normalize_dc=0.2, normalize_scale=1.5,
        )
        fade_n = engine._segment_fade_out_frames
        assert fade_n == int(fade_s * sr)
        blocks = _drain_mono_blocks(engine, fake_af)
        env = np.concatenate([b[:, 0] for b in blocks])
        # Last fade-out sample is exactly 0.0 (fade multiplies normalized sig).
        assert env[-1] == pytest.approx(0.0, abs=1e-6), (
            "fade-out tail must hit 0.0 even with a non-default affine"
        )
        # The mid (pre-fade-out) region equals the affine-normalized const.
        normed = (0.5 - 0.2) * 1.5
        seg_len = int(end_s * sr) - int(start_s * sr)
        fade_in_n = engine._segment_fade_in_frames
        mid = env[fade_in_n:seg_len - fade_n]
        assert np.allclose(mid, normed, atol=1e-5), (
            "non-fade region must be the normalized (affine) amplitude"
        )
        engine.stop()
