"""Plan 03-03 Task 2 — PlaybackEngine.set_skip_ranges + audio-thread hard-jump.

Pins:

* ``set_skip_ranges`` stores sorted, filtered (b > a) ranges atomically
  under ``self._lock``.
* Empty default state (fresh engine has no skip ranges).
* Mid-playback ``set_skip_ranges`` updates are thread-safe (no exception,
  ranges visible on the next callback iteration).
* When the playhead enters a skip range, the audio-thread callback
  advances ``_frames_played`` past the Trash, sets
  ``_pending_skip_to_sec`` to ``end_sec + 0.05`` (50 ms buffer), and
  fills silence. The GUI thread polls ``consume_pending_skip`` and
  calls ``seek()`` to restart the stream cleanly (Plan 03-06c).
* When playback never enters a skip range, position never jumps.
* ``stop()`` clears ``_pending_skip_to_sec`` under the lock.

These tests use a real ``PlaybackEngine`` with the sounddevice
``OutputStream`` and producer-thread plumbing — they exercise the actual
threading discipline rather than only mocking. The audio callback is
exercised by calling it directly (the PortAudio audio thread is replaced
by a synchronous in-test driver) which avoids ALSA / PulseAudio
dependencies on CI.

``default_cache_root`` is imported so conftest's _patch_targets covers
this module even though the engine itself never touches the cache root.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from marmelade.audio import playback as playback_mod
from marmelade.audio.playback import BLOCKSIZE, BUFFERSIZE, PlaybackEngine
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from tests.fixtures.synthesize import make_sine


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    """A 10-second 44.1 kHz mono sine — long enough to span a skip range."""
    path = tmp_path / "trash_skip_fixture.wav"
    make_sine(path, freq_hz=1000.0, amp=0.5, duration_s=10.0, sample_rate=44100, channels=1)
    return path


@pytest.fixture
def engine() -> PlaybackEngine:
    """Fresh engine with no playback in flight."""
    return PlaybackEngine()


class _FakeStatus:
    """Minimal sounddevice status stand-in — ``output_underflow`` always False."""

    output_underflow = False


# =========================================================================
# set_skip_ranges — storage discipline
# =========================================================================
def test_skip_ranges_empty_by_default(engine: PlaybackEngine) -> None:
    """A fresh engine has no skip ranges and no pending skip target."""
    assert engine._skip_ranges == []
    assert engine._pending_skip_to_sec is None


def test_set_skip_ranges_stores_sorted(engine: PlaybackEngine) -> None:
    """Ranges are sorted by start ascending under self._lock."""
    engine.set_skip_ranges([(5.0, 10.0), (1.0, 3.0)])
    assert engine._skip_ranges == [(1.0, 3.0), (5.0, 10.0)]


def test_set_skip_ranges_filters_invalid(engine: PlaybackEngine) -> None:
    """Ranges where end <= start are filtered out defensively."""
    engine.set_skip_ranges([(5.0, 3.0), (1.0, 2.0), (10.0, 10.0)])
    # Only the valid (1.0, 2.0) survives.
    assert engine._skip_ranges == [(1.0, 2.0)]


def test_set_skip_ranges_to_empty_clears(engine: PlaybackEngine) -> None:
    """Setting ``[]`` clears the list — Trash-untouch cycle relies on this."""
    engine.set_skip_ranges([(1.0, 2.0)])
    assert engine._skip_ranges == [(1.0, 2.0)]
    engine.set_skip_ranges([])
    assert engine._skip_ranges == []


def test_set_skip_ranges_coerces_to_float(engine: PlaybackEngine) -> None:
    """Ints are converted to floats so the audio-thread compare is uniform."""
    engine.set_skip_ranges([(1, 2), (3, 4)])
    assert engine._skip_ranges == [(1.0, 2.0), (3.0, 4.0)]
    assert all(
        isinstance(s, float) and isinstance(e, float)
        for s, e in engine._skip_ranges
    )


def test_set_skip_ranges_concurrent_access_does_not_raise(
    engine: PlaybackEngine,
) -> None:
    """Concurrent writers (GUI thread + a stray slot) don't deadlock or tear.

    Spawns 8 threads each calling set_skip_ranges 100 times. The threads
    finish without exception; the final state is one of the values
    written (we don't assert ordering — just absence of deadlock /
    crash).
    """
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def writer(value: int) -> None:
        try:
            barrier.wait(timeout=2.0)
            for _ in range(100):
                engine.set_skip_ranges([(float(value), float(value + 1))])
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert errors == [], f"concurrent set_skip_ranges raised: {errors!r}"
    # Final state has exactly one (s, s+1) entry — proves no torn read.
    final = engine._skip_ranges
    assert len(final) == 1
    s, e = final[0]
    assert e == s + 1.0


# =========================================================================
# _callback — audio-thread hard-jump
# =========================================================================
def _drive_callback(
    engine: PlaybackEngine,
    sample_rate: int,
    start_frame: int,
    frames_played: int,
) -> np.ndarray:
    """Synchronously invoke the audio callback at a chosen playback position.

    The real PortAudio thread is not running; we simulate one iteration
    by pre-loading a block onto the queue, setting the position state
    under the lock, and calling ``_callback`` directly. Returns the
    outdata buffer the callback wrote.
    """
    with engine._lock:
        engine._sample_rate = sample_rate
        engine._start_frame = start_frame
        engine._frames_played = frames_played
        engine._is_playing = True

    # Pre-load one queued block so the callback's tail-end
    # queue.get_nowait + lock block has data to consume on a non-skip
    # iteration. The skip path drains separately (bails before
    # get_nowait), so this is only needed on the no-skip code path.
    block = np.zeros((BLOCKSIZE, 1), dtype=np.float32)
    while not engine._queue.empty():
        try:
            engine._queue.get_nowait()
        except Exception:  # pragma: no cover — defensive
            break
    engine._queue.put_nowait(block)

    outdata = np.full((BLOCKSIZE, 1), 0.7, dtype=np.float32)
    status = _FakeStatus()
    engine._callback(outdata, BLOCKSIZE, None, status)
    return outdata


def test_callback_jumps_when_entering_skip_range(engine: PlaybackEngine) -> None:
    """At position 2.0s with skip [(2.0, 5.0)], callback advances to 5.05s.

    Plan 03-06c — the audio thread advances ``_frames_played`` and sets
    ``_pending_skip_to_sec`` to ``end_sec + 0.05`` (50 ms buffer past
    the half-open range end to avoid float-rounding re-entry). The GUI
    thread's playhead-poll consumes this and calls ``seek()``.
    """
    sr = 44100
    engine.set_skip_ranges([(2.0, 5.0)])
    # Position the playhead exactly at 2.0 s.
    out = _drive_callback(engine, sr, start_frame=0, frames_played=2 * sr)
    # The skip path advances _frames_played to put position at trash_end + 50 ms.
    pos = engine.position_seconds
    assert pos == pytest.approx(5.05, abs=1e-3), (
        f"expected hard-jump to 5.05 s (trash_end + 50 ms), got {pos}"
    )
    # The audio-thread fills silence on the skip callback.
    assert np.all(out == 0.0), "skip callback must fill silence in outdata"
    # GUI-thread seek signal set to trash_end + 50 ms.
    assert engine._pending_skip_to_sec == pytest.approx(5.05, abs=1e-9)


def test_callback_does_not_jump_outside_skip_range(engine: PlaybackEngine) -> None:
    """Position 1.0s with skip [(2.0, 5.0)] — no jump, no silence fill."""
    sr = 44100
    engine.set_skip_ranges([(2.0, 5.0)])
    _drive_callback(engine, sr, start_frame=0, frames_played=1 * sr)
    # Position advances by BLOCKSIZE frames on a normal callback (the
    # queue.get_nowait + tail _lock block runs since we don't skip).
    expected_pos = (1 * sr + BLOCKSIZE) / sr
    assert engine.position_seconds == pytest.approx(expected_pos, abs=1e-4)
    # No pending skip-to-seek.
    assert engine._pending_skip_to_sec is None


def test_callback_does_not_jump_after_skip_range_end(
    engine: PlaybackEngine,
) -> None:
    """Position 6.0s with skip [(2.0, 5.0)] — past the range, no jump."""
    sr = 44100
    engine.set_skip_ranges([(2.0, 5.0)])
    _drive_callback(engine, sr, start_frame=0, frames_played=6 * sr)
    expected_pos = (6 * sr + BLOCKSIZE) / sr
    assert engine.position_seconds == pytest.approx(expected_pos, abs=1e-4)
    assert engine._pending_skip_to_sec is None


def test_callback_jumps_at_range_start_boundary(
    engine: PlaybackEngine,
) -> None:
    """Skip range is half-open [start, end) — start IS inside, end IS NOT."""
    sr = 44100
    engine.set_skip_ranges([(3.0, 4.0)])
    # Position exactly at start.
    _drive_callback(engine, sr, start_frame=0, frames_played=3 * sr)
    assert engine.position_seconds == pytest.approx(4.05, abs=1e-3)
    assert engine._pending_skip_to_sec == pytest.approx(4.05, abs=1e-9)


def test_callback_does_not_jump_at_range_end_boundary(
    engine: PlaybackEngine,
) -> None:
    """Position exactly at range end is OUTSIDE the half-open interval."""
    sr = 44100
    engine.set_skip_ranges([(3.0, 4.0)])
    _drive_callback(engine, sr, start_frame=0, frames_played=4 * sr)
    expected_pos = (4 * sr + BLOCKSIZE) / sr
    assert engine.position_seconds == pytest.approx(expected_pos, abs=1e-4)
    assert engine._pending_skip_to_sec is None


def test_callback_jumps_first_matching_range_when_multiple(
    engine: PlaybackEngine,
) -> None:
    """Multiple ranges — first containing range wins (sorted scan)."""
    sr = 44100
    engine.set_skip_ranges([(1.0, 2.0), (4.0, 6.0)])
    _drive_callback(engine, sr, start_frame=0, frames_played=4 * sr + sr // 2)
    # Position was 4.5 s, inside (4.0, 6.0) — jumps to 6.05.
    assert engine.position_seconds == pytest.approx(6.05, abs=1e-3)
    assert engine._pending_skip_to_sec == pytest.approx(6.05, abs=1e-9)


# =========================================================================
# stop() clears pending skip state (03-06c)
# =========================================================================
def test_stop_clears_pending_skip(engine: PlaybackEngine) -> None:
    """A stale pending skip across stop->play cycles is a bug vector — stop clears it."""
    with engine._lock:
        engine._pending_skip_to_sec = 5.55
    engine.stop()
    assert engine._pending_skip_to_sec is None


# =========================================================================
# _callback + _produce concurrency — gap-closure UAT Test 4
# (deterministic manual-drain shape — no real OS thread per W-3)
# =========================================================================
def test_callback_silence_bail_with_pending_skip(
    engine: PlaybackEngine,
) -> None:
    """Plan 03-06c silence bail: while ``_pending_skip_to_sec`` is set,
    the callback fills silence and returns. The GUI thread (not the
    callback) clears the flag by calling ``consume_pending_skip``.
    """
    sr = 44100
    with engine._lock:
        engine._sample_rate = sr
        engine._pending_skip_to_sec = 5.55

    outdata = np.full((BLOCKSIZE, 1), 0.9, dtype=np.float32)
    engine._callback(outdata, BLOCKSIZE, None, _FakeStatus())

    assert np.all(outdata == 0.0), (
        "BL-01 invariant violated: outdata not zero during silence bail"
    )
    # GUI (not callback) clears the pending skip — must still be set.
    assert engine._pending_skip_to_sec == 5.55


def test_callback_pending_skip_signal_set_at_detection(
    engine: PlaybackEngine,
) -> None:
    """Plan 03-06c — at skip detection the callback sets
    ``_pending_skip_to_sec`` to ``trash_end + 0.05`` and advances
    ``_frames_played`` so the playhead reads as past the Trash. The
    GUI poll then sees the pending value and calls ``seek()`` to
    restart the stream cleanly — same path as user click-to-seek.
    """
    sr = 44100
    with engine._lock:
        engine._sample_rate = sr
        engine._start_frame = 0
        engine._frames_played = 10 * sr  # position 10.0 s
    engine.set_skip_ranges([(10.0, 15.0)])

    outdata = np.full((BLOCKSIZE, 1), 0.9, dtype=np.float32)
    engine._callback(outdata, BLOCKSIZE, None, _FakeStatus())

    # Pending skip-to-seek set with 50 ms buffer past trash_end.
    assert engine._pending_skip_to_sec == pytest.approx(15.05, abs=1e-9), (
        f"_pending_skip_to_sec should be 15.05 s, got {engine._pending_skip_to_sec}"
    )
    # _frames_played advanced so the visible playhead reads as past Trash.
    expected_frames = int(15.05 * sr)
    assert engine._frames_played == expected_frames, (
        f"_frames_played should advance to ~15.05 s, got {engine._frames_played}"
    )
    # Audio silenced for this callback (the GUI hasn't seek'd yet).
    assert np.all(outdata == 0.0), "audio must silence at detection until GUI seeks"


def test_consume_pending_skip_returns_and_clears_atomically(
    engine: PlaybackEngine,
) -> None:
    """Plan 03-06c — GUI-thread polls ``consume_pending_skip`` on each
    playhead-tick. Returns the pending target and clears the flag in
    one lock-protected operation so two consecutive polls cannot both
    fire a redundant ``seek()``.
    """
    with engine._lock:
        engine._pending_skip_to_sec = 7.5
    first = engine.consume_pending_skip()
    second = engine.consume_pending_skip()
    assert first == 7.5
    assert second is None  # cleared by the first call
    assert engine._pending_skip_to_sec is None


def test_callback_silences_until_gui_consumes_pending_skip(
    engine: PlaybackEngine,
) -> None:
    """Plan 03-06c — after detection, subsequent callbacks silence until
    the GUI thread consumes ``_pending_skip_to_sec`` and calls seek().
    This proves the audio-thread side of the contract: no stale audio
    leaks during the GUI-poll latency window (max ~33 ms at 30 Hz).

    The actual seek + stream restart is exercised by the integration
    tests in test_playback.py — this test pins the audio-thread
    silence-until-seen contract in isolation.
    """
    sr = 44100
    with engine._lock:
        engine._sample_rate = sr
        engine._start_frame = 0
        engine._frames_played = 5 * sr
        engine._is_playing = True
    engine.set_skip_ranges([(5.0, 5.5)])

    # Pre-fill the queue — these are pre-skip blocks that must NOT
    # reach outdata while the skip is pending.
    stale_block = np.full((BLOCKSIZE, 1), 0.42, dtype=np.float32)
    for _ in range(BUFFERSIZE):
        engine._queue.put_nowait(stale_block)

    # First callback: detection fires, _pending_skip_to_sec set,
    # _frames_played advanced past Trash, silence written.
    outdata = np.full((BLOCKSIZE, 1), 0.9, dtype=np.float32)
    engine._callback(outdata, BLOCKSIZE, None, _FakeStatus())
    assert engine._pending_skip_to_sec == pytest.approx(5.55, abs=1e-9)
    assert np.all(outdata == 0.0)

    # Next several callbacks (modelling the GUI-poll latency window):
    # _pending_skip_to_sec stays set, every callback silences. Stale
    # queue contents must NOT leak.
    for _ in range(5):
        outdata = np.full((BLOCKSIZE, 1), 0.9, dtype=np.float32)
        engine._callback(outdata, BLOCKSIZE, None, _FakeStatus())
        assert engine._pending_skip_to_sec == pytest.approx(5.55, abs=1e-9), (
            "_pending_skip_to_sec must persist until the GUI consumes it"
        )
        assert np.all(outdata == 0.0), (
            "audio must remain silenced while pending skip is queued"
        )

    # GUI consumes the pending skip — clears the flag and returns the value.
    skip_to = engine.consume_pending_skip()
    assert skip_to == pytest.approx(5.55, abs=1e-9)
    assert engine._pending_skip_to_sec is None

    # Defensive cleanup — drain queue so the engine fixture teardown is clean.
    while not engine._queue.empty():
        try:
            engine._queue.get_nowait()
        except queue.Empty:
            break


# =========================================================================
# Qt-free invariant — playback.py imports only stdlib + numpy + pedalboard
# =========================================================================
def test_playback_module_remains_qt_free() -> None:
    """N-3 invariant: playback.py MUST NOT import PySide6 or pyqtgraph.

    We scan only the import statements (lines starting with ``import`` or
    ``from``) so docstring mentions of the names do not produce a false
    positive. Mirror of the plan's grep gate
    ``! grep -E 'from PySide6|import pyqtgraph|from pyqtgraph'``.
    """
    src_path = Path(playback_mod.__file__)
    src = src_path.read_text(encoding="utf-8")
    bad_imports: list[str] = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        if "PySide6" in stripped or "pyqtgraph" in stripped or "PyQt6" in stripped:
            bad_imports.append(line)
    assert bad_imports == [], (
        f"playback.py contains Qt imports (N-3 violation): {bad_imports!r}"
    )
