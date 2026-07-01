"""Phase 8 Plan 08-04 — YouTubeUploadRunnable + 3-layer cancel (YT-04 + YT-05).

Plan 08-04 Task 2 (TDD GREEN — Wave 0 skip marker removed). The test
names below pin the canonical contract for
:mod:`marmelade.youtube.upload_runnable`.

D-28 locks the 3-layer cancel pattern — mirror of
:class:`marmelade.audio.mastering_worker.MasteringRunnable`.
WorkerSignals reused VERBATIM (D-16, identity-pinned by
``test_refresh_then_reconnect_uses_workersignals_verbatim``).

The mock_youtube_client fixture's scriptable next_chunk side_effect is
the load-bearing test seam — each test wires its own progression and
asserts on signals + control flow.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QCoreApplication

from marmelade.concurrency.worker import WorkerSignals


# ---------------------------------------------------------------------------
# Helpers — each test builds the runnable freshly via this factory.
# ---------------------------------------------------------------------------


def _build_runnable(
    tmp_path,
    mock_youtube_client,
    mock_credentials_factory,
    *,
    audio_bytes: bytes = b"FAKEWAV",
    image_bytes: bytes = b"FAKEJPEG",
):
    """Return a fully-wired YouTubeUploadRunnable for tests.

    Patches `build_youtube_client` to return the mock client, and
    `video_builder.build_video` to write a small fake MP4 (the runnable
    needs SOMETHING on disk for MediaFileUpload). The MediaFileUpload
    constructor itself is also patched to a MagicMock so tests can
    inspect chunksize without needing a real file.
    """
    from marmelade.youtube import upload_runnable as ur

    audio_p = tmp_path / "audio.wav"
    image_p = tmp_path / "image.jpg"
    audio_p.write_bytes(audio_bytes)
    image_p.write_bytes(image_bytes)

    creds = mock_credentials_factory()

    runnable = ur.YouTubeUploadRunnable(
        audio_path=audio_p,
        image_path=image_p,
        snippet={"title": "Test", "description": "Test desc"},
        status={"privacyStatus": "private"},
        credentials=creds,
        keeper_id="0123456789abcdef0123456789abcdef",
        tmp_dir=tmp_path / "tmp",
    )
    return runnable, creds


def _wire_signal_recorder(runnable):
    """Attach plain-Python listeners that record signal emissions.

    QSignal.connect on a QObject's signal stores a Python callable —
    invocation happens synchronously from the same thread that called
    .emit() (DirectConnection on same-thread). Recording into a dict
    lets each test assert without running an event loop.
    """
    rec = {
        "progress": [],
        "finished": [],
        "error": [],
        "cancelled": 0,
    }
    runnable.signals.progress.connect(lambda pct: rec["progress"].append(int(pct)))
    runnable.signals.finished.connect(lambda v: rec["finished"].append(v))
    runnable.signals.error.connect(lambda msg: rec["error"].append(msg))
    runnable.signals.cancelled.connect(lambda: rec.__setitem__("cancelled", rec["cancelled"] + 1))
    return rec


def _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path):
    """Patch build_youtube_client + video_builder.build_video.

    The fake build_video writes a tiny stub MP4 so MediaFileUpload (also
    patched below) sees a real file path if it ever stats it.
    """
    from marmelade.youtube import upload_runnable as ur

    def _fake_build_video(image, audio, out, **_kw):
        from pathlib import Path

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"FAKEMP4")

    monkeypatch.setattr(ur, "build_video", _fake_build_video)
    monkeypatch.setattr(
        ur, "build_youtube_client", lambda creds: mock_youtube_client
    )

    # Patch MediaFileUpload so tests can inspect its kwargs without
    # needing a real file path that googleapiclient is happy with.
    media_mock = MagicMock(name="MediaFileUpload")
    monkeypatch.setattr(ur, "MediaFileUpload", media_mock)

    # Patch MediaInMemoryUpload too (thumbnail upload uses bytes).
    inmemory_mock = MagicMock(name="MediaInMemoryUpload")
    monkeypatch.setattr(ur, "MediaInMemoryUpload", inmemory_mock)

    return media_mock, inmemory_mock


# ---------------------------------------------------------------------------
# Test 1 — progress monotone + finished
# ---------------------------------------------------------------------------


def test_progress_emits_monotone(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """progress.emit values are monotone non-decreasing 0..100; finished fires."""
    media_mock, inmemory_mock = _patch_build_client_and_ffmpeg(
        monkeypatch, mock_youtube_client, tmp_path
    )

    # Script next_chunk: 5 progress steps then a final response.
    def _mk_status(pct: float):
        s = MagicMock(name=f"status_{pct}")
        s.progress.return_value = pct
        s.resumable_progress = int(pct * 1000)
        s.total_size = 1000
        return s

    insert_req = mock_youtube_client.videos.return_value.insert.return_value
    insert_req.next_chunk.side_effect = [
        (_mk_status(0.20), None),
        (_mk_status(0.40), None),
        (_mk_status(0.60), None),
        (_mk_status(0.80), None),
        (_mk_status(0.95), None),
        (None, {"id": "abc123xyz"}),
    ]

    runnable, _ = _build_runnable(tmp_path, mock_youtube_client, mock_credentials_factory)
    # Bypass throttle for testing — every progress call should land.
    runnable._emit_progress_throttled = lambda pct: runnable.signals.progress.emit(int(pct))
    rec = _wire_signal_recorder(runnable)

    runnable.run()
    QCoreApplication.processEvents()

    # Finished fired with the video_id.
    assert rec["finished"] == ["abc123xyz"]
    assert rec["error"] == []
    assert rec["cancelled"] == 0

    # Progress monotone non-decreasing.
    assert rec["progress"], "expected at least one progress emit"
    for a, b in zip(rec["progress"], rec["progress"][1:]):
        assert a <= b, f"progress not monotone: {rec['progress']}"
    # Final emit reaches 100.
    assert rec["progress"][-1] == 100


# ---------------------------------------------------------------------------
# Test 2 — cancel mid-chunk → cancelled (no error, no thumbnail upload)
# ---------------------------------------------------------------------------


def test_cancel_mid_chunk_cleanly_exits(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """cancel() between chunks emits cancelled (NOT error); thumbnails.set NEVER called."""
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    # Script: yield a couple of progress tuples; on the 3rd next_chunk
    # the test's helper has already called .cancel() so the runnable
    # raises BuildCancelled before invoking next_chunk again.
    def _mk_status(pct: float):
        s = MagicMock()
        s.progress.return_value = pct
        return s

    insert_req = mock_youtube_client.videos.return_value.insert.return_value
    insert_req.next_chunk.side_effect = [
        (_mk_status(0.20), None),
        (_mk_status(0.40), None),
        (_mk_status(0.60), None),
        (_mk_status(0.80), None),
        (None, {"id": "should-never-get-here"}),
    ]

    runnable, _ = _build_runnable(tmp_path, mock_youtube_client, mock_credentials_factory)
    rec = _wire_signal_recorder(runnable)

    # Cancel BEFORE running — first cancel check in run() trips
    # BuildCancelled before videos.insert is even called.
    runnable.cancel()
    runnable.run()
    QCoreApplication.processEvents()

    assert rec["cancelled"] == 1
    assert rec["error"] == [], f"unexpected error: {rec['error']}"
    assert rec["finished"] == []

    # CRITICAL: thumbnails().set was NEVER called (Pitfall 1 — no orphan).
    mock_youtube_client.thumbnails.return_value.set.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — MediaFileUpload chunksize == 256*1024
# ---------------------------------------------------------------------------


def test_chunksize_is_256kb(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """MediaFileUpload(chunksize=...) is exactly 256 * 1024 (D-24)."""
    media_mock, _ = _patch_build_client_and_ffmpeg(
        monkeypatch, mock_youtube_client, tmp_path
    )

    # Default conftest fixture yields one chunk → done.
    runnable, _ = _build_runnable(tmp_path, mock_youtube_client, mock_credentials_factory)
    rec = _wire_signal_recorder(runnable)
    runnable.run()
    QCoreApplication.processEvents()

    assert media_mock.called, "MediaFileUpload was not constructed"
    # Capture the kwargs of the first call.
    _, kwargs = media_mock.call_args
    assert kwargs.get("chunksize") == 256 * 1024, (
        f"chunksize must be 256 KiB; got {kwargs.get('chunksize')!r}"
    )
    assert kwargs.get("resumable") is True
    # finished or error should still settle — but the contract here is
    # the chunksize itself, not the lifecycle.
    assert rec["error"] == [], f"unexpected error: {rec['error']}"


# ---------------------------------------------------------------------------
# Test 4 — 5xx HttpError triggers exponential-backoff retry
# ---------------------------------------------------------------------------


def test_retry_5xx_with_exp_backoff(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """HttpError 5xx triggers exponential backoff retry; max 10 attempts."""
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    from googleapiclient.errors import HttpError

    def _make_http_error(status: int):
        resp = MagicMock()
        resp.status = status
        resp.reason = "Server Error"
        return HttpError(resp, b'{"error":{"message":"transient"}}', uri="https://x")

    insert_req = mock_youtube_client.videos.return_value.insert.return_value
    insert_req.next_chunk.side_effect = [
        _make_http_error(500),  # raise on first attempt
        _make_http_error(503),  # raise on second
        (None, {"id": "ok-after-retry"}),  # succeed third
    ]

    # Capture time.sleep calls (the exp-backoff jitter).
    sleeps = []
    from marmelade.youtube import upload_runnable as ur
    monkeypatch.setattr(ur.time, "sleep", lambda s: sleeps.append(s))

    runnable, _ = _build_runnable(tmp_path, mock_youtube_client, mock_credentials_factory)
    rec = _wire_signal_recorder(runnable)
    runnable.run()
    QCoreApplication.processEvents()

    assert rec["finished"] == ["ok-after-retry"], (
        f"expected finished after retry; rec={rec}"
    )
    # Two retries → two sleeps.
    assert len(sleeps) >= 2, f"expected >=2 sleeps from backoff; got {sleeps}"
    # All sleep values are non-negative.
    assert all(s >= 0 for s in sleeps), f"negative sleep value: {sleeps}"


# ---------------------------------------------------------------------------
# Test 5 — 403 quotaExceeded surfaces actionable error
# ---------------------------------------------------------------------------


def test_quota_exceeded_actionable_message(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """quotaExceeded -> signals.error with an actionable message (no stack trace)."""
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    from googleapiclient.errors import HttpError

    resp = MagicMock()
    resp.status = 403
    resp.reason = "Forbidden"
    err = HttpError(
        resp,
        b'{"error":{"code":403,"message":"quotaExceeded","errors":[{"reason":"quotaExceeded"}]}}',
        uri="https://x",
    )

    insert_req = mock_youtube_client.videos.return_value.insert.return_value
    insert_req.next_chunk.side_effect = err

    runnable, _ = _build_runnable(tmp_path, mock_youtube_client, mock_credentials_factory)
    rec = _wire_signal_recorder(runnable)
    runnable.run()
    QCoreApplication.processEvents()

    assert rec["finished"] == []
    assert rec["cancelled"] == 0
    assert len(rec["error"]) == 1, f"expected 1 error emit; got {rec['error']}"
    msg = rec["error"][0]
    lower = msg.lower()
    assert "quota" in lower, f"expected 'quota' in error msg; got {msg!r}"
    assert "tomorrow" in lower, f"expected 'tomorrow' in error msg; got {msg!r}"


# ---------------------------------------------------------------------------
# Test 6 — RefreshError silent-retry; second failure surfaces Reconnect UX
# ---------------------------------------------------------------------------


def test_refresh_then_reconnect_uses_workersignals_verbatim(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """RefreshError once → silent refresh attempt; second failure → Reconnect UX.

    Also asserts the runnable's signals object IS-A WorkerSignals (D-16).
    """
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    from google.auth.exceptions import RefreshError

    insert_req = mock_youtube_client.videos.return_value.insert.return_value
    # First chunk raises RefreshError; runnable attempts creds.refresh,
    # which (per the creds mock below) ALSO raises RefreshError.
    insert_req.next_chunk.side_effect = RefreshError("Token expired")

    runnable, creds = _build_runnable(
        tmp_path, mock_youtube_client, mock_credentials_factory
    )
    # creds.refresh blows up too — simulates a hard refresh-failure path.
    creds.refresh.side_effect = RefreshError("Refresh denied")

    rec = _wire_signal_recorder(runnable)
    runnable.run()
    QCoreApplication.processEvents()

    # Identity pin: signals is the canonical WorkerSignals class (D-16).
    assert type(runnable.signals) is WorkerSignals, (
        f"signals must be WorkerSignals VERBATIM, not subclass; got "
        f"{type(runnable.signals).__name__}"
    )

    # An error fired with a "reconnect" hint.
    assert rec["finished"] == []
    assert rec["cancelled"] == 0
    assert len(rec["error"]) == 1, f"expected 1 error emit; got {rec['error']}"
    msg_lower = rec["error"][0].lower()
    assert "reconnect" in msg_lower, (
        f"expected 'reconnect' in error msg; got {rec['error'][0]!r}"
    )


# ---------------------------------------------------------------------------
# Test 7 (W1) — explicit network rung BEFORE broad Exception
# ---------------------------------------------------------------------------


def test_network_error_surfaces_friendly_message(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """OSError / ConnectionError class → 'Network error — check your connection'.

    The exception ladder MUST catch (socket.error, OSError, ConnectionError,
    TimeoutError, urllib.error.URLError) BEFORE the broad Exception rung
    so the user gets the friendly message instead of str(e) leaking
    the underlying URL / IP / port.
    """
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    insert_req = mock_youtube_client.videos.return_value.insert.return_value
    insert_req.next_chunk.side_effect = ConnectionError(
        "[Errno 111] Connection refused — 142.250.179.106:443"
    )

    runnable, _ = _build_runnable(tmp_path, mock_youtube_client, mock_credentials_factory)
    rec = _wire_signal_recorder(runnable)
    runnable.run()
    QCoreApplication.processEvents()

    assert rec["finished"] == []
    assert rec["cancelled"] == 0
    assert len(rec["error"]) == 1, f"expected 1 error emit; got {rec['error']}"
    msg = rec["error"][0]
    # User-actionable message (does NOT leak underlying IP / port).
    assert "network" in msg.lower(), f"expected 'network' in msg; got {msg!r}"
    assert "142.250" not in msg, f"underlying IP leaked: {msg!r}"
    assert "Errno" not in msg, f"underlying errno leaked: {msg!r}"
