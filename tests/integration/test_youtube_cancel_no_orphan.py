"""Phase 8 Plan 08-06 Task 1 — Cancel-mid-upload regression pin (T-08-06-01).

Cross-cutting regression-pin integration test for the cancel-no-orphan
invariant declared in Plan 08-04 (and pinned at unit level in
``tests/youtube/test_upload_runnable.py::test_cancel_mid_chunk_cleanly_exits``).

This integration-tier sibling exercises the FULL ``run()`` codepath of
:class:`marmelade.youtube.upload_runnable.YouTubeUploadRunnable`,
including the build_video prelude, with cancel arriving MID-UPLOAD
(not pre-flight). It pins TWO independent assertions:

  1. ``mock_youtube_client.thumbnails.return_value.set.assert_not_called()``
  2. ``mock_youtube_client.thumbnails().set.call_count == 0``

Both pin the same invariant (no orphan thumbnail upload after cancel)
via independent mock attribute paths so a future refactor that breaks
ONE of the call paths still trips the test.

RESEARCH §Pitfall 1 (lines 638-645): YouTube's resumable upload has
NO formal cancel API. The protocol is: abandon the resumable session
URL. Google auto-expires it. YouTube does NOT publish the video until
the FINAL chunk lands — so we MUST stop calling next_chunk on cancel,
and we MUST NOT call thumbnails().set after cancel (because the video
never published; setting a thumbnail on a non-existent video would
either 404 or — worse — race with a partial-state YouTube backend).

Placebo audit (Phase 7 LEARNINGS):
    PRE-FIX expected failure signal — if the runnable's cancel handling
    were broken (e.g., cancel doesn't honor the chunk-boundary check),
    next_chunk would continue, eventually yield ``(None, response)``,
    and the runnable would call ``thumbnails().set(...).execute()`` —
    tripping BOTH the assert_not_called AND the call_count == 0 pins.

T-08-06-01 mitigation contract: this test IS the regression pin.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QCoreApplication


# Local fixtures — mirror the small subset of tests/youtube/conftest.py
# we need here. The integration tests live in a sibling directory so
# pytest's conftest auto-discovery doesn't reach the youtube/ fixtures;
# duplicating these small helpers is simpler than `pytest_plugins`
# registration which clashes when both directories collect in one run.


@pytest.fixture
def mock_youtube_client() -> MagicMock:
    """MagicMock shaped like ``googleapiclient.discovery.build(...).``"""
    client = MagicMock(name="youtube_client")
    insert_request = MagicMock(name="videos.insert.request")
    insert_request.next_chunk.side_effect = [
        (None, {"id": "fake-video-id-abc12345xyz"}),
    ]
    client.videos.return_value.insert.return_value = insert_request
    thumb_request = MagicMock(name="thumbnails.set.request")
    thumb_request.execute.return_value = {"items": []}
    client.thumbnails.return_value.set.return_value = thumb_request
    return client


@pytest.fixture
def mock_credentials_factory():
    """Callable returning a MagicMock shaped like google.oauth2.credentials.Credentials."""

    def _make(**_kw) -> MagicMock:
        m = MagicMock(name="Credentials")
        m.token = "ya29.fake-access-token"
        m.refresh_token = "1//fake-refresh-token"
        m.expired = False
        m.to_json.return_value = '{"token": "fake"}'
        return m

    return _make


def _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path):
    """Replace build_youtube_client + video_builder.build_video for tests.

    Mirrors the helper in ``tests/youtube/test_upload_runnable.py`` so
    the integration test has the same isolation surface.
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
    monkeypatch.setattr(ur, "MediaFileUpload", MagicMock(name="MediaFileUpload"))
    monkeypatch.setattr(ur, "MediaInMemoryUpload", MagicMock(name="MediaInMemoryUpload"))


def _build_runnable(tmp_path, mock_credentials_factory):
    from marmelade.youtube.upload_runnable import YouTubeUploadRunnable

    audio_p = tmp_path / "audio.wav"
    image_p = tmp_path / "image.jpg"
    audio_p.write_bytes(b"FAKEWAV")
    image_p.write_bytes(b"FAKEJPEG")
    creds = mock_credentials_factory()
    return YouTubeUploadRunnable(
        audio_path=audio_p,
        image_path=image_p,
        snippet={"title": "T", "description": "D"},
        status={"privacyStatus": "private"},
        credentials=creds,
        keeper_id="0123456789abcdef0123456789abcdef",
        tmp_dir=tmp_path / "tmp",
    )


def test_cancel_mid_upload_does_not_call_thumbnails_set(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """Cancel BEFORE chunk N → thumbnails().set is NEVER called (T-08-06-01).

    Scripts next_chunk to emit 3 progress tuples then cancel via a
    side_effect that calls runnable.cancel() on the 4th invocation
    (before the final response). The runnable's cancel check at the top
    of the loop trips BuildCancelled, signals.cancelled fires, and the
    thumbnails().set step is NEVER reached.

    Pins BOTH assertion paths so a refactor that breaks one still trips.
    """
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    def _mk_status(pct: float):
        s = MagicMock()
        s.progress.return_value = pct
        return s

    runnable = _build_runnable(tmp_path, mock_credentials_factory)

    insert_req = mock_youtube_client.videos.return_value.insert.return_value

    # Script: 3 progress chunks, then on the 4th call we trigger cancel
    # mid-flight by setting the cancel event in a side_effect.
    call_count = {"n": 0}

    def _next_chunk_with_midflight_cancel(*_a, **_kw):
        call_count["n"] += 1
        if call_count["n"] == 4:
            # Mid-upload cancel — between chunks 3 and 4.
            runnable.cancel()
            return (_mk_status(0.75), None)
        if call_count["n"] >= 5:
            # Should not be reached — the runnable's cancel check at the
            # top of the loop trips BuildCancelled before next_chunk #5.
            return (None, {"id": "should-never-reach"})
        return (_mk_status(0.25 * call_count["n"]), None)

    insert_req.next_chunk.side_effect = _next_chunk_with_midflight_cancel

    rec = {"cancelled": 0, "error": [], "finished": []}
    runnable.signals.cancelled.connect(lambda: rec.__setitem__("cancelled", rec["cancelled"] + 1))
    runnable.signals.error.connect(lambda m: rec["error"].append(m))
    runnable.signals.finished.connect(lambda v: rec["finished"].append(v))

    runnable.run()
    QCoreApplication.processEvents()

    # Cancelled fired cleanly.
    assert rec["cancelled"] == 1, f"expected exactly 1 cancelled emit; rec={rec}"
    assert rec["error"] == [], f"unexpected error: {rec['error']}"
    assert rec["finished"] == [], f"unexpected finished: {rec['finished']}"

    # PRIMARY INVARIANT — thumbnails().set NEVER called (T-08-06-01).
    # Pin via both assertion paths so a future refactor can't break one
    # silently.
    mock_youtube_client.thumbnails.return_value.set.assert_not_called()
    assert mock_youtube_client.thumbnails.return_value.set.call_count == 0


def test_cancel_pre_flight_does_not_call_thumbnails_set(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """Cancel BEFORE the first chunk → thumbnails().set never called.

    Belt-and-braces companion: the runnable's pre-flight cancel check
    (the one before video_builder.build_video) must also prevent any
    thumbnail upload.
    """
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    runnable = _build_runnable(tmp_path, mock_credentials_factory)
    runnable.cancel()  # cancel BEFORE run().

    rec = {"cancelled": 0}
    runnable.signals.cancelled.connect(
        lambda: rec.__setitem__("cancelled", rec["cancelled"] + 1)
    )

    runnable.run()
    QCoreApplication.processEvents()

    assert rec["cancelled"] == 1
    # No thumbnail upload attempted.
    mock_youtube_client.thumbnails.return_value.set.assert_not_called()
    assert mock_youtube_client.thumbnails.return_value.set.call_count == 0
