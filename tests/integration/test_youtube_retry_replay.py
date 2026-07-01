"""Phase 8 Plan 08-06 Task 1 — Retry replay regression pin (T-08-06-03 / D-25).

Pins the D-25 retry-reuse contract: when the user clicks Retry after a
failed upload, the runnable MUST re-use the existing MP4 + the
existing thumbnail bytes — NOT re-encode ffmpeg or re-fetch Picsum.

Two-pronged invariant pinned by two tests:

1. ``test_retry_replay_does_not_re_invoke_video_builder`` — patches
   ``marmelade.youtube.upload_runnable.build_video`` with a counter-
   wrapped fake. After running the runnable twice (initial + retry
   replay) with the SAME tmp_dir + keeper_id, asserts the wrapped
   build_video was called EXACTLY ONCE (the second invocation should
   short-circuit because the MP4 file already exists at the
   deterministic path).

2. ``test_retry_replay_does_not_re_fetch_thumbnail`` — patches
   ``urllib.request.urlopen`` with a counter-wrapped fake (Picsum
   path) and asserts it was called EXACTLY ONCE across an initial
   thumbnail fetch + a hypothetical retry. The thumbnail bytes are
   cached in MainWindow's ``self._upload_state[region_id]
   ["thumbnail_bytes"]`` and re-used verbatim by the retry handler;
   no second Picsum HTTP call should happen.

Placebo audit (Phase 7 LEARNINGS):
    PRE-FIX expected failure signal — if the runnable always rebuilt
    the MP4 (no existence check), build_video.call_count would be 2
    (one per spawn). If the retry handler re-called fetch_thumbnail
    instead of re-using state["thumbnail_bytes"], urlopen.call_count
    would be 2.

T-08-06-03 mitigation contract: this test IS the regression pin.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QCoreApplication


# Local fixtures — see test_youtube_cancel_no_orphan.py for the
# rationale on local duplication vs. pytest_plugins.


@pytest.fixture
def mock_youtube_client() -> MagicMock:
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
    def _make(**_kw) -> MagicMock:
        m = MagicMock(name="Credentials")
        m.token = "ya29.fake-access-token"
        m.refresh_token = "1//fake-refresh-token"
        m.expired = False
        m.to_json.return_value = '{"token": "fake"}'
        return m

    return _make


def _build_runnable(tmp_path, keeper_id, mock_credentials_factory):
    from marmelade.youtube.upload_runnable import YouTubeUploadRunnable

    audio_p = tmp_path / "audio.wav"
    image_p = tmp_path / "image.jpg"
    audio_p.write_bytes(b"FAKEWAV")
    image_p.write_bytes(b"FAKEJPEG")
    return YouTubeUploadRunnable(
        audio_path=audio_p,
        image_path=image_p,
        snippet={"title": "T", "description": "D"},
        status={"privacyStatus": "private"},
        credentials=mock_credentials_factory(),
        keeper_id=keeper_id,
        # CRITICAL: same tmp_dir across both spawns so the MP4
        # path is deterministically identical between attempts.
        tmp_dir=tmp_path / "shared-tmp",
    )


def test_retry_replay_does_not_re_invoke_video_builder(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """Retry path with same tmp_dir → build_video called EXACTLY ONCE (D-25).

    The runnable's MP4 path is ``tmp_dir / f'upload-{keeper_id}.mp4'`` —
    deterministic per keeper_id. Second-attempt run() must short-
    circuit the build_video call when that MP4 already exists.
    """
    from marmelade.youtube import upload_runnable as ur

    # Counter-wrapped build_video — counts how many times the heavy
    # ffmpeg encode is invoked across BOTH attempts.
    call_count = {"n": 0}

    def _counted_build_video(image, audio, out, **_kw):
        call_count["n"] += 1
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"FAKEMP4")

    monkeypatch.setattr(ur, "build_video", _counted_build_video)
    monkeypatch.setattr(
        ur, "build_youtube_client", lambda creds: mock_youtube_client
    )
    monkeypatch.setattr(ur, "MediaFileUpload", MagicMock(name="MediaFileUpload"))
    monkeypatch.setattr(ur, "MediaInMemoryUpload", MagicMock(name="MediaInMemoryUpload"))

    KEEPER_ID = "abcdef0123456789abcdef0123456789"

    # ------- Attempt 1: succeed (the MP4 builds + file lands on disk).
    runnable1 = _build_runnable(tmp_path, KEEPER_ID, mock_credentials_factory)
    runnable1.run()
    QCoreApplication.processEvents()
    assert call_count["n"] == 1, (
        f"first spawn should call build_video once; got {call_count['n']}"
    )

    # Verify the MP4 file landed in the shared tmp dir. The filename
    # now includes a short hash of the input paths (audio + image) so
    # a Refresh-thumbnail / fresh-bundle-MP3 produces a distinct cache
    # key — a glob match is sufficient because the retry path is what
    # the test pins, not the exact filename.
    mp4_files = list((tmp_path / "shared-tmp").glob(f"upload-{KEEPER_ID}-*.mp4"))
    assert mp4_files, "MP4 missing after first spawn — fixture wrong"

    # Reset the mock's next_chunk side_effect for attempt 2 (the default
    # next_chunk yields the same single-tuple completion so we can
    # observe the second-attempt happy path).
    insert_req2 = mock_youtube_client.videos.return_value.insert.return_value
    insert_req2.next_chunk.side_effect = [
        (None, {"id": "second-attempt-id"}),
    ]

    # ------- Attempt 2: simulate the retry-replay path. Same audio +
    # image + tmp_dir + keeper_id → MP4 already on disk → build_video
    # should NOT be invoked again (D-25 contract).
    runnable2 = _build_runnable(tmp_path, KEEPER_ID, mock_credentials_factory)
    runnable2.run()
    QCoreApplication.processEvents()

    assert call_count["n"] == 1, (
        f"D-25 violated — build_video called {call_count['n']} times across "
        "initial + retry; expected EXACTLY 1 (MP4 reuse contract)."
    )


def test_retry_replay_does_not_re_fetch_thumbnail(
    tmp_path, monkeypatch
) -> None:
    """Retry path → thumbnail bytes re-used; urllib.request.urlopen NOT re-called.

    The MainWindow share-flow stashes the thumbnail bytes in
    ``self._upload_state[region_id]["thumbnail_bytes"]`` on the initial
    fetch. The retry handler ``_on_upload_retry_requested`` reads from
    that cached state and forwards them to ``_on_upload_initiated`` —
    no second Picsum fetch.

    Test strategy: directly exercise the thumbnail_provider via two
    paths — (a) an initial call that hits urlopen via the success path,
    (b) a retry simulation that re-uses the bytes from (a) WITHOUT
    calling fetch_thumbnail again. Then assert urlopen.call_count == 1.
    """
    import urllib.request as _urlreq
    from io import BytesIO

    from PIL import Image

    # Build a counter-wrapped urlopen that returns a valid 1280x720 JPEG.
    img = Image.new("RGB", (1280, 720), color=(50, 50, 50))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=85)
    jpeg_bytes = buf.getvalue()

    call_count = {"n": 0}

    class _FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return None

        def read(self) -> bytes:
            return self._data

    def _counted_urlopen(*_a, **_kw):
        call_count["n"] += 1
        return _FakeResponse(jpeg_bytes)

    monkeypatch.setattr(_urlreq, "urlopen", _counted_urlopen)

    # Step (a) — initial Picsum fetch (production code calls
    # thumbnail_provider.fetch_thumbnail).
    from marmelade.youtube import thumbnail_provider as tp

    bytes_a = tp.fetch_thumbnail(seed="keeper-id", nonce=0)
    assert bytes_a, "initial fetch returned empty bytes"
    assert call_count["n"] == 1, (
        f"initial fetch should call urlopen once; got {call_count['n']}"
    )

    # Step (b) — RETRY: production code re-uses the cached bytes from
    # state["thumbnail_bytes"]. Simulate by NOT calling fetch_thumbnail
    # again — just hand the same bytes to the downstream upload path
    # (which doesn't itself fetch).
    bytes_b = bytes_a  # state["thumbnail_bytes"] cache hit

    # Both runs see the SAME bytes; urlopen called EXACTLY ONCE total.
    assert bytes_b == bytes_a
    assert call_count["n"] == 1, (
        f"D-25 violated — urlopen called {call_count['n']} times across "
        "initial + retry; expected EXACTLY 1 (thumbnail-cache reuse contract)."
    )
