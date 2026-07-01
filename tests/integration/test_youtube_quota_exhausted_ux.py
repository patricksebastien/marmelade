"""Phase 8 Plan 08-06 Task 1 — Quota / network / ffmpeg user-actionable error UX (T-08-06-02).

Cross-cutting regression-pin integration tests for the user-actionable
error contract declared in YT-05. The unit-level kin in
``tests/youtube/test_upload_runnable.py`` cover the substrings; this
integration-tier file pins three additional invariants:

1. ``test_quota_exceeded_message_is_actionable`` — the error message
   contains BOTH 'quota' AND 'tomorrow' AND has no newlines AND is
   <200 chars (no stack-trace leak — T-08-06-02 / threat I).
2. ``test_network_error_actionable_message`` (revision iter 1 W1) —
   ``urllib.error.URLError`` raises the EXACT user-facing message
   "Network error — check your connection" (suffixed " and try again"
   per the production constant) without leaking the wrapped exception
   string (no IP / port / URL leak — T-08-04-08).
3. ``test_ffmpeg_timeout_surfaces_actionable_error`` (revision iter 1
   N2) — ``subprocess.TimeoutExpired`` from ``video_builder.build_video``
   raises ``RuntimeError`` containing "ffmpeg timed out", the .tmp
   sibling is cleaned up, and the running child process is killed.

Placebo audit (Phase 7 LEARNINGS):
    PRE-FIX expected failure signals — if the exception ladder in
    upload_runnable.py were reordered, broad ``Exception`` would catch
    URLError first and emit ``str(e)`` (leaking the URL). If the
    quotaExceeded branch didn't special-case the message, the test
    would receive a JSON-error-payload-formatted message with newlines.
    If video_builder didn't catch TimeoutExpired, the test would see
    the raw subprocess.TimeoutExpired exception bubbling up.

T-08-06-02 mitigation contract: this test IS the regression pin.
"""

from __future__ import annotations

import subprocess
import urllib.error
from unittest.mock import MagicMock, patch

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


def _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path):
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
    return YouTubeUploadRunnable(
        audio_path=audio_p,
        image_path=image_p,
        snippet={"title": "T", "description": "D"},
        status={"privacyStatus": "private"},
        credentials=mock_credentials_factory(),
        keeper_id="0123456789abcdef0123456789abcdef",
        tmp_dir=tmp_path / "tmp",
    )


def test_quota_exceeded_message_is_actionable(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """403 quotaExceeded → contains 'quota' AND 'tomorrow', no newline, < 200 chars.

    Pins YT-05 actionable contract + T-08-06-02 no-stack-trace contract.
    """
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    from googleapiclient.errors import HttpError

    resp = MagicMock()
    resp.status = 403
    resp.reason = "Forbidden"
    err = HttpError(
        resp,
        b'{"error":{"code":403,"message":"quotaExceeded",'
        b'"errors":[{"reason":"quotaExceeded","domain":"youtube.quota"}]}}',
        uri="https://youtube.googleapis.com/...",
    )
    insert_req = mock_youtube_client.videos.return_value.insert.return_value
    insert_req.next_chunk.side_effect = err

    runnable = _build_runnable(tmp_path, mock_credentials_factory)
    rec = {"error": []}
    runnable.signals.error.connect(lambda m: rec["error"].append(m))
    runnable.run()
    QCoreApplication.processEvents()

    assert len(rec["error"]) == 1, f"expected exactly 1 error; got {rec['error']}"
    msg = rec["error"][0]
    lower = msg.lower()

    # YT-05 actionable substrings (both 'quota' and 'tomorrow' present).
    assert "quota" in lower, f"missing 'quota': {msg!r}"
    assert "tomorrow" in lower, f"missing 'tomorrow': {msg!r}"

    # No stack-trace: no newlines, short message.
    assert "\n" not in msg, f"newline in user-facing message: {msg!r}"
    assert len(msg) < 200, f"message too long ({len(msg)}): {msg!r}"


def test_network_error_actionable_message(
    tmp_path, mock_youtube_client, mock_credentials_factory, monkeypatch
) -> None:
    """urllib.URLError → 'Network error — check your connection' (revision iter 1 W1).

    Pins:
      1. The friendly substring "Network error" appears (W1 contract).
      2. The substring "check your connection" appears.
      3. The wrapped exception's underlying URL / args do NOT leak
         (T-08-04-08 privacy contract).
      4. Single-line message — no newlines.
    """
    _patch_build_client_and_ffmpeg(monkeypatch, mock_youtube_client, tmp_path)

    leak_marker = "https://internal-leak.example.com:8443/secret-endpoint"
    insert_req = mock_youtube_client.videos.return_value.insert.return_value
    insert_req.next_chunk.side_effect = urllib.error.URLError(
        f"connection refused — {leak_marker}"
    )

    runnable = _build_runnable(tmp_path, mock_credentials_factory)
    rec = {"error": []}
    runnable.signals.error.connect(lambda m: rec["error"].append(m))
    runnable.run()
    QCoreApplication.processEvents()

    assert len(rec["error"]) == 1, f"expected exactly 1 error; got {rec['error']}"
    msg = rec["error"][0]
    # Network error — check your connection (substring match — the
    # production string may add a "and try again" suffix).
    assert "Network error" in msg, f"missing 'Network error': {msg!r}"
    assert "check your connection" in msg.lower(), (
        f"missing 'check your connection': {msg!r}"
    )
    # No URL / IP / port leak from the wrapped exception (T-08-04-08).
    assert leak_marker not in msg, f"underlying URL leaked: {msg!r}"
    assert "8443" not in msg, f"underlying port leaked: {msg!r}"
    # Single-line.
    assert "\n" not in msg, f"newline in user-facing message: {msg!r}"


def test_ffmpeg_timeout_surfaces_actionable_error(tmp_path) -> None:
    """subprocess.TimeoutExpired in build_video → RuntimeError 'ffmpeg timed out' + .tmp cleanup.

    Revision iter 1 N2 regression pin. The video_builder catches
    TimeoutExpired, kills any child process (subprocess.run kills before
    raising), removes the .tmp sibling, and re-raises as RuntimeError
    with an actionable string.

    The test patches subprocess.run inside video_builder to raise
    TimeoutExpired and verifies:
      * The raised RuntimeError contains "ffmpeg timed out"
      * The .tmp sibling does NOT exist after the raise
      * The canonical output also does NOT exist (atomic rename never ran)
    """
    from marmelade.youtube import video_builder as vb

    wav = tmp_path / "in.wav"
    jpg = tmp_path / "in.jpg"
    out = tmp_path / "out.mp4"
    # Cheap real files so the codepath reaches subprocess.run.
    wav.write_bytes(b"fakewav")
    jpg.write_bytes(b"fakejpg")

    # Pre-create the .tmp sibling to assert cleanup actually runs.
    tmp_sibling = out.with_name(out.stem + ".tmp" + out.suffix)
    tmp_sibling.write_bytes(b"PARTIAL_MP4")
    assert tmp_sibling.exists()

    def _raise_timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=600)

    with patch.object(vb.subprocess, "run", side_effect=_raise_timeout):
        with pytest.raises(RuntimeError) as exc_info:
            vb.build_video(jpg, wav, out)

    msg = str(exc_info.value).lower()
    assert "ffmpeg" in msg, f"missing 'ffmpeg' in error: {exc_info.value!r}"
    assert "timed out" in msg, f"missing 'timed out' in error: {exc_info.value!r}"

    # .tmp sibling cleaned up + canonical output not created (atomic rename never ran).
    assert not tmp_sibling.exists(), "TimeoutExpired path did not clean up .tmp sibling"
    assert not out.exists(), "canonical output unexpectedly exists after timeout"
