"""``YouTubeUploadRunnable`` — resumable upload + 3-layer cancel (Phase 8 YT-04 + YT-05).

1:1 structural mirror of
:class:`marmelade.audio.mastering_worker.MasteringRunnable` per D-28
(LOCKED — do NOT renegotiate the worker shape). REUSES
:class:`marmelade.concurrency.worker.WorkerSignals` VERBATIM per
D-16 — the test ``test_refresh_then_reconnect_uses_workersignals_verbatim``
pins identity via ``type(runnable.signals) is WorkerSignals``.

Qt-isolation (D-27): imports ``PySide6.QtCore`` for ``QRunnable`` and
``Slot`` ONLY. No widget-tier or GUI-tier PySide6 modules are imported.
The plan's source-grep gate pins zero import lines for those modules.

Pipeline (:meth:`run`):

1. Build MP4 via
   :func:`marmelade.youtube.video_builder.build_video` — synchronous
   prelude. Emits ``progress(5)`` after the build completes so the user
   sees motion before the upload's first chunk lands.
2. Resumable upload via
   :class:`googleapiclient.http.MediaFileUpload(chunksize=256*1024,
   resumable=True)` + ``request.next_chunk()`` loop per RESEARCH §Pattern
   2. ``self._is_cancelled()`` polled BEFORE each ``next_chunk()`` per
   RESEARCH §Pitfall 1.
3. Thumbnail upload via ``client.thumbnails().set(videoId=video_id,
   media_body=MediaInMemoryUpload(...))`` AFTER ``videos().insert()``
   returns the video ID. Best-effort — a thumbnail-upload failure does
   NOT fail the whole upload (the video is already published; the user
   can re-upload the thumbnail later via YouTube Studio).

Terminal-signal contract: exactly ONE of ``finished(video_id)`` /
``error(message)`` / ``cancelled()`` fires per :meth:`run` call.

Exception ladder (top-to-bottom, ORDER IS LOAD-BEARING — revision iter
1 W1 inserts the explicit network rung):

* :class:`BuildCancelled` — emit ``cancelled``. MUST be first because
  :class:`BuildCancelled` derives from :class:`RuntimeError` and would
  be swallowed by the broad ``Exception`` rung otherwise.
* :class:`googleapiclient.errors.HttpError`:
    * 5xx-retryable (500/502/503/504) → exponential backoff retry up
      to ``_MAX_RETRIES`` (10).
    * 403 ``quotaExceeded`` → emit
      ``"Daily YouTube upload quota exceeded — try again tomorrow."``
      (YT-05; surfaces in the dialog footer).
    * Other ``HttpError`` → emit ``str(e)`` as fallback.
* :class:`google.auth.exceptions.RefreshError` — attempt ONE silent
  ``creds.refresh(Request())`` then retry the upload from the start;
  if the refresh ALSO raises ``RefreshError`` emit
  ``"Reconnect YouTube — your authorization has expired."`` (D-25).
* ``(socket.error, OSError, ConnectionError, TimeoutError,
  urllib.error.URLError)`` — explicit network rung (revision iter 1
  W1). Emit ``"Network error — check your connection"``. The user-
  actionable message MUST NOT include ``str(e)`` so no IP/port/URL
  leaks into the dialog footer (Threat T-08-04-08).
* Broad ``Exception`` — fallback. Emit ``str(e) or type(e).__name__``.

Cancel discipline (D-24): :meth:`cancel` sets a :class:`threading.Event`
that :meth:`_is_cancelled` polls. Idempotent; safe to call multiple
times. Honored at chunk boundaries — the 256 KiB chunk size keeps the
cancel-acknowledge latency at roughly 1-2 seconds even on a slow uplink.

Credentials logging discipline (T-08-04-08 — pinned by the plan's
``grep -nE 'log\\..*creds'`` gate): credentials are NEVER logged. The
exception handlers emit ``str(e)`` only; ``self._credentials`` is never
serialized via ``log.*`` calls.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import socket
import threading
import time
import urllib.error
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaInMemoryUpload
from PySide6.QtCore import QRunnable, Slot

from marmelade.audio.peak_builder import BuildCancelled
from marmelade.concurrency.worker import WorkerSignals
from marmelade.youtube.client import build_youtube_client
from marmelade.youtube.video_builder import build_video


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants — frozen contract pinned by source-grep gates.
# ---------------------------------------------------------------------------

# D-24: 256 KiB chunks. Pinned by `test_chunksize_is_256kb` AND by the
# plan's acceptance grep on this exact literal expression.
_CHUNK_SIZE: int = 256 * 1024

# Retry budget for 5xx HttpError exponential backoff (RESEARCH §Pattern 2).
_MAX_RETRIES: int = 10

# 5xx statuses that warrant exponential-backoff retry. 4xx are surfaced
# immediately (403 quotaExceeded gets the special-cased actionable
# message; everything else falls through to str(e)).
_RETRYABLE_STATUSES: tuple[int, ...] = (500, 502, 503, 504)

# User-facing strings — pinned by tests so a future copy tweak fails fast.
_MSG_QUOTA_EXCEEDED: str = (
    "Daily YouTube upload quota exceeded — try again tomorrow."
)
_MSG_RECONNECT: str = (
    "Reconnect YouTube — your authorization has expired."
)
_MSG_NETWORK: str = (
    "Network error — check your connection and try again."
)


class YouTubeUploadRunnable(QRunnable):
    """Render an MP4 + upload to YouTube + set the thumbnail, on a pool thread.

    Constructor (keyword-only after ``credentials``):
        audio_path: WAV/MP3/FLAC source piped into ffmpeg as the audio
            stream (typically a mastered cache WAV or a tmp WAV from
            the source-proxy fallback path).
        image_path: JPEG/PNG used as the looped video frame AND as the
            YouTube thumbnail (the bytes are read once into memory for
            the ``thumbnails().set`` call after ``videos().insert``).
        snippet: ``{"title", "description", "tags"?}`` dict passed
            verbatim to ``videos.insert(body={"snippet": ...})``.
        status: ``{"privacyStatus": "private"|"unlisted"|"public"}``
            dict passed verbatim to ``videos.insert(body={"status":
            ...})``.
        credentials: Authenticated :class:`google.oauth2.credentials.
            Credentials` from :func:`marmelade.youtube.oauth.
            load_or_refresh`.
        keeper_id: 32-hex keeper UUID — stored on the runnable so the
            MainWindow slot can look up the right Region for sidecar
            persistence of ``youtube_video_id`` (D-30).
        tmp_dir: Scratch directory for the intermediate MP4. Caller
            (typically MainWindow) is responsible for cleanup; this
            runnable does NOT remove the MP4 on success because the
            Retry path may need to re-upload it (D-25).
    """

    def __init__(
        self,
        *,
        audio_path: str | os.PathLike,
        image_path: str | os.PathLike,
        snippet: dict,
        status: dict,
        credentials,
        keeper_id: str,
        tmp_dir: str | os.PathLike,
        tail_pad_sec: float | None = None,
    ) -> None:
        super().__init__()
        # CR-02 — Python refcount owns lifetime, not QThreadPool's
        # autoDelete. Same rationale as MasteringRunnable: MainWindow's
        # dict reference keeps the WorkerSignals QObject valid for late
        # ``_disconnect_*_tokens`` calls in the 3-layer cancel pattern.
        self.setAutoDelete(False)

        self._audio_path: Path = Path(audio_path)
        self._image_path: Path = Path(image_path)
        self._snippet: dict = dict(snippet)
        self._status: dict = dict(status)
        self._credentials = credentials
        self.keeper_id: str = keeper_id
        self._tmp_dir: Path = Path(tmp_dir)
        # Bundle uploads pass ``0`` here because the bundle MP3
        # already ends in a real per-keeper fade-out (applied in
        # bundle_builder). Per-keeper uploads leave this ``None`` so
        # video_builder falls back to its default 1.5 s breathing room.
        self._tail_pad_sec: float | None = (
            None if tail_pad_sec is None else float(tail_pad_sec)
        )

        # REUSE VERBATIM — DO NOT subclass per D-16. The 4-signal
        # contract (progress/finished/error/cancelled) is the
        # cross-worker invariant. Test
        # ``test_refresh_then_reconnect_uses_workersignals_verbatim``
        # pins ``type(self.signals) is WorkerSignals``.
        self.signals: WorkerSignals = WorkerSignals()
        self._cancel_event: threading.Event = threading.Event()

        # Throttle bookkeeping (mirrors mastering_worker.py lines 131-132).
        self._last_progress_emit_ts: float = 0.0
        self._last_progress_pct: int = -1

        # Cached image bytes — read once before the upload starts so a
        # mid-upload disk failure does not eat the runnable.
        self._image_bytes: bytes | None = None

    # ----- public API mirroring MasteringRunnable -----

    def cancel(self) -> None:
        """Request cooperative cancellation. Idempotent."""
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        """Bridge cancel-check used by video_builder + the upload loop."""
        return self._cancel_event.is_set()

    # ----- internal helpers -----

    def _emit_progress_throttled(self, pct: int) -> None:
        """Forward ``pct`` to ``self.signals.progress`` with rate limiting.

        Rule: always emit ``0`` and ``100``; otherwise drop emits that
        come within 100 ms of the previous emit. Mirrors
        :class:`MasteringRunnable._emit_progress_throttled` verbatim.
        """
        pct = int(pct)
        if pct == self._last_progress_pct:
            return
        now = time.monotonic()
        is_bookend = pct in (0, 100)
        if not is_bookend and (now - self._last_progress_emit_ts) < 0.1:
            return
        self._last_progress_emit_ts = now
        self._last_progress_pct = pct
        self.signals.progress.emit(pct)

    # ----- worker entry point -----

    @Slot()
    def run(self) -> None:  # noqa: C901 — flat try/except per mastering_worker.py
        """Worker entry point. Exactly one terminal signal fires per call.

        Exception ordering is load-bearing — see module docstring's
        exception-ladder block.
        """
        try:
            # Early cancel check before any I/O.
            if self._is_cancelled():
                raise BuildCancelled()

            # (1) Build MP4 via ffmpeg subprocess (synchronous prelude).
            # D-25 retry-reuse: the MP4 path is deterministic per
            # keeper_id, and the inputs (audio + image) are the same on
            # the retry path (D-25 contract — Retry re-uses the same
            # audio + thumbnail). Skip the expensive ffmpeg re-encode
            # when the MP4 already exists from a prior attempt — this
            # is the contract Plan 08-06 Task 1 regression-pins via
            # ``test_youtube_retry_replay.py``. The cancel/cleanup paths
            # below NEVER unlink the MP4 on error so the file survives
            # for retry; only success ``_cleanup_upload_state`` removes
            # it (the file lives in a per-process tmp_dir so the
            # filesystem reclaims it on shutdown either way).
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
            # Cache key includes the input paths so a Refresh-thumbnail
            # (different image_path) OR a freshly-built bundle MP3
            # (different audio_path tempfile each invocation) produces
            # a distinct cached MP4 — and a true D-25 retry of the same
            # inputs still hits the cache and skips re-encoding.
            # Without this, a bundle re-upload would reuse the OLD MP4
            # (cached by keeper_id="bundle") and the video would show
            # the previous thumbnail as its poster while
            # ``thumbnails().set`` correctly updates the YouTube
            # thumbnail — exactly the symptom the user reported.
            input_key = hashlib.sha1(
                f"{self._audio_path}|{self._image_path}".encode("utf-8")
            ).hexdigest()[:12]
            mp4_path = self._tmp_dir / f"upload-{self.keeper_id}-{input_key}.mp4"
            if not mp4_path.exists() or mp4_path.stat().st_size == 0:
                build_video(
                    self._image_path,
                    self._audio_path,
                    mp4_path,
                    cancel_check=self._is_cancelled,
                    tail_pad_sec=self._tail_pad_sec,
                )
            # First visible motion — encode done (or reused), upload
            # about to start.
            self._emit_progress_throttled(5)

            if self._is_cancelled():
                raise BuildCancelled()

            # (2) Cache thumbnail bytes for the after-insert thumbnails.set
            # call. Read once so a mid-upload disk failure doesn't eat
            # the runnable.
            self._image_bytes = Path(self._image_path).read_bytes()

            # (3) Build the YouTube client + the resumable insert request.
            client = build_youtube_client(self._credentials)
            media = MediaFileUpload(
                str(mp4_path),
                chunksize=_CHUNK_SIZE,
                resumable=True,
                mimetype="video/mp4",
            )
            request = client.videos().insert(
                part="snippet,status",
                body={"snippet": self._snippet, "status": self._status},
                media_body=media,
            )

            # (4) Resumable upload loop. cancel check BEFORE each chunk
            # so the cancel-acknowledge latency is bounded by ONE chunk
            # (~1-2 s at 256 KiB / typical uplink). Pitfall 1: no
            # formal cancel API — abandoning the session is fine because
            # YouTube doesn't publish until the FINAL chunk lands.
            response = None
            retry = 0
            tried_refresh = False
            while response is None:
                if self._is_cancelled():
                    raise BuildCancelled()
                try:
                    status_obj, response = request.next_chunk(num_retries=0)
                    if status_obj is not None:
                        self._emit_progress_throttled(
                            int(status_obj.progress() * 100)
                        )
                except HttpError as e:
                    if (
                        e.resp.status in _RETRYABLE_STATUSES
                        and retry < _MAX_RETRIES
                    ):
                        retry += 1
                        # Exponential backoff with jitter.
                        time.sleep(random.random() * (2 ** retry))
                        continue
                    if e.resp.status == 403 and _is_quota_exceeded(e):
                        self.signals.error.emit(_MSG_QUOTA_EXCEEDED)
                        return
                    # Other HttpError — surface as str(e). Re-raise into
                    # the outer try so the broad fallback ladder runs.
                    raise
                except RefreshError:
                    # D-25: attempt ONE silent re-auth via creds.refresh.
                    # If that also fails, surface Reconnect UX.
                    if tried_refresh:
                        self.signals.error.emit(_MSG_RECONNECT)
                        return
                    tried_refresh = True
                    try:
                        self._credentials.refresh(Request())
                    except RefreshError:
                        self.signals.error.emit(_MSG_RECONNECT)
                        return
                    except Exception:
                        # Any non-RefreshError surfaced by refresh()
                        # collapses to the same Reconnect UX (a partial
                        # refresh failure is operationally equivalent).
                        self.signals.error.emit(_MSG_RECONNECT)
                        return
                    # Rebuild the request with the refreshed credentials
                    # and try again.
                    client = build_youtube_client(self._credentials)
                    media = MediaFileUpload(
                        str(mp4_path),
                        chunksize=_CHUNK_SIZE,
                        resumable=True,
                        mimetype="video/mp4",
                    )
                    request = client.videos().insert(
                        part="snippet,status",
                        body={
                            "snippet": self._snippet,
                            "status": self._status,
                        },
                        media_body=media,
                    )
                    continue

            video_id = str(response["id"])

            # (5) Best-effort thumbnail upload. A failure here does NOT
            # fail the whole upload — the video is already published.
            try:
                client.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaInMemoryUpload(
                        self._image_bytes,
                        mimetype="image/jpeg",
                    ),
                ).execute()
            except Exception as thumb_err:
                # Swallow + log. NEVER log credentials (T-08-04-08).
                log.warning(
                    "Thumbnail upload failed for video %s: %s",
                    video_id,
                    thumb_err,
                )

            # Bookend emit: upload complete.
            self._last_progress_pct = -1  # force the 100% past the throttle
            self._emit_progress_throttled(100)
            self.signals.finished.emit(video_id)

        except BuildCancelled:
            # Not an error — clean exit.
            self.signals.cancelled.emit()
        # Explicit network rung — MUST come BEFORE broad Exception so
        # the friendly message wins over str(e). Revision iter 1 W1.
        except (
            socket.error,
            ConnectionError,
            TimeoutError,
            urllib.error.URLError,
            OSError,
        ) as e:
            # NEVER include str(e) — see T-08-04-08 (no IP/port/URL leak
            # into the dialog footer).
            log.info("YouTube upload network error: %s", type(e).__name__)
            self.signals.error.emit(_MSG_NETWORK)
        except Exception as e:
            # Broad fallback — surface a short message. NEVER log creds.
            msg = str(e) if str(e) else type(e).__name__
            self.signals.error.emit(msg)


def _is_quota_exceeded(err: HttpError) -> bool:
    """Return True iff ``err``'s payload identifies a quotaExceeded reason.

    HttpError carries ``error_details`` (a list of ``{reason: ...}``
    dicts) AND ``_get_reason()`` (string from the parsed body). Inspect
    both because the field naming has drifted across googleapiclient
    versions; the safe approach is to scan everything we can read.
    """
    try:
        body = (err.content or b"").decode("utf-8", errors="replace").lower()
    except Exception:
        body = ""
    if "quotaexceeded" in body or "quotaexceeded" in (err.reason or "").lower():
        return True
    try:
        details = err.error_details or []
    except Exception:
        details = []
    for d in details:
        if isinstance(d, dict) and "quotaexceeded" in str(d.get("reason", "")).lower():
            return True
    return False


__all__ = ["YouTubeUploadRunnable"]
