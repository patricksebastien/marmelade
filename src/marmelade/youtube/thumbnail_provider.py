"""Thumbnail provider — Picsum HTTP fetch + Pillow plain-color fallback (Phase 8 YT-03).

Qt-free per D-27 (N-3 invariant): zero ``PySide6.*`` imports. The two
public functions :func:`fetch_thumbnail` and :func:`_pillow_fallback`
are consumed by:

* :class:`marmelade.ui.upload_dialog.UploadDialog` (Phase A preview +
  the Refresh button).
* :class:`marmelade.youtube.upload_runnable.YouTubeUploadRunnable`
  (in-video poster — looped via ffmpeg per D-13 — and the
  ``youtube.thumbnails().set`` payload).

Reconciled aged decisions (CONTEXT.md §post_research_reconciliation):

* **R-01** — Image source is ``https://picsum.photos/1280/720``
  (Unsplash license, clear commercial use), NOT loremflickr.
* **R-02** — Image dimensions are 1280x720 (YouTube's actual recommended
  thumbnail size), NOT 1920x1080.
* **R-03** — Cache-bust query is ``?random=<nonce>`` (Picsum convention),
  NOT ``?lock=N``.

Failure semantics (D-16): three sequential HTTP attempts with a
2-second backoff between each. On all three failures the function
deterministically falls back to a Pillow-rendered plain-color JPEG whose
RGB triplet is ``hashlib.sha1(seed).digest()[:3]`` — same seed always
yields the same color, so a keeper that has lost network connectivity
still gets a recognizable visual identity.

Retried exceptions: ``(urllib.error.URLError, TimeoutError, OSError)`` —
the broad catch covers DNS resolution failure, connection reset, read
timeout, and the generic socket-error class (``OSError`` is the
base class for ``ConnectionError`` and ``socket.error`` in Python 3).
``socket.timeout`` is an alias of ``TimeoutError`` since Python 3.10.

Module-level constants are frozen so the source-grep gates in
``08-04-PLAN.md::acceptance_criteria`` can pin the contract:

* ``PICSUM_BASE_URL = "https://picsum.photos/1280/720"`` (R-01 + R-02)
* ``_MAX_RETRIES = 3`` (D-16)
* ``_BACKOFF_SEC = 2.0`` (D-16)
* ``_TIMEOUT_SEC = 10.0``
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.request
from io import BytesIO

from PIL import Image


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants — frozen contract pinned by source-grep gates.
# ---------------------------------------------------------------------------

PICSUM_BASE_URL: str = "https://picsum.photos/1280/720"
_MAX_RETRIES: int = 3
_BACKOFF_SEC: float = 2.0
_TIMEOUT_SEC: float = 10.0

# Image dimensions — pinned by source-grep + tests (R-02). Keep these
# named so a future change has to update the regression-pin too.
_THUMB_WIDTH: int = 1280
_THUMB_HEIGHT: int = 720

# JPEG quality for the Pillow fallback — high enough that the
# deterministic-color regression test can read the dominant RGB back
# within ±3 of the SHA1 derivation.
_FALLBACK_JPEG_QUALITY: int = 90


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def fetch_thumbnail(seed: str, nonce: int | str) -> bytes:
    """Fetch a 1280x720 JPEG thumbnail from Picsum, falling back deterministically.

    Args:
        seed: A stable identifier for the thumbnail subject — typically a
            keeper ``region_id`` (32-hex UUID) or, for bundle Share, a
            bundle hash. Only used by the Pillow fallback path for the
            deterministic color derivation.
        nonce: Cache-busting query value appended as ``?random=<nonce>``
            (R-03). The Refresh button in the upload dialog increments a
            per-session counter; pass an integer or any string Picsum will
            see as a "different" image request.

    Returns:
        Raw JPEG bytes. On the successful path these are whatever Picsum
        returned (typically ~50-250 KiB). On the fallback path they are a
        plain-color 1280x720 JPEG sized at roughly 10-30 KiB.

    The function never raises — every error path that would have surfaced
    a network exception falls through to :func:`_pillow_fallback` so a
    caller never has to handle ``URLError`` at this layer (the
    YouTubeUploadRunnable's exception ladder is one level up and is
    designed around the upload's own ``HttpError`` taxonomy, not the
    thumbnail's optional flourish).
    """
    url = f"{PICSUM_BASE_URL}?random={nonce}"

    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=_TIMEOUT_SEC) as resp:
                data = resp.read()
            # Sanity-check: a Picsum response should decode as a valid
            # image. A degenerate response (empty body, HTML error page)
            # falls through to the fallback.
            try:
                Image.open(BytesIO(data)).verify()
            except Exception as verify_err:
                log.info(
                    "Picsum returned non-image bytes (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    verify_err,
                )
                if attempt + 1 < _MAX_RETRIES:
                    time.sleep(_BACKOFF_SEC)
                continue
            return data
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            log.info(
                "Picsum fetch failed (attempt %d/%d): %s",
                attempt + 1,
                _MAX_RETRIES,
                err,
            )
            if attempt + 1 < _MAX_RETRIES:
                time.sleep(_BACKOFF_SEC)

    # All retries exhausted — deterministic Pillow fallback (D-16).
    log.warning(
        "Picsum fetch exhausted %d retries for seed=%r; using SHA1-color fallback.",
        _MAX_RETRIES,
        seed,
    )
    return _pillow_fallback(seed)


def _pillow_fallback(seed: str) -> bytes:
    """Render a 1280x720 plain-color JPEG whose RGB is ``sha1(seed)[:3]``.

    Deterministic: identical seeds yield byte-identical JPEGs (Pillow's
    JPEG encoder is deterministic for a given input + quality). Used as
    the network-failure fallback so an upload still ships SOMETHING
    visually identifying the keeper.

    Args:
        seed: The same identifier passed to :func:`fetch_thumbnail` —
            typically a keeper ``region_id`` (32-hex UUID) or a bundle
            hash. Encoded as UTF-8 before hashing.

    Returns:
        A 1280x720 JPEG byte string sized at roughly 10-30 KiB (the
        plain-color content compresses very well).
    """
    digest = hashlib.sha1(seed.encode("utf-8")).digest()
    rgb = (int(digest[0]), int(digest[1]), int(digest[2]))
    img = Image.new("RGB", (_THUMB_WIDTH, _THUMB_HEIGHT), color=rgb)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=_FALLBACK_JPEG_QUALITY)
    return buf.getvalue()


__all__ = [
    "PICSUM_BASE_URL",
    "fetch_thumbnail",
    "_pillow_fallback",
]
