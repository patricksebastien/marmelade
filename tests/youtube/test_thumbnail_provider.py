"""Phase 8 Plan 08-04 — thumbnail fetch + Pillow fallback (YT-03).

Plan 08-04 Task 1 (TDD GREEN — Wave 0 skip marker removed). The 3 test
names below pin the canonical contract for
:mod:`marmelade.youtube.thumbnail_provider`.

R-01: Picsum (not loremflickr) — Unsplash license, commercial use OK.
R-02: 1280x720 (not 1920x1080) — YouTube's actual recommended thumbnail size.
R-03: ``?random=N`` cache-bust convention (not ``?lock=N``).
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from PIL import Image


def test_fetch_success(fake_picsum_response, monkeypatch) -> None:
    """Successful fetch returns a 1280x720 JPEG byte string from Picsum.

    Uses the conftest fixture (`fake_picsum_response`) which monkeypatches
    `urllib.request.urlopen` (the stdlib reference) to return a fake JPEG.
    The thumbnail provider re-imports urlopen at function-call time so the
    monkeypatch applies inside `fetch_thumbnail`.
    """
    from marmelade.youtube import thumbnail_provider as tp

    fake_picsum_response()  # success mode

    data = tp.fetch_thumbnail("seed_abc", 1)
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0
    img = Image.open(BytesIO(data))
    assert img.size == (1280, 720)


def test_fetch_3_retry_then_fallback(monkeypatch) -> None:
    """URLError on all 3 attempts -> Pillow plain-color fallback fires.

    The retry loop catches `(urllib.error.URLError, TimeoutError, OSError)`
    and sleeps `_BACKOFF_SEC` between attempts. After 3 failures the
    function falls back to `_pillow_fallback(seed)` which returns a
    deterministic 1280x720 JPEG.
    """
    from marmelade.youtube import thumbnail_provider as tp
    import urllib.error
    import urllib.request as urlreq

    call_count = {"n": 0}

    def _raise(*_a, **_kw):
        call_count["n"] += 1
        raise urllib.error.URLError("simulated network failure")

    monkeypatch.setattr(urlreq, "urlopen", _raise)

    # Skip sleeps for fast test.
    monkeypatch.setattr(tp.time, "sleep", lambda _s: None)

    data = tp.fetch_thumbnail("keeper_abc", 5)

    # Three retry attempts before fallback.
    assert call_count["n"] == 3, f"expected 3 urlopen attempts, got {call_count['n']}"
    assert isinstance(data, (bytes, bytearray))
    # Decodes as 1280x720 JPEG (Pillow fallback engaged).
    img = Image.open(BytesIO(data))
    assert img.size == (1280, 720)


def test_fallback_color_deterministic_from_sha1() -> None:
    """Fallback RGB color is deterministically derived from sha1(seed).digest()[:3].

    Two calls with the same seed return byte-identical JPEGs. The dominant
    color is the SHA1-derived RGB triplet within JPEG compression tolerance.
    """
    from marmelade.youtube import thumbnail_provider as tp

    out1 = tp._pillow_fallback("keeper_xyz")
    out2 = tp._pillow_fallback("keeper_xyz")
    assert out1 == out2, "Pillow fallback must be deterministic for same seed"

    # Verify dominant color matches the SHA1 derivation within JPEG tolerance.
    expected_rgb = tuple(int(c) for c in hashlib.sha1(b"keeper_xyz").digest()[:3])
    img = Image.open(BytesIO(out1)).convert("RGB")
    # Pick a center pixel (uniform plain-color image).
    center_rgb = img.getpixel((640, 360))
    # JPEG quality=90 introduces small chroma drift; allow ±3 per channel.
    for i in range(3):
        assert (
            abs(center_rgb[i] - expected_rgb[i]) <= 3
        ), f"channel {i}: center={center_rgb}, expected={expected_rgb}"
