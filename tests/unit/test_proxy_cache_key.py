"""Unit tests for ``marmelade.audio.proxy_cache.cache_key`` (AUD-02).

The cache key is an xxhash-64 digest of ``size || mtime_ns || head_64KB
|| tail_64KB`` (or empty tail if file ≤ 128 KiB). Order matches RESEARCH
§Pattern 4 code blocks at lines 411-412 and 667-668 — the prose at line 83
is a known footnote inconsistency.

The tests below pin:
- Determinism for the same file.
- Invalidation on size, mtime, head, and tail content changes.
- Sub-100 ms timing budget on a 30 s WAV.
- A *frozen snapshot digest* — a regression pin that fails loudly on any
  future reorder/refactor of ``cache_key``.
- A *concat-order assertion* matching ``xxh64(size || mtime || head || tail)``
  and NOT ``xxh64(mtime || size || head || tail)`` (N-4 regression guard).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import xxhash

from marmelade.audio.proxy_cache import cache_key
from tests.fixtures.synthesize import make_sine


SAMPLE = 64 * 1024


def test_key_is_deterministic_for_same_file(tmp_path: Path) -> None:
    """Calling ``cache_key`` twice on the same file returns identical digests."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=2.0)
    k1 = cache_key(p)
    k2 = cache_key(p)
    assert k1 == k2
    assert len(k1) == 16
    assert all(c in "0123456789abcdef" for c in k1)


def test_key_changes_when_mtime_changes(tmp_path: Path) -> None:
    """Updating only ``mtime_ns`` invalidates the key."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=2.0)
    k1 = cache_key(p)

    # Push mtime forward by 1 second (ns precision).
    st = os.stat(p)
    new_ns = st.st_mtime_ns + 1_000_000_000
    os.utime(str(p), ns=(new_ns, new_ns))

    k2 = cache_key(p)
    assert k1 != k2


def test_key_changes_when_size_changes(tmp_path: Path) -> None:
    """Appending a byte (size grows) invalidates the key."""
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=2.0)
    k1 = cache_key(p)

    with open(p, "ab") as f:
        f.write(b"\x00")

    k2 = cache_key(p)
    assert k1 != k2


def test_key_changes_when_head_content_changes(tmp_path: Path) -> None:
    """Overwriting the first 4 KiB with new bytes invalidates the key.

    We restore the mtime after the in-place write so the size-and-mtime tuple
    is unchanged — only the head content has moved.
    """
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=2.0)
    k1 = cache_key(p)
    st = os.stat(p)
    saved_ns = st.st_mtime_ns

    with open(p, "r+b") as f:
        f.seek(0)
        f.write(b"\xff" * 4096)
    os.utime(str(p), ns=(saved_ns, saved_ns))

    k2 = cache_key(p)
    assert k1 != k2


def test_key_changes_when_tail_content_changes(tmp_path: Path) -> None:
    """Overwriting the last 4 KiB of a file > 128 KiB invalidates the key."""
    # Need a file > 128 KiB so cache_key reads the tail.
    p = tmp_path / "noise.bin"
    p.write_bytes(b"\x00" * (200 * 1024))  # 200 KiB > 128 KiB threshold
    k1 = cache_key(p)
    st = os.stat(p)
    saved_ns = st.st_mtime_ns

    with open(p, "r+b") as f:
        f.seek(-4096, os.SEEK_END)
        f.write(b"\xff" * 4096)
    os.utime(str(p), ns=(saved_ns, saved_ns))

    k2 = cache_key(p)
    assert k1 != k2


def test_key_is_sub_100ms_for_small_file(tmp_path: Path) -> None:
    """``cache_key`` is sub-100 ms on a 30 s WAV (~ 2.5 MiB)."""
    p = tmp_path / "sine30.wav"
    make_sine(p, duration_s=30.0)

    # Warm any FS caches first.
    cache_key(p)
    t0 = time.perf_counter()
    cache_key(p)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 100.0, f"cache_key took {elapsed_ms:.1f} ms (budget 100 ms)"


def test_cache_key_snapshot(tmp_path: Path) -> None:
    """Regression-pin: deterministic digest for a fixed-content file with frozen mtime.

    Any future reorder/refactor of ``cache_key`` changes this digest and the
    test fails loudly. The pinned value was computed during initial
    implementation by feeding the exact byte sequence
    ``size_LE || mtime_LE || head_64KB || tail_64KB`` to ``xxhash.xxh64``.
    """
    p = tmp_path / "snapshot.bin"
    p.write_bytes(b"x" * 200_000)
    ns = 1_700_000_000_000_000_000
    os.utime(str(p), ns=(ns, ns))

    assert cache_key(p) == "8bcd18d6368a7b38"


def test_cache_key_concat_order_is_size_then_mtime(tmp_path: Path) -> None:
    """N-4 — concat order MUST be ``size || mtime || head || tail`` per RESEARCH code blocks.

    Builds two stat-tuple-equivalent xxh64 digests by hand and asserts that
    ``cache_key`` matches the size-first form, NOT the mtime-first form.
    This pins the algorithm against drift away from RESEARCH §Pattern 4 code
    blocks (lines 411-412, 667-668).
    """
    p = tmp_path / "sine.wav"
    make_sine(p, duration_s=2.0)
    st = os.stat(p)

    with open(p, "rb") as f:
        head = f.read(SAMPLE)
        if st.st_size > SAMPLE * 2:
            f.seek(-SAMPLE, os.SEEK_END)
            tail = f.read(SAMPLE)
        else:
            tail = b""

    size_le = st.st_size.to_bytes(8, "little", signed=False)
    mtime_le = int(st.st_mtime_ns).to_bytes(8, "little", signed=False)

    # Canonical order per RESEARCH code blocks: size, then mtime.
    h_size_first = xxhash.xxh64()
    h_size_first.update(size_le)
    h_size_first.update(mtime_le)
    h_size_first.update(head)
    h_size_first.update(tail)

    # Drifted order matching the prose-only footnote inconsistency.
    h_mtime_first = xxhash.xxh64()
    h_mtime_first.update(mtime_le)
    h_mtime_first.update(size_le)
    h_mtime_first.update(head)
    h_mtime_first.update(tail)

    digest = cache_key(p)
    assert digest == h_size_first.hexdigest(), (
        f"cache_key digest {digest} did not match size-first xxh64 "
        f"{h_size_first.hexdigest()} — algorithm has drifted from RESEARCH "
        f"§Pattern 4 code blocks."
    )
    assert digest != h_mtime_first.hexdigest(), (
        "cache_key digest matches the mtime-first form — RESEARCH §Pattern 4 "
        "prose inconsistency may have leaked into implementation."
    )
