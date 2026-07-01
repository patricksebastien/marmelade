"""Unit tests for ``marmelade.audio.audio_proxy_cache`` (AUD-04).

Covers (Phase 2.1 Plan 01 — Qt-free cache surface for the audio proxy layer):

- Path resolver shape: ``<cache_root>/audio/<key>.proxy.wav``.
- Traversal-guard mitigation (T-02.1-01): ``_KEY_RE`` rejects non-hex,
  uppercase, too-short, and empty keys before any Path arithmetic.
- Freshness probe: returns the proxy path on HIT, None on MISS; a touched
  source mtime invalidates a previously-fresh proxy (D-04 cache_key reuse).
- ``expected_proxy_bytes`` formula (D-01 stereo + float32 + 128 B header
  overhead).
- ``check_disk_space`` enforces the 1 GiB safety margin (RESEARCH Open-Q-2).
- Cache-size scanner (D-09): returns 0 when ``audio/`` is absent; sums
  ``st_size`` over written dummy proxy files otherwise.
- Clear-cache helper (D-08): deletes ``audio/`` and returns bytes freed;
  idempotent (second call returns 0).
- ``cache_key`` is re-exported (D-04 — single-source freshness across the
  three caches).
- Module-level invariants: ``_KEY_RE.pattern`` matches the contract.

Qt-free: this module imports nothing from PyQt6/PySide6 — it relies only on
the ``tmp_path`` fixture so the suite runs without an event loop. The
``audio_proxy_cache`` module under test is the N-3 isolation member of the
audio backbone.
"""

from __future__ import annotations

import os
import shutil as _shutil_for_monkeypatch  # noqa: F401 — used as monkeypatch target
import time
from pathlib import Path

import pytest

from marmelade.audio.audio_proxy_cache import (
    _DISK_SAFETY_MARGIN_BYTES,
    _KEY_RE,
    audio_cache_size_bytes,
    audio_proxy_is_fresh,
    audio_proxy_path,
    cache_key,
    check_disk_space,
    clear_audio_cache,
    expected_proxy_bytes,
)


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------

def test_key_re_pattern_matches_contract() -> None:
    """``_KEY_RE.pattern`` MUST be ``^[0-9a-f]{16}$`` — same as proxy_cache."""
    assert _KEY_RE.pattern == r"^[0-9a-f]{16}$"


def test_disk_safety_margin_is_one_gib() -> None:
    """RESEARCH Open-Q-2 — 1 GiB margin locked at module level."""
    assert _DISK_SAFETY_MARGIN_BYTES == 1 * 1024**3


def test_cache_key_is_reexported_identity() -> None:
    """D-04 single-source freshness — re-export is the SAME object, not a copy.

    If a callee imports ``cache_key`` from ``audio_proxy_cache`` and another
    callee imports it from ``proxy_cache`` they must be looking at literally
    the same function so a future tweak to the hash inputs propagates to all
    three caches simultaneously.
    """
    from marmelade.audio.audio_proxy_cache import cache_key as A
    from marmelade.audio.proxy_cache import cache_key as B
    assert A is B


# ---------------------------------------------------------------------------
# audio_proxy_path — path resolver + traversal guard
# ---------------------------------------------------------------------------

def test_audio_proxy_path_resolves_under_cache_root(tmp_path: Path) -> None:
    key = "0123456789abcdef"
    p = audio_proxy_path(tmp_path, key)
    assert p == tmp_path / "audio" / f"{key}.proxy.wav"


def test_audio_proxy_path_does_not_create_directory(tmp_path: Path) -> None:
    """Path resolver is pure — builder owns the mkdir (PATTERNS.md §1)."""
    key = "0123456789abcdef"
    audio_proxy_path(tmp_path, key)
    assert not (tmp_path / "audio").exists()


def test_audio_proxy_path_rejects_traversal_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid cache key"):
        audio_proxy_path(tmp_path, "../etc/passwd")


def test_audio_proxy_path_rejects_uppercase_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid cache key"):
        audio_proxy_path(tmp_path, "ABCDEF0123456789")


def test_audio_proxy_path_rejects_too_short_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid cache key"):
        audio_proxy_path(tmp_path, "tooshort")


def test_audio_proxy_path_rejects_empty_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid cache key"):
        audio_proxy_path(tmp_path, "")


# ---------------------------------------------------------------------------
# audio_proxy_is_fresh — freshness probe (D-04 + D-13)
# ---------------------------------------------------------------------------

def _write_source(tmp_path: Path, *, name: str = "src.mp3", body: bytes = b"\x00" * 1024) -> Path:
    """Materialise a small source file for cache_key + freshness tests.

    ``cache_key`` reads size + mtime + 64 KiB head + 64 KiB tail; a 1 KiB body
    exercises the head-only branch (file ≤ 128 KiB), which keeps the test
    cheap.
    """
    src = tmp_path / name
    src.write_bytes(body)
    return src


def test_audio_proxy_is_fresh_returns_none_on_miss(tmp_path: Path) -> None:
    src = _write_source(tmp_path)
    assert audio_proxy_is_fresh(tmp_path, src) is None


def test_audio_proxy_is_fresh_returns_path_on_hit(tmp_path: Path) -> None:
    src = _write_source(tmp_path)
    key = cache_key(src)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / f"{key}.proxy.wav").write_bytes(b"\x00" * 16)

    fresh = audio_proxy_is_fresh(tmp_path, src)
    assert fresh == audio_dir / f"{key}.proxy.wav"
    assert fresh is not None and fresh.exists()


def test_audio_proxy_is_fresh_recomputes_key_when_source_changes(
    tmp_path: Path,
) -> None:
    """Touching the source mtime invalidates the previously-fresh proxy.

    D-04: a source change must invalidate ALL caches in one go. The probe
    re-computes ``cache_key(source)`` on every call — it does NOT trust an
    on-disk index.
    """
    src = _write_source(tmp_path)
    key_before = cache_key(src)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / f"{key_before}.proxy.wav").write_bytes(b"\x00" * 16)
    assert audio_proxy_is_fresh(tmp_path, src) is not None

    # Bump mtime by 2 s — enough to escape filesystem mtime granularity on
    # all common file systems (ext4 nanosecond, HFS+ 1 s, NTFS 100 ns).
    new_mtime = time.time() + 2.0
    os.utime(src, (new_mtime, new_mtime))
    key_after = cache_key(src)
    assert key_after != key_before

    # The probe must now MISS — the proxy at key_before is stale.
    assert audio_proxy_is_fresh(tmp_path, src) is None


# ---------------------------------------------------------------------------
# expected_proxy_bytes — disk-preflight math (D-14)
# ---------------------------------------------------------------------------

def test_expected_proxy_bytes_formula_30s_44100() -> None:
    """30 s @ 44.1 kHz stereo float32 + 128 B header overhead.

    PATTERNS.md §1: ``int(duration_s * sample_rate) * 2 * 4 + 128``.
    Per D-01 the proxy is always stereo even for mono fixtures.
    """
    assert expected_proxy_bytes(30.0, 44100) == int(30.0 * 44100) * 2 * 4 + 128


def test_expected_proxy_bytes_zero_duration_is_header_only() -> None:
    """Empty source → just the 128 B header overhead."""
    assert expected_proxy_bytes(0.0, 44100) == 128


# ---------------------------------------------------------------------------
# check_disk_space — 1 GiB safety margin (RESEARCH Open-Q-2)
# ---------------------------------------------------------------------------

def test_check_disk_space_refuses_when_below_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ok`` is False when free < expected + 1 GiB margin.

    500 MiB free vs. a 10 MiB request still fails because the 1 GiB margin
    is non-negotiable — RESEARCH Open-Q-2: "if the user is within 1 GiB of
    empty, refusing to write a 10 MB cache is the right call".
    """
    # Patch on the module under test (it imports ``shutil`` at module top).
    import marmelade.audio.audio_proxy_cache as mod

    class _FakeUsage:
        total = 100 * 1024**3
        used = 99 * 1024**3
        free = 500 * 1024**2  # 500 MiB — below 1 GiB margin

    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _path: _FakeUsage())
    ok, needed, free = check_disk_space(tmp_path, expected_bytes=10 * 1024**2)
    assert ok is False
    assert needed == 10 * 1024**2 + 1 * 1024**3
    assert free == 500 * 1024**2


def test_check_disk_space_passes_when_above_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ok`` is True when free ≥ expected + 1 GiB margin."""
    import marmelade.audio.audio_proxy_cache as mod

    class _FakeUsage:
        total = 100 * 1024**3
        used = 0
        free = 10 * 1024**3  # 10 GiB

    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _path: _FakeUsage())
    expected = 100 * 1024**2  # 100 MiB
    ok, needed, free = check_disk_space(tmp_path, expected_bytes=expected)
    assert ok is True
    assert needed == expected + 1 * 1024**3
    assert free == 10 * 1024**3


def test_check_disk_space_creates_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-flight mkdir(parents=True, exist_ok=True) — caller doesn't have to."""
    import marmelade.audio.audio_proxy_cache as mod

    class _FakeUsage:
        total = 100 * 1024**3
        used = 0
        free = 10 * 1024**3

    monkeypatch.setattr(mod.shutil, "disk_usage", lambda _path: _FakeUsage())
    new_root = tmp_path / "nested" / "cache"
    assert not new_root.exists()
    check_disk_space(new_root, expected_bytes=1024)
    assert new_root.exists()


# ---------------------------------------------------------------------------
# audio_cache_size_bytes — D-09 footer scanner
# ---------------------------------------------------------------------------

def test_audio_cache_size_bytes_returns_zero_when_absent(tmp_path: Path) -> None:
    """``cache_root/audio/`` does not exist — returns 0, NO error."""
    assert audio_cache_size_bytes(tmp_path) == 0


def test_audio_cache_size_bytes_sums_written_files(tmp_path: Path) -> None:
    """Sum ``st_size`` for two dummy proxies (16-hex keys per the contract).

    Uses real hex-shaped filenames so a future tightening (e.g. ignore
    non-conforming entries) would not silently break the assertion.
    """
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload_a = b"\x00" * 1234
    payload_b = b"\x00" * 5678
    (audio_dir / "0123456789abcdef.proxy.wav").write_bytes(payload_a)
    (audio_dir / "fedcba9876543210.proxy.wav").write_bytes(payload_b)

    assert audio_cache_size_bytes(tmp_path) == len(payload_a) + len(payload_b)


# ---------------------------------------------------------------------------
# clear_audio_cache — D-08 manual wipe
# ---------------------------------------------------------------------------

def test_clear_audio_cache_returns_zero_when_absent(tmp_path: Path) -> None:
    assert clear_audio_cache(tmp_path) == 0


def test_clear_audio_cache_deletes_and_reports_bytes_freed(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload = b"\x00" * 4096
    (audio_dir / "0123456789abcdef.proxy.wav").write_bytes(payload)

    freed = clear_audio_cache(tmp_path)
    assert freed == len(payload)
    assert not audio_dir.exists()


def test_clear_audio_cache_is_idempotent(tmp_path: Path) -> None:
    """Second call returns 0 — no double-free, no error."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "0123456789abcdef.proxy.wav").write_bytes(b"\x00" * 100)

    first = clear_audio_cache(tmp_path)
    second = clear_audio_cache(tmp_path)
    assert first == 100
    assert second == 0
