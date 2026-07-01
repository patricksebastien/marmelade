"""Unit tests for ``marmelade.audio.proxy_cache`` I/O (AUD-02).

Covers:
- Round-trip: write_proxy → load_proxy returns equal int16 pairs and a
  ProxyHeader matching the inputs.
- Path-traversal mitigation (T-02-01): proxy_path validates the key.
- Header validation (T-02-02): refuses version != 2, channels != 1,
  unrecognised flags, and length-vs-file-size lies — all via ProxyHeaderError.
- Atomic write: a partial ``.dat.tmp`` is never visible as ``.dat``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from marmelade.audio import proxy_cache
from marmelade.audio.proxy_cache import (
    DEFAULT_SAMPLES_PER_PIXEL,
    PROXY_VERSION,
    ProxyHeader,
    ProxyHeaderError,
    load_proxy,
    proxy_path,
    write_proxy,
)


SR = 44100
SPP = DEFAULT_SAMPLES_PER_PIXEL


def _make_pairs(n: int = 5000) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(-1000, 1000, size=(n, 2), dtype=np.int16)


def test_write_then_load_roundtrip(tmp_path: Path) -> None:
    """``write_proxy`` then ``load_proxy`` reproduces the pairs and header."""
    pairs = _make_pairs(5000)
    p = tmp_path / "proxies" / "abc" / "peaks.dat"

    write_proxy(p, sample_rate=SR, samples_per_pixel=SPP, pairs_int16=pairs)

    data, header = load_proxy(p)
    assert isinstance(header, ProxyHeader)
    assert header.version == PROXY_VERSION == 2
    assert header.flags == 0
    assert header.sample_rate == SR
    assert header.samples_per_pixel == SPP
    assert header.length == 5000
    assert header.channels == 1

    # data should be a memmap of shape (length, 2) int16, equal to input.
    assert data.shape == (5000, 2)
    assert data.dtype == np.int16
    assert np.array_equal(np.asarray(data), pairs)


def test_write_proxy_creates_parent_directories(tmp_path: Path) -> None:
    """``write_proxy`` creates missing parent dirs via mkdir(parents=True)."""
    pairs = _make_pairs(10)
    nested = tmp_path / "a" / "b" / "c" / "peaks.dat"
    assert not nested.parent.exists()
    write_proxy(nested, sample_rate=SR, samples_per_pixel=SPP, pairs_int16=pairs)
    assert nested.exists()


def test_write_proxy_is_atomic_no_tmp_leftover(tmp_path: Path) -> None:
    """After a successful write, no ``.dat.tmp`` sibling remains."""
    pairs = _make_pairs(100)
    p = tmp_path / "peaks.dat"
    write_proxy(p, sample_rate=SR, samples_per_pixel=SPP, pairs_int16=pairs)

    assert p.exists()
    siblings = list(tmp_path.iterdir())
    tmp_files = [s for s in siblings if str(s).endswith(".tmp")]
    assert tmp_files == []


def test_write_proxy_rejects_wrong_dtype(tmp_path: Path) -> None:
    """``write_proxy`` rejects non-int16 pair arrays."""
    bad = np.zeros((10, 2), dtype=np.int32)
    p = tmp_path / "peaks.dat"
    with pytest.raises((ValueError, AssertionError, TypeError)):
        write_proxy(p, sample_rate=SR, samples_per_pixel=SPP, pairs_int16=bad)


def test_write_proxy_rejects_wrong_shape(tmp_path: Path) -> None:
    """``write_proxy`` rejects shapes other than ``(N, 2)``."""
    bad = np.zeros((10, 3), dtype=np.int16)
    p = tmp_path / "peaks.dat"
    with pytest.raises((ValueError, AssertionError)):
        write_proxy(p, sample_rate=SR, samples_per_pixel=SPP, pairs_int16=bad)


def test_load_proxy_refuses_wrong_version(tmp_path: Path) -> None:
    """``load_proxy`` raises ``ProxyHeaderError`` for an unsupported version."""
    p = tmp_path / "peaks.dat"
    # Manually write a v99 header.
    with open(p, "wb") as f:
        f.write(struct.pack("<IIIIII", 99, 0, SR, SPP, 0, 1))
    with pytest.raises(ProxyHeaderError, match="version"):
        load_proxy(p)


def test_load_proxy_refuses_multichannel(tmp_path: Path) -> None:
    """``load_proxy`` raises ``ProxyHeaderError`` for channels != 1."""
    p = tmp_path / "peaks.dat"
    with open(p, "wb") as f:
        f.write(struct.pack("<IIIIII", 2, 0, SR, SPP, 0, 2))
    with pytest.raises(ProxyHeaderError, match="channel"):
        load_proxy(p)


def test_load_proxy_refuses_truncated_file(tmp_path: Path) -> None:
    """``load_proxy`` raises ``ProxyHeaderError`` when header.length lies."""
    p = tmp_path / "peaks.dat"
    with open(p, "wb") as f:
        # Claim length=10000 (= 40000 bytes of data) but only write 100 pairs.
        f.write(struct.pack("<IIIIII", 2, 0, SR, SPP, 10000, 1))
        f.write(np.zeros((100, 2), dtype=np.int16).tobytes())
    with pytest.raises(ProxyHeaderError):
        load_proxy(p)


def test_load_proxy_refuses_unrecognised_flags(tmp_path: Path) -> None:
    """``load_proxy`` raises ``ProxyHeaderError`` for unknown flag bits."""
    p = tmp_path / "peaks.dat"
    # Bit 31 set — undefined in v2 spec.
    bad_flags = 0x80000000
    with open(p, "wb") as f:
        f.write(struct.pack("<IIIIII", 2, bad_flags, SR, SPP, 0, 1))
    with pytest.raises(ProxyHeaderError):
        load_proxy(p)


def test_proxy_path_resolves_under_cache_root(tmp_path: Path) -> None:
    """``proxy_path`` returns ``cache_root / 'proxies' / key / 'peaks.dat'``."""
    key = "0123456789abcdef"
    p = proxy_path(tmp_path, key)
    assert p == tmp_path / "proxies" / key / "peaks.dat"


def test_proxy_path_rejects_traversal_key(tmp_path: Path) -> None:
    """T-02-01 — ``proxy_path`` raises ``ValueError`` for non-hex keys."""
    with pytest.raises(ValueError, match="Invalid cache key"):
        proxy_path(tmp_path, "../etc/passwd")


def test_proxy_path_rejects_uppercase_key(tmp_path: Path) -> None:
    """T-02-01 — only lowercase 16-char hex passes the gate."""
    with pytest.raises(ValueError, match="Invalid cache key"):
        proxy_path(tmp_path, "ABCDEF0123456789")


def test_proxy_path_rejects_wrong_length_key(tmp_path: Path) -> None:
    """T-02-01 — keys must be exactly 16 chars."""
    with pytest.raises(ValueError, match="Invalid cache key"):
        proxy_path(tmp_path, "abc123")


# ---------------------------------------------------------------------------
# CR-03 source-side: absolute bounds on header fields (T-07-01..T-07-04).
#
# These tests prove that load_proxy rejects implausibly-large header fields
# BEFORE the multiplicative `expected_bytes = length * 2 * dtype_size`
# calculation is reached. A hostile/corrupt header on a user-writable cache
# dir cannot drive np.memmap into a multi-GiB virtual mapping or trigger an
# ssize_t overflow on 32-bit Python builds.
#
# The test files are HEADER-ONLY (24 bytes) — no body is written. The new
# field bounds fire before the existing file-size check, so we can prove the
# precedence by asserting on the field name in the error message rather than
# on a "file size" or "too short" substring.
# ---------------------------------------------------------------------------


def test_load_proxy_rejects_implausibly_large_length(tmp_path: Path) -> None:
    """CR-03 / T-07-01 — `length > _MAX_LENGTH` is rejected before memmap."""
    p = tmp_path / "bad_length.dat"
    bad_length = proxy_cache._MAX_LENGTH + 1
    header = struct.pack("<IIIIII", PROXY_VERSION, 0, SR, SPP, bad_length, 1)
    p.write_bytes(header)

    with pytest.raises(ProxyHeaderError, match="length") as exc_info:
        load_proxy(p)
    # The new bound fires first — message names the field, not a file-size lie.
    assert "length" in str(exc_info.value).lower()
    # Bound value (or the offending value) should appear so the operator can
    # tell at a glance which ceiling was breached.
    msg = str(exc_info.value)
    assert str(proxy_cache._MAX_LENGTH) in msg or str(bad_length) in msg


@pytest.mark.parametrize(
    "bad_spp",
    [0, 1_000_001],  # 0 = nonsense / divide-by-zero;  _MAX_SAMPLES_PER_PIXEL + 1
)
def test_load_proxy_rejects_invalid_samples_per_pixel(
    tmp_path: Path, bad_spp: int
) -> None:
    """CR-03 / T-07-03 — `samples_per_pixel` must be in (0, _MAX_SAMPLES_PER_PIXEL]."""
    # Sanity: the upper-bound parameter must match the constant + 1.
    if bad_spp > 0:
        assert bad_spp == proxy_cache._MAX_SAMPLES_PER_PIXEL + 1
    p = tmp_path / f"bad_spp_{bad_spp}.dat"
    header = struct.pack("<IIIIII", PROXY_VERSION, 0, SR, bad_spp, 0, 1)
    p.write_bytes(header)

    with pytest.raises(ProxyHeaderError) as exc_info:
        load_proxy(p)
    msg = str(exc_info.value).lower()
    assert "samples_per_pixel" in msg or "spp" in msg


@pytest.mark.parametrize(
    "bad_sr",
    [0, 768_001],  # 0 = divide-by-zero in render_proxy;  _MAX_SAMPLE_RATE + 1
)
def test_load_proxy_rejects_invalid_sample_rate(
    tmp_path: Path, bad_sr: int
) -> None:
    """CR-03 / T-07-04 — `sample_rate` must be in (0, _MAX_SAMPLE_RATE]."""
    if bad_sr > 0:
        assert bad_sr == proxy_cache._MAX_SAMPLE_RATE + 1
    p = tmp_path / f"bad_sr_{bad_sr}.dat"
    header = struct.pack("<IIIIII", PROXY_VERSION, 0, bad_sr, SPP, 0, 1)
    p.write_bytes(header)

    with pytest.raises(ProxyHeaderError) as exc_info:
        load_proxy(p)
    msg = str(exc_info.value).lower()
    assert "sample_rate" in msg or "sample rate" in msg
