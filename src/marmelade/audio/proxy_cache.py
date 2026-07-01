"""On-disk BBC-audiowaveform-v2-compatible proxy cache (AUD-02).

This module is *pure Python* — no GUI-toolkit imports. The OS-cache-dir
helper that needs the writable-cache-location lookup lives at
:mod:`marmelade.paths` (``default_cache_root()``). N-3 invariant: the
``audio/`` package stays toolkit-free so its logic is unit-testable
without a graphical event loop.

File format (BBC audiowaveform v2):

    24-byte little-endian header followed by interleaved int16 min/max
    pairs (one pair per ``samples_per_pixel`` window).

    | Offset | Size | Type     | Field            | Value                |
    |--------|------|----------|------------------|----------------------|
    | 0      | 4    | int32 LE | version          | 2                    |
    | 4      | 4    | uint32 LE| flags            | 0 (bit 0 = 8-bit)    |
    | 8      | 4    | int32 LE | sample_rate (Hz) | e.g. 44100           |
    | 12     | 4    | int32 LE | samples_per_pixel| 256                  |
    | 16     | 4    | uint32 LE| length (pairs)   | frames // spp        |
    | 20     | 4    | int32 LE | channels         | 1 (Phase 1 mono)     |

Cache key (RESEARCH §Pattern 4 — see :func:`cache_key` docstring for the
exact concat order).

Security:
    * ``proxy_path`` requires the key to match ``^[0-9a-f]{16}$`` — cache
      subdirectories can never be user-controlled (T-02-01).
    * ``load_proxy`` validates version, flags, channels, and length-vs-file
      size BEFORE memmapping (T-02-02). On any mismatch a typed
      :class:`ProxyHeaderError` is raised so callers can rebuild from
      source.
    * ``write_proxy`` writes to a ``.tmp`` sibling and ``os.replace()``s into
      place atomically (T-02-04) — a half-written cache file is never
      visible to a concurrent reader.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xxhash

PROXY_VERSION = 2
DEFAULT_SAMPLES_PER_PIXEL = 256

# Header layout: 6 little-endian uint32 = 24 bytes.
_HEADER_FORMAT = "<IIIIII"
_HEADER_SIZE = 24
assert struct.calcsize(_HEADER_FORMAT) == _HEADER_SIZE

# Cache key shape: 16 lowercase hex chars (xxhash.xxh64 hexdigest length).
# T-02-01 — proxy_path validates against this regex.
_KEY_RE = re.compile(r"^[0-9a-f]{16}$")

# Bits defined in v2 flags. Bit 0: 0 = int16 data, 1 = int8 data. All other
# bits MUST be zero in a valid v2 file.
_FLAGS_KNOWN_MASK = 0x1

# Bytes-per-sample for the two flag-encoded dtypes.
_DTYPE_SIZE_INT16 = 2
_DTYPE_SIZE_INT8 = 1

# CR-03 fix: absolute upper bounds on the header's allocation-sensitive
# fields. Validated in ``load_proxy`` BEFORE the multiplicative
# ``expected_bytes = _HEADER_SIZE + length * 2 * dtype_size`` calculation so
# (a) ``np.memmap`` can never be tricked into a multi-GiB virtual mapping by
# a hostile/corrupt header that lies about ``length``, and (b) the
# multiplication cannot overflow ``ssize_t`` on 32-bit Python builds.
#
# Cache directory is user-writable per CLAUDE.md — a parallel process or a
# cache copied from another machine can plant a corrupt ``peaks.dat`` and
# this is the source-side gate for that threat (T-07-01..T-07-04).
_MAX_LENGTH = 64 * 1024 * 1024  # 64 M pairs — 3x headroom over the documented 21.6 M pairs worst case (8h @ sr=192_000 / spp=256). Anything larger is corrupt by construction.
_MAX_SAMPLES_PER_PIXEL = 1_000_000  # 1 M source samples per pair. Phase 1 uses 256; the bound is loose enough for future heatmap pyramids.
_MAX_SAMPLE_RATE = 768_000  # 768 kHz — covers professional audio (DSD-rate envelope). Phase 1 sees ≤ 192 kHz from pedalboard.

# Cache-key sampling window. RESEARCH §Pattern 4: read 64 KiB from each end.
_KEY_SAMPLE = 64 * 1024


@dataclass(frozen=True)
class ProxyHeader:
    """Parsed 24-byte BBC-audiowaveform v2 header.

    Attributes:
        version: Format version. Always 2 for Phase 1.
        flags: Bitfield. Bit 0 selects int8 (1) vs int16 (0) data; other
            bits MUST be zero.
        sample_rate: Source sample rate in Hz.
        samples_per_pixel: Number of source samples summarised by each pair.
        length: Number of min/max pairs (per channel).
        channels: Channel count. Always 1 for Phase 1 (mono mix-down).
    """

    version: int
    flags: int
    sample_rate: int
    samples_per_pixel: int
    length: int
    channels: int


class ProxyHeaderError(ValueError):
    """Raised by :func:`load_proxy` when the .dat header is unsupported or
    inconsistent with the file on disk.

    Callers (Plan 03's open handler) treat a ``ProxyHeaderError`` as a signal
    to discard the cache entry and rebuild from source.
    """


def cache_key(path: str | os.PathLike) -> str:
    """Return the 16-character hex cache key for ``path``.

    Concatenates: ``size (uint64 LE) || mtime_ns (uint64 LE) || head_64KB ||
    tail_64KB`` (or empty tail if file ≤ 128 KiB). Order matches RESEARCH
    §Pattern 4 code blocks (lines 411-412, 667-668).

    NOTE: RESEARCH §Pattern 4 prose at line 83 lists ``mtime || size`` — that
    is a known prose-vs-code inconsistency; the **code blocks are the
    contract**. Reordering changes every digest in the user's cache.

    Sub-100 ms regardless of file size — we read at most 128 KiB.
    """
    p = Path(path)
    st = os.stat(p)
    h = xxhash.xxh64()
    # (1) size FIRST — matches RESEARCH §Pattern 4 code blocks lines 411-412
    # and 667-668. N-4 invariant.
    h.update(st.st_size.to_bytes(8, "little", signed=False))
    # (2) mtime SECOND.
    h.update(int(st.st_mtime_ns).to_bytes(8, "little", signed=False))
    # (3) head 64 KiB.
    with open(p, "rb") as f:
        head = f.read(_KEY_SAMPLE)
        h.update(head)
        # (4) tail 64 KiB (only when file > 128 KiB so head and tail don't
        # overlap — matches RESEARCH guard ``st_size > SAMPLE * 2``).
        if st.st_size > _KEY_SAMPLE * 2:
            f.seek(-_KEY_SAMPLE, os.SEEK_END)
            h.update(f.read(_KEY_SAMPLE))
    return h.hexdigest()


def proxy_path(cache_root: Path, key: str) -> Path:
    """Return ``cache_root / 'proxies' / key / 'peaks.dat'``.

    T-02-01 mitigation: ``key`` MUST match ``^[0-9a-f]{16}$``. Anything else
    raises :class:`ValueError`. Cache subdirectory names can ONLY be 16-char
    lowercase hex digests; user-supplied filenames never reach the file
    system as path components.

    Does NOT create the directory — callers do that themselves immediately
    before writing.
    """
    if not _KEY_RE.match(key):
        raise ValueError(f"Invalid cache key: {key!r}")
    return Path(cache_root) / "proxies" / key / "peaks.dat"


def write_proxy(
    path: str | os.PathLike,
    sample_rate: int,
    samples_per_pixel: int,
    pairs_int16: np.ndarray,
) -> None:
    """Write a BBC-audiowaveform v2 .dat file atomically.

    Pre-conditions on ``pairs_int16``:
        * ``dtype == np.int16``
        * ``ndim == 2``
        * ``shape[1] == 2``

    Atomic write (T-02-04): the file is built at ``<path>.tmp`` and then
    ``os.replace()``d into place — a reader of the cache directory never
    observes a half-written ``peaks.dat``.
    """
    if pairs_int16.dtype != np.int16:
        raise ValueError(
            f"pairs_int16 dtype must be int16, got {pairs_int16.dtype}"
        )
    if pairs_int16.ndim != 2 or pairs_int16.shape[1] != 2:
        raise ValueError(
            f"pairs_int16 shape must be (N, 2), got {pairs_int16.shape}"
        )

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")  # e.g. peaks.dat.tmp

    length = int(pairs_int16.shape[0])
    # BBC audiowaveform v2 header: 6 little-endian uint32 = 24 bytes.
    # version, flags, sample_rate, samples_per_pixel, length, channels.
    header = struct.pack("<IIIIII", PROXY_VERSION, 0, int(sample_rate), int(samples_per_pixel), length, 1)
    with open(tmp, "wb") as f:
        f.write(header)
        # tofile is the documented fast path for raw binary numpy writes.
        pairs_int16.tofile(f)
    os.replace(tmp, p)


def load_proxy(path: str | os.PathLike) -> tuple[np.memmap, ProxyHeader]:
    """Read+validate the header and return ``(memmap, ProxyHeader)``.

    Header validation (T-02-02):
        * ``version`` MUST equal :data:`PROXY_VERSION` (2).
        * ``channels`` MUST equal 1 (Phase 1 supports mono only).
        * ``flags`` MUST set only bit 0 (any other bit set → reject).
        * ``length <= _MAX_LENGTH``, ``0 < samples_per_pixel <=
          _MAX_SAMPLES_PER_PIXEL``, ``0 < sample_rate <= _MAX_SAMPLE_RATE``
          (CR-03 source-side fix — absolute bounds enforced BEFORE the
          multiplicative size check so a hostile header cannot drive
          ``np.memmap`` past EOF or overflow ``ssize_t`` on 32-bit builds).
        * ``24 + length * 2 * dtype_size <= os.path.getsize(path)`` — the
          header.length field cannot lie about how many bytes follow.

    On success returns an ``np.memmap`` of shape ``(length, 2)`` with dtype
    ``int16`` (or ``int8`` when bit 0 is set) so the OS pages data in on
    demand — even an 8-hour proxy never sits whole in RAM.
    """
    p = Path(path)
    with open(p, "rb") as f:
        raw = f.read(_HEADER_SIZE)
    if len(raw) < _HEADER_SIZE:
        raise ProxyHeaderError(
            f"Proxy file too short for v2 header: {len(raw)} < {_HEADER_SIZE}"
        )

    version, flags, sample_rate, spp, length, channels = struct.unpack(
        _HEADER_FORMAT, raw
    )

    if version != PROXY_VERSION:
        raise ProxyHeaderError(
            f"Unsupported proxy version: {version} (expected {PROXY_VERSION})"
        )
    if channels != 1:
        raise ProxyHeaderError(
            f"Multi-channel proxies not yet supported: channels={channels}"
        )
    if flags & ~_FLAGS_KNOWN_MASK:
        raise ProxyHeaderError(
            f"Unrecognised flag bits set: 0x{flags:08x}"
        )

    # CR-03 fix: absolute bounds on length / spp / sample_rate BEFORE the
    # multiplicative size check (a hostile header lying about length cannot
    # drive np.memmap past EOF / into a multi-GiB virtual mapping, nor wrap
    # ssize_t on 32-bit Python builds).
    if length > _MAX_LENGTH:
        raise ProxyHeaderError(
            f"Header length implausibly large: {length} > {_MAX_LENGTH}"
        )
    if spp <= 0 or spp > _MAX_SAMPLES_PER_PIXEL:
        raise ProxyHeaderError(
            f"Header samples_per_pixel out of range: {spp} "
            f"(must be in (0, {_MAX_SAMPLES_PER_PIXEL}])"
        )
    if sample_rate <= 0 or sample_rate > _MAX_SAMPLE_RATE:
        raise ProxyHeaderError(
            f"Header sample_rate out of range: {sample_rate} "
            f"(must be in (0, {_MAX_SAMPLE_RATE}])"
        )

    dtype = np.int8 if (flags & 0x1) else np.int16
    dtype_size = _DTYPE_SIZE_INT8 if (flags & 0x1) else _DTYPE_SIZE_INT16
    file_size = os.path.getsize(p)
    expected_bytes = _HEADER_SIZE + length * 2 * dtype_size
    if expected_bytes > file_size:
        raise ProxyHeaderError(
            f"Header length exceeds file size: header claims "
            f"{expected_bytes} bytes, file is {file_size} bytes"
        )

    header = ProxyHeader(
        version=int(version),
        flags=int(flags),
        sample_rate=int(sample_rate),
        samples_per_pixel=int(spp),
        length=int(length),
        channels=int(channels),
    )

    data = np.memmap(
        str(p),
        dtype=dtype,
        mode="r",
        offset=_HEADER_SIZE,
        shape=(length, 2),
    )
    return data, header
