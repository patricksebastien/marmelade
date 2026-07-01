"""On-disk binary cache for the spectral proxy (Phase 11 — R-1).

Qt-free structural mirror of :mod:`marmelade.audio.heatmap_cache` (N-3
invariant: the audio package stays GUI-toolkit free so its logic is
unit-testable without an event loop). Where ``heatmap_cache`` persists a
single 1-D ``float32`` lane, this module persists the three sibling arrays
that back the Phase 11 spectral render modes:

    * ``mel.dat``      — 2-D ``uint8`` mel-magnitude image, ``(n_mels, n_frames)``
    * ``centroid.dat`` — 1-D ``float32`` spectral centroid, ``(n_frames,)``
    * ``bands.dat``    — 2-D ``float32`` low/mid/high band energies, ``(3, n_frames)``

All three live under ``<cache_root>/spectra/<key>/`` and share the
``proxy_cache.cache_key`` invalidation domain (D-18).

File format (version 1):

    36-byte little-endian header followed by the payload array.

    | Offset | Size | Type      | Field        | Notes                       |
    |--------|------|-----------|--------------|-----------------------------|
    | 0      | 4    | uint32 LE | version      | 1                           |
    | 4      | 4    | uint32 LE | flags        | 0 (reserved)                |
    | 8      | 4    | uint32 LE | sample_rate  | e.g. 48000                  |
    | 12     | 4    | uint32 LE | hop_length   | STFT hop in samples         |
    | 16     | 4    | uint32 LE | n_fft        | STFT window size            |
    | 20     | 4    | uint32 LE | n_mels       | mel-band count (image rows) |
    | 24     | 4    | uint32 LE | n_frames     | time columns                |
    | 28     | 4    | float32   | db_floor     | dBFS floor (e.g. -80.0)     |
    | 32     | 4    | float32   | db_ref       | dBFS reference (e.g. 0.0)   |

``db_floor`` / ``db_ref`` describe the dB window the mel image was
quantised over (see :func:`quantize_mel_db`) so the renderer's
``ImageItem.setLevels`` can map the stored ``uint8`` codes back to a dB
scale without re-deriving the window.

Security (mirrors :mod:`heatmap_cache` discipline):
    * :func:`spectral_path` requires ``key`` to match ``^[0-9a-f]{16}$``
      AND ``name`` to match ``^[a-z][a-z0-9_]{0,31}$`` BEFORE any
      filesystem use (T-11-02 traversal guard).
    * the loaders validate every absolute bound (``_MAX_*``, finite
      ``db_floor``/``db_ref``) BEFORE the multiplicative
      ``_HEADER_SIZE + n*itemsize <= filesize`` check (T-11-01 / CR-03
      bounds-before-multiply) so a hostile header cannot drive
      ``np.memmap`` past EOF or overflow ``ssize_t`` on 32-bit builds.
    * the writer builds a ``.tmp`` sibling and ``os.replace``s it into
      place atomically, unlinking any partial ``.tmp`` on
      ``BaseException`` before re-raising (T-11-03 atomic write).
"""

from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SPECTRAL_VERSION = 1

# Header layout: 7 little-endian uint32 + 2 little-endian float32 = 36 bytes.
# The literal `"<IIIIIIIff"` form below is the source-grep gate (Phase 1
# Pattern §"Inline-literal grep gates") — keep the assertion immediately
# after so a layout edit that desyncs the size fails at import.
_HEADER_FORMAT = "<IIIIIIIff"
_HEADER_SIZE = 36
assert struct.calcsize(_HEADER_FORMAT) == _HEADER_SIZE

# Cache key shape: 16 lowercase hex chars (xxhash.xxh64 hexdigest length) —
# IDENTICAL regex to heatmap_cache._KEY_RE / proxy_cache._KEY_RE so the
# spectral cache shares invalidation with the proxy/heatmap caches (D-18).
_KEY_RE = re.compile(r"^[0-9a-f]{16}$")

# Sibling .dat name component validator. A malicious future lane name like
# "../../etc/passwd" or "mel/../.." cannot smuggle path separators past the
# cache root (T-11-02). Identical regex to heatmap_cache._NAME_RE.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# CR-03 absolute bounds — applied BEFORE any multiplicative arithmetic in
# the loaders. A hostile/corrupt header on a user-writable cache dir cannot
# drive np.memmap into a multi-GiB virtual mapping or trigger ssize_t
# overflow. Values sit comfortably above 8 h @ ~1 col/s with headroom.
_MAX_N_MELS = 512  # mel-band rows; Phase 11 uses 128.
_MAX_FRAMES = 100_000  # time columns; 8 h @ ~1 col/s ≈ 28.8k, 3x headroom.
_MAX_SAMPLE_RATE = 768_000  # covers the DSD-rate envelope.
_MAX_N_FFT = 1 << 20  # rough cap; Phase 11 uses 2048.
_MAX_HOP = 1 << 20  # rough cap; Phase 11 hop ≈ sample_rate.

# Quantiser window defaults (RESEARCH Pattern 2 / D-02): a fixed 80 dB
# window referenced to 0 dBFS keeps quantisation single-pass deterministic
# (A2 / Pitfall 3) so a re-render is bit-identical to the cached build.
DB_REF = 0.0
DB_FLOOR = -80.0


@dataclass(frozen=True)
class SpectralHeader:
    """Parsed 36-byte spectral-cache v1 header.

    Attributes:
        version: Format version. Always 1 for Phase 11.
        flags: Bitfield reserved for future use. MUST be 0.
        sample_rate: Source sample rate in Hz (canonical 48000).
        hop_length: STFT hop in samples.
        n_fft: STFT window size in samples.
        n_mels: Mel-band count — the row dimension of ``mel.dat``.
        n_frames: Time-column count shared by all three siblings.
        db_floor: dBFS floor of the mel quantisation window (e.g. -80.0).
        db_ref: dBFS reference of the mel quantisation window (e.g. 0.0).
    """

    version: int
    flags: int
    sample_rate: int
    hop_length: int
    n_fft: int
    n_mels: int
    n_frames: int
    db_floor: float
    db_ref: float


class SpectralHeaderError(ValueError):
    """Raised by the loaders when a ``.dat`` header is unsupported or
    inconsistent with the file on disk.

    Callers (worker shells, the renderer) treat a ``SpectralHeaderError``
    as a signal to discard the cache entry and recompute from source.
    """


def spectral_path(cache_root: str | os.PathLike, key: str, name: str) -> Path:
    """Return ``cache_root / 'spectra' / key / f'{name}.dat'``.

    T-11-02 mitigation: ``key`` MUST match ``^[0-9a-f]{16}$`` AND ``name``
    MUST match ``^[a-z][a-z0-9_]{0,31}$`` BEFORE any filesystem use.
    Anything else raises :class:`ValueError`. Both components are validated
    so neither a hostile cache key nor a malicious lane name can escape the
    cache root via ``../`` traversal.

    Does NOT create the directory — :func:`write_spectral` does that
    immediately before writing.
    """
    if not _KEY_RE.match(key):
        raise ValueError(f"Invalid cache key: {key!r}")
    if not _NAME_RE.match(name):
        raise ValueError(f"Invalid spectral name: {name!r}")
    return Path(cache_root) / "spectra" / key / f"{name}.dat"


def quantize_mel_db(
    mel_mag: np.ndarray,
    *,
    db_floor: float = DB_FLOOR,
    db_ref: float = DB_REF,
) -> np.ndarray:
    """Quantise mel *magnitudes* to ``uint8`` over a fixed dB window.

    RESEARCH Pattern 2 / D-02: convert magnitude → dBFS, normalise over the
    ``[db_floor, db_ref]`` window, then map to ``uint8`` so the stored image
    is compact (1 B/cell) yet preserves the log-perceptual contrast the
    Magma spectrogram render expects.

    ``mag == 10**(db_ref/20)`` (1.0 at the 0 dBFS default) maps to ``255``;
    ``mag`` at or below ``db_floor`` maps to ``0`` (clipped). The window is
    fixed (not data-adaptive) so a re-render is bit-identical to the cached
    build (single-pass determinism, Pitfall 3).
    """
    span = float(db_ref) - float(db_floor)
    if span <= 0.0:
        raise ValueError(
            f"db_ref ({db_ref}) must be > db_floor ({db_floor})"
        )
    db = 20.0 * np.log10(np.maximum(np.asarray(mel_mag, dtype=np.float64), 1e-10))
    q = np.clip((db - float(db_floor)) / span, 0.0, 1.0)
    return np.round(q * 255.0).astype(np.uint8)


def _pack_header(header: SpectralHeader) -> bytes:
    """Pack a :class:`SpectralHeader` into its 36-byte little-endian form.

    The inline ``struct.pack(_HEADER_FORMAT, ...)`` is the write-side mirror
    of the read-side ``struct.unpack`` in the loaders.
    """
    return struct.pack(
        _HEADER_FORMAT,
        int(header.version),
        int(header.flags),
        int(header.sample_rate),
        int(header.hop_length),
        int(header.n_fft),
        int(header.n_mels),
        int(header.n_frames),
        float(header.db_floor),
        float(header.db_ref),
    )


def _atomic_write(path: Path, packed_header: bytes, array: np.ndarray) -> None:
    """Write ``packed_header`` + ``array`` to ``path`` atomically.

    Verbatim discipline of :func:`heatmap_cache.write_heatmap`: build at a
    ``.tmp`` sibling, ``os.replace`` into place, and unlink any partial
    ``.tmp`` on ``BaseException`` (the ``KeyboardInterrupt``-mid-write case)
    before re-raising. Guarantees a cancelled/crashed write leaves no
    partial ``.dat`` (T-11-03).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(packed_header)
            array.tofile(f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(str(tmp))
        except FileNotFoundError:
            pass
        raise


def write_spectral(
    cache_root: str | os.PathLike,
    key: str,
    *,
    sample_rate: int,
    mel: np.ndarray,
    centroid: np.ndarray | None = None,
    bands: np.ndarray | None = None,
    hop_length: int = 0,
    n_fft: int = 0,
    db_floor: float = DB_FLOOR,
    db_ref: float = DB_REF,
) -> None:
    """Write the mel image + optional centroid/bands siblings atomically.

    ``mel`` is a ``(n_mels, n_frames)`` float array of mel magnitudes already
    normalised into ``[0, 1]`` (the builder applies :func:`quantize_mel_db`'s
    dB window upstream and hands the normalised result here). It is stored as
    ``uint8`` (1 B/cell) via a linear ``round(clip(mel, 0, 1) * 255)`` so the
    on-disk codes round-trip back to the normalised magnitude within the
    quantisation step. ``db_floor``/``db_ref`` are recorded in every header
    so the renderer can map the stored codes to a dB scale.

    ``centroid`` (``(n_frames,)`` float) and ``bands`` (``(3, n_frames)``
    float) are stored as ``float32`` siblings when provided; ``None`` skips
    that sibling (e.g. the truncation/atomicity tests only need ``mel``).

    Each ``.dat`` is written through :func:`spectral_path` (traversal guard,
    T-11-02) and :func:`_atomic_write` (tmp + os.replace, T-11-03).
    """
    mel_arr = np.ascontiguousarray(mel)
    if mel_arr.ndim != 2:
        raise ValueError(
            f"mel must be 2-D (n_mels, n_frames), got ndim={mel_arr.ndim}"
        )
    n_mels, n_frames = int(mel_arr.shape[0]), int(mel_arr.shape[1])

    header = SpectralHeader(
        version=SPECTRAL_VERSION,
        flags=0,
        sample_rate=int(sample_rate),
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        n_mels=n_mels,
        n_frames=n_frames,
        db_floor=float(db_floor),
        db_ref=float(db_ref),
    )
    packed = _pack_header(header)

    mel_u8 = np.round(
        np.clip(mel_arr.astype(np.float64), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    _atomic_write(spectral_path(cache_root, key, "mel"), packed, mel_u8)

    if centroid is not None:
        centroid_arr = np.ascontiguousarray(centroid, dtype=np.float32)
        if centroid_arr.shape != (n_frames,):
            raise ValueError(
                f"centroid must be ({n_frames},), got {centroid_arr.shape}"
            )
        _atomic_write(
            spectral_path(cache_root, key, "centroid"), packed, centroid_arr
        )

    if bands is not None:
        bands_arr = np.ascontiguousarray(bands, dtype=np.float32)
        if bands_arr.shape != (3, n_frames):
            raise ValueError(
                f"bands must be (3, {n_frames}), got {bands_arr.shape}"
            )
        _atomic_write(
            spectral_path(cache_root, key, "bands"), packed, bands_arr
        )


def _read_header(path: Path) -> tuple[SpectralHeader, int]:
    """Read + bounds-validate the 36-byte header. Return ``(header, filesize)``.

    CR-03 (T-11-01): every absolute bound is checked BEFORE any caller
    multiplies ``n_mels * n_frames`` against the file size, so a hostile
    header cannot drive ``np.memmap`` into a multi-GiB mapping or overflow
    ``ssize_t``. Each check names its own field so the test suite can assert
    precedence. The multiplicative file-size guard lives in each loader
    (the payload dtype/shape differs per sibling).
    """
    p = Path(path)
    with open(p, "rb") as f:
        raw = f.read(_HEADER_SIZE)
    if len(raw) < _HEADER_SIZE:
        raise SpectralHeaderError(
            f"Spectral file too short for v1 header: {len(raw)} < {_HEADER_SIZE}"
        )

    (
        version,
        flags,
        sample_rate,
        hop_length,
        n_fft,
        n_mels,
        n_frames,
        db_floor,
        db_ref,
    ) = struct.unpack(_HEADER_FORMAT, raw)

    if version != SPECTRAL_VERSION:
        raise SpectralHeaderError(
            f"Unsupported spectral version: {version} (expected {SPECTRAL_VERSION})"
        )

    # CR-03 absolute bounds — ORDER MATTERS, applied BEFORE any
    # multiplicative arithmetic. Each names its field for precedence tests.
    if sample_rate <= 0 or sample_rate > _MAX_SAMPLE_RATE:
        raise SpectralHeaderError(
            f"Header sample_rate out of range: {sample_rate} "
            f"(must be in (0, {_MAX_SAMPLE_RATE}])"
        )
    if hop_length > _MAX_HOP:
        raise SpectralHeaderError(
            f"Header hop_length out of range: {hop_length} > {_MAX_HOP}"
        )
    if n_fft > _MAX_N_FFT:
        raise SpectralHeaderError(
            f"Header n_fft out of range: {n_fft} > {_MAX_N_FFT}"
        )
    if n_mels <= 0 or n_mels > _MAX_N_MELS:
        raise SpectralHeaderError(
            f"Header n_mels out of range: {n_mels} "
            f"(must be in (0, {_MAX_N_MELS}])"
        )
    if n_frames <= 0 or n_frames > _MAX_FRAMES:
        raise SpectralHeaderError(
            f"Header n_frames out of range: {n_frames} "
            f"(must be in (0, {_MAX_FRAMES}])"
        )
    if not math.isfinite(float(db_floor)) or not math.isfinite(float(db_ref)):
        raise SpectralHeaderError(
            f"Header db window must be finite, got "
            f"db_floor={db_floor!r}, db_ref={db_ref!r}"
        )

    header = SpectralHeader(
        version=int(version),
        flags=int(flags),
        sample_rate=int(sample_rate),
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        n_frames=int(n_frames),
        db_floor=float(db_floor),
        db_ref=float(db_ref),
    )
    return header, os.path.getsize(p)


def load_mel(path: str | os.PathLike) -> tuple[np.memmap, SpectralHeader]:
    """Read+validate ``mel.dat`` and return ``(uint8 memmap, header)``.

    The memmap has shape ``(n_mels, n_frames)`` and dtype ``uint8`` (mode
    ``'r'``) so the OS pages the image in on demand. The bounds-before-
    multiply check in :func:`_read_header` runs first; then the
    multiplicative ``_HEADER_SIZE + n_mels * n_frames * 1 <= filesize``
    guard runs BEFORE ``np.memmap`` (CR-03 / T-11-01).
    """
    p = Path(path)
    header, file_size = _read_header(p)
    expected_bytes = _HEADER_SIZE + header.n_mels * header.n_frames  # uint8 = 1 B
    if expected_bytes > file_size:
        raise SpectralHeaderError(
            f"Header mel size exceeds file: header claims {expected_bytes} "
            f"bytes, file is {file_size} bytes"
        )
    data = np.memmap(
        str(p),
        dtype=np.uint8,
        mode="r",
        offset=_HEADER_SIZE,
        shape=(header.n_mels, header.n_frames),
    )
    return data, header


def load_centroid(path: str | os.PathLike) -> tuple[np.memmap, SpectralHeader]:
    """Read+validate ``centroid.dat`` and return ``(float32 memmap, header)``.

    Shape ``(n_frames,)``. Same CR-03 discipline as :func:`load_mel`: bounds
    first, then ``_HEADER_SIZE + n_frames * 4 <= filesize`` BEFORE memmap.
    """
    p = Path(path)
    header, file_size = _read_header(p)
    expected_bytes = _HEADER_SIZE + header.n_frames * 4  # float32 = 4 B
    if expected_bytes > file_size:
        raise SpectralHeaderError(
            f"Header centroid size exceeds file: header claims {expected_bytes} "
            f"bytes, file is {file_size} bytes"
        )
    data = np.memmap(
        str(p),
        dtype=np.float32,
        mode="r",
        offset=_HEADER_SIZE,
        shape=(header.n_frames,),
    )
    return data, header


def load_bands(path: str | os.PathLike) -> tuple[np.memmap, SpectralHeader]:
    """Read+validate ``bands.dat`` and return ``(float32 memmap, header)``.

    Shape ``(3, n_frames)`` (low/mid/high). Same CR-03 discipline: bounds
    first, then ``_HEADER_SIZE + 3 * n_frames * 4 <= filesize`` BEFORE memmap.
    """
    p = Path(path)
    header, file_size = _read_header(p)
    expected_bytes = _HEADER_SIZE + 3 * header.n_frames * 4  # float32 = 4 B
    if expected_bytes > file_size:
        raise SpectralHeaderError(
            f"Header bands size exceeds file: header claims {expected_bytes} "
            f"bytes, file is {file_size} bytes"
        )
    data = np.memmap(
        str(p),
        dtype=np.float32,
        mode="r",
        offset=_HEADER_SIZE,
        shape=(3, header.n_frames),
    )
    return data, header
