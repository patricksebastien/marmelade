"""RED scaffold — R-1 spectral cache (header guard / atomic write / roundtrip).

Phase 11 Wave 0 (plan 11-01). PINs the not-yet-existing
:mod:`marmelade.audio.spectral_cache` API and its CR-03 hardening, mirroring the
existing :mod:`marmelade.audio.heatmap_cache` discipline (bounds-before-multiply,
``os.replace`` atomic write, traversal-guarded path builder).

Threat register (11-01 <threat_model>):
    * T-11-01 — a corrupt/hostile header must raise ``SpectralHeaderError``
      BEFORE any oversized ``np.memmap`` is created.
    * T-11-02 — ``spectral_path`` must reject traversal keys/names with
      ``ValueError`` before any ``Path`` join.

Production import lives inside each test so the module COLLECTS cleanly and is
RED on invocation until plan 11-02 lands the cache module.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest


def _cache():
    """Import the (later-wave) spectral cache module. Raises ImportError now."""
    from marmelade.audio import spectral_cache

    return spectral_cache


def test_roundtrip_mel_centroid_bands(tmp_path: Path) -> None:
    """R-1: write_spectral then load_* returns arrays equal within quant tol.

    The mel/centroid/band payloads are quantised on disk (uint8-ish for the
    image lanes), so equality is asserted within a small tolerance rather than
    bit-exact. Shapes must round-trip exactly.
    """
    sc = _cache()
    rng = np.random.default_rng(0)
    n_mels, n_frames = 64, 200
    mel = rng.random((n_mels, n_frames)).astype(np.float32)
    centroid = rng.random(n_frames).astype(np.float32)
    bands = rng.random((3, n_frames)).astype(np.float32)  # low/mid/high

    key = "0123456789abcdef"
    mel_path = sc.spectral_path(str(tmp_path), key, "mel")
    sc.write_spectral(
        str(tmp_path),
        key,
        sample_rate=48000,
        mel=mel,
        centroid=centroid,
        bands=bands,
    )

    loaded_mel, header = sc.load_mel(mel_path)
    loaded_mel = np.asarray(loaded_mel, dtype=np.float64)
    assert loaded_mel.shape == mel.shape
    # uint8 quantisation tolerance over a [0,1]-ish payload.
    assert np.max(np.abs(loaded_mel / loaded_mel.max() - mel / mel.max())) < 0.05

    loaded_centroid = np.asarray(
        sc.load_centroid(sc.spectral_path(str(tmp_path), key, "centroid"))[0]
    )
    assert loaded_centroid.shape == centroid.shape

    loaded_bands = np.asarray(
        sc.load_bands(sc.spectral_path(str(tmp_path), key, "bands"))[0]
    )
    assert loaded_bands.shape == bands.shape


def test_header_bounds_guard_rejects_oversize(tmp_path: Path) -> None:
    """R-1 / T-11-01: a header claiming dims beyond the bounds -> SpectralHeaderError.

    The guard MUST fire on the absolute-bounds check BEFORE the multiplicative
    expected-bytes computation (CR-03 mirror), so a hostile header cannot drive
    np.memmap into a multi-GiB mapping. We hand-craft a tiny file whose header
    advertises absurd dimensions and assert the loader rejects it.
    """
    sc = _cache()

    bad = tmp_path / "evil_mel.dat"
    # Header big enough to be parsed, dims absurd; payload deliberately tiny.
    # The exact header layout is owned by the production module — we only need
    # enough leading bytes that the reader gets past the "too short" check and
    # into the bounds check. 4096 bytes of 0xFF gives huge unsigned dims.
    bad.write_bytes(b"\xff" * 4096)

    with pytest.raises(sc.SpectralHeaderError):
        sc.load_mel(bad)


def test_header_size_guard_rejects_truncated(tmp_path: Path) -> None:
    """R-1 / T-11-01: header whose claimed size exceeds the file -> SpectralHeaderError.

    Even with in-bounds dimensions, the loader must verify
    ``header_size + n*itemsize <= filesize`` so a truncated/lying header is
    rejected instead of memmapping past EOF.
    """
    sc = _cache()
    rng = np.random.default_rng(1)
    n_mels, n_frames = 64, 200
    mel = rng.random((n_mels, n_frames)).astype(np.float32)
    key = "0123456789abcdef"
    sc.write_spectral(
        str(tmp_path), key, sample_rate=48000, mel=mel, centroid=None, bands=None
    )
    mel_path = sc.spectral_path(str(tmp_path), key, "mel")

    # Truncate the on-disk payload so the (valid) header now over-claims.
    raw = mel_path.read_bytes()
    mel_path.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(sc.SpectralHeaderError):
        sc.load_mel(mel_path)


def test_spectral_path_rejects_traversal_key_and_name(tmp_path: Path) -> None:
    """T-11-02: spectral_path rejects bad keys/names with ValueError pre-join.

    Mirrors heatmap_cache: key must match ^[0-9a-f]{16}$ and name must match
    ^[a-z][a-z0-9_]{0,31}$, validated BEFORE any Path join — so neither a
    hostile cache key nor a malicious lane name can escape the cache root.
    """
    sc = _cache()

    good_key = "0123456789abcdef"
    # Bad keys.
    for bad_key in ("../../etc/passwd", "GHIJKLMNOPQRSTUV", "short", "0123456789abcde"):
        with pytest.raises(ValueError):
            sc.spectral_path(str(tmp_path), bad_key, "mel")
    # Bad names.
    for bad_name in ("../mel", "mel/../..", "Mel", "9mel", "mel name"):
        with pytest.raises(ValueError):
            sc.spectral_path(str(tmp_path), good_key, bad_name)

    # Sanity: a well-formed key+name produces a path under the cache root.
    ok = sc.spectral_path(str(tmp_path), good_key, "mel")
    assert str(tmp_path) in str(ok)


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    """R-1: a completed write_spectral leaves the final .dat and no .tmp sibling.

    The writer must build at ``<path>.tmp`` then ``os.replace`` into place, so a
    successful write leaves exactly the final file (no lingering .tmp).
    """
    sc = _cache()
    rng = np.random.default_rng(2)
    mel = rng.random((32, 50)).astype(np.float32)
    key = "0123456789abcdef"
    sc.write_spectral(
        str(tmp_path), key, sample_rate=48000, mel=mel, centroid=None, bands=None
    )
    mel_path = sc.spectral_path(str(tmp_path), key, "mel")
    assert mel_path.exists()
    assert not mel_path.with_suffix(mel_path.suffix + ".tmp").exists()


def test_struct_header_is_documented_size() -> None:
    """Smoke: the module declares a fixed-size struct header (CR-03 layout).

    Not load-bearing for behaviour — just pins that the module exposes a struct
    header format so a future reviewer can grep the inline-literal gate. RED
    until the module exists.
    """
    sc = _cache()
    fmt = getattr(sc, "_HEADER_FORMAT", None)
    assert fmt is not None, "spectral_cache must declare a _HEADER_FORMAT"
    assert struct.calcsize(fmt) == getattr(sc, "_HEADER_SIZE")
