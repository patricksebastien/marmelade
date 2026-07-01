"""Unit tests for :mod:`marmelade.audio.naming_resolver` (Plan 03-04a — D-A4-1 + D-A4-2).

Pins (Qt-free; pure stdlib):

- Filename pattern locks ``{YYYY-MM-DD}_{HHMMSS}_{trait}.{ext}`` with explicit
  numeric inputs.
- Long offsets render as ``HHMMSS`` (zero-padded 6 digits) — including > 1h.
- First collision appends ``_02``; second collision appends ``_03``.
- 99 collisions don't crash — resolver finds the next free slot.
- Unsafe trait tokens (``../etc/passwd``) raise ``ValueError`` BEFORE path
  arithmetic — T-03-04a-01 mitigation.
- Unsupported extensions (e.g., ``aiff``) raise ``ValueError``.
- ``ext="wav"`` works alongside ``ext="mp3"`` — locks the dual-format support
  that Plan 03-04b consumes from MainWindow.
- ``dominant_trait_for_region`` returns ``"clip"`` when no heatmaps are
  cached for the source's cache_key.

quick-260701-muv note: the AI/DSP heatmap backend was removed as verified
dead code. ``_HEATMAP_REGISTRY`` is now permanently empty and
``dominant_trait_for_region`` always returns the ``"clip"`` fallback, so the
former energy-cache / slice-resolution / corrupt-file / registry-roster tests
(which imported the deleted heatmap subclasses + ``heatmap_cache``) were
removed along with the backend.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from marmelade.audio import naming_resolver
from marmelade.paths import default_cache_root  # noqa: F401 — patched by tmp_cache_dir


CACHE_KEY = "0123456789abcdef"


# ---------------------------------------------------------------------- helpers


def _make_source(tmp_path: Path, mtime_iso: str = "2026-04-03T12:34:56") -> Path:
    """Touch a fake source file and set its mtime to a known epoch.

    The resolver derives ``recorded_date`` from ``os.stat(source).st_mtime``
    so pinning mtime gives a deterministic ``YYYY-MM-DD`` prefix.
    """
    src = tmp_path / "source.wav"
    src.write_bytes(b"")  # zero-byte placeholder is fine — we don't read it.
    ts = datetime.fromisoformat(mtime_iso).timestamp()
    os.utime(str(src), (ts, ts))
    return src


# ------------------------------------------------------------ resolve_filename


def test_basic_pattern(tmp_path: Path) -> None:
    """region_start_sec=107 (= 0h 1m 47s) → ``2026-04-03_000147_loud.mp3``."""
    src = _make_source(tmp_path)
    out = naming_resolver.resolve_filename(
        source_path=src,
        region_start_sec=107.0,
        trait="loud",
        ext="mp3",
        output_dir=tmp_path,
    )
    assert out.name == "2026-04-03_000147_loud.mp3"


def test_long_offset_pattern(tmp_path: Path) -> None:
    """region_start_sec = 1h 14m 32s = 4472 → HHMMSS = ``011432``."""
    src = _make_source(tmp_path)
    out = naming_resolver.resolve_filename(
        source_path=src,
        region_start_sec=1 * 3600 + 14 * 60 + 32,
        trait="quiet",
        ext="mp3",
        output_dir=tmp_path,
    )
    assert out.name == "2026-04-03_011432_quiet.mp3"


def test_collision_suffix_starts_at_02(tmp_path: Path) -> None:
    """Pre-existing base file → resolver returns the ``_02`` variant."""
    src = _make_source(tmp_path)
    (tmp_path / "2026-04-03_000147_loud.mp3").write_bytes(b"")
    out = naming_resolver.resolve_filename(
        source_path=src,
        region_start_sec=107.0,
        trait="loud",
        ext="mp3",
        output_dir=tmp_path,
    )
    assert out.name == "2026-04-03_000147_loud_02.mp3"


def test_collision_suffix_increments(tmp_path: Path) -> None:
    """Base + ``_02`` taken → resolver returns ``_03``."""
    src = _make_source(tmp_path)
    (tmp_path / "2026-04-03_000147_loud.mp3").write_bytes(b"")
    (tmp_path / "2026-04-03_000147_loud_02.mp3").write_bytes(b"")
    out = naming_resolver.resolve_filename(
        source_path=src,
        region_start_sec=107.0,
        trait="loud",
        ext="mp3",
        output_dir=tmp_path,
    )
    assert out.name == "2026-04-03_000147_loud_03.mp3"


def test_collision_99_collisions_does_not_crash(tmp_path: Path) -> None:
    """Pre-create base + 98 collisions (_02 through _99) → resolver returns _100."""
    src = _make_source(tmp_path)
    (tmp_path / "2026-04-03_000147_loud.mp3").write_bytes(b"")
    for n in range(2, 100):  # _02 .. _99
        (tmp_path / f"2026-04-03_000147_loud_{n:02d}.mp3").write_bytes(b"")
    out = naming_resolver.resolve_filename(
        source_path=src,
        region_start_sec=107.0,
        trait="loud",
        ext="mp3",
        output_dir=tmp_path,
    )
    assert out.name == "2026-04-03_000147_loud_100.mp3"


def test_unsafe_trait_raises(tmp_path: Path) -> None:
    """A trait token with path separators must raise ValueError — T-03-04a-01."""
    src = _make_source(tmp_path)
    with pytest.raises(ValueError):
        naming_resolver.resolve_filename(
            source_path=src,
            region_start_sec=0.0,
            trait="../etc/passwd",
            ext="mp3",
            output_dir=tmp_path,
        )


def test_unsupported_ext_raises(tmp_path: Path) -> None:
    """``ext="aiff"`` is not in the allow-list — ValueError before any I/O."""
    src = _make_source(tmp_path)
    with pytest.raises(ValueError):
        naming_resolver.resolve_filename(
            source_path=src,
            region_start_sec=0.0,
            trait="loud",
            ext="aiff",
            output_dir=tmp_path,
        )


def test_ext_wav_accepted(tmp_path: Path) -> None:
    """``ext="wav"`` produces a ``.wav`` path — locks dual-format support (D-A4-4)."""
    src = _make_source(tmp_path)
    out = naming_resolver.resolve_filename(
        source_path=src,
        region_start_sec=107.0,
        trait="loud",
        ext="wav",
        output_dir=tmp_path,
    )
    assert out.suffix == ".wav"
    assert out.name == "2026-04-03_000147_loud.wav"


# --------------------------------------------------------- dominant_trait_for_region


def test_dominant_trait_for_region_no_cache_returns_clip(tmp_path: Path) -> None:
    """No heatmaps dir at all → fall back to ``"clip"`` (D-A4-2)."""
    label = naming_resolver.dominant_trait_for_region(
        cache_root=tmp_path,
        cache_key_hex=CACHE_KEY,
        region_start_sec=0.0,
        region_end_sec=10.0,
    )
    assert label == "clip"


def test_dominant_trait_for_region_unknown_heatmap_name_skipped(tmp_path: Path) -> None:
    """A ``.dat`` on disk cannot resolve against the (now empty) registry.

    quick-260701-muv: with the heatmap backend removed there are no
    registered subclasses, so any on-disk ``.dat`` is unusable and the
    resolver returns the ``"clip"`` fallback — the same observable answer
    as before the removal (the file never mapped to a live subclass).
    """
    heatmap_dir = tmp_path / "heatmaps" / CACHE_KEY
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    (heatmap_dir / "notavalidheatmap.dat").write_bytes(b"")
    label = naming_resolver.dominant_trait_for_region(
        cache_root=tmp_path,
        cache_key_hex=CACHE_KEY,
        region_start_sec=0.0,
        region_end_sec=10.0,
    )
    assert label == "clip"
