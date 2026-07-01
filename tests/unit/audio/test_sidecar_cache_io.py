"""Unit tests for :mod:`marmelade.audio.sidecar_cache` — round-trip + atomic + traversal guard.

Plan 03-01 Task 1 — RED first. Mirrors :mod:`tests.unit.test_heatmap_cache`
discipline: pin the regex contract, pin the schema-version constant, pin the
atomic-write idiom, pin the traversal guard.

Six core pins:
    * ``_KEY_RE`` regex pattern matches ``proxy_cache._KEY_RE`` byte-for-byte.
    * ``SCHEMA_VERSION`` constant equals 1.
    * ``sidecar_path`` rejects keys outside ``^[0-9a-f]{16}$`` BEFORE Path
      arithmetic (traversal guard — T-03-01-01).
    * ``sidecar_path`` returns the expected ``cache_root / 'sidecars' / '{key}.json'``.
    * Round-trip: ``save_sidecar`` + ``load_sidecar`` returns equivalent regions.
    * ``save_sidecar`` is atomic — no ``.tmp`` sibling remains after success.
    * ``save_sidecar`` creates the ``sidecars/`` subdirectory lazily.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import pytest

from marmelade.audio import sidecar_cache
from marmelade.audio.sidecar_cache import (
    Region,
    SCHEMA_VERSION,
    load_sidecar,
    save_sidecar,
    sidecar_path,
)
from marmelade.paths import default_cache_root  # noqa: F401 — patched by tmp_cache_dir


def test_key_re_pattern_pinned() -> None:
    """``_KEY_RE.pattern == r'^[0-9a-f]{16}$'`` — identical to proxy_cache / heatmap_cache."""
    assert sidecar_cache._KEY_RE.pattern == r"^[0-9a-f]{16}$"


def test_schema_version_is_one() -> None:
    """``SCHEMA_VERSION == 1`` — Phase 3 ships v1 of the JSON schema."""
    assert sidecar_cache.SCHEMA_VERSION == 1
    assert SCHEMA_VERSION == 1


def test_sidecar_path_rejects_traversal(tmp_path: Path) -> None:
    """A non-conforming key raises ValueError BEFORE any Path arithmetic (T-03-01-01)."""
    with pytest.raises(ValueError):
        sidecar_path(tmp_path, "../etc/passwd")


def test_sidecar_path_accepts_valid_key(tmp_path: Path) -> None:
    """A valid 16-hex key resolves to ``cache_root / 'sidecars' / '{key}.json'``."""
    p = sidecar_path(tmp_path, "0123456789abcdef")
    assert p == tmp_path / "sidecars" / "0123456789abcdef.json"


def test_round_trip_write_then_load(tmp_path: Path) -> None:
    """save_sidecar followed by load_sidecar returns equivalent Region list."""
    p = sidecar_path(tmp_path, "0123456789abcdef")
    regions_in = [
        Region(
            id="abc123",
            start_sec=1.5,
            end_sec=3.25,
            state="untouched",
            created_at="2026-05-16T00:00:00",
            note="",
        ),
        Region(
            id="def456",
            start_sec=10.0,
            end_sec=20.5,
            state="untouched",
            created_at="2026-05-16T00:00:00",
            note="a note",
        ),
    ]
    save_sidecar(p, regions_in)
    regions_out, _ = load_sidecar(p)
    assert len(regions_out) == 2
    assert [asdict(r) for r in regions_out] == [asdict(r) for r in regions_in]


def test_write_is_atomic_via_tmp_rename(tmp_path: Path) -> None:
    """After save_sidecar returns, no ``.tmp`` sibling remains in the directory."""
    p = sidecar_path(tmp_path, "0123456789abcdef")
    regions_in = [
        Region(
            id="abc123",
            start_sec=0.0,
            end_sec=1.0,
            state="untouched",
            created_at="2026-05-16T00:00:00",
            note="",
        )
    ]
    save_sidecar(p, regions_in)
    # No leftover .tmp anywhere in the directory.
    siblings = list(p.parent.iterdir())
    assert all(not s.name.endswith(".tmp") for s in siblings), (
        f"unexpected .tmp leftover: {siblings}"
    )
    assert p.exists()


def test_write_creates_sidecars_subdir(tmp_path: Path) -> None:
    """The 'sidecars/' subdir is created lazily on first save_sidecar."""
    p = sidecar_path(tmp_path, "0123456789abcdef")
    assert not p.parent.exists()  # not yet
    save_sidecar(p, [])
    assert p.parent.is_dir()
    assert p.parent.name == "sidecars"


def test_save_empty_list_then_load_returns_empty(tmp_path: Path) -> None:
    """An empty regions list round-trips as an empty list (not None)."""
    p = sidecar_path(tmp_path, "0123456789abcdef")
    save_sidecar(p, [])
    out, _ = load_sidecar(p)
    assert out == []


def test_cache_key_re_exported(tmp_path: Path) -> None:
    """``cache_key`` is re-exported from proxy_cache for single-import-surface."""
    from marmelade.audio.sidecar_cache import cache_key as sc_cache_key
    from marmelade.audio.proxy_cache import cache_key as pc_cache_key

    # Identity preserved — same function object, not a wrapper.
    assert sc_cache_key is pc_cache_key
