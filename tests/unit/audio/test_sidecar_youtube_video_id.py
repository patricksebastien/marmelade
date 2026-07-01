"""Phase 8 Plan 08-01 Task 4 — D-30 additive sidecar field (GREEN).

The one Wave-0-shippable test among Plan 08-01's RED stubs. Pins
Task 2's :attr:`Region.youtube_video_id` modifications across the
four sites:

* dataclass field declaration (default None)
* :func:`save_sidecar` omit-when-None pop
* :func:`_validate_payload` type-check read
* :class:`Region` constructor kwarg

Mirrors :mod:`tests.unit.audio.test_sidecar_mastering_field` (the
Phase 7 D-19 precedent) verbatim. Discriminating signal (placebo
audit per Phase 7 LEARNINGS): reverting any of the four sidecar_cache
sites makes one of these tests fail — the GREEN here is load-bearing.

No skip marker — this file is the Wave-0-shippable GREEN test, not
a stub. (The other Wave 0 RED stubs under ``tests/youtube/`` and
``tests/util/`` carry a module-level skip marker that downstream
plans remove as they implement each test.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marmelade.audio.sidecar_cache import (
    Region,
    SCHEMA_VERSION,
    load_sidecar,
    save_sidecar,
)


def test_legacy_sidecar_loads_with_none(tmp_path: Path) -> None:
    """Phase-7-era sidecar JSON (no ``youtube_video_id`` key) deserializes with None.

    D-30 backward-compat invariant — pre-Phase-8 sidecars (including
    every Phase 7 mastered keeper) load through ``load_sidecar``
    without warnings or errors, and every Region in the result has
    ``youtube_video_id is None``.
    """
    sidecar = tmp_path / "side.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "phase7keeper000001",
                "start_sec": 0.0,
                "end_sec": 60.0,
                "state": "keeper",
                "created_at": "2026-05-19T12:00:00",
                "note": "Phase 7 mastered keeper",
                "mastering": {
                    "limiter": {"enabled": True, "ceiling_dbtp": -1.0}
                },
                # NO "youtube_video_id" key — pre-Phase-8 sidecar shape.
            },
            {
                "id": "phase3regionA00001",
                "start_sec": 100.0,
                "end_sec": 110.0,
                "state": "untouched",
                "created_at": "2026-05-13T10:00:00",
                "note": "Phase 3 region with no mastering",
                # NO "mastering" and NO "youtube_video_id" keys —
                # pre-Phase-7 sidecar shape.
            },
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 2
    assert all(r.youtube_video_id is None for r in loaded)


def test_roundtrip_video_id(tmp_path: Path) -> None:
    """Region with youtube_video_id="dQw4w9WgXcQ" round-trips through save + load.

    D-30 happy path — a successful upload sets the field; subsequent
    sidecar save preserves the value; the next load returns the same
    Region with the same video_id.
    """
    sidecar = tmp_path / "side.json"
    r = Region(
        id="uploadedkeeper0001",
        start_sec=0.0,
        end_sec=120.0,
        state="keeper",
        youtube_video_id="dQw4w9WgXcQ",
    )
    save_sidecar(sidecar, [r])

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].youtube_video_id == "dQw4w9WgXcQ"


def test_validator_rejects_non_str(tmp_path: Path) -> None:
    """Malformed sidecar with youtube_video_id: 12345 (int) → quarantined.

    D-30 / T-08-01-04 — defense-in-depth type check. Non-str values
    raise :class:`SidecarValidationError`; ``load_sidecar`` catches and
    quarantines (renames to ``*.corrupt-<ts>``), returns empty list.
    The quarantine path is the standard Phase 3 D-A3-5 discipline; we
    verify both the quarantine file exists AND the returned list is
    empty.
    """
    sidecar = tmp_path / "side.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "badkeeper00000001",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "keeper",
                "created_at": "2026-05-22T12:00:00",
                "note": "",
                "youtube_video_id": 12345,  # int, not str — malformed
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert loaded == []

    quarantines = list(tmp_path.glob("side.json.corrupt-*"))
    assert len(quarantines) == 1, (
        f"expected exactly one quarantine sibling, got "
        f"{[p.name for p in quarantines]}"
    )


def test_omits_none_from_json(tmp_path: Path) -> None:
    """Saving a Region with youtube_video_id=None omits the key from JSON.

    D-30 backward-compat invariant — pre-Phase-8 readers must not see
    a new key. Mirrors the Phase 7 D-19 ``mastering`` omit-when-None
    discipline at ``save_sidecar`` lines 143-151.
    """
    sidecar = tmp_path / "side.json"
    r = Region(
        id="nevershared000001",
        start_sec=0.0,
        end_sec=30.0,
        state="keeper",
        youtube_video_id=None,
    )
    save_sidecar(sidecar, [r])

    with open(sidecar, "r", encoding="utf-8") as f:
        data = json.load(f)
    region = data["regions"][0]
    assert "youtube_video_id" not in region, (
        f"youtube_video_id must be omitted from JSON when None — saw "
        f"keys: {sorted(region.keys())}"
    )
