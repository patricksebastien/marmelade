"""Phase 7 Plan 07-02 Task 1 (RED) — Region.mastering additive sidecar field.

D-19 — additive sidecar schema. Adds an optional ``mastering: dict | None``
field to :class:`marmelade.audio.sidecar_cache.Region`. Existing Phase 3
sidecars (no ``mastering`` key in JSON) deserialize cleanly with
``mastering=None``. The validator restricts stage keys to
``_STAGE_ORDER`` (defense-in-depth; no attacker-controllable key names
propagate downstream into ``config_hash`` / cache filename composition).
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


def test_region_defaults_to_no_mastering() -> None:
    """A freshly constructed Region has ``mastering=None`` by default.

    D-19 — additive field. Existing Phase 3 callers do NOT need to pass
    ``mastering=``; absence means "no mastering applied; export uses
    source proxy".
    """
    r = Region(id="abc", start_sec=0.0, end_sec=1.0)
    assert r.mastering is None


def test_sidecar_round_trip_with_mastering(tmp_path: Path) -> None:
    """A Region with a mastering dict round-trips through save + load.

    Limiter-only config (Phase 7 default chain).
    """
    sidecar = tmp_path / "test.json"
    mastering = {
        "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0}
    }
    r = Region(
        id="id1234567890abcd",
        start_sec=0.0,
        end_sec=10.0,
        state="keeper",
        mastering=mastering,
    )
    save_sidecar(sidecar, [r])

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering == mastering


def test_sidecar_round_trip_without_mastering_omits_key(tmp_path: Path) -> None:
    """Saving a Region with ``mastering=None`` MUST omit the key from JSON.

    Defense-in-depth — older readers that don't know the key never see it.
    """
    sidecar = tmp_path / "test.json"
    r = Region(
        id="id1234567890abcd",
        start_sec=0.0,
        end_sec=10.0,
        state="keeper",
        mastering=None,
    )
    save_sidecar(sidecar, [r])

    with open(sidecar, "r", encoding="utf-8") as f:
        data = json.load(f)
    region = data["regions"][0]
    assert "mastering" not in region


def test_validator_rejects_bogus_stage_name(tmp_path: Path) -> None:
    """Invalid stage name (not in ``_STAGE_ORDER``) → file quarantined on load.

    The validator's job is to reject before any downstream consumer
    (config_hash, cache filename) sees attacker-controllable keys. On
    quarantine the file is renamed to ``*.corrupt-<ts>`` and an empty
    list returns (load_sidecar never raises — D-A3-5).
    """
    sidecar = tmp_path / "test.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "id1",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "keeper",
                "created_at": "2026-05-19T12:00:00",
                "note": "",
                "mastering": {"badstage": {"enabled": True}},
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Quarantine → empty list (no raise).
    loaded, _ = load_sidecar(sidecar)
    assert loaded == []
    # Confirm quarantine — original file should be gone, replaced with a
    # ``*.corrupt-...`` sibling.
    quarantines = list(tmp_path.glob("test.json.corrupt-*"))
    assert len(quarantines) == 1


def test_validator_accepts_subset_of_stages(tmp_path: Path) -> None:
    """Subset of ``_STAGE_ORDER`` keys is valid (no need to include all stages)."""
    sidecar = tmp_path / "test.json"
    # Only "limiter" — every other stage is implicitly disabled.
    mastering = {"limiter": {"enabled": True, "ceiling_dbtp": -1.0}}
    r = Region(
        id="id1234567890abcd",
        start_sec=0.0,
        end_sec=10.0,
        state="keeper",
        mastering=mastering,
    )
    save_sidecar(sidecar, [r])
    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering == mastering


def test_old_sidecar_without_mastering_loads_with_none(tmp_path: Path) -> None:
    """Phase 3-era sidecar JSON (no ``mastering`` key) deserializes with ``mastering=None``.

    D-19 — backward compat for pre-Phase-7 sidecars. The loader uses
    ``payload.get("mastering")`` so absence is harmless.
    """
    sidecar = tmp_path / "test.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "id1",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "keeper",
                "created_at": "2026-05-19T12:00:00",
                "note": "old phase 3 region",
                # No "mastering" key
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering is None


def test_validator_rejects_missing_enabled_flag(tmp_path: Path) -> None:
    """A stage dict without ``enabled: bool`` is malformed → quarantined."""
    sidecar = tmp_path / "test.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "id1",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "keeper",
                "created_at": "2026-05-19T12:00:00",
                "note": "",
                # Missing "enabled" inside the limiter stage dict.
                "mastering": {"limiter": {"ceiling_dbtp": -1.0}},
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert loaded == []
