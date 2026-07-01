"""quick-260621-gfq Task 2 — Region.mastering['normalize'] single source of truth.

Redesign (quick-260621-gfq): the standalone ``Region.normalize_enabled`` /
``Region.normalize_target_db`` fields are REMOVED. Per-keeper normalize now
lives EXCLUSIVELY in ``Region.mastering['normalize'] = {'enabled': bool,
'target_db': float}`` (default 0.0 dBFS).

Legacy sidecars carrying the old top-level ``normalize_enabled`` /
``normalize_target_db`` keys (the quick-260620-mgu shape) are migrated into
the mastering dict at load time. A sidecar already carrying
``mastering.normalize`` round-trips unchanged. Out-of-range / non-finite /
non-numeric target_db quarantines; ``mastering=None`` loads cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

from marmelade.audio.sidecar_cache import (
    Region,
    SCHEMA_VERSION,
    load_sidecar,
    save_sidecar,
)


def test_region_has_no_standalone_normalize_fields() -> None:
    """The standalone normalize attributes are gone from the dataclass."""
    r = Region(id="abc", start_sec=0.0, end_sec=1.0)
    assert not hasattr(r, "normalize_enabled")
    assert not hasattr(r, "normalize_target_db")


def test_mastering_normalize_round_trips(tmp_path: Path) -> None:
    """A Region with mastering.normalize enabled @ 0.0 round-trips save→load."""
    sidecar = tmp_path / "test.json"
    r = Region(
        id="id1234567890abcd",
        start_sec=0.0,
        end_sec=10.0,
        state="keeper",
        mastering={"normalize": {"enabled": True, "target_db": 0.0}},
    )
    save_sidecar(sidecar, [r])

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering is not None
    assert loaded[0].mastering["normalize"]["enabled"] is True
    assert loaded[0].mastering["normalize"]["target_db"] == 0.0


def test_legacy_top_level_normalize_enabled_migrates(tmp_path: Path) -> None:
    """Legacy top-level normalize_enabled=True migrates into mastering.normalize.

    The legacy enabled-without-target shape migrates to target 0.0 (the new
    default per locked decision #6, replacing the legacy -6.0).
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
                "normalize_enabled": True,
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering is not None
    assert loaded[0].mastering["normalize"]["enabled"] is True
    assert loaded[0].mastering["normalize"]["target_db"] == 0.0
    assert list(tmp_path.glob("test.json.corrupt-*")) == []


def test_legacy_explicit_target_db_preserved_on_migration(tmp_path: Path) -> None:
    """A legacy explicit normalize_target_db keeps its stored value on migration."""
    sidecar = tmp_path / "test.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "id1",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "keeper",
                "created_at": "x",
                "note": "",
                "normalize_enabled": True,
                "normalize_target_db": -12.0,
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert loaded[0].mastering["normalize"]["enabled"] is True
    assert loaded[0].mastering["normalize"]["target_db"] == -12.0


def test_legacy_no_normalize_keys_stays_unmastered(tmp_path: Path) -> None:
    """Legacy region with NO normalize keys → mastering stays None (no forced entry)."""
    sidecar = tmp_path / "test.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "id1",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "keeper",
                "created_at": "x",
                "note": "legacy region",
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering is None
    assert list(tmp_path.glob("test.json.corrupt-*")) == []


def test_mastering_none_loads_cleanly(tmp_path: Path) -> None:
    """A keeper with mastering omitted loads with mastering=None, no crash."""
    sidecar = tmp_path / "test.json"
    r = Region(id="id1234567890abcd", start_sec=0.0, end_sec=1.0, state="keeper")
    save_sidecar(sidecar, [r])
    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering is None


def test_out_of_range_target_db_quarantines(tmp_path: Path) -> None:
    """mastering.normalize.target_db out of [-60, 0] → quarantine."""
    sidecar = tmp_path / "test.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "id1",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "keeper",
                "created_at": "x",
                "note": "",
                "mastering": {"normalize": {"enabled": True, "target_db": 6.0}},
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert loaded == []
    assert len(list(tmp_path.glob("test.json.corrupt-*"))) == 1


def test_non_finite_target_db_quarantines(tmp_path: Path) -> None:
    """mastering.normalize.target_db = Infinity → quarantine."""
    sidecar = tmp_path / "test.json"
    raw = (
        '{"schema_version": %d, "regions": [{"id": "id1", '
        '"start_sec": 0.0, "end_sec": 1.0, "state": "keeper", '
        '"created_at": "x", "note": "", '
        '"mastering": {"normalize": {"enabled": true, "target_db": Infinity}}}]}'
        % SCHEMA_VERSION
    )
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(raw)

    loaded, _ = load_sidecar(sidecar)
    assert loaded == []
    assert len(list(tmp_path.glob("test.json.corrupt-*"))) == 1


def test_non_bool_normalize_enabled_quarantines(tmp_path: Path) -> None:
    """mastering.normalize.enabled non-bool → quarantine (generic enabled check)."""
    sidecar = tmp_path / "test.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "id1",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "keeper",
                "created_at": "x",
                "note": "",
                "mastering": {"normalize": {"enabled": "yes", "target_db": 0.0}},
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert loaded == []
    assert len(list(tmp_path.glob("test.json.corrupt-*"))) == 1


def test_in_range_target_db_round_trips(tmp_path: Path) -> None:
    """A non-default but in-range mastering.normalize.target_db is preserved."""
    sidecar = tmp_path / "test.json"
    r = Region(
        id="id1234567890abcd",
        start_sec=0.0,
        end_sec=10.0,
        state="keeper",
        mastering={"normalize": {"enabled": True, "target_db": -12.0}},
    )
    save_sidecar(sidecar, [r])
    loaded, _ = load_sidecar(sidecar)
    assert loaded[0].mastering["normalize"]["target_db"] == -12.0
