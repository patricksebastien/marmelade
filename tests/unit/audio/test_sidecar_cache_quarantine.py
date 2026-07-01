"""Unit tests for sidecar quarantine on load failure (D-A3-5).

Plan 03-01 Task 1 — RED first. Mirrors :mod:`tests.unit.test_heatmap_cache`
out-of-bounds discipline: ANY load failure (JSON parse, schema mismatch,
missing required field, bound violation, OSError) MUST:

* NEVER raise (sidecar load must not block file-open).
* Return an empty list.
* Quarantine the bad file by renaming to ``{key}.json.corrupt-{ISO-timestamp}``.

Missing file is NOT a failure — it's the "no sidecar yet" case.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from marmelade.audio import sidecar_cache
from marmelade.audio.sidecar_cache import (
    Region,
    load_sidecar,
    save_sidecar,
    sidecar_path,
)
from marmelade.paths import default_cache_root  # noqa: F401 — patched by tmp_cache_dir


_VALID_KEY = "0123456789abcdef"


def _quarantine_files(parent: Path, stem: str) -> list[Path]:
    """Return all files in ``parent`` matching ``{stem}.corrupt-*``."""
    return sorted(
        f for f in parent.iterdir() if f.name.startswith(f"{stem}.corrupt-")
    )


def test_corrupt_json_quarantined_returns_empty(tmp_path: Path) -> None:
    """Write garbage to the sidecar path → load returns [] and quarantine file appears."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not valid json {")
    out, _ = load_sidecar(p)
    assert out == []
    quarantines = _quarantine_files(p.parent, p.name)
    assert len(quarantines) == 1
    assert not p.exists()  # quarantine renamed the original


def test_schema_version_too_new_quarantined(tmp_path: Path) -> None:
    """schema_version > SCHEMA_VERSION → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema_version": 99, "regions": []}))
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1


def test_missing_regions_key_quarantined(tmp_path: Path) -> None:
    """Top-level dict missing 'regions' key → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema_version": 1}))
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1


def test_invalid_state_value_quarantined(tmp_path: Path) -> None:
    """Region with state='archived' (not in _VALID_STATES) → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema_version": 1,
        "regions": [
            {
                "id": "abc",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "archived",
                "created_at": "2026-05-16T00:00:00",
                "note": "",
            }
        ],
    }))
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1


def test_start_gte_end_quarantined(tmp_path: Path) -> None:
    """Region with start_sec >= end_sec → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema_version": 1,
        "regions": [
            {
                "id": "abc",
                "start_sec": 5.0,
                "end_sec": 3.0,
                "state": "untouched",
                "created_at": "2026-05-16T00:00:00",
                "note": "",
            }
        ],
    }))
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1


def test_too_many_regions_quarantined(tmp_path: Path) -> None:
    """4097 regions (one over _MAX_REGIONS=4096) → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    regions = [
        {
            "id": f"r{i}",
            "start_sec": float(i),
            "end_sec": float(i) + 0.5,
            "state": "untouched",
            "created_at": "2026-05-16T00:00:00",
            "note": "",
        }
        for i in range(sidecar_cache._MAX_REGIONS + 1)
    ]
    p.write_text(json.dumps({"schema_version": 1, "regions": regions}))
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1


def test_overlong_note_quarantined(tmp_path: Path) -> None:
    """Region with note len > _MAX_NOTE_LEN (200) → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema_version": 1,
        "regions": [
            {
                "id": "abc",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "state": "untouched",
                "created_at": "2026-05-16T00:00:00",
                "note": "a" * 201,
            }
        ],
    }))
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1


def test_load_on_missing_file_returns_empty_no_quarantine(tmp_path: Path) -> None:
    """A missing file is the 'no sidecar yet' case — return [], no quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    # Parent doesn't exist either — should still be safe.
    assert not p.exists()
    out, _ = load_sidecar(p)
    assert out == []
    # No quarantine files created.
    if p.parent.exists():
        assert _quarantine_files(p.parent, p.name) == []


def test_quarantine_filename_timestamp_format(tmp_path: Path) -> None:
    """Quarantine filename matches r'\\.json\\.corrupt-\\d{8}T\\d{6}\\d{6}$' (microsecond precision)."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"garbage")
    load_sidecar(p)
    quarantines = _quarantine_files(p.parent, p.name)
    assert len(quarantines) == 1
    # ``%Y%m%dT%H%M%S%f`` — 8 digits date + T + 6 digits time + 6 digits microseconds = 20 hex-ish digits.
    pattern = re.compile(r"\.json\.corrupt-\d{8}T\d{6}\d{6}$")
    assert pattern.search(quarantines[0].name), (
        f"quarantine name {quarantines[0].name!r} does not match microsecond-precision pattern"
    )


def test_two_corruptions_same_second_distinct_quarantine_files(tmp_path: Path) -> None:
    """Two corruptions within the same second produce distinct quarantine names (W-8 microsecond precision).

    Microsecond-precision in the timestamp ensures two corruptions in the
    same second do not produce the same filename — which would silently
    lose forensic data (the second rename would overwrite the first).
    """
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    # First corruption.
    p.write_bytes(b"garbage A")
    load_sidecar(p)
    quarantines_a = _quarantine_files(p.parent, p.name)
    assert len(quarantines_a) == 1
    # Within the same second (no sleep), write another corrupt file to the
    # SAME path and trigger another load → quarantine B.
    p.write_bytes(b"garbage B")
    load_sidecar(p)
    quarantines_all = _quarantine_files(p.parent, p.name)
    assert len(quarantines_all) == 2, (
        f"expected two distinct quarantine files, got {[q.name for q in quarantines_all]}"
    )
    # They are distinct files — microsecond precision guarantees this.
    assert quarantines_all[0].name != quarantines_all[1].name


def test_non_dict_root_quarantined(tmp_path: Path) -> None:
    """A JSON file whose root is not a dict (e.g., a list) → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(["not", "a", "dict"]))
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1


def test_negative_start_sec_quarantined(tmp_path: Path) -> None:
    """Region with start_sec < 0 → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema_version": 1,
        "regions": [
            {
                "id": "abc",
                "start_sec": -1.0,
                "end_sec": 1.0,
                "state": "untouched",
                "created_at": "2026-05-16T00:00:00",
                "note": "",
            }
        ],
    }))
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1


def test_non_finite_end_sec_quarantined(tmp_path: Path) -> None:
    """Region with non-finite end_sec (Infinity) → quarantine."""
    p = sidecar_path(tmp_path, _VALID_KEY)
    p.parent.mkdir(parents=True, exist_ok=True)
    # JSON does not natively support Infinity; emit it as the
    # JavaScript literal that Python's json.loads accepts.
    p.write_text(
        '{"schema_version": 1, "regions": [{"id": "abc", "start_sec": 0.0, '
        '"end_sec": Infinity, "state": "untouched", "created_at": "x", "note": ""}]}'
    )
    out, _ = load_sidecar(p)
    assert out == []
    assert len(_quarantine_files(p.parent, p.name)) == 1
