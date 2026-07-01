"""Unit tests for marker persistence in :mod:`marmelade.audio.sidecar_cache`.

quick-260701-jc5 Task 1 (MARK-01 data model + MARK-05 persistence).

Pins:
    * ``Marker`` dataclass round-trips through ``save_sidecar`` → ``load_sidecar``.
    * ``save_sidecar(path, regions, markers)`` writes BOTH arrays; ``load_sidecar``
      returns ``(regions, markers)``.
    * An OLD regions-only sidecar (no ``"markers"`` key) loads as zero markers AND
      its regions round-trip unchanged (backward compat).
    * Every invalid-marker path quarantines (negative / non-finite / non-numeric
      ``time_sec``, over-long ``label``, missing/empty ``id``, count over
      ``_MAX_MARKERS``).

Mirrors the fixtures/patterns of :mod:`tests.unit.audio.test_sidecar_cache_io` and
:mod:`tests.unit.audio.test_sidecar_cache_quarantine`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from marmelade.audio import sidecar_cache
from marmelade.audio.sidecar_cache import (
    Marker,
    Region,
    SCHEMA_VERSION,
    load_sidecar,
    save_sidecar,
    sidecar_path,
)

KEY = "0123456789abcdef"


def _p(tmp_path: Path) -> Path:
    return sidecar_path(tmp_path, KEY)


# --------------------------------------------------------------------------- #
# Data model + round-trip
# --------------------------------------------------------------------------- #


def test_marker_dataclass_defaults() -> None:
    """Marker(id, time_sec) fills label='' and an ISO created_at."""
    m = Marker(id="m1", time_sec=3.5)
    assert m.id == "m1"
    assert m.time_sec == 3.5
    assert m.label == ""
    assert isinstance(m.created_at, str) and m.created_at  # non-empty ISO string


def test_round_trip_regions_and_markers(tmp_path: Path) -> None:
    """save_sidecar(path, regions, markers) then load_sidecar returns both, equal."""
    p = _p(tmp_path)
    regions_in = [
        Region(
            id="r1",
            start_sec=1.0,
            end_sec=2.0,
            state="keeper",
            created_at="2026-07-01T00:00:00",
            note="hi",
        )
    ]
    markers_in = [
        Marker(id="m1", time_sec=0.0, label="", created_at="2026-07-01T00:00:00"),
        Marker(id="m2", time_sec=12.5, label="chorus", created_at="2026-07-01T00:00:01"),
    ]
    save_sidecar(p, regions_in, markers_in)
    regions_out, markers_out = load_sidecar(p)
    assert [asdict(r) for r in regions_out] == [asdict(r) for r in regions_in]
    assert [asdict(m) for m in markers_out] == [asdict(m) for m in markers_in]


def test_markers_default_empty_when_omitted(tmp_path: Path) -> None:
    """save_sidecar without a markers arg writes an empty markers array."""
    p = _p(tmp_path)
    save_sidecar(p, [])
    regions_out, markers_out = load_sidecar(p)
    assert regions_out == []
    assert markers_out == []
    # The key IS present in the payload (additive, explicit empty list).
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data.get("markers") == []


# --------------------------------------------------------------------------- #
# Backward compatibility — old sidecar with NO "markers" key
# --------------------------------------------------------------------------- #


def test_old_sidecar_without_markers_loads_zero_markers(tmp_path: Path) -> None:
    """A hand-built regions-only payload (no 'markers' key) loads as zero markers
    AND its regions round-trip unchanged."""
    p = _p(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": [
            {
                "id": "r1",
                "start_sec": 1.0,
                "end_sec": 5.0,
                "state": "untouched",
                "created_at": "2026-01-01T00:00:00",
                "note": "",
            }
        ],
        # NOTE: no "markers" key at all.
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    regions_out, markers_out = load_sidecar(p)
    assert markers_out == []
    assert len(regions_out) == 1
    assert regions_out[0].id == "r1"
    assert regions_out[0].start_sec == 1.0
    assert regions_out[0].end_sec == 5.0
    # File not quarantined — still present.
    assert p.exists()


# --------------------------------------------------------------------------- #
# Quarantine paths (T-jc5-01 / T-jc5-02 / T-jc5-03 / T-jc5-04)
# --------------------------------------------------------------------------- #


def _write_payload_with_markers(p: Path, markers: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "regions": [], "markers": markers}
    p.write_text(json.dumps(payload), encoding="utf-8")


def _assert_quarantined(p: Path) -> None:
    regions_out, markers_out = load_sidecar(p)
    assert (regions_out, markers_out) == ([], [])
    assert not p.exists()  # renamed to .corrupt-*
    corrupts = list(p.parent.glob(f"{p.name}.corrupt-*"))
    assert corrupts, "expected a quarantine sibling"


def test_negative_time_sec_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    _write_payload_with_markers(
        p, [{"id": "m1", "time_sec": -0.5, "label": "", "created_at": "x"}]
    )
    _assert_quarantined(p)


def test_nan_time_sec_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    # NaN/inf are not valid JSON literals via json.dumps unless allow_nan.
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        '{"schema_version": 1, "regions": [], "markers": '
        '[{"id": "m1", "time_sec": NaN, "label": "", "created_at": "x"}]}'
    )
    p.write_text(raw, encoding="utf-8")
    _assert_quarantined(p)


def test_inf_time_sec_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        '{"schema_version": 1, "regions": [], "markers": '
        '[{"id": "m1", "time_sec": Infinity, "label": "", "created_at": "x"}]}'
    )
    p.write_text(raw, encoding="utf-8")
    _assert_quarantined(p)


def test_non_numeric_time_sec_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    _write_payload_with_markers(
        p, [{"id": "m1", "time_sec": "nope", "label": "", "created_at": "x"}]
    )
    _assert_quarantined(p)


def test_over_long_label_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    _write_payload_with_markers(
        p,
        [{"id": "m1", "time_sec": 1.0, "label": "z" * 201, "created_at": "x"}],
    )
    _assert_quarantined(p)


def test_missing_id_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    _write_payload_with_markers(
        p, [{"time_sec": 1.0, "label": "", "created_at": "x"}]
    )
    _assert_quarantined(p)


def test_empty_id_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    _write_payload_with_markers(
        p, [{"id": "", "time_sec": 1.0, "label": "", "created_at": "x"}]
    )
    _assert_quarantined(p)


def test_non_str_id_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    _write_payload_with_markers(
        p, [{"id": 123, "time_sec": 1.0, "label": "", "created_at": "x"}]
    )
    _assert_quarantined(p)


def test_markers_not_a_list_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    _write_payload_with_markers(p, {"not": "a list"})
    _assert_quarantined(p)


def test_too_many_markers_quarantines(tmp_path: Path) -> None:
    p = _p(tmp_path)
    over = sidecar_cache._MAX_MARKERS + 1
    markers = [
        {"id": f"m{i}", "time_sec": float(i), "label": "", "created_at": "x"}
        for i in range(over)
    ]
    _write_payload_with_markers(p, markers)
    _assert_quarantined(p)


def test_max_markers_constant_present() -> None:
    assert isinstance(sidecar_cache._MAX_MARKERS, int)
    assert sidecar_cache._MAX_MARKERS >= 1
