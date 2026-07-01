"""Phase 07.1 Plan 03 — Region.mastering['ending_fx'] sidecar validation.

Makes ``Region.mastering['ending_fx']`` a first-class, validated sidecar field:

* It round-trips unchanged through ``save_sidecar`` → ``load_sidecar``.
* ``_validate_mastering_dict`` accepts ``ending_fx`` as an allowed stage (it is
  in ``chain._STAGE_ORDER`` from Plan 02), so a valid stage does NOT raise.
* ``_validate_stage_param_ranges`` range-checks the numeric ending_fx params
  (``tail_sec`` / ``wet`` / ``primary``) against the ``EndingFxStage`` Param
  min/max and rejects non-finite (NaN/inf) values — an out-of-range or
  non-finite value quarantines the sidecar at load (returns [], drops a
  ``.corrupt-*`` file).
* ``effect_type`` is ``kind="choice"`` (not numeric), so the numeric range loop
  skips it — the choice domain is enforced at the UI/builder layer, NOT here.
* Legacy sidecars with NO ``ending_fx`` key (or ``mastering=None``) load
  unchanged.
* The unknown-stage allowlist still rejects stage names not in _STAGE_ORDER
  (sanity that the allowlist was not loosened).

Mirrors the existing ``test_sidecar_normalize_field`` range-reject style
(compressor.ratio / normalize.target_db precedent).
"""

from __future__ import annotations

import json
from pathlib import Path

from marmelade.audio.mastering.stages import EndingFxStage
from marmelade.audio.sidecar_cache import (
    Region,
    SCHEMA_VERSION,
    SidecarValidationError,
    _validate_mastering_dict,
    load_sidecar,
    save_sidecar,
)

# A valid ending_fx stage dict (matches the EndingFxStage Param defaults /
# ranges). dub_echo is a real EFFECT_BUILDERS key; tail_sec/wet/primary are
# all in-range.
_VALID_ENDING_FX = {
    "enabled": True,
    "effect_type": "dub_echo",
    "tail_sec": 5.0,
    "wet": 1.0,
    "primary": 0.6,
}


def test_ending_fx_round_trips(tmp_path: Path) -> None:
    """A Region.mastering['ending_fx'] round-trips save→load with values/types preserved."""
    sidecar = tmp_path / "test.json"
    r = Region(
        id="id1234567890abcd",
        start_sec=0.0,
        end_sec=10.0,
        state="keeper",
        mastering={"ending_fx": dict(_VALID_ENDING_FX)},
    )
    save_sidecar(sidecar, [r])

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering is not None
    efx = loaded[0].mastering["ending_fx"]
    assert efx == _VALID_ENDING_FX
    # Types preserved (not silently coerced).
    assert efx["enabled"] is True
    assert isinstance(efx["effect_type"], str)
    assert isinstance(efx["tail_sec"], float)
    assert isinstance(efx["wet"], float)
    assert isinstance(efx["primary"], float)


def test_validate_mastering_dict_accepts_ending_fx() -> None:
    """_validate_mastering_dict does NOT raise for a valid ending_fx stage."""
    # Should not raise.
    _validate_mastering_dict({"ending_fx": dict(_VALID_ENDING_FX)})


def test_ending_fx_param_ranges_match_stage() -> None:
    """The valid fixture sits inside the EndingFxStage Param min/max bounds.

    Guards against the fixture drifting out of the declared ranges (which would
    make the 'accept' tests vacuous).
    """
    params = EndingFxStage().parameters()
    for key in ("tail_sec", "wet", "primary"):
        desc = params[key]
        val = float(_VALID_ENDING_FX[key])
        assert desc.min is not None and val >= float(desc.min)
        assert desc.max is not None and val <= float(desc.max)
    # effect_type is a choice param, not numeric.
    assert params["effect_type"].kind == "choice"


def test_out_of_range_tail_sec_quarantines(tmp_path: Path) -> None:
    """mastering.ending_fx.tail_sec above max (12.0) → quarantine."""
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
                "mastering": {
                    "ending_fx": {
                        "enabled": True,
                        "effect_type": "dub_echo",
                        "tail_sec": 1e9,
                        "wet": 1.0,
                        "primary": 0.6,
                    }
                },
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert loaded == []
    assert len(list(tmp_path.glob("test.json.corrupt-*"))) == 1


def test_non_finite_tail_sec_quarantines(tmp_path: Path) -> None:
    """mastering.ending_fx.tail_sec = NaN → quarantine (non-finite guard)."""
    sidecar = tmp_path / "test.json"
    raw = (
        '{"schema_version": %d, "regions": [{"id": "id1", '
        '"start_sec": 0.0, "end_sec": 1.0, "state": "keeper", '
        '"created_at": "x", "note": "", '
        '"mastering": {"ending_fx": {"enabled": true, '
        '"effect_type": "dub_echo", "tail_sec": NaN, '
        '"wet": 1.0, "primary": 0.6}}}]}'
        % SCHEMA_VERSION
    )
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(raw)

    loaded, _ = load_sidecar(sidecar)
    assert loaded == []
    assert len(list(tmp_path.glob("test.json.corrupt-*"))) == 1


def test_out_of_range_wet_quarantines(tmp_path: Path) -> None:
    """mastering.ending_fx.wet above max (1.0) → quarantine."""
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
                "mastering": {
                    "ending_fx": {
                        "enabled": True,
                        "effect_type": "dub_echo",
                        "tail_sec": 5.0,
                        "wet": 2.5,
                        "primary": 0.6,
                    }
                },
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert loaded == []
    assert len(list(tmp_path.glob("test.json.corrupt-*"))) == 1


def test_effect_type_choice_not_range_checked(tmp_path: Path) -> None:
    """effect_type is kind='choice' — the numeric range loop skips it.

    An arbitrary (even unknown) effect_type string does NOT quarantine at the
    sidecar layer: the choice domain is enforced at the UI/builder layer, and
    apply_ending_fx handles an unknown effect_type as a graceful no-op tail.
    The sidecar must round-trip whatever string is stored.
    """
    sidecar = tmp_path / "test.json"
    r = Region(
        id="id1234567890abcd",
        start_sec=0.0,
        end_sec=10.0,
        state="keeper",
        mastering={
            "ending_fx": {
                "enabled": True,
                "effect_type": "some_future_effect",
                "tail_sec": 5.0,
                "wet": 1.0,
                "primary": 0.6,
            }
        },
    )
    save_sidecar(sidecar, [r])
    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering["ending_fx"]["effect_type"] == "some_future_effect"
    assert list(tmp_path.glob("test.json.corrupt-*")) == []


def test_legacy_no_ending_fx_key_loads_unchanged(tmp_path: Path) -> None:
    """A mastering dict with other stages but NO ending_fx key loads unchanged."""
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
                "mastering": {"normalize": {"enabled": True, "target_db": 0.0}},
            }
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering is not None
    assert "ending_fx" not in loaded[0].mastering
    assert loaded[0].mastering["normalize"]["enabled"] is True
    assert list(tmp_path.glob("test.json.corrupt-*")) == []


def test_mastering_none_loads_cleanly(tmp_path: Path) -> None:
    """A fully legacy keeper with mastering=None loads as None, no crash."""
    sidecar = tmp_path / "test.json"
    r = Region(id="id1234567890abcd", start_sec=0.0, end_sec=1.0, state="keeper")
    save_sidecar(sidecar, [r])
    loaded, _ = load_sidecar(sidecar)
    assert len(loaded) == 1
    assert loaded[0].mastering is None


def test_unknown_stage_name_still_rejected() -> None:
    """A stage name NOT in _STAGE_ORDER still raises (allowlist not loosened)."""
    try:
        _validate_mastering_dict({"definitely_not_a_stage": {"enabled": True}})
    except SidecarValidationError:
        return
    raise AssertionError("expected SidecarValidationError for unknown stage name")
