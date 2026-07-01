"""Tests for ending_fx_presets — Phase 07.1 Plan 01 Task 2.

10 curated ending-FX presets (pure data) + a two-way ``match_ending_fx``
matcher mirroring the genre-preset module. The LOCKED input-shape contract:
``match_ending_fx`` takes the BARE ending_fx stage dict — NOT a
``{"ending_fx": ...}`` wrapper — exactly what Plan 04 passes via
``match_ending_fx(self._cfg.get("ending_fx", {}))``. Any config_hash
canonicalization/wrapping lives INSIDE the function.
"""

from __future__ import annotations

import pytest

from marmelade.audio.mastering.ending_fx_presets import (
    ENDING_FX_PRESETS,
    ending_fx_preset_names,
    match_ending_fx,
)
from marmelade.audio.mastering.stages.ending_fx import EFFECT_BUILDERS

# Locked lineup order.
LINEUP = [
    "Hall Wash",
    "Dub Echo",
    "Tape Stop",
    "Filter Close",
    "Shimmer Freeze",
    "Bitcrush Collapse",
    "Codec Rot",
    "Glitch Stutter",
    "Overdrive Bloom",
    "Smear",
]


def test_exactly_ten_presets_in_lineup_order():
    assert list(ENDING_FX_PRESETS.keys()) == LINEUP
    assert len(ENDING_FX_PRESETS) == 10


def test_preset_names_returns_lineup_without_custom():
    names = ending_fx_preset_names()
    assert names == LINEUP
    assert "Custom" not in names


@pytest.mark.parametrize("name", LINEUP)
def test_each_preset_is_a_valid_ending_fx_stage_dict(name):
    cfg = ENDING_FX_PRESETS[name]
    assert cfg["enabled"] is True
    assert cfg["effect_type"] in EFFECT_BUILDERS
    assert 0.5 <= cfg["tail_sec"] <= 12.0
    assert 0.0 <= cfg["wet"] <= 1.0
    assert 0.0 <= cfg["primary"] <= 1.0
    # All numeric values are floats so config_hash 6dp rounding round-trips.
    assert isinstance(cfg["tail_sec"], float)
    assert isinstance(cfg["wet"], float)
    assert isinstance(cfg["primary"], float)
    assert isinstance(cfg["onset_sec"], float)


@pytest.mark.parametrize("name", LINEUP)
def test_each_preset_has_onset_sec_in_range(name):
    cfg = ENDING_FX_PRESETS[name]
    assert isinstance(cfg["onset_sec"], float)
    assert 0.1 <= cfg["onset_sec"] <= 8.0


@pytest.mark.parametrize("name", LINEUP)
def test_onset_sec_round_trips_through_match(name):
    # Adding onset_sec to the schema must not break the two-way matcher.
    assert match_ending_fx(ENDING_FX_PRESETS[name]) == name


def test_onset_sec_param_range_validated_by_sidecar():
    # The sidecar validator reads EndingFxStage Param min/max automatically;
    # assert the declared range is [0.1, 8.0] (proven wired in Task 1).
    from marmelade.audio.mastering.stages.ending_fx import EndingFxStage

    onset = EndingFxStage().parameters()["onset_sec"]
    assert onset.min == 0.1
    assert onset.max == 8.0


def test_each_effect_type_maps_one_to_one_to_builders():
    used = {cfg["effect_type"] for cfg in ENDING_FX_PRESETS.values()}
    assert used == set(EFFECT_BUILDERS.keys())
    assert len(used) == 10  # no two presets share an effect_type


# ---------------------------------------------------------------------------
# match_ending_fx — BARE stage dict in (locked), two-way round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LINEUP)
def test_each_preset_round_trips_to_its_own_name(name):
    # Pass the preset's BARE stage dict directly — no {"ending_fx": ...} wrapper.
    assert match_ending_fx(ENDING_FX_PRESETS[name]) == name


def test_hall_wash_round_trip_explicit():
    assert match_ending_fx(ENDING_FX_PRESETS["Hall Wash"]) == "Hall Wash"


def test_non_matching_dict_returns_none_custom():
    # Same effect_type but an off-grid tail_sec → no preset matches → Custom.
    cfg = {
        "enabled": True,
        "effect_type": "hall_wash",
        "tail_sec": 99.0,  # not a preset value (and out of the curated grid)
        "wet": 1.0,
        "primary": 0.5,
    }
    assert match_ending_fx(cfg) is None


def test_empty_dict_returns_none():
    assert match_ending_fx({}) is None


def test_disabled_dict_returns_none():
    assert match_ending_fx({"enabled": False}) is None
    # A disabled stage carrying a preset's params still does not match (a
    # disabled stage canonicalizes to {"enabled": False}).
    disabled_hall = dict(ENDING_FX_PRESETS["Hall Wash"], enabled=False)
    assert match_ending_fx(disabled_hall) is None


# ---------------------------------------------------------------------------
# Re-exports from audio.mastering
# ---------------------------------------------------------------------------


def test_symbols_importable_from_audio_mastering():
    from marmelade.audio.mastering import (
        ENDING_FX_PRESETS as P,
        EndingFxStage,
        ending_fx_preset_names as names_fn,
        match_ending_fx as match_fn,
    )

    assert len(P) == 10
    assert EndingFxStage.name == "ending_fx"
    assert names_fn() == LINEUP
    assert match_fn(P["Smear"]) == "Smear"


def test_ending_fx_stage_importable_from_stages():
    from marmelade.audio.mastering.stages import EndingFxStage

    assert EndingFxStage.name == "ending_fx"
