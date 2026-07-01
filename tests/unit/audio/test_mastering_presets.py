"""quick-260623-p5b — genre mastering preset data tests.

Pins the pure-data contract of ``marmelade.audio.mastering.presets``:

  * names + locked lineup order ("Custom" NOT a member)
  * every preset is a COMPLETE 8-stage config mirroring _SESSION_DEFAULTS
  * every numeric param sits in its stage Param [min,max] range — proven by
    running the REAL ``_validate_stage_param_ranges`` on each preset config
  * match_preset round-trips each preset (config_hash) and returns None for
    an edited / unrelated config
  * the 10 config_hashes are mutually distinct
"""

from __future__ import annotations

import pytest

from marmelade.audio.mastering import (
    MASTERING_PRESETS,
    match_preset,
    preset_names,
)
from marmelade.audio.mastering.chain import _SESSION_DEFAULTS, config_hash
from marmelade.audio.sidecar_cache import _validate_stage_param_ranges

_LOCKED_ORDER = [
    "House",
    "Techno",
    "Dubstep",
    "Drum & Bass",
    "Trance",
    "EDM / Festival",
    "Hip-Hop / Trap",
    "Lo-fi",
    "Pop",
    "Ambient",
]


def test_preset_names_locked_order() -> None:
    """preset_names() returns exactly the 10 names in the locked order."""
    assert preset_names() == _LOCKED_ORDER
    # "Custom" is the dock sentinel, NOT a preset member.
    assert "Custom" not in preset_names()


def test_each_preset_is_complete_8_stage_config() -> None:
    """Every GENRE preset carries the 8 SESSION-chain stage keys.

    Phase 07.1 added ``ending_fx`` to ``_SESSION_DEFAULTS`` (so the per-keeper
    MasteringDialog + sidecar allowlist know the stage), but it is a
    PER-KEEPER-only stage — genre (session) presets do NOT carry it
    (07.1-CONTEXT deferred: no session-wide ending FX). quick-260625 adds
    ``vst3`` on the same footing (per-keeper-only external plugin slot), so the
    genre-preset stage set is ``_SESSION_DEFAULTS`` minus ``ending_fx``/``vst3``.
    quick-260626-o9y adds ``fade`` on the same footing — it is an output-time
    per-keeper stage that genre presets never carry — so it is also excluded.
    """
    session_stages = {
        s for s in _SESSION_DEFAULTS.keys() if s not in ("ending_fx", "vst3", "fade")
    }
    assert session_stages == {
        "normalize",
        "loudness",
        "highpass",
        "lowpass",
        "eq",
        "compressor",
        "limiter",
        "matchering",
    }
    for name, cfg in MASTERING_PRESETS.items():
        assert set(cfg.keys()) == session_stages, name
        for stage in session_stages:
            default_cfg = _SESSION_DEFAULTS[stage]
            # Each stage must carry the SAME param key set as the default.
            assert set(cfg[stage].keys()) == set(default_cfg.keys()), (
                f"{name}/{stage} param keys mismatch"
            )
            assert "enabled" in cfg[stage], f"{name}/{stage} missing enabled"


def test_common_stage_invariants() -> None:
    """The locked common-to-all invariants hold for every preset."""
    for name, cfg in MASTERING_PRESETS.items():
        assert cfg["highpass"]["enabled"] is True, name
        assert cfg["eq"]["enabled"] is True, name
        assert cfg["compressor"]["enabled"] is True, name
        assert cfg["limiter"]["enabled"] is True, name
        assert cfg["limiter"]["ceiling_dbtp"] == pytest.approx(-1.0), name
        assert cfg["limiter"]["release_ms"] == pytest.approx(100.0), name
        assert cfg["matchering"]["enabled"] is False, name
        assert cfg["matchering"]["reference_path"] == "", name
        assert cfg["normalize"]["enabled"] is False, name
        assert cfg["normalize"]["target_db"] == pytest.approx(0.0), name
        # lowpass OFF for all except Lo-fi.
        if name == "Lo-fi":
            assert cfg["lowpass"]["enabled"] is True
            assert cfg["lowpass"]["cutoff_hz"] == pytest.approx(16000.0)
        else:
            assert cfg["lowpass"]["enabled"] is False, name
        # loudness ON for all except Ambient (preserve dynamics).
        if name == "Ambient":
            assert cfg["loudness"]["enabled"] is False
        else:
            assert cfg["loudness"]["enabled"] is True, name


@pytest.mark.parametrize("name", _LOCKED_ORDER)
def test_preset_params_in_range(name: str) -> None:
    """Every preset config passes the REAL _validate_stage_param_ranges.

    T-p5b-01 — an out-of-range authored value fails the build here.
    """
    # Must not raise (would raise SidecarValidationError otherwise).
    _validate_stage_param_ranges(MASTERING_PRESETS[name])


@pytest.mark.parametrize("name", _LOCKED_ORDER)
def test_match_preset_round_trips(name: str) -> None:
    """match_preset(MASTERING_PRESETS[name]) == name for all 10."""
    assert match_preset(MASTERING_PRESETS[name]) == name


def test_match_preset_off_by_one_returns_none() -> None:
    """An enabled-param edit off-by-one breaks the match → None (Custom)."""
    import copy

    edited = copy.deepcopy(MASTERING_PRESETS["House"])
    edited["eq"]["low_db"] = 1.6  # was 1.5
    assert match_preset(edited) is None


def test_match_preset_empty_config_returns_none() -> None:
    """An unrelated / empty config matches no preset."""
    assert match_preset({}) is None
    # The empty-QSettings default (limiter-only) matches no preset either.
    assert match_preset(_SESSION_DEFAULTS) is None


def test_preset_hashes_are_distinct() -> None:
    """The 10 preset config_hashes are mutually distinct."""
    hashes = [config_hash(cfg) for cfg in MASTERING_PRESETS.values()]
    assert len(set(hashes)) == len(hashes)
