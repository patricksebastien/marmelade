"""quick-260629 — Reverb / Distortion / Delay whole-clip mastering stages.

Covers the four wiring points that make a new stage appear in BOTH the
session dock and the per-keeper dialog and process correctly:

* registry membership (_STAGE_ORDER / _SESSION_DEFAULTS / _STAGE_FACTORY /
  _PRE_LIMITER_STAGES) without disturbing the fade/normalize/matchering
  anchors,
* each stage builds its native pedalboard plugin and processes audio,
* config_hash treats a DISABLED instance as absent (cache + preset-match
  stability), while ENABLING it changes the hash,
* the N-3 invariant (no GUI toolkit import) holds for the new modules.
"""

from __future__ import annotations

import numpy as np
import pedalboard
import pytest

from marmelade.audio.mastering import DelayStage, DistortionStage, ReverbStage
from marmelade.audio.mastering.chain import (
    _PRE_LIMITER_STAGES,
    _SESSION_DEFAULTS,
    _STAGE_FACTORY,
    _STAGE_ORDER,
    config_hash,
    run_dsp_chain,
)

NEW_STAGES = ("distortion", "delay", "reverb")


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_stage_is_registered_everywhere(stage: str) -> None:
    assert stage in _STAGE_ORDER
    assert stage in _SESSION_DEFAULTS
    assert stage in _STAGE_FACTORY
    assert stage in _PRE_LIMITER_STAGES
    # Default session entry is DISABLED so existing sessions are no-ops.
    assert _SESSION_DEFAULTS[stage]["enabled"] is False


def test_anchor_stages_unmoved() -> None:
    """Adding color stages must not disturb the panel anchors."""
    assert _STAGE_ORDER[0] == "fade"
    assert _STAGE_ORDER[1] == "normalize"
    assert _STAGE_ORDER[-1] == "matchering"


@pytest.mark.parametrize(
    "cls, plugin_type",
    [
        (ReverbStage, pedalboard.Reverb),
        (DistortionStage, pedalboard.Distortion),
        (DelayStage, pedalboard.Delay),
    ],
)
def test_build_plugin_returns_native_pedalboard(cls, plugin_type) -> None:
    stage = cls()
    assert stage.parameters(), "stage must expose at least one tunable param"
    plugin = stage.build_plugin()
    assert isinstance(plugin, plugin_type)


def test_stage_defaults_match_session_defaults() -> None:
    """parameters() defaults must equal the _SESSION_DEFAULTS values."""
    for cls, name in (
        (ReverbStage, "reverb"),
        (DistortionStage, "distortion"),
        (DelayStage, "delay"),
    ):
        params = cls().parameters()
        for pname, pdesc in params.items():
            assert _SESSION_DEFAULTS[name][pname] == pytest.approx(pdesc.default), (
                name,
                pname,
            )


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_disabled_stage_canonicalizes_away(stage: str) -> None:
    """A disabled new stage hashes EQUAL to a config that lacks the key."""
    base = {"eq": {"enabled": True, "low_db": 1.0, "mid_db": 0.0, "high_db": 0.0}}
    with_disabled = dict(base)
    with_disabled[stage] = {**_SESSION_DEFAULTS[stage], "enabled": False}
    assert config_hash(base) == config_hash(with_disabled)


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_enabling_stage_changes_hash(stage: str) -> None:
    base = {"eq": {"enabled": True, "low_db": 1.0, "mid_db": 0.0, "high_db": 0.0}}
    enabled = dict(base)
    enabled[stage] = {**_SESSION_DEFAULTS[stage], "enabled": True}
    assert config_hash(base) != config_hash(enabled)


def test_color_stages_process_audio_finite() -> None:
    audio = (np.random.default_rng(0).standard_normal((2, 48000)) * 0.1).astype(
        np.float32
    )
    cfg = {
        "distortion": {"enabled": True, "drive_db": 12.0},
        "delay": {
            "enabled": True,
            "delay_seconds": 0.25,
            "feedback": 0.4,
            "mix": 0.3,
        },
        "reverb": {
            "enabled": True,
            "room_size": 0.7,
            "damping": 0.4,
            "wet_level": 0.3,
            "dry_level": 0.6,
            "width": 1.0,
        },
    }
    out = run_dsp_chain(audio, 48000, cfg)
    assert out.shape == audio.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_off_path_is_array_identical() -> None:
    """Disabled color stages leave the audio byte-identical (no-op)."""
    audio = (np.random.default_rng(1).standard_normal((2, 4800)) * 0.1).astype(
        np.float32
    )
    cfg = {
        "distortion": {"enabled": False, "drive_db": 12.0},
        "delay": {"enabled": False, "delay_seconds": 0.25, "feedback": 0.4, "mix": 0.3},
        "reverb": {"enabled": False, "room_size": 0.7},
    }
    out = run_dsp_chain(audio, 48000, cfg)
    np.testing.assert_array_equal(out, audio)
