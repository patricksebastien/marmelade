"""FadeStage + fade_params() unit coverage (quick-260626-o9y).

Fade is an OUTPUT-TIME virtual stage (applied at export/preview, never baked
into the mastered cache, dropped unconditionally from config_hash). This file
pins:

* FadeStage.parameters() shape (single ``duration_sec`` Param) + identity.
* build_plugin() raises (mirrors NormalizeStage — never a pedalboard plugin).
* fade_params() default / with-values / partial-dict behavior — the SINGLE
  source of truth read by export + preview.
* sidecar validation: fade.duration_sec range-checked to [0.0, 10.0]; legacy
  sidecars with no "fade" key still validate; a present valid fade passes.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# FadeStage parameters() + build_plugin()
# ---------------------------------------------------------------------------


def test_fade_stage_identity():
    from marmelade.audio.mastering.stages.fade import FadeStage

    assert FadeStage.name == "fade"
    assert FadeStage.display_name == "Fade in/out"


def test_fade_stage_parameters_shape():
    from marmelade.audio.mastering.stages.fade import FadeStage

    params = FadeStage().parameters()
    assert set(params.keys()) == {"duration_sec"}
    p = params["duration_sec"]
    assert p.kind == "float"
    assert p.default == 2.0
    assert p.min == 0.0
    assert p.max == 10.0
    assert p.step == 0.5
    assert p.unit == "s"
    # Output-time fade must NOT bust the mastered cache.
    assert p.requires_recompute is False


def test_fade_stage_build_plugin_raises():
    from marmelade.audio.mastering.stages.fade import FadeStage

    with pytest.raises(NotImplementedError):
        FadeStage().build_plugin()


def test_fade_stage_zero_qt_imports():
    """N-3: stages/fade.py must pull in no Qt."""
    import sys

    # Import fresh and assert no PySide6 was dragged in by THIS module's import.
    import marmelade.audio.mastering.stages.fade as fade_mod

    src = fade_mod.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    assert "PySide6" not in text
    assert "PyQt" not in text
    assert "QtWidgets" not in text
    assert "QtGui" not in text


# ---------------------------------------------------------------------------
# fade_params() — single source of truth for (enabled, duration_sec)
# ---------------------------------------------------------------------------


def test_fade_params_default_none():
    from marmelade.audio.mastering.stages.fade import fade_params

    assert fade_params(None) == (True, 2.0)


def test_fade_params_default_empty_dict():
    from marmelade.audio.mastering.stages.fade import fade_params

    assert fade_params({}) == (True, 2.0)


def test_fade_params_missing_fields_default():
    from marmelade.audio.mastering.stages.fade import fade_params

    assert fade_params({"fade": {}}) == (True, 2.0)


def test_fade_params_disabled_with_duration():
    from marmelade.audio.mastering.stages.fade import fade_params

    assert fade_params({"fade": {"enabled": False, "duration_sec": 5.0}}) == (
        False,
        5.0,
    )


def test_fade_params_enabled_zero_duration():
    from marmelade.audio.mastering.stages.fade import fade_params

    assert fade_params({"fade": {"enabled": True, "duration_sec": 0.0}}) == (
        True,
        0.0,
    )


def test_fade_params_coerces_types():
    from marmelade.audio.mastering.stages.fade import fade_params

    enabled, dur = fade_params({"fade": {"enabled": 1, "duration_sec": "3"}})
    assert enabled is True
    assert isinstance(dur, float)
    assert dur == 3.0


# ---------------------------------------------------------------------------
# Sidecar validation — range-check + legacy-absent
# ---------------------------------------------------------------------------


def test_sidecar_fade_out_of_range_rejected():
    from marmelade.audio.sidecar_cache import (
        SidecarValidationError,
        _validate_stage_param_ranges,
    )

    with pytest.raises(SidecarValidationError):
        _validate_stage_param_ranges(
            {"fade": {"enabled": True, "duration_sec": 99.0}}
        )


def test_sidecar_fade_valid_passes():
    from marmelade.audio.sidecar_cache import _validate_stage_param_ranges

    # Must NOT raise.
    _validate_stage_param_ranges({"fade": {"enabled": True, "duration_sec": 2.0}})


def test_sidecar_fade_absent_stays_valid():
    from marmelade.audio.sidecar_cache import _validate_stage_param_ranges

    # Legacy sidecar with no "fade" key.
    _validate_stage_param_ranges(
        {"limiter": {"enabled": True, "ceiling_dbtp": -1.0}}
    )


def test_validate_mastering_dict_accepts_fade():
    from marmelade.audio.sidecar_cache import _validate_mastering_dict

    # Must NOT raise.
    _validate_mastering_dict({"fade": {"enabled": True, "duration_sec": 2.0}})
