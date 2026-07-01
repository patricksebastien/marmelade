"""Tests for apply_ending_fx + EndingFxStage — Phase 07.1 Plan 01 Task 1.

The pure, Qt-free DSP core of Per-Keeper Ending FX. apply_ending_fx appends
``tail_sec`` of ring-out room, applies a per-effect pedalboard chain, mixes
wet/dry, and finishes with a short safety fade so the file always ends in
TRUE silence. EndingFxStage declares the tunable surface (effect_type,
tail_sec, wet, primary) and — like LoudnessStage — raises in build_plugin
(it is applied via a function, not a single pedalboard plugin).

Two locked contracts pinned here:
  * An UNKNOWN effect_type is a graceful dry-with-tail no-op — never raises
    in the render worker, applies NO effect.
  * The off path (disabled / None / no effect_type) is byte-identical.
"""

from __future__ import annotations

import numpy as np
import pedalboard
import pytest

from marmelade.audio.mastering.stages.ending_fx import (
    EFFECT_BUILDERS,
    EndingFxStage,
    apply_ending_fx,
)

SR = 48000

# The 10 curated effect_type ids (locked lineup, Task 2 presets map 1:1).
ALL_EFFECT_TYPES = [
    "hall_wash",
    "dub_echo",
    "tape_stop",
    "filter_close",
    "shimmer_freeze",
    "bitcrush_collapse",
    "codec_rot",
    "glitch_stutter",
    "overdrive_bloom",
    "smear",
]


def _signal(seconds: float = 1.0, channels: int = 2) -> np.ndarray:
    """A non-silent float32 stereo test signal (low-level pink-ish noise)."""
    n = int(seconds * SR)
    rng = np.random.default_rng(1234)
    return (rng.standard_normal((channels, n)).astype(np.float32)) * 0.1


# ---------------------------------------------------------------------------
# Off path — byte-identical pass-through
# ---------------------------------------------------------------------------


def test_disabled_returns_input_byte_identical():
    audio = _signal()
    out = apply_ending_fx(audio, SR, {"enabled": False, "effect_type": "hall_wash"})
    assert out.dtype == np.float32
    assert out.shape == audio.shape
    assert np.array_equal(out, audio)


def test_none_cfg_returns_input_byte_identical():
    audio = _signal()
    out = apply_ending_fx(audio, SR, None)
    assert np.array_equal(out, audio)


def test_missing_effect_type_returns_input_byte_identical():
    audio = _signal()
    out = apply_ending_fx(audio, SR, {"enabled": True, "tail_sec": 4.0})
    assert np.array_equal(out, audio)


def test_empty_effect_type_is_off_path():
    """An empty-string effect_type is the unset/off sentinel → byte-identical."""
    audio = _signal()
    out = apply_ending_fx(audio, SR, {"enabled": True, "effect_type": "", "tail_sec": 4.0})
    assert np.array_equal(out, audio)


# ---------------------------------------------------------------------------
# Enabled path — longer, silence-terminated, float32, all 10 effects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effect_type", ALL_EFFECT_TYPES)
def test_each_effect_longer_and_silence_terminated(effect_type):
    audio = _signal()
    tail_sec = 4.0
    cfg = {
        "enabled": True,
        "effect_type": effect_type,
        "tail_sec": tail_sec,
        "wet": 1.0,
        "primary": 0.5,
    }
    out = apply_ending_fx(audio, SR, cfg)

    assert out.dtype == np.float32
    assert out.ndim == 2
    assert out.shape[0] == audio.shape[0]  # channels preserved
    # Longer than input by ~tail_sec (resampling effects may shift a few samples)
    assert out.shape[1] >= audio.shape[1] + int(tail_sec * SR * 0.9)
    # Last ~5 ms is true silence (safety fade guarantees it)
    tail = out[:, -int(0.005 * SR):]
    assert np.abs(tail).max() < 1e-3
    # Very last sample is exactly zero
    assert out[:, -1].max() == 0.0


def test_sr_is_honored_no_internal_48k_assumption():
    """Effects are built at the passed sr; tail length scales with sr."""
    audio = _signal(seconds=0.5)
    sr = 44100
    cfg = {"enabled": True, "effect_type": "hall_wash", "tail_sec": 2.0, "wet": 1.0, "primary": 0.5}
    out = apply_ending_fx(audio, sr, cfg)
    assert out.shape[1] >= audio.shape[1] + int(2.0 * sr * 0.9)


def test_output_dtype_is_float32_even_from_float64_input():
    audio = _signal().astype(np.float64)
    cfg = {"enabled": True, "effect_type": "hall_wash", "tail_sec": 2.0, "wet": 1.0, "primary": 0.5}
    out = apply_ending_fx(audio, SR, cfg)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Unknown effect_type — graceful dry-with-tail no-op (locked contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bogus", ["bogus", "legacy_ir_load", "convolution", "HALL_WASH"])
def test_unknown_effect_type_graceful_no_op(bogus):
    """A hostile/legacy effect_type must never KeyError; it pass-throughs dry."""
    audio = _signal()
    tail_sec = 3.0
    cfg = {
        "enabled": True,
        "effect_type": bogus,
        "tail_sec": tail_sec,
        "wet": 1.0,
        "primary": 0.5,
    }
    # Must not raise.
    out = apply_ending_fx(audio, SR, cfg)

    # Build the reference: dry input padded with tail_room zeros, then the
    # SAME safety fade applied. The unknown path applies NO effect.
    tail_room = int(round(tail_sec * SR))
    padded = np.concatenate(
        [audio.astype(np.float32), np.zeros((audio.shape[0], tail_room), np.float32)],
        axis=1,
    )
    expected = _reference_dry_with_tail(padded, SR)

    assert out.shape == expected.shape
    assert out.shape[1] == audio.shape[1] + tail_room
    # True silence at end
    assert np.abs(out[:, -int(0.005 * SR):]).max() < 1e-3
    # Equals the dry-with-tail signal — no partial effect applied.
    assert np.allclose(out, expected, atol=1e-6)


def _reference_dry_with_tail(padded: np.ndarray, sr: int) -> np.ndarray:
    """Mirror apply_ending_fx's two-ramp safety fade on a dry padded array."""
    out = padded.astype(np.float32).copy()
    n = out.shape[1]
    long_len = int(min(0.05 * sr, n))
    if long_len > 0:
        out[:, -long_len:] *= np.linspace(1.0, 0.0, long_len, dtype=np.float32)
    short_len = int(min(0.01 * sr, n))
    if short_len > 0:
        out[:, -short_len:] *= np.linspace(1.0, 0.0, short_len, dtype=np.float32)
    out[:, -1] = 0.0
    return out


# ---------------------------------------------------------------------------
# Time-varying wet ENVELOPE — body stays dry, tail rings out (the bug fix)
# ---------------------------------------------------------------------------


def test_body_stays_dry_before_onset():
    """The keeper BODY before the onset window is value-identical to dry input."""
    audio = _signal(seconds=4.0)
    onset_sec = 2.0
    cfg = {
        "enabled": True,
        "effect_type": "hall_wash",
        "tail_sec": 4.0,
        "wet": 1.0,
        "primary": 0.5,
        "onset_sec": onset_sec,
    }
    out = apply_ending_fx(audio, SR, cfg)

    body_len = audio.shape[1]
    onset_n = int(onset_sec * SR)
    dry_region = body_len - onset_n
    assert dry_region > 0
    # The dry body before the onset ramp must be untouched.
    assert np.allclose(out[:, :dry_region], audio[:, :dry_region], atol=1e-6)


def test_effect_present_near_end_and_tail():
    """The appended tail is wet (rings out) and the onset window blended in."""
    audio = _signal(seconds=4.0)
    onset_sec = 2.0
    tail_sec = 4.0
    cfg = {
        "enabled": True,
        "effect_type": "hall_wash",
        "tail_sec": tail_sec,
        "wet": 1.0,
        "primary": 0.5,
        "onset_sec": onset_sec,
    }
    out = apply_ending_fx(audio, SR, cfg)

    body_len = audio.shape[1]
    tail_room = int(round(tail_sec * SR))

    # Dry-with-tail reference (no effect) — the tail would be pure silence.
    padded = np.concatenate(
        [audio.astype(np.float32), np.zeros((audio.shape[0], tail_room), np.float32)],
        axis=1,
    )
    expected_dry = _reference_dry_with_tail(padded, SR)

    # The tail region (before the final safety-fade window) must be NON-silent:
    # the reverb wash rings out into it.
    tail_region = out[:, body_len : body_len + tail_room - int(0.1 * SR)]
    assert np.abs(tail_region).max() > 1e-4

    # SOME sample in the last onset window of the body differs from dry — the
    # ramp blended the effect in before the region end.
    onset_n = int(onset_sec * SR)
    onset_window_out = out[:, body_len - onset_n : body_len]
    onset_window_dry = audio[:, body_len - onset_n : body_len]
    assert np.abs(onset_window_out - onset_window_dry).max() > 1e-4

    # The wet tail differs from the silent dry-with-tail reference tail.
    assert not np.allclose(
        out[:, body_len : body_len + tail_room - int(0.1 * SR)],
        expected_dry[:, body_len : body_len + tail_room - int(0.1 * SR)],
        atol=1e-4,
    )


def test_short_keeper_ramps_over_whole_body():
    """onset_sec longer than the body clamps to body_len; no raise, ends silent."""
    audio = _signal(seconds=0.5)
    onset_sec = 2.0  # longer than the 0.5 s body
    tail_sec = 4.0
    cfg = {
        "enabled": True,
        "effect_type": "hall_wash",
        "tail_sec": tail_sec,
        "wet": 1.0,
        "primary": 0.5,
        "onset_sec": onset_sec,
    }
    out = apply_ending_fx(audio, SR, cfg)

    body_len = audio.shape[1]
    tail_room = int(round(tail_sec * SR))
    assert out.shape[1] == body_len + tail_room
    # Ends in true silence.
    assert np.abs(out[:, -int(0.005 * SR):]).max() < 1e-3
    assert out[:, -1].max() == 0.0


def test_stage_declares_onset_sec_param():
    params = EndingFxStage().parameters()
    assert "onset_sec" in params
    onset = params["onset_sec"]
    assert onset.kind == "float"
    assert onset.min == 0.1
    assert onset.max == 8.0
    assert onset.default == 2.0
    assert onset.requires_recompute is True


# ---------------------------------------------------------------------------
# EFFECT_BUILDERS structure
# ---------------------------------------------------------------------------


def test_effect_builders_has_exactly_ten_known_ids():
    assert set(EFFECT_BUILDERS.keys()) == set(ALL_EFFECT_TYPES)
    assert len(EFFECT_BUILDERS) == 10


@pytest.mark.parametrize("effect_type", ALL_EFFECT_TYPES)
def test_builder_returns_list_of_plugins(effect_type):
    plugins = EFFECT_BUILDERS[effect_type](0.5)
    assert isinstance(plugins, list)
    assert len(plugins) >= 1
    for p in plugins:
        assert isinstance(p, pedalboard.Plugin)


# ---------------------------------------------------------------------------
# EndingFxStage declaration
# ---------------------------------------------------------------------------


def test_stage_name_and_display_name():
    assert EndingFxStage.name == "ending_fx"
    assert EndingFxStage.display_name


def test_stage_declares_four_params_with_ranges():
    params = EndingFxStage().parameters()
    assert set(params.keys()) == {"effect_type", "tail_sec", "wet", "primary", "onset_sec"}

    effect = params["effect_type"]
    assert effect.kind == "choice"
    assert set(effect.choices) >= set(ALL_EFFECT_TYPES)
    assert effect.default in effect.choices

    tail = params["tail_sec"]
    assert tail.kind == "float"
    assert tail.min == 0.5
    assert tail.max == 12.0

    wet = params["wet"]
    assert wet.kind == "float"
    assert wet.min == 0.0
    assert wet.max == 1.0

    primary = params["primary"]
    assert primary.kind == "float"
    assert primary.min == 0.0
    assert primary.max == 1.0

    # All require a re-render.
    for p in params.values():
        assert p.requires_recompute is True


def test_build_plugin_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        EndingFxStage().build_plugin()
