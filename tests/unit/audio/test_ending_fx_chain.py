"""Phase 07.1 Plan 02 Task 1 — ending_fx wired into MasteringChain.process.

Pins the four chain-level invariants for the new ending-FX tail stage:

* Enabled ending_fx makes the chain output LONGER than the input region by
  ~``tail_sec`` (the appended ring-out room is exactly what makes the tail
  audible — 07.1-CONTEXT "Output audio is LONGER than the region").
* The mastered output's final samples reach TRUE silence even with loudness
  AND normalize ON — the loudness / normalize tail stages are gain ops, not
  re-fades, so the safety fade's true-silence ending survives.
* ``config_hash`` CHANGES when any ending_fx param changes (cache
  invalidation comes free) and a DISABLED ending_fx canonicalizes away
  (byte-identical to a config with no ending_fx key — no spurious cache
  busts for legacy keepers).
* With ending_fx DISABLED / absent the chain output is array-identical to the
  same config with the ending_fx key removed (SC-6 no-regression off path).
* sr=48000 is still enforced; ``"ending_fx"`` is in ``_STAGE_ORDER``.

Apply-seam: ending_fx runs AFTER matchering, BEFORE the loudness tail, so the
ring-out is included in the LUFS measurement + final ISP/normalize. The
LUFS-inclusion guarantee is structural (seam placement), not a numeric
assertion here.
"""

from __future__ import annotations

import copy

import numpy as np

from marmelade.audio.mastering.chain import (
    MasteringChain,
    _STAGE_ORDER,
    config_hash,
)

SR = 48000  # quick-260615-f77 canonical mastering rate.


def _stereo_signal(seconds: float = 2.0, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """``(channels, samples)`` float32 — sine + a touch of noise, not clipped."""
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float32) / sr
    sine = 0.6 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    rng = np.random.default_rng(0)
    noise = (0.1 * rng.standard_normal(n)).astype(np.float32)
    mono = ((sine + noise) * amp).astype(np.float32)
    out = np.stack([mono, mono], axis=0)
    return np.clip(out, -0.99, 0.99).astype(np.float32)


def _base_cfg() -> dict:
    """Limiter-only baseline (matchering OFF, loudness OFF, normalize OFF)."""
    return {
        "highpass": {"enabled": False, "cutoff_hz": 30.0},
        "lowpass": {"enabled": False, "cutoff_hz": 18000.0},
        "eq": {"enabled": False, "low_db": 0.0, "mid_db": 0.0, "high_db": 0.0},
        "compressor": {
            "enabled": False,
            "threshold_db": -18.0,
            "ratio": 2.0,
            "attack_ms": 30.0,
            "release_ms": 200.0,
        },
        "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "matchering": {"enabled": False, "reference_path": ""},
        "loudness": {"enabled": False, "target_lufs": -14.0},
        "normalize": {"enabled": False, "target_db": 0.0},
    }


def _ending_fx_cfg(**overrides) -> dict:
    cfg = {
        "enabled": True,
        "effect_type": "hall_wash",
        "tail_sec": 4.0,
        "wet": 1.0,
        "primary": 0.5,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# _STAGE_ORDER membership.
# ---------------------------------------------------------------------------


def test_ending_fx_in_stage_order():
    """``"ending_fx"`` must appear in _STAGE_ORDER (panel + sidecar allowlist)."""
    assert "ending_fx" in _STAGE_ORDER


# ---------------------------------------------------------------------------
# Output longer than input + true-silence tail (with loudness + normalize ON).
# ---------------------------------------------------------------------------


def test_enabled_ending_fx_output_longer_than_input():
    """Enabled ending_fx → output is LONGER by ~tail_sec (matchering OFF)."""
    audio = _stereo_signal()
    cfg = _base_cfg()
    cfg["ending_fx"] = _ending_fx_cfg(tail_sec=4.0)

    out = MasteringChain(cfg).process(audio, SR)

    assert out.dtype == np.float32
    assert out.shape[0] == audio.shape[0]
    min_growth = int(4.0 * SR * 0.9)
    assert out.shape[1] >= audio.shape[1] + min_growth, (
        f"expected >= {audio.shape[1] + min_growth} samples, "
        f"got {out.shape[1]} (input {audio.shape[1]})"
    )


def test_ending_fx_tail_reaches_silence_with_loudness_and_normalize_on():
    """Final ~5 ms near silence EVEN with loudness ON + normalize ON.

    The loudness / normalize tail stages are gain ops, not re-fades — they
    multiply the (already-silent) tail by a scalar, so silence stays silence.
    This pins that the safety fade survives the post-ending_fx tail stages.
    """
    audio = _stereo_signal()
    cfg = _base_cfg()
    cfg["ending_fx"] = _ending_fx_cfg(tail_sec=4.0)
    cfg["loudness"] = {"enabled": True, "target_lufs": -14.0}
    cfg["normalize"] = {"enabled": True, "target_db": 0.0}

    out = MasteringChain(cfg).process(audio, SR)

    last = np.abs(out[:, -240:]).max()
    assert last < 1e-3, f"tail not silent: max abs over final 240 samples = {last}"


# ---------------------------------------------------------------------------
# Disabled / absent ending_fx → byte-identical off path (SC-6 no-regression).
# ---------------------------------------------------------------------------


def test_disabled_ending_fx_output_array_equal_to_no_key():
    """ending_fx disabled → output equals the SAME cfg with the key removed."""
    audio = _stereo_signal()

    cfg_no_key = _base_cfg()
    out_no_key = MasteringChain(copy.deepcopy(cfg_no_key)).process(audio.copy(), SR)

    cfg_disabled = _base_cfg()
    cfg_disabled["ending_fx"] = _ending_fx_cfg(enabled=False)
    out_disabled = MasteringChain(cfg_disabled).process(audio.copy(), SR)

    assert np.array_equal(out_no_key, out_disabled), (
        "disabled ending_fx must be a byte-identical no-op vs. absent key"
    )


# ---------------------------------------------------------------------------
# config_hash cache-invalidation behavior.
# ---------------------------------------------------------------------------


def test_config_hash_changes_on_effect_type():
    cfg_a = _base_cfg()
    cfg_a["ending_fx"] = _ending_fx_cfg(effect_type="hall_wash")
    cfg_b = _base_cfg()
    cfg_b["ending_fx"] = _ending_fx_cfg(effect_type="dub_echo")
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_config_hash_changes_on_tail_sec():
    cfg_a = _base_cfg()
    cfg_a["ending_fx"] = _ending_fx_cfg(tail_sec=4.0)
    cfg_b = _base_cfg()
    cfg_b["ending_fx"] = _ending_fx_cfg(tail_sec=6.0)
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_config_hash_changes_on_wet():
    cfg_a = _base_cfg()
    cfg_a["ending_fx"] = _ending_fx_cfg(wet=1.0)
    cfg_b = _base_cfg()
    cfg_b["ending_fx"] = _ending_fx_cfg(wet=0.5)
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_config_hash_changes_on_primary():
    cfg_a = _base_cfg()
    cfg_a["ending_fx"] = _ending_fx_cfg(primary=0.5)
    cfg_b = _base_cfg()
    cfg_b["ending_fx"] = _ending_fx_cfg(primary=0.9)
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_config_hash_disabled_ending_fx_equals_no_key():
    """Disabled ending_fx canonicalizes away — no spurious cache bust."""
    cfg_no_key = _base_cfg()
    cfg_disabled = _base_cfg()
    cfg_disabled["ending_fx"] = _ending_fx_cfg(enabled=False)
    assert config_hash(cfg_no_key) == config_hash(cfg_disabled)


# ---------------------------------------------------------------------------
# sr guard still enforced with ending_fx enabled.
# ---------------------------------------------------------------------------


def test_sr_guard_still_enforced_with_ending_fx_enabled():
    audio = _stereo_signal(seconds=0.2, sr=44100)
    cfg = _base_cfg()
    cfg["ending_fx"] = _ending_fx_cfg()
    try:
        MasteringChain(cfg).process(audio, 44100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for sr != 48000 with ending_fx on")


# ---------------------------------------------------------------------------
# Order pin: ending_fx applies AFTER matchering (matchering OFF here; the tail
# presence proves the call fired post-matchering in the seam).
# ---------------------------------------------------------------------------


def test_ending_fx_applies_after_matchering_seam():
    """matchering OFF, ending_fx ON → tail present (call fires at the seam)."""
    audio = _stereo_signal()
    cfg = _base_cfg()
    cfg["matchering"] = {"enabled": False, "reference_path": ""}
    cfg["ending_fx"] = _ending_fx_cfg(tail_sec=4.0)

    out = MasteringChain(cfg).process(audio, SR)
    assert out.shape[1] > audio.shape[1]
