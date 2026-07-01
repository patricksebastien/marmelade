"""Wave 0 RED stub — :func:`config_hash` golden test vectors A..E.

Vectors per RESEARCH §Pattern 2 lines 447-472:

* A — toggling a disabled stage's param does NOT change hash.
* B — enabling a stage DOES change hash.
* C — float precision normalization (6-decimal round).
* D — Matchering reference path matters.
* E — disabled-vs-disabled equality regardless of stale fields.

Phase 7 — Plan 01 Wave 0 (07-01-PLAN.md Task 1).
"""

from __future__ import annotations

import re

import pytest


def _baseline_cfg() -> dict:
    """Return the default disabled-everywhere shape (matchers _SESSION_DEFAULTS).

    All stages disabled except limiter (which is the canonical default).
    """
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
    }


def test_vector_a_disabled_param_toggle_invariant():
    """Vector A — toggling a disabled stage's param does NOT change hash."""
    from marmelade.audio.mastering.chain import config_hash

    cfg1 = _baseline_cfg()
    cfg2 = _baseline_cfg()
    cfg2["highpass"]["cutoff_hz"] = 120.0  # change a DISABLED stage's param
    assert config_hash(cfg1) == config_hash(cfg2)


def test_vector_b_enable_changes_hash():
    """Vector B — enabling a stage DOES change hash."""
    from marmelade.audio.mastering.chain import config_hash

    cfg1 = _baseline_cfg()
    cfg3 = _baseline_cfg()
    cfg3["highpass"]["enabled"] = True
    assert config_hash(cfg1) != config_hash(cfg3)


def test_vector_c_float_precision_normalization():
    """Vector C — 6-decimal round collapses near-equal floats."""
    from marmelade.audio.mastering.chain import config_hash

    cfg4 = _baseline_cfg()
    cfg4["highpass"]["enabled"] = True
    cfg4["highpass"]["cutoff_hz"] = 30.0000001
    cfg5 = _baseline_cfg()
    cfg5["highpass"]["enabled"] = True
    cfg5["highpass"]["cutoff_hz"] = 30.0
    assert config_hash(cfg4) == config_hash(cfg5)


def test_vector_d_matchering_reference_path_matters():
    """Vector D — different Matchering reference paths produce different hashes."""
    from marmelade.audio.mastering.chain import config_hash

    base = _baseline_cfg()
    base["highpass"]["enabled"] = True
    cfg6 = {**base, "matchering": {"enabled": True, "reference_path": "/a/ref1.wav"}}
    cfg7 = {**base, "matchering": {"enabled": True, "reference_path": "/a/ref2.wav"}}
    assert config_hash(cfg6) != config_hash(cfg7)


def test_vector_e_disabled_stale_fields_equality():
    """Vector E — two disabled-only forms with different stale fields hash equal."""
    from marmelade.audio.mastering.chain import config_hash

    cfg8 = {"compressor": {"enabled": False}}
    cfg9 = {"compressor": {"enabled": False, "ratio": 8.0, "attack_ms": 0.1}}
    assert config_hash(cfg8) == config_hash(cfg9)


def test_vector_f_fade_dropped_unconditionally():
    """Vector F (quick-260626-o9y) — fade contributes NOTHING to the hash.

    A cfg with fade.enabled=True, fade.enabled=False, AND no "fade" key at all
    must ALL hash EQUAL — fade is an output-time stage dropped unconditionally
    so a fade toggle / duration edit never busts the mastered cache.
    """
    from marmelade.audio.mastering.chain import config_hash

    base = _baseline_cfg()  # no "fade" key

    cfg_on = _baseline_cfg()
    cfg_on["fade"] = {"enabled": True, "duration_sec": 2.0}

    cfg_off = _baseline_cfg()
    cfg_off["fade"] = {"enabled": False, "duration_sec": 5.0}

    h_absent = config_hash(base)
    h_on = config_hash(cfg_on)
    h_off = config_hash(cfg_off)

    assert h_absent == h_on == h_off


def test_hash_is_12_hex_chars():
    """Sanity — hash is a 12-character lowercase hex string."""
    from marmelade.audio.mastering.chain import config_hash

    h = config_hash(_baseline_cfg())
    assert re.fullmatch(r"^[0-9a-f]{12}$", h), f"unexpected hash shape: {h!r}"
