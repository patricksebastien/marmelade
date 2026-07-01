"""quick-260621-gfq Task 1 — NormalizeStage + final-step chain integration.

Normalize is redesigned (quick-260621-gfq) as the FINAL stage of the
mastering chain — applied per keeper after limiter + LUFS makeup + ISP
verification + matchering. It is a real :class:`MasteringStage` subclass
exposing a single ``target_db`` Param (default 0.0 dBFS). The chain applies
it directly via :func:`marmelade.audio.normalize.normalize_array` (it is
NOT a pedalboard plugin — ``build_plugin`` raises ``NotImplementedError``).

Regression invariant: with normalize disabled (or the key absent) the
mastered output is array-identical to the pre-change pipeline. Enabled at
0 dB on a DC-offset signal: per-channel mean ≈ 0 and global peak ≈ 1.0.
"""

from __future__ import annotations

import numpy as np

from marmelade.audio.mastering import NormalizeStage
from marmelade.audio.mastering.chain import (
    _SESSION_DEFAULTS,
    _STAGE_ORDER,
    MasteringChain,
    config_hash,
)
from marmelade.audio.mastering.params import Param


SR = 48000


def test_normalize_is_first_in_stage_order() -> None:
    """``normalize`` leads the processing rows; ``matchering`` is the tail.

    NOTE: _STAGE_ORDER is the panel/display order, NOT the DSP processing
    order. quick-260626-o9y inserts the output-time ``fade`` row at index 0
    (top of the panel), so ``normalize`` is now the SECOND row — still shown
    near the top but APPLIED LAST in ``MasteringChain.process`` (after
    limiter/LUFS/ISP/matchering). ``matchering`` remains the tail of the
    display list. ``fade`` is never applied in process() at all.
    """
    assert _STAGE_ORDER[0] == "fade"
    assert _STAGE_ORDER[1] == "normalize"
    assert _STAGE_ORDER[-1] == "matchering"


def test_session_defaults_has_normalize() -> None:
    """``_SESSION_DEFAULTS['normalize']`` defaults to disabled @ 0.0 dB."""
    assert _SESSION_DEFAULTS["normalize"] == {"enabled": False, "target_db": 0.0}


def test_stage_single_target_db_param() -> None:
    """NormalizeStage exposes exactly one ``target_db`` float Param @ 0.0."""
    stage = NormalizeStage()
    params = stage.parameters()
    assert list(params.keys()) == ["target_db"]
    p = params["target_db"]
    assert isinstance(p, Param)
    assert p.kind == "float"
    assert p.default == 0.0
    assert p.min == -60.0
    assert p.max == 0.0
    assert p.unit == "dBFS"


def test_build_plugin_raises() -> None:
    """build_plugin is required by the ABC but normalize is not a plugin."""
    import pytest

    with pytest.raises(NotImplementedError):
        NormalizeStage().build_plugin()


def _dc_offset_stereo(sr: int = SR, dur_s: float = 0.5) -> np.ndarray:
    """Stereo sine + 0.2 DC offset, shape (2, n) float32."""
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float64) / sr
    mono = (0.2 * np.sin(2.0 * np.pi * 440.0 * t) + 0.2).astype(np.float32)
    return np.stack([mono, mono], axis=0)


def test_disabled_chain_is_array_identical() -> None:
    """normalize disabled (and key absent) → output identical to pre-change."""
    audio = _dc_offset_stereo()
    # Limiter-only chain (the pre-change default).
    base_cfg = {"limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0}}
    out_without = MasteringChain(dict(base_cfg)).process(audio.copy(), SR)

    # Same chain with normalize present but disabled.
    cfg_disabled = dict(base_cfg)
    cfg_disabled["normalize"] = {"enabled": False, "target_db": 0.0}
    out_disabled = MasteringChain(cfg_disabled).process(audio.copy(), SR)

    np.testing.assert_array_equal(out_without, out_disabled)


def test_enabled_at_0db_mean_zero_peak_one() -> None:
    """normalize enabled @ 0 dB on a DC signal → mean≈0, peak≈1.0.

    Limiter DISABLED so the normalize math is the only level operation and
    the 0 dBFS target is provable.
    """
    audio = _dc_offset_stereo()
    cfg = {
        "limiter": {"enabled": False, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "normalize": {"enabled": True, "target_db": 0.0},
    }
    out = MasteringChain(cfg).process(audio.copy(), SR)
    assert out.dtype == np.float32
    assert abs(float(out[0].mean())) < 1e-4
    assert abs(float(out[1].mean())) < 1e-4
    assert abs(float(np.abs(out).max()) - 1.0) < 1e-3


def test_config_hash_normalize_enable_changes_hash() -> None:
    """Enabling normalize changes the hash; target_db while disabled does not."""
    disabled_a = {"normalize": {"enabled": False, "target_db": 0.0}}
    disabled_b = {"normalize": {"enabled": False, "target_db": -12.0}}
    # Disabled-stage canonicalization: target_db change is invisible.
    assert config_hash(disabled_a) == config_hash(disabled_b)

    enabled = {"normalize": {"enabled": True, "target_db": 0.0}}
    assert config_hash(enabled) != config_hash(disabled_a)
