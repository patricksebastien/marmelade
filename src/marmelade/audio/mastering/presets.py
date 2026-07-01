"""Genre mastering presets — pure-data session-chain configs + matcher.

quick-260623-p5b — 10 one-click, export-ready genre mastering presets for
the Mastering dock. Each preset is a COMPLETE 8-stage session-chain config
mirroring :data:`marmelade.audio.mastering.chain._SESSION_DEFAULTS`'s shape,
so it round-trips cleanly through :func:`config_hash` and passes
:func:`marmelade.audio.sidecar_cache._validate_stage_param_ranges`.

N-3 invariant: this module lives under ``audio/mastering/`` and imports NO
Qt — only ``typing`` plus ``config_hash`` from the sibling ``chain`` module
(itself audio-tier; its lone ``QtCore.QSettings`` import is the documented
boundary crossing). It is therefore pure data + a hashing helper.

Common to ALL 10 presets (locked_preset_table):
    * highpass ON (per-genre cutoff_hz)
    * eq ON (per-genre low/mid/high gains)
    * compressor ON (per-genre threshold/ratio/attack/release)
    * limiter ON @ ceiling_dbtp -1.0 dBTP, release_ms 100.0
    * matchering OFF (reference_path "")
    * normalize OFF (target_db 0.0)
    * lowpass OFF (cutoff_hz 18000.0 placeholder) EXCEPT Lo-fi (ON @ 16000.0)
    * loudness ON at a per-genre target_lufs EXCEPT Ambient (OFF — preserve
      dynamics; placeholder target_lufs in range, canonicalized away by
      config_hash since the stage is disabled)

The dict insertion order IS the locked dock lineup order:
    House, Techno, Dubstep, Drum & Bass, Trance, EDM / Festival,
    Hip-Hop / Trap, Lo-fi, Pop, Ambient.
"""

from __future__ import annotations

from typing import Any

from marmelade.audio.mastering.chain import config_hash


def _preset(
    *,
    highpass_hz: float,
    eq_low: float,
    eq_mid: float,
    eq_high: float,
    comp_threshold: float,
    comp_ratio: float,
    comp_attack: float,
    comp_release: float,
    loudness_on: bool,
    target_lufs: float,
    lowpass_on: bool = False,
    lowpass_hz: float = 18000.0,
) -> dict[str, dict[str, Any]]:
    """Build a COMPLETE 8-stage session-chain config for one genre preset.

    Every stage key + every param mirrors ``_SESSION_DEFAULTS`` so the
    result hashes via :func:`config_hash` and validates against
    ``_validate_stage_param_ranges``. All numeric literals are floats so
    config_hash's 6dp float rounding round-trips cleanly.
    """
    return {
        "normalize": {"enabled": False, "target_db": 0.0},
        "loudness": {"enabled": bool(loudness_on), "target_lufs": float(target_lufs)},
        "highpass": {"enabled": True, "cutoff_hz": float(highpass_hz)},
        "lowpass": {"enabled": bool(lowpass_on), "cutoff_hz": float(lowpass_hz)},
        "eq": {
            "enabled": True,
            "low_db": float(eq_low),
            "mid_db": float(eq_mid),
            "high_db": float(eq_high),
        },
        "compressor": {
            "enabled": True,
            "threshold_db": float(comp_threshold),
            "ratio": float(comp_ratio),
            "attack_ms": float(comp_attack),
            "release_ms": float(comp_release),
        },
        "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "matchering": {"enabled": False, "reference_path": ""},
    }


# Insertion order = locked dock lineup order. "Custom" is NOT a member here;
# it is the dock combobox's index-0 sentinel handled by match_preset == None.
MASTERING_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "House": _preset(
        highpass_hz=30.0,
        eq_low=1.5, eq_mid=0.0, eq_high=2.0,
        comp_threshold=-18.0, comp_ratio=4.0, comp_attack=10.0, comp_release=150.0,
        loudness_on=True, target_lufs=-14.0,
    ),
    "Techno": _preset(
        highpass_hz=35.0,
        eq_low=2.0, eq_mid=-1.0, eq_high=1.0,
        comp_threshold=-20.0, comp_ratio=4.0, comp_attack=5.0, comp_release=120.0,
        loudness_on=True, target_lufs=-13.0,
    ),
    "Dubstep": _preset(
        highpass_hz=25.0,
        eq_low=3.0, eq_mid=-1.0, eq_high=2.5,
        comp_threshold=-20.0, comp_ratio=6.0, comp_attack=5.0, comp_release=100.0,
        loudness_on=True, target_lufs=-12.0,
    ),
    "Drum & Bass": _preset(
        highpass_hz=28.0,
        eq_low=2.5, eq_mid=0.0, eq_high=2.0,
        comp_threshold=-18.0, comp_ratio=5.0, comp_attack=3.0, comp_release=80.0,
        loudness_on=True, target_lufs=-12.0,
    ),
    "Trance": _preset(
        highpass_hz=35.0,
        eq_low=1.0, eq_mid=0.0, eq_high=2.5,
        comp_threshold=-18.0, comp_ratio=3.0, comp_attack=15.0, comp_release=200.0,
        loudness_on=True, target_lufs=-13.0,
    ),
    "EDM / Festival": _preset(
        highpass_hz=30.0,
        eq_low=2.0, eq_mid=-1.0, eq_high=2.5,
        comp_threshold=-20.0, comp_ratio=4.0, comp_attack=8.0, comp_release=120.0,
        loudness_on=True, target_lufs=-12.0,
    ),
    "Hip-Hop / Trap": _preset(
        highpass_hz=25.0,
        eq_low=3.0, eq_mid=-0.5, eq_high=1.5,
        comp_threshold=-18.0, comp_ratio=3.0, comp_attack=20.0, comp_release=180.0,
        loudness_on=True, target_lufs=-14.0,
    ),
    "Lo-fi": _preset(
        highpass_hz=40.0,
        eq_low=1.0, eq_mid=0.0, eq_high=-2.0,
        comp_threshold=-22.0, comp_ratio=2.0, comp_attack=30.0, comp_release=250.0,
        loudness_on=True, target_lufs=-16.0,
        lowpass_on=True, lowpass_hz=16000.0,
    ),
    "Pop": _preset(
        highpass_hz=40.0,
        eq_low=1.0, eq_mid=1.0, eq_high=2.0,
        comp_threshold=-18.0, comp_ratio=3.0, comp_attack=15.0, comp_release=150.0,
        loudness_on=True, target_lufs=-14.0,
    ),
    "Ambient": _preset(
        highpass_hz=30.0,
        eq_low=0.0, eq_mid=0.0, eq_high=1.0,
        comp_threshold=-24.0, comp_ratio=1.5, comp_attack=50.0, comp_release=400.0,
        # Ambient preserves dynamics — loudness OFF. The target_lufs is an
        # in-range placeholder (-14.0); config_hash canonicalizes a disabled
        # stage to {"enabled": False} so it never affects matching.
        loudness_on=False, target_lufs=-14.0,
    ),
}


def preset_names() -> list[str]:
    """Return the 10 genre preset names in the locked lineup order.

    "Custom" is NOT included — it is the dock combobox's index-0 sentinel.
    """
    return list(MASTERING_PRESETS.keys())


def match_preset(snapshot: dict[str, dict[str, Any]]) -> str | None:
    """Return the preset name whose config_hash equals ``snapshot``'s, else None.

    Reuses :func:`config_hash` so float rounding + disabled-stage param
    dropping are handled identically to keeper-divergence matching. A
    ``None`` return is the "Custom" sentinel — the snapshot matches no
    authored preset (e.g. the empty-QSettings limiter-only default, or a
    user-edited chain).
    """
    target = config_hash(snapshot)
    for name, cfg in MASTERING_PRESETS.items():
        if config_hash(cfg) == target:
            return name
    return None


__all__ = ["MASTERING_PRESETS", "preset_names", "match_preset"]
