"""Ending-FX presets — pure-data tail configs + two-way matcher (Phase 07.1).

10 curated per-keeper ending-FX presets for the Mastering dialog, mirroring
the genre-preset module :mod:`marmelade.audio.mastering.presets`. Each preset
is a single ending_fx STAGE dict (NOT a full 8-stage chain) of the exact
shape stored under ``Region.mastering["ending_fx"]``::

    {"enabled": True, "effect_type": <id>, "tail_sec": <float>,
     "onset_sec": <float>, "wet": <float>, "primary": <float>}

so it round-trips cleanly through :func:`config_hash` and passes the Plan-03
sidecar param-range validation. ``effect_type`` maps 1:1 to the
:data:`marmelade.audio.mastering.stages.ending_fx.EFFECT_BUILDERS` ids; the
per-effect ``tail_sec`` defaults are sourced from
:data:`...ending_fx.EFFECT_TAIL_DEFAULTS` (single source of truth — Task 1
locked the numbers).

N-3 invariant: this module lives under ``audio/mastering/`` and imports NO
Qt — only ``typing`` plus ``config_hash`` from the sibling ``chain`` module
and the effect ids/tail defaults from ``stages.ending_fx``. Pure data + a
hashing helper.

The dict insertion order IS the locked dock/dialog lineup order:
    Hall Wash, Dub Echo, Tape Stop, Filter Close, Shimmer Freeze,
    Bitcrush Collapse, Codec Rot, Glitch Stutter, Overdrive Bloom, Smear.

LOCKED INPUT-SHAPE CONTRACT: :func:`match_ending_fx` takes the BARE
ending_fx stage dict — NOT a ``{"ending_fx": ...}`` wrapper — exactly what
Plan 04 passes via ``match_ending_fx(self._cfg.get("ending_fx", {}))``. The
wrapping needed to reuse ``config_hash``'s disabled-stage / float-rounding
canonicalization is an implementation detail hidden INSIDE the function.
"""

from __future__ import annotations

from typing import Any

from marmelade.audio.mastering.chain import config_hash
from marmelade.audio.mastering.stages.ending_fx import EFFECT_TAIL_DEFAULTS


def _preset(
    effect_type: str,
    *,
    wet: float = 1.0,
    primary: float = 0.5,
    onset_sec: float = 2.0,
) -> dict[str, Any]:
    """Build one ending_fx STAGE dict for an effect_type.

    ``tail_sec`` is taken from the Task-1 per-effect defaults so the presets
    never invent numbers. ``onset_sec`` is how many seconds before the region
    end the effect blends in from dry (quick-260624-ph1). All numeric literals
    are floats so config_hash's 6dp float rounding round-trips cleanly.
    """
    return {
        "enabled": True,
        "effect_type": effect_type,
        "tail_sec": float(EFFECT_TAIL_DEFAULTS[effect_type]),
        "onset_sec": float(onset_sec),
        "wet": float(wet),
        "primary": float(primary),
    }


# Insertion order = locked lineup order. "Custom" is NOT a member here; it is
# the dialog combobox's index-0 sentinel handled by match_ending_fx == None.
ENDING_FX_PRESETS: dict[str, dict[str, Any]] = {
    # Slow washes get a long blend-in; rhythmic/aggressive get a short snap-in
    # (quick-260624-ph1 onset_sec — seconds before the region end the FX enters).
    "Hall Wash": _preset("hall_wash", onset_sec=3.0),
    # More feedback reads better for a dub echo.
    "Dub Echo": _preset("dub_echo", primary=0.6, onset_sec=2.0),
    "Tape Stop": _preset("tape_stop", onset_sec=1.0),
    "Filter Close": _preset("filter_close", onset_sec=2.0),
    "Shimmer Freeze": _preset("shimmer_freeze", onset_sec=3.0),
    "Bitcrush Collapse": _preset("bitcrush_collapse", onset_sec=1.5),
    "Codec Rot": _preset("codec_rot", onset_sec=2.0),
    "Glitch Stutter": _preset("glitch_stutter", onset_sec=1.0),
    "Overdrive Bloom": _preset("overdrive_bloom", onset_sec=2.0),
    "Smear": _preset("smear", onset_sec=3.0),
}


def ending_fx_preset_names() -> list[str]:
    """Return the 10 ending-FX preset names in the locked lineup order.

    "Custom" is NOT included — it is the dialog combobox's index-0 sentinel.
    """
    return list(ENDING_FX_PRESETS.keys())


def match_ending_fx(stage_cfg: dict) -> str | None:  # stage_cfg is the BARE ending_fx stage dict
    """Return the preset name whose config_hash equals ``stage_cfg``'s, else None.

    ``stage_cfg`` is the BARE ending_fx stage dict (the value stored under
    ``self._cfg["ending_fx"]``), NOT a ``{"ending_fx": ...}`` wrapper — Plan
    04 calls ``match_ending_fx(self._cfg.get("ending_fx", {}))``.

    Internally the bare dict is wrapped as ``{"ending_fx": stage_cfg}`` so it
    reuses the EXACT disabled-stage / float-rounding canonicalization that the
    genre matcher uses (:func:`config_hash` collapses a disabled stage to
    ``{"enabled": False}``, so a disabled/empty dict matches no enabled
    preset and returns ``None`` — the "Custom" sentinel).
    """
    target = config_hash({"ending_fx": stage_cfg})
    for name, preset_cfg in ENDING_FX_PRESETS.items():
        if config_hash({"ending_fx": preset_cfg}) == target:
            return name
    return None


__all__ = ["ENDING_FX_PRESETS", "ending_fx_preset_names", "match_ending_fx"]
