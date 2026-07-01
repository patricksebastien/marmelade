"""Reverb stage — whole-clip ambience via ``pedalboard.Reverb``.

A standard mastering/color stage (distinct from the per-keeper ``ending_fx``
tail ring-out, which only reverberates the LAST few seconds). This applies
``pedalboard.Reverb`` across the whole clip, exposing the five room knobs
(room size / damping / wet / dry / width). It is a normal pedalboard plugin
in the pre-limiter chain, so the true-peak limiter still guards the output.

The five Param defaults mirror ``pedalboard.Reverb``'s own constructor
defaults (verified: room_size=0.5, damping=0.5, wet_level=0.33,
dry_level=0.4, width=1.0) so an enabled-at-defaults reverb matches the
plugin's native voicing. ``freeze_mode`` is intentionally NOT exposed — it
holds the reverb tail infinitely and is unmusical for mastering.

N-3 invariant: this module imports no GUI toolkit (audio tier stays
toolkit-free per CLAUDE.md "Extensibility").
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class ReverbStage(MasteringStage):
    """Whole-clip reverb — room size / damping / wet / dry / width."""

    name: ClassVar[str] = "reverb"
    display_name: ClassVar[str] = "Reverb"

    # Mirror pedalboard.Reverb constructor defaults (single source of truth;
    # chain.py's factory reads these ClassVars instead of re-hardcoding them).
    ROOM_SIZE_DEFAULT: ClassVar[float] = 0.5
    DAMPING_DEFAULT: ClassVar[float] = 0.5
    WET_LEVEL_DEFAULT: ClassVar[float] = 0.33
    DRY_LEVEL_DEFAULT: ClassVar[float] = 0.4
    WIDTH_DEFAULT: ClassVar[float] = 1.0

    def parameters(self) -> dict[str, Param]:
        return {
            "room_size": Param(
                name="room_size",
                label="Room size",
                kind="float",
                default=self.ROOM_SIZE_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Reverb decay length (0 = tiny room, 1 = huge hall).",
            ),
            "damping": Param(
                name="damping",
                label="Damping",
                kind="float",
                default=self.DAMPING_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=1.0,
                step=0.05,
                description="High-frequency absorption (0 = bright, 1 = dark tail).",
            ),
            "wet_level": Param(
                name="wet_level",
                label="Wet",
                kind="float",
                default=self.WET_LEVEL_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Level of the reverberated (wet) signal.",
            ),
            "dry_level": Param(
                name="dry_level",
                label="Dry",
                kind="float",
                default=self.DRY_LEVEL_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Level of the unprocessed (dry) signal.",
            ),
            "width": Param(
                name="width",
                label="Width",
                kind="float",
                default=self.WIDTH_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Stereo width of the reverb (0 = mono, 1 = wide).",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        return pedalboard.Reverb(
            room_size=float(self._get("room_size", self.ROOM_SIZE_DEFAULT)),
            damping=float(self._get("damping", self.DAMPING_DEFAULT)),
            wet_level=float(self._get("wet_level", self.WET_LEVEL_DEFAULT)),
            dry_level=float(self._get("dry_level", self.DRY_LEVEL_DEFAULT)),
            width=float(self._get("width", self.WIDTH_DEFAULT)),
        )
