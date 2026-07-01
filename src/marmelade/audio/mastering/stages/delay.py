"""Delay stage — echo via ``pedalboard.Delay``.

A whole-clip echo with time / feedback / mix knobs. Placed in the
pre-limiter chain so the true-peak limiter guards the output.

``feedback`` is capped at 0.95 (not 1.0): a feedback of 1.0 is a
self-sustaining echo that never decays, which would ring forever and
keep the limiter working indefinitely. ``pedalboard.Delay``'s own
defaults are delay_seconds=0.5, feedback=0.0, mix=0.5; we pick a musical
default (0.375 s / 0.3 / 0.3) so an enabled delay is audible but tasteful.

N-3 invariant: no GUI toolkit imports (audio tier stays toolkit-free).
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class DelayStage(MasteringStage):
    """Echo — delay time / feedback / wet-dry mix."""

    name: ClassVar[str] = "delay"
    display_name: ClassVar[str] = "Delay"

    DELAY_SECONDS_DEFAULT: ClassVar[float] = 0.375
    FEEDBACK_DEFAULT: ClassVar[float] = 0.3
    MIX_DEFAULT: ClassVar[float] = 0.3

    def parameters(self) -> dict[str, Param]:
        return {
            "delay_seconds": Param(
                name="delay_seconds",
                label="Time",
                kind="float",
                default=self.DELAY_SECONDS_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=2.0,
                step=0.01,
                unit="s",
                description="Echo time between repeats.",
            ),
            "feedback": Param(
                name="feedback",
                label="Feedback",
                kind="float",
                default=self.FEEDBACK_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=0.95,
                step=0.05,
                description="How much of each echo feeds back (0.95 cap = no runaway).",
            ),
            "mix": Param(
                name="mix",
                label="Wet/Dry",
                kind="float",
                default=self.MIX_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Echo mix: 0.0 = dry, 1.0 = fully wet.",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        return pedalboard.Delay(
            delay_seconds=float(self._get("delay_seconds", self.DELAY_SECONDS_DEFAULT)),
            feedback=float(self._get("feedback", self.FEEDBACK_DEFAULT)),
            mix=float(self._get("mix", self.MIX_DEFAULT)),
        )
