"""Distortion stage — drive/saturation via ``pedalboard.Distortion``.

A single-knob waveshaping drive (``drive_db``) applied across the whole
clip. Adds warmth/grit; placed in the pre-limiter chain so the true-peak
limiter still guards the output even when driven hot.

``pedalboard.Distortion``'s native default is ``drive_db=25.0``; we mirror
it so an enabled-at-default distortion matches the plugin's own voicing.

N-3 invariant: no GUI toolkit imports (audio tier stays toolkit-free).
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class DistortionStage(MasteringStage):
    """Waveshaping drive — single ``drive_db`` knob."""

    name: ClassVar[str] = "distortion"
    display_name: ClassVar[str] = "Distortion"

    DRIVE_DB_DEFAULT: ClassVar[float] = 25.0

    def parameters(self) -> dict[str, Param]:
        return {
            "drive_db": Param(
                name="drive_db",
                label="Drive",
                kind="float",
                default=self.DRIVE_DB_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=60.0,
                step=1.0,
                unit="dB",
                description="Saturation drive — higher is grittier.",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        return pedalboard.Distortion(
            drive_db=float(self._get("drive_db", self.DRIVE_DB_DEFAULT)),
        )
