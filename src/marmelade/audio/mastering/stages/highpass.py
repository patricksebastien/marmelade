"""High-pass filter stage (6 dB/octave first-order HP — pedalboard 0.9.22).

UI-SPEC default: 30 Hz cutoff (Phase 7 — per-stage rows).
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class HighPassStage(MasteringStage):
    """Wrap :class:`pedalboard.HighpassFilter`."""

    name: ClassVar[str] = "highpass"
    display_name: ClassVar[str] = "High-pass filter"

    CUTOFF_HZ_DEFAULT: ClassVar[float] = 30.0

    def parameters(self) -> dict[str, Param]:
        return {
            "cutoff_hz": Param(
                name="cutoff_hz",
                label="Cutoff",
                kind="float",
                default=self.CUTOFF_HZ_DEFAULT,
                requires_recompute=True,
                min=20.0,
                max=500.0,
                step=1.0,
                unit="Hz",
                description="6 dB/octave first-order high-pass.",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        return pedalboard.HighpassFilter(
            cutoff_frequency_hz=float(self._get("cutoff_hz", self.CUTOFF_HZ_DEFAULT))
        )
