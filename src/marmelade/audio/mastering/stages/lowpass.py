"""Low-pass filter stage (pedalboard 0.9.22 — :class:`LowpassFilter`).

UI-SPEC default: 18 kHz cutoff (Phase 7 — per-stage rows).
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class LowPassStage(MasteringStage):
    """Wrap :class:`pedalboard.LowpassFilter`."""

    name: ClassVar[str] = "lowpass"
    display_name: ClassVar[str] = "Low-pass filter"

    CUTOFF_HZ_DEFAULT: ClassVar[float] = 18000.0

    def parameters(self) -> dict[str, Param]:
        return {
            "cutoff_hz": Param(
                name="cutoff_hz",
                label="Cutoff",
                kind="float",
                default=self.CUTOFF_HZ_DEFAULT,
                requires_recompute=True,
                min=1000.0,
                max=22050.0,
                step=100.0,
                unit="Hz",
                description="6 dB/octave first-order low-pass.",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        return pedalboard.LowpassFilter(
            cutoff_frequency_hz=float(self._get("cutoff_hz", self.CUTOFF_HZ_DEFAULT))
        )
