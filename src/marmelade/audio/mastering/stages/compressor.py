"""Compressor stage (pedalboard 0.9.22 — :class:`Compressor`).

UI-SPEC defaults: -18 dB threshold, 2:1 ratio, 30 ms attack, 200 ms release.
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class CompressorStage(MasteringStage):
    """Wrap :class:`pedalboard.Compressor`."""

    name: ClassVar[str] = "compressor"
    display_name: ClassVar[str] = "Compressor"

    THRESHOLD_DB_DEFAULT: ClassVar[float] = -18.0
    RATIO_DEFAULT: ClassVar[float] = 2.0
    ATTACK_MS_DEFAULT: ClassVar[float] = 30.0
    RELEASE_MS_DEFAULT: ClassVar[float] = 200.0

    def parameters(self) -> dict[str, Param]:
        return {
            "threshold_db": Param(
                name="threshold_db",
                label="Threshold",
                kind="float",
                default=self.THRESHOLD_DB_DEFAULT,
                requires_recompute=True,
                min=-60.0,
                max=0.0,
                step=1.0,
                unit="dB",
                description="Compression threshold (input level above which gain reduction occurs).",
            ),
            "ratio": Param(
                name="ratio",
                label="Ratio",
                kind="float",
                default=self.RATIO_DEFAULT,
                requires_recompute=True,
                min=1.0,
                max=20.0,
                step=0.1,
                unit=":1",
                description="Compression ratio (input dB above threshold : output dB).",
            ),
            "attack_ms": Param(
                name="attack_ms",
                label="Attack",
                kind="float",
                default=self.ATTACK_MS_DEFAULT,
                requires_recompute=True,
                min=0.1,
                max=500.0,
                step=1.0,
                unit="ms",
                description="Attack time (how fast gain reduction engages).",
            ),
            "release_ms": Param(
                name="release_ms",
                label="Release",
                kind="float",
                default=self.RELEASE_MS_DEFAULT,
                requires_recompute=True,
                min=10.0,
                max=2000.0,
                step=10.0,
                unit="ms",
                description="Release time (how fast gain reduction relaxes).",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        return pedalboard.Compressor(
            threshold_db=float(self._get("threshold_db", self.THRESHOLD_DB_DEFAULT)),
            ratio=float(self._get("ratio", self.RATIO_DEFAULT)),
            attack_ms=float(self._get("attack_ms", self.ATTACK_MS_DEFAULT)),
            release_ms=float(self._get("release_ms", self.RELEASE_MS_DEFAULT)),
        )
