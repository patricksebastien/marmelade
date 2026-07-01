"""Unified 3-band EQ stage (quick-260623-k7t — replaces the two shelf stages).

Gain-only, fixed-frequency 3-band EQ:

* Low  — low-shelf  primitive @ 100 Hz
* Mid  — peak       primitive @ 1000 Hz
* High — high-shelf primitive @ 10000 Hz

All three bands share a fixed ``Q = 0.7071`` and expose a single dB gain
each (range -12..+12, default 0.0, step 0.5). The center frequencies + Q
are the SINGLE SOURCE OF TRUTH — declared as ClassVars here and read by
``chain.py``'s ``_STAGE_FACTORY["eq"]`` (no duplicated literals).

N-3 invariant: this module imports no GUI toolkit (audio tier stays
toolkit-free per CLAUDE.md).
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class EqStage(MasteringStage):
    """3-band gain-only EQ — Low shelf / Mid peak / High shelf, fixed freqs."""

    name: ClassVar[str] = "eq"
    display_name: ClassVar[str] = "EQ"

    # Single source of truth for the band center frequencies + Q. The
    # chain.py factory reads these ClassVars instead of hardcoding
    # literals (locked decision — no duplicated 100/1000/10000/Q values).
    LOW_HZ: ClassVar[float] = 100.0
    MID_HZ: ClassVar[float] = 1000.0
    HIGH_HZ: ClassVar[float] = 10000.0
    Q_DEFAULT: ClassVar[float] = 0.7071

    LOW_DB_DEFAULT: ClassVar[float] = 0.0
    MID_DB_DEFAULT: ClassVar[float] = 0.0
    HIGH_DB_DEFAULT: ClassVar[float] = 0.0

    def parameters(self) -> dict[str, Param]:
        return {
            "low_db": Param(
                name="low_db",
                label="Low",
                kind="float",
                default=self.LOW_DB_DEFAULT,
                requires_recompute=True,
                min=-12.0,
                max=12.0,
                step=0.5,
                unit="dB",
                description="Low-band gain (low shelf, fixed 100 Hz).",
            ),
            "mid_db": Param(
                name="mid_db",
                label="Mid",
                kind="float",
                default=self.MID_DB_DEFAULT,
                requires_recompute=True,
                min=-12.0,
                max=12.0,
                step=0.5,
                unit="dB",
                description="Mid-band gain (peak, fixed 1000 Hz).",
            ),
            "high_db": Param(
                name="high_db",
                label="High",
                kind="float",
                default=self.HIGH_DB_DEFAULT,
                requires_recompute=True,
                min=-12.0,
                max=12.0,
                step=0.5,
                unit="dB",
                description="High-band gain (high shelf, fixed 10000 Hz).",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        # A nested Pedalboard IS a pedalboard.Plugin (verified 0.9.22) — it
        # appends cleanly to the outer chain in the apply loop.
        return pedalboard.Pedalboard(
            [
                pedalboard.LowShelfFilter(
                    cutoff_frequency_hz=self.LOW_HZ,
                    gain_db=float(self._get("low_db", self.LOW_DB_DEFAULT)),
                    q=self.Q_DEFAULT,
                ),
                pedalboard.PeakFilter(
                    cutoff_frequency_hz=self.MID_HZ,
                    gain_db=float(self._get("mid_db", self.MID_DB_DEFAULT)),
                    q=self.Q_DEFAULT,
                ),
                pedalboard.HighShelfFilter(
                    cutoff_frequency_hz=self.HIGH_HZ,
                    gain_db=float(self._get("high_db", self.HIGH_DB_DEFAULT)),
                    q=self.Q_DEFAULT,
                ),
            ]
        )
