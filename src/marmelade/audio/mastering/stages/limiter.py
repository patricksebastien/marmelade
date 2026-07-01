"""Limiter stage — true-peak-targeting via :class:`Limiter` + post-:class:`Gain`.

Phase 7 RESEARCH §Pattern 5 + Pitfall 1: pedalboard's
:class:`pedalboard.Limiter` is NOT a true-peak (ISP) limiter, and its
``threshold_db`` is NOT the output ceiling. The 2-plugin sub-chain
``Limiter(threshold_db = ceiling - 1) + Gain(gain_db = ceiling - 1)``
yields a sample-peak ceiling of ``ceiling - 1`` dBFS — provably below
``ceiling`` dBTP after the chain plus the optional ISP-verification
pass.

The :class:`LimiterStage` subclass exposes the parameter surface for
the gear-button dialog. The chain orchestrator does NOT call
:meth:`LimiterStage.build_plugin` — it calls the module-level
:func:`build_limiter_subchain` instead. :meth:`build_plugin` is
provided for symmetry with the other stage classes (and used by the
Wave 0 stage-shape test).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class LimiterStage(MasteringStage):
    """Declare the limiter's tunable surface (ceiling + release).

    The chain orchestrator builds the actual limiter sub-chain via
    :func:`build_limiter_subchain` so the Limiter + Gain pair is kept
    contiguous (Pitfall 1).
    """

    name: ClassVar[str] = "limiter"
    display_name: ClassVar[str] = "Limiter"

    CEILING_DBTP_DEFAULT: ClassVar[float] = -1.0
    RELEASE_MS_DEFAULT: ClassVar[float] = 100.0

    def parameters(self) -> dict[str, Param]:
        return {
            "ceiling_dbtp": Param(
                name="ceiling_dbtp",
                label="Ceiling",
                kind="float",
                default=self.CEILING_DBTP_DEFAULT,
                requires_recompute=True,
                min=-6.0,
                max=0.0,
                step=0.1,
                unit="dBTP",
                description=(
                    "True-peak ceiling. Default -1 dBTP is industry standard "
                    "for lossy transcoding headroom."
                ),
            ),
            "release_ms": Param(
                name="release_ms",
                label="Release",
                kind="float",
                default=self.RELEASE_MS_DEFAULT,
                requires_recompute=True,
                min=10.0,
                max=1000.0,
                step=10.0,
                unit="ms",
                description="Release time (how fast limiting relaxes).",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        """Return a SINGLE :class:`pedalboard.Limiter` (unit-test convenience).

        The chain orchestrator does NOT use this — it calls
        :func:`build_limiter_subchain` instead so the post-Gain pass is
        appended atomically. This method is provided so the stage-shape
        Wave 0 test (which asserts every stage class has a
        ``build_plugin()``) works.
        """
        # threshold_db = -2 dBFS by default — see ``build_limiter_subchain``
        # for the ceiling derivation rationale.
        return pedalboard.Limiter(
            threshold_db=-2.0,
            release_ms=float(self._get("release_ms", self.RELEASE_MS_DEFAULT)),
        )


def build_limiter_subchain(cfg: dict[str, Any]) -> list[pedalboard.Plugin]:
    """Return the two-plugin sub-chain enforcing -X dBTP via Limiter+Gain.

    Strategy: limit to ``ceiling - 1`` dBFS sample-peak, then apply
    ``Gain(gain_db = ceiling - 1)`` so the output sample-peak ceiling is
    ``ceiling - 1`` dBFS (~ 1 dB ISP headroom). The optional post-chain
    :func:`run_isp_verification` pass catches the rare cases where 1 dB
    isn't enough.

    For a -1 dBTP target: ``Limiter(threshold_db=-2.0) + Gain(gain_db=-2.0)``.
    """
    ceiling_dbtp = float(cfg.get("ceiling_dbtp", -1.0))
    release_ms = float(cfg.get("release_ms", 100.0))
    # Reserve 1 dB of ISP headroom (empirically sufficient for typical
    # jams; pathological signals get caught by run_isp_verification).
    sample_peak_ceiling_dbfs = ceiling_dbtp - 1.0
    return [
        pedalboard.Limiter(
            threshold_db=sample_peak_ceiling_dbfs,
            release_ms=release_ms,
        ),
        pedalboard.Gain(gain_db=sample_peak_ceiling_dbfs),
    ]
