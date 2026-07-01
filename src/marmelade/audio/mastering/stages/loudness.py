"""Loudness stage — absolute LUFS target (quick-260623-l7l).

A NEW virtual tail stage that COEXISTS with the existing relative LUFS
makeup. Musicians delivering to streaming (Spotify/Apple normalize to
~-14 LUFS) need an absolute integrated-loudness target, not the relative
makeup. The stage is default-disabled, so today's behavior is unchanged
when off.

Like :class:`NormalizeStage` / matchering, this is NOT a pedalboard plugin:
:meth:`marmelade.audio.mastering.chain.MasteringChain.process` applies it
directly via :func:`marmelade.audio.mastering.lufs.normalize_to_lufs_target`
(absolute target gain) followed by
:func:`marmelade.audio.mastering.lufs.run_isp_verification` (true-peak
guarantee). ``build_plugin`` therefore raises :class:`NotImplementedError`
(required by the :class:`MasteringStage` ABC but never called).

N-3 invariant: no PySide6 / QtWidgets / QtGui imports.
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class LoudnessStage(MasteringStage):
    """Declare the loudness stage's tunable surface (target LUFS).

    The chain orchestrator applies loudness directly via
    :func:`marmelade.audio.mastering.lufs.normalize_to_lufs_target` — this
    class only declares the ``target_lufs`` Param so the Mastering dock can
    render the stage's gear dialog (same auto-render path as the pedalboard
    stages and NormalizeStage).
    """

    name: ClassVar[str] = "loudness"
    display_name: ClassVar[str] = "Loudness (LUFS)"

    TARGET_LUFS_DEFAULT: ClassVar[float] = -14.0

    def parameters(self) -> dict[str, Param]:
        return {
            "target_lufs": Param(
                name="target_lufs",
                label="Target",
                kind="float",
                default=self.TARGET_LUFS_DEFAULT,
                requires_recompute=True,
                min=-30.0,
                max=-6.0,
                step=0.5,
                unit="LUFS",
                description=(
                    "Integrated-loudness target (BS.1770). -14 LUFS matches "
                    "Spotify/Apple streaming normalization."
                ),
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        """Not a pedalboard plugin — the chain applies loudness directly.

        Mirrors NormalizeStage / matchering: the orchestrator special-cases
        this stage (normalize_to_lufs_target + run_isp_verification) and
        never calls :meth:`build_plugin`. Required by the ABC; raising keeps
        the contract honest (any accidental factory call surfaces loudly).
        """
        raise NotImplementedError(
            "LoudnessStage is applied directly via normalize_to_lufs_target "
            "in MasteringChain.process, not as a pedalboard plugin."
        )
