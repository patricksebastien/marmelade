"""Normalize stage — the FINAL per-keeper mastering step (quick-260621-gfq).

Redesign rationale: normalize used to be a whole-file toolbar action AND a
separate export-time transform. quick-260621-gfq folds it into the mastering
chain as its tail stage so per-keeper normalize lives in ONE coherent place
(applied after limiter + LUFS makeup + ISP verification + matchering).

The stage exposes a single ``target_db`` Param (default 0.0 dBFS — locked
decision #6, replacing the old negative keeper default). It is NOT a pedalboard
plugin: :class:`marmelade.audio.mastering.chain.MasteringChain.process`
applies it directly via
:func:`marmelade.audio.normalize.normalize_array` (DC-removal +
peak-to-target), mirroring how matchering has no pedalboard ``build_plugin``
path. ``build_plugin`` therefore raises :class:`NotImplementedError` (it is
required by the :class:`MasteringStage` ABC but never called for this stage).
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class NormalizeStage(MasteringStage):
    """Declare the normalize stage's tunable surface (target dB).

    The chain orchestrator applies normalize directly via
    :func:`marmelade.audio.normalize.normalize_array` — this class only
    declares the ``target_db`` Param so the Mastering dock can render the
    stage's gear dialog (same auto-render path as the pedalboard stages).
    """

    name: ClassVar[str] = "normalize"
    display_name: ClassVar[str] = "Normalize"

    TARGET_DB_DEFAULT: ClassVar[float] = 0.0

    def parameters(self) -> dict[str, Param]:
        return {
            "target_db": Param(
                name="target_db",
                label="Target",
                kind="float",
                default=self.TARGET_DB_DEFAULT,
                requires_recompute=True,
                min=-60.0,
                max=0.0,
                step=0.5,
                unit="dBFS",
                description=(
                    "Peak target after DC-removal. Default 0 dBFS scales the "
                    "loudest sample to full scale (applied last in the chain)."
                ),
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        """Not a pedalboard plugin — the chain applies normalize directly.

        Mirrors matchering: the orchestrator special-cases this stage and
        never calls :meth:`build_plugin`. Required by the ABC; raising keeps
        the contract honest (any accidental factory call surfaces loudly).
        """
        raise NotImplementedError(
            "NormalizeStage is applied directly via normalize_array in "
            "MasteringChain.process, not as a pedalboard plugin."
        )
