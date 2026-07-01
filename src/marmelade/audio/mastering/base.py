"""Abstract :class:`MasteringStage` base class — Phase 7 D-01 / D-02 / D-15.

Mirrors Phase 6's :class:`marmelade.heatmaps.base.Heatmap` ABC. Each
concrete subclass:

* Declares the ``name`` and ``display_name`` ClassVars (stable identifier
  + UI label).
* Implements :meth:`parameters` returning ``dict[str, Param]`` — the
  tunable surface for the gear-button dialog (D-01).
* Implements :meth:`build_plugin` returning a constructed
  :class:`pedalboard.Plugin` ready to be appended to a
  :class:`pedalboard.Pedalboard`.

Param overrides — the worker injects a flat ``dict`` via
``stage._param_overrides`` immediately before building plugins; the
:meth:`_get` helper reads through that override map and falls back to
each subclass's own default (Phase 6 ``getattr`` idiom verbatim).

N-3 invariant: zero PySide6 imports. The audio tier (DSP + Cache) stays
toolkit-free per CLAUDE.md "Extensibility" + 07-CONTEXT D-15.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pedalboard

from marmelade.audio.mastering.params import Param  # D-02 — import, do not duplicate


class MasteringStage(ABC):
    """Abstract base for one DSP stage in the per-keeper mastering chain.

    Subclasses set the ``name`` and ``display_name`` ClassVars and
    override :meth:`parameters` + :meth:`build_plugin`. Worker code
    injects per-keeper overrides into ``self._param_overrides`` before
    calling :meth:`build_plugin`; :meth:`_get` is the canonical reader.

    Subclass template::

        class HighPassStage(MasteringStage):
            name = "highpass"
            display_name = "High-pass filter"

            def parameters(self) -> dict[str, Param]:
                return {
                    "cutoff_hz": Param(
                        name="cutoff_hz", label="Cutoff", kind="float",
                        default=30.0, requires_recompute=True,
                        min=20.0, max=500.0, step=1.0, unit="Hz",
                    ),
                }

            def build_plugin(self) -> pedalboard.Plugin:
                return pedalboard.HighpassFilter(
                    cutoff_frequency_hz=float(self._get("cutoff_hz", 30.0))
                )
    """

    # Subclasses MUST set these (string identifiers). Declared as
    # ``ClassVar`` so IDEs and dataclass-style introspection can find
    # them and so the N-3 invariant grep does not flag any false
    # positives on subclass-instance access.
    name: ClassVar[str]
    display_name: ClassVar[str]

    @abstractmethod
    def parameters(self) -> dict[str, Param]:
        """Return tunable parameters as a ``dict`` keyed by ``Param.name``.

        Insertion order determines the dialog layout order in Plan 02's
        gear-button dialog. Returning an empty dict is valid (the
        dialog hides the gear button for that stage).
        """

    @abstractmethod
    def build_plugin(self) -> pedalboard.Plugin:
        """Return a constructed :class:`pedalboard.Plugin` for this stage.

        Reads tunable values via :meth:`_get`. The orchestrator only
        calls :meth:`build_plugin` if the stage is enabled in the
        keeper's mastering config.
        """

    def _get(self, name: str, default: Any) -> Any:
        """Read a parameter value: per-instance override first, then default.

        Mirrors the Phase 6 idiom in
        :meth:`marmelade.heatmaps.energy.EnergyHeatmap.process` —
        ``getattr(self, "_param_overrides", {}).get(name, default)`` — so
        worker-thread injection works without subclass coupling.
        """
        return getattr(self, "_param_overrides", {}).get(name, default)
