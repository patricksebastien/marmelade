"""Mastering audio tier — pedalboard-wrapping DSP stages + chain orchestrator.

Public surface (import convenience):

    from marmelade.audio.mastering import (
        MasteringStage, MasteringChain, Param,
        HighPassStage, LowPassStage, EqStage,
        CompressorStage, LimiterStage,
        config_hash, load_session_chain_snapshot, _STAGE_ORDER,
    )

Architectural invariants (Phase 7 — 07-CONTEXT.md D-15):

* Qt-free except for the single deliberate ``QSettings`` import in
  :mod:`marmelade.audio.mastering.chain` (PATTERNS Reuse Discipline 5
  option (b)) — the N-3 invariant test enforces no ``QtWidgets`` /
  ``QtGui`` imports anywhere under this subpackage.
* ``MasteringStage`` mirrors Phase 6's :class:`Heatmap` ABC (D-01),
  reusing the frozen :class:`Param` descriptor from
  :mod:`marmelade.audio.mastering.params` (D-02 — import, do not
  duplicate; ``Param`` was relocated there from ``heatmaps.base`` in
  quick-260701-muv when the AI/DSP heatmap backend was removed).
* Stage order is FIXED at ``_STAGE_ORDER`` (D-03 + UI-SPEC) — the
  orchestrator iterates in this order; disabled stages are skipped.
"""

from __future__ import annotations

from marmelade.audio.mastering.params import Param  # D-02 — re-export, do not duplicate

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.stages import (
    CompressorStage,
    DelayStage,
    DistortionStage,
    EndingFxStage,
    EqStage,
    FadeStage,
    HighPassStage,
    LimiterStage,
    LoudnessStage,
    LowPassStage,
    NormalizeStage,
    ReverbStage,
    Vst3Stage,
    build_limiter_subchain,
)

# NOTE: ``MasteringChain``, ``run_dsp_chain``, ``config_hash``,
# ``load_session_chain_snapshot`` and ``_STAGE_ORDER`` are re-exported
# below from :mod:`marmelade.audio.mastering.chain` once Task 3
# lands the orchestrator. We rebind them here via a try/import so this
# module is importable in the Task-2-only intermediate state (the
# stages tests don't need the chain).
try:  # pragma: no cover — re-export shim
    from marmelade.audio.mastering.chain import (  # noqa: F401
        MasteringChain,
        _STAGE_ORDER,
        config_hash,
        load_session_chain_snapshot,
        run_dsp_chain,
        run_pre_limiter_stages,
    )
except ImportError:  # chain.py not yet present (Task 2 intermediate state)
    pass

# quick-260623-p5b — genre mastering presets (pure data + matcher). Imported
# unconditionally: presets.py depends only on chain.config_hash (re-exported
# above) and stdlib typing.
from marmelade.audio.mastering.presets import (  # noqa: F401,E402
    MASTERING_PRESETS,
    match_preset,
    preset_names,
)

# Phase 07.1 — per-keeper ending-FX presets (pure data + matcher). Depends
# only on chain.config_hash (re-exported above) and stages.ending_fx.
from marmelade.audio.mastering.ending_fx_presets import (  # noqa: F401,E402
    ENDING_FX_PRESETS,
    ending_fx_preset_names,
    match_ending_fx,
)


__all__ = [
    "MasteringStage",
    "Param",
    "HighPassStage",
    "LowPassStage",
    "EqStage",
    "CompressorStage",
    "LimiterStage",
    "LoudnessStage",
    "NormalizeStage",
    "EndingFxStage",
    "DelayStage",
    "DistortionStage",
    "ReverbStage",
    "FadeStage",
    "Vst3Stage",
    "build_limiter_subchain",
    # Chain orchestrator surface (Task 3 — re-exported lazily above).
    "MasteringChain",
    "run_dsp_chain",
    "run_pre_limiter_stages",
    "config_hash",
    "load_session_chain_snapshot",
    "_STAGE_ORDER",
    # quick-260623-p5b — genre mastering presets.
    "MASTERING_PRESETS",
    "match_preset",
    "preset_names",
    # Phase 07.1 — per-keeper ending-FX presets.
    "ENDING_FX_PRESETS",
    "ending_fx_preset_names",
    "match_ending_fx",
]
