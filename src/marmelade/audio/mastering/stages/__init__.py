"""Concrete :class:`MasteringStage` subclasses — Phase 7 D-01 / D-03.

Each module under this subpackage defines exactly one stage class. The
``__init__`` re-exports the six concrete classes plus the
:func:`build_limiter_subchain` helper used by the chain orchestrator to
enforce the -1 dBTP ceiling (Pitfall 1).
"""

from __future__ import annotations

from marmelade.audio.mastering.stages.compressor import CompressorStage
from marmelade.audio.mastering.stages.delay import DelayStage
from marmelade.audio.mastering.stages.distortion import DistortionStage
from marmelade.audio.mastering.stages.ending_fx import EndingFxStage
from marmelade.audio.mastering.stages.eq import EqStage
from marmelade.audio.mastering.stages.fade import FadeStage
from marmelade.audio.mastering.stages.highpass import HighPassStage
from marmelade.audio.mastering.stages.limiter import (
    LimiterStage,
    build_limiter_subchain,
)
from marmelade.audio.mastering.stages.loudness import LoudnessStage
from marmelade.audio.mastering.stages.lowpass import LowPassStage
from marmelade.audio.mastering.stages.normalize import NormalizeStage
from marmelade.audio.mastering.stages.reverb import ReverbStage
from marmelade.audio.mastering.stages.vst3 import Vst3Stage

__all__ = [
    "CompressorStage",
    "DelayStage",
    "DistortionStage",
    "EndingFxStage",
    "EqStage",
    "FadeStage",
    "HighPassStage",
    "LimiterStage",
    "LoudnessStage",
    "LowPassStage",
    "NormalizeStage",
    "ReverbStage",
    "Vst3Stage",
    "build_limiter_subchain",
]
