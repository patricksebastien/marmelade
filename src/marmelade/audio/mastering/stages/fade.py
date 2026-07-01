"""Fade stage — OUTPUT-TIME fade in/out (quick-260626-o9y).

Fade is a VIRTUAL stage: it is NEVER applied inside
:class:`marmelade.audio.mastering.chain.MasteringChain.process` and is NEVER a
pedalboard plugin. The actual fade is applied at EXPORT / PREVIEW time by
``main_window`` (the per-clip fade-in / fade-out envelope), reading its
``enabled`` flag + ``duration_sec`` from the keeper's mastering config via the
:func:`fade_params` helper below.

Because the fade is applied at output time and NOT baked into the mastered
cache, toggling it (or changing its duration) must NEVER bust the mastered
cache. :func:`marmelade.audio.mastering.chain.config_hash` therefore drops the
``fade`` stage UNCONDITIONALLY (whether enabled or disabled) — distinct from the
``ending_fx`` / ``vst3`` special-case which only drops a DISABLED instance.

This module mirrors :class:`NormalizeStage` exactly: it declares a single
tunable Param so the Mastering dialog / dock can auto-render the stage's gear
dialog, and ``build_plugin`` raises :class:`NotImplementedError` (required by
the :class:`MasteringStage` ABC but never called for this stage).

N-3 invariant: this module imports ZERO Qt — only ``pedalboard`` (for the
``build_plugin`` return type) and the frozen :class:`Param` descriptor.
"""

from __future__ import annotations

from typing import ClassVar

import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param


class FadeStage(MasteringStage):
    """Declare the output-time fade's tunable surface (duration in seconds).

    The fade is applied at export/preview by ``main_window`` (NOT in the
    mastering chain). This class only declares the ``duration_sec`` Param so
    the Mastering dialog/dock render the stage's gear dialog via the same
    auto-render path as the other stages.
    """

    name: ClassVar[str] = "fade"
    display_name: ClassVar[str] = "Fade in/out"

    DURATION_SEC_DEFAULT: ClassVar[float] = 2.0

    def parameters(self) -> dict[str, Param]:
        return {
            "duration_sec": Param(
                name="duration_sec",
                label="Duration",
                kind="float",
                default=self.DURATION_SEC_DEFAULT,
                # Output-time fade — must NOT bust the mastered cache, so it
                # never triggers a recompute and is dropped from config_hash.
                requires_recompute=False,
                min=0.0,
                max=10.0,
                step=0.5,
                unit="s",
                description=(
                    "Length of the fade-in and fade-out applied at "
                    "export/preview time. This fade is NOT baked into the "
                    "mastered cache and never changes the mastering hash — "
                    "edit it freely without re-rendering."
                ),
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        """Not a pedalboard plugin — the fade is applied at export/preview.

        Mirrors :class:`NormalizeStage`: the fade is applied directly by the
        output path (``main_window``), never as a pedalboard plugin nor inside
        :meth:`MasteringChain.process`. Required by the ABC; raising keeps the
        contract honest (any accidental factory call surfaces loudly).
        """
        raise NotImplementedError(
            "FadeStage is applied at export/preview time (output-time fade), "
            "not as a pedalboard plugin nor in MasteringChain.process."
        )


def fade_params(mastering_cfg: dict | None) -> tuple[bool, float]:
    """Return ``(enabled, duration_sec)`` for the output-time fade.

    SINGLE SOURCE OF TRUTH for the fade default — read by BOTH export and
    preview (do NOT duplicate the (True, 2.0) default elsewhere).

    Defaults to ``(True, 2.0)`` (reproducing today's forced 2.0 s fade) when:
      * ``mastering_cfg`` is ``None`` (legacy keeper with no mastering), OR
      * it has no ``"fade"`` key, OR
      * the ``fade`` sub-dict is missing a field.

    ``enabled`` is coerced via ``bool()`` and ``duration_sec`` via ``float()``
    so a tolerant (possibly QSettings-typed) config still yields native types.
    """
    if not isinstance(mastering_cfg, dict):
        return (True, FadeStage.DURATION_SEC_DEFAULT)
    fade_cfg = mastering_cfg.get("fade")
    if not isinstance(fade_cfg, dict):
        return (True, FadeStage.DURATION_SEC_DEFAULT)
    enabled = bool(fade_cfg.get("enabled", True))
    duration_sec = float(fade_cfg.get("duration_sec", FadeStage.DURATION_SEC_DEFAULT))
    return (enabled, duration_sec)
