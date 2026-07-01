"""quick-260623-l7l Task 1 — :class:`LoudnessStage` declarative surface.

The Loudness stage mirrors :class:`NormalizeStage`: a virtual tail stage
that declares ONE ``target_lufs`` Param so the mastering dock can render its
gear dialog. It is applied directly in :meth:`MasteringChain.process` (via
:func:`normalize_to_lufs_target` + :func:`run_isp_verification`), NOT as a
pedalboard plugin — so ``build_plugin`` raises ``NotImplementedError``.
"""

from __future__ import annotations

import pytest

from marmelade.audio.mastering import LoudnessStage
from marmelade.audio.mastering.params import Param


def test_name_and_display_name() -> None:
    stage = LoudnessStage()
    assert stage.name == "loudness"
    assert stage.display_name == "Loudness (LUFS)"


def test_parameters_single_target_lufs() -> None:
    params = LoudnessStage().parameters()
    assert set(params) == {"target_lufs"}
    p = params["target_lufs"]
    assert isinstance(p, Param)
    assert p.kind == "float"
    assert p.default == -14.0
    assert p.min == -30.0
    assert p.max == -6.0
    assert p.requires_recompute is True


def test_build_plugin_raises() -> None:
    with pytest.raises(NotImplementedError):
        LoudnessStage().build_plugin()
