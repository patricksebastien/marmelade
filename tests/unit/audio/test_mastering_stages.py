"""Wave 0 RED stub — concrete :class:`MasteringStage` subclasses.

Pinned invariants:
* Every subclass declares non-empty ``name`` and ``display_name`` ClassVars.
* ``parameters()`` returns a non-empty ``dict[str, Param]`` keyed by
  ``Param.name`` (D-01 contract).
* Every ``Param`` is the actual :class:`Param` re-export from
  :mod:`marmelade.heatmaps.base` (D-02 — no duplication).

Phase 7 — Plan 01 Wave 0 (07-01-PLAN.md Task 1).
"""

from __future__ import annotations

import pytest

from marmelade.audio.mastering.params import Param


_STAGE_CLASS_NAMES = (
    "HighPassStage",
    "LowPassStage",
    "EqStage",
    "CompressorStage",
    "LimiterStage",
)


def _stage_class(name: str):
    """Lookup a stage class by name; defer the import until test time.

    Deferring the import to test-body time (not module-load time) is
    necessary so the RED state — where the package does not yet exist —
    still allows pytest to collect this test module. Test bodies then
    fail with ``ModuleNotFoundError`` until Task 2 lands the GREEN
    implementation, at which point each test passes.
    """
    from marmelade.audio.mastering import stages  # noqa: WPS433 — local import

    return getattr(stages, name)


@pytest.mark.parametrize("stage_cls_name", _STAGE_CLASS_NAMES)
def test_stage_declares_classvars_and_parameters(stage_cls_name):
    """Each stage class has non-empty ``name`` / ``display_name`` and
    a non-empty ``parameters()`` dict keyed by ``Param.name``.
    """
    stage_cls = _stage_class(stage_cls_name)
    # ClassVars — non-empty strings.
    assert isinstance(stage_cls.name, str) and stage_cls.name, (
        f"{stage_cls.__name__}.name must be a non-empty str"
    )
    assert isinstance(stage_cls.display_name, str) and stage_cls.display_name, (
        f"{stage_cls.__name__}.display_name must be a non-empty str"
    )
    # Parameters dict — non-empty mapping keyed by Param.name.
    inst = stage_cls()
    params = inst.parameters()
    assert isinstance(params, dict)
    assert params, f"{stage_cls.__name__}.parameters() must return at least one Param"
    for key, p in params.items():
        assert isinstance(p, Param), (
            f"{stage_cls.__name__}.parameters()[{key!r}] is "
            f"{type(p).__name__}, expected Param"
        )
        assert key == p.name, (
            f"{stage_cls.__name__}.parameters() key {key!r} must equal "
            f"Param.name {p.name!r}"
        )


def test_eq_stage_surface():
    """quick-260623-k7t — EqStage exposes the 3-band Low/Mid/High surface."""
    eq_cls = _stage_class("EqStage")
    assert eq_cls.name == "eq"
    assert eq_cls.display_name == "EQ"

    params = eq_cls().parameters()
    assert list(params.keys()) == ["low_db", "mid_db", "high_db"]
    assert [p.label for p in params.values()] == ["Low", "Mid", "High"]
    for p in params.values():
        assert p.kind == "float"
        assert p.min == -12.0
        assert p.max == 12.0
        assert p.step == 0.5
        assert p.unit == "dB"
        assert p.requires_recompute is True


def test_stage_param_reexport_identity():
    """Stages must use the same ``Param`` dataclass as
    :mod:`marmelade.heatmaps.base` (D-02 — import, do not duplicate).
    """
    from marmelade.audio.mastering import Param as ReexportedParam

    assert ReexportedParam is Param
