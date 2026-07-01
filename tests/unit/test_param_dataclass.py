"""Phase 6 Plan 6-01 Wave 0 (RED) — Param dataclass __post_init__ invariants.

Mirrors tests/unit/test_heatmap_base.py lines 87-151 (frozen dataclass +
__post_init__ validation). HM-07b — T-06-02 mitigation.

This file is RED until Plan 6-01 Task 2 lands ``Param`` in
``marmelade.heatmaps.base``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from marmelade.audio.mastering.params import Param  # NEW — Plan 6-01 Task 2 adds this.


def _valid_param_kwargs(**overrides) -> dict:
    """Helper: a known-valid float-kind Param constructor kwarg set."""
    base = dict(
        name="silence_threshold",
        label="Silence threshold",
        kind="float",
        default=0.5,
        requires_recompute=False,
        min=0.0,
        max=1.0,
        step=0.01,
    )
    base.update(overrides)
    return base


def test_param_is_frozen() -> None:
    """The dataclass must be frozen — mutation raises FrozenInstanceError."""
    p = Param(**_valid_param_kwargs())
    with pytest.raises(FrozenInstanceError):
        p.name = "other"  # type: ignore[misc]


def test_param_rejects_empty_name() -> None:
    """Empty ``name`` must raise ValueError."""
    with pytest.raises(ValueError):
        Param(**_valid_param_kwargs(name=""))


def test_param_rejects_empty_label() -> None:
    """Empty ``label`` must raise ValueError."""
    with pytest.raises(ValueError):
        Param(**_valid_param_kwargs(label=""))


def test_float_kind_requires_min_max() -> None:
    """A float-kind Param missing ``min`` (or ``max``) raises ValueError."""
    kwargs = _valid_param_kwargs()
    kwargs["min"] = None
    with pytest.raises(ValueError):
        Param(**kwargs)
    kwargs = _valid_param_kwargs()
    kwargs["max"] = None
    with pytest.raises(ValueError):
        Param(**kwargs)


def test_float_kind_rejects_min_ge_max() -> None:
    """A float-kind Param with ``min >= max`` raises ValueError."""
    with pytest.raises(ValueError):
        Param(**_valid_param_kwargs(min=1.0, max=1.0))
    with pytest.raises(ValueError):
        Param(**_valid_param_kwargs(min=2.0, max=1.0))


def test_int_kind_requires_min_max() -> None:
    """An int-kind Param missing ``min``/``max`` raises ValueError."""
    with pytest.raises(ValueError):
        Param(
            name="window_size",
            label="Window size",
            kind="int",
            default=5,
            requires_recompute=True,
        )


def test_choice_kind_requires_choices_tuple() -> None:
    """A choice-kind Param without ``choices`` raises ValueError."""
    with pytest.raises(ValueError):
        Param(
            name="algo",
            label="Algorithm",
            kind="choice",
            default="a",
            requires_recompute=True,
        )


def test_choice_kind_default_must_be_in_choices() -> None:
    """A choice-kind Param whose ``default`` is not in ``choices`` raises ValueError."""
    with pytest.raises(ValueError):
        Param(
            name="algo",
            label="Algorithm",
            kind="choice",
            default="z",
            requires_recompute=True,
            choices=("a", "b", "c"),
        )


def test_valid_float_kind_constructs() -> None:
    """A known-valid float-kind Param constructs and exposes its fields."""
    p = Param(**_valid_param_kwargs())
    assert p.name == "silence_threshold"
    assert p.kind == "float"
    assert p.default == 0.5
    assert p.requires_recompute is False
    assert p.min == 0.0
    assert p.max == 1.0


def test_valid_choice_kind_constructs() -> None:
    """A choice-kind Param with all-required fields constructs."""
    p = Param(
        name="mode",
        label="Mode",
        kind="choice",
        default="adaptive",
        requires_recompute=True,
        choices=("adaptive", "manual"),
    )
    assert p.kind == "choice"
    assert p.choices == ("adaptive", "manual")
    assert p.default == "adaptive"


def test_valid_bool_kind_constructs() -> None:
    """A bool-kind Param needs neither min/max nor choices."""
    p = Param(
        name="enabled",
        label="Enabled",
        kind="bool",
        default=True,
        requires_recompute=False,
    )
    assert p.kind == "bool"
    assert p.default is True


# ---------------------------------------------------------------------------
# Phase 7 Plan 07-05 — browse_filter optional field.
# ---------------------------------------------------------------------------


def test_param_browse_filter_field_default_none() -> None:
    """``browse_filter`` defaults to None for every Phase 6 caller.

    Phase 6 callers (Energy, Talking, Danceability, BPM, Harmonic Params)
    do not pass ``browse_filter`` — they get the default ``None`` and
    ParamsDialog renders a bare combobox or float/int/bool widget without
    a Browse button. Backward-compat invariant for Phase 7 — adding the
    field MUST NOT change any Phase 6 dialog behavior.
    """
    p = Param(**_valid_param_kwargs())
    assert p.browse_filter is None


def test_param_browse_filter_can_be_set_on_choice_kind() -> None:
    """A choice-kind Param accepts a non-None ``browse_filter`` string.

    Phase 7 Plan 07-05 — the Matchering reference picker passes a Qt
    file-filter string (``"Audio files (*.wav *.flac);;All files (*)"``).
    ParamsDialog (Plan 07-02 Task 1) reads it via
    ``getattr(p, "browse_filter", None)`` and renders a Browse button
    next to the combobox.
    """
    p = Param(
        name="reference_path",
        label="Reference",
        kind="choice",
        default="",
        requires_recompute=True,
        choices=("",),
        browse_filter="Audio files (*.wav *.flac);;All files (*)",
    )
    assert p.browse_filter == "Audio files (*.wav *.flac);;All files (*)"


def test_param_browse_filter_is_purely_informational() -> None:
    """``__post_init__`` does NOT validate ``browse_filter`` — any string is OK.

    The field is a free-form Qt file-filter string. The Param dataclass
    doesn't know about Qt and can't usefully validate the filter syntax;
    that's the dialog's job (which delegates to QFileDialog). Pin the
    no-validation invariant so a future "be helpful" tweak doesn't break
    callers that use an unconventional filter shape.
    """
    p = Param(
        name="something",
        label="Something",
        kind="choice",
        default="x",
        requires_recompute=False,
        choices=("x", "y"),
        browse_filter="garbage-string-not-really-a-qt-filter",
    )
    assert p.browse_filter == "garbage-string-not-really-a-qt-filter"
