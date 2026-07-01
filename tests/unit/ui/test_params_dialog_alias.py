"""Phase 7 Plan 07-02 Task 1 — generic ParamsDialog contract.

The generic :class:`marmelade.ui.params_dialog.ParamsDialog` is the
shared params modal for mastering stages (and was formerly the Phase 6
heatmap params dialog). quick-260621-dt4 retired the heatmap panel and
its ``HeatmapParamsDialog`` backward-compat shim; the constructor
signature + browse-filter-optional contracts pinned here still hold for
the kept mastering callers.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QPushButton

from marmelade.audio.mastering.params import Param
from marmelade.ui.params_dialog import ParamsDialog


def test_phase_6_constructor_signature_preserved(qtbot, qapp) -> None:
    """Phase 6 callers pass ``(title, params, current_values, parent)`` kwargs.

    The new generic ``ParamsDialog`` must accept these same kwargs
    without raising.
    """
    params = {
        "f": Param(
            name="f",
            label="Float",
            kind="float",
            default=0.5,
            requires_recompute=False,
            min=0.0,
            max=1.0,
            step=0.1,
        ),
    }
    dialog = ParamsDialog(
        title="Test",
        params=params,
        current_values={},
        parent=None,
    )
    qtbot.add_widget(dialog)
    # Sanity — widget for "f" exists and exposes the spin attribute (the
    # float-widget container's value source).
    assert "f" in dialog._widgets
    assert hasattr(dialog._widgets["f"], "_spin")


def test_browse_filter_attribute_optional_no_crash(qtbot, qapp) -> None:
    """A choice-kind Param without a ``browse_filter`` attribute must not crash.

    Plan 07-02 Task 1 — the generic dialog reads ``getattr(p,
    "browse_filter", None)`` so old Param descriptors (without the
    Phase-7-Plan-5 Matchering extension) keep working — combobox renders
    with no Browse button.
    """
    params = {
        "c": Param(
            name="c",
            label="Choice",
            kind="choice",
            default="a",
            requires_recompute=False,
            choices=("a", "b", "c"),
        ),
    }
    dialog = ParamsDialog(
        title="Test",
        params=params,
        current_values={},
        parent=None,
    )
    qtbot.add_widget(dialog)
    # Widget is a QComboBox (single-widget — no Browse companion since
    # browse_filter is absent).
    w = dialog._widgets["c"]
    assert isinstance(w, QComboBox)
    # Optional: assert no QPushButton sibling labeled "Browse..." in the
    # form row (defense-in-depth — the absence of browse_filter must NOT
    # introduce a stray button).
    parent = w.parent()
    if parent is not None:
        browse_buttons = [
            b
            for b in parent.findChildren(QPushButton)
            if b.text().lower().startswith("browse")
        ]
        # No Browse button must exist in any parent chain for this Param.
        assert browse_buttons == []
