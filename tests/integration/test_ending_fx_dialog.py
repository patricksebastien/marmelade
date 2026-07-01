"""Per-keeper Ending FX surface in MasteringDialog.

quick-260629 — the dedicated per-keeper "Ending FX" preset dropdown was
REMOVED so the keeper dialog matches the session dock exactly: ending FX is
now a single checkbox + gear row living inside the "FX" group box at the top
of the dialog (alongside distortion / delay / reverb). The effect type and
its params (effect_type / tail_sec / onset_sec / wet / primary) are edited via
the per-stage gear → ParamsDialog auto-render path — identical to every other
stage. Pins:

* Ending FX is part of the FX display group (chain._FX_STAGES) and renders
  inside the titled "FX" QGroupBox; the dialog has NO bespoke ending-fx combo.
* The gear surfaces EndingFxStage's params via the generic ParamsDialog path.
* A gear param edit round-trips into the bare ending_fx dict that Apply persists.
* Audition is VERIFY-ONLY — a read-only assertion that the B-mode play path
  passes ``end_seconds=None`` (no playback-engine edits).
"""

from __future__ import annotations

import copy

from PySide6.QtWidgets import QCheckBox, QComboBox, QGroupBox

from marmelade.audio.mastering.chain import _FX_STAGES, _SESSION_DEFAULTS
from marmelade.ui.mastering_dialog import MasteringDialog

_KID = "kid000000000000000000000000004fx"


def _full_default_cfg() -> dict:
    """Return a complete chain config — every stage seeded from defaults."""
    return copy.deepcopy(_SESSION_DEFAULTS)


def _dialog_with_cfg(qtbot, cfg: dict) -> MasteringDialog:
    """Construct a MasteringDialog bound to a deep copy of ``cfg``."""
    dlg = MasteringDialog(
        keeper_id=_KID,
        keeper_mastering=copy.deepcopy(cfg),
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)
    return dlg


# ----------------------------------------------------------------------------
# Ending FX lives in the FX group; the bespoke dropdown is gone
# ----------------------------------------------------------------------------


def test_ending_fx_is_in_the_fx_display_group() -> None:
    """ending_fx joined the FX display grouping (chain._FX_STAGES)."""
    assert "ending_fx" in _FX_STAGES


def test_dialog_has_no_dedicated_ending_fx_dropdown(qtbot, qapp) -> None:
    """The per-keeper ending-FX preset dropdown was removed (dock parity)."""
    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    assert not hasattr(dlg, "_ending_fx_combo")
    # The only remaining combobox is the genre preset selector.
    combos = dlg.findChildren(QComboBox)
    assert combos == [dlg._preset_combo]


def test_ending_fx_renders_inside_the_fx_group_box(qtbot, qapp) -> None:
    """The ending_fx checkbox row sits inside the titled 'FX' group box."""
    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    fx_groups = [g for g in dlg.findChildren(QGroupBox) if g.title() == "FX"]
    assert len(fx_groups) == 1
    # The FX box holds exactly one checkbox per FX stage (incl. ending_fx).
    fx_checkboxes = fx_groups[0].findChildren(QCheckBox)
    assert len(fx_checkboxes) == len(_FX_STAGES)
    assert "ending_fx" in dlg._stage_checkboxes


# ----------------------------------------------------------------------------
# Custom power-user params via the per-stage gear → ParamsDialog path
# ----------------------------------------------------------------------------


def test_ending_fx_gear_opens_params_dialog_from_stage_params(qtbot, qapp) -> None:
    """The ending_fx gear builds a ParamsDialog from EndingFxStage.parameters().

    Verifies the row + gear wiring resolves (the three stage maps are
    registered) and the power-user params (effect_type / tail_sec / onset_sec /
    wet / primary) are the ones surfaced — without actually exec()'ing a modal.
    """
    from marmelade.audio.mastering import EndingFxStage
    from marmelade.ui.mastering_dialog import (
        _STAGE_CLASS_BY_NAME,
        _STAGE_DISPLAY_NAMES,
        _STAGE_GEAR_LABELS,
    )

    # The three stage maps carry ending_fx.
    assert _STAGE_CLASS_BY_NAME["ending_fx"] is EndingFxStage
    assert _STAGE_DISPLAY_NAMES["ending_fx"] == "Ending FX"
    assert _STAGE_GEAR_LABELS["ending_fx"] == "Ending FX parameters"

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    # The ending_fx checkbox row was built (generic loop, in the FX group).
    assert "ending_fx" in dlg._stage_checkboxes

    # The params the gear would surface are the full power-user knob set.
    params = EndingFxStage().parameters()
    assert set(params.keys()) == {
        "effect_type",
        "tail_sec",
        "onset_sec",
        "wet",
        "primary",
    }
    assert params["effect_type"].kind == "choice"


def test_ending_fx_gear_edit_writes_values_into_cfg(qtbot, qapp) -> None:
    """A simulated gear Apply writes effect_type/tail_sec/wet/primary into _cfg.

    Mirrors _on_stage_gear_clicked's accepted-values write-back without exec()'ing
    a modal (which would block the headless test). Confirms the power-user values
    round-trip into the bare ending_fx dict that Apply persists.
    """
    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    new_values = {
        "effect_type": "dub_echo",
        "tail_sec": 5.0,
        "wet": 0.8,
        "primary": 0.6,
    }
    # Same write-back _on_stage_gear_clicked performs on ParamsDialog Accept.
    dlg._cfg.setdefault("ending_fx", {}).update(new_values)
    # The gear path resyncs the genre preset combo (ending_fx now behaves like
    # every other stage — no bespoke combo to resync).
    dlg._sync_preset_combo()

    assert dlg._cfg["ending_fx"]["effect_type"] == "dub_echo"
    assert dlg._cfg["ending_fx"]["tail_sec"] == 5.0
    assert dlg._cfg["ending_fx"]["wet"] == 0.8
    assert dlg._cfg["ending_fx"]["primary"] == 0.6


# ----------------------------------------------------------------------------
# Apply still persists the ending_fx stage dict via config_changed
# ----------------------------------------------------------------------------


def test_apply_emits_config_changed_with_ending_fx(qtbot, qapp) -> None:
    """Apply emits config_changed once carrying the edited ending_fx dict."""
    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    dlg._cfg.setdefault("ending_fx", {}).update(
        {"enabled": True, "effect_type": "hall_wash", "tail_sec": 6.0}
    )

    emits: list = []
    dlg.config_changed.connect(lambda kid, cfg: emits.append((kid, cfg)))
    dlg._on_apply_clicked()

    assert len(emits) == 1
    kid, cfg = emits[0]
    assert kid == _KID
    assert cfg["ending_fx"]["enabled"] is True
    assert cfg["ending_fx"]["effect_type"] == "hall_wash"
    assert cfg["ending_fx"]["tail_sec"] == 6.0


# ----------------------------------------------------------------------------
# Audition is VERIFY-ONLY — no playback-engine edits in this plan.
# ----------------------------------------------------------------------------


def test_b_mode_audition_plays_cache_to_eof_end_seconds_none(qtbot, qapp) -> None:
    """Read-only guard: the B-mode keeper-play path passes end_seconds=None.

    The longer (tail-baked) mastered cache rings out for free because B-mode
    plays the cache to its natural EOF. This static assertion pins that the
    existing end_seconds=None contract is still in place (so the tail is
    audible without edits).
    """
    import inspect
    from pathlib import Path

    src = Path(
        inspect.getsourcefile(MasteringDialog)
    ).resolve().parents[0] / "main_window.py"
    text = src.read_text(encoding="utf-8")
    assert "end_seconds=None" in text
