"""Phase 7 Plan 07-02 Task 3 (RED) — MasteringDialog modal per-keeper editor.

D-05 (modal) + D-04 (snapshot-not-link) + UI-SPEC §MasteringDialog
interactions. Pins:

* Dialog requires ``keeper_mastering != None`` at construction
  (defensive assertion — Plan 07-03 owns the snapshot-at-creation
  semantics; the dialog never re-snapshots).
* Pre-populated values come from the keeper's mastering dict.
* Reset to session chain overwrites the in-dialog state with
  ``load_session_chain_snapshot()`` and re-renders widgets.
* Apply emits ``config_changed(keeper_id, cfg)`` and accepts.
* Discard changes (Cancel-role with relabel) rejects silently.
* ``isModal() is True``.
* Per-stage gear opens a ParamsDialog with the correct UI-SPEC title.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDialog

from marmelade.audio.mastering.chain import (
    _SESSION_DEFAULTS,
    load_session_chain_snapshot,
)
from marmelade.ui.mastering_dialog import MasteringDialog


def _full_default_cfg() -> dict:
    """Return a complete chain config — every stage seeded from defaults."""
    import copy

    return copy.deepcopy(_SESSION_DEFAULTS)


def test_opens_with_keeper_config_when_mastering_set(qtbot, qapp) -> None:
    """Dialog opens with the keeper's mastering dict pre-loaded.

    Limiter with ``ceiling_dbtp = -2.0`` (different from default -1.0) —
    the dialog's internal config must reflect that.
    """
    keeper_cfg = _full_default_cfg()
    keeper_cfg["limiter"] = {
        "enabled": True,
        "ceiling_dbtp": -2.0,
        "release_ms": 50.0,
    }
    dlg = MasteringDialog(
        keeper_id="kid00000000000000000000000000001",
        keeper_mastering=keeper_cfg,
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)

    assert dlg._cfg["limiter"]["enabled"] is True
    assert dlg._cfg["limiter"]["ceiling_dbtp"] == pytest.approx(-2.0)
    assert dlg._cfg["limiter"]["release_ms"] == pytest.approx(50.0)


def test_rejects_mastering_none_on_construction(qtbot, qapp) -> None:
    """MasteringDialog requires keeper_mastering != None (defensive contract).

    Plan 07-03 owns the snapshot-at-keeper-creation hook; the dialog must
    NOT re-snapshot to avoid breaking D-04's snapshot-not-link semantics.
    """
    with pytest.raises(AssertionError):
        MasteringDialog(
            keeper_id="kid00000000000000000000000000001",
            keeper_mastering=None,
            keeper_range="00:00:10 – 00:00:20",
        )


def test_reset_to_session_chain_overwrites_cfg_and_rerenders_widgets(
    qtbot, qapp
) -> None:
    """Reset button overwrites cfg with load_session_chain_snapshot() output."""
    keeper_cfg = _full_default_cfg()
    # Custom: HP enabled, differs from session defaults.
    keeper_cfg["highpass"] = {"enabled": True, "cutoff_hz": 80.0}
    dlg = MasteringDialog(
        keeper_id="kid00000000000000000000000000001",
        keeper_mastering=keeper_cfg,
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)

    dlg._on_reset_to_session_chain()

    expected = load_session_chain_snapshot()
    assert dlg._cfg == expected
    # Per-stage checkbox visual states match the new cfg.
    for stage, row in dlg._stage_checkboxes.items():
        assert row.isChecked() == bool(expected[stage]["enabled"])


def test_apply_emits_config_changed_with_keeper_id_and_cfg(qtbot, qapp) -> None:
    """Click Apply → config_changed(keeper_id, cfg) fires + dialog accepts."""
    keeper_cfg = _full_default_cfg()
    keeper_id = "kid00000000000000000000000000002"
    dlg = MasteringDialog(
        keeper_id=keeper_id,
        keeper_mastering=keeper_cfg,
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)

    with qtbot.waitSignal(dlg.config_changed, timeout=1000) as blocker:
        dlg._on_apply_clicked()

    payload = blocker.args
    assert payload[0] == keeper_id
    assert payload[1] == keeper_cfg
    assert dlg.result() == QDialog.DialogCode.Accepted


def test_discard_changes_emits_no_signal_and_rejects(qtbot, qapp) -> None:
    """Discard changes (Cancel-role button) → no config_changed; dialog rejects."""
    keeper_cfg = _full_default_cfg()
    dlg = MasteringDialog(
        keeper_id="kid00000000000000000000000000003",
        keeper_mastering=keeper_cfg,
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)

    spy: list[tuple] = []
    dlg.config_changed.connect(lambda kid, cfg: spy.append((kid, cfg)))

    dlg.reject()
    assert spy == []
    assert dlg.result() == QDialog.DialogCode.Rejected


def test_modal_property_set(qtbot, qapp) -> None:
    """Dialog must be modal (D-05)."""
    keeper_cfg = _full_default_cfg()
    dlg = MasteringDialog(
        keeper_id="kid00000000000000000000000000004",
        keeper_mastering=keeper_cfg,
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)
    assert dlg.isModal() is True


def test_per_stage_gear_opens_paramsdialog_with_correct_title(
    qtbot, qapp, monkeypatch
) -> None:
    """Compressor gear opens a ParamsDialog titled "Compressor parameters".

    UI-SPEC §Per-stage gear button — accessible name + tooltip table.
    Captures the title via a replacement ParamsDialog class that
    bypasses Qt entirely (records the constructor args + returns Rejected
    from exec). Avoids the recursion that arises if we monkeypatch
    __init__ in place and try to call the original.
    """
    keeper_cfg = _full_default_cfg()
    dlg = MasteringDialog(
        keeper_id="kid00000000000000000000000000005",
        keeper_mastering=keeper_cfg,
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)

    captured_titles: list[str] = []

    class _StubParamsDialog:
        def __init__(self, *args, **kwargs):
            captured_titles.append(kwargs.get("title", ""))

        def exec(self):
            return QDialog.DialogCode.Rejected

        def accepted_values(self):
            return {}

    # Replace the symbol the dialog module looked up at import time.
    from marmelade.ui import mastering_dialog as md_module

    monkeypatch.setattr(md_module, "ParamsDialog", _StubParamsDialog)

    dlg._on_stage_gear_clicked("compressor")

    assert "Compressor parameters" in captured_titles


# ----------------------------------------------------------------------------
# quick-260624-h78 — per-keeper genre preset combobox
#
# Mirrors the main MasteringDock preset combobox (quick-260623-p5b) but scoped
# to a single keeper's in-dialog working config (self._cfg) rather than the
# session-wide QSettings chain. The preset data + matcher are REUSED as-is from
# audio/mastering/presets.py (N-3 holds — no duplication).
# ----------------------------------------------------------------------------


def _dialog_with_cfg(qtbot, cfg: dict) -> MasteringDialog:
    """Construct a MasteringDialog bound to ``cfg`` and register with qtbot."""
    import copy

    dlg = MasteringDialog(
        keeper_id="kid0000000000000000000000000h78a",
        keeper_mastering=copy.deepcopy(cfg),
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)
    return dlg


def test_preset_combo_lists_custom_first_then_presets_in_order(qtbot, qapp) -> None:
    """_preset_combo has 'Custom' at index 0 then the 10 presets in lockstep."""
    from marmelade.audio.mastering import preset_names

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    combo = dlg._preset_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["Custom"] + preset_names()
    assert combo.count() == 11


def test_apply_preset_overwrites_cfg_and_flips_checkboxes(qtbot, qapp) -> None:
    """_apply_preset('Techno') deep-copies the preset into _cfg + syncs checkboxes."""
    from marmelade.audio.mastering import MASTERING_PRESETS

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    dlg._apply_preset("Techno")

    assert dlg._cfg == MASTERING_PRESETS["Techno"]
    # _cfg must be an independent deep copy (mutating it must not touch the source).
    assert dlg._cfg is not MASTERING_PRESETS["Techno"]
    for stage, params in MASTERING_PRESETS["Techno"].items():
        checkbox = dlg._stage_checkboxes[stage]
        assert checkbox.isChecked() is bool(params.get("enabled", False)), stage


def test_selecting_preset_via_combobox_applies_to_cfg(qtbot, qapp) -> None:
    """setCurrentIndex(findText('Dubstep')) drives the real currentIndexChanged path."""
    from marmelade.audio.mastering import MASTERING_PRESETS

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    target = dlg._preset_combo.findText("Dubstep")
    dlg._preset_combo.setCurrentIndex(target)

    assert dlg._cfg == MASTERING_PRESETS["Dubstep"]


def test_custom_selection_is_a_noop(qtbot, qapp) -> None:
    """_on_preset_selected(0) ('Custom') does not mutate _cfg."""
    import copy

    cfg = _full_default_cfg()
    dlg = _dialog_with_cfg(qtbot, cfg)
    before = copy.deepcopy(dlg._cfg)
    dlg._on_preset_selected(0)
    assert dlg._cfg == before


def test_open_time_sync_matches_preset(qtbot, qapp) -> None:
    """A dialog opened with a preset cfg points the combobox at that preset."""
    import copy

    from marmelade.audio.mastering import MASTERING_PRESETS

    dlg = _dialog_with_cfg(qtbot, copy.deepcopy(MASTERING_PRESETS["House"]))
    assert dlg._preset_combo.currentText() == "House"


def test_open_time_sync_non_matching_is_custom(qtbot, qapp) -> None:
    """A dialog opened with a non-preset cfg points the combobox at 'Custom'."""
    cfg = _full_default_cfg()
    # Perturb a param so no preset config_hash matches.
    cfg.setdefault("limiter", {})["enabled"] = True
    cfg["limiter"]["ceiling_dbtp"] = -7.3
    dlg = _dialog_with_cfg(qtbot, cfg)
    assert dlg._preset_combo.currentText() == "Custom"


def test_manual_toggle_resyncs_combo_to_custom_without_double_apply(
    qtbot, qapp
) -> None:
    """After applying a preset, a genuine checkbox toggle moves combo to Custom.

    The cfg must reflect EXACTLY the one toggle (not a re-applied preset), and
    no spurious config_changed emit fires (apply only happens on the Apply btn).
    """
    import copy

    from marmelade.audio.mastering import MASTERING_PRESETS

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    dlg._apply_preset("Pop")
    applied = copy.deepcopy(dlg._cfg)

    emits: list = []
    dlg.config_changed.connect(lambda kid, cfg: emits.append((kid, cfg)))

    # Toggle one currently-enabled stage OFF — a genuine user edit.
    enabled_stage = next(
        s for s, p in MASTERING_PRESETS["Pop"].items() if p.get("enabled")
    )
    checkbox = dlg._stage_checkboxes[enabled_stage]
    checkbox.setChecked(False)

    assert dlg._preset_combo.currentText() == "Custom"
    assert emits == [], "manual toggle must not emit config_changed"
    # cfg differs from the applied preset only by that single enabled flag.
    expected = copy.deepcopy(applied)
    expected[enabled_stage]["enabled"] = False
    assert dlg._cfg == expected


def test_apply_persists_selected_preset_exactly_once(qtbot, qapp) -> None:
    """After _apply_preset('Trance'), Apply emits config_changed once w/ the preset."""
    from marmelade.audio.mastering import MASTERING_PRESETS

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    dlg._apply_preset("Trance")

    emits: list = []
    dlg.config_changed.connect(lambda kid, cfg: emits.append((kid, cfg)))
    dlg._on_apply_clicked()

    assert len(emits) == 1
    kid, cfg = emits[0]
    assert kid == "kid0000000000000000000000000h78a"
    assert cfg == MASTERING_PRESETS["Trance"]


# ----------------------------------------------------------------------
# quick-260626 close-to-commit hardening — Apply + VST3 gear are LOCKED
# while the out-of-process editor QProcess is alive, and re-enabled in the
# finished/on_done path. This makes the prime timing-race (Apply clicked
# before the editor closed → stale `enabled=False` cfg → silent passthrough)
# structurally impossible. We monkeypatch the configure_vst3 seam (the Qt
# QProcess glue is NOT unit-tested headlessly per its module docstring) and
# drive its on_started / on_done callbacks directly, mirroring the
# injectable-loader / fake-process pattern used for the worker.
# ----------------------------------------------------------------------


class _FakeConfigureVst3:
    """Records configure_vst3 callbacks so a test can drive the editor lifecycle.

    Replaces ``marmelade.ui.vst3_config.configure_vst3``. On call it fires
    ``on_started`` synchronously (matching the real helper, which calls it
    right after QProcess.start()) and stashes ``on_done`` + ``cfg`` so the
    test can simulate the editor closing later.
    """

    def __init__(self) -> None:
        self.cfg: dict | None = None
        self.on_done = None
        self.on_started = None
        self.calls = 0

    def __call__(self, parent, cfg, on_done=None, on_started=None) -> bool:
        self.calls += 1
        self.cfg = cfg
        self.on_done = on_done
        self.on_started = on_started
        if on_started is not None:
            on_started()  # editor process started → caller should lock
        return True

    def finish(self, *, changed: bool, captured_enabled: bool = True) -> None:
        """Simulate the editor closing: mutate cfg (on success) then on_done."""
        if changed and self.cfg is not None:
            # The real QProcess.finished handler captures enabled+state here.
            self.cfg["enabled"] = captured_enabled
            self.cfg["plugin_path"] = "/fake/oXygen.vst3"
            self.cfg["state_b64"] = "ZmFrZS1zdGF0ZQ=="
        if self.on_done is not None:
            self.on_done(changed)


def test_vst3_editor_disables_apply_and_gear_while_running(
    qtbot, qapp, monkeypatch
) -> None:
    """Opening the VST3 editor disables Apply + the VST3 gear while it runs."""
    import marmelade.ui.vst3_config as vst3_config

    fake = _FakeConfigureVst3()
    monkeypatch.setattr(vst3_config, "configure_vst3", fake)

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    assert dlg._apply_btn.isEnabled()
    assert dlg._stage_gears["vst3"].isEnabled()

    dlg._on_vst3_gear_clicked()

    # on_started fired → both locked while the editor process is alive.
    assert fake.calls == 1
    assert not dlg._apply_btn.isEnabled(), "Apply must be locked mid-edit"
    assert not dlg._stage_gears["vst3"].isEnabled(), "gear must be locked"


def test_vst3_editor_reenables_apply_and_gear_after_finished(
    qtbot, qapp, monkeypatch
) -> None:
    """When the editor closes (success), Apply + gear re-enable and checkbox ticks."""
    import marmelade.ui.vst3_config as vst3_config

    fake = _FakeConfigureVst3()
    monkeypatch.setattr(vst3_config, "configure_vst3", fake)

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    dlg._on_vst3_gear_clicked()
    assert not dlg._apply_btn.isEnabled()

    # Editor closes, capturing enabled=True state.
    fake.finish(changed=True, captured_enabled=True)

    assert dlg._apply_btn.isEnabled(), "Apply re-enabled after editor finished"
    assert dlg._stage_gears["vst3"].isEnabled(), "gear re-enabled after finish"
    # Captured state is reflected: cfg enabled + row checkbox ticked.
    assert dlg._cfg["vst3"]["enabled"] is True
    assert dlg._stage_checkboxes["vst3"].isChecked() is True


def test_vst3_editor_reenables_apply_and_gear_after_failure(
    qtbot, qapp, monkeypatch
) -> None:
    """Even on editor failure (changed=False) the lock is released (no strand)."""
    import marmelade.ui.vst3_config as vst3_config

    fake = _FakeConfigureVst3()
    monkeypatch.setattr(vst3_config, "configure_vst3", fake)

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    dlg._on_vst3_gear_clicked()
    assert not dlg._apply_btn.isEnabled()

    # Worker exited with an error → changed=False, cfg NOT mutated.
    fake.finish(changed=False)

    assert dlg._apply_btn.isEnabled(), "Apply re-enabled even after failure"
    assert dlg._stage_gears["vst3"].isEnabled()
    # No spurious enable of the vst3 stage on failure.
    assert dlg._stage_checkboxes["vst3"].isChecked() is False


# ----------------------------------------------------------------------
# quick-260626-mih — Loudness (LUFS) / Normalize mutual exclusion (DIALOG)
#
# chain.py BYPASSES the Normalize tail whenever Loudness is enabled, so "both
# checked" is a meaningless UI state. Checking one auto-unchecks the other
# (cfg + checkbox); unchecking a stage NEVER touches its sibling. The preset
# apply path flips checkboxes under its own QSignalBlocker, so it bypasses the
# mutual-exclusion handler — preset integrity is preserved.
# ----------------------------------------------------------------------


def test_checking_loudness_unchecks_normalize_dialog(qtbot, qapp) -> None:
    """Checking Loudness auto-unchecks Normalize (cfg + checkbox)."""
    cfg = _full_default_cfg()
    cfg["normalize"]["enabled"] = True
    dlg = _dialog_with_cfg(qtbot, cfg)
    assert dlg._stage_checkboxes["normalize"].isChecked() is True

    dlg._stage_checkboxes["loudness"].setChecked(True)

    assert dlg._cfg["loudness"]["enabled"] is True
    assert dlg._cfg["normalize"]["enabled"] is False
    assert dlg._stage_checkboxes["normalize"].isChecked() is False


def test_checking_normalize_unchecks_loudness_dialog(qtbot, qapp) -> None:
    """Checking Normalize auto-unchecks Loudness (cfg + checkbox)."""
    cfg = _full_default_cfg()
    cfg["loudness"]["enabled"] = True
    dlg = _dialog_with_cfg(qtbot, cfg)
    assert dlg._stage_checkboxes["loudness"].isChecked() is True

    dlg._stage_checkboxes["normalize"].setChecked(True)

    assert dlg._cfg["normalize"]["enabled"] is True
    assert dlg._cfg["loudness"]["enabled"] is False
    assert dlg._stage_checkboxes["loudness"].isChecked() is False


def test_unchecking_loudness_leaves_normalize_untouched_dialog(qtbot, qapp) -> None:
    """Unchecking Loudness must NOT touch Normalize (only a CHECK flips sibling).

    Drive the whole scenario through checkbox toggles: enable Normalize, then
    check Loudness (Normalize auto-off), then uncheck Loudness — Normalize must
    stay OFF (no spurious re-enable / flip).
    """
    cfg = _full_default_cfg()
    dlg = _dialog_with_cfg(qtbot, cfg)

    dlg._stage_checkboxes["normalize"].setChecked(True)
    assert dlg._cfg["normalize"]["enabled"] is True

    # Checking loudness forces normalize OFF.
    dlg._stage_checkboxes["loudness"].setChecked(True)
    assert dlg._cfg["normalize"]["enabled"] is False
    assert dlg._stage_checkboxes["normalize"].isChecked() is False

    # Unchecking loudness must leave normalize untouched (still OFF).
    dlg._stage_checkboxes["loudness"].setChecked(False)
    assert dlg._cfg["loudness"]["enabled"] is False
    assert dlg._cfg["normalize"]["enabled"] is False
    assert dlg._stage_checkboxes["normalize"].isChecked() is False


def test_preset_apply_preserves_exact_flags_despite_mutual_exclusion_dialog(
    qtbot, qapp
) -> None:
    """A preset enabling loudness + disabling normalize keeps its EXACT flags.

    The mutual-exclusion handler must NOT fire through the preset apply path
    (which flips checkboxes under its own QSignalBlocker). 'House' has loudness
    ON + normalize OFF — applying it must yield exactly that, with every other
    stage flag matching the preset dict.
    """
    from marmelade.audio.mastering import MASTERING_PRESETS

    name = "House"
    assert MASTERING_PRESETS[name]["loudness"]["enabled"] is True
    assert MASTERING_PRESETS[name]["normalize"]["enabled"] is False

    dlg = _dialog_with_cfg(qtbot, _full_default_cfg())
    dlg._apply_preset(name)

    # cfg matches the preset EXACTLY (no mutual-exclusion side effects).
    assert dlg._cfg == MASTERING_PRESETS[name]
    # Every checkbox reflects the preset's enabled flag.
    for stage, params in MASTERING_PRESETS[name].items():
        cb = dlg._stage_checkboxes[stage]
        assert cb.isChecked() is bool(params.get("enabled", False)), stage
