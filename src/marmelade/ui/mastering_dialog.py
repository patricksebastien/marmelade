"""Modal per-keeper MasteringDialog (Phase 7 Plan 07-02 Task 3 — D-05).

Per-keeper editor surface for the mastering chain. Bound to a single
keeper's ``mastering`` dict (NOT QSettings — that's the dock's job).
The dialog composes per-stage rows of ``[checkbox] [stage name]
[gear button]`` plus a bottom button row with "Reset to session chain"
(left), Apply + Discard changes (right).

D-05 — modal: the user must Apply or Discard before returning to the
waveform. Qt's modal-dialog semantics suppress global ApplicationShortcuts
so spacebar / A / B keypresses while the dialog is open are inactive.

D-04 — snapshot-not-link: the dialog NEVER re-snapshots the session
chain on open. Plan 07-03 owns the snapshot-at-keeper-creation hook
(every new keeper receives ``Region.mastering = load_session_chain_snapshot()``
at creation) plus the legacy-migration path for keepers persisted
before Phase 7 lands. The dialog therefore receives a non-None dict
every time; the constructor asserts this loudly. Re-snapshotting at
dialog-open time would produce a DIFFERENT snapshot than the keeper
originally received at creation (the session chain may have been
edited in the meantime), defeating D-04.

Reset-to-session-chain (the explicit user gesture) is the ONLY path
that re-syncs a keeper to the current session — it lives on the
dialog's left auxiliary button and is invoked deliberately.

UI-SPEC contracts realized here:
    * §"MasteringDialog (modal per-keeper editor) — D-05 modal" (lines 290-307)
    * §"Per-stage gear button — accessible name + tooltip table" (lines 272-285)
    * §"MasteringDialog interactions" (lines 565-573)
    * Dim-1 ban on bare "Cancel" — the QDialogButtonBox.Cancel button's
      text is overridden to ``"Discard changes"``.
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt, QSignalBlocker, QSize, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from marmelade.audio.mastering import (
    MASTERING_PRESETS,
    CompressorStage,
    DelayStage,
    DistortionStage,
    EndingFxStage,
    EqStage,
    FadeStage,
    HighPassStage,
    ReverbStage,
    LimiterStage,
    LoudnessStage,
    LowPassStage,
    NormalizeStage,
    match_preset,
    preset_names,
)
from marmelade.audio.mastering.chain import (
    _FX_STAGES,
    _STAGE_ORDER,
    load_session_chain_snapshot,
)
from marmelade.paths import matchering_reference_dir
from marmelade.ui.icons import _gear_icon
from marmelade.ui.matchering_reference_picker import (
    build_reference_param,
    scan_reference_dir,
)
from marmelade.ui.params_dialog import ParamsDialog


# UI-SPEC §"Per-stage gear button — accessible name + tooltip table" —
# strings used for setAccessibleName + setToolTip + ParamsDialog title.
# Matchering is excluded (Plan 05 owns its bespoke picker UI).
_STAGE_GEAR_LABELS: dict[str, str] = {
    # quick-260626-o9y — output-time fade in/out duration (auto-rendered).
    "fade": "Fade in/out duration",
    "highpass": "Highpass filter parameters",
    "lowpass": "Lowpass filter parameters",
    "eq": "EQ (Low / Mid / High) parameters",
    "compressor": "Compressor parameters",
    # quick-260629 — whole-clip color stages (auto-rendered gear dialogs).
    "distortion": "Distortion (drive) parameters",
    "delay": "Delay (time / feedback / mix) parameters",
    "reverb": "Reverb (room / damping / wet / dry / width) parameters",
    "limiter": "Limiter parameters",
    "matchering": "Matchering reference parameters",
    # quick-260621-gfq — normalize is the FINAL chain stage (auto-rendered).
    "normalize": "Normalize parameters",
    # quick-260623-l7l — gear opens a single target-LUFS ParamsDialog @ -14.0.
    "loudness": "Loudness target (LUFS) parameters",
    # Phase 07.1-04 — gear opens the custom power-user ParamsDialog
    # (effect_type / tail_sec / wet / primary) for the per-keeper ending FX.
    "ending_fx": "Ending FX parameters",
    # quick-260625 — gear opens the VST3 picker + native editor flow.
    "vst3": "Choose VST3 plugin and open its editor",
}

# Per-stage display label shown in the stage row (next to the checkbox).
# quick-260622-upg — plain "Matchering" (dropped the parenthetical prose
# suffix) to match the session dock label.
_STAGE_DISPLAY_NAMES: dict[str, str] = {
    # quick-260626-o9y — output-time fade in/out row (top of the panel).
    "fade": "Fade in/out",
    "highpass": "High-pass filter",
    "lowpass": "Low-pass filter",
    "eq": "EQ",
    "compressor": "Compressor",
    # quick-260629 — whole-clip color stages.
    "distortion": "Distortion",
    "delay": "Delay",
    "reverb": "Reverb",
    "limiter": "Limiter",
    "matchering": "Matchering",
    "normalize": "Normalize",
    # quick-260623-l7l — absolute LUFS loudness target row.
    "loudness": "Loudness (LUFS)",
    # Phase 07.1-04 — per-keeper ending FX tail row.
    "ending_fx": "Ending FX",
    # quick-260625 — external VST3 plugin slot row.
    "vst3": "VST3 plugin",
}

# Stage class lookup for the per-stage ParamsDialog. Matchering uses a
# bespoke picker UI in Plan 05 — it has no MasteringStage subclass.
_STAGE_CLASS_BY_NAME = {
    # quick-260626-o9y — gear opens the single duration_sec ParamsDialog @ 2.0s.
    "fade": FadeStage,
    "highpass": HighPassStage,
    "lowpass": LowPassStage,
    "eq": EqStage,
    "compressor": CompressorStage,
    # quick-260629 — whole-clip color stages render via the same generic
    # auto-render gear path (float sliders from each stage's parameters()).
    "distortion": DistortionStage,
    "delay": DelayStage,
    "reverb": ReverbStage,
    "limiter": LimiterStage,
    # quick-260621-gfq — gear opens the single target-dB ParamsDialog @ 0.0.
    "normalize": NormalizeStage,
    # quick-260623-l7l — same auto-render path renders the loudness
    # target-LUFS dialog.
    "loudness": LoudnessStage,
    # Phase 07.1-04 — the gear opens the custom power-user ParamsDialog from
    # EndingFxStage.parameters() (effect_type choice + tail_sec/wet/primary
    # floats), reusing the auto-render path with ZERO new gear code.
    "ending_fx": EndingFxStage,
}


class MasteringDialog(QDialog):
    """Modal per-keeper mastering chain editor.

    Args:
        keeper_id: Region UUID — included in the ``config_changed``
            signal payload so MainWindow can route to the right
            sidecar field + KeeperRow widget.
        keeper_mastering: Per-keeper mastering chain config dict. MUST
            NOT be None — Plan 07-03 owns the snapshot-at-creation
            invariant. The dialog raises ``AssertionError`` loudly if
            this contract is violated (so the missing wiring is
            obvious rather than silently re-snapshotting).
        keeper_range: Display string ``"00:14:32 – 00:18:07"`` used in
            the window title.
        parent: Optional parent widget (typically MainWindow).

    Signals:
        config_changed(keeper_id: str, mastering: dict): emitted by
            ``_on_apply_clicked``. Discard changes does NOT emit
            (drops the in-dialog edits).
    """

    config_changed = Signal(str, dict)

    def __init__(
        self,
        keeper_id: str,
        keeper_mastering: dict | None,
        keeper_range: str,
        parent: QWidget | None = None,
    ) -> None:
        # Defensive contract — D-04. Plan 07-03 must have populated
        # ``Region.mastering`` BEFORE this dialog is constructed (either
        # at keeper-creation time or via the legacy-migration auto-snapshot
        # path on first open). Re-snapshotting here would break D-04 by
        # rebinding the keeper to the CURRENT session chain instead of
        # the snapshot the keeper originally captured.
        assert keeper_mastering is not None, (
            "MasteringDialog requires keeper.mastering != None — see Plan "
            "07-03 for snapshot semantics"
        )

        super().__init__(parent)
        self.setWindowTitle(f"Mastering — {keeper_range}")
        self.setModal(True)  # D-05
        self.resize(480, 520)  # UI-SPEC default size
        # quick-260625 — floor so shrinking the dialog can never clip the
        # pinned button row (content scrolls inside the QScrollArea instead).
        self.setMinimumSize(420, 320)

        self._keeper_id = keeper_id
        # Mutable working copy + initial snapshot for diff-on-Apply.
        self._cfg: dict = copy.deepcopy(keeper_mastering)
        self._initial_cfg: dict = copy.deepcopy(self._cfg)

        # Per-stage widget registry — used by Reset-to-session-chain to
        # re-render checkbox states after overwriting self._cfg.
        self._stage_checkboxes: dict[str, QCheckBox] = {}
        # quick-260626 close-to-commit hardening — per-stage gear button
        # registry, used to disable the VST3 gear while its editor QProcess is
        # alive (so the user cannot re-open / re-Apply mid-edit). The Apply
        # button is captured into self._apply_btn in _build_ui.
        self._stage_gears: dict[str, QPushButton] = {}
        self._apply_btn: QPushButton | None = None

        # quick-260624-h78 — re-entrancy guard. Set True while
        # _apply_preset overwrites self._cfg + flips checkboxes and while
        # _sync_preset_combo drives setCurrentIndex, so neither path
        # re-triggers the other (mirrors the dock's guard).
        self._applying_preset: bool = False

        self._build_ui()

    # ------------------------------------------------------ layout

    def _build_ui(self) -> None:
        # quick-260625 — the chain has 9 stage rows plus two combos and
        # captions, which overflowed the fixed dialog height and clipped the
        # bottom button row (Reset / Apply / Discard changes). Put the
        # scrollable content in a QScrollArea and keep the button row OUTSIDE
        # it, pinned at the bottom, so the buttons are ALWAYS visible no matter
        # the dialog/screen height; the stage list scrolls when space is tight.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)  # UI-SPEC §Spacing
        outer.setSpacing(8)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)

        # quick-260629 — FX sub-section pinned at the VERY TOP of the dialog
        # (above the "Per-keeper chain" header + preset droplist), mirroring the
        # session dock: distortion/delay/reverb lifted into a titled "FX" group
        # box. The remaining stages render as flat rows below the header, in
        # _STAGE_ORDER order. DISPLAY-only — _STAGE_ORDER (snapshot + anchors)
        # and the DSP apply order are unchanged. See chain._FX_STAGES.
        fx_group = QGroupBox("FX")
        fx_layout = QVBoxLayout(fx_group)
        fx_layout.setContentsMargins(6, 4, 6, 4)
        fx_layout.setSpacing(4)
        for stage in _FX_STAGES:
            fx_layout.addWidget(self._build_stage_row(stage))
        body.addWidget(fx_group)

        # Section header.
        header = QLabel("Per-keeper chain")
        header.setStyleSheet("font-size: 10pt; font-weight: 600;")
        body.addWidget(header)

        caption = QLabel(
            "Changes apply only to THIS keeper. Use Reset to session "
            "chain to re-sync with the session defaults."
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #9CA3AF; font-size: 10pt;")
        body.addWidget(caption)

        body.addSpacing(8)

        # quick-260624-h78 — genre preset selector at the top of the stage
        # area. Index 0 is the "Custom" sentinel (no preset matches); 1..10
        # are the genre presets in their locked lineup order. Mirrors the
        # main MasteringDock combobox but scoped to THIS keeper's chain.
        self._preset_combo = QComboBox(self)
        self._preset_combo.addItem("Custom")
        self._preset_combo.addItems(preset_names())
        self._preset_combo.setToolTip(
            "Apply a genre mastering preset to this keeper's chain"
        )
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        body.addWidget(self._preset_combo)

        body.addSpacing(8)

        # quick-260629 — the dedicated per-keeper "Ending FX" preset dropdown
        # was REMOVED so the keeper dialog matches the session dock exactly:
        # ending FX is now just a checkbox + gear row inside the FX group at the
        # top of the body (effect_type + tail/onset/wet/primary are edited via
        # the gear ParamsDialog, identical to the dock). The curated ending-FX
        # presets remain available at the session level; per-keeper editing is
        # the single checkbox+param surface.
        #
        # Per-stage rows — the FX sub-section (incl. ending_fx) was rendered at
        # the very top of the body above; here we render only the NON-FX stages
        # in _STAGE_ORDER.
        for stage in _STAGE_ORDER:
            if stage in _FX_STAGES:
                continue
            row = self._build_stage_row(stage)
            body.addWidget(row)

        body.addStretch(1)

        # Mount the scrollable content; the button row below stays pinned.
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Bottom button row — Reset on left, Apply / Discard changes on right.
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        # Dim-1 — relabel the Cancel-role button to make the destructive
        # consequence explicit ("Discard changes" vs bare "Cancel").
        cancel_btn = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setText("Discard changes")
        # Apply button wiring.
        apply_btn = button_box.button(QDialogButtonBox.StandardButton.Apply)
        apply_btn.clicked.connect(self._on_apply_clicked)
        cancel_btn.clicked.connect(self.reject)
        # quick-260626 close-to-commit hardening — keep a handle so the VST3
        # flow can disable Apply while the out-of-process editor is open.
        self._apply_btn = apply_btn

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 8, 0, 0)
        self._reset_btn = QPushButton("Reset to session chain")
        self._reset_btn.clicked.connect(self._on_reset_to_session_chain)
        bottom.addWidget(self._reset_btn, 0)
        bottom.addStretch(1)
        bottom.addWidget(button_box, 0)
        outer.addLayout(bottom)

        # quick-260624-h78 — reflect the keeper's in-dialog cfg into the
        # combobox AFTER the checkboxes exist. Guarded so the programmatic
        # setCurrentIndex does not re-enter the apply path.
        self._sync_preset_combo()

    def _build_stage_row(self, stage: str) -> QWidget:
        """One row: ``[checkbox] [stage display name] [gear button]``."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        checkbox = QCheckBox()
        checkbox.setChecked(bool(self._cfg.get(stage, {}).get("enabled", False)))

        def _on_toggle(checked: bool, n: str = stage) -> None:
            # quick-260624-h78 — a flip driven BY _apply_preset is suppressed
            # at the source via QSignalBlocker, so reaching here with the guard
            # set would be a programmatic flip we must NOT treat as a user edit.
            # (Belt-and-suspenders against a future un-blocked setChecked.)
            if self._applying_preset:
                return
            # Mutate the working cfg in place — Apply persists this dict.
            self._cfg.setdefault(n, {})["enabled"] = bool(checked)
            # quick-260626-mih — Loudness (LUFS) and Normalize are mutually
            # exclusive: chain.py BYPASSES the Normalize tail whenever Loudness
            # is enabled, so "both checked" is a meaningless UI state. Checking
            # one auto-unchecks the other (cfg + checkbox). Only force the
            # sibling OFF on a CHECK (checked True) — an uncheck must NOT touch
            # the sibling. The sibling setChecked(False) is wrapped in
            # QSignalBlocker so it does not re-enter this handler (no recursion,
            # no double resync). The preset apply path flips checkboxes under
            # its own QSignalBlocker, so it bypasses this handler entirely —
            # preset integrity is preserved with no change there.
            if checked and n in ("loudness", "normalize"):
                sibling = "normalize" if n == "loudness" else "loudness"
                self._cfg.setdefault(sibling, {})["enabled"] = False
                cb = self._stage_checkboxes.get(sibling)
                if cb is not None:
                    with QSignalBlocker(cb):
                        cb.setChecked(False)
            # Re-point the genre preset combo at the matching preset / "Custom".
            self._sync_preset_combo()

        checkbox.toggled.connect(_on_toggle)
        self._stage_checkboxes[stage] = checkbox
        h.addWidget(checkbox, 0)

        name_label = QLabel(_STAGE_DISPLAY_NAMES[stage])
        h.addWidget(name_label, 1)

        gear = QPushButton()
        gear.setFixedSize(24, 24)
        gear.setIconSize(QSize(20, 20))
        gear.setIcon(_gear_icon())
        gear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        accessible = _STAGE_GEAR_LABELS[stage]
        gear.setAccessibleName(accessible)
        gear.setToolTip(accessible)
        # Default-arg closure binding (Phase 1 LEARNINGS) — capture
        # ``stage`` by value to avoid late-binding in the for-loop.
        gear.clicked.connect(
            lambda _checked=False, n=stage: self._on_stage_gear_clicked(n)
        )
        # quick-260626 — register the gear so the VST3 flow can disable it
        # (and Apply) while its out-of-process editor QProcess is alive.
        self._stage_gears[stage] = gear
        h.addWidget(gear, 0)
        return row

    # ------------------------------------------------------ slots

    def _on_stage_gear_clicked(self, stage: str) -> None:
        """Open a stage-scoped ParamsDialog for the per-stage parameters.

        Matchering is special (D-03): no MasteringStage subclass; instead
        the picker UI lives INSIDE a ParamsDialog populated dynamically
        from :func:`scan_reference_dir`. Plan 07-05 wires this here so
        the gear button surfaces the reference-library combobox + Browse
        button + inline empty-state guidance.

        For non-matchering stages: instantiate the stage class, read its
        ``parameters()`` dict, pre-populate values from the in-dialog
        cfg, open the modal dialog. On Apply, write the new values back
        into the in-dialog cfg.
        """
        if stage == "matchering":
            return self._on_matchering_gear_clicked()
        if stage == "vst3":
            return self._on_vst3_gear_clicked()

        stage_cls = _STAGE_CLASS_BY_NAME.get(stage)
        if stage_cls is None:
            # Defensive: no class + not matchering. Unknown stage.
            return
        stage_instance = stage_cls()
        params = stage_instance.parameters()
        if not params:
            return
        # Current values — read from the working cfg first, falling back to
        # Param.default when a key is absent.
        current_stage_cfg = self._cfg.get(stage, {})
        current_values: dict = {}
        for name, p in params.items():
            current_values[name] = current_stage_cfg.get(name, p.default)

        title = _STAGE_GEAR_LABELS[stage]
        dlg = ParamsDialog(
            title=title,
            params=params,
            current_values=current_values,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_values = dlg.accepted_values()
            self._cfg.setdefault(stage, {}).update(new_values)
            # quick-260624-h78 — a param edit may move the cfg onto / off a
            # genre preset; resync the genre combobox. (quick-260629 — ending_fx
            # no longer has its own dropdown, so it resyncs the genre combo like
            # every other stage.)
            self._sync_preset_combo()

    def _on_vst3_gear_clicked(self) -> None:
        """Open the VST3 picker + out-of-process native editor flow.

        Mutates the in-dialog ``vst3`` cfg (plugin_path / plugin_name /
        state_b64 / enabled) via the shared :func:`configure_vst3` helper. The
        editor is hosted out-of-process (quick-260626-fw2) so the flow is now
        ASYNC: ``configure_vst3`` returns immediately and the row-checkbox sync
        happens in the ``on_done`` callback once the editor's worker exits. The
        genre preset combo is deliberately NOT touched — ``vst3`` is a
        per-keeper-only stage that never participates in a genre preset (and
        config_hash drops a disabled vst3 entirely, so preset matching is
        unaffected).

        Close-to-commit hardening (quick-260626): the cfg is mutated
        (``enabled`` / ``state_b64`` captured) only in the QProcess.finished
        handler, AFTER the user closes the editor. If the user could click
        Apply (or re-open the gear) while the editor is still open, they would
        commit a stale cfg (``enabled`` still False → silent passthrough at
        master time), which is the prime timing-race failure. We therefore
        DISABLE Apply + the VST3 gear the moment the editor process starts
        (``on_started``) and RE-ENABLE them in ``_vst3_done`` (the
        finished/on_done hook), after the cfg has been captured. This makes the
        race structurally impossible: Apply cannot fire until the editor closed
        and state was captured.
        """
        from marmelade.ui.vst3_config import configure_vst3

        cfg = self._cfg.setdefault("vst3", {})

        def _vst3_started() -> None:
            self._set_vst3_editing(True)

        def _vst3_done(changed: bool) -> None:
            # Re-enable Apply + the gear regardless of success/failure — the
            # editor process has exited, so the lock no longer protects
            # anything (and leaving Apply disabled would strand the dialog).
            self._set_vst3_editing(False)
            if not changed:
                return
            # Tick the VST3 row checkbox to the captured `enabled` state so the
            # user gets a clear visual signal that state was captured.
            checkbox = self._stage_checkboxes.get("vst3")
            if checkbox is not None:
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(bool(cfg.get("enabled", False)))

        configure_vst3(
            self, cfg, on_done=_vst3_done, on_started=_vst3_started
        )

    def _set_vst3_editing(self, editing: bool) -> None:
        """Lock/unlock the Apply + VST3 gear while the editor QProcess is alive.

        quick-260626 close-to-commit hardening. While ``editing`` is True the
        Apply button and the VST3 gear are disabled so the user cannot commit a
        stale cfg (``enabled`` not yet captured) or re-launch a second editor
        mid-edit. Both are re-enabled when the editor process finishes.
        """
        if self._apply_btn is not None:
            self._apply_btn.setEnabled(not editing)
        gear = self._stage_gears.get("vst3")
        if gear is not None:
            gear.setEnabled(not editing)

    def _on_matchering_gear_clicked(self) -> None:
        """Open the Matchering reference picker (Plan 07-05).

        Picker UX strategy A: the picker is a ParamsDialog with a
        choice-kind ``reference_path`` Param whose ``browse_filter`` is
        set. ParamsDialog renders the combobox + Browse button + (when
        the library dir is empty) the inline empty-state guidance
        label.

        On Apply, the selected value is resolved:

        * If the value matches a filename in the scan
          (:func:`scan_reference_dir`), the in-cfg ``reference_path``
          becomes ``matchering_reference_dir() / filename`` (absolute
          path inside the library — ``is_one_off = False``).
        * Otherwise (Browse-picked absolute path), the value is the
          absolute path verbatim and ``is_one_off = True`` (transient
          T-7-01 carve-out per :func:`_validate_reference_path`).
        """
        current_stage_cfg = self._cfg.setdefault("matchering", {})
        current_value = str(current_stage_cfg.get("reference_path", ""))
        # Build a Param dynamically with the current library + custom
        # entry preserved.
        param = build_reference_param(current_value)
        params = {"reference_path": param}
        current_values = {"reference_path": param.default}

        dlg = ParamsDialog(
            title=_STAGE_GEAR_LABELS["matchering"],
            params=params,
            current_values=current_values,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_values = dlg.accepted_values()
        selected = str(new_values.get("reference_path", ""))

        # Resolve filename vs. absolute path. The scan returns
        # ``(filename, abs_path)`` tuples; if selected matches a name
        # there, we have a library entry. Otherwise selected is the
        # absolute path the Browse button appended to the combobox.
        scan = {name: abs_path for name, abs_path in scan_reference_dir()}
        if selected in scan:
            current_stage_cfg["reference_path"] = str(scan[selected])
            current_stage_cfg["is_one_off"] = False
        elif selected:
            current_stage_cfg["reference_path"] = selected
            current_stage_cfg["is_one_off"] = True
        else:
            # User explicitly chose empty — clear the reference.
            current_stage_cfg["reference_path"] = ""
            current_stage_cfg["is_one_off"] = False

        # quick-260624-h78 — a matchering reference edit may move the cfg onto /
        # off a preset; resync the combobox.
        self._sync_preset_combo()

    def _on_reset_to_session_chain(self) -> None:
        """Overwrite the in-dialog cfg with a fresh session-chain snapshot.

        UI-SPEC §"Bottom button row" — also re-renders the per-stage
        checkbox visual states from the new cfg so the user sees the
        change immediately. The on-disk keeper state is NOT touched —
        the user still needs to click Apply to commit.
        """
        self._cfg = load_session_chain_snapshot()
        for stage, checkbox in self._stage_checkboxes.items():
            # quick-260624-h78 — block each checkbox's toggled signal so the
            # re-render does not cascade into _on_toggle (no per-checkbox cfg
            # write or resync); a single _sync_preset_combo runs afterwards.
            with QSignalBlocker(checkbox):
                checkbox.setChecked(
                    bool(self._cfg.get(stage, {}).get("enabled", False))
                )
        # quick-260624-h78 — point the combobox at the matching preset / Custom.
        self._sync_preset_combo()

    def _on_apply_clicked(self) -> None:
        """Emit ``config_changed(keeper_id, cfg)`` and accept the dialog."""
        self.config_changed.emit(self._keeper_id, self._cfg)
        self.accept()

    # ------------------------------------------------- preset combobox

    def _on_preset_selected(self, index: int) -> None:
        """Apply the selected genre preset to THIS keeper's working cfg.

        Index 0 ("Custom") is a no-op; index >0 applies the corresponding
        preset. Returns immediately when ``_applying_preset`` is set, so the
        programmatic ``setCurrentIndex`` in :meth:`_sync_preset_combo` does
        not re-enter the apply path (signal-loop guard). Mirrors the dock.
        """
        if self._applying_preset:
            return
        if index <= 0:
            return
        self._apply_preset(self._preset_combo.itemText(index))

    def _apply_preset(self, name: str) -> None:
        """Overwrite the in-dialog working cfg with preset ``name``.

        Per-keeper analog of the dock's ``_apply_preset`` AND the existing
        :meth:`_on_reset_to_session_chain`: deep-copies the preset's full
        8-stage chain into ``self._cfg`` and flips each stage checkbox to the
        preset's enabled state (signals blocked so no per-checkbox cfg write /
        resync fires). Does NOT emit ``config_changed`` and does NOT touch
        QSettings — persistence happens only when the user clicks Apply.
        """
        self._applying_preset = True
        try:
            self._cfg = copy.deepcopy(MASTERING_PRESETS[name])
            for stage, checkbox in self._stage_checkboxes.items():
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(
                        bool(self._cfg.get(stage, {}).get("enabled", False))
                    )
        finally:
            self._applying_preset = False

    def _sync_preset_combo(self) -> None:
        """Point the preset combobox at the preset matching the working cfg.

        Computes :func:`match_preset` over ``self._cfg`` and selects that
        preset's combobox item, or index 0 ("Custom") when no preset matches.
        The ``setCurrentIndex`` runs under the ``_applying_preset`` guard so it
        never triggers :meth:`_on_preset_selected`'s apply path. The prior
        guard value is restored (mirror the dock's ``prev``-restore form) so
        nested calls (e.g. from inside _apply_preset's flow) behave.
        """
        name = match_preset(self._cfg)
        if name is None:
            target = 0
        else:
            target = self._preset_combo.findText(name)
            if target < 0:
                target = 0
        prev = self._applying_preset
        self._applying_preset = True
        try:
            self._preset_combo.setCurrentIndex(target)
        finally:
            self._applying_preset = prev


    # ------------------------------------------------------ public read-only

    @property
    def applied_config(self) -> dict:
        """Return the dialog's current config dict — useful post-accept."""
        return self._cfg


__all__ = ["MasteringDialog"]
