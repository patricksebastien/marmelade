"""Always-visible session-chain editor (Phase 7 Plan 07-03 Task 1 — D-06).

The MasteringDock is the source of truth for the session chain. Every
checkbox toggle + every per-stage gear-button Apply persists instantly
to ``QSettings("Marmelade","Marmelade")`` under the key prefix
``mastering/session/<stage>/<param>`` (UI-SPEC §"No 'Apply' button on
the session dock body" — instant persistence, no buffering).

Tabified with the existing Keepers QDockWidget (UI-SPEC §"Layout
Architecture — Phase 7 deltas"). Hidden by default (D-06); shown via
``View → Mastering panel`` toggle (state persisted under
``mastering/panel/visible``).

D-04 — snapshot-not-link semantics: edits here apply only to NEW
keepers (Plan 07-03 owns the snapshot-at-keeper-creation hook). Existing
keepers' ``Region.mastering`` dicts are independent snapshots and are
NEVER mutated by session-chain edits. This module emits the
``session_chain_changed`` signal on every edit so MainWindow can refresh
the divergence badge state for each existing keeper (the badge state
depends on ``config_hash(keeper.mastering) == config_hash(session)``).

T-7-03 mitigation — QSettings key namespace discipline:
    QSettings key segments come EXCLUSIVELY from class constants:
        * stage names from :data:`_STAGE_ORDER` (literal tuple in
          ``audio/mastering/chain.py``).
        * param names from ``stage.parameters().keys()`` (declared in
          each stage class's source).
    No code path constructs a QSettings key from user input. The
    ``test_t_7_03_no_user_controlled_qsettings_keys`` test ast-parses
    this module to enforce the discipline. The same test asserts no
    bare ``QSettings()`` call (D-16 / T-06-01 carry-over).

Defense-in-depth: even if a future contributor introduced a
user-controlled key segment, the on-disk QSettings format is
plaintext INI on Linux / plist on macOS / registry on Windows — the
user's own keystrokes already have full read/write to that store. The
"injection" attack class is not relevant to a single-user desktop app;
the mitigation is still applied as a code-hygiene discipline.

Stage class lookup:
    Matchering has no :class:`MasteringStage` subclass (D-03 — operates
    on whole-clip + reference, not per-sample DSP). For the dock's gear
    button, opening the Matchering ParamsDialog lands in Plan 05 (the
    bespoke reference-picker UX). In Plan 03 the gear button is wired
    but the slot returns early for ``matchering`` — Plan 05 will lift
    the gate.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSettings, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
    LimiterStage,
    LoudnessStage,
    LowPassStage,
    NormalizeStage,
    ReverbStage,
    match_preset,
    preset_names,
)
from marmelade.audio.mastering.chain import (
    _FX_STAGES,
    _SESSION_DEFAULTS,
    _STAGE_ORDER,
    _coerce_like,
    load_session_chain_snapshot,
)
from marmelade.ui.icons import _master_icon_with_badge
from marmelade.ui.matchering_reference_picker import (
    build_reference_param,
    scan_reference_dir,
)
from marmelade.ui.params_dialog import ParamsDialog


# UI-SPEC §"Mastering dock — labels and copy" — per-stage row labels.
# quick-260622-upg — the Matchering row label is plain "Matchering"; the
# old parenthetical prose suffix was dropped to shrink the panel.
_STAGE_DISPLAY_NAMES: dict[str, str] = {
    # quick-260626-o9y — output-time fade in/out row (top of the dock).
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
    # quick-260621-gfq — normalize is the FINAL chain stage (auto-rendered
    # as the last dock row).
    "normalize": "Normalize",
    # quick-260623-l7l — absolute LUFS loudness target row.
    "loudness": "Loudness (LUFS)",
    # quick-260626-ked — external VST3 plugin slot, now surfaced session-wide.
    "vst3": "VST3 plugin",
    # quick-260626-o9y — Ending FX is now a normal session-dock stage row
    # (was per-keeper-only). Plain row only — NO bespoke preset combo here.
    "ending_fx": "Ending FX",
}

# UI-SPEC §"Per-stage gear button — accessible name + tooltip table".
# Used for setAccessibleName + setToolTip AND as the ParamsDialog title
# so screen-reader / hover / opened-dialog-title stay in sync.
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
    # quick-260621-gfq — gear opens a single target-dB ParamsDialog @ 0.0.
    "normalize": "Normalize parameters",
    # quick-260623-l7l — gear opens a single target-LUFS ParamsDialog @ -14.0.
    "loudness": "Loudness target (LUFS) parameters",
    # quick-260626-ked — gear picks a .vst3 + opens its native editor.
    "vst3": "Choose VST3 plugin and open its editor",
    # quick-260626-o9y — gear opens the ending-FX power-user ParamsDialog.
    "ending_fx": "Ending FX parameters",
}

# Stage class lookup for the per-stage ParamsDialog. Matchering has no
# MasteringStage subclass (Plan 05 owns the bespoke picker UI).
_STAGE_CLASS_BY_NAME: dict[str, type] = {
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
    # quick-260621-gfq — the existing _on_stage_gear_clicked path renders
    # the normalize target-dB dialog with no special-casing.
    "normalize": NormalizeStage,
    # quick-260623-l7l — same auto-render path renders the loudness
    # target-LUFS dialog.
    "loudness": LoudnessStage,
    # quick-260626-o9y — Ending FX renders via the same generic auto-render
    # gear path (effect_type choice + tail_sec/onset_sec/wet/primary floats).
    "ending_fx": EndingFxStage,
}


class MasteringDock(QWidget):
    """Always-visible session-chain editor widget.

    Composes 7 per-stage rows (UI-SPEC §"Mastering dock — labels and
    copy"). Each row exposes a checkbox + label + gear button. Toggling
    the checkbox writes ``mastering/session/<stage>/enabled`` instantly;
    clicking the gear opens a :class:`ParamsDialog` for the stage's
    full param surface and writes each accepted value under
    ``mastering/session/<stage>/<param>``.

    Signals:
        session_chain_changed(): emitted after EVERY edit (checkbox
            toggle OR per-stage Apply). No payload — receivers re-read
            the snapshot themselves via :func:`load_session_chain_snapshot`.

    Args:
        parent: Optional parent widget (typically the surrounding
            :class:`QDockWidget` constructed by MainWindow).
    """

    session_chain_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Per-stage widget registry — tests and MainWindow drive
        # ``stage_checkbox(name)`` for direct lookup; the production
        # signal flow does not reach into this dict.
        self._stage_checkboxes: dict[str, QCheckBox] = {}

        # quick-260626-ked — gear-button registry (QPushButton, the dock's
        # gear type). Needed by _on_vst3_gear_clicked to lock/unlock the vst3
        # gear while the out-of-process editor is open. Symmetric with
        # _stage_checkboxes; populated in _build_stage_row.
        self._stage_gears: dict[str, QPushButton] = {}

        # quick-260623-p5b — re-entrancy guard for the preset combobox.
        # Set True while _apply_preset writes QSettings + flips checkboxes
        # and while _sync_preset_combo drives setCurrentIndex, so neither
        # the combobox's currentIndexChanged nor a checkbox's toggled
        # re-enters the apply path (the signal-loop trap).
        self._applying_preset: bool = False

        # Read the current session snapshot ONCE so the initial
        # checkbox states reflect what's on disk (and the Phase 6
        # ``_clear_qsettings_mastering`` fixture's empty-state honors
        # ``_SESSION_DEFAULTS`` — Limiter ON, others OFF).
        snapshot = load_session_chain_snapshot()

        self._build_ui(snapshot)

    # ------------------------------------------------------ layout

    def _build_ui(self, snapshot: dict[str, dict[str, Any]]) -> None:
        outer = QVBoxLayout(self)
        # quick-260623-csc — tightened the md token 16 -> 8px to reclaim
        # ~16px horizontal on each side so the panel honors the new 130px
        # narrow-left-sidebar minimum without clipping stage rows.
        # quick-260623-d84 — RIGHT margin zeroed so dock content butts tight
        # against the resize handle / splitter (the user-reported right dead
        # space); 6px L/T/B keeps stage rows from clipping at the 130px
        # narrow-dock minimum.
        outer.setContentsMargins(6, 6, 0, 6)
        outer.setSpacing(6)

        # quick-260629 — FX sub-section pinned at the VERY TOP of the dock
        # (above the "Mastering chain" header + preset droplist): the whole-clip
        # color stages (distortion/delay/reverb) are lifted into a titled "FX"
        # group box. The remaining stages render as flat rows below the header,
        # in _STAGE_ORDER order. DISPLAY-only — _STAGE_ORDER (snapshot + anchors)
        # and the DSP apply order are unchanged. See chain._FX_STAGES.
        fx_group = QGroupBox("FX", self)
        fx_layout = QVBoxLayout(fx_group)
        fx_layout.setContentsMargins(6, 4, 6, 4)
        fx_layout.setSpacing(4)
        for stage in _FX_STAGES:
            fx_layout.addWidget(self._build_stage_row(stage, snapshot))
        outer.addWidget(fx_group)

        # Section header (UI-SPEC §"Dock body"). The secondary caption was
        # dropped to keep the dock compact and maximise waveform width.
        header = QLabel("Mastering chain")
        header.setStyleSheet("font-size: 10pt; font-weight: 600;")
        outer.addWidget(header)

        # quick-260623-p5b — genre preset selector below the header.
        # Index 0 is the "Custom" sentinel (no preset matches); 1..10 are the
        # genre presets in their locked lineup order. Kept compact (the user
        # wants the dock narrow) — a single bare QComboBox, no extra label.
        self._preset_combo = QComboBox(self)
        self._preset_combo.addItem("Custom")
        self._preset_combo.addItems(preset_names())
        self._preset_combo.setToolTip(
            "Apply a genre mastering preset to the session chain"
        )
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        outer.addWidget(self._preset_combo)

        outer.addSpacing(8)

        # quick-260626-o9y — every _STAGE_ORDER stage now renders a session
        # dock row (fade + ending_fx included). Ending FX was previously
        # skipped here as per-keeper-only; it is now a normal session-dock
        # stage row (plain checkbox + gear → ParamsDialog, NO bespoke preset
        # combo). Fade renders at the top as the output-time fade row.
        for stage in _STAGE_ORDER:
            if stage in _FX_STAGES:
                continue
            row = self._build_stage_row(stage, snapshot)
            outer.addWidget(row)

        outer.addStretch(1)

        # Reflect the on-disk session chain into the combobox AFTER the
        # checkboxes exist. Guarded so the programmatic setCurrentIndex does
        # not trigger an apply.
        self._sync_preset_combo()

    def _build_stage_row(
        self, stage: str, snapshot: dict[str, dict[str, Any]]
    ) -> QWidget:
        """One row: ``[checkbox] [stage display name] [gear button]``.

        T-7-03 — ``stage`` comes from the loop over ``_STAGE_ORDER`` (a
        literal tuple in ``chain.py``). No user input flows into the
        widget layout or the connected slot's QSettings key.
        """
        row = QWidget(self)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        # quick-260623-d84 — tightened 8 -> 4px to pull the checkbox and gear
        # snug to the stretch-1 label (still above the ~2px readability floor).
        h.setSpacing(4)

        # Checkbox — reflects ``mastering/session/<stage>/enabled``.
        checkbox = QCheckBox(row)
        checkbox.setChecked(bool(snapshot.get(stage, {}).get("enabled", False)))
        # Default-arg closure binding (Phase 1 LEARNINGS — late-binding
        # would capture the loop's terminal value `"matchering"` for
        # every connected slot).
        checkbox.toggled.connect(
            lambda checked, name=stage: self._on_stage_enabled_changed(
                name, checked
            )
        )
        self._stage_checkboxes[stage] = checkbox
        h.addWidget(checkbox, 0)

        # Display label (Label 10pt semibold per UI-SPEC §Typography).
        name_label = QLabel(_STAGE_DISPLAY_NAMES[stage], row)
        name_label.setStyleSheet("font-size: 10pt; font-weight: 600;")
        h.addWidget(name_label, 1)

        # Gear button — 24x24, icon-only, opens stage-scoped ParamsDialog.
        gear = QPushButton(row)
        # quick-260623-d84 — use the painted-gear composite so the dock gear
        # is pixel-identical to the Keeper row's Master gear (both from
        # _paint_gear_icon); _gear_icon() prefers an OS theme glyph that
        # renders differently per desktop.
        gear.setIcon(_master_icon_with_badge("none"))
        gear.setFixedSize(24, 24)
        gear.setIconSize(QSize(20, 20))
        gear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        accessible = _STAGE_GEAR_LABELS[stage]
        gear.setAccessibleName(accessible)
        gear.setToolTip(accessible)
        gear.clicked.connect(
            lambda _checked=False, name=stage: self._on_stage_gear_clicked(name)
        )
        # quick-260626-ked — register the gear so _on_vst3_gear_clicked can
        # lock/unlock it while the out-of-process VST3 editor is open.
        self._stage_gears[stage] = gear
        h.addWidget(gear, 0)

        return row

    # ------------------------------------------------------ public accessors

    def stage_checkbox(self, stage: str) -> QCheckBox:
        """Return the QCheckBox widget for ``stage`` — used by tests + MainWindow.

        Raises:
            KeyError: if ``stage`` is not a known stage name (programming
                error — only ``_STAGE_ORDER`` entries are valid).
        """
        return self._stage_checkboxes[stage]

    # ------------------------------------------------------ slots

    def _on_stage_enabled_changed(self, stage: str, checked: bool) -> None:
        """Persist the new enabled-state and emit ``session_chain_changed``.

        T-7-03 — the key segment ``stage`` is from ``_STAGE_ORDER``
        (asserted via the test's ast-grep). No user input flows in.
        """
        # quick-260623-p5b — a checkbox flip driven BY _apply_preset is
        # suppressed at the source via QSignalBlocker, so reaching here means
        # a genuine user toggle. (The guard is belt-and-suspenders against a
        # future code path that mutates a checkbox without blocking.)
        if self._applying_preset:
            return
        s = QSettings("Marmelade", "Marmelade")
        s.setValue(f"mastering/session/{stage}/enabled", bool(checked))
        # quick-260626-mih — Loudness (LUFS) and Normalize are mutually
        # exclusive: chain.py BYPASSES the Normalize tail whenever Loudness is
        # enabled, so "both checked" is a meaningless UI state. Checking one
        # auto-unchecks the other (QSettings + checkbox). Only force the
        # sibling OFF on a CHECK (checked True) — an uncheck must NOT touch the
        # sibling. The sibling write goes BEFORE the single s.sync() so it is
        # flushed in the same call; the sibling setChecked(False) is wrapped in
        # QSignalBlocker so it does NOT re-enter this handler and cannot trigger
        # a second emit/sync (EXACTLY one s.sync() + one emit per user toggle).
        if checked and stage in ("loudness", "normalize"):
            sibling = "normalize" if stage == "loudness" else "loudness"
            s.setValue(f"mastering/session/{sibling}/enabled", False)
            cb = self._stage_checkboxes.get(sibling)
            if cb is not None:
                with QSignalBlocker(cb):
                    cb.setChecked(False)
        s.sync()
        # Re-point the preset combo at the matching preset / "Custom" BEFORE
        # the emit (the resync is guarded so it never re-enters apply).
        self._sync_preset_combo()
        self.session_chain_changed.emit()

    # ------------------------------------------------- preset combobox

    def _on_preset_selected(self, index: int) -> None:
        """Apply the selected genre preset to the session chain.

        Index 0 ("Custom") is a no-op; index >0 applies the corresponding
        preset. Returns immediately when ``_applying_preset`` is set, so the
        programmatic ``setCurrentIndex`` in :meth:`_sync_preset_combo` does
        not re-enter the apply path (signal-loop guard).
        """
        if self._applying_preset:
            return
        if index <= 0:
            return
        self._apply_preset(self._preset_combo.itemText(index))

    def _apply_preset(self, name: str) -> None:
        """Overwrite the session chain QSettings with preset ``name``.

        Writes every ``mastering/session/<stage>/<param>`` key, flips each
        stage checkbox to the preset's enabled state (with checkbox signals
        blocked so no per-checkbox QSettings write / emit / resync fires),
        then emits ``session_chain_changed`` EXACTLY ONCE after the guard is
        cleared.

        T-7-03 / T-p5b-02 — every key segment comes from the stage/param
        constants in the preset config dict; the preset display ``name`` is
        NEVER used as a key segment.
        """
        cfg = MASTERING_PRESETS[name]
        self._applying_preset = True
        try:
            s = QSettings("Marmelade", "Marmelade")
            for stage, params in cfg.items():
                for param, value in params.items():
                    s.setValue(f"mastering/session/{stage}/{param}", value)
            # Tidy bonus — drop any stale one-off matchering flag so a prior
            # Browse selection does not linger under the preset chain.
            s.remove("mastering/session/matchering/is_one_off")
            s.sync()
            # Flip checkboxes with their signals blocked so setChecked does
            # NOT cascade into _on_stage_enabled_changed (no per-checkbox
            # write, emit, or resync — defuses the signal loop).
            for stage, checkbox in self._stage_checkboxes.items():
                if stage not in cfg:
                    continue
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(bool(cfg[stage]["enabled"]))
        finally:
            self._applying_preset = False
        # Exactly one emit per apply, after the guard is cleared.
        self.session_chain_changed.emit()

    def _sync_preset_combo(self) -> None:
        """Point the preset combobox at the preset matching the session chain.

        Computes :func:`match_preset` over the current QSettings snapshot and
        selects that preset's combobox item, or index 0 ("Custom") when no
        preset matches. The ``setCurrentIndex`` runs under the
        ``_applying_preset`` guard so it never triggers
        :meth:`_on_preset_selected`'s apply path.
        """
        name = match_preset(load_session_chain_snapshot())
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

    def _on_stage_gear_clicked(self, stage: str) -> None:
        """Open a stage-scoped :class:`ParamsDialog` and persist on Apply.

        For non-Matchering stages: instantiate the stage class, read its
        ``parameters()`` dict, pre-populate values from QSettings (falling
        back to each Param's default), open the modal dialog. On Apply,
        write each value back under
        ``mastering/session/<stage>/<param>`` and emit
        ``session_chain_changed``.

        Matchering (Plan 07-05): the bespoke reference-picker is composed
        inside a ParamsDialog with a dynamically-populated combobox + the
        Browse button + (when the library dir is empty) the inline
        empty-state guidance label. Handled by
        :meth:`_on_matchering_gear_clicked`.

        T-7-03 — ``stage`` is from ``_STAGE_ORDER``; ``pname`` is iterated
        over ``stage_instance.parameters().keys()`` which are declared as
        literals in the stage class source. The grep test pins this
        invariant on the source.
        """
        if stage == "matchering":
            return self._on_matchering_gear_clicked()
        # quick-260626-ked — the external VST3 plugin slot has no
        # MasteringStage subclass; it opens an out-of-process native editor
        # via the shared configure_vst3 flow (mirrors the per-keeper dialog).
        if stage == "vst3":
            return self._on_vst3_gear_clicked()

        stage_cls = _STAGE_CLASS_BY_NAME.get(stage)
        if stage_cls is None:
            # Unknown stage — silent no-op.
            return
        stage_instance = stage_cls()
        params = stage_instance.parameters()
        if not params:
            return

        # Read current values from QSettings (or fall back to defaults).
        s = QSettings("Marmelade", "Marmelade")
        current_values: dict[str, Any] = {}
        for pname, p in params.items():
            default = _SESSION_DEFAULTS.get(stage, {}).get(pname, p.default)
            raw = s.value(f"mastering/session/{stage}/{pname}", default)
            current_values[pname] = _coerce_like(raw, p.default)

        dlg = ParamsDialog(
            title=_STAGE_GEAR_LABELS[stage],
            params=params,
            current_values=current_values,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_values = dlg.accepted_values()
        for pname, value in new_values.items():
            s.setValue(f"mastering/session/{stage}/{pname}", value)
        s.sync()
        # quick-260623-p5b — a per-stage param edit can move the chain into
        # or out of a preset match; resync the combobox (guarded).
        self._sync_preset_combo()
        self.session_chain_changed.emit()

    def _on_matchering_gear_clicked(self) -> None:
        """Open the Matchering reference picker for the SESSION chain.

        Same picker shape as the per-keeper MasteringDialog (Plan 07-05
        Task 3): a ParamsDialog populated dynamically from
        :func:`scan_reference_dir`. Persists the selection under
        ``mastering/session/matchering/{reference_path,is_one_off}``.
        T-7-03 — both literal QSettings key segments are static.
        """
        s = QSettings("Marmelade", "Marmelade")
        current_value = str(
            s.value(
                "mastering/session/matchering/reference_path",
                _SESSION_DEFAULTS["matchering"].get("reference_path", ""),
            )
            or ""
        )
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
        scan = {name: abs_path for name, abs_path in scan_reference_dir()}
        if selected in scan:
            s.setValue(
                "mastering/session/matchering/reference_path", str(scan[selected])
            )
            s.setValue("mastering/session/matchering/is_one_off", False)
        elif selected:
            s.setValue("mastering/session/matchering/reference_path", selected)
            s.setValue("mastering/session/matchering/is_one_off", True)
        else:
            s.setValue("mastering/session/matchering/reference_path", "")
            s.setValue("mastering/session/matchering/is_one_off", False)
        s.sync()
        # quick-260623-p5b — resync the preset combobox (guarded).
        self._sync_preset_combo()
        self.session_chain_changed.emit()

    def _on_vst3_gear_clicked(self) -> None:
        """Open the VST3 picker + out-of-process native editor for the SESSION.

        quick-260626-ked — surfaces the external VST3 plugin slot as a
        session-chain row, mirroring the per-keeper MasteringDialog's async
        flow. The native editor runs out-of-process (the editor's native call
        blocks its host thread until the window closes), so this is ASYNC:
        :func:`configure_vst3` returns immediately and the cfg is mutated +
        persisted later in the ``on_done`` callback once the worker exits.

        Close-to-commit hardening: while the editor process is open the vst3
        gear + vst3 checkbox + preset combo are DISABLED (``on_started``) so the
        user cannot mutate the row or apply a genre preset over a not-yet-
        captured cfg; all three re-enable in ``_done`` (success AND failure).

        On a successful capture the four ``mastering/session/vst3/*`` keys are
        written, the checkbox is ticked to the captured ``enabled`` under a
        QSignalBlocker (so it does NOT re-enter ``_on_stage_enabled_changed`` and
        double-write/double-emit), the preset combo is resynced (an ENABLED
        vst3 → ``config_hash`` no longer drops it → ``match_preset`` returns
        None → combo shows "Custom"), and ``session_chain_changed`` emits once.

        T-7-03 — every ``setValue`` key is the literal-prefixed f-string
        ``f"mastering/session/vst3/{k}"`` where ``k`` is a literal param name
        from ``_SESSION_DEFAULTS["vst3"]``; ``vst3`` is a literal segment and
        the plugin_path/state_b64 values are NEVER key segments. The QSettings
        org/app pair is always explicit.
        """
        # Local import — keep the QProcess/Qt-glue off the module top-level
        # (mirrors the per-keeper dialog).
        from marmelade.ui.vst3_config import configure_vst3

        # Build a mutable cfg from QSettings, falling back to defaults per key
        # and coercing types exactly as load_session_chain_snapshot does.
        s = QSettings("Marmelade", "Marmelade")
        defaults = _SESSION_DEFAULTS["vst3"]
        cfg: dict[str, Any] = {}
        for k, default in defaults.items():
            raw = s.value(f"mastering/session/vst3/{k}", default)
            cfg[k] = _coerce_like(raw, default)

        gear = self._stage_gears.get("vst3")
        checkbox = self._stage_checkboxes.get("vst3")

        def _started() -> None:
            # Lock the vst3 row + preset combo while the editor process is open.
            if gear is not None:
                gear.setEnabled(False)
            if checkbox is not None:
                checkbox.setEnabled(False)
            self._preset_combo.setEnabled(False)

        def _done(changed: bool) -> None:
            # Always re-enable — the editor process has exited, so the lock no
            # longer protects anything (leaving it disabled would strand the
            # dock). Re-enable on BOTH success and failure.
            if gear is not None:
                gear.setEnabled(True)
            if checkbox is not None:
                checkbox.setEnabled(True)
            self._preset_combo.setEnabled(True)
            if not changed:
                return
            # Persist the captured cfg. Fresh handle — the worker finished
            # later, so do not rely on a long-lived closure-captured one.
            sw = QSettings("Marmelade", "Marmelade")
            for k in defaults:
                sw.setValue(f"mastering/session/vst3/{k}", cfg[k])
            sw.sync()
            # Tick the checkbox to the captured enabled WITHOUT re-entering
            # _on_stage_enabled_changed (no double-write / double-emit).
            if checkbox is not None:
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(bool(cfg.get("enabled", False)))
            # An enabled vst3 → config_hash keeps it → no preset match → Custom.
            self._sync_preset_combo()
            self.session_chain_changed.emit()

        configure_vst3(self, cfg, on_done=_done, on_started=_started)


__all__ = ["MasteringDock"]
