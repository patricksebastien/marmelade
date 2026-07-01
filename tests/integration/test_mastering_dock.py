"""Phase 7 Plan 07-03 Task 1 — MasteringDock widget pins.

The MasteringDock is the always-visible session-chain editor (D-06).
Tabified with the Keepers dock, hidden by default. Edits flow instantly
to ``QSettings("Marmelade","Marmelade")`` under the key prefix
``mastering/session/<stage>/<param>`` (no Apply button — UI-SPEC §"No
'Apply' button on the session dock body").

This test module pins the WIDGET-LEVEL behaviour (Task 1). The
MainWindow-integration tests live in :mod:`test_session_chain_snapshot`
(Task 2 — dock construction + View menu + snapshot-at-creation).

D-16 + T-7-03 — every QSettings touch uses the explicit
``("Marmelade","Marmelade")`` org/app pair AND every key segment
after ``mastering/session/`` comes from a class constant (no user input
flows into the persistence key namespace).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QDialog, QPushButton

from marmelade.audio.mastering.chain import (
    _SESSION_DEFAULTS,
    _STAGE_ORDER,
)
from marmelade.ui.mastering_dock import MasteringDock


# UI-SPEC §"Mastering dock — labels and copy" — stage row labels.
_EXPECTED_LABELS: dict[str, str] = {
    "highpass": "High-pass filter",
    "lowpass": "Low-pass filter",
    "eq": "EQ",
    "compressor": "Compressor",
    # quick-260629 — whole-clip color stages (drive / echo / ambience).
    "distortion": "Distortion",
    "delay": "Delay",
    "reverb": "Reverb",
    "limiter": "Limiter",
    # quick-260622-upg — plain "Matchering" (dropped "(reference match)").
    "matchering": "Matchering",
    # quick-260621-gfq — normalize is the FINAL chain stage, auto-rendered
    # as the last dock row.
    "normalize": "Normalize",
    # quick-260623-l7l — absolute LUFS loudness target row.
    "loudness": "Loudness (LUFS)",
    # quick-260626-ked — external VST3 plugin slot, now surfaced session-wide.
    "vst3": "VST3 plugin",
    # quick-260626-o9y — output-time fade in/out row (top of the dock).
    "fade": "Fade in/out",
    # quick-260626-o9y — Ending FX is now a normal session-dock row.
    "ending_fx": "Ending FX",
}


def test_dock_has_seven_stage_rows(qtbot, qapp) -> None:
    """Dock exposes one row per stage in ``_STAGE_ORDER``.

    Each row is built from a checkbox + label + gear button (UI-SPEC
    §"Per-stage row layout"). Asserting on count + per-stage label
    presence pins the layout without coupling to private attribute names.
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    # quick-260626-o9y — the dock now renders EVERY _STAGE_ORDER stage (no
    # skip). Fade was added at index 0 and the prior ending_fx skip was
    # removed, so the session dock renders all 11 entries of _STAGE_ORDER
    # (fade, normalize, loudness, highpass, lowpass, eq, compressor, vst3,
    # limiter, ending_fx, matchering).
    session_stages = tuple(_STAGE_ORDER)
    checkboxes = dock.findChildren(QCheckBox)
    assert len(checkboxes) == len(session_stages) == len(_STAGE_ORDER)

    # The label text the dock surfaces — one per session stage. We accept ANY
    # QLabel descendant that contains the expected string so the test does not
    # couple to private layout details.
    from PySide6.QtWidgets import QLabel

    labels = [lbl.text() for lbl in dock.findChildren(QLabel)]
    for stage in session_stages:
        assert _EXPECTED_LABELS[stage] in labels, (
            f"missing stage label for {stage!r} (expected "
            f"{_EXPECTED_LABELS[stage]!r}, got {labels!r})"
        )


def test_default_limiter_checked_on_empty_qsettings(qtbot, qapp) -> None:
    """D-13 — Limiter is checked by default on first construction.

    The autouse ``_clear_qsettings_mastering`` fixture wipes the
    mastering sub-tree before this test runs, so the dock falls back to
    :data:`_SESSION_DEFAULTS` (which has ``limiter.enabled = True``).
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    assert dock.stage_checkbox("limiter").isChecked() is True
    # quick-260626-o9y — fade also defaults ON (reproduces today's forced
    # 2.0 s fade). Ending FX now has a session checkbox (the dock renders it),
    # defaulting OFF. So limiter AND fade are the two ON defaults; every other
    # stage (ending_fx included) defaults OFF.
    assert dock.stage_checkbox("fade").isChecked() is True
    for stage in _STAGE_ORDER:
        if stage in ("limiter", "fade"):
            continue
        assert dock.stage_checkbox(stage).isChecked() is False, (
            f"{stage!r} should default OFF (UI-SPEC default), got True"
        )


def test_checkbox_toggle_writes_qsettings_instantly(qtbot, qapp) -> None:
    """Toggling a stage checkbox persists to QSettings on the same call.

    UI-SPEC §"No 'Apply' button on the session dock body" — edits flow
    to ``mastering/session/<stage>/enabled`` immediately, no buffering.
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    # Compressor defaults OFF — flip it ON and verify the QSettings
    # value rounds-trips through the explicit org/app pair.
    dock.stage_checkbox("compressor").setChecked(True)

    s = QSettings("Marmelade", "Marmelade")
    raw = s.value("mastering/session/compressor/enabled")
    # QSettings serializes booleans as strings on some platforms — accept
    # either bool True or a truthy string.
    if isinstance(raw, str):
        assert raw.lower() in ("true", "1", "yes"), raw
    else:
        assert bool(raw) is True


def test_session_chain_changed_signal_fires_on_toggle(qtbot, qapp) -> None:
    """The ``session_chain_changed`` signal fires on every dock edit.

    MainWindow subscribes to this signal to refresh divergence badges on
    existing keepers when the session config_hash changes.
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    with qtbot.waitSignal(dock.session_chain_changed, timeout=1000):
        dock.stage_checkbox("compressor").setChecked(True)


def test_dock_reflects_existing_qsettings_value_on_construction(
    qtbot, qapp
) -> None:
    """On construction the dock reads QSettings and reflects it into checkboxes.

    Pre-seeds ``mastering/session/highpass/enabled=true`` before building
    the dock; the highpass row's checkbox must be checked.
    """
    s = QSettings("Marmelade", "Marmelade")
    s.setValue("mastering/session/highpass/enabled", True)
    s.sync()

    dock = MasteringDock()
    qtbot.add_widget(dock)

    assert dock.stage_checkbox("highpass").isChecked() is True


def test_gear_opens_paramsdialog_with_current_qsettings_value(
    qtbot, qapp, monkeypatch
) -> None:
    """Clicking a stage's gear opens a ParamsDialog reading current values.

    Pre-seeds the compressor threshold and verifies the dialog sees the
    pre-seeded value (not just the stage default).
    """
    s = QSettings("Marmelade", "Marmelade")
    s.setValue("mastering/session/compressor/threshold_db", -24.0)
    s.sync()

    dock = MasteringDock()
    qtbot.add_widget(dock)

    captured: list[dict] = []

    class _StubParamsDialog:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

        def exec(self):
            return QDialog.DialogCode.Rejected

        def accepted_values(self):
            return {}

    from marmelade.ui import mastering_dock as md_module

    monkeypatch.setattr(md_module, "ParamsDialog", _StubParamsDialog)

    # Trigger the compressor gear handler directly — production code
    # connects the QPushButton.clicked signal to the same slot.
    dock._on_stage_gear_clicked("compressor")

    assert captured, "ParamsDialog was not constructed"
    current_values = captured[0].get("current_values", {})
    assert current_values.get("threshold_db") == pytest.approx(-24.0)


def test_gear_apply_writes_qsettings_and_fires_signal(
    qtbot, qapp, monkeypatch
) -> None:
    """ParamsDialog Apply → values land in QSettings + signal fires.

    Stub the dialog so ``exec`` returns ``Accepted`` and ``accepted_values``
    returns a known mutation; assert QSettings round-trip.
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    new_values = {
        "threshold_db": -20.0,
        "ratio": 3.0,
        "attack_ms": 12.0,
        "release_ms": 250.0,
    }

    class _StubParamsDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def accepted_values(self):
            return new_values

    from marmelade.ui import mastering_dock as md_module

    monkeypatch.setattr(md_module, "ParamsDialog", _StubParamsDialog)

    with qtbot.waitSignal(dock.session_chain_changed, timeout=1000):
        dock._on_stage_gear_clicked("compressor")

    s = QSettings("Marmelade", "Marmelade")
    for pname, expected in new_values.items():
        raw = s.value(f"mastering/session/compressor/{pname}")
        # QSettings round-trips floats to strings on some platforms.
        assert float(raw) == pytest.approx(float(expected)), (
            f"compressor/{pname}: expected {expected}, got {raw!r}"
        )


# ----------------------------- T-7-03 mitigation (key namespace audit)


_MASTERING_DOCK_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "marmelade"
    / "ui"
    / "mastering_dock.py"
)


def test_t_7_03_no_bare_qsettings_call() -> None:
    """The dock NEVER calls bare ``QSettings()`` — D-16 / T-06-01 carry-over.

    Bare-form derives the org/app from ``QCoreApplication`` defaults
    which the test conftest monkeypatches — the explicit
    ``("Marmelade","Marmelade")`` pair bypasses that monkeypatch
    in production, so we forbid the bare form to keep tests faithful.
    """
    src = _MASTERING_DOCK_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "QSettings":
            if not node.args:
                offenders.append(f"line {node.lineno}: bare QSettings() call")
    assert not offenders, (
        f"mastering_dock.py must use QSettings('Marmelade','Marmelade') — "
        f"offending bare calls: {offenders}"
    )


def test_t_7_03_no_user_controlled_qsettings_keys() -> None:
    """Every ``setValue`` key literal starts with ``mastering/session/``.

    The dock's only QSettings write surface is the session-chain
    namespace. Static-scan the source: every ``.setValue(...)`` argument
    must be an f-string whose literal prefix is ``mastering/session/``
    (the variable segments are stage names from ``_STAGE_ORDER`` and
    param names from ``stage.parameters().keys()`` — both class
    constants, not user input).
    """
    src = _MASTERING_DOCK_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "setValue":
            continue
        if not node.args:
            continue
        key_node = node.args[0]
        # Acceptable shapes:
        #   - bare str literal starting with "mastering/session/"
        #   - f-string whose first part is a Str constant with that prefix
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            if not key_node.value.startswith("mastering/session/"):
                offenders.append(
                    f"line {key_node.lineno}: {key_node.value!r}"
                )
            continue
        if isinstance(key_node, ast.JoinedStr):
            first = key_node.values[0] if key_node.values else None
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.startswith("mastering/session/")
            ):
                continue
            offenders.append(
                f"line {key_node.lineno}: f-string does not start with "
                f"'mastering/session/'"
            )
            continue
        offenders.append(
            f"line {node.lineno}: setValue() key is not a literal or "
            f"prefix-pinned f-string (T-7-03 grep guard)"
        )
    assert not offenders, (
        f"mastering_dock.py setValue() calls violate T-7-03 namespace "
        f"discipline: {offenders}"
    )


# ----------------------------- quick-260623-p5b — genre preset combobox


from PySide6.QtWidgets import QComboBox  # noqa: E402

from marmelade.audio.mastering import (  # noqa: E402
    MASTERING_PRESETS,
    match_preset,
    preset_names,
)
from marmelade.audio.mastering.chain import (  # noqa: E402
    config_hash,
    load_session_chain_snapshot,
)


def test_preset_combo_first_item_is_custom_then_10_presets(qtbot, qapp) -> None:
    """The combobox lists 'Custom' first then the 10 presets in locked order."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    combo = dock.findChild(QComboBox)
    assert combo is not None
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["Custom"] + preset_names()
    assert combo.count() == 11


def test_preset_combo_initial_selection_is_custom_on_empty_qsettings(
    qtbot, qapp
) -> None:
    """Empty QSettings (limiter-only default) matches no preset → 'Custom'."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    assert dock._preset_combo.currentIndex() == 0
    assert dock._preset_combo.currentText() == "Custom"


def test_selecting_preset_writes_full_qsettings_and_updates_checkboxes(
    qtbot, qapp
) -> None:
    """Selecting a preset writes every stage/param + flips checkboxes to match."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    name = "Dubstep"
    cfg = MASTERING_PRESETS[name]
    idx = dock._preset_combo.findText(name)
    dock._preset_combo.setCurrentIndex(idx)

    s = QSettings("Marmelade", "Marmelade")
    # Every stage/param round-trips to QSettings.
    for stage, params in cfg.items():
        for param, value in params.items():
            raw = s.value(f"mastering/session/{stage}/{param}")
            assert raw is not None, f"{stage}/{param} not written"
            if isinstance(value, bool):
                if isinstance(raw, str):
                    got = raw.lower() in ("true", "1", "yes")
                else:
                    got = bool(raw)
                assert got is value, f"{stage}/{param}={raw!r}"
            elif isinstance(value, (int, float)):
                assert float(raw) == pytest.approx(float(value)), (
                    f"{stage}/{param}={raw!r}"
                )
            else:
                assert str(raw) == value, f"{stage}/{param}={raw!r}"

    # Checkboxes reflect the preset's enabled state.
    for stage, params in cfg.items():
        assert dock.stage_checkbox(stage).isChecked() is bool(params["enabled"]), (
            stage
        )

    # The snapshot now round-trips to the same preset.
    assert match_preset(load_session_chain_snapshot()) == name


def test_selecting_preset_emits_session_chain_changed_exactly_once(
    qtbot, qapp
) -> None:
    """A preset apply fires session_chain_changed EXACTLY once (no loop)."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    emits: list[int] = []
    dock.session_chain_changed.connect(lambda: emits.append(1))

    idx = dock._preset_combo.findText("House")
    with qtbot.waitSignal(dock.session_chain_changed, timeout=1000):
        dock._preset_combo.setCurrentIndex(idx)

    assert len(emits) == 1, f"expected exactly 1 emit, got {len(emits)}"


def test_selecting_custom_index_0_is_a_noop(qtbot, qapp) -> None:
    """Index 0 ('Custom') writes nothing and emits nothing."""
    # Seed a known preset first so the combo is NOT already at index 0.
    dock = MasteringDock()
    qtbot.add_widget(dock)
    dock._preset_combo.setCurrentIndex(dock._preset_combo.findText("Pop"))

    # Wipe QSettings so we can detect any spurious write from the Custom path.
    s = QSettings("Marmelade", "Marmelade")
    s.remove("mastering")
    s.sync()

    emits: list[int] = []
    dock.session_chain_changed.connect(lambda: emits.append(1))

    dock._preset_combo.setCurrentIndex(0)

    assert emits == [], "Custom selection must not emit"
    # No QSettings session keys were written by the Custom path.
    s2 = QSettings("Marmelade", "Marmelade")
    s2.beginGroup("mastering/session")
    try:
        assert s2.allKeys() == [], "Custom selection must not write QSettings"
    finally:
        s2.endGroup()


def test_manual_toggle_drives_combo_to_custom_without_extra_emit(
    qtbot, qapp
) -> None:
    """A manual checkbox toggle re-points the combo to 'Custom' (one emit)."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    # Apply a preset so the combo is on a named preset.
    dock._preset_combo.setCurrentIndex(dock._preset_combo.findText("Techno"))
    assert dock._preset_combo.currentText() == "Techno"

    emits: list[int] = []
    dock.session_chain_changed.connect(lambda: emits.append(1))

    # Flip a stage so the chain no longer matches Techno.
    dock.stage_checkbox("highpass").setChecked(
        not dock.stage_checkbox("highpass").isChecked()
    )

    # Exactly one emit (the genuine toggle), and the combo re-synced to Custom.
    assert len(emits) == 1, f"expected 1 emit from the toggle, got {len(emits)}"
    assert dock._preset_combo.currentIndex() == 0
    assert dock._preset_combo.currentText() == "Custom"


def test_toggle_back_into_exact_match_reflects_that_preset(qtbot, qapp) -> None:
    """Editing the chain into an exact preset match re-points the combo to it."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    # Seed every QSettings key for Ambient EXCEPT leave highpass disabled, so
    # the chain does NOT yet match Ambient; then toggle highpass on to land an
    # exact match.
    name = "Ambient"
    cfg = MASTERING_PRESETS[name]
    s = QSettings("Marmelade", "Marmelade")
    for stage, params in cfg.items():
        for param, value in params.items():
            s.setValue(f"mastering/session/{stage}/{param}", value)
    # Break the match: disable highpass (Ambient has it ON).
    s.setValue("mastering/session/highpass/enabled", False)
    s.sync()

    # Rebuild so the checkbox states reflect the seeded (non-matching) chain.
    dock2 = MasteringDock()
    qtbot.add_widget(dock2)
    assert dock2._preset_combo.currentText() == "Custom"

    # Toggle highpass ON → now the chain matches Ambient exactly.
    dock2.stage_checkbox("highpass").setChecked(True)
    assert dock2._preset_combo.currentText() == name


def test_apply_preset_does_not_cascade_per_checkbox_writes(qtbot, qapp) -> None:
    """During apply, checkbox flips do NOT each fire their own emit/write.

    Pins the signal-loop defusal: a single apply produces exactly one emit
    even though it flips multiple checkboxes (highpass/eq/compressor/limiter
    on, others off).
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    emits: list[int] = []
    dock.session_chain_changed.connect(lambda: emits.append(1))

    dock._apply_preset("Lo-fi")

    assert len(emits) == 1
    # Lo-fi flips lowpass ON (the one preset that does).
    assert dock.stage_checkbox("lowpass").isChecked() is True
    assert match_preset(load_session_chain_snapshot()) == "Lo-fi"


# ----------------------------- quick-260626-ked — session-wide VST3 row


def _qsettings_truthy(raw) -> bool:
    """QSettings serializes booleans as strings on some platforms."""
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.lower() in ("true", "1", "yes")
    return bool(raw)


def test_dock_renders_vst3_row(qtbot, qapp) -> None:
    """The dock surfaces a VST3 row — both a checkbox AND a gear button."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    assert dock.stage_checkbox("vst3") is not None
    assert dock._stage_gears.get("vst3") is not None


def test_vst3_checkbox_toggle_writes_qsettings_and_emits(qtbot, qapp) -> None:
    """Toggling the vst3 checkbox ON persists enabled + fires the signal.

    Mirrors the compressor toggle path — the checkbox lambda is built
    generically in _build_stage_row, so vst3 inherits the instant-write +
    emit behaviour the moment its row renders.
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    with qtbot.waitSignal(dock.session_chain_changed, timeout=1000):
        dock.stage_checkbox("vst3").setChecked(True)

    s = QSettings("Marmelade", "Marmelade")
    raw = s.value("mastering/session/vst3/enabled")
    assert _qsettings_truthy(raw), raw


def test_vst3_gear_success_locks_then_unlocks_and_persists(
    qtbot, qapp, monkeypatch
) -> None:
    """Gear → configure_vst3 → on_started locks, on_done(True) persists+unlocks.

    Monkeypatch the SOURCE symbol (``_on_vst3_gear_clicked`` does a LOCAL
    ``from marmelade.ui.vst3_config import configure_vst3``) and drive the
    callbacks synchronously — never launch the real worker.
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    # Snapshot of the three lock surfaces' enabled-state captured DURING
    # on_started, so we can assert they were disabled at that moment.
    locked_during_started: dict[str, bool] = {}

    def _fake_configure_vst3(parent, cfg, on_done=None, on_started=None):
        # (a) it received a dict cfg + both callbacks.
        assert isinstance(cfg, dict)
        assert on_done is not None and on_started is not None
        # (b) on_started fires → the three surfaces must be DISABLED now.
        on_started()
        locked_during_started["gear"] = dock._stage_gears["vst3"].isEnabled()
        locked_during_started["checkbox"] = dock.stage_checkbox("vst3").isEnabled()
        locked_during_started["combo"] = dock._preset_combo.isEnabled()
        # (c) mutate cfg to a configured + enabled state (as the real worker
        # does in its finished handler on exit_code 0).
        cfg.update(
            {
                "plugin_path": "/x.vst3",
                "plugin_name": "oXygen",
                "state_b64": "QUJD",
                "enabled": True,
            }
        )
        # (d) capture succeeded.
        on_done(True)
        return True

    monkeypatch.setattr(
        "marmelade.ui.vst3_config.configure_vst3", _fake_configure_vst3
    )

    emits: list[int] = []
    dock.session_chain_changed.connect(lambda: emits.append(1))

    dock._on_vst3_gear_clicked()

    # During on_started all three surfaces were disabled.
    assert locked_during_started == {
        "gear": False,
        "checkbox": False,
        "combo": False,
    }
    # After on_done they are all re-enabled.
    assert dock._stage_gears["vst3"].isEnabled() is True
    assert dock.stage_checkbox("vst3").isEnabled() is True
    assert dock._preset_combo.isEnabled() is True

    # The four vst3 keys persisted.
    s = QSettings("Marmelade", "Marmelade")
    assert _qsettings_truthy(s.value("mastering/session/vst3/enabled"))
    assert str(s.value("mastering/session/vst3/plugin_path")) == "/x.vst3"
    assert str(s.value("mastering/session/vst3/plugin_name")) == "oXygen"
    assert str(s.value("mastering/session/vst3/state_b64")) == "QUJD"

    # Checkbox reflects the captured enabled.
    assert dock.stage_checkbox("vst3").isChecked() is True
    # Exactly one emit.
    assert len(emits) == 1, f"expected exactly 1 emit, got {len(emits)}"
    # An enabled vst3 → no preset match → combo shows Custom.
    assert dock._preset_combo.currentText() == "Custom"


def test_vst3_gear_failure_reenables_without_persist(
    qtbot, qapp, monkeypatch
) -> None:
    """on_done(False) re-enables the surfaces but persists nothing + no emit."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    # Ensure no pre-existing enabled key.
    s = QSettings("Marmelade", "Marmelade")
    s.remove("mastering/session/vst3/enabled")
    s.sync()

    def _fake_configure_vst3(parent, cfg, on_done=None, on_started=None):
        on_started()
        # Worker error → cfg NOT mutated, on_done(False).
        on_done(False)
        return False

    monkeypatch.setattr(
        "marmelade.ui.vst3_config.configure_vst3", _fake_configure_vst3
    )

    emits: list[int] = []
    dock.session_chain_changed.connect(lambda: emits.append(1))

    dock._on_vst3_gear_clicked()

    # Surfaces re-enabled despite the failure.
    assert dock._stage_gears["vst3"].isEnabled() is True
    assert dock.stage_checkbox("vst3").isEnabled() is True
    assert dock._preset_combo.isEnabled() is True

    # Nothing persisted truthy, nothing emitted.
    s2 = QSettings("Marmelade", "Marmelade")
    assert not _qsettings_truthy(s2.value("mastering/session/vst3/enabled"))
    assert emits == [], f"failure path must not emit, got {emits}"


def test_enabled_vst3_forces_custom_preset_disabled_vst3_keeps_match(
    qtbot, qapp
) -> None:
    """Caveat pin: enabled vst3 → Custom; disabled vst3 → preset still matches.

    config_hash drops a DISABLED vst3 entirely, so a preset still matches with
    vst3 off. An ENABLED vst3 is kept in the hash → match_preset returns None →
    the combo reads "Custom".
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    # Apply a known preset (vst3 still disabled) → the combo shows that preset.
    name = "House"
    idx = dock._preset_combo.findText(name)
    dock._preset_combo.setCurrentIndex(idx)
    assert dock._preset_combo.currentText() == name
    # Disabled vst3 is dropped from config_hash → the preset still matches.
    assert match_preset(load_session_chain_snapshot()) == name

    # Now seed an ENABLED vst3 with a plugin + state.
    s = QSettings("Marmelade", "Marmelade")
    s.setValue("mastering/session/vst3/enabled", True)
    s.setValue("mastering/session/vst3/plugin_path", "/x.vst3")
    s.setValue("mastering/session/vst3/state_b64", "QUJD")
    s.sync()

    # An enabled vst3 → no preset match.
    assert match_preset(load_session_chain_snapshot()) is None
    # Resync the combo → it must read Custom (index 0).
    dock._sync_preset_combo()
    assert dock._preset_combo.currentIndex() == 0
    assert dock._preset_combo.currentText() == "Custom"


def test_session_snapshot_round_trips_configured_vst3(qtbot, qapp) -> None:
    """Inheritance pin: load_session_chain_snapshot carries a configured vst3.

    Proves a NEW keeper inherits the session vst3 with NO chain.py change —
    load_session_chain_snapshot iterates _STAGE_ORDER (includes "vst3") and
    fills missing keys from _SESSION_DEFAULTS["vst3"].
    """
    s = QSettings("Marmelade", "Marmelade")
    s.setValue("mastering/session/vst3/enabled", True)
    s.setValue("mastering/session/vst3/plugin_path", "/x.vst3")
    s.setValue("mastering/session/vst3/plugin_name", "oXygen")
    s.setValue("mastering/session/vst3/state_b64", "QUJD")
    s.sync()

    snapshot = load_session_chain_snapshot()
    assert snapshot["vst3"] == {
        "enabled": True,
        "plugin_path": "/x.vst3",
        "plugin_name": "oXygen",
        "state_b64": "QUJD",
    }


# ----------------------------- quick-260626-mih — Loudness/Normalize exclusion
#
# chain.py BYPASSES the Normalize tail whenever Loudness is enabled, so "both
# checked" is a meaningless UI state. Checking one auto-unchecks the other
# (QSettings + checkbox); unchecking a stage NEVER touches its sibling. Each
# user toggle emits session_chain_changed EXACTLY once and runs exactly one
# s.sync(). The preset apply path flips checkboxes under its own QSignalBlocker
# so the mutual-exclusion handler does not fire through it.


def test_checking_loudness_unchecks_normalize_dock(qtbot, qapp) -> None:
    """Checking Loudness writes normalize/enabled=False + unchecks its checkbox."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    # Enable normalize first (a genuine toggle).
    dock.stage_checkbox("normalize").setChecked(True)
    assert dock.stage_checkbox("normalize").isChecked() is True

    # Now check loudness → normalize forced OFF.
    dock.stage_checkbox("loudness").setChecked(True)

    s = QSettings("Marmelade", "Marmelade")
    assert _qsettings_truthy(s.value("mastering/session/loudness/enabled")) is True
    assert _qsettings_truthy(s.value("mastering/session/normalize/enabled")) is False
    assert dock.stage_checkbox("normalize").isChecked() is False


def test_checking_normalize_unchecks_loudness_dock(qtbot, qapp) -> None:
    """Checking Normalize writes loudness/enabled=False + unchecks its checkbox."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    dock.stage_checkbox("loudness").setChecked(True)
    assert dock.stage_checkbox("loudness").isChecked() is True

    dock.stage_checkbox("normalize").setChecked(True)

    s = QSettings("Marmelade", "Marmelade")
    assert _qsettings_truthy(s.value("mastering/session/normalize/enabled")) is True
    assert _qsettings_truthy(s.value("mastering/session/loudness/enabled")) is False
    assert dock.stage_checkbox("loudness").isChecked() is False


def test_unchecking_loudness_leaves_normalize_untouched_dock(qtbot, qapp) -> None:
    """Unchecking Loudness must NOT touch Normalize (only a CHECK flips sibling)."""
    dock = MasteringDock()
    qtbot.add_widget(dock)

    dock.stage_checkbox("normalize").setChecked(True)
    dock.stage_checkbox("loudness").setChecked(True)  # normalize forced OFF
    assert dock.stage_checkbox("normalize").isChecked() is False

    dock.stage_checkbox("loudness").setChecked(False)  # uncheck — no sibling touch

    s = QSettings("Marmelade", "Marmelade")
    assert _qsettings_truthy(s.value("mastering/session/loudness/enabled")) is False
    assert _qsettings_truthy(s.value("mastering/session/normalize/enabled")) is False
    assert dock.stage_checkbox("normalize").isChecked() is False


def test_mutual_exclusion_toggle_emits_exactly_once_dock(qtbot, qapp) -> None:
    """A user toggle that flips the sibling still emits session_chain_changed once.

    The sibling setChecked(False) is wrapped in QSignalBlocker so it does NOT
    re-enter the handler — exactly one emit per user toggle. The preset combo
    resyncs without crashing.
    """
    dock = MasteringDock()
    qtbot.add_widget(dock)

    dock.stage_checkbox("normalize").setChecked(True)

    emits: list[int] = []
    dock.session_chain_changed.connect(lambda: emits.append(1))

    with qtbot.waitSignal(dock.session_chain_changed, timeout=1000):
        dock.stage_checkbox("loudness").setChecked(True)

    assert len(emits) == 1, f"expected exactly 1 emit, got {len(emits)}"
    # The sibling really flipped + the combo resynced without crashing.
    assert dock.stage_checkbox("normalize").isChecked() is False
    assert dock._preset_combo.currentIndex() >= 0


def test_preset_apply_preserves_exact_flags_despite_mutual_exclusion_dock(
    qtbot, qapp
) -> None:
    """A preset enabling loudness + disabling normalize keeps its EXACT flags.

    'House' has loudness ON + normalize OFF — applying it via the combo must
    yield exactly that in QSettings + checkboxes (the mutual-exclusion handler
    must NOT fire through the preset apply path).
    """
    name = "House"
    assert MASTERING_PRESETS[name]["loudness"]["enabled"] is True
    assert MASTERING_PRESETS[name]["normalize"]["enabled"] is False

    dock = MasteringDock()
    qtbot.add_widget(dock)

    dock._preset_combo.setCurrentIndex(dock._preset_combo.findText(name))

    s = QSettings("Marmelade", "Marmelade")
    assert _qsettings_truthy(s.value("mastering/session/loudness/enabled")) is True
    assert _qsettings_truthy(s.value("mastering/session/normalize/enabled")) is False
    # Checkboxes reflect the preset's enabled flags exactly.
    for stage, params in MASTERING_PRESETS[name].items():
        if stage == "ending_fx":
            continue  # per-keeper-only stage; no session row
        assert dock.stage_checkbox(stage).isChecked() is bool(
            params.get("enabled", False)
        ), stage
    # The snapshot still round-trips to the preset (exclusion did not perturb it).
    assert match_preset(load_session_chain_snapshot()) == name
