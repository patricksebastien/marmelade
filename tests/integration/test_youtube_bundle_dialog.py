"""Phase 8 Plan 08-05 Task 3 — BundleDialog + bundle Share button + MainWindow orchestration.

Pins:
* D-19 — bundle Share button at top of Keepers dock alongside Master & Export All.
* D-02 — bundle button DISABLED when ANY keeper is unmastered (tooltip explains).
* D-03 — single thumbnail for the whole bundle (NOT per-keeper sequence).
* D-04 — spacer QDoubleSpinBox default 2.0 s, range 0-10 s.
* D-05 — fallback QListWidget(InternalMove) reorder via use_fallback_reorder=True.
* D-06 — Save-to file picker; no QSettings persistence of save_path.
* D-23 — title fields are user-editable.

Composition over inheritance: BundleDialog copies UploadDialog's Phase A/B
widget vocabulary; the future mixin extraction is Plan 08-06 Task 3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDoubleSpinBox

from marmelade.audio.sidecar_cache import Region
from marmelade.ui.bundle_dialog import BundleDialog
from marmelade.ui.keepers_sidebar import KeepersSidebar


# Region UUID fixtures.
RID_A = "0123456789abcdef0123456789abcde0"
RID_B = "0123456789abcdef0123456789abcde1"
RID_C = "0123456789abcdef0123456789abcde2"


def _make_region(rid: str, start_sec: float) -> Region:
    return Region(
        id=rid,
        start_sec=float(start_sec),
        end_sec=float(start_sec + 10.0),
        state="keeper",
        note="",
    )


# A minimal 1x1 transparent JPEG (placeholder for the thumbnail bytes
# arg — BundleDialog stashes the bytes for the upload signal payload
# but does not need them to be a valid image to render the QLabel).
_FAKE_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605"
    "08070707090908064646464646464646464646464646464646464646464646"
)


def _make_dialog(qtbot, *, use_fallback_reorder: bool = False) -> BundleDialog:
    dlg = BundleDialog(
        keepers=[
            (RID_A, "00:00:10 – 00:00:20"),
            (RID_B, "00:00:30 – 00:00:40"),
            (RID_C, "00:00:50 – 00:01:00"),
        ],
        initial_title="bundle of three",
        initial_description="2026-05-22 bundle — exported from Marmelade",
        initial_privacy="private",
        initial_thumbnail_bytes=_FAKE_JPEG,
        initial_spacer_sec=2.0,
        use_fallback_reorder=use_fallback_reorder,
    )
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# Test 1 — Dialog renders Phase A widgets
# ---------------------------------------------------------------------------


def test_dialog_renders_phase_a_widgets(qtbot) -> None:
    """BundleDialog Phase A surfaces the required widgets."""
    dlg = _make_dialog(qtbot)
    # Phase A is the default visible page.
    assert dlg._stack.currentIndex() == 0
    # Required Phase A widgets — names follow UploadDialog's vocabulary.
    assert hasattr(dlg, "_title_edit")
    assert hasattr(dlg, "_description_edit")
    assert hasattr(dlg, "_privacy_combo")
    assert hasattr(dlg, "_spacer_spinbox")
    assert hasattr(dlg, "_thumbnail_label")
    assert hasattr(dlg, "_refresh_btn")
    assert hasattr(dlg, "_export_only_btn")
    assert hasattr(dlg, "_export_and_upload_btn")


# ---------------------------------------------------------------------------
# Test 2 — Spacer spinbox default 2.0, range 0..10
# ---------------------------------------------------------------------------


def test_spacer_spinbox_defaults_and_range(qtbot) -> None:
    """QDoubleSpinBox: range 0..10, decimals ≥1, default 2.0 (D-04)."""
    dlg = _make_dialog(qtbot)
    sb: QDoubleSpinBox = dlg._spacer_spinbox
    assert sb.minimum() == 0.0
    assert sb.maximum() == 10.0
    assert sb.value() == 2.0
    # Suffix " s" so the user reads "2.0 s".
    assert " s" in sb.suffix() or "s" in sb.suffix()


# ---------------------------------------------------------------------------
# Test 3 — Export MP3 only click emits export_mp3_only_requested
# ---------------------------------------------------------------------------


def test_export_mp3_only_button_emits_signal(qtbot) -> None:
    """Click on Export MP3 only emits (save_path_placeholder, spacer_sec, ordered_ids).

    Plan 08-05 signal carries spacer_sec + ordered_region_ids; the
    MainWindow slot is the one that opens QFileDialog for the actual
    save path. The dialog itself emits with save_path = "" so the slot
    knows it must prompt.
    """
    dlg = _make_dialog(qtbot)
    received: list[tuple] = []
    dlg.export_mp3_only_requested.connect(
        lambda *args: received.append(args)
    )

    dlg._spacer_spinbox.setValue(1.5)
    qtbot.mouseClick(dlg._export_only_btn, Qt.MouseButton.LeftButton)
    assert len(received) == 1
    save_path, spacer_sec, ordered_ids = received[0]
    assert save_path == ""  # MainWindow prompts for the actual path
    assert spacer_sec == 1.5
    assert ordered_ids == [RID_A, RID_B, RID_C]


# ---------------------------------------------------------------------------
# Test 4 — Export + Upload click emits export_and_upload_requested with full payload
# ---------------------------------------------------------------------------


def test_export_and_upload_button_emits_signal(qtbot) -> None:
    """Click on Export + Upload emits the full snippet/status/thumbnail payload."""
    dlg = _make_dialog(qtbot)
    received: list[tuple] = []
    dlg.export_and_upload_requested.connect(
        lambda *args: received.append(args)
    )

    dlg._spacer_spinbox.setValue(3.0)
    dlg._title_edit.setText("custom title")
    dlg._description_edit.setText("custom desc")
    # Set privacy to public via userData lookup.
    idx = dlg._privacy_combo.findData("public")
    assert idx >= 0
    dlg._privacy_combo.setCurrentIndex(idx)
    # Set license to the Standard YouTube License.
    lic_idx = dlg._license_combo.findData("youtube")
    assert lic_idx >= 0
    dlg._license_combo.setCurrentIndex(lic_idx)

    qtbot.mouseClick(dlg._export_and_upload_btn, Qt.MouseButton.LeftButton)
    assert len(received) == 1
    (
        save_path,
        spacer_sec,
        ordered_ids,
        title,
        desc,
        privacy,
        license_val,
        jpeg_bytes,
    ) = received[0]
    assert save_path == ""
    assert spacer_sec == 3.0
    assert ordered_ids == [RID_A, RID_B, RID_C]
    assert title == "custom title"
    assert desc == "custom desc"
    assert privacy == "public"
    assert license_val == "youtube"
    assert jpeg_bytes == _FAKE_JPEG


def test_license_combo_defaults_to_creative_commons(qtbot) -> None:
    """BundleDialog Phase A License combo defaults to creativeCommon (CC-BY)."""
    dlg = _make_dialog(qtbot)
    assert dlg._license_combo.currentData() == "creativeCommon"
    datas = {
        dlg._license_combo.itemData(i) for i in range(dlg._license_combo.count())
    }
    assert datas == {"creativeCommon", "youtube"}


# ---------------------------------------------------------------------------
# Test 5 — Refresh thumbnail emits signal
# ---------------------------------------------------------------------------


def test_refresh_thumbnail_button_emits_signal(qtbot) -> None:
    """Refresh button click emits refresh_thumbnail_requested."""
    dlg = _make_dialog(qtbot)
    received: list = []
    dlg.refresh_thumbnail_requested.connect(lambda: received.append(True))

    qtbot.mouseClick(dlg._refresh_btn, Qt.MouseButton.LeftButton)
    assert received == [True]


# ---------------------------------------------------------------------------
# Test 6 — Cancel button emits cancel_requested (Phase B)
# ---------------------------------------------------------------------------


def test_phase_b_cancel_button_emits_cancel(qtbot) -> None:
    """Phase B Cancel button emits cancel_requested."""
    dlg = _make_dialog(qtbot)
    dlg.set_phase_b()
    received: list = []
    dlg.cancel_requested.connect(lambda: received.append(True))

    # Use .click() (direct invocation) rather than qtbot.mouseClick —
    # the latter requires a mapped/shown window and hangs the offscreen
    # platform when the dialog has been swapped to Phase B without
    # show()n. .click() generates the clicked signal synchronously.
    dlg._cancel_button.click()
    assert received == [True]

    # Cleanup: swap back to Phase A so the dialog teardown doesn't
    # trigger the Phase B closeEvent confirm prompt (which would hang
    # the offscreen test runner waiting for a QMessageBox response).
    dlg._stack.setCurrentIndex(0)


# ---------------------------------------------------------------------------
# Test 7 — Fallback reorder mode toggles QListWidget(InternalMove)
# ---------------------------------------------------------------------------


def test_fallback_reorder_mode_enables_internal_move(qtbot) -> None:
    """use_fallback_reorder=True flips the keepers QListWidget to InternalMove."""
    from PySide6.QtWidgets import QAbstractItemView

    dlg = _make_dialog(qtbot, use_fallback_reorder=True)
    assert hasattr(dlg, "_keepers_list")
    assert dlg._keepers_list.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove
    # Apply Order button only exists/is configured-visible in fallback mode.
    # The dialog is not show()n in tests so isVisible() is always False —
    # check the configured visibility flag via isHidden() instead.
    assert hasattr(dlg, "_apply_order_btn")
    assert not dlg._apply_order_btn.isHidden(), (
        "Apply Order button should be visible (not hidden) in fallback mode"
    )


def test_default_mode_keepers_list_is_read_only(qtbot) -> None:
    """Default (non-fallback) mode: keepers list is NOT InternalMove."""
    from PySide6.QtWidgets import QAbstractItemView

    dlg = _make_dialog(qtbot, use_fallback_reorder=False)
    assert dlg._keepers_list.dragDropMode() == QAbstractItemView.DragDropMode.NoDragDrop


# ---------------------------------------------------------------------------
# Test 8 — Bundle Share button in sidebar: position + disabled-state + tooltip
# ---------------------------------------------------------------------------


def test_bundle_button_exists_at_top_of_sidebar(qtbot) -> None:
    """KeepersSidebar._bundle_button is a top-level widget (D-19)."""
    sidebar = KeepersSidebar()
    qtbot.addWidget(sidebar)
    assert hasattr(sidebar, "_bundle_button")
    # The bundle button sits in the outer QVBoxLayout alongside the
    # batch button (NOT inside the QStackedWidget).
    outer_layout = sidebar.layout()
    # Quick-260615-l4y two-button redesign: Master (_batch_button) at
    # index 0, Export (_export_button) at index 1, Share (_bundle_button)
    # at index 2.
    assert outer_layout.indexOf(sidebar._batch_button) == 0
    assert outer_layout.indexOf(sidebar._export_button) == 1
    assert outer_layout.indexOf(sidebar._bundle_button) == 2


def test_bundle_button_disabled_when_no_keepers(qtbot) -> None:
    """Empty Keepers dock → bundle button disabled with explanatory tooltip."""
    sidebar = KeepersSidebar()
    qtbot.addWidget(sidebar)
    assert sidebar._bundle_button.isEnabled() is False
    # Disabled tooltip mentions mastering requirement OR keeper requirement.
    tt = sidebar._bundle_button.toolTip()
    assert "Master" in tt or "keeper" in tt.lower(), (
        f"disabled tooltip should explain the gate; got {tt!r}"
    )


def test_bundle_button_disabled_when_keepers_unmastered(qtbot, monkeypatch) -> None:
    """Keepers present but unmastered → bundle button disabled (D-02 gate)."""
    sidebar = KeepersSidebar()
    qtbot.addWidget(sidebar)
    sidebar.add_row(_make_region(RID_A, 10.0))
    # By default mastering=None; bundle button stays disabled because
    # _refresh_bundle_button_enabled finds no fresh cache.
    assert sidebar._bundle_button.isEnabled() is False
    tt = sidebar._bundle_button.toolTip()
    assert "Master" in tt, (
        f"disabled tooltip should mention mastering; got {tt!r}"
    )


def test_bundle_button_emits_signal_when_clicked(qtbot, monkeypatch) -> None:
    """Bundle button click emits bundle_share_requested when enabled."""
    sidebar = KeepersSidebar()
    qtbot.addWidget(sidebar)
    # Force-enable by patching the gate (real enable path requires fresh
    # mastered caches on disk — overkill for this unit-ish test).
    sidebar._bundle_button.setEnabled(True)

    received: list = []
    sidebar.bundle_share_requested.connect(lambda: received.append(True))

    qtbot.mouseClick(sidebar._bundle_button, Qt.MouseButton.LeftButton)
    assert received == [True]


# ---------------------------------------------------------------------------
# Test 9 — Source greps: _PhaseAB mixin NOT extracted yet (08-06 owns)
# ---------------------------------------------------------------------------


def test_main_window_coerces_qsettings_spacer_to_float() -> None:
    """RESEARCH Pitfall 4 — main_window reads bundle_spacer_sec from QSettings as float.

    QSettings.value() on Linux returns ``str`` for any non-Qt-native
    type written via setValue() — so a saved ``2.0`` round-trips as
    ``"2.0"`` and a downstream ``QDoubleSpinBox.setValue("2.0")``
    silently coerces to 0.0. Either an explicit ``float()`` cast or a
    ``_coerce_like`` helper at the read site is required.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    src = repo_root / "src" / "marmelade" / "ui" / "main_window.py"
    text = src.read_text()
    # Look for either explicit float(...) cast around a value(...spacer...)
    # call OR the spacer arg passed to BundleDialog wrapped in float().
    import re

    has_explicit_float = bool(
        re.search(r"float\s*\([^)]*value\([^)]*spacer", text)
        or re.search(r"initial_spacer_sec\s*=\s*float\(", text)
    )
    has_coerce = "_coerce_like" in text or "coerce_like" in text
    assert has_explicit_float or has_coerce, (
        "main_window must coerce QSettings bundle_spacer_sec value to float "
        "(RESEARCH Pitfall 4 — Linux QSettings.value returns str)"
    )


def test_main_window_reuses_youtube_upload_runnable() -> None:
    """MainWindow imports YouTubeUploadRunnable (REUSED, not duplicated)."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    src = repo_root / "src" / "marmelade" / "ui" / "main_window.py"
    text = src.read_text()
    assert "YouTubeUploadRunnable" in text


def test_main_window_wires_bundle_share_requested() -> None:
    """MainWindow connects sidebar.bundle_share_requested to _on_bundle_share_requested."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    src = repo_root / "src" / "marmelade" / "ui" / "main_window.py"
    text = src.read_text()
    assert "self._keepers_sidebar.bundle_share_requested.connect" in text
    assert "def _on_bundle_share_requested" in text


def test_main_window_has_bundle_orchestration_slots() -> None:
    """MainWindow defines the 4 bundle slots per the plan."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    src = repo_root / "src" / "marmelade" / "ui" / "main_window.py"
    text = src.read_text()
    assert "def _on_bundle_share_requested" in text
    assert "def _on_bundle_export_mp3_only" in text
    assert "def _on_bundle_export_and_upload" in text
    assert "def _on_bundle_cancel_requested" in text


# ---------------------------------------------------------------------------
# Test 10 — _refresh_bundle_button_enabled is defined on KeepersSidebar
# ---------------------------------------------------------------------------


def test_sidebar_has_refresh_bundle_button_enabled() -> None:
    """KeepersSidebar exposes _refresh_bundle_button_enabled helper."""
    assert hasattr(KeepersSidebar, "_refresh_bundle_button_enabled")
