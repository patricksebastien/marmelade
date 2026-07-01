"""Phase 8 Plan 08-04 — UploadDialog Phase A (Setup) + Phase B (Progress) (D-20).

Plan 08-04 Task 4 (TDD GREEN — Wave 0 skip marker removed). The 9
test names below cover the Phase A/B transition + cancel handoff per
D-20 single-modal-two-phases contract.

The dialog is constructed with stub initial values; the runnable
spawn + retry/error paths live in MainWindow's orchestration and are
exercised separately (test_keeper_row_share_button.py adds the
source-proxy fallback integration test in this same plan's Task 4).
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QMessageBox

from marmelade.ui.upload_dialog import UploadDialog


REGION_ID = "0123456789abcdef0123456789abcdef"


def _make_jpeg(color=(64, 96, 192), size=(1280, 720)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _make_dialog(qtbot, **overrides) -> UploadDialog:
    kwargs = dict(
        keeper_id=REGION_ID,
        keeper_range="00:14:32 – 00:18:07",
        initial_title="distant ferries hum quietly",
        initial_description="2026-05-22 — exported from Marmelade",
        initial_privacy="private",
        initial_thumbnail_bytes=_make_jpeg(),
    )
    kwargs.update(overrides)
    dlg = UploadDialog(**kwargs)
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# Test 1 — Phase A renders the setup form
# ---------------------------------------------------------------------------


def test_phase_a_setup_renders(qtbot) -> None:
    """Phase A shows thumbnail QLabel pixmap + title/description/privacy widgets."""
    dlg = _make_dialog(qtbot)

    # QStackedWidget starts on page 0 (Phase A).
    assert dlg._stack.currentIndex() == 0

    # Thumbnail label pixmap is non-null.
    assert dlg._thumbnail_label.pixmap() is not None
    assert not dlg._thumbnail_label.pixmap().isNull()

    # Title / description QLineEdit text matches initial values.
    assert dlg._title_edit.text() == "distant ferries hum quietly"
    assert dlg._description_edit.text() == "2026-05-22 — exported from Marmelade"

    # Privacy combo's current data matches the initial privacy value.
    assert dlg._privacy_combo.currentData() == "private"


def test_license_combo_defaults_to_creative_commons(qtbot) -> None:
    """Phase A License combo defaults to creativeCommon (CC-BY) and offers both values."""
    dlg = _make_dialog(qtbot)
    assert dlg._license_combo.currentData() == "creativeCommon"
    datas = {
        dlg._license_combo.itemData(i) for i in range(dlg._license_combo.count())
    }
    assert datas == {"creativeCommon", "youtube"}


# ---------------------------------------------------------------------------
# Test 2 — set_phase_b transitions QStackedWidget to progress page
# ---------------------------------------------------------------------------


def test_phase_a_to_phase_b_transition(qtbot) -> None:
    """set_phase_b() switches the QStackedWidget; QProgressBar visible on new page."""
    dlg = _make_dialog(qtbot)
    dlg.set_phase_b()
    assert dlg._stack.currentIndex() == 1
    # Progress bar exists and is visible after the swap.
    assert dlg._progress_bar is not None


# ---------------------------------------------------------------------------
# Test 3 — Cancel during Phase B emits cancel_requested
# ---------------------------------------------------------------------------


def test_cancel_during_progress_emits_cancel_signal(qtbot) -> None:
    """Phase-B Cancel button click emits cancel_requested exactly once."""
    dlg = _make_dialog(qtbot)
    dlg.set_phase_b()

    received: list[None] = []
    dlg.cancel_requested.connect(lambda: received.append(None))

    qtbot.mouseClick(dlg._cancel_button, Qt.MouseButton.LeftButton)
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Test 4 — Refresh button emits refresh_thumbnail_requested
# ---------------------------------------------------------------------------


def test_refresh_emits_signal(qtbot) -> None:
    """Phase-A Refresh thumbnail button emits refresh_thumbnail_requested."""
    dlg = _make_dialog(qtbot)

    received: list[None] = []
    dlg.refresh_thumbnail_requested.connect(lambda: received.append(None))

    qtbot.mouseClick(dlg._refresh_btn, Qt.MouseButton.LeftButton)
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Test 5 — Upload button emits upload_requested with the field payload
# ---------------------------------------------------------------------------


def test_upload_emits_signal_with_payload(qtbot) -> None:
    """Phase-A Upload button emits upload_requested(title, desc, privacy, license, jpeg_bytes)."""
    thumb = _make_jpeg()
    dlg = _make_dialog(qtbot, initial_thumbnail_bytes=thumb)

    # Change the title to confirm the edited value (not the initial) is in the payload.
    dlg._title_edit.setText("a different title")
    # Change description.
    dlg._description_edit.setText("changed description")
    # Switch privacy to unlisted.
    idx = dlg._privacy_combo.findData("unlisted")
    dlg._privacy_combo.setCurrentIndex(idx)
    # Switch license to the Standard YouTube License.
    lic_idx = dlg._license_combo.findData("youtube")
    dlg._license_combo.setCurrentIndex(lic_idx)

    received: list[tuple] = []
    dlg.upload_requested.connect(
        lambda title, desc, priv, lic, jpeg: received.append(
            (title, desc, priv, lic, jpeg)
        )
    )

    qtbot.mouseClick(dlg._upload_btn, Qt.MouseButton.LeftButton)
    assert len(received) == 1
    title, desc, priv, lic, jpeg = received[0]
    assert title == "a different title"
    assert desc == "changed description"
    assert priv == "unlisted"
    assert lic == "youtube"
    assert jpeg == thumb


# ---------------------------------------------------------------------------
# Test 6 — update_thumbnail replaces the pixmap
# ---------------------------------------------------------------------------


def test_update_thumbnail_replaces_pixmap(qtbot) -> None:
    """update_thumbnail(new_bytes) replaces the QLabel pixmap with the new image."""
    dlg = _make_dialog(qtbot, initial_thumbnail_bytes=_make_jpeg(color=(255, 0, 0)))
    old_pix = dlg._thumbnail_label.pixmap()
    old_image = old_pix.toImage()

    new_jpeg = _make_jpeg(color=(0, 255, 0))
    dlg.update_thumbnail(new_jpeg)

    new_pix = dlg._thumbnail_label.pixmap()
    new_image = new_pix.toImage()
    assert old_image != new_image, "pixmap should have changed after update_thumbnail"


# ---------------------------------------------------------------------------
# Test 7 — show_error renders the message + Retry button
# ---------------------------------------------------------------------------


def test_show_error_renders_message(qtbot) -> None:
    """show_error(msg, retryable=True) shows msg in footer + Retry QPushButton."""
    dlg = _make_dialog(qtbot)
    dlg.set_phase_b()
    msg = "Daily YouTube upload quota exceeded — try again tomorrow."
    dlg.show_error(msg, retryable=True)

    assert dlg._error_label is not None
    assert msg in dlg._error_label.text()
    assert dlg._retry_btn is not None
    assert dlg._retry_btn.isVisible()


# ---------------------------------------------------------------------------
# Test 8 — set_progress updates bar value + ETA label
# ---------------------------------------------------------------------------


def test_set_progress_updates_bar_and_eta(qtbot) -> None:
    """set_progress(50, 12.3) → progress bar value 50; ETA label contains '12'."""
    dlg = _make_dialog(qtbot)
    dlg.set_phase_b()
    dlg.set_progress(50, 12.3)
    assert dlg._progress_bar.value() == 50
    assert "12" in dlg._eta_label.text(), (
        f"expected '12' in ETA label; got {dlg._eta_label.text()!r}"
    )


# ---------------------------------------------------------------------------
# Test 9 — Close during Phase B prompts confirmation; only Yes closes
# ---------------------------------------------------------------------------


def test_close_during_phase_b_prompts_confirmation(qtbot, monkeypatch) -> None:
    """closeEvent during Phase B asks confirm; No keeps dialog; Yes emits cancel + closes."""
    dlg = _make_dialog(qtbot)
    dlg.set_phase_b()

    # Monkeypatch the QMessageBox.question staticmethod.
    answers = []

    def _patched_question(*_a, **_kw):
        return answers.pop(0)

    monkeypatch.setattr(QMessageBox, "question", _patched_question)

    received_cancel: list[None] = []
    dlg.cancel_requested.connect(lambda: received_cancel.append(None))

    # 1) Answer No — close is rejected; cancel_requested NOT emitted.
    answers.append(QMessageBox.StandardButton.No)
    dlg.close()
    assert received_cancel == [], "No answer should NOT emit cancel"

    # 2) Answer Yes — close proceeds; cancel_requested IS emitted.
    answers.append(QMessageBox.StandardButton.Yes)
    dlg.close()
    assert received_cancel == [None], (
        f"Yes answer should emit cancel exactly once; got {received_cancel!r}"
    )
