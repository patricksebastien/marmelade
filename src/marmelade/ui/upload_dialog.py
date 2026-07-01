"""Modal per-keeper UploadDialog — Phase A (Setup) + Phase B (Progress) (D-20).

Single QDialog hosting two pages swapped via :class:`QStackedWidget`:

* **Page 0 (Phase A — Setup)** — thumbnail preview, Refresh thumbnail
  button, Title QLineEdit (default = pseudo-poem from
  :func:`marmelade.util.poem_generator.generate`), Description
  QLineEdit, Privacy QComboBox (Private / Unlisted / Public, initial
  selection from QSettings per D-21), Upload + Cancel bottom row.
* **Page 1 (Phase B — Progress)** — QProgressBar 0..100, ETA QLabel,
  Cancel QPushButton always visible (D-24 + D-20).

D-20: the same modal stays open through BOTH phases — no separate
progress window. User stays focused.

D-23: title field is ALWAYS user-editable. The pseudo-poem is a
starting point; the user can replace it before clicking Upload.

D-25: on upload failure the dialog stays open and surfaces an inline
error label + Retry button. Token-expired errors trigger a silent
re-auth attempt one level up (in :class:`YouTubeUploadRunnable`);
only on refresh-failure does the dialog show the message.

closeEvent: title-bar X during Phase A closes silently (no upload
started). During Phase B it raises a ``QMessageBox.question`` confirm
prompt; only Yes proceeds and emits ``cancel_requested``.

The dialog NEVER spawns the upload itself — it emits signals and
:class:`MainWindow` orchestrates the
:class:`marmelade.youtube.upload_runnable.YouTubeUploadRunnable`
spawn + retry/restart cycle. This separation keeps the dialog
testable without a real upload queue.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from marmelade.ui._phase_ab_mixin import _PhaseABMixin


_THUMB_PREVIEW_W: int = 320
_THUMB_PREVIEW_H: int = 180


class UploadDialog(_PhaseABMixin, QDialog):
    """Modal per-keeper YouTube upload dialog.

    Args:
        keeper_id: Region UUID — included in callers' state lookup.
        keeper_range: Display string ``"00:14:32 – 00:18:07"`` used in
            the window title.
        initial_title: Pre-filled title text (typically
            :func:`marmelade.util.poem_generator.generate`). User-
            editable per D-23.
        initial_description: Pre-filled description text.
        initial_privacy: One of ``"private"``, ``"unlisted"``,
            ``"public"`` — read from QSettings per D-21.
        initial_thumbnail_bytes: JPEG bytes for the thumbnail preview.
            Loaded into the QLabel pixmap via
            :meth:`QPixmap.loadFromData` and scaled to 320x180.
        parent: Optional parent widget (typically MainWindow).

    Signals:
        upload_requested(title, description, privacy, license, jpeg_bytes):
            emitted when the user clicks Upload. MainWindow's
            :meth:`_on_upload_initiated` slot receives these and spawns
            the runnable. ``license`` is one of YouTube's two accepted
            values: ``"creativeCommon"`` (CC-BY, default) or
            ``"youtube"`` (Standard YouTube License).
        cancel_requested(): emitted when the user clicks Cancel (in
            Phase B) OR confirms the title-bar X prompt in Phase B.
        retry_requested(): emitted when the user clicks the Retry
            button that appears in the error footer.
        refresh_thumbnail_requested(): emitted when the user clicks the
            Refresh thumbnail button in Phase A. MainWindow re-fetches
            with an incremented nonce.
    """

    # Class-level thumb preview constants — read by _PhaseABMixin's
    # _apply_thumbnail_pixmap so the mixin does not need to know the
    # dialog-specific sizes.
    _THUMB_PREVIEW_W: int = _THUMB_PREVIEW_W
    _THUMB_PREVIEW_H: int = _THUMB_PREVIEW_H

    upload_requested = Signal(str, str, str, str, bytes)
    cancel_requested = Signal()
    retry_requested = Signal()
    refresh_thumbnail_requested = Signal()

    def __init__(
        self,
        *,
        keeper_id: str,
        keeper_range: str,
        initial_title: str,
        initial_description: str,
        initial_privacy: str,
        initial_thumbnail_bytes: bytes,
        initial_license: str = "creativeCommon",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Share to YouTube — {keeper_range}")
        self.setModal(True)  # D-20
        self.resize(560, 580)

        self.keeper_id = keeper_id
        self._initial_thumbnail_bytes: bytes = initial_thumbnail_bytes
        self._current_thumbnail_bytes: bytes = initial_thumbnail_bytes

        # Error-footer widgets are created lazily by :meth:`show_error`.
        self._error_label: QLabel | None = None
        self._retry_btn: QPushButton | None = None

        self._build_ui(
            initial_title=initial_title,
            initial_description=initial_description,
            initial_privacy=initial_privacy,
            initial_license=initial_license,
        )

    # ----------------------------------------------------------- public API
    #
    # Phase 8 Plan 08-06 Task 3 — set_phase_b / set_progress / show_error /
    # update_thumbnail / _apply_thumbnail_pixmap are provided by
    # ``_PhaseABMixin``. The methods used to live in this class
    # verbatim; they were extracted into the mixin so BundleDialog can
    # share one implementation (revision iter 1 W6 close-out).

    # ----------------------------------------------------------- internal

    def _build_ui(
        self,
        *,
        initial_title: str,
        initial_description: str,
        initial_privacy: str,
        initial_license: str,
    ) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(8)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_phase_a(
            initial_title=initial_title,
            initial_description=initial_description,
            initial_privacy=initial_privacy,
            initial_license=initial_license,
        ))
        self._stack.addWidget(self._build_phase_b())
        self._stack.setCurrentIndex(0)
        outer.addWidget(self._stack)

    def _build_phase_a(
        self,
        *,
        initial_title: str,
        initial_description: str,
        initial_privacy: str,
        initial_license: str,
    ) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Thumbnail preview (320x180 scaled).
        self._thumbnail_label = QLabel()
        self._thumbnail_label.setFixedSize(_THUMB_PREVIEW_W, _THUMB_PREVIEW_H)
        self._thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumbnail_label.setStyleSheet(
            "border: 1px solid #2F2F33; background-color: #1B1B1C;"
        )
        self._apply_thumbnail_pixmap(self._initial_thumbnail_bytes)
        thumb_row = QHBoxLayout()
        thumb_row.addStretch(1)
        thumb_row.addWidget(self._thumbnail_label, 0)
        thumb_row.addStretch(1)
        layout.addLayout(thumb_row)

        # Refresh thumbnail button.
        self._refresh_btn = QPushButton("Refresh thumbnail")
        self._refresh_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._refresh_btn.clicked.connect(self.refresh_thumbnail_requested.emit)
        refresh_row = QHBoxLayout()
        refresh_row.addStretch(1)
        refresh_row.addWidget(self._refresh_btn, 0)
        refresh_row.addStretch(1)
        layout.addLayout(refresh_row)

        layout.addSpacing(8)

        # Title field — pre-filled but always user-editable (D-23).
        title_label = QLabel("Title")
        title_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(title_label)
        self._title_edit = QLineEdit(initial_title)
        self._title_edit.setMaxLength(100)  # YouTube caps title at 100 chars.
        layout.addWidget(self._title_edit)

        # Description field.
        desc_label = QLabel("Description")
        desc_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(desc_label)
        self._description_edit = QLineEdit(initial_description)
        self._description_edit.setMaxLength(5000)  # YouTube cap.
        layout.addWidget(self._description_edit)

        # Privacy combo.
        privacy_label = QLabel("Privacy")
        privacy_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(privacy_label)
        self._privacy_combo = QComboBox()
        self._privacy_combo.addItem("Private — only you", userData="private")
        self._privacy_combo.addItem("Unlisted — anyone with the link", userData="unlisted")
        self._privacy_combo.addItem("Public — anyone can find it", userData="public")
        # Initial selection.
        idx = self._privacy_combo.findData(initial_privacy)
        if idx == -1:
            idx = self._privacy_combo.findData("private")
        self._privacy_combo.setCurrentIndex(idx if idx != -1 else 0)
        layout.addWidget(self._privacy_combo)

        # License combo — maps directly to YouTube's status.license field.
        # Only two values are accepted by the API: creativeCommon / youtube.
        license_label = QLabel("License")
        license_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(license_label)
        self._license_combo = QComboBox()
        self._license_combo.addItem(
            "Creative Commons – Attribution (CC-BY)", userData="creativeCommon"
        )
        self._license_combo.addItem(
            "Standard YouTube License", userData="youtube"
        )
        lic_idx = self._license_combo.findData(initial_license)
        if lic_idx == -1:
            lic_idx = self._license_combo.findData("creativeCommon")
        self._license_combo.setCurrentIndex(lic_idx if lic_idx != -1 else 0)
        layout.addWidget(self._license_combo)

        # Format combo (common audio formats). Shares the populator
        # with BundleDialog so both flows offer the same list.
        format_label = QLabel("Format")
        format_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(format_label)
        self._format_combo = QComboBox()
        from marmelade.ui.format_choices import populate_format_combo
        populate_format_combo(self._format_combo, default="mp3_320")
        layout.addWidget(self._format_combo)

        layout.addStretch(1)

        # Bottom button row.
        self._upload_btn = QPushButton("Upload")
        self._upload_btn.setDefault(True)
        self._upload_btn.clicked.connect(self._on_upload_clicked)
        self._setup_cancel_btn = QPushButton("Cancel")
        self._setup_cancel_btn.clicked.connect(self.reject)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(self._setup_cancel_btn, 0)
        bottom.addWidget(self._upload_btn, 0)
        layout.addLayout(bottom)

        return page

    def _build_phase_b(self) -> QWidget:
        page = QWidget()
        self._phase_b_layout = QVBoxLayout(page)
        self._phase_b_layout.setContentsMargins(0, 0, 0, 0)
        self._phase_b_layout.setSpacing(8)

        heading = QLabel("Uploading to YouTube…")
        heading.setStyleSheet("font-size: 12pt; font-weight: 600;")
        self._phase_b_layout.addWidget(heading)

        # Determinate 0..100 progress bar.
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._phase_b_layout.addWidget(self._progress_bar)

        # ETA label.
        self._eta_label = QLabel("ETA: —")
        self._eta_label.setStyleSheet("color: #9CA3AF; font-size: 10pt;")
        self._phase_b_layout.addWidget(self._eta_label)

        self._phase_b_layout.addStretch(1)

        # Cancel button — always visible during upload (D-20).
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._cancel_button = QPushButton("Cancel upload")
        self._cancel_button.clicked.connect(self.cancel_requested.emit)
        button_row.addWidget(self._cancel_button, 0)
        self._phase_b_layout.addLayout(button_row)

        return page

    # Phase 8 Plan 08-06 Task 3 — _apply_thumbnail_pixmap moved to
    # _PhaseABMixin (revision iter 1 W6 close-out).

    def _on_upload_clicked(self) -> None:
        title = self._title_edit.text()
        desc = self._description_edit.text()
        privacy = self._privacy_combo.currentData() or "private"
        license_val = self._license_combo.currentData() or "creativeCommon"
        self.upload_requested.emit(
            str(title),
            str(desc),
            str(privacy),
            str(license_val),
            bytes(self._current_thumbnail_bytes),
        )

    # ------------------------------------------------------ Qt overrides

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Title-bar X handling per RESEARCH Open Question 5.

        Phase A: close silently (no upload was ever started).
        Phase B: prompt confirm. Only Yes proceeds — emits
        :attr:`cancel_requested` and accepts the close.
        """
        if self._stack.currentIndex() == 1:
            answer = QMessageBox.question(
                self,
                "Cancel upload?",
                "Your video has not been published. Cancel the upload?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel_requested.emit()
        super().closeEvent(event)


__all__ = ["UploadDialog"]
