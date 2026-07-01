"""Modal BundleDialog — Phase A (Setup) + Phase B (Progress) for the bundle Share path.

Phase 8 Plan 08-05 Task 3 — implements D-03 (single thumbnail for the
whole bundle, NOT a per-keeper sequence) + D-04 (user-configurable
silent spacer QDoubleSpinBox, default 2.0 s, range 0..10 s) + D-05
fallback path (``use_fallback_reorder=True`` flips the keepers list to
``QAbstractItemView.InternalMove`` per RESEARCH Pattern 5) + D-06
(Save-to file picker; no QSettings persistence of the save path) +
D-23 (title field always user-editable).

Composition over inheritance — copies :class:`marmelade.ui.
upload_dialog.UploadDialog`'s Phase A/B widget vocabulary verbatim
rather than subclassing. The two dialogs share enough widget shape
that a ``_PhaseAB`` mixin extraction is a sensible follow-up; that
extraction is Plan 08-06 Task 3 (added per revision iter 1 W6 — see
the plan's <output> note). For Plan 08-05 we ship duplication and let
the polish plan close the loop.

The dialog NEVER spawns the build / upload itself — it emits signals
and :class:`MainWindow` orchestrates the
:class:`marmelade.audio.bundle_builder.build_bundle` call,
:func:`marmelade.youtube.video_builder.build_video` follow-up, and
:class:`marmelade.youtube.upload_runnable.YouTubeUploadRunnable`
spawn + retry/restart cycle. This separation keeps the dialog
testable without a real upload queue.

D-04 spacer-seconds QDoubleSpinBox details:
    * range [0.0, 10.0]
    * decimals = 1 (so the user enters 1.5 etc.)
    * single-step = 0.5
    * suffix = " s"
    * initial value: ``initial_spacer_sec`` arg (defaults to 2.0); the
      MainWindow caller reads QSettings ``youtube/bundle_spacer_sec``
      with explicit ``float()`` coercion per RESEARCH Pitfall 4 before
      passing it in.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from marmelade.ui._phase_ab_mixin import _PhaseABMixin


_THUMB_PREVIEW_W: int = 320
_THUMB_PREVIEW_H: int = 180


class BundleDialog(_PhaseABMixin, QDialog):
    """Modal bundle upload + export dialog.

    Args:
        keepers: ``[(region_id, display_label), ...]`` in current user-
            arranged order. The dialog renders these in a
            :class:`QListWidget` so the user can verify the order before
            committing. In default mode the list is read-only; in
            fallback mode (``use_fallback_reorder=True``) it becomes
            drag-and-droppable via ``InternalMove``.
        initial_title: Pre-filled title text (typically pseudo-poem from
            :func:`marmelade.util.poem_generator.generate`). User-
            editable per D-23.
        initial_description: Pre-filled description text.
        initial_privacy: One of ``"private"``, ``"unlisted"``,
            ``"public"`` — read from QSettings per D-21.
        initial_thumbnail_bytes: JPEG bytes for the thumbnail preview.
        initial_spacer_sec: D-04 default 2.0 s. Caller is responsible
            for clamping to [0, 10] before passing in (the spinbox
            enforces the same range on the user side).
        use_fallback_reorder: RESEARCH Pattern 5 escape hatch — when
            True the keepers QListWidget becomes drag-and-droppable
            (InternalMove) and an "Apply Order" button appears that
            emits :attr:`reorder_requested` with the rearranged ids.
        parent: Optional parent widget (typically MainWindow).

    Signals:
        export_mp3_only_requested(save_path, spacer_sec, ordered_ids):
            emitted on Export MP3 only click. ``save_path`` is an
            empty string — MainWindow is responsible for the
            ``QFileDialog.getSaveFileName`` call (this keeps the
            dialog filesystem-free for testing).
        export_and_upload_requested(save_path, spacer_sec, ordered_ids,
            title, description, privacy, license, jpeg_bytes): emitted on
            Export + Upload click. ``license`` is one of YouTube's two
            accepted values: ``"creativeCommon"`` (CC-BY, default) or
            ``"youtube"`` (Standard YouTube License).
        cancel_requested(): emitted on the Phase B Cancel button.
        refresh_thumbnail_requested(): emitted on the Refresh
            thumbnail button in Phase A.
        reorder_requested(ordered_region_ids): emitted on the fallback
            mode's Apply Order button — MainWindow forwards the new
            order back to the KeepersSidebar.
    """

    # Class-level thumb preview constants — read by _PhaseABMixin's
    # _apply_thumbnail_pixmap so the mixin does not need to know the
    # dialog-specific sizes.
    _THUMB_PREVIEW_W: int = _THUMB_PREVIEW_W
    _THUMB_PREVIEW_H: int = _THUMB_PREVIEW_H

    export_mp3_only_requested = Signal(str, float, list)
    export_and_upload_requested = Signal(str, float, list, str, str, str, str, bytes)
    cancel_requested = Signal()
    refresh_thumbnail_requested = Signal()
    reorder_requested = Signal(list)

    def __init__(
        self,
        *,
        keepers: list[tuple[str, str]],
        initial_title: str,
        initial_description: str,
        initial_privacy: str,
        initial_thumbnail_bytes: bytes,
        initial_spacer_sec: float = 2.0,
        initial_license: str = "creativeCommon",
        use_fallback_reorder: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Share bundle ({len(keepers)} keepers) to YouTube")
        self.setModal(True)
        self.resize(620, 700)

        self._keepers: list[tuple[str, str]] = list(keepers)
        self._initial_thumbnail_bytes: bytes = initial_thumbnail_bytes
        self._current_thumbnail_bytes: bytes = initial_thumbnail_bytes
        self._use_fallback_reorder: bool = bool(use_fallback_reorder)

        # Error-footer widgets are created lazily by :meth:`show_error`.
        self._error_label: QLabel | None = None
        self._retry_btn: QPushButton | None = None

        self._build_ui(
            initial_title=initial_title,
            initial_description=initial_description,
            initial_privacy=initial_privacy,
            initial_spacer_sec=initial_spacer_sec,
            initial_license=initial_license,
        )

    # ----------------------------------------------------------- public API
    #
    # Phase 8 Plan 08-06 Task 3 — set_phase_b / set_progress / show_error /
    # update_thumbnail are provided by ``_PhaseABMixin``. The methods
    # used to live in this class verbatim; they were extracted into
    # the mixin so UploadDialog can share one implementation (revision
    # iter 1 W6 close-out).

    def get_ordered_region_ids(self) -> list[str]:
        """Return the current keeper order as seen in the dialog.

        In default mode this matches the input order (the list is
        read-only). In fallback mode the user may have reordered via
        ``InternalMove``; this method walks the QListWidget rows in
        index order and returns the ids the user stored in each row's
        UserRole data.
        """
        out: list[str] = []
        for i in range(self._keepers_list.count()):
            item = self._keepers_list.item(i)
            if item is None:
                continue
            rid = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(rid, str):
                out.append(rid)
        return out

    # ----------------------------------------------------------- internal

    def _build_ui(
        self,
        *,
        initial_title: str,
        initial_description: str,
        initial_privacy: str,
        initial_spacer_sec: float,
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
            initial_spacer_sec=initial_spacer_sec,
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
        initial_spacer_sec: float,
        initial_license: str,
    ) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # --- Keepers order list (D-05) ---
        order_label = QLabel("Bundle order (drag the handle on each keeper to reorder):")
        order_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(order_label)

        self._keepers_list = QListWidget()
        self._keepers_list.setMaximumHeight(120)
        for rid, label in self._keepers:
            # Add display label, stash region_id in UserRole so
            # get_ordered_region_ids can read it back.
            from PySide6.QtWidgets import QListWidgetItem

            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, rid)
            self._keepers_list.addItem(item)

        if self._use_fallback_reorder:
            # RESEARCH Pattern 5 — InternalMove + Apply Order button.
            self._keepers_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self._keepers_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        else:
            # Default mode: read-only. Rows reorder via the in-sidebar
            # drag-handle in keepers_sidebar.py.
            self._keepers_list.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
            self._keepers_list.setSelectionMode(
                QAbstractItemView.SelectionMode.NoSelection
            )
            self._keepers_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self._keepers_list)

        # Apply Order button — only meaningful in fallback mode.
        self._apply_order_btn = QPushButton("Apply order")
        self._apply_order_btn.setVisible(self._use_fallback_reorder)
        self._apply_order_btn.clicked.connect(
            lambda: self.reorder_requested.emit(self.get_ordered_region_ids())
        )
        layout.addWidget(self._apply_order_btn)

        # --- Spacer spinbox (D-04) ---
        spacer_row = QHBoxLayout()
        spacer_lbl = QLabel("Silence between keepers:")
        spacer_lbl.setStyleSheet("font-weight: 600;")
        spacer_row.addWidget(spacer_lbl)
        self._spacer_spinbox = QDoubleSpinBox()
        self._spacer_spinbox.setRange(0.0, 10.0)
        self._spacer_spinbox.setDecimals(1)
        self._spacer_spinbox.setSingleStep(0.5)
        self._spacer_spinbox.setSuffix(" s")
        self._spacer_spinbox.setValue(float(initial_spacer_sec))
        spacer_row.addWidget(self._spacer_spinbox)
        spacer_row.addStretch(1)
        layout.addLayout(spacer_row)

        # --- Thumbnail preview (D-03) ---
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

        self._refresh_btn = QPushButton("Refresh thumbnail")
        self._refresh_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._refresh_btn.clicked.connect(self.refresh_thumbnail_requested.emit)
        refresh_row = QHBoxLayout()
        refresh_row.addStretch(1)
        refresh_row.addWidget(self._refresh_btn, 0)
        refresh_row.addStretch(1)
        layout.addLayout(refresh_row)

        layout.addSpacing(8)

        # --- Title (D-23 user-editable) ---
        title_label = QLabel("Title")
        title_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(title_label)
        self._title_edit = QLineEdit(initial_title)
        self._title_edit.setMaxLength(100)
        layout.addWidget(self._title_edit)

        # --- Description ---
        desc_label = QLabel("Description")
        desc_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(desc_label)
        self._description_edit = QLineEdit(initial_description)
        self._description_edit.setMaxLength(5000)
        layout.addWidget(self._description_edit)

        # --- Privacy combo ---
        privacy_label = QLabel("Privacy")
        privacy_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(privacy_label)
        self._privacy_combo = QComboBox()
        self._privacy_combo.addItem("Private — only you", userData="private")
        self._privacy_combo.addItem("Unlisted — anyone with the link", userData="unlisted")
        self._privacy_combo.addItem("Public — anyone can find it", userData="public")
        idx = self._privacy_combo.findData(initial_privacy)
        if idx == -1:
            idx = self._privacy_combo.findData("private")
        self._privacy_combo.setCurrentIndex(idx if idx != -1 else 0)
        layout.addWidget(self._privacy_combo)

        # --- License combo ---
        # Maps directly to YouTube's status.license field. Only two
        # values are accepted by the API: creativeCommon / youtube.
        # Grouped with the output-decision controls, directly after
        # Privacy (mirrors UploadDialog).
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

        # --- Format combo (common audio formats) ---
        # Placed below Privacy so the dialog reads as a stack of
        # output-decision controls (Spacer → Title → Description →
        # Privacy → Format). Driven by the shared
        # :func:`marmelade.ui.format_choices.populate_format_combo`
        # helper so UploadDialog ships an identical list.
        format_label = QLabel("Format")
        format_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(format_label)
        self._format_combo = QComboBox()
        from marmelade.ui.format_choices import populate_format_combo
        populate_format_combo(self._format_combo, default="mp3_320")
        layout.addWidget(self._format_combo)

        layout.addStretch(1)

        # --- Bottom button row: two action buttons + Cancel (D-06) ---
        # D-06: file picker prompt happens in MainWindow's slot, NOT here.
        # The dialog emits save_path="" and the slot opens
        # QFileDialog.getSaveFileName.
        self._export_only_btn = QPushButton("Export MP3")
        self._export_only_btn.clicked.connect(self._on_export_only_clicked)
        self._export_and_upload_btn = QPushButton("Upload to YouTube")
        self._export_and_upload_btn.setDefault(True)
        self._export_and_upload_btn.clicked.connect(self._on_export_and_upload_clicked)
        self._setup_cancel_btn = QPushButton("Cancel")
        self._setup_cancel_btn.clicked.connect(self.reject)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(self._setup_cancel_btn, 0)
        bottom.addWidget(self._export_only_btn, 0)
        bottom.addWidget(self._export_and_upload_btn, 0)
        layout.addLayout(bottom)

        return page

    def _build_phase_b(self) -> QWidget:
        page = QWidget()
        self._phase_b_layout = QVBoxLayout(page)
        self._phase_b_layout.setContentsMargins(0, 0, 0, 0)
        self._phase_b_layout.setSpacing(8)

        heading = QLabel("Building bundle + uploading to YouTube…")
        heading.setStyleSheet("font-size: 12pt; font-weight: 600;")
        self._phase_b_layout.addWidget(heading)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._phase_b_layout.addWidget(self._progress_bar)

        self._eta_label = QLabel("ETA: —")
        self._eta_label.setStyleSheet("color: #9CA3AF; font-size: 10pt;")
        self._phase_b_layout.addWidget(self._eta_label)

        self._phase_b_layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.cancel_requested.emit)
        button_row.addWidget(self._cancel_button, 0)
        self._phase_b_layout.addLayout(button_row)

        return page

    # Phase 8 Plan 08-06 Task 3 — _apply_thumbnail_pixmap moved to
    # _PhaseABMixin (revision iter 1 W6 close-out).

    def _on_export_only_clicked(self) -> None:
        """Export MP3 only — emit signal with empty save_path; slot opens picker."""
        ordered_ids = self.get_ordered_region_ids()
        spacer_sec = float(self._spacer_spinbox.value())
        # save_path = "" sentinel: MainWindow slot is responsible for
        # the QFileDialog.getSaveFileName prompt (D-06 — no QSettings
        # persistence of the path, no auto-naming).
        self.export_mp3_only_requested.emit("", spacer_sec, ordered_ids)

    def _on_export_and_upload_clicked(self) -> None:
        """Export MP3 + Upload — emit signal with full snippet/status payload."""
        ordered_ids = self.get_ordered_region_ids()
        spacer_sec = float(self._spacer_spinbox.value())
        title = self._title_edit.text()
        desc = self._description_edit.text()
        privacy = self._privacy_combo.currentData() or "private"
        license_val = self._license_combo.currentData() or "creativeCommon"
        self.export_and_upload_requested.emit(
            "",
            spacer_sec,
            ordered_ids,
            str(title),
            str(desc),
            str(privacy),
            str(license_val),
            bytes(self._current_thumbnail_bytes),
        )

    # ------------------------------------------------------ Qt overrides

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Title-bar X handling — same vocabulary as UploadDialog.

        Phase A: close silently (no work started). Phase B: confirm
        prompt; only Yes proceeds and emits :attr:`cancel_requested`.
        """
        if self._stack.currentIndex() == 1:
            answer = QMessageBox.question(
                self,
                "Cancel bundle?",
                "Your bundle has not been published. Cancel the bundle build/upload?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.cancel_requested.emit()
        super().closeEvent(event)


__all__ = ["BundleDialog"]
