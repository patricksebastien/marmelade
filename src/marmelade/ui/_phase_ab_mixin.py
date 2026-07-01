"""Shared Phase A/B widget mixin (Plan 08-06 Task 3 / W6).

Extracts set_phase_b / set_progress / show_error / update_thumbnail /
_apply_thumbnail_pixmap from UploadDialog + BundleDialog. Inheriting
class must set: _stack, _progress_bar, _eta_label, _thumbnail_label,
_phase_b_layout, _THUMB_PREVIEW_W / _H (class consts). retry_requested
Signal is optional (Retry click → emit if defined).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QPushButton


class _PhaseABMixin:
    """Shared Phase A (Setup) / Phase B (Progress) helper."""

    _error_label: QLabel | None
    _retry_btn: QPushButton | None

    def set_phase_b(self) -> None:
        """Swap the QStackedWidget to the Phase B (Progress) page."""
        self._stack.setCurrentIndex(1)

    def set_progress(self, pct: int, eta_seconds: float | None) -> None:
        """Update the progress bar + ETA label (None → 'ETA: —')."""
        self._progress_bar.setValue(int(pct))
        if eta_seconds is None:
            self._eta_label.setText("ETA: —")
            return
        secs = int(eta_seconds)
        eta = f"~{secs // 60}m {secs % 60}s" if secs >= 60 else f"~{secs}s"
        self._eta_label.setText(f"ETA: {eta}")

    def show_error(self, message: str, retryable: bool) -> None:
        """Show ``message`` in a destructive-styled footer (+ Retry button)."""
        if self._error_label is None:
            self._error_label = QLabel("")
            self._error_label.setWordWrap(True)
            self._error_label.setStyleSheet(
                "color: #FF6B6B; font-size: 10pt; font-weight: 600;"
            )
            self._phase_b_layout.addWidget(self._error_label)
        if self._retry_btn is None:
            self._retry_btn = QPushButton("Retry")
            # Wire to retry_requested if the dialog defines that Signal.
            sig = getattr(self, "retry_requested", None)
            if sig is not None and hasattr(sig, "emit"):
                self._retry_btn.clicked.connect(sig.emit)
            self._phase_b_layout.addWidget(self._retry_btn)
        self._error_label.setText(message)
        self._error_label.setVisible(True)
        self._retry_btn.setVisible(bool(retryable))

    def update_thumbnail(self, jpeg_bytes: bytes) -> None:
        """Replace the thumbnail QLabel pixmap with ``jpeg_bytes``."""
        self._current_thumbnail_bytes = jpeg_bytes
        self._apply_thumbnail_pixmap(jpeg_bytes)

    def _apply_thumbnail_pixmap(self, jpeg_bytes: bytes) -> None:
        """Render ``jpeg_bytes`` into _thumbnail_label (dark-grey fallback)."""
        thumb_w = getattr(self, "_THUMB_PREVIEW_W", 320)
        thumb_h = getattr(self, "_THUMB_PREVIEW_H", 180)
        pix = QPixmap()
        loaded = pix.loadFromData(jpeg_bytes)
        if not loaded or pix.isNull():
            pix = QPixmap(thumb_w, thumb_h)
            pix.fill(Qt.GlobalColor.darkGray)
        scaled = pix.scaled(
            thumb_w, thumb_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumbnail_label.setPixmap(scaled)


__all__ = ["_PhaseABMixin"]
