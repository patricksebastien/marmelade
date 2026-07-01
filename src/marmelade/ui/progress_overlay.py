"""Modeless progress overlay shown during a proxy build.

UI-SPEC §Copywriting > Loading:
    Heading: "Preparing waveform"
    Body:    "{filename} · {duration} · first open — building a downsampled
              proxy. This may take up to a minute for an 8-hour file."
    Cancel:  "Stop building proxy"

The overlay is a child widget of the WaveformView (NOT a separate
top-level window) so it scrolls / resizes / hides with the main view. It
uses ``setParent(waveform_view)`` and manual ``setGeometry(0, 0, w, h)``
sizing — see :meth:`resize_to_parent`. The backdrop is the UI-SPEC
secondary surface ``#252526`` at 80 % alpha so the waveform underneath is
faintly visible.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProgressOverlay(QWidget):
    """Modeless overlay with a heading, body, progress bar, and cancel button.

    Public attributes:
        cancel_button: ``QPushButton`` labelled "Stop building proxy" —
            wired by MainWindow to :meth:`PeakBuilderRunnable.cancel`.

    Public methods:
        set_progress(pct: int): Forward the worker's ``progress`` signal
            to the progress bar.
        set_body(text: str): Update the body label (called once after the
            user picks a file so the UI-SPEC body shows filename + duration).
        resize_to_parent(): Re-anchor the overlay to cover the parent
            widget. Called by MainWindow's ``resizeEvent`` hook.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("ProgressOverlay")
        # Auto-fill background so the 80 % alpha tint actually shows.
        self.setAutoFillBackground(True)
        # Reasonable defaults — MainWindow calls resize_to_parent right after
        # construction so the overlay fully covers the WaveformView.
        self.setGeometry(0, 0, parent.width(), parent.height())

        # Outer layout — center the inner card.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(48, 48, 48, 48)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame(self)
        card.setObjectName("ProgressOverlayCard")
        card.setFrameShape(QFrame.Shape.NoFrame)
        card.setMaximumWidth(560)
        card.setMinimumWidth(360)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(12)

        # Heading — UI-SPEC §Copywriting > Loading.
        self._heading = QLabel("Preparing waveform", card)
        self._heading.setStyleSheet(
            "font-size: 14pt; font-weight: 600; color: #E6E6E6;"
        )
        card_layout.addWidget(self._heading)

        # Body — populated by set_body() with the per-file substitution.
        self._body = QLabel("", card)
        self._body.setWordWrap(True)
        self._body.setStyleSheet(
            "font-size: 10pt; font-weight: 400; color: #9CA3AF;"
        )
        card_layout.addWidget(self._body)

        # Determinate 0..100 progress bar.
        self._progress_bar = QProgressBar(card)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        card_layout.addWidget(self._progress_bar)

        # Cancel button — UI-SPEC label verbatim.
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Stop building proxy", card)
        button_row.addWidget(self.cancel_button)
        card_layout.addLayout(button_row)

        outer.addWidget(card)

        # 80 % alpha secondary-surface backdrop covers the whole overlay
        # rectangle; the inner card sits at full opacity.
        self.setStyleSheet(
            "#ProgressOverlay { background-color: rgba(37, 37, 38, 200); }"
            "#ProgressOverlayCard { background-color: #252526; "
            "border: 1px solid #2F2F33; border-radius: 6px; }"
        )

    # ------------------------------------------------------------- public API
    def set_progress(self, pct: int) -> None:
        """Forward worker progress to the progress bar (0..100)."""
        self._progress_bar.setValue(int(pct))

    def set_body(self, text: str) -> None:
        """Populate the per-file body label (filename · duration · …)."""
        self._body.setText(text)

    def set_heading(self, text: str) -> None:
        """Override the heading text (default "Preparing waveform").

        Phase 2.1 HUMAN-UAT request #3 — MainWindow re-uses this overlay
        for the audio proxy build and needs to swap the heading so the
        user can tell the two builds apart ("Preparing audio proxy" vs.
        "Preparing waveform").
        """
        self._heading.setText(text)

    def resize_to_parent(self) -> None:
        """Resize this overlay to fully cover its parent widget."""
        p = self.parentWidget()
        if p is not None:
            self.setGeometry(0, 0, p.width(), p.height())
