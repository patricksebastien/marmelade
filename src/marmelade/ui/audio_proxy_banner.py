"""Compact in-band progress banner for the audio-proxy build.

Phase 2.1 HUMAN-UAT request #3 (final form). The full-screen
:class:`ProgressOverlay` is correctly built and ``isVisible()`` returns
True, but on at least some Linux + Qt + PyQtGraph compositor stacks the
overlay does not actually paint over the ``pg.GraphicsLayoutWidget``
that lives inside the WaveformView. Rather than fight the compositor,
we render the audio-proxy progress as a small horizontal banner pinned
to the top-center of the WaveformView — the waveform stays visible
underneath, click-to-seek is gated off in ``MainWindow._on_seek_requested``,
and the banner is a plain ``QFrame`` (no fancy backgrounds, no
``setAutoFillBackground``-vs-stylesheet conflict) so its painting is
robust.

Visual shape:

    ┌─────────────────────────────────────────────────┐
    │ Preparing audio proxy                           │
    │ <basename> · <duration> · click-to-play locked  │
    │ [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░] 67%   [Stop building] │
    └─────────────────────────────────────────────────┘

Sized ~600 × 100, centered horizontally near the top of the waveform.
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


class AudioProxyProgressBanner(QFrame):
    """Compact progress banner shown during the audio-proxy build.

    Public attributes
    -----------------
    cancel_button : QPushButton
        Wired by MainWindow to the active ``AudioProxyRunnable.cancel``.
        The label reads "Stop building proxy" to match the existing
        :class:`~marmelade.ui.progress_overlay.ProgressOverlay`
        copywriting.

    Public methods
    --------------
    set_progress(pct: int)
        Forward the worker's ``progress`` signal (0..100) to the bar.
    set_body(text: str)
        Update the per-file body line (filename · duration · note).
    position_over_parent()
        Center the banner horizontally near the top of the parent
        widget. MainWindow calls this on show and on parent resize.
    """

    BANNER_WIDTH = 560
    BANNER_HEIGHT = 110
    TOP_MARGIN_PX = 16

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("AudioProxyProgressBanner")
        # Solid background (no rgba/stylesheet-on-stylesheet) — keeps
        # painting predictable on every Qt/X11 compositor combination.
        self.setAutoFillBackground(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "#AudioProxyProgressBanner { "
            "background-color: #252526; "
            "border: 1px solid #2F2F33; "
            "border-radius: 6px; "
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        self._heading = QLabel("Preparing audio proxy", self)
        self._heading.setStyleSheet(
            "font-size: 11pt; font-weight: 600; color: #E6E6E6; "
            "background: transparent;"
        )
        layout.addWidget(self._heading)

        self._body = QLabel("", self)
        self._body.setWordWrap(True)
        self._body.setStyleSheet(
            "font-size: 9pt; font-weight: 400; color: #9CA3AF; "
            "background: transparent;"
        )
        layout.addWidget(self._body)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(
            "QProgressBar { "
            "background-color: #1E1E1E; "
            "border: 1px solid #2F2F33; "
            "border-radius: 3px; "
            "text-align: center; "
            "color: #E6E6E6; "
            "} "
            "QProgressBar::chunk { "
            "background-color: #4A9EFF; "
            "border-radius: 2px; "
            "}"
        )
        bottom_row.addWidget(self._progress_bar, stretch=1)

        self.cancel_button = QPushButton("Stop building proxy", self)
        self.cancel_button.setStyleSheet(
            "QPushButton { "
            "background-color: #2F2F33; "
            "color: #E6E6E6; "
            "border: 1px solid #3A3A3F; "
            "border-radius: 3px; "
            "padding: 4px 10px; "
            "} "
            "QPushButton:hover { background-color: #3A3A3F; }"
        )
        bottom_row.addWidget(self.cancel_button)

        layout.addLayout(bottom_row)

        # Initial geometry — MainWindow calls position_over_parent() right
        # after construction so the banner sits at top-center.
        self.setFixedSize(self.BANNER_WIDTH, self.BANNER_HEIGHT)

    # ------------------------------------------------------------- public API
    def set_progress(self, pct: int) -> None:
        """Forward worker progress to the progress bar (0..100)."""
        self._progress_bar.setValue(int(pct))

    def set_body(self, text: str) -> None:
        """Populate the per-file body line."""
        self._body.setText(text)

    def set_heading(self, text: str) -> None:
        """Override the heading (defaults to 'Preparing audio proxy')."""
        self._heading.setText(text)

    def configure_building(self, heading: str, body: str) -> None:
        """Switch the banner to BUILDING state (progress bar + Stop button).

        Default mode set by MainWindow on audio-proxy MISS spawn.
        """
        self._heading.setText(heading)
        self._body.setText(body)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self.cancel_button.setText("Stop building proxy")
        self.cancel_button.show()

    def configure_unavailable(self, heading: str, body: str) -> None:
        """Switch the banner to NOT-BUILT state (no bar, Build button).

        Used after the user cancels the build (or the build errors) —
        playback is unavailable until the proxy is rebuilt. The
        ``cancel_button`` is reused as the action button with label
        "Build proxy"; MainWindow rewires its ``clicked`` signal to
        re-spawn the worker.
        """
        self._heading.setText(heading)
        self._body.setText(body)
        self._progress_bar.hide()
        self.cancel_button.setText("Build proxy")
        self.cancel_button.show()

    def position_over_parent(self) -> None:
        """Center horizontally near the top of the parent widget.

        Called by MainWindow on show and on parent resize so the banner
        stays anchored when the user resizes the window.
        """
        parent = self.parentWidget()
        if parent is None:
            return
        x = max(0, (parent.width() - self.BANNER_WIDTH) // 2)
        y = self.TOP_MARGIN_PX
        self.move(x, y)

    def position_over_widget(self, anchor: QWidget) -> None:
        """Center horizontally near the top of an arbitrary widget.

        Used when the banner is a child of MainWindow (not the anchor
        widget) — necessary because QWidget children of a
        ``pg.GraphicsLayoutWidget``-containing parent can fail to paint
        on some Linux+Qt+PyQtGraph compositor stacks (the original
        symptom the banner was built to dodge in the first place).
        """
        parent = self.parentWidget()
        if parent is None or anchor is None:
            return
        # Top-left of the anchor expressed in the banner's parent coords.
        top_left_in_anchor = anchor.rect().topLeft()
        top_left_in_parent = anchor.mapTo(parent, top_left_in_anchor)
        x = top_left_in_parent.x() + max(
            0, (anchor.width() - self.BANNER_WIDTH) // 2
        )
        y = top_left_in_parent.y() + self.TOP_MARGIN_PX
        self.move(x, y)
