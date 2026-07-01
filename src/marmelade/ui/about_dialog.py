"""About Marmelade — modal dialog with attribution + open-source credits.

quick-260626-pbl: a proper About surface reachable from Help → About Marmelade.
Shows the app name, tagline, version, the bundled "Bald Approved" credits image,
the author, a clickable external website link, and the full open-source credits
list (each dependency + its license).

Resource loading mirrors :mod:`marmelade.ui.theme` (``importlib.resources.files``)
so the bundled image resolves correctly whether running from source or a wheel.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import marmelade

# Packaged credits image (read-only resource; mirrors theme.QSS_PATH).
IMAGE_PATH: Path = Path(
    str(files("marmelade.resources").joinpath("baldapprouved.png"))
)

_TAGLINE = (
    "Find and extract the good moments from long, unstructured jam sessions."
)

_OSS_CREDITS = [
    "PySide6 / Qt for Python (LGPL v3)",
    "PyQtGraph (MIT)",
    "NumPy (BSD)",
    "pedalboard — Spotify (GPL v3)",
    "soundfile / libsndfile (BSD)",
    "sounddevice (MIT)",
    "Essentia + essentia-tensorflow (AGPL v3)",
    "TensorFlow & TensorFlow Hub / YAMNet (Apache 2.0)",
    "librosa (ISC)",
    "soxr (LGPL)",
    "Matchering (GPL v3)",
    "pyloudnorm (MIT)",
    "Pillow (HPND)",
    "FFmpeg via imageio-ffmpeg (LGPL/GPL)",
    "google-api-python-client & google-auth-oauthlib (Apache 2.0)",
    "keyring (MIT)",
    "platformdirs (MIT)",
    "xxhash (BSD)",
    "Thanks to the open-source community.",
]


class AboutDialog(QDialog):
    """Modal "About Marmelade" dialog (attribution + OSS credits)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("About Marmelade")
        self.setMinimumSize(460, 700)

        layout = QVBoxLayout(self)

        # (1) App name — bold/larger.
        name_label = QLabel("Marmelade", self)
        name_font = name_label.font()
        name_font.setPointSize(max(name_font.pointSize() + 8, 20))
        name_font.setBold(True)
        name_label.setFont(name_font)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)

        # (2) Tagline.
        tagline_label = QLabel(_TAGLINE, self)
        tagline_label.setWordWrap(True)
        tagline_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tagline_label)

        # (3) Version.
        version_label = QLabel(f"Version {marmelade.__version__}", self)
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        # (4) Bundled image + caption.
        image_label = QLabel(self)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap(str(IMAGE_PATH))
        if not pix.isNull():
            scaled = pix.scaledToWidth(
                275, Qt.TransformationMode.SmoothTransformation
            )
            image_label.setPixmap(scaled)
            # Reserve the full pixmap height so the credits browser's stretch
            # can't squeeze this label and clip the (centered) image top/bottom.
            image_label.setFixedHeight(scaled.height())
        layout.addWidget(image_label)

        caption_label = QLabel("Bald Approved™", self)
        caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(caption_label)

        # (5) Author.
        author_label = QLabel("Created by Patrick Sébastien Coulombe", self)
        author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(author_label)

        # (6) Website link — opens externally.
        self._website_label = QLabel(
            '<a href="https://workinprogress.ca">workinprogress.ca</a>', self
        )
        self._website_label.setTextFormat(Qt.TextFormat.RichText)
        self._website_label.setOpenExternalLinks(True)
        self._website_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._website_label)

        # (7) Open-source credits — scrollable, opens links externally.
        credits = QTextBrowser(self)
        credits.setOpenExternalLinks(True)
        credits.setReadOnly(True)
        credits.setPlainText("\n".join(_OSS_CREDITS))
        layout.addWidget(credits, stretch=1)
        self._credits_browser = credits

        # (8) Close button.
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close, self
        )
        button_box.rejected.connect(self.reject)
        close_btn = button_box.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.reject)
        layout.addWidget(button_box)
