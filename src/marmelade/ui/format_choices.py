"""Shared audio-format options for the YouTube share dialogs.

Both :class:`marmelade.ui.bundle_dialog.BundleDialog` and
:class:`marmelade.ui.upload_dialog.UploadDialog` show the same
"Format" QComboBox so the user picks the same output set in either
flow. Centralised here so adding a new format is a one-file change.

The actual encoder dispatch (bundle_builder / ffmpeg) consumes the
``userData`` string. Order is most-common-first.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox


# (label, userData) tuples — order is the display order in the combo.
# Lossy first (most common for YouTube share), then lossless.
FORMAT_CHOICES: list[tuple[str, str]] = [
    ("MP3 (320 kbps)", "mp3_320"),
    ("MP3 (256 kbps)", "mp3_256"),
    ("MP3 (192 kbps)", "mp3_192"),
    ("MP3 (128 kbps)", "mp3_128"),
    ("AAC (256 kbps, .m4a)", "aac_256"),
    ("WAV (16-bit PCM)", "wav_16"),
    ("WAV (24-bit PCM)", "wav_24"),
    ("FLAC (lossless)", "flac"),
]


def populate_format_combo(combo: QComboBox, *, default: str = "mp3_320") -> None:
    """Populate ``combo`` with :data:`FORMAT_CHOICES` and pre-select ``default``.

    Args:
        combo: The empty :class:`QComboBox` to populate.
        default: The ``userData`` value to pre-select. Falls back to
            the first entry if ``default`` is not in the list.
    """
    combo.clear()
    for label, key in FORMAT_CHOICES:
        combo.addItem(label, userData=key)
    idx = combo.findData(default)
    combo.setCurrentIndex(idx if idx != -1 else 0)
