"""Plan 02.1-04 — D-14: disk-preflight failure refuses worker spawn.

Pins:

* When ``check_disk_space`` (imported into main_window) returns ``(False, ...)``,
  ``QMessageBox.warning`` is invoked with a "Not enough disk space" title.
* No ``AudioProxyRunnable`` is enqueued — ``_current_proxy_runnable`` stays
  None.
* The spacebar shortcut is left disabled (no proxy means no constant-time
  seek; the user can still inspect the waveform visually).
* The cache file is NOT created (``audio_proxy_is_fresh`` still returns None).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from marmelade.audio.audio_proxy_cache import audio_proxy_is_fresh
from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_disk_preflight_failure_shows_warning_and_skips_worker(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-14: insufficient disk space → friendly dialog, no worker spawn."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "disk_pressure.mp3"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=3.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
    )

    # Force the preflight check to fail by patching the imported reference
    # inside main_window. Return (ok=False, needed, free) where needed > free.
    from marmelade.ui import main_window as mw_module

    fake_check = lambda _root, _expected: (False, 10 * 1024**3, 100 * 1024**2)
    monkeypatch.setattr(mw_module, "check_disk_space", fake_check)

    # Spy on QMessageBox.warning to capture the dialog without showing it.
    captured: list[tuple] = []

    def fake_warning(parent, title, text, *args, **kwargs):
        captured.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(fake_warning))

    window = MainWindow()
    qtbot.addWidget(window)

    window._open_file(str(src))

    # Exactly one warning, with the locked title prefix and the friendly
    # need/free wording.
    assert len(captured) == 1, f"expected 1 QMessageBox.warning, got {len(captured)}"
    title, body = captured[0]
    assert "Not enough disk space" in title
    assert "Need" in body and "Free" in body

    # No worker spawned.
    assert window._current_proxy_runnable is None

    # Spacebar disabled.
    assert window._shortcut_play_pause.isEnabled() is False

    # No cache file written.
    cache_root = default_cache_root()
    assert audio_proxy_is_fresh(cache_root, src) is None
