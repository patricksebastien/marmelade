"""Integration: AUD-02 — re-opening the same file reuses the cached proxy.

The test patches :class:`PeakBuilderRunnable.__init__` to detect spawn
attempts. First open is a cache MISS that builds the proxy normally; second
open of the same path must skip the worker entirely.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from marmelade.audio.peak_builder_worker import PeakBuilderRunnable
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


@pytest.fixture
def main_window(qtbot, qapp, tmp_cache_dir: Path):
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


def test_reopen_uses_cache(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """First open builds, second open of the same path skips the worker."""
    wav = tmp_path / "fixture.wav"
    make_sine(wav, freq_hz=1000.0, amp=0.5, duration_s=3.0, sample_rate=44100, channels=1)

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(wav), "Audio files (*.wav *.flac *.mp3)")),
    )

    # First open — cache MISS. We allow the runnable to construct and run.
    with qtbot.waitSignal(
        main_window.render_complete, timeout=15000, raising=True
    ):
        main_window._action_open_file()

    # Close the file so the second open is a fresh open flow.
    main_window._close_file()

    # Second open — patch PeakBuilderRunnable.__init__ to track calls.
    original_init = PeakBuilderRunnable.__init__
    call_count = {"n": 0}

    def tracking_init(self, *args, **kwargs):
        call_count["n"] += 1
        original_init(self, *args, **kwargs)

    with patch.object(PeakBuilderRunnable, "__init__", tracking_init):
        with qtbot.waitSignal(
            main_window.render_complete, timeout=5000, raising=True
        ):
            main_window._action_open_file()

    # Cache HIT → no runnable instantiated.
    assert call_count["n"] == 0, "Expected cache HIT (no worker), got fresh build"
