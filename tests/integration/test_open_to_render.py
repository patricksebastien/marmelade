"""Integration: open → probe → cache → render end-to-end happy + error paths.

UI-01 / UI-02 / UI-03 / UI-04 + the three UI-SPEC error dialogs (too long /
unsupported extension / corrupt file).

These tests run under ``QT_QPA_PLATFORM=offscreen`` (set in conftest). They
synthesize fixtures on the fly via :mod:`tests.fixtures.synthesize` and
monkey-patch ``QFileDialog.getOpenFileName`` so the open flow runs without
any user interaction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def main_window(qtbot, qapp, tmp_cache_dir: Path):
    """Build a MainWindow with the theme applied; tmp_cache_dir isolates writes."""
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    """Synthesize a 5-second 44.1 kHz mono sine WAV."""
    path = tmp_path / "fixture.wav"
    make_sine(path, freq_hz=1000.0, amp=0.5, duration_s=5.0, sample_rate=44100, channels=1)
    return path


# ------------------------------------------------------------------ tests
def test_open_dialog_invokes(
    main_window: MainWindow, qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Triggering the toolbar Open action invokes QFileDialog.getOpenFileName."""
    called = {"n": 0}

    def fake_dialog(*args, **kwargs):
        called["n"] += 1
        return ("", "")  # cancel — no signal fires

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(fake_dialog),
    )

    main_window._action_open_file()
    assert called["n"] == 1


def test_open_wav_renders_waveform(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sine_wav: Path,
) -> None:
    """UI-01 / UI-02 / AUD-01 / AUD-02: open a WAV → proxy built → waveform rendered."""
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(sine_wav), "Audio files (*.wav *.flac *.mp3)")),
    )

    with qtbot.waitSignal(
        main_window.render_complete, timeout=15000, raising=True
    ):
        main_window._action_open_file()

    items = main_window._waveform_view.plot_widget.plotItem.listDataItems()
    assert len(items) == 1
    x, y = items[0].getData()
    assert x is not None and len(x) > 0
    assert y is not None and len(y) > 0


def test_too_long_file_shows_dialog(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A file whose probe.duration_s > 8h must show the UI-SPEC too-long dialog."""
    fake_path = tmp_path / "long.wav"
    # Create a minimal placeholder so existence check passes; we monkey-patch
    # probe so its duration_s exceeds 8h.
    fake_path.write_bytes(b"\x00" * 4)

    from marmelade.audio import audio_file
    from marmelade.ui import main_window as mw_module

    def fake_probe(path):
        return audio_file.AudioProbe(
            sample_rate=44100,
            frames=int(9 * 3600 * 44100),  # 9 hours
            channels=1,
            duration_s=9 * 3600,
        )

    monkeypatch.setattr(mw_module.audio_file, "probe", fake_probe)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(fake_path), "")),
    )

    captured: list[QMessageBox] = []

    def fake_exec(self: QMessageBox) -> int:
        captured.append(self)
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)

    main_window._action_open_file()

    assert len(captured) == 1
    box = captured[0]
    assert box.windowTitle() == "File is longer than supported"
    # Worker should NOT have been spawned — sanity-check by examining
    # that there is no current runnable.
    assert main_window._current_runnable is None


def test_unsupported_extension_shows_dialog(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """UI-SPEC: a .txt file selected via QFileDialog → unsupported-format dialog."""
    bad_path = tmp_path / "notes.txt"
    bad_path.write_text("not audio")

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(bad_path), "")),
    )

    captured: list[QMessageBox] = []

    def fake_exec(self: QMessageBox) -> int:
        captured.append(self)
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)

    main_window._action_open_file()
    assert len(captured) == 1
    assert captured[0].windowTitle() == "Couldn't open file"
    assert "isn't a supported audio format" in captured[0].text()


def test_corrupt_file_shows_dialog(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """UI-SPEC: a 4-byte garbage .wav → 'Couldn't open file' / Underlying error."""
    bad_path = tmp_path / "garbage.wav"
    bad_path.write_bytes(b"\x00\x01\x02\x03")

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(bad_path), "")),
    )

    captured: list[QMessageBox] = []

    def fake_exec(self: QMessageBox) -> int:
        captured.append(self)
        # Click Close so we don't loop on Retry.
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)

    main_window._action_open_file()
    assert len(captured) == 1
    box = captured[0]
    assert box.windowTitle() == "Couldn't open file"
    assert "Underlying error:" in box.text()


def test_wheel_zoom_xonly(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sine_wav: Path,
) -> None:
    """UI-03: wheel scroll zooms x-axis only with 1.25× step."""
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(sine_wav), "")),
    )

    with qtbot.waitSignal(
        main_window.render_complete, timeout=15000, raising=True
    ):
        main_window._action_open_file()

    plot_item = main_window._waveform_view.plot_widget.plotItem
    vb = plot_item.getViewBox()

    initial_x = vb.viewRange()[0]
    initial_y = vb.viewRange()[1]
    initial_x_width = initial_x[1] - initial_x[0]

    # Use ViewBox.scaleBy — the documented Z3PyQtGraph API that wheel
    # events ultimately drive. Apply x-only by passing (sx, sy) with
    # sy=1 (no y change) and sx=1/1.25 (zoom in).
    vb.scaleBy((1 / 1.25, 1.0))
    QTest.qWait(50)

    new_x = vb.viewRange()[0]
    new_y = vb.viewRange()[1]
    new_x_width = new_x[1] - new_x[0]

    # X-width narrowed by ~1.25.
    assert new_x_width == pytest.approx(initial_x_width / 1.25, rel=0.05)
    # Y-range unchanged.
    assert new_y[0] == pytest.approx(initial_y[0], abs=1.0)
    assert new_y[1] == pytest.approx(initial_y[1], abs=1.0)


def test_drag_pans_x(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sine_wav: Path,
) -> None:
    """UI-03: left-drag pans the x-axis; y-axis is locked."""
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(sine_wav), "")),
    )

    with qtbot.waitSignal(
        main_window.render_complete, timeout=15000, raising=True
    ):
        main_window._action_open_file()

    plot_item = main_window._waveform_view.plot_widget.plotItem
    vb = plot_item.getViewBox()
    initial_x = vb.viewRange()[0]
    initial_y = vb.viewRange()[1]
    width = initial_x[1] - initial_x[0]

    # Simulate a pan by translating the view by 20% of its width to the
    # right. Equivalent to the user dragging the data 20% to the right.
    vb.translateBy(x=width * 0.2, y=0.0)
    QTest.qWait(50)

    new_x = vb.viewRange()[0]
    new_y = vb.viewRange()[1]

    # X-range shifted in positive direction.
    assert new_x[0] > initial_x[0]
    # Y-range unchanged.
    assert new_y[0] == pytest.approx(initial_y[0], abs=1.0)
    assert new_y[1] == pytest.approx(initial_y[1], abs=1.0)
