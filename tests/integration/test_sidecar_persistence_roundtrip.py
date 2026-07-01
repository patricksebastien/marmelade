"""Plan 03-01 Task 3 — end-to-end sidecar persistence roundtrip.

Four pins covering MainWindow's sidecar load on file-open + atomic save on
every region mutation (REG-04):

* Shift+drag → wait for ``regions_changed`` → the sidecar JSON exists on
  disk and contains one region with the drawn range.
* Open file → Shift+drag → close → re-open the same file → the region is
  restored on the timeline.
* Two regions persist with their distinct ranges in the JSON ``regions``
  list (Plan 01 doesn't enforce chronological order — Plan 02 does).
* The on-disk path is ``default_cache_root() / 'sidecars' / f'{cache_key}.json'``.

Test discipline mirrors :mod:`tests.integration.test_reopen_uses_cache` —
QFileDialog patched, qtbot.waitSignal on render_complete +
audio_proxy_complete + regions_changed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog

from marmelade.audio.sidecar_cache import cache_key, sidecar_path
from marmelade.paths import default_cache_root  # noqa: F401 — patched by tmp_cache_dir
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


@pytest.fixture
def main_window(qtbot, qapp, tmp_cache_dir: Path):
    """Construct MainWindow + size it wide enough for both side docks.

    Plan 03-02 adds a right-side Keepers dock (min width 280 px) next to
    the existing left Layers dock (min width 160 px). At a 1000 px window
    width the central WaveformView shrinks to ~560 px, which collapses
    the data-x mapping enough that this file's x-pixel-based ``_shift_drag``
    coordinates land on top of each other. Resizing the window to 1600 px
    keeps the central viewport at ~1160 px so the legacy x-coords (100,
    200, 400, 600, …) still map to distinct, in-bounds data seconds.
    """
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.resize(1600, 600)
    QApplication.processEvents()
    return window


def _send_press(view, x: int, modifier=Qt.KeyboardModifier.ShiftModifier, y: int = 30) -> None:
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        viewport.mapToGlobal(QPoint(x, y)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        modifier,
    )
    QApplication.sendEvent(viewport, press)


def _send_move(view, x: int, modifier=Qt.KeyboardModifier.ShiftModifier, y: int = 30) -> None:
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        pos,
        viewport.mapToGlobal(QPoint(x, y)).toPointF(),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        modifier,
    )
    QApplication.sendEvent(viewport, move)


def _send_release(view, x: int, modifier=Qt.KeyboardModifier.ShiftModifier, y: int = 30) -> None:
    viewport = view.graphics_layout.viewport()
    pos = QPointF(float(x), float(y))
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        pos,
        viewport.mapToGlobal(QPoint(x, y)).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        modifier,
    )
    QApplication.sendEvent(viewport, release)


def _shift_drag(view, x_start: int, x_end: int) -> None:
    """Shift+drag from x_start to x_end through one intermediate move."""
    midpoint = (x_start + x_end) // 2
    _send_press(view, x=x_start)
    _send_move(view, x=midpoint)
    _send_release(view, x=x_end)


def _open_via_action(window: MainWindow, wav: Path, monkeypatch, qtbot) -> None:
    """Open ``wav`` via _action_open_file with QFileDialog patched to return wav."""
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(wav), "Audio files (*.wav *.flac *.mp3)")),
    )
    with qtbot.waitSignal(window.render_complete, timeout=15000, raising=True):
        window._action_open_file()
    QApplication.processEvents()


def test_shift_drag_writes_sidecar(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_cache_dir: Path,
) -> None:
    """Shift+drag → wait for regions_changed → sidecar JSON exists with one region."""
    wav = tmp_path / "fixture.wav"
    make_sine(
        wav,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=5.0,
        sample_rate=44100,
        channels=1,
    )
    _open_via_action(main_window, wav, monkeypatch, qtbot)

    # Sanity: sidecar does not yet exist.
    key = cache_key(wav)
    sp = sidecar_path(tmp_cache_dir, key)
    assert not sp.exists()

    # Shift+drag and wait for the atomic-save to land.
    view = main_window._waveform_view
    with qtbot.waitSignal(main_window.regions_changed, timeout=2000) as blocker:
        _shift_drag(view, x_start=200, x_end=400)

    # The signal payload is the sidecar path as a str.
    assert blocker.args[0] == str(sp)
    assert sp.exists(), f"sidecar should exist at {sp}"
    # Parse — exactly one region with a positive duration.
    data = json.loads(sp.read_text())
    assert data["schema_version"] == 1
    assert len(data["regions"]) == 1
    r = data["regions"][0]
    assert r["state"] == "untouched"
    assert r["start_sec"] < r["end_sec"]
    assert re.fullmatch(r"[0-9a-f]{32}", r["id"])


def test_close_and_reopen_restores_regions(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_cache_dir: Path,
) -> None:
    """Open → Shift+drag → close → re-open → the region is on the timeline."""
    wav = tmp_path / "fixture.wav"
    make_sine(
        wav,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=5.0,
        sample_rate=44100,
        channels=1,
    )
    _open_via_action(main_window, wav, monkeypatch, qtbot)
    view = main_window._waveform_view

    # Draw a region and wait for the save.
    with qtbot.waitSignal(main_window.regions_changed, timeout=2000):
        _shift_drag(view, x_start=200, x_end=400)
    overlay = main_window._regions_overlay
    assert len(overlay._regions) == 1
    drawn_start, drawn_end = next(iter(overlay._regions.values())).getRegion()

    # Close the file and re-open it.
    main_window._close_file()
    QApplication.processEvents()
    assert len(overlay._regions) == 0, "overlay should clear on close"

    _open_via_action(main_window, wav, monkeypatch, qtbot)
    QApplication.processEvents()
    # The region should be restored.
    assert len(overlay._regions) == 1
    restored_start, restored_end = next(iter(overlay._regions.values())).getRegion()
    # Float tolerance — JSON round-trip preserves float64 exactly so equal,
    # but allow a tiny epsilon for any pyqtgraph coercion.
    assert abs(restored_start - drawn_start) < 0.01
    assert abs(restored_end - drawn_end) < 0.01


def test_two_regions_persist_distinct_ranges(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_cache_dir: Path,
) -> None:
    """Two Shift+drags produce two distinct regions in the JSON regions list."""
    wav = tmp_path / "fixture.wav"
    make_sine(
        wav,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=10.0,
        sample_rate=44100,
        channels=1,
    )
    _open_via_action(main_window, wav, monkeypatch, qtbot)
    view = main_window._waveform_view

    with qtbot.waitSignal(main_window.regions_changed, timeout=2000):
        _shift_drag(view, x_start=100, x_end=200)
    with qtbot.waitSignal(main_window.regions_changed, timeout=2000):
        _shift_drag(view, x_start=400, x_end=600)

    key = cache_key(wav)
    sp = sidecar_path(tmp_cache_dir, key)
    data = json.loads(sp.read_text())
    assert len(data["regions"]) == 2
    # Both ranges valid and distinct.
    ranges = [(r["start_sec"], r["end_sec"]) for r in data["regions"]]
    assert ranges[0] != ranges[1]
    for s, e in ranges:
        assert s < e


def test_sidecar_path_is_inside_default_cache_root_sidecars(
    main_window: MainWindow,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tmp_cache_dir: Path,
) -> None:
    """The on-disk file path satisfies ``parent == cache_root / 'sidecars'`` AND ``name == f'{cache_key}.json'``."""
    wav = tmp_path / "fixture.wav"
    make_sine(
        wav,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=3.0,
        sample_rate=44100,
        channels=1,
    )
    _open_via_action(main_window, wav, monkeypatch, qtbot)
    view = main_window._waveform_view

    with qtbot.waitSignal(main_window.regions_changed, timeout=2000):
        _shift_drag(view, x_start=200, x_end=400)

    key = cache_key(wav)
    assert re.fullmatch(r"[0-9a-f]{16}", key)
    expected = tmp_cache_dir / "sidecars" / f"{key}.json"
    assert expected.exists()
    assert expected.parent == tmp_cache_dir / "sidecars"
    assert expected.name == f"{key}.json"
