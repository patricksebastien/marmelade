"""Phase 8 Plan 08-06 Task 1 — sidecar youtube_video_id app-restart e2e (T-08-06-05).

End-to-end pin: open a file via MainWindow, draw a region, mark it as
a keeper, set ``region.youtube_video_id`` via the same code path the
production ``_on_youtube_upload_finished`` slot uses, save the sidecar,
CLOSE the file, RE-OPEN it on a fresh MainWindow instance, and assert
the region carries the same ``youtube_video_id`` after the round-trip.

This pin distinguishes from the unit-level
``tests/unit/audio/test_sidecar_youtube_video_id.py`` by exercising
the full MainWindow lifecycle: cache-key resolution, sidecar load on
file-open, in-memory Region carrier through the overlay, and
``_on_regions_changed`` atomic-save on every mutation.

Placebo audit (Phase 7 LEARNINGS):
    PRE-FIX expected failure signal — if MainWindow's overlay layer
    didn't propagate ``youtube_video_id`` through ``regions_data()``,
    or if ``save_sidecar`` accidentally dropped the field on an
    additive schema upgrade, the reopened MainWindow's overlay would
    show the region but with ``youtube_video_id is None``.

T-08-06-05 mitigation contract: this test IS the regression pin.
"""

from __future__ import annotations

import json
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


def _send_shift_drag(view, x_start: int, x_end: int, y: int = 30) -> None:
    """Shift+drag from x_start to x_end through one intermediate move."""
    viewport = view.graphics_layout.viewport()
    midpoint = (x_start + x_end) // 2

    def _ev(typ, x, btn_pressed):
        return QMouseEvent(
            typ,
            QPointF(float(x), float(y)),
            viewport.mapToGlobal(QPoint(x, y)).toPointF(),
            Qt.MouseButton.LeftButton if btn_pressed else Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton if btn_pressed else Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ShiftModifier,
        )

    QApplication.sendEvent(viewport, _ev(QEvent.Type.MouseButtonPress, x_start, True))
    QApplication.sendEvent(viewport, _ev(QEvent.Type.MouseMove, midpoint, True))
    QApplication.sendEvent(viewport, _ev(QEvent.Type.MouseButtonRelease, x_end, True))


def _open_via_action(window, wav, monkeypatch, qtbot):
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(wav), "Audio files (*.wav *.flac *.mp3)")),
    )
    with qtbot.waitSignal(window.render_complete, timeout=15000, raising=True):
        window._action_open_file()
    QApplication.processEvents()


def test_sidecar_youtube_video_id_roundtrip_e2e(
    qtbot, qapp, monkeypatch, tmp_path: Path, tmp_cache_dir: Path
) -> None:
    """Open → draw → set youtube_video_id → save → close → reopen → verify roundtrip.

    Pins T-08-06-05 — Region.youtube_video_id survives a full app-
    restart cycle through MainWindow's file-open / sidecar-load
    machinery.
    """
    theme.apply_theme(QApplication.instance())

    wav = tmp_path / "fixture.wav"
    make_sine(wav, freq_hz=1000.0, amp=0.5, duration_s=5.0, sample_rate=44100, channels=1)

    # ----- Session 1: open + draw + set video_id + save -----
    window1 = MainWindow()
    qtbot.addWidget(window1)
    window1.show()
    window1.resize(1600, 600)
    QApplication.processEvents()

    _open_via_action(window1, wav, monkeypatch, qtbot)

    # Draw a region.
    view = window1._waveform_view
    with qtbot.waitSignal(window1.regions_changed, timeout=2000):
        _send_shift_drag(view, x_start=200, x_end=400)

    overlay = window1._regions_overlay
    assert len(overlay._regions) == 1, "expected 1 region after shift+drag"

    # Read the region id from the overlay's regions_data().
    regions = overlay.regions_data()
    assert len(regions) == 1
    region_id = regions[0].id

    # Set youtube_video_id via the SAME code path
    # _on_youtube_upload_finished uses (the overlay's setter that
    # persists the value on the per-region widget so regions_data()
    # round-trips it). Plan 08-06 Task 1 added this setter as part of
    # closing the D-30 persistence loop.
    expected_video_id = "test_yt_id_xyz_e2e"
    overlay.set_youtube_video_id(region_id, expected_video_id)

    # Trigger the save path.
    with qtbot.waitSignal(window1.regions_changed, timeout=2000):
        window1._on_regions_changed()

    # Sanity — the on-disk JSON now carries the field.
    key = cache_key(wav)
    sp = sidecar_path(tmp_cache_dir, key)
    on_disk = json.loads(sp.read_text())
    assert len(on_disk["regions"]) == 1
    assert on_disk["regions"][0].get("youtube_video_id") == expected_video_id, (
        f"youtube_video_id missing or wrong in sidecar JSON: "
        f"{on_disk['regions'][0]}"
    )

    # ----- Close + dispose session 1 -----
    window1._close_file()
    QApplication.processEvents()
    window1.close()
    QApplication.processEvents()

    # ----- Session 2: fresh MainWindow → reopen the same file -----
    window2 = MainWindow()
    qtbot.addWidget(window2)
    window2.show()
    window2.resize(1600, 600)
    QApplication.processEvents()

    _open_via_action(window2, wav, monkeypatch, qtbot)
    QApplication.processEvents()

    overlay2 = window2._regions_overlay
    assert len(overlay2._regions) == 1, "expected 1 region restored from sidecar"

    regions2 = overlay2.regions_data()
    assert len(regions2) == 1
    assert regions2[0].id == region_id, "region id changed across restart"
    # PRIMARY ASSERTION — youtube_video_id survived the round-trip.
    assert regions2[0].youtube_video_id == expected_video_id, (
        f"D-30 violated — youtube_video_id lost across app restart; "
        f"got {regions2[0].youtube_video_id!r}, expected "
        f"{expected_video_id!r}"
    )
