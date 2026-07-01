"""Integration tests for :class:`marmelade.audio.export_worker.ExportRunnable`
and MainWindow export wiring (Plan 03-04b — EXP-01/03 D-A4-3/4/5).

The runnable mirrors the Phase 2.1 AudioProxyRunnable 4-signal contract
verbatim. MainWindow surfaces TWO Export actions (MP3 + WAV) from the
right-click context menu per CONTEXT D-A4-4 LOCKED.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from pedalboard.io import AudioFile
from PySide6.QtCore import QSettings, QThreadPool
from PySide6.QtWidgets import QApplication, QFileDialog

from marmelade.audio.audio_proxy_cache import audio_proxy_path, cache_key
from marmelade.audio.export_worker import ExportRunnable
from marmelade.audio.peak_builder import BuildCancelled
from marmelade.concurrency.worker import WorkerSignals
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


SR = 44100


def _make_stereo_float32_proxy(path: Path, duration_s: float, sr: int = SR) -> None:
    total = int(duration_s * sr)
    t = np.arange(total, dtype=np.float64) / sr
    mono = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    with sf.SoundFile(
        str(path), mode="w", samplerate=sr, channels=2,
        subtype="FLOAT", format="RF64",
    ) as f:
        f.write(stereo)


# ==================================================== ExportRunnable contract


@pytest.fixture(autouse=True)
def _qapp(qapp):
    return qapp


def test_export_runnable_setAutoDelete_false(tmp_path: Path) -> None:
    """CR-02 — setAutoDelete(False) so MainWindow's stored ref is safe."""
    src = tmp_path / "src.proxy.wav"
    dst = tmp_path / "out.wav"
    runnable = ExportRunnable(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=SR,
        fade_frames=0, fmt="wav", sample_rate=SR,
    )
    assert runnable.autoDelete() is False


def test_export_runnable_signals_is_worker_signals_verbatim(tmp_path: Path) -> None:
    """D-16 — signals must be EXACTLY WorkerSignals, not a subclass."""
    src = tmp_path / "src.proxy.wav"
    dst = tmp_path / "out.wav"
    runnable = ExportRunnable(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=SR,
        fade_frames=0, fmt="wav", sample_rate=SR,
    )
    assert type(runnable.signals) is WorkerSignals


def test_export_runnable_emits_finished_with_dst_path(qtbot, tmp_path: Path) -> None:
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 3.0)
    dst = tmp_path / "out.wav"
    runnable = ExportRunnable(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=SR * 2,
        fade_frames=SR // 2, fmt="wav", sample_rate=SR,
    )
    payloads: list = []
    runnable.signals.finished.connect(payloads.append)

    with qtbot.waitSignal(runnable.signals.finished, timeout=30000):
        QThreadPool.globalInstance().start(runnable)

    assert payloads, "finished did not emit"
    assert Path(payloads[0]) == dst
    assert dst.exists()


def test_export_runnable_emits_progress_then_finished(qtbot, tmp_path: Path) -> None:
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 5.0)
    dst = tmp_path / "out.wav"
    runnable = ExportRunnable(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=SR * 5,
        fade_frames=0, fmt="wav", sample_rate=SR,
    )
    progress_log: list[int] = []
    runnable.signals.progress.connect(progress_log.append)

    with qtbot.waitSignal(runnable.signals.finished, timeout=30000):
        QThreadPool.globalInstance().start(runnable)

    assert len(progress_log) >= 1
    assert all(0 <= p <= 100 for p in progress_log)


def test_export_runnable_error_emits_error_signal(qtbot, tmp_path: Path) -> None:
    """Non-existent source → signals.error fires with a non-empty message."""
    bad_src = tmp_path / "does_not_exist.proxy.wav"
    dst = tmp_path / "out.wav"
    runnable = ExportRunnable(
        proxy_path=bad_src, dst_path=dst,
        start_frame=0, end_frame=SR,
        fade_frames=0, fmt="wav", sample_rate=SR,
    )
    error_log: list[str] = []
    runnable.signals.error.connect(error_log.append)

    with qtbot.waitSignal(runnable.signals.error, timeout=10000):
        QThreadPool.globalInstance().start(runnable)

    assert error_log
    assert error_log[0]  # non-empty


def test_export_runnable_round_trip_mp3(qtbot, tmp_path: Path) -> None:
    """End-to-end MP3 round trip via QRunnable — re-readable, samplerate preserved."""
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 3.0)
    dst = tmp_path / "out.mp3"
    runnable = ExportRunnable(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=SR * 2,
        fade_frames=SR // 2, fmt="mp3", sample_rate=SR,
    )
    with qtbot.waitSignal(runnable.signals.finished, timeout=30000):
        QThreadPool.globalInstance().start(runnable)

    assert dst.exists()
    with AudioFile(str(dst), "r") as f:
        assert f.samplerate == SR


def test_export_runnable_round_trip_wav_subtype_float(
    qtbot, tmp_path: Path
) -> None:
    src = tmp_path / "src.proxy.wav"
    _make_stereo_float32_proxy(src, 3.0)
    dst = tmp_path / "out.wav"
    runnable = ExportRunnable(
        proxy_path=src, dst_path=dst,
        start_frame=0, end_frame=SR * 2,
        fade_frames=SR // 2, fmt="wav", sample_rate=SR,
    )
    with qtbot.waitSignal(runnable.signals.finished, timeout=30000):
        QThreadPool.globalInstance().start(runnable)
    assert sf.info(str(dst)).subtype == "FLOAT"


# ===================================================== MainWindow integration


def _open_and_wait(window: MainWindow, qtbot, src: Path) -> None:
    """Open ``src`` and wait for both the audio-proxy and render to settle."""
    render_done = {"v": False}
    window.render_complete.connect(lambda: render_done.update(v=True))
    proxy_done = {"v": False}
    window.audio_proxy_complete.connect(lambda _p: proxy_done.update(v=True))
    window._open_file(str(src))
    # For WAV sources, audio_proxy_complete never fires — render_complete is enough.
    # For non-WAV, both must settle.
    if src.suffix.lower() == ".wav":
        qtbot.waitUntil(lambda: render_done["v"], timeout=15000)
    else:
        qtbot.waitUntil(
            lambda: render_done["v"] and proxy_done["v"], timeout=20000
        )


def test_main_window_has_export_complete_signal(qtbot, qapp) -> None:
    """MainWindow exposes ``export_complete = Signal(str)`` as a public test seam."""
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    # Class-level signal must exist.
    assert hasattr(window, "export_complete")


def test_overlay_has_export_requested_signal(qtbot, qapp) -> None:
    """RegionsOverlay must expose ``export_requested = Signal(str, str)``."""
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    assert hasattr(window._regions_overlay, "export_requested")


def test_both_export_actions_present_and_enabled_for_keeper(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Keeper region's context menu has BOTH MP3 + WAV Export actions, BOTH enabled."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=3.0, sample_rate=SR, channels=1, fmt="wav")

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    # Create a region and mark Keeper.
    from marmelade.audio.sidecar_cache import Region
    overlay = window._regions_overlay
    overlay.set_regions(
        [Region(id="rk1", start_sec=0.5, end_sec=2.0, state="keeper")]
    )
    region = overlay._regions["rk1"]
    menus = region.getContextMenus(None)
    actions = menus[0].actions()
    labels = [a.text() for a in actions]
    assert "Export this region as MP3…" in labels
    assert "Export this region as WAV…" in labels
    # Both enabled.
    for a in actions:
        if a.text() in ("Export this region as MP3…", "Export this region as WAV…"):
            assert a.isEnabled(), f"{a.text()} should be enabled for keeper"


def test_both_export_actions_disabled_for_non_keeper(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Untouched/Trash region's context menu has the actions but BOTH disabled."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=3.0, sample_rate=SR, channels=1, fmt="wav")

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    from marmelade.audio.sidecar_cache import Region
    overlay = window._regions_overlay
    overlay.set_regions(
        [Region(id="ru1", start_sec=0.5, end_sec=2.0, state="untouched")]
    )
    region = overlay._regions["ru1"]
    menus = region.getContextMenus(None)
    actions = menus[0].actions()
    for a in actions:
        if a.text() in ("Export this region as MP3…", "Export this region as WAV…"):
            assert not a.isEnabled()


def test_mp3_export_via_context_menu(
    qtbot, qapp, monkeypatch, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """End-to-end MP3 export: open file → keeper region → trigger MP3 action → file written."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=4.0, sample_rate=SR, channels=1, fmt="wav")

    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *args, **kwargs: str(export_dir),
    )
    # Make sure QSettings is clean for this test (clears any export_dir from prior tests).
    QSettings("Marmelade", "Marmelade").remove("export_dir")

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    from marmelade.audio.sidecar_cache import Region
    overlay = window._regions_overlay
    overlay.set_regions(
        [Region(id="rk1", start_sec=0.5, end_sec=2.5, state="keeper")]
    )

    region = overlay._regions["rk1"]
    menus = region.getContextMenus(None)
    actions = menus[0].actions()
    mp3_action = next(a for a in actions if a.text() == "Export this region as MP3…")

    payloads: list[str] = []
    window.export_complete.connect(payloads.append)
    with qtbot.waitSignal(window.export_complete, timeout=30000):
        mp3_action.trigger()

    assert payloads
    out = Path(payloads[0])
    assert out.exists()
    assert out.suffix == ".mp3"
    assert out.parent == export_dir
    with AudioFile(str(out), "r") as f:
        assert f.samplerate == SR


def test_wav_export_via_context_menu(
    qtbot, qapp, monkeypatch, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """End-to-end WAV export — same flow as MP3, both formats first-class (B-5)."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=4.0, sample_rate=SR, channels=1, fmt="wav")

    export_dir = tmp_path / "exports_wav"
    export_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *args, **kwargs: str(export_dir),
    )
    QSettings("Marmelade", "Marmelade").remove("export_dir")

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    from marmelade.audio.sidecar_cache import Region
    overlay = window._regions_overlay
    overlay.set_regions(
        [Region(id="rk2", start_sec=0.5, end_sec=2.5, state="keeper")]
    )
    region = overlay._regions["rk2"]
    menus = region.getContextMenus(None)
    actions = menus[0].actions()
    wav_action = next(a for a in actions if a.text() == "Export this region as WAV…")

    payloads: list[str] = []
    window.export_complete.connect(payloads.append)
    with qtbot.waitSignal(window.export_complete, timeout=30000):
        wav_action.trigger()

    out = Path(payloads[0])
    assert out.exists()
    assert out.suffix == ".wav"
    info = sf.info(str(out))
    assert info.subtype == "FLOAT"
    assert info.frames > 0


def test_first_export_prompts_for_directory(
    qtbot, qapp, monkeypatch, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Clear QSettings → first export triggers QFileDialog.getExistingDirectory."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=3.0, sample_rate=SR, channels=1, fmt="wav")

    export_dir = tmp_path / "first_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    QSettings("Marmelade", "Marmelade").remove("export_dir")

    call_count = {"n": 0}

    def fake_dialog(*args, **kwargs):
        call_count["n"] += 1
        return str(export_dir)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", fake_dialog)

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    from marmelade.audio.sidecar_cache import Region
    window._regions_overlay.set_regions(
        [Region(id="rk3", start_sec=0.2, end_sec=1.5, state="keeper")]
    )

    with qtbot.waitSignal(window.export_complete, timeout=30000):
        window._on_export_region_requested("rk3", "mp3")

    assert call_count["n"] == 1
    # QSettings now remembers the choice.
    s = QSettings("Marmelade", "Marmelade")
    remembered = s.value("export_dir", "")
    assert Path(str(remembered)) == export_dir


def test_subsequent_exports_use_remembered_dir(
    qtbot, qapp, monkeypatch, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """QSettings prepopulated → QFileDialog must NEVER be invoked."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=3.0, sample_rate=SR, channels=1, fmt="wav")

    export_dir = tmp_path / "remembered_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    s = QSettings("Marmelade", "Marmelade")
    s.setValue("export_dir", str(export_dir))
    s.sync()

    sentinel = MagicMock(side_effect=AssertionError("dialog should not be called"))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", sentinel)

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    from marmelade.audio.sidecar_cache import Region
    window._regions_overlay.set_regions(
        [Region(id="rk4", start_sec=0.2, end_sec=1.5, state="keeper")]
    )

    payloads: list[str] = []
    window.export_complete.connect(payloads.append)
    with qtbot.waitSignal(window.export_complete, timeout=30000):
        window._on_export_region_requested("rk4", "mp3")

    out = Path(payloads[0])
    assert out.parent == export_dir
    sentinel.assert_not_called()


def test_export_uses_proxy_not_source(
    qtbot, qapp, monkeypatch, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Open an MP3 source → export reads from the .proxy.wav, NOT the MP3."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "source.mp3"
    make_sine(src, duration_s=3.0, sample_rate=SR, channels=1, fmt="mp3")

    export_dir = tmp_path / "proxy_check_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *args, **kwargs: str(export_dir),
    )
    QSettings("Marmelade", "Marmelade").remove("export_dir")

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    # The proxy now exists for this source.
    canonical_proxy = audio_proxy_path(default_cache_root(), cache_key(src))
    assert canonical_proxy.exists()

    # Verify _current_playback_path is the proxy, NOT the source.
    assert window._current_playback_path == canonical_proxy

    # Stub _spawn_export_worker to capture the proxy_path arg.
    captured = {}
    orig_spawn = window._spawn_export_worker

    def capture_spawn(**kwargs):
        captured.update(kwargs)
        orig_spawn(**kwargs)

    window._spawn_export_worker = capture_spawn

    from marmelade.audio.sidecar_cache import Region
    window._regions_overlay.set_regions(
        [Region(id="rk5", start_sec=0.3, end_sec=1.5, state="keeper")]
    )

    with qtbot.waitSignal(window.export_complete, timeout=30000):
        window._on_export_region_requested("rk5", "mp3")

    assert Path(captured["proxy_path"]) == canonical_proxy
    assert Path(captured["proxy_path"]) != src


def test_filename_matches_pattern(
    qtbot, qapp, monkeypatch, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Exported filename follows {YYYY-MM-DD}_{HHMMSS}_{trait}.{ext} pattern."""
    import os
    from datetime import datetime

    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=4.0, sample_rate=SR, channels=1, fmt="wav")
    # Pin mtime to a known date for filename determinism.
    ts = datetime.fromisoformat("2026-04-03T12:34:56").timestamp()
    os.utime(str(src), (ts, ts))

    export_dir = tmp_path / "exports_pattern"
    export_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *args, **kwargs: str(export_dir),
    )
    QSettings("Marmelade", "Marmelade").remove("export_dir")

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    from marmelade.audio.sidecar_cache import Region
    # Region start at exactly 1 minute 5 seconds (HHMMSS = 000105).
    # But our fixture is only 4s — use offset 2 → HHMMSS = 000002.
    window._regions_overlay.set_regions(
        [Region(id="rk6", start_sec=2.0, end_sec=3.5, state="keeper")]
    )

    payloads: list[str] = []
    window.export_complete.connect(payloads.append)
    with qtbot.waitSignal(window.export_complete, timeout=30000):
        window._on_export_region_requested("rk6", "mp3")

    out = Path(payloads[0])
    # Pattern: 2026-04-03_000002_<trait>.mp3 ; trait fallback is "clip" when no heatmaps.
    assert out.name.startswith("2026-04-03_000002_")
    assert out.name.endswith(".mp3")


def test_cancel_button_visible_during_export_clears_after_finish(
    qtbot, qapp, monkeypatch, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Status-bar cancel × button hidden after the export's terminal signal."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=3.0, sample_rate=SR, channels=1, fmt="wav")

    export_dir = tmp_path / "exports_cancel_btn"
    export_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *args, **kwargs: str(export_dir),
    )
    QSettings("Marmelade", "Marmelade").remove("export_dir")

    window = MainWindow()
    qtbot.addWidget(window)
    _open_and_wait(window, qtbot, src)

    from marmelade.audio.sidecar_cache import Region
    window._regions_overlay.set_regions(
        [Region(id="rk7", start_sec=0.2, end_sec=1.5, state="keeper")]
    )

    with qtbot.waitSignal(window.export_complete, timeout=30000):
        window._on_export_region_requested("rk7", "mp3")
    # After finished, cancel button is hidden.
    assert not window._status_export_cancel.isVisible()
