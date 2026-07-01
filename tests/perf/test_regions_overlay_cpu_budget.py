"""Plan 03-04b — Wave 0 paint-budget gate for the RegionsOverlay.

UI-SPEC §Performance: with 100 LinearRegionItem instances on the
WaveformView, mean ``setXRange`` wall-clock must stay under 50 ms.

Always runs (including under offscreen) — exercises PyQtGraph's paint
pipeline via setXRange + qtbot.wait(0) which dispatches the queued
paint events even in offscreen mode.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from marmelade.audio import peak_builder, proxy_cache
from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


SR = 44100


def _prewarm_cache(src: Path) -> Path:
    key = proxy_cache.cache_key(src)
    proxy_p = proxy_cache.proxy_path(default_cache_root(), key)
    proxy_p.parent.mkdir(parents=True, exist_ok=True)
    peak_builder.build_proxy(src, proxy_p, samples_per_pixel=256)
    return proxy_p


@pytest.mark.perf
def test_100_regions_paint_under_50ms_per_setXRange(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Mean setXRange wall-clock < 50 ms with 100 regions on the overlay."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=60.0, sample_rate=SR, channels=1, fmt="wav")
    _prewarm_cache(src)

    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.render_complete, timeout=10000, raising=True):
        window._open_file(str(src))

    # Install 100 untouched regions spread across the timeline.
    duration = 60.0
    region_w = duration / 200.0  # 0.3s wide
    regions = [
        Region(
            id=f"r{i:03d}",
            start_sec=(i * duration / 100.0),
            end_sec=(i * duration / 100.0) + region_w,
            state="untouched",
        )
        for i in range(100)
    ]
    window._regions_overlay.set_regions(regions)

    vb = window._waveform_view.waveform_plot.getViewBox()

    durations_ms: list[float] = []
    for i in range(10):
        t0 = time.perf_counter()
        # Walk across the timeline so the paint pipeline is exercised
        # each iteration (no-change setXRange may short-circuit).
        x_lo = i * 1.0
        x_hi = x_lo + 30.0
        vb.setXRange(x_lo, x_hi, padding=0)
        qtbot.wait(0)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(durations_ms) / len(durations_ms)
    assert mean_ms < 50.0, (
        f"100 regions × setXRange mean wall-clock = {mean_ms:.2f} ms exceeds "
        f"50 ms budget. UI-SPEC §Performance pan ≥ 30 fps (33 ms/frame); the "
        f"50 ms budget gives headroom over the 30 fps target while still "
        f"catching a paint-pipeline regression on the region overlay."
    )


@pytest.mark.perf
def test_drag_create_then_setXRange_under_budget(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Synthesize a region drag-create, then 10 setXRange — total under 500 ms."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "fixture.wav"
    make_sine(src, duration_s=30.0, sample_rate=SR, channels=1, fmt="wav")
    _prewarm_cache(src)

    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.render_complete, timeout=10000, raising=True):
        window._open_file(str(src))

    overlay = window._regions_overlay
    overlay.start_draft(5.0)
    overlay.update_draft(10.0)
    overlay.commit_draft(10.0)

    vb = window._waveform_view.waveform_plot.getViewBox()
    t0 = time.perf_counter()
    for i in range(10):
        x_lo = i * 0.5
        x_hi = x_lo + 15.0
        vb.setXRange(x_lo, x_hi, padding=0)
        qtbot.wait(0)
    total_ms = (time.perf_counter() - t0) * 1000.0
    assert total_ms < 500.0, (
        f"drag-create + 10× setXRange total = {total_ms:.2f} ms exceeds "
        f"500 ms ceiling."
    )
