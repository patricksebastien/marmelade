"""quick-260621-gfq Tasks 4 + 6 — normalize dock surface + overlay state + WYSIWYG.

Task 4 (dock/overlay/state): the Mastering dock renders a Normalize row;
the keeper-row toggle and the overlay both read/write the SAME per-keeper
``mastering['normalize']`` state (default 0.0 dBFS); ``regions_data()`` emits
the mastering dict (no standalone fields).

Task 6 (in-place WYSIWYG): ``WaveformView.set_region_normalize`` transforms
only the toggled keeper's span of the rendered envelope in place (peak→target,
DC removed), leaves the viewport X-range unchanged, and reverts on disable.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pytest

from marmelade.audio.sidecar_cache import Region
from marmelade.ui.regions_overlay import RegionsOverlay
from marmelade.ui.mastering_dock import MasteringDock
from marmelade.ui.waveform_view import WaveformView


# ----------------------------------------------------------------- dock

def test_dock_renders_normalize_checkbox(qtbot, qapp) -> None:
    """The dock auto-renders a 'normalize' stage row with a checkbox + gear."""
    dock = MasteringDock()
    qtbot.add_widget(dock)
    # The normalize checkbox is addressable by stage name.
    cb = dock.stage_checkbox("normalize")
    assert cb is not None
    # Default OFF (the session default).
    assert cb.isChecked() is False


def test_dock_normalize_gear_dialog_default_zero(qtbot, qapp, monkeypatch) -> None:
    """The normalize gear opens a ParamsDialog whose target_db defaults to 0.0."""
    from PySide6.QtWidgets import QDialog

    dock = MasteringDock()
    qtbot.add_widget(dock)

    captured: list[dict] = []

    class _StubParamsDialog:
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)

        def exec(self):
            return QDialog.DialogCode.Rejected

        def accepted_values(self):
            return {}

    from marmelade.ui import mastering_dock as md_module

    monkeypatch.setattr(md_module, "ParamsDialog", _StubParamsDialog)
    dock._on_stage_gear_clicked("normalize")

    assert captured, "ParamsDialog was not constructed for the normalize gear"
    current_values = captured[0].get("current_values", {})
    assert current_values.get("target_db") == pytest.approx(0.0)


# ----------------------------------------------------------------- overlay state

@pytest.fixture
def overlay_with_region(qtbot, qapp):
    glw = pg.GraphicsLayoutWidget()
    qtbot.addWidget(glw)
    plot = glw.addPlot()
    overlay = RegionsOverlay(plot_item=plot, duration_s_provider=lambda: 100.0)
    overlay.set_regions(
        [Region(id="rrr1", start_sec=10.0, end_sec=20.0, state="keeper")]
    )
    yield overlay
    try:
        overlay.clear()
    except Exception:
        pass


def test_set_get_normalize_round_trips_through_mastering(overlay_with_region) -> None:
    """set_normalize(True) then get_normalize → (True, 0.0); regions_data carries it."""
    overlay = overlay_with_region
    overlay.set_normalize("rrr1", True)
    assert overlay.get_normalize("rrr1") == (True, 0.0)

    data = overlay.regions_data()
    region = next(r for r in data if r.id == "rrr1")
    assert region.mastering is not None
    assert region.mastering["normalize"]["enabled"] is True
    assert region.mastering["normalize"]["target_db"] == 0.0


def test_set_normalize_seeds_full_session_snapshot(overlay_with_region) -> None:
    """Toggling normalize ON for a mastering=None keeper seeds a full snapshot.

    The MIGRATION-GAP GUARD: the seed comes from load_session_chain_snapshot()
    (never an empty {}), so the resulting mastering dict carries the session
    defaults plus normalize enabled.
    """
    overlay = overlay_with_region
    overlay.set_normalize("rrr1", True)
    data = overlay.regions_data()
    region = next(r for r in data if r.id == "rrr1")
    # The snapshot seed includes all stages (e.g. limiter from defaults).
    assert "limiter" in region.mastering
    assert region.mastering["normalize"]["enabled"] is True


def test_set_normalize_off_round_trips(overlay_with_region) -> None:
    """Disabling normalize leaves an explicit enabled=False entry."""
    overlay = overlay_with_region
    overlay.set_normalize("rrr1", True)
    overlay.set_normalize("rrr1", False)
    assert overlay.get_normalize("rrr1") == (False, 0.0)


# ----------------------------------------------------------------- WYSIWYG (Task 6)

def _make_proxy(n_pairs: int = 1000, amp: int = 8000) -> np.ndarray:
    """A small (N, 2) int16 min/max proxy with a uniform amplitude band."""
    arr = np.empty((n_pairs, 2), dtype=np.int16)
    arr[:, 0] = -amp
    arr[:, 1] = amp
    return arr


def test_set_region_normalize_scales_span_and_keeps_viewport(qtbot, qapp) -> None:
    """Enabling normalize scales the span's envelope; viewport X-range unchanged."""
    view = WaveformView()
    qtbot.add_widget(view)
    sr = 48000
    spp = 256
    proxy = _make_proxy(n_pairs=2000, amp=8000)
    view.render_proxy(proxy, sr, spp)

    x_before = view.waveform_plot.viewRange()[0]
    peak_before = float(np.abs(view._rendered_y).max())

    dur = view._duration_s
    view.set_region_normalize(0.0, dur, True, 0.0)

    x_after = view.waveform_plot.viewRange()[0]
    # Viewport X-range must NOT move (no auto-zoom — locked decision #1 opt 2).
    assert x_before == pytest.approx(x_after)
    # The span's envelope scaled UP toward full int16 scale (target 0 dB).
    peak_after = float(np.abs(view._rendered_y).max())
    assert peak_after > peak_before


def test_set_region_normalize_reverts_on_disable(qtbot, qapp) -> None:
    """Disabling normalize restores the original span values."""
    view = WaveformView()
    qtbot.add_widget(view)
    sr = 48000
    spp = 256
    proxy = _make_proxy(n_pairs=2000, amp=8000)
    view.render_proxy(proxy, sr, spp)

    original = view._rendered_y.copy()
    dur = view._duration_s
    view.set_region_normalize(0.0, dur, True, 0.0)
    assert not np.array_equal(view._rendered_y, original)

    view.set_region_normalize(0.0, dur, False, 0.0)
    np.testing.assert_array_equal(view._rendered_y, original)
