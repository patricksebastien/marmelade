"""Integration: spectral-mode reference legends (quick-260629-vui).

Pins the four INSET reference overlays added to the three SPECTRAL render
modes of :class:`marmelade.ui.waveform_view.WaveformView`:

    1. SPECTROGRAM — Hz tick labels on the left axis (``_FreqAxisItem``),
       y-positioned via the mel center-freq table.
    2. SPECTROGRAM — a vertical Magma dB colorbar inset (``_mag_colorbar``)
       labelled from the live spectral header (db_ref top / db_floor bottom).
    3. CENTROID — a vertical Magma centroid-Hz log colorbar
       (``_centroid_colorbar``), labelled 100 / 1k / 10k.
    4. RGB_BAND — a static colored band legend (``_rgb_legend``) with the
       real low/mid/high split Hz from spectral_builder.

All four are anchored INSIDE the plot ViewBox and must NOT change the plot's
outer geometry or the reserved 40 px left-axis width (the waveform is
X-aligned to the heatmap lanes below; a reflow would misalign them). Each
overlay is visible ONLY in its mode; non-spectral modes stay byte-identical
(blank left ticks, no overlays).

The tests stash fake spectral data directly via ``set_spectral_data`` and
toggle the render mode through ``render_mode_combo`` (the live seam — assert
real ``.isVisible()`` / real tick strings, not method existence; project
memory: re-verify GREEN claims against the live seam).
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from marmelade.audio.render_modes import (
    CENTROID_FMAX_HZ,
    CENTROID_FMIN_HZ,
    RenderMode,
)
from marmelade.audio.spectral_builder import (
    FMAX,
    N_MELS,
    _BAND_HI_HZ,
    _BAND_LO_HZ,
)
from marmelade.audio.spectral_cache import DB_FLOOR, DB_REF
from marmelade.ui import theme
from marmelade.ui.waveform_view import _FREQ_TICK_HZ, _MEL_FREQS, WaveformView


@pytest.fixture
def waveform_view(qtbot, qapp):
    theme.apply_theme(QApplication.instance())
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(800, 200)
    return view


class _FakeHeader:
    """Minimal stand-in for the spectral-cache SpectralHeader dataclass."""

    def __init__(self, db_floor: float, db_ref: float) -> None:
        self.db_floor = db_floor
        self.db_ref = db_ref


def _fake_mel(n_cols: int = 200) -> np.ndarray:
    """A (N_MELS, n_cols) uint8 mel image (row 0 = low freq)."""
    rng = np.random.default_rng(0)
    return (rng.random((N_MELS, n_cols)) * 255).astype(np.uint8)


def _stash(view: WaveformView, *, header=None) -> None:
    """Stash a full set of fake spectral arrays so every spectral mode renders."""
    mel = _fake_mel()
    n_cols = mel.shape[1]
    centroid = np.linspace(0.1, 0.9, n_cols).astype(np.float32)
    bands = np.abs(np.random.default_rng(1).random((3, n_cols))).astype(np.float32)
    view.set_spectral_data(mel, centroid, bands, header)


def _select(view: WaveformView, mode: RenderMode) -> None:
    view.render_mode_combo.setCurrentIndex(list(RenderMode).index(mode))
    QApplication.processEvents()


# --------------------------------------------------------------------------
# Task 1 — _FreqAxisItem mel-mapped Hz ticks
# --------------------------------------------------------------------------


def test_freq_axis_maps_1000hz_to_mel_y(waveform_view: WaveformView) -> None:
    """1000 Hz maps to the mel-stretched y within 1e-6 (freq mapping)."""
    view = waveform_view
    _stash(view)
    _select(view, RenderMode.SPECTROGRAM)

    y_min, y_max = view.waveform_plot.viewRange()[1]
    left = view.waveform_plot.getAxis("left")

    frac_row = np.interp(1000.0, _MEL_FREQS, np.arange(N_MELS))
    expected_y = y_min + (frac_row / (N_MELS - 1)) * (y_max - y_min)

    got_y = left._hz_to_y(1000.0, y_min, y_max)
    assert got_y == pytest.approx(expected_y, abs=1e-6)


def test_freq_axis_orientation_low_bottom_high_top(
    waveform_view: WaveformView,
) -> None:
    """Low Hz maps NEAR y_min (bottom); high Hz NEAR y_max (top)."""
    view = waveform_view
    _stash(view)
    _select(view, RenderMode.SPECTROGRAM)

    y_min, y_max = view.waveform_plot.viewRange()[1]
    left = view.waveform_plot.getAxis("left")

    y_low = left._hz_to_y(100.0, y_min, y_max)
    y_high = left._hz_to_y(16000.0, y_min, y_max)

    span = y_max - y_min
    assert abs(y_low - y_min) < 0.25 * span, "100 Hz should sit near the bottom"
    assert abs(y_high - y_max) < 0.25 * span, "16 kHz should sit near the top"
    assert y_low < y_high, "low freq must be below high freq"


def test_freq_axis_labels_in_spectrogram(waveform_view: WaveformView) -> None:
    """In SPECTROGRAM the left axis emits non-empty Hz tick strings."""
    view = waveform_view
    _stash(view)
    _select(view, RenderMode.SPECTROGRAM)

    left = view.waveform_plot.getAxis("left")
    y_min, y_max = view.waveform_plot.viewRange()[1]
    tv = left.tickValues(y_min, y_max, y_max - y_min)
    # Flatten all tick values across levels.
    values = [v for _spacing, vals in tv for v in vals]
    assert values, "SPECTROGRAM left axis must emit Hz ticks"
    strings = left.tickStrings(values, 1.0, 1.0)
    nonblank = [s for s in strings if s]
    assert nonblank, "tick strings must be non-empty in SPECTROGRAM"
    # 1k / 10k labels are produced for the standard tick set.
    assert any(s in ("100", "1k", "10k", "20k") for s in nonblank)


def test_freq_axis_delegation_blank_when_not_spectrogram(
    waveform_view: WaveformView,
) -> None:
    """Non-spectral modes produce blank left ticks (byte-identical gutter)."""
    view = waveform_view
    _stash(view)
    _select(view, RenderMode.CLASSIC)

    left = view.waveform_plot.getAxis("left")
    # Whatever tick values exist, every label must be blank in CLASSIC.
    strings = left.tickStrings([0.0, 100.0, -100.0], 1.0, 1.0)
    assert all(s == "" for s in strings), (
        "non-spectral modes must render blank left ticks"
    )
    # The 40 px gutter geometry is unchanged.
    assert left.width() == pytest.approx(40.0)


# --------------------------------------------------------------------------
# Task 2 — dB + centroid colorbar insets + visibility wiring
# --------------------------------------------------------------------------


def test_mag_colorbar_labels_from_live_header(waveform_view: WaveformView) -> None:
    """Magnitude colorbar reads db_ref (top) / db_floor (bottom) from the header."""
    view = waveform_view
    _stash(view, header=_FakeHeader(db_floor=-80.0, db_ref=0.0))
    _select(view, RenderMode.SPECTROGRAM)

    labels = view._mag_colorbar.labels  # {text: value 0..1 from bottom}
    # Top (value 1.0) is db_ref, bottom (value 0.0) is db_floor.
    top = {v: t for t, v in labels.items()}.get(1.0)
    bottom = {v: t for t, v in labels.items()}.get(0.0)
    assert top == "0", f"top label should be db_ref '0', got {top}"
    assert bottom == "-80", f"bottom label should be db_floor '-80', got {bottom}"


def test_mag_colorbar_labels_update_with_header(
    waveform_view: WaveformView,
) -> None:
    """A different header (db_floor=-60) updates the bottom label."""
    view = waveform_view
    _stash(view, header=_FakeHeader(db_floor=-60.0, db_ref=0.0))
    _select(view, RenderMode.SPECTROGRAM)

    labels = view._mag_colorbar.labels
    bottom = {v: t for t, v in labels.items()}.get(0.0)
    assert bottom == "-60", f"bottom label should track header db_floor, got {bottom}"


def test_mag_colorbar_labels_fallback_no_header(
    waveform_view: WaveformView,
) -> None:
    """With header=None the labels fall back to DB_FLOOR / DB_REF."""
    view = waveform_view
    _stash(view, header=None)
    _select(view, RenderMode.SPECTROGRAM)

    labels = view._mag_colorbar.labels
    top = {v: t for t, v in labels.items()}.get(1.0)
    bottom = {v: t for t, v in labels.items()}.get(0.0)
    assert top == str(int(DB_REF))
    assert bottom == str(int(DB_FLOOR))


def test_centroid_colorbar_log_positioned_labels(
    waveform_view: WaveformView,
) -> None:
    """Centroid colorbar labels 100 / 1k / 10k log-positioned in [fmin, fmax]."""
    view = waveform_view
    _stash(view)
    _select(view, RenderMode.CENTROID)

    labels = view._centroid_colorbar.labels  # {text: value 0..1}
    lo = np.log(CENTROID_FMIN_HZ)
    hi = np.log(CENTROID_FMAX_HZ)
    for hz, text in ((100.0, "100"), (1000.0, "1k"), (10000.0, "10k")):
        assert text in labels, f"missing centroid label {text}"
        expected = (np.log(hz) - lo) / (hi - lo)
        assert labels[text] == pytest.approx(expected, abs=1e-6)


def test_colorbars_visible_only_in_their_mode(
    waveform_view: WaveformView,
) -> None:
    """mag bar visible ONLY in SPECTROGRAM, centroid bar ONLY in CENTROID."""
    view = waveform_view
    _stash(view, header=_FakeHeader(db_floor=-80.0, db_ref=0.0))

    expected = {
        RenderMode.CLASSIC: (False, False),
        RenderMode.DB: (False, False),
        RenderMode.ENERGY: (False, False),
        RenderMode.SPECTROGRAM: (True, False),
        RenderMode.CENTROID: (False, True),
        RenderMode.RGB_BAND: (False, False),
    }
    for mode, (mag_vis, cen_vis) in expected.items():
        _select(view, mode)
        assert view._mag_colorbar.isVisible() is mag_vis, (
            f"mag colorbar visibility wrong in {mode}"
        )
        assert view._centroid_colorbar.isVisible() is cen_vis, (
            f"centroid colorbar visibility wrong in {mode}"
        )


def test_colorbar_no_reflow(waveform_view: WaveformView) -> None:
    """Entering SPECTROGRAM must not resize the plot or the 40 px left axis."""
    view = waveform_view
    _stash(view)
    QApplication.processEvents()
    vb = view.waveform_plot.getViewBox()
    left = view.waveform_plot.getAxis("left")
    geo_before = vb.geometry()
    width_before = left.width()

    _select(view, RenderMode.SPECTROGRAM)

    assert view.waveform_plot.getViewBox().geometry() == geo_before, (
        "ViewBox geometry changed entering SPECTROGRAM (colorbar reflowed it)"
    )
    assert left.width() == pytest.approx(width_before), (
        "left-axis width changed entering SPECTROGRAM"
    )
    assert width_before == pytest.approx(40.0)


# --------------------------------------------------------------------------
# Task 3 — RGB band legend inset + full-mode visibility sweep
# --------------------------------------------------------------------------


def test_rgb_legend_content_from_imported_splits(
    waveform_view: WaveformView,
) -> None:
    """RGB legend HTML carries the real 250 / 4000 splits + red/green/blue."""
    view = waveform_view
    _stash(view)
    _select(view, RenderMode.RGB_BAND)

    html = view._rgb_legend.toHtml() if hasattr(view._rgb_legend, "toHtml") else ""
    # TextItem stores its source HTML; read it back via the document or our attr.
    if not html:
        html = view._rgb_legend_html
    assert str(int(_BAND_LO_HZ)) in html, "low/mid split (250) missing from legend"
    assert str(int(_BAND_HI_HZ)) in html, "mid/high split (4000) missing from legend"
    # Color markup present (any of the three band colors).
    lowered = html.lower()
    assert any(c in lowered for c in ("red", "#ff0000", "rgb(255")), "red span missing"
    assert any(
        # #00b000 is the legibility-darkened green used on the white fill.
        c in lowered for c in ("green", "#00ff00", "#00b000", "lime", "rgb(0,255")
    ), "green span missing"
    assert any(c in lowered for c in ("blue", "#0000ff", "rgb(0,0,255")), "blue span missing"


def test_rgb_legend_visible_only_in_rgb_band(waveform_view: WaveformView) -> None:
    """RGB legend visible ONLY in RGB_BAND, hidden in all other modes."""
    view = waveform_view
    _stash(view)

    expected = {
        RenderMode.CLASSIC: False,
        RenderMode.DB: False,
        RenderMode.ENERGY: False,
        RenderMode.SPECTROGRAM: False,
        RenderMode.CENTROID: False,
        RenderMode.RGB_BAND: True,
    }
    for mode, vis in expected.items():
        _select(view, mode)
        assert view._rgb_legend.isVisible() is vis, (
            f"rgb legend visibility wrong in {mode}"
        )


def test_full_mode_overlay_sweep(waveform_view: WaveformView) -> None:
    """One sweep over all 6 modes: each overlay shows ONLY in its mode."""
    view = waveform_view
    _stash(view, header=_FakeHeader(db_floor=-80.0, db_ref=0.0))
    left = view.waveform_plot.getAxis("left")

    def freq_axis_active() -> bool:
        y_min, y_max = view.waveform_plot.viewRange()[1]
        tv = left.tickValues(y_min, y_max, y_max - y_min)
        values = [v for _s, vals in tv for v in vals]
        if not values:
            return False
        return any(s for s in left.tickStrings(values, 1.0, 1.0))

    # (freq_axis, mag_bar, centroid_bar, rgb_legend)
    expected = {
        RenderMode.CLASSIC: (False, False, False, False),
        RenderMode.DB: (False, False, False, False),
        RenderMode.ENERGY: (False, False, False, False),
        RenderMode.SPECTROGRAM: (True, True, False, False),
        RenderMode.CENTROID: (False, False, True, False),
        RenderMode.RGB_BAND: (False, False, False, True),
    }
    for mode, (fx, mag, cen, rgb) in expected.items():
        _select(view, mode)
        assert freq_axis_active() is fx, f"freq axis wrong in {mode}"
        assert view._mag_colorbar.isVisible() is mag, f"mag bar wrong in {mode}"
        assert view._centroid_colorbar.isVisible() is cen, (
            f"centroid bar wrong in {mode}"
        )
        assert view._rgb_legend.isVisible() is rgb, f"rgb legend wrong in {mode}"


def test_rgb_legend_positioned_inside_viewbox(
    waveform_view: WaveformView,
) -> None:
    """Regression: the RGB legend must sit INSIDE the ViewBox PIXEL rect.

    The TextItem is a direct child of the ViewBox, so its pos is in ViewBox
    LOCAL PIXELS (the space GradientLegend anchors in) — NOT data coords. The
    original code positioned it in data coords (seconds × int16 amplitude),
    parking it ~32 000 px below the canvas: ``isVisible()`` was True but the
    legend never appeared. Assert it lands in the top-right pixel corner.
    """
    view = waveform_view
    _stash(view)
    _select(view, RenderMode.RGB_BAND)
    QApplication.processEvents()

    vb = view.waveform_plot.getViewBox()
    rect = vb.boundingRect()
    pos = view._rgb_legend.pos()

    # Inside the ViewBox's pixel rect…
    assert rect.left() <= pos.x() <= rect.right(), f"x={pos.x()} outside {rect}"
    assert rect.top() <= pos.y() <= rect.bottom(), f"y={pos.y()} outside {rect}"
    # …anchored BOTTOM-RIGHT (right half, bottom half)…
    assert pos.x() > rect.center().x(), "legend not in the right half"
    assert pos.y() > rect.center().y(), "legend not in the bottom half"
    # …in PIXELS, not the old data-coord parking spot (tens of thousands down).
    assert pos.y() <= rect.bottom(), f"y={pos.y()} looks like data coords"


def test_reset_render_mode_to_classic_drops_old_spectral(
    waveform_view: WaveformView,
) -> None:
    """Opening a new sound resets to CLASSIC and drops the old spectral data.

    Mirrors the new-file-open path: a file opened while SPECTROGRAM was active
    must not show the PREVIOUS file's stashed mel — the mode returns to CLASSIC,
    the spectral arrays are dropped, and every spectral overlay is hidden.
    """
    view = waveform_view
    _stash(view)
    _select(view, RenderMode.SPECTROGRAM)
    assert view._render_mode is RenderMode.SPECTROGRAM
    assert view._has_spectral_data()

    view.reset_render_mode_to_classic()

    assert view._render_mode is RenderMode.CLASSIC
    assert view.render_mode_combo.currentIndex() == list(RenderMode).index(
        RenderMode.CLASSIC
    )
    assert not view._has_spectral_data(), "stale spectral data not dropped"
    assert not view._spectro_img.isVisible()
    assert not view._color_backdrop_img.isVisible()
    assert not view._mag_colorbar.isVisible()
    assert not view._centroid_colorbar.isVisible()
    assert not view._rgb_legend.isVisible()


def test_rgb_legend_has_white_fill_and_black_border(
    waveform_view: WaveformView,
) -> None:
    """The RGB legend is a white box with a black border (matches the rest)."""
    view = waveform_view
    fill = view._rgb_legend.fill
    border = view._rgb_legend.border
    assert fill.color().getRgb()[:3] == (255, 255, 255), "fill not white"
    assert border.color().getRgb()[:3] == (0, 0, 0), "border not black"
