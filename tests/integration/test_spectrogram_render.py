"""RED scaffold — R-4 spectrogram render (region differentiation + y-orientation).

Phase 11 Wave 0 (plan 11-01). PINs the not-yet-existing spectrogram render path
on :class:`marmelade.ui.waveform_view.WaveformView`:

    * ``WaveformView.set_spectral_data(mel, centroid, bands, header)`` stashes
      precomputed spectral arrays for the spectral render modes.
    * Selecting ``RenderMode.SPECTROGRAM`` shows an ``ImageItem`` (the mel image)
      and hides the line ``PlotDataItem``.
    * The rendered mel visibly DIFFERENTIATES a silence->sweep->noise fixture:
      low intensity in the silence columns, a rising diagonal ridge in the
      sweep, broadband fill in the noise.
    * Y-orientation (Pitfall #4 — pyqtgraph imageAxisOrder row-major): a
      low-frequency-only segment lights the BOTTOM mel rows (row 0).

The mel is synthesised directly here (a hand-built mel-like array) so the render
test does NOT depend on the builder wave — it only exercises set_spectral_data +
the SPECTROGRAM transform. RED until the render path lands.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from marmelade.ui import theme
from marmelade.ui.waveform_view import WaveformView


@pytest.fixture
def waveform_view(qtbot, qapp):
    theme.apply_theme(QApplication.instance())
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(800, 200)
    return view


def _silence_sweep_noise_mel(n_mels: int = 64, seg_frames: int = 60) -> np.ndarray:
    """Hand-build a (n_mels, 3*seg_frames) mel-like array, row-major (row0=low f).

    * silence  — near-zero everywhere.
    * sweep    — a single hot bin per column rising from row 0 to row n_mels-1.
    * noise    — broadband fill (all rows hot).
    """
    rng = np.random.default_rng(0)
    silence = rng.random((n_mels, seg_frames)).astype(np.float32) * 0.01
    sweep = np.zeros((n_mels, seg_frames), dtype=np.float32)
    for c in range(seg_frames):
        row = int(round((n_mels - 1) * c / max(seg_frames - 1, 1)))
        sweep[row, c] = 1.0
    noise = (0.6 + 0.4 * rng.random((n_mels, seg_frames))).astype(np.float32)
    return np.concatenate([silence, sweep, noise], axis=1), seg_frames


def _set_spectral(view: WaveformView, mel: np.ndarray) -> None:
    # header/centroid/bands are optional for the pure spectrogram path; pass
    # None where the production signature allows it.
    view.set_spectral_data(mel, None, None, None)


def test_spectrogram_shows_image_hides_line(waveform_view: WaveformView) -> None:
    """R-4: SPECTROGRAM mode shows an ImageItem and hides the line item."""
    import pyqtgraph as pg
    from marmelade.audio.render_modes import RenderMode

    mel, _seg = _silence_sweep_noise_mel()
    _set_spectral(waveform_view, mel)
    waveform_view.render_mode_combo.setCurrentIndex(
        list(RenderMode).index(RenderMode.SPECTROGRAM)
    )
    QApplication.processEvents()

    plot = waveform_view.waveform_plot
    images = [it for it in plot.items if isinstance(it, pg.ImageItem)]
    assert images, "SPECTROGRAM mode must add a pyqtgraph ImageItem"
    assert any(it.isVisible() for it in images), "the mel ImageItem must be visible"


def test_spectrogram_differentiates_regions(waveform_view: WaveformView) -> None:
    """R-4: silence / sweep / noise columns are visibly distinct in the mel image."""
    from marmelade.audio.render_modes import RenderMode

    mel, seg = _silence_sweep_noise_mel()
    _set_spectral(waveform_view, mel)
    waveform_view.render_mode_combo.setCurrentIndex(
        list(RenderMode).index(RenderMode.SPECTROGRAM)
    )
    QApplication.processEvents()

    rendered = np.asarray(waveform_view._rendered_spectral_image, dtype=np.float64)
    # Image is (n_mels, n_cols) row-major. Per-column mean intensity.
    col_mean = rendered.mean(axis=0)
    n_cols = col_mean.shape[0]
    third = n_cols // 3
    silence_mean = float(col_mean[:third].mean())
    sweep_mean = float(col_mean[third : 2 * third].mean())
    noise_mean = float(col_mean[2 * third :].mean())

    assert silence_mean < sweep_mean, "silence must be dimmer than the sweep"
    assert sweep_mean < noise_mean, "broadband noise must be brighter than the sweep"


def test_spectrogram_y_orientation_low_freq_bottom(waveform_view: WaveformView) -> None:
    """R-4 / Pitfall #4: a low-freq-only segment lights the BOTTOM mel rows.

    Build a mel where only the lowest few rows are hot; assert the rendered
    image's bottom rows (index 0..k) carry the energy, confirming row-major
    imageAxisOrder with row 0 = low frequency at the bottom.
    """
    from marmelade.audio.render_modes import RenderMode

    n_mels, n_cols = 64, 120
    mel = np.full((n_mels, n_cols), 0.01, dtype=np.float32)
    mel[:6, :] = 1.0  # only the lowest 6 mel rows hot
    _set_spectral(waveform_view, mel)
    waveform_view.render_mode_combo.setCurrentIndex(
        list(RenderMode).index(RenderMode.SPECTROGRAM)
    )
    QApplication.processEvents()

    rendered = np.asarray(waveform_view._rendered_spectral_image, dtype=np.float64)
    row_mean = rendered.mean(axis=1)
    bottom_energy = float(row_mean[:6].mean())
    top_energy = float(row_mean[6:].mean())
    assert bottom_energy > top_energy, (
        "low-frequency energy did not land in the bottom mel rows "
        "(imageAxisOrder y-orientation, Pitfall #4)"
    )
