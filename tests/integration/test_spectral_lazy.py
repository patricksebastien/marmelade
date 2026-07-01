"""RED scaffold — R-3 lazy spectral build + cancel + cache-hit (qtbot).

Phase 11 Wave 0 (plan 11-01). The spectral lanes are EXPENSIVE (FFT/STFT
precompute), so they must be LAZY: opening a file does NO spectral work; the
build is triggered only when the user selects a spectral render mode, runs in
the background (UI responsive), is cancellable (no partial .dat left), and a
second selection after a completed build is a cache HIT (no rebuild).

PINs the not-yet-existing lazy-build seam:
    * ``WaveformView.spectral_build_requested`` signal
    * MainWindow ``_spawn_spectral_worker`` (analogous to
      ``_spawn_audio_proxy_worker``)
    * ``marmelade.audio.spectral_cache`` on-disk layout

RED until the lazy spectral path lands (plans 11-02..11-04). Each production
import lives inside its test so the module COLLECTS cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog

from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow

# tests/integration imports the synth fixtures via the same path discipline as
# its siblings (tests/ is on sys.path under pytest rootdir).
from tests.fixtures.synthesize import make_sine

# Importing default_cache_root at module level so the tmp_cache_dir fixture can
# redirect spectral writes through the per-test cache directory (conftest
# Pitfall #10 discipline — the conftest patch list references this binding).
from marmelade.paths import default_cache_root  # noqa: F401


@pytest.fixture
def main_window(qtbot, qapp, tmp_cache_dir: Path):
    from PySide6.QtWidgets import QApplication

    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    return window


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    return make_sine(tmp_path / "lazy.wav", freq_hz=440.0, duration_s=3.0, sample_rate=48000)


def _open(main_window, qtbot, monkeypatch, path: Path) -> None:
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **kw: (str(path), "Audio files (*.wav *.flac *.mp3)")),
    )
    with qtbot.waitSignal(main_window.render_complete, timeout=15000, raising=True):
        main_window._action_open_file()


def _spectra_dir(cache_root: Path) -> Path:
    # The spectral cache lives under a dedicated subtree, mirroring the
    # heatmaps/ layout of heatmap_cache.
    return Path(cache_root) / "spectra"


def test_open_no_spectral(
    main_window: MainWindow, qtbot, monkeypatch, sine_wav: Path, tmp_cache_dir: Path
) -> None:
    """R-3: opening a file performs NO spectral work (lazy)."""
    _open(main_window, qtbot, monkeypatch, sine_wav)

    spectra = _spectra_dir(tmp_cache_dir)
    dats = list(spectra.rglob("*.dat")) if spectra.exists() else []
    assert dats == [], f"opening a file wrote spectral data eagerly: {dats}"

    # And no spectral worker should be running just from opening.
    assert not getattr(main_window, "_spectral_worker", None), (
        "a spectral worker was spawned on open — must be lazy (R-3)"
    )


def test_spectral_build_cancellable(
    main_window: MainWindow, qtbot, monkeypatch, sine_wav: Path, tmp_cache_dir: Path
) -> None:
    """R-3: selecting a spectral mode on a cold cache spawns a cancellable build.

    Drives the lazy seam: switching to a spectral render mode emits
    ``spectral_build_requested``; MainWindow spawns a background worker; the user
    can cancel; a cancelled build leaves no mel.dat / mel.dat.tmp.
    """
    from marmelade.audio.render_modes import RenderMode  # noqa: F401

    _open(main_window, qtbot, monkeypatch, sine_wav)
    view = main_window._waveform_view

    # The lazy trigger: the view must expose a spectral_build_requested signal.
    signal = view.spectral_build_requested  # AttributeError -> RED today

    spawned = {"started": False}
    signal.connect(lambda *a, **k: spawned.__setitem__("started", True))

    # Select a spectral mode -> should request a build.
    from marmelade.audio.render_modes import RenderMode as RM

    idx = list(RM).index(RM.SPECTROGRAM)
    view.render_mode_combo.setCurrentIndex(idx)
    qtbot.wait(50)
    assert spawned["started"], "selecting a spectral mode did not request a build (R-3)"

    # Cancel the in-flight build via MainWindow's cancel entry point.
    main_window._cancel_spectral_build()
    qtbot.wait(50)

    spectra = _spectra_dir(tmp_cache_dir)
    leftovers = list(spectra.rglob("*.dat")) + list(spectra.rglob("*.tmp")) if spectra.exists() else []
    assert leftovers == [], f"cancelled spectral build left partial files: {leftovers}"


def test_spectral_cache_hit_no_rebuild(
    main_window: MainWindow, qtbot, monkeypatch, sine_wav: Path, tmp_cache_dir: Path
) -> None:
    """R-3: re-selecting a spectral mode after a completed build is a cache HIT.

    After one completed build, re-selecting the same mode must NOT spawn a new
    worker — it renders from the on-disk cache / in-memory stash. We assert no
    second ``spectral_build_requested`` triggers an actual worker spawn.
    """
    _open(main_window, qtbot, monkeypatch, sine_wav)
    view = main_window._waveform_view
    from marmelade.audio.render_modes import RenderMode as RM

    spectro_idx = list(RM).index(RM.SPECTROGRAM)
    classic_idx = list(RM).index(RM.CLASSIC)

    # First selection -> build completes.
    with qtbot.waitSignal(main_window.spectral_build_complete, timeout=15000, raising=True):
        view.render_mode_combo.setCurrentIndex(spectro_idx)

    spawn_count = {"n": 0}
    orig = main_window._spawn_spectral_worker

    def counting_spawn(*a, **k):
        spawn_count["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(main_window, "_spawn_spectral_worker", counting_spawn)

    # Toggle away and back -> must be served from cache, no new worker.
    view.render_mode_combo.setCurrentIndex(classic_idx)
    view.render_mode_combo.setCurrentIndex(spectro_idx)
    qtbot.wait(100)
    assert spawn_count["n"] == 0, "re-selecting spectral mode rebuilt instead of cache HIT (R-3)"


def test_rebuild_spectral_cache_deletes_and_recreates(
    main_window: MainWindow, qtbot, monkeypatch, sine_wav: Path, tmp_cache_dir: Path
) -> None:
    """quick-260629: View → Rebuild spectrogram deletes + recomputes the cache.

    After a completed build the on-disk .dat siblings exist; triggering the
    rebuild action deletes them and re-runs the background build, leaving a
    fresh cache (and emitting spectral_build_complete on completion).
    """
    from marmelade.audio.render_modes import RenderMode as RM

    _open(main_window, qtbot, monkeypatch, sine_wav)
    view = main_window._waveform_view
    spectro_idx = list(RM).index(RM.SPECTROGRAM)

    # First build → cache populated.
    with qtbot.waitSignal(
        main_window.spectral_build_complete, timeout=15000, raising=True
    ):
        view.render_mode_combo.setCurrentIndex(spectro_idx)
    spectra = _spectra_dir(tmp_cache_dir)
    assert "mel.dat" in {p.name for p in spectra.rglob("*.dat")}

    # The View → Rebuild spectrogram action is enabled once a file is open.
    assert main_window._action_rebuild_spectral.isEnabled()

    # Trigger the rebuild → deletes then rebuilds; wait for the fresh build.
    with qtbot.waitSignal(
        main_window.spectral_build_complete, timeout=15000, raising=True
    ):
        main_window._action_rebuild_spectral.trigger()

    # The cache was recreated (a true rebuild, not just left in place).
    assert "mel.dat" in {p.name for p in spectra.rglob("*.dat")}, (
        "rebuild did not recreate the spectral cache"
    )
