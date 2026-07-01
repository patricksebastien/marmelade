"""quick-260615-f77 — non-48 kHz sources are resampled to 48 kHz on open.

Pins the resample-on-open branch wired into ``MainWindow._open_file``:

* Opening a 44.1 kHz WAV produces a 48 kHz RF64 working file under
  ``<cache_root>/resampled48k/<key>.wav`` and rebinds playback to it
  (engine sample rate becomes 48000).
* Opening a 48 kHz WAV performs NO conversion (no working file created).
* Reopening the same 44.1 kHz source reuses the fresh working file
  (no double-convert).
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
from PySide6.QtWidgets import QApplication

from marmelade.audio import proxy_cache
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine

# NOTE: do NOT import default_cache_root at module level here — the
# tmp_cache_dir fixture monkeypatches ``main_window.default_cache_root``
# (the production caller) to the per-test cache root, so this test must
# read the cache root from the yielded ``tmp_cache_dir`` value instead.


def test_non_48k_source_resampled_to_48k_on_open(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """A 44.1 kHz WAV is converted to a 48 kHz RF64 working file on open."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "src_44100.wav"
    make_sine(
        src,
        freq_hz=440.0,
        amp=0.5,
        duration_s=1.0,
        sample_rate=44100,
        channels=2,
        fmt="wav",
    )

    window = MainWindow()
    qtbot.addWidget(window)

    render_done = {"v": False}
    window.render_complete.connect(lambda: render_done.update(v=True))
    window._open_file(str(src))
    qtbot.waitUntil(lambda: render_done["v"], timeout=15000)

    key = proxy_cache.cache_key(src)
    working = tmp_cache_dir / "resampled48k" / f"{key}.wav"
    assert working.is_file(), "48 kHz working file was not created"
    info = sf.info(str(working))
    assert info.samplerate == 48000
    assert "RF64" in info.format
    assert info.channels == 2

    # Playback rebound to the converted 48 kHz working file.
    assert window._playback_engine.sample_rate == 48000


def test_48k_source_not_converted_on_open(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """A 48 kHz WAV is used as-is — no working file is created."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "src_48000.wav"
    make_sine(
        src,
        freq_hz=440.0,
        amp=0.5,
        duration_s=1.0,
        sample_rate=48000,
        channels=2,
        fmt="wav",
    )

    window = MainWindow()
    qtbot.addWidget(window)

    render_done = {"v": False}
    window.render_complete.connect(lambda: render_done.update(v=True))
    window._open_file(str(src))
    qtbot.waitUntil(lambda: render_done["v"], timeout=15000)

    key = proxy_cache.cache_key(src)
    working = tmp_cache_dir / "resampled48k" / f"{key}.wav"
    assert not working.exists(), "48 kHz source must not be converted"
    assert window._playback_engine.sample_rate == 48000


def test_reopen_non_48k_reuses_working_file(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Reopening the same 44.1 kHz source reuses the fresh working file."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "src_reopen.wav"
    make_sine(
        src,
        freq_hz=440.0,
        amp=0.5,
        duration_s=1.0,
        sample_rate=44100,
        channels=2,
        fmt="wav",
    )

    window = MainWindow()
    qtbot.addWidget(window)

    render_done = {"v": False}
    window.render_complete.connect(lambda: render_done.update(v=True))

    window._open_file(str(src))
    qtbot.waitUntil(lambda: render_done["v"], timeout=15000)

    key = proxy_cache.cache_key(src)
    working = tmp_cache_dir / "resampled48k" / f"{key}.wav"
    assert working.is_file()
    mtime_first = working.stat().st_mtime_ns

    # Second open — the fresh working file must be reused (not rewritten).
    render_done["v"] = False
    window._open_file(str(src))
    qtbot.waitUntil(lambda: render_done["v"], timeout=15000)

    assert working.stat().st_mtime_ns == mtime_first, (
        "working file was reconverted on reopen (should be reused)"
    )
