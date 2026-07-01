"""Live-seam MainWindow integration test for the Markers feature.

quick-260701-jc5 Task 3 (MARK-01..MARK-05 wiring). Per project memory
(Phase 9 integration gap — unit-only tests let unwired features ship), this
drives the REAL MainWindow: it opens a small fixture, primes the playback +
sidecar paths exactly as a live session would, then asserts the panel row +
overlay line + sidecar persistence move together.

Assertions:
    1. Activating the "m" shortcut adds a panel row AND an overlay line AND
       writes the marker to the sidecar (load it back, assert present).
    2. Clicking the [+] Add-marker button does the same.
    3. Editing a MarkerRow label persists to the sidecar AND updates the
       overlay line's label.
    4. Deleting a marker removes the row, the overlay line, AND the sidecar
       entry.
    5. Jumping (row click) seeks + plays (spy on engine.seek / engine.play).
    6. Reopening the same file's sidecar repopulates panel + overlay.
    7. "m" is a no-op while a QLineEdit has focus.

Runs under QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication, QLineEdit

from marmelade.audio import sidecar_cache
from marmelade.audio.sidecar_cache import Marker, load_sidecar
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


def _make_proxy_wav(tmp_path: Path, seconds: float = 3.0) -> Path:
    sr = 48000
    n = int(seconds * sr)
    audio = (np.random.RandomState(0).randn(n, 2) * 0.05).astype("float32")
    p = tmp_path / "proxy.wav"
    sf.write(str(p), audio, sr, subtype="FLOAT", format="WAV")
    return p


@pytest.fixture
def window_open(qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path):
    """Open a file so _current_playback_path + _current_sidecar_path are set."""
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None
        and window._current_playback_path is not None,
        timeout=15000,
    )
    return window


# =========================================================================
# 0 — construction: the Markers dock exists, sits below Keepers
# =========================================================================
def test_markers_dock_exists_below_keepers(window_open, qtbot) -> None:
    window = window_open
    assert hasattr(window, "_dock_markers")
    assert hasattr(window, "_markers_sidebar")
    assert hasattr(window, "_markers_overlay")
    # The [+] button is enabled once a file is open.
    assert window._markers_sidebar._add_button.isEnabled() is True


# =========================================================================
# 1 — the "m" shortcut adds row + overlay line + sidecar entry
# =========================================================================
def test_m_shortcut_adds_marker_everywhere(window_open, qtbot, monkeypatch) -> None:
    window = window_open
    # Drive a known live playhead position.
    monkeypatch.setattr(
        type(window._playback_engine), "position_seconds",
        property(lambda self: 1.25),
    )

    assert window._markers_sidebar.marker_count() == 0
    assert window._markers_overlay.marker_count() == 0

    window._action_add_marker()

    assert window._markers_sidebar.marker_count() == 1
    assert window._markers_overlay.marker_count() == 1

    # Sidecar persistence — load it back off disk.
    _regions, markers = load_sidecar(window._current_sidecar_path)
    assert len(markers) == 1
    assert markers[0].time_sec == pytest.approx(1.25, abs=1e-6)

    # The new marker's label field grabs focus so the user can type at once.
    # Use focusWidget() (not hasFocus) so the assertion holds under the
    # offscreen platform where the top-level window is never "active".
    mid = next(iter(window._markers_sidebar._rows.keys()))
    row = window._markers_sidebar.find_row(mid)
    assert window._markers_sidebar.focusWidget() is row._label_edit


# =========================================================================
# 2 — the [+] button adds a marker at the same source
# =========================================================================
def test_plus_button_adds_marker(window_open, qtbot, monkeypatch) -> None:
    window = window_open
    monkeypatch.setattr(
        type(window._playback_engine), "position_seconds",
        property(lambda self: 2.0),
    )
    window._markers_sidebar._add_button.click()

    assert window._markers_sidebar.marker_count() == 1
    assert window._markers_overlay.marker_count() == 1
    _regions, markers = load_sidecar(window._current_sidecar_path)
    assert len(markers) == 1
    assert markers[0].time_sec == pytest.approx(2.0, abs=1e-6)


# =========================================================================
# 3 — editing a label persists + updates the overlay line label
# =========================================================================
def test_edit_label_persists_and_updates_overlay(
    window_open, qtbot, monkeypatch
) -> None:
    window = window_open
    monkeypatch.setattr(
        type(window._playback_engine), "position_seconds",
        property(lambda self: 0.5),
    )
    window._action_add_marker()
    mid = next(iter(window._markers_sidebar._rows.keys()))
    row = window._markers_sidebar.find_row(mid)

    row._label_edit.setText("verse")
    row._label_edit.editingFinished.emit()

    # Overlay line label updated.
    line = window._markers_overlay._markers[mid]
    assert line.label.format == "verse"
    # Sidecar persisted.
    _regions, markers = load_sidecar(window._current_sidecar_path)
    assert markers[0].label == "verse"


# =========================================================================
# 4 — delete removes row + overlay line + sidecar entry
# =========================================================================
def test_delete_removes_everywhere(window_open, qtbot, monkeypatch) -> None:
    window = window_open
    monkeypatch.setattr(
        type(window._playback_engine), "position_seconds",
        property(lambda self: 1.0),
    )
    window._action_add_marker()
    mid = next(iter(window._markers_sidebar._rows.keys()))

    window._on_marker_delete(mid)

    assert window._markers_sidebar.marker_count() == 0
    assert window._markers_overlay.marker_count() == 0
    _regions, markers = load_sidecar(window._current_sidecar_path)
    assert markers == []


# =========================================================================
# 5 — jump seeks + plays
# =========================================================================
def test_jump_seeks_and_plays(window_open, qtbot, monkeypatch) -> None:
    window = window_open
    monkeypatch.setattr(
        type(window._playback_engine), "position_seconds",
        property(lambda self: 1.75),
    )
    window._action_add_marker()
    mid = next(iter(window._markers_sidebar._rows.keys()))

    seek_spy = MagicMock()
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "seek", seek_spy)
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    window._on_marker_jump(mid)

    assert seek_spy.called, "jump must seek"
    assert seek_spy.call_args.args[0] == pytest.approx(1.75, abs=1e-6)
    assert play_spy.called, "jump must play"


def test_play_button_seeks_and_plays(window_open, qtbot, monkeypatch) -> None:
    """The per-row Play button (before the timecode) seeks + plays at the
    marker — same path as the row-body jump."""
    window = window_open
    monkeypatch.setattr(
        type(window._playback_engine), "position_seconds",
        property(lambda self: 2.5),
    )
    window._action_add_marker()
    mid = next(iter(window._markers_sidebar._rows.keys()))
    row = window._markers_sidebar.find_row(mid)

    seek_spy = MagicMock()
    play_spy = MagicMock()
    monkeypatch.setattr(window._playback_engine, "seek", seek_spy)
    monkeypatch.setattr(window._playback_engine, "play", play_spy)

    row._play.click()

    assert seek_spy.called, "play button must seek"
    assert seek_spy.call_args.args[0] == pytest.approx(2.5, abs=1e-6)
    assert play_spy.called, "play button must play"


# =========================================================================
# 6 — reopening the sidecar repopulates panel + overlay
# =========================================================================
def test_reload_repopulates_panel_and_overlay(
    window_open, qtbot, monkeypatch
) -> None:
    window = window_open
    sp = window._current_sidecar_path
    # Write a sidecar with two markers directly, then re-load via the
    # MainWindow's own load path.
    sidecar_cache.save_sidecar(
        sp,
        [],
        [
            Marker(id="m1", time_sec=0.3, label="a"),
            Marker(id="m2", time_sec=2.5, label="b"),
        ],
    )
    # Clear the live panel/overlay, then drive the real reload seam.
    window._markers_sidebar.clear()
    window._markers_overlay.clear()
    window._current_markers = []

    # Re-run the sidecar-for-key load (the seam file-open uses). Derive the
    # cache root from the live sidecar path (sidecars live at
    # ``cache_root/sidecars/{key}.json``) so this matches the root the window
    # used at open time regardless of the test-module import binding.
    key = window._current_cache_key
    cache_root = sp.parent.parent
    window._load_sidecar_for_key(cache_root, key)

    assert window._markers_sidebar.marker_count() == 2
    assert window._markers_overlay.marker_count() == 2


# =========================================================================
# 7 — "m" is a no-op while a QLineEdit has focus
# =========================================================================
def test_m_noop_while_lineedit_focused(window_open, qtbot, monkeypatch) -> None:
    window = window_open
    monkeypatch.setattr(
        type(window._playback_engine), "position_seconds",
        property(lambda self: 1.0),
    )
    # Force the focus widget to be a QLineEdit.
    le = QLineEdit(window)
    monkeypatch.setattr(QApplication, "focusWidget", staticmethod(lambda: le))

    window._action_add_marker()

    assert window._markers_sidebar.marker_count() == 0
    assert window._markers_overlay.marker_count() == 0


# =========================================================================
# 8 — opening a new file clears the previous file's markers
# =========================================================================
def test_new_file_clears_markers(window_open, qtbot, monkeypatch, tmp_path) -> None:
    window = window_open
    monkeypatch.setattr(
        type(window._playback_engine), "position_seconds",
        property(lambda self: 1.0),
    )
    window._action_add_marker()
    assert window._markers_sidebar.marker_count() == 1

    # Build a distinct second file.
    d2 = tmp_path / "second"
    d2.mkdir(parents=True, exist_ok=True)
    src2 = _make_proxy_wav(d2, seconds=2.0)
    window._open_file(str(src2))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None
        and window._current_playback_path is not None,
        timeout=15000,
    )
    assert window._markers_sidebar.marker_count() == 0
    assert window._markers_overlay.marker_count() == 0
