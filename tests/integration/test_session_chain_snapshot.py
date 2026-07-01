"""Phase 7 Plan 07-03 Task 2 — MainWindow Mastering dock + snapshot semantics.

End-to-end pins:

* MainWindow constructs the Mastering dock in the LEFT dock area
  (quick-260621-dt4); Keepers is standalone on the RIGHT.
* New-keeper creation snapshots the current session chain into the
  keeper's ``Region.mastering`` field (D-04 — snapshot, not link).
* Editing the session chain AFTER a keeper is created does NOT mutate
  that keeper's mastering field (snapshot-not-link semantics).
* Legacy keepers (pre-Phase-7 sidecar JSONs with ``mastering=None``)
  auto-snapshot-and-persist exactly once on first load. Subsequent
  loads see the persisted value and skip the migration branch.
* ``session_chain_changed`` triggers per-keeper badge refresh — the
  badge state reflects ``config_hash(keeper.mastering) ==
  config_hash(session)`` for every existing keeper.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QDockWidget

from marmelade.audio.mastering.chain import (
    _SESSION_DEFAULTS,
    config_hash,
    load_session_chain_snapshot,
)
from marmelade.audio import sidecar_cache
from marmelade.audio.proxy_cache import cache_key
from marmelade.audio.sidecar_cache import Region, sidecar_path
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patches at module load
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from marmelade.ui.mastering_dock import MasteringDock


def _make_proxy_wav(tmp_path: Path, seconds: float = 1.0) -> Path:
    """Create a tiny 44.1 kHz stereo proxy WAV used to drive MainWindow."""
    sr = 44100
    n = int(seconds * sr)
    audio = (np.random.RandomState(0).randn(n, 2) * 0.05).astype("float32")
    p = tmp_path / "src.wav"
    sf.write(str(p), audio, sr, subtype="FLOAT", format="WAV")
    return p


# ----------------------------- Dock construction + View menu


def test_mastering_dock_is_left_keepers_standalone_right(
    qtbot, qapp, tmp_cache_dir: Path
) -> None:
    """MainWindow builds the Mastering dock on the LEFT; Keepers is standalone RIGHT.

    quick-260621-dt4 contract: the retired DSP/AI/Math Heatmaps panel left
    the LEFT dock area free, so the Mastering dock (formerly a tab sibling
    of Keepers on the right) now lives in the LEFT area as a standalone
    dock. Keepers stays in the RIGHT area with NO tab sibling.
    """
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)

    assert hasattr(window, "_mastering_dock"), "MainWindow must expose _mastering_dock"
    dock: QDockWidget = window._mastering_dock
    assert isinstance(dock, QDockWidget)
    assert dock.objectName() == "MasteringDock"
    assert isinstance(dock.widget(), MasteringDock)
    assert dock.isHidden() is False, (
        "Mastering dock must be VISIBLE on first launch"
    )
    assert (
        window.dockWidgetArea(dock) == Qt.DockWidgetArea.LeftDockWidgetArea
    ), "Mastering dock must be in the LEFT dock area"

    # Keepers is standalone on the RIGHT — no tab sibling.
    assert (
        window.dockWidgetArea(window._dock_keepers)
        == Qt.DockWidgetArea.RightDockWidgetArea
    )
    assert window.tabifiedDockWidgets(window._dock_keepers) == [], (
        "Keepers must be standalone (no tabify sibling)"
    )


# ----------------------------- Snapshot semantics (D-04)


def test_creation_snapshots_session_into_keeper_mastering(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """K-key keeper creation copies ``load_session_chain_snapshot()`` into the keeper.

    Workflow:
        1. Pre-seed Compressor=enabled in session QSettings.
        2. Open a WAV file in MainWindow.
        3. Add a draft region and mark it as keeper.
        4. The resulting sidecar Region.mastering equals the snapshot —
           NOT None.
    """
    theme.apply_theme(QApplication.instance())
    s = QSettings("Marmelade", "Marmelade")
    s.setValue("mastering/session/compressor/enabled", True)
    s.sync()

    src = _make_proxy_wav(tmp_path)

    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None
        and window._current_proxy_p is not None,
        timeout=15000,
    )

    # Add a draft region via the overlay's public draft-API + commit it.
    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None

    # Mark as keeper via the same slot the K key uses.
    overlay.set_state(region.id, "keeper")

    # Verify the sidecar has been re-saved AND the keeper's mastering
    # field is the session snapshot.
    regions = window._regions_overlay.regions_data()
    keeper = next((r for r in regions if r.id == region.id), None)
    assert keeper is not None
    assert keeper.mastering is not None, (
        "New keeper must inherit the session-chain snapshot (D-04)"
    )
    assert keeper.mastering["compressor"]["enabled"] is True, (
        "Snapshot must reflect the pre-seeded Compressor=enabled state"
    )

    # And the sidecar JSON on disk reflects the same.
    saved, _ = sidecar_cache.load_sidecar(window._current_sidecar_path)
    saved_keeper = next((r for r in saved if r.id == region.id), None)
    assert saved_keeper is not None
    assert saved_keeper.mastering is not None
    assert saved_keeper.mastering["compressor"]["enabled"] is True


def test_session_edit_does_not_mutate_existing_keeper(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """D-04 snapshot-not-link — session edits do NOT touch existing keepers.

    Workflow:
        1. Open a file, create a keeper while session has Compressor=False.
        2. Verify keeper.mastering[compressor].enabled is False.
        3. Toggle Compressor=True in the dock (signal → no keeper mutation).
        4. Reload the sidecar from disk; keeper.mastering[compressor].enabled
           is STILL False.
    """
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)

    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None,
        timeout=15000,
    )

    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None
    overlay.set_state(region.id, "keeper")

    keeper_before = next(
        r for r in overlay.regions_data() if r.id == region.id
    )
    assert keeper_before.mastering["compressor"]["enabled"] is False

    # Now toggle the session via the MasteringDock — emits
    # session_chain_changed; D-04 requires no keeper mutation.
    window._mastering_widget.stage_checkbox("compressor").setChecked(True)

    # Reload from disk — keeper still has compressor disabled.
    reloaded, _ = sidecar_cache.load_sidecar(window._current_sidecar_path)
    keeper_after = next(r for r in reloaded if r.id == region.id)
    assert keeper_after.mastering["compressor"]["enabled"] is False, (
        "D-04 snapshot-not-link: session edits must not mutate existing keepers"
    )


def test_session_chain_changed_refreshes_keeper_badges(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Toggling session chain → existing keeper's badge flips check ↔ star.

    Setup: keeper A is created while session is at defaults; its
    config_hash matches the session, badge is "check". Toggling
    Compressor on flips the session hash → A's hash now differs → badge
    transitions to "star".
    """
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)

    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None,
        timeout=15000,
    )

    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None
    overlay.set_state(region.id, "keeper")

    row = window._keepers_sidebar.find_row(region.id)
    assert row is not None
    # Initial state — keeper used the session snapshot, hashes match.
    assert row._badge_state == "check", (
        f"keeper using session chain must start at 'check' "
        f"(got {row._badge_state!r})"
    )

    # Toggle session Compressor → emits session_chain_changed →
    # MainWindow slot recomputes the badge state for each keeper.
    window._mastering_widget.stage_checkbox("compressor").setChecked(True)

    # Drive Qt's event loop once so the queued slot runs.
    QApplication.processEvents()

    assert row._badge_state == "star", (
        f"after session diverges, keeper badge must transition to 'star' "
        f"(got {row._badge_state!r})"
    )


# ----------------------------- Legacy migration


def _write_legacy_sidecar(
    sidecar: Path, keeper_id: str, start: float, end: float
) -> None:
    """Write a pre-Phase-7 sidecar JSON (no mastering field at all).

    Mirrors the on-disk shape Phase 3 left behind: the ``mastering``
    key is omitted entirely (loading produces ``mastering=None``).
    """
    payload = {
        "schema_version": 1,
        "regions": [
            {
                "id": keeper_id,
                "start_sec": start,
                "end_sec": end,
                "state": "keeper",
                "created_at": "2026-05-19T00:00:00",
                "note": "",
            }
        ],
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_legacy_keeper_with_mastering_none_auto_migrates_on_load(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Loading a legacy sidecar auto-snapshots + persists the session chain.

    The keeper had ``mastering=None``; after the file open completes,
    the in-memory keeper's mastering field is non-None AND the sidecar
    file on disk has been rewritten with the populated mastering field
    (so subsequent loads skip the migration branch entirely).
    """
    theme.apply_theme(QApplication.instance())
    # Pre-seed Compressor=enabled so the snapshot is distinguishable
    # from a stale "all defaults" fallback.
    s = QSettings("Marmelade", "Marmelade")
    s.setValue("mastering/session/compressor/enabled", True)
    s.sync()

    src = _make_proxy_wav(tmp_path)
    # Compute the sidecar path the MainWindow will derive on open.
    src_key = cache_key(src)
    sp = sidecar_path(tmp_cache_dir, src_key)
    keeper_id = "0123456789abcdef0123456789abcdef"
    _write_legacy_sidecar(sp, keeper_id, 0.1, 0.5)

    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None,
        timeout=15000,
    )

    # In-memory keeper has mastering populated.
    region = window._regions_overlay.get_region(keeper_id)
    assert region is not None
    in_mem_mastering = window._regions_overlay.get_mastering(keeper_id)
    assert in_mem_mastering is not None
    assert in_mem_mastering["compressor"]["enabled"] is True

    # The migration must have re-persisted the sidecar to disk.
    saved, _ = sidecar_cache.load_sidecar(sp)
    saved_keeper = next((r for r in saved if r.id == keeper_id), None)
    assert saved_keeper is not None
    assert saved_keeper.mastering is not None
    assert saved_keeper.mastering["compressor"]["enabled"] is True


def test_session_edit_after_load_does_not_remigrate_existing_legacy_keeper(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """D-04 snapshot-not-link holds even for migrated legacy keepers.

    After the first load auto-snapshots a legacy keeper, subsequent
    session-chain edits must NOT re-snapshot. The keeper's mastering
    field is now its own independent snapshot.
    """
    theme.apply_theme(QApplication.instance())
    # Session starts with Compressor=enabled at load time.
    s = QSettings("Marmelade", "Marmelade")
    s.setValue("mastering/session/compressor/enabled", True)
    s.sync()

    src = _make_proxy_wav(tmp_path)
    src_key = cache_key(src)
    sp = sidecar_path(tmp_cache_dir, src_key)
    keeper_id = "abcdef0123456789abcdef0123456789"
    _write_legacy_sidecar(sp, keeper_id, 0.1, 0.5)

    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None,
        timeout=15000,
    )

    # Edit session chain AFTER load — flip Compressor off.
    window._mastering_widget.stage_checkbox("compressor").setChecked(False)
    QApplication.processEvents()

    # Reload from disk — the migrated keeper still has Compressor=True
    # (its mastering field was captured at LOAD time and is independent).
    saved, _ = sidecar_cache.load_sidecar(sp)
    saved_keeper = next((r for r in saved if r.id == keeper_id), None)
    assert saved_keeper is not None
    assert saved_keeper.mastering is not None
    assert saved_keeper.mastering["compressor"]["enabled"] is True, (
        "Migration runs ONCE per legacy keeper — session edits never "
        "retroactively change it (D-04 holds even after migration)"
    )
