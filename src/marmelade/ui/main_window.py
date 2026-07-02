"""QMainWindow shell — menu, toolbar, dock, status bar, central waveform view.

Plan 03 (this revision) wires :attr:`MainWindow.file_open_requested` (Plan 01)
to the audio backbone (Plan 02) via :class:`PeakBuilderRunnable` on
:class:`QThreadPool`. The Architectural Responsibility Map locks:

* GUI thread: probe, duration check, cache lookup, load_proxy (sync memmap),
  render_proxy, dialog display, ProgressOverlay management.
* Worker thread (QThreadPool.globalInstance()): peak_builder.build_proxy.
* Cross-thread bridge: WorkerSignals queued signal delivery only — NEVER
  direct widget mutation from the worker (RESEARCH Pitfall #4).

N-3: every writable-location lookup goes through ``marmelade.paths`` —
:func:`default_cache_root` for the proxy cache and :func:`default_open_dir`
for the file-open initial directory. The GUI tier does NOT import any
toolkit-level path utility directly (W-5 / N-3 single source of truth).

Test seam — :attr:`render_complete` signal fires exactly once per
successful open (cache HIT in :meth:`_open_file`, or cache MISS in
:meth:`_on_proxy_ready`). The signal has NO ``_for_test`` suffix
(W-6 fix) — production code is welcome to subscribe to it (a future
autosave-zoom-state feature could).
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6.QtCore import (
    QDeadlineTimer,
    QEventLoop,
    QSettings,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeySequence, QResizeEvent, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStatusBar,
    QStyle,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from marmelade.audio import audio_file, naming_resolver, proxy_cache
from marmelade.audio.render_modes import RenderMode
from marmelade.audio.audio_proxy_cache import (
    audio_cache_size_bytes,
    audio_proxy_is_fresh,
    audio_proxy_path,
    cache_key,
    check_disk_space,
    clear_audio_cache,
    expected_proxy_bytes,
)
from marmelade.audio.audio_proxy_worker import AudioProxyRunnable
from marmelade.audio.spectral_builder_worker import SpectralProxyRunnable
from marmelade.audio import spectral_cache
from marmelade.audio.export_worker import ExportRunnable
from marmelade.audio.resample_to_48k import (
    CANONICAL_SAMPLE_RATE,
    resample_to_48k,
)
from marmelade.concurrency.worker import WorkerSignals
from marmelade.audio.peak_builder_worker import PeakBuilderRunnable
from marmelade.audio.playback import PlaybackEngine, PlaybackError
from marmelade.audio.proxy_cache import ProxyHeaderError
from marmelade.audio import sidecar_cache
from marmelade.audio.sidecar_cache import Marker, Region, sidecar_path
from marmelade.paths import (
    default_cache_root,
    default_open_dir,
    matchering_reference_dir,
)
from marmelade.ui.dialogs import (
    show_corrupt_file,
    show_too_long,
    show_unsupported_format,
)
from marmelade.ui.icons import (
    _fit_to_view_icon,
    _follow_playhead_icon,
    _region_select_icon,
    _zoom_in_icon,
    _zoom_out_icon,
)
from marmelade.ui.keepers_sidebar import KeepersSidebar
from marmelade.ui.markers_sidebar import MarkersSidebar
from marmelade.ui.ab_toggle import ABToggleWidget
from marmelade.ui.mastering_dialog import MasteringDialog
from marmelade.ui.mastering_dock import MasteringDock
from marmelade.ui.settings_dialog import SettingsDialog
from marmelade.youtube import oauth as _yt_oauth
from marmelade.audio.mastering.chain import (
    config_hash,
    load_session_chain_snapshot,
)
# quick-260626-o9y — output-time fade is now config-driven (enabled + duration
# from the keeper's mastering config) instead of a forced 2.0 s. fade_params is
# the single source of truth for the (enabled, duration_sec) read at the 4
# export/preview fade sites below.
from marmelade.audio.mastering.stages.fade import fade_params
from marmelade.audio.mastering_cache import (
    is_mastered_cache_fresh,
    mastered_cache_path,
)
from marmelade.audio.mastering_worker import MasteringRunnable
from marmelade.audio.export_builder import export_region
from marmelade.audio.bundle_builder import build_bundle, BuildCancelled
from marmelade.youtube.video_builder import build_video
from marmelade.ui.progress_overlay import ProgressOverlay
from marmelade.ui.regions_overlay import MarkersOverlay, RegionsOverlay
from marmelade.ui.bundle_dialog import BundleDialog
from marmelade.ui.upload_dialog import UploadDialog
from marmelade.ui.waveform_view import WaveformView
from marmelade.util import poem_generator as _poem_generator
from marmelade.youtube import thumbnail_provider as _thumbnail_provider
from marmelade.youtube.upload_runnable import YouTubeUploadRunnable

logger = logging.getLogger(__name__)


_SUPPORTED_EXTS = {".wav", ".flac", ".mp3"}


# quick-260625 — playhead visual sync trim (seconds).
#
# ``engine.position_seconds`` already reports the AUDIBLE playback position
# (it subtracts the audio output latency). But the VISUAL playhead is updated
# by the 30 Hz poll timer and then PyQtGraph has to repaint, so the drawn line
# lags the sound by the GUI/render pipeline delay. On systems where that delay
# exceeds the audio latency the playhead trails the sound ("the sound plays
# before the playhead reaches it").
#
# This constant draws the playhead this many seconds AHEAD of the true audible
# position to cancel that pipeline lag — equivalent to nudging the waveform
# left under the playhead so the sounding feature sits under the line. POSITIVE
# moves the playhead forward (right); raise it if the sound still leads the
# playhead, lower it (toward 0, or negative) if the playhead now leads the
# sound. Purely cosmetic: it never affects audio, seek targets, or export.
#
# Tuning log: 0.08 s still left the sound leading the playhead on the dev
# machine (likely the audio server buffers more than PortAudio reports), so
# the default was raised to 0.15 s.
#
# This is only the DEFAULT — the live value is editable in Preferences and
# persisted in QSettings under _PLAYHEAD_OFFSET_SETTINGS_KEY.
_PLAYHEAD_VISUAL_OFFSET_SEC = 0.15
_PLAYHEAD_OFFSET_SETTINGS_KEY = "playback/playhead_visual_offset_sec"


def _keeper_play_offsets(
    start_sec: float, end_sec: float, mode: str
) -> tuple[float, bool]:
    """Pure start-offset + fade-in-suppression computation for keeper play
    (quick-260622-sr8). Single source of truth for the three keeper play
    buttons (Play / middle / end).

    Plain float math — imports no Qt — so it stays unit-testable in
    isolation. The fade_sec auto-scaling (``min(2.0, region_duration/2)``)
    stays at the CALL site (``_on_keeper_play``), not here, so this helper
    is a pure offset/flag computation.

    Args:
        start_sec: keeper start (source-time seconds).
        end_sec: keeper end (source-time seconds).
        mode: one of ``"start"`` / ``"middle"`` / ``"end"``. Any
            unrecognized value degrades to ``"start"`` (T-sr8-02 — a
            future caller typo never crashes, just plays from the start).

    Returns:
        ``(start_seconds, suppress_fade_in)`` — ``suppress_fade_in`` is True
        for BOTH ``"middle"`` AND ``"end"``; only ``"start"`` gets a fade-in.
        The fade-OUT is applied for every mode by the caller regardless of
        this flag (quick-260622-ud0 — the engine now does asymmetric per-end
        fades, so suppressing the fade-in no longer drops the fade-out).
          * ``"start"`` → ``(start_sec, False)`` — fade-in AND fade-out apply.
          * ``"middle"`` → ``(start_sec + (end_sec - start_sec) / 2, True)``
            — start at the midpoint, suppress the fade-IN (fade-out kept).
          * ``"end"`` → ``(max(start_sec, end_sec - 5.0), True)`` —
            5 s before the keeper end, clamped so a <5 s keeper never starts
            before its own start (T-sr8-01); suppress the fade-IN (the user
            request — "play 5 sec before the end should not have fade in"),
            fade-out kept.
    """
    if mode == "middle":
        return (start_sec + (end_sec - start_sec) / 2.0, True)
    if mode == "end":
        return (max(start_sec, end_sec - 5.0), True)
    # "start" and any unrecognized mode (T-sr8-02 safe default).
    return (start_sec, False)


# quick-260626-kw — Keepers dock width: default a bit wider than the 340 px
# minimum, persisted across restarts (saved on close, restored on launch).
_KEEPERS_DOCK_WIDTH_KEY = "keepers/dock_width"
_KEEPERS_DOCK_DEFAULT_WIDTH = 420
# quick-260626-kw2 — full window geometry (size + position + maximized state)
# persisted across restarts; saveGeometry()/restoreGeometry() round-trip the
# maximized flag for free.
_WINDOW_GEOMETRY_KEY = "window/geometry"


class MainWindow(QMainWindow):
    """Marmelade main window — UI-SPEC chrome + Plan 03 open-to-render flow.

    Signals:
        file_open_requested(str): Plan 01 contract — emitted when the user
            picks a file via the QFileDialog. Plan 03 connects this to
            :meth:`_open_file`.
        render_complete(): Plan 03 test seam — emitted EXACTLY ONCE per
            successful open, on BOTH the cache-HIT and cache-MISS success
            paths. Failure paths (error / cancelled) never emit it. Tests
            ``qtbot.waitSignal`` on this to know the render finished.
    """

    # Signal carrying the absolute path of the user-chosen audio file.
    file_open_requested = Signal(str)
    # Test seam (W-6: no `_for_test` suffix; production code may subscribe).
    render_complete = Signal()
    # quick-260621-dt4 — the ``heatmap_complete`` and ``lane_rebound`` test
    # seams were removed alongside the retired DSP/AI/Math heatmap panel +
    # pipeline. The kept heatmap BACKEND (heatmap_cache, heatmaps/*,
    # heatmap_lane, naming_resolver) is driven by the export pipeline, not
    # by these UI signals.
    # Plan 02.1-04 test seam — success-only emission discipline. Fires from
    # ``_on_audio_proxy_finished`` AFTER
    # ``prime(proxy_path)`` + ``_update_cache_size_footer()`` succeeded.
    # NOT emitted on error/cancel paths. Payload is the proxy WAV path
    # as a str so test assertions can ``Path(payload)`` it directly.
    audio_proxy_complete = Signal(str)
    # Plan 03-01 test seam — emitted after sidecar_cache.save_sidecar
    # completes for ANY region mutation (create / edge-drag finish /
    # delete). Payload is the sidecar JSON path as a str. Tests use
    # ``qtbot.waitSignal(window.regions_changed, ...)`` as the
    # synchronisation point for "the sidecar is on disk, you can now
    # assert on its contents".
    regions_changed = Signal(str)
    # Plan 03-04b test seam — emitted EXACTLY ONCE per successful export
    # with the exported file path as payload. Mirrors the success-only
    # discipline of ``audio_proxy_complete``: error / cancel paths do NOT
    # emit it. Tests ``qtbot.waitSignal(window.export_complete, ...)`` as
    # the synchronisation point for "the file is on disk".
    export_complete = Signal(str)
    # Phase 7 Plan 07-02 Task 4 test seam — emitted EXACTLY ONCE per
    # successful MasteringRunnable.finished. Payload is the keeper_id
    # so tests can ``qtbot.waitSignal(window.mastering_complete, ...)``
    # and filter by keeper. Mirrors the audio_proxy_complete success-only
    # discipline (error/cancel do NOT emit it).
    mastering_complete = Signal(str)
    # Phase 11 Plan 11-07 test seam (R-3) — emitted EXACTLY ONCE per
    # successful SpectralProxyRunnable.finished (the spectral arrays are
    # loaded + handed to the WaveformView). Mirrors the audio_proxy_complete
    # success-only discipline (error/cancel do NOT emit it). Tests
    # ``qtbot.waitSignal(window.spectral_build_complete, ...)`` as the
    # synchronisation point for "the spectral render is ready".
    spectral_build_complete = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Marmelade")
        # quick-260626-kw2 — restore the saved window geometry (size + position
        # + maximized state) if present; else the default 1280x800.
        _geo = QSettings().value(_WINDOW_GEOMETRY_KEY)
        if _geo is None or not self.restoreGeometry(_geo):
            self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        # Phase 7 Plan 07-05 — auto-create the Matchering reference library
        # directory on first launch (D-12). Silent — no status-bar message,
        # no QMessageBox. Wrapped in try/except OSError so a read-only $HOME
        # cannot crash app startup; we log a warning and continue (the user
        # can still operate the app — only the Matchering reference picker
        # will show empty until they manually create the dir).
        try:
            matchering_reference_dir().mkdir(parents=True, exist_ok=True)
        except OSError:
            # Read-only $HOME or permission error — non-fatal.
            # No logger import in this module by convention; the dock UI
            # will simply show the empty-state guidance label.
            pass

        # Central widget — the WaveformView fills the central area. It owns
        # the waveform_plot that is the X-link anchor for every heatmap lane
        # (Phase 2 X-link discipline).
        self._waveform_view = WaveformView(self)
        self.setCentralWidget(self._waveform_view)

        # Plan 02-05 — PlaybackEngine constructed BEFORE the toolbar build
        # so the toolbar's Follow-Playhead action can consult
        # ``engine.is_available`` to decide whether to enable itself. On
        # systems without libportaudio2 the engine constructs OK (the dlopen
        # failure was caught in playback.py) but every play() raises
        # PlaybackError; the toolbar simply disables itself.
        self._playback_engine = PlaybackEngine()

        # Plan 03-01 — RegionsOverlay attached to the waveform's PlotItem.
        # Duration provider reads lazily from the WaveformView (which is
        # populated by render_proxy on every file open) so this overlay
        # can be constructed before any file has been loaded.
        self._regions_overlay = RegionsOverlay(
            plot_item=self._waveform_view.waveform_plot,
            duration_s_provider=lambda: self._waveform_view._duration_s,
            parent=self,
        )
        self._waveform_view.set_regions_overlay(self._regions_overlay)
        self._regions_overlay.regions_changed.connect(self._on_regions_changed)
        # Plan 03-02 — Edit-menu enable/disable wiring. The overlay's
        # ``hover_changed(object)`` signal carries the new hovered region
        # id (or ``None`` on leave); the slot below flips the four Edit
        # actions to enabled/disabled accordingly.
        self._regions_overlay.hover_changed.connect(
            self._on_overlay_hover_changed
        )
        # Plan 03-04b — wire Export request from regions overlay. Slot
        # consumes ``(region_id, fmt)`` — fmt in ``{"mp3", "wav"}`` from
        # the two context-menu Export actions per CONTEXT D-A4-4 LOCKED.
        self._regions_overlay.export_requested.connect(
            self._on_export_region_requested
        )
        # quick-260701-jc5 (MARK-03) — MarkersOverlay shares the SAME PlotItem
        # as the RegionsOverlay so marker lines pan/zoom in lockstep with the
        # waveform. ``_current_markers`` is the single in-memory source of
        # truth for the panel + overlay + sidecar (populated on load, mutated
        # on add/edit/delete). Constructed before any file is open.
        self._markers_overlay = MarkersOverlay(
            plot_item=self._waveform_view.waveform_plot,
            parent=self,
        )
        self._current_markers: list[Marker] = []
        # Path of the sidecar JSON for the currently-open source file
        # (captured in _open_file after cache_key is computed). None when
        # no file is open — the _on_regions_changed slot bails in that case.
        self._current_sidecar_path: Optional[Path] = None
        # Plan 03-04b — additional source bookkeeping for naming_resolver
        # and the export pipeline. ``_current_playback_path`` (Phase 2.1)
        # is the proxy path passed to ExportRunnable; ``_current_source_path``
        # is the original source for ``resolve_filename``'s mtime →
        # recorded_date lookup; ``_current_cache_key`` is the 16-hex key
        # for ``dominant_trait_for_region``.
        self._current_source_path: Optional[Path] = None
        self._current_cache_key: Optional[str] = None
        # Single in-flight export worker handle — same shape as the audio
        # proxy and peak-builder runnable slots. Spawn-new cancels in-flight.
        self._current_export_runnable: Optional[ExportRunnable] = None

        # Build the four UI-SPEC chrome regions.
        # Plan 03-02 — Plan 02 also adds the right Keepers dock and the
        # Edit menu (created inside ``_build_menus``). Order matters:
        # ``_build_toolbar`` must run AFTER the WaveformView is constructed
        # (the Region Select QAction.toggled handler reaches into it).
        self._build_menus()
        self._build_toolbar()
        self._build_left_dock()
        self._build_right_dock()
        # quick-260701-jc5 (MARK-02) — build the Markers dock AFTER the Keepers
        # dock so ``self._dock_keepers`` exists for the splitDockWidget call.
        self._build_markers_dock()
        self._build_status_bar()

        # Plan 03-07 / W-6 ordering: this .connect() MUST land AFTER
        # self._build_menus() runs because _build_menus creates
        # _action_export_hovered_mp3 / _action_export_hovered_wav, which
        # the refresh helper accesses. Subscribing to regions_changed
        # here covers the case where a region's state mutates mid-hover
        # (e.g. user presses T while hovering a Keeper, demoting it to
        # Trash) — hover_changed does not re-fire in that case but the
        # Export submenu must still flip to disabled.
        self._regions_overlay.regions_changed.connect(
            self._refresh_export_hovered_actions_enabled
        )

        # Wire the empty-state Open button to the shared slot.
        self._waveform_view.open_button.clicked.connect(self._action_open_file)

        # Connect file_open_requested to the Plan 03 open handler.
        self.file_open_requested.connect(self._open_file)

        # ProgressOverlay lives over the WaveformView; hidden by default.
        self._overlay = ProgressOverlay(self._waveform_view)
        self._overlay.hide()
        # Phase 2.1 HUMAN-UAT #3 (final-final) — compact inline banner
        # used for the audio-proxy build. CRUCIAL: parent is MainWindow,
        # NOT WaveformView. QWidget children of WaveformView (which
        # contains a pg.GraphicsLayoutWidget doing GPU painting) can
        # fail to composite on some Linux+Qt+PyQtGraph stacks — the
        # exact symptom that killed the full-screen ProgressOverlay
        # approach. Status-bar widgets (children of MainWindow) render
        # fine on the same systems, so we use the same parent here.
        # `position_over_widget(self._waveform_view)` re-anchors the
        # banner using mapped coordinates whenever WaveformView moves
        # or resizes.
        from marmelade.ui.audio_proxy_banner import AudioProxyProgressBanner

        self._audio_proxy_banner = AudioProxyProgressBanner(self)
        self._audio_proxy_banner.hide()
        # When the user clicks "Stop building proxy", cancel the runnable.
        # Connection is established per-build in _open_file.

        # Track the in-flight runnable (semi-private — tests read this
        # deliberately for the N-5 distinct-instance assertion).
        self._current_runnable: Optional[PeakBuilderRunnable] = None
        # Plan 02.1-04 — single in-flight audio-proxy worker handle. NOT a
        # dict (Phase 2.1 RESEARCH §"_heatmap_runnables dict registry"): an
        # audio file can have at most one canonical proxy build in flight,
        # so a per-name registry is the wrong shape. The cancel preamble in
        # _open_file targets ONLY this attribute via _mw_proxy_conn_* tokens.
        self._current_proxy_runnable: Optional[AudioProxyRunnable] = None
        # Phase 11 Plan 11-07 (R-3) — the in-flight lazy spectral build.
        # Mirrors _current_proxy_runnable: stored so the cancel entry point
        # (_cancel_spectral_build) + the terminal handlers' runnable-identity
        # double-guard can target ONLY this worker via the SEPARATE
        # _mw_spectral_conn_* token namespace (so a targeted disconnect never
        # nukes an unrelated qtbot.waitSignal watcher). None when no spectral
        # build is running.
        self._current_spectral_runnable: Optional["SpectralProxyRunnable"] = None
        # True while an audio-proxy build owns the ProgressOverlay (UX
        # upgrade — Phase 2.1 HUMAN-UAT request #3). The waveform proxy's
        # completion paths consult this to decide whether to hide the
        # overlay (False = ok to hide; True = leave it up because the
        # audio proxy is still building and the overlay is its UI).
        self._audio_proxy_overlay_active: bool = False
        # Stashed (path, probe) tuple captured at spawn time so the
        # banner's "Build proxy" button (shown after a user cancel or a
        # build error) can re-spawn the worker without re-running the
        # cancel preamble. None when no audio-proxy build is pending or
        # available for retry.
        self._audio_proxy_retry_args: Optional[
            tuple[Path, "audio_file.AudioProbe"]
        ] = None
        # Path of the currently-open file (for status bar updates).
        self._current_path: Optional[Path] = None
        # Path to the canonical proxy WAV (Phase 2.1) used by the
        # mastering chain and by playback as the seek-friendly source.
        # None until a primed path is available; set by the proxy
        # finished path + the WAV-skip / cache-HIT branches.
        self._current_proxy_p: Optional[Path] = None
        # Path the playback engine should open() — proxy WAV for non-WAV
        # sources, source path for native WAV. Decoupled from _current_path
        # because the source path drives waveform + heatmap reads, while
        # playback must go through the canonical proxy to get O(1) seek
        # (AUD-04 / Phase 2.1 SC-4). None until a primed path is available
        # (set by WAV-skip / cache-HIT / _on_audio_proxy_finished).
        self._current_playback_path: Optional[Path] = None
        # CR-04 generation token: incremented on every _open_file call; worker
        # signal closures capture the value at submit time and slots compare it
        # against self._open_generation to drop stale terminal signals.
        self._open_generation: int = 0
        # CR-05 re-entrancy guard: True while _action_open_file is mid-flight
        # (including the cancel-and-drain spinloop). A second click is silently
        # dropped while this is set.
        self._open_in_progress: bool = False

        # Phase 7 Plan 07-02 Task 4 — per-keeper mastering worker registry.
        # Keyed by keeper_id (region UUID). Mirrors the 3-layer cancel-restart
        # discipline (D-18): generation token + runnable identity + targeted
        # disconnect (the runnable's setAutoDelete(False) keeps the C++
        # object alive for late slot calls).
        self._mastering_runnables: dict[str, MasteringRunnable] = {}
        self._mastering_generation: int = 0
        # Phase 8 Plan 08-04 — per-keeper YouTube upload state.
        # Mirrors the 3-layer cancel-restart discipline (D-28):
        #   * ``_upload_generation`` — monotonically-increasing token.
        #     Each new spawn bumps the counter and stores the token on
        #     the runnable signal closures so stale signals from a
        #     prior runnable are dropped by the slot's generation check.
        #   * ``_upload_runnables`` — region_id → YouTubeUploadRunnable.
        #     setAutoDelete(False) on the runnable keeps the C++ object
        #     alive for late slot calls (mirrors the mastering pattern).
        #   * ``_upload_state`` — region_id → dict with at minimum:
        #       ``audio_source_path``: Path piped into ffmpeg (mastered
        #           cache when fresh, tmp WAV from export_region otherwise).
        #       ``tmp_audio_to_cleanup``: Path | None — the tmp WAV the
        #           finished/error/cancelled slots must unlink. None when
        #           the audio_source_path is a mastered cache (R-05 bypass).
        #       ``dialog``: UploadDialog instance.
        #       ``thumbnail_bytes``: bytes — current Picsum/Pillow JPEG.
        #       ``nonce``: int — per-dialog Picsum cache-bust counter.
        self._upload_generation: int = 0
        self._upload_runnables: dict[str, YouTubeUploadRunnable] = {}
        self._upload_state: dict[str, dict] = {}
        # Phase 8 Plan 08-05 — bundle Share state. There is at most
        # ONE bundle in flight at a time (the bundle button is a global
        # multi-keeper action, not per-row), so a single Optional dict
        # suffices instead of the per-region dict the per-keeper Share
        # path uses. None when no bundle dialog is open.
        self._bundle_state: Optional[dict] = None
        # Phase 8 Plan 08-02 — SettingsDialog (Preferences) reference.
        # Held so the YouTube connect/disconnect slots can push state back
        # into the dialog after first_time_connect / disconnect completes.
        # Re-created on every View → Preferences… click.
        self._settings_dialog: SettingsDialog | None = None
        # Phase 7 Plan 07-06 Task 2 — Master & Export All batch state.
        # Phase A walks ``_master_all_queue`` sequentially, spawning one
        # MasteringRunnable at a time. Cancel bumps ``_master_all_generation``
        # so any in-flight signal becomes stale and is dropped by the
        # generation-guarded slots. ``_master_all_failed_ids`` collects
        # per-keeper failures for the Phase A → Phase B transition toast.
        # Phase C reuses Phase 3's _current_export_runnable; this batch
        # holds the per-keeper export queue + the run-time bookkeeping.
        self._master_all_generation: int = 0
        self._master_all_queue: list[Region] = []
        self._master_all_total: int = 0
        self._master_all_completed_count: int = 0
        self._master_all_failed_ids: set[str] = set()
        self._master_all_failure_msgs: dict[str, str] = {}
        self._master_all_target_dir: Optional[Path] = None
        self._master_all_format: Optional[str] = None
        # Phase C bookkeeping — separate queue + completion counter so
        # cancel in Phase A does not corrupt Phase C state.
        self._export_all_queue: list[Region] = []
        self._export_all_finished_count: int = 0
        self._export_all_failed_ids: set[str] = set()
        # Quick-260615-l4y — batch-export in-flight sentinel. Replaces the
        # removed sidebar ``phase_c_running`` state as the loop-control
        # guard for the three export-all slots (_on_export_all_next /
        # _on_export_all_failure / _on_export_cancelled). Set True in
        # _on_export_all_requested, cleared in _on_export_all_complete.
        self._export_all_in_flight: bool = False
        # WR-02 (Phase 7 review) — count of keepers that were excluded
        # from Phase C because their Phase A mastering failed; surfaced
        # in the _on_export_all_complete toast so the user knows why
        # the final export count is less than their keeper count.
        self._export_all_skipped_count: int = 0

        # Plan 02-05 — per-lane playheads + 30 Hz QTimer + spacebar shortcut.
        #
        # ``_lane_playheads`` is a dict[name -> InfiniteLine] keyed by the
        # PlotItem the playhead lives in. The waveform's playhead is keyed
        # ``"waveform"``; each active heatmap lane's playhead is keyed by the
        # heatmap name (e.g. ``"energy"``). On every 30 Hz tick we update
        # ALL entries in lockstep because a PyQtGraph QGraphicsItem can only
        # belong to one QGraphicsScene at a time (W1) — sharing a single
        # InfiniteLine across PlotItems silently fails.
        self._lane_playheads: dict[str, pg.InfiniteLine] = {
            "waveform": self._waveform_view.playhead,
        }
        # 30 Hz QTimer — drives the playhead update. 33 ms matches the
        # paint-budget the perf gates were sized for; faster polling adds
        # CPU spend without visual improvement (the playhead won't move
        # more than ~one viewport pixel per tick at typical zoom levels).
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(33)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        # Spacebar play/pause QShortcut. ApplicationShortcut context per
        # locked D-14b — Phase 2 has no text-input widgets so this and
        # WindowShortcut behave identically; the literal is locked for the
        # Phase 3+ override hook. Disabled if the audio backend is
        # unavailable (libportaudio2 missing) — same gating as the toolbar
        # action; a status-bar message communicates the degraded state.
        self._shortcut_play_pause = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._shortcut_play_pause.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_play_pause.activated.connect(self._action_toggle_playback)
        if not self._playback_engine.is_available:
            self._shortcut_play_pause.setEnabled(False)
            self.statusBar().showMessage(
                "Audio playback unavailable (libportaudio2 missing)", 0
            )

        # Phase 7 Plan 07-04 — A/B preview state + shortcuts.
        # ``_selected_keeper_id`` tracks the most-recently-clicked
        # KeeperRow (UI-SPEC §"A/B Preview Toolbar Toggle" line 550).
        # Two ApplicationShortcuts drive the same set_state transitions
        # as the toolbar widget's sub-button clicks — A → source proxy,
        # B → mastered cache. Same scope as the spacebar (D-14b
        # precedent — works regardless of which child widget has focus,
        # but modal dialogs naturally suppress it per Qt semantics).
        self._selected_keeper_id: Optional[str] = None
        # Plan 07-10e — track which keeper-row's Play button should show
        # the active highlight (the row whose audio the engine is
        # currently playing, if any). Cleared on engine stop / EOF
        # transitions and on a waveform click; set by _on_keeper_play
        # when starting playback for a specific keeper.
        # quick-260622-tit — _currently_playing_mode records WHICH of the
        # three buttons (start / middle / end) was last clicked, so the
        # highlight lands on the right button. There is no pause behavior.
        self._currently_playing_keeper_id: Optional[str] = None
        self._currently_playing_mode: Optional[str] = None
        # quick-260625 — playhead visual sync trim (seconds). Editable in
        # Preferences, persisted in QSettings. Defaults to the module constant.
        try:
            self._playhead_visual_offset_sec = float(
                QSettings().value(
                    _PLAYHEAD_OFFSET_SETTINGS_KEY, _PLAYHEAD_VISUAL_OFFSET_SEC
                )
            )
        except (TypeError, ValueError):
            self._playhead_visual_offset_sec = _PLAYHEAD_VISUAL_OFFSET_SEC
        self._shortcut_ab_a = QShortcut(QKeySequence("A"), self)
        self._shortcut_ab_a.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_ab_a.activated.connect(
            lambda: self._on_ab_shortcut_pressed("A")
        )
        self._shortcut_ab_b = QShortcut(QKeySequence("B"), self)
        self._shortcut_ab_b.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_ab_b.activated.connect(
            lambda: self._on_ab_shortcut_pressed("B")
        )
        # quick-260701-jc5 (MARK-01) — "M" drops a marker at the current
        # playhead position. ApplicationShortcut scope matches the A/B/spacebar
        # convention. ``_action_add_marker`` guards against a focused QLineEdit
        # (so the marker-label / keeper-note fields receive the keystroke) and
        # against no file being open.
        self._shortcut_add_marker = QShortcut(QKeySequence("M"), self)
        self._shortcut_add_marker.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self._shortcut_add_marker.activated.connect(self._action_add_marker)

        # quick-260630-dqd — number-key shortcuts ("1".."N") that switch the
        # waveform/spectral render mode. The binding count is REGISTRY-DRIVEN:
        # one QShortcut per ``RenderMode`` member (DQD-1), so adding a 7th mode
        # auto-binds key "7" with no code change here — there is no hardcoded
        # mode/key list. Each shortcut routes through
        # ``render_mode_combo.setCurrentIndex`` (the combo is the single source
        # of truth — DQD-2), so the "View:" combo stays visually synced and the
        # existing ``currentIndexChanged`` -> ``_on_render_mode_changed`` path
        # (in-place re-render + spectral lazy-build seam) runs unchanged. Same
        # ApplicationShortcut scope as the spacebar/A/B convention above. The
        # shortcuts are stored on a list attribute so they are not GC'd.
        self._render_mode_shortcuts: list[QShortcut] = []
        for i in range(len(list(RenderMode))):
            sc = QShortcut(QKeySequence(str(i + 1)), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(
                lambda checked=False, idx=i: self._on_render_mode_shortcut(idx)
            )
            self._render_mode_shortcuts.append(sc)

        # Wire the WaveformView's click-to-seek signal to our seek handler.
        self._waveform_view.seek_requested.connect(self._on_seek_requested)
        # Phase 11 Plan 11-07 (R-3) — the lazy spectral-build seam. The view
        # emits this exactly once when a spectral render mode is selected on a
        # cold cache (no stashed spectral arrays); we check the cache (HIT ->
        # load + render, no worker) else spawn a background build. Opening a
        # file does NOT fire this (lazy — REQ-3 a); only a user mode-select does.
        self._waveform_view.spectral_build_requested.connect(
            self._spawn_spectral_worker
        )

        # Phase 7 Plan 07-04 — wire the Keepers sidebar selection signal
        # to the A/B toggle's enable-state refresh. The sidebar emits
        # ``selection_changed(region_id)`` on row left-click; the slot
        # updates ``_selected_keeper_id`` and re-evaluates the toggle.
        self._keepers_sidebar.selection_changed.connect(
            self._on_keeper_selection_changed
        )
        # Phase 7 Plan 07-04 — re-evaluate the toggle when mastering
        # finishes for ANY keeper (the cache may have just landed for
        # the currently-selected keeper).
        self.mastering_complete.connect(
            lambda _kid: self._refresh_ab_toggle_enabled_state()
        )
        # Phase 8 Plan 08-05 — the "Share All as Bundle…" button
        # depends on EVERY keeper having a fresh mastered cache (D-02).
        # When mastering finishes for any keeper the cache state may now
        # satisfy the bundle gate, so re-evaluate it here. Without this
        # connection the button only refreshes on add_row / remove_row /
        # clear / probe-install, which means a "Master All Keepers" run
        # leaves the bundle button stale until the next app launch.
        self.mastering_complete.connect(
            lambda _kid: self._keepers_sidebar.refresh_bundle_button()
        )

        # Persist the active file's full path on every successful render
        # so we can auto-reopen it on the next launch. ``render_complete``
        # fires on BOTH cache-HIT and cache-MISS success branches and
        # NEVER on error / cancel, so it's the right success-only hook.
        self.render_complete.connect(self._persist_last_file)

        # Auto-reopen the last successfully-opened file on app start.
        # ``QTimer.singleShot(0, ...)`` defers until the event loop
        # spins so the WaveformView's empty-state is already painted
        # (giving the user a brief "No audio loaded" → file-loading
        # transition rather than a frozen window during proxy build).
        QTimer.singleShot(0, self._try_reopen_last_file)

    def _persist_last_file(self) -> None:
        """Save ``self._current_path`` to QSettings ``last_file``.

        Wired to :attr:`render_complete` so we only persist paths that
        successfully rendered — a corrupt file that throws during open
        will NOT poison the auto-reopen slot.
        """
        path = self._current_path
        if path is None:
            return
        try:
            QSettings("Marmelade", "Marmelade").setValue(
                "last_file", str(path)
            )
        except Exception:
            # Never fail a render because we couldn't persist; the
            # user can always re-open manually.
            pass

    def _try_reopen_last_file(self) -> None:
        """Auto-reopen the last successfully-opened file on app start.

        Reads ``last_file`` from QSettings. If the path is set AND the
        file still exists on disk, emits ``file_open_requested`` so the
        normal open pipeline kicks in. If the path is missing or the
        file is gone, leaves the empty-state ("No audio loaded") in
        place — the user can pick a new file via File → Open / toolbar.
        """
        try:
            settings = QSettings("Marmelade", "Marmelade")
            raw = settings.value("last_file", "")
            if not isinstance(raw, str) or not raw:
                return
            p = Path(raw)
            if not p.is_file():
                return
            self.file_open_requested.emit(str(p))
        except Exception:
            # Any QSettings / filesystem hiccup just falls back to the
            # empty state — never crash the app on startup.
            pass

    # ------------------------------------------------------------------ chrome
    def _build_menus(self) -> None:
        """Menu bar: File / View / Help with UI-SPEC items + cross-platform shortcuts."""
        menu_bar = self.menuBar()

        # --- File menu ---
        file_menu = menu_bar.addMenu("File")

        self._action_open = QAction("Open audio file…", self)
        self._action_open.setShortcut(QKeySequence.StandardKey.Open)
        self._action_open.triggered.connect(self._action_open_file)
        file_menu.addAction(self._action_open)

        recent_menu = file_menu.addMenu("Open recent")
        recent_menu.setEnabled(False)  # Phase 1: empty placeholder.
        self._menu_recent = recent_menu

        self._action_close = QAction("Close file", self)
        self._action_close.setShortcut(QKeySequence.StandardKey.Close)
        self._action_close.setEnabled(False)  # No file loaded yet.
        self._action_close.triggered.connect(self._close_file)
        file_menu.addAction(self._action_close)

        # Plan 02.1-04 D-08 — manual "Clear audio proxy cache" menu action.
        # Deletes <cache_root>/audio/ subtree and refreshes the status-bar
        # footer. No automatic eviction in this phase (LRU + size cap are
        # deferred to a future polish phase per CONTEXT § Deferred Ideas).
        self._action_clear_audio_cache = QAction(
            "Clear audio proxy cache", self
        )
        self._action_clear_audio_cache.triggered.connect(
            self._action_clear_audio_cache_slot
        )
        file_menu.addAction(self._action_clear_audio_cache)

        # Plan 03-04b D-A4-3 — "Change default export folder" menu action.
        # The first export triggers a QFileDialog automatically; this menu
        # lets the user re-choose later without doing an export.
        self._action_change_export_dir = QAction(
            "Change default export folder…", self
        )
        self._action_change_export_dir.triggered.connect(
            self._action_change_export_dir_slot
        )
        file_menu.addAction(self._action_change_export_dir)

        file_menu.addSeparator()

        self._action_quit = QAction("Exit", self)
        self._action_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self._action_quit.triggered.connect(self.close)
        file_menu.addAction(self._action_quit)

        # --- Edit menu (Plan 03-02 — region state mutation shortcuts) ---
        # Inserted between File and View per Qt convention. Each QAction
        # carries a single-letter QKeySequence so the standard
        # Qt.ShortcutContext.WindowShortcut behavior fires the shortcut
        # only when the focus is NOT on a child widget that consumes the
        # key (RESEARCH §Pitfall #5 — QLineEdit in the Keepers row note
        # input naturally suppresses K/T/U so the user can type those
        # letters into the note). The slots add a defense-in-depth
        # ``isinstance(QApplication.focusWidget(), QLineEdit)`` bail-out
        # for the explicit-trigger path (menu click via mouse).
        edit_menu = menu_bar.addMenu("Edit")

        self._action_mark_keeper = QAction("Mark as Keeper", self)
        self._action_mark_keeper.setShortcut(QKeySequence("K"))
        self._action_mark_keeper.setEnabled(False)
        self._action_mark_keeper.triggered.connect(
            lambda: self._mark_hovered_region("keeper")
        )
        edit_menu.addAction(self._action_mark_keeper)

        self._action_mark_trash = QAction("Mark as Trash", self)
        self._action_mark_trash.setShortcut(QKeySequence("T"))
        self._action_mark_trash.setEnabled(False)
        self._action_mark_trash.triggered.connect(
            lambda: self._mark_hovered_region("trash")
        )
        edit_menu.addAction(self._action_mark_trash)

        self._action_unmark = QAction("Unmark", self)
        self._action_unmark.setShortcut(QKeySequence("U"))
        self._action_unmark.setEnabled(False)
        self._action_unmark.triggered.connect(
            lambda: self._mark_hovered_region("untouched")
        )
        edit_menu.addAction(self._action_unmark)

        self._action_delete_region = QAction("Delete region", self)
        self._action_delete_region.setShortcut(QKeySequence.StandardKey.Delete)
        self._action_delete_region.setEnabled(False)
        self._action_delete_region.triggered.connect(
            self._delete_hovered_region
        )
        edit_menu.addAction(self._action_delete_region)

        # --- Edit menu: Export hovered region submenu (Plan 03-07 gap-closure) ---
        # Plan 03-07 / UAT Test 7 — redundant Export entry point.
        # The right-click context menu (Plan 03-05) is the primary path;
        # this Edit-menu submenu adds a keyboard-accessible fallback so
        # SC-4 (User can extract any region as MP3 or WAV) survives any
        # right-click regression. Mirrors the right-click context menu's
        # hover-targeting + keeper-only-export rules per CONTEXT D-A4-4.
        # Separator visually groups the export actions apart from the
        # state-mutation actions above.
        edit_menu.addSeparator()
        self._menu_export_hovered = edit_menu.addMenu("Export hovered region as")
        self._action_export_hovered_mp3 = QAction("MP3…", self)
        self._action_export_hovered_mp3.setEnabled(False)
        self._action_export_hovered_mp3.triggered.connect(
            lambda: self._on_export_hovered_region("mp3")
        )
        self._menu_export_hovered.addAction(self._action_export_hovered_mp3)
        self._action_export_hovered_wav = QAction("WAV…", self)
        self._action_export_hovered_wav.setEnabled(False)
        self._action_export_hovered_wav.triggered.connect(
            lambda: self._on_export_hovered_region("wav")
        )
        self._menu_export_hovered.addAction(self._action_export_hovered_wav)

        # --- View menu ---
        view_menu = menu_bar.addMenu("View")

        self._action_zoom_in = QAction("Zoom in", self)
        self._action_zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self._action_zoom_in.setEnabled(False)  # Enabled after first render.
        self._action_zoom_in.triggered.connect(
            lambda: self._waveform_view.zoom(1.25)
        )
        view_menu.addAction(self._action_zoom_in)

        self._action_zoom_out = QAction("Zoom out", self)
        self._action_zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self._action_zoom_out.setEnabled(False)
        self._action_zoom_out.triggered.connect(
            lambda: self._waveform_view.zoom(1.0 / 1.25)
        )
        view_menu.addAction(self._action_zoom_out)

        self._action_zoom_fit = QAction("Zoom to fit", self)
        self._action_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        self._action_zoom_fit.setEnabled(False)
        self._action_zoom_fit.triggered.connect(self._waveform_view.fit_view)
        view_menu.addAction(self._action_zoom_fit)

        # quick-260629 — manual rebuild of the on-disk spectral cache for the
        # current file. The spectrogram is built lazily + cached under
        # <cache_root>/spectra/<key>/{mel,centroid,bands}.dat; this deletes that
        # entry and re-runs the background build (useful if a cache went stale
        # or corrupt). Enabled only while a file is open.
        view_menu.addSeparator()
        self._action_rebuild_spectral = QAction("Rebuild spectrogram", self)
        self._action_rebuild_spectral.setEnabled(False)
        self._action_rebuild_spectral.setStatusTip(
            "Delete and recompute the cached spectrogram for the current file"
        )
        self._action_rebuild_spectral.triggered.connect(
            self._rebuild_spectral_cache
        )
        view_menu.addAction(self._action_rebuild_spectral)

        # --- Phase 8 Plan 08-02 — Preferences (introduces D-10 Settings
        # panel surface). Opens SettingsDialog with the YouTube connection
        # section. RESEARCH §"Settings panel surface" Option C — minimal
        # Preferences pane behind View → Preferences….
        view_menu.addSeparator()
        self._action_preferences = QAction("Preferences…", self)
        self._action_preferences.triggered.connect(self._on_open_preferences)
        view_menu.addAction(self._action_preferences)

        # Phase 7 Plan 07-07 iter-4 — the "View → Mastering panel"
        # checkable QAction was removed. The Mastering dock is now the
        # always-visible LEFT dock (see _build_left_dock). Qt's main-window
        # dock-area context menu still allows the user to undock/re-dock
        # individual docks; removing this menu item just drops the
        # redundant top-level toggle that couldn't be made to work reliably
        # on Linux/Wayland (07-UAT test 1 iter-1..3 failed UAT re-run).

        # --- Help menu ---
        help_menu = menu_bar.addMenu("Help")
        self._action_about = QAction("About Marmelade", self)
        self._action_about.triggered.connect(self._on_about)  # quick-260626-pbl
        help_menu.addAction(self._action_about)

    def _build_toolbar(self) -> None:
        """Toolbar: Open + Zoom Fit/In/Out — non-movable, non-floatable."""
        toolbar = QToolBar("Main toolbar", self)
        toolbar.setObjectName("MainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        self._toolbar = toolbar

        style = self.style()

        # 1. Open
        open_icon = style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        self._tb_open = QAction(open_icon, "Open audio file", self)
        self._tb_open.setToolTip("Open audio file")
        self._tb_open.triggered.connect(self._action_open_file)
        toolbar.addAction(self._tb_open)

        # 2. Zoom Fit — custom corner-bracket icon (replaces the
        # SP_FileDialogContentsView placeholder which read as a list view).
        self._tb_zoom_fit = QAction(_fit_to_view_icon(), "Fit waveform to view", self)
        self._tb_zoom_fit.setToolTip("Fit waveform to view")
        self._tb_zoom_fit.setEnabled(False)
        self._tb_zoom_fit.triggered.connect(self._waveform_view.fit_view)
        toolbar.addAction(self._tb_zoom_fit)

        # 3. Zoom In — magnifying-glass + (replaces SP_ArrowUp).
        self._tb_zoom_in = QAction(_zoom_in_icon(), "Zoom in", self)
        self._tb_zoom_in.setToolTip("Zoom in")
        self._tb_zoom_in.setEnabled(False)
        self._tb_zoom_in.triggered.connect(
            lambda: self._waveform_view.zoom(1.25)
        )
        toolbar.addAction(self._tb_zoom_in)

        # 4. Zoom Out — magnifying-glass − (replaces SP_ArrowDown).
        self._tb_zoom_out = QAction(_zoom_out_icon(), "Zoom out", self)
        self._tb_zoom_out.setToolTip("Zoom out")
        self._tb_zoom_out.setEnabled(False)
        self._tb_zoom_out.triggered.connect(
            lambda: self._waveform_view.zoom(1.0 / 1.25)
        )
        toolbar.addAction(self._tb_zoom_out)

        # 5. Follow Playhead (Plan 02-05) — checkable; default OFF (quick-260629:
        # the user opted out of auto-paging the view by default). Disabled when
        # the audio backend is unavailable (libportaudio2 missing on Linux).
        # Custom grey chevron icon so it sits in the same hue family as the
        # other toolbar icons (replaces the OS-themed SP_ArrowForward which read
        # as a colored / out-of-family arrow).
        self._tb_follow_playhead = QAction(_follow_playhead_icon(), "Follow playhead", self)
        self._tb_follow_playhead.setToolTip("Follow playhead")
        self._tb_follow_playhead.setCheckable(True)
        self._tb_follow_playhead.setChecked(False)  # default OFF
        self._tb_follow_playhead.setEnabled(self._playback_engine.is_available)
        toolbar.addAction(self._tb_follow_playhead)

        # 6. Region Select mode (Plan 03-02 — CONTEXT D-A1-1).
        # Checkable; default OFF on app start (the user must opt in,
        # because pan is the primary navigation gesture once a file is
        # open). Tooltip is the 3-line UI-SPEC §Copywriting block;
        # ``\n`` line breaks render as a multi-line Qt tooltip. The
        # custom dashed-marquee icon replaces SP_FileDialogDetailedView
        # which read as a generic list view.
        self._tb_region_select = QAction(_region_select_icon(), "Region select mode", self)
        self._tb_region_select.setToolTip(
            "Region select mode (toggle)\n"
            "Drag on the waveform to mark a region. Middle-drag pans.\n"
            "Shift+drag always marks a region — works with this off too."
        )
        self._tb_region_select.setCheckable(True)
        self._tb_region_select.setChecked(False)  # mode-OFF on start (D-A1-1)
        self._tb_region_select.toggled.connect(self._on_region_select_toggled)
        toolbar.addAction(self._tb_region_select)

        # 6b. "View:" render-mode selector — relocated from inside WaveformView
        # onto the toolbar, between Region-select and A/B preview (user
        # request). WaveformView still OWNS ``render_mode_combo`` and all its
        # wiring (currentIndexChanged -> _on_render_mode_changed re-render +
        # spectral lazy-build); here we only reparent the existing combo via
        # addWidget. The waveform view is constructed (line ~309) before this
        # toolbar is built, so the combo already exists. Number-key shortcuts
        # + tests are unaffected — they still reach it as
        # ``self._waveform_view.render_mode_combo``.
        self._tb_view_label = QLabel("View:", toolbar)
        self._tb_view_label.setContentsMargins(8, 0, 4, 0)
        toolbar.addWidget(self._tb_view_label)
        toolbar.addWidget(self._waveform_view.render_mode_combo)

        # 7. Phase 7 Plan 07-04 — A/B preview toolbar toggle (D-13).
        # Composite QWidget (48×24 px) inserted via QToolBar.addWidget,
        # which wraps it in a QWidgetAction. UI-SPEC §"Layout
        # Architecture" line 54 places it after Region Select.
        # Disabled by default — enabled state is driven by
        # ``_refresh_ab_toggle_enabled_state`` based on keeper selection
        # + mastering + cache freshness.
        self._ab_toggle = ABToggleWidget(self)
        self._ab_toggle.set_enabled(False)
        # Plan 07-09 — permanent discoverability tooltip. Survives re-enable
        # in _refresh_ab_toggle_enabled_state (which used to clear it to "").
        # The em-dash is U+2014; preserve verbatim — pinned by integration
        # test (test_permanent_tooltip_set_at_construct_time). Plan 07-04
        # shipped invisible A/B labels (clipped by inherited QPushButton
        # padding); the diagnosis at .planning/debug/ab-widget-broken-keys-
        # icon-tooltip.md proved the user has no visual cue without this
        # tooltip. Plan 07-09 makes it the canonical discoverability cue.
        self._ab_default_tooltip = (
            "A/B preview — A = source, B = mastered. "
            "Click a keeper row with a Ready mastered cache to enable."
        )
        self._ab_toggle.setToolTip(self._ab_default_tooltip)
        toolbar.addWidget(self._ab_toggle)
        self._ab_toggle.state_changed.connect(self._on_ab_state_changed)

        # 8. Playback timestamp — placed right after the A/B toggle
        # with a fixed 24-px gap so it visually leads the toolbar's
        # right-hand half. Updated on every 30 Hz playback tick via
        # ``_on_playback_tick`` so the user gets continuous feedback
        # for where the playhead is in the source.
        gap = QWidget(toolbar)
        gap.setFixedWidth(24)
        toolbar.addWidget(gap)

        self._tb_time_label = QLabel("0:00", toolbar)
        self._tb_time_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        # Big and monospaced so the numbers don't jitter and the timestamp
        # reads at a glance. The font-size MUST live in this widget's own
        # stylesheet: the app-wide QSS (app.qss) sets `QLabel { font-size:
        # 10pt }` / `QToolBar { font-size: 10pt }`, and a Qt style sheet
        # overrides QFont/setFont() — so setPointSize() here silently did
        # nothing. A widget-level rule beats the app-level one. 24pt ≈ 2x
        # the 10pt the toolbar was actually rendering (user request); the
        # toolbar grows to fit the taller label.
        self._tb_time_label.setStyleSheet(
            "color: #E6E6E6; padding: 0 12px; font-size: 18pt; "
            "font-weight: 600; "
            'font-family: "SF Mono", "Consolas", "JetBrains Mono", '
            '"DejaVu Sans Mono", monospace;'
        )
        self._tb_time_label.setToolTip("Current playback position")
        toolbar.addWidget(self._tb_time_label)

        # quick-260621-gfq — the toolbar Normalize button + dB spinbox were
        # REMOVED. Normalize is now the FINAL per-keeper mastering-chain stage,
        # surfaced in the Mastering dock + the keeper-row Normalize toggle.

    def _build_left_dock(self) -> None:
        """Left sidebar: 'Mastering' dock.

        quick-260621-dt4 retired the DSP/AI/Math "Heatmaps" panel that used
        to occupy the LEFT area. The Mastering panel (formerly a tab sibling
        of Keepers on the right) is now the primary left-side surface,
        constructed here in the LEFT dock area. The
        ``session_chain_changed`` → ``_on_session_chain_changed`` wiring is
        preserved verbatim so mastering edits still refresh every keeper's
        divergence badge (D-04 — does NOT mutate any keeper).
        """
        mastering_dock = QDockWidget("Mastering", self)
        mastering_dock.setObjectName("MasteringDock")
        # quick-260623-csc — suppress the native QDockWidget title-bar strip
        # (the gray "Mastering" header the user reads as a tab). An empty
        # QWidget replaces the rendered title bar; the dock is a permanent
        # left sidebar (its visibility toggle was removed in Phase 7), so the
        # drag/float/close affordance is intentionally dropped. windowTitle()
        # stays "Mastering" — setTitleBarWidget does not touch that property,
        # so test_main_window_skeleton's windowTitle assertion still passes.
        mastering_dock.setTitleBarWidget(QWidget(mastering_dock))
        mastering_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        # quick-260623-csc — narrowed further (160 -> 130) now that the title
        # bar is gone and content margins are trimmed to 8px; a stage row
        # ([checkbox] [stage label] [gear]) still fits unclipped.
        mastering_dock.setMinimumWidth(130)
        self._mastering_widget = MasteringDock(mastering_dock)
        mastering_dock.setWidget(self._mastering_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, mastering_dock)
        self._mastering_dock = mastering_dock
        # Wire session-chain edits → divergence-badge refresh for every
        # existing keeper. D-04 — does NOT mutate any keeper.
        self._mastering_widget.session_chain_changed.connect(
            self._on_session_chain_changed
        )

    def _build_right_dock(self) -> None:
        """Right sidebar: 'Keepers (N)' dock — Plan 03-02 UI-07.

        Mirrors :meth:`_build_left_dock` shape. The dock holds a
        :class:`KeepersSidebar` whose four aggregated row signals
        (jump / play / note / delete) are connected to MainWindow slots
        that drive the playback engine and the regions overlay. The
        dock title carries the live keeper count via
        ``KeepersSidebar.set_dock_title_callback`` — every add/remove on
        the sidebar fires the callback so the title stays in sync
        without each callsite remembering to update it.

        UI-SPEC §Layout Architecture pins min width 280 px and the
        Right/Left DockWidgetArea allowed set (the user may park the
        Keepers panel on the left if they like — Qt convention).
        """
        dock = QDockWidget("Keepers (0)", self)
        dock.setObjectName("KeepersDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        # Plan 03-07 / W-7 — raised from 280 to 340 px to fit the two new
        # per-row MP3 + WAV QToolButtons (~36 px each) without collapsing
        # the note QLineEdit below readable width. UI-SPEC §Layout
        # Architecture documents the new min-width.
        dock.setMinimumWidth(340)

        self._keepers_sidebar = KeepersSidebar(dock)
        self._keepers_sidebar.setMinimumWidth(340)
        self._keepers_sidebar.jump_requested.connect(self._on_keeper_jump)
        self._keepers_sidebar.play_requested.connect(self._on_keeper_play)
        # quick-260622-sr8 — the two new play-position buttons reuse the same
        # handler with the start_mode injected (no second playback code path).
        self._keepers_sidebar.play_middle_requested.connect(
            lambda rid: self._on_keeper_play(rid, start_mode="middle")
        )
        self._keepers_sidebar.play_end_requested.connect(
            lambda rid: self._on_keeper_play(rid, start_mode="end")
        )
        self._keepers_sidebar.delete_requested.connect(self._on_keeper_delete)
        # quick-260620-mgu NORM-04 — per-keeper Normalize toggle. Persists the
        # field to the sidecar via the overlay carrier.
        self._keepers_sidebar.normalize_changed.connect(
            self._on_keeper_normalize_changed
        )
        # Phase 7 Plan 07-02 Task 4 — Master button + right-click Cancel
        # mastering. The mastering_requested slot opens MasteringDialog;
        # cancel_mastering_requested aborts an in-flight runnable.
        self._keepers_sidebar.mastering_requested.connect(
            self._on_keeper_mastering_requested
        )
        self._keepers_sidebar.cancel_mastering_requested.connect(
            self._on_keeper_mastering_cancel_requested
        )
        # Quick-260615-l4y — two-button batch flow signals.
        # ``master_all_requested`` (Master idle click) starts a modal-free
        # mastering pass over every keeper; ``mastering_cancel_requested``
        # (Master running click) aborts the loop with the 3-layer pattern;
        # ``export_all_requested`` (Export click) opens the output-folder /
        # format modal and runs the sequential batch export.
        self._keepers_sidebar.master_all_requested.connect(
            self._on_master_all_requested
        )
        self._keepers_sidebar.mastering_cancel_requested.connect(
            self._on_master_all_cancel_requested
        )
        self._keepers_sidebar.export_all_requested.connect(
            self._on_export_all_requested
        )
        # Phase 8 Plan 08-04 Task 4 — per-keeper Share button. The slot
        # owns the audio-source routing decision (R-05 + D-02 fallback)
        # and opens a modal UploadDialog.
        self._keepers_sidebar.share_requested.connect(
            self._on_share_requested
        )
        # Phase 8 Plan 08-05 Task 3 — top-of-sidebar bundle Share button
        # (D-19). The slot opens a modal BundleDialog and orchestrates
        # the bundle build → ffmpeg → YouTubeUploadRunnable chain. Also
        # install the freshness-probe so the sidebar can disable the
        # button when any keeper lacks a fresh mastered cache (D-02).
        self._keepers_sidebar.bundle_share_requested.connect(
            self._on_bundle_share_requested
        )
        self._keepers_sidebar.set_mastered_cache_fresh_probe(
            self._is_keeper_mastered_cache_fresh
        )
        # Live ``Keepers (N)`` dock title — sidebar fires this on every
        # add_row / remove_row / clear mutation so MainWindow doesn't have
        # to remember to setWindowTitle at each callsite.
        self._keepers_sidebar.set_dock_title_callback(
            lambda n: dock.setWindowTitle(f"Keepers ({n})")
        )
        dock.setWidget(self._keepers_sidebar)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        # quick-260626-kw — restore the persisted Keepers dock width (default a
        # bit wider than the 340 px minimum). resizeDocks is deferred to the
        # next event-loop turn so it lands AFTER the initial layout pass.
        self._keepers_dock = dock
        try:
            _kw = int(
                QSettings().value(
                    _KEEPERS_DOCK_WIDTH_KEY, _KEEPERS_DOCK_DEFAULT_WIDTH
                )
            )
        except (TypeError, ValueError):
            _kw = _KEEPERS_DOCK_DEFAULT_WIDTH
        _kw = max(_kw, 340)
        QTimer.singleShot(
            0,
            lambda: self.resizeDocks(
                [dock], [_kw], Qt.Orientation.Horizontal
            ),
        )
        self._dock_keepers = dock

    def _build_markers_dock(self) -> None:
        """Markers dock (quick-260701-jc5 — MARK-02), placed BELOW Keepers.

        Mirrors :meth:`_build_right_dock`: a :class:`MarkersSidebar` inside a
        ``QDockWidget("Markers (0)")`` whose add/jump/delete/label signals are
        wired to MainWindow handlers, with a live "Markers (N)" title callback.
        Positioned below the Keepers dock via ``splitDockWidget`` so both share
        the right dock area vertically. Called AFTER ``_build_right_dock`` so
        ``self._dock_keepers`` already exists.
        """
        dock = QDockWidget("Markers (0)", self)
        dock.setObjectName("MarkersDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
        )

        self._markers_sidebar = MarkersSidebar(dock)
        # Signals-up: the [+] button routes to the SAME add action as the "m"
        # shortcut (locked decision #2 — one live-playhead position source).
        self._markers_sidebar.add_requested.connect(self._action_add_marker)
        self._markers_sidebar.jump_requested.connect(self._on_marker_jump)
        self._markers_sidebar.delete_requested.connect(self._on_marker_delete)
        self._markers_sidebar.label_edited.connect(self._on_marker_label_edited)
        # Live "Markers (N)" title.
        self._markers_sidebar.set_dock_title_callback(
            lambda n: dock.setWindowTitle(f"Markers ({n})")
        )
        dock.setWidget(self._markers_sidebar)

        # Place BELOW the Keepers dock (shared right area, split vertically).
        self.splitDockWidget(
            self._dock_keepers, dock, Qt.Orientation.Vertical
        )
        self._dock_markers = dock

    def _build_status_bar(self) -> None:
        """Status bar with left metadata + right zoom label.

        Plan 02.1-04 adds two new widgets:
        * ``_status_proxy_progress`` (transient ``addWidget``) — shows the
          "Preparing audio proxy: NN%" text only during an active proxy
          build. Hidden by default. (D-09 / SC-2)
        * ``_status_cache_size`` (permanent ``addPermanentWidget``) — shows
          the running on-disk audio-cache size as ``"Cache: N.NN GiB"``.
          Refreshed after app start, file open completes, proxy build
          completes, and the "Clear audio proxy cache" menu fires. (D-09)
        """
        status_bar = QStatusBar(self)
        status_bar.setSizeGripEnabled(True)
        self.setStatusBar(status_bar)

        self._status_left = QLabel("")
        self._status_left.setProperty("role", "caption")
        status_bar.addWidget(self._status_left, 1)

        # Plan 02.1-04 D-09 — transient progress text. Hidden by default;
        # shown only during an active proxy build. Sits between the
        # file-metadata left widget and the right-pinned zoom + cache-size
        # widgets so the user sees it alongside the file name.
        self._status_proxy_progress = QLabel("")
        self._status_proxy_progress.setProperty("role", "caption")
        self._status_proxy_progress.hide()
        status_bar.addWidget(self._status_proxy_progress)

        self._status_zoom = QLabel("")
        self._status_zoom.setProperty("role", "caption")
        status_bar.addPermanentWidget(self._status_zoom)

        # Plan 03-04b — export progress widget pair. UI-SPEC §Status-bar
        # export progress: a label that reads "Exporting clip: NN%" /
        # "Exported {filename}" / "Export cancelled" / "Export failed: …"
        # plus a tiny `×` cancel button visible only during an active
        # export. Both empty/hidden at construction.
        self._status_export = QLabel("")
        self._status_export.setObjectName("StatusExport")
        self._status_export.setStyleSheet(
            "color: #9CA3AF; font-family: monospace; font-size: 8pt;"
        )
        status_bar.addPermanentWidget(self._status_export)
        self._status_export_cancel = QToolButton()
        self._status_export_cancel.setText("×")
        self._status_export_cancel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._status_export_cancel.setToolTip("Cancel export")
        self._status_export_cancel.setFixedWidth(16)
        self._status_export_cancel.setVisible(False)
        self._status_export_cancel.clicked.connect(self._on_export_cancel_clicked)
        status_bar.addPermanentWidget(self._status_export_cancel)

        # Plan 02.1-04 D-09 — permanent right-pinned cache-size footer.
        # addPermanentWidget so a transient showMessage() cannot overwrite
        # it. Populated at construction time so the user sees the running
        # cache footprint from the moment the window appears.
        self._status_cache_size = QLabel("")
        self._status_cache_size.setProperty("role", "caption")
        status_bar.addPermanentWidget(self._status_cache_size)
        self._update_cache_size_footer()

        # Phase 2.1 HUMAN-UAT — pinned-right version+build SHA so the
        # user can verify which commit the running process is actually on.
        # Added in response to "are you sure i am seeing the last version
        # of the app, ... maybe adding Version: XYZ in the footer would
        # help". Read once at import time from `marmelade.__build__`.
        from marmelade import __build__, __version__

        self._status_version = QLabel(f"v{__version__} · {__build__}")
        self._status_version.setProperty("role", "caption")
        self._status_version.setToolTip(
            f"Marmelade v{__version__} build {__build__}"
        )
        status_bar.addPermanentWidget(self._status_version)

    # --------------------------------------------------------------- helpers
    def _basename(self, path: str | Path) -> str:
        """UI-SPEC §Window title + §Status bar: basename only, never absolute path."""
        return Path(path).name

    def _fmt_duration(self, seconds: float) -> str:
        """Format ``seconds`` per UI-SPEC: HH:MM:SS for >=1h, MM:SS.s otherwise."""
        if seconds >= 3600:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h}:{m:02d}:{s:02d}"
        m = int(seconds // 60)
        s = seconds - 60 * m
        return f"{m}:{s:04.1f}"

    def _update_status_for_loaded(self, probe: "audio_file.AudioProbe", path: Path) -> None:
        """Populate the status bar after a successful render."""
        basename = self._basename(path)
        text = (
            f"{basename} · {self._fmt_duration(probe.duration_s)} · "
            f"{probe.sample_rate} Hz · {probe.channels} ch"
        )
        self._status_left.setText(text)
        self.setWindowTitle(f"{basename} — Marmelade")
        # Enable the View > Zoom actions (Plan 01 left them disabled).
        for action in (
            self._action_close,
            self._action_zoom_in,
            self._action_zoom_out,
            self._action_zoom_fit,
            self._action_rebuild_spectral,
            self._tb_zoom_fit,
            self._tb_zoom_in,
            self._tb_zoom_out,
        ):
            action.setEnabled(True)
        # quick-260621-gfq — the toolbar Normalize button + dB spinbox were
        # removed; normalize is now the mastering chain's final stage.

    # --------------------------------------------------------------- file open
    def _action_open_file(self) -> None:
        """Shared Open slot — menu + toolbar + empty-state button all call this.

        CR-05: gated by self._open_in_progress so a second click during the
        cancel-and-drain spinloop is silently dropped (eliminates re-entrant
        QFileDialog hazard). Try/finally guarantees the flag clears even if
        the user cancels the QFileDialog mid-dialog or an exception is raised
        inside the body — subsequent clicks remain functional.
        """
        if self._open_in_progress:
            return
        self._open_in_progress = True
        try:
            # Phase 6 T-06-01 — explicit ("Marmelade", "Marmelade")
            # org/app pair (NEVER the bare no-arg form) so the last_dir
            # preference shares the same QSettings namespace as the
            # heatmap params + export folder + Phase 1 default_open_dir,
            # and inherits the conftest test-mode sandbox cleanly.
            settings = QSettings("Marmelade", "Marmelade")
            default_dir = settings.value("last_dir", default_open_dir())
            if not isinstance(default_dir, str):
                default_dir = str(default_dir) if default_dir is not None else ""

            path, _selected_filter = QFileDialog.getOpenFileName(
                self,
                "Open audio file",
                default_dir,
                "Audio files (*.wav *.flac *.mp3);;All files (*)",
            )

            if not path:
                # Cancel is silent — UI-SPEC §File open.
                return

            # Persist the parent directory for next time.
            settings.setValue("last_dir", str(Path(path).parent))

            # Emit so external subscribers (Plan 01 contract) see it AND drive
            # our own _open_file slot (connected in __init__).
            self.file_open_requested.emit(path)
        finally:
            self._open_in_progress = False

    def _open_file(self, path: str) -> None:
        """Plan 03 open flow: probe → duration check → extension → cache → worker.

        Runs entirely on the GUI thread up to the cache-lookup decision. On
        cache HIT, ``render_proxy`` runs synchronously and ``render_complete``
        fires immediately. On cache MISS, a :class:`PeakBuilderRunnable` is
        started on the global :class:`QThreadPool`; ``render_complete`` fires
        from :meth:`_on_proxy_ready`.

        If a build is already in flight, cancel it first and wait for the
        ``cancelled`` signal (with a 2 s safety timeout) before starting the
        new one.
        """
        p = Path(path)
        basename = self._basename(p)

        # CR-04: stamp this open with a fresh generation token. Worker signal
        # closures will capture this value; slots compare against the current
        # generation to drop stale signals from previously-cancelled workers.
        self._open_generation += 1
        gen = self._open_generation

        # quick-260629 — opening a different sound resets the waveform to CLASSIC
        # and drops any stashed spectral data, so the previous file's
        # spectrogram / centroid / band surfaces are never shown against the new
        # audio. The new file's render (and a re-selected spectral mode's lazy
        # build) repopulate from scratch.
        self._waveform_view.reset_render_mode_to_classic()

        # If a build is already in flight, cancel it and wait for cancellation
        # to drain before proceeding (idempotent _open_file per behavior).
        if self._current_runnable is not None:
            old = self._current_runnable
            # CR-02: disconnect ONLY our own slot wiring BEFORE cancelling so
            # late terminal signals from the cancelled worker can't fire into
            # the next file's GUI state. We pass the QMetaObject.Connection
            # tokens we stored at connect time so external listeners (e.g.
            # a test's qtbot.waitSignal slot on `cancelled`) remain attached
            # — a bare `signals.<x>.disconnect()` would nuke every connection
            # including the test watcher and break N-5. progress is left
            # wired because overlay updates are visually idempotent and the
            # overlay is about to be replaced anyway.
            _conn = getattr(old, "_mw_conn_finished", None)
            if _conn is not None:
                try:
                    old.signals.finished.disconnect(_conn)
                except (RuntimeError, TypeError):
                    pass
            _conn = getattr(old, "_mw_conn_error", None)
            if _conn is not None:
                try:
                    old.signals.error.disconnect(_conn)
                except (RuntimeError, TypeError):
                    pass
            _conn = getattr(old, "_mw_conn_cancelled", None)
            if _conn is not None:
                try:
                    old.signals.cancelled.disconnect(_conn)
                except (RuntimeError, TypeError):
                    pass
            old.cancel()
            # REVIEW-GAPS CR-01 + CR-02 + WR-01 unified fix (plan 01-10):
            # Reset state synchronously. The CR-02 disconnect above already
            # neutralised this worker's terminal signals; CR-04's generation
            # guard would also drop them. So the cancelled-slot side effects
            # (overlay hide, runnable clear) will never fire — we run them
            # here in the cancel preamble so BOTH the HIT and MISS branches
            # below inherit a clean state. CR-05 re-entrancy is preserved
            # by the `_open_in_progress` flag at the entry of
            # `_action_open_file` (lines 305-332).
            self._current_runnable = None
            self._overlay.hide()

        # Plan 02.1-04 — D-10 audio-proxy cancel preamble. Same shape as the
        # peak-builder block above (targeted disconnect via stored
        # ``_mw_proxy_conn_*`` tokens, then ``.cancel()``). The disconnect
        # MUST happen BEFORE ``cancel()`` so a late terminal signal from
        # the cancelled worker cannot fire into the next file's GUI state.
        # Separate token namespace (``_mw_proxy_conn_*``) from the peak
        # builder's ``_mw_conn_*`` so the two preambles cannot collide.
        if self._current_proxy_runnable is not None:
            old_proxy = self._current_proxy_runnable
            for token_name, signal_name in (
                ("_mw_proxy_conn_finished", "finished"),
                ("_mw_proxy_conn_error", "error"),
                ("_mw_proxy_conn_cancelled", "cancelled"),
            ):
                _conn = getattr(old_proxy, token_name, None)
                if _conn is not None:
                    try:
                        getattr(old_proxy.signals, signal_name).disconnect(_conn)
                    except (RuntimeError, TypeError):
                        pass
            old_proxy.cancel()
            # Synchronous state reset — mirror of the peak-builder preamble
            # above. The CR-02 disconnect already neutralised the cancelled
            # worker's terminal signals; generation-token guards in the
            # slots are belt-and-suspenders. Resetting here keeps the
            # WAV-skip / cache-HIT / MISS branches below inheriting a
            # clean state.
            self._current_proxy_runnable = None
            self._status_proxy_progress.hide()
            # If the audio-proxy banner was up, tear it down here too so
            # the next file's spawn starts from a clean state. Clear the
            # retry args so a stale Build click doesn't try to rebuild
            # the previous file.
            if self._audio_proxy_overlay_active:
                self._audio_proxy_banner.hide()
                self._audio_proxy_overlay_active = False
            self._audio_proxy_retry_args = None
            # Clear the playback path — the next branch below (WAV-skip /
            # cache-HIT / MISS) will re-set it. Leaving the old proxy path
            # in place would let a play() between cancel and finish open
            # the previous file's proxy.
            self._current_playback_path = None
            self._shortcut_play_pause.setEnabled(
                self._playback_engine.is_available
            )

        # Plan 02-05 — stop any active playback for the previous file.
        # ``engine.stop()`` is idempotent so calling on a fresh engine
        # (no prior file) is a no-op. Reset the playhead to 0 on every
        # new file (UX choice — natural starting position).
        try:
            self._playback_engine.stop()
        except Exception:
            pass
        self._playback_timer.stop()
        self._waveform_view.playhead.setValue(0.0)

        # Plan 03-01 — clear any in-flight regions from the previous file.
        # The next file's sidecar load (after cache_key is computed below)
        # repopulates the overlay. Clearing the sidecar path bookkeeping
        # ensures _on_regions_changed bails until set_regions has run.
        self._regions_overlay.clear()
        self._current_sidecar_path = None
        # Plan 03-02 — empty the Keepers dock alongside the overlay
        # (RegionsOverlay.clear does NOT emit regions_changed, so the
        # sidebar would otherwise still show the previous file's rows
        # until the next sidecar load triggers a refresh).
        if hasattr(self, "_keepers_sidebar"):
            self._keepers_sidebar.clear()
        # quick-260701-jc5 — clear the previous file's markers (panel + overlay
        # + in-memory list). The next file's sidecar load repopulates them.
        if hasattr(self, "_markers_overlay"):
            self._markers_overlay.clear()
        if hasattr(self, "_markers_sidebar"):
            self._markers_sidebar.clear()
            # A file is being opened — enable the [+] Add-marker button.
            self._markers_sidebar.set_add_enabled(True)
        self._current_markers = []

        # Bug #1 fix — prime the playback engine with sample_rate +
        # duration BEFORE any gate that might return early. Without this,
        # a user who clicks to seek BEFORE pressing spacebar loses the
        # seek target (seek() falls back to _start_frame=0 when
        # _sample_rate==0; D-15 lazy compute means play() never runs at
        # open time to populate it). Single call site here covers BOTH
        # the cache-HIT path (which returns after render_complete.emit())
        # AND the cache-MISS path (which continues via _on_proxy_ready)
        # — both inherit a primed engine. Priming a file that the gates
        # below reject is harmless (engine state for an unopened file is
        # never consumed). Wrapped in try/except per the bug spec: a
        # priming failure NEVER breaks the file-open flow.
        try:
            self._playback_engine.prime(str(path))
        except Exception:
            pass

        # (1) Extension check FIRST — cheapest gate; a file with an
        # unsupported extension never gets a chance to crash pedalboard
        # (Rule 1 fix: the plan's numbered ordering reads as
        # probe→duration→ext, but its own test (.txt fixture) requires
        # the unsupported-format dialog rather than the corrupt-file
        # dialog. Reordering the check to extension-first preserves the
        # intended user-visible behavior for non-audio files.).
        if p.suffix.lower() not in _SUPPORTED_EXTS:
            show_unsupported_format(self, basename)
            return

        # (2) Probe — pedalboard rejection or missing file → corrupt-file dialog.
        try:
            probe = audio_file.probe(p)
        except (FileNotFoundError, ValueError) as e:
            result = show_corrupt_file(self, basename, str(e))
            if result == QMessageBox.StandardButton.Retry:
                self._action_open_file()
            return

        # (3) Duration check — > 8h → too-long dialog. NO worker spawned.
        if probe.duration_s > audio_file.MAX_DURATION_S:
            show_too_long(
                self, basename, self._fmt_duration(probe.duration_s)
            )
            return

        # (3.25) quick-260615-f77 — canonical-rate normalization. The
        # mastering chain now hard-requires sr=48000 (chain.py guard,
        # reverses Phase 2.1 D-04). 48 kHz sources are used as-is (the
        # common case). A non-48 kHz source (e.g. 44.1 kHz, rare) is
        # converted to a 48 kHz RF64 working file HERE, BEFORE any
        # downstream branch (playback prime decision, proxy build, cache
        # key, analysis), so the entire flow operates on the canonical
        # rate and the converted keepers master successfully.
        if probe.sample_rate != CANONICAL_SAMPLE_RATE:
            working_root = default_cache_root() / "resampled48k"
            try:
                working_root.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            working_key = proxy_cache.cache_key(p)
            working_path = working_root / f"{working_key}.wav"

            # Reuse a fresh working file (opening the same 44.1 kHz source
            # twice must not reconvert). cache_key folds in source mtime +
            # size, so a stale key never collides with a changed source.
            need_convert = not working_path.exists()
            if need_convert:
                # Synchronous convert on the GUI thread for v1 — mirrors
                # the build_bundle "small enough for v1" precedent. A wait
                # cursor signals the (rare) conversion; an 8 h non-48 kHz
                # source would block the UI, which is acceptable for the
                # rare path until user feedback demands the async
                # _spawn_audio_proxy_worker pattern (quick-260615-f77).
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.statusBar().showMessage(
                    f"Converting {basename} to 48 kHz…"
                )
                try:
                    resample_to_48k(p, working_path)
                except BuildCancelled:
                    # Aborted convert — treat as an aborted open.
                    return
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "Could not convert audio",
                        f"Failed to convert {basename} to 48 kHz.\n{exc}",
                    )
                    return
                finally:
                    QApplication.restoreOverrideCursor()
                    self.statusBar().clearMessage()

            # Rebind EVERY local the downstream flow consumes to the
            # converted 48 kHz working file, then re-prime playback (the
            # earlier prime at the top of _open_file used the source path).
            p = working_path
            path = str(working_path)
            basename = self._basename(working_path)
            probe = audio_file.probe(working_path)
            try:
                self._playback_engine.prime(str(working_path))
            except Exception:
                pass

        # (3.5) Plan 02.1-04 — audio-proxy path resolution (D-07): the
        # MainWindow owns the "which audio path to prime" decision; the
        # PlaybackEngine stays format-agnostic. Four branches:
        #
        #   * WAV source (D-05 / SC-6) — skip the proxy entirely. The
        #     existing ``prime(str(path))`` above already primed with the
        #     source path; just enable the spacebar synchronously.
        #
        #   * Cache HIT (D-13 / SC-5) — re-prime with the proxy path so
        #     playback uses the canonical float32-stereo WAV (constant-time
        #     seek). NO worker spawned, NO progress UI shown, NO
        #     ``audio_proxy_complete`` emission (the test seam fires on
        #     BUILD completion only).
        #
        #   * Disk-preflight failure (D-14) — friendly QMessageBox.warning,
        #     spacebar left disabled (no proxy → can't constant-time seek),
        #     fall through to the waveform pipeline so the user can still
        #     inspect the file visually.
        #
        #   * MISS — enqueue ``AudioProxyRunnable``; disable spacebar
        #     until ``signals.finished`` arrives (Option 1 UX from
        #     CONTEXT). ``_on_audio_proxy_finished`` re-primes with the
        #     proxy path and re-enables the spacebar.
        #
        # The branch falls through (does NOT return) so the existing
        # waveform-proxy pipeline below still renders the visual layer.
        if p.suffix.lower() == ".wav":
            # D-05 — WAV-source skip. ``prime(source_path)`` already ran
            # above; just enable the spacebar synchronously. Playback reads
            # the source directly (native WAV is already O(1) seek).
            self._current_playback_path = p
            self._shortcut_play_pause.setEnabled(
                self._playback_engine.is_available
            )
        else:
            audio_cache_root = default_cache_root()
            fresh_proxy = audio_proxy_is_fresh(audio_cache_root, p)
            if fresh_proxy is not None:
                # D-13 — cache HIT. Re-prime with the canonical proxy.
                try:
                    self._playback_engine.prime(str(fresh_proxy))
                except Exception:
                    pass
                # Playback must go through the proxy WAV, not the source
                # MP3 — otherwise pedalboard's O(n) MP3 seek defeats the
                # whole phase (SC-4).
                self._current_playback_path = fresh_proxy
                self._shortcut_play_pause.setEnabled(
                    self._playback_engine.is_available
                )
                self._update_cache_size_footer()
            else:
                # D-14 — disk-preflight gate.
                expected = expected_proxy_bytes(
                    probe.duration_s, probe.sample_rate
                )
                ok, needed, free = check_disk_space(
                    audio_cache_root, expected
                )
                if not ok:
                    # Friendly error per RESEARCH §"Error message wording".
                    QMessageBox.warning(
                        self,
                        "Not enough disk space",
                        "Not enough disk space for audio proxy.\n"
                        f"Need: ~{needed / 1024**3:.1f} GiB ({basename})\n"
                        f"Free: {free / 1024**3:.1f} GiB\n"
                        "Free some space or use File → "
                        "Clear audio proxy cache and try again.",
                    )
                    # Continue with the waveform render so the user can
                    # still inspect visually — playback stays disabled.
                    self._shortcut_play_pause.setEnabled(False)
                else:
                    # MISS — spawn the audio-proxy worker.
                    self._spawn_audio_proxy_worker(p, probe, gen)
                    self._shortcut_play_pause.setEnabled(False)

        # (4) Cache lookup.
        cache_root = default_cache_root()
        key = proxy_cache.cache_key(p)
        proxy_p = proxy_cache.proxy_path(cache_root, key)
        # Plan 03-04b — stash source path + cache_key for the export
        # pipeline. The Phase 2.1 ``_current_playback_path`` (set on the
        # WAV-skip / cache-HIT / audio-proxy success branches above) is
        # the proxy path passed to ExportRunnable; these two are used by
        # ``naming_resolver`` for the filename pattern and dominant-trait
        # lookup respectively.
        self._current_source_path = p
        self._current_cache_key = key

        # (5) Cache HIT branch — try load_proxy first.
        if proxy_p.exists():
            try:
                arr, header = proxy_cache.load_proxy(proxy_p)
                self._render_loaded_proxy(arr, header, probe, p)
                # Plan 03-01 — load sidecar (REG-04). Runs AFTER
                # _render_loaded_proxy so the WaveformView's _duration_s is
                # populated (RegionsOverlay reads it lazily for bounds).
                self._load_sidecar_for_key(cache_root, key)
                # render_complete fires on BOTH HIT and MISS success branches.
                self.render_complete.emit()
                # Plan 02-04 — D-15 lazy compute. The heatmap pipeline is
                # NOT auto-fired here. The sidebar's Energy checkbox is
                # the user's trigger; until they toggle it, no worker
                # spawns and no heatmap activity happens. The checkbox
                # itself was reset to unchecked by the cancel preamble at
                # the top of this method.
                return
            except (ProxyHeaderError, OSError, ValueError, MemoryError):
                # CR-03 fix: OSError / ValueError / MemoryError can come from
                # np.memmap on systems with limited virtual-address space
                # (containers, sandboxes, 32-bit Python); treat them as
                # cache-file-corruption and rebuild from source. ProxyHeaderError
                # is the in-bounds header-corruption case from Plan 01-07.
                # All four collapse to the same disposition: delete the bad
                # cache file (robust to a concurrent deletion via
                # missing_ok=True) and fall through to MISS rebuild.
                proxy_p.unlink(missing_ok=True)

        # (6) Cache MISS — start the worker.
        proxy_p.parent.mkdir(parents=True, exist_ok=True)
        runnable = PeakBuilderRunnable(p, proxy_p)
        self._current_runnable = runnable
        # Stash the probe and path so _on_proxy_ready can finish the render.
        self._current_probe = probe
        self._current_path = p
        self._current_proxy_p = proxy_p

        # Show the overlay with the UI-SPEC body — but ONLY if the audio
        # proxy isn't already owning it. When opening a non-WAV cache-MISS
        # file, audio proxy spawned just above this block (line ~793) and
        # has put its modal body on the overlay. Overwriting it here would
        # erase the audio-proxy progress UI and the user would see the
        # fast waveform build text instead of the long audio-proxy build
        # text. Waveform proxy runs silently behind the overlay until it
        # finishes; the cancel button is wired to the audio runnable, not
        # the waveform one (waveform is fast — cancelling it isn't a
        # user-meaningful action while audio proxy is the long-running
        # blocker).
        if not self._audio_proxy_overlay_active:
            self._overlay.set_heading("Preparing waveform")
            self._overlay.set_body(
                f"{basename} · {self._fmt_duration(probe.duration_s)} · "
                "first open — building a downsampled proxy. "
                "This may take up to a minute for an 8-hour file."
            )
            self._overlay.set_progress(0)
            self._overlay.resize_to_parent()
            self._overlay.show()
            self._overlay.raise_()

        # Connect signals — fresh connections per build so a stale signal
        # from an earlier cancelled build can never trigger us.
        # CR-04: capture `gen` and `runnable` by value (default-arg trick) so
        # a later _open_file overwriting `self._open_generation` /
        # `self._current_runnable` does NOT mutate the closure. A bare
        # `lambda obj: self._on_proxy_ready(gen, runnable, obj)` would
        # late-bind both free variables and break the guard.
        #
        # We retain the QMetaObject.Connection tokens on the runnable so
        # CR-02's disconnect-on-cancel / disconnect-on-close can remove
        # ONLY our wiring (not, e.g., a test's qtbot.waitSignal listener
        # that may also be attached to the same signal). A bare
        # `signals.<x>.disconnect()` would nuke every connection.
        # Only route waveform progress to the overlay if it's not owned
        # by the audio proxy (HUMAN-UAT #3 — audio proxy is the user's
        # blocker; waveform progress is incidental).
        if not self._audio_proxy_overlay_active:
            runnable.signals.progress.connect(self._overlay.set_progress)
        runnable._mw_conn_finished = runnable.signals.finished.connect(
            lambda obj, g=gen, r=runnable: self._on_proxy_ready(g, r, obj)
        )
        runnable._mw_conn_error = runnable.signals.error.connect(
            lambda msg, g=gen, r=runnable: self._on_proxy_error(g, r, msg)
        )
        runnable._mw_conn_cancelled = runnable.signals.cancelled.connect(
            lambda g=gen, r=runnable: self._on_proxy_cancelled(g, r)
        )
        # Connect the overlay's cancel button to this runnable's cancel.
        # Disconnect any previous wiring first. PySide6 emits a
        # RuntimeWarning when there is no existing connection — this is
        # cosmetic; the actual disconnect is wrapped in try/except so the
        # warning does not block functionality, and the new connect()
        # below always wires the current runnable.
        # Only rebind the cancel button if the audio proxy isn't using
        # it. The audio proxy's spawn block (earlier in _open_file) has
        # already wired the button to its own runnable; cancellation of
        # the long-running build is what the user sees as "cancel" from
        # the modal.
        if not self._audio_proxy_overlay_active:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    self._overlay.cancel_button.clicked.disconnect()
                except (RuntimeError, TypeError):
                    pass
            self._overlay.cancel_button.clicked.connect(runnable.cancel)

        QThreadPool.globalInstance().start(runnable)

    # ------------------------------------------------------- worker callbacks
    # -------------------------------------------------- Plan 03-01 sidecar
    def _load_sidecar_for_key(self, cache_root: Path, key: str) -> None:
        """Load the sidecar JSON for ``key`` and populate the regions overlay.

        Called from BOTH the cache-HIT (synchronous) and the cache-MISS
        (``_on_proxy_ready``) branches AFTER the WaveformView's
        ``_duration_s`` is set so the overlay's bounds are clamped to the
        source range. ``load_sidecar`` never raises — failures quarantine
        the file and return ``[]`` (D-A3-5).
        """
        try:
            sp = sidecar_path(cache_root, key)
        except ValueError:
            # _KEY_RE rejected the key — should not happen since cache_key
            # produced it, but degrade silently.
            self._current_sidecar_path = None
            self._regions_overlay.set_regions([])
            # quick-260701-jc5 — keep marker state consistent on the degrade.
            self._current_markers = []
            self._markers_overlay.clear()
            self._markers_sidebar.clear()
            return
        self._current_sidecar_path = sp
        # quick-260701-jc5 — load_sidecar now returns (regions, markers). The
        # markers side is captured + pushed to the panel/overlay in Task 3's
        # marker wiring; here we unpack it so the region path stays correct.
        regions, markers = sidecar_cache.load_sidecar(sp)
        self._regions_overlay.set_regions(regions)
        # quick-260701-jc5 (MARK-05) — repopulate the Markers panel + overlay
        # from the freshly loaded marker list. This is the seam that makes
        # markers survive close/reopen. Store the list as the single source of
        # truth so subsequent mutations persist the FULL set.
        self._current_markers = list(markers)
        self._markers_overlay.set_markers(markers)
        self._markers_sidebar.clear()
        for _m in markers:
            self._markers_sidebar.add_row(_m)
        # Phase 7 Plan 07-03 Task 2 — legacy-migration auto-snapshot.
        # Pre-Phase-7 sidecars persisted keepers with ``mastering=None``
        # (the field is additive per D-19). On first load, populate
        # those keepers with a fresh session-chain snapshot and
        # re-persist the sidecar EXACTLY ONCE. Subsequent loads see the
        # populated field and skip the migration branch entirely
        # (snapshot-not-link semantics preserved per D-04).
        migrated_any = self._snapshot_session_into_unmastered_keepers()
        if migrated_any:
            # quick-260701-jc5 — carry markers through the migration re-write
            # so a legacy-keeper auto-snapshot does not drop the marker array.
            sidecar_cache.save_sidecar(
                sp,
                self._regions_overlay.regions_data(),
                self._current_markers,
            )
        # Plan 03-02 — repopulate the Keepers dock from the freshly loaded
        # region list. ``set_regions`` does NOT emit ``regions_changed``
        # (Plan 01 design — that signal is reserved for user-initiated
        # mutations so it doesn't trigger a save-after-load cycle), so we
        # explicitly refresh the dock here on file open.
        if hasattr(self, "_keepers_sidebar"):
            self._refresh_keepers_dock()
        # Plan 03-03 — push the initial Trash skip ranges into the engine.
        # ``set_regions`` does NOT emit ``regions_changed`` so the wiring in
        # ``_on_regions_changed`` doesn't fire automatically; we drive it
        # here to keep file-open state consistent with what the user marked
        # in a prior session.
        self._push_trash_ranges_to_engine()

    def _on_regions_changed(self) -> None:
        """Atomic sidecar save + Keepers refresh on any region mutation.

        Plan 01 contract (D-A3-4): atomic write of the in-memory region
        list to the sidecar JSON path captured at file-open time, then
        public ``regions_changed(str)`` test-seam emission. Plan 02
        EXTENDS this slot to also rebuild the right-side Keepers dock
        from the current region list so the panel stays in sync after
        every create / state-change / edge-resize / delete.

        Phase 7 Plan 07-03 — snapshot-at-keeper-creation hook (D-04).
        Before saving the sidecar, populate ``mastering`` on every
        keeper whose field is still None (newly-marked-as-keeper
        regions inherit a session-chain snapshot exactly once). The
        in-memory overlay is mutated first so the subsequent
        ``regions_data()`` call serializes the populated dict to disk.

        Bails if no source is open (``_current_sidecar_path is None``) —
        the overlay's :class:`pg.LinearRegionItem.sigRegionChangeFinished`
        signal can fire during a teardown sequence, and we must not
        write a sidecar for a non-existent source. The Keepers refresh
        still runs (it's a UI-only no-op when the sidebar is empty).
        """
        if self._current_sidecar_path is not None:
            # D-04 — snapshot newly-marked keepers BEFORE save so the
            # on-disk Region carries the snapshot from creation onward.
            self._snapshot_session_into_unmastered_keepers()
            regions = self._regions_overlay.regions_data()
            # quick-260701-jc5 — pass the live markers so a region-only
            # mutation does not drop markers from the sidecar.
            sidecar_cache.save_sidecar(
                self._current_sidecar_path, regions, self._current_markers
            )
            self.regions_changed.emit(str(self._current_sidecar_path))
        # Plan 03-02 — refresh Keepers sidebar to match current state.
        # Constructor-order safety: this slot is wired in __init__ AFTER
        # _build_right_dock has run, so _keepers_sidebar always exists by
        # the time the first regions_changed fires. The guard is for the
        # narrow window between the overlay's hover_changed emit and the
        # right dock build during partial init.
        if hasattr(self, "_keepers_sidebar"):
            self._refresh_keepers_dock()
        # Plan 03-03 (D-A2-3) — Trash playback skip. Runs on every regions
        # mutation so a Trash mark applies immediately to playback.
        self._push_trash_ranges_to_engine()

    def _push_trash_ranges_to_engine(self) -> None:
        """Push trash_minus_keepers ranges to PlaybackEngine.set_skip_ranges (D-A2-3).

        The GUI tier owns the Keeper-punch-through subtraction (D-A2-5)
        — the engine sees a flat, sorted, non-overlapping list of skip
        ranges. Wraps the engine call in try/except so a transient
        engine-state issue never blocks region UX.
        """
        if not hasattr(self, "_playback_engine") or self._playback_engine is None:
            return
        skip_ranges = self._regions_overlay.trash_minus_keepers()
        try:
            self._playback_engine.set_skip_ranges(skip_ranges)
        except Exception:
            pass

    def _refresh_keepers_dock(self) -> None:
        """Rebuild Keepers rows from current region list (Plan 03-02).

        UI-SPEC §Layout Architecture: the Keepers panel shows ONLY
        Keeper-state regions, sorted chronologically by ``start_sec``.
        We tear down + repopulate rather than diffing because the
        region count is bounded by Plan 01's ``_MAX_REGIONS = 4096``
        schema cap and the rebuild is O(N) at worst — well under one
        frame for any realistic session.

        Phase 7 Plan 07-02 Task 4 — also set the initial divergence
        badge state per keeper (none / check / star) based on the
        keeper's mastering field vs the current session-chain hash.
        Computed once per refresh; subsequent Apply paths update the
        badge directly via :meth:`_on_mastering_config_applied`.
        """
        self._keepers_sidebar.clear()
        regions = self._regions_overlay.regions_data()
        keeper_regions = sorted(
            (reg for reg in regions if reg.state == "keeper"),
            key=lambda x: x.start_sec,
        )
        if not keeper_regions:
            return
        # Compute session hash once for all keepers in this refresh.
        try:
            session_hash = config_hash(load_session_chain_snapshot())
        except Exception:
            session_hash = None
        for r in keeper_regions:
            row = self._keepers_sidebar.add_row(r)
            if r.mastering is None:
                row.set_mastering_badge("none")
            elif session_hash is not None and config_hash(r.mastering) == session_hash:
                row.set_mastering_badge("check")
            else:
                row.set_mastering_badge("star")

    # --------------------------- Phase 7 Plan 07-03 — dock + snapshot slots

    # Phase 7 Plan 07-07 iter-4 — the toggle slot
    # ``_on_mastering_panel_action_toggled`` and the visibility-sync slot
    # ``_on_mastering_dock_visibility_changed`` were removed alongside the
    # View → Mastering panel QAction. The Mastering dock is always visible
    # in the LEFT dock area (see ``_build_left_dock``). Qt's main-window
    # dock-area context menu still allows undocking/re-docking individual
    # docks without a top-level View toggle.

    def _on_session_chain_changed(self) -> None:
        """MasteringDock edited → refresh divergence badges (D-04 no-mutate).

        D-04 — session-chain edits do NOT propagate to existing keepers.
        This slot only re-computes each keeper row's badge state by
        comparing ``config_hash(keeper.mastering) ==
        config_hash(session_snapshot)``. The keepers' mastering dicts
        are NEVER mutated by this slot.
        """
        try:
            session_hash = config_hash(load_session_chain_snapshot())
        except Exception:
            session_hash = None
        for r in self._regions_overlay.regions_data():
            if r.state != "keeper":
                continue
            row = self._keepers_sidebar.find_row(r.id)
            if row is None:
                continue
            if r.mastering is None:
                row.set_mastering_badge("none")
            elif (
                session_hash is not None
                and config_hash(r.mastering) == session_hash
            ):
                row.set_mastering_badge("check")
            else:
                row.set_mastering_badge("star")

        # Phase 7 Plan 07-04 — session edits do not change keeper.mastering
        # (D-04), but they may shift the visible badge OR the user's
        # mental model of what "A/B" means. Re-evaluate the toggle so
        # state stays consistent.
        self._refresh_ab_toggle_enabled_state()

    def _snapshot_session_into_unmastered_keepers(self) -> bool:
        """Populate ``mastering`` on every keeper whose field is None.

        Used by two call sites:

        1. ``_on_regions_changed`` — when a region just transitioned to
           keeper state, it has ``mastering=None`` from creation; the
           snapshot lands BEFORE the sidecar save so the on-disk
           Region carries the snapshot (D-04 — snapshot-at-creation).
        2. ``_load_sidecar_for_key`` — legacy-migration path for
           pre-Phase-7 sidecars whose keepers persisted before the
           snapshot hook existed.

        D-04: each keeper is snapshotted EXACTLY ONCE. Subsequent
        session edits never re-snapshot — the keeper's mastering dict
        is an independent copy from that moment forward.

        Returns:
            ``True`` if at least one keeper was newly populated;
            ``False`` otherwise (caller can skip a redundant save).
        """
        any_snapshotted = False
        snapshot: dict | None = None
        for r in self._regions_overlay.regions_data():
            if r.state != "keeper":
                continue
            if self._regions_overlay.get_mastering(r.id) is not None:
                continue
            # Lazy-load the snapshot once per call so repeated invocations
            # share the same QSettings read.
            if snapshot is None:
                snapshot = load_session_chain_snapshot()
            # Deep-copy so each keeper has an independent dict (D-04 —
            # subsequent edits to one keeper's chain must not leak into
            # other keepers).
            import copy as _copy

            self._regions_overlay.set_mastering(r.id, _copy.deepcopy(snapshot))
            any_snapshotted = True
        return any_snapshotted

    # ----------------------------------- Plan 03-02 region-mode + Edit slots
    def _on_region_select_toggled(self, checked: bool) -> None:
        """Toolbar Region Select QAction.toggled handler (D-A1-1).

        Forwards the toggle straight to :meth:`WaveformView.set_region_select_mode`
        which flips the gesture-handler branch and swaps the viewport
        cursor. Kept as a separate slot so future polish (e.g. status
        bar hint "Region select: ON — drag to mark") can hook here
        without re-touching the toolbar build.
        """
        self._waveform_view.set_region_select_mode(bool(checked))

    def _on_overlay_hover_changed(self, region_id) -> None:
        """Enable/disable Edit-menu actions based on hover state.

        Wired to :attr:`RegionsOverlay.hover_changed` (payload = region
        id or None). When None (mouse left the region), the four Edit
        actions are disabled so the user can't trigger a no-op
        K/T/U/Delete with no target. When non-None, they enable —
        Qt's WindowShortcut context already suppresses the QAction's
        shortcut while a QLineEdit has focus, so the user can type
        K/T/U into a note input without triggering region mutation.

        Plan 03-07 — also refresh the Export submenu's enable/disable
        because its rule (keeper-only) is a strict subset of the K/T/U/
        Delete hover rule.
        """
        enabled = region_id is not None
        self._action_mark_keeper.setEnabled(enabled)
        self._action_mark_trash.setEnabled(enabled)
        self._action_unmark.setEnabled(enabled)
        self._action_delete_region.setEnabled(enabled)
        # Plan 03-07 — Export submenu has the extra keeper-only gate.
        self._refresh_export_hovered_actions_enabled()

    def _refresh_export_hovered_actions_enabled(self) -> None:
        """Enable Edit > Export submenu only when a Keeper is hovered.

        Plan 03-07 gap-closure (UAT Test 7). Called from:

        - :meth:`_on_overlay_hover_changed` (hover transitions in/out of
          regions).
        - :attr:`RegionsOverlay.regions_changed` (state mutations on the
          hovered region — e.g. user presses T while hovering a Keeper,
          demoting it to Trash).

        Reads the live ``_current_state`` from the hovered region widget
        so a mid-hover state change correctly toggles the export
        availability. CONTEXT D-A4-4 LOCKED dual-format keeper-only rule.
        """
        rid = self._regions_overlay.hovered_region_id
        enabled = False
        if rid is not None:
            region = self._regions_overlay.get_region(rid)
            if region is not None and region._current_state == "keeper":
                enabled = True
        self._action_export_hovered_mp3.setEnabled(enabled)
        self._action_export_hovered_wav.setEnabled(enabled)

    def _on_export_hovered_region(self, fmt: str) -> None:
        """Edit > Export submenu trigger — Plan 03-07 fallback entry point.

        Reads the currently-hovered region id and delegates to the same
        :meth:`_on_export_region_requested` slot that the right-click
        context menu uses (single export pipeline, three entry points
        per Plan 03-07 UI-SPEC §Export flow).

        QLineEdit-focus defense-in-depth: per RESEARCH §Pitfall #5, bail
        if a note QLineEdit has focus — the user is typing into a Keeper
        note input and accidentally clicked the menu. Mirrors
        :meth:`_mark_hovered_region`'s bail-out (above).

        Race-safety: re-check the keeper state at trigger time. The
        setEnabled(True) call happened at hover time; if the user demoted
        the region between then and the menu click, the trigger must
        still bail.
        """
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            return
        rid = self._regions_overlay.hovered_region_id
        if rid is None:
            return
        region = self._regions_overlay.get_region(rid)
        if region is None or region._current_state != "keeper":
            return
        self._on_export_region_requested(rid, fmt)

    def _mark_hovered_region(self, state: str) -> None:
        """Apply ``state`` to the currently-hovered region (no-op when none).

        Per RESEARCH §Pitfall #5: defense-in-depth check that the focus
        widget is NOT a QLineEdit. The Edit menu QActions use the
        default ``Qt.ShortcutContext.WindowShortcut`` which already
        suppresses the shortcut when a QLineEdit consumes the key, but
        a user clicking the menu item by mouse while a note input has
        focus would still trigger this slot. The bail-out here makes the
        contract symmetric: while the user is typing a note, K/T/U
        always type into the note, never mutate a region.
        """
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            return
        self._regions_overlay.set_state_of_hovered(state)
        # Sidecar save + Keepers refresh happens via regions_changed
        # signal (wired in __init__).

    def _delete_hovered_region(self) -> None:
        """Delete the currently-hovered region — same QLineEdit gate as above."""
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            return
        self._regions_overlay.delete_hovered()

    # ---------------------------------------------- quick-260701-jc5 markers
    def _action_add_marker(self) -> None:
        """Drop a marker at the current playhead ("m" key AND [+] button).

        quick-260701-jc5 (MARK-01). Locked decision #2 — the [+] button and
        the "m" shortcut share this single position source. Bails when:

        * a QLineEdit has focus (the marker-label / keeper-note fields keep
          the "m" keystroke — mirrors :meth:`_mark_hovered_region`); OR
        * no file is open (no playback path or no sidecar path).

        The playhead position is clamped to ``[0, duration_seconds]`` so a
        stray out-of-range read can never write an invalid ``time_sec``.
        """
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            return
        if (
            self._current_playback_path is None
            or self._current_sidecar_path is None
        ):
            return
        try:
            pos = float(self._playback_engine.position_seconds)
        except Exception:
            pos = 0.0
        try:
            duration = float(self._playback_engine.duration_seconds)
        except Exception:
            duration = 0.0
        if duration > 0.0:
            pos = max(0.0, min(pos, duration))
        else:
            pos = max(0.0, pos)
        marker = Marker(id=uuid.uuid4().hex, time_sec=pos, label="")
        self._current_markers.append(marker)
        row = self._markers_sidebar.add_row(marker)
        self._markers_overlay.add_marker(marker)
        self._persist_markers()
        # Focus the new marker's label field so the user can type the label
        # right away (quick — same for the "m" key and the [+] button).
        row.focus_label()

    def _on_marker_jump(self, marker_id: str) -> None:
        """MarkerRow click → seek the playhead to the marker + start playback.

        Mirrors the seek+play sequence in :meth:`_on_keeper_play` (A-mode
        source-proxy playback). Bails when no file/source is open.
        """
        if self._current_playback_path is None:
            return
        marker = next(
            (m for m in self._current_markers if m.id == marker_id), None
        )
        if marker is None:
            return
        target = float(marker.time_sec)
        try:
            self._playback_engine.seek(target)
        except Exception:
            pass
        try:
            self._playback_engine.play(
                str(self._current_playback_path), start_seconds=target
            )
        except Exception:
            pass

    def _on_marker_delete(self, marker_id: str) -> None:
        """Delete a marker — remove the row, the overlay line, and persist."""
        self._current_markers = [
            m for m in self._current_markers if m.id != marker_id
        ]
        self._markers_sidebar.remove_row(marker_id)
        self._markers_overlay.remove_marker(marker_id)
        self._persist_markers()

    def _on_marker_label_edited(self, marker_id: str, label: str) -> None:
        """Persist an edited marker label + update the overlay line's text."""
        # Clamp to the sidecar cap so the persisted value never quarantines.
        clamped = (label or "")[:200]
        for m in self._current_markers:
            if m.id == marker_id:
                m.label = clamped
                break
        self._markers_overlay.update_label(marker_id, clamped)
        self._persist_markers()

    def _persist_markers(self) -> None:
        """Atomic sidecar save of BOTH regions and markers (one write).

        Bails when no source is open. Reuses the SAME sidecar path as the
        region save so markers and regions round-trip together — a region
        mutation therefore preserves markers and vice-versa.
        """
        if self._current_sidecar_path is None:
            return
        sidecar_cache.save_sidecar(
            self._current_sidecar_path,
            self._regions_overlay.regions_data(),
            self._current_markers,
        )

    def _on_keeper_jump(self, region_id: str) -> None:
        """KeeperRow single-click → seek the playhead to the region's start.

        Reads the region's start_sec from the overlay's internal map
        (the LinearRegionItem's ``getRegion()`` returns the live edges,
        which may differ from the row's start_sec snapshot if the user
        resized between the click and now). PlaybackEngine.seek is
        wrapped in a try/except so an un-primed engine (no file open)
        degrades silently — a stray jump_requested before a file is
        loaded is a UX no-op, not an exception.
        """
        region = self._regions_overlay.get_region(region_id)
        if region is None:
            return
        start_sec, _end_sec = region.getRegion()
        try:
            self._playback_engine.seek(float(start_sec))
        except Exception:
            pass

    def _on_keeper_play(self, region_id: str, start_mode: str = "start") -> None:
        """KeeperRow play button / double-click → smart playback.

        Plan 07-10e — extended from the Phase 2 seek-and-resume idiom;
        quick-260622-tit removed the pause behavior:

        * If the keeper has a fresh mastered cache → switch toggle to B,
          select the keeper, play the CACHE from the per-mode offset. The
          just-clicked button gets the active highlight.
        * If no cache (or cache stale) → fall back to A-mode, play the
          source proxy starting at the per-mode offset.
        * Clicking ANY keeper button (including the same one again) always
          fires its action immediately — it NEVER pauses. The highlight
          moves to the just-clicked button; the previous one clears.

        Requires ``_current_playback_path`` to be set (WAV-skip /
        cache-HIT / audio-proxy success branches all set it); a play
        request before that is a UX no-op.
        """
        region = self._regions_overlay.get_region(region_id)
        if region is None or self._current_playback_path is None:
            return
        start_sec, end_sec = region.getRegion()

        # quick-260621-iuc — keeper Play must STOP at the segment end AND apply
        # the same linear fade as the export/master path.
        # quick-260626-o9y — the fade is now CONFIG-DRIVEN: read the keeper's
        # fade enabled flag + duration from its mastering config (single source
        # of truth fade_params) instead of a forced 2.0 s. When fade is disabled
        # fade_sec is 0.0 (no fade at either end). Otherwise clamp the configured
        # duration to half the region so the in/out fades never overlap.
        region_duration = float(end_sec) - float(start_sec)
        mastering = self._regions_overlay.get_mastering(region_id)
        fade_enabled, fade_dur = fade_params(mastering)
        fade_sec = min(fade_dur, region_duration / 2.0) if fade_enabled else 0.0

        # quick-260622-sr8 — Play=start / middle=middle / end=ending. The pure
        # helper is the single source of truth for the start offset + whether
        # the fade-IN is suppressed (middle mode only). fade_sec stays computed
        # at this call site (per-mode application below).
        offset_start, suppress_fade_in = _keeper_play_offsets(
            float(start_sec), float(end_sec), start_mode
        )
        # quick-260622-ud0 — the engine now applies asymmetric per-end fades,
        # so suppressing the fade-IN no longer drops the fade-OUT. Fade-in is
        # dropped for middle/end (suppress_fade_in True); fade-out is ALWAYS
        # applied for every mode.
        play_fade_in = 0.0 if suppress_fade_in else fade_sec
        play_fade_out = fade_sec

        # quick-260622-vwr — read the keeper's normalize state ONCE, before the
        # cache_path branch. A-mode applies the SAME WYSIWYG affine the
        # normalized waveform render uses; B-mode ignores it (the mastered
        # cache already baked normalize in as the chain's final stage).
        norm_enabled, norm_target_db = self._regions_overlay.get_normalize(region_id)

        # quick-260622-tit — no pause toggle. Clicking any button (even the
        # one already active) always re-fires its playback action below;
        # the highlight is then moved to the just-clicked button.

        # Check for a fresh mastered cache for this keeper.
        cache_path = None
        try:
            src_key = proxy_cache.cache_key(self._current_path)
            # quick-260626-o9y — ``mastering`` was already read above for the
            # config-driven fade; reuse it here for the cache lookup.
            if mastering is not None:
                chash = config_hash(mastering)
                candidate = mastered_cache_path(
                    default_cache_root(), src_key, region_id, chash
                )
                if is_mastered_cache_fresh(candidate):
                    cache_path = candidate
        except Exception:
            cache_path = None

        if cache_path is not None:
            # B-mode path: cache available — play it from offset 0
            # (== keeper start in source-time).
            self._selected_keeper_id = region_id
            self._refresh_ab_toggle_enabled_state()
            if self._ab_toggle.state != "B":
                self._ab_failclosed_in_progress = True
                try:
                    self._ab_toggle.set_state("B")
                finally:
                    self._ab_failclosed_in_progress = False
            try:
                # quick-260621-iuc — the mastered cache IS the keeper segment
                # (starts at 0, length ≈ region_duration) and is UN-faded:
                # mastering applies no fades (those live only in the export
                # stage). So we fade HERE. No double-fade risk. end_seconds=None
                # plays to the cache's natural EOF and the engine computes the
                # fade-out relative to _duration_frames.
                # quick-260622-sr8 — the cache spans the keeper
                # [start_sec, end_sec] mapped to cache-time [0, duration].
                # Translate the source-time offset into cache-time.
                # quick-260622-ud0 — pass asymmetric fades: fade-in only for
                # start mode (play_fade_in), fade-out for every mode.
                cache_offset = offset_start - float(start_sec)
                # quick-260622-vwr — B-mode passes the DEFAULT affine
                # (0.0 / 1.0): the mastered cache ALREADY has normalize baked
                # in as the mastering chain's FINAL stage (chain.py applies
                # normalize_array last), so re-applying it here would
                # DOUBLE-normalize.
                self._playback_engine.play(
                    str(cache_path),
                    start_seconds=float(cache_offset),
                    end_seconds=None,
                    fade_in_seconds=play_fade_in,
                    fade_out_seconds=play_fade_out,
                )
                self._playback_timer.start()
            except Exception:
                return
            # Visual playhead at the source-time offset (start/middle/end).
            for line in self._lane_playheads.values():
                line.setValue(float(offset_start))
        else:
            # A-mode fallback: no fresh cache → source proxy at start_sec.
            if self._ab_toggle.state != "A":
                self._ab_failclosed_in_progress = True
                try:
                    self._ab_toggle.set_state("A")
                finally:
                    self._ab_failclosed_in_progress = False
            try:
                # quick-260621-iuc — A-mode plays the source proxy windowed to
                # the keeper [start_sec, end_sec) with the same auto-scaled
                # fade as the export. The engine stops at end_sec (sets EOF →
                # CallbackStop) and the existing _on_playback_tick cleanup
                # clears the row glyph; no new GUI stop logic.
                # quick-260622-sr8 — start at the per-mode offset; stop-at-end
                # (end_seconds) stays for ALL modes.
                # quick-260622-ud0 — asymmetric fades: middle/end drop the
                # fade-in (play_fade_in=0.0) but ALL modes keep the fade-out
                # (play_fade_out=fade_sec).
                # quick-260622-vwr — when Normalize is ON, compute the SAME
                # DC-remove + peak-to-target affine the WYSIWYG waveform render
                # applies, over the FULL keeper span [start_sec, end_sec] (NOT
                # the audition sub-window) so the gain is constant across
                # play/middle/end and matches the whole-span display normalize.
                # Computed from _current_playback_path (the proxy A-mode plays)
                # so the domains match. Normalize OFF -> identity affine.
                if norm_enabled:
                    norm_dc, norm_scale = (
                        self._playback_engine.compute_segment_normalize_params(
                            str(self._current_playback_path),
                            float(start_sec),
                            float(end_sec),
                            float(norm_target_db),
                        )
                    )
                else:
                    norm_dc, norm_scale = 0.0, 1.0
                self._playback_engine.play(
                    str(self._current_playback_path),
                    start_seconds=float(offset_start),
                    end_seconds=float(end_sec),
                    fade_in_seconds=play_fade_in,
                    fade_out_seconds=play_fade_out,
                    normalize_dc=norm_dc,
                    normalize_scale=norm_scale,
                )
                self._playback_timer.start()
            except Exception:
                return
            for line in self._lane_playheads.values():
                line.setValue(float(offset_start))

        self._currently_playing_keeper_id = region_id
        self._currently_playing_mode = start_mode
        self._refresh_keeper_row_play_icons()

    def _refresh_keeper_row_play_icons(self) -> None:
        """Sync the per-keeper-row active highlight with engine state.

        Plan 07-10e / quick-260622-tit — highlights the just-clicked
        button (start / middle / end) on the row matching
        ``_currently_playing_keeper_id`` (only while the engine is
        playing) via ``set_active_mode(self._currently_playing_mode)``
        and clears the highlight on every other row with
        ``set_active_mode(None)``. There is no pause glyph; active state
        is conveyed purely by the highlight. Defensive: silent no-op
        when the sidebar is missing the row (e.g., user just deleted the
        keeper).
        """
        active_id = (
            self._currently_playing_keeper_id
            if self._playback_engine.is_playing
            else None
        )
        for region_data in self._regions_overlay.regions_data():
            row = self._keepers_sidebar._rows.get(region_data.id)
            if row is None:
                continue
            if region_data.id == active_id:
                row.set_active_mode(self._currently_playing_mode)
            else:
                row.set_active_mode(None)

    def _set_playing_row_highlight(self, pos: float) -> None:
        """Tint the keeper row the playhead is currently inside (quick-260625).

        Whole-row "now playing" highlight, distinct from the per-button accent
        in :meth:`_refresh_keeper_row_play_icons`. Driven by the 30 Hz playhead
        tick with the SOURCE-time ``pos`` so the tint follows the playhead into
        whatever keeper region it is passing through (full-file playback as well
        as a single keeper audition). When the playhead is not inside any
        keeper that has a sidebar row, falls back to the explicitly-played
        keeper so a keeper audition stays lit through inter-region gaps. Each
        row's :meth:`KeeperRow.set_playing` no-ops when its state is unchanged,
        so looping every row per tick is cheap.
        """
        rows = self._keepers_sidebar._rows
        active_id: Optional[str] = None
        for region_data in self._regions_overlay.regions_data():
            if (
                region_data.start_sec <= pos < region_data.end_sec
                and region_data.id in rows
            ):
                active_id = region_data.id
                break
        if active_id is None and self._playback_engine.is_playing:
            # No region under the playhead — keep the played keeper lit.
            active_id = self._currently_playing_keeper_id
        for rid, row in rows.items():
            row.set_playing(rid == active_id)

    def _maybe_autoplay_keeper_on_enter(self, pos: float) -> None:
        """Auto-audition a keeper the instant continuous playback enters it.

        quick-260629 — when the playhead streams into a keeper section (the
        SAME containment test that lights its sidebar row in
        :meth:`_set_playing_row_highlight`) and we are not already playing
        that keeper, fire the EXACT action of clicking the row's ▶ Play
        button: :meth:`_on_keeper_play` with ``"start"``. That reuses the
        established cache→B / no-cache→A logic, the config-driven fades, and
        the per-row highlight — so entering a mastered keeper auditions its
        mastered version from the start, with no new playback plumbing.

        Re-fire guard (the tick runs at 30 Hz): only trigger on an ENTER
        transition — ``region.id != self._currently_playing_keeper_id``. Once
        ``_on_keeper_play`` sets ``_currently_playing_keeper_id`` to this
        keeper, subsequent ticks while the playhead stays inside it are
        no-ops. Runs only while the engine is playing — a paused/stopped
        engine cannot cross a boundary, and this must never start audio on
        its own.

        Uses the true audible source-time ``pos`` (the B-mode translation in
        :meth:`_on_playback_tick` already ran), so containment works in both
        A and B modes.

        Only fires while the A/B toggle is on **A** (plain source playback).
        In B we are already auditioning a keeper's mastered cache (reached
        either by this auto-play or by a deliberate click/▶ into that keeper),
        so re-firing would fight the user's chosen offset — and a B-mode
        bounded cache cannot stream across into another keeper anyway.
        """
        if not self._playback_engine.is_playing:
            return
        if not hasattr(self, "_ab_toggle") or self._ab_toggle.state != "A":
            return
        rows = self._keepers_sidebar._rows
        for region_data in self._regions_overlay.regions_data():
            if (
                region_data.start_sec <= pos < region_data.end_sec
                and region_data.id in rows
            ):
                if region_data.id != self._currently_playing_keeper_id:
                    self._on_keeper_play(region_data.id, "start")
                return

    def _on_keeper_normalize_changed(
        self, region_id: str, enabled: bool
    ) -> None:
        """KeeperRow Normalize toggle flipped → persist to sidecar (gfq).

        quick-260621-gfq — normalize is now the FINAL mastering-chain stage.
        Updates the single source of truth (``mastering['normalize']``) via the
        overlay setter (default 0 dB target), drives the existing
        ``_on_regions_changed`` save path, AND re-renders ONLY this keeper's
        waveform region in place (WYSIWYG preview of the eventual mastered
        output's final step; no viewport move).
        """
        self._regions_overlay.set_normalize(region_id, bool(enabled), 0.0)
        self._on_regions_changed()
        # In-place WYSIWYG re-render of this keeper's span (locked decision #1
        # option 2). Read the live bounds + the persisted target.
        widget = self._regions_overlay.get_region(region_id)
        if widget is not None:
            start_s, end_s = widget.getRegion()
            lo, hi = (float(start_s), float(end_s))
            if lo > hi:
                lo, hi = hi, lo
            _enabled, target_db = self._regions_overlay.get_normalize(region_id)
            self._waveform_view.set_region_normalize(
                lo, hi, bool(enabled), float(target_db)
            )

    def _on_keeper_delete(self, region_id: str) -> None:
        """KeeperRow Delete button click → delete the region.

        Drives :meth:`RegionsOverlay.delete` which removes the region
        from the plot, clears any stale hover, and emits
        ``regions_changed`` — which fires :meth:`_on_regions_changed`,
        which saves the sidecar AND refreshes the Keepers dock.
        """
        self._regions_overlay.delete(region_id)

    # --------------------------------- Phase 7 Plan 07-02 mastering slots
    def _format_time_range(self, start_sec: float, end_sec: float) -> str:
        """Format a HH:MM:SS – HH:MM:SS range (em-dash per UI-SPEC)."""

        def _hhmmss(s: float) -> str:
            total = max(0, int(s))
            return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

        return f"{_hhmmss(start_sec)} – {_hhmmss(end_sec)}"

    def _on_keeper_mastering_requested(self, region_id: str) -> None:
        """KeeperRow Master button click → open the modal MasteringDialog.

        Phase 7 Plan 07-02 Task 4. Looks up the keeper's current
        mastering config from the regions overlay; if mastering is None
        (legacy keeper persisted before Phase 7 / before Plan 07-03's
        snapshot hook lands) we auto-snapshot from the session chain so
        the dialog satisfies its defensive non-None contract. The
        auto-snapshot path is the legacy-migration covered by Plan 07-03
        — owning it here is a Rule 3 blocker fix (the dialog needs a
        non-None dict; without this branch every existing keeper would
        crash on Master button click).
        """
        # Resolve the region to get its start/end + current mastering.
        target: Region | None = None
        for r in self._regions_overlay.regions_data():
            if r.id == region_id:
                target = r
                break
        if target is None:
            return  # defensive — race against overlay teardown

        keeper_mastering = target.mastering
        if keeper_mastering is None:
            # Legacy-migration auto-snapshot (Plan 07-03 contract). The
            # snapshot is persisted immediately so a subsequent Discard
            # changes still leaves the keeper with a valid mastering
            # dict (rather than silently re-snapshotting on every open).
            keeper_mastering = load_session_chain_snapshot()
            self._regions_overlay.set_mastering(region_id, keeper_mastering)
            if self._current_sidecar_path is not None:
                sidecar_cache.save_sidecar(
                    self._current_sidecar_path,
                    self._regions_overlay.regions_data(),
                    self._current_markers,  # quick-260701-jc5 — keep markers
                )

        keeper_range = self._format_time_range(target.start_sec, target.end_sec)
        dlg = MasteringDialog(
            keeper_id=region_id,
            keeper_mastering=keeper_mastering,
            keeper_range=keeper_range,
            parent=self,
        )
        dlg.config_changed.connect(self._on_mastering_config_applied)
        dlg.exec()

    def _on_mastering_config_applied(
        self, keeper_id: str, cfg: dict
    ) -> None:
        """MasteringDialog Apply → persist cfg + update badge + maybe spawn runnable.

        Phase 7 Plan 07-02 Task 4. End-to-end:
            1. Look up the previous mastering state for the keeper.
            2. Update the regions overlay + save sidecar.
            3. Compute the new badge state (none / check / star).
            4. Update the row's badge + status.
            5. If transitioning from no mastering OR config_hash changed
               AND the cache file is stale → spawn a single-keeper
               MasteringRunnable.
        """
        previous_mastering = self._regions_overlay.get_mastering(keeper_id)
        self._regions_overlay.set_mastering(keeper_id, cfg)
        if self._current_sidecar_path is not None:
            sidecar_cache.save_sidecar(
                self._current_sidecar_path,
                self._regions_overlay.regions_data(),
                self._current_markers,  # quick-260701-jc5 — keep markers
            )
            self.regions_changed.emit(str(self._current_sidecar_path))

        # Badge state computation — share the same hash-based rules with
        # the test contract in test_divergence_badge.py.
        new_hash = config_hash(cfg)
        session_hash = config_hash(load_session_chain_snapshot())
        badge_state = "check" if new_hash == session_hash else "star"

        row = self._keepers_sidebar.find_row(keeper_id)
        if row is not None:
            row.set_mastering_badge(badge_state)

        # Phase 7 Plan 07-04 — re-evaluate A/B toggle after the config
        # changes (the new config_hash may match a different cache file).
        if (
            self._selected_keeper_id is not None
            and self._selected_keeper_id == keeper_id
        ):
            self._refresh_ab_toggle_enabled_state()

        # Single-keeper runnable spawn — only when:
        #   (a) previous mastering was None (first Apply), OR
        #   (b) config_hash changed (would create a new cache filename).
        # In either case the cache file at the new path is the freshness
        # signal; if it already exists fresh, skip the runnable.
        #
        # Plan 07-08 Part B-0 — the no-source guard previously checked
        # ``_current_proxy_p`` (the peak-builder's peaks.dat binary),
        # which is unrelated to the audio source. The legacy site is
        # now unified with the Phase A site (see comment at
        # ``_kick_next_master_all`` line ~2471): ``_current_playback_path``
        # is the canonical audio source. We keep ``_current_path`` in
        # the guard too (it tracks the open-file path used for the
        # cache_key) but drop the bogus ``_current_proxy_p`` check.
        if self._current_path is None or self._current_playback_path is None:
            return  # no source open — nothing to render

        # Plan 07-08 Part B — look up the keeper Region from the overlay
        # so the spawn can forward region bounds (start_sec / end_sec)
        # as source-proxy frames. Mirrors the canonical pattern at
        # ``_on_keeper_mastering_requested`` lines ~2020-2026 (which
        # is also a region-lookup-by-id branch). Fail closed if the
        # lookup misses — a race between region deletion and the
        # dialog Apply signal would otherwise spawn a runnable with no
        # bounds. Status-bar message + early return matches the WR-05
        # error-UX pattern elsewhere in this file.
        target: Region | None = None
        for r in self._regions_overlay.regions_data():
            if r.id == keeper_id:
                target = r
                break
        if target is None:
            self.statusBar().showMessage(
                "Mastering aborted: keeper region not found.", 10000
            )
            return

        previous_hash = (
            config_hash(previous_mastering) if previous_mastering else None
        )
        needs_render = previous_mastering is None or previous_hash != new_hash
        if not needs_render:
            return

        src_key = proxy_cache.cache_key(self._current_path)
        dst = mastered_cache_path(
            default_cache_root(), src_key, keeper_id, new_hash
        )
        if is_mastered_cache_fresh(dst):
            if row is not None:
                row.set_mastering_status("Ready", "#7FBFFF")
            return

        # Plan 07-08 Part B — read the source-proxy sample rate so we
        # can convert keeper.start_sec / end_sec to frame indices.
        # sf.info failure is surfaced via the status bar + early
        # return (mirroring the Phase A site's per-keeper Failed
        # handling, adapted to the single-keeper UX).
        try:
            src_sr = int(
                sf.info(str(self._current_playback_path)).samplerate
            )
        except Exception as exc:
            self.statusBar().showMessage(
                f"Mastering failed: source proxy unreadable ({exc}).",
                10000,
            )
            return

        # Cancel any in-flight runnable for this keeper, then spawn anew.
        self._on_keeper_mastering_cancel_requested(keeper_id)
        self._mastering_generation += 1
        gen = self._mastering_generation
        # Plan 07-08 Part B-0 — pass ``_current_playback_path`` (the
        # audio source WAV) NOT ``_current_proxy_p`` (peaks.dat — the
        # peak-builder's binary). Plan 07-06's source-path unification
        # only landed at the Phase A site; this is the matching legacy
        # fix. Without this, MasteringRunnable was reading non-audio
        # bytes via sf.read, producing either silent worker errors or
        # garbage cache content since Plan 07-02 shipped.
        #
        # Plan 07-08 Part B — forward keeper region bounds so the
        # cache file ends up region-bounded instead of full-source.
        runnable = MasteringRunnable(
            self._current_playback_path,
            dst,
            keeper_id,
            cfg,
            start_frame=int(target.start_sec * src_sr),
            end_frame=int(target.end_sec * src_sr),
        )
        runnable.signals.progress.connect(
            lambda pct, kid=keeper_id, g=gen: self._on_mastering_progress(
                pct, kid, g
            )
        )
        runnable.signals.finished.connect(
            lambda path, kid=keeper_id, g=gen: self._on_mastering_finished(
                path, kid, g
            )
        )
        runnable.signals.error.connect(
            lambda msg, kid=keeper_id, g=gen: self._on_mastering_error(
                msg, kid, g
            )
        )
        runnable.signals.cancelled.connect(
            lambda kid=keeper_id, g=gen: self._on_mastering_cancelled(kid, g)
        )
        if row is not None:
            row.set_mastering_status("Mastering 0%", "#9CA3AF")
        self._mastering_runnables[keeper_id] = runnable
        self._dispatch_mastering_runnable(runnable)

    def _dispatch_mastering_runnable(self, runnable: MasteringRunnable) -> None:
        """Start a mastering render — GUI thread for VST3 chains, else the pool.

        quick-260625 — a VST3/JUCE plugin pins its message manager to the
        thread that opened its editor (the GUI thread). Rendering such a chain
        on a ``QThreadPool`` worker thread then DEADLOCKS waiting on that
        manager — the "stuck at 5%" hang. So a keeper whose chain has the VST3
        stage enabled renders synchronously ON THE GUI THREAD instead.

        The GUI thread can't repaint while ``run()`` blocks, so we defer via
        ``QTimer.singleShot(0, ...)``: the dialog closes / progress paints
        first, and the master-all queue (whose ``finished`` slot kicks the next
        render) serialises through the event loop instead of recursing. The
        per-keeper clip is short, so the brief freeze is acceptable — and
        consistent with the modal plugin editor. All non-VST3 keepers keep the
        fully responsive worker path.
        """
        cfg = getattr(runnable, "_mastering_cfg", {}) or {}
        vst3_enabled = bool(cfg.get("vst3", {}).get("enabled", False))
        if vst3_enabled:
            QTimer.singleShot(0, runnable.run)
        else:
            QThreadPool.globalInstance().start(runnable)

    def _on_mastering_progress(self, pct: int, keeper_id: str, gen: int) -> None:
        """Update row status with the current MasteringRunnable percentage."""
        if gen != self._mastering_generation:
            return  # stale signal
        row = self._keepers_sidebar.find_row(keeper_id)
        if row is not None:
            row.set_mastering_status(f"Mastering {int(pct)}%", "#9CA3AF")

    def _on_mastering_finished(self, path: str, keeper_id: str, gen: int) -> None:
        """MasteringRunnable terminal-success signal → "Ready" + test seam."""
        if gen != self._mastering_generation:
            return
        row = self._keepers_sidebar.find_row(keeper_id)
        if row is not None:
            row.set_mastering_status("Ready", "#7FBFFF")
        self._mastering_runnables.pop(keeper_id, None)
        self.mastering_complete.emit(keeper_id)

    def _on_mastering_error(self, msg: str, keeper_id: str, gen: int) -> None:
        """MasteringRunnable error → "Failed" + carry msg in tooltip.

        WR-05 (Phase 7 review) — when the error is a config-shape problem
        (Matchering reference path outside library, missing reference
        file, regex-rejected keeper_id/source_cache_key/config_hash),
        also surface a user-readable summary in the status bar so the
        cause is discoverable without hovering the gear-button tooltip.
        Heuristic dispatch on message substring keeps the audio-tier
        free of tagged-exception coupling.
        """
        if gen != self._mastering_generation:
            return
        row = self._keepers_sidebar.find_row(keeper_id)
        if row is not None:
            row.set_mastering_status("Failed", "#E5484D")
            row._master.setToolTip(f"Mastering failed: {msg}")
        self._mastering_runnables.pop(keeper_id, None)
        # WR-05 — status-bar surfacing for config-shape errors.
        lower = msg.lower() if msg else ""
        summary: str | None = None
        if "outside the configured" in lower or "reference library" in lower:
            summary = (
                "Mastering failed: reference track must live inside the "
                "configured reference library, or be re-picked via Browse."
            )
        elif "does not point to a file" in lower:
            summary = (
                "Mastering failed: the Matchering reference file is missing."
            )
        elif "invalid keeper_id" in lower or "invalid source_cache_key" in lower \
                or "invalid config_hash" in lower:
            summary = (
                "Mastering failed: invalid cache key. Try re-creating the Keeper."
            )
        if summary is not None:
            self.statusBar().showMessage(summary, 10000)

    def _on_mastering_cancelled(self, keeper_id: str, gen: int) -> None:
        """MasteringRunnable cancellation → clear row status."""
        if gen != self._mastering_generation:
            return
        row = self._keepers_sidebar.find_row(keeper_id)
        if row is not None:
            row.set_mastering_status("", "#9CA3AF")
        self._mastering_runnables.pop(keeper_id, None)

    def _on_keeper_mastering_cancel_requested(self, keeper_id: str) -> None:
        """Right-click "Cancel mastering" → cancel any in-flight runnable.

        Generation bump ensures any in-flight signal that emits after
        the cancel call lands as stale and is ignored by the slot
        generation-checks.

        CR-03 (Phase 7 review) — the generation bump exists precisely
        BECAUSE the worker's cancelled() may take up to ~30 s (matchering
        mid-call). That means the cancelled-slot pop() will be reached
        with a stale generation and return early — so the runnable
        would otherwise leak in the dict. Pop synchronously here, and
        clear the row's UI side-effects too, since the cancelled slot
        will be dropped as stale.
        """
        runnable = self._mastering_runnables.pop(keeper_id, None)
        if runnable is not None:
            runnable.cancel()
        self._mastering_generation += 1
        row = self._keepers_sidebar.find_row(keeper_id)
        if row is not None:
            try:
                row.set_mastering_status("", "#9CA3AF")
            except RuntimeError:
                pass

    # ----------------------- Phase 7 Plan 07-06 Task 2 — Master & Export All

    # ------------------------------------------------------------------
    # Phase 8 Plan 08-04 — per-keeper Share-to-YouTube orchestration.
    # ------------------------------------------------------------------

    def _on_share_requested(self, region_id: str) -> None:
        """KeeperRow Share button click → open the modal UploadDialog.

        Implements the R-05 + D-02 audio-source routing (revision iter
        1 B1) — owns the mastered-cache-vs-source-proxy decision in
        ONE place so the KeeperRow Share button can stay always-enabled
        per B1.

        Flow:
            1. Resolve the keeper Region from the overlay.
            2. Gate on OAuth — if not connected, surface the Preferences
               dialog and return (the user has to Connect YouTube
               before any upload can succeed).
            3. Determine ``audio_source_path``:
                 * Mastered cache fresh → feed cache path directly to
                   ffmpeg (R-05 bypass — no export_region call).
                 * Mastered cache NOT fresh → call ``export_region(*,
                   source_path=self._current_playback_path, ...)`` to
                   materialise the unmastered audio (with Phase 3
                   fades + sample-rate handling) into a tmp WAV.
                   The tmp WAV is registered for cleanup after upload
                   success / failure / cancel.
            4. Fetch initial Picsum thumbnail bytes.
            5. Construct UploadDialog with default title (poem_generator)
               + default privacy (last-used QSettings value, D-21).
            6. Connect the 4 dialog signals to MainWindow slots; stash
               the per-keeper upload state; ``dlg.exec()``.

        CRITICAL (Phase 7 LEARNINGS Surprise carry-forward): the source
        for the unmastered-fallback path is ``self._current_playback_path``
        (the audio proxy WAV) — NEVER the peaks binary attribute (which
        holds the peak-builder's .dat output). Passing the peaks binary
        to ``export_region`` would silently render garbage; this bug bit
        Phase 7 from Plan 07-02 through 07-08 and the integration test
        ``test_share_button_on_unmastered_keeper_uses_source_proxy_fallback``
        regression-pins it.
        """
        # 1. Resolve the region from the overlay.
        target: Region | None = None
        for r in self._regions_overlay.regions_data():
            if r.id == region_id:
                target = r
                break
        if target is None:
            return  # defensive — race against overlay teardown

        # 2. OAuth gate — must be connected before showing the dialog.
        try:
            creds = _yt_oauth.load_or_refresh()
        except Exception:
            creds = None
        if creds is None:
            self.statusBar().showMessage(
                "Connect YouTube in Preferences before sharing.", 5000
            )
            # Open the SettingsDialog so the user has a direct path to
            # Connect. (Defensive — if a future iteration adds an
            # auto-prompt this is the natural attachment point.)
            self._on_open_preferences()
            return

        # 3. Determine audio source path (R-05 + D-02 routing).
        if self._current_path is None or self._current_playback_path is None:
            self.statusBar().showMessage(
                "Open a file first to share its keepers.", 5000
            )
            return

        audio_source_path: Path
        tmp_audio_to_cleanup: Path | None = None
        try:
            src_key = proxy_cache.cache_key(self._current_path)
        except Exception as exc:
            self.statusBar().showMessage(
                f"Share unavailable: cache key error ({exc}).", 5000
            )
            return

        cache_p: Path | None = None
        if target.mastering is not None:
            try:
                chash = config_hash(target.mastering)
                cache_p = mastered_cache_path(
                    default_cache_root(), src_key, region_id, chash
                )
            except Exception:
                cache_p = None

        if cache_p is not None and is_mastered_cache_fresh(cache_p):
            # R-05 bypass — the mastered cache WAV is already the
            # materialised audio. No tmp WAV; nothing to clean up.
            audio_source_path = cache_p
        else:
            # D-02 fallback — materialise the unmastered audio via
            # export_region. The Phase 3 fade-in/out + sample-rate
            # handling apply transparently.
            #
            # Read the source-proxy sample rate so we can convert
            # start_sec / end_sec to frame indices. sf.info failure
            # surfaces as a status-bar message + early return.
            try:
                src_sr = int(
                    sf.info(str(self._current_playback_path)).samplerate
                )
            except Exception as exc:
                self.statusBar().showMessage(
                    f"Share failed: source proxy unreadable ({exc}).", 5000
                )
                return
            start_f = int(target.start_sec * src_sr)
            end_f = int(target.end_sec * src_sr)
            region_dur = max(0.0, target.end_sec - target.start_sec)
            # quick-260626-o9y — config-driven fade (was forced 2.0 s). Read
            # the keeper's fade enabled flag + duration via fade_params;
            # disabled → 0.0 (no fade), else clamp to half the region.
            fade_enabled, fade_dur = fade_params(target.mastering)
            fade_sec = min(fade_dur, region_dur / 2.0) if fade_enabled else 0.0
            fade_frames = max(0, int(fade_sec * src_sr))
            tmp_fd, tmp_name = tempfile.mkstemp(
                suffix=".wav", prefix=f"jamextract-share-{region_id[:8]}-"
            )
            os.close(tmp_fd)
            tmp_audio_to_cleanup = Path(tmp_name)
            try:
                # CRITICAL — source_path is _current_playback_path (the
                # audio proxy WAV). Passing the peaks binary attribute
                # would silently render garbage; Phase 7 LEARNINGS
                # Surprise carry-forward, regression-pinned by
                # test_share_button_on_unmastered_keeper_uses_source_proxy_fallback.
                export_region(
                    self._current_playback_path,  # proxy_path (unused
                    # when source_path is set, but the positional kwarg
                    # is still required by the signature).
                    tmp_audio_to_cleanup,
                    start_f,
                    end_f,
                    fade_frames,
                    "wav",
                    src_sr,
                    source_path=self._current_playback_path,
                )
            except Exception as exc:
                # Cleanup tmp before bailing.
                try:
                    tmp_audio_to_cleanup.unlink(missing_ok=True)
                except OSError:
                    pass
                self.statusBar().showMessage(
                    f"Share failed: audio export error ({exc}).", 5000
                )
                return
            audio_source_path = tmp_audio_to_cleanup

        # 4. Fetch initial Picsum thumbnail bytes. fetch_thumbnail
        # never raises — falls back to the Pillow deterministic-color
        # JPEG on network failure.
        initial_nonce = 0
        thumbnail_bytes = _thumbnail_provider.fetch_thumbnail(
            seed=region_id, nonce=initial_nonce
        )

        # 5. Build dialog kwargs from QSettings + poem_generator.
        s = QSettings("Marmelade", "Marmelade")
        initial_privacy = str(
            s.value("youtube/privacy_default", "private")
        )
        initial_title = _poem_generator.generate()
        initial_description = ""
        keeper_range = self._format_time_range(target.start_sec, target.end_sec)

        dlg = UploadDialog(
            keeper_id=region_id,
            keeper_range=keeper_range,
            initial_title=initial_title,
            initial_description=initial_description,
            initial_privacy=initial_privacy,
            initial_thumbnail_bytes=thumbnail_bytes,
            parent=self,
        )

        # Stash per-keeper upload state BEFORE wiring signals so the
        # slots can look up the dialog + audio source on every
        # invocation.
        self._upload_state[region_id] = {
            "audio_source_path": audio_source_path,
            "tmp_audio_to_cleanup": tmp_audio_to_cleanup,
            "dialog": dlg,
            "thumbnail_bytes": thumbnail_bytes,
            "nonce": initial_nonce,
            "credentials": creds,
        }

        # 6. Wire dialog signals (default-arg closure binding).
        dlg.upload_requested.connect(
            lambda title, desc, priv, lic, jpeg, rid=region_id, d=dlg:
                self._on_upload_initiated(
                    title, desc, priv, lic, jpeg, rid, d
                )
        )
        dlg.cancel_requested.connect(
            lambda rid=region_id: self._on_upload_cancel_requested(rid)
        )
        dlg.retry_requested.connect(
            lambda rid=region_id, d=dlg: self._on_upload_retry_requested(rid, d)
        )
        dlg.refresh_thumbnail_requested.connect(
            lambda rid=region_id, d=dlg: self._on_refresh_thumbnail_requested(rid, d)
        )

        try:
            dlg.exec()
        finally:
            # Clean up tmp WAV on dialog close (success / failure /
            # cancel all funnel through here). The terminal slots also
            # call _cleanup_upload_state — this is the defense-in-depth
            # branch for the "user rejects dialog before clicking
            # Upload" path.
            self._cleanup_upload_state(region_id)

    def _on_upload_initiated(
        self,
        title: str,
        description: str,
        privacy: str,
        license: str,
        thumbnail_bytes: bytes,
        region_id: str,
        dlg: UploadDialog,
    ) -> None:
        """UploadDialog Upload click → spawn YouTubeUploadRunnable + set Phase B."""
        state = self._upload_state.get(region_id)
        if state is None:
            return  # defensive — dialog raced ahead of state stash

        # Persist privacy choice to QSettings per D-21.
        s = QSettings("Marmelade", "Marmelade")
        s.setValue("youtube/privacy_default", privacy)
        s.sync()

        # Update the cached thumbnail bytes in state so a Retry uses
        # the same image (D-25: retry re-uses audio + thumbnail).
        state["thumbnail_bytes"] = thumbnail_bytes

        # Persist a tmp JPEG so the upload runnable can read it back.
        tmp_img_p = Path(tempfile.mkstemp(
            suffix=".jpg", prefix=f"jamextract-share-{region_id[:8]}-"
        )[1])
        tmp_img_p.write_bytes(thumbnail_bytes)
        state["tmp_image_to_cleanup"] = tmp_img_p

        # Cancel any in-flight runnable for this keeper, then spawn anew.
        self._on_upload_cancel_requested(region_id)
        self._upload_generation += 1
        gen = self._upload_generation

        runnable = YouTubeUploadRunnable(
            audio_path=state["audio_source_path"],
            image_path=tmp_img_p,
            snippet={"title": title, "description": description, "tags": []},
            status={"privacyStatus": privacy, "license": license},
            credentials=state["credentials"],
            keeper_id=region_id,
            tmp_dir=Path(tempfile.gettempdir()) / "marmelade-uploads",
        )
        # Default-arg closure binding (Phase 1 LEARNINGS) — capture
        # keeper_id + generation token by value at connect time.
        runnable.signals.progress.connect(
            lambda pct, kid=region_id, g=gen:
                self._on_youtube_upload_progress(pct, kid, g)
        )
        runnable.signals.finished.connect(
            lambda video_id, kid=region_id, g=gen:
                self._on_youtube_upload_finished(video_id, kid, g)
        )
        runnable.signals.error.connect(
            lambda msg, kid=region_id, g=gen:
                self._on_youtube_upload_error(msg, kid, g)
        )
        runnable.signals.cancelled.connect(
            lambda kid=region_id, g=gen:
                self._on_youtube_upload_cancelled(kid, g)
        )
        self._upload_runnables[region_id] = runnable
        QThreadPool.globalInstance().start(runnable)

        # Swap the dialog to Phase B.
        dlg.set_phase_b()

    def _on_youtube_upload_progress(
        self, pct: int, keeper_id: str, gen: int
    ) -> None:
        """Forward worker progress to the dialog's progress bar."""
        if gen != self._upload_generation:
            return  # stale signal
        state = self._upload_state.get(keeper_id)
        if state is None:
            return
        dlg = state.get("dialog")
        if dlg is None:
            return
        # ETA not yet wired (Plan 08-06 follow-up) — pass None so the
        # label renders "ETA: —".
        try:
            dlg.set_progress(int(pct), None)
        except RuntimeError:
            # Dialog already destroyed — drop the update.
            pass

    def _on_youtube_upload_finished(
        self, video_id: str, keeper_id: str, gen: int
    ) -> None:
        """Persist video_id to the sidecar + close the dialog."""
        if gen != self._upload_generation:
            return
        state = self._upload_state.get(keeper_id)
        if state is None:
            return
        self._upload_runnables.pop(keeper_id, None)
        # Persist Region.youtube_video_id via the overlay's setter +
        # the existing _on_regions_changed save pathway (Plan 08-01
        # ships the additive sidecar field per D-30; Plan 08-06 Task
        # 1 wires the overlay carrier so the value survives a full
        # app-restart cycle — regression-pinned by
        # test_sidecar_youtube_video_id_roundtrip_e2e.py).
        try:
            self._regions_overlay.set_youtube_video_id(
                keeper_id, str(video_id)
            )
            # Trigger a save so the new id lands on disk immediately
            # (the overlay's set_youtube_video_id is a mutator-only
            # method; it does NOT emit regions_changed by itself
            # because callers may batch multiple mutations).
            self._on_regions_changed()
        except Exception:
            pass
        # Status bar — 8 second display per the plan's success copy.
        self.statusBar().showMessage(
            f"Uploaded to YouTube: {video_id}", 8000
        )
        dlg = state.get("dialog")
        if dlg is not None:
            try:
                dlg.accept()
            except RuntimeError:
                pass
        self._cleanup_upload_state(keeper_id)

    def _on_youtube_upload_error(
        self, msg: str, keeper_id: str, gen: int
    ) -> None:
        """Surface the runnable's error message in the dialog footer."""
        if gen != self._upload_generation:
            return
        state = self._upload_state.get(keeper_id)
        if state is None:
            return
        self._upload_runnables.pop(keeper_id, None)
        dlg = state.get("dialog")
        if dlg is not None:
            try:
                dlg.show_error(msg, retryable=True)
            except RuntimeError:
                pass
        # NOTE: do NOT cleanup upload state here — the user may click
        # Retry which needs the same audio source + image. Cleanup
        # happens on dialog close (success / cancel / dismiss).

    def _on_youtube_upload_cancelled(self, keeper_id: str, gen: int) -> None:
        """Worker confirmed cancel — close dialog, no sidecar write."""
        if gen != self._upload_generation:
            return
        state = self._upload_state.get(keeper_id)
        if state is None:
            return
        self._upload_runnables.pop(keeper_id, None)
        dlg = state.get("dialog")
        if dlg is not None:
            try:
                dlg.reject()
            except RuntimeError:
                pass
        self._cleanup_upload_state(keeper_id)

    def _on_upload_cancel_requested(self, region_id: str) -> None:
        """UploadDialog Cancel button → cancel the in-flight runnable.

        Idempotent. The runnable's ``cancelled`` signal eventually fires
        and ``_on_youtube_upload_cancelled`` rejects the dialog.
        """
        runnable = self._upload_runnables.pop(region_id, None)
        if runnable is not None:
            runnable.cancel()
        # Bump generation so any stale signals from the cancelled
        # runnable are dropped.
        self._upload_generation += 1

    def _on_upload_retry_requested(
        self, region_id: str, dlg: UploadDialog
    ) -> None:
        """UploadDialog Retry button → re-spawn the runnable, same audio + image."""
        state = self._upload_state.get(region_id)
        if state is None:
            return
        # Re-spawn with the same audio source + thumbnail. Read back
        # the privacy / title / description from the dialog's current
        # widgets (the user may have edited them in the error footer).
        title = dlg._title_edit.text()
        description = dlg._description_edit.text()
        privacy = dlg._privacy_combo.currentData() or "private"
        license_val = dlg._license_combo.currentData() or "creativeCommon"
        self._on_upload_initiated(
            str(title), str(description), str(privacy), str(license_val),
            state["thumbnail_bytes"], region_id, dlg,
        )

    def _on_refresh_thumbnail_requested(
        self, region_id: str, dlg: UploadDialog
    ) -> None:
        """UploadDialog Refresh button → re-fetch with incremented nonce."""
        state = self._upload_state.get(region_id)
        if state is None:
            return
        state["nonce"] = int(state.get("nonce", 0)) + 1
        new_bytes = _thumbnail_provider.fetch_thumbnail(
            seed=region_id, nonce=state["nonce"]
        )
        state["thumbnail_bytes"] = new_bytes
        try:
            dlg.update_thumbnail(new_bytes)
        except RuntimeError:
            pass

    def _cleanup_upload_state(self, region_id: str) -> None:
        """Remove tmp files + drop the state dict for ``region_id``."""
        state = self._upload_state.pop(region_id, None)
        if state is None:
            return
        for key in ("tmp_audio_to_cleanup", "tmp_image_to_cleanup"):
            p = state.get(key)
            if p is not None:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Phase 8 Plan 08-05 — Bundle Share orchestration (D-01 + D-02 + D-19).
    # ------------------------------------------------------------------

    def _is_keeper_mastered_cache_fresh(self, region_id: str) -> bool:
        """Probe — does ``region_id`` have a fresh mastered cache WAV on disk?

        Installed on the KeepersSidebar via
        :meth:`KeepersSidebar.set_mastered_cache_fresh_probe` so the
        sidebar can enable/disable the bundle button per D-02 without
        importing the paths/config_hash machinery itself.

        Returns False on any of: no source file open, no keeper with
        that id, keeper has no mastering chain, cache_key/config_hash
        compute failure, or the cache WAV is missing/empty.
        """
        if self._current_path is None:
            return False
        target: Region | None = None
        for r in self._regions_overlay.regions_data():
            if r.id == region_id:
                target = r
                break
        if target is None or target.mastering is None:
            return False
        try:
            src_key = proxy_cache.cache_key(self._current_path)
            chash = config_hash(target.mastering)
            cache_p = mastered_cache_path(
                default_cache_root(), src_key, region_id, chash
            )
        except Exception:
            return False
        return is_mastered_cache_fresh(cache_p)

    def _on_bundle_share_requested(self) -> None:
        """Bundle Share button click → open the modal BundleDialog.

        Flow (D-01 + D-02 + D-19):
            1. Collect the bundle order via
               ``self._keepers_sidebar.current_order()`` — this is the
               user-arranged order (D-05 drag-handle reorder).
            2. Gate on D-02: every keeper MUST have a fresh mastered
               cache. The sidebar button is already disabled when this
               isn't true, but the slot defends against a race where
               the user clicks just as a cache invalidates.
            3. OAuth gate — surface Preferences if not connected.
            4. Build the BundleDialog with:
                 - keepers list (region_id + display label) in current order
                 - initial_spacer_sec from QSettings (explicit float coerce
                   per RESEARCH Pitfall 4)
                 - initial_title from poem_generator
                 - initial_description = "<YYYY-MM-DD> bundle …"
                 - initial_privacy from QSettings (D-21 carry-forward)
                 - initial_thumbnail_bytes from a one-shot Picsum fetch
            5. Wire the 4 dialog signals, exec().
        """
        if self._current_path is None or self._current_playback_path is None:
            self.statusBar().showMessage(
                "Open a file first to share its keepers.", 5000
            )
            return
        ordered_ids = self._keepers_sidebar.current_order()
        if not ordered_ids:
            return
        # D-02 gate — defend against race.
        for rid in ordered_ids:
            if not self._is_keeper_mastered_cache_fresh(rid):
                self.statusBar().showMessage(
                    "Master all keepers before sharing the bundle.", 5000
                )
                return
        # OAuth gate.
        try:
            creds = _yt_oauth.load_or_refresh()
        except Exception:
            creds = None
        if creds is None:
            self.statusBar().showMessage(
                "Connect YouTube in Preferences before sharing.", 5000
            )
            self._on_open_preferences()
            return

        # Build the (region_id, display_label) list. The label uses the
        # same HH:MM:SS – HH:MM:SS format the per-keeper Share dialog
        # uses (mirrors _format_time_range).
        keepers_for_dialog: list[tuple[str, str]] = []
        keeper_paths: list[Path] = []
        for rid in ordered_ids:
            target: Region | None = None
            for r in self._regions_overlay.regions_data():
                if r.id == rid:
                    target = r
                    break
            if target is None:
                continue
            label = self._format_time_range(target.start_sec, target.end_sec)
            keepers_for_dialog.append((rid, label))
            try:
                src_key = proxy_cache.cache_key(self._current_path)
                chash = config_hash(target.mastering)
                cache_p = mastered_cache_path(
                    default_cache_root(), src_key, rid, chash
                )
                keeper_paths.append(cache_p)
            except Exception:
                self.statusBar().showMessage(
                    f"Bundle failed: cannot resolve cache for keeper.", 5000
                )
                return

        # QSettings reads — explicit float() coerce per RESEARCH Pitfall 4.
        s = QSettings("Marmelade", "Marmelade")
        try:
            initial_spacer_sec = float(s.value("youtube/bundle_spacer_sec", 2.0))
        except (TypeError, ValueError):
            initial_spacer_sec = 2.0
        initial_privacy = str(s.value("youtube/privacy_default", "private"))
        initial_title = _poem_generator.generate()
        initial_description = ""
        # Bundle seed for thumbnail — sha1 of the ordered ids so refreshes
        # are deterministic.
        import hashlib as _hashlib
        bundle_seed = _hashlib.sha1(
            ("bundle:" + ",".join(ordered_ids)).encode("utf-8")
        ).hexdigest()[:16]
        thumbnail_bytes = _thumbnail_provider.fetch_thumbnail(
            seed=bundle_seed, nonce=0
        )

        dlg = BundleDialog(
            keepers=keepers_for_dialog,
            initial_title=initial_title,
            initial_description=initial_description,
            initial_privacy=initial_privacy,
            initial_thumbnail_bytes=thumbnail_bytes,
            initial_spacer_sec=initial_spacer_sec,
            parent=self,
        )
        # Stash bundle state for the slot lifecycle. Bundle uses a
        # parallel dict from per-keeper upload_state so signals don't
        # cross-pollinate.
        self._bundle_state = {
            "dialog": dlg,
            "keeper_paths": keeper_paths,
            "ordered_ids": ordered_ids,
            "thumbnail_bytes": thumbnail_bytes,
            "nonce": 0,
            "bundle_seed": bundle_seed,
            "credentials": creds,
            "runnable": None,
            "tmp_mp4_to_cleanup": None,
            "tmp_image_to_cleanup": None,
            "tmp_mp3_to_cleanup": None,
            "mp3_save_path": None,
        }
        # Wire dialog signals.
        dlg.export_mp3_only_requested.connect(
            lambda _sp, sec, ids, d=dlg:
                self._on_bundle_export_mp3_only(_sp, sec, ids, d)
        )
        dlg.export_and_upload_requested.connect(
            lambda _sp, sec, ids, ti, de, pr, lic, jp, d=dlg:
                self._on_bundle_export_and_upload(
                    _sp, sec, ids, ti, de, pr, lic, jp, d
                )
        )
        dlg.cancel_requested.connect(self._on_bundle_cancel_requested)
        dlg.refresh_thumbnail_requested.connect(
            lambda d=dlg: self._on_bundle_refresh_thumbnail(d)
        )
        try:
            dlg.exec()
        finally:
            self._cleanup_bundle_state()

    def _on_bundle_export_mp3_only(
        self,
        _save_path_placeholder: str,
        spacer_sec: float,
        ordered_ids: list,
        dlg: BundleDialog,
    ) -> None:
        """BundleDialog Export MP3 only → file picker + bundle_builder.build_bundle."""
        state = self._bundle_state
        if state is None:
            return
        # Persist the spacer choice for next time (D-04 default carry).
        s = QSettings("Marmelade", "Marmelade")
        s.setValue("youtube/bundle_spacer_sec", float(spacer_sec))
        s.sync()
        # D-06 — user picks the save path each time; no QSettings persistence.
        default_name = f"bundle_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.mp3"
        save_path_str, _filter = QFileDialog.getSaveFileName(
            self,
            "Save bundle MP3 as…",
            str(Path.home() / default_name),
            "MP3 audio (*.mp3)",
        )
        if not save_path_str:
            return
        save_path = Path(save_path_str)
        state["mp3_save_path"] = save_path
        # Swap dialog to Phase B for progress.
        try:
            dlg.set_phase_b()
        except RuntimeError:
            pass
        # Run the build synchronously on the GUI thread for v1 — the
        # bundle is small (mastered caches are typically <60s per
        # keeper) and the user is staring at the dialog. A QRunnable
        # wrapper is a Plan 08-06 follow-up if user feedback demands
        # responsive progress on giant bundles.
        # quick-260615-f77: mastered-cache WAVs are now produced at the
        # source rate (48000 for 48 kHz sources), and build_bundle's
        # fail-fast observed_sr != sample_rate check would raise on the
        # old 44100 literal. The keepers all share one source, so the
        # engine's loaded rate is the correct expected rate.
        bundle_sr = self._playback_engine.sample_rate
        bundle_sr = bundle_sr if bundle_sr > 0 else 48000
        try:
            build_bundle(
                state["keeper_paths"],
                spacer_sec,
                save_path,
                sample_rate=bundle_sr,
                progress_cb=lambda pct: self._on_bundle_build_progress(pct, dlg),
            )
        except Exception as exc:
            try:
                dlg.show_error(f"Bundle build failed: {exc}", retryable=False)
            except RuntimeError:
                pass
            return
        self.statusBar().showMessage(
            f"Bundle exported to {save_path}", 8000
        )
        try:
            dlg.accept()
        except RuntimeError:
            pass

    def _on_bundle_export_and_upload(
        self,
        _save_path_placeholder: str,
        spacer_sec: float,
        ordered_ids: list,
        title: str,
        description: str,
        privacy: str,
        license: str,
        thumbnail_bytes: bytes,
        dlg: BundleDialog,
    ) -> None:
        """BundleDialog "Upload to YouTube" → temp bundle MP3 → ffmpeg MP4 → YouTube upload.

        User-direction simplification: the local-save side-effect was
        moved to the dedicated "Export MP3" button. This handler now
        builds the bundle MP3 to a temp file, uploads, and cleans it up
        on dialog teardown via :meth:`_cleanup_bundle_state` →
        ``tmp_mp3_to_cleanup``.
        """
        state = self._bundle_state
        if state is None:
            return
        # Persist spacer + privacy preferences.
        s = QSettings("Marmelade", "Marmelade")
        s.setValue("youtube/bundle_spacer_sec", float(spacer_sec))
        s.setValue("youtube/privacy_default", privacy)
        s.sync()
        # Build the bundle MP3 to a temp file — no save dialog, no
        # local copy persists once the upload (success or failure)
        # finishes and _cleanup_bundle_state runs.
        tmp_mp3_fd, tmp_mp3_str = tempfile.mkstemp(
            suffix=".mp3", prefix="jamextract-bundle-"
        )
        try:
            os.close(tmp_mp3_fd)
        except OSError:
            pass
        mp3_path = Path(tmp_mp3_str)
        state["tmp_mp3_to_cleanup"] = mp3_path
        state["mp3_save_path"] = mp3_path
        state["thumbnail_bytes"] = thumbnail_bytes

        try:
            dlg.set_phase_b()
        except RuntimeError:
            pass

        # Step 1 — build the bundle MP3 (synchronous for v1).
        # quick-260615-f77: pass the real engine rate (48000 for 48 kHz
        # sources) so build_bundle's observed_sr check does not raise.
        bundle_sr = self._playback_engine.sample_rate
        bundle_sr = bundle_sr if bundle_sr > 0 else 48000
        try:
            build_bundle(
                state["keeper_paths"],
                spacer_sec,
                mp3_path,
                sample_rate=bundle_sr,
                progress_cb=lambda pct: self._on_bundle_build_progress(pct, dlg),
            )
        except Exception as exc:
            try:
                dlg.show_error(f"Bundle build failed: {exc}", retryable=False)
            except RuntimeError:
                pass
            return

        # Step 2 — write the thumbnail JPEG to a tmp file (the upload
        # runnable will read it back for ffmpeg + thumbnails().set).
        tmp_img_p = Path(tempfile.mkstemp(
            suffix=".jpg", prefix="jamextract-bundle-"
        )[1])
        tmp_img_p.write_bytes(thumbnail_bytes)
        state["tmp_image_to_cleanup"] = tmp_img_p

        # Step 3 — spawn the YouTubeUploadRunnable (REUSED from Plan
        # 08-04). The runnable's run() also builds the MP4 via
        # video_builder.build_video as its synchronous prelude, so we
        # don't need to invoke build_video separately here.
        self._upload_generation += 1
        gen = self._upload_generation
        runnable = YouTubeUploadRunnable(
            audio_path=mp3_path,
            image_path=tmp_img_p,
            snippet={"title": title, "description": description, "tags": []},
            status={"privacyStatus": privacy, "license": license},
            credentials=state["credentials"],
            keeper_id="bundle",
            tmp_dir=Path(tempfile.gettempdir()) / "marmelade-uploads",
            # The bundle MP3 already ends with the last keeper's
            # fade-out (applied in bundle_builder), so the default
            # 1.5 s silent video tail would just feel like dead air —
            # ask video_builder to skip the pad for bundles.
            tail_pad_sec=0.0,
        )
        state["runnable"] = runnable
        runnable.signals.progress.connect(
            lambda pct, g=gen: self._on_bundle_upload_progress(pct, g, dlg)
        )
        runnable.signals.finished.connect(
            lambda video_id, g=gen: self._on_bundle_upload_finished(
                video_id, g, dlg
            )
        )
        runnable.signals.error.connect(
            lambda msg, g=gen: self._on_bundle_upload_error(msg, g, dlg)
        )
        runnable.signals.cancelled.connect(
            lambda g=gen: self._on_bundle_upload_cancelled(g, dlg)
        )
        QThreadPool.globalInstance().start(runnable)

    def _on_bundle_build_progress(self, pct: int, dlg: BundleDialog) -> None:
        """Forward bundle-build progress to the dialog (Phase B 0..50)."""
        # Build progress occupies the lower half of the bar; upload
        # progress will occupy the upper half. This keeps the user's
        # perception of "progress moving" honest for the two-step flow.
        try:
            dlg.set_progress(int(pct // 2), None)
        except RuntimeError:
            pass

    def _on_bundle_upload_progress(
        self, pct: int, gen: int, dlg: BundleDialog
    ) -> None:
        """Forward upload progress to the dialog (Phase B 50..100)."""
        if gen != self._upload_generation:
            return
        try:
            dlg.set_progress(50 + int(pct // 2), None)
        except RuntimeError:
            pass

    def _on_bundle_upload_finished(
        self, video_id: str, gen: int, dlg: BundleDialog
    ) -> None:
        """Bundle upload finished — close dialog + status-bar message.

        D-30 + W7 known-limit: bundle video_id is NOT persisted to the
        sidecar (only per-keeper video_ids land via Plan 08-04's
        sidecar additive field). The status-bar message is the user's
        ephemeral feedback.
        """
        if gen != self._upload_generation:
            return
        self.statusBar().showMessage(
            f"Bundle uploaded to YouTube: {video_id}", 10000
        )
        try:
            dlg.accept()
        except RuntimeError:
            pass

    def _on_bundle_upload_error(
        self, msg: str, gen: int, dlg: BundleDialog
    ) -> None:
        """Bundle upload error — surface in dialog footer."""
        if gen != self._upload_generation:
            return
        try:
            dlg.show_error(msg, retryable=True)
        except RuntimeError:
            pass

    def _on_bundle_upload_cancelled(self, gen: int, dlg: BundleDialog) -> None:
        """Bundle upload cancelled by user — close the dialog cleanly."""
        if gen != self._upload_generation:
            return
        try:
            dlg.reject()
        except RuntimeError:
            pass

    def _on_bundle_cancel_requested(self) -> None:
        """BundleDialog Phase B Cancel → cancel the in-flight runnable.

        Idempotent. The runnable's ``cancelled`` signal eventually
        fires and ``_on_bundle_upload_cancelled`` rejects the dialog.
        """
        state = self._bundle_state
        if state is None:
            return
        runnable = state.get("runnable")
        if runnable is not None:
            try:
                runnable.cancel()
            except Exception:
                pass
        # Bump generation so stale signals are dropped.
        self._upload_generation += 1

    def _on_bundle_refresh_thumbnail(self, dlg: BundleDialog) -> None:
        """BundleDialog Refresh thumbnail → re-fetch with incremented nonce."""
        state = self._bundle_state
        if state is None:
            return
        state["nonce"] = int(state.get("nonce", 0)) + 1
        new_bytes = _thumbnail_provider.fetch_thumbnail(
            seed=state.get("bundle_seed", "bundle"), nonce=state["nonce"]
        )
        state["thumbnail_bytes"] = new_bytes
        try:
            dlg.update_thumbnail(new_bytes)
        except RuntimeError:
            pass

    def _cleanup_bundle_state(self) -> None:
        """Remove tmp files + drop the bundle state dict."""
        state = self._bundle_state
        if state is None:
            return
        p = state.get("tmp_image_to_cleanup")
        if p is not None:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        p = state.get("tmp_mp4_to_cleanup")
        if p is not None:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        p = state.get("tmp_mp3_to_cleanup")
        if p is not None:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        self._bundle_state = None

    # ------------------------------------------------------------------
    # Phase 8 Plan 08-02 — View → Preferences slot + YouTube connect/disconnect.
    # ------------------------------------------------------------------

    def _on_about(self) -> None:  # quick-260626-pbl
        """Help → About Marmelade → open the modal AboutDialog.

        Lazy-imports AboutDialog inside the slot (mirrors the Preferences
        pattern) to avoid any import-cycle / startup cost.
        """
        from marmelade.ui.about_dialog import AboutDialog

        AboutDialog(self).exec()

    def _on_open_preferences(self) -> None:
        """View → Preferences… → open the modal SettingsDialog (D-10).

        Reads the current OAuth state from
        :mod:`marmelade.youtube.oauth`, constructs the dialog, wires the
        connect/disconnect signals to the local slots, and exec()s. The
        dialog reference is stored on ``self._settings_dialog`` so the
        connect/disconnect slots can push state back into it on completion.

        ``load_or_refresh`` may raise
        :class:`google.auth.exceptions.RefreshError` if the persisted
        refresh token has been revoked at Google — we treat that as
        "not connected" for display purposes (Plan 08-04 will surface
        the "Reconnect YouTube" UX at upload time per D-25).
        """
        from google.auth.exceptions import RefreshError

        try:
            creds = _yt_oauth.load_or_refresh()
        except RefreshError:
            creds = None

        is_connected = creds is not None
        channel_name: str | None = None
        if is_connected:
            try:
                channel_name = _yt_oauth.channel_info(creds).get("title") or None
            except Exception:
                # channel_info network failure — fall back to a generic label.
                # Local credentials are still considered "connected".
                channel_name = "your YouTube channel"

        self._settings_dialog = SettingsDialog(
            is_connected=is_connected,
            channel_name=channel_name,
            playhead_offset_sec=self._playhead_visual_offset_sec,
            parent=self,
        )
        self._settings_dialog.youtube_connect_requested.connect(
            self._on_youtube_connect
        )
        self._settings_dialog.youtube_disconnect_requested.connect(
            self._on_youtube_disconnect
        )
        self._settings_dialog.playhead_offset_changed.connect(
            self._on_playhead_offset_changed
        )
        self._settings_dialog.exec()

    def _on_playhead_offset_changed(self, value: float) -> None:
        """Preferences → live-apply + persist the playhead visual offset.

        quick-260625 — fires on every spinbox change so the user can tune the
        sync while audio plays and watch the playhead shift. Persisted to
        QSettings so it survives restarts. Cosmetic only (see
        :data:`_PLAYHEAD_VISUAL_OFFSET_SEC`).
        """
        self._playhead_visual_offset_sec = float(value)
        QSettings().setValue(_PLAYHEAD_OFFSET_SETTINGS_KEY, float(value))

    def _on_youtube_connect(self) -> None:
        """SettingsDialog Connect button → drive the OAuth flow + update UI.

        Calls :func:`marmelade.youtube.oauth.first_time_connect`
        synchronously (the loopback flow blocks until the browser callback
        fires). On success queries :func:`channel_info` for the channel
        title and pushes both into the dialog via
        :meth:`SettingsDialog.update_connection_state`. On failure
        (browser closed, network error, invalid_client from a placeholder
        client_id, etc.) surfaces a ``QMessageBox.warning`` and leaves
        the dialog at "Not connected".
        """
        try:
            creds = _yt_oauth.first_time_connect()
            channel_name = _yt_oauth.channel_info(creds).get("title") or "your YouTube channel"
        except Exception as exc:  # noqa: BLE001 — surface to user as warning
            QMessageBox.warning(
                self,
                "YouTube Connection Failed",
                f"Could not complete the YouTube connection:\n\n{exc}",
            )
            return
        if self._settings_dialog is not None:
            self._settings_dialog.update_connection_state(True, channel_name)

    def _on_youtube_disconnect(self) -> None:
        """SettingsDialog Disconnect button → revoke + clear + update UI.

        :func:`marmelade.youtube.oauth.disconnect` is best-effort on the
        revoke POST (D-09 / T-08-02-10) — never raises on network failure;
        the local keyring + plaintext-fallback entry are cleared
        unconditionally. The dialog flips back to "Not connected".
        """
        try:
            _yt_oauth.disconnect()
        except Exception as exc:  # noqa: BLE001 — defensive
            QMessageBox.warning(
                self,
                "YouTube Disconnect Failed",
                f"Could not fully disconnect:\n\n{exc}",
            )
            return
        if self._settings_dialog is not None:
            self._settings_dialog.update_connection_state(False, None)

    def _on_master_all_requested(self) -> None:
        """KeepersSidebar Master button (idle state) → kick off mastering.

        Quick-260615-l4y — the "Master All Keepers" button starts the
        mastering pass immediately with no modal; the user only sees the
        output folder + format picker when they later click the separate
        Export button (see :meth:`_on_export_all_requested`). Master is a
        no-friction precompute the user can re-run at any time; export is
        the explicit "where + how" decision.

        Bumps the generation, builds the master-all queue, transitions the
        Master button to the running state, and kicks off the first
        MasteringRunnable.
        """
        if self._current_path is None or self._current_playback_path is None:
            self.statusBar().showMessage(
                "Wait for audio proxy build to complete.", 5000
            )
            return

        keepers = [
            r
            for r in self._regions_overlay.regions_data()
            if r.state == "keeper"
        ]
        if not keepers:
            # Defensive — the button should be disabled in this state.
            return

        # No modal — start mastering immediately. target_dir / fmt are
        # collected later via the export modal when the user clicks the
        # separate Export button.
        self._kickoff_master_all()

    def _build_export_all_confirmation_dialog(self, keeper_count: int):
        """Build (but do not exec) the "Export N Keepers" output-options dialog.

        Modal: output folder (read-only QLineEdit + Browse, default from
        QSettings ``export_dir``) + format combobox (MP3 320 kbps / WAV
        float32, default MP3). Ok-role "Start export"; Cancel-role
        "Don't export now". Shown only when the user clicks the Export
        button — the mastering pass itself runs modal-free.

        Factored out so tests can exercise dialog construction without
        modal-loop quirks. Stores the user's choices via setProperty
        before accept() so callers can read them after exec().
        """
        from PySide6.QtWidgets import (
            QComboBox,
            QDialogButtonBox,
            QFormLayout,
            QPushButton as _QPushButton,
        )

        s = QSettings("Marmelade", "Marmelade")
        default_dir_raw = s.value("export_dir", "")
        default_dir = (
            Path(str(default_dir_raw))
            if default_dir_raw
            else Path.home() / "Marmelade Exports"
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Export {keeper_count} mastered Keepers")
        outer = QVBoxLayout(dlg)

        form = QFormLayout()
        dir_edit = QLineEdit(str(default_dir))
        dir_edit.setReadOnly(True)
        browse_btn = _QPushButton("Browse…")

        def _on_browse() -> None:
            chosen = QFileDialog.getExistingDirectory(
                dlg, "Choose target folder", dir_edit.text()
            )
            if chosen:
                dir_edit.setText(chosen)

        browse_btn.clicked.connect(_on_browse)
        # Pair the line-edit + browse into a single row.
        dir_row = QWidget(dlg)
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.addWidget(dir_edit, 1)
        dir_layout.addWidget(browse_btn, 0)

        fmt_combo = QComboBox(dlg)
        fmt_combo.addItem("MP3 (320 kbps)", userData="mp3")
        fmt_combo.addItem("WAV (float32)", userData="wav")
        fmt_combo.setCurrentIndex(0)

        form.addRow("Output folder:", dir_row)
        form.addRow("Format:", fmt_combo)
        outer.addLayout(form)

        # QDialogButtonBox with renamed buttons.
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            dlg,
        )
        ok_btn = box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setText("Start export")
        cancel_btn = box.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_btn is not None:
            cancel_btn.setText("Don't export now")
        outer.addWidget(box)

        def _on_accept() -> None:
            target = Path(dir_edit.text())
            fmt = fmt_combo.currentData() or "mp3"
            dlg.setProperty("master_all_target_dir", target)
            dlg.setProperty("master_all_format", str(fmt))
            # Persist the chosen dir so future picks default to it.
            s.setValue("export_dir", str(target))
            dlg.accept()

        box.accepted.connect(_on_accept)
        box.rejected.connect(dlg.reject)
        return dlg

    def _kickoff_master_all(
        self,
        target_dir: Path | str | None = None,
        fmt: str | None = None,
    ) -> None:
        """Kick off the batch mastering pass. Public test seam.

        Production: invoked by :meth:`_on_master_all_requested` with no
        arguments — the user clicked "Master All Keepers" and the export
        folder/format isn't decided yet (the modal for those runs later
        when the user clicks the separate Export button).

        Tests: may pass ``target_dir`` + ``fmt`` directly to pre-seed
        the export destination and skip the export modal. When both are
        passed, the values are stashed on ``self._master_all_target_dir``
        / ``self._master_all_format`` so the test can drive straight
        through to the export loop without exec'ing a real modal.
        """
        if target_dir is not None:
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

        keepers = sorted(
            (
                r
                for r in self._regions_overlay.regions_data()
                if r.state == "keeper"
            ),
            key=lambda r: r.start_sec,
        )
        if not keepers:
            return

        # Only overwrite when callers passed real values. Mastering
        # itself doesn't read these — they're stashed here as a
        # pre-seed so tests can skip the export modal.
        if target_dir is not None:
            self._master_all_target_dir = target_dir
        if fmt is not None:
            self._master_all_format = fmt
        self._master_all_generation += 1
        self._master_all_queue = list(keepers)
        self._master_all_total = len(keepers)
        self._master_all_completed_count = 0
        self._master_all_failed_ids = set()
        self._master_all_failure_msgs = {}

        # Transition the Master button to the running (cancel) state.
        self._keepers_sidebar.set_batch_state("running")
        # Update dock title to show progress.
        self._update_keepers_dock_title_master_all(0)

        # Kick off the first runnable.
        self._kick_next_master_all()

    def _update_keepers_dock_title_master_all(self, completed: int) -> None:
        """Set the Keepers dock title to ``Keepers (N) — Mastering K/N``."""
        n = self._master_all_total
        try:
            self._dock_keepers.setWindowTitle(
                f"Keepers ({n}) — Mastering {completed}/{n}"
            )
        except RuntimeError:
            pass

    def _kick_next_master_all(self) -> None:
        """Pop the next keeper off the queue and spawn its MasteringRunnable.

        Per UI-SPEC + RESEARCH §Pattern 7: walks keepers sequentially
        (one MasteringRunnable at a time on the global QThreadPool).
        Cache HIT short-circuits — set badge to Ready, recurse via
        QTimer.singleShot to avoid stack growth.
        """
        gen = self._master_all_generation
        if not self._master_all_queue:
            self._on_master_all_complete()
            return

        keeper = self._master_all_queue.pop(0)
        kid = keeper.id

        # No mastering snapshot — skip Phase A for this keeper; Phase C
        # will still export it from the source proxy.
        if keeper.mastering is None:
            QTimer.singleShot(0, self._kick_next_master_all)
            return

        try:
            src_key = proxy_cache.cache_key(self._current_path)
            chash = config_hash(keeper.mastering)
            target = mastered_cache_path(
                default_cache_root(), src_key, kid, chash
            )
        except Exception as exc:
            # Bad config / key — mark failure, continue.
            logger.exception(
                "Master All: keeper %s config/cache-key resolution failed", kid
            )
            self._master_all_failed_ids.add(kid)
            self._master_all_failure_msgs[kid] = str(exc)
            row = self._keepers_sidebar.find_row(kid)
            if row is not None:
                row.set_mastering_status("Failed", "#E5484D")
            self._master_all_completed_count += 1
            self._update_keepers_dock_title_master_all(
                self._master_all_completed_count
            )
            QTimer.singleShot(0, self._kick_next_master_all)
            return

        # Cache HIT — skip mastering, mark Ready.
        if is_mastered_cache_fresh(target):
            row = self._keepers_sidebar.find_row(kid)
            if row is not None:
                row.set_mastering_status("Ready", "#7FBFFF")
            self._master_all_completed_count += 1
            self.mastering_complete.emit(kid)
            self._update_keepers_dock_title_master_all(
                self._master_all_completed_count
            )
            QTimer.singleShot(0, self._kick_next_master_all)
            return

        # Spawn the runnable. ``_current_playback_path`` is the canonical
        # 44.1 kHz stereo float32 audio source (either the source WAV for
        # native-WAV opens, or the audio-proxy WAV under the cache for
        # non-WAV opens). NOT ``_current_proxy_p`` — that variable holds
        # the peak-builder's ``peaks.dat`` output, which is not audio.
        #
        # Plan 07-08 — read the source-proxy sample rate so we can
        # forward keeper.start_sec / end_sec as frame indices. sf.info
        # failure is treated as a per-keeper failure (same UX as the
        # cache-target resolution failure above): mark Failed + recurse.
        try:
            src_sr = int(
                sf.info(str(self._current_playback_path)).samplerate
            )
        except Exception as exc:
            logger.exception("Master All: keeper %s sf.info failed", kid)
            self._master_all_failed_ids.add(kid)
            self._master_all_failure_msgs[kid] = f"sf.info failed: {exc}"
            row = self._keepers_sidebar.find_row(kid)
            if row is not None:
                row.set_mastering_status("Failed", "#E5484D")
            self._master_all_completed_count += 1
            self._update_keepers_dock_title_master_all(
                self._master_all_completed_count
            )
            QTimer.singleShot(0, self._kick_next_master_all)
            return

        runnable = MasteringRunnable(
            self._current_playback_path,
            target,
            kid,
            keeper.mastering,
            start_frame=int(keeper.start_sec * src_sr),
            end_frame=int(keeper.end_sec * src_sr),
        )
        # Default-arg closures (Phase 1 LEARNINGS late-binding guard).
        runnable.signals.progress.connect(
            lambda pct, k=kid, g=gen: self._on_master_all_progress(pct, k, g)
        )
        runnable.signals.finished.connect(
            lambda path, k=kid, g=gen: self._on_master_all_keeper_finished(
                path, k, g
            )
        )
        runnable.signals.error.connect(
            lambda msg, k=kid, g=gen: self._on_master_all_keeper_error(
                msg, k, g
            )
        )
        runnable.signals.cancelled.connect(
            lambda k=kid, g=gen: self._on_master_all_keeper_cancelled(k, g)
        )
        row = self._keepers_sidebar.find_row(kid)
        if row is not None:
            row.set_mastering_status("Mastering 0%", "#9CA3AF")
        self._mastering_runnables[kid] = runnable
        self._dispatch_mastering_runnable(runnable)

    def _on_master_all_progress(
        self, pct: int, keeper_id: str, gen: int
    ) -> None:
        """Per-keeper progress emit during Phase A."""
        if gen != self._master_all_generation:
            return
        try:
            row = self._keepers_sidebar.find_row(keeper_id)
            if row is not None:
                row.set_mastering_status(
                    f"Mastering {int(pct)}%", "#9CA3AF"
                )
        except RuntimeError:
            # Window/widget torn down between cancel + late signal.
            pass

    def _on_master_all_keeper_finished(
        self, path: str, keeper_id: str, gen: int
    ) -> None:
        """A Phase A keeper finished successfully — recurse to the next."""
        if gen != self._master_all_generation:
            return
        try:
            row = self._keepers_sidebar.find_row(keeper_id)
            if row is not None:
                row.set_mastering_status("Ready", "#7FBFFF")
        except RuntimeError:
            return
        self._mastering_runnables.pop(keeper_id, None)
        self._master_all_completed_count += 1
        self.mastering_complete.emit(keeper_id)
        self._update_keepers_dock_title_master_all(
            self._master_all_completed_count
        )
        QTimer.singleShot(0, self._kick_next_master_all)

    def _on_master_all_keeper_error(
        self, msg: str, keeper_id: str, gen: int
    ) -> None:
        """A Phase A keeper raised — mark Failed, continue with the rest."""
        if gen != self._master_all_generation:
            return
        try:
            row = self._keepers_sidebar.find_row(keeper_id)
            if row is not None:
                row.set_mastering_status("Failed", "#E5484D")
                try:
                    row._master.setToolTip(f"Mastering failed: {msg}")
                except RuntimeError:
                    pass
        except RuntimeError:
            return
        logger.error("Master All: keeper %s failed: %s", keeper_id, msg)
        self._mastering_runnables.pop(keeper_id, None)
        self._master_all_failed_ids.add(keeper_id)
        self._master_all_failure_msgs[keeper_id] = msg
        self._master_all_completed_count += 1
        self._update_keepers_dock_title_master_all(
            self._master_all_completed_count
        )
        QTimer.singleShot(0, self._kick_next_master_all)

    def _on_master_all_keeper_cancelled(
        self, keeper_id: str, gen: int
    ) -> None:
        """A Phase A keeper was cancelled — terminate the queue.

        Cancel does NOT recurse to the next keeper — the entire Phase A
        loop is short-circuited per UI-SPEC §"Master & Export All —
        in-flight progress" line 343 (Cancel mastering reverts to
        Phase B with whatever was already successful).
        """
        if gen != self._master_all_generation:
            return
        try:
            row = self._keepers_sidebar.find_row(keeper_id)
            if row is not None:
                row.set_mastering_status("", "#9CA3AF")
        except RuntimeError:
            pass
        self._mastering_runnables.pop(keeper_id, None)
        # Do NOT recurse. Phase A is terminated by the cancel slot.

    def _on_master_all_cancel_requested(self) -> None:
        """Phase A cancel — 3-layer pattern (D-08 + D-18).

        Layer 1: bump ``_master_all_generation`` so any in-flight
                 signal becomes stale and is dropped by the slots.
        Layer 2: call ``cancel()`` on every in-flight runnable.
        Layer 3: clear ``_master_all_queue`` so not-yet-started keepers
                 are skipped.

        Then transition via _on_master_all_complete (which reverts the
        Master button to idle and re-probes the Export gate) with
        whatever successful work survives.
        """
        # Layer 1.
        self._master_all_generation += 1
        # Layer 2 + 3.
        # CR-03 (Phase 7 review) — pop synchronously while cancelling.
        # The generation bump above means the cancelled-slot pop will
        # see a stale generation and return early, leaking the
        # MasteringRunnable in the dict forever. Drain the dict here,
        # cancel each runnable, and clear the row UI synchronously.
        in_flight = list(self._mastering_runnables.items())
        self._mastering_runnables.clear()
        for keeper_id, runnable in in_flight:
            try:
                runnable.cancel()
            except Exception:
                pass
            row = self._keepers_sidebar.find_row(keeper_id)
            if row is not None:
                try:
                    row.set_mastering_status("", "#9CA3AF")
                except RuntimeError:
                    pass
        self._master_all_queue = []
        # Status-bar acknowledgement so the user knows the click registered.
        self.statusBar().showMessage(
            "Cancelling mastering — current stage will finish first…",
            5000,
        )
        # Do NOT block the GUI thread waiting for QThreadPool here —
        # the cancelled() signal arrives async; revert to idle
        # immediately via _on_master_all_complete. The runnable's
        # terminal signal will be ignored via the generation guard. If 0
        # keepers completed, the toast in _on_master_all_complete will be
        # the destructive one.
        QTimer.singleShot(0, self._on_master_all_complete)

    def _on_master_all_complete(self) -> None:
        """Batch master finished — revert Master to idle, open Export gate.

        Quick-260615-l4y — always reverts the Master button to its idle
        ("Master All Keepers") state and re-probes the Export + bundle
        buttons so the freshly-written mastered caches enable them. There
        is no longer a phase_b "Export N" transition; the Export button is
        a persistent sibling gated purely by cache freshness.
        """
        total = self._master_all_total
        failed = len(self._master_all_failed_ids)
        ok_count = max(0, self._master_all_completed_count - failed)

        # Surface the real first failure cause in the toast so the user
        # sees WHAT went wrong without hovering each row badge. The full
        # traceback is in the log (mastering_worker + the failure sites
        # above); the toast carries a truncated representative string.
        first_msg = ""
        if self._master_all_failure_msgs:
            first_msg = next(iter(self._master_all_failure_msgs.values())) or ""
        if len(first_msg) > 200:
            first_msg = first_msg[:200] + "…"

        if total > 0 and failed > 0 and ok_count == 0:
            self.statusBar().showMessage(
                f"All {total} Keepers failed to master: {first_msg} "
                "— see log for full traceback",
                10000,
            )
        elif failed > 0:
            self.statusBar().showMessage(
                f"Mastered {ok_count}/{total}; {failed} failed: {first_msg} "
                "— see log for full traceback",
                10000,
            )

        # Restore dock title — Phase A done.
        try:
            self._dock_keepers.setWindowTitle(f"Keepers ({total})")
        except RuntimeError:
            pass

        # Always revert the Master button to idle, then re-probe the
        # Export + bundle buttons so the freshly-written mastered caches
        # open their freshness gates.
        self._keepers_sidebar.set_batch_state("idle")
        self._keepers_sidebar.refresh_export_button()
        self._keepers_sidebar.refresh_bundle_button()

    def _on_export_all_requested(self) -> None:
        """Export button click — output-options modal + sequential export.

        Quick-260615-l4y — the Export button is a persistent sibling of
        the Master button, enabled once every keeper has a fresh mastered
        cache. This slot owns the output folder / format modal and runs
        the sequential batch export, guarded by the
        ``_export_all_in_flight`` sentinel.

        Each keeper goes through Phase 3's ExportRunnable (same pipeline
        used by per-region export); when ``keeper.mastering`` is set
        AND the mastered cache is fresh, we pass the cache path via the
        D-20 ``source_path`` keyword. Otherwise ``source_path=None`` →
        existing Phase 3 behavior (source proxy is the audio source).
        """
        if self._current_path is None or self._current_playback_path is None:
            return

        keepers = sorted(
            (
                r
                for r in self._regions_overlay.regions_data()
                if r.state == "keeper"
            ),
            key=lambda r: r.start_sec,
        )
        if not keepers:
            return

        # Open the output-options modal if target_dir / fmt haven't been
        # pre-seeded by a test. Cancel → no state change (the Export
        # button stays enabled so the user can retry).
        if self._master_all_target_dir is None or self._master_all_format is None:
            dlg = self._build_export_all_confirmation_dialog(len(keepers))
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            target_dir = dlg.property("master_all_target_dir")
            fmt = dlg.property("master_all_format")
            if not target_dir or not fmt:
                return
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            self._master_all_target_dir = target_dir
            self._master_all_format = str(fmt)

        # Quick-260615-l4y — mark the batch export in-flight. This sentinel
        # replaces the removed sidebar phase_c_running state and is the
        # loop-control guard the three export-all slots read. It MUST be
        # set True before _kick_next_export_all is first invoked below.
        self._export_all_in_flight = True
        # Disable the Export button for the duration so a double-click
        # can't spawn a second queue (T-l4y-02). Re-enabled via the
        # freshness re-probe in _on_export_all_complete.
        self._keepers_sidebar._export_button.setEnabled(False)

        # WR-02 (Phase 7 review) — skip keepers whose mastering pass
        # failed. Previously, those keepers were exported from the
        # source proxy (the mastered cache freshness check returned
        # False, so source_path fell back to the proxy) without any
        # indication to the user that they got unmastered output
        # despite clicking Export. Now we exclude them from the queue
        # and surface the count in the final toast via
        # _export_all_skipped_count below.
        failed = self._master_all_failed_ids
        skipped_ids = [k.id for k in keepers if k.id in failed]
        self._export_all_skipped_count = len(skipped_ids)
        self._export_all_queue = [k for k in keepers if k.id not in failed]
        self._export_all_finished_count = 0
        self._export_all_failed_ids = set()

        # Connect a one-shot finished/cancelled/error listener that
        # drives the sequential loop. We use the existing
        # ``export_complete`` signal as the loop pump for finished
        # because it has the success-only discipline we need; failures
        # are observed via the existing _on_export_error path.
        # The export_complete signal already drains to a slot the
        # tests can wait on.
        # Re-use the existing single-export pipeline (spawn-new
        # cancels old) but route the post-finish hop ourselves by
        # listening once.
        self._export_complete_loop_conn = self.export_complete.connect(
            self._on_export_all_next
        )
        self._kick_next_export_all()

    def _kick_next_export_all(self) -> None:
        """Pop next keeper, build per-keeper export args, spawn the worker."""
        if not self._export_all_queue:
            self._on_export_all_complete()
            return

        keeper = self._export_all_queue.pop(0)
        target_dir = self._master_all_target_dir
        fmt = self._master_all_format
        assert target_dir is not None and fmt is not None

        # Naming uses the keeper's start_sec (per CONTEXT D-A4-1) and
        # the dominant trait derived from cached heatmaps.
        region_start = float(keeper.start_sec)
        region_end = float(keeper.end_sec)

        cache_root = default_cache_root()
        if self._current_cache_key is not None:
            trait = naming_resolver.dominant_trait_for_region(
                cache_root, self._current_cache_key,
                region_start, region_end,
            )
        else:
            trait = "clip"
        try:
            dst_path = naming_resolver.resolve_filename(
                source_path=self._current_source_path or self._current_path,
                region_start_sec=region_start,
                trait=trait,
                ext=fmt,
                output_dir=target_dir,
            )
        except (ValueError, RuntimeError) as exc:
            # Naming failed — count as failure, continue with next.
            self._export_all_failed_ids.add(keeper.id)
            QTimer.singleShot(0, self._kick_next_export_all)
            return

        engine_sr = self._playback_engine.sample_rate
        sr = engine_sr if engine_sr > 0 else 44100

        # Decide source_path override per D-20.
        source_path: Path | None = None
        if keeper.mastering is not None:
            try:
                src_key = proxy_cache.cache_key(self._current_path)
                chash = config_hash(keeper.mastering)
                cache_p = mastered_cache_path(
                    default_cache_root(), src_key, keeper.id, chash
                )
                if is_mastered_cache_fresh(cache_p):
                    source_path = cache_p
            except Exception:
                source_path = None

        # Frame range — when source_path is the mastered cache, that
        # file's frame timeline holds the WHOLE region (the cache was
        # rendered for exactly that region by the MasteringRunnable),
        # so start=0, end=total_frames. When source_path is None we
        # use the source-proxy frame timeline (existing Phase 3 path).
        if source_path is not None:
            try:
                info = sf.info(str(source_path))
                start_frame = 0
                end_frame = int(info.frames)
            except Exception:
                # Fall back to source-proxy frames if sf.info fails.
                source_path = None
                start_frame = int(region_start * sr)
                end_frame = int(region_end * sr)
        else:
            start_frame = int(region_start * sr)
            end_frame = int(region_end * sr)

        region_duration = (end_frame - start_frame) / float(sr)
        # quick-260626-o9y — config-driven fade (was forced 2.0 s). Read the
        # keeper's fade enabled flag + duration via fade_params; disabled →
        # 0.0 (no fade), else clamp to half the region so the in/out fades
        # never overlap.
        fade_enabled, fade_dur = fade_params(keeper.mastering)
        fade_sec = (
            min(fade_dur, max(0.0, region_duration / 2.0)) if fade_enabled else 0.0
        )
        fade_frames = int(fade_sec * sr)

        # quick-260621-gfq — export no longer normalizes. Normalize is the
        # mastering chain's final stage, so a normalize-enabled keeper carries
        # its normalized bytes through the mastered-cache export path
        # (source_path). Raw export streams the source verbatim.
        self._spawn_export_worker(
            proxy_path=Path(self._current_playback_path),
            dst_path=dst_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fade_frames=fade_frames,
            fmt=fmt,
            sample_rate=sr,
            source_path=source_path,
        )

    def _on_export_all_next(self, _dst_path: str) -> None:
        """``export_complete`` arrived during a batch export — kick next keeper.

        Quick-260615-l4y — guarded by the ``_export_all_in_flight``
        sentinel so this loop pump stays inert for unrelated single-keeper
        exports (which also emit ``export_complete``).
        """
        if not self._export_all_in_flight:
            return  # not a batch export — defensive, ignore.
        self._export_all_finished_count += 1
        QTimer.singleShot(0, self._kick_next_export_all)

    def _on_export_all_complete(self) -> None:
        """Batch export done — final toast + clear the in-flight sentinel.

        Quick-260615-l4y — funnel point for the complete, failure-drain,
        and cancel paths. Clears ``_export_all_in_flight`` and re-probes
        the Export button (re-enabling it if the caches are still fresh).
        """
        # Disconnect the loop pump.
        try:
            self.export_complete.disconnect(self._on_export_all_next)
        except (RuntimeError, TypeError):
            pass

        target = self._master_all_target_dir
        n = self._export_all_finished_count
        # WR-02 (Phase 7 review) — surface unmastered-skip count so the
        # user knows why their export count is less than their keeper
        # count. _export_all_skipped_count is set in _on_export_all_requested.
        skipped = getattr(self, "_export_all_skipped_count", 0)
        if skipped > 0:
            self.statusBar().showMessage(
                f"Exported {n} Keepers to {target} "
                f"— skipped {skipped} unmastered Keeper"
                f"{'s' if skipped != 1 else ''}",
                10000,
            )
        else:
            self.statusBar().showMessage(
                f"Exported {n} Keepers to {target}", 10000,
            )
        # Quick-260615-l4y — clear the in-flight sentinel (covers the
        # complete, failure-drain, and cancel paths — all funnel here)
        # and re-probe the Export button so it re-enables if the caches
        # are still fresh. The Master button is already idle.
        self._export_all_in_flight = False
        self._keepers_sidebar.refresh_export_button()
        # WR-07 (Phase 7 review) — symmetric reset of master + export
        # scratch fields. Not a correctness bug today (the next batch
        # cycle re-initializes them in _kickoff_master_all +
        # _on_export_all_requested) but a defensive read of any of these
        # between completion and the next cycle would otherwise return
        # stale data. Keep _master_all_target_dir + _master_all_format so
        # the next cycle's confirmation dialog defaults match the last one.
        self._export_all_queue = []
        self._export_all_finished_count = 0
        self._export_all_failed_ids = set()
        self._export_all_skipped_count = 0
        self._master_all_queue = []
        self._master_all_failed_ids = set()
        self._master_all_failure_msgs = {}

    # ----------------------- Phase 7 Plan 07-04 — A/B preview wiring

    def _on_keeper_selection_changed(self, region_id: str) -> None:
        """KeepersSidebar.selection_changed → track + refresh A/B toggle.

        UI-SPEC §"A/B Preview Toolbar Toggle" line 550 — most-recently-
        row-clicked Keeper is the A/B selection. The same click also
        emits ``jump_requested`` (the dual-purpose seek + select
        gesture); the sidebar forwards both signals from the same row
        click.
        """
        self._selected_keeper_id = region_id or None
        self._refresh_ab_toggle_enabled_state()
        # quick-260629 — selecting a keeper auto-switches the A/B preview to
        # mastered (B) when that keeper has a fresh mastered cache (the toggle
        # is enabled iff so), else back to source (A). Mirrors the Play-button
        # behaviour (_on_keeper_play) for plain row/region selection. A genuine
        # A<->B change routes through _on_ab_state_changed, which reseeds the
        # engine (and immediately pauses again when not playing, so selecting a
        # keeper never starts audio). When already on B and a DIFFERENT mastered
        # keeper is selected mid-playback, force a reseed so the live audio
        # follows the newly-selected keeper's cache.
        if hasattr(self, "_ab_toggle"):
            target = "B" if self._ab_toggle.is_enabled else "A"
            if self._ab_toggle.state != target:
                self._ab_toggle.set_state(target)
            elif target == "B" and self._playback_engine.is_playing:
                self._on_ab_state_changed("B")

    def _refresh_ab_toggle_enabled_state(self) -> None:
        """Re-evaluate the A/B toggle's enable state + tooltip.

        Three disable conditions per UI-SPEC §"A/B Preview Toolbar
        Toggle" lines 540-556:

        1. ``_selected_keeper_id is None`` — no keeper selected.
        2. The selected keeper's ``mastering`` field is None.
        3. The selected keeper has ``mastering`` but no fresh mastered
           cache exists at the keeper's current config_hash.

        Plan 07-09 tooltip contract:

        * The permanent discoverability tooltip (``self._ab_default_tooltip``,
          set in :meth:`_build_toolbar`) is restored on every re-enable AND
          on the ``kid is None`` disabled branch — so hovering an enabled
          OR no-keeper-selected widget always gives a discoverability hint.
        * Disabled-with-context branches (``mastering is None``,
          cache-pending) override the default with a state-specific message.
        * The pre-file-load branch (``current_path / proxy_p is None``)
          keeps ``setToolTip("")`` — that's an empty-state where the toolbar
          shouldn't compete with the file-open UI.

        Verbatim tooltip strings per UI-SPEC §"Error states" lines
        422-423 + the cache-pending plan-defined string + the Plan 07-09
        discoverability default.
        """
        if not hasattr(self, "_ab_toggle"):
            return  # called before the toolbar built — silently bail

        kid = self._selected_keeper_id
        if kid is None:
            self._ab_toggle.set_enabled(False)
            # Plan 07-09 — restore the discoverability default so the user
            # hovers the (visible) toolbar widget and learns the keeper-row
            # prerequisite. Previously cleared to "".
            self._ab_toggle.setToolTip(self._ab_default_tooltip)
            return

        mastering = self._regions_overlay.get_mastering(kid)
        if mastering is None:
            self._ab_toggle.set_enabled(False)
            self._ab_toggle.setToolTip(
                "A/B preview needs a mastered Keeper. "
                "Click the gear button on a Keeper row to configure mastering."
            )
            return

        # Cache freshness check — requires a current source file with
        # a decodable audio path. _current_playback_path is the audio
        # proxy (or source WAV for native WAV); _current_proxy_p is the
        # peak-builder's peaks.dat binary and is NOT the right gate for
        # the audio-swap path (Plan 07-10 fix — same source-path rule
        # Plan 07-08 enforced for mastering spawns).
        if self._current_path is None or self._current_playback_path is None:
            self._ab_toggle.set_enabled(False)
            # Pre-file-load — defer to the empty-state UI; do not show the
            # discoverability tooltip yet (Plan 07-09).
            self._ab_toggle.setToolTip("")
            return

        try:
            src_key = proxy_cache.cache_key(self._current_path)
            chash = config_hash(mastering)
            cache_p = mastered_cache_path(
                default_cache_root(), src_key, kid, chash
            )
        except Exception:
            # Defensive — bad config_hash inputs etc. Treat as disabled.
            self._ab_toggle.set_enabled(False)
            self._ab_toggle.setToolTip("")
            return

        if not is_mastered_cache_fresh(cache_p):
            self._ab_toggle.set_enabled(False)
            self._ab_toggle.setToolTip(
                "Mastered cache is being rendered. "
                "A/B preview becomes available when the row badge shows Ready."
            )
            return

        # All three conditions met — enable + restore the discoverability
        # default (Plan 07-09; previously cleared to ""). The permanent
        # tooltip gives a hover hint even when the widget is functional.
        self._ab_toggle.set_enabled(True)
        self._ab_toggle.setToolTip(self._ab_default_tooltip)

    def _on_ab_state_changed(self, new_state: str) -> None:
        """ABToggleWidget.state_changed → swap PlaybackEngine source.

        Press B during playback → reseed engine with mastered cache at
        current playhead position (full restart, audible click is
        accepted UX per D-13).

        Fail-closed (T-7-05): if the mastered cache is not fresh at
        switch time, revert toggle to A + status-bar toast + skip the
        ``play(...)`` call entirely. Source playback continues
        unchanged.
        """
        # Phase 7 Plan 07-04 — re-entrance guard for the fail-closed
        # revert. When the cache-missing branch calls
        # ``self._ab_toggle.set_state("A")`` it would normally re-enter
        # this slot via ``state_changed`` and reseed playback on the
        # source proxy — but the plan contract is "Do NOT call
        # playback_engine.play(...) — the swap is aborted, source
        # playback continues unchanged." So the revert is purely a
        # widget-state update; the audio engine is untouched.
        if getattr(self, "_ab_failclosed_in_progress", False):
            return

        kid = self._selected_keeper_id
        if kid is None:
            return  # defensive — shouldn't be reachable when widget enabled

        # Plan 07-10b — get the keeper's source-timeline bounds so we can
        # translate the engine's position between the source timeline
        # (A: full proxy) and the cache timeline (B: keeper-bounded audio
        # written by Plan 07-08). Without translation, pressing B with
        # the playhead inside the keeper region jumps the seek offset to
        # the SOURCE timeline value — which is past the cache file's EOF
        # because the cache only contains the keeper-bounded slice. The
        # engine raises ValueError on seek-past-EOF (user-reported 07-10).
        region_widget = self._regions_overlay.get_region(kid)
        if region_widget is None:
            return  # defensive — keeper region went away between select and press
        keeper_start_sec, keeper_end_sec = region_widget.getRegion()
        keeper_duration_sec = max(0.0, float(keeper_end_sec) - float(keeper_start_sec))

        if new_state == "A":
            # Plan 07-10 fix — was self._current_proxy_p (the peak-builder's
            # peaks.dat binary, NOT a decodable audio file). The legacy
            # mastering spawn had the same bug; Plan 07-08 fixed that site,
            # this is the matching fix for the A/B audio-swap path.
            target = self._current_playback_path
        else:  # "B"
            mastering = self._regions_overlay.get_mastering(kid)
            if (
                mastering is None
                or self._current_path is None
                or self._current_playback_path is None
            ):
                return  # defensive

            try:
                src_key = proxy_cache.cache_key(self._current_path)
                chash = config_hash(mastering)
                target = mastered_cache_path(
                    default_cache_root(), src_key, kid, chash
                )
            except Exception:
                return  # defensive — bad cache key inputs

            # T-7-05 mitigation — re-check freshness IMMEDIATELY before
            # swapping. The enable-state check ran at selection time;
            # the file may have been deleted by an external process
            # between then and now.
            if not is_mastered_cache_fresh(target):
                # Fail-closed: revert toggle to A (no-op-if-already-A),
                # show destructive toast, do NOT call play(). The
                # re-entrance guard prevents the revert's state_changed
                # emission from kicking off a source-proxy reseed.
                self._ab_failclosed_in_progress = True
                try:
                    self._ab_toggle.set_state("A")
                finally:
                    self._ab_failclosed_in_progress = False
                self.statusBar().showMessage(
                    "Mastered preview unavailable — cache is missing. "
                    "Re-master this keeper.",
                    10000,
                )
                return

        if target is None:
            return  # defensive — no playback target available

        # Capture current position + playing state, then reseed.
        was_playing = self._playback_engine.is_playing
        try:
            position = self._playback_engine.position_seconds
        except Exception:
            position = 0.0
        position = max(0.0, position)

        # Plan 07-10b — translate position between source and cache timelines.
        # The engine reports position relative to whatever file is currently
        # loaded (source proxy for A-state, keeper-bounded cache for B-state).
        # Clamp into the legal range for the destination file so engine.seek
        # never sees an out-of-bounds offset (the af.seek ValueError that
        # spammed stderr in the user-reported 07-10 trace).
        if new_state == "B":
            # Source → Cache: subtract keeper.start_sec, clamp to keeper duration.
            translated_position = position - float(keeper_start_sec)
            translated_position = max(0.0, min(translated_position, keeper_duration_sec))
        else:  # "A"
            # Cache → Source: add keeper.start_sec, clamp to keeper region in
            # source timeline so playback resumes inside the keeper region the
            # user is auditioning, not somewhere unrelated.
            translated_position = position + float(keeper_start_sec)
            translated_position = max(
                float(keeper_start_sec),
                min(translated_position, float(keeper_end_sec)),
            )

        try:
            self._playback_engine.play(str(target), start_seconds=translated_position)
            if not was_playing:
                # Plan §"Switch behavior" — if paused, immediately
                # pause again so the state matches what the user had
                # before pressing A/B. play() then immediate pause()
                # preserves the position cleanly through the engine's
                # state machine.
                try:
                    self._playback_engine.pause()
                except Exception:
                    pass
        except PlaybackError as e:
            # Surface play failures via the same QMessageBox path as
            # the spacebar shortcut for consistency.
            QMessageBox.warning(self, "Couldn't start playback", str(e))
        except ValueError as e:
            # Plan 07-10b — defensive net: even with the clamp above, any
            # future drift (e.g., float rounding past cache EOF, race with
            # external cache deletion) should NOT escape as an uncaught
            # ValueError that spams stderr. Surface a clean status-bar
            # message and let source playback continue unchanged.
            self.statusBar().showMessage(
                f"A/B preview seek failed — {e}",
                5000,
            )

    def _on_ab_shortcut_pressed(self, state: str) -> None:
        """A/B QShortcut handler — delegates to the toggle widget.

        Bail-out cases:
        * Widget disabled — shortcut is also disabled (no swap). Plan 07-09:
          emits a 3-second status-bar diagnostic explaining the keeper-row
          prerequisite, so the user sees visible feedback instead of a
          silent no-op. The audio engine remains untouched.
        * Modal dialog active — Qt's ``ApplicationShortcut`` semantics
          already suppress activation while a modal is up, but we
          defense-in-depth check ``QApplication.activeModalWidget()``
          so direct test invocations also honor the suppression.
        * Focused QLineEdit — see CR-02 below; the keystroke should
          reach the line-edit, not trigger an A/B swap.

        On valid press, calls ``self._ab_toggle.set_state(state)`` —
        the widget's STATE-KEY semantics suppress same-state no-ops,
        and ``state_changed`` fires the actual source swap via
        ``_on_ab_state_changed``.
        """
        if not hasattr(self, "_ab_toggle"):
            return
        if not self._ab_toggle.isEnabled():
            # Plan 07-09 — surface diagnostic feedback for the otherwise-
            # silent bail. The widget is disabled because the user hasn't
            # selected a keeper row with a Ready mastered cache (per
            # _refresh_ab_toggle_enabled_state's three-condition gate).
            # Without this status-bar nudge, the user perceives the A/B
            # keys as dead. 3-second message; the try/except mirrors the
            # libshiboken-guarded UI mutations elsewhere in MainWindow
            # (C++ object already deleted during teardown is harmless here).
            try:
                self.statusBar().showMessage(
                    "A/B preview: click a keeper row with a Ready mastered cache first",
                    3000,
                )
            except (RuntimeError, AttributeError):
                # libshiboken: window may have been torn down between
                # shortcut activation and slot dispatch. Silent skip.
                pass
            return
        if QApplication.activeModalWidget() is not None:
            return  # modal dialog up — defense-in-depth bail
        # CR-02 (Phase 7 review) — ``ApplicationShortcut`` consumes the
        # keystroke even when a QLineEdit has focus, so a user typing
        # "A" or "B" into a Keeper note would lose the character AND
        # fire a spurious A/B preview swap. Mirror the K/T/U pattern
        # used by ``_mark_hovered_region`` (see RESEARCH §Pitfall #5).
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            return
        self._ab_toggle.set_state(state)

    def _on_render_mode_shortcut(self, index: int) -> None:
        """Number-key (``"1".."N"``) render-mode switch — quick-260630-dqd.

        ``index`` is the zero-based render-mode index the pressed key maps to
        (key "1" -> 0, key "2" -> 1, …); the binding loop in ``__init__``
        derives one shortcut per ``RenderMode`` member so this handler never
        needs a hardcoded mode list (DQD-1).

        Bail-out cases (mirror the A/B / K/T/U convention, RESEARCH §Pitfall #5):
        * Focused QLineEdit — the user is typing into a Keeper-note field; the
          digit must reach the line-edit, not trigger a mode switch (T-dqd-02).
        * Out-of-range index — a number key with no corresponding mode (e.g.
          "7" when 6 modes exist) is a no-op rather than an IndexError or an
          invalid combo index (T-dqd-01); the guard reads the LIVE combo count.

        On a valid press we drive ``render_mode_combo.setCurrentIndex(index)``.
        The combo is the single source of truth (DQD-2): its
        ``currentIndexChanged`` fires ``_on_render_mode_changed``, which updates
        ``_render_mode``, re-renders the cached proxy in place, and (for a
        cold-cache spectral mode) emits ``spectral_build_requested``. We do NOT
        touch ``_render_mode`` or call ``_on_render_mode_changed`` directly —
        routing through the combo keeps the "View:" control visually synced.
        """
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            return  # user typing into a keeper-note field — defense-in-depth
        combo = self._waveform_view.render_mode_combo
        if index < 0 or index >= combo.count():
            return  # number key with no corresponding mode — no-op
        combo.setCurrentIndex(index)

    def _hide_overlay_if_waveform_owned(self) -> None:
        """Hide ProgressOverlay only if the audio proxy isn't using it.

        Waveform-proxy terminal slots (ready / error / cancelled) used to
        call ``self._overlay.hide()`` directly. When the audio proxy owns
        the overlay (Phase 2.1 HUMAN-UAT request #3), waveform finishing
        first must NOT hide the modal — the audio proxy is still building
        and its UI must stay visible. Funnels every waveform-side hide
        through this guard so the rule lives in one place.
        """
        if not self._audio_proxy_overlay_active:
            self._overlay.hide()

    def _on_proxy_ready(
        self, gen: int, runnable: PeakBuilderRunnable, proxy_p_obj: object
    ) -> None:
        """Worker finished successfully — load_proxy and render on GUI thread.

        CR-04: drop stale signals from previously-cancelled workers via the
        generation-token + runnable-identity double-guard. The generation
        token catches the most common case (a newer _open_file ran); the
        identity check is belt-and-suspenders for refactor accidents.
        """
        if gen != self._open_generation or runnable is not self._current_runnable:
            return  # CR-04: stale signal from a previously-cancelled open — drop silently
        proxy_p = Path(str(proxy_p_obj))
        try:
            arr, header = proxy_cache.load_proxy(proxy_p)
        except ProxyHeaderError as e:
            # Worker emitted finished with a bad header — treat as build error.
            self._hide_overlay_if_waveform_owned()
            self._waveform_view.clear()
            self._current_runnable = None
            show_corrupt_file(
                self,
                self._basename(self._current_path or proxy_p),
                str(e),
            )
            return

        probe = getattr(self, "_current_probe", None)
        path = self._current_path
        if probe is None or path is None:
            # Defensive — should not happen, but degrade gracefully.
            self._hide_overlay_if_waveform_owned()
            self._current_runnable = None
            return
        self._render_loaded_proxy(arr, header, probe, path)
        # Plan 03-01 — load sidecar (REG-04). Runs AFTER _render_loaded_proxy
        # so the WaveformView's _duration_s is populated for the overlay's
        # lazy bounds provider. Mirror of the cache-HIT branch in _open_file.
        cache_root = default_cache_root()
        key = proxy_cache.cache_key(path)
        self._load_sidecar_for_key(cache_root, key)
        self._hide_overlay_if_waveform_owned()
        self._current_runnable = None
        self.render_complete.emit()
        # Plan 02-04 — D-15 lazy compute. The Energy heatmap no longer
        # auto-fires on the cache-MISS success branch. The sidebar's
        # Energy checkbox is the user's trigger; this slot only completes
        # the waveform render path. The checkbox was reset to unchecked
        # by the cancel preamble at the top of _open_file.

    def _on_proxy_error(
        self, gen: int, runnable: PeakBuilderRunnable, msg: str
    ) -> None:
        """Worker raised — hide overlay, show corrupt-file dialog, clear view.

        CR-04: drop stale signals via the same double-guard as _on_proxy_ready.
        """
        if gen != self._open_generation or runnable is not self._current_runnable:
            return  # CR-04: stale signal from a previously-cancelled open — drop silently
        self._hide_overlay_if_waveform_owned()
        basename = (
            self._basename(self._current_path)
            if self._current_path is not None
            else "(unknown)"
        )
        self._waveform_view.clear()
        self._current_runnable = None
        show_corrupt_file(self, basename, msg)

    def _on_proxy_cancelled(
        self, gen: int, runnable: PeakBuilderRunnable
    ) -> None:
        """Worker was cancelled — hide overlay, leave view in empty state.

        CR-04: drop stale signals via the same double-guard. Note this slot
        is the COMMON case for stale signals (a cancelled worker emits
        `cancelled` after _open_file has already moved on); the guard
        ensures we don't accidentally hide the new file's overlay.
        """
        if gen != self._open_generation or runnable is not self._current_runnable:
            return  # CR-04: stale signal from a previously-cancelled open — drop silently
        self._hide_overlay_if_waveform_owned()
        self._current_runnable = None
        # WaveformView already in empty state; build_proxy cleaned up the .tmp.

    # ---------------------------------------- audio-proxy worker + slots
    # Plan 02.1-04 — the audio-proxy pipeline. The MISS branch in
    # ``_open_file`` calls ``_spawn_audio_proxy_worker`` which constructs an
    # ``AudioProxyRunnable``, stashes it on ``self._current_proxy_runnable``,
    # records four ``_mw_proxy_conn_*`` connection tokens, and submits to
    # the global ``QThreadPool``. Exactly one terminal signal fires per
    # build:
    #   * ``finished(str)`` → ``_on_audio_proxy_finished`` re-primes
    #     ``PlaybackEngine`` with the proxy path, re-enables the spacebar,
    #     refreshes the cache-size footer, and emits ``audio_proxy_complete``.
    #   * ``error(str)``    → ``_on_audio_proxy_error`` surfaces a
    #     QMessageBox.warning; spacebar stays disabled.
    #   * ``cancelled()``   → ``_on_audio_proxy_cancelled`` silently clears
    #     state; no user-facing dialog because cancel is voluntary.
    # All three slots use the same double-guard prologue
    # (generation-token mismatch OR runnable-identity mismatch → return).
    def _spawn_audio_proxy_worker(
        self, p: Path, probe: "audio_file.AudioProbe", gen: int
    ) -> None:
        """D-10 + D-11 + D-16/17/18 — spawn the audio-proxy build worker.

        Default-arg closure binding on every signal connect (Phase 1
        LEARNINGS §"Late-binding closure capture would break the
        generation-token guard"). The four ``QMetaObject.Connection``
        tokens are stashed on the runnable so the cancel preamble can
        target ONLY our wiring — a bare ``signal.disconnect()`` would nuke
        test watchers attached via ``qtbot.waitSignal``.
        """
        audio_cache_root = default_cache_root()
        key = cache_key(p)
        proxy_p = audio_proxy_path(audio_cache_root, key)
        proxy_p.parent.mkdir(parents=True, exist_ok=True)

        runnable = AudioProxyRunnable(p, proxy_p)
        self._current_proxy_runnable = runnable

        # Show the transient progress widget. setText + show; updates via
        # the progress slot below. Initial text is "0%" so the widget has
        # something visible immediately if the first progress emit lags.
        self._status_proxy_progress.setText("Preparing audio proxy: 0%")
        self._status_proxy_progress.show()

        # UX upgrade (Phase 2.1 HUMAN-UAT request #3) — surface the audio
        # proxy build as a modal overlay (like the file-open progress)
        # instead of a quiet status-bar text update. The overlay is a
        # child of the WaveformView; while visible, it intercepts mouse
        # events so the user cannot click-to-seek mid-build (which would
        # be a no-op anyway since `_current_playback_path` is None until
        # `_on_audio_proxy_finished` lands). Audio takes priority on the
        # overlay over the waveform proxy: if waveform also spawns a
        # cache-MISS build below, it skips its own overlay setup (see
        # `_spawn_proxy_worker` guard) so the audio body text stays put.
        try:
            basename = p.name
            duration_s = float(probe.duration_s)
        except Exception:  # pragma: no cover — probe always has these
            basename = str(p)
            duration_s = 0.0
        # Phase 2.1 HUMAN-UAT #3 (final) — inline progress banner anchored
        # at top-center of the WaveformView. Waveform stays visible
        # underneath; click-to-seek is gated off (see _on_seek_requested)
        # so the user can't accidentally seek mid-build (which would be
        # a no-op anyway — _current_playback_path is None until the
        # proxy finishes). The flag name is kept as
        # `_audio_proxy_overlay_active` for compatibility with the
        # cancel preamble + terminal handlers + waveform-proxy guards,
        # though it now means "audio proxy banner is shown" rather than
        # "audio proxy is using the full-screen overlay".
        # Stash the open args so the "Build proxy" button on the
        # unavailable banner can re-spawn the worker without re-running
        # the cancel preamble / probe pipeline. Cleared on close +
        # cleared by the cancel preamble on a fresh _open_file.
        self._audio_proxy_retry_args = (p, probe)
        self._audio_proxy_banner.configure_building(
            heading="Preparing audio proxy",
            body=(
                f"{basename} · {self._fmt_duration(duration_s)} · "
                "click-to-play locked until proxy completes"
            ),
        )
        # Anchor over the WaveformView even though the banner is a child
        # of MainWindow (sidesteps PyQtGraph compositing issues; see
        # __init__ comment on `_audio_proxy_banner` for full rationale).
        self._audio_proxy_banner.position_over_widget(self._waveform_view)
        self._audio_proxy_banner.show()
        self._audio_proxy_banner.raise_()
        self._audio_proxy_overlay_active = True
        # Hook the banner's cancel button to THIS audio runnable. Disconnect
        # any prior wiring (e.g., a previous file's runnable) so the button
        # cancels the right worker. PySide6 emits a cosmetic RuntimeWarning
        # on a no-prior-connection disconnect; suppress and continue.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                self._audio_proxy_banner.cancel_button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._audio_proxy_banner.cancel_button.clicked.connect(runnable.cancel)

        # No race guard on progress — stale-flicker is harmless and the
        # widget is about to be replaced or hidden anyway.
        runnable.signals.progress.connect(self._on_audio_proxy_progress)
        # Capture gen + runnable BY VALUE (default-arg trick) so a later
        # ``_open_file`` overwriting state does NOT mutate the closure.
        runnable._mw_proxy_conn_finished = runnable.signals.finished.connect(
            lambda obj, g=gen, r=runnable: self._on_audio_proxy_finished(
                g, r, obj
            )
        )
        runnable._mw_proxy_conn_error = runnable.signals.error.connect(
            lambda msg, g=gen, r=runnable: self._on_audio_proxy_error(
                g, r, msg
            )
        )
        runnable._mw_proxy_conn_cancelled = runnable.signals.cancelled.connect(
            lambda g=gen, r=runnable: self._on_audio_proxy_cancelled(g, r)
        )

        QThreadPool.globalInstance().start(runnable)

    def _on_audio_proxy_progress(self, pct: int) -> None:
        """Worker progress (0..100) — update the transient status text.

        No race guard needed: stale flicker before the real terminal signal
        lands is harmless and the widget is hidden by the cancel preamble
        anyway.
        """
        self._status_proxy_progress.setText(
            f"Preparing audio proxy: {int(pct)}%"
        )
        # Drive the inline banner's progress bar too (HUMAN-UAT #3 final).
        if self._audio_proxy_overlay_active:
            self._audio_proxy_banner.set_progress(int(pct))

    def _on_audio_proxy_finished(
        self,
        gen: int,
        runnable: AudioProxyRunnable,
        proxy_p_obj: object,
    ) -> None:
        """Worker finished — re-prime playback with the proxy path.

        Double-guard (generation token + runnable identity) drops stale
        signals from a previously-cancelled worker. Emits
        ``audio_proxy_complete`` LAST so test watchers see a fully-settled
        UI state.
        """
        if (
            gen != self._open_generation
            or runnable is not self._current_proxy_runnable
        ):
            return  # CR-04: stale signal from a previously-cancelled open
        proxy_path = Path(str(proxy_p_obj))
        try:
            self._playback_engine.prime(str(proxy_path))
        except Exception:
            pass
        # Route playback through the proxy from here on — the toolbar
        # play handler at _action_toggle_playback reads this to decide
        # which path to hand to ``engine.play()``. Without this, play()
        # would re-open the source MP3 and pay pedalboard's O(n) seek
        # cost on every spacebar press (SC-4 regression).
        self._current_playback_path = proxy_path
        self._status_proxy_progress.hide()
        # Tear down the modal overlay. The waveform proxy may have
        # rendered behind it in the meantime; hiding now reveals the
        # finished waveform with playback enabled.
        if self._audio_proxy_overlay_active:
            self._audio_proxy_banner.hide()
            self._audio_proxy_overlay_active = False
        # Clear retry args — proxy is built, nothing to rebuild from.
        self._audio_proxy_retry_args = None
        self._current_proxy_runnable = None
        self._shortcut_play_pause.setEnabled(
            self._playback_engine.is_available
        )
        self._update_cache_size_footer()
        # Test seam — success-only emission discipline. Tests use
        # ``qtbot.waitSignal(window.audio_proxy_complete, ...)`` as the
        # synchronisation point for "the proxy is ready, you can now
        # assert on the cache contents".
        self.audio_proxy_complete.emit(str(proxy_path))

    def _on_audio_proxy_error(
        self,
        gen: int,
        runnable: AudioProxyRunnable,
        msg: str,
    ) -> None:
        """Worker raised — surface a QMessageBox.warning; spacebar stays disabled.

        Double-guard drops stale signals. We do NOT emit
        ``audio_proxy_complete`` — the test seam is success-only.
        """
        if (
            gen != self._open_generation
            or runnable is not self._current_proxy_runnable
        ):
            return  # CR-04: stale signal from a previously-cancelled open
        self._status_proxy_progress.hide()
        self._current_proxy_runnable = None
        # Spacebar stays disabled — playback engine has no prime'd proxy.
        # Show the build-failed dialog AND keep the banner up in
        # unavailable state so the user has a one-click retry surface.
        QMessageBox.warning(self, "Audio proxy build failed", msg)
        self._show_audio_proxy_unavailable(
            heading="Audio proxy build failed",
            body_suffix="click Build to retry",
        )
        self._update_cache_size_footer()

    def _on_audio_proxy_cancelled(
        self,
        gen: int,
        runnable: AudioProxyRunnable,
    ) -> None:
        """Worker was cancelled — silently clear state. No user-facing dialog.

        Cancel is voluntary (the user opened another file or quit). The
        builder removed the ``.tmp`` already; the worker did a defensive
        second-pass unlink. Nothing to report.
        """
        if (
            gen != self._open_generation
            or runnable is not self._current_proxy_runnable
        ):
            return  # CR-04: stale signal from a previously-cancelled open
        self._status_proxy_progress.hide()
        self._current_proxy_runnable = None
        # Keep the banner up in unavailable state so the user has a
        # one-click "Build proxy" retry surface. Without this they'd be
        # stranded — waveform clicks gated, no playback path, no UI to
        # restart the build (Phase 2.1 HUMAN-UAT follow-up).
        self._show_audio_proxy_unavailable(
            heading="Audio proxy not built",
            body_suffix="click Build to enable playback",
        )
        # NO audio_proxy_complete emission — success-only seam.
        self._update_cache_size_footer()

    # ============================================================ Phase 11 (R-3)
    # Lazy / cancellable / cache-reusing spectral build. Structural mirror of
    # the _spawn_audio_proxy_worker family above (cache-HIT fast path, generation
    # token + runnable-identity double-guard, separate _mw_spectral_conn_* token
    # namespace, reuse of the audio-proxy progress banner). Opening a file does
    # NOT spawn a spectral worker — this runs ONLY from the WaveformView's
    # spectral_build_requested signal (REQ-3 a, lazy).
    def _rebuild_spectral_cache(self) -> None:
        """View → Rebuild spectrogram: delete + recompute the spectral cache.

        quick-260629. The spectrogram is cached on disk under
        ``<cache_root>/spectra/<key>/{mel,centroid,bands}.dat`` (``key`` =
        ``cache_key(source)``). This deletes that entry, clears the view's
        stashed arrays, and re-spawns the background build so a stale or corrupt
        cache can be regenerated. A no-op when no file is open. Cancels any
        in-flight build first so the fresh one is the only writer.
        """
        p = self._current_source_path
        if p is None:
            return
        # Stop any build already running so it cannot race the rebuild.
        self._cancel_spectral_build()
        # Delete the three cached siblings so the next build is a true rebuild
        # (the cache-HIT fast path in _spawn_spectral_worker will now MISS).
        cache_root = default_cache_root()
        key = cache_key(p)
        for name in ("mel", "centroid", "bands"):
            try:
                spectral_cache.spectral_path(cache_root, key, name).unlink()
            except (FileNotFoundError, OSError, ValueError):
                pass
        # Drop the view's stashed arrays so a stale surface isn't shown while
        # the rebuild runs (the Classic silhouette stays visible meanwhile).
        self._waveform_view.set_spectral_data(
            mel=None, centroid=None, bands=None, header=None
        )
        # Re-trigger the build. The cache is now empty → guaranteed MISS →
        # background worker; on completion set_spectral_data re-renders.
        self._spawn_spectral_worker(None)

    def _spawn_spectral_worker(self, mode: object) -> None:
        """Cache-HIT fast path else spawn the background spectral build (R-3).

        ``mode`` is the selected :class:`RenderMode` (payload of
        ``WaveformView.spectral_build_requested``); it is informational here —
        all three spectral lanes (mel/centroid/bands) are built/loaded together
        so any spectral mode shares the same cache entry.

        CACHE-HIT (REQ-3 d): if the three ``.dat`` siblings load cleanly we hand
        them straight to ``WaveformView.set_spectral_data`` and return WITHOUT
        spawning a worker. A corrupt/oversized header (``SpectralHeaderError``,
        T-11-01) or a missing sibling (``FileNotFoundError``/``OSError``) is
        treated as a MISS — discard and rebuild.

        MISS: bump the open-generation token, spawn ``SpectralProxyRunnable`` on
        the global ``QThreadPool`` with the audio-proxy progress banner + cancel,
        and wire finished/error/cancelled into the SEPARATE
        ``_mw_spectral_conn_*`` token namespace with default-arg closures
        capturing ``gen`` + ``runnable`` (mirrors the audio-proxy spawn).
        """
        p = self._current_source_path
        if p is None:
            return  # nothing open — defensive; the view only emits after open.

        cache_root = default_cache_root()
        key = cache_key(p)

        # --- CACHE-HIT fast path FIRST (REQ-3 d). All three siblings must load.
        try:
            mel, header = spectral_cache.load_mel(
                spectral_cache.spectral_path(cache_root, key, "mel")
            )
            centroid, _ = spectral_cache.load_centroid(
                spectral_cache.spectral_path(cache_root, key, "centroid")
            )
            bands, _ = spectral_cache.load_bands(
                spectral_cache.spectral_path(cache_root, key, "bands")
            )
        except (
            spectral_cache.SpectralHeaderError,
            FileNotFoundError,
            OSError,
            ValueError,
        ):
            mel = None  # type: ignore[assignment]
        if mel is not None:
            # HIT — render from disk, no worker. set_spectral_data re-renders
            # the surface in-place when a spectral mode is active.
            self._waveform_view.set_spectral_data(
                mel=mel, centroid=centroid, bands=bands, header=header
            )
            self.spectral_build_complete.emit(str(p))
            return

        # --- CACHE-MISS — spawn a background build (UI stays responsive).
        # Reuse the open-generation token (a fresh _open_file bumps it, so a
        # stale spectral worker for the previous file is dropped by the guard).
        gen = self._open_generation
        runnable = SpectralProxyRunnable(p, cache_root)
        self._current_spectral_runnable = runnable

        # Reuse the audio-proxy progress banner as the spectral build surface.
        self._status_proxy_progress.setText("Building spectrogram: 0%")
        self._status_proxy_progress.show()
        self._spectral_banner_active = True
        self._audio_proxy_banner.configure_building(
            heading="Building spectrogram",
            body=f"{p.name} · computing spectral lanes…",
        )
        self._audio_proxy_banner.position_over_widget(self._waveform_view)
        self._audio_proxy_banner.show()
        self._audio_proxy_banner.raise_()

        # Hook the banner cancel button to THIS spectral runnable. PySide6
        # emits a cosmetic RuntimeWarning on a no-prior-connection disconnect;
        # suppress and continue (mirror of the audio-proxy spawn).
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                self._audio_proxy_banner.cancel_button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._audio_proxy_banner.cancel_button.clicked.connect(runnable.cancel)

        runnable.signals.progress.connect(self._on_spectral_progress)
        # Capture gen + runnable BY VALUE (default-arg trick) so a later
        # _open_file / re-select overwriting state does NOT mutate the closure.
        # SEPARATE _mw_spectral_conn_* namespace so the cancel disconnect never
        # touches the audio-proxy tokens or a qtbot.waitSignal watcher.
        runnable._mw_spectral_conn_finished = runnable.signals.finished.connect(
            lambda obj, g=gen, r=runnable: self._on_spectral_finished(g, r, obj)
        )
        runnable._mw_spectral_conn_error = runnable.signals.error.connect(
            lambda msg, g=gen, r=runnable: self._on_spectral_error(g, r, msg)
        )
        runnable._mw_spectral_conn_cancelled = runnable.signals.cancelled.connect(
            lambda g=gen, r=runnable: self._on_spectral_cancelled(g, r)
        )

        QThreadPool.globalInstance().start(runnable)

    def _cancel_spectral_build(self) -> None:
        """Cancel the in-flight spectral build, if any. Idempotent.

        Targeted disconnect via the stored ``_mw_spectral_conn_*`` tokens BEFORE
        ``cancel()`` so a late terminal signal from the cancelled worker cannot
        fire into the next render state (T-11-10). Then synchronously reset the
        UI + runnable so no partial render is shown (the builder removes any
        partial ``*.dat.tmp`` before raising ``BuildCancelled`` — T-11-03).
        """
        runnable = self._current_spectral_runnable
        if runnable is None:
            return
        for token_name, signal_name in (
            ("_mw_spectral_conn_finished", "finished"),
            ("_mw_spectral_conn_error", "error"),
            ("_mw_spectral_conn_cancelled", "cancelled"),
        ):
            _conn = getattr(runnable, token_name, None)
            if _conn is not None:
                try:
                    getattr(runnable.signals, signal_name).disconnect(_conn)
                except (RuntimeError, TypeError):
                    pass
        runnable.cancel()
        self._current_spectral_runnable = None
        self._hide_spectral_progress()

    def _hide_spectral_progress(self) -> None:
        """Tear down the spectral build progress UI (banner + status text)."""
        self._status_proxy_progress.hide()
        if getattr(self, "_spectral_banner_active", False):
            self._audio_proxy_banner.hide()
            self._spectral_banner_active = False

    def _on_spectral_progress(self, pct: int) -> None:
        """Worker progress (0..100) — update the spectral build progress UI.

        No race guard needed: stale flicker before the terminal signal lands is
        harmless and the banner is hidden by the terminal handlers anyway.
        """
        self._status_proxy_progress.setText(f"Building spectrogram: {int(pct)}%")
        if getattr(self, "_spectral_banner_active", False):
            self._audio_proxy_banner.set_progress(int(pct))

    def _on_spectral_finished(
        self,
        gen: int,
        runnable: SpectralProxyRunnable,
        cache_root_obj: object,
    ) -> None:
        """Worker finished — load the three memmaps + hand them to the view.

        Double-guard (generation token + runnable identity) drops stale signals
        from a previously-cancelled worker (T-11-10). Emits
        ``spectral_build_complete`` LAST so test watchers see a settled state.
        """
        if (
            gen != self._open_generation
            or runnable is not self._current_spectral_runnable
        ):
            return  # stale signal from a previously-cancelled / superseded build
        cache_root = Path(str(cache_root_obj))
        p = self._current_source_path
        if p is None:
            self._current_spectral_runnable = None
            self._hide_spectral_progress()
            return
        key = cache_key(p)
        try:
            mel, header = spectral_cache.load_mel(
                spectral_cache.spectral_path(cache_root, key, "mel")
            )
            centroid, _ = spectral_cache.load_centroid(
                spectral_cache.spectral_path(cache_root, key, "centroid")
            )
            bands, _ = spectral_cache.load_bands(
                spectral_cache.spectral_path(cache_root, key, "bands")
            )
        except (
            spectral_cache.SpectralHeaderError,
            FileNotFoundError,
            OSError,
            ValueError,
        ) as e:
            # The freshly-written cache failed to load — surface as an error
            # rather than a silent no-op so the user knows the render is empty.
            self._current_spectral_runnable = None
            self._hide_spectral_progress()
            QMessageBox.warning(self, "Spectral build failed", str(e))
            return
        self._waveform_view.set_spectral_data(
            mel=mel, centroid=centroid, bands=bands, header=header
        )
        self._current_spectral_runnable = None
        self._hide_spectral_progress()
        self._update_cache_size_footer()
        # Success-only seam — tests synchronise on this.
        self.spectral_build_complete.emit(str(p))

    def _on_spectral_error(
        self,
        gen: int,
        runnable: SpectralProxyRunnable,
        msg: str,
    ) -> None:
        """Worker raised — surface a warning; no partial render. Double-guarded."""
        if (
            gen != self._open_generation
            or runnable is not self._current_spectral_runnable
        ):
            return  # stale signal from a previously-cancelled / superseded build
        self._current_spectral_runnable = None
        self._hide_spectral_progress()
        QMessageBox.warning(self, "Spectral build failed", msg)
        self._update_cache_size_footer()
        # NO spectral_build_complete emission — success-only seam.

    def _on_spectral_cancelled(
        self,
        gen: int,
        runnable: SpectralProxyRunnable,
    ) -> None:
        """Worker cancelled — silently clear state. No partial render. Double-guarded.

        Cancel is voluntary (the user toggled away or quit). The builder removed
        any partial ``*.dat.tmp`` (T-11-03); nothing to report.
        """
        if (
            gen != self._open_generation
            or runnable is not self._current_spectral_runnable
        ):
            return  # stale signal from a previously-cancelled / superseded build
        self._current_spectral_runnable = None
        self._hide_spectral_progress()
        # NO spectral_build_complete emission — success-only seam.

    def _show_audio_proxy_unavailable(
        self, heading: str, body_suffix: str
    ) -> None:
        """Swap the banner to the NOT-BUILT state (cancel / error follow-up).

        Keeps the banner visible after a build cancel or error so the
        user has a single-click retry surface. Without this they end up
        stranded — waveform clicks gated by ``_audio_proxy_overlay_active``,
        no playback path, no UI to restart the build. Wires the banner's
        action button to re-spawn the worker via
        ``self._audio_proxy_retry_args``.

        Hidden by: cancel preamble (next ``_open_file``) and
        ``_close_file``. Replaced by the BUILDING state when the user
        clicks Build.
        """
        if self._audio_proxy_retry_args is None:
            # Nothing to retry against (defensive — should not happen
            # because cancel/error only fire after a spawn). Hide the
            # banner instead of leaving it in a half-state.
            if self._audio_proxy_overlay_active:
                self._audio_proxy_banner.hide()
                self._audio_proxy_overlay_active = False
            return
        p, probe = self._audio_proxy_retry_args
        basename = p.name
        duration_s = float(probe.duration_s)
        self._audio_proxy_banner.configure_unavailable(
            heading=heading,
            body=(
                f"{basename} · {self._fmt_duration(duration_s)} · "
                f"{body_suffix}"
            ),
        )
        # Disconnect prior wiring (cancel-from-building) and rewire to
        # re-spawn. Capture gen from a NEW generation token? No — re-
        # spawning reuses the current open's generation; the cancel
        # preamble bumps generation only on a NEW file open. This means
        # if the user cancels then clicks Build, the new runnable fires
        # signals into the current open's slots — correct.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                self._audio_proxy_banner.cancel_button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._audio_proxy_banner.cancel_button.clicked.connect(
            lambda: self._retry_audio_proxy_build()
        )
        # Banner is still on top — re-position in case the window was
        # resized while the build was running.
        self._audio_proxy_banner.position_over_widget(self._waveform_view)
        self._audio_proxy_banner.show()
        self._audio_proxy_banner.raise_()
        self._audio_proxy_overlay_active = True

    def _retry_audio_proxy_build(self) -> None:
        """Banner's "Build proxy" button slot — re-spawn the worker.

        Reuses the stashed ``(path, probe)`` captured at the original
        spawn time. The generation token is NOT bumped here (no new
        file open); the new worker emits into the current open's slots.
        """
        if self._audio_proxy_retry_args is None:
            return
        p, probe = self._audio_proxy_retry_args
        # _spawn_audio_proxy_worker rewires the cancel button back to
        # the new runnable + flips the banner to BUILDING state.
        self._spawn_audio_proxy_worker(p, probe, self._open_generation)

    def _update_cache_size_footer(self) -> None:
        """D-09 — refresh the right-pinned cache-size status-bar label."""
        size = audio_cache_size_bytes(default_cache_root())
        self._status_cache_size.setText(f"Cache: {size / 1024**3:.2f} GiB")

    def _action_clear_audio_cache_slot(self) -> None:
        """D-08 — manual cache wipe. No automatic eviction in this phase.

        Deletes ``<cache_root>/audio/`` subtree (best-effort —
        ``shutil.rmtree(ignore_errors=True)`` inside ``clear_audio_cache``)
        and refreshes the status-bar footer. A transient ``showMessage``
        reports the bytes freed so the user gets feedback (UI-SPEC
        §Copywriting expects a friendly tone).
        """
        audio_cache_root = default_cache_root()
        freed = clear_audio_cache(audio_cache_root)
        self._update_cache_size_footer()
        self.statusBar().showMessage(
            f"Cleared audio proxy cache: {freed / 1024**3:.2f} GiB freed",
            5000,
        )

    # ------------------------------------------------------ shared render path
    def _render_loaded_proxy(
        self,
        arr,
        header: "proxy_cache.ProxyHeader",
        probe: "audio_file.AudioProbe",
        path: Path,
    ) -> None:
        """Drive WaveformView.render_proxy + status bar update.

        Called from both the cache-HIT path (sync) and the cache-MISS
        path (after the worker's finished signal). NEVER emits
        render_complete here — the caller does, so failure paths never
        emit it.
        """
        self._waveform_view.render_proxy(
            arr,
            sample_rate=header.sample_rate,
            samples_per_pixel=header.samples_per_pixel,
        )
        self._update_status_for_loaded(probe, path)
        self._current_path = path

    # ------------------------------------------------------ playback pipeline
    # Plan 02-05 — spacebar / toolbar / click-to-seek wiring around the
    # PlaybackEngine. The engine itself is Qt-free; this section is the
    # only place that bridges its atomic position counter into the GUI
    # thread via a 30 Hz QTimer poll. The audio callback NEVER touches a
    # widget — see playback.py module docstring.
    def _action_toggle_playback(self) -> None:
        """Spacebar / toolbar play/pause toggle.

        No-op when no file is open OR the audio backend is unavailable.
        When playing, calls ``engine.pause()`` and stops the QTimer. When
        paused, calls ``engine.play(path, start_seconds=position_seconds)``
        — playback always resumes from the engine's authoritative position
        so the user can click-to-seek then press space and have it Just
        Work. :class:`PlaybackError` from ``engine.play`` is caught and
        surfaced via a :class:`QMessageBox.warning` so a transient device
        error doesn't crash the app.

        CR-01 fix — resume position is read from
        ``self._playback_engine.position_seconds`` (the engine's
        authoritative state), NOT from
        ``self._waveform_view.playhead.value()``. The playhead InfiniteLine
        is a render artifact updated by the 30 Hz QTimer poll
        (``_on_playback_tick``) — when the timer is stopped (paused state)
        the line retains the LAST tick's value. After
        ``engine.seek(new_pos)`` runs while paused, the engine's
        ``_start_frame`` advances to ``new_pos`` but the playhead line
        stays at the pre-seek position because no tick polls. Resuming
        from the playhead would replay from the stale pre-seek position;
        resuming from ``engine.position_seconds`` plays from the seek
        target as the user expects.
        """
        if self._current_path is None or not self._playback_engine.is_available:
            return
        # AUD-04 / SC-4 — playback must go through the proxy WAV (when one
        # exists) so MP3 sources don't pay pedalboard's O(n) seek cost on
        # every play. ``_current_playback_path`` is set by WAV-skip,
        # cache-HIT, and ``_on_audio_proxy_finished``; None means no primed
        # playback target yet (proxy still building, or disk-preflight
        # failed) — bail out quietly.
        if self._current_playback_path is None:
            return
        try:
            if self._playback_engine.is_playing:
                self._playback_engine.pause()
                self._playback_timer.stop()
            else:
                # Plan 07-10e — resume the file the engine is currently
                # loaded with. When the user pressed B (or smart-clicked
                # into a keeper region), the engine was switched to the
                # mastered cache. The legacy code unconditionally re-played
                # self._current_playback_path (the source proxy) which
                # silently reverted the user's B-mode preview. Fall back
                # to the source proxy only on a cold start (no file loaded
                # in the engine yet).
                engine_path = self._playback_engine._current_path
                if engine_path is None:
                    engine_path = self._current_playback_path
                # Read the engine's authoritative position, not the
                # playhead InfiniteLine (which is a render artifact that
                # lags during paused seek — see method docstring).
                start = self._playback_engine.position_seconds
                # Defensive clamp — Phase 2.1 HUMAN-UAT bug #2 surfaced
                # this when a proxy was shorter than the waveform claimed
                # (standard WAV 4 GiB truncation, now fixed by RF64).
                # Without the clamp, a click near the right edge of the
                # waveform raises `ValueError: Cannot seek to position N
                # frames, which is beyond end of file`. Stay 10 ms shy of
                # EOF so the producer thread reads at least one block
                # before EOF rather than failing immediately on seek.
                duration = self._playback_engine.duration_seconds
                if duration > 0.0:
                    start = min(start, max(0.0, duration - 0.01))
                self._playback_engine.play(
                    str(engine_path), start_seconds=max(0.0, start)
                )
                self._playback_timer.start()
        except PlaybackError as e:
            QMessageBox.warning(self, "Couldn't start playback", str(e))

    def _on_playback_tick(self) -> None:
        """30 Hz QTimer slot — update every playhead + page the view if Follow.

        Reads the engine's position counter via the lock-protected
        ``position_seconds`` property (cheap; no allocation on the audio
        thread). Updates every playhead InfiniteLine in lockstep so the
        waveform's playhead and each lane's playhead stay visually
        synchronised. If Follow-Playhead is checked, page-flips the view
        when the playhead exits the visible range.

        Also drains any pending Trash skip (03-06c) — when the audio
        thread enters a Trash range, it advances the position counter
        and asks us to re-seek past the range. seek() restarts the
        stream cleanly at trash_end + 50 ms (same path as click-seek).
        """
        skip_to = self._playback_engine.consume_pending_skip()
        if skip_to is not None:
            try:
                self._playback_engine.seek(float(skip_to))
            except Exception:
                pass
        pos = self._playback_engine.position_seconds
        # Plan 07-10c — when the engine is playing the keeper-bounded
        # mastered cache (B-state), engine.position_seconds reports
        # CACHE-timeline (0..keeper_duration). The InfiniteLines live
        # on the SOURCE-timeline waveform plot, so without this
        # translation the playhead jumps to 0..keeper_duration on the
        # source waveform rather than tracking where the audio is
        # actually sounding in the source. Translate by adding
        # keeper.start_sec so the visual playhead stays anchored to
        # the keeper region the user sees on the waveform.
        if (
            hasattr(self, "_ab_toggle")
            and self._ab_toggle.state == "B"
            and self._selected_keeper_id is not None
        ):
            region_widget = self._regions_overlay.get_region(
                self._selected_keeper_id
            )
            if region_widget is not None:
                keeper_start_sec, _ = region_widget.getRegion()
                pos = float(keeper_start_sec) + pos
        # quick-260625 — draw the playhead a touch AHEAD of the true audible
        # position to cancel the GUI/render pipeline lag so the line sits on
        # the waveform feature you are actually hearing (see the constant).
        # Cosmetic only; ``pos`` (the real audible position) is untouched and
        # still drives nothing but this displayed value.
        pos_display = max(0.0, pos + self._playhead_visual_offset_sec)
        for line in self._lane_playheads.values():
            line.setValue(pos_display)
        # quick-260625 — follow the playhead with the "now playing" row tint.
        # ``pos_display`` is SOURCE-time (the B-mode translation above already
        # ran), so containment against region start/end works in every mode;
        # using the displayed value keeps the tint aligned with the line.
        self._set_playing_row_highlight(pos_display)
        # quick-260629 — when continuous playback streams INTO a keeper
        # section, auto-audition it exactly as if the user had clicked the
        # row's ▶ Play (from start) button. Driven off the TRUE audible
        # source-time ``pos`` (not the cosmetic ``pos_display``); fires once
        # per enter transition (see the re-fire guard inside).
        self._maybe_autoplay_keeper_on_enter(pos)
        # Update the centered toolbar timestamp on every tick so the
        # user gets continuous visual feedback for where the playhead is.
        self._update_toolbar_time_label(pos_display)
        if self._tb_follow_playhead.isChecked():
            self._follow_view_to_playhead(pos_display)
        if not self._playback_engine.is_playing:
            self._playback_timer.stop()
            # quick-260625 — playback ended: drop the "now playing" row tint
            # on every row (it can be lit during full-file playback even when
            # no specific keeper was the play source).
            for row in self._keepers_sidebar._rows.values():
                row.set_playing(False)
            # Plan 07-10e / quick-260622-tit — engine just stopped (EOF,
            # error, or stop via another path). Clear the per-row active
            # highlight so no button stays highlighted after playback ends.
            if self._currently_playing_keeper_id is not None:
                self._currently_playing_keeper_id = None
                self._currently_playing_mode = None
                self._refresh_keeper_row_play_icons()

    def _update_toolbar_time_label(self, pos: float) -> None:
        """Format ``pos`` (seconds) and set it on the centered toolbar QLabel.

        Mirrors the waveform's _TimeAxisItem formatter: ``M:SS`` for
        durations under an hour, ``H:MM:SS`` for longer files. The label
        always shows at least ``0:00`` so the toolbar layout doesn't
        jump between zero-width and labeled states.
        """
        if pos < 0:
            pos = 0.0
        total = int(round(pos))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        text = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        try:
            self._tb_time_label.setText(text)
        except RuntimeError:
            # Label deleted during teardown — ignore.
            pass

    def _follow_view_to_playhead(self, pos: float) -> None:
        """Page-flip-on-edge follow behavior — RESEARCH §Pattern 7.

        When the playhead exits the visible range, snap the x-range so the
        playhead is at the LEFT edge plus a 5% margin. The view width is
        preserved (the user's zoom level survives the page-flip). When the
        playhead is still inside the visible range, no change — the view
        is stable.
        """
        vb = self._waveform_view.waveform_plot.getViewBox()
        (xmin, xmax), _ = vb.viewRange()
        width = xmax - xmin
        if width <= 0:
            return
        if pos > xmax or pos < xmin:
            new_min = max(0.0, pos - width * 0.05)
            new_max = new_min + width
            vb.setXRange(new_min, new_max, padding=0)

    def _find_keeper_with_fresh_cache_at(
        self, source_time_sec: float
    ) -> tuple[str, float, float, Path] | tuple[None, None, None, None]:
        """Hit-test: keeper region containing source_time with fresh mastered cache.

        Plan 07-10e — used by smart waveform click. Returns
        ``(region_id, start_sec, end_sec, cache_path)`` for the FIRST keeper
        whose source-timeline span contains ``source_time_sec`` AND whose
        mastered cache file is fresh at the current config_hash. Returns
        ``(None, None, None, None)`` if no such keeper exists.

        Order matches the overlay's iteration order — overlapping keepers
        (rare) resolve to the first declared. Bounds are read from the
        live region widget (``widget.getRegion()``) so edge-drag mutations
        between selection time and click time pick up the latest values.
        """
        if (
            self._current_path is None
            or self._current_playback_path is None
        ):
            return None, None, None, None
        try:
            src_key = proxy_cache.cache_key(self._current_path)
        except Exception:
            return None, None, None, None
        for region_data in self._regions_overlay.regions_data():
            if region_data.state != "keeper":
                continue
            widget = self._regions_overlay.get_region(region_data.id)
            if widget is None:
                continue
            start_sec, end_sec = widget.getRegion()
            if not (float(start_sec) <= source_time_sec <= float(end_sec)):
                continue
            mastering = self._regions_overlay.get_mastering(region_data.id)
            if mastering is None:
                continue
            try:
                chash = config_hash(mastering)
                cache_p = mastered_cache_path(
                    default_cache_root(), src_key, region_data.id, chash
                )
            except Exception:
                continue
            if not is_mastered_cache_fresh(cache_p):
                continue
            return (
                region_data.id,
                float(start_sec),
                float(end_sec),
                cache_p,
            )
        return None, None, None, None

    def _swap_engine_to(self, target_path: Path, target_offset_sec: float) -> None:
        """Engine.play/seek to ``target_path`` at ``target_offset_sec``.

        Plan 07-10e — preserves pause state. If the engine is already
        loaded with ``target_path`` uses ``engine.seek`` (which only
        auto-restarts when the engine was playing). Otherwise calls
        ``engine.play`` then immediately ``pause`` if playback was paused
        — a brief blip is acceptable for a click action that has to
        swap the loaded audio file.
        """
        if not self._playback_engine.is_available:
            return
        target_path_str = str(target_path)
        engine_path = self._playback_engine._current_path
        same_file = engine_path is not None and str(engine_path) == target_path_str
        was_playing = self._playback_engine.is_playing
        try:
            if same_file:
                self._playback_engine.seek(target_offset_sec)
            else:
                self._playback_engine.play(
                    target_path_str, start_seconds=target_offset_sec
                )
                if not was_playing:
                    try:
                        self._playback_engine.pause()
                    except Exception:
                        pass
        except (PlaybackError, ValueError):
            pass

    def _on_seek_requested(self, seconds: float) -> None:
        """WaveformView.seek_requested handler — Plan 07-10e smart click-to-seek.

        Behavior contract (user request):
          * Click inside a keeper region with a fresh mastered cache →
            switch toggle to **B**, select that keeper, play the cache
            file at cache_offset = source_click - keeper.start_sec.
          * Click outside any keeper-with-cache → switch toggle to **A**,
            play the source proxy at the clicked source-time.

        The visual playhead is always positioned at the user's clicked
        source-time (the waveform display is source-timeline; see the
        07-10c sibling fix in :meth:`_on_playback_tick` for the live-
        tick counterpart that translates B-state engine-position back
        to source-time for the same reason).

        Phase 2.1 HUMAN-UAT #3: during audio-proxy build the click would
        be a no-op anyway (``_current_playback_path`` is None), but the
        visual playhead jump would mislead — gate the whole handler so
        the waveform feels cleanly disabled while the inline banner is up.
        """
        if self._current_path is None:
            return
        if self._audio_proxy_overlay_active:
            return
        if self._current_playback_path is None:
            return
        seek_s_source = max(0.0, float(seconds))
        # Reflect the new playhead position in the toolbar timestamp
        # immediately — without this the label only updates while the
        # engine is actively playing, so a paused-then-clicked state
        # looks stale.
        self._update_toolbar_time_label(seek_s_source)

        # Plan 07-10f / quick-260622-tit — a waveform click is not a
        # per-row Play-button gesture, so the per-row active highlight
        # must clear. The engine may still play cache (via the smart
        # hit-test below) but the row highlights stop pointing at any
        # specific row.
        if self._currently_playing_keeper_id is not None:
            self._currently_playing_keeper_id = None
            self._currently_playing_mode = None
            self._refresh_keeper_row_play_icons()

        (
            target_keeper_id,
            keeper_start_sec,
            keeper_end_sec,
            cache_path,
        ) = self._find_keeper_with_fresh_cache_at(seek_s_source)

        if target_keeper_id is not None and cache_path is not None:
            # Click inside a keeper with cache → B mode.
            keeper_duration_sec = max(
                0.0, keeper_end_sec - keeper_start_sec
            )
            cache_offset_sec = max(
                0.0,
                min(
                    seek_s_source - keeper_start_sec,
                    keeper_duration_sec,
                ),
            )
            # Update selection + refresh toggle (tooltip / enabled state).
            self._selected_keeper_id = target_keeper_id
            self._refresh_ab_toggle_enabled_state()
            # Set toggle visual to B without re-firing _on_ab_state_changed
            # — we will swap the engine ourselves with the exact cache
            # offset, instead of letting it re-read engine.position. The
            # _ab_failclosed_in_progress guard suppresses re-entrance.
            self._ab_failclosed_in_progress = True
            try:
                self._ab_toggle.set_state("B")
            finally:
                self._ab_failclosed_in_progress = False
            # Visual playhead at source-time of the click.
            for line in self._lane_playheads.values():
                line.setValue(seek_s_source)
            self._swap_engine_to(cache_path, cache_offset_sec)
        else:
            # Click outside any keeper-with-cache → A mode.
            if self._ab_toggle.state != "A":
                self._ab_failclosed_in_progress = True
                try:
                    self._ab_toggle.set_state("A")
                finally:
                    self._ab_failclosed_in_progress = False
            # Source-proxy duration clamp — Phase 2.1 HUMAN-UAT bug #2.
            # Only apply when engine is already on source (else the
            # engine.duration value reflects the cache, not the source).
            engine_path = self._playback_engine._current_path
            on_source = (
                engine_path is not None
                and str(engine_path) == str(self._current_playback_path)
            )
            if on_source:
                duration = self._playback_engine.duration_seconds
                if duration > 0.0:
                    seek_s_source = min(
                        seek_s_source, max(0.0, duration - 0.01)
                    )
            for line in self._lane_playheads.values():
                line.setValue(seek_s_source)
            self._swap_engine_to(self._current_playback_path, seek_s_source)

    # ---------------------------------------------------------------- export
    # Plan 03-04b — Region export pipeline (EXP-01/03 D-A4-3/4/5).
    #
    # Right-click → "Export this region as MP3…" OR "Export this region as
    # WAV…" fires ``RegionsOverlay.export_requested(region_id, fmt)`` which
    # routes here. The slot computes the destination filename on the GUI
    # thread (cheap — walks cached heatmaps for the dominant trait, scans
    # the output dir for ``_NN`` collision), then spawns an ExportRunnable
    # on the global QThreadPool. The 4-signal contract reuses Phase 2.1's
    # WorkerSignals verbatim (D-16).

    def _export_settings(self) -> QSettings:
        """Return a QSettings handle scoped to the Marmelade app namespace.

        Using an explicit org/app pair keeps the export-folder preference
        isolated from pytest-qt's default-app QSettings (which would
        otherwise leak the preference between unrelated test invocations)
        while still landing under the per-test ``~/.qttest`` sandbox
        established by ``QStandardPaths.setTestModeEnabled(True)`` in the
        conftest fixture.
        """
        return QSettings("Marmelade", "Marmelade")

    def _action_change_export_dir_slot(self) -> None:
        """File → Change default export folder… — pick a new dir via QFileDialog (D-A4-3)."""
        settings = self._export_settings()
        current = settings.value("export_dir", "")
        current = str(current) if current else str(Path.home())
        new_dir = QFileDialog.getExistingDirectory(
            self, "Choose default export folder", current
        )
        if not new_dir:
            return
        settings.setValue("export_dir", new_dir)
        self.statusBar().showMessage(
            f"Default export folder set to {new_dir}", 5000
        )

    def _resolve_export_dir(self) -> Optional[Path]:
        """Return the remembered export folder, or prompt and return.

        Returns ``None`` if the user cancelled the picker. The first export
        sees no remembered folder and triggers the picker; subsequent
        exports skip the picker and write straight to the remembered dir.
        """
        settings = self._export_settings()
        remembered = settings.value("export_dir", "")
        if remembered:
            p = Path(str(remembered))
            if p.is_dir():
                return p
        default = str(Path.home() / "Marmelade Exports")
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose default export folder", default
        )
        if not chosen:
            return None
        settings.setValue("export_dir", chosen)
        return Path(chosen)

    def _on_export_region_requested(
        self, region_id: str, fmt: str
    ) -> None:
        """Right-click → Export this region as MP3/WAV…  (UI-SPEC §Export flow).

        ``fmt`` comes from the context-menu action: ``"mp3"`` or ``"wav"``.
        Both paths are first-class per CONTEXT D-A4-4 LOCKED dual-format.
        Validates the fmt arg defensively (two-layer defense — the export
        builder ALSO validates, T-03-04b-05).
        """
        if fmt not in ("mp3", "wav"):
            return  # defensive — should never happen
        if self._current_playback_path is None:
            self.statusBar().showMessage(
                "Wait for audio proxy build to complete.", 5000
            )
            return
        if self._current_cache_key is None or self._current_source_path is None:
            return
        region_widget = self._regions_overlay.get_region(region_id)
        if region_widget is None:
            return
        start_sec, end_sec = region_widget.getRegion()
        region_start = float(start_sec)
        region_end = float(end_sec)

        output_dir = self._resolve_export_dir()
        if output_dir is None:
            return  # user cancelled picker
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # WR-05 — surface mkdir failures (permission denied on a system
            # path like /etc, no-space, read-only filesystem) as a friendly
            # QMessageBox rather than letting the export worker eventually
            # fail with an unhandled OSError.
            QMessageBox.warning(
                self,
                "Export folder unavailable",
                f"Cannot create export folder:\n{output_dir}\n\n{exc}",
            )
            return
        # WR-05 — writability sandbox check. The user picks the folder via
        # QFileDialog so a malicious target is gated by a deliberate user
        # gesture, but an accidental pick (e.g., the user's read-only
        # archive mount, or a system path that mkdir happened to satisfy
        # because it already existed) should fail FAST with a friendlier
        # error than the export worker's eventual os.replace exception.
        if not os.access(str(output_dir), os.W_OK):
            QMessageBox.warning(
                self,
                "Export folder not writable",
                f"Cannot write to the chosen export folder:\n{output_dir}\n\n"
                "Pick a different folder via File > Set default export folder.",
            )
            return

        # Compute dominant trait on the GUI thread — cheap walk of cached
        # heatmaps; the slice arithmetic is O(samples_in_region) which is
        # tiny for any realistic region.
        cache_root = default_cache_root()
        trait = naming_resolver.dominant_trait_for_region(
            cache_root, self._current_cache_key, region_start, region_end,
        )

        # Resolve filename with collision suffix — ext comes from fmt.
        dst_path = naming_resolver.resolve_filename(
            source_path=self._current_source_path,
            region_start_sec=region_start,
            trait=trait,
            ext=fmt,
            output_dir=output_dir,
        )

        # Compute frame range + fade frames (D-A4-5: linear).
        # quick-260626-o9y — the per-side fade duration is now CONFIG-DRIVEN
        # (was a forced min(2.0, region_duration/2.0)). Read the keeper's fade
        # enabled flag + duration via fade_params; disabled → 0.0, else clamp
        # to half the region.
        # WR-04: read sample_rate via the public accessor so the GUI
        # tier respects the engine's encapsulation boundary; the
        # accessor reads under the engine's _lock so a misread between
        # prime() and play() is impossible.
        engine_sr = self._playback_engine.sample_rate
        sr = engine_sr if engine_sr > 0 else 44100
        start_frame = int(region_start * sr)
        end_frame = int(region_end * sr)
        region_duration = region_end - region_start
        fade_enabled, fade_dur = fade_params(
            self._regions_overlay.get_mastering(region_id)
        )
        fade_sec = min(fade_dur, region_duration / 2.0) if fade_enabled else 0.0
        fade_frames = int(fade_sec * sr)

        # quick-260621-gfq — export no longer normalizes (normalize is the
        # mastering chain's final stage). This single-export path streams the
        # source proxy verbatim; normalized bytes come from mastering.
        self._spawn_export_worker(
            proxy_path=Path(self._current_playback_path),
            dst_path=dst_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fade_frames=fade_frames,
            fmt=fmt,
            sample_rate=sr,
        )

    def _spawn_export_worker(
        self,
        proxy_path: Path,
        dst_path: Path,
        start_frame: int,
        end_frame: int,
        fade_frames: int,
        fmt: str,
        sample_rate: int,
        *,
        source_path: Path | None = None,
    ) -> None:
        """Construct + start ExportRunnable. Mirrors AudioProxyRunnable spawn.

        New export cancels any in-flight one — same spawn-new-cancels-old
        discipline as the audio proxy and peak builder runnables. Targeted
        ``_mw_export_conn_*`` connection tokens let the cancel preamble
        disconnect ONLY our wiring (test watchers stay attached).

        Phase 7 Plan 07-06 — ``source_path`` keyword-only override (D-20).
        When provided, the export reads from ``source_path`` instead of
        ``proxy_path``. Phase C of the Master & Export All flow uses
        this to route mastered keepers through their mastered-cache
        WAV file.
        """
        if self._current_export_runnable is not None:
            old = self._current_export_runnable
            for token_name, signal_name in (
                ("_mw_export_conn_progress", "progress"),
                ("_mw_export_conn_finished", "finished"),
                ("_mw_export_conn_error", "error"),
                ("_mw_export_conn_cancelled", "cancelled"),
            ):
                _conn = getattr(old, token_name, None)
                if _conn is not None:
                    try:
                        getattr(old.signals, signal_name).disconnect(_conn)
                    except (RuntimeError, TypeError):
                        pass
            try:
                old.cancel()
            except Exception:
                pass
            self._current_export_runnable = None

        # quick-260621-gfq — export never normalizes. Normalize is the
        # mastering chain's final stage; the mastered-cache export path
        # (source_path) carries already-normalized bytes, and raw export
        # streams the source verbatim.
        runnable = ExportRunnable(
            proxy_path=proxy_path,
            dst_path=dst_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fade_frames=fade_frames,
            fmt=fmt,
            sample_rate=sample_rate,
            source_path=source_path,
        )
        runnable._mw_export_conn_progress = runnable.signals.progress.connect(
            self._on_export_progress
        )
        runnable._mw_export_conn_finished = runnable.signals.finished.connect(
            self._on_export_finished
        )
        runnable._mw_export_conn_error = runnable.signals.error.connect(
            self._on_export_error
        )
        runnable._mw_export_conn_cancelled = runnable.signals.cancelled.connect(
            self._on_export_cancelled
        )
        self._current_export_runnable = runnable
        # Show progress widget pair.
        self._status_export.setText("Exporting clip: 0%")
        self._status_export_cancel.setVisible(True)
        QThreadPool.globalInstance().start(runnable)

    def _on_export_progress(self, pct: int) -> None:
        self._status_export.setText(f"Exporting clip: {int(pct)}%")

    def _on_export_finished(self, dst_path: str) -> None:
        filename = Path(dst_path).name
        self._status_export.setText(f"Exported {filename}")
        self._status_export_cancel.setVisible(False)
        self._clear_export_status_after_delay(5000)
        self._current_export_runnable = None
        self.export_complete.emit(dst_path)

    def _on_export_error(self, msg: str) -> None:
        short = msg.splitlines()[0] if msg else "unknown error"
        self._status_export.setText(f"Export failed: {short}")
        self._status_export.setToolTip(msg)
        self._status_export_cancel.setVisible(False)
        self._clear_export_status_after_delay(10000)
        self._current_export_runnable = None
        # CR-01 (Phase 7 review) — a batch export must still advance on
        # per-keeper export failure. ``export_complete`` (success-only)
        # is the loop pump, so an error would otherwise wedge the batch
        # indefinitely. Count this keeper as failed, advance the loop.
        # Quick-260615-l4y — guarded by the _export_all_in_flight sentinel.
        if self._export_all_in_flight:
            QTimer.singleShot(0, self._kick_next_export_all)

    def _on_export_cancelled(self) -> None:
        self._status_export.setText("Export cancelled")
        self._status_export_cancel.setVisible(False)
        self._clear_export_status_after_delay(3000)
        self._current_export_runnable = None
        # CR-01 (Phase 7 review) — During a batch export, the user clicking
        # the status-bar × on a per-keeper export is treated as terminal
        # for the whole batch (we don't have a way to ask "skip this one
        # or abort all" from a single click). Drain to completion.
        # Quick-260615-l4y — guarded by the _export_all_in_flight sentinel.
        if self._export_all_in_flight:
            QTimer.singleShot(0, self._on_export_all_complete)

    def _on_export_cancel_clicked(self) -> None:
        """Status-bar × button slot — cancel the in-flight export."""
        if self._current_export_runnable is not None:
            try:
                self._current_export_runnable.cancel()
            except Exception:
                pass

    def _clear_export_status_after_delay(self, msec: int) -> None:
        """Auto-clear the status-bar export label after ``msec`` (UI-SPEC).

        The QTimer.singleShot lambda guards against the case where the
        MainWindow (and therefore the status-bar label) has been torn
        down before the timer fires — common during pytest teardown when
        multiple windows are constructed back-to-back. The bare attribute
        access ``self._status_export.setText("")`` would raise libshiboken
        ``Internal C++ object already deleted`` and surface as a noisy
        captured exception in the next test's output.
        """
        QTimer.singleShot(msec, self._clear_export_status_safe)

    def _clear_export_status_safe(self) -> None:
        """Clear the export status label, swallowing teardown-race RuntimeError."""
        try:
            self._status_export.setText("")
            self._status_export.setToolTip("")
        except RuntimeError:
            # Window/label has been torn down — drop silently.
            pass

    # ---------------------------------------------------------------- close
    def _close_file(self) -> None:
        """File > Close: clear the view, reset title, clear status bar.

        CR-02: disconnect terminal signals from the in-flight runnable
        BEFORE dropping the reference, so a late-arriving cancelled/finished/
        error from the cancelled worker cannot fire into the next file's
        GUI state.
        """
        # If a build is in flight, cancel it.
        if self._current_runnable is not None:
            old = self._current_runnable
            # CR-02: disconnect ONLY our own slot wiring BEFORE cancel + drop
            # reference, so a late-arriving terminal signal from the cancelled
            # worker can no longer reach GUI slots and contaminate the next
            # file's state. We use the QMetaObject.Connection tokens stored
            # at connect time so external listeners (e.g. a test's qtbot
            # watcher on `cancelled`) remain attached — a bare
            # `signals.<x>.disconnect()` would nuke every connection.
            _conn = getattr(old, "_mw_conn_finished", None)
            if _conn is not None:
                try:
                    old.signals.finished.disconnect(_conn)
                except (RuntimeError, TypeError):
                    pass
            _conn = getattr(old, "_mw_conn_error", None)
            if _conn is not None:
                try:
                    old.signals.error.disconnect(_conn)
                except (RuntimeError, TypeError):
                    pass
            _conn = getattr(old, "_mw_conn_cancelled", None)
            if _conn is not None:
                try:
                    old.signals.cancelled.disconnect(_conn)
                except (RuntimeError, TypeError):
                    pass
            old.cancel()
            self._current_runnable = None
        self._overlay.hide()
        self._waveform_view.clear()
        self.setWindowTitle("Marmelade")
        self._status_left.setText("")
        self._status_zoom.setText("")
        self._current_path = None
        self._current_playback_path = None
        # Plan 03-01 — clear the regions overlay and forget the sidecar
        # path so a subsequent open re-loads from disk and a stray
        # sigRegionChangeFinished after teardown does not save.
        self._regions_overlay.clear()
        self._current_sidecar_path = None
        # Plan 03-02 — close-file also empties the Keepers dock.
        if hasattr(self, "_keepers_sidebar"):
            self._keepers_sidebar.clear()
        # quick-260701-jc5 — close-file clears markers + disables [+].
        if hasattr(self, "_markers_overlay"):
            self._markers_overlay.clear()
        if hasattr(self, "_markers_sidebar"):
            self._markers_sidebar.clear()
            self._markers_sidebar.set_add_enabled(False)
        self._current_markers = []
        # Tear down the banner if it was lingering in unavailable state
        # (cancelled or errored build the user never retried).
        if self._audio_proxy_overlay_active:
            self._audio_proxy_banner.hide()
            self._audio_proxy_overlay_active = False
        self._audio_proxy_retry_args = None
        for action in (
            self._action_close,
            self._action_zoom_in,
            self._action_zoom_out,
            self._action_zoom_fit,
            self._action_rebuild_spectral,
            self._tb_zoom_fit,
            self._tb_zoom_in,
            self._tb_zoom_out,
        ):
            action.setEnabled(False)

    # -------------------------------------------------- exit / close-event hook
    def closeEvent(self, event: QCloseEvent) -> None:
        """D-12 — cooperative cancel any in-flight audio-proxy build (800 ms).

        Pattern is verbatim from ``02.1-RESEARCH.md`` §Open-Q-1 + PATTERNS.md
        §5 with ``super().closeEvent(event)`` appended:

        * If a proxy build is in flight we ``.cancel()`` the runnable, then
          pump a local ``QEventLoop`` until either a terminal signal
          (``cancelled`` / ``finished`` / ``error``) fires OR a
          ``QDeadlineTimer(800)`` expires. The deadline covers the
          worst-case in-flight ``BLOCK_SAMPLES`` decode (~100 ms at
          typical MP3 throughput) with ~8x headroom, while keeping
          user-perceived exit snappy.
        * Beyond the deadline OS process death halts the worker; a
          leftover ``.tmp`` is acceptable — the File > Clear audio proxy
          cache menu (D-08) covers manual cleanup, and the startup
          ``.tmp`` GC pass is deferred per RESEARCH §Open-Q-4 to a
          future 2.1.1 polish phase.
        * ``ExcludeUserInputEvents`` mirrors Phase 1's re-entrancy
          discipline (``_action_open_file`` lines 412-439) — the user
          cannot accidentally retrigger Open during the cancel wait.
        * ``self._current_proxy_runnable = None`` is cleared inside this
          hook so any terminal signal slipping past the deadline is
          silently dropped by the slot-prologue identity guards (see
          ``_on_audio_proxy_finished`` lines 1027-1031 et al.).

        This is the structural fix for the libshiboken-UAF concern
        documented in Plan 02.1-04 SUMMARY § "Deferred Issues" — by
        cancelling in-flight workers cooperatively before the QMainWindow
        tear-down, the workers no longer outlive their parent QObjects
        and the C++/Python lifetime invariant is preserved.
        """
        # CR-04 (Phase 7 review) — Phase 7 added three worker classes
        # (MasteringRunnable, ExportRunnable, PeakBuilderRunnable) that
        # can be in flight at window-close time. None of them used to
        # be cancelled here, so they would outlive the QMainWindow and
        # fire terminal signals into a torn-down Python wrapper (most
        # row-update slots are wrapped in try/except RuntimeError, but
        # not all — e.g. self.mastering_complete.emit on a dead QObject).
        # Mirror the Phase 2.1 audio-proxy drain pattern: cooperatively
        # cancel each class, then bound the wait at QThreadPool.
        for runnable in list(self._mastering_runnables.values()):
            try:
                runnable.cancel()
            except Exception:
                pass
        self._mastering_runnables.clear()
        if self._current_export_runnable is not None:
            try:
                self._current_export_runnable.cancel()
            except Exception:
                pass
            self._current_export_runnable = None
        if self._current_runnable is not None:
            try:
                self._current_runnable.cancel()
            except Exception:
                pass
            self._current_runnable = None
        if self._current_proxy_runnable is not None:
            old = self._current_proxy_runnable
            old.cancel()
            deadline = QDeadlineTimer(800)
            loop = QEventLoop()
            latch = {"done": False}
            # WR-07 — store the (signal, Connection) tokens so the
            # ``finally:`` block can disconnect them. Without this, the
            # lambdas hold strong references to ``latch`` and ``loop``
            # past closeEvent return — preventing GC of the QEventLoop
            # and producing "signal emitted on dead object" warnings if
            # a terminal signal arrives after the deadline fires.
            conns: list[tuple[Signal, object]] = []
            try:
                for sig in (
                    old.signals.cancelled,
                    old.signals.finished,
                    old.signals.error,
                ):
                    # Tuple form so both side effects fire from one lambda.
                    token = sig.connect(
                        lambda *_: (latch.update(done=True), loop.quit())
                    )
                    conns.append((sig, token))
                while not latch["done"] and not deadline.hasExpired():
                    loop.processEvents(
                        QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents,
                        max(1, int(deadline.remainingTime())),
                    )
            finally:
                # Disconnect every lambda we connected. Connection tokens
                # may have been auto-invalidated if the signal source's
                # underlying QObject was destroyed concurrently — guard
                # against (RuntimeError, TypeError) to keep the close
                # path robust under teardown races.
                for sig, token in conns:
                    try:
                        sig.disconnect(token)
                    except (RuntimeError, TypeError):
                        pass
            # Whether the latch fired OR the deadline expired, accept
            # the close: process death halts any straggler worker
            # thread, and the slot-prologue guards drop late signals.
            self._current_proxy_runnable = None
        # CR-04 (Phase 7 review) — give the global QThreadPool a bounded
        # window to honor the cooperative cancels we just issued before
        # C++ tear-down. Cancellation is cooperative (matchering mid-call
        # is the worst case but the user's already accepted a longer
        # wait by closing during a master-all). 2 s is enough to drain a
        # typical pedalboard pass without making the close feel hung;
        # leftover .tmp / matchering temp_dir stragglers are handled by
        # existing cleanup paths.
        QThreadPool.globalInstance().waitForDone(2000)
        # quick-260626-kw — persist the current Keepers dock width so it is
        # restored on next launch (see _build_right_dock).
        keepers_dock = getattr(self, "_keepers_dock", None)
        if keepers_dock is not None:
            QSettings().setValue(
                _KEEPERS_DOCK_WIDTH_KEY, int(keepers_dock.width())
            )
        # quick-260626-kw2 — persist full window geometry (incl. maximized
        # state) so the next launch reopens the way the user left it.
        QSettings().setValue(_WINDOW_GEOMETRY_KEY, self.saveGeometry())
        super().closeEvent(event)

    # ------------------------------------------------------------ resize hook
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Reposition floating overlays when visible."""
        super().resizeEvent(event)
        if self._overlay.isVisible():
            self._overlay.resize_to_parent()
        if self._audio_proxy_banner.isVisible():
            self._audio_proxy_banner.position_over_widget(self._waveform_view)
