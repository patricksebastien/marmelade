"""pytest configuration — must be the FIRST executable thing imported by pytest.

The QT_QPA_PLATFORM environment variable MUST be set before any Qt module is
imported (RESEARCH Pitfall #6). We use ``os.environ.setdefault`` as the first
line of this file so that:

* a developer who forgets to ``export QT_QPA_PLATFORM=offscreen`` still gets
  headless mode automatically, and
* CI runs that DO set the env var keep whatever value they provided.
"""

from __future__ import annotations

import os

# Must run before any Qt import. Do not move this.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication


# Phase 7 — autouse QSettings sanitizer for the mastering integration test
# modules. Mirrors the Phase 6 ``_clear_qsettings_heatmaps`` discipline
# (LEARNINGS lesson: explicit ``("Marmelade","Marmelade")`` org/app
# pair bypasses bare ``QSettings()`` monkeypatch — the worker uses the
# explicit pair and would otherwise see stale state from a previous test).
#
# Scoped by ``request.node.fspath`` glob so unit tests (which never write
# to ``mastering/...``) are not impacted. The list is curated narrowly to
# avoid touching unrelated tests.
_MASTERING_TEST_MODULE_GLOBS = (
    "test_mastering*",
    "test_master_export*",
    "test_session_chain*",
    "test_ab_*",
    "test_divergence_badge*",
    "test_matchering_*",
)


@pytest.fixture(autouse=True)
def _clear_qsettings_mastering(request: pytest.FixtureRequest):
    """Wipe the ``mastering/`` QSettings sub-tree before and after each test.

    Only fires for integration test modules matching the curated globs
    in :data:`_MASTERING_TEST_MODULE_GLOBS`. Other tests pay zero cost.

    Uses the explicit ``("Marmelade","Marmelade")`` org/app pair —
    matches the worker's :func:`load_session_chain_snapshot` reader so
    the worker cannot pick up a stale value left over from a previous
    test in the same pytest invocation. See Phase 6 LEARNINGS lesson on
    bare-QSettings monkeypatch pitfalls.
    """
    fspath = str(getattr(request.node, "fspath", ""))
    if not any(g.strip("*") in fspath for g in _MASTERING_TEST_MODULE_GLOBS):
        # Faster: glob-like substring match is sufficient for our test
        # module names (no overlap risk because the globs are unique).
        yield
        return
    s = QSettings("Marmelade", "Marmelade")
    s.remove("mastering")
    s.sync()
    try:
        yield
    finally:
        s.remove("mastering")
        s.sync()


@pytest.fixture(scope="session")
def qapp_cls():
    """Return the QApplication class for pytest-qt to instantiate."""
    return QApplication


@pytest.fixture(scope="session")
def qapp_args() -> list[str]:
    """Args passed to QApplication by pytest-qt's `qapp` fixture."""
    return ["--platform", "offscreen"]


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the Marmelade cache root to ``tmp_path / 'cache'``.

    Two mechanisms are layered (closes REVIEW.md CR-06):

    (1) ``monkeypatch.setattr`` pins
        :func:`marmelade.paths.default_cache_root` to ``tmp_path / 'cache'``
        so OUR app's cache writes land in the per-test directory regardless of
        what ``QStandardPaths`` resolves. This is the load-bearing mechanism —
        without it the helper would resolve to
        ``~/.qttest/cache/Marmelade-test`` (the Qt test-mode prefix on
        Linux) which persists across pytest runs and produces false cache
        HITs in integration tests.

    (2) ``QStandardPaths.setTestModeEnabled(True)`` is also set so any
        non-``default_cache_root`` Qt utility that consults
        ``QStandardPaths`` sees the Qt test-mode sandbox instead of the
        user's real paths during the test body.

    Teardown:
        After the test body returns, ``QStandardPaths.setTestModeEnabled(False)``
        restores the process-global toggle so subsequent tests in the same
        pytest invocation see the user's normal paths unless they request
        ``tmp_cache_dir`` themselves. The ``monkeypatch.setattr`` on the org
        name, app name, and ``default_cache_root`` is unwound by pytest's
        own monkeypatch teardown. Closes REVIEW.md CR-06.
    """
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # CR-06 fix part 1: pin the helper to the per-test directory. String
    # form (not object form) so the patch survives module reimport and
    # matches the existing QCoreApplication monkeypatch style below.
    #
    # We must patch the source-module attribute AND every importer that
    # uses ``from marmelade.paths import default_cache_root`` —
    # because those importers captured a direct reference to the original
    # function object at import time, and patching the source module's
    # attribute does NOT rebind names in already-imported namespaces. See
    # plan 01-06 SUMMARY (deviation Rule 1) for why the originally
    # specified fix was incomplete for this codebase.
    fake = lambda: cache_root  # noqa: E731 — tiny lambda is clearer than def here
    _patch_targets = (
        "marmelade.paths.default_cache_root",  # source of truth
        "marmelade.ui.main_window.default_cache_root",  # production caller
        # Test-module bindings — only patched if the module is already
        # imported by pytest at fixture-setup time. ``monkeypatch.setattr``
        # with ``raising=False`` is a no-op on a not-yet-imported module,
        # which is the safe default for tests that never use the helper.
    )
    for target in _patch_targets:
        monkeypatch.setattr(target, fake)
    # The two test modules below import default_cache_root at module level
    # and consume it via that local binding — patch defensively so the
    # fixture is correct regardless of test selection order.
    for test_target in (
        "tests.perf.test_render_frame_budget.default_cache_root",
        "tests.unit.test_conftest_tmp_cache_dir.default_cache_root",
        "tests.unit.test_paths.default_cache_root",
        # quick-260701-muv — the Plan 02-02 heatmap-cache + energy-budget test
        # modules were deleted with the AI/DSP heatmap backend, so their
        # conftest monkeypatch string-targets are removed here (a string
        # target forces an import of the module, which no longer exists;
        # ``raising=False`` only suppresses a missing attribute, not a
        # missing module).
        # quick-260621-dt4 — the Plan 02-03/02-04 heatmap lane/cache/toggle/
        # recompute integration test modules were deleted with the retired
        # heatmap panel + pipeline, so their conftest patch entries are gone.
        # Plan 02.1-04 — audio-proxy MainWindow integration test modules
        # that import default_cache_root at module level (RESEARCH §Pitfall
        # #10 — `monkeypatch.setattr` does NOT rebind from-imports, so each
        # importer needs its own patch). `raising=False` so a partial test
        # selection that does not collect a particular module is fine.
        # (test_audio_proxy_paint_budget is added by Plan 05.)
        "tests.integration.test_main_window_audio_proxy_mp3_open.default_cache_root",
        "tests.integration.test_main_window_audio_proxy_wav_skip.default_cache_root",
        "tests.integration.test_main_window_audio_proxy_cache_hit.default_cache_root",
        "tests.integration.test_main_window_audio_proxy_cancel_restart.default_cache_root",
        "tests.integration.test_main_window_audio_proxy_play_button_disabled.default_cache_root",
        "tests.integration.test_main_window_audio_proxy_status_bar.default_cache_root",
        "tests.integration.test_main_window_audio_proxy_disk_pressure.default_cache_root",
        "tests.integration.test_main_window_clear_audio_cache.default_cache_root",
        # Plan 02.1-05 — closeEvent exit-during-build pin + paint-budget perf
        # gate import ``default_cache_root`` at module level. Same rebinding
        # discipline as Plan 02.1-04 above (RESEARCH §Pitfall #10).
        "tests.integration.test_main_window_audio_proxy_exit_during_build.default_cache_root",
        "tests.perf.test_audio_proxy_paint_budget.default_cache_root",
        # Phase 2.1 HUMAN-UAT bug #1 fix — regression pin asserting
        # ``engine.play()`` opens the proxy WAV, not the source MP3, after
        # the proxy completes. See test module docstring.
        "tests.integration.test_main_window_audio_proxy_playback_uses_proxy.default_cache_root",
        # Phase 2.1 HUMAN-UAT bug #2 fix — defensive seek-bounds clamp
        # so a click past engine.duration_seconds can't crash play().
        "tests.integration.test_main_window_audio_proxy_seek_bounds.default_cache_root",
        # Phase 2.1 HUMAN-UAT request #3 — audio-proxy modal overlay UX.
        "tests.integration.test_main_window_audio_proxy_overlay.default_cache_root",
        # Plan 03-01 — sidecar persistence test modules that import
        # default_cache_root at module level (same Pitfall #10 discipline).
        "tests.unit.audio.test_sidecar_cache_io.default_cache_root",
        "tests.unit.audio.test_sidecar_cache_quarantine.default_cache_root",
        "tests.integration.test_sidecar_persistence_roundtrip.default_cache_root",
        # Plan 03-02 — Wave 2 region UX tests. Same Pitfall #10 discipline
        # (`raising=False` so unselected modules don't fail at fixture
        # setup). Overlay-only tests don't need a full MainWindow but the
        # patch is cheap and keeps the discipline uniform.
        "tests.integration.test_regions_overlay_hover_target_delete.default_cache_root",
        "tests.integration.test_regions_overlay_context_menu.default_cache_root",
        "tests.integration.test_regions_overlay_resize.default_cache_root",
        "tests.integration.test_keepers_sidebar.default_cache_root",
        # Plan 03-03 — Wave 3 Trash playback skip test modules. Same Pitfall
        # #10 discipline: each module imports default_cache_root at module
        # level for consistency with the rest of the suite even when the
        # algorithm under test does not write to the cache.
        # (quick-260621-dt4 deleted test_trash_heatmap_mask with the panel.)
        "tests.unit.test_trash_keeper_subtract.default_cache_root",
        "tests.integration.test_trash_playback_skip.default_cache_root",
        # Plan 03-04a — Wave 4 export-pipeline foundation: naming_resolver
        # (Qt-free filename + dominant-trait lookup) imports
        # ``default_cache_root`` at module level.
        # (quick-260701-muv removed the test_energy_dominant_trait entry with
        # the deleted heatmap backend.)
        "tests.unit.audio.test_naming_resolver.default_cache_root",
        # Plan 03-04b — Wave 5 export pipeline: export_builder + ExportRunnable
        # tests + MainWindow integration tests + perf gate for the region overlay.
        "tests.unit.audio.test_export_fade_curve.default_cache_root",
        "tests.unit.audio.test_export_builder.default_cache_root",
        "tests.integration.test_export_export_runnable.default_cache_root",
        "tests.perf.test_regions_overlay_cpu_budget.default_cache_root",
        # Plan 03-07 — Wave 7 gap-closure: redundant Export entry point
        # (Edit menu submenu). The test module imports default_cache_root at
        # module level for the tmp_cache_dir fixture; same Pitfall #10
        # discipline. (The KeepersSidebar per-row MP3/WAV buttons were
        # removed in quick-260625 — per-keeper export now lives only in the
        # right-click + Edit menus; clip export via "Export All Keepers".)
        "tests.integration.test_main_window_export_edit_menu.default_cache_root",
        # quick-260701-muv — the Plan 05-01 BPM and Plan 05-02 Harmonic unit +
        # perf heatmap test modules were deleted with the AI/DSP heatmap
        # backend, so their conftest monkeypatch string-targets are removed
        # here (a string target forces an import of the now-missing module;
        # ``raising=False`` only suppresses a missing attribute).
        # quick-260621-dt4 — the Plan 05-03 rhythm/harmonic-toggle and the
        # Phase 6 gear-Apply / qsettings-roundtrip integration test modules
        # were deleted with the retired heatmap panel + pipeline, so their
        # conftest patch entries are gone.
        # Phase 7 Plan 07-03 — session-chain snapshot integration tests
        # import default_cache_root at module level so the tmp_cache_dir
        # fixture redirects sidecar + mastered-cache writes through the
        # per-test directory (Pitfall #10 discipline).
        "tests.integration.test_session_chain_snapshot.default_cache_root",
        # Phase 7 Plan 07-04 — A/B preview integration tests import
        # ``default_cache_root`` at module level for the same reason —
        # the tests pre-stage a fake mastered cache file at
        # ``default_cache_root() / 'mastered' / ...`` and the production
        # ``_refresh_ab_toggle_enabled_state`` slot reads via the same
        # function. Both ends must resolve to the same per-test directory.
        "tests.integration.test_ab_toggle_disabled.default_cache_root",
        "tests.integration.test_ab_switch.default_cache_root",
        # Phase 7 Plan 07-06 — Master & Export All integration tests +
        # cache invalidation pin. All read mastered_cache_path via
        # default_cache_root(); the orchestration test runs real
        # MasteringRunnables that write into ``default_cache_root() /
        # 'mastered' / ...``. Pitfall #10 discipline.
        "tests.integration.test_master_export_all.default_cache_root",
        "tests.integration.test_export_uses_mastered_cache.default_cache_root",
        "tests.integration.test_mastering_cache_invalidation.default_cache_root",
        # Phase 8 Plan 08-06 — sidecar youtube_video_id app-restart e2e test
        # module imports default_cache_root at module level (Pitfall #10
        # discipline). The test opens a real audio file via MainWindow,
        # mutates a region's youtube_video_id, saves the sidecar, closes
        # the file, then re-opens on a fresh MainWindow — all on the
        # per-test cache root.
        "tests.integration.test_sidecar_youtube_video_id_roundtrip_e2e.default_cache_root",
    ):
        # raising=False: tests not collected in this invocation may not be
        # importable as attributes; skip without error.
        monkeypatch.setattr(test_target, fake, raising=False)

    # Enable test mode — QStandardPaths returns paths under a private prefix
    # (e.g. ~/.qttest on Linux) instead of the real user dirs. Preserved for
    # any non-`default_cache_root` Qt utility that consults it.
    QStandardPaths.setTestModeEnabled(True)

    # Use a per-test org/app name so QSettings + QStandardPaths land in an
    # isolated namespace. Reverted via monkeypatch teardown.
    monkeypatch.setattr(
        QCoreApplication,
        "organizationName",
        lambda: "Marmelade-test",
    )
    monkeypatch.setattr(
        QCoreApplication,
        "applicationName",
        lambda: "Marmelade-test",
    )

    yield cache_root

    # CR-06 fix part 2: restore the process-global toggle so subsequent
    # tests in the same pytest invocation see the user's real paths
    # unless they too request `tmp_cache_dir`.
    QStandardPaths.setTestModeEnabled(False)
