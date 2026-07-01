"""Phase 7 Plan 07-04 Task 2 (RED) — A/B toolbar toggle disabled-state pins.

Pins the three disable conditions from UI-SPEC §"A/B Preview Toolbar
Toggle — D-13" lines 540-556:

* Disabled when no keeper is selected.
* Disabled when the selected keeper has ``mastering is None``.
* Disabled when the selected keeper has ``mastering`` but no fresh
  mastered cache exists for the keeper's current config_hash.
* Enabled iff keeper selected + mastering set + cache fresh.
* Enabled-state refreshes after a ``mastering_complete`` signal fires
  for the currently-selected keeper.

Selection model — most-recently-row-clicked in Keepers sidebar (per
plan §"Selection tracking" decision). The sidebar emits
``selection_changed(region_id)`` on left-click of the row body; the
MainWindow tracks ``_selected_keeper_id`` and re-runs the enable-state
check.

Phase 7 Plan 07-09 additions:

* Permanent discoverability tooltip set at toolbar construct time and
  restored (not cleared to "") on every re-enable — the original
  Plan 07-04 docstring claim that "labels are self-explanatory" is
  contradicted by the diagnosis (debug/ab-widget-broken-keys-icon-
  tooltip.md): the labels were clipped to invisibility. The permanent
  tooltip is now the canonical discoverability cue.
* When the A or B shortcut fires while the widget is disabled, a
  3-second status-bar diagnostic message explains the prerequisite —
  the bail still happens silently in terms of audio, but the user gets
  visible feedback.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from marmelade.audio import sidecar_cache
from marmelade.audio.mastering.chain import (
    _SESSION_DEFAULTS,
    config_hash,
    load_session_chain_snapshot,
)
from marmelade.audio.mastering_cache import mastered_cache_path
from marmelade.audio.proxy_cache import cache_key
from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch
from marmelade.ui import theme
from marmelade.ui.ab_toggle import ABToggleWidget
from marmelade.ui.main_window import MainWindow


# Verbatim UI-SPEC strings (lines 422-423 + plan-defined) — grep these
# literals to confirm the implementation matches.
_TOOLTIP_NO_MASTERING = (
    "A/B preview needs a mastered Keeper. "
    "Click the gear button on a Keeper row to configure mastering."
)
_TOOLTIP_CACHE_PENDING = (
    "Mastered cache is being rendered. "
    "A/B preview becomes available when the row badge shows Ready."
)
# Plan 07-09 — permanent discoverability tooltip (set at toolbar construct
# time, restored on every re-enable). Em-dash is U+2014; verbatim string
# pinned by grep + test (`grep "Click a keeper row" src/ tests/`).
_TOOLTIP_DISCOVERABILITY = (
    "A/B preview — A = source, B = mastered. "
    "Click a keeper row with a Ready mastered cache to enable."
)
# Plan 07-09 — status-bar diagnostic when A/B shortcut fires while
# widget is disabled. 3-second duration. Verbatim pinned by grep + test.
_STATUSBAR_BAIL_MESSAGE = (
    "A/B preview: click a keeper row with a Ready mastered cache first"
)


def _make_proxy_wav(tmp_path: Path, seconds: float = 1.0) -> Path:
    sr = 44100
    n = int(seconds * sr)
    audio = (np.random.RandomState(0).randn(n, 2) * 0.05).astype("float32")
    p = tmp_path / "proxy.wav"
    sf.write(str(p), audio, sr, subtype="FLOAT", format="WAV")
    return p


def _write_fake_mastered_cache(
    cache_root: Path,
    src_key: str,
    keeper_id: str,
    cfg: dict,
) -> Path:
    """Create a placeholder mastered WAV at the expected cache path."""
    chash = config_hash(cfg)
    dst = mastered_cache_path(cache_root, src_key, keeper_id, chash)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sr = 44100
    sf.write(str(dst), np.zeros((sr, 2), dtype="float32"), sr,
             subtype="PCM_24", format="WAV")
    return dst


def _open_file_and_wait(window: MainWindow, qtbot, src: Path) -> None:
    """Open a file in MainWindow and wait until proxy + sidecar are primed."""
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None
        and window._current_proxy_p is not None,
        timeout=15000,
    )


def _make_keeper_region(rid: str, start: float, end: float, mastering=None) -> Region:
    return Region(
        id=rid,
        start_sec=start,
        end_sec=end,
        state="keeper",
        note="",
        mastering=mastering,
    )


# =========================================================================
# Test 1 — Toolbar exposes ABToggleWidget after Region Select
# =========================================================================
def test_toolbar_includes_ab_toggle_widget_after_region_select(
    qtbot, qapp, tmp_cache_dir: Path
) -> None:
    """The toolbar's 7th visible item is the A/B toggle widget.

    Order per UI-SPEC §"Layout Architecture" line 54:
    [Open][Fit][ZoomIn][ZoomOut][Follow][Region Select][ A | B ].
    """
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    assert hasattr(window, "_ab_toggle"), "MainWindow must expose _ab_toggle"
    assert isinstance(window._ab_toggle, ABToggleWidget)
    # The widget is inserted via toolbar.addWidget — find its QAction
    # via QToolBar.widgetForAction.
    actions = window._toolbar.actions()
    # 6 prior toolbar actions + ABToggleWidget + a spacer gap widget + the
    # time label = 9 (quick-260621-gfq removed the trailing Normalize spinbox
    # + Normalize action). ABToggleWidget is still at index 6.
    assert len(actions) == 9, (
        f"toolbar should have 9 actions (incl. ABToggleWidget); got {len(actions)}"
    )
    ab_action = actions[6]
    assert window._toolbar.widgetForAction(ab_action) is window._ab_toggle


# =========================================================================
# Test 2 — Disabled when no keeper is selected
# =========================================================================
def test_disabled_no_keeper_selected(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """No keeper selected → A/B disabled (no copy)."""
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _open_file_and_wait(window, qtbot, src)

    assert window._selected_keeper_id is None
    assert window._ab_toggle.isEnabled() is False


# =========================================================================
# Test 3 — Disabled when selected keeper has mastering=None
# =========================================================================
def test_disabled_keeper_selected_but_mastering_none(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Keeper selected but ``mastering is None`` — disabled + tooltip.

    The MainWindow normally auto-snapshots on creation (D-04), so to
    reach this state we explicitly null out the mastering field on a
    keeper after creation — this corresponds to a legacy sidecar that
    has not yet been migrated, OR a defensive bail-out case.
    """
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _open_file_and_wait(window, qtbot, src)

    # Create a keeper region directly via the overlay.
    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None
    overlay.set_state(region.id, "keeper")

    # Force mastering=None to simulate the legacy / defensive case.
    overlay.set_mastering(region.id, None)

    # Simulate the user clicking the row to select the keeper.
    window._on_keeper_selection_changed(region.id)
    assert window._selected_keeper_id == region.id

    assert window._ab_toggle.isEnabled() is False
    assert window._ab_toggle.toolTip() == _TOOLTIP_NO_MASTERING


# =========================================================================
# Test 4 — Disabled when mastering set but cache missing
# =========================================================================
def test_disabled_mastering_set_but_cache_missing(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Keeper has mastering=dict but no cache file on disk → disabled."""
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _open_file_and_wait(window, qtbot, src)

    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None
    overlay.set_state(region.id, "keeper")
    # The session snapshot was auto-applied (D-04); explicitly set a
    # custom mastering to ensure the dict is present. Even if the
    # snapshot already populated it, this idempotently confirms.
    custom = copy.deepcopy(_SESSION_DEFAULTS)
    overlay.set_mastering(region.id, custom)

    # No cache file yet — the runnable would write it, but we have not
    # started one (and disabled it with mastering=set ≠ None).
    window._on_keeper_selection_changed(region.id)
    assert window._ab_toggle.isEnabled() is False
    # Cache-pending plan-defined tooltip — verbatim per plan action.
    assert window._ab_toggle.toolTip() == _TOOLTIP_CACHE_PENDING


# =========================================================================
# Test 5 — Enabled when keeper selected + mastering set + cache fresh
# =========================================================================
def test_enabled_when_keeper_selected_with_fresh_cache(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """All three conditions met → A/B enabled, no tooltip."""
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _open_file_and_wait(window, qtbot, src)

    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None
    overlay.set_state(region.id, "keeper")

    cfg = copy.deepcopy(_SESSION_DEFAULTS)
    overlay.set_mastering(region.id, cfg)

    # Materialise a fresh cache file at the expected path.
    src_key = cache_key(window._current_path)
    _write_fake_mastered_cache(default_cache_root(), src_key, region.id, cfg)

    window._on_keeper_selection_changed(region.id)
    assert window._ab_toggle.isEnabled() is True
    # Plan 07-09 — when the widget enables, the permanent discoverability
    # tooltip is restored (NOT cleared). The original Plan 07-04 contract
    # said "labels are self-explanatory" but the diagnosis proved the
    # labels were clipped to invisibility; the permanent tooltip is now
    # the canonical discoverability cue and survives re-enable.
    assert window._ab_toggle.toolTip() == _TOOLTIP_DISCOVERABILITY


# =========================================================================
# Test 6 — Toggle enables after mastering_complete signal fires
# =========================================================================
def test_toggle_enables_after_mastering_complete_signal(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """mastering=set + cache missing initially → disabled. After cache lands +
    the ``mastering_complete`` signal fires, the toggle becomes enabled.
    """
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _open_file_and_wait(window, qtbot, src)

    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None
    overlay.set_state(region.id, "keeper")
    cfg = copy.deepcopy(_SESSION_DEFAULTS)
    overlay.set_mastering(region.id, cfg)

    window._on_keeper_selection_changed(region.id)
    assert window._ab_toggle.isEnabled() is False  # cache missing

    # Drop the cache file in place, then fire the test seam signal.
    src_key = cache_key(window._current_path)
    _write_fake_mastered_cache(default_cache_root(), src_key, region.id, cfg)
    # Emit the existing test-seam to invoke the refresh.
    window.mastering_complete.emit(region.id)

    assert window._ab_toggle.isEnabled() is True


# =========================================================================
# Phase 7 Plan 07-09 — permanent discoverability tooltip + status-bar bail
# =========================================================================

def test_permanent_tooltip_set_at_construct_time(
    qtbot, qapp, tmp_cache_dir: Path
) -> None:
    """Toolbar construction sets a permanent discoverability tooltip.

    Per Plan 07-09: ``_build_toolbar`` assigns ``self._ab_default_tooltip``
    and calls ``self._ab_toggle.setToolTip(self._ab_default_tooltip)``
    BEFORE the toolbar is shown. The tooltip MUST be set even with no
    file loaded — that's the whole point of the discoverability cue
    (the user sees the empty toolbar and hovers to learn what A/B does).
    """
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)
    assert hasattr(window, "_ab_default_tooltip"), (
        "MainWindow must expose _ab_default_tooltip after _build_toolbar"
    )
    assert window._ab_default_tooltip == _TOOLTIP_DISCOVERABILITY, (
        f"Default tooltip drift detected. Expected verbatim "
        f"{_TOOLTIP_DISCOVERABILITY!r}, got {window._ab_default_tooltip!r}. "
        "The em-dash is U+2014 — check the Edit tool preserved it."
    )
    # The widget itself must carry the tooltip at this moment (post-
    # construct, pre-file-open) — disabled-state default branch in
    # _refresh_ab_toggle_enabled_state (`kid is None`) restores the
    # default, so this asserts both the construct-time setToolTip AND
    # the kid-None branch behavior.
    assert window._ab_toggle.toolTip() == _TOOLTIP_DISCOVERABILITY


def test_tooltip_restored_when_widget_enables(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """When the widget enables, the discoverability tooltip is restored.

    Plan 07-09 changes the "all three met" branch of
    ``_refresh_ab_toggle_enabled_state`` from ``setToolTip("")`` to
    ``setToolTip(self._ab_default_tooltip)``. Mirrors the existing
    ``test_enabled_when_keeper_selected_with_fresh_cache`` fixture but
    asserts the tooltip-content contract explicitly with verbatim match.
    """
    theme.apply_theme(QApplication.instance())
    src = _make_proxy_wav(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    _open_file_and_wait(window, qtbot, src)

    overlay = window._regions_overlay
    overlay.start_draft(0.1)
    overlay.update_draft(0.5)
    region = overlay.commit_draft(0.5)
    assert region is not None
    overlay.set_state(region.id, "keeper")

    cfg = copy.deepcopy(_SESSION_DEFAULTS)
    overlay.set_mastering(region.id, cfg)

    src_key = cache_key(window._current_path)
    _write_fake_mastered_cache(default_cache_root(), src_key, region.id, cfg)

    window._on_keeper_selection_changed(region.id)
    assert window._ab_toggle.isEnabled() is True
    # The tooltip MUST be the verbatim discoverability string — NOT
    # empty, NOT one of the disabled-state context tooltips. This pins
    # the Plan 07-09 contract that the permanent tooltip survives the
    # re-enable refresh.
    assert window._ab_toggle.toolTip() == _TOOLTIP_DISCOVERABILITY, (
        f"Expected discoverability tooltip after re-enable; got "
        f"{window._ab_toggle.toolTip()!r}. Likely _refresh_ab_toggle_"
        f"enabled_state still clears with setToolTip('') instead of "
        f"setToolTip(self._ab_default_tooltip) (Plan 07-09)."
    )


def test_shortcut_bail_emits_status_bar_message(
    qtbot, qapp, tmp_cache_dir: Path
) -> None:
    """Disabled-shortcut bail surfaces a 3-second status-bar diagnostic.

    Plan 07-09 changes ``_on_ab_shortcut_pressed`` so that when the
    widget is disabled, the bail path emits a status-bar message
    explaining the prerequisite (instead of being completely silent).
    The audio engine is untouched (no swap occurs); this is purely a
    user-feedback nudge.

    Verifies the verbatim message string via ``statusBar().currentMessage()``.
    """
    theme.apply_theme(QApplication.instance())
    window = MainWindow()
    qtbot.addWidget(window)

    # Widget should be naturally disabled with no file / no selection.
    assert window._ab_toggle.isEnabled() is False

    # Fire the slot directly — bypasses the QShortcut wiring but
    # exercises the same code path the shortcut would.
    window._on_ab_shortcut_pressed("A")

    qtbot.waitUntil(
        lambda: window.statusBar().currentMessage() == _STATUSBAR_BAIL_MESSAGE,
        timeout=1000,
    )
    assert window.statusBar().currentMessage() == _STATUSBAR_BAIL_MESSAGE, (
        f"Expected verbatim status-bar bail message after disabled A/B "
        f"shortcut. Got: {window.statusBar().currentMessage()!r}. "
        "Likely _on_ab_shortcut_pressed bails without calling "
        "self.statusBar().showMessage(...) (Plan 07-09)."
    )

    # And the same for B — the bail path is symmetric.
    window.statusBar().clearMessage()
    window._on_ab_shortcut_pressed("B")
    qtbot.waitUntil(
        lambda: window.statusBar().currentMessage() == _STATUSBAR_BAIL_MESSAGE,
        timeout=1000,
    )
    assert window.statusBar().currentMessage() == _STATUSBAR_BAIL_MESSAGE
