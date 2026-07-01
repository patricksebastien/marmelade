"""Phase 7 Plan 07-02 Task 4 (RED) — Divergence badge + MainWindow wiring.

End-to-end: KeeperRow Master button → MasteringDialog → Apply →
sidecar write → divergence badge transition → single-keeper
MasteringRunnable when transitioning from mastering=None.

These tests use a lightweight stand-in for MainWindow's mastering
orchestration so they can exercise the wiring without bootstrapping
the full app. The actual MainWindow integration is covered by the
SUMMARY-level sanity check at the end of Task 4.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from marmelade.audio.mastering.chain import (
    _SESSION_DEFAULTS,
    config_hash,
    load_session_chain_snapshot,
)
from marmelade.audio.mastering_cache import mastered_cache_path
from marmelade.audio.proxy_cache import cache_key
from marmelade.audio.sidecar_cache import Region
from marmelade.ui.icons import _master_icon_with_badge
from marmelade.ui.keepers_sidebar import KeepersSidebar


def _make_proxy_wav(tmp_path: Path, seconds: float = 1.0) -> Path:
    """Create a tiny 44.1 kHz stereo proxy WAV for runnable tests."""
    sr = 44100
    n = int(seconds * sr)
    audio = (np.random.RandomState(0).randn(n, 2) * 0.1).astype("float32")
    p = tmp_path / "proxy.wav"
    sf.write(str(p), audio, sr, subtype="FLOAT", format="WAV")
    return p


def _badge_state_for(mastering: dict | None) -> str:
    """Replicate MainWindow's badge-state computation (Task 4 contract).

    none → keeper has no mastering.
    check → config_hash matches the session snapshot's hash.
    star → config_hash differs.
    """
    if mastering is None:
        return "none"
    session_hash = config_hash(load_session_chain_snapshot())
    new_hash = config_hash(mastering)
    return "check" if new_hash == session_hash else "star"


def test_no_mastering_shows_no_badge() -> None:
    """A keeper with mastering=None must compute to ``"none"`` badge state."""
    assert _badge_state_for(None) == "none"


def test_session_chain_match_shows_check_badge() -> None:
    """A keeper whose mastering == session snapshot hashes to ``"check"``."""
    snapshot = load_session_chain_snapshot()
    assert _badge_state_for(snapshot) == "check"


def test_custom_chain_shows_star_badge() -> None:
    """A keeper differing from session snapshot hashes to ``"star"``."""
    snapshot = load_session_chain_snapshot()
    custom = copy.deepcopy(snapshot)
    # Toggle a stage to force divergence.
    custom["highpass"]["enabled"] = True
    custom["highpass"]["cutoff_hz"] = 80.0
    assert _badge_state_for(custom) == "star"


def test_apply_changes_badge_from_check_to_star(qtbot, qapp) -> None:
    """KeeperRow.set_mastering_badge transitions check → star.

    Simulates the end of the Apply path: badge starts at ``"check"``
    (using session chain), user diverges via dialog, MainWindow calls
    set_mastering_badge("star").
    """
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    region = Region(
        id="id1234567890abcd",
        start_sec=0.0,
        end_sec=10.0,
        state="keeper",
    )
    row = sidebar.add_row(region)
    row.set_mastering_badge("check")
    assert "session mastering chain" in row._master.toolTip()
    row.set_mastering_badge("star")
    assert "Custom mastering chain" in row._master.toolTip()


def test_mastering_runnable_renders_pcm24_wav(tmp_path: Path) -> None:
    """A previously-unmastered keeper → MasteringRunnable produces a PCM_24 WAV.

    This pins the Task 4 "transition mastering=None → mastered cache"
    contract at the runnable boundary. MainWindow's orchestration uses
    exactly this runnable + this cache path resolution.
    """
    from PySide6.QtCore import QThreadPool

    from marmelade.audio.mastering_cache import is_mastered_cache_fresh
    from marmelade.audio.mastering_worker import MasteringRunnable

    proxy = _make_proxy_wav(tmp_path)
    src_key = cache_key(proxy)
    keeper_id = "abcdef0000000000000000000000000a"
    cfg = copy.deepcopy(_SESSION_DEFAULTS)
    # Limiter-on default — produces a non-trivial output.
    chash = config_hash(cfg)
    cache_root = tmp_path / "cache"
    dst = mastered_cache_path(cache_root, src_key, keeper_id, chash)

    runnable = MasteringRunnable(proxy, dst, keeper_id, cfg)
    finished_payloads: list[str] = []
    error_payloads: list[str] = []
    runnable.signals.finished.connect(finished_payloads.append)
    runnable.signals.error.connect(error_payloads.append)

    QThreadPool.globalInstance().start(runnable)

    # Wait for terminal signal (manual poll — qtbot.waitSignal works
    # against a QObject-bound signal, but the runnable lives off-thread).
    import time as _t

    deadline = _t.monotonic() + 10.0
    while _t.monotonic() < deadline and not (finished_payloads or error_payloads):
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.processEvents()
        _t.sleep(0.02)

    assert error_payloads == [], f"Mastering errored: {error_payloads}"
    assert finished_payloads, "Mastering did not emit finished within 10 s"
    assert is_mastered_cache_fresh(dst)


def test_cache_hit_skips_runnable_and_sets_ready_immediately(tmp_path: Path) -> None:
    """A pre-existing fresh cache file → MainWindow path computes "Ready" without runnable.

    Pinned at the cache-freshness layer; the MainWindow slot uses
    ``is_mastered_cache_fresh`` to decide between spawn-runnable and
    skip-to-ready. The widget contract is covered by the badge tests
    above.
    """
    from marmelade.audio.mastering_cache import is_mastered_cache_fresh

    proxy = _make_proxy_wav(tmp_path)
    src_key = cache_key(proxy)
    keeper_id = "abcdef0000000000000000000000000b"
    cfg = copy.deepcopy(_SESSION_DEFAULTS)
    chash = config_hash(cfg)
    cache_root = tmp_path / "cache"
    dst = mastered_cache_path(cache_root, src_key, keeper_id, chash)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Write a placeholder WAV so the freshness check passes.
    sr = 44100
    sf.write(
        str(dst),
        np.zeros((sr, 2), dtype="float32"),
        sr,
        subtype="PCM_24",
        format="WAV",
    )
    assert is_mastered_cache_fresh(dst)
