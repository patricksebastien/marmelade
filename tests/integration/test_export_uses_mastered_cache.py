"""Phase 7 Plan 07-06 Task 2 — Phase C export routing.

Focused unit-style integration test for the D-20 routing rule:

* keeper.mastering is set AND mastered cache is fresh →
  ``export_region(source_path=<mastered_cache>)``.
* otherwise → ``export_region(source_path=None)`` (source proxy is the
  audio source, existing Phase 3 behavior).

The parent integration test
:mod:`tests.integration.test_master_export_all` exercises the
end-to-end flow including Phase A first; this module pins the
routing rule alone for clearer failure diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from marmelade.audio import sidecar_cache
from marmelade.audio.mastering.chain import (
    config_hash,
    load_session_chain_snapshot,
)
from marmelade.audio.mastering_cache import mastered_cache_path
from marmelade.audio.proxy_cache import cache_key as proxy_cache_key
from marmelade.audio.sidecar_cache import Region
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


SR = 44100


def _make_source_wav(path: Path, seconds: float = 3.0) -> Path:
    n = int(seconds * SR)
    audio = (
        0.5
        * np.sin(2.0 * np.pi * 440.0 * np.arange(n, dtype=np.float64) / SR)
    ).astype(np.float32)
    stereo = np.stack([audio, audio], axis=1)
    sf.write(str(path), stereo, SR, subtype="FLOAT", format="WAV")
    return path


def test_export_phase_c_routes_mastered_keepers_through_source_path(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """Phase C: mastered keeper → source_path = cache; unmastered → source_path = None.

    Setup: pre-stage a mastered cache file on disk for the FIRST keeper
    so the orchestrator sees ``is_mastered_cache_fresh = True`` and
    routes through ``source_path``. The SECOND keeper has no
    ``mastering`` field — routes through the source proxy.
    """
    theme.apply_theme(QApplication.instance())
    src = _make_source_wav(tmp_path / "src.wav", seconds=3.0)

    window = MainWindow()
    qtbot.addWidget(window)
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: window._current_sidecar_path is not None
        and window._current_playback_path is not None,
        timeout=15000,
    )

    snapshot = load_session_chain_snapshot()
    # 32-hex IDs — mastered_cache_path validates against ^[0-9a-f]{32}$.
    mastered_keeper = Region(
        id="a" * 32,
        start_sec=0.1,
        end_sec=0.6,
        state="keeper",
        note="",
        mastering=snapshot,
    )
    plain_keeper = Region(
        id="b" * 32,
        start_sec=1.0,
        end_sec=1.5,
        state="keeper",
        note="",
        mastering=None,
    )
    window._regions_overlay.set_regions([mastered_keeper, plain_keeper])
    window._on_regions_changed()

    # NOTE (Plan 07-08): This test pins the Phase C ROUTING rule only
    # (mastered keeper → source_path = cache_p; unmastered →
    # source_path = None). We pre-stage a region-bounded cache here
    # to skip Phase A's MasteringRunnable — that worker's
    # region-bounded behavior is pinned end-to-end in
    # tests/integration/test_master_export_all.py and
    # tests/integration/audio/test_mastering_runnable.py. If you
    # change those, also revisit this stub.
    #
    # Pre-stage the mastered cache file for the mastered keeper so the
    # orchestrator's is_mastered_cache_fresh check returns True without
    # running an actual MasteringRunnable.
    src_key = proxy_cache_key(src)
    chash = config_hash(snapshot)
    cache_p = mastered_cache_path(
        default_cache_root(), src_key, mastered_keeper.id, chash
    )
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    # Tiny WAV sized to the keeper region — encodes intent (the stub
    # is region-length, NOT a magic 0.5 s number) so the contract is
    # readable. The is_mastered_cache_fresh check only requires the
    # file to exist + be non-empty; content shape matters only for
    # the Plan 07-08 region-bound semantics pinned elsewhere.
    _make_source_wav(
        cache_p,
        seconds=(mastered_keeper.end_sec - mastered_keeper.start_sec),
    )

    # Set up Phase C state by hand — we skip Phase A entirely.
    window._master_all_target_dir = tmp_path / "out"
    window._master_all_format = "wav"

    captured: list[dict[str, Any]] = []
    import marmelade.ui.main_window as mw

    def capture_spawn(self, **kwargs):
        captured.append(kwargs)
        # Simulate immediate finish so the loop progresses.
        self.export_complete.emit(str(kwargs["dst_path"]))

    monkeypatch.setattr(mw.MainWindow, "_spawn_export_worker", capture_spawn)

    window._on_export_all_requested()
    qtbot.waitUntil(lambda: len(captured) == 2, timeout=10000)

    # Find each call by which keeper it corresponds to. We use the
    # dst_path filename: the naming_resolver derives ``HHMMSS_`` from
    # the region's start_sec, so the mastered keeper (start 0.1 → 000000)
    # and the plain keeper (start 1.0 → 000001) are distinguishable.
    by_source_path = {
        "with_override": next(
            c for c in captured if c.get("source_path") is not None
        ),
        "without_override": next(
            c for c in captured if c.get("source_path") is None
        ),
    }

    # The "with_override" call MUST correspond to the mastered keeper.
    assert Path(by_source_path["with_override"]["source_path"]) == cache_p
    # The "without_override" call is the plain keeper — uses the
    # source proxy verbatim (existing Phase 3 path).
    assert by_source_path["without_override"].get("source_path") is None
