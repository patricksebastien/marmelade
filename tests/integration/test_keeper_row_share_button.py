"""Phase 8 Plan 08-04 — KeeperRow Share button + source-proxy fallback (R-05 / B1).

Plan 08-04 Task 3 ships the Share button itself + 8 GREEN tests
pinning YT-02 + plan-boundary invariants for Plan 08-05 (drag-handle
and bundle button MUST NOT appear here — Plan 08-05 owns those).

Plan 08-04 Task 4 adds ONE more test
(``test_share_button_on_unmastered_keeper_uses_source_proxy_fallback``)
that pins the R-05 + D-02 fallback contract end-to-end through
MainWindow.

R-05: per-keeper Share bypasses ``export_region`` when a fresh
mastered cache exists (feeds the cache WAV directly into the ffmpeg
subprocess). The SC #6 source-proxy fallback path is retained — when
no fresh mastered cache exists, per-keeper Share calls
``export_region(*, source_path=audio_proxy_path)`` to materialise the
unmastered audio (with Phase 3 fades + sample-rate handling) BEFORE
feeding into ffmpeg.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from marmelade.ui.keepers_sidebar import KeeperRow, KeepersSidebar


# A valid 32-hex region_id fixture so we don't sprinkle UUIDs everywhere.
REGION_ID = "0123456789abcdef0123456789abcdef"


def _make_row(qtbot, *, region_id: str = REGION_ID) -> KeeperRow:
    row = KeeperRow(
        region_id=region_id,
        start_sec=10.0,
        end_sec=20.0,
        state="keeper",
        note="",
    )
    qtbot.addWidget(row)
    return row


# ---------------------------------------------------------------------------
# Test 1 — Share button at the right insertion point (after _master, before _master_status)
# ---------------------------------------------------------------------------


def test_share_button_present_at_index_after_master(qtbot) -> None:
    """_share QPushButton sits AFTER _master AND BEFORE _master_status."""
    row = _make_row(qtbot)

    layout = row.layout()
    idx_share = layout.indexOf(row._share)
    idx_master = layout.indexOf(row._master)
    idx_master_status = layout.indexOf(row._master_status)

    assert idx_share != -1, "_share not in layout"
    assert idx_master != -1, "_master not in layout"
    assert idx_master_status != -1, "_master_status not in layout"

    assert idx_master < idx_share, (
        f"share must come AFTER master; share={idx_share}, master={idx_master}"
    )
    assert idx_share < idx_master_status, (
        f"share must come BEFORE master_status; share={idx_share}, "
        f"status={idx_master_status}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Always enabled (B1)
# ---------------------------------------------------------------------------


def test_share_button_always_enabled(qtbot) -> None:
    """Share button is enabled at construction (B1: no per-row gate)."""
    row = _make_row(qtbot)
    assert row._share.isEnabled() is True


# ---------------------------------------------------------------------------
# Test 3 — Click emits share_requested with region_id
# ---------------------------------------------------------------------------


def test_share_button_emits_signal_when_clicked(qtbot) -> None:
    """qtbot.mouseClick on _share emits share_requested(region_id)."""
    row = _make_row(qtbot)

    received: list[str] = []
    row.share_requested.connect(received.append)

    qtbot.mouseClick(row._share, Qt.MouseButton.LeftButton)
    assert received == [REGION_ID]


# ---------------------------------------------------------------------------
# Test 4 — Single tooltip string (B1 — no enabled/disabled split)
# ---------------------------------------------------------------------------


def test_share_button_tooltip_is_single_string(qtbot) -> None:
    """Tooltip text equals the single always-enabled string per B1."""
    row = _make_row(qtbot)
    assert (
        row._share.toolTip()
        == "Share to YouTube — uploads as MP4 with abstract image"
    )


# ---------------------------------------------------------------------------
# Test 5 — set_share_enabled method REMOVED (B1)
# ---------------------------------------------------------------------------


def test_no_set_share_enabled_method() -> None:
    """KeeperRow.set_share_enabled was REMOVED per revision iter 1 B1."""
    assert not hasattr(KeeperRow, "set_share_enabled")


# ---------------------------------------------------------------------------
# Test 6 — Wayland-safe icon: ≥100 non-transparent pixels
# ---------------------------------------------------------------------------


def test_share_icon_renders_non_background_pixels(qtbot) -> None:
    """_share_icon() rendered to 24x24 has ≥100 non-transparent pixels.

    Regression pin per Phase 7 LEARNINGS Surprise #9 (Wayland-safe icon
    contract) and 08-PATTERNS.md Shared Pattern 1. A Unicode glyph that
    Qt cannot actually render at small sizes would emit fewer than ~50
    visible pixels; QPainter-drawn shapes hit the floor reliably.
    """
    from marmelade.ui.icons import _share_icon

    icon = _share_icon()
    pix = icon.pixmap(24, 24)
    assert not pix.isNull()
    img = pix.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    count = 0
    for y in range(img.height()):
        for x in range(img.width()):
            argb = img.pixel(x, y)
            alpha = (argb >> 24) & 0xFF
            if alpha > 0:
                count += 1
    assert count >= 100, (
        f"share icon has only {count} non-transparent pixels; need ≥100 "
        "(Wayland safety — Phase 7 LEARNINGS)"
    )


# ---------------------------------------------------------------------------
# Test 7 — KeepersSidebar forwards share_requested
# ---------------------------------------------------------------------------


def test_keepers_sidebar_forwards_share_requested(qtbot) -> None:
    """KeepersSidebar.share_requested fires with region_id after row click."""
    from marmelade.audio.sidecar_cache import Region

    sidebar = KeepersSidebar()
    qtbot.addWidget(sidebar)

    region = Region(
        id=REGION_ID,
        start_sec=10.0,
        end_sec=20.0,
        state="keeper",
        note="",
    )
    row = sidebar.add_row(region)
    assert row._share.isEnabled() is True

    received: list[str] = []
    sidebar.share_requested.connect(received.append)

    qtbot.mouseClick(row._share, Qt.MouseButton.LeftButton)
    assert received == [REGION_ID], (
        f"expected sidebar.share_requested with region_id; got {received!r}"
    )


# ---------------------------------------------------------------------------
# Tests 8 & 9 were plan-boundary asserts that Plan 08-04 had NOT preempted
# Plan 08-05's surface (drag-handle + bundle button). Plan 08-05 now ships
# those widgets — the boundary tests would invert the assertion (must exist)
# but that's redundant with the Plan 08-05 tests in
# ``test_keepers_sidebar_drag_reorder.py`` and ``test_youtube_bundle_dialog.py``
# which pin the widgets directly. Removed here per Plan 08-05 Task 2 (drag-
# handle) and Task 3 (bundle button) which now own those surfaces.


# ---------------------------------------------------------------------------
# Plan 08-04 Task 4 — source-proxy fallback (B1 — R-05 + D-02 contract).
# ---------------------------------------------------------------------------


def test_share_button_on_unmastered_keeper_uses_source_proxy_fallback(
    qtbot, tmp_path, monkeypatch
) -> None:
    """Share on an unmastered keeper routes through export_region(source_path=audio_proxy).

    Pins the R-05 + D-02 fallback contract end-to-end:
    * KeeperRow Share button is ALWAYS enabled.
    * No fresh mastered cache → MainWindow.``_on_share_requested``
      calls ``export_region(*, source_path=self._current_playback_path,
      ...)`` to materialise the unmastered audio (with Phase 3 fades)
      into a tmp WAV.
    * The tmp WAV path is registered for cleanup on the upload state
      dict.
    * The peaks.dat binary at ``_current_proxy_p`` is NEVER passed to
      export_region (Phase 7 LEARNINGS Surprise carry-forward).
    """
    from PySide6.QtCore import QTimer

    from marmelade.audio.sidecar_cache import Region
    from marmelade.ui import main_window as mw_mod
    from marmelade.ui.main_window import MainWindow

    # Stub the file picker so MainWindow boots without prompting.
    window = MainWindow()
    qtbot.addWidget(window)

    # Wire up the audio_source paths the slot reads. The slot uses
    # _current_playback_path (the WAV proxy), NOT _current_proxy_p
    # (peaks.dat). The playback path must be a REAL WAV so the
    # slot's `sf.info(...).samplerate` read succeeds; the peaks /
    # source paths can be opaque blobs because the slot never opens
    # them.
    import numpy as np
    import soundfile as sf

    playback_p = tmp_path / "audio_proxy.wav"
    peaks_p = tmp_path / "audio.peaks.dat"
    src_p = tmp_path / "source.wav"
    # 30 seconds of silent stereo float32 at 44.1 kHz so the
    # [10s, 20s] keeper region maps to valid frames.
    silent = np.zeros((44100 * 30, 2), dtype=np.float32)
    sf.write(str(playback_p), silent, 44100, subtype="PCM_24")
    peaks_p.write_bytes(b"FAKEPEAKS")
    src_p.write_bytes(b"FAKESOURCE")
    window._current_playback_path = playback_p
    window._current_proxy_p = peaks_p
    window._current_path = src_p

    # Add an unmastered keeper to the regions overlay.
    region_id = REGION_ID
    region = Region(
        id=region_id,
        start_sec=10.0,
        end_sec=20.0,
        state="keeper",
        note="",
    )
    # Use the production set_regions API to install the region into the
    # overlay (it's a dict keyed by region_id, not a list — direct
    # append would AttributeError).
    window._regions_overlay.set_regions([region])
    window._keepers_sidebar.add_row(region)

    # Force the freshness check to return False so the fallback path
    # is selected. cache_key / config_hash / mastered_cache_path must
    # NOT raise for a None mastering (None is the unmastered case).
    monkeypatch.setattr(
        mw_mod, "is_mastered_cache_fresh", lambda _p: False, raising=True
    )

    # OAuth gate — pretend we're connected so the slot proceeds past
    # the credentials check.
    fake_creds = object()
    monkeypatch.setattr(
        mw_mod._yt_oauth, "load_or_refresh", lambda: fake_creds, raising=True
    )

    # Mock thumbnail fetch — return tiny bytes so the dialog can build.
    from marmelade.youtube import thumbnail_provider as tp
    monkeypatch.setattr(tp, "fetch_thumbnail", lambda seed, nonce: _MAKE_TINY_JPEG)

    # Spy on export_region — record its kwargs so the test can assert
    # source_path was _current_playback_path (not _current_proxy_p).
    export_calls = []
    from marmelade.audio import export_builder as eb

    def _spy_export_region(*args, **kwargs):
        # Match the export_builder signature shape.
        export_calls.append({"args": args, "kwargs": kwargs})
        # Write a tiny tmp WAV so subsequent code (if any) can read it.
        dst = kwargs.get("dst_path") if "dst_path" in kwargs else (
            args[1] if len(args) >= 2 else None
        )
        if dst is not None:
            from pathlib import Path
            Path(dst).write_bytes(b"FAKETMPWAV")

    monkeypatch.setattr(mw_mod, "export_region", _spy_export_region, raising=True)

    # Capture the upload state INSIDE the QTimer callback (before the
    # dialog rejects). MainWindow's `_on_share_requested` cleans up
    # _upload_state in its `finally` block when `dlg.exec()` returns,
    # so we have to snapshot the state while the modal is still open.
    captured_state: dict = {}

    def _capture_and_close():
        from marmelade.ui.upload_dialog import UploadDialog

        # Snapshot the state dict for region_id (deep-copy the entries
        # we care about — the original dict is mutated by the cleanup).
        st = window._upload_state.get(region_id)
        if st is not None:
            captured_state.update({
                "audio_source_path": st.get("audio_source_path"),
                "tmp_audio_to_cleanup": st.get("tmp_audio_to_cleanup"),
                "registered": True,
            })
        for w in window.findChildren(UploadDialog):
            w.reject()

    QTimer.singleShot(50, _capture_and_close)

    # Drive the slot — this is the exact path the Share button click
    # would invoke (we call directly to keep the test deterministic).
    window._on_share_requested(region_id)

    # ---- Assertions ----
    assert len(export_calls) == 1, (
        f"export_region must be called exactly once for the source-proxy "
        f"fallback path; got {len(export_calls)} calls"
    )
    kwargs = export_calls[0]["kwargs"]
    # source_path MUST be _current_playback_path (the WAV proxy),
    # NEVER _current_proxy_p (peaks.dat).
    src_path = kwargs.get("source_path")
    assert src_path is not None, (
        f"export_region must be called with source_path kwarg; got {kwargs!r}"
    )
    assert str(src_path) == str(playback_p), (
        f"source_path must be _current_playback_path ({playback_p}); "
        f"got {src_path!r} — Phase 7 LEARNINGS Surprise carry-forward "
        "(peaks.dat must NEVER reach export_region)"
    )
    assert str(src_path) != str(peaks_p), (
        f"source_path must NOT be _current_proxy_p (peaks.dat path "
        f"{peaks_p}); got {src_path!r}"
    )

    # The tmp WAV path was registered for cleanup on the upload state
    # while the dialog was open. (After dialog close, the finally block
    # in `_on_share_requested` cleans up the entry — which is why we
    # snapshot via the QTimer callback above.)
    assert captured_state.get("registered") is True, (
        f"upload state for {region_id} was not registered while the "
        f"dialog was open; captured_state={captured_state!r}"
    )
    assert captured_state.get("tmp_audio_to_cleanup") is not None, (
        "tmp WAV path must be registered for cleanup on upload state"
    )


# Small JPEG for the dialog thumbnail (tests don't need a 1280×720).
_MAKE_TINY_JPEG: bytes = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01"
    b"\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07"
    b"\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14"
    b"\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444"
    b"\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01"
    b"\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01"
    b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06"
    b"\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02"
    b"\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11"
    b"\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1"
    b"\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
    b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98"
    b"\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5"
    b"\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2"
    b"\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7"
    b"\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda"
    b"\x00\x08\x01\x01\x00\x00?\x00\xfb\xd0\xff\xd9"
)
