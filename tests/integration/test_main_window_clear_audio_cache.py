"""Plan 02.1-04 — D-08: File → Clear audio proxy cache menu action.

Pins:

* The action is wired (``window._action_clear_audio_cache`` is a QAction).
* Triggering the action deletes the audio cache subtree.
* The cache-size footer text reverts to ``"Cache: 0.00 GiB"`` after the
  clear.
* A transient status-bar message containing "Cleared audio proxy cache"
  is shown to the user.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from marmelade.audio.audio_proxy_cache import audio_cache_size_bytes
from marmelade.paths import default_cache_root
from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow
from tests.fixtures.synthesize import make_sine


def test_clear_audio_cache_action_clears_subtree_and_refreshes_footer(
    qtbot, qapp, tmp_cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trigger the menu action; cache vanishes; footer + status message updated."""
    theme.apply_theme(QApplication.instance())
    src = tmp_path / "clear_cache.mp3"
    make_sine(
        src,
        freq_hz=1000.0,
        amp=0.5,
        duration_s=3.0,
        sample_rate=44100,
        channels=1,
        fmt="mp3",
    )

    window = MainWindow()
    qtbot.addWidget(window)

    # Pre-warm the cache by completing an MP3 open. Drain both pipelines
    # so teardown sees a clean event loop (see mp3_open test rationale).
    proxy_done = {"v": False}
    render_done = {"v": False}
    window.audio_proxy_complete.connect(lambda _p: proxy_done.update(v=True))
    window.render_complete.connect(lambda: render_done.update(v=True))
    window._open_file(str(src))
    qtbot.waitUntil(
        lambda: proxy_done["v"] and render_done["v"],
        timeout=15000,
    )

    cache_root = default_cache_root()
    pre_size = audio_cache_size_bytes(cache_root)
    assert pre_size > 0, "expected the proxy build to populate the cache"

    # Spy on the status-bar transient showMessage call.
    captured: list[tuple] = []
    real_status_bar = window.statusBar()
    real_show_message = real_status_bar.showMessage

    def spy_show_message(msg, timeout=0):
        captured.append((msg, timeout))
        return real_show_message(msg, timeout)

    monkeypatch.setattr(real_status_bar, "showMessage", spy_show_message)

    # Trigger the action.
    window._action_clear_audio_cache.trigger()

    # Cache subtree gone.
    assert audio_cache_size_bytes(cache_root) == 0

    # Footer text reset.
    assert window._status_cache_size.text() == "Cache: 0.00 GiB"

    # Status-bar transient message captured with the expected wording.
    assert any("Cleared audio proxy cache" in msg for msg, _ in captured), (
        f"no Clear-cache status message captured (saw {captured!r})"
    )
