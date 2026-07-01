"""quick-260622-sr8 Task 1 — play-from-middle / play-from-end icons + the
pure start-offset helper.

Two deliverables under test:

  (a) ICONS — ``_play_middle_icon`` / ``_play_end_icon`` in
      ``marmelade.ui.icons``. Wayland-safe (QPainter geometry, no
      Unicode glyphs). Each must clear the >=100 non-transparent-pixel
      regression floor used by the other custom icons, and the two new
      icons must be visually distinct from each other AND from the
      Qt standard ``SP_MediaPlay`` glyph.

  (b) HELPER — ``_keeper_play_offsets(start_sec, end_sec, mode)`` in
      ``marmelade.ui.main_window``. Pure float math, no Qt. The single
      source of truth for the start offset + fade-in suppression flag the
      three keeper-play buttons route through. Returns
      ``(start_seconds: float, suppress_fade_in: bool)``.

The math is asserted against the helper's return value — NOT re-implemented
here (the helper is the contract).
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QStyle

from marmelade.ui.icons import _play_end_icon, _play_middle_icon
from marmelade.ui.main_window import _keeper_play_offsets


def _non_transparent_pixel_count(icon, w: int = 24, h: int = 24) -> int:
    image = icon.pixmap(w, h).toImage()
    count = 0
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).alpha() > 0:
                count += 1
    return count


# --------------------------------------------------------------------------
# (a) ICON render assertions
# --------------------------------------------------------------------------


def test_play_middle_icon_renders_non_transparent(qapp) -> None:
    """_play_middle_icon clears the >=100 non-transparent-pixel floor."""
    icon = _play_middle_icon()
    assert not icon.isNull()
    assert _non_transparent_pixel_count(icon) >= 100


def test_play_end_icon_renders_non_transparent(qapp) -> None:
    """_play_end_icon clears the >=100 non-transparent-pixel floor."""
    icon = _play_end_icon()
    assert not icon.isNull()
    assert _non_transparent_pixel_count(icon) >= 100


def test_middle_and_end_icons_are_distinct(qapp) -> None:
    """The two new icons must be visually distinct pixmaps."""
    middle = _play_middle_icon().pixmap(24, 24).toImage()
    end = _play_end_icon().pixmap(24, 24).toImage()
    assert middle != end


def test_new_icons_distinct_from_standard_play(qapp) -> None:
    """Both new icons must differ from the Qt SP_MediaPlay triangle."""
    play = (
        QApplication.style()
        .standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        .pixmap(24, 24)
        .toImage()
    )
    middle = _play_middle_icon().pixmap(24, 24).toImage()
    end = _play_end_icon().pixmap(24, 24).toImage()
    assert middle != play
    assert end != play


# --------------------------------------------------------------------------
# (b) _keeper_play_offsets math
# --------------------------------------------------------------------------


def test_start_mode_returns_start_no_suppress() -> None:
    assert _keeper_play_offsets(10.0, 40.0, "start") == (10.0, False)


def test_middle_mode_returns_midpoint_and_suppresses_fade_in() -> None:
    # midpoint of [10, 40] is 25; fade-in suppressed.
    assert _keeper_play_offsets(10.0, 40.0, "middle") == (25.0, True)


def test_end_mode_returns_end_minus_5_and_suppresses_fade_in() -> None:
    # quick-260622-ud0 — end mode now suppresses the fade-IN (fade-out kept).
    assert _keeper_play_offsets(10.0, 40.0, "end") == (35.0, True)


def test_end_mode_clamps_short_keeper_to_start() -> None:
    # 3s keeper [10, 13] — end-5 would be 8.0 < start; clamp to start.
    # quick-260622-ud0 — fade-in suppressed for end mode.
    assert _keeper_play_offsets(10.0, 13.0, "end") == (10.0, True)


def test_unrecognized_mode_degrades_to_start() -> None:
    # T-sr8-02 — a typo / future mode degrades to "start" behavior.
    assert _keeper_play_offsets(10.0, 40.0, "wat") == (10.0, False)
