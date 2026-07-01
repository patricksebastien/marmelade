"""quick-260622-tit — _play_start_icon Wayland-safe pixel-floor pins.

Task 1 adds ``_play_start_icon`` as the horizontal MIRROR of
``_play_end_icon`` — a RIGHT-pointing triangle (apex at right) plus a
vertical "start" bar pinned to the LEFT edge (reads as ``|▶``). It
replaces the green SP_MediaPlay system triangle on the keeper-row Play
button so Play and End form a consistent toolbar-grey mirrored pair.

Contract (mirrors the existing icon regression floor in
``tests/integration/test_youtube_icons_pixel_count_regression.py``):

1. ``_play_start_icon`` renders ≥ 100 non-transparent pixels at 24×24
   (Wayland-safety floor — Phase 7 LEARNINGS Surprise #9 carry-forward).
2. ``_play_start_icon`` is visually DISTINCT from ``_play_middle_icon``
   (different rendered pixel footprint — start is a triangle+bar, middle
   is centered+flanking bars).
3. Both ``_play_start_icon`` and ``_play_end_icon`` are non-null and
   clear the floor (loose mirror-pair assertion).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QImage, QPainter, QPixmap


def _render_icon_to_image(icon, size: int = 24) -> QImage:
    """Render ``icon`` to a transparent QPixmap, return as QImage.

    Uses ``QIcon.paint`` so the result reflects exactly what Qt draws
    when the icon ships on a real button.
    """
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        icon.paint(painter, QRect(0, 0, size, size))
    finally:
        painter.end()
    return pix.toImage()


def _count_non_transparent_pixels(img: QImage) -> int:
    """Return the number of pixels with alpha > 0."""
    count = 0
    for y in range(img.height()):
        for x in range(img.width()):
            if img.pixelColor(x, y).alpha() > 0:
                count += 1
    return count


def _visible_pixel_set(img: QImage) -> frozenset:
    """Return the set of (x, y) coords whose pixel alpha > 0."""
    coords = set()
    for y in range(img.height()):
        for x in range(img.width()):
            if img.pixelColor(x, y).alpha() > 0:
                coords.add((x, y))
    return frozenset(coords)


def test_play_start_icon_renders_minimum_pixels(qapp) -> None:
    """_play_start_icon renders ≥ 100 non-transparent pixels at 24×24."""
    from marmelade.ui.icons import _play_start_icon

    icon = _play_start_icon()
    assert not icon.isNull(), "_play_start_icon returned a null QIcon"
    img = _render_icon_to_image(icon, size=24)
    n = _count_non_transparent_pixels(img)
    assert n >= 100, (
        f"_play_start_icon rendered only {n} non-transparent pixels at "
        "24x24 — Wayland-safe contract requires >= 100 (Phase 7 LEARNINGS "
        "Surprise #9 carry-forward)."
    )


def test_play_start_icon_distinct_from_middle(qapp) -> None:
    """_play_start_icon's rendered footprint differs from _play_middle_icon."""
    from marmelade.ui.icons import _play_middle_icon, _play_start_icon

    start_img = _render_icon_to_image(_play_start_icon(), size=24)
    middle_img = _render_icon_to_image(_play_middle_icon(), size=24)
    assert _visible_pixel_set(start_img) != _visible_pixel_set(middle_img), (
        "_play_start_icon must be visually distinct from _play_middle_icon "
        "(start = right-pointing triangle + left start-bar; middle = "
        "centered + flanking bars)."
    )


def test_play_start_and_end_form_mirror_pair(qapp) -> None:
    """Both start and end icons are non-null and clear the pixel floor."""
    from marmelade.ui.icons import _play_end_icon, _play_start_icon

    start_icon = _play_start_icon()
    end_icon = _play_end_icon()
    assert not start_icon.isNull()
    assert not end_icon.isNull()
    start_n = _count_non_transparent_pixels(_render_icon_to_image(start_icon))
    end_n = _count_non_transparent_pixels(_render_icon_to_image(end_icon))
    assert start_n >= 100
    assert end_n >= 100
