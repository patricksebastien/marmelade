"""Phase 8 Plan 08-06 Task 1 — Wayland-safe icon pixel-count regression pins (T-08-06-04).

Pins both ``_share_icon`` AND ``_drag_handle_icon`` against future
PySide6 / system-theme changes that could re-introduce Phase 7
Surprise #9 (invisible icons on Wayland because QSS specificity
overrides the painted-glyph fixed-size).

Three tests:

1. ``test_share_icon_renders_minimum_pixels`` — ≥100 non-transparent
   pixels in the 24×24 QPixmap.
2. ``test_drag_handle_icon_renders_minimum_pixels`` — same contract for
   the 6-dot drag-handle.
3. ``test_drag_handle_icon_contrast_against_light_bg`` — average
   brightness of the icon's non-transparent pixels is < 200/255 so the
   icon reads on a light system theme (mid-grey #9CA3AF target).

Wayland safety contract (Phase 7 LEARNINGS Surprise #9 carry-forward):
the painted-glyph approach (QPainter + QPainterPath) must produce a
pixmap with enough non-transparent pixels that the icon is visible at
24×24 regardless of GPU / scaling / theme.

Placebo audit (Phase 7 LEARNINGS):
    PRE-FIX expected failure signal — if a future refactor accidentally
    replaced the painter rendering with a Unicode glyph (which fails on
    Wayland at small sizes), the pixel count would drop below 100
    (transparent fill + missing-glyph rendering).

T-08-06-04 mitigation contract: this test IS the regression pin.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap


# No youtube fixtures needed here — icon tests are pure-Qt.


def _render_icon_to_image(icon, size=24) -> QImage:
    """Render ``icon`` to a transparent QPixmap, return as QImage.

    Uses ``QIcon.paint(painter, rect)`` so the result reflects exactly
    what Qt draws when the icon ships on a real button — captures any
    theme / size adjustment QIcon applies internally.
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
            c = img.pixelColor(x, y)
            if c.alpha() > 0:
                count += 1
    return count


def _average_brightness_of_visible_pixels(img: QImage) -> float:
    """Return average perceived brightness (0..255) of alpha>0 pixels.

    Brightness = 0.299*R + 0.587*G + 0.114*B (rec601 luma).
    """
    total = 0.0
    n = 0
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)
            if c.alpha() == 0:
                continue
            r, g, b = c.red(), c.green(), c.blue()
            total += 0.299 * r + 0.587 * g + 0.114 * b
            n += 1
    if n == 0:
        return 0.0
    return total / n


def test_share_icon_renders_minimum_pixels(qapp) -> None:
    """_share_icon renders ≥ 100 non-transparent pixels at 24×24.

    Wayland-safety regression pin against future PySide6 / QSS changes.
    """
    from marmelade.ui.icons import _share_icon

    icon = _share_icon()
    img = _render_icon_to_image(icon, size=24)
    n = _count_non_transparent_pixels(img)
    assert n >= 100, (
        f"_share_icon rendered only {n} non-transparent pixels at 24x24 "
        "— Wayland-safe contract requires >= 100 (Phase 7 LEARNINGS "
        "Surprise #9 carry-forward)."
    )


def test_drag_handle_icon_renders_minimum_pixels(qapp) -> None:
    """_drag_handle_icon renders ≥ 100 non-transparent pixels at 24×24.

    Wayland-safety regression pin against future PySide6 / QSS changes.
    """
    from marmelade.ui.icons import _drag_handle_icon

    icon = _drag_handle_icon()
    img = _render_icon_to_image(icon, size=24)
    n = _count_non_transparent_pixels(img)
    assert n >= 100, (
        f"_drag_handle_icon rendered only {n} non-transparent pixels at "
        "24x24 — Wayland-safe contract requires >= 100 (Phase 7 LEARNINGS "
        "Surprise #9 carry-forward)."
    )


def test_drag_handle_icon_contrast_against_light_bg(qapp) -> None:
    """_drag_handle_icon's average pixel brightness is < 200/255.

    The drag-handle is painted in mid-grey ``#9CA3AF`` so it contrasts
    against light system themes. Average brightness above 200/255 means
    the icon would visually disappear on a white-ish background.
    """
    from marmelade.ui.icons import _drag_handle_icon

    icon = _drag_handle_icon()
    img = _render_icon_to_image(icon, size=24)
    avg = _average_brightness_of_visible_pixels(img)
    assert 0 < avg < 200, (
        f"_drag_handle_icon avg brightness = {avg:.1f} (expected 0 < x < 200 "
        "for theme-contrast safety on light system themes)."
    )
