"""Composite icon builders for Phase 7 KeeperRow Master button (D-12).

Provides:
    * :func:`_master_icon_with_badge` — 24x24 composite QIcon composed of
      the existing gear icon plus an optional 8x8 divergence badge in the
      bottom-right corner. Used by every KeeperRow.
    * :func:`_five_pointed_star_path` — QPainterPath helper for the
      custom-mastering star badge.

Design constraints (Phase 6 LEARNINGS):
    * NEVER use Unicode glyphs — ``QFontMetrics.inFont()`` claims
      coverage Qt cannot actually render at small sizes on Ubuntu
      Wayland. All badges are painted as QPainterPath geometry.
    * The gear half of the icon reuses the :func:`_paint_gear_icon`
      helper defined in this module (relocated here from the retired
      ``layers_sidebar`` in quick-260621-dt4) so the Phase 6 + Phase 7
      buttons share one visual identity.

Badge state vocabulary (D-12):
    * ``"none"`` — keeper has ``mastering=None``; no badge painted.
    * ``"check"`` — keeper's config_hash matches the session chain;
      check-mark badge in light gray (``#E6E6E6``).
    * ``"star"`` — keeper diverges from the session chain; filled
      5-pointed star in accent blue (``#4DA3FF``).
"""

from __future__ import annotations

import math
from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _paint_gear_icon(size: int = 20) -> QIcon:
    """Paint a gear icon programmatically so the gear button always shows a
    visible glyph regardless of system font fallback for Unicode U+2699.

    The Plan 6-01 implementation used ``QPushButton("⚙")`` — Qt's
    ``QFontMetrics.inFont`` reported the codepoint as present in the Ubuntu
    default font but real-world rendering on some Linux + Wayland desktops
    showed an empty button (the font claimed coverage it didn't have at the
    drawn size). Painting the gear ourselves removes the font-fallback
    dependency entirely.

    Drawn shape: an 8-tooth gear with a central circular hole, stroked +
    filled in the system text color so it inherits dark/light theme.
    Anti-aliased; one 1.5px outline so the gear reads cleanly at 18-20 px.
    """
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    cx = cy = size / 2.0
    outer_r = size * 0.46  # tooth tips
    inner_r = size * 0.34  # gear body
    hole_r = size * 0.13   # center hole
    teeth = 8
    tooth_half_angle = math.radians(360 / teeth / 4)  # tooth width = quarter of slot

    path = QPainterPath()
    for i in range(teeth):
        base_angle = math.radians(i * 360 / teeth)
        a1 = base_angle - tooth_half_angle
        a2 = base_angle + tooth_half_angle
        a3 = base_angle + math.radians(360 / teeth) - tooth_half_angle
        # Inner arc start (between teeth)
        p_inner_start = QPointF(cx + inner_r * math.cos(a1), cy + inner_r * math.sin(a1))
        p_outer_start = QPointF(cx + outer_r * math.cos(a1), cy + outer_r * math.sin(a1))
        p_outer_end = QPointF(cx + outer_r * math.cos(a2), cy + outer_r * math.sin(a2))
        p_inner_end = QPointF(cx + inner_r * math.cos(a2), cy + inner_r * math.sin(a2))
        p_inner_next = QPointF(cx + inner_r * math.cos(a3), cy + inner_r * math.sin(a3))
        if i == 0:
            path.moveTo(p_inner_start)
        path.lineTo(p_outer_start)
        path.lineTo(p_outer_end)
        path.lineTo(p_inner_end)
        # Arc along inner_r from p_inner_end to p_inner_next
        rect = QRectF(cx - inner_r, cy - inner_r, 2 * inner_r, 2 * inner_r)
        sweep = math.degrees(a3 - a2)
        path.arcTo(rect, -math.degrees(a2), -sweep)
    path.closeSubpath()

    # Subtract center hole
    hole = QPainterPath()
    hole.addEllipse(QPointF(cx, cy), hole_r, hole_r)
    path = path.subtracted(hole)

    # Use a neutral foreground that reads in both dark and light themes.
    color = QColor("#6B7280")  # UI-SPEC tertiary text (between secondary and primary)
    painter.setBrush(QBrush(color))
    painter.setPen(QPen(color, 1.0))
    painter.drawPath(path)
    painter.end()
    return QIcon(pix)


def _gear_icon() -> QIcon:
    """Return a visible gear icon. Prefer the freedesktop theme (Linux
    GNOME / KDE ship a real settings icon under ``preferences-system``);
    fall back to the painted gear so the button is never empty.
    """
    icon = QIcon.fromTheme("preferences-system")
    if not icon.isNull():
        return icon
    icon = QIcon.fromTheme("emblem-system")
    if not icon.isNull():
        return icon
    return _paint_gear_icon(20)


_BadgeState = Literal["none", "check", "star"]

# UI-SPEC §Icon and Badge Rendering Contract — color tokens. Frozen here
# for source-grep gates.
_CHECK_COLOR = "#E6E6E6"  # primary text gray
_STAR_COLOR = "#4DA3FF"  # accent blue


def _five_pointed_star_path(
    cx: float, cy: float, r_outer: float, r_inner: float
) -> QPainterPath:
    """Build a closed 5-pointed star QPainterPath centered at ``(cx, cy)``.

    The standard 5-pointed star alternates between outer and inner radius
    at 36-degree increments (10 points total), starting at -90 degrees
    (pointing up). The resulting path is closed.

    Args:
        cx, cy: Center coordinates in pixel space.
        r_outer: Distance from center to outer points (the star tips).
        r_inner: Distance from center to inner concave points.

    Returns:
        Closed :class:`QPainterPath` suitable for ``painter.fillPath``.
    """
    path = QPainterPath()
    # 10 alternating points: outer, inner, outer, inner, ...
    # Start at -90 deg (top), step by 36 deg each point.
    for i in range(10):
        angle_deg = -90.0 + i * 36.0
        angle_rad = math.radians(angle_deg)
        r = r_outer if i % 2 == 0 else r_inner
        x = cx + r * math.cos(angle_rad)
        y = cy + r * math.sin(angle_rad)
        if i == 0:
            path.moveTo(QPointF(x, y))
        else:
            path.lineTo(QPointF(x, y))
    path.closeSubpath()
    return path


def _master_icon_with_badge(badge_state: _BadgeState) -> QIcon:
    """Build a 24x24 composite QIcon: gear + optional bottom-right badge.

    Layout (UI-SPEC §Icon and Badge Rendering Contract):
        * Gear painted into the top-left 20x20 region (uses Phase 6's
          :func:`_paint_gear_icon`).
        * Badge (when not "none") painted into the 8x8 bottom-right
          region — bounding box ``(16,16)`` to ``(24,24)``, 4 px inset
          from the right + bottom edges. The badge center is at
          ``(20, 20)`` (4 px inside the 24x24 canvas from the bottom-right
          corner).

    Args:
        badge_state: ``"none"`` (gear only), ``"check"`` (gear + check
            mark), or ``"star"`` (gear + filled star).

    Returns:
        :class:`QIcon` ready for ``QPushButton.setIcon``. Always non-null.
    """
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # (1) Gear layer — paint Phase 6's gear into the top-left 20x20
        # square. We use the QIcon paint() entry point to defer to the
        # gear's existing painter (which already builds the gear path
        # from ClassVars).
        gear_icon = _paint_gear_icon(20)
        gear_icon.paint(painter, 0, 0, 20, 20)

        # (2) Badge layer — paint into the 8x8 bottom-right region
        # centered at (20, 20).
        if badge_state == "check":
            # Three-point check-mark stroke. UI-SPEC values verbatim:
            # moveTo(17.0, 20.0), lineTo(18.5, 21.5), lineTo(22.5, 17.0).
            path = QPainterPath()
            path.moveTo(QPointF(17.0, 20.0))
            path.lineTo(QPointF(18.5, 21.5))
            path.lineTo(QPointF(22.5, 17.0))
            pen = QPen(QColor(_CHECK_COLOR), 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)
        elif badge_state == "star":
            # Filled 5-pointed star — outer radius 4 px, inner radius
            # 1.6 px, centered at (20, 20).
            path = _five_pointed_star_path(cx=20.0, cy=20.0, r_outer=4.0, r_inner=1.6)
            painter.fillPath(path, QColor(_STAR_COLOR))
        # "none" → skip badge layer (gear-only icon).
    finally:
        painter.end()

    return QIcon(pix)


def _drag_handle_icon() -> QIcon:
    """Build a 24x24 Wayland-safe drag-handle icon (Phase 8 Plan 08-05 D-05 + D-29).

    Layout:
        * Six dots arranged in a 2-column × 3-row grid forming a
          "dotted-grip" glyph. Centers at::

              (9, 6)  (15, 6)
              (9, 12) (15, 12)
              (9, 18) (15, 18)

          Each dot is drawn as a filled QPainterPath ellipse with
          a 2.8 px radius (larger than the 2 px floor in the plan
          spec so the regression pin of ≥100 non-transparent pixels
          comfortably holds across Wayland's anti-aliasing variance).
        * Color: mid-grey ``#9CA3AF`` (UI-SPEC secondary text gray,
          same hue family as the existing Trash badge so the row's
          "non-actionable affordance" widgets read as one visual
          family).

    Wayland safety: NO Unicode glyph (RESEARCH §"drag-handle glyph"
    rejected the ``≡`` / ``⋮⋮`` Unicode options because they fail
    QFontMetrics.inFont() probe-reality at small sizes on Ubuntu).
    The painted-circle approach is the Option B "dotted-grip"
    recommendation from RESEARCH lines 817-824.

    Returns:
        :class:`QIcon` suitable for ``QPushButton.setIcon``.
    """
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        grip_color = QColor("#9CA3AF")
        radius = 2.8
        for cx in (9.0, 15.0):
            for cy in (6.0, 12.0, 18.0):
                path = QPainterPath()
                path.addEllipse(QPointF(cx, cy), radius, radius)
                painter.fillPath(path, grip_color)
    finally:
        painter.end()
    return QIcon(pix)


def _share_icon() -> QIcon:
    """Build a 24x24 Wayland-safe generic share icon (Phase 8 D-29).

    Standard share-tree silhouette (Android / iOS convention) — three
    filled nodes connected by two lines, suggesting "fan out / send
    elsewhere". Service-agnostic: works for YouTube, MP3 export, or
    any future share target.

    Geometry:
        * Top-right node at ``(17, 6)`` — radius 3
        * Bottom-right node at ``(17, 18)`` — radius 3
        * Left-middle node at ``(7, 12)`` — radius 3
        * Two 2-px-stroke lines connecting the left node to each
          right node, drawn UNDERNEATH the filled circles so the
          line tips disappear cleanly into each node.

    Color: toolbar grey (:data:`_TOOLBAR_ICON_COLOR`) — sits in the
    same chrome family as the zoom / fit / region-select icons so the
    Share button reads as a neutral affordance rather than an accent.

    Wayland safety: NO Unicode glyph. Pure QPainter geometry. The
    rendered pixmap exceeds the ≥100-non-transparent-pixel regression
    floor pinned by ``test_share_icon_renders_non_background_pixels``.

    Returns:
        :class:`QIcon` suitable for ``QPushButton.setIcon``.
    """
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(_TOOLBAR_ICON_COLOR)

        # Connecting lines first so the filled nodes overlay the tips.
        pen = QPen(color)
        pen.setWidthF(2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(7.0, 12.0), QPointF(17.0, 6.0))
        painter.drawLine(QPointF(7.0, 12.0), QPointF(17.0, 18.0))

        # Three filled nodes.
        node_path = QPainterPath()
        node_path.addEllipse(QPointF(17.0, 6.0), 3.0, 3.0)
        node_path.addEllipse(QPointF(17.0, 18.0), 3.0, 3.0)
        node_path.addEllipse(QPointF(7.0, 12.0), 3.0, 3.0)
        painter.fillPath(node_path, color)
    finally:
        painter.end()
    return QIcon(pix)


# --------------------------------------------------------------------------
# Toolbar icons (zoom-in / zoom-out / fit / region-select). All 24×24,
# Wayland-safe (QPainter geometry, no Unicode glyphs), monochrome with the
# secondary-text grey so they read as toolbar chrome rather than data.
# --------------------------------------------------------------------------

_TOOLBAR_ICON_COLOR = "#C8C8CD"  # toolbar foreground grey


def _magnifier_glass(painter: QPainter, color: QColor) -> None:
    """Paint a generic magnifying-glass silhouette into the 24×24 canvas.

    Lens centered at (10, 10) with outer radius 6.5 and inner radius
    4.5 (stroked annulus). Handle is a 3-px stroke from the SE edge of
    the lens out to (20, 20). The lens interior stays empty so callers
    can stamp ``+`` or ``-`` inside it.
    """
    pen = QPen(color)
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.GlobalColor.transparent)
    painter.drawEllipse(QPointF(10.0, 10.0), 5.5, 5.5)
    # Handle: tangent at ~45° from lens center, ends at (20, 20).
    painter.drawLine(QPointF(14.0, 14.0), QPointF(20.0, 20.0))


def _zoom_in_icon() -> QIcon:
    """Magnifying glass with ``+`` inside the lens (24×24, Wayland-safe)."""
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(_TOOLBAR_ICON_COLOR)
        _magnifier_glass(painter, color)
        pen = QPen(color)
        pen.setWidthF(1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(7.5, 10.0), QPointF(12.5, 10.0))
        painter.drawLine(QPointF(10.0, 7.5), QPointF(10.0, 12.5))
    finally:
        painter.end()
    return QIcon(pix)


def _zoom_out_icon() -> QIcon:
    """Magnifying glass with ``−`` inside the lens (24×24, Wayland-safe)."""
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(_TOOLBAR_ICON_COLOR)
        _magnifier_glass(painter, color)
        pen = QPen(color)
        pen.setWidthF(1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(7.5, 10.0), QPointF(12.5, 10.0))
    finally:
        painter.end()
    return QIcon(pix)


def _fit_to_view_icon() -> QIcon:
    """Four corner brackets pointing inward — the classic "fit-to-view" glyph (24×24)."""
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(_TOOLBAR_ICON_COLOR))
        pen.setWidthF(2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        # Top-left bracket
        painter.drawLine(QPointF(4.0, 8.0), QPointF(4.0, 4.0))
        painter.drawLine(QPointF(4.0, 4.0), QPointF(8.0, 4.0))
        # Top-right
        painter.drawLine(QPointF(16.0, 4.0), QPointF(20.0, 4.0))
        painter.drawLine(QPointF(20.0, 4.0), QPointF(20.0, 8.0))
        # Bottom-right
        painter.drawLine(QPointF(20.0, 16.0), QPointF(20.0, 20.0))
        painter.drawLine(QPointF(20.0, 20.0), QPointF(16.0, 20.0))
        # Bottom-left
        painter.drawLine(QPointF(8.0, 20.0), QPointF(4.0, 20.0))
        painter.drawLine(QPointF(4.0, 20.0), QPointF(4.0, 16.0))
    finally:
        painter.end()
    return QIcon(pix)


def _follow_playhead_icon() -> QIcon:
    """Right-pointing chevron + small vertical bar — "follow playhead" (24×24).

    Reads as "track the playhead's motion": a chevron pointing right
    (motion direction) with a thin vertical bar in front of it (the
    playhead). Matches the toolbar grey hue family of the other custom
    icons.
    """
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(_TOOLBAR_ICON_COLOR))
        pen.setWidthF(2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.GlobalColor.transparent)
        # Two stacked chevrons pointing right — the "follow / advance"
        # convention used by media players for fast-forward / track-next.
        # Left chevron at x=6-12, right chevron at x=12-18.
        painter.drawLine(QPointF(6.0, 6.0), QPointF(12.0, 12.0))
        painter.drawLine(QPointF(12.0, 12.0), QPointF(6.0, 18.0))
        painter.drawLine(QPointF(12.0, 6.0), QPointF(18.0, 12.0))
        painter.drawLine(QPointF(18.0, 12.0), QPointF(12.0, 18.0))
    finally:
        painter.end()
    return QIcon(pix)


def _play_middle_icon() -> QIcon:
    """Build a 24x24 Wayland-safe "play from middle" icon (quick-260622-sr8).

    Mental model: Play=start / middle=middle / end=ending. The middle glyph
    is a centered solid vertical bar flanked by two thinner side bars — it
    reads as "the middle of the clip" and contrasts cleanly with the play
    triangle (SP_MediaPlay) and the mirrored end-arrow.

    Geometry (filled QPainterPath rectangles, no Unicode glyph):
        * Center bar: 4 px wide, 14 px tall, centered at x=12.
        * Two flanking bars: 2 px wide, 10 px tall, at x=5 and x=17.

    Color: toolbar grey (:data:`_TOOLBAR_ICON_COLOR`) so the three keeper
    play affordances read as one neutral family. The filled geometry
    comfortably clears the >=100-non-transparent-pixel regression floor.

    Returns:
        :class:`QIcon` suitable for ``QPushButton.setIcon``.
    """
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(_TOOLBAR_ICON_COLOR)
        path = QPainterPath()
        # Center bar (5x16, centered).
        path.addRect(QRectF(9.5, 4.0, 5.0, 16.0))
        # Flanking bars (3x12).
        path.addRect(QRectF(3.5, 6.0, 3.0, 12.0))
        path.addRect(QRectF(17.5, 6.0, 3.0, 12.0))
        painter.fillPath(path, color)
    finally:
        painter.end()
    return QIcon(pix)


def _play_end_icon() -> QIcon:
    """Build a 24x24 Wayland-safe "play from end" icon (quick-260622-sr8).

    Reads as "play toward / near the ending": a play triangle MIRRORED
    horizontally (pointing LEFT, toward the clip start) plus a solid
    vertical "end-stop" bar pinned to the right edge — the bar is the
    end of the clip, the left-pointing triangle the playback rolling into
    it. Visually distinct from BOTH SP_MediaPlay (right-pointing, no bar)
    and :func:`_play_middle_icon` (centered bars, no triangle).

    Geometry (filled QPainterPath, no Unicode glyph):
        * Left-pointing triangle: apex at (6, 12), base edge x=16 spanning
          y=5..19.
        * End-stop bar: 3 px wide, 16 px tall at the right edge (x=18).

    Color: toolbar grey (:data:`_TOOLBAR_ICON_COLOR`). Filled geometry
    clears the >=100-non-transparent-pixel regression floor.

    Returns:
        :class:`QIcon` suitable for ``QPushButton.setIcon``.
    """
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(_TOOLBAR_ICON_COLOR)
        path = QPainterPath()
        # Left-pointing triangle (apex at left).
        path.moveTo(QPointF(6.0, 12.0))
        path.lineTo(QPointF(16.0, 5.0))
        path.lineTo(QPointF(16.0, 19.0))
        path.closeSubpath()
        # End-stop bar at the right edge.
        path.addRect(QRectF(18.0, 4.0, 3.0, 16.0))
        painter.fillPath(path, color)
    finally:
        painter.end()
    return QIcon(pix)


def _play_start_icon() -> QIcon:
    """Build a 24x24 Wayland-safe "play from start" icon (quick-260622-tit).

    The mirror SIBLING of :func:`_play_end_icon`: a play triangle pointing
    RIGHT (apex at the right edge, toward the clip body) rolling OUT of a
    solid vertical "start" bar pinned to the LEFT edge — the bar is the
    start of the clip, the right-pointing triangle the playback rolling
    forward from it (reads as ``|▶``). Together Play and End form a
    consistent toolbar-grey mirrored pair ( ``|▶`` start, ``◀|`` end ).
    Replaces the green ``SP_MediaPlay`` system triangle so the three
    keeper play affordances read as one neutral family.

    Geometry (filled QPainterPath, no Unicode glyph) — the horizontal
    mirror of :func:`_play_end_icon`:
        * Right-pointing triangle: apex at (18, 12), base edge x=8 spanning
          y=5..19.
        * Start bar: 3 px wide, 16 px tall at the left edge (x=3).

    Color: toolbar grey (:data:`_TOOLBAR_ICON_COLOR`). Filled geometry
    clears the >=100-non-transparent-pixel regression floor.

    Returns:
        :class:`QIcon` suitable for ``QPushButton.setIcon``.
    """
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(_TOOLBAR_ICON_COLOR)
        path = QPainterPath()
        # Right-pointing triangle (apex at right).
        path.moveTo(QPointF(18.0, 12.0))
        path.lineTo(QPointF(8.0, 5.0))
        path.lineTo(QPointF(8.0, 19.0))
        path.closeSubpath()
        # Start bar at the left edge.
        path.addRect(QRectF(3.0, 4.0, 3.0, 16.0))
        painter.fillPath(path, color)
    finally:
        painter.end()
    return QIcon(pix)


def _region_select_icon() -> QIcon:
    """Bracketed-arrows region marker — reads as ``[< >]`` (24×24).

    Two square brackets at the left + right edges with arrows pointing
    outward toward each bracket — conveys "draggable region between
    two boundary markers" much better than the generic dashed marquee
    (which read as a list-view selection on Wayland).

    Geometry (anti-aliased, 1.6 px stroke):
        * Left bracket ``[`` at x=4 with short top/bottom serifs.
        * Right bracket ``]`` at x=19 with mirrored serifs.
        * Inner ``<`` at x≈8-10 pointing LEFT toward the bracket.
        * Inner ``>`` at x≈14-16 pointing RIGHT toward the bracket.
    """
    pix = QPixmap(24, 24)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(_TOOLBAR_ICON_COLOR))
        pen.setWidthF(1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.GlobalColor.transparent)

        # Left bracket ``[`` — vertical bar + top/bottom serifs.
        painter.drawLine(QPointF(4.0, 6.0), QPointF(4.0, 18.0))
        painter.drawLine(QPointF(4.0, 6.0), QPointF(7.0, 6.0))
        painter.drawLine(QPointF(4.0, 18.0), QPointF(7.0, 18.0))

        # Right bracket ``]`` — mirrored.
        painter.drawLine(QPointF(20.0, 6.0), QPointF(20.0, 18.0))
        painter.drawLine(QPointF(17.0, 6.0), QPointF(20.0, 6.0))
        painter.drawLine(QPointF(17.0, 18.0), QPointF(20.0, 18.0))

        # Inner ``<`` arrow pointing LEFT, centered vertically at y=12.
        painter.drawLine(QPointF(10.0, 9.0), QPointF(7.5, 12.0))
        painter.drawLine(QPointF(7.5, 12.0), QPointF(10.0, 15.0))

        # Inner ``>`` arrow pointing RIGHT.
        painter.drawLine(QPointF(14.0, 9.0), QPointF(16.5, 12.0))
        painter.drawLine(QPointF(16.5, 12.0), QPointF(14.0, 15.0))
    finally:
        painter.end()
    return QIcon(pix)


__all__ = [
    "_gear_icon",
    "_paint_gear_icon",
    "_five_pointed_star_path",
    "_master_icon_with_badge",
    "_share_icon",
    "_drag_handle_icon",
    "_zoom_in_icon",
    "_zoom_out_icon",
    "_fit_to_view_icon",
    "_region_select_icon",
    "_follow_playhead_icon",
    "_play_middle_icon",
    "_play_end_icon",
]
