"""Region overlay on the WaveformView's PlotItem (Phase 3 — REG-01/02/03).

Plain Python class (NOT a QWidget subclass) that owns PyQtGraph
``LinearRegionItem`` instances installed on a host PlotItem. The owning
:class:`marmelade.ui.waveform_view.WaveformView` routes region-create
gestures into this overlay via its extended eventFilter.

Phase 3 surface — Plan 01 thin slice + Plan 02 region UX:

* ``ResizeOnlyRegion(LinearRegionItem)`` — subclass that disables
  body-drag (CONTEXT D-A1-4 + UI-SPEC §Interaction Contract). Edge resize
  stays enabled via the child InfiniteLines' own ``mouseDragEvent``
  handlers. Plan 02 adds:

  * ``setAcceptHoverEvents(True)`` opt-in (RESEARCH §Pitfall #11).
  * ``hoverEnterEvent`` / ``hoverLeaveEvent`` overrides that flip the
    overlay's ``hovered_region_id`` and emit ``hover_changed``.
  * ``getContextMenus(event)`` returning ``list[QMenu]`` (RESEARCH
    §Pitfall #8) with the four region actions:
    Mark as Keeper / Mark as Trash / Unmark / Delete region.
  * ``_current_state`` attribute tracking the region's state so
    ``regions_data`` returns the live state.
  * ``_note`` and ``_created_at`` carried on the widget so the
    sidecar round-trip is faithful.

* ``RegionsOverlay(QObject)`` — owns the per-region widget dict plus a
  draft LinearRegionItem during in-progress Shift+drag gestures. Plan 02
  adds:

  * ``hovered_region_id`` (str | None) — read by MainWindow K/T/U/Delete
    Edit-menu slots.
  * ``hover_changed = Signal(object)`` — payload = new hovered id or
    ``None``. MainWindow connects to enable/disable Edit-menu actions.
  * ``set_state(region_id, state)`` — mutates a region's state, re-applies
    brush/hover/pen, emits ``regions_changed``.
  * ``delete(region_id)`` — removes a region from the plot + dict +
    clears hover if matching.
  * ``set_state_of_hovered(state)`` / ``delete_hovered()`` — convenience
    wrappers consumed by MainWindow shortcut/menu slots.

UI-SPEC tokens (NONE may be ``#4DA3FF`` — playhead-reserved):

* Draft:      brush ``#FFC857`` alpha=64, pen ``#FFC857`` alpha=220
* Untouched:  brush ``#6B8FA8`` alpha=48, hover alpha=72, pen alpha=200
* Keeper:     brush ``#FFC857`` alpha=56, hover alpha=88, pen alpha=220
* Trash:      brush ``#3F3F3F`` alpha=144, hover alpha=168, pen
                ``#5C636B`` alpha=200

Qt cleanup discipline (Phase 1 LEARNINGS §"removeItem alone leaks"):
when removing a region, call ``plot_item.removeItem(region)`` AND then
``region.deleteLater()`` to free the QGraphicsItem.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Callable

import pyqtgraph as pg
from PySide6.QtCore import QObject, QPoint, Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import QMenu

from marmelade.audio.sidecar_cache import Marker, Region


# UI-SPEC §Color — region-state PyQtGraph brushes. None may be
# ``#4DA3FF`` (reserved for playhead per Phase 1 UI-SPEC).
_BRUSH_UNTOUCHED = pg.mkBrush(QColor(0x6B, 0x8F, 0xA8, 48))
_HOVER_UNTOUCHED = pg.mkBrush(QColor(0x6B, 0x8F, 0xA8, 72))
_PEN_UNTOUCHED = pg.mkPen(QColor(0x6B, 0x8F, 0xA8, 200), width=1)

# Plan 02 — Keeper (warm-yellow) + Trash (greyed) per-state tokens.
_BRUSH_KEEPER = pg.mkBrush(QColor(0xFF, 0xC8, 0x57, 56))
_HOVER_KEEPER = pg.mkBrush(QColor(0xFF, 0xC8, 0x57, 88))
_PEN_KEEPER = pg.mkPen(QColor(0xFF, 0xC8, 0x57, 220), width=1)

_BRUSH_TRASH = pg.mkBrush(QColor(0x3F, 0x3F, 0x3F, 144))
_HOVER_TRASH = pg.mkBrush(QColor(0x3F, 0x3F, 0x3F, 168))
_PEN_TRASH = pg.mkPen(QColor(0x5C, 0x63, 0x6B, 200), width=1)

# Draft (in-progress Shift+drag) — Plan 01 styling preserved.
_BRUSH_DRAFT = pg.mkBrush(QColor(0xFF, 0xC8, 0x57, 64))
_PEN_DRAFT = pg.mkPen(QColor(0xFF, 0xC8, 0x57, 220), width=1)

# Per-state 3-tuple (brush, hover_brush, pen) — single source of truth
# for ``_apply_state``. RESEARCH §Pitfall #2 mandates all three setters
# in concert: ``setBrush`` alone leaves the edges at the previous color.
_STATE_STYLES: dict[str, tuple] = {
    "untouched": (_BRUSH_UNTOUCHED, _HOVER_UNTOUCHED, _PEN_UNTOUCHED),
    "keeper": (_BRUSH_KEEPER, _HOVER_KEEPER, _PEN_KEEPER),
    "trash": (_BRUSH_TRASH, _HOVER_TRASH, _PEN_TRASH),
}

_VALID_STATES = frozenset(_STATE_STYLES.keys())


def _merge_intervals(
    ranges: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping or touching half-open intervals (Plan 03-03).

    Pure-Python helper consumed by :meth:`RegionsOverlay.trash_minus_keepers`
    (D-A2-5 Keeper-punch-through). Plan 04's naming_resolver may reuse this
    helper for region-set dominant-trait aggregation, so it stays module-level
    rather than nested inside the method.

    Input MUST be sorted ascending by start (the caller's responsibility —
    :meth:`keeper_trash_ranges` already sorts). Half-open semantics:
    intervals ``(a, b)`` and ``(b, c)`` merge to ``(a, c)``. This matches
    the audio-thread skip-range scan in
    :meth:`PlaybackEngine._callback` which uses ``start <= pos < end``.
    """
    if not ranges:
        return []
    out: list[tuple[float, float]] = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def _apply_state(region: "ResizeOnlyRegion", state: str) -> None:
    """Apply per-state brush + hover brush + pen in concert.

    RESEARCH §Pitfall #2: ``setBrush`` alone leaves the edges at the
    previous color, producing a visual mismatch. ``setPen`` on the
    parent ``LinearRegionItem`` does NOT exist on PyQtGraph 0.13.7
    (verified via attribute introspection); we set the pen on each
    child :class:`pyqtgraph.InfiniteLine` (``region.lines[0]`` and
    ``region.lines[1]``) directly. ``setHoverBrush`` updates the
    visual when the user hovers the region.
    """
    brush, hover_brush, pen = _STATE_STYLES[state]
    region.setBrush(brush)
    region.setHoverBrush(hover_brush)
    # LinearRegionItem doesn't expose setPen — propagate to each child line.
    for line in region.lines:
        line.setPen(pen)


class ResizeOnlyRegion(pg.LinearRegionItem):
    """``LinearRegionItem`` subclass that disables body-drag.

    UI-SPEC §Interaction Contract (Region edit affordances):

        "Body-drag (drag the middle of a region to shift it) is **disabled**
         — subclass LinearRegionItem and override mouseDragEvent to no-op."

    Edge-drag (resize) remains enabled via ``self.lines[0]`` and
    ``self.lines[1]`` (the child :class:`pyqtgraph.InfiniteLine` instances) —
    those still fire their own ``mouseDragEvent`` when the user grabs an
    edge handle.

    Plan 02 additions:

    * Hover targeting via ``hoverEnterEvent`` / ``hoverLeaveEvent`` —
      keeps :attr:`RegionsOverlay.hovered_region_id` in sync so the K/T/U
      Edit-menu actions know which region to mutate.
    * Right-click context menu via :meth:`getContextMenus` returning a
      ``list[QMenu]`` (RESEARCH §Pitfall #8 — list, not single QMenu).
    * ``_current_state`` attribute carries the region's live state so
      :meth:`RegionsOverlay.regions_data` can serialize the right value
      to the sidecar.
    * ``_created_at`` / ``_note`` carry the schema fields end-to-end —
      loaded regions preserve their original metadata; new regions get
      a fresh ISO timestamp + empty note.
    """

    def __init__(
        self,
        region_id: str,
        *args,
        overlay: "RegionsOverlay | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.region_id = region_id
        self._overlay = overlay
        # Plan 02 — region state tracker (untouched|keeper|trash). The
        # owning :meth:`RegionsOverlay.set_state` updates this alongside
        # re-applying the styles, so :meth:`RegionsOverlay.regions_data`
        # can report the live state at serialize time.
        self._current_state: str = "untouched"
        # Schema-field carriers — populated by ``set_regions`` for loaded
        # regions or by ``commit_draft`` for fresh ones.
        self._created_at: str = ""
        self._note: str = ""
        # Phase 7 Plan 07-02 Task 4 — per-keeper mastering chain config
        # (D-19 sidecar field). None = no mastering applied; dict =
        # mastered cache override applies on export. Carried on the
        # widget so ``regions_data()`` round-trips the field through
        # save_sidecar verbatim.
        self._mastering: dict | None = None
        # Phase 8 Plan 08-06 Task 1 — per-keeper YouTube video id
        # (D-30 sidecar field). None = never uploaded; str = the id
        # returned by the most recent successful upload. Carried on
        # the widget so ``regions_data()`` round-trips the field
        # through ``save_sidecar`` verbatim (mirrors the
        # ``_mastering`` carrier above).
        self._youtube_video_id: str | None = None
        # quick-260621-gfq — per-keeper normalize state now lives INSIDE
        # ``_mastering['normalize']`` (single source of truth), not in
        # standalone widget attrs. The setter/getter route through the
        # ``_mastering`` carrier above (seeding it from the session snapshot
        # when None per the MIGRATION-GAP GUARD).
        # RESEARCH §Pitfall #11 — hover events do NOT fire by default on
        # QGraphicsItems; the opt-in is required for ``hoverEnterEvent`` /
        # ``hoverLeaveEvent`` to receive dispatches from the scene.
        self.setAcceptHoverEvents(True)
        # 03-05b — context menu must outlive the popup() call. Without an
        # instance-attribute or parent QWidget, PySide6 can drop the
        # Python shadow before Qt paints (research 03-05b §Cause A). Build
        # lazily on first right-click, reuse thereafter, re-parent each
        # popup so Wayland gets a transient parent (research §Cause B).
        self._context_menu: QMenu | None = None

    def mouseDragEvent(self, ev):  # noqa: N802 — Qt method name
        # Body-drag disabled (CONTEXT D-A1-4 + UI-SPEC §Interaction Contract).
        # Edge-drag is handled independently by self.lines[0] / self.lines[1].
        ev.ignore()

    # ---------------------------------------------------- hover target
    def hoverEnterEvent(self, ev):  # noqa: N802 — Qt method name
        """Flip the overlay's hovered_region_id to this region's id.

        The ``super()`` call is wrapped in try/except because PyQtGraph's
        LinearRegionItem inherits Qt's strictly-typed hoverEnterEvent —
        tests that synthesize hover events via plain objects (without
        constructing a QGraphicsSceneHoverEvent) would raise TypeError
        on the super() call but we still want to exercise our overlay
        side-effects.
        """
        if self._overlay is not None:
            self._overlay.hovered_region_id = self.region_id
            self._overlay.hover_changed.emit(self.region_id)
        try:
            super().hoverEnterEvent(ev)
        except TypeError:
            # Synthetic event (tests) — PyQtGraph's base accepts only
            # QGraphicsSceneHoverEvent. The overlay-side bookkeeping
            # above is what we actually need.
            pass

    def hoverLeaveEvent(self, ev):  # noqa: N802 — Qt method name
        """Clear hovered_region_id iff this region was the hovered one."""
        if (
            self._overlay is not None
            and self._overlay.hovered_region_id == self.region_id
        ):
            self._overlay.hovered_region_id = None
            self._overlay.hover_changed.emit(None)
        try:
            super().hoverLeaveEvent(ev)
        except TypeError:
            pass

    # --------------------------------------------- scene-level event routing
    # The three overrides below (hoverEvent, mouseClickEvent, raiseContextMenu)
    # are the gap-closure for UAT Test 3 — Plan 03-05. They mirror the
    # canonical ROI pattern (pyqtgraph/graphicsItems/ROI.py:728-806). The
    # base ``LinearRegionItem`` (LinearRegionItem.py:332-345) does NOT route
    # idle RightButton clicks anywhere: its ``mouseClickEvent`` only acts
    # when ``self.moving == True`` (to cancel an in-progress drag), and its
    # ``hoverEvent`` never claims RightButton on the scene's click-accept
    # bookkeeping. Without all three of these overrides, ``getContextMenus``
    # below is dead code at runtime — the scene never reaches it.
    #
    # These are scene-level methods (QGraphicsItem hoverEvent /
    # mouseClickEvent), NOT the Qt-widget-level
    # ``hoverEnterEvent`` / ``hoverLeaveEvent`` overrides above. The two
    # layers coexist: our enter/leave overrides flip the overlay's
    # ``hovered_region_id`` for the Edit-menu K/T/U shortcuts; the scene
    # ``hoverEvent`` below tells PyQtGraph's scene that this item claims
    # RightButton, which is how ``mouseClickEvent`` later receives the
    # right-click for the popup.
    def hoverEvent(self, ev):  # noqa: N802 — PyQtGraph scene API
        """Accept RightButton clicks during hover (gap-closure for UAT Test 3).

        Mirrors ROI.py:728-744. PyQtGraph's GraphicsScene needs the item
        to claim RightButton during hover so ``sendClickEvent`` later
        finds this item as the acceptedItem for right-clicks. WITHOUT
        this override, :meth:`mouseClickEvent` below never fires for
        RightButton on idle regions because
        ``LinearRegionItem.hoverEvent`` (LinearRegionItem.py:341) never
        calls ``ev.acceptClicks(RightButton)``.

        We delegate to ``super().hoverEvent(ev)`` first to preserve
        LinearRegionItem's existing mouseHover brush-swap behaviour
        (LinearRegionItem.py:341-345 → setMouseHover), then add the
        RightButton click-accept. The try/except guards the synthetic-
        event tests (mirrors :meth:`hoverEnterEvent` pattern).
        """
        try:
            super().hoverEvent(ev)
        except TypeError:
            pass
        if not getattr(ev, "isExit", lambda: True)():
            try:
                ev.acceptClicks(Qt.MouseButton.RightButton)
            except (AttributeError, TypeError):
                pass

    def mouseClickEvent(self, ev):  # noqa: N802 — PyQtGraph scene API
        """Route right-click → raiseContextMenu (gap-closure for UAT Test 3).

        ``LinearRegionItem.mouseClickEvent`` (LinearRegionItem.py:332-339)
        only handles RightButton mid-drag (to cancel by reverting the
        line positions); idle right-clicks are silently dropped. We add
        the canonical ROI pattern (ROI.py:801-806): if RightButton AND
        not currently moving, raise our context menu and accept.

        Why ``super()`` first then our right-click handling: when the
        user right-clicks DURING a drag (rare — mid-resize cancel),
        ``LinearRegionItem.mouseClickEvent`` reverts line positions AND
        emits ``sigRegionChangeFinished``, which our overlay's
        ``_on_region_changed`` translates into a ``regions_changed``
        emission → sidecar write. That is correct — the user explicitly
        cancelled mid-drag, and re-persisting the reverted state is the
        right outcome. For an idle right-click, ``super()`` is a no-op
        and the RightButton branch below fires cleanly.
        """
        try:
            super().mouseClickEvent(ev)
        except TypeError:
            pass
        try:
            is_right = ev.button() == Qt.MouseButton.RightButton
        except (AttributeError, TypeError):
            return
        if is_right and not self.moving:
            self.raiseContextMenu(ev)
            try:
                ev.accept()
            except (AttributeError, TypeError):
                pass

    def raiseContextMenu(self, ev):  # noqa: N802 — PyQtGraph scene API
        """Build + popup the per-region context QMenu (gap-closure for UAT Test 3).

        Mirrors ROI.py:772-778 exactly. The QMenu instance comes from
        our own :meth:`getContextMenus` below so the existing lambda
        action-trigger plumbing (regions_overlay.py — Mark as
        Keeper/Trash/Unmark + Export MP3/WAV + Delete) dispatches
        correctly. No second copy, no re-parenting.

        ``scene().addParentContextMenus(self, menu, ev)`` walks the
        scene-item tree for any parent-supplied menus
        (PlotItem / ViewBox menus). On this plot we set
        ``setMenuEnabled(False)`` (waveform_view.py:246) so the parent
        walk finds none — but we still call ``addParentContextMenus``
        to honor the PyQtGraph idiom in case a future plot adds a
        parent menu.
        """
        menus = self.getContextMenus(ev)
        if not menus:
            return
        menu = menus[0]
        scene = self.scene()
        if scene is not None:
            # Honor the PyQtGraph idiom (ROI.py:776) — but the scene's
            # built-in ``Export...`` action survives even when our PlotItem
            # has ``setMenuEnabled(False)`` (waveform_view.py:246). We DO
            # NOT want that ``Export...`` (or any other parent-walked
            # actions) in the user-facing region menu — UI-SPEC §Region
            # context menu locks the six-action ordering and the user
            # explicitly does NOT want the scene's stock chain. So we
            # snapshot the menu's own actions BEFORE the parent walk,
            # call addParentContextMenus for idiom compliance, then trim
            # any newly-added actions back out.
            existing_actions = list(menu.actions())
            menu = scene.addParentContextMenus(self, menu, ev)
            for act in list(menu.actions()):
                if act not in existing_actions:
                    menu.removeAction(act)
            # 03-05b §Cause B — Wayland compositor requires a transient
            # parent QWindow to accept a popup. Re-parent the menu to the
            # scene's first view each popup; the view is the long-lived
            # QGraphicsView owning the plot. Harmless on X11/macOS/Windows.
            views = scene.views()
            if views:
                try:
                    menu.setParent(views[0], menu.windowFlags())
                except (AttributeError, TypeError):
                    pass
        try:
            pos = ev.screenPos()
            menu.popup(QPoint(int(pos.x()), int(pos.y())))
        except (AttributeError, TypeError):
            return

    # --------------------------------------------- right-click context menu
    def getContextMenus(self, event):  # noqa: N802 — PyQtGraph API
        """Return a list[QMenu] for PyQtGraph's right-click dispatch.

        Lazy-build the menu on first call and store on ``self._context_menu``
        so two consecutive right-clicks reuse the SAME QMenu (research
        03-05b §Cause A — PySide6 GC'd parent-less fresh QMenus before
        Qt could paint them; mirrors ROI.py:160 / 781-789). Per-call we
        only refresh the keeper-only enabled state on the two Export
        actions; everything else (lambdas, ordering, separators) is built
        once and reused.
        """
        if self._context_menu is None:
            self._context_menu = self._build_context_menu()
        # Refresh keeper-only Export enabled state — the only per-call
        # mutation needed; ordering and lambdas are stable.
        for act in self._context_menu.actions():
            if act.text().startswith("Export this region as"):
                act.setEnabled(self._current_state == "keeper")
        return [self._context_menu]

    def _build_context_menu(self) -> QMenu:
        """Build the six-action QMenu once. Lambdas close over stable
        per-region values via default-arg binding (Phase 1 LEARNINGS
        §"Late-binding closure capture")."""
        rid = self.region_id
        overlay = self._overlay
        menu = QMenu()
        # Order matches UI-SPEC §Copywriting + Edit menu items.
        action_keeper = QAction("Mark as Keeper", menu)
        action_keeper.triggered.connect(
            lambda _checked=False, _rid=rid, _ov=overlay: (
                _ov.set_state(_rid, "keeper") if _ov is not None else None
            )
        )
        menu.addAction(action_keeper)

        action_trash = QAction("Mark as Trash", menu)
        action_trash.triggered.connect(
            lambda _checked=False, _rid=rid, _ov=overlay: (
                _ov.set_state(_rid, "trash") if _ov is not None else None
            )
        )
        menu.addAction(action_trash)

        action_unmark = QAction("Unmark", menu)
        action_unmark.triggered.connect(
            lambda _checked=False, _rid=rid, _ov=overlay: (
                _ov.set_state(_rid, "untouched") if _ov is not None else None
            )
        )
        menu.addAction(action_unmark)

        menu.addSeparator()

        # Plan 03-04b — Export actions (D-A4-4 LOCKED dual-format). BOTH
        # formats surfaced from day one; the user picks MP3 (sharing) OR
        # WAV (DAW handoff) from the same menu. Enabled ONLY when the
        # region's state is "keeper" — toggled per popup in getContextMenus.
        action_export_mp3 = QAction("Export this region as MP3…", menu)
        action_export_mp3.triggered.connect(
            lambda _checked=False, _rid=rid, _ov=overlay: (
                _ov.export_requested.emit(_rid, "mp3")
                if _ov is not None else None
            )
        )
        menu.addAction(action_export_mp3)

        action_export_wav = QAction("Export this region as WAV…", menu)
        action_export_wav.triggered.connect(
            lambda _checked=False, _rid=rid, _ov=overlay: (
                _ov.export_requested.emit(_rid, "wav")
                if _ov is not None else None
            )
        )
        menu.addAction(action_export_wav)

        menu.addSeparator()

        action_delete = QAction("Delete region", menu)
        action_delete.triggered.connect(
            lambda _checked=False, _rid=rid, _ov=overlay: (
                _ov.delete(_rid) if _ov is not None else None
            )
        )
        menu.addAction(action_delete)

        return menu


class RegionsOverlay(QObject):
    """Region overlay on the WaveformView's PlotItem.

    Owns:
        * A dict of ``ResizeOnlyRegion`` keyed by ``region_id`` — one
          LinearRegionItem subclass per saved region. The owning
          :class:`MainWindow` reads ``self._regions`` to drive the
          KeepersSidebar (Plan 02).
        * One draft ``LinearRegionItem`` (Plan 01 only — Plan 02 swaps to
          a per-state brush). Visible only during an in-progress
          Shift+drag, converted to a confirmed ``Region`` on
          MouseButtonRelease.

    Signals:
        regions_changed: Emitted after every commit or sigRegionChangeFinished
            or :meth:`set_state` / :meth:`delete` mutation so MainWindow
            can call ``sidecar_cache.save_sidecar(...)``.
        hover_changed(object): Emitted on hover enter (payload = region_id
            str) and hover leave (payload = ``None``). MainWindow Task 3
            consumes this to enable/disable the Edit-menu K/T/U/Delete
            actions.

    Construction args:
        plot_item: ``pg.PlotItem`` — the WaveformView's main plot,
            ``view.waveform_plot``. ResizeOnlyRegion instances are added
            to / removed from this PlotItem.
        duration_s_provider: zero-arg callable returning the current
            source duration in seconds. Used as ``bounds`` for newly
            created LinearRegionItem instances so drags cannot exceed
            the source range (RESEARCH §Pitfall #7). Read LAZILY (at
            region-create time) so a MainWindow can construct the
            overlay before any file is open (duration may be 0.0).
    """

    regions_changed = Signal()
    # Plan 02 — hover-target signal. Payload = new hovered region_id (str)
    # on enter, ``None`` on leave. MainWindow Task 3 connects to enable /
    # disable Edit-menu K/T/U/Delete actions.
    hover_changed = Signal(object)
    # Plan 03-04b — export request from the right-click context menu.
    # Payload = (region_id, fmt) where fmt is "mp3" or "wav". MainWindow's
    # ``_on_export_region_requested`` slot dispatches on the fmt arg.
    # Both formats are first-class per CONTEXT D-A4-4 LOCKED dual-format.
    export_requested = Signal(str, str)

    def __init__(
        self,
        plot_item: pg.PlotItem,
        duration_s_provider: Callable[[], float],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._plot_item = plot_item
        self._duration_s_provider = duration_s_provider
        # Per-region widget registry. Dict for O(1) lookup by id (Plan 02
        # hover-target wiring needs this).
        self._regions: dict[str, ResizeOnlyRegion] = {}
        # Draft region during in-progress Shift+drag. None when idle.
        self._draft: pg.LinearRegionItem | None = None
        self._draft_start_x: float | None = None
        # Plan 02 — currently-hovered region id (str) or None when no
        # region is under the cursor. Maintained by per-region
        # ``hoverEnterEvent`` / ``hoverLeaveEvent`` overrides on
        # ResizeOnlyRegion.
        self.hovered_region_id: str | None = None

    # ---------------------------------------------------------------- helpers
    def _bounds(self) -> list[float]:
        """Return ``[0.0, duration_s]`` clamping bounds for new regions.

        Reads duration LAZILY so the overlay can be constructed before any
        file is open (duration may be 0.0). When duration is 0.0 we omit
        bounds to avoid pinning every region to zero-width.
        """
        duration = float(self._duration_s_provider() or 0.0)
        if duration > 0.0:
            return [0.0, duration]
        return [0.0, float("inf")]

    def _make_region_widget(self, region: Region) -> ResizeOnlyRegion:
        """Build a styled ``ResizeOnlyRegion`` for the given Region data.

        Plan 02 — per-state styling. The widget is constructed with the
        untouched brush; ``_apply_state`` then swaps the trio
        (brush/hover/pen) to the correct state. ``_current_state`` /
        ``_created_at`` / ``_note`` are populated on the widget so
        :meth:`regions_data` round-trips faithfully.
        """
        widget = ResizeOnlyRegion(
            region_id=region.id,
            overlay=self,
            values=(region.start_sec, region.end_sec),
            orientation="vertical",
            brush=_BRUSH_UNTOUCHED,
            hoverBrush=_HOVER_UNTOUCHED,
            movable=True,
            bounds=self._bounds(),
            swapMode="sort",
        )
        # Apply the right per-state styling (brush+hover+pen in concert).
        state = region.state if region.state in _VALID_STATES else "untouched"
        _apply_state(widget, state)
        widget._current_state = state
        widget._created_at = region.created_at
        widget._note = region.note
        # Phase 7 Plan 07-02 Task 4 — preserve the loaded mastering dict
        # (or None) so subsequent save_sidecar calls round-trip the field
        # without losing it through ``regions_data()``.
        widget._mastering = region.mastering
        # Phase 8 Plan 08-06 Task 1 — preserve the loaded
        # youtube_video_id (or None) for the same reason; mirrors the
        # ``_mastering`` carrier above. ``_on_youtube_upload_finished``
        # in MainWindow calls :meth:`set_youtube_video_id` after a
        # successful upload so a subsequent ``save_sidecar`` persists
        # the id (D-30).
        widget._youtube_video_id = region.youtube_video_id
        # quick-260621-gfq — per-keeper normalize is carried inside
        # ``_mastering['normalize']`` (set above from region.mastering); no
        # standalone normalize attrs to seed any more.
        # On commit of an edge-drag, emit regions_changed so the
        # MainWindow can atomic-save the sidecar.
        widget.sigRegionChangeFinished.connect(self._on_region_changed)
        return widget

    # ----------------------------------------------------------- public API
    def set_regions(self, regions: list[Region]) -> None:
        """Replace the on-screen region set from a fresh data list.

        Clears existing items (with the ``removeItem`` + ``deleteLater``
        cleanup discipline from Phase 1 LEARNINGS) then rebuilds from the
        given list. Does NOT emit ``regions_changed`` — this is a
        load-time call from MainWindow._open_file, not a user mutation.
        """
        # Clear existing.
        for widget in list(self._regions.values()):
            self._plot_item.removeItem(widget)
            try:
                widget.deleteLater()
            except RuntimeError:
                # Already deleted by Qt — defensive.
                pass
        self._regions.clear()
        # Reset hover state — a stale id from a deleted region is a bug
        # vector for Edit-menu actions.
        self.hovered_region_id = None
        # Rebuild.
        for r in regions:
            widget = self._make_region_widget(r)
            self._plot_item.addItem(widget)
            self._regions[r.id] = widget

    def start_draft(self, x: float) -> None:
        """Start an in-progress Shift+drag draft at data-x ``x``.

        Builds a translucent yellow ``LinearRegionItem`` (Keeper-warm
        signal — UI-SPEC §Color) and adds it to the plot. Subsequent
        :meth:`update_draft` calls re-set the region range; the
        :meth:`commit_draft` call on MouseButtonRelease converts it to
        a persisted ``Region``.
        """
        # Tear down any stale draft (defensive — should not happen in
        # normal usage but a missed release event must not strand a
        # draft on the plot forever).
        if self._draft is not None:
            self._discard_draft()
        self._draft = pg.LinearRegionItem(
            values=(x, x),
            orientation="vertical",
            brush=_BRUSH_DRAFT,
            pen=_PEN_DRAFT,
            movable=False,
            bounds=self._bounds(),
        )
        self._plot_item.addItem(self._draft)
        self._draft_start_x = float(x)

    def update_draft(self, x: float) -> None:
        """Update the in-progress draft range to span (start_x, x), sorted."""
        if self._draft is None or self._draft_start_x is None:
            return
        lo, hi = sorted((self._draft_start_x, float(x)))
        self._draft.setRegion((lo, hi))

    def commit_draft(
        self, x: float, min_width_sec: float = 0.001
    ) -> Region | None:
        """Convert the draft to a persisted Region or discard it.

        Returns the new :class:`Region` on commit, or ``None`` if the
        draft was discarded (drag width below ``min_width_sec``).

        Emits ``regions_changed`` on commit. The MainWindow's connected
        slot atomic-saves the sidecar JSON.
        """
        if self._draft is None or self._draft_start_x is None:
            return None
        start_x = float(self._draft_start_x)
        end_x = float(x)
        lo, hi = sorted((start_x, end_x))
        # Dead-zone — a Shift+click without movement should not create a
        # region. The pixel-side equivalent (SEEK_THRESHOLD_PX in
        # WaveformView) protects from below-threshold pixel deltas; this
        # data-space guard protects from zero-data-x drags too.
        if (hi - lo) < min_width_sec:
            self._discard_draft()
            return None
        # Build the persisted Region. UUID4 hex (32 chars) — single
        # source of truth for region ids.
        region = Region(
            id=uuid.uuid4().hex,
            start_sec=lo,
            end_sec=hi,
            # Selecting a region in the viewport marks it as a Keeper
            # right away — the common case. Right-click → Unmark still
            # resets it to "untouched" when the user wants that.
            state="keeper",
            created_at=datetime.now().isoformat(),
            note="",
        )
        # Tear down the draft visual and install the persisted widget.
        self._discard_draft()
        widget = self._make_region_widget(region)
        self._plot_item.addItem(widget)
        self._regions[region.id] = widget
        self.regions_changed.emit()
        return region

    def get_region(self, region_id: str) -> ResizeOnlyRegion | None:
        """Return the per-region widget for ``region_id`` or ``None`` if absent.

        WR-04 — public O(1) accessor so MainWindow slots
        (``_on_keeper_jump`` / ``_on_keeper_play`` / ``_on_keeper_note_changed``
        / ``_on_export_region_requested``) do not reach into the private
        ``_regions`` dict. Returns the widget itself (not a copy of
        the :class:`Region` data) so callers can read
        ``widget.getRegion()`` to pick up live edge-drag mutations.
        The alternative public accessor :meth:`regions_data` returns a
        fresh ``list[Region]``, which is O(N) per call and a bad fit for
        single-id lookups.
        """
        return self._regions.get(region_id)

    def set_state(self, region_id: str, state: str) -> None:
        """Mutate a region's state — re-apply styling + emit regions_changed.

        Validates ``state`` against the closed enum ``_VALID_STATES``
        (untouched|keeper|trash). Invalid state raises :class:`ValueError`.

        Defensive against ``region_id`` not in ``_regions`` (the region
        may have been deleted concurrently) — silent no-op.

        No-op when the requested state equals the current state — this
        prevents redundant sidecar writes when a context-menu action is
        triggered against a region already in that state.
        """
        if state not in _VALID_STATES:
            raise ValueError(
                f"invalid region state: {state!r} "
                f"(expected one of {sorted(_VALID_STATES)})"
            )
        region = self._regions.get(region_id)
        if region is None:
            return
        if region._current_state == state:
            return
        _apply_state(region, state)
        region._current_state = state
        self.regions_changed.emit()

    def delete(self, region_id: str) -> None:
        """Remove a region from the plot + dict + clear hover if matching.

        Defensive against ``region_id`` not in ``_regions`` (silent
        no-op). RESEARCH §Pitfall #4 — clears the stale hovered_region_id
        when the deleted region is the currently-hovered one; otherwise
        a stale Edit-menu action would target a deleted region.
        """
        region = self._regions.get(region_id)
        if region is None:
            return
        self._plot_item.removeItem(region)
        try:
            region.deleteLater()
        except RuntimeError:
            pass
        del self._regions[region_id]
        if self.hovered_region_id == region_id:
            self.hovered_region_id = None
        self.regions_changed.emit()

    def set_state_of_hovered(self, state: str) -> None:
        """Convenience — set_state on the currently-hovered region or no-op.

        Consumed by MainWindow's Edit-menu K/T/U slots. The slot also
        checks for QLineEdit focus (defense-in-depth against Pitfall #5),
        so this method only needs the hovered-id guard.
        """
        if self.hovered_region_id is None:
            return
        self.set_state(self.hovered_region_id, state)

    def delete_hovered(self) -> None:
        """Convenience — delete the currently-hovered region or no-op."""
        if self.hovered_region_id is None:
            return
        self.delete(self.hovered_region_id)

    def set_mastering(self, region_id: str, mastering: dict | None) -> None:
        """Update the per-region mastering chain config.

        Phase 7 Plan 07-02 Task 4 — MainWindow calls this after the
        MasteringDialog's Apply has been processed so a subsequent
        ``regions_data() -> save_sidecar`` round-trips the new dict.
        No-op if ``region_id`` is unknown (defensive — overlay teardown
        can race with a stray Apply signal in test code).
        """
        widget = self._regions.get(region_id)
        if widget is None:
            return
        widget._mastering = mastering

    def get_mastering(self, region_id: str) -> dict | None:
        """Read the per-region mastering chain config (or None if absent).

        Phase 7 Plan 07-02 Task 4 — used by MainWindow when opening the
        MasteringDialog so the dialog receives the keeper's persisted
        dict (NOT a fresh session snapshot — D-04).
        """
        widget = self._regions.get(region_id)
        if widget is None:
            return None
        return getattr(widget, "_mastering", None)

    def set_youtube_video_id(self, region_id: str, video_id: str | None) -> None:
        """Update the per-region YouTube video id (Phase 8 D-30).

        MainWindow's ``_on_youtube_upload_finished`` calls this with
        the id returned by the successful upload so a subsequent
        ``regions_data() -> save_sidecar`` round-trips the new value.
        No-op if ``region_id`` is unknown (defensive — overlay teardown
        can race with a stray finished signal in test code).
        """
        widget = self._regions.get(region_id)
        if widget is None:
            return
        widget._youtube_video_id = video_id

    def get_youtube_video_id(self, region_id: str) -> str | None:
        """Read the per-region YouTube video id (or None if never uploaded).

        Phase 8 D-30 — used by future "Open on YouTube" flows that
        check whether a keeper has an upload record.
        """
        widget = self._regions.get(region_id)
        if widget is None:
            return None
        return getattr(widget, "_youtube_video_id", None)

    def set_normalize(
        self, region_id: str, enabled: bool, target_db: float = 0.0
    ) -> None:
        """Update the per-region normalize state (quick-260621-gfq).

        Single source of truth: ``widget._mastering['normalize']``. The
        keeper-row Normalize toggle and the Mastering dock both drive this
        same per-keeper state. When the widget has no mastering dict yet
        (a legacy ``mastering=None`` keeper), seed it from the session
        snapshot — NEVER an empty ``{}`` — so toggling normalize ON produces
        a full session-default mastering dict with normalize enabled
        (MIGRATION-GAP GUARD; matches the dock's session defaults).

        No-op if ``region_id`` is unknown (defensive — mirrors
        :meth:`set_mastering`).
        """
        widget = self._regions.get(region_id)
        if widget is None:
            return
        mastering = getattr(widget, "_mastering", None)
        if not isinstance(mastering, dict):
            # Authoritative seed branch — full session-snapshot dict so the
            # keeper gains the same default chain the dock shows (D-04).
            from marmelade.audio.mastering.chain import (
                load_session_chain_snapshot,
            )

            mastering = load_session_chain_snapshot()
            widget._mastering = mastering
        norm = mastering.setdefault("normalize", {})
        norm["enabled"] = bool(enabled)
        norm["target_db"] = float(target_db)

    def get_normalize(self, region_id: str) -> tuple[bool, float]:
        """Read the per-region normalize state (quick-260621-gfq).

        Returns ``(enabled, target_db)`` read from
        ``widget._mastering['normalize']``. Safe-default ``(False, 0.0)``
        when ``region_id`` is unknown OR the keeper has no normalize entry.
        """
        widget = self._regions.get(region_id)
        if widget is None:
            return (False, 0.0)
        mastering = getattr(widget, "_mastering", None)
        norm = (mastering or {}).get("normalize", {}) if mastering else {}
        if not isinstance(norm, dict):
            norm = {}
        return (
            bool(norm.get("enabled", False)),
            float(norm.get("target_db", 0.0)),
        )

    def regions_data(self) -> list[Region]:
        """Return a fresh ``list[Region]`` matching the current overlay state.

        Walks ``_regions.values()`` reading each widget's current
        ``getRegion()`` (which reflects edge-drag mutations) plus the
        widget's stored ``region_id`` / ``_current_state`` / ``_created_at``
        / ``_note``. Used by :class:`MainWindow` to build the payload for
        ``sidecar_cache.save_sidecar`` on every mutation.
        """
        out: list[Region] = []
        for region_id, widget in self._regions.items():
            start_s, end_s = widget.getRegion()
            # Guarantee start < end on read — defensive against any
            # corner case where edge-drag flipped past the other edge
            # (PyQtGraph's swapMode='sort' should prevent this).
            lo, hi = (float(start_s), float(end_s))
            if lo > hi:
                lo, hi = hi, lo
            if hi - lo <= 0.0:
                # Skip zero-width — should never happen, defensive.
                continue
            out.append(
                Region(
                    id=region_id,
                    start_sec=lo,
                    end_sec=hi,
                    state=widget._current_state,
                    created_at=widget._created_at,
                    note=widget._note,
                    mastering=getattr(widget, "_mastering", None),
                    youtube_video_id=getattr(
                        widget, "_youtube_video_id", None
                    ),
                )
            )
        return out

    # ----------------------------- Plan 03-03 — Trash playback skip helpers
    def keeper_trash_ranges(
        self,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return ``(keepers, trash)`` ranges sorted ascending by start.

        Each list is sorted by start but NOT merged — regions of the
        same state may still overlap each other (D-A2-5 allows multiple
        independent regions over the same range). Merging is the
        consumer's job (see :meth:`trash_minus_keepers` below).

        Untouched regions and zero-width (or negative-width) regions are
        excluded — they have no effect on playback skip or heatmap mask.
        """
        keepers: list[tuple[float, float]] = []
        trash: list[tuple[float, float]] = []
        for region in self._regions.values():
            s, e = region.getRegion()
            if e <= s:
                continue
            state = region._current_state
            if state == "keeper":
                keepers.append((float(s), float(e)))
            elif state == "trash":
                trash.append((float(s), float(e)))
        keepers.sort()
        trash.sort()
        return (keepers, trash)

    def trash_minus_keepers(self) -> list[tuple[float, float]]:
        """Compute Trash ranges with Keeper-punch-through subtracted (D-A2-5).

        Pipeline:

        1. Pull per-state ranges via :meth:`keeper_trash_ranges`.
        2. Merge overlapping Trash ranges (so a downstream consumer
           sees a flat, non-overlapping list).
        3. Subtract every Keeper from each merged Trash range — a
           Keeper inside Trash splits the Trash into two carved
           sub-ranges; a Keeper engulfing a Trash eliminates it;
           a Keeper at an edge trims the Trash.

        Returns a sorted, non-overlapping list of ``(start_sec, end_sec)``
        tuples. Consumed by:

        * :meth:`PlaybackEngine.set_skip_ranges` — audio-thread skip
          scan in ``_callback``.
        * :meth:`HeatmapLaneView.render` ``trash_mask`` kwarg — visual
          blanking of Trash ranges on the heatmap lane.

        Empty list when there are no Trash regions OR Keepers fully
        engulf every Trash range.
        """
        keepers, trash = self.keeper_trash_ranges()
        if not trash:
            return []
        merged_trash = _merge_intervals(trash)
        if not keepers:
            return merged_trash
        merged_keepers = _merge_intervals(keepers)
        result: list[tuple[float, float]] = []
        for ts, te in merged_trash:
            # Carve every overlapping Keeper out of this Trash range.
            carved: list[tuple[float, float]] = [(ts, te)]
            for ks, ke in merged_keepers:
                if ke <= ts or ks >= te:
                    # Keeper is disjoint from this Trash range.
                    continue
                new_carved: list[tuple[float, float]] = []
                for cs, ce in carved:
                    if ke <= cs or ks >= ce:
                        # Keeper is disjoint from this carved sub-range —
                        # preserve it as-is.
                        new_carved.append((cs, ce))
                        continue
                    # Subtract [ks, ke) from [cs, ce).
                    if ks > cs:
                        new_carved.append((cs, ks))
                    if ke < ce:
                        new_carved.append((ke, ce))
                    # else: the Keeper consumes the full carved slice.
                carved = new_carved
            result.extend(carved)
        return sorted(result)

    def clear(self) -> None:
        """Full teardown — wipe every region + any in-progress draft.

        Called from MainWindow's new-file-open preamble. Same
        ``removeItem`` + ``deleteLater`` discipline as
        :meth:`set_regions`.
        """
        for widget in list(self._regions.values()):
            self._plot_item.removeItem(widget)
            try:
                widget.deleteLater()
            except RuntimeError:
                pass
        self._regions.clear()
        self.hovered_region_id = None
        self._discard_draft()

    # --------------------------------------------------------------- private
    def _discard_draft(self) -> None:
        """Tear down the in-progress draft (if any)."""
        if self._draft is None:
            return
        self._plot_item.removeItem(self._draft)
        try:
            self._draft.deleteLater()
        except RuntimeError:
            pass
        self._draft = None
        self._draft_start_x = None

    def _on_region_changed(self) -> None:
        """Slot wired to every ResizeOnlyRegion.sigRegionChangeFinished.

        Emits ``regions_changed`` so MainWindow can save the sidecar.
        The in-memory dict-of-widgets is the source of truth for the
        start/end values — :meth:`regions_data` reads them on demand.
        """
        self.regions_changed.emit()


# ---------------------------------------------------------------------------
# quick-260701-jc5 — MarkersOverlay (MARK-03)
# ---------------------------------------------------------------------------

# Marker accent — a vivid green DISTINCT from the white/blue playhead
# (``#FFFFFF`` / reserved ``#4DA3FF``), the amber Keeper/Draft region edges
# (``#FFC857``), the blue-grey Untouched edges (``#6B8FA8``), and the grey
# Trash edges (``#5C636B``). Locked here as a source-grep token.
_MARKER_PEN_COLOR = "#3ECF8E"
# Label anchored near the TOP of the plot (y-fraction 0.05 from the top edge
# in InfiniteLine label position terms — 0.0 is the bottom, 1.0 the top).
_MARKER_LABEL_POSITION = 0.95


class MarkersOverlay(QObject):
    """Point-in-time marker lines on the WaveformView's PlotItem.

    Sibling of :class:`RegionsOverlay` — owns its own registry of read-only
    (``movable=False``) :class:`pyqtgraph.InfiniteLine` items, one per marker,
    added to the SAME ``plot_item`` the RegionsOverlay uses so they pan/zoom
    in lockstep with the waveform (locked decision #3 — markers are read-only
    on the waveform; editing/deleting happens in the Markers panel).

    Qt cleanup discipline (Phase 1 LEARNINGS §"removeItem alone leaks"):
    every removal calls ``plot_item.removeItem(line)`` AND ``line.deleteLater()``.

    Construction args:
        plot_item: ``pg.PlotItem`` — the WaveformView's main plot
            (``view.waveform_plot``), identical to the RegionsOverlay's.
    """

    def __init__(
        self,
        plot_item: pg.PlotItem,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._plot_item = plot_item
        # Per-marker line registry keyed by marker id for O(1) lookup.
        self._markers: dict[str, pg.InfiniteLine] = {}

    # ---------------------------------------------------------------- helpers
    def _make_line(self, marker: Marker) -> pg.InfiniteLine:
        """Build a read-only vertical InfiniteLine for ``marker``.

        Distinct green pen (:data:`_MARKER_PEN_COLOR`), ``movable=False``,
        label = the marker's label rendered near the top of the plot.
        """
        line = pg.InfiniteLine(
            pos=float(marker.time_sec),
            angle=90,
            movable=False,
            pen=pg.mkPen(_MARKER_PEN_COLOR, width=1),
            label=marker.label or "",
            labelOpts={
                "position": _MARKER_LABEL_POSITION,
                "color": _MARKER_PEN_COLOR,
                "movable": False,
                "fill": (0, 0, 0, 120),
            },
        )
        return line

    # ----------------------------------------------------------- public API
    def set_markers(self, markers: list[Marker]) -> None:
        """Replace all marker lines with one line per marker in ``markers``.

        Clears the existing lines (``removeItem`` + ``deleteLater``) then adds
        a fresh line per marker. Used on file-open / reload.
        """
        self.clear()
        for m in markers:
            self.add_marker(m)

    def add_marker(self, marker: Marker) -> None:
        """Add a single marker line to the plot + registry."""
        # Idempotent — replace an existing line for the same id.
        if marker.id in self._markers:
            self.remove_marker(marker.id)
        line = self._make_line(marker)
        self._plot_item.addItem(line)
        self._markers[marker.id] = line

    def remove_marker(self, marker_id: str) -> None:
        """Remove a marker line (removeItem + deleteLater — no leak)."""
        line = self._markers.pop(marker_id, None)
        if line is None:
            return
        self._plot_item.removeItem(line)
        try:
            line.deleteLater()
        except RuntimeError:
            pass

    def update_label(self, marker_id: str, label: str) -> None:
        """Update an existing marker line's label text in place."""
        line = self._markers.get(marker_id)
        if line is None:
            return
        # pg.InfiniteLine exposes ``setText`` on its label via ``label``.
        try:
            line.label.setFormat(label or "")
        except (AttributeError, RuntimeError):
            # Defensive — if the label object is not present, rebuild the
            # line's text via the public setText path when available.
            try:
                line.setText(label or "")
            except (AttributeError, RuntimeError):
                pass

    def clear(self) -> None:
        """Remove every marker line + reset the registry."""
        for marker_id in list(self._markers.keys()):
            self.remove_marker(marker_id)

    def marker_count(self) -> int:
        """Number of marker lines currently on the plot."""
        return len(self._markers)
