"""Phase 7 Plan 07-04 Task 1 — ABToggleWidget composite toolbar widget.

A small two-button toolbar widget for A/B audio source preview. Owns
the STATE-KEY semantics from D-13:

* ``A`` and ``B`` are STATE keys, not togglers — pressing the active
  key is a no-op, never flips to the other side.
* Sub-button clicks AND keyboard shortcuts (wired by :class:`MainWindow`
  in Task 2) both call :meth:`set_state` with the target state.

The widget itself is shortcut-free so it can be unit-tested without
a top-level window. MainWindow installs the ``A`` / ``B``
``ApplicationShortcut`` instances and routes them through this
widget's :meth:`set_state` method.

UI-SPEC contract — locked here for source-grep gates:

* Composite QWidget: ``QHBoxLayout`` margins (0,0,0,0) spacing 0,
  containing ``[button_a][divider][button_b]`` in that order.
* Each sub-button: ``setFixedSize(24, 24)``, ``setFlat(True)``,
  ASCII text ``"A"`` / ``"B"`` (Wayland-safe — Phase 6 LEARNINGS).
* 1 px vertical divider via ``QFrame.VLine`` (UI-SPEC §"A/B toolbar
  toggle visual states" line 179).
* Both sub-buttons have ``setFocusPolicy(Qt.FocusPolicy.NoFocus)`` so
  keyboard focus stays on the central widget.

Phase 6 LEARNINGS — NO Unicode in load-bearing labels (``QFontMetrics.
inFont`` lies at small sizes on Ubuntu Wayland). ASCII ``A`` and ``B``
both render reliably across platforms.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)


# UI-SPEC §"A/B toolbar toggle visual states" object-name selectors.
# QSS lives in app.qss (the active sub-button gets a #4DA3FF border + an
# accent-tinted background; the inactive sub-button is transparent).
_ACTIVE_OBJECT_NAME = "ABButtonActive"
_INACTIVE_OBJECT_NAME = "ABButtonInactive"

# Phase 1 divider token reused for the 1 px vertical line between the
# two sub-buttons (UI-SPEC §"Mastering dock background and per-stage row
# tokens" — same `#2F2F33` border token).
_DIVIDER_COLOR = "#2F2F33"


class ABToggleWidget(QWidget):
    """Composite A/B preview toggle.

    Two sub-buttons (``A`` / ``B``) wrapped in a horizontal layout with
    a 1 px vertical divider. Click → :meth:`set_state`. State changes
    emit :attr:`state_changed` with the new state string. Same-state
    presses are no-ops.

    Signals:
        state_changed(str): Emitted on every state transition. Payload
            is the new state (``"A"`` or ``"B"``). NEVER emitted when
            :meth:`set_state` is called with the current state
            (STATE-KEY semantics per D-13).

    Public API:
        :attr:`state` (read-only): current state, ``"A"`` or ``"B"``.
        :attr:`is_enabled` (read-only): mirror of :meth:`isEnabled`.
        :meth:`set_state` (state: str): transition to the given state.
        :meth:`set_enabled` (enabled: bool): forward to
            :meth:`setEnabled` — when False, sub-button clicks are
            suppressed by Qt automatically and ``state_changed`` does
            not fire.

    Default state on construction is ``"A"`` per UI-SPEC line 554
    ("Default state on enable: A (source)").
    """

    # Emitted with the new state string after EVERY transition. Never
    # emitted on same-state presses (STATE-KEY no-op).
    state_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._state: str = "A"  # UI-SPEC line 554 — default to A on construct.

        # UI-SPEC §Spacing Scale line 97 — composite widget 48×24 px
        # (two 24×24 sub-buttons + a 1 px divider; the divider's width
        # is absorbed by the layout's QFrame size hint).
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._button_a = QPushButton("A", self)
        self._button_a.setFixedSize(24, 24)
        self._button_a.setFlat(True)
        self._button_a.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._button_a.setObjectName(_ACTIVE_OBJECT_NAME)  # default A active
        self._button_a.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Default-arg closure binding (Phase 1 LEARNINGS) — the lambda
        # closes over the STATE constant, not the loop variable, so this
        # is safe even though the widget construction does not loop.
        self._button_a.clicked.connect(
            lambda _checked=False: self.set_state("A")
        )

        # 1 px vertical divider between the two sub-buttons per UI-SPEC.
        self._divider = QFrame(self)
        self._divider.setFrameShape(QFrame.Shape.VLine)
        self._divider.setFrameShadow(QFrame.Shadow.Plain)
        self._divider.setLineWidth(1)
        # Set the divider's foreground color via palette (works across
        # Wayland/X11). QSS could do this too but palette is more robust
        # against theme stylesheets that override frame-foreground.
        pal = self._divider.palette()
        pal.setColor(QPalette.ColorRole.WindowText, QColor(_DIVIDER_COLOR))
        self._divider.setPalette(pal)
        self._divider.setFixedWidth(1)
        self._divider.setFixedHeight(24)

        self._button_b = QPushButton("B", self)
        self._button_b.setFixedSize(24, 24)
        self._button_b.setFlat(True)
        self._button_b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._button_b.setObjectName(_INACTIVE_OBJECT_NAME)
        self._button_b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._button_b.clicked.connect(
            lambda _checked=False: self.set_state("B")
        )

        layout.addWidget(self._button_a)
        layout.addWidget(self._divider)
        layout.addWidget(self._button_b)

        # Overall widget size is 24 (A) + 1 (divider) + 24 (B) = 49 px;
        # UI-SPEC §Spacing Scale line 97 names 48 px which is the rounded
        # nominal — we honor the geometry exactly here (49 px) and let the
        # layout absorb the 1 px overshoot. The contract is "48×24 widget
        # made of two 24×24 sub-buttons plus a 1 px divider".
        self.setFixedHeight(24)

    # ----------------------------------------------------------- public API
    @property
    def state(self) -> str:
        """Current state — ``"A"`` (source) or ``"B"`` (mastered)."""
        return self._state

    @property
    def is_enabled(self) -> bool:
        """Mirror of :meth:`isEnabled` for ergonomic test asserts."""
        return self.isEnabled()

    def set_state(self, new_state: str) -> None:
        """Transition to ``new_state``. STATE-KEY no-op if already there.

        Args:
            new_state: ``"A"`` or ``"B"``. Other values raise
                :class:`AssertionError` (defensive — both call sites are
                inside this codebase).

        Behavior:
            * If ``new_state == self._state``: returns silently. No
              repaint. No :attr:`state_changed` emission. This is the
              load-bearing STATE-KEY semantics from D-13 — pressing A
              while on A is a no-op, never a toggle to B.
            * Else: updates internal state, swaps the QSS object names
              on both sub-buttons (active / inactive), repaints by
              cycling unpolish/polish, and emits
              :attr:`state_changed` with the new state.
        """
        assert new_state in ("A", "B"), (
            f"ABToggleWidget.set_state: invalid state {new_state!r} "
            "(expected 'A' or 'B')"
        )
        if new_state == self._state:
            return  # STATE-KEY no-op — same key pressed twice.
        self._state = new_state
        if new_state == "A":
            self._button_a.setObjectName(_ACTIVE_OBJECT_NAME)
            self._button_b.setObjectName(_INACTIVE_OBJECT_NAME)
        else:
            self._button_a.setObjectName(_INACTIVE_OBJECT_NAME)
            self._button_b.setObjectName(_ACTIVE_OBJECT_NAME)
        # Force a repaint after object-name changes — Qt's QSS engine
        # caches selectors by object name; unpolish/polish cycles the
        # cache so the new selectors apply immediately.
        for btn in (self._button_a, self._button_b):
            style = btn.style()
            if style is not None:
                style.unpolish(btn)
                style.polish(btn)
            btn.update()
        self.state_changed.emit(new_state)

    def set_enabled(self, enabled: bool) -> None:
        """Forward to :meth:`setEnabled`.

        When disabled, Qt automatically suppresses click events on the
        sub-buttons so :meth:`set_state` is never called and
        :attr:`state_changed` does not fire. The disabled-text visual
        style is provided by Qt's ``:disabled`` QSS pseudo-selector per
        UI-SPEC §"A/B toolbar toggle visual states" line 176.
        """
        self.setEnabled(bool(enabled))


__all__ = ["ABToggleWidget"]
