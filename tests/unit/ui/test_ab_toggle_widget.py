"""Phase 7 Plan 07-04 Task 1 (RED) — ABToggleWidget unit tests.

Pins the composite A/B toolbar widget contract:

* Default state is A.
* ``set_state("B")`` from A emits ``state_changed("B")``.
* STATE-KEY semantics — pressing the active key is a no-op:
  - ``set_state("A")`` while already on A emits NOTHING.
  - ``set_state("B")`` twice in a row emits exactly once.
* Sub-button clicks drive ``set_state`` (no extra hookup needed).
* Disabled state suppresses signal emissions on click.
* ASCII-only labels — Wayland-safe per Phase 6 LEARNINGS (no Unicode
  above 127 in either button's text). UI-SPEC §"Icon and Badge
  Rendering Contract — A/B toolbar toggle rendering" lines 506-520.

Phase 7 Plan 07-09 additions (RED):

* Render-to-pixmap regression guard: under ``theme.apply_theme(app)`` the
  ABToggleWidget sub-buttons MUST render their "A" / "B" glyphs as visible
  light pixels (≥ 50 pixels with any RGB channel > 200). Pre-fix this test
  fails (zero light pixels — diagnosis evidence at debug/ab-widget-broken-
  keys-icon-tooltip.md timestamp 00:05).
* Active vs inactive visual distinct — after ``set_state("B")`` the B
  sub-button has a measurably different background sample than the now-
  inactive A sub-button (the active state's #4DA3FF accent border + tinted
  background vs transparent on inactive).
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton

from marmelade.ui import theme
from marmelade.ui.ab_toggle import ABToggleWidget


def test_default_state_is_A(qtbot, qapp) -> None:
    """Widget constructs with state == 'A' per UI-SPEC line 554."""
    w = ABToggleWidget()
    qtbot.addWidget(w)
    assert w.state == "A"


def test_set_state_B_emits_state_changed(qtbot, qapp) -> None:
    """Transition A→B emits state_changed with payload 'B' exactly once."""
    w = ABToggleWidget()
    qtbot.addWidget(w)
    payloads: list[str] = []
    w.state_changed.connect(payloads.append)
    w.set_state("B")
    assert payloads == ["B"]
    assert w.state == "B"


def test_state_key_semantics_pressing_A_while_on_A_is_no_op(qtbot, qapp) -> None:
    """STATE-KEY: re-pressing the active state does NOT emit (D-13)."""
    w = ABToggleWidget()
    qtbot.addWidget(w)
    assert w.state == "A"
    payloads: list[str] = []
    w.state_changed.connect(payloads.append)
    w.set_state("A")
    assert payloads == [], "set_state on the active key must be a no-op"
    assert w.state == "A"


def test_state_key_semantics_pressing_B_while_on_B_is_no_op(
    qtbot, qapp
) -> None:
    """STATE-KEY: B then B again — signal fires once, not twice."""
    w = ABToggleWidget()
    qtbot.addWidget(w)
    payloads: list[str] = []
    w.state_changed.connect(payloads.append)
    w.set_state("B")  # fires
    w.set_state("B")  # no-op
    assert payloads == ["B"]
    assert w.state == "B"


def test_button_click_drives_set_state(qtbot, qapp) -> None:
    """Clicking the B sub-button transitions state to 'B'."""
    w = ABToggleWidget()
    qtbot.addWidget(w)
    payloads: list[str] = []
    w.state_changed.connect(payloads.append)
    qtbot.mouseClick(w._button_b, Qt.MouseButton.LeftButton)
    assert w.state == "B"
    assert payloads == ["B"]


def test_set_enabled_false_disables_widget(qtbot, qapp) -> None:
    """When widget is disabled, sub-button clicks do NOT emit."""
    w = ABToggleWidget()
    qtbot.addWidget(w)
    w.set_enabled(False)
    assert w.isEnabled() is False
    payloads: list[str] = []
    w.state_changed.connect(payloads.append)
    qtbot.mouseClick(w._button_b, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(w._button_a, Qt.MouseButton.LeftButton)
    assert payloads == [], "Disabled widget must suppress click → set_state"


def test_ASCII_labels_only_no_unicode(qtbot, qapp) -> None:
    """A/B sub-buttons use ASCII text — no Unicode > 127.

    Phase 6 LEARNINGS — Wayland renders some Unicode glyphs unreliably
    even when QFontMetrics.inFont returns True. ASCII letters always
    render. UI-SPEC §"Icon and Badge Rendering Contract" line 511.
    """
    w = ABToggleWidget()
    qtbot.addWidget(w)
    assert w._button_a.text() == "A"
    assert w._button_b.text() == "B"
    for ch in w._button_a.text() + w._button_b.text():
        assert ord(ch) < 128, f"Non-ASCII char {ch!r} (U+{ord(ch):04X}) in A/B label"


def test_widget_has_two_qpushbutton_children(qtbot, qapp) -> None:
    """Composite construction has _button_a and _button_b QPushButtons."""
    w = ABToggleWidget()
    qtbot.addWidget(w)
    assert isinstance(w._button_a, QPushButton)
    assert isinstance(w._button_b, QPushButton)
    # Per UI-SPEC §Spacing Scale line 97 — each sub-button is 24×24 px.
    assert w._button_a.size().width() == 24
    assert w._button_a.size().height() == 24
    assert w._button_b.size().width() == 24
    assert w._button_b.size().height() == 24


def test_set_state_invalid_raises(qtbot, qapp) -> None:
    """Defensive contract — only 'A' or 'B' accepted; other values raise."""
    w = ABToggleWidget()
    qtbot.addWidget(w)
    with pytest.raises((AssertionError, ValueError)):
        w.set_state("C")


# =========================================================================
# Phase 7 Plan 07-09 (RED) — render-pixel regression guards
# =========================================================================

def _count_light_pixels(pixmap) -> int:
    """Count pixels in ``pixmap`` with any RGB channel > 200.

    This is the discriminating threshold from
    debug/ab-widget-broken-keys-icon-tooltip.md timestamps 00:05/00:06:
    a button rendered WITH the broken (text-clipped) state shows 0 light
    pixels (only the dark #2F2F33 background + #4DA3FF accent border —
    both have at least one channel ≤ 200). A button rendering "A"/"B"
    text glyphs on top shows ≥ 50 light pixels for the text strokes.
    """
    image = pixmap.toImage()
    count = 0
    for x in range(image.width()):
        for y in range(image.height()):
            c = image.pixelColor(x, y)
            if c.red() > 200 or c.green() > 200 or c.blue() > 200:
                count += 1
    return count


def _sample_background(pixmap, x: int = 2, y: int = 2) -> tuple[int, int, int]:
    """Sample a pixel at ``(x, y)`` away from the central text glyph.

    Returns ``(r, g, b)`` of the sample. The (2, 2) corner offset on a
    24×24 button lands inside the border / background area, not over
    the text glyph (which centers around (12, 12)).
    """
    image = pixmap.toImage()
    c = image.pixelColor(x, y)
    return (c.red(), c.green(), c.blue())


def test_ab_toggle_renders_text_pixels_under_qss(qtbot, qapp) -> None:
    """Sub-buttons render "A" / "B" text glyphs as visible light pixels.

    The diagnosis (debug/ab-widget-broken-keys-icon-tooltip.md timestamp
    00:05) proved experimentally that under the pre-fix app.qss, the two
    sub-buttons render with ZERO light pixels — the "A"/"B" text is
    clipped by the inherited ``QPushButton { padding: 6px 16px }`` rule
    (16+16=32 px horizontal padding on a 24-px-wide button). After the
    Plan 07-09 QSS rules land, each sub-button must show ≥ 50 light
    pixels (the text glyph strokes).

    The pixel-count threshold (50) is conservative — a real "A" or "B"
    glyph at 10pt 600-weight has ~150-300 light pixels; the floor of
    50 gives margin for font fallback variations across platforms while
    still detecting the "0 light pixels" failure mode unambiguously.
    """
    theme.apply_theme(QApplication.instance())
    w = ABToggleWidget()
    qtbot.addWidget(w)
    w.show()
    w.repaint()
    qapp.processEvents()

    pix_a = w._button_a.grab()
    count_a = _count_light_pixels(pix_a)
    assert count_a >= 50, (
        f"Expected >=50 light pixels in A sub-button (text glyph); got {count_a}. "
        "Likely the #ABButtonActive QSS rule did not apply — check app.qss for "
        "the rule block (Plan 07-09)."
    )

    pix_b = w._button_b.grab()
    count_b = _count_light_pixels(pix_b)
    assert count_b >= 50, (
        f"Expected >=50 light pixels in B sub-button (text glyph); got {count_b}. "
        "Likely the #ABButtonInactive QSS rule did not apply — check app.qss for "
        "the rule block (Plan 07-09)."
    )


def test_ab_toggle_renders_active_state_visual_distinct_from_inactive(
    qtbot, qapp
) -> None:
    """Active sub-button has a measurably different background than inactive.

    Default state is A → A is active (accent border + tinted background),
    B is inactive (transparent border + transparent background). Sample
    pixels at (1, 12) — that lands on the LEFT BORDER of each sub-button,
    where the active state's #4DA3FF accent border (blue=255) is the
    discriminating signal vs the inactive's transparent border.

    After ``set_state("B")``, the roles swap and the assertion repeats
    inverted — the B button's border now has the accent blue, A is
    transparent.
    """
    theme.apply_theme(QApplication.instance())
    w = ABToggleWidget()
    qtbot.addWidget(w)
    w.show()
    w.repaint()
    qapp.processEvents()

    # Default state: A is active. Sample the leftmost border column
    # (x=0, y=12) — the active state paints #4DA3FF there.
    pix_a_active = w._button_a.grab()
    pix_b_inactive = w._button_b.grab()
    a_border = _sample_background(pix_a_active, x=0, y=12)
    b_border = _sample_background(pix_b_inactive, x=0, y=12)
    # Active has the #4DA3FF (77, 163, 255) accent border — its blue
    # channel is much higher than the inactive's (which is transparent
    # over the toolbar's #1E1E1E background — blue near 30).
    assert a_border[2] > b_border[2] + 50, (
        f"Active A border should be visibly more blue than inactive B border. "
        f"A border RGB={a_border}, B border RGB={b_border}. "
        "Likely the #ABButtonActive vs #ABButtonInactive border QSS rules did "
        "not apply differentially (Plan 07-09)."
    )

    # Flip state.
    w.set_state("B")
    w.repaint()
    qapp.processEvents()

    pix_a_inactive = w._button_a.grab()
    pix_b_active = w._button_b.grab()
    a_border2 = _sample_background(pix_a_inactive, x=0, y=12)
    b_border2 = _sample_background(pix_b_active, x=0, y=12)
    assert b_border2[2] > a_border2[2] + 50, (
        f"After set_state('B'), B border should be visibly more blue than A. "
        f"A border RGB={a_border2}, B border RGB={b_border2}. "
        "Likely the object-name swap in ABToggleWidget.set_state did not "
        "re-trigger the #ABButtonActive QSS rule (Plan 07-09)."
    )
