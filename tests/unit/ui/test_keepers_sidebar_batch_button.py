"""Quick-260615-l4y — Two-button Master / Export sidebar redesign.

Pins the top-of-sidebar layout AFTER the morphing one-button machine
was split into two persistent buttons:

    KeepersSidebar._batch_button  — the MASTER button (two states):
        idle    → "Master All Keepers"   (enabled iff ≥1 keeper)
        running → "Cancel mastering"      (always enabled while in-flight)
    KeepersSidebar._export_button — the EXPORT button, freshness-gated
        by the same ``_mastered_cache_fresh_probe`` the bundle button uses.

Outer-layout order: Master (index 0), Export (index 1), Share/bundle
(index 2), then the QStackedWidget.

Signals:
    master_all_requested      = Signal()   (Master click in idle state)
    mastering_cancel_requested = Signal()   (Master click while running)
    export_all_requested       = Signal()   (Export click when enabled)

Behavior pins:

1. Two persistent buttons at the top of the sidebar; neither morphs
   into the other's role.
2. Empty sidebar: Master disabled with its literal disabled tooltip;
   Export disabled.
3. ≥1 keeper, no fresh cache: Master ENABLED; Export DISABLED with a
   "Master all keepers first…" tooltip.
4. ≥1 keeper + probe returns True for every row: Export ENABLED.
5. Installing the probe (set_mastered_cache_fresh_probe) refreshes the
   Export button.
6. Master state machine reduced to {idle, running}. While running the
   Export button is DISABLED regardless of cache freshness.
7. Click routing: Master idle → master_all_requested; Master running →
   mastering_cancel_requested; Export enabled → export_all_requested.
8. Removing the last row re-disables both buttons.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QVBoxLayout

from marmelade.audio.sidecar_cache import Region
from marmelade.ui.keepers_sidebar import KeepersSidebar


def _add_keeper(sidebar: KeepersSidebar, idx: int = 0) -> str:
    """Inject a Region into the sidebar's populated state via the public API."""
    rid = f"id{idx:032d}"
    region = Region(
        id=rid,
        start_sec=float(idx * 10),
        end_sec=float(idx * 10 + 5),
        state="keeper",
    )
    sidebar.add_row(region)
    return rid


# ---------------------------------------------------------------- Pin 1
def test_master_button_visible_at_top_of_sidebar(qtbot, qapp) -> None:
    """The Master button is the first widget in the sidebar's outer layout."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    outer = sidebar.layout()
    assert isinstance(outer, QVBoxLayout), (
        f"Outer layout must be QVBoxLayout (got {type(outer).__name__})"
    )
    first_widget = outer.itemAt(0).widget()
    assert isinstance(first_widget, QPushButton)
    assert first_widget is sidebar._batch_button
    assert first_widget.text() == "Master All Keepers", (
        f"Initial Master label must be the idle string; "
        f"got {first_widget.text()!r}"
    )


def test_two_buttons_layout_order_master_export_share(qtbot, qapp) -> None:
    """Outer layout order: Master (0), Export (1), Share/bundle (2)."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    outer = sidebar.layout()
    assert outer.indexOf(sidebar._batch_button) == 0
    assert outer.indexOf(sidebar._export_button) == 1
    assert outer.indexOf(sidebar._bundle_button) == 2


# ---------------------------------------------------------------- Pin 2
def test_master_button_disabled_when_no_keepers(qtbot, qapp) -> None:
    """Empty sidebar — Master disabled + literal disabled tooltip."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    btn = sidebar._batch_button
    assert not btn.isEnabled(), "Empty sidebar must disable Master button"
    assert btn.toolTip() == (
        "Create at least one Keeper to enable batch mastering."
    ), f"Disabled tooltip mismatch: {btn.toolTip()!r}"


def test_export_button_disabled_when_no_keepers(qtbot, qapp) -> None:
    """Empty sidebar — Export disabled."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    assert not sidebar._export_button.isEnabled()


# ---------------------------------------------------------------- Pin 3
def test_master_enabled_export_disabled_without_fresh_cache(qtbot, qapp) -> None:
    """≥1 keeper, no probe → Master enabled, Export disabled w/ tooltip."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    assert sidebar._batch_button.isEnabled(), (
        "Master button must be enabled whenever ≥1 keeper exists"
    )
    assert not sidebar._export_button.isEnabled(), (
        "Export must be disabled with no fresh mastered cache"
    )
    tt = sidebar._export_button.toolTip()
    assert "Master all keepers first" in tt, (
        f"Export disabled tooltip should explain the gate; got {tt!r}"
    )


def test_master_button_enabled_tooltip(qtbot, qapp) -> None:
    """≥1 keeper → Master enabled + literal enabled tooltip."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    btn = sidebar._batch_button
    assert btn.isEnabled()
    assert btn.toolTip() == (
        "Master every Keeper so they're ready to audition and export."
    ), f"Enabled tooltip mismatch: {btn.toolTip()!r}"


# ---------------------------------------------------------------- Pin 4
def test_export_enabled_when_all_rows_fresh(qtbot, qapp) -> None:
    """≥1 keeper + probe True for every row → Export enabled."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    _add_keeper(sidebar, idx=2)
    sidebar.set_mastered_cache_fresh_probe(lambda rid: True)
    assert sidebar._export_button.isEnabled(), (
        "Export must enable when every keeper has a fresh mastered cache"
    )


def test_export_disabled_when_one_row_not_fresh(qtbot, qapp) -> None:
    """Probe False for any row → Export disabled."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    rid1 = _add_keeper(sidebar, idx=1)
    _add_keeper(sidebar, idx=2)
    sidebar.set_mastered_cache_fresh_probe(lambda rid: rid == rid1)
    assert not sidebar._export_button.isEnabled()


# ---------------------------------------------------------------- Pin 5
def test_probe_install_refreshes_export_button(qtbot, qapp) -> None:
    """Installing a True-probe with ≥1 keeper present enables Export."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    assert not sidebar._export_button.isEnabled()
    sidebar.set_mastered_cache_fresh_probe(lambda rid: True)
    assert sidebar._export_button.isEnabled(), (
        "set_mastered_cache_fresh_probe must trigger an Export refresh"
    )


def test_refresh_export_button_public_hook(qtbot, qapp) -> None:
    """refresh_export_button() re-probes after external cache change."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    fresh = {"v": False}
    sidebar.set_mastered_cache_fresh_probe(lambda rid: fresh["v"])
    assert not sidebar._export_button.isEnabled()
    fresh["v"] = True
    sidebar.refresh_export_button()
    assert sidebar._export_button.isEnabled()


# ---------------------------------------------------------------- Pin 6
def test_master_state_machine_running(qtbot, qapp) -> None:
    """set_batch_state('running') → label 'Cancel mastering', enabled."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    sidebar.set_batch_state("running")
    assert sidebar._batch_button.text() == "Cancel mastering"
    assert sidebar._batch_button.isEnabled()


def test_master_state_machine_back_to_idle(qtbot, qapp) -> None:
    """set_batch_state('idle') after running reverts to the idle label."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    sidebar.set_batch_state("running")
    sidebar.set_batch_state("idle")
    assert sidebar._batch_button.text() == "Master All Keepers"
    assert sidebar._batch_button.isEnabled()


def test_export_disabled_while_mastering_in_flight(qtbot, qapp) -> None:
    """While running, Export is disabled regardless of cache freshness."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    sidebar.set_mastered_cache_fresh_probe(lambda rid: True)
    assert sidebar._export_button.isEnabled()
    sidebar.set_batch_state("running")
    assert not sidebar._export_button.isEnabled(), (
        "Export must be disabled while a batch master is in-flight"
    )
    # Reverting to idle re-evaluates freshness → Export re-enabled.
    sidebar.set_batch_state("idle")
    assert sidebar._export_button.isEnabled()


def test_set_batch_state_accepts_ok_count_arg(qtbot, qapp) -> None:
    """Legacy ok_count arg is accepted as a no-op (caller compat)."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    # Must not raise.
    sidebar.set_batch_state("running", ok_count=3)
    sidebar.set_batch_state("idle", ok_count=2)
    assert sidebar._batch_button.text() == "Master All Keepers"


# ---------------------------------------------------------------- Pin 7
def test_click_master_idle_emits_master_all_requested(qtbot, qapp) -> None:
    """Idle Master click → master_all_requested fires."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    with qtbot.waitSignal(sidebar.master_all_requested, timeout=1000):
        qtbot.mouseClick(sidebar._batch_button, Qt.MouseButton.LeftButton)


def test_click_master_running_emits_cancel(qtbot, qapp) -> None:
    """Master click while running → mastering_cancel_requested fires."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    sidebar.set_batch_state("running")
    with qtbot.waitSignal(sidebar.mastering_cancel_requested, timeout=1000):
        qtbot.mouseClick(sidebar._batch_button, Qt.MouseButton.LeftButton)


def test_click_export_emits_export_all_requested(qtbot, qapp) -> None:
    """Enabled Export click → export_all_requested fires."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    _add_keeper(sidebar, idx=1)
    sidebar.set_mastered_cache_fresh_probe(lambda rid: True)
    assert sidebar._export_button.isEnabled()
    with qtbot.waitSignal(sidebar.export_all_requested, timeout=1000):
        qtbot.mouseClick(sidebar._export_button, Qt.MouseButton.LeftButton)


# ---------------------------------------------------------------- Pin 8
def test_remove_last_row_disables_both_buttons(qtbot, qapp) -> None:
    """Removing all rows re-disables both the Master and Export buttons."""
    sidebar = KeepersSidebar()
    qtbot.add_widget(sidebar)
    rid = _add_keeper(sidebar, idx=0)
    sidebar.set_mastered_cache_fresh_probe(lambda rid: True)
    assert sidebar._batch_button.isEnabled()
    assert sidebar._export_button.isEnabled()
    sidebar.remove_row(rid)
    assert not sidebar._batch_button.isEnabled()
    assert not sidebar._export_button.isEnabled()
