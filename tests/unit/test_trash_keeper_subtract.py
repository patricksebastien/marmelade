"""Plan 03-03 Task 1 — pure unit tests for the trash_minus_keepers algorithm.

Three concentric layers:

* ``_merge_intervals(...)`` — the module-private helper that fuses
  overlapping or touching half-open intervals. Six pins.
* ``RegionsOverlay.keeper_trash_ranges`` — extracts the per-state ranges
  from the overlay's region registry. Four pins.
* ``RegionsOverlay.trash_minus_keepers`` — Keeper-punch-through subtraction
  (D-A2-5). Nine pins covering the full interval-arithmetic surface.

We avoid constructing a real :class:`RegionsOverlay` (it requires a
``pg.PlotItem`` host + a Qt event loop). Instead a small ``_StubOverlay``
class binds the unbound methods of ``RegionsOverlay`` to a plain object
with a populated ``_regions`` dict — the methods only read
``_regions.items()`` and call each region's ``getRegion()`` + read
``_current_state``, so the stub is shape-equivalent.

Default ``default_cache_root`` is imported so ``conftest.py`` can patch
it consistently with the rest of the test suite (kept for uniformity
even though this module never writes to the cache).
"""

from __future__ import annotations

import pytest

from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target
from marmelade.ui.regions_overlay import (
    RegionsOverlay,
    _merge_intervals,
)


# ----------------------------------------------------------------- stubs
class _StubRegion:
    """Minimal stand-in for ``ResizeOnlyRegion`` — the algorithm only
    reads ``getRegion()`` (start, end) and ``_current_state``.
    """

    def __init__(self, start: float, end: float, state: str) -> None:
        self._s = float(start)
        self._e = float(end)
        self._current_state = state

    def getRegion(self) -> tuple[float, float]:  # noqa: N802 — PyQtGraph API
        return (self._s, self._e)


class _StubOverlay:
    """Stub that re-uses the unbound ``RegionsOverlay`` methods.

    We don't subclass :class:`RegionsOverlay` because that pulls in
    :class:`QObject` initialisation requiring a QApplication. Pure
    attribute-binding is all the algorithm needs.
    """

    keeper_trash_ranges = RegionsOverlay.keeper_trash_ranges
    trash_minus_keepers = RegionsOverlay.trash_minus_keepers

    def __init__(self, regions: list[_StubRegion]) -> None:
        self._regions = {f"r{i}": r for i, r in enumerate(regions)}


# ====================================================================
# _merge_intervals — module-level helper
# ====================================================================
def test_merge_intervals_empty() -> None:
    assert _merge_intervals([]) == []


def test_merge_intervals_no_overlap() -> None:
    assert _merge_intervals([(0.0, 1.0), (2.0, 3.0)]) == [(0.0, 1.0), (2.0, 3.0)]


def test_merge_intervals_overlap() -> None:
    assert _merge_intervals([(0.0, 5.0), (3.0, 7.0)]) == [(0.0, 7.0)]


def test_merge_intervals_touching() -> None:
    # Touching intervals (b == c) are merged. The skip-range list is
    # used downstream as half-open [start, end) so touching ranges
    # collapse cleanly without leaving a zero-width gap.
    assert _merge_intervals([(0.0, 5.0), (5.0, 10.0)]) == [(0.0, 10.0)]


def test_merge_intervals_engulf() -> None:
    # Second interval fully inside the first; merge returns the wider one.
    assert _merge_intervals([(0.0, 10.0), (3.0, 5.0)]) == [(0.0, 10.0)]


def test_merge_intervals_chain() -> None:
    # Three overlapping then a disjoint pair — produces two output ranges.
    out = _merge_intervals([(0.0, 2.0), (1.0, 4.0), (3.0, 6.0), (10.0, 12.0)])
    assert out == [(0.0, 6.0), (10.0, 12.0)]


# ====================================================================
# keeper_trash_ranges — extracts per-state ranges from regions dict
# ====================================================================
def test_keeper_trash_ranges_empty() -> None:
    ov = _StubOverlay([])
    assert ov.keeper_trash_ranges() == ([], [])


def test_keeper_trash_ranges_skips_untouched() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(0, 10, "untouched"),
            _StubRegion(20, 30, "keeper"),
            _StubRegion(40, 50, "trash"),
        ]
    )
    keepers, trash = ov.keeper_trash_ranges()
    assert keepers == [(20.0, 30.0)]
    assert trash == [(40.0, 50.0)]


def test_keeper_trash_ranges_skips_zero_width() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(5, 5, "keeper"),  # zero width
            _StubRegion(10, 5, "keeper"),  # negative width
            _StubRegion(20, 30, "keeper"),
        ]
    )
    keepers, trash = ov.keeper_trash_ranges()
    assert keepers == [(20.0, 30.0)]
    assert trash == []


def test_keeper_trash_ranges_sorted_by_start() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(40, 50, "trash"),
            _StubRegion(10, 20, "trash"),
            _StubRegion(25, 30, "trash"),
        ]
    )
    keepers, trash = ov.keeper_trash_ranges()
    assert trash == [(10.0, 20.0), (25.0, 30.0), (40.0, 50.0)]
    assert keepers == []


# ====================================================================
# trash_minus_keepers — Keeper-punch-through subtraction (D-A2-5)
# ====================================================================
def test_no_trash_returns_empty() -> None:
    ov = _StubOverlay([_StubRegion(0, 10, "keeper")])
    assert ov.trash_minus_keepers() == []


def test_no_keepers_returns_merged_trash() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(0, 5, "trash"),
            _StubRegion(3, 8, "trash"),
            _StubRegion(20, 30, "trash"),
        ]
    )
    assert ov.trash_minus_keepers() == [(0.0, 8.0), (20.0, 30.0)]


def test_keeper_inside_trash_splits_trash() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(0, 20, "trash"),
            _StubRegion(5, 10, "keeper"),
        ]
    )
    assert ov.trash_minus_keepers() == [(0.0, 5.0), (10.0, 20.0)]


def test_keeper_engulfing_trash_eliminates_it() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(5, 10, "trash"),
            _StubRegion(0, 20, "keeper"),
        ]
    )
    assert ov.trash_minus_keepers() == []


def test_keeper_at_left_edge_of_trash() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(0, 10, "trash"),
            _StubRegion(0, 3, "keeper"),
        ]
    )
    assert ov.trash_minus_keepers() == [(3.0, 10.0)]


def test_keeper_at_right_edge_of_trash() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(0, 10, "trash"),
            _StubRegion(7, 10, "keeper"),
        ]
    )
    assert ov.trash_minus_keepers() == [(0.0, 7.0)]


def test_two_keepers_carve_three_subranges() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(0, 30, "trash"),
            _StubRegion(5, 10, "keeper"),
            _StubRegion(15, 20, "keeper"),
        ]
    )
    assert ov.trash_minus_keepers() == [(0.0, 5.0), (10.0, 15.0), (20.0, 30.0)]


def test_disjoint_keeper_no_effect() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(0, 10, "trash"),
            _StubRegion(100, 200, "keeper"),
        ]
    )
    assert ov.trash_minus_keepers() == [(0.0, 10.0)]


def test_two_trash_with_one_keeper_in_each() -> None:
    ov = _StubOverlay(
        [
            _StubRegion(0, 10, "trash"),
            _StubRegion(20, 30, "trash"),
            _StubRegion(3, 5, "keeper"),
            _StubRegion(25, 28, "keeper"),
        ]
    )
    assert ov.trash_minus_keepers() == [
        (0.0, 3.0),
        (5.0, 10.0),
        (20.0, 25.0),
        (28.0, 30.0),
    ]


# ====================================================================
# Pin: returned list is sorted ascending (used by audio-thread scan)
# ====================================================================
def test_result_sorted_ascending() -> None:
    """Carved sub-ranges must be sorted — the audio-thread _callback
    is O(N) over the list; downstream binary-search optimization
    requires sorted input."""
    ov = _StubOverlay(
        [
            _StubRegion(40, 50, "trash"),
            _StubRegion(0, 10, "trash"),
            _StubRegion(20, 30, "trash"),
            _StubRegion(22, 26, "keeper"),
        ]
    )
    out = ov.trash_minus_keepers()
    starts = [s for s, _ in out]
    assert starts == sorted(starts)
