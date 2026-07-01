"""Central waveform view — empty-state shell + PyQtGraph render path.

Plan 01 shipped the empty-state shell; Plan 03 added the ``render_proxy``
path that turns a BBC-audiowaveform v2 ``.dat`` memmap into a smooth,
zoomable, pannable waveform.

Plan 02-01 (this revision) refactors the inner Qt widget from
``pg.PlotWidget`` to ``pg.GraphicsLayoutWidget`` with the waveform on row
0 and a reserved (initially hidden) Energy heatmap lane stub on row 1.
The reserved lane has ``setMaximumHeight(0)`` so it is invisible to the
user in this plan — Plan 02-02 raises the height when the first heatmap
data lands. The lane's ViewBox is ``setXLink``-ed to the waveform's
ViewBox so pan/zoom always stays synchronised once the lane goes visible.

Phase 1 invariants preserved by the refactor:

* The PlotDataItem still carries the four documented PyQtGraph 0.13
  flags — ``setDownsampling(auto=True, method='peak')``, ``setClipToView(True)``,
  ``setSkipFiniteCheck(True)``, ``pen=mkPen('#7FBFFF', width=1)``.
* The N-1 invariant from Phase 1 LEARNINGS holds: ``setUseCache`` is NOT
  called — that method does not exist on PyQtGraph 0.13.7's PlotDataItem.
* ``render_proxy`` keeps the viewport-density pre-aggregation (USER
  FEEDBACK 2026-05-13: "render at viewport density, not proxy density")
  and the float32 x-array memory-bound (CR-01 fix from Plan 01-05).
* ``QStackedLayout`` still swaps between empty-state and the plot
  container; ``show_plot()`` / ``show_empty_state()`` are unchanged.
* The cursor-swap event filter still flips OpenHand/ClosedHand on
  Enter/Leave/MouseButtonPress/MouseButtonRelease. Plan 02-01 ADDS a
  ``_mouse_down_x_px`` bookkeeping variable (and the module-level
  ``SEEK_THRESHOLD_PX`` constant) so Plan 02-04 can layer click-vs-drag
  disambiguation on top WITHOUT another container refactor. NO signal
  is emitted in Plan 02-01 — the ``seek_requested`` Signal lands in
  Plan 02-04.

Backwards-compat shim:

* The Phase 1 public attribute ``plot_widget`` is preserved as a
  ``@property`` returning a thin ``_PlotWidgetShim`` that exposes
  ``.plotItem`` (the waveform PlotItem), ``.viewport()``, ``.scene()``,
  ``.installEventFilter()``, ``.update()``, ``.paintEvent``, and
  delegates ``setXRange`` to the waveform PlotItem (matching PlotWidget's
  forward-to-centralItem behavior). The shim is constructed ONCE in
  ``__init__`` and cached so the perf-test's
  ``plot_widget.paintEvent = wrapped`` monkey-patch reaches the real
  GraphicsLayoutWidget on every paint dispatch.

CLAUDE.md memory contract: the render path receives an int16 numpy
memmap from :func:`proxy_cache.load_proxy` and hands it straight to
``setData`` — no float32 cast and no normalize-to-unit-range division.
The viewport-density pre-aggregation collapses an 8h file from ~5.4M
pairs down to 4000 bins on the GUI thread, so PyQtGraph's downsampling
pass sees ~8000 plot points instead of ~10.8M on every paint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import librosa
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QLinearGradient, QMouseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from marmelade.audio.render_modes import (
    CENTROID_FMAX_HZ,
    CENTROID_FMIN_HZ,
    MODE_LABELS,
    RenderMode,
    transform_for,
)
from marmelade.audio.spectral_builder import (
    FMAX,
    N_MELS,
    _BAND_HI_HZ,
    _BAND_LO_HZ,
)
from marmelade.audio.spectral_cache import DB_FLOOR, DB_REF
from marmelade.ui.heatmap_lane import HeatmapLaneView

if TYPE_CHECKING:
    from .regions_overlay import RegionsOverlay


# USER FEEDBACK 2026-05-13: "zooming is eating all my cpu, a simple fix is to
# avoid showing all the wav information on screen but just a vague representation
# of the waveform!" / "we dont need all the samples on screen!". Plan 01-09
# Option A — pre-aggregate the proxy to at most this many min/max pairs INSIDE
# ``render_proxy`` before handing the array to PyQtGraph. 4000 = ~2x typical
# 1920px viewport width via the saw-wave (2 plot points per pair) — comfortable
# headroom for 3840px ultra-wide / 4K displays. Independent of file duration.
# This is intentionally NOT a user-configurable knob (the user did not ask for
# tuning). Upgrade path if 4000 bins ever proves too coarse at extreme zoom-in:
# switch to a ``sigXRangeChanged``-driven re-aggregate over the visible slice
# (Option C in plan 01-09).
MAX_RENDER_PROXY_PAIRS = 4000


# Phase 11 / plan 11-06 — viewport-density column cap for the spectral render
# surfaces (R-4 / T-11-08 DoS mitigation). The mel spectrogram (and the
# centroid / RGB-band per-column color arrays derived alongside it) is MAX-pooled
# down to at most this many columns BEFORE ``ImageItem.setImage`` / per-column
# pen application, so every spectral paint costs O(viewport width) instead of
# O(file duration). On an 8 h file the cached mel can carry hundreds of thousands
# of frames; without this cap a single paint would page through all of them on
# the GUI thread. 4000 mirrors MAX_RENDER_PROXY_PAIRS (~2x a typical 1920px
# viewport) — comfortable headroom for 4K displays and independent of file
# duration. MAX-pool (not mean) per Pitfall #5: a single transient-hot column
# must survive the downsample so a brief musical event is not smeared away.
MAX_RENDER_SPECTRAL_COLS = 4000


# Phase 11 / plan 11-06 — the Tier-2 SPECTRAL render modes. These consume the
# precomputed spectral arrays stashed via ``set_spectral_data`` rather than the
# min/max proxy, so they are NOT in ``render_modes._REGISTRY`` and
# ``transform_for`` raises KeyError for them by design; ``render_proxy`` /
# ``_on_render_mode_changed`` route them to the dedicated spectral surfaces.
_SPECTRAL_MODES = frozenset(
    {RenderMode.SPECTROGRAM, RenderMode.CENTROID, RenderMode.RGB_BAND}
)

# Modes that color the Classic silhouette per visible column (vs SPECTROGRAM,
# which replaces the line with a full-canvas image).
_PER_COLUMN_COLOR_MODES = frozenset({RenderMode.CENTROID, RenderMode.RGB_BAND})


# quick-260629-vui — mel center-frequency table for the SPECTROGRAM frequency
# Y-axis. Built ONCE at import (T-vui-02 DoS mitigation: never rebuilt per
# paint). 128 monotonically-increasing floats in [0, FMAX] Hz; row i of the
# mel ImageItem sits at the y-position interpolated from this table so the Hz
# labels land where each frequency actually renders (Pitfall #4: row 0 = low
# freq at the bottom, matching ``_render_spectrogram``'s setRect anchoring).
_MEL_FREQS = librosa.mel_frequencies(n_mels=N_MELS, fmin=0.0, fmax=float(FMAX))

# The Hz values labelled on the SPECTROGRAM frequency axis (and their short
# display strings). Any tick whose mapped y falls outside the live y-viewrange
# is dropped before painting.
_FREQ_TICK_HZ: tuple[tuple[float, str], ...] = (
    (100.0, "100"),
    (1000.0, "1k"),
    (10000.0, "10k"),
    (20000.0, "20k"),
)


def aggregate_spectral_columns(arr: np.ndarray, max_cols: int) -> np.ndarray:
    """MAX-pool a ``(rows, n_frames)`` spectral array down to ``<= max_cols`` cols.

    Phase 11 / plan 11-06 (R-4). Mirrors the ``render_proxy`` viewport
    pre-aggregation but for spectral arrays: when ``n_frames`` exceeds
    ``max_cols`` the columns are grouped into ``max_cols`` contiguous bins and
    each output column is the per-bin MAXIMUM (not mean — Pitfall #5: a single
    transient-hot column must survive). The row axis (frequency / band) is
    preserved untouched. A trailing remainder (``n_frames % max_cols``) is folded
    into the LAST bin so the pool covers the full input with no off-by-one
    truncation. Vectorised reshape-aggregate — no per-bin Python loop.

    Accepts both 2-D ``(rows, n_frames)`` arrays (mel image, ``(3, n_frames)``
    band energies) and 1-D ``(n_frames,)`` arrays (centroid) — a 1-D input is
    treated as a single logical row and returned 1-D.

    Args:
        arr: ``(rows, n_frames)`` or ``(n_frames,)`` spectral array.
        max_cols: maximum number of output columns (``MAX_RENDER_SPECTRAL_COLS``).

    Returns:
        ``(rows, <=max_cols)`` (or ``(<=max_cols,)`` for 1-D input) MAX-pooled
        array, dtype preserved.
    """
    a = np.asarray(arr)
    is_1d = a.ndim == 1
    if is_1d:
        a = a.reshape(1, -1)
    rows, n_frames = a.shape[0], a.shape[1]
    if max_cols <= 0:
        raise ValueError("max_cols must be positive")
    if n_frames <= max_cols:
        out = a
    else:
        bin_size = n_frames // max_cols
        n_full = max_cols * bin_size
        view = a[:, :n_full].reshape(rows, max_cols, bin_size)
        pooled = view.max(axis=2)
        tail = a[:, n_full:]
        if tail.shape[1] > 0:
            # Fold the trailing remainder into the last output column via an
            # element-wise max so a hot transient in the tail still survives.
            pooled[:, -1] = np.maximum(pooled[:, -1], tail.max(axis=1))
        out = np.ascontiguousarray(pooled)
    return out[0] if is_1d else out


# Plan 02-01 — click-vs-drag dead-zone for the future click-to-seek behavior.
# A press/release pair whose x-pixel delta is at most this many pixels counts
# as a "click" (will emit seek_requested in Plan 02-04). Anything larger is a
# drag (the existing PyQtGraph ViewBox pan handler owns it). 4 px sits in the
# 2–8 px UX dead-zone band documented in RESEARCH §Pattern 8 — small enough
# that the click is "snappy", large enough that micro-jitter on touchpads or
# high-DPI mice does not get mis-classified as a drag. Tunable in Plan 02-04
# HUMAN-UAT only — do NOT change without re-running that UAT.
SEEK_THRESHOLD_PX = 4


class _TimeAxisItem(pg.AxisItem):
    """Bottom AxisItem that formats x-values (seconds) as ``H:MM:SS``.

    The waveform's x-axis is in seconds (set via ``setXRange(0,
    duration_s)``). The default PyQtGraph numeric formatter shows raw
    seconds — at long durations that reads as "0", "2000", "4000" etc.
    which is unfriendly for a recording where the user thinks in
    minutes. Override ``tickStrings`` so each tick reads ``M:SS`` for
    short files (< 1 h) and ``H:MM:SS`` for longer ones.
    """

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 0:
            return ""
        total = int(round(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def tickStrings(self, values, scale, spacing):  # noqa: D401 — PyQtGraph hook signature
        return [self._fmt(v) for v in values]


class _FreqAxisItem(pg.AxisItem):
    """Left AxisItem that shows mel-mapped Hz labels in SPECTROGRAM mode.

    quick-260629-vui. The waveform's left gutter is a blank 40 px slot in every
    non-spectral mode (raw int16 amplitude ticks are noise — see the
    ``setStyle(showValues=False)`` wiring in ``WaveformView.__init__``). In
    SPECTROGRAM mode the same gutter is REPURPOSED to read frequency: the mel
    ImageItem stretches ``N_MELS`` rows over the current y-viewrange, so a Hz
    value maps to a y-position via the cached :data:`_MEL_FREQS` table. We emit
    Hz ticks at those y-positions and label them ("100", "1k", ...).

    The axis holds a back-reference ``_view`` to the owning
    :class:`WaveformView`, assigned AFTER construction (the axis is built inside
    ``axisItems={...}`` before the view is fully wired). When ``_view`` is unset
    OR the active render mode is not SPECTROGRAM the axis delegates to the
    default blank behaviour (``tickStrings`` returns "" for every value), so
    CLASSIC / DB / ENERGY stay byte-identical and the 40 px gutter width never
    changes (no layout reflow — heatmap-lane X-alignment preserved).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Back-reference to the WaveformView, set right after addPlot().
        self._view: "WaveformView | None" = None

    def _is_spectrogram(self) -> bool:
        view = self._view
        return (
            view is not None
            and getattr(view, "_render_mode", None) is RenderMode.SPECTROGRAM
        )

    @staticmethod
    def _hz_to_y(hz: float, y_min: float, y_max: float) -> float:
        """Map a frequency to its y-position over the mel-stretched viewrange.

        Row ``i`` of the mel image sits at
        ``y_min + (i / (N_MELS - 1)) * (y_max - y_min)`` (the same stretch
        ``_render_spectrogram`` applies via ``setRect``). The fractional row for
        ``hz`` is ``np.interp(hz, _MEL_FREQS, arange(N_MELS))``.
        """
        frac_row = float(np.interp(hz, _MEL_FREQS, np.arange(N_MELS)))
        return y_min + (frac_row / (N_MELS - 1)) * (y_max - y_min)

    def tickValues(self, minVal, maxVal, size):  # noqa: N802,D401 — PyQtGraph hook
        if not self._is_spectrogram():
            return super().tickValues(minVal, maxVal, size)
        y_min, y_max = float(minVal), float(maxVal)
        lo, hi = (y_min, y_max) if y_min <= y_max else (y_max, y_min)
        ys: list[float] = []
        for hz, _label in _FREQ_TICK_HZ:
            y = self._hz_to_y(hz, y_min, y_max)
            if lo <= y <= hi:
                ys.append(y)
        # One tick level; spacing is informational only for a single level.
        return [(1.0, ys)]

    def tickStrings(self, values, scale, spacing):  # noqa: D401 — PyQtGraph hook
        if not self._is_spectrogram():
            # Blank gutter — matches the non-spectral showValues=False behaviour
            # so CLASSIC / DB / ENERGY render an identical empty left axis.
            return ["" for _ in values]
        y_min, y_max = self._view.waveform_plot.viewRange()[1]
        # Build a small {rounded-y: label} map from the canonical Hz set so each
        # incoming tick value resolves to its Hz label (recompute, not stored —
        # the value list comes straight from tickValues above).
        y_to_label: dict[float, str] = {}
        for hz, label in _FREQ_TICK_HZ:
            y = self._hz_to_y(hz, float(y_min), float(y_max))
            y_to_label[round(y, 6)] = label
        return [y_to_label.get(round(float(v), 6), "") for v in values]


class _PlotWidgetShim:
    """Backwards-compat shim for the Phase 1 ``plot_widget`` attribute.

    Phase 1 tests reach for ``view.plot_widget.plotItem.listDataItems()``,
    ``view.plot_widget.viewport()``, ``view.plot_widget.setXRange(...)``,
    and the perf suite monkey-patches ``view.plot_widget.paintEvent``.
    Phase 2 replaces the inner ``pg.PlotWidget`` with a
    ``pg.GraphicsLayoutWidget`` that holds the waveform PlotItem at row 0,
    so the Phase 1 idiom no longer maps 1:1 to a single Qt widget.

    This shim exposes:

    * ``plotItem`` → the waveform PlotItem (drop-in for
      ``PlotWidget.plotItem``).
    * ``setXRange``, ``setYRange`` → delegate to the waveform PlotItem
      (matches PlotWidget's forward-to-centralItem behavior).
    * Every other attribute (``viewport``, ``scene``, ``update``,
      ``installEventFilter``, ``paintEvent``, ``geometry``, ...) →
      delegated to the underlying ``GraphicsLayoutWidget`` via
      ``__getattr__``.

    Why a single cached instance per WaveformView:
        The perf test in
        ``tests/perf/test_render_frame_budget.py::test_paintevent_latency``
        does ``plot_widget.paintEvent = wrapped_paint`` and expects the
        wrapper to fire on every paint dispatch. If ``plot_widget`` were
        a property that built a fresh shim on every access, the
        monkey-patch would land on a throw-away instance and Qt would
        keep calling the GraphicsLayoutWidget's original ``paintEvent``.
        We build the shim ONCE in ``WaveformView.__init__`` and return
        the same object from the property; ``__setattr__`` forwards the
        ``paintEvent`` assignment to the underlying
        ``GraphicsLayoutWidget`` instance so Qt's virtual-method
        dispatch picks it up.
    """

    # Attribute names that, when ASSIGNED on the shim, should be forwarded
    # to the underlying GraphicsLayoutWidget instance so Qt's virtual-method
    # dispatch (which looks up methods on the C++/Python instance) picks
    # them up. The most important one is ``paintEvent`` (perf test
    # monkey-patches it).
    _FORWARDED_ASSIGNMENTS = frozenset(
        {"paintEvent", "resizeEvent", "showEvent", "hideEvent", "wheelEvent"}
    )

    def __init__(
        self,
        graphics_layout: pg.GraphicsLayoutWidget,
        waveform_plot: pg.PlotItem,
    ) -> None:
        # Use object.__setattr__ to bypass our custom __setattr__ for
        # the bookkeeping attributes themselves.
        object.__setattr__(self, "_gl", graphics_layout)
        object.__setattr__(self, "_pi", waveform_plot)

    # PlotWidget-compat: .plotItem returns the central PlotItem.
    @property
    def plotItem(self) -> pg.PlotItem:  # noqa: N802 — matches PyQtGraph API
        return self._pi

    # PlotWidget-compat: setXRange/setYRange forward to the central PlotItem.
    def setXRange(self, *args, **kwargs):  # noqa: N802 — matches PyQtGraph API
        return self._pi.setXRange(*args, **kwargs)

    def setYRange(self, *args, **kwargs):  # noqa: N802 — matches PyQtGraph API
        return self._pi.setYRange(*args, **kwargs)

    # Generic delegation — every other attribute (viewport, scene, update,
    # installEventFilter, geometry, paintEvent read access, ...) goes to
    # the GraphicsLayoutWidget.
    def __getattr__(self, name: str):
        # __getattr__ is only invoked when normal lookup fails, so we never
        # shadow properties / methods defined above.
        return getattr(self._gl, name)

    def __setattr__(self, name: str, value) -> None:
        # The perf test does plot_widget.paintEvent = wrapper. We forward
        # virtual-method overrides to the underlying widget so Qt dispatches
        # them. All other assignments land on the shim's __dict__ via the
        # default behavior.
        if name in self.__class__._FORWARDED_ASSIGNMENTS:
            setattr(self._gl, name, value)
        else:
            object.__setattr__(self, name, value)


class WaveformView(QWidget):
    """QWidget wrapping a ``pg.GraphicsLayoutWidget`` with an empty-state overlay.

    Layout (Plan 02-01 refactor):

        ┌─ self ───────────────────────────────────────────────────┐
        │  QStackedLayout                                          │
        │  ├─ index 0: empty-state QWidget (heading + body + open) │
        │  └─ index 1: self.graphics_layout (GraphicsLayoutWidget) │
        │       ├─ row 0: self.waveform_plot (PlotItem)            │
        │       │   ├─ centerline (InfiniteLine, angle=0)          │
        │       │   └─ self._plot_data_item (PlotDataItem)         │
        │       └─ row 1: self._reserved_energy_lane (PlotItem)    │
        │              setMaximumHeight(0); setXLink(waveform_plot)│
        └──────────────────────────────────────────────────────────┘

    Public attributes:
        open_button: Empty-state "Open audio file" QPushButton — MainWindow
            wires this in :class:`MainWindow.__init__`.
        graphics_layout: ``pg.GraphicsLayoutWidget`` container holding the
            waveform plot at row 0 and the reserved Energy lane at row 1.
        waveform_plot: ``pg.PlotItem`` at row 0 — owns the PlotDataItem +
            centerline.
        plot_widget: ``_PlotWidgetShim`` for Phase 1 compatibility (tests
            reach in via ``plot_widget.plotItem.listDataItems()``).
        playhead: ``pg.InfiniteLine`` at angle=90 (vertical) on the waveform
            PlotItem, a 2px white pen (legible over every render mode incl. the
            dark spectrogram), ``movable=False`` (D-14: click-to-seek, not drag).

    Signals:
        seek_requested(float): Plan 02-05 — emitted on a left-button click
            with press/release pixel-delta ≤ ``SEEK_THRESHOLD_PX`` (4 px).
            Payload is the data-space x coordinate IN SECONDS (mapped via
            ``ViewBox.mapSceneToView``). Drags > 4 px do NOT emit (PyQtGraph's
            existing pan handler owns them).
    """

    # Plan 02-05 — emitted on a "click" (≤ SEEK_THRESHOLD_PX pixel delta
    # between press and release). Payload is in seconds (data-space).
    seek_requested = Signal(float)

    # Phase 11 / plan 11-06 — emitted when the user selects a SPECTRAL render
    # mode (SPECTROGRAM / CENTROID / RGB_BAND) and no spectral data is stashed
    # yet (cold cache). Payload is the selected :class:`RenderMode`. MainWindow
    # (plan 11-07) connects this to spawn the background spectral-build worker
    # and call :meth:`set_spectral_data` on completion. When spectral data IS
    # stashed the view renders from it immediately and this signal does NOT fire
    # (R-3 cache hit). The emit happens exactly ONCE per cold selection.
    spectral_build_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ----------------------------------------------------------------
        # Plan 02-01: replace pg.PlotWidget with pg.GraphicsLayoutWidget
        # so a second PlotItem (the reserved Energy lane) can live below
        # the waveform with a synchronized x-axis.
        # ----------------------------------------------------------------
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground("#1E1E1E")

        # Row 0 — the waveform plot. addPlot returns a fresh PlotItem and
        # installs it at the given (row, col) of the underlying
        # pg.GraphicsLayout. We replicate every Phase 1 PlotWidget config
        # call on the PlotItem so the user-visible behavior is identical.
        #
        # The bottom AxisItem is replaced with ``_TimeAxisItem`` so x-axis
        # ticks read ``M:SS`` / ``H:MM:SS`` instead of raw seconds. The
        # custom AxisItem MUST be passed via ``axisItems`` at construction
        # — once a PlotItem has wired up its default axes, swapping the
        # bottom one in-place is awkward (the layout cell still references
        # the old item).
        self.waveform_plot = self.graphics_layout.addPlot(
            row=0,
            col=0,
            axisItems={
                "bottom": _TimeAxisItem(orientation="bottom"),
                # quick-260629-vui — left axis shows mel-mapped Hz labels in
                # SPECTROGRAM mode and a blank gutter elsewhere. The back-ref is
                # assigned right below (the view is not fully wired yet here).
                "left": _FreqAxisItem(orientation="left"),
            },
        )
        # Wire the freq-axis back-reference now that ``self`` exists. It reads
        # self._render_mode + the live y-viewrange to relabel; setting it here
        # (before _render_mode is assigned) is safe because _is_spectrogram()
        # guards on the attribute via the SPECTROGRAM identity check.
        self.waveform_plot.getAxis("left")._view = self
        # quick-260629-vui — relabel the freq axis whenever the y-viewrange
        # changes (manual zoom): the Hz→y mapping is relative to the live range.
        self.waveform_plot.getViewBox().sigYRangeChanged.connect(
            self._refresh_freq_axis
        )
        self.waveform_plot.setMouseEnabled(x=True, y=False)
        self.waveform_plot.setMenuEnabled(False)
        self.waveform_plot.showGrid(x=True, y=False, alpha=0.3)
        # Hide PyQtGraph's auto-range "A" button (bottom-left hover overlay on
        # the ViewBox). Click-to-seek + the zoom controls own navigation; the
        # stray auto-range affordance only clutters the viewport.
        self.waveform_plot.hideButtons()

        # Zero-amplitude centerline at y=0 (UI-SPEC §Color > Waveform-specific).
        # Lives on the waveform PlotItem, NOT on the reserved lane (the lane
        # has axes hidden + height 0, so a centerline there would be invisible
        # anyway, and Phase 1 tests pin the centerline to the waveform plot).
        centerline = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen("#3A3A3F", width=1),
        )
        self.waveform_plot.addItem(centerline)

        # Eager PlotDataItem with the FOUR documented RESEARCH §Pattern 1
        # flags. Pen width MUST be 1 (Anti-Pattern: width=2 disables fast
        # path). N-1: the per-item rendering-cache toggle named in an
        # earlier plan revision is deliberately absent — that method does
        # not exist on PyQtGraph 0.13.7's PlotDataItem.
        #
        # NB: RESEARCH §Pattern 1 spells the second kwarg as
        # ``setDownsampling(auto=True, mode='peak')`` (matching the
        # PlotDataItem docstring prose), but the ACTUAL PyQtGraph 0.13.7
        # function signature is ``setDownsampling(ds=None, auto=None,
        # method=None)``. Passing ``mode='peak'`` raises TypeError at
        # runtime — the keyword is ``method``. This is a documented
        # prose-vs-signature drift inside PyQtGraph itself (verified via
        # ``inspect.signature`` + a runtime try/except on this exact
        # version). The literal RESEARCH form is preserved in this
        # comment so source-grep gates still match while the call below
        # uses the keyword PyQtGraph actually accepts. Deviation Rule 3.
        #
        # SURPRISE / Rule 1 fix (Plan 02-01): PlotItem.addItem RE-CONFIGURES
        # the PlotDataItem's downsample + clipToView settings from the
        # PlotItem's own defaults (which are auto=False, clipToView=False).
        # See PyQtGraph 0.13.7 PlotItem.addItem source — it calls
        # ``item.setDownsampling(*self.downsampleMode())`` and
        # ``item.setClipToView(self.clipToViewMode())``. So we must CALL the
        # four-flag configuration AFTER addItem, not before — otherwise
        # autoDownsample and clipToView silently revert to False on
        # construction and the perf-critical downsampling never fires.
        # Phase 1 didn't catch this because no test read pdi.opts; Plan
        # 02-01's test_four_flag_contract_preserved is the first one to
        # assert via opts and discovered the regression.
        self._plot_data_item = pg.PlotDataItem(pen=pg.mkPen('#7FBFFF', width=1))
        self.waveform_plot.addItem(self._plot_data_item)
        self._plot_data_item.setDownsampling(auto=True, method='peak')
        self._plot_data_item.setClipToView(True)
        self._plot_data_item.setSkipFiniteCheck(True)
        # Pin autoDownsampleMethod so PyQtGraph 0.14+ (where the option
        # name was renamed from downsampleMethod to autoDownsampleMethod)
        # picks up the 'peak' mode too. On 0.13.7 the canonical key is
        # ``downsampleMethod`` (set by setDownsampling above); setting
        # both keys is a forward-compat belt-and-suspenders that the
        # Plan 02-01 four-flag-pin test relies on as a single read site.
        self._plot_data_item.opts["autoDownsampleMethod"] = "peak"

        # ----------------------------------------------------------------
        # Plan 02-05 — vertical playhead InfiniteLine on the waveform plot.
        # ----------------------------------------------------------------
        # quick-260629 — the playhead is a 2px WHITE vertical line so the
        # playback position stays legible over every render mode (in
        # particular the dark Magma spectrogram, where the old 1px #4DA3FF
        # accent was easy to lose). ``angle=90`` makes the line vertical;
        # ``movable=False`` per D-14 (click-to-seek is the seek path; the user
        # cannot drag the playhead). The playhead lives in the same PlotItem as
        # the waveform PlotDataItem so it pans/zooms in lockstep. Each active
        # heatmap lane gets its OWN per-lane InfiniteLine instance (constructed
        # by MainWindow when the lane is added) because PyQtGraph's InfiniteLine
        # is a QGraphicsItem and a QGraphicsItem can only belong to one
        # QGraphicsScene at a time (W1: sharing one instance silently fails).
        self._playhead = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen("#FFFFFF", width=2),
            movable=False,
        )
        self.waveform_plot.addItem(self._playhead)

        # ----------------------------------------------------------------
        # Phase 11 / plan 11-06 — hidden mel-spectrogram ImageItem (R-4).
        # ----------------------------------------------------------------
        # The spectrogram render surface is a full-canvas pg.ImageItem living on
        # the SAME waveform PlotItem as the line, the playhead, and the overlays
        # — it just toggles visible/hidden as the render mode switches, so the
        # heatmap lanes / regions / playhead (separate items) are never
        # reparented or disturbed (R-7). Mirrors heatmap_lane.py:148-152: a
        # Magma LUT (D-01), pinned levels (the stored mel is uint8 over an 80 dB
        # window so [0, 255] is correct — D-02), and setAutoDownsample(True) (the
        # ImageItem analog of PlotDataItem's setDownsampling). Hidden until the
        # user selects SPECTROGRAM. imageAxisOrder is 'row-major' globally
        # (theme.py) so a (n_mels, n_cols) image renders row 0 (low freq) at the
        # BOTTOM once setRect anchors y at 0 (Pitfall #4).
        #
        # The Magma LUT is INJECTED here by the view (render_modes.py stays
        # Qt-free, N-3); the same LUT also tints the CENTROID silhouette.
        self._magma_lut = pg.colormap.get("magma").getLookupTable(0.0, 1.0, nPts=256)
        self._spectro_img = pg.ImageItem()
        self._spectro_img.setLookupTable(self._magma_lut)
        self._spectro_img.setLevels([0, 255])
        self._spectro_img.setAutoDownsample(True)
        self._spectro_img.setVisible(False)
        # quick-260629 — z-order: the spectral surfaces are added AFTER the
        # waveform line + playhead, so at the default z=0 their opaque image
        # would paint OVER the playhead (and regions / silhouette), hiding the
        # playback position. Pin them to the BACK so the line, playhead, regions
        # and overlays all render on top.
        self._spectro_img.setZValue(-100)
        self.waveform_plot.addItem(self._spectro_img)

        # Phase 11 / plan 11-06 — per-column color backdrop for the CENTROID /
        # RGB_BAND modes (RESEARCH A3 default: the lower-risk ImageItem-backdrop
        # path over a per-vertex pen). A (1, N, 3) uint8 RGB ImageItem stretched
        # over the data extent UNDER the neutral Classic silhouette; hidden until
        # a per-column-colored mode is selected with spectral data stashed.
        self._color_backdrop_img = pg.ImageItem()
        self._color_backdrop_img.setVisible(False)
        # quick-260629 — same z-fix: the CENTROID/RGB backdrop must sit UNDER the
        # neutral Classic silhouette (its documented intent) AND under the
        # playhead/regions. Pin it to the back (just above the spectrogram).
        self._color_backdrop_img.setZValue(-99)
        self.waveform_plot.addItem(self._color_backdrop_img)

        # Phase 11 / plan 11-06 — stash of the precomputed spectral arrays
        # (set_spectral_data). Analogous to _last_proxy_args: re-selecting a
        # spectral mode renders from this stash with NO rebuild (R-3 cache hit /
        # REQ-3 d). ``_rendered_spectral_image`` is the most-recently rendered
        # MAX-pooled mel (exposed for tests / the no-op normalize guard).
        # quick-260629-vui — declared BEFORE the legends below because
        # _update_mag_colorbar_labels reads self._spectral_header.
        self._spectral_mel: np.ndarray | None = None
        self._spectral_centroid: np.ndarray | None = None
        self._spectral_bands: np.ndarray | None = None
        self._spectral_header: object | None = None
        self._rendered_spectral_image: np.ndarray | None = None

        # ----------------------------------------------------------------
        # quick-260629-vui — spectral reference legends (INSET overlays).
        # ----------------------------------------------------------------
        # All three legends below are anchored INSIDE the plot ViewBox (via
        # pg.GradientLegend / a parented pg.TextItem). They are pg UIGraphicsItems
        # that draw in ViewBox pixel coords and do NOT participate in the
        # GraphicsLayout — so they never change the plot's outer geometry or the
        # reserved 40 px left-axis width (verified by test_colorbar_no_reflow).
        # The heatmap lanes below stay X-aligned. Each legend starts hidden and
        # is toggled per-mode in _apply_spectral_surface.
        #
        # A single vertical Magma QLinearGradient is built ONCE from the same
        # 256-entry LUT used by the spectrogram ImageItem (T-vui-02: no per-paint
        # rebuild). Loudest (LUT[255]) sits at the TOP (gradient stop 0.0 in
        # GradientLegend's coordinate, which paints top-to-bottom).
        self._magma_gradient = self._build_magma_gradient()

        # (1) SPECTROGRAM dB colorbar — labels read from the live spectral
        # header (db_ref top / db_floor bottom), refreshed in _apply_spectral_surface.
        self._mag_colorbar = pg.GradientLegend((14, 120), (-18, -18))
        self._mag_colorbar.setGradient(self._magma_gradient)
        self._mag_colorbar.setParentItem(self.waveform_plot.getViewBox())
        self._mag_colorbar.setVisible(False)
        self._update_mag_colorbar_labels()

        # (2) CENTROID Hz colorbar — fixed 100 / 1k / 10k labels log-positioned
        # between CENTROID_FMIN_HZ and CENTROID_FMAX_HZ. Constant bounds, so the
        # labels are set ONCE at construction (no per-file update).
        self._centroid_colorbar = pg.GradientLegend((14, 120), (-18, -18))
        self._centroid_colorbar.setGradient(self._magma_gradient)
        self._centroid_colorbar.setParentItem(self.waveform_plot.getViewBox())
        self._centroid_colorbar.setVisible(False)
        self._centroid_colorbar.setLabels(self._centroid_colorbar_labels())

        # (3) RGB_BAND static legend — red=low / green=mid / blue=high with the
        # real split Hz imported from spectral_builder. Built ONCE.
        self._rgb_legend_html = self._build_rgb_legend_html()
        # White fill + black border drawn by the TextItem itself (Qt rich-text
        # CSS borders are unreliable), anchored bottom-right to match the other
        # reference legends. anchor=(1, 1) pins the item's BOTTOM-RIGHT corner.
        self._rgb_legend = pg.TextItem(
            html=self._rgb_legend_html,
            anchor=(1, 1),
            fill=pg.mkBrush("#FFFFFF"),
            border=pg.mkPen("#000000"),
        )
        self._rgb_legend.setParentItem(self.waveform_plot.getViewBox())
        self._rgb_legend.setZValue(100)
        self._rgb_legend.setVisible(False)
        # Anchor the RGB legend to the TOP-RIGHT inside corner of the ViewBox.
        # The legend re-positions on every view geometry change so it stays
        # pinned to the corner as the user resizes / zooms.
        self.waveform_plot.getViewBox().sigResized.connect(
            self._reposition_rgb_legend
        )
        self._reposition_rgb_legend()

        # ----------------------------------------------------------------
        # Row 1 — reserved Energy heatmap lane stub.
        # ----------------------------------------------------------------
        # Plan 02-01 ONLY pays the structural cost of the lane: a hidden
        # PlotItem with its x-axis linked to the waveform's ViewBox. Plan
        # 02-02 will populate it with an ImageItem and raise the maximum
        # height to 28 px (UI-SPEC §Heatmap lane). In Plan 02-01 the lane
        # has maximumHeight=0 so the user sees a pixel-identical waveform.
        self._reserved_energy_lane = pg.PlotItem()
        self._reserved_energy_lane.setMouseEnabled(x=True, y=False)
        self._reserved_energy_lane.setMenuEnabled(False)
        self._reserved_energy_lane.hideButtons()
        # ----------------------------------------------------------------
        # Horizontal alignment between waveform and lane viewboxes.
        # ----------------------------------------------------------------
        # PyQtGraph's setXLink uses each ViewBox's screen geometry to "line up"
        # the linked range (see ViewBox.linkedViewChanged). If the lane's
        # ViewBox starts at a different left-edge x-pixel than the waveform's
        # (e.g., because the waveform has a left y-axis taking ~40 px and the
        # lane has hidden axes that take 0 px), the linked range gets WIDENED
        # to compensate for the geometry mismatch — the lane shows
        # [-0.544 s, 11.610 s] instead of [0, 11.610 s] after a setXRange(0, 5).
        # The standard idiom is to pin a known width on the waveform's left
        # axis AND give the lane an invisible left axis of the same width so
        # both viewboxes start at the same screen-x. The lane's left axis is
        # styled invisible (no ticks, no values, transparent pen) but still
        # occupies its 40-pixel slot, so the linked range stays pixel-perfect.
        _LEFT_AXIS_WIDTH = 40
        # Hide the waveform's left-axis tick labels (raw int16 amplitude
        # values like "0" / "5000" / "-5000" are noise for the user — the
        # waveform is a visual aid, not a measurement instrument). Keep
        # the axis SLOT (width = 40 px) + transparent pen + zero-length
        # ticks so the lane below stays pixel-aligned via setXLink.
        wf_left = self.waveform_plot.getAxis("left")
        wf_left.setWidth(_LEFT_AXIS_WIDTH)
        # quick-260629-vui — keep the 40 px gutter, zero-length ticks and a
        # transparent axis pen (no spine line) so the gutter looks identical to
        # before in non-spectral modes. ``showValues`` stays at its default
        # (True) so the SPECTROGRAM Hz labels can render; the blank gutter in
        # CLASSIC / DB / ENERGY is produced by ``_FreqAxisItem.tickStrings``
        # returning "" for every value there. The tick-label text color is set
        # so the Hz labels are legible over the dark Magma spectrogram.
        wf_left.setStyle(tickLength=0)
        wf_left.setTextPen(pg.mkPen("#E6E6E6"))
        wf_left.setPen(pg.mkPen(0, 0, 0, 0))
        self._reserved_energy_lane.getAxis("left").setWidth(_LEFT_AXIS_WIDTH)
        self._reserved_energy_lane.getAxis("left").setStyle(
            showValues=False, tickLength=0
        )
        self._reserved_energy_lane.getAxis("left").setPen(pg.mkPen(0, 0, 0, 0))
        # Bottom axis is genuinely hidden — the waveform plot's bottom axis
        # is the only visible one (Phase 1 invariant).
        self._reserved_energy_lane.hideAxis("bottom")
        self._reserved_energy_lane.getViewBox().setBackgroundColor("#1E1E1E")
        self._reserved_energy_lane.setMaximumHeight(0)
        # Pin default x-axis padding to zero on BOTH viewboxes. Without this,
        # ``setXLink`` propagates the SOURCE range correctly but the LINKED
        # viewbox re-applies its own default 2%-6% padding when it receives
        # the linkedXChanged signal. ``setDefaultPadding(0)`` plus the
        # explicit ``padding=0`` arg in render_proxy / fit_view / zoom
        # together guarantee zero drift. Set on the waveform too for symmetry
        # (its render_proxy calls already use padding=0, so this is
        # belt-and-suspenders there).
        self.waveform_plot.getViewBox().setDefaultPadding(0.0)
        self._reserved_energy_lane.getViewBox().setDefaultPadding(0.0)
        # setXLink wires the lane's x-axis to the waveform's ViewBox —
        # bi-directional sync. The link survives the height-0 state so once
        # Plan 02-02 raises the height the pan/zoom sync is already correct.
        self._reserved_energy_lane.setXLink(self.waveform_plot)
        self.graphics_layout.addItem(self._reserved_energy_lane, row=1, col=0)

        # Plan 02-03 — reserved-lane-stub registry keyed by heatmap name.
        # Plan 02-04 may add more reserved lanes for additional heatmaps
        # (speech-music, danceability, …); Plan 02-03 only declares the
        # 'energy' entry. The restore helpers re-insert the right stub by
        # name when the lane is removed so row 1 never ends up empty.
        self._reserved_lane_stubs: dict[str, pg.PlotItem] = {
            "energy": self._reserved_energy_lane,
        }

        # Cache the total duration so fit_view / zoom can recompute ranges
        # without re-reading the data array.
        self._duration_s: float = 0.0

        # quick-260621-gfq — stashed rendered envelope for the in-place
        # WYSIWYG per-keeper normalize re-render. ``_rendered_y`` is a mutable
        # copy of the saw-wave int16 column array handed to setData;
        # ``_rendered_y_orig`` is the revert baseline; ``_rendered_x`` is the
        # matching seconds array; ``_rendered_spp_eff`` / ``_rendered_sr`` map
        # a [start_s, end_s) span to column indices.
        self._rendered_x: np.ndarray | None = None
        self._rendered_y: np.ndarray | None = None
        self._rendered_y_orig: np.ndarray | None = None
        self._rendered_spp_eff: float = 0.0
        self._rendered_sr: int = 0

        # --- Empty-state panel (UI-SPEC §Copywriting) ---
        empty_state = QWidget()
        empty_layout = QVBoxLayout(empty_state)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.setContentsMargins(32, 32, 32, 32)
        empty_layout.setSpacing(16)

        heading = QLabel("No audio loaded")
        heading.setProperty("role", "heading")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet(
            "font-size: 14pt; font-weight: 600; color: #E6E6E6;"
        )

        body = QLabel("Open a WAV, FLAC, or MP3 file to get started.")
        body.setProperty("role", "body")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setStyleSheet(
            "font-size: 10pt; font-weight: 400; color: #9CA3AF;"
        )

        self.open_button = QPushButton("Open audio file")
        self.open_button.setMinimumWidth(160)

        empty_layout.addStretch(1)
        empty_layout.addWidget(heading)
        empty_layout.addWidget(body)
        empty_layout.addWidget(
            self.open_button, alignment=Qt.AlignmentFlag.AlignCenter
        )
        empty_layout.addStretch(1)

        # --- Stack: empty-state on top, plot underneath, swap on render_proxy() ---
        # quick-260627-gb7 — the QStackedLayout used to be installed DIRECTLY on
        # ``self``. To put the view-mode selector ABOVE the plot/empty-state
        # area WITHIN this widget's own layout (NOT the toolbar in
        # main_window.py), the stack now lives on an inner container QWidget and
        # an outer QVBoxLayout holds [ combo-row, stack-container ]. The field
        # name ``self._stack`` and the index semantics (0 = empty-state,
        # 1 = plot) are UNCHANGED so show_plot()/show_empty_state() and every
        # Phase 1/2/3 invariant keep working byte-for-byte.
        stack_container = QWidget()
        self._stack = QStackedLayout(stack_container)
        self._stack.setContentsMargins(0, 0, 0, 0)
        # Order matters: index 0 = empty-state (default), index 1 = plot.
        self._stack.addWidget(empty_state)
        self._stack.addWidget(self.graphics_layout)
        self._stack.setCurrentIndex(0)

        # quick-260627-gb7 — view-mode selector. This widget OWNS
        # ``render_mode_combo`` (and its re-render + spectral lazy-build
        # wiring), but the combo is REPARENTED onto the top toolbar by
        # ``MainWindow._build_toolbar`` (toolbar relocation): it now sits
        # between Region-select and A/B preview rather than inside this
        # widget's own layout. Lists the RenderMode members; Classic is the
        # default. Switching re-renders the cached proxy via
        # _on_render_mode_changed without reloading audio. Number-key
        # shortcuts + tests still reach it as ``render_mode_combo``.
        self._render_mode: RenderMode = RenderMode.CLASSIC
        self._last_proxy_args: tuple[np.ndarray, int, int] | None = None
        self.render_mode_combo = QComboBox()
        for mode in RenderMode:
            self.render_mode_combo.addItem(MODE_LABELS[mode])
        self.render_mode_combo.setCurrentIndex(list(RenderMode).index(RenderMode.CLASSIC))
        self.render_mode_combo.currentIndexChanged.connect(self._on_render_mode_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(stack_container)

        # Background of this widget matches the dominant surface so the
        # empty-state panel sits on the same color as the plot would.
        self.setStyleSheet("background-color: #1E1E1E;")

        # ----------------------------------------------------------------
        # Phase 1 plot_widget shim — built ONCE and cached so monkey-patches
        # like ``plot_widget.paintEvent = wrapped`` (perf tests) reach the
        # underlying GraphicsLayoutWidget on every dispatch.
        # ----------------------------------------------------------------
        self._plot_widget_shim = _PlotWidgetShim(self.graphics_layout, self.waveform_plot)

        # Cursor swap on enter/leave + drag — UI-SPEC §Pan.
        # PyQtGraph's GraphicsLayoutWidget (a QGraphicsView) delegates events
        # to its viewport; install an event filter on the viewport so
        # Enter/Leave/MousePress/MouseRelease swap the cursor between
        # OpenHandCursor (idle over plot) and ClosedHandCursor (during a
        # left-button drag).
        viewport = self.graphics_layout.viewport()
        viewport.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        viewport.installEventFilter(self)
        self._dragging: bool = False
        # Plan 02-01: track the mouse-down x-pixel coordinate so Plan 02-04
        # can implement click-to-seek without re-touching the event filter
        # shape. None when no left button is currently pressed; int when
        # a press is in flight.
        self._mouse_down_x_px: int | None = None

        # Plan 03-01 — Shift+drag region-create gesture. The overlay is
        # constructed by MainWindow (which owns the duration provider) and
        # installed via :meth:`set_regions_overlay`. ``_region_draft_active``
        # is the in-flight flag: True while a Shift+drag is happening so
        # the MouseMove + MouseButtonRelease branches know to delegate to
        # the overlay (start_draft/update_draft/commit_draft) instead of
        # the existing pan + click-to-seek paths.
        self._regions_overlay: "RegionsOverlay | None" = None
        self._region_draft_active: bool = False

        # Plan 03-02 — Region Select mode toggle (CONTEXT D-A1-1). When
        # ``True``, a plain Left+drag (no Shift required) creates a region.
        # The toolbar's Region Select QAction (mode-OFF on app start) is
        # the user-facing toggle; MainWindow wires the QAction.toggled
        # signal to :meth:`set_region_select_mode`. Shift+drag continues
        # to work regardless of mode (combo gesture).
        self._region_select_mode_active: bool = False

        # Plan 03-02 — middle-mouse pan. PyQtGraph 0.13.7's ViewBox does
        # NOT pan on middle-button by default; we synthesize the pan in
        # the eventFilter so the user-discoverable contract "middle-drag
        # pans" works regardless of which gesture mode (region-select on
        # or off) the user is in. The press / move / release branches in
        # eventFilter consult these three fields.
        self._middle_pan_active: bool = False
        self._middle_pan_last_x: int = 0
        self._middle_pan_last_y: int = 0

    # ----------------------------------------------------- backwards-compat
    @property
    def plot_widget(self) -> _PlotWidgetShim:
        """Phase 1 compatibility shim — see :class:`_PlotWidgetShim`."""
        return self._plot_widget_shim

    # ----------------------------------------------------- playhead accessor
    @property
    def playhead(self) -> pg.InfiniteLine:
        """The waveform-plot playhead InfiniteLine (Plan 02-05).

        The line lives on ``self.waveform_plot``. Per-lane playheads (one
        per active heatmap lane) live on their respective PlotItems and
        are tracked by MainWindow in ``self._lane_playheads`` because a
        single InfiniteLine instance cannot be shared across QGraphicsScenes
        (W1).
        """
        return self._playhead

    # ---------------------------------------------------------- Plan-01 shims
    def show_plot(self) -> None:
        """Swap from empty-state to plot view. Plan 01 compatibility shim."""
        self._stack.setCurrentIndex(1)

    def show_empty_state(self) -> None:
        """Reset to empty-state. Plan 01 compatibility shim."""
        self._stack.setCurrentIndex(0)

    # --------------------------------------------------------- render contract
    def render_proxy(
        self,
        proxy_arr: np.ndarray,
        sample_rate: int,
        samples_per_pixel: int,
    ) -> None:
        """Render a BBC-v2 proxy ``(N, 2)`` int16 array as a saw-wave waveform.

        Pre-conditions:
            * ``proxy_arr.ndim == 2`` and ``proxy_arr.shape[1] == 2`` —
              interleaved min/max pairs as written by
              :func:`proxy_cache.write_proxy`.
            * ``proxy_arr.dtype == np.int16`` — handed straight to
              ``setData`` without any cast or copy (CLAUDE.md memory
              contract + RESEARCH §Architectural Responsibility Map).

        The x array is twice the pair count because each pair emits two
        points (RESEARCH §Pattern 1 saw-wave trick). x units are seconds
        starting at 0; y units are raw int16 amplitudes (the y-axis is
        locked via ``setMouseEnabled(y=False)``, so the visual range
        ``[-32768, 32767]`` matches the data domain without any rescale).
        """
        # USER FEEDBACK 2026-05-13 / plan 01-09 Option A — viewport-density
        # pre-aggregation: when the proxy contains more than MAX_RENDER_PROXY_PAIRS
        # (4000) min/max pairs, aggregate it down to exactly 4000 bins BEFORE
        # handing the array to PyQtGraph. The user complaint was "zooming is
        # eating all my cpu" because ``setDownsampling(auto=True, method='peak')``
        # still pages through the full proxy density on every paint (and
        # ``setXRange`` — what the mouse-wheel zoom dispatches — triggers a
        # repaint). At spp=256 / sr=44100 an 8h file is ≈ 5.4M pairs (≈ 10.8M
        # plot points after saw-wave doubling); after aggregation PyQtGraph sees
        # at most 8000 plot points regardless of file duration. CPU per paint
        # becomes O(viewport_width) instead of O(file_duration). The four
        # RESEARCH §Pattern 1 PlotDataItem flags
        # (setDownsampling/setClipToView/setSkipFiniteCheck/pen width=1) stay
        # configured in __init__ — they just have orders-of-magnitude less data
        # to page through on every paint. Upgrade path if 4000 bins ever proves
        # too coarse at extreme zoom-in: Option C — sigXRangeChanged-driven
        # re-aggregate over the visible slice. N-1 invariant preserved (the
        # per-item rendering-cache toggle from the earlier plan revision stays
        # absent); plan 01-05 float32 x-array fix preserved (no float64
        # regression — zero-copy y when pass-through, small fixed-size allocated
        # aggregate when not).
        original_length = int(proxy_arr.shape[0])
        if original_length > MAX_RENDER_PROXY_PAIRS:
            # Aggregate to exactly MAX_RENDER_PROXY_PAIRS bins that UNIFORMLY
            # span the WHOLE proxy. The old reshape used integer-floor
            # ``bin_size = original_length // MAX`` and folded the remainder
            # into the last bin — so the first MAX bins covered only
            # ``MAX * bin_size`` pairs (e.g. 8000 of 11250 on a 60 s file)
            # while the x-axis below spaced them as if each spanned
            # ``original_length / MAX`` pairs. That mismatch STRETCHED the
            # waveform horizontally by ``(original/MAX) / bin_size`` (≈ 1.40×
            # on a 60 s clip): a transient at true t=1.0 s drew at x≈1.40 s,
            # so audio led the on-screen feature by a TIME-PROPORTIONAL amount
            # no constant playhead offset could cancel. (Negligible on long
            # files where floor≈ratio — hence it hid until short clips.)
            #
            # ``np.linspace`` edges give MAX bins evenly covering all
            # ``original_length`` pairs (widths differ by at most 1 pair), so
            # the uniform ``effective_spp = spp * original_length / length``
            # spacing below is now CORRECT. ``reduceat`` reduces each
            # ``[edges[i], edges[i+1])`` segment with no Python per-bin loop
            # (still O(original_length), one pass) — edges are strictly
            # increasing because ``original_length > MAX`` ⇒ step > 1.
            edges = np.linspace(
                0, original_length, MAX_RENDER_PROXY_PAIRS + 1
            ).astype(np.intp)
            mins = np.minimum.reduceat(proxy_arr[:, 0], edges[:-1])
            maxs = np.maximum.reduceat(proxy_arr[:, 1], edges[:-1])
            # ``np.empty`` is intentional vs ``np.zeros`` — both columns are
            # written below before ``setData``, so no uninitialised memory
            # leaks into the plot (T-09-failure-mode).
            aggregate = np.empty((MAX_RENDER_PROXY_PAIRS, 2), dtype=np.int16)
            aggregate[:, 0] = mins
            aggregate[:, 1] = maxs
            render_arr = aggregate
        else:
            # Pass-through branch: short files (≤ MAX_RENDER_PROXY_PAIRS pairs,
            # i.e. ≤ ~23 s at sr=44100/spp=256) keep their original int16
            # memmap. Behaviour is byte-identical to plan 01-05.
            render_arr = proxy_arr

        # Saw-wave x axis: each (min, max) pair contributes two points
        # spaced by ``effective_spp / (2 * sample_rate)`` seconds, where
        # ``effective_spp = samples_per_pixel * (original_length / length)``.
        # Pass-through: effective_spp == samples_per_pixel. Aggregated 8h file:
        # effective_spp ≈ 345_600 (each bin spans ~7.84 s) — the deliberate
        # "vague representation" the user asked for.
        length = int(render_arr.shape[0])
        n_points = 2 * length
        effective_spp = samples_per_pixel * (original_length / length)
        # Memory-bounded x axis (CR-01 fix, plan 01-05): float32 is sufficient
        # — for the pass-through case (spp=256) the per-sample spacing is
        # 256/(2*44100) ≈ 2.9 ms; the float32 ULP at the 8h domain end
        # (28_800 s) is ≈ 0.004 s, far below one paint period (33 ms at 30 fps).
        # Post-aggregation (plan 01-09) the effective spp grows by
        # ``original_length / length`` so per-bin spacing at the 8h target is
        # ≈ 7.2 s — orders of magnitude above the float32 ULP, so the ULP
        # analysis trivially still applies. Allocating int32 first and then
        # converting to float32 would double-allocate; we compute float32
        # directly from np.arange to keep peak memory at one (2 * length) × 4
        # byte array. The earlier float64 path peaked at ≈ 144 MiB on an
        # 8-hour proxy and violated the CLAUDE.md memory contract on the GUI
        # thread. Plan 01-09's aggregation collapses this further: post-fix
        # n_points = 8000 → x-array is ≈ 32 KiB, a free bonus on top of the
        # plan-01-05 fix.
        x = np.arange(n_points, dtype=np.float32) * np.float32(
            effective_spp / (2.0 * sample_rate)
        )

        # quick-260627-gb7 — cache the proxy args BEFORE the mode transform so a
        # later view-mode change can re-run render_proxy from the cached array
        # WITHOUT reloading audio (_on_render_mode_changed). We stash the
        # ORIGINAL proxy_arr (not the aggregate) + sr + spp so the re-render
        # path is byte-for-byte the same as the original render.
        self._last_proxy_args = (proxy_arr, sample_rate, samples_per_pixel)

        # quick-260627-gb7 — render-mode dispatch. The transform turns the
        # aggregated ``render_arr`` int16 (min,max) array into the flat saw-wave
        # ``y_flat`` + the ``(y_min, y_max)`` range. CLASSIC is the identity
        # passthrough — its ``y_flat`` is value-identical to the previous
        # ``render_arr.reshape(-1)`` int16 saw-wave (CLAUDE.md memory contract:
        # CLASSIC stays the zero-extra-work path; dB/Energy materialise a small
        # ≤4000-bin float32 derived curve, O(viewport), not O(file)).
        #
        # Phase 11 / plan 11-06 — SPECTRAL modes are NOT in render_modes._REGISTRY
        # (they consume spectral arrays, not min/max pairs — transform_for raises
        # KeyError BY DESIGN). For the line silhouette they reuse the CLASSIC
        # transform: SPECTROGRAM hides the line entirely (the ImageItem is the
        # surface) and CENTROID / RGB_BAND keep the Classic min/max silhouette and
        # color it per-column via a backdrop. So we always build the line from a
        # registry transform, substituting CLASSIC for the spectral modes.
        line_mode = (
            RenderMode.CLASSIC
            if self._render_mode in _SPECTRAL_MODES
            else self._render_mode
        )
        y_flat, y_range = transform_for(line_mode)(render_arr)

        # quick-260621-gfq — stash the rendered envelope so the keeper-row
        # Normalize toggle can transform a [start_s, end_s) span IN PLACE
        # without re-reading the proxy or moving the viewport. ``_rendered_y``
        # is a MUTABLE copy; ``_rendered_y_orig`` is the revert baseline. In
        # CLASSIC mode this is the int16 saw-wave layout the normalize WYSIWYG
        # math expects (so set_region_normalize is byte-identical to before);
        # in dB/Energy modes it holds the derived float curve and
        # set_region_normalize EARLY-RETURNS (it only operates on the int16
        # saw-wave layout — guarded by self._render_mode). The per-point time
        # spacing is ``effective_spp / (2 * sample_rate)`` seconds.
        self._rendered_x = x
        self._rendered_y = np.array(y_flat, copy=True)
        self._rendered_y_orig = self._rendered_y.copy()
        self._rendered_spp_eff = float(effective_spp)
        self._rendered_sr = int(sample_rate)

        self._plot_data_item.setData(x=x, y=self._rendered_y)
        # Duration uses the ORIGINAL proxy length (not the aggregated length)
        # so the visible x domain spans the full file regardless of aggregation.
        # Aggregation changes y density, not x extent.
        self._duration_s = (original_length * samples_per_pixel) / float(sample_rate)
        # Pin the visible range to the data domain so the user sees the
        # whole file by default; y-axis is locked anyway. Calls land on
        # waveform_plot (the PlotItem) directly — setXLink propagates the
        # x-range to the reserved energy lane automatically. The y-range comes
        # from the active mode's transform (CLASSIC: (-32768, 32767); Energy:
        # single-sided (0, 32767)).
        self.waveform_plot.setXRange(0.0, self._duration_s, padding=0)
        self.waveform_plot.setYRange(*y_range, padding=0)

        # Phase 11 / plan 11-06 — toggle the spectral surfaces (ImageItem /
        # color backdrop) for the active mode. In a 1-D mode this hides both and
        # leaves the line as drawn above; in SPECTROGRAM it shows the image and
        # hides the line; in CENTROID / RGB_BAND it keeps the line visible and
        # lays a per-column color backdrop beneath it (when spectral data is
        # stashed). Overlays / playhead / lanes are independent items and are
        # untouched (R-7).
        self._apply_spectral_surface()

        # Swap from empty-state to plot view.
        self._stack.setCurrentIndex(1)

    # ------------------------------------------------- Phase 11 spectral surface
    def set_spectral_data(
        self,
        mel: np.ndarray | None,
        centroid: np.ndarray | None,
        bands: np.ndarray | None,
        header: object | None,
    ) -> None:
        """Stash the precomputed spectral arrays for the spectral render modes.

        Phase 11 / plan 11-06 (R-3 / R-4 / R-5 / R-6). MainWindow (plan 11-07)
        calls this when the background spectral build completes (or on a cache
        hit) with:

            * ``mel``      — ``(n_mels, n_frames)`` uint8 mel image (row 0 = low
              freq), already quantised over the 80 dB window so the ImageItem
              ``setLevels([0, 255])`` is correct.
            * ``centroid`` — ``(n_frames,)`` per-column spectral centroid (Hz or
              normalised) for the CENTROID tint. May be ``None``.
            * ``bands``    — ``(3, n_frames)`` low/mid/high band energies for the
              RGB_BAND mode. May be ``None``.
            * ``header``   — the spectral-cache header (``db_floor`` / ``db_ref``
              documentation); the view does not strictly require it.

        Stashing is analogous to ``_last_proxy_args``: once stashed, re-selecting
        a spectral mode renders from this stash with NO rebuild (cache hit). If
        the current mode is already a spectral mode, the surface re-renders
        immediately so a completed build paints without a further user action.
        """
        self._spectral_mel = None if mel is None else np.asarray(mel)
        self._spectral_centroid = None if centroid is None else np.asarray(centroid)
        self._spectral_bands = None if bands is None else np.asarray(bands)
        self._spectral_header = header
        # quick-260629-vui — refresh the dB colorbar labels from the freshly
        # stashed header (db_floor / db_ref) so a completed build relabels the
        # bar even before the next mode dispatch.
        self._update_mag_colorbar_labels()
        # If a spectral mode is already active, paint the freshly-arrived data.
        if self._render_mode in _SPECTRAL_MODES:
            self._apply_spectral_surface()

    def _has_spectral_data(self) -> bool:
        """True when at least the mel image has been stashed (R-3 cache state)."""
        return self._spectral_mel is not None

    def _apply_spectral_surface(self) -> None:
        """Toggle / render the spectral surfaces for the current render mode.

        Phase 11 / plan 11-06. Pure view-side dispatch (no audio reload):

            * non-spectral mode → hide the ImageItem + color backdrop, show line.
            * SPECTROGRAM       → hide the line, render + show the mel ImageItem.
            * CENTROID/RGB_BAND → hide the line, render + show the color backdrop
              (quick-260629 — the opaque color backdrop IS the surface; leaving
              the Classic silhouette on top hid the color and read as plain
              Classic. Hiding the line lets the per-column color show, with the
              playhead + regions still on top because the backdrop sits at the
              back z-band).

        When a spectral mode is selected but no spectral data is stashed the
        surfaces stay hidden and the line stays visible (the build-request is
        emitted by :meth:`_on_render_mode_changed`, not here).
        """
        mode = self._render_mode
        if mode not in _SPECTRAL_MODES:
            self._spectro_img.setVisible(False)
            self._color_backdrop_img.setVisible(False)
            self._plot_data_item.setVisible(True)
            # quick-260629-vui — hide every spectral legend and relabel the freq
            # axis (now blank) so non-spectral modes are byte-identical.
            self._hide_spectral_legends()
            self._refresh_freq_axis()
            return

        if not self._has_spectral_data():
            # Cold cache — leave the current (Classic-silhouette) line in place
            # until set_spectral_data arrives; do not blank the canvas.
            self._spectro_img.setVisible(False)
            self._color_backdrop_img.setVisible(False)
            self._plot_data_item.setVisible(True)
            # quick-260629-vui — no surface yet, so no legend either.
            self._hide_spectral_legends()
            self._refresh_freq_axis()
            return

        if mode is RenderMode.SPECTROGRAM:
            self._render_spectrogram()
            self._color_backdrop_img.setVisible(False)
            self._plot_data_item.setVisible(False)
            self._spectro_img.setVisible(True)
            # quick-260629-vui — Hz freq axis (via _FreqAxisItem, relabelled in
            # _render_spectrogram) + dB colorbar from the live header; hide the
            # other two legends.
            self._update_mag_colorbar_labels()
            self._mag_colorbar.setVisible(True)
            self._centroid_colorbar.setVisible(False)
            self._rgb_legend.setVisible(False)
        else:  # CENTROID or RGB_BAND
            self._render_color_backdrop(mode)
            self._spectro_img.setVisible(False)
            # quick-260629 — hide the Classic silhouette so the color backdrop
            # is the visible surface (was True, which left the plain waveform
            # on top and hid the color). The playhead/regions render above the
            # backdrop because it lives in the back z-band (setZValue(-99)).
            self._plot_data_item.setVisible(False)
            self._color_backdrop_img.setVisible(True)
            # quick-260629-vui — per-mode legend: centroid colorbar in CENTROID,
            # RGB band legend in RGB_BAND. The mag colorbar + Hz axis are
            # SPECTROGRAM-only, so hide them here.
            self._mag_colorbar.setVisible(False)
            self._centroid_colorbar.setVisible(mode is RenderMode.CENTROID)
            self._rgb_legend.setVisible(mode is RenderMode.RGB_BAND)
            if mode is RenderMode.RGB_BAND:
                self._reposition_rgb_legend()
            # Freq axis is blank outside SPECTROGRAM — relabel to clear it.
            self._refresh_freq_axis()

    def _hide_spectral_legends(self) -> None:
        """Hide all four spectral legends (quick-260629-vui).

        Called from the non-spectral and cold-cache early-returns so no legend
        lingers when switching out of a spectral mode. The Hz freq axis is
        handled separately via ``_refresh_freq_axis`` (it blanks itself through
        ``_FreqAxisItem.tickStrings`` whenever the mode is not SPECTROGRAM).
        """
        self._mag_colorbar.setVisible(False)
        self._centroid_colorbar.setVisible(False)
        self._rgb_legend.setVisible(False)

    def _render_spectrogram(self) -> None:
        """MAX-pool the stashed mel to viewport density and push to the ImageItem.

        Phase 11 / plan 11-06 (R-4). The stored mel ``(n_mels, n_frames)`` is
        MAX-pooled to ``<= MAX_RENDER_SPECTRAL_COLS`` columns
        (:func:`aggregate_spectral_columns`, ``.max`` per Pitfall #5 — a single
        hot column survives), then handed to ``setImage(..., autoLevels=False)``
        (levels pinned to ``[0, 255]`` at construction) + ``setRect(0, 0,
        duration_s, n_mels)`` so row 0 (low freq) sits at the BOTTOM
        (imageAxisOrder row-major — Pitfall #4). The pooled image is stashed on
        ``_rendered_spectral_image`` for tests / the normalize no-op guard.
        """
        mel = self._spectral_mel
        if mel is None or mel.ndim != 2 or mel.shape[1] == 0:
            return
        pooled = aggregate_spectral_columns(mel, MAX_RENDER_SPECTRAL_COLS)
        self._rendered_spectral_image = pooled
        self._spectro_img.setImage(pooled, autoLevels=False)
        # x-extent in seconds = the full file duration (aggregation changes
        # column width, not extent). The y-extent is STRETCHED over the current
        # view y-range (the waveform amplitude domain, e.g. (-32768, 32767))
        # rather than the raw n_mels bin count — otherwise a ~128-tall image
        # renders as a thin sliver near y=0 inside a ±32768 viewport. Mirrors
        # _render_color_backdrop. setRect maps image row 0 → y_min (bottom),
        # so low freq stays at the bottom (row-major imageAxisOrder, Pitfall #4)
        # as long as the height is positive.
        y_min, y_max = self.waveform_plot.viewRange()[1]
        self._spectro_img.setRect(
            0.0, float(y_min), float(self._duration_s), float(y_max - y_min)
        )
        # quick-260629-vui — relabel the Hz freq axis against the (now current)
        # y-range so the labels track a fresh spectrogram render.
        self._refresh_freq_axis()

    def _render_color_backdrop(self, mode: RenderMode) -> None:
        """Render the per-column CENTROID / RGB_BAND color backdrop (R-5 / R-6).

        Phase 11 / plan 11-06 (RESEARCH A3 ImageItem-backdrop path). Builds a
        ``(N, 3)`` uint8 RGB color array from the stashed spectral arrays via the
        Qt-free math in :mod:`render_modes` — ``centroid_tint_colors`` (with the
        view's injected Magma LUT) for CENTROID, ``rgb_band_colors`` for
        RGB_BAND — after MAX-pooling the source arrays to the viewport column
        count. The colors are pushed as a ``(1, N, 3)`` RGB ImageItem stretched
        over the full data extent UNDER the neutral Classic silhouette.
        """
        from marmelade.audio.render_modes import (
            centroid_tint_colors,
            rgb_band_colors,
        )

        if mode is RenderMode.CENTROID:
            centroid = self._spectral_centroid
            if centroid is None or np.asarray(centroid).size == 0:
                self._color_backdrop_img.setVisible(False)
                return
            pooled = aggregate_spectral_columns(
                np.asarray(centroid), MAX_RENDER_SPECTRAL_COLS
            )
            colors = centroid_tint_colors(pooled, lut=self._magma_lut)
        else:  # RGB_BAND
            bands = self._spectral_bands
            if bands is None or np.asarray(bands).ndim != 2 or bands.shape[0] < 3:
                self._color_backdrop_img.setVisible(False)
                return
            pooled = aggregate_spectral_columns(
                np.asarray(bands), MAX_RENDER_SPECTRAL_COLS
            )
            colors = rgb_band_colors(pooled[0], pooled[1], pooled[2])

        if colors.shape[0] == 0:
            self._color_backdrop_img.setVisible(False)
            return
        # (1, N, 3) RGB image stretched over the full data extent. The y-extent
        # spans the current y-range so the backdrop tints the whole silhouette
        # band. RGB images bypass the LUT/levels (3 channels), so no setLevels.
        img = colors.reshape(1, -1, 3)
        self._color_backdrop_img.setImage(img, autoLevels=False)
        y_min, y_max = self.waveform_plot.viewRange()[1]
        self._color_backdrop_img.setRect(
            0.0, float(y_min), float(self._duration_s), float(y_max - y_min)
        )

    # ---------------------------------------------- quick-260629-vui legends
    def _build_magma_gradient(self) -> QLinearGradient:
        """Build a vertical Magma QLinearGradient ONCE from ``self._magma_lut``.

        quick-260629-vui (T-vui-02). Samples ~16 evenly-spaced stops from the
        256-entry LUT so the colorbar shows the same Magma ramp as the
        spectrogram ImageItem with no per-paint cost. GradientLegend paints its
        gradient top-to-bottom, so stop 0.0 is the TOP — we put the loudest
        color (LUT[255]) there so "loud" reads at the top of the bar.
        """
        lut = self._magma_lut
        n = lut.shape[0]
        grad = QLinearGradient()
        n_stops = 16
        for i in range(n_stops):
            t = i / (n_stops - 1)  # 0.0 (top) .. 1.0 (bottom)
            # Invert so the top (t=0) samples the brightest LUT entry.
            lut_idx = int(round((1.0 - t) * (n - 1)))
            r, g, b = (int(c) for c in lut[lut_idx][:3])
            grad.setColorAt(t, QColor(r, g, b))
        return grad

    def _update_mag_colorbar_labels(self) -> None:
        """Refresh the dB colorbar labels from the live spectral header.

        quick-260629-vui (T-vui-01). Reads ``db_floor`` / ``db_ref`` from
        ``self._spectral_header`` with the ``spectral_cache`` constants as the
        fallback (header may be ``None``). The values are only formatted into a
        label string — never used to index an array — so a hostile header cannot
        drive an out-of-bounds access. GradientLegend label dict is
        ``{text: value}`` where ``value`` is 0..1 measured from the BOTTOM, so
        db_ref sits at 1.0 (top) and db_floor at 0.0 (bottom).
        """
        db_floor = float(getattr(self._spectral_header, "db_floor", DB_FLOOR))
        db_ref = float(getattr(self._spectral_header, "db_ref", DB_REF))
        mid = (db_ref + db_floor) / 2.0
        self._mag_colorbar.setLabels(
            {
                f"{int(db_ref)}": 1.0,
                f"{int(mid)}": 0.5,
                f"{int(db_floor)}": 0.0,
            }
        )

    @staticmethod
    def _centroid_colorbar_labels() -> dict[str, float]:
        """Log-positioned 100 / 1k / 10k labels for the centroid colorbar.

        quick-260629-vui. ``pos = (log(hz) - log(fmin)) / (log(fmax) - log(fmin))``
        with the CENTROID_FMIN_HZ / CENTROID_FMAX_HZ bounds (LOG scale, matching
        the centroid-tint LUT mapping). Constant bounds → computed once.
        """
        lo = np.log(CENTROID_FMIN_HZ)
        hi = np.log(CENTROID_FMAX_HZ)
        out: dict[str, float] = {}
        for hz, text in ((100.0, "100"), (1000.0, "1k"), (10000.0, "10k")):
            out[text] = float((np.log(hz) - lo) / (hi - lo))
        return out

    def _build_rgb_legend_html(self) -> str:
        """Static RGB band legend HTML from the imported split Hz.

        quick-260629-vui. low (< _BAND_LO_HZ) in red, mid (_BAND_LO_HZ..
        _BAND_HI_HZ) in green, high (>= _BAND_HI_HZ) in blue — the same
        red/green/blue channel mapping ``rgb_band_colors`` uses. Split numbers
        come from the imported constants, never hardcoded literals.
        """
        lo = int(_BAND_LO_HZ)
        hi = int(_BAND_HI_HZ)
        # Colored ■ swatches (pure R/G/B match the rgb_band_colors channel
        # mapping low→R / mid→G / high→B exactly); label text is black for
        # legibility on the white TextItem fill (set in __init__).
        return (
            "<div style='padding:1px; white-space:nowrap;'>"
            f"<span style='color:#FF0000;'>&#9632;</span>"
            f"<span style='color:#000000;'> low &lt;{lo} Hz</span><br>"
            f"<span style='color:#00B000;'>&#9632;</span>"
            f"<span style='color:#000000;'> mid {lo}-{hi} Hz</span><br>"
            f"<span style='color:#0000FF;'>&#9632;</span>"
            f"<span style='color:#000000;'> high &ge;{hi} Hz</span>"
            "</div>"
        )

    def _reposition_rgb_legend(self) -> None:
        """Pin the RGB legend to the BOTTOM-RIGHT inside corner of the ViewBox.

        quick-260629-vui. The TextItem is a DIRECT child of the ViewBox
        (``setParentItem``), so its ``setPos`` is in the ViewBox's LOCAL PIXEL
        coordinate system — the SAME space ``pg.GradientLegend`` anchors the
        colorbars in — NOT data coordinates. The earlier version positioned it
        in data coords (seconds × int16 amplitude), which placed it ~32 000 px
        off-screen: ``isVisible()`` was True but nothing was ever painted in
        view. ``boundingRect`` gives the ViewBox's local pixel rect (origin
        top-left, y increasing downward); anchor (1, 1) puts the legend's
        bottom-right corner at the inset point.
        """
        vb = self.waveform_plot.getViewBox()
        if vb is None:
            return
        rect = vb.boundingRect()  # ViewBox-local PIXELS, not data coords
        inset = 8.0
        x = rect.right() - inset
        y = rect.bottom() - inset
        self._rgb_legend.setPos(x, y)

    def _refresh_freq_axis(self) -> None:
        """Force the left freq-axis to re-tick after a y-range change.

        quick-260629-vui. The Hz→y mapping depends on the live y-viewrange, so
        when the viewbox y-range changes (manual zoom or a fresh spectrogram
        render) the cached tick picture must be invalidated so the labels
        relabel at their new positions.
        """
        left = self.waveform_plot.getAxis("left")
        left.picture = None
        left.update()

    def _on_render_mode_changed(self, index: int) -> None:
        """Combo handler — switch the active render mode + re-render the cache.

        quick-260627-gb7. Maps the combo ``index`` to a :class:`RenderMode`
        (combo items are populated in ``RenderMode`` order), updates
        ``self._render_mode`` and, if a proxy is cached, re-runs
        :meth:`render_proxy` with the CACHED args — re-rendering in place with
        the new transform and WITHOUT touching / reloading the loaded audio.

        Phase 11 / plan 11-06 (R-3 lazy build). When the new mode is a SPECTRAL
        mode (SPECTROGRAM / CENTROID / RGB_BAND) and no spectral data is stashed
        yet (cold cache), emit ``spectral_build_requested(mode)`` exactly once and
        leave the current render in place — MainWindow (plan 11-07) spawns the
        background worker and calls :meth:`set_spectral_data` on completion. When
        spectral data IS stashed, the surface renders from it immediately and no
        signal fires (cache hit). The re-render below repaints the line silhouette
        for the spectral modes and toggles the spectral surface visibility.
        """
        modes = list(RenderMode)
        if not (0 <= index < len(modes)):
            return
        self._render_mode = modes[index]
        if self._last_proxy_args is not None:
            self.render_proxy(*self._last_proxy_args)
        else:
            # No proxy cached (e.g. a test that stashes spectral data directly
            # without loading audio): render_proxy won't run, so apply the
            # spectral surface dispatch here so a stashed mel still paints.
            self._apply_spectral_surface()
        # Lazy build request: cold-cache selection of a spectral mode asks
        # MainWindow to build the spectral arrays. Emitted AFTER the re-render
        # above so the line silhouette is already in place when the build starts.
        if (
            self._render_mode in _SPECTRAL_MODES
            and not self._has_spectral_data()
        ):
            self.spectral_build_requested.emit(self._render_mode)

    def clear(self) -> None:
        """Swap back to empty-state and clear the PlotDataItem data."""
        self._plot_data_item.setData([], [])
        self._duration_s = 0.0
        # quick-260621-gfq — drop the stashed envelope so a stale span
        # transform can't fire against a cleared plot.
        self._rendered_x = None
        self._rendered_y = None
        self._rendered_y_orig = None
        self._rendered_spp_eff = 0.0
        self._rendered_sr = 0
        # quick-260627-gb7 — drop the cached proxy args so a stale view-mode
        # switch can't re-render against a cleared plot.
        self._last_proxy_args = None
        # Phase 11 / plan 11-06 — drop the stashed spectral arrays + hide the
        # spectral surfaces so a stale spectral render can't fire on the next
        # (different) file. The next file's build re-stashes via set_spectral_data.
        self._spectral_mel = None
        self._spectral_centroid = None
        self._spectral_bands = None
        self._spectral_header = None
        self._rendered_spectral_image = None
        self._spectro_img.setVisible(False)
        self._color_backdrop_img.setVisible(False)
        self._mag_colorbar.setVisible(False)
        self._centroid_colorbar.setVisible(False)
        self._rgb_legend.setVisible(False)
        self._plot_data_item.setVisible(True)
        self._stack.setCurrentIndex(0)

    def reset_render_mode_to_classic(self) -> None:
        """Reset to CLASSIC + drop stale spectral data — called on new-file open.

        quick-260629. Loading a DIFFERENT sound must not carry the previous
        file's render mode or spectral surfaces (spectrogram / centroid /
        band / freq axis / colorbars / legend) into the new file. ``clear()``
        only runs on error/close paths, not on a successful open, so without
        this a file opened while SPECTROGRAM was selected would re-render the
        PREVIOUS file's stashed mel.

        The combo signal is BLOCKED while resetting the index so no spurious
        in-place re-render or lazy spectral-build request fires; ``_render_mode``
        is set directly and the stale spectral arrays are dropped + every
        spectral overlay hidden. The new file's own ``render_proxy`` (and, if
        the user re-selects a spectral mode, the lazy build) repopulate.
        """
        classic_idx = list(RenderMode).index(RenderMode.CLASSIC)
        blocked = self.render_mode_combo.blockSignals(True)
        try:
            self.render_mode_combo.setCurrentIndex(classic_idx)
        finally:
            self.render_mode_combo.blockSignals(blocked)
        self._render_mode = RenderMode.CLASSIC
        # Drop stale spectral arrays so the new file cannot show the old surface.
        self._spectral_mel = None
        self._spectral_centroid = None
        self._spectral_bands = None
        self._spectral_header = None
        self._rendered_spectral_image = None
        # Hide every spectral overlay; show the Classic line.
        self._spectro_img.setVisible(False)
        self._color_backdrop_img.setVisible(False)
        self._mag_colorbar.setVisible(False)
        self._centroid_colorbar.setVisible(False)
        self._rgb_legend.setVisible(False)
        self._plot_data_item.setVisible(True)

    def set_region_normalize(
        self,
        start_s: float,
        end_s: float,
        enabled: bool,
        target_db: float,
    ) -> None:
        """In-place WYSIWYG per-keeper normalize re-render (quick-260621-gfq).

        Transforms ONLY the rendered-envelope columns covering ``[start_s,
        end_s)``: DC-removes the span and peak-scales it to ``target_db``
        (the mastering chain's final-stage preview), or restores the original
        span values when ``enabled`` is False. The viewport X-range is NOT
        changed (locked decision #1 option 2 — no auto-zoom). O(viewport)
        regardless of file length (operates on the already-aggregated
        ≤4000-bin envelope — T-gfq-02).

        No-op when nothing is rendered yet.

        Classic-only contract (quick-260627-gb7): this WYSIWYG span re-render
        operates on the int16 saw-wave layout that the CLASSIC render mode
        produces. In the dB / Energy view modes the rendered ``y`` is a derived
        float curve (log-amplitude / rectified envelope), so this method
        EARLY-RETURNS (no-op) rather than corrupting that derived curve — the
        normalize preview is a Classic-mode-only affordance (T-gb7-03).
        """
        # quick-260627-gb7 — guard: only the CLASSIC int16 saw-wave layout is a
        # valid normalize target. Switch back to Classic to preview normalize.
        if self._render_mode is not RenderMode.CLASSIC:
            return
        if self._rendered_y is None or self._rendered_y_orig is None:
            return
        if self._rendered_sr <= 0 or self._rendered_spp_eff <= 0.0:
            return
        # Each rendered point spans ``effective_spp / (2 * sr)`` seconds (the
        # saw-wave doubling halves the per-pair spacing).
        per_point = self._rendered_spp_eff / (2.0 * self._rendered_sr)
        if per_point <= 0.0:
            return
        n = int(self._rendered_y.shape[0])
        col_lo = max(0, int(start_s / per_point))
        col_hi = min(n, int(end_s / per_point))
        if col_hi <= col_lo:
            return

        if enabled:
            # Reuse the chain's scale math (_compute_scale) so the display
            # preview agrees with the mastered output. The rendered envelope
            # is int16; normalize it in [-1, 1] float space (divide by the
            # int16 full-scale), DC-remove, peak-to-target via _compute_scale
            # (its eps-clamp keeps a silent span at scale 1.0 — no noise-floor
            # amplification), then scale back to int16. A 0 dB target maps the
            # loudest sample to ±INT16_MAX.
            from marmelade.audio.normalize import _compute_scale

            int16_max = 32767.0
            span = self._rendered_y[col_lo:col_hi].astype(np.float64) / int16_max
            mean = float(span.mean())
            centered = span - mean
            peak = float(np.abs(centered).max()) if centered.size else 0.0
            scale = _compute_scale(peak, target_db)
            transformed = centered * scale * int16_max
            # ROUND before the int16 cast (np.round) to avoid the systematic
            # truncation/DC bias a bare ``.astype(int16)`` would introduce.
            # Clip to the int16 domain so a target at/above the current peak
            # never wraps on cast.
            transformed = np.clip(transformed, -int16_max, int16_max)
            self._rendered_y[col_lo:col_hi] = np.round(transformed).astype(
                self._rendered_y.dtype, copy=False
            )
        else:
            # Revert this span to the stashed baseline.
            self._rendered_y[col_lo:col_hi] = self._rendered_y_orig[
                col_lo:col_hi
            ]

        # Re-set data WITHOUT any setXRange/setYRange — viewport unchanged.
        self._plot_data_item.setData(x=self._rendered_x, y=self._rendered_y)

    # ----------------------------------------------------- heatmap lane registry
    # Plan 02-03 — public lane-registry methods on the WaveformView. The
    # lane registry on MainWindow (`MainWindow._heatmap_lanes`) is the
    # canonical owner of the lane instances; WaveformView owns the layout
    # slot and the reserved height-0 stub that lives there when no real
    # lane is attached. Plan 02-04's sidebar toggle uses
    # ``add_heatmap_lane`` / ``remove_heatmap_lane`` to swap lanes in and
    # out without further refactoring.
    def add_heatmap_lane(self, name: str, lane: HeatmapLaneView) -> None:
        """Install ``lane.plot_item`` at row 1 of the graphics layout.

        Removes the height-0 reserved stub for ``name`` first so the row-1
        slot ends up holding a single PlotItem. The lane's own
        ``setMaximumHeight(28)`` (set in :class:`HeatmapLaneView.__init__`)
        raises the row from invisible to 28 px tall.
        """
        stub = self._reserved_lane_stubs.get(name)
        if stub is not None:
            current = self.graphics_layout.getItem(1, 0)
            if current is stub:
                self.graphics_layout.removeItem(stub)
        # Horizontal alignment discipline (see this module's
        # §"Horizontal alignment between viewboxes"): the waveform plot
        # reserves a 40 px invisible LEFT-axis slot so its ViewBox starts at a
        # fixed left screen-x. ``HeatmapLaneView`` hides its left axis (width
        # 0), so its ViewBox would otherwise be ~40 px WIDER than the
        # waveform's — and ``setXLink`` reconciles the linked range by the
        # ratio of ViewBox widths, drifting the lane's left edge when the
        # plot is narrow. Give the lane the SAME 40 px invisible slot so both
        # ViewBoxes share width regardless of the overall plot width (the
        # drift is latent at wide geometries and surfaces when the waveform
        # plot is narrow).
        _LEFT_AXIS_WIDTH = 40
        lane_left = lane.plot_item.getAxis("left")
        lane_left.show()
        lane_left.setWidth(_LEFT_AXIS_WIDTH)
        lane_left.setStyle(showValues=False, tickLength=0)
        lane_left.setPen(pg.mkPen(0, 0, 0, 0))
        self.graphics_layout.addItem(lane.plot_item, row=1, col=0)

    def remove_heatmap_lane(self, name: str) -> None:
        """Restore the reserved height-0 stub for ``name`` at row 1.

        MainWindow owns the lane instance; it MUST call
        ``lane.remove(self.graphics_layout)`` itself BEFORE invoking this
        helper so the lane's PlotItem is removed and ``deleteLater``-ed
        first. This helper just re-installs the height-0 reserved stub so
        row 1 is not left empty (avoids a transient "empty row 1" layout
        flicker between lane removal and stub restoration). Idempotent —
        calling on an already-restored row is a no-op.
        """
        self._restore_reserved_lane_stub(name)

    def _restore_reserved_lane_stub(self, name: str) -> None:
        """Re-add the reserved height-0 stub for ``name`` at row 1.

        Idempotent: if the row-1 slot already holds the reserved stub for
        ``name``, this is a no-op. If the slot holds a different occupant
        (a real lane, or another heatmap's stub), the caller is
        responsible for removing it first — this method only ADDS the
        reserved stub when row 1 is empty or holds something else.
        Plan 02-04 cancel preamble calls this AFTER iterating its own
        ``_heatmap_lanes.items()`` and calling ``lane.remove(layout)`` on
        each.
        """
        stub = self._reserved_lane_stubs.get(name)
        if stub is None:
            return
        current = self.graphics_layout.getItem(1, 0)
        if current is stub:
            # Already restored — idempotent no-op.
            return
        self.graphics_layout.addItem(stub, row=1, col=0)

    def _reset_to_reserved_lane_stub(self) -> None:
        """Sweep — ensure each registered reserved-lane stub is restored.

        Iterates ``self._reserved_lane_stubs`` and re-adds any stub whose
        slot has been displaced. Plan 02-04 cancel preamble uses this
        AFTER removing every lane in ``MainWindow._heatmap_lanes`` so a
        cancelled file-switch leaves the layout in its initial "reserved
        stub only" shape regardless of how many lanes were attached.
        """
        for name in self._reserved_lane_stubs:
            self._restore_reserved_lane_stub(name)

    # ------------------------------------------------------------- zoom + fit
    def fit_view(self) -> None:
        """Restore the x-range to ``[0, duration_s]`` (UI-SPEC §Zoom > Fit)."""
        if self._duration_s > 0.0:
            self.waveform_plot.setXRange(0.0, self._duration_s, padding=0)

    def zoom(self, step: float) -> None:
        """Zoom the x-axis by ``step`` (1.25 in, 1/1.25 out) centered on the view center.

        UI-SPEC §Zoom: 1.25× per wheel notch / toolbar click. Y-axis is
        locked (``setMouseEnabled(y=False)``).
        """
        if step <= 0:
            return
        view_range = self.waveform_plot.viewRange()
        x_min, x_max = view_range[0]
        width = x_max - x_min
        if width <= 0:
            return
        center = 0.5 * (x_min + x_max)
        new_half = (width / step) * 0.5
        self.waveform_plot.setXRange(center - new_half, center + new_half, padding=0)

    # ----------------------------------- Plan 03-01 region overlay attachment
    def set_regions_overlay(self, overlay: "RegionsOverlay") -> None:
        """Install the :class:`RegionsOverlay` (Plan 03-01).

        Called by :class:`MainWindow.__init__` after both the WaveformView
        and the RegionsOverlay are constructed. The overlay owns the
        per-region LinearRegionItem widgets on this view's PlotItem; the
        eventFilter below routes Shift+drag gestures into the overlay.
        """
        self._regions_overlay = overlay

    # -------------------------------------------- Plan 03-02 region mode toggle
    def set_region_select_mode(self, active: bool) -> None:
        """Toggle the Region Select mode (Plan 03-02 / CONTEXT D-A1-1).

        When ``active`` is True:

        * A plain Left+drag on the waveform creates a region (no Shift
          required). Shift+drag continues to work as before — the combo
          gesture is always available.
        * The viewport cursor swaps to ``Qt.CrossCursor`` (visual feedback
          that the next click will start a region, not a pan).

        When ``active`` is False:

        * Plain Left+drag falls through to the Phase 1 pan path.
        * Only Shift+drag creates regions.
        * The viewport cursor reverts to ``Qt.OpenHandCursor`` (idle pan
          affordance per UI-SPEC §Pan).

        Toolbar wiring: :class:`MainWindow` connects its
        ``_tb_region_select.toggled`` signal to this method, so a single
        QAction click flips the mode.
        """
        self._region_select_mode_active = bool(active)
        viewport = self.graphics_layout.viewport()
        if self._region_select_mode_active:
            viewport.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            viewport.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def _viewport_x_to_data_x(self, px: int) -> float:
        """Map a viewport-pixel x to a data-coordinate x (seconds).

        Uses the GraphicsLayoutWidget → scene → ViewBox.mapSceneToView
        chain (NOT QMouseEvent.scenePosition() — that carries
        window-relative coords when delivered through the
        ``viewport.eventFilter`` path, leaking chrome offset into the
        result, per Plan 02-05's chrome-offset regression pin).
        """
        vb = self.waveform_plot.getViewBox()
        viewport_pt = QPointF(float(px), 0.0)
        scene_pt = self.graphics_layout.mapToScene(viewport_pt.toPoint())
        data_pt = vb.mapSceneToView(scene_pt)
        return float(data_pt.x())

    # ------------------------------------------------- cursor swap event filter
    def eventFilter(self, obj, event: QEvent) -> bool:
        """Swap cursor over the plot per UI-SPEC §Pan; track mouse-down x for Plan 02-04.

        OpenHandCursor when the cursor is idle over the plot; ClosedHandCursor
        during an active left-button drag. We rely on PyQtGraph's ViewBox to
        actually perform the pan — we only swap the cursor.

        Plan 02-01: ADDS ``self._mouse_down_x_px = int(event.position().x())`` on
        left-button press and clears it on release. NO signal is emitted —
        Plan 02-04 will compute ``abs(up_x - _mouse_down_x_px) <= SEEK_THRESHOLD_PX``
        in the release branch and emit ``seek_requested`` when the click is
        in the dead-zone. We pay the bookkeeping cost in Plan 02-01 so Plan
        02-04 doesn't need to re-touch this filter's shape.
        """
        viewport = self.graphics_layout.viewport()
        if obj is viewport:
            etype = event.type()
            if etype == QEvent.Type.Enter:
                if not self._dragging:
                    viewport.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            elif etype == QEvent.Type.Leave:
                if not self._dragging:
                    viewport.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            elif etype == QEvent.Type.MouseButtonPress:
                mouse_event = event  # type: QMouseEvent
                # Plan 03-02 — middle-mouse pan branch. PyQtGraph 0.13.7's
                # ViewBox does NOT pan on middle-button natively, so we
                # synthesize the pan ourselves. Middle-button drag pans
                # the x-axis regardless of Region Select mode — the
                # user-discoverable contract is "middle-drag pans". We
                # consume the event so the click-to-seek + region-create
                # paths never see it.
                if mouse_event.button() == Qt.MouseButton.MiddleButton:
                    self._middle_pan_active = True
                    self._middle_pan_last_x = int(mouse_event.position().x())
                    self._middle_pan_last_y = int(mouse_event.position().y())
                    viewport.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                    event.accept()
                    return True
                if mouse_event.button() == Qt.MouseButton.LeftButton:
                    # Plan 03-01 + Plan 03-02 — region-create branch.
                    # Shift is checked AT MouseButtonPress (CONTEXT D-A1-1) —
                    # holding Shift mid-drag does not retroactively turn a
                    # pan into a region-create. Plan 03-02 ADDS the
                    # toolbar-mode condition: when Region Select mode is
                    # ON, a plain Left+drag (no modifier) also starts a
                    # draft. The combo gesture (Shift+drag) is always
                    # available so the keyboard-only power user never
                    # loses access regardless of the toolbar toggle.
                    shift_held = bool(
                        mouse_event.modifiers()
                        & Qt.KeyboardModifier.ShiftModifier
                    )
                    create_region = (
                        (shift_held or self._region_select_mode_active)
                        and self._regions_overlay is not None
                    )
                    if create_region:
                        x_data = self._viewport_x_to_data_x(
                            int(mouse_event.position().x())
                        )
                        self._regions_overlay.start_draft(x_data)
                        self._region_draft_active = True
                        # Crosshair cursor signals the region-draft is in
                        # flight. Reset on release. We do NOT set
                        # _dragging or _mouse_down_x_px — those belong to
                        # the pan + click-to-seek path which we are
                        # explicitly bypassing for this gesture.
                        viewport.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                        event.accept()
                        return True
                    self._dragging = True
                    # Plan 02-01: record press x-pixel for the future
                    # click-to-seek dead-zone check in Plan 02-04.
                    self._mouse_down_x_px = int(mouse_event.position().x())
                    viewport.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            elif etype == QEvent.Type.MouseMove:
                # Plan 03-02 — middle-mouse pan: translate the waveform
                # ViewBox by the pixel delta mapped to data coordinates.
                # Must be checked BEFORE the region-draft branch so a
                # middle-drag never bleeds into the draft path.
                if self._middle_pan_active:
                    mouse_event = event
                    x_now = int(mouse_event.position().x())
                    y_now = int(mouse_event.position().y())
                    dx_px = x_now - self._middle_pan_last_x
                    self._middle_pan_last_x = x_now
                    self._middle_pan_last_y = y_now
                    vb = self.waveform_plot.getViewBox()
                    # Map a pixel delta in scene coordinates to a data-x
                    # delta, then translate the ViewBox by the inverse
                    # (drag right → view shifts left, like a DAW).
                    pt1 = vb.mapSceneToView(QPointF(0.0, 0.0))
                    pt2 = vb.mapSceneToView(QPointF(float(dx_px), 0.0))
                    vb.translateBy(x=-(pt2.x() - pt1.x()), y=0)
                    event.accept()
                    return True
                # Plan 03-01 — update the in-progress region draft on
                # every MouseMove while Shift+drag is active. Plain
                # MouseMove (no draft) falls through to PyQtGraph's
                # default pan handler unchanged.
                if (
                    self._region_draft_active
                    and self._regions_overlay is not None
                ):
                    mouse_event = event
                    x_data = self._viewport_x_to_data_x(
                        int(mouse_event.position().x())
                    )
                    self._regions_overlay.update_draft(x_data)
                    return True
            elif etype == QEvent.Type.MouseButtonRelease:
                mouse_event = event
                # Plan 03-02 — middle-mouse pan release. Restore the
                # appropriate idle cursor based on the current mode (cross
                # when Region Select mode is ON, openhand otherwise). MUST
                # be checked BEFORE the LeftButton branches below — a
                # middle-button release would otherwise fall through to
                # the click-to-seek dead-zone check (which compares
                # against a never-recorded _mouse_down_x_px for the
                # middle-button press) and is harmless but messy.
                if (
                    self._middle_pan_active
                    and mouse_event.button() == Qt.MouseButton.MiddleButton
                ):
                    self._middle_pan_active = False
                    viewport.setCursor(
                        QCursor(
                            Qt.CursorShape.CrossCursor
                            if self._region_select_mode_active
                            else Qt.CursorShape.OpenHandCursor
                        )
                    )
                    event.accept()
                    return True
                # Plan 03-01 — commit the region draft if one is in flight.
                # MUST be checked BEFORE the existing click-to-seek dead-zone
                # branch below — otherwise a Shift+drag release would also
                # emit a spurious seek_requested for the release coordinate.
                if (
                    self._region_draft_active
                    and self._regions_overlay is not None
                    and mouse_event.button() == Qt.MouseButton.LeftButton
                ):
                    x_data = self._viewport_x_to_data_x(
                        int(mouse_event.position().x())
                    )
                    self._regions_overlay.commit_draft(x_data)
                    self._region_draft_active = False
                    # Clear press bookkeeping so the dead-zone check below
                    # does not fire on this same release.
                    self._mouse_down_x_px = None
                    self._dragging = False
                    # Plan 03-02 — restore the mode-appropriate idle cursor.
                    # When Region Select mode is ON the user-facing
                    # affordance must stay as CrossCursor so the next
                    # plain-drag is visually advertised; only revert to
                    # OpenHandCursor when mode is OFF (Shift+drag path).
                    viewport.setCursor(
                        QCursor(
                            Qt.CursorShape.CrossCursor
                            if self._region_select_mode_active
                            else Qt.CursorShape.OpenHandCursor
                        )
                    )
                    event.accept()
                    return True
                if mouse_event.button() == Qt.MouseButton.LeftButton:
                    self._dragging = False
                    # Plan 02-05 — click-vs-drag disambiguation:
                    # If the press/release pixel delta is ≤ SEEK_THRESHOLD_PX
                    # we treat the gesture as a "click" and emit
                    # ``seek_requested(seconds)`` with the data-space x.
                    # Mapping path: viewport-pixel → scene (via
                    # ``graphics_layout.mapToScene``) → data (via
                    # ``ViewBox.mapSceneToView``). We MUST NOT use
                    # ``QMouseEvent.scenePosition()`` here — when the event
                    # is delivered through ``viewport.eventFilter`` (not via
                    # a QGraphicsView chain) it carries window-relative, not
                    # scene-relative, coordinates, which leaks chrome offset
                    # (menubar/toolbar/sidebar) into the emitted seconds.
                    # Anything larger than the threshold is a drag
                    # (PyQtGraph's ViewBox pan handler owns it; no signal
                    # fired). A release without a prior press
                    # (``_mouse_down_x_px`` is None) is ignored — defensive
                    # against stale state.
                    if self._mouse_down_x_px is not None:
                        up_x = int(mouse_event.position().x())
                        delta = abs(up_x - self._mouse_down_x_px)
                        if delta <= SEEK_THRESHOLD_PX:
                            vb = self.waveform_plot.getViewBox()
                            viewport_pos = mouse_event.position()
                            scene_pos = self.graphics_layout.mapToScene(viewport_pos.toPoint())
                            data_pt = vb.mapSceneToView(scene_pos)
                            self.seek_requested.emit(float(data_pt.x()))
                    # Always clear the press-coord bookkeeping after a
                    # release so a stray release (e.g. focus change) doesn't
                    # double-fire the next time around.
                    self._mouse_down_x_px = None
                    # Plan 03-02 — preserve CrossCursor when mode is ON.
                    viewport.setCursor(
                        QCursor(
                            Qt.CursorShape.CrossCursor
                            if self._region_select_mode_active
                            else Qt.CursorShape.OpenHandCursor
                        )
                    )
        return super().eventFilter(obj, event)
