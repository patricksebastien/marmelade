"""One heatmap rendered as an ImageItem strip below the waveform (Plan 02-03).

Each heatmap lane is a thin ``pg.PlotItem`` (28 px tall — D-06) that
re-renders a 1-D ``float32`` array of heatmap values into three discrete
visual bands (silent / quiet / loud — D-03). The lane:

* Reuses the ``imageAxisOrder='row-major'`` global config flipped at
  ``theme.py`` module-load (Plan 02-01) so the underlying ``ImageItem``
  reads the band-index array in the expected orientation.
* Mean-aggregates inputs longer than :data:`MAX_RENDER_HEATMAP_BINS` (4000)
  down to that bin count BEFORE thresholding — same viewport-density
  discipline as :func:`marmelade.ui.waveform_view.render_proxy`'s
  ``MAX_RENDER_PROXY_PAIRS`` aggregation, but using MEAN per bin (D-05 —
  energy averages cleanly; max-per-bin biases bright).
* Thresholds at render time (NOT inside the algorithm) so a future
  recompute / threshold adjustment never has to rebuild the .dat cache —
  the algorithm stores adaptive thresholds in
  :class:`marmelade.heatmaps.base.HeatmapResult` /
  :class:`marmelade.audio.heatmap_cache.HeatmapHeader` and the renderer
  applies them on the GUI thread (D-03 + RESEARCH §Open Q #1 resolution).

The three-row uint8 RGBA LUT (:data:`DEFAULT_LUT`) is shared across every
lane (MVP shape — Plan 02-04 may add a sidebar control to swap LUTs per
heatmap). The loud-band color :literal:`#5A8FBF` is a desaturated cyan-blue
variant of the waveform's :literal:`#7FBFFF`; it MUST NOT be the playhead
accent :literal:`#4DA3FF` (UI-SPEC + D-06 — accent reserved for the
playhead).

ImageItem contract — NOT the four PlotDataItem flags:
    Phase 1's four-flag PlotDataItem contract
    (``setDownsampling`` + ``setClipToView`` + ``setSkipFiniteCheck`` +
    pen width=1) does NOT apply here. ``ImageItem`` has its own
    downsampling API (``setAutoDownsample``); the safe Phase-2 idiom is
    ``ImageItem.setLookupTable(lut)`` + ``setLevels([0, 2])`` +
    ``setAutoDownsample(True)`` (RESEARCH §Pattern 3).

The lane is plain Python — NOT a ``QWidget`` subclass. It owns a
``pg.PlotItem`` exposed as :attr:`HeatmapLaneView.plot_item`. The owning
``WaveformView`` installs this PlotItem at row 1 of its
``pg.GraphicsLayoutWidget`` via :meth:`WaveformView.add_heatmap_lane`.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg


# USER FEEDBACK 2026-05-13 / CONTEXT D-05 — mean-aggregate when the
# heatmap has more values than viewport pixels. Structural twin of
# ``waveform_view.MAX_RENDER_PROXY_PAIRS`` but SEPARATE constant per
# CONTEXT.md discretion ("future tuning is independent"). MEAN not MAX
# per D-05: energy averages cleanly; max-per-bin biases bright. 4000
# bins is ~2× a typical 1920-px viewport so PyQtGraph's downsample pass
# sees comfortable headroom for 3840-px ultra-wide / 4K displays.
# Upgrade path if 4000 ever proves too coarse at extreme zoom-in: switch
# to a ``sigXRangeChanged``-driven re-aggregate over the visible slice
# (Option C in plan 01-09 — never adopted for the waveform; reuse if
# the heatmap needs it before the waveform does).
MAX_RENDER_HEATMAP_BINS = 4000


# Three-row uint8 RGBA LUT — silent / quiet / loud.
#
#   Row 0 — silent — #1E1E1E (UI-SPEC dominant surface; the lane "disappears"
#                              into the background when the section is silent).
#   Row 1 — quiet  — #2F3A47 (cool grey-blue, low saturation).
#   Row 2 — loud   — #5A8FBF (desaturated cyan-blue variant of the waveform's
#                              #7FBFFF). NOT #4DA3FF — that color is reserved
#                              for the playhead per UI-SPEC + D-06.
#
# Alpha 0xFF on every row — the lane is opaque against the dominant surface.
DEFAULT_LUT = np.array(
    [
        [0x1E, 0x1E, 0x1E, 0xFF],
        [0x2F, 0x3A, 0x47, 0xFF],
        [0x5A, 0x8F, 0xBF, 0xFF],
    ],
    dtype=np.uint8,
)


class HeatmapLaneView:
    """One heatmap rendered as a 28-px-tall ImageItem strip.

    Lifecycle:
        1. Construct: ``lane = HeatmapLaneView(name, waveform_plot, lut)``.
           This builds a ``pg.PlotItem`` configured for the lane (no axes,
           28 px max height, x-linked to ``waveform_plot``) and installs a
           single ``pg.ImageItem`` with the given LUT.
        2. Render: ``lane.render(values, sample_rate, samples_per_value,
           silent_quiet_threshold, quiet_loud_threshold)`` aggregates
           ``values`` down to at most :data:`MAX_RENDER_HEATMAP_BINS` cells
           via MEAN per bin (D-05), thresholds the aggregate into a uint8
           band-index array (silent=0, quiet=1, loud=2), and pushes it to
           the ImageItem.
        3. Remove: ``lane.remove(layout)`` removes the PlotItem from the
           given ``pg.GraphicsLayoutWidget`` and calls ``deleteLater()`` on
           it (Pitfall #8 — ``removeItem`` alone leaks).

    Attributes:
        plot_item: The owned ``pg.PlotItem``. The owning ``WaveformView``
            installs it at row 1 of its ``GraphicsLayoutWidget``.
        last_rendered_band_indices: The most recent uint8 band-index array
            produced by :meth:`render` — exposed for offscreen unit tests
            to observe the render result without inspecting the ImageItem
            backing buffer. ``None`` until the first :meth:`render` call.
    """

    def __init__(
        self,
        name: str,
        waveform_plot: pg.PlotItem,
        lut: np.ndarray = DEFAULT_LUT,
    ) -> None:
        self._name = name
        self._plot = pg.PlotItem()

        # Lane PlotItem config: pannable x-axis, locked y-axis, no menu,
        # no axes, dominant-surface background.
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.getViewBox().setBackgroundColor("#1E1E1E")
        # D-06 — lane height 24-32 px discretionary band; we pick 28 px
        # so the heatmap reads as a distinct strip without dominating the
        # vertical layout. ``setMaximumHeight`` (NOT ``setFixedHeight``) so
        # Plan 02-04 / Plan 02-05 may stack multiple lanes without each
        # one forcing the row geometry.
        self._plot.setMaximumHeight(28)
        # Pin default x-axis padding to zero so a linked setXRange(_, _,
        # padding=0) doesn't drift via the linked-viewbox's default
        # padding (the same alignment discipline Plan 02-01 applies to
        # the reserved row-1 stub).
        self._plot.getViewBox().setDefaultPadding(0.0)
        # setXLink wires the lane's x-axis to the waveform's ViewBox —
        # bi-directional sync. Established at construction so pan/zoom
        # already works the moment the lane is added to the layout (the
        # owning WaveformView matches the left-axis widths so the linked
        # range does not geometry-drift per Plan 02-01 alignment fix).
        self._plot.setXLink(waveform_plot)

        # ImageItem with the 3-row uint8 RGBA LUT and pinned levels.
        # setLevels([0, 2]) makes the LUT row-index = band-index lookup
        # deterministic regardless of input data range. autoDownsample on
        # ImageItem is the analog of PlotDataItem's setDownsampling.
        self._img = pg.ImageItem()
        self._img.setLookupTable(lut)
        self._img.setLevels([0, 2])
        self._img.setAutoDownsample(True)
        self._plot.addItem(self._img)

        # Public attributes for unit-test observation — expose the most
        # recent uint8 band-index array AND the most recent
        # data-coordinate duration so tests don't have to inspect the
        # ImageItem backing buffer (in (1, n_bins) shape after setImage)
        # OR reverse-engineer the QGraphicsItem transform that
        # ``setRect`` installs. Both default to ``None`` until first
        # render.
        self.last_rendered_band_indices: np.ndarray | None = None
        self.last_rendered_duration_s: float | None = None
        # Plan 03-03 — kwargs from the most recent render() call, exposed
        # via :meth:`last_render_args` so MainWindow can re-render with
        # a freshly computed Trash mask without reaching into private
        # lane state (Qt-tier correct per the RESEARCH Architectural
        # Responsibility Map — render state lives on the lane, NOT on
        # MainWindow). The mask itself is intentionally NOT stashed:
        # MainWindow recomputes it from the live regions set on every
        # mutation.
        self._last_render_args: dict | None = None

    @property
    def plot_item(self) -> pg.PlotItem:
        """The owned ``pg.PlotItem`` — installed at the host layout's row 1."""
        return self._plot

    def render(
        self,
        values: np.ndarray,
        sample_rate: int,
        samples_per_value: int,
        silent_quiet_threshold: float,
        quiet_loud_threshold: float,
        trash_mask: np.ndarray | None = None,
    ) -> None:
        """Aggregate + threshold + draw.

        Args:
            values: 1-D ``float32`` array of heatmap values (one per
                ``samples_per_value`` source samples).
            sample_rate: Source sample rate in Hz — used to compute the
                ImageItem's x-extent (data-coordinate width).
            samples_per_value: Source samples per heatmap value. Matches
                the proxy's ``samples_per_pixel`` (256 by default) so the
                heatmap and waveform share their x-axis without
                resampling.
            silent_quiet_threshold: Adaptive threshold separating silent
                and quiet bands. Values ``< threshold`` → band 0 (silent).
            quiet_loud_threshold: Adaptive threshold separating quiet and
                loud bands. Values ``≥ threshold`` → band 2 (loud);
                in-between → band 1 (quiet).
            trash_mask: Plan 03-03 D-A2-3. Optional boolean array; when
                provided AND its size matches the aggregated render
                resolution, masked indices are forced to band 0 (silent /
                background — the lane visually disappears into the
                dominant surface across the Trash range). If the mask
                length is a clean multiple of the render bin count, an
                OR-per-bin aggregation collapses it to the render
                resolution. Mismatched sizes are silently dropped (no
                crash) — defensive only; MainWindow passes the correct
                shape via :meth:`last_render_args`.

        The aggregation uses the SAME viewport-density discipline as
        :func:`marmelade.ui.waveform_view.render_proxy` — when
        ``values.size > MAX_RENDER_HEATMAP_BINS``, the array is reshaped
        into ``(MAX_RENDER_HEATMAP_BINS, bin_size)`` and the row-mean
        becomes the per-bin value. The trailing remainder (size
        ``values.size % MAX_RENDER_HEATMAP_BINS``) is folded into the LAST
        bin via a weighted mean so the aggregation covers the full input
        without an off-by-one truncation (mirror of waveform_view's
        ``proxy_arr[n_full:]`` tail fold).
        """
        # Plan 03-03 — stash the non-mask kwargs for :meth:`last_render_args`.
        # The mask is intentionally NOT stashed: MainWindow recomputes it
        # on every regions_changed mutation, so a stashed mask would go
        # stale immediately. Done at the TOP of render so the stash is
        # always consistent with the call that's about to execute.
        self._last_render_args = {
            "values": values,
            "sample_rate": sample_rate,
            "samples_per_value": samples_per_value,
            "silent_quiet_threshold": silent_quiet_threshold,
            "quiet_loud_threshold": quiet_loud_threshold,
        }

        original_length = int(values.size)

        if original_length > MAX_RENDER_HEATMAP_BINS:
            bin_size = original_length // MAX_RENDER_HEATMAP_BINS
            n_full = MAX_RENDER_HEATMAP_BINS * bin_size
            view = values[:n_full].reshape(MAX_RENDER_HEATMAP_BINS, bin_size)
            aggregated = view.mean(axis=1).astype(np.float32, copy=False)
            tail = values[n_full:]
            if tail.size > 0:
                # Weighted mean — mirror waveform_view's tail fold.
                # aggregated[-1] currently holds the mean of bin_size
                # samples; the tail adds tail.size more samples. The
                # combined mean is the sum / total-count.
                aggregated[-1] = (
                    float(aggregated[-1]) * bin_size + float(tail.sum())
                ) / (bin_size + int(tail.size))
            render_values = aggregated
        else:
            render_values = values

        # Threshold-at-render — D-03 three discrete bands. Default = 0
        # (silent); we then set band 2 (loud) and band 1 (quiet) via
        # boolean masks. The order matters: setting band 2 first means
        # the band-1 mask must explicitly exclude values ≥ quiet_loud
        # threshold (it does, via the second comparand).
        band_idx = np.zeros_like(render_values, dtype=np.uint8)
        band_idx[render_values >= quiet_loud_threshold] = 2
        band_idx[
            (render_values >= silent_quiet_threshold)
            & (render_values < quiet_loud_threshold)
        ] = 1

        # Plan 03-03 — apply Trash render-mask (D-A2-3). Masked indices
        # are forced to band 0 (silent / background = #1E1E1E per the
        # Phase 2 LUT — the lane disappears into the dominant surface
        # over the Trash range). The simple v1 path reuses the existing
        # band-0 row rather than extending DEFAULT_LUT to a 4th row.
        # RESEARCH §Pitfall #12 confirmed band-0 reuse is sufficient.
        if trash_mask is not None and trash_mask.size > 0:
            if trash_mask.size == band_idx.size:
                band_idx[trash_mask.astype(bool)] = 0
            elif trash_mask.size >= band_idx.size:
                ratio = trash_mask.size // band_idx.size
                if ratio > 1 and trash_mask.size >= ratio * band_idx.size:
                    # Collapse to render bin count via OR-aggregation.
                    mask_agg = (
                        trash_mask[: ratio * band_idx.size]
                        .reshape(band_idx.size, ratio)
                        .any(axis=1)
                    )
                    band_idx[mask_agg] = 0
                # else: non-integer ratio — defensive drop (no crash).
            # else: mask shorter than render bins — defensive drop.

        # Expose the intermediate band-index array for offscreen unit
        # tests — saves them from reverse-engineering the ImageItem
        # backing buffer.
        self.last_rendered_band_indices = band_idx

        # Push to the ImageItem. autoLevels=False because we set
        # setLevels([0, 2]) at construction — auto-leveling would scan
        # the new buffer and recompute, defeating the LUT row mapping.
        # Reshape (n,) → (1, n) because imageAxisOrder='row-major' (set
        # in theme.py at module load) reads ImageItem.image as (rows,
        # cols); the lane is a single 1-row strip.
        self._img.setImage(band_idx.reshape(1, -1), autoLevels=False)

        # x-extent in seconds — uses ORIGINAL length so the visible time
        # domain spans the full input regardless of aggregation
        # (aggregation changes cell width, not total extent).
        duration_s = (original_length * samples_per_value) / float(sample_rate)
        self._img.setRect(0.0, 0.0, duration_s, 1.0)
        self.last_rendered_duration_s = duration_s

    def last_render_args(self) -> dict | None:
        """Return the kwargs from the most recent ``render`` call (Plan 03-03).

        Plan 03-03 — :meth:`marmelade.ui.main_window.MainWindow._refresh_heatmap_trash_mask`
        uses this to re-render the lane with a freshly computed Trash
        mask whenever regions change, WITHOUT reaching into private lane
        state (Qt-tier correct per the RESEARCH Architectural
        Responsibility Map — render kwargs belong on the lane, not on
        the window).

        Returns ``None`` if :meth:`render` has never been called. The
        returned dict references the same arrays the caller passed in;
        callers MUST NOT mutate them in place.
        """
        return self._last_render_args

    def rebind_thresholds(
        self,
        silent_quiet_threshold: float,
        quiet_loud_threshold: float,
    ) -> None:
        """Re-render the lane with new thresholds against the cached values.

        Phase 6 — display-time Smart Apply path (D-06). Reuses the kwargs
        stashed at the last :meth:`render` call (``_last_render_args``)
        for everything except the two threshold floats. Plan 03-03's
        Trash-mask refresh path already established this stashing
        discipline; Phase 6 re-uses it for the gear-Apply re-band path.

        The cached ``.dat`` file is NOT touched — only the render-time
        thresholding changes. This is what makes the Energy gear Apply
        feel instant (no worker spawn, no cache rewrite, no progress UI).

        Raises:
            RuntimeError: if :meth:`render` has never been called.
        """
        if self._last_render_args is None:
            raise RuntimeError(
                "rebind_thresholds called before first render()"
            )
        kwargs = dict(self._last_render_args)
        kwargs["silent_quiet_threshold"] = float(silent_quiet_threshold)
        kwargs["quiet_loud_threshold"] = float(quiet_loud_threshold)
        self.render(**kwargs)

    def remove(self, layout: pg.GraphicsLayoutWidget) -> None:
        """Remove this lane's PlotItem from ``layout`` and schedule deletion.

        Pitfall #8 (RESEARCH § Worker / Qt cleanup) — ``removeItem`` alone
        keeps the PlotItem reachable from the underlying ``QGraphicsScene``
        and leaks. Calling ``deleteLater()`` after ``removeItem`` schedules
        the C++ object for destruction on the next event-loop tick.

        Pre-deletion ``setXLink(None)``: PyQtGraph stores back-references
        on the waveform_plot's ViewBox so a linkedXChanged still tries to
        propagate to a half-deleted lane ViewBox during the
        ``deleteLater`` window — which crashes with "Internal C++ object
        already deleted" when the linked view re-checks bounds. Unhooking
        the x-link before scheduling deletion neutralises that race
        (verified via the deleteLater unit pin under offscreen Qt).
        """
        # Unhook setXLink before deleteLater so the waveform_plot's
        # ViewBox does not try to propagate bounds into the destroyed
        # lane ViewBox during the post-removeItem event-loop tick.
        self._plot.setXLink(None)
        layout.removeItem(self._plot)
        self._plot.deleteLater()
