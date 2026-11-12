"""The Plots window's heatmap view, clustering and PCA runs, and the numbers behind the picture.

``_PlotsViewsMixin`` is a part of :class:`nanoscope.gui.plots_panel.PlotsWindow`, listed
before ``QtWidgets.QMainWindow`` in its bases. It holds the heatmap's colour map, colour
bar and draw, the cluster legend and the PCA text helpers, ``run_clustering`` and
``run_pca`` with their warnings and summary, and the helpers that turn the drawn figure
into exportable numbers (the series labels, the cluster of each drawn point, the plot
data table and the PCA loadings). The histogram, scatter, KDE and PCA draws themselves
stay in ``PlotsWindow``, with ``save_image`` and ``export_data``.

It is a plain-Python mixin: no ``__init__``, no signals and no state of its own, so every
attribute its methods read (``self._plot``, ``self._cluster`` ...) is set by
``PlotsWindow.__init__``. The constants and helpers it uses come from
:mod:`nanoscope.gui.plots_constants`; it never imports ``plots_panel`` or another part.

Every method below was moved out of ``plots_panel.py`` without an edit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from ..analysis import event_shape, populations
from ..core.sources import PlotSource, amplitude_comparability, source_comparability
from .categorical import category_colour, category_counts, legend_swatch
from .plots_constants import (
    HISTOGRAM,
    SCATTER,
    PCA,
    KDE,
    _PLOT_TYPES,
    _HEATMAP_COLORMAPS,
    _COLOUR_CLUSTER,
    _NORM_COUNT,
    _NORM_DENSITY,
    _NORM_PROB,
    _METHOD_IDS,
    _K_AUTO,
    _plot_quantity_bases,
)


class _PlotsViewsMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    def _heatmap_colormap(self) -> pg.ColorMap:
        """The style's heatmap colour map, falling back to viridis for a name this
        build does not carry (a newer build's template, a hand-edit) rather than raising
        in the middle of a draw."""
        name = self._style.heatmap_colormap
        if name not in _HEATMAP_COLORMAPS:
            name = "viridis"
        try:
            return pg.colormap.get(name)
        except Exception:   # noqa: BLE001 - a missing bundled map must not break a plot
            return pg.colormap.get(name, source="matplotlib")

    def _remove_heatmap_bar(self) -> None:
        """Take the colour bar out of the PlotItem's own layout.

        It lives IN the PlotItem's layout -- not in a widget beside it -- because export
        renders the PlotItem, and a colour bar the saved figure does not carry leaves
        the colours unreadable. ``plot.clear()`` only empties the ViewBox, so the bar is
        detached here by hand (the same reason ``_drop_legend`` exists), and the column
        it occupied is released so the next plot gets its width back.
        """
        bar = getattr(self, "_heatmap_bar", None)
        self._heatmap_bar = None
        self._heatmap_image = None
        if bar is None:
            return
        layout = self._plot.layout
        layout.removeItem(bar)
        bar.setParentItem(None)
        scene = bar.scene()
        if scene is not None:
            scene.removeItem(bar)
        layout.setColumnFixedWidth(4, 0)
        layout.setColumnSpacing(4, 0)

    def _style_heatmap_bar(self) -> None:
        """Foreground, font and colour map of the colour bar, from the style. Called by
        the draw and by the appearance-only path, which must never re-bin."""
        bar = getattr(self, "_heatmap_bar", None)
        image = getattr(self, "_heatmap_image", None)
        if bar is None:
            return
        cmap = self._heatmap_colormap()
        bar.setColorMap(cmap)
        if image is not None:
            image.setColorMap(cmap)
        fg = self._fg_hex()
        for side in ("left", "right", "top", "bottom"):
            axis = bar.getAxis(side)
            axis.setPen(pg.mkPen(fg))
            axis.setTextPen(pg.mkPen(fg))
            axis.setStyle(tickFont=self._qfont(self._tick_pt()))
        bar.getAxis("left").setLabel("Event count", **self._label_style())

    def _draw_heatmap(self, drawn: list[tuple[PlotSource, pd.DataFrame]]) -> None:
        """A 2-D EVENT-COUNT image of two columns, for the first ticked source.

        THE KERNEL IS EVENT WAVEFORMS' OWN (``event_shape.density_2d``): edges spanning
        the source's min..max per axis (geometric on a log axis), points dropped where
        either coordinate is non-finite or non-positive on a log axis. One definition of
        "an event-count heatmap" in the app, so this one and that window's agree bin for
        bin over the same rows -- a second binning here would be a second answer.

        COUNTS ARE COUNTS. The colour is the number of events in a cell, not a density
        and not smoothed; the KDE page is the smoothed estimate, and the guide's rule is
        that the two are different quantities. Nothing is normalised, so the Normalize
        combo (a histogram setting) does not apply here.

        LOG AXES THE KDE PAGE'S WAY: the image is placed in log10 view coordinates by
        hand (``setLogMode`` only transforms PlotDataItems), so the decade ticks the
        log axis draws read the data correctly. The time unit scales the ticks only.
        """
        xcol, ycol = self._heatmap_x.currentText(), self._heatmap_y.currentText()
        if not xcol or not ycol:
            self._message("Pick numeric X and Y columns.")
            return
        log_x, log_y = self._heatmap_logx.isChecked(), self._heatmap_logy.isChecked()
        src, df = drawn[0]
        skipped = len(drawn) - 1
        if xcol not in df.columns or ycol not in df.columns:
            self._message(f"“{src.label}” has no “{xcol}” or “{ycol}” column to bin.")
            return
        x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=np.float64)
        nb = int(self._heatmap_bins.value())
        counts, xe, ye = event_shape.density_2d(x, y, xbins=nb, ybins=nb,
                                                xlog=log_x, ylog=log_y)
        kept = int(counts.sum())
        if kept == 0:
            self._message(
                "No events fall on these axes — every row is non-finite, or non-positive "
                "on a log axis. Check the log toggles.")
            return

        details = [f"Event-count heatmap {ycol} vs {xcol}   bins = {nb} × {nb}   "
                   f"source = {src.label}   n = {kept}",
                   *self._incoming_notes()]
        if skipped:
            details.append(
                f"ONE SOURCE AT A TIME: this heatmap shows the first ticked source only "
                f"(“{src.label}”); {skipped} other ticked source(s) are not drawn. Untick "
                "it to see the next ticked source.")
        dropped = int(x.size - kept)
        if dropped:
            details.append(f"{dropped} of {x.size} row(s) dropped (non-finite, or "
                           "non-positive on a log axis).")
        # THE SAME VERDICT THE SCATTER AND KDE PAGES GIVE FOR THESE COLUMNS -- and Event
        # waveforms' own heatmap, which audits the amplitude whenever it is on an axis. A
        # count image of `amplitude_pA` with stale or missing selections is exactly as
        # unreadable as a scatter of it, and a heatmap that alone stayed silent would be
        # the page to export from to lose the caveat. Judged over the DRAWN source only,
        # not `_comparability()`'s every drawn frame: the other ticked sources are not
        # in this image, so their baselines cannot make it incomparable.
        verdict = source_comparability([src], (xcol, ycol), frames=[df])
        details.extend(self._column_notes([xcol, ycol], [src]))
        details.extend(verdict.warnings)
        details.extend(self._detection_notes([src]))
        details.append(
            "colour = events per cell (counts, not a KDE density and not normalised); "
            f"cells span the source's own range, {'log10-spaced' if log_x else 'linear'} "
            f"on x and {'log10-spaced' if log_y else 'linear'} on y — the binning "
            "Event waveforms uses. Lasso and point-click selection are not available on "
            "a heatmap.")

        self._clear_plot()
        self._cache_figure_source(src)
        xe_v = np.log10(xe) if log_x else np.asarray(xe, dtype=np.float64)
        ye_v = np.log10(ye) if log_y else np.asarray(ye, dtype=np.float64)
        cmax = max(float(counts.max()), 1.0)
        # density_2d stores counts[x_bin, y_bin]; pin the orientation (Event waveforms'
        # rule), because the app's global imageAxisOrder is not this item's to assume.
        image = pg.ImageItem(axisOrder="col-major")
        image.setImage(counts, levels=(0.0, cmax))
        image.setRect(QtCore.QRectF(float(xe_v[0]), float(ye_v[0]),
                                    float(xe_v[-1] - xe_v[0]), float(ye_v[-1] - ye_v[0])))
        # What `plot_source_data` exports: the drawn counts with their edges in DATA
        # units, read off the item so the CSV is the picture, not a re-binning.
        image._heatmap_edges = (np.asarray(xe, dtype=np.float64).copy(),
                                np.asarray(ye, dtype=np.float64).copy())
        self._plot.addItem(image)
        self._heatmap_image = image
        # `colorMapMenu=False`: pyqtgraph's own right-click menu recolours the bar and the
        # image behind `StyleSettings.heatmap_colormap`'s back, so the picture would
        # disagree with the combo, undo, templates and the saved project -- and the next
        # style change would silently put the style's map back. The Style combo is the
        # one door.
        bar = pg.ColorBarItem(values=(0.0, cmax), colorMap=self._heatmap_colormap(),
                              label="Event count", interactive=False, width=15,
                              colorMapMenu=False)
        self._heatmap_bar = bar
        bar.setImageItem(image, insert_in=self._plot)
        self._style_heatmap_bar()

        self._plot.setLogMode(x=log_x, y=log_y)
        self._plot.setLabel("bottom", self._axis_label_for("bottom", xcol, log_x),
                            **self._label_style())
        self._plot.setLabel("left", self._axis_label_for("left", ycol, log_y),
                            **self._label_style())
        self._apply_title(f"Event counts {ycol} vs {xcol} — {self._legend_name(src)}",
                          warn=self._warn_headline(self._stale_headline([xcol, ycol],
                                                                        [src]),
                                                   verdict.headline))
        self._apply_background()
        self._apply_fonts()
        # Ranged on the image by hand, as the KDE page is -- autoRange sees no
        # PlotDataItem -- and then the user's pin wins, as on every page.
        self._plot.vb.setRange(xRange=(float(xe_v[0]), float(xe_v[-1])),
                               yRange=(float(ye_v[0]), float(ye_v[-1])), padding=0)
        # What Reset view repeats: autoRange() would pad the image's bounds instead.
        self._kde_view_frame = (float(xe_v[0]), float(xe_v[-1]),
                                float(ye_v[0]), float(ye_v[-1]))
        self._apply_stored_ranges(log_x, log_y)
        self._set_details("\n".join(details))

    def _legend_clusters(self, labels: np.ndarray) -> None:
        """One legend swatch per cluster present, in cluster order (dropped last).

        A cluster-coloured cloud is an overlay of populations in one colour channel,
        so it needs naming exactly as a multi-source overlay does. The swatches are
        zero-point scatter items: they exist to be legend samples, never to draw."""
        for cid, n in category_counts(labels):
            # legend_swatch's body, not a copy of it: this used to hand-write the
            # identical ScatterPlotItem call, so a pen or alpha added to the shared
            # helper would have changed the report window's swatches and silently
            # left this panel's alone -- two pictures drifting apart one colour at
            # a time, which is the failure `categorical` exists to prevent.
            swatch = legend_swatch(cid, size=self._style.point_size)
            swatch._presentation_role = "category"
            name = "dropped (non-finite features)" if cid < 0 else f"cluster {cid}"
            source = next((getattr(item, "_presentation_cluster_source", "")
                           for item in self._scatter_items
                           if hasattr(item, "_presentation_cluster_labels")), "")
            self._tag_series(swatch, "data", f"{source}:cluster:{int(cid)}", name,
                             identity_kind="category", colour=QtGui.QColor(category_colour(cid)).name())
            self._legend_add(swatch, self._legend_entry(name, n),
                             kind="cluster", key=str(name))

    @staticmethod
    def _evr_pct(ratio) -> str:
        """One explained-variance percentage, or the refusal that replaces it.

        NEVER a bare format spec on a value that can be non-finite -- the same rule
        the scatter's R² already follows two hundred lines above ("R² undefined (y
        does not vary)") and the PSD panel's ``_fmt_q``. ``explained_variance_ratio_``
        is ``explained_variance_ / total_var``; with every selected feature constant
        that denominator is zero and sklearn hands back ``nan``, which this window
        printed verbatim into its title, both axis labels, Details and an exported
        PNG. :func:`populations.pca` now refuses that input outright, so this is the
        second line of the same defence rather than the only one -- and the one that
        holds if a future solver produces a non-finite ratio for a reason nobody has
        thought of yet.
        """
        r = float(ratio)
        return f"{r * 100:.1f}%" if np.isfinite(r) else "undefined"

    @staticmethod
    def _pca_dof_tag(res: populations.PCAResult) -> str:
        """"exact by construction", when the cumulative percentage cannot be anything else.

        THE SAME DOF HONESTY THE SCATTER'S R² ALREADY GETS
        (:func:`nanoscope.analysis.stats.r2_dof_tag`). Centred data of ``n`` rows has
        rank at most ``n - 1``, and a projection onto ``p`` features has rank at most
        ``p``; when the number of components shown reaches that rank the components
        span the whole space and the cumulative share is 100.0% for ANY data
        whatsoever. Six pure-noise tables at n = 2 and n = 3 all reported "cumulative
        100.0%" with nothing beside it, in the window whose scatter, on the same two
        rows, prints "R² = 1.000 (exact by construction)".
        """
        rank = int(res.meta.get("max_rank", 0) or 0)
        shown = min(int(res.explained_variance_ratio.size), 2)
        return "exact by construction" if rank and shown >= rank else ""

    @classmethod
    def _explained_variance_text(cls, res: populations.PCAResult) -> str:
        evr = res.explained_variance_ratio
        parts = [f"PC{i + 1} {cls._evr_pct(r)}" for i, r in enumerate(evr[:2])]
        cum = float(np.sum(evr[:2]))
        tag = cls._pca_dof_tag(res)
        return (f"Explained variance: {', '.join(parts)}  "
                f"(cumulative {cls._evr_pct(cum)}{'; ' + tag if tag else ''})")

    @staticmethod
    def _pca_dof_caveat(res: populations.PCAResult) -> str:
        """The full sentence behind :meth:`_pca_dof_tag`, for Details. ``""`` when moot."""
        rank = int(res.meta.get("max_rank", 0) or 0)
        shown = min(int(res.explained_variance_ratio.size), 2)
        if not rank or shown < rank:
            return ""
        n, p = int(res.n_kept), len(res.columns)
        why = (f"{n} rows centred on their own mean span at most {n - 1} dimension"
               f"{'' if n - 1 == 1 else 's'}" if (n - 1) <= p
               else f"{p} features span at most {p} dimensions")
        return (f"CUMULATIVE 100% HERE IS EXACT BY CONSTRUCTION, not a finding: {why}, "
                f"and {shown} component{'' if shown == 1 else 's'} already cover"
                f"{'s' if shown == 1 else ''} that space — so ANY {n} rows whatsoever, "
                f"noise included, would report the same cumulative share. Read the "
                f"per-component split as arithmetic, not as evidence that these "
                f"features are collinear.")

    # -- clustering / PCA runs -------------------------------------------------------------------
    def run_clustering(self) -> populations.ClusterResult | None:
        """Cluster the current rows over the selected features and recolour the view."""
        df = self._current_frame()
        if df is None or df.shape[0] == 0:
            self._message("No rows to cluster.")
            return None
        features = self.feature_selection()
        if len(features) < 2:
            self._set_details("Clustering needs at least 2 feature columns selected.")
            return None
        method = _METHOD_IDS[self._method_combo.currentText()]
        if method == "gmm" and self._k_mode.currentText() == _K_AUTO:
            k: int | None = None
        else:
            k = self._k_spin.value()
        try:
            res = populations.cluster(df, features, method=method, k=k)
        except (ValueError, RuntimeError) as exc:
            self._set_details(f"Clustering failed: {exc}")
            return None

        self._cluster_result = res
        # Make the grouping visible on EVERY view that can carry it: the scatter (and
        # PCA) switch to cluster colouring, and so does the histogram — a clustering
        # that could be seen in one view and not in another was the missing half of
        # this workflow. The histogram split stays off while several sources are
        # overlaid, where colour already means source.
        self._updating = True
        self._colour_by.setCurrentText(_COLOUR_CLUSTER)
        if self._hist_colour_by.isEnabled():
            self._hist_colour_by.setCurrentText(_COLOUR_CLUSTER)
        self._updating = False
        self._armed_redraw()   # "Run clustering" is a run gesture; it clears the gate too
        self._set_details(self._cluster_summary())
        # The result itself is durable v11 state even when the controls already had
        # these values (for example, Run clustering pressed a second time).
        self.workspaceChanged.emit()
        return res

    def run_pca(self) -> populations.PCAResult | None:
        """Project the selected features to 2 components and show the scatter."""
        df = self._current_frame()
        if df is None or df.shape[0] == 0:
            self._message("No rows for PCA.")
            return None
        features = self.feature_selection()
        if len(features) < 2:
            self._set_details("PCA needs at least 2 feature columns selected.")
            return None
        try:
            # Project v11 records the seed with the PCA snapshot.  Use an explicit
            # integer here so a deliberate rerun has a reproducible recipe instead of
            # persisting ``None`` (which delegates randomness to library defaults).
            res = populations.pca(df, features, n_components=2, random_state=0)
        except (ValueError, RuntimeError) as exc:
            self._set_details(f"PCA failed: {exc}")
            return None

        self._pca_result = res
        self._updating = True
        self._plot_type_combo.setCurrentText(PCA)
        self._pages.setCurrentIndex(_PLOT_TYPES.index(PCA))
        self._updating = False
        self._armed_redraw()   # "Run PCA" is a run gesture; it clears the gate too
        self.workspaceChanged.emit()
        return res

    def _cluster_amplitude_warnings(self) -> tuple[str, ...]:
        res = self._cluster_result
        frame = self._current_frame()
        if res is None or "amplitude_pA" not in res.columns or frame is None:
            return ()
        primary = self.primary_source()
        return (amplitude_comparability([("Clustering features", frame)])
                + tuple(self._column_notes(["amplitude_pA"], [primary])))

    def _cluster_summary(self) -> str:
        res = self._cluster_result
        if res is None:
            return ""
        auto = res.meta.get("requested_k") is None and res.method == "gmm"
        head = (f"Clustering: {res.method}, k = {res.chosen_k}"
                f"{' (auto by BIC)' if auto else ''}, n = {res.n_kept} kept")
        dropped = int((res.labels == populations.DROPPED_LABEL).sum())
        if dropped:
            head += f", {dropped} dropped (non-finite features)"
        sizes = "sizes: " + ", ".join(f"{cid}:{n}" for cid, n in sorted(res.sizes.items()))
        means = "means (original units):\n" + res.means.to_string(float_format=lambda x: f"{x:.4g}")
        parts = [head, sizes, means]
        # The caveats FIRST among the trailing lines, and never silently: this readout used
        # to print "sizes: 0:189, 1:1" with no comment that a cluster of one event is not a
        # population. See populations.ClusterResult.warnings.
        for note in getattr(res, "warnings", ()):
            parts.append("! " + str(note))
        parts.extend("! " + note for note in self._cluster_amplitude_warnings())
        return "\n".join(parts)

    # -- export: the numbers behind the picture --------------------------------------------------
    def _export_series_labels(self) -> tuple[list[str], list[str]]:
        """``(series names, fit names)`` for the current picture, in DRAW order.

        Taken from ``_legend_offered``, which the draw fills as it adds each item and
        which holds the FULL label -- not the elided text that goes onto the legend
        (``_LEGEND_LABEL_CHARS`` = 34 turns a recording filename into
        "500 pM Apo ferri…-500 mV.csv", which is not a name a data file should carry),
        and not nothing when the user has unticked *Show legend*: ``_legend_add``
        records the offer before either gate. So the exported ``series`` column is what
        the figure calls that series, whether or not the figure is currently showing it.

        The fit entries are separated by their entry KIND rather than by position,
        because a fit that refused (too few events, an empty range) leaves a gap in
        the fit sequence but not in the series one.

        A ``cluster`` ENTRY IS A SERIES ON ONE PAGE AND A SWATCH ON TWO, and the
        difference is not cosmetic -- getting it wrong writes a mislabelled file.
        ``_draw_histogram``'s cluster split really does draw one histogram per cluster
        and offers each bar set under ``kind="cluster"``, so there the cluster names
        ARE the series names. ``_draw_scatter`` and ``_draw_pca`` draw ONE
        ScatterPlotItem with a brush per point and offer ``_legend_clusters``' swatches
        -- items that exist only to sit in the legend and never to draw. Treating those
        as series labelled 200 points of a two-cluster scatter "cluster 0 (n = 100)"
        (measured). On those two pages the series is the SOURCE and the cluster is a
        per-point column; see :meth:`_cluster_of_drawn_points`.

        FALLBACK: the KDE draw offers legend entries only when several sources are
        overlaid, and a single-source scatter offers none at all, so nothing is
        recorded. The series is then the drawn source, which is what the checked-source
        list already says.
        """
        series: list[str] = []
        fits: list[str] = []
        applied = getattr(self, "_figure_applied_config", None)
        swatches_only = (applied.identity.plot_type if applied else self._plot_type()) in (SCATTER, PCA)
        for entry_id, label in self._legend_offered.items():
            if entry_id.startswith("fit:"):
                fits.append(str(label))
            elif entry_id.startswith("cluster:") and swatches_only:
                continue                       # a legend swatch, not a drawn series
            elif entry_id.startswith("kde:"):
                continue                       # the KDE key row; its curves are named below
            else:
                series.append(str(label))
        if not series:
            # The figure's name for each source, not the raw override -- the one
            # resolver every writer that names a source on or beside the figure reads.
            series = list(self._committed_legend_names().values())
        return series, fits

    def _cluster_of_drawn_points(self) -> np.ndarray | None:
        """The cluster id behind every drawn marker, in draw order — or None.

        READ OFF THE BRUSHES THE DRAW SET, not recomputed from the frame. The draw
        derives them with ``self._cluster_result.labels.reindex(index)`` over an index
        it builds and discards (the finite/positive mask), so the only ways to get them
        back are to re-derive that mask -- a second copy of the draw's rule, free to
        drift from it silently -- or to read the picture. The colour IS what the figure
        shows, and ``categorical.category_colour`` is the same function that put it
        there, so the mapping cannot disagree with the legend beside it.

        An unrecognised colour becomes NaN rather than a guess: a fabricated cluster id
        in a CSV is worse than an admitted gap.

        THE GATE IS THE OFFERED SWATCHES, NOT ``self._cluster_result``. A clustering
        stays on the window after the user switches *Colour by* back to None or ticks a
        second source, and both of those repaint the markers from the SOURCE palette --
        which is Okabe-Ito, the same palette ``category_colour`` draws from, so a
        colour lookup would match and stamp every row with a cluster the picture is not
        showing. ``_legend_clusters`` is the only producer of ``cluster:`` entries on
        these two pages and it runs only when the markers really are cluster-coloured,
        so its offers are the exact condition.
        """
        if self._cluster_result is None or not self._scatter_items:
            return None
        if not any(k.startswith("cluster:") for k in self._legend_offered):
            return None
        labels = np.asarray(self._cluster_result.labels, dtype=int)
        by_rgb: dict[int, int] = {}
        for cid in set(labels.tolist()) | {populations.DROPPED_LABEL}:
            # ``rgb()`` drops the alpha, which ``category_brushes`` is free to vary.
            by_rgb[QtGui.QColor(category_colour(int(cid))).rgb()] = int(cid)
        out: list[float] = []
        for item in self._scatter_items:
            captured = getattr(item, "_presentation_cluster_labels", None)
            if captured is not None:
                out.extend(float(cid) for cid in captured)
                continue
            brushes = item.data["brush"]
            for brush in brushes:
                cid = by_rgb.get(QtGui.QColor(brush.color()).rgb())
                out.append(np.nan if cid is None else float(cid))
        return np.asarray(out, dtype=float)

    def _plot_data(self) -> pd.DataFrame:
        """The numbers behind the plot on screen, as the frame :meth:`export_data` writes.

        Split out from the writer so a test -- and a future *Copy data* -- can ask for
        the table without a file dialog anywhere near it.

        WHAT EACH PLOT TYPE CONTRIBUTES, and why that is enough to redraw it:

        * **Histogram** -- one ``bins`` row per bar per series, carrying
          ``bin_left``/``bin_right`` and the bar's DRAWN height, so a reader re-plots
          the exact bars rather than re-binning and hoping. The height is in whatever
          the *y* combo says (count / density / probability), and the y column is NAMED
          that, because a density and a count are different numbers from the same bars.
          A fitted curve adds ``fit`` rows sampled where it was drawn -- already scaled
          to the same y quantity, so the curve and the bars are directly comparable.
        * **Scatter** -- one ``points`` row per plotted marker, which is the figure
          exactly: the rows the draw dropped (non-finite, or non-positive under a log
          axis) are absent here for the same reason they are absent from the picture,
          and Details says how many they were. A best-fit line adds its ``fit`` samples.
        * **KDE** -- the ``grid`` the contours were cut from (128x128 densities per
          source, in data coordinates) plus one ``level`` row per contour. Contour
          VERTICES would reproduce only the levels the user happened to choose; the
          grid re-contours at any level, in any tool, and is what the density actually
          is.
        * **Histogram with a KDE overlay** -- the bars as above, plus ``kde`` rows: each
          series' curve sampled where it was drawn, in the bars' own y quantity, named
          for the series it smooths (never taking a fit's name).
        * **Event-count heatmap** -- one ``cell`` row per bin, empty bins included: its x
          and y edges in data units and its count, read off the drawn image.
        * **PCA** -- one ``points`` row per score, plus a ``loading`` row per feature
          (its weight on PC1 and PC2). Without the loadings a PCA scatter is a cloud
          nobody can interpret: the axes are combinations of columns, and the CSV would
          not say which.

        Empty (no rows) when nothing is drawn -- a message, a refusal, an empty frame.
        """
        from ..plot.figures import plot_source_data
        from types import SimpleNamespace

        applied = getattr(self, "_figure_applied_config", None) or self.config()
        identity = applied.identity
        kind = identity.plot_type
        log_x, log_y = applied.log_state
        if kind == HISTOGRAM:
            # The y axis of a histogram is not a column of the events table; it is what
            # the normalization combo made of the counts, and ``_y_label`` is the same
            # function that names the QUANTITY on the plot.
            #
            # `_y_label`, NOT `_y_axis_label`, AND THAT IS DELIBERATE (S6-03). The picture
            # needs its unit spelled out because its x TICKS are re-scaled for display; a
            # CSV has no display scale, and its x column is the schema's own unit-bearing
            # name -- `dwell_s`, `depth_pA` -- so a column called `density` beside `dwell_s`
            # is per-second by construction. Writing `density (1/s)` here would rename a
            # user-facing column for no gain, break
            # `test_the_height_column_is_named_for_the_quantity_it_holds`, and disturb
            # `plot_source_data`'s `density_name in (x_name, y_name)` guard, whose own
            # comment names this exact histogram case. Keeping BOTH the values and this name
            # fixed is also what lets the whole export path stay untouched by S6-03:
            # measured, the exported bars still integrate to 1.0000000000000053 in stored
            # units, in every time unit, before and after.
            x_name = identity.x_column or "x"
            y_name = {_NORM_COUNT: "count", _NORM_DENSITY: "density",
                      _NORM_PROB: "probability"}[applied.histogram.normalization]
        elif kind == PCA:
            x_name, y_name = "PC1", "PC2"
        else:
            x_name, y_name = identity.x_column or "x", identity.y_column or "y"
        if x_name == y_name:
            # A scatter of a column against itself is legal and draws a diagonal line;
            # two identical column names would silently collapse into one in the frame.
            y_name = f"{y_name}_y"
        series, fits = self._export_series_labels()
        # Confidence bounds are decorations, not additional independently fitted
        # populations. Exclude them so a visibility toggle cannot invent fit rows.
        numerical_items = SimpleNamespace(items=[item for item in self._plot.items
                                                 if not getattr(item, "_presentation_band", False)])
        # A KDE curve is named for the SERIES whose bars it smooths -- by the index the
        # draw stamped on it, so a series the KDE skipped (too few values) cannot shift
        # the next curve onto the wrong name.
        kdes = [series[i] if isinstance(i, int) and 0 <= i < len(series) else ""
                for i in (getattr(node, "_kde_series", None) for node in numerical_items.items
                          if getattr(node, "_plot_data_part", None) == "kde")]
        df = plot_source_data(numerical_items, x_name=x_name, y_name=y_name,
                              log_x=log_x, log_y=log_y,
                              series_labels=series, fit_labels=fits, kde_labels=kdes)
        if kind in (SCATTER, PCA):
            # THE COLOUR CHANNEL IS DATA HERE. On these two pages a clustering is drawn
            # as one item with a brush per point, so without this column the CSV would
            # carry the cloud and lose the very partition the picture is about.
            df = self._with_cluster_column(df)
        if kind == PCA and len(df) and self._pca_result is not None:
            df = pd.concat([df, self._pca_loading_rows(x_name, y_name)],
                           ignore_index=True)
        if kind != PCA and len(df):
            from .quantity_axis import export_display_columns
            bases = _plot_quantity_bases(applied)
            xq, yq = bases["bottom"].quantity, bases["left"].quantity
            columns = {x_name: xq, y_name: yq}
            if kind == HISTOGRAM:
                columns.update(bin_left=xq, bin_right=xq)
            elif kind == KDE and "density" in df:
                from ..core.quantity_display import density_quantity
                if all(q.known and q.quantity not in ("unknown", "sample", "count", "categorical")
                       for q in (xq, yq)):
                    columns["density"] = density_quantity([xq, yq],
                        measures=["log10" if log_x else "linear", "log10" if log_y else "linear"])
            df = export_display_columns(df, columns)
        if hasattr(self, "_nonlinear_fitter"):
            extra = self._nonlinear_fitter.export_rows()
            if extra:
                df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True, sort=False)
        return df

    def _with_cluster_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``cluster`` column to the ``points`` rows of *df*, when there is one.

        Nullable ``Int64``, so an uncoloured row is ``<NA>`` in the CSV and never a 0
        that would read as "cluster 0". Added only when the count matches the drawn
        points exactly -- a partial alignment would put the right ids against the wrong
        markers, which is the one outcome worse than no column at all.
        """
        if not len(df):
            return df
        ids = self._cluster_of_drawn_points()
        if ids is None:
            return df
        mask = (df["part"] == "points").to_numpy()
        if int(mask.sum()) != ids.size:
            return df
        column = np.full(len(df), np.nan)
        column[mask] = ids
        return df.assign(cluster=pd.Series(column, index=df.index).astype("Int64"))

    def _pca_loading_rows(self, x_name: str, y_name: str) -> pd.DataFrame:
        """One row per feature: its loading on PC1 and on PC2.

        ``PCAResult.components`` is ``(n_components, n_features)``, so a component is a
        ROW of it and a feature is a column -- transposed here rather than in the
        reader's head. A one-component result (two features, one of them constant) has
        no PC2 to report, and gets NaN there rather than a fabricated zero.
        """
        res = self._pca_result
        comps = np.asarray(res.components, dtype=float)
        pc1 = comps[0] if comps.shape[0] > 0 else np.full(len(res.columns), np.nan)
        pc2 = comps[1] if comps.shape[0] > 1 else np.full(len(res.columns), np.nan)
        return pd.DataFrame({
            "series": [str(c) for c in res.columns],
            "part": "loading",
            x_name: pc1,
            y_name: pc2,
        })
