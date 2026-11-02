"""The standalone event-plots window: the density map + the event ensemble.

Opened from ``Analysis ▸ Event waveforms…``, this is the GUI
face of :mod:`nanoscope.analysis.event_shape` -- the two figures every
translocation paper leads with:

* **Density** -- the 2-D dwell-vs-blockage heatmap, an X metric against a Y metric
  binned into a colour-mapped image with MARGINAL histograms along the top (X) and
  right (Y). X and Y each choose among ``dwell_s`` (default X, log by default),
  ``depth_pA`` (default Y), ``frac_depth`` and the on-demand ECD / charge deficit.
* **Ensemble** -- every accepted event resampled onto a common axis and overlaid
  (faint), with the bold mean and median waveform drawn on top. Real-time alignment is
  at each event's start, centre or extremum (its deepest sample, by the table's own
  polarity); an optional band (mean +/- 1 SD, or the median's IQR) shades the spread;
  and the events can be narrowed to one of the curation subsets the Event gallery
  offers (always within the accepted events, which is what exports honour).

Design mirrors :class:`~nanoscope.gui.voltage_dep_panel.VDepWindow` and
:class:`~nanoscope.gui.plots_panel.PlotsWindow`: a single, lazily-built top-level
window owned by :class:`~nanoscope.gui.main_window.MainWindow`, rendered with
pyqtgraph so it inherits the app's dark theme. Like PlotsWindow it never reaches
into another panel -- the owner snapshots the ACTIVE detection's EventSet and
pushes it via :meth:`set_source`; the window pulls that detection's calibrated
signal straight from the :class:`~nanoscope.core.session.Session` it was handed
and recomputes the ECD and the ensemble traces ON DEMAND (nothing is stored back
onto the frozen EventSet). A :attr:`refreshRequested` signal lets the owner
re-snapshot on demand.

The ECD, density and ensemble are computed purely in
:mod:`nanoscope.analysis.event_shape`; this module is only the interactive shell
around them.

The title and both axis labels are typed once for both views, so they go back to automatic
(``gui.quantity_reset``) when the heatmap's X or Y, the view shown, or Normalize time changes.
The change goes through the control's own signal: a pick and a setter do it, a project restore
and a template (which set ``_updating`` or block the signal) do not. A range pinned in Axis
properties belongs to one view's axis (``map.<side>``, ``ensemble.<side>``), and goes with the
words only when that axis's quantity moved: the heatmap's X takes the map's x pin, its Y the map's
y pin, Normalize time the ensemble's x pin; the view shown takes none. Log x and Log y take the
pin of their axis of the map and not the words: the map draws log10 values on a linear axis, so a
range pinned over the other state frames nothing. A Log scale changed in Axis properties takes no
pin, since that dialog commits the range typed beside it (:meth:`EventPlotWindow._log_from_figure`).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..analysis import event_shape
from ..analysis.curation_view import (FILTER_ACCEPTED, FILTER_ALL, FILTER_REJECTED,
                                      curation_progress, filter_mask, filter_options)
from ..analysis.events import EventSet
from ..core.session import Session
from ..core.sources import amplitude_comparability, amplitude_units_label
from ..plot.style import OKABE_ITO
from .analysis_catalog import ANALYSIS_LABELS
from .export_paths import ask_export_path
from .figure_text import (SHARED_LABELS_REASON, TextTarget, appearance_text_host,
                          apply_text_style, figure_text_for, plain_text)
from .plot_appearance import (PlotAppearance, PlotAppearanceWidget, apply_plot_appearance,
                              apply_tick_family, reframe_axis_pins)
from .plot_interaction import (axis_name, image_value, install_plot_interaction,
                               reset_plot_history, view_button, window_readout)
from .plot_templates import WindowTemplateControls
from .plot_workflow import (WarningSummary, bounded_controls, control_group,
                            ensure_item_legend, install_plot_workflow)
from .plot_export import (DATA_FILTERS, IMAGE_FORMATS, choose_live_plot_export,
                          data_write_failure_text, require_csv_row_budget,
                          write_live_plot_image, write_plot_csv)

__all__ = ["EventPlotWindow"]

# View labels (also the QTabWidget page order).
DENSITY = "Density"  # stable workspace key; the visible label is deliberately more precise
DENSITY_LABEL = "Event-count heatmap"
ENSEMBLE = "Ensemble"
#: The keys each view's Axis properties pins are kept under (``PlotAppearance.for_view``). A
#: bare pin, saved before pins were kept per view, is the map's: the view the window opens on.
PIN_MAP, PIN_ENSEMBLE = "map", "ensemble"

# Metric ids (X/Y choices for the density map). ``ecd`` is computed on demand.
_M_DWELL = "dwell_s"
_M_DEPTH = "depth_pA"
_M_FRAC = "frac_depth"
_M_ECD = "ecd"
_M_AMPLITUDE = "amplitude_pA"
_METRICS: tuple[tuple[str, str], ...] = (
    (_M_DWELL, "Dwell (s)"),
    (_M_DEPTH, "Median depth"),
    (_M_FRAC, "dI/I₀"),
    (_M_ECD, "ECD (charge deficit)"),
    (_M_AMPLITUDE, "Selected amplitude (Curate)"),
)

# Long-form keeps every visible sample and its event identity easy to understand,
# but millions of Python dict rows can exhaust RAM before pandas even writes the CSV.
# Refuse before constructing them; the user can lower *Points* or curate a smaller
# accepted set and gets a precise, non-lossy explanation from the export dialog.
MAX_ENSEMBLE_CSV_ROWS = 250_000

#: What the status line calls a metric after a change (quantity_reset): the name without its unit.
_METRIC_PHRASE = {_M_DWELL: "dwell", _M_DEPTH: "median depth", _M_FRAC: "dI/I₀",
                  _M_ECD: "ECD", _M_AMPLITUDE: "selected amplitude"}

_COLORMAPS = ("viridis", "magma", "plasma", "inferno", "cividis")

# Ensemble variability bands: (visible text, event_shape.ensemble_band kind). "" = none,
# the default, so a saved or reopened ensemble looks exactly as it did before bands existed.
_BAND_NONE = ""
_BANDS: tuple[tuple[str, str], ...] = (
    ("None", _BAND_NONE),
    ("Mean ± 1 SD", "sd"),
    ("Median IQR (25–75 %)", "iqr"),
)
# The ensemble subset default: every accepted event, which is what the window drew before
# the subset existed. The other subsets are analysis.curation_view's filters -- the Event
# gallery's, by the same names -- intersected with the accepted rows. "All" and "Rejected
# only" are not offered: ``accepted`` is the column exports honour (events.SCHEMA), and a
# published ensemble of rejected events is a picture of what curation threw away.
_SUBSET_DEFAULT = FILTER_ACCEPTED
_SUBSET_EXCLUDED = (FILTER_ALL, FILTER_REJECTED)

_MSG_HEX = "#8b9199"
_EVENT_HEX = OKABE_ITO[5]        # sky blue -- the faint per-event overlay lines
_MEAN_HEX = OKABE_ITO[1]        # vermillion -- the ensemble mean
_MEDIAN_HEX = OKABE_ITO[2]      # bluish green -- the ensemble median
_BAR_HEX = OKABE_ITO[0]         # blue -- the marginal histograms


class EventPlotWindow(QtWidgets.QMainWindow):
    """Density heatmap + event ensemble for the active detection's accepted events.

    Fed by the owner via :meth:`set_source` (the active detection's ``(doc_id,
    channel, EventSet)``); it pulls the calibrated signal from the Session and
    recomputes everything on demand. The owner builds it once, pushes data, and
    wires :attr:`refreshRequested` to re-snapshot.

    Emits:
        refreshRequested(): the user clicked *Refresh*; the owner should re-snapshot
            the active detection and call :meth:`set_source` again.
    """

    refreshRequested = Signal()
    #: A recipe control changed.  The owner persists it in the v11 analytical
    #: workspace; this window remains deliberately ignorant of project files.
    workspaceChanged = Signal()

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle(f"NanoScope — {ANALYSIS_LABELS['event_waveforms']}")
        self.resize(1040, 760)

        # Pushed-in source state.
        self._doc_id: str | None = None
        self._channel: int | None = None
        self._eventset: EventSet | None = None
        self._signal: np.ndarray | None = None
        self._fs: float | None = None
        self._units: str = "pA"

        # Live results (kept for tests + save).
        self._density_counts: np.ndarray | None = None
        # Keep the bin edges with the counts.  A density matrix without its physical
        # coordinates cannot be rendered faithfully after a project/recovery restore.
        self._density_x_edges: np.ndarray | None = None
        self._density_y_edges: np.ndarray | None = None
        self._density_image: pg.ImageItem | None = None
        self._top_marginal_item: pg.BarGraphItem | None = None
        self._right_marginal_item: pg.BarGraphItem | None = None
        self._ensemble: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        # The variability band beside the ensemble: (lower, upper, events per point), a
        # reduction of the matrix above that is saved and exported with it.
        self._ensemble_band: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._band_items: tuple = ()
        # (polarity, where it came from) of the last extremum-aligned ensemble, for Details.
        self._ensemble_alignment: tuple[int, str] | None = None
        # How many events the last ensemble was asked to draw (the subset, before the
        # edge skip), so a restored snapshot reports its skip count against the same base.
        self._ensemble_input_count: int | None = None
        # Of those, how many an extremum alignment left out because their own span holds
        # no finite sample to find a deepest point in: skipped, but NOT for running off
        # the trace, so the title and the CSV count them apart from the edge skips.
        self._ensemble_unalignable: int = 0
        self._events_item: pg.PlotDataItem | None = None
        self._mean_item: pg.PlotDataItem | None = None
        self._median_item: pg.PlotDataItem | None = None
        # Automatic titles describe the numerical result currently on screen, not a
        # newly calculated result. Keep them when editing labels or a saved snapshot.
        self._density_auto_title = ""
        self._amplitude_warnings: tuple[str, ...] = ()
        self._amplitude_snapshot_units = "units unstated"
        self._amplitude_frozen_snapshot = False
        self._ensemble_auto_title = ""
        self._appearance = PlotAppearance()
        # What each figure on screen shows, for the reset rule (quantity_reset): the typed
        # words belong to it, so a change is measured against THIS and not against the
        # controls. None: that figure shows no quantity yet (empty, or not drawn).
        self._density_shown: tuple[str, str] | None = None
        #: The map's (Log x, Log y) on screen: with one ticked it draws log10 values on a linear
        #: axis, so a range pinned over the other state frames nothing.
        self._density_logs: tuple[bool, bool] | None = None
        self._normalize_shown: bool | None = None
        self._view_shown = DENSITY

        self._updating = False
        self._figure_log = False        # up while Axis properties changes Log scale
        self._build_ui()
        self._redraw()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._build_density_page(), DENSITY_LABEL)
        self._tabs.addTab(self._build_ensemble_page(), ENSEMBLE)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        # ⌘Z on the page, not on these controls: an edit made in the figure is undone from
        # the keyboard whatever the Section shows (FIG-4); a page behind another Explorer
        # tab is hidden, so it does not answer.
        self._appearance_widget = PlotAppearanceWidget(
            self._appearance, include_colors=True, include_markers=False,
            multi_view_labels=True, shortcut_parent=self)
        self._appearance_widget.changed.connect(self._on_appearance_changed)
        self._appearance_section = bounded_controls(self._appearance_widget)
        layout.addWidget(self._appearance_section)
        self._data_section = control_group(
            "Data", "Uses the active detection’s accepted events. Analyze / refresh updates "
            "this snapshot. The Event-count heatmap bins actual event counts; it is not a "
            "smoothed kernel-density estimate (KDE). Ensemble displays aligned event traces.")
        layout.addWidget(self._data_section)
        layout.addWidget(self._tabs, 1)
        self.setCentralWidget(central)

        # Refresh lives in the toolbar so both views share it.
        bar = QtWidgets.QToolBar()
        bar.setParent(self)
        self._legacy_toolbar = bar
        bar.setMovable(False)
        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip("Re-snapshot the active detection's accepted events.")
        self._refresh_btn.clicked.connect(self.refreshRequested.emit)
        bar.addWidget(self._refresh_btn)
        appearance = bar.addAction("Typography and lines")
        self._appearance_action = appearance
        appearance.setCheckable(True)
        appearance.toggled.connect(self._show_appearance)
        appearance.setToolTip("Style the existing plots without recalculating events.")
        self._workflow = install_plot_workflow(
            self, sections={"Data": (self._data_section,),
                            "Analysis": (self._density_analysis_controls, self._ensemble_analysis_controls),
                            "Appearance": (self._appearance_section, self._density_palette),
                            "Export": (self._density_export_controls, self._ensemble_export_controls)},
            actions=(("Analyze / refresh", self.refreshRequested.emit),),
            exports=(("Export current figure…", lambda: self.save_image()),
                     ("Export data (CSV)…", lambda: self.export_data())))
        self._workflow.adopt_reset_figure(
            self._appearance_widget, keeps="the Event-count heatmap colormap")
        bar.hide()  # legacy programmatic toggle remains available; workflow owns visible chrome
        # One template for both views: the heatmap's metrics, bins and colormap with the
        # ensemble's alignment, and the shared appearance -- never the bound detection.
        self._templates = WindowTemplateControls(
            "event-waveforms", capture=self.template_settings,
            recall=self.apply_template_settings,
            note="A template keeps both views' settings (heatmap metrics, log axes, bins, "
                 "colormap; ensemble normalisation, alignment (extremum included), points, "
                 "padding, Events subset and band), the view shown and the appearance. It never "
                 "keeps the detection or its events. Recalling one redraws a view from the "
                 "events already bound only if its settings differ, as changing those controls "
                 "does; a changed colormap only recolours, and a changed band only re-shades.")
        layout.insertWidget(layout.indexOf(self._data_section) + 1, self._templates)
        self._workflow.add_section_widget("Export", self._templates)
        self._install_plot_interaction()

    def _install_plot_interaction(self) -> None:
        """View only (plot_interaction): Reset view, zoom history, app menu, pointer readout.

        The density map draws LOG10 VALUES ON A LINEAR AXIS when Log x / Log y is ticked
        (``_to_plot``; its axis label says "(log10)"), so the readout is told which axes hold
        logs and shows the real value, named without the "(log10)". The name is the map's
        DRAWN axis title, so an Appearance label typed over the metric's name is the name
        the readout uses too (the marginals have no titles of their own). The marginals are
        linked to the map and share its history, so their menu means the same view.
        """
        exports = (("Export current figure…", lambda: self.save_image()),
                   ("Export data (CSV)…", lambda: self.export_data()))
        self._readout = window_readout(self)

        def drawn_name(side: str, combo: QtWidgets.QComboBox):
            def name() -> str:
                text = axis_name(self._density_main.getAxis(side))
                if text.endswith("(log10)"):
                    text = text[:-len("(log10)")].rstrip()
                return text or self._metric_label(combo.currentData())
            return name

        x_name, y_name = drawn_name("bottom", self._x_combo), drawn_name("left", self._y_combo)
        x_log, y_log = self._logx.isChecked, self._logy.isChecked

        def density_count(vx: float, vy: float) -> str:
            value = image_value(self._density_image, vx, vy)
            return "" if value is None else f"Event count {value:.0f}"

        # In-figure editing (figure_text) on the map and the ensemble, one controller per
        # PlotItem: the marginals and the colour bar are NOT given controllers of their own;
        # their labels are extra targets of the map's (plan G4, the six-install risk). Log is
        # the map's own Log x / Log y (one store), which re-bins as a click on them does.
        def log_box(side: str) -> QtWidgets.QCheckBox:
            return self._logx if side == "bottom" else self._logy

        self._density_text_host = appearance_text_host(
            self._density_main, self._appearance_widget,
            axis_hooks={"log": self._log_from_figure,
                        "log_state": lambda side: log_box(side).isChecked()},
            extra_targets=self._density_label_targets, shared_labels=SHARED_LABELS_REASON,
            pin_view=PIN_MAP, primary_view=PIN_MAP)
        self._density_nav = install_plot_interaction(
            self._density_main, readout=self._readout, exports=exports, x_name=x_name,
            y_name=y_name, x_log=x_log, y_log=y_log, extra=density_count,
            text=self._density_text_host)
        controller = figure_text_for(self._density_main)
        for key, target in self._density_label_targets().items():
            controller._wrap(target.item, lambda _ev, key=key: key)
        install_plot_interaction(self._density_top, readout=self._readout, exports=exports,
                                 navigation=self._density_nav, x_name=x_name, x_log=x_log,
                                 y_name="count")
        install_plot_interaction(self._density_right, readout=self._readout, exports=exports,
                                 navigation=self._density_nav, y_name=y_name, y_log=y_log,
                                 x_name="count")
        self._ensemble_nav = install_plot_interaction(
            self._ensemble_plot, readout=self._readout, exports=exports,
            text=appearance_text_host(
                self._ensemble_plot, self._appearance_widget,
                axis_hooks={"log_reason": "The ensemble draws current against time on "
                                          "linear axes."},
                shared_labels=SHARED_LABELS_REASON, pin_view=PIN_ENSEMBLE, primary_view=PIN_MAP))
        navigations = (self._density_nav, self._ensemble_nav)
        self._workflow.add_view_control(view_button(
            lambda: self._density_nav if self.current_view() == DENSITY else self._ensemble_nav,
            navigations))

    def _density_label_targets(self) -> dict:
        """The map's own labels outside its PlotItem: the colour bar's and the marginals'."""
        return {key: TextTarget(axis, auto, noun="label")
                for key, (axis, auto) in self._density_labels().items()}

    def _density_labels(self) -> dict:
        # pyqtgraph draws a vertical colour bar's label on its LEFT axis; ``.axis`` is the
        # tick axis on the right, whose label is hidden, so editing it changed nothing seen.
        bar = self._colorbar
        label_axis = bar.getAxis("bottom" if bar.horizontal else "left")
        return {"colorbar.label": (label_axis, "Event count"),
                "marginal.left.label": (self._density_top.getAxis("left"), "count"),
                "marginal.bottom.label": (self._density_right.getAxis("bottom"), "count")}

    def label_text(self, key: str) -> str:
        """A colour-bar or marginal label as drawn (plain text)."""
        axis, _auto = self._density_labels()[key]
        return plain_text(axis.labelText) if axis.labelText else ""

    def _show_appearance(self, visible: bool) -> None:
        self._workflow.set_section("Appearance" if visible else "Data")
        # The Explorer can be shorter than this standalone window's preferred
        # size. Keep the axes visible when the optional controls are expanded.
        for widget, original in ((self._density_glw, 460), (self._ensemble_widget, 420)):
            widget.setMinimumHeight(240 if visible else original)

    def _on_appearance_changed(self, state: PlotAppearance) -> None:
        self._appearance = state
        if not self._updating:
            self._apply_appearance()
            self.workspaceChanged.emit()

    def set_appearance_state(self, mapping) -> None:
        self._appearance = PlotAppearance.from_mapping(mapping)
        self._appearance_widget.set_state(self._appearance)
        # A restored figure starts its appearance history over (design §4.2).
        self._appearance_widget.clear_history()
        if not self._updating:
            self._apply_appearance()

    def appearance_state(self) -> dict:
        return self._appearance.to_dict()

    def _frame(self, plot: pg.PlotItem) -> None:
        """A draw frames its data, then a range typed in Axis properties is the view again
        (FIG-2): the mark tells the next apply that this was a draw, so an appearance step
        alone still keeps a zoom made by hand."""
        plot.autoRange()
        reframe_axis_pins(plot)
        self._apply_appearance()

    def _apply_appearance(self) -> None:
        # Each view draws only the pins typed on it (figures2-1): a time range pinned on the
        # ensemble blanked the map, whose x is log10 dwell.
        apply_plot_appearance(self._density_main,
                              self._appearance.for_view(PIN_MAP, primary=PIN_MAP),
                              plot_widget=self._density_glw)
        # The map's element styles and axis pins are the map's: a marginal shares only the
        # figure-wide look, and its count label is styled below as its own element.
        marginal = replace(self._appearance, title=None, x_label=None, y_label=None,
                           text=None, axes=None, fit_label=None)
        for plot, item in ((self._density_top, self._top_marginal_item),
                           (self._density_right, self._right_marginal_item)):
            apply_plot_appearance(plot, marginal, data_items=(item,))
        for key, (axis, auto) in self._density_labels().items():
            original = getattr(axis, "_nanoscope_label_style", None)
            if original is None:
                original = axis._nanoscope_label_style = dict(axis.labelStyle)
            axis.labelStyle = dict(original)  # a style taken off must go, not linger
            apply_text_style(axis, self._appearance.text_style(key), auto,
                             self._appearance.font_family)
        apply_tick_family(self._colorbar.axis, self._appearance.font_family)
        # Mean and median are summaries, not fits: preserve their individual
        # colors and solid/dashed identities while sharing typography/line width.
        ensure_item_legend(self._ensemble_plot, ((self._events_item, "Events"),
                          (self._mean_item, "Mean"), (self._median_item, "Median")))
        self._appearance_widget.set_legend_available(self.current_view() == ENSEMBLE)
        self._appearance_widget.data_color.setToolTip(
            "Colour of the marginal count curves and individual ensemble traces. "
            "The heatmap uses its separate colormap control.")
        self._appearance_widget.fit_color.setEnabled(self.current_view() == ENSEMBLE)
        self._appearance_widget.fit_color.setToolTip(
            "Colour of the mean and median summary curves; these are not fitted models."
            if self.current_view() == ENSEMBLE else
            "The event-count heatmap has no fitted curve. Its colours use the colormap control.")
        apply_plot_appearance(
            self._ensemble_plot, self._appearance.for_view(PIN_ENSEMBLE, primary=PIN_MAP),
            plot_widget=self._ensemble_widget,
            data_items=(self._events_item,), fit_items=(self._mean_item, self._median_item))
        controller = figure_text_for(self._ensemble_plot)
        if controller is not None:
            controller.refresh()        # the legend may be new since the last draw
        if len(self._band_items) == 3:
            # The band follows the summary curve it surrounds (its colour may just have
            # changed); restyling a brush is presentation only.
            fill, lo_item, hi_item = self._band_items
            hex_color = self._band_anchor_hex()
            fill.setBrush(_alpha_brush(hex_color, 60))
            for edge in (lo_item, hi_item):
                edge.setPen(_alpha_pen(hex_color, 110, 1))

    def _build_density_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._density_analysis_controls = QtWidgets.QWidget()
        row1 = QtWidgets.QHBoxLayout(self._density_analysis_controls)
        row1.setContentsMargins(0, 0, 0, 0)
        row1.addWidget(QtWidgets.QLabel("X"))
        self._x_combo = QtWidgets.QComboBox()
        self._y_combo = QtWidgets.QComboBox()
        for mid, label in _METRICS:
            self._x_combo.addItem(label, mid)
            self._y_combo.addItem(label, mid)
        self._x_combo.setCurrentIndex(0)        # dwell_s
        self._y_combo.setCurrentIndex(1)        # depth_pA
        self._x_combo.setToolTip("The X-axis event metric for the density map.")
        self._y_combo.setToolTip("The Y-axis event metric for the density map.")
        row1.addWidget(self._x_combo)
        row1.addWidget(QtWidgets.QLabel("Y"))
        row1.addWidget(self._y_combo)

        self._logx = QtWidgets.QCheckBox("Log x")
        self._logx.setChecked(True)             # dwell reads on a log axis by convention
        self._logy = QtWidgets.QCheckBox("Log y")
        self._logx.setToolTip("Bin the X metric in log space (drops non-positive values).")
        self._logy.setToolTip("Bin the Y metric in log space (drops non-positive values).")
        row1.addWidget(self._logx)
        row1.addWidget(self._logy)

        row1.addWidget(QtWidgets.QLabel("Bins"))
        self._bins = QtWidgets.QSpinBox()
        self._bins.setRange(4, 400)
        self._bins.setValue(40)
        self._bins.setToolTip("Number of bins along each axis.")
        row1.addWidget(self._bins)

        self._density_palette = QtWidgets.QWidget()
        palette = QtWidgets.QHBoxLayout(self._density_palette)
        palette.setContentsMargins(0, 0, 0, 0)
        palette.addWidget(QtWidgets.QLabel("Event-count heatmap colormap"))
        self._cmap_combo = QtWidgets.QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setToolTip("Sequential colour map for the density (perceptually uniform).")
        palette.addWidget(self._cmap_combo)
        palette.addStretch(1)
        row1.addStretch(1)
        root.addWidget(self._density_analysis_controls)
        root.addWidget(self._density_palette)

        self._density_export_controls = QtWidgets.QWidget()
        row2 = QtWidgets.QHBoxLayout(self._density_export_controls)
        row2.setContentsMargins(0, 0, 0, 0)
        self._d_title = QtWidgets.QLineEdit()
        self._d_title.setPlaceholderText("title (auto)")
        self._d_xlabel = QtWidgets.QLineEdit()
        self._d_xlabel.setPlaceholderText("x label (auto)")
        self._d_ylabel = QtWidgets.QLineEdit()
        self._d_ylabel.setPlaceholderText("y label (auto)")
        for legacy in (self._d_title, self._d_xlabel, self._d_ylabel):
            legacy.setParent(self)
            legacy.hide()  # shared Appearance labels supersede these legacy config controls
        self._d_save = QtWidgets.QPushButton("Export current figure…")
        self._d_save.setToolTip(
            "Export the density map, marginals, and colour bar as PNG, SVG, or PDF.")
        self._d_save.clicked.connect(lambda: self.save_image())
        row2.addWidget(self._d_save)
        self._d_data = QtWidgets.QPushButton("Export data (CSV)…")
        self._d_data.setToolTip(
            "Export exact density bin edges/counts and source-event identities.")
        self._d_data.clicked.connect(lambda: self.export_data())
        row2.addWidget(self._d_data)
        root.addWidget(self._density_export_controls)

        self._amplitude_warning_summary = WarningSummary(title="Selected amplitude warnings")
        self._amplitude_warning_label = self._amplitude_warning_summary
        root.addWidget(self._amplitude_warning_summary)

        # The density scene: top marginal + main image + right marginal + colour bar.
        self._density_glw = pg.GraphicsLayoutWidget()
        self._density_glw.setMinimumSize(560, 460)
        self._density_top = self._density_glw.addPlot(row=0, col=0)
        self._density_main = self._density_glw.addPlot(row=1, col=0)
        self._density_right = self._density_glw.addPlot(row=1, col=1)
        # No colour-map menu on the bar: its pick would recolour the map behind the
        # Colormap control, which is the one the page saves (plot_interaction's rule).
        self._colorbar = pg.ColorBarItem(values=(0, 1), colorMap=_get_colormap("viridis"),
                                         label="Event count", colorMapMenu=False)
        self._density_glw.addItem(self._colorbar, row=1, col=2)

        self._density_main.showGrid(x=True, y=True, alpha=0.15)
        self._density_top.setXLink(self._density_main)
        self._density_right.setYLink(self._density_main)
        self._density_top.setMaximumHeight(90)
        self._density_right.setMaximumWidth(90)
        for m, axis in ((self._density_top, "bottom"), (self._density_right, "left")):
            m.getAxis(axis).setStyle(showValues=False)
            m.setMouseEnabled(x=False, y=False)
        self._density_top.getAxis("left").setLabel("count")
        self._density_right.getAxis("bottom").setLabel("count")
        root.addWidget(self._density_glw, 1)
        return page

    def _build_ensemble_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._ensemble_analysis_controls = QtWidgets.QWidget()
        rows = QtWidgets.QVBoxLayout(self._ensemble_analysis_controls)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(4)
        row1 = QtWidgets.QHBoxLayout()
        rows.addLayout(row1)
        self._normalize = QtWidgets.QCheckBox("Normalize time")
        self._normalize.setChecked(True)
        self._normalize.setToolTip(
            "Duration-normalize each event onto a common [0, 1] axis (align is then immaterial).")
        row1.addWidget(self._normalize)

        row1.addWidget(QtWidgets.QLabel("Align"))
        self._align_combo = QtWidgets.QComboBox()
        # The texts ARE the kernel's ids (and the saved/exported value), as before.
        self._align_combo.addItems(list(event_shape.ENSEMBLE_ALIGNS))
        self._align_combo.setEnabled(False)     # only meaningful when NOT normalizing time
        self._align_combo.setToolTip(
            "Real-time alignment reference (used only when 'Normalize time' is off).\n"
            "• start / center — each event's first sample or midpoint at t = 0;\n"
            "• extremum — each event's deepest sample at t = 0: the sample of maximum "
            "excursion, read with the event table's polarity, so a blockade that lowers "
            "the current and one that raises it are both aligned on their deepest point.")
        row1.addWidget(self._align_combo)

        row1.addWidget(QtWidgets.QLabel("Points"))
        self._npoints = QtWidgets.QSpinBox()
        self._npoints.setRange(8, 4000)
        self._npoints.setValue(120)
        self._npoints.setToolTip("Samples on the common axis each event is resampled to.")
        row1.addWidget(self._npoints)

        row1.addWidget(QtWidgets.QLabel("Pad %"))
        self._pad = QtWidgets.QSpinBox()
        self._pad.setRange(0, 200)
        self._pad.setValue(20)
        self._pad.setToolTip("Pre/post padding as a percentage of each event's dwell.")
        row1.addWidget(self._pad)
        row1.addStretch(1)

        row1b = QtWidgets.QHBoxLayout()
        rows.addLayout(row1b)
        row1b.addWidget(QtWidgets.QLabel("Events"))
        self._subset_combo = QtWidgets.QComboBox()
        self._subset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._subset_combo.addItem(_subset_text(_SUBSET_DEFAULT), _SUBSET_DEFAULT)
        self._subset_combo.setToolTip(
            "Which accepted events the ensemble uses: all of them, or one of the Event "
            "gallery's curation subsets (unreviewed, unlabelled, a label, hand-made or "
            "detected) within them. Rejected events are never drawn. The Event-count "
            "heatmap always uses every accepted event.")
        row1b.addWidget(self._subset_combo)
        row1b.addWidget(QtWidgets.QLabel("Band"))
        self._band_combo = QtWidgets.QComboBox()
        for text, kind in _BANDS:
            self._band_combo.addItem(text, kind)
        self._band_combo.setToolTip(
            "Shade the spread of the events around a summary curve: mean ± 1 sample SD "
            "around the solid mean, or the 25th–75th percentile band around the dashed "
            "median. Drawn from the traces already on screen — choosing one never "
            "re-reads the recording. Details gives how many events each point uses.")
        row1b.addWidget(self._band_combo)
        row1b.addStretch(1)
        root.addWidget(self._ensemble_analysis_controls)

        self._ensemble_export_controls = QtWidgets.QWidget()
        row2 = QtWidgets.QHBoxLayout(self._ensemble_export_controls)
        row2.setContentsMargins(0, 0, 0, 0)
        self._e_title = QtWidgets.QLineEdit()
        self._e_title.setPlaceholderText("title (auto)")
        self._e_title.setParent(self)
        self._e_title.hide()
        self._e_save = QtWidgets.QPushButton("Export current figure…")
        self._e_save.setToolTip("Export the event ensemble as PNG, SVG, or PDF.")
        self._e_save.clicked.connect(lambda: self.save_image())
        row2.addWidget(self._e_save)
        self._e_data = QtWidgets.QPushButton("Export data (CSV)…")
        self._e_data.setToolTip(
            "Export individual resampled traces, mean, median, the band and its events "
            "per point, and event identities.")
        self._e_data.clicked.connect(lambda: self.export_data())
        row2.addWidget(self._e_data)
        root.addWidget(self._ensemble_export_controls)

        # The ensemble's own readout: what was aligned on, which events, and what the band
        # is and how many events each point uses. Collapsed to its "Full details" button.
        self._ensemble_notes = WarningSummary(title="Ensemble notes")
        root.addWidget(self._ensemble_notes)

        self._ensemble_widget = pg.PlotWidget()
        self._ensemble_widget.setMinimumSize(560, 420)
        self._ensemble_plot: pg.PlotItem = self._ensemble_widget.getPlotItem()
        self._ensemble_plot.showGrid(x=True, y=True, alpha=0.15)
        root.addWidget(self._ensemble_widget, 1)

        # Wire ALL change signals last so building the controls above never redrew.
        self._x_combo.currentIndexChanged.connect(self._on_density_changed)
        self._y_combo.currentIndexChanged.connect(self._on_density_changed)
        self._logx.toggled.connect(self._on_density_changed)
        self._logy.toggled.connect(self._on_density_changed)
        self._bins.valueChanged.connect(self._on_density_changed)
        self._cmap_combo.currentTextChanged.connect(self._on_density_colormap_changed)
        self._d_title.textChanged.connect(self._on_density_labels_changed)
        self._d_xlabel.textChanged.connect(self._on_density_labels_changed)
        self._d_ylabel.textChanged.connect(self._on_density_labels_changed)
        self._normalize.toggled.connect(self._on_normalize_toggled)
        self._align_combo.currentTextChanged.connect(self._on_ensemble_changed)
        self._npoints.valueChanged.connect(self._on_ensemble_changed)
        self._pad.valueChanged.connect(self._on_ensemble_changed)
        self._subset_combo.currentIndexChanged.connect(self._on_ensemble_changed)
        self._band_combo.currentIndexChanged.connect(self._on_band_changed)
        self._e_title.textChanged.connect(self._on_ensemble_title_changed)
        self._tabs.currentChanged.connect(self._on_view_changed)
        return page

    # -- data in ---------------------------------------------------------------------------------
    def set_source(self, doc_id: str | None, channel: int | None,
                   eventset: EventSet | None, *, defer_analysis: bool = False) -> None:
        """Install the active detection's events + its calibrated signal, then redraw.

        Pass the FULL detection EventSet (the window filters to its ACCEPTED events
        itself, mirroring PlotsWindow). ``None`` / an all-rejected set clears the
        window to its empty state. The calibrated signal is pulled from the Session
        for ``(doc_id, channel)`` -- the same trace the events were detected on.

        ``defer_analysis=True`` is the project-restore path.  It binds the durable
        source and calibrated trace but deliberately does not call ``density_2d`` or
        ``event_ensemble``; the workspace adapter installs the frozen numerical
        arrays immediately afterward and paints them through
        :meth:`_render_workspace_snapshot`.
        """
        self._doc_id = None
        self._channel = None
        self._eventset = None
        self._signal = None
        self._fs = None
        self._units = "pA"
        if eventset is not None and doc_id is not None and channel is not None \
                and self._session.has_document(doc_id):
            try:
                ch = self._session.calibrated(doc_id, channel)
            except (KeyError, TypeError):
                ch = None
            if ch is not None and ch.sample_rate_hz:
                self._doc_id = str(doc_id)
                self._channel = int(channel)
                self._eventset = eventset
                self._signal = np.asarray(ch.data)
                self._fs = float(ch.sample_rate_hz)
                self._units = ch.units or "pA"
        for combo in (self._x_combo, self._y_combo):
            index = combo.findData(_M_AMPLITUDE)
            item = combo.model().item(index)
            available = self._eventset is not None and _M_AMPLITUDE in self._eventset.df
            item.setEnabled(available)
            item.setToolTip("Selected amplitude applied in Curate; use consistent methods and "
                            "interval scopes and review stale-value warnings before pooling."
                            if available else "Apply an amplitude measurement in Curate to enable this metric.")
        self._refresh_subset_options()
        if defer_analysis:
            # Never leave another project's result available behind a newly bound
            # source if the artifact later proves malformed.  Clearing Qt items and
            # cached arrays is presentation bookkeeping, not an analysis redraw.
            self._density_counts = None
            self._density_x_edges = None
            self._density_y_edges = None
            self._ensemble = None
            self._ensemble_band = None
            self._band_items = ()
            self._ensemble_alignment = None
            self._ensemble_input_count = None
            self._ensemble_unalignable = 0
            self._ensemble_notes.set_warnings(())
            self._density_main.clear()
            self._density_top.clear()
            self._density_right.clear()
            self._ensemble_plot.clear()
            reset_plot_history(self._density_main, self._ensemble_plot)
            self._density_image = None
            self._top_marginal_item = None
            self._right_marginal_item = None
            self._events_item = None
            self._mean_item = None
            self._median_item = None
            return
        self._redraw()

    def _accepted(self) -> EventSet | None:
        """The accepted-only EventSet, or ``None`` when there is nothing to plot."""
        if self._eventset is None or self._signal is None or not self._fs:
            return None
        acc = self._eventset.accepted
        return acc if len(acc.df) else None

    # -- ensemble subset -------------------------------------------------------------------------
    def _ensemble_subset(self) -> str:
        mode = self._subset_combo.currentData()
        return str(mode) if mode else _SUBSET_DEFAULT

    def _refresh_subset_options(self) -> None:
        """Offer the gallery's curation subsets the bound detection actually has.

        The same list the Event gallery builds (``curation_view.filter_options`` over the
        whole frame: labels found in it, hand-made/detected once it holds a hand-made row),
        without the two that would draw rejected events. The current choice is kept even
        when the new source no longer offers it -- a window never drops the subset the user
        is looking at, it reports that the subset is empty instead.
        """
        current = self._ensemble_subset()
        df = self._eventset.df if self._eventset is not None else None
        modes = [m for m in filter_options(curation_progress(df), None, current)
                 if m not in _SUBSET_EXCLUDED]
        if current not in modes:
            modes.append(current)
        blocked = self._subset_combo.blockSignals(True)
        try:
            self._subset_combo.clear()
            for mode in modes:
                self._subset_combo.addItem(_subset_text(mode), mode)
            self._subset_combo.setCurrentIndex(max(0, self._subset_combo.findData(current)))
        finally:
            self._subset_combo.blockSignals(blocked)

    def set_ensemble_subset(self, mode: str) -> None:
        """Select an ensemble subset by its curation-filter name (added if not offered)."""
        mode = str(mode or _SUBSET_DEFAULT)
        if mode in _SUBSET_EXCLUDED:
            raise ValueError(f"the ensemble never draws rejected events; got {mode!r}")
        index = self._subset_combo.findData(mode)
        if index < 0:
            self._subset_combo.addItem(_subset_text(mode), mode)
            index = self._subset_combo.count() - 1
        self._subset_combo.setCurrentIndex(index)

    def _ensemble_events(self) -> EventSet | None:
        """The accepted events of the chosen subset, or ``None`` when there are none."""
        acc = self._accepted()
        if acc is None:
            return None
        mode = self._ensemble_subset()
        if mode == _SUBSET_DEFAULT:
            return acc
        try:
            keep = filter_mask(acc.df, mode)
        except ValueError:          # a saved subset this build cannot parse keeps nothing
            keep = np.zeros(len(acc.df), dtype=bool)
        if not keep.any():
            return None
        return EventSet(acc.df[keep].copy(), {**acc.provenance, "ensemble_subset": mode})

    # -- current control state -------------------------------------------------------------------
    def _metric_values(self, metric_id: str, acc: EventSet) -> np.ndarray:
        """The chosen metric's per-event array from the ACCEPTED events (+ signal for ECD)."""
        if metric_id == _M_ECD:
            return event_shape.charge_deficit(self._signal, self._fs, acc)
        if metric_id == _M_AMPLITUDE and metric_id not in acc.df:
            return np.full(len(acc.df), np.nan, dtype=np.float64)
        return acc.df[metric_id].to_numpy(dtype=np.float64)

    def _metric_label(self, metric_id: str) -> str:
        base = dict(_METRICS)[metric_id]
        if metric_id == _M_DEPTH:
            return f"Median depth ({self._units})"
        if metric_id == _M_AMPLITUDE:
            # Units travel with the computed bins, never a newly bound live table.
            return f"Selected amplitude ({self._amplitude_snapshot_units})"
        if metric_id == _M_ECD:
            return f"ECD ({self._units}·s)"
        return base

    # -- redraw dispatch -------------------------------------------------------------------------
    def _on_density_changed(self, *_a) -> None:
        if not self._updating:
            shown, logs = self._density_shown, self._density_logs
            moved = shown not in (None, self._density_quantity())
            logged = logs not in (None, self._density_log_state()) and not self._figure_log
            self._redraw_density()
            if moved:       # X is the map's x axis and Y its y axis: only the one that moved
                x_id, y_id = self._density_quantity()
                axes = tuple((PIN_MAP, side) for side, was, now in (
                    ("bottom", shown[0], x_id), ("left", shown[1], y_id)) if was != now)
                self._reset_words(f"{_METRIC_PHRASE[y_id]} against {_METRIC_PHRASE[x_id]}", axes)
            elif logged:    # log10 values on a linear axis: the pin goes, the words stay
                axes = tuple((PIN_MAP, side) for side, was, now in zip(
                    ("bottom", "left"), logs, self._density_log_state()) if was != now)
                self._reset_words(self._log_phrase(), axes, only=())
            self.workspaceChanged.emit()

    def _density_quantity(self) -> tuple[str, str]:
        """The two metrics the Event-count heatmap would show now."""
        return self._x_combo.currentData(), self._y_combo.currentData()

    def _density_log_state(self) -> tuple[bool, bool]:
        """The map's Log x and Log y as the check boxes hold them now."""
        return self._logx.isChecked(), self._logy.isChecked()

    def _log_phrase(self) -> str:
        """What the status line says the map shows after Log x or Log y moved."""
        x_id, y_id = self._density_quantity()
        x, y = (f"log10 of {_METRIC_PHRASE[m]}" if on else _METRIC_PHRASE[m]
                for m, on in zip((x_id, y_id), self._density_log_state()))
        return f"{y} against {x}"

    def _log_from_figure(self, side: str, on: bool) -> None:
        """Axis properties' Log scale, routed to the map's own Log x / Log y: it redraws as a
        click on the box does, but takes no pin off, because the dialog commits the range
        typed beside it (for the axis as it is about to be drawn: unticking Log scale there to
        type a range in seconds is how a person pins the map's x, which opens on a log axis)."""
        self._figure_log = True
        try:
            (self._logx if side == "bottom" else self._logy).setChecked(bool(on))
        finally:
            self._figure_log = False

    def _reset_words(self, shows: str, axes: tuple = (), only: tuple[str, ...] | None = None
                     ) -> None:
        """A figure's quantity changed: the typed words, and the pin of each axis in ``axes``
        (the ones whose quantity moved), go back to automatic, as one step of this window's
        appearance Undo, and the status line (the one under the canvas, which the Explorer
        shows) says what went, or loses the sentence of an earlier change. ``only=()`` leaves
        the words and takes the pins alone."""
        self._appearance_widget.reset_typed_words(shows, self.statusBar(), only=only, axes=axes,
                                                  primary=PIN_MAP)

    def _on_ensemble_changed(self, *_a) -> None:
        if not self._updating:
            self._redraw_ensemble()
            self.workspaceChanged.emit()

    def _on_density_labels_changed(self, *_a) -> None:
        if not self._updating:
            self._apply_density_labels()
            self.workspaceChanged.emit()

    def _on_density_colormap_changed(self, *_a) -> None:
        if not self._updating:
            # LUT replacement is presentation only: do not replace image data,
            # recompute an ECD metric, re-bin events, or disturb the ViewBox range.
            if self._density_image is not None:
                cmap = _get_colormap(self._cmap_combo.currentText())
                self._density_image.setColorMap(cmap)
                self._colorbar.setColorMap(cmap)
            self.workspaceChanged.emit()

    def _on_ensemble_title_changed(self, *_a) -> None:
        if not self._updating:
            self._apply_ensemble_title()
            self.workspaceChanged.emit()

    def _apply_density_labels(self) -> None:
        """Update the existing populated plot; leave empty/error messages intact."""
        if self._density_image is None:
            return
        x_id, y_id = self._x_combo.currentData(), self._y_combo.currentData()
        xlabel = self._d_xlabel.text().strip() or (
            self._metric_label(x_id) + (" (log10)" if self._logx.isChecked() else ""))
        ylabel = self._d_ylabel.text().strip() or (
            self._metric_label(y_id) + (" (log10)" if self._logy.isChecked() else ""))
        self._density_main.setLabel("bottom", xlabel)
        self._density_main.setLabel("left", ylabel)
        title = self._d_title.text().strip() or self._density_auto_title
        self._density_main.setTitle(title)
        self._show_amplitude_warnings()
        self._apply_appearance()

    def _show_amplitude_warnings(self) -> None:
        self._density_glw.setToolTip("\n\n".join(self._amplitude_warnings))
        self._amplitude_warning_label.setText("\n".join(self._amplitude_warnings))
        self._amplitude_warning_summary.set_warnings(self._amplitude_warnings)

    def _apply_ensemble_title(self) -> None:
        if self._mean_item is not None:
            self._ensemble_plot.setTitle(
                self._e_title.text().strip() or self._ensemble_auto_title)
            self._apply_appearance()

    def _on_view_changed(self, *_a) -> None:
        shown, self._view_shown = self._view_shown, self.current_view()
        if not self._updating:
            self._apply_appearance()
            if shown != self._view_shown:
                self._reset_words("the ensemble" if self._view_shown == ENSEMBLE
                                  else "the event-count heatmap")
            self.workspaceChanged.emit()

    def _on_normalize_toggled(self, on: bool) -> None:
        self._align_combo.setEnabled(not on)   # align only bites in real-time mode
        moved = not self._updating and self._normalize_shown not in (None, bool(on))
        self._on_ensemble_changed()
        if moved:       # normalised or real time is the ensemble's x axis; current stays current
            self._reset_words("the ensemble against " + (
                "time normalised to each dwell" if on else "time in seconds"),
                ((PIN_ENSEMBLE, "bottom"),))

    def _redraw(self) -> None:
        self._redraw_density()
        self._redraw_ensemble()

    # -- density ---------------------------------------------------------------------------------
    def _redraw_density(self) -> None:
        self._density_shown = self._density_logs = None
        self._amplitude_frozen_snapshot = False
        self._amplitude_warnings = ()
        self._amplitude_warning_label.hide()
        self._amplitude_warning_summary.set_warnings(())
        self._density_main.clear()
        self._density_top.clear()
        self._density_right.clear()
        reset_plot_history(self._density_main)   # new geometry: old ranges mean nothing now
        self._density_image = None
        self._top_marginal_item = None
        self._right_marginal_item = None
        self._density_counts = None
        self._density_x_edges = None
        self._density_y_edges = None

        acc = self._accepted()
        if acc is None:
            self._density_message(
                "No accepted events.\nRun detection and accept some events, then Refresh.")
            return

        x_id = self._x_combo.currentData()
        y_id = self._y_combo.currentData()
        if _M_AMPLITUDE in (x_id, y_id):
            self._amplitude_warnings = amplitude_comparability([("Accepted events", acc.df)])
            self._amplitude_snapshot_units = amplitude_units_label([acc.df])
            self._show_amplitude_warnings()
        logx, logy = self._logx.isChecked(), self._logy.isChecked()
        x = self._metric_values(x_id, acc)
        y = self._metric_values(y_id, acc)
        nb = int(self._bins.value())
        counts, xe, ye = event_shape.density_2d(
            x, y, xbins=nb, ybins=nb, xlog=logx, ylog=logy)
        self._density_counts = counts
        self._density_x_edges = np.asarray(xe, dtype=np.float64).copy()
        self._density_y_edges = np.asarray(ye, dtype=np.float64).copy()
        if counts.sum() == 0:
            self._density_message(
                "No events fall on these axes.\nCheck the log toggles (non-positive values are "
                "dropped for a log axis).")
            return

        cmap = _get_colormap(self._cmap_combo.currentText())
        xe_p = _to_plot(xe, logx)
        ye_p = _to_plot(ye, logy)
        cmax = float(counts.max())

        # density_2d stores counts[x_bin, y_bin]. Pin this item's orientation;
        # the application's global imageAxisOrder may be row-major for other views.
        img = pg.ImageItem(axisOrder="col-major")
        img.setImage(counts)
        img.setRect(QtCore.QRectF(
            xe_p[0], ye_p[0], xe_p[-1] - xe_p[0], ye_p[-1] - ye_p[0]))
        img.setColorMap(cmap)
        img.setLevels((0.0, max(cmax, 1.0)))
        self._density_main.addItem(img)
        self._density_image = img

        self._colorbar.setColorMap(cmap)
        self._colorbar.setLevels((0.0, max(cmax, 1.0)))
        self._colorbar.setImageItem(img)

        # Marginals: the X and Y projections of the 2-D counts, in the SAME plot coords.
        brush = _alpha_brush(_BAR_HEX, 150)
        xc = 0.5 * (xe_p[:-1] + xe_p[1:])
        yc = 0.5 * (ye_p[:-1] + ye_p[1:])
        xw = float(xe_p[1] - xe_p[0]) * 0.96
        yw = float(ye_p[1] - ye_p[0]) * 0.96
        top_bar = pg.BarGraphItem(
            x=xc, height=counts.sum(axis=1), width=xw, brush=brush, pen=None)
        right_bar = pg.BarGraphItem(
            y=yc, height=yw, width=counts.sum(axis=0), x0=0.0, brush=brush, pen=None)
        self._density_top.addItem(top_bar)
        self._density_right.addItem(right_bar)
        self._top_marginal_item = top_bar
        self._right_marginal_item = right_bar

        self._density_shown, self._density_logs = (x_id, y_id), (logx, logy)
        self._density_auto_title = (
            f"{self._metric_label(y_id)} vs {self._metric_label(x_id)}   ·   "
            f"n = {len(acc.df)} events")
        if _M_AMPLITUDE in (x_id, y_id):
            self._density_auto_title = (
                f"{self._metric_label(y_id)} vs {self._metric_label(x_id)}   ·   "
                f"n = {int(counts.sum())} of {len(acc.df)} accepted events")
        self._apply_density_labels()
        self._frame(self._density_main)

    def _density_message(self, text: str) -> None:
        self._density_main.setTitle(None)
        self._density_main.titleLabel.setText("")
        self._density_main.setLabel("bottom", "")
        self._density_main.setLabel("left", "")
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=_MSG_HEX)
        self._density_main.addItem(item)
        self._density_main.setXRange(0, 1, padding=0)
        self._density_main.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)

    # -- ensemble --------------------------------------------------------------------------------
    def _redraw_ensemble(self) -> None:
        self._normalize_shown = None
        self._ensemble_plot.clear()
        reset_plot_history(self._ensemble_plot)  # new geometry: old ranges mean nothing now
        self._events_item = None
        self._mean_item = None
        self._median_item = None
        self._ensemble = None
        self._ensemble_band = None
        self._band_items = ()
        self._ensemble_alignment = None
        self._ensemble_input_count = None
        self._ensemble_unalignable = 0

        if self._accepted() is None:
            self._ensemble_message(
                "No accepted events.\nRun detection and accept some events, then Refresh.")
            return
        acc = self._ensemble_events()
        if acc is None:
            self._ensemble_message(
                f"No accepted events in the subset “{self._ensemble_subset()}”.\n"
                "Choose another subset under Events.")
            return

        normalize = self._normalize.isChecked()
        align = self._align_combo.currentText()
        npoints = int(self._npoints.value())
        pad_frac = float(self._pad.value()) / 100.0
        polarity = None
        if align == "extremum" and not normalize:
            # Resolved HERE, once, so Details and the CSV's identity replay name the very
            # polarity the kernel aligned with rather than re-deriving it later.
            self._ensemble_alignment = event_shape.ensemble_polarity(self._signal, acc)
            polarity = self._ensemble_alignment[0]
            self._ensemble_unalignable = _count_unalignable(self._signal, acc, polarity)
        T, matrix, mean, median = event_shape.event_ensemble(
            self._signal, self._fs, acc, n_points=npoints, pad_frac=pad_frac,
            align=align, normalize_time=normalize, polarity=polarity)
        self._ensemble = (T, matrix, mean, median)
        self._ensemble_input_count = int(len(acc.df))

        if matrix.shape[0] == 0:
            self._ensemble_message(
                "Every accepted event is too close to the trace edge for this padding.")
            return

        # All events as ONE PlotDataItem (NaN-separated) so 1e3+ events stay cheap.
        gap = np.array([np.nan])
        xs = np.concatenate([np.concatenate([T, gap]) for _ in range(matrix.shape[0])])
        ys = np.concatenate([np.concatenate([row, gap]) for row in matrix])
        self._events_item = pg.PlotDataItem(
            x=xs, y=ys, connect="finite", pen=_alpha_pen(_EVENT_HEX, 60, 1))
        self._ensemble_plot.addItem(self._events_item)

        self._mean_item = pg.PlotDataItem(x=T, y=mean, pen=pg.mkPen(_MEAN_HEX, width=2.5))
        self._median_item = pg.PlotDataItem(
            x=T, y=median, pen=pg.mkPen(_MEDIAN_HEX, width=2.5, style=Qt.DashLine))
        self._compute_ensemble_band()
        self._add_band_items()          # under everything, the faint events included
        self._ensemble_plot.addItem(self._mean_item)
        self._ensemble_plot.addItem(self._median_item)

        skipped = len(acc.df) - matrix.shape[0]
        self._normalize_shown = normalize
        xlabel = "Normalized time (dwell)" if normalize else "Time (s)"
        self._ensemble_plot.setLabel("bottom", xlabel)
        self._ensemble_plot.setLabel("left", f"Current ({self._units})")
        self._ensemble_auto_title = self._ensemble_title(matrix.shape[0], skipped)
        self._apply_ensemble_title()
        self._refresh_ensemble_notes()
        self._frame(self._ensemble_plot)

    # -- ensemble band ---------------------------------------------------------------------------
    def _band_kind(self) -> str:
        return str(self._band_combo.currentData() or _BAND_NONE)

    def set_band(self, kind: str) -> None:
        """Select the variability band by kind: ``""`` (none), ``"sd"`` or ``"iqr"``."""
        index = self._band_combo.findData(str(kind or _BAND_NONE))
        if index < 0:
            raise ValueError(f"unknown ensemble band {kind!r}")
        self._band_combo.setCurrentIndex(index)

    def ensemble_band(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """``(lower, upper, events per point)`` of the band on screen, or ``None``."""
        return self._ensemble_band

    def _compute_ensemble_band(self) -> None:
        """The chosen band from the ensemble matrix ALREADY computed -- never the trace."""
        self._ensemble_band = None
        kind = self._band_kind()
        if not kind or self._ensemble is None:
            return
        matrix = np.asarray(self._ensemble[1], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return
        self._ensemble_band = event_shape.ensemble_band(matrix, kind)

    def _remove_band_items(self) -> None:
        for item in self._band_items:
            self._ensemble_plot.removeItem(item)
        self._band_items = ()

    def _add_band_items(self) -> None:
        """Draw ``self._ensemble_band`` as a fill between two curves (NaN points left open)."""
        self._remove_band_items()
        kind = self._band_kind()
        if not kind or self._ensemble is None or self._ensemble_band is None:
            return
        T = np.asarray(self._ensemble[0], dtype=np.float64)
        lower, upper, _n = (np.asarray(v, dtype=np.float64) for v in self._ensemble_band)
        if lower.shape != T.shape or upper.shape != T.shape:
            return
        hex_color = self._band_anchor_hex()
        edge = _alpha_pen(hex_color, 110, 1)
        # connect="finite" splits both edges at the same NaN points (the band shares one
        # mask), and FillBetweenItem pairs their sub-paths -- a gap where fewer than two
        # events contribute stays a gap instead of being bridged by a straight edge.
        lo_item = pg.PlotCurveItem(x=T, y=lower, connect="finite", pen=edge)
        hi_item = pg.PlotCurveItem(x=T, y=upper, connect="finite", pen=edge)
        fill = pg.FillBetweenItem(lo_item, hi_item, brush=_alpha_brush(hex_color, 60))
        for item in (fill, lo_item, hi_item):
            # Below every other item wherever it is added: under the summary curves, and
            # under the faint per-event lines too, so the shading never tints the traces
            # it summarises.
            item.setZValue(-1)
            self._ensemble_plot.addItem(item)
        self._band_items = (fill, lo_item, hi_item)

    def _band_anchor_hex(self) -> str:
        """The band takes the colour of the curve it surrounds, appearance edits included."""
        item = self._mean_item if self._band_kind() == "sd" else self._median_item
        if item is not None:
            try:
                return pg.mkPen(item.opts["pen"]).color().name()
            except (KeyError, TypeError):
                pass
        return _MEAN_HEX if self._band_kind() == "sd" else _MEDIAN_HEX

    def _on_band_changed(self, *_a) -> None:
        if self._updating:
            return
        # A reduction of the traces on screen: no kernel, no trace read, and the user's
        # zoom is put back exactly (adding items to an auto-ranging view would refit it).
        view = self._ensemble_plot.getViewBox()
        ranges = [list(axis) for axis in view.viewRange()]
        auto = list(view.autoRangeEnabled())
        view.disableAutoRange()
        try:
            self._compute_ensemble_band()
            if self._mean_item is not None:
                self._add_band_items()
                self._ensemble_auto_title = self._ensemble_title(
                    *self._ensemble_kept_and_skipped())
                self._apply_ensemble_title()
            else:
                self._remove_band_items()
            self._refresh_ensemble_notes()
        finally:
            view.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
            for axis, enabled in enumerate(auto):
                if enabled:
                    view.enableAutoRange(axis=axis, enable=enabled)
        self.workspaceChanged.emit()

    def _ensemble_kept_and_skipped(self) -> tuple[int, int]:
        """``(drawn, skipped)`` for the ensemble on screen, live or restored.

        :meth:`_ensemble_skips` splits ``skipped`` into edge skips and events with no
        finite sample to align on.
        """
        n_kept = int(np.asarray(self._ensemble[1]).shape[0]) if self._ensemble else 0
        base = self._ensemble_input_count
        if base is None:
            base = int(getattr(self, "_nanoscope_workspace_accepted_event_count", n_kept))
        return n_kept, max(0, int(base) - n_kept)

    def _ensemble_skips(self, skipped: int) -> tuple[int, int]:
        """``(edge-skipped, no finite sample)`` out of the ``skipped`` events not drawn."""
        unalignable = min(max(0, int(self._ensemble_unalignable or 0)), max(0, int(skipped)))
        return max(0, int(skipped)) - unalignable, unalignable

    def _ensemble_title(self, n_kept: int, skipped: int) -> str:
        mode = self._ensemble_subset()
        band = {"sd": ", shaded mean ± 1 SD", "iqr": ", shaded IQR (25–75 %)"}.get(
            self._band_kind(), "")
        edge, unalignable = self._ensemble_skips(skipped)
        return (f"Ensemble of {n_kept} event(s)"
                + ("" if mode == _SUBSET_DEFAULT else f"   ·   {mode}")
                + (f"   ·   {edge} edge-skipped" if edge else "")
                + (f"   ·   {unalignable} with no finite sample" if unalignable else "")
                + "   —   mean (solid) + median (dashed)" + band)

    def _ensemble_detail_lines(self, cautions: list[str] | None = None) -> list[str]:
        """What the ensemble on screen is: subset, alignment, and the band with its n.

        Lines that qualify what is drawn (events with no finite sample to align on, points
        where the curves and their band are not defined, a band with too few events) are
        also appended to ``cautions``, which the ensemble's notes show as its summary.
        """
        if self._ensemble is None:
            return []
        matrix = np.asarray(self._ensemble[1])
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return []
        lines = [f"Events: {_subset_text(self._ensemble_subset())} "
                 f"({matrix.shape[0]} drawn)."]
        if not self._normalize.isChecked() and self._align_combo.currentText() == "extremum":
            if self._ensemble_alignment is not None:
                pol, source = self._ensemble_alignment
                direction = "decrease" if pol < 0 else "increase"
                lines.append(
                    f"Aligned at each event's extremum: the sample inside the event where "
                    f"polarity × (current − baseline) is largest, with polarity {pol:+d} "
                    f"({source}), so the deepest point of a current {direction} sits at "
                    f"t = 0. The window reaches the longest dwell plus the padding to both "
                    f"sides of it, so every drawn event lies wholly inside.")
            else:
                lines.append("Aligned at each event's extremum (the polarity used was not "
                             "saved with this snapshot).")
            _edge, unalignable = self._ensemble_skips(self._ensemble_kept_and_skipped()[1])
            if unalignable:
                lines.append(
                    f"{unalignable} event(s) have no finite sample inside their own span, "
                    f"so they have no deepest sample to align on and are not drawn. They "
                    f"are counted apart from the events whose window runs off the trace.")
                if cautions is not None:
                    cautions.append(lines[-1])
        kind = self._band_kind()
        if kind and self._ensemble_band is not None:
            lines.append(
                "Band: mean ± 1 sample standard deviation (ddof = 1), around the solid mean."
                if kind == "sd" else
                "Band: interquartile range, the 25th to 75th percentile (linear "
                "interpolation), around the dashed median.")
            # n is the count behind the summary curve at each point: every drawn event,
            # or 0 where the curve itself is undefined (event_shape.ensemble_point_counts).
            n = np.asarray(self._ensemble_band[2], dtype=np.int64)
            n_rows = int(matrix.shape[0])
            undefined = int(np.count_nonzero(n < n_rows))
            if n.size and n_rows < 2:
                lines.append("The band needs at least 2 events: one value has no spread, so "
                             "no band is drawn.")
                if cautions is not None:
                    cautions.append(lines[-1])
            elif n.size and not undefined:
                lines.append(f"Every point of the band uses n = {n_rows} events, the same "
                             f"events as its curve.")
            elif n.size:
                lines.append(
                    f"At {undefined} of {n.size} points at least one event's window holds a "
                    f"non-finite sample. The mean and median take every drawn event at every "
                    f"point, so neither is defined there, and the band is left out with "
                    f"them; everywhere else it uses all n = {n_rows} events. Export data "
                    f"(CSV) gives n_events_at_point, 0 at those points.")
                if cautions is not None:
                    cautions.append(lines[-1])
        return lines

    def _refresh_ensemble_notes(self) -> None:
        """The ensemble's readout beside the figure: cautions as the summary, all in Details.

        Kept out of the title and the artwork, which carry only what the figure shows.
        """
        cautions: list[str] = []
        lines = self._ensemble_detail_lines(cautions)
        self._ensemble_notes.set_warnings(tuple(cautions), full_text="\n".join(lines))

    def _ensemble_message(self, text: str) -> None:
        self._ensemble_notes.set_warnings(())
        self._ensemble_plot.setTitle(None)
        self._ensemble_plot.titleLabel.setText("")
        self._ensemble_plot.setLabel("bottom", "")
        self._ensemble_plot.setLabel("left", "")
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=_MSG_HEX)
        self._ensemble_plot.addItem(item)
        self._ensemble_plot.setXRange(0, 1, padding=0)
        self._ensemble_plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)

    # -- public accessors (owner + tests) --------------------------------------------------------
    def details_text(self) -> str:
        """Retained view diagnostics and source provenance for figure bundles."""
        density = self.current_view() == DENSITY
        plot = self._density_main if density else self._ensemble_plot
        blocks = [DENSITY_LABEL if density else ENSEMBLE]
        if plot.titleLabel.text:
            blocks.append(str(plot.titleLabel.text))
        blocks.extend(item.textItem.toPlainText() for item in plot.items
                      if isinstance(item, pg.TextItem))
        if density:
            blocks.extend(self._amplitude_warnings)
        else:
            blocks.append("\n".join(self._ensemble_detail_lines()))
        blocks.append("Source provenance\n" + json.dumps(
            self._export_common("density" if density else "ensemble"),
            ensure_ascii=False, indent=2, default=self._json_default))
        return "\n\n".join(block for block in blocks if block)

    def current_view(self) -> str:
        return DENSITY if self._tabs.currentIndex() == 0 else ENSEMBLE

    def set_view(self, name: str) -> None:
        if name == DENSITY:
            name = DENSITY_LABEL
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == name:
                self._tabs.setCurrentIndex(i)
                return

    # -- window templates ------------------------------------------------------------------------
    def template_settings(self) -> tuple[dict, dict]:
        """``(settings, appearance)`` for a window template, keyed as the project saves them
        (``capture_event_plots``), without the source or the frozen counts."""
        return ({"view": self.current_view(),
                 "x_metric": self._x_combo.currentData(), "y_metric": self._y_combo.currentData(),
                 "log_x": self._logx.isChecked(), "log_y": self._logy.isChecked(),
                 "bins": self._bins.value(), "colormap": self._cmap_combo.currentText(),
                 "normalize_time": self._normalize.isChecked(),
                 "align": self._align_combo.currentText(), "n_points": self._npoints.value(),
                 "pad_percent": self._pad.value(),
                 "ensemble_subset": self._ensemble_subset(),
                 "ensemble_band": self._band_kind()},
                self.appearance_state())

    def apply_template_settings(self, settings: dict, appearance: dict) -> None:
        """A recalled template, as one gesture: every control set, then each view redrawn
        at most once, and only if one of ITS settings moved.

        This window has no draft -- each of these controls redraws its view from the bound
        events as it moves -- so the recall does what setting them by hand would, once
        instead of a dozen times. A heatmap or ensemble setting that did not change does not
        redraw: that would re-bin over the zoom, and on a snapshot restored from a project
        whose recording is missing it would erase the saved count grid
        (``_render_workspace_snapshot``). The colormap, the view shown and the appearance are
        presentation only. Validated whole before any control moves.
        """
        current, _appearance = self.template_settings()
        merged = {**current, **{key: settings[key] for key in current if key in settings}}
        metrics = {self._x_combo.itemData(i) for i in range(self._x_combo.count())}
        spins = {"bins": self._bins, "n_points": self._npoints, "pad_percent": self._pad}
        if (merged["view"] not in (DENSITY, ENSEMBLE)
                or merged["x_metric"] not in metrics or merged["y_metric"] not in metrics
                or merged["colormap"] not in _COLORMAPS
                or merged["align"] not in event_shape.ENSEMBLE_ALIGNS
                or merged["ensemble_band"] not in {kind for _text, kind in _BANDS}
                or type(merged["ensemble_subset"]) is not str
                or merged["ensemble_subset"] in _SUBSET_EXCLUDED
                or any(type(merged[key]) is not bool
                       for key in ("log_x", "log_y", "normalize_time"))
                or any(type(merged[key]) is not int
                       or not spin.minimum() <= merged[key] <= spin.maximum()
                       for key, spin in spins.items())):
            raise ValueError("Its event-waveform settings are not ones this window offers.")
        previous, self._updating = self._updating, True
        try:
            self.set_x_metric(merged["x_metric"])
            self.set_y_metric(merged["y_metric"])
            self._logx.setChecked(merged["log_x"])
            self._logy.setChecked(merged["log_y"])
            self.set_bins(merged["bins"])
            self.set_colormap(merged["colormap"])
            self.set_normalize_time(merged["normalize_time"])
            self.set_align(merged["align"])
            self.set_n_points(merged["n_points"])
            self.set_pad_percent(merged["pad_percent"])
            self.set_ensemble_subset(merged["ensemble_subset"] or _SUBSET_DEFAULT)
            self.set_band(merged["ensemble_band"])
            self.set_view(merged["view"])
        finally:
            self._updating = previous
        self._align_combo.setEnabled(not merged["normalize_time"])
        # What moved, read back from the controls, each sent where its signal sends a hand
        # edit (the connections after _build_ui).
        after, _appearance = self.template_settings()
        moved = {key for key in after if after[key] != current[key]}
        if moved & {"x_metric", "y_metric", "log_x", "log_y", "bins"}:
            self._redraw_density()                  # _on_density_changed
        elif "colormap" in moved and self._density_image is not None:
            # _on_density_colormap_changed: the LUT only, never the counts or the range.
            cmap = _get_colormap(self._cmap_combo.currentText())
            self._density_image.setColorMap(cmap)
            self._colorbar.setColorMap(cmap)
        if moved & {"normalize_time", "align", "n_points", "pad_percent", "ensemble_subset"}:
            self._redraw_ensemble()                 # _on_ensemble_changed (band included)
        elif "ensemble_band" in moved:
            self._on_band_changed()                 # the band only: no kernel, zoom kept
        if "view" in moved:
            self._apply_appearance()                # _on_view_changed
        # The shared look answers through _on_appearance_changed, zoom-preserving.
        self._appearance_widget.apply_appearance(appearance)
        if moved:
            self.workspaceChanged.emit()

    def set_x_metric(self, metric_id: str) -> None:
        self._set_combo_data(self._x_combo, metric_id)

    def set_y_metric(self, metric_id: str) -> None:
        self._set_combo_data(self._y_combo, metric_id)

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, data: str) -> None:
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def set_density_log(self, x: bool, y: bool) -> None:
        self._updating = True
        self._logx.setChecked(bool(x))
        self._logy.setChecked(bool(y))
        self._updating = False
        self._on_density_changed()      # redraws, and a pin goes as it does for a click on a box

    def set_bins(self, n: int) -> None:
        self._bins.setValue(int(n))

    def set_colormap(self, name: str) -> None:
        self._cmap_combo.setCurrentText(str(name))

    def set_normalize_time(self, on: bool) -> None:
        self._normalize.setChecked(bool(on))

    def set_align(self, align: str) -> None:
        self._align_combo.setCurrentText(str(align))

    def set_n_points(self, n: int) -> None:
        self._npoints.setValue(int(n))

    def set_pad_percent(self, pct: int) -> None:
        self._pad.setValue(int(pct))

    def density_counts(self) -> np.ndarray | None:
        return None if self._density_counts is None else self._density_counts.copy()

    def density_edges(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The physical bin edges paired with :meth:`density_counts`.

        Counts alone are ambiguous, especially for logarithmic axes.  Returning owned
        copies also makes a GUI-thread project snapshot safe to serialize on a worker.
        """
        if self._density_x_edges is None or self._density_y_edges is None:
            return None
        return self._density_x_edges.copy(), self._density_y_edges.copy()

    def _render_workspace_snapshot(self) -> None:
        """Draw already-installed density/ensemble arrays without analysis kernels.

        Project restore installs the frozen numerical results directly, then calls this
        seam.  It must never call ``density_2d`` or ``event_ensemble``: reopening a project
        displays the saved analysis and only an explicit user refresh computes a new one.
        """
        self._amplitude_frozen_snapshot = _M_AMPLITUDE in (
            self._x_combo.currentData(), self._y_combo.currentData())
        self._view_shown = self.current_view()      # restore sets the tab with its signal blocked
        self._render_density_snapshot()
        self._render_ensemble_snapshot()

    def _render_density_snapshot(self) -> None:
        self._density_shown = self._density_logs = None
        self._density_main.clear()
        self._density_top.clear()
        self._density_right.clear()
        reset_plot_history(self._density_main)   # new geometry: old ranges mean nothing now
        self._density_image = None
        self._top_marginal_item = None
        self._right_marginal_item = None

        counts = self._density_counts
        xe = self._density_x_edges
        ye = self._density_y_edges
        if counts is None or xe is None or ye is None:
            self._density_message("No saved density result is available.")
            return
        counts = np.asarray(counts, dtype=np.float64)
        xe = np.asarray(xe, dtype=np.float64)
        ye = np.asarray(ye, dtype=np.float64)
        if (counts.ndim != 2 or counts.size == 0 or xe.ndim != 1 or ye.ndim != 1
                or xe.size != counts.shape[0] + 1 or ye.size != counts.shape[1] + 1):
            self._density_message("The saved density grid is incomplete or malformed.")
            return
        if not np.isfinite(counts).all() or np.any(counts < 0.0):
            self._density_message("The saved density counts are not finite and non-negative.")
            return
        if counts.sum() == 0:
            self._density_message(
                "No events fall on these axes.\nCheck the log toggles (non-positive values are "
                "dropped for a log axis).")
            return

        logx, logy = self._logx.isChecked(), self._logy.isChecked()
        try:
            xe_p = _to_plot(xe, logx)
            ye_p = _to_plot(ye, logy)
        except (FloatingPointError, TypeError, ValueError):
            self._density_message("The saved density axes cannot be rendered.")
            return
        if not np.isfinite(xe_p).all() or not np.isfinite(ye_p).all():
            self._density_message("The saved density axes cannot be rendered.")
            return

        cmap = _get_colormap(self._cmap_combo.currentText())
        cmax = float(counts.max())
        # Saved grids retain the same counts[x_bin, y_bin] contract as live results.
        img = pg.ImageItem(axisOrder="col-major")
        img.setImage(counts)
        img.setRect(QtCore.QRectF(
            float(xe_p[0]), float(ye_p[0]),
            float(xe_p[-1] - xe_p[0]), float(ye_p[-1] - ye_p[0])))
        img.setColorMap(cmap)
        img.setLevels((0.0, max(cmax, 1.0)))
        self._density_main.addItem(img)
        self._density_image = img

        self._colorbar.setColorMap(cmap)
        self._colorbar.setLevels((0.0, max(cmax, 1.0)))
        self._colorbar.setImageItem(img)

        brush = _alpha_brush(_BAR_HEX, 150)
        xc = 0.5 * (xe_p[:-1] + xe_p[1:])
        yc = 0.5 * (ye_p[:-1] + ye_p[1:])
        top_bar = pg.BarGraphItem(
            x=xc, height=counts.sum(axis=1),
            width=float(np.median(np.diff(xe_p))) * 0.96, brush=brush, pen=None)
        right_bar = pg.BarGraphItem(
            y=yc, height=float(np.median(np.diff(ye_p))) * 0.96,
            width=counts.sum(axis=0), x0=0.0, brush=brush, pen=None)
        self._density_top.addItem(top_bar)
        self._density_right.addItem(right_bar)
        self._top_marginal_item = top_bar
        self._right_marginal_item = right_bar

        x_id = self._x_combo.currentData()
        y_id = self._y_combo.currentData()
        self._density_shown, self._density_logs = (x_id, y_id), (logx, logy)
        n_events = int(getattr(self, "_nanoscope_workspace_accepted_event_count",
                               round(float(counts.sum()))))
        drawn_count = int(round(float(counts.sum())))
        count_text = (f"{drawn_count} of {n_events} accepted events"
                      if _M_AMPLITUDE in (x_id, y_id) and drawn_count != n_events
                      else f"{n_events} events")
        self._density_auto_title = (
            f"{self._metric_label(y_id)} vs {self._metric_label(x_id)}   ·   "
            f"n = {count_text}")
        self._apply_density_labels()
        self._frame(self._density_main)

    def _render_ensemble_snapshot(self) -> None:
        self._normalize_shown = None
        self._ensemble_plot.clear()
        reset_plot_history(self._ensemble_plot)  # new geometry: old ranges mean nothing now
        self._events_item = None
        self._mean_item = None
        self._median_item = None
        self._band_items = ()
        if self._ensemble is None:
            self._ensemble_message("No saved event ensemble is available.")
            return
        try:
            T, matrix, mean, median = (np.asarray(value, dtype=np.float64)
                                       for value in self._ensemble)
        except (TypeError, ValueError):
            self._ensemble_message("The saved event ensemble is malformed.")
            return
        if (T.ndim != 1 or matrix.ndim != 2 or mean.ndim != 1 or median.ndim != 1
                or matrix.shape[1] != T.size or mean.size != T.size or median.size != T.size):
            self._ensemble_message("The saved event ensemble is incomplete or malformed.")
            return
        if matrix.shape[0] == 0:
            self._ensemble_message(
                "Every accepted event is too close to the trace edge for this padding.")
            return

        gap = np.array([np.nan])
        xs = np.concatenate([np.concatenate([T, gap]) for _ in range(matrix.shape[0])])
        ys = np.concatenate([np.concatenate([row, gap]) for row in matrix])
        self._events_item = pg.PlotDataItem(
            x=xs, y=ys, connect="finite", pen=_alpha_pen(_EVENT_HEX, 60, 1))
        self._ensemble_plot.addItem(self._events_item)
        self._mean_item = pg.PlotDataItem(x=T, y=mean, pen=pg.mkPen(_MEAN_HEX, width=2.5))
        self._median_item = pg.PlotDataItem(
            x=T, y=median, pen=pg.mkPen(_MEDIAN_HEX, width=2.5, style=Qt.DashLine))
        # The saved band is drawn as saved. A snapshot that names a band but carries no
        # usable arrays (written by hand, or damaged) gets it reduced from the saved
        # matrix -- the same arithmetic over the same numbers, never the trace.
        band = self._ensemble_band
        if self._band_kind():
            try:
                band = None if band is None else tuple(
                    np.asarray(v, dtype=np.float64) for v in band)
            except (TypeError, ValueError):
                band = None
            if band is None or len(band) != 3 or any(v.shape != T.shape for v in band):
                self._compute_ensemble_band()
            else:
                self._ensemble_band = (band[0], band[1], band[2].astype(np.int64))
        else:
            self._ensemble_band = None
        self._add_band_items()
        self._ensemble_plot.addItem(self._mean_item)
        self._ensemble_plot.addItem(self._median_item)

        normalize = self._normalize.isChecked()
        self._normalize_shown = normalize
        self._ensemble_plot.setLabel(
            "bottom", "Normalized time (dwell)" if normalize else "Time (s)")
        self._ensemble_plot.setLabel("left", f"Current ({self._units})")
        self._ensemble_auto_title = self._ensemble_title(*self._ensemble_kept_and_skipped())
        self._apply_ensemble_title()
        self._refresh_ensemble_notes()
        self._frame(self._ensemble_plot)

    def ensemble_traces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        return self._ensemble

    def has_data(self) -> bool:
        return self._accepted() is not None

    # -- export ----------------------------------------------------------------------------------
    def save_image(self, path: str | None = None) -> Path | None:
        """Export the current view as PNG, SVG or PDF by extension.

        With ``path`` None publication size/format and destination dialogs are shown;
        explicit paths keep the legacy native-size contract. Returns the written path, or
        None when the user cancelled OR the write failed (a failed write is
        reported in a dialog, never returned as a path).
        """
        density = self.current_view() == DENSITY
        default = "event_density.png" if density else "event_ensemble.png"
        # The GraphicsLayout root owns the map, marginals, and colour bar.
        target = self.figure_export_target()
        export_kwargs = {}
        if path is None:
            selection = choose_live_plot_export(self, target, default)
            if selection is None:
                return None
            chosen, options = selection
            path = str(chosen)
            export_kwargs = dict(width=options.width, height=options.height,
                                 physical_width_mm=options.physical_width_mm,
                                 physical_height_mm=options.physical_height_mm)
            if options.frozen_snapshot is not None:
                export_kwargs["frozen_snapshot"] = options.frozen_snapshot
            if options.extension == ".png" and not options.native:
                export_kwargs["dpi"] = options.dpi
        out = Path(path)
        from ..plot.figures import image_write_failure_text
        try:
            notes = write_live_plot_image(out, target, **export_kwargs)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Save event-plot image",
                image_write_failure_text(out, exc, formats=IMAGE_FORMATS))
            return None
        # What the replace changed about the FILE (symlink followed, hard-linked names
        # left holding the previous contents) — unrecoverable after the fact, so it is
        # disclosed here the way File ▸ Save Project discloses it (EXP3-02).
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Save event-plot image", "\n\n".join(notes))
        return out

    def figure_export_target(self):
        return self._density_glw.ci if self.current_view() == DENSITY else self._ensemble_plot

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return list(value)
        return str(value)

    def _export_common(self, view: str) -> dict:
        provenance = dict(self._eventset.provenance) if self._eventset is not None else {}
        return {
            "view": view, "source_identity_available": bool(self._doc_id),
            "source_doc_id": self._doc_id or "",
            "source_channel": np.nan if self._channel is None else self._channel,
            "source_units": self._units,
            "provenance_json": json.dumps(
                provenance, ensure_ascii=False, default=self._json_default),
        }

    def _density_plot_data(self) -> pd.DataFrame:
        counts = self._density_counts
        xe, ye = self._density_x_edges, self._density_y_edges
        if counts is None or xe is None or ye is None or not np.asarray(counts).size:
            return pd.DataFrame()
        counts = np.asarray(counts, dtype=np.float64)
        xe = np.asarray(xe, dtype=np.float64)
        ye = np.asarray(ye, dtype=np.float64)
        x_id, y_id = self._x_combo.currentData(), self._y_combo.currentData()
        logx, logy = self._logx.isChecked(), self._logy.isChecked()
        common = {
            **self._export_common("density"), "x_metric": x_id, "y_metric": y_id,
            "log_x": logx, "log_y": logy, "bins": self._bins.value(),
            "colormap": self._cmap_combo.currentText(),
            "title": self._d_title.text(), "x_label": self._d_xlabel.text(),
            "y_label": self._d_ylabel.text(),
        }
        if _M_AMPLITUDE in (x_id, y_id):
            common["amplitude_warnings_json"] = json.dumps(self._amplitude_warnings)
            common["amplitude_units"] = self._amplitude_snapshot_units
        rows: list[dict] = []
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                rows.append({
                    "part": "density_bin", "x_bin": i, "y_bin": j,
                    "x_left": xe[i], "x_right": xe[i + 1],
                    "y_left": ye[j], "y_right": ye[j + 1],
                    "x_center": 0.5 * (xe[i] + xe[i + 1]),
                    "y_center": 0.5 * (ye[j] + ye[j + 1]),
                    "count": counts[i, j], "event_row_id": "",
                    "event_row_id_json": "", "x_value": np.nan,
                    "y_value": np.nan, "included_in_view": True, **common,
                })

        # Bin rows reproduce the visible map exactly. Source-event rows add the identities
        # behind it when the live EventSet is available; a restored orphan snapshot simply
        # leaves them absent rather than inventing positional ids.
        acc = self._accepted()
        if _M_AMPLITUDE in (x_id, y_id) and self._amplitude_frozen_snapshot:
            # Frozen bins do not prove that a rebound source still has the same
            # per-event choices. Export those bins without attaching today's rows.
            acc = None
        if acc is not None:
            x = self._metric_values(x_id, acc)
            y = self._metric_values(y_id, acc)
            keep = np.isfinite(x) & np.isfinite(y)
            if logx:
                keep &= x > 0
            if logy:
                keep &= y > 0
            event_ids = (acc.df["event_id"].tolist() if "event_id" in acc.df
                         else [None] * len(acc.df))
            recordings = (acc.df["recording"].tolist() if "recording" in acc.df
                          else [None] * len(acc.df))
            pore_ids = (acc.df["pore_id"].tolist() if "pore_id" in acc.df
                        else [None] * len(acc.df))
            for pos, (row_id, event_id, recording, pore_id, xv, yv, kept) in enumerate(zip(
                    acc.df.index, event_ids, recordings, pore_ids, x, y, keep)):
                amplitude_audit = {}
                if _M_AMPLITUDE in (x_id, y_id):
                    from ..analysis.event_amplitude import AMPLITUDE_COLUMNS
                    amplitude_audit = {"event_" + c: acc.df.iloc[pos][c]
                                       for c in AMPLITUDE_COLUMNS if c in acc.df}
                rows.append({
                    "part": "source_event", "x_bin": np.nan, "y_bin": np.nan,
                    "x_left": np.nan, "x_right": np.nan, "y_left": np.nan,
                    "y_right": np.nan, "x_center": np.nan, "y_center": np.nan,
                    "count": np.nan, "event_row_id": row_id,
                    "event_row_id_json": json.dumps(
                        row_id, ensure_ascii=False, default=self._json_default),
                    "event_id": event_id, "event_recording": recording,
                    "event_pore_id": pore_id,
                    "x_value": xv, "y_value": yv,
                    "included_in_view": bool(kept), **common, **amplitude_audit,
                })
        return pd.DataFrame(rows)

    def _ensemble_kept_identities(self, n_rows: int) -> list[dict]:
        """Source identities for ensemble rows in exact input POSITION order.

        Dataframe indices are labels, not positions, and need not be unique.  A dict
        keyed by index therefore maps two same-labelled rows onto one event.  The
        ensemble kernel walks accepted rows positionally; replay that keep mask and
        use ``iloc`` so even duplicate indices retain their own event_id/recording.
        """
        acc = self._ensemble_events()
        if acc is None or self._signal is None:
            return []
        df = acc.df
        starts = df["start_idx"].to_numpy(dtype=np.int64)
        ends = df["end_idx"].to_numpy(dtype=np.int64)
        lengths = ends - starts
        n = int(self._signal.size)
        pad_frac = float(self._pad.value()) / 100.0
        valid = lengths > 0
        if self._normalize.isChecked():
            valid &= (starts - pad_frac * lengths >= 0.0)
            valid &= (ends + pad_frac * lengths <= n)
        elif np.any(valid):
            align = self._align_combo.currentText()
            refs = None
            if align == "extremum":
                if self._ensemble_alignment is None:
                    return []       # a snapshot that never recorded its polarity
                refs = event_shape.extremum_indices(
                    self._signal, starts, ends, self._ensemble_alignment[0])
            # The kernel's own window arithmetic, so the replay cannot drift from it.
            geometry = event_shape.realtime_window(starts, ends, pad_frac, align, refs)
            if geometry is None:
                return []
            a_all, total, _ref, valid = geometry
            valid = valid & (a_all >= 0) & (a_all + total <= n)
        positions = np.flatnonzero(valid)
        if positions.size != int(n_rows):
            return []
        identities: list[dict] = []
        for position in positions:
            row = df.iloc[int(position)]
            identities.append({
                "event_row_id": df.index[int(position)],
                "event_id": row.get("event_id") if "event_id" in df.columns else None,
                "event_recording": (row.get("recording")
                                    if "recording" in df.columns else None),
                "event_pore_id": row.get("pore_id") if "pore_id" in df.columns else None,
            })
        return identities

    def _ensemble_kept_row_ids(self, n_rows: int) -> list[object]:
        """Backward-compatible source-index accessor; mapping is positional."""
        return [identity["event_row_id"]
                for identity in self._ensemble_kept_identities(n_rows)]

    def _ensemble_plot_data(self) -> pd.DataFrame:
        if self._ensemble is None:
            return pd.DataFrame()
        time, matrix, mean, median = (np.asarray(value) for value in self._ensemble)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return pd.DataFrame()
        row_count = int(matrix.shape[0]) * int(matrix.shape[1])
        require_csv_row_budget(
            row_count, limit=MAX_ENSEMBLE_CSV_ROWS,
            description=(f"event ensemble ({matrix.shape[0]:,} events × "
                         f"{matrix.shape[1]:,} points)"),
            advice=("Lower Ensemble › Points, or curate a smaller accepted event set, "
                    "then export again."))
        identities = self._ensemble_kept_identities(matrix.shape[0])
        identity_available = len(identities) == int(matrix.shape[0])
        # The counts the title gives, from the input size frozen with the draw: the live
        # table may be absent (a reopened project) or curated since, and neither changes
        # how many events this figure left out.
        _drawn, skipped = self._ensemble_kept_and_skipped()
        edge_skipped, unalignable = self._ensemble_skips(skipped)
        # The band columns are written whether or not a band is chosen: band_kind says
        # which, and n_events_at_point is the n behind the mean and median on the same
        # row in either case -- every drawn event, or 0 where those are NaN because some
        # event's sample there is not finite (the band is NaN there with them).
        kind = self._band_kind()
        band = self._ensemble_band if kind else None
        if band is not None:
            band_lower, band_upper, n_at_point = (np.asarray(v) for v in band)
        else:
            band_lower = band_upper = np.full(matrix.shape[1], np.nan)
            n_at_point = event_shape.ensemble_point_counts(matrix)
        alignment = self._ensemble_alignment
        extremum = (not self._normalize.isChecked()
                    and self._align_combo.currentText() == "extremum")
        common = {
            **self._export_common("ensemble"),
            "normalize_time": self._normalize.isChecked(),
            "align": self._align_combo.currentText(), "n_points": self._npoints.value(),
            "pad_percent": self._pad.value(), "edge_skipped_count": edge_skipped,
            "no_finite_sample_count": unalignable,
            "ensemble_subset": self._ensemble_subset(),
            "align_polarity": (alignment[0] if extremum and alignment else np.nan),
            "align_polarity_source": (alignment[1] if extremum and alignment else ""),
            "band_kind": {"sd": "mean_pm_1sd", "iqr": "median_iqr_25_75"}.get(kind, "none"),
        }
        rows: list[dict] = []
        for trace_i, trace in enumerate(matrix):
            identity = identities[trace_i] if identity_available else {}
            row_id = identity.get("event_row_id")
            row_json = (json.dumps(row_id, ensure_ascii=False, default=self._json_default)
                        if identity_available else "")
            for sample_i, (tv, value) in enumerate(zip(time, trace)):
                rows.append({
                    "part": "event_trace", "trace_index": trace_i,
                    "event_row_id": row_id,
                    "event_row_id_json": row_json,
                    "identity_available": identity_available,
                    "event_id": identity.get("event_id"),
                    "event_recording": identity.get("event_recording"),
                    "event_pore_id": identity.get("event_pore_id"),
                    "sample_index": sample_i, "time": tv, "event_current": value,
                    "mean_current": mean[sample_i], "median_current": median[sample_i],
                    "band_lower": band_lower[sample_i], "band_upper": band_upper[sample_i],
                    "n_events_at_point": int(n_at_point[sample_i]),
                    **common,
                })
        return pd.DataFrame(rows)

    def _plot_data(self) -> pd.DataFrame:
        """The exact numeric values behind the currently visible event-plot view."""
        return (self._density_plot_data() if self.current_view() == DENSITY
                else self._ensemble_plot_data())

    def export_data(self, path: str | None = None) -> Path | None:
        """Export the current density or ensemble view and provenance as staged CSV."""
        density = self.current_view() == DENSITY
        default = "event_density.csv" if density else "event_ensemble.csv"
        if path is None:
            chosen = ask_export_path(
                self, title="Export event-plot data", suggestion=default,
                filters=DATA_FILTERS)
            if chosen is None:
                return None
            path = str(chosen)
        out = Path(path)
        try:
            notes = write_plot_csv(out, self._plot_data())
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(
                self, "Export event-plot data", data_write_failure_text(out, exc))
            return None
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Export event-plot data", "\n\n".join(notes))
        return out


# ================================================================================================
# small pyqtgraph helpers
# ================================================================================================

def _subset_text(mode: str) -> str:
    """The Events combo's text for a curation filter: every subset is WITHIN accepted."""
    return "All accepted" if mode == _SUBSET_DEFAULT else f"Accepted · {mode}"


def _count_unalignable(signal: np.ndarray, events: EventSet, polarity: int) -> int:
    """Events an extremum alignment drops for having no finite sample in their own span.

    The kernel drops them through the same ``extremum_indices`` (``-1``: nothing to align
    on). Only spans that lie at least partly on the trace count: an event wholly off it
    has no samples at all and is an edge skip, as it is for start and centre.
    """
    df = events.accepted.df
    n = int(np.asarray(signal).size)
    starts = np.clip(df["start_idx"].to_numpy(dtype=np.int64), 0, n)
    ends = np.clip(df["end_idx"].to_numpy(dtype=np.int64), 0, n)
    refs = event_shape.extremum_indices(signal, starts, ends, polarity)
    return int(np.count_nonzero((ends > starts) & (refs < 0)))


def _get_colormap(name: str) -> pg.ColorMap:
    """A pyqtgraph ColorMap by name, via matplotlib (a hard dependency), viridis on failure."""
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:
        return pg.colormap.get("viridis", source="matplotlib")


def _to_plot(edges: np.ndarray, log: bool) -> np.ndarray:
    """Edges in the coordinate the image/axis is drawn in: log10 when the axis is logged."""
    return np.log10(edges) if log else np.asarray(edges, dtype=np.float64)


def _alpha_brush(hex_color: str, alpha: int) -> QtGui.QBrush:
    c = QtGui.QColor(hex_color)
    c.setAlpha(int(alpha))
    return pg.mkBrush(c)


def _alpha_pen(hex_color: str, alpha: int, width: int) -> QtGui.QPen:
    c = QtGui.QColor(hex_color)
    c.setAlpha(int(alpha))
    return pg.mkPen(c, width=width)
