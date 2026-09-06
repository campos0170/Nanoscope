"""The standalone event-plots window: the density map + the event ensemble.

Opened from ``Analysis ▸ Event plots (density / ensemble)…``, this is the GUI
face of :mod:`nanoscope.analysis.event_shape` -- the two figures every
translocation paper leads with:

* **Density** -- the 2-D dwell-vs-blockage heatmap, an X metric against a Y metric
  binned into a colour-mapped image with MARGINAL histograms along the top (X) and
  right (Y). X and Y each choose among ``dwell_s`` (default X, log by default),
  ``depth_pA`` (default Y), ``frac_depth`` and the on-demand ECD / charge deficit.
* **Ensemble** -- every accepted event resampled onto a common axis and overlaid
  (faint), with the bold mean and median waveform drawn on top.

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
from ..analysis.events import EventSet
from ..core.session import Session
from ..core.sources import amplitude_comparability, amplitude_units_label
from ..plot.style import OKABE_ITO
from .export_paths import ask_export_path
from .plot_appearance import PlotAppearance, PlotAppearanceWidget, apply_plot_appearance
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

_COLORMAPS = ("viridis", "magma", "plasma", "inferno", "cividis")

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
        self.setWindowTitle("NanoScope — Event plots")
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

        self._updating = False
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
        self._appearance_widget = PlotAppearanceWidget(
            self._appearance, include_colors=True, include_markers=False,
            multi_view_labels=True)
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
        bar.hide()  # legacy programmatic toggle remains available; workflow owns visible chrome

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
        if not self._updating:
            self._apply_appearance()

    def appearance_state(self) -> dict:
        return self._appearance.to_dict()

    def _apply_appearance(self) -> None:
        apply_plot_appearance(self._density_main, self._appearance, plot_widget=self._density_glw)
        marginal = replace(self._appearance, title=None, x_label=None, y_label=None)
        for plot, item in ((self._density_top, self._top_marginal_item),
                           (self._density_right, self._right_marginal_item)):
            apply_plot_appearance(plot, marginal, data_items=(item,))
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
            self._ensemble_plot, self._appearance, plot_widget=self._ensemble_widget,
            data_items=(self._events_item,), fit_items=(self._mean_item, self._median_item))

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
        self._colorbar = pg.ColorBarItem(values=(0, 1), colorMap=_get_colormap("viridis"), label="Event count")
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
        row1 = QtWidgets.QHBoxLayout(self._ensemble_analysis_controls)
        row1.setContentsMargins(0, 0, 0, 0)
        self._normalize = QtWidgets.QCheckBox("Normalize time")
        self._normalize.setChecked(True)
        self._normalize.setToolTip(
            "Duration-normalize each event onto a common [0, 1] axis (align is then immaterial).")
        row1.addWidget(self._normalize)

        row1.addWidget(QtWidgets.QLabel("Align"))
        self._align_combo = QtWidgets.QComboBox()
        self._align_combo.addItems(["start", "center"])
        self._align_combo.setEnabled(False)     # only meaningful when NOT normalizing time
        self._align_combo.setToolTip(
            "Real-time alignment reference (used only when 'Normalize time' is off).")
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
            "Export individual resampled traces, mean, median, and event identities.")
        self._e_data.clicked.connect(lambda: self.export_data())
        row2.addWidget(self._e_data)
        root.addWidget(self._ensemble_export_controls)

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
        if defer_analysis:
            # Never leave another project's result available behind a newly bound
            # source if the artifact later proves malformed.  Clearing Qt items and
            # cached arrays is presentation bookkeeping, not an analysis redraw.
            self._density_counts = None
            self._density_x_edges = None
            self._density_y_edges = None
            self._ensemble = None
            self._density_main.clear()
            self._density_top.clear()
            self._density_right.clear()
            self._ensemble_plot.clear()
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
            self._redraw_density()
            self.workspaceChanged.emit()

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
        if not self._updating:
            self._apply_appearance()
            self.workspaceChanged.emit()

    def _on_normalize_toggled(self, on: bool) -> None:
        self._align_combo.setEnabled(not on)   # align only bites in real-time mode
        self._on_ensemble_changed()

    def _redraw(self) -> None:
        self._redraw_density()
        self._redraw_ensemble()

    # -- density ---------------------------------------------------------------------------------
    def _redraw_density(self) -> None:
        self._amplitude_frozen_snapshot = False
        self._amplitude_warnings = ()
        self._amplitude_warning_label.hide()
        self._amplitude_warning_summary.set_warnings(())
        self._density_main.clear()
        self._density_top.clear()
        self._density_right.clear()
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

        img = pg.ImageItem()
        # ImageItem is col-major by default: image[i, j] -> X-bin i, Y-bin j, exactly counts2d.
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

        self._density_auto_title = (
            f"{self._metric_label(y_id)} vs {self._metric_label(x_id)}   ·   "
            f"n = {len(acc.df)} events")
        if _M_AMPLITUDE in (x_id, y_id):
            self._density_auto_title = (
                f"{self._metric_label(y_id)} vs {self._metric_label(x_id)}   ·   "
                f"n = {int(counts.sum())} of {len(acc.df)} accepted events")
        self._apply_density_labels()
        self._density_main.autoRange()

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
        self._ensemble_plot.clear()
        self._events_item = None
        self._mean_item = None
        self._median_item = None
        self._ensemble = None

        acc = self._accepted()
        if acc is None:
            self._ensemble_message(
                "No accepted events.\nRun detection and accept some events, then Refresh.")
            return

        normalize = self._normalize.isChecked()
        align = self._align_combo.currentText()
        npoints = int(self._npoints.value())
        pad_frac = float(self._pad.value()) / 100.0
        T, matrix, mean, median = event_shape.event_ensemble(
            self._signal, self._fs, acc, n_points=npoints, pad_frac=pad_frac,
            align=align, normalize_time=normalize)
        self._ensemble = (T, matrix, mean, median)

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
        self._ensemble_plot.addItem(self._mean_item)
        self._ensemble_plot.addItem(self._median_item)

        n_acc = len(acc.df)
        skipped = n_acc - matrix.shape[0]
        xlabel = "Normalized time (dwell)" if normalize else "Time (s)"
        self._ensemble_plot.setLabel("bottom", xlabel)
        self._ensemble_plot.setLabel("left", f"Current ({self._units})")
        self._ensemble_auto_title = (f"Ensemble of {matrix.shape[0]} event(s)"
                + (f"   ·   {skipped} edge-skipped" if skipped else "")
                + "   —   mean (solid) + median (dashed)")
        self._apply_ensemble_title()
        self._ensemble_plot.autoRange()

    def _ensemble_message(self, text: str) -> None:
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
        self._redraw_density()
        self.workspaceChanged.emit()

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
        self._render_density_snapshot()
        self._render_ensemble_snapshot()

    def _render_density_snapshot(self) -> None:
        self._density_main.clear()
        self._density_top.clear()
        self._density_right.clear()
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
        img = pg.ImageItem()
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
        self._density_main.autoRange()

    def _render_ensemble_snapshot(self) -> None:
        self._ensemble_plot.clear()
        self._events_item = None
        self._mean_item = None
        self._median_item = None
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
        self._ensemble_plot.addItem(self._mean_item)
        self._ensemble_plot.addItem(self._median_item)

        normalize = self._normalize.isChecked()
        self._ensemble_plot.setLabel(
            "bottom", "Normalized time (dwell)" if normalize else "Time (s)")
        self._ensemble_plot.setLabel("left", f"Current ({self._units})")
        n_accepted = int(getattr(self, "_nanoscope_workspace_accepted_event_count",
                                 matrix.shape[0]))
        skipped = max(0, n_accepted - matrix.shape[0])
        self._ensemble_auto_title = (f"Ensemble of {matrix.shape[0]} event(s)"
                + (f"   ·   {skipped} edge-skipped" if skipped else "")
                + "   —   mean (solid) + median (dashed)")
        self._apply_ensemble_title()
        self._ensemble_plot.autoRange()

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
        acc = self._accepted()
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
            width = int(lengths[valid].max())
            wpad = int(round(pad_frac * width))
            total = width + 2 * wpad
            if self._align_combo.currentText() == "center":
                a_all = np.round((starts + ends) / 2.0 - total / 2.0).astype(np.int64)
            else:
                a_all = starts - wpad
            valid &= (a_all >= 0) & (a_all + total <= n)
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
        skipped = max(0, (len(self._accepted().df) if self._accepted() is not None else
                          matrix.shape[0]) - matrix.shape[0])
        common = {
            **self._export_common("ensemble"),
            "normalize_time": self._normalize.isChecked(),
            "align": self._align_combo.currentText(), "n_points": self._npoints.value(),
            "pad_percent": self._pad.value(), "edge_skipped_count": skipped,
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
