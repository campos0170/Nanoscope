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

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..analysis import event_shape
from ..analysis.events import EventSet
from ..core.session import Session
from ..plot.style import OKABE_ITO
from .export_paths import ask_export_path

__all__ = ["EventPlotWindow"]

# View labels (also the QTabWidget page order).
DENSITY = "Density"
ENSEMBLE = "Ensemble"

# Metric ids (X/Y choices for the density map). ``ecd`` is computed on demand.
_M_DWELL = "dwell_s"
_M_DEPTH = "depth_pA"
_M_FRAC = "frac_depth"
_M_ECD = "ecd"
_METRICS: tuple[tuple[str, str], ...] = (
    (_M_DWELL, "Dwell (s)"),
    (_M_DEPTH, "Depth"),
    (_M_FRAC, "dI/I₀"),
    (_M_ECD, "ECD (charge deficit)"),
)

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

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("NanoScope — Event plots")
        self.resize(1040, 760)

        # Pushed-in source state.
        self._eventset: EventSet | None = None
        self._signal: np.ndarray | None = None
        self._fs: float | None = None
        self._units: str = "pA"

        # Live results (kept for tests + save).
        self._density_counts: np.ndarray | None = None
        self._density_image: pg.ImageItem | None = None
        self._top_marginal_item: pg.BarGraphItem | None = None
        self._right_marginal_item: pg.BarGraphItem | None = None
        self._ensemble: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._events_item: pg.PlotDataItem | None = None
        self._mean_item: pg.PlotDataItem | None = None
        self._median_item: pg.PlotDataItem | None = None

        self._updating = False
        self._build_ui()
        self._redraw()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._build_density_page(), DENSITY)
        self._tabs.addTab(self._build_ensemble_page(), ENSEMBLE)
        self.setCentralWidget(self._tabs)

        # Refresh lives in the toolbar so both views share it.
        bar = QtWidgets.QToolBar()
        bar.setMovable(False)
        self.addToolBar(bar)
        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip("Re-snapshot the active detection's accepted events.")
        self._refresh_btn.clicked.connect(self.refreshRequested.emit)
        bar.addWidget(self._refresh_btn)

    def _build_density_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        row1 = QtWidgets.QHBoxLayout()
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

        row1.addWidget(QtWidgets.QLabel("Colormap"))
        self._cmap_combo = QtWidgets.QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setToolTip("Sequential colour map for the density (perceptually uniform).")
        row1.addWidget(self._cmap_combo)
        row1.addStretch(1)
        root.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self._d_title = QtWidgets.QLineEdit()
        self._d_title.setPlaceholderText("title (auto)")
        self._d_xlabel = QtWidgets.QLineEdit()
        self._d_xlabel.setPlaceholderText("x label (auto)")
        self._d_ylabel = QtWidgets.QLineEdit()
        self._d_ylabel.setPlaceholderText("y label (auto)")
        row2.addWidget(QtWidgets.QLabel("Title"))
        row2.addWidget(self._d_title, 1)
        row2.addWidget(QtWidgets.QLabel("X label"))
        row2.addWidget(self._d_xlabel, 1)
        row2.addWidget(QtWidgets.QLabel("Y label"))
        row2.addWidget(self._d_ylabel, 1)
        self._d_save = QtWidgets.QPushButton("Save image…")
        self._d_save.setToolTip("Export the density map (with marginals) as a PNG.")
        self._d_save.clicked.connect(lambda: self.save_image())
        row2.addWidget(self._d_save)
        root.addLayout(row2)

        # The density scene: top marginal + main image + right marginal + colour bar.
        self._density_glw = pg.GraphicsLayoutWidget()
        self._density_glw.setMinimumSize(560, 460)
        self._density_top = self._density_glw.addPlot(row=0, col=0)
        self._density_main = self._density_glw.addPlot(row=1, col=0)
        self._density_right = self._density_glw.addPlot(row=1, col=1)
        self._colorbar = pg.ColorBarItem(values=(0, 1), colorMap=_get_colormap("viridis"))
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

        row1 = QtWidgets.QHBoxLayout()
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
        root.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        self._e_title = QtWidgets.QLineEdit()
        self._e_title.setPlaceholderText("title (auto)")
        row2.addWidget(QtWidgets.QLabel("Title"))
        row2.addWidget(self._e_title, 1)
        self._e_save = QtWidgets.QPushButton("Save image…")
        self._e_save.setToolTip("Export the event ensemble as a PNG.")
        self._e_save.clicked.connect(lambda: self.save_image())
        row2.addWidget(self._e_save)
        root.addLayout(row2)

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
        self._cmap_combo.currentTextChanged.connect(self._on_density_changed)
        self._d_title.textChanged.connect(self._on_density_changed)
        self._d_xlabel.textChanged.connect(self._on_density_changed)
        self._d_ylabel.textChanged.connect(self._on_density_changed)
        self._normalize.toggled.connect(self._on_normalize_toggled)
        self._align_combo.currentTextChanged.connect(self._on_ensemble_changed)
        self._npoints.valueChanged.connect(self._on_ensemble_changed)
        self._pad.valueChanged.connect(self._on_ensemble_changed)
        self._e_title.textChanged.connect(self._on_ensemble_changed)
        return page

    # -- data in ---------------------------------------------------------------------------------
    def set_source(self, doc_id: str | None, channel: int | None,
                   eventset: EventSet | None) -> None:
        """Install the active detection's events + its calibrated signal, then redraw.

        Pass the FULL detection EventSet (the window filters to its ACCEPTED events
        itself, mirroring PlotsWindow). ``None`` / an all-rejected set clears the
        window to its empty state. The calibrated signal is pulled from the Session
        for ``(doc_id, channel)`` -- the same trace the events were detected on.
        """
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
                self._eventset = eventset
                self._signal = np.asarray(ch.data)
                self._fs = float(ch.sample_rate_hz)
                self._units = ch.units or "pA"
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
        return acc.df[metric_id].to_numpy(dtype=np.float64)

    def _metric_label(self, metric_id: str) -> str:
        base = dict(_METRICS)[metric_id]
        if metric_id == _M_DEPTH:
            return f"Depth ({self._units})"
        if metric_id == _M_ECD:
            return f"ECD ({self._units}·s)"
        return base

    # -- redraw dispatch -------------------------------------------------------------------------
    def _on_density_changed(self, *_a) -> None:
        if not self._updating:
            self._redraw_density()

    def _on_ensemble_changed(self, *_a) -> None:
        if not self._updating:
            self._redraw_ensemble()

    def _on_normalize_toggled(self, on: bool) -> None:
        self._align_combo.setEnabled(not on)   # align only bites in real-time mode
        self._on_ensemble_changed()

    def _redraw(self) -> None:
        self._redraw_density()
        self._redraw_ensemble()

    # -- density ---------------------------------------------------------------------------------
    def _redraw_density(self) -> None:
        self._density_main.clear()
        self._density_top.clear()
        self._density_right.clear()
        self._density_image = None
        self._top_marginal_item = None
        self._right_marginal_item = None
        self._density_counts = None

        acc = self._accepted()
        if acc is None:
            self._density_message(
                "No accepted events.\nRun detection and accept some events, then Refresh.")
            return

        x_id = self._x_combo.currentData()
        y_id = self._y_combo.currentData()
        logx, logy = self._logx.isChecked(), self._logy.isChecked()
        x = self._metric_values(x_id, acc)
        y = self._metric_values(y_id, acc)
        nb = int(self._bins.value())
        counts, xe, ye = event_shape.density_2d(
            x, y, xbins=nb, ybins=nb, xlog=logx, ylog=logy)
        self._density_counts = counts
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

        xlabel = self._d_xlabel.text().strip() or (
            self._metric_label(x_id) + (" (log10)" if logx else ""))
        ylabel = self._d_ylabel.text().strip() or (
            self._metric_label(y_id) + (" (log10)" if logy else ""))
        self._density_main.setLabel("bottom", xlabel)
        self._density_main.setLabel("left", ylabel)
        title = self._d_title.text().strip() or (
            f"{self._metric_label(y_id)} vs {self._metric_label(x_id)}   ·   "
            f"n = {len(acc.df)} events")
        self._density_main.setTitle(title)
        self._density_main.autoRange()

    def _density_message(self, text: str) -> None:
        self._density_main.setTitle(None)
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
        auto = (f"Ensemble of {matrix.shape[0]} event(s)"
                + (f"   ·   {skipped} edge-skipped" if skipped else "")
                + "   —   mean (solid) + median (dashed)")
        self._ensemble_plot.setTitle(self._e_title.text().strip() or auto)
        self._ensemble_plot.autoRange()

    def _ensemble_message(self, text: str) -> None:
        self._ensemble_plot.setTitle(None)
        self._ensemble_plot.setLabel("bottom", "")
        self._ensemble_plot.setLabel("left", "")
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=_MSG_HEX)
        self._ensemble_plot.addItem(item)
        self._ensemble_plot.setXRange(0, 1, padding=0)
        self._ensemble_plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)

    # -- public accessors (owner + tests) --------------------------------------------------------
    def current_view(self) -> str:
        return self._tabs.tabText(self._tabs.currentIndex())

    def set_view(self, name: str) -> None:
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

    def ensemble_traces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        return self._ensemble

    def has_data(self) -> bool:
        return self._accepted() is not None

    # -- export ----------------------------------------------------------------------------------
    def save_image(self, path: str | None = None) -> Path | None:
        """Export the CURRENT view (density scene or ensemble plot) as a PNG.

        With ``path`` None a file dialog is shown; returns the written path, or
        None when the user cancelled OR the write failed (a failed write is
        reported in a dialog, never returned as a path).
        """
        density = self.current_view() == DENSITY
        default = "event_density.png" if density else "event_ensemble.png"
        if path is None:
            chosen = ask_export_path(
                self, title="Save event-plot image", suggestion=default,
                filters="PNG image (*.png);;All files (*)")   # writes exactly this file
            if chosen is None:
                return None
            path = str(chosen)
        target = self._density_glw.scene() if density else self._ensemble_plot
        out = Path(path)
        # STAGED, like every other export (USER_GUIDE §12). pyqtgraph's ImageExporter
        # ends in ``QImage.save``, which OPENS AND TRUNCATES its target and reports
        # failure by RETURNING False — writing ``out`` directly destroyed the previous
        # export for any format QImage cannot write and then said "The file was not
        # created". Rendering into a staged sibling costs the user nothing on failure,
        # which is what lets the message below say so truthfully (EXP3-06).
        from ..plot.figures import image_write_failure_text, save_plot_image

        exporter = ImageExporter(target)
        try:
            notes = save_plot_image(out, lambda staged: exporter.export(str(staged)))
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Save event-plot image", image_write_failure_text(out, exc))
            return None
        # What the replace changed about the FILE (symlink followed, hard-linked names
        # left holding the previous contents) — unrecoverable after the fact, so it is
        # disclosed here the way File ▸ Save Project discloses it (EXP3-02).
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Save event-plot image", "\n\n".join(notes))
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
