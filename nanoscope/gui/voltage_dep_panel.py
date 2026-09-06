"""The standalone voltage-dependence window: the classic nanopore voltage series.

Opened from ``Analysis ▸ Voltage dependence…``, this window gathers one
``(voltage, metric)`` point from every open recording that has BOTH a known
applied voltage AND a stored detection (:func:`nanoscope.analysis.voltage_dep.vdep_points`),
scatters the chosen event metric against the applied voltage, overlays a fitted
trend (a straight line, or the ``A·exp(V/V0)`` capture-rate model), and reports
the recovered parameters, R² and n alongside a per-recording table. It is the
aggregate the user builds after running :class:`batch detection
<nanoscope.gui.analysis_worker.BatchDetectController>` across their
-1000..-200 mV runs: dial the metric (capture rate, dwell, depth, dI/I0, count),
the fit (none / linear / exponential), and the axis conventions (|V| or signed V,
log or linear y) and read how the physics moves with voltage.

Design mirrors :class:`~nanoscope.gui.iv_panel.IVWindow`: a single, lazily-built
top-level window owned by :class:`~nanoscope.gui.main_window.MainWindow`, rendered
with pyqtgraph so it inherits the app's dark theme. Its data is the open documents
plus their detection results, so :meth:`refresh` re-gathers straight from the
:class:`~nanoscope.core.session.Session` (recordings may have been detected since),
and a *Refresh* button and *Save image* export sit alongside the plot.

The fit runs on whatever x-axis is currently plotted (|V| or signed V), so its
overlaid curve always lands on the scatter; log-y is offered because the capture
rate is exponential in voltage and reads as a straight line on a log-count axis.

WHAT WAS POOLED IS ON SCREEN, AND IN THE EXPORTED PNG
-----------------------------------------------------
This window is the only place several recordings are put on one axis, so it is
the only place that has to show what it pooled. It never calls
:func:`~nanoscope.analysis.voltage_dep.vdep_points`: it calls
:func:`~nanoscope.analysis.voltage_dep.vdep_series`, which hands back the points
AND the verdict together, so there is no way to render a trend here without
holding the verdict on it. Three surfaces carry it:

* a warning banner above the plot, shown only when there is something to say --
  different baseline windows under an amplitude metric, a merged set pooled with an
  unmerged one, an uncalibrated gain under an absolute-current metric, or a pooled
  detection carrying its own caveat -- naming the groups and what to do;
* the plot TITLE, which is what an exported PNG keeps: the merge state always, the
  baseline fingerprint whenever the metric is measured against the baseline, and the
  CALIBRATION whenever the metric is an absolute current. A saved ``V0 = 0.441 V``
  now says on its face that it is the merged number, and a saved
  ``V₀ = -0.432 V`` over ``Median depth (pA)`` says ``gain uncalibrated``;
* the per-recording table's *Baseline*, *Merge* and *Gain* columns, and a "Pooled
  provenance" block in the details pane.

TWO KINDS OF HONESTY, AND THEY ARE NOT THE SAME KIND
-----------------------------------------------------
The comparability verdict is about the POOL: these recordings may not be put on one
axis. A detection caveat is about ONE RECORDING: this point is biased. The window
shows both and does not conflate them -- the caveat block never claims the pool is
incomparable, and the ⚠ on the fit headline still means only "the pool is not
comparable". Before 2026-08-22 the caveats reached the Summary panel and the Plots
details pane and stopped there, so the −700 mV recording's *"the tracked baseline
sits 4.20 pA INSIDE the states … a voltage series measured this way puts one
recording out of line with the rest"* never reached the window that plots the
voltage series.

None of it hides a point or refuses a fit. An incomparable pool is still drawn --
it is simply no longer drawn silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

from ..analysis import stats, voltage_dep
from ..core.session import Session
from ..core.sources import gain_label
from ..plot.style import OKABE_ITO
from .combos import choose_combo_item
from .export_paths import ask_export_path
from .plot_appearance import PlotAppearance, PlotAppearanceWidget, apply_plot_appearance
from .plot_workflow import (WarningSummary, bounded_controls, control_group,
                            ensure_item_legend, install_plot_workflow)
from .plot_export import (DATA_FILTERS, IMAGE_FILTERS, IMAGE_FORMATS,
                          choose_live_plot_export, data_write_failure_text, write_live_plot_image,
                          write_plot_csv)

__all__ = ["VDepWindow"]

_DATA_HEX = OKABE_ITO[0]        # blue -- the (V, metric) points
_FIT_HEX = OKABE_ITO[1]         # vermillion -- the fitted trend
_MSG_HEX = "#8b9199"
_WARN_HEX = OKABE_ITO[4]        # orange -- the not-comparable banner
_POINT_ALPHA = 200

# The banner that carries a comparability warning. Styled loud on purpose: the
# defect this replaced was a silently pooled, silently wrong trend.
_WARN_CSS = (
    "color:#0d0f12; background:%s; border-radius:4px; padding:6px 8px; "
    "font-family: Menlo, monospace; font-size: 11px;" % _WARN_HEX
)

#: Tallest the banner may grow before it scrolls instead. ~9 lines at 11 px: enough
#: for a three-recording baseline disagreement and the gain caution without the plot
#: below it losing more than a slice, and the rest is one flick of the wheel away
#: (and, whole, in the details pane).
_WARN_MAX_H = 150

#: Character budget per line of the plot TITLE before it wraps to the next one.
#: The title carries the fit AND everything qualifying it, and at the window's
#: default 880 px the pyqtgraph title CLIPS at about 95 characters -- measured: the
#: three-recording as-probed title ran to 133 and lost its last segment, i.e. the
#: exported PNG silently dropped part of the very statement the title exists to
#: keep. Wrapped, not shortened: nothing is worth dropping.
_TITLE_WRAP = 72

# Metric combo label -> voltage_dep metric id (order defines the combo order).
#
# "Median depth (pA)" carries its unit like every other dimensioned metric here.
# It was the ONE metric with no unit anywhere -- combo, axis, table header or
# details -- for a column of picoamps, on the window whose whole job is putting
# several pores' picoamps on one axis. The unit is also what makes the gain
# CAUTION legible: "not comparable across recordings" means something specific
# once the reader can see the number is an absolute current.
_METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("Capture rate (Hz)", "capture_rate_hz"),
    ("Median dwell (s)", "median_dwell_s"),
    ("Median depth (pA)", "median_depth"),
    ("Median dI/I₀", "median_frac_depth"),
    ("Event count", "n_events"),
)
_METRIC_ID = dict(_METRIC_LABELS)
_METRIC_LABEL = {v: k for k, v in _METRIC_LABELS}

# Metric id -> y-axis label (the metric's physical name/units). Identical to the
# combo labels by construction: the axis, the combo, the table header and the
# details pane are four renderings of one metric and must not name it four ways.
_Y_LABEL: dict[str, str] = dict(_METRIC_LABEL)

# Fit combo labels (also the voltage_dep model ids, plus "none").
_FIT_NONE = "none"
_FIT_LINEAR = "linear"
_FIT_EXP = "exponential"

#: The per-recording table's columns. Index 2 is filled with the metric's own label
#: at fill time (the axis, the combo and this header are one metric named once).
_TABLE_COLUMNS: tuple[str, ...] = (
    "Recording", "V (mV)", "metric", "V source", "Duration (s)",
    "Baseline", "Merge", "Gain")
_COL_RECORDING = 0
_COL_VOLTAGE = 1
_COL_METRIC = 2
_COL_V_SOURCE = 3
_COL_DURATION = 4
_COL_BASELINE = 5
_COL_MERGE = 6
_COL_GAIN = 7

#: How a row that is on the plot but NOT in the fit is marked, in the sibling I-V
#: window's own words (``iv_panel._finish_details``: "— DROPPED, not in the fit").
#: One app, one phrase for one fact.
_NOT_IN_FIT_MARK = "  — DROPPED, not in the fit"

#: Free parameters in EITHER trend: (slope, intercept) for the linear one,
#: (A, V0) for the exponential one. What
#: :func:`nanoscope.analysis.stats.r2_dof_tag` subtracts from the recording count to
#: decide whether the printed R² has any residual degrees of freedom left to be
#: evidence over. The same constant the Plots scatter keeps for its own two lines.
_VDEP_FIT_PARAMS = 2


class VDepWindow(QtWidgets.QMainWindow):
    """Scatter an event metric vs applied voltage and fit the voltage trend.

    Fed nothing directly: it reads the session's open documents + their detection
    results on every :meth:`refresh`. The owner builds it once, shows it, and calls
    :meth:`refresh`; the window's own *Refresh* button does the same, so detecting
    another recording and clicking Refresh brings it into the plot.
    """

    #: Emitted only for a user's durable workspace edits. Calls to the public
    #: setters and :meth:`refresh` are deliberately silent so project restore and
    #: MainWindow's data refreshes do not immediately dirty the project again.
    workspaceChanged = Signal()

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None, *,
                 defer_initial_refresh: bool = False) -> None:
        """Build the window; defer gathering/fitting for project restore when requested."""
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("NanoScope — Voltage dependence")
        self.resize(880, 700)

        self._series: voltage_dep.VDepSeries = voltage_dep.vdep_comparability(
            (), metric="capture_rate_hz")
        self._points: list[voltage_dep.VDepPoint] = []
        #: recording -> why it is plotted but NOT in the fit. Written by
        #: :meth:`_redraw` from the fit's own report, read by :meth:`_fill_table`.
        self._not_in_fit: dict[str, str] = {}
        self._fit: dict[str, Any] | None = None
        #: THE APPLIED trend, as distinct from the SELECTED one. Owner note #12 -- "the
        #: app applies the fit as soon as the user selects it; better to have the user
        #: confirm" -- is about the Plots window's histogram, but this window offers the
        #: same gesture over the same kind of model, and a confirmation step that exists
        #: on one fit combo and not on the other is a rule the user has to remember per
        #: window. :meth:`_fit_model` reads THIS; the combo is only a proposal.
        self._applied_fit: str = _FIT_NONE
        self._scatter_item: pg.PlotDataItem | None = None
        self._fit_item: pg.PlotDataItem | None = None
        self._message_item: pg.TextItem | None = None
        self._appearance = PlotAppearance()

        self._build_ui()
        if not defer_initial_refresh:
            self.refresh()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Controls row: metric + fit + axis toggles. Items are added BEFORE the
        # change signals are connected so building the combos never triggers a redraw.
        self._analysis_section = QtWidgets.QWidget()
        controls = QtWidgets.QHBoxLayout(self._analysis_section)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QtWidgets.QLabel("Metric"))
        self._metric_combo = QtWidgets.QComboBox()
        self._metric_combo.addItems([lbl for lbl, _ in _METRIC_LABELS])
        self._metric_combo.setToolTip(
            "Which per-recording event summary to plot against voltage.")
        controls.addWidget(self._metric_combo)

        controls.addWidget(QtWidgets.QLabel("Fit"))
        self._fit_combo = QtWidgets.QComboBox()
        self._fit_combo.addItems([_FIT_NONE, _FIT_LINEAR, _FIT_EXP])
        self._fit_combo.setToolTip(
            "Overlay a trend: a straight line, or the A·exp(V/V0) capture-rate model. "
            "Press Apply fit to draw the one you have chosen.")
        controls.addWidget(self._fit_combo)

        # THE CONFIRMATION STEP (owner note #12), the same gate the Plots window's two
        # fit combos now carry. A trend line that appears the instant the dropdown is
        # opened is a model asserted over the user's points before they have said they
        # want it -- and on a four-recording voltage series, where every point is a
        # whole recording, an unasked-for exponential is exactly the kind of claim that
        # ends up screenshotted.
        self._fit_apply_btn = QtWidgets.QPushButton("Apply fit")
        self._fit_apply_btn.clicked.connect(self._apply_fit_from_user)
        controls.addWidget(self._fit_apply_btn)

        self._abs_v = QtWidgets.QCheckBox("|V|")
        self._abs_v.setToolTip("Plot against the voltage MAGNITUDE |V| instead of signed V.")
        controls.addWidget(self._abs_v)

        self._log_y = QtWidgets.QCheckBox("Log y")
        self._log_y.setToolTip(
            "Log-scale the metric axis — an exponential capture rate reads as a line.")
        controls.addWidget(self._log_y)

        self._accepted_only = QtWidgets.QCheckBox("Accepted only")
        self._accepted_only.setChecked(True)
        self._accepted_only.setToolTip("Summarise only the curated-in (accepted) events.")
        controls.addWidget(self._accepted_only)

        controls.addStretch(1)
        self._headline = QtWidgets.QLabel("")
        self._headline.setStyleSheet("font-weight:600; font-size:14px;")
        controls.addWidget(self._headline)
        root.addWidget(self._analysis_section)

        # Buttons row.
        bar = QtWidgets.QHBoxLayout()
        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip(
            "Re-gather (V, metric) from every open recording — pick up any detected since.")
        self._refresh_btn.clicked.connect(self._refresh_from_user)
        self._save_btn = QtWidgets.QPushButton("Export current figure…")
        self._save_btn.setToolTip("Export the voltage-dependence plot as PNG, SVG, or PDF.")
        self._save_btn.clicked.connect(lambda: self.save_image())
        self._save_data_btn = QtWidgets.QPushButton("Export data (CSV)…")
        self._save_data_btn.setToolTip(
            "Export exact plotted points and fitted trend with pooled provenance.")
        self._save_data_btn.clicked.connect(lambda: self.export_data())
        bar.addWidget(self._refresh_btn)
        bar.addWidget(self._save_btn)
        bar.addWidget(self._save_data_btn)
        bar.addStretch(1)
        root.addLayout(bar)

        self._appearance_widget = PlotAppearanceWidget(self._appearance)
        self._appearance_widget.changed.connect(self._appearance_from_user)
        self._appearance_section = bounded_controls(self._appearance_widget)
        root.addWidget(self._appearance_section)
        self._data_section = control_group(
            "Data", "Summarizes the detected events in each open recording against voltage. "
            "Analyze / refresh updates the points. Choose the metric and fit in Analysis.")
        self._data_section.layout().addWidget(self._accepted_only)
        self._export_section = QtWidgets.QWidget()
        export_layout = QtWidgets.QHBoxLayout(self._export_section)
        export_layout.addWidget(self._save_btn)
        export_layout.addWidget(self._save_data_btn)
        export_layout.addStretch(1)
        root.addWidget(self._data_section)
        root.addWidget(self._export_section)
        root.addWidget(self._headline)
        self._refresh_btn.hide()

        # The comparability banner. Above the plot, hidden when there is nothing to
        # say, so its presence alone is the signal.
        #
        # INSIDE A HEIGHT-CAPPED SCROLL AREA, because the banner's length is not
        # bounded by anything: each pooled recording adds a line to every group it
        # disagrees about, and a detection's own caveat is a PARAGRAPH. Measured on
        # the owner's three recordings as probed -- three baselines, an uncalibrated
        # gain and two detection caveats -- the free-height banner took 290 px of an
        # 880x700 window, squeezing the plot to 280 px and pushing the per-recording
        # table and the details pane off the bottom entirely. A warning that hides
        # the numbers it is warning about is its own defect.
        self._warning_label = QtWidgets.QLabel("")
        self._warning_label.setWordWrap(True)
        self._warning_label.setTextFormat(Qt.PlainText)
        self._warning_label.setStyleSheet(_WARN_CSS)
        self._warning_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._warning_label.setParent(self)
        self._warning_label.hide()  # retained verbatim for the existing read/export API
        self._warning_scroll = WarningSummary(title="Comparability warnings", maximum_height=_WARN_MAX_H)
        root.addWidget(self._warning_scroll)

        splitter = QtWidgets.QSplitter(Qt.Vertical)
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumSize(480, 320)
        self._plot: pg.PlotItem = self._plot_widget.getPlotItem()
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        splitter.addWidget(self._plot_widget)

        bottom = QtWidgets.QSplitter(Qt.Horizontal)
        details_box = QtWidgets.QGroupBox("Fit details")
        dlay = QtWidgets.QVBoxLayout(details_box)
        dlay.setContentsMargins(6, 6, 6, 6)
        self._details = QtWidgets.QPlainTextEdit()
        self._details.setReadOnly(True)
        self._details.setMaximumHeight(200)
        self._details.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        dlay.addWidget(self._details)
        bottom.addWidget(details_box)

        table_box = QtWidgets.QGroupBox("Per recording")
        tlay = QtWidgets.QVBoxLayout(table_box)
        tlay.setContentsMargins(6, 6, 6, 6)
        # EIGHT columns, not three: what was pooled travels with the value it
        # produced, in the same row, so a reader of the table cannot miss that two
        # rows were measured under different baselines -- or, since 2026-08-22,
        # that all three rows are raw uncalibrated pA. Two more since 2026-08-24:
        # the ANALYSED DURATION, because an "Event count" is a rate times that number
        # and the window used to show the count and never the duration; and where the
        # row's VOLTAGE came from, because a bias read off a filename is a much weaker
        # claim than one measured on the command channel and the x-axis of this fit
        # can be made of both. `V (mV)` stays column 1 and stays a bare number.
        self._table = QtWidgets.QTableWidget(0, len(_TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_TABLE_COLUMNS))
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        tlay.addWidget(self._table)
        bottom.addWidget(table_box)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self._workflow = install_plot_workflow(
            self, sections={"Data": (self._data_section,), "Analysis": (self._analysis_section,),
                            "Appearance": (self._appearance_section,), "Export": (self._export_section,)},
            actions=(("Analyze / refresh", self._refresh_from_user),
                     ("Apply fit", self._apply_fit_from_user)),
            exports=(("Export current figure…", lambda: self.save_image()),
                     ("Export data (CSV)…", lambda: self.export_data())))

        # Wire the change signals AFTER construction so populating them never redrew.
        self._metric_combo.currentTextChanged.connect(self.refresh)
        self._fit_combo.currentTextChanged.connect(self._on_fit_selection_changed)
        self._abs_v.toggled.connect(self._redraw)
        self._log_y.toggled.connect(self._redraw)
        self._accepted_only.toggled.connect(self.refresh)
        # ``activated`` and ``clicked`` are user-only signals; unlike the state
        # change signals above they do not fire while a saved project is restored
        # through the public setters.
        self._metric_combo.activated.connect(self._notify_workspace_changed)
        self._fit_combo.activated.connect(self._notify_workspace_changed)
        self._abs_v.clicked.connect(self._notify_workspace_changed)
        self._log_y.clicked.connect(self._notify_workspace_changed)
        self._accepted_only.clicked.connect(self._notify_workspace_changed)
        self._sync_fit_apply_btn()

    def _appearance_from_user(self, state: PlotAppearance) -> None:
        self._appearance = state
        self._apply_appearance()
        self.workspaceChanged.emit()

    def set_appearance_state(self, state) -> None:
        """Restore graphical overrides silently, without re-gathering or fitting."""
        self._appearance = (state if isinstance(state, PlotAppearance)
                            else PlotAppearance.from_mapping(state))
        self._appearance_widget.set_state(self._appearance)
        self._apply_appearance()

    def appearance_state(self) -> dict:
        return self._appearance.to_dict()

    def _apply_appearance(self) -> None:
        ensure_item_legend(self._plot, ((self._scatter_item, "Per recording"),
                                       (self._fit_item, "Applied trend")))
        self._appearance_widget.set_legend_available(self._scatter_item is not None)
        apply_plot_appearance(
            self._plot, self._appearance, plot_widget=self._plot_widget,
            data_items=(self._scatter_item,), fit_items=(self._fit_item,))

    def figure_export_target(self):
        return self._plot

    # -- current control state -------------------------------------------------------------------
    def _metric(self) -> str:
        return _METRIC_ID.get(self._metric_combo.currentText(), "capture_rate_hz")

    def _fit_model(self) -> str:
        """The trend that is DRAWN — the applied one, never the combo's selection."""
        return self._applied_fit

    # -- the fit APPLY gate (owner note #12) -----------------------------------------------------
    def _on_fit_selection_changed(self, _model: str) -> None:
        """A fit was chosen: arm Apply if a PERSON chose it, commit if code did.

        Qt's ``currentTextChanged`` fires the same way for a click and for
        ``setCurrentText``, and the two need opposite treatment: browsing the dropdown
        must not fit anything, while :meth:`set_fit` -- how the owner and the tests
        drive this window -- has already made the choice and must take effect at once.
        A user's selection arrives with the popup on screen or the combo holding focus;
        code touches neither. (Same rule, same wording, as
        ``plots_panel.PlotsWindow._from_the_user``.)
        """
        view = self._fit_combo.view()
        if (view is not None and view.isVisible()) or self._fit_combo.hasFocus():
            self._sync_fit_apply_btn()
        else:
            self._apply_fit()

    def _apply_fit(self) -> None:
        """Commit the selected trend and redraw with it."""
        self._applied_fit = self._fit_combo.currentText()
        self._sync_fit_apply_btn()
        self._redraw()

    def _apply_fit_from_user(self) -> None:
        """Commit an explicitly requested fit and mark the workspace changed."""
        self._apply_fit()
        self.workspaceChanged.emit()

    def _notify_workspace_changed(self, *_args: object) -> None:
        """Bridge Qt's user-only widget signals to the parameterless API signal."""
        self.workspaceChanged.emit()

    def _sync_fit_apply_btn(self) -> None:
        """Show on the button that the chosen trend is NOT the drawn one."""
        pending = self._fit_combo.currentText() != self._applied_fit
        self._fit_apply_btn.setEnabled(pending)
        self._fit_apply_btn.setText("Apply fit ●" if pending else "Apply fit")
        if hasattr(self, "_workflow"):
            button = self._workflow.primary_actions[1]
            button.setEnabled(pending)
            button.setText(self._fit_apply_btn.text())
        self._fit_apply_btn.setToolTip(
            f"Draw “{self._fit_combo.currentText()}” — the plot still shows "
            f"“{self._applied_fit}”." if pending else
            f"“{self._applied_fit}” is the trend on the plot.")

    # -- data ------------------------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-gather the points from the session (metric + curation) and redraw.

        Goes through :func:`~nanoscope.analysis.voltage_dep.vdep_series`, never
        ``vdep_points``: the verdict on pooling these recordings arrives with the
        points, so it cannot be forgotten by this method or by a future one.
        """
        self._series = voltage_dep.vdep_series(
            self._session, metric=self._metric(),
            accepted_only=self._accepted_only.isChecked())
        self._points = list(self._series.points)
        self._redraw()

    def _refresh_from_user(self) -> None:
        """Refresh from the button and record that explicit workspace action."""
        self.refresh()
        self.workspaceChanged.emit()

    def _render_workspace_snapshot(self) -> None:
        """Draw the installed project snapshot without gathering or refitting it.

        The workspace adapter restores the exact series, fit, drop annotations and
        explanatory text before invoking this seam.  The ordinary :meth:`_redraw`
        path intentionally computes a new trend; project restoration must instead
        render the frozen numerical result and wait for an explicit Refresh before
        touching the analysis kernels again.
        """
        saved_fit = self._fit
        saved_not_in_fit = dict(self._not_in_fit)
        saved_details = self._details.toPlainText()
        saved_headline = self._headline.text()
        saved_warning = self.warning_text()
        try:
            self._fill_table()
            if not self._points:
                self._message("No usable recordings.")
                return

            use_abs = self._abs_v.isChecked()
            log_y = self._log_y.isChecked()
            x_values = np.array(
                [abs(point.v_volts) if use_abs else point.v_volts
                 for point in self._points], dtype=np.float64)
            y_values = np.array([point.value for point in self._points], dtype=np.float64)
            finite = np.isfinite(x_values) & np.isfinite(y_values)
            if log_y:
                finite &= y_values > 0
            x_values, y_values = x_values[finite], y_values[finite]
            if x_values.size == 0:
                self._message("No finite points to plot (check the log-y toggle).")
                return

            self._clear()
            point_color = QtGui.QColor(_DATA_HEX)
            point_color.setAlpha(_POINT_ALPHA)
            self._scatter_item = pg.PlotDataItem(
                x=x_values, y=y_values, pen=None, symbol="o", symbolSize=11,
                symbolPen=pg.mkPen("#0d0f12"), symbolBrush=pg.mkBrush(point_color))
            self._plot.addItem(self._scatter_item)
            self._plot.setLabel(
                "bottom", "Applied |voltage| (V)" if use_abs else "Applied voltage (V)")
            self._plot.setLabel(
                "left", _Y_LABEL.get(self._metric(), "metric") + (" (log)" if log_y else ""))
            if saved_fit is not None:
                self._draw_fit_curve(x_values, saved_fit)
                self._apply_fit_summary(saved_fit, use_abs)
            else:
                self._set_title(self._provenance_segments(self._series), _MSG_HEX)
            self._plot.setLogMode(x=False, y=log_y)
            self._plot.autoRange()
        finally:
            # The rendering helpers normally derive these.  Here the values in the
            # project are authoritative and must remain byte-for-byte presentable.
            self._fit = saved_fit
            self._not_in_fit = saved_not_in_fit
            self._details.setPlainText(saved_details)
            self._headline.setText(saved_headline)
            self._set_warnings((saved_warning,) if saved_warning else ())
            self._apply_appearance()

    def points(self) -> list[voltage_dep.VDepPoint]:
        return list(self._points)

    def series(self) -> voltage_dep.VDepSeries:
        """The gathered points together with the comparability verdict on them."""
        return self._series

    def warnings(self) -> list[str]:
        """The comparability warnings currently on show (empty when comparable)."""
        return list(self._series.warnings)

    def warning_text(self) -> str:
        """The banner's text — '' exactly when the banner is hidden.

        Read the TEXT, not ``isVisible()``: a widget in a window that has not been
        shown yet reports invisible whatever its own flag says, and ``_set_warnings``
        is the single writer of both, always in step.
        """
        return self._warning_label.text()

    def fit(self) -> dict[str, Any] | None:
        return None if self._fit is None else dict(self._fit)

    def plot_item(self) -> pg.PlotItem:
        return self._plot

    def details_text(self) -> str:
        return self._details.toPlainText()

    # -- public setters (owner + tests) ----------------------------------------------------------
    def set_metric(self, metric: str) -> None:
        """Select the metric by its id (``"capture_rate_hz"`` ...) or its combo label.

        RAISES on a name this window does not offer. It used to fall through to
        ``QComboBox.setCurrentText``, which on a non-editable combo does nothing at all
        for unmatched text: ``set_metric("median depth (pA)")`` left the selection on
        "Capture rate (Hz)" and the window then reported capture-rate points, a
        capture-rate fit and a capture-rate details pane under the requested heading.
        That happened during the verification pass of this batch -- the same defect the
        report logged as Low-21 in a different window (see
        :func:`nanoscope.gui.combos.choose_combo_item`).
        """
        label = _METRIC_LABEL.get(metric, metric)
        choose_combo_item(self._metric_combo, label, "metric")

    def set_fit(self, model: str) -> None:
        """Select the fit overlay: ``"none"`` / ``"linear"`` / ``"exponential"``.

        RAISES on anything else -- see :meth:`set_metric`.

        APPLIED IMMEDIATELY: naming the model IS the confirmation the Apply button
        collects from a mouse (see :meth:`_on_fit_selection_changed`)."""
        choose_combo_item(self._fit_combo, model, "fit")
        self._apply_fit()

    def set_abs_voltage(self, on: bool) -> None:
        self._abs_v.setChecked(bool(on))

    def set_log_y(self, on: bool) -> None:
        self._log_y.setChecked(bool(on))

    def set_accepted_only(self, on: bool) -> None:
        self._accepted_only.setChecked(bool(on))

    # -- rendering -------------------------------------------------------------------------------
    def _clear(self) -> None:
        self._plot.clear()
        self._plot.setTitle(None)
        self._plot.titleLabel.setText("")
        self._scatter_item = self._fit_item = self._message_item = None

    def _set_warnings(self, warnings) -> None:
        """Show (or hide) the banner. The ONE writer of that label.

        Takes every block the banner carries -- the comparability warnings and the
        pooled detections' own caveats -- because a second writer is how one of them
        ends up overwriting the other. The label and its scroll area are shown and
        hidden together, so ``warning_text() == ""`` still means "no banner".
        """
        text = "\n\n".join(warnings)
        self._warning_label.setText(text)
        self._warning_scroll.set_warnings(tuple(warnings))

    def _message(self, text: str) -> None:
        self._clear()
        self._fit = None
        self._set_warnings(())      # nothing was pooled, so nothing to warn about
        self._headline.setText("")
        self._plot.setLogMode(x=False, y=False)
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=_MSG_HEX)
        self._plot.addItem(item)
        self._plot.setLabel("bottom", "")
        self._plot.setLabel("left", "")
        self._plot.setXRange(0, 1, padding=0)
        self._plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)
        self._message_item = item
        self._details.setPlainText(text)
        self._fill_table()

    def _redraw(self) -> None:
        try:
            self._redraw_analysis()
        finally:
            self._apply_appearance()

    def _redraw_analysis(self) -> None:
        pts = self._points
        self._not_in_fit = {}      # re-derived below; never carried over from a past fit
        self._fill_table()   # the table shows every gathered point, independent of the fit
        if not pts:
            # "No usable recordings" is a lie when there ARE recordings and this
            # window threw them all out: name them and their causes, the way the
            # sibling I-V window does when every point is refused.
            lines = ["No usable recordings.\nDetect events on recordings that have a "
                     "known voltage (batch-detect the whole series), then Refresh."]
            if self._series.excluded:
                lines = ["No recording could contribute a point to this series."]
                lines.extend(self._exclusion_lines(self._series))
            self._message("\n".join(lines))
            return

        metric = self._metric()
        use_abs = self._abs_v.isChecked()
        log_y = self._log_y.isChecked()

        x_all = np.array([abs(p.v_volts) if use_abs else p.v_volts for p in pts],
                         dtype=np.float64)
        y_all = np.array([p.value for p in pts], dtype=np.float64)
        finite = np.isfinite(x_all) & np.isfinite(y_all)
        notes: list[str] = []
        if log_y:
            pos = y_all > 0
            if int(np.sum(finite & ~pos)):
                notes.append(f"{int(np.sum(finite & ~pos))} non-positive point(s) dropped for log y.")
            finite &= pos
        x, y = x_all[finite], y_all[finite]
        # The points BEHIND x/y, in the same order, so a drop reported by index can be
        # turned back into the recording's name. Without this the fit can say "n = 2"
        # over three plotted points and nothing on screen can say which two.
        plotted = [p for p, keep in zip(pts, finite) if keep]
        if x.size == 0:
            self._message("No finite points to plot (check the log-y toggle).")
            return

        self._clear()
        point_color = QtGui.QColor(_DATA_HEX)
        point_color.setAlpha(_POINT_ALPHA)
        # A PlotDataItem (line disabled, symbol markers) -- NOT a bare ScatterPlotItem --
        # because only PlotDataItem implements setLogMode: with the log-y toggle on (the
        # capture-rate feature), a ScatterPlotItem's points keep their linear coordinates
        # while the axis switches to log10, so a rate of 24 is drawn at 10^24 and the plot
        # is unreadable. A PlotDataItem log-transforms its points with the axis.
        self._scatter_item = pg.PlotDataItem(
            x=x, y=y, pen=None, symbol="o", symbolSize=11,
            symbolPen=pg.mkPen("#0d0f12"), symbolBrush=pg.mkBrush(point_color))
        self._plot.addItem(self._scatter_item)

        xlabel = "Applied |voltage| (V)" if use_abs else "Applied voltage (V)"
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", _Y_LABEL.get(metric, "metric") + (" (log)" if log_y else ""))

        # The headline count of the details pane. Rewritten at the end of this method
        # once the fit is known: two counts on one screen must never disagree without
        # a sentence between them, and the fit's ``n`` is printed ten lines below this.
        header = (f"{_Y_LABEL.get(metric, metric)} vs {'|V|' if use_abs else 'V'} "
                  f"over {x.size} recording(s)")
        details: list[str] = [header]

        # The verdict FIRST, above the numbers it qualifies -- a reader who stops
        # after one line of the details pane still sees it.
        series = self._series
        caveats = self._caveat_lines(series)
        excluded = self._exclusion_block(series)
        # THE BANNER CARRIES WHAT IS MISSING TOO. A recording that never reached the
        # plot cannot be seen anywhere else on this window, which is exactly why its
        # absence was invisible; the comparability verdict and the per-recording
        # caveats can at least be checked against the rows they are about.
        self._set_warnings(tuple(series.warnings)
                           + (("\n".join(excluded),) if excluded else ())
                           + (("\n".join(caveats),) if caveats else ()))
        if series.warnings:
            details.append("")
            details.extend(series.warnings)
        if series.duration_warnings:
            # NOT part of the comparability banner above: "these recordings were not
            # measured the same way" and "this NUMBER is part physics and part file
            # length" are different claims (voltage_dep.duration_comparability).
            details.append("")
            details.extend(series.duration_warnings)
        if excluded:
            details.append("")
            details.extend(excluded)
        if caveats:
            details.append("")
            details.extend(caveats)
        details.append("")
        details.extend(self._provenance_lines(series))

        self._fit = None
        self._headline.setText("")
        model = self._fit_model()
        if model != _FIT_NONE and x.size >= 2:
            try:
                fit = voltage_dep.vdep_fit(x, y, model=model)
            except ValueError as exc:
                details.append(f"Fit unavailable: {exc}")
            else:
                self._fit = fit
                self._draw_fit_curve(x, fit)
                details.append("")
                details.extend(self._apply_fit_summary(fit, use_abs))
                # WHAT THE FIT LEFT OUT, wherever the fit is reported. vdep_fit hands
                # back the indices it could not use and why; this is where they become
                # recordings the reader can find in the table above.
                drop_lines = self._fit_drop_lines(fit, plotted)
                if drop_lines:
                    details.extend(drop_lines)
                    details[0] = (f"{header}   ·   {int(fit.get('n', 0))} of them in "
                                  f"the fit")
                    self._fill_table()      # re-mark the rows now that the fit is known
        elif model != _FIT_NONE:
            details.append("Need >= 2 recordings at different voltages to fit a trend.")
        if self._fit is None:
            self._set_title(self._figure_method_segments(series), _MSG_HEX)

        self._plot.setLogMode(x=False, y=log_y)
        self._plot.autoRange()
        if notes:
            details.append("")
            details.extend(notes)
        self._details.setPlainText("\n".join(details))

    def _draw_fit_curve(self, x: np.ndarray, fit: dict[str, Any]) -> None:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            hi = lo + 1.0
        grid = np.linspace(lo, hi, 200)
        try:
            curve = voltage_dep.vdep_predict(fit, grid)
        except (ValueError, ZeroDivisionError, KeyError):
            return
        self._fit_item = pg.PlotDataItem(x=grid, y=curve, pen=pg.mkPen(_FIT_HEX, width=2))
        self._plot.addItem(self._fit_item)

    # -- what was pooled -------------------------------------------------------------------------
    @staticmethod
    def _one_or_mixed(groups, mixed: str, prefix: str = "") -> str:
        """The single shared value of a provenance group, or ``MIXED ... (n)``."""
        if not groups:
            return "—"
        if len(groups) == 1:
            return prefix + groups[0][0]
        return f"{mixed} ({len(groups)})"

    def _set_title(self, segments, colour: str) -> None:
        """Write the plot title from its segments. The ONE writer of that label.

        Wraps (see :meth:`_wrap_title`) AND makes room for the wrap:
        ``PlotItem.setTitle`` hard-codes a 30 px title row and a 30 px cap on the
        label, so a two- or three-line title is drawn with its FIRST line clipped
        away behind the plot's top edge — trading the horizontal clipping the wrap
        exists to prevent for a vertical one. Measured on the owner's three
        recordings as probed: a 3-line title lost its whole first line, ``V₀`` and
        ``R²`` included.
        """
        text = self._wrap_title(segments)
        if not text:
            self._plot.setTitle(None)
            self._plot.titleLabel.setText("")
            return
        self._plot.setTitle(text, color=colour)
        if "<br>" not in text:
            return          # one line: leave pyqtgraph's own 30 px exactly as it was
        label = self._plot.titleLabel
        try:
            height = max(30.0, float(label.item.boundingRect().height()) + 6.0)
        except (AttributeError, TypeError):     # pragma: no cover - pyqtgraph internals
            height = 30.0 + 19.0 * text.count("<br>")
        label.setMaximumHeight(height)
        self._plot.layout.setRowFixedHeight(0, height)

    @staticmethod
    def _wrap_title(segments) -> str:
        """Join title segments with the separator, breaking a line before it clips.

        Returns pyqtgraph-ready markup: ``LabelItem`` renders the text as HTML, so
        ``<br>`` is a line break there. Breaks BETWEEN segments only -- a wrapped
        fingerprint or a wrapped "MIXED baselines (3)" would be worse than the
        clipping this exists to prevent.
        """
        lines: list[str] = []
        current = ""
        for seg in [s for s in segments if s]:
            candidate = seg if not current else f"{current}   ·   {seg}"
            if current and len(candidate) > _TITLE_WRAP:
                lines.append(current)
                current = seg
            else:
                current = candidate
        if current:
            lines.append(current)
        return "<br>".join(lines)

    def _provenance_segments(self, series: voltage_dep.VDepSeries) -> list[str]:
        """Legacy complete provenance summary, no longer used on the plot canvas.

        Kept for compatibility with callers of ``_provenance_title``. Live plots
        use ``_figure_method_segments``; Details uses ``_provenance_lines`` plus
        the full warning blocks, independently of whether a fit is displayed.
        """
        parts = [self._one_or_mixed(series.merge_groups, "MIXED merge states")]
        if series.depends_on_baseline:
            parts.append(self._one_or_mixed(
                series.baseline_groups, "MIXED baselines", prefix="baseline "))
        if series.depends_on_gain:
            parts.append(self._one_or_mixed(
                series.gain_groups, "MIXED gains", prefix="gain "))
        # THE DURATION, on the surface an exported PNG keeps. Load-bearing only for a
        # count (a count is a rate times a duration), so it is stated whenever this
        # metric reads it: the shared span when the recordings agree, and the range
        # with its ratio when they do not. A figure showing an "Event count" trend
        # now carries, on its face, the spans that trend was fitted over.
        if voltage_dep.depends_on_duration(series.metric):
            groups = series.duration_groups
            if series.duration_warnings:
                spans = sorted(p.analyzed_duration_s for p in series.points
                               if p.analyzed_duration_s is not None)
                span = (f" {spans[0]:.3g}–{spans[-1]:.3g} s, {spans[-1] / spans[0]:.2f}×"
                        if len(spans) >= 2 and spans[0] > 0 else "")
                parts.append(f"⚠ UNEQUAL analysed durations{span}")
            elif len(groups) == 1:
                parts.append(f"over {groups[0][0]} each")
        # Whether the x-axis is measured or read off a filename. Stated only when the
        # series MIXES them: one shared source is already in the provenance block, but
        # a law fitted through both kinds of x at once is a different claim.
        if len(series.voltage_source_groups) > 1:
            parts.append("MIXED voltage sources ("
                         + ", ".join(k for k, _ in series.voltage_source_groups) + ")")
        n_caveats = len(series.detection_warnings)
        if n_caveats:
            parts.append(f"⚠ {n_caveats} detection caveat(s)")
        # What is NOT in the pool, on the title too -- an exported PNG of a truncated
        # series must not look like an exported PNG of a whole one.
        if series.excluded:
            parts.append(f"⚠ {len(series.excluded)} recording(s) NOT in this series")
        if series.warnings:
            parts.insert(0, "⚠ NOT COMPARABLE")
        return parts

    @staticmethod
    def _figure_method_segments(series: voltage_dep.VDepSeries) -> list[str]:
        """Ordinary shared method labels, without warning prose on the canvas.

        Complete mixed/uncalibrated/excluded-source qualifications remain in the
        warning summary, Details and numerical export. Do not remove substrings
        from user titles or fit labels: those are deliberately authored content.
        """
        parts = []
        for groups, prefix, applicable in (
                (series.merge_groups, "", True),
                (series.baseline_groups, "baseline ", series.depends_on_baseline),
                (series.gain_groups, "gain ", series.depends_on_gain)):
            if applicable and len(groups) == 1:
                value = str(groups[0][0])
                if value and value.casefold() not in {"uncalibrated", "unknown", "—"}:
                    parts.append(prefix + value)
        if (voltage_dep.depends_on_duration(series.metric)
                and not series.duration_warnings and len(series.duration_groups) == 1):
            parts.append(f"over {series.duration_groups[0][0]} each")
        return parts

    def _provenance_title(self, series: voltage_dep.VDepSeries) -> str:
        """Legacy wrapped provenance summary; not assigned to live plot titles."""
        return self._wrap_title(self._provenance_segments(series))

    def _provenance_lines(self, series: voltage_dep.VDepSeries) -> list[str]:
        """The 'Pooled provenance' block: what measured every point on this plot."""
        lines = ["Pooled provenance"]

        def block(label: str, groups, note: str = "") -> None:
            if len(groups) == 1:
                key, recs = groups[0]
                lines.append(f"  {label:<10}{key}   ({len(recs)} recording(s)){note}")
                return
            lines.append(f"  {label:<10}{len(groups)} different{note}")
            for key, recs in groups:
                lines.append(f"    {key} — {', '.join(recs)}")

        unread = "   [not read by this metric]"
        block("Merge:", series.merge_groups)
        block("Baseline:", series.baseline_groups,
              "" if series.depends_on_baseline else unread)
        block("Gain:", series.gain_groups,
              "" if series.depends_on_gain else unread)
        # The two properties this module judges itself, rendered by the same block
        # writer so they read as part of one list rather than as an afterthought.
        # A count IS proportional to the duration; every other metric divided it out
        # or never touched it, and says so in the same words the other rows use.
        block("Duration:", series.duration_groups,
              "" if voltage_dep.depends_on_duration(series.metric) else unread)
        # No "[not read]" note on this one: EVERY metric here is plotted against the
        # x-axis these sources produced, so it is read by all of them.
        block("V source:", series.voltage_source_groups)
        return lines

    @staticmethod
    def _exclusion_lines(series: voltage_dep.VDepSeries) -> list[str]:
        """One line per open recording that is NOT a point here, value and reason."""
        out: list[str] = []
        for e in series.excluded:
            detail = f"{e.detail},  " if e.detail else ""
            out.append(f"  {e.recording}:  {detail}{e.reason}")
        return out

    @classmethod
    def _exclusion_block(cls, series: voltage_dep.VDepSeries) -> list[str]:
        """The 'NOT IN THIS SERIES' block, or ``[]`` when every recording placed.

        The half of the honesty that no other surface of this window can carry: the
        banner, the provenance block, the table and the "over N recording(s)" count
        all describe what IS pooled, so a recording that never reached the plot was
        invisible on every one of them. The sibling I-V window has listed its own
        refusals since CF-5, in these same words, for the same recordings.
        """
        if not series.excluded:
            return []
        n = len(series.excluded)
        # "...covers N of the M open", never "...is fitted over N": the FIT may drop a
        # plotted point of its own (a non-positive value has no logarithm), and that
        # count is reported by _fit_drop_lines. Two counts on one screen must not
        # disagree, so each one states only what it actually counted.
        lines = [f"NOT IN THIS SERIES — {n} open recording(s) could not contribute a "
                 f"point, so this series covers {len(series.points)} of the "
                 f"{len(series.points) + n} open recording(s):"]
        lines.extend(cls._exclusion_lines(series))
        return lines

    def _fit_drop_lines(self, fit: dict[str, Any], plotted: list) -> list[str]:
        """Name the recordings the FIT discarded, and remember them for the table.

        A point can be on the plot, in the table and in the "over N recording(s)"
        count and still not be in the fit -- the exponential capture-rate model is a
        line through ``(V, ln value)``, so a 0 Hz recording (a pore that captured
        nothing at low bias, or one whose events the user rejected) has no logarithm
        and leaves. That drop used to be visible only as two counts ten lines apart
        disagreeing, with no sentence between them, even though this same window
        already announces the byte-identical drop when the log-y toggle causes it.
        """
        dropped = [int(i) for i in fit.get("dropped", ()) if 0 <= int(i) < len(plotted)]
        if not dropped:
            return []
        reason = fit.get("drop_reason", "") or "not usable by this fit"
        names = [plotted[i].recording for i in dropped]
        self._not_in_fit = {name: reason for name in names}
        return [
            f"DROPPED FROM THE FIT — {len(names)} of the {int(fit.get('n_input', 0))} "
            f"plotted recording(s) are NOT in the trend above ({reason}): "
            f"{', '.join(names)}. They are still on the plot and in the table, marked "
            f"there; the R² and n below describe the {int(fit.get('n', 0))} that were "
            f"fitted."
        ]

    @staticmethod
    def _caveat_lines(series: voltage_dep.VDepSeries) -> list[str]:
        """The pooled detections' OWN warnings, as ``recording: warning`` lines.

        Empty when there are none, so the block's presence is the signal. Rendered
        the way ``Analysis ▸ Plots`` renders the same tuples for the same
        recordings — these caveats reached the Summary panel and the Plots details
        pane and stopped there, never reaching the one window that turns those
        recordings into a fitted law.
        """
        caveats = series.detection_warnings
        if not caveats:
            return []
        recs = len({rec for rec, _ in caveats})
        lines = [f"DETECTION CAVEATS — {recs} of the {len(series.points)} pooled "
                 f"recording(s) carry a warning from the detection that produced "
                 f"their events. These do not make the pool incomparable; they make "
                 f"those points wrong:"]
        lines.extend(f"  {rec}: {w}" for rec, w in caveats)
        return lines

    def _apply_fit_summary(self, fit: dict[str, Any], use_abs: bool) -> list[str]:
        """Set the headline from the fit and return its detail lines + R²/n."""
        p = fit.get("params", {})
        r2, n = fit.get("r2", float("nan")), fit.get("n", 0)
        # NEVER let a non-finite R² reach a bare format spec (the PSD panel's _fmt_q
        # rule). `populations.linear_fit` returns r2 = nan for exactly one cause: the
        # response has no variance across the pooled recordings (SStot == 0), e.g.
        # identical capture rates at +600 and -600 mV. "R² = nan" in the title and
        # the Goodness line read as a computation failure -- say the cause instead.
        # The title takes the short form (its 72-char wrap budget is real: a clean
        # single-provenance title must stay on one line); the Goodness line below
        # carries the full cause, and the two surfaces agree.
        # ...AND THE MIRROR CASE, through the SAME writer the Plots scatter uses
        # (:func:`nanoscope.analysis.stats.r2_dof_tag`). Both trends here have TWO free
        # parameters -- (slope, intercept) and (A, V0) -- so over two recordings the
        # curve passes through both points whatever they are and R² is 1 by
        # construction. "Goodness of fit: R² = 1.00000 · n = 2 recording(s)" was the
        # strongest-looking line the panel can print, on the weakest evidence it can
        # have. The short form rides with the number in the title (its 72-char wrap
        # budget is real); the sentence goes in the Goodness line below.
        r2_finite = np.isfinite(r2)
        dof_tag = stats.r2_dof_tag(n, _VDEP_FIT_PARAMS)
        r2_short = (f"R² = {r2:.4f}" + (f" ({dof_tag})" if dof_tag else "")
                    if r2_finite else "R² undefined")
        vsym = "|V|" if use_abs else "V"
        # The headline number and the title both carry the pooled provenance: the
        # fit is a statement ABOUT those event sets, and it is wrong to read it
        # without knowing which ones.
        prov = self._figure_method_segments(self._series)
        warn = "⚠ " if self._series.warnings else ""
        colour = _WARN_HEX if self._series.warnings else _FIT_HEX
        if fit.get("model") == "exponential":
            self._headline.setText(f"{warn}V₀ = {p['V0']:.4g} V")
            lines = [
                f"Exponential:  value = {p['A']:.4g} · exp({vsym} / {p['V0']:.4g})",
                f"              A = {p['A']:.5g}   V₀ = {p['V0']:.5g} V",
            ]
            lead = [f"V₀ = {p['V0']:.4g} V", r2_short]
        else:
            self._headline.setText(f"{warn}slope = {p['slope']:.4g}")
            lines = [f"Linear:  value = {p['slope']:.5g} · {vsym} + {p['intercept']:.5g}"]
            lead = [f"slope = {p['slope']:.4g}", r2_short]
        self._set_title(lead + prov, colour)
        if r2_finite:
            dof_note = stats.r2_dof_caveat(n, _VDEP_FIT_PARAMS, model="trend",
                                           noun="recording")
            lines.append(f"Goodness of fit:  R² = {r2:.5f}   ·   n = {n} recording(s)"
                         + (f"\n                  {dof_note}" if dof_note else ""))
        else:
            lines.append(
                f"Goodness of fit:  R² undefined — the metric is identical across the "
                f"pooled recording(s), so a trend explains nothing and no "
                f"goodness-of-fit exists to report   ·   n = {n} recording(s)")
        return lines

    def _fill_table(self) -> None:
        """Fill the per-recording table from every gathered point (sorted by voltage).

        Also marks the rows that are on the plot but NOT in the fit, from
        ``self._not_in_fit`` (see :meth:`_redraw`) -- so the table and the fit's ``n``
        cannot describe two different sets of recordings without saying so.
        """
        pts = sorted(self._points, key=lambda p: p.v_volts)
        metric = self._metric()
        headers = list(_TABLE_COLUMNS)
        headers[_COL_METRIC] = _Y_LABEL.get(metric, "metric")
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(pts))
        for r, p in enumerate(pts):
            why = self._not_in_fit.get(p.recording, "")
            name = QtWidgets.QTableWidgetItem(
                p.recording + (_NOT_IN_FIT_MARK if why else ""))
            if why:
                name.setToolTip(why)
            self._table.setItem(r, _COL_RECORDING, name)
            self._table.setItem(r, _COL_VOLTAGE,
                                QtWidgets.QTableWidgetItem(f"{p.v_volts * 1000.0:.4g}"))
            self._table.setItem(r, _COL_METRIC, QtWidgets.QTableWidgetItem(f"{p.value:.6g}"))
            # Where this row's x-coordinate came from: a command-channel median, or a
            # number parsed out of the file name. The I-V window prints the same tag
            # against the same recording, from the same resolver.
            self._table.setItem(r, _COL_V_SOURCE,
                                QtWidgets.QTableWidgetItem(p.voltage_source or "unstated"))
            # How much signal produced this row's number. Load-bearing for a COUNT
            # (a count is a rate times this) and informative for everything else.
            self._table.setItem(r, _COL_DURATION, QtWidgets.QTableWidgetItem(
                "unknown" if p.analyzed_duration_s is None
                else f"{p.analyzed_duration_s:.3f}"))
            self._table.setItem(r, _COL_BASELINE,
                                QtWidgets.QTableWidgetItem(p.baseline_fingerprint))
            self._table.setItem(r, _COL_MERGE, QtWidgets.QTableWidgetItem(p.merge_state))
            # The gain the row's number is IN, beside the number. Read through the
            # same labeller the pooled Gain groups use, so the table cell and the
            # title cannot describe one recording two ways.
            self._table.setItem(r, _COL_GAIN, QtWidgets.QTableWidgetItem(
                gain_label(p.calibration_fingerprint)))

    # -- export ----------------------------------------------------------------------------------
    def save_image(self, path: str | None = None) -> Path | None:
        """Export the current voltage-dependence plot as PNG, SVG or PDF by extension.

        Returns the written path, or None when the user cancelled OR the write
        failed (a failed write is reported in a dialog, never returned as a path).
        """
        export_kwargs = {}
        if path is None:
            chosen = choose_live_plot_export(
                self, self._plot, suggestion="voltage_dependence.png")
            if chosen is None:
                return None
            selected_path, options = chosen
            path = str(selected_path)
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
            notes = write_live_plot_image(out, self._plot, **export_kwargs)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Save voltage-dependence image",
                image_write_failure_text(out, exc, formats=IMAGE_FORMATS))
            return None
        # What the replace changed about the FILE (symlink followed, hard-linked names
        # left holding the previous contents) — unrecoverable after the fact, so it is
        # disclosed here the way File ▸ Save Project discloses it (EXP3-02).
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Save voltage-dependence image", "\n\n".join(notes))
        self._details.setPlainText(self.details_text() + f"\n\nSaved image → {out.name}")
        return out

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return str(value)

    def _plot_data(self) -> pd.DataFrame:
        """Exact visible voltage-series points and trend with pooled provenance."""
        if self._scatter_item is None:
            return pd.DataFrame()

        metric = self._metric()
        use_abs = self._abs_v.isChecked()
        log_y = self._log_y.isChecked()
        fit = self._fit or {}
        fit_fields = {
            "fit_model": fit.get("model", "none"),
            "fit_params_json": json.dumps(
                fit.get("params", {}), ensure_ascii=False, default=self._json_default),
            "fit_r2": fit.get("r2", np.nan),
            "fit_n": fit.get("n", np.nan),
            "fit_drop_reason": fit.get("drop_reason", ""),
        }
        warnings = (list(self._series.warnings) + list(self._series.duration_warnings)
                    + [f"{recording}: {warning}"
                       for recording, warning in self._series.detection_warnings]
                    + [f"{item.recording}: {item.reason}"
                       for item in self._series.excluded])
        common = {
            "view": "voltage_dependence", "metric": metric,
            "metric_label": _Y_LABEL.get(metric, metric),
            "absolute_voltage": use_abs, "log_y": log_y,
            "accepted_only": self._accepted_only.isChecked(),
            "applied_fit": self._fit_model(),
            "warnings_json": json.dumps(warnings, ensure_ascii=False),
            "details": self.details_text(), **fit_fields,
            "series_source_doc_ids_json": json.dumps(
                [point.doc_id for point in self._points if point.doc_id],
                ensure_ascii=False),
        }
        rows: list[dict] = []
        for point in self._points:
            x = abs(point.v_volts) if use_abs else point.v_volts
            finite = np.isfinite(x) and np.isfinite(point.value)
            positive = point.value > 0 or not log_y
            if finite and positive:
                part = "point"
                warning = self._not_in_fit.get(point.recording, "")
                xv, yv = float(x), float(point.value)
            else:
                part = "excluded_from_view"
                warning = ("non-positive for log y" if finite and not positive
                           else "non-finite voltage or metric")
                xv = float(x) if np.isfinite(x) else np.nan
                yv = float(point.value) if np.isfinite(point.value) else np.nan
            rows.append({
                "part": part,
                "source_identity_kind": ("document_id" if point.doc_id
                                         else "recording_label"),
                "source_identity_available": bool(point.doc_id),
                "source_doc_id": point.doc_id or None,
                "source_recording": point.recording, "voltage_V": xv,
                "metric_value": yv, "voltage_source": point.voltage_source,
                "analyzed_duration_s": point.analyzed_duration_s,
                "baseline_fingerprint": point.baseline_fingerprint,
                "merge_state": point.merge_state,
                "calibration_fingerprint_json": json.dumps(
                    point.calibration_fingerprint, ensure_ascii=False,
                    default=self._json_default),
                "signal_domain": point.signal_domain,
                "source_warnings_json": json.dumps(
                    list(point.detection_warnings), ensure_ascii=False),
                "warning": warning, "warning_detail": "", **common,
            })

        for exclusion in self._series.excluded:
            rows.append({
                "part": "excluded_from_series",
                "source_identity_kind": "document_id",
                "source_identity_available": bool(exclusion.doc_id),
                "source_doc_id": exclusion.doc_id,
                "source_recording": exclusion.recording, "voltage_V": np.nan,
                "metric_value": np.nan, "voltage_source": "",
                "analyzed_duration_s": np.nan, "baseline_fingerprint": "",
                "merge_state": "", "calibration_fingerprint_json": "",
                "signal_domain": "", "source_warnings_json": "[]",
                "warning": exclusion.reason, "warning_detail": exclusion.detail,
                **common,
            })

        if self._fit_item is not None:
            x, y = self._fit_item.getData()
            for xv, yv in zip(np.asarray(x), np.asarray(y)):
                rows.append({
                    "part": "fit", "source_identity_kind": "",
                    "source_identity_available": False,
                    "source_doc_id": None,
                    "source_recording": "", "voltage_V": float(xv),
                    "metric_value": float(yv), "voltage_source": "",
                    "analyzed_duration_s": np.nan, "baseline_fingerprint": "",
                    "merge_state": "", "calibration_fingerprint_json": "",
                    "signal_domain": "", "source_warnings_json": "[]",
                    "warning": "", "warning_detail": "", **common,
                })
        return pd.DataFrame(rows)

    def export_data(self, path: str | None = None) -> Path | None:
        """Export the exact visible voltage-series view and provenance as staged CSV."""
        if path is None:
            chosen = ask_export_path(
                self, title="Export voltage-dependence data",
                suggestion="voltage_dependence.csv", filters=DATA_FILTERS)
            if chosen is None:
                return None
            path = str(chosen)
        out = Path(path)
        try:
            notes = write_plot_csv(out, self._plot_data())
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(
                self, "Export voltage-dependence data",
                data_write_failure_text(out, exc))
            return None
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Export voltage-dependence data", "\n\n".join(notes))
        self._details.setPlainText(self.details_text() + f"\n\nExported data → {out.name}")
        return out
