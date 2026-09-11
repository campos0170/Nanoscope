"""Explicit, snapshot-backed distribution summaries of existing event values.

This optional page never detects, fits, recalibrates, or changes event rows. Draft
controls and live sources are separate from the committed figure: only Generate
computes ECDF/quartiles/KDE. Restoring a project paints stored draw coordinates.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..analysis.distribution_summary import summarize_distribution
from ..core.analysis_workspace import _plot_source_recipe, fingerprint_frame
from ..core.sources import amplitude_units_label, signal_domain_of, source_comparability
from ..core.source_style import compact_source_labels, validate_source_appearance
from .analysis_catalog import ANALYSIS_LABELS
from .axes import ReadableLinearAxis
from .export_paths import ask_export_path
from .plot_appearance import PlotAppearance, PlotAppearanceWidget, apply_plot_appearance
from .plot_export import (DATA_FILTERS, IMAGE_FORMATS, choose_live_plot_export,
                          write_live_plot_image, write_plot_csv)
from .plot_workflow import (WarningSummary, bounded_controls, control_group,
                            ensure_item_legend, install_plot_workflow)
from .source_appearance import SourceAppearanceEditor


VIEWS = (("ecdf", "ECDF"), ("survival", "Empirical survival (uncensored)"),
         ("box", "Box (linear quartiles)"), ("violin", "Violin (Scott KDE)"))
METRICS = (("dwell_s", "Dwell (s)"), ("depth_pA", "Median depth (source units)"),
           ("frac_depth", "Fractional depth ΔI/I₀"),
           ("amplitude_pA", "Selected amplitude (recorded units)"))
METHODS = {
    "ecdf": "Unweighted ECDF F(x) = count(X ≤ x) / n. Ties share a right-continuous jump.",
    "survival": "Empirical P(X > x) = count(X > x) / n for uncensored observations. "
                "This is not Kaplan–Meier and does not correct missed or censored events.",
    "box": "Q1, median and Q3 use NumPy linear-interpolated quantiles. Whiskers are "
           "the most extreme observed values within 1.5 IQR; circles show all outliers.",
    "violin": "Gaussian KDE with Scott bandwidth (n⁻¹/⁵ × sample standard deviation). "
              "Only the observed range is drawn; tails are not renormalized. Each "
              "source has equal peak width, independent of its event count. No KDE "
              "is inferred for fewer than 3 observations or constant data.",
}
_PLOT_COLUMNS = ["source_key", "source_label", "primitive", "role", "point", "x", "y"]


def _display_source_label(label, number):
    """Keep plot labels bounded; full names stay in tables, provenance and notes."""
    return label if len(label) <= 48 else f"{number + 1} · {label[:45]}…"


class DistributionSummaryWindow(QtWidgets.QMainWindow):
    workspaceChanged = QtCore.Signal()
    # The host may synchronously update sources before the explicit Generate action.
    # Its handler must not call analyze again.
    refreshRequested = QtCore.Signal()

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self._session = session
        self.setWindowTitle(f"NanoScope — {ANALYSIS_LABELS['distribution_summaries']}")
        self.resize(1000, 740)
        self.setMinimumSize(650, 480)
        self._sources = {}
        self._source_stamps = {}
        self._selected_keys = []
        self._loading = False
        self._stale = False
        self._applied_config = {}
        self._source_recipes = []
        self._snapshot = {}
        self._tables = {}
        self._data_items = []
        self._appearance = PlotAppearance()
        self._source_appearance = {}

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)
        self._status = QtWidgets.QLabel("Select event sources, then Generate. No analysis runs on opening this page.")
        self._status.setTextFormat(QtCore.Qt.PlainText)
        self._status.setWordWrap(True)
        self._status.setMaximumHeight(65)
        layout.addWidget(self._status)
        self._warnings = WarningSummary(title="Distribution notes", maximum_height=100)
        layout.addWidget(self._warnings)

        data = QtWidgets.QGroupBox("Data · each source remains a separate distribution")
        data_layout = QtWidgets.QGridLayout(data)
        self._source_list = QtWidgets.QListWidget()
        self._source_list.setAccessibleName("Distribution event sources")
        self._source_list.setMaximumHeight(115)
        self._accepted_only = QtWidgets.QCheckBox("Accepted events only")
        self._accepted_only.setChecked(True)
        self._metric_combo = QtWidgets.QComboBox()
        for key, label in METRICS:
            self._metric_combo.addItem(label, key)
        self._metric_combo.setAccessibleName("Distribution measurement")
        data_layout.addWidget(self._source_list, 0, 0, 3, 1)
        data_layout.addWidget(QtWidgets.QLabel("Measurement"), 0, 1)
        data_layout.addWidget(self._metric_combo, 0, 2)
        data_layout.addWidget(self._accepted_only, 1, 1, 1, 2)
        note = QtWidgets.QLabel("Stored values are used as recorded. Missing/non-finite values are disclosed; no imputation or pooling.")
        note.setWordWrap(True)
        data_layout.addWidget(note, 2, 1, 1, 2)

        analysis = QtWidgets.QGroupBox("Analysis · opt-in descriptive summaries")
        analysis_layout = QtWidgets.QGridLayout(analysis)
        self._view_combo = QtWidgets.QComboBox()
        for key, label in VIEWS:
            self._view_combo.addItem(label, key)
        self._view_combo.setAccessibleName("Distribution summary type")
        self._violin_points = QtWidgets.QSpinBox()
        self._violin_points.setRange(32, 2048)
        self._violin_points.setValue(256)
        self._violin_points.setAccessibleName("Violin density grid points")
        self._violin_points.setToolTip("Evaluation grid only. Scott bandwidth is unchanged. No source subsampling.")
        self._method_note = QtWidgets.QLabel()
        self._method_note.setWordWrap(True)
        self._method_note.setTextFormat(QtCore.Qt.PlainText)
        analysis_layout.addWidget(QtWidgets.QLabel("Summary"), 0, 0)
        analysis_layout.addWidget(self._view_combo, 0, 1)
        analysis_layout.addWidget(QtWidgets.QLabel("Violin grid points"), 0, 2)
        analysis_layout.addWidget(self._violin_points, 0, 3)
        analysis_layout.addWidget(self._method_note, 1, 0, 1, 4)

        self._appearance_widget = PlotAppearanceWidget(self._appearance)
        # There is no fitted curve on this page. Keep the common preset vocabulary,
        # but explicitly disable its inapplicable fit palette control.
        if hasattr(self._appearance_widget, "fit_color"):
            self._appearance_widget.fit_color.setEnabled(False)
            self._appearance_widget.fit_color.setToolTip("Distribution summaries have no fitted curve.")
        appearance_host = QtWidgets.QWidget()
        appearance_layout = QtWidgets.QVBoxLayout(appearance_host)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        appearance_layout.addWidget(self._appearance_widget)
        self._source_appearance_editor = SourceAppearanceEditor(session=session)
        self._source_appearance_editor.changed.connect(self._source_appearance_from_user)
        appearance_layout.addWidget(self._source_appearance_editor)
        appearance = bounded_controls(appearance_host, maximum_height=235)
        export = control_group("Export the committed figure", "Export preserves the last generated figure, "
                               "including its notes and exact visible coordinates; unapplied controls do not change it. "
                               "Use Figure Library to save its numerical snapshot, source bindings and appearance together.")
        for widget in (data, analysis, appearance, export):
            layout.addWidget(widget)
        self._plot_widget = pg.PlotWidget(background="#111316", axisItems={
            side: ReadableLinearAxis(orientation=side) for side in ("bottom", "left")})
        self._plot_widget.setMinimumHeight(150)
        self._plot = self._plot_widget.getPlotItem()
        self._plot.setTitle("Distribution summaries · Generate to begin")
        layout.addWidget(self._plot_widget, 1)
        self._workflow = install_plot_workflow(
            self, sections={"Data": (data,), "Analysis": (analysis,),
                            "Appearance": (appearance,), "Export": (export,)},
            actions=(("Generate", self._generate_from_user),),
            exports=(("Export current figure…", self.save_image),
                     ("Export plotted coordinates CSV…", self.save_csv)))
        self._workflow.export_button.setEnabled(False)
        self._source_list.itemChanged.connect(self._source_selection_changed)
        self._metric_combo.currentIndexChanged.connect(self._draft_changed)
        self._accepted_only.toggled.connect(self._draft_changed)
        self._view_combo.currentIndexChanged.connect(self._draft_changed)
        self._violin_points.valueChanged.connect(self._draft_changed)
        self._appearance_widget.changed.connect(self._appearance_from_user)
        self._update_method_note()

    def config(self):
        """Current draft controls, not the recipe of an already generated figure."""
        return {"source_keys": list(self._selected_keys),
                "accepted_only": self._accepted_only.isChecked(),
                "metric": self._metric_combo.currentData(),
                "kind": self._view_combo.currentData(),
                "violin_points": self._violin_points.value()}

    def set_config(self, config):
        """Silently restore draft controls. This never generates a distribution."""
        self._loading = True
        try:
            self._selected_keys = [str(k) for k in config.get("source_keys", self._selected_keys)]
            for combo, name in ((self._metric_combo, "metric"), (self._view_combo, "kind")):
                index = combo.findData(config.get(name))
                if index >= 0:
                    combo.setCurrentIndex(index)
            self._accepted_only.setChecked(bool(config.get("accepted_only", True)))
            self._violin_points.setValue(int(config.get("violin_points", 256)))
            self._populate_sources()
            self._update_method_note()
        finally:
            self._loading = False

    def selected_sources(self):
        return [self._sources[key] for key in self._selected_keys if key in self._sources]

    def _populate_sources(self):
        blocked = QtCore.QSignalBlocker(self._source_list)
        self._source_list.clear()
        for key, source in self._sources.items():
            item = QtWidgets.QListWidgetItem(str(source.label))
            item.setToolTip(str(source.label))
            item.setData(QtCore.Qt.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if key in self._selected_keys else QtCore.Qt.Unchecked)
            self._source_list.addItem(item)
        # Keep missing draft bindings explicit rather than silently selecting others.
        for key in self._selected_keys:
            if key not in self._sources:
                item = QtWidgets.QListWidgetItem(f"Unavailable source: {key}")
                item.setToolTip(f"Unavailable source: {key}")
                item.setData(QtCore.Qt.UserRole, key)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                self._source_list.addItem(item)
        del blocked

    def set_sources(self, sources, selected_keys=None, analyze=False):
        """Offer live data without replacing the committed numerical snapshot.

        Content stamps are retained separately from live frames so an in-place
        curation mutation is detected too. Reopening an unchanged page stays current.
        """
        offered = {str(source.key): source for source in sources}
        if len(offered) != len(sources):
            raise ValueError("Distribution sources must have unique keys.")
        stamps = {key: (fingerprint_frame(source.df),
                        repr(dict(source.provenance)), repr(source.calibration_fingerprint),
                        tuple(source.warnings)) for key, source in offered.items()}
        before = tuple((key, self._source_stamps.get(key)) for key in self._selected_keys)
        self._sources = offered
        if selected_keys is not None:
            self._selected_keys = [str(key) for key in selected_keys]
        elif not self._selected_keys and not self._snapshot:
            current = [key for key, source in offered.items() if source.current]
            self._selected_keys = current[:1] or list(offered)[:1]
        self._source_stamps = stamps
        self._populate_sources()
        self._sync_source_appearance_editor()
        after = tuple((key, stamps.get(key)) for key in self._selected_keys)
        if before != after:
            self._stale = bool(self._snapshot)
            self._update_status()
            self.workspaceChanged.emit()
        if analyze:
            self.analyze()

    def _source_selection_changed(self, *_args):
        self._selected_keys = [str(self._source_list.item(i).data(QtCore.Qt.UserRole))
                               for i in range(self._source_list.count())
                               if self._source_list.item(i).checkState() == QtCore.Qt.Checked]
        self._draft_changed()

    def _update_method_note(self):
        kind = self._view_combo.currentData()
        self._method_note.setText(METHODS[kind])
        self._violin_points.setEnabled(kind == "violin")

    def _draft_changed(self, *_args):
        if self._loading:
            return
        self._update_method_note()
        self._stale = bool(self._snapshot)
        self._update_status()
        self.workspaceChanged.emit()

    def _update_status(self):
        if not self._snapshot:
            text = "Select event sources, then Generate. No analysis runs on opening this page."
        else:
            counts = ", ".join(f"{series['label']}: n={series['n']}"
                               for series in self._snapshot["series"])
            text = ("Unapplied data/analysis changes · showing the last generated snapshot. "
                    if self._stale else "Generated snapshot · ") + counts
        self._status.setToolTip(text)
        self._status.setAccessibleDescription(text)
        self._status.setText(text if len(text) <= 220 else text[:217] + "…")
        self._workflow.primary_actions[0].setText("Generate ●" if self._stale else "Generate")
        self._workflow.export_button.setEnabled(self.has_data())

    def _generate_from_user(self):
        self.refreshRequested.emit()
        try:
            self.analyze()
        except (TypeError, ValueError, OverflowError) as exc:
            QtWidgets.QMessageBox.warning(self, "Generate distribution summary", str(exc))

    def refresh(self):
        """Explicit programmatic generation; the host should offer current sources first."""
        return self.analyze()

    def analyze(self):
        """Compute a new figure transactionally; on failure retain the old snapshot."""
        config = self.config()
        if not config["source_keys"]:
            raise ValueError("Select at least one event source.")
        if any(key not in self._sources for key in config["source_keys"]):
            raise ValueError("A selected source is unavailable. Rebind it or uncheck it before Generate.")
        sources = self.selected_sources()
        metric, kind = config["metric"], config["kind"]
        tables, series, primitives, warnings, frames, recipes = {}, [], [], [], [], []
        for number, source in enumerate(sources):
            frame = source.df.copy(deep=True)
            if metric not in frame:
                raise ValueError(f"{source.label}: the {metric!r} measurement is unavailable.")
            if config["accepted_only"]:
                if "accepted" not in frame:
                    warnings.append(f"{source.label}: no accepted flag is stored; all source rows are included.")
                else:
                    frame = frame.loc[frame["accepted"].fillna(False).astype(bool)].copy(deep=True)
            frames.append(frame)
            # Nullable/object integer columns must be checked before pandas can
            # round them while coercing missing or nonnumeric cells to float64.
            for value in frame[metric]:
                if isinstance(value, (bool, np.bool_, complex, np.complexfloating,
                                      datetime.date, datetime.timedelta, np.datetime64, np.timedelta64)):
                    raise ValueError(f"{source.label}: measurements must be real numbers in the recorded units, not boolean, complex or time-typed values.")
                if (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                        and abs(int(value)) > 2 ** 53):
                    raise ValueError(f"{source.label}: integer measurements above 2**53 cannot be plotted exactly.")
                if isinstance(value, (str, Decimal)):
                    try:
                        exact = Decimal(value)
                    except InvalidOperation:
                        continue  # Nonnumeric text is disclosed as an exclusion.
                    if exact.is_finite() and exact == exact.to_integral_value() and abs(exact) > 2 ** 53:
                        raise ValueError(f"{source.label}: integer measurements above 2**53 cannot be plotted exactly.")
            numeric = pd.to_numeric(frame[metric], errors="coerce")
            values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
            summary = summarize_distribution(values, kind=kind, violin_points=config["violin_points"])
            table_key = f"source_{number}"
            tables[table_key] = frame
            tables[f"samples_{number}"] = pd.DataFrame({"value": summary.samples})
            tables[f"summary_{number}"] = pd.DataFrame({
                "x": summary.x, "y": summary.y if len(summary.y) else np.full(len(summary.x), np.nan)})
            info = {"key": str(source.key), "label": str(source.label), "n": summary.n,
                    "n_total": summary.n_total, "n_excluded": summary.n_excluded,
                    "stats": dict(summary.stats), "source_table": table_key}
            series.append(info)
            recipes.append(_plot_source_recipe(self, source))
            warnings.extend(f"{source.label}: {note}" for note in (*source.warnings, *summary.warnings))
            color = str(source.colour)
            if not QtGui.QColor(color).isValid():
                color = "#56b4e9"
            self._draw_commands(summary, number, info, color, tables, primitives)
        verdict = source_comparability(sources, (metric,), frames=frames)
        warnings.extend(verdict.warnings)
        warnings = list(dict.fromkeys(warnings))
        label = self._metric_label(metric, sources, frames)
        title = dict(VIEWS)[kind]
        # Method-defining language belongs to the exported figure, not just its UI.
        if kind == "survival":
            title = "Empirical survival · uncensored, not Kaplan–Meier"
        elif kind == "violin":
            title = "Violin · Scott KDE, observed range, equal peak widths"
        elif kind == "box":
            title = "Box · linear quartiles, 1.5-IQR sample whiskers"
        snapshot = {"series": series, "primitives": primitives, "warnings": warnings,
                    "title": title, "x_label": label if kind in ("ecdf", "survival") else "Source",
                    "y_label": ("Empirical F(x) = P(X ≤ x)" if kind == "ecdf" else
                                "Empirical P(X > x)" if kind == "survival" else label),
                    "categorical": kind in ("box", "violin")}
        config["method"] = METHODS[kind]
        self._applied_config = deepcopy(config)
        self._source_recipes = recipes
        self._snapshot, self._tables = snapshot, tables
        self._stale = False
        self._render_snapshot()
        self._update_status()
        self.workspaceChanged.emit()
        return True

    @staticmethod
    def _metric_label(metric, sources, frames):
        if metric == "dwell_s":
            return "Dwell (s)"
        if metric == "amplitude_pA":
            return f"Selected amplitude ({amplitude_units_label(frames)})"
        if metric == "frac_depth":
            return "Fractional depth ΔI/I₀ (dimensionless)"
        domains = list(dict.fromkeys(signal_domain_of(source.provenance) for source in sources))
        return f"Median depth ({domains[0] if len(domains) == 1 else 'mixed source units / domains'})"

    @staticmethod
    def _draw_commands(summary, number, info, color, tables, primitives):
        """Freeze all display geometry at generation, including relative violin width."""
        def add(role, x, y, *, symbols=False, fill=False):
            if not len(x):
                return
            key = f"draw_{len(primitives)}"
            tables[key] = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
            primitives.append({"table": key, "source_key": info["key"], "source_label": info["label"],
                               "role": role, "color": color, "symbols": symbols, "fill": fill,
                               "legend": f"{_display_source_label(info['label'], number)} · n={summary.n}"})
        if not summary.n:
            return
        if summary.kind in ("ecdf", "survival"):
            x, y = summary.step_coordinates()
            add("step", x, y)
        elif summary.kind == "box":
            s = summary.stats
            left, right = number - .28, number + .28
            add("box", [left, right, right, left, left],
                [s["q1"], s["q1"], s["q3"], s["q3"], s["q1"]], fill=True)
            add("median", [left, right], [s["median"], s["median"]])
            for edge, quartile in (("lower_whisker", "q1"), ("upper_whisker", "q3")):
                add(edge, [number, number], [s[edge], s[quartile]])
                add(edge + "_cap", [number - .14, number + .14], [s[edge], s[edge]])
            outside = summary.samples[(summary.samples < s["lower_fence"]) |
                                      (summary.samples > s["upper_fence"])]
            add("outlier", np.full(len(outside), number), outside, symbols=True)
        elif summary.stats.get("bandwidth"):
            widths = .36 * summary.y / summary.y.max()
            x = np.r_[number - widths, (number + widths)[::-1], number - widths[0]]
            y = np.r_[summary.x, summary.x[::-1], summary.x[0]]
            add("violin", x, y, fill=True)
        else:
            add("observed_value", np.full(len(summary.x), number), summary.x, symbols=True)

    def _render_snapshot(self):
        """Painting only. Stored coordinates are authoritative; no estimators here."""
        self._plot.clear()
        self._data_items = []
        entries, seen = [], set()
        for primitive in self._snapshot.get("primitives", []):
            points = self._tables[primitive["table"]]
            color = QtGui.QColor(primitive["color"])
            options = {"pen": None if primitive["symbols"] else pg.mkPen(color, width=1.7)}
            if primitive["symbols"]:
                options.update(symbol="o", symbolSize=6, symbolBrush=color, symbolPen=pg.mkPen(color))
            if primitive["fill"]:
                color.setAlpha(60)
                options.update(fillLevel="enclosed", fillBrush=pg.mkBrush(color))
            item = self._plot.plot(points["x"].to_numpy(copy=True), points["y"].to_numpy(copy=True), **options)
            self._data_items.append(item)
            if primitive["source_key"] not in seen:
                entries.append((item, primitive["legend"]))
                seen.add(primitive["source_key"])
        # Older editable snapshots stored a generated warning prefix in this
        # internal title. Suppress only that prefix for rendering; retain the
        # snapshot and its warnings, and never filter user appearance titles.
        title = self._snapshot.get("title", "Distribution summaries · Generate to begin")
        self._plot.setTitle(title.removeprefix("⚠ Distribution notes · "))
        self._plot.setLabel("bottom", self._snapshot.get("x_label", ""))
        self._plot.setLabel("left", self._snapshot.get("y_label", ""))
        ticks = ([(i, _display_source_label(s["label"], i)) for i, s in enumerate(self._snapshot.get("series", []))]
                 if self._snapshot.get("categorical") else None)
        bottom = self._plot.getAxis("bottom")
        # Labels already state the recorded units. Never silently scale ticks,
        # especially dimensionless probabilities, which must remain in [0, 1].
        bottom.enableAutoSIPrefix(False)
        bottom.setTicks([ticks] if ticks is not None else None)
        self._plot.getAxis("left").enableAutoSIPrefix(False)
        ensure_item_legend(self._plot, entries)
        if self._plot.legend is not None:
            self._plot.legend.show()
        self._appearance_widget.set_legend_available(bool(entries))
        self._sync_source_appearance_editor()
        self._warnings.set_warnings(self._snapshot.get("warnings", []), full_text=self.details_text())
        self._apply_appearance()
        self._plot.enableAutoRange()
        if ticks:
            # Category indices are not physical measurements. Reserve a complete
            # source slot even for a single box/violin, without rescaling its data.
            self._plot.setXRange(-.6, len(ticks) - .4, padding=0)

    def details_text(self):
        if not self._snapshot:
            return ""
        blocks = [self._applied_config.get("method", "")]
        for series in self._snapshot["series"]:
            text = (f"{series['label']}: {series['n']} finite of {series['n_total']} selected observations; "
                    f"{series['n_excluded']} non-finite excluded.")
            if series["stats"]:
                text += "\n" + "; ".join(f"{key}={value:.10g}" for key, value in series["stats"].items())
            blocks.append(text)
        blocks.extend(self._snapshot["warnings"])
        if self._source_appearance:
            blocks.append("Figure source appearance (original identities and numerical tables unchanged):\n" +
                          "\n".join(f"{key}: {style}" for key, style in self._source_appearance.items()))
        return "\n\n".join(blocks)

    def warning_text(self):
        return "\n\n".join(self._snapshot.get("warnings", []))

    def _appearance_from_user(self, state):
        self._appearance = state
        self._apply_appearance()
        self.workspaceChanged.emit()

    def _apply_appearance(self):
        apply_plot_appearance(self._plot, self._appearance, plot_widget=self._plot_widget,
                              data_items=self._data_items)
        self._apply_source_appearance()

    def _sync_source_appearance_editor(self):
        if self._snapshot:
            colors = {p["source_key"]: p["color"] for p in self._snapshot.get("primitives", [])}
            sources = [{**series, "color": colors.get(series["key"], "#0072b2")}
                       for series in self._snapshot.get("series", [])]
        else:
            sources = list(self._sources.values())
        self._source_appearance_editor.set_sources(sources)
        self._source_appearance_editor.set_state(self._source_appearance)

    def _source_appearance_from_user(self, state):
        self.set_source_appearance_state(state)
        self.workspaceChanged.emit()

    def source_appearance_state(self):
        return deepcopy(self._source_appearance)

    def set_source_appearance_state(self, state):
        self._source_appearance = validate_source_appearance(state)
        self._source_appearance_editor.set_state(self._source_appearance)
        self._apply_appearance()
        self._warnings.set_warnings(self._snapshot.get("warnings", []), full_text=self.details_text())

    def _apply_source_appearance(self):
        """Restyle existing frozen primitives; no summary, data or view changes."""
        series = self._snapshot.get("series", [])
        names = compact_source_labels(series, self._source_appearance)
        counts = {source["key"]: source["n"] for source in series}
        entries, seen = [], set()
        ranges = deepcopy(self._plot.viewRange())
        auto = list(self._plot.getViewBox().autoRangeEnabled())
        self._plot.disableAutoRange()
        try:
            for item, primitive in zip(self._data_items, self._snapshot.get("primitives", [])):
                key = primitive["source_key"]
                color = (self._source_appearance.get(key, {}).get("color") or
                         self._appearance.data_color or primitive["color"])
                for option, setter in (("pen", item.setPen), ("symbolPen", item.setSymbolPen)):
                    if option == "symbolPen" and not primitive["symbols"]:
                        continue
                    existing = item.opts.get(option)
                    if existing is not None:
                        pen = pg.mkPen(existing)
                        pen.setColor(QtGui.QColor(color))
                        setter(pen)
                for option, setter in (("fillBrush", item.setFillBrush), ("symbolBrush", item.setSymbolBrush)):
                    if option == "symbolBrush" and not primitive["symbols"]:
                        continue
                    existing = item.opts.get(option)
                    if existing is not None:
                        brush = pg.mkBrush(existing)
                        tint = QtGui.QColor(color)
                        tint.setAlpha(brush.color().alpha())
                        brush.setColor(tint)
                        setter(brush)
                if key not in seen:
                    entries.append((item, f"{names[key]} · n={counts[key]}"))
                    seen.add(key)
            ensure_item_legend(self._plot, entries)
            if self._plot.legend is not None:
                self._plot.legend.setVisible(bool(entries) and self._appearance.legend_visible is not False)
            if self._snapshot.get("categorical"):
                self._plot.getAxis("bottom").setTicks([
                    [(i, names[source["key"]]) for i, source in enumerate(series)]])
        finally:
            self._plot.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
            for axis, enabled in enumerate(auto):
                if enabled:
                    self._plot.enableAutoRange(axis=axis, enable=enabled)

    def appearance_state(self):
        return self._appearance.to_dict()

    def set_appearance_state(self, state):
        self._appearance = state if isinstance(state, PlotAppearance) else PlotAppearance.from_mapping(state)
        self._appearance_widget.set_state(self._appearance)
        self._apply_appearance()

    def has_data(self):
        return bool(self._snapshot.get("primitives"))

    def figure_applied_config(self):
        return deepcopy(self._applied_config)

    def figure_source_recipes(self):
        return deepcopy(self._source_recipes)

    def figure_export_target(self):
        return self._plot

    def _plot_data(self):
        """Exact visible coordinates, not a regenerated ECDF/KDE or draft recipe."""
        parts = []
        for i, primitive in enumerate(self._snapshot.get("primitives", [])):
            frame = self._tables[primitive["table"]].copy(deep=True)
            frame.insert(0, "point", np.arange(len(frame)))
            frame.insert(0, "role", primitive["role"])
            frame.insert(0, "primitive", i)
            frame.insert(0, "source_label", primitive["source_label"])
            frame.insert(0, "source_key", primitive["source_key"])
            parts.append(frame)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_PLOT_COLUMNS)

    def snapshot_state(self):
        payload = {"schema_version": 1, "draft": self.config(),
                   "applied": deepcopy(self._applied_config), "appearance": self.appearance_state(),
                   "source_appearance": deepcopy(self._source_appearance),
                   "source_recipes": self.figure_source_recipes(), "snapshot": deepcopy(self._snapshot),
                   "stale": self._stale, "view_range": deepcopy(self._plot.viewRange())}
        return payload, {key: frame.copy(deep=True) for key, frame in self._tables.items()}

    def restore_state(self, payload, tables):
        """Load a frozen figure without statistical computation or live source reads."""
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValueError("Unsupported distribution summary snapshot version.")
        def validate_config(value, *, empty=False):
            if not isinstance(value, dict):
                raise ValueError("Invalid saved distribution settings.")
            if empty and not value:
                return
            keys = value.get("source_keys")
            if (not isinstance(keys, list) or any(not isinstance(key, str) for key in keys)
                    or len(set(keys)) != len(keys)
                    or value.get("metric") not in dict(METRICS)
                    or value.get("kind") not in dict(VIEWS)
                    or type(value.get("accepted_only")) is not bool
                    or type(value.get("violin_points")) is not int
                    or not 32 <= value["violin_points"] <= 2048
                    or not isinstance(value.get("method", ""), str)):
                raise ValueError("Invalid saved distribution settings.")
        draft = deepcopy(payload.get("draft"))
        applied = deepcopy(payload.get("applied", {}))
        validate_config(draft)
        validate_config(applied, empty=True)
        snapshot = deepcopy(payload.get("snapshot", {}))
        if not isinstance(snapshot, dict) or not isinstance(tables, dict):
            raise ValueError("Invalid saved distribution snapshot.")
        if any(not isinstance(key, str) or not isinstance(frame, pd.DataFrame) for key, frame in tables.items()):
            raise ValueError("Invalid saved distribution table.")
        restored = {key: frame.copy(deep=True) for key, frame in tables.items()}
        if snapshot and (not applied or any(not isinstance(snapshot.get(key), str)
                           for key in ("title", "x_label", "y_label"))
                         or type(snapshot.get("categorical")) is not bool
                         or any(not isinstance(snapshot.get(key), list)
                                for key in ("series", "primitives", "warnings"))
                         or any(not isinstance(note, str) for note in snapshot["warnings"])):
            raise ValueError("Invalid saved distribution description.")
        series_keys = set()
        for series in snapshot.get("series", []):
            if (not isinstance(series, dict)
                    or any(not isinstance(series.get(key), str) for key in ("key", "label", "source_table"))
                    or series["key"] in series_keys or series["source_table"] not in restored
                    or any(type(series.get(key)) is not int or series[key] < 0
                           for key in ("n", "n_total", "n_excluded"))
                    or series["n"] + series["n_excluded"] != series["n_total"]
                    or len(restored[series["source_table"]]) != series["n_total"]
                    or not isinstance(series.get("stats"), dict)
                    or any(not isinstance(key, str) or type(value) not in (float, int)
                           or not np.isfinite(value) for key, value in series["stats"].items())):
                raise ValueError("Invalid saved distribution source summary.")
            series_keys.add(series["key"])
        if snapshot and series_keys != set(applied["source_keys"]):
            raise ValueError("Saved distribution sources disagree with the applied settings.")
        # Validate before replacing anything, including every coordinate passed to Qt.
        for primitive in snapshot.get("primitives", []):
            if (not isinstance(primitive, dict)
                    or any(not isinstance(primitive.get(key), str)
                           for key in ("table", "source_key", "source_label", "role", "color", "legend"))
                    or primitive["source_key"] not in series_keys
                    or any(type(primitive.get(key)) is not bool for key in ("symbols", "fill"))):
                raise ValueError("Invalid saved distribution drawing instruction.")
            table = restored.get(primitive["table"])
            if table is None or list(table.columns) != ["x", "y"]:
                raise ValueError("A distribution drawing table is missing or malformed.")
            if any(table[col].dtype.kind not in "iuf" for col in ("x", "y")):
                raise ValueError("Distribution drawing coordinates must be real numeric arrays.")
            if not np.isfinite(table.to_numpy()).all():
                raise ValueError("Distribution drawing coordinates must be finite.")
            if not QtGui.QColor(str(primitive.get("color", ""))).isValid():
                raise ValueError("Invalid saved distribution source color.")
        recipes = deepcopy(payload.get("source_recipes", []))
        if not isinstance(recipes, list) or any(not isinstance(recipe, dict) for recipe in recipes):
            raise ValueError("Invalid saved distribution source recipes.")
        if type(payload.get("stale", False)) is not bool or not isinstance(payload.get("appearance", {}), dict):
            raise ValueError("Invalid saved distribution presentation state.")
        appearance = PlotAppearance.from_mapping(payload.get("appearance", {}))
        source_appearance = validate_source_appearance(payload.get("source_appearance", {}))
        ranges = payload.get("view_range")
        if ranges is not None:
            try:
                ranges = np.asarray(ranges, dtype=float)
                if ranges.shape != (2, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
                    raise ValueError
                ranges = ranges.tolist()
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid saved distribution view range.") from exc
        self.set_config(draft)
        self._snapshot, self._tables = snapshot, restored
        self._applied_config = applied
        self._source_recipes = recipes
        self._appearance = appearance
        self._source_appearance = source_appearance
        self._appearance_widget.set_state(self._appearance)
        self._stale = bool(payload.get("stale", False))
        self._render_snapshot()
        if ranges is not None:
            self._plot.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
        self._update_status()

    def save_image(self, path=None):
        if not self.has_data():
            return None
        kwargs = {}
        if path is None:
            chosen = choose_live_plot_export(self, self._plot, suggestion="distribution_summary.png")
            if chosen is None:
                return None
            path, options = chosen
            kwargs = {"width": options.width, "height": options.height,
                      "physical_width_mm": options.physical_width_mm,
                      "physical_height_mm": options.physical_height_mm,
                      "frozen_snapshot": options.frozen_snapshot}
            if options.extension == ".png" and not options.native:
                kwargs["dpi"] = options.dpi
        out = Path(path)
        from ..plot.figures import image_write_failure_text
        try:
            notes = write_live_plot_image(out, self._plot, **kwargs)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export distribution figure",
                                         image_write_failure_text(out, exc, formats=IMAGE_FORMATS))
            return None
        if notes:
            QtWidgets.QMessageBox.information(self, "Export distribution figure", "\n\n".join(notes))
        return out

    def save_csv(self, path=None):
        if not self.has_data():
            return None
        if path is None:
            path = ask_export_path(self, title="Export distribution coordinates",
                                   suggestion="distribution_summary.csv", filters=DATA_FILTERS)
            if path is None:
                return None
        out = Path(path)
        frame = self._plot_data()
        frame["applied_config_json"] = json.dumps(self.figure_applied_config(), ensure_ascii=False)
        frame["source_recipes_json"] = json.dumps(self.figure_source_recipes(), ensure_ascii=False)
        frame["notes"] = self.details_text()
        try:
            notes = write_plot_csv(out, frame)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export distribution coordinates", str(exc))
            return None
        if notes:
            QtWidgets.QMessageBox.information(self, "Export distribution coordinates", "\n\n".join(notes))
        return out
