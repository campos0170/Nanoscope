"""Explicit, snapshot-backed publication comparisons of existing event tables.

Generating prepares plot geometry, never detection/calibration/fitted results.
Styling and restore draw that saved geometry; they do not rerun statistics or
swarm layout. Canvas magnification is a viewer aid, not a numerical axis edit.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.analysis_workspace import _plot_source_recipe, fingerprint_frame
from ..core.sources import amplitude_units_label, signal_domain_of, source_comparability
from ..core.source_style import validate_source_appearance
from .analysis_catalog import ANALYSIS_LABELS
from .export_paths import ask_export_path
from .plot_appearance import PlotAppearance, PlotAppearanceWidget
from .plot_export import DATA_FILTERS, IMAGE_FORMATS, choose_live_plot_export, write_live_plot_image, write_plot_csv
from .plot_workflow import WarningSummary, bounded_controls, control_group, install_plot_workflow
from .source_appearance import SourceAppearanceEditor


VIEWS = (("facets", "Faceted comparisons"), ("pairs", "Pair-plot matrix"),
         ("events", "Individual-event overlays"))
METRICS = (("dwell_s", "Dwell (s)"), ("depth_pA", "Median depth (recorded units)"),
           ("amplitude_pA", "Selected amplitude (recorded units)"),
           ("frac_depth", "Fractional depth ΔI/I₀"),
           ("ecd_pA_s", "Event area (recorded units · s)"), ("snr", "SNR"),
           ("rise_s", "Rise time (s)"), ("fall_s", "Fall time (s)"))
_METRIC_KEYS = {key for key, _ in METRICS}
_KINDS = {key for key, _ in VIEWS}
METHODS = {
    "facets": "One scatter panel per recording, with shared numerical axes. No pooling, regression, smoothing or uncertainty estimation.",
    "pairs": "Two to five features: scatter panels off the diagonal and separate-source count histograms on the diagonal. Bins are shared per feature. Missing values are excluded per panel, not by deleting every row with any missing feature.",
    "events": "Every retained observation is drawn as a strip or swarm point. Optional boxes use NanoScope's linear quartiles and 1.5-IQR sample whiskers, not confidence intervals. Horizontal spreading is frozen on Generate and is not a measured variable. Larger markers may overlap without moving any event.",
}


class _ComparisonCanvas(QtWidgets.QGraphicsView):
    """A bounded magnifier for a fixed publication canvas, not an analysis editor."""

    viewChanged = QtCore.Signal()

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setMinimumHeight(180)
        self.setAccessibleName("Comparison figure canvas")
        self.setBackgroundBrush(QtGui.QColor("#dce3e7"))

    def zoom(self, factor):
        scale = self.transform().m11() * factor
        if .01 <= scale <= 100:
            self.scale(factor, factor)
            self.viewChanged.emit()

    def wheelEvent(self, event):  # noqa: N802
        if event.angleDelta().y():
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        super().mouseReleaseEvent(event)
        self.viewChanged.emit()


class ComparisonFiguresWindow(QtWidgets.QMainWindow):
    workspaceChanged = QtCore.Signal()
    refreshRequested = QtCore.Signal()

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self._session = session
        self.setWindowTitle(f"NanoScope — {ANALYSIS_LABELS['comparison_figures']}")
        self.resize(1100, 800)
        self.setMinimumSize(700, 500)
        self._sources = {}
        self._source_stamps = {}
        self._selected_keys = []
        self._loading = False
        self._stale = False
        self._snapshot = {}
        self._tables = {}
        self._applied_config = {}
        self._source_recipes = []
        self._appearance = PlotAppearance()
        self._source_appearance = {}
        self._figure_item = None
        self._fit_pending = False

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)
        self._status = QtWidgets.QLabel()
        self._status.setTextFormat(QtCore.Qt.PlainText)
        self._status.setWordWrap(True)
        self._status.setMaximumHeight(50)
        layout.addWidget(self._status)

        data = QtWidgets.QGroupBox("Data · existing measurements, separately identified sources")
        data_layout = QtWidgets.QGridLayout(data)
        self._source_list = QtWidgets.QListWidget()
        self._source_list.setAccessibleName("Comparison event sources")
        self._source_list.setMaximumHeight(105)
        self._accepted_only = QtWidgets.QCheckBox("Accepted events only")
        self._accepted_only.setChecked(True)
        self._view_combo = QtWidgets.QComboBox()
        for key, label in VIEWS:
            self._view_combo.addItem(label, key)
        self._view_combo.setAccessibleName("Comparison figure type")
        data_layout.addWidget(self._source_list, 0, 0, 3, 1)
        data_layout.addWidget(QtWidgets.QLabel("Figure"), 0, 1)
        data_layout.addWidget(self._view_combo, 0, 2)
        data_layout.addWidget(self._accepted_only, 1, 1, 1, 2)
        note = QtWidgets.QLabel("Uses stored values and identities. No automatic recalibration, imputation or subsampling.")
        note.setWordWrap(True)
        data_layout.addWidget(note, 2, 1, 1, 2)

        analysis = QtWidgets.QWidget()
        analysis_layout = QtWidgets.QVBoxLayout(analysis)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        self._facet_controls = QtWidgets.QGroupBox("Facets · one panel per source")
        facets = QtWidgets.QGridLayout(self._facet_controls)
        self._x_combo = self._metric_combo("Faceted x measurement", "dwell_s")
        self._y_combo = self._metric_combo("Faceted y measurement", "depth_pA")
        self._columns = QtWidgets.QSpinBox()
        self._columns.setRange(1, 3)
        self._columns.setValue(2)
        self._columns.setAccessibleName("Facet columns")
        self._log_x = QtWidgets.QCheckBox("Log x")
        self._log_y = QtWidgets.QCheckBox("Log y")
        for row, (label, control) in enumerate((("X", self._x_combo), ("Y", self._y_combo), ("Columns", self._columns))):
            facets.addWidget(QtWidgets.QLabel(label), row, 0)
            facets.addWidget(control, row, 1)
        facets.addWidget(self._log_x, 0, 2)
        facets.addWidget(self._log_y, 1, 2)
        analysis_layout.addWidget(self._facet_controls)

        self._pair_controls = QtWidgets.QGroupBox("Pair matrix · select two to five features")
        pairs = QtWidgets.QHBoxLayout(self._pair_controls)
        self._features = QtWidgets.QListWidget()
        self._features.setMaximumHeight(115)
        self._features.setAccessibleName("Pair matrix features")
        for key, label in METRICS:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if key in {"dwell_s", "depth_pA", "ecd_pA_s", "snr"} else QtCore.Qt.Unchecked)
            self._features.addItem(item)
        pairs.addWidget(self._features, 1)
        pair_options = QtWidgets.QVBoxLayout()
        pair_options.addWidget(QtWidgets.QLabel("Diagonal histogram bins"))
        self._bins = QtWidgets.QSpinBox()
        self._bins.setRange(5, 200)
        self._bins.setValue(40)
        self._bins.setAccessibleName("Pair histogram bins")
        pair_options.addWidget(self._bins)
        pair_note = QtWidgets.QLabel("Linear axes; separate-source counts. No KDE or fitted lines.")
        pair_note.setWordWrap(True)
        pair_options.addWidget(pair_note)
        pairs.addLayout(pair_options)
        analysis_layout.addWidget(self._pair_controls)

        self._event_controls = QtWidgets.QGroupBox("Individual observations · optional descriptive box overlay")
        events = QtWidgets.QGridLayout(self._event_controls)
        self._metric = self._metric_combo("Individual-event measurement", "depth_pA")
        self._points = QtWidgets.QComboBox()
        self._points.addItem("Strip · deterministic horizontal spread", "strip")
        self._points.addItem("Swarm · non-overlapping layout", "swarm")
        self._points.setAccessibleName("Individual-event point layout")
        self._show_box = QtWidgets.QCheckBox("Overlay box summary (linear quartiles; 1.5 IQR)")
        self._show_box.setChecked(True)
        self._event_log_y = QtWidgets.QCheckBox("Log measurement axis")
        events.addWidget(QtWidgets.QLabel("Measurement"), 0, 0)
        events.addWidget(self._metric, 0, 1)
        events.addWidget(QtWidgets.QLabel("Points"), 1, 0)
        events.addWidget(self._points, 1, 1)
        events.addWidget(self._show_box, 2, 0, 1, 2)
        events.addWidget(self._event_log_y, 3, 0, 1, 2)
        analysis_layout.addWidget(self._event_controls)
        self._method_note = QtWidgets.QLabel()
        self._method_note.setWordWrap(True)
        self._method_note.setTextFormat(QtCore.Qt.PlainText)
        analysis_layout.addWidget(self._method_note)
        analysis_scroll = bounded_controls(analysis, maximum_height=240)

        self._appearance_widget = PlotAppearanceWidget(self._appearance, multi_view_labels=True)
        self._appearance_widget.data_color.setEnabled(False)
        self._appearance_widget.data_color.setToolTip("Comparison figures retain each source's identity colour; a single colour override is not applied.")
        self._appearance_widget.fit_color.setEnabled(False)
        self._appearance_widget.fit_color.setToolTip("Comparison figures do not fit models.")
        appearance_content = QtWidgets.QWidget()
        appearance_layout = QtWidgets.QVBoxLayout(appearance_content)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        self._source_appearance_editor = SourceAppearanceEditor(session=session)
        appearance_layout.addWidget(self._source_appearance_editor)
        appearance_layout.addWidget(self._appearance_widget)
        appearance = bounded_controls(appearance_content, maximum_height=235)
        export = control_group("Export the generated figure", "Export the full publication canvas as PNG, PDF or SVG. "
            "Canvas zoom is only a magnifier. Save to Figure Library keeps observations, plotted geometry, "
            "settings and full Details together. Draft controls never change a saved picture.")
        for widget in (data, analysis_scroll, appearance, export):
            layout.addWidget(widget)

        canvas_bar = QtWidgets.QHBoxLayout()
        canvas_bar.addWidget(QtWidgets.QLabel("Canvas magnification · exports retain the full figure"), 1)
        self._fit_button = QtWidgets.QPushButton("Fit canvas")
        self._zoom_out = QtWidgets.QPushButton("−")
        self._zoom_in = QtWidgets.QPushButton("+")
        self._zoom_out.setAccessibleName("Zoom out comparison canvas")
        self._zoom_in.setAccessibleName("Zoom in comparison canvas")
        for button in (self._fit_button, self._zoom_out, self._zoom_in):
            canvas_bar.addWidget(button)
        layout.addLayout(canvas_bar)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._view = _ComparisonCanvas(self._scene, self)
        layout.addWidget(self._view, 1)
        self._warnings = WarningSummary(title="Comparison notes", maximum_height=135)
        self._warnings.toggle.setText("Details — methods and provenance")
        layout.addWidget(self._warnings)
        self._workflow = install_plot_workflow(self,
            sections={"Data": (data,), "Analysis": (analysis_scroll,), "Appearance": (appearance,), "Export": (export,)},
            actions=(("Generate", self._generate_from_user),),
            exports=(("Export current figure…", self.save_image), ("Export plotted values CSV…", self.save_csv)))
        self._source_list.itemChanged.connect(self._source_selection_changed)
        self._accepted_only.toggled.connect(self._draft_changed)
        for combo in (self._view_combo, self._x_combo, self._y_combo, self._metric, self._points):
            combo.currentIndexChanged.connect(self._draft_changed)
        for check in (self._log_x, self._log_y, self._event_log_y, self._show_box):
            check.toggled.connect(self._draft_changed)
        self._bins.valueChanged.connect(self._draft_changed)
        self._columns.valueChanged.connect(self._draft_changed)
        self._features.itemChanged.connect(self._draft_changed)
        self._appearance_widget.changed.connect(self._appearance_from_user)
        self._source_appearance_editor.changed.connect(self._source_appearance_from_user)
        self._fit_button.clicked.connect(self.fit_canvas)
        self._zoom_out.clicked.connect(lambda: self._view.zoom(1 / 1.25))
        self._zoom_in.clicked.connect(lambda: self._view.zoom(1.25))
        self._view.viewChanged.connect(self._view_changed)
        self._update_method_note()
        self._update_status()

    @staticmethod
    def _metric_combo(name, default):
        combo = QtWidgets.QComboBox()
        combo.setAccessibleName(name)
        combo.setMinimumContentsLength(14)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        for key, label in METRICS:
            combo.addItem(label, key)
        combo.setCurrentIndex(combo.findData(default))
        return combo

    def config(self):
        kind = self._view_combo.currentData()
        return {"source_keys": list(self._selected_keys), "accepted_only": self._accepted_only.isChecked(),
                "kind": kind, "x": self._x_combo.currentData(), "y": self._y_combo.currentData(),
                "metric": self._metric.currentData(), "features": [self._features.item(i).data(QtCore.Qt.UserRole)
                    for i in range(self._features.count()) if self._features.item(i).checkState() == QtCore.Qt.Checked],
                "bins": self._bins.value(), "columns": self._columns.value(),
                "points": self._points.currentData(), "show_box": self._show_box.isChecked(),
                "log_x": self._log_x.isChecked() if kind == "facets" else False,
                "log_y": (self._event_log_y.isChecked() if kind == "events" else
                          self._log_y.isChecked() if kind == "facets" else False)}

    @staticmethod
    def _validate_config(config):
        if not isinstance(config, dict) or config.get("kind") not in _KINDS:
            raise ValueError("Invalid comparison figure type.")
        if any(config.get(key) not in _METRIC_KEYS for key in ("x", "y", "metric")):
            raise ValueError("Invalid comparison measurement.")
        for key in ("accepted_only", "show_box", "log_x", "log_y"):
            if type(config.get(key)) is not bool:
                raise ValueError("Invalid comparison option: " + key)
        for key, low, high in (("bins", 5, 200), ("columns", 1, 3)):
            if type(config.get(key)) is not int or not low <= config[key] <= high:
                raise ValueError("Invalid comparison option: " + key)
        if config.get("points") not in {"strip", "swarm"}:
            raise ValueError("Invalid individual-event layout.")
        for key in ("source_keys", "features"):
            values = config.get(key)
            if (not isinstance(values, list) or any(not isinstance(v, str) for v in values)
                    or len(values) != len(set(values))):
                raise ValueError("Invalid comparison selection: " + key)
        if any(key not in _METRIC_KEYS for key in config["features"]):
            raise ValueError("Invalid pair-matrix feature.")

    def set_config(self, config):
        merged = {**self.config(), **deepcopy(config)}
        self._validate_config(merged)
        previous = self._loading
        self._loading = True
        try:
            self._selected_keys = list(merged["source_keys"])
            for combo, key in ((self._view_combo, "kind"), (self._x_combo, "x"),
                               (self._y_combo, "y"), (self._metric, "metric"), (self._points, "points")):
                combo.setCurrentIndex(combo.findData(merged[key]))
            self._accepted_only.setChecked(merged["accepted_only"])
            self._show_box.setChecked(merged["show_box"])
            self._log_x.setChecked(merged["log_x"])
            self._log_y.setChecked(merged["log_y"])
            self._event_log_y.setChecked(merged["log_y"])
            self._bins.setValue(merged["bins"])
            self._columns.setValue(merged["columns"])
            for i in range(self._features.count()):
                item = self._features.item(i)
                item.setCheckState(QtCore.Qt.Checked if item.data(QtCore.Qt.UserRole) in merged["features"] else QtCore.Qt.Unchecked)
            self._populate_sources()
            self._update_method_note()
        finally:
            self._loading = previous

    def selected_sources(self):
        return [self._sources[key] for key in self._selected_keys if key in self._sources]

    def _populate_sources(self):
        blocker = QtCore.QSignalBlocker(self._source_list)
        self._source_list.clear()
        for key in [*self._sources, *[key for key in self._selected_keys if key not in self._sources]]:
            label = str(self._sources[key].label) if key in self._sources else "Unavailable source: " + key
            item = QtWidgets.QListWidgetItem(label if len(label) <= 110 else label[:107] + "…")
            item.setToolTip(label)
            item.setData(QtCore.Qt.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if key in self._selected_keys else QtCore.Qt.Unchecked)
            self._source_list.addItem(item)
        del blocker
        self._source_appearance_editor.set_sources(self._snapshot.get("sources") or self._sources.values())

    def set_sources(self, sources, selected_keys=None, analyze=False):
        sources = list(sources)
        offered = {str(source.key): source for source in sources}
        if len(offered) != len(sources):
            raise ValueError("Comparison sources must have unique identities.")
        stamps = {key: (fingerprint_frame(source.df), repr(dict(source.provenance)),
                        repr(source.calibration_fingerprint), tuple(source.warnings)) for key, source in offered.items()}
        before = tuple((key, self._source_stamps.get(key)) for key in self._selected_keys)
        self._sources = offered
        if selected_keys is not None:
            self._selected_keys = list(dict.fromkeys(str(key) for key in selected_keys))
        elif not self._selected_keys and not self._snapshot:
            self._selected_keys = [key for key, source in offered.items() if source.current][:1] or list(offered)[:1]
        self._source_stamps = stamps
        self._populate_sources()
        if before != tuple((key, stamps.get(key)) for key in self._selected_keys):
            self._stale = bool(self._snapshot)
            self._update_status()
            self.workspaceChanged.emit()
        if analyze:
            self.analyze()

    def _source_selection_changed(self, *_args):
        self._selected_keys = [str(self._source_list.item(i).data(QtCore.Qt.UserRole))
            for i in range(self._source_list.count()) if self._source_list.item(i).checkState() == QtCore.Qt.Checked]
        self._draft_changed()

    def _update_method_note(self):
        kind = self._view_combo.currentData()
        for widget, key in ((self._facet_controls, "facets"), (self._pair_controls, "pairs"), (self._event_controls, "events")):
            widget.setVisible(kind == key)
        limits = (" Limits: 8 sources; 50,000 selected rows." if kind == "facets" else
                  " Limits: 8 sources; 10,000 selected rows; 250,000 drawn points." if kind == "pairs" else
                  " Limits: 8 sources; 50,000 selected rows. Swarm: 500 per source / 2,000 total; use Strip for larger sets.")
        self._method_note.setText(METHODS[kind] + limits)
        # A matrix's axes refer to different features. One global text override
        # would mislabel them; retain per-feature scientific labels.
        label_kind = self._applied_config.get("kind", kind)
        for name in ("x_label", "y_label"):
            widget = getattr(self._appearance_widget, name, None)
            if widget is not None:
                widget.setEnabled(label_kind != "pairs")
                widget.setToolTip("Pair matrices retain each feature's own label." if label_kind == "pairs" else "Presentation-only label override.")

    def _draft_changed(self, *_args):
        if self._loading:
            return
        self._update_method_note()
        self._stale = bool(self._snapshot)
        self._update_status()
        self.workspaceChanged.emit()

    def _update_status(self):
        if self._snapshot:
            counts = "; ".join(f"{source['label']}: {source['n']} selected rows" for source in self._snapshot.get("sources", []))
            text = ("Unapplied data/analysis changes · showing the previous figure. " if self._stale else "Generated snapshot · ") + counts
        else:
            text = "Choose stored event sources and a figure type, then Generate. Opening this page runs no analysis."
        self._status.setText(text if len(text) <= 220 else text[:217] + "…")
        self._status.setToolTip(text)
        self._status.setAccessibleDescription(text)
        self._workflow.primary_actions[0].setText("Generate ●" if self._stale else "Generate")
        self._workflow.export_button.setEnabled(self.has_data())
        for button in (self._fit_button, self._zoom_in, self._zoom_out):
            button.setEnabled(self.has_data())

    def _generate_from_user(self):
        self.refreshRequested.emit()
        try:
            self.analyze()
        except (TypeError, ValueError, OverflowError, RuntimeError, ImportError) as exc:
            QtWidgets.QMessageBox.warning(self, "Generate comparison figure", str(exc))

    def refresh(self):
        return self.analyze()

    @staticmethod
    def _metric_labels(sources, frames):
        domains = list(dict.fromkeys(signal_domain_of(source.provenance) for source in sources))
        domain = domains[0] if len(domains) == 1 else "mixed source units / domains"
        return {"dwell_s": "Dwell (s)", "depth_pA": f"Median depth ({domain})",
                "amplitude_pA": f"Selected amplitude ({amplitude_units_label(frames)})",
                "frac_depth": "Fractional depth ΔI/I₀ (dimensionless)",
                "ecd_pA_s": f"Event area ({domain} · s)", "snr": "SNR (dimensionless)",
                "rise_s": "Rise time (s)", "fall_s": "Fall time (s)"}

    def analyze(self):
        """Prepare and render transactionally. The source tables are read-only."""
        from ..plot.comparison_figures import prepare_comparison

        config = self.config()
        self._validate_config(config)
        if not config["source_keys"]:
            raise ValueError("Select at least one event source.")
        if any(key not in self._sources for key in config["source_keys"]):
            raise ValueError("A selected source is unavailable. Rebind or uncheck it before Generate.")
        sources = self.selected_sources()
        frames, descriptors, notes, recipes = [], [], [], []
        for source in sources:
            frame = source.df.copy(deep=True)
            # Positional identity is independent of duplicate index labels/event IDs.
            # This reserved column is internal to the private rendering copy only.
            if "_nanoscope_row" in frame:
                raise ValueError("Source has a reserved comparison identity column: _nanoscope_row.")
            frame["_nanoscope_row"] = np.arange(len(frame), dtype=np.int64)
            if config["accepted_only"]:
                if "accepted" not in frame:
                    notes.append(f"{source.label}: no accepted flag is stored; all source rows are included.")
                else:
                    frame = frame.loc[frame["accepted"].fillna(False).astype(bool)].copy(deep=True)
            frames.append(frame)
            descriptors.append({"key": str(source.key), "label": str(source.label), "color": str(source.colour), "frame": frame})
            recipes.append(_plot_source_recipe(self, source))
            notes.extend(f"{source.label}: {note}" for note in source.warnings)
        metrics = (config["features"] if config["kind"] == "pairs" else
                   [config["x"], config["y"]] if config["kind"] == "facets" else [config["metric"]])
        notes.extend(source_comparability(sources, metrics, frames=frames).warnings)
        config["labels"] = self._metric_labels(sources, frames)
        snapshot, tables = prepare_comparison(descriptors, config)
        snapshot = deepcopy(snapshot)
        snapshot["warnings"] = list(dict.fromkeys([*snapshot.get("warnings", []), *notes]))
        snapshot["details"] = "\n\n".join(filter(None, [METHODS[config["kind"]], str(snapshot.get("details", "")), *notes]))
        item = self._make_figure_item(snapshot, tables, self._appearance, self._source_appearance)
        self._snapshot, self._tables = snapshot, {key: frame.copy(deep=True) for key, frame in tables.items()}
        self._source_recipes = recipes
        self._applied_config = deepcopy(config)
        self._stale = False
        self._source_appearance_editor.set_sources(snapshot["sources"])
        self._update_method_note()
        self._install_figure_item(item, fit=True)
        self._warnings.set_warnings(self._snapshot["warnings"], full_text=self.details_text())
        self._update_status()
        self.workspaceChanged.emit()
        return True

    @staticmethod
    def _make_figure_item(snapshot, tables, appearance, source_appearance=None):
        from ..plot.comparison_figures import render_comparison
        from .matplotlib_figure_item import MatplotlibFigureItem

        figure = render_comparison(snapshot, tables, appearance, source_appearance=source_appearance)
        try:
            return MatplotlibFigureItem(figure)
        finally:
            # Renderer returns a detached Figure, never a pyplot-managed window.
            figure.clear()

    def _install_figure_item(self, item, *, fit=False):
        view = self._view_state() if self._figure_item is not None else None
        self._scene.clear()
        self._figure_item = item
        self._scene.addItem(item)
        self._scene.setSceneRect(item.boundingRect())
        if fit:
            self._fit_pending = not self.isVisible()
            self.fit_canvas(emit=False)
        elif view is not None:
            self._restore_view(view)

    def fit_canvas(self, _checked=False, *, emit=True):
        if self._figure_item is None:
            return
        self._view.fitInView(self._figure_item.boundingRect(), QtCore.Qt.KeepAspectRatio)
        if emit:
            self._view_changed()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self._fit_pending:
            self._fit_pending = False
            self.fit_canvas(emit=False)

    def _view_state(self):
        center = self._view.mapToScene(self._view.viewport().rect().center())
        return {"scale": float(self._view.transform().m11()), "center": [center.x(), center.y()]}

    @staticmethod
    def _validate_view(view):
        if (not isinstance(view, dict) or type(view.get("scale")) not in (int, float)
                or not math.isfinite(view["scale"]) or not .001 <= view["scale"] <= 1000
                or not isinstance(view.get("center"), list) or len(view["center"]) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e9 for value in view["center"])):
            raise ValueError("Invalid comparison canvas view.")

    def _restore_view(self, view):
        self._view.setTransform(QtGui.QTransform.fromScale(view["scale"], view["scale"]))
        self._view.centerOn(*view["center"])

    def _view_changed(self):
        if not self._loading and self.has_data():
            self.workspaceChanged.emit()

    def _appearance_from_user(self, state):
        previous = self._appearance
        try:
            self.set_appearance_state(state)
        except (TypeError, ValueError, RuntimeError, ImportError) as exc:
            self._appearance_widget.set_state(previous)
            QtWidgets.QMessageBox.warning(self, "Style comparison figure", str(exc))
            return
        self.workspaceChanged.emit()

    def appearance_state(self):
        return self._appearance.to_dict()

    def set_appearance_state(self, state):
        appearance = state if isinstance(state, PlotAppearance) else PlotAppearance.from_mapping(state)
        item = self._make_figure_item(self._snapshot, self._tables, appearance, self._source_appearance) if self._snapshot else None
        self._appearance = appearance
        self._appearance_widget.set_state(appearance)
        if item is not None:
            self._install_figure_item(item)

    def details_text(self):
        text = str(self._snapshot.get("details", ""))
        if self._source_appearance:
            aliases = [f"{key}: {entry}" for key, entry in self._source_appearance.items()]
            text += "\n\nFigure-local appearance (source identities and measured values unchanged):\n" + "\n".join(aliases)
        return text

    def source_appearance_state(self):
        return deepcopy(self._source_appearance)

    def set_source_appearance_state(self, state):
        overrides = validate_source_appearance(state)
        item = self._make_figure_item(self._snapshot, self._tables, self._appearance, overrides) if self._snapshot else None
        self._source_appearance = overrides
        self._source_appearance_editor.set_state(overrides)
        if item is not None:
            self._install_figure_item(item)
            self._warnings.set_warnings(self._snapshot["warnings"], full_text=self.details_text())

    def _source_appearance_from_user(self, state):
        previous = self.source_appearance_state()
        try:
            self.set_source_appearance_state(state)
        except (TypeError, ValueError, RuntimeError, ImportError) as exc:
            self._source_appearance_editor.set_state(previous)
            QtWidgets.QMessageBox.warning(self, "Style comparison sources", str(exc))
            return
        self.workspaceChanged.emit()

    def warning_text(self):
        return "\n\n".join(self._snapshot.get("warnings", []))

    def has_data(self):
        return bool(self._snapshot and self._figure_item is not None)

    def figure_applied_config(self):
        return deepcopy(self._applied_config)

    def figure_source_recipes(self):
        return deepcopy(self._source_recipes)

    def figure_export_target(self):
        if self._figure_item is None:
            raise ValueError("Generate a comparison figure first.")
        return self._figure_item

    def _plot_data(self):
        parts = []
        panels = {panel["id"]: panel for panel in self._snapshot.get("panels", [])}
        labels = {source["key"]: source["label"] for source in self._snapshot.get("sources", [])}
        for name in ("plotted", "histograms", "boxes"):
            frame = self._tables.get(name)
            if frame is not None and not frame.empty:
                frame = frame.copy(deep=True)
                frame.insert(0, "primitive", name)
                if "source_key" in frame:
                    frame["source_label"] = frame["source_key"].map(labels)
                if "panel_id" in frame:
                    for column, key in (("x_measurement", "x"), ("y_measurement", "y"),
                                        ("x_axis_label", "x_label"), ("y_axis_label", "y_label"),
                                        ("x_scale", "xscale"), ("y_scale", "yscale")):
                        frame[column] = frame["panel_id"].map({pid: panel.get(key, "") for pid, panel in panels.items()})
                frame["accepted_only"] = bool(self._applied_config.get("accepted_only", True))
                parts.append(frame)
        return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()

    def snapshot_state(self):
        return ({"schema_version": 1, "draft": self.config(), "applied": self.figure_applied_config(),
                 "appearance": self.appearance_state(), "snapshot": deepcopy(self._snapshot),
                 "source_appearance": self.source_appearance_state(),
                 "source_recipes": self.figure_source_recipes(), "stale": self._stale,
                 "canvas_view": self._view_state()},
                {key: frame.copy(deep=True) for key, frame in self._tables.items()})

    def restore_state(self, payload, tables):
        """Validate and redraw frozen geometry without touching source data or kernels."""
        from ..plot.comparison_figures import validate_comparison

        if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
                or payload["schema_version"] != 1 or type(payload.get("stale")) is not bool
                or not isinstance(payload.get("appearance"), dict)
                or not isinstance(payload.get("source_recipes"), list)
                or any(not isinstance(recipe, dict) for recipe in payload["source_recipes"])):
            raise ValueError("Invalid comparison figure snapshot.")
        self._validate_config(payload.get("draft"))
        self._validate_view(payload.get("canvas_view"))
        snapshot = deepcopy(payload.get("snapshot"))
        applied = deepcopy(payload.get("applied"))
        if not isinstance(snapshot, dict) or not isinstance(applied, dict) or not isinstance(tables, dict):
            raise ValueError("Invalid comparison snapshot resources.")
        if any(not isinstance(key, str) or not isinstance(frame, pd.DataFrame) for key, frame in tables.items()):
            raise ValueError("Comparison resources must be named tables.")
        restored = {key: frame.copy(deep=True) for key, frame in tables.items()}
        appearance = PlotAppearance.from_mapping(payload["appearance"])
        source_appearance = validate_source_appearance(payload.get("source_appearance", {}))
        item = None
        if snapshot:
            self._validate_config(applied)
            validate_comparison(snapshot, restored)
            if [source["key"] for source in snapshot["sources"]] != applied["source_keys"]:
                raise ValueError("Saved comparison sources disagree with the applied settings.")
            if any(applied.get(key) != value for key, value in snapshot["config"].items()):
                raise ValueError("Saved comparison geometry disagrees with the applied settings.")
            item = self._make_figure_item(snapshot, restored, appearance, source_appearance)
        elif applied or restored:
            raise ValueError("Empty comparison snapshot has unexpected analytical resources.")
        previous = self._loading
        self._loading = True
        try:
            self.set_config(payload["draft"])
            self._snapshot, self._tables = snapshot, restored
            self._applied_config = applied
            self._update_method_note()
            self._source_recipes = deepcopy(payload["source_recipes"])
            self._appearance = appearance
            self._source_appearance = source_appearance
            self._source_appearance_editor.set_state(source_appearance)
            self._source_appearance_editor.set_sources(snapshot.get("sources") or self._sources.values())
            self._appearance_widget.set_state(appearance)
            self._stale = payload["stale"]
            self._fit_pending = False
            if item is not None:
                self._install_figure_item(item)
            else:
                self._scene.clear()
                self._figure_item = None
            self._restore_view(payload["canvas_view"])
            self._warnings.set_warnings(snapshot.get("warnings", []), full_text=self.details_text())
            self._update_status()
        finally:
            self._loading = previous

    def save_image(self, path=None):
        if not self.has_data():
            return None
        target = self.figure_export_target()
        kwargs = {}
        if path is None:
            chosen = choose_live_plot_export(self, target, suggestion="comparison_figure.png")
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
        try:
            notes = write_live_plot_image(out, target, **kwargs)
        except Exception as exc:
            from ..plot.figures import image_write_failure_text
            QtWidgets.QMessageBox.warning(self, "Export comparison figure", image_write_failure_text(out, exc, formats=IMAGE_FORMATS))
            return None
        if notes:
            QtWidgets.QMessageBox.information(self, "Export comparison figure", "\n\n".join(notes))
        return out

    def save_csv(self, path=None):
        if not self.has_data():
            return None
        if path is None:
            path = ask_export_path(self, title="Export comparison plotted values",
                                   suggestion="comparison_figure.csv", filters=DATA_FILTERS)
            if path is None:
                return None
        out = Path(path)
        frame = self._plot_data()
        # Rich provenance is deliberately in the bundle/Details, not repeated
        # thousands of times in each CSV row. Each row identifies its primitive.
        try:
            notes = write_plot_csv(out, frame)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export comparison values", str(exc))
            return None
        if notes:
            QtWidgets.QMessageBox.information(self, "Export comparison values", "\n\n".join(notes))
        return out
