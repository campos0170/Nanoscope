"""Explicit, snapshot-backed distribution summaries of existing event values.

This optional page never detects, fits, recalibrates, or changes event rows. Draft
controls and live sources are separate from the committed figure: only Generate
computes ECDF/quartiles/KDE. Restoring a project paints stored draw coordinates.

The typed title and axis labels belong to the measurement and summary the committed figure
shows (``gui.quantity_reset``): the controls only edit a draft, so :meth:`analyze` puts them
back to automatic when it commits another pair, not on a first draw, a repeat, a restore, or
the Generate that commits a recalled template's own pair while the words and pins are still the
template's (Undo of the recall, or a word or pin changed after it, makes that Generate an
ordinary change of pair). So does a range pinned on the value axis, when the MEASUREMENT is
what changed: the value is x in ECDF and Survival and y in Box and Violin, so the pins of both
pairs' value axes go together. A summary type moves no axis (each pair keeps its own pins, as
it always has), and the axis of a pair that is not the value never does.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, InvalidOperation
import datetime
import html
import json
import weakref
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
from .figure_text import (AppearanceTextHost, TextStyle, TextTarget, figure_text_for,
                          plain_text)
from .plot_appearance import (PlotAppearance, PlotAppearanceWidget, apply_plot_appearance,
                              reframe_axis_pins)
from .plot_export import (DATA_FILTERS, IMAGE_FORMATS, choose_live_plot_export,
                          write_live_plot_image, write_plot_csv)
from .plot_interaction import (install_plot_interaction, reset_plot_history, view_button,
                               window_readout)
from .plot_templates import WindowTemplateControls
from .plot_workflow import (ExportEntry, FormatSpec, InspectorGroup, PageAdapter, PendingState,
                            WarningSummary, bounded_controls, control_group, ensure_item_legend,
                            form_rows, install_plot_workflow)
from .quantity_reset import promise, promise_stands
from .source_appearance import SourceAppearanceEditor, with_source_appearance_edit


VIEWS = (("ecdf", "ECDF"), ("survival", "Empirical survival (uncensored)"),
         ("box", "Box (linear quartiles)"), ("violin", "Violin (Scott KDE)"))
METRICS = (("dwell_s", "Dwell (ms)"), ("depth_pA", "Median depth (source units)"),
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
#: What the status line says the plot shows after a measurement or summary change
#: (quantity_reset): "the ECDF of median depth".
_SHOWS_KIND = {"ecdf": "the ECDF", "survival": "the empirical survival", "box": "a box plot",
               "violin": "a violin plot"}
_SHOWS_METRIC = {"dwell_s": "dwell time", "depth_pA": "median depth",
                 "frac_depth": "fractional depth", "amplitude_pA": "selected amplitude"}
_PLOT_COLUMNS = ["source_key", "source_label", "primitive", "role", "point", "x", "y"]
#: A range typed in Axis properties belongs to what the axes show (figures2-2): ECDF and
#: Survival put the value on x and a fraction on y, Box and Violin the sources on x and the
#: value on y. Each pair keeps its own pins (``PlotAppearance.for_view``); a bare pin, saved
#: before pins were kept per pair, is the curves' (ECDF, the Type the page opens on).
PIN_CURVES, PIN_CATEGORIES = "curves", "categories"
#: The axis each pair draws the measurement on, for the pins a new measurement takes off
#: (quantity_reset): x for the curves, y for box and violin. The status line calls it the value
#: axis ("Value range"), since it is x in one pair and y in the other.
_VALUE_AXES = ((PIN_CURVES, "bottom", "Value"), (PIN_CATEGORIES, "left", "Value"))


def _pins_for(snapshot) -> str:
    """The pins the drawn figure (``snapshot``) frames with."""
    return PIN_CATEGORIES if (snapshot or {}).get("categorical") else PIN_CURVES


class _SummaryTextHost(AppearanceTextHost):
    """In-figure editing for Distribution summaries. A legend name IS the Source names
    editor's display label (one store): both routes edit the same appearance history
    value, with no second legend-entry wording store."""

    def __init__(self, panel, plot, widget) -> None:
        panel_ref = weakref.ref(panel)
        super().__init__(plot, widget, axis_hooks={
            "log_reason": "Distribution summaries are drawn on linear axes."},
            pin_view=lambda: _pins_for(getattr(panel_ref(), "_snapshot", None)),
            primary_view=PIN_CURVES)
        self._panel_ref = panel_ref

    def text_targets(self) -> dict:
        out = {key: target for key, target in super().text_targets().items()
               if not key.startswith("legend.entry.")}
        panel, plot = self._panel_ref(), self.plot
        legend = getattr(plot, "legend", None) if plot is not None else None
        if panel is None or legend is None:
            return out
        for _sample, label in list(legend.items):
            source = panel._legend_sources.get(plain_text(label.text))
            if source is None:
                continue
            key, name = source
            typed = panel._source_appearance.get(key, {}).get("label") or None
            out[f"legend.entry.{key}"] = TextTarget(
                label, name, TextStyle(text=typed), "legend entry", frozenset({"edit"}))
        return out

    def apply_text(self, key: str, style: TextStyle) -> None:
        panel = self._panel_ref()
        if key.startswith("legend.entry."):
            if panel is not None:
                panel._rename_source_from_figure(key[len("legend.entry."):], style.text)
            return
        super().apply_text(key, style)

    def apply_text_run(self, key: str, style: TextStyle) -> None:
        """Format typing uses the same source-name store as inline rename."""
        if key.startswith("legend.entry."):
            panel = self._panel_ref()
            if panel is not None:
                panel._rename_source_from_figure(key[len("legend.entry."):], style.text, run=True)
            return
        super().apply_text_run(key, style)


#: What the Update plot tooltip calls a draft control that differs from the figure on screen.
_PENDING_REASON = {"source_keys": "the ticked sources", "accepted_only": "accepted-only",
                   "metric": "the measurement", "kind": "the summary type",
                   "violin_points": "the violin grid"}


def _display_source_label(label, number):
    """Keep plot labels bounded; full names stay in tables, provenance and notes."""
    return label if len(label) <= 48 else f"{number + 1} · {label[:45]}…"


class DistributionSummaryWindow(QtWidgets.QMainWindow):
    workspaceChanged = QtCore.Signal()
    # The host may synchronously update sources before the explicit Generate action.
    # Its handler must not call analyze again.
    refreshRequested = QtCore.Signal()
    #: The pending state of the figure moved (a draft control, the sources, a Generate or a
    #: restore). Carries nothing; the Explorer's Update plot reads ``pending()`` of the adapter.
    pendingChanged = QtCore.Signal()

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
        self._hosted = False
        self._stale = False
        self._applied_config = {}
        self._source_recipes = []
        self._snapshot = {}
        self._tables = {}
        self._data_items = []
        self._appearance = PlotAppearance(source_appearance={})
        self._pins_drawn = PIN_CURVES      # whose pins the plot holds (``_pins_for``)
        self._source_appearance = {}
        #: What a recalled template promised (``quantity_reset.promise``): the (measurement,
        #: summary) its words and value-axis pins came with. Its first Generate commits that
        #: pair, so those words are not stale -- while they and the pins are still the
        #: template's (``promise_stands``).
        self._words_for = None

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
        note = self._data_note = QtWidgets.QLabel(
            "Stored values are used as recorded. Missing/non-finite values are disclosed; no imputation or pooling.")
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

        # ⌘Z on the page, not on these controls: an edit made in the figure is undone from
        # the keyboard whatever the Section shows (FIG-4); a page behind another Explorer
        # tab is hidden, so it does not answer.
        # Reset figure keeps the legend-name carriers: the names live in Source names
        # and colors, which Reset leaves alone, so dropping them would be a step that
        # changes nothing on screen.
        self._appearance_widget = PlotAppearanceWidget(
            self._appearance, shortcut_parent=self,
            reset_keeps=lambda key: key.startswith("legend.entry."))
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
        self._source_appearance_editor.changed.connect(lambda state: self._source_appearance_from_user(
            state, source_key=self._source_appearance_editor.source.currentData()))
        appearance_layout.addWidget(self._source_appearance_editor)
        self._source_appearance_editor.hide()  # Compatibility API; shared Format owns visible series edits.
        appearance = bounded_controls(appearance_host, maximum_height=235)
        export = control_group("Export the committed figure", "Export preserves the last generated figure, "
                               "including its notes and exact visible coordinates; unapplied controls do not change it. "
                               "Use Figure Library to save its numerical snapshot, source bindings and appearance together.")
        # Named recipes for another recording: summary, measurement and appearance, never the
        # ticked sources or the generated snapshot. Recall is a draft; Generate commits it.
        self._templates = WindowTemplateControls(
            "distribution-summaries", capture=self.template_settings,
            recall=self.apply_template_settings,
            note="A template keeps the measurement, summary type, violin grid, accepted-only "
                 "choice and appearance. It never keeps your ticked sources, per-source colours "
                 "or the generated figure. Recalling one changes the controls only: press "
                 "Generate to use them.")
        export.layout().addWidget(self._templates)
        # The four Section boxes keep their names: the Explorer's inspector takes their
        # controls out and leaves them behind, hidden (``inspector_adapter``).
        self._data_section, self._analysis_section = data, analysis
        self._appearance_section, self._export_section = appearance, export
        for widget in (data, analysis, appearance, export):
            layout.addWidget(widget)
        self._plot_widget = pg.PlotWidget(background="#111316", axisItems={
            side: ReadableLinearAxis(orientation=side) for side in ("bottom", "left")})
        self._plot_widget.setMinimumHeight(150)
        self._plot = self._plot_widget.getPlotItem()
        self._plot.setTitle("Distribution summaries · Generate to begin")
        layout.addWidget(self._plot_widget, 1)
        # One list for the right-click menu and the Explorer's Export ▾: they cannot disagree.
        # Greyed out until a figure has been generated.
        exports = self._exports = (
            ("Export current figure…", lambda: self.save_image(), self.has_data),
            ("Export plotted coordinates CSV…", lambda: self.save_csv(), self.has_data))
        self._workflow = install_plot_workflow(
            self, sections={"Data": (data,), "Analysis": (analysis,),
                            "Appearance": (appearance,), "Export": (export,)},
            actions=(("Generate", self._generate_from_user),),
            exports=(("Export current figure…", self.save_image),
                     ("Export plotted coordinates CSV…", self.save_csv)))
        self._workflow.export_button.setEnabled(False)
        self._workflow.adopt_reset_figure(self._appearance_widget, takes_ranges=True)
        # View only (plot_interaction). Reset view is this page's own framing, not a bare
        # auto-range: a box/violin keeps its full category slots. The menu's exports are
        # greyed out with the Export button, until a figure has been generated.
        self._readout = window_readout(self)
        self._legend_sources: dict[str, tuple[str, str]] = {}
        self._plot_nav = install_plot_interaction(
            self._plot, readout=self._readout, reset=self._frame_view,
            exports=exports,
            text=_SummaryTextHost(self, self._plot, self._appearance_widget))
        self._workflow.add_view_control(view_button(lambda: self._plot_nav, (self._plot_nav,)))
        self._source_list.itemChanged.connect(self._source_selection_changed)
        self._metric_combo.currentIndexChanged.connect(self._draft_changed)
        self._accepted_only.toggled.connect(self._draft_changed)
        self._view_combo.currentIndexChanged.connect(self._draft_changed)
        self._violin_points.valueChanged.connect(self._draft_changed)
        self._appearance_widget.changed.connect(self._appearance_from_user)
        self._update_method_note()

    def inspector_adapter(self) -> PageAdapter:
        """How the Analysis Explorer hosts this page (Phase 1, "one layout for every page").

        Built once and cached; nothing but ``AnalysisExplorer.add_page`` (through
        ``InspectorHost.adopt``) and the hosting test helpers call it. Building it is the
        move's first half: the Measurement, Summary and Violin rows leave the two grids for
        one form (the SAME widget instances), and the boxes stay behind as hidden husks. It
        computes nothing: ``update`` is the existing Generate handler and ``pending`` only
        compares the draft controls with the figure on screen."""
        if getattr(self, "_inspector_adapter", None) is None:
            # as wide as its words, as in the grid: a click in the middle of the row's empty
            # space would otherwise miss a check box that fills the column
            self._accepted_only.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            plot = form_rows([("Measurement", self._metric_combo), ("Summary", self._view_combo),
                              ("Violin grid points", self._violin_points)])
            self._inspector_adapter = PageAdapter.from_widget(
                self, "distribution_summaries", self._appearance_widget,
                groups=[InspectorGroup("sources", "Sources", [
                            self._source_list, self._accepted_only, self._data_note]),
                        InspectorGroup("plot", "Plot", [plot, self._method_note]),
                        InspectorGroup("recipes", "Recipes", [self._templates], expanded=False)],
                format=FormatSpec(
                    self._appearance_widget,
                    controllers=lambda: [c for c in (figure_text_for(self._plot),) if c]),
                update=self._generate_from_user,
                update_tooltip="Compute the summary of the ticked sources and draw it. "
                               "Nothing is computed before this.",
                pending=self._pending,
                pending_changed=self.pendingChanged,
                exports=lambda: [ExportEntry(text, run, enabled, kind) for (text, run, enabled), kind
                                 in zip(self._exports, ("figure", "data"))],
                navigation=lambda: self._plot_nav,
                navigations=lambda: (self._plot_nav,),
                status_bar=self.statusBar,
                chrome=[self._workflow],        # hidden when hosted; its objects stay alive
                on_adopt=self._on_adopt)
        return self._inspector_adapter

    def _on_adopt(self):
        """Hosting steps: the boxes whose controls moved out, and the Appearance and Export
        sections the inspector and the toolbar replace, stay constructed and hidden. The page
        was budgeted for a page with its controls on it (650x480)."""
        self._hosted = True
        self._templates.note.setText(self._templates.note.text().replace("Generate to use", "Update plot to use"))
        for hidden in (self._data_section, self._analysis_section, self._appearance_section,
                       self._export_section):
            hidden.hide()
        self.setMinimumSize(min(self.minimumWidth(), 280), min(self.minimumHeight(), 360))
        self._update_status()
        if not self._snapshot:
            self._plot.setTitle("Distribution summaries · Update plot to begin")

    def _update_verb(self) -> str:
        return "Update plot" if self._hosted else "Generate"

    def _pending(self) -> PendingState:
        """How many independent changes wait for Generate: the draft controls that differ
        from the figure on screen, never fewer than 1 while the figure is marked stale (a
        control moved and moved back leaves the mark), 1 before the first Generate."""
        if not self._snapshot:
            return PendingState(1, ("not generated yet",)) if self._selected_keys else PendingState()
        config, applied = self.config(), self._applied_config
        moved = [_PENDING_REASON[name] for name, value in config.items()
                 if value != applied.get(name)]
        if not moved and self._stale:
            moved = ["an earlier change to the controls or sources"]
        return PendingState(max(len(moved), int(self._stale)), tuple(moved))

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

    def template_settings(self):
        """``(settings, appearance)`` for a window template: the draft recipe without sources."""
        settings = self.config()
        settings.pop("source_keys", None)
        return settings, self.appearance_state()

    def apply_template_settings(self, settings, appearance):
        """A recalled template: the draft controls and one appearance step, never Generate.

        Validated whole before anything moves, so a damaged template changes nothing rather
        than half the controls. The committed figure keeps its numbers; only its appearance
        follows, which is an appearance-only edit. Like a hand edit, it marks the figure
        unapplied only when a draft control actually moved.
        """
        before, _appearance = self.template_settings()
        draft = {key: settings[key] for key in ("accepted_only", "metric", "kind", "violin_points")
                 if key in settings}
        if ("metric" in draft and draft["metric"] not in dict(METRICS)
                or "kind" in draft and draft["kind"] not in dict(VIEWS)
                or "accepted_only" in draft and type(draft["accepted_only"]) is not bool
                or "violin_points" in draft and (type(draft["violin_points"]) is not int
                                                 or not 32 <= draft["violin_points"] <= 2048)):
            raise ValueError("Its distribution settings are not ones this page offers.")
        self.set_config({**self.config(), **draft})
        if self.template_settings()[0] != before:
            self._draft_changed()
        self._appearance_widget.apply_appearance(appearance)
        config = self.config()
        self._words_for = promise((config["metric"], config["kind"]),
                                  self._appearance_widget.state(),
                                  axes=_VALUE_AXES, primary=PIN_CURVES)

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
            text = f"Select event sources, then {self._update_verb()}. No analysis runs on opening this page."
        else:
            counts = ", ".join(f"{series['label']}: n={series['n']}"
                               for series in self._snapshot["series"])
            text = ("Unapplied data/analysis changes: " + "; ".join(self._pending().reasons)
                    + f". Use {self._update_verb()}; showing the last generated result. "
                    if self._stale else "Generated result · ") + counts
        self._status.setToolTip(text)
        self._status.setAccessibleDescription(text)
        self._status.setText(text if len(text) <= 220 else text[:217] + "…")
        self._workflow.primary_actions[0].setText("Generate ●" if self._stale else "Generate")
        self._workflow.export_button.setEnabled(self.has_data())
        self.pendingChanged.emit()

    def _generate_from_user(self):
        self.refreshRequested.emit()
        try:
            return self.analyze()
        except (TypeError, ValueError, OverflowError) as exc:
            QtWidgets.QMessageBox.warning(self, self._update_verb() + " — distribution summary", str(exc))
            return False

    def refresh(self):
        """Explicit programmatic generation; the host should offer current sources first."""
        return self.analyze()

    def analyze(self):
        """Compute a new figure transactionally; on failure retain the old snapshot."""
        config = self.config()
        if not config["source_keys"]:
            raise ValueError("Select at least one event source.")
        if any(key not in self._sources for key in config["source_keys"]):
            raise ValueError(f"A selected source is unavailable. Rebind it or uncheck it before {self._update_verb()}.")
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
        # The figure on screen shows the OLD measurement and summary until this commit: the
        # words typed on it go back to automatic when the new ones replace it.
        previous, words_for, self._words_for = self._applied_config, self._words_for, None
        moved = (bool(previous) and (previous["metric"], previous["kind"]) != (metric, kind)
                 and not promise_stands(words_for, (metric, kind), self._appearance_widget.state(),
                                        axes=_VALUE_AXES, primary=PIN_CURVES))
        self._applied_config = deepcopy(config)
        self._source_recipes = recipes
        self._snapshot, self._tables = snapshot, tables
        self._stale = False
        self._render_snapshot()
        self._update_status()
        if moved:
            self._appearance_widget.reset_typed_words(
                f"{_SHOWS_KIND[kind]} of {_SHOWS_METRIC[metric]}", self.statusBar(),
                axes=_VALUE_AXES if previous["metric"] != metric else (), primary=PIN_CURVES)
        self.workspaceChanged.emit()
        return True

    @staticmethod
    def _metric_label(metric, sources, frames):
        if metric == "dwell_s":
            return "Dwell (ms)"
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
        reset_plot_history(self._plot)   # new geometry: old zoom ranges mean nothing now
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
            from .figure_series import tag_pg_series
            tag_pg_series(item, view="summary", role="data", identity=primitive["source_key"],
                          identity_kind="source", label=primitive["legend"], color=primitive["color"])
            self._data_items.append(item)
            if primitive["source_key"] not in seen:
                entries.append((item, primitive["legend"]))
                seen.add(primitive["source_key"])
        # Older editable snapshots stored a generated warning prefix in this
        # internal title. Suppress only that prefix for rendering; retain the
        # snapshot and its warnings, and never filter user appearance titles.
        title = self._snapshot.get("title", f"Distribution summaries · {self._update_verb()} to begin")
        self._plot.setTitle(title.removeprefix("⚠ Distribution notes · "))
        x_label, y_label = self._snapshot.get("x_label", ""), self._snapshot.get("y_label", "")
        if self._applied_config.get("metric") == "dwell_s":
            if self._snapshot.get("categorical"):
                y_label = "Dwell (ms)"
            else:
                x_label = "Dwell (ms)"
        self._plot.setLabel("bottom", x_label)
        self._plot.setLabel("left", y_label)
        # The Appearance boxes name the automatic text of the figure on screen.
        for field, text in (("title", title.removeprefix("⚠ Distribution notes · ")),
                            ("x_label", x_label),
                            ("y_label", y_label)):
            self._appearance_widget.set_automatic_text(field, text)
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
        if self._pins_drawn != _pins_for(self._snapshot):
            # A Type whose axes show other quantities (ECDF/Survival <-> Box/Violin): the
            # other pair's pins leave the plot BEFORE it is framed, or taking them off after
            # would hand those axes back to auto-range and lose the category slots.
            self._apply_appearance()
        # Framed first, then styled: a range pinned in Axis properties is the figure's
        # framing, so a Generate (new data) ends on the pin rather than on the data.
        self._frame_view()
        reframe_axis_pins(self._plot)
        self._apply_appearance()

    def _frame_view(self):
        """The page's own framing of the drawn figure: what a draw ends with and what
        Reset view returns to. View only; the stored coordinates are not read again."""
        self._plot.enableAutoRange()
        series = self._snapshot.get("series", [])
        if self._snapshot.get("categorical") and series:
            # Category indices are not physical measurements. Reserve a complete
            # source slot even for a single box/violin, without rescaling its data.
            self._plot.setXRange(-.6, len(series) - .4, padding=0)

    def details_text(self):
        if not self._snapshot:
            return ""
        blocks = [self._applied_config.get("method", "")]
        for series in self._snapshot["series"]:
            text = (f"{series['label']}: {series['n']} finite of {series['n_total']} selected observations; "
                    f"{series['n_excluded']} non-finite excluded.")
            if series["stats"]:
                text += "\n" + "; ".join(f"{key}={value * (1000. if self._applied_config.get('metric') == 'dwell_s' and key in ('q1', 'median', 'q3', 'lower_whisker', 'upper_whisker', 'iqr', 'lower_fence', 'upper_fence', 'bandwidth') else 1.):.10g}" +
                    (" ms" if self._applied_config.get('metric') == 'dwell_s' and key in ('q1', 'median', 'q3', 'lower_whisker', 'upper_whisker', 'iqr', 'lower_fence', 'upper_fence', 'bandwidth') else "")
                    for key, value in series["stats"].items())
            blocks.append(text)
        blocks.extend(self._snapshot["warnings"])
        if self._source_appearance:
            blocks.append("Figure source appearance (original identities and numerical tables unchanged):\n" +
                          "\n".join(f"{key}: {style}" for key, style in self._source_appearance.items()))
        return "\n\n".join(blocks)

    def warning_text(self):
        return "\n\n".join(self._snapshot.get("warnings", []))

    def _appearance_from_user(self, state):
        overrides = validate_source_appearance(self._source_appearance if state.source_appearance is None
                                               else state.source_appearance)
        self._appearance = replace(state, source_appearance=overrides)
        self._source_appearance = overrides
        self._source_appearance_editor.set_state(overrides)
        if state != self._appearance:
            self._appearance_widget.set_state(self._appearance)
        self._apply_appearance()
        self.workspaceChanged.emit()

    def _apply_appearance(self):
        pins = _pins_for(self._snapshot)
        if self._applied_config:
            from ..core.quantity_display import quantity_for
            from .quantity_axis import present_page_axes
            metric = self._applied_config.get("metric", "")
            side = "left" if self._snapshot.get("categorical") else "bottom"
            quantities = {side: quantity_for(metric)}
            other = "bottom" if side == "left" else "left"
            quantities[other] = quantity_for("source" if other == "bottom" else "probability",
                declared_unit="-", quantity="categorical" if other == "bottom" else "dimensionless")
            present_page_axes(self, self._plot, quantities, view=pins, primary=PIN_CURVES)
        apply_plot_appearance(self._plot, replace(self._appearance.for_view(pins, primary=PIN_CURVES), series_styles=None),
                              plot_widget=self._plot_widget, data_items=self._data_items)
        self._pins_drawn = pins
        self._apply_source_appearance()
        from .figure_series import apply_pg_series_styles
        apply_pg_series_styles(self._plot,self._appearance,data_items=self._data_items)
        from .figure_series import sync_series_legend
        sync_series_legend(self._plot)

    def _sync_source_appearance_editor(self):
        if self._snapshot:
            colors = {p["source_key"]: p["color"] for p in self._snapshot.get("primitives", [])}
            sources = [{**series, "color": colors.get(series["key"], "#0072b2")}
                       for series in self._snapshot.get("series", [])]
        else:
            sources = list(self._sources.values())
        self._source_appearance_editor.set_sources(sources)
        self._source_appearance_editor.set_state(self._source_appearance)

    def _source_appearance_from_user(self, state, *, run=False, source_key=None):
        updated = with_source_appearance_edit(self._appearance_widget.state(), state,
            view='summary', source_key=source_key)
        if run:
            self._appearance_widget._edit("source_appearance", updated.source_appearance)
        else:
            self._appearance_widget.apply_appearance(updated)

    def source_appearance_state(self):
        return deepcopy(self._source_appearance)

    def set_source_appearance_state(self, state):
        """Restore the compatibility source map into the one appearance/history value."""
        self._source_appearance = validate_source_appearance(state)
        self._appearance = replace(self._appearance, source_appearance=self._source_appearance)
        self._appearance_widget.set_state(self._appearance)
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
        # A shared style's data color is the default for an unclassified series.
        # These primitives already carry source identity colors. Preserve those,
        # while retaining deliberate figure overrides and legacy unbound looks.
        global_color = self._appearance.data_color
        binding = self._appearance.style_binding
        if binding and "data_color" not in binding.get("overrides", {}):
            global_color = None
        self._plot.disableAutoRange()
        try:
            for item, primitive in zip(self._data_items, self._snapshot.get("primitives", [])):
                key = primitive["source_key"]
                descriptor = getattr(item, '_nanoscope_series_descriptor', None)
                if descriptor is not None:
                    item._nanoscope_series_descriptor = replace(descriptor, label=names[key])
                color = (self._source_appearance.get(key, {}).get("color") or
                         global_color or primitive["color"])
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
                    # A name is literal text (owner, Q-T1): the legend row is rich text,
                    # so "Apo <1>" would otherwise lose its "<1>" as a tag.
                    entries.append((item, html.escape(f"{names[key]} · n={counts[key]}",
                                                      quote=False)))
                    self._legend_sources[f"{names[key]} · n={counts[key]}"] = (key, names[key])
                    seen.add(key)
            ensure_item_legend(self._plot, entries)
            controller = figure_text_for(self._plot)
            if controller is not None:
                controller.refresh()        # the legend may be new since the last draw
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

    def _rename_source_from_figure(self, source_key: str, text: str | None, *,
                                   run: bool = False) -> None:
        """Canvas and source-table renames use the same source map and Undo step."""
        new = text.strip() if text and text.strip() else None
        self._source_appearance_from_user(self._with_label(source_key, new), run=run)

    def _with_label(self, source_key: str, label: str | None, names=None) -> dict:
        """The Source names store (or ``names``) with ``source_key``'s display label set
        (None: the captured one)."""
        state = deepcopy(self._source_appearance if names is None else names)
        entry = dict(state.get(source_key, {}))
        if label:
            entry["label"] = label
        else:
            entry.pop("label", None)
        if entry:
            state[source_key] = entry
        else:
            state.pop(source_key, None)
        return state

    def appearance_state(self):
        return self._appearance.to_dict()

    def set_appearance_state(self, state):
        appearance = state if isinstance(state, PlotAppearance) else PlotAppearance.from_mapping(state)
        source = self._source_appearance if appearance.source_appearance is None else appearance.source_appearance
        self._source_appearance = validate_source_appearance(source)
        self._appearance = replace(appearance, source_appearance=self._source_appearance)
        self._source_appearance_editor.set_state(self._source_appearance)
        self._appearance_widget.set_state(self._appearance)
        self._appearance_widget.clear_history()
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
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_PLOT_COLUMNS)
        from .quantity_axis import export_display_columns
        return export_display_columns(frame, {name: basis.quantity for name, side in
            (("x", "bottom"), ("y", "left")) if
            (basis := getattr(self._plot.getAxis(side), "_nanoscope_quantity_basis", None)) is not None})

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
        appearance = replace(appearance, source_appearance=(source_appearance
            if appearance.source_appearance is None else appearance.source_appearance))
        source_appearance = validate_source_appearance(appearance.source_appearance)
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
        self._appearance_widget.clear_history()
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
