"""Explicit, snapshot-backed publication comparisons of existing event tables.

Generating prepares plot geometry, never detection/calibration/fitted results.
Styling and restore draw that saved geometry; they do not rerun statistics or
swarm layout. Canvas magnification is a viewer aid, not a numerical axis edit.

The typed title and axis labels belong to the figure type and measurements the committed
figure shows (``gui.quantity_reset``): the controls only edit a draft, so :meth:`analyze` puts
them back to automatic when it commits another quantity, not on a first draw, a repeat, a
restore, or the Generate that commits a recalled template's own quantity while the words are
still the template's (Undo of the recall, or a word typed after it, makes that Generate an
ordinary change of quantity).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.analysis_workspace import _plot_source_recipe, fingerprint_frame
from ..core.sources import amplitude_units_label, signal_domain_of, source_comparability
from ..core.source_style import validate_source_appearance
from .analysis_catalog import ANALYSIS_LABELS
from .column_checklist import ColumnChecklist
from .export_paths import ask_export_path
from .plot_appearance import PlotAppearance, PlotAppearanceWidget
from .plot_export import DATA_FILTERS, IMAGE_FORMATS, choose_live_plot_export, write_live_plot_image, write_plot_csv
from .plot_templates import WindowTemplateControls
from .plot_workflow import (ExportEntry, FormatSpec, InspectorGroup, PageAdapter, PendingState,
                            WarningSummary, bounded_controls, control_group, form_rows,
                            install_plot_workflow)
from .quantity_reset import promise, cleared_words, cleared_pins, reset_sentence
from .source_appearance import SourceAppearanceEditor, with_source_appearance_edit
from .widget_transfer import prepare_widget_move


VIEWS = (("facets", "Faceted comparisons"), ("pairs", "Pair-plot matrix"),
         ("events", "Individual-event overlays"))
METRICS = (("dwell_s", "Dwell (ms)"), ("depth_pA", "Median depth (recorded units)"),
           ("amplitude_pA", "Selected amplitude (recorded units)"),
           ("frac_depth", "Fractional depth ΔI/I₀"),
           ("ecd_pA_s", "Event area (recorded units · s)"), ("snr", "SNR"),
           ("rise_s", "Rise time (ms)"), ("fall_s", "Fall time (ms)"))
_METRIC_KEYS = {key for key, _ in METRICS}
#: What the status line calls a measurement after a figure change (quantity_reset).
_SHOWS_METRIC = {"dwell_s": "dwell time", "depth_pA": "median depth",
                 "amplitude_pA": "selected amplitude", "frac_depth": "fractional depth",
                 "ecd_pA_s": "event area", "snr": "SNR", "rise_s": "rise time",
                 "fall_s": "fall time"}
#: The pair matrix's starting features, and what its Default button puts back.
PAIR_DEFAULT_FEATURES = ("dwell_s", "depth_pA", "ecd_pA_s", "snr")
_KINDS = {key for key, _ in VIEWS}
#: What the Update plot tooltip calls a draft control that differs from the figure on screen.
_PENDING_REASON = {"source_keys": "the ticked sources", "accepted_only": "accepted-only",
                   "kind": "the figure type", "x": "the x measurement", "y": "the y measurement",
                   "metric": "the measurement", "features": "the pair features",
                   "bins": "the histogram bins", "columns": "the facet columns",
                   "points": "the point layout", "show_box": "the box overlay",
                   "log_x": "log x", "log_y": "log y"}
METHODS = {
    "facets": "One scatter panel per recording, with shared numerical axes. No pooling, regression, smoothing or uncertainty estimation.",
    "pairs": "Two to five features: scatter panels off the diagonal and separate-source count histograms on the diagonal. Bins are shared per feature. Missing values are excluded per panel, not by deleting every row with any missing feature.",
    "events": "Every retained observation is drawn as a strip or swarm point. Optional boxes use NanoScope's linear quartiles and 1.5-IQR sample whiskers, not confidence intervals. Horizontal spreading is frozen on Generate and is not a measured variable. Larger markers may overlap without moving any event.",
}


class _WrapCheckBox(QtWidgets.QCheckBox):
    """A check box whose sentence wraps when its column is narrower than the sentence.

    A plain ``QCheckBox`` never wraps, so "Overlay box summary (linear quartiles; 1.5 IQR)"
    (306 px) kept the Explorer's 290 px inspector column from fitting. Same text, same
    signals; on a wide column it is drawn on one line as before."""

    _FLAGS = (QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap
              | QtCore.Qt.TextShowMnemonic)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ``Minimum`` (a check box's default) makes the one-line width the floor.
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)

    def _gap(self):
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""
        return self.style().subElementRect(QtWidgets.QStyle.SE_CheckBoxContents, opt, self).left()

    def minimumSizeHint(self):  # noqa: N802
        base = super().minimumSizeHint()
        word = max((self.fontMetrics().horizontalAdvance(w) for w in self.text().split()), default=0)
        return QtCore.QSize(min(base.width(), self._gap() + word), base.height())

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        room = max(width - self._gap(), 1)
        text = self.fontMetrics().boundingRect(QtCore.QRect(0, 0, room, 10_000), self._FLAGS, self.text())
        return max(self.sizeHint().height(), text.height() + 4)

    def paintEvent(self, event):  # noqa: N802
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        text, opt.text = opt.text, ""
        painter.drawControl(QtWidgets.QStyle.CE_CheckBox, opt)      # the box alone
        painter.drawItemText(self.style().subElementRect(QtWidgets.QStyle.SE_CheckBoxContents, opt, self),
                             self._FLAGS, self.palette(), self.isEnabled(), text,
                             QtGui.QPalette.WindowText)


class _ComparisonCanvas(QtWidgets.QGraphicsView):
    """A bounded magnifier for a fixed publication canvas, not an analysis editor."""

    viewChanged = QtCore.Signal()
    geometryChanged = QtCore.Signal()

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        #: True while the canvas shows the whole figure and nobody has zoomed or panned
        #: it. Then a resize fits it again, so a figure generated with the canvas tall
        #: (Section ▸ Data) is still whole when Analysis's controls shrink the canvas;
        #: before, it kept the old scale and opened scrolled into the figure, its title
        #: cut off. The scene rect is the figure's (see ``_install_figure_item``), so a
        #: restyled figure of another size is fitted as it now is.
        self._fitted = False
        self._pan_from = None
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setMinimumHeight(180)
        self.setAccessibleName("Comparison figure canvas")
        self.setBackgroundBrush(QtGui.QColor("#dce3e7"))

    def fit(self, rect):
        self._fitted = True
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def release_fit(self):
        """The view is the user's now (a zoom, a pan, a restored view): stop re-fitting."""
        self._fitted = False

    def zoom(self, factor):
        scale = self.transform().m11() * factor
        if .01 <= scale <= 100:
            self.release_fit()
            self.scale(factor, factor)
            self.viewChanged.emit()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._fitted and not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.geometryChanged.emit()

    def _scroll_position(self):
        return (self.horizontalScrollBar().value(), self.verticalScrollBar().value())

    def mousePressEvent(self, event):  # noqa: N802
        self._pan_from = self._scroll_position()
        super().mousePressEvent(event)

    def wheelEvent(self, event):  # noqa: N802
        if event.angleDelta().y():
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        super().mouseReleaseEvent(event)
        if self._pan_from is not None and self._pan_from != self._scroll_position():
            self.release_fit()      # a hand-drag pan; a plain click keeps the fit
        self._pan_from = None
        self.viewChanged.emit()


class ComparisonFiguresWindow(QtWidgets.QMainWindow):
    workspaceChanged = QtCore.Signal()
    formatViewsChanged = QtCore.Signal()
    refreshRequested = QtCore.Signal()
    #: The pending state of the figure moved (a draft control, the sources, a Generate or a
    #: restore). Carries nothing; the Explorer's Update plot reads ``pending()`` of the adapter.
    pendingChanged = QtCore.Signal()

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
        self._hosted = False
        self._stale = False
        self._snapshot = {}
        self._tables = {}
        self._applied_config = {}
        self._source_recipes = []
        self._appearance = PlotAppearance(source_appearance={})
        self._text_controllers = []
        self._source_appearance = {}
        #: What a recalled template promised (``quantity_reset.promise``): the quantity its
        #: words came with. Its first Generate commits that quantity, so those words are not
        #: stale -- while they are still the template's (``promise_stands``).
        self._words_for = None
        self._panel_reset_keys = ()
        #: The sentence the last Generate led the status line with ("Title reset: ..."), until
        #: what it names is typed or pinned again (the Undo of the reset, a title typed since).
        self._reset_said = ""
        self._figure_item = None
        self._fit_pending = False
        #: Set while the Explorer's Style ▾ restyles this page: a figure that will not render
        #: is then reported to the caller (``_quiet_error``) instead of opened as a dialog.
        self._quiet = False
        self._quiet_error = ""

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)
        self._status = QtWidgets.QLabel()
        self._status.setTextFormat(QtCore.Qt.PlainText)
        self._status.setWordWrap(True)
        # Let the bounded status text wrap fully when the hosted canvas is narrow.
        # A fixed 50 px cap cut the last source/count line in the pending state.
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
        note = self._data_note = QtWidgets.QLabel(
            "Uses stored values and identities. No automatic recalibration, imputation or subsampling.")
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
        # The shared checklist: a click anywhere on a row ticks it (only the 18 px box did),
        # and the two-to-five rule is live ("4 of 8 selected · pick 2–5", the rows that
        # would cross it greyed and saying why) instead of being met at Generate. The draft
        # key `features` keeps METRICS order, as before.
        self._features = ColumnChecklist(noun="features", minimum=2, maximum=5)
        self._features.set_entries([(key, label, key) for key, label in METRICS],
                                   PAIR_DEFAULT_FEATURES, PAIR_DEFAULT_FEATURES)
        self._features.set_visible_rows(6)
        self._features.layout().setContentsMargins(0, 0, 0, 0)
        self._features.view.setAccessibleName("Pair matrix features")
        pairs.addWidget(self._features, 1)
        pair_options = QtWidgets.QVBoxLayout()
        pair_options.addWidget(QtWidgets.QLabel("Diagonal histogram bins"))
        self._bins = QtWidgets.QSpinBox()
        self._bins.setRange(5, 200)
        self._bins.setValue(40)
        self._bins.setAccessibleName("Pair histogram bins")
        pair_options.addWidget(self._bins)
        pair_note = self._pair_note = QtWidgets.QLabel(
            "Linear axes; separate-source counts. No KDE or fitted lines.")
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
        self._show_box = _WrapCheckBox("Overlay box summary (linear quartiles; 1.5 IQR)")
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

        # ⌘Z on the page, not on these controls: Reset figure (in the page toolbar) and the rule's
        # reset at Generate are made from any Section, and their step is undone from the keyboard
        # whatever the Section shows (FIG-4); a page behind another Explorer tab is hidden, so it
        # does not answer.
        self._appearance_widget = PlotAppearanceWidget(self._appearance, multi_view_labels=True,
                                                       shortcut_parent=self)
        self._appearance_widget.style_vocabulary = "matplotlib_appearance"
        self._appearance_widget.data_color.setEnabled(False)
        self._appearance_widget.data_color.setToolTip("Comparison figures retain each source's identity colour; a single colour override is not applied.")
        self._appearance_widget.fit_color.setEnabled(False)
        self._appearance_widget.fit_color.setToolTip("Comparison figures do not fit models.")
        appearance_content = QtWidgets.QWidget()
        appearance_layout = QtWidgets.QVBoxLayout(appearance_content)
        appearance_layout.setContentsMargins(0, 0, 0, 0)
        self._source_appearance_editor = SourceAppearanceEditor(session=session)
        appearance_layout.addWidget(self._source_appearance_editor)
        self._source_appearance_editor.hide()  # Compatibility API; shared Format owns visible series edits.
        appearance_layout.addWidget(self._appearance_widget)
        appearance = bounded_controls(appearance_content, maximum_height=235)
        export = control_group("Export the generated figure", "Export the full publication canvas as PNG, PDF or SVG. "
            "Canvas zoom is only a magnifier. Save to Figure Library keeps observations, plotted geometry, "
            "settings and full Details together. Draft controls never change a saved picture.")
        # Named recipes for another recording: figure type, measurements, layout and
        # appearance, never the ticked sources or the generated figure. Recall is a draft.
        self._templates = WindowTemplateControls(
            "comparison-figures", capture=self.template_settings,
            recall=self.apply_template_settings,
            note="A template keeps the figure type, measurements, pair features, bins, layout, "
                 "log axes, accepted-only choice and appearance. It never keeps your ticked "
                 "sources, per-source colours or the generated figure. Recalling one changes "
                 "the controls only: press Generate to use them.")
        export.layout().addWidget(self._templates)
        # The four Section boxes keep their names: the Explorer's inspector takes their
        # controls out and leaves them behind, hidden (``inspector_adapter``).
        self._data_section, self._analysis_section = data, analysis_scroll
        self._appearance_section, self._export_section = appearance, export
        for widget in (data, analysis_scroll, appearance, export):
            layout.addWidget(widget)

        # Keep the explanatory sentence above the controls. In a hosted page the
        # canvas can be only 300 px wide while Figures and the inspector are open;
        # a single row otherwise forces the whole figure beyond that viewport.
        canvas_bar = QtWidgets.QVBoxLayout()
        canvas_bar.setSpacing(3)
        self._canvas_note = QtWidgets.QLabel("Canvas magnification · exports retain the full figure")
        self._canvas_note.setWordWrap(True)
        canvas_bar.addWidget(self._canvas_note)
        canvas_buttons = QtWidgets.QHBoxLayout()
        self._fit_button = QtWidgets.QPushButton("Fit canvas")
        self._zoom_out = QtWidgets.QPushButton("−")
        self._zoom_in = QtWidgets.QPushButton("+")
        self._zoom_out.setAccessibleName("Zoom out comparison canvas")
        self._zoom_in.setAccessibleName("Zoom in comparison canvas")
        for button in (self._fit_button, self._zoom_out, self._zoom_in):
            canvas_buttons.addWidget(button)
        canvas_buttons.addStretch(1)
        canvas_bar.addLayout(canvas_buttons)
        layout.addLayout(canvas_bar)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._view = _ComparisonCanvas(self._scene, self)
        from .matplotlib_series import MatplotlibSeriesSelection
        self._series_selection = MatplotlibSeriesSelection(self._view, self)
        layout.addWidget(self._view, 1)
        self._warnings = WarningSummary(title="Comparison notes", maximum_height=135)
        self._warnings.toggle.setText("Details")
        self._warnings.toggle.setAccessibleName("Comparison methods and provenance")
        self._warnings.toggle.setToolTip("Show all comparison notes, methods and provenance.")
        layout.addWidget(self._warnings)
        # The page's export list, defined once; Explorer Export ▾ reads it too (``inspector_adapter``).
        self._exports = (("Export current figure…", self.save_image),
                         ("Export plotted values CSV…", self.save_csv))
        self._workflow = install_plot_workflow(self,
            sections={"Data": (data,), "Analysis": (analysis_scroll,), "Appearance": (appearance,), "Export": (export,)},
            actions=(("Generate", self._generate_from_user),),
            exports=self._exports)
        self._workflow.adopt_reset_figure(self._appearance_widget)
        self._source_list.itemChanged.connect(self._source_selection_changed)
        self._accepted_only.toggled.connect(self._draft_changed)
        for combo in (self._view_combo, self._x_combo, self._y_combo, self._metric, self._points):
            combo.currentIndexChanged.connect(self._draft_changed)
        for check in (self._log_x, self._log_y, self._event_log_y, self._show_box):
            check.toggled.connect(self._draft_changed)
        self._bins.valueChanged.connect(self._draft_changed)
        self._columns.valueChanged.connect(self._draft_changed)
        self._features.changed.connect(self._draft_changed)
        self._appearance_widget.changed.connect(self._appearance_from_user)
        self._appearance_widget.resetLapsed.connect(self._reset_sentence_lapsed)
        self._source_appearance_editor.changed.connect(lambda state: self._source_appearance_from_user(
            state, source_key=self._source_appearance_editor.source.currentData()))
        self._fit_button.clicked.connect(self.fit_canvas)
        self._zoom_out.clicked.connect(lambda: self._view.zoom(1 / 1.25))
        self._zoom_in.clicked.connect(lambda: self._view.zoom(1.25))
        self._view.viewChanged.connect(self._view_changed)
        self._update_method_note()
        self._update_status()

    def inspector_adapter(self) -> PageAdapter:
        """How the Analysis Explorer hosts this page (Phase 1, "one layout for every page").

        Built once and cached; nothing but ``AnalysisExplorer.add_page`` (through
        ``InspectorHost.adopt``) and the hosting test helpers call it. Building it is the
        move's first half: the figure-type row and the three per-type boxes are re-flowed
        into forms (the SAME widget instances; ``_update_method_note`` still shows one box
        at a time), and the Data, Analysis, Appearance and Export boxes stay behind, hidden.
        It computes nothing: ``update`` is the existing Generate handler and ``pending``
        only compares the draft controls with the figure on screen. There is no pointer
        readout and no Reset view: the canvas is a magnifier (Fit canvas, − and + stay on
        its bar), and there is nothing on it to click until the figure is editable."""
        if getattr(self, "_inspector_adapter", None) is None:
            for box in (self._accepted_only, self._log_x, self._log_y, self._event_log_y):
                # as wide as its words, as in the grid: a click in the middle of the row's
                # empty space would otherwise miss a check box that fills the column
                box.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            kind = form_rows([("Figure", self._view_combo)])
            self._reflow(self._facet_controls, [("X", self._x_combo), ("Y", self._y_combo),
                                                ("Columns", self._columns)],
                         [self._log_x, self._log_y])
            pair = form_rows([("Diagonal histogram bins", self._bins)])
            prepare_widget_move(self._features)
            pair.layout().insertRow(0, self._features)
            prepare_widget_move(self._pair_note)
            pair.layout().addRow(self._pair_note)
            self._place(self._pair_controls, pair)
            self._reflow(self._event_controls, [("Measurement", self._metric),
                                                ("Points", self._points)],
                         [self._show_box, self._event_log_y], label_above=True)
            self._inspector_adapter = PageAdapter.from_widget(
                self, "comparison_figures", self._appearance_widget,
                groups=[InspectorGroup("sources", "Sources", [
                            self._source_list, self._accepted_only, self._data_note]),
                        InspectorGroup("plot", "Plot", [
                            kind, self._method_note, self._facet_controls,
                            self._pair_controls, self._event_controls]),
                        InspectorGroup("recipes", "Recipes", [self._templates], expanded=False)],
                format=FormatSpec(
                    self._appearance_widget,
                    controllers=lambda: self._text_controllers,
                    views_changed=self.formatViewsChanged,
                    series_provider=self.series_descriptors,
                    series_selected=self._series_selection.selected,
                    series_select=self._series_selection.select),
                update=self._generate_from_user,
                update_tooltip="Prepare the figure from the ticked sources and draw it. "
                               "Nothing is computed before this.",
                pending=self._pending,
                pending_changed=self.pendingChanged,
                apply_look=self._apply_look,
                exports=lambda: [ExportEntry(text, run, self.has_data, kind) for (text, run), kind
                                 in zip(self._exports, ("figure", "data"))],
                status_bar=self.statusBar,
                chrome=[self._workflow],        # hidden when hosted; its objects stay alive
                on_adopt=self._on_adopt)
        return self._inspector_adapter

    def series_descriptors(self):
        from ..plot.comparison_figures import comparison_series
        return comparison_series(self._snapshot, self._source_appearance, self._appearance)

    @staticmethod
    def _place(box, content):
        """``content`` is the whole of ``box`` now (its old grid is left empty, hidden labels
        and all). The box's title is a heading inside the content that wraps: a group box
        title never does, and "Individual observations · optional descriptive box overlay"
        alone is wider than the inspector column."""
        heading = QtWidgets.QLabel(box.title())
        heading.setWordWrap(True)
        heading.setTextFormat(QtCore.Qt.PlainText)
        content.layout().insertRow(0, heading)
        box.setAccessibleName(box.title())
        box.setTitle("")
        for label in box.findChildren(QtWidgets.QLabel,
                                      options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly):
            label.hide()
        layout = box.layout()
        if isinstance(layout, QtWidgets.QGridLayout):
            layout.addWidget(content, layout.rowCount(), 0, 1, max(layout.columnCount(), 1))
        else:
            layout.addWidget(content)

    def _reflow(self, box, rows, wide, *, label_above=False):
        """A per-type box as one form: labelled ``rows``, then the ``wide`` check boxes.
        ``label_above`` puts each label over its control: a 253 px combo and its label would
        otherwise set the form's minimum width."""
        content = form_rows(rows)
        if label_above:
            content.layout().setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        for widget in wide:
            prepare_widget_move(widget)
            content.layout().addRow(widget)
        self._place(box, content)

    def _on_adopt(self) -> None:
        """Hosting steps: the boxes whose controls moved out, and the Appearance and Export
        sections the inspector and the toolbar replace, stay constructed and hidden. The page
        was budgeted for a page with its controls on it (700x500)."""
        self._hosted = True
        self._templates.note.setText(self._templates.note.text().replace("press Generate", "press Update plot"))
        for hidden in (self._data_section, self._analysis_section, self._appearance_section,
                       self._export_section):
            hidden.hide()
        self._canvas_note.setWordWrap(True)         # the canvas bar's sentence no longer sets the page's width
        self._update_method_note()      # the inspector showed the three boxes; one belongs on screen
        self.setMinimumSize(min(self.minimumWidth(), 280), min(self.minimumHeight(), 360))
        self._update_status()

    def _update_verb(self) -> str:
        return "Update plot" if self._hosted else "Generate"

    def _apply_look(self, look, keep_hand) -> bool:
        """Style ▾ over several pages: ONE appearance step, and a figure that will not render
        comes back as the reason (``StyleResult.failed``) instead of a modal dialog."""
        self._quiet, self._quiet_error = True, ""
        try:
            moved = self._appearance_widget.apply_look(look, keep_hand=keep_hand)
            error = self._quiet_error
        finally:
            self._quiet, self._quiet_error = False, ""
        if error:
            raise RuntimeError(error)
        return moved

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

    @staticmethod
    def _metric_combo(name, default):
        combo = QtWidgets.QComboBox()
        combo.setAccessibleName(name)
        combo.setMinimumContentsLength(14)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        for key, label in METRICS:
            combo.addItem(label, key)
            combo.setItemData(combo.count() - 1, key, QtCore.Qt.ToolTipRole)
        combo.setCurrentIndex(combo.findData(default))
        return combo

    def config(self):
        kind = self._view_combo.currentData()
        return {"source_keys": list(self._selected_keys), "accepted_only": self._accepted_only.isChecked(),
                "kind": kind, "x": self._x_combo.currentData(), "y": self._y_combo.currentData(),
                "metric": self._metric.currentData(), "features": self._features.checked(),
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
            self._features.set_checked(merged["features"])
            self._populate_sources()
            self._update_method_note()
        finally:
            self._loading = previous

    def template_settings(self):
        """``(settings, appearance)`` for a window template: the draft recipe without sources."""
        settings = self.config()
        settings.pop("source_keys", None)
        return settings, self.appearance_state()

    def apply_template_settings(self, settings, appearance):
        """A recalled template: the draft controls and one appearance step, never Generate.

        ``set_config`` validates the merged recipe before any control moves, so a damaged
        template is refused whole. Sources stay as ticked. Like a hand edit, it marks the
        figure unapplied only when a draft control actually moved.
        """
        before, _appearance = self.template_settings()
        draft = {key: value for key, value in settings.items() if key != "source_keys"}
        self.set_config(draft)
        if self.template_settings()[0] != before:
            self._draft_changed()
        self._appearance_widget.apply_appearance(appearance)
        self._words_for = self._word_promise(self.config(), self._appearance_widget.state())

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
        self._method_note.setText(METHODS[kind].replace("Generate", self._update_verb()) + limits)
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

    def _update_status(self, lead=""):
        if self._snapshot:
            counts = "; ".join(f"{source['label']}: {source['n']} selected rows" for source in self._snapshot.get("sources", []))
            text = ("Unapplied data/analysis changes: " + "; ".join(self._pending().reasons)
                    + f". Use {self._update_verb()}; showing the previous figure. "
                    if self._stale else "Generated result · ") + counts
        else:
            text = f"Choose stored event sources and a figure type, then {self._update_verb()}. Opening this page runs no analysis."
        if lead:
            text = f"{lead} {text}"
        self._status.setText(text if len(text) <= 220 else text[:217] + "…")
        self._status.setToolTip(text)
        self._status.setAccessibleDescription(text)
        self._workflow.primary_actions[0].setText("Generate ●" if self._stale else "Generate")
        self._workflow.export_button.setEnabled(self.has_data())
        for button in (self._fit_button, self._zoom_in, self._zoom_out):
            button.setEnabled(self.has_data())
        self.pendingChanged.emit()

    def _generate_from_user(self):
        self.refreshRequested.emit()
        try:
            self.analyze()
        except (TypeError, ValueError, OverflowError, RuntimeError, ImportError) as exc:
            QtWidgets.QMessageBox.warning(self, self._update_verb() + " — comparison figure", str(exc))

    def refresh(self):
        return self.analyze()

    @staticmethod
    def _metric_labels(sources, frames):
        domains = list(dict.fromkeys(signal_domain_of(source.provenance) for source in sources))
        domain = domains[0] if len(domains) == 1 else "mixed source units / domains"
        return {"dwell_s": "Dwell (ms)", "depth_pA": f"Median depth ({domain})",
                "amplitude_pA": f"Selected amplitude ({amplitude_units_label(frames)})",
                "frac_depth": "Fractional depth ΔI/I₀ (dimensionless)",
                "ecd_pA_s": f"Event area ({domain} · s)", "snr": "SNR (dimensionless)",
                "rise_s": "Rise time (ms)", "fall_s": "Fall time (ms)"}

    def analyze(self):
        """Prepare and render transactionally. The source tables are read-only."""
        from ..plot.comparison_figures import prepare_comparison

        config = self.config()
        self._validate_config(config)
        if not config["source_keys"]:
            raise ValueError("Select at least one event source.")
        if any(key not in self._sources for key in config["source_keys"]):
            raise ValueError(f"A selected source is unavailable. Rebind or uncheck it before {self._update_verb()}.")
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
        # Compute presentation reset before drawing: an old range may be invalid for
        # the incoming quantity/log scale. Install it as one history step only after
        # the new frozen geometry was successfully rendered and committed.
        before = self._appearance_widget.state()
        promised = self._words_for is not None and self._words_for == self._word_promise(config, before)
        moved = (bool(self._applied_config) and not promised
                 and self._quantity(self._applied_config) != self._quantity(config))
        appearance, reset_names, reset_axes, reset_keys = self._quantity_reset_state(
            snapshot, before, words=moved, preserve=promised)
        item = self._make_figure_item(snapshot, tables, appearance, self._source_appearance)
        self._words_for = None
        self._snapshot, self._tables = snapshot, {key: frame.copy(deep=True) for key, frame in tables.items()}
        self._source_recipes = recipes
        self._applied_config = deepcopy(config)
        self._stale = False
        self._source_appearance_editor.set_sources(snapshot["sources"])
        self._update_method_note()
        self._install_figure_item(item, fit=True)
        self._warnings.set_warnings(self._snapshot["warnings"], full_text=self.details_text())
        self._panel_reset_keys = reset_keys
        if appearance != before:
            self._appearance_widget.apply_appearance(appearance)
        self._reset_said = reset_sentence(reset_names, self._shows(config))
        self._appearance_widget._reset_took = (reset_names, reset_axes, "") if self._reset_said else None
        self._update_status(self._reset_said)
        self.workspaceChanged.emit()
        return True

    @staticmethod
    def _panel_word_keys(state):
        from ..plot.matplotlib_targets import storage_key
        roles = {storage_key('', key).rsplit('.', 1)[1]
                 for key in ('title', 'axis.bottom.label', 'axis.left.label')}
        return tuple(key for key, style in (state.text or ())
                     if key.startswith('annotation.mpl.') and key.rsplit('.', 1)[-1] in roles
                     and style.text is not None)

    def _word_promise(self, config, state):
        return (promise(self._quantity(config), state),
                tuple((key, state.text_style(key).text) for key in self._panel_word_keys(state)),
                tuple((key, pin.range(), pin.tick_major) for key, pin in (state.axes or ())
                      if key.startswith('mpl_')))

    def _quantity_reset_state(self, snapshot, before, *, words, preserve):
        from ..plot.comparison_figures import comparison_axis_quantities
        from ..plot.matplotlib_targets import panel_view
        if preserve or not self._snapshot:
            return before, (), (), ()
        state, names = cleared_words(before) if words else (before, ())
        keys = self._panel_word_keys(before) if words else ()
        for key in keys:
            state = state.with_text_style(key, replace(state.text_style(key), text=None))
        if keys: names += ('Panel titles and labels',)
        previous = comparison_axis_quantities(self._snapshot)
        incoming = comparison_axis_quantities(snapshot)
        axes = tuple((panel_view(panel), side, 'Panel X' if side == 'bottom' else 'Panel Y')
                     for (panel, side), quantity in previous.items()
                     if incoming.get((panel, side), quantity if not words else None) != quantity)
        state, pins = cleared_pins(state, axes)
        return state, tuple(dict.fromkeys((*names, *pins))), axes, keys

    @staticmethod
    def _quantity(config):
        """What a figure shows: its type, and the measurements on its axes."""
        kind = config["kind"]
        return kind, (tuple(config["features"]) if kind == "pairs" else
                      (config["x"], config["y"]) if kind == "facets" else (config["metric"],))

    @staticmethod
    def _shows(config):
        """What the status line says the plot now shows (quantity_reset)."""
        kind = config["kind"]
        if kind == "pairs":
            return f"a pair matrix of {len(config['features'])} measurements"
        if kind == "facets":
            return (f"{_SHOWS_METRIC[config['y']]} against {_SHOWS_METRIC[config['x']]}, "
                    "one panel per source")
        return f"individual events of {_SHOWS_METRIC[config['metric']]}"

    @staticmethod
    def _make_figure_item(snapshot, tables, appearance, source_appearance=None):
        from ..plot.comparison_figures import render_comparison
        from .matplotlib_figure_item import MatplotlibFigureItem

        figure = render_comparison(snapshot, tables, appearance, source_appearance=source_appearance,
                                   automatic_quantity_labels=True)
        try:
            return MatplotlibFigureItem(figure)
        finally:
            # Renderer returns a detached Figure, never a pyplot-managed window.
            figure.clear()

    def _install_figure_item(self, item, *, fit=False):
        view = (QtGui.QTransform(self._view.transform()),
                self._view.horizontalScrollBar().value(), self._view.verticalScrollBar().value()) \
               if self._figure_item is not None else None
        self._scene.clear()
        self._figure_item = item
        self._scene.addItem(item)
        self._scene.setSceneRect(item.boundingRect())
        self._series_selection.set_item(item)
        if fit:
            self._fit_pending = not self.isVisible()
            self.fit_canvas(emit=False)
        elif view is not None:
            # Redraw in the same canvas: integer scroll positions preserve the exact
            # viewport. centerOn(sceneCenter) rounds and can drift one pixel per edit.
            transform, horizontal, vertical = view
            self._view.setTransform(transform)
            self._view.horizontalScrollBar().setValue(horizontal)
            self._view.verticalScrollBar().setValue(vertical)
        from .matplotlib_text_controller import synchronize_controllers
        self._text_controllers = synchronize_controllers(
            self, item, self._appearance_widget, self._view, self._text_controllers)
        self.formatViewsChanged.emit()

    def fit_canvas(self, _checked=False, *, emit=True):
        if self._figure_item is None:
            return
        self._view.fit(self._figure_item.boundingRect())
        if emit:
            self._view_changed()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self._fit_pending:
            self._fit_pending = False
            self.fit_canvas(emit=False)

    def _view_state(self):
        center = self._view.mapToScene(self._view.viewport().rect().center())
        # "fitted": the view was still Fit canvas, so a reopened project fits the figure to
        # the canvas it opens in rather than restoring a scale taken at another height.
        return {"scale": float(self._view.transform().m11()), "center": [center.x(), center.y()],
                "fitted": bool(self._view._fitted)}

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

    def _reset_sentence_lapsed(self):
        """What the sentence on the status line said was automatic is typed or pinned again."""
        if self._reset_said:
            self._reset_said = ""
            self._update_status()

    def _appearance_from_user(self, state):
        if self._reset_said and any(state.text_style(key).text is not None for key in self._panel_reset_keys):
            self._reset_sentence_lapsed()
            self._appearance_widget._reset_took = None
        previous = self._appearance
        try:
            self._install_appearance(state)
        except (TypeError, ValueError, RuntimeError, ImportError) as exc:
            self._appearance_widget.set_state(previous)
            self._source_appearance_editor.set_state(self._source_appearance)
            if self._quiet:         # Style ▾ over several pages: the caller reports it, no dialog
                self._quiet_error = str(exc) or type(exc).__name__
                return
            QtWidgets.QMessageBox.warning(self, "Style comparison figure", str(exc))
            return
        self.workspaceChanged.emit()

    def appearance_state(self):
        return self._appearance.to_dict()

    def set_appearance_state(self, state):
        """A restore: the installed figure starts its appearance history over (design
        §4.2). The user's own edits come through :meth:`_install_appearance` instead, or
        each one would erase the history it is recorded in."""
        self._install_appearance(state)
        self._appearance_widget.clear_history()

    def _install_appearance(self, state):
        appearance = state if isinstance(state, PlotAppearance) else PlotAppearance.from_mapping(state)
        overrides = validate_source_appearance(self._source_appearance if appearance.source_appearance is None
                                               else appearance.source_appearance)
        appearance = replace(appearance, source_appearance=overrides)
        item = self._make_figure_item(self._snapshot, self._tables, appearance, overrides) if self._snapshot else None
        self._appearance = appearance
        self._source_appearance = overrides
        self._source_appearance_editor.set_state(overrides)
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
        self._install_appearance(replace(self._appearance, source_appearance=validate_source_appearance(state)))
        if self._snapshot:
            self._warnings.set_warnings(self._snapshot["warnings"], full_text=self.details_text())

    def _source_appearance_from_user(self, state, *, source_key=None):
        self._appearance_widget.apply_appearance(with_source_appearance_edit(
            self._appearance_widget.state(), state, view='comparison', source_key=source_key))

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
        appearance = replace(appearance, source_appearance=(source_appearance
            if appearance.source_appearance is None else appearance.source_appearance))
        source_appearance = validate_source_appearance(appearance.source_appearance)
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
            self._appearance_widget.clear_history()
            self._stale = payload["stale"]
            self._fit_pending = False
            if item is not None:
                self._install_figure_item(item)
            else:
                for controller in self._text_controllers:
                    controller.retire()
                    controller.deleteLater()
                self._text_controllers = []
                self._series_selection.set_item(None)
                self._scene.clear()
                self._figure_item = None
                self.formatViewsChanged.emit()
            self._restore_view(payload["canvas_view"])
            if payload["canvas_view"].get("fitted") is True and self._figure_item is not None:
                self._fit_pending = not self.isVisible()
                self._view.fit(self._figure_item.boundingRect())
            else:
                self._view.release_fit()    # the saved view is the user's; a resize keeps it
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
