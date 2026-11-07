"""Presentation-only workflow and warning chrome shared by analytical pages."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from PySide6 import QtCore, QtGui, QtWidgets

from .widget_transfer import prepare_widget_move


SECTIONS = ("Data", "Analysis", "Appearance", "Export")

#: What each old Section name means once the controls live in the Explorer's inspector
#: (a group key, or ``"@format"`` for the Format tab). A page's adapter may override it.
DEFAULT_REVEAL = {"Data": "sources", "Analysis": "plot", "Appearance": "@format",
                  "Export": "recipes"}


# -- the page / shell interface -------------------------------------------------------------
# Plain data: the Analysis Explorer's inspector and toolbar (gui/analysis_inspector.py,
# analysis_toolbar.py) reach a page ONLY through a PageAdapter that the page itself writes
# (``inspector_adapter()``). Nothing here imports another gui module, and nothing here
# computes: every callable is an existing handler of the page, every state reader reads
# state the page already keeps.

@dataclass(frozen=True)
class PendingState:
    count: int = 0                     # independent pending changes; 0 = nothing to update
    reasons: tuple[str, ...] = ()      # short phrases, most recent last; the tooltip lists them
    busy: bool = False                 # a background gather is running (I-V): Update plot is greyed


@dataclass(frozen=True)
class HistoryState:
    can_undo: bool = False
    can_redo: bool = False
    undo_text: str = "Nothing to undo"  # the tooltip the page already writes: "Undo: title bold on"
    redo_text: str = "Nothing to redo"
    mark: int = 0                      # depth of the undo stack; Style's batch undo compares it


@dataclass(frozen=True)
class ExportEntry:
    text: str                          # the page's own wording: "Export plotted coordinates CSV…"
    run: Callable[[], object]
    enabled: Callable[[], bool] | None = None   # evaluated when the menu is about to show
    kind: str = "figure"               # "figure" | "data" | "recipe" | "page" (orders and separates the menu)


@dataclass
class InspectorGroup:
    key: str                           # stable, persisted: "sources" "plot" "fit" "groups" "estimator" "layout" "recipes" ...
    title: str                         # "Sources", "Plot", "Fit" ...
    widgets: Sequence[QtWidgets.QWidget]   # EXISTING widgets (or form_rows containers), moved into the group, in order
    expanded: bool = True              # the default; the user's choice is persisted
    visible_when: Callable[[], bool] | None = None   # follows the page's own view (Event tab, Noise view)
    tooltip: str = ""


@dataclass
class FormatSpec:
    appearance: object                 # the page's PlotAppearanceWidget (duck-typed; the shell never imports it)
    controllers: Callable[[], Sequence[object]]      # FigureTextControllers of the plots ON SCREEN now
    extras: Mapping[str, Sequence[QtWidgets.QWidget]] = field(default_factory=dict)
        # page-owned Format widgets keyed by element panel: "figure" "legend" "series" "fit_label"
        # "caption" "colour_map". Moved into that panel, in order.
    defaults: Callable[[], object] | None = None
        # a PlotAppearance holding the page's real defaults; Distributions only
    views_changed: object | None = None            # a signal: which plot is on screen changed (Event tab, Noise view)
    project_sources: Callable[[], None] | None = None  # shell-owned, project appearance only
    series_provider: Callable[[], Sequence[object]] | None = None
    series_selected: object | None = None   # renderer signal(str stable series ID)
    series_select: Callable[[str | None], None] | None = None  # renderer selection overlay
    axis_log_target: Callable[[str, str], tuple[str, QtWidgets.QWidget] | None] | None = None
        # (plot id, axis side) -> existing inspector group/Format target and the page's
        # own Log control. Navigation only: the Format panel never owns a second Log box.


@dataclass
class PageAdapter:
    key: str                           # the Explorer's key (page id today)
    page: QtWidgets.QWidget
    groups: list[InspectorGroup]       # Data & analysis tab, top to bottom
    format: FormatSpec
    # the verb
    update: Callable[[], None]         # the page's existing primary handler
    update_tooltip: str = ""
    pending: Callable[[], PendingState] = PendingState
    pending_changed: object | None = None          # a signal, or None: the shell then recomputes on workspaceChanged
    # history (the page's own; the keys stay on the page)
    undo: Callable[[], bool] = lambda: False
    redo: Callable[[], bool] = lambda: False
    history: Callable[[], HistoryState] = HistoryState
    history_changed: object | None = None          # a signal
    # look
    apply_look: Callable[[object, bool], bool] = lambda look, keep_hand: False   # (PlotAppearance, keep_hand) -> moved
    reset_figure: Callable[[], None] = lambda: None
    # export
    exports: Callable[[], Sequence[ExportEntry]] = lambda: ()
    # view
    navigation: Callable[[], object | None] = lambda: None        # the PlotNavigation view_button resolves
    navigations: Callable[[], Sequence[object]] = lambda: ()
    status_bar: Callable[[], QtWidgets.QStatusBar | None] = lambda: None
    canvas_actions: Sequence[QtGui.QAction] = ()   # Lasso select (Distributions)
    # what the old Section names mean here (the compatibility shim)
    reveal_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_REVEAL))
    # lifecycle
    chrome: Sequence[QtWidgets.QWidget] = ()     # bars/toolbars hidden when hosted
    on_adopt: Callable[[], None] = lambda: None    # page-specific hosting steps (minimum size, readout, shortcuts)
    on_retire: Callable[[], None] = lambda: None
    fallback: bool = False             # compatibility marker; every real page supplies a native adapter
    kind: str | None = None            # tool kind; key is the independent figure id when hosted

    def __post_init__(self) -> None:
        if self.kind is None:
            self.kind = self.key

    @classmethod
    def from_widget(cls, page, key, widget, **fields) -> "PageAdapter":
        """An adapter whose history, look and reset come from a ``PlotAppearanceWidget``.

        Duck-typed (this module imports no gui module): ``widget.undo`` / ``redo``,
        ``undo_button.toolTip()`` / ``isEnabled()`` (the tooltips already say "Undo: title
        bold on"), ``history_mark()``, ``historyChanged``, ``apply_look`` and
        ``reset_figure``. Anything the caller passes in ``fields`` wins; ``groups`` and
        ``format`` are the caller's and have no default."""
        def history() -> HistoryState:
            undo, redo = widget.undo_button, widget.redo_button
            if undo is None or redo is None:       # a widget built without history
                return HistoryState()
            return HistoryState(
                can_undo=undo.isEnabled(), can_redo=redo.isEnabled(),
                undo_text=undo.toolTip(), redo_text=redo.toolTip(), mark=widget.history_mark())

        defaults = dict(
            undo=widget.undo, redo=widget.redo, history=history,
            history_changed=getattr(widget, "historyChanged", None),
            apply_look=lambda look, keep_hand: widget.apply_look(look, keep_hand=keep_hand),
            reset_figure=widget.reset_figure)
        return cls(key=key, page=page, **{**defaults, **fields})


def form_rows(rows: Sequence[tuple[str, QtWidgets.QWidget]], *, parent=None) -> QtWidgets.QWidget:
    """A new container with one ``QFormLayout`` that ADOPTS the existing ``widgets`` (a move).

    The label text comes from the tuple; the inline labels of the strip the widgets leave
    stay behind in its hidden husk. Minimum width 0, so a long combo shrinks instead of
    widening the inspector. Every adopted widget is shown (``setVisible(True)``): a Section
    switch may have hidden it, and a page that wants one hidden hides it again afterwards."""
    box = QtWidgets.QWidget(parent)
    box.setMinimumWidth(0)
    form = QtWidgets.QFormLayout(box)
    form.setContentsMargins(0, 0, 0, 0)
    form.setSpacing(4)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
    for label, widget in rows:
        widget.setMinimumWidth(0)
        prepare_widget_move(widget)
        form.addRow(label, widget)
        widget.setVisible(True)
    return box


class PlotWorkflowBar(QtWidgets.QWidget):
    """Switch control visibility without invoking any calculation or changing values."""

    sectionChanged = QtCore.Signal(str)

    def __init__(self, *, actions: Sequence[tuple[str, Callable]] = (),
                 exports: Sequence[tuple[str, Callable]] = (), parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sharedPlotWorkflow")
        self.setAccessibleName("Plot workflow")
        self._sections: dict[str, tuple[QtWidgets.QWidget, ...]] = {}
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Section"))
        self.sections = QtWidgets.QComboBox()
        self.sections.addItems(SECTIONS)
        self.sections.setAccessibleName("Plot control section")
        first, last = (QtGui.QKeySequence(f"Ctrl+{n}").toString(
            QtGui.QKeySequence.SequenceFormat.NativeText) for n in (1, len(SECTIONS)))
        self.sections.setToolTip(f"Data → Analysis → Appearance → Export ({first}–{last}). "
                                 "Changing section only shows controls; it never analyzes or "
                                 "changes your data.")
        self.sections.currentTextChanged.connect(self._show_section)
        layout.addWidget(self.sections)
        # Ctrl+1–4 (⌘1–4 on macOS) choose a section as the combo does. Window-wide, yet
        # only the page in front answers: a shortcut on a hidden widget never matches, and
        # the Explorer hides every other page's bar with its tab. Nothing else in the
        # Explorer or a page window binds these keys (Signal Conditioning's Ctrl+1–3 live
        # on its own widget, in another window); Ctrl+0 stays the main window's Reset.
        self.section_shortcuts: list[QtGui.QShortcut] = []
        for number, name in enumerate(SECTIONS, 1):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{number}"), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda name=name: self.set_section(name))
            self.section_shortcuts.append(shortcut)
        layout.addStretch(1)
        self.primary_actions: list[QtWidgets.QPushButton] = []
        for title, callback in actions:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, fn=callback: fn())
            button.setAccessibleName(title)
            button.setToolTip("Explicit analysis action. Appearance and section changes do not run it.")
            layout.addWidget(button)
            self.primary_actions.append(button)
        self.export_button = QtWidgets.QToolButton()
        self.export_button.setText("Export ▾")
        self.export_button.setAccessibleName("Export current plot")
        self.export_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.export_menu = QtWidgets.QMenu(self.export_button)
        for title, callback in exports:
            self.export_menu.addAction(title, callback)
        self.export_button.setMenu(self.export_menu)
        self.export_button.setEnabled(bool(exports))
        layout.addWidget(self.export_button)
        #: The page's Reset figure button, once :meth:`adopt_reset_figure` has put it here.
        self.reset_button: QtWidgets.QPushButton | None = None
        #: Hosted mode (:meth:`attach_inspector`): the Explorer's inspector shows the
        #: controls, so a Section name no longer toggles any widget.
        self._reveal: Callable[[str], None] | None = None

    def adopt_reset_figure(self, appearance, keeps: str = "",
                           takes_ranges: bool = False) -> QtWidgets.QPushButton:
        """Show ``appearance``'s Reset figure button here, left of Export ▾.

        The same button its Appearance section had (``PlotAppearanceWidget.take_reset_button``),
        so it resets the figure's words and look as one Undo step whichever Section is
        showing. ``keeps`` names the page's own Appearance rows that the shared look does not
        hold, and so Reset figure leaves (a colormap): the tool tip says so.
        ``takes_ranges`` is for a page with Axis properties, whose pinned ranges and major tick
        spacings are part of the appearance this button puts back (Distributions keeps its range
        in the Axis range boxes and has its own button text; Comparison figures has no Axis
        properties): the tool tip says so too."""
        button = appearance.take_reset_button()
        if takes_ranges:
            button.setToolTip(f"{button.toolTip()} A range or tick spacing typed in Axis "
                              "properties goes with the look.")
        if keeps:
            button.setToolTip(f"{button.toolTip()} Here it also keeps {keeps}.")
        layout = self.layout()
        layout.insertWidget(layout.indexOf(self.export_button), button)
        self.reset_button = button
        return button

    def add_view_control(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        """Put a view-only control (``plot_interaction.view_button``, or a pointer readout
        given ``stretch``) after the section selector and any view control already there.

        On the left with the other non-analytical control, so the explicit analysis verbs
        and Export keep their place at the right end of the bar."""
        layout = self.layout()
        index = layout.indexOf(self.sections) + 1
        while index < layout.count() and layout.itemAt(index).widget() is not None and \
                layout.itemAt(index).widget().property("nanoscopeViewControl"):
            index += 1
        widget.setProperty("nanoscopeViewControl", True)
        layout.insertWidget(index, widget, stretch)

    def set_sections(self, mapping: Mapping[str, Sequence[QtWidgets.QWidget]]) -> None:
        self._sections = {name: tuple(mapping.get(name, ())) for name in SECTIONS}
        self._show_section(self.sections.currentText())

    def add_section_widget(self, name: str, widget: QtWidgets.QWidget) -> None:
        """Show ``widget`` with section ``name`` too, for a control a page adds after the bar.

        The window templates (``gui.plot_templates``) join a page's Export section this way
        where that section is a row of buttons rather than a group they could sit inside.
        """
        if name not in SECTIONS:
            raise ValueError(f"Unknown plot section: {name}")
        self._sections[name] = (*self._sections.get(name, ()), widget)
        # Only this widget's visibility: re-running _show_section would re-announce the
        # section to listeners that already know it. Hosted, the inspector owns what shows.
        if self._reveal is None:
            widget.setVisible(self.sections.currentText() == name)

    def set_section(self, name: str) -> None:
        if name not in SECTIONS:
            raise ValueError(f"Unknown plot section: {name}")
        if self.sections.currentText() == name:
            self._show_section(name)
        else:
            self.sections.setCurrentText(name)

    def _show_section(self, name: str) -> None:
        if self._reveal is not None:
            # Hosted: the name is only a request to look at a part of the inspector.
            self.sectionChanged.emit(name)
            self._reveal(name)
            return
        shown = set(self._sections.get(name, ()))
        all_widgets = {w for widgets in self._sections.values() for w in widgets}
        for widget in all_widgets:
            widget.setVisible(widget in shown)
        self.sectionChanged.emit(name)

    def attach_inspector(self, reveal: Callable[[str], None]) -> None:
        """Hosted mode. ``_show_section(name)`` no longer toggles any widget: it emits
        ``sectionChanged`` and calls ``reveal(name)``. The four Ctrl+1..4 shortcuts are
        disabled (a disabled shortcut takes no part in Qt's ambiguity rule, so the
        Explorer's own keys are the only ones on those combinations)."""
        self._reveal = reveal
        for shortcut in self.section_shortcuts:
            shortcut.setEnabled(False)

    def detach_chrome(self) -> None:
        """Hide the bar (and the ``QToolBar`` it sits in). Every object stays alive: the
        Section combo, the verbs and Export ▾ are the compatibility facade behind the
        attribute names ``core/analysis_workspace.py`` and the tests read."""
        parent = self.parentWidget()
        self.hide()
        if isinstance(parent, QtWidgets.QToolBar):
            parent.hide()


def install_plot_workflow(window: QtWidgets.QMainWindow, *, sections, actions=(), exports=()):
    """Attach the same fixed workflow bar to a standalone or Explorer-embedded page."""
    toolbar = QtWidgets.QToolBar("Plot workflow", window)
    toolbar.setObjectName("sharedPlotWorkflowToolbar")
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    bar = PlotWorkflowBar(actions=actions, exports=exports)
    bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    toolbar.addWidget(bar)
    window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
    bar.set_sections(sections)
    return bar


def control_group(title: str, text: str) -> QtWidgets.QGroupBox:
    """A small explanatory section when the page consumes a fixed analysis source."""
    box = QtWidgets.QGroupBox(title)
    layout = QtWidgets.QVBoxLayout(box)
    label = QtWidgets.QLabel(text)
    label.setTextFormat(QtCore.Qt.PlainText)
    label.setWordWrap(True)
    layout.addWidget(label)
    return box


def bounded_controls(widget: QtWidgets.QWidget, maximum_height=260) -> QtWidgets.QScrollArea:
    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setMinimumHeight(80)
    scroll.setMaximumHeight(maximum_height)
    return scroll


def ensure_item_legend(plot, entries) -> None:
    """Provide an opt-in legend while preserving the historical hidden default."""
    pairs = tuple((item, name) for item, name in entries if item is not None)
    legend = getattr(plot, "legend", None)
    if legend is None:
        legend = plot.addLegend()
        legend.hide()
    if getattr(plot, "_nanoscope_workflow_legend_pairs", None) == pairs:
        return
    legend.clear()
    for item, name in pairs:
        legend.addItem(item, name)
    plot._nanoscope_workflow_legend_pairs = pairs


class WarningSummary(QtWidgets.QWidget):
    """A persistent first warning/count, with every original word in bounded detail."""

    def __init__(self, parent=None, *, title="Warnings", maximum_height=150) -> None:
        super().__init__(parent)
        self._title = title
        self._full = ""
        self._summary_text = ""
        self.setObjectName("plotWarningSummary")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QtWidgets.QHBoxLayout()
        self.summary = QtWidgets.QLabel()
        self.summary.setTextFormat(QtCore.Qt.PlainText)
        self.summary.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.summary.setWordWrap(False)  # _fit_summary supplies complete bounded lines
        self.summary.setMaximumHeight(48)
        self.summary.setMinimumWidth(1)
        self.summary.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.summary.setStyleSheet("color:#ffd479;")
        self.summary.setAccessibleName(title + " summary")
        self.summary.installEventFilter(self)
        row.addWidget(self.summary, 1)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText("Full details")
        self.toggle.setCheckable(True)
        self.toggle.setArrowType(QtCore.Qt.RightArrow)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        row.addWidget(self.toggle)
        layout.addLayout(row)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(maximum_height)
        self.details.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.details.setAccessibleName("Full " + title.lower())
        self.details.hide()
        layout.addWidget(self.details)
        self.toggle.toggled.connect(self._expand)
        self.hide()

    def _expand(self, checked: bool) -> None:
        self.details.setVisible(checked)
        self.toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

    def eventFilter(self, watched, event):  # noqa: N802 (Qt)
        if watched is self.summary and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.FontChange):
            self._fit_summary()
        return super().eventFilter(watched, event)

    def _fit_summary(self):
        """Show at most three whole lines; every omitted word remains in Details."""
        text = self._summary_text
        width = max(1, self.summary.contentsRect().width())
        layout = QtGui.QTextLayout(text, self.summary.font())
        option = QtGui.QTextOption()
        option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        lines = []
        layout.beginLayout()
        for index in range(3):
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            start = line.textStart()
            if index == 2:
                lines.append(self.summary.fontMetrics().elidedText(
                    text[start:].strip(), QtCore.Qt.ElideRight, width))
            else:
                lines.append(text[start:start + line.textLength()].strip())
        layout.endLayout()
        shown = "\n".join(lines)
        if self.summary.text() != shown:
            self.summary.setText(shown)
        height = self.summary.fontMetrics().lineSpacing() * max(1, len(lines)) + 2
        self.summary.setMinimumHeight(height)
        self.summary.setMaximumHeight(height)

    def set_warnings(self, warnings: Sequence[str], *, full_text: str | None = None) -> None:
        blocks = [str(w) for w in warnings if str(w).strip()]
        self._full = "\n\n".join(blocks) if full_text is None else str(full_text)
        self.details.setPlainText(self._full)
        first = " ".join(blocks[0].split()) if blocks else ""
        # The count and explicit disclosure survive truncation; full text is never lost.
        if len(first) > 170:
            first = first[:167].rstrip() + "…"
        self._summary_text = f"⚠ {self._title} ({len(blocks)}): {first}" if blocks else ""
        self._fit_summary()
        self.summary.setToolTip(self._full)
        self.setVisible(bool(self._full))

    def full_text(self) -> str:
        return self._full

    # Compatibility for existing warning-label read APIs: the retained text, not
    # the visually abbreviated summary, remains the authoritative warning string.
    def text(self) -> str:
        return self._full

    def setText(self, text: str) -> None:
        self.set_warnings((str(text),) if text else (), full_text=str(text))
