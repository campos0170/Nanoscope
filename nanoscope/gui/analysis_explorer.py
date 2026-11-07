"""One home for NanoScope's analytical views.

The individual analysis widgets deliberately continue to own their calculations,
controls, exports, and project-workspace adapters.  :class:`AnalysisExplorer` is
only presentation chrome: it embeds those existing widgets as lazy pages, keeps a
small amount of shared context visible, and routes one Refresh button to the page
that is currently in front of the user.

Keeping the pages lazy is important.  Constructing the noise and event views can
run numerical work, and opening a project must not eagerly create (or recompute)
every analytical result merely so that a tab exists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Final

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets
from .analysis_catalog import ANALYSIS_LABELS, PAGE_IDS, PAGE_HOME, STATUS_HELP, FIGURE_NAVIGATION_HELP
from .analysis_home import AnalysisHome
from .analysis_inspector import InspectorHost
from .analysis_toolbar import ExplorerToolbar
from .plot_styles import StyleMenu, StyleResult, StyleService
from .plot_workflow import FormatSpec, PageAdapter, ExportEntry
from .figure_list import FigureList

__all__ = [
    "AnalysisExplorer",
    "PAGE_HOME",
    "PAGE_DISTRIBUTIONS",
    "PAGE_DISTRIBUTION_SUMMARIES",
    "PAGE_COMPARISON_FIGURES",
    "PAGE_EVENT_WAVEFORMS",
    "PAGE_NOISE",
    "PAGE_IV",
    "PAGE_VOLTAGE_SERIES",
    "PAGE_SUBPOPULATIONS",
]


PAGE_DISTRIBUTIONS: Final = "distributions"
PAGE_DISTRIBUTION_SUMMARIES: Final = "distribution_summaries"
PAGE_COMPARISON_FIGURES: Final = "comparison_figures"
PAGE_EVENT_WAVEFORMS: Final = "event_waveforms"
PAGE_NOISE: Final = "noise"
PAGE_IV: Final = "iv"
PAGE_VOLTAGE_SERIES: Final = "voltage_series"
PAGE_SUBPOPULATIONS: Final = "subpopulations"

_PAGE_LABELS: Final[dict[str, str]] = {key: ANALYSIS_LABELS[key] for key in PAGE_IDS}

_HEADER_HEIGHT: Final = 24      # the one-line figure header

_STATUS_STYLES: Final[dict[str, str]] = {
    "current": "color:#79c99e; background:#193126; border:1px solid #315a43;",
    "stale": "color:#ffca78; background:#382a16; border:1px solid #6e5126;",
    "snapshot": "color:#8dccff; background:#172d3d; border:1px solid #315770;",
    "empty": "color:#a7adb5; background:#282c31; border:1px solid #454b52;",
    "home": "color:#b7d6e8; background:#202c34; border:1px solid #405464;",
}


class _ElidedLabel(QtWidgets.QLabel):
    """A one-line label that shrinks and draws its text elided on the right. ``text()`` is
    always the full text (the header's context is read by tests and by the tooltip)."""

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt override
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def elided_text(self) -> str:
        """The text as drawn at the label's current width."""
        return self.fontMetrics().elidedText(
            self.text(), QtCore.Qt.TextElideMode.ElideRight, self.contentsRect().width())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
        painter.drawText(self.contentsRect(),
                         int(self.alignment()) | int(QtCore.Qt.AlignmentFlag.AlignVCenter),
                         self.elided_text())


class AnalysisExplorer(QtWidgets.QMainWindow):
    """A tabbed shell around the application's existing analysis widgets.

    ``context_provider`` and ``status_provider`` are intentionally callbacks rather
    than references to ``MainWindow`` or ``Session``.  The shell remains ignorant of
    scientific state; its owner is still the one place that resolves which recording
    and which persisted snapshot a page describes.
    """

    visibilityChanged = QtCore.Signal(bool)
    saveFigureRequested = QtCore.Signal(str)
    saveBundleRequested = QtCore.Signal(str)
    figureLibraryRequested = QtCore.Signal()
    launchRequested = QtCore.Signal(str)
    savedFigureActivated = QtCore.Signal(str)
    snapshotRequested = QtCore.Signal(str)
    compositionRequested = QtCore.Signal(object)
    recipeRequested = QtCore.Signal()
    saveRecipeRequested = QtCore.Signal(str)
    manageRecipesRequested = QtCore.Signal(str)
    exportSelectedRequested = QtCore.Signal()
    savedDeleteRequested = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        context_provider: Callable[[], str] | None = None,
        status_provider: Callable[[str, QtWidgets.QWidget], tuple[str, str]] | None = None,
        availability_provider: Callable[[str], tuple[bool, str]] | None = None,
        source_session=None,
        source_provider: Callable | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NanoScope — Analysis Explorer")
        self.resize(1280, 820)
        self.setMinimumSize(840, 560)

        self._context_provider = context_provider
        self._status_provider = status_provider
        self._availability_provider = availability_provider
        self._source_session = source_session
        self._source_provider = source_provider
        self._pages: dict[str, QtWidgets.QWidget] = {}
        self._refreshers: dict[str, Callable[[], None] | None] = {}
        self._activators: dict[str, Callable[[], None] | None] = {}
        self._closing = False
        self._close_scheduled = False
        self._page_close_origin: QtWidgets.QWidget | None = None
        self._programmatic_tab_change = False
        self._figure_workspace = None
        # Application view preference, independent of project data and active selection.
        self._view_settings = QtCore.QSettings()
        saved_figures = self._view_settings.value("analysisExplorer/figuresVisible", True)
        self._figures_visible = (saved_figures if isinstance(saved_figures, bool)
                                 else str(saved_figures).lower() != "false")
        self._figure_links: list[tuple[object, object]] = []
        self._saved_figure_provider = lambda: ()
        self._saved_figure_status = None
        self._saved_view = None
        self._figure_refresh_timer = QtCore.QTimer(self)
        self._figure_refresh_timer.setSingleShot(True)
        self._figure_refresh_timer.timeout.connect(self.refresh_figures)

        # The one toolbar and the right-hand inspector (Phase 1). Both are bound to the page
        # in front through its PageAdapter; neither knows a page's attributes.
        self._toolbar = ExplorerToolbar(self)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._toolbar)
        self._inspector = InspectorHost()
        self._style = StyleService(self, self)
        self._style_menu = StyleMenu(self, self._style)
        self._toolbar.set_style_menu(self._style_menu.menu())

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(5)
        header = self._build_header()
        self._build_retired_buttons(header)
        self._refusal_route: str | None = None
        root.addWidget(self._build_refusal_banner())

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(False)
        self._tabs.setTabsClosable(False)
        self._tabs.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self._home = AnalysisHome(availability_provider=availability_provider)
        self._home.launchRequested.connect(self.launchRequested)
        self._home.availabilityRefreshed.connect(self._sync_refusal)
        # A double-click, Enter or Explain requirements on a tool that cannot open yet is
        # refused by Home's own gate, so the owner's refusal branch never sees it.
        self._home.launchRefused.connect(self.show_refusal)
        self._tabs.addTab(self._home, ANALYSIS_LABELS[PAGE_HOME])
        self._tabs.currentChanged.connect(self._on_tab_activated)
        # The body: the tool tabs on the left, the inspector on the right. A hidden
        # inspector (Analysis home, the toolbar's toggle) leaves the tabs the whole width.
        self._body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._body.setObjectName("explorerBody")
        self._body.setChildrenCollapsible(False)
        # The one-line figure header sits above the tab bar, where the Figures list will
        # leave it in Phase 2.
        left = QtWidgets.QWidget()
        left.setObjectName("explorerFigureColumn")
        column = QtWidgets.QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        column.addWidget(header)
        self._pending_notice = QtWidgets.QLabel()
        self._pending_notice.setObjectName("analysisPendingNotice")
        self._pending_notice.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._pending_notice.setWordWrap(True)
        self._pending_notice.setStyleSheet("color:#ffca78; padding:3px 6px;")
        self._pending_notice.setAccessibleName("Pending plot changes")
        self._pending_notice.hide()
        column.addWidget(self._pending_notice)
        column.addWidget(self._tabs, 1)
        self._body.addWidget(left)
        self._body.addWidget(self._inspector)
        self._body.setStretchFactor(0, 1)
        self._body.setStretchFactor(1, 0)
        self._body.splitterMoved.connect(self._remember_inspector_width)
        body_row = QtWidgets.QHBoxLayout()
        self._figures = FigureList()
        self._figures.hide()
        body_row.addWidget(self._figures)
        body_row.addWidget(self._body, 1)
        root.addLayout(body_row, 1)
        self.setCentralWidget(central)
        self._connect_toolbar()
        self._connect_figure_list()
        # ⌘1 and ⌘2 are the inspector's two tabs, and ⌘Return (the toolbar's) is Update
        # plot, on every hosted page. The pages' own Ctrl+1 to 4 are handed over at adoption.
        self._tab_shortcuts: list[QtGui.QShortcut] = []
        for index, keys in enumerate(("Ctrl+1", "Ctrl+2")):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda index=index: self._inspector.set_tab(index))
            shortcut.setEnabled(False)
            self._tab_shortcuts.append(shortcut)
        settings = self._inspector.settings()
        self._toolbar.inspector_toggle.setChecked(settings.visible)
        self._inspector.set_user_visible(settings.visible)
        self.refresh_header()

    def _connect_toolbar(self) -> None:
        toolbar = self._toolbar
        toolbar.refreshFromRecordingsRequested.connect(self.refresh_current)
        toolbar.saveFigureRequested.connect(
            lambda: self.saveFigureRequested.emit(self.current_page_id() or ""))
        toolbar.saveBundleRequested.connect(
            lambda: self.saveBundleRequested.emit(self.current_page_id() or ""))
        toolbar.figureLibraryRequested.connect(self.figureLibraryRequested)
        toolbar.exportSelectedRequested.connect(self.exportSelectedRequested)
        toolbar.inspector_toggle.toggled.connect(self._toggle_inspector)
        toolbar.figures_action.toggled.connect(self.set_figures_visible)
        toolbar.synced.connect(self._sync_chip_help)
        toolbar.synced.connect(self._sync_style_reference)
        toolbar.synced.connect(lambda: toolbar.set_duplicate_enabled(self._figures.duplicate_button.isEnabled()))

    def _connect_figure_list(self) -> None:
        figures = self._figures
        figures.liveActivated.connect(self._activate_figure)
        figures.savedActivated.connect(self.savedFigureActivated)
        figures.selectionChanged.connect(self._figure_selection_changed)
        figures.newRequested.connect(self._new_figure)
        figures.duplicateRequested.connect(lambda key: self._figure_command("duplicate", key))
        self._toolbar.duplicateRequested.connect(figures.duplicate_current)
        figures.renameRequested.connect(self._rename_figure)
        figures.deleteRequested.connect(self._delete_figures)
        figures.restoreRequested.connect(lambda: self._figure_command("restore_deleted"))
        figures.snapshotRequested.connect(self._request_snapshot)
        figures.compositionRequested.connect(self.compositionRequested)
        figures.recipeRequested.connect(self.recipeRequested)
        figures.libraryRequested.connect(self.figureLibraryRequested)

    def figure_workspace(self):
        return self._figure_workspace

    def figures_list(self) -> FigureList:
        return self._figures

    def bind_figure_workspace(self, controller) -> None:
        """Bind a record owner; ``None`` retains the standalone tool-tab contract."""
        for signal, slot in self._figure_links:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._figure_links.clear()
        self._style._drop_batch()
        self._figure_workspace = controller
        self._toolbar.set_figure_commands(self._figures.new_menu if controller is not None else None)
        self._figures.set_toolbar_commands_available(controller is not None)
        self._apply_figures_visibility()
        self._tabs.tabBar().setVisible(controller is None)
        if controller is not None:
            for name, slot in (("recordsChanged", self._schedule_figures_refresh),
                               ("recordChanged", self._schedule_figures_refresh),
                               ("selectionChanged", self._schedule_figures_refresh),
                               ("currentChanged", self._figure_current_changed),
                               ("requestFailed", self._figure_request_failed)):
                signal = getattr(controller, name)
                signal.connect(slot)
                self._figure_links.append((signal, slot))
        self.refresh_figures()

    def set_figures_visible(self, shown: bool) -> None:
        """Collapse only navigation; preserve selected records, controls and results."""
        self._toolbar.commit_focus()
        self._figures_visible = bool(shown)
        self._view_settings.setValue("analysisExplorer/figuresVisible", self._figures_visible)
        self._apply_figures_visibility()

    def _apply_figures_visibility(self) -> None:
        self._figures.setVisible(self._figure_workspace is not None and self._figures_visible)
        with QtCore.QSignalBlocker(self._toolbar.figures_action):
            self._toolbar.figures_action.setChecked(self._figures_visible)
        # Preserve the chosen inspector width; released space belongs to the canvas.
        self._apply_inspector_width()

    def _schedule_figures_refresh(self, *_args) -> None:
        # A selection signal may still be traversing its QTreeWidgetItem. Replace rows
        # after that gesture returns, coalescing record/history notifications as well.
        self._figure_refresh_timer.start(0)

    def set_saved_figure_provider(self, provider: Callable | None, *, status_provider=None) -> None:
        self._saved_figure_provider = provider or (lambda: ())
        self._saved_figure_status = status_provider
        self._schedule_figures_refresh()

    def schedule_figures_refresh(self) -> None:
        """Refresh row metadata after a source/project mutation, without opening pages."""
        self._schedule_figures_refresh()

    def refresh_figures(self, *_args) -> None:
        controller = self._figure_workspace
        records = controller.records() if controller is not None else ()
        unsupported = getattr(controller, "unsupported_records", lambda: ())()
        saved = tuple(self._saved_figure_provider())
        if self._saved_view is not None:
            saved_id = self._saved_view.current_id()
            if saved_id and any(entry.artifact_id == saved_id for entry in saved):
                self._saved_view.refresh(saved_id)
            elif self._tabs.indexOf(self._saved_view) >= 0:
                self.clear_saved_figure()
        statuses = {}
        for artifact in saved:
            if self._saved_figure_status is not None:
                statuses[artifact.artifact_id] = self._saved_figure_status(artifact)
            else:
                from ..core.figure_library import figure_status
                statuses[artifact.artifact_id] = figure_status(artifact, None)
        self._figures.set_records(records, saved,
                                 selected=controller.selected_ids() if controller is not None else (),
                                 current=controller.current_id() if controller is not None else None,
                                 unsupported=unsupported, loaded=self._pages, saved_status=statuses)
        for record in records:
            page = self._pages.get(record.spec.figure_id)
            if page is not None:
                self._tabs.setTabText(self._tabs.indexOf(page), record.spec.name)
        self.refresh_header()

    def _sync_style_reference(self) -> None:
        """Read the live history value or a dormant record; never materialize a view."""
        if self.saved_figure_selected():
            self._toolbar.style_button.setText("Frozen snapshot")
            self._toolbar.style_button.setToolTip("This saved artwork is frozen. Style its live figure and capture a new snapshot to change the look.")
            self._toolbar.style_button.setAccessibleName("Frozen snapshot style")
            return
        owner = self._figure_workspace
        key = owner.current_id() if owner is not None else self.current_page_id()
        record = owner.get(key) if owner is not None and key else None
        adapter = self._toolbar.adapter()
        appearance = getattr(getattr(adapter, "format", None), "appearance", None)
        read = getattr(appearance, "state", None)
        if callable(read) and (owner is None or self._toolbar.bound_key() == key):
            value = read()
            binding = getattr(value, "style_binding", None)
            texts = dict(getattr(value, "text", None) or ())
            overrides = any(any(field != "text" and setting is not None for field, setting in
                                (style.to_dict() if hasattr(style, "to_dict") else style).items())
                            for style in texts.values())
            fit = getattr(value, "fit_label", None)
            overrides = overrides or any(getattr(fit, field, None) is not None for field in ("size_pt", "color"))
            self._toolbar.set_style_reference(binding, local_overrides=overrides, has_figure=True)
            return
        presentation = getattr(getattr(record, "spec", None), "presentation", None)
        binding = None
        overrides = False
        if presentation is not None:
            if presentation.style_ref:
                binding = {"ref": presentation.style_ref, "snapshot": presentation.style_snapshot,
                           "overrides": presentation.overrides}
            overrides = any(bool(text.overrides) for text in getattr(record.spec, "texts", ()))
        self._toolbar.set_style_reference(binding, local_overrides=overrides, has_figure=record is not None)

    def _figure_command(self, command: str, *args):
        controller = self._figure_workspace
        if controller is None:
            return None
        self._toolbar.commit_focus()
        if command == "duplicate" and args and controller.get(args[0]) is None:
            if self._saved_view is not None and self._saved_view.show_entry(args[0]):
                return self._saved_view.duplicate_current()
            return None
        return getattr(controller, command)(*args)

    def _activate_figure(self, figure_id: str) -> None:
        self._figure_command("activate", figure_id)

    def _figure_current_changed(self, figure_id: str) -> None:
        # Activation/materialization is the controller's transaction; this never builds.
        if figure_id in self._pages:
            # Reopening the same live figure from a menu is still an intentional
            # navigation from a saved row, even when the owner's current ID did
            # not change. Keep an existing live multi-selection when its current
            # row already matches. New rows are selected by the scheduled refresh.
            if (self._figures.current_id() != figure_id
                    and figure_id in self._figures._items):
                self._figures.restore_selection((figure_id,), current_id=figure_id)
            self._programmatic_tab_change = True
            try:
                self._tabs.setCurrentWidget(self._pages[figure_id])
            finally:
                self._programmatic_tab_change = False
            self._bind_front()
        self._schedule_figures_refresh()

    def _figure_selection_changed(self) -> None:
        if self._figure_workspace is not None:
            self._figure_workspace.set_selected(self._figures.selected_ids())
        self._sync_batch_export()
        self._toolbar.sync()

    def _sync_batch_export(self):
        enabled = bool(self._figures.selected_ids() or self._figures.selected_ids("saved"))
        self._toolbar.set_batch_export_enabled(enabled, available=self._figure_workspace is not None)

    def _new_figure(self, kind: str, recipe) -> None:
        self._figure_command("create", kind, recipe)

    def _request_snapshot(self, figure_id: str) -> None:
        if figure_id:
            self._toolbar.commit_focus()
            self.snapshotRequested.emit(figure_id)

    def _rename_figure(self, figure_id: str) -> None:
        owner = self._figure_workspace
        record = owner.get(figure_id) if owner is not None else None
        if record is None:
            if self._saved_view is not None and self._saved_view.show_entry(figure_id):
                self._saved_view.rename_current()
            return
        name, accepted = QtWidgets.QInputDialog.getText(
            self, "Rename figure", "Figure name:", text=record.spec.name)
        if accepted and name.strip() and owner is self._figure_workspace:
            self._figure_command("rename", figure_id, name.strip())

    def _delete_figures(self, ids) -> None:
        owner = self._figure_workspace
        if owner is None:
            return
        records = [owner.get(key) for key in ids]
        unsupported = {entry["id"]: entry for entry in getattr(owner, "unsupported_records", lambda: ())()}
        saved = {entry.artifact_id: entry for entry in self._saved_figure_provider()}
        names = [record.spec.name if record is not None else
                 str(saved[key].payload.get("name") or "Saved figure") if key in saved else
                 str(unsupported[key].get("name") or "Unsupported figure")
                 for key, record in zip(ids, records) if record is not None or key in unsupported or key in saved]
        if not names:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete figures?", "Delete " + ", ".join(f"‘{name}’" for name in names)
            + "?\nOnly the selected entries are removed. Unselected snapshots and compositions remain."
            + ("\nDeleted live figures can be restored until this project closes."
               if any(record is not None for record in records) else ""),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer == QtWidgets.QMessageBox.StandardButton.Yes and owner is self._figure_workspace:
            for key, record in zip(ids, records):
                if record is not None and owner.get(key) is record:
                    self._figure_command("delete", key)
                elif record is None and key in unsupported:
                    current = {entry["id"]: entry for entry in getattr(owner, "unsupported_records", lambda: ())()}
                    if current.get(key) == unsupported[key]:
                        self._figure_command("delete", key)
            remaining = {entry.artifact_id: entry for entry in self._saved_figure_provider()}
            saved_ids = tuple(key for key in ids if key in saved and remaining.get(key) is saved[key])
            if saved_ids:
                self.savedDeleteRequested.emit(saved_ids)

    def _figure_request_failed(self, figure_id: str, reason: str) -> None:
        record = self._figure_workspace.get(figure_id) if self._figure_workspace is not None else None
        name = record.spec.name if record is not None else "Figure"
        self._refusal_route = None
        self._refusal_label.setText(f"{name}: {reason}")
        self._refusal.show()

    def _sync_chip_help(self) -> None:
        """The status chip's tooltip: the shared help, and while the page has pending
        changes, why it is out of date (the same reasons Update plot's tooltip lists)."""
        if self.saved_figure_selected():
            self._status_label.setToolTip(self._saved_view.current_status()[1])
            self._pending_notice.hide()
            return
        text = STATUS_HELP
        if self._toolbar.pending_count() > 0:
            reasons = "; ".join(self._toolbar.pending_reasons()) or "changes waiting for Update plot"
            text = f"{STATUS_HELP}\n\nOut of date: {reasons}."
            self._pending_notice.setText(f"Pending: {reasons}. Use Update plot to apply these settings.")
            self._pending_notice.setAccessibleDescription(self._pending_notice.text())
            self._pending_notice.show()
        else:
            self._pending_notice.hide()
        self._status_label.setToolTip(text)

    def _toggle_inspector(self, shown: bool) -> None:
        self._inspector.set_user_visible(shown)
        self._apply_inspector_width()

    def _apply_inspector_width(self) -> None:
        """Give a visible inspector its remembered width (clamped to 320..560)."""
        if self._inspector.isHidden():
            return
        width = self._inspector.settings().width
        sizes = self._body.sizes()
        if len(sizes) == 2 and sizes[1] == width:
            return
        total = max(sum(sizes), self.width() - 16)
        self._body.setSizes([max(0, total - width), width])

    def _remember_inspector_width(self, _pos: int, _index: int) -> None:
        if not self._inspector.isHidden():
            sizes = self._body.sizes()
            if len(sizes) == 2 and sizes[1] > 0:
                self._inspector.settings().width = sizes[1]

    # -- the Phase 1 shell API -------------------------------------------------------------------
    def toolbar(self) -> ExplorerToolbar:
        return self._toolbar

    def inspector(self) -> InspectorHost:
        return self._inspector

    def _open_project_sources(self) -> None:
        if self._source_session is None or not callable(self._source_provider):
            return
        from .project_sources import ProjectSourcesDialog
        self._toolbar.commit_focus()
        dialog = ProjectSourcesDialog(self._source_session, self._source_provider, self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def adapter(self, page_id: str) -> PageAdapter | None:
        """The adapter an installed page is hosted through, else None."""
        return self._inspector.adapter(page_id)

    def update_current(self) -> None:
        """Update plot: commit the focused edit in the inspector, then the front page's own
        verb. The toolbar button and ⌘Return land here."""
        self._toolbar.run_update()

    def reveal(self, target: str) -> bool:
        """Show ``target`` (a group key, ``"@format"``, an old Section name) of the front page."""
        return self._inspector.reveal(target)

    def apply_look(self, look, *, scope: str = "page", keep_hand: bool = True) -> StyleResult:
        """Apply a look (a ``PlotAppearance``) to the front page or to every open page."""
        return self._style.apply(self._style.name_of(look), look, scope=scope,
                                 keep_hand=keep_hand)

    def _bind_front(self) -> None:
        """Point the toolbar and the inspector at the page in front. Moves nothing."""
        if self.saved_figure_selected():
            self._inspector.show_page(None)
            self._toolbar.bind_saved(self._saved_view.current_id(), self._saved_view.exports)
            for shortcut in self._tab_shortcuts:
                shortcut.setEnabled(False)
            return
        key = self.current_page_id()
        key = key if key in self._pages else None
        self._inspector.show_page(key)
        adapter = self._inspector.adapter(key) if key is not None else None
        hosted = adapter is not None and self._inspector.has_content(key)
        self._toolbar.bind(key if hosted else None, adapter if hosted else None)
        for shortcut in self._tab_shortcuts:
            shortcut.setEnabled(hosted)
        if hosted:
            self._apply_inspector_width()

    def saved_figure_selected(self) -> bool:
        return self._saved_view is not None and self._tabs.currentWidget() is self._saved_view

    def show_saved_figure(self, figure_id: str, view) -> bool:
        """Display one frozen entry in the canvas area, using the project list for selection."""
        self._toolbar.commit_focus()
        if not view.show_entry(figure_id):
            return False
        if self._saved_view is not None and self._saved_view is not view:
            self.clear_saved_figure()
        self._saved_view = view
        self._programmatic_tab_change = True
        try:
            if self._tabs.indexOf(view) < 0:
                self._tabs.addTab(view, "Saved figure")
            self._tabs.setCurrentWidget(view)
        finally:
            self._programmatic_tab_change = False
        # A direct row selection already owns its exact multi-selection. A
        # programmatic Save/Duplicate/Compose must instead select its new result.
        # Do not rebuild the tree during its currentItemChanged callback.
        if self._figures.editable_id() != figure_id:
            if figure_id not in self._figures._items:
                self.refresh_figures()
            self._figures.restore_selection((figure_id,), current_id=figure_id)
        self.clear_refusal()
        self._bind_front()
        self.refresh_header()
        self.show()
        return True

    def clear_saved_figure(self) -> None:
        """Remove a deleted/outgoing project's frozen view without activating a live page."""
        view = self._saved_view
        if view is None:
            return
        self._programmatic_tab_change = True
        try:
            if self._tabs.currentWidget() is view:
                self._tabs.setCurrentWidget(self._home)
            index = self._tabs.indexOf(view)
            if index >= 0:
                self._tabs.removeTab(index)
            view.hide()
        finally:
            self._programmatic_tab_change = False
        self._bind_front()

    def _build_header(self) -> QtWidgets.QWidget:
        """The one-line figure header: ``[Analysis home]  context ... [CURRENT]``, 24 px."""
        header = QtWidgets.QFrame()
        header.setObjectName("AnalysisExplorerHeader")
        header.setFixedHeight(_HEADER_HEIGHT)
        header.setStyleSheet(
            "#AnalysisExplorerHeader { background:#20252b; border:1px solid #383f47; "
            "border-radius:6px; }")
        row = QtWidgets.QHBoxLayout(header)
        row.setContentsMargins(4, 1, 6, 1)
        row.setSpacing(8)

        self._home_btn = QtWidgets.QPushButton(ANALYSIS_LABELS[PAGE_HOME])
        self._home_btn.setObjectName("analysisExplorerHome")
        self._home_btn.setFlat(True)
        self._home_btn.setFixedHeight(_HEADER_HEIGHT - 6)
        # The header is dark whatever the theme, so its button carries its own colours.
        self._home_btn.setStyleSheet(
            "#analysisExplorerHome { color:#9cc7ea; background:transparent; border:none; "
            "padding:0 8px; } #analysisExplorerHome:hover { color:#d4e9f8; }")
        self._home_btn.setToolTip("Browse every analysis and its input requirements without running it.\n\n" + FIGURE_NAVIGATION_HELP)
        self._home_btn.clicked.connect(self.show_home)
        row.addWidget(self._home_btn)

        self._context_label = _ElidedLabel("No recording selected")
        self._context_label.setObjectName("analysisExplorerContext")
        self._context_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._context_label.setStyleSheet("color:#b7bec7;")
        self._context_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        row.addWidget(self._context_label, 1)

        self._status_label = QtWidgets.QLabel("NO DATA")
        self._status_label.setObjectName("analysisExplorerStatus")
        self._status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._status_label.setMinimumWidth(78)
        row.addWidget(self._status_label)
        self._set_status("NO DATA", "empty")
        return header

    def _build_retired_buttons(self, header: QtWidgets.QWidget) -> None:
        """The buttons of the two bands the toolbar replaced, kept as HIDDEN objects.

        ``Refresh current`` is Update plot ▾ ▸ Refresh from recordings, and ``Save to Figure
        Library…``, ``Save figure bundle…`` and ``Figure Library`` are Export ▾ items. The
        objects stay (a hidden widget is the repo's way to retire one) with their
        connections and their enabled state, so what read or clicked them still works."""
        self._refresh_btn = QtWidgets.QPushButton("Refresh current", header)
        self._refresh_btn.setObjectName("analysisExplorerRefresh")
        self._refresh_btn.setToolTip(
            "Refresh only the analytical view in the selected tab from the current "
            "recordings. Other tabs are left unchanged.")
        self._refresh_btn.clicked.connect(self.refresh_current)
        self._save_figure_btn = QtWidgets.QPushButton("Save to Figure Library…", header)
        self._save_figure_btn.setToolTip(
            "Save a named, frozen picture with exact plotted data, analysis settings and provenance in this project.")
        self._save_bundle_btn = QtWidgets.QPushButton("Save figure bundle…", header)
        self._save_bundle_btn.setToolTip(
            "Export the current picture, plotted numbers, configuration and provenance together as one ZIP file.")
        self._figure_library_btn = QtWidgets.QPushButton("Figure Library", header)
        self._save_figure_btn.clicked.connect(
            lambda: self.saveFigureRequested.emit(self.current_page_id() or ""))
        self._save_bundle_btn.clicked.connect(
            lambda: self.saveBundleRequested.emit(self.current_page_id() or ""))
        self._figure_library_btn.clicked.connect(self.figureLibraryRequested)
        for button in (self._refresh_btn, self._save_figure_btn, self._save_bundle_btn,
                       self._figure_library_btn):
            button.hide()

    def _build_refusal_banner(self) -> QtWidgets.QWidget:
        """Why a chosen tool did not open, said here rather than behind this window."""
        self._refusal = QtWidgets.QFrame()
        self._refusal.setObjectName("analysisExplorerRefusal")
        self._refusal.setStyleSheet(
            "#analysisExplorerRefusal { background:#382a16; border:1px solid #6e5126; "
            "border-radius:6px; }")
        row = QtWidgets.QHBoxLayout(self._refusal)
        row.setContentsMargins(10, 6, 6, 6)
        self._refusal_label = QtWidgets.QLabel()
        self._refusal_label.setObjectName("analysisExplorerRefusalText")
        self._refusal_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._refusal_label.setWordWrap(True)
        self._refusal_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._refusal_label.setStyleSheet("color:#ffca78;")
        row.addWidget(self._refusal_label, 1)
        dismiss = QtWidgets.QToolButton()
        dismiss.setText("Dismiss")
        dismiss.setAccessibleName("Dismiss this message")
        dismiss.clicked.connect(self.clear_refusal)
        row.addWidget(dismiss, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self._refusal.hide()
        return self._refusal

    def show_refusal(self, route: str, reason: str) -> None:
        """Say why ``route`` cannot open: a banner here, and Home showing its requirements.

        The owner used to post the reason to the main window's status bar, behind this
        window, so choosing an unavailable tool looked like nothing happening. Nothing is
        constructed or run; Home only selects the entry, whose detail pane explains it.
        """
        if route in self._home.route_ids():
            self._home.set_current_route(route)
        self.show_home()
        self._refusal_route = route
        label = ANALYSIS_LABELS.get(route, "This tool")
        self._refusal_label.setText(f"{label} cannot open yet. {reason}".strip())
        self._refusal.show()

    def refusal_text(self) -> str:
        """The banner's sentence while it is shown, else ``""``."""
        return self._refusal_label.text() if not self._refusal.isHidden() else ""

    @QtCore.Slot()
    def clear_refusal(self) -> None:
        self._refusal_route = None
        self._refusal.hide()

    @QtCore.Slot()
    def _sync_refusal(self) -> None:
        # The refused tool became available (a recording opened): the sentence is stale.
        if self._refusal_route is not None and self._home.availability(self._refusal_route)[0]:
            self.clear_refusal()

    # -- pages -----------------------------------------------------------------------------
    def show_home(self) -> None:
        """Navigate to the persistent catalog, without touching any analytical page."""
        self._programmatic_tab_change = True
        try:
            self._tabs.setCurrentWidget(self._home)
        finally:
            self._programmatic_tab_change = False
        self.refresh_header()
        self.show()
        self.raise_()
        self.activateWindow()

    def home(self) -> AnalysisHome:
        return self._home

    def add_page(
        self,
        page_id: str,
        widget: QtWidgets.QWidget,
        *,
        refresh: Callable[[], None] | None = None,
        activate: Callable[[], None] | None = None,
        kind: str | None = None,
        name: str | None = None,
    ) -> None:
        """Embed ``widget`` once and remember how its shared Refresh is routed."""
        record = self._figure_workspace.get(page_id) if self._figure_workspace is not None else None
        kind = kind or (record.spec.kind if record is not None else page_id)
        if kind not in _PAGE_LABELS:
            raise KeyError(f"Unknown Analysis Explorer page kind: {kind!r}")
        label = name or (record.spec.name if record is not None else _PAGE_LABELS[kind])
        existing = self._pages.get(page_id)
        if existing is not None and existing is not widget:
            if self._figure_workspace is not None:
                raise ValueError("Remove a figure's existing surface before installing its replacement.")
            # A project transition normally removes disposable pages explicitly.  Keep
            # this defensive path for callers/tests that retire a legacy page attribute
            # directly: one page id still means one widget, never two hidden copies.
            old = self.remove_page(page_id)
            if old is not None:
                old.close()
                old.deleteLater()
            existing = None
        if existing is widget:
            self._refreshers[page_id] = refresh
            self._activators[page_id] = activate
            return

        # QMainWindow and SubpopReportWindow deliberately construct themselves as
        # top-level windows.  Clearing Window *after* reparenting turns the unchanged
        # widget into an ordinary page while preserving its central widget/toolbars.
        widget.setParent(self._tabs)
        widget.setWindowFlags(QtCore.Qt.WindowType.Widget)
        widget.setProperty("nanoscopeAnalysisExplorerPage", True)
        widget.installEventFilter(self)
        self._pages[page_id] = widget
        self._refreshers[page_id] = refresh
        self._activators[page_id] = activate
        self._programmatic_tab_change = True
        try:
            self._tabs.addTab(widget, label)
        finally:
            self._programmatic_tab_change = False

        changed = getattr(widget, "workspaceChanged", None)
        if changed is not None and hasattr(changed, "connect"):
            changed.connect(self.refresh_header)
        # Adopt: the page's controls move into the inspector, once, for its lifetime here.
        make = getattr(widget, "inspector_adapter", None)
        adapter = (make() if callable(make) else
                   PageAdapter(key=page_id, page=widget, groups=[],
                               format=FormatSpec(None, lambda: ()), update=lambda: None, kind=kind))
        if adapter.key != page_id or adapter.kind != kind:
            adapter = replace(adapter, key=page_id, kind=kind)
        if self._source_session is not None and callable(self._source_provider):
            adapter.format.project_sources = self._open_project_sources
        if self._figure_workspace is not None:
            from .figure_recipes import SUPPORTED
            original_exports = adapter.exports
            def recipe_exports():
                result = list(original_exports())
                if kind in SUPPORTED:
                    result.append(ExportEntry("Save as recipe…", lambda: self.saveRecipeRequested.emit(page_id), kind="recipe"))
                result.append(ExportEntry("Manage recipes…" if kind in SUPPORTED else "About recipes / saved recipes…",
                                          lambda: self.manageRecipesRequested.emit(kind), kind="recipe"))
                return result
            groups = [group for group in adapter.groups if group.key != "recipes"]
            if kind == PAGE_DISTRIBUTIONS:
                # Saved templates now live under New figure, but JSON sidecars
                # still have a distinct full-configuration import contract. Move
                # the native button, preserving its validator, warnings and Undo.
                from .plot_workflow import InspectorGroup
                import_button = getattr(widget, "_import_cfg_btn", None)
                if import_button is not None:
                    controls = [import_button]
                    history_note = getattr(widget, "_history_note", None)
                    if history_note is not None:
                        controls.append(history_note)
                    groups.append(InspectorGroup("configuration_import", "Import plot settings",
                        controls, expanded=False,
                        tooltip="Import a JSON plot configuration into this figure, including its source selections and typed text. Undo restores the previous settings."))
            adapter = replace(adapter, groups=groups, exports=recipe_exports)
        self._inspector.adopt(page_id, adapter)

    def show_page(
        self,
        page_id: str,
        widget: QtWidgets.QWidget,
        *,
        refresh: Callable[[], None] | None = None,
        activate: Callable[[], None] | None = None,
        kind: str | None = None,
        name: str | None = None,
    ) -> None:
        """Add/select ``widget`` and focus this one top-level window."""
        self.add_page(page_id, widget, refresh=refresh, activate=activate, kind=kind, name=name)
        self._programmatic_tab_change = True
        try:
            self._tabs.setCurrentWidget(widget)
        finally:
            self._programmatic_tab_change = False
        widget.setProperty("nanoscopeAnalysisExplorerClosing", False)
        widget.show()
        self.refresh_header()
        self.show()
        self.raise_()
        self.activateWindow()

    def page(self, page_id: str) -> QtWidgets.QWidget | None:
        return self._pages.get(page_id)

    def raise_page(self, page_id: str) -> bool:
        """Bring an installed page and this window forward, as a click on its tab would
        (so its activator runs). ``False`` when the page is not installed."""
        page = self._pages.get(page_id)
        if page is None:
            if (self._saved_view is not None
                    and any(entry.artifact_id == page_id for entry in self._saved_figure_provider())):
                return self.show_saved_figure(page_id, self._saved_view)
            return False
        if self._tabs.currentWidget() is not page:
            self._tabs.setCurrentWidget(page)
        self.refresh_header()
        self.show()
        self.raise_()
        self.activateWindow()
        return True

    def page_count(self) -> int:
        return len(self._pages)

    def remove_page(self, page_id: str) -> QtWidgets.QWidget | None:
        """Forget an about-to-be-destroyed lazy page and remove its tab."""
        page = self._pages.pop(page_id, None)
        self._refreshers.pop(page_id, None)
        self._activators.pop(page_id, None)
        if page is None:
            return None
        # The bound adapter goes before the page is handed back for deleteLater.
        self._toolbar.forget(page_id)
        self._inspector.retire(page_id)
        index = self._tabs.indexOf(page)
        if index >= 0:
            # Removing the current tab synchronously emits ``currentChanged`` for the
            # next tab. During a project reset that next page belongs to the outgoing
            # project too; treating the signal like a user click would recompute it just
            # before it is removed. Keep the visual change, suppress only its activator.
            self._programmatic_tab_change = True
            try:
                # QTabWidget must remove its own tab and stacked-layout item together.
                # Forget only that item's Python wrapper before Qt deletes the C++
                # item; taking it out first would desynchronize the tab indices.
                stack = page.parentWidget()
                # itemAt reparents Python ownership under this native stack wrapper.
                # Keep it alive: dropping the temporary wrapper invalidates a still-
                # live page's native child wrappers (including its QActions).
                self._tab_stack_owner = stack
                layout = stack.layout() if stack is not None else None
                item_index = layout.indexOf(page) if layout is not None else -1
                if item_index >= 0:
                    item = layout.itemAt(item_index)
                    if isinstance(item, QtWidgets.QWidgetItem):
                        shiboken6.invalidate(item)
                self._tabs.removeTab(index)
            finally:
                self._programmatic_tab_change = False
        page.removeEventFilter(self)
        self.refresh_header()
        return page

    def current_page_id(self) -> str | None:
        current = self._tabs.currentWidget()
        if self._saved_view is not None and current is self._saved_view:
            return self._saved_view.current_id()
        if current is self._home:
            return PAGE_HOME
        return next((key for key, page in self._pages.items() if page is current), None)

    def page_ids(self) -> tuple[str, ...]:
        """Installed pages in their visible tab order."""
        return tuple(
            key
            for index in range(self._tabs.count())
            for key, page in self._pages.items()
            if page is self._tabs.widget(index)
        )

    # -- shared chrome ----------------------------------------------------------------------
    @QtCore.Slot()
    def refresh_current(self) -> None:
        page_id = self.current_page_id()
        if page_id == PAGE_HOME:
            self.refresh_header()
            return
        refresh = self._refreshers.get(page_id or "")
        if refresh is None:
            return
        refresh()
        self.refresh_header()

    @QtCore.Slot(int)
    def _on_tab_activated(self, _index: int) -> None:
        """Bring a directly-clicked tab current before its next paint.

        Menu/launcher routing prepares a page itself and suppresses this callback.  A
        click on the tab bar has no such route, so its lightweight activator lets the
        owner rebind live data.  Saved/frozen-state policy remains the owner's decision.
        """
        page_id = self.current_page_id()
        page = self._pages.get(page_id or "")
        if page is not None:
            page.setProperty("nanoscopeAnalysisExplorerClosing", False)
            self.clear_refusal()     # it explained Home's selection, not this page
        if not self._programmatic_tab_change:
            activate = self._activators.get(page_id or "")
            if activate is not None:
                activate()
        self._bind_front()
        self.refresh_header()

    @QtCore.Slot()
    def refresh_header(self) -> None:
        if self.saved_figure_selected():
            entry = self._saved_view.current_entry()
            payload = entry.payload if entry is not None else {}
            context = " · ".join(str(value) for value in
                                 (payload.get("name", "Saved figure"), payload.get("source_label", "")) if value)
            self._context_label.setText(context)
            self._context_label.setToolTip(context)
            status, reason = self._saved_view.current_status()
            self._set_status(status, "stale" if status == "Outdated" else "snapshot")
            self._status_label.setToolTip(reason)
            self._refresh_btn.setEnabled(False)
            self._save_figure_btn.setEnabled(False)
            self._save_bundle_btn.setEnabled(False)
            self._toolbar.set_library_enabled(False)
            self._sync_batch_export()
            self._toolbar.sync()
            return
        context = "No recording selected"
        if self._context_provider is not None:
            try:
                context = str(self._context_provider() or context)
            except RuntimeError:
                pass
        if self._figure_workspace is not None:
            record = self._figure_workspace.get(self.current_page_id())
            if record is not None:
                context = f"{record.spec.name} · {context}"
        self._context_label.setText(context)
        self._context_label.setToolTip(context)

        page_id = self.current_page_id()
        page = self._pages.get(page_id or "")
        text, state = ("NO DATA", "empty")
        if page_id == PAGE_HOME:
            text, state = ("Analysis home", "home")
            self._home.refresh_availability()
        elif page_id is not None and page is not None and self._status_provider is not None:
            try:
                text, state = self._status_provider(page_id, page)
            except RuntimeError:
                pass
        self._set_status(str(text), str(state))
        enabled = page_id == PAGE_HOME or self._refreshers.get(page_id or "") is not None
        self._refresh_btn.setEnabled(enabled)
        self._refresh_btn.setText("Refresh availability" if page_id == PAGE_HOME else "Refresh current")
        self._refresh_btn.setToolTip(
            "Recheck which tools are available. No analytical view is constructed or recomputed."
            if page_id == PAGE_HOME else
            "Refresh only the selected analytical view from its current sources. Other tabs are unchanged.")
        self._save_figure_btn.setEnabled(page is not None and state != "empty")
        self._save_bundle_btn.setEnabled(page is not None and state != "empty")
        # The toolbar's Export ▾ shell items follow the same rule, and its other buttons
        # re-read the front page; the inspector re-reads what it shows. This is the one
        # refresh route for the silent paths that emit nothing (a project open, a restore).
        self._toolbar.set_library_enabled(page is not None and state != "empty")
        self._sync_batch_export()
        self._toolbar.sync()
        self._inspector.refresh_current()

    def _set_status(self, text: str, state: str) -> None:
        state = state if state in _STATUS_STYLES else "empty"
        shown = text.strip().upper() or "NO DATA"
        self._status_label.setText(shown)
        self._status_label.setToolTip(STATUS_HELP)
        self._status_label.setStyleSheet(
            _STATUS_STYLES[state] +
            " border-radius:8px; padding:2px 7px; font-size:10px; font-weight:700;")

    # -- lifetime ---------------------------------------------------------------------------
    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        # A legacy page may still expose its own Close button.  Let its closeEvent run
        # (PlotsWindow uses it to checkpoint and stop an advisory worker), then close the
        # containing Explorer as the user would expect.  Merely switching tabs generates
        # Hide, not Close, and therefore never closes the shell.
        if (event.type() == QtCore.QEvent.Type.Close and watched in self._pages.values()
                and not self._closing and not self._close_scheduled):
            watched.setProperty("nanoscopeAnalysisExplorerClosing", True)
            self._page_close_origin = watched
            self._close_scheduled = True
            QtCore.QTimer.singleShot(0, self._close_after_page)
        return super().eventFilter(watched, event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.visibilityChanged.emit(True)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802 - Qt override
        super().hideEvent(event)
        self.visibilityChanged.emit(False)

    def _close_after_page(self) -> None:
        self._close_scheduled = False
        if not self._closing:
            self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt override
        if self._figure_workspace is not None:
            # Closing the workspace window hides it. Records, pending controls and their
            # results survive; Delete and project close are explicit controller operations.
            self._toolbar.commit_focus()
            super().closeEvent(event)
            return
        # Child widgets do not receive Close merely because their parent is hidden.
        # Send it explicitly so the original standalone-window cleanup contracts remain
        # intact, most importantly PlotsWindow's worker cancellation/checkpoint/reset.
        self._closing = True
        origin, self._page_close_origin = self._page_close_origin, None
        try:
            for page in tuple(self._pages.values()):
                # The legacy page whose own Close initiated this shell close has
                # already completed its closeEvent/checkpoint/reset.  Closing it a
                # second time would checkpoint that reset state over the real one.
                if page is origin:
                    continue
                try:
                    page.setProperty("nanoscopeAnalysisExplorerClosing", True)
                    page.close()
                except RuntimeError:
                    pass
        finally:
            self._closing = False
        super().closeEvent(event)
        # The marker is for the synchronous Close/Hide sequence only.  Pages are kept
        # alive and may be shown again; a later tab switch must once again read as a tab
        # switch, not as a second close.
        for page in tuple(self._pages.values()):
            try:
                page.setProperty("nanoscopeAnalysisExplorerClosing", False)
            except RuntimeError:
                pass
