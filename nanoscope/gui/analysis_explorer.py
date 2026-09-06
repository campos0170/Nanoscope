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
from typing import Final

from PySide6 import QtCore, QtGui, QtWidgets

__all__ = [
    "AnalysisExplorer",
    "PAGE_DISTRIBUTIONS",
    "PAGE_DISTRIBUTION_SUMMARIES",
    "PAGE_EVENT_WAVEFORMS",
    "PAGE_NOISE",
    "PAGE_IV",
    "PAGE_VOLTAGE_SERIES",
    "PAGE_SUBPOPULATIONS",
]


PAGE_DISTRIBUTIONS: Final = "distributions"
PAGE_DISTRIBUTION_SUMMARIES: Final = "distribution_summaries"
PAGE_EVENT_WAVEFORMS: Final = "event_waveforms"
PAGE_NOISE: Final = "noise"
PAGE_IV: Final = "iv"
PAGE_VOLTAGE_SERIES: Final = "voltage_series"
PAGE_SUBPOPULATIONS: Final = "subpopulations"

_PAGE_LABELS: Final[dict[str, str]] = {
    PAGE_DISTRIBUTIONS: "Distributions",
    PAGE_DISTRIBUTION_SUMMARIES: "Distribution summaries",
    PAGE_EVENT_WAVEFORMS: "Event waveforms",
    PAGE_NOISE: "Noise",
    PAGE_IV: "I-V",
    PAGE_VOLTAGE_SERIES: "Voltage series",
    PAGE_SUBPOPULATIONS: "Subpopulations",
}

_STATUS_STYLES: Final[dict[str, str]] = {
    "current": "color:#79c99e; background:#193126; border:1px solid #315a43;",
    "stale": "color:#ffca78; background:#382a16; border:1px solid #6e5126;",
    "snapshot": "color:#8dccff; background:#172d3d; border:1px solid #315770;",
    "empty": "color:#a7adb5; background:#282c31; border:1px solid #454b52;",
}


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

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        context_provider: Callable[[], str] | None = None,
        status_provider: Callable[[str, QtWidgets.QWidget], tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("NanoScope — Analysis Explorer")
        self.resize(1280, 820)
        self.setMinimumSize(840, 560)

        self._context_provider = context_provider
        self._status_provider = status_provider
        self._pages: dict[str, QtWidgets.QWidget] = {}
        self._refreshers: dict[str, Callable[[], None] | None] = {}
        self._activators: dict[str, Callable[[], None] | None] = {}
        self._closing = False
        self._close_scheduled = False
        self._page_close_origin: QtWidgets.QWidget | None = None
        self._programmatic_tab_change = False

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)
        root.addWidget(self._build_header())
        figure_row = QtWidgets.QHBoxLayout()
        self._save_figure_btn = QtWidgets.QPushButton("Save to Figure Library…")
        self._save_figure_btn.setToolTip(
            "Save a named, frozen picture with exact plotted data, analysis settings and provenance in this project.")
        self._save_bundle_btn = QtWidgets.QPushButton("Save figure bundle…")
        self._save_bundle_btn.setToolTip(
            "Export the current picture, plotted numbers, configuration and provenance together as one ZIP file.")
        self._figure_library_btn = QtWidgets.QPushButton("Figure Library")
        self._save_figure_btn.clicked.connect(
            lambda: self.saveFigureRequested.emit(self.current_page_id() or ""))
        self._save_bundle_btn.clicked.connect(
            lambda: self.saveBundleRequested.emit(self.current_page_id() or ""))
        self._figure_library_btn.clicked.connect(self.figureLibraryRequested)
        for button in (self._save_figure_btn, self._save_bundle_btn, self._figure_library_btn):
            figure_row.addWidget(button)
        figure_row.addStretch(1)
        root.addLayout(figure_row)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(False)
        self._tabs.setTabsClosable(False)
        self._tabs.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self._tabs.currentChanged.connect(self._on_tab_activated)
        root.addWidget(self._tabs, 1)
        self.setCentralWidget(central)

    def _build_header(self) -> QtWidgets.QWidget:
        header = QtWidgets.QFrame()
        header.setObjectName("AnalysisExplorerHeader")
        header.setStyleSheet(
            "#AnalysisExplorerHeader { background:#20252b; border:1px solid #383f47; "
            "border-radius:6px; }")
        row = QtWidgets.QHBoxLayout(header)
        row.setContentsMargins(10, 6, 8, 6)
        row.setSpacing(9)

        title = QtWidgets.QLabel("Analysis Explorer")
        title.setStyleSheet("font-size:14px; font-weight:600;")
        row.addWidget(title)

        self._context_label = QtWidgets.QLabel("No recording selected")
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

        self._refresh_btn = QtWidgets.QPushButton("Refresh current")
        self._refresh_btn.setObjectName("analysisExplorerRefresh")
        self._refresh_btn.setToolTip(
            "Refresh only the analytical view in the selected tab from the current "
            "recordings. Other tabs are left unchanged.")
        self._refresh_btn.clicked.connect(self.refresh_current)
        row.addWidget(self._refresh_btn)
        self._set_status("NO DATA", "empty")
        return header

    # -- pages -----------------------------------------------------------------------------
    def add_page(
        self,
        page_id: str,
        widget: QtWidgets.QWidget,
        *,
        refresh: Callable[[], None] | None = None,
        activate: Callable[[], None] | None = None,
    ) -> None:
        """Embed ``widget`` once and remember how its shared Refresh is routed."""
        if page_id not in _PAGE_LABELS:
            raise KeyError(f"Unknown Analysis Explorer page: {page_id!r}")
        existing = self._pages.get(page_id)
        if existing is not None and existing is not widget:
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
            self._tabs.addTab(widget, _PAGE_LABELS[page_id])
        finally:
            self._programmatic_tab_change = False

        changed = getattr(widget, "workspaceChanged", None)
        if changed is not None and hasattr(changed, "connect"):
            changed.connect(self.refresh_header)

    def show_page(
        self,
        page_id: str,
        widget: QtWidgets.QWidget,
        *,
        refresh: Callable[[], None] | None = None,
        activate: Callable[[], None] | None = None,
    ) -> None:
        """Add/select ``widget`` and focus this one top-level window."""
        self.add_page(page_id, widget, refresh=refresh, activate=activate)
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

    def page_count(self) -> int:
        return len(self._pages)

    def remove_page(self, page_id: str) -> QtWidgets.QWidget | None:
        """Forget an about-to-be-destroyed lazy page and remove its tab."""
        page = self._pages.pop(page_id, None)
        self._refreshers.pop(page_id, None)
        self._activators.pop(page_id, None)
        if page is None:
            return None
        index = self._tabs.indexOf(page)
        if index >= 0:
            # Removing the current tab synchronously emits ``currentChanged`` for the
            # next tab. During a project reset that next page belongs to the outgoing
            # project too; treating the signal like a user click would recompute it just
            # before it is removed. Keep the visual change, suppress only its activator.
            self._programmatic_tab_change = True
            try:
                self._tabs.removeTab(index)
            finally:
                self._programmatic_tab_change = False
        page.removeEventFilter(self)
        self.refresh_header()
        return page

    def current_page_id(self) -> str | None:
        current = self._tabs.currentWidget()
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
        if not self._programmatic_tab_change:
            activate = self._activators.get(page_id or "")
            if activate is not None:
                activate()
        self.refresh_header()

    @QtCore.Slot()
    def refresh_header(self) -> None:
        context = "No recording selected"
        if self._context_provider is not None:
            try:
                context = str(self._context_provider() or context)
            except RuntimeError:
                pass
        self._context_label.setText(context)
        self._context_label.setToolTip(context)

        page_id = self.current_page_id()
        page = self._pages.get(page_id or "")
        text, state = ("NO DATA", "empty")
        if page_id is not None and page is not None and self._status_provider is not None:
            try:
                text, state = self._status_provider(page_id, page)
            except RuntimeError:
                pass
        self._set_status(str(text), str(state))
        enabled = self._refreshers.get(page_id or "") is not None
        self._refresh_btn.setEnabled(enabled)
        self._save_figure_btn.setEnabled(page is not None and state != "empty")
        self._save_bundle_btn.setEnabled(page is not None and state != "empty")

    def _set_status(self, text: str, state: str) -> None:
        state = state if state in _STATUS_STYLES else "empty"
        shown = text.strip().upper() or "NO DATA"
        self._status_label.setText(shown)
        self._status_label.setToolTip(
            "Current: this view reflects its live source.\n"
            "Stale: controls or source data changed after the visible result.\n"
            "Saved snapshot: a project result is shown without recomputing it.")
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
