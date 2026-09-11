"""Read-only analysis catalog. Selecting an entry never constructs its analytical page."""
from collections.abc import Callable

from PySide6 import QtCore, QtWidgets

from .analysis_catalog import ANALYSIS_ENTRIES, ENTRY_BY_ROUTE, STATUS_HELP


class _ToolList(QtWidgets.QListWidget):
    def keyPressEvent(self, event):  # noqa: N802
        # Native item activation keys vary by platform. Make Enter explicit on
        # macOS as well, without requiring a double-click or a mouse.
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            item = self.currentItem()
            if item is not None:
                self.itemActivated.emit(item)
            event.accept()
            return
        super().keyPressEvent(event)


class AnalysisHome(QtWidgets.QWidget):
    launchRequested = QtCore.Signal(str)

    def __init__(self, parent=None, *, availability_provider: Callable[[str], tuple[bool, str]] | None = None):
        super().__init__(parent)
        self._availability_provider = availability_provider
        self._gates = {}
        root = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Choose a tool by the question you want to answer. This home lists every analysis, "
            "including tools you have not opened. Browsing this list does not run analysis.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        self._search = QtWidgets.QLineEdit()
        self._search.setObjectName("analysisHomeSearch")
        self._search.setPlaceholderText("Find a tool, plot type or input…")
        self._search.setClearButtonEnabled(True)
        root.addWidget(self._search)
        split = QtWidgets.QSplitter()
        self._list = _ToolList()
        self._list.setObjectName("analysisHomeTools")
        self._list.setMinimumWidth(265)
        self._list.setAlternatingRowColors(True)
        self._items = {}
        for entry in ANALYSIS_ENTRIES:
            item = QtWidgets.QListWidgetItem(entry.label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry.route)
            item.setToolTip(f"{entry.description}\n\nInputs: {entry.inputs}")
            item.setSizeHint(QtCore.QSize(270, 33))
            self._list.addItem(item)
            self._items[entry.route] = item
        split.addWidget(self._list)
        details_scroll = QtWidgets.QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        details = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(details)
        detail_layout.setContentsMargins(14, 4, 10, 8)
        self._group = QtWidgets.QLabel()
        self._group.setStyleSheet("color:#8098a8;")
        self._title = QtWidgets.QLabel()
        self._title.setStyleSheet("font-size:20px; font-weight:600;")
        self._description = QtWidgets.QLabel()
        self._inputs = QtWidgets.QLabel()
        self._availability = QtWidgets.QLabel()
        self._availability.setObjectName("analysisHomeAvailability")
        self._availability.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._availability.setAccessibleName("Tool availability and requirements")
        for label in (self._group, self._title, self._description, self._inputs, self._availability):
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            detail_layout.addWidget(label)
        self._open = QtWidgets.QPushButton()
        self._open.setObjectName("analysisHomeOpen")
        self._open.setToolTip("Open the selected tool. An unavailable tool explains its requirements without opening it.")
        detail_layout.addWidget(self._open, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        detail_layout.addSpacing(15)
        help_title = QtWidgets.QLabel("Understanding result status")
        help_title.setStyleSheet("font-weight:600;")
        detail_layout.addWidget(help_title)
        self._status_help = QtWidgets.QLabel(STATUS_HELP)
        self._status_help.setWordWrap(True)
        self._status_help.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        detail_layout.addWidget(self._status_help)
        detail_layout.addStretch(1)
        details_scroll.setWidget(details)
        split.addWidget(details_scroll)
        split.setStretchFactor(1, 1)
        split.setSizes([315, 750])
        root.addWidget(split, 1)
        self._list.currentItemChanged.connect(self._show_current)
        self._list.itemActivated.connect(lambda _item: self.request_current())
        self._open.clicked.connect(self.request_current)
        self._search.textChanged.connect(self._filter)
        self._list.setCurrentRow(0)
        self.refresh_availability()

    def route_ids(self):
        return tuple(self._items)

    def current_route(self):
        item = self._list.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None and not item.isHidden() else None

    def set_current_route(self, route):
        if route not in self._items:
            raise KeyError(route)
        self._search.clear()
        self._list.setCurrentItem(self._items[route])

    def _availability_for(self, route):
        if self._availability_provider is None:
            return True, "Open the tool to review its inputs."
        try:
            available, reason = self._availability_provider(route)
            if type(available) is not bool or not isinstance(reason, str):
                raise ValueError("Invalid availability result")
            return available, reason
        except (RuntimeError, TypeError, ValueError, KeyError):
            return False, "Availability could not be checked. Return to the main workspace and select a recording."

    def refresh_availability(self):
        """Only call the supplied read-only predicates, never page factories or refreshers."""
        for route, item in self._items.items():
            available, reason = self._availability_for(route)
            self._gates[route] = (available, reason)
            entry = ENTRY_BY_ROUTE[route]
            item.setToolTip(f"{entry.description}\n\nInputs: {entry.inputs}\n\n"
                            f"{'Available' if available else 'Needs input'}: {reason}")
            item.setData(QtCore.Qt.ItemDataRole.AccessibleTextRole,
                         f"{entry.label}. {'Available' if available else 'Needs input'}. {reason}")
        self._show_current()

    def _show_current(self, *_args):
        route = self.current_route()
        if route is None:
            self._group.clear()
            self._title.setText("No matching tools")
            self._description.setText("Try a plot name such as ECDF, a measurement such as conductance, or an input such as events.")
            self._inputs.clear()
            self._availability.clear()
            self._open.hide()
            return
        entry = ENTRY_BY_ROUTE[route]
        gate = self._gates.get(route)
        available, reason = gate if gate is not None else self._availability_for(route)
        self._group.setText(entry.group)
        self._title.setText(entry.label)
        self._description.setText(entry.description)
        self._inputs.setText("Inputs and interpretation\n" + entry.inputs)
        self._availability.setText(("Available" if available else "Needs input") + (" — " + reason if reason else ""))
        self._open.setText("Open " + entry.label if available else "Explain requirements")
        self._open.setAccessibleDescription(reason or entry.inputs)
        self._open.show()

    def _filter(self, value):
        query = value.strip().casefold()
        visible = []
        for route, item in self._items.items():
            entry = ENTRY_BY_ROUTE[route]
            hidden = query not in f"{entry.label} {entry.group} {entry.description} {entry.inputs}".casefold()
            item.setHidden(hidden)
            if not hidden:
                visible.append(item)
        if self.current_route() is None and visible:
            self._list.setCurrentItem(visible[0])
        self._show_current()

    def request_current(self):
        route = self.current_route()
        if route is None:
            return
        available, reason = self._availability_for(route)
        self._gates[route] = (available, reason)
        self._show_current()
        if available:
            self.launchRequested.emit(route)
        else:
            self._availability.setFocus()
