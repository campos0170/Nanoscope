"""The project's figure index. Listing records never constructs an analytical page."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .analysis_catalog import ANALYSIS_LABELS, NEW_FIGURES, ENTRY_BY_ROUTE

ID = QtCore.Qt.ItemDataRole.UserRole
CATEGORY = int(ID) + 1
STATUS = int(ID) + 2


class _FigureNameDelegate(QtWidgets.QStyledItemDelegate):
    """Keep a long name's distinct suffix visible without changing its stored label."""
    def initStyleOption(self, option, index):  # noqa: N802 - Qt override
        super().initStyleOption(option, index)
        option.textElideMode = QtCore.Qt.TextElideMode.ElideMiddle


class FigureList(QtWidgets.QWidget):
    liveActivated = QtCore.Signal(str)
    savedActivated = QtCore.Signal(str)
    selectionChanged = QtCore.Signal()
    newRequested = QtCore.Signal(str, object)
    recipeRequested = QtCore.Signal()
    duplicateRequested = QtCore.Signal(str)
    renameRequested = QtCore.Signal(str)
    deleteRequested = QtCore.Signal(object)
    restoreRequested = QtCore.Signal()
    snapshotRequested = QtCore.Signal(str)
    compositionRequested = QtCore.Signal(object)
    libraryRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("projectFigures")
        self.setMinimumWidth(180)
        self.setMaximumWidth(340)
        self._items = {}
        self._current_live = None
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 4, 0)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(QtWidgets.QLabel("Figures"), 1)
        self.new_button = QtWidgets.QToolButton()
        self.new_button.setText("New ▾")
        self.new_button.setAccessibleName("New figure")
        self.new_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.new_menu = QtWidgets.QMenu(self)
        self.new_menu.setToolTipsVisible(True)
        for label, kind, recipe in NEW_FIGURES:
            action = self.new_menu.addAction(label)
            requirement = ENTRY_BY_ROUTE[kind].inputs
            action.setToolTip(requirement + "\nCreates an empty figure; Update plot runs the analysis.")
            action.setStatusTip(requirement)
            action.triggered.connect(lambda _checked=False, kind=kind, recipe=recipe:
                                     self.newRequested.emit(kind, dict(recipe) if recipe else None))
        self.new_menu.addSeparator()
        self.new_menu.addAction("From recipe…", self.recipeRequested.emit)
        self.new_button.setMenu(self.new_menu)
        heading.addWidget(self.new_button)
        column.addLayout(heading)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setObjectName("projectFiguresList")
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Figure", "State"])
        self.tree.setItemDelegateForColumn(0, _FigureNameDelegate(self.tree))
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setAccessibleName("Live figures, snapshots and compositions")
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemActivated.connect(lambda item, _column: self._activate(item))
        column.addWidget(self.tree, 1)
        actions = QtWidgets.QGridLayout()
        self._action_layout = actions
        self.duplicate_button = QtWidgets.QPushButton("Duplicate")
        self.rename_button = QtWidgets.QPushButton("Rename…")
        self.delete_button = QtWidgets.QPushButton("Delete…")
        self.restore_button = QtWidgets.QPushButton("Restore deleted")
        self.snapshot_button = QtWidgets.QPushButton("Snapshot")
        self.snapshot_button.setToolTip("Capture the current live figure; saved snapshots remain unchanged.")
        self.compose_button = QtWidgets.QPushButton("Compose…")
        for index, button in enumerate((self.duplicate_button, self.rename_button, self.delete_button,
                                        self.restore_button, self.snapshot_button, self.compose_button)):
            actions.addWidget(button, index // 2, index % 2)
        column.addLayout(actions)
        self.duplicate_button.clicked.connect(self.duplicate_current)
        self.rename_button.clicked.connect(lambda: self._emit_current(self.renameRequested))
        self.delete_button.clicked.connect(lambda: self.deleteRequested.emit(self.deletable_ids()))
        self.restore_button.clicked.connect(self.restoreRequested.emit)
        self.snapshot_button.clicked.connect(lambda: self.snapshotRequested.emit(self.current_id() or ""))
        self.compose_button.clicked.connect(lambda: self.compositionRequested.emit(self.selected_ids("saved")))
        library = QtWidgets.QPushButton("Snapshot details…")
        library.clicked.connect(self.libraryRequested.emit)
        column.addWidget(library)
        self.set_records((), ())

    def set_toolbar_commands_available(self, available: bool) -> None:
        """Keep one visible New/Duplicate pair when the Explorer supplies its toolbar.

        The original controls and signals remain available to standalone hosts. Hiding
        duplicates does not rebuild the tree or change the current/multiple selection.
        """
        self.new_button.setVisible(not available)
        self.duplicate_button.setVisible(not available)
        buttons = (self.duplicate_button, self.rename_button, self.delete_button,
                   self.restore_button, self.snapshot_button, self.compose_button)
        for button in buttons:
            self._action_layout.removeWidget(button)
        if available:
            for index, button in enumerate((self.rename_button, self.delete_button,
                                             self.snapshot_button, self.compose_button)):
                self._action_layout.addWidget(button, index // 2, index % 2)
            self._action_layout.addWidget(self.restore_button, 2, 0, 1, 2)
        else:
            for index, button in enumerate(buttons):
                self._action_layout.addWidget(button, index // 2, index % 2)

    def selected_ids(self, category="live") -> tuple[str, ...]:
        return tuple(item.data(0, ID) for item in self.tree.selectedItems()
                     if item.data(0, ID) and (item.data(0, CATEGORY) == category
                     or category == "saved" and item.data(0, CATEGORY) in ("snapshot", "composition")))

    def restore_selection(self, ids, current_id=None):
        """Restore the original rows without activating or materializing any figure."""
        wanted = set(ids)
        with QtCore.QSignalBlocker(self.tree):
            self.tree.clearSelection()
            self.tree.setCurrentItem(self._items.get(current_id), 0,
                                     QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
            for key, item in self._items.items():
                item.setSelected(key in wanted)
        self._sync_actions()
        self.selectionChanged.emit()

    def current_id(self, category="live") -> str | None:
        item = self.tree.currentItem()
        return item.data(0, ID) if item is not None and item.data(0, CATEGORY) == category else None

    def deletable_ids(self) -> tuple[str, ...]:
        return self.selected_ids("live") + self.selected_ids("saved") + self.selected_ids("unsupported")

    def editable_id(self) -> str | None:
        """The selected live or saved figure, never a previous live selection."""
        return self.current_id() or self.current_id("snapshot") or self.current_id("composition")

    def set_records(self, records, saved, *, selected=(), current=None, unsupported=(),
                    loaded=(), saved_status=None) -> None:
        """Refresh names/status by durable ID, preserving saved-row selection and expansion."""
        old_saved = self.selected_ids("saved") + self.selected_ids("unsupported")
        old_current = self.tree.currentItem()
        old_id = old_current.data(0, ID) if old_current is not None else None
        activated_live = current is not None and current != self._current_live
        if activated_live:
            # New/activate owns the foreground. A status refresh, by contrast,
            # keeps the saved selection and multi-selection exactly as it was.
            old_saved = ()
        expanded = {self.tree.topLevelItem(i).text(0): self.tree.topLevelItem(i).isExpanded()
                    for i in range(self.tree.topLevelItemCount())}
        self._current_live = current
        loaded = set(loaded)
        saved_status = saved_status or {}
        with QtCore.QSignalBlocker(self.tree):
            self.tree.clear()
            self._items = {}
            groups = {}
            for category, label in (("live", "Live"), ("snapshot", "Snapshots"),
                                     ("composition", "Compositions")):
                group = QtWidgets.QTreeWidgetItem(self.tree, [label])
                group.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                group.setExpanded(expanded.get(label, True))
                groups[category] = group
            for record in records:
                spec = record.spec
                status = str(getattr(spec.result, "status", "") or "")
                item = QtWidgets.QTreeWidgetItem(groups["live"], [spec.name])
                item.setData(0, ID, spec.figure_id)
                item.setData(0, CATEGORY, "live")
                label = {"current": "Current", "stale": "Stale", "empty": "No data",
                         "frozen": "Saved result"}.get(status, "Unknown")
                dormant = spec.figure_id not in loaded
                item.setText(1, label + (" · Dormant" if dormant else ""))
                item.setData(0, STATUS, status)
                detail = ("Dormant: its stored figure is not loaded; displaying this row runs no analysis."
                          if dormant else "The editable figure is loaded.")
                tip = f"{spec.name}\n{ANALYSIS_LABELS.get(spec.kind, 'Figure')}\n{label}\n{detail}"
                item.setToolTip(0, tip)
                item.setToolTip(1, tip)
                item.setData(0, QtCore.Qt.ItemDataRole.AccessibleTextRole, f"{spec.name}. {item.text(1)}")
                self._items[spec.figure_id] = item
            if unsupported:
                group = QtWidgets.QTreeWidgetItem(self.tree, ["Unsupported"])
                group.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                group.setExpanded(expanded.get("Unsupported", True))
                for entry in unsupported:
                    item = QtWidgets.QTreeWidgetItem(group, [str(entry.get("name") or "Unsupported figure")])
                    item.setData(0, ID, entry["id"])
                    item.setData(0, CATEGORY, "unsupported")
                    item.setToolTip(0, item.text(0) + "\n" + str(entry.get("reason") or "This figure is preserved but cannot be opened by this build."))
                    self._items[entry["id"]] = item
            for artifact in saved:
                payload = artifact.payload
                category = "composition" if payload.get("page_id") == "composition" else "snapshot"
                item = QtWidgets.QTreeWidgetItem(groups[category], [str(payload.get("name") or "Saved figure")])
                item.setData(0, ID, artifact.artifact_id)
                item.setData(0, CATEGORY, category)
                status, reason = saved_status.get(artifact.artifact_id, ("Saved snapshot",
                    "Source lineage has not been checked; the saved image and data remain unchanged."))
                item.setText(1, str(status))
                item.setData(0, STATUS, str(status))
                tip = f"{item.text(0)}\n{payload.get('source_label', '')}\nSaved {payload.get('created_at', '')}\n{status}: {reason}"
                item.setToolTip(0, tip)
                item.setToolTip(1, tip)
                item.setData(0, QtCore.Qt.ItemDataRole.AccessibleTextRole, f"{item.text(0)}. {status}")
                self._items[artifact.artifact_id] = item
            chosen = self._items.get(current if activated_live else old_id if old_id in old_saved else current)
            if chosen is not None:
                self.tree.setCurrentItem(chosen, 0, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
            for key in (*selected, *old_saved):
                if key in self._items:
                    self._items[key].setSelected(True)
        self._sync_actions()

    def _current_changed(self, item, _old) -> None:
        self._sync_actions()
        self._activate(item)

    def _activate(self, item) -> None:
        if item is None or not item.data(0, ID):
            return
        if item.data(0, CATEGORY) == "unsupported":
            return
        signal = self.liveActivated if item.data(0, CATEGORY) == "live" else self.savedActivated
        signal.emit(item.data(0, ID))

    def _selection_changed(self) -> None:
        self._sync_actions()
        self.selectionChanged.emit()

    def duplicate_current(self) -> None:
        self._emit_current(self.duplicateRequested)

    def _emit_current(self, signal) -> None:
        key = self.editable_id()
        if key:
            signal.emit(key)

    def _sync_actions(self) -> None:
        live = self.current_id() is not None
        self.duplicate_button.setEnabled(self.editable_id() is not None)
        self.rename_button.setEnabled(self.editable_id() is not None)
        self.delete_button.setEnabled(bool(self.deletable_ids()))
        self.snapshot_button.setEnabled(live)
        self.compose_button.setEnabled(len(self.selected_ids("saved")) >= 2)
