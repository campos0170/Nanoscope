"""One project-wide source appearance table over already available source metadata."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .source_appearance import SourceAppearanceEditor


class ProjectSourcesDialog(QtWidgets.QDialog):
    """Read a catalog; write only Session's exact-key presentation registries.

    The provider is MainWindow's ordinary cached-result discovery. We keep only
    scalar metadata, never a second copy of a source table, figure, or snapshot.
    """

    def __init__(self, session, source_provider, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Project sources")
        self.resize(760, 620)
        self._session = session
        self._source_provider = source_provider
        self._rows = {}
        self._refreshing = False
        column = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Names and colors for exact recording, segment, channel and detector identities. "
            "This table reads existing results; it does not run an analysis.")
        note.setWordWrap(True)
        column.addWidget(note)
        self.table = QtWidgets.QTreeWidget()
        self.table.setObjectName("projectSourcesTable")
        self.table.setAccessibleName("Project sources")
        self.table.setColumnCount(5)
        self.table.setHeaderLabels(("Source", "Channel", "Detector / kind", "Rows", "Color"))
        self.table.setRootIsDecorated(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for index in range(1, 5):
            self.table.header().setSectionResizeMode(index, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        column.addWidget(self.table, 1)
        self.editor = SourceAppearanceEditor(session=session, project_only=True)
        self.editor.source.currentIndexChanged.connect(self._select_editor_source)
        self.table.currentItemChanged.connect(self._select_table_source)
        # The table owns selection; retain the editor's exact-ID combo internally.
        self.editor.layout().setRowVisible(self.editor.source, False)
        column.addWidget(self.editor)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        refresh = buttons.addButton("Refresh source list", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        refresh.clicked.connect(self.refresh)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)
        session.sourceAppearanceChanged.connect(self.refresh)
        self.refresh()

    def refresh(self, *_args):
        """Update names and availability without a generation/refresh figure action."""
        if self._refreshing:
            return
        self._refreshing = True
        try:
            selected = self.editor.source.currentData()
            labels = self._session.source_labels.labels
            inks = self._session.colours.inks
            rows = {}
            for source in self._source_provider():
                key = str(source.key)
                rows[key] = {
                    "key": key, "label": str(labels.get(key) or source.label), "color": str(inks.get(key) or source.colour),
                    "channel": "—" if source.channel is None else str(source.channel),
                    "kind": str(source.detector_label or source.kind),
                    "count": str(source.n_rows), "available": True,
                }
            # Project defaults may outlive an open recording. Keep their exact IDs
            # visible and editable, without pretending their data are available.
            for key in sorted(set(labels) | set(inks)):
                if key not in rows:
                    rows[key] = {"key": key, "label": labels.get(key) or key,
                                 "color": inks.get(key, "#0072b2"), "channel": "—",
                                 "kind": "Saved default (unavailable)", "count": "—", "available": False}
            self._rows = rows
            state = {key: {**({"label": labels[key]} if key in labels else {}),
                           **({"color": inks[key]} if key in inks else {})} for key in rows}
            table_blocker = QtCore.QSignalBlocker(self.table)
            self.table.clear()
            for key, row in rows.items():
                item = QtWidgets.QTreeWidgetItem((row["label"], row["channel"], row["kind"], row["count"], row["color"]))
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, key)
                item.setToolTip(0, row["label"] + "\nExact source identity: " + key)
                item.setForeground(4, QtGui.QBrush(QtGui.QColor(row["color"])))
                self.table.addTopLevelItem(item)
                if key == selected:
                    self.table.setCurrentItem(item)
            del table_blocker
            self.editor.set_sources(rows.values())
            self.editor.set_state(state)
            self._select_editor_source()
        finally:
            self._refreshing = False

    def _select_table_source(self, item, _previous=None):
        if item is not None:
            index = self.editor.source.findData(item.data(0, QtCore.Qt.ItemDataRole.UserRole))
            if index >= 0:
                self.editor.source.setCurrentIndex(index)

    def _select_editor_source(self, *_args):
        key = self.editor.source.currentData()
        for index in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(index)
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == key:
                self.table.setCurrentItem(item)
                return
