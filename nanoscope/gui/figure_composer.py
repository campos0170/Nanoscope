"""Arrange project-owned frozen figures without opening their source analyses."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .figure_bundle import FigureSnapshot
from .figure_composition import DEFAULT_LABEL_STYLE


class FigureComposerDialog(QtWidgets.QDialog):
    """Explicit preview/commit boundary: the saved result is the approved preview."""

    def __init__(self, entries, parent=None, *, current_id=None, initial_layout=None, name="Multi-panel figure"):
        super().__init__(parent)
        self.setWindowTitle("Revise publication figure" if initial_layout is not None else "Compose publication figure")
        self.resize(1120, 780)
        self._entries = {entry.artifact_id: entry for entry in entries}
        self._snapshots = {}
        self._minimum_panels = 1 if initial_layout is not None else 2
        self.snapshot = None
        self.panel_ids = []
        self._pixmap = QtGui.QPixmap()
        root = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            ("Revise the embedded original panels, then save a new figure. No live sources are reopened. "
             if initial_layout is not None else "Choose 2–12 saved figures, set their order and labels, then preview. ") +
            "Original figures and live analyses remain unchanged. Shared axes are available "
            "only for compatible, fully captured Cartesian plots. Spans place panels in the first free grid cells. "
            "Complex panels keep their own axes and legends.")
        intro.setWordWrap(True)
        root.addWidget(intro)
        form = QtWidgets.QFormLayout()
        self._name = QtWidgets.QLineEdit(name)
        self._name.setMaxLength(200)
        form.addRow("Figure name:", self._name)
        settings = QtWidgets.QHBoxLayout()
        self._columns = QtWidgets.QSpinBox()
        self._columns.setRange(1, 4)
        self._columns.setValue(2)
        settings.addWidget(QtWidgets.QLabel("Columns:"))
        settings.addWidget(self._columns)
        self._share_x = QtWidgets.QCheckBox("Share X range")
        self._share_y = QtWidgets.QCheckBox("Share Y range")
        self._shared_legend = QtWidgets.QCheckBox("One shared legend")
        self._shared_legend.setToolTip(
            "Newly captured standard Cartesian legends only. Every panel must have identical labels, "
            "entry order and sample styles. Displays their key once below the panels; different or "
            "customized legends are never silently removed.")
        if initial_layout is not None:
            self._columns.setValue(initial_layout["columns"])
            self._share_x.setChecked(initial_layout["share_x"])
            self._share_y.setChecked(initial_layout["share_y"])
            self._shared_legend.setChecked(initial_layout.get("shared_legend", False))
        for check in (self._share_x, self._share_y):
            check.setToolTip(
                "Requires matching axis labels, units and linear/log mode. Uses the union "
                "of the saved view ranges on detached plot copies, without changing any numbers.")
            settings.addWidget(check)
        settings.addWidget(self._shared_legend)
        settings.addStretch()
        form.addRow(settings)
        style = (initial_layout or {}).get("label_style", DEFAULT_LABEL_STYLE)
        appearance = QtWidgets.QHBoxLayout()
        self._spacing = QtWidgets.QDoubleSpinBox()
        self._spacing.setRange(0, 200)
        self._spacing.setDecimals(3)
        self._spacing.setSuffix(" px")
        self._spacing.setValue((initial_layout or {}).get("spacing", 20))
        appearance.addWidget(QtWidgets.QLabel("Panel spacing:"))
        appearance.addWidget(self._spacing)
        self._label_font = QtWidgets.QFontComboBox()
        self._label_font.setCurrentFont(QtGui.QFont(style["family"]))
        self._label_font.setMaximumWidth(230)
        appearance.addWidget(QtWidgets.QLabel("Panel labels:"))
        appearance.addWidget(self._label_font)
        self._label_size = QtWidgets.QSpinBox()
        self._label_size.setRange(6, 72)
        self._label_size.setSuffix(" px")
        self._label_size.setValue(style["size_px"])
        self._label_size.setToolTip("Native composition pixels. Library export scales the complete frozen figure.")
        appearance.addWidget(self._label_size)
        self._label_bold = QtWidgets.QCheckBox("Bold")
        self._label_bold.setChecked(style["bold"])
        appearance.addWidget(self._label_bold)
        self._label_color = QtWidgets.QPushButton(style["color"])
        self._label_color.clicked.connect(self._choose_label_color)
        appearance.addWidget(self._label_color)
        appearance.addStretch()
        form.addRow(appearance)
        root.addLayout(form)
        split = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._table = QtWidgets.QTableWidget(0, 5)
        self._table.setObjectName("figureComposerPanels")
        self._table.setHorizontalHeaderLabels(["Use", "Label", "Saved figure", "Rows", "Cols"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(0, 42)
        self._table.setColumnWidth(1, 96)
        self._table.setColumnWidth(3, 52)
        self._table.setColumnWidth(4, 52)
        self._table.setMinimumWidth(480)
        ordered = list(entries)
        if current_id in self._entries:
            ordered.sort(key=lambda entry: entry.artifact_id != current_id)
        for row, entry in enumerate(ordered):
            self._table.insertRow(row)
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable |
                         QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(QtCore.Qt.CheckState.Checked if initial_layout is not None or row < 2 else QtCore.Qt.CheckState.Unchecked)
            use.setData(QtCore.Qt.ItemDataRole.UserRole, entry.artifact_id)
            label = QtWidgets.QTableWidgetItem(initial_layout["panel_labels"][row] if initial_layout is not None
                                               else chr(65 + row) if row < 26 else str(row + 1))
            title = QtWidgets.QTableWidgetItem(str(entry.payload.get("name", "Figure")))
            title.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            title.setToolTip(str(entry.payload.get("source_label", "")))
            for col, item in enumerate((use, label, title)):
                self._table.setItem(row, col, item)
            spans = (initial_layout or {}).get("panel_spans")
            for col, value in enumerate(spans[row] if spans is not None else (1, 1), 3):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setToolTip("Number of grid rows/columns occupied by this panel (1–4). "
                                "Column span must fit the grid. Panels are fitted without cropping.")
                self._table.setItem(row, col, item)
        left_layout.addWidget(self._table, 1)
        order = QtWidgets.QHBoxLayout()
        self._up = QtWidgets.QPushButton("Move up")
        self._down = QtWidgets.QPushButton("Move down")
        self._up.clicked.connect(lambda: self._move(-1))
        self._down.clicked.connect(lambda: self._move(1))
        order.addWidget(self._up)
        order.addWidget(self._down)
        order.addStretch()
        left_layout.addLayout(order)
        split.addWidget(left)
        self._preview = QtWidgets.QLabel("Preview the composition before saving.")
        self._preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(260, 220)
        self._preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                    QtWidgets.QSizePolicy.Policy.Expanding)
        self._preview.setStyleSheet("background:#24272b; border:1px solid #666;")
        split.addWidget(self._preview)
        split.setStretchFactor(1, 1)
        split.setSizes([520, 580])
        root.addWidget(split, 1)
        self._message = QtWidgets.QLabel("Preview captures layout only. No analyses will run.")
        self._message.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._message.setWordWrap(True)
        root.addWidget(self._message)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self._preview_btn = buttons.addButton("Update preview", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self._save_btn = buttons.addButton("Save as new figure" if initial_layout is not None else "Save to Figure Library",
                                           QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self._save_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self.update_preview)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)
        self._table.itemChanged.connect(self._invalidate)
        self._columns.valueChanged.connect(self._invalidate)
        self._share_x.toggled.connect(self._invalidate)
        self._share_y.toggled.connect(self._invalidate)
        self._shared_legend.toggled.connect(self._invalidate)
        self._spacing.valueChanged.connect(self._invalidate)
        self._label_font.currentFontChanged.connect(self._invalidate)
        self._label_size.valueChanged.connect(self._invalidate)
        self._label_bold.toggled.connect(self._invalidate)
        self._name.textChanged.connect(self._name_changed)

    def figure_name(self):
        return self._name.text().strip()

    def _choose_label_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._label_color.text()), self,
                                                "Panel-label color")
        if color.isValid():
            self._label_color.setText(color.name())
            self._invalidate()

    def _name_changed(self, *_args):
        self._save_btn.setEnabled(self.snapshot is not None and bool(self.figure_name()))

    def _invalidate(self, *_args):
        self.snapshot = None
        self.panel_ids = []
        self._save_btn.setEnabled(False)
        self._pixmap = QtGui.QPixmap()
        self._preview.setText("Layout changed. Update the preview before saving.")
        self._message.setText("Preview needs updating; the previous preview cannot be saved.")

    def _move(self, delta):
        row = self._table.currentRow()
        other = row + delta
        if row < 0 or not 0 <= other < self._table.rowCount():
            return
        with QtCore.QSignalBlocker(self._table):
            first = [self._table.takeItem(row, col) for col in range(self._table.columnCount())]
            second = [self._table.takeItem(other, col) for col in range(self._table.columnCount())]
            for col in range(self._table.columnCount()):
                self._table.setItem(row, col, second[col])
                self._table.setItem(other, col, first[col])
            self._table.selectRow(other)
        self._invalidate()

    def update_preview(self):
        from .figure_composition import compose_figures
        selected, labels, spans = [], [], []
        for row in range(self._table.rowCount()):
            use = self._table.item(row, 0)
            if use.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(use.data(QtCore.Qt.ItemDataRole.UserRole))
                labels.append(self._table.item(row, 1).text().strip())
                spans.append([self._table.item(row, col).text().strip() for col in (3, 4)])
        self._invalidate()
        try:
            if not self._minimum_panels <= len(selected) <= 12:
                raise ValueError(f"Choose between {self._minimum_panels} and 12 saved figures.")
            if any(len(label) > 32 for label in labels):
                raise ValueError("Panel labels must be at most 32 characters.")
            try:
                spans = [[int(value) for value in span] for span in spans]
            except ValueError as exc:
                raise ValueError("Rows and Cols must be whole numbers from 1–4.") from exc
            for key in selected:
                if key not in self._snapshots:
                    entry = self._entries[key]
                    self._snapshots[key] = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
            result = compose_figures(
                [self._snapshots[key] for key in selected], labels, self._columns.value(),
                share_x=self._share_x.isChecked(), share_y=self._share_y.isChecked(),
                panel_spans=spans, spacing=self._spacing.value(),
                label_style={"family": self._label_font.currentFont().family(),
                             "size_px": self._label_size.value(), "bold": self._label_bold.isChecked(),
                             "color": self._label_color.text()}, shared_legend=self._shared_legend.isChecked())
            self._pixmap.loadFromData(result.png_bytes, "PNG")
            if self._pixmap.isNull():
                raise ValueError("The composition preview could not be decoded.")
        except Exception as exc:
            self._message.setText(str(exc))
            return
        self.snapshot = result
        self.panel_ids = selected
        self._render_preview()
        self._message.setText(
            f"{len(selected)} panels · {round(result.width)} × {round(result.height)} native pixels. "
            "Save this preview as a new snapshot, then use Library export for mm/inches, DPI, SVG or PDF.")
        self._name_changed()

    def _render_preview(self):
        if not self._pixmap.isNull():
            self._preview.setPixmap(self._pixmap.scaled(
                self._preview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "_pixmap"):
            self._render_preview()

    def accept(self):
        if self.snapshot is not None and self.figure_name():
            super().accept()
