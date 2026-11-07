"""Arrange project-owned frozen figures without opening their source analyses."""
from __future__ import annotations

from copy import deepcopy
import weakref

from PySide6 import QtCore, QtGui, QtWidgets

from ..plot import figure_fonts as ff
from ..core.figure_style import (FigureStyle, builtin_styles, style_binding,
    normalize_style_binding, logical_pixels_to_points)
from ..core.figure_composition_style import panel_style, panel_overrides_from_legacy
from .figure_bundle import FigureSnapshot
from .figure_composition import DEFAULT_LABEL_STYLE
from .font_picker import FontPicker


def _paint_swatch(button: QtWidgets.QPushButton, hex_color: str) -> None:
    """The chosen colour ON the button, with legible ink and its hex as the label: the
    Distributions colour swatch, so the two read the same."""
    color = QtGui.QColor(hex_color)
    luma = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    button.setText(hex_color)
    button.setStyleSheet(
        f"background-color: {hex_color}; color: {'#000000' if luma > 140 else '#ffffff'}; "
        "border: 1px solid #555; padding: 2px;")


class FigureComposerDialog(QtWidgets.QDialog):
    """Explicit preview/commit boundary: the saved result is the approved preview."""

    def __init__(self, entries, parent=None, *, current_id=None, initial_layout=None, name="Multi-panel figure",
                 available_styles=None, default_style=None):
        super().__init__(parent)
        owner = parent
        while owner is not None and getattr(owner, '_figure_workspace', None) is None:
            owner = owner.parent() if hasattr(owner, 'parent') else None
        workspace = getattr(owner, '_figure_workspace', None)
        self._style_owner = weakref.ref(owner) if owner is not None else None
        self._style_workspace = weakref.ref(workspace) if workspace is not None else None
        self._style_epoch = getattr(workspace, '_epoch', None)
        self._available_styles = available_styles
        self._label_binding = deepcopy((initial_layout or {}).get('style_binding'))
        if self._label_binding:
            self._label_binding = normalize_style_binding(self._label_binding)
        self._label_native = deepcopy((initial_layout or {}).get('label_style', DEFAULT_LABEL_STYLE))
        self._label_overrides = deepcopy((initial_layout or {}).get('label_overrides',
            panel_overrides_from_legacy(self._label_native)))
        self._writing_letters = False
        # Revision always restores the saved binding/pixels. Only a new composition
        # takes today's project default; an unreadable registry leaves native ink.
        if initial_layout is None:
            if default_style is None and workspace is not None and workspace.style_registry() is not None:
                default_style = workspace.default_style()
            if default_style is not None:
                self._label_binding = style_binding(default_style)
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
        style = panel_style(self._label_binding, self._label_overrides, legacy=self._label_native)
        style_row = QtWidgets.QHBoxLayout()
        self._style_button = QtWidgets.QToolButton()
        self._style_button.setText('Style')
        self._style_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._style_menu = QtWidgets.QMenu(self._style_button)
        self._style_menu.aboutToShow.connect(self._fill_style_menu)
        self._style_button.setMenu(self._style_menu)
        style_row.addWidget(self._style_button)
        self._style_name = QtWidgets.QLabel()
        style_row.addWidget(self._style_name)
        self._keep_label_edits = QtWidgets.QCheckBox('Keep manual label edits')
        self._keep_label_edits.setChecked(True)
        self._keep_label_edits.setToolTip('Applying a style retains explicit letter typography. Panel words and positions are always retained.')
        style_row.addWidget(self._keep_label_edits)
        style_row.addStretch()
        form.addRow(style_row)
        appearance = QtWidgets.QHBoxLayout()
        spacing = (initial_layout or {}).get("spacing", 20)
        self._spacing = QtWidgets.QDoubleSpinBox()
        self._spacing.setRange(0, 200)
        # Whole pixels: "20 px", not "20.000 px". A composition saved with a fractional
        # gap keeps its decimals when revised, so a revision never moves its panels.
        self._spacing.setDecimals(0 if float(spacing).is_integer() else 3)
        self._spacing.setSuffix(" px")
        self._spacing.setValue(spacing)
        appearance.addWidget(QtWidgets.QLabel("Panel spacing:"))
        appearance.addWidget(self._spacing)
        # The shared font picker. Its first row is the figure default, Arial; a revised
        # composition shows the family it was saved with (an extra row when the list lacks
        # it, kept as saved).
        self._label_font = FontPicker()
        self._label_font.set_value(style["family"])
        self._label_font.setToolTip(
            "Panel-label font. The first row is Arial, the figure default; "
            "Other font… lists every font installed on this computer except the system's "
            "private fonts and symbol fonts such as Wingdings.")
        self._label_font.setMaximumWidth(230)
        appearance.addWidget(QtWidgets.QLabel("Panel labels:"))
        appearance.addWidget(self._label_font)
        self._label_size = QtWidgets.QDoubleSpinBox()
        self._label_size.setDecimals(3)
        self._label_size.setRange(.001, 1000)
        self._label_size.setSuffix(" pt")
        self._label_size.setValue(logical_pixels_to_points(style["size_px"]))
        self._label_size.setToolTip("Physical points at the composition's 96-dpi coordinate basis. Export scales the complete frozen figure.")
        appearance.addWidget(self._label_size)
        self._label_bold = QtWidgets.QCheckBox("Bold")
        self._label_bold.setChecked(style["bold"])
        appearance.addWidget(self._label_bold)
        self._label_italic = QtWidgets.QCheckBox('Italic')
        self._label_italic.setChecked(style.get('italic', False))
        appearance.addWidget(self._label_italic)
        self._label_color_value = style["color"]
        self._label_color = QtWidgets.QPushButton()
        self._label_color.setToolTip("Panel-label colour")
        self._label_color.setFixedHeight(24)
        _paint_swatch(self._label_color, self._label_color_value)
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
        self._label_font.changed.connect(self._font_edited)
        self._label_size.valueChanged.connect(self._size_edited)
        self._label_bold.toggled.connect(self._bold_edited)
        self._label_italic.toggled.connect(self._italic_edited)
        self._name.textChanged.connect(self._name_changed)
        self._refresh_style_name()

    def _style_context_current(self):
        if self._style_workspace is None:
            return True
        workspace = self._style_workspace()
        owner = self._style_owner() if self._style_owner is not None else None
        return (workspace is not None and workspace._epoch == self._style_epoch
                and owner is not None and getattr(owner, '_figure_workspace', None) is workspace)

    @QtCore.Slot()
    def _fill_style_menu(self):
        self._style_menu.clear()
        if not self._style_context_current():
            action = self._style_menu.addAction('Project changed; reopen this composer')
            action.setEnabled(False)
            return
        offered = list(builtin_styles())
        provider = self._available_styles
        workspace = self._style_workspace() if self._style_workspace is not None else None
        if provider is None and workspace is not None:
            provider = workspace.available_styles
        if callable(provider):
            offered.extend(provider())
        seen = set()
        for style in offered:
            identity = (style.style_id, style.revision)
            if identity in seen:
                continue
            seen.add(identity)
            self._style_menu.addAction(style.name, lambda checked=False, value=style: self.apply_style(value))
        from .computer_styles import styles
        self._style_menu.addSection('On this computer')
        try:
            local = styles()
        except ValueError as exc:
            self._style_menu.addAction(str(exc)).setEnabled(False)
        else:
            if not local:
                self._style_menu.addAction('None saved on this computer').setEnabled(False)
            for style in local:
                self._style_menu.addAction(style.name, lambda checked=False, value=style: self.apply_style(value))

    def _refresh_style_name(self):
        name = self._label_binding['snapshot']['name'] if self._label_binding else 'Original label appearance'
        self._style_name.setText(name + (' · manual edits' if self._label_overrides else ''))

    def apply_style(self, style):
        if not self._style_context_current():
            self._message.setText('The project changed. Reopen the composer before applying a style.')
            return False
        # A new style changes only the composition's own lettering. Source panels
        # and their exact embedded resources have no path through this operation.
        self._label_native = panel_style(self._label_binding, self._label_overrides,
                                         legacy=self._label_native)
        self._label_binding = style_binding(style)
        if not self._keep_label_edits.isChecked():
            self._label_overrides = {}
        self._write_letter_controls()
        self._invalidate()
        return True

    def _write_letter_controls(self):
        style = panel_style(self._label_binding, self._label_overrides, legacy=self._label_native)
        self._writing_letters = True
        try:
            self._label_font.set_value(style['family'])
            self._label_size.setValue(logical_pixels_to_points(style['size_px']))
            self._label_bold.setChecked(style['bold'])
            self._label_italic.setChecked(style.get('italic', False))
            self._label_color_value = style['color']
            _paint_swatch(self._label_color, self._label_color_value)
        finally:
            self._writing_letters = False
        self._refresh_style_name()

    def _letter_edited(self, key, value):
        if self._writing_letters:
            return
        if not self._label_binding:
            # Editing an old composition starts a local pinned style without
            # inventing journal defaults for its other explicit native typography.
            self._label_binding = style_binding(FigureStyle('composition.custom', 1,
                'Custom lettering', provenance={'inherit_native_defaults': True}))
        self._label_overrides[key] = value
        self._refresh_style_name()
        self._invalidate()

    @QtCore.Slot()
    def _font_edited(self):
        self._letter_edited('family', self.label_family())

    @QtCore.Slot(float)
    def _size_edited(self, value):
        self._letter_edited('size_pt', value)

    @QtCore.Slot(bool)
    def _bold_edited(self, value):
        self._letter_edited('bold', value)

    @QtCore.Slot(bool)
    def _italic_edited(self, value):
        self._letter_edited('italic', value)

    def figure_name(self):
        return self._name.text().strip()

    def label_family(self):
        """The family panel labels are drawn in: the named row, or Arial."""
        return ff.effective_family(self._label_font.value())

    def _choose_label_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._label_color_value), self,
                                                "Panel-label color")
        if color.isValid():
            self._label_color_value = color.name()
            _paint_swatch(self._label_color, self._label_color_value)
            self._letter_edited('color', self._label_color_value)

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
                label_style=self._label_native,
                style_binding=self._label_binding, label_overrides=self._label_overrides,
                shared_legend=self._shared_legend.isChecked())
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
