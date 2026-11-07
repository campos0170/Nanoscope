"""Edit a project style's look as a new immutable revision."""
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.figure_spec import thaw_json
from .font_picker import FontPicker


class ProjectStyleEditor(QtWidgets.QDialog):
    def __init__(self, style, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit style — " + style.name)
        self._original = thaw_json(style.look)
        self._original_columns = thaw_json(style.columns)
        self.resize(560, 760)
        column = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("Saves a new revision. Existing figures keep their look until you apply it.")
        note.setWordWrap(True)
        column.addWidget(note)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(content)
        scroll.setWidget(content)
        column.addWidget(scroll, 1)
        self.family = FontPicker()
        self.family.set_value(self._original.get("font_family"))
        form.addRow("Font", self.family)
        self.sizes = {}
        for key, label in (("body_size_pt", "Body"), ("title_size_pt", "Title"),
                           ("axis_size_pt", "Axis labels"), ("tick_size_pt", "Tick numbers"),
                           ("legend_size_pt", "Legend"), ("caption_size_pt", "Caption"),
                           ("fit_label_size_pt", "Fit label"), ("panel_size_pt", "Panel letter"),
                           ("line_width_pt", "Line width"),
                           ("marker_size_pt", "Marker size")):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(0, 1000)
            spin.setSingleStep(.25)
            spin.setSpecialValueText("Inherit")
            spin.setSuffix(" pt")
            spin.setValue(self._original.get(key, 0))
            self.sizes[key] = spin
            form.addRow(label, spin)
        self.background = QtWidgets.QComboBox()
        for label, value in (("Inherit", None), ("Light", "light"), ("Dark", "dark")):
            self.background.addItem(label, value)
        self.background.setCurrentIndex(max(0, self.background.findData(self._original.get("background"))))
        form.addRow("Background", self.background)
        self.grid = QtWidgets.QComboBox()
        for label, value in (("Inherit", None), ("Show", True), ("Hide", False)):
            self.grid.addItem(label, value)
        self.grid.setCurrentIndex(max(0, self.grid.findData(self._original.get("grid"))))
        form.addRow("Grid", self.grid)
        self.frame = QtWidgets.QComboBox()
        for label, value in (("Native frame", None), ("Closed box", True), ("Open top and right", False)):
            self.frame.addItem(label, value)
        self.frame.setCurrentIndex(max(0, self.frame.findData(self._original.get("frame"))))
        form.addRow("Frame", self.frame)
        self.palette = QtWidgets.QLineEdit(', '.join(self._original.get('palette', ())))
        self.palette.setPlaceholderText("Inherit — or comma-separated colors, e.g. #0072b2, #d55e00")
        self.palette.setAccessibleName("Palette colors")
        form.addRow("Palette", self.palette)
        palette_note = QtWidgets.QLabel(
            "Palette defaults apply to series without identity colors. Existing source and class colors "
            "stay fixed; use Apply palette in Format → Series to change this figure deliberately.")
        palette_note.setWordWrap(True)
        form.addRow(palette_note)
        self._shown_palette = self.palette.text()
        self.colors = {}
        for key, label in (("data_color", "Data color"), ("fit_color", "Fit color")):
            field = QtWidgets.QLineEdit(self._original.get(key, ""))
            field.setPlaceholderText("Inherit")
            self.colors[key] = field
            form.addRow(label, field)
        self.canvas_sizes = {}
        from ..core.figure_style import CANVAS_MAX_MM
        for key, label in (("canvas_width_mm", "Default canvas width"),
                           ("canvas_height_mm", "Default canvas height")):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setObjectName("styleCanvas_" + key)
            spin.setDecimals(4)
            spin.setRange(0, CANVAS_MAX_MM)
            spin.setSingleStep(1)
            spin.setSpecialValueText("Native / keep aspect")
            spin.setSuffix(" mm")
            spin.setValue(self._original.get(key, 0))
            spin.setToolTip("Export default only. With one dimension set, keep the captured aspect ratio. With neither, keep native size.")
            self.canvas_sizes[key] = spin
            form.addRow(label, spin)
        self._shown_canvas = {key: spin.value() for key, spin in self.canvas_sizes.items()}
        canvas_note = QtWidgets.QLabel("These physical dimensions seed export choices. They do not resize or recompute the analytical canvas. Set both to choose an explicit export aspect ratio.")
        canvas_note.setWordWrap(True)
        form.addRow(canvas_note)
        self.column_widths = {}
        for key, label in (("single_mm", "Single-column width"),
                           ("double_mm", "Double-column width")):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setObjectName("styleColumn_" + key)
            spin.setDecimals(4)
            spin.setRange(0, 10000)
            spin.setSingleStep(1)
            spin.setSpecialValueText("Not set")
            spin.setSuffix(" mm")
            spin.setValue(self._original_columns.get(key, 0))
            spin.setToolTip("Export width for this style. Not set removes this named preset.")
            self.column_widths[key] = spin
            form.addRow(label, spin)
        self.column_note = QtWidgets.QLabel(
            "Widths are reference values, not verified publication requirements. "
            "Export keeps the captured aspect ratio; resizing scales artwork and fonts. "
            "Native size keeps the captured figure dimensions.")
        self.column_note.setWordWrap(True)
        form.addRow(self.column_note)
        self._shown_columns = {key: spin.value() for key, spin in self.column_widths.items()}
        self._shown = {key: spin.value() for key, spin in self.sizes.items()}
        self._shown.update(font_family=self.family.value(), background=self.background.currentData(),
                           grid=self.grid.currentData(), frame=self.frame.currentData())
        self._shown.update({key: field.text() for key, field in self.colors.items()})
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        column.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)
        self._build_preview(style, column)

    def _build_preview(self, style, column):
        import pyqtgraph as pg
        from .figure_series import tag_pg_series
        self._preview_style_base = style
        group = QtWidgets.QGroupBox('Style preview — synthetic points')
        layout = QtWidgets.QVBoxLayout(group)
        self.preview = pg.PlotWidget()
        self.preview.setAccessibleName('Synthetic style preview')
        self.preview.setFixedHeight(210)
        self.preview.setMinimumWidth(0)
        plot = self.preview.getPlotItem()
        plot.setMenuEnabled(False); plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setTitle('Figure title')
        plot.setLabel('bottom', 'Time (ms)'); plot.setLabel('left', 'Signal (pA)')
        plot.addLegend(offset=(-5, 5))
        xs = [0., 1., 2., 3., 4.]
        values = ([1., 1.4, 1.2, 1.8, 1.6], [1.6, 2., 1.8, 2.4, 2.2], [2.2, 2.6, 2.4, 3., 2.8])
        self._preview_curves = []
        for index, ys in enumerate(values):
            curve = plot.plot(xs, ys, pen=pg.mkPen('#0072b2', width=1),
                              symbol='o', symbolSize=4, symbolBrush='#0072b2',
                              name=f'Series {index + 1}')
            tag_pg_series(curve, view='style-preview', identity=str(index), label=f'Series {index + 1}')
            self._preview_curves.append(curve)
        self._preview_fit = plot.plot(xs, [.8, 1.1, 1.4, 1.7, 2.],
                                      pen=pg.mkPen('#d55e00', width=1), name='Illustrative fit')
        self._preview_labels = {}
        for key, text, x, y in (('annotation.panel.preview', 'A', .1, 4.),
                                ('annotation.fit', 'Fit label', 1.7, 3.8),
                                ('annotation.caption', 'Caption', .1, .35),
                                ('body', 'Body', 2., .35)):
            item = pg.TextItem(text, anchor=(0, 0)); item.setPos(x, y)
            plot.addItem(item, ignoreBounds=True)
            self._preview_labels[key] = item
        plot.setRange(xRange=(-.2, 4.5), yRange=(-.25, 4.25), padding=0)
        layout.addWidget(self.preview)
        self.preview_note = QtWidgets.QLabel()
        self.preview_note.setWordWrap(True)
        layout.addWidget(self.preview_note)
        # A bounded view stays outside the scrolling control form.
        column.insertWidget(column.count() - 2, group)
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True); self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._update_preview)
        self.family.changed.connect(self._queue_preview)
        for spin in (*self.sizes.values(), *self.canvas_sizes.values(), *self.column_widths.values()):
            spin.valueChanged.connect(self._queue_preview)
        for combo in (self.background, self.grid, self.frame):
            combo.currentIndexChanged.connect(self._queue_preview)
        for edit in (self.palette, *self.colors.values()):
            edit.textChanged.connect(self._queue_preview)
        self._update_preview()

    @QtCore.Slot()
    def _queue_preview(self, *_args):
        self._preview_timer.start()

    @QtCore.Slot()
    def _update_preview(self):
        from dataclasses import replace
        from ..core.figure_style import resolve_bound_text, resolve_series_color
        from .figure_style_binding import bind_native_style
        from .font_picker import figure_qfont
        from .plot_appearance import PlotAppearance, apply_plot_appearance
        from .figure_export_style import canvas_dimensions, column_presets
        try:
            style = replace(self._preview_style_base, look=self.look(), columns=self.columns())
            native = bind_native_style({}, style)
            state = PlotAppearance.from_mapping(native)
            # These synthetic curves have no recording/category identity. Show
            # exactly the palette defaults the shared resolver assigns to them.
            series = {curve._nanoscope_series_descriptor.series_id: {
                'color': resolve_series_color(state.style_binding, index=index, native_color='#0072b2')}
                for index, curve in enumerate(self._preview_curves)}
            state = replace(state, series_styles=series)
            apply_plot_appearance(self.preview.getPlotItem(), state, plot_widget=self.preview,
                                  data_items=self._preview_curves, fit_items=(self._preview_fit,))
            foreground = '#eeeeee' if state.background == 'dark' else '#202226'
            for key, item in self._preview_labels.items():
                values, _origins = resolve_bound_text(state.style_binding, key)
                font = figure_qfont(values.get('family'))
                if values.get('size_pt') is not None: font.setPointSizeF(values['size_pt'])
                font.setBold(bool(values.get('bold', False))); font.setItalic(bool(values.get('italic', False)))
                item.setFont(font); item.setColor(values.get('color', foreground))
            dimensions = canvas_dimensions(style, (100., 65.))
            sizes = ('Native export dimensions' if dimensions is None else
                     f'Export default {dimensions[0]:g} × {dimensions[1]:g} mm')
            presets, _note = column_presets(style)
            widths = ' · '.join(f'{width:g} mm' for _label, width in presets)
            self.preview_note.setText('Synthetic illustration; screen size is not print scale. ' + sizes +
                                      (f'. Column references: {widths}.' if widths else '.'))
            self._preview_state = state
            self.error.clear()
        except (ValueError, TypeError) as exc:
            self.error.setText('Preview kept at the last valid style: ' + str(exc))

    def look(self):
        values = dict(self._original)
        family = self.family.value()
        if family != self._shown["font_family"]:
            if family:
                values["font_family"] = family
            else:
                values.pop("font_family", None)
        for key, spin in self.sizes.items():
            if spin.value() == self._shown[key]:
                continue
            if spin.value():
                values[key] = spin.value()
            else:
                values.pop(key, None)
        for key, combo in (("background", self.background), ("grid", self.grid), ("frame", self.frame)):
            if combo.currentData() == self._shown[key]:
                continue
            if combo.currentData() is None:
                values.pop(key, None)
            else:
                values[key] = combo.currentData()
        for key, field in self.colors.items():
            if field.text() == self._shown[key]:
                continue
            value = field.text().strip()
            if value and not QtGui.QColor(value).isValid():
                raise ValueError("Enter a valid color name or hexadecimal color.")
            if value:
                values[key] = QtGui.QColor(value).name()
            else:
                values.pop(key, None)
        if self.palette.text() != self._shown_palette:
            text = self.palette.text().strip()
            if not text:
                values.pop('palette', None)
            else:
                colors = [QtGui.QColor(part.strip()) for part in text.split(',')]
                if not 1 <= len(colors) <= 256 or any(not color.isValid() for color in colors):
                    raise ValueError("Enter 1–256 valid colors separated by commas.")
                values['palette'] = [color.name() for color in colors]
        from ..core.figure_style import CANVAS_MIN_MM, CANVAS_MAX_MM
        for key, spin in self.canvas_sizes.items():
            if spin.value() == self._shown_canvas[key]:
                continue
            value = spin.value()
            if value:
                if not CANVAS_MIN_MM <= value <= CANVAS_MAX_MM:
                    raise ValueError(f"Canvas dimensions must be between {CANVAS_MIN_MM:g} and {CANVAS_MAX_MM:g} mm.")
                values[key] = value
            else:
                values.pop(key, None)
        return values

    def columns(self):
        """Preserve every untouched stored value; deliberate edits define new widths."""
        values = dict(self._original_columns)
        changed = False
        for key, spin in self.column_widths.items():
            if spin.value() == self._shown_columns[key]:
                continue
            changed = True
            if spin.value():
                values[key] = spin.value()
            else:
                values.pop(key, None)
        if changed:
            values["basis"] = "user-defined"
            values["label"] = "User-defined columns"
        return values

    def accept(self):
        try:
            self.look()
            self.columns()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()
