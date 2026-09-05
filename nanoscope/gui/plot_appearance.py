"""Shared, opt-in styling of existing plots, with no analysis callbacks.

Empty state preserves the original appearance. Clearing overrides restores the
original graphical properties. Titles, labels, units, arrays and fits stay intact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class PlotAppearance:
    font_family: str | None = None
    title_size: float | None = None
    axis_size: float | None = None
    tick_size: float | None = None
    grid: bool | None = None
    background: str | None = None
    data_color: str | None = None
    fit_color: str | None = None
    line_width: float | None = None
    marker_size: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe overrides only; an empty mapping means the original style."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "PlotAppearance":
        """Ignore unknown/invalid saved fields without changing analysis defaults."""
        values = values if isinstance(values, Mapping) else {}
        valid: dict[str, Any] = {}
        family = values.get("font_family")
        if isinstance(family, str) and family.strip() and not any(c in family for c in ";{}<>"):
            valid["font_family"] = family.strip()[:200]
        for key, low, high in (("title_size", 6, 48), ("axis_size", 6, 36),
                               ("tick_size", 6, 30), ("line_width", .25, 10),
                               ("marker_size", 1, 30)):
            value = values.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value) and low <= value <= high:
                    valid[key] = float(value)
        if isinstance(values.get("grid"), bool):
            valid["grid"] = values["grid"]
        if values.get("background") in ("dark", "light"):
            valid["background"] = values["background"]
        for key in ("data_color", "fit_color"):
            color = values.get(key)
            if isinstance(color, str) and QtGui.QColor(color).isValid():
                valid[key] = QtGui.QColor(color).name()
        return cls(**valid)


class PlotAppearanceWidget(QtWidgets.QWidget):
    """Compact common controls. ``set_state`` is silent; user edits emit once."""

    changed = QtCore.Signal(object)

    def __init__(self, state: PlotAppearance | None = None, parent=None, *,
                 include_colors=True, compact=False, include_markers=True):
        super().__init__(parent)
        self._state = state or PlotAppearance()
        self._loading = False
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(4)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(["Choose appearance preset…", "Publication · light", "Presentation · dark"])
        self.preset.setToolTip("Appearance only: never changes data, fitting, axis scales, or analysis settings.")
        self.preset.activated.connect(self._apply_preset)
        layout.addWidget(self.preset, 0, 0, 1, 3)
        self.reset_button = QtWidgets.QPushButton("Reset appearance")
        self.reset_button.setToolTip("Remove appearance overrides; keep all data and analytical settings.")
        self.reset_button.clicked.connect(self._reset)
        layout.addWidget(self.reset_button, 0, 3)
        self.font_family = QtWidgets.QComboBox()
        self.font_family.setEditable(True)
        self.font_family.addItems(["Default", *QtGui.QFontDatabase.families()])
        self.font_family.setMaxVisibleItems(12)
        self.font_family.setMinimumContentsLength(12)
        self.font_family.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.font_family.currentTextChanged.connect(
            lambda text: self._edit("font_family", None if text == "Default" else text))
        self._add_control(layout, 1, 0, "Font", self.font_family)
        self.background = QtWidgets.QComboBox()
        for label, value in (("Original", None), ("Light", "light"), ("Dark", "dark")):
            self.background.addItem(label, value)
        self.background.currentIndexChanged.connect(
            lambda: self._edit("background", self.background.currentData()))
        self._add_control(layout, 1, 2, "Background", self.background)
        for key, label, row, col, maximum in (
                ("title_size", "Title (pt)", 2, 0, 48),
                ("axis_size", "Axis labels (pt)", 2, 2, 36),
                ("tick_size", "Ticks (pt)", 3, 0, 30),
                ("line_width", "Line width", 4, 0, 10),
                ("marker_size", "Marker size", 4, 2, 30)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0, maximum)
            spin.setDecimals(1)
            spin.setSingleStep(.5 if key == "line_width" else 1)
            spin.setSpecialValueText("Default")
            spin.valueChanged.connect(lambda value, field=key: self._edit(field, value or None))
            setattr(self, key, spin)
            self._add_control(layout, row, col, label, spin)
        self.grid = QtWidgets.QComboBox()
        for label, value in (("Original", None), ("Show", True), ("Hide", False)):
            self.grid.addItem(label, value)
        self.grid.currentIndexChanged.connect(lambda: self._edit("grid", self.grid.currentData()))
        self._add_control(layout, 3, 2, "Grid", self.grid)
        for key, label, col in (("data_color", "Data color", 0), ("fit_color", "Fit color", 2)):
            button = QtWidgets.QPushButton("Default…")
            button.clicked.connect(lambda checked=False, field=key: self._choose_color(field))
            setattr(self, key, button)
            self._add_control(layout, 5, col, label, button)
        note = QtWidgets.QLabel("Appearance only. Titles, units, provenance, data and fits are preserved.")
        note.setWordWrap(True)
        layout.addWidget(note, 6, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        if not include_colors:
            for row, column in ((1, 2), (1, 3), (5, 0), (5, 1), (5, 2), (5, 3)):
                layout.itemAtPosition(row, column).widget().hide()
            # Presets include palettes; the host already owns its palette controls.
            self.preset.hide()
        if not include_markers:
            layout.itemAtPosition(4, 2).widget().hide()
            self.marker_size.hide()
        if compact:
            pairs = [(layout.itemAtPosition(row, col).widget(),
                      layout.itemAtPosition(row, col + 1).widget())
                     for row, col in ((1, 0), (1, 2), (2, 0), (2, 2), (3, 0),
                                      (3, 2), (4, 0), (4, 2), (5, 0), (5, 2))]
            while layout.count():
                layout.takeAt(0)
            row = 0
            if include_colors:
                layout.addWidget(self.preset, row, 0, 1, 2)
                row += 1
            for label, control in pairs:
                if control.isHidden():
                    continue
                layout.addWidget(label, row, 0)
                layout.addWidget(control, row, 1)
                row += 1
            layout.addWidget(self.reset_button, row, 0, 1, 2)
            layout.addWidget(note, row + 1, 0, 1, 2)
            layout.setColumnStretch(3, 0)
        self.set_state(self._state)

    @staticmethod
    def _add_control(layout, row, column, label, control):
        label_widget = QtWidgets.QLabel(label)
        label_widget.setBuddy(control)
        layout.addWidget(label_widget, row, column)
        layout.addWidget(control, row, column + 1)

    def state(self) -> PlotAppearance:
        return self._state

    def set_state(self, state: PlotAppearance | Mapping[str, Any]):
        self._state = state if isinstance(state, PlotAppearance) else PlotAppearance.from_mapping(state)
        self._loading = True
        try:
            self.font_family.setCurrentText(self._state.font_family or "Default")
            for key in ("title_size", "axis_size", "tick_size", "line_width", "marker_size"):
                getattr(self, key).setValue(getattr(self._state, key) or 0)
            for key in ("background", "grid"):
                combo = getattr(self, key)
                combo.setCurrentIndex(combo.findData(getattr(self._state, key)))
            for key in ("data_color", "fit_color"):
                button = getattr(self, key)
                value = getattr(self._state, key)
                button.setText(f"{value}…" if value else "Default…")
                button.setStyleSheet(f"border-bottom: 3px solid {value};" if value else "")
        finally:
            self._loading = False

    def _edit(self, key, value):
        if self._loading:
            return
        updated = PlotAppearance.from_mapping({**self._state.to_dict(), key: value})
        if updated != self._state:
            self._state = updated
            self.changed.emit(updated)

    def _choose_color(self, field):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(getattr(self._state, field) or "#0072B2"), self,
            "Choose data color" if field == "data_color" else "Choose fit color")
        if color.isValid():
            self._edit(field, color.name())
            self.set_state(self._state)

    def _reset(self):
        if self._state != PlotAppearance():
            self.set_state(PlotAppearance())
            self.changed.emit(self._state)

    def _apply_preset(self, index):
        if index not in (1, 2):
            return
        self.set_state(PlotAppearance(
            font_family="Arial", title_size=14, axis_size=12, tick_size=10,
            grid=False if index == 1 else True,
            background="light" if index == 1 else "dark", data_color="#0072b2",
            fit_color="#d55e00", line_width=2, marker_size=8 if index == 1 else 11))
        self.preset.setCurrentIndex(0)
        self.changed.emit(self._state)


def _item_defaults(item):
    defaults = getattr(item, "_nanoscope_appearance_defaults", None)
    if defaults is None:
        defaults = {key: item.opts.get(key) for key in
                    ("pen", "brush", "size", "symbolPen", "symbolBrush", "symbolSize")}
        item._nanoscope_appearance_defaults = defaults
    return defaults


def _style_data_item(item, state, color):
    if not isinstance(item, (pg.PlotDataItem, pg.ScatterPlotItem)):
        return
    defaults = _item_defaults(item)
    pen = defaults["pen"]
    if isinstance(item, pg.PlotDataItem):
        if pen is not None:
            pen = pg.mkPen(pen)
            if color:
                pen.setColor(QtGui.QColor(color))
            if state.line_width is not None:
                pen.setWidthF(state.line_width)
        item.setPen(pen)
        brush = defaults["symbolBrush"]
        if color:
            tint = QtGui.QColor(color)
            if brush is not None:
                tint.setAlpha(pg.mkBrush(brush).color().alpha())
            brush = pg.mkBrush(tint)
        item.setSymbolBrush(brush)
        item.setSymbolSize(state.marker_size if state.marker_size is not None else defaults["symbolSize"])
    else:
        brush = defaults["brush"]
        if color:
            tint = QtGui.QColor(color)
            if brush is not None:
                tint.setAlpha(pg.mkBrush(brush).color().alpha())
            brush = pg.mkBrush(tint)
        item.setBrush(brush)
        item.setSize(state.marker_size if state.marker_size is not None else defaults["size"])


def apply_plot_appearance(plot: pg.PlotItem, state: PlotAppearance, *,
                          plot_widget=None, data_items: Sequence = (),
                          fit_items: Sequence = ()) -> None:
    """Style existing items, without ``clear``, ``setData``, fitting, or autoRange.

    Original graphical properties are recorded on first opt-in so Reset restores
    them. Data/fit items replaced by explicit analysis refresh acquire their own
    original styles. Existing manual zoom is retained.
    """
    defaults = getattr(plot, "_nanoscope_appearance_defaults", None)
    if defaults is None and not state.to_dict():
        return
    if defaults is None:
        defaults = {
            "title_opts": dict(plot.titleLabel.opts),
            "axes": {side: {"label": dict(plot.getAxis(side).labelStyle),
                            "font": plot.getAxis(side).style.get("tickFont"),
                            "pen": plot.getAxis(side).pen(),
                            "text_pen": plot.getAxis(side).textPen(),
                            "grid": plot.getAxis(side).grid}
                     for side in ("bottom", "left", "top", "right")},
            "background": plot_widget.backgroundBrush() if plot_widget is not None else None,
        }
        plot._nanoscope_appearance_defaults = defaults
    view = plot.getViewBox()
    ranges = [list(axis) for axis in view.viewRange()]
    auto = list(view.autoRangeEnabled())
    view.disableAutoRange()
    try:
        foreground = {"light": "#20242b", "dark": "#e6e9ef"}.get(state.background)
        if plot_widget is not None:
            background = {"light": "#ffffff", "dark": "#151a21"}.get(state.background)
            plot_widget.setBackground(background if background else defaults["background"])
        for side, original in defaults["axes"].items():
            axis = plot.getAxis(side)
            label_style = dict(original["label"])
            if state.font_family:
                label_style["font-family"] = state.font_family
            if state.axis_size is not None:
                label_style["font-size"] = f"{state.axis_size:g}pt"
            if foreground:
                label_style["color"] = foreground
            axis.labelStyle = label_style
            axis._updateLabel()  # preserve text, units and automatic SI-prefix state
            font = original["font"]
            if state.font_family or state.tick_size is not None:
                font = QtGui.QFont(font or QtWidgets.QApplication.font())
                if state.font_family:
                    font.setFamily(state.font_family)
                if state.tick_size is not None:
                    font.setPointSizeF(state.tick_size)
            axis.setStyle(tickFont=font)
            axis.setPen(foreground or original["pen"])
            axis.setTextPen(foreground or original["text_pen"])
            axis.setGrid(original["grid"] if state.grid is None else
                         (0.15 * 255 if state.grid and side in ("bottom", "left") else False))
        title = plot.titleLabel
        last_title_color = defaults.get("applied_title_color", defaults["title_opts"].get("color"))
        if title.opts.get("color") != last_title_color:
            # A fresh scientific render can deliberately recolor its fit/warning
            # title. Keep that original color available for palette reset.
            defaults["title_opts"]["color"] = title.opts.get("color")
        title.opts = dict(defaults["title_opts"])
        if state.font_family:
            title.opts["family"] = state.font_family
        if state.title_size is not None:
            title.opts["size"] = f"{state.title_size:g}pt"
        if foreground:
            title.opts["color"] = foreground
        title.setText(title.text)
        defaults["applied_title_color"] = title.opts.get("color")
        if title.text:
            height = max(30.0, float(title.item.boundingRect().height()) + 6)
            title.setMaximumHeight(height)
            plot.layout.setRowFixedHeight(0, height)
        for item in data_items:
            _style_data_item(item, state, state.data_color)
        for item in fit_items:
            _style_data_item(item, state, state.fit_color)
    finally:
        view.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
        for axis, enabled in enumerate(auto):
            if enabled:
                view.enableAutoRange(axis=axis, enable=enabled)
