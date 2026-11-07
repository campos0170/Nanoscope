"""The Format tab of the Analysis Explorer's inspector: the figure's elements, one by one.

An element list ("Figure", "Title", "X axis", "Legend", "Series", "Fit label" ...) over ONE
properties panel that follows the selection. A click on a text of the figure selects it, a pick
in the list selects it, and the same panel shows it either way. Built on the override model that
already exists (``PlotAppearance``: an element's own style, then the figure-wide field, then
the window's default); nothing is stored that was not stored before, and the project stays at
format 12.

* **The page's ``PlotAppearanceWidget`` stays the state owner and the single appearance
  history.** Its labelled rows are MOVED (not rebuilt) into the panels, with their labels
  (``control._nanoscope_row_label``); the widget itself becomes a hidden husk that still owns
  ``_state``, ``_history``, ``changed`` and every attribute name. New code is a registry of
  where each row goes, the two ``..Fields`` widgets of ``figure_text`` (the popovers' own),
  and the panels that hold them.
* **Selection has no event filter.** ``FigureTextController`` says ``claimed(key)`` from the
  click wrappers it already installs and ``cleared()`` from the scene's ``sigMouseClicked``
  (see ``figure_text``); this module only listens. The outline and the mini toolbar
  (``format_toolbar``) are children of the view's viewport.
* **Lazy construction.** The pane is a skeleton at adoption; the list and the footer are
  built at the first show, each panel at the first time it is shown, once, never per click,
  selection or redraw (a panel is RETARGETED: ``show_ref`` updates it in place). The list
  uses ``currentItemChanged`` only, there is no ``itemChanged``, and a rebuild is deferred and
  happens only when the set of rows changed.
* **A write starts from the element's current style.** Every writer (a panel row, the mini
  toolbar, Paste format) changes one field of the style the target has now
  (``FigureTextController.set_style_property``), because ``PlotAppearance.with_text_style``
  puts a style's text into the title, a label or the fit label's template.

A page still hosted by the Foundation's fallback adapter (its Format widgets are the whole
Appearance section) keeps the Foundation's single column; its own adapter turns the panels on.

This module never imports a page, and it installs no event filter.
"""
from __future__ import annotations

import functools
import weakref
from dataclasses import dataclass, replace
from typing import NamedTuple

from PySide6 import QtCore, QtGui, QtWidgets

from .figure_text import (HIDDEN_TEXT, AxisFields, AxisState, SetHereDot, TextStyle,
                          TextStyleFields, TextTarget, _alive, shown_text)
from .plot_appearance import PlotAppearance
from .plot_workflow import FormatSpec
from .widget_transfer import prepare_widget_move

__all__ = ["ElementList", "ElementRef", "FormatClipboard", "FormatPane", "SelectionHub",
           "SetHereDot", "resolve"]

_UNLIMITED = 16777215       # QWIDGETSIZE_MAX
_ROLE = QtCore.Qt.ItemDataRole.UserRole

#: The panel that shows a UI-only key. (Text keys are told apart in :func:`_kind_of`.)
FIGURE = "figure"
LEGEND = "legend"
COLOUR_MAP = "colour_map"
NO_CONTROLLERS = ("This figure is drawn by matplotlib. Click-to-select arrives with "
                  "editable matplotlib figures.")


class ElementRef(NamedTuple):
    """One selectable element of a figure. UI only, never saved."""

    plot: str       # "" on single-plot pages; the pin view id ("psd" "map" "embedding" ...) elsewhere
    key: str        # a text key from figure_text.py or a UI-only key: "figure" "axis.bottom"
                    # "axis.left" "legend" "series.data" "series.fit" "series.<source key>"


# -- the shared state ------------------------------------------------------------------------------
class SelectionHub(QtCore.QObject):
    """The element selected in one page's Format tab, and who may change it. One per
    ``PagePane``; its parent is the pane."""

    selectionChanged = QtCore.Signal(object, bool)    # ElementRef | None, from_canvas
    elementsChanged = QtCore.Signal()
    formatChanged = QtCore.Signal()                   # coalesced with a 0 ms single-shot

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._current: ElementRef | None = None
        self._controllers: list[weakref.ref] = []
        self._refs: list[ElementRef] = []
        self._format_pending = False

    def select(self, ref: ElementRef | None, *, from_canvas: bool = False) -> None:
        if ref == self._current:
            return
        self._current = ref
        self.selectionChanged.emit(ref, bool(from_canvas))

    def clear(self) -> None:
        self.select(None)

    def current(self) -> ElementRef | None:
        return self._current

    def register(self, controller) -> None:
        """Remember a ``FigureTextController`` (weakly: the hub never keeps a plot alive)."""
        self._controllers = [ref for ref in self._controllers if ref() is not None]
        if all(ref() is not controller for ref in self._controllers):
            self._controllers.append(weakref.ref(controller))

    def controllers(self) -> list:
        return [c for c in (ref() for ref in self._controllers) if c is not None]

    def refs(self) -> list[ElementRef]:
        """The elements the list shows now."""
        return list(self._refs)

    def set_refs(self, refs) -> None:
        refs = list(refs)
        if refs != self._refs:
            self._refs = refs
            self.elementsChanged.emit()

    def request_format_changed(self) -> None:
        """Say the figure's look or words moved, once per event-loop turn."""
        if not self._format_pending:
            self._format_pending = True
            QtCore.QTimer.singleShot(0, self, self._emit_format_changed)

    def _emit_format_changed(self) -> None:
        self._format_pending = False
        self.formatChanged.emit()


# -- where a value comes from ----------------------------------------------------------------------
#: The figure-wide field behind a text element's size, by the kind of element.
def _size_field(key: str) -> str | None:
    if key == "title":
        return "title_size"
    if key.startswith("axis.") and key.endswith(".label"):
        return "axis_size"
    if key.startswith("axis.") and key.endswith(".ticks"):
        return "tick_size"
    if key.startswith("legend.entry."):
        return "legend_size"
    return None


def _size_role(key: str) -> str:
    return {"title_size": "Title", "axis_size": "Axis labels", "tick_size": "Ticks",
            "legend_size": "Legend text"}.get(_size_field(key) or "", "Figure")


def resolve(state, defaults, ref: ElementRef, prop: str, *, storage_key=None) -> tuple[object, str]:
    """``(effective value, source)`` of ``prop`` for ``ref``, ``source`` in
    ``{"element", "figure", "default"}`` (Phase 3 adds "style", and no panel changes).

    ``prop`` is a ``TextStyle`` field for a text element and a ``PlotAppearance`` field for
    the figure itself. ``element``: the element's own override (the figure's, for a
    figure-wide field that differs from ``defaults``); ``figure``: the element has none and
    the figure-wide field for its role (``font_family``, ``title_size``...) is not the page's
    default; ``default``: neither, and the value is the page's default. ``defaults`` is a
    ``PlotAppearance`` holding the page's real defaults (Distributions reports seven concrete
    values at their defaults, which "not None" would mark as set), or None.

    Nothing here reads a drawn item or builds anything."""
    key = ref.key
    binding = getattr(state, "style_binding", None)
    if binding:
        from .figure_style_binding import NATIVE_LOOK_KEYS, resolved_binding, resolve_bound_text
        if key == FIGURE and prop in NATIVE_LOOK_KEYS:
            resolved = resolved_binding(binding)
            origin = resolved.origins.get(NATIVE_LOOK_KEYS[prop], "default")
            return getattr(state, prop, None), "element" if origin == "figure" else origin
        if key != FIGURE:
            explicit = state.text_style(storage_key or key).to_dict()
            values, origins = resolve_bound_text(binding, key, explicit)
            if prop in values:
                return values[prop], origins[prop]
    if key == FIGURE:
        value = getattr(state, prop, None)
        base = getattr(defaults, prop, None) if defaults is not None else None
        if value is not None and value != base:
            return value, "element"
        return base, "default"
    style = state.text_style(storage_key or key) if hasattr(state, "text_style") else None
    value = getattr(style, prop, None) if style is not None else None
    if value is not None:
        return value, "element"
    field = "font_family" if prop == "family" else _size_field(key) if prop == "size_pt" else None
    if field is not None:
        figure = getattr(state, field, None)
        base = getattr(defaults, field, None) if defaults is not None else None
        if figure is not None and figure != base:
            return figure, "figure"
    own = getattr(defaults, prop, None) if defaults is not None else None
    if own is None and field is not None and defaults is not None:
        own = getattr(defaults, field, None)
    return own, "default"


class FormatClipboard:
    """Copy format / Paste format, in memory only: one per Explorer, not the system
    clipboard, not ``QSettings``, never saved."""

    def __init__(self) -> None:
        self._held: object | None = None

    def put(self, value: object) -> None:
        self._held = value

    def get(self) -> object | None:
        return self._held

    def clear(self) -> None:
        self._held = None

    @classmethod
    def for_window(cls, widget: QtWidgets.QWidget) -> "FormatClipboard":
        """The one clipboard of ``widget``'s top-level window (the Explorer)."""
        top = widget.window()
        held = getattr(top, "_nanoscope_format_clipboard", None)
        if not isinstance(held, cls):
            held = cls()
            try:
                top._nanoscope_format_clipboard = held
            except (AttributeError, RuntimeError):
                pass
        return held


@dataclass(frozen=True)
class _Copied:
    """What Copy format holds: a look, never words. ``kind`` is what it may be pasted onto."""

    kind: str                              # "text" | "axis" | "figure"
    style: TextStyle | None = None         # a text element's style, text removed
    label: TextStyle | None = None         # an axis: its label's and its numbers' styles
    ticks: TextStyle | None = None
    look: PlotAppearance | None = None     # the figure: ``style_only()``
    series: dict | None = None             # series ink only; no identity or visibility
    vocabulary: str = "plot_appearance"     # native units of an unbound copied figure


class ElementList(QtWidgets.QTreeWidget):
    """The object list at the top of the Format tab (accessible name "Figure elements")."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("formatElementList")
        self.setHeaderHidden(True)
        self.setAccessibleName("Figure elements")
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setMinimumHeight(96)
        self.setMaximumHeight(150)

    def ref_of(self, item: QtWidgets.QTreeWidgetItem | None) -> ElementRef | None:
        data = item.data(0, _ROLE) if item is not None else None
        return ElementRef(*data) if isinstance(data, (tuple, list)) and len(data) == 2 else None

    def find(self, ref: ElementRef | None) -> QtWidgets.QTreeWidgetItem | None:
        """The row of ``ref``, looked up by KEY each time (no row is kept between rebuilds)."""
        if ref is None:
            return None
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            if self.ref_of(item) == ref:
                return item
            stack.extend(item.child(i) for i in range(item.childCount()))
        return None


class _Panel(QtWidgets.QWidget):
    """A properties panel: built once, then RETARGETED by :meth:`show_ref` (no widget is
    created or destroyed on a selection)."""

    kind = ""

    def __init__(self, pane: "FormatPane") -> None:
        super().__init__()
        self.setObjectName(f"formatPanel_{self.kind}")
        self._pane = weakref.ref(pane)
        self.ref: ElementRef | None = None
        self._column = QtWidgets.QVBoxLayout(self)
        self._column.setContentsMargins(6, 6, 6, 6)

    @property
    def pane(self) -> "FormatPane | None":
        return self._pane()

    def show_ref(self, ref: ElementRef) -> None:
        self.ref = ref

    def refresh(self) -> None:
        if self.ref is not None:
            self.show_ref(self.ref)

    # -- building blocks ----------------------------------------------------------------------
    def _note(self, text: str) -> QtWidgets.QLabel:
        note = QtWidgets.QLabel(text)
        note.setWordWrap(True)
        note.setEnabled(False)               # the grey words
        return note

    def _form(self, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QFormLayout:
        form = QtWidgets.QFormLayout(parent) if parent is not None else QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        return form

    def _move_row(self, form: QtWidgets.QFormLayout, control: QtWidgets.QWidget | None,
                  dot: SetHereDot | None = None, text: str = "") -> QtWidgets.QWidget | None:
        """Move ``control`` and its label (``control._nanoscope_row_label``) into ``form`` as
        one row. A control the widget had hidden keeps its hidden flag and its row is hidden."""
        if control is None:
            return None
        label = getattr(control, "_nanoscope_row_label", None)
        hidden = control.isHidden() or (label is not None and label.isHidden())
        prepare_widget_move(control)
        if label is not None:
            prepare_widget_move(label)
        field: QtWidgets.QWidget = control
        if dot is not None:
            field = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(field)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(control, 1)
            row.addWidget(dot)
        form.addRow(label if label is not None else text, field)
        if hidden:
            form.setRowVisible(field, False)
        else:
            # A reparented widget is shown again by a queued call; say so now, so it is
            # visible the moment its panel is (a widget the page hid keeps its own flag).
            for shown in (control, label):
                if shown is not None:
                    shown.setVisible(True)
        return field

    def _move_extras(self, key: str) -> None:
        """The page's own Format widgets for this panel, moved in, in order."""
        pane = self.pane
        for widget in (pane.extras(key) if pane is not None else ()):
            if getattr(widget, "_nanoscope_row_label", None) is not None:
                form = self._form()
                self._move_row(form, widget)
                self._column.addLayout(form)
                continue
            prepare_widget_move(widget)
            if isinstance(widget, QtWidgets.QScrollArea):
                widget.setMaximumHeight(_UNLIMITED)
            hidden = widget.isHidden()
            self._column.addWidget(widget)
            if not hidden:
                widget.setVisible(True)


class _FieldPanel(_Panel):
    """A panel of figure-wide fields of the appearance widget, each a moved row with a dot."""

    #: ((field name, form index), ...) filled by the subclasses
    def _field_dot(self, field: str) -> SetHereDot:
        dot = SetHereDot(prop=field)
        dot.cleared.connect(functools.partial(self._clear_field, field))
        self._dots[field] = dot
        return dot

    def __init__(self, pane: "FormatPane") -> None:
        super().__init__(pane)
        self._dots: dict[str, SetHereDot] = {}

    def _add_field(self, form: QtWidgets.QFormLayout, field: str) -> None:
        pane = self.pane
        control = getattr(pane.widget(), field, None) if pane is not None else None
        if control is not None:
            self._move_row(form, control, self._field_dot(field))

    def _clear_field(self, field: str) -> None:
        pane = self.pane
        if pane is not None:
            pane.write_figure(**{field: None})

    def show_ref(self, ref: ElementRef) -> None:
        super().show_ref(ref)
        pane = self.pane
        if pane is None:
            return
        state, defaults = pane.state(), pane.defaults()
        for field, dot in self._dots.items():
            _value, source = resolve(state, defaults, ElementRef("", FIGURE), field)
            dot.set_source(source)


class FigurePanel(_FieldPanel):
    """Font, background, grid, the role sizes (title, axis labels, ticks, legend text), the
    page's own figure-wide widgets, and Reset look."""

    kind = "figure"

    def __init__(self, pane: "FormatPane") -> None:
        super().__init__(pane)
        form = self._form()
        for field in ("font_family", "background", "grid", "frame"):
            self._add_field(form, field)
        self._column.addLayout(form)
        sizes = QtWidgets.QGroupBox("Text sizes")
        sizes.setObjectName("formatTextSizes")
        size_form = self._form(sizes)
        for field in ("title_size", "axis_size", "tick_size", "legend_size"):
            self._add_field(size_form, field)
        self._column.addWidget(sizes)
        self._column.addWidget(self._note(
            "These are the figure's own sizes. A text with its own size says so in its panel."))
        self._move_extras("figure")
        self.reset_look = QtWidgets.QPushButton("Reset look")
        self.reset_look.setToolTip(
            "Put the figure's fonts, sizes, colours and every element's style back to the "
            "defaults, as one Undo step. The words you typed, axis ranges, where annotations "
            "sit and legend names are kept.")
        self.reset_look.clicked.connect(self._reset_look)
        self._column.addWidget(self.reset_look)
        self._column.addStretch(1)

    def _reset_look(self, _checked: bool = False) -> None:
        pane = self.pane
        if pane is not None:
            pane.reset_look()


class LegendPanel(_FieldPanel):
    """Show or hide the legend, and the page's own legend widgets."""

    kind = "legend"

    def __init__(self, pane: "FormatPane") -> None:
        super().__init__(pane)
        form = self._form()
        self._add_field(form, "legend_visible")
        self._column.addLayout(form)
        self._column.addWidget(self._note(
            "Click a legend name on the figure to restyle or rename it. Its text size is in "
            "Figure ▸ Text sizes."))
        self._move_extras("legend")
        self._column.addStretch(1)


class SeriesPanel(_FieldPanel):
    """Whole-figure defaults and independently identified series share one panel."""
    kind = "series"

    def __init__(self, pane):
        super().__init__(pane)
        self._loading = False
        self._series_id = None
        self.global_box = QtWidgets.QGroupBox("Data and fit defaults")
        self.global_box.setObjectName("formatSeriesFigureWide")
        form = self._form(self.global_box)
        for field in ("line_width", "marker_size", "data_color", "fit_color"):
            self._add_field(form, field)
        self._column.addWidget(self.global_box)
        self.series_box = QtWidgets.QGroupBox("Selected series")
        form = self._form(self.series_box)
        self.series_name = QtWidgets.QLabel(); self.series_name.setWordWrap(True)
        form.addRow(self.series_name)
        self.color = QtWidgets.QPushButton("From figure")
        self.color.clicked.connect(self._choose_color)
        self.line_width = QtWidgets.QDoubleSpinBox(); self.marker_size = QtWidgets.QDoubleSpinBox()
        for control in (self.line_width, self.marker_size):
            control.setRange(0., 1000.); control.setDecimals(3); control.setSingleStep(.25)
            control.setSpecialValueText("From figure"); control.setSuffix(" pt")
        self.line_width.valueChanged.connect(lambda v: self._write("line_width_pt", v or None))
        self.marker_size.valueChanged.connect(lambda v: self._write("marker_size_pt", v or None))
        self.visible = QtWidgets.QComboBox()
        for label, value in (("From figure", None), ("Show", True), ("Hide", False)):
            self.visible.addItem(label, value)
        self.visible.currentIndexChanged.connect(lambda _index: self._write("visible", self.visible.currentData()))
        self._series_controls = {"color":self.color, "line_width_pt":self.line_width,
                                 "marker_size_pt":self.marker_size, "visible":self.visible}
        self._series_dots = {}
        for field, label in (("color","Colour"),("line_width_pt","Line width"),
                             ("marker_size_pt","Marker size"),("visible","Visible")):
            dot = SetHereDot(prop=field); dot.cleared.connect(lambda field=field: self._write(field,None))
            self._series_dots[field] = dot
            row=QtWidgets.QWidget(); layout=QtWidgets.QHBoxLayout(row); layout.setContentsMargins(0,0,0,0)
            layout.addWidget(self._series_controls[field],1);layout.addWidget(dot);form.addRow(label,row)
        self._column.addWidget(self.series_box)
        self.palette = QtWidgets.QPushButton("Apply style palette to data series")
        self.palette.setToolTip("Set figure-local series colours from this style's palette. Project source colours stay unchanged.")
        self.palette.clicked.connect(self._apply_palette);self._column.addWidget(self.palette)
        self._move_extras("series")
        if callable(pane._spec.project_sources):
            button=QtWidgets.QPushButton("Project sources…");button.clicked.connect(pane._spec.project_sources)
            button.setObjectName("formatProjectSources")
            button.setToolTip("Edit names and colours for exact project source identities.")
            self._column.addWidget(button)
        self._column.addStretch(1)

    def _write(self, field, value):
        if not self._loading and self._series_id and self.pane is not None:
            self.pane.write_series(self._series_id, {field:value})

    def _choose_color(self):
        pane=self.pane
        if pane is None or self._series_id is None:return
        descriptor=pane.series_descriptor(self._series_id)
        if descriptor is None:return
        own=(pane.state().series_styles or {}).get(self._series_id,{})
        color=QtWidgets.QColorDialog.getColor(QtGui.QColor(own.get("color") or descriptor.color or "#0072b2"), self,"Series colour")
        if color.isValid():self._write("color",color.name())

    def _palette(self):
        pane=self.pane
        if pane is None or not pane.state().style_binding:return ()
        from .figure_style_binding import resolved_binding
        return tuple(resolved_binding(pane.state().style_binding).values.get("palette",()))

    def _apply_palette(self):
        palette=self._palette();pane=self.pane
        if not palette or pane is None:return
        targets=[d for d in pane.series_descriptors() if d.role == "data" and "color" in d.capabilities]
        from ..core.figure_spec import thaw_json
        values=thaw_json(pane.state().series_styles or {})
        for index,descriptor in enumerate(targets):
            values.setdefault(descriptor.series_id,{})["color"]=palette[index % len(palette)]
        pane.widget().apply_appearance(replace(pane.state(),series_styles=values or None))

    def show_ref(self, ref):
        super().show_ref(ref)
        pane=self.pane
        if pane is None:return
        descriptor=pane.series_descriptor(ref.key)
        self._series_id=descriptor.series_id if descriptor else None
        self.global_box.setVisible(descriptor is None);self.series_box.setVisible(descriptor is not None)
        self.palette.setEnabled(bool(self._palette()) and any(d.role == "data" for d in pane.series_descriptors()))
        if descriptor is None:return
        self._loading=True
        try:
            own=(pane.state().series_styles or {}).get(descriptor.series_id,{})
            self.series_name.setText(descriptor.label)
            self.color.setText(own.get("color") or "From figure")
            self.line_width.setValue(own.get("line_width_pt",0));self.marker_size.setValue(own.get("marker_size_pt",0))
            self.visible.setCurrentIndex(self.visible.findData(own.get("visible")))
            for field,control in self._series_controls.items():
                control.setEnabled(field in descriptor.capabilities)
                inherited = "style" if pane.state().style_binding else "figure"
                self._series_dots[field].set_source("element" if field in own else inherited)
        finally:self._loading=False


class ColourMapPanel(_Panel):
    """The page's colour map and level controls, moved in whole."""

    kind = "colour_map"

    def __init__(self, pane: "FormatPane") -> None:
        super().__init__(pane)
        self._move_extras("colour_map")
        self._column.addStretch(1)


class TextPanel(_Panel):
    """One text element: its words (the moved Title / X label / Y label boxes, or one box
    for the others) and its font, size, bold, italic, colour and alignment, each with its
    "set here" marker. Serves the title, axis labels and numbers, legend entries, colour-bar
    and marginal labels and any other annotation."""

    kind = "text"

    #: key -> (appearance attribute of the words box, caption)
    _WORDS = {"title": "title", "axis.bottom.label": "x_label", "axis.left.label": "y_label"}

    def __init__(self, pane: "FormatPane") -> None:
        super().__init__(pane)
        app = pane.widget()
        self._words_form = self._form()
        self._moved: dict[str, QtWidgets.QWidget] = {}
        # Only the shared text panel owns these existing wording rows. The fit-label
        # and caption panels have their own generic words box; moving the same rows
        # into either subclass would steal them from an already-built text panel.
        for key, attr in (self._WORDS.items() if self.kind == "text" else ()):
            box = getattr(app, attr, None)
            if box is not None and pane._appearance_has(attr):
                self._moved[key] = self._move_row(self._words_form, box)
            elif box is not None:
                self._move_row(self._words_form, box)
                self._words_form.setRowVisible(box, False)
        self.words = QtWidgets.QLineEdit()
        self.words.setAccessibleName("Text")
        self.words.textEdited.connect(self._words_typed)
        self._words_form.addRow("Text", self.words)
        self._column.addLayout(self._words_form)
        self._locked = self._note("")
        self._column.addWidget(self._locked)
        self.fields = TextStyleFields(live=True, inherit_label="Figure font", dots=True)
        self.fields.changed.connect(self._written)
        self.fields.editingFinished.connect(self._finish_run)
        self.words.editingFinished.connect(self._finish_run)
        self._column.addWidget(self.fields)
        self.figure_link = QtWidgets.QToolButton()
        self.figure_link.setText("Figure text sizes…")
        self.figure_link.setAutoRaise(True)
        self.figure_link.setToolTip("Some of this text's values come from the figure: they "
                                    "are set in Figure ▸ Text sizes.")
        self.figure_link.clicked.connect(self._to_figure)
        self._column.addWidget(self.figure_link)
        self._move_extras(self.kind)
        self._column.addStretch(1)

    # -- showing ------------------------------------------------------------------------------
    def show_ref(self, ref: ElementRef) -> None:
        super().show_ref(ref)
        pane = self.pane
        if pane is None:
            return
        target = pane.target_for(ref)
        controller = pane.controller_for(ref)
        independent_words = bool(getattr(getattr(controller, "host", None),
                                         "independent_text", False))
        for key, field in self._moved.items():
            self._words_form.setRowVisible(field, key == ref.key and not independent_words)
        own = ref.key in self._moved and not independent_words
        caps = target.capabilities if target is not None else frozenset()
        generic = (not own) and "edit" in caps
        self._words_form.setRowVisible(self.words, generic)
        if generic and target is not None:
            typed = target.style.text
            shown = "" if typed in (None, HIDDEN_TEXT) else typed
            if self.words.text() != shown:        # a keystroke's own redraw must not move the cursor
                self.words.setText(shown)
            self.words.setPlaceholderText(f"Automatic: {target.auto}" if target.auto
                                          else "Automatic")
        locked = pane.label_locked(ref) if ref.key.endswith(".label") else ""
        self._locked.setText(locked)
        self._locked.setVisible(bool(locked))
        has_style = target is not None and "style" in caps
        self.fields.setVisible(has_style)
        if has_style:
            resolved, notes = pane.resolve_text(ref)
            self.fields.set_target(ref.key, target, resolved, notes=notes)
            self.figure_link.setVisible(any(src == "figure" for _v, src in resolved.values()))
        else:
            self.figure_link.setVisible(False)

    # -- writing ------------------------------------------------------------------------------
    def _written(self, prop: str, value) -> None:
        pane = self.pane
        if pane is not None and self.ref is not None:
            pane.write_text(self.ref, prop, value, run=(prop == "size_pt"))

    def _words_typed(self, text: str) -> None:
        pane = self.pane
        if pane is not None and self.ref is not None:
            pane.write_text(self.ref, "text", TextStyle.from_mapping({"text": text}).text,
                            run=True)

    def _finish_run(self) -> None:
        pane = self.pane
        controller = pane.controller_for(self.ref) if pane is not None else None
        if controller is not None:
            controller.finish_style_run()

    def _to_figure(self, _checked: bool = False) -> None:
        pane = self.pane
        if pane is not None:
            pane.select(ElementRef("", FIGURE))


class FitLabelPanel(TextPanel):
    """The fit label: its wording and style (font, bold, italic, size, colour) like any text,
    then the page's whole "Fit label" group (show, the numbers, position, in legend)."""

    kind = "fit_label"

    def __init__(self, pane: "FormatPane") -> None:
        super().__init__(pane)
        group = getattr(pane.widget(), "fit_label_group", None)
        if group is not None:
            # The group's own Text, Size and Colour rows say what the rows above say: the
            # same stores (``PlotAppearance.fit_label`` through ``with_text_style``).
            form = group.layout()
            if isinstance(form, QtWidgets.QFormLayout):
                for name in ("fit_label_template", "fit_label_size", "fit_label_color"):
                    control = getattr(pane.widget(), name, None)
                    if control is not None:
                        form.setRowVisible(control, False)
            hidden = group.isHidden()           # until the host says it draws a fit label
            prepare_widget_move(group)
            self._column.insertWidget(self._column.count() - 1, group)
            if not hidden:
                group.setVisible(True)


class CaptionPanel(TextPanel):
    """The provenance caption: its wording and style, and the page's own caption widgets."""

    kind = "caption"


class AxisPanel(_Panel):
    """One axis (X or Y): range with Auto, major tick spacing and the read-only Log mirror,
    the live version of "Axis properties…". Reads ``host.axis_state(side)`` and writes
    ``host.apply_axis(side, AxisState)``, the two calls the popover makes, so it is right on
    every page that has a controller. Its label and numbers are elements of their own."""

    kind = "axis"

    def __init__(self, pane: "FormatPane") -> None:
        super().__init__(pane)
        self.fields = AxisFields(AxisState(), live=True)
        self.fields.edited.connect(self._edited)
        self._column.addWidget(self.fields)
        self.range_dot = SetHereDot(prop="range")
        self.range_dot.cleared.connect(self._auto_range)
        self.major_dot = SetHereDot(prop="tick_major")
        self.major_dot.cleared.connect(self._auto_major)
        self.dots = QtWidgets.QWidget()
        dots = QtWidgets.QFormLayout(self.dots)
        dots.setContentsMargins(0, 0, 0, 0)
        dots.addRow("Range", self.range_dot)
        dots.addRow("Major tick", self.major_dot)
        self._column.addWidget(self.dots)
        self.log_link = QtWidgets.QToolButton()
        self.log_link.setText("Change Log in Data & analysis")
        self.log_link.setAutoRaise(True)
        self.log_link.setToolTip("Log changes what is binned or fitted, so it stays with the "
                                 "page's own controls.")
        self.log_link.clicked.connect(self._to_data)
        self._column.addWidget(self.log_link)
        self._column.addStretch(1)

    def _side(self) -> str:
        return (self.ref.key.split(".")[1] if self.ref is not None else "bottom")

    def show_ref(self, ref: ElementRef) -> None:
        super().show_ref(ref)
        pane = self.pane
        controller = pane.controller_for(ref) if pane is not None else None
        state = None
        if controller is not None:
            try:
                state = controller.host.axis_state(self._side())
            except (RuntimeError, AttributeError, NotImplementedError):
                state = None
        self.setEnabled(state is not None)
        if state is None:
            return
        self.fields.set_state(state)
        self.range_dot.set_source("element" if state.range is not None else "default")
        self.major_dot.set_source("element" if state.tick_major is not None else "default")
        self.dots.layout().setRowVisible(self.major_dot, state.tick_major_available)
        destination = pane.axis_log_target(ref) if pane is not None else None
        self.log_link.setVisible(state.log_available and destination is not None)
        self.log_link.setText("Change Log in Colour map" if destination is not None
                             and destination[0] == "@format:colour_map"
                             else "Change Log in Data & analysis")

    def _edited(self) -> None:
        pane = self.pane
        controller = pane.controller_for(self.ref) if pane is not None and self.ref else None
        if controller is None:
            return
        value = self.fields.value()
        if value is None:
            self.fields.error.setText("Type a range whose minimum is below its maximum "
                                      "(and above 0 on a log axis).")
            self.fields.error.show()
            return
        self.fields.error.hide()
        side = self._side()
        current = controller.host.axis_state(side)
        controller._apply_axis(side, replace(current, range=value.range,
                                             tick_major=value.tick_major))

    def _auto_range(self) -> None:
        self._set_pin(range=None)

    def _auto_major(self) -> None:
        self._set_pin(tick_major=None)

    def _set_pin(self, **changes) -> None:
        pane = self.pane
        controller = pane.controller_for(self.ref) if pane is not None and self.ref else None
        if controller is not None:
            side = self._side()
            controller._apply_axis(side, replace(controller.host.axis_state(side), **changes))

    def _to_data(self, _checked: bool = False) -> None:
        pane = self.pane
        if pane is not None and self.ref is not None:
            pane.reveal_axis_log(self.ref)


# -- the pane --------------------------------------------------------------------------------------
_PANELS = {"figure": FigurePanel, "text": TextPanel, "axis": AxisPanel, "legend": LegendPanel,
           "series": SeriesPanel, "fit_label": FitLabelPanel, "colour_map": ColourMapPanel,
           "caption": CaptionPanel}


def _kind_of(key: str) -> str:
    if key == FIGURE:
        return "figure"
    if key == LEGEND:
        return "legend"
    if key == COLOUR_MAP:
        return "colour_map"
    if key.startswith("series."):
        return "series"
    if key.startswith("axis.") and key.count(".") == 1:
        return "axis"
    if key == "annotation.fit":
        return "fit_label"
    if key == "annotation.provenance":
        return "caption"
    return "text"


class FormatPane(QtWidgets.QWidget):
    """The Format tab of one page: a skeleton at adoption, filled on first use.

    The element list over one properties panel that follows the selection, and a footer
    (Copy format, Paste format, Reset element). ``spec.appearance`` is the page's
    ``PlotAppearanceWidget`` whose rows the panels hold; ``spec.controllers()`` the
    ``FigureTextController`` of every plot on screen; ``spec.extras`` the page's own widgets
    per panel (``figure legend series fit_label caption colour_map``).

    A page whose Format widgets are still the whole Appearance section (the Foundation's
    fallback adapter) is shown as the Foundation showed it, in one column, until it has an
    adapter of its own."""

    def __init__(self, spec: FormatSpec, hub: SelectionHub,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("analysisFormatPane")
        self._spec = spec
        self._hub = hub
        self._built = False
        self._body: QtWidgets.QWidget | None = None       # the legacy single column
        self._legacy = self._is_legacy()
        appearance = self.widget()
        available = getattr(appearance, "_nanoscope_format_controls", None)
        if available is None:
            # Keep the page's original capabilities with its state owner. A retired
            # pane leaves unselected wording rows hidden; those presentation flags
            # must not become missing controls when the same page is hosted again.
            available = frozenset(
                name for name in ("title", "x_label", "y_label", "legend_visible", "line_width")
                if (control := getattr(appearance, name, None)) is not None and not control.isHidden()
            )
            if appearance is not None:
                appearance._nanoscope_format_controls = available
        self._available_controls = available
        self._panels: dict[str, _Panel] = {}
        self._rows: list[tuple[ElementRef, str, ElementRef | None]] = []
        self._wired: list[weakref.ref] = []
        self._series_wired: list[weakref.ref] = []
        self._stale = False
        self._rebuild_pending = False
        self._husk_hidden = False
        self._retired = False
        self._canvas_claim_pending = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._layout = layout
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # The fallback holds the page's 4-column appearance widget (483 px wide) whole, so it
        # scrolls sideways where needed. The panels never will.
        self._scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded if self._legacy
            else QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        if self._legacy:
            layout.addWidget(self._scroll)
        # Listening costs no widget: a click on the canvas must find the pane before the
        # Format tab was ever shown.
        if not self._legacy:
            self._sync_controllers()
            if self._spec.series_selected is not None:
                self._spec.series_selected.connect(self._on_series_selected)

    # -- what the page gave -------------------------------------------------------------------
    def widget(self):
        """The page's ``PlotAppearanceWidget`` (the state owner), or None."""
        return self._spec.appearance if not self._legacy else None

    def extras(self, key: str):
        return tuple((self._spec.extras or {}).get(key, ()))

    def series_descriptors(self):
        provider = self._spec.series_provider
        if callable(provider):
            values = tuple(provider())
        else:
            from .figure_series import pg_series_descriptors
            values = pg_series_descriptors([c.plot for c in self._controllers() if getattr(c,"plot",None) is not None])
        found = {}
        for descriptor in values:
            found.setdefault(descriptor.series_id, descriptor)
        return tuple(found.values())

    def series_descriptor(self, key):
        return next((d for d in self.series_descriptors() if d.series_id == key), None)

    def write_series(self, key, changes):
        descriptor = self.series_descriptor(key)
        if descriptor is None or self.widget() is None:return
        from ..core.figure_spec import thaw_json
        values = thaw_json(self.state().series_styles or {})
        own = values.setdefault(key,{})
        for field,value in changes.items():
            if field not in descriptor.capabilities:continue
            if value is None:own.pop(field,None)
            else:own[field]=value
        if not own:values.pop(key,None)
        self.widget().apply_appearance(replace(self.state(),series_styles=values or None))

    @QtCore.Slot(str)
    def _on_series_selected(self, key):
        if self._retired or self.series_descriptor(key) is None:return
        if not self._canvas_claim_pending:
            self._canvas_claim_pending = True
            QtCore.QTimer.singleShot(0, self, self._finish_canvas_click)
        self._ensure_built()
        self._hub.select(ElementRef("",key),from_canvas=True)
        self.show_inspector_tab(1)

    def state(self) -> PlotAppearance:
        widget = self.widget()
        return widget.state() if widget is not None else PlotAppearance()

    def defaults(self) -> PlotAppearance:
        """What the page draws when nothing is set (``FormatSpec.defaults``; else
        ``PlotAppearance()``). Used only to tell a set value from a default one."""
        provider = self._spec.defaults
        if callable(provider):
            try:
                value = provider()
            except (RuntimeError, AttributeError):
                value = None
            if isinstance(value, PlotAppearance):
                return value
        return PlotAppearance()

    def _is_legacy(self) -> bool:
        """The Foundation's single column: no usable appearance widget, or one the page's
        Format widgets (the fallback's Appearance section) already hold whole."""
        app = self._spec.appearance
        needs = ("font_family", "title", "state", "apply_appearance", "apply_look")
        if not isinstance(app, QtWidgets.QWidget) or not all(hasattr(app, n) for n in needs):
            return True
        for widget in (self._spec.extras or {}).get("figure", ()):
            if widget is app or (isinstance(widget, QtWidgets.QWidget) and widget.isAncestorOf(app)):
                return True
        return False

    def is_legacy(self) -> bool:
        return self._legacy

    # -- controllers --------------------------------------------------------------------------
    def _controllers(self) -> list:
        try:
            return [c for c in self._spec.controllers() if c is not None and _alive(c)]
        except (RuntimeError, AttributeError):
            return []

    def _sync_controllers(self) -> None:
        """Listen to every controller on screen now (a connection costs no widget)."""
        if self._retired:
            return
        for controller in self._controllers():
            emitter = getattr(getattr(controller,"plot",None),"_nanoscope_series_selection",None)
            if emitter is not None and not any(ref() is emitter for ref in self._series_wired):
                self._series_wired.append(weakref.ref(emitter))
                emitter.selected.connect(self._on_series_selected)
            if any(ref() is controller for ref in self._wired):
                continue
            self._wired.append(weakref.ref(controller))
            self._hub.register(controller)
            controller.claimed.connect(self._on_claimed)
            controller.cleared.connect(self._on_cleared)
            controller.editingChanged.connect(self._on_editing_changed)

    @staticmethod
    def plot_id(controller) -> str:
        return str(getattr(getattr(controller, "host", None), "pin_view", "") or "")

    def controller_for(self, ref: ElementRef | None):
        """The controller that draws ``ref`` (the one whose view id is ``ref.plot``; the first
        for an element of no view in particular, such as Figure)."""
        controllers = self._controllers()
        plot = ref.plot if ref is not None else ""
        for controller in controllers:
            if self.plot_id(controller) == plot:
                return controller
        return controllers[0] if controllers and not plot else None

    def target_for(self, ref: ElementRef | None) -> TextTarget | None:
        controller = self.controller_for(ref)
        if controller is not None and ref:
            return controller.target(ref.key)
        # Matplotlib pages retain the existing wording controls even though selectable
        # canvas objects and per-element styling arrive in a later phase.
        field = {"title": "title", "axis.bottom.label": "x_label",
                 "axis.left.label": "y_label"}.get(ref.key if ref else "")
        if field in self._available_controls:
            return TextTarget(None, "", self.state().text_style(ref.key), "text",
                              frozenset({"edit"}))
        return None

    # -- building -----------------------------------------------------------------------------
    def _ensure_built(self) -> None:
        if self._built:
            return
        self._built = True
        if self._legacy:
            self._build_legacy()
            return
        widget = self.widget()
        self._hide_husk(widget)
        body = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        self.list = ElementList()
        self.list.currentItemChanged.connect(self._list_changed)
        column.addWidget(self.list)
        self.note = QtWidgets.QLabel(NO_CONTROLLERS)
        self.note.setObjectName("formatNoCanvasNote")
        self.note.setWordWrap(True)
        self.note.setEnabled(False)
        column.addWidget(self.note)
        self._stack = QtWidgets.QStackedWidget()
        self._scroll.setWidget(self._stack)
        column.addWidget(self._scroll, 1)
        footer = QtWidgets.QWidget()
        buttons = QtWidgets.QHBoxLayout(footer)
        buttons.setContentsMargins(4, 0, 4, 4)
        self.copy_button = QtWidgets.QPushButton("Copy format")
        self.paste_button = QtWidgets.QPushButton("Paste format")
        self.reset_button = QtWidgets.QPushButton("Reset element")
        self.copy_button.setToolTip("Hold this element's look (never its words) to paste onto "
                                    "another element.")
        self.paste_button.setToolTip("Apply the copied look to this element; its own words stay.")
        self.reset_button.setToolTip("Clear every style this element has set on its own, as one "
                                     "Undo step. Its words are kept.")
        self.copy_button.clicked.connect(self.copy_format)
        self.paste_button.clicked.connect(self.paste_format)
        self.reset_button.clicked.connect(self.reset_element)
        for button in (self.copy_button, self.paste_button, self.reset_button):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        column.addWidget(footer)
        self._layout.addWidget(body)
        self._body = body
        if self.isVisible():
            body.show()          # built inside a show event: do not wait for the queued show
        self._hub.selectionChanged.connect(self._on_selection)
        self._hub.formatChanged.connect(self._on_format_changed)
        if widget is not None:
            for name in ("changed", "restored", "historyChanged"):
                signal = getattr(widget, name, None)
                if signal is not None:
                    signal.connect(self._appearance_moved)
        self._sync_controllers()
        self._fill_list()
        self._clipboard = FormatClipboard.for_window(self)
        if self._hub.current() is None:
            self._hub.select(ElementRef("", FIGURE))
        else:
            self._on_selection(self._hub.current(), False)

    def _hide_husk(self, widget) -> None:
        """The appearance widget becomes a hidden husk: its rows live in the panels."""
        if widget is not None and not self._husk_hidden:
            self._husk_hidden = True
            widget.hide()

    def _build_legacy(self) -> None:
        widgets = list(self._spec.extras.get("figure", ())) if self._spec.extras else []
        if not widgets and isinstance(self._spec.appearance, QtWidgets.QWidget):
            widgets = [self._spec.appearance]
        body = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(body)
        column.setContentsMargins(6, 6, 6, 6)
        for widget in widgets:
            prepare_widget_move(widget)
            if isinstance(widget, QtWidgets.QScrollArea):
                widget.setMaximumHeight(_UNLIMITED)     # one scroll area for the whole tab
            column.addWidget(widget)
            widget.setVisible(True)
        column.addStretch(1)
        self._body = body
        self._scroll.setWidget(body)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._ensure_built()
        super().showEvent(event)
        if not self._legacy:
            self._sync_controllers()
            self._stale = False
            self._refresh_now()

    def retire_overlays(self) -> None:
        """Hide and retire every outline and mini toolbar (``hide()``, then ``deleteLater``)."""
        from .format_toolbar import retire

        for controller in (ref() for ref in self._wired):
            if controller is None or not _alive(controller):
                continue
            for name in ("_nanoscope_outline", "_nanoscope_minibar"):
                retire(getattr(controller, name, None))
                setattr(controller, name, None)

    def retire(self) -> None:
        """Disconnect the retained pane before its page can be hosted again elsewhere."""
        if self._retired:
            return
        self._retired = True
        self.retire_overlays()
        for emitter in (ref() for ref in self._series_wired):
            if emitter is not None and _alive(emitter):
                emitter.selected.disconnect(self._on_series_selected)
        self._series_wired.clear()
        if not self._legacy and self._spec.series_selected is not None:
            try:
                self._spec.series_selected.disconnect(self._on_series_selected)
            except RuntimeError:
                pass  # Its native renderer may already have been destroyed.
        if callable(self._spec.series_select):
            try:
                self._spec.series_select(None)
            except RuntimeError:
                pass
        for controller in (ref() for ref in self._wired):
            if controller is None or not _alive(controller):
                continue
            for signal, slot in ((controller.claimed, self._on_claimed),
                                 (controller.cleared, self._on_cleared),
                                 (controller.editingChanged, self._on_editing_changed)):
                signal.disconnect(slot)
        self._wired.clear()
        widget = self.widget()
        if self._built and widget is not None and _alive(widget):
            for name in ("changed", "restored", "historyChanged"):
                signal = getattr(widget, name, None)
                if signal is not None:
                    signal.disconnect(self._appearance_moved)

    # -- the element list ---------------------------------------------------------------------
    def elements(self) -> tuple[ElementRef, ...]:
        if self._legacy:
            return ()
        return tuple(ref for ref, _label, _parent in self._plan())

    def _plan(self) -> list[tuple[ElementRef, str, ElementRef | None]]:
        """The rows, top to bottom: ``(ref, text, parent)``. Computed from keys and capability
        flags only; nothing is built."""
        rows: list[tuple[ElementRef, str, ElementRef | None]] = [
            (ElementRef("", FIGURE), "Figure", None)]
        controllers = self._controllers()
        if not controllers:
            for field, key, label in (("title", "title", "Title"),
                                      ("x_label", "axis.bottom.label", "X axis label"),
                                      ("y_label", "axis.left.label", "Y axis label")):
                if self._appearance_has(field):
                    rows.append((ElementRef("", key), label, None))
        many = len(controllers) > 1
        seen_title = False
        legend_rows: list[tuple[ElementRef, str, ElementRef | None]] = []
        tail: list[tuple[ElementRef, str, ElementRef | None]] = []
        legend_hidden = None
        for controller in controllers:
            plot_id = self.plot_id(controller)
            targets = controller.targets()
            panel_label = str(getattr(controller.host, "panel_label", "") or plot_id)
            suffix = f" ({panel_label})" if many and panel_label else ""
            independent_title = bool(getattr(controller.host, "independent_title", False))
            if "title" in targets and (not seen_title or independent_title):
                seen_title = True
                rows.append((ElementRef(plot_id, "title"),
                             "Title" + suffix if independent_title else
                             "Title (all views)" if self._spec.views_changed is not None
                             and plot_id else "Title", None))
            for side, word in (("bottom", "X axis"), ("left", "Y axis")):
                label, ticks = f"axis.{side}.label", f"axis.{side}.ticks"
                if label not in targets and ticks not in targets:
                    continue
                parent = ElementRef(plot_id, f"axis.{side}")
                rows.append((parent, word + suffix, None))
                if label in targets:
                    rows.append((ElementRef(plot_id, label), "Label", parent))
                if ticks in targets:
                    rows.append((ElementRef(plot_id, ticks), "Numbers", parent))
            for key in targets:
                if key == "colorbar.label":
                    rows.append((ElementRef(plot_id, key), "Colour bar label" + suffix, None))
                elif key.startswith("marginal.") and key.endswith(".label"):
                    side = key.split(".")[1]
                    rows.append((ElementRef(plot_id, key), f"Marginal label ({side})" + suffix,
                                 None))
            entries = [key for key in targets if key.startswith("legend.entry.")]
            legend = getattr(controller.plot, "legend", None) if controller.plot else None
            independent_legend = bool(getattr(controller.host, "independent_legend", False))
            if entries and (not legend_rows or independent_legend):
                legend_hidden = legend is not None and _alive(legend) and not legend.isVisible()
                parent = ElementRef(plot_id, LEGEND)
                legend_rows.append((parent, ("Legend (hidden)" if legend_hidden else "Legend")
                                    + (suffix if independent_legend else ""),
                                    None))
                for key in entries:
                    target = targets[key]
                    name = shown_text(target.style, target.auto) or target.auto or "Unnamed series"
                    legend_rows.append((ElementRef(plot_id, key),
                                        name if len(name) <= 36 else name[:35] + "…", parent))
            for key, text in (("annotation.fit", "Fit label"),
                              ("annotation.limit", "Below-limit label"),
                              ("annotation.provenance", "Caption")):
                if key in targets:
                    tail.append((ElementRef(plot_id, key), text + suffix, None))
            for key in targets:
                if key.startswith("annotation.") and key not in (
                        "annotation.fit", "annotation.limit", "annotation.provenance"):
                    tail.append((ElementRef(plot_id, key), "Annotation" + suffix, None))
        if not legend_rows and self._appearance_has("legend_visible"):
            legend_rows.append((ElementRef("", LEGEND), "Legend", None))
        rows.extend(legend_rows)
        if self._appearance_has("line_width") or self.extras("series") or callable(self._spec.project_sources):
            rows.append((ElementRef("", "series.data"), "Series", None))
        descriptors = self.series_descriptors()
        if descriptors and not any(ref.key == "series.data" for ref,_label,_parent in rows):
            rows.append((ElementRef("", "series.data"), "Series", None))
        for descriptor in descriptors:
            hidden = (self.state().series_styles or {}).get(descriptor.series_id,{}).get("visible") is False
            rows.append((ElementRef("",descriptor.series_id), descriptor.label + (" (hidden)" if hidden else ""),
                         ElementRef("","series.data")))
        # A generated caption can have controls without an editable canvas target
        # (Distributions' provenance switch). It must be reachable even while off.
        if (any(self._extra_visible(widget) for widget in self.extras("caption"))
                and not any(ref.key == "annotation.provenance" for ref, _text, _parent in tail)):
            tail.append((ElementRef("", "annotation.provenance"), "Caption", None))
        rows.extend(tail)
        if any(self._extra_visible(widget) for widget in self.extras(COLOUR_MAP)):
            rows.append((ElementRef("", COLOUR_MAP), "Colour map", None))
        return rows

    @staticmethod
    def _extra_visible(widget) -> bool:
        predicate = getattr(widget, "_nanoscope_format_visible", None)
        return bool(predicate()) if callable(predicate) else True

    def _appearance_has(self, name: str) -> bool:
        # Panels hide rows while retargeting; that does not remove their capabilities.
        return name in self._available_controls

    def _fill_list(self) -> None:
        plan = self._plan()
        self._rows = plan
        blocked = self.list.blockSignals(True)
        try:
            self.list.clear()
            made: dict[ElementRef, QtWidgets.QTreeWidgetItem] = {}
            for ref, text, parent in plan:
                item = QtWidgets.QTreeWidgetItem([text])
                item.setData(0, _ROLE, (ref.plot, ref.key))
                if parent is None or parent not in made:
                    self.list.addTopLevelItem(item)
                else:
                    made[parent].addChild(item)
                made[ref] = item
            made.clear()                      # keys are kept, rows are not
            self.list.expandAll()
            item = self.list.find(self._hub.current())
            if item is not None:
                self.list.setCurrentItem(item)
        finally:
            self.list.blockSignals(blocked)
        self.note.setVisible(not self._controllers())
        self._hub.set_refs(ref for ref, _t, _p in plan)

    def _schedule_rebuild(self) -> None:
        if not self._rebuild_pending:
            self._rebuild_pending = True
            QtCore.QTimer.singleShot(0, self, self._rebuild)

    def _rebuild(self) -> None:
        self._rebuild_pending = False
        if not self._built or self._legacy or self._retired:
            return
        self._sync_controllers()
        self._fill_list()
        current = self._hub.current()
        if current is not None and current not in self._hub.refs():
            fallback = ElementRef(current.plot, LEGEND) if current.key.startswith(
                "legend.entry.") else ElementRef("", FIGURE)
            self._hub.select(fallback if fallback in self._hub.refs()
                             else ElementRef("", FIGURE))

    # -- selection ----------------------------------------------------------------------------
    def select(self, ref: ElementRef | None, *, from_canvas: bool = False) -> None:
        self._ensure_built()
        self._hub.select(ref, from_canvas=from_canvas)

    def show_element(self, ref: ElementRef | str) -> None:
        self._ensure_built()
        if self._legacy:
            return
        if isinstance(ref, str):
            found = next((r for r in self._hub.refs() if r.key == ref), None)
            ref = found if found is not None else ElementRef("", ref)
        self._hub.select(ref)

    def _list_changed(self, current, _previous) -> None:
        ref = self.list.ref_of(current)
        if ref is not None:
            self._hub.select(ref, from_canvas=False)

    def _on_selection(self, ref: ElementRef | None, from_canvas: bool) -> None:
        if not self._built or self._legacy or self._retired:
            return
        blocked = self.list.blockSignals(True)
        try:
            item = self.list.find(ref)
            if item is not None:
                self.list.setCurrentItem(item)
            elif ref is None:
                self.list.setCurrentItem(None)
        finally:
            self.list.blockSignals(blocked)
        self._show_panel(ref if ref is not None else ElementRef("", FIGURE))
        self._update_overlays(ref)
        self._update_footer(ref)
        if from_canvas:
            self.show_inspector_tab(1)

    @QtCore.Slot(str)
    def _on_claimed(self, key: str) -> None:
        """A click on the canvas landed on ``key``: select it and show this tab."""
        controller = self.sender()
        if controller is None or self._retired:
            return
        if not self._canvas_claim_pending:
            self._canvas_claim_pending = True
            QtCore.QTimer.singleShot(0, self, self._finish_canvas_click)
        self._ensure_built()
        ref = ElementRef(self.plot_id(controller), key)
        self._hub.select(ref, from_canvas=True)
        self.show_inspector_tab(1)

    def _finish_canvas_click(self) -> None:
        self._canvas_claim_pending = False

    @QtCore.Slot()
    def _on_cleared(self) -> None:
        """A left click on nothing of ours: the selection ends (the panel stays on Figure)."""
        if self._retired or self._hub.current() is None:
            return
        if self._canvas_claim_pending:
            return              # another controller in this scene saw the same claimed click
        self._hub.clear()

    @QtCore.Slot()
    def _on_editing_changed(self) -> None:
        bar = getattr(self.sender(), "_nanoscope_minibar", None)
        if bar is not None and _alive(bar):
            bar.reposition()

    def show_inspector_tab(self, index: int) -> None:
        """Make ``index`` the inspector's tab (0 Data & analysis, 1 Format): the host is the
        nearest ancestor named ``analysisInspector``. No host (a bare pane) changes nothing."""
        node = self.parentWidget()
        while node is not None:
            if node.objectName() == "analysisInspector" and hasattr(node, "set_tab"):
                node.set_tab(index)
                return
            node = node.parentWidget()

    def axis_log_target(self, ref: ElementRef) -> tuple[str, QtWidgets.QWidget] | None:
        """The adapter names the exact existing control, never inferred from its label."""
        target = self._spec.axis_log_target
        if target is None:
            return None
        return target(ref.plot, ref.key.split(".")[1])

    def reveal_axis_log(self, ref: ElementRef) -> None:
        destination = self.axis_log_target(ref)
        if destination is None:
            return
        target, control = destination
        node = self.parentWidget()
        while node is not None:
            if node.objectName() == "analysisInspector" and hasattr(node, "reveal"):
                node.reveal(target, focus=control)
                return
            node = node.parentWidget()

    # -- panels -------------------------------------------------------------------------------
    def panel_for(self, key: str) -> QtWidgets.QWidget:
        self._ensure_built()
        if self._legacy:
            return self._body
        return self._panel(_kind_of(key))

    def _panel(self, kind: str) -> _Panel:
        panel = self._panels.get(kind)
        if panel is None:
            panel = _PANELS[kind](self)
            self._panels[kind] = panel
            self._stack.addWidget(panel)
            panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                QtWidgets.QSizePolicy.Policy.Ignored)
        return panel

    def _show_panel(self, ref: ElementRef) -> None:
        panel = self._panel(_kind_of(ref.key))
        for other in self._panels.values():
            other.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                QtWidgets.QSizePolicy.Policy.Ignored)
        panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                            QtWidgets.QSizePolicy.Policy.Preferred)
        self._stack.setCurrentWidget(panel)
        panel.show_ref(ref)
        self._stack.updateGeometry()

    def current_panel(self) -> _Panel | None:
        widget = self._stack.currentWidget() if self._built and not self._legacy else None
        return widget if isinstance(widget, _Panel) else None

    # -- keeping current ----------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read the selected element's values (coalesced)."""
        if self._legacy or self._retired:
            return
        self._sync_controllers()
        if self._built:
            self._hub.request_format_changed()

    def _appearance_moved(self, *_args) -> None:
        if not self._retired:
            self._hub.request_format_changed()

    def _on_format_changed(self) -> None:
        if not self._built or self._legacy or self._retired:
            return
        self._sync_controllers()
        self._update_overlays(self._hub.current())
        if not self.isVisible():
            self._stale = True              # reads fresh on its first show
            return
        self._refresh_now()

    def _refresh_now(self) -> None:
        if not self._built:
            return
        if [(r, t, p) for r, t, p in self._plan()] != self._rows:
            self._schedule_rebuild()
        panel = self.current_panel()
        if panel is not None:
            panel.refresh()
        self._update_footer(self._hub.current())

    # -- the outline and the mini toolbar -----------------------------------------------------
    def _update_overlays(self, ref: ElementRef | None) -> None:
        from .format_toolbar import MiniTextToolbar, SelectionOutline

        series_key = ref.key if ref is not None and self.series_descriptor(ref.key) is not None else None
        if callable(self._spec.series_select):
            self._spec.series_select(series_key)
        if series_key is not None:
            for controller in self._controllers():
                bar = getattr(controller, "_nanoscope_minibar", None)
                if bar is not None and _alive(bar):bar.show_key(None)
                plot = getattr(controller, "plot", None)
                owns = series_key in getattr(plot, "_nanoscope_series_targets", {})
                outline = getattr(controller, "_nanoscope_outline", None)
                view = controller.view()
                if owns and view is not None and (outline is None or not _alive(outline)):
                    outline = SelectionOutline(view.viewport());outline.attach(controller)
                    controller._nanoscope_outline = outline
                if outline is not None and _alive(outline):outline.show_key(series_key if owns else None)
            return
        target_controller = self.controller_for(ref) if ref is not None else None
        for controller in self._controllers():
            outline = getattr(controller, "_nanoscope_outline", None)
            bar = getattr(controller, "_nanoscope_minibar", None)
            if controller is not target_controller:
                for overlay in (outline, bar):
                    if overlay is not None and _alive(overlay):
                        overlay.show_key(None)
        if ref is None or target_controller is None:
            return
        key = ref.key
        outlined = key == LEGEND or key.startswith(("axis.", "legend.entry.", "annotation.",
                                                    "colorbar.", "marginal.")) or key == "title"
        view = target_controller.view()
        if view is None:
            return
        if outlined:
            outline = getattr(target_controller, "_nanoscope_outline", None)
            if outline is None or not _alive(outline):
                outline = SelectionOutline(view.viewport())
                outline.attach(target_controller)
                target_controller._nanoscope_outline = outline
            outline.show_key(key)
        else:
            outline = getattr(target_controller, "_nanoscope_outline", None)
            if outline is not None and _alive(outline):
                outline.show_key(None)
        target = target_controller.target(key)
        if target is not None and ({"style", "edit"} & set(target.capabilities)):
            bar = getattr(target_controller, "_nanoscope_minibar", None)
            if bar is None or not _alive(bar):
                bar = MiniTextToolbar(view.viewport())
                bar.attach(target_controller)
                target_controller._nanoscope_minibar = bar
            bar.show_key(key)
        else:
            bar = getattr(target_controller, "_nanoscope_minibar", None)
            if bar is not None and _alive(bar):
                bar.show_key(None)

    # -- writes (every one starts from what the element has NOW) ---------------------------------
    def resolve_text(self, ref: ElementRef):
        """``({prop: (value, source)}, {prop: grey words})`` for a text element."""
        state, defaults = self.state(), self.defaults()
        host = getattr(self.controller_for(ref), "host", None)
        lookup = getattr(host, "storage_key", None)
        storage_key = lookup(ref.key) if callable(lookup) else ref.key
        resolved = {prop: resolve(state, defaults, ref, prop, storage_key=storage_key)
                    for prop in TextStyleFields.PROPS}
        notes = {}
        value, source = resolved["size_pt"]
        if source == "figure" and isinstance(value, (int, float)):
            notes["size_pt"] = f"{_size_role(ref.key)} {value:g} pt"
        return resolved, notes

    def label_locked(self, ref: ElementRef) -> str:
        """Why this axis label's words are not typed on the figure (several views share one)."""
        controller = self.controller_for(ref)
        side = ref.key.split(".")[1] if ref.key.startswith("axis.") else ""
        if controller is None or not side:
            return ""
        try:
            return str(controller.host.axis_state(side).label_locked or "")
        except (RuntimeError, AttributeError, NotImplementedError):
            return ""

    def write_text(self, ref: ElementRef, prop: str, value, *, run: bool = False) -> None:
        controller = self.controller_for(ref)
        if controller is not None:
            controller.set_style_property(ref.key, prop, value, run=run)

    def write_figure(self, **changes) -> None:
        widget = self.widget()
        if widget is not None:
            widget.apply_appearance(replace(widget.state(), **changes))

    def reset_look(self) -> None:
        """Reset look: the figure-wide look and every element style back to the defaults;
        words, pins, annotation places and legend-entry styles stay. One Undo step."""
        widget = self.widget()
        if widget is not None:
            state = widget.state()
            # Visibility is content selection; Reset look clears ink while keeping it.
            series = {key: {"visible": value["visible"]}
                      for key, value in (state.series_styles or {}).items() if "visible" in value}
            if state.style_binding:
                from ..core.figure_spec import FrozenMap, thaw_json
                binding = thaw_json(state.style_binding)
                binding["overrides"] = {}
                text = tuple((key, style if key.startswith("legend.entry.") else
                              TextStyle(text=style.text)) for key, style in (state.text or ()))
                label = replace(state.fit_label, size_pt=None, color=None) if state.fit_label else None
                widget.apply_appearance(replace(state, style_binding=FrozenMap(binding),
                                                text=text or None, fit_label=label,
                                                series_styles=series or None))
            else:
                widget.apply_appearance(replace(state.restyled(PlotAppearance()), series_styles=series or None))

    def _write_styles(self, controller, styles: dict[str, TextStyle], *,
                      auto_range_of: str = "") -> None:
        """Write several elements' styles (and, for ``auto_range_of`` a side, take that axis'
        pinned range and major tick off) as ONE step, using the page's bulk writer or its
        appearance widget. Older hosts without either keep the single-element route."""
        widget = self.widget()
        host = getattr(controller, "host", None)
        apply_many = getattr(host, "apply_format_styles", None)
        if callable(apply_many):
            apply_many(styles, auto_range_of=auto_range_of)
            return
        if widget is not None and getattr(host, "widget", None) is widget:
            state = widget.state()
            for key, style in styles.items():
                lookup = getattr(host, "storage_key", None)
                state = state.with_text_style(lookup(key) if callable(lookup) else key, style)
            if auto_range_of:
                state = state.without_axis_pin(host.pin_view, auto_range_of,
                                               primary=host.primary_view)
            widget.apply_appearance(state)
            return
        for key, style in styles.items():
            controller.apply_style(key, style)
        if auto_range_of:
            state = controller.host.axis_state(auto_range_of)
            if state.range is not None or state.tick_major is not None:
                controller._apply_axis(auto_range_of, replace(state, range=None,
                                                              tick_major=None))

    # -- footer -------------------------------------------------------------------------------
    def _update_footer(self, ref: ElementRef | None) -> None:
        if not self._built or self._legacy:
            return
        ref = ref if ref is not None else ElementRef("", FIGURE)
        kind = _kind_of(ref.key)
        target = self.target_for(ref) if kind in ("text", "fit_label", "caption") else None
        styled = target is not None and "style" in target.capabilities
        copyable = styled or kind in ("axis", "figure") or self.series_descriptor(ref.key) is not None
        held = self._clipboard.get()
        accepts = isinstance(held, _Copied) and (
            (held.kind == "text" and styled) or (held.kind == "axis" and kind == "axis")
            or (held.kind == "figure" and kind == "figure")
            or (held.kind == "series" and self.series_descriptor(ref.key) is not None))
        self.copy_button.setEnabled(copyable)
        self.paste_button.setEnabled(bool(accepts))
        self.reset_button.setEnabled(styled or kind in ("axis", "figure", "legend", "series"))
        self.reset_button.setText("Reset to style" if self.state().style_binding else "Reset element")

    def copy_format(self, _checked: bool = False) -> None:
        """Hold the selected element's look in memory (never its words)."""
        ref = self._hub.current() or ElementRef("", FIGURE)
        kind = _kind_of(ref.key)

        def copied_text(key, target):
            if target is None:
                return None
            resolved, _notes = self.resolve_text(ElementRef(ref.plot, key))
            values = {prop: value for prop, (value, _origin) in resolved.items()
                      if value is not None}
            return replace(target.style, text=None, **values)
        if kind == "figure":
            look = self.state().style_only()
            # The fit label's number switches and visibility are not a figure look.
            text = tuple((key, style) for key, style in (look.text or ())
                         if key != "annotation.fit") or None
            self._clipboard.put(_Copied("figure", look=replace(look, fit_label=None, text=text),
                                        vocabulary=self.widget().style_vocabulary))
        elif kind == "series" and self.series_descriptor(ref.key) is not None:
            descriptor = self.series_descriptor(ref.key)
            own = dict((self.state().series_styles or {}).get(ref.key,{}))
            from ..core.figure_style import logical_pixels_to_points
            width = self.state().line_width
            marker = self.state().marker_size
            # Native appearance sizes are logical pixels; series overrides use points.
            if self.widget().style_vocabulary in ("plot_appearance", "style_settings"):
                width = logical_pixels_to_points(width) if width is not None else None
                marker = logical_pixels_to_points(marker) if marker is not None else None
            drawn = descriptor.extra.get("effective_style", {})
            effective = {"color":own.get("color") or drawn.get("color") or descriptor.color,
                "line_width_pt":own.get("line_width_pt",drawn.get("line_width_pt",width)),
                "marker_size_pt":own.get("marker_size_pt",drawn.get("marker_size_pt",marker))}
            self._clipboard.put(_Copied("series",series={k:v for k,v in effective.items() if v is not None and k in descriptor.capabilities}))
        elif kind == "axis":
            controller = self.controller_for(ref)
            side = ref.key.split(".")[1]
            label = controller.target(f"axis.{side}.label") if controller else None
            ticks = controller.target(f"axis.{side}.ticks") if controller else None
            if label is None and ticks is None:
                return
            self._clipboard.put(_Copied(
                "axis", label=copied_text(f"axis.{side}.label", label),
                ticks=copied_text(f"axis.{side}.ticks", ticks)))
        else:
            target = self.target_for(ref)
            if target is None or "style" not in target.capabilities:
                return
            style = copied_text(ref.key, target)
            if "align" not in target.capabilities:
                style = replace(style, align=None)
            self._clipboard.put(_Copied("text", style=style))
        self._update_footer(ref)

    def paste_format(self, _checked: bool = False) -> None:
        """Apply the copied look to the selected element. Its own style is REBUILT from its
        current one, never replaced, so a typed title, label wording or legend name stays."""
        held = self._clipboard.get()
        ref = self._hub.current() or ElementRef("", FIGURE)
        kind = _kind_of(ref.key)
        if not isinstance(held, _Copied):
            return
        if held.kind == "figure" and kind == "figure":
            widget = self.widget()
            if widget is not None and held.look is not None:
                before = widget.state()
                look = held.look
                if not look.style_binding and held.vocabulary != widget.style_vocabulary:
                    from ..core.figure_style_mapping import import_legacy_look, export_legacy_look
                    physical = import_legacy_look(look.to_dict(), held.vocabulary).look
                    look = PlotAppearance.from_mapping(export_legacy_look(physical,
                        widget.style_vocabulary, look.to_dict(), _authoritative=True))
                after = before.restyled(look, vocabulary=widget.style_vocabulary)
                # Figure Paste preserves the destination fit label in its entirety.
                after = after.with_text_style("annotation.fit", before.text_style("annotation.fit"))
                widget.apply_appearance(replace(after, fit_label=before.fit_label))
        elif held.kind == "series" and self.series_descriptor(ref.key) is not None:
            self.write_series(ref.key, held.series or {})
        elif held.kind == "axis" and kind == "axis":
            controller = self.controller_for(ref)
            side = ref.key.split(".")[1]
            styles = {}
            for key, copied in ((f"axis.{side}.label", held.label), (f"axis.{side}.ticks", held.ticks)):
                target = controller.target(key) if controller else None
                if target is not None and copied is not None:
                    styles[key] = replace(copied, text=target.style.text, align=None)
            if controller is not None and styles:
                self._write_styles(controller, styles)
        elif held.kind == "text" and held.style is not None and kind in (
                "text", "fit_label", "caption"):
            controller = self.controller_for(ref)
            target = controller.target(ref.key) if controller else None
            if target is not None and "style" in target.capabilities:
                new = replace(held.style, text=target.style.text)
                if "align" not in target.capabilities:
                    new = replace(new, align=None)
                controller.apply_style(ref.key, new)

    def reset_element(self, _checked: bool = False) -> None:
        """Clear what the selected element set on its own, as one Undo step. Words stay."""
        ref = self._hub.current() or ElementRef("", FIGURE)
        kind = _kind_of(ref.key)
        if kind == "figure":
            self.reset_look()
        elif kind == "legend":
            self.write_figure(legend_visible=None, legend_size=None)
        elif kind == "series":
            descriptor = self.series_descriptor(ref.key)
            if descriptor is not None:
                self.write_series(ref.key,{field:None for field in descriptor.capabilities})
            else:
                self.write_figure(line_width=None, marker_size=None, data_color=None, fit_color=None)
        elif kind == "axis":
            controller = self.controller_for(ref)
            if controller is None:
                return
            side = ref.key.split(".")[1]
            styles = {}
            for key in (f"axis.{side}.label", f"axis.{side}.ticks"):
                target = controller.target(key)
                if target is not None:
                    styles[key] = TextStyle(text=target.style.text)
            self._write_styles(controller, styles, auto_range_of=side)
        else:
            controller = self.controller_for(ref)
            target = controller.target(ref.key) if controller else None
            if target is not None and "style" in target.capabilities:
                controller.apply_style(ref.key, TextStyle(text=target.style.text))
