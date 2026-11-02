"""In-figure editing of every piece of text in a pyqtgraph plot, and of its axes.

ONE MODULE, EVERY LIVE PLOT WINDOW. Double-click a title, an axis label, a legend entry or an
annotation (the fit label, a colour-bar label) to retype it in place; double-click the tick
area for Axis properties; right-click any of them for Edit…, Text style…, Automatic and Hide,
which join the page's own app menu (``plot_interaction``). Each commit is ONE step of the
window's appearance Undo, because the controller never stores anything: it hands the new
:class:`TextStyle` to the window's :class:`FigureTextHost`, and the window writes it where its
appearance already lives (``PlotAppearance.text`` through :func:`appearance_text_host`, or
Distributions' own ``StyleSettings``).

NO PYTHON EVENT FILTER ON ANY GRAPHICS VIEW. An ``eventFilter`` on the view made the PySide
construction segfault family about four times as frequent (``plot_interaction.watch_exit``
says how it was measured, and it was measured again on 2026-11-01). Every entry point here is
pyqtgraph's own Python dispatch or a Qt signal:

* the title, the axes and the legend get a per-INSTANCE ``mouseClickEvent`` wrap, holding the
  controller by weak reference and calling the original for every event it does not consume
  (the AxisItem forwards clicks to its ViewBox; the ViewBox raises the app menu);
* annotations are our own item subclasses (``fit_label.FitLabelItem``,
  :class:`EditableLabelItem`) that override ``mouseClickEvent`` and say so with signals;
* the legend's Qt double click arrives as ``LegendItem.sigDoubleClicked``.

THE DOUBLE CLICK IS A GATE, NOT ``ev.double()``. pyqtgraph 0.14 turns Qt's double click into
``MouseClickEvent(double=True)`` only for the order Press, Release, DblClick, Release. For
Qt 6's Press, Release, Press, DblClick, Release the scene forwards the second PRESS as a
single click and drops the double (``GraphicsScene.mouseReleaseEvent`` sends
``clickEvents[0]``), so :class:`DoubleClickGate` also takes two single clicks on one element
within the platform's interval and drag distance. Each pair is taken once, whichever route
reports it.

The inline editor is a ``QLineEdit`` child of the viewport; the popovers are ``Qt.Popup``
frames parented to the window. Each is deleted with ``deleteLater`` after it commits or
cancels, and every later reference goes through :func:`_alive`, because pages call
``plot.clear()`` (and close) while an editor may still be open.

TYPED TEXT IS LITERAL. :func:`rich_text` escapes it for Qt's rich text, so ``I < 5 pA & τ > 1
ms`` draws as typed; ``_{…}`` and ``^{…}`` are the one markup, for sub- and superscripts.
"""
from __future__ import annotations

import html
import math
import re
import weakref
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol

import pyqtgraph as pg
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from ..plot import figure_fonts as ff
from .font_picker import FontPicker, html_font_family

__all__ = [
    "AXIS_SIDES", "HIDDEN_TEXT", "AxisPin", "AxisPropertiesPopover", "AxisState",
    "DoubleClickGate", "EditableLabelItem", "FigureTextController", "FigureTextHost",
    "InlineTextEditor", "TextStyle", "TextStylePopover", "TextTarget", "apply_text_style",
    "appearance_text_host", "attach_figure_text", "css_args", "figure_text_for",
    "label_args", "rich_text", "shown_text", "valid_element_key", "view_pin_key",
]

#: A text that draws nothing: what a box holding only spaces stores (``HIDDEN_LABEL`` in
#: ``plot_appearance``, the same value). ``None`` is the window's automatic text.
HIDDEN_TEXT = " "
AXIS_SIDES = ("bottom", "left", "top", "right")

#: The element keys a style may be stored under. Anything else in a saved map is dropped,
#: as ``PlotAppearance.from_mapping`` drops any other bad field.
_ELEMENT_KEY = re.compile(
    r"^(title|view\.[\w-]{1,40}\.title|axis\.(bottom|left|top|right)\.(label|ticks)"
    r"|legend\.entry\..{1,200}|colorbar\.label|marginal\.(bottom|left|top|right)\.label"
    r"|annotation\.[\w.-]{1,60})$", re.DOTALL)

#: Colours offered in the popovers: the Okabe-Ito set the presets already use, so a
#: styled label matches the data and fit colours. A modal QColorDialog would close the
#: popup it was opened from, so the popover offers these and a typed hex code instead.
SWATCHES = ("#000000", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9",
            "#7f7f7f")


def valid_element_key(key: Any) -> bool:
    return isinstance(key, str) and bool(_ELEMENT_KEY.match(key))


def _clean_text(value: Any) -> str | None:
    """The Appearance boxes' rule: empty is automatic (None), only spaces is hidden."""
    if not isinstance(value, str):
        return None
    if value.strip():
        return value.strip()[:1000]
    return HIDDEN_TEXT if value else None


@dataclass(frozen=True)
class TextStyle:
    """One text element's overrides. Every field None = inherit: the element override wins,
    then the figure-wide field (``font_family``, ``title_size``…), then the window default."""

    text: str | None = None          # None automatic, " " hidden, else the user's literal text
    family: str | None = None
    size_pt: float | None = None
    bold: bool | None = None
    italic: bool | None = None
    color: str | None = None
    align: str | None = None         # title only: left | center | right

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def is_default(self) -> bool:
        return self == TextStyle()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "TextStyle":
        """Keep the valid fields only; a malformed one is dropped, never an error."""
        values = values if isinstance(values, Mapping) else {}
        valid: dict[str, Any] = {}
        text = _clean_text(values.get("text"))
        if text is not None:
            valid["text"] = text
        family = values.get("family")
        if isinstance(family, str) and family.strip() and not any(c in family for c in ";{}<>'\""):
            valid["family"] = family.strip()[:200]
        size = values.get("size_pt")
        if (isinstance(size, (int, float)) and not isinstance(size, bool) and math.isfinite(size)
                and 4 <= size <= 72):
            valid["size_pt"] = float(size)
        for key in ("bold", "italic"):
            if isinstance(values.get(key), bool):
                valid[key] = values[key]
        color = values.get("color")
        if isinstance(color, str) and QtGui.QColor(color).isValid():
            valid["color"] = QtGui.QColor(color).name()
        if values.get("align") in ("left", "center", "right"):
            valid["align"] = values["align"]
        return cls(**valid)


def shown_text(style: TextStyle, auto: str) -> str:
    """What the element draws, as plain text: the user's, the automatic, or nothing."""
    if style.text is None:
        return auto
    return "" if style.text == HIDDEN_TEXT else style.text


_SUBSUP = re.compile(r"([_^])\{([^{}]*)\}")


def rich_text(user_text: str) -> str:
    """``user_text`` for Qt's rich text: escaped, with ``_{…}``/``^{…}`` as sub/superscript.

    Escaped FIRST, so the markup can only ever produce the two tags it names: ``a_{<b>}``
    is a subscript ``<b>`` drawn literally, never a bold tag.
    """
    text = html.escape(str(user_text), quote=False)
    text = _SUBSUP.sub(lambda m: f"<{'sub' if m.group(1) == '_' else 'sup'}>{m.group(2)}"
                       f"</{'sub' if m.group(1) == '_' else 'sup'}>", text)
    return text.replace("\n", "<br>")


_TAG = re.compile(r"<[^>]*>")
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def plain_text(rich: str) -> str:
    """The text a rich-text string draws (automatic labels carry tags and entities).

    Read the way Qt's rich text reads it (tags dropped, white space collapsed and trimmed per
    line, ``<br>`` a new line, entities decoded), in Python: a QTextDocument per call, on every
    controller refresh, was the one new native crash site in the 2026-11-01 crash-rate runs.
    """
    lines = []
    for part in _BREAK.split(str(rich or "")):
        line = " ".join(_TAG.sub("", part).split())
        lines.append(html.unescape(line).replace("\xa0", " "))
    return "\n".join(lines)


def css_args(style: TextStyle) -> dict[str, str]:
    """The style as the CSS an AxisItem's ``labelStyle`` takes."""
    css: dict[str, str] = {}
    if style.color:
        css["color"] = style.color
    if style.size_pt:
        css["font-size"] = f"{style.size_pt:g}pt"
    if style.family:
        css["font-family"] = html_font_family(style.family)
    if style.bold is not None:
        css["font-weight"] = "bold" if style.bold else "normal"
    if style.italic is not None:
        css["font-style"] = "italic" if style.italic else "normal"
    return css


def label_args(style: TextStyle) -> dict[str, Any]:
    """The style as ``LabelItem.setText`` keywords."""
    args: dict[str, Any] = {}
    if style.color:
        args["color"] = style.color
    if style.size_pt:
        args["size"] = f"{style.size_pt:g}pt"
    if style.family:
        args["family"] = html_font_family(style.family)
    if style.bold is not None:
        args["bold"] = style.bold
    if style.italic is not None:
        args["italic"] = style.italic
    if style.align:
        args["justify"] = style.align
    return args


def apply_text_style(item, style: TextStyle, auto: str, family: str | None = None) -> None:
    """Draw ``style`` on a window's own text item (a colour-bar or marginal label, an
    annotation): a LabelItem, a TextItem or an AxisItem's label. ``auto`` is the window's
    automatic text, as rich text; the user's text is escaped. The item's family is the
    style's own, else ``family`` (the figure's), else Arial."""
    text = auto if style.text is None else ("" if style.text == HIDDEN_TEXT
                                           else rich_text(style.text))
    css_family = html_font_family(style.family or family)
    if isinstance(item, pg.AxisItem):
        css = dict(item.labelStyle)
        css["font-family"] = css_family
        css.update(css_args(style))
        item.labelText = text
        item.labelStyle = css
        item._updateLabel()
    elif isinstance(item, pg.LabelItem):
        item.setText(text, **{**item.opts, "family": css_family, **label_args(style)})
        item.setVisible(bool(text.strip()))
    elif isinstance(item, pg.TextItem):
        css = "; ".join(f"{k}: {v}" for k, v in {"font-family": css_family,
                                                  **css_args(style)}.items())
        item.setHtml(f"<span style='{css}'>{text}</span>" if css else text)
        item.setVisible(bool(text.strip()))


@dataclass(frozen=True)
class AxisPin:
    """The saved per-axis state of ``PlotAppearance.axes``, in DISPLAY units.

    ``min``/``max`` pin the range (both or neither); ``tick_major`` is the major tick
    spacing (decades on a log axis); ``log`` is kept for a window whose own log control
    stores it here. None everywhere = automatic.
    """

    min: float | None = None
    max: float | None = None
    log: bool | None = None
    tick_major: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def is_default(self) -> bool:
        return self == AxisPin()

    def range(self) -> tuple[float, float] | None:
        return None if self.min is None or self.max is None else (self.min, self.max)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "AxisPin":
        values = values if isinstance(values, Mapping) else {}

        def number(key):
            value = values.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                return float(value)
            return None

        lo, hi, major = number("min"), number("max"), number("tick_major")
        valid: dict[str, Any] = {}
        if lo is not None and hi is not None and lo < hi:
            valid.update(min=lo, max=hi)
        if major is not None and major > 0:
            valid["tick_major"] = major
        if isinstance(values.get("log"), bool):
            valid["log"] = values["log"]
        return cls(**valid)


@dataclass(frozen=True)
class AxisState:
    """What Axis properties shows and commits for one axis. ``label`` follows the
    TextStyle text rule (None automatic, " " hidden); ``range`` None is Auto."""

    label: str | None = None
    label_auto: str = ""
    units: str = ""
    range: tuple[float, float] | None = None
    shown_range: tuple[float, float] = (0.0, 1.0)
    log: bool = False
    log_available: bool = False
    log_reason: str = "This window has no log scale for this axis."
    tick_pt: float | None = None
    tick_major: float | None = None
    #: False where the window keeps no major tick spacing: the row is not offered, rather
    #: than taking a value that changes nothing.
    tick_major_available: bool = True
    #: Why the label cannot be typed here (one label shared by several views), or "".
    label_locked: str = ""


@dataclass(frozen=True)
class TextTarget:
    """One editable element: the item a click lands on, its automatic text (plain), its
    current style, the word the menu uses for it, and what may be done to it."""

    item: Any
    auto: str = ""
    style: TextStyle = TextStyle()
    noun: str = "text"
    capabilities: frozenset = field(default_factory=lambda: frozenset({"edit", "style", "hide"}))


class FigureTextHost(Protocol):
    """What a window gives :func:`attach_figure_text`. Every write is ONE Undo step."""

    def text_targets(self) -> dict[str, TextTarget]: ...

    def apply_text(self, key: str, style: TextStyle) -> None: ...

    def axis_state(self, side: str) -> AxisState: ...

    def apply_axis(self, side: str, values: AxisState) -> None: ...

    def menu_entries(self, key: str) -> list: ...   # optional extra (title, callback)

    def background_entries(self) -> list: ...   # optional: a right-click on the plot itself


def _alive(obj) -> bool:
    """True while ``obj``'s C++ side exists (a page's clear or close deletes it)."""
    try:
        return obj is not None and shiboken6.isValid(obj)
    except (RuntimeError, TypeError):
        return False


# -- the double-click gate -------------------------------------------------------------------
class DoubleClickGate:
    """Is this left click the second of a double click? ``ev.double()``, OR two single clicks
    on the same element within the platform's double-click interval and drag distance.
    Each pair is taken once, whichever route reports it (see the module docstring)."""

    def __init__(self) -> None:
        self._last = None          # (key, time, scene pos) of the last single click
        self._consumed: dict = {}  # key -> time a double was taken for it

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def _interval() -> float:
        return QtWidgets.QApplication.doubleClickInterval() / 1000.0

    def __call__(self, ev, key) -> bool:
        now = float(ev.time()) if hasattr(ev, "time") else 0.0
        if ev.double():
            return self._take(key, now)
        last, self._last = self._last, (key, now, ev.scenePos())
        if last is None or last[0] != key or now - last[1] > self._interval():
            return False
        moved = ev.scenePos() - last[2]
        if abs(moved.x()) + abs(moved.y()) > QtWidgets.QApplication.startDragDistance():
            return False
        return self._take(key, now)

    def _take(self, key, now: float) -> bool:
        if now - self._consumed.get(key, -1e9) < self._interval():
            return False           # this pair was already taken by the other route
        self._consumed[key] = now
        self._last = None
        return True

    def take(self, key) -> bool:
        """A double reported by another route (``LegendItem.sigDoubleClicked``)."""
        import time

        return self._take(key, time.perf_counter())


#: One gate for the process: a double click is one pair of clicks wherever it lands.
GATE = DoubleClickGate()


class EditableLabelItem(pg.LabelItem):
    """A window's own annotation (a provenance caption, a panel letter) that asks to be
    edited: register it with :meth:`FigureTextController.refresh` through the host."""

    editRequested = QtCore.Signal()
    menuRequested = QtCore.Signal(object)

    def mouseClickEvent(self, ev):  # noqa: N802 - pyqtgraph API, our own subclass
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and GATE(ev, ("label", id(self))):
            ev.accept()
            self.editRequested.emit()
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.menuRequested.emit(ev.screenPos())
        else:
            ev.ignore()


# -- the editors ---------------------------------------------------------------------------
class InlineTextEditor(QtWidgets.QLineEdit):
    """A line editor laid over the text. Return or leaving it commits; Escape cancels."""

    committed = QtCore.Signal(str, str)     # key, text
    cancelled = QtCore.Signal(str)

    def __init__(self, key: str, text: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(text, parent)
        self.key = key
        self.initial = text
        self._done = False
        self.setObjectName("plotInlineTextEditor")
        self.setAccessibleName(f"Edit {key}")
        self.setPlaceholderText("Empty = automatic · a space hides it")
        self.setToolTip("Return keeps it, Esc cancels. Empty = automatic, one space hides it. "
                        "_{…} and ^{…} make sub- and superscripts.")
        self.selectAll()
        self.editingFinished.connect(self._finish)

    def keyPressEvent(self, ev):  # noqa: N802 - our own widget, not a view filter
        if ev.key() == QtCore.Qt.Key.Key_Escape:
            self.cancel()
            return
        super().keyPressEvent(ev)

    def cancel(self) -> None:
        if self._done:
            return
        self._done = True
        self.cancelled.emit(self.key)
        self.hide()
        self.deleteLater()

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self.hide()
        self.committed.emit(self.key, self.text())
        self.deleteLater()


class _Popover(QtWidgets.QFrame):
    def __init__(self, parent) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self._done = False

    def closeEvent(self, ev):  # noqa: N802 - our own widget
        super().closeEvent(ev)
        if not self._done:
            self._done = True
            self.deleteLater()

    def finish(self) -> None:
        self._done = True
        self.hide()
        self.deleteLater()


class _ColorRow(QtWidgets.QWidget):
    """Default, the swatches, and a hex box. No modal dialog: it would close the popup."""

    def __init__(self, value: str | None, parent=None) -> None:
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.default = QtWidgets.QToolButton(text="Auto")
        self.default.setToolTip("The window's own colour")
        self.default.clicked.connect(lambda: self.set_value(None))
        row.addWidget(self.default)
        for colour in SWATCHES:
            button = QtWidgets.QToolButton()
            button.setFixedSize(18, 18)
            button.setStyleSheet(f"background: {colour}; border: 1px solid #888;")
            button.setToolTip(colour)
            button.clicked.connect(lambda _=False, c=colour: self.set_value(c))
            row.addWidget(button)
        self.hex = QtWidgets.QLineEdit()
        self.hex.setPlaceholderText("#rrggbb")
        self.hex.setMaximumWidth(80)
        row.addWidget(self.hex)
        self.set_value(value)

    def set_value(self, value: str | None) -> None:
        self.hex.setText(value or "")

    def value(self) -> str | None:
        text = self.hex.text().strip()
        return QtGui.QColor(text).name() if text and QtGui.QColor(text).isValid() else None


class TextStylePopover(_Popover):
    """Text, font, size, bold, italic, colour and (title only) alignment of one element."""

    committed = QtCore.Signal(str, object)

    def __init__(self, key: str, style: TextStyle, auto: str, parent=None, *, align=False,
                 editable=True) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName("plotTextStylePopover")
        form = QtWidgets.QFormLayout(self)
        self.text = QtWidgets.QLineEdit("" if style.text is None else style.text)
        self.text.setPlaceholderText(f"Automatic: {auto}" if auto else "Automatic")
        if editable:
            form.addRow("Text", self.text)
        else:
            self.text.hide()
        self.use_family = QtWidgets.QCheckBox("Custom font")
        self.use_family.setChecked(bool(style.family))
        # "Arial" is a family here, not "unset": a text that follows the figure's font is this
        # box unticked, so the first row stores Arial itself.
        self.family = FontPicker(first_row=(ff.DEFAULT_FAMILY, ff.DEFAULT_FAMILY))
        # Not the picker's own sentence ("The first row is the figure default"): here the first
        # row is an explicit Arial, and the box counts only while Custom font is ticked.
        self.family.setToolTip(
            "Font for this text, used only while Custom font is ticked. The first row is Arial "
            "itself, whatever the figure's font is; untick Custom font to follow the figure's "
            f"font. {ff.OTHER_FONT_LABEL} lists every font installed on this computer except the "
            "system's private fonts and symbol fonts such as Wingdings.")
        self.family.set_value(style.family)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.use_family)
        row.addWidget(self.family, 1)
        form.addRow("Font", row)
        self.size = QtWidgets.QDoubleSpinBox()
        self.size.setRange(0, 72)
        self.size.setDecimals(1)
        self.size.setSpecialValueText("Default")
        self.size.setSuffix(" pt")
        self.size.setValue(style.size_pt or 0)
        self.bold = QtWidgets.QToolButton(text="B", checkable=True)
        self.bold.setAccessibleName("Bold")
        self.bold.setChecked(bool(style.bold))
        self.italic = QtWidgets.QToolButton(text="I", checkable=True)
        self.italic.setAccessibleName("Italic")
        self.italic.setChecked(bool(style.italic))
        srow = QtWidgets.QHBoxLayout()
        for widget in (self.size, self.bold, self.italic):
            srow.addWidget(widget)
        form.addRow("Size / style", srow)
        self.color = _ColorRow(style.color)
        form.addRow("Colour", self.color)
        self.align = QtWidgets.QComboBox()
        for label, value in (("Automatic", None), ("Left", "left"), ("Center", "center"),
                             ("Right", "right")):
            self.align.addItem(label, value)
        self.align.setCurrentIndex(max(0, self.align.findData(style.align)))
        if align:
            form.addRow("Alignment", self.align)
        else:
            self.align.hide()
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.reset_button = buttons.addButton("Reset text style",
                                              QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.close)
        self.reset_button.clicked.connect(self.reset_style)
        form.addRow(buttons)
        self._text_kept = style.text

    def value(self) -> TextStyle:
        return TextStyle.from_mapping({
            "text": self.text.text() if self.text.isVisibleTo(self) else self._text_kept,
            "family": self.family.value() if self.use_family.isChecked() else None,
            "size_pt": self.size.value() or None,
            "bold": True if self.bold.isChecked() else None,
            "italic": True if self.italic.isChecked() else None,
            "color": self.color.value(), "align": self.align.currentData()})

    def accept(self) -> None:
        value = self.value()
        self.finish()
        self.committed.emit(self.key, value)

    def reset_style(self) -> None:
        """Every style back to automatic; the text stays the user's."""
        value = TextStyle(text=self._text_kept if not self.text.isVisibleTo(self)
                          else TextStyle.from_mapping({"text": self.text.text()}).text)
        self.finish()
        self.committed.emit(self.key, value)


class AxisPropertiesPopover(_Popover):
    """Label, units (read-only), range with Auto, log, tick label size and major tick
    spacing of one axis, typed in the units the axis shows."""

    committed = QtCore.Signal(str, object)

    def __init__(self, side: str, state: AxisState, parent=None) -> None:
        super().__init__(parent)
        self.side = side
        self._state = state
        self.setObjectName("plotAxisPropertiesPopover")
        form = QtWidgets.QFormLayout(self)
        self.label = QtWidgets.QLineEdit("" if state.label is None else state.label)
        self.label.setPlaceholderText(f"Automatic: {state.label_auto}" if state.label_auto
                                      else "Automatic")
        if state.label_locked:
            self.label.setReadOnly(True)
            self.label.setToolTip(state.label_locked)
        form.addRow("Label", self.label)
        self.units = QtWidgets.QLineEdit(state.units)
        self.units.setReadOnly(True)        # units belong to the data, not the style
        self.units.setToolTip("The unit is the data's. Change the time unit in Data to "
                              "rescale; a custom label replaces the whole text.")
        form.addRow("Units", self.units)
        lo, hi = state.range if state.range is not None else state.shown_range
        self.lo, self.hi = QtWidgets.QLineEdit(f"{lo:.6g}"), QtWidgets.QLineEdit(f"{hi:.6g}")
        self.auto = QtWidgets.QCheckBox("Auto")
        self.auto.setChecked(state.range is None)
        self.auto.toggled.connect(self._sync)
        rrow = QtWidgets.QHBoxLayout()
        for widget in (self.lo, QtWidgets.QLabel("to"), self.hi, self.auto):
            rrow.addWidget(widget)
        form.addRow("Range", rrow)
        self.log = QtWidgets.QCheckBox("Log scale")
        self.log.setChecked(state.log)
        self.log.setEnabled(state.log_available)
        if not state.log_available:
            self.log.setToolTip(state.log_reason)
        form.addRow("", self.log)
        self.tick_size = QtWidgets.QDoubleSpinBox()
        self.tick_size.setRange(0, 30)
        self.tick_size.setDecimals(1)
        self.tick_size.setSpecialValueText("Default")
        self.tick_size.setValue(state.tick_pt or 0)
        form.addRow("Tick labels (pt)", self.tick_size)
        self.major = QtWidgets.QDoubleSpinBox()
        self.major.setRange(0, 1e12)
        self.major.setDecimals(6)
        self.major.setSpecialValueText("Automatic")
        self.major.setValue(state.tick_major or 0)
        self.major.setToolTip("Spacing of the labelled ticks, in the axis' units (decades on "
                              "a log axis). Automatic lets the axis choose.")
        form.addRow("Major tick every", self.major)
        form.setRowVisible(self.major, state.tick_major_available)
        self.error = QtWidgets.QLabel("")
        self.error.setStyleSheet("color: #b00020;")
        self.error.hide()
        form.addRow(self.error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.close)
        form.addRow(buttons)
        self._sync()

    def _sync(self) -> None:
        for widget in (self.lo, self.hi):
            widget.setEnabled(not self.auto.isChecked())

    def value(self) -> AxisState | None:
        rng = None
        if not self.auto.isChecked():
            try:
                lo, hi = float(self.lo.text()), float(self.hi.text())
            except ValueError:
                return None
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                return None
            if self.log.isChecked() and lo <= 0:
                return None
            rng = (lo, hi)
        return replace(self._state, label=_clean_text(self.label.text()), range=rng,
                       log=self.log.isChecked(), tick_pt=self.tick_size.value() or None,
                       tick_major=self.major.value() or None)

    def accept(self) -> None:
        value = self.value()
        if value is None:
            self.error.setText("Type a range whose minimum is below its maximum "
                               "(and above 0 on a log axis).")
            self.error.show()
            return
        self.finish()
        self.committed.emit(self.side, value)


# -- the controller ------------------------------------------------------------------------
class FigureTextController(QtCore.QObject):
    """Every in-figure entry point of ONE PlotItem. It stores nothing: the host does.

    ``opened`` says which editor opened for which key (``"inline"``, ``"style"``,
    ``"axis"``), so a test can count them.
    """

    opened = QtCore.Signal(str, str)

    def __init__(self, plot: pg.PlotItem, host: FigureTextHost) -> None:
        super().__init__(plot.getViewBox())
        self._plot_ref = weakref.ref(plot)
        self.host = host
        self.editor: InlineTextEditor | None = None
        self.popover: QtWidgets.QFrame | None = None
        self.last_menu: QtWidgets.QMenu | None = None
        self._legend = None

    # -- plumbing ---------------------------------------------------------------------------
    @property
    def plot(self) -> pg.PlotItem | None:
        plot = self._plot_ref()
        return plot if _alive(plot) else None

    def _view(self) -> QtWidgets.QGraphicsView | None:
        plot = self.plot
        if plot is None or plot.scene() is None or not plot.scene().views():
            return None
        return plot.scene().views()[0]

    def targets(self) -> dict[str, TextTarget]:
        try:
            return dict(self.host.text_targets())
        except RuntimeError:     # an item of a page being cleared
            return {}

    def target(self, key: str) -> TextTarget | None:
        return self.targets().get(key)

    def install(self) -> "FigureTextController":
        plot = self.plot
        if plot is None:
            return self
        self._wrap(plot.titleLabel, lambda ev: "title")
        plot.titleLabel.setCursor(QtCore.Qt.CursorShape.IBeamCursor)
        for side in AXIS_SIDES:
            self._wrap_axis(plot.getAxis(side), side)
        self.refresh()
        return self

    def refresh(self) -> None:
        """Pick up what the window added since: a legend, annotations (the fit label)."""
        plot = self.plot
        if plot is None:
            return
        legend = getattr(plot, "legend", None)
        if legend is not None and legend is not self._legend and _alive(legend):
            self._legend = legend
            self._wrap(legend, self._legend_key)
            legend.sigDoubleClicked.connect(self._legend_double)
        ref = weakref.ref(self)
        for key, target in self.targets().items():
            item = target.item
            if not _alive(item) or not hasattr(item, "editRequested"):
                continue
            if getattr(item, "_nanoscope_text_controller", None) is ref:
                continue
            item._nanoscope_text_controller = ref
            item.editRequested.connect(lambda *_a, k=key, r=ref: r() and r()._edit_or_style(k))
            if hasattr(item, "menuRequested"):
                item.menuRequested.connect(
                    lambda pos, k=key, r=ref: r() and r()._raise_menu_at(k, pos))
            item.setCursor(QtCore.Qt.CursorShape.OpenHandCursor
                           if hasattr(item, "moved") else QtCore.Qt.CursorShape.IBeamCursor)

    def _wrap(self, item, resolve: Callable) -> None:
        """Wrap ``item.mouseClickEvent`` on this instance: double click edits, right click
        adds the text entries to the app menu, anything else goes to the original."""
        if getattr(item, "_nanoscope_text_wrapped", False):
            return
        item._nanoscope_text_wrapped = True
        original = getattr(item, "mouseClickEvent", None)
        ref = weakref.ref(self)

        def click(ev, original=original, resolve=resolve):
            ctl = ref()
            key = resolve(ev) if ctl is not None else None
            if key is not None and ctl.target(key) is not None:
                if ev.button() == QtCore.Qt.MouseButton.LeftButton and GATE(ev, (id(ctl), key)):
                    ev.accept()
                    ctl._edit_or_style(key)
                    return
                if ev.button() == QtCore.Qt.MouseButton.RightButton:
                    ev.accept()
                    ctl._raise_menu(key, ev)
                    return
            if original is not None:
                original(ev)
            else:
                ev.ignore()

        item.mouseClickEvent = click

    def _wrap_axis(self, axis: pg.AxisItem, side: str) -> None:
        if getattr(axis, "_nanoscope_text_wrapped", False):
            return
        axis._nanoscope_text_wrapped = True
        axis.label.setCursor(QtCore.Qt.CursorShape.IBeamCursor)
        original = axis.mouseClickEvent         # bound: forwards to the linked ViewBox
        ref = weakref.ref(self)

        def click(ev, original=original, axis=axis, side=side):
            ctl = ref()
            if ctl is None or not axis.isVisible():
                original(ev)
                return
            on_label = (axis.label.isVisible() and axis.label.mapRectToParent(
                axis.label.boundingRect()).contains(ev.pos()))
            key = f"axis.{side}.label" if on_label else f"axis.{side}.ticks"
            known = ctl.target(f"axis.{side}.label") is not None
            if known and ev.button() == QtCore.Qt.MouseButton.LeftButton and GATE(ev, (id(ctl), key)):
                ev.accept()
                if on_label:
                    ctl.edit_inline(key)
                else:
                    ctl.open_axis_properties(side)
                return
            if known and ev.button() == QtCore.Qt.MouseButton.RightButton:
                # The menu of what was hit: the label's on the label, the tick labels'
                # (their Text style…, Axis properties…) on the numbers.
                ev.accept()
                ctl._raise_menu(key if ctl.target(key) is not None else f"axis.{side}.label", ev)
                return
            original(ev)

        axis.mouseClickEvent = click

    def _legend_key(self, ev) -> str | None:
        return self._legend_row_at(ev.scenePos())

    def _legend_row_at(self, scene_pos) -> str | None:
        legend = self._legend
        if not _alive(legend):
            return None
        rows = {id(target.item): key for key, target in self.targets().items()
                if key.startswith("legend.entry.")}
        for sample, label in list(legend.items):
            if not (_alive(label) and _alive(sample)):
                continue
            if label.sceneBoundingRect().contains(scene_pos) or (
                    sample.sceneBoundingRect().contains(scene_pos)):
                return rows.get(id(label))
        return None

    def _legend_double(self, _legend, ev) -> None:
        key = self._legend_row_at(ev.scenePos())
        if key is not None and GATE.take((id(self), key)):
            self._edit_or_style(key)

    # -- the menu -----------------------------------------------------------------------------
    def menu_entries(self, key: str) -> list[tuple[str, Callable[[], None]]]:
        """The entries a right-click on ``key`` adds above the page's own."""
        target = self.target(key)
        if target is None:
            return []
        caps, noun = target.capabilities, target.noun
        style = target.style
        entries: list[tuple[str, Callable[[], None]]] = []
        legend = key.startswith("legend.entry.")
        if "edit" in caps:
            entries.append(("Rename…" if legend else f"Edit {noun}…",
                            lambda: self.edit_inline(key)))
        if "style" in caps:
            entries.append(("Text style…", lambda: self.open_style(key)))
        if "edit" in caps:
            entries.append(("Reset name" if legend else f"Automatic {noun}",
                            lambda: self._set(key, replace(style, text=None))))
        if "hide" in caps:
            entries.append(("Hide entry" if legend else f"Hide {noun}",
                            lambda: self._set(key, replace(style, text=HIDDEN_TEXT))))
        if key.startswith("axis.") and key.endswith(".ticks"):
            # A hidden label leaves nothing to right-click: its numbers bring it back.
            label_key = key[:-len("ticks")] + "label"
            label = self.target(label_key)
            if label is not None and "edit" in label.capabilities and label.style.text == HIDDEN_TEXT:
                entries.append(("Show label", lambda: self._set(
                    label_key, replace(label.style, text=None))))
        if key.startswith("axis."):
            side = key.split(".")[1]
            entries.append(("Axis properties…", lambda: self.open_axis_properties(side)))
        extra = getattr(self.host, "menu_entries", None)
        if callable(extra):
            entries.extend(extra(key) or [])
        return entries

    def background_entries(self) -> list[tuple[str, Callable[[], None]]]:
        """The entries a right-click on the plot itself (no text under it) adds above the
        page's own: "Add title…" while the title draws nothing -- an empty automatic title
        (I-V, Voltage dependence) or a hidden one -- because such a title has no row left
        to click; "Show title" too when a hidden one has automatic text to come back to.
        "Add title (this view)…" where the views share one title ("(both figures)" in
        Subpopulations: ``AppearanceTextHost.title_scope``). Then the host's own
        (``background_entries``, optional), such as Distributions' Show detection-limit label."""
        entries: list[tuple[str, Callable[[], None]]] = []
        target = self.target("title")
        if (target is not None and "edit" in target.capabilities
                and not shown_text(target.style, target.auto).strip()):
            # A window whose views share one title (Noise, Event waveforms, Subpopulations)
            # says what the title covers -- the figure(s) on screen, until the view changes --
            # as the Appearance box's "Title (this view)" does.
            scope = getattr(self.host, "title_scope", "")
            entries.append((f"Add title ({scope})…" if scope else "Add title…", self.add_title))
            if target.style.text == HIDDEN_TEXT and target.auto.strip():
                style = target.style
                entries.append(("Show title",
                                lambda: self._set("title", replace(style, text=None))))
        # A host's own: a hidden annotation has nothing left to right-click either.
        extra = getattr(self.host, "background_entries", None)
        if callable(extra):
            entries.extend(extra() or [])
        return entries

    def _raise_menu(self, key: str, ev) -> None:
        plot = self.plot
        if plot is not None and getattr(plot, "_nanoscope_menu_factory", None) is not None:
            # The page's app menu, with this element's entries on top (plot_interaction).
            plot._nanoscope_menu_context = key
            plot.getViewBox().raiseContextMenu(ev)
            return
        self._raise_menu_at(key, ev.screenPos())

    def _raise_menu_at(self, key: str, screen_pos) -> QtWidgets.QMenu | None:
        plot = self.plot
        if plot is not None and getattr(plot, "_nanoscope_menu_factory", None) is not None:
            menu = plot._nanoscope_menu_factory(key)
        else:
            view = self._view()
            menu = QtWidgets.QMenu(view)
            menu.setObjectName("plotTextMenu")
            for title, callback in self.menu_entries(key):
                menu.addAction(title, callback)
        self.last_menu = menu
        menu.popup(QtCore.QPoint(int(screen_pos.x()), int(screen_pos.y())))
        return menu

    # -- the editors ----------------------------------------------------------------------------
    def close_editors(self) -> None:
        """Cancel an open editor and close a popover (a page clearing its plot calls it)."""
        if _alive(self.editor):
            self.editor.cancel()
        self.editor = None
        if _alive(self.popover):
            self.popover.close()
        self.popover = None

    def _edit_or_style(self, key: str) -> None:
        target = self.target(key)
        if target is None:
            return
        if "edit" in target.capabilities:
            self.edit_inline(key)
        elif "style" in target.capabilities:
            self.open_style(key)

    def _anchor_rect(self, target: TextTarget) -> QtCore.QRect | None:
        view = self._view()
        item = target.item
        if view is None or not _alive(item):
            return None
        shape = item.label if isinstance(item, pg.AxisItem) else item
        rect = view.mapFromScene(shape.sceneBoundingRect()).boundingRect()
        plot = self.plot
        if plot is not None and item is plot.titleLabel and not item.isVisible():
            # A title that draws nothing has a 0 px row: open over the top of the plot,
            # where the title will appear, rather than half outside the view.
            top = view.mapFromScene(plot.getViewBox().sceneBoundingRect()).boundingRect()
            rect = QtCore.QRect(top.left(), top.top(), top.width(), 26)
        return rect

    def add_title(self) -> InlineTextEditor | None:
        """"Add title…": the inline editor where the title goes. Return keeps what was
        typed, one Undo step, and the title row appears with it; Esc, or Return on the
        box left empty, leaves the figure as it was."""
        return self.edit_inline("title")

    def edit_inline(self, key: str) -> InlineTextEditor | None:
        target = self.target(key)
        view = self._view()
        rect = self._anchor_rect(target) if target is not None else None
        if target is None or view is None or rect is None:
            return None
        if _alive(self.editor):
            self.editor.cancel()
        text = shown_text(target.style, target.auto)
        editor = InlineTextEditor(key, text, view.viewport())
        room = view.viewport().rect()
        width = int(min(max(rect.width(), 180), room.width()))
        # Centred on the element, but never past the view's edges: a ViewBox flush with
        # the viewport maps a pixel or two beyond it once its edges are rounded.
        x = max(0, min(int(rect.center().x() - width / 2), room.width() - width))
        y = max(0, min(int(rect.center().y() - 13), room.height() - 26))
        editor.setGeometry(x, y, width, 26)
        ref = weakref.ref(self)
        editor.committed.connect(lambda k, t, r=ref: r() and r()._commit_text(k, t))
        editor.show()
        editor.setFocus()
        self.editor = editor
        self.opened.emit("inline", key)
        return editor

    def _commit_text(self, key: str, text: str) -> None:
        target = self.target(key)
        editor = self.editor
        self.editor = None
        if target is None:
            return
        if editor is not None and _alive(editor) and text == editor.initial:
            # Opened on what the element shows and left as it was: nothing changes. An
            # automatic text stays automatic, and an empty box on a title or label that
            # draws nothing ("Add title…") does not turn a hidden one automatic.
            return
        self._set(key, replace(target.style, text=TextStyle.from_mapping({"text": text}).text))

    def _set(self, key: str, style: TextStyle) -> None:
        try:
            self.host.apply_text(key, style)
        except RuntimeError:     # the page was torn down under the gesture
            pass

    def _place(self, popover: QtWidgets.QFrame, target: TextTarget | None) -> None:
        view = self._view()
        rect = self._anchor_rect(target) if target is not None else None
        if view is not None and rect is not None:
            popover.move(view.mapToGlobal(rect.bottomLeft()))
        popover.adjustSize()
        popover.show()

    def open_style(self, key: str) -> TextStylePopover | None:
        target = self.target(key)
        view = self._view()
        if target is None or view is None:
            return None
        self.close_editors()
        popover = TextStylePopover(key, target.style, target.auto, view.window(),
                                   align=("align" in target.capabilities),
                                   editable=("edit" in target.capabilities))
        ref = weakref.ref(self)
        popover.committed.connect(lambda k, s, r=ref: r() and r()._set(k, s))
        self._place(popover, target)
        self.popover = popover
        self.opened.emit("style", key)
        return popover

    def open_axis_properties(self, side: str) -> AxisPropertiesPopover | None:
        view = self._view()
        if view is None:
            return None
        try:
            state = self.host.axis_state(side)
        except (RuntimeError, AttributeError, NotImplementedError):
            return None
        self.close_editors()
        popover = AxisPropertiesPopover(side, state, view.window())
        ref = weakref.ref(self)
        popover.committed.connect(lambda s, v, r=ref: r() and r()._apply_axis(s, v))
        plot = self.plot
        target = TextTarget(item=plot.getAxis(side)) if plot is not None else None
        self._place(popover, target)
        self.popover = popover
        self.opened.emit("axis", f"axis.{side}.ticks")
        return popover

    def _apply_axis(self, side: str, values: AxisState) -> None:
        try:
            self.host.apply_axis(side, values)
        except RuntimeError:
            pass


def attach_figure_text(plot, host: FigureTextHost) -> FigureTextController:
    """Make ``plot``'s text editable in the figure, through ``host``. Once per PlotItem: a
    second call returns the controller already there (with its host replaced)."""
    if isinstance(plot, pg.PlotWidget):
        plot = plot.getPlotItem()
    controller = figure_text_for(plot)
    if controller is not None:
        controller.host = host
        controller.refresh()
        return controller
    controller = FigureTextController(plot, host).install()
    plot._nanoscope_figure_text = controller
    return controller


def figure_text_for(plot) -> FigureTextController | None:
    if isinstance(plot, pg.PlotWidget):
        plot = plot.getPlotItem()
    controller = getattr(plot, "_nanoscope_figure_text", None)
    return controller if _alive(controller) else None


# -- the generic host of the PlotAppearance windows ------------------------------------------
def _display_to_view(axis: pg.AxisItem, value: float) -> float:
    """A typed DISPLAY value as the ViewBox coordinate (``plot_interaction.axis_value_text``
    inverted: log axes hold log10, the tick scale and a log display factor multiply)."""
    factor = float(getattr(axis, "scale", 1.0) or 1.0)
    if axis.logMode:
        factor *= float(getattr(axis, "display_factor", 1.0) or 1.0)
        return math.log10(value / factor)
    return value / factor


def _view_to_display(axis: pg.AxisItem, value: float) -> float:
    factor = float(getattr(axis, "scale", 1.0) or 1.0)
    if axis.logMode:
        factor *= float(getattr(axis, "display_factor", 1.0) or 1.0)
        try:
            return (10.0 ** value) * factor
        except OverflowError:
            return math.inf
    return value * factor


class AppearanceTextHost:
    """:class:`FigureTextHost` for the windows whose appearance is a ``PlotAppearance``.

    Every write goes through ``widget.apply_appearance``: one appearance Undo step, and the
    window's existing ``changed`` handler redraws exactly as for an edit in the Appearance
    section. Title and axis-label TEXT stays in the ``title``/``x_label``/``y_label`` fields
    the boxes edit, so there is one store for each label.

    ``axis_hooks`` (all optional): ``{"log": callable(side, on), "log_state":
    callable(side) -> bool, "log_reason": str, "label_auto": callable(side) -> str}``.
    ``extra_targets``: a callable returning ``{key: item or TextTarget}`` for the window's
    own text (``colorbar.label``, ``marginal.left.label``…), whose styles the window draws
    with :func:`apply_text_style` from ``state.text_style(key)``.
    ``shared_labels``: in a window whose views share ONE x and y label (Noise, Event
    waveforms, Subpopulations), the reason axis-label text is not typed on a figure; an
    edit on one view would relabel views that show other quantities. Style still applies.
    ``pin_view`` (a name, or a callable giving the one drawn now): in a window whose views
    show different quantities on one appearance (Noise, Event waveforms, Distribution
    summaries, Subpopulations), the view whose Axis properties pins are its own, kept as
    ``<view>.<side>`` (:func:`view_pin_key`), so a range typed on one view never frames
    another.
    ``primary_view``: the view a bare ``<side>`` pin, saved before pins were kept per view,
    belongs to (``PlotAppearance.for_view``).
    """

    def __init__(self, plot: pg.PlotItem, widget, *, axis_hooks: Mapping | None = None,
                 extra_targets: Callable[[], Mapping] | Mapping | None = None,
                 shared_labels: str = "", pin_view: str | Callable[[], str] = "",
                 primary_view: str = "") -> None:
        self._plot_ref = weakref.ref(plot)
        self._widget_ref = weakref.ref(widget)
        self.axis_hooks = dict(axis_hooks or {})
        self.extra_targets = extra_targets
        self.shared_labels = str(shared_labels or "")
        self._pin_view = pin_view
        self.primary_view = str(primary_view or "")
        self._annotations: dict[str, weakref.ref] = {}

    @property
    def title_scope(self) -> str:
        """What a title typed on this figure covers ("this view", "both figures"), as the
        window's Title box says; "" where the window has one figure."""
        widget = self._widget_ref()
        return str(getattr(widget, "title_scope", "") or "")

    @property
    def pin_view(self) -> str:
        """The view this figure's pins are kept under now; "" in a single-view window."""
        view = self._pin_view() if callable(self._pin_view) else self._pin_view
        return str(view or "")

    def _pin(self, state, side: str) -> AxisPin:
        """This view's pin on ``side``: never another view's (``PlotAppearance.for_view``)."""
        return state.for_view(self.pin_view, primary=self.primary_view).axis_pin(side)

    @property
    def plot(self):
        plot = self._plot_ref()
        return plot if _alive(plot) else None

    @property
    def widget(self):
        widget = self._widget_ref()
        return widget if _alive(widget) else None

    def state(self):
        from .plot_appearance import PlotAppearance

        widget = self.widget
        return widget.state() if widget is not None else PlotAppearance()

    def register_fit_label(self, item, key: str = "annotation.fit") -> None:
        """Make ``item`` (a ``fit_label.FitLabelItem``) editable, and keep where it is
        dragged (:meth:`register_annotation`)."""
        self._annotations[key] = weakref.ref(item)
        self.register_annotation(item, key)

    def register_annotation(self, item, key: str) -> None:
        """Keep where ``item`` is dragged: each drag (its ``moved(offset)`` signal, as
        ``fit_label.FitLabelItem`` and its subclasses send on release) is ONE appearance
        step that stores the offset under ``key`` (``PlotAppearance.with_annotation_offset``:
        the fit label's in ``fit_label``, any other ``annotation.<id>`` in ``annotations``),
        so Undo, templates and projects keep it. The window draws the item at
        ``state.annotation_offset(key)`` (None: its own place) on every apply. Its text,
        if editable, is the window's target as before (``extra_targets``). Once per item."""
        ref = weakref.ref(self)
        item.moved.connect(lambda offset, r=ref, k=key: r() and r()._moved(k, offset))
        controller = figure_text_for(self.plot) if self.plot is not None else None
        if controller is not None:
            controller.refresh()

    def _moved(self, key: str, offset) -> None:
        widget = self.widget
        if widget is not None:
            widget.apply_appearance(widget.state().with_annotation_offset(key, offset))

    def _defaults(self) -> dict:
        plot = self.plot
        return getattr(plot, "_nanoscope_appearance_defaults", None) or {}

    def automatic_text(self, key: str) -> str:
        plot = self.plot
        if plot is None:
            return ""
        defaults = self._defaults()
        if key == "title":
            return plain_text(defaults.get("title_text", plot.titleLabel.text) or "")
        if key.startswith("axis.") and key.endswith(".label"):
            side = key.split(".")[1]
            hook = self.axis_hooks.get("label_auto")
            if callable(hook):
                return str(hook(side) or "")
            axis = plot.getAxis(side)
            text = defaults.get("axes", {}).get(side, {}).get("text", axis.labelText)
            units = f" ({axis.labelUnits})" if axis.labelUnits else ""
            return plain_text(text or "") + units
        return ""

    def text_targets(self) -> dict[str, TextTarget]:
        plot = self.plot
        if plot is None:
            return {}
        state = self.state()
        out: dict[str, TextTarget] = {
            "title": TextTarget(plot.titleLabel, self.automatic_text("title"),
                                state.text_style("title"), "title",
                                frozenset({"edit", "style", "hide", "align"}))}
        for side in ("bottom", "left"):
            axis = plot.getAxis(side)
            if not axis.isVisible():
                continue
            out[f"axis.{side}.label"] = TextTarget(
                axis, self.automatic_text(f"axis.{side}.label"),
                state.text_style(f"axis.{side}.label"), "label",
                frozenset({"style"}) if self.shared_labels else
                frozenset({"edit", "style", "hide"}))
            out[f"axis.{side}.ticks"] = TextTarget(
                axis, "", state.text_style(f"axis.{side}.ticks"), "tick labels",
                frozenset({"style"}))
        legend = getattr(plot, "legend", None)
        if legend is not None and _alive(legend):
            for _sample, label in list(legend.items):
                if not _alive(label):
                    continue
                name = getattr(label, "_nanoscope_auto_text", None)
                if name is None:
                    name = label._nanoscope_auto_text = label.text
                key = f"legend.entry.{plain_text(name)}"[:213]
                out[key] = TextTarget(label, plain_text(name), state.text_style(key),
                                      "legend entry", frozenset({"edit", "style"}))
        for key, ref in list(self._annotations.items()):
            item = ref()
            if not _alive(item):
                continue
            from .fit_label import automatic_template

            # The editor opens on the TEMPLATE ("{equation} · {r2} · {n}"), so retyped
            # words keep live numbers instead of freezing the ones on screen.
            out[key] = TextTarget(item, automatic_template(state.fit_label_state()),
                                  state.text_style(key), "fit label",
                                  frozenset({"edit", "style", "hide"}))
        extra = self.extra_targets() if callable(self.extra_targets) else (self.extra_targets or {})
        for key, value in dict(extra).items():
            if isinstance(value, TextTarget):
                out[key] = replace(value, style=state.text_style(key))
            elif _alive(value):
                auto = plain_text(value.labelText if isinstance(value, pg.AxisItem)
                                  else getattr(value, "text", "") if isinstance(value, pg.LabelItem)
                                  else value.toPlainText() if isinstance(value, pg.TextItem) else "")
                out[key] = TextTarget(value, auto, state.text_style(key))
        return out

    def apply_text(self, key: str, style: TextStyle) -> None:
        widget = self.widget
        if widget is not None:
            widget.apply_appearance(widget.state().with_text_style(key, style))

    def axis_state(self, side: str) -> AxisState:
        plot = self.plot
        axis = plot.getAxis(side)
        vb = plot.getViewBox()
        index = 0 if side in ("bottom", "top") else 1
        lo, hi = (_view_to_display(axis, v) for v in vb.viewRange()[index])
        state = self.state()
        pin = self._pin(state, side)
        log_hook = self.axis_hooks.get("log")
        log_state = self.axis_hooks.get("log_state")
        label = state.text_style(f"axis.{side}.label").text
        return AxisState(
            label=label, label_auto=self.automatic_text(f"axis.{side}.label"),
            units=str(axis.labelUnits or ""), range=pin.range(), shown_range=(lo, hi),
            log=bool(log_state(side)) if callable(log_state) else bool(axis.logMode),
            log_available=callable(log_hook),
            log_reason=str(self.axis_hooks.get("log_reason")
                           or "This window has no log scale for this axis."),
            tick_pt=state.text_style(f"axis.{side}.ticks").size_pt, tick_major=pin.tick_major,
            label_locked=self.shared_labels)

    def apply_axis(self, side: str, values: AxisState) -> None:
        widget = self.widget
        if widget is None:
            return
        state = widget.state()
        before = self.axis_state(side)
        lo, hi = values.range if values.range is not None else (None, None)
        view = self.pin_view
        key = view_pin_key(view, side)
        pin = replace(self._pin(state, side), min=lo, max=hi, tick_major=values.tick_major)
        if not self.shared_labels:
            state = state.with_text_style(f"axis.{side}.label", replace(
                state.text_style(f"axis.{side}.label"), text=values.label))
        state = state.with_text_style(f"axis.{side}.ticks", replace(
            state.text_style(f"axis.{side}.ticks"), size_pt=values.tick_pt))
        state = state.with_axis_pin(key, pin)
        if key != side and view == self.primary_view:
            # A bare pin read above (saved before pins were kept per view) is now this
            # view's own key, so clearing it here cannot bring the old one back.
            state = state.with_axis_pin(side, AxisPin())
        widget.apply_appearance(state)
        hook = self.axis_hooks.get("log")
        if callable(hook) and values.log != before.log:
            hook(side, values.log)

    def menu_entries(self, key: str) -> list:
        return []


def view_pin_key(view: str, side: str) -> str:
    """The ``PlotAppearance.axes`` key of one view's pin: ``<view>.<side>``, or ``side`` in
    a window with one view."""
    return f"{view}.{side}" if view else side


def appearance_text_host(plot, widget, *, axis_hooks=None, extra_targets=None,
                         shared_labels: str = "", pin_view: str | Callable[[], str] = "",
                         primary_view: str = "") -> AppearanceTextHost:
    """The generic :class:`FigureTextHost` of a ``PlotAppearanceWidget`` window."""
    if isinstance(plot, pg.PlotWidget):
        plot = plot.getPlotItem()
    return AppearanceTextHost(plot, widget, axis_hooks=axis_hooks, extra_targets=extra_targets,
                              shared_labels=shared_labels, pin_view=pin_view,
                              primary_view=primary_view)

#: The reason a multi-view window's axis label is renamed in Appearance, not on a figure.
SHARED_LABELS_REASON = ("This window's views share one X and one Y label (Appearance ▸ "
                        "X label / Y label), so it is renamed there; renaming it "
                        "here would relabel views that show other quantities.")
