"""The on-plot fit label: where a fit's equation and numbers go instead of the title.

A fit used to BECOME the plot title (Distributions scatter and histogram, I-V, Voltage
dependence), so a typed title lost the result and an automatic one lost the title. The fit
label is its own element: one line per fitted source, in the source's colour, top-left by
default, draggable, editable in place, and on by default whenever a fit is applied, so no
number visible on a figure before this change disappears from it.

THE WORDS ARE THE USER'S, THE NUMBERS NEVER ARE. The label's text is a TEMPLATE with live
tokens -- ``{equation} {r2} {n} {ci} {bic} {ks} {source}`` -- so a re-fit puts new numbers
into the user's own words. And no template can drop a refusal: when R² is asked for and
the fit cannot honestly report it (zero degrees of freedom, an undefined total variance),
the refusal tag is printed whatever the template says (:func:`format_fit_label`).

Windows build :class:`FitReadout` values from their fits and draw them with
:meth:`FitLabelItem.apply_state`; the label's state is ``PlotAppearance.fit_label`` (or
Distributions' own copy), so it travels with Undo, templates and projects.
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from .figure_text import GATE, HIDDEN_TEXT, rich_text

__all__ = ["FIT_FIELDS", "FIT_TOKENS", "PLACEMENTS", "FitLabelItem", "FitLabelState", "automatic_template",
           "FitReadout", "format_fit_label", "valid_offset"]

#: The numbers a label can show, in the order the automatic text joins them.
FIT_FIELDS = ("equation", "r2", "n", "ci", "bic", "ks")
#: What a template may name: the numbers, and the source's legend name.
FIT_TOKENS = FIT_FIELDS + ("source",)
PLACEMENTS = ("auto", "top-left", "top-right", "bottom-left", "bottom-right", "custom")
_TOKEN = re.compile(r"\{(" + "|".join(FIT_TOKENS) + r")\}")
#: Pixels between the label and the corner of the plot it sits in.
_MARGIN = (10.0, 8.0)


def valid_offset(value: Any) -> tuple[float, float] | None:
    """A dragged annotation's saved place, ``(x, y)`` px from the plot's top-left corner,
    or None for anything else (a bad value in a saved file is dropped, never an error).
    One rule for the fit label and every other annotation a drag places."""
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and abs(v) <= 20000 for v in value)):
        return (float(value[0]), float(value[1]))
    return None


@dataclass(frozen=True)
class FitLabelState:
    visible: bool = True
    fields: tuple[str, ...] = ("equation", "r2", "n")
    template: str = ""                 # "" automatic; " " hidden; else the user's words
    placement: str = "top-left"        # "auto" | a corner | "custom"
    offset: tuple[float, float] | None = None   # px from the top-left corner, when custom
    size_pt: float | None = None
    color: str | None = None
    in_legend: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = {key: value for key, value in asdict(self).items() if value is not None}
        out["fields"] = list(self.fields)
        if self.offset is not None:
            out["offset"] = list(self.offset)
        return out

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None, *,
                     legacy: Mapping[str, Any] | None = None) -> "FitLabelState":
        """Tolerant: a bad field is dropped. With no saved label, ``legacy`` (a pre-label
        configuration's ``show_r2`` / ``show_gof`` / ``fit_values_on_plot``) picks the
        fields; the label is visible either way, because before it the fit was in the
        title, so a hidden label would take the numbers off a figure that showed them."""
        if not isinstance(values, Mapping):
            legacy = legacy if isinstance(legacy, Mapping) else {}
            fields = ["equation"]
            if legacy.get("show_r2", True) is not False:
                fields.append("r2")
            fields.append("n")
            if legacy.get("show_gof") is True:
                fields.append("ks")
            return cls(fields=tuple(fields))
        valid: dict[str, Any] = {}
        for key in ("visible", "in_legend"):
            if isinstance(values.get(key), bool):
                valid[key] = values[key]
        fields = values.get("fields")
        if isinstance(fields, (list, tuple)):
            valid["fields"] = tuple(f for f in FIT_FIELDS if f in fields)
        template = values.get("template")
        if isinstance(template, str):
            valid["template"] = (template.strip()[:500] if template.strip()
                                 else (HIDDEN_TEXT if template else ""))
        if values.get("placement") in PLACEMENTS:
            valid["placement"] = values["placement"]
        offset = valid_offset(values.get("offset"))
        if offset is not None:
            valid["offset"] = offset
        size = values.get("size_pt")
        if (isinstance(size, (int, float)) and not isinstance(size, bool) and math.isfinite(size)
                and 4 <= size <= 72):       # Text style…'s range (TextStyle)
            valid["size_pt"] = float(size)
        color = values.get("color")
        if isinstance(color, str) and QtGui.QColor(color).isValid():
            valid["color"] = QtGui.QColor(color).name()
        state = cls(**valid)
        if state.placement == "custom" and state.offset is None:
            state = replace(state, placement="top-left")
        return state


@dataclass(frozen=True)
class FitReadout:
    """One fitted source's numbers, as the plain text each token prints.

    ``parts`` maps a :data:`FIT_FIELDS` name to its text ("R² = 0.981 (118 dof)"), only for
    the numbers this fit has. ``refusals`` are the tags no template may drop ("R² undefined:
    zero degrees of freedom"). ``source`` is the legend name, ``colour`` the source's colour.
    """

    parts: Mapping[str, str]
    refusals: tuple[str, ...] = ()
    source: str | None = None
    colour: str | None = None


def _line(readout: FitReadout, state: FitLabelState, prefix: bool) -> str:
    parts = {key: str(value) for key, value in dict(readout.parts).items() if value}
    asks_r2 = "r2" in state.fields or "{r2}" in state.template
    refusal = " · ".join(readout.refusals) if (asks_r2 and readout.refusals) else ""
    if state.template == HIDDEN_TEXT:
        return html.escape(refusal, quote=False)
    if state.template:
        values = dict(parts, source=readout.source or "")
        if "r2" not in values and refusal:
            values["r2"] = refusal        # {r2} of a fit with no R² prints why
        out, end = [], 0
        for match in _TOKEN.finditer(state.template):
            out.append(rich_text(state.template[end:match.start()]))
            value = values.get(match.group(1), "—")
            # The legend name as the legend draws it (its _{…}/^{…} too); numbers and
            # refusals are the fit's own text, escaped and nothing more.
            out.append(rich_text(value) if match.group(1) == "source"
                       else html.escape(value, quote=False))
            end = match.end()
        out.append(rich_text(state.template[end:]))
        text = "".join(out)
    else:
        chosen = [parts[key] for key in state.fields if key in parts]
        text = html.escape(" · ".join(chosen), quote=False)
        if prefix and readout.source:
            text = f"{rich_text(readout.source)}: {text}" if text else ""
    plain = html.unescape(re.sub(r"<[^>]+>", "", text))
    missing = [tag for tag in readout.refusals if asks_r2 and tag not in plain]
    if missing:
        tags = html.escape(" · ".join(missing), quote=False)
        text = f"{text} · {tags}" if text else tags
    return text


def automatic_template(state: FitLabelState) -> str:
    """The automatic text written as a template: what the inline editor opens on, so
    editing the words keeps the numbers live instead of freezing them."""
    return " · ".join("{%s}" % name for name in state.fields)


def format_fit_label(readouts: Sequence[FitReadout], state: FitLabelState) -> list[tuple[str, str | None]]:
    """``(html line, colour)`` per readout that has anything to say.

    A refusal tag is printed whenever R² is asked for (a field or the ``{r2}`` token),
    whatever the template says -- a label is never more confident than its fit.
    """
    if not state.visible:
        return []
    prefix = len(readouts) > 1
    lines = []
    for readout in readouts:
        text = _line(readout, state, prefix)
        if text.strip():
            lines.append((text, state.color or readout.colour))
    return lines


class FitLabelItem(pg.LabelItem):
    """The fit label on a plot: parented to the ViewBox, so zoom and pan leave it where it
    is. A drag moves it (never the view) and says where once, on release (``moved``); a
    double click asks to edit it (``editRequested``); a right click asks for the menu
    (``menuRequested``, the screen position)."""

    moved = QtCore.Signal(tuple)
    editRequested = QtCore.Signal()
    menuRequested = QtCore.Signal(object)

    def __init__(self, view_box: pg.ViewBox | None = None) -> None:
        super().__init__("", justify="left")
        self.setObjectName("plotFitLabel")
        self.setZValue(30)
        self._readouts: tuple[FitReadout, ...] = ()
        self._state = FitLabelState()
        self._offset = (_MARGIN[0], _MARGIN[1])
        self._drag_from: tuple[float, float] | None = None
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Fit label: drag to move, double-click to edit its text "
                        "(the numbers stay live)")
        if view_box is not None:
            self.setParentItem(view_box)
            self._place()

    # -- drawing --------------------------------------------------------------------------
    def apply_state(self, state: FitLabelState, readouts: Sequence[FitReadout] = (),
                    style=None) -> None:
        """Draw ``readouts`` as ``state`` says; hidden when there is nothing to show, when
        the label is off, or when its numbers are in the legend instead. ``style`` (a
        ``TextStyle``, from Text style…) adds font, bold and italic."""
        self._state = state
        self._readouts = tuple(readouts)
        lines = [] if state.in_legend else format_fit_label(self._readouts, state)
        body = "<br>".join(f"<span style='color: {colour}'>{text}</span>" if colour else text
                           for text, colour in lines)
        for key in ("family", "bold", "italic", "size"):
            self.opts.pop(key, None)      # setText merges: a style taken off must go
        options: dict[str, Any] = {"justify": "left"}
        if state.size_pt:
            options["size"] = f"{state.size_pt:g}pt"
        if style is not None:
            for key, value in (("family", style.family), ("bold", style.bold),
                               ("italic", style.italic)):
                if value is not None:
                    options[key] = value
        self.setText(body, **options)
        self.setVisible(bool(lines))
        self._place()

    def state(self) -> FitLabelState:
        return self._state

    def lines(self) -> list[tuple[str, str | None]]:
        return [] if self._state.in_legend else format_fit_label(self._readouts, self._state)

    def plain_text(self) -> str:
        return self.item.toPlainText()

    def _corner(self) -> tuple[tuple[float, float], tuple[float, float]]:
        placement = self._state.placement
        if placement == "custom" and self._state.offset is not None:
            return (0.0, 0.0), self._state.offset
        x = 1.0 if placement.endswith("right") else 0.0
        y = 1.0 if placement.startswith("bottom") else 0.0
        return (x, y), (_MARGIN[0] * (1 - 2 * x), _MARGIN[1] * (1 - 2 * y))

    def _place(self) -> None:
        if self.parentItem() is None:
            return
        corner, offset = self._corner()
        self._offset = (float(offset[0]), float(offset[1])) if corner == (0.0, 0.0) else self._offset
        self.anchor(itemPos=corner, parentPos=corner, offset=offset)

    def offset(self) -> tuple[float, float]:
        """Where the label's top-left sits, in px from the plot's top-left corner."""
        parent = self.parentItem()
        if parent is None:
            return self._offset
        here = self.mapToParent(QtCore.QPointF(0, 0))
        return (float(here.x()), float(here.y()))

    # -- pyqtgraph's own dispatch (no event filter) --------------------------------------
    def hoverEvent(self, ev):  # noqa: N802 - pyqtgraph API
        if not ev.isExit():
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)

    def mouseClickEvent(self, ev):  # noqa: N802 - pyqtgraph API
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and GATE(ev, ("fit", id(self))):
            ev.accept()
            self.editRequested.emit()
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.menuRequested.emit(ev.screenPos())
        else:
            ev.ignore()

    def mouseDragEvent(self, ev):  # noqa: N802 - pyqtgraph API
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        ev.accept()           # the drag is the label's: the ViewBox must not pan
        if ev.isStart():
            self._drag_from = self.offset()
        if self._drag_from is None:
            return
        delta = ev.scenePos() - ev.buttonDownScenePos()
        offset = (self._drag_from[0] + delta.x(), self._drag_from[1] + delta.y())
        self.anchor(itemPos=(0, 0), parentPos=(0, 0), offset=offset)
        if ev.isFinish():
            self._drag_from = None
            self._offset = offset
            self._state = replace(self._state, placement="custom", offset=offset)
            self.moved.emit(offset)
