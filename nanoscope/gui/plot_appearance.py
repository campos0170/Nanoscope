"""Shared, opt-in styling of existing plots, with no analysis callbacks.

Empty state preserves the original appearance. Clearing overrides restores the
original graphical properties. Text overrides never change units, arrays or fits.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import functools
import math
import json
import re
from typing import Any, Callable, Mapping, Sequence

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .axes import set_major_tick
from .figure_text import (AXIS_SIDES, AxisPin, TextStyle, _alive, _display_to_view, css_args,
                          label_args, plain_text, rich_text, valid_element_key, view_pin_key)
from .fit_label import FIT_FIELDS, FitLabelState, valid_offset
from .font_picker import FontPicker, figure_family, figure_qfont, html_font_family
from .key_sequences import redo_sequences, undo_sequences

#: How many appearance values one widget's history keeps, and how long a run of edits
#: may stay open before it becomes one undo step. EQUAL to ``plots_panel``'s
#: ``_HISTORY_DEPTH`` and ``_HISTORY_COALESCE_MS``, on purpose: design §4.2 gives every
#: plot window the Plots window's depth and pause, and two windows of one app must not
#: disagree about how far back Undo reaches or how long a pause ends a run. Only those
#: two: what joins a run is this widget's own rule (any edit, a combo pick included),
#: where the Plots window commits every change but typing at once and ends typing when
#: the box loses focus (``PlotsWindow._history_touch``). Restated rather than imported,
#: because ``plots_panel`` hosts this widget and so imports THIS module.
#: tests/gui/test_plot_appearance_history.py holds the two pairs equal. No byte ceiling:
#: an entry is fifteen scalars, where the Plots window's measured 1.5 kB entries already
#: cost only ~320 kB at this depth.
APPEARANCE_HISTORY_DEPTH = 200
APPEARANCE_COALESCE_MS = 700

#: A Title / X label / Y label that draws nothing: what a box holding only spaces stores.
#: Empty (or never touched) means the window's automatic label.
HIDDEN_LABEL = " "

#: The sizes a figure draws, per field: ``from_mapping`` drops anything outside, and each
#: size spin box runs 0 ("Default") then this range, so the two cannot disagree.
_SIZE_RANGES: dict[str, tuple[float, float]] = {
    "title_size": (6, 48), "axis_size": (6, 36), "tick_size": (6, 30),
    "line_width": (.25, 10), "marker_size": (1, 30), "legend_size": (6, 30),
}


@dataclass(frozen=True)
class PlotAppearance:
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    legend_visible: bool | None = None
    legend_size: float | None = None
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
    #: Per-element styles, element key -> TextStyle, as sorted pairs so the value stays
    #: hashable. The TEXT of the title and the two axis labels stays in the three fields
    #: above (one store per label); the map holds their style only.
    text: tuple[tuple[str, TextStyle], ...] | None = None
    #: The fit label (``fit_label.FitLabelState``); None = its defaults.
    fit_label: FitLabelState | None = None
    #: Per-axis pins, side -> AxisPin (range, major tick), as sorted pairs. In a window
    #: whose views show different quantities the key is ``<view>.<side>`` (:meth:`for_view`).
    axes: tuple[tuple[str, AxisPin], ...] | None = None
    #: Where a drag put each annotation other than the fit label (V-dep's provenance
    #: caption…): element key -> (x, y) px from the plot's top-left corner, as sorted
    #: pairs. An annotation not here sits at its window's own place. The fit label's
    #: place stays in ``fit_label`` (placement "custom" and its offset).
    annotations: tuple[tuple[str, tuple[float, float]], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe overrides only; an empty mapping means the original style."""
        out = {f.name: getattr(self, f.name) for f in fields(self)
               if getattr(self, f.name) is not None
               and f.name not in ("text", "fit_label", "axes", "annotations")}
        if self.text:
            out["text"] = {key: style.to_dict() for key, style in self.text}
        if self.fit_label is not None:
            out["fit_label"] = self.fit_label.to_dict()
        if self.axes:
            out["axes"] = {side: pin.to_dict() for side, pin in self.axes}
        if self.annotations:
            # A mapping per annotation, as the fit label's: room for more than the offset.
            out["annotations"] = {key: {"offset": list(offset)} for key, offset in self.annotations}
        return out

    # -- per-element text -----------------------------------------------------------------
    def text_style(self, key: str) -> TextStyle:
        """The element's whole style, its text included, wherever each part is stored."""
        style = dict(self.text or ()).get(key, TextStyle())
        legacy = _LEGACY_TEXT.get(key)
        if legacy is not None:
            style = replace(style, text=getattr(self, legacy))
        elif key == FIT_LABEL_KEY:
            label = self.fit_label_state()
            style = replace(style, text=label.template or None, size_pt=label.size_pt,
                            color=label.color)
        return style

    def with_text_style(self, key: str, style: TextStyle) -> "PlotAppearance":
        """A copy with ``key``'s style replaced. The title's and axis labels' text go to
        their own fields, the fit label's text, size and colour to ``fit_label``."""
        if not valid_element_key(key):
            return self
        style = TextStyle.from_mapping(style.to_dict())
        changes: dict[str, Any] = {}
        legacy = _LEGACY_TEXT.get(key)
        if legacy is not None:
            changes[legacy] = style.text
            style = replace(style, text=None)
        elif key == FIT_LABEL_KEY:
            # Through from_mapping, so the state kept is the one a project reopens.
            changes["fit_label"] = FitLabelState.from_mapping(
                {**self.fit_label_state().to_dict(), "template": style.text or "",
                 "size_pt": style.size_pt, "color": style.color})
            style = replace(style, text=None, size_pt=None, color=None)
        items = dict(self.text or ())
        if style.is_default():
            items.pop(key, None)
        else:
            items[key] = style
        changes["text"] = tuple(sorted(items.items())) or None
        return replace(self, **changes)

    def axis_pin(self, side: str) -> AxisPin:
        return dict(self.axes or ()).get(side, AxisPin())

    def with_axis_pin(self, side: str, pin: AxisPin) -> "PlotAppearance":
        if not valid_pin_key(side):
            return self
        pin = AxisPin.from_mapping(pin.to_dict())
        items = dict(self.axes or ())
        if pin.is_default():
            items.pop(side, None)
        else:
            items[side] = pin
        return replace(self, axes=tuple(sorted(items.items())) or None)

    def without_axis_pin(self, view: str, side: str, *, primary: str = "") -> "PlotAppearance":
        """A copy with the range and major tick that :meth:`for_view` draws on ``side`` of
        ``view`` taken off, so the axis frames its data again: the view's own key and, in
        the ``primary`` view, the bare ``side`` an older project saved (the two keys
        ``AppearanceTextHost.apply_axis`` moves a pin between). ``log`` stays: it is how
        an axis is drawn, not what it frames."""
        keys = [view_pin_key(view, side)]
        if view == primary and keys[0] != side:
            keys.append(side)
        out = self
        for key in keys:
            out = out.with_axis_pin(key, replace(out.axis_pin(key), min=None, max=None,
                                                 tick_major=None))
        return out

    def for_view(self, view: str, *, primary: str = "") -> "PlotAppearance":
        """What ONE view of a multi-view window draws: the pins typed on it (``<view>.<side>``,
        ``figure_text.view_pin_key``) on its sides, and no other view's -- a range typed in
        Hz on the PSD is no All-points range in pA. A bare ``<side>`` pin, saved before pins
        were kept per view, is the ``primary`` view's (the one the window opens on) only.
        ``view`` "" (a window with one view) is this appearance unchanged."""
        if not view:
            return self
        own = {key: pin for key, pin in (self.axes or ()) if "." not in key and view == primary}
        own.update((key[len(view) + 1:], pin) for key, pin in (self.axes or ())
                   if key.startswith(view + "."))
        return replace(self, axes=tuple(sorted(own.items())) or None)

    def fit_label_state(self, legacy: Mapping[str, Any] | None = None) -> FitLabelState:
        """The fit label as drawn: the saved one, else defaults (``legacy`` picks fields)."""
        if self.fit_label is not None:
            return self.fit_label
        return FitLabelState.from_mapping(None, legacy=legacy)

    # -- dragged annotations --------------------------------------------------------------
    def annotation_offset(self, key: str) -> tuple[float, float] | None:
        """Where annotation ``key`` was dragged to, px from the plot's top-left corner, or
        None: at its window's own place. The fit label's is its custom ``fit_label`` offset."""
        if key == FIT_LABEL_KEY:
            label = self.fit_label_state()
            return label.offset if label.placement == "custom" else None
        return dict(self.annotations or ()).get(key)

    def with_annotation_offset(self, key: str, offset) -> "PlotAppearance":
        """A copy with annotation ``key`` dragged to ``offset``, or (None) back at its
        window's own place. The fit label's goes to ``fit_label`` (placement "custom"),
        where its Position control reads it. An offset no project could reopen is no change."""
        place = valid_offset(offset) if offset is not None else None
        if offset is not None and place is None:
            return self
        if key == FIT_LABEL_KEY:
            label = self.fit_label_state()
            if place is not None:
                return replace(self, fit_label=replace(label, placement="custom", offset=place))
            if label.placement != "custom":
                return self
            return replace(self, fit_label=replace(label, placement="top-left", offset=None))
        if not (valid_element_key(key) and key.startswith("annotation.")):
            return self
        items = dict(self.annotations or ())
        if place is None:
            items.pop(key, None)
        else:
            items[key] = place
        return replace(self, annotations=tuple(sorted(items.items())) or None)

    def style_only(self) -> "PlotAppearance":
        """What a PRESET carries: the look, never the wording (owner, Q-T2). Typed title
        and axis labels, each element's text, the fit label's words, legend-entry styles
        (tied to sources), axis pins (tied to data) and where annotations were dragged
        all stay with the figure."""
        items = {key: replace(style, text=None) for key, style in (self.text or ())
                 if not key.startswith("legend.entry.")}
        items = {key: style for key, style in items.items() if not style.is_default()}
        label = None if self.fit_label is None else replace(self.fit_label, template="",
                                                             offset=None,
                                                             placement=("top-left" if self.fit_label.placement == "custom"
                                                                        else self.fit_label.placement))
        return replace(self, title=None, x_label=None, y_label=None,
                       text=tuple(sorted(items.items())) or None, fit_label=label, axes=None,
                       annotations=None)

    def restyled(self, look: "PlotAppearance") -> "PlotAppearance":
        """``look``'s style with THIS appearance's wording, pins and annotation places: a
        preset applied."""
        look = look.style_only()
        items = dict(look.text or ())
        for key, style in (self.text or ()):
            if key.startswith("legend.entry."):
                items[key] = style
            elif style.text is not None:
                items[key] = replace(items.get(key, TextStyle()), text=style.text)
        label = look.fit_label
        if label is None:
            label = self.fit_label
        elif self.fit_label is not None:
            label = replace(label, template=self.fit_label.template,
                            placement=self.fit_label.placement, offset=self.fit_label.offset)
        return replace(look, title=self.title, x_label=self.x_label, y_label=self.y_label,
                       text=tuple(sorted(items.items())) or None, fit_label=label,
                       axes=self.axes, annotations=self.annotations)

    def template_form(self) -> "PlotAppearance":
        """What a window TEMPLATE carries (owner, Q-T6): wording and element styles, but no
        legend-entry names or styles, which belong to the sources a template leaves out.
        Dragged annotation places stay, as the fit label's dragged offset does."""
        items = {key: style for key, style in (self.text or ())
                 if not key.startswith("legend.entry.")}
        return replace(self, text=tuple(sorted(items.items())) or None)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "PlotAppearance":
        """Ignore unknown/invalid saved fields without changing analysis defaults."""
        values = values if isinstance(values, Mapping) else {}
        valid: dict[str, Any] = {}
        text = values.get("text")
        if isinstance(text, Mapping):
            items = {}
            for key, style in list(text.items())[:400]:
                if not valid_element_key(key) or not isinstance(style, Mapping):
                    continue
                style = TextStyle.from_mapping(style)
                if key in _LEGACY_TEXT or key == FIT_LABEL_KEY:
                    style = replace(style, text=None)
                if not style.is_default():
                    items[key] = style
            if items:
                valid["text"] = tuple(sorted(items.items()))
        if isinstance(values.get("fit_label"), Mapping):
            valid["fit_label"] = FitLabelState.from_mapping(values["fit_label"])
        axes = values.get("axes")
        if isinstance(axes, Mapping):
            pins = {side: AxisPin.from_mapping(pin) for side, pin in axes.items()
                    if valid_pin_key(side) and isinstance(pin, Mapping)}
            pins = {side: pin for side, pin in pins.items() if not pin.is_default()}
            if pins:
                valid["axes"] = tuple(sorted(pins.items()))
        annotations = values.get("annotations")
        if isinstance(annotations, Mapping):
            places = {}
            for key, place in list(annotations.items())[:100]:
                offset = valid_offset(place.get("offset")) if isinstance(place, Mapping) else None
                if (offset is not None and valid_element_key(key)
                        and key.startswith("annotation.") and key != FIT_LABEL_KEY):
                    places[key] = offset
            if places:
                valid["annotations"] = tuple(sorted(places.items()))
        # One rule in every plot window (owner, design 13.4 Q1): an emptied box is the
        # automatic label, as one never touched, and a box holding only spaces hides the
        # label (kept as one space, :data:`HIDDEN_LABEL`).
        for key in ("title", "x_label", "y_label"):
            text = values.get(key)
            if isinstance(text, str) and text.strip():
                valid[key] = text.strip()[:1000]
            elif isinstance(text, str) and text:
                valid[key] = HIDDEN_LABEL
        family = values.get("font_family")
        if isinstance(family, str) and family.strip() and not any(c in family for c in ";{}<>"):
            valid["font_family"] = family.strip()[:200]
        for key, (low, high) in _SIZE_RANGES.items():
            value = values.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value) and low <= value <= high:
                    valid[key] = float(value)
        if isinstance(values.get("grid"), bool):
            valid["grid"] = values["grid"]
        if isinstance(values.get("legend_visible"), bool):
            valid["legend_visible"] = values["legend_visible"]
        if values.get("background") in ("dark", "light"):
            valid["background"] = values["background"]
        for key in ("data_color", "fit_color"):
            color = values.get(key)
            if isinstance(color, str) and QtGui.QColor(color).isValid():
                valid[key] = QtGui.QColor(color).name()
        return cls(**valid)


#: The shipped looks, by their entry in the preset combo. Values, so that a history step
#: which lands exactly on one can be named after it ("Undo: Publication · light preset").
_BUILTIN_PRESETS: dict[str, PlotAppearance] = {
    "Publication · light": PlotAppearance(
        font_family="Arial", title_size=14, axis_size=12, tick_size=10, grid=False,
        background="light", data_color="#0072b2", fit_color="#d55e00", line_width=2,
        marker_size=8),
    "Presentation · dark": PlotAppearance(
        font_family="Arial", title_size=14, axis_size=12, tick_size=10, grid=True,
        background="dark", data_color="#0072b2", fit_color="#d55e00", line_width=2,
        marker_size=11),
}

#: The element keys whose TEXT lives in a field of its own (the Appearance boxes edit it).
_LEGACY_TEXT = {"title": "title", "axis.bottom.label": "x_label", "axis.left.label": "y_label"}
#: The fit label's element key: its text, size and colour live in ``fit_label``.
FIT_LABEL_KEY = "annotation.fit"

#: How a step names each field, and the word for "no override" in that control's own
#: vocabulary: the size spins and colour buttons say "Default", everything else
#: "Automatic" -- the label boxes' placeholder, and the three-way combos since ux L3 (one
#: word for "not overridden"). An emptied label box is that too; a box of spaces
#: (:data:`HIDDEN_LABEL`) draws no label, so a step names it "hidden". Every field needs
#: an entry: :func:`_describe_step` reads one for each field that moved.
_STEP_WORDS: dict[str, tuple[str, str]] = {
    "title": ("title", "automatic"), "x_label": ("x label", "automatic"),
    "y_label": ("y label", "automatic"), "legend_visible": ("legend", "automatic"),
    "legend_size": ("legend text", "default"), "font_family": ("font", "default"),
    "title_size": ("title size", "default"), "axis_size": ("axis label size", "default"),
    "tick_size": ("tick size", "default"), "grid": ("grid", "automatic"),
    "background": ("background", "automatic"), "data_color": ("data color", "default"),
    "fit_color": ("fit color", "default"), "line_width": ("line width", "default"),
    "marker_size": ("marker size", "default"), "text": ("text style", "automatic"),
    "fit_label": ("fit label", "automatic"), "axes": ("axis", "automatic"),
    "annotations": ("annotation position", "automatic"),
}
_POINT_FIELDS = frozenset({"title_size", "axis_size", "tick_size", "legend_size"})
_TEXT_FIELDS = ("title", "x_label", "y_label")
#: The fields one element's style can move together (its text and its style).
_ELEMENT_FIELDS = frozenset({"title", "x_label", "y_label", "text", "fit_label"})
_SIDE_WORDS = {"bottom": "x axis", "left": "y axis", "top": "top axis", "right": "right axis"}
#: A ``PlotAppearance.axes`` key: a side, or ``<view>.<side>`` (``PlotAppearance.for_view``).
_PIN_KEY = re.compile(rf"(?:[a-z_]{{1,32}}\.)?(?:{'|'.join(AXIS_SIDES)})")


def valid_pin_key(key) -> bool:
    """Whether ``key`` can name a pin in ``PlotAppearance.axes``."""
    return isinstance(key, str) and _PIN_KEY.fullmatch(key) is not None


def _element_name(key: str) -> str:
    names = {"title": "title", "axis.bottom.label": "x label", "axis.left.label": "y label",
             "colorbar.label": "colour-bar label", FIT_LABEL_KEY: "fit label",
             "annotation.provenance": "caption"}
    if key in names:
        return names[key]
    parts = key.split(".")
    if key.startswith("axis.") and key.endswith(".ticks"):
        return f"{_SIDE_WORDS.get(parts[1], parts[1])} ticks"
    if key.startswith("axis.") or key.startswith("marginal."):
        return f"{parts[1]} label"
    if key.startswith("legend.entry."):
        name = key[len("legend.entry."):]
        return f"legend ‘{name if len(name) <= 20 else name[:19] + '…'}’"
    return key.split(".", 1)[-1].replace(".", " ")


def _spell_style(attr: str, value) -> str:
    if attr == "text":
        if value is None:
            return "text automatic"
        if not value.strip():
            return "hidden"
        return f"‘{value if len(value) <= 24 else value[:23] + '…'}’"
    word = {"family": "font", "size_pt": "size", "color": "colour", "align": "aligned",
            "bold": "bold", "italic": "italic"}[attr]
    if value is None:
        return f"{word} automatic"
    if isinstance(value, bool):
        return f"{word} {'on' if value else 'off'}"
    if attr == "size_pt":
        return f"size {value:g} pt"
    return f"{word} {value}"


def _describe_elements(before: "PlotAppearance", after: "PlotAppearance") -> str | None:
    """"title bold on", "y label colour #d55e00": one element's style step, or None."""
    keys = {key for key, _ in (before.text or ())} | {key for key, _ in (after.text or ())}
    keys |= set(_LEGACY_TEXT)
    if before.fit_label_state() != after.fit_label_state():
        keys.add(FIT_LABEL_KEY)
    moved = [(key, attr.name) for key in sorted(keys) for attr in fields(TextStyle)
             if getattr(before.text_style(key), attr.name) != getattr(after.text_style(key), attr.name)]
    if not moved or len({key for key, _ in moved}) != 1:
        return None
    key = moved[0][0]
    said = [_spell_style(attr, getattr(after.text_style(key), attr)) for _, attr in moved]
    if len(said) > 2:
        return f"{_element_name(key)} style ({len(said)} changes)"
    return f"{_element_name(key)} {', '.join(said)}"


def _describe_axes(before: "PlotAppearance", after: "PlotAppearance") -> str:
    keys = sorted({key for key, _ in (before.axes or ())} | {key for key, _ in (after.axes or ())})
    moved = [key for key in keys if before.axis_pin(key) != after.axis_pin(key)]
    # A bare pin saved before pins were kept per view, moved onto its view's key: one axis.
    bare = {key for key in moved if "." not in key and any(m.endswith("." + key) for m in moved)}
    moved = [key for key in moved if key not in bare]
    if len(moved) != 1:
        return f"{len(moved)} axes"
    key = moved[0]
    side = key.rpartition(".")[2]
    old, new = before.axis_pin(side if side in bare else key), after.axis_pin(key)
    name = _SIDE_WORDS[side]
    said = []
    if old.range() != new.range():
        said.append(f"range {new.min:g}–{new.max:g}" if new.range() else "range automatic")
    if old.tick_major != new.tick_major:
        said.append(f"major tick {new.tick_major:g}" if new.tick_major else "major tick automatic")
    if old.log != new.log:
        said.append("log automatic" if new.log is None else f"log {'on' if new.log else 'off'}")
    return f"{name} {', '.join(said)}"


def _describe_annotations(before: "PlotAppearance", after: "PlotAppearance") -> str:
    """"caption moved", as a fit-label drag is "fit label moved"; "caption position
    automatic" when it goes back to its window's own place."""
    old, new = dict(before.annotations or ()), dict(after.annotations or ())
    keys = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
    if len(keys) != 1:
        return f"{len(keys)} annotations moved"
    return f"{_element_name(keys[0])} {'moved' if keys[0] in new else 'position automatic'}"


def _describe_fit_label(before: "PlotAppearance", after: "PlotAppearance") -> str:
    old, new = before.fit_label_state(), after.fit_label_state()
    if old.visible != new.visible:
        return f"fit label {'shown' if new.visible else 'hidden'}"
    if old.in_legend != new.in_legend:
        return f"fit label in legend {'on' if new.in_legend else 'off'}"
    if (old.placement, old.offset) != (new.placement, new.offset):
        return "fit label moved" if new.placement == "custom" else f"fit label {new.placement}"
    if old.fields != new.fields:
        return "fit label numbers " + (", ".join(new.fields) or "none")
    return "fit label style"


def _spell(key: str, value) -> str:
    if value is None:
        return _STEP_WORDS[key][1]
    if key in _TEXT_FIELDS and not value.strip():
        return "hidden"
    if isinstance(value, bool):
        return "shown" if value else "hidden"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if key in _TEXT_FIELDS:
        return f"‘{value if len(value) <= 24 else value[:23] + '…'}’"
    return str(value)


def _describe_step(before: PlotAppearance, after: PlotAppearance) -> str:
    """Name the step from ``before`` to ``after`` for its Undo or Redo tooltip.

    Read off the two VALUES, so the stack holds nothing but appearances. A single field is
    named with both of its values, whatever moved it: a size typed back to Default is
    that, not a reset. Several fields that land exactly on a shipped preset are that
    preset, several that land on no override at all are a reset, and any other several
    are counted.
    """
    moved = [f.name for f in fields(PlotAppearance)
             if getattr(before, f.name) != getattr(after, f.name)]
    if moved == ["axes"]:
        return _describe_axes(before, after)
    if moved == ["annotations"]:
        return _describe_annotations(before, after)
    if moved and set(moved) <= _ELEMENT_FIELDS and ("text" in moved or "fit_label" in moved):
        element = _describe_elements(before, after)
        if element is not None:
            return element
        if moved == ["fit_label"]:
            return _describe_fit_label(before, after)
    if moved == ["text"]:
        changed = set(before.text or ()) ^ set(after.text or ())
        return f"{len({key for key, _ in changed})} text styles"
    if len(moved) == 1:
        key = moved[0]
        old, new = getattr(before, key), getattr(after, key)
        said_old, said_new = _spell(key, old), _spell(key, new)
        if key in _POINT_FIELDS:    # "12 → 14 pt": the unit once, on a side with a number
            if new is not None:
                said_new += " pt"
            elif old is not None:
                said_old += " pt"
        return f"{_STEP_WORDS[key][0]} {said_old} → {said_new}"
    for name, preset in _BUILTIN_PRESETS.items():
        # A preset keeps the figure's wording, so the step lands on the preset's LOOK.
        if after.style_only() == preset and before.style_only() != preset:
            return f"{name} preset"
    if not after.to_dict():
        return "reset figure"
    return f"{len(moved)} appearance changes"


class _AppearanceHistory:
    """A bounded undo/redo stack of :class:`PlotAppearance` VALUES, local to one widget.

    ``plots_panel._ConfigHistory``'s shape, and its arguments, without the origin
    machinery its import disclosures need: ``base`` is the value the controls are believed
    to show, ``_past`` what came before it (oldest first), ``_future`` what an undo
    stepped off. :meth:`note` records by DIFF, so a caller that says "something may have
    moved" too often costs one comparison, and walking a combo away and back pushes
    nothing. Qt-free.
    """

    __slots__ = ("base", "_past", "_future", "dropped")

    def __init__(self, base: PlotAppearance) -> None:
        self.base = base
        self._past: list[PlotAppearance] = []
        self._future: list[PlotAppearance] = []
        #: Steps thrown away at :data:`APPEARANCE_HISTORY_DEPTH`. Counted rather than
        #: merely dropped, because a silent bound looks exactly like a broken Undo button.
        self.dropped = 0

    def depth(self) -> tuple[int, int]:
        """``(steps back, steps forward)``."""
        return len(self._past), len(self._future)

    def last_step(self) -> tuple[PlotAppearance, PlotAppearance] | None:
        """``(before, after)`` of the step an undo would take back."""
        return (self._past[-1], self.base) if self._past else None

    def next_step(self) -> tuple[PlotAppearance, PlotAppearance] | None:
        """``(before, after)`` of the step a redo would replay."""
        return (self.base, self._future[-1]) if self._future else None

    def note(self, now: PlotAppearance) -> bool:
        """Record that the controls moved from :attr:`base` to ``now``. True if they had."""
        if now == self.base:
            return False
        self._past.append(self.base)
        self.base = now
        # A new step ends the redo branch: "forward" would lead somewhere no longer
        # reachable from here.
        self._future.clear()
        while len(self._past) > APPEARANCE_HISTORY_DEPTH:
            del self._past[0]
            self.dropped += 1
        return True

    def undo(self) -> PlotAppearance | None:
        if not self._past:
            return None
        self._future.append(self.base)
        self.base = self._past.pop()
        return self.base

    def redo(self) -> PlotAppearance | None:
        if not self._future:
            return None
        self._past.append(self.base)
        self.base = self._future.pop()
        return self.base

    def rebase(self, state: PlotAppearance) -> None:
        """Move the base WITHOUT recording a step."""
        self.base = state

    def reset(self, base: PlotAppearance) -> None:
        """Forget every step, and the ceiling count with them (``_ConfigHistory.reset``)."""
        self.base = base
        self._past.clear()
        self._future.clear()
        self.dropped = 0


class _SizeSpinBox(QtWidgets.QDoubleSpinBox):
    """A size whose 0 is Default and whose next value is the smallest size a figure draws.

    Run from 0 by single steps, the title, axis, tick and legend sizes showed 1-5 pt while
    ``from_mapping`` dropped them, so the figure and the history did nothing (sweep 10,
    S10-PLOT2). A step up from below ``floor`` lands on it and a step down below it lands on
    Default; a typed size below it is not taken until it grows (``"1"`` on the way to
    ``"12"``), and Return or leaving the box puts the last size back.
    """

    def __init__(self, floor: float, maximum: float) -> None:
        super().__init__()
        self._floor = floor
        self.setRange(0, maximum)

    def stepBy(self, steps: int) -> None:  # noqa: N802 -- Qt API
        target = self.value() + steps * self.singleStep()
        if not 0 < target < self._floor:
            super().stepBy(steps)
            return
        self.setValue(self._floor if steps > 0 else self.minimum())
        if self.style().styleHint(QtWidgets.QStyle.SH_SpinBox_SelectOnStep, None, self):
            self.selectAll()

    def validate(self, text: str, pos: int):
        state, text, pos = super().validate(text, pos)
        if (state == QtGui.QValidator.State.Acceptable
                and 0 < self.valueFromText(text) < self._floor):
            state = QtGui.QValidator.State.Intermediate
        return state, text, pos


class _PresetStore(QtCore.QObject):
    """Says the saved presets or templates changed from OUTSIDE the widget that lists them.

    Opening a project can add presets and templates (``gui.plot_preset_sync``) while windows
    that list them are open. Each lister connects a bound method, which Qt disconnects when
    that widget is destroyed -- unlike walking ``QApplication.allWidgets()``, which hands out
    wrappers for widgets already on their way out and crashed in testing.
    """

    changed = QtCore.Signal()


_PRESET_STORE: _PresetStore | None = None


def preset_store() -> _PresetStore:
    """The one :class:`_PresetStore` of this process."""
    global _PRESET_STORE
    if _PRESET_STORE is None:
        _PRESET_STORE = _PresetStore()
    return _PRESET_STORE


def _delete_all(objects, *_args) -> None:
    for obj in objects:
        if _alive(obj):
            obj.deleteLater()


class PlotAppearanceWidget(QtWidgets.QWidget):
    """Compact common controls. ``set_state`` is silent; user edits emit once.

    With ``history`` (the default) the widget keeps its own Undo and Redo of its
    appearance, apart from Edit ▸ Undo (design §4.2): every user change is one step, a run
    of edits closer together than :data:`APPEARANCE_COALESCE_MS` is one, and an undo or
    redo emits ``changed`` exactly as the user's own edit would, so the host re-applies it
    the same way. ``set_state`` is never a step. A host whose figure is replaced from
    outside -- a restore, a window blanked for an incoming project, a reset of everything
    -- calls :meth:`clear_history` after it.

    ``multi_view_labels`` is for a window whose views share ONE title and one pair of axis labels
    (Noise, Event waveforms, Comparison figures, Subpopulations): ``True``, or a few words for
    what the typed words cover (Subpopulations passes ``"both figures"``). The Title, X label
    and Y label captions say it (``Title (this view)``), as does the menu entry that adds a title
    on the figure, and so do their tool tips: those words belong to what is on screen, and go
    back to automatic when the figure comes to show something else (``gui.quantity_reset``).
    :attr:`title_scope` holds the words, "" for a one-figure window.

    ``shortcuts`` binds the platform's Undo and Redo keys to that history, scoped to the
    widget's window. A host whose window already binds those keys passes False: the same
    key bound twice in one window is ambiguous, and Qt then fires NEITHER. The Plots window
    passes ``history=False`` -- its own configuration stack already holds appearance, and
    one window must not have two stacks.

    ``shortcut_parent`` (default: this widget) is the widget those keys are created on. Qt
    answers a window shortcut only while its parent is visible, so on this widget the keys
    go quiet whenever its Appearance section is folded away -- while an edit made in the
    figure itself (``figure_text``) is still one of its steps. A page passes its own root
    to keep ⌘Z live in every section; a page behind another Explorer tab is hidden, so it
    still does not answer.

    ``reset_keeps`` (default: none) names the element keys whose text styles Reset
    figure leaves as they are: a page whose steps carry state stored elsewhere
    (Distribution summaries' legend names, kept in Source names and colors) passes them, so
    a Reset does not drop them as a step that changes nothing on screen.

    The Reset figure button (``reset_button``) sits in this section until a page with a
    toolbar takes it (:meth:`take_reset_button`); a window without one, the study figure
    editor, keeps it here.
    """

    changed = QtCore.Signal(object)

    def __init__(self, state: PlotAppearance | None = None, parent=None, *,
                 include_colors=True, compact=False, include_markers=True,
                 include_labels=True, include_legend=True,
                 multi_view_labels: bool | str = False, history=True, shortcuts=True,
                 shortcut_parent: QtWidgets.QWidget | None = None,
                 reset_keeps: Callable[[str], bool] | None = None):
        super().__init__(parent)
        self._state = state or PlotAppearance()
        self._reset_keeps = reset_keeps
        self._reset_taken = False       # see take_reset_button
        #: The sentence the last quantity change put on the host's status bar ("" for none).
        self._reset_said = ""
        self._loading = False
        #: How deep this widget is inside its own ``changed`` emission. A ``set_state``
        #: arriving then is the host answering the value it was just handed, not an
        #: install from outside (see :meth:`set_state`).
        self._emitting = 0
        self._history = _AppearanceHistory(self._state) if history else None
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(4)
        # TEXT FIRST (ux H2): the words on the figure are what people come here to change,
        # so Title, X label and Y label lead, above the preset and the font rows.
        self._multi_view_labels = bool(multi_view_labels)
        #: What the typed words cover in a window with several views: the Title box's caption
        #: and the figure's "Add title" entry say it. "" in a window with one figure.
        self.title_scope = ("" if not multi_view_labels else
                            "this view" if multi_view_labels is True else str(multi_view_labels))
        text_rows = []
        for key, caption in (("title", "Title"), ("x_label", "X label"), ("y_label", "Y label")):
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(self._automatic_placeholder(""))
            edit.setToolTip("Text only: axis scaling and numerical values stay unchanged. "
                            "Typed text is shown as typed; _{…} and ^{…} make sub- and "
                            "superscripts. Include the scientific unit in a custom axis label."
                            + (" Typed words go back to automatic when the figure comes to show "
                               "something else (another view, metric or embedding); the status "
                               "line says so and Undo brings them back." if self.title_scope
                               else ""))
            edit.textEdited.connect(lambda text, field=key: self._edit(field, text))
            setattr(self, key, edit)
            label = QtWidgets.QLabel(caption + (f" ({self.title_scope})" if self.title_scope else ""))
            label.setBuddy(edit)
            label.setVisible(include_labels)
            edit.setVisible(include_labels)
            text_rows.append((label, edit))
        for row, (label, edit) in enumerate(text_rows):
            layout.addWidget(label, row, 0)
            layout.addWidget(edit, row, 1, 1, 3)
        top = len(text_rows)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(["Choose appearance preset…", *_BUILTIN_PRESETS])
        self.preset.setToolTip("Appearance only: never changes data, fitting, axis scales, or analysis settings.")
        self.preset.activated.connect(self._apply_preset)
        self.reset_button = QtWidgets.QPushButton("Reset figure")
        self.reset_button.setAccessibleName("Reset figure")
        self.reset_button.setToolTip(
            "Put the text you typed on the figure (title, axis labels, fit-label wording, "
            "caption) and its look back to their defaults, as one Undo step. Data, fits and "
            "analysis settings are kept.")
        self.reset_button.clicked.connect(self._reset)
        # Undo and Redo sit beside Reset, the only other buttons here that move the whole
        # appearance. Without a history the cell is the bare button, as it always was.
        self.undo_button = self.redo_button = self._burst = self._reset_row = None
        reset_cell = self.reset_button
        if self._history is not None:
            self.undo_button = QtWidgets.QPushButton("Undo")
            self.undo_button.setAccessibleName("Undo appearance change")
            self.undo_button.clicked.connect(self.undo)
            self.redo_button = QtWidgets.QPushButton("Redo")
            self.redo_button.setAccessibleName("Redo appearance change")
            self.redo_button.clicked.connect(self.redo)
            for button in (self.undo_button, self.redo_button):
                # The text's own width, not the style's 80 px push-button floor. At the
                # floor the compact row wanted 308 px where the rest of the column wants
                # 287, so it would have widened every sidebar it sits in (measured
                # offscreen); at this width the column is as wide as it always was.
                button.setFixedWidth(button.fontMetrics().horizontalAdvance(button.text()) + 20)
            reset_cell = QtWidgets.QWidget()
            reset_cell.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            row_layout = QtWidgets.QHBoxLayout(reset_cell)
            row_layout.setContentsMargins(0, 0, 0, 0)
            for button in (self.reset_button, self.undo_button, self.redo_button):
                row_layout.addWidget(button)
            self._reset_row = row_layout
            self._burst = QtCore.QTimer(self)
            self._burst.setSingleShot(True)
            self._burst.setInterval(APPEARANCE_COALESCE_MS)
            self._burst.timeout.connect(self._close_burst)
            if shortcuts:
                # Window-scoped, and live only while this widget is on screen: Qt matches
                # a WindowShortcut whose parent is visible and enabled in the active
                # window, so a page behind another Explorer tab, or an Appearance section
                # the workflow bar has folded away, does not answer. Each resolved key
                # once (key_sequences says why). A focused text box keeps the key for its
                # own typing, as everywhere else in the app. On ``shortcut_parent`` when
                # the host gives one (the class docstring says why).
                owner = shortcut_parent if shortcut_parent is not None else self
                self._shortcuts: list[QtGui.QShortcut] = []
                for keys, slot in ((undo_sequences(), self.undo), (redo_sequences(), self.redo)):
                    for key in keys:
                        shortcut = QtGui.QShortcut(key, owner)
                        self._shortcuts.append(shortcut)
                        shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
                        shortcut.activated.connect(slot)
                if owner is not self:
                    # Gone with this widget, not with the page: a widget built again on
                    # the same page would otherwise find its keys bound twice (ambiguous).
                    self.destroyed.connect(functools.partial(_delete_all, tuple(self._shortcuts)))
        # Three buttons need two columns of the four; the preset gives up the third, which
        # costs the grid nothing where the extra column would have cost it the row's width.
        wide = reset_cell is not self.reset_button
        layout.addWidget(self.preset, top, 0, 1, 2 if wide else 3)
        layout.addWidget(reset_cell, top, 2 if wide else 3, 1, 2 if wide else 1)
        # The short list (Arial first, the unset default) and Other font…; a picked row
        # reports once, and showing a stored value (``_write``) never does.
        self.font_family = FontPicker()
        self.font_family.changed.connect(lambda family: self._edit("font_family", family))
        self._add_control(layout, top + 1, 0, "Font", self.font_family)
        self.background = QtWidgets.QComboBox()
        # "Automatic", the one word for "not overridden" (ux L3), as the label boxes say it.
        for label, value in (("Automatic", None), ("Light", "light"), ("Dark", "dark")):
            self.background.addItem(label, value)
        self.background.currentIndexChanged.connect(
            lambda: self._edit("background", self.background.currentData()))
        self._add_control(layout, top + 1, 2, "Background", self.background)
        for key, label, row, col in (
                ("title_size", "Title (pt)", 2, 0),
                ("axis_size", "Axis labels (pt)", 2, 2),
                ("tick_size", "Ticks (pt)", 3, 0),
                ("line_width", "Line width", 4, 0),
                ("marker_size", "Marker size", 4, 2)):
            spin = _SizeSpinBox(*_SIZE_RANGES[key])
            spin.setDecimals(1)
            spin.setSingleStep(.5 if key == "line_width" else 1)
            spin.setSpecialValueText("Default")
            spin.valueChanged.connect(lambda value, field=key: self._edit(field, value or None))
            setattr(self, key, spin)
            self._add_control(layout, top + row, col, label, spin)
        self.grid = QtWidgets.QComboBox()
        for label, value in (("Automatic", None), ("Show", True), ("Hide", False)):
            self.grid.addItem(label, value)
        self.grid.currentIndexChanged.connect(lambda: self._edit("grid", self.grid.currentData()))
        self._add_control(layout, top + 3, 2, "Grid", self.grid)
        for key, label, col in (("data_color", "Data color", 0), ("fit_color", "Fit color", 2)):
            button = QtWidgets.QPushButton("Default…")
            button.clicked.connect(lambda checked=False, field=key: self._choose_color(field))
            setattr(self, key, button)
            self._add_control(layout, top + 5, col, label, button)
        note = QtWidgets.QLabel("Appearance only. Axis scales, numerical values and scientific settings are preserved.")
        note.setWordWrap(True)
        layout.addWidget(note, top + 6, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        if not include_colors:
            for row, column in ((5, 0), (5, 1), (5, 2), (5, 3)):
                layout.itemAtPosition(top + row, column).widget().hide()
            # Category-colored views still share typography and canvas presets.
            # Their host ignores data/fit overrides so category identities stay intact.
        if not include_markers:
            layout.itemAtPosition(top + 4, 2).widget().hide()
            self.marker_size.hide()
        if compact:
            pairs = [(layout.itemAtPosition(top + row, col).widget(),
                      layout.itemAtPosition(top + row, col + 1).widget())
                     for row, col in ((1, 0), (1, 2), (2, 0), (2, 2), (3, 0),
                                      (3, 2), (4, 0), (4, 2), (5, 0), (5, 2))]
            while layout.count():
                layout.takeAt(0)
            for row, (label, edit) in enumerate(text_rows):
                layout.addWidget(label, row, 0)
                layout.addWidget(edit, row, 1)
            row = top
            layout.addWidget(self.preset, row, 0, 1, 2)
            row += 1
            for label, control in pairs:
                if control.isHidden():
                    continue
                layout.addWidget(label, row, 0)
                layout.addWidget(control, row, 1)
                row += 1
            layout.addWidget(reset_cell, row, 0, 1, 2)
            layout.addWidget(note, row + 1, 0, 1, 2)
            layout.setColumnStretch(3, 0)
        columns = 2 if compact else 4
        row = layout.rowCount()
        self.legend_visible = QtWidgets.QComboBox()
        for label, value in (("Automatic", None), ("Show", True), ("Hide", False)):
            self.legend_visible.addItem(label, value)
        self.legend_visible.currentIndexChanged.connect(
            lambda: self._edit("legend_visible", self.legend_visible.currentData()))
        self.legend_size = _SizeSpinBox(*_SIZE_RANGES["legend_size"])
        self.legend_size.setSpecialValueText("Default")
        self.legend_size.valueChanged.connect(lambda value: self._edit("legend_size", value or None))
        for label, control in (("Legend", self.legend_visible), ("Legend text (pt)", self.legend_size)):
            caption = QtWidgets.QLabel(label)
            layout.addWidget(caption, row, 0)
            layout.addWidget(control, row, 1, 1, columns - 1)
            caption.setVisible(include_legend)
            control.setVisible(include_legend)
            row += 1
        self.fit_label_group = self._build_fit_label_group()
        layout.addWidget(self.fit_label_group, row, 0, 1, columns)
        row += 1
        self.user_presets = QtWidgets.QComboBox()
        self.user_presets.setToolTip("Reusable appearance only — no source, column, fit, bin or analysis settings.")
        self.user_presets.activated.connect(self._activate_user_preset)
        layout.addWidget(self.user_presets, row, 0, 1, columns)
        row += 1
        self.save_preset_button = QtWidgets.QPushButton("Save appearance preset…")
        self.save_preset_button.setToolTip("Save the look: fonts, sizes, colours and styles. "
                                           "Titles and labels you typed stay with their figure.")
        self.delete_preset_button = QtWidgets.QPushButton("Delete selected preset")
        self.save_preset_button.clicked.connect(self._ask_save_preset)
        self.delete_preset_button.clicked.connect(self._delete_selected_preset)
        layout.addWidget(self.save_preset_button, row, 0, 1, columns)
        layout.addWidget(self.delete_preset_button, row + 1, 0, 1, columns)
        self.refresh_user_presets()
        preset_store().changed.connect(self.refresh_user_presets)
        self.set_state(self._state)

    @staticmethod
    def _add_control(layout, row, column, label, control):
        label_widget = QtWidgets.QLabel(label)
        label_widget.setBuddy(control)
        layout.addWidget(label_widget, row, column)
        layout.addWidget(control, row, column + 1)

    def _automatic_placeholder(self, text: str) -> str:
        if self._multi_view_labels and not text:
            return "Automatic (each view's own)"
        return f"Automatic: {text}" if text else "Automatic"

    def set_automatic_text(self, field: str, text: str) -> None:
        """Show the window's automatic ``title`` / ``x_label`` / ``y_label`` as the box's
        placeholder ("Automatic: Dwell (ms)"), so it is easy to copy and change."""
        box = getattr(self, field, None) if field in _TEXT_FIELDS else None
        if box is not None:
            box.setPlaceholderText(self._automatic_placeholder(plain_text(text)))

    # -- the fit label section --------------------------------------------------------------
    _FIT_FIELD_WORDS = {"equation": "Equation", "r2": "R²", "n": "n", "ci": "CI",
                        "bic": "BIC", "ks": "KS"}
    _FIT_FIELD_TIPS = {
        "r2": "The fit's R². Ticked, a fit whose R² is undefined or has no residual degrees of "
              "freedom says so here instead of printing a bare number. Unticked, the "
              "automatic text makes no claim about R² and Details still says; wording of your "
              "own that names {r2} prints it either way.",
        "ks": "The numerical Kolmogorov–Smirnov statistic of a histogram fit (Distributions). "
              "The verdict and warnings stay in Details and in figure-bundle provenance, not "
              "on the plot or in Save image artwork. This does not change the fit."}

    def _build_fit_label_group(self) -> QtWidgets.QGroupBox:
        """Show, the numbers, Text (template), Size, Colour, Position, In legend instead.
        Hidden until the host says it draws a fit label (:meth:`set_fit_label_available`)."""
        group = QtWidgets.QGroupBox("Fit label")
        group.setObjectName("plotAppearanceFitLabel")
        form = QtWidgets.QFormLayout(group)
        form.setContentsMargins(6, 4, 6, 4)
        self.fit_label_show = QtWidgets.QCheckBox("Show the fit on the plot")
        self.fit_label_show.toggled.connect(lambda on: self._edit_fit_label(visible=on))
        form.addRow(self.fit_label_show)
        # THREE TO A LINE: six boxes on one line were 460 px wide, so in Distributions' 344 px
        # column BIC and KS sat off its right edge -- and with Fit values on plot gone this row
        # is the only way to tick them.
        row = QtWidgets.QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.fit_label_fields: dict[str, QtWidgets.QCheckBox] = {}
        for index, name in enumerate(FIT_FIELDS):
            box = QtWidgets.QCheckBox(self._FIT_FIELD_WORDS[name])
            box.setToolTip(self._FIT_FIELD_TIPS.get(name, ""))
            box.toggled.connect(lambda _on: self._edit_fit_label(fields=tuple(
                f for f in FIT_FIELDS if self.fit_label_fields[f].isChecked())))
            self.fit_label_fields[name] = box
            row.addWidget(box, index // 3, index % 3)
        row.setColumnStretch(3, 1)       # in a wide strip the spare width goes here, not between boxes
        form.addRow("Numbers", row)
        self.fit_label_template = QtWidgets.QLineEdit()
        self.fit_label_template.setPlaceholderText("Automatic: the numbers ticked above")
        self.fit_label_template.setToolTip(
            "Your words, with live numbers: {equation} {r2} {n} {ci} {bic} {ks} {source}. "
            "A re-fit keeps your words and updates the numbers. An R² refusal (for example "
            "zero degrees of freedom) is always shown when R² is asked for.")
        self.fit_label_template.textEdited.connect(lambda text: self._edit_fit_label(template=text))
        form.addRow("Text", self.fit_label_template)
        self.fit_label_size = _SizeSpinBox(6, 72)
        self.fit_label_size.setSpecialValueText("Default")
        self.fit_label_size.valueChanged.connect(lambda v: self._edit_fit_label(size_pt=v or None))
        form.addRow("Size (pt)", self.fit_label_size)
        self.fit_label_color = QtWidgets.QComboBox()
        self.fit_label_color.addItem("Automatic (the fit's colour)", None)
        for colour in ("#000000", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#7f7f7f"):
            self.fit_label_color.addItem(colour, colour)
        self.fit_label_color.currentIndexChanged.connect(
            lambda _i: self._edit_fit_label(color=self.fit_label_color.currentData()))
        form.addRow("Colour", self.fit_label_color)
        self.fit_label_position = QtWidgets.QComboBox()
        for label, value in (("Automatic", "auto"), ("Top left", "top-left"),
                             ("Top right", "top-right"), ("Bottom left", "bottom-left"),
                             ("Bottom right", "bottom-right"), ("Where I dragged it", "custom")):
            self.fit_label_position.addItem(label, value)
        self.fit_label_position.setToolTip("Drag the label on the plot to put it anywhere.")
        self.fit_label_position.activated.connect(self._fit_label_position_picked)
        form.addRow("Position", self.fit_label_position)
        self.fit_label_in_legend = QtWidgets.QCheckBox("In legend instead")
        self.fit_label_in_legend.toggled.connect(lambda on: self._edit_fit_label(in_legend=on))
        form.addRow(self.fit_label_in_legend)
        group.hide()
        return group

    def set_fit_label_available(self, available: bool, *, in_legend: bool = True) -> None:
        """Show the Fit label section: the host draws a fit label (``fit_label``).

        ``in_legend=False`` for a window that does not put the label's lines in its legend
        row: there "In legend instead" would take the numbers, and the R² refusal no
        template may drop, off the figure."""
        self.fit_label_group.setVisible(bool(available))
        self.fit_label_in_legend.setVisible(bool(in_legend))

    def _fit_label_position_picked(self, _index: int) -> None:
        placement = self.fit_label_position.currentData()
        label = self._state.fit_label_state()
        if placement == "custom" and label.offset is None:
            self._write(self._state)       # nothing dragged yet: nothing to go back to
            return
        self._edit_fit_label(placement=placement,
                             offset=label.offset if placement == "custom" else None)

    def _edit_fit_label(self, **changes) -> None:
        if self._loading:
            return
        self._edit("fit_label", replace(self._state.fit_label_state(), **changes).to_dict())

    def state(self) -> PlotAppearance:
        return self._state

    def set_state(self, state: PlotAppearance | Mapping[str, Any]):
        """Adopt ``state`` silently: a restore, a recipe load, a host's reset or its answer.

        Never an undo step. From outside, it first closes any open run of edits -- that
        run was the user's and stays a step -- then moves the history's base without one.
        From INSIDE this widget's own ``changed``, where a host normalises or refuses the
        value it was just handed, it only moves the controls: the step being recorded is
        the user's, and it records what the host left on screen, so a refused edit records
        nothing.
        """
        state = state if isinstance(state, PlotAppearance) else PlotAppearance.from_mapping(state)
        outside = self._history is not None and not self._emitting
        if outside:
            self._close_burst()
        self._write(state)
        if outside:
            self._history.rebase(self._state)
            self._sync_history_buttons()

    def _write(self, state: PlotAppearance) -> None:
        """Put ``state`` on the controls without any of them reporting an edit."""
        self._state = state
        self._loading = True
        try:
            self.font_family.set_value(self._state.font_family)
            for key in ("title_size", "axis_size", "tick_size", "line_width", "marker_size"):
                getattr(self, key).setValue(getattr(self._state, key) or 0)
            for key in ("title", "x_label", "y_label"):
                getattr(self, key).setText(getattr(self._state, key) or "")
            self.legend_size.setValue(self._state.legend_size or 0)
            self.legend_visible.setCurrentIndex(self.legend_visible.findData(self._state.legend_visible))
            for key in ("background", "grid"):
                combo = getattr(self, key)
                combo.setCurrentIndex(combo.findData(getattr(self._state, key)))
            for key in ("data_color", "fit_color"):
                button = getattr(self, key)
                value = getattr(self._state, key)
                button.setText(f"{value}…" if value else "Default…")
                button.setStyleSheet(f"border-bottom: 3px solid {value};" if value else "")
            label = self._state.fit_label_state()
            self.fit_label_show.setChecked(label.visible)
            for name, box in self.fit_label_fields.items():
                box.setChecked(name in label.fields)
            if self.fit_label_template.text() != label.template:
                self.fit_label_template.setText(label.template)
            self.fit_label_size.setValue(label.size_pt or 0)
            index = self.fit_label_color.findData(label.color)
            if index < 0 and label.color:
                self.fit_label_color.addItem(label.color, label.color)
                index = self.fit_label_color.count() - 1
            self.fit_label_color.setCurrentIndex(max(0, index))
            self.fit_label_position.setCurrentIndex(
                max(0, self.fit_label_position.findData(label.placement)))
            self.fit_label_in_legend.setChecked(label.in_legend)
        finally:
            self._loading = False

    def set_legend_available(self, available: bool) -> None:
        for widget in (self.legend_visible, self.legend_size):
            widget.setEnabled(available)
            widget.setToolTip("Appearance of existing legend entries." if available else
                              "This view has no legend entries; no data or legend is invented.")

    _PRESETS_KEY = "plot_appearance/user_presets_v1"

    @classmethod
    def user_preset_states(cls) -> dict[str, dict]:
        raw = QtCore.QSettings().value(cls._PRESETS_KEY, "{}")
        try:
            values = json.loads(raw) if isinstance(raw, str) else {}
            if not isinstance(values, dict):
                return {}
            # Style only, on the way out as on the way in: a preset saved before presets
            # dropped wording (or carried in by a project) must not stamp its title either.
            return {name: PlotAppearance.from_mapping(state).style_only().to_dict()
                    for name, state in values.items()
                    if isinstance(name, str) and name.strip() and isinstance(state, dict)}
        except (TypeError, ValueError):
            return {}

    def refresh_user_presets(self) -> None:
        selected = self.user_presets.currentData()
        with QtCore.QSignalBlocker(self.user_presets):
            self.user_presets.clear()
            self.user_presets.addItem("Choose saved appearance…", None)
            for name in sorted(self.user_preset_states(), key=str.casefold):
                self.user_presets.addItem(name, name)
            self.user_presets.setCurrentIndex(max(0, self.user_presets.findData(selected)))

    def showEvent(self, event) -> None:
        # Another open analytical page may have saved a preset since construction.
        self.refresh_user_presets()
        super().showEvent(event)

    def save_user_preset(self, name: str, *, overwrite: bool = False) -> bool:
        name = str(name).strip()[:100]
        if not name:
            return False
        values = self.user_preset_states()
        if name in values and not overwrite:
            return False
        # A preset is a LOOK (owner, Q-T2): a preset made on I-V used to stamp its title onto
        # the noise spectrum. The wording stays with the figure it was typed on.
        values[name] = self.state().style_only().to_dict()
        settings = QtCore.QSettings()
        settings.setValue(self._PRESETS_KEY, json.dumps(values, sort_keys=True, allow_nan=False))
        settings.sync()
        self.refresh_user_presets()
        self.user_presets.setCurrentIndex(self.user_presets.findData(name))
        return settings.status() == QtCore.QSettings.NoError

    def apply_user_preset(self, name: str) -> bool:
        values = self.user_preset_states()
        if name not in values:
            return False
        self._apply(self._state.restyled(PlotAppearance.from_mapping(values[name])))
        return True

    def apply_appearance(self, state: PlotAppearance | Mapping[str, Any]) -> None:
        """Install ``state`` as the USER's change: announced, and one Undo step.

        For a host recalling a window template (``gui.plot_templates``), whose appearance is
        a gesture exactly like picking a saved appearance -- never :meth:`set_state`, which
        is the silent restore and would leave the step out of this plot's history.
        """
        self._apply(state if isinstance(state, PlotAppearance)
                    else PlotAppearance.from_mapping(state))

    def reset_typed_words(self, shows: str, status: Any = None,
                          only: Sequence[str] | None = None,
                          axes: Sequence[Sequence[str]] = (), primary: str = "") -> str:
        """The figure's quantity changed: its typed title, axis labels and fit-label wording
        go back to automatic, and so does the pin of each axis in ``axes``, as ONE step of this
        history. Returns the status sentence, which names only what was cleared ("Title, Y
        label and Y range reset: the plot now shows ``shows``.").

        The look stays (``gui.quantity_reset`` has the rule). A gesture like a preset: announced
        through ``changed`` after the host has answered, so the host redraws as for any edit,
        and Undo brings the words and the pins back. Nothing typed and no pin on ``axes``
        means no step, no ``changed`` and ``""``.

        ``axes`` are the axes whose quantity moved, ``(view, side)`` or ``(view, side, name)``
        (``quantity_reset.pinned_axes``), and ``primary`` the window's primary view, whose
        pin a bare side key may be. ``status`` is the host's status bar when the sentence is
        read there: it is shown for ten seconds, and a change that resets nothing takes the
        earlier sentence off it (``quantity_reset.announce``). ``only`` limits the words, e.g.
        ``("fit_label",)``; ``()`` is no word at all, a pin alone.
        """
        from .quantity_reset import announce, cleared_pins, cleared_words, reset_sentence

        state, names = cleared_words(self._state, only)
        state, pins = cleared_pins(state, axes, primary)
        if names or pins:
            self._apply(state)
        said = reset_sentence(names + pins, shows)
        if status is not None:
            self._reset_said = announce(status, said, self._reset_said)
        return said

    def delete_user_preset(self, name: str) -> bool:
        values = self.user_preset_states()
        if name not in values:
            return False
        del values[name]
        settings = QtCore.QSettings()
        settings.setValue(self._PRESETS_KEY, json.dumps(values, sort_keys=True, allow_nan=False))
        settings.sync()
        self.refresh_user_presets()
        return settings.status() == QtCore.QSettings.NoError

    def _activate_user_preset(self, index: int) -> None:
        name = self.user_presets.itemData(index)
        if name:
            self.apply_user_preset(name)

    def _ask_save_preset(self) -> None:
        name, accepted = QtWidgets.QInputDialog.getText(self, "Save appearance", "Preset name:")
        if not accepted or not name.strip():
            return
        overwrite = name.strip()[:100] in self.user_preset_states()
        if overwrite and QtWidgets.QMessageBox.question(
                self, "Replace appearance preset?", f"Replace the saved appearance ‘{name.strip()[:100]}’?") != QtWidgets.QMessageBox.Yes:
            return
        self.save_user_preset(name, overwrite=overwrite)

    def _delete_selected_preset(self) -> None:
        name = self.user_presets.currentData()
        if name and QtWidgets.QMessageBox.question(
                self, "Delete appearance preset?", f"Delete the saved appearance ‘{name}’?") == QtWidgets.QMessageBox.Yes:
            self.delete_user_preset(name)

    def _edit(self, key, value):
        """A control moved: one property, held open as part of a run of edits.

        Spin boxes, the editable font box and the label boxes report every keystroke, so
        "14" typed into Title (pt) is two values a few hundred milliseconds apart. The run
        becomes ONE step once :data:`APPEARANCE_COALESCE_MS` passes without another edit,
        or sooner when Undo, Redo, a one-gesture change or ``set_state`` closes it.
        """
        if self._loading:
            return
        updated = PlotAppearance.from_mapping({**self._state.to_dict(), key: value})
        if updated != self._state:
            self._state = updated
            self._announce(updated)
            if self._history is not None:
                self._burst.start()
                self._sync_history_buttons()

    def _apply(self, state: PlotAppearance, *, always: bool = False) -> None:
        """A one-gesture change -- a preset, a saved appearance, a colour, Reset: one step.

        Recorded after the host has answered ``changed``, so the step holds what the host
        left on screen. ``always`` announces even a state these controls already show, as
        a built-in preset always has: the Plots window installs whole configurations
        without writing them back here, so its controls can lag the window, and the preset
        must still reach it. Announcing an unchanged state records no step.
        """
        self._close_burst()      # a run of edits still open is its own, earlier step
        if state == self._state and not always:
            return
        self._write(state)
        self._announce(state)
        self._close_burst()

    def _announce(self, state: PlotAppearance) -> None:
        self._emitting += 1
        try:
            self.changed.emit(state)
        finally:
            self._emitting -= 1

    def _choose_color(self, field):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(getattr(self._state, field) or "#0072B2"), self,
            "Choose data color" if field == "data_color" else "Choose fit color")
        if color.isValid():
            self._apply(PlotAppearance.from_mapping({**self._state.to_dict(), field: color.name()}))

    def _reset(self):
        state = PlotAppearance()
        if self._reset_keeps is not None:
            for key, style in self._state.text or ():
                if self._reset_keeps(key):
                    state = state.with_text_style(key, style)
        self._apply(state)

    def take_reset_button(self) -> QtWidgets.QPushButton:
        """Hand the Reset figure button to the page's toolbar: the SAME button, out of this
        section, still clicking :meth:`_reset` and still named by ``reset_button``. Undo and
        Redo stay, side by side where Reset was. Taking it again returns it where it is."""
        button = self.reset_button
        if not self._reset_taken:
            self._reset_taken = True
            button.setParent(None)      # its layout lets go of a child that leaves
            if self._reset_row is not None:
                self._reset_row.addStretch(1)
        return button

    def _apply_preset(self, index):
        presets = tuple(_BUILTIN_PRESETS.values())
        if not 1 <= index <= len(presets):
            return
        self.preset.setCurrentIndex(0)
        self._apply(self._state.restyled(presets[index - 1]), always=True)

    # -- history ------------------------------------------------------------------------
    def undo(self) -> bool:
        """Step the appearance back one change. True if it moved.

        A run of edits still open is committed first, so Undo right after typing takes
        back the typing rather than the step before it.
        """
        if self._history is None:
            return False
        self._close_burst()
        return self._step(self._history.undo, self._history.redo)

    def redo(self) -> bool:
        """Step the appearance forward one change. True if it moved."""
        if self._history is None:
            return False
        self._close_burst()
        return self._step(self._history.redo, self._history.undo)

    def clear_history(self) -> None:
        """Forget every step, for a host whose figure was just replaced.

        A step back would restyle what the window no longer shows, so the history ends
        with the data it was made on, as the Plots window's does
        (``PlotsWindow.reset_state``). Call it after the ``set_state`` that installs the
        replacement.
        """
        if self._history is None:
            return
        self._burst.stop()
        self._history.reset(self._state)
        self._sync_history_buttons()

    def _step(self, move, unmove) -> bool:
        """Move the stack one step with ``move`` and hand the host the state it lands on.

        ``unmove`` is ``move``'s inverse, for a host that refuses the step.
        """
        left = self._state
        state = move()
        if state is None:
            return False
        self._write(state)
        # ``changed`` exactly as after a user edit: the host re-applies and re-saves the
        # value the way it always does, so an undo cannot be styled differently from the
        # edit it takes back.
        self._announce(state)
        if state != left and self._state == left:
            # The host REFUSED it and put back what it showed, as Comparison figures does
            # when the figure will not render. Nothing moved, so the stack moves back too:
            # re-basing here would lose the step, and leave a Redo that replays a step to
            # where the plot already is.
            unmove()
            self._sync_history_buttons()
            return False
        # Re-based onto what is on the controls once the host has answered, as the Plots
        # window re-bases after its own installs: a base that disagreed with the controls
        # would make the next edit push a phantom step.
        self._history.rebase(self._state)
        self._sync_history_buttons()
        return True

    def _close_burst(self) -> None:
        """Close any open run of edits: one step, if the controls moved from the base."""
        if self._history is None:
            return
        self._burst.stop()
        self._history.note(self._state)
        self._sync_history_buttons()

    def _sync_history_buttons(self) -> None:
        """Enable and name Undo and Redo from the stack, counting an OPEN run of edits.

        An open run is offered to Undo (committing it is the first thing :meth:`undo`
        does) and withdraws Redo, which committing it will clear -- the Plots window's
        rule (``PlotsWindow._sync_config_buttons``), in its words.
        """
        history = self._history
        if history is None or self.undo_button is None:
            return
        pending = history.base != self._state
        back, forward = history.depth()
        want_undo, want_redo = bool(back or pending), bool(forward and not pending)
        # A focused button about to be disabled hands focus to its partner on the same
        # row first. Left to Qt, focus walks the tab chain, and the scroll area these
        # controls sit in scrolls after it -- measured in the Plots window, where it put a
        # different control under a pointer that had not moved
        # (``PlotsWindow._retire_history_button``).
        for dying, wanted, partner, partner_wanted in (
                (self.undo_button, want_undo, self.redo_button, want_redo),
                (self.redo_button, want_redo, self.undo_button, want_undo)):
            if not wanted and dying.hasFocus():
                if partner_wanted:
                    partner.setEnabled(True)
                    partner.setFocus()
                else:
                    dying.clearFocus()
        self.undo_button.setEnabled(want_undo)
        self.redo_button.setEnabled(want_redo)
        lost = history.dropped
        plural = "s" if lost != 1 else ""
        ceiling = (f". Its oldest history was dropped at the {APPEARANCE_HISTORY_DEPTH}-step "
                   f"ceiling ({lost} step{plural} lost).") if lost else ""
        step = (history.base, self._state) if pending else history.last_step()
        if step is not None:
            self.undo_button.setToolTip(f"Undo: {_describe_step(*step)}{ceiling}")
        elif lost:
            self.undo_button.setToolTip(
                f"Nothing left to undo — this plot's oldest appearance history was dropped "
                f"at the {APPEARANCE_HISTORY_DEPTH}-step ceiling ({lost} step{plural} lost).")
        else:
            self.undo_button.setToolTip("Nothing to undo yet.")
        step = history.next_step()
        self.redo_button.setToolTip(
            "Redo is cleared by a new change." if pending and step else
            f"Redo: {_describe_step(*step)}" if step else "Nothing to redo.")


def _item_defaults(item):
    defaults = getattr(item, "_nanoscope_appearance_defaults", None)
    if defaults is None:
        defaults = {key: item.opts.get(key) for key in
                    ("pen", "brush", "fillBrush", "size", "symbolPen", "symbolBrush", "symbolSize")}
        item._nanoscope_appearance_defaults = defaults
    return defaults


def _style_data_item(item, state, color):
    if not isinstance(item, (pg.PlotDataItem, pg.ScatterPlotItem, pg.BarGraphItem)):
        return
    defaults = _item_defaults(item)
    pen = defaults["pen"]
    if isinstance(item, pg.BarGraphItem):
        brush = defaults["brush"]
        if color:
            tint = QtGui.QColor(color)
            if brush is not None:
                tint.setAlpha(pg.mkBrush(brush).color().alpha())
            brush = pg.mkBrush(tint)
        if pen is not None or state.line_width is not None:
            pen = pg.mkPen(pen if pen is not None else
                           (color or pg.mkBrush(brush).color()))
            if color:
                pen.setColor(QtGui.QColor(color))
            if state.line_width is not None:
                pen.setWidthF(state.line_width)
        # Geometry fields (x/y/width/height) are never passed through this seam.
        item.setOpts(brush=brush, pen=pen)
    elif isinstance(item, pg.PlotDataItem):
        if pen is not None:
            pen = pg.mkPen(pen)
            if color:
                pen.setColor(QtGui.QColor(color))
            if state.line_width is not None:
                pen.setWidthF(state.line_width)
        item.setPen(pen)
        fill = defaults["fillBrush"]
        if fill is not None:
            brush = pg.mkBrush(fill)
            if color:
                tint = QtGui.QColor(color)
                tint.setAlpha(brush.color().alpha())
                brush = pg.mkBrush(tint)
            item.setFillBrush(brush)
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
                          fit_items: Sequence = (), title_prefix: str = "") -> None:
    """Style existing items, without ``clear``, ``setData``, fitting, or autoRange.

    Original graphical properties are recorded on first opt-in so Reset restores
    them. Data/fit items replaced by explicit analysis refresh acquire their own
    original styles. Existing manual zoom is retained.
    """
    defaults = getattr(plot, "_nanoscope_appearance_defaults", None)
    if defaults is None and not state.to_dict():
        _draw_in_figure_family(plot, None)
        return
    if defaults is None:
        defaults = {
            "title_opts": dict(plot.titleLabel.opts),
            "title_text": plot.titleLabel.text,
            "applied_title_text": plot.titleLabel.text,
            "axes": {side: {"label": dict(plot.getAxis(side).labelStyle),
                            "text": plot.getAxis(side).labelText,
                            "applied_text": plot.getAxis(side).labelText,
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
    elements = dict(state.text or ())
    # Read by figure_composition: a style its model cannot replay makes a frozen panel.
    plot._nanoscope_text_elements = elements
    try:
        foreground = {"light": "#20242b", "dark": "#e6e9ef"}.get(state.background)
        if plot_widget is not None:
            background = {"light": "#ffffff", "dark": "#151a21"}.get(state.background)
            plot_widget.setBackground(background if background else defaults["background"])
        for side, original in defaults["axes"].items():
            axis = plot.getAxis(side)
            if axis.labelText != original["applied_text"]:
                original["text"] = axis.labelText
            override = state.x_label if side == "bottom" else state.y_label if side == "left" else None
            # The user's label is LITERAL text (owner, Q-T1): escaped, so "a<b> c" draws as
            # typed instead of turning b into a bold tag. The window's own label is its HTML.
            axis.labelText = rich_text(override) if override else original["text"]
            original["applied_text"] = axis.labelText
            label_style = dict(original["label"])
            label_style["font-family"] = html_font_family(state.font_family)     # unset: Arial
            if state.axis_size is not None:
                label_style["font-size"] = f"{state.axis_size:g}pt"
            if foreground:
                label_style["color"] = foreground
            # Element override, then the figure-wide fields above, then the window's own.
            label_style.update(css_args(elements.get(f"axis.{side}.label", TextStyle())))
            ticks = elements.get(f"axis.{side}.ticks", TextStyle())
            # Always a font of our own, in the figure family (Arial when unset, or when the
            # family is not installed: its substitute), never the application font.
            font = figure_qfont(ticks.family or state.font_family,
                                original["font"] or QtWidgets.QApplication.font())
            if state.tick_size is not None:
                font.setPointSizeF(state.tick_size)
            if ticks.size_pt:
                font.setPointSizeF(ticks.size_pt)
            if ticks.bold is not None:
                font.setBold(ticks.bold)
            if ticks.italic is not None:
                font.setItalic(ticks.italic)
            axis.setStyle(tickFont=font)
            axis.setPen(foreground or original["pen"])
            axis.setTextPen(ticks.color or foreground or original["text_pen"])
            # After setTextPen, which writes its own colour into the label's CSS.
            axis.labelStyle = label_style
            axis._updateLabel()  # preserve text, units and automatic SI-prefix state
            axis.setGrid(original["grid"] if state.grid is None else
                         (0.15 * 255 if state.grid and side in ("bottom", "left") else False))
            pin = state.axis_pin(side)
            if pin.tick_major is not None or getattr(axis, "major_tick", None) is not None \
                    or "tickSpacing" in vars(axis):
                set_major_tick(axis, pin.tick_major)
        title = plot.titleLabel
        if title.text != defaults["applied_title_text"]:
            defaults["title_text"] = title.text
        last_title_color = defaults.get("applied_title_color", defaults["title_opts"].get("color"))
        if title.opts.get("color") != last_title_color:
            # A fresh scientific render can deliberately recolor its fit/warning
            # title. Keep that original color available for palette reset.
            defaults["title_opts"]["color"] = title.opts.get("color")
        title.opts = dict(defaults["title_opts"])
        title.opts["family"] = html_font_family(state.font_family)               # unset: Arial
        if state.title_size is not None:
            title.opts["size"] = f"{state.title_size:g}pt"
        if foreground:
            title.opts["color"] = foreground
        title.opts.update(label_args(elements.get("title", TextStyle())))
        title.setText(title_prefix + rich_text(state.title) if state.title else defaults["title_text"])
        # setTitle(None) hides the row. An authored title must make it visible
        # again; Reset to an empty automatic title, or a hidden one, must hide it
        # without a gap.
        shown = bool(title.text.strip())
        title.setVisible(shown)
        defaults["applied_title_text"] = title.text
        defaults["applied_title_color"] = title.opts.get("color")
        if shown:
            height = max(30.0, float(title.item.boundingRect().height()) + 6)
            title.setMaximumHeight(height)
            plot.layout.setRowFixedHeight(0, height)
        else:
            # Both, as pyqtgraph's own setTitle(None): the label's minimum height (its
            # text's, 23 px for a space) otherwise wins over the row's 0 and leaves a band.
            title.setMaximumHeight(0)
            plot.layout.setRowFixedHeight(0, 0)
        for item in data_items:
            _style_data_item(item, state, state.data_color)
        for item in fit_items:
            _style_data_item(item, state, state.fit_color)
        legend = getattr(plot, "legend", None)
        if legend is not None:
            if "legend" not in defaults:
                defaults["legend"] = {"visible": legend.isVisible(),
                                      "size": legend.opts.get("labelTextSize", "9pt"),
                                      "color": legend.opts.get("labelTextColor")}
            original = defaults["legend"]
            legend.setVisible(original["visible"] if state.legend_visible is None else
                              bool(state.legend_visible and legend.items))
            legend.setLabelTextSize(f"{state.legend_size:g}pt" if state.legend_size else original["size"])
            # Axis text already follows the canvas theme; legend text must do
            # the same or a white publication preset leaves pale labels on white.
            legend.setLabelTextColor(foreground or original["color"] or pg.getConfigOption("foreground"))
            apply_legend_family(legend, state.font_family)
            _style_legend_entries(legend, elements, state.font_family)
    finally:
        view.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
        pins = _apply_axis_pins(plot, state, auto)
        for axis, enabled in enumerate(auto):
            if enabled and axis not in pins:
                view.enableAutoRange(axis=axis, enable=enabled)


def apply_legend_family(legend, family: str | None) -> None:
    """Draw every legend row in the figure family (Arial when unset). The row's own text is
    not touched: pyqtgraph's ``LegendItem`` has no family option, so it goes on the label."""
    css = html_font_family(family)
    changed = False
    for _sample, label in list(legend.items):
        if label.opts.get("family") != css:
            label.opts["family"] = css
            label.setText(label.text)
            changed = True
    if changed:
        # The legend's own size is only recomputed when a row is added or removed, from the
        # rows' geometry, which the layout only refreshes when it is activated. A row whose
        # text just changed width would leave both stale, and a legend anchored by its far
        # edge (Distributions keeps a dragged legend that way) would move by the difference
        # at the next draw.
        legend.layout.activate()
        legend.updateSize()


def apply_tick_family(axis: pg.AxisItem, family: str | None) -> None:
    """Draw an axis' tick numbers in the figure family (Arial when unset), at the size they
    have. For the axes ``apply_plot_appearance`` does not reach, a colour bar's."""
    font = axis.style.get("tickFont")
    if font is None or font.family() != figure_family(family):
        axis.setStyle(tickFont=figure_qfont(family, font or QtWidgets.QApplication.font()))


def _draw_in_figure_family(plot: pg.PlotItem, family: str | None) -> None:
    """An untouched figure keeps its original look in everything but its text family: that
    is the figure default, Arial, and not the application font a journal does not have.

    Only what differs is written, so the redraw that follows every analysis costs a
    comparison per element, and nothing is recorded as the figure's original: the first
    real appearance step captures this, Arial, as what Reset goes back to."""
    css = html_font_family(family)
    title = plot.titleLabel
    if title.opts.get("family") != css:
        title.opts["family"] = css
        title.setText(title.text)
    for side in ("bottom", "left", "top", "right"):
        axis = plot.getAxis(side)
        if axis.labelStyle.get("font-family") != css:
            axis.labelStyle["font-family"] = css
            axis._updateLabel()
        apply_tick_family(axis, family)
    legend = getattr(plot, "legend", None)
    if legend is not None:
        apply_legend_family(legend, family)


def _style_legend_entries(legend, elements: Mapping[str, TextStyle],
                          family: str | None = None) -> None:
    """Each legend row's rename and style (``legend.entry.<name>``), escaped like any typed
    text. The window's own name is remembered per row, so Reset name puts it back. A row's
    own family wins; without one it is the figure's (``family``, Arial when unset)."""
    for _sample, label in list(legend.items):
        if getattr(label, "_nanoscope_applied_text", None) != label.text:
            label._nanoscope_auto_text = label.text      # the window (re)named it
        auto = getattr(label, "_nanoscope_auto_text", label.text)
        style = elements.get(f"legend.entry.{plain_text(auto)}"[:213], TextStyle())
        if style.is_default() and label.text == auto and not getattr(label, "_nanoscope_styled", False):
            label._nanoscope_applied_text = label.text
            continue
        text = auto if style.text is None else ("" if not style.text.strip() else rich_text(style.text))
        # LabelItem.setText MERGES into its options, so a style taken off must be taken out.
        for key in ("family", "bold", "italic"):
            label.opts.pop(key, None)
        label.opts["family"] = html_font_family(family)
        label.opts["justify"] = "left"
        label.opts["color"] = legend.opts.get("labelTextColor")
        label.opts["size"] = legend.opts.get("labelTextSize", "9pt")
        label.setText(text, **label_args(style))
        label._nanoscope_styled = not style.is_default()
        label._nanoscope_applied_text = label.text


def _apply_axis_pins(plot: pg.PlotItem, state: PlotAppearance,
                     auto: Sequence[object] = (True, True)) -> set[int]:
    """Pin the bottom and left ranges the user typed in Axis properties (display units).

    The pin is the page's framing from then on: ``plot_interaction``'s Reset view returns
    to it (``_nanoscope_axis_pins``), where it would otherwise auto-range. Removing a pin
    gives the axis back to auto-range. Returns the ViewBox axes now pinned.

    A pin moves the view only when it is new or changed, when the axis was still
    auto-ranging (a fresh draw), or after :func:`reframe_axis_pins` (a draw that framed
    its data itself): an unrelated appearance step (a title size, a fit-label drag, Undo)
    keeps the user's manual zoom, as every other appearance step does.
    """
    view = plot.getViewBox()
    previous = getattr(plot, "_nanoscope_axis_pins", {}) or {}
    reframe = bool(getattr(plot, "_nanoscope_axis_reframe", False))
    plot._nanoscope_axis_reframe = False
    pins: dict[int, tuple[float, float]] = {}
    for side, index in (("bottom", 0), ("left", 1)):
        rng = state.axis_pin(side).range()
        if rng is None:
            continue
        axis = plot.getAxis(side)
        try:
            lo, hi = (_display_to_view(axis, value) for value in rng)
        except (ValueError, ZeroDivisionError):
            continue                         # a pin a log axis cannot show (<= 0)
        if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
            pins[index] = (lo, hi)
    for index, (lo, hi) in pins.items():
        view.enableAutoRange(axis=index, enable=False)
        if reframe or previous.get(index) != (lo, hi) or auto[index]:
            (view.setXRange if index == 0 else view.setYRange)(lo, hi, padding=0)
    for index in previous:
        if index not in pins:
            view.enableAutoRange(axis=index, enable=True)
    plot._nanoscope_axis_pins = pins
    return set(pins)


def reframe_axis_pins(plot) -> None:
    """A data redraw re-frames: the next apply moves the view to the pin.

    For a window whose redraw puts new data on the plot and frames it itself (I-V's and
    Voltage dependence's ``autoRange()``) before re-applying its appearance. That
    ``autoRange()`` turns auto-range OFF, so to :func:`_apply_axis_pins` the pin looks
    unchanged and the data's framing looks like a manual zoom to keep: Refresh dropped a
    range typed in Axis properties while Axis properties still showed it. Call this
    between the draw and the apply (``try: draw … finally: reframe_axis_pins(plot);
    apply``). Appearance-only steps never call it, so they keep a manual zoom as before.
    A plot with no pin keeps its draw's framing; the next apply uses the mark up.
    """
    if isinstance(plot, pg.PlotWidget):
        plot = plot.getPlotItem()
    plot._nanoscope_axis_reframe = True
