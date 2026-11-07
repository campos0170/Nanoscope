"""Renderer-local text identities and geometry; no Qt or analytical operations.

A retained Figure owns the artists. A frozen canvas receives only these immutable
rectangles/strings, and the page receives edits through its existing history owner.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

from matplotlib.transforms import Bbox
from matplotlib.ticker import MultipleLocator

from ..core.figure_style import resolve_bound_text
from .style import font_family_list


def panel_view(panel: str) -> str:
    return "mpl_" + hashlib.sha256(str(panel).encode()).hexdigest()[:24].translate(
        str.maketrans("0123456789", "ghijklmnop"))


def storage_key(panel: str, key: str) -> str:
    """Fits the established PlotAppearance key grammar, independent of words/order."""
    if panel == "figure" and (key == "title" or key.startswith("legend.entry.")):
        return key
    return "annotation.mpl." + hashlib.sha256(str(panel).encode()).hexdigest()[:24] + "." + hashlib.sha256(key.encode()).hexdigest()[:20]


@dataclass(frozen=True)
class MatplotlibTarget:
    panel: str
    key: str
    automatic: str
    rect: tuple[float, float, float, float]
    noun: str
    capabilities: frozenset[str]
    panel_label: str
    visible: bool = True


@dataclass(frozen=True)
class MatplotlibAxis:
    panel: str
    side: str
    label: str
    shown_range: tuple[float, float]
    log: bool
    label_locked: str = ""
    major_available: bool = True
    basis: object = None


def bind_panel(axis, panel_id: str, *, label: str, labels_locked: str = "") -> None:
    axis._nanoscope_panel_id = str(panel_id)
    axis._nanoscope_panel_label = str(label)
    axis._nanoscope_labels_locked = labels_locked


def panel_identity(axis, index: int) -> str:
    if hasattr(axis, "_nanoscope_event_panel_identity"):
        # An event's annotation must never migrate to the next event at this ordinal.
        return "event:" + str(axis._nanoscope_event_panel_identity)
    return str(getattr(axis, "_nanoscope_panel_id", None) or axis.get_gid() or f"panel:{index}")


def _remember_auto(artist, text=None):
    if not hasattr(artist, "_nanoscope_automatic_text"):
        artist._nanoscope_automatic_text = artist.get_text() if text is None else str(text)
    return artist._nanoscope_automatic_text


def artist_targets(figure):
    """Yield stable role identities and their existing Text artists (ticks are a group)."""
    if figure._suptitle is not None:
        yield "figure", "title", [figure._suptitle], "title", frozenset({"edit", "style", "hide", "align"}), "Figure"
    for index, axis in enumerate(figure.axes):
        if not axis.get_visible():
            continue
        panel = panel_identity(axis, index)
        label = str(getattr(axis, "_nanoscope_panel_label", f"Panel {index + 1}"))
        yield panel, "title", [axis.title], "title", frozenset({"edit", "style", "hide", "align"}), label
        locked = bool(getattr(axis, "_nanoscope_labels_locked", ""))
        for side, mpl_axis in (("bottom", axis.xaxis), ("left", axis.yaxis)):
            if not axis.axison or not mpl_axis.get_visible():
                continue
            caps = {"style"} if locked else {"edit", "style", "hide"}
            yield panel, f"axis.{side}.label", [mpl_axis.label], "axis label", frozenset(caps), label
            ticks = list(mpl_axis.get_ticklabels()) + [mpl_axis.offsetText]
            yield panel, f"axis.{side}.ticks", ticks, "tick labels", frozenset({"style"}), label
        legend = axis.get_legend()
        if legend is not None:
            yield from _legend_targets(legend, panel, label)
        for ordinal, text in enumerate(axis.texts):
            key = getattr(text, "_nanoscope_target_key", None) or "annotation." + hashlib.sha256((text.get_gid() or f"text:{ordinal}").encode()).hexdigest()[:20]
            yield panel, key, [text], "annotation", frozenset({"edit", "style", "hide"}), label
    for legend in figure.legends:
        yield from _legend_targets(legend, "figure", "Figure")
    for ordinal, text in enumerate(figure.texts):
        if text is figure._suptitle:
            continue
        key = getattr(text, "_nanoscope_target_key", None) or ("annotation.provenance" if ordinal == 0 else f"annotation.caption{ordinal}")
        yield "figure", key, [text], "caption", frozenset({"edit", "style", "hide"}), "Figure"


def _legend_targets(legend, panel, label):
    keys = getattr(legend, "_nanoscope_source_keys", ())
    for index, text in enumerate(legend.get_texts()):
        # Comparison supplies stable source keys. Retained study artists retain their
        # identity for the lifetime of the editor; wording never becomes the key.
        source = keys[index] if index < len(keys) else (text.get_gid() or f"entry{index}")
        yield panel, f"legend.entry.{source}", [text], "legend entry", frozenset({"edit", "style", "hide"}), label


def remember_automatic_text(figure) -> None:
    for _panel, _key, artists, _noun, _caps, _label in artist_targets(figure):
        for artist in artists:
            _remember_auto(artist)


def apply_text_overrides(figure, appearance: Mapping, *, binding: Mapping | None = None) -> None:
    """Apply look and literal wording to existing artists; never move scientific points."""
    text_map = appearance.get("text") or {}
    binding = binding or appearance.get("style_binding")
    if figure._suptitle is None and appearance.get("title"):
        text = figure.suptitle("")
        _remember_auto(text, "")
    for panel, key, artists, _noun, caps, _label in artist_targets(figure):
        style = text_map.get(storage_key(panel, key), {})
        style = dict(style)
        if binding:
            effective, _origins = resolve_bound_text(binding, key, style)
            style = {**style, **effective}
        for artist in artists:
            _remember_auto(artist)
            if style.get("family") is not None:
                artist.set_fontfamily(font_family_list(style["family"]))
            if style.get("size_pt") is not None:
                artist.set_fontsize(style["size_pt"])
            if style.get("bold") is not None:
                artist.set_fontweight("bold" if style["bold"] else "normal")
            if style.get("italic") is not None:
                artist.set_fontstyle("italic" if style["italic"] else "normal")
            if style.get("color") is not None:
                artist.set_color(style["color"])
            if "align" in caps and style.get("align") is not None:
                artist.set_horizontalalignment(style["align"])
            if "edit" in caps and style.get("text") is not None:
                artist.set_text(str(style["text"]).strip())
                # User input is literal. A dollar in a sample name is not mathtext.
                artist.set_parse_math(False)
            elif key == "title" and panel == "figure" and appearance.get("title"):
                artist.set_text(appearance["title"].strip())
                artist.set_parse_math(False)
    for index, axis in enumerate(figure.axes):
        panel = panel_identity(axis, index)
        for side, mpl_axis, setter in (("bottom", axis.xaxis, axis.set_xlim), ("left", axis.yaxis, axis.set_ylim)):
            pin = (appearance.get("axes") or {}).get(panel_view(panel) + "." + side, {})
            lo, hi = pin.get("min"), pin.get("max")
            basis = getattr(axis, "_nanoscope_quantity_axes", {}).get(side)
            witness = (appearance.get("quantity_display") or {}).get("axes", {}).get(panel_view(panel) + "." + side)
            if basis is not None and witness is not None:
                from ..core.quantity_display import AxisDisplayBasis, display_to_view
                saved = AxisDisplayBasis.from_dict(witness)
                if saved.quantity.key != basis.quantity.key or saved.quantity.stored_unit != basis.quantity.stored_unit:
                    raise ValueError("The saved axis range describes another scientific quantity.")
                # Saved pins name their own display basis, including a legacy known basis.
                lo = None if lo is None else display_to_view(lo, saved)
                hi = None if hi is None else display_to_view(hi, saved)
            log = (axis.get_xscale() if side == "bottom" else axis.get_yscale()) == "log"
            if lo is not None and hi is not None:
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi)) or lo >= hi or (log and lo <= 0):
                    raise ValueError("A Matplotlib range must be finite, increasing and positive on a log axis.")
                setter(lo, hi)
            step = pin.get("tick_major")
            if step is not None:
                if not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
                    raise ValueError("Major tick spacing must be finite and positive.")
                if log:
                    raise ValueError("Changing logarithmic tick spacing is not supported by this renderer.")
                if basis is not None and witness is not None:
                    step /= saved.quantity.factor
                mpl_axis.set_major_locator(MultipleLocator(step))


def capture_targets(figure, renderer) -> tuple[tuple[MatplotlibTarget, ...], tuple[MatplotlibAxis, ...]]:
    """Freeze hit rectangles after the very same layout/draw used for the canvas."""
    height = float(figure.get_figheight() * figure.dpi)
    targets = []
    for panel, key, artists, noun, caps, label in artist_targets(figure):
        boxes = []
        for artist in artists:
            box = artist.get_window_extent(renderer)
            if all(math.isfinite(v) for v in box.bounds) and box.width > 0 and box.height > 0 and artist.get_visible():
                boxes.append(box)
        if boxes:
            box = Bbox.union(boxes)
            rect = (float(box.x0), height - float(box.y1), float(box.width), float(box.height))
        else:
            # Empty/hidden elements remain reachable in Format and inline Add title.
            axis = artists[0].axes if artists else None
            box = axis.bbox if axis is not None else figure.bbox
            rect = (float(box.x0), height - float(box.y1), max(24., float(box.width)), 22.)
        auto = "" if key.endswith(".ticks") else (_remember_auto(artists[0]) if artists else "")
        targets.append(MatplotlibTarget(panel, key, auto, rect, noun, caps, label,
                                       any(a.get_visible() and a.get_text().strip() for a in artists)))
    if not any(t.panel == "figure" and t.key == "title" for t in targets):
        targets.insert(0, MatplotlibTarget("figure", "title", "", (0., 0., float(figure.bbox.width), 24.), "title", frozenset({"edit", "style", "hide", "align"}), "Figure", False))
    axes = []
    for index, axis in enumerate(figure.axes):
        if not axis.get_visible() or not axis.axison: continue
        panel = panel_identity(axis, index)
        for side, mpl_axis, values, scale in (("bottom", axis.xaxis, axis.get_xlim(), axis.get_xscale()), ("left", axis.yaxis, axis.get_ylim(), axis.get_yscale())):
            if not mpl_axis.get_visible(): continue
            basis = getattr(axis, "_nanoscope_quantity_axes", {}).get(side)
            if basis is not None:
                from ..core.quantity_display import view_to_display
                values = tuple(view_to_display(float(value), basis) for value in values)
            axes.append(MatplotlibAxis(panel, side, _remember_auto(mpl_axis.label), tuple(map(float, values)), scale == "log", str(getattr(axis, "_nanoscope_labels_locked", "")), scale == "linear", basis))
    return tuple(targets), tuple(axes)
