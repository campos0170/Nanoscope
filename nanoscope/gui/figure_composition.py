"""Frozen multi-panel figures and deliberately restricted detached shared axes.

Composition never consults a recording. Sharing ranges requires a saved model of
supported Cartesian graphics; it creates new, hidden plots and never modifies
the original items, calculations, fit parameters or numerical table resources.
"""
from __future__ import annotations

import base64
from dataclasses import replace
import json
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .figure_bundle import (
    FigureSnapshot, MAX_SNAPSHOT_BYTES, capture_figure_snapshot, encode_figure_table,
)

CARTESIAN_RESOURCE = "rendering/cartesian.json"
PICTURE_RESOURCE = "rendering/figure.qpic"
MAX_PANELS = 12
MAX_LABEL_LENGTH = 32
DEFAULT_LABEL_STYLE = {"family": "Arial", "size_px": 18, "bold": True, "color": "#111111"}
_NS = "http://www.w3.org/2000/svg"
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_POINTS = 1_000_000
_FILES = {"png": "figure.png", "svg": "figure.svg", "data_csv": "data.csv",
          "data_parquet": "data.parquet", "config_json": "config.json",
          "provenance_json": "provenance.json"}
_AXIS_KINDS = ("standard", "readable-log", "readable-linear",
               "population-decade", "spectrum-readable")
_AXIS_METHODS = ("tickStrings", "tickValues", "logTickStrings", "logTickValues",
                 "generateDrawSpecs", "drawPicture", "choose_notation", "format_decade",
                 "format_minor", "without_collisions", "label_rank")


def _axis_class(kind):
    """Fixed local allowlist: saved resources never select an import or a callable."""
    if kind == "standard":
        return pg.AxisItem
    from .axes import ReadableLogAxis, ReadableLinearAxis
    if kind == "readable-log":
        return ReadableLogAxis
    if kind == "readable-linear":
        return ReadableLinearAxis
    if kind == "population-decade":
        from .plots_panel import _DecadeLabelAxis
        return _DecadeLabelAxis
    if kind == "spectrum-readable":
        from .psd_panel import _ReadableAxis
        return _ReadableAxis
    raise ValueError("Unsupported saved Cartesian axis formatter.")


def _axis_formatter(state):
    # Version-1 models predating formatter capture used exact pg.AxisItem only.
    descriptor = state.get("formatter", {"kind": "standard"})
    if not isinstance(descriptor, dict) or descriptor.get("kind") not in _AXIS_KINDS:
        raise ValueError("Unsupported saved Cartesian axis formatter.")
    kind = descriptor["kind"]
    if set(descriptor) != ({"kind"} if kind == "standard" else {"kind", "display_factor"}):
        raise ValueError("Invalid saved Cartesian axis formatter fields.")
    if kind != "standard":
        factor = descriptor["display_factor"]
        if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid saved Cartesian axis display factor.")
        if kind == "readable-linear" and state["log"]:
            raise ValueError("A linear-only Cartesian axis cannot be logarithmic.")
    return descriptor


def _capture_axis_formatter(axis):
    kind = next((kind for kind in _AXIS_KINDS if type(axis) is _axis_class(kind)), None)
    if kind is None or any(name in vars(axis) for name in _AXIS_METHODS):
        raise ValueError("Custom axis formatters do not support shared axes.")
    descriptor = {"kind": kind}
    if kind != "standard":
        if any(getattr(axis, name) != getattr(type(axis), name)
               for name in ("LABEL_GAP_PX", "MAX_PLAIN_DIGITS")):
            raise ValueError("Customized readable-axis formatting is unsupported.")
        descriptor["display_factor"] = axis.display_factor
    return _axis_formatter({"formatter": descriptor, "log": bool(axis.logMode)})


def _scatter_style(item):
    """Preserve uniform effective spot overrides, without rewriting live spot data."""
    style = None
    for row in item.data:
        if not row["visible"]:
            raise ValueError("Hidden individual scatter points require frozen composition.")
        current = {"symbol": _symbol(row["symbol"] if row["symbol"] is not None else item.opts["symbol"]),
                   "pen": _pen(row["pen"] if row["pen"] is not None else item.opts["pen"]),
                   "brush": _brush(row["brush"] if row["brush"] is not None else item.opts["brush"]),
                   "size": float(row["size"] if row["size"] != -1 else item.opts["size"])}
        if style is not None and current != style:
            raise ValueError("Per-point scatter styles require frozen composition without shared axes.")
        style = current
    return style if style is not None else {
        "symbol": _symbol(item.opts["symbol"]), "pen": _pen(item.opts["pen"]),
        "brush": _brush(item.opts["brush"]), "size": item.opts["size"]}


def _json(value) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _array(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim > 1 or array.dtype.kind not in "biuf" or array.size > _MAX_POINTS:
        raise ValueError("Only bounded one-dimensional real numerical arrays are supported.")
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "base64": base64.b64encode(array.tobytes(order="C")).decode("ascii")}


def _restore_array(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "base64"}:
        raise ValueError("Invalid Cartesian numerical array.")
    dtype = np.dtype(value["dtype"])
    shape = value["shape"]
    if (dtype.kind not in "biuf" or dtype.itemsize > 16 or not isinstance(shape, list)
            or len(shape) > 1 or any(type(n) is not int or not 0 <= n <= _MAX_POINTS for n in shape)):
        raise ValueError("Invalid Cartesian numerical array size or dtype.")
    payload = base64.b64decode(value["base64"], validate=True)
    if len(payload) != dtype.itemsize * math.prod(shape):
        raise ValueError("Cartesian numerical array byte count does not match.")
    return np.frombuffer(payload, dtype=dtype).copy().reshape(shape)


def _brush(value):
    brush = pg.mkBrush(value)
    if brush.style() not in (QtCore.Qt.BrushStyle.NoBrush, QtCore.Qt.BrushStyle.SolidPattern):
        raise ValueError("Gradient, texture and patterned brushes do not support shared axes.")
    return {"rgba": list(brush.color().getRgb()), "none": brush.style() == QtCore.Qt.BrushStyle.NoBrush}


def _restore_brush(value):
    brush = QtGui.QBrush(QtGui.QColor(*value["rgba"]))
    if value["none"]:
        brush.setStyle(QtCore.Qt.BrushStyle.NoBrush)
    return brush


def _pen(value):
    pen = pg.mkPen(value)
    return {"brush": _brush(pen.brush()), "width": pen.widthF(), "style": pen.style().value,
            "cap": pen.capStyle().value, "join": pen.joinStyle().value,
            "cosmetic": pen.isCosmetic(), "dash": pen.dashPattern(),
            "dash_offset": pen.dashOffset(), "miter": pen.miterLimit()}


def _restore_pen(value):
    pen = QtGui.QPen(_restore_brush(value["brush"]), float(value["width"]))
    pen.setStyle(QtCore.Qt.PenStyle(value["style"]))
    pen.setCapStyle(QtCore.Qt.PenCapStyle(value["cap"]))
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle(value["join"]))
    pen.setCosmetic(value["cosmetic"])
    if value["style"] == QtCore.Qt.PenStyle.CustomDashLine.value:
        pen.setDashPattern(value["dash"])
    pen.setDashOffset(value["dash_offset"])
    pen.setMiterLimit(value["miter"])
    return pen


def _font(value):
    return value.toString() if value is not None else None


def _restore_font(value):
    if value is None:
        return None
    font = QtGui.QFont()
    if not font.fromString(value):
        raise ValueError("Invalid saved Cartesian font.")
    return font


def _visible(item):
    return item.isVisibleTo(item.parentItem())


def _symbol(value):
    from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
    if value is not None and (not isinstance(value, str) or value not in Symbols):
        raise ValueError("Custom scatter paths do not support shared axes.")
    return value


def _capture_series(item):
    if (not item.transform().isIdentity() or item.pos() != QtCore.QPointF(0, 0)
            or item.rotation() != 0 or item.scale() != 1):
        raise ValueError("Transformed data graphics do not support shared axes.")
    common = {"visible": _visible(item), "opacity": item.opacity(), "z": item.zValue()}
    if type(item) is pg.PlotDataItem:
        opts = item.opts
        if any(opts.get(key) for key in ("fftMode", "derivativeMode", "phasemapMode", "subtractMeanMode")):
            raise ValueError("Plot-side analytical transforms cannot be repeated by composition.")
        if opts.get("shadowPen") is not None or opts.get("data") is not None:
            raise ValueError("Custom line shadows or point metadata are unsupported.")
        if opts.get("alphaMode") or opts.get("alphaHint", 1.0) != 1.0 or opts.get("pointMode"):
            raise ValueError("Adaptive alpha and point-only line modes are unsupported.")
        connect = opts.get("connect", "auto")
        if not isinstance(connect, str):
            raise ValueError("Custom connection masks are unsupported.")
        if connect not in {"auto", "all", "finite", "pairs"}:
            raise ValueError("Unsupported line connection style.")
        simple = {key: opts.get(key) for key in (
            "fillLevel", "fillOutline", "stepMode", "symbolSize", "pxMode", "antialias",
            "downsample", "autoDownsample", "downsampleMethod", "autoDownsampleFactor",
            "clipToView", "dynamicRangeLimit", "dynamicRangeHyst", "skipFiniteCheck")}
        return {**common, "kind": "line", "x": _array(item.xData), "y": _array(item.yData),
                "pen": _pen(opts["pen"]), "symbol": _symbol(opts.get("symbol")),
                "symbol_pen": _pen(opts.get("symbolPen")),
                "symbol_brush": _brush(opts.get("symbolBrush")),
                "fill_brush": _brush(opts.get("fillBrush")), "options": simple,
                "connect": connect, "name": item.name()}
    if type(item) is pg.ScatterPlotItem:
        if item.opts.get("compositionMode") is not None or np.any(item.data["hovered"]):
            raise ValueError("Special compositing or active hover styles are unsupported.")
        style = _scatter_style(item)
        return {**common, "kind": "scatter", "x": _array(item.data["x"]),
                "y": _array(item.data["y"]), **style, "pxMode": item.opts["pxMode"],
                "antialias": item.opts["antialias"], "name": item.opts.get("name")}
    if type(item) is pg.BarGraphItem:
        opts = item.opts
        if opts.get("pens") is not None or opts.get("brushes") is not None:
            raise ValueError("Per-bar styles require frozen composition without shared axes.")
        return {**common, "kind": "bar", "coordinates": {
            key: _array(opts.get(key)) for key in ("x", "y", "x0", "y0", "x1", "y1", "width", "height")},
            "pen": _pen(opts.get("pen")), "brush": _brush(opts.get("brush")),
            "name": opts.get("name")}
    raise ValueError("Only standard line, scatter and bar graphics support shared axes.")


def _capture_legend(plot):
    """Capture standard legends without guessing associations from colors or labels."""
    legend = plot.legend
    if legend is None or not _visible(legend):
        return None
    from pyqtgraph.graphicsItems.LegendItem import ItemSample
    if (type(legend) is not pg.LegendItem or legend.sampleType is not ItemSample
            or not legend.items or legend.size is not None or legend.parentItem() is not plot.vb
            or any(name in vars(legend) for name in ("paint", "boundingRect"))
            or not legend.transform().isIdentity() or legend.rotation() or legend.scale() != 1
            or legend.offset is None):
        raise ValueError("Only standard, automatically sized legends support detached composition.")
    if set(legend.childItems()) != {child for pair in legend.items for child in pair}:
        raise ValueError("Extra legend annotations require frozen composition.")
    entries = []
    for sample, label in legend.items:
        if (type(sample) is not ItemSample or type(label) is not pg.LabelItem
                or sample.item not in plot.items or label.angle != 0
                or sample.childItems() or label.childItems() != [label.item] or label.item.childItems()
                or not _visible(sample) or not _visible(label)
                or any(not item.transform().isIdentity() or item.rotation() or item.scale() != 1
                       or item.opacity() != 1 for item in (sample, label, label.item))
                or any(name in vars(sample) for name in ("paint", "boundingRect"))
                or not isinstance(label.text, str) or not label.text.strip()
                or len(label.text) > 500 or any(c in label.text for c in "<>\r\n")
                or label.opts.get("justify", "left") != "left"):
            raise ValueError("Customized or ambiguous legend entries require frozen composition.")
        opts = label.opts
        entries.append({"series_index": plot.items.index(sample.item), "text": label.text,
                        "font": _font(label.item.font()),
                        "color": list(pg.mkColor(opts.get("color") or pg.getConfigOption("foreground")).getRgb()),
                        "size": opts.get("size", "9pt"), "bold": bool(opts.get("bold", False)),
                        "italic": bool(opts.get("italic", False))})
    return {"entries": entries, "offset": list(legend.offset),
            "anchor": {"item": list(getattr(legend, "_GraphicsWidgetAnchor__itemAnchor")),
                       "parent": list(getattr(legend, "_GraphicsWidgetAnchor__parentAnchor")),
                       "offset": list(getattr(legend, "_GraphicsWidgetAnchor__offset"))},
            "columns": legend.columnCount, "frame": bool(legend.frame),
            "pen": _pen(legend.opts["pen"]), "brush": _brush(legend.opts["brush"]),
            "hspace": legend.layout.horizontalSpacing(), "vspace": legend.layout.verticalSpacing(),
            "margins": list(legend.layout.getContentsMargins()),
            "opacity": legend.opacity(), "z": legend.zValue()}


def _validate_legend(model):
    legend = model.get("legend")
    if legend is None:
        if model["version"] == 3:
            raise ValueError("Version-3 Cartesian models require a captured legend.")
        return
    if model["version"] != 3 or not isinstance(legend, dict):
        raise ValueError("Captured legends require Cartesian model version 3.")
    entries = legend.get("entries")
    if (not isinstance(entries, list) or not 1 <= len(entries) <= 1000
            or type(legend.get("columns")) is not int or not 1 <= legend["columns"] <= 1000
            or type(legend.get("frame")) is not bool):
        raise ValueError("Invalid saved legend layout.")
    for entry in entries:
        if (not isinstance(entry, dict) or type(entry.get("series_index")) is not int
                or not 0 <= entry["series_index"] < len(model["series"])
                or not isinstance(entry.get("text"), str) or not entry["text"].strip()
                or len(entry["text"]) > 500 or any(c in entry["text"] for c in "<>\r\n")
                or not isinstance(entry.get("size"), str)
                or not re.fullmatch(r"\d+(?:\.\d+)?(?:pt|px)", entry["size"])
                or not 1 <= float(entry["size"][:-2]) <= 200
                or not isinstance(entry.get("color"), list) or len(entry["color"]) != 4
                or any(type(n) is not int or not 0 <= n <= 255 for n in entry["color"])
                or any(type(entry.get(key)) is not bool for key in ("bold", "italic"))):
            raise ValueError("Invalid saved legend entry.")
        _restore_font(entry.get("font"))
    for key, length in (("offset", 2), ("margins", 4)):
        if (not isinstance(legend.get(key), list) or len(legend[key]) != length
                or any(type(n) not in (int, float) or not math.isfinite(n) or abs(n) > 32768
                       for n in legend[key])):
            raise ValueError("Invalid saved legend geometry.")
    anchor = legend.get("anchor")
    if not isinstance(anchor, dict) or set(anchor) != {"item", "parent", "offset"}:
        raise ValueError("Invalid saved legend anchor.")
    for key, values in anchor.items():
        if (not isinstance(values, list) or len(values) != 2
                or any(type(n) not in (int, float) or not math.isfinite(n)
                       or not (-32768 <= n <= 32768 if key == "offset" else 0 <= n <= 1) for n in values)):
            raise ValueError("Invalid saved legend anchor.")
    for key in ("hspace", "vspace", "opacity", "z"):
        if type(legend.get(key)) not in (int, float) or not math.isfinite(legend[key]):
            raise ValueError("Invalid saved legend style.")
    if not 0 <= legend["opacity"] <= 1 or any(abs(legend[k]) > 200 for k in ("hspace", "vspace")):
        raise ValueError("Invalid saved legend spacing or opacity.")


def _restore_legend(plot, state, series):
    legend = plot.addLegend(offset=tuple(state["offset"]), colCount=state["columns"],
                            frame=state["frame"], pen=_restore_pen(state["pen"]),
                            brush=_restore_brush(state["brush"]),
                            horSpacing=state["hspace"], verSpacing=state["vspace"])
    legend.layout.setContentsMargins(*state["margins"])
    for entry in state["entries"]:
        legend.addItem(series[entry["series_index"]], entry["text"])
        label = legend.items[-1][1]
        label.item.setFont(_restore_font(entry["font"]))
        label.setText(entry["text"], color=QtGui.QColor(*entry["color"]), size=entry["size"],
                      bold=entry["bold"], italic=entry["italic"])
    legend.setOpacity(state["opacity"])
    legend.setZValue(state["z"])
    legend.updateSize()
    legend.layout.activate()
    anchor = state["anchor"]
    legend.anchor(itemPos=tuple(anchor["item"]), parentPos=tuple(anchor["parent"]), offset=tuple(anchor["offset"]))
    return legend


def capture_cartesian_model(item) -> bytes | None:
    """Return a restricted re-renderable model, or None without modifying the item.

    Image/density/heatmap panels, extra annotations, custom graphics/transforms,
    custom ticks, varying per-point palettes and customized legends are deliberately refused.
    Exact built-in readable-axis classes retain their notation and display factor;
    a saved logarithmic millisecond axis never becomes a logarithmic seconds axis.
    Their exact frozen pictures can still be composed without shared axes.
    """
    if not isinstance(item, pg.PlotItem):
        return None
    try:
        legend = _capture_legend(item)
        if item.vb.state.get("aspectLocked") is not False:
            return None
        if getattr(item.vb, "_selection_overlay", None) is not None:
            return None
        if set(item.vb.addedItems) != set(item.items):
            return None
        if (item.titleLabel.opts.get("justify", "center") != "center"
                or any(not graphic.transform().isIdentity() or graphic.rotation() != 0
                       or graphic.scale() != 1 for graphic in (item.titleLabel, item.titleLabel.item))):
            return None
        series = [_capture_series(series) for series in item.items]
        axes = {}
        for side in ("left", "bottom", "right", "top"):
            axis = item.getAxis(side)
            formatter = _capture_axis_formatter(axis)
            if (getattr(axis, "_tickLevels", None) is not None
                    or getattr(axis, "_tickSpacing", None) is not None
                    or any(name in vars(axis) for name in ("tickStrings", "tickValues", "logTickStrings"))
                    or not axis.transform().isIdentity() or axis.rotation() != 0
                    or QtWidgets.QGraphicsItem.scale(axis) != 1
                    or not axis.label.transform().isIdentity()
                    or axis.label.rotation() != (-90 if side in ("left", "right") else 0)
                    or axis.label.scale() != 1):
                return None
            style = dict(axis.style)
            style["tickFont"] = _font(style.get("tickFont"))
            axes[side] = {"text": axis.labelText, "units": axis.labelUnits,
                          "formatter": formatter,
                          "log": bool(axis.logMode), "scale": axis.scale,
                          "auto_si": axis.autoSIPrefix, "visible": _visible(axis),
                          "label_style": dict(axis.labelStyle), "style": style,
                          "label_font": _font(axis.label.font()),
                          "pen": _pen(axis.pen()), "text_pen": _pen(axis.textPen()),
                          "tick_pen": _pen(axis.tickPen()), "grid": axis.grid}
        title = {"text": item.titleLabel.text, "options": dict(item.titleLabel.opts),
                 "font": _font(item.titleLabel.item.font())}
        scene = item.scene()
        if scene is None or not scene.views():
            return None
        rect = item.sceneBoundingRect()
        canvas = [rect.width(), rect.height()]
        ranges = item.viewRange()
        if not all(math.isfinite(v) for axis in ranges for v in axis):
            return None
        # Old readers ignore additional fields; version 2 prevents one from
        # replaying a readable ms-log formatter as a standard seconds axis.
        version = 2 if any(state["formatter"]["kind"] != "standard" for state in axes.values()) else 1
        if legend is not None:
            version = 3
        model = {"format": "nanoscope-cartesian-model", "version": version,
                 "canvas": canvas, "ranges": ranges, "axes": axes, "title": title,
                 "background": _brush(scene.views()[0].backgroundBrush()),
                 "inverted": [bool(item.vb.state["xInverted"]), bool(item.vb.state["yInverted"])],
                 "margins": list(item.layout.getContentsMargins()), "series": series}
        if legend is not None:
            model["legend"] = legend
            _validate_legend(model)
        payload = _json(model)
        return payload if len(payload) <= _MAX_MODEL_BYTES else None
    except (TypeError, ValueError, AttributeError, OverflowError):
        return None


def _model(snapshot):
    payload = dict(snapshot.additional_resources).get(CARTESIAN_RESOURCE)
    if payload is None:
        raise ValueError("A selected figure has no supported Cartesian model. "
                         "Use independent axes or save a new simple line/scatter/bar figure.")
    if not isinstance(payload, bytes) or len(payload) > _MAX_MODEL_BYTES:
        raise ValueError("Invalid or oversized Cartesian model.")
    model = json.loads(payload)
    if (not isinstance(model, dict) or model.get("format") != "nanoscope-cartesian-model"
            or type(model.get("version")) is not int or model["version"] not in (1, 2, 3)):
        raise ValueError("Unsupported Cartesian model format.")
    if not isinstance(model.get("series"), list) or len(model["series"]) > 1000:
        raise ValueError("Invalid Cartesian series count.")
    for size in model["canvas"]:
        if not math.isfinite(size) or not 0 < size <= 32768:
            raise ValueError("Invalid Cartesian canvas size.")
    if len(model["ranges"]) != 2 or any(
            len(pair) != 2 or not all(math.isfinite(v) for v in pair) or pair[0] >= pair[1]
            for pair in model["ranges"]):
        raise ValueError("Invalid Cartesian view ranges.")
    if not isinstance(model.get("axes"), dict) or set(model["axes"]) != {"left", "bottom", "right", "top"}:
        raise ValueError("Invalid Cartesian axes.")
    for state in model["axes"].values():
        if (not isinstance(state, dict) or not isinstance(state.get("text"), str)
                or not isinstance(state.get("units"), str) or type(state.get("log")) is not bool
                or type(state.get("auto_si")) is not bool
                or type(state.get("scale")) not in (int, float)
                or not math.isfinite(state["scale"]) or state["scale"] <= 0):
            raise ValueError("Invalid Cartesian axis identity.")
        if model["version"] >= 2 and "formatter" not in state:
            raise ValueError("Version-2 Cartesian axes require an explicit formatter descriptor.")
        formatter = _axis_formatter(state)
        if model["version"] == 1 and formatter["kind"] != "standard":
            raise ValueError("Readable Cartesian axis formatters require model version 2.")
    _validate_legend(model)
    return model


def shared_axes_compatibility(snapshots, share_x=True, share_y=True) -> tuple[bool, str]:
    """Check labels, units, scale, log mode and exact scientific tick formatting."""
    try:
        models = [_model(snapshot) for snapshot in snapshots]
        if not models:
            return False, "Select at least one saved figure."
        for side, enabled in (("bottom", share_x), ("left", share_y)):
            if not enabled:
                continue
            signatures = [{key: model["axes"][side][key]
                           for key in ("text", "units", "scale", "auto_si", "log")}
                          for model in models]
            for signature, model in zip(signatures, models):
                signature["formatter"] = _axis_formatter(model["axes"][side])
            if any(not str(signature["text"]).strip() for signature in signatures):
                return False, "Shared axes require explicit axis labels; unlabelled axes are ambiguous."
            if any(value != signatures[0] for value in signatures[1:]):
                return False, f"The {'X' if side == 'bottom' else 'Y'} labels, units, scale, log modes or tick formatters differ."
            axis = 0 if side == "bottom" else 1
            if any(model["inverted"][axis] != models[0]["inverted"][axis] for model in models):
                return False, f"The {'X' if axis == 0 else 'Y'} axis directions differ."
        return True, "Compatible saved Cartesian models; only detached copies will receive the shared ranges."
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        return False, str(exc)


def _clone_model(model, canvas, ranges):
    axes = {side: _axis_class(_axis_formatter(state)["kind"])(orientation=side)
            for side, state in model["axes"].items()}
    widget = pg.PlotWidget(background=_restore_brush(model["background"]), axisItems=axes)
    try:
        return _populate_clone(widget, model, canvas, ranges)
    except BaseException:
        widget.close()
        widget.deleteLater()
        raise


def _populate_clone(widget, model, canvas, ranges):
    widget.resize(round(canvas[0]), round(canvas[1]))
    plot = widget.getPlotItem()
    plot.setGeometry(QtCore.QRectF(0, 0, *canvas))
    plot.layout.setContentsMargins(*model["margins"])
    for side, state in model["axes"].items():
        axis = plot.getAxis(side)
        style = dict(state["style"])
        style["tickFont"] = _restore_font(style.get("tickFont"))
        offset = style.get("tickTextOffset")
        if isinstance(offset, (list, tuple)):
            style["tickTextOffset"] = offset[0 if side in ("left", "right") else 1]
        axis.setStyle(**style)
        axis.setPen(_restore_pen(state["pen"]))
        axis.setTextPen(_restore_pen(state["text_pen"]))
        axis.setTickPen(_restore_pen(state["tick_pen"]))
        axis.setLabel(state["text"], units=state["units"], **state["label_style"])
        axis.label.setFont(_restore_font(state["label_font"]))
        axis.setScale(state["scale"])
        formatter = _axis_formatter(state)
        if formatter["kind"] != "standard":
            axis.set_display_factor(formatter["display_factor"])
        axis.enableAutoSIPrefix(state["auto_si"])
        if not state["auto_si"]:
            # Pyqtgraph's setter can compute an SI factor even while disabling
            # its label prefix. Never replay that hidden tick multiplier.
            axis.autoSIPrefixScale = 1.0
            axis._updateLabel()
        axis.setGrid(state["grid"])
        axis.setVisible(state["visible"])
    title_options = {key: value for key, value in model["title"]["options"].items()
                     if key in {"color", "size", "bold", "italic"}}
    plot.setTitle(model["title"]["text"], **title_options)
    plot.titleLabel.item.setFont(_restore_font(model["title"]["font"]))
    plot.setLogMode(x=model["axes"]["bottom"]["log"], y=model["axes"]["left"]["log"])
    for side, state in model["axes"].items():
        plot.getAxis(side).setLogMode(state["log"])
    restored_series = []
    for state in model["series"]:
        kind = state["kind"]
        if kind == "line":
            data = pg.PlotDataItem(_restore_array(state["x"]), _restore_array(state["y"]),
                                  pen=_restore_pen(state["pen"]), connect=state["connect"],
                                  symbol=state["symbol"], symbolPen=_restore_pen(state["symbol_pen"]),
                                  symbolBrush=_restore_brush(state["symbol_brush"]),
                                  fillBrush=_restore_brush(state["fill_brush"]),
                                  name=state["name"], **state["options"])
        elif kind == "scatter":
            data = pg.ScatterPlotItem(x=_restore_array(state["x"]), y=_restore_array(state["y"]),
                                     pen=_restore_pen(state["pen"]), brush=_restore_brush(state["brush"]),
                                     symbol=state["symbol"], size=state["size"], pxMode=state["pxMode"],
                                     antialias=state["antialias"], name=state["name"])
        elif kind == "bar":
            coords = {key: _restore_array(value) for key, value in state["coordinates"].items()}
            data = pg.BarGraphItem(**coords, pen=_restore_pen(state["pen"]),
                                   brush=_restore_brush(state["brush"]), name=state["name"])
        else:
            widget.close()
            raise ValueError("Unsupported saved Cartesian series.")
        plot.addItem(data)
        data.setVisible(state["visible"])
        data.setOpacity(state["opacity"])
        data.setZValue(state["z"])
        restored_series.append(data)
    if model.get("legend") is not None:
        _restore_legend(plot, model["legend"], restored_series)
    plot.invertX(model["inverted"][0])
    plot.invertY(model["inverted"][1])
    plot.setRange(xRange=ranges[0], yRange=ranges[1], padding=0, disableAutoRange=True)
    # Detached widgets have no event-loop resize pass. Settle their graphics
    # layout explicitly before measuring axes (otherwise height can remain 20
    # while the label requires 39 pixels, overlapping the tick row).
    plot.layout.activate()
    plot.scene().prepareForPaint()
    return widget


def _legend_signature(model):
    legend = model.get("legend")
    if legend is None:
        raise ValueError("Every panel needs a newly saved, fully captured standard legend to share it.")
    entries = []
    for entry in legend["entries"]:
        series = dict(model["series"][entry["series_index"]])
        for key in ("x", "y", "coordinates", "name"):
            series.pop(key, None)
        entries.append({**{key: value for key, value in entry.items() if key != "series_index"},
                        "sample_style": series})
    return {**{key: value for key, value in legend.items() if key not in ("offset", "anchor", "z", "entries")},
            "entries": entries}


def shared_legend_compatibility(snapshots) -> tuple[bool, str]:
    """Share only identical label/style mappings, never merge or invent a key."""
    try:
        signatures = [_legend_signature(_model(snapshot)) for snapshot in snapshots]
        if not signatures:
            return False, "Select at least one saved figure."
        if any(signature != signatures[0] for signature in signatures[1:]):
            return False, "Legends have different labels, order, sample styles or typography. Keep individual legends."
        return True, "Identical captured legend entries will appear once below the panels."
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        return False, str(exc)


def _shared_pictures(snapshots, share_x, share_y, shared_legend=False):
    compatible, reason = shared_axes_compatibility(snapshots, share_x, share_y)
    if not compatible:
        raise ValueError(reason)
    models = [_model(snapshot) for snapshot in snapshots]
    if shared_legend:
        compatible, reason = shared_legend_compatibility(snapshots)
        if not compatible:
            raise ValueError(reason)
    canvas = (max(m["canvas"][0] for m in models), max(m["canvas"][1] for m in models))
    ranges = [[list(pair) for pair in model["ranges"]] for model in models]
    for axis, enabled in enumerate((share_x, share_y)):
        if enabled:
            union = [min(model["ranges"][axis][0] for model in models),
                     max(model["ranges"][axis][1] for model in models)]
            for value in ranges:
                value[axis] = union[:]
    widgets = []
    try:
        legend_snapshot = None
        if shared_legend:
            legend_widget = _clone_model(models[0], canvas, models[0]["ranges"])
            widgets.append(legend_widget)
            # Exporters render all scene items within the target bounds, not
            # only its descendants. Isolate the legend so curves behind it
            # cannot leak into the standalone key.
            legend = legend_widget.getPlotItem().legend
            legend.opts["offset"] = None
            legend.setParentItem(None)
            legend_widget.scene().removeItem(legend)
            legend_view = pg.GraphicsView()
            widgets.append(legend_view)
            legend_view.setBackgroundBrush(_restore_brush(models[0]["background"]))
            legend_view.scene().addItem(legend)
            legend.setPos(0, 0)
            legend.layout.activate()
            legend_snapshot = capture_figure_snapshot(
                legend, data=pd.DataFrame(),
                config={"kind": "shared-legend"}, provenance={"analysis_recomputed": False})
        panel_widgets = []
        for model, bounds in zip(models, ranges):
            detached = {**model, "legend": None} if shared_legend else model
            widget = _clone_model(detached, canvas, bounds)
            widgets.append(widget)
            panel_widgets.append(widget)
        # Identical plot rectangles make the common ranges visually align too.
        for side in ("left", "right", "bottom", "top"):
            axes = [widget.getPlotItem().getAxis(side) for widget in panel_widgets]
            extent = max(axis.width() if side in ("left", "right") else axis.height() for axis in axes)
            for axis in axes:
                (axis.setWidth if side in ("left", "right") else axis.setHeight)(extent)
        pictures = []
        for widget, snapshot, bounds in zip(panel_widgets, snapshots, ranges):
            # The common fixed axis extents invalidated each detached layout.
            widget.getPlotItem().layout.activate()
            widget.scene().prepareForPaint()
            pictures.append(capture_figure_snapshot(
                widget.getPlotItem(), data=snapshot.dataframe(), config=snapshot.config(),
                provenance={**snapshot.provenance(), "composition_view_ranges": bounds,
                            "scientific_recomputation": False}))
        return (pictures, ranges, legend_snapshot) if shared_legend else (pictures, ranges)
    finally:
        for widget in widgets:
            widget.close()
            widget.deleteLater()


def _load_picture(snapshot):
    payload = dict(snapshot.additional_resources).get(PICTURE_RESOURCE)
    if payload is None:
        raise ValueError("A selected figure lacks frozen vector commands. Capture it again before composing.")
    buffer = QtCore.QBuffer()
    buffer.setData(QtCore.QByteArray(payload))
    buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
    picture = QtGui.QPicture()
    if not picture.load(buffer):
        raise ValueError("A selected figure has invalid frozen vector commands.")
    return picture


def _svg_panel(payload, parent, panel_id, x, y, width, height):
    root = ET.fromstring(payload)
    viewbox = root.attrib.get("viewBox")
    if root.tag != f"{{{_NS}}}svg" or not viewbox:
        raise ValueError("A selected figure has an invalid frozen SVG.")
    ids = {node.attrib["id"]: panel_id + "_" + node.attrib["id"] for node in root.iter() if "id" in node.attrib}
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if key == "id":
                node.set(key, ids[value])
            else:
                value = re.sub(r"url\(#([^)]*)\)", lambda m: f"url(#{ids.get(m[1], m[1])})", value)
                if value.startswith("#") and value[1:] in ids:
                    value = "#" + ids[value[1:]]
                node.set(key, value)
    root.set("x", str(x))
    root.set("y", str(y))
    root.set("width", str(width))
    root.set("height", str(height))
    root.set("preserveAspectRatio", "xMidYMid meet")
    ET.SubElement(parent, f"{{{_NS}}}g", {"id": panel_id}).append(root)


def _png_dimensions(payload):
    """Inspect the actual PNG header before allocating a decompressed image."""
    buffer = QtCore.QBuffer()
    buffer.setData(QtCore.QByteArray(payload))
    buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
    reader = QtGui.QImageReader(buffer, b"PNG")
    size = reader.size()
    if not reader.canRead() or not size.isValid() or size.isEmpty():
        raise ValueError("A selected figure has an invalid frozen PNG.")
    if max(size.width(), size.height()) > 32768 or size.width() * size.height() > 100_000_000:
        raise ValueError("A frozen PNG exceeds 32,768 pixels per side or 100 megapixels.")
    return size


def validate_publication_layout(count, columns, panel_spans=None, spacing=20, label_style=None):
    """Bound layout metadata before allocation. Spans use row-major first-fit packing."""
    if type(columns) is not int or not 1 <= columns <= 4:
        raise ValueError("Choose between 1 and 4 columns.")
    if type(spacing) not in (int, float) or not math.isfinite(spacing) or not 0 <= spacing <= 200:
        raise ValueError("Panel spacing must be between 0 and 200 native pixels.")
    spans = [[1, 1] for _ in range(count)] if panel_spans is None else panel_spans
    if (not isinstance(spans, (tuple, list)) or len(spans) != count
            or any(not isinstance(span, (tuple, list)) or len(span) != 2
                   or type(span[0]) is not int or not 1 <= span[0] <= 4
                   or type(span[1]) is not int or not 1 <= span[1] <= columns for span in spans)):
        raise ValueError("Give every panel a row span from 1–4 and a column span within the grid width.")
    style = dict(DEFAULT_LABEL_STYLE) if label_style is None else label_style
    if (not isinstance(style, dict) or set(style) != set(DEFAULT_LABEL_STYLE)
            or not isinstance(style["family"], str) or not style["family"].strip()
            or len(style["family"]) > 100 or any(c in style["family"] for c in "\r\n")
            or type(style["size_px"]) is not int or not 6 <= style["size_px"] <= 72
            or type(style["bold"]) is not bool
            or not isinstance(style["color"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", style["color"])):
        raise ValueError("Invalid panel-label typography. Choose a font, 6–72 native pixels and an RGB color.")
    occupied, positions = set(), []
    for row_span, column_span in spans:
        for row in range(4 * count):
            for col in range(columns - column_span + 1):
                cells = {(r, c) for r in range(row, row + row_span)
                         for c in range(col, col + column_span)}
                if not cells & occupied:
                    occupied.update(cells)
                    positions.append((row, col, row_span, column_span))
                    break
            else:
                continue
            break
    return [list(span) for span in spans], float(spacing), dict(style), positions


def compose_figures(snapshots, labels, columns, share_x=False, share_y=False, *,
                    panel_spans=None, spacing=20, label_style=None, shared_legend=False) -> FigureSnapshot:
    """Compose 1–12 frozen figures; original resources retain all numerical data.

    The top-level table is an explicit panel index. Every panel's exact CSV,
    Parquet, configuration, provenance, images and original additional resources
    live below ``panels/panel_NNN/``. No concatenation coerces numerical dtypes.
    """
    snapshots = list(snapshots)
    if not 1 <= len(snapshots) <= MAX_PANELS:
        raise ValueError("Select between 1 and 12 saved figures.")
    if any(not isinstance(snapshot, FigureSnapshot) for snapshot in snapshots):
        raise TypeError("Composition requires saved FigureSnapshot objects.")
    if type(columns) is not int or not 1 <= columns <= 4:
        raise ValueError("Choose between 1 and 4 columns.")
    advanced_layout = panel_spans is not None or spacing != 20 or label_style is not None or shared_legend
    if panel_spans is None:
        columns = min(columns, len(snapshots))
    spans, gap, label_style, positions = validate_publication_layout(
        len(snapshots), columns, panel_spans, spacing, label_style)
    if type(shared_legend) is not bool:
        raise ValueError("Shared legend must be enabled or disabled explicitly.")
    labels = [chr(65 + i) for i in range(len(snapshots))] if labels is None else list(labels)
    if len(labels) != len(snapshots) or any(
            not isinstance(label, str) or len(label) > MAX_LABEL_LENGTH or "\n" in label or "\r" in label
            for label in labels):
        raise ValueError("Provide one single-line panel label of at most 32 characters per figure.")
    total_bytes = 0
    for snapshot in snapshots:
        metadata, blobs = snapshot.to_storage()
        FigureSnapshot.from_storage(metadata, blobs)
        total_bytes += sum(map(len, blobs.values()))
    if total_bytes > MAX_SNAPSHOT_BYTES:
        raise ValueError("The original panels exceed the 512 MiB composite storage budget.")
    for snapshot in snapshots:
        _png_dimensions(snapshot.png_bytes)
    # Refuse oversized compositions before creating detached widgets or images.
    planned_w = max(snapshot.width for snapshot in snapshots)
    planned_h = max(snapshot.height for snapshot in snapshots)
    if share_x or share_y or shared_legend:
        models = [_model(snapshot) for snapshot in snapshots]
        planned_w = max(model["canvas"][0] for model in models)
        planned_h = max(model["canvas"][1] for model in models)
    font = QtGui.QFont(label_style["family"])
    font.setBold(label_style["bold"])
    font.setPixelSize(label_style["size_px"])
    metrics = QtGui.QFontMetricsF(font)
    header = max(32, math.ceil(metrics.height()) + 12) if any(labels) else 0
    rows = max(row + row_span for row, _, row_span, _ in positions)
    planned_width = math.ceil(columns * planned_w + (columns - 1) * gap)
    planned_height = math.ceil(rows * (planned_h + header) + (rows - 1) * gap)
    if max(planned_width, planned_height) > 32768 or planned_width * planned_height > 100_000_000:
        raise ValueError("The composite exceeds 32,768 pixels per side or 100 megapixels. "
                         "Use fewer or smaller panels.")
    ranges = None
    rendered = snapshots
    legend_snapshot = None
    if shared_legend:
        rendered, ranges, legend_snapshot = _shared_pictures(
            snapshots, bool(share_x), bool(share_y), shared_legend=True)
    elif share_x or share_y:
        rendered, ranges = _shared_pictures(snapshots, bool(share_x), bool(share_y))
    cell_w = max(snapshot.width for snapshot in rendered)
    cell_h = max(snapshot.height for snapshot in rendered)
    width = math.ceil(columns * cell_w + (columns - 1) * gap)
    height = math.ceil(rows * (cell_h + header) + (rows - 1) * gap)
    legend_top = height + gap
    if legend_snapshot is not None:
        if legend_snapshot.width > width:
            raise ValueError("The shared legend is wider than the composition. Widen the grid or shorten source legend labels.")
        height = math.ceil(legend_top + legend_snapshot.height)
    if (not all(math.isfinite(v) and 0 < v <= 32768 for v in (width, height))
            or width * height > 100_000_000):
        raise ValueError("The composite exceeds 32,768 pixels per side or 100 megapixels. "
                         "Use fewer or smaller panels.")
    if any(metrics.horizontalAdvance(label) > span[1] * cell_w + (span[1] - 1) * gap - 12
           for label, span in zip(labels, spans)):
        raise ValueError("A panel label is too wide for its canvas. Shorten that label.")
    pictures = []
    for snapshot in rendered:
        _png_dimensions(snapshot.png_bytes)
        image = QtGui.QImage.fromData(snapshot.png_bytes)
        if image.isNull():
            raise ValueError("A selected figure has an invalid frozen PNG.")
        pictures.append(_load_picture(snapshot))
    composite = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
    composite.fill(QtCore.Qt.GlobalColor.white)
    picture = QtGui.QPicture()
    raster, vector = QtGui.QPainter(composite), QtGui.QPainter(picture)
    svg = ET.Element(f"{{{_NS}}}svg", {"width": str(width), "height": str(height),
                                      "viewBox": f"0 0 {width} {height}"})
    ET.SubElement(svg, f"{{{_NS}}}rect", {"width": str(width), "height": str(height), "fill": "white"})
    try:
        vector.fillRect(QtCore.QRectF(0, 0, width, height), QtCore.Qt.GlobalColor.white)
        for index, (snapshot, panel_picture, label) in enumerate(zip(rendered, pictures, labels)):
            row, col, row_span, col_span = positions[index]
            x, top = col * (cell_w + gap), row * (cell_h + header + gap)
            area_w = col_span * cell_w + (col_span - 1) * gap
            area_h = row_span * (cell_h + header) + (row_span - 1) * gap - header
            scale = min(area_w / snapshot.width, area_h / snapshot.height)
            panel_w, panel_h = snapshot.width * scale, snapshot.height * scale
            left, y = x + (area_w - panel_w) / 2, top + header + (area_h - panel_h) / 2
            label_y = top + (header - metrics.height()) / 2 + metrics.ascent()
            for painter in (raster, vector):
                painter.setFont(font)
                painter.setPen(QtGui.QColor(label_style["color"]))
                painter.drawText(QtCore.QPointF(x + 4, label_y), label)
            # Replay vector commands for the preview too. Enlarging a spanned
            # panel must not simply magnify its low-resolution PNG thumbnail.
            for painter in (raster, vector):
                painter.save()
                try:
                    painter.setClipRect(QtCore.QRectF(left, y, panel_w, panel_h))
                    painter.translate(left, y)
                    painter.scale(scale, scale)
                    if not panel_picture.play(painter):
                        raise ValueError("A frozen panel could not be replayed.")
                finally:
                    painter.restore()
            panel_id = f"panel_{index + 1:03d}"
            ET.SubElement(svg, f"{{{_NS}}}text", {"x": str(x + 4), "y": str(label_y),
                          "font-family": label_style["family"], "font-size": str(label_style["size_px"]),
                          "font-weight": "bold" if label_style["bold"] else "normal",
                          "fill": label_style["color"], "id": panel_id + "_label"}).text = label
            _svg_panel(snapshot.svg_bytes, svg, panel_id, left, y, panel_w, panel_h)
        if legend_snapshot is not None:
            left = (width - legend_snapshot.width) / 2
            legend_rect = QtCore.QRectF(left, legend_top, legend_snapshot.width, legend_snapshot.height)
            raster.drawImage(legend_rect, QtGui.QImage.fromData(legend_snapshot.png_bytes))
            vector.save()
            try:
                vector.setClipRect(legend_rect)
                vector.translate(left, legend_top)
                if not _load_picture(legend_snapshot).play(vector):
                    raise ValueError("The captured shared legend could not be replayed.")
            finally:
                vector.restore()
            _svg_panel(legend_snapshot.svg_bytes, svg, "shared_legend", left, legend_top,
                       legend_snapshot.width, legend_snapshot.height)
    finally:
        raster.end()
        vector.end()
    picture.setBoundingRect(QtCore.QRect(0, 0, width, height))
    image_buffer, picture_buffer = QtCore.QBuffer(), QtCore.QBuffer()
    for buffer in (image_buffer, picture_buffer):
        buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    if not composite.save(image_buffer, "PNG") or not picture.save(picture_buffer):
        raise ValueError("The composite rendering could not be encoded.")
    resources = {PICTURE_RESOURCE: bytes(picture_buffer.data())}
    panels, records = [], []
    for index, (snapshot, label) in enumerate(zip(snapshots, labels)):
        panel_id = f"panel_{index + 1:03d}"
        prefix = "panels/" + panel_id + "/"
        metadata, blobs = snapshot.to_storage()
        files = {**_FILES, **{key: filename for filename, key in metadata["resources"].items()}}
        mapping = {key: prefix + filename for key, filename in files.items()}
        resources.update({mapping[key]: payload for key, payload in blobs.items()})
        panels.append({"panel_id": panel_id, "label": label, "snapshot": metadata,
                       "blob_resources": mapping, "source_config": snapshot.config(),
                       "source_provenance": snapshot.provenance()})
        records.append({"panel_id": panel_id, "label": label,
                        "data_csv_resource": prefix + "data.csv",
                        "data_parquet_resource": prefix + "data.parquet"})
    catalog = pd.DataFrame(records)
    config = {"kind": "figure-composition", "version": 2 if advanced_layout else 1, "columns": columns,
              "panel_labels": labels, "share_x": bool(share_x), "share_y": bool(share_y),
              "view_ranges": ranges, "panels": panels}
    if advanced_layout:
        config.update(panel_spans=spans, spacing=gap, label_style=label_style, shared_legend=shared_legend)
    provenance = {"kind": "frozen multi-panel composition", "analysis_recomputed": False,
                  "source_figures_modified": False,
                  "shared_axis_contract": "range unions applied only to detached Cartesian graphics",
                  "shared_legend_contract": "identical captured label/style mappings shown once below panels"
                      if shared_legend else "original panel legends retained",
                  "table_contract": "Top-level data is a panel index; every exact original numerical table "
                                    "and snapshot resource is retained under panels/panel_NNN/.",
                  "panel_count": len(snapshots), "panels": [p["source_provenance"] for p in panels]}
    result = FigureSnapshot(
        png_bytes=bytes(image_buffer.data()),
        svg_bytes=ET.tostring(svg, encoding="utf-8", xml_declaration=True),
        data_csv_bytes=catalog.to_csv(index=False).encode("utf-8"),
        data_parquet_bytes=encode_figure_table(catalog), config_json_bytes=_json(config),
        provenance_json_bytes=_json(provenance), width=width, height=height,
        additional_resources=tuple(resources.items()))
    metadata, blobs = result.to_storage()
    return FigureSnapshot.from_storage(metadata, blobs)
