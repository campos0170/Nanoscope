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


def capture_cartesian_model(item) -> bytes | None:
    """Return a restricted re-renderable model, or None without modifying the item.

    Image/density/heatmap panels, extra annotations, custom graphics/transforms,
    custom ticks, varying per-point palettes and visible legends are deliberately refused.
    Exact built-in readable-axis classes retain their notation and display factor;
    a saved logarithmic millisecond axis never becomes a logarithmic seconds axis.
    Their exact frozen pictures can still be composed without shared axes.
    """
    if not isinstance(item, pg.PlotItem):
        return None
    try:
        if item.legend is not None and _visible(item.legend):
            return None
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
        model = {"format": "nanoscope-cartesian-model", "version": version,
                 "canvas": canvas, "ranges": ranges, "axes": axes, "title": title,
                 "background": _brush(scene.views()[0].backgroundBrush()),
                 "inverted": [bool(item.vb.state["xInverted"]), bool(item.vb.state["yInverted"])],
                 "margins": list(item.layout.getContentsMargins()), "series": series}
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
            or type(model.get("version")) is not int or model["version"] not in (1, 2)):
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
        if model["version"] == 2 and "formatter" not in state:
            raise ValueError("Version-2 Cartesian axes require an explicit formatter descriptor.")
        formatter = _axis_formatter(state)
        if model["version"] == 1 and formatter["kind"] != "standard":
            raise ValueError("Readable Cartesian axis formatters require model version 2.")
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
        axis.setGrid(state["grid"])
        axis.setVisible(state["visible"])
    title_options = {key: value for key, value in model["title"]["options"].items()
                     if key in {"color", "size", "bold", "italic"}}
    plot.setTitle(model["title"]["text"], **title_options)
    plot.titleLabel.item.setFont(_restore_font(model["title"]["font"]))
    plot.setLogMode(x=model["axes"]["bottom"]["log"], y=model["axes"]["left"]["log"])
    for side, state in model["axes"].items():
        plot.getAxis(side).setLogMode(state["log"])
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
    plot.invertX(model["inverted"][0])
    plot.invertY(model["inverted"][1])
    plot.setRange(xRange=ranges[0], yRange=ranges[1], padding=0, disableAutoRange=True)
    # Detached widgets have no event-loop resize pass. Settle their graphics
    # layout explicitly before measuring axes (otherwise height can remain 20
    # while the label requires 39 pixels, overlapping the tick row).
    plot.layout.activate()
    plot.scene().prepareForPaint()
    return widget


def _shared_pictures(snapshots, share_x, share_y):
    compatible, reason = shared_axes_compatibility(snapshots, share_x, share_y)
    if not compatible:
        raise ValueError(reason)
    models = [_model(snapshot) for snapshot in snapshots]
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
        for model, bounds in zip(models, ranges):
            widgets.append(_clone_model(model, canvas, bounds))
        # Identical plot rectangles make the common ranges visually align too.
        for side in ("left", "right", "bottom", "top"):
            axes = [widget.getPlotItem().getAxis(side) for widget in widgets]
            extent = max(axis.width() if side in ("left", "right") else axis.height() for axis in axes)
            for axis in axes:
                (axis.setWidth if side in ("left", "right") else axis.setHeight)(extent)
        pictures = []
        for widget, snapshot, bounds in zip(widgets, snapshots, ranges):
            # The common fixed axis extents invalidated each detached layout.
            widget.getPlotItem().layout.activate()
            widget.scene().prepareForPaint()
            pictures.append(capture_figure_snapshot(
                widget.getPlotItem(), data=snapshot.dataframe(), config=snapshot.config(),
                provenance={**snapshot.provenance(), "composition_view_ranges": bounds,
                            "scientific_recomputation": False}))
        return pictures, ranges
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


def compose_figures(snapshots, labels, columns, share_x=False, share_y=False) -> FigureSnapshot:
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
    columns = min(columns, len(snapshots))
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
    if share_x or share_y:
        models = [_model(snapshot) for snapshot in snapshots]
        planned_w = max(model["canvas"][0] for model in models)
        planned_h = max(model["canvas"][1] for model in models)
    planned_rows = math.ceil(len(snapshots) / columns)
    planned_width = math.ceil(columns * planned_w + (columns - 1) * 20)
    planned_height = math.ceil(planned_rows * (planned_h + 32) + (planned_rows - 1) * 20)
    if max(planned_width, planned_height) > 32768 or planned_width * planned_height > 100_000_000:
        raise ValueError("The composite exceeds 32,768 pixels per side or 100 megapixels. "
                         "Use fewer or smaller panels.")
    ranges = None
    rendered = snapshots
    if share_x or share_y:
        rendered, ranges = _shared_pictures(snapshots, bool(share_x), bool(share_y))
    cell_w = max(snapshot.width for snapshot in rendered)
    cell_h = max(snapshot.height for snapshot in rendered)
    gap, header = 20, 32
    rows = math.ceil(len(snapshots) / columns)
    width = math.ceil(columns * cell_w + (columns - 1) * gap)
    height = math.ceil(rows * (cell_h + header) + (rows - 1) * gap)
    if (not all(math.isfinite(v) and 0 < v <= 32768 for v in (width, height))
            or width * height > 100_000_000):
        raise ValueError("The composite exceeds 32,768 pixels per side or 100 megapixels. "
                         "Use fewer or smaller panels.")
    font = QtGui.QFont("Arial", 12)
    font.setBold(True)
    font.setPixelSize(18)
    if any(QtGui.QFontMetricsF(font).horizontalAdvance(label) > cell_w - 12 for label in labels):
        raise ValueError("A panel label is too wide for its canvas. Shorten that label.")
    images = []
    pictures = []
    for snapshot in rendered:
        _png_dimensions(snapshot.png_bytes)
        image = QtGui.QImage.fromData(snapshot.png_bytes)
        if image.isNull():
            raise ValueError("A selected figure has an invalid frozen PNG.")
        images.append(image)
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
        for index, (snapshot, image, panel_picture, label) in enumerate(zip(rendered, images, pictures, labels)):
            x, top = (index % columns) * (cell_w + gap), (index // columns) * (cell_h + header + gap)
            scale = min(cell_w / snapshot.width, cell_h / snapshot.height)
            panel_w, panel_h = snapshot.width * scale, snapshot.height * scale
            left, y = x + (cell_w - panel_w) / 2, top + header + (cell_h - panel_h) / 2
            for painter in (raster, vector):
                painter.setFont(font)
                painter.setPen(QtGui.QColor("#111111"))
                painter.drawText(QtCore.QPointF(x + 4, top + 23), label)
            raster.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            raster.drawImage(QtCore.QRectF(left, y, panel_w, panel_h), image)
            vector.save()
            try:
                vector.setClipRect(QtCore.QRectF(left, y, panel_w, panel_h))
                vector.translate(left, y)
                vector.scale(scale, scale)
                if not panel_picture.play(vector):
                    raise ValueError("A frozen panel could not be replayed.")
            finally:
                vector.restore()
            panel_id = f"panel_{index + 1:03d}"
            ET.SubElement(svg, f"{{{_NS}}}text", {"x": str(x + 4), "y": str(top + 23),
                          "font-family": "Arial", "font-size": "18", "font-weight": "bold",
                          "fill": "#111111", "id": panel_id + "_label"}).text = label
            _svg_panel(snapshot.svg_bytes, svg, panel_id, left, y, panel_w, panel_h)
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
    config = {"kind": "figure-composition", "version": 1, "columns": columns,
              "panel_labels": labels, "share_x": bool(share_x), "share_y": bool(share_y),
              "view_ranges": ranges, "panels": panels}
    provenance = {"kind": "frozen multi-panel composition", "analysis_recomputed": False,
                  "source_figures_modified": False,
                  "shared_axis_contract": "range unions applied only to detached Cartesian graphics",
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
