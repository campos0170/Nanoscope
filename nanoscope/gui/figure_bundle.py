"""Frozen publication pictures and portable, lossless figure bundles.

Capture reads the existing graphics items and caller-supplied table only. It never
calls an analysis, fit, plot redraw, or a view-range setter. Separate residual
widgets are captured independently and composed without reparenting live items.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from pyqtgraph.exporters import ImageExporter, SVGExporter

from ..core.atomic_write import staged_path, staged_write, take_identity_notes
from ..core.figure_tables import (
    _object_scalar, _restore_object_scalar, _needs_object_codec,
    _table_parquet_bytes, _table_from_parquet, encode_figure_table, decode_figure_table,
)
from ..plot import pg_svg

_BLOB_FIELDS = {
    "png": "png_bytes", "svg": "svg_bytes", "data_csv": "data_csv_bytes",
    "data_parquet": "data_parquet_bytes", "config_json": "config_json_bytes",
    "provenance_json": "provenance_json_bytes",
}
_FILES = {"png": "figure.png", "svg": "figure.svg", "data_csv": "data.csv",
          "data_parquet": "data.parquet", "config_json": "config.json",
          "provenance_json": "provenance.json"}
_SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG_NS)
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
_PICTURE_RESOURCE = "rendering/figure.qpic"


def _json_bytes(value: Any) -> bytes:
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Figure metadata must be JSON-shaped, not {type(item).__name__}.")
    return json.dumps(value, default=encode, allow_nan=False, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _resource_name(name: str) -> str:
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", name)
            or name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/"))
            or name.split("/")[0].casefold() in {*_FILES.values(), "manifest.json"}):
        raise ValueError("Additional figure resources require safe, unique relative filenames.")
    return name


def _size(item) -> tuple[float, float]:
    rect = item.sceneRect() if isinstance(item, QtWidgets.QGraphicsScene) else item.sceneBoundingRect()
    width, height = float(rect.width()), float(rect.height())
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("The plot must have a finite, positive export area.")
    return width, height


def _encode_png(image: QtGui.QImage) -> bytes:
    if image.isNull():
        raise ValueError("The current plot did not render an image.")
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise ValueError("The figure could not be encoded as PNG.")
    return bytes(buffer.data())


def _capture_image(item, *, max_edge: int | None = None) -> QtGui.QImage:
    width, height = _size(item)
    if max_edge is None and (max(width, height) > 32768 or width * height > 100_000_000):
        raise ValueError("The native figure is too large to capture safely; reduce its canvas size.")
    exporter = ImageExporter(item)
    if max_edge is not None:
        scale = min(1.0, max_edge / max(width, height))
        exporter.parameters().param("width").setValue(
            max(1, round(width * scale)), blockSignal=exporter.widthChanged)
        exporter.parameters().param("height").setValue(
            max(1, round(height * scale)), blockSignal=exporter.heightChanged)
    result = exporter.export(toBytes=True)
    if not isinstance(result, QtGui.QImage) or result.isNull():
        raise ValueError("The current plot did not render an image.")
    return result.copy()


def _capture_svg(item) -> bytes:
    # The same pending-paint reconciliation used by all existing SVG exports.
    if isinstance(item, QtWidgets.QGraphicsScene):
        prepare = getattr(item, "prepareForPaint", None)
        if prepare is not None:
            prepare()
    else:
        pg_svg.refresh_for_export(item)
    return bytes(SVGExporter(item).export(toBytes=True))


def _composition_size(sizes, gap=12.0):
    width = max(w for w, _ in sizes)
    heights = [h * width / w for w, h in sizes]
    return width, sum(heights) + gap * (len(sizes) - 1), heights


def figure_native_size(item, *, residual_item=None, include_residual=False):
    """Native publication canvas size, without rendering or changing any item."""
    if not include_residual:
        return _size(item)
    if residual_item is None:
        raise ValueError("No residual panel is available for this figure.")
    width, height, _ = _composition_size([_size(item), _size(residual_item)])
    return width, height


def _compose_images(images: list[QtGui.QImage]) -> QtGui.QImage:
    width, height, heights = _composition_size([(i.width(), i.height()) for i in images])
    if width * height > 100_000_000 or max(width, height) > 32768:
        raise ValueError("The combined figure is too large to render safely.")
    combined = QtGui.QImage(math.ceil(width), math.ceil(height), QtGui.QImage.Format.Format_ARGB32)
    combined.fill(images[0].pixelColor(0, 0))
    painter = QtGui.QPainter(combined)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        top = 0.0
        for image, panel_height in zip(images, heights):
            painter.drawImage(QtCore.QRectF(0, top, width, panel_height), image)
            top += panel_height + 12.0
    finally:
        painter.end()
    return combined


def _compose_svg(panels: list[bytes], sizes) -> tuple[bytes, float, float]:
    width, height, heights = _composition_size(sizes)
    root = ET.Element(f"{{{_SVG_NS}}}svg", {
        "width": str(width), "height": str(height), "viewBox": f"0 0 {width} {height}"})
    top = 0.0
    for index, (payload, size, panel_height) in enumerate(zip(panels, sizes, heights)):
        panel = ET.fromstring(payload)
        # IDs are document-wide, even in nested SVGs. Prefix both declarations and
        # references so two independent exporters' clip paths never collide.
        ids = {node.attrib["id"]: f"panel{index}_{node.attrib['id']}"
               for node in panel.iter() if "id" in node.attrib}
        for node in panel.iter():
            for key, value in list(node.attrib.items()):
                if key == "id":
                    node.set(key, ids[value])
                    continue
                value = re.sub(r"url\(#([^)]*)\)",
                               lambda m: f"url(#{ids.get(m[1], m[1])})", value)
                if value.startswith("#") and value[1:] in ids:
                    value = "#" + ids[value[1:]]
                node.set(key, value)
        # Qt's SVG Tiny renderer silently skips nested <svg> documents. Use an
        # ordinary group with the same viewport transform, so the saved vector
        # also renders both panels in publication PNG/PDF exports.
        box = panel.attrib.get("viewBox", f"0 0 {size[0]} {size[1]}")
        try:
            vx, vy, vw, vh = map(float, re.split(r"[\s,]+", box.strip()))
        except (ValueError, TypeError) as exc:
            raise ValueError("A captured panel has an invalid SVG viewBox.") from exc
        if not all(math.isfinite(v) for v in (vx, vy, vw, vh)) or vw <= 0 or vh <= 0:
            raise ValueError("A captured panel has an invalid SVG viewBox.")
        if index == 0:
            background = next((child for child in panel
                               if child.tag == f"{{{_SVG_NS}}}rect"
                               and child.attrib.get("width") == "100%"
                               and child.attrib.get("height") == "100%"), None)
            if background is not None:
                ET.SubElement(root, f"{{{_SVG_NS}}}rect", {
                    "width": str(width), "height": str(height),
                    "fill": background.attrib.get("fill", "none"),
                    "fill-opacity": background.attrib.get("fill-opacity", "1")})
        # A group's percentage lengths refer to the outer SVG viewport, unlike
        # a nested SVG. Pin each exported background to its own original panel.
        for child in panel:
            if child.tag == f"{{{_SVG_NS}}}rect":
                for key, extent in (("width", vw), ("height", vh)):
                    if child.attrib.get(key) == "100%":
                        child.set(key, str(extent))
        structural = {"width", "height", "viewBox", "x", "y", "version", "baseProfile",
                      "preserveAspectRatio", "transform"}
        viewport = ET.SubElement(root, f"{{{_SVG_NS}}}g", {
            "transform": f"translate(0 {top}) scale({width / vw} {panel_height / vh}) "
                         f"translate({-vx} {-vy})"})
        content = ET.SubElement(viewport, f"{{{_SVG_NS}}}g", {
            key: value for key, value in panel.attrib.items() if key not in structural})
        if "transform" in panel.attrib:
            content.set("transform", panel.attrib["transform"])
        content.extend(list(panel))
        top += panel_height + 12.0
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), width, height


def capture_plot_preview(item, *, residual_item=None,
                         include_residual: bool = False) -> QtGui.QImage:
    """Bounded detached bitmap for the sizing dialog; never re-runs a plot."""
    images = [_capture_image(item, max_edge=1200)]
    if include_residual:
        if residual_item is None:
            raise ValueError("No residual panel is available for this figure.")
        images.append(_capture_image(residual_item, max_edge=1200))
    return images[0] if len(images) == 1 else _compose_images(images)


def capture_figure_render(item, *, residual_item=None, include_residual=False):
    """Return immutable PNG/SVG bytes and dimensions of the existing picture."""
    items = [item]
    if include_residual:
        if residual_item is None:
            raise ValueError("No residual panel is available for this figure.")
        items.append(residual_item)
    sizes = [_size(panel) for panel in items]
    images = [_capture_image(panel) for panel in items]
    vectors = [_capture_svg(panel) for panel in items]
    if len(items) == 1:
        return _encode_png(images[0]), vectors[0], *sizes[0]
    svg, width, height = _compose_svg(vectors, sizes)
    return _encode_png(_compose_images(images)), svg, width, height


def capture_figure_picture(item, *, residual_item=None, include_residual=False) -> bytes:
    """Record vector painter commands, including clipping and canvas background.

    QtSvg supports only SVG Tiny and cannot faithfully replay pyqtgraph's clip
    paths. QPicture records the same scene rendering used by native PNG/PDF,
    without later needing a live item or a scientific recomputation.
    """
    items = [item]
    if include_residual:
        if residual_item is None:
            raise ValueError("No residual panel is available for this figure.")
        items.append(residual_item)
    sizes = [_size(panel) for panel in items]
    width, height, heights = _composition_size(sizes)
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    if not painter.isActive():
        raise ValueError("The figure's vector rendering could not be recorded.")
    try:
        top = 0.0
        painter.fillRect(QtCore.QRectF(0, 0, width, height), ImageExporter(items[0]).parameters()["background"])
        for panel, panel_height in zip(items, heights):
            exporter = ImageExporter(panel)
            target = QtCore.QRectF(0, top, width, panel_height)
            background = exporter.parameters()["background"]
            painter.save()
            try:
                painter.setClipRect(target)
                painter.fillRect(target, background)
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
                try:
                    exporter.setExportMode(True, {"antialias": True, "background": background,
                                                  "painter": painter, "resolutionScale": 1.0})
                    exporter.getScene().render(painter, target, exporter.getSourceRect())
                finally:
                    exporter.setExportMode(False)
            finally:
                painter.restore()
            top += panel_height + 12.0
    finally:
        painter.end()
    picture.setBoundingRect(QtCore.QRect(0, 0, math.ceil(width), math.ceil(height)))
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    if not picture.save(buffer):
        raise ValueError("The figure's vector rendering could not be saved.")
    return bytes(buffer.data())


@dataclass(frozen=True)
class FigureSnapshot:
    png_bytes: bytes
    svg_bytes: bytes
    data_csv_bytes: bytes
    data_parquet_bytes: bytes
    config_json_bytes: bytes
    provenance_json_bytes: bytes
    width: float
    height: float
    includes_residual: bool = False
    additional_resources: tuple[tuple[str, bytes], ...] = ()

    def dataframe(self) -> pd.DataFrame:
        return _table_from_parquet(self.data_parquet_bytes)

    def config(self):
        return json.loads(self.config_json_bytes)

    def provenance(self):
        return json.loads(self.provenance_json_bytes)

    def to_storage(self) -> tuple[dict, dict[str, bytes]]:
        blobs = {name: getattr(self, field) for name, field in _BLOB_FIELDS.items()}
        resources = {}
        for index, (filename, payload) in enumerate(self.additional_resources):
            _resource_name(filename)
            if filename.casefold() in {name.casefold() for name in resources} or not isinstance(payload, bytes):
                raise ValueError("Additional figure resources must have unique names and byte payloads.")
            key = f"resource_{index}"
            resources[filename] = key
            blobs[key] = payload
        metadata = {"format": "nanoscope-figure-snapshot", "version": 1,
                    "width": self.width, "height": self.height,
                    "includes_residual": self.includes_residual,
                    "resources": resources,
                    "sha256": {name: hashlib.sha256(blob).hexdigest()
                               for name, blob in blobs.items()}}
        _json_bytes(metadata)
        return metadata, blobs

    @classmethod
    def from_storage(cls, metadata: Mapping, blobs: Mapping[str, bytes]):
        if not isinstance(metadata, Mapping) or not isinstance(blobs, Mapping):
            raise ValueError("Invalid figure snapshot metadata or payload mapping.")
        if (metadata.get("format") != "nanoscope-figure-snapshot"
                or type(metadata.get("version")) is not int or metadata["version"] != 1):
            raise ValueError("Unsupported figure snapshot format.")
        if (not {"width", "height", "includes_residual", "sha256"} <= metadata.keys()
                or not _BLOB_FIELDS.keys() <= blobs.keys()):
            raise ValueError("The figure snapshot is missing required metadata or payloads.")
        if any(type(metadata[key]) not in (int, float) for key in ("width", "height")):
            raise ValueError("Invalid frozen figure dimensions.")
        width, height = float(metadata["width"]), float(metadata["height"])
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError("Invalid frozen figure dimensions.")
        if type(metadata.get("includes_residual")) is not bool:
            raise ValueError("Invalid frozen residual-panel option.")
        checksums = metadata["sha256"]
        if not isinstance(checksums, Mapping):
            raise ValueError("Invalid figure checksum mapping.")
        resource_map = metadata.get("resources", {})
        if not isinstance(resource_map, Mapping):
            raise ValueError("Invalid figure resource mapping.")
        resource_keys, resource_names = set(), set()
        for filename, key in resource_map.items():
            _resource_name(filename)
            if (not isinstance(key, str) or not re.fullmatch(r"resource_[0-9]+", key)
                    or key in resource_keys or filename.casefold() in resource_names):
                raise ValueError("Invalid or duplicate figure resource mapping.")
            resource_keys.add(key)
            resource_names.add(filename.casefold())
        if not resource_keys <= blobs.keys():
            raise ValueError("The figure snapshot is missing an additional resource.")
        if set(checksums) != set(_BLOB_FIELDS) | resource_keys or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in checksums.values()):
            raise ValueError("Invalid or incomplete figure checksum mapping.")
        values = {}
        total = 0
        for name, field in _BLOB_FIELDS.items():
            value = blobs[name]
            if not isinstance(value, bytes):
                raise ValueError("Frozen figure payloads must be bytes.")
            if hashlib.sha256(value).hexdigest() != checksums.get(name):
                raise ValueError(f"Frozen figure {name} checksum does not match.")
            total += len(value)
            values[field] = value
        resources = []
        for filename, key in resource_map.items():
            value = blobs[key]
            if (not isinstance(value, bytes)
                    or hashlib.sha256(value).hexdigest() != checksums.get(key)):
                raise ValueError(f"Frozen figure {filename} checksum does not match.")
            total += len(value)
            resources.append((filename, value))
        if total > MAX_SNAPSHOT_BYTES:
            raise ValueError("The frozen figure exceeds the 512 MiB storage limit.")
        result = cls(**values, width=width, height=height,
                     includes_residual=metadata["includes_residual"],
                     additional_resources=tuple(resources))
        _json_bytes(result.config())
        _json_bytes(result.provenance())
        return result


def capture_figure_snapshot(item, *, data: pd.DataFrame, config=None, provenance=None,
                            residual_item=None, include_residual=False,
                            additional_resources: Mapping[str, bytes] | None = None) -> FigureSnapshot:
    """Freeze the picture plus the EXACT table the caller says it represents.

    CSV is human-readable (17-digit floats). Parquet retains numerical columns
    natively; schema-tagged JSON scalars preserve heterogeneous object columns,
    missing-value distinctions and index. No sampling or metric selection happens.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("A figure snapshot requires its exact plotted DataFrame.")
    config_json = _json_bytes({} if config is None else config)
    provenance_json = _json_bytes({} if provenance is None else provenance)
    resources = tuple((name, value) for name, value in (additional_resources or {}).items())
    for name, value in resources:
        _resource_name(name)
        if not isinstance(value, bytes):
            raise TypeError("Additional figure resources must contain bytes.")
        if name.casefold() == _PICTURE_RESOURCE:
            raise ValueError("The vector rendering resource is reserved for the captured figure.")
    table = data.copy(deep=True)
    parquet = _table_parquet_bytes(table)
    csv = table.to_csv(index=False, float_format="%.17g").encode("utf-8")
    png, svg, width, height = capture_figure_render(
        item, residual_item=residual_item, include_residual=include_residual)
    resources += ((_PICTURE_RESOURCE, capture_figure_picture(
        item, residual_item=residual_item, include_residual=include_residual)),)
    snapshot = FigureSnapshot(png, svg, csv, parquet, config_json,
                              provenance_json, width, height, bool(include_residual), resources)
    metadata, blobs = snapshot.to_storage()
    return FigureSnapshot.from_storage(metadata, blobs)


def write_figure_bundle(path: str | Path, snapshot: FigureSnapshot) -> Path:
    """Atomically write a portable ZIP; an earlier bundle survives any failure."""
    metadata, blobs = snapshot.to_storage()
    FigureSnapshot.from_storage(metadata, blobs)
    files = {**_FILES, **{key: name for name, key in metadata["resources"].items()}}
    manifest = {**metadata, "files": files,
                "data_contract": "data.csv has 17-digit floats. data.parquet keeps numerical columns native; "
                                 "its nanoscope.figure_table_codec schema metadata identifies any object "
                                 "columns/index levels encoded as tagged JSON scalars. FigureSnapshot.dataframe() "
                                 "restores their exact types and missing values. No pickle is used.",
                "picture_contract": "frozen visible items; no analysis or fit was re-run"}
    out = Path(path)
    with staged_write(out) as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, filename in files.items():
                archive.writestr(filename, blobs[name])
            archive.writestr("manifest.json", _json_bytes(manifest))
    return out


def render_frozen_picture(payload: bytes, target: str | Path, *, width=0, height=0,
                          native_size) -> None:
    """Replay immutable Qt vector commands with their original clipping and fonts."""
    picture = QtGui.QPicture()
    buffer = QtCore.QBuffer()
    buffer.setData(QtCore.QByteArray(payload))
    buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
    if not picture.load(buffer):
        raise ValueError("The frozen vector rendering is invalid.")
    native_w, native_h = map(float, native_size)
    w, h = float(width), float(height)
    if w <= 0 and h <= 0:
        w, h = native_w, native_h
    elif w <= 0:
        w = h * native_w / native_h
    elif h <= 0:
        h = w * native_h / native_w
    if not all(math.isfinite(v) and v > 0 for v in (w, h, native_w, native_h)):
        raise ValueError("The frozen figure requires finite, positive export dimensions.")
    out = Path(target)
    if out.suffix.lower() == ".pdf":
        device = QtGui.QPdfWriter(str(out))
        device.setResolution(72)
        device.setPageSize(QtGui.QPageSize(QtCore.QSizeF(w, h), QtGui.QPageSize.Unit.Point,
                                           "figure", QtGui.QPageSize.SizeMatchPolicy.ExactMatch))
        device.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
    elif out.suffix.lower() == ".png":
        if max(w, h) > 32768 or w * h > 100_000_000:
            raise ValueError("The frozen figure exceeds the safe raster size.")
        device = QtGui.QImage(max(1, round(w)), max(1, round(h)), QtGui.QImage.Format.Format_ARGB32)
        device.fill(QtCore.Qt.GlobalColor.transparent)
    else:
        raise ValueError("Frozen vector rendering supports PNG and PDF only.")
    painter = QtGui.QPainter(device)
    if not painter.isActive():
        raise ValueError("The frozen figure could not be rendered.")
    try:
        scale = min(w / native_w, h / native_h)
        painter.translate((w - native_w * scale) / 2, (h - native_h * scale) / 2)
        painter.scale(scale, scale)
        # QPicture accounts for device DPI when played. Counter that conversion
        # because our explicit canvas transform already establishes pixel/point units.
        painter.scale(picture.logicalDpiX() / device.logicalDpiX(),
                      picture.logicalDpiY() / device.logicalDpiY())
        if not picture.play(painter):
            raise ValueError("The frozen vector rendering could not be replayed.")
    finally:
        painter.end()
    if isinstance(device, QtGui.QImage) and not device.save(str(out), "PNG"):
        raise ValueError("The frozen PNG could not be written.")


def write_frozen_figure_image(path: str | Path, snapshot: FigureSnapshot,
                              options) -> list[str]:
    """Atomically export a saved picture using the live publication sizing contract.

    Native PNG/SVG exports retain the captured bytes exactly. Physical PNG and
    PDF are painted from stored vector commands, never enlarged from the
    thumbnail, and never access a recording, analysis, or live plot. A residual
    panel, if captured, is part of the frozen picture and cannot be added later.
    """
    from .plot_export import apply_png_resolution, apply_svg_physical_size
    from .plot_export_options import ExportOptions, resolve_export_options

    if not isinstance(snapshot, FigureSnapshot) or not isinstance(options, ExportOptions):
        raise TypeError("A frozen image export requires a FigureSnapshot and ExportOptions.")
    out = Path(path)
    if out.suffix.lower() != options.extension:
        raise ValueError("The destination extension must match the selected image format.")
    if options.include_residual and not snapshot.includes_residual:
        raise ValueError("This saved figure does not contain a residual panel.")
    validated = resolve_export_options(
        (snapshot.width, snapshot.height), extension=options.extension,
        native=options.native, width=options.physical_width_mm,
        height=options.physical_height_mm, dpi=options.dpi)
    if (validated.width, validated.height) != (options.width, options.height):
        raise ValueError("The export pixel/point dimensions do not match its physical size.")
    with staged_path(out) as staged:
        if options.extension == ".svg":
            staged.write_bytes(snapshot.svg_bytes)
            if not options.native:
                apply_svg_physical_size(staged, width_mm=options.physical_width_mm,
                                        height_mm=options.physical_height_mm)
        elif options.extension == ".png" and options.native:
            if QtGui.QImage.fromData(snapshot.png_bytes, "PNG").isNull():
                raise ValueError("The frozen PNG is invalid.")
            staged.write_bytes(snapshot.png_bytes)
        else:
            picture = dict(snapshot.additional_resources).get(_PICTURE_RESOURCE)
            if picture is None:
                raise ValueError("This saved figure lacks a faithful vector rendering. "
                                 "Export its native PNG/SVG, or capture a new snapshot.")
            render_frozen_picture(picture, staged, width=options.width,
                                  height=options.height,
                                  native_size=(snapshot.width, snapshot.height))
            if options.extension == ".png":
                apply_png_resolution(staged, dpi=options.dpi)
        if not staged.is_file() or staged.stat().st_size == 0:
            raise ValueError("The frozen figure did not produce an image.")
    return take_identity_notes(out)
