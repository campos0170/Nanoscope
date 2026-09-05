"""Shared staged export primitives for live pyqtgraph analysis panels.

The numerical tables remain panel-owned: each panel knows what its picture means.
This module owns only the format switch and the staged-write/error wording so PNG,
SVG, PDF and CSV behave identically across those panels.
"""
from __future__ import annotations

from pathlib import Path
import math
import struct
import zlib
import xml.dom.minidom as _xml

from PySide6 import QtWidgets
from pyqtgraph.exporters import ImageExporter

from ..plot import pg_svg
from ..plot.figures import export_plot_pdf, save_plot_data, save_plot_image
from .export_paths import ask_export_path
from .plot_export_options import ExportOptions, choose_export_options, resolve_export_options

IMAGE_FILTERS = (
    "PNG image (*.png);;SVG vector image (*.svg);;"
    "PDF vector document (*.pdf);;All files (*)"
)
DATA_FILTERS = "CSV data (*.csv);;All files (*)"
IMAGE_FORMATS = (".png", ".svg", ".pdf")


def _format_path(path: str | Path, extension: str) -> Path:
    """Match the chosen format while retaining unrecognised filename suffixes."""
    out = Path(path)
    if out.suffix.lower() == extension:
        return out
    if out.suffix.lower() in IMAGE_FORMATS:
        return out.with_suffix(extension)
    return Path(str(out) + extension)


def choose_live_plot_export(parent, item, suggestion: str,
                            initial_width: int = 0, initial_height: int = 0
                            ) -> tuple[Path, ExportOptions] | None:
    """Choose current-figure sizing and then confirm the exact destination file.

    This is for interactive calls only; programmatic ``save_image(path)`` callers
    retain their existing export behavior without another dialog.
    """
    rect = item.sceneRect() if isinstance(item, QtWidgets.QGraphicsScene) else item.sceneBoundingRect()
    extension = Path(suggestion).suffix.lower()
    if extension not in IMAGE_FORMATS:
        extension = ".png"
    try:
        options = choose_export_options(
            parent, rect.size(), initial_format=extension,
            initial_width=initial_width, initial_height=initial_height)
    except ValueError as exc:
        QtWidgets.QMessageBox.warning(parent, "Export current figure", str(exc))
        return None
    if options is None:
        return None
    labels = {".png": "PNG image", ".pdf": "PDF vector document", ".svg": "SVG vector image"}
    chosen = ask_export_path(
        parent, title="Export current figure", suggestion=str(_format_path(suggestion, options.extension)),
        filters=f"{labels[options.extension]} (*{options.extension})",
        outputs=lambda path: [_format_path(path, options.extension)])
    if chosen is None:
        return None
    return _format_path(chosen, options.extension), options


def apply_svg_physical_size(path: str | Path, *, width_mm: float,
                            height_mm: float) -> None:
    """Set a staged SVG's physical viewport without changing its viewBox or paths.

    Call this inside ``save_plot_image``'s render callback, before the staged file
    replaces the user's destination. Invalid size or SVG content fails safely there.
    """
    options = resolve_export_options(
        (1, 1), extension="svg", native=False, width=width_mm, height=height_mm)
    staged = Path(path)
    document = _xml.parseString(staged.read_bytes())
    try:
        root = document.documentElement
        if root.localName != "svg":
            raise ValueError("The staged vector export is not an SVG document.")
        if not root.hasAttribute("viewBox"):
            raise ValueError("The staged SVG has no viewBox; physical sizing cannot be applied safely.")
        root.setAttribute("width", f"{options.physical_width_mm:.10g}mm")
        root.setAttribute("height", f"{options.physical_height_mm:.10g}mm")
        staged.write_bytes(document.toxml(encoding="utf-8"))
    finally:
        document.unlink()


class PlotDataTooLarge(ValueError):
    """An exact long-form CSV would be large enough to freeze the GUI.

    This is a refusal, not a reduction: callers catch it through the same staged
    export path, report its actionable message, and leave both the visible result
    and any previous file untouched.
    """


def apply_png_resolution(path: str | Path, *, dpi: float) -> None:
    """Set staged PNG resolution metadata, leaving every pixel chunk untouched."""
    if not math.isfinite(dpi) or not 1 <= dpi <= 9600:
        raise ValueError("PNG resolution must be between 1 and 9600 DPI.")
    staged = Path(path)
    source = staged.read_bytes()
    signature = b"\x89PNG\r\n\x1a\n"
    if not source.startswith(signature):
        raise ValueError("The staged raster export is not a PNG image.")
    ppm = round(dpi / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    resolution = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
    chunks = [signature]
    offset = len(signature)
    ended = inserted = False
    while offset + 12 <= len(source):
        size = struct.unpack_from(">I", source, offset)[0]
        end = offset + size + 12
        if end > len(source):
            raise ValueError("The staged PNG contains a truncated chunk.")
        kind = source[offset + 4:offset + 8]
        if kind != b"pHYs":
            chunks.append(source[offset:end])
        if kind == b"IHDR":
            chunks.append(resolution)
            inserted = True
        offset = end
        if kind == b"IEND":
            ended = True
            break
    if not inserted or not ended or offset != len(source):
        raise ValueError("The staged PNG structure is incomplete.")
    staged.write_bytes(b"".join(chunks))


def require_csv_row_budget(row_count: int, *, limit: int, description: str,
                           advice: str) -> None:
    """Refuse an oversized exact table before allocating its rows."""
    count, ceiling = int(row_count), int(limit)
    if count <= ceiling:
        return
    raise PlotDataTooLarge(
        f"The visible {description} contains {count:,} exact values. Writing its "
        f"long-form CSV would require {count:,} data rows, above NanoScope's "
        f"interactive safety limit of {ceiling:,}. No values were sampled or omitted, "
        f"and no file was written. {advice}")


class _SceneItem:
    """The tiny item interface ``export_plot_pdf`` needs for a whole graphics scene.

    A spectrogram's colour bar is a sibling in its scene, not necessarily inside the
    PlotItem's export bounds.  Wrapping the scene lets the established PDF renderer paint
    the same rectangle the PNG/SVG exporters receive without modifying that renderer.
    """

    def __init__(self, scene: QtWidgets.QGraphicsScene) -> None:
        self._scene = scene

    def scene(self) -> QtWidgets.QGraphicsScene:
        return self._scene

    def sceneBoundingRect(self):
        return self._scene.sceneRect()

    def childItems(self):
        return self._scene.items()


def write_live_plot_image(target: str | Path, item, *, width: int = 0,
                          height: int = 0, physical_width_mm: float | None = None,
                          physical_height_mm: float | None = None,
                          dpi: float | None = None) -> list[str]:
    """Write a pyqtgraph item/scene as PNG, SVG or PDF through one staged seam."""
    out = Path(target)
    suffix = out.suffix.lower()
    w, h = int(width), int(height)

    if (physical_width_mm is None) != (physical_height_mm is None):
        raise ValueError("Both physical width and height are required.")

    if suffix == ".svg":
        # A whole GraphicsScene is the correct export target when a colour bar is a
        # sibling of the plot.  ``pg_svg.refresh_for_export`` normally reaches the
        # scene through ``item.scene()``; the scene itself has no such owner, so run
        # pyqtgraph's identical pre-paint pass explicitly before exporting it.
        if isinstance(item, QtWidgets.QGraphicsScene):
            prepare = getattr(item, "prepareForPaint", None)

            def render(staged):
                if prepare is not None:
                    prepare()
                return pg_svg.export_svg(item, staged, refresh=False)
        else:
            render = lambda staged: pg_svg.export_svg(item, staged)  # noqa: E731
        if physical_width_mm is not None:
            svg_render = render

            def render(staged):
                result = svg_render(staged)
                apply_svg_physical_size(
                    staged, width_mm=physical_width_mm, height_mm=physical_height_mm)
                return result
    elif suffix == ".pdf":
        pdf_item = _SceneItem(item) if isinstance(item, QtWidgets.QGraphicsScene) else item
        render = lambda staged: export_plot_pdf(  # noqa: E731
            pdf_item, staged, width=w, height=h)
    else:
        exporter = ImageExporter(item)
        params = exporter.parameters()
        if w > 0 and h > 0:
            params.param("width").setValue(w, blockSignal=exporter.widthChanged)
            params.param("height").setValue(h, blockSignal=exporter.heightChanged)
        elif w > 0:
            params["width"] = w
        elif h > 0:
            params["height"] = h
        render = lambda staged: exporter.export(str(staged))  # noqa: E731
        if suffix == ".png" and dpi is not None:
            png_render = render

            def render(staged):
                result = png_render(staged)
                if result is not False:
                    apply_png_resolution(staged, dpi=dpi)
                return result
    return save_plot_image(out, render)


def write_plot_csv(target: str | Path, data) -> list[str]:
    """Write a panel-produced DataFrame through the established staged CSV seam."""
    return save_plot_data(Path(target), data)


def data_write_failure_text(target: str | Path, exc: BaseException) -> str:
    """Truthful CSV failure text after :func:`write_plot_csv` abandoned its staging."""
    out = Path(target)
    if out.exists():
        state = (f"The existing {out.name} was left unchanged; it is the earlier export, "
                 "not the data currently on screen.")
    else:
        state = f"No {out.name} file was created."
    return f"Could not export plot data to {out.name}.\n\n{exc}\n\n{state}"
