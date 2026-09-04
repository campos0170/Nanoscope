"""Shared staged export primitives for live pyqtgraph analysis panels.

The numerical tables remain panel-owned: each panel knows what its picture means.
This module owns only the format switch and the staged-write/error wording so PNG,
SVG, PDF and CSV behave identically across those panels.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets
from pyqtgraph.exporters import ImageExporter

from ..plot import pg_svg
from ..plot.figures import export_plot_pdf, save_plot_data, save_plot_image

IMAGE_FILTERS = (
    "PNG image (*.png);;SVG vector image (*.svg);;"
    "PDF vector document (*.pdf);;All files (*)"
)
DATA_FILTERS = "CSV data (*.csv);;All files (*)"
IMAGE_FORMATS = (".png", ".svg", ".pdf")


class PlotDataTooLarge(ValueError):
    """An exact long-form CSV would be large enough to freeze the GUI.

    This is a refusal, not a reduction: callers catch it through the same staged
    export path, report its actionable message, and leave both the visible result
    and any previous file untouched.
    """


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
                          height: int = 0) -> list[str]:
    """Write a pyqtgraph item/scene as PNG, SVG or PDF through one staged seam."""
    out = Path(target)
    suffix = out.suffix.lower()
    w, h = int(width), int(height)

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
