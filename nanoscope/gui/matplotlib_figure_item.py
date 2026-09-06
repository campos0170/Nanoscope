"""Frozen Matplotlib drawings for NanoScope's existing figure workflow.

Only ``update_figure`` visits the supplied artists. Display, capture, composition
and export replay immutable vector commands, never an estimator. SVG is retained
from Matplotlib itself; PNG/PDF and composition use QPicture, not QtSvg's lossy
SVG Tiny clipping implementation. Text uses vector glyph outlines.
"""
from __future__ import annotations

import io
import math

import numpy as np
from matplotlib import rc_context
from matplotlib.backend_bases import RendererBase
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from PySide6 import QtCore, QtGui, QtWidgets


def _qt_path(path, transform, height):
    """Convert transformed vertices without simplifying plotted data."""
    result = QtGui.QPainterPath()
    result.setFillRule(QtCore.Qt.FillRule.WindingFill)
    for vertices, code in path.iter_segments(transform, remove_nans=True,
                                              simplify=False, curves=True):
        points = np.asarray(vertices, dtype=float).reshape(-1, 2)
        if code == MplPath.CLOSEPOLY:
            result.closeSubpath()
        elif code == MplPath.MOVETO:
            result.moveTo(float(points[0, 0]), height - float(points[0, 1]))
        elif code == MplPath.LINETO:
            result.lineTo(float(points[0, 0]), height - float(points[0, 1]))
        elif code == MplPath.CURVE3:
            result.quadTo(float(points[0, 0]), height - float(points[0, 1]),
                          float(points[1, 0]), height - float(points[1, 1]))
        elif code == MplPath.CURVE4:
            result.cubicTo(float(points[0, 0]), height - float(points[0, 1]),
                           float(points[1, 0]), height - float(points[1, 1]),
                           float(points[2, 0]), height - float(points[2, 1]))
    return result


def _color(value, gc):
    red, green, blue, alpha = to_rgba(value)
    if gc.get_forced_alpha():
        alpha = gc.get_alpha()
    return QtGui.QColor.fromRgbF(red, green, blue, alpha)


class _PictureRenderer(RendererBase):
    """Translate already prepared artists into native vector painter commands."""

    def __init__(self, painter, width, height, dpi):
        super().__init__()
        self.painter = painter
        self.width, self.height, self.dpi = width, height, dpi

    def get_canvas_width_height(self):
        return self.width, self.height

    def points_to_pixels(self, points):
        return points * self.dpi / 72.0

    def flipy(self):
        return False

    def _clip(self, gc):
        clip = gc.get_clip_rectangle()
        if clip is not None:
            x, y, width, height = clip.bounds
            self.painter.setClipRect(QtCore.QRectF(x, self.height - y - height, width, height),
                                     QtCore.Qt.ClipOperation.IntersectClip)
        path, transform = gc.get_clip_path()
        if path is not None:
            self.painter.setClipPath(_qt_path(path, transform, self.height),
                                     QtCore.Qt.ClipOperation.IntersectClip)

    def draw_path(self, gc, path, transform, rgbFace=None):
        if gc.get_hatch():
            raise ValueError("Hatched artists are not supported in comparison figures.")
        painter = self.painter
        painter.save()
        try:
            self._clip(gc)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing,
                                  bool(gc.get_antialiased()))
            linewidth = float(self.points_to_pixels(gc.get_linewidth()))
            if linewidth > 0:
                pen = QtGui.QPen(_color(gc.get_rgb(), gc), linewidth)
                pen.setCapStyle({"butt": QtCore.Qt.PenCapStyle.FlatCap,
                                 "round": QtCore.Qt.PenCapStyle.RoundCap,
                                 "projecting": QtCore.Qt.PenCapStyle.SquareCap}[gc.get_capstyle()])
                pen.setJoinStyle({"miter": QtCore.Qt.PenJoinStyle.MiterJoin,
                                  "round": QtCore.Qt.PenJoinStyle.RoundJoin,
                                  "bevel": QtCore.Qt.PenJoinStyle.BevelJoin}[gc.get_joinstyle()])
                offset, dashes = gc.get_dashes()
                if dashes is not None and len(dashes):
                    pen.setDashPattern([max(1e-6, float(value) / gc.get_linewidth())
                                        for value in dashes])
                    pen.setDashOffset(float(offset) / gc.get_linewidth())
                painter.setPen(pen)
            else:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush if rgbFace is None
                              else QtGui.QBrush(_color(rgbFace, gc)))
            painter.drawPath(_qt_path(path, transform, self.height))
        finally:
            painter.restore()

    def draw_text(self, gc, x, y, text, prop, angle, ismath=False, mtext=None):
        text_gc = self.new_gc()
        text_gc.copy_properties(gc)
        self._draw_text_as_path(text_gc, x, y, text, prop, angle, ismath)

    def draw_tex(self, *args, **kwargs):
        raise ValueError("External TeX rendering is not supported in comparison figures.")

    def draw_gouraud_triangles(self, *args, **kwargs):
        raise ValueError("Gouraud-shaded artists are not supported in comparison figures.")

    def draw_image(self, gc, x, y, im, transform=None):
        if transform is not None:
            raise ValueError("Transformed image artists are not supported in comparison figures.")
        pixels = np.ascontiguousarray(np.flipud(im), dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("A comparison image must contain RGBA pixels.")
        height, width = pixels.shape[:2]
        image = QtGui.QImage(pixels.data, width, height, int(pixels.strides[0]),
                             QtGui.QImage.Format.Format_RGBA8888).copy()
        self.painter.save()
        try:
            self._clip(gc)
            self.painter.setOpacity(gc.get_alpha())
            self.painter.drawImage(QtCore.QPointF(x, self.height - y - height), image)
        finally:
            self.painter.restore()


class MatplotlibFigureItem(QtWidgets.QGraphicsObject):
    """Static vector canvas compatible with Figure Library/export/composition.

    Add to a QGraphicsScene shown by a QGraphicsView before using export helpers.
    Updates are transactional and never alter a view's zoom/scrollbar values.
    The source Figure is not retained and cannot be consulted by later exports.
    """

    def __init__(self, figure: Figure, parent=None):
        super().__init__(parent)
        self._rect = QtCore.QRectF()
        self._picture = QtGui.QPicture()
        self._svg = b""
        self.update_figure(figure)

    def figure_export_target(self):
        return self

    def native_figure_svg(self) -> bytes:
        """Native vector protocol, intentionally detached from Matplotlib."""
        return self._svg

    def boundingRect(self):
        return QtCore.QRectF(self._rect)

    def mapRectToDevice(self, rect):
        # pyqtgraph's exporters ask their target for its default pixel size. A
        # static publication canvas owns that size; incidental viewer zoom does
        # not resize or crop the saved artwork.
        return QtCore.QRectF(rect)

    def paint(self, painter, option, widget=None):
        painter.save()
        try:
            # Outer scene/export transforms already establish display units.
            device = painter.device()
            painter.scale(self._picture.logicalDpiX() / device.logicalDpiX(),
                          self._picture.logicalDpiY() / device.logicalDpiY())
            self._picture.play(painter)
        finally:
            painter.restore()

    def update_figure(self, figure: Figure) -> None:
        if not isinstance(figure, Figure):
            raise TypeError("A comparison canvas requires a Matplotlib Figure.")
        width, height = map(float, figure.get_size_inches() * figure.dpi)
        dpi = float(figure.dpi)
        if (not all(math.isfinite(v) and v > 0 for v in (width, height, dpi))
                or max(width, height) > 32768 or width * height > 100_000_000):
            raise ValueError("The comparison figure requires a finite, safely sized canvas.")
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        if not painter.isActive():
            raise ValueError("The comparison figure could not be recorded.")
        try:
            figure.draw(_PictureRenderer(painter, width, height, dpi))
        finally:
            painter.end()
        picture.setBoundingRect(QtCore.QRect(0, 0, math.ceil(width), math.ceil(height)))
        stream = io.BytesIO()
        canvas = figure.canvas
        layout_engine = figure.get_layout_engine()
        try:
            # A second backend must not solve constrained/tight layout again:
            # both frozen formats describe the axes positions just recorded.
            figure.set_layout_engine("none")
            with rc_context({"svg.fonttype": "path", "svg.hashsalt": "nanoscope-comparison",
                             "savefig.bbox": None, "svg.image_inline": True}):
                figure.savefig(stream, format="svg", transparent=False,
                               metadata={"Date": None}, bbox_inches=None)
        finally:
            figure.set_canvas(canvas)
            figure.set_layout_engine(layout_engine)
        svg = stream.getvalue()
        if not svg or picture.isNull():
            raise ValueError("The comparison figure did not produce a vector drawing.")
        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(0, 0, width, height)
        self._picture, self._svg = picture, svg
        self.update()
