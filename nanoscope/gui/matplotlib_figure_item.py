"""Frozen Matplotlib drawings for NanoScope's existing figure workflow.

Only ``update_figure`` visits the supplied artists. Display, capture, composition
and export replay immutable vector commands, never an estimator. SVG is retained
from Matplotlib itself; PNG/PDF and composition use QPicture, not QtSvg's lossy
SVG Tiny clipping implementation. Literal text retains editable font commands;
mathematical glyph layout retains its exact vector-path fallback.
"""
from __future__ import annotations

import io
import math
import weakref

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


_QT_FONT_FAMILIES = {}


def _font_families(prop):
    """Register the exact faces Matplotlib resolved, including bundled fallback fonts."""
    from matplotlib import font_manager
    families = []
    for filename in font_manager.fontManager._find_fonts_by_props(prop):
        if filename not in _QT_FONT_FAMILIES:
            identifier = QtGui.QFontDatabase.addApplicationFont(str(filename))
            names = tuple(QtGui.QFontDatabase.applicationFontFamilies(identifier)) if identifier >= 0 else ()
            if not names:
                raise ValueError('The resolved figure font could not be retained for editable export.')
            _QT_FONT_FAMILIES[filename] = names
        families.extend(name for name in _QT_FONT_FAMILIES[filename] if name not in families)
    return families


def restore_retained_font_faces(descriptors):
    """Restore recorded face names from installed/bundled fonts before QPicture replay.

    No font programs or arbitrary paths are accepted from a project. A missing recorded
    face refuses faithful PDF replay; the original SVG/PNG bytes remain exportable.
    """
    from matplotlib.font_manager import FontProperties, findfont
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) > 256:
        raise ValueError('The retained font metadata exceeds the replay budget.')
    for descriptor in descriptors:
        families = descriptor.get('families') if isinstance(descriptor, dict) else None
        if (not isinstance(families, (list, tuple)) or not 1 <= len(families) <= 32
                or any(not isinstance(name, str) or not 0 < len(name) <= 200 for name in families)):
            raise ValueError('The retained figure font metadata is invalid.')
        options = {key: descriptor.get(key, default) for key, default in
                   (('style', 'normal'), ('weight', 'normal'), ('stretch', 'normal'))}
        try:
            for family in families:
                findfont(FontProperties(family=[family], **options), fallback_to_default=False)
            _font_families(FontProperties(family=list(families), **options))
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError('A recorded figure font is unavailable for faithful replay. '
                             'Use its original PNG/SVG or restore the recorded font. ' + str(exc)) from exc


class _PictureRenderer(RendererBase):
    """Translate already prepared artists into native vector painter commands."""

    def __init__(self, painter, width, height, dpi):
        super().__init__()
        self.painter = painter
        self.width, self.height, self.dpi = width, height, dpi
        self.qt_font_families = set()
        self.qt_font_faces = {}

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
        if ismath:
            text_gc = self.new_gc()
            text_gc.copy_properties(gc)
            self._draw_text_as_path(text_gc, x, y, text, prop, angle, ismath)
            return
        # Store text commands, so QPdfWriter embeds fonts rather than glyph outlines.
        # The renderer's coordinates are figure pixels; QFont sizes are device points.
        from matplotlib.font_manager import weight_dict, stretch_dict
        families = _font_families(prop)
        self.qt_font_families.update(families)
        descriptor = {'families': families, 'style': prop.get_style(),
                      'weight': prop.get_weight(), 'stretch': prop.get_stretch()}
        self.qt_font_faces[repr(descriptor)] = descriptor
        font = QtGui.QFont()
        font.setFamilies(families)
        font.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
        stretch = prop.get_stretch()
        # Matplotlib's CSS selection scale is100..900 (normal500); Qt takes
        # percentages (normal100). Passing500 stretches all literal text fivefold.
        numeric_stretch = stretch_dict[stretch] if isinstance(stretch, str) else float(stretch)
        font.setStretch(round(float(np.interp(numeric_stretch,
            [100, 200, 300, 400, 500, 600, 700, 800, 900],
            [50, 62, 75, 87, 100, 112, 125, 150, 200]))))
        font.setPointSizeF(float(prop.get_size_in_points()) * self.dpi / self.painter.device().logicalDpiY())
        weight = prop.get_weight()
        font.setWeight(QtGui.QFont.Weight(int(weight_dict.get(weight, weight) if isinstance(weight, str) else weight)))
        font.setItalic(prop.get_style() in ('italic', 'oblique'))
        painter = self.painter
        painter.save()
        try:
            self._clip(gc)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            painter.setFont(font)
            painter.setPen(_color(gc.get_rgb(), gc))
            painter.translate(float(x), self.height - float(y))
            painter.rotate(-float(angle))
            painter.drawText(QtCore.QPointF(0., 0.), str(text))
        finally:
            painter.restore()

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

    seriesSelected = QtCore.Signal(str)

    def __init__(self, figure: Figure, parent=None):
        super().__init__(parent)
        self._rect = QtCore.QRectF()
        self._picture = QtGui.QPicture()
        self._svg = b""
        self._targets, self._axes = (), ()
        self._legend_rects = {}
        self._series_geometry = {}
        self._controllers = []
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton | QtCore.Qt.MouseButton.RightButton)
        self.update_figure(figure)

    def text_targets(self):
        return self._targets

    def axis_targets(self):
        return self._axes

    def legend_rect(self, panel):
        return self._legend_rects.get(panel)

    def set_text_controllers(self, controllers):
        self._controllers = [weakref.ref(c) for c in controllers]

    def _hit(self, point):
        # The smallest painted text wins where a long tick-group extent overlaps it.
        hits = [target for target in self._targets if target.visible
                and QtCore.QRectF(*target.rect).adjusted(-2, -2, 2, 2).contains(point)]
        return min(hits, key=lambda t: t.rect[2] * t.rect[3]) if hits else None

    def _text_event(self, event, *, double=False):
        target = self._hit(event.pos())
        controllers = [c for ref in self._controllers if (c := ref()) is not None]
        controller = next((c for c in controllers if target is not None and c.panel == target.panel), None)
        if controller is None:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                for other in controllers: other.cleared.emit()
            return False
        controller.claimed.emit(target.key)
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            controller._raise_menu_at(target.key, event.screenPos())
        elif double:
            if target.key.endswith(".ticks"):
                controller.open_axis_properties(target.key.split(".")[1])
            else:
                controller._edit_or_style(target.key)
        event.accept()
        return True

    def series_geometry(self):
        return self._series_geometry

    def series_at(self, point):
        return next((key for key, geometry in reversed(tuple(self._series_geometry.items()))
                     if geometry.hit(point)), None)

    def mousePressEvent(self, event):  # noqa: N802
        if self._text_event(event):
            return
        key = self.series_at(event.pos()) if event.button() == QtCore.Qt.MouseButton.LeftButton else None
        if key is not None:
            self.seriesSelected.emit(key)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if not self._text_event(event, double=True): super().mouseDoubleClickEvent(event)

    def figure_export_target(self):
        return self

    def native_figure_svg(self) -> bytes:
        """Native vector protocol, intentionally detached from Matplotlib."""
        return self._svg

    def capture_rendering_metadata(self):
        """Facts recorded with this immutable drawing, without revisiting its Figure."""
        from copy import deepcopy
        return deepcopy(self._rendering_metadata)

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
            renderer = _PictureRenderer(painter, width, height, dpi)
            figure.draw(renderer)
            from ..plot.matplotlib_targets import capture_targets, panel_identity
            targets, axes = capture_targets(figure, renderer)
            from .matplotlib_series import capture_series_geometry
            series_geometry = capture_series_geometry(figure, height)
            legend_rects = {}
            for panel, legend in [*(('figure', legend) for legend in figure.legends),
                                  *((panel_identity(axis, i), axis.get_legend())
                                    for i, axis in enumerate(figure.axes) if axis.get_legend() is not None)]:
                if not legend.get_visible():
                    continue
                box = legend.get_window_extent(renderer)
                legend_rects[panel] = (float(box.x0), height - float(box.y1),
                                       float(box.width), float(box.height))
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
            with rc_context({"svg.fonttype": "none", "svg.hashsalt": "nanoscope-comparison",
                             "savefig.bbox": None, "svg.image_inline": True}):
                figure.savefig(stream, format="svg", transparent=False,
                               metadata={"Date": None}, bbox_inches=None)
        finally:
            figure.set_canvas(canvas)
            figure.set_layout_engine(layout_engine)
        from ..plot.svg_fonts import editable_font_stacks
        svg = editable_font_stacks(stream.getvalue())
        if not svg or picture.isNull():
            raise ValueError("The comparison figure did not produce a vector drawing.")
        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(0, 0, width, height)
        self._picture, self._svg = picture, svg
        import matplotlib
        from matplotlib.text import Text
        from ..plot.quantity_axes import figure_quantity_metadata
        self._rendering_metadata = {
            "renderer": "matplotlib-to-QPicture", "matplotlib_version": matplotlib.__version__,
            "display_basis": figure_quantity_metadata(figure),
            "figure_dpi": dpi, "retained_qt_font_families": sorted(renderer.qt_font_families),
            "retained_text_faces": [renderer.qt_font_faces[key] for key in sorted(renderer.qt_font_faces)],
            "picture_logical_dpi": [picture.logicalDpiX(), picture.logicalDpiY()],
            "requested_font_families": sorted({family for text in figure.findobj(Text)
                                                for family in text.get_fontfamily()}),
            "svg_text_policy": "editable text with fallback families",
            "picture_text_policy": "literal text commands; mathematical glyph paths",
        }
        self._targets, self._axes = targets, axes
        self._legend_rects = legend_rects
        self._series_geometry = series_geometry
        self.update()
