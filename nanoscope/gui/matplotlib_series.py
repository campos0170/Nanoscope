"""Frozen artist hit geometry and viewport-only series selection decoration."""
from dataclasses import dataclass, field
import weakref

import numpy as np
from matplotlib.collections import Collection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PySide6 import QtCore, QtGui, QtWidgets

from .figure_text import _alive


@dataclass
class SeriesGeometry:
    bounds: QtCore.QRectF = field(default_factory=QtCore.QRectF)
    paths: list = field(default_factory=list)
    points: list = field(default_factory=list)

    def hit(self, point):
        if not self.bounds.adjusted(-5, -5, 5, 5).contains(point):
            return False
        if any(path.contains(point) for path in self.paths):
            return True
        for centers, radii in self.points:
            squared = (centers[:, 0] - point.x()) ** 2 + (centers[:, 1] - point.y()) ** 2
            if np.any(squared <= (radii + 3.) ** 2):
                return True
        return False


def capture_series_geometry(figure, height):
    """Copy display geometry once, after layout; never retain artists or estimators."""
    from .matplotlib_figure_item import _qt_path
    result = {}

    def entry(key):
        return result.setdefault(key, SeriesGeometry())

    def clip_for(artist):
        box = artist.get_clip_box() if artist.get_clip_on() else None
        return (QtCore.QRectF(float(box.x0), height - float(box.y1), float(box.width), float(box.height))
                if box is not None else None)

    def add_path(key, path, artist, *, filled=False):
        if not key or path.isEmpty():
            return
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(max(6., float(artist.get_linewidth()) * figure.dpi / 72. + 4.)
                         if callable(getattr(artist, 'get_linewidth', None)) else 6.)
        hit = stroker.createStroke(path)
        if filled:
            hit = hit.united(path)
        clip = clip_for(artist)
        if clip is not None:
            boundary = QtGui.QPainterPath(); boundary.addRect(clip)
            hit = hit.intersected(boundary)
        if hit.isEmpty():
            return
        target = entry(key); target.paths.append(hit)
        target.bounds = target.bounds.united(hit.boundingRect())

    def add_points(keys, centers, sizes, artist):
        centers = np.asarray(np.ma.filled(centers, np.nan), dtype=float).reshape(-1, 2).copy()
        if not len(centers):
            return
        centers[:, 1] = height - centers[:, 1]
        sizes = np.asarray(sizes, dtype=float).ravel()
        if not len(sizes):
            return
        radii = np.sqrt(np.maximum(0, sizes[np.arange(len(centers)) % len(sizes)])) * figure.dpi / 144.
        keys = np.asarray([keys] * len(centers) if isinstance(keys, str) else keys, dtype=object)
        if len(keys) != len(centers):
            return  # A mismatched capability never guesses a point's identity.
        valid = np.all(np.isfinite(centers), axis=1) & np.isfinite(radii) & (radii > 0)
        clip = clip_for(artist)
        if clip is not None:
            valid &= ((centers[:, 0] >= clip.left()) & (centers[:, 0] <= clip.right()) &
                      (centers[:, 1] >= clip.top()) & (centers[:, 1] <= clip.bottom()))
        for key in dict.fromkeys(keys[valid]):
            if not key:
                continue
            mask = valid & (keys == key)
            xy, radius = centers[mask], radii[mask]
            low, high = np.min(xy - radius[:, None], axis=0), np.max(xy + radius[:, None], axis=0)
            xy.setflags(write=False); radius.setflags(write=False)
            target = entry(key); target.points.append((xy, radius))
            target.bounds = target.bounds.united(QtCore.QRectF(float(low[0]), float(low[1]),
                max(.1, float(high[0] - low[0])), max(.1, float(high[1] - low[1]))))

    for artist in figure.findobj():
        key = getattr(artist, '_nanoscope_series_id', None)
        point_keys = getattr(artist, '_nanoscope_series_point_ids', None)
        path_keys = getattr(artist, '_nanoscope_series_path_ids', None)
        if not (key or point_keys is not None or path_keys is not None) or not artist.get_visible():
            continue
        if artist.axes is not None and not artist.axes.get_visible():
            continue
        if isinstance(artist, Line2D):
            if artist.get_linestyle() not in ('None', '', ' '):
                add_path(key, _qt_path(artist.get_path(), artist.get_transform(), height), artist)
            if artist.get_marker() not in (None, 'None', '', ' '):
                add_points(key, artist.get_transform().transform(artist.get_path().vertices),
                           [artist.get_markersize() ** 2], artist)
        elif isinstance(artist, PathCollection):
            add_points(point_keys if point_keys is not None else key,
                       artist.get_offset_transform().transform(artist.get_offsets()), artist.get_sizes(), artist)
        elif isinstance(artist, Patch):
            add_path(key, _qt_path(artist.get_path(), artist.get_transform(), height), artist,
                     filled=artist.get_fill() and artist.get_facecolor()[3] > 0)
        elif isinstance(artist, Collection):
            for index, path in enumerate(artist.get_paths()):
                current = key if path_keys is None else path_keys[index % len(path_keys)]
                # Collection line widths are arrays; use the conservative six-pixel hit band.
                converted = _qt_path(path, artist.get_transform(), height)
                stroker = QtGui.QPainterPathStroker(); stroker.setWidth(6.)
                hit = stroker.createStroke(converted)
                clip = clip_for(artist)
                if clip is not None:
                    boundary = QtGui.QPainterPath(); boundary.addRect(clip); hit = hit.intersected(boundary)
                if current and not hit.isEmpty():
                    target = entry(current); target.paths.append(hit)
                    target.bounds = target.bounds.united(hit.boundingRect())
    return result


class _SeriesOutline(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('matplotlibSeriesOutline')
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.hide()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.ColorRole.Highlight), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class MatplotlibSeriesSelection(QtCore.QObject):
    """Stable selection owner across frozen redraws; decoration never enters exports."""
    selected = QtCore.Signal(str)

    def __init__(self, view, parent):
        super().__init__(parent)
        self._view = weakref.ref(view)
        self._item = lambda: None
        self._key = None
        self.outline = _SeriesOutline(view.viewport())
        view.viewChanged.connect(self.reposition)
        view.horizontalScrollBar().valueChanged.connect(self.reposition)
        view.verticalScrollBar().valueChanged.connect(self.reposition)
        view.geometryChanged.connect(self.reposition)

    def set_item(self, item):
        old = self._item()
        if old is not None and _alive(old):
            old.seriesSelected.disconnect(self._picked)
        self._item = weakref.ref(item) if item is not None else lambda: None
        if item is not None:
            item.seriesSelected.connect(self._picked)
        self.reposition()

    @QtCore.Slot(str)
    def _picked(self, key):
        self.select(key)
        self.selected.emit(key)

    def select(self, key):
        self._key = key
        self.reposition()

    @QtCore.Slot()
    def reposition(self, *_args):
        view, item = self._view(), self._item()
        geometry = item.series_geometry().get(self._key) if item is not None and _alive(item) else None
        if view is None or not _alive(view) or geometry is None:
            self.outline.hide()
            return
        rect = item.deviceTransform(view.viewportTransform()).mapRect(geometry.bounds).toAlignedRect()
        self.outline.setGeometry(rect.adjusted(-2, -2, 2, 2).intersected(view.viewport().rect()))
        self.outline.setVisible(not self.outline.geometry().isEmpty())
        self.outline.raise_()
