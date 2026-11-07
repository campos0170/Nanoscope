"""Presentation-only point masks; never replace coordinates or class/row identities."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui
from shiboken6 import isValid


def scatter_mask_scene_bounds(item, mask=None):
    """Read the visible marker footprints of one mask in scene coordinates.

    No coordinates, membership arrays or items are copied or retained. Pixel-sized
    markers are measured in the existing viewport, including their current size
    overrides; zero-size and clipped-out points do not extend the outline.
    """
    if item is None or not isValid(item) or not item.isVisible():
        return None
    scene = item.scene()
    views = scene.views() if scene is not None else ()
    if not views:
        return None
    view = views[0]
    to_device = item.deviceTransform(view.viewportTransform())
    to_scene, invertible = view.viewportTransform().inverted()
    if not invertible:
        return None
    view_box = item.getViewBox()
    clip = (view.viewportTransform().mapRect(view_box.sceneBoundingRect())
            if view_box is not None else None)
    points = item.points()
    if mask is None:
        chosen = range(len(points))
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or mask.size != len(points):
            return None  # A retiring/replaced draw cannot supply a matching mask.
        chosen = np.flatnonzero(mask)
    bounds = None
    for index in chosen:
        point = points[int(index)]
        radius = float(point.size()) / 2.
        if radius <= 0:
            continue
        if item.opts.get('pxMode', True):
            center = to_device.map(point.pos())
            footprint = QtCore.QRectF(center.x() - radius, center.y() - radius,
                                     radius * 2, radius * 2)
        else:
            center = point.pos()
            footprint = to_device.mapRect(QtCore.QRectF(
                center.x() - radius, center.y() - radius, radius * 2, radius * 2))
        if clip is not None:
            footprint = footprint.intersected(clip)
        if not footprint.isEmpty():
            bounds = footprint if bounds is None else bounds.united(footprint)
    return to_scene.mapRect(bounds) if bounds is not None else None


def reset_point_sizes(item, size):
    """Capture the current global footprint before per-category overrides."""
    item._presentation_series_sizes = np.full(len(item.points()), float(size), dtype=float)
    return item._presentation_series_sizes


def apply_point_style(item, mask, *, color=None, size=None, visible=True):
    """Style selected existing SpotItems, preserving coordinate and metadata arrays.

    A hidden category has a zero footprint and no selection ring. Its measurements
    stay available to raw export and its captured membership never depends on ink.
    Native selection handlers can read ``_presentation_series_sizes`` when painting
    rings so they do not overwrite explicit marker sizes or reveal hidden points.
    """
    points = item.points()
    if mask is None:
        chosen = range(len(points))
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or mask.size != len(points):
            raise ValueError('Series point mask does not match the retained markers')
        chosen = np.flatnonzero(mask)
    sizes = getattr(item, '_presentation_series_sizes', None)
    if sizes is None or len(sizes) != len(points):
        sizes = np.asarray([point.size() for point in points], dtype=float)
    for j in chosen:
        point = points[j]
        if color is not None:
            brush = pg.mkBrush(point.brush())
            tint = QtGui.QColor(color)
            tint.setAlpha(brush.color().alpha())
            point.setBrush(pg.mkBrush(tint))
        if size is not None:
            sizes[j] = float(size)
        if not visible:
            sizes[j] = 0.
            point.setPen(pg.mkPen(None))
        point.setSize(float(sizes[j]))
    item._presentation_series_sizes = sizes
    return sizes
