"""Figure-local series controls for the custom Distributions canvas.

Items carry only plain identities. Rendering applies presentation to existing artists;
class membership, coordinates, and native figure history remain owned by PlotsWindow.
"""
from dataclasses import replace
from functools import partial
import weakref
import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui
from shiboken6 import isValid

from ..core.figure_series import SeriesDescriptor, stable_series_id
from ..core.figure_style import resolve_series_color, points_to_logical_pixels
from .categorical import category_colour
from .series_points import reset_point_sizes, apply_point_style, scatter_mask_scene_bounds


def tag_series(item, view, role, key, label, *, identity_kind='source', colour=None):
    capabilities = ['color', 'visible']
    if isinstance(item, (pg.PlotDataItem, pg.IsocurveItem)):
        capabilities.append('line_width_pt')
    if isinstance(item, pg.ScatterPlotItem) or (
            isinstance(item, pg.PlotDataItem) and item.opts.get('symbol') is not None):
        capabilities.append('marker_size_pt')
    descriptor = SeriesDescriptor(
        stable_series_id('distributions.' + view, role, key), str(label), role,
        identity_kind, str(key) if identity_kind != 'none' else '', colour,
        tuple(capabilities))
    item._figure_series_meta = descriptor.to_dict()
    from .figure_series import tag_pg_series
    tag_pg_series(item, series_id=descriptor.series_id, role=role, identity=key,
        identity_kind=identity_kind, label=label, color=colour, capabilities=capabilities)
    if isinstance(item, pg.ScatterPlotItem):
        # Point clicks already own exact event inspection and lasso selection. The
        # Format series list supplies selection without a competing click callback.
        item._nanoscope_series_selection_wired = True



def _groups(page):
    """Combine real artists and legend proxies by their captured semantic identity."""
    groups = {}
    names = page._committed_legend_names()
    objects = list(page._plot.items) + [row[0] for row in page._presentation_legend.values()]
    for item in dict.fromkeys(objects):
        labels = getattr(item, '_presentation_cluster_labels', None)
        if labels is not None:
            source = getattr(item, '_presentation_cluster_source', '')
            view = (getattr(page, '_figure_applied_config', None) or page.config()).plot_type
            for cid in sorted(set(int(value) for value in labels)):
                key = f'{source}:cluster:{cid}'
                descriptor = SeriesDescriptor(
                    stable_series_id('distributions.' + view, 'data', key),
                    'Dropped events' if cid < 0 else f'Cluster {cid}', 'data', 'category', key,
                    QtGui.QColor(category_colour(cid)).name(), ('color', 'marker_size_pt', 'visible'))
                groups.setdefault(descriptor.series_id, (descriptor, []))[1].append((item, labels == cid))
            continue
        raw = getattr(item, '_figure_series_meta', None)
        if not raw:
            continue
        descriptor = SeriesDescriptor.from_dict(raw)
        if descriptor.identity_kind == 'source' and descriptor.identity_key in names:
            label = names[descriptor.identity_key]
            suffix = ' fit' if descriptor.role == 'fit' else ' KDE' if descriptor.role == 'kde' else ''
            descriptor = replace(descriptor, label=label + suffix)
        groups.setdefault(descriptor.series_id, (descriptor, []))[1].append((item, None))
    return list(groups.values())


def _colour(page, descriptor, index):
    style = page._style.series_styles.get(descriptor.series_id, {})
    return resolve_series_color(page._style.style_binding,
        role='fit' if descriptor.role == 'fit' else 'data', index=index,
        native_color=page._style.fit_color if descriptor.role == 'fit' else page._style.data_color,
        identity_color=descriptor.color, explicit_color=style.get('color'))


def descriptors(page):
    from .figure_series import effective_pg_item_style
    from ..core.figure_style import logical_pixels_to_points
    output = []
    for i, (descriptor, targets) in enumerate(_groups(page)):
        effective = effective_pg_item_style(targets[0][0]) if targets else {}
        effective['color'] = _colour(page, descriptor, i)
        if any(mask is not None for _item, mask in targets):
            own = page._style.series_styles.get(descriptor.series_id, {})
            effective['marker_size_pt'] = own.get('marker_size_pt', logical_pixels_to_points(page._style.point_size))
        output.append(replace(descriptor, color=effective['color'],
            extra={**dict(descriptor.extra), 'effective_style':effective}))
    return output


def _tint(value, color):
    original = pg.mkBrush(value)
    tint = QtGui.QColor(color)
    tint.setAlpha(original.color().alpha())
    return pg.mkBrush(tint)


def _series_scene_bounds(targets, key):
    """Resolve current paint bounds without retaining a page, draw or mask array."""
    bounds = None
    for reference, category in targets.get(key, ()):
        item = reference()
        if item is None or not isValid(item) or not item.isVisible() or item.scene() is None:
            continue
        if isinstance(item, pg.ScatterPlotItem):
            labels = getattr(item, '_presentation_cluster_labels', None)
            if category is not None and labels is None:
                continue
            rect = scatter_mask_scene_bounds(item, None if category is None else labels == category)
        else:
            rect = item.sceneBoundingRect()
        if rect is not None and not rect.isEmpty():
            bounds = rect if bounds is None else bounds.united(rect)
    return bounds


def _bounds_callback(grouped, plot_items):
    # Legend proxies can share the series identity but are not painted data. The
    # callback keeps scalar category IDs and weak refs only, not the temporary
    # boolean masks created while applying styles or the owning page/plot.
    real_items = set(plot_items)
    targets = {}
    for descriptor, items in grouped:
        refs = []
        for item, mask in items:
            if item not in real_items:
                continue
            category = (int(descriptor.identity_key.rsplit(':cluster:', 1)[1])
                        if mask is not None else None)
            refs.append((weakref.ref(item), category))
        targets[descriptor.series_id] = tuple(refs)
    return partial(_series_scene_bounds, targets)


def apply_series_styles(page):
    # Reset point-size memo to the current global style before per-category overrides.
    # The selection-ring pass then reads this memo instead of erasing an explicit size.
    for item in page._scatter_items:
        reset_point_sizes(item, page._style.point_size)
    grouped = _groups(page)
    for index, (descriptor, targets) in enumerate(grouped):
        style = page._style.series_styles.get(descriptor.series_id, {})
        color = _colour(page, descriptor, index)
        width = points_to_logical_pixels(style['line_width_pt']) if 'line_width_pt' in style else None
        marker = points_to_logical_pixels(style['marker_size_pt']) if 'marker_size_pt' in style else page._style.point_size
        visible = style.get('visible', True)
        for item, mask in targets:
            if mask is None:
                item.setVisible(visible and (not getattr(item, '_presentation_band', False)
                                             or page._style.show_ci_band))
            if isinstance(item, pg.ScatterPlotItem):
                apply_point_style(item, mask, color=color, size=marker, visible=visible)
                if not len(item.points()):
                    item.setBrush(pg.mkBrush(color))
                    item.setSize(marker)
            elif isinstance(item, (pg.PlotDataItem, pg.IsocurveItem)):
                existing = item.opts.get('pen') if isinstance(item, pg.PlotDataItem) else item.pen
                if existing is not None:
                    pen = pg.mkPen(existing)
                    pen.setColor(QtGui.QColor(color))
                    pen.setWidthF(width if width is not None else
                        page._style.line_width + getattr(item, '_presentation_width_extra', 0))
                    item.setPen(pen)
                if isinstance(item, pg.PlotDataItem):
                    brush = item.opts.get('fillBrush')
                    if brush is not None:
                        item.setFillBrush(_tint(brush, color))
                    if item.opts.get('symbol') is not None:
                        item.setSymbolBrush(pg.mkBrush(color))
                        item.setSymbolSize(marker)
            elif isinstance(item, pg.FillBetweenItem):
                item.setBrush(_tint(item.brush() if callable(item.brush) else item.brush, color))

    # Publish curve hit targets through the common Format selection seam. Class
    # masks stay with the native point-selection owner and remain selectable in the
    # series list. This does not add a second geometry or estimator path.
    from .figure_series import apply_pg_series_styles, tag_pg_series
    ordinary = []
    for descriptor, targets in grouped:
        for item, mask in targets:
            if mask is not None or getattr(item, '_presentation_band', False):
                continue
            tag_pg_series(item, series_id=descriptor.series_id, role=descriptor.role,
                identity=descriptor.identity_key, identity_kind=descriptor.identity_kind,
                label=descriptor.label, color=descriptor.color, capabilities=descriptor.capabilities)
            if isinstance(item, pg.ScatterPlotItem):
                item._nanoscope_series_selection_wired = True
            ordinary.append(item)
    apply_pg_series_styles(page._plot, page._appearance_widget.state(), data_items=ordinary,
                           view='distributions')
    # The original point signal still owns exact event inspection and selection.
    # This additional presentation listener selects the matching series in Format,
    # including a class mask, without consuming the event or changing its row IDs.
    for item in page._scatter_items:
        if getattr(item, '_nanoscope_points_format_wired', False):continue
        owner = weakref.ref(page); sink = weakref.ref(page._plot._nanoscope_series_selection)
        def picked(clicked, points, *_args, owner=owner, sink=sink):
            host, emitter = owner(), sink()
            if host is None or emitter is None or not points:return
            index = points[0].index()
            for descriptor, targets in _groups(host):
                if any(target is clicked and (mask is None or 0 <= index < len(mask) and mask[index])
                       for target, mask in targets):
                    emitter.selected.emit(descriptor.series_id);return
        item.sigClicked.connect(picked)
        item._nanoscope_points_format_wired = True
    page._plot._nanoscope_series_targets = {
        descriptor.series_id: (replace(descriptor, color=_colour(page, descriptor, i)),
                               [weakref.ref(item) for item, _mask in targets])
        for i, (descriptor, targets) in enumerate(grouped)}
    page._plot._nanoscope_series_bounds = _bounds_callback(grouped, page._plot.items)
