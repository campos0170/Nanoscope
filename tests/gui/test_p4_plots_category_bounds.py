"""Category outlines follow retained point masks, without changing scientific data."""
from dataclasses import replace
import weakref

import nanoscope  # noqa: F401: initialize the existing Qt bootstrap before pyqtgraph
import numpy as np
import pyqtgraph as pg
from shiboken6 import delete

from nanoscope.core.figure_series import SeriesDescriptor, stable_series_id
from nanoscope.gui.figure_text import figure_text_for
from nanoscope.gui.format_toolbar import SelectionOutline
from nanoscope.gui.plots_series import _bounds_callback
from . import p1_hosting as H


def test_plots_category_outline_uses_visible_mask_and_preserves_native_points(qtbot):
    page = H.page_builders()['distributions'](); qtbot.addWidget(page)
    page.set_plot_type('Scatter')
    page._clear_plot()
    page.resize(1200, 900); page.show()
    labels = np.array([0, 0, 1, 1])
    item = pg.ScatterPlotItem(x=[0., 1., 9., 10.], y=[0., 1., 9., 10.],
        size=7, data=['row-a', 'row-b', 'row-c', 'row-d'])
    item._presentation_cluster_labels = labels.copy()
    item._presentation_cluster_source = 'source-a'
    item._presentation_role = 'category'
    page._plot.addItem(item); page._scatter_items = [item]
    page._legend_clusters(labels)
    page._plot.setRange(xRange=(-1., 11.), yRange=(-1., 11.), padding=0)
    page._apply_series_styles()
    qtbot.wait(10)  # let the shown viewport acquire its actual transform
    raw = tuple(value.copy() for value in item.getData())
    before_range = page._plot.getViewBox().viewRange()
    keys = {int(entry.identity_key.rsplit(':', 1)[1]): entry.series_id
            for entry in page.figure_series()}
    callback = page._plot._nanoscope_series_bounds
    first, second = callback(keys[0]), callback(keys[1])
    assert first is not None and second is not None
    assert first.right() < second.left()
    assert first.width() < item.sceneBoundingRect().width() / 2.
    for index in (0, 1):
        assert first.contains(item.mapToScene(item.points()[index].pos()))
    controller = figure_text_for(page._plot)
    outline = SelectionOutline(controller.view().viewport()); outline.attach(controller)
    outline.show_key(keys[0])
    expected = controller.view().mapFromScene(first).boundingRect().adjusted(-2, -2, 2, 2)
    assert outline.geometry() == expected.intersected(controller.view().viewport().rect())

    # One invisible point no longer enlarges its category; hiding the category
    # removes its outline without removing its records or hiding another category.
    item.points()[1].setSize(0.)
    assert callback(keys[0]).width() < first.width()
    page._style = replace(page._style, series_styles={keys[0]: {'visible': False}})
    page._apply_series_styles()
    assert page._plot._nanoscope_series_bounds(keys[0]) is None
    assert page._plot._nanoscope_series_bounds(keys[1]) is not None
    outline.reposition()
    assert not outline.isVisible()
    assert page._plot.getViewBox().viewRange() == before_range
    for actual, original in zip(item.getData(), raw):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(item._presentation_cluster_labels, labels)
    assert [point.data() for point in item.points()] == ['row-a', 'row-b', 'row-c', 'row-d']


def test_category_bounds_callback_keeps_only_weak_items_and_scalar_identities(qtbot):
    widget = pg.PlotWidget(); qtbot.addWidget(widget)
    widget.resize(500, 400); widget.show()
    item = pg.ScatterPlotItem(x=[0., 1.], y=[0., 1.], size=7)
    item._presentation_cluster_labels = np.array([3, 4])
    widget.addItem(item)
    proxy = pg.ScatterPlotItem(x=[999.], y=[999.], size=7)
    descriptor = SeriesDescriptor(stable_series_id('distributions.Scatter', 'data', 's:cluster:3'),
        'Cluster 3', identity_kind='category', identity_key='s:cluster:3')
    mask = item._presentation_cluster_labels == 3
    callback = _bounds_callback([(descriptor, [(item, mask), (proxy, None)])], [item])
    # A callback kept by a retired outline cannot keep old data, page or items
    # alive. The legend proxy is intentionally absent from its selected bounds.
    references = callback.args[0][descriptor.series_id]
    assert len(references) == 1
    reference, category = references[0]
    assert isinstance(reference, weakref.ReferenceType) and reference() is item
    assert type(category) is int and category == 3
    assert callback.func.__closure__ is None and not callback.keywords
    qtbot.wait(10)
    assert callback(descriptor.series_id) is not None
    widget.removeItem(item)
    delete(item)
    assert callback(descriptor.series_id) is None
    assert callback('unknown-series') is None
    delete(proxy)
