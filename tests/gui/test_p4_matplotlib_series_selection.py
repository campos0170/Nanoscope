"""Real artist hits select stable Format rows; viewport outlines never enter exports."""
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtTest

from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
from nanoscope.gui.format_inspector import ElementRef, FormatPane, SelectionHub
from nanoscope.gui.study_figure_window import StudyFigureWindow
from .test_comparison_figures_panel import source, forbid_prepare


def click_series(view, item, key):
    geometry = item.series_geometry()[key]
    centers = geometry.points[0][0]
    center = centers[len(centers) // 2]
    point = QtCore.QPointF(float(center[0]), float(center[1]))
    assert item.series_at(point) == key
    assert item._hit(point) is None
    QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton,
                           pos=view.mapFromScene(item.mapToScene(point)))


def test_comparison_format_row_and_canvas_hit_share_selection_without_changing_export(qtbot, monkeypatch):
    page = ComparisonFiguresWindow(); qtbot.addWidget(page)
    page.resize(1100, 800); page.show()
    page.set_sources([source('a'), source('b', 'B')], selected_keys=['a', 'b'])
    assert page.analyze(); page.fit_canvas()
    forbid_prepare(monkeypatch)
    spec = page.inspector_adapter().format
    hub = SelectionHub(page)
    pane = FormatPane(spec, hub); qtbot.addWidget(pane); pane.show()
    first, second = page.series_descriptors()
    item = page._figure_item
    exported = item.native_figure_svg()
    tables = deepcopy(page._tables)
    mark = page._appearance_widget.history_mark()
    pane.select(ElementRef('', first.series_id))
    assert page._series_selection.outline.isVisible()
    assert page._series_selection.outline.parentWidget() is page._view.viewport()
    assert item.native_figure_svg() == exported
    click_series(page._view, item, second.series_id)
    qtbot.waitUntil(lambda: hub.current() == ElementRef('', second.series_id))
    assert page._series_selection._key == second.series_id
    assert item.native_figure_svg() == exported and page._appearance_widget.history_mark() == mark
    # Real separate clicks have an event-loop boundary. The first claim's
    # zero-delay coalescer must finish before delivering the second QTest event.
    qtbot.waitUntil(lambda: not pane._canvas_claim_pending)
    viewport = page._view.viewport().rect()
    blank_pixel = None
    for y in range(8, viewport.height() - 8, 13):
        for x in range(8, viewport.width() - 8, 13):
            pixel = QtCore.QPoint(x, y)
            local = item.mapFromScene(page._view.mapToScene(pixel))
            if (page._view.itemAt(pixel) is item and item._hit(local) is None
                    and item.series_at(local) is None):
                blank_pixel = pixel; break
        if blank_pixel is not None:
            break
    assert blank_pixel is not None and viewport.contains(blank_pixel)
    QtTest.QTest.mouseClick(page._view.viewport(), QtCore.Qt.LeftButton, pos=blank_pixel)
    qtbot.waitUntil(lambda: hub.current() is None)
    assert not page._series_selection.outline.isVisible()
    for key, frame in tables.items():
        pd.testing.assert_frame_equal(page._tables[key], frame, check_exact=True)
    pane.retire()


def test_mixed_colour_retained_collection_picks_exact_original_group_after_recolour(qtbot):
    figure = Figure(figsize=(5, 3)); FigureCanvasAgg(figure)
    axis = figure.subplots()
    points = axis.scatter([1., 2., 3., 4.], [2., 3., 1., 4.],
                          c=['#123456', '#abcdef', '#123456', '#abcdef'], s=36)
    axis.set_xlim(0., 5.); axis.set_ylim(0., 5.)
    editor = StudyFigureWindow(figure, pd.DataFrame({'raw': [1., 2.]}), {},
                               name='Mixed', provenance={}, owner_kind='report')
    qtbot.addWidget(editor); editor.resize(1200, 800); editor.show(); editor.fit()
    target = next(d for d in editor._styler.series_descriptors() if d.identity_key == '#abcdef')
    before = points.get_offsets().copy()
    exported = editor._item.native_figure_svg()
    history = editor.history_mark()
    click_series(editor._view, editor._item, target.series_id)
    qtbot.waitUntil(lambda: editor._selection_hub.current() == ElementRef('', target.series_id))
    assert editor._series_selection.outline.isVisible()
    assert editor._item.native_figure_svg() == exported and editor.history_mark() == history
    editor._format_pane.write_series(target.series_id, {'color': '#ff0088'})
    updated = next(d for d in editor._styler.series_descriptors() if d.series_id == target.series_id)
    assert updated.extra['effective_style']['color'] == '#ff0088'
    assert editor._series_selection._key == target.series_id
    click_series(editor._view, editor._item, target.series_id)
    assert editor._selection_hub.current() == ElementRef('', target.series_id)
    np.testing.assert_array_equal(points.get_offsets(), before)
