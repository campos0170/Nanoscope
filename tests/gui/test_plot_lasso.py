"""Population lasso reads the committed picture and changes selection only."""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401 -- Qt bootstrap before PySide
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui.plot_lasso import RowIdentityIndex, toggled_rows, unique_rows
from nanoscope.gui.plots_panel import PlotsWindow
from .test_plots_organization_and_selection import _source


@pytest.fixture
def window(qapp, qtbot):
    frame = pd.DataFrame({
        "event_id": [17, 23, 41, 59, 71, 83],
        "dwell_s": [1., 10., 100., -1., np.nan, 1000.],
        "depth_pA": [10., 100., 1000., 100., np.inf, 1000.],
        "accepted": [True, True, True, True, True, False],
    }, index=[0., 0., np.nan, 0., 0., 0.])
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source("A", frame)])
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    w.set_scatter_log(True, True)
    w.resize(1150, 750)
    w.show()
    qapp.processEvents()
    w._plot.setRange(xRange=(-.3, 2.3), yRange=(.7, 3.3), padding=0)
    qapp.processEvents()
    return w


def _rect_polygon(rect):
    return QtGui.QPolygonF([rect.topLeft(), rect.topRight(), rect.bottomRight(),
                           rect.bottomLeft(), rect.topLeft()])


def _around(w, key="A", index=0, margin=12):
    item = w._event_point_items[key][0]
    position = item.mapToScene(item.points()[index].pos())
    return _rect_polygon(QtCore.QRectF(position.x()-margin, position.y()-margin,
                                      2*margin, 2*margin))


def _gesture(w, qtbot, polygon, modifiers=QtCore.Qt.NoModifier):
    points = [w._plot_widget.mapFromScene(point) for point in polygon]
    qtbot.mousePress(w._plot_widget.viewport(), QtCore.Qt.LeftButton, modifiers, points[0])
    for point in points[1:]:
        qtbot.mouseMove(w._plot_widget.viewport(), point)
    qtbot.mouseRelease(w._plot_widget.viewport(), QtCore.Qt.LeftButton, modifiers, points[-1])


def _forbid_science(w, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Selection attempted to rerender, collect sources, or run analysis")
    for name in ("_redraw", "_drawn_frames", "_draw_histogram", "_draw_scatter",
                 "_draw_kde", "_draw_pca", "_overlay_fit", "_overlay_scatter_fit"):
        monkeypatch.setattr(w, name, fail)


def test_lasso_real_mouse_gesture_preserves_picture_and_scientific_state(window, qtbot, monkeypatch):
    w = window
    assert [point.data() for point in w._scatter_item.points()] == [17, 23, 41]
    assert w._lasso_action.isVisible() and w._lasso_action.isEnabled()
    assert "Pan/zoom" in w._selection_mode_label.text()
    before_frame = w._sources["A"].df.copy(deep=True)
    before_data = w._plot_data().copy(deep=True)
    before_config = w.figure_applied_config()
    before_sources = w.figure_source_recipes()
    before_range = np.asarray(w._plot.viewRange()).copy()
    before_items = tuple(w._plot.items)
    before_fit, before_pca, before_cluster = w.fit_result(), w._pca_result, w._cluster_result
    changed = QtTest.QSignalSpy(w.workspaceChanged)
    selected = QtTest.QSignalSpy(w.eventRowsSelected)
    _forbid_science(w, monkeypatch)

    w._lasso_action.trigger()
    assert w._lasso.enabled and "Lasso selection" in w._selection_mode_label.text()
    _gesture(w, qtbot, _around(w, index=1))

    assert selected.count() == 1
    assert selected.at(0) == ["A", (23,)]
    assert w._selected_event_rows == {"A": [23]}
    assert changed.count() == 0
    assert not w._lasso.drawing and not w._lasso.overlay.isVisible()
    assert tuple(w._plot.items) == before_items
    assert w.fit_result() is before_fit and w._pca_result is before_pca
    assert w._cluster_result is before_cluster
    assert w.figure_applied_config() == before_config
    assert w.figure_source_recipes() == before_sources
    np.testing.assert_array_equal(w._plot.viewRange(), before_range)
    pd.testing.assert_frame_equal(w._sources["A"].df, before_frame, check_exact=True)
    pd.testing.assert_frame_equal(w._plot_data(), before_data, check_exact=True)
    monkeypatch.undo()  # Widget teardown may legitimately read its source frames.


@pytest.mark.parametrize("modifier", [QtCore.Qt.ControlModifier, QtCore.Qt.MetaModifier,
                                     QtCore.Qt.ShiftModifier])
def test_lasso_modifiers_toggle_exact_ids_once(window, qtbot, modifier):
    w = window
    w._lasso_action.trigger()
    _gesture(w, qtbot, _around(w, index=0))
    _gesture(w, qtbot, _around(w, index=1), modifier)
    assert w._selected_event_rows == {"A": [17, 23]}
    _gesture(w, qtbot, _around(w, index=0), modifier)
    assert w._selected_event_rows == {"A": [23]}


def test_lasso_nonfinite_log_rejection_clipping_and_hidden_items(window, qapp):
    w = window
    entire = _rect_polygon(w._plot.getViewBox().sceneBoundingRect())
    assert w._lasso_hits(entire) == {"A": [17, 23, 41]}
    assert w._lasso_hits(_around(w, index=1)) == {"A": [23]}
    w._plot.setRange(xRange=(-.3, .3), yRange=(.7, 1.3), padding=0)
    qapp.processEvents()
    assert w._lasso_hits(entire) == {"A": [17]}
    w._scatter_item.hide()
    assert w._lasso_hits(entire) == {}


def test_lasso_pca_uses_committed_kept_ids_not_frame_positions(window, qapp, monkeypatch):
    w = window
    w.set_feature_selection(["dwell_s", "depth_pA"])
    result = w.run_pca()
    w._plot.autoRange()
    qapp.processEvents()
    qapp.processEvents()  # Settle child transforms after the new title resizes the view.
    assert result is not None
    assert [point.data() for point in w._scatter_item.points()] == [17, 23, 41, 59]
    polygon = _around(w, index=3)
    before = result.scores.copy()
    _forbid_science(w, monkeypatch)
    try:
        w._on_lasso_completed(polygon)
        assert w._selected_event_rows == {"A": [59]}
        assert w._pca_result is result
        np.testing.assert_array_equal(result.scores, before)
    finally:
        monkeypatch.undo()


def test_empty_lasso_clears_plain_selection_but_modifiers_leave_it(window):
    w = window
    w._on_lasso_completed(_around(w))
    received = []
    w.eventRowsSelected.connect(lambda key, rows: received.append((key, rows)))
    rect = w._plot.getViewBox().sceneBoundingRect()
    empty = _rect_polygon(QtCore.QRectF(rect.left()+20, rect.top()+20, 10, 10))
    w._on_lasso_completed(empty, QtCore.Qt.ShiftModifier)
    assert w._selected_event_rows == {"A": [17]} and received == []
    w._on_lasso_completed(empty)
    assert w._selected_event_rows == {} and received == [("A", ())]


def test_multi_source_lasso_asks_for_one_source_and_cancel_keeps_selection(window, monkeypatch, qapp):
    w = window
    first = w._sources["A"]
    second = _source("B", first.df.copy(deep=True))
    w.set_data([first, second])
    w.set_sources(["A", "B"])
    w._plot.autoRange()
    qapp.processEvents()
    qapp.processEvents()
    polygon = _around(w)
    assert w._lasso_hits(polygon) == {"A": [17], "B": [17]}
    w._apply_event_row_selection("A", [17])
    dialogs = []

    def choose(*args):
        dialogs.append(args)
        return args[3][1], True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", choose)
    w._on_lasso_completed(polygon, QtCore.Qt.ControlModifier)
    assert w._selected_event_rows == {"B": [17]}
    assert "multiple sources" in dialogs[0][2]
    assert all("1 events" in label for label in dialogs[0][3])
    assert w._event_point_items["A"][0].points()[0].pen().style() == QtCore.Qt.NoPen
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *a: ("", False))
    w._on_lasso_completed(polygon)
    assert w._selected_event_rows == {"B": [17]}


def test_lasso_source_chooser_replot_generation_aborts_selection(window, monkeypatch):
    w = window
    def choose(hits):
        w._lasso_generation += 1  # A new applied draw arrived during the modal event loop.
        return "A"
    monkeypatch.setattr(w, "_choose_lasso_source", choose)
    w._on_lasso_completed(_around(w))
    assert not w._selected_event_rows
    assert "plot changed" in w.statusBar().currentMessage()


@pytest.mark.parametrize("cancel", ["escape", "right"])
def test_cancel_lasso_restores_pan_zoom_without_selection(window, qtbot, cancel):
    w = window
    w._apply_event_row_selection("A", [17])
    signals = QtTest.QSignalSpy(w.eventRowsSelected)
    before = np.asarray(w._plot.viewRange()).copy()
    w._lasso_action.trigger()
    first, second = [w._plot_widget.mapFromScene(p) for p in list(_around(w, index=1))[:2]]
    qtbot.mousePress(w._plot_widget.viewport(), QtCore.Qt.LeftButton, pos=first)
    qtbot.mouseMove(w._plot_widget.viewport(), second)
    assert w._lasso.drawing and w._lasso.overlay.isVisible()
    assert w._lasso.overlay not in w._plot.items
    if cancel == "escape":
        qtbot.keyPress(w._plot_widget, QtCore.Qt.Key_Escape)
    else:
        qtbot.mouseClick(w._plot_widget.viewport(), QtCore.Qt.RightButton, pos=second)
        context = QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, second)
        assert w._lasso.eventFilter(w._plot_widget.viewport(), context)
    qtbot.mouseRelease(w._plot_widget.viewport(), QtCore.Qt.LeftButton, pos=second)
    assert not w._lasso.enabled and not w._lasso_action.isChecked()
    assert "Pan/zoom" in w._selection_mode_label.text()
    assert not w._lasso.drawing and not w._lasso.overlay.isVisible()
    assert w._selected_event_rows == {"A": [17]} and signals.count() == 0
    np.testing.assert_array_equal(w._plot.viewRange(), before)


@pytest.mark.parametrize("change", ["resize", "range", "redraw"])
def test_changed_canvas_cancels_incomplete_gesture(window, qtbot, qapp, change):
    w = window
    w._lasso_action.trigger()
    position = w._plot_widget.mapFromScene(_around(w)[0])
    qtbot.mousePress(w._plot_widget.viewport(), QtCore.Qt.LeftButton, pos=position)
    assert w._lasso.drawing
    if change == "resize":
        w.resize(w.width()+40, w.height()+20)
    elif change == "range":
        w._plot.setRange(xRange=(-1, 3), yRange=(0, 4), padding=0)
    else:
        w._redraw()
    qapp.processEvents()
    assert not w._lasso.drawing and not w._lasso.overlay.isVisible()
    qtbot.mouseRelease(w._plot_widget.viewport(), QtCore.Qt.LeftButton, pos=position)
    assert not w._selected_event_rows


def test_non_event_plot_disables_lasso_and_ordinary_pan_remains_available(window, qtbot):
    w = window
    w._lasso_action.trigger()
    w.set_plot_type("Histogram")
    assert not w._lasso.enabled and not w._lasso_action.isEnabled()
    rect = w._lasso._data_rect()
    first = rect.center().toPoint()
    second = first + QtCore.QPoint(40, 30)
    before = np.asarray(w._plot.viewRange()).copy()
    qtbot.mousePress(w._plot_widget.viewport(), QtCore.Qt.LeftButton, pos=first)
    qtbot.mouseMove(w._plot_widget.viewport(), second)
    qtbot.mouseRelease(w._plot_widget.viewport(), QtCore.Qt.LeftButton, pos=second)
    assert not w._lasso.drawing
    assert not np.array_equal(w._plot.viewRange(), before)


def test_identity_index_is_linear_for_ordinary_event_ids_and_handles_unhashable_labels():
    comparisons = []
    def equal(left, right):
        comparisons.append(None)
        return PlotsWindow._same_row_identity(left, right)
    values = list(range(3000))
    index = RowIdentityIndex(values, equal)
    assert all(index.contains(value) for value in values)
    assert len(comparisons) == len(values)
    assert unique_rows([[1, 2], [1, 2], [3, 4]], equal) == [[1, 2], [3, 4]]
    assert toggled_rows([1, 2, 2], [2, 3, 3], equal) == [1, 3]


@pytest.mark.parametrize("size", [1, 2, 6, 15])
def test_selection_preserves_enabled_autorange_and_source_brushes(window, qapp, size):
    w = window
    w._style = replace(w._style, point_size=size)
    w._paint_event_point_selection()
    w._plot.enableAutoRange()
    qapp.processEvents()
    before = np.asarray(w._plot.viewRange()).copy()
    auto = list(w._plot.getViewBox().state["autoRange"])
    brushes = [point.brush().color().rgba() for point in w._scatter_item.points()]
    for rows in ([17, 23, 41], [], [23], []):
        w._apply_event_row_selection("A", rows)
        qapp.processEvents()
        np.testing.assert_array_equal(w._plot.viewRange(), before)
        assert w._plot.getViewBox().state["autoRange"] == auto
        assert [point.brush().color().rgba() for point in w._scatter_item.points()] == brushes


def test_hidden_spot_is_not_selected(window, qapp):
    w = window
    polygon = _around(w)
    w._scatter_item.points()[0].setVisible(False)
    qapp.processEvents()
    assert w._lasso_hits(polygon) == {}


def test_actual_lasso_routes_stable_ids_to_event_table_and_trace(qapp, qtbot, tmp_path):
    from nanoscope.gui.main_window import MainWindow
    from .test_analysis_explorer import _install_events
    w = MainWindow(recovery_enabled=False)
    qtbot.addWidget(w)
    doc_id, events = _install_events(w, tmp_path)
    before = events.df.copy(deep=True)
    w._open_plots()
    plot = w._plots_window
    source = next(item for item in plot.sources()
                  if item.doc_id == doc_id and item.module_id == "detect.schmitt")
    plot.set_plot_type("Scatter")
    plot.set_scatter_columns("dwell_s", "depth_pA")
    w.resize(1300, 850)
    w.show()
    plot._plot.autoRange()
    qapp.processEvents()
    plot._lasso_action.trigger()
    _gesture(plot, qtbot, _around(plot, key=source.key, index=1))
    assert plot._selected_event_rows == {source.key: [11]}
    assert w._analysis_panel._table.selected_event_ids() == [11]
    assert w._analysis_panel._selected_id == 11
    pd.testing.assert_frame_equal(events.df, before, check_exact=True)
    w._dirty = False
