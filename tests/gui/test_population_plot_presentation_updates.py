"""Pure text/canvas edits retain committed population-plot results and zoom."""
from __future__ import annotations

import os
from unittest.mock import Mock

import nanoscope  # noqa: F401 -- initialize the Qt plugin path first
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtTest

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

from .test_plots_panel import _settle, _two_blob_events


@pytest.fixture
def window(qtbot):
    window = PlotsWindow(Session())
    qtbot.addWidget(window)
    window.set_data({"Synthetic events": _two_blob_events()})
    window.show()
    qtbot.wait(10)
    return window


def _watch_render_and_analysis(window, monkeypatch):
    watchers = []
    for name in ("_redraw", "_drawn_frames", "_draw_histogram", "_draw_scatter",
                 "_draw_kde", "_draw_pca", "_overlay_fit", "_overlay_scatter_fit"):
        watcher = Mock(wraps=getattr(window, name))
        monkeypatch.setattr(window, name, watcher)
        watchers.append(watcher)
    return watchers


def _exercise_text_and_canvas(window):
    window.set_title("Publication figure")
    window._x_label_edit.setText("Horizontal axis")
    window._y_label_edit.setText("Vertical axis")
    window.set_font_size(13)
    window.set_font_family("Helvetica")
    window._title_size.setValue(16)
    window._axis_label_size.setValue(14)
    window._tick_size.setValue(11)
    window._legend_font_size.setValue(9)
    window._on_style_title_bold(True)
    window._on_style_title_italic(True)
    window._grid_check.setChecked(False)
    window.set_background("light")
    window._history_commit()


@pytest.mark.parametrize("kind", ["Histogram", "Scatter", "KDE", "PCA"])
def test_text_and_canvas_keep_committed_items_numerics_and_zoom(window, qtbot, monkeypatch, kind):
    if kind == "PCA":
        window.set_feature_selection(["dwell_s", "amplitude", "snr"])
        window.run_pca()
    else:
        window.set_plot_type(kind)
        if kind == "Histogram":
            window.set_histogram_column("dwell_s")
        elif kind == "Scatter":
            window.set_scatter_columns("dwell_s", "amplitude")
        else:
            window.set_kde_columns("dwell_s", "amplitude")
    qtbot.wait(10)
    items = tuple(window._plot.items)
    before = window._plot_data()
    pca = window._pca_result
    cluster = window._cluster_result
    source_frames = {key: source.df.copy(deep=True) for key, source in window._sources.items()}
    axis_scale = tuple(window._plot.getAxis(side).scale for side in ("bottom", "left"))
    window._plot.setRange(xRange=(0.00015, 0.00045), yRange=(15, 45), padding=0)
    qtbot.wait(10)
    view = np.asarray(window._plot.viewRange()).copy()
    auto = list(window._plot.vb.state["autoRange"])
    watchers = _watch_render_and_analysis(window, monkeypatch)
    changed = QtTest.QSignalSpy(window.workspaceChanged)

    _exercise_text_and_canvas(window)
    qtbot.wait(10)

    for watcher in watchers:
        watcher.assert_not_called()
    assert changed.count() > 0
    assert tuple(window._plot.items) == items
    np.testing.assert_array_equal(window._plot.viewRange(), view)
    assert window._plot.vb.state["autoRange"] == auto
    assert tuple(window._plot.getAxis(side).scale for side in ("bottom", "left")) == axis_scale
    assert window._pca_result is pca and window._cluster_result is cluster
    assert window._plot.titleLabel.text.endswith("Publication figure")
    assert window._plot.getAxis("bottom").labelText == "Horizontal axis"
    assert window._plot.getAxis("left").labelText == "Vertical axis"
    assert window._plot.getAxis("bottom").style["tickFont"].pointSize() == 11
    assert window._plot_widget.backgroundBrush().color().name() == "#ffffff"
    assert not window._stale
    for key, frame in source_frames.items():
        pd.testing.assert_frame_equal(window._sources[key].df, frame, check_exact=True)
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)


def test_histogram_fit_and_residual_arrays_are_not_replaced(window, qtbot, monkeypatch):
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")
    window._fit_values_check.setChecked(True)
    window._residual_check.setChecked(True)
    _settle(window)
    fit = window.fit_result()
    assert fit is not None
    item_pairs = [(item, item.xData, item.yData, item.xData.copy(), item.yData.copy())
                  for item in (*window._hist_items, *window._fit_items,
                               *window._resid.listDataItems())]
    fit_values = window._fit_values_item
    fit_text = fit_values.text
    details = window.details_text()
    pending = dict(window._pending_notes)
    window._plot.setRange(xRange=(0.0001, 0.0006), yRange=(0, 20), padding=0)
    view = np.asarray(window._plot.viewRange()).copy()
    watchers = _watch_render_and_analysis(window, monkeypatch)

    _exercise_text_and_canvas(window)
    qtbot.wait(10)

    for watcher in watchers:
        watcher.assert_not_called()
    assert window.fit_result() is fit
    assert window._fit_values_item is fit_values
    assert window._fit_values_item.text == fit_text
    assert window.details_text() == details
    assert window._pending_notes == pending
    np.testing.assert_array_equal(window._plot.viewRange(), view)
    for item, x, y, xcopy, ycopy in item_pairs:
        assert item.xData is x and item.yData is y
        np.testing.assert_array_equal(item.xData, xcopy)
        np.testing.assert_array_equal(item.yData, ycopy)


def test_pending_axis_does_not_change_rendered_units_during_a_style_edit(window, monkeypatch):
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    old_label = window._plot.getAxis("bottom").labelText
    old_scale = window._plot.getAxis("bottom").scale
    item = window._hist_item
    before = item.yData.copy()
    monkeypatch.setattr(window, "_from_the_user", lambda *_a: True)
    window._hist_col.setCurrentText("amplitude")
    assert window._stale
    watchers = _watch_render_and_analysis(window, monkeypatch)

    window.set_title("The still-committed dwell distribution")
    window.set_font_size(15)
    window._x_label_edit.setText("Custom caption")
    window._x_label_edit.clear()

    for watcher in watchers:
        watcher.assert_not_called()
    assert window._stale
    assert window._hist_item is item
    np.testing.assert_array_equal(item.yData, before)
    assert window._plot.getAxis("bottom").labelText == old_label
    assert window._plot.getAxis("bottom").scale == old_scale
    assert window._hist_col.currentText() == "amplitude"


def test_custom_title_keeps_warning_in_details_and_defaults_restore(window, monkeypatch):
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    window._apply_title("Automatic scientific title", warn="NOT COMPARABLE: baseline mismatch")
    window._set_details(window.details_text())
    labels = [window._plot.getAxis(side).labelText for side in ("bottom", "left")]
    watchers = _watch_render_and_analysis(window, monkeypatch)

    _exercise_text_and_canvas(window)
    assert window._plot.titleLabel.text == "Publication figure"
    assert "NOT COMPARABLE: baseline mismatch" in window.details_text()
    window.set_title("")
    window._x_label_edit.clear()
    window._y_label_edit.clear()
    window.set_font_family("")
    window._on_style_title_bold(False)
    window._on_style_title_italic(False)

    for watcher in watchers:
        watcher.assert_not_called()
    assert window._plot.titleLabel.text == "Automatic scientific title"
    assert "NOT COMPARABLE: baseline mismatch" in window.details_text()
    assert [window._plot.getAxis(side).labelText for side in ("bottom", "left")] == labels
    assert not window._plot.titleLabel.item.font().bold()
    assert not window._plot.titleLabel.item.font().italic()
    assert window._plot.getAxis("bottom").labelStyle["font-family"] == window._qfont().family()


def test_never_run_prompt_survives_presentation_edits(qtbot, monkeypatch):
    window = PlotsWindow(Session())
    qtbot.addWidget(window)
    message = window._message_item
    watchers = _watch_render_and_analysis(window, monkeypatch)
    _exercise_text_and_canvas(window)
    for watcher in watchers:
        watcher.assert_not_called()
    assert not window._plot_armed
    assert window._message_item is message
    assert window._hist_item is None and window._scatter_item is None
