"""Styling is opt-in, reversible, and cannot recalculate scientific results."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui

from nanoscope.analysis import iv, voltage_dep
from nanoscope.core.session import Session
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.plot_appearance import (
    PlotAppearance, PlotAppearanceWidget, apply_plot_appearance,
)
from nanoscope.gui.voltage_dep_panel import VDepWindow


@pytest.fixture(params=["iv", "vdep"])
def scientific_window(request, qtbot):
    if request.param == "iv":
        window = IVWindow(Session(), defer_initial_refresh=True)
        window._points = [iv.IVPoint(v, 100 * v + 2, "pA", "measured", str(i), f"run {i}")
                          for i, v in enumerate((-.2, -.5, -.8, -1.0))]
    else:
        window = VDepWindow(Session(), defer_initial_refresh=True)
        window._points = [voltage_dep.VDepPoint(
            v, -100 * v + 2, f"run {i}", baseline_fingerprint="baseline-a",
            analyzed_duration_s=10, voltage_source="measured", doc_id=str(i))
                          for i, v in enumerate((-.2, -.5, -.8, -1.0))]
        window._series = voltage_dep.vdep_comparability(window._points, metric="capture_rate_hz")
        window._applied_fit = "linear"
    window._redraw()
    qtbot.addWidget(window)
    return window


def _xy(item):
    x, y = item.getData()
    return np.array(x, copy=True), np.array(y, copy=True)


def _forbid_analysis(monkeypatch, window):
    def fail(*args, **kwargs):
        pytest.fail("An appearance edit called a scientific/data-refresh path")
    for module, name in ((iv, "iv_fit"), (iv, "iv_points"),
                         (voltage_dep, "vdep_fit"), (voltage_dep, "vdep_series")):
        monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(window, "refresh", fail)
    monkeypatch.setattr(window, "_redraw", fail)


def test_appearance_edit_preserves_points_fit_provenance_and_zoom(scientific_window, monkeypatch, qtbot):
    w = scientific_window
    old_points = w.points()
    old_fit = copy.deepcopy(w.fit())
    old_scatter = _xy(w._scatter_item)
    old_line = _xy(w._fit_item)
    old_details = w.details_text()
    old_export = w._plot_data().copy(deep=True)
    old_title = w._plot.titleLabel.text
    old_labels = [(w._plot.getAxis(s).labelText, w._plot.getAxis(s).labelUnits)
                  for s in ("left", "bottom")]
    w._plot.setRange(xRange=(-.65, -.25), yRange=(-55, 70), padding=0)
    before_range = copy.deepcopy(w._plot.viewRange())
    changes = []
    w.workspaceChanged.connect(lambda: changes.append(True))
    _forbid_analysis(monkeypatch, w)
    w._appearance_widget._apply_preset(1)
    w._appearance_widget.title_size.setValue(20)
    qtbot.wait(10)
    assert len(changes) == 2
    assert w.points() == old_points
    assert w.fit() == old_fit
    assert w.details_text() == old_details
    pd.testing.assert_frame_equal(w._plot_data(), old_export, check_exact=True)
    assert w._plot.titleLabel.text == old_title
    assert [(w._plot.getAxis(s).labelText, w._plot.getAxis(s).labelUnits)
            for s in ("left", "bottom")] == old_labels
    for before, after in zip(old_scatter + old_line, _xy(w._scatter_item) + _xy(w._fit_item)):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(w._plot.viewRange(), before_range, rtol=0, atol=1e-12)
    assert w._plot_widget.backgroundBrush().color().name() == "#ffffff"
    assert w._fit_item.opts["pen"].widthF() == 2
    assert w._appearance.title_size == 20


def test_empty_appearance_is_noop_and_reset_restores_original(scientific_window, monkeypatch):
    w = scientific_window
    assert w._appearance.to_dict() == {}
    assert not hasattr(w._plot, "_nanoscope_appearance_defaults")
    baseline_background = w._plot_widget.backgroundBrush()
    baseline_pen = QtGui.QPen(w._fit_item.opts["pen"])
    baseline_title_opts = dict(w._plot.titleLabel.opts)
    baseline_label_style = dict(w._plot.getAxis("left").labelStyle)
    baseline_font = w._plot.getAxis("left").style["tickFont"]
    baseline_grid = w._plot.getAxis("left").grid
    _forbid_analysis(monkeypatch, w)
    w.set_appearance_state({"background": "light", "fit_color": "#ff0000",
                            "line_width": 4, "axis_size": 19, "tick_size": 17,
                            "title_size": 22, "font_family": "Arial", "grid": False})
    assert w._fit_item.opts["pen"].color().name() == "#ff0000"
    w._appearance_widget.reset_button.click()
    assert w._appearance.to_dict() == {}
    assert w._plot_widget.backgroundBrush() == baseline_background
    assert w._fit_item.opts["pen"] == baseline_pen
    assert w._plot.titleLabel.opts == baseline_title_opts
    assert w._plot.getAxis("left").labelStyle == baseline_label_style
    assert w._plot.getAxis("left").style["tickFont"] == baseline_font
    assert w._plot.getAxis("left").grid == baseline_grid


def test_saved_appearance_restore_is_silent_and_survives_explicit_redraw(scientific_window):
    w = scientific_window
    changes = []
    w.workspaceChanged.connect(lambda: changes.append(True))
    w.set_appearance_state({"fit_color": "#ff0000", "line_width": 4, "marker_size": 7})
    encoded = json.loads(json.dumps(w._appearance.to_dict()))
    original_fit = copy.deepcopy(w.fit())
    original_line = _xy(w._fit_item)
    w._redraw()
    assert changes == []
    assert w._appearance.to_dict() == encoded
    assert w.fit() == original_fit
    assert w._fit_item.opts["pen"].color().name() == "#ff0000"
    for before, after in zip(original_line, _xy(w._fit_item)):
        np.testing.assert_array_equal(before, after)


def test_shared_helper_preserves_dash_and_source_colors(qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    item = widget.plot([0, 1, 2], [1, 3, 2], pen=pg.mkPen("#8d44aa", width=1,
                                                       style=QtCore.Qt.DashLine))
    apply_plot_appearance(widget.getPlotItem(), PlotAppearance(line_width=3),
                          plot_widget=widget, data_items=(item,))
    assert item.opts["pen"].color().name() == "#8d44aa"
    assert item.opts["pen"].style() == QtCore.Qt.DashLine
    assert item.opts["pen"].widthF() == 3


def test_state_validation_and_widget_restore(qtbot):
    assert PlotAppearance.from_mapping(["bad"]).to_dict() == {}
    state = PlotAppearance.from_mapping({"title_size": float("nan"), "tick_size": -1,
                                        "data_color": "invalid", "line_width": True,
                                        "background": "light", "grid": False,
                                        "font_family": "Arial", "detector": "cusum"})
    assert state.to_dict() == {"font_family": "Arial", "background": "light", "grid": False}
    widget = PlotAppearanceWidget(include_colors=False)
    qtbot.addWidget(widget)
    changes = []
    widget.changed.connect(changes.append)
    widget.set_state(state)
    assert widget.state() == state
    assert changes == []
    assert widget.background.isHidden()
    assert widget.data_color.isHidden()
    assert widget.fit_color.isHidden()
    assert widget.preset.isHidden()


def test_compact_widget_fits_narrow_sidebar(qtbot):
    widget = PlotAppearanceWidget(include_colors=False, compact=True)
    qtbot.addWidget(widget)
    widget.resize(300, 400)
    widget.show()
    assert widget.minimumSizeHint().width() <= 300
    assert widget.font_family.geometry().right() <= 300


def test_gui_export_uses_options_but_path_export_is_unchanged(scientific_window, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from nanoscope.gui import iv_panel, voltage_dep_panel
    module = iv_panel if isinstance(scientific_window, IVWindow) else voltage_dep_panel
    w = scientific_window
    options = SimpleNamespace(width=1024, height=768, physical_width_mm=86,
                              physical_height_mm=64.5, extension=".png", native=False, dpi=300)
    calls = []
    monkeypatch.setattr(module, "choose_live_plot_export",
                        lambda *a, **k: (tmp_path / "figure.png", options))
    monkeypatch.setattr(module, "write_live_plot_image",
                        lambda path, item, **kwargs: calls.append((path, kwargs)))
    assert w.save_image() == tmp_path / "figure.png"
    assert calls[-1][1] == {"width": 1024, "height": 768,
                            "physical_width_mm": 86, "physical_height_mm": 64.5, "dpi": 300}
    assert w.save_image(str(tmp_path / "legacy.png")) == tmp_path / "legacy.png"
    assert calls[-1][1] == {}
