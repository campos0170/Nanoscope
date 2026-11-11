"""Styling is opt-in, reversible, and cannot recalculate scientific results."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui

from nanoscope.analysis import iv, voltage_dep
from nanoscope.core.figure_quantity_basis import read_quantity_bases
from nanoscope.core.session import Session
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.plot_appearance import (
    PlotAppearance, PlotAppearanceWidget, apply_plot_appearance,
)
from nanoscope.gui.voltage_dep_panel import VDepWindow


def test_publication_background_keeps_legend_text_legible_and_resettable(qtbot):
    widget = pg.PlotWidget(background="#111316")
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    legend = plot.addLegend(labelTextColor="#bbbbbb")
    item = plot.plot([0, 1], [1, 2], name="Condition A")
    before = _xy(item)
    apply_plot_appearance(plot, PlotAppearance(background="light"), plot_widget=widget)
    assert legend.opts["labelTextColor"].name() == "#20242b"
    assert legend.items[0][1].opts["color"].name() == "#20242b"
    apply_plot_appearance(plot, PlotAppearance(background="dark"), plot_widget=widget)
    assert legend.opts["labelTextColor"].name() == "#e6e9ef"
    apply_plot_appearance(plot, PlotAppearance(), plot_widget=widget)
    assert legend.opts["labelTextColor"].name() == "#bbbbbb"
    for original, current in zip(before, _xy(item)):
        np.testing.assert_array_equal(current, original)


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


def _look(appearance):
    """``appearance`` without the witness of the quantity each axis shows: the part that is look.
    2026-11-11: a page that presents units (Voltage dependence, not I-V) always carries the
    witness, which ``present_page_axes`` writes at every draw (8661d35, quantity display); an
    appearance that has never been styled is empty only in this sense."""
    return replace(appearance, quantity_display=None)


def _bases(window):
    """The quantity each axis of ``window`` shows according to its appearance's witness, as
    ``{axis: (quantity key, display unit)}``: Voltage dependence shows the capture rate in Hz,
    I-V presents no units and records none."""
    return {axis: (basis.quantity.key, basis.quantity.display_unit)
            for axis, basis in read_quantity_bases(window._appearance.quantity_display).items()}


def _drawn_quantity(window):
    return {} if isinstance(window, IVWindow) else {"left": ("capture_rate_hz", "Hz")}


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
    """An appearance nobody has touched applies nothing to the plot, and Reset figure puts back
    what the plot looked like before.

    2026-11-11: the first half, "applies nothing to an untouched plot", is checked for I-V only
    (the ``_nanoscope_appearance_defaults`` assertion below). Voltage dependence presents units, so
    its appearance holds the witness of the quantity it shows from its first draw (8661d35), and
    an appearance holding only the witness is not empty to ``apply_plot_appearance``: it misses
    the early return that leaves a plot alone, so the draw's styling pass records the plot's
    defaults before this test can read them, and every baseline below is taken after that pass.
    For Voltage dependence the test therefore pins the second half (Reset figure restores that
    baseline, and the witness is what is drawn), not that nothing was styled. Whether the early
    return should ignore the witness, as ``_describe_step`` does, is the owner's choice; the app
    is as released."""
    w = scientific_window
    # 2026-11-11: Voltage dependence's appearance holds the witness of the quantity it shows from
    # its first draw, so it is "empty" apart from that, and its first draw already goes through the
    # styling pass (with nothing to style). I-V, which presents no units, is empty and untouched.
    assert _look(w._appearance) == PlotAppearance()
    assert _bases(w) == _drawn_quantity(w)
    if isinstance(w, IVWindow):
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
    assert _look(w._appearance) == PlotAppearance() and _bases(w) == _drawn_quantity(w)
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
    assert not widget.background.isHidden()
    assert not widget.preset.isHidden()
    assert widget.data_color.isHidden() and widget.fit_color.isHidden()
    assert widget.data_color.isHidden()
    assert widget.fit_color.isHidden()


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
                              physical_height_mm=64.5, extension=".png", native=False, dpi=300,
                              frozen_snapshot=None)
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
