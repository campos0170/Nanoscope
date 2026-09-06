"""Built-in scientific tick formatting is replayed without changing source data."""
from dataclasses import replace
import json
import struct
import zlib

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import pyqtgraph as pg

from nanoscope.core.session import Session
from nanoscope.gui import figure_composition as C
from nanoscope.gui.axes import ReadableLogAxis, ReadableLinearAxis
from nanoscope.gui.figure_bundle import capture_figure_snapshot
from nanoscope.gui.plots_panel import PlotsWindow, _DecadeLabelAxis
from nanoscope.gui.psd_panel import _ReadableAxis


def _snapshot(plot, frame=None):
    payload = C.capture_cartesian_model(plot)
    assert payload is not None
    return capture_figure_snapshot(plot, data=pd.DataFrame() if frame is None else frame,
                                   additional_resources={C.CARTESIAN_RESOURCE: payload})


def _with_model(snapshot, model):
    return replace(snapshot, additional_resources=tuple(
        (key, json.dumps(model).encode() if key == C.CARTESIAN_RESOURCE else value)
        for key, value in snapshot.additional_resources))


@pytest.mark.parametrize("log_x", [False, True])
def test_real_population_scatter_replays_ms_ticks_and_uniform_spot_styles(qtbot, monkeypatch, log_x):
    panel = PlotsWindow(Session())
    qtbot.addWidget(panel)
    frame = pd.DataFrame({"dwell_s": np.geomspace(.0001, .1, 30),
                          "amplitude": np.linspace(10., 20., 30), "accepted": True})
    panel.set_data({"synthetic": frame})
    panel.set_plot_type("Scatter")
    panel.set_scatter_columns("dwell_s", "amplitude")
    panel.set_time_unit("ms")
    panel.set_scatter_log(log_x, False)
    panel._run_plot()
    panel.resize(1200, 820)
    panel.show()
    qtbot.wait(10)
    plot = panel._plot
    before = panel._plot_data().copy(deep=True)
    ranges = np.array(plot.viewRange())
    spot = panel._scatter_item
    x, y, sizes = (spot.data[key].copy() for key in ("x", "y", "size"))
    pens = [C._pen(pen) for pen in spot.data["pen"]]
    snapshot = _snapshot(plot, before)
    model = C._model(snapshot)
    axis = plot.getAxis("bottom")
    assert model["version"] == 2
    descriptor = model["axes"]["bottom"]["formatter"]
    assert descriptor == {"kind": "population-decade", "display_factor": 1000. if log_x else 1.}
    assert axis.scale == (1. if log_x else 1000.)
    values, spacing = ([-3., -2., -1.], 1.) if log_x else ([.001, .002, .003], .001)
    assert axis.tickStrings(values, axis.scale, spacing) == (["1", "10", "100"] if log_x else ["1", "2", "3"])
    clone = C._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    copied_axis = clone.getPlotItem().getAxis("bottom")
    assert type(copied_axis) is _DecadeLabelAxis
    assert copied_axis.tickStrings(values, copied_axis.scale, spacing) == axis.tickStrings(values, axis.scale, spacing)
    copied_spot = next(item for item in clone.getPlotItem().items if type(item) is pg.ScatterPlotItem)
    np.testing.assert_array_equal(copied_spot.data["x"], x)
    np.testing.assert_array_equal(copied_spot.data["y"], y)
    assert C._scatter_style(copied_spot) == C._scatter_style(spot)
    monkeypatch.setattr(panel, "_run_plot", lambda *a, **k: pytest.fail("analysis reran"))
    monkeypatch.setattr(plot, "setRange", lambda *a, **k: pytest.fail("source zoom changed"))
    result = C.compose_figures([snapshot, snapshot], ["A", "B"], 2, share_x=True, share_y=True)
    assert result.provenance()["analysis_recomputed"] is False
    np.testing.assert_array_equal(plot.viewRange(), ranges)
    for key, expected in (("x", x), ("y", y), ("size", sizes)):
        np.testing.assert_array_equal(spot.data[key], expected)
    assert [C._pen(pen) for pen in spot.data["pen"]] == pens
    assert_frame_equal(panel._plot_data(), before, check_exact=True)
    # Mixed selection rings carry meaningful appearance; never flatten them.
    spot.points()[0].setPen(pg.mkPen("orange", width=2))
    assert C.capture_cartesian_model(plot) is None


@pytest.mark.parametrize("axis_class,log", [(ReadableLogAxis, True), (ReadableLinearAxis, False),
                                           (_DecadeLabelAxis, True), (_ReadableAxis, True)])
def test_exact_known_axis_class_and_tick_labels_survive(qtbot, axis_class, log):
    widget = pg.PlotWidget(axisItems={"bottom": axis_class(orientation="bottom"),
                                     "left": axis_class(orientation="left")})
    qtbot.addWidget(widget)
    widget.resize(480, 320)
    widget.show()
    widget.plot([.001, .01, .1], [1e-5, 1e-4, 1e-3])
    widget.setLabel("bottom", "Frequency", units="Hz")
    widget.setLabel("left", "PSD", units="pA²/Hz")
    widget.setLogMode(x=log, y=log)
    qtbot.wait(10)
    snapshot = _snapshot(widget.getPlotItem())
    model = C._model(snapshot)
    clone = C._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    for side in ("bottom", "left"):
        original, copied = widget.getPlotItem().getAxis(side), clone.getPlotItem().getAxis(side)
        assert type(copied) is axis_class
        assert (copied.labelText, copied.labelUnits, copied.logMode) == (
            original.labelText, original.labelUnits, original.logMode)
        for values, spacing in (([-5., -2., 0., 2.], 1.), ([-1.7, -.3, .2], .1)):
            assert copied.tickStrings(values, copied.scale, spacing) == original.tickStrings(values, original.scale, spacing)
        if axis_class is _ReadableAxis:
            assert copied.tickStrings([-5., 0., 2.], 1., 1.) == ["10⁻⁵", "10⁰", "10²"]


def test_actual_psd_model_keeps_power_notation_and_original_samples(qtbot, tmp_path):
    from .test_psd import _fed_window
    panel, _ = _fed_window(qtbot, tmp_path, n=4096)
    panel.resize(1200, 820)
    panel.show()
    qtbot.wait(10)
    arrays = [(item.xData.copy(), item.yData.copy()) for item in panel._plot.listDataItems()]
    ranges = np.asarray(panel._plot.viewRange())
    snapshot = _snapshot(panel._plot, panel._plot_data())
    model = C._model(snapshot)
    clone = C._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    for side in ("left", "bottom"):
        original = panel._plot.getAxis(side)
        copied = clone.getPlotItem().getAxis(side)
        assert type(copied) is _ReadableAxis
        assert copied.labelText == original.labelText
        assert copied.tickStrings([-5., 0., 2.], 1., 1.) == original.tickStrings([-5., 0., 2.], 1., 1.)
    for original, copied, (x, y) in zip(panel._plot.listDataItems(), clone.listDataItems(), arrays):
        np.testing.assert_array_equal(original.xData, x)
        np.testing.assert_array_equal(original.yData, y)
        np.testing.assert_array_equal(copied.xData, x)
        np.testing.assert_array_equal(copied.yData, y)
        np.testing.assert_array_equal(copied.getData()[0], original.getData()[0])
        np.testing.assert_array_equal(copied.getData()[1], original.getData()[1])
    np.testing.assert_array_equal(panel._plot.viewRange(), ranges)


@pytest.fixture
def basic_snapshot(qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    widget.plot([1, 2, 3], [4, 5, 6])
    widget.setLabel("bottom", "Time", units="s")
    widget.setLabel("left", "Current", units="pA")
    qtbot.wait(10)
    return widget, _snapshot(widget.getPlotItem())


def test_legacy_standard_axis_models_still_match_and_replay(basic_snapshot, qtbot):
    _, snapshot = basic_snapshot
    model = C._model(snapshot)
    for axis in model["axes"].values():
        axis.pop("formatter")
    legacy = _with_model(snapshot, model)
    assert C.shared_axes_compatibility([snapshot, legacy])[0]
    clone = C._clone_model(C._model(legacy), model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    assert type(clone.getPlotItem().getAxis("bottom")) is pg.AxisItem


def test_detached_layout_settles_before_axis_measurement_and_freezing(basic_snapshot, qtbot, monkeypatch):
    _, snapshot = basic_snapshot
    model = C._model(snapshot)
    model["axes"]["bottom"]["style"].update(autoExpandTextSpace=False, tickTextHeight=34)
    clone = C._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    bottom = clone.getPlotItem().getAxis("bottom")
    assert bottom.height() >= bottom.minimumHeight()
    assert bottom.label.pos().y() >= 34 + bottom.style["tickTextOffset"][1]
    altered = _with_model(snapshot, model)
    captured = []
    original = C.capture_figure_snapshot
    def checked(plot, **kwargs):
        assert plot.layout.isActivated()
        axis = plot.getAxis("bottom")
        assert axis.height() >= axis.minimumHeight()
        captured.append(axis.height())
        return original(plot, **kwargs)
    monkeypatch.setattr(C, "capture_figure_snapshot", checked)
    C.compose_figures([snapshot, altered], ["A", "B"], 2, share_x=True)
    assert len(captured) == 2 and captured[0] == captured[1]


@pytest.mark.parametrize("descriptor", [
    {"kind": "remote-import"}, {"kind": "standard", "display_factor": 1},
    {"kind": "readable-log"}, {"kind": "readable-log", "display_factor": True},
    {"kind": "readable-log", "display_factor": 0},
    {"kind": "readable-log", "display_factor": -1},
    {"kind": "readable-log", "display_factor": "1000"},
    {"kind": "readable-log", "display_factor": float("inf")},
    {"kind": "readable-log", "display_factor": float("nan")},
    {"kind": "readable-log", "display_factor": 1, "callable": "tickStrings"},
])
def test_untrusted_axis_formatter_schema_is_rejected(basic_snapshot, descriptor):
    _, snapshot = basic_snapshot
    model = C._model(snapshot)
    model["axes"]["bottom"]["formatter"] = descriptor
    altered = _with_model(snapshot, model)
    assert not C.shared_axes_compatibility([altered, altered])[0]
    with pytest.raises(ValueError, match="axis"):
        C.compose_figures([altered], ["A"], 1, share_x=True)


def test_matching_labels_cannot_hide_a_different_display_factor(basic_snapshot):
    _, snapshot = basic_snapshot
    model = C._model(snapshot)
    model["version"] = 2
    model["axes"]["bottom"]["formatter"] = {"kind": "population-decade", "display_factor": 1000.}
    first = _with_model(snapshot, model)
    model["axes"]["bottom"]["formatter"]["display_factor"] = 1.
    second = _with_model(snapshot, model)
    assert not C.shared_axes_compatibility([first, second], share_x=True, share_y=False)[0]
    assert C.shared_axes_compatibility([first, second], share_x=False, share_y=True)[0]


def test_version_one_cannot_claim_new_formatter_semantics(basic_snapshot):
    _, snapshot = basic_snapshot
    model = C._model(snapshot)
    assert model["version"] == 1
    model["axes"]["bottom"]["formatter"] = {"kind": "population-decade", "display_factor": 1000.}
    altered = _with_model(snapshot, model)
    with pytest.raises(ValueError, match="version 2"):
        C._model(altered)


def test_version_two_cannot_silently_default_a_missing_formatter(basic_snapshot):
    _, snapshot = basic_snapshot
    model = C._model(snapshot)
    model["version"] = 2
    model["axes"]["bottom"].pop("formatter")
    with pytest.raises(ValueError, match="explicit formatter"):
        C._model(_with_model(snapshot, model))


@pytest.mark.parametrize("custom", ["subclass", "instance_method", "tick_spacing", "gap", "linear_log"])
def test_unknown_or_overridden_readable_axes_fail_closed(basic_snapshot, custom):
    widget, _ = basic_snapshot
    axis_class = ReadableLogAxis
    if custom == "subclass":
        class CustomAxis(ReadableLogAxis):
            pass
        axis_class = CustomAxis
    elif custom == "linear_log":
        axis_class = ReadableLinearAxis
    axis = axis_class(orientation="bottom")
    widget.getPlotItem().setAxisItems({"bottom": axis})
    if custom == "instance_method":
        axis.choose_notation = lambda values: True
    elif custom == "tick_spacing":
        axis.setTickSpacing(1, .1)
    elif custom == "gap":
        axis.LABEL_GAP_PX = 20.
    elif custom == "linear_log":
        widget.setLogMode(x=True)
    assert C.capture_cartesian_model(widget.getPlotItem()) is None


@pytest.mark.parametrize("width,height", [(32769, 10), (10001, 10000)])
def test_png_header_limit_cannot_be_bypassed_with_small_metadata(basic_snapshot, monkeypatch, width, height):
    _, snapshot = basic_snapshot
    png = bytearray(snapshot.png_bytes)
    assert png[12:16] == b"IHDR"
    png[16:24] = struct.pack(">II", width, height)
    png[29:33] = struct.pack(">I", zlib.crc32(png[12:29]) & 0xffffffff)
    forged = replace(snapshot, png_bytes=bytes(png), width=100, height=100)
    monkeypatch.setattr(C, "_load_picture", lambda *a, **k: pytest.fail("oversized PNG was decoded"))
    with pytest.raises(ValueError, match="PNG exceeds"):
        C.compose_figures([forged], ["A"], 1)
