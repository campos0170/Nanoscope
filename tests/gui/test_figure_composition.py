"""Frozen composition preserves exact scientific resources and source view state."""
from dataclasses import replace
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from nanoscope.gui.figure_bundle import (FigureSnapshot, capture_figure_snapshot,
                                         decode_figure_table, write_frozen_figure_image)
from nanoscope.gui import figure_composition as C
from nanoscope.gui.plot_export_options import resolve_export_options


@pytest.fixture
def sources(qtbot):
    widgets, snapshots, frames = [], [], []
    for index, color in enumerate(("#124a90", "#bc2430")):
        widget = pg.PlotWidget(background="w")
        qtbot.addWidget(widget)
        widget.resize(500, 340)
        widget.show()
        x = np.array([1., 2., 3., 4.]) + index * 3
        y = np.array([2., 3., 1., 4.]) + index * 10
        widget.plot(x, y, pen=pg.mkPen(color, width=2), symbol="o", symbolBrush=color)
        widget.setLabel("bottom", "Time", units="s")
        widget.setLabel("left", "Current", units="A")
        widget.setTitle(f"Source {index + 1}")
        widget.setXRange(float(x.min()) - .5, float(x.max()) + .5, padding=0)
        widget.setYRange(float(y.min()) - .5, float(y.max()) + .5, padding=0)
        qtbot.wait(10)
        frame = pd.DataFrame({"x": x, "y": y, "source_id": pd.Series(["", 9, None, pd.NA], dtype=object)})
        model = C.capture_cartesian_model(widget.getPlotItem())
        assert model is not None
        snapshot = capture_figure_snapshot(widget.getPlotItem(), data=frame,
            config={"fit_parameters": [2.5, 9.0], "source": index},
            provenance={"recording": f"synthetic-{index}", "analysis_recomputed": False},
            additional_resources={C.CARTESIAN_RESOURCE: model})
        widgets.append(widget)
        snapshots.append(snapshot)
        frames.append(frame)
    return widgets, snapshots, frames


def test_frozen_composition_preserves_every_panel_resource_exactly(sources):
    widgets, snapshots, frames = sources
    views = [widget.viewRange() for widget in widgets]
    before = [snapshot.to_storage() for snapshot in snapshots]
    result = C.compose_figures(snapshots, ["A · controls", "B · treated"], 2)
    resources = dict(result.additional_resources)
    assert C.PICTURE_RESOURCE in resources
    panels = result.config()["panels"]
    for index, panel in enumerate(panels):
        mapping = panel["blob_resources"]
        restored = FigureSnapshot.from_storage(panel["snapshot"],
                                               {key: resources[path] for key, path in mapping.items()})
        assert restored.to_storage() == before[index]
        assert_frame_equal(restored.dataframe(), frames[index], check_exact=True)
        assert_frame_equal(decode_figure_table(resources[mapping["data_parquet"]]), frames[index], check_exact=True)
    assert [widget.viewRange() for widget in widgets] == views
    assert [snapshot.to_storage() for snapshot in snapshots] == before
    assert result.provenance()["analysis_recomputed"] is False
    assert result.dataframe()["panel_id"].tolist() == ["panel_001", "panel_002"]
    svg = ET.fromstring(result.svg_bytes)
    ids = [element.attrib["id"] for element in svg.iter() if "id" in element.attrib]
    assert len(ids) == len(set(ids))
    assert {"panel_001", "panel_002", "panel_001_label", "panel_002_label"} <= set(ids)
    assert "A · controls" in result.svg_bytes.decode("utf-8")


@pytest.mark.parametrize("custom", ["subclass", "tick_spacing", "tick_method", "title", "axis_label"])
def test_unmodeled_axis_and_title_formatting_fail_closed(sources, custom):
    widgets, _, _ = sources
    plot = widgets[0].getPlotItem()
    if custom == "subclass":
        from nanoscope.gui.plots_panel import _DecadeLabelAxis
        class UnknownAxis(_DecadeLabelAxis):
            pass
        plot.setAxisItems({"bottom": UnknownAxis(orientation="bottom")})
    elif custom == "tick_spacing":
        plot.getAxis("bottom").setTickSpacing(major=1, minor=.1)
    elif custom == "tick_method":
        plot.getAxis("bottom").tickStrings = lambda values, scale, spacing: ["custom"] * len(values)
    elif custom == "title":
        plot.titleLabel.setText("Left-aligned title", justify="left")
    else:
        plot.getAxis("bottom").label.setRotation(30)
    assert C.capture_cartesian_model(plot) is None


def test_composition_exports_both_panels_as_native_and_publication_png_pdf(sources, tmp_path):
    _, snapshots, _ = sources
    result = C.compose_figures(snapshots, ["A", "B"], 2)
    for extension in (".png", ".pdf", ".svg"):
        options = resolve_export_options((result.width, result.height), extension=extension,
                                         native=False, width=180,
                                         height=180 * result.height / result.width, dpi=150)
        target = tmp_path / ("composite" + extension)
        write_frozen_figure_image(target, result, options)
        assert target.stat().st_size > 100
        if extension == ".png":
            image = QtGui.QImage(str(target)).convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
            blue = (pixels[:, :, 2] > 100) & (pixels[:, :, 0] < 80) & (pixels[:, :, 1] < 120)
            red = (pixels[:, :, 0] > 120) & (pixels[:, :, 1] < 90) & (pixels[:, :, 2] < 100)
            assert blue.sum() > 20 and red.sum() > 20
            assert np.where(blue)[1].mean() < image.width() / 2 < np.where(red)[1].mean()


def test_shared_axes_are_range_union_on_detached_clones_only(sources, monkeypatch):
    widgets, snapshots, _ = sources
    before_ranges = [widget.viewRange() for widget in widgets]
    before_arrays = [(item.xData.copy(), item.yData.copy())
                     for widget in widgets for item in widget.listDataItems()]
    for widget in widgets:
        monkeypatch.setattr(widget.getPlotItem(), "setRange", lambda *a, **k: pytest.fail("source view mutated"))
        for item in widget.listDataItems():
            monkeypatch.setattr(item, "setData", lambda *a, **k: pytest.fail("source series recomputed"))
    assert C.shared_axes_compatibility(snapshots)[0]
    result = C.compose_figures(snapshots, ["A", "B"], 2, share_x=True, share_y=True)
    ranges = result.config()["view_ranges"]
    assert ranges[0] == ranges[1]
    assert ranges[0][0] == [min(v[0][0] for v in before_ranges), max(v[0][1] for v in before_ranges)]
    assert ranges[0][1] == [min(v[1][0] for v in before_ranges), max(v[1][1] for v in before_ranges)]
    assert [widget.viewRange() for widget in widgets] == before_ranges
    for item, (x, y) in zip((item for w in widgets for item in w.listDataItems()), before_arrays):
        np.testing.assert_array_equal(item.xData, x)
        np.testing.assert_array_equal(item.yData, y)
    assert result.provenance()["source_figures_modified"] is False


def test_unshared_direction_can_keep_different_units_and_ranges(sources):
    widgets, snapshots, _ = sources
    widgets[1].setLabel("left", "Voltage", units="V")
    model = C.capture_cartesian_model(widgets[1].getPlotItem())
    snapshots[1] = replace(snapshots[1], additional_resources=tuple(
        (name, model if name == C.CARTESIAN_RESOURCE else payload)
        for name, payload in snapshots[1].additional_resources))
    assert not C.shared_axes_compatibility(snapshots)[0]
    assert C.shared_axes_compatibility(snapshots, share_x=True, share_y=False)[0]
    result = C.compose_figures(snapshots, ["A", "B"], 2, share_x=True)
    ranges = result.config()["view_ranges"]
    assert ranges[0][0] == ranges[1][0]
    assert ranges[0][1] != ranges[1][1]
    with pytest.raises(ValueError, match="labels, units"):
        C.compose_figures(snapshots, ["A", "B"], 2, share_y=True)


@pytest.mark.parametrize("change", ["units", "label", "log", "unlabelled"])
def test_shared_axis_mismatch_and_unlabelled_axes_are_refused(sources, change):
    widgets, snapshots, _ = sources
    plot = widgets[1].getPlotItem()
    if change == "log":
        plot.setLogMode(x=True)
    elif change == "units":
        plot.setLabel("bottom", "Time", units="ms")
    elif change == "unlabelled":
        plot.setLabel("bottom", "", units="")
    else:
        plot.setLabel("bottom", "Dwell", units="s")
    model = C.capture_cartesian_model(plot)
    assert model is not None
    snapshots[1] = replace(snapshots[1], additional_resources=tuple(
        (name, model if name == C.CARTESIAN_RESOURCE else payload)
        for name, payload in snapshots[1].additional_resources))
    assert not C.shared_axes_compatibility(snapshots, share_x=True, share_y=False)[0]
    with pytest.raises(ValueError):
        C.compose_figures(snapshots, ["A", "B"], 2, share_x=True)


@pytest.mark.parametrize("kind", ["scatter", "bar", "line", "empty", "constant"])
def test_supported_basic_cartesian_models_can_be_replayed(qtbot, kind):
    widget = pg.PlotWidget(background="w")
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    if kind == "scatter":
        widget.addItem(pg.ScatterPlotItem(x=[1, 2, 3], y=[2, 3, 4], symbol="o"))
    elif kind == "bar":
        widget.addItem(pg.BarGraphItem(x=[1, 2, 3], height=[2, 3, 4], width=.8))
    elif kind != "empty":
        widget.plot([1, 2, 3], [2, 2, 2] if kind == "constant" else [2, 3, 4])
    widget.setLabel("bottom", "Time", units="s")
    widget.setLabel("left", "Current", units="A")
    qtbot.wait(10)
    model = C.capture_cartesian_model(widget.getPlotItem())
    assert model is not None
    snapshot = capture_figure_snapshot(widget.getPlotItem(), data=pd.DataFrame(),
                                        additional_resources={C.CARTESIAN_RESOURCE: model})
    result = C.compose_figures([snapshot, snapshot], ["A", "B"], 2, share_x=True, share_y=True)
    assert result.width > 0 and result.height > 0


def test_log_curve_replay_does_not_double_log_or_change_original_samples(qtbot):
    widget = pg.PlotWidget(background="w")
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    x, y = np.array([1., 10., 100.]), np.array([2., 20., 200.])
    curve = widget.plot(x, y)
    widget.setLabel("bottom", "Time", units="s")
    widget.setLabel("left", "Current", units="A")
    widget.setLogMode(x=True, y=True)
    qtbot.wait(10)
    payload = C.capture_cartesian_model(widget.getPlotItem())
    model = json.loads(payload)
    np.testing.assert_array_equal(C._restore_array(model["series"][0]["x"]), x)
    clone = C._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    copied = clone.listDataItems()[0]
    np.testing.assert_array_equal(copied.xData, x)
    np.testing.assert_array_equal(copied.yData, y)
    np.testing.assert_array_equal(copied.getData()[0], curve.getData()[0])
    np.testing.assert_array_equal(copied.getData()[1], curve.getData()[1])


@pytest.mark.parametrize("kind", ["image", "annotation", "errorbar", "legend", "custom_ticks", "palette"])
def test_complex_or_scientifically_annotated_panels_cannot_silently_lose_items(qtbot, kind):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.plot([1, 2], [3, 4], name="source")
    if kind == "image":
        widget.addItem(pg.ImageItem(np.ones((3, 3))))
    elif kind == "annotation":
        widget.addItem(pg.TextItem("scientific caution"))
    elif kind == "errorbar":
        widget.addItem(pg.ErrorBarItem(x=np.array([1]), y=np.array([3]), height=np.array([1])))
    elif kind == "legend":
        widget.addLegend()
    elif kind == "custom_ticks":
        widget.getAxis("bottom").setTicks([[(1, "custom")]])
    else:
        widget.addItem(pg.ScatterPlotItem(x=[1, 2], y=[3, 4], brush=["red", "blue"]))
    assert C.capture_cartesian_model(widget.getPlotItem()) is None


def test_missing_model_keeps_frozen_composition_available(sources):
    _, snapshots, _ = sources
    no_model = replace(snapshots[0], additional_resources=tuple(
        (name, value) for name, value in snapshots[0].additional_resources if name != C.CARTESIAN_RESOURCE))
    assert C.compose_figures([no_model, snapshots[1]], ["A", "B"], 2).width > 0
    assert not C.shared_axes_compatibility([no_model, snapshots[1]])[0]


def test_invalid_labels_counts_and_canvas_refused_before_allocating(sources, monkeypatch):
    _, snapshots, _ = sources
    for figures, labels, columns in (([], [], 1), (snapshots, ["A"], 2),
                                    (snapshots, ["A" * 33, "B"], 2), (snapshots, ["A", "B"], 0)):
        with pytest.raises(ValueError):
            C.compose_figures(figures, labels, columns)
    large = replace(snapshots[0], width=20_000., height=20_000.)
    monkeypatch.setattr(C, "_load_picture", lambda *a, **k: pytest.fail("allocated an oversized composite"))
    with pytest.raises(ValueError, match="megapixels"):
        C.compose_figures([large, large], ["A", "B"], 2)
