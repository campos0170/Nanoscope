"""Publication layout affects drawing only and cannot discard a scientific key."""
from dataclasses import replace
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui

from nanoscope.gui import figure_composition as C
from nanoscope.gui.figure_bundle import capture_figure_snapshot
from nanoscope.gui.figure_composition_revision import extract_composition
from nanoscope.gui.figure_composer import FigureComposerDialog
from .test_figure_composition import sources
from .test_figure_composer import entries


def test_spans_spacing_and_typography_restore_exact_originals(sources):
    _, snapshots, _ = sources
    snapshots = [snapshots[0], snapshots[1], snapshots[1]]
    before = [snapshot.to_storage() for snapshot in snapshots]
    style = {"family": "Arial", "size_px": 30, "bold": False, "color": "#453377"}
    result = C.compose_figures(snapshots, ["A", "B", "C"], 3,
        panel_spans=[[2, 2], [1, 1], [1, 1]], spacing=38, label_style=style)
    panels, layout = extract_composition(result)
    assert result.config()["version"] == 2
    assert [panel.snapshot.to_storage() for panel in panels] == before
    assert layout["panel_spans"] == [[2, 2], [1, 1], [1, 1]]
    assert layout["spacing"] == 38 and layout["label_style"] == style
    assert C.validate_publication_layout(3, 3, layout["panel_spans"])[3] == [
        (0, 0, 2, 2), (0, 2, 1, 1), (1, 2, 1, 1)]
    svg = ET.fromstring(result.svg_bytes)
    ns = {"s": C._NS}
    a = svg.find("s:g[@id='panel_001']/s:svg", ns)
    b = svg.find("s:g[@id='panel_002']/s:svg", ns)
    assert float(a.attrib["width"]) > 1.9 * float(b.attrib["width"])
    label = svg.find("s:text[@id='panel_001_label']", ns)
    assert label.attrib["font-size"] == "30"
    assert label.attrib["font-weight"] == "normal" and label.attrib["fill"] == "#453377"
    assert result.provenance()["analysis_recomputed"] is False


@pytest.mark.parametrize("change", ["overwide", "zero", "bool", "spacing", "nan", "font", "color"])
def test_invalid_layout_refused_before_render(sources, monkeypatch, change):
    _, snapshots, _ = sources
    args = {"panel_spans": [[1, 1], [1, 1]], "spacing": 20,
            "label_style": dict(C.DEFAULT_LABEL_STYLE)}
    if change == "overwide": args["panel_spans"][0] = [1, 3]
    if change == "zero": args["panel_spans"][0] = [0, 1]
    if change == "bool": args["panel_spans"][0] = [True, 1]
    if change == "spacing": args["spacing"] = 201
    if change == "nan": args["spacing"] = float("nan")
    if change == "font": args["label_style"]["size_px"] = 500
    if change == "color": args["label_style"]["color"] = "transparent"
    monkeypatch.setattr(C, "_load_picture", lambda *_: pytest.fail("Rendered invalid metadata"))
    with pytest.raises(ValueError): C.compose_figures(snapshots, ["A", "B"], 2, **args)


def test_legacy_layout_still_extracts_defaults_and_unknown_v2_fields_fail(sources):
    _, snapshots, _ = sources
    legacy = C.compose_figures(snapshots, ["A", "B"], 2)
    assert legacy.config()["version"] == 1
    assert extract_composition(legacy)[1] == {"columns": 2, "share_x": False,
                                             "share_y": False, "panel_labels": ["A", "B"]}
    config = legacy.config()
    config["version"] = 2
    with pytest.raises(ValueError, match="malformed"):
        extract_composition(replace(legacy, config_json_bytes=json.dumps(config).encode()))


@pytest.fixture
def legend_sources(qtbot):
    plots, snapshots = [], []
    for i in range(2):
        widget = pg.PlotWidget(background="w")
        qtbot.addWidget(widget)
        widget.resize(500, 340)
        widget.addLegend()
        widget.plot([1., 2., 3.], np.array([1., 3., 2.]) + i, pen=pg.mkPen("#174f9a", width=2), name="Control")
        widget.plot([1., 2., 3.], np.array([3., 2., 4.]) + i, pen=pg.mkPen("#b53137", width=2), name="Treated")
        widget.setLabel("bottom", "Time", units="s")
        widget.setLabel("left", "Current", units="A")
        widget.show()
        qtbot.wait(10)
        plot = widget.getPlotItem()
        model = C.capture_cartesian_model(plot)
        assert model is not None and json.loads(model)["version"] == 3
        snapshots.append(capture_figure_snapshot(plot,
            data=pd.DataFrame({"time_s": [1., 2., 3.], "current_A": np.array([1., 3., 2.]) + i}),
            additional_resources={C.CARTESIAN_RESOURCE: model}))
        plots.append(widget)
    return plots, snapshots


def test_shared_legend_is_drawn_once_preserving_models_numbers_and_live_keys(legend_sources, monkeypatch):
    plots, snapshots = legend_sources
    before = [snapshot.to_storage() for snapshot in snapshots]
    ranges = [plot.viewRange() for plot in plots]
    for widget in plots:
        monkeypatch.setattr(widget.getPlotItem().legend, "hide", lambda: pytest.fail("Live legend hidden"))
        for series in widget.listDataItems():
            monkeypatch.setattr(series, "setData", lambda *_a, **_k: pytest.fail("Analysis repeated"))
    assert C.shared_legend_compatibility(snapshots)[0]
    result = C.compose_figures(snapshots, ["A", "B"], 2, share_x=True, shared_legend=True)
    assert [panel.snapshot.to_storage() for panel in extract_composition(result)[0]] == before
    assert [plot.viewRange() for plot in plots] == ranges
    assert all(plot.getPlotItem().legend.isVisible() for plot in plots)
    svg = ET.fromstring(result.svg_bytes)
    texts = ["".join(node.itertext()) for node in svg.iter() if node.tag == f"{{{C._NS}}}text"]
    assert texts.count("Control") == 1 and texts.count("Treated") == 1
    assert extract_composition(result)[1]["shared_legend"] is True
    individual = C.compose_figures(snapshots, ["A", "B"], 2, share_x=True)
    svg = ET.fromstring(individual.svg_bytes)
    texts = ["".join(node.itertext()) for node in svg.iter() if node.tag == f"{{{C._NS}}}text"]
    assert texts.count("Control") == 2 and texts.count("Treated") == 2


@pytest.mark.parametrize("change", ["color", "label", "missing", "annotation"])
def test_shared_legend_refuses_incompatible_or_unmodeled_keys(legend_sources, change):
    _, snapshots = legend_sources
    if change == "missing":
        snapshots[1] = replace(snapshots[1], additional_resources=tuple(
            (key, value) for key, value in snapshots[1].additional_resources if key != C.CARTESIAN_RESOURCE))
    else:
        model = C._model(snapshots[1])
        if change == "color": model["series"][0]["pen"]["brush"]["rgba"] = [0, 255, 0, 255]
        if change == "label": model["legend"]["entries"][0]["text"] = "Other condition"
        if change == "annotation": model["legend"]["entries"][0]["text"] = "<b>Unmodeled</b>"
        snapshots[1] = replace(snapshots[1], additional_resources=tuple(
            (key, C._json(model) if key == C.CARTESIAN_RESOURCE else value)
            for key, value in snapshots[1].additional_resources))
    assert not C.shared_legend_compatibility(snapshots)[0]
    with pytest.raises(ValueError): C.compose_figures(snapshots, ["A", "B"], 2, shared_legend=True)
    assert C.compose_figures(snapshots, ["A", "B"], 2).width > 0


def test_custom_legend_annotation_and_sample_fail_closed(legend_sources):
    plots, _ = legend_sources
    legend = plots[0].getPlotItem().legend
    extra = pg.TextItem("Important qualification")
    extra.setParentItem(legend)
    assert C.capture_cartesian_model(plots[0].getPlotItem()) is None


def test_standalone_legend_has_no_underlying_plot_pixels_and_preserves_text_color(legend_sources):
    _, snapshots = legend_sources
    _, _, legend = C._shared_pictures(snapshots, False, False, shared_legend=True)
    image = QtGui.QImage.fromData(legend.png_bytes).convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
    red = (pixels[:, :, 0] > 120) & (pixels[:, :, 1] < 90) & (pixels[:, :, 2] < 100)
    blue = (pixels[:, :, 2] > 100) & (pixels[:, :, 0] < 80) & (pixels[:, :, 1] < 120)
    assert red.any() and blue.any()
    assert np.where(red | blue)[1].max() <= 30  # Only the 20 px legend samples, no curves behind them.
    model = C._model(snapshots[0])
    clone = C._clone_model(model, model["canvas"], model["ranges"])
    try:
        color = clone.getPlotItem().legend.items[0][1].opts["color"]
        assert list(color.getRgb()) == model["legend"]["entries"][0]["color"]
    finally:
        clone.close()
        clone.deleteLater()


def test_publication_controls_invalidate_preview_and_move_spans_with_panel(entries, qtbot):
    dialog = FigureComposerDialog(entries)
    qtbot.addWidget(dialog)
    dialog._columns.setValue(3)
    dialog._table.item(0, 3).setText("2")
    dialog._table.item(0, 4).setText("2")
    dialog._spacing.setValue(32)
    dialog._label_size.setValue(26)
    dialog.update_preview()
    assert dialog.snapshot is not None
    assert dialog.snapshot.config()["panel_spans"] == [[2, 2], [1, 1]]
    dialog._table.selectRow(0)
    dialog._move(1)
    assert dialog.snapshot is None
    assert dialog._table.item(1, 3).text() == "2"
    dialog.update_preview()
    assert dialog.snapshot.config()["panel_spans"] == [[1, 1], [2, 2]]
    dialog._shared_legend.setChecked(True)
    assert dialog.snapshot is None
    dialog.update_preview()
    assert dialog.snapshot is None and not dialog._save_btn.isEnabled()
    assert "Cartesian model" in dialog._message.text()
