"""Capture records observed renderer facts without changing older snapshot bytes."""
import json
from dataclasses import replace

import pandas as pd
import pyqtgraph as pg
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from nanoscope.gui.figure_bundle import FigureSnapshot, capture_figure_snapshot
from nanoscope.gui.matplotlib_figure_item import MatplotlibFigureItem


def test_pg_capture_records_device_facts_and_keeps_caller_provenance(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    plot.plot([1, 2], [3, 4])
    source = {"source": "original"}
    snapshot = capture_figure_snapshot(plot.getPlotItem(), data=pd.DataFrame(), provenance=source)
    facts = snapshot.provenance()["nanoscope_capture_rendering"]
    assert source == {"source": "original"} and snapshot.provenance()["source"] == "original"
    assert facts["version"] == 1 and facts["qt_version"] and facts["pyqtgraph_version"]
    assert facts["renderers"][0]["renderer"] == "Qt graphics scene"
    assert all(value > 0 for value in facts["renderers"][0]["picture_logical_dpi"])
    # Prior captures have no added field and remain exactly readable/re-exportable.
    old = replace(snapshot, provenance_json_bytes=json.dumps(source).encode())
    metadata, blobs = old.to_storage()
    assert FigureSnapshot.from_storage(metadata, blobs) == old


def test_mpl_capture_records_frozen_dpi_and_requested_family_without_live_figure(qtbot):
    figure = Figure(figsize=(3, 2), dpi=125)
    axis = figure.subplots(); axis.plot([1, 2], [3, 4])
    axis.set_title("Recorded", family="DejaVu Sans")
    item = MatplotlibFigureItem(figure)
    view = QtWidgets.QGraphicsView(); qtbot.addWidget(view)
    scene = QtWidgets.QGraphicsScene(view); view.setScene(scene); scene.addItem(item)
    figure.clear()
    snapshot = capture_figure_snapshot(item, data=pd.DataFrame())
    facts = snapshot.provenance()["nanoscope_capture_rendering"]["renderers"][0]
    assert facts["renderer"] == "matplotlib-to-QPicture" and facts["figure_dpi"] == 125
    assert "DejaVu Sans" in facts["requested_font_families"]
    assert facts["matplotlib_version"]
