"""Retained exports disclose display units while preserving raw scientific coordinates."""
import io
import json
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from nanoscope.plot.quantity_axes import time_axis
from nanoscope.gui.matplotlib_figure_item import MatplotlibFigureItem
from nanoscope.gui.study_figure_window import _plotted_arrays
from nanoscope.gui.figure_bundle import capture_figure_snapshot


def test_retained_milliseconds_witness_survives_without_rewriting_raw_arrays(qtbot):
    figure = Figure(figsize=(3, 2), dpi=100)
    axis = figure.subplots()
    raw = np.array([.001, .002, .003])
    axis.plot(raw, [1., 2., 3.])
    basis = time_axis(axis, 'bottom', 'dwell_s', stored_unit='s', label='Dwell (ms)')
    resources = _plotted_arrays(figure)
    evidence = json.loads(resources['plotted_arrays.json'])['display_basis']
    assert evidence['axes'][0]['basis'] == basis.to_dict()
    quantity = evidence['axes'][0]['basis']['quantity']
    assert (quantity['stored_unit'], quantity['display_unit'], quantity['factor']) == ('s', 'ms', 1000.)
    with np.load(io.BytesIO(resources['plotted_arrays.npz'])) as arrays:
        np.testing.assert_array_equal(arrays['a0_l0_x'], raw)
    item = MatplotlibFigureItem(figure)
    view = QtWidgets.QGraphicsView(); qtbot.addWidget(view)
    scene = QtWidgets.QGraphicsScene(view); view.setScene(scene); scene.addItem(item)
    figure.clear()
    snapshot = capture_figure_snapshot(item, data=pd.DataFrame({'dwell_s': raw}))
    captured = snapshot.provenance()['nanoscope_capture_rendering']['renderers'][0]['display_basis']
    assert captured == evidence
    np.testing.assert_array_equal(snapshot.dataframe()['dwell_s'], raw)
