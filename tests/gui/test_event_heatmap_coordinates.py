"""Heatmap artwork must place counts[x_bin, y_bin] at the exported bin coordinates.

Neither global pyqtgraph image defaults nor restoring a frozen grid may transpose
the event census. Check actual rendered pixels as well as the numeric exports.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

import nanoscope  # noqa: F401 -- bootstrap Qt before GUI imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui
from pyqtgraph.exporters import ImageExporter

from nanoscope.analysis import event_shape
from nanoscope.core.analysis_workspace import capture_event_plots, restore_event_plots
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project
from nanoscope.core.session import Session
from nanoscope.gui.event_plot_panel import DENSITY, EventPlotWindow

_fed_window = runpy.run_path(str(Path(__file__).with_name("test_event_plot.py")))["_fed_window"]


@pytest.fixture(params=["row-major", "col-major"])
def image_axis_order(request):
    previous = pg.getConfigOption("imageAxisOrder")
    pg.setConfigOption("imageAxisOrder", request.param)
    try:
        yield request.param
    finally:
        pg.setConfigOption("imageAxisOrder", previous)


def _assert_grid_mapping(window, counts, x_edges, y_edges):
    image = window._density_image
    assert image.axisOrder == "col-major"
    np.testing.assert_array_equal(image.image, counts)
    np.testing.assert_array_equal(window.density_counts(), counts)
    for actual, expected in zip(window.density_edges(), (x_edges, y_edges)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(window._top_marginal_item.opts["height"], counts.sum(axis=1))
    np.testing.assert_array_equal(window._right_marginal_item.opts["width"], counts.sum(axis=0))

    # Inspect the rendered image, not just the array or the enclosing rectangle:
    # square grids can otherwise look plausible while putting the peak at (j, i).
    image.render()
    assert (image.qimage.width(), image.qimage.height()) == counts.shape
    x_plot = np.log10(x_edges) if window._logx.isChecked() else x_edges
    y_plot = np.log10(y_edges) if window._logy.isChecked() else y_edges
    i, j = np.unravel_index(np.argmax(counts), counts.shape)
    peak_color = QtGui.QColor(*[int(value) for value in image.lut[-1]])
    assert image.qimage.pixelColor(int(i), int(j)) == peak_color
    mapped = image.mapToParent(QtCore.QPointF(float(i) + 0.5, float(j) + 0.5))
    np.testing.assert_allclose(
        [mapped.x(), mapped.y()],
        [(x_plot[i] + x_plot[i + 1]) / 2, (y_plot[j] + y_plot[j + 1]) / 2],
        rtol=1e-12, atol=1e-12)

    # All exported bin identities/physical edges retain the original scientific
    # convention, independently of whether the display uses log coordinates.
    data = window._density_plot_data()
    bins = data.loc[data["part"] == "density_bin"]
    np.testing.assert_array_equal(bins["count"].to_numpy().reshape(counts.shape), counts)
    peak = bins.loc[bins["count"].idxmax()]
    assert (peak["x_bin"], peak["y_bin"]) == (i, j)
    np.testing.assert_array_equal(
        [peak["x_left"], peak["x_right"], peak["y_left"], peak["y_right"]],
        [x_edges[i], x_edges[i + 1], y_edges[j], y_edges[j + 1]])
    return (int(i), int(j)), peak_color


@pytest.mark.parametrize("log_x,log_y", [(False, False), (True, False), (True, True)])
def test_live_heatmap_pixel_coordinates_ignore_global_image_order(
        qtbot, tmp_path, image_axis_order, log_x, log_y):
    window = _fed_window(qtbot, tmp_path)
    window.set_bins(5)
    window.set_density_log(log_x, log_y)
    # Run the real kernel against the same source columns, not a transposed or
    # guessed fixture. Deliberately assert the maximum is off the diagonal.
    accepted = window._eventset.accepted.df
    counts, xe, ye = event_shape.density_2d(
        accepted["dwell_s"].to_numpy(), accepted["depth_pA"].to_numpy(),
        xbins=5, ybins=5, xlog=log_x, ylog=log_y)
    peak = np.unravel_index(np.argmax(counts), counts.shape)
    assert peak[0] != peak[1]
    _assert_grid_mapping(window, counts, xe, ye)


def test_non_square_snapshot_reopens_and_exports_at_correct_coordinates(
        qtbot, tmp_path, monkeypatch, image_axis_order):
    source = EventPlotWindow(Session())
    qtbot.addWidget(source)
    source.set_view(DENSITY)
    source.set_density_log(True, False)
    counts = np.array([[1, 0, 2, 0, 3], [0, 4, 0, 1, 0], [2, 13, 0, 0, 1]], dtype=float)
    xe = np.geomspace(1e-4, 1e-1, 4)
    ye = np.linspace(10.0, 110.0, 6)
    source._density_counts = counts.copy()
    source._density_x_edges = xe.copy()
    source._density_y_edges = ye.copy()
    source._render_workspace_snapshot()
    _assert_grid_mapping(source, counts, xe, ye)

    # Exercise actual on-disk project serialization, not only widget assignments.
    path = tmp_path / "asymmetric-heatmap.nanoscope"
    state = ProjectState(workspace=AnalyticalWorkspace(artifacts=[capture_event_plots(source)]))
    save_project(path, state)
    loaded = load_project(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Reopening/exporting the frozen heatmap ran analysis")

    for name in ("density_2d", "event_ensemble", "charge_deficit"):
        monkeypatch.setattr(event_shape, name, forbidden)
    restored = EventPlotWindow(Session())
    qtbot.addWidget(restored)
    report = restore_event_plots(restored, loaded.workspace.artifacts[0])
    assert report.skipped == ()
    peak, peak_color = _assert_grid_mapping(restored, counts, xe, ye)
    restored.resize(1300, 900)
    restored.show()
    qtbot.wait(20)

    # The real image-export path draws the same heatmap, marginals and colour bar.
    # Sample the exported PNG at the peak's known scene coordinate; asserting only
    # that the file exists would miss a visually transposed scientific figure.
    target = restored.figure_export_target()
    exporter = ImageExporter(target)
    png_path = tmp_path / "asymmetric-heatmap.png"
    assert restored.save_image(str(png_path)) == png_path
    exported = QtGui.QImage(str(png_path))
    assert not exported.isNull()
    source_rect = exporter.getSourceRect()
    # Stay within the peak bin but off axis gridlines, which are drawn over it.
    point = restored._density_image.mapToScene(QtCore.QPointF(peak[0] + 0.37, peak[1] + 0.43))
    px = round((point.x() - source_rect.left()) * exported.width() / source_rect.width())
    py = round((point.y() - source_rect.top()) * exported.height() / source_rect.height())
    assert exported.pixelColor(px, py) == peak_color
    csv_path = tmp_path / "asymmetric-heatmap.csv"
    assert restored.export_data(str(csv_path)) == csv_path
    bins = pd.read_csv(csv_path)
    np.testing.assert_array_equal(bins["count"].to_numpy().reshape(counts.shape), counts)
    np.testing.assert_array_equal(restored.density_counts(), counts)
    # Vector exports use the same ImageItem; exercise the production SVG/PDF paths.
    for extension, signature in (("svg", b"<?xml"), ("pdf", b"%PDF")):
        vector_path = tmp_path / f"asymmetric-heatmap.{extension}"
        assert restored.save_image(str(vector_path)) == vector_path
        assert vector_path.read_bytes().lstrip().startswith(signature)
