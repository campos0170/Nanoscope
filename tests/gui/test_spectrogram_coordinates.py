"""Spectrogram pixels must retain time-X/frequency-Y under application image defaults."""
from __future__ import annotations

import os
import runpy
from pathlib import Path

import nanoscope  # noqa: F401 -- Qt bootstrap.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore

from nanoscope.core.analysis_workspace import capture_psd, restore_psd
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project
from nanoscope.core.session import Session
from nanoscope.gui.psd_panel import PSDWindow, VIEW_SPECTROGRAM

_fed_window = runpy.run_path(str(Path(__file__).with_name("test_psd.py")))["_fed_window"]


def _assert_time_frequency_mapping(window):
    sg = window.spectrogram()
    image = window._spec_image
    assert sg.times.size != sg.freqs.size
    assert image.axisOrder == "col-major"
    np.testing.assert_array_equal(image.image, np.log10(sg.psd).T.astype(np.float32))
    image.render()
    assert (image.qimage.width(), image.qimage.height()) == (sg.times.size, sg.freqs.size)
    # Check a known off-diagonal physical cell in the actual QImage. This remains
    # valid even when multiple PSD cells saturate at the chosen display percentiles.
    t_index, f_index = 1, sg.freqs.size // 3
    mapped = image.mapToParent(QtCore.QPointF(t_index + 0.5, f_index + 0.5))
    np.testing.assert_allclose([mapped.x(), mapped.y()],
                               [sg.times[t_index], sg.freqs[f_index]], rtol=1e-12)
    data = window._plot_data()
    cell = data[(data["part"] == "spectrogram_cell")
                & (data["time_index"] == t_index) & (data["frequency_index"] == f_index)].iloc[0]
    assert cell["psd_value"] == sg.psd[f_index, t_index]
    return sg.psd.copy()


@pytest.mark.parametrize("order", ["row-major", "col-major"])
def test_spectrogram_live_and_reopened_grid_ignore_global_image_order(qtbot, tmp_path, order):
    previous = pg.getConfigOption("imageAxisOrder")
    pg.setConfigOption("imageAxisOrder", order)
    try:
        source, _doc_id = _fed_window(qtbot, tmp_path, n=16_384)
        source.set_view(VIEW_SPECTROGRAM)
        expected = _assert_time_frequency_mapping(source)
        state = ProjectState(workspace=AnalyticalWorkspace(artifacts=[capture_psd(source)]))
        path = tmp_path / "spectrogram.nanoscope"
        save_project(path, state)
        restored = PSDWindow(Session())
        qtbot.addWidget(restored)
        report = restore_psd(restored, load_project(path).workspace.artifacts[0])
        assert report.skipped == ()
        np.testing.assert_array_equal(_assert_time_frequency_mapping(restored), expected)
    finally:
        pg.setConfigOption("imageAxisOrder", previous)
