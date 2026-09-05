"""Spectrum appearance edits retain exact result objects and manual ranges."""
from __future__ import annotations

import os
from unittest.mock import Mock

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtTest

from nanoscope.analysis import allpoints, psd
from nanoscope.core.analysis_workspace import capture_psd, restore_psd
from nanoscope.core.session import Session
from nanoscope.gui.psd_panel import PSDWindow, VIEWS
from .test_psd import _fed_window


@pytest.mark.parametrize("view", [key for key, _ in VIEWS])
def test_spectrum_presentation_is_not_analysis(qtbot, tmp_path, monkeypatch, view):
    window, _ = _fed_window(qtbot, tmp_path)
    window.set_view(view)
    window.show()
    qtbot.wait(10)
    plot = window.active_plot_item()
    plot.setRange(xRange=(1, 2), yRange=(1, 2), padding=0)
    qtbot.wait(10)
    ranges = np.asarray(plot.viewRange()).copy()
    arrays = (window._freqs, window._psd, window._plot_x, window._plot_y)
    copies = tuple(a.copy() for a in arrays)
    results = (window._fft, window._spec, window._hist, window._summary)
    items = (window._curve_item, window._fit_item, window._fft_item,
             window._spec_image, window._hist_item)
    automatic_title = plot.titleLabel.text
    labels = [(plot.getAxis(side).labelText, plot.getAxis(side).labelUnits)
              for side in ("bottom", "left")]
    watched = []
    for module, names in ((psd, ("welch_psd", "fft_magnitude", "spectrogram_psd")),
                          (allpoints, ("all_points_histogram", "level_peaks"))):
        for name in names:
            watcher = Mock(wraps=getattr(module, name))
            monkeypatch.setattr(module, name, watcher)
            watched.append(watcher)
    changed = QtTest.QSignalSpy(window.workspaceChanged)
    window._title_edit.setText("Edited figure")
    window._bg_combo.setCurrentText("Light")
    window._appearance_widget.title_size.setValue(18)
    window._appearance_widget.line_width.setValue(2)
    window._spec_cmap_combo.setCurrentText("magma")
    qtbot.wait(10)
    assert plot.titleLabel.text == "Edited figure"
    assert changed.count() == 5
    window._title_edit.clear()
    assert plot.titleLabel.text == automatic_title
    assert labels == [(plot.getAxis(side).labelText, plot.getAxis(side).labelUnits)
                      for side in ("bottom", "left")]
    for watcher in watched:
        watcher.assert_not_called()
    for before, expected, actual in zip(arrays, copies,
            (window._freqs, window._psd, window._plot_x, window._plot_y)):
        assert actual is before
        np.testing.assert_array_equal(actual, expected)
    for before, actual in zip(results, (window._fft, window._spec, window._hist, window._summary)):
        assert actual is before
    assert items == (window._curve_item, window._fit_item, window._fft_item,
                     window._spec_image, window._hist_item)
    np.testing.assert_allclose(plot.viewRange(), ranges, rtol=0, atol=1e-12)


def test_spectrum_appearance_roundtrip_and_orphan_edit(qtbot, tmp_path, monkeypatch):
    original, _ = _fed_window(qtbot, tmp_path)
    original.set_view("allpoints")
    original.set_appearance_state({"title_size": 18, "line_width": 2, "grid": False})
    artifact = capture_psd(original)
    restored = PSDWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(restored)
    for module, names in ((psd, ("welch_psd", "fft_magnitude", "spectrogram_psd")),
                          (allpoints, ("all_points_histogram", "level_peaks"))):
        for name in names:
            monkeypatch.setattr(module, name, Mock(side_effect=AssertionError("unexpected analysis")))
    report = restore_psd(restored, artifact)
    assert not report.skipped
    assert restored._appearance == original._appearance
    assert capture_psd(restored).payload["config"]["appearance"] == artifact.payload["config"]["appearance"]
    stored_psd = restored._psd
    stored_hist = restored._hist
    restored._title_edit.setText("Saved figure")
    assert restored._psd is stored_psd
    assert restored._hist is stored_hist
    assert restored.active_plot_item().titleLabel.text == "Saved figure"
