"""Previews relabel retained arrays without launching a preview or processing job."""
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg

from nanoscope.gui.baseline_panel import BaselinePanel
from nanoscope.gui.baseline_workspace import BaselineCorrectionWorkspace
from nanoscope.gui.signal_conditioning import (
    SignalConditioningWindow, _duration_text, _scaled_seconds, _seconds_text,
)


def test_baseline_preview_keeps_second_geometry_and_matches_run_window_text(qtbot):
    panel = BaselinePanel()
    workspace = BaselineCorrectionWorkspace(panel)
    qtbot.addWidget(workspace)
    time = np.array([.001, .002, .003])
    raw = np.array([-10., -20., -10.])
    workspace.set_preview(time, raw, np.full(3, -10.), raw + 10.)
    assert workspace.corrected_plot.getAxis('bottom').scale == 1000.
    assert 'ms' in workspace.corrected_plot.getAxis('bottom').labelText
    np.testing.assert_array_equal(workspace.raw_curve.getData()[0], time)
    np.testing.assert_array_equal(workspace.corrected_curve.getData()[0], time)
    panel.set_run_windows({'baseline_rough_s': .08, 'baseline_base_s': .016})
    assert 'rough 80 ms / base 16 ms' in panel.run_windows_label.text()


def test_conditioning_duration_and_response_helpers_leave_the_result_untouched():
    values = np.array([0., .000025, .003, 1.])
    expected = values.copy()
    for span in (.0001, .1, 100.):
        shown, unit = _scaled_seconds(values, span)
        assert unit == 'ms'
        np.testing.assert_array_equal(shown, values * 1000.)
        assert not np.shares_memory(shown, values)
    np.testing.assert_array_equal(values, expected)
    assert _duration_text(.000025) == '0.025 ms'
    assert _duration_text(1.) == '1e+03 ms'
    assert _duration_text(None) == '—'
    assert _seconds_text(40.) == '40000 ms'


def test_conditioning_time_item_scales_ticks_only_without_creating_a_worker(qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    item = widget.getPlotItem()
    owner = SimpleNamespace(_on_user_range=lambda *args: None, _refit_axis_labels=lambda: None)
    SignalConditioningWindow._time_plot_item(owner, item)
    curve = item.plot([.001, .002], [1., 2.])
    assert item.getAxis('bottom').scale == 1000.
    np.testing.assert_array_equal(curve.getData()[0], [.001, .002])
