"""Presentation edits must not become new analyses or discard a user's view."""
from __future__ import annotations

import os
from unittest.mock import Mock

import nanoscope  # noqa: F401 -- Qt bootstrap precedes the Qt imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtTest

from nanoscope.analysis import event_shape
from nanoscope.core.session import Session
from nanoscope.gui.event_plot_panel import ENSEMBLE, EventPlotWindow

from .test_event_plot import _fed_window


@pytest.fixture
def window(qtbot, tmp_path):
    result = _fed_window(qtbot, tmp_path, n_events=7)
    result.show()
    qtbot.wait(10)
    return result


def _watch_kernels(monkeypatch):
    watchers = {}
    for name in ("density_2d", "event_ensemble", "charge_deficit"):
        watcher = Mock(wraps=getattr(event_shape, name))
        monkeypatch.setattr(event_shape, name, watcher)
        watchers[name] = watcher
    return watchers


def _assert_no_analysis(watchers):
    for watcher in watchers.values():
        watcher.assert_not_called()


def _remember_numerics(window):
    arrays = (window._density_counts, window._density_x_edges,
              window._density_y_edges, *window._ensemble)
    items = (window._density_image, window._top_marginal_item,
             window._right_marginal_item, window._events_item,
             window._mean_item, window._median_item)
    return arrays, tuple(value.copy() for value in arrays), items


def _assert_same_numerics(window, remembered):
    arrays, copies, items = remembered
    current = (window._density_counts, window._density_x_edges,
               window._density_y_edges, *window._ensemble)
    for original, copy, actual in zip(arrays, copies, current):
        assert actual is original, "Appearance must retain the frozen numeric arrays"
        np.testing.assert_array_equal(actual, copy)
    assert (window._density_image, window._top_marginal_item,
            window._right_marginal_item, window._events_item,
            window._mean_item, window._median_item) == items


@pytest.mark.parametrize("control,value,axis", [
    ("_d_title", "Custom event-count map", None),
    ("_d_xlabel", "Charge deficit (pA·s)", "bottom"),
    ("_d_ylabel", "Event depth (pA)", "left"),
    ("_cmap_combo", "magma", None),
])
def test_density_presentation_retains_results_zoom_and_source(
        window, qtbot, monkeypatch, control, value, axis):
    # ECD is deliberately on demand; typing a label must not calculate it again.
    window.set_x_metric("ecd")
    window._density_main.setRange(xRange=(-1.7, -1.2), yRange=(60, 120), padding=0)
    qtbot.wait(10)
    view = np.asarray(window._density_main.viewRange()).copy()
    auto_range = list(window._density_main.vb.state["autoRange"])
    remembered = _remember_numerics(window)
    image_array = window._density_image.image
    image_levels = np.asarray(window._density_image.getLevels()).copy()
    before_export = window._density_plot_data()
    source = (window._doc_id, window._channel, window._eventset, window._signal, window._fs)
    source_frame = window._eventset.df.copy(deep=True)
    watchers = _watch_kernels(monkeypatch)
    changed = QtTest.QSignalSpy(window.workspaceChanged)

    widget = getattr(window, control)
    if control == "_cmap_combo":
        old_lut = np.asarray(window._density_image.lut).copy()
        widget.setCurrentText(value)
    else:
        widget.setText(value)
    qtbot.wait(10)

    _assert_no_analysis(watchers)
    assert changed.count() == 1
    _assert_same_numerics(window, remembered)
    assert window._density_image.image is image_array
    np.testing.assert_array_equal(window._density_image.getLevels(), image_levels)
    np.testing.assert_array_equal(window._density_main.viewRange(), view)
    assert window._density_main.vb.state["autoRange"] == auto_range
    assert (window._doc_id, window._channel, window._eventset, window._signal, window._fs) == source
    pd.testing.assert_frame_equal(window._eventset.df, source_frame, check_exact=True)
    if control == "_cmap_combo":
        assert not np.array_equal(window._density_image.lut, old_lut)
        np.testing.assert_array_equal(
            window._colorbar.colorMap().getLookupTable(nPts=256),
            window._density_image.getColorMap().getLookupTable(nPts=256))
    elif axis:
        assert window._density_main.getAxis(axis).labelText == value
    else:
        assert window._density_main.titleLabel.text == value

    # The export schema/source identities/values remain exact; only the existing
    # explicitly presentation-related columns are expected to record this edit.
    appearance = ["title", "x_label", "y_label", "colormap"]
    pd.testing.assert_frame_equal(
        before_export.drop(columns=appearance),
        window._density_plot_data().drop(columns=appearance), check_exact=True)


def test_ensemble_title_retains_results_zoom_and_export(window, qtbot, monkeypatch):
    window.set_view(ENSEMBLE)
    qtbot.wait(10)
    window._ensemble_plot.setRange(xRange=(0.1, 0.7), yRange=(870, 980), padding=0)
    remembered = _remember_numerics(window)
    view = np.asarray(window._ensemble_plot.viewRange()).copy()
    original_title = window._ensemble_plot.titleLabel.text
    before_export = window._ensemble_plot_data()
    watchers = _watch_kernels(monkeypatch)
    changed = QtTest.QSignalSpy(window.workspaceChanged)

    window._e_title.setText("Accepted event waveforms")
    qtbot.wait(10)
    assert window._ensemble_plot.titleLabel.text == "Accepted event waveforms"
    window._e_title.clear()
    qtbot.wait(10)
    assert window._ensemble_plot.titleLabel.text == original_title
    _assert_no_analysis(watchers)
    assert changed.count() == 2
    _assert_same_numerics(window, remembered)
    np.testing.assert_array_equal(window._ensemble_plot.viewRange(), view)
    pd.testing.assert_frame_equal(before_export, window._ensemble_plot_data(), check_exact=True)


def test_saved_orphan_snapshot_can_be_styled_without_reanalysis(window, qtbot, monkeypatch):
    arrays = _remember_numerics(window)[0]
    window.set_source(None, None, None, defer_analysis=True)
    window._density_counts, window._density_x_edges, window._density_y_edges = arrays[:3]
    window._ensemble = tuple(arrays[3:])
    # This is saved metadata, not a count that may be inferred from filtered bins.
    window._nanoscope_workspace_accepted_event_count = 11
    watchers = _watch_kernels(monkeypatch)
    window._render_workspace_snapshot()
    remembered = _remember_numerics(window)
    density_title = window._density_main.titleLabel.text
    ensemble_title = window._ensemble_plot.titleLabel.text
    assert "n = 11 events" in density_title
    assert "4 edge-skipped" in ensemble_title
    changed = QtTest.QSignalSpy(window.workspaceChanged)

    window._d_title.setText("Saved density")
    window._d_title.clear()
    window._d_xlabel.setText("Saved horizontal axis")
    window.set_colormap("cividis")
    window._e_title.setText("Saved ensemble")
    window._e_title.clear()
    qtbot.wait(10)

    _assert_no_analysis(watchers)
    assert changed.count() == 6
    _assert_same_numerics(window, remembered)
    assert window._density_main.titleLabel.text == density_title
    assert window._ensemble_plot.titleLabel.text == ensemble_title
    assert window._eventset is None and window._doc_id is None


def test_empty_state_edits_keep_message_items(qtbot, monkeypatch):
    window = EventPlotWindow(Session())
    qtbot.addWidget(window)
    density_items = tuple(window._density_main.items)
    ensemble_items = tuple(window._ensemble_plot.items)
    watchers = _watch_kernels(monkeypatch)
    changed = QtTest.QSignalSpy(window.workspaceChanged)

    window._d_title.setText("A title for later")
    window._d_xlabel.setText("An axis label for later")
    window.set_colormap("magma")
    window._e_title.setText("Another title for later")

    _assert_no_analysis(watchers)
    assert changed.count() == 4
    assert tuple(window._density_main.items) == density_items
    assert tuple(window._ensemble_plot.items) == ensemble_items
    assert window._density_counts is None and window._ensemble is None


def test_analysis_controls_still_recompute_the_corresponding_result(window, monkeypatch):
    watchers = _watch_kernels(monkeypatch)
    changed = QtTest.QSignalSpy(window.workspaceChanged)
    window.set_bins(24)
    assert watchers["density_2d"].call_count == 1
    watchers["event_ensemble"].assert_not_called()
    assert window._density_counts.shape == (24, 24)
    assert window._density_counts.sum() == 7

    window.set_n_points(80)
    assert watchers["density_2d"].call_count == 1
    assert watchers["event_ensemble"].call_count == 1
    assert window._ensemble[1].shape == (7, 80)
    assert changed.count() == 2


def test_workspace_restore_guard_suppresses_presentation_signals(window, monkeypatch):
    watchers = _watch_kernels(monkeypatch)
    changed = QtTest.QSignalSpy(window.workspaceChanged)
    original_title = window._density_main.titleLabel.text
    window._updating = True
    try:
        window._d_title.setText("Restored density title")
        window._e_title.setText("Restored ensemble title")
        window.set_colormap("plasma")
    finally:
        window._updating = False
    assert changed.count() == 0
    assert window._density_main.titleLabel.text == original_title
    window._render_workspace_snapshot()
    _assert_no_analysis(watchers)
    assert window._density_main.titleLabel.text == "Restored density title"
    assert window._ensemble_plot.titleLabel.text == "Restored ensemble title"
