"""Noise spectrum ▸ Spectrogram: user colour limits and the log-frequency display.

Both are presentation of the spectrogram already computed: neither may call
``spectrogram_psd`` or reset the user's zoom, both are saved in the workspace snapshot,
and the log-frequency axis is a labelled DISPLAY resampling -- Export data keeps the
original linear-frequency grid. Synthetic white noise only (``test_psd``'s builder).
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from unittest.mock import Mock

import nanoscope  # noqa: F401 -- Qt bootstrap.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtTest

from nanoscope.analysis import psd as psd_mod
from nanoscope.core.analysis_workspace import capture_psd, restore_psd
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project
from nanoscope.core.session import Session
from nanoscope.gui import psd_panel
from nanoscope.gui.psd_panel import VIEW_SPECTROGRAM, PSDWindow

_fed_window = runpy.run_path(str(Path(__file__).with_name("test_psd.py")))["_fed_window"]


def _spectrogram_window(qtbot, tmp_path):
    window, _doc_id = _fed_window(qtbot, tmp_path, n=65_536)
    window.set_view(VIEW_SPECTROGRAM)
    assert window.spectrogram() is not None
    return window


def _forbid_spectrogram(monkeypatch):
    monkeypatch.setattr(psd_mod, "spectrogram_psd",
                        Mock(side_effect=AssertionError("spectrogram recomputed")))


def _caption(window) -> str:
    return window._captions[VIEW_SPECTROGRAM].toPlainText()


def _metadata(window):
    data = window._plot_data()
    return data[data["part"] == "metadata"].iloc[0], data[data["part"] == "spectrogram_cell"]


# ================================================================================================
# E4 colour limits
# ================================================================================================

def test_auto_limits_are_the_documented_percentiles(qtbot, tmp_path):
    window = _spectrogram_window(qtbot, tmp_path)
    z = np.log10(window.spectrogram().psd)
    expected = np.percentile(z[np.isfinite(z)], [2.0, 99.8])
    np.testing.assert_allclose(window._spec_image.getLevels(), expected, rtol=1e-6)
    np.testing.assert_allclose(window.spectrogram_levels(), expected)
    caption = _caption(window)
    assert "2nd–99.8th percentile of the spectrogram's cells on its original frequency grid" \
        in caption
    # The guide says the caption gives the limits' linear values in either mode.
    assert f"({10.0 ** expected[0]:.3g} … {10.0 ** expected[1]:.3g} pA²/Hz)" in caption


def test_manual_limits_relevel_without_recomputing_or_moving_the_zoom(qtbot, tmp_path, monkeypatch):
    window = _spectrogram_window(qtbot, tmp_path)
    sg = window.spectrogram()
    auto = window.spectrogram_levels()
    view = window._spec_plot.getViewBox()
    view.setRange(xRange=(0.2, 0.6), yRange=(1_000.0, 5_000.0), padding=0)
    ranges = np.asarray(view.viewRange()).copy()
    image_before = np.asarray(window._spec_image.image).copy()
    _forbid_spectrogram(monkeypatch)
    changed = QtTest.QSignalSpy(window.workspaceChanged)

    window._spec_levels_combo.setCurrentIndex(window._spec_levels_combo.findData("manual"))
    # Manual starts from the limits on screen, so the picture does not jump.
    np.testing.assert_allclose((window._spec_level_lo.value(), window._spec_level_hi.value()),
                               auto, atol=1e-3)
    assert window._spec_level_lo.isEnabled() and window._spec_level_hi.isEnabled()
    window._spec_level_lo.setValue(-3.0)
    window._spec_level_hi.setValue(-1.0)
    assert tuple(window._spec_image.getLevels()) == (-3.0, -1.0)
    assert tuple(window._spec_bar.levels()) == (-3.0, -1.0)
    np.testing.assert_allclose(view.viewRange(), ranges, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(window._spec_image.image, image_before)
    assert window.spectrogram() is sg
    assert changed.count() == 3
    assert "set by hand (0.001 … 0.1 pA²/Hz)" in _caption(window)
    meta, _cells = _metadata(window)
    assert meta["colour_level_mode"] == "manual"
    assert (meta["colour_level_low_log10_psd"], meta["colour_level_high_log10_psd"]) == (-3.0, -1.0)

    window._spec_levels_combo.setCurrentIndex(window._spec_levels_combo.findData("auto"))
    np.testing.assert_allclose(window._spec_image.getLevels(), auto, rtol=1e-6)
    assert not window._spec_level_lo.isEnabled()


def test_manual_limits_roundtrip_through_a_project(qtbot, tmp_path, monkeypatch):
    window = _spectrogram_window(qtbot, tmp_path)
    window.set_spectrogram_levels((-4.5, -0.5))
    artifact = capture_psd(window)
    config = artifact.payload["config"]
    assert config["spectrogram_levels_mode"] == "manual"
    assert (config["spectrogram_level_low"], config["spectrogram_level_high"]) == (-4.5, -0.5)
    path = tmp_path / "levels.nanoscope"
    save_project(path, ProjectState(workspace=AnalyticalWorkspace(artifacts=[artifact])))
    restored = PSDWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(restored)
    _forbid_spectrogram(monkeypatch)
    report = restore_psd(restored, load_project(path).workspace.artifacts[0])
    assert report.skipped == ()
    assert tuple(restored._spec_image.getLevels()) == (-4.5, -0.5)
    assert restored._spec_levels_combo.currentData() == "manual"


def test_high_is_always_above_low_on_screen_in_force_and_in_the_project(qtbot, tmp_path):
    window = _spectrogram_window(qtbot, tmp_path)
    lo_spin, hi_spin = window._spec_level_lo, window._spec_level_hi
    # An unordered pair keeps Low and gets High a decade above it -- in the boxes as well
    # as on the image, so the controls never disagree with the limits in force.
    window.set_spectrogram_levels((-2.0, -3.0))
    assert (lo_spin.value(), hi_spin.value()) == (-2.0, -1.0)
    assert tuple(window._spec_image.getLevels()) == (-2.0, -1.0)
    config = capture_psd(window).payload["config"]
    assert (config["spectrogram_level_low"], config["spectrogram_level_high"]) == (-2.0, -1.0)
    # The boxes bound each other, so a Low pushed past High stops below it.
    assert lo_spin.maximum() < hi_spin.value() and hi_spin.minimum() > lo_spin.value()
    lo_spin.setValue(5.0)
    assert lo_spin.value() < hi_spin.value()
    assert tuple(window._spec_image.getLevels()) == (lo_spin.value(), hi_spin.value())
    # A pair above the old High is set whole, not clamped by the old bounds.
    window.set_spectrogram_levels((2.0, 4.0))
    assert (lo_spin.value(), hi_spin.value()) == (2.0, 4.0)
    # A hand-edited snapshot with High below Low is put in order, and the restore says so.
    artifact = capture_psd(window)
    artifact.payload["config"]["spectrogram_level_low"] = 1.0
    artifact.payload["config"]["spectrogram_level_high"] = 0.5
    restored = PSDWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(restored)
    report = restore_psd(restored, artifact)
    assert any("High at or below Low" in w for w in report.warnings)
    assert (restored._spec_level_lo.value(), restored._spec_level_hi.value()) == (1.0, 2.0)
    assert tuple(restored._spec_image.getLevels()) == (1.0, 2.0)


# ================================================================================================
# E5 log-frequency display
# ================================================================================================

def test_log_rows_preserve_the_band_integral_and_skip_nonfinite_bins():
    freqs = np.arange(0.0, 129.0) * 10.0                      # 0, 10, ... 1280 Hz
    rng = np.random.default_rng(4)
    psd = rng.uniform(1.0, 5.0, (freqs.size, 7))
    edges, rows = psd_panel._log_frequency_rows(freqs, psd, 10.0, n_rows=64)
    assert edges[0] == pytest.approx(5.0) and edges[-1] == pytest.approx(1285.0)
    assert rows.shape == (64, 7)
    # Area-weighted: the integral over the rows equals the integral over bins 1..N.
    np.testing.assert_allclose((rows * np.diff(edges)[:, None]).sum(axis=0),
                               (psd[1:] * 10.0).sum(axis=0), rtol=1e-10)
    # A constant spectrum stays exactly that constant on every row.
    _e, flat = psd_panel._log_frequency_rows(freqs, np.full((freqs.size, 2), 3.0), 10.0)
    np.testing.assert_allclose(flat, 3.0)
    # A non-finite bin contributes no width; a row inside it alone is NaN, never 0.
    holed = psd.copy()
    holed[1] = np.nan
    e2, r2 = psd_panel._log_frequency_rows(freqs, holed, 10.0, n_rows=64)
    inside = (e2[1:] <= 15.0) & (e2[:-1] >= 5.0)
    assert inside.any() and np.isnan(r2[inside]).all()
    assert np.isfinite(r2[~inside]).all()
    assert psd_panel._log_frequency_rows(np.zeros(1), np.ones((1, 3)), 10.0) is None


def test_log_frequency_axis_is_a_display_resampling_with_the_export_on_the_linear_grid(
        qtbot, tmp_path, monkeypatch):
    window = _spectrogram_window(qtbot, tmp_path)
    sg = window.spectrogram()
    _meta, linear_cells = _metadata(window)
    view = window._spec_plot.getViewBox()
    view.setRange(xRange=(0.2, 0.6), yRange=(100.0, 1_000.0), padding=0)
    _forbid_spectrogram(monkeypatch)
    levels = window._spec_image.getLevels()

    window._spec_logf_check.setChecked(True)
    image = np.asarray(window._spec_image.image)
    assert image.shape == (sg.times.size, psd_panel._SPEC_LOG_ROWS)
    assert view.state["logMode"][1] is True
    rect = window._spec_image.mapRectToParent(window._spec_image.boundingRect())
    edges = window._spec_log_edges
    assert rect.top() == pytest.approx(np.log10(edges[0]))
    assert rect.bottom() == pytest.approx(np.log10(edges[-1]))
    (x0, x1), (y0, y1) = view.viewRange()
    assert (x0, x1) == pytest.approx((0.2, 0.6))
    assert (y0, y1) == pytest.approx((2.0, 3.0))               # the same 100–1000 Hz
    np.testing.assert_allclose(window._spec_image.getLevels(), levels)  # scale unchanged
    assert window.spectrogram() is sg
    caption = _caption(window)
    assert "LOG FREQUENCY AXIS — A DISPLAY RESAMPLING" in caption
    assert "0 Hz bin has no place on a log axis" in caption
    assert "display resampled" in window._spec_plot.getAxis("left").labelText
    meta, cells = _metadata(window)
    assert meta["spectrogram_frequency_axis"].startswith("log (display resampled")
    assert meta["display_log_rows"] == psd_panel._SPEC_LOG_ROWS
    # Export data is the ORIGINAL linear grid, cell for cell.
    np.testing.assert_array_equal(cells["psd_value"].to_numpy(),
                                  linear_cells["psd_value"].to_numpy())
    np.testing.assert_array_equal(cells["display_log10_psd"].to_numpy(),
                                  linear_cells["display_log10_psd"].to_numpy())
    np.testing.assert_array_equal(cells["frequency_hz"].to_numpy(),
                                  linear_cells["frequency_hz"].to_numpy())

    window._spec_logf_check.setChecked(False)
    np.testing.assert_array_equal(window._spec_image.image,
                                  np.log10(sg.psd).T.astype(np.float32))
    assert view.state["logMode"][1] is False
    (_x0, _x1), (y0, y1) = view.viewRange()
    assert (y0, y1) == pytest.approx((100.0, 1_000.0))
    assert "DISPLAY RESAMPLING" not in _caption(window)


def test_log_frequency_axis_roundtrips_through_the_snapshot(qtbot, tmp_path, monkeypatch):
    window = _spectrogram_window(qtbot, tmp_path)
    window._spec_logf_check.setChecked(True)
    artifact = capture_psd(window)
    assert artifact.payload["config"]["spectrogram_log_frequency"] is True
    restored = PSDWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(restored)
    _forbid_spectrogram(monkeypatch)
    report = restore_psd(restored, artifact)
    assert report.skipped == ()
    assert restored._spec_logf_check.isChecked()
    assert np.asarray(restored._spec_image.image).shape[1] == psd_panel._SPEC_LOG_ROWS
    assert "DISPLAY RESAMPLING" in restored._captions[VIEW_SPECTROGRAM].toPlainText()


def test_log_frequency_there_and_back_returns_to_the_fitted_view(qtbot, tmp_path, monkeypatch):
    window = _spectrogram_window(qtbot, tmp_path)
    view = window._spec_plot.getViewBox()
    fitted = np.asarray(view.viewRange()).copy()
    _forbid_spectrogram(monkeypatch)
    window._spec_logf_check.setChecked(True)
    # The whole image was on screen, so the log axis is fitted too, padded as a fit pads.
    edges = window._spec_log_edges
    (y0, y1) = view.viewRange()[1]
    assert y0 < np.log10(edges[0]) and y1 > np.log10(edges[-1])
    np.testing.assert_allclose(view.viewRange()[0], fitted[0], rtol=0, atol=1e-12)
    window._spec_logf_check.setChecked(False)
    np.testing.assert_allclose(view.viewRange(), fitted, rtol=0, atol=1e-9)
    sg = window.spectrogram()
    df = float(sg.freqs[1] - sg.freqs[0])
    assert view.viewRange()[1][0] < -df / 2.0          # the 0 Hz bin is back on screen


def test_a_time_only_zoom_survives_the_axis_switch(qtbot, tmp_path):
    window = _spectrogram_window(qtbot, tmp_path)
    view = window._spec_plot.getViewBox()
    view.enableAutoRange()
    view.setXRange(0.2, 0.6, padding=0)                 # time narrowed, frequency auto
    assert view.autoRangeEnabled()[0] is False and view.autoRangeEnabled()[1]
    window._spec_logf_check.setChecked(True)
    assert tuple(view.viewRange()[0]) == pytest.approx((0.2, 0.6))
    assert view.autoRangeEnabled()[1]                   # frequency still auto-ranges
    edges = window._spec_log_edges
    (y0, y1) = view.viewRange()[1]
    assert y0 < np.log10(edges[0]) and y1 > np.log10(edges[-1])
    window._spec_logf_check.setChecked(False)
    assert tuple(view.viewRange()[0]) == pytest.approx((0.2, 0.6))


def test_log_frequency_survives_a_spectrogram_recompute(qtbot, tmp_path):
    window = _spectrogram_window(qtbot, tmp_path)
    window._spec_logf_check.setChecked(True)
    window._spec_nperseg_combo.setCurrentText("1024")    # a NEW spectrogram, log still on
    sg = window.spectrogram()
    assert sg.freqs.size == 513
    image = np.asarray(window._spec_image.image)
    assert image.shape == (sg.times.size, psd_panel._SPEC_LOG_ROWS)
    view = window._spec_plot.getViewBox()
    assert view.state["logMode"][1] is True
    edges = window._spec_log_edges
    df = float(sg.freqs[1] - sg.freqs[0])
    assert edges[0] == pytest.approx(sg.freqs[1] - df / 2.0)
    rect = window._spec_image.mapRectToParent(window._spec_image.boundingRect())
    assert rect.top() == pytest.approx(np.log10(edges[0]))
    assert rect.bottom() == pytest.approx(np.log10(edges[-1]))
    assert "display resampled" in window._spec_plot.getAxis("left").labelText
    assert "DISPLAY RESAMPLING" in _caption(window)
    meta, cells = _metadata(window)
    assert meta["display_log_rows"] == psd_panel._SPEC_LOG_ROWS
    assert len(cells) == sg.freqs.size * sg.times.size      # the new LINEAR grid
    np.testing.assert_array_equal(np.unique(cells["frequency_hz"].to_numpy()), sg.freqs)


# ================================================================================================
# the pointer readout and the zoom history on the log-frequency axis (plot_interaction)
# ================================================================================================

def test_the_log_frequency_readout_reads_the_original_cell_not_the_display_row(qtbot, tmp_path):
    from PySide6.QtCore import QPointF
    from nanoscope.gui import plot_interaction as pi
    window = _spectrogram_window(qtbot, tmp_path)
    window._spec_logf_check.setChecked(True)
    sg, grid, image = window.spectrogram(), window._spec_grid_z, window._spec_image
    rect = image.mapRectToView(image.boundingRect())
    i = 3
    vx = rect.left() + (i + 0.5) * rect.width() / sg.times.size
    differs = 0
    for k in range(1, sg.freqs.size):
        vy = float(np.log10(sg.freqs[k]))
        value = window._spectrogram_cell_value(vx, vy)
        assert value == pytest.approx(float(grid[i, k]))
        resampled = pi.image_value(image, vx, vy)
        differs += resampled is not None and abs(resampled - float(grid[i, k])) > 1e-6
    assert differs, "no display row averaged several bins: the check would prove nothing"
    hover = window._spec_plot.scene()._nanoscope_hover
    options = next(o for plot, _ro, o in hover._entries if plot is window._spec_plot)
    k = sg.freqs.size - 3
    text = pi.readout_text(window._spec_plot, QPointF(vx, float(np.log10(sg.freqs[k]))),
                           **options)
    assert f"{float(sg.freqs[k]):.5g}" in text              # the true frequency, not log10
    assert f"{float(grid[i, k]):.4g}" in text


def test_switching_the_frequency_axis_restarts_the_zoom_history(qtbot, tmp_path):
    from nanoscope.gui import plot_interaction as pi
    window = _spectrogram_window(qtbot, tmp_path)
    view = window._spec_plot.getViewBox()
    nav = pi.navigation_for(window._spec_plot)
    view.setRange(xRange=(0.1, 0.4), yRange=(1_000.0, 5_000.0), padding=0)
    view.sigRangeChangedManually.emit([True, True])
    qtbot.wait(pi.SETTLE_MS + 100)
    assert nav.can_go_back()
    window._spec_logf_check.setChecked(True)               # Hz -> log10 Hz: new coordinates
    assert len(nav.history()) == 1 and not nav.can_go_back()
    np.testing.assert_allclose(view.viewRange()[0], (0.1, 0.4))   # the view itself is kept
    view.setRange(yRange=(3.2, 3.5), padding=0)
    view.sigRangeChangedManually.emit([True, True])
    qtbot.wait(pi.SETTLE_MS + 100)
    kept = nav.history()
    window._spec_levels_combo.setCurrentIndex(window._spec_levels_combo.findData("manual"))
    window._spec_level_lo.setValue(-3.0)                   # levels only: history kept
    qtbot.wait(pi.SETTLE_MS + 100)
    assert nav.history() == kept
