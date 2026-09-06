"""Warnings stay in Details/provenance, never auto-prefixed onto figure titles."""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import voltage_dep
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.plot_appearance import PlotAppearance, apply_plot_appearance
from nanoscope.gui.subpop_window import SubpopReportWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow


@pytest.mark.parametrize("kind", ["ecdf", "survival", "box", "violin"])
def test_summary_automatic_and_legacy_titles_keep_warnings_in_details(
        kind, qtbot, monkeypatch):
    from nanoscope.gui import distribution_summary_panel as module

    warning = "Source calibration requires verification."
    window = DistributionSummaryWindow()
    qtbot.addWidget(window)
    source = PlotSource("measurements", "Measured events",
                        pd.DataFrame({"dwell_s": [1., 2., 3., 4.]}),
                        kind="manual", doc_key="measurements", current=True,
                        warnings=(warning,))
    window.set_sources([source])
    window._view_combo.setCurrentIndex(window._view_combo.findData(kind))
    window.analyze()
    title = window._plot.titleLabel.text
    assert title and "⚠" not in title and "Distribution notes" not in title
    assert warning in window.warning_text() and warning in window.details_text()
    assert window._snapshot["title"] == title
    if kind == "survival":
        assert "uncensored, not Kaplan–Meier" in title
    elif kind == "violin":
        assert "Scott KDE, observed range, equal peak widths" in title
    elif kind == "box":
        assert "linear quartiles, 1.5-IQR sample whiskers" in title
    payload, tables = window.snapshot_state()
    payload["snapshot"]["title"] = "⚠ Distribution notes · " + title
    before_payload, before_tables = copy.deepcopy(payload), copy.deepcopy(tables)

    def forbidden(*args, **kwargs):
        pytest.fail("Restoring a legacy generated title recomputed summary statistics")

    monkeypatch.setattr(module, "summarize_distribution", forbidden)
    restored = DistributionSummaryWindow()
    qtbot.addWidget(restored)
    restored.restore_state(payload, tables)
    assert restored._plot.titleLabel.text == title
    assert warning in restored.details_text()
    assert restored._snapshot["title"] == payload["snapshot"]["title"]
    assert payload == before_payload
    for key in tables:
        pd.testing.assert_frame_equal(tables[key], before_tables[key], check_exact=True)
    authored = "⚠ Distribution notes · User's chosen title"
    restored.set_appearance_state({"title": authored})
    assert restored._plot.titleLabel.text == authored
    pd.testing.assert_frame_equal(restored._plot_data(), window._plot_data(), check_exact=True)


def _helper(filename, name):
    return runpy.run_path(str(Path(__file__).with_name(filename)))[name]


@pytest.mark.parametrize("kind", ["iv", "psd", "psd-title", "density", "ensemble"])
def test_empty_state_clears_stale_auto_title_without_losing_message(kind, qtbot, monkeypatch):
    import pyqtgraph as pg
    from nanoscope.gui.event_plot_panel import EventPlotWindow
    from nanoscope.gui.psd_panel import PSDWindow

    if kind == "iv":
        window = IVWindow(Session(), defer_initial_refresh=True)
        plot, message = window._plot, window._message
    elif kind in {"psd", "psd-title"}:
        window = PSDWindow(Session(), defer_initial_refresh=True)
        plot, message = window._plot, window._message
    else:
        window = EventPlotWindow(Session())
        if kind == "density":
            plot, message = window._density_main, window._density_message
        else:
            plot, message = window._ensemble_plot, window._ensemble_message
    qtbot.addWidget(window)
    plot.setTitle("Previous automatic result")
    window.set_appearance_state({"title_size": 16})
    before = window._plot_data().copy(deep=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Empty-state presentation invoked analysis or refresh")

    for name in ("refresh", "_redraw"):
        if hasattr(window, name):
            monkeypatch.setattr(window, name, forbidden)
    text = "No usable data for this view."
    message(text)
    if kind == "psd-title":
        # Exercise the separate explicit empty-title writer as well as clear.
        plot.setTitle("Previous automatic result")
        window._apply_title(None)
    window._apply_appearance()
    assert plot.titleLabel.text == ""
    assert not plot.titleLabel.isVisible()
    assert any(isinstance(item, pg.TextItem) and item.textItem.toPlainText() == text
               for item in plot.items)
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)


def test_authored_title_reveals_hidden_row_and_reset_removes_it(qtbot):
    import pyqtgraph as pg

    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    curve = plot.plot([0., 1., 2.], [2., 4., 3.])
    plot.setTitle(None)
    plot.titleLabel.setText("")
    plot.setRange(xRange=(.2, 1.5), yRange=(2.2, 3.5), padding=0)
    previous = copy.deepcopy(plot.viewRange())
    xy = tuple(np.array(v, copy=True) for v in curve.getData())
    apply_plot_appearance(plot, PlotAppearance(title="Authored title"), plot_widget=widget)
    assert plot.titleLabel.isVisible()
    assert plot.titleLabel.text == "Authored title"
    assert plot.layout.rowMaximumHeight(0) >= 30
    apply_plot_appearance(plot, PlotAppearance(), plot_widget=widget)
    assert not plot.titleLabel.isVisible()
    assert plot.titleLabel.text == ""
    assert plot.layout.rowMaximumHeight(0) == 0
    np.testing.assert_allclose(plot.viewRange(), previous, rtol=0, atol=1e-12)
    for before, after in zip(xy, curve.getData()):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("kind", ["psd", "fft", "spectrogram", "allpoints",
                                 "iv", "summary", "subpop"])
def test_authored_titles_are_not_warning_prefixed_and_edits_do_not_analyze(
        kind, qtbot, tmp_path, monkeypatch):
    warning = "Retained warning: source calibration needs verification."
    if kind in {"psd", "fft", "spectrogram", "allpoints"}:
        window, _ = _helper("test_psd.py", "_fed_window")(qtbot, tmp_path, n=16384)
        window.set_view(kind)
        window._details.setPlainText(window.details_text() + "\n" + warning)
        summary = window._warning_summary
        plots = [window.active_plot_item()]
    elif kind == "iv":
        window = IVWindow(_helper("test_iv.py", "_ohmic_session")(tmp_path))
        qtbot.addWidget(window)
        window._details.setPlainText(window.details_text() + "\n" + warning)
        summary, plots = window._warning_summary, [window._plot]
    elif kind == "summary":
        window = DistributionSummaryWindow()
        qtbot.addWidget(window)
        source = PlotSource("measurements", "Measured events",
                            pd.DataFrame({"dwell_s": [1., 2., 3., 4.]}),
                            kind="manual", doc_key="measurements", current=True)
        window.set_sources([source])
        window.analyze()
        window._snapshot["warnings"].append(warning)
        summary, plots = window._warnings, [window._plot]
    else:
        window = SubpopReportWindow()
        qtbot.addWidget(window)
        summary, plots = window._caveat_summary, list(window.figure_export_items())
    summary.set_warnings([warning], full_text=warning)
    old_details = window.details_text()
    old_data = window._plot_data().copy(deep=True)
    before = []
    for plot in plots:
        plot.setRange(xRange=(.2, .6), yRange=(.1, .8), padding=0)
        before.append(copy.deepcopy(plot.viewRange()))

    def forbidden(*args, **kwargs):
        pytest.fail("Appearance change invoked scientific analysis or data refresh")

    for name in ("refresh", "_redraw", "analyze", "set_report"):
        if hasattr(window, name):
            monkeypatch.setattr(window, name, forbidden)
    # Even a user-authored warning symbol must be preserved verbatim: no text filter.
    title = "⚠ User-authored calibration study"
    window.set_appearance_state({"title": title, "title_size": 16})
    for plot, previous in zip(plots, before):
        assert plot.titleLabel.text == title
        np.testing.assert_allclose(plot.viewRange(), previous, rtol=0, atol=1e-12)
    assert summary.full_text() == warning
    assert warning in window.details_text()
    assert window.details_text() == old_details
    pd.testing.assert_frame_equal(window._plot_data(), old_data, check_exact=True)


@pytest.mark.parametrize("metric", ["median_depth", "n_events"])
@pytest.mark.parametrize("model", ["none", "linear"])
def test_voltage_no_fit_and_fit_keep_complete_details_and_export_metadata(
        metric, model, qtbot, tmp_path, monkeypatch):
    window = VDepWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(window)
    # Selecting the metric on an empty session does not supply synthetic points.
    window.set_metric(metric)
    points = [voltage_dep.VDepPoint(
        v, value, f"run-{i}", baseline_fingerprint=f"baseline-{i}",
        merge_state="unmerged" if i != 1 else "merged: gap 10 ms",
        calibration_fingerprint=(1., 0.), detection_warnings=(f"detector caveat {i}",),
        analyzed_duration_s=10. * (i + 1),
        voltage_source="nominal" if i == 1 else "measured", doc_id=str(i))
        for i, (v, value) in enumerate([(-.2, 10.), (-.5, 21.), (-.8, 38.)])]
    exclusion = voltage_dep.VDepExclusion("missing", "missing.dat", "voltage unknown")
    series = voltage_dep.vdep_comparability(points, metric=metric, excluded=[exclusion])
    window._points = points
    window._series = series
    window._applied_fit = model
    window._redraw()
    title = window._plot.titleLabel.text
    for forbidden in ("⚠", "MIXED", "NOT COMPARABLE", "UNCALIBRATED", "uncalibrated",
                      "UNEQUAL", "detection caveat", "NOT in this series"):
        assert forbidden not in title
    if model == "linear":
        assert "slope" in title and "R²" in title
    expected = (list(series.warnings) + list(series.duration_warnings)
                + [f"{recording}: {warning}" for recording, warning in series.detection_warnings]
                + [f"{item.recording}: {item.reason}" for item in series.excluded])
    assert series.warnings and series.detection_warnings
    assert metric != "n_events" or series.duration_warnings
    data = window._plot_data().copy(deep=True)
    assert json.loads(data.iloc[0].warnings_json) == expected
    for warning in list(series.warnings) + list(series.duration_warnings):
        assert warning in window.details_text()
    for text in ("missing.dat", "voltage unknown", "DETECTION CAVEATS", "detector caveat 0",
                 "Pooled provenance", "V source:", "nominal", "measured"):
        assert text in window.details_text()
    old_details, old_fit = window.details_text(), copy.deepcopy(window.fit())
    original_xy = tuple(np.array(v, copy=True) for v in window._scatter_item.getData())

    def forbidden(*args, **kwargs):
        pytest.fail("Export or appearance invoked voltage-series science")

    monkeypatch.setattr(voltage_dep, "vdep_fit", forbidden)
    monkeypatch.setattr(voltage_dep, "vdep_series", forbidden)
    window.set_appearance_state({"title": "Voltage study", "title_size": 17})
    pd.testing.assert_frame_equal(window._plot_data(), data, check_exact=True)
    assert window.details_text() == old_details and window.fit() == old_fit
    output = window.save_image(str(tmp_path / f"{metric}-{model}.svg"))
    svg = output.read_text(encoding="utf-8")
    assert "Voltage study" in svg
    for text in ("MIXED", "NOT COMPARABLE", "detector caveat", "UNCALIBRATED", "UNEQUAL"):
        assert text not in svg
    assert json.loads(window._plot_data().iloc[0].warnings_json) == expected
    for before, after in zip(original_xy, window._scatter_item.getData()):
        np.testing.assert_array_equal(before, after)
    window.set_appearance_state({})
    assert window._plot.titleLabel.text == title
    assert window._plot.titleLabel.isVisible() == bool(title)
