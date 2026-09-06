"""Optional amplitudes remain identifiable in tables, plots, exports and snapshots."""
import json

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.analysis.event_amplitude import invalidate_event_amplitudes
from nanoscope.core import analysis_workspace as workspace
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.sources import (
    PlotSource, amplitude_comparability, discover_plot_sources, source_comparability,
)
from nanoscope.gui.event_plot_panel import EventPlotWindow
from nanoscope.gui.panels import EventTable
from nanoscope.gui.plots_panel import PlotsWindow, _numeric_columns
from .test_event_amplitude_curation import _session_with_doc, _events_for, _inspector


@pytest.fixture
def amplitudes(qapp, qtbot, tmp_path):
    session, doc_id, channel = _session_with_doc(tmp_path)
    events = _events_for(session, doc_id, channel)
    inspector = _inspector(qtbot, session)
    inspector.set_events(doc_id, channel, events.df, provenance=dict(events.provenance))
    for event_id in events.df.event_id:
        inspector.show_event(int(event_id))
        assert inspector._amplitude.apply_current(method="mean")
    return session, doc_id, channel, events, inspector


def test_same_method_can_be_compared_but_mixed_methods_are_disclosed(amplitudes):
    _, _, _, events, inspector = amplitudes
    assert amplitude_comparability([("same", events.df)]) == ()
    inspector.show_event(0)
    assert inspector._amplitude.apply_current(method="peak")
    messages = amplitude_comparability([("mixed", events.df)])
    assert any("NOT COMPARABLE" in msg for msg in messages)
    source = PlotSource("test", "test", events.df, provenance=dict(events.provenance))
    verdict = source_comparability([source], ["amplitude_pA"])
    assert "CHECK SELECTED AMPLITUDES" in verdict.headline
    assert any("mixed estimators" in msg for msg in verdict.warnings)
    assert source_comparability([source], ["depth_pA"]).warnings == ()


def test_comparability_uses_only_the_rows_actually_plotted(amplitudes):
    _, _, _, events, inspector = amplitudes
    inspector.show_event(0)
    assert inspector._amplitude.apply_current(method="peak")
    source = PlotSource("test", "test", events.df, provenance=dict(events.provenance))
    verdict = source_comparability([source], ["amplitude_pA"], frames=[events.df.iloc[1:]])
    assert verdict.warnings == ()


def test_stale_missing_and_unaudited_values_are_never_silent(amplitudes):
    _, _, _, events, _ = amplitudes
    frame = events.df.copy(deep=True)
    invalidate_event_amplitudes(frame, [0], reason="test")
    frame.loc[1, "amplitude_provenance_json"] = "{}"
    messages = amplitude_comparability([("events", frame)])
    assert any("no selected amplitude" in msg for msg in messages)
    assert any("stale or unverified" in msg for msg in messages)
    assert any("provenance" in msg for msg in messages)
    assert np.isnan(frame.iloc[0].amplitude_pA)


def test_table_materializes_amplitude_columns_and_keeps_selection(amplitudes, qtbot):
    _, _, _, events, inspector = amplitudes
    table = EventTable()
    qtbot.addWidget(table)
    for column in tuple(events.df.columns):
        if column.startswith("amplitude_"):
            events.df.drop(columns=column, inplace=True)
    table.set_events(events)
    table.select_event(0)
    before_ids = table.selected_event_ids()
    inspector.show_event(0)
    assert inspector._amplitude.apply_current(method="mean")
    table.refresh_all()
    assert "amplitude_pA" in table._model._columns
    assert table.selected_event_ids() == before_ids
    assert table.view.isColumnHidden(table._model._columns.index("amplitude_provenance_json"))
    assert not table.view.isColumnHidden(table._model._columns.index("amplitude_pA"))
    depth = table._model._columns.index("depth_pA")
    assert "median depth" in table._model.headerData(depth, QtCore.Qt.Horizontal)
    table.set_stale_columns(("depth_pA",), "baseline changed")
    assert "amplitude_pA" in table.stale_columns


def test_numeric_pickers_expose_value_not_audit_coordinates(amplitudes):
    _, _, _, events, _ = amplitudes
    cols = _numeric_columns(events.df)
    assert "amplitude_pA" in cols
    assert {c for c in cols if c.startswith("amplitude_")} == {"amplitude_pA"}


def test_population_picker_labels_and_warns_mixed_amplitudes(amplitudes, qtbot):
    session, _, _, events, inspector = amplitudes
    inspector.show_event(0)
    assert inspector._amplitude.apply_current(method="peak")
    window = PlotsWindow(session)
    qtbot.addWidget(window)
    window.set_data({"test": events.df}, provenance={"test": dict(events.provenance)})
    window.set_histogram_column("amplitude_pA")
    assert "Selected amplitude" in window._plot.getAxis("bottom").labelText
    assert "CHECK SELECTED AMPLITUDES" in window._plot.titleLabel.text


def test_pca_and_cluster_colours_warn_about_amplitude_features(amplitudes, qtbot):
    session, _, _, events, inspector = amplitudes
    inspector.show_event(0)
    assert inspector._amplitude.apply_current(method="peak")
    window = PlotsWindow(session)
    qtbot.addWidget(window)
    window.set_data({"test": events.df}, provenance={"test": dict(events.provenance)})
    window.set_feature_selection(["amplitude_pA", "dwell_s"])
    assert window.run_pca() is not None
    assert "CHECK SELECTED AMPLITUDES" in window._plot.titleLabel.text
    window.set_cluster_method("k-means")
    window.set_cluster_k(2)
    assert window.run_clustering() is not None
    assert "mixed estimators" in window._cluster_summary()
    # Neither axis uses the amplitude, but the colours still do.
    window.set_plot_type("Scatter")
    window.set_scatter_columns("depth_pA", "dwell_s")
    window.set_colour_by("Cluster")
    assert "CHECK SELECTED AMPLITUDES" in window._plot.titleLabel.text
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    assert "CHECK SELECTED AMPLITUDES" in window._plot.titleLabel.text


def test_density_partial_amplitudes_and_orphan_snapshot_keep_warning_units(amplitudes, qtbot):
    session, doc_id, channel, events, _ = amplitudes
    invalidate_event_amplitudes(events.df, [0], reason="test")
    window = EventPlotWindow(session)
    qtbot.addWidget(window)
    window.set_source(doc_id, channel, events)
    window.set_y_metric("amplitude_pA")
    assert "Selected amplitude (pA)" in window._density_main.getAxis("left").labelText
    assert window.density_counts().sum() == len(events.df) - 1
    assert "CHECK SELECTED AMPLITUDES" in window._density_main.titleLabel.text
    assert "3 of 4" in window._density_main.titleLabel.text
    live_export = window._density_plot_data()
    source_rows = live_export[live_export.part == "source_event"]
    assert "event_amplitude_provenance_json" in source_rows
    assert json.loads(source_rows.iloc[1].event_amplitude_provenance_json)["method"] == "mean"
    artifact = workspace.capture_event_plots(window)
    orphan = EventPlotWindow(type(session)())
    qtbot.addWidget(orphan)
    result = workspace.restore_event_plots(orphan, artifact)
    assert not result.skipped
    assert "Selected amplitude (pA)" in orphan._density_main.getAxis("left").labelText
    assert orphan._amplitude_warnings == window._amplitude_warnings
    assert "3 of 4" in orphan._density_main.titleLabel.text
    exported = orphan._density_plot_data()
    assert set(exported.amplitude_units) == {"pA"}
    assert json.loads(exported.amplitude_warnings_json.iloc[0])


def test_frozen_amplitude_bins_keep_saved_units_when_another_table_is_bound(amplitudes, qtbot):
    session, doc_id, channel, events, _ = amplitudes
    window = EventPlotWindow(session)
    qtbot.addWidget(window)
    window.set_source(doc_id, channel, events)
    window.set_y_metric("amplitude_pA")
    saved = workspace.capture_event_plots(window)
    events.df["amplitude_units"] = "dimensionless"
    rebound = EventPlotWindow(session)
    qtbot.addWidget(rebound)
    rebound.set_source(doc_id, channel, events, defer_analysis=True)
    assert not workspace.restore_event_plots(rebound, saved).skipped
    assert "Selected amplitude (pA)" in rebound._density_main.getAxis("left").labelText
    exported = rebound._density_plot_data()
    assert set(exported.amplitude_units) == {"pA"}
    assert set(exported.part) == {"density_bin"}


def test_superseded_calibration_also_marks_selected_amplitude(amplitudes, qtbot):
    session, doc_id, channel, events, _ = amplitudes
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=channel,
        params={}, calibration_fingerprint=session.calibration(doc_id, channel).fingerprint(),
        payload=events))
    session.set_calibration(doc_id, channel, ChannelCalibration(gain=2, target_units="pA"),
                            write_sidecar=False)
    window = PlotsWindow(session)
    qtbot.addWidget(window)
    window.set_data(discover_plot_sources(
        session.documents(), session.results.get,
        calibration_fingerprint=lambda d, c: session.calibration(d, c).fingerprint()))
    window.set_histogram_column("amplitude_pA")
    assert "SUPERSEDED CALIBRATION" in window._plot.titleLabel.text
    assert "amplitude_pA" in window.details_text()


def test_all_stale_density_shows_reason_even_without_image(amplitudes, qtbot):
    session, doc_id, channel, events, _ = amplitudes
    invalidate_event_amplitudes(events.df, reason="test")
    window = EventPlotWindow(session)
    qtbot.addWidget(window)
    window.set_source(doc_id, channel, events)
    window.set_y_metric("amplitude_pA")
    assert "stale" in window._amplitude_warning_label.text()
    assert not window._amplitude_warning_label.isHidden()
