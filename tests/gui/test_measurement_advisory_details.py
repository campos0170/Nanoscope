"""Stored-run QC and acquisition facts stay readable without changing measurements."""
from __future__ import annotations

import os
import runpy
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import nanoscope  # noqa: F401 -- Qt bootstrap.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.measurement_quality import build_measurement_quality
from nanoscope.core.acquisition import acquisition_snapshot_messages
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.session import Session
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.analysis_worker import AnalysisController
from nanoscope.gui.trace_view import TraceView

_helpers = runpy.run_path(str(Path(__file__).with_name("test_event_plot.py")))


@pytest.fixture
def panel(qapp, qtbot, tmp_path):
    app = QtWidgets.QApplication.instance()
    org, name = app.organizationName(), app.applicationName()
    app.setOrganizationName("NanoScopeTest")
    app.setApplicationName("MeasurementAdvisoryDetails")
    QtCore.QSettings().clear()
    session = Session()
    trace = TraceView(session)
    qtbot.addWidget(trace)
    controller = AnalysisController(session)
    panel = AnalysisPanel(session, controller, trace)
    qtbot.addWidget(panel)
    doc, events = _helpers["_doc_and_events"](tmp_path, n_events=3)
    doc_id = session.add_document(doc)
    session.add_layer(doc_id, 0)
    panel.set_selection(doc_id, 0)
    events.provenance.update(baseline_method="constant", polarity=1,
                             polarity_evidence={"verdict": "down"})
    events.provenance["measurement_quality"] = build_measurement_quality(
        doc.channels[0], np.full(doc.meta.n_samples, _helpers["BASE"]), events.df,
        doc.meta.sample_rate_hz, events.provenance)
    result = AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=events,
        warnings=("SENTINEL: retain this unrelated detector caveat",))
    session.add_result(result)
    try:
        yield panel, session, result
    finally:
        controller.abort()
        QtCore.QSettings().clear()
        app.setOrganizationName(org)
        app.setApplicationName(name)


def test_summary_preserves_recorded_warnings_and_unknown_acquisition_without_mutation(panel):
    panel, _session, result = panel
    before = result.payload.df.copy(deep=True)
    provenance = deepcopy(result.payload.provenance)
    panel._show_result_advisories(result)
    shown = panel._stats.warnings_text()
    assert "SENTINEL" in shown
    assert "Advisory checks only" in shown
    assert "Accepted is a separate curation state" in shown
    assert "Acquisition bandwidth was not recorded" in shown
    assert "Acquisition bandwidth was not recorded" in panel._result_notice.toolTip()
    assert not panel._result_notice.isHidden()
    pd.testing.assert_frame_equal(before, result.payload.df, check_exact=True)
    assert provenance == result.payload.provenance


def test_summary_refresh_keeps_acceptance_separate_and_marks_measurement_edits_stale(panel):
    panel, _session, result = panel
    before = panel._stats.warnings_text()
    result.payload.df.loc[0, "accepted"] = False
    panel._refresh_summary_from_table()
    assert panel._stats.warnings_text() == before
    result.payload.df.loc[0, "baseline_pA"] += 1.0
    panel._refresh_summary_from_table()
    shown = panel._stats.warnings_text()
    assert "Measurement quality is stale" in shown
    assert "Existing acceptance is unchanged" in shown
    assert "SENTINEL" in shown
    assert not result.payload.df.loc[0, "accepted"]


def test_clearing_recorded_warnings_does_not_suppress_quality_or_acquisition_details(panel):
    panel, session, result = panel
    session.add_result(replace(result, warnings=()))
    shown = panel._stats.warnings_text()
    assert "SENTINEL" not in shown
    assert "Advisory checks only" in shown
    assert all(message in panel._result_notice.toolTip()
               for message in acquisition_snapshot_messages(result.payload.provenance))
