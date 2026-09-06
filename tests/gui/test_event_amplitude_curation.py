"""Explicit amplitude choices and drag edits cannot silently change detector science."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.event_amplitude import AMPLITUDE_COLUMNS, amplitude_is_current
from nanoscope.analysis.events import EventSet, SCHEMA
from nanoscope.analysis.modules import DetectSchmitt, detection_signal
from nanoscope.analysis.reconcile import CONSENSUS_MODULE_ID
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_inspector import EventInspectorWindow
from nanoscope.gui.main_window import MainWindow


def _synth_doc(tmp_path):
    rng = np.random.default_rng(4)
    data = np.full(100_000, 100.0) + rng.normal(0, 1.5, 100_000)
    for start, end in ((10000, 10300), (30000, 30250), (50000, 50400), (70000, 70200)):
        data[start:end] -= 30
    data = data.astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "amplitude.dat", format_name="synthetic",
                         n_samples=len(data), sample_rate_hz=100_000.0,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(data,))


def _session_with_doc(tmp_path):
    session = Session()
    return session, session.add_document(_synth_doc(tmp_path)), 0


def _events_for(session, doc_id, channel):
    ch = session.calibrated(doc_id, channel)
    df, prov = detect_events(ch.data, ch.sample_rate_hz, min_dwell_s=5e-4)
    return EventSet(df, prov)


def _inspector(qtbot, session):
    inspector = EventInspectorWindow(session)
    qtbot.addWidget(inspector)
    return inspector


def _levels_for(session, doc_id, channel, events):
    from nanoscope.analysis.modules import fit_levels
    ch = session.calibrated(doc_id, channel)
    return fit_levels(events.df, ch.data, ch.sample_rate_hz, doc_id=doc_id, channel=channel,
                      baseline=100.0, polarity=int(events.provenance["polarity"]),
                      provenance=dict(events.provenance)).payload


builders = SimpleNamespace(FS=100_000.0, _synth_doc=_synth_doc,
                           _session_with_doc=_session_with_doc, _events_for=_events_for,
                           _inspector=_inspector, _levels_for=_levels_for)


@pytest.fixture
def curated(qapp, qtbot, tmp_path):
    session, doc_id, ch = builders._session_with_doc(tmp_path)
    events = builders._events_for(session, doc_id, ch)
    inspector = builders._inspector(qtbot, session)
    inspector.set_events(doc_id, ch, events.df, provenance=dict(events.provenance))
    return inspector, events


def test_opening_and_navigating_materializes_no_amplitude(curated):
    inspector, events = curated
    before = events.df.copy(deep=True)
    inspector.next_event()
    inspector.prev_event()
    pd.testing.assert_frame_equal(events.df, before)
    assert not set(AMPLITUDE_COLUMNS).intersection(events.df)
    row = events.df.iloc[0]
    polarity = events.provenance["polarity"]
    assert inspector._level_line.value() == pytest.approx(row.baseline_pA + polarity * row.depth_pA)
    assert not inspector._level_line.movable


@pytest.mark.parametrize("method", ["median", "mean", "peak", "mode"])
def test_apply_method_preserves_every_existing_column(curated, method):
    inspector, events = curated
    before = events.df.copy(deep=True)
    controls = inspector._amplitude
    controls.method.setCurrentIndex(controls.method.findData(method))
    assert controls.apply_current(), controls.note.text()
    row = events.df.iloc[0]
    assert row.amplitude_method == method
    assert amplitude_is_current(row)
    pd.testing.assert_frame_equal(events.df[list(before)], before)
    context = controls.context(row)
    start, end = int(row.start_idx), int(row.end_idx)
    base = context.baseline if np.ndim(context.baseline) == 0 else context.baseline[start:end]
    dev = context.polarity * (context.current[start:end] - base)
    if method in ("median", "mean", "peak"):
        reduce = {"median": np.median, "mean": np.mean, "peak": np.max}[method]
        assert row.amplitude_pA == pytest.approx(reduce(dev))
    assert inspector._level_line.value() == pytest.approx(row.amplitude_level_pA)


def test_drag_release_is_manual_amplitude_only_and_keeps_zoom(curated):
    inspector, events = curated
    before = events.df.copy(deep=True)
    controls = inspector._amplitude
    edits, boundaries = [], []
    inspector.amplitudeEdited.connect(edits.append)
    inspector.eventEdited.connect(boundaries.append)
    inspector._plot.setXRange(0.099, 0.104, padding=0)
    inspector._plot.setYRange(50, 110, padding=0)
    view = inspector._plot.getViewBox().viewRange()
    controls.edit_level.setChecked(True)
    assert inspector._level_line.movable
    target = float(before.iloc[0].baseline_pA) - 45.0
    inspector._level_line.setPos(target)
    assert not set(AMPLITUDE_COLUMNS).intersection(events.df)
    inspector._level_line.sigPositionChangeFinished.emit(inspector._level_line)
    assert edits == [int(before.iloc[0].event_id)]
    assert boundaries == []
    assert events.df.iloc[0].amplitude_method == "manual"
    assert events.df.iloc[0].amplitude_pA == pytest.approx(45.0, abs=1e-4)
    assert inspector._level_line.movable
    np.testing.assert_allclose(inspector._plot.getViewBox().viewRange(), view)
    pd.testing.assert_frame_equal(events.df[list(before)], before)


def test_numeric_level_apply_and_invalid_polarity_drag_are_atomic(curated):
    inspector, events = curated
    controls = inspector._amplitude
    controls.edit_level.setChecked(True)
    controls.level.setValue(float(events.df.iloc[0].baseline_pA) - 22)
    assert controls.apply_current()
    saved = events.df.copy(deep=True)
    inspector._level_line.setPos(float(events.df.iloc[0].baseline_pA) + 25)
    inspector._level_line.sigPositionChangeFinished.emit(inspector._level_line)
    pd.testing.assert_frame_equal(events.df, saved)
    assert "not changed" in controls.note.text()
    assert inspector._level_line.value() == pytest.approx(saved.iloc[0].amplitude_level_pA)


def test_numeric_level_outside_waveform_stays_visible_without_changing_x(curated):
    inspector, events = curated
    controls = inspector._amplitude
    inspector._plot.setXRange(0.099, 0.104, padding=0)
    inspector._plot.setYRange(65, 110, padding=0)
    x_before = inspector._plot.getViewBox().viewRange()[0]
    controls.edit_level.setChecked(True)
    controls.level.setValue(40.0)
    assert controls.apply_current()
    x_range, y_range = inspector._plot.getViewBox().viewRange()
    np.testing.assert_allclose(x_range, x_before)
    assert y_range[0] < 40 < y_range[1]
    inspector.next_event()
    inspector.prev_event()
    y_range = inspector._plot.getViewBox().viewRange()[1]
    assert y_range[0] < 40 < y_range[1]


def test_switching_away_from_manual_relocks_red_line(curated):
    inspector, _events = curated
    controls = inspector._amplitude
    controls.edit_level.setChecked(True)
    assert inspector._level_line.movable
    controls.method.setCurrentIndex(controls.method.findData("selected_mean"))
    assert not controls.edit_level.isChecked()
    assert not inspector._level_line.movable
    assert controls.manual_row.isHidden()


def test_selected_interval_is_separate_from_event_mask(curated):
    inspector, events = curated
    controls = inspector._amplitude
    original = events.df[list(SCHEMA)].copy(deep=True)
    row = events.df.iloc[0]
    controls.method.setCurrentIndex(controls.method.findData("selected_mean"))
    controls.start.setValue(int(row.start_idx) + 20)
    controls.end.setValue(int(row.end_idx) - 20)
    assert controls.apply_current()
    result = events.df.iloc[0]
    assert result.amplitude_start_idx == int(row.start_idx) + 20
    assert result.amplitude_end_idx == int(row.end_idx) - 20
    assert json.loads(result.amplitude_provenance_json)["interval_scope"] == "selected_interval"
    pd.testing.assert_frame_equal(events.df[list(SCHEMA)], original)


def test_fitted_state_selection_does_not_refit_or_modify_levels(curated):
    inspector, events = curated
    levels = builders._levels_for(inspector._session, inspector._doc_id, inspector._channel, events)
    before = levels.copy(deep=True)
    inspector.set_levels(levels)
    controls = inspector._amplitude
    controls.method.setCurrentIndex(controls.method.findData("selected_median"))
    assert controls.state.count() > 1
    controls.state.setCurrentIndex(1)
    expected = controls.state.currentData()
    assert controls.apply_current()
    row = events.df.iloc[0]
    assert (row.amplitude_start_idx, row.amplitude_end_idx) == expected[:2]
    assert json.loads(row.amplitude_provenance_json)["signal_identity"]["selected_state"] == expected[2]
    pd.testing.assert_frame_equal(levels, before)


def test_adjusting_fitted_state_bounds_drops_state_identity(curated):
    inspector, events = curated
    inspector.set_levels(builders._levels_for(
        inspector._session, inspector._doc_id, inspector._channel, events))
    controls = inspector._amplitude
    controls.method.setCurrentIndex(controls.method.findData("selected_mean"))
    controls.state.setCurrentIndex(1)
    assert controls._selected_state is not None
    controls.start.setValue(controls.start.value() + 1)
    assert controls.apply_current()
    audit = json.loads(events.df.iloc[0].amplitude_provenance_json)
    assert "selected_state" not in audit["signal_identity"]


def test_reset_clears_only_optional_values_and_navigation_keeps_choices(curated):
    inspector, events = curated
    before = events.df.copy(deep=True)
    assert inspector._amplitude.apply_current(method="mean")
    inspector.next_event()
    assert inspector._amplitude.method.currentData() == "median"
    inspector.prev_event()
    assert inspector._amplitude.method.currentData() == "mean"
    assert inspector._amplitude.reset_current()
    assert events.df.iloc[0][list(AMPLITUDE_COLUMNS)].isna().all()
    pd.testing.assert_frame_equal(events.df[list(before)], before)


def test_mask_resize_invalidates_optional_amplitude(curated):
    inspector, events = curated
    assert inspector._amplitude.apply_current(method="peak")
    row = events.df.iloc[0]
    raw = inspector._session.calibrated(inspector._doc_id, inspector._channel).data
    inspector._apply_edited_bounds(0, int(row.start_idx) + 3, int(row.end_idx) - 3, raw)
    assert events.df.iloc[0].amplitude_status == "stale"
    assert np.isnan(events.df.iloc[0].amplitude_pA)
    assert "stale" in inspector._amplitude.note.text()


def test_stale_and_reconciled_tables_cannot_edit_level(curated):
    inspector, events = curated
    before = events.df.copy(deep=True)
    inspector.set_stale_columns(("depth_pA",), "changed baseline")
    assert not inspector._amplitude.apply_current(method="mean")
    assert not inspector._level_line.movable
    assert not inspector._level_line.isVisible()
    inspector.set_stale_columns(())
    inspector.set_events(inspector._doc_id, inspector._channel, events.df,
                         CONSENSUS_MODULE_ID, dict(events.provenance))
    assert not inspector._amplitude.apply_current(method="mean")
    pd.testing.assert_frame_equal(events.df, before)


def test_waveform_replays_filter_and_red_line_uses_same_domain(qapp, qtbot, tmp_path):
    session, doc_id, ch = builders._session_with_doc(tmp_path)
    raw = session.calibrated(doc_id, ch).data
    df, prov = detect_events(raw, builders.FS, filter_cutoff_hz=2000, min_dwell_s=5e-4)
    inspector = builders._inspector(qtbot, session)
    inspector.set_events(doc_id, ch, df, provenance=prov)
    expected = detection_signal(raw, builders.FS, prov)
    x, y = inspector._curve.getData()
    lo = inspector._win_lo
    np.testing.assert_array_equal(y, expected[lo:lo + len(y)])
    assert not np.array_equal(y, raw[lo:lo + len(y)])
    assert inspector._amplitude.apply_current(method="mean")
    assert inspector._baseline_line.value() - inspector._level_line.value() == pytest.approx(df.iloc[0].amplitude_pA)


def test_normalized_measurement_and_saved_amplitude_are_dimensionless(
        qapp, qtbot, tmp_path, context_factory):
    session, doc_id, ch = builders._session_with_doc(tmp_path)
    raw = session.calibrated(doc_id, ch).data
    result = DetectSchmitt().run(context_factory(
        raw, fs=builders.FS,
        params={"min_dwell_s": 5e-4, "normalize_enabled": True, "normalize_mode": "robust"}))
    events = result.payload
    inspector = builders._inspector(qtbot, session)
    inspector.set_events(doc_id, ch, events.df, provenance=dict(events.provenance))
    assert inspector._units == "dimensionless"
    controls = inspector._amplitude
    assert controls.apply_current(method="mean"), controls.note.text()
    row = events.df.iloc[0]
    assert row.amplitude_units == "dimensionless"
    assert "dimensionless" in controls.level.suffix()
    context = controls.context(row)
    a, b = int(row.start_idx), int(row.end_idx)
    base = context.baseline if np.ndim(context.baseline) == 0 else context.baseline[a:b]
    expected = np.mean(context.polarity * (context.current[a:b] - base))
    assert row.amplitude_pA == pytest.approx(expected)
    audit = json.loads(row.amplitude_provenance_json)
    assert audit["signal_identity"]["processing"]["normalize_enabled"] is True


def test_apply_all_is_explicit_atomic_and_preserves_detector_columns(curated, monkeypatch):
    inspector, events = curated
    controls = inspector._amplitude
    before = events.df.copy(deep=True)
    controls.method.setCurrentIndex(controls.method.findData("mean"))
    edits = []
    inspector.amplitudeEdited.connect(edits.append)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *_: QtWidgets.QMessageBox.No)
    controls.apply_all()
    pd.testing.assert_frame_equal(events.df, before)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *_: QtWidgets.QMessageBox.Yes)
    controls.apply_all()
    assert edits == [-1]
    assert set(events.df.amplitude_method) == {"mean"}
    assert all(amplitude_is_current(row) for _, row in events.df.iterrows())
    pd.testing.assert_frame_equal(events.df[list(before)], before)


def test_wired_edit_undo_redo_and_rerun_guard(qapp, qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    doc = builders._synth_doc(tmp_path)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    events = builders._events_for(window._session, doc_id, 0)
    before = events.df.copy(deep=True)
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
        baseline_fingerprint=events.provenance["baseline_fingerprint"], payload=events))
    refits = []
    monkeypatch.setattr(window, "_refit_levels_for_event", lambda *_: refits.append(True))
    window._open_event_inspector()
    inspector = window._event_inspector
    before = events.df.copy(deep=True)  # installation adds the standard curation metadata
    assert inspector._amplitude.apply_current(method="peak"), inspector._amplitude.note.text()
    after = events.df.copy(deep=True)
    assert refits == []
    assert "amplitude_pA" in window._analysis_panel._table._model._columns
    assert window._analysis_panel.hand_work_at_stake(after)["n_amplitude_curated"] == 1
    window._undo_events()
    qtbot.wait(10)
    restored = window._analysis_panel.current_detection().payload.df
    pd.testing.assert_frame_equal(restored, before)
    window._redo_events()
    qtbot.wait(10)
    pd.testing.assert_frame_equal(window._analysis_panel.current_detection().payload.df, after)
    assert refits == []
    window._analysis_controller.abort()
