"""Optional amplitude curation survives durable storage and ordinary event history."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.event_amplitude import (
    AMPLITUDE_COLUMNS, amplitude_is_current, measure_event_amplitude, write_event_amplitude,
    invalidate_event_amplitudes,
)
from nanoscope.analysis.events import EventSet, SCHEMA
from nanoscope.analysis.scoring import write_events_export
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.history import EventHistory, _hand_curation, describe_change
from nanoscope.core.project import ProjectState, ResultRec, load_project, save_project
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta


def _frame():
    signal = np.full(4096, 100.0)
    signal[1000:1200] = 70.0
    signal[2000:2200] = 60.0
    frame = characterize(
        signal, np.full(signal.size, 100.0), np.array([1000, 2000]),
        np.array([1200, 2200]), 50_000.0, -1, noise_rms=1.0,
        recording="synthetic", pore_id="pore")
    return frame, signal


def _curate(frame, position=0):
    _, signal = _frame()
    measured = measure_event_amplitude(
        signal, 100.0, frame.iloc[position], polarity=-1, method="manual",
        units="pA", manual_level=69.0, provenance={"source": "test"})
    assert write_event_amplitude(frame, position, measured)


def test_optional_amplitude_roundtrip_keeps_frozen_schema_and_nullable_types(tmp_path):
    frame, _ = _frame()
    original = frame[list(SCHEMA)].copy(deep=True)
    _curate(frame)
    state = ProjectState(results=[ResultRec(
        "/data/synthetic.csv", 0, "detect.schmitt", frame, {"fs": 50_000.0})])
    restored = load_project(save_project(tmp_path / "amplitudes.nanoscope", state))
    back = restored.results[0].events
    assert_frame_equal(back, frame)
    assert_frame_equal(back[list(SCHEMA)], original)
    assert pd.isna(back.iloc[1]["amplitude_method"])
    assert json.loads(back.iloc[0]["amplitude_provenance_json"])["estimator_version"] == 1
    assert amplitude_is_current(back.iloc[0])


def test_legacy_project_does_not_materialize_amplitude_choices(tmp_path):
    frame, _ = _frame()
    state = ProjectState(results=[ResultRec(
        "/data/synthetic.csv", 0, "detect.schmitt", frame, {})])
    back = load_project(save_project(tmp_path / "legacy.nanoscope", state)).results[0].events
    assert not set(AMPLITUDE_COLUMNS).intersection(back.columns)
    assert_frame_equal(EventSet(back).df, frame)


def test_event_csv_exports_amplitude_and_its_provenance_without_changing_depth(tmp_path):
    frame, _ = _frame()
    _curate(frame)
    paths = write_events_export(frame, tmp_path / "events.csv", provenance={"units": "pA"})
    back = pd.read_csv(paths["data"])
    sidecar = json.loads(paths["meta"].read_text())
    assert set(AMPLITUDE_COLUMNS) <= set(back.columns)
    assert set(AMPLITUDE_COLUMNS) <= set(sidecar["columns"])
    assert back.iloc[0]["depth_pA"] == frame.iloc[0]["depth_pA"]
    assert back.iloc[0]["amplitude_pA"] == 31.0
    assert json.loads(back.iloc[0]["amplitude_provenance_json"])["signal_identity"]["source"] == "test"
    assert amplitude_is_current(back.iloc[0])


def test_first_amplitude_choice_has_specific_label_and_counts_as_hand_work():
    frame, _ = _frame()
    after = frame.copy(deep=True)
    _curate(after)
    assert describe_change(frame, after) == "Edit event amplitude"
    assert describe_change(after, frame) == "Edit event amplitude"
    assert _hand_curation(frame)["n_amplitude_curated"] == 0
    assert _hand_curation(after)["n_amplitude_curated"] == 1
    after.loc[0, "amplitude_status"] = "stale"
    assert _hand_curation(after)["n_amplitude_curated"] == 1


def test_amplitude_only_edit_undo_redo_restores_whole_optional_payload(tmp_path):
    frame, signal = _frame()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "synthetic.csv", format_name="synthetic", n_samples=signal.size,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(signal.astype(np.float32),)))
    key = (doc_id, 0, "detect.schmitt")
    result = AnalysisResult(
        module_id=key[2], module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint="synthetic", payload=EventSet(frame))
    session.add_result(result)
    history = EventHistory(session)
    assert not history.checkpoint(key)
    frozen = frame[list(SCHEMA)].copy(deep=True)
    _curate(frame)
    curated = frame[list(AMPLITUDE_COLUMNS)].copy(deep=True)
    assert history.checkpoint(key)
    assert history.undo_label(key) == "Edit event amplitude"
    assert not history.checkpoint(key), "an unchanged nullable payload must not invent undo steps"
    assert history.undo(key) == "Edit event amplitude"
    restored = session.results.get(*key).payload.df
    assert not set(AMPLITUDE_COLUMNS).intersection(restored.columns)
    assert_frame_equal(restored[list(SCHEMA)], frozen)
    assert history.redo(key) == "Edit event amplitude"
    restored = session.results.get(*key).payload.df
    assert_frame_equal(restored[list(AMPLITUDE_COLUMNS)], curated)
    assert_frame_equal(restored[list(SCHEMA)], frozen)
    assert not history.checkpoint(key)


def test_baseline_remeasure_that_only_invalidates_amplitude_is_named_correctly(tmp_path):
    frame, signal = _frame()
    _curate(frame)
    session = Session()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "reference.csv", format_name="synthetic", n_samples=signal.size,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(signal.astype(np.float32),)))
    result = AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"baseline_window_s": 0.5}, calibration_fingerprint="test",
        baseline_fingerprint="before", payload=EventSet(frame))
    session.add_result(result)
    key = (doc_id, 0, "detect.schmitt")
    history = EventHistory(session)
    assert not history.checkpoint(key)
    updated = frame.copy(deep=True)
    invalidate_event_amplitudes(updated, reason="baseline reference changed")
    session.add_result(replace(result, params={"baseline_window_s": 1.0},
                               baseline_fingerprint="after", payload=EventSet(updated)))
    assert history.checkpoint(key)
    assert history.undo_label(key) == "Re-measure depths"
    assert history.undo(key) == "Re-measure depths"
    assert amplitude_is_current(session.results.get(*key).payload.df.iloc[0])
