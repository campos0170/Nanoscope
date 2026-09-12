"""Metadata-only lifecycle, project identity, and detector snapshot integration."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt
from nanoscope.core.acquisition import AcquisitionBandwidth
from nanoscope.core.acquisition_state import (
    WORKSPACE_KEY, capture_acquisition_overrides, restore_acquisition_overrides,
)
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import (
    AnalyticalWorkspace, DocumentRef, ProjectState, load_project, save_project,
)
from nanoscope.core.segment import persistent_key
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

HEADER = AcquisitionBandwidth(5000, "reported", "file_header",
                              header_fields=(("fSignalLowpassFilter", "5000"),))
USER = AcquisitionBandwidth(2000, "reported", "user", "Bessel", 8, "Amplifier record")


def _doc(path, *, reverse=False):
    data = np.random.default_rng(813).normal(120, 1, 12000).astype(np.float32)
    for start, stop in ((2000, 2200), (6000, 6250), (10000, 10100)):
        data[start:stop] -= 30
    infos = [ChannelInfo(0, "current", "pA", ChannelKind.CURRENT, acquisition_bandwidth=HEADER),
             ChannelInfo(1, "voltage", "mV", ChannelKind.VOLTAGE)]
    arrays = [data, np.full(data.size, -100., dtype=np.float32)]
    if reverse:
        infos = [replace(info, index=i) for i, info in enumerate(reversed(infos))]
        arrays.reverse()
    return TraceDocument(
        RecordingMeta(path, "synthetic", len(data), 50000., Provenance.FILE_META, tuple(infos)),
        tuple(arrays))


def _live(tmp_path, *, reverse=False):
    session = Session()
    doc = _doc(tmp_path / "recording.dat", reverse=reverse)
    doc_id = session.add_document(doc)
    return session, doc, doc_id


def _run(session, doc_id, detector=DetectSchmitt):
    return detector().run(SimpleAnalysisContext(
        session.calibrated(doc_id, 0), session.document(doc_id).meta, {}))


def test_set_reset_is_metadata_only_and_emits_no_calibration_or_result_change(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    session.set_calibration(key, 0, ChannelCalibration(gain=1.1), write_sidecar=False)
    result = _run(session, key)
    session.add_result(result)
    before = result.payload.df.copy(deep=True)
    before_provenance = deepcopy(result.payload.provenance)
    calibrated = session.calibrated_array(key, 0)
    raw = doc.channels[0].copy()
    changes, cal_changes, result_changes = [], [], []
    session.acquisitionBandwidthChanged.connect(lambda *args: changes.append(args))
    session.calibrationChanged.connect(lambda *args: cal_changes.append(args))
    session.resultsChanged.connect(lambda *args: result_changes.append(args))
    session.set_acquisition_bandwidth(key, 0, USER)
    assert session.acquisition_bandwidth(key, 0) == USER
    assert session.calibrated(key, 0).info.acquisition_bandwidth == USER
    assert doc.meta.channels[0].acquisition_bandwidth == HEADER
    assert session.calibrated_array(key, 0) is calibrated
    assert session.results.get(key, 0, result.module_id) is result
    assert_frame_equal(result.payload.df, before, check_exact=True)
    assert result.payload.provenance == before_provenance
    np.testing.assert_array_equal(doc.channels[0], raw)
    session.set_acquisition_bandwidth(key, 0, USER)  # idempotent
    session.set_acquisition_bandwidth(key, 0, None)
    assert session.acquisition_bandwidth(key, 0) == HEADER
    assert session.acquisition_bandwidth_override(key, 0) is None
    assert len(changes) == 2
    assert cal_changes == result_changes == []
    assert list(tmp_path.iterdir()) == []  # no acquisition or calibration sidecar


@pytest.mark.parametrize("channel", [-1, True, 1.5, "0", 99])
def test_invalid_channel_cannot_create_hidden_or_misbound_override(qapp, tmp_path, channel):
    session, doc, key = _live(tmp_path)
    with pytest.raises((IndexError, ValueError, TypeError)):
        session.set_acquisition_bandwidth(key, channel, USER)
    assert session.acquisition_bandwidth(key, 0) == HEADER
    assert session.acquisition_bandwidth_override(key, 1) is None


def test_only_valid_user_objects_can_override_header(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    for value in (HEADER, {}, "unknown"):
        with pytest.raises(ValueError):
            session.set_acquisition_bandwidth(key, 0, value)
    assert session.acquisition_bandwidth(key, 0) == HEADER


def test_segment_inherits_but_does_not_link_live_parent_override(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    session.set_acquisition_bandwidth(key, 0, USER)
    segment_id = session.create_segment(key, 1000, 3000, name="event region")
    segment = session.document(segment_id)
    assert segment.meta.channels[0].acquisition_bandwidth == HEADER
    assert session.acquisition_bandwidth(segment_id, 0) == USER
    np.testing.assert_array_equal(segment.channels[0], doc.channels[0][1000:3000])
    session.set_acquisition_bandwidth(key, 0, None)
    assert session.acquisition_bandwidth(segment_id, 0) == USER
    records = capture_acquisition_overrides(session)
    assert any(r["doc_id"] == persistent_key(segment) for r in records)
    session.set_acquisition_bandwidth(key, 0, USER)
    session.close_document(key)
    with pytest.raises(KeyError):
        session.acquisition_bandwidth_override(key, 0)
    assert not any(doc_id == key for doc_id, _ch in session._acquisition_overrides)
    assert session.acquisition_bandwidth(segment_id, 0) == USER


@pytest.mark.parametrize("detector", [DetectSchmitt, DetectCusum])
def test_detector_snapshots_header_and_override_without_changing_numerical_schema(qapp, tmp_path,
                                                                                 detector):
    session, doc, key = _live(tmp_path)
    before = _run(session, key, detector)
    assert len(before.payload.df) > 0
    session.set_acquisition_bandwidth(key, 0, USER)
    after = _run(session, key, detector)
    assert_frame_equal(after.payload.df, before.payload.df, check_exact=True)
    assert after.params == before.params
    assert after.calibration_fingerprint == before.calibration_fingerprint
    assert after.baseline_fingerprint == before.baseline_fingerprint
    old = before.payload.provenance["acquisition_bandwidth"]
    new = after.payload.provenance["acquisition_bandwidth"]
    assert AcquisitionBandwidth.from_dict(old["effective"]) == HEADER
    assert AcquisitionBandwidth.from_dict(new["effective"]) == USER
    assert AcquisitionBandwidth.from_dict(new["file_header"]) == HEADER
    assert new["resolution_assessment"] == "not_assessed"
    for a, b in zip(after.overlays, before.overlays):
        np.testing.assert_array_equal(a.starts, b.starts)
        np.testing.assert_array_equal(a.stops, b.stops)


def test_project_workspace_roundtrip_remaps_by_channel_identity(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    session.set_acquisition_bandwidth(key, 0, USER)
    records = capture_acquisition_overrides(session)
    state = ProjectState(documents=[DocumentRef(persistent_key(doc))],
        workspace=AnalyticalWorkspace(extra={WORKSPACE_KEY: records}))
    path = save_project(tmp_path / "bandwidth.nanoscope", state)
    restored = load_project(path)
    other, reordered, other_id = _live(tmp_path, reverse=True)
    messages = restore_acquisition_overrides(other, other_id, restored.workspace.extra[WORKSPACE_KEY])
    assert messages  # channel moved, said out loud
    assert other.acquisition_bandwidth(other_id, 1) == USER
    assert other.acquisition_bandwidth_override(other_id, 0) is None
    assert reordered.meta.channels[1].acquisition_bandwidth == HEADER
    rerecords = capture_acquisition_overrides(other, restored.workspace.extra[WORKSPACE_KEY])
    assert len(rerecords) == 1
    assert rerecords[0]["channel"] == 1


def test_missing_recordings_are_retained_and_explicit_reset_removes_applied_override(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    session.set_acquisition_bandwidth(key, 0, USER)
    records = capture_acquisition_overrides(session)
    absent = deepcopy(records[0])
    absent["doc_id"] = str(tmp_path / "missing.dat")
    assert absent in capture_acquisition_overrides(session, [absent])
    session.set_acquisition_bandwidth(key, 0, None)
    assert capture_acquisition_overrides(session, records) == []


def test_malformed_unresolved_override_is_not_erased_on_resave(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    bad = {"doc_id": persistent_key(doc), "channel": 0, "channel_name": "current",
           "channel_kind": "current", "bandwidth": {"status": "future-schema"}}
    messages = restore_acquisition_overrides(session, key, [bad])
    assert messages
    assert session.acquisition_bandwidth(key, 0) == HEADER
    assert capture_acquisition_overrides(session, [bad]) == [bad]


def test_missing_channel_record_is_preserved_not_applied(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    session.set_acquisition_bandwidth(key, 0, USER)
    record = capture_acquisition_overrides(session)[0]
    del record["channel"]
    session.set_acquisition_bandwidth(key, 0, None)
    assert restore_acquisition_overrides(session, key, [record])
    assert session.acquisition_bandwidth_override(key, 0) is None
    assert capture_acquisition_overrides(session, [record]) == [record]


def test_malformed_extension_entries_do_not_crash_capture_and_are_retained(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    malformed = [None, "not a record", 7, {"future": True}]
    assert restore_acquisition_overrides(session, key, malformed)
    assert capture_acquisition_overrides(session, malformed) == malformed


def test_duplicate_restore_is_reported_and_does_not_override_first_entry(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    original = {"doc_id": persistent_key(doc), "channel": 0, "channel_name": "current",
                "channel_kind": "current", "bandwidth": USER.to_dict()}
    second = deepcopy(original)
    second["bandwidth"]["cutoff_hz"] = 3000
    assert "duplicate" in " ".join(restore_acquisition_overrides(session, key, [original, second]))
    assert session.acquisition_bandwidth(key, 0) == USER


def test_child_reset_to_file_header_survives_project_rederivation(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    session.set_acquisition_bandwidth(key, 0, USER)
    child_id = session.create_segment(key, 1000, 3000, name="reset child")
    child = session.document(child_id)
    session.set_acquisition_bandwidth(child_id, 0, None)
    assert session.acquisition_bandwidth(child_id, 0) == HEADER
    records = capture_acquisition_overrides(session)
    assert all(row["doc_id"] != persistent_key(child) for row in records)
    state = ProjectState(documents=[DocumentRef(persistent_key(doc))],
        workspace=AnalyticalWorkspace(extra={WORKSPACE_KEY: records}))
    saved = load_project(save_project(tmp_path / "child-reset.nanoscope", state))
    other, other_doc, other_id = _live(tmp_path)
    restore_acquisition_overrides(other, other_id, saved.workspace.extra[WORKSPACE_KEY],
                                  clear_existing=True)
    restored_child = other.create_segment(other_id, 1000, 3000, name="reset child",
                                         segment_uid=child.origin.segment_uid)
    assert other.acquisition_bandwidth(restored_child, 0) == USER  # temporary inheritance
    restore_acquisition_overrides(other, restored_child, saved.workspace.extra[WORKSPACE_KEY],
                                  match_key=child.origin.segment_uid, clear_existing=True)
    assert other.acquisition_bandwidth(restored_child, 0) == HEADER
    assert other.acquisition_bandwidth_override(restored_child, 0) is None
    assert other.acquisition_bandwidth(other_id, 0) == USER


def test_absent_legacy_section_does_not_clear_segment_inheritance(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    session.set_acquisition_bandwidth(key, 0, USER)
    child_id = session.create_segment(key, 1000, 3000, name="legacy child")
    assert restore_acquisition_overrides(session, child_id, [], clear_existing=False) == ()
    assert session.acquisition_bandwidth(child_id, 0) == USER


@pytest.mark.parametrize("unreadable", [None, {}, "unknown", 3])
def test_explicit_unreadable_section_cannot_be_silently_overwritten(qapp, tmp_path, unreadable):
    session, doc, key = _live(tmp_path)
    with pytest.raises(ValueError, match="unreadable"):
        capture_acquisition_overrides(session, unreadable)


def test_future_record_fields_are_preserved_without_applying_unknown_semantics(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    future = {"doc_id": persistent_key(doc), "channel": 0, "channel_name": "current",
              "channel_kind": "current", "bandwidth": USER.to_dict(),
              "future_transfer_function": {"cascade": ["stage1", "stage2"]}}
    assert restore_acquisition_overrides(session, key, [future])
    assert session.acquisition_bandwidth(key, 0) == HEADER
    assert capture_acquisition_overrides(session, [future]) == [future]
