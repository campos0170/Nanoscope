"""Curated workspace geometry/decisions enter study science without re-detection."""
from dataclasses import asdict, replace
from copy import deepcopy
import threading

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import incubation_detection as reference
from nanoscope.analysis import manual_detection as manual
from nanoscope.analysis.cancel import OperationCancelled


@pytest.fixture
def recording():
    fs, n = 10_000., 5001
    detrended = np.resize(np.array([-.15, -.1, 0, .1, .15], np.float32), n)
    detrended[100:140] = -2.
    detrended[400:440] = -.1 + np.linspace(-.02, .02, 40, dtype=np.float32)
    raw = -200 - np.arange(n) * .01 + detrended
    return reference.PreparedRecording("zn_30h", raw, detrended, fs, n / fs,
        {"tag": "zn_30h", "event_polarity": 1, "qc_mad_noise_pA": .1},
        {"event_polarity": 1, "preprocessing": "synthetic prepared input"})


@pytest.fixture
def events():
    return pd.DataFrame(dict(event_id=[18, 7], start_idx=[400, 100], end_idx=[440, 140],
        accepted=[True, False], depth_pA=[17.5, 22.5], flag=["molecule=A", "flagged"],
        review=["reviewed", "reviewed"], original_start_idx=[399, 99],
        original_end_idx=[441, 141]))


def measured(recording, events, **kwargs):
    return manual.measure_imported_events(recording, events, manual.MeasurementSettings(),
        source_provenance={"source_key": "source-A", "module_id": "detection"}, **kwargs)


def test_native_exclusive_bounds_and_curation_remain_exact(recording, events):
    before = events.copy(deep=True)
    result = measured(recording, events)
    table = result.catalog
    assert table.source_event_id.tolist() == [7, 18]
    assert table.event_id.tolist() == ["zn_30h:source:7", "zn_30h:source:18"]
    assert table.start_idx.tolist() == [100, 400]
    assert table.end_idx.tolist() == [139, 439]
    assert table.dwell_ms.tolist() == [4., 4.]
    assert table.onset_s.tolist() == [.01, .04]
    assert table.end_s.tolist() == [.014, .044]
    assert table.accepted.tolist() == [False, True]
    assert table.passes_gate.tolist() == [True, False]
    assert table.source_depth_pA.tolist() == [22.5, 17.5]
    assert not np.array_equal(table.depth_pA, table.source_depth_pA)
    assert table.source_flag.tolist() == ["flagged", "molecule=A"]
    assert table.source_original_end_idx.tolist() == [141, 441]
    assert result.provenance["accepted_count"] == 1
    assert "model" not in result.provenance
    assert "decode_settings" not in result.provenance
    assert result.provenance["input_mode"] == "imported_events"
    pd.testing.assert_frame_equal(events, before)


def test_adjacent_single_sample_and_last_sample_events(recording):
    native = pd.DataFrame(dict(event_id=[1, 2, 3], start_idx=[0, 1, 5000],
                              end_idx=[1, 2, 5001], accepted=[True] * 3))
    table = measured(recording, native).catalog
    assert table.end_idx.tolist() == [0, 1, 5000]
    assert table.dwell_ms.tolist() == [.1] * 3
    assert table.end_s.iloc[-1] == recording.duration_s


@pytest.mark.parametrize("column,value,message", [
    ("event_id", [7, 7], "unique"),
    ("event_id", [np.nan, 7], "integer"),
    ("event_id", [np.inf, 7], "integer"),
    ("event_id", [1.2, 7], "integer"),
    ("event_id", [True, False], "integer"),
    ("event_id", [-1, 7], "nonnegative"),
    ("start_idx", [-1, 100], "nonnegative"),
    ("start_idx", [400., 100.], "integer"),
    ("end_idx", [400, 140], "nonempty"),
    ("end_idx", [5002, 140], "within"),
    ("start_idx", [130, 100], "overlap"),
    ("accepted", [1, 0], "boolean"),
    ("accepted", ["True", "False"], "boolean"),
    ("accepted", [True, None], "boolean"),
])
def test_invalid_native_contract_is_refused(recording, events, column, value, message):
    events[column] = value
    with pytest.raises(ValueError, match=message):
        measured(recording, events)


def test_uint64_values_cannot_wrap_to_negative_ids(recording, events):
    events["event_id"] = pd.Series([2**63, 7], dtype="uint64")
    with pytest.raises(ValueError, match="int64"):
        measured(recording, events)


@pytest.mark.parametrize("missing", ["event_id", "start_idx", "end_idx", "accepted"])
def test_required_source_evidence_is_not_invented(recording, events, missing):
    with pytest.raises(ValueError, match="require columns"):
        measured(recording, events.drop(columns=missing))


def test_nullable_integer_and_boolean_native_columns_are_supported(recording, events):
    for name in ("event_id", "start_idx", "end_idx"):
        events[name] = events[name].astype("Int64")
    events["accepted"] = events.accepted.astype("boolean")
    table = measured(recording, events).catalog
    assert table.source_event_id.dtype == np.dtype("int64")
    assert table.accepted.dtype == np.dtype("bool")
    assert table.accepted.tolist() == [False, True]


def test_empty_import_retains_typed_identity_measurements_and_curation(recording):
    native = pd.DataFrame(columns=["event_id", "start_idx", "end_idx", "accepted", "flag"])
    result = measured(recording, native)
    assert result.catalog.empty
    assert result.catalog.source_event_id.dtype == np.dtype("int64")
    assert result.catalog.start_idx.dtype == np.dtype("int64")
    assert result.catalog.accepted.dtype == np.dtype("bool")
    assert result.catalog.passes_gate.dtype == np.dtype("bool")
    assert result.provenance["accepted_count"] == result.provenance["candidate_count"] == 0


def test_source_snapshot_does_not_follow_later_curation_or_nested_metadata(recording, events):
    events["lineage"] = [{"parents": [18]}, {"parents": [7]}]
    provenance = {"source_key": "original", "params": {"threshold": [4.]}}
    result = manual.measure_imported_events(recording, events, manual.MeasurementSettings(),
                                            source_provenance=provenance)
    events.loc[0, "accepted"] = False
    events.at[1, "lineage"]["parents"].append(99)
    provenance["params"]["threshold"].append(100.)
    assert result.catalog.accepted.tolist() == [False, True]
    assert result.catalog.source_lineage.iloc[0] == {"parents": [7]}
    assert result.provenance["imported_events"]["source_provenance"] == {
        "source_key": "original", "params": {"threshold": [4.]}}


def test_remeasurement_preserves_native_ids_and_decisions(recording, events):
    first = measured(recording, events)
    second = manual.measure_imported_events(recording, events, manual.MeasurementSettings(min_depth_sigma=100.),
        source_provenance=first.provenance["imported_events"]["source_provenance"])
    assert first.catalog.event_id.equals(second.catalog.event_id)
    assert first.catalog.accepted.equals(second.catalog.accepted)
    assert first.catalog.passes_gate.any()
    assert not second.catalog.passes_gate.any()
    assert first.catalog.depth_pA.equals(second.catalog.depth_pA)


def test_upward_source_does_not_flip_prepared_downward_signal(recording, events):
    # Preparation already orients an upward raw source downward for the study.
    upward = replace(recording, raw=-recording.raw,
        metadata={**recording.metadata, "event_polarity": -1},
        provenance={**recording.provenance, "event_polarity": -1})
    downward = measured(recording, events)
    result = measured(upward, events)
    for column in ("depth_peak_pA", "depth_sust_pA", "fracblock", "I0_local_pA"):
        np.testing.assert_array_equal(result.catalog[column], downward.catalog[column])
    assert result.provenance["event_polarity"] == -1


def test_existing_decoder_measurement_and_import_have_identical_science(recording, tmp_path):
    model = tmp_path / "model.keras"
    model.write_bytes(b"frozen model fixture")
    probability = np.zeros(len(recording.raw), np.float32)
    probability[100:140] = .9
    probability[400:440] = .9
    inference = manual.InferenceResult(probability, {}, {
        **recording.provenance, "recording_identity": manual.recording_identity(recording),
        "probability_sha256": manual._array_hash(probability),
        "inference_settings": asdict(manual.InferenceSettings()),
        "model": reference._fingerprint(model), "inference": {},
    })
    candidates = manual.find_candidates(recording, inference, manual.DecodeSettings())
    expected = manual.measure_candidates(recording, candidates, manual.MeasurementSettings())
    native = candidates.copy()
    native["end_idx"] += 1
    native["event_id"] = np.arange(len(native), dtype=np.int64)
    native["accepted"] = expected.catalog.accepted
    actual = measured(recording, native)
    common = [column for column in actual.catalog if column in expected.catalog and column != "event_id"]
    pd.testing.assert_frame_equal(actual.catalog[common], expected.catalog[common])
    for key in ("baseline_pA", "noise_mad_pA", "quiet_sample_count", "i0_relative_drift", "i0_detail"):
        assert actual.provenance[key] == expected.provenance[key]


@pytest.mark.parametrize("phase", ["before", "during"])
def test_cancelled_import_publishes_nothing_and_preserves_input(recording, events, phase):
    stop = threading.Event()
    if phase == "before":
        stop.set()
    before = deepcopy(events)
    def progress(fraction, message):
        if fraction > .2:
            stop.set()
    with pytest.raises(OperationCancelled):
        measured(recording, events, cancel=stop, progress=progress)
    pd.testing.assert_frame_equal(events, before)


def test_prepared_array_change_during_measurement_is_refused(recording, events):
    def mutate(fraction, message):
        if fraction > .2:
            recording.detrended[0] = -999.
    with pytest.raises(ValueError, match="Prepared recording changed"):
        measured(recording, events, progress=mutate)


def test_checkpoint_retains_all_imported_source_columns_and_evidence(recording, events, tmp_path):
    from nanoscope.analysis.manual_workspace import save_workspace, load_workspace
    result = measured(recording, events)
    save_workspace({"analysis": result}, tmp_path / "workspace")
    restored = load_workspace(tmp_path / "workspace")["analysis"]
    pd.testing.assert_frame_equal(restored.catalog, result.catalog)
    assert restored.provenance == result.provenance


def test_downstream_rates_and_waveforms_use_imported_decisions(recording, events, tmp_path):
    from nanoscope.analysis import manual_comparisons as comparisons
    from nanoscope.analysis.incubation_study import StudyRecording
    analysis = measured(recording, events)
    condition = StudyRecording(recording.tag, "recording.abf", "recording.abf", 30., "pore-A", 3.)
    result = comparisons.compare_recordings([analysis], [condition],
        comparisons.ComparisonSettings(n_blocks=2, n_boot=10, sensitivity_blocks=(2,)),
        folder=tmp_path / "comparisons")
    assert result.events.event_id.tolist() == ["zn_30h:source:18"]
    assert result.summary.n_events.tolist() == [1]
    assert result.summary.rate_per_s.tolist() == [1 / recording.duration_s]
    # The accepted shallow event must survive even though the study gate fails.
    assert result.events.passes_gate.tolist() == [False]
    features = comparisons.extract_features([analysis], [condition])
    assert features.event_id.tolist() == ["zn_30h:source:18"]
    assert features.start_idx.tolist() == [400]
    assert features.end_idx.tolist() == [439]
