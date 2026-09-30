"""Scientific contract and stale-stage checks for user-driven student stages."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import threading

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import incubation_detection as reference
from nanoscope.analysis import manual_detection as manual
from nanoscope.analysis.cancel import OperationCancelled


class FakeStudent:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, values, verbose=0):
        self.batch_sizes.append(len(values))
        return np.where(values < -0.05, 0.9, 0.01).astype(np.float32)


def make_inputs(tmp_path, monkeypatch, *, n=5001, fs=10_000, window=64, model=None):
    detrended = np.resize(np.array([-0.15, -0.1, 0, 0.1, 0.15], np.float32), n)
    detrended[100:140] = -2.0
    detrended[400:440] = -0.4
    raw = -200 - np.arange(n) * 0.01 + detrended
    source = tmp_path / "recording.raw"
    source.write_bytes(b"original raw recording")
    model_path = tmp_path / "student.keras"
    model_path.write_bytes(b"original frozen student")
    recording = reference.PreparedRecording("zn_30h", raw, detrended, fs, n / fs,
        {"tag": "zn_30h", "fs_hz": fs, "event_polarity": 1},
        {"rawfile": reference._fingerprint(source), "preprocessing": "synthetic prepared input"})
    from nanoscope.analysis import ml_detect
    student = model or FakeStudent()
    monkeypatch.setattr(ml_detect, "clear_model_cache", lambda: None)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda path: SimpleNamespace(
        channels=1, window=window, model=student))
    settings = manual.InferenceSettings(window_samples=window, stride_samples=max(window // 2, 1))
    return recording, model_path, settings, student


def prepare_probability(recording, model_path, settings, values):
    # An independently constructed probability-stage fixture, with the same
    # public checkpoint format as the real inference stage.
    values = np.asarray(values, np.float32)
    return manual.InferenceResult(values, {}, {
        **recording.provenance,
        "recording_identity": manual.recording_identity(recording),
        "probability_sha256": manual._array_hash(values),
        "inference_settings": asdict(settings),
        "model": reference._fingerprint(model_path),
        "inference": {},
    })


def test_stages_default_science_matches_verified_frozen_contract(tmp_path, monkeypatch):
    recording, model, settings, student = make_inputs(tmp_path, monkeypatch)
    fractions = []
    inference = manual.run_inference(recording, model, settings,
        progress=lambda f, m: fractions.append(f))
    expected_probability, expected_coverage = reference._predict_notebook(
        FakeStudent(), recording.detrended, settings.window_samples)
    assert np.array_equal(inference.probability, expected_probability)
    assert not inference.probability.flags.writeable
    assert max(student.batch_sizes) == 64
    assert inference.coverage["uncovered_tail_samples"] == expected_coverage["uncovered_tail_samples"]
    assert fractions == sorted(fractions) and fractions[-1] == 1
    candidates = manual.find_candidates(recording, inference, manual.DecodeSettings())
    expected_regions = reference._student_regions(expected_probability, recording.fs_hz)
    assert list(candidates.itertuples(index=False, name=None)) == expected_regions
    baseline, noise, quiet = reference._quiet_stats(recording.detrended)
    expected, _ = reference._build_catalog(recording, expected_regions, baseline, noise, quiet)
    result = manual.measure_candidates(recording, candidates, manual.MeasurementSettings())
    pd.testing.assert_frame_equal(result.catalog[expected.columns], expected)
    assert np.array_equal(result.catalog.accepted, result.catalog.passes_gate)
    assert result.catalog.passes_gate.any() and (~result.catalog.passes_gate).any()
    assert result.catalog.event_id.tolist() == [f"zn_30h:{i}" for i in range(len(candidates))]
    assert result.provenance["measurement_settings"] == asdict(manual.MeasurementSettings())
    assert result.provenance["candidate_count"] == len(candidates)


@pytest.mark.parametrize("gap,merge,min_ms,expected", [
    (4, 0.05, 0.1, [(10, 25)]),
    (5, 0.05, 0.1, []),
    (5, 0.06, 0.1, [(10, 26)]),
    (5, 0.05, 0.05, [(10, 15), (21, 26)]),
])
def test_decode_merge_uses_inclusive_gap_then_final_duration(tmp_path, monkeypatch, gap, merge, min_ms, expected):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, n=60, fs=100_000, window=8)
    probability = np.zeros(60, np.float32)
    probability[10:16] = .4
    probability[16 + gap:22 + gap] = .4
    inference = prepare_probability(recording, model, settings, probability)
    got = manual.find_candidates(recording, inference,
        manual.DecodeSettings(smooth=1, merge_ms=merge, min_ms=min_ms))
    assert list(got.itertuples(index=False, name=None)) == expected


@pytest.mark.parametrize("fs,min_ms,minimum", [
    (10_000, .29, 3), (10_000, .3, 3),
    (10_000, np.nextafter(.3, np.inf), 3),
    (10_000, np.nextafter(.3, -np.inf), 3),
    (10_000, .300000001, 4), (10_000, .299999999, 3),
    (50_000, .1, 5), (50_000, np.nextafter(.1, np.inf), 5),
    (50_000, .100000001, 6), (44_100, .1, 5),
])
def test_minimum_duration_never_truncates_fractional_sample(tmp_path, monkeypatch, fs, min_ms, minimum):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, n=80, fs=fs, window=8)
    probability = np.zeros(80, np.float32)
    probability[10:10 + minimum - 1] = .9
    probability[40:40 + minimum] = .9
    inference = prepare_probability(recording, model, settings, probability)
    candidates = manual.find_candidates(recording, inference,
        manual.DecodeSettings(smooth=1, merge_ms=0, min_ms=min_ms))
    assert candidates.to_numpy().tolist() == [[40, 40 + minimum - 1]]
    sampling = candidates.attrs[manual._ATTR]["decode_sampling"]
    assert sampling["minimum_samples"] == minimum
    assert sampling["effective_minimum_ms"] == minimum / fs * 1000
    assert sampling["minimum_duration_rounding"] == "ceiling_with_8ulp_integer_tolerance"
    measured = manual.measure_candidates(recording, candidates, manual.MeasurementSettings())
    assert measured.provenance["decode_sampling"] == sampling
    assert np.all(measured.catalog.dwell_ms >= min_ms - 16 * np.spacing(min_ms))


def test_old_candidate_checkpoint_keeps_explicit_legacy_rounding(tmp_path, monkeypatch):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, n=80, window=8)
    probability = np.zeros(80, np.float32); probability[10:12] = .9
    inference = prepare_probability(recording, model, settings, probability)
    candidates = manual.find_candidates(recording, inference,
        manual.DecodeSettings(smooth=1, merge_ms=0, min_ms=.2))
    # A pre-fix candidate checkpoint stored requested .29 ms but retained this
    # two-sample event; loading/measurement must not silently rewrite its bounds.
    stage = candidates.attrs[manual._ATTR]
    stage.pop("decode_sampling"); stage["decode_settings"]["min_ms"] = .29
    measured = manual.measure_candidates(recording, candidates, manual.MeasurementSettings())
    assert measured.catalog.dwell_ms.tolist() == [.2]
    assert measured.provenance["decode_sampling"]["minimum_duration_rounding"] == "legacy_truncate"
    assert measured.provenance["decode_sampling"]["minimum_samples"] == 2


def test_threshold_and_smoothing_controls_change_candidates_without_inference(tmp_path, monkeypatch):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, n=60, window=8)
    probability = np.zeros(60, np.float32)
    probability[10:16] = .2
    probability[12] = .5
    probability[30:35] = .7
    inference = prepare_probability(recording, model, settings, probability)
    decode = manual.DecodeSettings(smooth=1, merge_ms=0, min_ms=0)
    low = manual.find_candidates(recording, inference, decode)
    raised = manual.find_candidates(recording, inference, replace(decode, enter=.6))
    narrow = manual.find_candidates(recording, inference, replace(decode, exit=.3))
    smoothed = manual.find_candidates(recording, inference, replace(decode, smooth=3))
    assert low.to_numpy().tolist() == [[10, 15], [30, 34]]
    assert raised.to_numpy().tolist() == [[30, 34]]
    assert narrow.to_numpy().tolist() == [[12, 12], [30, 34]]
    assert smoothed.to_numpy().tolist() == [[30, 34]]


@pytest.mark.parametrize("n,policy,covered,windows", [
    (7, "exclude_last_start", 0, 0), (8, "exclude_last_start", 0, 0),
    (8, "include_last_start", 8, 1), (7, "anchor_end", 0, 0),
    (16, "exclude_last_start", 12, 2), (16, "include_last_start", 16, 3),
    (17, "include_last_start", 16, 3), (17, "anchor_end", 17, 4),
])
def test_explicit_complete_window_tail_policies(tmp_path, monkeypatch, n, policy, covered, windows):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, n=n, window=8)
    result = manual.run_inference(recording, model, replace(settings, tail_policy=policy))
    assert result.coverage["covered_sample_count"] == covered
    assert result.coverage["covered_stop_idx_exclusive"] == covered
    assert result.coverage["uncovered_tail_samples"] == n - covered
    assert result.coverage["n_windows"] == windows
    assert np.all(result.probability[covered:] == 0)


def test_clip_stride_and_batch_are_consumed_by_inference(tmp_path, monkeypatch):
    recording, model, settings, student = make_inputs(tmp_path, monkeypatch)
    first = manual.run_inference(recording, model, settings)
    second = manual.run_inference(recording, model, replace(settings,
        clip_sigma=100, stride_samples=16, batch_size=7))
    assert not np.array_equal(first.probability, second.probability)
    assert second.coverage["n_windows"] > first.coverage["n_windows"]
    assert second.coverage["max_overlap_count"] == 4
    assert student.batch_sizes[-1] <= 7
    assert second.provenance["inference_settings"]["batch_size"] == 7


def test_measurement_settings_change_gate_and_local_i0_but_keep_all_candidates(tmp_path, monkeypatch):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, n=7001)
    # Nonlinear raw drift makes both block duration and smoothing window choices
    # observable instead of testing a ramp, whose rolling median is unchanged.
    recording.raw[:] = -100 - 100 * np.sin(np.arange(len(recording.raw)) / 1000) ** 2
    probability = np.zeros(len(recording.raw), np.float32)
    probability[100:140] = .9
    probability[400:440] = .9
    probability[3100:3120] = .9
    inference = prepare_probability(recording, model, settings, probability)
    candidates = manual.find_candidates(recording, inference, manual.DecodeSettings())
    default = manual.measure_candidates(recording, candidates, manual.MeasurementSettings())
    tighter = manual.measure_candidates(recording, candidates, manual.MeasurementSettings(min_depth_sigma=20))
    shorter = manual.measure_candidates(recording, candidates, manual.MeasurementSettings(i0_window_s=.1))
    blocks = manual.measure_candidates(recording, candidates, manual.MeasurementSettings(i0_window_s=.1, i0_block_s=.02))
    assert len(default.catalog) == len(tighter.catalog) == len(shorter.catalog) == 3
    assert tighter.catalog.passes_gate.sum() < default.catalog.passes_gate.sum()
    assert not np.array_equal(default.catalog.I0_local_pA, shorter.catalog.I0_local_pA)
    assert not np.array_equal(shorter.catalog.I0_local_pA, blocks.catalog.I0_local_pA)
    assert np.array_equal(default.catalog.depth_sust_pA, tighter.catalog.depth_sust_pA)


def test_zero_local_i0_reference_is_refused_not_clamped(tmp_path, monkeypatch):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, n=5000, fs=10_000, window=8)
    # Block 0's raw median is exactly zero while the recording-wide median stays
    # a normal -200, isolating the per-candidate local reference from the global one.
    recording.raw[:] = -200.0
    recording.raw[:1000] = 0.0
    probability = np.zeros(5000, np.float32)
    probability[10:50] = .9
    inference = prepare_probability(recording, model, settings, probability)
    candidates = manual.find_candidates(recording, inference,
        manual.DecodeSettings(smooth=1, merge_ms=0, min_ms=0))
    assert candidates.to_numpy().tolist() == [[10, 49]]
    with pytest.raises(ValueError, match="I0"):
        manual.measure_candidates(recording, candidates,
            manual.MeasurementSettings(i0_block_s=0.1, i0_window_s=0.1))


def test_zero_global_i0_reference_is_refused_not_clamped(tmp_path, monkeypatch):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, n=5000, fs=10_000, window=8)
    recording.raw[:] = 0.0
    probability = np.zeros(5000, np.float32)
    probability[10:50] = .9
    inference = prepare_probability(recording, model, settings, probability)
    candidates = manual.find_candidates(recording, inference,
        manual.DecodeSettings(smooth=1, merge_ms=0, min_ms=0))
    with pytest.raises(ValueError, match="I0"):
        manual.measure_candidates(recording, candidates, manual.MeasurementSettings())


def test_source_change_between_stages_is_refused(tmp_path, monkeypatch):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch)
    inference = manual.run_inference(recording, model, settings)
    (tmp_path / "recording.raw").write_bytes(b"replaced recording")
    with pytest.raises(ValueError, match="Source changed"):
        manual.find_candidates(recording, inference, manual.DecodeSettings())


@pytest.mark.parametrize("target", ["recording", "probability", "model"])
def test_stale_stage_cannot_be_laundered_into_new_candidates(tmp_path, monkeypatch, target):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch)
    inference = manual.run_inference(recording, model, settings)
    if target == "recording":
        recording.detrended[10] = -999
    elif target == "probability":
        inference.probability = inference.probability.copy()
        inference.probability[10] = .999
    else:
        model.write_bytes(b"other frozen model")
    with pytest.raises(ValueError, match="changed"):
        manual.find_candidates(recording, inference, manual.DecodeSettings())


def test_model_replaced_during_inference_is_refused(tmp_path, monkeypatch):
    class MutatingStudent(FakeStudent):
        def predict(self, values, verbose=0):
            (tmp_path / "student.keras").write_bytes(b"replaced during inference")
            return super().predict(values, verbose)
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch, model=MutatingStudent())
    with pytest.raises(ValueError, match="Source changed"):
        manual.run_inference(recording, model, settings)


def test_checkpoint_roundtrip_retains_stage_link_and_inclusive_bounds(tmp_path, monkeypatch):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch)
    inference = manual.run_inference(recording, model, settings)
    np.savez(tmp_path / "probability.npz", probability=inference.probability)
    with np.load(tmp_path / "probability.npz", allow_pickle=False) as saved:
        restored = manual.InferenceResult(saved["probability"],
            json.loads(json.dumps(inference.coverage)), json.loads(json.dumps(inference.provenance)))
    candidates = manual.find_candidates(recording, restored, manual.DecodeSettings())
    candidates.to_csv(tmp_path / "candidates.csv", index=False)
    reloaded = pd.read_csv(tmp_path / "candidates.csv")
    reloaded.attrs = json.loads(json.dumps(candidates.attrs))
    result = manual.measure_candidates(recording, reloaded, manual.MeasurementSettings())
    assert len(result.catalog) == len(candidates)
    assert np.array_equal(result.catalog.end_idx, candidates.end_idx)
    np.testing.assert_allclose(result.catalog.end_s,
        (result.catalog.end_idx + 1) / recording.fs_hz, rtol=0, atol=0)
    reloaded.loc[0, "end_idx"] += 1
    with pytest.raises(ValueError, match="bounds changed"):
        manual.measure_candidates(recording, reloaded, manual.MeasurementSettings())


def test_zero_candidates_are_typed_and_measurable(tmp_path, monkeypatch):
    recording, model, settings, _ = make_inputs(tmp_path, monkeypatch)
    probability = np.zeros(len(recording.raw), np.float32)
    candidates = manual.find_candidates(recording,
        prepare_probability(recording, model, settings, probability), manual.DecodeSettings())
    result = manual.measure_candidates(recording, candidates, manual.MeasurementSettings())
    assert result.catalog.empty
    assert result.catalog.start_idx.dtype == np.int64
    assert result.catalog.passes_gate.dtype == bool
    assert result.catalog.accepted.dtype == bool
    assert result.provenance["accepted_count"] == 0


def test_cancel_before_inference_does_not_call_student(tmp_path, monkeypatch):
    recording, model, settings, student = make_inputs(tmp_path, monkeypatch)
    token = threading.Event(); token.set()
    with pytest.raises(OperationCancelled):
        manual.run_inference(recording, model, settings, cancel=token)
    assert student.batch_sizes == []


def test_window_mismatch_is_refused_before_prediction(tmp_path, monkeypatch):
    recording, model, settings, student = make_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="does not match model"):
        manual.run_inference(recording, model, replace(settings, window_samples=128))
    assert student.batch_sizes == []


@pytest.mark.parametrize("factory,kwargs", [
    (manual.InferenceSettings, {"clip_sigma": 0}),
    (manual.InferenceSettings, {"clip_sigma": np.nan}),
    (manual.InferenceSettings, {"stride_samples": 4096}),
    (manual.InferenceSettings, {"window_samples": 8.5}),
    (manual.InferenceSettings, {"window_samples": True}),
    (manual.InferenceSettings, {"batch_size": 0}),
    (manual.InferenceSettings, {"tail_policy": "guess"}),
    (manual.InferenceSettings, {"normalization": "none"}),
    (manual.DecodeSettings, {"enter": .1, "exit": .2}),
    (manual.DecodeSettings, {"enter": 1.1}),
    (manual.DecodeSettings, {"exit": -1}),
    (manual.DecodeSettings, {"smooth": 4}),
    (manual.DecodeSettings, {"smooth": 0}),
    (manual.DecodeSettings, {"merge_ms": -1}),
    (manual.DecodeSettings, {"min_ms": np.inf}),
    (manual.MeasurementSettings, {"min_depth_sigma": -1}),
    (manual.MeasurementSettings, {"i0_block_s": 0}),
    (manual.MeasurementSettings, {"i0_window_s": "5"}),
])
def test_invalid_settings_are_refused(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
