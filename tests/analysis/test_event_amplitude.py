"""Optional amplitudes never redefine or mutate detector characterization."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.event_amplitude import (
    AMPLITUDE_COLUMNS,
    METHODS,
    amplitude_is_current,
    clear_event_amplitude,
    invalidate_event_amplitudes,
    measure_event_amplitude,
    write_event_amplitude,
)
from nanoscope.analysis.events import SCHEMA


def _event(depths=(40.0,) * 9 + (150.0,), *, polarity=-1, drift=False):
    baseline = np.linspace(900, 1100, len(depths) + 4) if drift else np.full(len(depths) + 4, 1000.0)
    current = baseline.copy()
    current[2:-2] += polarity * np.asarray(depths)
    df = characterize(current, baseline, [2], [len(current) - 2], 10_000, polarity, noise_rms=1)
    return current, baseline, df


def _measurement(current, baseline, row, *, method="median", polarity=-1, **kwargs):
    return measure_event_amplitude(current, baseline, row, method=method,
                                   polarity=polarity, units="pA", **kwargs)


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("method,expected", [("median", 40), ("mean", 51), ("peak", 150)])
def test_distinct_whole_event_measurements_preserve_science(method, expected, polarity):
    current, baseline, df = _event(polarity=polarity)
    original_current, original_baseline, original_df = current.copy(), baseline.copy(), df.copy(deep=True)
    result = _measurement(current, baseline, df.iloc[0], method=method, polarity=polarity)
    assert result["amplitude_pA"] == expected
    assert result["amplitude_level_pA"] == 1000 + polarity * expected
    assert df.iloc[0].depth_pA == 40
    np.testing.assert_array_equal(current, original_current)
    np.testing.assert_array_equal(baseline, original_baseline)
    pd.testing.assert_frame_equal(df, original_df)
    assert not set(AMPLITUDE_COLUMNS) & set(SCHEMA)
    assert amplitude_is_current({**df.iloc[0].to_dict(), **result})


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("method", ["median", "mean", "peak", "mode"])
def test_drift_uses_samplewise_baseline_not_raw_current_statistic(polarity, method):
    current, baseline, df = _event([50] * 21, polarity=polarity, drift=True)
    result = _measurement(current, baseline, df.iloc[0], method=method, polarity=polarity)
    assert result["amplitude_pA"] == pytest.approx(50)
    assert result["amplitude_baseline_pA"] == np.mean(baseline[2:-2])
    assert result["amplitude_level_pA"] == pytest.approx(np.mean(baseline[2:-2]) + polarity * 50)


def test_drift_median_is_median_deviation_not_difference_of_medians():
    baseline = np.array([100.0, 120, 100, 180, 90])
    dev = np.array([5.0, 10, 40, 15, 20])
    current = baseline - dev
    row = {"start_idx": 0, "end_idx": 5, "baseline_pA": 118, "depth_pA": 15}
    result = _measurement(current, baseline, row)
    assert result["amplitude_pA"] == 15
    assert np.median(baseline) - np.median(current) != 15
    assert result["amplitude_level_pA"] == 103


@pytest.mark.parametrize("method,expected", [("selected_median", 40), ("selected_mean", 95)])
def test_selected_interval_is_half_open_and_never_trims(method, expected):
    current, baseline, df = _event()
    selection = (10, 12) if method == "selected_mean" else (2, 11)
    result = _measurement(current, baseline, df.iloc[0], method=method, selection=selection,
                          provenance={"selected_state": 2})
    assert result["amplitude_pA"] == expected
    assert (result["amplitude_start_idx"], result["amplitude_end_idx"]) == selection
    metadata = json.loads(result["amplitude_provenance_json"])
    assert metadata["interval_scope"] == "selected_interval"
    assert metadata["signal_identity"]["selected_state"] == 2
    assert amplitude_is_current({**df.iloc[0].to_dict(), **result})


@pytest.mark.parametrize("polarity", [-1, 1])
def test_manual_level_uses_mean_baseline_over_whole_event(polarity):
    current, baseline, df = _event(polarity=polarity, drift=True)
    reference = np.mean(baseline[2:-2])
    result = _measurement(current, baseline, df.iloc[0], method="manual", polarity=polarity,
                          manual_level=reference + polarity * 75)
    assert result["amplitude_pA"] == 75
    assert result["amplitude_level_pA"] == reference + polarity * 75
    assert json.loads(result["amplitude_provenance_json"])["interval_scope"] == "whole_event"
    with pytest.raises(ValueError, match="nonnegative"):
        _measurement(current, baseline, df.iloc[0], method="manual", polarity=polarity,
                     manual_level=reference - polarity)


def test_noise_changes_peak_but_median_remains_near_flat_depth():
    rng = np.random.default_rng(241)
    current, baseline, df = _event(80 + rng.normal(0, 3, 3000))
    median = _measurement(current, baseline, df.iloc[0])["amplitude_pA"]
    mean = _measurement(current, baseline, df.iloc[0], method="mean")["amplitude_pA"]
    peak = _measurement(current, baseline, df.iloc[0], method="peak")["amplitude_pA"]
    mode = _measurement(current, baseline, df.iloc[0], method="mode")["amplitude_pA"]
    assert abs(median - 80) < 0.2
    assert abs(mean - 80) < 0.2
    assert abs(mode - 80) < 1.5
    assert peak > 88


def test_peak_is_signed_excursion_not_largest_absolute_noise():
    result = _measurement(np.array([900.0, 1200.0, 950.0]), 1000,
                          {"start_idx": 0, "end_idx": 3}, method="peak")
    assert result["amplitude_pA"] == 100


def test_scalar_baseline_is_explicitly_supported():
    current, baseline, df = _event()
    assert _measurement(current, 1000, df.iloc[0]) == _measurement(current, baseline, df.iloc[0])


def test_nonfinite_outside_selected_measurement_does_not_enter_estimator():
    current, baseline, df = _event()
    current[0] = np.nan
    current[11] = np.inf
    baseline[11] = np.nan
    assert _measurement(current, baseline, df.iloc[0], method="selected_mean",
                        selection=(2, 11))["amplitude_pA"] == 40


@pytest.mark.parametrize("kwargs", [
    {"method": "not_a_method"}, {"polarity": 0}, {"polarity": True},
    {"polarity": 2}, {"units": ""}, {"units": None},
    {"method": "selected_mean"}, {"method": "selected_median", "selection": (1, 10)},
    {"method": "selected_mean", "selection": (2, 13)},
    {"method": "selected_mean", "selection": (3, 3)},
    {"method": "selected_mean", "selection": (4, 3)},
    {"method": "selected_mean", "selection": (2.5, 5)},
    {"method": "selected_mean", "selection": (True, 5)},
    {"selection": (2, 10)}, {"manual_level": 920},
    {"method": "manual"}, {"method": "manual", "manual_level": np.inf},
    {"method": "manual", "manual_level": np.nan},
    {"provenance": {"source_path": "/private/sample.abf"}},
    {"provenance": {"identity": "/private/sample.abf"}},
    {"provenance": {"bad": np.nan}}, {"provenance": {"object": object()}},
])
def test_invalid_arguments_refuse_without_mutation(kwargs):
    current, baseline, df = _event()
    before = df.copy(deep=True)
    args = {"method": "median", "polarity": -1, "units": "pA", **kwargs}
    with pytest.raises(ValueError):
        measure_event_amplitude(current, baseline, df.iloc[0], **args)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("start,end", [(0, 0), (-1, 5), (3, 2), (2, 15), (2.1, 5), (True, 5)])
def test_invalid_event_bounds_are_not_clipped(start, end):
    with pytest.raises(ValueError):
        _measurement(np.zeros(14), 1000, {"start_idx": start, "end_idx": end})


@pytest.mark.parametrize("which", ["current", "baseline"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_measured_samples_refused(which, bad):
    current, baseline, df = _event()
    (current if which == "current" else baseline)[3] = bad
    with pytest.raises(ValueError):
        _measurement(current, baseline, df.iloc[0])


@pytest.mark.parametrize("current,baseline", [
    (np.ones((2, 5)), 0), (np.ones(5), np.zeros(4)),
    (np.ones(5), np.zeros((1, 5))), (np.array([]), 0),
    (np.ones(5), np.nan), (np.ones(5), ["bad"] * 5),
])
def test_bad_signal_shapes_and_baselines_refused(current, baseline):
    with pytest.raises(ValueError):
        _measurement(current, baseline, {"start_idx": 0, "end_idx": 5})


def test_written_extras_are_typed_and_core_frame_is_unchanged():
    current, baseline, single = _event()
    df = pd.concat([single, single], ignore_index=True)
    df.index = [42, 42]  # Writer accepts positions, including duplicate labels.
    original = df.copy(deep=True)
    measurement = _measurement(current, baseline, df.iloc[1], method="mean")
    assert write_event_amplitude(df, 1, measurement)
    pd.testing.assert_frame_equal(df[list(SCHEMA)], original)
    assert df.iloc[1].amplitude_pA == 51
    assert df.iloc[0][list(AMPLITUDE_COLUMNS)].isna().all()
    for column, (dtype, _, _) in AMPLITUDE_COLUMNS.items():
        assert str(df[column].dtype) == dtype
    assert not write_event_amplitude(df, 1, measurement)
    assert amplitude_is_current(df.iloc[1])
    assert not amplitude_is_current(df.iloc[0])


@pytest.mark.parametrize("edit", [
    {"amplitude_pA": np.nan}, {"amplitude_pA": -1}, {"amplitude_pA": 51},
    {"amplitude_method": "unknown"}, {"amplitude_start_idx": 3},
    {"amplitude_end_idx": 99}, {"amplitude_level_pA": np.inf},
    {"amplitude_provenance_json": "not json"}, {"amplitude_provenance_json": "[]"},
    {"amplitude_status": "stale"}, {"depth_pA": 500},
])
def test_write_refuses_invalid_or_core_edits_atomically(edit):
    current, baseline, df = _event()
    measurement = _measurement(current, baseline, df.iloc[0])
    measurement.update(edit)
    original = df.copy(deep=True)
    assert not write_event_amplitude(df, 0, measurement)
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("position", [-1, 1, 42, 0.5, True])
def test_invalid_write_position_does_not_create_columns(position):
    current, baseline, df = _event()
    original = df.copy(deep=True)
    assert not write_event_amplitude(df, position, _measurement(current, baseline, df.iloc[0]))
    assert not clear_event_amplitude(df, position)
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("column,value", [
    ("start_idx", 3), ("end_idx", 11), ("baseline_pA", 1001),
    ("depth_pA", 41), ("mean_dev_pA", 52), ("max_dev_pA", 151),
    ("dwell_s", 0.01), ("ecd_pA_s", 0.5),
])
def test_changed_core_measurement_invalidates_fingerprint(column, value):
    current, baseline, df = _event()
    assert write_event_amplitude(df, 0, _measurement(current, baseline, df.iloc[0]))
    df.loc[0, column] = value
    assert not amplitude_is_current(df.iloc[0])


def test_curation_labels_and_id_renumbering_do_not_change_measurement():
    current, baseline, df = _event()
    assert write_event_amplitude(df, 0, _measurement(current, baseline, df.iloc[0]))
    df.loc[0, ["accepted", "flag", "event_id", "recording"]] = [False, "manual_reject", 45, "new label"]
    assert amplitude_is_current(df.iloc[0])


def test_signal_identity_is_safe_json_and_compares_original_subset():
    current, baseline, df = _event()
    df.loc[0, "rise_s"] = np.nan
    identity = {"baseline_source": "recording", "baseline_fingerprint": "abc123",
                "calibration_fingerprint": (np.float64(1), np.float64(0)),
                "sample_rate_hz": np.int64(10_000)}
    result = _measurement(current, baseline, df.iloc[0], provenance=identity)
    row = {**df.iloc[0].to_dict(), **result}
    metadata = json.loads(result["amplitude_provenance_json"])
    assert metadata["signal_identity"]["calibration_fingerprint"] == [1.0, 0.0]
    assert metadata["event_fingerprint"]["rise_s"] is None
    assert "NaN" not in result["amplitude_provenance_json"]
    assert amplitude_is_current(row, {**identity, "additional_safe_context": 5})
    assert not amplitude_is_current(row, {**identity, "baseline_fingerprint": "changed"})
    assert not amplitude_is_current(row, {})


def test_csv_roundtrip_validity():
    current, baseline, df = _event([40.123456789] * 9 + [150.987654321], drift=True)
    result = _measurement(current, baseline, df.iloc[0], method="mean")
    assert write_event_amplitude(df, 0, result)
    restored = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    assert amplitude_is_current(restored.iloc[0])


def test_clear_restores_blanks_only_in_optional_fields_and_is_idempotent():
    current, baseline, df = _event()
    original = df.copy(deep=True)
    assert not clear_event_amplitude(df, 0)
    assert list(df) == list(original)
    assert write_event_amplitude(df, 0, _measurement(current, baseline, df.iloc[0]))
    assert clear_event_amplitude(df, 0)
    assert df.iloc[0][list(AMPLITUDE_COLUMNS)].isna().all()
    assert not clear_event_amplitude(df, 0)
    pd.testing.assert_frame_equal(df[list(SCHEMA)], original)


def test_invalidation_retains_audit_metadata_but_removes_numeric_values():
    current, baseline, df = _event()
    original = df.copy(deep=True)
    measurement = _measurement(current, baseline, df.iloc[0], method="peak")
    assert write_event_amplitude(df, 0, measurement)
    assert invalidate_event_amplitudes(df, [0], reason="event_mask_changed") == 1
    assert df.iloc[0].amplitude_method == "peak"
    assert df.iloc[0].amplitude_status == "stale"
    assert json.loads(df.iloc[0].amplitude_provenance_json)["invalidation_reason"] == "event_mask_changed"
    for key, (dtype, _, _) in AMPLITUDE_COLUMNS.items():
        if dtype != "string":
            assert pd.isna(df.iloc[0][key])
    assert not amplitude_is_current(df.iloc[0])
    assert invalidate_event_amplitudes(df) == 0
    pd.testing.assert_frame_equal(df[list(SCHEMA)], original)


def test_invalidation_is_positional_selective_and_atomic_on_bad_rows():
    current, baseline, single = _event()
    df = pd.concat([single] * 3, ignore_index=True)
    df.index = [20, 30, 40]
    for position in (0, 2):
        assert write_event_amplitude(df, position, _measurement(current, baseline, df.iloc[position]))
    before = df.copy(deep=True)
    with pytest.raises(ValueError):
        invalidate_event_amplitudes(df, [0, 3])
    pd.testing.assert_frame_equal(df, before)
    assert invalidate_event_amplitudes(df, [2, 2]) == 1
    assert amplitude_is_current(df.iloc[0])
    assert pd.isna(df.iloc[1].amplitude_status)
    assert df.iloc[2].amplitude_status == "stale"


def test_legacy_frames_are_never_migrated_implicitly():
    _, _, df = _event()
    original = df.copy(deep=True)
    assert not amplitude_is_current(df.iloc[0])
    assert invalidate_event_amplitudes(df) == 0
    assert not clear_event_amplitude(df, 0)
    pd.testing.assert_frame_equal(df, original)
    assert METHODS == ("median", "mean", "peak", "mode", "selected_median", "selected_mean", "manual")
