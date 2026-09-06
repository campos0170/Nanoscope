"""No old amplitude becomes the measurement of a resized/merged/re-measured event."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nanoscope.analysis.baseline import DEFAULT_BASELINE
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.event_amplitude import AMPLITUDE_COLUMNS
from nanoscope.analysis.events import SCHEMA, EventSet
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.analysis.remeasure import (
    curation_snapshot, merge_guard_snapshot, merge_late_curation, remeasure_events,
)


FS = 50_000.0


def _events():
    signal = np.full(50_000, 100.0)
    for start, end in ((10_000, 10_300), (10_310, 10_700), (30_000, 30_300)):
        signal[start:end] = 70.0
    signal += np.random.default_rng(22).normal(0.0, 0.05, signal.size)
    frame, provenance = detect_events(
        signal, FS, baseline=100.0, polarity=-1, min_dwell_s=1e-4,
        recording="synthetic", pore_id="pore")
    assert len(frame) == 3
    return EventSet(frame, provenance), signal


def _curate(frame, positions=(0,)):
    for col, (dtype, *_rest) in AMPLITUDE_COLUMNS.items():
        missing = np.nan if dtype == "float64" else pd.NA
        frame[col] = pd.Series([missing] * len(frame), dtype=dtype)
    for position in positions:
        row = frame.iloc[position]
        values = {
            "amplitude_pA": 31.0, "amplitude_method": "manual_level",
            "amplitude_level_pA": 69.0, "amplitude_baseline_pA": 100.0,
            "amplitude_start_idx": int(row["start_idx"]),
            "amplitude_end_idx": int(row["end_idx"]),
            "amplitude_units": "pA", "amplitude_status": "current",
            "amplitude_provenance_json": json.dumps({"test_event": int(row["event_id"])}),
        }
        for col, value in values.items():
            frame.iat[position, frame.columns.get_loc(col)] = value


def test_merge_invalidates_only_merged_amplitude_and_keeps_scientific_result():
    events, signal = _events()
    control, count = merge_by_gap(events, signal, FS, max_gap_s=0.001)
    assert count == 1
    _curate(events.df, (0, 2))
    before = events.df.copy(deep=True)
    merged, count = merge_by_gap(events, signal, FS, max_gap_s=0.001)
    assert count == 1
    assert_frame_equal(merged.df[list(SCHEMA)], control.df[list(SCHEMA)])
    assert merged.df.iloc[0]["amplitude_status"] == "stale"
    assert pd.isna(merged.df.iloc[0]["amplitude_pA"])
    assert merged.df.iloc[0]["amplitude_method"] == "manual_level"
    assert merged.df.iloc[1]["amplitude_pA"] == 31.0
    assert merged.df.iloc[1]["amplitude_status"] == "current"
    assert_frame_equal(events.df, before)


def test_remeasure_invalidates_previous_amplitude_without_altering_schema_result():
    events, signal = _events()
    control = remeasure_events(events, signal, FS, config=DEFAULT_BASELINE)
    _curate(events.df)
    before = events.df.copy(deep=True)
    measured = remeasure_events(events, signal, FS, config=DEFAULT_BASELINE)
    assert_frame_equal(measured.events.df[list(SCHEMA)], control.events.df[list(SCHEMA)])
    assert measured.events.df.iloc[0]["amplitude_status"] == "stale"
    assert pd.isna(measured.events.df.iloc[0]["amplitude_pA"])
    assert measured.events.df.iloc[0]["amplitude_method"] == "manual_level"
    assert_frame_equal(events.df, before)


def test_baseline_definition_change_invalidates_even_when_measured_scalars_coincide():
    events, signal = _events()
    measured = remeasure_events(events, signal, FS, config=DEFAULT_BASELINE).events
    expected = remeasure_events(measured, signal, FS, config=DEFAULT_BASELINE).events
    _curate(measured.df)
    measured.provenance["baseline_fingerprint"] = "different-historical-definition"
    out = remeasure_events(measured, signal, FS, config=DEFAULT_BASELINE).events
    assert_frame_equal(out.df[list(SCHEMA)], expected.df[list(SCHEMA)])
    assert out.df.iloc[0]["amplitude_status"] == "stale"
    assert pd.isna(out.df.iloc[0]["amplitude_pA"])


@pytest.mark.parametrize("already_curated", [False, True])
def test_late_amplitude_edit_refuses_remeasure_install_and_preserves_live_choice(already_curated):
    events, signal = _events()
    if already_curated:
        _curate(events.df)
    before = curation_snapshot(events.df)
    result = remeasure_events(events, signal, FS, config=DEFAULT_BASELINE)
    if not already_curated:
        _curate(events.df)
    events.df.loc[0, "amplitude_level_pA"] = 65.0
    live_copy = events.df.copy(deep=True)
    with pytest.raises(ValueError, match="amplitude was edited"):
        merge_late_curation(result.events.df, events.df, before)
    assert_frame_equal(events.df, live_copy)


def test_unchanged_amplitude_does_not_block_late_acceptance_carry():
    events, signal = _events()
    _curate(events.df)
    before = curation_snapshot(events.df)
    result = remeasure_events(events, signal, FS, config=DEFAULT_BASELINE)
    events.df.loc[1, "accepted"] = False
    installed = merge_late_curation(result.events.df, events.df, before)
    assert not installed.iloc[1]["accepted"]
    assert installed.iloc[0]["amplitude_status"] == "stale"
    assert pd.isna(installed.iloc[0]["amplitude_pA"])


def test_merge_guard_notices_first_amplitude_choice_or_changed_level():
    events, _ = _events()
    pristine = merge_guard_snapshot(events.df)
    _curate(events.df)
    curated = merge_guard_snapshot(events.df)
    assert not pristine.equals(curated)
    assert curated.equals(merge_guard_snapshot(events.df))
    events.df.loc[0, "amplitude_level_pA"] = 62.0
    assert not curated.equals(merge_guard_snapshot(events.df))


def test_legacy_remeasure_does_not_create_amplitude_columns():
    events, signal = _events()
    out = remeasure_events(events, signal, FS, config=DEFAULT_BASELINE)
    assert not set(AMPLITUDE_COLUMNS).intersection(out.events.df.columns)
