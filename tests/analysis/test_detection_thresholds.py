"""The detector's thresholds as trace rails, and the marks on edges a person moved.

SYNTHETIC ONLY. Every recording here is an array built in this file.

THE INDEPENDENT ANSWER IS THE DETECTOR'S OWN ARRAYS. `preview_capture.capture_detection_trace`
hands back the exact ``(current, baseline)`` pair `detect_events` / `detect_events_cusum`
thresholded, so "the rail is what the detector compared with" is asserted against the
detector itself rather than against a second call of the reproduction under test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import decimate_track
from nanoscope.analysis.boundary_marks import boundary_marks
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.detect_cusum import detect_events_cusum
from nanoscope.analysis.detection_thresholds import (
    RAIL_DRIFT, RAIL_ENTER, RAIL_EXIT, ThresholdsUnavailable, decimate_extrema,
    detection_threshold_tracks, threshold_refusal)
from nanoscope.analysis.events import SOURCE_COL, SOURCE_MANUAL, stamp_original_bounds
from nanoscope.analysis.modules import detection_signal
from nanoscope.analysis.preview_capture import capture_detection_trace
from nanoscope.analysis.scoring import ADDED_COLUMN

FS = 50_000.0
EVENTS = [(20_000 + i * 18_000, 20_000 + i * 18_000 + 250 + 40 * i) for i in range(8)]
N = 170_000


def _trace(polarity: int = -1, i0: float = -600.0, depth: float = 60.0, seed: int = 5):
    """An open pore with a slow 20 pA sag, 3 pA noise and eight events of ``depth``.

    ``polarity`` is the direction an event moves the SIGNED current: -1 downward. The sag is
    what makes a flat "median + k sigma" line wrong, so the test can tell a track from one.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = i0 + 20.0 * np.sin(2 * np.pi * t / (N / FS)) + rng.normal(0.0, 3.0, N)
    for s, e in EVENTS:
        x[s:e] += polarity * depth
    return np.ascontiguousarray(x, dtype=np.float32)


def _schmitt(raw, **kw):
    with capture_detection_trace() as cap:
        df, prov = detect_events(raw, FS, filter_cutoff_hz=10_000.0, min_dwell_s=1e-4, **kw)
    cur, base = cap.require_single()
    return df, prov, np.asarray(cur, dtype=np.float64), np.asarray(base, dtype=np.float64)


# ================================================================================================
# A. The rails are what the detector compared with.
# ================================================================================================
@pytest.mark.parametrize("polarity", [-1, +1])
def test_the_schmitt_rails_are_the_detectors_moving_baseline_plus_its_thresholds(polarity):
    raw = _trace(polarity)
    df, prov, _cur, base = _schmitt(raw)
    assert len(df) == len(EVENTS)
    assert prov["polarity"] == polarity
    loc = detection_signal(raw, FS, prov)
    tracks = detection_threshold_tracks(loc, FS, prov, module_id="detect.schmitt")
    by = {r.label: r for r in tracks.rails}
    assert set(by) == {RAIL_ENTER, RAIL_EXIT}
    sigma = float(prov["noise_rms_pA"])
    for label, thr in ((RAIL_ENTER, prov["threshold_in_pA"]),
                       (RAIL_EXIT, prov["threshold_out_pA"])):
        rail = by[label]
        expected = base[rail.idx] + polarity * float(thr)
        # Within a hundredth of a sigma of the detector's own comparison level, everywhere.
        np.testing.assert_allclose(rail.values, expected, atol=0.01 * sigma)
    # A TRACK, not a line: the sag moves it by tens of pA, which a flat line would miss.
    enter = by[RAIL_ENTER].values
    assert float(enter.max() - enter.min()) > 20.0
    assert ("below" if polarity < 0 else "above") in tracks.description


def test_only_samples_inside_detected_events_cross_the_full_resolution_enter_rail():
    """The picture's promise: where the compared signal goes past the dashed rail there is
    an event, and every event goes past it."""
    raw = _trace(-1)
    df, prov, _cur, _base = _schmitt(raw)
    loc = detection_signal(raw, FS, prov).astype(np.float64)
    tracks = detection_threshold_tracks(loc, FS, prov, module_id="detect.schmitt", max_points=0)
    enter = next(r for r in tracks.rails if r.label == RAIL_ENTER)
    assert enter.idx.size == loc.size
    beyond = -1 * (loc - enter.values) >= 0.0
    inside = np.zeros(loc.size, dtype=bool)
    for s, e in zip(df["start_idx"], df["end_idx"]):
        inside[int(s):int(e)] = True
        assert beyond[int(s):int(e)].any(), "an event that never crossed its enter rail"
    assert not (beyond & ~inside).any(), "the signal crossed the rail outside every event"


def test_a_span_starts_and_ends_at_the_exit_rail_and_the_sentence_says_so():
    """`detect._hysteresis_runs` takes the run past the EXIT level and keeps it when its peak
    passes the enter level. A sentence saying "entered at k_in" sent a user checking a span's
    start to the dashed rail, which the signal crosses a few samples after the span began."""
    raw = _trace(-1).astype(np.float64)
    for s, _e in EVENTS:                 # a 200-sample ramp in: the two crossings come apart
        raw[s - 200:s] -= 60.0 * np.linspace(0.0, 1.0, 200, endpoint=False)
    raw = raw.astype(np.float32)
    df, prov, _cur, _base = _schmitt(raw)
    assert len(df) == len(EVENTS)
    loc = detection_signal(raw, FS, prov).astype(np.float64)
    tracks = detection_threshold_tracks(loc, FS, prov, module_id="detect.schmitt", max_points=0)
    by = {r.label: r.values for r in tracks.rails}
    past_exit = -1 * (loc - by[RAIL_EXIT]) > 0.0       # strict, as the detector compares
    past_enter = -1 * (loc - by[RAIL_ENTER]) > 0.0
    for s, e in zip(df["start_idx"].astype(int), df["end_idx"].astype(int)):
        assert past_exit[s] and not past_exit[s - 1], "a span that does not start at the exit rail"
        assert past_exit[e - 1] and not past_exit[e], "a span that does not end at the exit rail"
        first_enter = s + int(np.argmax(past_enter[s:e]))
        assert past_enter[first_enter] and first_enter > s, \
            "on a ramp the span begins well before the signal reaches the dashed rail"
    text = tracks.description
    assert "exit level" in text and "enter level" in text
    assert text.index("exit level") < text.index("enter level")
    assert "entered when" not in text


def test_the_cusum_rail_is_the_drift_level_and_h_is_said_not_drawn():
    raw = _trace(-1, depth=30.0)
    with capture_detection_trace() as cap:
        df, prov = detect_events_cusum(raw, FS, filter_cutoff_hz=10_000.0, min_dwell_s=1e-4)
    _cur, base = cap.require_single()
    assert len(df) >= 1
    loc = detection_signal(raw, FS, prov)
    tracks = detection_threshold_tracks(loc, FS, prov, module_id="detect.cusum")
    assert [r.label for r in tracks.rails] == [RAIL_DRIFT]
    rail = tracks.rails[0]
    expected = np.asarray(base, dtype=np.float64)[rail.idx] - float(prov["cusum_k"]) * float(
        prov["noise_rms_pA"])
    np.testing.assert_allclose(rail.values, expected, atol=0.01 * float(prov["noise_rms_pA"]))
    assert "not a current" in tracks.description


# ================================================================================================
# B. Refusals: say so, draw nothing.
# ================================================================================================
@pytest.fixture(scope="module")
def schmitt_prov():
    raw = _trace(-1)
    _df, prov, _cur, _base = _schmitt(raw)
    return raw, prov


@pytest.mark.parametrize("module_id, word", [
    ("detect.keras", "ML table"), ("detect.manual", "hand-made"),
    ("detect.reconciled", "reconciled")])
def test_tables_without_a_threshold_of_this_kind_say_so(schmitt_prov, module_id, word):
    _raw, prov = schmitt_prov
    reason = threshold_refusal(module_id, prov)
    assert reason is not None and word in reason


@pytest.mark.parametrize("key, word", [
    ("hp_enabled", "high-passed"), ("detrend_enabled", "drift removal"),
    ("normalize_enabled", "Normalized")])
def test_a_chain_that_moves_the_signal_off_the_trace_level_is_refused(schmitt_prov, key, word):
    _raw, prov = schmitt_prov
    reason = threshold_refusal("detect.schmitt", {**prov, key: True})
    assert reason is not None and word in reason


def test_the_split_high_pass_stamp_alone_is_refused(schmitt_prov):
    _raw, prov = schmitt_prov
    assert "high-passed" in threshold_refusal(
        "detect.schmitt", {**prov, "highpass_localization_only": True})


def test_a_remeasured_or_unstamped_or_resized_table_is_refused(schmitt_prov):
    _raw, prov = schmitt_prov
    assert "re-measured" in threshold_refusal("detect.schmitt", {**prov, "remeasured": {}})
    unstamped = {k: v for k, v in prov.items() if k != "baseline_fingerprint"}
    assert "does not record which open-pore estimator" in threshold_refusal(
        "detect.schmitt", unstamped)
    assert "samples" in threshold_refusal("detect.schmitt", prov, n_samples=N - 1)
    assert threshold_refusal("detect.schmitt", prov, n_samples=N) is None


def test_a_record_the_reproduction_does_not_land_on_draws_nothing(schmitt_prov):
    """Tampered sigma stands for every way the rebuild can miss: another estimator, another
    chain, another calibration. The rail would not be the detector's, so it is refused."""
    raw, prov = schmitt_prov
    loc = detection_signal(raw, FS, prov)
    with pytest.raises(ThresholdsUnavailable, match="would not be the one"):
        detection_threshold_tracks(loc, FS, {**prov, "noise_rms_pA": prov["noise_rms_pA"] * 1.05},
                                   module_id="detect.schmitt")
    with pytest.raises(ThresholdsUnavailable, match="would not be the one"):
        detection_threshold_tracks(loc + 5.0, FS, prov, module_id="detect.schmitt")


# ================================================================================================
# C. Decimation keeps brief features.
# ================================================================================================
def test_extrema_decimation_keeps_a_one_sample_step_that_uniform_knots_lose():
    track = np.zeros(1_000_003)
    track[500_001] = 7.0
    idx, vals = decimate_extrema(track, 2_000)
    assert idx[0] == 0 and idx[-1] == track.size - 1
    assert idx.size <= 2_002
    assert vals.max() == 7.0 and 500_001 in idx
    assert np.all(np.diff(idx) > 0), "knots must stay in time order"
    _uidx, uvals = decimate_track(track, 2_000)
    assert uvals.max() == 0.0, "the uniform rule the baseline rail uses would have lost it"


@pytest.mark.parametrize("case", ["flat", "slope"])
def test_extrema_decimation_draws_a_mid_bin_step_at_its_own_sample(case):
    """A moving baseline can step where the event mask hands the short pass a change. Its min
    and max alone put the knot before the step at the bin's START, so the step drew as a ramp a
    bin wide (9.9 pA off at worst on a 10 pA step). On a SLOPE the step is neither the bin's min
    nor its max. The straight lines between the kept knots must BE the full-length track."""
    n = 3_000_000
    track = np.full(n, -800.0) if case == "flat" else np.linspace(-820.0, -780.0, n)
    track[1_500_150:] += 10.0
    idx, vals = decimate_extrema(track)
    assert idx.size <= 60_002 and np.all(np.diff(idx) > 0)
    drawn = np.interp(np.arange(n), idx, vals)
    np.testing.assert_allclose(drawn, track, atol=1e-6)
    # ...where the min/max rule alone did not: the case is a real one.
    _u, uvals = decimate_track(track, 2_000)
    assert np.abs(np.interp(np.arange(n), _u, uvals) - track).max() > 1.0


def test_extrema_decimation_keeps_its_budget():
    rng = np.random.default_rng(2)
    for n, m in ((1_000_003, 2_000), (10_007, 60), (5_000, 7)):
        idx, _v = decimate_extrema(rng.normal(size=n), m)
        assert idx.size <= m + 2 and idx[0] == 0 and idx[-1] == n - 1
        assert np.all(np.diff(idx) > 0)


def test_a_short_track_is_returned_whole():
    idx, vals = decimate_extrema(np.arange(10.0), 2_000)
    assert idx.tolist() == list(range(10))
    np.testing.assert_array_equal(vals, np.arange(10.0))


# ================================================================================================
# D. Which edges a person moved or drew.
# ================================================================================================
def _stamped_frame():
    raw = _trace(-1)
    df, _prov, _c, _b = _schmitt(raw)
    return stamp_original_bounds(df).copy()


def test_a_fresh_detection_has_no_marks():
    marks = boundary_marks(_stamped_frame())
    assert not marks.hand.any()
    assert not marks.start_corrected.any() and not marks.end_corrected.any()


def test_a_moved_edge_is_marked_on_its_own_side_and_revert_takes_it_away():
    df = _stamped_frame()
    df.loc[1, "start_idx"] = int(df.loc[1, "start_idx"]) + 5     # a mask drag, left edge
    df.loc[2, "end_idx"] = int(df.loc[3, "end_idx"])             # a merge stretched row 2
    marks = boundary_marks(df)
    assert marks.start_corrected.tolist()[1] and not marks.end_corrected.tolist()[1]
    assert marks.end_corrected.tolist()[2] and not marks.start_corrected.tolist()[2]
    assert int(marks.start_corrected.sum() + marks.end_corrected.sum()) == 2
    df.loc[1, "start_idx"] = int(df.loc[1, "orig_start_idx"])    # Revert
    assert not boundary_marks(df).start_corrected[1]


def test_a_real_merge_marks_the_far_edge_it_moved_and_not_the_near_one():
    """Through `merge.merge_by_gap`, not a hand-set column: one event broken into two
    fragments by a brief return toward the open pore, detected as two, merged back."""
    from nanoscope.analysis.events import EventSet
    from nanoscope.analysis.merge import merge_by_gap

    raw = _trace(-1)
    raw[60_000:60_400] -= 60.0
    raw[60_400:60_410] += 60.0          # a 10-sample recovery: two fragments, one passage
    raw[60_410:60_800] -= 60.0
    df, prov, _c, _b = _schmitt(raw)
    events = EventSet(stamp_original_bounds(df), prov)
    loc = detection_signal(raw, FS, prov)
    out = merge_by_gap(events, loc, FS, max_gap_s=0.001)
    merged_df = out[0].df if isinstance(out, tuple) else out.events.df
    row = merged_df.index[(merged_df["start_idx"] <= 60_005) & (merged_df["end_idx"] >= 60_790)]
    assert len(row) == 1, "the two fragments were not merged"
    marks = boundary_marks(merged_df)
    i = int(np.flatnonzero(merged_df.index == row[0])[0])
    assert not marks.start_corrected[i] and marks.end_corrected[i]
    assert int(marks.start_corrected.sum() + marks.end_corrected.sum()) == 1


def test_hand_made_rows_unknown_originals_and_mixed_merges():
    df = _stamped_frame()
    df[ADDED_COLUMN] = False
    df[SOURCE_COL] = pd.array(["detect.schmitt"] * len(df), dtype="string")
    df.loc[0, SOURCE_COL] = SOURCE_MANUAL                         # drawn by hand
    df.loc[0, ["orig_start_idx", "orig_end_idx"]] = pd.NA
    df.loc[0, ADDED_COLUMN] = True
    df.loc[4, ["orig_start_idx", "orig_end_idx"]] = pd.NA         # saved before the record
    df.loc[4, "start_idx"] = int(df.loc[4, "start_idx"]) - 3
    df.loc[5, SOURCE_COL] = "mixed"                               # detected + hand merge,
    df.loc[5, ADDED_COLUMN] = True                                # whose `added` a merge ORs,
    df.loc[5, ["orig_start_idx", "orig_end_idx"]] = pd.NA         # hand-made fragment first
    df.loc[6, SOURCE_COL] = "mixed"                               # mixed WITHOUT a hand row:
    df.loc[6, ["orig_start_idx", "orig_end_idx"]] = pd.NA         # two detectors agreeing
    marks = boundary_marks(df)
    assert marks.hand.tolist()[0] and not marks.start_corrected[0] and not marks.end_corrected[0]
    assert not marks.start_corrected[4], "unknown is not a correction"
    assert marks.start_corrected[5] and marks.end_corrected[5]
    assert not marks.start_corrected[6] and not marks.end_corrected[6], \
        "'mixed' alone does not say a person touched the row"
    assert int(marks.hand.sum()) == 1


def test_a_merge_with_a_hand_row_whose_pair_is_recorded_marks_only_the_edge_it_moved():
    """Detected fragment first: the merged row carries ITS original pair, so it is compared
    like any detected row. The start is the detector's own; the end, the hand row's, moved."""
    df = _stamped_frame()
    df[ADDED_COLUMN] = False
    df[SOURCE_COL] = pd.array(["detect.schmitt"] * len(df), dtype="string")
    df.loc[2, SOURCE_COL] = "mixed"
    df.loc[2, ADDED_COLUMN] = True
    df.loc[2, "end_idx"] = int(df.loc[2, "end_idx"]) + 400
    marks = boundary_marks(df)
    assert not marks.start_corrected[2] and marks.end_corrected[2]
    assert int(marks.start_corrected.sum() + marks.end_corrected.sum()) == 1
    assert not marks.hand.any()


def _reconcile_inputs(schmitt_df, cusum_df, prov):
    from nanoscope.analysis.events import EventSet, stamp_source_status
    from nanoscope.analysis.scoring import ensure_added_column
    from nanoscope.core.analysis import AnalysisResult

    out = []
    for module_id, df in (("detect.schmitt", schmitt_df), ("detect.cusum", cusum_df)):
        if SOURCE_COL not in df.columns:
            df = stamp_source_status(df, module_id)
        ensure_added_column(df)
        out.append(AnalysisResult(
            module_id=module_id, module_version="1", doc_id="doc-1", channel=0, params={},
            calibration_fingerprint=(1.0, 0.0), payload=EventSet(df, dict(prov)),
            baseline_fingerprint=prov.get("baseline_fingerprint")))
    return out


def test_a_reconciled_row_where_two_detectors_agreed_is_not_marked():
    """The review's case: a Schmitt table saved before the original pair existed, reconciled
    with a stamped CUSUM table. Every row comes back ``mixed`` and nobody edited any of them."""
    from nanoscope.analysis.reconcile import reconcile_detections

    raw = _trace(-1)
    df, prov, _c, _b = _schmitt(raw)
    legacy = df.copy()
    # Every other legacy row two samples shorter, so it wins its cluster (`reconcile._canonical`
    # prefers the shorter of two equally similar members) and the reconciled row carries the
    # legacy row's UNKNOWN original pair: the case that read "moved by hand".
    legacy.loc[::2, "end_idx"] = legacy.loc[::2, "end_idx"] - 2
    stamped = stamp_original_bounds(df).copy()
    out = reconcile_detections(_reconcile_inputs(legacy, stamped, prov)).result.payload.df
    assert (out[SOURCE_COL].astype(str) == "mixed").all() and not out[ADDED_COLUMN].any()
    assert out["orig_start_idx"].isna().any() and out["orig_start_idx"].notna().any()
    marks = boundary_marks(out)
    assert not marks.hand.any()
    assert not marks.start_corrected.any() and not marks.end_corrected.any()


def test_a_reconciled_row_whose_span_came_from_a_hand_made_event_is_marked_hand_made():
    """A curator's boundary wins the cluster (`reconcile._canonical`), so that row's edges ARE a
    person's although its combined source reads ``mixed`` and its ``added`` is False."""
    from nanoscope.analysis.reconcile import reconcile_detections

    raw = _trace(-1)
    df, prov, _c, _b = _schmitt(raw)
    with_hand = stamp_original_bounds(df).copy()
    with_hand[ADDED_COLUMN] = False
    with_hand[SOURCE_COL] = pd.array(["detect.schmitt"] * len(df), dtype="string")
    with_hand.loc[3, SOURCE_COL] = SOURCE_MANUAL                  # redrawn by hand
    with_hand.loc[3, ADDED_COLUMN] = True
    with_hand.loc[3, ["orig_start_idx", "orig_end_idx"]] = pd.NA
    with_hand.loc[3, "start_idx"] = int(with_hand.loc[3, "start_idx"]) - 20
    out = reconcile_detections(
        _reconcile_inputs(with_hand, stamp_original_bounds(df).copy(), prov)).result.payload.df
    marks = boundary_marks(out)
    hit = np.flatnonzero(out["start_idx"].to_numpy() == int(with_hand.loc[3, "start_idx"]))
    assert hit.size == 1, "the hand-made span did not win its cluster"
    assert str(out[SOURCE_COL].iloc[hit[0]]) == "mixed" and not bool(out[ADDED_COLUMN].iloc[hit[0]])
    assert marks.hand.tolist() == [i == hit[0] for i in range(len(out))]
    assert not marks.start_corrected.any() and not marks.end_corrected.any()


def test_a_table_without_the_original_columns_marks_only_hand_rows():
    df = _stamped_frame().drop(columns=["orig_start_idx", "orig_end_idx"])
    df[ADDED_COLUMN] = False
    df.loc[3, ADDED_COLUMN] = True
    marks = boundary_marks(df)
    assert marks.hand.tolist() == [i == 3 for i in range(len(df))]
    assert not marks.start_corrected.any() and not marks.end_corrected.any()
