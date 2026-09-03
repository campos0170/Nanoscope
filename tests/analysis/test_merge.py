"""Tests for analysis.merge: post-detection sub-event merging.

Merging is a curation-style edit that stitches a shattered translocation back
together AFTER detection. These tests pin the contract:

* a run of fragments within the window collapses to ONE event spanning
  ``[first.start, last.end]``, re-characterized (dwell = full span, depth re-medianed);
* a far event stays separate; the merge is transitive along a chain;
* ``index_range`` limits which events are eligible;
* ``merge_region`` collapses everything overlapping the region regardless of gap, and
  is a no-op with fewer than two overlaps;
* a merged span must still BE one translocation: both entry points refuse a group whose
  fragments cover less than ``MIN_MERGE_OCCUPANCY`` of it, and say so in the returned
  ``MergeOutcome`` rather than reporting a refusal as "nothing to merge";
* ``accepted`` for a merged event is the CONSERVATIVE combination of the run (AND, not
  OR) and ``flag`` is re-derived + inherited rather than blanked -- see
  ``test_a_hand_rejected_fragment_rejects_the_merge`` and
  ``tests/analysis/test_flag_invariant.py``; ``event_id`` stays contiguous
  and unique; the frozen 20-column schema / dtypes are untouched; ``rate_hz`` survives;
* the grouping obeys the SAME gap rule as the detector's own ``merge_gap_s``.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.detect import (
    detect_events,
    gap_window_samples,
    group_starts_by_gap,
    split_low_occupancy_groups,
)
from nanoscope.analysis.events import SCHEMA, EventSet
from nanoscope.analysis.merge import (
    LOW_OCCUPANCY_TOKEN,
    MARGINAL_MERGE_OCCUPANCY,
    MIN_MERGE_OCCUPANCY,
    merge_by_gap,
    merge_region,
    span_occupancy,
)

FS = 50_000.0
I0 = -1000.0
DEPTH = 300.0
NOISE = 8.0


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _trace(spans, *, n=50_000, depth=DEPTH, polarity=-1, i0=I0, noise=NOISE, seed=0):
    """A flat-baseline current trace with square excursions over each [start, end) span."""
    rng = np.random.default_rng(seed)
    x = np.full(n, i0, dtype=np.float64) + rng.normal(0.0, noise, n)
    for s, e in spans:
        x[s:e] += polarity * depth
    return x.astype(np.float32)


def _detect(spans, **kw) -> tuple[EventSet, np.ndarray]:
    """Detect on a flat-baseline trace (explicit baseline, no detect-time merge)."""
    seed = kw.pop("seed", 0)
    n = kw.pop("n", 50_000)
    depth = kw.pop("depth", DEPTH)
    polarity = kw.pop("polarity", -1)
    i0 = kw.pop("i0", I0)
    noise = kw.pop("noise", NOISE)
    signal = _trace(spans, n=n, depth=depth, polarity=polarity, i0=i0, noise=noise, seed=seed)
    df, prov = detect_events(signal, FS, baseline=i0, min_dwell_s=1e-4, **kw)
    return EventSet(df, prov), signal


# Three fragments 20 samples apart (one translocation the detector split) + a far event.
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000)]
_FAR = (30_000, 30_300)


# ---------------------------------------------------------------------------
# merge_by_gap -- the headline behaviour
# ---------------------------------------------------------------------------
def test_three_fragments_merge_into_one_far_event_stays():
    es, signal = _detect(_FRAGMENTS + [_FAR])
    assert len(es) == 4

    frag = es.df[es.df["start_idx"] < 20_000]
    first_start = int(frag["start_idx"].min())
    last_end = int(frag["end_idx"].max())

    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3)   # window = 50 samples
    assert n == 2                       # 4 rows -> 2 rows: two merged away
    assert len(merged) == 2

    row = merged.df[merged.df["start_idx"] == first_start]
    assert len(row) == 1
    row = row.iloc[0]
    # dwell is the FULL span (first fragment start .. last fragment end).
    assert int(row["end_idx"]) == last_end
    assert float(row["dwell_s"]) == pytest.approx((last_end - first_start) / FS, abs=1 / FS)

    # the far event is untouched (same bounds, still present)
    assert (merged.df["start_idx"] == int(es.df[es.df["start_idx"] > 20_000]["start_idx"].iloc[0])).any()


def test_merged_depth_is_re_median_over_the_full_span():
    es, signal = _detect(_FRAGMENTS + [_FAR])
    frag = es.df[es.df["start_idx"] < 20_000]
    first_start, last_end = int(frag["start_idx"].min()), int(frag["end_idx"].max())

    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    row = merged.df[merged.df["start_idx"] == first_start].iloc[0]

    # The re-median over the whole span (including the recrossing gaps) recovers the
    # true blockade depth, NOT some fragment average or the sum.
    baseline = float(np.median(np.concatenate(
        [signal[first_start - 500:first_start], signal[last_end:last_end + 500]])))
    expected = float(np.median(-1.0 * (signal[first_start:last_end].astype(np.float64) - baseline)))
    assert float(row["depth_pA"]) == pytest.approx(expected, abs=3.0)
    assert float(row["depth_pA"]) == pytest.approx(DEPTH, abs=15.0)


def test_transitive_chain_collapses_to_one():
    # Four fragments, each within the window of the next: one chain -> one event.
    spans = [(10_000, 10_200), (10_230, 10_450), (10_470, 10_700), (10_730, 11_000)]
    es, signal = _detect(spans)
    assert len(es) == 4
    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert len(merged) == 1
    assert n == 3
    assert int(merged.df["end_idx"].iloc[0]) == int(es.df["end_idx"].max())


def test_far_apart_events_do_not_merge():
    spans = [(10_000, 10_300), (20_000, 20_300), (30_000, 30_300)]
    es, signal = _detect(spans)
    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3)   # 50-sample window
    assert n == 0
    assert merged is es                # true no-op returns the original


def test_index_range_restricts_scope():
    # Two clusters of three fragments each, far apart. Restrict to the first cluster.
    c1 = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000)]
    c2 = [(30_000, 30_200), (30_220, 30_500), (30_520, 31_000)]
    es, signal = _detect(c1 + c2)
    assert len(es) == 6

    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3, index_range=(9_000, 12_000))
    assert n == 2                      # only cluster 1 collapsed (3 -> 1)
    assert len(merged) == 4            # 1 merged + 3 untouched from cluster 2
    # cluster 2 is untouched: still three separate events in [29k, 32k]
    c2_rows = merged.df[(merged.df["start_idx"] > 20_000)]
    assert len(c2_rows) == 3


def test_a_hand_rejected_fragment_rejects_the_merge():
    """C1. ``accepted`` used to be the OR of the run -- "real if any fragment was" --
    which quietly REVERSED a curator's decision: reject two of three fragments, merge,
    and the merged event came back accepted. It is an AND now: one hand-rejected
    fragment rejects the whole merge, because a merge is not standing to overrule a
    person."""
    es, signal = _detect(_FRAGMENTS)
    df = es.df.sort_values("start_idx").reset_index(drop=True)
    df.loc[[0, 1], "accepted"] = False          # a bare reject: no flag, i.e. by hand
    es2 = EventSet(df, es.provenance)
    merged, _ = merge_by_gap(es2, signal, FS, max_gap_s=1e-3)
    assert len(merged) == 1
    assert bool(merged.df["accepted"].iloc[0]) is False

    # ...and rejecting every fragment obviously still rejects it.
    df.loc[:, "accepted"] = False
    es3 = EventSet(df, es.provenance)
    merged3, _ = merge_by_gap(es3, signal, FS, max_gap_s=1e-3)
    assert bool(merged3.df["accepted"].iloc[0]) is False

    # ...and an all-accepted run still merges to an accepted event.
    es4, sig4 = _detect(_FRAGMENTS)
    merged4, _ = merge_by_gap(es4, sig4, FS, max_gap_s=1e-3)
    assert bool(merged4.df["accepted"].iloc[0]) is True
    assert str(merged4.df["flag"].iloc[0]) == ""


def test_event_id_is_contiguous_and_unique_after_merge():
    es, signal = _detect(_FRAGMENTS + [_FAR])
    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    ids = merged.df["event_id"].to_numpy()
    assert list(ids) == list(range(len(merged)))
    assert len(set(ids.tolist())) == len(ids)


def test_schema_columns_and_dtypes_match_a_fresh_characterization():
    es, signal = _detect(_FRAGMENTS + [_FAR])
    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    # A freshly detected set is the reference schema.
    ref, _sig = _detect([(5_000, 5_300)])
    assert list(merged.df.columns) == list(SCHEMA)
    assert list(merged.df.columns) == list(ref.df.columns)
    for col in SCHEMA:
        assert merged.df[col].dtype == ref.df[col].dtype, col


def test_rate_hz_survives_a_merge():
    es, signal = _detect(_FRAGMENTS + [_FAR])
    dur = es.provenance["analyzed_duration_s"]
    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert merged.provenance["analyzed_duration_s"] == pytest.approx(dur)
    assert merged.rate_hz() == pytest.approx(len(merged) / dur)


def test_provenance_records_the_merge():
    es, signal = _detect(_FRAGMENTS + [_FAR])
    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    m = merged.provenance["merged"]
    assert m["rule"] == "gap"
    assert m["window_s"] == pytest.approx(1e-3)
    assert m["n_before"] == 4 and m["n_after"] == 2


# ---------------------------------------------------------------------------
# merge_region -- explicit selection overrides the gap threshold
# ---------------------------------------------------------------------------
def test_merge_region_collapses_all_in_region_regardless_of_gap():
    """The selection overrides the WINDOW: gaps far beyond any sensible merge window
    still collapse when the user drags over them.

    CHANGED FIXTURE (was 10k/14k/18k, i.e. ~3700-sample gaps between 300-sample events,
    an occupancy of 0.11): that set is now REFUSED by the occupancy floor, and is pinned
    as a refusal by ``test_merge_region_refuses_a_mostly_open_pore_selection`` below. The
    gaps here are 200 samples (4 ms) against a 0.15 ms default window -- still an order
    of magnitude beyond anything the window would merge, which is what this test is
    about -- while the span stays 69 % covered, so it is a REPAIR of one shattered
    translocation rather than three events glued to the open pore between them.
    """
    spans = [(10_000, 10_300), (10_500, 10_800), (11_000, 11_300)]
    es, signal = _detect(spans)
    assert len(es) == 3
    first_start = int(es.df["start_idx"].min())
    last_end = int(es.df["end_idx"].max())

    merged, n = merge_region(es, signal, FS, t0_s=9_000 / FS, t1_s=12_000 / FS)
    assert len(merged) == 1
    assert n == 2
    assert int(merged.df["start_idx"].iloc[0]) == first_start
    assert int(merged.df["end_idx"].iloc[0]) == last_end
    assert merged.provenance["merged"]["rule"] == "region"
    # ...and the gaps really are far outside the window, so "regardless of gap" is not vacuous.
    ungrouped, n_gap = merge_by_gap(es, signal, FS, max_gap_s=1.5e-4)
    assert n_gap == 0 and len(ungrouped) == 3


def test_merge_region_is_a_noop_with_fewer_than_two_overlaps():
    spans = [(10_000, 10_300), (30_000, 30_300)]
    es, signal = _detect(spans)
    # Region touching only ONE event -> no-op.
    merged, n = merge_region(es, signal, FS, t0_s=9_500 / FS, t1_s=10_800 / FS)
    assert n == 0
    assert merged is es
    # Region touching NO event -> no-op.
    merged2, n2 = merge_region(es, signal, FS, t0_s=15_000 / FS, t1_s=16_000 / FS)
    assert n2 == 0


def test_merge_region_leaves_events_outside_untouched():
    # The pair inside the region is 200 samples apart (was 3700): dense enough to pass the
    # occupancy floor, so this test keeps testing what it is named for -- that the event
    # OUTSIDE the region is not touched -- rather than the floor.
    spans = [(10_000, 10_300), (10_500, 10_800), (40_000, 40_300)]
    es, signal = _detect(spans)
    merged, n = merge_region(es, signal, FS, t0_s=9_000 / FS, t1_s=15_000 / FS)
    assert n == 1                      # the first two collapse
    assert len(merged) == 2
    # the lone far event survives with its bounds
    far = merged.df[merged.df["start_idx"] > 20_000]
    assert len(far) == 1
    assert int(far["end_idx"].iloc[0]) == 40_300


# ===========================================================================
# I10 — "the merging is not giving good results". Three reproduced defects:
# the grouping rule compared against the wrong end, transitivity had no brake,
# and merge_region had no brake at all.
# ===========================================================================

# -- (1) the grouping rule compares against the GROUP, not the predecessor ---
def test_group_starts_by_gap_carries_a_running_max_end():
    """A long event with a short one nested inside it must not split the group.

    ``gaps = s[1:] - e[:-1]`` compared every event with its IMMEDIATE predecessor's end.
    On these rows the third event starts 10 samples BEFORE the first one ends, and was
    still split off -- because it was measured against the nested short event's end
    (10400), giving a phantom gap of 9590 against a 50-sample window."""
    starts = np.array([10_000, 10_250, 19_990], dtype=np.int64)
    ends = np.array([20_000, 10_400, 20_100], dtype=np.int64)

    mask = group_starts_by_gap(starts, ends, gap_window_samples(1e-3, FS))  # 50 samples
    assert mask.tolist() == [True, False, False]                # ONE group, not two

    # ...and the group's span is the first start to the run's MAXIMUM end.
    assert (int(starts[mask][0]), int(ends.max())) == (10_000, 20_100)


def test_a_new_group_resets_the_running_max_end():
    """The running maximum must RESET at a group start -- ``np.maximum.accumulate`` would
    not, so one long early event would bind every later event to itself forever."""
    starts = np.array([0, 10, 100_000], dtype=np.int64)
    ends = np.array([50_000, 60, 100_100], dtype=np.int64)
    mask = group_starts_by_gap(starts, ends, gap_window_samples(1e-3, FS))
    assert mask.tolist() == [True, False, True]     # the far event is its own group


def test_the_gap_rule_is_unchanged_for_ordinary_non_overlapping_spans():
    """The fix must not move the ordinary case: consecutive, non-overlapping spans are
    grouped exactly as the vectorised rule grouped them."""
    starts = np.array([0, 100, 400, 1_000], dtype=np.int64)
    ends = np.array([50, 300, 500, 1_100], dtype=np.int64)
    for window in (0, 50, 100, 500, 5_000):
        got = group_starts_by_gap(starts, ends, window)
        want = np.ones(4, dtype=bool)
        want[1:] = (starts[1:] - ends[:-1]) > window
        assert got.tolist() == want.tolist(), window


# -- (2) transitivity has a brake: the occupancy floor -----------------------
#: 40 clean 0.4 ms events spaced 8 ms apart -- a perfectly ordinary train of
#: translocations, nothing shattered about it. At the 10 ms window the merge panel's
#: own text recommends, the transitive rule swallowed all 40 into ONE row: dwell
#: 312.40 ms, depth 0.70 pA against a 300 pA ground truth, accepted, flag ''.
_TRAIN = [(500 + i * 400, 520 + i * 400) for i in range(40)]


def test_a_train_of_clean_events_is_not_merged_into_one():
    es, signal = _detect(_TRAIN)
    assert len(es) == 40

    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-2)   # the recommended window
    assert n == 0, "a regularly spaced train of separate events is not one translocation"
    assert len(merged) == 40
    # every row is still its own event, with the true depth and dwell
    assert float(merged.df["depth_pA"].median()) == pytest.approx(DEPTH, abs=15.0)
    assert float(merged.df["dwell_s"].max()) < 1e-3


def test_the_refusal_is_reported_not_silent():
    es, signal = _detect(_TRAIN)
    out = merge_by_gap(es, signal, FS, max_gap_s=1e-2)
    # (eventset, n_merged) still unpacks -- the refusal rides along as an attribute.
    _events, n = out
    assert n == 0
    assert out.n_refused == 1
    assert out.refused_occupancy[0] == pytest.approx(20 / 400.0, abs=0.02)
    assert out.eventset is es and out.n_merged == 0


def test_occupancy_refusal_does_not_stop_the_genuine_merges_in_the_same_table():
    """A shattered event and a train in one table: the shattered one merges, the train
    is refused, and the refusal is counted rather than swallowed."""
    train = [(20_000 + i * 400, 20_020 + i * 400) for i in range(20)]
    es, signal = _detect(_FRAGMENTS + train)
    before = len(es)
    assert before == 3 + 20

    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-2)
    assert n == 2                                   # the three fragments -> one row
    assert len(merged) == before - 2
    out = merge_by_gap(es, signal, FS, max_gap_s=1e-2)
    assert out.n_refused == 1
    assert merged.provenance["merged"]["refused_low_occupancy"] == [
        pytest.approx(20 / 400.0, abs=0.02)]


def test_a_marginal_merge_is_kept_and_tokened_in_provenance():
    """Between the floor and :data:`MARGINAL_MERGE_OCCUPANCY` the merge is KEPT -- slow
    multi-level translocations really do look like that -- but it is recorded by name, so
    a table that drifted into runaway territory can be audited after the fact."""
    spans = [(10_000, 10_300), (10_500, 10_800), (11_000, 11_300)]   # 69 % occupied
    es, signal = _detect(spans)
    merged, n = merge_by_gap(es, signal, FS, max_gap_s=5e-3)
    assert n == 2 and len(merged) == 1

    token = merged.provenance["merged"][LOW_OCCUPANCY_TOKEN]
    assert len(token) == 1
    assert token[0]["start_idx"] == 10_000 and token[0]["end_idx"] == 11_300
    assert MIN_MERGE_OCCUPANCY <= token[0]["occupancy"] < MARGINAL_MERGE_OCCUPANCY

    # ...and a clean stitch leaves the provenance alone (no token, no refusals).
    es2, signal2 = _detect(_FRAGMENTS)
    clean, _ = merge_by_gap(es2, signal2, FS, max_gap_s=1e-3)
    assert LOW_OCCUPANCY_TOKEN not in clean.provenance["merged"]
    assert "refused_low_occupancy" not in clean.provenance["merged"]


def test_span_occupancy_is_coverage_over_the_merged_span():
    assert span_occupancy(np.array([0]), np.array([10])) == 1.0          # single event
    assert span_occupancy(np.array([0, 10]), np.array([5, 15])) == pytest.approx(10 / 15)
    # overlapping / nested spans must not manufacture occupancy above 1.
    assert span_occupancy(np.array([0, 2]), np.array([100, 4])) == 1.0
    assert span_occupancy(np.zeros(0), np.zeros(0)) == 1.0               # nothing to judge


def test_overlapping_sources_use_interval_union_for_the_occupancy_brake():
    """A nested duplicate must not hide the mostly-open-pore gap to a third event.

    Summing the three source lengths gives 209 / 260 = 0.804, above the clean-merge
    threshold.  Their interval union is only 110 / 260 = 0.423, below the refusal
    floor: samples 1..100 belong to both of the first two rows but exist only once in
    the trace.
    """
    starts = np.array([0, 1, 250])
    ends = np.array([100, 100, 260])

    occupancy = span_occupancy(starts, ends)
    assert occupancy == pytest.approx(110 / 260)
    assert occupancy < MIN_MERGE_OCCUPANCY

    groups, refused, marginal = split_low_occupancy_groups(
        starts, ends, np.array([True, False, False])
    )
    np.testing.assert_array_equal(groups, np.array([True, True, True]))
    assert refused == pytest.approx([110 / 260])
    assert marginal == []


# -- (3) merge_region has the same brake ------------------------------------
def test_merge_region_refuses_a_mostly_open_pore_selection():
    """One drag over two 4 ms events 600 ms apart used to produce a single 604 ms event
    of depth ~0.05 pA -- 0.017 % of the true blockade -- accepted, with a blank flag."""
    spans = [(5_000, 5_200), (35_000, 35_200)]      # 4 ms events, 600 ms apart
    es, signal = _detect(spans)
    assert len(es) == 2

    out = merge_region(es, signal, FS, t0_s=4_000 / FS, t1_s=36_000 / FS)
    merged, n = out
    assert n == 0
    assert merged is es                             # the table is untouched
    # ...and it is a REFUSAL, not the "fewer than two events overlap" no-op.
    assert out.n_refused == 1
    assert out.refused_occupancy[0] < MIN_MERGE_OCCUPANCY

    noop = merge_region(es, signal, FS, t0_s=4_000 / FS, t1_s=6_000 / FS)   # one event
    assert noop[1] == 0 and noop.n_refused == 0


# ---------------------------------------------------------------------------
# robustness / edge cases
# ---------------------------------------------------------------------------
def test_empty_eventset_is_a_noop():
    es, signal = _detect([(10_000, 10_300)])
    empty = EventSet(es.df.iloc[0:0].copy(), es.provenance)
    merged, n = merge_by_gap(empty, signal, FS, max_gap_s=1e-3)
    assert n == 0
    assert len(merged) == 0
    r_merged, r_n = merge_region(empty, signal, FS, t0_s=0.0, t1_s=1.0)
    assert r_n == 0 and len(r_merged) == 0


def test_single_event_is_a_noop():
    es, signal = _detect([(10_000, 10_300)])
    assert len(es) == 1
    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert n == 0
    assert merged is es


def test_events_flush_to_the_signal_edge_merge_cleanly():
    # First fragment starts at sample 0; a merge must not index out of bounds.
    spans = [(0, 300), (320, 600)]
    es, signal = _detect(spans, n=40_000)
    assert len(es) == 2
    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert n == 1
    assert len(merged) == 1
    row = merged.df.iloc[0]
    assert int(row["start_idx"]) == int(es.df["start_idx"].min())
    assert int(row["end_idx"]) == int(es.df["end_idx"].max())
    assert np.isfinite(float(row["depth_pA"]))


def test_polarity_enhancement_merges_with_positive_depth():
    # Upward (enhancement) events on a negative baseline: depth stays a positive magnitude.
    spans = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000)]
    es, signal = _detect(spans, polarity=1)
    assert es.provenance["polarity"] == 1
    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert len(merged) == 1
    assert float(merged.df["depth_pA"].iloc[0]) > 0


def test_non_schema_added_column_is_preserved_across_merge():
    es, signal = _detect(_FRAGMENTS + [_FAR])
    df = es.df.copy()
    df["added"] = False
    df.loc[df["start_idx"].idxmin(), "added"] = True    # a hand-added fragment
    es2 = EventSet(df, es.provenance)
    merged, _ = merge_by_gap(es2, signal, FS, max_gap_s=1e-3)
    assert "added" in merged.df.columns
    # the merged run inherited the added=True marker (OR across the run)
    row = merged.df.sort_values("start_idx").iloc[0]
    assert bool(row["added"]) is True


# ---------------------------------------------------------------------------
# ONE rule: post-hoc merge == detect-time merge_gap_s
# ---------------------------------------------------------------------------
def test_post_hoc_merge_matches_detect_time_merge_gap():
    signal = _trace(_FRAGMENTS + [_FAR])
    # detect-time merge
    df_dt, _ = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4, merge_gap_s=1e-3)
    # post-hoc merge of the un-merged detection
    df_raw, prov_raw = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    merged, _ = merge_by_gap(EventSet(df_raw, prov_raw), signal, FS, max_gap_s=1e-3)

    dt = df_dt.sort_values("start_idx").reset_index(drop=True)
    ph = merged.df.sort_values("start_idx").reset_index(drop=True)
    # Same number of events, same spans: the grouping rule is identical.
    assert len(dt) == len(ph)
    assert list(dt["start_idx"]) == list(ph["start_idx"])
    assert list(dt["end_idx"]) == list(ph["end_idx"])


# ===========================================================================
# N1 — the merged row's NOISE, and therefore its SNR.
#
# ``snr = depth_pA / noise_rms``, and every row the DETECTOR produced divides by one
# number: ``noise_rms(current - baseline_track)`` over the whole recording. A merged
# row took ``1.4826 * MAD`` of the RAW, pooled, un-detrended flank current instead --
# and ``filters.noise_rms``'s own docstring warns that MAD is taken about the median,
# so a drift or a level step inflates it. The flanks of a MERGED span are the worst
# case, because the pad is as wide as the merged event.
#
# Measured on the user's -600 mV recording, 349 merged spans, detector noise 3.68 pA:
# the pooled raw flank MAD came out at 43.3 pA (11.8x), so merged rows landed at a
# median SNR of 0.85 while the unmerged rows in the SAME table sat at 6.27 -- and any
# SNR-based curation threw the merged events out for being noisy. After the fix the
# merged rows read 8.68 against the same 6.27.
# ===========================================================================

def _burst_trace(n=60_000, *, seed=5, noise=NOISE):
    """A trace where the merged span's flanks sit at DIFFERENT open-pore levels.

    A slow level change under a burst of fragments is the shape that breaks a pooled
    flank MAD: each flank is quiet, but together they straddle a step, so the pooled
    median absolute deviation reports the step rather than the noise. A flat-baseline
    trace cannot show the bug at all -- which is why every test above stayed green.
    """
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64)
    x[30_000:] = I0 - 150.0                 # the open pore steps mid-burst
    x += rng.normal(0.0, noise, n)
    frags = [(29_600, 29_800), (29_850, 30_100), (30_150, 30_400)]
    for s, e in frags:
        x[s:e] += -1 * DEPTH
    for s, e in _ISOLATED:                  # far, isolated events: the unmerged control
        x[s:e] += -1 * DEPTH
    return x.astype(np.float32), frags


#: Isolated events far from the burst -- they pass through the merge untouched, so they
#: are the same-table control the merged rows' SNR has to be comparable with.
_ISOLATED = [(10_000, 10_300), (15_000, 15_300), (45_000, 45_300), (52_000, 52_300)]


def test_a_merged_rows_snr_is_on_the_same_scale_as_its_unmerged_neighbours():
    """The user-visible half: SNR-based curation must not single out merged rows."""
    signal, frags = _burst_trace()
    df, prov = detect_events(signal, FS, min_dwell_s=1e-4)
    es = EventSet(df, prov)
    merged, n = merge_by_gap(es, signal, FS, max_gap_s=2e-3)
    # TWO source rows absorbed: the detector emits all three fragments separately
    # (7 spans -- the 3 fragments plus the 4 isolated controls), and the 2 ms merge
    # collapses the run into ONE row spanning the whole burst, and therefore the step.
    #
    # An earlier note here asserted only ``n >= 1``, on the reasoning that "the mask
    # dilation bridges 1 ms gaps": the event-masked baseline dilates its mask by
    # ``dilate_ms = 1.0`` each side, so -- the argument went -- the burst's 1 ms
    # inter-fragment gaps were already closed and the detector emitted the last two
    # fragments as one event. That is wrong twice over, and the weakening it bought
    # is reverted here.
    #
    # 1. Dilation grows the mask the BASELINE interpolates across. It does not join
    #    two detected spans, which the Schmitt trigger cuts out of
    #    ``current - baseline``. Measured on this exact fixture: ``dilate_ms=0.0``
    #    gives the SAME seven spans -- only the baseline fingerprint moves
    #    (sha256:8e331d... -> sha256:aa5961...).
    # 2. The event-count collapse on the owner's -600 mV recording that motivated the
    #    weakening is caused by the baseline LEVEL, not by the dilation: turning the
    #    dilation off gives 263 events, FEWER than the 315 with it on, and a plain
    #    200 ms block median with no mask at all already gives 327.
    #
    # The merge does earn its keep on these gaps. Do not weaken this back.
    #
    # The margin is real, not knife-edge: the fixture's two inter-fragment gaps are 50 and
    # 100 samples (1.0 and 2.0 ms), so ``n == 2`` holds unchanged at max_gap_s = 1, 2, 3 and
    # 5 ms. Only ``dilate_ms=5`` breaks it, and it does so by wrecking the DETECTION (6
    # nonsense spans starting at 28,429 instead of the 7 real ones) -- which is the same
    # point again: what moves this fixture is the baseline the mask produces, not the
    # bridging of gaps.
    assert n == 2, (
        f"fixture must merge the whole three-fragment burst into one row "
        f"(2 source rows absorbed), got n={n} from {len(df)} detected spans"
    )

    spans = set(zip(df["start_idx"], df["end_idx"]))
    is_merged = np.array([(int(s), int(e)) not in spans
                          for s, e in zip(merged.df["start_idx"], merged.df["end_idx"])])
    assert is_merged.any() and (~is_merged).any()

    m = float(np.median(merged.df["snr"][is_merged]))
    u = float(np.median(merged.df["snr"][~is_merged]))
    assert m > 0.5 * u, (
        f"merged rows read SNR {m:.2f} against {u:.2f} for the unmerged rows in the "
        f"same table -- the merged row is being measured against a different noise"
    )


def test_a_merged_row_divides_by_the_detectors_own_noise():
    """States it exactly: the SNR denominator IS ``provenance['noise_rms_pA']``, the
    number every other row in the table used."""
    signal, _ = _burst_trace()
    df, prov = detect_events(signal, FS, min_dwell_s=1e-4)
    merged, n = merge_by_gap(EventSet(df, prov), signal, FS, max_gap_s=2e-3)
    assert n == 2        # all three fragments; see the note above the previous test

    spans = set(zip(df["start_idx"], df["end_idx"]))
    new = [r for _, r in merged.df.iterrows()
           if (int(r["start_idx"]), int(r["end_idx"])) not in spans]
    assert len(new) == 1, f"exactly one merged row expected, got {len(new)}"
    row = new[0]
    # ...and it spans the whole burst, across the open-pore step -- which is the only
    # reason this row's noise estimate is worth pinning.
    assert int(row["start_idx"]) <= 29_600 and int(row["end_idx"]) >= 30_400
    assert float(row["snr"]) == pytest.approx(
        float(row["depth_pA"]) / float(prov["noise_rms_pA"]), rel=1e-9
    )


def test_the_fallback_flank_noise_is_detrended_per_flank():
    """With no stored noise to fall back on, the local estimate must still be a NOISE
    estimate: the two flanks are detrended separately, so a step between them cannot
    masquerade as noise (MAD is shift-invariant, so removing one POOLED median -- what
    the baseline does -- would change nothing)."""
    from nanoscope.analysis.characterize import span_noise_rms

    rng = np.random.default_rng(11)
    left = rng.normal(-1000.0, 4.0, 3000)
    right = rng.normal(-1150.0, 4.0, 3000)      # 150 pA lower: a step, not noise
    window = np.concatenate([left, np.full(500, -1300.0), right])

    got = span_noise_rms(left, right, window, provenance=None)
    assert got == pytest.approx(4.0, rel=0.15), got
    # ...and the pooled raw MAD -- what it used to do -- is nowhere near it.
    pooled = float(1.4826 * np.median(np.abs(np.concatenate([left, right])
                                             - np.median(np.concatenate([left, right])))))
    assert pooled > 10 * got


def test_the_detectors_noise_wins_over_any_local_estimate():
    from nanoscope.analysis.characterize import span_noise_rms

    rng = np.random.default_rng(12)
    left = rng.normal(0.0, 4.0, 500)
    right = rng.normal(0.0, 4.0, 500)
    assert span_noise_rms(left, right, np.concatenate([left, right]),
                          {"noise_rms_pA": 7.25}) == pytest.approx(7.25)
    # ...but a missing / unusable stored noise falls through to the local estimate.
    for bad in ({}, {"noise_rms_pA": 0.0}, {"noise_rms_pA": None},
                {"noise_rms_pA": float("nan")}):
        got = span_noise_rms(left, right, np.concatenate([left, right]), bad)
        assert got == pytest.approx(4.0, rel=0.25), (bad, got)


def test_span_noise_is_always_finite_and_positive():
    """characterize refuses a non-positive noise_rms, so the helper must never hand it
    one -- not for empty flanks, not for a dead-flat window."""
    from nanoscope.analysis.characterize import span_noise_rms

    empty = np.zeros(0)
    assert span_noise_rms(empty, empty, empty, None) > 0
    assert span_noise_rms(empty, empty, np.zeros(100), None) > 0
    assert span_noise_rms(np.zeros(50), np.zeros(50), np.zeros(150), None) > 0
