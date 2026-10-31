"""A dwell or depth fit records WHICH values it fitted, and the judge compares them, not only how
many (docs/EDITING_FOUNDATION_DESIGN.md 7.2, decision 10).

THE GAP. ``judge_stored_fit`` (the Results dock's Fits tab) and ``stored_fit_problem`` (Export
figure) asked one question of the events: is the accepted count the one the fit recorded
(``n_input_events``)? Three ordinary edits keep that count and change the sample:

* a swap — one event rejected and another accepted;
* a resize — an edge dragged in the Event inspector, which moves that event's dwell;
* a re-measure — a new open-pore estimator, which moves every event's depth.

After any of them the tab went on printing the fit as current, and the export put it on the
figure.

THE FIX. ``fit_dwell`` / ``fit_depth`` record ``input_column`` (``dwell_s`` / ``frac_depth``) and
``input_digest`` — sha256 over the ``start_idx`` and that column of the rows the fitter was handed,
in event order — plus the table's ``fs`` and estimator stamp. The judge selects the same rows from
the table on screen and compares digests: a different digest is ``FIT_EVENTS_CHANGED`` at an equal
count, and the verdict's ``digest_moved`` says so. Only the fitted column counts, so a resize
leaves the depth fit current and a re-measure leaves the dwell fit current. A fit without a digest
(an earlier build, or the Plots window) keeps the count rule. An event is named by its start
because a merge renumbers ``event_id`` and re-sorts the rows: the first build hashed ids, and a
merge that touched no fitted event read as a change of all of them.

On 27ec0eb the swap, the resize and the re-measure all read "current" (measured), and their tests
below fail there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet, empty_df
from nanoscope.analysis.merge import merge_region
from nanoscope.analysis.modules import (
    FIT_CURRENT,
    FIT_EVENTS_CHANGED,
    fit_depth,
    fit_dwell,
    judge_stored_fit,
    stored_fit_problem,
)
from nanoscope.analysis.stats import fit_from_json, fit_to_json
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.project import _df_to_parquet_bytes, _parquet_bytes_to_df

FS = 50_000.0
T_MIN = 3e-5
BASELINE = "sha256:31991df77507"

#: The ``FitResult.meta`` keys this change adds, pinned by their spelling: a saved project is
#: read by older builds, so the names are part of the file format.
COLUMN_KEY = "input_column"
DIGEST_KEY = "input_digest"

#: What Export figure prints when it refuses a stored fit for its events.
EXPORT_STALE = "saved fit is stale (the accepted events changed since it ran)"


# ================================================================================================
# builders
# ================================================================================================

def _frame(n: int = 44, accepted: int = 40) -> pd.DataFrame:
    """``n`` events, the first ``accepted`` curated in: distinct dwells and depths, so a swap
    moves the sample even where it keeps the count."""
    rng = np.random.default_rng(0)
    dwell = T_MIN + rng.exponential(4e-4, n)
    df = empty_df().reindex(range(n)).reset_index(drop=True)
    df["event_id"] = np.arange(n, dtype=np.int64)
    df["start_idx"] = np.arange(n, dtype=np.int64) * 5_000
    df["end_idx"] = df["start_idx"] + np.round(dwell * FS).astype(np.int64)
    df["dwell_s"] = dwell
    df["baseline_pA"] = 1_000.0
    df["frac_depth"] = np.abs(rng.normal(0.3, 0.03, n))
    df["depth_pA"] = df["frac_depth"] * 1_000.0
    df["filter_limited"] = False
    df["accepted"] = np.r_[np.ones(accepted, dtype=bool), np.zeros(n - accepted, dtype=bool)]
    df["flag"] = ""
    df["recording"] = "synthetic"
    df["pore_id"] = "p1"
    return df


def _table(df: pd.DataFrame, provenance: dict | None = None) -> AnalysisResult:
    """A stored Schmitt detection holding ``df``."""
    return AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id="doc", channel=0, params={},
        calibration_fingerprint=(1.0, 0.0),
        payload=EventSet(df.copy(), dict(provenance or {"min_dwell_s": T_MIN, "fs": FS})),
        baseline_fingerprint=BASELINE)


#: The one translocation :func:`_detected` splits: two square fragments 20 samples apart.
_SPLIT = ((6_500, 6_620), (6_640, 6_900))


def _detected() -> tuple[EventSet, np.ndarray]:
    """A real detector's table and its trace: 44 square events of distinct widths on a flat
    baseline, and the :data:`_SPLIT` pair between the fifth and the sixth, so most events come
    after it."""
    spans = [(1_000 + 1_250 * i, 1_040 + 1_250 * i + 9 * i) for i in range(44)]
    rng = np.random.default_rng(0)
    signal = np.full(60_000, -1_000.0) + rng.normal(0.0, 8.0, 60_000)
    for s, e in sorted([*spans, *_SPLIT]):
        signal[s:e] -= 300.0
    signal = signal.astype(np.float32)
    df, prov = detect_events(signal, FS, baseline=-1_000.0, min_dwell_s=1e-4)
    return EventSet(df, prov), signal


def _fits(table: AnalysisResult, *, use_filter_limited: bool = False,
          ) -> tuple[AnalysisResult, AnalysisResult]:
    """The pair the Fit button makes of ``table``."""
    kw = dict(doc_id="doc", channel=0, n_boot=20, use_filter_limited=use_filter_limited,
              calibration_fingerprint=table.calibration_fingerprint,
              detection_module_id=table.module_id)
    return fit_dwell(table, **kw), fit_depth(table, **kw)


def _swapped(df: pd.DataFrame) -> pd.DataFrame:
    """Event 0 curated out and event 41 curated in: the same 40 accepted, not the same 40."""
    out = df.copy()
    out.loc[0, "accepted"] = False
    out.loc[41, "accepted"] = True
    return out


# ================================================================================================
# the stamp
# ================================================================================================

def test_each_fit_records_the_column_it_fitted_and_a_digest_of_the_rows_it_was_handed():
    table = _table(_frame())
    dwell, depth = _fits(table)

    assert dwell.payload.meta[COLUMN_KEY] == "dwell_s"
    assert depth.payload.meta[COLUMN_KEY] == "frac_depth"
    for fit in (dwell, depth):
        assert str(fit.payload.meta[DIGEST_KEY]).startswith("sha256:")
        # The table's own rate and estimator stamps, spelled as the table spells them.
        assert fit.payload.meta["fs"] == FS
        assert fit.payload.meta["baseline_fingerprint"] == BASELINE
        assert fit.payload.meta["n_input_events"] == 40        # the count is still recorded
    assert dwell.payload.meta[DIGEST_KEY] != depth.payload.meta[DIGEST_KEY]


def test_rows_whose_events_cannot_be_named_record_no_digest_and_keep_the_count_rule():
    """No integer ``start_idx`` (a hand-built frame): nothing to name the rows by, so no digest,
    and the judge falls back to the count exactly as before."""
    df = _frame()
    df["start_idx"] = np.nan
    table = _table(df)
    dwell, _ = _fits(table)

    assert DIGEST_KEY not in dwell.payload.meta
    assert judge_stored_fit(dwell, _table(_swapped(df))).reason == FIT_CURRENT


# ================================================================================================
# the verdict
# ================================================================================================

def test_a_table_that_has_not_changed_keeps_both_fits_current_through_a_project_round_trip():
    """No false alarm: the same events, and the same events read back from the project file
    (parquet for the table, JSON for the fit), hash as they did when they were fitted."""
    df = _frame()
    dwell, depth = _fits(_table(df))
    for fit in (dwell, depth):
        verdict = judge_stored_fit(fit, _table(df.copy()))
        assert verdict.reason == FIT_CURRENT and not verdict.digest_moved

    reopened_table = _table(_parquet_bytes_to_df(_df_to_parquet_bytes(df)))
    for fit in (dwell, depth):
        reopened_fit = AnalysisResult(
            module_id=fit.module_id, module_version="", doc_id="doc", channel=0, params={},
            calibration_fingerprint=fit.calibration_fingerprint,
            payload=fit_from_json(fit_to_json(fit.payload)))
        assert reopened_fit.payload.meta[DIGEST_KEY] == fit.payload.meta[DIGEST_KEY]
        assert judge_stored_fit(reopened_fit, reopened_table).reason == FIT_CURRENT


def test_merging_two_rejected_fragments_changes_no_fitted_event():
    """A merge renumbers ``event_id`` from 0 and re-sorts the rows by start
    (``merge._rebuild``). Two rejected fragments merge into one rejected row (``accepted`` is
    combined with AND), so the count stays and no fitted value moves. With the rows named by
    their ids, every accepted event after the pair read as changed: the Fits tab called both
    fits stale and Export figure refused them. Also here: an event an older build appended
    after the rest, which the merge moves back into start order."""
    events, signal = _detected()
    df = events.df.copy()
    split = df["start_idx"].between(_SPLIT[0][0] - 10, _SPLIT[1][1])
    assert int(split.sum()) == 2                                # the detector split it
    df.loc[split, "accepted"] = False
    df = pd.concat([df.drop(index=8), df.loc[[8]]], ignore_index=True)    # the appended event
    dwell, depth = _fits(_table(df, events.provenance))
    assert DIGEST_KEY in dwell.payload.meta and DIGEST_KEY in depth.payload.meta

    merged, n_merged = merge_region(EventSet(df, events.provenance), signal, FS,
                                    t0_s=(_SPLIT[0][0] - 10) / FS, t1_s=(_SPLIT[1][1] + 10) / FS)
    assert n_merged == 1
    # The same events with the same values, renamed and re-sorted.
    before = df[df["accepted"]].sort_values("start_idx")
    after = merged.df[merged.df["accepted"]]
    for column in ("start_idx", "dwell_s", "frac_depth"):
        assert np.array_equal(after[column].to_numpy(), before[column].to_numpy()), column
    assert not np.array_equal(after["event_id"].to_numpy(), before["event_id"].to_numpy())
    assert merged.df["start_idx"].is_monotonic_increasing
    assert not df["start_idx"].is_monotonic_increasing

    table = _table(merged.df, merged.provenance)
    for fit in (dwell, depth):
        verdict = judge_stored_fit(fit, table)
        assert verdict.reason == FIT_CURRENT and not verdict.digest_moved, verdict
        assert stored_fit_problem(fit, table) == ""


def test_swapping_one_accepted_event_for_another_at_the_same_count_changes_both_fits():
    df = _frame()
    dwell, depth = _fits(_table(df))
    swapped = _table(_swapped(df))

    for fit in (dwell, depth):
        verdict = judge_stored_fit(fit, swapped)
        assert verdict.reason == FIT_EVENTS_CHANGED, verdict
        assert (verdict.n_fit, verdict.n_now) == (40, 40)        # the count cannot see it
        assert verdict.digest_moved                              # ...so the verdict says what can
        assert verdict.made_on == "detect.schmitt"


def test_resizing_one_event_changes_the_dwell_fit_and_not_the_depth_fit():
    df = _frame()
    dwell, depth = _fits(_table(df))
    resized = df.copy()
    resized.loc[3, "dwell_s"] *= 1.5                             # an edge dragged in the inspector
    resized.loc[3, "end_idx"] = resized.loc[3, "start_idx"] + round(resized.loc[3, "dwell_s"] * FS)
    table = _table(resized)

    moved = judge_stored_fit(dwell, table)
    assert moved.reason == FIT_EVENTS_CHANGED, moved
    assert moved.digest_moved and moved.n_fit == moved.n_now == 40
    assert judge_stored_fit(depth, table).reason == FIT_CURRENT


def test_a_remeasure_changes_the_depth_fit_and_not_the_dwell_fit():
    """A new open-pore estimator moves every depth and no dwell."""
    df = _frame()
    dwell, depth = _fits(_table(df))
    remeasured = df.copy()
    remeasured["baseline_pA"] = 1_000.0 / 1.07
    remeasured["frac_depth"] = remeasured["frac_depth"] * 1.07
    table = _table(remeasured)

    moved = judge_stored_fit(depth, table)
    assert moved.reason == FIT_EVENTS_CHANGED, moved
    assert moved.digest_moved and moved.n_fit == moved.n_now == 40
    assert judge_stored_fit(dwell, table).reason == FIT_CURRENT


def test_rows_the_fit_excluded_are_not_part_of_what_it_fitted():
    """The digest covers the rows the fitter was HANDED. A filter-limited event is excluded by
    default, so resizing it moves nothing the fit used; a fit that took filter-limited events
    in is judged on them. The count rule still stands beside the digest: rejecting the excluded
    event changes no fitted value, but it changes the count the fit recorded."""
    df = _frame()
    df.loc[2, "filter_limited"] = True
    dwell, depth = _fits(_table(df))
    dwell_all, depth_all = _fits(_table(df), use_filter_limited=True)
    # The judge selects the rows by the fit's own choice: a fit that took the filter-limited
    # event in is current on the table it was made from.
    assert judge_stored_fit(dwell_all, _table(df)).reason == FIT_CURRENT
    assert judge_stored_fit(depth_all, _table(df)).reason == FIT_CURRENT

    resized = df.copy()
    resized.loc[2, "dwell_s"] *= 1.5
    table = _table(resized)

    assert judge_stored_fit(dwell, table).reason == FIT_CURRENT
    assert judge_stored_fit(depth, table).reason == FIT_CURRENT
    assert judge_stored_fit(dwell_all, table).reason == FIT_EVENTS_CHANGED

    rejected = df.copy()
    rejected.loc[2, "accepted"] = False
    verdict = judge_stored_fit(dwell, _table(rejected))
    assert verdict.reason == FIT_EVENTS_CHANGED
    assert (verdict.n_fit, verdict.n_now) == (40, 39) and not verdict.digest_moved


def test_a_fit_without_a_digest_keeps_the_count_rule():
    """What an earlier build saved: no digest, so a swap is invisible to it, exactly as before,
    and a change of count is still caught."""
    df = _frame()
    dwell, depth = _fits(_table(df))
    for fit in (dwell, depth):
        fit.payload.meta.pop(DIGEST_KEY)
        fit.payload.meta.pop(COLUMN_KEY)

        assert judge_stored_fit(fit, _table(_swapped(df))).reason == FIT_CURRENT
        curated = df.copy()
        curated.loc[0, "accepted"] = False
        verdict = judge_stored_fit(fit, _table(curated))
        assert verdict.reason == FIT_EVENTS_CHANGED
        assert (verdict.n_fit, verdict.n_now) == (40, 39) and not verdict.digest_moved


# ================================================================================================
# the export applies the same rule, in its own words
# ================================================================================================

def test_the_export_refuses_a_fit_of_events_that_changed_at_the_same_count():
    df = _frame()
    dwell, depth = _fits(_table(df))
    resized = df.copy()
    resized.loc[3, "dwell_s"] *= 1.5

    assert stored_fit_problem(dwell, _table(df)) == ""
    assert stored_fit_problem(dwell, _table(_swapped(df))) == EXPORT_STALE
    assert stored_fit_problem(depth, _table(_swapped(df))) == EXPORT_STALE
    assert stored_fit_problem(dwell, _table(resized)) == EXPORT_STALE
    assert stored_fit_problem(depth, _table(resized)) == ""      # its column did not move
