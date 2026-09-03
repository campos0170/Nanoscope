"""S7-DET4 / S7 MISSED-1 / S7 MISSED-2 -- one invariant for row-derived provenance.

``AnalysisResult.provenance`` and ``.warnings`` carry facts that are STATEMENTS ABOUT
THE ROWS (``n_events``, ``n_filter_limited``, ``n_clipped_boundary`` and the sentences
``modules._build_warnings`` composes from them). Sweep 7 measured three doors that
changed the rows and left those facts stale, each having fixed ONE key at its own
call site:

* Merge all (317 -> 247 rows): ``n_events`` stayed 317; the Stats panel read "1 of
  317 event(s)..." over a 247-row table; the subpopulation report printed "71 of the
  detector's 317 region(s) were curated OUT" for one curated-out row (S7-DET4).
* Re-measure depths: the install replaced the result's warnings with the NEW
  baseline's list alone, deleting the filter-limited and boundary-clipped caveats
  (S7 MISSED-1); provenance kept ``n_filter_limited = 3`` over rows flagging 2, into
  the exported sidecar (S7 MISSED-2).

The fix is ONE function per layer -- `events.reconcile_row_counts` for the counts and
`modules.reconcile_result` for the counts + the rebuilt warnings -- called by the
merge engine, the re-measure engine and every GUI install path. These tests pin the
invariant at the engines and the resolver the report reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.baseline import DEFAULT_BASELINE, BaselineConfig
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis import events as EV
from nanoscope.analysis import modules as M
from nanoscope.analysis.events import CLIPPED_COUNT_KEY, FLAG_CLIPPED, EventSet
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.analysis.remeasure import remeasure_events
from nanoscope.core.analysis import AnalysisResult

FS = 50_000.0
I0 = -1000.0


def _signal(spans, *, n=60_000, depth=300.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] -= depth
    return x.astype(np.float32)


_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]


def _detected(x, **kw) -> EventSet:
    df, prov = detect_events(x, FS, baseline=I0, min_dwell_s=1e-4, **kw)
    return EventSet(df, prov)


def _result(events: EventSet, warnings=()) -> AnalysisResult:
    return AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id="doc", channel=0,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=events,
        scalars={"n_events": float(len(events.df)), "rate_hz": 0.0},
        warnings=tuple(warnings))


# ===========================================================================
# reconcile_row_counts -- the counts
# ===========================================================================
def test_every_row_derived_count_is_re_read_from_the_rows_and_absent_keys_stay_absent():
    df = pd.DataFrame({
        "flag": ["", FLAG_CLIPPED, "", ""],
        "filter_limited": [True, False, True, False],
        "accepted": [True, False, True, True],
    })
    prov = {"n_events": 317, "n_filter_limited": 3, CLIPPED_COUNT_KEY: 0, "merged": {"x": 1}}
    out = EV.reconcile_row_counts(prov, df)
    assert out["n_events"] == 4
    assert out["n_filter_limited"] == 2
    assert out[CLIPPED_COUNT_KEY] == 1
    assert out["merged"] == {"x": 1}, "history keys are not the function's to touch"
    assert prov["n_events"] == 317, "the caller's dict is never mutated"
    # A provenance that never carried a count does not grow one.
    assert "n_events" not in EV.reconcile_row_counts({"merged": {}}, df)
    assert EV.reconcile_row_counts(None, df) == {}
    assert EV.n_filter_limited(df) == 2


# ===========================================================================
# the merge engine -- n_events follows the merged table (S7-DET4)
# ===========================================================================
def test_the_merge_engine_restamps_n_events_to_the_merged_table():
    x = _signal(_FRAGMENTS)
    ev = _detected(x)
    assert len(ev.df) == 4 and ev.provenance["n_events"] == 4
    merged, n = merge_by_gap(ev, x, FS, max_gap_s=0.010)
    assert n == 2 and len(merged.df) == 2
    assert merged.provenance["n_events"] == 2, (
        "provenance['n_events'] must describe the merged table, or the subpopulation "
        "report reads the folded rows as rows the user curated OUT")
    assert merged.provenance["merged"]["n_before"] == 4
    assert merged.provenance["merged"]["n_after"] == 2


# ===========================================================================
# reconcile_result -- counts AND sentences, preserving what is not the rows'
# ===========================================================================
def _three_rows_one_clipped() -> pd.DataFrame:
    x = _signal(_FRAGMENTS)
    df = _detected(x).df.iloc[:3].copy().reset_index(drop=True)
    df.loc[0, "flag"] = FLAG_CLIPPED
    df.loc[0, "accepted"] = False
    df.loc[1, "filter_limited"] = True
    return df


def test_reconcile_result_rebuilds_the_composed_sentences_from_the_rows_and_keeps_the_baselines_own():
    x = _signal(_FRAGMENTS)
    ev = _detected(x)
    prov = dict(ev.provenance)
    prov.update({
        "n_events": 317, "n_filter_limited": 3, CLIPPED_COUNT_KEY: 1,
        "baseline_warnings": ["THE BASELINE'S OWN ADVISORY"],
        "merged": {"n_before": 317, "n_after": 247},
        "baseline_fingerprint": "sha256:abc",
    })
    stale = _result(EventSet(ev.df, prov), warnings=("1 of 317 event(s) ...",))
    new = M.reconcile_result(stale, EventSet(_three_rows_one_clipped(), prov))

    p = new.payload.provenance
    assert p["n_events"] == 3
    assert p["n_filter_limited"] == 1
    assert p[CLIPPED_COUNT_KEY] == 1
    assert p["merged"] == {"n_before": 317, "n_after": 247}
    assert p["baseline_fingerprint"] == "sha256:abc"
    joined = "\n".join(new.warnings)
    assert "1 of 3 event(s) were still above threshold" in joined
    assert "1 of 3 event(s) (33%) are filter-limited" in joined
    assert "THE BASELINE'S OWN ADVISORY" in joined, (
        "the baseline's own advisories travel in provenance['baseline_warnings'] and "
        "must survive a rebuild -- that is what the re-measure used to lose")
    assert "1 of 317" not in joined
    assert new.scalars["n_events"] == 3.0


def test_reconcile_result_refuses_to_have_its_derived_fields_overridden():
    x = _signal(_FRAGMENTS)
    res = _result(_detected(x))
    with pytest.raises(TypeError):
        M.reconcile_result(res, warnings=("x",))
    with pytest.raises(TypeError):
        M.reconcile_result(res, payload=res.payload)
    # ...but the caller's OWN fields ride through, as the re-measure install needs.
    out = M.reconcile_result(res, params={"k": 1}, baseline_fingerprint="sha256:new")
    assert out.params == {"k": 1} and out.baseline_fingerprint == "sha256:new"


# ===========================================================================
# the re-measure engine -- the new baseline's advisories are STAMPED, and the
# counts follow the rows (S7 MISSED-1 / MISSED-2)
# ===========================================================================
def test_the_remeasure_engine_stamps_the_new_baselines_warnings_and_re_reads_every_count():
    x = _signal(_FRAGMENTS, n=200_000)
    df, prov = detect_events(
        x, FS, baseline_config=BaselineConfig(method="block_median", window_s=0.01),
        polarity=-1, min_dwell_s=1e-4)
    prov = dict(prov)
    prov["n_filter_limited"] = 99          # a stale parallel count, deliberately wrong
    out = remeasure_events(EventSet(df, prov), x, FS, config=DEFAULT_BASELINE,
                           polarity=-1, param_source="proposed")
    p = out.events.provenance
    assert "baseline_warnings" in p, (
        "the new estimator's own advisories must ride in provenance, where "
        "modules._build_warnings finds them, or the install has nothing to rebuild from")
    assert list(p["baseline_warnings"]) == list(out.comparison.warnings)
    assert p["n_filter_limited"] == int(out.events.df["filter_limited"].sum())
    assert p["n_events"] == len(out.events.df)
    assert p[CLIPPED_COUNT_KEY] == 0
    assert p["baseline_param_source"] == "proposed"


# ===========================================================================
# the report's curation line reads the reconciled count, and says what it is
# ===========================================================================
def test_rows_in_hand_no_longer_calls_merged_away_rows_curated_out():
    x = _signal(_FRAGMENTS)
    ev = _detected(x)
    merged, _ = merge_by_gap(ev, x, FS, max_gap_s=0.010)
    df = merged.df.copy()
    df.loc[0, "accepted"] = False               # the user curates out exactly ONE
    es = EventSet(df, merged.provenance)
    meta, note = SR.rows_in_hand(es.accepted, int(df["accepted"].sum()))
    assert meta["n_detected"] == 2
    assert note is not None
    assert "1 of the 2 region(s) left in the table after a merge" in note
    assert "folded 4 rows into 2" in note
    assert "of the detector's" not in note, "a merged table is not the detector's"


def test_rows_in_hand_keeps_the_detectors_wording_when_nothing_was_merged():
    x = _signal(_FRAGMENTS)
    ev = _detected(x)
    df = ev.df.copy()
    df.loc[0, "accepted"] = False
    meta, note = SR.rows_in_hand(EventSet(df, ev.provenance).accepted, 3)
    assert "1 of the detector's 4 region(s)" in note
