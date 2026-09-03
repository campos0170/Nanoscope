"""CLOCK — the analysed-span clock at the seam, one step past every obvious case.

The wired half of this repair is ``tests/gui/test_rp_stationarity_clock_wired.py``:
a real ``MainWindow``, the real ML detector over the real "Visible view" default,
and the false sentence it printed. This file is the seam underneath it, and it is
written to go PAST the case that would have caught nothing:

* the obvious sub-range moves the START of the span. The one here that the
  shipped code gets wrong in a way no start-moving test can see is the span that
  starts at 0 and STOPS EARLY — a "Visible view" framed on the front of the
  recording — where the whole-file clock puts the boundary past the last event
  and calls every single one of them "early";
* the obvious provenance carries ``analysed_start_idx``. The one here that
  nothing else in the suite exercises is a stored ML result from an older build,
  which carries only ``range_start``/``range_stop``;
* the obvious table has one scope. The one here has three, because "every scope
  in this table shares ONE boundary" is the property that makes two scopes'
  ``early_frac`` comparable, and a per-scope clock would pass a single-scope test;
* the obvious guard is "the whole-recording path still works". The one here also
  pins that a stored span reaching past the trace it was handed is REFUSED rather
  than trusted, because that span would put the midpoint outside every onset.

WHAT IS FROZEN AND MUST NOT MOVE: both detectors stamp ``(0, n)`` on a
whole-recording detection, so this repair is a no-op there — asserted below
against the literal expression it replaced, on an EventSet, a bare frame and a
provenance-less set alike.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import nanoscope
from nanoscope.analysis import event_features as EF
from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import (
    ANALYSED_START_KEY,
    ANALYSED_STOP_KEY,
    REQUIRED,
    EventSet,
    span_provenance,
)

FS = 50_000.0


# ===========================================================================
# 0 — which tree is this?
# ===========================================================================


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    """Passes in BOTH trees; a failure below cannot be an import that never happened."""
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    for mod in (SR, EF):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents, f"{mod.__name__} at {p} is not under {root}"
        assert p.stat().st_size > 20_000, f"{p} is {p.stat().st_size} B — a stub?"
    print(f"\n[CLOCK/gates] nanoscope resolved from {root}")


# ===========================================================================
# builders
# ===========================================================================


def _rows(starts, dwell: int, depth: float, baseline: float) -> pd.DataFrame:
    """The frozen 20-column schema, filled honestly enough to build an EventSet."""
    rows = []
    for i, s in enumerate(starts):
        s = int(s)
        rows.append({
            "event_id": i, "recording": "r", "pore_id": "p1",
            "start_idx": s, "end_idx": s + dwell,
            "t_start_s": s / FS, "dwell_s": dwell / FS,
            "baseline_pA": baseline, "depth_pA": depth,
            "frac_depth": depth / baseline, "ecd_pA_s": depth * dwell / FS,
            "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / 1.2, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
    return pd.DataFrame(rows)[list(REQUIRED)]


def _recording(n_samples: int, starts, dwell: int = 90, depth: float = 40.0):
    """A trace with an event at each start, and the schema frame naming them."""
    rng = np.random.default_rng(7)
    baseline = 200.0
    x = np.full(n_samples, baseline) + rng.normal(0.0, 1.2, n_samples)
    for s in starts:
        x[int(s):int(s) + dwell] -= depth
    return (np.ascontiguousarray(x, dtype=np.float32),
            _rows(starts, dwell, depth, baseline))


def _set(df: pd.DataFrame, prov: dict | None) -> EventSet:
    full = {"analyzed_duration_s": 1.0}
    full.update(prov or {})
    return EventSet(df.copy(), full)


def _one_row() -> pd.DataFrame:
    return _rows([10], 10, 40.0, 200.0)


# ===========================================================================
# 1 — analysed_span_s: what clock, from what
# ===========================================================================


def test_the_span_comes_from_the_rows_own_provenance():
    df = _one_row()
    n = 1_000_000
    got = SR.analysed_span_s(_set(df, span_provenance(660_000, n)), n, FS)
    assert got == (13.2, 20.0)


def test_a_legacy_ml_result_carries_only_range_start_and_is_still_read():
    """The step past the canonical keys: a project saved before the span contract."""
    df = _one_row()
    n = 1_000_000
    legacy = _set(df, {"range_start": 200_000, "range_stop": 600_000})
    assert ANALYSED_START_KEY not in legacy.provenance
    assert SR.analysed_span_s(legacy, n, FS) == (4.0, 12.0)


@pytest.mark.parametrize("prov", [None, {}, {"detector": "schmitt_hysteresis"}])
def test_no_span_in_provenance_falls_back_to_the_whole_trace(prov):
    """A bare frame and a legacy set both take the clock this repair replaced."""
    df = _one_row()
    n = 500_000
    events = df if prov is None else _set(df, prov)
    assert SR.analysed_span_s(events, n, FS) == (0.0, n / FS)


def test_a_stored_span_reaching_past_the_trace_is_refused_not_trusted():
    """A span from a longer document would put the midpoint outside every onset."""
    df = _one_row()
    n = 100_000
    far = _set(df, {ANALYSED_START_KEY: 900_000, ANALYSED_STOP_KEY: 1_000_000})
    assert SR.analysed_span_s(far, n, FS) == (0.0, n / FS)
    # ...and one that merely overhangs the end is CLAMPED, not discarded.
    over = _set(df, {ANALYSED_START_KEY: 40_000, ANALYSED_STOP_KEY: 1_000_000})
    assert SR.analysed_span_s(over, n, FS) == (0.8, 2.0)


def test_a_clock_with_no_rate_is_refused():
    df = _one_row()
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="not a positive sample rate"):
            SR.analysed_span_s(_set(df, None), 1000, bad)


def test_the_whole_recording_span_is_bit_for_bit_the_expression_it_replaced():
    """THE FROZEN PATH. ``detect_events`` stamps ``(0, n)``; nothing may move there."""
    df = _one_row()
    for n in (1, 3, 7, 12_502_500, 1_100_000):
        old = (0.0, float(n) / float(FS))
        assert SR.analysed_span_s(_set(df, span_provenance(0, n)), n, FS) == old
        assert SR.analysed_span_s(df, n, FS) == old


# ===========================================================================
# 2 — the wiring: feature_table cuts the halves at the analysed span
# ===========================================================================


def _halves(trace, events, fs=FS):
    feats, _gate = SR.feature_table(trace, trace, fs, events, recording="r")
    return feats


def test_events_spread_evenly_over_a_sub_range_are_not_all_late():
    """THE DEFECT, at the seam. 300 events evenly spread over the last 40 % of the
    file; the whole-file clock said 0 % of them were early."""
    n = 1_000_000                                   # 20 s
    i0 = 600_000                                    # the detector looked at 12-20 s
    starts = np.linspace(i0 + 2_000, n - 4_000, 300).astype(int)
    trace, df = _recording(n, starts)
    feats = _halves(trace, _set(df, span_provenance(i0, n)))
    early = float((feats["time_half"] == "early").mean())
    print(f"\n[CLOCK/gates] sub-range early_frac = {early:.4f} over {len(feats)} events")
    assert 0.45 <= early <= 0.55, f"evenly spread over its own span, got {early:.4f}"
    # ...and the boundary really is the span's midpoint, not anything near it.
    mid = 0.5 * (i0 / FS + n / FS)
    assert np.array_equal(feats["time_half"].to_numpy() == "early",
                          feats["onset_s"].to_numpy() <= mid)


def test_a_span_that_starts_at_zero_and_stops_early_is_the_case_a_start_moving_test_misses():
    """ONE STEP PAST the obvious sub-range: the user framed the FRONT of the file.

    The span's own midpoint splits these events in half; the whole-file clock sits
    beyond the last of them and calls every one "early" — the mirror image of the
    defect the wired run reproduced, and invisible to any test that only moves the
    span's start.
    """
    n = 1_000_000
    i1 = 400_000                                    # detected in 0-8 s of a 20 s file
    starts = np.linspace(4_000, i1 - 4_000, 300).astype(int)
    trace, df = _recording(n, starts)
    on_span = _halves(trace, _set(df, span_provenance(0, i1)))
    whole = _halves(trace, df)                       # no provenance -> the old clock
    e_span = float((on_span["time_half"] == "early").mean())
    e_whole = float((whole["time_half"] == "early").mean())
    print(f"\n[CLOCK/gates] front-framed: span clock {e_span:.4f}, file clock {e_whole:.4f}")
    assert e_whole == 1.0, "fixture guard: the file clock should swallow them all"
    assert 0.45 <= e_span <= 0.55


def test_every_scope_in_one_table_shares_the_one_boundary():
    """Three scopes, one span, one midpoint — which is what makes two scopes'
    early_frac comparable at all. A per-scope clock passes a single-scope test."""
    n = 1_000_000
    i0 = 500_000
    starts = np.linspace(i0 + 2_000, n - 4_000, 300).astype(int)
    trace, df = _recording(n, starts)
    # A is the span's first third, B its middle third (which straddles the shared
    # midpoint), C its last third.
    scope = np.array(["A" if i < 100 else ("B" if i < 200 else "C")
                      for i in range(len(df))], dtype=object)
    feats, _ = SR.feature_table(trace, trace, FS, _set(df, span_provenance(i0, n)),
                                recording="r", scope=scope)
    mid = 0.5 * (i0 / FS + n / FS)
    for name, sub in feats.groupby("scope"):
        assert np.array_equal(sub["time_half"].to_numpy() == "early",
                              sub["onset_s"].to_numpy() <= mid), (
            f"scope {name} was cut somewhere other than the shared boundary")
    fracs = {k: float((v["time_half"] == "early").mean())
             for k, v in feats.groupby("scope")}
    print(f"\n[CLOCK/gates] per-scope early_frac on one boundary: {fracs}")
    assert fracs["A"] == 1.0        # wholly before the shared midpoint
    assert fracs["C"] == 0.0        # wholly after it
    assert 0.0 < fracs["B"] < 1.0   # the one that straddles it
    # ...and NONE of them is 0.50 by construction, which is what a per-scope clock
    # (or the median split this whole line of repair replaced) would have given all
    # three of them.
    assert sorted(fracs.values()) != [0.5, 0.5, 0.5]


def test_the_whole_recording_wiring_is_unchanged_by_the_repair():
    """The frozen path through the wiring: an EventSet stamped ``(0, n)`` and a
    bare frame produce the SAME column, and it is the whole-file split."""
    n = 2_000_000
    starts = np.linspace(2_000, 395_000, 300).astype(int)   # first 8 s of 40 s
    trace, df = _recording(n, starts)
    a = _halves(trace, df)
    b = _halves(trace, _set(df, span_provenance(0, n)))
    assert np.array_equal(a["time_half"].to_numpy(), b["time_half"].to_numpy())
    assert float((a["time_half"] == "early").mean()) == 1.0


# ===========================================================================
# 3 — the report says which clock it used
# ===========================================================================


def _report(trace, events, **kw):
    return SR.run_subpopulations(
        trace, trace, FS, events, recording="r",
        config=SR.SubpopConfig(k_grid=(1,), n_splits=2, n_boot=0,
                               with_tsne=False, with_umap=False),
        **kw,
    )


def test_the_report_names_the_span_its_halves_were_cut_at():
    n = 400_000
    i0 = 200_000
    starts = np.linspace(i0 + 2_000, n - 4_000, 60).astype(int)
    trace, df = _recording(n, starts)
    rep = _report(trace, _set(df, span_provenance(i0, n)), duration_s=(n - i0) / FS)
    assert rep.meta["span_start_s"] == pytest.approx(4.0)
    assert rep.meta["span_stop_s"] == pytest.approx(8.0)
    text = rep.text()
    assert "analysed span  : 4.000 - 8.000 s" in text, text[:600]
    assert "ANALYSED SPAN'S CLOCK" in text
    sub = [w for w in rep.warnings if w.startswith("SUB-RANGE:")]
    assert len(sub) == 1, rep.warnings
    print(f"\n[CLOCK/gates] {sub[0]}")
    assert "4.000-8.000 s of a 8.000 s recording" in sub[0]
    assert "50.0 %" in sub[0]


def test_the_printed_clock_is_the_column_s_clock_and_not_a_second_resolution():
    """ONE STEP PAST 'the header prints 4.000-8.000': the header must print what the
    COLUMN was cut at, even when the caller overrides the span — which is the only
    way to tell a read-back from a second, independent resolution that happens to
    agree."""
    n = 400_000
    starts = np.linspace(2_000, n - 4_000, 60).astype(int)
    trace, df = _recording(n, starts)
    # A span the provenance does NOT say: if the report re-resolved the clock it
    # would print the whole 0-8 s and contradict its own early_frac column.
    forced = (2.0, 6.0)
    feats, _ = SR.feature_table(trace, trace, FS, _set(df, span_provenance(0, n)),
                                recording="r", span_s=forced)
    assert feats.attrs["analysed_span_s"] == forced
    mid = 4.0
    assert np.array_equal(feats["time_half"].to_numpy() == "early",
                          feats["onset_s"].to_numpy() <= mid)
    rep = SR.run_subpopulations(
        trace, trace, FS, _set(df, span_provenance(0, n)), recording="r",
        span_s=forced, duration_s=n / FS,
        config=SR.SubpopConfig(k_grid=(1,), n_splits=2, n_boot=0,
                               with_tsne=False, with_umap=False),
    )
    assert (rep.meta["span_start_s"], rep.meta["span_stop_s"]) == forced
    assert "analysed span  : 2.000 - 6.000 s" in rep.text()
    assert np.array_equal(
        rep.features["time_half"].to_numpy() == "early",
        rep.features["onset_s"].to_numpy() <= mid)


def test_a_bare_duration_handed_to_the_override_is_refused_by_name():
    """``time_half`` takes a bare duration as a span, so a caller reaching for this
    override will hand one over. Indexing it said "'float' object is not
    subscriptable", which names neither the argument nor the two forms."""
    n = 200_000
    starts = np.linspace(2_000, n - 4_000, 40).astype(int)
    trace, df = _recording(n, starts)
    for bad in (4.0, "4.0", (1.0, 2.0, 3.0), object()):
        with pytest.raises(ValueError, match="is not a .t0, t1. pair of seconds"):
            SR.feature_table(trace, trace, FS, _set(df, None), span_s=bad)
    # ...and it is refused BEFORE the gate and the features run, not after them:
    # a wrong clock is a wrong report, and the caller should hear about it while
    # the answer is still ahead of them.
    calls: list[int] = []
    real = EF.sustained_gate
    try:
        EF.sustained_gate = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        with pytest.raises(ValueError, match="is not a .t0, t1. pair of seconds"):
            SR.feature_table(trace, trace, FS, _set(df, None), span_s=4.0)
    finally:
        EF.sustained_gate = real
    assert calls == [], "the gate ran before the clock was checked"


def test_the_clock_survives_on_the_table_all_the_way_to_the_report():
    """The report reads its clock BACK off the cut table rather than resolving a
    second one, so the attribute carrying it must survive the pipeline."""
    n = 400_000
    i0 = 100_000
    starts = np.linspace(i0 + 2_000, n - 4_000, 60).astype(int)
    trace, df = _recording(n, starts)
    rep = _report(trace, _set(df, span_provenance(i0, n)), duration_s=(n - i0) / FS)
    assert rep.features.attrs["analysed_span_s"] == (2.0, 8.0)
    assert (rep.meta["span_start_s"], rep.meta["span_stop_s"]) == (2.0, 8.0)


def test_a_whole_recording_report_grows_no_sub_range_caveat():
    """The frozen reports must not sprout a line. Both detectors stamp ``(0, n)``."""
    n = 400_000
    starts = np.linspace(2_000, n - 4_000, 60).astype(int)
    trace, df = _recording(n, starts)
    for events in (df, _set(df, span_provenance(0, n))):
        rep = _report(trace, events, duration_s=n / FS)
        assert not [w for w in rep.warnings if w.startswith("SUB-RANGE:")]
        assert rep.meta["span_start_s"] == 0.0
        assert rep.meta["span_stop_s"] == pytest.approx(n / FS)
        assert "analysed span  : 0.000 - 8.000 s" in rep.text()
