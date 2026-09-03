"""Migration: record, mark stale, offer — never silently re-measure, never refuse.

The engine half of §4.2. Everything the GUI offers has to be true here first:

* a re-measure rewrites amplitudes and NOTHING else — the geometry guard is a raise,
  not a docstring, and this file proves it fires;
* curation survives it, through `events.inherited_curation` reached by handing each
  row to `characterize.recharacterize_span` as its own ``prior``;
* the ``baseline_`` provenance NAMESPACE is replaced, not the eight parameter keys,
  so a stale health block cannot describe an estimator that never ran;
* the comparison report and the button that acts on it are ONE computation, so the
  report cannot describe something the button would not do;
* on a recording without the pathology the report says ~1.0x and means it.

This module is new, so nothing here has a "before" to fail against; the two
regressions this change fixes are in the GUI half (``tests/gui/test_baseline_migration.py``
— the restore that dropped the stamp, and the staleness check that never looked at
it), and both are marked FAILING BEFORE there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import remeasure as R
from nanoscope.analysis.baseline import DEFAULT_BASELINE, BaselineConfig
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    FLAG_NON_FINITE,
    SCHEMA,
    EventSet,
    flag_tokens,
    has_flag,
)

FS = 50_000.0

#: The estimator every pre-change row in this project was measured by. Named here
#: once, as the fixture's "before", so no test invents its own idea of the old world.
OLD = BaselineConfig(method="block_median", window_s=0.01)

APO = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")


# ===========================================================================
# fixtures
# ===========================================================================
def bursty(n=400_000, i0=-100.0, depth=60.0, noise=2.0, dwell=250, period=2000,
           seed=3) -> np.ndarray:
    """A trace whose events are long against half a 10 ms block.

    ``dwell=250`` (5 ms) at ``period=2000`` (40 ms) is deliberately chosen: the block
    median is badly wrong (it stamps 34.8 pA where the true depth is 60.0) while the
    detection still finds 201 events against 200 real ones, so the geometry the
    re-measure is asked to preserve is real geometry rather than an artefact of the
    broken baseline. Pushed harder (33% duty) the OLD estimator loses a third of the
    events outright, which is a detection story, not a migration one.
    """
    rng = np.random.default_rng(seed)
    x = np.full(n, i0) + rng.normal(0, noise, n)
    t = 500
    while t + dwell < n:
        x[t:t + dwell] -= depth
        t += period
    return x.astype(np.float32)


def clustered(n=400_000, i0=-100.0, depth=60.0, noise=2.0, dwell=150, gap=60,
              per_burst=4, period=3000, seed=7) -> np.ndarray:
    """Bursts of events with SUB-PAD gaps between them — the owner's own structure.

    ``rise_s``/``fall_s`` clip their search window to the neighbouring events
    (``_EDGE_PAD_SAMPLES`` = 64), so a defect in that bound is only reachable when a
    neighbour is inside the pad. On the apo recording 1399 of 1641 events are; on
    ``bursty`` (40 ms apart) none are. Four 3 ms events separated by 1.2 ms, every
    60 ms: 400 detected events, 133 of them with a neighbour inside the pad, 20%
    duty, so the old block median is still a usable "before".

    Measured with the fix removed (``_restore_set_wide_rise_fall`` stubbed out), a
    same-config re-measure on this fixture moves ``rise_s`` on 39 rows and ``fall_s``
    on 133; with it, on none. The same run on ``bursty`` moves nothing either way,
    which is why the fixture exists.
    """
    rng = np.random.default_rng(seed)
    x = np.full(n, i0) + rng.normal(0, noise, n)
    t = 500
    while t + per_burst * (dwell + gap) < n:
        u = t
        for _ in range(per_burst):
            x[u:u + dwell] -= depth
            u += dwell + gap
        t += period
    return x.astype(np.float32)


def clean(n=400_000, **kw) -> np.ndarray:
    """Short, sparse events: the block median lands on the open pore and the two
    estimators agree. This is the "re-run without anxiety" file of §4.2.7."""
    return bursty(n=n, dwell=150, period=1000, **kw)


def detected(x: np.ndarray, config: BaselineConfig = OLD, fs: float = FS) -> EventSet:
    df, prov = detect_events(x, fs, baseline_config=config, polarity=-1,
                             min_dwell_s=1e-4, recording="rec", pore_id="pore")
    return EventSet(df, prov)


def curated(ev: EventSet, *, reject=(2, 3), note=4) -> EventSet:
    """A hand rejection and a curator's own flag token — the two things §4.2.3 says
    a re-measure must carry over."""
    df = ev.df.copy()
    for i in reject:
        df.loc[i, "accepted"] = False
    df.loc[note, "flag"] = "curator_note"
    return EventSet(df, dict(ev.provenance))


# ===========================================================================
# 1. WHAT A RE-MEASURE MOVES, AND WHAT IT MAY NOT
# ===========================================================================
def test_the_amplitudes_move_and_the_geometry_does_not():
    x = bursty()
    ev = detected(x)
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    before, after = ev.df, out.events.df

    for col in R.GEOMETRY_COLUMNS + ("event_id", "recording", "pore_id"):
        assert (before[col].to_numpy() == after[col].to_numpy()).all(), col
    # ...and the amplitudes did move, or the test above is pinning a no-op.
    assert float(np.median(after["depth_pA"])) > 1.5 * float(np.median(before["depth_pA"]))
    assert set(after.columns) == set(SCHEMA) | (set(before.columns) - set(SCHEMA))
    assert list(after.columns) == list(before.columns)


def test_a_same_config_remeasure_moves_no_column_at_all():
    """Re-measure against the estimator the rows were ALREADY measured by, and every
    one of the 20 columns must come back bit-identical. Nothing was measured
    differently, so nothing may move.

    This is the test that was missing, and it is the reason a real defect shipped.
    ``rise_s`` and ``fall_s`` were declared unaffected in the release note, badged as
    unaffected in the table, and excluded from the geometry guard -- three deliberate
    decisions, all derived from the same untested assumption about which columns a
    rebuild touches. Measured on the owner's 1641-event apo detection, re-measured
    against its own stored config: ``baseline_pA``, ``depth_pA``, ``ecd_pA_s``,
    ``snr`` and ``dwell_s`` were bit-identical while ``rise_s`` moved on **812** rows
    and ``fall_s`` on **920**, 331 and 341 of them from a finite value to NaN.

    The cause was structural, not arithmetic: ``rise_fall_over_set`` clips its
    search window to the NEIGHBOURING events, and a row-by-row rebuild presents it
    with one event, so the bound degenerates. Split by neighbour distance it is
    unambiguous -- 812 of the 1399 rows with a neighbour inside the +-64-sample pad
    moved, and 0 of the 83 isolated ones. ``_restore_set_wide_rise_fall`` recomputes
    the whole table in one pass, which is the only shape in which the answer is the
    one the detector gave.

    The fixture is `clustered`, not `bursty`: on a trace whose events are 40 ms
    apart every search window is isolated and this test passes with the defect still
    in place. It is checked at the bottom that the fixture really does have
    neighbours inside the pad.
    """
    x = clustered()
    ev = detected(x, OLD)
    stored = BaselineConfig.reproducible_from(ev.provenance)
    assert stored is not None and stored.fingerprint(FS) == OLD.fingerprint(FS)

    out = R.remeasure_events(ev, x, FS, config=stored, polarity=-1)
    before, after = ev.df, out.events.df
    assert list(after.columns) == list(before.columns)

    moved = []
    for col in SCHEMA:
        a, b = before[col].to_numpy(), after[col].to_numpy()
        if a.dtype.kind == "f":
            same = np.array_equal(a, b, equal_nan=True)
        else:
            same = np.array_equal(a, b)
        if not same:
            n_bad = int((~((a == b) | (pd.isna(a) & pd.isna(b)))).sum())
            moved.append(f"{col} ({n_bad} rows)")
    assert not moved, (
        "a re-measure against the rows' OWN estimator changed " + ", ".join(moved)
    )

    # ...and the fixture really does have the neighbour density that makes the
    # defect reachable, or the assertion above pins nothing. Both halves matter: a
    # neighbour inside the +-64-sample pad is what the lost bound reads into, and a
    # measurable rise time is what there is to lose.
    s = before["start_idx"].to_numpy()
    e = before["end_idx"].to_numpy()
    gaps = s[1:] - e[:-1]
    assert int((gaps <= 64).sum()) > 100, (
        f"only {int((gaps <= 64).sum())} rows have a neighbour inside the search "
        "pad; this fixture cannot reach the defect"
    )
    assert float(np.isfinite(before["rise_s"].to_numpy()).mean()) > 0.5, (
        "the fixture has no measurable rise times, so this test cannot see them move"
    )


def test_the_geometry_guard_raises_rather_than_returning_a_moved_table():
    """The promise the offered button makes is "it re-measures, it does not
    re-detect". A silent boundary shift would break it in the one way the user
    cannot see: the curation would still be attached, to different events.

    Asserted by corrupting the rebuilt row on its way out, so the guard is proved to
    FIRE rather than merely to exist.
    """
    x = bursty()
    ev = detected(x)
    real = R.recharacterize_span

    def shifted(*a, **kw):
        row = real(*a, **kw)
        row = row.copy()
        row.at[row.index[0], "end_idx"] = int(row["end_idx"].iloc[0]) + 1
        return row

    R.recharacterize_span = shifted
    try:
        with pytest.raises(ValueError, match="re-measure moved 'end_idx'"):
            R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    finally:
        R.recharacterize_span = real


def test_event_id_is_identity_not_a_measurement():
    """A merge reassigns ``event_id`` because a merge destroys rows. A re-measure
    keeps every row, so reassigning would break every curation reference, plot
    selection and Event Inspector target keyed on it."""
    x = bursty()
    ev = detected(x)
    df = ev.df.copy()
    # A hand-added event appends out of start order; the answer must still come back
    # in the INPUT's row order with the INPUT's ids.
    df = pd.concat([df.iloc[5:], df.iloc[:5]], ignore_index=False)
    ev = EventSet(df.reset_index(drop=True), dict(ev.provenance))
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    assert list(out.events.df["event_id"]) == list(ev.df["event_id"])
    assert list(out.events.df["start_idx"]) == list(ev.df["start_idx"])


# ===========================================================================
# 2. CURATION SURVIVES
# ===========================================================================
def test_a_hand_rejection_and_a_curator_flag_survive_a_remeasure():
    x = bursty()
    ev = curated(detected(x))
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    after = out.events.df
    assert not bool(after.loc[2, "accepted"])
    assert not bool(after.loc[3, "accepted"])
    assert has_flag(after.loc[4, "flag"], "curator_note")
    # ...and the rows nobody touched are still accepted: `accepted` is combined with
    # AND, which must not mean "reject everything". (Row 0 of this fixture is a
    # boundary-clipped noise excursion the detector rejects on its own merits, so it
    # is not an untouched row and must not be used as one.)
    assert bool(after.loc[1, "accepted"]) and bool(after.loc[5, "accepted"])
    assert int(after["accepted"].sum()) == int(ev.df["accepted"].sum())


def test_an_automatic_rejection_is_re_judged_not_inherited():
    """A row rejected only because a rule fired is re-judged from the new
    measurement; inheriting it would leave the row rejected with its reason erased,
    invisible to the very filter meant to find it. Here the rule is ``non_finite``,
    which the new measurement no longer triggers."""
    x = bursty()
    ev = detected(x)
    df = ev.df.copy()
    df.loc[1, "accepted"] = False
    df.loc[1, "flag"] = FLAG_NON_FINITE          # an AUTOMATIC reason, re-derivable
    out = R.remeasure_events(EventSet(df, dict(ev.provenance)), x, FS,
                             config=DEFAULT_BASELINE, polarity=-1)
    after = out.events.df
    assert bool(after.loc[1, "accepted"])
    assert FLAG_NON_FINITE not in flag_tokens(after.loc[1, "flag"])


# ===========================================================================
# 3. PROVENANCE — the namespace is the boundary
# ===========================================================================
def test_the_whole_baseline_namespace_is_replaced_not_the_parameter_keys():
    """`BaselineResult.provenance` emits DERIVED keys beside the eight parameters.
    Updating only the parameters leaves a health block and a fallback reason
    describing an estimator that no longer measured anything here — the same lesson
    ``modules._detect_kwargs`` learned about the derived fingerprint."""
    x = bursty()
    ev = detected(x)
    prov = dict(ev.provenance)
    prov["baseline_fallback"] = "a reason from the previous estimator"
    prov["baseline_health"] = {"stale": True}
    ev = EventSet(ev.df, prov)

    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    new = out.events.provenance
    assert new.get("baseline_fallback") is None
    assert new["baseline_health"] != {"stale": True}
    assert new["baseline_fingerprint"] == DEFAULT_BASELINE.fingerprint(FS)
    assert new["baseline_method"] == "masked_median"
    # everything OUTSIDE the namespace is untouched: the noise scale, the analysed
    # span, the polarity and the filter rise are properties of the DETECTION.
    for key in ("noise_rms_pA", "polarity", "analyzed_duration_s"):
        if key in ev.provenance:
            assert new[key] == ev.provenance[key], key


def test_the_remeasure_record_names_where_the_table_came_from():
    x = bursty()
    ev = detected(x)
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    rec = out.events.provenance[R.REMEASURED_KEY]
    assert rec["n_events"] == len(ev.df)
    assert rec["from_fingerprint"] == ev.provenance["baseline_fingerprint"]
    assert rec["to_fingerprint"] == DEFAULT_BASELINE.fingerprint(FS)
    assert "block_median" in rec["from_description"]
    assert rec["columns"] == list(R.REMEASURED_COLUMNS)
    assert json.loads(json.dumps(rec)) == rec        # a project persists this verbatim


def test_the_export_sidecar_contract_carries_the_remeasure_record():
    """Without it a migrated CSV and a freshly detected one are indistinguishable in
    the sidecar, which is exactly what old exports cannot answer."""
    from nanoscope.analysis.scoring import BASELINE_CONTRACT_KEYS

    assert R.REMEASURED_KEY in BASELINE_CONTRACT_KEYS


# ===========================================================================
# 4. THE REPORT AND THE BUTTON ARE ONE COMPUTATION
# ===========================================================================
def test_compare_returns_exactly_the_remeasure_s_own_comparison():
    x = bursty()
    ev = detected(x)
    report = R.compare_estimators(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    acted = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    assert np.array_equal(report.new_depth_pA, acted.comparison.new_depth_pA)
    assert np.array_equal(report.new_depth_pA, acted.events.df["depth_pA"].to_numpy())
    assert report.median_ratio == acted.comparison.median_ratio


def test_compare_installs_nothing():
    x = bursty()
    ev = curated(detected(x))
    before = ev.df.copy(deep=True)
    prov_before = dict(ev.provenance)
    R.compare_estimators(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    pd.testing.assert_frame_equal(ev.df, before)
    assert dict(ev.provenance) == prov_before


def test_on_a_pathological_recording_the_report_states_the_factor():
    x = bursty()
    ev = detected(x)
    c = R.compare_estimators(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    # Measured on this fixture: 34.8 -> 60.0 pA, ratio 1.73, every baseline moved.
    assert c.n_events == len(ev.df)
    assert c.median_ratio > 1.5
    assert c.n_baseline_moved == c.n_events
    assert c.frac_baseline_moved == 1.0
    assert c.changed
    assert "block_median" in c.old_description
    assert "masked median" in c.new_description
    assert f"{c.n_baseline_moved} of {c.n_events}" in c.summary()


def test_on_a_clean_recording_the_report_says_so_and_means_it():
    """§4.2.7's whole point: the app cannot tell a pathological recording from a
    clean one by looking at a stored row, so the report must be able to say "nothing
    happens here" as confidently as it says "15x"."""
    x = clean()
    ev = detected(x)
    c = R.compare_estimators(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    # Measured: 1.013 here against 1.73 on `bursty` and 4.73 on the owner's own apo
    # file. The two regimes are two orders of magnitude apart in this statistic,
    # which is why `MATERIAL_RATIO_TOLERANCE`'s exact value is not load-bearing.
    assert abs(c.median_ratio - 1.0) < R.MATERIAL_RATIO_TOLERANCE
    assert c.n_baseline_moved == 0
    assert not c.changed
    assert "re-measuring is safe" in c.summary()


def test_the_ratio_statistics_exclude_the_rows_they_cannot_describe():
    """A ratio against a zero or non-finite stored depth is not a number he can act
    on. Folding those into the median would make the report lie in the
    safe-looking direction, so they are NaN and ``n_ratio`` says how many rows are
    left."""
    x = bursty()
    ev = detected(x)
    df = ev.df.copy()
    df.loc[0, "depth_pA"] = 0.0
    df.loc[1, "depth_pA"] = np.nan
    c = R.compare_estimators(EventSet(df, dict(ev.provenance)), x, FS,
                             config=DEFAULT_BASELINE, polarity=-1)
    assert not np.isfinite(c.depth_ratio[0]) and not np.isfinite(c.depth_ratio[1])
    assert c.n_ratio == c.n_events - 2
    assert np.isfinite(c.median_ratio)


def test_frac_depth_over_one_is_counted_and_not_blamed_on_the_current_gain():
    """§4.1: the corrected baseline is what makes dI/I0 > 1 visible, and this report
    is where the owner is looking when it becomes visible. Out of scope to FIX,
    in scope to SAY -- but to say TRUTHFULLY.

    The sentence used to read "a calibration question about this channel's current
    gain". ``characterize`` computes ``frac_depth = depth_pA / |baseline_pA|``, so
    scaling the whole channel by any gain factor leaves it EXACTLY unchanged: the
    report was pointing the owner at a knob that provably cannot move the number it
    was complaining about. What dI/I0 > 1 actually says is that the current
    magnitude grows during the event -- a conductive event, which is a category his
    own corpus names in its filenames -- or an amplifier OFFSET, which shifts the
    open-pore level without touching the excursion.

    The gain-invariance is asserted here rather than asserted in prose, because it
    is the whole reason the wording changed."""
    # open pore -35 pA with events reaching -120 pA: the same shape as the owner's
    # own apo file, where the corrected frac_depth comes out above 2.
    x = bursty(i0=-35.0, depth=85.0, noise=1.5)
    ev = detected(x)
    c = R.compare_estimators(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    assert c.n_frac_depth_gt1 > 0
    text = c.summary()
    assert "conductive" in text
    assert "current-gain question" in text and "cannot move it" in text
    assert "calibration question about this channel's current gain" not in text
    assert c.to_dict()["n_frac_depth_gt1"] == c.n_frac_depth_gt1

    # frac_depth is a ratio: a 1000x gain on the whole channel moves it by nothing.
    c10 = R.compare_estimators(
        _scaled(ev, 1000.0), x * 1000.0, FS, config=DEFAULT_BASELINE, polarity=-1
    )
    a = c.new_frac_depth
    b = c10.new_frac_depth
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 100
    # rtol at float32's own precision: the trace is float32, so a 1000x gain is not
    # an exactly representable rescaling of every sample and the baseline estimate
    # differs in its last bits. The claim is invariance of the RATIO, and 1e-5 is
    # five orders of magnitude tighter than anything that could make frac_depth
    # cross 1.
    assert np.allclose(a[ok], b[ok], rtol=1e-5), (
        "frac_depth moved under a pure gain change; the report's wording depends "
        "on it being invariant"
    )


def _scaled(ev, k):
    """The same EventSet with every pA column multiplied by ``k``."""
    from nanoscope.analysis.events import EventSet

    df = ev.df.copy()
    for col in ("baseline_pA", "depth_pA", "mean_dev_pA", "max_dev_pA"):
        df[col] = df[col] * k
    df["ecd_pA_s"] = df["ecd_pA_s"] * k
    prov = dict(ev.provenance or {})
    if "noise_rms_pA" in prov:
        prov["noise_rms_pA"] = float(prov["noise_rms_pA"]) * k
    return EventSet(df, prov)


# ===========================================================================
# 5. DESCRIBING THE OLD WORLD — absence is the signal
# ===========================================================================
def test_a_table_with_no_stamp_is_named_as_the_block_median_it_was():
    """"Unknown" would be true and useless. A table carrying no
    ``baseline_fingerprint`` predates the stamp, and the estimator in force before
    the stamp existed was the 10 ms block median — which is the fact the owner needs
    in order to judge the change."""
    assert "block_median @ 10 ms" in R.describe_stored_baseline({})
    assert "predates" in R.describe_stored_baseline(None)
    named = R.describe_stored_baseline(
        {"baseline_fingerprint": "sha256:deadbeefcafe",
         "baseline_method": "masked_median", "baseline_rough_s": 2.0,
         "baseline_base_s": 0.5, "baseline_decim": None, "baseline_mask_sigma": 3.5,
         "baseline_dilate_ms": 1.0, "baseline_window_s": 0.5,
         "baseline_despike_size": 0})
    assert "masked median" in named and "2000 ms" in named


# ===========================================================================
# 6. EDGES
# ===========================================================================
def test_an_empty_table_is_stamped_not_refused():
    """It WAS measured — of nothing — by the new estimator. Leaving it marked stale
    for ever would be a lie, and refusing would violate "never refuse to open"."""
    x = bursty()
    ev = detected(x)
    empty = EventSet(ev.df.iloc[:0].copy(), dict(ev.provenance))
    out = R.remeasure_events(empty, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    assert len(out.events.df) == 0
    assert out.fingerprint == DEFAULT_BASELINE.fingerprint(FS)
    assert out.comparison.n_events == 0
    assert out.comparison.summary() == "No events to compare."


def test_one_bad_sample_far_from_every_event_does_not_change_the_table():
    """A whole-recording estimate that REFUSES on one non-finite sample anywhere
    turns that sample into a TABLE-WIDE change of measurement. The blast radius has
    to stay at the rows whose own pad reads it."""
    x = bursty()
    ev = detected(x)
    good = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)

    y = x.copy()
    starts = ev.df["start_idx"].to_numpy(dtype=np.int64)
    ends = ev.df["end_idx"].to_numpy(dtype=np.int64)
    # A sample no event's pad can reach. The pad is max(event length, 3 ms) either
    # side, so the midpoint of a full inter-event gap between two REAL events is
    # ~17 ms clear of both. (Row 0 of this fixture is a boundary-clipped noise
    # excursion, hence the gap between rows 5 and 6 rather than 0 and 1.)
    gap = int((starts[6] + ends[5]) // 2)
    pad = int(0.006 * FS)
    assert gap - ends[5] > pad and starts[6] - gap > pad
    y[gap] = np.inf
    hurt = R.remeasure_events(ev, y, FS, config=DEFAULT_BASELINE, polarity=-1)

    a, b = good.events.df, hurt.events.df
    poisoned = ~np.isfinite(b["depth_pA"].to_numpy()) | \
        np.array([has_flag(f, FLAG_NON_FINITE) for f in b["flag"]])
    assert poisoned.sum() <= 2, "one bad sample must not poison the table"
    ok = ~poisoned
    assert np.allclose(a.loc[ok, "baseline_pA"], b.loc[ok, "baseline_pA"], atol=1e-6)


def test_a_bad_sample_inside_a_row_s_pad_still_poisons_that_row():
    """The other half of the same rule: a non-finite sample inside a row's own
    measurement window is INSIDE its measurement and must taint it, by
    ``recharacterize_span``'s flank rule and no second rule of ours."""
    x = bursty()
    ev = detected(x)
    y = x.copy()
    y[int(ev.df.loc[0, "start_idx"]) + 5] = np.nan
    out = R.remeasure_events(ev, y, FS, config=DEFAULT_BASELINE, polarity=-1)
    assert has_flag(out.events.df.loc[0, "flag"], FLAG_NON_FINITE)
    assert not bool(out.events.df.loc[0, "accepted"])


def test_a_rate_the_detection_did_not_run_at_is_refused():
    """Found by the geometry guard on the owner's own file: ``t_start_s`` and
    ``dwell_s`` are derived from the rate, so re-measuring at 50000.0 a table
    detected at 50000.001192 silently rewrites both."""
    x = bursty()
    ev = detected(x)
    with pytest.raises(ValueError, match="detected at .* but this re-measure"):
        R.remeasure_events(ev, x, FS + 1.0, config=DEFAULT_BASELINE, polarity=-1)


def test_it_polls_the_ambient_cancel_token():
    x = bursty()
    ev = detected(x)
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)


def test_the_detector_s_polarity_is_preferred_over_a_fresh_measurement():
    """A re-measure must reproduce the frame the rows were DETECTED in. Flipping the
    polarity would not be a better baseline, it would be a different experiment —
    and `derive_polarity` returns +1 on the owner's own raw drifting traces."""
    x = bursty()
    ev = detected(x)
    prov = dict(ev.provenance)
    assert prov.get("polarity") == -1
    stated = R.remeasure_events(EventSet(ev.df, prov), x, FS,
                                config=DEFAULT_BASELINE)
    explicit = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    assert np.array_equal(stated.baseline, explicit.baseline)
    assert stated.events.provenance["baseline_polarity"] == -1


# ===========================================================================
# 7. THE RELEASE NOTE (§4.2.6)
# ===========================================================================
def test_the_release_note_states_the_columns_and_the_measured_factor():
    note = R.release_note("v9.9")
    assert "v9.9" in note
    for col in R.REMEASURED_COLUMNS:
        assert col in note, col
    for col in ("start_idx", "end_idx", "dwell_s"):
        assert col in note, col
    # The MEASURED factor, not the spec's. 4.65x is what 1641 apo events actually
    # move by (median 23.70 -> 110.17 pA); the spec's 14.7x is a different
    # detection's statistic and does not reproduce on this table.
    assert "4.65x" in note and "10 ms" in note
    assert "14.7x" not in note
    assert "meta.json" in note


def test_the_release_note_does_not_call_rise_and_fall_unaffected():
    """It did, and they moved on 1243 of 1641 rows.

    Two separate reasons, and the note has to survive both: the row-by-row rebuild
    used to lose the neighbour bound (fixed -- see
    ``test_a_same_config_remeasure_moves_no_column_at_all``), and a depth that moves
    4.7x moves the 10%/90% levels the crossing times are read at, which is real and
    permanent. A note that lists them under "Unaffected" is wrong either way."""
    note = R.release_note("v9.9")
    head, _, tail = note.partition("Unaffected:")
    assert tail, "the note no longer has an Unaffected section; update this test"
    unaffected = tail.split("\n\n")[0]
    for col in ("rise_s", "fall_s"):
        assert col not in unaffected, f"{col} is listed as unaffected"
    assert "rise_s" in head or "rise_s" in tail


def test_the_release_note_says_a_remeasure_is_not_a_redetection():
    """The dwell column is the one a re-measure deliberately does NOT fix, and on
    the owner's file it is the column the old estimator got most wrong: median dwell
    0.44 ms stored against 17.56 ms on a fresh detection, because a baseline that
    dives into each blockade shatters it into fragments. A note that only talks
    about depth invites a depth-vs-dwell scatter with a correct x and a 40x-wrong y."""
    note = R.release_note("v9.9")
    assert "17.56" in note and "0.44" in note
    assert "RE-DETECT" in note


def test_the_shipped_release_note_is_generated_from_the_code():
    """The note and the behaviour it describes can only change together. A doc file
    edited by hand is a doc file that goes stale the first time the estimator moves,
    and this one is the artefact the owner attaches to data he has already
    published."""
    from nanoscope import __version__

    doc = Path(__file__).resolve().parents[2] / "docs" / "RELEASE_NOTE_BASELINE.md"
    assert doc.exists(), doc
    assert R.release_note(f"v{__version__}") in doc.read_text()


# ===========================================================================
# 8. THE REAL RECORDING
# ===========================================================================
@pytest.mark.corpus
def test_the_owners_apo_recording_moves_by_the_factor_the_report_states():
    """The migration, measured on the file it exists for.

    Measured on ``_0006_`` (2,993,798 samples, 50 kHz, channel 2), Schmitt at
    ``min_dwell_s=1e-4`` under the 10 ms block median: 1641 events, median
    ``depth_pA`` 23.70 pA against a stamped baseline of -120.9 pA. Re-measured
    against the app default (masked median, 0.5 s / 0.1 s): median depth 110.2 pA,
    a 4.73x per-event median ratio, and 1636 of 1641 events (99.7%) with a stamped
    baseline that moves more than 10 pA.

    The 14.7x of §0.1 is a different detection's population ratio, not this one's;
    what both say, and what this asserts, is that the change is far outside anything
    a reader of the old table could have absorbed as noise.
    """
    from nanoscope.core.types import ChannelKind
    from nanoscope.io.base import open_document

    if not APO.exists():
        pytest.skip(f"{APO} not present")
    doc = open_document(str(APO))
    # THREE channels on this file (ch0 all-zero, ch1 the command voltage, ch2 the
    # current) against two in the ferritin corpus. Never hard-code the index.
    ch = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT][0]
    cur = np.asarray(doc.channels[ch.index], dtype=np.float32)
    fs = float(doc.meta.sample_rate_hz)

    # The FILE's rate, not the module constant: 50000.001192 Hz here, and
    # `_require_same_rate` refuses the mismatch rather than letting the geometry
    # guard report it as a moved ``t_start_s``.
    ev = detected(cur, OLD, fs)
    assert 1500 < len(ev.df) < 1800
    assert 20 < float(np.median(ev.df["depth_pA"])) < 30

    out = R.remeasure_events(ev, cur, fs, config=DEFAULT_BASELINE, polarity=-1)
    c = out.comparison
    assert float(np.median(out.events.df["depth_pA"])) > 90
    assert c.median_ratio > 4.0
    assert c.frac_baseline_moved > 0.95
    assert c.changed
    # ...and the geometry did not move a sample.
    for col in R.GEOMETRY_COLUMNS:
        assert (ev.df[col].to_numpy() == out.events.df[col].to_numpy()).all(), col
    # §4.1's calibration question, made visible by the corrected baseline.
    assert c.n_frac_depth_gt1 > 0.9 * c.n_events


# ===========================================================================
# 9. CURATION MADE WHILE THE JOB RUNS
# ===========================================================================
# The re-measure is dispatched off the GUI thread and the event table stays live
# for the ~3.3 s it takes on the owner's own file. Discarding the clicks made in
# that window is the same failure as discarding them on reload, on a shorter fuse.
def test_a_curation_made_during_the_job_is_carried_onto_the_result():
    x = bursty()
    ev = detected(x)
    before = R.curation_snapshot(ev.df)
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)

    live = ev.df.copy()                      # the user clicked while it ran
    live.loc[3, "accepted"] = False
    live.loc[7, "flag"] = "late_note"

    merged = R.merge_late_curation(out.events.df, live, before)
    assert not bool(merged.loc[3, "accepted"])
    assert has_flag(merged.loc[7, "flag"], "late_note")
    # ...and the MEASUREMENT is still the re-measured one, not the live table's.
    assert np.array_equal(merged["depth_pA"].to_numpy(),
                          out.events.df["depth_pA"].to_numpy())


def test_a_late_curation_does_not_resurrect_a_re_derived_flag():
    """The carry goes through `inherited_curation`, not by copying cells: the
    re-measured row has just RE-DERIVED its automatic tokens against the corrected
    baseline, and pasting the pre-measurement ones back would restore a rejection
    the correction has fixed."""
    x = bursty()
    ev = detected(x)
    before = R.curation_snapshot(ev.df)
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)

    live = ev.df.copy()
    live.loc[3, "accepted"] = False
    live.loc[3, "flag"] = FLAG_NON_FINITE     # an AUTOMATIC token, re-derivable
    merged = R.merge_late_curation(out.events.df, live, before)
    assert bool(merged.loc[3, "accepted"])
    assert FLAG_NON_FINITE not in flag_tokens(merged.loc[3, "flag"])


def test_no_late_curation_returns_the_result_untouched():
    x = bursty()
    ev = detected(x)
    before = R.curation_snapshot(ev.df)
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    assert R.merge_late_curation(out.events.df, ev.df, before) is out.events.df


def test_a_table_whose_shape_changed_under_the_job_is_refused_not_guessed():
    """A merge (or a hand-added event, or a fresh detection) during the job leaves
    rows that are not the rows that were measured. Installing over them would undo
    the edit, so the answer has to be dropped and SAID."""
    x = bursty()
    ev = detected(x)
    before = R.curation_snapshot(ev.df)
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1)
    merged_live = ev.df.drop(index=3).reset_index(drop=True)
    with pytest.raises(ValueError, match="changed while the re-measure was running"):
        R.merge_late_curation(out.events.df, merged_live, before)
