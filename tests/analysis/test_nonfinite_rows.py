"""C41 — a single NaN/Inf must not come back as an accepted, unflagged event row.

The defect, measured on a 3-event trace with ONE ``np.inf`` inside event 0::

    clean                depth=20.140960  ecd= 0.12069  mean_dev= 20.115  accepted=True  flag=''
    one +inf inside ev0  depth=20.135567  ecd=-inf      mean_dev=-inf     accepted=True  flag=''
    one NaN inside ev0   depth=20.140960  ecd=nan       mean_dev=nan      accepted=True  flag=''

``depth_pA`` is a MEDIAN, so it reads straight through the poisoned sample and comes
back a plausible wrong number while ``ecd_pA_s`` and ``mean_dev_pA`` are ``-inf`` — and
the row is ACCEPTED with no flag. Exporting the accepted rows then writes literal
``-inf`` into two columns of the frozen 20-column schema, and it detonates hundreds of
steps later in ``stats._clean_1d`` or ``figures._column`` with an error that names no row.

The batch that preceded this one gave ``detect_events`` and ``psd`` non-finite guards and
left ``characterize`` without one — so the guarded paths are exactly the ones that read a
FILE, and the unguarded ones are exactly the ones that rebuild a row (a merge, an Event
Inspector resize, a hand-added event) from an array that can go non-finite AFTER the load:
a float32 calibration overflow, a detrend, a flank read outside the event.

So the rule is a fourth AUTOMATIC REJECTION RULE in ``characterize``, beside the other
three, in the one function every row in the app is built by. A caller cannot forget it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import derive_polarity
from nanoscope.analysis.characterize import characterize, recharacterize_span
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    DERIVED_FLAGS,
    FLAG_NON_FINITE,
    FLAG_ZERO_BASELINE,
    SCHEMA,
    EventSet,
    has_flag,
    span_provenance,
)
from nanoscope.analysis.measure import measure_selection
from nanoscope.analysis.merge import merge_by_gap

FS = 50_000.0
I0 = -1000.0
DEPTH = 300.0
NOISE = 8.0
SPANS = [(10_000, 10_400), (20_000, 20_400), (30_000, 30_400)]

ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
_needs_abf = pytest.mark.skipif(not ABF.exists(), reason="owner's -600 mV recording not present")


# ===========================================================================
# builders
# ===========================================================================
def _trace(*, n=50_000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, NOISE, n)
    for s, e in SPANS:
        x[s:e] -= DEPTH
    return x


def _rows(current, baseline=None, **kw) -> pd.DataFrame:
    starts = np.array([s for s, _ in SPANS], dtype=np.int64)
    ends = np.array([e for _, e in SPANS], dtype=np.int64)
    base = np.full(np.asarray(current).size, I0) if baseline is None else baseline
    return characterize(current, base, starts, ends, FS, -1, noise_rms=NOISE, **kw)


def _prov(n, **extra):
    return {"polarity": -1, "noise_rms_pA": NOISE, **span_provenance(0, n), **extra}


# ===========================================================================
# THE DEFECT
# ===========================================================================
@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_one_poisoned_sample_no_longer_produces_an_accepted_unflagged_row(value):
    """§2.3 exactly: one non-finite sample inside event 0 of three."""
    x = _trace()
    x[10_050] = value
    row = _rows(x).iloc[0]

    assert has_flag(row["flag"], FLAG_NON_FINITE)
    assert bool(row["accepted"]) is False


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_the_clean_rows_beside_it_are_bit_identical(value):
    """The rule must reject the poisoned row and touch nothing else -- a rule that moves
    a neighbouring number is a worse defect than the one it fixes."""
    clean = _rows(_trace())
    x = _trace()
    x[10_050] = value
    poisoned = _rows(x)

    for col in SCHEMA:
        a, b = clean[col].to_numpy()[1:], poisoned[col].to_numpy()[1:]
        assert np.array_equal(a, b), col


def test_a_wholly_finite_table_acquires_nothing():
    """The control. Over-flagging would reject every good event in the app."""
    rows = _rows(_trace())
    assert not has_flag(rows["flag"], FLAG_NON_FINITE).any()
    assert bool(rows["accepted"].all())
    assert list(rows["flag"]) == ["", "", ""]


def test_the_poisoned_row_is_the_only_one_flagged():
    x = _trace()
    x[10_050] = np.inf
    rows = _rows(x)
    assert list(has_flag(rows["flag"], FLAG_NON_FINITE)) == [True, False, False]


def test_a_non_finite_BASELINE_is_caught_even_though_every_sample_is_finite():
    """The second half of the rule. A baseline read from a poisoned flank puts no bad
    sample inside the event at all, and ``baseline_pA``/``frac_depth``/``snr`` are then
    quietly NaN on an accepted row."""
    base = np.full(50_000, I0)
    base[10_100] = np.nan
    row = _rows(_trace(), baseline=base).iloc[0]

    assert has_flag(row["flag"], FLAG_NON_FINITE)
    assert bool(row["accepted"]) is False


def test_a_nan_baseline_is_not_reported_as_a_ZERO_baseline():
    """``|baseline| > 0`` is False for NaN as well as for 0, so a NaN baseline used to
    come out claiming ``zero_baseline`` -- a reason that is not true, and was the only
    thing the row said about itself."""
    base = np.full(50_000, I0)
    base[10_100] = np.nan
    row = _rows(_trace(), baseline=base).iloc[0]
    assert not has_flag(row["flag"], FLAG_ZERO_BASELINE)


def test_a_genuinely_zero_baseline_still_says_zero_baseline():
    """The control on that narrowing."""
    row = _rows(_trace(), baseline=np.zeros(50_000)).iloc[0]
    assert has_flag(row["flag"], FLAG_ZERO_BASELINE)
    assert not has_flag(row["flag"], FLAG_NON_FINITE)


def test_n_levels_is_not_fitted_through_an_inf():
    """The sub-state fit's prefix sums over an Inf are ``inf - inf``: a NaN split gain and
    a numpy RuntimeWarning per poisoned event, printed from a GUI worker with nothing to
    say which row it came from. A poisoned event reports the unfittable default instead."""
    x = _trace()
    x[10_050] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rows = _rows(x)
    assert int(rows["n_levels"].iloc[0]) == 1


# ===========================================================================
# every rebuild path inherits it, because the rule is in the one function
# ===========================================================================
def test_the_rebuild_path_inherits_the_rule():
    """``recharacterize_span`` is the merge, the Event Inspector resize and the
    hand-added event. None of them was guarded; all three are now, without being
    touched."""
    x = _trace()
    x[10_050] = np.inf
    row = recharacterize_span(x, FS, 10_000, 10_400, provenance=_prov(x.size),
                              prior=None).iloc[0]
    assert has_flag(row["flag"], FLAG_NON_FINITE)
    assert bool(row["accepted"]) is False


def test_a_merge_over_a_poisoned_span_is_rejected_not_averaged_out():
    """A merge is where it hurts most: the merged span is longer than either fragment, so
    a single Inf anywhere in it decides ``ecd_pA_s`` for the whole thing."""
    x = _trace()
    x[10_050] = np.inf
    rows = pd.concat([
        recharacterize_span(x, FS, 10_000, 10_150, provenance=_prov(x.size), prior=None),
        recharacterize_span(x, FS, 10_200, 10_400, provenance=_prov(x.size), prior=None),
    ], ignore_index=True)
    rows["event_id"] = np.arange(len(rows), dtype=np.int64)

    merged, n = merge_by_gap(EventSet(rows, _prov(x.size)), x, FS, max_gap_s=2e-3)
    assert n == 1
    assert has_flag(merged.df["flag"].iloc[0], FLAG_NON_FINITE)
    assert bool(merged.df["accepted"].iloc[0]) is False
    assert len(merged.accepted.df) == 0


def test_the_flag_is_re_derived_so_a_clean_re_measurement_clears_it():
    """It is a statement about THIS measurement, so it belongs in ``DERIVED_FLAGS``: a row
    re-measured on a repaired trace must stop claiming it, exactly as a resized event stops
    claiming it is clipped. Inheriting it would be C21 in a fourth costume."""
    assert FLAG_NON_FINITE in DERIVED_FLAGS
    prior = pd.Series({"flag": FLAG_NON_FINITE, "accepted": False})
    row = recharacterize_span(_trace(), FS, 10_000, 10_400,
                              provenance=_prov(50_000), prior=prior).iloc[0]
    assert str(row["flag"]) == ""
    assert bool(row["accepted"]) is True


def test_a_hand_rejection_still_outranks_the_new_rule():
    """``accepted`` is AND, never OR -- the new rule may not become a way to un-reject."""
    prior = pd.Series({"flag": "", "accepted": False})       # a bare reject == by hand
    row = recharacterize_span(_trace(), FS, 10_000, 10_400,
                              provenance=_prov(50_000), prior=prior).iloc[0]
    assert bool(row["accepted"]) is False


def test_a_curators_own_token_survives_beside_the_new_one():
    x = _trace()
    x[10_050] = np.inf
    prior = pd.Series({"flag": "operator_note", "accepted": True})
    row = recharacterize_span(x, FS, 10_000, 10_400, provenance=_prov(x.size),
                              prior=prior).iloc[0]
    assert has_flag(row["flag"], "operator_note")
    assert has_flag(row["flag"], FLAG_NON_FINITE)
    assert bool(row["accepted"]) is False


# ===========================================================================
# what the fix actually buys: the -inf stops reaching the export
# ===========================================================================
def test_the_accepted_subset_is_finite_which_is_what_every_export_writes():
    """``accepted`` is what exports, both histograms, the scatter and every fit honour.
    The poisoned row is still IN the table -- the user must see it -- and out of the
    numbers."""
    x = _trace()
    x[10_050] = np.inf
    es = EventSet(_rows(x), _prov(x.size))

    numeric = [c for c, (d, _, _) in SCHEMA.items() if d == "float64"]
    kept = es.accepted.df
    assert len(kept) == 2
    for col in numeric:
        v = kept[col].to_numpy(dtype=np.float64)
        assert np.isfinite(v[~np.isnan(v)]).all(), col      # rise/fall may be NaN by design
    assert not np.isinf(np.nan_to_num(kept[numeric].to_numpy(np.float64), nan=0.0)).any()
    assert len(es.df) == 3


def test_the_row_is_kept_not_dropped():
    """A dropped row is a row the user cannot go and look at. The whole point of the flag
    vocabulary is that a rejection carries its reason instead of vanishing."""
    x = _trace()
    x[10_050] = np.nan
    rows = _rows(x)
    assert len(rows) == 3
    assert int(rows["start_idx"].iloc[0]) == 10_000


# ===========================================================================
# the two other doors the report named
# ===========================================================================
def test_derive_polarity_refuses_a_non_finite_trace_instead_of_guessing_plus_one():
    """One NaN makes the median NaN, hence the MAD NaN, hence ``sigma > 0`` False -- so it
    took the FLAT-TRACE exit and returned +1 for a trace whose direction it never measured.
    +1 is the wrong sign for every blockade in this app at negative bias."""
    x = _trace()
    assert derive_polarity(x) == -1                      # measured, before anything is broken
    x[10_050] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        derive_polarity(x)


def test_the_rebuild_paths_fall_back_rather_than_crash_on_that_refusal():
    """``merge._polarity`` and ``characterize._resolve_polarity`` catch it (they have a
    defensible fallback); the row is flagged by the rule either way."""
    from nanoscope.analysis.characterize import _resolve_polarity
    from nanoscope.analysis.merge import _polarity

    x = _trace()
    x[10_050] = np.inf
    assert _polarity(None, x) == -1
    assert _resolve_polarity(None, None, x) == -1

    row = recharacterize_span(x, FS, 10_000, 10_400,
                              provenance={"noise_rms_pA": NOISE}, prior=None).iloc[0]
    assert has_flag(row["flag"], FLAG_NON_FINITE)


def test_measure_selection_refuses_a_poisoned_selection():
    """The hand-measured half. An order statistic reads straight through an Inf, so the
    live readout printed a confident ``amplitude`` beside a ``mean`` of ``inf``."""
    x = _trace()
    x[10_050] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        measure_selection(x, FS, 10_000, 10_400)

    # a selection elsewhere in the same trace is measurable, and unchanged
    ok = measure_selection(x, FS, 30_000, 30_400)
    assert np.isfinite(ok["amplitude"])
    assert ok == measure_selection(_trace(), FS, 30_000, 30_400)


def test_a_poisoned_FLANK_is_refused_and_the_refusal_names_the_flank():
    """The open-pore reference is read from windows OUTSIDE the selection, and those
    have their own guard now. They used to refuse only BY ACCIDENT and only for an Inf:
    ``filters.noise_rms`` raised "x contains non-finite samples" -- naming its own local
    variable, from a module the user has never heard of -- while a flank NaN reached the
    level estimator and came out as "cannot convert float NaN to integer"."""
    x = _trace()
    x[10_450] = np.inf                       # in the right-hand flank, not the selection
    with pytest.raises(ValueError, match="open-pore window beside selection"):
        measure_selection(x, FS, 10_000, 10_400)


def test_measure_selection_names_how_many_and_where():
    x = _trace()
    x[10_050] = np.inf
    x[10_060] = np.nan
    with pytest.raises(ValueError, match=r"2 of 400 samples are non-finite.*index 10,050"):
        measure_selection(x, FS, 10_000, 10_400)


def test_a_bounded_preview_reports_what_it_read():
    """A live preview reads a STRIDED subsample, so it can legitimately stride past the bad
    sample -- and this module's whole contract is that a preview reports what it read.
    Anything that PERSISTS a measurement re-measures with ``max_samples=None``, which is
    the call that refuses."""
    x = _trace(n=400_000)
    x[200_001] = np.inf                      # a sample stride 200 steps straight over
    preview = measure_selection(x, FS, 100_000, 300_000, max_samples=1_000)
    assert preview["approximate"] is True and int(preview["stride"]) == 200
    assert np.isfinite(preview["mean"])

    with pytest.raises(ValueError, match="non-finite"):
        measure_selection(x, FS, 100_000, 300_000)


# ===========================================================================
# the detectors: guarded upstream, and guarded here too
# ===========================================================================
def test_detect_events_still_refuses_the_whole_recording_up_front():
    """The load-boundary guard the previous batch added is not weakened by this one: a
    detector handed a poisoned trace refuses outright rather than producing a table with
    one rejected row in it."""
    x = _trace()
    x[10_050] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        detect_events(x, FS, baseline=I0, min_dwell_s=1e-4)


@_needs_abf
def test_one_poisoned_sample_in_a_real_translocation():
    """A real event out of the owner's -600 mV recording, with one sample of it set to
    Inf: the row is rejected and named, and the depth it would otherwise have reported is
    still a perfectly plausible number."""
    from nanoscope.io.base import open_document

    doc = open_document(ABF)
    ci = next(i for i, c in enumerate(doc.meta.channels) if c.kind.name == "CURRENT")
    cur = np.ascontiguousarray(doc.channels[ci][1_000_000:1_500_000], dtype=np.float64)
    fs = float(doc.meta.sample_rate_hz)

    df, prov = detect_events(cur, fs, min_dwell_s=2e-5)
    ev = df[df["dwell_s"] > 1e-3].iloc[0]
    s, e = int(ev["start_idx"]), int(ev["end_idx"])

    poisoned = cur.copy()
    poisoned[(s + e) // 2] = np.inf
    row = recharacterize_span(poisoned, fs, s, e, provenance=prov,
                              prior=df.loc[ev.name]).iloc[0]

    assert has_flag(row["flag"], FLAG_NON_FINITE)
    assert bool(row["accepted"]) is False
    assert np.isfinite(float(row["depth_pA"]))          # the plausible wrong number
    assert not np.isfinite(float(row["ecd_pA_s"]))      # beside the one that is not

    clean = recharacterize_span(cur, fs, s, e, provenance=prov,
                                prior=df.loc[ev.name]).iloc[0]
    assert str(clean["flag"]) == ""
    assert bool(clean["accepted"]) is True


# ===========================================================================
# THE RESIDUAL: +/-Inf survived IN the rejected row's cells
# ===========================================================================
# The rule above catches the ROW -- it is flagged, it is not accepted, and every
# shipped export honours ``accepted``, so nothing left the machine carrying an
# infinity. What it did not do is take the ``-inf`` OUT of ``ecd_pA_s`` and
# ``mean_dev_pA``, and +/-Inf is the only non-value that survives ``dropna()``:
# ``df['ecd_pA_s'].dropna().mean()`` is ``-inf``, and so is ``describe()``, a
# histogram, and every downstream notebook that treats NaN as missing. Untick
# "Use accepted events only" in the Plots panel and the pair reaches ``stats``.
#
# Two fixes, each at ITS OWN invariant rather than at one call site:
#   * the SOURCE -- the `EventSet` door demotes every infinity to NaN, this app's
#     own "not measured" sentinel, which the same row already carries in ``snr``;
#   * the CONSUMER -- ``stats._clean_1d``, the one validator every fit in the
#     module passes through, names the offending rows instead of a bare count.


def test_no_schema_column_ever_holds_an_infinity():
    x = _trace()
    x[10_050] = np.inf
    rows = _rows(x)
    # characterize itself still produces them -- that is arithmetic, not a policy...
    assert not np.isfinite(float(rows["ecd_pA_s"].iloc[0]))
    assert np.isinf(float(rows["ecd_pA_s"].iloc[0]))

    es = EventSet(rows, _prov(x.size))
    numeric = [c for c, (d, _, _) in SCHEMA.items() if d == "float64"]
    v = es.df[numeric].to_numpy(dtype=np.float64)
    assert not np.isinf(v).any(), "an EventSet column still holds +/-Inf"
    # ...and the row is still there, still rejected, still saying why.
    assert len(es.df) == 3
    assert has_flag(es.df["flag"].iloc[0], FLAG_NON_FINITE)
    assert bool(es.df["accepted"].iloc[0]) is False
    assert np.isnan(float(es.df["ecd_pA_s"].iloc[0]))
    assert np.isnan(float(es.df["mean_dev_pA"].iloc[0]))
    assert es.provenance["n_infinite_cells_demoted"] == 2


def test_an_infinity_no_longer_survives_dropna():
    """The whole reason a value is changed rather than merely flagged."""
    x = _trace()
    x[10_050] = np.inf
    raw = _rows(x)["ecd_pA_s"]
    assert not np.isfinite(raw.dropna().mean())          # the defect, in one line

    es = EventSet(_rows(x), _prov(x.size))
    assert np.isfinite(es.df["ecd_pA_s"].dropna().mean())
    assert len(es.df["ecd_pA_s"].dropna()) == 2


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_a_finite_number_on_a_poisoned_row_is_left_exactly_as_measured(value):
    """Only the non-finite CELLS move. ``depth_pA`` is a median and reads straight
    through the bad sample -- it is still what the instrument recorded, and the
    ``non_finite`` flag is what says not to trust it."""
    x = _trace()
    x[10_050] = value
    rows = _rows(x)
    es = EventSet(rows.copy(), _prov(x.size))
    for col, (dtype, _, _) in SCHEMA.items():
        if dtype != "float64":
            continue
        a = rows[col].to_numpy(np.float64)
        b = es.df[col].to_numpy(np.float64)
        keep = np.isfinite(a)
        assert np.array_equal(a[keep], b[keep]), col


def test_a_clean_table_is_not_copied_at_all():
    """Every well-formed recording takes this path; it must cost one ``isinf`` per
    float column and nothing else, and must not hand back a different object."""
    df = _rows(_trace())
    es = EventSet(df, _prov(50_000))
    assert es.df is df
    assert "n_infinite_cells_demoted" not in es.provenance


def test_every_producer_inherits_it_because_the_rule_is_at_the_door():
    """A merge, an Event Inspector resize and a hand-added event all end up in an
    `EventSet`; so does a project reopened from disk. None of them is touched."""
    x = _trace()
    x[10_050] = np.inf
    rows = pd.concat([
        recharacterize_span(x, FS, 10_000, 10_150, provenance=_prov(x.size), prior=None),
        recharacterize_span(x, FS, 10_200, 10_400, provenance=_prov(x.size), prior=None),
    ], ignore_index=True)
    rows["event_id"] = np.arange(len(rows), dtype=np.int64)
    merged, _ = merge_by_gap(EventSet(rows, _prov(x.size)), x, FS, max_gap_s=2e-3)

    numeric = [c for c, (d, _, _) in SCHEMA.items() if d == "float64"]
    assert not np.isinf(merged.df[numeric].to_numpy(np.float64)).any()
    assert has_flag(merged.df["flag"].iloc[0], FLAG_NON_FINITE)


def test_a_refusal_to_fit_names_the_rows_it_is_refusing_over():
    """``stats._clean_1d`` is the one validator every fit in the module passes through.
    It used to say only "contains 1 non-finite value(s). Fix or drop them upstream",
    which is nothing a user can act on: the Plots panel handed over a whole column and
    nothing on screen says which row is bad."""
    from nanoscope.analysis.stats import _clean_1d

    x = _trace()
    x[10_050] = np.inf
    es = EventSet(_rows(x), _prov(x.size))
    col = es.df.set_index("event_id")["frac_depth"].copy()
    col.iloc[0] = np.nan                       # the poisoned row, as the panel sees it

    with pytest.raises(ValueError) as exc:
        _clean_1d(col, "frac_depth", positive=False)
    msg = str(exc.value)
    assert "at rows 0" in msg, msg
    assert FLAG_NON_FINITE in msg and "accepted events only" in msg

    # a bare array can only be quoted by position, and the message says so
    with pytest.raises(ValueError, match="at positions 0"):
        _clean_1d(col.to_numpy(), "frac_depth", positive=False)


def test_the_refusal_quotes_the_users_own_row_labels_and_stops_at_a_readable_number():
    from nanoscope.analysis.stats import _clean_1d

    v = np.arange(100.0) + 1.0
    v[[3, 17, 42]] = np.nan
    s = pd.Series(v, index=[f"ev{i}" for i in range(100)])
    with pytest.raises(ValueError) as exc:
        _clean_1d(s, "dwell_s", positive=True)
    assert "at rows ev3, ev17, ev42" in str(exc.value)

    v2 = np.arange(100.0) + 1.0
    v2[:20] = np.nan
    with pytest.raises(ValueError) as exc:
        _clean_1d(pd.Series(v2), "dwell_s", positive=True)
    assert "and 12 more" in str(exc.value)


def test_the_demotion_is_positional_so_a_repeated_index_is_not_over_blanked():
    """An event table's index is not guaranteed unique -- a subset, or a concat without
    ``ignore_index``. Label-based assignment would blank every row sharing the label."""
    x = _trace()
    x[10_050] = np.inf
    rows = _rows(x)
    rows.index = [7, 7, 7]                     # the pathological case, explicitly
    es = EventSet(rows, _prov(x.size))
    assert np.isnan(float(es.df["ecd_pA_s"].iloc[0]))
    assert np.isfinite(float(es.df["ecd_pA_s"].iloc[1]))
    assert np.isfinite(float(es.df["ecd_pA_s"].iloc[2]))
    assert es.provenance["n_infinite_cells_demoted"] == 2
