"""The ANALYSED SPAN invariant: a row's boundary judgement is stable under rebuilds.

``clipped_by_boundary`` is not a property of a row on its own -- it is a relation
between the row and the span the detector looked at. ``recharacterize_span`` used to
take an ``n_total`` argument for that span and **no caller anywhere passed it**, so
every rebuild re-judged the row against the WHOLE RECORDING. A row the ML detector had
correctly flagged as cut by the user's zoom rectangle found itself nowhere near an edge
and -- ``clipped_by_boundary`` being a DERIVED flag -- dropped it and un-rejected
itself. Measured, a 600-sample event with the ML range starting mid-event, the row
carrying ``flag='clipped_by_boundary', accepted=False``::

    as detected    [200300,200600): accepted=True  flag=''  dwell=6.000 ms
    1-sample nudge [200300,200599): accepted=True  flag=''  dwell=5.980 ms

Note the first line: **no edit was required**. A plain ``merge_by_gap`` lost it, and a
dwell that is purely an artifact of where the user drew the rectangle entered the event
table, both histograms, the censored dwell fit and every export as a fully-measured,
accepted event -- while provenance still claimed ``n_clipped_boundary=1``.

The fix is at the invariant, not at the call sites: the span travels WITH THE ROWS in
the EventSet's own provenance (``events.span_provenance`` writes it, both detectors
stamp it, ``events.analysed_span`` reads it), and the one function every rebuild path
goes through reads it from the provenance it is already handed. There is no argument
for a fourth rebuild path to forget.

The GUI half -- the ML detector, a segment, and a project round-trip -- is in
``tests/gui/test_flag_invariant_gui.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.characterize import recharacterize_span
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    ANALYSED_START_KEY,
    ANALYSED_STOP_KEY,
    CLIPPED_COUNT_KEY,
    FLAG_CLIPPED,
    EventSet,
    analysed_span,
    empty_df,
    has_flag,
    n_clipped_boundary,
    span_provenance,
)
from nanoscope.analysis.merge import merge_by_gap

FS = 50_000.0
I0 = -1000.0
DEPTH = 300.0
NOISE = 8.0

ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
_needs_abf = pytest.mark.skipif(not ABF.exists(), reason="owner's -600 mV recording not present")


# ===========================================================================
# builders
# ===========================================================================
def _trace(spans, *, n=500_000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, NOISE, n)
    for s, e in spans:
        x[s:e] -= DEPTH
    return x.astype(np.float32)


def _prov(lo, hi, **extra):
    """A detection provenance for the sub-range ``[lo, hi)``."""
    return {"polarity": -1, "noise_rms_pA": NOISE, **span_provenance(lo, hi), **extra}


def _rebuild(signal, start, end, prov, prior=None):
    return recharacterize_span(signal, FS, start, end, provenance=prov, prior=prior).iloc[0]


# ===========================================================================
# the carrier
# ===========================================================================
def test_span_provenance_and_analysed_span_are_one_writer_and_one_reader():
    prov = span_provenance(200_300, 260_000)
    assert prov == {ANALYSED_START_KEY: 200_300, ANALYSED_STOP_KEY: 260_000}
    assert analysed_span(prov, 999) == (200_300, 260_000)


def test_a_span_that_analysed_nothing_is_refused_at_the_writer():
    """A reversed or empty span would flag every row in the table; it is not a span."""
    with pytest.raises(ValueError, match="empty or reversed"):
        span_provenance(500, 500)
    with pytest.raises(ValueError, match="empty or reversed"):
        span_provenance(500, 100)


def test_a_legacy_eventset_with_no_span_falls_back_to_the_whole_signal():
    """The documented fallback: an EventSet saved before the span existed carries none.
    ``(0, n)`` is right for every whole-recording detection AND is exactly what the code
    did before, so no legacy row changes its judgement."""
    assert analysed_span(None, 1_000) == (0, 1_000)
    assert analysed_span({}, 1_000) == (0, 1_000)
    assert analysed_span({"polarity": -1}, 1_000) == (0, 1_000)


def test_a_legacy_ml_result_is_read_back_from_range_start_stop():
    """The one legacy case ``(0, n)`` gets WRONG is an ML sub-range result -- which is
    covered, because the ML detector stamped ``range_start``/``range_stop`` long before
    anything read them."""
    assert analysed_span({"range_start": 10, "range_stop": 90}, 1_000) == (10, 90)
    # the canonical pair wins when both are present
    both = {"range_start": 10, "range_stop": 90, **span_provenance(20, 80)}
    assert analysed_span(both, 1_000) == (20, 80)


@pytest.mark.parametrize("stored", [
    {ANALYSED_START_KEY: 90, ANALYSED_STOP_KEY: 10},        # reversed
    {ANALYSED_START_KEY: 10, ANALYSED_STOP_KEY: 10},        # empty
    {ANALYSED_START_KEY: "x", ANALYSED_STOP_KEY: 90},       # not a number
    {ANALYSED_START_KEY: 10},                               # half a pair
    {ANALYSED_START_KEY: 1.5, ANALYSED_STOP_KEY: 100.5},    # not integral
    {ANALYSED_START_KEY: 10, ANALYSED_STOP_KEY: 90.25},     # one of the two
])
def test_an_unusable_stored_span_is_ignored_rather_than_trusted(stored):
    """Trusting a degenerate stored pair would flag every row in the table.

    The fractional pairs are the case the docstring claimed and the code did not do:
    ``int()`` TRUNCATES, so ``(1.5, 100.5)`` came back ``(1, 100)`` and was used verbatim
    as the clip bounds. A sample index is an integer; a stored pair that is not one is
    not a span this reader will guess at."""
    assert analysed_span(stored, 1_000) == (0, 1_000)


def test_an_integral_span_stored_as_text_or_as_a_float_still_reads():
    """The complement -- a JSON/parquet round trip may hand back ``"10"`` or ``10.0``,
    and both name sample 10 exactly."""
    assert analysed_span({ANALYSED_START_KEY: "10", ANALYSED_STOP_KEY: "90"}, 1_000) == (10, 90)
    assert analysed_span({ANALYSED_START_KEY: 10.0, ANALYSED_STOP_KEY: 90.0}, 1_000) == (10, 90)


def test_recharacterize_span_has_no_span_override_to_get_wrong():
    """The design claim, enforced. A ``span=`` override existed briefly, no caller ever
    passed it, and it took precedence over the provenance span with NO validation --
    ``span=(0, 0)`` was used verbatim, bypassing ``analysed_span``'s own refusal of an
    empty pair, and a fourth rebuild path passing it wrongly got the pre-fix behaviour
    back. The argument a caller cannot forget is the one that does not exist."""
    import inspect

    assert "span" not in inspect.signature(recharacterize_span).parameters
    with pytest.raises(TypeError, match="span"):
        recharacterize_span(_trace([(1_000, 1_400)], n=20_000), FS, 1_000, 1_400,
                            provenance=_prov(0, 20_000), prior=None, span=(0, 0))


def test_reading_one_eventsets_provenance_cannot_rewrite_anothers():
    """``EventSet.provenance`` is a getter that WRITES the reconciled count back into
    the stored dict. Keeping the caller's dict by reference therefore made a bare READ
    of one EventSet silently rewrite ``n_clipped_boundary`` in a dict somebody else still
    owned -- and that dict is what a project save writes to disk."""
    signal = _trace([(200_000, 200_600), (210_000, 210_400)], n=300_000)
    df, prov = detect_events(signal, FS, min_dwell_s=2e-5)
    prov[CLIPPED_COUNT_KEY] = 0

    flagged = df.copy()
    flagged.loc[flagged.index[0], "flag"] = FLAG_CLIPPED
    subset = EventSet(flagged, prov)

    assert subset.provenance is not prov, "the EventSet kept the caller's dict"
    _ = subset.provenance                                   # a bare read
    assert prov[CLIPPED_COUNT_KEY] == 0, "reading one EventSet rewrote the detector's own"
    assert subset.provenance[CLIPPED_COUNT_KEY] == 1        # ... while still reconciling


# ===========================================================================
# THE DEFECT
# ===========================================================================
def test_a_zoom_clipped_row_keeps_its_flag_through_a_rebuild():
    """§2.2 exactly. The ML range starts mid-event; the row is clipped by the RANGE, not
    by the recording, and the recording edge is 300 k samples away."""
    signal = _trace([(200_000, 200_600)])
    prov = _prov(200_300, 260_000)
    prior = pd.Series({"flag": FLAG_CLIPPED, "accepted": False})

    row = _rebuild(signal, 200_300, 200_600, prov, prior)
    assert has_flag(row["flag"], FLAG_CLIPPED), "the rebuild judged it against the recording"
    assert bool(row["accepted"]) is False

    # the 1-sample nudge from the report -- an Event Inspector mask resize
    row = _rebuild(signal, 200_300, 200_599, prov, prior)
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False


def test_the_boundary_judgement_is_stable_under_any_number_of_rebuilds():
    """The property the fix guarantees. One rebuild losing the flag is the defect; the
    row must also not acquire or lose it on the fifth."""
    signal = _trace([(200_000, 200_600)])
    prov = _prov(200_300, 260_000)

    row = _rebuild(signal, 200_300, 200_600, prov, None)
    for _ in range(5):
        row = _rebuild(signal, int(row["start_idx"]), int(row["end_idx"]), prov, row)
        assert has_flag(row["flag"], FLAG_CLIPPED)
        assert bool(row["accepted"]) is False

    interior = _rebuild(signal, 210_000, 210_400, prov, None)
    for _ in range(5):
        interior = _rebuild(signal, int(interior["start_idx"]), int(interior["end_idx"]),
                            prov, interior)
        assert str(interior["flag"]) == ""
        assert bool(interior["accepted"]) is True


def test_a_row_that_is_not_clipped_does_not_acquire_the_flag():
    """The control. Over-flagging is its own defect: it would reject good events."""
    signal = _trace([(210_000, 210_400)])
    row = _rebuild(signal, 210_000, 210_400, _prov(200_300, 260_000), None)
    assert str(row["flag"]) == ""
    assert bool(row["accepted"]) is True


# ===========================================================================
# THE DEFECT THE FIX ITSELF INTRODUCED
# ===========================================================================
@pytest.mark.parametrize("start,end,where", [
    (100_000, 100_400, "wholly BEFORE the analysed span"),
    (300_000, 300_400, "wholly AFTER the analysed span"),
    (200_300 - 400, 200_300, "ending exactly at the span's first sample"),
    (260_000, 260_400, "starting exactly at the span's last sample + 1"),
])
def test_an_event_outside_the_analysed_span_is_not_called_clipped(start, end, where):
    """Giving the rebuild a span the row can be OUTSIDE of is what the span fix bought,
    and the clipped rule -- ``(starts <= lo) | (ends >= hi)`` -- had no case for it.

    The user's gesture: run the ML detector on a zoom rectangle, then drag the measure
    cursors over a real translocation somewhere ELSE in the recording and press "Mark
    selection as missed event". The event is fully inside the file, both flanks clean,
    nothing cut it -- and it came back ``accepted=False, flag='clipped_by_boundary'``,
    telling the user it was "still above threshold at the first or last analysed sample"
    and to widen the range. Their explicit "this IS an event" was overridden by a reason
    that was not true of it, and ``n_clipped_boundary`` counted it."""
    signal = _trace([(start, end)])
    row = _rebuild(signal, start, end, _prov(200_300, 260_000), None)

    assert not has_flag(row["flag"], FLAG_CLIPPED), where
    assert str(row["flag"]) == ""
    assert bool(row["accepted"]) is True
    assert float(row["dwell_s"]) == pytest.approx((end - start) / FS)


def test_a_row_overlapping_one_edge_of_the_span_is_still_clipped():
    """The complement, and the reason the rule is 'overlaps AND touches' rather than
    'touches': the two rows below differ only in whether the span could have cut them."""
    signal = _trace([(200_100, 200_500), (259_800, 260_200)])
    prov = _prov(200_300, 260_000)

    low = _rebuild(signal, 200_100, 200_500, prov, None)     # straddles the low edge
    assert has_flag(low["flag"], FLAG_CLIPPED) and bool(low["accepted"]) is False

    high = _rebuild(signal, 259_800, 260_200, prov, None)    # straddles the high edge
    assert has_flag(high["flag"], FLAG_CLIPPED) and bool(high["accepted"]) is False


def test_a_whole_recording_detection_still_flags_its_first_and_last_event():
    """The regression this narrowing must NOT cause. With the default span ``(0, n)``
    nothing is outside it, so an event touching sample 0 or sample n is clipped exactly
    as before -- the ordinary Schmitt case, which must not move."""
    from nanoscope.analysis.characterize import characterize

    n = 20_000
    rng = np.random.default_rng(3)
    sig = np.full(n, I0) + rng.normal(0.0, NOISE, n)
    sig[0:200] -= DEPTH
    sig[9_000:9_400] -= DEPTH
    sig[n - 200:n] -= DEPTH
    df = characterize(sig, np.full(n, I0),
                      np.array([0, 9_000, n - 200], dtype=np.int64),
                      np.array([200, 9_400, n], dtype=np.int64),
                      FS, -1, noise_rms=NOISE)
    assert [has_flag(f, FLAG_CLIPPED) for f in df["flag"]] == [True, False, True]
    assert df["accepted"].tolist() == [False, True, False]


@pytest.mark.parametrize("start,end,where", [
    (200_300, 200_600, "start"),
    (259_800, 260_000, "end"),
    (200_300, 260_000, "both"),
])
def test_clipped_at_the_start_at_the_end_and_at_both_all_survive(start, end, where):
    signal = _trace([(200_000, 200_600), (259_800, 260_400)])
    row = _rebuild(signal, start, end, _prov(200_300, 260_000), None)
    assert has_flag(row["flag"], FLAG_CLIPPED), where
    assert bool(row["accepted"]) is False


def test_a_row_exactly_one_sample_inside_each_edge_is_not_clipped():
    """The boundary of the boundary rule: ``start <= lo`` / ``end >= hi``."""
    signal = _trace([(200_000, 200_600), (259_800, 260_400)])
    prov = _prov(200_300, 260_000)
    assert str(_rebuild(signal, 200_301, 200_600, prov)["flag"]) == ""
    assert str(_rebuild(signal, 259_800, 259_999, prov)["flag"]) == ""


# ===========================================================================
# the merge
# ===========================================================================
def _ml_shaped_eventset(signal, spans, lo, hi):
    """Rows built the way an ML sub-range detection builds them, as an EventSet whose
    provenance carries the range."""
    prov = _prov(lo, hi)
    rows = [recharacterize_span(signal, FS, s, e, provenance=prov, prior=None)
            for s, e in spans]
    df = pd.concat(rows, ignore_index=True)
    df["event_id"] = np.arange(len(df), dtype=np.int64)
    return EventSet(df, {**prov, CLIPPED_COUNT_KEY: n_clipped_boundary(df)}), prov


def test_a_merge_over_ml_sub_range_rows_keeps_the_clip_no_edit_required():
    """"No edit is required" -- the headline of §2.2. A plain gap merge lost the flag."""
    signal = _trace([(200_000, 200_600)])
    es, _ = _ml_shaped_eventset(signal, [(200_300, 200_450), (200_470, 200_600)],
                                200_300, 260_000)
    assert int(es.provenance[CLIPPED_COUNT_KEY]) == 1

    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert n == 1 and len(merged.df) == 1
    row = merged.df.iloc[0]
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False
    assert len(merged.accepted.df) == 0


def test_merging_one_clipped_row_with_one_that_is_not_gives_a_clipped_row():
    """The merged span CONTAINS the clipped fragment, so it runs off the same edge: the
    re-derivation and the union agree, and the re-derivation is the one that is right in
    general (a merge of two interior fragments must NOT be flagged)."""
    signal = _trace([(200_000, 200_600)])
    es, _ = _ml_shaped_eventset(signal, [(200_300, 200_450), (200_470, 200_600)],
                                200_300, 260_000)
    flags = [str(f) for f in es.df["flag"]]
    assert flags[0] == FLAG_CLIPPED and flags[1] == ""      # only ONE is clipped

    merged, _ = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert has_flag(merged.df["flag"].iloc[0], FLAG_CLIPPED)


def test_merging_two_interior_rows_inside_a_sub_range_stays_unflagged():
    """The complement, and why the flag is RE-DERIVED rather than unioned."""
    signal = _trace([(210_000, 210_200), (210_220, 210_400)])
    es, _ = _ml_shaped_eventset(signal, [(210_000, 210_200), (210_220, 210_400)],
                                200_300, 260_000)
    merged, n = merge_by_gap(es, signal, FS, max_gap_s=1e-3)
    assert n == 1
    assert str(merged.df["flag"].iloc[0]) == ""
    assert bool(merged.df["accepted"].iloc[0]) is True


# ===========================================================================
# provenance may not disagree with the rows
# ===========================================================================
def test_a_disagreeing_clipped_count_is_not_constructible():
    """The hunt's tell: ``n_clipped_boundary=1`` with zero rows carrying the flag. The
    count is a statement ABOUT THE ROWS, so ``EventSet`` re-reads it from them -- at the
    one constructor every detection, merge, hand-add and project restore passes through,
    rather than in each producer one at a time."""
    signal = _trace([(200_000, 200_600)])
    es, prov = _ml_shaped_eventset(signal, [(210_000, 210_400)], 200_300, 260_000)
    assert int(es.provenance[CLIPPED_COUNT_KEY]) == 0

    lying = EventSet(es.df, {**prov, CLIPPED_COUNT_KEY: 7})
    assert int(lying.provenance[CLIPPED_COUNT_KEY]) == 0

    # ...and the other direction: a clipped row that provenance forgot to count
    clipped, prov2 = _ml_shaped_eventset(signal, [(200_300, 200_600)], 200_300, 260_000)
    understated = EventSet(clipped.df, {**prov2, CLIPPED_COUNT_KEY: 0})
    assert int(understated.provenance[CLIPPED_COUNT_KEY]) == 1


def test_a_hand_added_event_that_runs_off_the_edge_updates_the_count():
    """The reachable construction: the analysis panel appends a hand-added row to the
    detection's df and rewraps it with a COPY of the old provenance."""
    signal = _trace([(200_000, 200_600)])
    es, prov = _ml_shaped_eventset(signal, [(210_000, 210_400)], 200_300, 260_000)
    added = recharacterize_span(signal, FS, 200_300, 200_600, provenance=prov, prior=None)
    combined = pd.concat([es.df, added], ignore_index=True)
    combined["event_id"] = np.arange(len(combined), dtype=np.int64)

    rewrapped = EventSet(combined, dict(es.provenance))
    assert int(rewrapped.provenance[CLIPPED_COUNT_KEY]) == 1


def test_the_count_is_re_read_after_the_frame_is_edited_in_place():
    """Reconciling at CONSTRUCTION is the shape of fix this project has twice watched come
    back: the Event Inspector edits the very frame the EventSet holds, IN PLACE, and never
    rebuilds the object. A count fixed only in ``__post_init__`` therefore goes stale on
    the first mask resize -- and that stale number is what a project save writes to disk.
    Deriving it where it is READ covers every mutation as well as every construction."""
    signal = _trace([(200_000, 200_600)])
    es, _ = _ml_shaped_eventset(signal, [(200_300, 200_600)], 200_300, 260_000)
    assert int(es.provenance[CLIPPED_COUNT_KEY]) == 1

    # the resize that legitimately un-clips the row, written straight into the frame
    fixed = recharacterize_span(signal, FS, 200_400, 200_600,
                                provenance=_prov(200_300, 260_000), prior=None).iloc[0]
    for col in ("start_idx", "flag", "accepted"):
        es.df.at[es.df.index[0], col] = fixed[col]
    assert not has_flag(es.df["flag"].iloc[0], FLAG_CLIPPED)
    assert int(es.provenance[CLIPPED_COUNT_KEY]) == 0

    # ...and the other direction: an edit that newly clips a row
    clipped = recharacterize_span(signal, FS, 200_300, 200_600,
                                  provenance=_prov(200_300, 260_000), prior=None).iloc[0]
    for col in ("start_idx", "flag", "accepted"):
        es.df.at[es.df.index[0], col] = clipped[col]
    assert int(es.provenance[CLIPPED_COUNT_KEY]) == 1


def test_a_set_that_never_carried_the_count_does_not_grow_one():
    """Provenance stays additive by its own choice -- the reconciliation corrects a key
    that is there, it does not invent one."""
    es = EventSet(empty_df(), {"detector": "csv"})
    assert CLIPPED_COUNT_KEY not in es.provenance


def test_n_clipped_boundary_is_the_counts_one_definition():
    df = empty_df()
    assert n_clipped_boundary(df) == 0
    signal = _trace([(200_000, 200_600)])
    es, _ = _ml_shaped_eventset(signal, [(200_300, 200_600)], 200_300, 260_000)
    assert n_clipped_boundary(es.df) == int(has_flag(es.df["flag"], FLAG_CLIPPED).sum()) == 1


# ===========================================================================
# the Schmitt detector, and segments
# ===========================================================================
def test_the_schmitt_detector_stamps_the_span_it_analysed():
    signal = _trace([(1_000, 1_400)], n=50_000)
    _, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4)
    assert analysed_span(prov, -1) == (0, 50_000)


def test_a_segments_own_edges_are_its_span_and_survive_a_rebuild():
    """A segment is a derived recording with its own doc_id and its own SEGMENT-LOCAL
    indices, so its span is ``(0, n_segment)``: an event cut by the segment boundary is
    genuinely clipped (the user's cut landed on it) and must keep saying so, while an
    event in the middle of the segment must never be flagged by the PARENT's edges."""
    parent = _trace([(199_900, 200_500), (210_000, 210_400)])
    seg = np.ascontiguousarray(parent[200_000:220_000])
    df, prov = detect_events(seg, FS, baseline=I0, min_dwell_s=1e-4)
    assert analysed_span(prov, -1) == (0, seg.size)

    clipped = int(np.flatnonzero(has_flag(df["flag"], FLAG_CLIPPED))[0])
    assert int(df["start_idx"].iloc[clipped]) == 0

    row = recharacterize_span(seg, FS, 0, int(df["end_idx"].iloc[clipped]),
                              provenance=prov, prior=df.iloc[clipped]).iloc[0]
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False

    interior = df[~has_flag(df["flag"], FLAG_CLIPPED)].iloc[0]
    back = recharacterize_span(seg, FS, int(interior["start_idx"]), int(interior["end_idx"]),
                               provenance=prov, prior=interior).iloc[0]
    assert str(back["flag"]) == ""
    assert bool(back["accepted"]) is True


# ===========================================================================
# the real recording
# ===========================================================================
@_needs_abf
def test_the_span_invariant_on_the_owners_recording():
    """A real translocation, with a real analysed range starting inside it. The rebuild
    must keep the flag; the same event judged against the whole recording must not have
    it -- which is what makes this fixture capable of catching the defect."""
    from nanoscope.io.base import open_document

    doc = open_document(ABF)
    ci = next(i for i, c in enumerate(doc.meta.channels)
              if c.kind.name == "CURRENT")
    cur = np.asarray(doc.channels[ci], dtype=np.float64)
    fs = float(doc.meta.sample_rate_hz)

    df, prov = detect_events(cur, fs, min_dwell_s=2e-5)
    assert analysed_span(prov, -1) == (0, cur.size)
    ev = df[df["dwell_s"] > 4e-3].iloc[0]
    s, e = int(ev["start_idx"]), int(ev["end_idx"])
    mid = (s + e) // 2

    ranged = {**prov, **span_provenance(mid, min(mid + 500_000, cur.size))}
    row = recharacterize_span(cur, fs, mid, e, provenance=ranged, prior=None).iloc[0]
    assert has_flag(row["flag"], FLAG_CLIPPED)
    assert bool(row["accepted"]) is False
    assert float(row["dwell_s"]) < float(ev["dwell_s"])     # the dwell is a lower bound

    whole = recharacterize_span(cur, fs, mid, e, provenance=prov, prior=None).iloc[0]
    assert str(whole["flag"]) == ""


# ===========================================================================
# the structural guard
# ===========================================================================
_SRC = Path(__file__).resolve().parents[2] / "nanoscope"


def _called_name(node: ast.Call) -> str | None:
    """The function name a call site spells, bare OR as a module attribute.

    Matching only ``ast.Name`` let ``_char_mod.recharacterize_span(sig, fs, a, b,
    prior=None)`` -- a rebuild with no provenance, the §2.2 defect itself -- pass
    this guard in an out-of-tree repo copy. Same helper as
    ``tests/analysis/test_flag_invariant.py``; both guards are on a call SHAPE and
    have to see every spelling of it.
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def test_every_rebuild_hands_over_the_provenance_that_carries_the_span():
    """The span lives in provenance precisely so no caller has to remember it -- which
    only holds while every caller hands the provenance over. ``noise_rms_pA``,
    ``max_dwell_s``, ``filter_rise_s`` and the polarity ride the same carrier, so a call
    without it is silently measuring on a different basis as well."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and _called_name(node) == "recharacterize_span"):
                continue
            if not any(kw.arg == "provenance" for kw in node.keywords):
                offenders.append(f"{path.relative_to(_SRC.parent)}:{node.lineno}")
    assert not offenders, (
        f"recharacterize_span() is called without provenance= at {offenders}; the "
        "analysed span, the noise scale, the max_dwell_s limit and the polarity all "
        "travel in it, so the rebuilt row would be judged against the wrong span"
    )


def test_the_span_keys_have_exactly_one_writer():
    """Two spellings of the span is how the reader and the writer drift apart."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC.parent))
        if rel == "nanoscope/analysis/events.py":       # defines them
            continue
        tree = ast.parse(path.read_text())
        docstrings = {
            ast.get_docstring(n, clean=False)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in (ANALYSED_START_KEY, ANALYSED_STOP_KEY)
                    and node.value not in docstrings):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        f"the analysed-span provenance keys are spelled by hand at {offenders}; write "
        "them with events.span_provenance() and read them with events.analysed_span()"
    )
