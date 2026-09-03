"""EXPL3-06 — the subpopulation report may not call a curated subset "detected regions".

``run_subpopulations`` is documented to take ``events.accepted``, and the app's entry
point passes exactly that. So on every run the app produces, the number the report printed
under the words ``detected regions`` was the count of events the user curated IN. On a
300-event detection with 180 curated out the exported file read

    detected regions        : 120

and headlined *"120 of 120 detected region(s) passed the gate"*, while the app's own status
bar for the same run said *"Grouping 120 accepted event(s) into subpopulations"*. A regex
for ``accept|curat`` over the whole report text found nothing: the document itemises every
other row the pipeline drops — the ≥σ gate and the too-short-to-measure floor, each with a
count and a named mechanism — and never named the largest removal of all.

The repair is the one the CLOCK repair already taught this module: the report NAMES what
its own number counts, reading the fact off the provenance that travels with the rows
(:func:`subpop_report.rows_in_hand`) instead of re-deriving it or trusting the caller to
remember. ``EventSet.accepted`` stamps ``provenance["filtered"] = "accepted"``; the
detector stamps ``provenance["n_events"]``. Both are already there — nothing new is
measured and nothing is guessed.

These tests pin all four properties the fix has to have:
  1. rows that ARE the detection are still called "detected" (no report that was accurate
     changes);
  2. rows that say they were curated are called "accepted", everywhere the count is
     printed — the ledger, the headline, the gate caveat;
  3. the number removed is stated with the detection's own total when the rows carry it;
  4. when they do not carry it, the report says curation happened AND says it cannot state
     the count — rather than inventing one.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import REQUIRED, EventSet

FS = 50_000.0
N_DETECTED = 160


def _synth(seed: int = 5, n: int = 260_000):
    """A trace with two obvious event classes and the table that describes them."""
    rng = np.random.default_rng(seed)
    baseline, sigma = 200.0, 2.0
    sig = (baseline + rng.normal(0.0, sigma, n)).astype(np.float32)
    rows, pos, i = [], 500, 0
    while pos + 400 < n and i < N_DETECTED:
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        depth = 60.0 if deep else 30.0
        sig[pos:pos + dwell] -= depth
        rows.append({
            "event_id": i, "recording": "synth", "pore_id": "p1",
            "start_idx": pos, "end_idx": pos + dwell, "t_start_s": pos / FS,
            "dwell_s": dwell / FS, "baseline_pA": baseline, "depth_pA": depth,
            "frac_depth": depth / baseline, "ecd_pA_s": depth * dwell / FS,
            "max_dev_pA": depth, "mean_dev_pA": depth, "rise_s": 1 / FS,
            "fall_s": 1 / FS, "snr": depth / sigma, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
        pos += int(rng.integers(500, 1100))
        i += 1
    return sig, pd.DataFrame(rows)[REQUIRED]


CHEAP = SR.SubpopConfig(k_grid=(1, 2), n_splits=2, n_boot=0, with_tsne=False,
                        with_umap=False)


def _run(sig, events):
    return SR.run_subpopulations(sig, sig.astype(np.float64), FS, events,
                                 recording="synth", duration_s=len(sig) / FS,
                                 config=CHEAP)


@pytest.fixture(scope="module")
def trace_and_table():
    return _synth()


# ================================================================================================
# rows_in_hand — the resolver, on its own
# ================================================================================================

def test_an_unfiltered_row_set_is_still_called_the_detection():
    meta, note = SR.rows_in_hand(EventSet(_synth()[1]), 160)
    assert meta["region_noun"] == SR.DETECTED_NOUN
    assert note is None, "a report that was already accurate must not grow a caveat"


def test_a_curated_row_set_is_named_and_its_removal_counted():
    _sig, df = _synth()
    df.loc[df.index[:100], "accepted"] = False
    es = EventSet(df, {"n_events": len(df)})
    meta, note = SR.rows_in_hand(es.accepted, len(df) - 100)
    assert meta["region_noun"] == SR.ACCEPTED_NOUN
    assert meta["n_detected"] == len(df)
    assert f"{100} of the detector's {len(df)}" in note


def test_a_curated_row_set_with_no_detector_count_says_it_cannot_state_the_number():
    """Honest about the hole rather than silent about the removal, and it does NOT
    reconstruct a total out of the event ids."""
    _sig, df = _synth()
    df.loc[df.index[:100], "accepted"] = False
    meta, note = SR.rows_in_hand(EventSet(df).accepted, len(df) - 100)
    assert meta["region_noun"] == SR.ACCEPTED_NOUN
    assert meta["n_detected"] is None
    assert "cannot state it" in note
    assert not re.search(r"\b100\b", note), "a count that is not known must not be printed"


def test_a_stale_detector_count_smaller_than_the_rows_is_refused():
    """The detector stamps ``n_events`` once and the table can legitimately grow (a
    hand-added missed event). Subtracting a stale stamp would print a negative removal —
    a fabricated number in the line that exists to stop a fabricated number."""
    _sig, df = _synth()
    meta, note = SR.rows_in_hand(EventSet(df, {"n_events": 3}).accepted, len(df))
    assert meta["n_detected"] is None
    assert "cannot state it" in note


def test_an_unrecognised_filter_mark_is_still_not_called_the_detection():
    """``accepted`` is the only mark anything stamps today. A report that treated an
    unknown mark as "the detection" would be this defect returning the moment a second
    filter is added, so an unknown mark is named with its own word and disclosed."""
    _sig, df = _synth()
    meta, note = SR.rows_in_hand(EventSet(df, {"filtered": "one_pore"}), len(df))
    assert meta["region_noun"] == "one_pore region"
    assert "carrying the mark 'one_pore'" in note
    assert "cannot say what it removed" in note


def test_curated_with_nothing_actually_removed_carries_no_removal_caveat():
    _sig, df = _synth()
    meta, note = SR.rows_in_hand(EventSet(df, {"n_events": len(df)}).accepted, len(df))
    assert meta["region_noun"] == SR.ACCEPTED_NOUN
    assert note is None


# ================================================================================================
# the report, through the real pipeline
# ================================================================================================

@pytest.fixture(scope="module")
def curated_report(trace_and_table):
    sig, df = trace_and_table
    df = df.copy()
    rng = np.random.default_rng(11)
    df.loc[rng.choice(df.index, size=60, replace=False), "accepted"] = False
    es = EventSet(df, {"n_events": len(df), "analyzed_duration_s": len(sig) / FS})
    return _run(sig, es.accepted), len(df), len(df) - 60


def test_the_ledger_line_is_labelled_with_the_word_for_its_own_number(curated_report):
    rep, n_detected, n_accepted = curated_report
    text = rep.text()
    assert f"accepted regions        : {n_accepted}" in text
    assert "detected regions        :" not in text, (
        "the curated-in count must not be labelled with the word for the detected count")
    assert rep.n_fired == n_accepted


def test_the_report_states_how_many_were_curated_out_and_out_of_how_many(curated_report):
    rep, n_detected, n_accepted = curated_report
    text = rep.text()
    assert f"curated out (not below) : {n_detected - n_accepted}" in text
    assert f"of {n_detected} detected region(s)" in text


def test_the_headline_every_surface_repeats_says_accepted(curated_report):
    """The headline is the single author of the claim — it reaches the report window's
    header and the app's status bar verbatim, so the noun has to be right THERE."""
    rep, _n_detected, n_accepted = curated_report
    assert rep.headline().startswith(
        f"{rep.n_kept} of {n_accepted} accepted region(s) passed the gate")
    assert "detected region(s) passed the gate" not in rep.headline()


def test_the_curation_caveat_is_first_and_names_the_mechanism(curated_report):
    rep, n_detected, n_accepted = curated_report
    assert rep.warnings, "a curated run always has something to disclose"
    first = rep.warnings[0]
    assert first.startswith("CURATION:")
    assert f"{n_detected - n_accepted} of the detector's {n_detected} region(s)" in first
    assert "curated OUT before this run started" in first


def test_the_gate_caveat_counts_the_same_set_it_is_about(curated_report):
    """The gate caveat is a fraction OF THE CURATED-IN SET; it used to say 'detected'."""
    rep, _n_detected, _n_accepted = curated_report
    gate = [w for w in rep.warnings if w.startswith("GATE:")]
    for w in gate:
        assert "accepted region(s) reached" in w
        assert "detected region(s)" not in w


def test_a_reader_of_the_exported_file_alone_learns_that_curation_happened(curated_report):
    """``write_report`` produces a document read WITHOUT the app beside it. This is the
    property the defect was reported on, stated as itself."""
    rep, _n_detected, _n_accepted = curated_report
    assert re.search(r"accept|curat", rep.text(), re.I)


# ================================================================================================
# the report that was already accurate must not move
# ================================================================================================

@pytest.fixture(scope="module")
def uncurated_report(trace_and_table):
    sig, df = trace_and_table
    return _run(sig, EventSet(df.copy(), {"analyzed_duration_s": len(sig) / FS}))


def test_an_uncurated_run_still_says_detected_regions(uncurated_report):
    text = uncurated_report.text()
    assert f"detected regions        : {uncurated_report.n_fired}" in text
    assert "accepted regions" not in text
    assert "curated out (not below)" not in text
    assert not [w for w in uncurated_report.warnings if w.startswith("CURATION:")]


def test_the_two_nouns_are_the_same_width_so_the_column_does_not_move(uncurated_report):
    assert len(SR.DETECTED_NOUN) == len(SR.ACCEPTED_NOUN)
    for line in uncurated_report.text().splitlines():
        if line.startswith(("detected regions", "accepted regions",
                            "curated out", "event-excluded baseline")):
            assert line.index(":") == 24, line
