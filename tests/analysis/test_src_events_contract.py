"""Section 7 — "each event should store Source ... and Status: unreviewed / accepted /
rejected", and section 8's "preserve provenance".

TWO FACTS WERE MISSING PER ROW, and this file is the contract for both.

* **Source.** Which detector authored a row was knowable only from the ``module_id`` on
  the enclosing ``AnalysisResult`` — per TABLE. The moment a table stops being one
  detector's output, which it does the first time somebody hand-adds a missed event into
  the same result, the table can no longer say which row is which.
* **Status.** ``accepted`` is a boolean, so "I reviewed this and kept it" and "I never
  looked at it" are the same cell, and a detection arrives with every row already
  accepted. That is the half that makes "curation determines the final dataset" untrue
  today.

The shape under test is that REVIEWEDNESS IS STORED AND STATUS IS DERIVED. ``accepted``
keeps its exact present meaning and its single writer — every export, figure and fit that
reads it reads the same rows as before — and the new column answers only "has a person
ruled on this row at all". A second three-valued column beside ``accepted`` was the
alternative, and it is how this app already earned a provenance count that disagreed with
its own table.

Both columns are declared OUTSIDE ``SCHEMA``, which is the compatibility mechanism this
codebase has now used three times (``scoring.ADDED_COLUMN``, ``ORIGINAL_BOUNDS_SCHEMA``,
this): ``EventSet.__post_init__`` raises only on a missing REQUIRED column, so an old
project still loads, and ``characterize`` ends with ``df[list(SCHEMA)]``, so every rebuild
path must re-apply them deliberately. The owner was told he would not have to migrate
every saved project; these tests are that promise.

The migration is HONEST, which is the other half of the file: an old row whose
reviewedness nobody recorded reads UNKNOWN — not "unreviewed" (a claim that it is a fresh
candidate) and not "reviewed" (a claim that somebody looked).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.characterize import recharacterize_span
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    FLAG_CLIPPED,
    KIND_CONVENTIONAL,
    KIND_MANUAL,
    KIND_ML,
    KIND_MIXED,
    KIND_UNKNOWN,
    REQUIRED,
    REVIEW_COL,
    REVIEW_DONE,
    REVIEW_NONE,
    REVIEW_VALUES,
    SCHEMA,
    SOURCE_COL,
    SOURCE_KIND_BY_MODULE,
    SOURCE_MANUAL,
    SOURCE_MIXED,
    SOURCE_STATUS_SCHEMA,
    STATUS_ACCEPTED,
    STATUS_AUTO_REJECTED,
    STATUS_REJECTED,
    STATUS_UNKNOWN,
    STATUS_UNREVIEWED,
    EventSet,
    carry_source_status,
    combine_reviews,
    combine_sources,
    empty_df,
    ensure_source_status,
    event_status,
    has_source_status,
    is_reviewed,
    mark_reviewed,
    review_of,
    source_disagrees_with_added,
    source_for_module,
    source_kind,
    source_of,
    stamp_source_status,
    status_of,
)
from nanoscope.analysis.scoring import ADDED_COLUMN, ensure_added_column

FS = 50_000.0
N = 40_000
I0 = -900.0
_SPANS = [(5_000, 5_600), (15_000, 15_500), (25_000, 25_800)]


def _signal(seed: int = 0) -> np.ndarray:
    """Three square blockades on a noisy open pore, the fixture the sibling
    ``test_original_bounds.py`` uses, so the two contracts are measured on one trace."""
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 4.0, N)
    for s, e in _SPANS:
        x[s:e] -= 250.0
    return np.ascontiguousarray(x, dtype=np.float64)


@pytest.fixture
def detected():
    """``(signal, df, provenance)`` straight from the real Schmitt detector — no columns
    beyond the frozen 20, which is what a pre-change detection looks like."""
    sig = _signal()
    df, prov = detect_events(sig, FS, baseline=I0, min_dwell_s=1e-4)
    return sig, df, prov


# ================================================================================================
# the compatibility mechanism — why neither column is in the frozen schema
# ================================================================================================

def test_neither_column_is_declared_inside_the_frozen_twenty():
    """If either lands in ``SCHEMA`` the whole design is undone: ``SCHEMA`` is ``REQUIRED``,
    so every project saved before this change stops loading."""
    assert SOURCE_COL not in SCHEMA and SOURCE_COL not in REQUIRED
    assert REVIEW_COL not in SCHEMA and REVIEW_COL not in REQUIRED
    assert set(SOURCE_STATUS_SCHEMA) == {SOURCE_COL, REVIEW_COL}
    assert len(SCHEMA) == 20, "the frozen 20 moved; that is the one thing this may not do"


def test_an_event_set_from_before_this_change_still_loads(detected):
    """The migration promise, stated as the one call that would have broken it."""
    _, df, prov = detected
    assert not has_source_status(df)
    es = EventSet(df, prov)                       # must not raise
    assert len(es) == len(df)


def test_the_declared_dtypes_are_what_the_columns_actually_carry(detected):
    """``SOURCE_STATUS_SCHEMA`` is a contract a reader consults instead of guessing, the
    same as ``SCHEMA`` and ``ORIGINAL_BOUNDS_SCHEMA``. It has to be true."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt")
    for col, (dtype, _unit, _desc) in SOURCE_STATUS_SCHEMA.items():
        assert str(out[col].dtype) == "string" == dtype


def test_the_review_column_is_not_a_nullable_bool_because_a_merge_would_raise_on_it():
    """The dtype is load-bearing, and this is the measurement behind it.

    ``merge._merged_row`` carries every non-schema column across a merged run with
    ``if is_bool_dtype(series): series.to_numpy(dtype=bool).any()``. ``is_bool_dtype`` is
    True for pandas' nullable ``boolean`` and ``to_numpy(dtype=bool)`` raises on a missing
    value — so an old project (every row UNKNOWN) reopened and then merged would raise out
    of a Qt slot. A ``string`` column takes the other branch.
    """
    nullable_bool = pd.Series([True, pd.NA], dtype="boolean")
    assert pd.api.types.is_bool_dtype(nullable_bool)
    with pytest.raises(ValueError, match="missing values"):
        nullable_bool.to_numpy(dtype=bool)

    ours = pd.Series([REVIEW_DONE, pd.NA], dtype="string")
    assert not pd.api.types.is_bool_dtype(ours)


def test_a_merge_carries_the_pair_across_without_raising():
    """The same claim end to end, through the real merge engine, on a frame whose
    reviewedness is partly unknown — the exact state an un-migrated project is in."""
    from nanoscope.analysis.merge import merge_by_gap

    rng = np.random.default_rng(3)
    sig = np.full(N, I0) + rng.normal(0.0, 4.0, N)
    for s, e in [(5_000, 5_600), (5_700, 6_300)]:     # one translocation, two fragments
        sig[s:e] -= 250.0
    df, prov = detect_events(np.ascontiguousarray(sig), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == 2
    ensure_source_status(df, source="detect.schmitt", review=REVIEW_NONE)
    df.loc[0, REVIEW_COL] = pd.NA                 # an un-migrated row in a live table

    new_events, n_merged = merge_by_gap(EventSet(df, prov), sig, FS, max_gap_s=0.01)
    merged = new_events.df
    assert n_merged == 1 and len(merged) == 1
    assert has_source_status(merged)


# ================================================================================================
# SOURCE — one spelling, taken from the module_id that already exists
# ================================================================================================

def test_the_source_token_is_the_module_id_verbatim_so_no_caller_invents_a_spelling():
    """Identity by design: there is no second vocabulary to keep in step with
    ``sources.DETECTOR_NAMES``, which is already the one writer of "Schmitt"/"CUSUM"/
    "Keras" for a legend."""
    for module_id in ("detect.schmitt", "detect.cusum", "detect.keras"):
        assert source_for_module(module_id) == module_id


def test_every_detector_the_app_can_run_has_a_declared_kind():
    """A fourth detector cannot be registered without landing in the kind map — otherwise
    "conventional detector / ML detector" would silently start reporting "unknown"."""
    from nanoscope.core.sources import DETECTION_MODULE_IDS

    assert set(DETECTION_MODULE_IDS) <= set(SOURCE_KIND_BY_MODULE)
    assert source_kind("detect.schmitt") == source_kind("detect.cusum") == KIND_CONVENTIONAL
    assert source_kind("detect.keras") == KIND_ML
    assert source_kind(SOURCE_MANUAL) == KIND_MANUAL
    assert source_kind(SOURCE_MIXED) == KIND_MIXED


def test_the_manual_token_cannot_be_confused_with_a_detector_id():
    """Every id this app registers is dotted; ``manual`` is not, so a stored cell is
    unambiguous without a second column saying which kind of token it holds."""
    from nanoscope.core.sources import DETECTION_MODULE_IDS

    assert SOURCE_MANUAL not in DETECTION_MODULE_IDS
    assert all("." in mid for mid in DETECTION_MODULE_IDS)
    assert "." not in SOURCE_MANUAL


def test_the_column_is_not_the_provenance_key_that_shares_its_name(detected):
    """Detection provenance has carried a ``source`` key since before this column existed,
    and it holds the RECORDING NAME. The two land side by side in the events export — the
    CSV gets the column, the ``.meta.json`` sidecar gets the key — so a reader must never
    fold them together."""
    sig, _, _ = detected
    named, prov = detect_events(sig, FS, baseline=I0, min_dwell_s=1e-4,
                                recording="apo-1000mV")
    assert "source" in prov, "the collision this test exists for has gone away"
    assert prov["source"] == "apo-1000mV"                     # the FILE, not a detector
    out = stamp_source_status(named, "detect.schmitt")
    assert out[SOURCE_COL].iloc[0] == "detect.schmitt"        # the DETECTOR, not a file
    assert out[SOURCE_COL].iloc[0] != prov["source"]


def test_an_unknown_detector_id_reads_back_as_itself_rather_than_as_a_guess():
    """The rule ``sources.detector_name`` already follows: an id this build does not ship
    is not evidence of anything, so it is neither dropped nor bucketed."""
    assert source_for_module("detect.experimental") == "detect.experimental"
    assert source_kind("detect.experimental") == KIND_UNKNOWN


def test_a_blank_module_id_is_refused_instead_of_being_stamped_as_unknown():
    """Stamping ``pd.NA`` for a missing id would record "no detector authored this row",
    which is what MANUAL means — and mislabelling a detector's row as hand-added moves the
    detector's own score, because ``added`` counts as a false negative."""
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="module_id"):
            source_for_module(bad)


# ================================================================================================
# STATUS — stored reviewedness, derived status, ``accepted`` untouched
# ================================================================================================

def test_a_fresh_detection_is_a_pile_of_candidates_not_of_accepted_events(detected):
    """Section 7's missing half. Every row a detector produces is UNREVIEWED, and that is
    now readable — while ``accepted`` still says exactly what it said before."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt")
    assert set(out[REVIEW_COL]) == {REVIEW_NONE}
    assert list(event_status(out)) == [STATUS_UNREVIEWED] * len(out)
    assert out["accepted"].tolist() == df["accepted"].tolist()


def test_accepted_is_still_the_single_answer_to_what_is_in_the_final_dataset(detected):
    """The invariant every export, figure and fit rests on: the new column changes which
    rows can be DESCRIBED, never which rows are SELECTED."""
    _, df, prov = detected
    before = EventSet(df.copy(), prov).accepted.df[list(SCHEMA)]
    stamped = stamp_source_status(df, "detect.schmitt")
    mark_reviewed(stamped)
    after = EventSet(stamped, prov).accepted.df[list(SCHEMA)]
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_status_is_derived_and_is_never_a_column_of_its_own(detected):
    """Two facts in two places is how this app got a provenance count that disagreed with
    its own table. There is no ``status`` column to disagree with anything."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt")
    assert "status" not in out.columns
    assert set(out.columns) == set(REQUIRED) | {SOURCE_COL, REVIEW_COL}


def test_the_row_reader_and_the_frame_reader_cannot_answer_differently(detected):
    """One rule, two shapes: a table column, a summary count and a per-event read-out all
    have to agree, so they share the four rules rather than each implementing them."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt").copy()
    out.loc[0, REVIEW_COL] = REVIEW_DONE
    out.loc[1, REVIEW_COL] = REVIEW_DONE
    out.loc[1, "accepted"] = False
    out.loc[2, REVIEW_COL] = pd.NA
    assert list(event_status(out)) == [status_of(r) for _, r in out.iterrows()]
    assert list(event_status(out)) == [STATUS_ACCEPTED, STATUS_REJECTED, STATUS_UNKNOWN]


def test_a_row_the_app_rejected_is_not_reported_as_a_curators_rejection(detected):
    """``is_automatic_rejection`` is already THE definition of "the app rejected this row,
    not a person" — three existing readers rely on it. Reporting a boundary-clipped row as
    "rejected" would attribute to the curator a decision he never made."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt").copy()
    out.loc[0, "accepted"] = False
    out.loc[0, "flag"] = FLAG_CLIPPED
    assert event_status(out)[0] == STATUS_AUTO_REJECTED
    assert status_of(out.iloc[0]) == STATUS_AUTO_REJECTED


def test_a_hand_added_event_clipped_by_the_span_edge_is_still_the_apps_rejection(detected):
    """The case that decides the precedence. A hand-added row is REVIEWED (a person put it
    there) and can arrive ``accepted=False`` because the selection ran off the end — and
    calling that "rejected" would say the curator threw away the event he had just added."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt").copy()
    ensure_added_column(out)
    out.loc[2, ADDED_COLUMN] = True
    out.loc[2, SOURCE_COL] = SOURCE_MANUAL
    out.loc[2, REVIEW_COL] = REVIEW_DONE
    out.loc[2, "accepted"] = False
    out.loc[2, "flag"] = FLAG_CLIPPED
    assert event_status(out)[2] == STATUS_AUTO_REJECTED


def test_a_detection_from_before_this_change_reads_unknown_and_is_never_backfilled(detected):
    """The honest migration. An old row's reviewedness was not recorded, so it is UNKNOWN —
    not "unreviewed", which would claim it is a fresh candidate nobody has seen, on a table
    the owner may have curated for an hour."""
    _, df, _ = detected
    assert list(event_status(df)) == [STATUS_UNKNOWN] * len(df)
    assert source_of(df.iloc[0]) is None
    assert review_of(df.iloc[0]) is None
    assert is_reviewed(df.iloc[0]) is None


def test_an_old_frame_still_names_the_detectors_own_rejects(detected):
    """Rules 1 and 4 need no new column, so the pre-change rows are not uniformly opaque:
    what the FLAGS already say is still said."""
    _, df, _ = detected
    df = df.copy()
    df.loc[1, "accepted"] = False
    df.loc[1, "flag"] = FLAG_CLIPPED
    assert list(event_status(df)) == [STATUS_UNKNOWN, STATUS_AUTO_REJECTED, STATUS_UNKNOWN]


# ================================================================================================
# ensure — the caller has to say which it means
# ================================================================================================

def test_ensure_defaults_to_unknown_so_a_caller_who_does_not_know_cannot_assert(detected):
    """Both defaults are ``pd.NA``. The only way to put "unreviewed" on a row is to say so,
    which is the difference between a migration and a fabrication."""
    _, df, _ = detected
    df = df.copy()
    ensure_source_status(df)
    assert has_source_status(df)
    assert df[SOURCE_COL].isna().all() and df[REVIEW_COL].isna().all()
    assert list(event_status(df)) == [STATUS_UNKNOWN] * len(df)


def test_ensure_fills_in_place_because_the_table_and_the_inspector_share_one_frame(detected):
    """``scoring.ensure_added_column``'s idiom, for its reason: the event table, the Event
    Inspector and the undo recorder hold the SAME object, so a copy would give the columns
    to nobody."""
    _, df, _ = detected
    same = ensure_source_status(df, source="detect.cusum", review=REVIEW_NONE)
    assert same is df
    assert set(df[SOURCE_COL]) == {"detect.cusum"}


def test_ensure_never_overwrites_a_value_somebody_already_recorded(detected):
    """Idempotent, so it can be called on every display without erasing curation."""
    _, df, _ = detected
    df = df.copy()
    ensure_source_status(df, source="detect.schmitt", review=REVIEW_NONE)
    mark_reviewed(df, np.array([True, False, False]))
    ensure_source_status(df, source="detect.keras", review=REVIEW_NONE)
    assert set(df[SOURCE_COL]) == {"detect.schmitt"}
    assert df[REVIEW_COL].tolist() == [REVIEW_DONE, REVIEW_NONE, REVIEW_NONE]


def test_a_blank_source_string_is_refused_because_it_is_not_the_same_as_unknown(detected):
    _, df, _ = detected
    with pytest.raises(ValueError, match="unknown"):
        ensure_source_status(df.copy(), source="")


def test_a_review_token_outside_the_vocabulary_is_refused(detected):
    _, df, _ = detected
    with pytest.raises(ValueError, match="review"):
        ensure_source_status(df.copy(), review="maybe")
    assert set(REVIEW_VALUES) == {REVIEW_DONE, REVIEW_NONE}


# ================================================================================================
# ``added`` keeps its job, and ``source`` agrees with it
# ================================================================================================

def test_a_hand_added_row_is_manual_whatever_the_caller_says_the_detector_was(detected):
    """``added`` is the existing recorded fact — the detector score reads it as a false
    negative and the training export ships it — so ``source`` is stamped to AGREE with it
    rather than to contradict it, in one place."""
    _, df, _ = detected
    df = df.copy()
    ensure_added_column(df)
    df.loc[1, ADDED_COLUMN] = True
    ensure_source_status(df, source="detect.schmitt", review=REVIEW_NONE)
    assert df[SOURCE_COL].tolist() == ["detect.schmitt", SOURCE_MANUAL, "detect.schmitt"]
    assert not source_disagrees_with_added(df).any()


def test_the_added_flag_wins_for_source_and_says_nothing_about_review(detected):
    """CHANGED, and the change is the point. This asserted that ``ensure_source_status``
    writes REVIEWED for every ``added`` row, on the reasoning that creating an event is a
    curation gesture. True of the GESTURE, false of the FLAG: ``merge._merged_row`` ORs
    ``added`` across a run, so a merged row inherits it from one hand-added fragment and
    was then promoted from "nobody recorded a ruling" to "reviewed" by the install door —
    undoing, one function later, exactly what ``combine_reviews`` had just refused to do
    (see the merge test below). The door cannot tell creating from inheriting, so it no
    longer guesses: ``source`` still agrees with ``added``, ``review`` stays as recorded,
    and the gesture records its own reviewedness through ``mark_reviewed``."""
    _, df, _ = detected
    df = df.copy()
    ensure_added_column(df)
    df.loc[1, ADDED_COLUMN] = True
    ensure_source_status(df, source="detect.schmitt", review=REVIEW_NONE)
    assert df[SOURCE_COL].tolist() == ["detect.schmitt", SOURCE_MANUAL, "detect.schmitt"]
    assert df[REVIEW_COL].tolist()[0] == REVIEW_NONE
    assert df[REVIEW_COL].tolist()[1] is pd.NA        # not claimed either way
    assert event_status(df)[1] == STATUS_UNKNOWN

    # ...and the gesture that created it is what says a person ruled.
    mark_reviewed(df, np.array([False, True, False]))
    assert df[REVIEW_COL].tolist()[1] == REVIEW_DONE
    assert event_status(df)[1] == STATUS_ACCEPTED


def test_added_is_not_derived_from_source_and_keeps_its_own_column(detected):
    """Deriving ``added`` would change a number the detector's score already reports, and
    this change is description, not measurement."""
    _, df, _ = detected
    df = df.copy()
    ensure_added_column(df)
    df.loc[1, ADDED_COLUMN] = True
    ensure_source_status(df, source="detect.schmitt", review=REVIEW_NONE)
    assert ADDED_COLUMN in df.columns
    assert df[ADDED_COLUMN].dtype == bool
    assert df[ADDED_COLUMN].tolist() == [False, True, False]


def test_a_contradiction_between_the_two_is_reported_and_not_quietly_repaired(detected):
    """Rewriting one of two recorded facts to match the other is how the true one is lost."""
    _, df, _ = detected
    df = df.copy()
    ensure_added_column(df)
    df.loc[1, ADDED_COLUMN] = True
    ensure_source_status(df, source="detect.schmitt", review=REVIEW_NONE)
    df.loc[1, SOURCE_COL] = "detect.keras"                # somebody wrote a lie
    bad = source_disagrees_with_added(df)
    assert bad.tolist() == [False, True, False]
    assert df.loc[1, SOURCE_COL] == "detect.keras", "reported, not repaired"


def test_an_unstamped_row_disagrees_with_nothing(detected):
    """``pd.NA`` makes no claim, so it cannot contradict one."""
    _, df, _ = detected
    df = df.copy()
    ensure_added_column(df)
    df.loc[1, ADDED_COLUMN] = True
    ensure_source_status(df)                              # everything unknown
    assert not source_disagrees_with_added(df).any()


# ================================================================================================
# the detection writer
# ================================================================================================

def test_stamping_a_curated_table_a_second_time_leaves_it_alone(detected):
    """``stamp_original_bounds``' rule, for the same reason: re-stamping after curation
    would write the current table under the detector's name."""
    _, df, _ = detected
    once = stamp_source_status(df, "detect.schmitt")
    mark_reviewed(once, np.array([True, False, False]))
    again = stamp_source_status(once, "detect.cusum")
    assert again is once
    assert set(once[SOURCE_COL]) == {"detect.schmitt"}


def test_a_re_detection_may_overwrite_but_still_may_not_claim_a_hand_added_row(detected):
    """The one caller entitled to ``overwrite`` genuinely replaced the rows — and a
    hand-added event is still not the detector's, no matter who is stamping.

    CHANGED with the door above, for the same reason: the ``added`` row's ``review`` was
    asserted REVIEWED and is now ``pd.NA``. A detector cannot state whether a person has
    ruled on a row it did not author — ``unreviewed`` would erase a ruling and
    ``reviewed`` would invent one — and ``added`` is ORed across a merge, so it is not by
    itself evidence that anybody looked."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt")
    ensure_added_column(out)
    out.loc[2, ADDED_COLUMN] = True
    forced = stamp_source_status(out, "detect.keras", overwrite=True)
    assert forced[SOURCE_COL].tolist() == ["detect.keras", "detect.keras", SOURCE_MANUAL]
    assert forced[REVIEW_COL].tolist()[:2] == [REVIEW_NONE, REVIEW_NONE]
    assert forced[REVIEW_COL].tolist()[2] is pd.NA


def test_the_detection_writer_returns_a_copy_and_leaves_the_input_alone(detected):
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt")
    assert out is not df
    assert not has_source_status(df)


def test_an_empty_detection_still_carries_the_columns(detected):
    """So an empty result round-trips through save/load with the same shape as a full one."""
    out = stamp_source_status(empty_df(), "detect.schmitt")
    assert has_source_status(out) and len(out) == 0
    assert list(event_status(out)) == []


# ================================================================================================
# the rebuild path — ``characterize`` drops everything outside SCHEMA
# ================================================================================================

def test_a_rebuilt_row_arrives_without_the_pair_which_is_why_the_door_exists(detected):
    """``recharacterize_span`` is the ONE function behind the mask drag, the merge and the
    re-measure, and it ends at ``df[list(SCHEMA)]``."""
    sig, df, prov = detected
    row = recharacterize_span(
        sig, FS, int(df["start_idx"].iloc[0]), int(df["end_idx"].iloc[0]),
        provenance=prov, prior=df.iloc[[0]], polarity=-1,
        recording="r", pore_id="p", units="pA")
    assert not has_source_status(row)


def test_the_rebuild_door_puts_the_pair_back_by_event_id(detected):
    """Alignment is by ``event_id``, because "the row's IDENTITY is not a measurement" —
    a re-measure preserves it by design."""
    _, df, _ = detected
    stamped = stamp_source_status(df, "detect.keras")
    mark_reviewed(stamped, np.array([False, True, False]))
    rebuilt = stamped[list(SCHEMA)].iloc[[2, 0, 1]].reset_index(drop=True)   # reordered
    back = carry_source_status(rebuilt, stamped)
    assert back[SOURCE_COL].tolist() == ["detect.keras"] * 3
    assert back[REVIEW_COL].tolist() == [REVIEW_NONE, REVIEW_NONE, REVIEW_DONE]


def test_the_rebuild_door_is_a_no_op_on_a_table_that_never_had_the_pair(detected):
    """Safe to call unconditionally, so a rebuild path has one line and not a branch."""
    _, df, _ = detected
    rebuilt = df[list(SCHEMA)].copy()
    assert carry_source_status(rebuilt, df) is rebuilt


def test_the_rebuild_door_refuses_to_guess_when_it_cannot_align(detected):
    """A merge reassigns ``event_id`` and changes the row count. A mis-aligned provenance
    column is worse than an absent one, because it reads as a fact."""
    _, df, _ = detected
    stamped = stamp_source_status(df, "detect.schmitt")
    merged_shape = stamped[list(SCHEMA)].iloc[[0, 1]].copy()
    merged_shape["event_id"] = [7, 8]
    with pytest.raises(ValueError, match="align"):
        carry_source_status(merged_shape, stamped)


def test_a_re_measure_brings_the_pair_through_untouched(detected):
    """The other rebuild path. ``remeasure`` copies every non-schema column per row from
    the source row, so curation survives a change of open-pore estimator — which it must,
    since a re-measure revises the BASELINE and says nothing about who found the event or
    who has looked at it."""
    from nanoscope.analysis.remeasure import remeasure_events

    sig, df, prov = detected
    stamped = stamp_source_status(df, "detect.cusum")
    mark_reviewed(stamped, np.array([False, True, False]))
    out = remeasure_events(EventSet(stamped, prov), sig, FS)
    after = out.events.df
    assert after[SOURCE_COL].tolist() == ["detect.cusum"] * 3
    assert after[REVIEW_COL].tolist() == [REVIEW_NONE, REVIEW_DONE, REVIEW_NONE]


def test_a_rebuilt_row_with_no_match_reads_unknown_rather_than_borrowing_a_neighbour(detected):
    _, df, _ = detected
    stamped = stamp_source_status(df, "detect.schmitt")
    rebuilt = stamped[list(SCHEMA)].copy()
    back = carry_source_status(rebuilt, stamped.iloc[:3])
    assert back[SOURCE_COL].notna().all()
    prior = stamped.copy()
    prior[SOURCE_COL] = pd.array([pd.NA] * len(prior), dtype="string")
    assert carry_source_status(rebuilt, prior)[SOURCE_COL].isna().all()


# ================================================================================================
# merging fragments — the conservative direction is the load-bearing one
# ================================================================================================

def test_two_sources_in_one_merged_row_are_named_rather_than_resolved():
    """Picking either contributor would put words in somebody's mouth — and picking
    ``manual`` would move the detector's score."""
    assert combine_sources(["detect.schmitt", "detect.schmitt"]) == "detect.schmitt"
    assert combine_sources(["detect.schmitt", SOURCE_MANUAL]) == SOURCE_MIXED
    assert combine_sources([pd.NA, "detect.cusum"]) == "detect.cusum"
    assert combine_sources([pd.NA, pd.NA]) is pd.NA


def test_a_merged_event_is_reviewed_only_when_every_fragment_was():
    """Claiming a merged row is reviewed because one fragment was is how a "final dataset"
    ends up holding spans nobody looked at."""
    assert combine_reviews([REVIEW_DONE, REVIEW_DONE]) == REVIEW_DONE
    assert combine_reviews([REVIEW_DONE, REVIEW_NONE]) == REVIEW_NONE
    assert combine_reviews([REVIEW_DONE, pd.NA]) is pd.NA
    assert combine_reviews([]) is pd.NA


# ================================================================================================
# the curation gesture
# ================================================================================================

def test_marking_a_row_reviewed_writes_reviewedness_and_nothing_else(detected):
    """Which way the ruling went is ``accepted``, whose writer is the panel. A second
    writer for that column is the defect this design exists to avoid."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt")
    before = out[list(SCHEMA)].copy()
    n = mark_reviewed(out, np.array([True, False, True]))
    assert n == 2
    pd.testing.assert_frame_equal(before, out[list(SCHEMA)])
    assert out[REVIEW_COL].tolist() == [REVIEW_DONE, REVIEW_NONE, REVIEW_DONE]


def test_marking_reports_how_many_cells_actually_moved(detected):
    """So a caller can skip a no-op undo checkpoint."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt")
    assert mark_reviewed(out) == 3
    assert mark_reviewed(out) == 0
    assert mark_reviewed(out, reviewed=False) == 3


def test_marking_a_frame_that_never_had_the_columns_leaves_the_others_unknown(detected):
    """The materialisation must not backfill the rows it was not asked about."""
    _, df, _ = detected
    df = df.copy()
    assert mark_reviewed(df, np.array([False, True, False])) == 1
    assert df[REVIEW_COL].tolist()[0] is pd.NA
    assert list(event_status(df)) == [STATUS_UNKNOWN, STATUS_ACCEPTED, STATUS_UNKNOWN]


# ================================================================================================
# the pair survives the trip to disk
# ================================================================================================

def test_the_pair_round_trips_through_the_projects_parquet_member(detected):
    """A project save writes ``EventSet.df`` to parquet; provenance that does not survive
    a reopen is not provenance."""
    from nanoscope.core.project import _df_to_parquet_bytes, _parquet_bytes_to_df

    _, df, _ = detected
    out = stamp_source_status(df, "detect.cusum")
    mark_reviewed(out, np.array([True, False, False]))
    out.loc[2, REVIEW_COL] = pd.NA
    back = _parquet_bytes_to_df(_df_to_parquet_bytes(out))
    assert str(back[SOURCE_COL].dtype) == "string"
    assert back[REVIEW_COL].tolist()[:2] == [REVIEW_DONE, REVIEW_NONE]
    assert back[REVIEW_COL].isna().tolist() == [False, False, True]
    assert list(event_status(back)) == list(event_status(out))


# ================================================================================================
# THE INVARIANT — no detected number moves
# ================================================================================================

def test_stamping_moves_no_measured_number(detected):
    """This change is DESCRIPTION, not measurement. The frozen 20 hash identically before
    and after — asserted here on the frame, and measured separately against the pre-change
    tree on the owner's own -1000 mV recording."""
    from pandas.util import hash_pandas_object

    _, df, _ = detected
    before = hash_pandas_object(df[list(SCHEMA)], index=True).sum()
    out = stamp_source_status(df, "detect.schmitt")
    ensure_source_status(out, review=REVIEW_NONE)
    mark_reviewed(out, np.array([True, False, False]))
    after = hash_pandas_object(out[list(SCHEMA)], index=True).sum()
    assert before == after
    assert list(out[list(SCHEMA)].columns) == list(SCHEMA)
