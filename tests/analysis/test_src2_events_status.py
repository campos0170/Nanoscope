"""The source/review contract, at the four places it answered its own question twice.

Every test here reproduces a defect that shipped an hour ago and pins the fix. They were
all run against a copy of the pre-fix module (loaded out of tree by ``prefix_plugin.py``
in place of ``nanoscope.analysis.events``, so no source was edited back and forth) and
every one of them fails there:

* **A person's rejection read "unreviewed".** ``event_status`` rule 3 called a row that a
  curator explicitly rejected a candidate nobody had judged — while
  ``core.history._hand_curation`` read the IDENTICAL row through the IDENTICAL authority
  (``~accepted & ~is_automatic_rejection(flag)``) and counted it as a hand rejection. Two
  answers to one question, and the wired export published the wrong one.
* **A merge promoted "nobody recorded a ruling" to "reviewed".** ``combine_reviews``
  refuses to inherit reviewedness across a merge; ``ensure_source_status`` handed it back
  one function later because the merged row inherits ``added=True``.
* **``mark_reviewed`` returned 0 while mutating the frame**, so the checkpoint it exists
  to gate was skipped.
* **The row reader and the frame reader disagreed** on a frame without ``accepted`` and
  on a ``review`` cell with a leading space.

Plus the two claims that had to be made true rather than rewritten: the ``string`` dtype
justification, re-measured on this tree, and ``carry_source_status``' "THE REBUILD DOOR".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import events as ev
from nanoscope.analysis import scoring
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    FLAG_CLIPPED,
    REVIEW_COL,
    REVIEW_DONE,
    REVIEW_NONE,
    SOURCE_COL,
    SOURCE_MANUAL,
    STATUS_ACCEPTED,
    STATUS_AUTO_REJECTED,
    STATUS_REJECTED,
    STATUS_UNKNOWN,
    STATUS_UNREVIEWED,
    EventSet,
    ensure_source_status,
    event_status,
    mark_reviewed,
    source_disagrees_with_added,
    stamp_source_status,
    status_of,
)
from nanoscope.analysis.scoring import ADDED_COLUMN, ensure_added_column
from nanoscope.core import history

FS = 50_000.0
I0 = -900.0
N = 40_000


@pytest.fixture(scope="module")
def detected():
    """A real three-event detection: synthetic float32 at the owner's sampling rate."""
    rng = np.random.default_rng(11)
    sig = (np.full(N, I0) + rng.normal(0.0, 3.0, N)).astype(np.float32)
    for s, e in [(5_000, 5_400), (15_000, 15_500), (25_000, 25_300)]:
        sig[s:e] -= 250.0
    sig = np.ascontiguousarray(sig)
    df, prov = detect_events(sig, FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == 3
    return sig, df, prov


# ================================================================================================
# D1 — a person's rejection is a person's rejection, in both readers and in the export
# ================================================================================================

def test_a_rejection_nobody_can_explain_automatically_is_a_persons_rejection(detected):
    """The app must not answer one question two ways. ``core.history._hand_curation`` is
    the existing authority — it reads ``(~accepted) & ~is_automatic_rejection(flag)`` as a
    hand rejection, in the warning that tells the user what a re-detection will destroy —
    and the status derivation now agrees with it instead of adding a third opinion.

    Nothing in the app calls ``mark_reviewed`` yet, so this row's ``review`` cell still
    says 'unreviewed'; that is exactly the state in which the two readers used to
    contradict each other."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt").copy()
    out.loc[out.index[1], "accepted"] = False              # a curator pressed Reject

    assert history._hand_curation(out)["n_hand_rejected"] == 1
    assert list(event_status(out)) == [STATUS_UNREVIEWED, STATUS_REJECTED,
                                       STATUS_UNREVIEWED]
    assert status_of(out.iloc[1]) == STATUS_REJECTED
    # storage untouched: the derivation says what the pair MEANS, not what was written
    assert out[REVIEW_COL].iloc[1] == REVIEW_NONE
    assert out["accepted"].tolist() == [True, False, True]


def test_a_legacy_rejection_reads_rejected_and_not_unknown(detected):
    """The same rule where reviewedness was never recorded at all — a project saved
    before these columns existed. ``accepted=False`` with nothing automatic to explain it
    is a person's decision whoever wrote the file, which is why ``_hand_curation`` counts
    it without consulting ``review``."""
    _, df, _ = detected
    out = df.copy()                                        # no source/review columns
    out.loc[out.index[0], "accepted"] = False
    assert not ev.has_source_status(out)
    assert list(event_status(out)) == [STATUS_REJECTED, STATUS_UNKNOWN, STATUS_UNKNOWN]
    assert history._hand_curation(out)["n_hand_rejected"] == 1


def test_the_apps_own_rejection_is_still_not_attributed_to_the_curator(detected):
    """Rule 1 still runs first and still wins: ``is_automatic_rejection`` is THE
    definition of "the app rejected this row, not a person", and a boundary-clipped row
    must not be reported as a curator's decision just because it is not accepted."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt").copy()
    out.loc[out.index[0], "accepted"] = False
    out.loc[out.index[0], "flag"] = FLAG_CLIPPED
    assert event_status(out)[out.index[0]] == STATUS_AUTO_REJECTED
    assert status_of(out.iloc[0]) == STATUS_AUTO_REJECTED
    assert history._hand_curation(out)["n_hand_rejected"] == 0


def test_the_wired_export_publishes_the_rejection_rather_than_calling_it_unreviewed(
        detected, tmp_path):
    """The sidecar is already wired, so the wrong answer was already leaving the machine:
    ``curation.by_status`` counted a rejected row as an un-judged candidate."""
    _, df, _ = detected
    out = stamp_source_status(df, "detect.schmitt").copy()
    ensure_added_column(out)
    out.loc[out.index[1], "accepted"] = False

    paths = scoring.write_events_export(out, tmp_path / "ev.csv", provenance={"n_events": 3})
    block = json.loads(paths["meta"].read_text())[scoring.CURATION_BLOCK_KEY]
    assert block["by_status"] == {STATUS_UNREVIEWED: 2, STATUS_REJECTED: 1}


# ================================================================================================
# D-DESIGN-2 — creating an event is a curation gesture; inheriting an ``added`` flag is not
# ================================================================================================

def test_a_merge_does_not_promote_an_unrecorded_ruling_to_reviewed():
    """The measured gesture sequence: a project saved before these columns existed, one
    hand-added event, Merge all. ``merge_by_gap`` yields ``review=<NA>`` on purpose, and
    the install door used to overwrite it with 'reviewed' because the merged row inherits
    ``added=True`` from the hand-added fragment — so a merge marked as reviewed a span
    nobody had ruled on."""
    from nanoscope.analysis.merge import merge_by_gap

    rng = np.random.default_rng(0)
    sig = (np.full(50_000, -1000.0) + rng.normal(0.0, 8.0, 50_000)).astype(np.float32)
    for s, e in [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]:
        sig[s:e] -= 300.0
    sig = np.ascontiguousarray(sig)
    df, prov = detect_events(sig, FS, baseline=-1000.0, min_dwell_s=1e-4)

    ensure_added_column(df)
    df.loc[df["start_idx"] > 10_400, ADDED_COLUMN] = True        # the hand-added fragment
    ensure_source_status(df, source="detect.schmitt")            # legacy rows: unknown
    assert df[REVIEW_COL].iloc[0] is pd.NA

    merged, _ = merge_by_gap(EventSet(df, prov), sig, FS, max_gap_s=1e-3)
    row = merged.df[merged.df["start_idx"] < 20_000].iloc[0]
    assert bool(row[ADDED_COLUMN]) and row[REVIEW_COL] is pd.NA  # merge was conservative

    ensure_source_status(merged.df, source="detect.schmitt")     # the install door
    after = merged.df[merged.df["start_idx"] < 20_000].iloc[0]
    assert after[REVIEW_COL] is pd.NA, "a merge is not a ruling"
    assert status_of(after) == STATUS_UNKNOWN
    # SOURCE is the half that is legitimately inherited: the row really does have two
    # authors, and `merge` names that rather than resolving it. A mixed row is not a
    # contradiction of ``added`` -- it says a person is AMONG the authors.
    assert after[SOURCE_COL] == ev.SOURCE_MIXED
    assert not source_disagrees_with_added(merged.df).any()


def test_the_install_door_fills_source_from_added_and_leaves_review_alone(detected):
    """The unit form of the same separation, and the half that must NOT change: ``source``
    is still stamped to agree with ``added`` (the detector score reads that flag as a
    false negative, and a contradiction between the two is a bug worth reporting)."""
    _, df, _ = detected
    out = df.copy()
    ensure_added_column(out)
    out.loc[out.index[1], ADDED_COLUMN] = True
    ensure_source_status(out, source="detect.cusum", review=REVIEW_NONE)

    assert out[SOURCE_COL].tolist() == ["detect.cusum", SOURCE_MANUAL, "detect.cusum"]
    assert out[REVIEW_COL].tolist()[1] is pd.NA
    assert not source_disagrees_with_added(out).any()
    assert list(event_status(out)) == [STATUS_UNREVIEWED, STATUS_UNKNOWN,
                                       STATUS_UNREVIEWED]


def test_the_detector_does_not_state_a_persons_reviewedness_for_a_row_it_did_not_author(
        detected):
    """``stamp_source_status`` with ``overwrite`` is the re-detection's writer. It keeps
    ``manual`` for an ``added`` row — that row is not the detector's — and now leaves the
    reviewedness unknown rather than asserting 'reviewed' from a flag a merge can OR into
    place."""
    _, df, _ = detected
    once = stamp_source_status(df, "detect.schmitt")
    ensure_added_column(once)
    once.loc[once.index[2], ADDED_COLUMN] = True
    forced = stamp_source_status(once, "detect.keras", overwrite=True)
    assert forced[SOURCE_COL].tolist()[2] == SOURCE_MANUAL
    assert forced[REVIEW_COL].tolist()[2] is pd.NA


# ================================================================================================
# D-DESIGN-3 — a mutation that reports it did nothing
# ================================================================================================

def test_a_mark_that_changes_nothing_leaves_the_frame_a_snapshot_cannot_tell_apart(
        detected):
    """``core.history._snapshot`` compares WHOLE frames, so a column that appears midway
    is a difference the undo stack has no record of. An empty selection returns 0 — and
    now that 0 is true: nothing is materialised, nothing is filled."""
    _, df, _ = detected
    out = df.copy()
    ensure_added_column(out)
    out.loc[out.index[2], ADDED_COLUMN] = True
    before_cols, before_status = list(out.columns), list(event_status(out))

    assert mark_reviewed(out, rows=np.array([], dtype=np.int64)) == 0
    assert list(out.columns) == before_cols
    assert list(event_status(out)) == before_status


def test_the_count_includes_the_cells_the_materialisation_door_filled(detected):
    """The other half: when the call DOES touch the frame, the number it returns covers
    everything it changed — the ``review`` cell it was asked for and the ``source`` cells
    ``ensure_source_status`` filled on the way — so a caller may key an undo checkpoint on
    it. A hand-added row means one extra ``source`` cell moves from unknown to 'manual'."""
    _, df, _ = detected
    out = df.copy()
    ensure_added_column(out)
    out.loc[out.index[2], ADDED_COLUMN] = True

    n = mark_reviewed(out, np.array([True, False, False]))
    assert n == 2, "one review cell written, one source cell filled by the door"
    assert out[REVIEW_COL].tolist() == [REVIEW_DONE, pd.NA, pd.NA]
    assert out[SOURCE_COL].tolist() == [pd.NA, pd.NA, SOURCE_MANUAL]
    assert mark_reviewed(out, np.array([True, False, False])) == 0


# ================================================================================================
# D-DESIGN-4 — one rule in two shapes, or it is not one rule
# ================================================================================================

def test_a_padded_review_token_reads_the_same_per_row_and_per_frame():
    """One leading space — all a CSV round-trip or a hand-edited project needs — used to
    read ACCEPTED per row (``review_of`` strips) and UNKNOWN in the same table's column
    (the vectorised path compared raw). A table column, a summary count and a per-row
    read-out may not disagree."""
    df = pd.DataFrame({
        "accepted": [True, False],
        "flag": pd.array(["", ""], dtype="string"),
        SOURCE_COL: pd.array(["detect.schmitt", "detect.schmitt"], dtype="string"),
        REVIEW_COL: pd.array([" reviewed", "reviewed "], dtype="string"),
    })
    assert list(event_status(df)) == [status_of(r) for _, r in df.iterrows()]
    assert list(event_status(df)) == [STATUS_ACCEPTED, STATUS_REJECTED]


def test_an_unrecognised_review_token_is_unknown_in_both_shapes():
    """And the normalisation is shared in the other direction too: a spelling this build
    does not know is not evidence that a person ruled, in EITHER reader."""
    df = pd.DataFrame({
        "accepted": [True],
        "flag": pd.array([""], dtype="string"),
        REVIEW_COL: pd.array(["True"], dtype="string"),      # e.g. from a boolean column
    })
    assert status_of(df.iloc[0]) == STATUS_UNKNOWN
    assert list(event_status(df)) == [STATUS_UNKNOWN]


def test_both_readers_refuse_a_frame_that_has_no_accepted_column():
    """Status is a statement ABOUT ``accepted``. The frame reader raised and the row
    reader assumed True, so the same frame was 'accepted' per row and undescribable per
    table; they now refuse identically."""
    df = pd.DataFrame({
        "event_id": [0],
        "flag": pd.array([""], dtype="string"),
        SOURCE_COL: pd.array([SOURCE_MANUAL], dtype="string"),
        REVIEW_COL: pd.array([REVIEW_DONE], dtype="string"),
    })
    with pytest.raises(ValueError, match="accepted"):
        event_status(df)
    with pytest.raises(ValueError, match="accepted"):
        status_of(df.iloc[0])


def test_a_padded_source_cell_is_not_reported_as_a_contradiction():
    """``source_of`` and ``source_kind`` strip; ``source_disagrees_with_added`` did not,
    so ``' manual '`` on a hand-added row was reported as a contradiction that is not
    there — and that function exists to find REAL ones."""
    df = pd.DataFrame({
        SOURCE_COL: pd.array([" manual ", "detect.schmitt"], dtype="string"),
        ADDED_COLUMN: [True, False],
    })
    assert ev.source_of({SOURCE_COL: " manual "}) == SOURCE_MANUAL
    assert ev.source_kind(" manual ") == ev.KIND_MANUAL
    assert source_disagrees_with_added(df).tolist() == [False, False]


# ================================================================================================
# the two recorded claims that had to be true rather than tidy
# ================================================================================================

def test_the_dtype_justification_is_true_of_the_tree_it_ships_in():
    """The contract argues ``string`` over nullable ``boolean``. The reason it gave — a
    merge would raise on a ``boolean`` ``review`` — is FALSE here, because this change
    itself put the name-routing ahead of ``merge``'s bool branch. Re-measured, all four
    facts the rewritten paragraph now rests on:"""
    from nanoscope.analysis.merge import merge_by_gap

    # 1. the pandas fact, unchanged
    nullable = pd.Series([True, pd.NA], dtype="boolean")
    assert pd.api.types.is_bool_dtype(nullable)
    with pytest.raises(ValueError, match="missing values"):
        nullable.to_numpy(dtype=bool)

    rng = np.random.default_rng(3)
    sig = (np.full(20_000, I0) + rng.normal(0.0, 4.0, 20_000)).astype(np.float32)
    for s, e in [(5_000, 5_600), (5_700, 6_300)]:
        sig[s:e] -= 250.0
    sig = np.ascontiguousarray(sig)
    df, prov = detect_events(sig, FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == 2

    # 2. the generic bool carry in ``merge`` STILL raises — the hazard is live for any
    #    other non-schema boolean column with a missing value.
    other = df.copy()
    other["some_extra"] = pd.array([True, pd.NA], dtype="boolean")
    with pytest.raises(ValueError, match="missing values"):
        merge_by_gap(EventSet(other, prov), sig, FS, max_gap_s=0.01)

    # 3. ...but no longer for THESE two columns, and only because they are routed by name
    #    first. So the recorded reason was stale; it is one rename from being true again.
    boolean_review = df.copy()
    boolean_review[SOURCE_COL] = pd.array(["detect.schmitt"] * 2, dtype="string")
    boolean_review[REVIEW_COL] = pd.array([True, pd.NA], dtype="boolean")
    merged, _ = merge_by_gap(EventSet(boolean_review, prov), sig, FS, max_gap_s=0.01)
    assert merged.df[REVIEW_COL].iloc[0] is pd.NA

    # 4. what the dtype actually rests on: the vocabulary is words. A ``boolean`` column
    #    does not fail loudly, it reads ALL-UNKNOWN — the reviewedness is gone, silently.
    assert list(event_status(boolean_review)) == [STATUS_UNKNOWN, STATUS_UNKNOWN]


def test_the_rebuild_door_does_not_claim_callers_it_does_not_have():
    """``carry_source_status`` called itself THE REBUILD DOOR while no production code
    called it — the rebuild paths carry the pair generically instead. Either it is the
    door and the paths go through it, or the sentence goes: this test holds the two
    together, so wiring it up is what unlocks the claim."""
    root = Path(ev.__file__).resolve().parent.parent
    callers = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if p.name != "events.py" and "carry_source_status(" in p.read_text()
    )
    doc = ev.carry_source_status.__doc__ or ""
    disclaimed = 'NOT "THE REBUILD DOOR"' in doc
    if callers:
        assert not disclaimed, (
            f"carry_source_status is now called by {callers}: it IS a rebuild door, and "
            "the docstring still says it is not one."
        )
    else:
        assert disclaimed, (
            "carry_source_status has no production caller, so its docstring must not "
            "claim to be the door every rebuild path goes through."
        )
