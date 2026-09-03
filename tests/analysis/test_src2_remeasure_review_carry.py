"""``review`` is a DESCRIPTION. It may never move a MEASUREMENT, and it may never
throw away another job's answer.

Both defects here come from the same shortcut: ``review`` was added to the existing
curation machinery by widening things that already existed, instead of being given its
own diff and its own snapshot.

FAILING BEFORE -- a review-only click rewrote the frozen-20 ``flag`` cell.
    ``merge_late_curation`` built ONE changed-mask as the union over
    ``accepted``/``flag``/``review``, and every row in that mask entered the loop that
    REBUILDS ``flag`` from tokens. The rebuild is not the identity: on a row nobody
    edited, with the only gesture a "mark reviewed" click elsewhere in the table,
    ``pd.NA -> ''``, ``'short; noisy' -> 'short;noisy'``, ``'short;short' ->
    'short'``, ``'short;' -> 'short'``. The last three move the frozen-20 hash
    outright; the first moves the cell while a CSV hash looks stable, and shows up on
    a project round trip.

    A fresh ``characterize`` answer happens to hold canonical flags, so on the
    engine's own output the damage was LATENT rather than absent -- one non-canonical
    cell away. A CSV re-import gives ``pd.NA`` on every unflagged row (60 of 61 on the
    synthetic detection below) and a hand-typed ``'noise; long tail'`` keeps its
    space; ``merge_late_curation`` is a public function that takes the frame it is
    given. The unit cases below are therefore the repro, and the end-to-end case is
    the guard that says so.

FAILING BEFORE -- the merge-all install guard read the widened snapshot.
    ``analysis_panel._merge_outcome`` drops a finished "Merge all" when
    ``curation_snapshot(live).equals(curation)`` is False, on the ground that a merge
    deletes rows and cannot carry an accept/reject made while it ran. Once the
    snapshot carried ``review``, one "mark reviewed" click during the merge made that
    guard fire and a completed merge was discarded. :func:`merge_guard_snapshot` is
    the narrowed question -- the verdict, which a merge would silently undo -- and
    ``review`` is deliberately outside it, because a merge RE-DERIVES review from its
    fragments rather than overwriting it with a stale copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import remeasure as R
from nanoscope.analysis.baseline import DEFAULT_BASELINE, BaselineConfig
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    REVIEW_COL,
    REVIEW_DONE,
    REVIEW_NONE,
    SCHEMA,
    EventSet,
    stamp_source_status,
)

FS = 50_000.0
OLD = BaselineConfig(method="block_median", window_s=0.01)

#: Every shape the ``flag`` column is actually seen holding. ``pd.NA`` is what a
#: clean detection writes; the rest come from a CSV re-import, a hand-typed flag and
#: a concat of a partial row.
FLAG_SHAPES = ["", pd.NA, "short; noisy", "short;short", "short;", "rejected_by_hand"]


def _frame(n: int = 6, flag=None) -> pd.DataFrame:
    """A minimal table carrying the whole frozen 20 plus ``review``."""
    df = pd.DataFrame({c: np.arange(n, dtype=float) for c in SCHEMA})
    df["event_id"] = np.arange(n, dtype=np.int64)
    df["start_idx"] = np.arange(n, dtype=np.int64) * 100
    df["end_idx"] = df["start_idx"] + 50
    df["n_levels"] = np.ones(n, dtype=np.int64)
    df["filter_limited"] = np.zeros(n, dtype=bool)
    df["accepted"] = np.ones(n, dtype=bool)
    df["recording"] = pd.array(["r"] * n, dtype="string")
    df["pore_id"] = pd.array(["p"] * n, dtype="string")
    df["flag"] = pd.array([flag] * n, dtype="string")
    df[REVIEW_COL] = pd.array([REVIEW_NONE] * n, dtype="string")
    return df[list(SCHEMA) + [REVIEW_COL]]


def _same_cells(a: pd.Series, b: pd.Series) -> bool:
    """Cell for cell, ``pd.NA`` included -- and ``pd.NA`` is NOT ``''``."""
    if list(a.index) != list(b.index) or a.dtype != b.dtype:
        return False
    for x, y in zip(a.to_numpy(dtype=object), b.to_numpy(dtype=object)):
        x_na, y_na = pd.isna(x), pd.isna(y)
        if x_na or y_na:
            if not (x_na and y_na):
                return False
        elif x != y:
            return False
    return True


def _signal(n=120_000, i0=-100.0, depth=60.0, noise=2.0, dwell=250, period=2000,
            seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, i0) + rng.normal(0.0, noise, n)
    t = 500
    while t + dwell < n:
        x[t:t + dwell] -= depth
        t += period
    return x.astype(np.float32)


# ===========================================================================
# 1. A REVIEW-ONLY CLICK MOVES NO MEASURED CELL
# ===========================================================================
@pytest.mark.parametrize("flag", FLAG_SHAPES)
def test_a_review_only_click_leaves_the_flag_cell_exactly_as_measured(flag):
    """FAILING BEFORE, on four of the six shapes. The curator's only gesture is
    "mark reviewed" on row 2; ``flag`` is a frozen-20 column and nobody touched it.
    """
    before = _frame(flag=flag)
    live = before.copy(deep=True)
    live.loc[2, REVIEW_COL] = REVIEW_DONE
    out = before.copy(deep=True)

    merged = R.merge_late_curation(out.copy(deep=True), live, before)

    assert _same_cells(merged["flag"], out["flag"]), (
        f"a review click rewrote flag: {out['flag'].iloc[2]!r} -> "
        f"{merged['flag'].iloc[2]!r}")
    assert merged["accepted"].equals(out["accepted"])
    assert merged.loc[2, REVIEW_COL] == REVIEW_DONE
    assert str(merged[REVIEW_COL].dtype) == "string"


def test_a_review_only_click_moves_none_of_the_frozen_twenty_on_a_real_measurement():
    """PASSES BEFORE AND AFTER, and is kept for exactly that reason -- see the module
    docstring. End to end through a real detection and a real re-measure, the
    engine's own ``flag`` cells are canonical, so the pre-fix rebuild happened to be
    the identity and no number moved. This pins the whole-table invariant (all 20
    measured columns, byte for byte) so that a future caller handing
    ``merge_late_curation`` a re-imported or hand-flagged frame cannot quietly turn
    the latent case back into the live one.
    """
    x = _signal()
    df, prov = detect_events(x, FS, baseline_config=OLD, polarity=-1,
                             min_dwell_s=1e-4, recording="rec", pore_id="pore")
    # What `modules.DetectSchmitt.run` stamps on the way out of a real detection.
    ev = EventSet(stamp_source_status(df, "detect.schmitt"), prov)
    assert len(ev.df) > 5 and REVIEW_COL in ev.df.columns

    before = R.curation_snapshot(ev.df)
    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE,
                             polarity=-1).events.df
    live = ev.df.copy(deep=True)
    live.loc[live.index[3], REVIEW_COL] = REVIEW_DONE

    merged = R.merge_late_curation(out.copy(deep=True), live, before)

    moved = [c for c in SCHEMA if not _same_cells(merged[c], out[c])]
    assert moved == [], f"a review click moved measured columns: {moved}"
    assert merged[REVIEW_COL].iloc[3] == REVIEW_DONE


def test_the_review_carry_itself_survives_the_narrowing():
    """The control for the fix: narrowing the diff must not throw the carry away.
    A review-only change still returns a NEW frame with the click in it -- it is not
    the untouched-table fast path.
    """
    before = _frame(flag="")
    live = before.copy(deep=True)
    live.loc[1, REVIEW_COL] = REVIEW_DONE
    live.loc[4, REVIEW_COL] = pd.NA
    out = before.copy(deep=True)

    merged = R.merge_late_curation(out, live, before)

    assert merged is not out
    assert merged[REVIEW_COL].iloc[1] == REVIEW_DONE
    assert pd.isna(merged[REVIEW_COL].iloc[4])
    assert merged[REVIEW_COL].iloc[0] == REVIEW_NONE


def test_a_review_click_beside_a_real_flag_edit_carries_both_and_confuses_neither():
    """Two rows, two different gestures in the same three seconds: row 1 is marked
    reviewed, row 3 gets a hand flag. The flag edit must land on row 3 ONLY.
    """
    before = _frame(flag=pd.NA)
    live = before.copy(deep=True)
    live.loc[1, REVIEW_COL] = REVIEW_DONE
    live.loc[3, "flag"] = "noise"
    out = before.copy(deep=True)

    merged = R.merge_late_curation(out.copy(deep=True), live, before)

    assert merged["flag"].iloc[3] == "noise"
    assert merged[REVIEW_COL].iloc[1] == REVIEW_DONE
    untouched = [i for i in range(len(before)) if i != 3]
    assert all(pd.isna(merged["flag"].iloc[i]) for i in untouched), (
        "the flag rebuild reached a row whose only change was review")


def test_an_untouched_table_is_still_returned_unchanged():
    """Control. No gesture at all -- neither mask fires, and ``out`` comes back as
    itself rather than as a copy.
    """
    before = _frame(flag="short;")
    out = before.copy(deep=True)
    assert R.merge_late_curation(out, before.copy(deep=True), before) is out


# ===========================================================================
# 2. THE MERGE-ALL INSTALL GUARD ASKS THE NARROWER QUESTION
# ===========================================================================
def test_the_merge_guard_snapshot_holds_the_verdict_and_not_the_review():
    """FAILING BEFORE -- there was one snapshot for two different questions."""
    df = _frame(flag="")
    snap = R.merge_guard_snapshot(df)

    assert list(snap.columns) == ["event_id", "accepted", "flag"]
    assert REVIEW_COL not in snap.columns
    # ...while the re-measure's snapshot, a different question, still carries it.
    assert REVIEW_COL in R.curation_snapshot(df).columns


def test_a_mark_reviewed_click_during_a_merge_is_not_read_as_curation():
    """FAILING BEFORE. One review click during a "Merge all" made the install guard
    read "the user curated while the job ran" and threw the finished merge away.
    """
    df = _frame(flag="")
    snap = R.merge_guard_snapshot(df)

    live = df.copy(deep=True)
    live.loc[2, REVIEW_COL] = REVIEW_DONE
    assert R.merge_guard_snapshot(live).equals(snap), (
        "a review click still looks like curation to the merge guard")


@pytest.mark.parametrize("col, value", [("accepted", False), ("flag", "noise")])
def test_the_verdict_the_guard_protects_still_drops_the_merge(col, value):
    """The other control. Narrowing must not blind the guard to what it exists for:
    a merge deletes rows, so an accept/reject or a hand flag made while it ran has no
    honest row to be carried onto and the answer must still be dropped.
    """
    df = _frame(flag="")
    snap = R.merge_guard_snapshot(df)
    live = df.copy(deep=True)
    live.loc[2, col] = value
    assert not R.merge_guard_snapshot(live).equals(snap)


def test_the_merge_guard_snapshot_reads_a_table_that_predates_review():
    """A project saved before ``review`` existed still loads, so every read of the
    column is guarded on presence -- this one by construction.
    """
    legacy = _frame(flag="").drop(columns=[REVIEW_COL])
    snap = R.merge_guard_snapshot(legacy)
    assert list(snap.columns) == ["event_id", "accepted", "flag"]
    assert snap.equals(R.merge_guard_snapshot(legacy.copy(deep=True)))
