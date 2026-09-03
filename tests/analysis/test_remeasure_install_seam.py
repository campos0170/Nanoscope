"""The re-measure DISPATCH/INSTALL seam: one seam, two defects, one sitting.

A re-measure is dispatched against one frame, runs for 3-4 s off the GUI thread, and
is installed against another. Everything here is about that gap.

FAILING BEFORE -- the job read a frame that moved under it.
    ``_launch_remeasure`` handed the worker the LIVE ``EventSet`` with no copy, so an
    Event Inspector mask drag ~1.5 s into an 800-event job rewrote ``start_idx`` /
    ``end_idx`` while ``for k, i in enumerate(order)`` was still reading rows out of
    it -- and ``_assert_geometry_held``, an assertion about the ENGINE, then compared
    its output against an input that had changed and fired at the user. The whole job
    was discarded and the status bar was left reading "Re-measuring 800 events…".

FAILING BEFORE -- the install reconciled by rule instead of by diff.
    The only precondition was ``np.array_equal(live_ids, before_ids)``. Measured on
    the owner's own file: event 10 dragged (144689, 145934) -> (144709, 145914),
    Re-measure pressed, Ctrl+Z during the 4.15 s job. The table correctly returned to
    24.900 ms; the install put 24.100 ms back, under the sentence "Detection geometry
    and curation unchanged." Synthetic magnitude: 12.000 ms / 300.19 pA on screen ->
    24.000 ms / 150.62 pA installed -- 100% dwell error, 2.0x depth error.

FAILING BEFORE -- ``merge_late_curation`` was monotone in both curation columns.
    It carried curation through ``inherited_curation``, which is a REBUILD's rule
    (union the flags, AND the acceptance) and is therefore structurally incapable of
    expressing a retraction. Every LOOSENING edit made during the job was discarded:
    a rejection retracted, a flag cleared, a flag cleared on a row that had never been
    rejected at all. One realistic click -- 40 rows hand-rejected earlier, "Accept
    all" pressed during a 3.3 s job -- showed 361 accepted on screen and installed
    321, with every accepted-gated number reporting 321.

    The tested half was exactly the half that worked
    (``tests/analysis/test_remeasure.py:690`` exercises a late *reject* and a late
    flag *add*). Both are still pinned below, marked as controls.

THE PRECONDITION A READER OF THIS FILE NEEDS. Write the retraction test with
``zero_baseline`` as the pre-job rejection and it passes with the defect in place:
``recharacterize_span`` used to re-derive that row to accepted, so the late accept
appeared to work. ``test_the_zero_baseline_variant_is_not_a_repro`` is that case,
kept and labelled so nobody re-derives it and concludes the report was wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.baseline import DEFAULT_BASELINE, BaselineConfig
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import SCHEMA, EventSet, flag_tokens

FS = 50_000.0

#: The estimator every pre-change row was measured by.
OLD = BaselineConfig(method="block_median", window_s=0.01)


def _signal(n=200_000, i0=-100.0, depth=60.0, noise=2.0, dwell=250, period=2000,
            seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, i0) + rng.normal(0.0, noise, n)
    t = 500
    while t + dwell < n:
        x[t:t + dwell] -= depth
        t += period
    return x.astype(np.float32)


def _detected(x, config=OLD) -> EventSet:
    df, prov = detect_events(x, FS, baseline_config=config, polarity=-1,
                             min_dwell_s=1e-4, recording="rec", pore_id="pore")
    return EventSet(df, prov)


def _with_curation(ev: EventSet, row: int, flag: str, accepted: bool) -> EventSet:
    df = ev.df.copy()
    df.loc[row, "flag"] = flag
    df.loc[row, "accepted"] = accepted
    return EventSet(df, dict(ev.provenance))


def _dispatch(ev: EventSet, x: np.ndarray, config=DEFAULT_BASELINE):
    """What the GUI does at dispatch: snapshot the curation, run the job."""
    from nanoscope.analysis import remeasure as R

    return R.curation_snapshot(ev.df), R.remeasure_events(
        ev, x, FS, config=config, polarity=-1)


# ===========================================================================
# 1. THE FRAME THE JOB MEASURES IS FROZEN AT DISPATCH
# ===========================================================================
def test_a_table_edited_under_the_running_job_does_not_reach_the_measurement():
    """FAILING BEFORE. The engine held the caller's frame, so a mask drag landing
    mid-loop measured half the table at the old span and half at the new -- and then
    the engine's own self-check fired at the user, discarding the whole job.

    The drag here is fired from the progress callback, which is exactly where the GUI
    thread gets in: ``progress`` is called from inside the row loop.
    """
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    frame = ev.df
    was = (int(frame.at[0, "start_idx"]), int(frame.at[0, "end_idx"]))
    fired = {"n": 0}

    def drag(frac, _msg):
        if 0.2 < frac < 0.9 and not fired["n"]:
            fired["n"] = 1
            frame.at[0, "start_idx"] = was[0] - 200
            frame.at[0, "end_idx"] = was[1] + 200

    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1,
                             progress=drag)
    assert fired["n"], "the drag never fired; this test proves nothing"
    # The caller's frame really did move...
    assert (int(frame.at[0, "start_idx"]), int(frame.at[0, "end_idx"])) != was
    # ...and the answer is a measurement of the spans as they were at dispatch.
    assert (int(out.events.df.at[0, "start_idx"]),
            int(out.events.df.at[0, "end_idx"])) == was


def test_the_answer_carries_the_dispatch_time_geometry_for_every_row():
    """The freeze is whole-table, not row 0. Every row the loop had not reached yet
    must also come back at its dispatch-time span, or half the table is measured in
    one world and half in another with nothing saying which."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    frame = ev.df
    at_dispatch = frame[list(R.GEOMETRY_COLUMNS)].copy(deep=True)
    fired = {"n": 0}

    def drag(frac, _msg):
        if frac > 0.2 and not fired["n"]:
            fired["n"] = 1
            frame["start_idx"] = frame["start_idx"].to_numpy() - 7
            frame["end_idx"] = frame["end_idx"].to_numpy() + 7

    out = R.remeasure_events(ev, x, FS, config=DEFAULT_BASELINE, polarity=-1,
                             progress=drag)
    assert fired["n"]
    for col in R.GEOMETRY_COLUMNS:
        assert np.array_equal(out.events.df[col].to_numpy(),
                              at_dispatch[col].to_numpy()), col


# ===========================================================================
# 2. THE INSTALL COMPARES WHAT WAS MEASURED AGAINST WHAT THE STORE HOLDS
# ===========================================================================
def test_a_span_the_user_undid_is_refused_not_reinstalled():
    """FAILING BEFORE, and this is C-4's own sequence. The user widens a mask, presses
    Re-measure, then Ctrl+Z while it runs. The store is back at the original span; the
    answer measured the widened one. Installing it puts the undone edit back."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    row = 2
    widened = ev.df.copy()
    widened.loc[row, "start_idx"] = int(widened.loc[row, "start_idx"]) - 60
    widened.loc[row, "end_idx"] = int(widened.loc[row, "end_idx"]) + 60
    widened.loc[row, "t_start_s"] = int(widened.loc[row, "start_idx"]) / FS
    widened.loc[row, "dwell_s"] = (int(widened.loc[row, "end_idx"])
                                   - int(widened.loc[row, "start_idx"])) / FS
    before, out = _dispatch(EventSet(widened, dict(ev.provenance)), x)

    # Ctrl+Z: the store is back at the original geometry.
    live = ev.df
    measured_ms = float(out.events.df.loc[row, "dwell_s"]) * 1e3
    on_screen_ms = float(live.loc[row, "dwell_s"]) * 1e3
    assert abs(measured_ms - on_screen_ms) > 1.0, "the undo did not move the dwell"

    with pytest.raises(ValueError, match="geometry changed while the re-measure"):
        R.merge_late_curation(out.events.df, live, before)


def test_the_refusal_names_the_column_and_the_row_it_found():
    """A refusal the user cannot act on is a refusal he learns to click past. The
    sentence goes to the status bar verbatim."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    before, out = _dispatch(ev, x)
    live = ev.df.copy()
    live.loc[4, "end_idx"] = int(live.loc[4, "end_idx"]) + 33

    with pytest.raises(ValueError) as exc:
        R.merge_late_curation(out.events.df, live, before)
    msg = str(exc.value)
    assert "end_idx" in msg and "row 4" in msg
    assert "no longer holds" in msg


def test_every_geometry_column_is_checked_not_only_the_indices():
    """``t_start_s`` and ``dwell_s`` are geometry too -- a re-measure at a different
    rate moves them without touching an index, and the install would have installed
    a table whose times describe a different recording."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    before, out = _dispatch(ev, x)
    for col in R.GEOMETRY_COLUMNS:
        live = ev.df.copy()
        live.loc[3, col] = live.loc[3, col] + (1 if col.endswith("_idx") else 1e-3)
        with pytest.raises(ValueError, match="geometry changed"):
            R.merge_late_curation(out.events.df, live, before)


def test_a_reordered_table_is_refused_by_the_id_check():
    """A CONTROL that passed before and passes now: the row-identity half of the
    precondition already worked, and widening it to geometry must not weaken it."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    before, out = _dispatch(ev, x)
    live = ev.df.drop(index=3).reset_index(drop=True)
    with pytest.raises(ValueError, match="changed while the re-measure was running"):
        R.merge_late_curation(out.events.df, live, before)


def test_an_untouched_table_installs_unchanged():
    """A CONTROL, and the one the widened precondition could plausibly have broken:
    on the ordinary workflow nothing has moved, so nothing may be refused and the
    result object itself comes back."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    before, out = _dispatch(ev, x)
    assert R.merge_late_curation(out.events.df, ev.df, before) is out.events.df


# ===========================================================================
# 3. THE CURATION CARRY IS A PER-COLUMN DIFF, IN BOTH DIRECTIONS
# ===========================================================================
@pytest.mark.parametrize(
    "name,pre_flag,pre_acc,live_flag,live_acc",
    [
        ("retract a bare hand rejection", "", False, "", True),
        ("retract a rejection carrying a note", "noise", False, "", True),
        ("clear a note on an accepted row", "noise", True, "", True),
        ("retract but keep the note", "noise", False, "noise", True),
    ],
)
def test_a_loosening_edit_made_during_the_job_survives_the_install(
        name, pre_flag, pre_acc, live_flag, live_acc):
    """FAILING BEFORE, all four. ``merge_late_curation`` could only ADD flag tokens
    and only REMOVE acceptance, so every one of these was silently discarded and the
    installed table disagreed with the screen."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    row = 3
    pre = _with_curation(_detected(x), row, pre_flag, pre_acc)
    before, out = _dispatch(pre, x)

    live = pre.df.copy()
    live.loc[row, "flag"] = live_flag
    live.loc[row, "accepted"] = live_acc

    merged = R.merge_late_curation(out.events.df, live, before)
    assert bool(merged.loc[row, "accepted"]) is live_acc, name
    assert set(flag_tokens(merged.loc[row, "flag"])) == set(flag_tokens(live_flag)), name
    # ...and it is still the RE-MEASURED row, not the live table's numbers.
    assert np.array_equal(merged["depth_pA"].to_numpy(),
                          out.events.df["depth_pA"].to_numpy())
    assert len(ev.df) == len(merged)


def test_accept_all_pressed_during_the_job_does_not_revert_forty_rejections():
    """FAILING BEFORE, and this is the click that costs a number. 40 rows hand-rejected
    earlier, "Accept all" pressed during the job: the screen said 361 accepted and the
    installed table said 321, so the summary, the exports and the fits all reported
    321 with nothing on screen to say so."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x)
    # Rows the DETECTOR accepted with no reason of its own: a rejection on one of
    # these can only be a person's, which is what makes the retraction a person's too.
    clean = [i for i in ev.df.index
             if bool(ev.df.loc[i, "accepted"]) and not flag_tokens(ev.df.loc[i, "flag"])]
    rejected = clean[:40]
    assert len(rejected) >= 20, "fixture too small to make the point"
    df = ev.df.copy()
    df.loc[rejected, "accepted"] = False
    pre = EventSet(df, dict(ev.provenance))
    before, out = _dispatch(pre, x)

    live = pre.df.copy()
    live["accepted"] = True                    # "Accept all", mid-job
    on_screen = int(live["accepted"].sum())

    merged = R.merge_late_curation(out.events.df, live, before)
    # The answer alone carries all 40 rejections forward -- that is the loss.
    assert sum(1 for i in rejected
               if not bool(out.events.df.loc[i, "accepted"])) == len(rejected)
    # ...and the install must show what the screen showed.
    assert all(bool(merged.loc[i, "accepted"]) for i in rejected)
    assert int(merged["accepted"].sum()) == int(out.events.df["accepted"].sum()) \
        + len(rejected)
    assert on_screen - int(merged["accepted"].sum()) == int((~out.events.df[
        "accepted"].to_numpy()).sum()) - len(rejected)


@pytest.mark.parametrize(
    "name,pre_flag,pre_acc,live_flag,live_acc",
    [
        ("a late rejection", "", True, "", False),
        ("a late flag added", "", True, "late_note", True),
    ],
)
def test_a_tightening_edit_still_carries(name, pre_flag, pre_acc, live_flag, live_acc):
    """CONTROLS that passed before and pass now -- the half that already worked, and
    the half the old test file covered. A diff carry that lost these would have traded
    one direction for the other."""
    from nanoscope.analysis import remeasure as R

    x = _signal()
    row = 3
    pre = _with_curation(_detected(x), row, pre_flag, pre_acc)
    before, out = _dispatch(pre, x)
    live = pre.df.copy()
    live.loc[row, "flag"] = live_flag
    live.loc[row, "accepted"] = live_acc

    merged = R.merge_late_curation(out.events.df, live, before)
    assert bool(merged.loc[row, "accepted"]) is live_acc, name
    assert set(flag_tokens(merged.loc[row, "flag"])) == set(flag_tokens(live_flag)), name


def test_a_derived_token_is_never_carried_in_either_direction():
    """PASSES BOTH BEFORE AND AFTER, and it is the invariant the rewrite had to buy
    the retraction WITHOUT spending: ``out`` has just re-derived its automatic tokens
    against the corrected baseline, so neither half of a stale one may travel. The old
    code got this right for the reason it got the retraction wrong (a union cannot
    carry a derived token it excludes); the diff carry has to get it right on
    purpose."""
    from nanoscope.analysis import remeasure as R
    from nanoscope.analysis.events import FLAG_NON_FINITE

    x = _signal()
    row = 3
    pre = _with_curation(_detected(x), row, FLAG_NON_FINITE, False)
    before, out = _dispatch(pre, x)
    # The engine re-derived it: this row is finite under the new estimator.
    assert FLAG_NON_FINITE not in flag_tokens(out.events.df.loc[row, "flag"])

    # ADD direction: a derived token appearing live is not a curator's decision.
    live = pre.df.copy()
    live.loc[row, "flag"] = ""
    live.loc[row, "accepted"] = True
    merged = R.merge_late_curation(out.events.df, live, before)
    assert FLAG_NON_FINITE not in flag_tokens(merged.loc[row, "flag"])

    live2 = _detected(x).df.copy()
    live2.loc[row, "flag"] = FLAG_NON_FINITE
    live2.loc[row, "accepted"] = False
    before2, out2 = _dispatch(_detected(x), x)
    merged2 = R.merge_late_curation(out2.events.df, live2, before2)
    assert FLAG_NON_FINITE not in flag_tokens(merged2.loc[row, "flag"])
    assert bool(merged2.loc[row, "accepted"]) is True


def test_a_late_accept_cannot_un_reject_a_row_the_re_measure_found_unmeasurable():
    """PASSES BOTH BEFORE AND AFTER -- the ceiling the diff carry must not break
    through. The carry is a DIFF, not an override: ``characterize`` ANDs the automatic
    rules over any inherited acceptance, and a click made before the measurement
    existed is not evidence about it, so a row the re-measure has just flagged
    ``non_finite`` stays rejected exactly as it would if the click had landed a second
    later. The old monotone rule could not violate this; a diff carry written as a
    verbatim copy of ``live`` would."""
    from nanoscope.analysis import remeasure as R
    from nanoscope.analysis.events import FLAG_NON_FINITE, has_flag

    x = _signal().astype(np.float64)
    ev = _detected(x)
    row = 3
    lo = int(ev.df.loc[row, "start_idx"])
    x[lo + 3] = np.inf                       # poison this row only
    pre = _with_curation(ev, row, "", False)
    before, out = _dispatch(pre, x)
    assert has_flag(out.events.df.loc[row, "flag"], FLAG_NON_FINITE)
    assert not bool(out.events.df.loc[row, "accepted"])

    live = pre.df.copy()
    live.loc[row, "accepted"] = True         # the retraction, mid-job
    merged = R.merge_late_curation(out.events.df, live, before)
    assert not bool(merged.loc[row, "accepted"])
    assert has_flag(merged.loc[row, "flag"], FLAG_NON_FINITE)


def test_the_zero_baseline_variant_is_not_a_repro():
    """PASSES BOTH BEFORE AND AFTER, deliberately, and it is here as a warning.

    If the pre-job rejection was purely automatic -- a ``DERIVED_FLAGS`` token --
    ``recharacterize_span`` re-derived the row to accepted and the late accept
    appeared to work even with the defect fully in place. Anyone who writes the
    regression test this way sees green and concludes the report was wrong. The
    retraction cases above all use a HAND rejection for that reason.
    """
    from nanoscope.analysis import remeasure as R
    from nanoscope.analysis.events import FLAG_ZERO_BASELINE

    x = _signal()
    row = 3
    pre = _with_curation(_detected(x), row, FLAG_ZERO_BASELINE, False)
    before, out = _dispatch(pre, x)
    live = pre.df.copy()
    live.loc[row, "accepted"] = True
    live.loc[row, "flag"] = ""
    merged = R.merge_late_curation(out.events.df, live, before)
    assert bool(merged.loc[row, "accepted"])


# ===========================================================================
# 4. WHAT A RE-MEASURE MOVES, THROUGH THE SEAM THAT INSTALLS IT
# ===========================================================================
def test_a_same_config_install_is_bit_identical_on_all_twenty_columns():
    """PASSES BOTH BEFORE AND AFTER, deliberately: this is the NON-REGRESSION the
    freeze and the widened precondition could plausibly have broken, not a repro.

    The invariant this seam could break without anyone noticing, and it HAS been
    broken once by ``rise_s``/``fall_s``, so it is asserted column by column rather
    than on a summary. Re-measure against the estimator the rows were already
    measured by, install it, and every one of the 20 columns must come back
    bit-identical -- including the two the release note once called unaffected."""
    from nanoscope.analysis import remeasure as R

    # `clustered`-style: neighbours inside the +-64-sample rise/fall pad, which is
    # the only geometry in which the rise/fall bound can degenerate.
    rng = np.random.default_rng(7)
    x = np.full(200_000, -100.0) + rng.normal(0.0, 2.0, 200_000)
    t = 500
    while t + 4 * 210 < 200_000:
        u = t
        for _ in range(4):
            x[u:u + 150] -= 60.0
            u += 210
        t += 3000
    x = x.astype(np.float32)

    ev = _detected(x, OLD)
    stored = BaselineConfig.reproducible_from(ev.provenance)
    assert stored is not None
    before, out = _dispatch(ev, x, config=stored)
    merged = R.merge_late_curation(out.events.df, ev.df, before)

    assert list(merged.columns) == list(ev.df.columns)
    for col in SCHEMA:
        a, b = ev.df[col].to_numpy(), merged[col].to_numpy()
        if a.dtype.kind == "f":
            assert np.array_equal(a, b, equal_nan=True), col
        else:
            assert np.array_equal(a, b), col
    assert "rise_s" in SCHEMA and "fall_s" in SCHEMA


def test_a_changed_config_moves_the_amplitudes_and_nothing_else():
    """PASSES BOTH BEFORE AND AFTER -- the other half of the same non-regression.

    Against a DIFFERENT estimator the six
    amplitude columns must all move, and nothing outside ``REMEASURED_COLUMNS`` may:
    not the geometry, not the ids, not the curation.

    ``n_levels`` is the one exception, and it is not this seam's doing: it is counted
    by ``characterize`` from the event's OWN depth, so an open-pore reference that
    moves 4.7x re-counts the sub-levels beneath it. Measured identical on the
    unmodified engine. It is called out here rather than folded into the set because
    it is NOT in ``REMEASURED_COLUMNS`` and therefore is not among the columns the
    results table badges as stale -- a true statement about the current app that a
    future reader should not mistake for this test being loose.
    """
    from nanoscope.analysis import remeasure as R

    x = _signal()
    ev = _detected(x, OLD)
    df = ev.df.copy()
    df.loc[2, "accepted"] = False
    df.loc[4, "flag"] = "curator_note"
    pre = EventSet(df, dict(ev.provenance))
    before, out = _dispatch(pre, x)
    merged = R.merge_late_curation(out.events.df, pre.df, before)

    moved = set()
    for col in SCHEMA:
        a, b = pre.df[col].to_numpy(), merged[col].to_numpy()
        same = (np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f"
                else np.array_equal(a, b))
        if not same:
            moved.add(col)
    assert set(R.AFFECTED_COLUMNS) <= moved, sorted(moved)
    assert moved <= set(R.REMEASURED_COLUMNS) | {"n_levels"}, \
        sorted(moved - set(R.REMEASURED_COLUMNS) - {"n_levels"})
    assert not (moved & set(R.GEOMETRY_COLUMNS))
    assert "accepted" not in moved and "flag" not in moved and "event_id" not in moved
    assert not bool(merged.loc[2, "accepted"])
    assert "curator_note" in flag_tokens(merged.loc[4, "flag"])
