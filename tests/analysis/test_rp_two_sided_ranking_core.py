"""RP -- the ladder looks DOWN as well as up, and the screen stops claiming otherwise.

THE DEFECT
==========
`baseline_probe.candidate_windows` walked the grid strictly upward from the proposal
(``if rough <= first[0]: continue``), so a SHORTER window was never estimated. But
`baseline_probe.rank_windows`' own documented objective is "lowest cost, ties broken by
the SHORTEST window", and on the -600 mV apo recording those two things disagree
outright:

    ==============  =====================  =========================
    -600 mV window  ``baseline_intrusion``  estimated by the old search?
    ==============  =====================  =========================
    0.3 / 100 ms    0.002888               no -- shorter
    0.5 / 100 ms    0.003155               no -- shorter
    **1 / 300 ms**  **0.0**                **no -- shorter, and it WINS**
    2 / 500 ms      0.0  (the proposal)    yes, and it was chosen
    ==============  =====================  =========================

...and then the screen said **"Nothing can beat a baseline that follows none of the
states, so no wider window was estimated."** The second clause is true. The first is
false under the app's own objective: what beats a clean window is a SHORTER clean one,
on the tie-break, and the search had never looked there. The same sentence is printed
on -500 mV, where it is equally untested and happens to be right.

WHAT IS ASSERTED HERE
=====================
The invariant: **the app may not claim an optimality it did not test.** Either it
tests the shorter side, or it says it did not. Five hunks carry that and every test
below names the one it fires on:

  A  `candidate_windows`   -- two-sided, floored below by `base_floor_s`
  B  `rank_windows`        -- the two-phase walk, the descent gate, the cap
  C  `_ranking_reason`     -- what was estimated, what was not, and the supersession
  D  `propose_baseline`    -- the rule's numbers labelled as the RULE's
  E  `_rank_declined`      -- the decline sentences, still true of a two-sided ladder

THE STEP PAST THE OBVIOUS CASE
==============================
The RK suite's unit tests all ran on one synthetic whose proposal INTRUDES, which is
the only shape at which the descent cannot fire at all: phase 2 is gated on the
proposal reading zero. So this file's fixtures are chosen for the k that shows it --
`CLOGGED` has a clean proposal AND a rung below it that ties on the median (the
-600 mV shape), `DEEP` has four such clogs, and `RISING` has an intruding proposal
so the descent must NOT run. A fixture on which the bug cannot appear proves nothing,
and one of those shipped last batch.

AMENDED BY THE BL BATCH (the worst-state criterion), and why that was owed
==========================================================================
The two-sided ladder shipped by THIS batch moved -600 mV to 1 s / 300 ms on a tie the
median cannot judge: that window's tracked baseline rides into the recording's 521 ms
clog (56% of the one state's own excursion followed, median exactly 0.0 -- one state
among 239 cannot move a median), the clog stops being detected at all and ~30 shallow
events are invented in its place. `rank_windows` now reads the WORST single state
beside the median (`baseline_probe.WORST_STATE_INTRUSION_MAX`), a rung that rides
loses to one that stays out, and -600 mV keeps the rule's own 2 s / 500 ms. Several
tests below asserted the ridden window as the right answer, and one asserted the
sentence "nothing WIDER can beat a baseline that follows none of the states" -- a
false general claim (-600 mV's wider window IS better; "follows none" was only the
median). Those tests are amended here to the two-criteria objective; the mechanism
and the corpus evidence live in ``tests/analysis/test_bl_worststate_ranking.py``.

Nothing here asserts a screen; ``tests/gui/test_rp_two_sided_ranking_wired.py`` does.
"""
from __future__ import annotations

import os
import pathlib
import time

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope

FS = 50_000.0
CORPUS = pathlib.Path("/Users/abrahamcampos/Desktop/Research-Data/FerritinModeler/CSVData")
APO = {
    -500: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0008_process data_-500 mV.abf"),
    -600: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0006_Process_-600 mV.abf"),
    -700: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0005_Process_-700 mV.abf"),
}

#: What the app OFFERS on each recording with the two-sided ladder AND the worst-state
#: criterion. -600 mV's descent runs (its proposal is clean) and estimates 1 s /
#: 300 ms, which ties on the median -- and rides into the 521 ms clog (worst state
#: 0.556 > 0.5), so it loses and the rule's own window stands.
EXPECTED = {
    "apo-500": (1.0, 0.3),            # unchanged -- 0.5 / 0.1 IS estimated, at 0.006467
    "apo-600": (2.0, 0.5),            # 1.0 / 0.3 ties on the median but RIDES (0.556)
    "apo-700": (1.0, 0.3),            # unchanged -- proposal intrudes, no descent
    "ferritin_data1": (1.0, 0.3),     # unchanged -- 1.0 / 0.3 has no state that long
    "ferritin_data2": (2.0, 0.5),     # 1.0 / 0.3 RIDES a real 395 ms state (0.782)
    "ferritin_data3": (0.03, 0.01),   # gating: not ranked, and must stay unranked
}

#: How many windows the search ESTIMATES per recording. The cost of the change, and
#: the reason the two stop rules are gated the way they are: phase 2 runs only when
#: phase 1 stopped after one estimate, so the two never both spend. -600 mV spends 2:
#: its descent's FIRST rung rides and stops it.
ESTIMATES = {
    "apo-500": 2, "apo-600": 2, "apo-700": 2,
    "ferritin_data1": 2, "ferritin_data2": 3, "ferritin_data3": 0,
}

_CACHE: dict = {}

#: The sentence this whole batch is about. It must not be on any screen again.
FALSE_CLAIM = ("Nothing can beat a baseline that follows none of the states, so no "
               "wider window was estimated.")


def api(obj, name):
    """``getattr`` as an ASSERTION, so a package without the fix fails on the CLAIM
    rather than on an AttributeError that reads like a broken test."""
    assert hasattr(obj, name), (
        f"{getattr(obj, '__name__', type(obj).__name__)} has no {name!r}: the search "
        "does not look below the proposal, which is the defect this file is about")
    return getattr(obj, name)


def where_from():
    """Print AND CHECK the package under test.

    A revert run from inside the repo silently tests the LIVE tree -- PYTHONPATH does
    not beat cwd when cwd holds the package. The revert runner sets
    ``NANOSCOPE_EXPECT_TREE``; this fails rather than reporting a green revert that
    was never actually run against the reverted code.
    """
    import nanoscope
    p = pathlib.Path(nanoscope.__file__).resolve()
    print(f"\nnanoscope resolved from: {p}")
    want = os.environ.get("NANOSCOPE_EXPECT_TREE")
    if want:
        root = pathlib.Path(want).resolve()
        assert str(p).startswith(str(root)), (
            f"nanoscope resolved from {p}, not from {root}: this run is testing a "
            "different tree than it was pointed at")
    return p


def _apo(mv):
    p = APO[mv]
    if not p.exists():
        pytest.skip(f"recording absent: {p}")
    key = f"apo{mv}"
    if key not in _CACHE:
        from nanoscope.io.base import open_document
        doc = open_document(p)
        _CACHE[key] = (np.asarray(doc.channels[2], np.float64),
                       float(doc.meta.sample_rate_hz))
    return _CACHE[key]


def _corpus(tag):
    import pandas as pd
    from nanoscope.analysis import ml_preprocess as mp
    csv = CORPUS / f"{tag}.csv.gz"
    if not csv.exists():
        pytest.skip(f"corpus absent: {csv}")
    if tag not in _CACHE:
        raw = np.asarray(pd.read_csv(csv, usecols=["current_raw_pA"])["current_raw_pA"],
                         np.float32)
        _CACHE[tag] = (np.asarray(mp.notch_sequential(raw, FS), np.float64), FS)
    return _CACHE[tag]


def load(tag):
    return _apo(int(tag[3:])) if tag.startswith("apo") else _corpus(tag)


# ===========================================================================
# fixtures -- and WHY each one is shaped the way it is
# ===========================================================================
def telegraph(dwell_s, duty, *, dur=60.0, seed=11, depth=110.0, i0=-34.8,
              noise=4.84, shape=4.0, fs=FS, clogs=()):
    """A two-level pore, optionally with long partial-clog states in it.

    ``clogs`` is what makes these fixtures different from the RK file's: a handful of
    ~600 ms states is what pushes ``longest_state_s`` (the p99.9 of the closed dwells)
    up far enough that the rule proposes a 2 s rough window, which is the only shape
    on which a rung BELOW the proposal can be the one that wins. One long state is not
    enough -- p99.9 of ~300 dwells interpolates between the top two, so a single 600 ms
    state reads as ~290 ms and the rule proposes 1 s.

    Arrival times are exponential for the reason the RK file's generator gives: a
    periodic event train beats against the probe's 20 ms gating block grid and scores
    the fixture as a stepping pore.
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    x = np.full(n, float(i0))
    t = 0.0
    gap = dwell_s * (1.0 - duty) / duty
    while True:
        t += rng.exponential(gap)
        d = rng.gamma(shape, dwell_s / shape)
        s, e = int(round(t * fs)), int(round((t + d) * fs))
        if e >= n:
            break
        x[s:e] -= depth
        t += d
    for start_s, dur_s, dep in clogs:
        a = int(start_s * fs)
        x[a:a + int(dur_s * fs)] -= dep
    x += rng.normal(0.0, noise, n)
    return x.astype(np.float32)


CLOGS_3 = ((12.0, 0.60, 40.0), (31.0, 0.62, 40.0), (48.0, 0.58, 40.0))
CLOGS_4 = ((10.0, 0.60, 40.0), (25.0, 0.62, 40.0), (40.0, 0.58, 40.0),
           (52.0, 0.61, 40.0))

#: THE -600 mV SHAPE: the rule proposes 2 s / 500 ms, that window reads a clean zero
#: on BOTH criteria, and 1 s / 300 ms one rung below it ties on the MEDIAN -- while
#: riding into the ~600 ms clogs (worst state 0.966). The upward-only search never
#: estimated the shorter rung; the two-sided-with-guard search estimates it, and the
#: worst-state criterion is what makes it lose the tie the median cannot judge.
CLOGGED = dict(dwell_s=0.020, duty=0.10, shape=2.0, clogs=CLOGS_3)
#: Four clogs instead of three; the same descent-stops-on-the-riding-rung shape from a
#: different draw, so the guard is not pinned to one seed. (Before the worst-state
#: criterion this fixture's descent ran to the ladder's own floor on median readings
#: alone -- every rung down to 0.3 s / 100 ms read at or below zero while riding.)
DEEP = dict(dwell_s=0.025, duty=0.12, shape=2.0, clogs=CLOGS_4, seed=7)
#: The proposal INTRUDES (6.8e-06 > 0) and the winner is a WIDER window. The descent
#: must not run here at all -- every rung below the proposal is shorter than a window
#: that already follows the states -- and that gate is what keeps the change cheap.
RISING = dict(dwell_s=0.030, duty=0.15, shape=2.0, clogs=CLOGS_4, seed=3)


def trace(spec):
    key = repr(sorted(spec.items(), key=lambda kv: kv[0]))
    if key not in _CACHE:
        _CACHE[key] = telegraph(**spec)
    return _CACHE[key]


def ranked(spec):
    x = trace(spec)
    rule = P.propose_baseline(P.probe_timescales(x, FS, polarity=-1))
    return x, rule, api(P, "rank_windows")(x, FS, rule)


def cfg(rough, base):
    return B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)


def windows(candidates):
    return [(c.rough_s, c.base_s) for c in candidates]


# ===========================================================================
# A -- the ladder is two-sided, and the floor is what stops it going down
# ===========================================================================
def test_the_ladder_holds_rungs_below_the_proposal():
    """A. THE HUNK. ``if rough <= first[0]: continue`` made a shorter window
    unreachable, and every claim the screen made about optimality was made without
    one. The ladder must contain the proposal (or "ranked" can mean "replaced without
    comparison") and rungs on BOTH sides of it."""
    where_from()
    x = trace(CLOGGED)
    rule = P.propose_baseline(P.probe_timescales(x, FS, polarity=-1))
    ladder = api(P, "candidate_windows")(rule, x.size, FS)
    proposed = (float(rule.config.rough_s), float(rule.config.base_s))

    assert proposed in ladder, (proposed, ladder)
    below = [w for w in ladder if w[0] < proposed[0]]
    above = [w for w in ladder if w[0] > proposed[0]]
    assert below, ("the ladder is still one-sided: no rung below the proposal, so the "
                   "tie-break the module documents cannot ever be exercised", ladder)
    assert above, ("this fixture has no rungs above the proposal either, so it is not "
                   "testing a two-sided ladder", ladder)
    assert [r for r, _ in ladder] == sorted(r for r, _ in ladder), ladder
    assert len(set(r for r, _ in ladder)) == len(ladder), ("a rough window appears "
                                                           "twice", ladder)
    assert ladder == ((0.3, 0.1), (0.5, 0.1), (1.0, 0.3), (2.0, 0.5), (3.0, 1.0)), ladder


def test_the_ladder_stops_going_down_where_the_floor_binds():
    """A. THE BOUND ON THE WAY DOWN, and why it is `base_floor_s` and not the grid.

    `propose_baseline` carries ``rough_s`` UP whenever the floor lifts ``base_s``
    (``rough = max(rough, snap_to_grid(ROUGH_BASE_RATIO * base))``), so a window whose
    ``base_s`` had to be lifted onto the floor is one the rule CANNOT produce -- and
    those are exactly the rungs ``base_vs_state`` condemns, at ratios outside the
    2.5-5.5x band the proposer maintains. A ladder that lifted them would be searching
    over new windows rather than over the rule's own, and 0.1 s / 100 ms is a rough
    window equal to its base window.
    """
    where_from()
    x = trace(CLOGGED)
    probe = P.probe_timescales(x, FS, polarity=-1)
    rule = P.propose_baseline(probe)
    ladder = api(P, "candidate_windows")(rule, x.size, FS)
    floor = api(P, "base_floor_s")(probe)
    assert floor is not None and floor == 0.1, floor

    shortest = min(r for r, _ in ladder)
    assert shortest == 0.3, (
        "the ladder's floor moved; 0.3 s is the shortest rough window whose "
        "rule-derived base_s (100 ms) already clears the floor here", ladder)
    for rough, base in ladder:
        assert base >= floor, (
            "a rung below the floor Baseline health certifies", (rough, base), floor)
        assert base <= rough, (rough, base)
        if rough < float(rule.config.rough_s):
            assert base == P.snap_to_grid(rough / P.ROUGH_BASE_RATIO), (
                "a DOWNWARD rung whose base_s was lifted onto the floor -- the rule "
                "would have carried rough_s up instead, so this is not a window it "
                "can produce", (rough, base))
            assert 2.5 <= rough / base <= 5.5, (
                "a downward rung outside the ratio band the proposer maintains",
                (rough, base), rough / base)
    # ...and the rung one grid point below the shortest really is excluded BY THE
    # FLOOR, so this test is about the bound and not about the grid running out.
    assert P.snap_to_grid(0.2 / P.ROUGH_BASE_RATIO) < floor, (
        "0.2 s no longer derives a base_s below the floor, so nothing here shows why "
        "the ladder stopped at 0.3 s")


def test_the_ladder_never_reaches_a_window_whose_base_is_the_grids_own_clamp():
    """A. The other downward guard, on the branch `base_floor_s` returns ``None``.

    With no floor (an unmeasurable state dwell, or the gating branch) the descent has
    nothing to stop it but the grid, and `snap_to_grid` CLAMPS at ``GRID_MIN_S``: at
    ``rough_s`` = 0.002 s the derived ``base_s`` is 0.0005 s, the clamp returns
    0.002 s, and the ladder would hold a rough window EQUAL to its base window. The
    rule cannot produce that either.
    """
    where_from()
    probe = P.probe_timescales(trace(CLOGGED), FS, polarity=-1)
    # A probe with no measurable median state: `base_floor_s` returns None, exactly as
    # it does on a recording too short for a reference window.
    import dataclasses
    blind = dataclasses.replace(probe, state_dwell_s=None)
    assert api(P, "base_floor_s")(blind) is None
    rule = P.BaselineProposal(config=cfg(0.5, 0.1), param_source="proposed",
                              reason="synthetic", probe=blind)
    ladder = api(P, "candidate_windows")(rule, int(60 * FS), FS)
    assert (0.5, 0.1) in ladder, ladder
    assert min(r for r, _ in ladder) < 0.5, ("nothing descends here", ladder)
    for rough, base in ladder:
        assert base <= rough, (rough, base)
        if rough < 0.5:
            assert base < rough, (
                "a rough window equal to its base window reached the ladder",
                (rough, base), ladder)
            assert rough / P.ROUGH_BASE_RATIO >= P.GRID_MIN_S, (
                "base_s here is the grid's lower clamp, not the rule's ratio",
                (rough, base))


# ===========================================================================
# B -- the walk: the descent, its gate, its stop and its cost
# ===========================================================================
def test_the_descent_takes_the_shorter_window_that_ties_on_both_criteria():
    """B. **THE PARAMETER VALUE THE OLD TESTS NEVER PICKED**, amended by BL.

    Phase 2 is gated on the PROPOSAL being clean, and every synthetic in the RK file
    proposes a window that intrudes -- the one shape at which the descent cannot
    fire. Here the descent runs from a hand-clean 2 s / 500 ms on the CLOG-FREE
    telegraph (`CLOGGED` minus its clogs): 1 s / 300 ms one rung below reads a clean
    zero AND rides into nothing, so two windows measure the SAME on both criteria and
    the documented rule is "ties broken by the shortest window" -- the shorter one
    must be taken. The clog-free trace is the point: on `CLOGGED` itself the shorter
    rung's tie is a tie on the median alone (its baseline rides into the ~600 ms
    clogs, worst state 0.966), and taking it there was the BL defect --
    ``test_a_rung_that_rides_loses_the_tie_the_median_cannot_judge`` asserts that
    side.
    """
    where_from()
    spec = dict(CLOGGED)
    spec["clogs"] = ()
    x = trace(spec)
    probe = P.probe_timescales(x, FS, polarity=-1)
    rule = P.BaselineProposal(config=cfg(2.0, 0.5), param_source="proposed",
                              reason="synthetic", probe=probe)
    pr = api(P, "rank_windows")(x, FS, rule)

    got = (float(pr.config.rough_s), float(pr.config.base_s))
    assert got == (1.0, 0.3), (
        "the search takes the LONGER of two windows that measure the same on both "
        "criteria", got, [c.to_dict() for c in api(pr, "ranking")])

    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert (1.0, 0.3) in by, ("the rung below the proposal was never estimated", by)
    assert P.intrusion_cost(by[(1.0, 0.3)].fraction) == 0.0
    assert by[(1.0, 0.3)].rides is False, (
        "the winner rides here, so this fixture no longer demonstrates a LEGITIMATE "
        "tie", by[(1.0, 0.3)].to_dict())
    assert P.intrusion_cost(by[(2.0, 0.5)].fraction) == 0.0, (
        "the proposal no longer ties with the winner, so nothing here demonstrates "
        "the tie-break", by[(2.0, 0.5)].to_dict())
    assert by[(2.0, 0.5)].rides is False
    assert by[(1.0, 0.3)].chosen is True
    assert by[(2.0, 0.5)].chosen is False
    assert sum(c.chosen for c in pr.ranking) == 1


def test_a_rung_that_rides_loses_the_tie_the_median_cannot_judge():
    """B. **THE BL AMENDMENT, ON THE FIXTURE THAT USED TO ASSERT THE OPPOSITE.**

    On `CLOGGED` the rule proposes 2 s / 500 ms and the descent estimates 1 s /
    300 ms, which ties at a median cost of 0.0 -- and rides into the ~600 ms clogs
    (worst state 0.966 against `WORST_STATE_INTRUSION_MAX`'s 0.5). This test asserted
    that the shorter rung WINS that tie; that was the blindness: a median over ~280
    states cannot see one state ridden fully into, and on the real -600 mV recording
    the same choice un-detects a 521 ms clog and invents ~30 shallow events. A rung
    that rides loses to one that stays out, whatever the medians say.
    """
    where_from()
    x, rule, pr = ranked(CLOGGED)
    assert (float(rule.config.rough_s), float(rule.config.base_s)) == (2.0, 0.5), (
        "the rule no longer proposes 2 s / 500 ms here, so this fixture is not the "
        "-600 mV shape any more", rule.config.describe())

    got = (float(pr.config.rough_s), float(pr.config.base_s))
    assert got == (2.0, 0.5), (
        "the search takes the rung whose baseline rides into the clogs, on the "
        "strength of a median that cannot see them",
        got, [c.to_dict() for c in api(pr, "ranking")])

    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert (1.0, 0.3) in by, (
        "the riding rung was never estimated, so nothing here shows the guard "
        "judging it", by)
    assert P.intrusion_cost(by[(1.0, 0.3)].fraction) == 0.0, (
        "the shorter rung no longer ties on the median, so this fixture no longer "
        "demonstrates the tie the guard exists to break", by[(1.0, 0.3)].to_dict())
    assert by[(1.0, 0.3)].rides is True, (
        "the shorter rung no longer rides here; the fixture has lost the shape",
        by[(1.0, 0.3)].to_dict())
    assert float(by[(1.0, 0.3)].worst_fraction) > P.WORST_STATE_INTRUSION_MAX
    assert by[(2.0, 0.5)].rides is False
    assert by[(2.0, 0.5)].chosen is True
    assert by[(1.0, 0.3)].chosen is False


def test_the_descent_stops_at_the_first_window_that_is_not_clean():
    """B. The mirror of the stop-at-clean rule, and it is NOT exact -- see
    `rank_windows` for the measurement it rests on. What is asserted here is the
    OBSERVABLE consequence: the first not-clean rung below the proposal is estimated
    (so the stop is a measurement), and nothing below THAT is (so it is a stop). On
    `CLOGGED` the stopper is a rung that RIDES (median 0.0, worst state 0.966) --
    amended by BL: before the worst-state criterion the descent read that rung as
    clean and walked past it to 0.5 s / 100 ms."""
    where_from()
    x, rule, pr = ranked(CLOGGED)
    ladder = api(P, "candidate_windows")(rule, x.size, FS)
    est = windows(pr.ranking)

    assert (1.0, 0.3) in est, ("the descent stopped without measuring anything", est)
    stopper = next(c for c in pr.ranking if (c.rough_s, c.base_s) == (1.0, 0.3))
    assert stopper.rides and P.intrusion_cost(stopper.fraction) == 0.0, (
        "1 s / 300 ms no longer rides at a clean median here, so it is not the "
        "median-blind stopper this fixture exists to produce",
        [c.to_dict() for c in pr.ranking])
    assert (0.5, 0.1) in ladder and (0.3, 0.1) in ladder, ladder
    assert (0.5, 0.1) not in est and (0.3, 0.1) not in est, (
        "the search descended PAST a window that rides into the states, which is a "
        "walk with no stop rule at all", est)


def test_the_descent_does_not_run_from_a_proposal_that_already_intrudes():
    """B. THE GATE, which is what makes the change cost one estimate rather than five.

    Every rung below an intruding proposal is SHORTER than a window that already
    follows the states. `RISING` is that case: the proposal reads 6.8e-06, the winner
    is a WIDER window, and not one rung below the proposal may be estimated.
    """
    where_from()
    x, rule, pr = ranked(RISING)
    proposed = (float(rule.config.rough_s), float(rule.config.base_s))
    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert proposed in by, ("the proposal itself was not estimated", by)
    assert P.intrusion_cost(by[proposed].fraction) > 0.0, (
        "the proposal is clean on this fixture, so it cannot show that the descent "
        "is gated", by[proposed].to_dict())
    assert float(pr.config.rough_s) > proposed[0], (
        "the winner is not above the proposal, so this fixture is not the shape the "
        "gate is about", pr.config.describe())
    assert not [w for w in windows(pr.ranking) if w[0] < proposed[0]], (
        "the search descended below a window that already follows the states -- that "
        "is unbounded cost for windows that cannot win", windows(pr.ranking))
    # ...and the ladder really did hold rungs down there to descend into.
    ladder = api(P, "candidate_windows")(rule, x.size, FS)
    assert [w for w in ladder if w[0] < proposed[0]], ladder


def test_the_walk_never_estimates_more_than_the_cap():
    """B. `RANK_MAX_CANDIDATES` moved: it used to truncate `candidate_windows` and now
    bounds the ESTIMATES. Truncating a two-sided ladder from the top deletes the
    winner rather than the rungs the search was never going to reach -- on -600 mV a
    four-rung cap applied to the ladder would have kept 0.3 / 0.5 / 1 / 2 and dropped
    3, which is harmless, but on ferritin_data2 it would have kept 0.03 / 0.05 / 0.1 /
    0.2 and dropped the window the app actually offers."""
    where_from()
    for name, spec in (("CLOGGED", CLOGGED), ("DEEP", DEEP), ("RISING", RISING)):
        x, rule, pr = ranked(spec)
        ladder = api(P, "candidate_windows")(rule, x.size, FS)
        assert len(pr.ranking) <= api(P, "RANK_MAX_CANDIDATES"), (name, pr.ranking)
        assert len(ladder) > api(P, "RANK_MAX_CANDIDATES"), (
            f"{name}: the ladder is no longer bigger than the cap, so nothing here "
            "shows that the cap is spent on estimates rather than on rungs", ladder)


def test_the_ranking_rows_come_back_shortest_first_whatever_order_they_were_measured():
    """B. The two phases estimate in two directions, and `proposal_summary` and
    `_ranking_reason` both render these rows in order. A ladder printed in walk order
    is a table the reader cannot scan for the shortest window that reaches zero -- and
    on the -600 mV recording the walk order is 2 s, 1 s, 0.5 s."""
    where_from()
    for name, spec in (("CLOGGED", CLOGGED), ("DEEP", DEEP), ("RISING", RISING)):
        _, _, pr = ranked(spec)
        rough = [c.rough_s for c in pr.ranking]
        assert rough == sorted(rough), (name, rough)
        rows = [k for k, _ in P.proposal_summary(pr, FS) if k.startswith("Ranked:")]
        assert len(rows) == len(pr.ranking) == len(set(rows)), (name, rows)


def test_the_two_sided_walk_is_still_cancellable():
    """B. Cancellation is ambient (`analysis.cancel`) and the walk is the most
    expensive thing the probe job does. A second loop is a second place to forget."""
    where_from()
    x = trace(CLOGGED)
    rule = P.propose_baseline(P.probe_timescales(x, FS, polarity=-1))
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            api(P, "rank_windows")(x, FS, rule)


# ===========================================================================
# C -- the paragraph: what was tested, what was not, and what was superseded
# ===========================================================================
def test_the_screen_no_longer_claims_an_optimality_it_did_not_test():
    """C. **THE SENTENCE, TWICE OVER.** "Nothing can beat a baseline that follows
    none of the states" was false under the module's own objective, and it was
    printed on both apo recordings whose proposal reads zero. THE FIRST REPLACEMENT
    WAS FALSE TOO, and this test asserted it verbatim: "nothing WIDER can beat a
    baseline that follows none of the states" -- on -600 mV the wider window IS
    better, because "follows none of the states" was only ever the MEDIAN and the
    shorter window had ridden into the clog. A test pinning that sentence was a test
    defending a false general claim. What is asserted now: every optimality claim is
    scoped to the objective and the ladder walked, the worst-state criterion is named
    with its threshold, and the windows never estimated are counted, not judged."""
    where_from()
    _, rule, pr = ranked(CLOGGED)
    reason = pr.reason
    assert FALSE_CLAIM not in reason, (
        "the app still claims nothing can beat this window while never having "
        "estimated the shorter one that ties with it on the median", reason)
    assert ("nothing WIDER can beat a baseline that follows none of the states"
            not in reason), (
        "the unscoped claim is back: a wider window CAN beat one that follows none "
        "of the states at the median -- -600 mV's does, by not riding into the clog",
        reason)
    assert "under this ranking's objective" in reason, (
        "the optimality claim is not scoped to the objective that produced it",
        reason)
    assert "nothing WIDER can beat a window already clean on both criteria" in reason
    assert "not a measurement of any window left unestimated" in reason, reason
    # the worst-state criterion is NAMED, with its threshold, and the rung it rejects
    # carries the number that rejected it
    assert "50% of any one state's excursion" in reason, (
        "the criterion the ranking now decides on is not named to the user", reason)
    assert "RIDES INTO its worst single state" in reason, reason
    assert "1 s / 300 ms" in reason and "96.6%" in reason, reason
    # the short side: what WAS estimated, and what was not
    assert "the nearest one, 1 s / 300 ms, was estimated" in reason, reason
    assert ("2 shorter windows on the grid (0.3 s / 100 ms; 0.5 s / 100 ms) were "
            "NOT estimated") in reason, (
        "the paragraph does not say which windows the search never tried", reason)
    assert "rather than a claim that they are worse" in reason, reason
    # and it does not claim to have searched the whole grid
    assert "SHORTEST window on the grid" not in reason, (
        "the paragraph still says 'on the grid' about a search that stops early",
        reason)
    assert "SHORTEST window ESTIMATED" in reason, reason


def test_the_paragraph_says_when_it_ran_out_of_ladder_rather_than_out_of_evidence():
    """C. The other end of the same honesty: when the descent reaches the ladder's
    own floor there is NO shorter window to disclose, and the sentence that counts
    them would be a count of zero. "3 shorter windows were not estimated" and "there
    is no shorter window" are different facts about the world.

    AMENDED BY BL: `DEEP` used to reach this branch through `rank_windows` -- its
    descent read every rung's MEDIAN as clean all the way down while the rungs rode
    into its clogs (worst state 0.945-1.007), which is the blindness the worst-state
    criterion removes; its descent now stops on the first riding rung. No fixture
    reaches the floor branch through the walk any more, so it is exercised where it
    lives, on `_ranking_reason` itself with a hand-built clean ladder -- exactly how
    this file already tests the two branches below it.
    """
    where_from()
    ranking = (
        _candidate(0.3, 0.1, 0.0, chosen=True),
        _candidate(0.5, 0.1, -1e-6),
        _candidate(1.0, 0.3, 0.0),
        _candidate(2.0, 0.5, 0.0),
    )
    text = ranking_reason(ranking, ranking[0], (2.0, 0.5), LADDER_5)
    assert "There is no shorter window on the grid whose tracking window Baseline " \
           "health would certify" in text, text
    assert "was NOT estimated" not in text, (
        "the paragraph counts shorter windows that do not exist", text)


def test_the_paragraph_says_the_rules_window_is_not_the_offer_when_it_is_not():
    """C + D. **THE SECOND FALSE-BY-OMISSION DEFECT.** On -700 mV the derivation
    declares the floor's 0.5 s / 100 ms three times and the ranking paragraph then
    takes 1 s / 300 ms, with nothing in between saying the first two numbers had been
    superseded. The reader gets three declarations of a window the app is not
    offering. The correction has to come AFTER the claim it corrects and has to name
    both windows. (`RISING`, not `CLOGGED`: since the BL amendment the CLOGGED
    ranking CONFIRMS the rule's window -- the tie that used to supersede it was a tie
    on the blind median -- and RISING is the fixture whose ranking still moves the
    offer, upward.)"""
    where_from()
    _, rule, pr = ranked(RISING)
    proposed = P._fmt_window(float(rule.config.rough_s), float(rule.config.base_s))
    offered = P._fmt_window(float(pr.config.rough_s), float(pr.config.base_s))
    assert proposed != offered
    assert f"THE {proposed} DERIVED ABOVE IS THE RULE'S WINDOW AND NOT WHAT THE APP " \
           f"OFFERS" in pr.reason, pr.reason
    assert f"SUPERSEDES it with {offered}" in pr.reason, pr.reason
    assert pr.reason.index("the rule's rough window") < pr.reason.index(
        "IS THE RULE'S WINDOW AND NOT WHAT THE APP OFFERS"), (
        "the correction comes before the claim it corrects", pr.reason)


def test_the_paragraph_does_not_announce_a_supersession_that_did_not_happen():
    """C. The counter-case, and the reason the sentence is conditional: on `RISING`
    the ranking MOVES the window. On a recording where the ranking confirms the
    rule's own window, announcing that it was superseded would be a new false
    sentence in place of the old one. Since the BL amendment `CLOGGED` itself is that
    recording -- the rule proposes 2 s / 500 ms, the descent estimates the shorter
    rung, the worst-state criterion rejects it, and the offer is the rule's own
    window -- so the counter-case is asserted on the real walk rather than on a
    hand-built proposal."""
    where_from()
    x, rule, pr = ranked(CLOGGED)
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (
        float(rule.config.rough_s), float(rule.config.base_s)), (
        "the ranking no longer confirms the rule's window here, so the test proves "
        "nothing", pr.config.describe())
    assert "NOT WHAT THE APP OFFERS" not in pr.reason, (
        "the app announces that it superseded a window it in fact kept", pr.reason)
    assert "SUPERSEDES" not in pr.reason, pr.reason
    assert "RANKED BY MEASURED INTRUSION" in pr.reason, pr.reason


def test_propose_baseline_alone_labels_its_numbers_and_promises_nothing_else():
    """D, ON THE NON-DEFAULT CODE PATH. `propose_baseline` is public and is called
    without ranking (the RK core file does it, and so does
    ``test_baseline_probe.py``'s acceptance test). The label has to be true THERE too,
    which is why it is a label rather than a forward reference to a paragraph that may
    not follow: "the rule's rough window 2 s" is true of the rule with or without a
    ranking; "see the ranking below" would not be."""
    where_from()
    probe = P.probe_timescales(trace(CLOGGED), FS, polarity=-1)
    rule = P.propose_baseline(probe)
    assert rule.ranking == ()
    assert "the rule's rough window 2 s" in rule.reason, rule.reason
    assert "the rule's base window 500 ms" in rule.reason, rule.reason
    assert "RANKED BY MEASURED INTRUSION" not in rule.reason, rule.reason
    assert "NOT WHAT THE APP OFFERS" not in rule.reason, (
        "an unranked proposal claims something was superseded", rule.reason)


def _candidate(rough, base, fraction, chosen=False):
    return P.BaselineCandidate(
        rough_s=rough, base_s=base, fraction=fraction, n_states=200,
        cost=P.intrusion_cost(fraction), chosen=chosen)


LADDER_5 = ((0.3, 0.1), (0.5, 0.1), (1.0, 0.3), (2.0, 0.5), (3.0, 1.0))


def ranking_reason(ranking, best, proposed, ladder):
    """`_ranking_reason`, with its SIGNATURE checked first.

    Calling a two-argument function with four arguments fails on a ``TypeError`` that
    reads like a broken test rather than like the claim being made. The claim is that
    the paragraph generator is TOLD what the rule proposed and what the ladder held --
    without either it cannot report a supersession or count what it never estimated.
    """
    import inspect
    params = list(inspect.signature(P._ranking_reason).parameters)
    assert params == ["ranking", "best", "proposed", "ladder"], (
        "_ranking_reason is not given the window the rule proposed, nor the ladder it "
        "walked, so it can neither say what it superseded nor count what it never "
        "estimated -- which is the defect this file is about", params)
    return P._ranking_reason(ranking, best, proposed, ladder)


def test_the_paragraph_does_not_say_there_is_no_shorter_window_when_it_estimated_one():
    """C. **A DEFECT FOUND INSIDE THIS FIX, AND THE BRANCH IT LIVES ON.**

    The "no shorter window exists" sentence was written against ``unseen_below``,
    which counts rungs below the SHORTEST ESTIMATED window. When the descent runs off
    the bottom of the ladder and the last rung is the one that STOPS it, that count is
    zero while a shorter window plainly exists -- it was estimated, it is named two
    sentences earlier, and it lost. The paragraph then said "there is no shorter
    window on the grid whose tracking window Baseline health would certify", which is
    false about a window it had just printed.

    No corpus recording and none of the three fixtures above reach that branch (a
    scan of 84 synthetic configurations found none), so it is exercised where it
    lives: on `_ranking_reason` itself, with a ladder and a ranking built by hand.
    A branch that only a real recording can reach is a branch that ships untested.
    """
    where_from()
    ranking = (
        _candidate(0.3, 0.1, 0.004),                      # estimated, and it stopped
        _candidate(0.5, 0.1, 0.0, chosen=True),           # the winner
        _candidate(1.0, 0.3, -1e-6),
        _candidate(2.0, 0.5, 0.0),                        # the rule's own window
    )
    text = ranking_reason(ranking, ranking[1], (2.0, 0.5), LADDER_5)
    assert "0.3 s / 100 ms" in text, text
    assert "There is no shorter window on the grid" not in text, (
        "the paragraph denies the existence of a window it estimated and printed",
        text)
    assert "Every shorter window on the grid was estimated" in text, text
    assert "was NOT estimated" not in text, (
        "it counts unestimated shorter windows where there are none", text)


def test_the_paragraph_reports_the_short_side_when_the_winner_still_intrudes():
    """C. The other half of the same disclosure, on the branch where NOTHING is clean.

    There the descent never runs (it is gated on a clean proposal), so every rung
    below the winner is unestimated -- and the paragraph has to say so rather than
    let "this is the least-intruding window" be read as a statement about the grid.
    """
    where_from()
    ranking = (
        _candidate(1.0, 0.3, 0.02, chosen=True),
        _candidate(2.0, 0.5, 0.03),
    )
    text = ranking_reason(ranking, ranking[0], (1.0, 0.3), LADDER_5)
    # "least-intruding window ESTIMATED" was amended by BL: with the worst-state
    # criterion in the objective, "least-intruding" alone no longer describes the
    # selection rule (a rider with the lowest median loses to a non-rider with a
    # higher one), so the sentence names the objective instead of one half of it.
    assert "best window ESTIMATED under this ranking's objective" in text, text
    assert "NOT a clean one" in text, text
    assert "least-intruding window on the grid" not in text, text
    assert "2 shorter windows on the grid (0.3 s / 100 ms; 0.5 s / 100 ms) were NOT " \
           "estimated" in text, text
    assert "does not descend below a window that already follows the states" in text, \
        text
    assert "NOT WHAT THE APP OFFERS" not in text, (
        "the winner IS the proposal here", text)


# ===========================================================================
# E -- the branches that must still decline, on a two-sided ladder
# ===========================================================================
def test_a_stepping_pore_is_still_not_ranked_in_either_direction():
    """E. The gating exemption is the same clause of the contract that exempts a
    stepping pore from the floor, and a two-sided search must not defeat it: with
    `base_floor_s` returning ``None`` there, the descent would have had nothing to
    stop it and ferritin_data3 would walk off its shipped 0.03 / 0.01 s in the
    DOWNWARD direction as well as the upward one."""
    where_from()
    rng = np.random.default_rng(3)
    n = 1_500_000
    x = np.full(n, -34.8)
    k = int(0.035 * FS)
    lvl = 0.0
    for s in range(0, n, k):
        lvl = -296.0 if lvl == 0.0 else 0.0
        x[s:s + k] += lvl
    x = (x + rng.normal(0.0, 13.8, n)).astype(np.float32)

    probe = P.probe_timescales(x, FS, polarity=-1)
    if not probe.gating_credible:
        pytest.skip("the gating fixture did not read as a stepping pore here")
    rule = P.propose_baseline(probe)
    pr = api(P, "rank_windows")(x, FS, rule)
    assert pr.config.rough_s == rule.config.rough_s
    assert pr.config.base_s == rule.config.base_s
    assert api(pr, "ranking") == ()
    assert "NOT ranked" in pr.reason, pr.reason


def test_a_proposal_at_the_ceiling_declines_and_says_something_true_about_why():
    """E. The decline sentence had to change with the ladder. It used to read "so two
    candidates could not be measured against the same states", which a two-sided
    ladder makes FALSE -- two candidates BELOW the ceiling could be. What cannot be
    measured there is the window being REPLACED, and a ranking that never estimates it
    is the "replaced without comparison" the ladder exists to prevent."""
    where_from()
    x = trace(CLOGGED)
    probe = P.probe_timescales(x, FS, polarity=-1)
    wide = P.BaselineProposal(
        config=cfg(B.reference_rough_s(x.size, FS), 0.5),
        param_source="proposed", reason="synthetic", probe=probe)
    pr = api(P, "rank_windows")(x, FS, wide)
    assert api(pr, "ranking") == ()
    assert "NOT ranked" in pr.reason, pr.reason
    assert "reference window" in pr.reason, pr.reason
    assert "two candidates could not be measured against the same states" \
        not in pr.reason, (
        "the decline still says something a two-sided ladder makes false", pr.reason)
    assert "the window being replaced could not be measured" in pr.reason, pr.reason


# ===========================================================================
# THE CORPUS
# ===========================================================================
@pytest.mark.corpus
@pytest.mark.parametrize("tag", sorted(EXPECTED))
def test_the_corpus_moves_exactly_the_one_recording_the_charter_names(tag):
    """THE BLAST RADIUS, on all six recordings in evidence.

    AMENDED BY BL: the worst-state criterion moves TWO rows against the median-only
    two-sided ladder, both back off a window that rides. -600 mV's descent estimates
    1.0 / 0.3 (median tie at 0.0, rides the 521 ms clog at 0.556) and keeps the
    rule's 2.0 / 0.5; ferritin_data2's upward walk passes THROUGH 1.0 / 0.3 (median
    0.0, rides a real 395 ms deep state at 0.782) to 2.0 / 0.5. None of the six is
    held by slack: -500 mV's proposal is clean on both criteria, so the descent DOES
    run there and estimates 0.5 s / 100 ms at 0.006467; -700 mV and ferritin_data1
    propose windows that intrude, so the descent is gated off, and data1's winner
    has no state longer than its window at all; ferritin_data3 is held by the gating
    exemption, which the ladder must not defeat from below.
    """
    where_from()
    x, fs = load(tag)
    pr = P.probe_and_propose(x, fs, polarity=-1)
    got = (float(pr.config.rough_s), float(pr.config.base_s))
    assert got == pytest.approx(EXPECTED[tag]), pr.config.describe()
    assert len(pr.ranking) == ESTIMATES[tag], (
        "the search estimated a different number of windows than the cost note says",
        tag, windows(pr.ranking))
    health = B.estimate_baseline_result(
        x, fs, config=pr.config, polarity=-1).health
    assert health.ok, [c.name for c in health.checks if not c.ok]


@pytest.mark.corpus
def test_the_six_hundred_millivolt_ladder_is_the_charters_table():
    """THE -600 mV LADDER, with every number measured here over the one shared
    reference mask, so a revert fails on the VALUE and not only on the window.

    AMENDED BY BL. The MEDIAN column is unchanged -- 1.0 / 0.3 and 2.0 / 0.5 still
    both read exactly 0.0 -- and the tie between them is now judged by the criterion
    the median cannot carry: 1.0 / 0.3 rides into the clog region (worst state 0.556
    on a 473 ms run) and 2.0 / 0.5 does not (its only state longer than its own
    500 ms window is the 521 ms clog itself, sailed clean over at -0.112). The
    descent therefore stops on its first rung, and 0.5 / 0.1 -- which the
    median-only walk used to estimate -- is never reached.
    """
    where_from()
    x, fs = _apo(-600)
    ref = B.reference_states(
        x, fs, config=B.DEFAULT_BASELINE.replace(
            rough_s=B.reference_rough_s(x.size, fs) / 2.0))
    assert ref is not None

    def intrusion(rough, base):
        got = B.config_intrusion(x, fs, config=cfg(rough, base), polarity=-1,
                                 state_mask=ref[0])
        assert got is not None, (rough, base)
        return got

    assert intrusion(0.3, 0.1)["fraction"] == pytest.approx(0.002888, abs=5e-6)
    assert intrusion(0.5, 0.1)["fraction"] == pytest.approx(
        0.0031545741324921135, abs=1e-9)
    one = intrusion(1.0, 0.3)
    two = intrusion(2.0, 0.5)
    assert one["fraction"] == 0.0
    assert two["fraction"] == 0.0
    assert one["worst_fraction"] == pytest.approx(0.5557, abs=5e-4), one
    assert one["worst_dwell_s"] == pytest.approx(0.473, abs=5e-3), one
    assert two["worst_fraction"] == pytest.approx(-0.112, abs=5e-3), two
    assert two["worst_dwell_s"] == pytest.approx(0.5215, abs=5e-3), (
        "2.0 / 0.5's worst qualifying state is no longer the 521 ms clog itself",
        two)
    assert one["worst_fraction"] > P.WORST_STATE_INTRUSION_MAX >= two["worst_fraction"]

    pr = P.probe_and_propose(x, fs, polarity=-1)
    assert windows(pr.ranking) == [(1.0, 0.3), (2.0, 0.5)], pr.ranking
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (2.0, 0.5)
    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert by[(1.0, 0.3)].rides is True and by[(1.0, 0.3)].chosen is False
    assert by[(2.0, 0.5)].rides is False and by[(2.0, 0.5)].chosen is True
    assert (0.5, 0.1) not in windows(pr.ranking), (
        "0.5 s / 100 ms was estimated; the descent should have stopped one rung "
        "earlier, at the first window that rides into a state")


@pytest.mark.corpus
def test_the_five_hundred_millivolt_file_descends_and_declines_to_move():
    """THE COUNTER-CASE, and the one that shows the descent is not a licence.

    -500 mV reads -0.00040 at its own proposal -- a clean zero to `intrusion_cost` --
    which is precisely the condition that fires the descent. It therefore DOES
    estimate 0.5 s / 100 ms, measures 0.006467 there, and stays at 1 s / 300 ms. A
    version of this change that kept -500 mV by never looking would be indistinguish-
    able from this one on the window alone, which is why the estimate is asserted.
    """
    where_from()
    x, fs = _apo(-500)
    pr = P.probe_and_propose(x, fs, polarity=-1)
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (1.0, 0.3)
    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert (0.5, 0.1) in by, (
        "the descent did not run on the one recording whose proposal is clean",
        windows(pr.ranking))
    assert by[(0.5, 0.1)].fraction == pytest.approx(0.006466877265713259, abs=1e-9)
    assert by[(1.0, 0.3)].fraction == pytest.approx(
        -0.00040053280177352197, abs=1e-9)
    assert by[(1.0, 0.3)].chosen is True


@pytest.mark.corpus
def test_the_ferritin_data3_exemption_survives_a_two_sided_search():
    """THE HARD CONSTRAINT, checked explicitly rather than assumed.

    ferritin_data3 reads ``base_vs_state`` **2.9940119760479043** against a threshold
    of 3.0 -- 0.2% of clearance -- and is held out of the ranking by the gating
    judgement rather than by that margin. Its 0.03 / 0.01 s reads an intrusion of 0.49
    because its "states" are conductance levels, so a ranking that did not decline
    would walk it off its shipped configuration; and with `base_floor_s` returning
    ``None`` there, a two-sided ladder would have walked it DOWNWARD as well.
    """
    where_from()
    x, fs = _corpus("ferritin_data3")
    pr = P.probe_and_propose(x, fs, polarity=-1)
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (0.03, 0.01)
    assert pr.ranking == (), pr.ranking
    assert "NOT ranked" in pr.reason, pr.reason
    res = B.characterisation_baseline(x, fs, config=pr.config, polarity=-1,
                                      health=True, cache=False)
    check = res.health.check("base_vs_state")
    assert check is not None
    assert check.value == 2.9940119760479043, check.value
    assert B.BASE_STATE_RATIO_MIN == 3.0
    assert check.level == "ok"


@pytest.mark.corpus
def test_the_published_windows_are_still_reachable_bit_for_bit():
    """THE BOUND ON ALL OF THIS. The owner has PUBLISHED from -700 mV at 0.1 / 0.03 s
    and from -600 mV at 2.0 / 0.5 s. The ranking changes what the app OFFERS; it may
    not move a byte of what an entered window measures."""
    where_from()
    x7, fs7 = _apo(-700)
    res7 = B.characterisation_baseline(x7, fs7, config=cfg(0.1, 0.03), polarity=-1,
                                       health=True, cache=False)
    assert res7.fingerprint == "sha256:1a29aabf5fb1", res7.fingerprint
    assert not res7.health.ok, "the published -700 mV window is no longer condemned"

    x6, fs6 = _apo(-600)
    res6 = B.characterisation_baseline(x6, fs6, config=cfg(2.0, 0.5), polarity=-1,
                                       health=True, cache=False)
    assert res6.fingerprint == "sha256:ce8ae31f10bd", res6.fingerprint
    assert res6.health.ok, [c.name for c in res6.health.checks if not c.ok]


@pytest.mark.corpus
@pytest.mark.parametrize("tag", ["apo-500", "apo-600", "apo-700"])
def test_the_search_costs_what_the_cost_note_says(tag):
    """THE INTERACTIVE COST, measured rather than asserted in a comment.

    The descent only runs where phase 1 stopped after one estimate, so the two phases
    never both spend. Measured here as the wall clock of `rank_windows` alone, on the
    owner's own recordings, against a bound generous enough not to be a flake and
    tight enough to catch a walk that lost its stop rule (which would estimate the
    whole ladder: 5 rungs on all three).
    """
    where_from()
    x, fs = load(tag)
    rule = P.propose_baseline(P.probe_timescales(x, fs, polarity=-1))
    t0 = time.perf_counter()
    pr = api(P, "rank_windows")(x, fs, rule)
    elapsed = time.perf_counter() - t0
    ladder = api(P, "candidate_windows")(rule, x.size, fs)
    print(f"{tag}: {len(pr.ranking)} of {len(ladder)} rungs estimated in "
          f"{elapsed:.2f} s")
    assert len(pr.ranking) == ESTIMATES[tag], windows(pr.ranking)
    assert len(pr.ranking) < len(ladder), (
        "every rung on the ladder was estimated, so neither stop rule fired", ladder)
    assert elapsed < 12.0, (tag, elapsed)


# ===========================================================================
# WHAT THE CHARTERED MOVE COSTS, MEASURED AND PINNED
# ===========================================================================
@pytest.mark.corpus
def test_the_shorter_window_rides_into_the_five_hundred_millisecond_clog():
    """**THE MEASUREMENT THAT POINTED THE OTHER WAY -- AND THE OBJECTIVE HAS NOW
    LEARNED IT.** This test was written as the pinned cost of the RP move, with the
    note "if the objective ever learns about the case a median cannot see, this test
    is where the change will show up". It has. `baseline_probe`'s module docstring
    opens on this exact state:

        "The apo ``_0006_`` file has a **521 ms partial-clog state at t = 37.13 s**,
        and a 500 ms rough window sitting inside a 521 ms state *is* that state ...
        It needs ``rough_s = 2.0``."

    A 1 s rough window is not 500 ms, but the clog still fills 52% of it -- more than
    half, so the rough MEDIAN inside the clog IS the clog level. Measured on the
    recording, the tracked baseline through the clog:

    ==============  ==================  ====================================
    window          baseline in the clog  what the clog becomes
    ==============  ==================  ====================================
    2 s / 500 ms    -27.8 pA            ONE event, 519 ms, depth 28.9 pA
    **1 s / 300 ms**  **-54.5 pA**      **not an event at all**
    ==============  ==================  ====================================

    The raw current there is -57.7 pA against an open pore of -34.5, so at
    1 s / 300 ms the baseline follows 87% of a 23 pA state. ``baseline_intrusion``'s
    MEDIAN cannot see it -- one state among 239 -- and all eight health checks stay
    green; what sees it is the WORST-state reading the same measurement now carries
    (0.556 > `WORST_STATE_INTRUSION_MAX`'s 0.5), which is why the ranking rejects
    this window. The estimator numbers below are unchanged from the day this test
    was the cost of the move; what changed is that the app no longer offers the
    window that produces them.
    """
    where_from()
    x, fs = _apo(-600)
    a, b = int(37.13 * fs), int(37.65 * fs)
    assert float(np.median(x[a:b])) == pytest.approx(-57.678, abs=0.01), (
        "the 521 ms clog is not where this test thinks it is")

    def in_clog(rough, base):
        res = B.characterisation_baseline(
            x, fs, config=cfg(rough, base), polarity=-1, health=True, cache=False)
        return float(np.median(np.asarray(res.baseline, float)[a:b]))

    shipped = in_clog(2.0, 0.5)
    rejected = in_clog(1.0, 0.3)
    print(f"\n-600 mV clog: 2 s / 500 ms -> {shipped:.3f} pA, "
          f"1 s / 300 ms -> {rejected:.3f} pA")
    assert shipped == pytest.approx(-27.770, abs=0.05), (
        "the shipped window no longer sails over the clog", shipped)
    assert rejected == pytest.approx(-54.466, abs=0.05), (
        "1 s / 300 ms no longer rides into the clog through the estimator -- if the "
        "estimator changed, the guard's evidence needs re-measuring", rejected)
    assert abs(rejected - shipped) > 20.0, (shipped, rejected)

    # ...the MEDIAN is still blind to it -- that has not changed and cannot: one
    # state does not move a median over 239 -- and the WORST-state reading beside it
    # is what the ranking now judges the window on. Asserted together, so nobody has
    # to take either half on trust.
    ref = B.reference_states(
        x, fs, config=B.DEFAULT_BASELINE.replace(
            rough_s=B.reference_rough_s(x.size, fs) / 2.0))
    got = B.config_intrusion(x, fs, config=cfg(1.0, 0.3), polarity=-1,
                             state_mask=ref[0])
    assert got is not None and got["fraction"] == 0.0, got
    assert int(got["n_states"]) > 100, (
        "the intrusion is a median over this many states; one clog cannot move it",
        got)
    assert got["worst_fraction"] == pytest.approx(0.5557, abs=5e-4), got
    assert got["worst_fraction"] > P.WORST_STATE_INTRUSION_MAX, (
        "the worst-state reading no longer condemns the riding window here, so the "
        "guard has lost the recording it was built for", got)
