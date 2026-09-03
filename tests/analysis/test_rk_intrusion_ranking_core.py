"""RK -- the proposer chooses its window by MEASURED INTRUSION, not by legality.

THE DEFECT
==========
`baseline_probe.base_floor_s` is a LOWER BOUND, and `baseline_probe.propose_baseline`
never looked above it: it took the shortest ``base_s`` on its grid that
``base_vs_state`` tolerates and stopped. On the owner's -700 mV apo recording that is
0.5 s / 100 ms, whose ``baseline_intrusion`` reads **0.00428** -- the tracked baseline
follows 0.43% of every state's own depth, so every ``depth_pA`` and every ``frac_depth``
in the table reads about that much shallow, while all eight health checks stay green.
One grid point up, at **1.0 s / 300 ms** -- a window the same rule already proposes for
the -500 mV recording -- the same statistic reads **exactly 0.0**, and the stamped open
pore moves from -48.857 pA to -48.218 against the app's own event-excluded estimate of
-48.218 (1.33% out to 0.0005% out).

The app was already computing the statistic that ranks those two windows. It compared
them to nothing.

WHAT IS ASSERTED HERE
=====================
The invariant, not the symptom: **when the app can measure which of two legal windows
intrudes less, it must choose the one that does.** Five hunks carry it and every test
below names the one it fires on, so a revert of any single hunk fails a test that says
which:

  H1  `baseline.config_intrusion`            -- the health block's own intrusion
                                                number, callable BEFORE a run exists
  H2  `baseline_probe.rank_windows` + `candidate_windows` + `intrusion_cost`
                                             -- the ladder, the ceiling, the
                                                one-sided objective and the tie-break
  H3  `baseline_probe.probe_and_propose`     -- the app's one entry point ranks
  H4  `BaselineProposal.ranking` + `proposal_summary`
                                             -- the evidence travels with the answer
  H5  (the screen; `tests/gui/test_rk_intrusion_ranking_wired.py`)

Nothing here asserts a screen.
"""
from __future__ import annotations

import pathlib

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

#: What the corpus proposes once the windows are RANKED. Three of the six move, and
#: they move together because they are one reading: a 100 ms tracking window follows
#: about half a percent of each state on all three (0.00428 / 0.00509 / 0.00531), and
#: no threshold separates the chartered recording from the other two.
#:
#: RP MADE THE LADDER TWO-SIDED, AND BL PUT A SECOND CRITERION ON IT. The worst
#: single state (longer than the candidate's own tracking window) is read beside the
#: median, because a median over hundreds of states cannot see ONE state the
#: baseline rides fully into -- `baseline_probe.WORST_STATE_INTRUSION_MAX`. Two rows
#: here are set by that criterion: -600 mV's descent estimates the shorter 1.0 / 0.3
#: (median tie at 0.0) and REJECTS it for riding the 521 ms clog at 0.556, and
#: ferritin_data2's upward walk passes through 1.0 / 0.3 (median 0.0, rides a real
#: 395 ms deep state at 0.782) to 2.0 / 0.5. -500 mV descends too (its proposal is
#: clean) and measures 0.5 / 0.1 at 0.006467, so it stays where it is on evidence
#: rather than on slack. This table asserted (1.0, 0.3) for both amended rows -- the
#: blind median's answer -- until the BL batch; the evidence is in
#: ``tests/analysis/test_bl_worststate_ranking.py``.
EXPECTED = {
    "apo-500": (1.0, 0.3),            # unchanged -- 0.5 / 0.1 estimated, 0.006467
    "apo-600": (2.0, 0.5),            # 1.0 / 0.3 ties at 0.0 and RIDES the clog (BL)
    "apo-700": (1.0, 0.3),            # WAS 0.5 / 0.1, at baseline_intrusion 0.00428
    "ferritin_data1": (1.0, 0.3),     # WAS 0.5 / 0.1, at 0.00509
    "ferritin_data2": (2.0, 0.5),     # 1.0 / 0.3 RIDES a 395 ms deep state (BL)
    "ferritin_data3": (0.03, 0.01),   # gating: not ranked, and must stay unranked
}

_CACHE: dict = {}


def api(obj, name):
    """``getattr`` as an ASSERTION that the surface exists at all, so a package
    without the fix fails on the CLAIM rather than on an AttributeError that reads
    like a broken test."""
    assert hasattr(obj, name), (
        f"{getattr(obj, '__name__', type(obj).__name__)} has no {name!r}: nothing in "
        "the proposer ranks candidate windows by measured intrusion, which is the "
        "defect this file is about"
    )
    return getattr(obj, name)


def where_from():
    """Print (and return) the package under test. A revert run from inside the repo
    silently tests the LIVE tree -- PYTHONPATH does not beat cwd when cwd holds the
    package -- so every test here says where it resolved from."""
    import nanoscope
    p = pathlib.Path(nanoscope.__file__).resolve()
    print(f"\nnanoscope resolved from: {p}")
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


def telegraph(dwell_s, duty, *, dur=60.0, seed=11, depth=110.0, i0=-34.8,
              noise=4.84, shape=4.0, fs=FS):
    """A two-level pore at a chosen state length and occupancy -- the same generator
    ``tests/analysis/test_fup_proposer_floor.py`` uses, so the two files are talking
    about the same fixture, with the dwell distribution's SHAPE exposed.

    ``dur`` is 60 s because the ranking's CEILING is `baseline.reference_rough_s`,
    which is ``duration / 4`` until it saturates at 5 s: a shorter fixture would have
    no ladder to rank on and would prove nothing.
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    x = np.full(n, float(i0))
    spans, t = [], 0.0
    gap = dwell_s * (1.0 - duty) / duty
    while True:
        t += rng.exponential(gap)
        d = rng.gamma(shape, dwell_s / shape)
        s, e = int(round(t * fs)), int(round((t + d) * fs))
        if e >= n:
            break
        x[s:e] -= depth
        spans.append((s, e))
        t += d
    x += rng.normal(0.0, noise, n)
    return x.astype(np.float32), spans


#: THE LADDER FIXTURE. 30 ms states at 20% occupancy with a shape-2 dwell tail, which
#: is the only synthetic in this file whose proposed window actually intrudes: the rule
#: offers 0.3 s / 100 ms, whose baseline follows 0.000653 of each state's depth, and
#: the ranking walks 0.3 / 0.1 -> 0.5 / 0.1 (0.000583) -> 1.0 / 0.3 (zero) and stops.
#:
#: **It is two orders of magnitude cleaner than the owner's own recordings** (0.00428
#: on the -700 mV file at the same 100 ms tracking window), and that gap is the honest
#: limit of a synthetic here: a Gaussian-noise telegraph with no drift and no
#: heavy-tailed clogs simply does not push a rolling median into its states the way a
#: real pore does. Every magnitude claim in this file is therefore made on the CORPUS
#: tests below; the synthetics are here for the STRUCTURE -- the ladder, the ceiling,
#: the stop rule, the tie-break, the branches that decline and the strings.
LADDER = dict(dwell_s=0.030, duty=0.20, shape=2.0, dur=60.0)


def ladder_trace():
    x, _ = telegraph(LADDER["dwell_s"], LADDER["duty"],
                     dur=LADDER["dur"], shape=LADDER["shape"])
    return x


def gating_trace(fs=FS, n=1_500_000):
    """A pore whose OPEN LEVEL steps every ~35 ms -- ferritin_data3's shape, and the
    one case that must NOT be ranked."""
    rng = np.random.default_rng(3)
    x = np.full(n, -34.8)
    k = int(0.035 * fs)
    lvl = 0.0
    for s in range(0, n, k):
        lvl = -296.0 if lvl == 0.0 else 0.0
        x[s:s + k] += lvl
    x += rng.normal(0.0, 13.8, n)
    return x.astype(np.float32)


def cfg(rough, base):
    return B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)


# ===========================================================================
# 1. H1 -- the ranking statistic IS the health block's number
# ===========================================================================
@pytest.mark.parametrize("rough,base", [(0.5, 0.1), (1.0, 0.3)])
def test_the_ranking_reads_the_health_blocks_own_intrusion_bit_for_bit(rough, base):
    """H1. THE INVARIANT UNDER THE WHOLE CHANGE.

    A proposer that ranked on its OWN estimate of intrusion would be choosing windows
    against a statistic the health block then disagrees with -- the exact shape of the
    defect the floor was invented to close one layer down (the proposer sized base_s
    off the p99.9 while the check divided by the median, so the app condemned what it
    had just offered). `baseline.config_intrusion` therefore is not an estimate of
    ``baseline_intrusion``; it is that measurement, run early.

    Asserted as EQUALITY OF FLOATS, deliberately. ``approx`` here would pass over a
    reading taken with a different ``min_depth_pA``, a different dilation or a
    different mask -- which is precisely the class of drift this test exists to catch,
    and all three of those change the number by less than a percent on this fixture.
    """
    where_from()
    x, _ = telegraph(0.020, 0.10)
    probe = P.probe_timescales(x, FS, polarity=-1)
    ref = B.reference_states(
        x, FS, config=B.DEFAULT_BASELINE.replace(
            rough_s=B.reference_rough_s(x.size, FS) / 2.0))
    assert ref is not None, "no reference window on this fixture; the test proves nothing"

    got = api(B, "config_intrusion")(
        x, FS, config=cfg(rough, base), polarity=-1, state_mask=ref[0])
    health = B.characterisation_baseline(
        x, FS, config=cfg(rough, base), polarity=-1, health=True, cache=False).health

    assert health.intrusion_fraction is not None, (
        "the health block did not measure an intrusion on this fixture, so there is "
        "nothing to compare against")
    assert got is not None
    assert got["fraction"] == health.intrusion_fraction, (
        got["fraction"], health.intrusion_fraction)
    assert got["n_states"] == health.intrusion_n_states
    assert got["intrusion_pA"] == health.intrusion_pA
    assert got["depth_pA"] == health.intrusion_depth_pA
    assert probe.usable


def test_config_intrusion_reports_not_measured_rather_than_no_intrusion():
    """H1. A mask with no state deep enough to measure gives ``None``, never 0.0.

    `intrusion_cost` treats 0.0 as "this window cannot be beaten" and stops the search
    there, so a `config_intrusion` that answered 0.0 for "I could not tell" would end
    the ranking on the first rung, every time, on any recording it cannot measure."""
    where_from()
    x, _ = telegraph(0.020, 0.10)
    empty = np.zeros(x.size, dtype=bool)
    assert api(B, "config_intrusion")(
        x, FS, config=cfg(0.5, 0.1), polarity=-1, state_mask=empty) is None


# ===========================================================================
# 2. H2 -- the ladder, the ceiling and the bound
# ===========================================================================
def test_the_ladder_contains_the_proposal_and_stops_below_the_reference_window():
    """H2. THE CEILING, and why it is `baseline.reference_rough_s` rather than the
    recording.

    Every candidate's intrusion is read over the ONE wide-pass mask
    `baseline.reference_states` returns, and that function returns ``None`` the moment
    the configured ``rough_s`` reaches the reference window. A candidate above the
    ceiling would therefore be measured against a DIFFERENT mask from the ones below
    it, and the ranking would stop comparing like with like. It is also exactly where
    `base_floor_s`'s guarantee ends, so the two bounds are one line rather than two.

    UPDATED FOR THE TWO-SIDED LADDER (RP). Two assertions here were written against a
    ladder that began at the proposal and only walked up, and both had to go:

    * ``ladder[0] == the proposal`` -- the proposal is now somewhere INSIDE the
      ladder, not at its foot. What the contract actually needs is that the ladder
      CONTAINS it, which is what "ranked" can never mean "replaced without comparison"
      rests on, and that is what is asserted;
    * ``len(ladder) <= RANK_MAX_CANDIDATES`` -- that constant bounds the ESTIMATES,
      and truncating a two-sided ladder from the top deletes the winner rather than
      the rungs the search was never going to reach. The bound on the estimates is
      asserted where it now lives, in
      ``test_rp_two_sided_ranking_core.py::test_the_walk_never_estimates_more_than_the_cap``.
    """
    where_from()
    x = ladder_trace()
    proposal = P.propose_baseline(P.probe_timescales(x, FS, polarity=-1))
    ladder = api(P, "candidate_windows")(proposal, x.size, FS)

    ceiling = B.reference_rough_s(x.size, FS)
    assert (float(proposal.config.rough_s), float(proposal.config.base_s)) in ladder, (
        "the ladder does not contain the proposal, so 'ranked' can mean 'replaced "
        "without comparison'", ladder)
    assert len(ladder) >= 2, ("nothing to rank against on this fixture", ladder, ceiling)
    assert all(r < ceiling for r, _ in ladder), (ladder, ceiling)
    assert [r for r, _ in ladder] == sorted(r for r, _ in ladder)
    assert all(b <= r for r, b in ladder), ladder


def test_the_ladder_is_spelled_on_the_grid_the_panel_hands_back():
    """H2. ``3 * 10.0**-1`` is ``0.30000000000000004``, one ULP above ``0.3``, and a
    `baseline.BaselineConfig` fingerprint is a hash of the fields' ``repr``. A ladder
    built the arithmetic way would give a proposed 1.0 / 0.3 s window a DIFFERENT
    identity from the same window typed into the panel's spin box, and two
    identically-windowed recordings would read NOT COMPARABLE in Analysis > Plots.
    `snap_to_grid` already carries that lesson; the ladder has to carry it too."""
    where_from()
    x = ladder_trace()
    proposal = P.propose_baseline(P.probe_timescales(x, FS, polarity=-1))
    for rough, base in api(P, "candidate_windows")(proposal, x.size, FS):
        assert repr(rough) == repr(float(f"{rough:g}")), rough
        assert repr(base) == repr(float(f"{base:g}")), base
        assert B.DEFAULT_BASELINE.replace(
            rough_s=rough, base_s=base, decim=None).fingerprint(FS) == \
            B.DEFAULT_BASELINE.replace(
                rough_s=float(f"{rough:g}"), base_s=float(f"{base:g}"),
                decim=None).fingerprint(FS)


def test_the_objective_is_one_sided_because_the_check_it_searches_is_one_sided():
    """H2. ``baseline_intrusion`` condemns on ``fraction > BASELINE_INTRUSION_MAX``
    and on nothing else, so a baseline sitting slightly ABOVE the local open pore
    reads negative and the check does not distinguish it from zero.

    This is not a detail: the -500 mV recording reads **-0.00040** at its own proposal.
    Ranked on the signed value it would beat a window reading exactly 0.0 on the
    strength of a sign the check ignores; ranked on ``abs`` it would LOSE to one, and
    the -500 mV proposal would move to 2.0 / 0.5 s on the same non-evidence. Clipping
    at zero is what makes both of those impossible AND what makes the stop rule exact.
    """
    where_from()
    cost = api(P, "intrusion_cost")
    assert cost(0.05) == 0.05
    assert cost(0.0) == 0.0
    assert cost(-0.00040053280177352197) == 0.0
    assert cost(-1.0) == 0.0
    assert cost(None) is None
    assert cost(float("nan")) is None
    assert cost(float("inf")) is None


def test_a_tie_goes_to_the_shortest_window_and_the_search_stops_at_the_first_zero():
    """H2. THE TIE-BREAK, AND THE THING THAT MAKES IT AFFORDABLE.

    On the -700 mV recording 1.0 / 0.3 and 2.0 / 0.5 BOTH read exactly 0.0, so
    something has to separate them and "the first one tried" is not a reason. The rule
    is: lowest measured intrusion, ties to the SHORTEST window -- because a longer
    window tracks slower drift less well, which is what ``length_tracking`` and
    ``rough_sigma_ratio`` exist to measure, and the cost is visible on the corpus (on
    the -500 mV file, the genuinely non-stationary one, ``overshoot`` goes 0.000227 at
    1.0 / 0.3 to 0.002446 at 2.0 / 0.5 while the intrusion does not improve at all).

    That tie-break is what makes the stop rule EXACT rather than a heuristic: the
    objective is bounded below by zero and ties go to the shorter window, so nothing
    above a zero can win and nothing above a zero is estimated. Asserted here as the
    observable consequence -- the last rung measured reads zero, and no rung above it
    appears in the ladder at all, even though a wider one WOULD also read zero.
    """
    where_from()
    x = ladder_trace()
    proposal = api(P, "rank_windows")(
        x, FS, P.propose_baseline(P.probe_timescales(x, FS, polarity=-1)))
    ranking = api(proposal, "ranking")
    assert ranking, "nothing was ranked on this fixture; the test proves nothing"
    assert ranking[-1].cost == 0.0, [c.to_dict() for c in ranking]
    assert ranking[-1].chosen is True
    assert sum(c.chosen for c in ranking) == 1
    assert all(c.cost > 0.0 for c in ranking[:-1]), [c.to_dict() for c in ranking]
    assert (proposal.config.rough_s, proposal.config.base_s) == (
        ranking[-1].rough_s, ranking[-1].base_s)

    ladder = api(P, "candidate_windows")(
        P.propose_baseline(P.probe_timescales(x, FS, polarity=-1)), x.size, FS)
    assert len(ladder) > len(ranking), (
        "the ladder held no window above the winner, so 'the search stopped' is not "
        "what is being demonstrated here", ladder, len(ranking))
    # ...and the window it declined to estimate really would have tied.
    nxt = ladder[len(ranking)]
    ref = B.reference_states(
        x, FS, config=B.DEFAULT_BASELINE.replace(
            rough_s=B.reference_rough_s(x.size, FS) / 2.0))
    got = api(B, "config_intrusion")(
        x, FS, config=cfg(*nxt), polarity=-1, state_mask=ref[0])
    assert got is not None and api(P, "intrusion_cost")(got["fraction"]) == 0.0, (
        "the next rung does not tie, so the tie-break is not what chose the shorter "
        "window here", nxt, got)


def test_the_ranking_is_cancellable_like_everything_else_that_loops():
    """H2. Cancellation is ambient (`analysis.cancel`), and the ranking is the most
    expensive thing the probe job now does -- several seconds of estimator per
    candidate. A ranking that could not be interrupted would put the Cancel button
    back in the state the whole cancel invariant exists to remove."""
    where_from()
    x = ladder_trace()
    proposal = P.propose_baseline(P.probe_timescales(x, FS, polarity=-1))
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            api(P, "rank_windows")(x, FS, proposal)


# ===========================================================================
# 3. H2 -- the branches that must NOT be ranked
# ===========================================================================
def test_a_stepping_pore_is_not_ranked_and_says_so():
    """H2, and the same clause of the contract that exempts a gating pore from the
    FLOOR. ``baseline_intrusion`` renders NOT APPLICABLE on a credibly gating pore --
    there the long states ARE baseline levels, so the share of them the baseline
    follows is supposed to be large -- and ranking on a number the check itself calls
    inapplicable would be ranking on nothing.

    Measured, and this is why the exemption is load-bearing rather than tidy:
    ferritin_data3's 0.03 / 0.01 s reads an intrusion of 0.49 against a 0.03 threshold
    because its "states" are levels. Ranked on that, it would walk the ladder to the
    ceiling and off its shipped configuration."""
    where_from()
    x = gating_trace()
    probe = P.probe_timescales(x, FS, polarity=-1)
    if not probe.gating_credible:
        pytest.skip("the gating fixture did not read as a stepping pore here")
    proposal = P.propose_baseline(probe)
    ranked = api(P, "rank_windows")(x, FS, proposal)
    assert ranked.config.rough_s == proposal.config.rough_s
    assert ranked.config.base_s == proposal.config.base_s
    assert api(ranked, "ranking") == ()
    assert "NOT ranked" in ranked.reason, ranked.reason
    assert "steps" in ranked.reason, ranked.reason


def test_a_fallback_proposal_is_left_alone_and_not_given_a_second_negation():
    """H2. A probe that measured nothing already says at length that it measured
    nothing and fell back to `baseline.DEFAULT_BASELINE`. Appending "...and they were
    not ranked either" adds a second negation with no new fact in it -- and the
    sentence it would append ("these are the windows the rule derives from this
    recording's timescales") is not even true of the global default. The fallback
    proposal must come back IDENTICAL, including by identity of its config."""
    where_from()
    short = np.zeros(1000, dtype=np.float32)
    proposal = P.propose_baseline(P.probe_timescales(short, FS, polarity=-1))
    assert proposal.param_source == "default"
    ranked = api(P, "rank_windows")(short, FS, proposal)
    assert ranked is proposal
    assert ranked.config is B.DEFAULT_BASELINE
    assert "NOT ranked" not in ranked.reason, ranked.reason


def test_a_recording_with_no_room_above_the_proposal_says_it_was_not_ranked():
    """H2. NOT SILENCE. "The shortest window the floor certifies" and "the
    least-intruding window on the grid" are different claims, and a user reading the
    same two numbers either way has no way to tell which one he is being offered."""
    where_from()
    x, _ = telegraph(0.020, 0.10, dur=30.0)
    probe = P.probe_timescales(x, FS, polarity=-1)
    # A proposal already at/above the reference window -- the shape the rule reaches on
    # a recording whose longest state is a good fraction of its whole length.
    wide = P.BaselineProposal(
        config=cfg(B.reference_rough_s(x.size, FS), 0.5),
        param_source="proposed", reason="synthetic", probe=probe)
    ranked = api(P, "rank_windows")(x, FS, wide)
    assert api(ranked, "ranking") == ()
    assert "NOT ranked" in ranked.reason, ranked.reason
    assert "reference window" in ranked.reason, ranked.reason


# ===========================================================================
# 4. H3 -- the app's one entry point ranks
# ===========================================================================
def test_probe_and_propose_ranks_where_propose_baseline_alone_does_not():
    """H3. `gui.analysis_panel` calls `probe_and_propose` and nothing else, so if the
    ranking is not on that path the whole change is unreachable from the app. Two
    batches ago an agent shipped exactly this shape of half-change.

    `propose_baseline` keeps its behaviour and its signature -- it turns a
    `BaselineProbe` into windows and has no trace to measure anything on -- so the two
    answers legitimately differ, and this asserts BOTH halves of that."""
    where_from()
    x = ladder_trace()
    probe = P.probe_timescales(x, FS, polarity=-1)
    unranked = P.propose_baseline(probe)
    ranked = P.probe_and_propose(x, FS, polarity=-1)

    assert api(unranked, "ranking") == (), (
        "propose_baseline ranked on its own; it has no trace to rank with")
    assert ranked.ranking, "probe_and_propose did not rank"
    assert ranked.config.base_s > unranked.config.base_s, (
        "this fixture no longer distinguishes the two, so the test proves nothing",
        unranked.config.describe(), ranked.config.describe())
    assert ranked.param_source == "proposed"
    assert ranked.probe is not None


def test_the_ranked_window_is_still_certified_by_every_health_check():
    """H2/H3. Ranking walks UP the grid, and up the grid ``overshoot`` and
    ``rough_sigma_ratio`` can degrade -- neither is bounded by the proposer. M1's Apply
    gate is the backstop for that, but a proposal that routinely reached it would turn
    a backstop back into a workflow step, which is the state this app has already paid
    to leave. So the window ranking chooses must come back with all eight green."""
    where_from()
    x = ladder_trace()
    proposal = P.probe_and_propose(x, FS, polarity=-1)
    assert api(proposal, "ranking")
    health = B.characterisation_baseline(
        x, FS, config=proposal.config, polarity=-1, health=True, cache=False).health
    assert health.ok, [c.name for c in health.checks if not c.ok]
    assert health.intrusion_fraction is not None
    assert P.intrusion_cost(health.intrusion_fraction) == proposal.ranking[-1].cost


# ===========================================================================
# 5. H4 -- the evidence travels with the answer
# ===========================================================================
def test_the_summary_names_every_window_that_was_estimated():
    """H4. The claim being made is a COMPARISON, and a comparison with one row in it
    is an assertion. Every candidate estimated gets a row, the winner is marked, and
    the labels are unique -- callers read these rows back through
    ``dict(proposal_summary(...))`` and a repeated label would silently drop a
    candidate out of the evidence."""
    where_from()
    x = ladder_trace()
    proposal = P.probe_and_propose(x, FS, polarity=-1)
    rows = P.proposal_summary(proposal, FS)
    ranked = [(k, v) for k, v in rows if k.startswith("Ranked:")]

    assert len(ranked) == len(api(proposal, "ranking")) >= 2, ranked
    assert len(dict(rows)) == len(rows), ("a label is repeated", [k for k, _ in rows])
    assert sum("CHOSEN" in v for _, v in ranked) == 1, ranked
    chosen = next(c for c in proposal.ranking if c.chosen)
    assert f"Ranked: {chosen.rough_s:g} s / {chosen.base_s * 1e3:g} ms" in dict(rows)
    for k, v in ranked:
        assert "state depth" in v, (k, v)
    # ...AND NO VALUE CARRIES A "<" INTO THE PANEL'S RICH TEXT. `summary_label` is
    # rendered as HTML, and Qt's parser reads a "<" as the start of a tag and deletes
    # everything up to the next ">" -- that is how the overshoot criterion vanished
    # from the health block while the source still contained it. The panel escapes
    # these rows now (H5), and the values still do not rely on it: a marker spelled
    # "<- chosen" would be invisible on the one screen it exists for. (">" alone is
    # harmless and the pre-existing rows already carry it: "-> base_s >= 100 ms".)
    assert not any("<" in v for _, v in rows), rows


def test_a_negative_intrusion_is_not_rendered_as_a_negative_quantity():
    """H4. The -500 mV recording's chosen window reads **-0.00040**, and
    `intrusion_cost` calls that "followed none of it" because the check does. A row
    printing a bare "-0.040%" next to a CHOSEN mark, under a reason line claiming the
    baseline follows none of the states, is the panel disagreeing with itself in the
    one place a user checks the claim."""
    where_from()
    c = api(P, "BaselineCandidate")(rough_s=1.0, base_s=0.3, fraction=-0.0004,
                                    n_states=845, cost=0.0, chosen=True)
    api(P.BaselineProposal, "ranking")
    proposal = P.BaselineProposal(
        config=cfg(1.0, 0.3), param_source="proposed", reason="", probe=None,
        ranking=(c,))
    value = dict(P.proposal_summary(proposal, FS))["Ranked: 1 s / 300 ms"]
    assert "NONE" in value, value
    assert "-0.040%" in value, value
    assert "CHOSEN" in value, value


def test_the_reason_names_the_window_that_lost_and_what_it_costs():
    """H4. "1 s / 300 ms was chosen" is not a justification. The paragraph has to name
    the window the rule WOULD have offered, what it measured, and what that costs a
    depth -- otherwise the user cannot tell that anything was overridden, which is the
    same finding the floor's own reason line already carries one layer down."""
    where_from()
    x = ladder_trace()
    proposal = P.probe_and_propose(x, FS, polarity=-1)
    reason = proposal.reason
    loser = api(proposal, "ranking")[0]
    best = proposal.ranking[-1]
    assert "RANKED BY MEASURED INTRUSION" in reason, reason
    assert f"{loser.rough_s:g} s / {loser.base_s * 1e3:g} ms" in reason, reason
    assert f"{loser.fraction * 100:.3f}%" in reason, reason
    assert f"{best.rough_s:g} s / {best.base_s * 1e3:g} ms is taken" in reason, reason
    assert "shallow" in reason, reason
    assert "SHORTEST" in reason, reason


# ===========================================================================
# 6. THE CORPUS -- what ranking by measurement actually gives
# ===========================================================================
@pytest.mark.corpus
@pytest.mark.parametrize("tag", sorted(EXPECTED))
def test_the_corpus_proposes_the_least_intruding_window_on_its_grid(tag):
    """H3, against the six recordings in evidence, and the honest blast radius.

    THREE PROPOSALS MOVE, NOT ONE. 0.00428 (-700 mV), 0.00509 (ferritin_data1) and
    0.00531 (ferritin_data2) is ONE phenomenon at three recordings -- a 100 ms tracking
    window follows about half a percent of each state -- and no threshold separates the
    chartered recording from the other two. A rule that moved only the -700 mV file
    would be a rule fitted to the answer.

    The -500 mV file that does NOT move is not preserved by slack either: it reads a
    cost of zero at its own proposal, which is exactly the case in which the two-sided
    search DESCENDS, and the rung below it (0.5 s / 100 ms) is estimated at 0.006467
    and loses. -600 mV takes the same descent, and the rung below ties at 0.0 -- and
    LOSES anyway, because it rides into the 521 ms clog (worst state 0.556, the BL
    criterion); data2's walk passes through its own riding rung the same way.
    ferritin_data3 is held by the gating exemption.

    The estimator still reproduces the notebook's shipped 0.5 / 0.1 s bit-for-bit when
    those windows are entered; what moved is which windows the app OFFERS.
    """
    where_from()
    x, fs = _apo(int(tag[3:])) if tag.startswith("apo") else _corpus(tag)
    proposal = P.probe_and_propose(x, fs, polarity=-1)
    got = (float(proposal.config.rough_s), float(proposal.config.base_s))
    assert got == pytest.approx(EXPECTED[tag]), proposal.config.describe()
    assert proposal.param_source == "proposed"

    health = B.estimate_baseline_result(
        x, fs, config=proposal.config, polarity=-1).health
    assert health.ok, [c.name for c in health.checks if not c.ok]


@pytest.mark.corpus
def test_the_seven_hundred_millivolt_file_lands_on_the_window_that_intrudes_none():
    """THE CHARTERED CHANGE, with the numbers on both sides of it and the one the
    floor stopped at in the middle.

    The floor took it from 0.1 / 0.03 s (``baseline_intrusion`` 0.0416, condemned) to
    0.5 / 0.1 s (0.00428, certified). The ranking takes it from there to 1.0 / 0.3 s
    (exactly 0.0). Every one of those three numbers is asserted here, measured through
    `baseline.config_intrusion` over the one shared reference mask, so a revert of the
    ranking fails on the value rather than on the window.
    """
    where_from()
    x, fs = _apo(-700)
    proposal = P.probe_and_propose(x, fs, polarity=-1)
    assert (float(proposal.config.rough_s),
            float(proposal.config.base_s)) == pytest.approx((1.0, 0.3))

    ref = B.reference_states(
        x, fs, config=B.DEFAULT_BASELINE.replace(
            rough_s=B.reference_rough_s(x.size, fs) / 2.0))
    assert ref is not None

    def intrusion(rough, base):
        got = api(B, "config_intrusion")(
            x, fs, config=cfg(rough, base), polarity=-1, state_mask=ref[0])
        assert got is not None, (rough, base)
        return got["fraction"]

    assert intrusion(0.1, 0.03) == pytest.approx(0.0416, abs=5e-4)
    assert intrusion(0.5, 0.1) == pytest.approx(0.004278771600903435, abs=1e-9)
    assert intrusion(1.0, 0.3) == 0.0

    # ...and the two rungs the app actually estimated are exactly those last two.
    assert [(c.rough_s, c.base_s) for c in proposal.ranking] == [(0.5, 0.1), (1.0, 0.3)]
    assert proposal.ranking[0].chosen is False
    assert proposal.ranking[1].chosen is True


@pytest.mark.corpus
def test_the_published_seven_hundred_millivolt_window_is_still_reachable():
    """THE BOUND ON ALL OF THIS. The owner has PUBLISHED from 0.1 / 0.03 s. Nothing
    here may make that window unreachable or move a byte of what it produces -- the
    ranking changes what the app OFFERS, never what an entered window measures."""
    where_from()
    x, fs = _apo(-700)
    res = B.characterisation_baseline(
        x, fs, config=cfg(0.1, 0.03), polarity=-1, health=True, cache=False)
    assert res.fingerprint == "sha256:1a29aabf5fb1", res.fingerprint
    assert res.health.intrusion_fraction == pytest.approx(0.0416, abs=5e-4)
    assert not res.health.ok, "the published window is no longer condemned"
