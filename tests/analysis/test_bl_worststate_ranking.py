"""BL -- the ranking sees the catastrophic state its median is structurally blind to.

THE DEFECT
==========
`baseline_probe.rank_windows` scored every candidate window on ONE number:
``baseline_intrusion``'s MEDIAN over the recording's states. A median over hundreds
of states cannot be moved by ONE state, however fully the baseline enters it, and
that case is in evidence. On the owner's -600 mV apo recording (the ``_0006_`` file)
there is a 521 ms partial clog at t = 37.13 s, depth ~29 pA:

    ==============  ============  ===========  =======================================
    window          median        worst state  what detection finds at 36.5-38.5 s
    ==============  ============  ===========  =======================================
    2 s / 500 ms    0.0           -0.112       the clog as ONE 519.5 ms event
    1 s / 300 ms    **0.0**       **0.556**    clog NOT DETECTED; ~30 shallow events
                                               invented on a baseline below -45 pA
    ==============  ============  ===========  =======================================

The two-sided ladder (RP) read that as a TIE and took the shorter window; all eight
health checks stayed green both ways, because the intrusion check reports the same
blind median. The proposal was a window that un-detects a half-second blockade.

THE FIX, AND WHAT IS ASSERTED HERE
==================================
`baseline.tracked_intrusion` now reports the WORST single state beside the median
(``worst_fraction`` / ``worst_dwell_s`` / ``worst_t_s``), taken only over the states
LONGER than the candidate's own ``base_s`` -- a shorter state's per-state ratio is
noise (0.495 on a 2.7 ms ferritin_data2 state, within 1% of the threshold), and the
mechanism needs a state the tracking median can dwell inside. `rank_windows` judges
it against `baseline_probe.WORST_STATE_INTRUSION_MAX` (0.5 -- a rolling median sits
inside a state exactly when the state fills more than half its window, and the
corpus separates at that line by 9x: ridden windows read >= 0.556, windows that stay
out read <= 0.061 or have no qualifying state at all). The objective is
lexicographic: **a rung that rides into a state loses to one that stays out; then
the median; then the shortest window.** Four hunks:

  W1  `baseline.tracked_intrusion` / `config_intrusion` -- the worst state measured,
      over the dwell-gated states
  W2  `BaselineCandidate.rides` + `WORST_STATE_INTRUSION_MAX` -- the criterion
  W3  `rank_windows` -- the lexicographic objective, and both stop rules on it
  W4  `_candidate_phrase` / `_ranking_reason` / `proposal_summary` -- every claim
      scoped to the objective, the criterion named, the rider's number printed

The criterion also found the SAME failure on a second recording it was not built
from: ferritin_data2's median-only offer (1.0 / 0.3) rides a real 395 ms deep state
at t = 99.3 s to 78%, and now walks up to 2.0 / 0.5 -- asserted in the corpus tables
of the RP and RK files. The BOUNDS of the fix are asserted here too: a tie between
two windows clean on BOTH criteria still goes to the shorter one, and the -500 /
-700 mV proposals are bit-for-bit what they were before the criterion existed --
windows, ranking rows and measured fractions.

The screen-level path is ``tests/gui/test_rp_two_sided_ranking_wired.py``, amended by
this batch to the guarded offer.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P

FS = 50_000.0
APO = {
    -500: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0008_process data_-500 mV.abf"),
    -600: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0006_Process_-600 mV.abf"),
    -700: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0005_Process_-700 mV.abf"),
}

_CACHE: dict = {}

#: The two sentences the RP rewrite printed in place of the one it removed, both
#: untrue as claims about baselines (-600 mV's wider window IS better than the
#: "clean" shorter one) and neither allowed back on any screen.
UNSCOPED_WIDER_CLAIM = ("nothing WIDER can beat a baseline that follows none of "
                        "the states")
UNSCOPED_TIE_CLAIM = "could at best tie, and would lose the tie-break"


def api(obj, name):
    """``getattr`` as an ASSERTION, so a package without the fix fails on the CLAIM
    rather than on an AttributeError that reads like a broken test."""
    assert hasattr(obj, name), (
        f"{getattr(obj, '__name__', type(obj).__name__)} has no {name!r}: the "
        "ranking cannot see the state its median is blind to, which is the defect "
        "this file is about")
    return getattr(obj, name)


def where_from():
    """Print AND CHECK the package under test (see the RP core file for why)."""
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


def telegraph(dwell_s, duty, *, dur=60.0, seed=11, depth=110.0, i0=-34.8,
              noise=4.84, shape=4.0, fs=FS, clogs=()):
    """The RP core file's generator, verbatim: a two-level pore with optional long
    partial-clog states. ``clogs`` of ~600 ms is what makes a 1 s rough window ride
    (the state fills > half the window, so the rolling median IS the clog level)."""
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

#: The -600 mV shape (the RP file's `CLOGGED`): the rule proposes 2 s / 500 ms, the
#: descent estimates 1 s / 300 ms, and that rung ties on the median while riding into
#: the ~600 ms clogs at a worst state of 0.966.
CLOGGED = dict(dwell_s=0.020, duty=0.10, shape=2.0, clogs=CLOGS_3)


def trace(spec):
    key = repr(sorted(spec.items(), key=lambda kv: kv[0]))
    if key not in _CACHE:
        _CACHE[key] = telegraph(**spec)
    return _CACHE[key]


def cfg(rough, base):
    return B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)


def ref_mask(x, fs):
    ref = B.reference_states(
        np.asarray(x, np.float64), fs, config=B.DEFAULT_BASELINE.replace(
            rough_s=B.reference_rough_s(x.size, fs) / 2.0))
    assert ref is not None, "no reference states here; the test proves nothing"
    return ref[0]


# ===========================================================================
# W1 -- the measurement: the worst state travels beside the median
# ===========================================================================
def test_config_intrusion_reports_the_worst_state_and_the_median_is_untouched():
    """W1. THE HUNK. `tracked_intrusion` returned four medians and nothing else, so
    no caller could see the one state a median cannot show. The worst trio is new;
    the medians must be the health block's own numbers still, asserted as EQUALITY OF
    FLOATS against the health block exactly as the RK suite pins them -- a guard that
    moved the median while adding the worst would trade one blindness for another."""
    where_from()
    x = trace(CLOGGED)
    mask = ref_mask(x, FS)
    got = api(B, "config_intrusion")(
        x, FS, config=cfg(1.0, 0.3), polarity=-1, state_mask=mask)
    assert got is not None
    for key in ("worst_fraction", "worst_dwell_s", "worst_t_s"):
        assert key in got, (key, sorted(got))

    # The riding window: the worst state is one of the ~600 ms clogs, followed
    # almost entirely, at one of the three planted times.
    assert got["worst_fraction"] > 0.9, got
    assert got["worst_fraction"] >= got["fraction"], got
    assert 0.45 <= got["worst_dwell_s"] <= 0.75, got
    assert any(abs(got["worst_t_s"] - t0) < 0.1 for t0, _, _ in CLOGS_3), got

    # ...and the median half is bit-identical to what the health block reports for
    # the same window, so W1 changed a report's SIZE and not its numbers.
    health = B.characterisation_baseline(
        x, FS, config=cfg(1.0, 0.3), polarity=-1, health=True, cache=False).health
    assert got["fraction"] == health.intrusion_fraction
    assert got["n_states"] == health.intrusion_n_states
    assert got["intrusion_pA"] == health.intrusion_pA
    assert got["depth_pA"] == health.intrusion_depth_pA

    # The window that sails over the clogs reads a small worst state -- the two
    # populations this criterion separates, on one fixture.
    wide = api(B, "config_intrusion")(
        x, FS, config=cfg(2.0, 0.5), polarity=-1, state_mask=mask)
    assert wide is not None
    assert wide["worst_fraction"] < 0.1, wide


def test_the_worst_state_is_read_only_over_states_the_window_could_dwell_inside():
    """W1. THE DWELL GATE, exercised directly on `tracked_intrusion`. The per-state
    fraction is a ratio of two median differences and on a SHORT state it is noise --
    ferritin_data2 reads 0.495 on a 2.7 ms state at 2.0 / 0.5 s, within 1% of the
    threshold the real failure clears by 11%. So the worst is taken only over the
    states longer than ``worst_min_dwell_s`` (the caller's ``base_s``), and when no
    state is that long the trio is ``None`` -- "nothing to ride into", never 0.0.
    Built by hand so each state's followed share is exact."""
    where_from()
    fs = 1000.0
    n = 20_000
    x = np.zeros(n)
    b = np.zeros(n)
    mask = np.zeros(n, dtype=bool)
    # a SHORT state (100 ms) the baseline follows 80% of -- the noise-shaped case
    a0, a1 = 2000, 2100
    x[a0:a1] = -100.0
    b[a0:a1] = -80.0
    mask[a0:a1] = True
    # a LONG state (1 s) the baseline stays out of -- the case the criterion is for
    c0, c1 = 10_000, 11_000
    x[c0:c1] = -100.0
    mask[c0:c1] = True

    ungated = B.tracked_intrusion(x, b, mask, fs)
    assert ungated is not None
    assert ungated["worst_fraction"] == pytest.approx(0.8, abs=1e-6), ungated
    assert ungated["worst_dwell_s"] == pytest.approx(0.1, abs=2e-3), ungated

    gated = B.tracked_intrusion(x, b, mask, fs, worst_min_dwell_s=0.5)
    assert gated is not None
    assert gated["worst_fraction"] == pytest.approx(0.0, abs=1e-6), (
        "the gate did not exclude the short state from the worst reading", gated)
    assert gated["worst_dwell_s"] == pytest.approx(1.0, abs=2e-3), gated
    # the median half must not move with the gate -- it is the health block's number
    assert gated["fraction"] == ungated["fraction"]
    assert gated["n_states"] == ungated["n_states"] == 2

    none_long = B.tracked_intrusion(x, b, mask, fs, worst_min_dwell_s=2.0)
    assert none_long is not None
    assert none_long["worst_fraction"] is None, (
        "with no state long enough the worst must be None -- an absent measurement, "
        "not a zero", none_long)
    assert none_long["worst_dwell_s"] is None
    assert none_long["worst_t_s"] is None
    assert none_long["fraction"] == ungated["fraction"]


# ===========================================================================
# W2 -- the criterion
# ===========================================================================
def test_rides_is_a_threshold_on_the_worst_state_never_on_the_median():
    """W2. ``rides`` is `BaselineCandidate`'s reading of the worst state against
    `WORST_STATE_INTRUSION_MAX`, and ``None`` reads as False -- "not shown to ride",
    never "shown not to" -- because an unmeasurable candidate already cannot win."""
    where_from()
    thr = api(P, "WORST_STATE_INTRUSION_MAX")
    assert thr == 0.5, (
        "the threshold moved; it was placed in the measured factor-of-two gap "
        "between every ridden window in evidence (>= 0.556) and every window that "
        "stays out (<= 0.273), at the point where a rolling median crosses onto the "
        "state's side of the window", thr)

    def cand(worst, fraction=0.0):
        return P.BaselineCandidate(rough_s=1.0, base_s=0.3, fraction=fraction,
                                   n_states=200, cost=P.intrusion_cost(fraction),
                                   worst_fraction=worst, worst_dwell_s=0.5,
                                   worst_t_s=37.1)

    assert api(cand(0.51), "rides") is True
    assert cand(thr).rides is False, "the bound is exclusive: AT the line is not past it"
    assert cand(0.49).rides is False
    assert cand(None).rides is False
    # ...and the median has no say in it, in either direction.
    assert cand(0.9, fraction=0.0).rides is True
    assert cand(0.1, fraction=0.2).rides is False


# ===========================================================================
# W3 -- the objective: a rider loses, in both walk directions
# ===========================================================================
def test_a_riding_rung_loses_the_tie_the_median_cannot_judge():
    """W3, THE DESCENT SIDE (the -600 mV shape). The rule proposes 2 s / 500 ms, the
    descent estimates 1 s / 300 ms, the two tie at a median of exactly zero -- and
    the shorter one rides (worst 0.966), so the tie is broken the only way it can be
    judged: toward the window that stays out of the states."""
    where_from()
    x = trace(CLOGGED)
    rule = P.propose_baseline(P.probe_timescales(x, FS, polarity=-1))
    assert (float(rule.config.rough_s), float(rule.config.base_s)) == (2.0, 0.5), (
        "the fixture no longer proposes 2 s / 500 ms; it has lost the -600 mV shape",
        rule.config.describe())
    pr = api(P, "rank_windows")(x, FS, rule)
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (2.0, 0.5), (
        "the ranking takes the rung that rides into the clogs",
        [c.to_dict() for c in pr.ranking])
    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert P.intrusion_cost(by[(1.0, 0.3)].fraction) == 0.0, (
        "the rider no longer ties on the median, so this is not the tie the guard "
        "exists to break", by[(1.0, 0.3)].to_dict())
    assert by[(1.0, 0.3)].rides is True
    assert by[(2.0, 0.5)].rides is False
    assert by[(2.0, 0.5)].chosen is True
    # ...and the descent STOPPED on the rider: nothing shorter was estimated.
    assert min(c.rough_s for c in pr.ranking) == 1.0, (
        "the search descended past a riding rung", [c.to_dict() for c in pr.ranking])


def test_a_riding_proposal_is_walked_up_and_superseded():
    """W3, THE ASCENT SIDE. A proposal that rides at a clean median used to stop
    phase 1 on the spot (cost zero read as unbeatable); now it is not clean, the walk
    continues upward, and the offer is the window that stays out -- announced as a
    supersession, since the rule's window is no longer what the app offers."""
    where_from()
    x = trace(CLOGGED)
    probe = P.probe_timescales(x, FS, polarity=-1)
    rule = P.BaselineProposal(config=cfg(1.0, 0.3), param_source="proposed",
                              reason="synthetic", probe=probe)
    pr = api(P, "rank_windows")(x, FS, rule)
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (2.0, 0.5), (
        "a proposal riding at a clean median still stops the upward walk",
        [c.to_dict() for c in pr.ranking])
    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert by[(1.0, 0.3)].rides and by[(1.0, 0.3)].chosen is False
    assert "SUPERSEDES it with 2 s / 500 ms" in pr.reason, pr.reason


def test_a_tie_between_two_windows_clean_on_both_criteria_still_goes_to_the_shorter():
    """W3, THE BOUND ON THE FIX. The guard breaks ties the median cannot judge; it
    must not break the tie-break itself. On the clog-free telegraph a hand-clean
    2 s / 500 ms descends to 1 s / 300 ms, both read a clean median AND ride into
    nothing, and the shorter window still wins."""
    where_from()
    spec = dict(CLOGGED)
    spec["clogs"] = ()
    x = trace(spec)
    probe = P.probe_timescales(x, FS, polarity=-1)
    rule = P.BaselineProposal(config=cfg(2.0, 0.5), param_source="proposed",
                              reason="synthetic", probe=probe)
    pr = api(P, "rank_windows")(x, FS, rule)
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (1.0, 0.3), (
        "the worst-state criterion broke the shortest-window tie-break",
        [c.to_dict() for c in pr.ranking])
    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert by[(1.0, 0.3)].rides is False and by[(2.0, 0.5)].rides is False
    assert P.intrusion_cost(by[(1.0, 0.3)].fraction) == 0.0
    assert P.intrusion_cost(by[(2.0, 0.5)].fraction) == 0.0


# ===========================================================================
# W4 -- the sentences: the criterion named, every claim scoped, the number shown
# ===========================================================================
def test_the_reason_names_the_criterion_and_scopes_every_optimality_claim():
    """W4. BL-5b: the RP rewrite replaced one false optimality sentence with two new
    ones. Neither may return, the worst-state criterion must be NAMED with its
    threshold, and the rider must be shown the number that made it lose."""
    where_from()
    x = trace(CLOGGED)
    pr = P.probe_and_propose(x, FS, polarity=-1)
    reason = pr.reason
    assert UNSCOPED_WIDER_CLAIM not in reason, reason
    assert UNSCOPED_TIE_CLAIM not in reason, reason
    # the criterion, named with its threshold, in the user's words
    assert "50% of any one state's excursion" in reason, reason
    assert "loses to any window that stays out" in reason, reason
    # the rider's own numbers: share followed, dwell, where to look
    assert "RIDES INTO its worst single state" in reason, reason
    assert "96.6%" in reason, reason
    assert "cannot move a median" in reason, reason
    # the optimality claims are scoped to the objective and the ladder
    assert "under this ranking's objective" in reason, reason
    assert "on the ladder walked" in reason, reason
    assert "not a measurement of any window left unestimated" in reason, reason


def test_the_summary_rows_carry_both_criteria_so_the_loser_is_explicable():
    """W4. `proposal_summary` renders the evidence table. A rider's row must say it
    rides and by how much -- otherwise the -600 mV table shows a shorter window with
    "follows NONE" losing to a longer one for no visible reason, which is the panel
    disagreeing with its own verdict. The chosen row carries its worst state too, so
    the two rows are comparable on the criterion that separated them."""
    where_from()
    x = trace(CLOGGED)
    pr = P.probe_and_propose(x, FS, polarity=-1)
    rows = dict(P.proposal_summary(pr, FS))
    rider = rows["Ranked: 1 s / 300 ms"]
    chosen = rows["Ranked: 2 s / 500 ms"]
    assert "RIDES INTO" in rider and "96.6%" in rider, rider
    assert "CHOSEN" not in rider, rider
    assert "CHOSEN" in chosen, chosen
    assert "worst single state" in chosen, chosen
    assert "RIDES" not in chosen, chosen
    # no angle bracket may enter the panel's rich text (the RK lesson, kept)
    assert not any("<" in v for v in rows.values()), rows


# ===========================================================================
# THE CORPUS -- the chartered recording, and the two that must not move
# ===========================================================================
@pytest.mark.corpus
def test_the_six_hundred_millivolt_one_click_detects_the_clog_again():
    """(a) OF THE CHARTER, through the app's own entry point and its own detector.

    `probe_and_propose` (the panel's one call) must offer 2.0 / 0.5 with the riding
    1.0 / 0.3 named and rejected in the ladder -- and detection AT THE PROPOSED
    WINDOWS must find the 37.13 s clog as one long event and invent nothing on a
    collapsed baseline. Numbers pinned from the before/after measurement of this
    batch: 276 detected (vs 311 at the ridden window), 11 events in 36.5-38.5 s of
    which ONE is the 519.5 ms clog, and NO event stamped on a local baseline below
    -45 pA."""
    where_from()
    x, fs = _apo(-600)
    pr = P.probe_and_propose(x, fs, polarity=-1)
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (2.0, 0.5), (
        "the one-click proposal is the window that rides into the clog",
        pr.config.describe())
    by = {(c.rough_s, c.base_s): c for c in pr.ranking}
    assert set(by) == {(1.0, 0.3), (2.0, 0.5)}, by
    assert by[(1.0, 0.3)].rides is True
    assert float(by[(1.0, 0.3)].worst_fraction) == pytest.approx(0.5557, abs=5e-4)
    assert float(by[(1.0, 0.3)].worst_dwell_s) == pytest.approx(0.473, abs=5e-3)
    assert by[(2.0, 0.5)].rides is False and by[(2.0, 0.5)].chosen is True

    from nanoscope.analysis.detect import detect_events
    df, prov = detect_events(x, fs, polarity=-1, baseline_config=pr.config,
                             baseline_param_source=pr.param_source)
    assert len(df) == 276, ("the -600 mV detected count at the proposed windows",
                            len(df))
    stretch = df[(df["t_start_s"] >= 36.5) & (df["t_start_s"] <= 38.5)]
    assert len(stretch) == 11, ("the clog region's event count moved", len(stretch))
    clog = stretch[stretch["dwell_s"] > 0.4]
    assert len(clog) == 1, ("the 521 ms clog is not ONE long event at the proposed "
                            "windows", clog.to_dict("records"))
    assert float(clog["t_start_s"].iloc[0]) == pytest.approx(37.13, abs=0.01)
    assert float(clog["dwell_s"].iloc[0]) == pytest.approx(0.5195, abs=1e-3)
    assert float(clog["depth_pA"].iloc[0]) == pytest.approx(28.87, abs=0.05)
    assert int((df["baseline_pA"] < -45.0).sum()) == 0, (
        "the proposed windows stamp events on a collapsed local baseline")


@pytest.mark.corpus
def test_what_the_rejected_window_would_have_cost_measured_through_detection():
    """THE MECHANISM, pinned from the harm side so a revert fails on the physical
    consequence and not only on a window pair. At the rejected 1.0 / 0.3 the clog is
    not detected at all and ~30 shallow events appear on a local baseline below
    -45 pA -- the numbers the guard's threshold is buying."""
    where_from()
    x, fs = _apo(-600)
    from nanoscope.analysis.detect import detect_events
    df, _ = detect_events(x, fs, polarity=-1, baseline_config=cfg(1.0, 0.3),
                          baseline_param_source="user")
    assert len(df) == 311, len(df)
    clog = df[(df["dwell_s"] > 0.4) & (df["t_start_s"] > 37.0)
              & (df["t_start_s"] < 37.3)]
    assert len(clog) == 0, (
        "1 s / 300 ms now detects the clog -- if the estimator changed, the guard's "
        "evidence needs re-measuring", clog.to_dict("records"))
    deep = df[df["baseline_pA"] < -45.0]
    assert len(deep) >= 25, (
        "the invented shallow events are gone; whichever way, this is the number "
        "the worst-state criterion was weighed against", len(deep))
    assert float(deep["t_start_s"].min()) == pytest.approx(37.14, abs=0.05)


@pytest.mark.corpus
@pytest.mark.parametrize("mv,loser_fraction,winner_fraction", [
    (-500, 0.006466877265713259, -0.00040053280177352197),
    (-700, 0.004278771600903435, 0.0),
])
def test_the_five_and_seven_hundred_millivolt_proposals_are_unchanged(
        mv, loser_fraction, winner_fraction):
    """(b) OF THE CHARTER: the criterion must not move the recordings it was not
    built for. Windows, ranking rows AND measured fractions pinned to the values
    recorded from this tree before the criterion existed -- and the worst states of
    both winners asserted BELOW the threshold, so "unchanged" is a measurement of
    clearance rather than an accident of ordering."""
    where_from()
    x, fs = _apo(mv)
    pr = P.probe_and_propose(x, fs, polarity=-1)
    assert (float(pr.config.rough_s), float(pr.config.base_s)) == (1.0, 0.3), (
        f"the {mv} mV proposal moved", pr.config.describe())
    assert [(c.rough_s, c.base_s) for c in pr.ranking] == [(0.5, 0.1), (1.0, 0.3)], (
        pr.ranking)
    assert pr.ranking[0].fraction == pytest.approx(loser_fraction, abs=1e-9)
    assert pr.ranking[1].fraction == pytest.approx(winner_fraction, abs=1e-9)
    assert pr.ranking[1].chosen is True
    for c in pr.ranking:
        assert c.rides is False, (
            f"a {mv} mV rung now reads as riding; the guard has left its measured "
            "clearance", c.to_dict())
    assert pr.ranking[1].worst_fraction is None, (
        "the winner now has a state longer than its 300 ms tracking window; the "
        "measured shape of this recording -- no state a tracking median could dwell "
        "inside -- is gone, and the clearance this file documents with it",
        pr.ranking[1].to_dict())
