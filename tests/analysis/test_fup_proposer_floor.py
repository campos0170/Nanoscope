"""M2 -- the proposer is taught the rule its own health checks enforce.

THE DEFECT
==========
`baseline_health`'s ``base_vs_state`` divides ``base_s`` by the MEDIAN state dwell and
condemns the run below `baseline.BASE_STATE_RATIO_MIN` (3.0). `baseline_probe`'s rule
sized ``base_s`` off ``rough_s``, which is sized off the **p99.9** of the dwells. Those
two statistics differ by 1.5x on the -700 mV apo recording, so the proposer offered
0.1 / 0.03 s there -- **1.3819x a floor of 3.0**, deterministically, for ever -- and the
health block condemned what the proposer had just offered. The consequence, found only
by the integration pass: because M1 made the Apply button REFUSE a condemned proposal,
one press of Apply stopped applying anything at all on that recording and the owner's
default -700 mV numbers changed silently.

WHAT IS ASSERTED HERE
=====================
The invariant, not the symptom: **a configuration these checks condemn as too short to
track must not be proposable.** Four hunks carry it and every test below names the one
it fires on, so a revert of any single hunk fails a test that says which:

  H1  `baseline.reference_state_dwell_s`  -- the check's own divisor, measurable
                                             before any configuration exists
  H2  `baseline_probe.snap_up_to_grid`    -- a bound snapped to the log-NEAREST grid
                                             point is not a bound
  H3  `baseline_probe.BaselineProbe.state_dwell_s` + `base_floor_s`
  H4  the floor itself, in `propose_baseline`'s non-gating branch

The strings the USER reads live in `tests/gui/test_fup_proposer_wired.py`, driven
through a real `MainWindow`. Nothing here asserts a screen.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P

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

#: The proposals that must NOT move, and the three that are chartered to. Measured on
#: this tree through `probe_and_propose` on the conditioned channel-2 trace, now that
#: `rank_windows` chooses by measured intrusion -- the MEDIAN and, since the BL
#: batch, the WORST single state (`baseline_probe.WORST_STATE_INTRUSION_MAX`; see
#: ``tests/analysis/test_bl_worststate_ranking.py``) -- rather than by the floor's
#: lower bound. The two rows marked (BL) asserted the blind median's 1.0 / 0.3 here
#: until that batch; on both recordings that window's baseline rides into a real
#: 400-500 ms deep state at a median of exactly 0.0.
EXPECTED = {
    "apo-500": (1.0, 0.3),            # reads intrusion -0.00040 at rung 1: unchanged
    "apo-600": (2.0, 0.5),            # 1.0 / 0.3 RIDES the 521 ms clog, 0.556 (BL)
    "apo-700": (1.0, 0.3),            # WAS 0.5 / 0.1, at baseline_intrusion 0.004279
    "ferritin_data1": (1.0, 0.3),     # WAS 0.5 / 0.1, at 0.005094
    "ferritin_data2": (2.0, 0.5),     # 1.0 / 0.3 RIDES a 395 ms state, 0.782 (BL)
    "ferritin_data3": (0.03, 0.01),   # gating: exempt from the floor AND the ranking
}

_CACHE: dict = {}


def api(obj, name):
    """``getattr`` as an ASSERTION that the surface exists at all, so a package
    without the fix fails on the CLAIM rather than on an AttributeError that reads
    like a broken test."""
    assert hasattr(obj, name), (
        f"{getattr(obj, '__name__', type(obj).__name__)} has no {name!r}: the "
        "proposer is not bounded by BASE_STATE_RATIO_MIN anywhere, which is the "
        "defect this file is about"
    )
    return getattr(obj, name)


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


def telegraph(dwell_s, duty, *, dur=15.0, seed=7, depth=110.0, i0=-34.8,
              noise=4.84, fs=FS):
    """A two-level pore at a chosen state length and occupancy. Same generator as
    ``tests/analysis/test_baseline_state_timescale.py`` so the two files are talking
    about the same fixture; ``dur`` is longer here because the probe needs at least
    `baseline.REFERENCE_WINDOWS_MIN` reference windows before it reports anything."""
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    x = np.full(n, float(i0))
    spans, t = [], 0.0
    gap = dwell_s * (1.0 - duty) / duty
    while True:
        t += rng.exponential(gap)
        d = rng.gamma(4.0, dwell_s / 4.0)
        s, e = int(round(t * fs)), int(round((t + d) * fs))
        if e >= n:
            break
        if spans and s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
        t += d
    for s, e in spans:
        x[s:e] -= depth
    x += rng.normal(0.0, noise, n)
    return x.astype(np.float32), [(s, e) for s, e in spans if e > s]


# ===========================================================================
# 1. H1 -- the proposer and the check divide by the SAME number
# ===========================================================================
def test_the_probe_reports_the_number_base_vs_state_divides_by():
    """H1 + H3. The whole defect in one sentence: the check measured the MEDIAN state
    and the proposer had only the p99.9, so the proposer could not have obeyed the
    check even in principle.

    Asserted on a synthetic rather than on a recording so it runs everywhere: a 20 ms
    telegraph at 12% duty. `BaselineProbe.state_dwell_s` must be the median (~20 ms)
    and NOT `longest_state_s`, and the two must be far enough apart that a test
    passing on one would fail on the other.
    """
    x, _spans = telegraph(0.020, 0.12)
    probe = P.probe_timescales(x, FS, polarity=-1)
    d = api(probe, "state_dwell_s")
    assert d is not None and np.isfinite(d), (
        f"the probe reports no median state dwell ({d!r}), so nothing it proposes "
        "can be bounded by BASE_STATE_RATIO_MIN"
    )
    assert 0.010 <= d <= 0.040, f"median state {d * 1e3:.1f} ms for a 20 ms telegraph"
    assert probe.longest_state_s > d * 1.5, (
        "state_dwell_s is being read off the same statistic as longest_state_s; the "
        "check divides by the median and the proposer sized on the p99.9, and that "
        "gap IS the defect", probe.longest_state_s, d)


@pytest.mark.corpus
@pytest.mark.parametrize("tag", ["apo-700", "ferritin_data2"])
def test_the_probes_state_dwell_is_bit_identical_to_the_checks_divisor(tag):
    """H1, at the seam. Not "close to" and not "correlates with": the health block
    divides ``base_s`` by ``state_dwell_s`` over `reference_states`' mask, and
    `reference_state_dwell_s` must be that same measurement to the last bit -- a proxy
    would let the floor land under the bound it is enforcing.

    Both directions are checked: the probe's field, and the ratio the rendered check
    reports.
    """
    x, fs = _apo(-700) if tag == "apo-700" else _corpus("ferritin_data2")
    probe = P.probe_timescales(x, fs, polarity=-1)
    proposal = P.propose_baseline(probe)
    health = B.estimate_baseline_result(
        x, fs, config=proposal.config, polarity=-1).health

    divisor = api(health, "state_dwell_s")
    mine = api(probe, "state_dwell_s")
    assert float(mine) == float(divisor), (
        "the proposer is dividing by a DIFFERENT number from the one the check "
        "divides by", mine, divisor)
    # and the ratio the user is shown falls straight out of it
    assert float(proposal.config.base_s) / float(mine) == pytest.approx(
        float(health.base_state_ratio), rel=1e-12)


# ===========================================================================
# 2. H2 -- a floor snapped to the nearest grid point is not a floor
# ===========================================================================
def test_snap_up_never_lands_below_what_it_was_given():
    """H2. THE PROPERTY, over the whole grid and three decades between it: the result
    is a grid point, it is >= the input, and no smaller grid point is.

    `snap_to_grid` fails this by construction and that is the point of the function
    existing -- see the next test for the recording it cost.
    """
    snap_up = api(P, "snap_up_to_grid")
    grid = sorted({float(f"{m:g}e{d}") for d in range(-4, 3) for m in P.GRID_MANTISSAS
                   if P.GRID_MIN_S - 1e-12 <= float(f"{m:g}e{d}") <= P.GRID_MAX_S + 1e-12})
    probes = sorted(set(grid) | {g * f for g in grid for f in (0.51, 0.99, 1.01, 1.7)})
    for v in probes:
        if not (P.GRID_MIN_S < v < P.GRID_MAX_S):
            continue
        got = snap_up(v)
        assert got in grid, (v, got)
        assert got >= v * (1 - 1e-9), f"snap_up({v}) = {got} is BELOW its input"
        smaller = [g for g in grid if g < got and g >= v * (1 - 1e-9)]
        assert not smaller, (v, got, smaller)
    for g in grid:
        assert snap_up(g) == g, f"a grid point moved: {g} -> {snap_up(g)}"


def test_the_nearest_grid_point_would_still_have_been_condemned():
    """H2, with the number that forced it. On the -700 mV recording the bound is
    ``3.0 x 21.71 ms = 65.13 ms``. Log-NEAREST puts that at 0.05 s, where
    ``base_vs_state`` reads **2.3031 and still condemns** (measured through the
    estimator, all eight checks: 0.2/0.05 s scores 2.3030861903313187 and an intrusion
    of 0.0272). Snapping UP puts it at 0.1 s -- 4.6062 and 0.0043, all green.

    Arithmetic only, so it runs without the recording; the estimator numbers above are
    reproduced by `test_the_seven_hundred_millivolt_file_is_proposed_a_certified_window`.
    """
    want = B.BASE_STATE_RATIO_MIN * 0.021709999482393265
    assert P.snap_to_grid(want) == 0.05 and 0.05 < want * (1 / B.BASE_STATE_RATIO_MIN) * 3
    assert P.snap_to_grid(want) < want, "the premise moved: nearest is no longer below"
    assert api(P, "snap_up_to_grid")(want) == 0.1
    assert 0.05 / 0.021709999482393265 < B.BASE_STATE_RATIO_MIN, (
        "log-nearest lands on a window base_vs_state condemns")
    assert 0.1 / 0.021709999482393265 > B.BASE_STATE_RATIO_MIN


# ===========================================================================
# 3. H3/H4 -- the floor, and the exemption that is the same clause
# ===========================================================================
def test_the_floor_is_the_checks_own_bound_snapped_up():
    """H3. `base_floor_s` is ONE function because two callers need one answer -- the
    rule that applies it and the summary line that reports it."""
    x, _ = telegraph(0.020, 0.12)
    probe = P.probe_timescales(x, FS, polarity=-1)
    floor = api(P, "base_floor_s")(probe)
    assert floor is not None
    assert floor == P.snap_up_to_grid(B.BASE_STATE_RATIO_MIN * probe.state_dwell_s)
    assert floor >= B.BASE_STATE_RATIO_MIN * probe.state_dwell_s


def test_no_probe_and_an_unusable_probe_have_no_floor():
    """H3. There is nothing to bound when the proposal is `DEFAULT_BASELINE` by
    identity, and a floor invented there would be a number with nothing behind it."""
    floor_of = api(P, "base_floor_s")
    assert floor_of(None) is None
    short = P.probe_timescales(np.zeros(20_000) - 34.8 + np.random.default_rng(1)
                               .normal(0, 4.8, 20_000), FS)
    assert short.usable is False
    assert floor_of(short) is None


@pytest.mark.parametrize("dwell_ms,duty", [(5, 0.05), (5, 0.12), (20, 0.05),
                                           (20, 0.12), (60, 0.05), (60, 0.12)])
def test_the_proposal_is_never_one_base_vs_state_condemns(dwell_ms, duty):
    """**THE LOAD-BEARING TEST**, and the invariant stated as a property rather than
    as one recording: over six synthetic pores spanning a 12x range of state length,
    what `propose_baseline` offers is never a window `baseline_health`'s
    ``base_vs_state`` condemns.

    This is the assertion the -700 mV file failed for the whole life of the proposer.
    """
    x, _spans = telegraph(dwell_ms * 1e-3, duty)
    proposal = P.probe_and_propose(x, FS, polarity=-1)
    health = B.estimate_baseline_result(
        x, FS, config=proposal.config, polarity=-1).health
    c = health.check("base_vs_state")
    assert c is not None, "no state-timescale check ran; this test proves nothing"
    assert c.level == "ok", (
        f"the proposer offered {proposal.config.describe()} and its own health block "
        f"condemns it: base_vs_state = {c.value} against {c.threshold}"
    )
    if c.value is not None:
        assert c.value >= B.BASE_STATE_RATIO_MIN


def test_the_floor_is_only_a_guarantee_while_the_check_reads_the_reference_mask():
    """**THE HOLE IN MY OWN FIX, pinned so it cannot be read as folklore.**

    `base_floor_s` divides by `baseline.reference_state_dwell_s`, which is always
    measured at `baseline.reference_rough_s`. ``base_vs_state`` normally divides by
    the same thing -- but when the CONFIGURED ``rough_s`` reaches that window,
    `baseline.reference_states` returns ``None`` and the check measures the state
    dwell on the configured pass instead. Different mask, different number, and the
    floor was built on the other one.

    Measured here: 3 s of a 120 ms-state pore, reference window 0.75 s. The floor
    lifts ``base_s`` 0.1 -> 0.3 s and carries ``rough_s`` to 1.0, past the reference
    window; the configured pass then reads a ~109 ms state where the reference pass
    read ~77 ms, and ``base_vs_state`` still warns at ~2.75 against 3.0.

    It is an IMPROVEMENT and not a repair: without the floor the same recording is
    proposed at 1.29x. This test exists so that a later reader of "no longer
    proposable" finds the exception, and so that anyone who closes it has to delete
    a named test rather than quietly widen a claim.
    """
    x, _spans = telegraph(0.120, 0.20, dur=3.0)
    probe = P.probe_timescales(x, FS, polarity=-1)
    floor = api(P, "base_floor_s")(probe)
    assert floor is not None
    proposal = P.propose_baseline(probe)
    assert float(proposal.config.base_s) == floor, "the floor did not bind here"

    w = B.reference_rough_s(x.size, FS)
    assert float(proposal.config.rough_s) >= w, (
        "the premise moved: the proposal no longer passes the reference window",
        proposal.config.rough_s, w)
    assert B.reference_states(x, FS, config=proposal.config) is None

    health = B.estimate_baseline_result(
        x, FS, config=proposal.config, polarity=-1).health
    c = health.check("base_vs_state")
    assert c is not None and c.level == "warn", (
        "the known exception has closed -- good, but then this test and the two "
        "docstrings that name it are stale", c)
    assert c.value == pytest.approx(2.75, abs=0.3), c.value
    assert float(health.state_dwell_s) != float(probe.state_dwell_s), (
        "the two are measuring the same mask after all, so the exception is not "
        "what these docstrings say it is")

    # ...and the remedy on screen must NOT send the user to the probe here.
    assert "OR PRESS PROBE AGAIN" not in c.message, c.message
    assert "is not guaranteed to clear this one" in c.message, c.message

    # the improvement, measured against the unfloored rule on the same trace
    unfloored = B.DEFAULT_BASELINE.replace(
        rough_s=P.snap_to_grid(P.ROUGH_MULTIPLE * probe.longest_state_s),
        base_s=P.snap_to_grid(P.snap_to_grid(P.ROUGH_MULTIPLE * probe.longest_state_s)
                              / P.ROUGH_BASE_RATIO), decim=None)
    before = B.estimate_baseline_result(
        x, FS, config=unfloored, polarity=-1).health.base_state_ratio
    assert before < c.value, (before, c.value)


def test_the_floor_on_a_trace_with_no_events_is_noise_and_still_helps():
    """The second thing the floor is not. On 60 s of pure Gaussian noise there are no
    states, so `baseline.state_dwell_s` measures the dilated NOISE runs -- for the
    check and for the floor alike, which is the contract. What says the trace is empty
    is ``mask_fraction_floor``, and it fires.

    Recorded because a floor built on noise is a real thing to know about, and because
    the direction is still right: without it this trace is proposed a rough window
    EQUAL to its base window.
    """
    x = (-34.8 + np.random.default_rng(2).normal(0, 4.8, 3_000_000)).astype(np.float32)
    probe = P.probe_timescales(x, FS, polarity=-1)
    proposal = P.propose_baseline(probe)
    health = B.estimate_baseline_result(
        x, FS, config=proposal.config, polarity=-1).health
    assert "mask_fraction_floor" in [c.name for c in health.checks if not c.ok]
    unfloored_rough = P.snap_to_grid(P.ROUGH_MULTIPLE * probe.longest_state_s)
    unfloored_base = P.snap_to_grid(unfloored_rough / P.ROUGH_BASE_RATIO)
    assert unfloored_rough == unfloored_base, (
        "the premise moved", unfloored_rough, unfloored_base)
    assert proposal.config.rough_s > proposal.config.base_s


def test_the_probe_report_stays_strict_json():
    """`BaselineProbe.to_dict` promises "flat and JSON-safe" and the floor added two
    keys to it. ``nan`` is not JSON, so ``allow_nan=False`` is the test."""
    import json
    for x in (telegraph(0.020, 0.12)[0],
              (-34.8 + np.random.default_rng(5).normal(0, 4.8, 1_500_000)
               ).astype(np.float32)):
        d = P.probe_timescales(x, FS, polarity=-1).to_dict()
        assert "probe_state_dwell_s" in d and "probe_base_floor_s" in d
        json.dumps(d, allow_nan=False)


def test_a_gating_pore_is_exempt_here_exactly_as_it_is_in_the_check():
    """H4. Not an exception to the contract -- the same clause of it.
    ``base_vs_state`` writes ``short = (not gating) and ...``; `base_floor_s` returns
    ``None`` on `BaselineProbe.gating_credible`, which is the same predicate. On a
    stepping pore the long states ARE baseline levels and ``base_s`` is SUPPOSED to be
    shorter than they are.
    """
    fs, n = FS, 1_500_000
    rng = np.random.default_rng(3)
    x = np.full(n, -34.8)
    k = int(0.035 * fs)                       # ~35 ms level dwell
    lvl = 0.0
    for s in range(0, n, k):
        lvl = -296.0 if lvl == 0.0 else 0.0
        x[s:s + k] += lvl
    x += rng.normal(0.0, 13.8, n)
    probe = P.probe_timescales(x.astype(np.float32), fs, polarity=-1)
    if not probe.gating_credible:
        pytest.skip("the gating fixture did not read as a stepping pore here")
    assert api(P, "base_floor_s")(probe) is None, (
        "a floor was applied on a stepping pore; on ferritin_data3 that moves the "
        "shipped 0.03 / 0.01 s to 0.1 / 0.02 and breaks the corpus contract")


# ===========================================================================
# 4. THE CORPUS CONTRACT -- exactly one proposal moves
# ===========================================================================
@pytest.mark.corpus
@pytest.mark.parametrize("tag", sorted(EXPECTED))
def test_exactly_the_seven_hundred_millivolt_proposal_moves(tag):
    """H4, against the six recordings in evidence. Five must be byte-identical to what
    the proposer offered before the floor existed; -700 mV is the one that is
    chartered to move, and it moves onto a window all eight checks certify.

    ferritin_data3 is the tight one and it is held by the gating exemption rather than
    by slack: it reads ``base_vs_state`` **2.9940119760479043** against a threshold of
    3.0, 0.2% inside. A floor that reached it would move it.
    """
    if tag.startswith("apo"):
        x, fs = _apo(int(tag[3:]))
    else:
        x, fs = _corpus(tag)
    proposal = P.probe_and_propose(x, fs, polarity=-1)
    got = (float(proposal.config.rough_s), float(proposal.config.base_s))
    assert got == pytest.approx(EXPECTED[tag]), proposal.config.describe()
    assert proposal.param_source == "proposed"

    health = B.estimate_baseline_result(
        x, fs, config=proposal.config, polarity=-1).health
    assert health.ok, [c.name for c in health.checks if not c.ok]


@pytest.mark.corpus
def test_the_seven_hundred_millivolt_file_is_proposed_a_certified_window():
    """THE CHARTERED CHANGE, with the numbers on both sides of it.

    Before: 0.1 / 0.03 s, ``base_vs_state`` 1.3819 and ``baseline_intrusion`` 0.0416,
    health NOT ok, two checks condemning. After: 0.5 / 0.1 s, 4.6062 and 0.0043, all
    eight green. The intermediate 0.2 / 0.05 -- where a log-NEAREST floor would have
    landed -- is asserted too, because it is still condemned and is the reason
    `snap_up_to_grid` exists.
    """
    x, fs = _apo(-700)
    proposal = P.probe_and_propose(x, fs, polarity=-1)
    assert (float(proposal.config.rough_s),
            float(proposal.config.base_s)) == pytest.approx((1.0, 0.3))

    def health_at(rough, base):
        return B.estimate_baseline_result(
            x, fs, config=B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base,
                                                     decim=None),
            polarity=-1).health

    now = health_at(0.5, 0.1)
    assert now.ok and not [c.name for c in now.checks if not c.ok]
    assert now.base_state_ratio == pytest.approx(4.606, abs=0.01)
    assert now.intrusion_fraction == pytest.approx(0.0043, abs=0.002)

    was = health_at(0.1, 0.03)
    assert not was.ok
    assert {c.name for c in was.checks if not c.ok} == {"base_vs_state",
                                                        "baseline_intrusion"}
    assert was.base_state_ratio == pytest.approx(1.382, abs=0.01)

    nearest = health_at(0.2, 0.05)
    assert not nearest.ok, (
        "the premise of snap_up_to_grid has moved: log-nearest now certifies")
    assert nearest.base_state_ratio == pytest.approx(2.303, abs=0.01)


@pytest.mark.corpus
def test_the_probe_still_costs_what_it_is_budgeted_to_cost():
    """The floor buys its number with one more decimated median at the reference
    window. Measured 0.376 s of the -700 mV probe on this machine; the guard is loose
    on purpose (a shared machine, and a probe is off-thread), and exists to catch the
    order of magnitude rather than the second decimal."""
    x, fs = _apo(-700)
    probe = P.probe_timescales(x, fs, polarity=-1)
    assert probe.elapsed_s < 15.0, probe.elapsed_s


# ===========================================================================
# 5. STRUCTURE -- and where `nanoscope` came from
# ===========================================================================
def test_the_proposer_still_touches_only_the_three_fields_it_may():
    """The floor must not have become a fourth thing the probe decides. Every config
    a proposal carries is ``DEFAULT_BASELINE.replace(rough_s, base_s, decim)``, and
    ``mask_sigma`` / ``dilate_ms`` / ``window_s`` / ``despike_size`` / ``method`` are
    what `reference_state_dwell_s` measured its own reference mask with -- if the
    proposer could move them, the floor would be measured against a mask the run does
    not use."""
    d = B.DEFAULT_BASELINE
    for x, pol in ((telegraph(0.020, 0.12)[0], -1), (telegraph(0.005, 0.05)[0], -1)):
        cfg = P.probe_and_propose(x, FS, polarity=pol).config
        assert cfg.method == d.method
        assert cfg.mask_sigma == d.mask_sigma
        assert cfg.dilate_ms == d.dilate_ms
        assert cfg.window_s == d.window_s
        assert cfg.despike_size == d.despike_size
        assert cfg.decim is None


def test_nanoscope_resolved_from_the_tree_under_test():
    """A revert run from inside the repo silently tests the LIVE tree: ``PYTHONPATH``
    does not beat cwd when cwd contains the package. Every run of this file records
    where the package came from, so a green revert can be told from a green fix."""
    import nanoscope
    where = pathlib.Path(nanoscope.__file__).resolve()
    print(f"\nnanoscope resolved from: {where}")
    assert where.name == "__init__.py"
    assert (where.parent / "analysis" / "baseline_probe.py").exists()
