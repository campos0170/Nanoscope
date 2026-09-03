"""The second occupancy measurement, and the comparison that was missing.

THE DEFECT (hunt 3, C-2)
=======================
At a routine higher concentration -- the owner's own experimental parameters with
only the arrival rate varied -- the probe proposed ``rough_s`` 0.03 s for a recording
whose events are 18 ms long. A rough window shorter than the events FOLLOWS them
instead of exposing them, so the residual inside a blockade goes to zero, the mask
finds almost nothing, and the baseline lands inside the events. Measured: median
event depth **44.5 pA against a true 110**, event count inflated 23% -- with
``BaselineHealth.ok`` True and **zero warnings**, while the CORRECT configuration
(2.0/0.5, 110.17 pA) was the one that got a warning. The health block was exactly
inverted, so the app steered the user away from the right answer.

WHAT WAS ACTUALLY MISSING, AND WHERE IT IS NOW
==============================================
Not `baseline_probe.GATING_THRESHOLD`: raising that only moves the occupancy at which
the misfire starts. What made the wrong number SILENT is that the app held TWO
occupancy measurements -- the probe's 0.3535 (the true occupancy to four digits) and
the run's own 0.0991 -- and compared them nowhere. Every mask check compared against
a CONSTANT (`MASK_FRACTION_FLOOR`, `MASK_FRACTION_CEILING`), and a run can sit
squarely inside that band with three quarters of its events missing.

So the comparison is now an invariant of the health block itself
(`baseline.reference_mask_fraction` + the ``mask_reference`` check), not a value
threaded in from whoever happened to have run a probe: it fires the same on a config
typed by hand, restored from a project file or applied from a proposal, because it
re-measures the reference rather than being handed one.

THE LOAD-BEARING TEST IS NOT "DOES THE FLAG FIRE"
================================================
It is :func:`test_health_ok_never_certifies_depths_that_are_wrong` -- *does* ``ok``
*ever read True on an estimate whose per-event depth is more than a few percent off*
-- swept over occupancy and dwell/window ratio rather than asserted at one point. No
test in the tree asserted that, which is why this survived two hunts.

THE BOUND, AND THE SEVEN TESTS THAT PASS EITHER WAY
===================================================
This cannot fire on any recording the owner has on disk (his apo file scores 0.472
against a gating threshold of 8.0). Fifteen of the tests here fail against a
programmatic revert of the fix, each on its own assertion. Seven pass both, all of
them labelled CONTROL in their own docstrings, and they are the ones that would catch
this change doing harm: the four corpus/apo bound tests are written with **no new
API** so the identical assertions run against both packages, and the other three pin
the premises the fix rests on -- that no baseline-error metric could have seen the
defect, that a genuinely stepping pore still takes the level branch, and that the
floor case is still left to the floor check.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P

FS = 50_000.0
I0 = -34.8
DEPTH = 110.0
NOISE = 4.84

CORPUS = pathlib.Path("/Users/abrahamcampos/Desktop/Research-Data/FerritinModeler/CSVData")
APO = pathlib.Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")


# ===========================================================================
# the fixture: a renewal process at the owner's own parameters
# ===========================================================================
def renewal(rate, *, seed=3, dur=20.0, dwell_s=0.018, depth=DEPTH, i0=I0,
            noise=NOISE, fs=FS):
    """``(trace, true spans, occupancy)``. Exponential gaps AND exponential dwells.

    Poisson arrivals rather than a period, because a period that is a multiple of the
    probe's 20 ms gating block makes the block medians alternate deterministically and
    the fixture then tests the fixture. Overlapping arrivals are merged, which is what
    a pore does.
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    x = np.full(n, float(i0))
    spans, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        d = rng.exponential(dwell_s)
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
    spans = [(s, e) for s, e in spans if e > s]
    return x, spans, sum(e - s for s, e in spans) / n


def measured_depth(x, baseline, spans):
    """The number the user reads: median over the true events of ``baseline - x``."""
    d = np.asarray(baseline) - np.asarray(x)
    return float(np.median([np.median(d[s:e]) for s, e in spans]))


def cfg(rough, base=None):
    return B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base or rough / 4.0,
                                      decim=None)


def health_of(x, config, fs=FS):
    return B.estimate_baseline_result(x, fs, config=config, polarity=-1).health


def api(obj, name):
    """``getattr``, as an ASSERTION about the cross-check existing at all.

    Every test below reaches for the new surface through this, so a package without
    the cross-check fails on the claim being made rather than on an ``AttributeError``
    that reads like a broken test.
    """
    assert hasattr(obj, name), (
        f"{getattr(obj, '__name__', type(obj).__name__)} has no {name!r}: the second "
        "occupancy measurement -- the one the run's mask fraction has to be compared "
        "against -- is not there, which is the C-2 defect"
    )
    return getattr(obj, name)


def reference_of(h):
    """The reference reading, with the assertion that it exists at all.

    Spelled as an assertion rather than an attribute access so that a package without
    the cross-check fails these tests on their own claim instead of on an
    ``AttributeError`` that reads like a broken test.
    """
    names = {f.name for f in dataclasses.fields(type(h))}
    assert {"mask_fraction_reference", "mask_reference_ratio",
            "mask_reference_rough_s"} <= names, (
        "BaselineHealth carries no reference occupancy: the run's own mask fraction "
        "is being compared against constants only, which is the C-2 defect"
    )
    return h.mask_fraction_reference, h.mask_reference_ratio, h.mask_reference_rough_s


# ===========================================================================
# 1. THE PROPERTY -- the one that was missing
# ===========================================================================
#: (mean dwell ms, arrival rate /s). Chosen to sweep occupancy 0.09..0.35 and the
#: dwell-to-window ratio across three decades, which is the plane the failure lives
#: in -- not a single reproduction of the reported case.
_SWEEP = [(18, 30), (18, 20), (18, 5), (40, 8), (40, 3), (80, 5), (6, 20)]


def test_health_ok_never_certifies_depths_that_are_wrong():
    """**The load-bearing test.** `BaselineHealth.ok` is a certificate, and a
    certificate that can be issued for a 0.6%-of-truth depth certifies nothing.

    Swept over 7 (dwell, rate) pairs x 4 rough windows. BEFORE the cross-check
    existed, 6 of those 28 estimates read ``ok`` with per-event depths of 0.005x to
    0.06x the truth -- every check passing, no warning anywhere, because every mask
    check compared against a constant. AFTER, the worst depth surviving ``ok`` is
    0.985x.

    The tolerance is 0.95, deliberately looser than the 0.985 measured: this test
    exists to catch a certificate issued for a HALVED depth, and pinning it at the
    measured value would turn a 1% shift in the fixture's RNG into a failure that
    says nothing. What must never move is the direction -- see the second assertion,
    which is the one that fails without the cross-check.
    """
    certified, worst = [], (1.0, None)
    for dwell_ms, rate in _SWEEP:
        x, spans, occ = renewal(rate, dwell_s=dwell_ms * 1e-3)
        assert 0.05 < occ < 0.45, f"fixture drifted: occupancy {occ}"
        for rough in (0.03, 0.1, 0.5, 2.0):
            res = B.estimate_baseline_result(x, FS, config=cfg(rough), polarity=-1)
            ratio = measured_depth(x, res.baseline, spans) / DEPTH
            if res.health.ok:
                certified.append((dwell_ms, rate, rough, round(ratio, 4)))
                if ratio < worst[0]:
                    worst = (ratio, (dwell_ms, rate, rough))

    bad = [c for c in certified if c[3] < 0.95]
    assert not bad, (
        "health.ok is True on estimates whose per-event depth is wrong by more than "
        f"5%: {bad}. Every one of these has a mask fraction inside "
        f"[{B.MASK_FRACTION_FLOOR}, {B.MASK_FRACTION_CEILING}] and no other check "
        "objects -- the missing measurement is this recording's own occupancy, "
        "measured over a window wide enough to see over its events."
    )
    assert len(certified) >= 8, f"the sweep certified almost nothing: {certified}"
    assert worst[0] >= 0.95, worst


# ===========================================================================
# 2. THE REPORTED CASE, END TO END
# ===========================================================================
def test_the_thirty_five_percent_occupancy_case_is_no_longer_certified():
    """The measurement in the report, reproduced: 35% occupancy, 18 ms dwells, the
    probe's own proposal applied. Before: depth 0.07x of truth, ``ok`` True, zero
    warnings. The mask fraction is 9.8% -- comfortably inside the band -- and it is
    the reference at 39.1% that says the window is inside the events."""
    x, spans, occ = renewal(30, dur=60.0)
    assert occ == pytest.approx(0.354, abs=0.01)

    h = health_of(x, cfg(0.03, 0.01))
    ref, ratio, ref_rough = reference_of(h)
    assert B.MASK_FRACTION_FLOOR < h.mask_fraction < B.MASK_FRACTION_CEILING, (
        "the whole point is that the ABSOLUTE band is satisfied: "
        f"{h.mask_fraction}"
    )
    assert ref == pytest.approx(occ, abs=0.05), (ref, occ)
    assert ratio > api(B, "MASK_REFERENCE_MAX_RATIO")
    assert ref_rough == B.REFERENCE_ROUGH_S
    assert not h.ok, "a 7.8 pA reading of a 110 pA event may not be certified healthy"
    assert h.check("mask_reference").level == "warn"
    # When this test was written this list was exactly ``["mask_reference"]`` -- no
    # other check could see the defect, which is why it needed its own. The two
    # state-timescale checks added afterwards (`BASE_STATE_RATIO_MIN`,
    # `BASELINE_INTRUSION_MAX`) also fire here, and that is not a weakening of the
    # claim: at rough 0.03 / base 0.01 this fixture's baseline is inside the events
    # by 98% of their depth, so a check that measures intrusion is SUPPOSED to say
    # so. What still holds -- and is what the C-2 fix is about -- is that
    # ``mask_reference`` is the only one of the SIX ORIGINAL checks that sees it.
    fired = [c.name for c in h.checks if not c.ok]
    assert "mask_reference" in fired
    # EIGHT NAMES, SIX CHECKS. ``mask_fraction`` has three spellings -- floor,
    # ceiling and in-band -- and exactly one of them is ever emitted, so this set
    # names the six checks that existed before ``mask_reference``, plus itself.
    _ORIGINAL_CHECK_NAMES = {"length_tracking", "length_rough", "mask_fraction_floor",
                             "mask_fraction_ceiling", "mask_fraction", "mask_reference",
                             "rough_sigma_ratio", "overshoot"}
    assert [c for c in fired if c in _ORIGINAL_CHECK_NAMES] == ["mask_reference"], (
        "no check that existed before the cross-check sees this -- that is why it "
        f"needed its own; fired = {fired}"
    )
    # ...and the complement is pinned too, so this does not quietly become a test
    # that any future check may fire on this fixture. The two state-timescale checks
    # SHOULD fire here (the baseline is 98% inside the events); nothing else may
    # appear without someone editing this line and saying why.
    assert set(fired) - _ORIGINAL_CHECK_NAMES <= {"base_vs_state", "baseline_intrusion"}, (
        f"an unexpected check fires on the 35%-occupancy fixture: fired = {fired}"
    )

    res = B.estimate_baseline_result(x, FS, config=cfg(0.03, 0.01), polarity=-1)
    assert measured_depth(x, res.baseline, spans) / DEPTH < 0.5


def test_only_the_per_event_depth_can_see_it_so_no_absolute_metric_could():
    """CONTROL, PASSES BOTH -- WHY the fix had to be a comparison and not a threshold.

    On the wrong estimate the BASELINE is still within ~1.5 pA of the true open pore,
    because two thirds of the samples genuinely are open pore. Every scalar the app
    reports about the baseline itself -- level, drift, noise, overshoot -- is
    therefore fine. Only the per-event depth is destroyed, and only a second
    occupancy measurement can reach it."""
    x, spans, _ = renewal(30, dur=60.0)
    wrong = B.estimate_baseline_result(x, FS, config=cfg(0.03, 0.01), polarity=-1)
    right = B.estimate_baseline_result(x, FS, config=cfg(2.0, 0.5), polarity=-1)

    assert abs(float(np.median(wrong.baseline)) - I0) < 2.0
    assert abs(float(np.median(right.baseline)) - I0) < 2.0
    assert measured_depth(x, right.baseline, spans) == pytest.approx(DEPTH, rel=0.02)
    assert measured_depth(x, wrong.baseline, spans) < 0.5 * DEPTH
    assert wrong.health.overshoot < B.OVERSHOOT_FLAG, (
        "overshoot -- the one published QC scalar that resists a too-short window -- "
        "does not resist THIS one"
    )


# ===========================================================================
# 3. THE COMPARISON IS A COMPARISON OF THE SAME THING
# ===========================================================================
def test_the_two_fractions_differ_only_by_the_window():
    """A ratio between two masks measured with different thresholds would be a
    measurement of the thresholds. ``mask_sigma``, ``dilate_ms`` and ``despike_size``
    are the caller's own on both sides; only ``rough_s`` (and the throttle derived
    from it) moves."""
    x, _, _ = renewal(20, dwell_s=0.018)
    tight = B.DEFAULT_BASELINE.replace(rough_s=0.1, base_s=0.025, decim=None,
                                       mask_sigma=5.0, dilate_ms=0.2)
    got = api(B, "reference_mask_fraction")(x, FS, config=tight)
    assert got is not None
    frac, w = got
    _, mask, _, _ = B._rough_pass(x, FS, config=tight.replace(
        rough_s=w, base_s=w / api(B, "ROUGH_BASE_RATIO"), decim=None))
    assert frac == pytest.approx(float(mask.mean()), abs=1e-12)
    loose = B.reference_mask_fraction(x, FS, config=tight.replace(mask_sigma=3.0))[0]
    assert loose > frac, "a looser threshold must move BOTH sides, not one"


def test_the_reference_is_not_measured_when_it_would_not_be_wider():
    """A config already running at or above the reference window IS the wide
    measurement; comparing it with itself would be a tautology reported as a check."""
    x, _, _ = renewal(20)
    wide = api(B, "REFERENCE_ROUGH_S")
    assert B.reference_mask_fraction(x, FS, config=cfg(wide)) is None
    h = health_of(x, cfg(wide, 1.0))
    ref, ratio, _ = reference_of(h)
    assert ref is None and ratio is None
    assert h.check("mask_reference") is None


def test_a_short_recording_says_not_measured_rather_than_guessing():
    """Below `REFERENCE_WINDOWS_MIN` x `REFERENCE_MIN_ROUGH_S` = 2 s there is no
    window wide enough to see over a state worth sizing anything against, and a
    reference measured on a shrunken window would be the recording's length rather
    than the pore's timescale. The check reports NOT MEASURED; it must never report
    agreement it did not measure."""
    x = (I0 + np.random.default_rng(5).normal(0, NOISE, 50_000))     # 1 s
    assert api(B, "reference_rough_s")(x.size, FS) < B.REFERENCE_MIN_ROUGH_S
    assert B.reference_mask_fraction(x, FS, config=B.DEFAULT_BASELINE) is None
    h = health_of(x, B.DEFAULT_BASELINE)
    assert reference_of(h) == (None, None, None)
    assert h.check("mask_reference") is None


def test_the_floor_case_is_left_to_the_floor_check():
    """PASSES BOTH -- a CONTROL, and the non-regression the new check could most
    plausibly have broken.

    An event-free trace and a 60%-duty inversion both mask ~nothing. A ratio against
    ~zero is arithmetic, not evidence, and the floor check's message already tells a
    user the two opposite things it can mean. The two traces must still fire exactly
    the same two checks -- the property
    ``test_the_inversion_and_the_empty_trace_are_not_told_apart_by_the_checks``
    pins in ``test_masked_baseline.py``.
    """
    rng = np.random.default_rng(3)
    empty = (-100.0 + rng.normal(0, 5.0, 1_000_000)).astype(np.float32)
    busy = (-100.0 + rng.normal(0, 5.0, 1_000_000)).astype(np.float32)
    for s in range(0, busy.size - 6_000, 10_000):
        busy[s:s + 6_000] -= 200.0

    fired = lambda h: sorted(c.name for c in h.checks if not c.ok)
    h_empty = B.estimate_baseline_result(empty, FS, polarity=-1).health
    h_busy = B.estimate_baseline_result(busy, FS, polarity=-1).health
    assert fired(h_empty) == fired(h_busy) == ["mask_fraction_floor", "overshoot"]
    assert h_empty.check("mask_reference") is None
    assert h_busy.check("mask_reference") is None


def test_the_verdict_survives_the_decimation_it_is_measured_at():
    """The reference's decimation is DERIVED from the reference window, not carried
    over from a config whose windows were 200x narrower. Carrying it costs 25 000
    taps on a 10 ms ``base_s`` (>120 s on ferritin_data3, against 62 ms re-derived),
    and the fraction itself is genuinely decim-dependent on a stepping pore. What has
    to be stable is the VERDICT, and it is: 3.7x..4.0x across a 64x decim range on
    the case that matters."""
    x, _, _ = renewal(30, dur=60.0)
    conf = cfg(0.03, 0.01)
    _, mask, _, _ = B._rough_pass(x, FS, config=conf)
    run = float(mask.mean())
    w = api(B, "reference_rough_s")(x.size, FS)
    base = B.derive_decim(w / B.ROUGH_BASE_RATIO, FS)
    ratios = []
    for mult in (0.25, 1, 4, 16):
        ref_cfg = conf.replace(rough_s=w, base_s=w / B.ROUGH_BASE_RATIO,
                               decim=int(max(1, base * mult)))
        _, m, _, _ = B._rough_pass(x, FS, config=ref_cfg)
        ratios.append(float(m.mean()) / run)
    assert min(ratios) > api(B, "MASK_REFERENCE_MAX_RATIO"), ratios
    assert max(ratios) / min(ratios) < 1.5, ratios


# ===========================================================================
# 4. ONE SCALE, WRITTEN ONCE
# ===========================================================================
def test_the_probe_and_the_health_block_measure_at_one_window():
    """The two occupancies are compared against each other. Two constants that are
    meant to be comparable, spelled in two modules, is how the comparison quietly
    becomes a comparison of the two window choices instead."""
    assert P.PROBE_ROUGH_S is api(B, "REFERENCE_ROUGH_S")
    assert P.PROBE_MIN_ROUGH_S is B.REFERENCE_MIN_ROUGH_S
    assert P.PROBE_WINDOWS_MIN is B.REFERENCE_WINDOWS_MIN
    assert P.ROUGH_BASE_RATIO is B.ROUGH_BASE_RATIO

    shared = {"PROBE_ROUGH_S", "PROBE_MIN_ROUGH_S", "PROBE_WINDOWS_MIN",
              "ROUGH_BASE_RATIO"}
    tree = ast.parse(pathlib.Path(P.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in shared:
                    assert isinstance(node.value, ast.Name), (
                        f"{t.id} is spelled as a literal in baseline_probe.py; it must "
                        "BE baseline's constant, not a copy of its value"
                    )


def test_the_sizing_formula_is_written_once():
    """`reference_rough_s` is the formula. A second spelling of
    ``min(5.0, duration / 4)`` is how the probe's window and the health block's
    reference start disagreeing by a factor nobody notices."""
    fns = [n for n in ast.walk(ast.parse(pathlib.Path(P.__file__).read_text()))
           if isinstance(n, ast.FunctionDef) and n.name == "probe_timescales"]
    assert len(fns) == 1
    called = {n.func.id for n in ast.walk(fns[0])
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "reference_rough_s" in called, (
        "probe_timescales sizes its own window instead of asking for the one the "
        "health block will compare it against"
    )
    assert api(B, "reference_rough_s")(60 * 50_000, FS) == B.REFERENCE_ROUGH_S
    assert B.reference_rough_s(4 * 50_000, FS) == 1.0
    assert B.reference_rough_s(0, FS) == 0.0


# ===========================================================================
# 5. THE PROPOSAL
# ===========================================================================
def test_the_gating_branch_is_declined_when_events_explain_the_score():
    """The gating statistic cannot tell a pore stepping between conductance levels
    from a pore blocked a quarter of the time by events long against its 20 ms block
    -- and the rule INVERTS on the answer, so the false positive does not degrade the
    proposal, it replaces it with one measured against the wrong quantity. The second
    measurement that separates them is the probe's own mask fraction, which was
    already being taken and read by nothing."""
    x, _, occ = renewal(30, dur=60.0)
    probe = P.probe_timescales(x, FS, polarity=-1)

    assert probe.gating_score > P.GATING_THRESHOLD, "the misfire itself must reproduce"
    assert probe.gating is True, "the raw statistic is unchanged and still says yes"
    assert probe.mask_fraction > B.MASK_FRACTION_CEILING
    assert api(probe, "gating_credible") is False

    proposal = P.propose_baseline(probe)
    assert proposal.config.rough_s >= 0.3, (
        f"the level branch was taken anyway: {proposal.config.describe()}"
    )
    assert proposal.param_source == "proposed"


def test_the_declined_branch_proposes_a_window_that_measures_the_depth():
    """The decision is worth taking only because the dwell branch is RIGHT here. On
    the reported case the level branch measures 44.5 pA of a 110 pA event; the dwell
    branch measures it to better than 1%."""
    x, spans, _ = renewal(30, dur=60.0)
    proposal = P.probe_and_propose(x, FS, polarity=-1)
    res = B.estimate_baseline_result(x, FS, config=proposal.config, polarity=-1)
    assert measured_depth(x, res.baseline, spans) == pytest.approx(DEPTH, rel=0.02)


def test_a_real_stepping_pore_still_takes_the_level_branch():
    """CONTROL, PASSES BOTH -- written with no new API on purpose, so the identical
    assertions run against the reverted package.

    A pore whose OPEN level steps every 35 ms is the case the gating branch exists
    for, and it must survive the new condition. It does because a stepping pore
    DISTRIBUTES its time over levels rather than sitting below one, so its one-sided
    mask stays well under `MASK_FRACTION_CEILING` -- the same shape as ferritin_data3,
    which masks 15.4% while scoring 13.8.
    """
    rng = np.random.default_rng(5)
    n = 1_200_000
    x = np.full(n, -400.0) + rng.normal(0.0, 14.0, n)
    k = int(0.035 * FS)
    for i, s in enumerate(range(0, n, k)):
        if i % 2:
            x[s:s + k] += 290.0
    probe = P.probe_timescales(x.astype(np.float32), FS, polarity=-1)
    assert probe.gating and probe.gating_score > P.GATING_THRESHOLD
    assert probe.mask_fraction < B.MASK_FRACTION_CEILING
    assert P.propose_baseline(probe).config.rough_s <= 0.1


def test_gating_stays_the_raw_statistic_and_the_judgement_is_separate():
    """``gating`` is DEFINED as ``gating_score > GATING_THRESHOLD``, it is displayed
    as that statistic, and a field that quietly answered a different question than
    its name would be the same defect one layer down. The judgement is its own
    property and its own provenance key."""
    x, _, _ = renewal(30, dur=60.0)
    probe = P.probe_timescales(x, FS, polarity=-1)
    assert probe.gating == (probe.gating_score > P.GATING_THRESHOLD)
    d = probe.to_dict()
    assert "probe_gating_credible" in d, (
        "the probe reports the raw flag but not whether it is believed"
    )
    assert d["probe_gating"] is True and d["probe_gating_credible"] is False
    assert json.loads(json.dumps(d)) == d


def test_the_reason_line_names_the_branch_it_declined():
    """A proposal a user cannot interrogate is an auto-apply with an extra click, and
    a reason that still read "Gating pore: the open-pore level itself steps" would be
    confidently wrong about the recording in front of him."""
    x, _, _ = renewal(30, dur=60.0)
    proposal = P.probe_and_propose(x, FS, polarity=-1)
    assert "LEVEL branch is declined" in proposal.reason
    assert "read as events" in proposal.reason
    assert "Gating pore" not in proposal.reason


def test_the_summary_row_does_not_read_yes_for_a_declined_branch():
    x, _, _ = renewal(30, dur=60.0)
    rows = dict(P.proposal_summary(P.probe_and_propose(x, FS, polarity=-1), FS))
    row = rows["Pore stepping (gating)"]
    assert not row.startswith("yes"), row
    assert "events, not levels" in row


# ===========================================================================
# 6. PLUMBING
# ===========================================================================
def test_the_reference_numbers_reach_provenance_and_stay_json_safe():
    """The GUI renders the health block from provenance, so a diagnostic that exists
    only as a Python attribute is a diagnostic the user never sees."""
    x, _, _ = renewal(30, dur=60.0)
    res = B.estimate_baseline_result(x, FS, config=cfg(0.03, 0.01), polarity=-1)
    prov = B.BaselineResult(baseline=res.baseline, config=res.config, fs=FS,
                            polarity=-1, health=res.health).provenance()
    block = prov["baseline_health"]
    assert {"mask_fraction_reference", "mask_reference_ratio",
            "mask_reference_rough_s"} <= set(block), sorted(block)
    assert json.loads(json.dumps(block)) == block
    assert block["mask_fraction_reference"] == pytest.approx(
        res.health.mask_fraction_reference)
    assert block["mask_reference_ratio"] > B.MASK_REFERENCE_MAX_RATIO
    assert block["mask_reference_rough_s"] == B.REFERENCE_ROUGH_S
    assert any(c["name"] == "mask_reference" and c["level"] == "warn"
               for c in block["checks"])
    assert block["ok"] is False


# ===========================================================================
# 7. THE BOUND -- on the owner's own recordings, what moves is disclosed at the
#    pin that moved (two BL-5 proposals, below) and everything else is pinned
# ===========================================================================
_KNOWN_GOOD = {
    "ferritin_data1": (B.BaselineConfig(method="robust_median", window_s=0.5, decim=50),
                       (1.0, 0.3)),
    # (2.0, 0.5), not the (1.0, 0.3) first pinned here: the old pin defended the
    # median-only ranking's blindness to one catastrophic state (BL-5).  data2 holds
    # a real 395 ms deep state at t = 99.303 s that the 1 s / 300 ms baseline follows
    # 78.2% of the way into while its MEDIAN intrusion over the other ~278 states
    # reads exactly 0.0 -- one ridden state cannot move a median over hundreds.
    # Under the worst-state criterion (`baseline_probe.WORST_STATE_INTRUSION_MAX`)
    # that rung now loses to 2 s / 500 ms, which follows no single state at all
    # (worst_fraction None) and certifies all-eight-green.
    "ferritin_data2": (B.BaselineConfig(method="masked_median", rough_s=0.5, base_s=0.1,
                                        decim=50, despike_size=3), (2.0, 0.5)),
    "ferritin_data3": (B.BaselineConfig(method="masked_median", rough_s=0.03,
                                        base_s=0.01, decim=10), (0.03, 0.01)),
}


_CACHE: dict[str, object] = {}


def _corpus_trace(tag):
    import pandas as pd

    from nanoscope.analysis import ml_preprocess as mp
    csv = CORPUS / f"{tag}.csv.gz"
    if not csv.exists():
        pytest.skip(f"corpus absent: {csv}")
    if tag not in _CACHE:
        raw = np.asarray(pd.read_csv(csv, usecols=["current_raw_pA"])["current_raw_pA"],
                         np.float32)
        _CACHE[tag] = mp.notch_sequential(raw, FS)
    return _CACHE[tag]


def _apo():
    if not APO.exists():
        pytest.skip(f"recording absent: {APO}")
    if "apo" not in _CACHE:
        from nanoscope.io.base import open_document
        doc = open_document(APO)
        _CACHE["apo"] = (np.asarray(doc.channels[2], np.float64),
                         float(doc.meta.sample_rate_hz))
    return _CACHE["apo"]


@pytest.mark.corpus
@pytest.mark.parametrize("tag", sorted(_KNOWN_GOOD))
def test_the_owners_corpus_proposes_and_certifies_exactly_what_it_did(tag):
    """CONTROL, PASSES BOTH -- and the bound the whole change is held to. Written
    with no new API on purpose, so the IDENTICAL assertions run against the reverted
    package and "nothing moved" is a comparison rather than a claim.

    ferritin_data3 is the ONE genuinely gating recording in evidence: it scores 13.8
    while masking 15.4%, so the new credibility condition must leave its level branch
    alone.

    **ferritin_data2's pin has since moved, so for that ONE tag this is no longer a
    control against every older package.** The worst-state ranking criterion (BL-5)
    changed its proposal from (1.0, 0.3) to (2.0, 0.5) on purpose -- see the comment
    on the pin itself for the measured ride -- and holding the old value here would
    have pinned the blindness the criterion removes. data1 and data3 still hold what
    they always did, and the health assertions run at the config the owner actually
    used, which has not moved for any tag.
    """
    config, want = _KNOWN_GOOD[tag]
    x = _corpus_trace(tag)
    proposal = P.probe_and_propose(np.asarray(x, np.float64), FS, polarity=-1)
    assert (proposal.config.rough_s, proposal.config.base_s) == pytest.approx(want)
    assert proposal.param_source == "proposed"

    h = B.estimate_baseline_result(x, FS, config=config, polarity=-1).health
    assert h.ok, h.warnings
    assert not h.warnings


@pytest.mark.corpus
def test_the_apo_recording_is_untouched_at_every_window_he_might_try():
    """CONTROL for the C-2 cross-check: his -600 mV apo file, same proposal, and no
    ``mask_reference`` objection at any of the three windows in evidence.

    **The blanket ``h.ok`` at all three windows was wrong and has been removed.** As
    first written this test asserted no warning at 0.03/0.01 as well. That window is
    a 10 ms tracking median on a pore whose states last 20 ms, and it is measurably
    catastrophic: the baseline sits 48.4% of the way inside the states, so every
    depth read from it is about half of the truth. Asserting a clean bill of health
    for it pinned the defect that
    ``tests/analysis/test_baseline_state_timescale.py`` exists to remove. The two
    windows that measure this recording correctly still certify silently, which is
    the non-regression this test is for.

    **The (1.0, 0.3) proposal pin was wrong too and now reads (2.0, 0.5).** As first
    written this test pinned the median-only ranking's offer, and that offer is the
    BL-5 defect on this very recording: at 1 s / 300 ms the tracked baseline rides
    55.6% of the way into a 473 ms blocked state at t = 38.3 s, so the 521 ms partial
    clog at 37.13 s stops being detectable at all and ~33 shallow events are invented
    on the collapsed local baseline. The worst-state criterion rejects that rung and
    the offer is now 2 s / 500 ms, which detects the clog as ONE long event; asserting
    (1.0, 0.3) here defended exactly the blindness
    ``tests/analysis/test_bl_worststate_ranking.py`` exists to remove. The health
    loop below is genuinely untouched: all three windows in evidence still certify
    the same way they always did.
    """
    x, fs = _apo()
    proposal = P.probe_and_propose(x, fs, polarity=-1)
    assert (proposal.config.rough_s, proposal.config.base_s) == (2.0, 0.5)
    assert proposal.probe.gating is False
    assert proposal.probe.gating_score < P.GATING_THRESHOLD

    for rough, base in ((1.0, 0.3), (2.0, 0.5), (0.5, 0.1)):
        h = B.estimate_baseline_result(x, fs, config=cfg(rough, base),
                                       polarity=-1).health
        assert h.ok and not h.warnings, (rough, h.warnings)

    h = B.estimate_baseline_result(x, fs, config=cfg(0.03, 0.01), polarity=-1).health
    assert h.check("mask_reference").level == "ok", (
        "the C-2 cross-check must still be silent here -- at 10% occupancy a narrow "
        "rough window has too little to hide in, and reading this window as healthy "
        "was never something mask_reference could have caught"
    )
    assert not h.ok, "a baseline half-way inside the events may not be certified"


@pytest.mark.corpus
def test_the_owners_recordings_sit_clear_of_the_new_threshold():
    """The margin, measured rather than assumed -- a bound that holds by 1% is not a
    bound. His two masked-median corpus files read 0.98 and 0.08, and the apo file
    0.99 / 1.14 / 1.30 across the three windows in evidence: 2.0x, 26x and 1.5x clear
    of `MASK_REFERENCE_MAX_RATIO` on the tightest of them.

    The apo file at 0.03/0.01 is the interesting one. It is a window 66x too narrow
    for that recording, and the ratio still reads 1.30 -- because at 9% occupancy a
    narrow window has too little to hide in. The failure needs the OCCUPANCY, which
    is exactly why raising `baseline_probe.GATING_THRESHOLD` would not have helped.
    """
    for tag, (config, _) in sorted(_KNOWN_GOOD.items()):
        if config.method != "masked_median":
            continue
        h = B.estimate_baseline_result(_corpus_trace(tag), FS, config=config,
                                       polarity=-1).health
        _, ratio, rough = reference_of(h)
        assert ratio < B.MASK_REFERENCE_MAX_RATIO, (tag, ratio)
        assert rough == B.REFERENCE_ROUGH_S

    x, fs = _apo()
    for rough, base in ((2.0, 0.5), (0.5, 0.1), (0.03, 0.01)):
        h = B.estimate_baseline_result(x, fs, config=cfg(rough, base),
                                       polarity=-1).health
        _, ratio, _ = reference_of(h)
        assert ratio < B.MASK_REFERENCE_MAX_RATIO, (rough, ratio)
        assert h.check("mask_reference").level == "ok"
