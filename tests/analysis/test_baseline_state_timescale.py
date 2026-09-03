"""The tracking window against the EVENT timescale -- the seventh and eighth checks.

THE DEFECT
==========
On the owner's -700 mV apoferritin recording the app's own probe measured a longest
state of 31 ms and proposed ``base_s`` = 0.03 s. A baseline tracker whose window is
the size of the states it must ignore rides up into them: the tracked baseline sat
**-51.7 pA against a locally reconstructed open pore of -48.2 pA** (7.2%), the median
depth read 97.00 pA instead of 100.48, and the median dI/I0 read 1.900 instead of
2.103 -- which is the whole of the "anomaly" that made -700 mV the odd file out in a
three-point voltage series (1.90 against 2.36 and 2.57).

**All six health checks passed, in green, on that configuration** -- on the screen
captioned "These six checks were run on the windows PROPOSED above... This is the
evidence for the Apply click." They test the window against the RECORDING length
(6049x, 1815x -- both vast), the mask fraction against two constants, the mask
against a wide-window mask, the rough pass's sigma, and overshoot. Every one of them
is about ``rough_s`` -- which excursions become events -- or about the recording's
length. Nothing looked at ``base_s`` against the events, so 30 ms against a 21 ms
state certified exactly as cleanly as 500 ms against it.

WHAT IS NOW THERE
=================
Two checks, and they are deliberately two:

``base_vs_state``
    ``base_s`` over `baseline.state_dwell_s` -- the median state measured over the
    REFERENCE window, the same wide window `reference_mask_fraction` uses. The
    actionable one: it says how far to widen. `BASE_STATE_RATIO_MIN` = 3.0.
``baseline_intrusion``
    `baseline.tracked_intrusion` -- for every state, how far the finished baseline
    sits inside it against the open pore taken from the flanks on either side. The
    direct one: it is the measurement that found the defect, and it is very nearly
    the fraction by which the depths read shallow. `BASELINE_INTRUSION_MAX` = 0.03.

Each catches configurations the other does not, both measured and both pinned below
(:func:`test_neither_check_subsumes_the_other`).

THE LOAD-BEARING TEST IS NOT "DOES THE FLAG FIRE"
=================================================
It is :func:`test_health_ok_never_certifies_a_depth_the_baseline_ate` -- *does*
``health.ok`` *ever read True on a configuration whose per-event depth is more than a
few percent off* -- swept over the dwell/window ratio and the duty cycle rather than
asserted at the one reported point, and :func:`test_the_same_property_on_his_own_
recordings` which asks it again of the three real files. That framing is what the C-2
fix used; asking only "does it fire on -700 mV" is what let this class survive.

WHAT MUST NOT MOVE
==================
A check that cries wolf on correct settings is one the owner learns to click past,
which returns us here. :func:`test_the_probed_windows_and_the_corpus_are_untouched`
and :func:`test_the_margins_are_measured_not_assumed` hold the whole change to his
own data: -500 mV and -600 mV at their probed windows, all three ferritin corpus
files, and -700 mV at every window that measures it correctly, all silent and all
with the same verdict they had before.

TESTS THAT PASS EITHER WAY, AND SAY SO
======================================
Against a programmatic revert of the fix, **17 of the 24 tests here fail and 7 pass**.
The 7 are labelled CONTROL in their own docstrings and are written with NO new API on
purpose, so the identical assertions run against both packages: the five corpus/apo
bound tests (what must not move), :func:`test_the_original_six_checks_could_not_see_it`
(why this needed a check of its own at all) and
:func:`test_the_floor_case_is_still_left_to_the_floor_check` (the non-regression the
new checks could most plausibly have broken). A test that passes both ways pins
nothing, and these seven say which they are.

The revert is ``scratchpad/FIXTWO/revert.py`` -- 19 hunks, each asserting it fired
exactly once -- with ``guard.py``, which prints where ``nanoscope`` resolved from and
refuses to run unless it is the reverted copy and the reverted surface is really
gone.
"""

from __future__ import annotations

import ast
import json
import pathlib
import time

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P

FS = 50_000.0
I0 = -34.8
DEPTH = 110.0
NOISE = 4.84

CORPUS = pathlib.Path("/Users/abrahamcampos/Desktop/Research-Data/FerritinModeler/CSVData")
APO = {
    -500: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0008_process data_-500 mV.abf"),
    -600: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0006_Process_-600 mV.abf"),
    -700: pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0005_Process_-700 mV.abf"),
}
#: What `baseline_probe` proposes for each, and what the report drove the app with.
#: -600 is (2.0, 0.5) again since the BL batch: the RP ladder briefly offered
#: (1.0, 0.3) on a median tie, and that window rides into the recording's 521 ms
#: clog (worst state 0.556 -- `baseline_probe.WORST_STATE_INTRUSION_MAX`; see
#: ``tests/analysis/test_bl_worststate_ranking.py``). The -34.790 pA the docstrings
#: in this file quote for -600 was always the 2.0 / 0.5 measurement.
PROBED = {-500: (1.0, 0.3), -600: (2.0, 0.5), -700: (0.1, 0.03)}


# ===========================================================================
# fixtures
# ===========================================================================
def telegraph(dwell_s, duty, *, dur=15.0, seed=7, depth=DEPTH, i0=I0,
              noise=NOISE, fs=FS):
    """``(trace, true spans, occupancy)`` -- a two-level pore at a chosen duty.

    Exponential gaps and gamma(4) dwells: the owner's own recordings are a narrow
    dwell distribution (CV 0.18) rather than an exponential one, and a periodic
    fixture would make the block medians alternate deterministically and test the
    fixture. Overlapping arrivals merge, which is what a pore does.
    """
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
    spans = [(s, e) for s, e in spans if e > s]
    return x.astype(np.float32), spans, sum(e - s for s, e in spans) / n


def measured_depth(x, baseline, spans):
    """The number the user reads: median over the true events of ``baseline - x``."""
    d = np.asarray(baseline, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    return float(np.median([np.median(d[s:e]) for s, e in spans]))


def cfg(rough, base=None):
    return B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base or rough / 4.0,
                                      decim=None)


def fired(h):
    return [c.name for c in h.checks if not c.ok]


def api(obj, name):
    """``getattr``, as an ASSERTION that the surface exists at all.

    A package without the fix must fail these tests on the claim being made, not on
    an ``AttributeError`` that reads like a broken test.
    """
    assert hasattr(obj, name), (
        f"{getattr(obj, '__name__', type(obj).__name__)} has no {name!r}: the "
        "tracking window is not measured against the event timescale anywhere, "
        "which is the defect"
    )
    return getattr(obj, name)


_CACHE: dict = {}


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
        _CACHE[tag] = mp.notch_sequential(raw, FS)
    return _CACHE[tag]


#: The three corpus configurations that measured the persisted training data.
_KNOWN_GOOD = {
    "ferritin_data1": B.BaselineConfig(method="robust_median", window_s=0.5, decim=50),
    "ferritin_data2": B.BaselineConfig(method="masked_median", rough_s=0.5, base_s=0.1,
                                       decim=50, despike_size=3),
    "ferritin_data3": B.BaselineConfig(method="masked_median", rough_s=0.03,
                                       base_s=0.01, decim=10),
}


# ===========================================================================
# 1. THE PROPERTY -- the one that was missing
# ===========================================================================
#: (mean dwell ms, duty). Nine pores spanning a 12x range of state length and a 5x
#: range of occupancy -- the plane the failure lives in, not one reproduction of it.
_PORES = [(d, q) for d in (5, 20, 60) for q in (0.05, 0.12, 0.25)]
#: ``base_s``, with ``rough_s`` at the matched `ROUGH_BASE_RATIO`, spanning 160x.
_WINDOWS = (0.005, 0.02, 0.05, 0.2, 0.8)


def test_health_ok_never_certifies_a_depth_the_baseline_ate():
    """**The load-bearing test.** `BaselineHealth.ok` is a certificate, and one that
    can be issued for a depth the baseline has eaten certifies nothing.

    Swept over 9 (dwell, duty) pores x 5 tracking windows. BEFORE these two checks
    existed, 8 of those 45 estimates read ``ok`` with per-event depths between 0.4%
    and 97.6% of the truth -- every check passing, no warning anywhere. AFTER, the
    worst depth surviving ``ok`` is 0.998 of the truth.

    The tolerance is 0.98, deliberately looser than the 0.998 measured: this test
    exists to catch a certificate issued for a depth the baseline has taken a real
    bite out of, and pinning it at the measured value would turn a 1% shift in the
    fixture's RNG into a failure that says nothing. What must never move is the
    direction -- see the second assertion, which is the one that fails without the
    checks.
    """
    certified, worst = [], (1.0, None)
    for dwell_ms, duty in _PORES:
        x, spans, occ = telegraph(dwell_ms * 1e-3, duty)
        assert 0.03 < occ < 0.35, f"fixture drifted: dwell {dwell_ms} duty {duty} -> {occ}"
        for base in _WINDOWS:
            res = B.estimate_baseline_result(
                x, FS, config=cfg(base * B.ROUGH_BASE_RATIO, base), polarity=-1)
            ratio = measured_depth(x, res.baseline, spans) / DEPTH
            if res.health.ok:
                certified.append((dwell_ms, duty, base, round(ratio, 4)))
                if ratio < worst[0]:
                    worst = (ratio, (dwell_ms, duty, base))

    bad = [c for c in certified if c[3] < 0.98]
    assert not bad, (
        "health.ok is True on estimates whose per-event depth is wrong by more than "
        f"2%: {bad}. Every one of these has a tracking window comparable to (or "
        "shorter than) the states it is supposed to ignore, so the baseline sits "
        "inside them -- and every check that existed before compares rough_s, the "
        "mask, or the recording length against a constant."
    )
    assert len(certified) >= 12, f"the sweep certified almost nothing: {certified}"
    assert worst[0] >= 0.98, worst


@pytest.mark.corpus
def test_the_same_property_on_his_own_recordings():
    """The property again, on the three real recordings rather than a generator, with
    ``rough_s`` pinned at 2.0 s so ONLY the tracking window moves -- the mask, the
    spans and the reference depth are then identical in every row and the depth
    difference cannot be anything but ``base_s``.

    Truth is each recording's own depth at ``base_s`` = 0.5 s, where all three read
    an intrusion of 0.000. Measured worst certified: 0.980 (-500 mV at 0.1 s).
    """
    for mv in (-500, -600, -700):
        x, fs = _apo(mv)
        ref = api(B, "reference_states")(x, fs, config=cfg(2.0, 0.5))
        assert ref is not None
        mask, _ = ref
        s, e = api(B, "state_runs")(mask, fs, dilate_ms=1.0)
        assert s.size > 100, (mv, s.size)

        def depth_at(base):
            res = B.estimate_baseline_result(x, fs, config=cfg(2.0, base), polarity=-1)
            d = float(np.median([np.median(res.baseline[a:c] - x[a:c])
                                 for a, c in zip(s, e)]))
            return d, res.health

        truth, _ = depth_at(0.5)
        for base in (0.03, 0.05, 0.07, 0.1, 0.3):
            d, h = depth_at(base)
            if h.ok:
                assert d / truth >= 0.97, (
                    f"{mv} mV certified base_s={base} s, whose median depth is "
                    f"{d:.2f} pA against {truth:.2f} pA at 0.5 s -- "
                    f"{1 - d / truth:.1%} eaten by the baseline"
                )


# ===========================================================================
# 2. THE REPORTED CASE, END TO END
# ===========================================================================
@pytest.mark.corpus
def test_the_seven_hundred_millivolt_proposal_is_no_longer_certified():
    """The measurement in report SUMMARY.md 4.1, reproduced through the app's own
    entry point: 0.1/0.03 on this file refuses to certify.

    Before, it was green on all six checks. The numbers here are the ones the report
    quotes, re-measured: a 21.7 ms median state against a 30 ms window (1.4x), and a
    baseline 4.0 pA -- 4.0% of the excursion -- inside the states.

    THE WINDOW IS NOW DRIVEN DIRECTLY, AND THE PROPOSER IS ASSERTED NOT TO OFFER IT.
    This test was written when ``probe_and_propose`` still offered 0.1/0.03 here, so
    it took the condemned window from the proposal. The proposer has since been taught
    the bound its own ``base_vs_state`` check enforces (a floor at
    ``BASE_STATE_RATIO_MIN x`` the median state dwell) and now offers 0.5/0.1, which
    certifies -- the repair the alarm below was standing in for. The subject of this
    test is the CONDEMNED window and what the checks say about it, so it names that
    window itself, exactly as its sibling ``test_the_original_six_checks_could_not_see_it``
    two tests below already did. The deleted line is replaced by its mirror image
    rather than by nothing: the proposal must NOT be the condemned pair, which fails
    if the floor is reverted. ``tests/analysis/test_fup_proposer_floor.py`` pins what
    the proposal now IS.
    """
    x, fs = _apo(-700)
    proposal = P.probe_and_propose(x, fs, polarity=-1)
    assert (proposal.config.rough_s,
            proposal.config.base_s) != pytest.approx(PROBED[-700]), (
        "the proposer offered the condemned window again -- the floor that stops it "
        f"has gone: {proposal.config.describe()}"
    )
    assert B.estimate_baseline_result(
        x, fs, config=proposal.config, polarity=-1).health.ok, (
        "what the proposer offers on this file must itself certify: "
        f"{proposal.config.describe()}")

    h = B.estimate_baseline_result(x, fs, config=cfg(*PROBED[-700]), polarity=-1).health
    assert not h.ok, "a window 1.4x the state dwell may not be certified healthy"
    assert set(fired(h)) == {"base_vs_state", "baseline_intrusion"}

    dwell = api(h, "state_dwell_s")
    assert dwell == pytest.approx(0.0217, abs=0.003), dwell
    assert h.base_state_ratio == pytest.approx(1.38, abs=0.1)
    assert h.base_state_ratio < B.BASE_STATE_RATIO_MIN
    assert h.intrusion_fraction == pytest.approx(0.040, abs=0.008)
    assert h.intrusion_fraction > B.BASELINE_INTRUSION_MAX
    assert h.intrusion_pA == pytest.approx(3.9, abs=1.0), (
        "the report reconstructed -48.218 pA against a stamped -51.661; this is the "
        "same 3.4-4.2 pA, measured per state"
    )
    assert h.gating_states is False, "this pore does not step; it is blocked"


@pytest.mark.corpus
def test_the_original_six_checks_could_not_see_it():
    """CONTROL, PASSES BOTH -- WHY this needed checks of its own, written with no new
    API so the identical assertions run against the reverted package.

    Every check that existed before reads ``ok`` on the -700 mV proposal. Three of
    them are not close: the recording is 6049 tracking windows long, the mask
    fraction is mid-band, and the wide-window mask agrees with the configured one to
    0.1%. The published QC scalars are all healthy too -- which is the point, because
    two thirds of the samples genuinely are open pore, so no metric of the BASELINE
    can see a bias that only shows up where the events are.
    """
    x, fs = _apo(-700)
    h = B.estimate_baseline_result(x, fs, config=cfg(*PROBED[-700]), polarity=-1).health

    for name in ("length_tracking", "length_rough", "mask_fraction",
                 "mask_reference", "rough_sigma_ratio", "overshoot"):
        c = h.check(name)
        assert c is not None and c.level == "ok", (name, c)
    assert h.check("length_tracking").value > 6000
    assert B.MASK_FRACTION_FLOOR < h.mask_fraction < B.MASK_FRACTION_CEILING
    assert h.mask_reference_ratio == pytest.approx(1.0, abs=0.2)
    assert h.overshoot < B.OVERSHOOT_FLAG
    assert abs(h.open_pore_drift_sigma) < 1.0


@pytest.mark.corpus
def test_the_correct_window_for_that_file_certifies_silently():
    """The other half of the reported case: the same recording at the windows that
    measure it. 1.0/0.3 and 2.0/0.5 read an intrusion of 0.000 and warn about
    nothing, so the check is telling him something specific about ONE window rather
    than about the file."""
    x, fs = _apo(-700)
    for rough, base in ((0.5, 0.1), (1.0, 0.3), (2.0, 0.5)):
        h = B.estimate_baseline_result(x, fs, config=cfg(rough, base), polarity=-1).health
        assert h.ok and not h.warnings, (rough, base, h.warnings)
        assert h.base_state_ratio > B.BASE_STATE_RATIO_MIN
        assert h.intrusion_fraction < B.BASELINE_INTRUSION_MAX


# ===========================================================================
# 3. IT MUST NOT CRY WOLF
# ===========================================================================
@pytest.mark.corpus
@pytest.mark.parametrize("tag", sorted(_KNOWN_GOOD))
def test_the_corpus_configurations_are_untouched(tag):
    """CONTROL, PASSES BOTH, no new API. The three configurations that measured the
    persisted training corpus still certify with no warning at all.

    ferritin_data3 is the one that matters: it is the genuinely gating pore, its
    ``base_s`` is deliberately SHORTER than its level dwells, and judged as an event
    pore it would read as 94% intruded. See
    :func:`test_a_stepping_pore_is_not_judged_as_an_event_pore`.
    """
    h = B.estimate_baseline_result(_corpus(tag), FS, config=_KNOWN_GOOD[tag],
                                   polarity=-1).health
    assert h.ok, h.warnings
    assert not h.warnings


@pytest.mark.corpus
@pytest.mark.parametrize("mv", (-500, -600))
def test_the_probed_windows_and_the_corpus_are_untouched(mv):
    """CONTROL, PASSES BOTH, no new API. His -500 mV and -600 mV recordings at the
    windows his own probe proposes -- the two of the three the report re-measured and
    found correct to the last digit (-27.161 and -34.790 pA against an independent
    reconstruction, difference 0.000).

    -700 mV is excluded on purpose and it is the whole finding: its probed window is
    the one that is wrong.
    """
    x, fs = _apo(mv)
    proposal = P.probe_and_propose(x, fs, polarity=-1)
    assert (proposal.config.rough_s, proposal.config.base_s) == pytest.approx(PROBED[mv])
    h = B.estimate_baseline_result(x, fs, config=proposal.config, polarity=-1).health
    assert h.ok, h.warnings
    assert not h.warnings


@pytest.mark.corpus
def test_the_margins_are_measured_not_assumed():
    """A bound that holds by 1% is not a bound. The clearance on every configuration
    that must stay silent, measured rather than asserted:

    ==========================  =====  ========  ==============================
    configuration               ratio  intruded  clearance
    ==========================  =====  ========  ==============================
    -500 mV probed (1.0/0.3)    12.4   -0.000    4.1x / >7x
    -600 mV probed (2.0/0.5)    24.1    0.000    8.0x / >7x
    -700 mV at 0.5/0.1           4.6    0.004    1.5x / 7.0x
    -600 mV at 0.5/0.1           4.8    0.003    1.6x / 10x
    ferritin_data2 (0.5/0.1)    36.2    0.004    12x  / 7.5x
    ==========================  =====  ========  ==============================

    The tightest is the RATIO at 1.5x, on a window nobody proposes for that file --
    and the two checks are independent there, because its intrusion is 7x clear.
    """
    rows = [("apo -500 probed", *_apo(-500), cfg(*PROBED[-500])),
            ("apo -600 probed", *_apo(-600), cfg(*PROBED[-600])),
            ("apo -700 @0.5/0.1", *_apo(-700), cfg(0.5, 0.1)),
            ("apo -600 @0.5/0.1", *_apo(-600), cfg(0.5, 0.1)),
            ("ferritin_data2", _corpus("ferritin_data2"), FS,
             _KNOWN_GOOD["ferritin_data2"])]
    for name, x, fs, config in rows:
        h = B.estimate_baseline_result(x, fs, config=config, polarity=-1).health
        assert h.base_state_ratio > 1.4 * B.BASE_STATE_RATIO_MIN, (name, h.base_state_ratio)
        assert h.intrusion_fraction < B.BASELINE_INTRUSION_MAX / 5.0, (
            name, h.intrusion_fraction)
        assert h.check("base_vs_state").level == "ok", name
        assert h.check("baseline_intrusion").level == "ok", name


# ===========================================================================
# 4. A STEPPING PORE IS A DIFFERENT RECORDING
# ===========================================================================
@pytest.mark.corpus
def test_a_stepping_pore_is_not_judged_as_an_event_pore():
    """On a pore whose OPEN LEVEL steps, a tracking window shorter than the long
    states is the point rather than the defect, and both checks decline to judge.

    This is not a soft spot to be tightened later. ferritin_data3 -- the one
    genuinely gating recording in evidence -- reads a ratio of **2.99** against a
    threshold of 3.0 and an intrusion of **0.94** at the configuration that produced
    the persisted training data. Without the gating judgement the corpus contract
    breaks on the loudest warning this module can emit. `GATING_THRESHOLD` is
    therefore load-bearing here and its margin is the thing to check: data3 scores
    13.8 against 8.0, and the two apo files score 0.5 and 1.7.
    """
    x = _corpus("ferritin_data3")
    h = B.estimate_baseline_result(x, FS, config=_KNOWN_GOOD["ferritin_data3"],
                                   polarity=-1).health
    assert api(h, "gating_states") is True
    assert h.gating_score > B.GATING_THRESHOLD
    assert h.base_state_ratio < B.BASE_STATE_RATIO_MIN, (
        "if data3 ever stops being the tight case, re-derive the constant rather "
        f"than relaxing this: {h.base_state_ratio}"
    )
    assert h.intrusion_fraction > 0.5
    assert h.check("base_vs_state").level == "ok"
    assert h.check("baseline_intrusion").level == "ok"
    assert "NOT APPLICABLE" in h.check("base_vs_state").message
    assert h.ok and not h.warnings

    for mv in (-600, -700):
        xa, fs = _apo(mv)
        ha = B.estimate_baseline_result(xa, fs, config=cfg(*PROBED[mv]),
                                        polarity=-1).health
        assert ha.gating_states is False
        assert ha.gating_score < B.GATING_THRESHOLD / 4.0, (mv, ha.gating_score)


def test_a_high_score_is_not_believed_when_events_explain_it():
    """The gating exemption inherits `BaselineProbe.gating_credible`'s second
    condition, and it has to: a heavily blocked pore scores high on a statistic that
    cannot tell "the level moved" from "the level was covered". At 35% occupancy the
    C-2 fixture scores 17 -- past the threshold -- and the exemption must NOT apply,
    or the one configuration that fix exists to catch would be waved through here.
    """
    x, spans, occ = telegraph(0.018, 0.35, dur=20.0)
    assert occ > B.MASK_FRACTION_CEILING
    h = B.estimate_baseline_result(x, FS, config=cfg(0.03, 0.01), polarity=-1).health
    assert h.gating_score > B.GATING_THRESHOLD, h.gating_score
    assert api(h, "gating_states") is False, (
        "a pore that is simply blocked a third of the time is not a stepping pore"
    )
    assert not h.ok
    assert "base_vs_state" in fired(h) and "baseline_intrusion" in fired(h)


def test_the_gating_question_is_asked_once():
    """The proposal INVERTS on this judgement and so does the verdict. Two spellings
    of one statistic is how the app proposes a window on one reading of a recording
    and certifies it on another."""
    assert P.GATING_THRESHOLD is api(B, "GATING_THRESHOLD")
    assert P.GATING_BLOCK_S is api(B, "GATING_BLOCK_S")

    tree = ast.parse(pathlib.Path(P.__file__).read_text())
    assigned = {t.id for node in tree.body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    assert not ({"GATING_THRESHOLD", "GATING_BLOCK_S"} & assigned), (
        "baseline_probe re-spells a gating constant instead of importing baseline's"
    )
    fns = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "probe_timescales"]
    assert len(fns) == 1
    called = {n.func.id for n in ast.walk(fns[0])
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "gating_spread" in called, (
        "probe_timescales computes the gating statistic itself instead of asking for "
        "the one the health block will judge with"
    )


# ===========================================================================
# 5. THE TWO CHECKS ARE TWO
# ===========================================================================
@pytest.mark.corpus
def test_neither_check_subsumes_the_other():
    """Both directions, on real recordings.

    ferritin_data2 at ``base_s`` 0.02 s scores a comfortable 7.7x on the ratio -- its
    median state is 2.6 ms -- and is intruded 0.064, because its state distribution
    has a long tail the median cannot see. The -700 mV file at 0.05 s scores 2.3x and
    is intruded only 0.028. A single check would have missed one of them, and the
    first is the shape the -700 mV failure would have taken on a burstier pore.
    """
    d2 = B.estimate_baseline_result(
        _corpus("ferritin_data2"), FS,
        config=_KNOWN_GOOD["ferritin_data2"].replace(base_s=0.02, decim=None),
        polarity=-1).health
    assert d2.base_state_ratio > 2.0 * B.BASE_STATE_RATIO_MIN, d2.base_state_ratio
    assert d2.intrusion_fraction > B.BASELINE_INTRUSION_MAX
    assert fired(d2) == ["baseline_intrusion"]

    x, fs = _apo(-700)
    a7 = B.estimate_baseline_result(x, fs, config=cfg(2.0, 0.05), polarity=-1).health
    assert a7.base_state_ratio < B.BASE_STATE_RATIO_MIN
    assert a7.intrusion_fraction < B.BASELINE_INTRUSION_MAX
    assert fired(a7) == ["base_vs_state"]


# ===========================================================================
# 6. THE MEASUREMENTS ARE THE MEASUREMENTS THEY CLAIM TO BE
# ===========================================================================
def test_the_state_dwell_ignores_the_dilation_kernel_runs():
    """A one-sided threshold GROWN by ``+-dilate_ms`` turns one noise sample into a
    run of exactly ``2k+1``. Below the event rate at which those stop being a
    minority of the RUNS, a median run length is a reading of ``mask_sigma``, not of
    the pore: on this fixture 34 states share the mask with ~60 noise runs and the
    unfiltered median reads 2.02 ms for a 20 ms state -- a 10x error, in the
    direction that certifies a window ten times too narrow.
    """
    x, spans, _ = telegraph(0.020, 0.05)
    got = api(B, "reference_states")(x, FS, config=cfg(0.08, 0.02))
    assert got is not None
    mask, _ = got
    raw = api(B, "state_dwell_s")(mask, FS)
    kept = B.state_dwell_s(mask, FS, dilate_ms=B.DEFAULT_BASELINE.dilate_ms)
    dilation_s = (2 * int(B.DEFAULT_BASELINE.dilate_ms * 1e-3 * FS) + 1) / FS

    assert raw == pytest.approx(dilation_s, rel=0.1), (
        "the fixture no longer has enough noise runs to make the point; raise the "
        f"trace length or lower the duty: {raw}"
    )
    assert kept == pytest.approx(0.020, rel=0.35), kept
    s, e = B.state_runs(mask, FS, dilate_ms=B.DEFAULT_BASELINE.dilate_ms)
    assert np.all((e - s) > 2 * int(B.DEFAULT_BASELINE.dilate_ms * 1e-3 * FS) + 1)
    assert s.size >= 0.6 * len(spans), (s.size, len(spans))


@pytest.mark.corpus
def test_the_intrusion_reference_is_the_baseline_not_the_surviving_samples():
    """The obvious spelling of "the open pore beside this state" is the median of the
    UNMASKED samples in the flanks, and it is biased: the mask is one-sided, so the
    survivors' median sits ABOVE the level. Measured on ferritin_data2 at its
    known-good configuration, that spelling reads 4.3 pA of intrusion where the
    baseline-referenced one reads 0.9 -- it would have warned on a corpus file, on an
    artefact of its own reference.
    """
    x = np.asarray(_corpus("ferritin_data2"), np.float64)
    config = _KNOWN_GOOD["ferritin_data2"]
    res = B.estimate_baseline_result(x, FS, config=config, polarity=-1)
    mask, _ = api(B, "reference_states")(x, FS, config=config)
    honest = api(B, "tracked_intrusion")(x, res.baseline, mask, FS,
                                         dilate_ms=config.dilate_ms)

    # the biased spelling, computed here so the claim is a measurement
    s, e = B.state_runs(mask, FS, dilate_ms=config.dilate_ms)
    pad = int(B.INTRUSION_PAD_S * FS)
    fl = int(B.INTRUSION_FLANK_S * FS)
    biased = []
    for a, c in zip(s[:400], e[:400]):
        parts = []
        for q0, q1 in ((max(0, a - pad - fl), max(0, a - pad)),
                       (min(x.size, c + pad), min(x.size, c + pad + fl))):
            free = ~mask[q0:q1]
            if free.sum() > 20:
                parts.append(float(np.median(x[q0:q1][free])))
        if parts:
            biased.append(float(np.mean(parts)) - float(np.median(res.baseline[a:c])))
    assert np.median(biased) > 3.0 * honest["intrusion_pA"], (
        np.median(biased), honest["intrusion_pA"])
    assert honest["intrusion_pA"] < 2.0
    assert res.health.check("baseline_intrusion").level == "ok"


def test_the_dwell_is_measured_over_the_reference_window():
    """A rough window shorter than the states TRACKS them instead of exposing them,
    so the configured mask's runs are a reading of ``rough_s``. Sizing ``base_s``
    against them would compare the window with a number that shrinks as the window
    does -- the check would then pass by making the estimate worse."""
    x, _, _ = telegraph(0.040, 0.12)
    tight = cfg(0.02, 0.005)
    wide = cfg(2.0, 0.5)
    _, tight_mask, _, _ = B._rough_pass(x, FS, config=tight)
    ref_mask, w = api(B, "reference_states")(x, FS, config=tight)

    assert w == B.reference_rough_s(x.size, FS)
    d_tight = B.state_dwell_s(tight_mask, FS, dilate_ms=1.0)
    d_ref = B.state_dwell_s(ref_mask, FS, dilate_ms=1.0)
    assert d_ref > 3.0 * d_tight, (d_ref, d_tight)
    assert d_ref == pytest.approx(0.040, rel=0.4), d_ref

    # and the health block used the reference one
    h = B.estimate_baseline_result(x, FS, config=tight, polarity=-1).health
    assert h.state_dwell_s == pytest.approx(d_ref, rel=1e-9)
    # the same recording measured at a window that works, for contrast
    assert B.estimate_baseline_result(x, FS, config=wide, polarity=-1).health.ok


# ===========================================================================
# 7. NOT MEASURED IS NOT THE SAME AS OK
# ===========================================================================
def test_a_recording_too_short_for_a_reference_reports_nothing():
    """Below ``REFERENCE_WINDOWS_MIN * REFERENCE_MIN_ROUGH_S`` = 2 s there is no
    window wide enough to see over a state worth sizing anything against. The checks
    must be ABSENT, never reported as agreement they did not measure."""
    x = (I0 + np.random.default_rng(5).normal(0, NOISE, 50_000)).astype(np.float32)
    h = B.estimate_baseline_result(x, FS, config=B.DEFAULT_BASELINE, polarity=-1).health
    assert h.check("base_vs_state") is None
    assert h.check("baseline_intrusion") is None
    assert api(h, "state_dwell_s") is None and h.intrusion_fraction is None


def test_the_floor_case_is_still_left_to_the_floor_check():
    """CONTROL, PASSES BOTH, no new API. An event-free trace and a 60%-duty inversion
    both mask ~nothing, and a state timescale measured on ~no states is arithmetic
    rather than evidence. The floor check's message already says the two opposite
    things it can mean, and the two traces must still fire exactly the same pair."""
    rng = np.random.default_rng(3)
    empty = (-100.0 + rng.normal(0, 5.0, 1_000_000)).astype(np.float32)
    busy = (-100.0 + rng.normal(0, 5.0, 1_000_000)).astype(np.float32)
    for s in range(0, busy.size - 6_000, 10_000):
        busy[s:s + 6_000] -= 200.0

    for x in (empty, busy):
        h = B.estimate_baseline_result(x, FS, polarity=-1).health
        assert sorted(fired(h)) == ["mask_fraction_floor", "overshoot"]


def test_the_non_masked_methods_get_no_state_checks():
    """``robust_median`` and the block methods have no pass 1, so they have no mask
    to measure a state timescale from -- the same reason their mask diagnostics are
    ``None``. ferritin_data1's known-good configuration is one of them."""
    x, _, _ = telegraph(0.020, 0.12)
    h = B.estimate_baseline_result(
        x, FS, config=B.BaselineConfig(method="robust_median", window_s=0.5,
                                       decim=50), polarity=-1).health
    assert h.check("base_vs_state") is None
    assert h.check("baseline_intrusion") is None
    assert api(h, "state_dwell_s") is None


# ===========================================================================
# 8. PLUMBING AND COST
# ===========================================================================
def test_the_numbers_reach_provenance_and_stay_json_safe():
    """The GUI renders the health block from provenance, so a diagnostic that exists
    only as a Python attribute is one the owner never sees. Additive keys only --
    the 20-column EventSet schema is frozen and this is where new information goes."""
    x, _, _ = telegraph(0.020, 0.12)
    res = B.estimate_baseline_result(x, FS, config=cfg(0.04, 0.01), polarity=-1)
    prov = B.BaselineResult(baseline=res.baseline, config=res.config, fs=FS,
                            polarity=-1, health=res.health).provenance()
    block = prov["baseline_health"]
    assert {"state_dwell_s", "base_state_ratio", "intrusion_fraction",
            "intrusion_pA", "intrusion_depth_pA", "intrusion_n_states",
            "gating_score", "gating_states"} <= set(block), sorted(block)
    assert json.loads(json.dumps(block)) == block
    assert block["base_state_ratio"] == pytest.approx(res.health.base_state_ratio)
    assert block["ok"] is False
    names = [c["name"] for c in block["checks"]]
    assert names.index("base_vs_state") > names.index("mask_reference")
    assert any(c["name"] == "base_vs_state" and c["level"] == "warn"
               for c in block["checks"])


def test_the_messages_say_what_is_wrong_and_what_to_do():
    """The owner reads these to decide whether to trust a depth. A check that only
    names a number is a check he has to go and re-derive."""
    x, _, _ = telegraph(0.020, 0.12)
    h = B.estimate_baseline_result(x, FS, config=cfg(0.04, 0.01), polarity=-1).health
    m = h.check("base_vs_state").message
    assert "base_s" in m and "WIDEN base_s to at least" in m
    assert "median state this pore spends below its open pore" in m
    assert "depths read shallow" in m
    assert "RIDES INTO them" in m
    assert h.check("base_vs_state").threshold.startswith(">= 3")

    m2 = h.check("baseline_intrusion").message
    assert "INSIDE the states" in m2 and "pA" in m2
    assert "depth_pA and frac_depth" in m2
    assert h.check("baseline_intrusion").value == pytest.approx(h.intrusion_fraction)


@pytest.mark.corpus
def test_the_two_measurements_are_cheap_on_twelve_million_samples():
    """The health block is opt-in because it already costs a second pass-1 median.
    These two must not be what makes it unusable: the state timescale is free (it
    reads the mask `reference_mask_fraction` already built), and the intrusion is a
    Python loop over STATES, not samples. Budget: under 15% of the block, and under
    1 s on the 12.5 M-sample -500 mV recording."""
    x, fs = _apo(-500)
    config = cfg(*PROBED[-500])
    B.estimate_baseline_result(x, fs, config=config, polarity=-1, health=False)

    mask, _ = api(B, "reference_states")(x, fs, config=config)
    base = B.estimate_baseline(x, fs, method="masked_median", polarity=-1,
                               rough_s=config.rough_s, base_s=config.base_s)
    t = time.perf_counter()
    got = api(B, "tracked_intrusion")(x, base, mask, fs, dilate_ms=config.dilate_ms)
    t_intr = time.perf_counter() - t
    t = time.perf_counter()
    B.gating_spread(x, fs)
    t_gate = time.perf_counter() - t

    assert got is not None and got["n_states"] > 100
    assert t_intr < 1.0, f"intrusion took {t_intr:.2f} s on {x.size} samples"
    assert t_gate < 3.0, f"gating spread took {t_gate:.2f} s"

    t = time.perf_counter()
    B.estimate_baseline_result(x, fs, config=config, polarity=-1)
    whole = time.perf_counter() - t
    assert (t_intr + t_gate) < 0.35 * whole, (t_intr, t_gate, whole)
