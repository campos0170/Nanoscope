"""DET3-03 — a pore blocked most of the time, measured upside down and certified.

THE DEFECT
==========
On a synthetic pore blocked 81% of a 30 s recording (open 100 pA, blocked 60 pA, 277
blockades, true dI/I0 0.400) the app's own journey — probe, apply, detect — settles the
tracked baseline on the BLOCKED level (60.04 pA), reports ``BaselineHealth.ok`` True
with ZERO failing checks, and returns the OPEN-PORE RETURNS as the event population,
labelled 'enhancement' at confidence 1.00 with a median dI/I0 of 0.665 and a median
dwell of 14.9 ms against a true blockade dwell of 64.1 ms. Nothing anywhere said
occupancy, inversion or open pore.

``overshoot`` is the check whose own message names this exact regime, and it read
0.000. It could not do otherwise: it is computed on the CANONICAL frame, the frame is
chosen by a polarity that `derive_polarity` measured from the same trace, and past ~50%
occupancy the global median has moved onto the blockade — so the flip the scalar exists
to detect is applied to the scalar's own frame first and cancels. The repo's own
95%-duty inversion fixture reads 2941.7 with ``polarity=-1`` pinned and 0.0003 with the
polarity derived, which is what every path through the app does.

NOT A REGRESSION FROM THE WORST-STATE CRITERION. The pre-worst-state package
reproduces this byte for byte on the same fixture (proposal 2.0/0.5, baseline 60.04 pA,
eight checks ok, overshoot 0.0) — the blindness predates it.

WHAT IS FIXED, AND WHAT HONESTLY CANNOT BE
==========================================
The block carries TWO frame-free numbers now, and it needed both. ``level_occupancy``
is ``|current - baseline|``, two-sided, in no frame at all, so there is nothing for a
flipped polarity to cancel — but on an inverted estimate that number RISES with the
blockade fraction (0.791 at 81% blocked, 0.897 at 90%, 0.942 at 95%, 0.970 at 97%), so
keying the advisory to it alone caught the shallowest case by 0.9 percentage points and
went silent on every worse one. The app got quieter the more wrong it was, which is the
subject of :func:`test_the_advisory_does_not_go_quiet_as_the_inversion_gets_worse`.

`alternative_level` is the other half: the second mode of the recording's own amplitude
histogram, present in all four of those rows (0.191 / 0.094 / 0.054 / 0.028). Whenever
it holds `ALT_LEVEL_OCCUPANCY_MIN` of the recording the result carries
`level_occupancy_warning`, which names both candidate currents and both readings of
dI/I0. And the comparison of the two occupancies gives the row a failing line that
REACHES a baseline sitting on the blockade: `LEVEL_DECISIVE_RATIO` fails wherever the
second level holds at least half the first's share, which on this fixture is the whole
55%–67% duty band — a baseline on the blocked level, condemned, in the very regime
``overshoot``'s own message names and cannot see.

What no version of this can do is say WHICH of two levels is the open pore, and the
reason is measured rather than conceded — it is the subject of
:func:`test_the_inverted_estimate_and_a_valid_one_are_the_same_place` below. A
21%-off-level recording is arithmetically identical to a 79%-blocked one whose
open-pore returns got masked, and on every statistic this module holds, the INVERTED
estimate looks HEALTHIER than a valid fixture ``test_baseline_occupancy_reference.py``
requires to stay certified. A threshold that condemned the first would condemn the
second; one placed between them would be fitted to two fixtures rather than to the
physics. Four candidate discriminators were measured against the owner's own four
recordings and each died on them — the list is in `alternative_level`'s docstring —
which is why the advisory fires on his files too. That is the honest outcome: on every
one of them the block picked the open pore by majority occupancy and verified nothing,
exactly as it did on the synthetic.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P

FS = 50_000.0
OPEN_PA, BLOCKED_PA = 100.0, 60.0
APO_600 = pathlib.Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")


def clogged(mean_open_ms=20.0, mean_blocked_ms=80.0, *, seed=0, dur=30.0):
    """A pore that spends most of its life blocked. Returns ``(trace, duty)``.

    Exponential open and blocked dwells, so the blockades are a renewal process and not
    a period the windows could align to.
    """
    n = int(FS * dur)
    rng = np.random.default_rng(seed)
    x = np.full(n, OPEN_PA)
    i = blocked = 0
    while i < n:
        o = max(1, int(rng.exponential(mean_open_ms) * 1e-3 * FS))
        b = max(1, int(rng.exponential(mean_blocked_ms) * 1e-3 * FS))
        j, k = min(n, i + o), min(n, i + o + b)
        if k > j:
            x[j:k] = BLOCKED_PA
            blocked += k - j
        i = k
    return (x + rng.normal(0.0, 1.5, n)).astype(np.float32), blocked / n


def renewal(rate, *, seed=3, dur=20.0, dwell_s=0.018, depth=110.0, i0=-34.8,
            noise=4.84, fs=FS):
    """``test_baseline_occupancy_reference``'s own fixture, reproduced here so this
    file's comparison is against the trace THAT file certifies, not a lookalike:
    exponential gaps AND exponential dwells, overlapping arrivals merged."""
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
    return x.astype(np.float32), [(s, e) for s, e in spans if e > s]


def probed(x, fs=FS):
    """The panel's own chain: probe -> propose -> characterise WITH health, at the
    polarity the probe measured (which is what the GUI hands the health block)."""
    proposal = P.probe_and_propose(x, fs)
    result = B.characterisation_baseline(
        x, fs, config=proposal.config, health=True,
        polarity=proposal.probe.polarity)
    return proposal, result


# ===========================================================================
# 1. THE REGIME — the block no longer goes quiet on it
# ===========================================================================
def test_the_mostly_blocked_pore_is_no_longer_certified_in_silence():
    x, duty = clogged()
    assert duty > 0.75, f"fixture drifted: blocked {duty:.0%}"
    proposal, result = probed(x)

    # The estimate itself is unchanged -- this is a fix to what the app SAYS, not to
    # what it computes, and the baseline is still the blocked level.
    assert (proposal.config.rough_s, proposal.config.base_s) == (2.0, 0.5)
    assert abs(result.median_pA - BLOCKED_PA) < 5.0, result.median_pA

    row = result.health.check("level_occupancy")
    assert row is not None, "the frame-free row is missing from the block"
    assert row.value < B.LEVEL_OCCUPANCY_MIN, (
        f"level occupancy {row.value:.4f} is above the certification margin on a pore "
        f"blocked {duty:.0%} of the time"
    )

    # ...and the result says so, in a sentence that names BOTH readings rather than
    # asserting the benign one -- the idiom mask_fraction_floor already uses.
    said = [w for w in result.warnings if "NOT CERTIFIED" in w]
    assert said, f"nothing on the result mentions it: {result.warnings!r}"
    sentence = said[0]
    assert "BLOCKED" in sentence and "open pore" in sentence
    assert "got masked as events" in sentence
    # Both candidate currents, in the reader's own units, and both readings of the
    # number the reader is actually going to quote. Parsed out of the sentence rather
    # than matched as literals, so this asserts on the ARITHMETIC and survives a
    # rewording: dI/I0 is 40/60 = 0.667 under the reading the app took and 40/100 =
    # 0.400 under the other one, and the whole defect is that the first was printed
    # with nothing beside it.
    here, there = re.search(
        r"reads ([0-9.]+) against this baseline and ([0-9.]+) against the other",
        sentence).groups()
    assert float(here) == pytest.approx(0.667, abs=0.01), sentence
    assert float(there) == pytest.approx(0.400, abs=0.01), sentence
    assert re.search(r"another at 100\.\d pA", sentence), sentence


DUTY_LADDER = [
    # (mean blocked dwell ms, roughly how much of the recording the pore is blocked
    # for). One fixture, one knob: the SAME pore, clogged harder. The first row is the
    # dossier's own case; the rest are the ones a line drawn on the tracked level's own
    # occupancy could never reach, because that occupancy climbs from 0.791 to 0.970
    # across them.
    (80.0, 0.81), (200.0, 0.90), (400.0, 0.95), (900.0, 0.97),
]


@pytest.mark.parametrize("mean_blocked_ms, want_duty", DUTY_LADDER)
def test_the_advisory_does_not_go_quiet_as_the_inversion_gets_worse(
        mean_blocked_ms, want_duty):
    """THE REPAIR, and the predicate the first version of this fix failed.

    Moving the fixture's blocked dwell makes the pore MORE clogged -- a strictly worse
    instance of the same defect -- and the tracked level's own occupancy RISES with it,
    straight through the certification margin. An advisory keyed to that number alone
    therefore fired on the 81% case and said nothing at 90%, 95% or 97%: the app got
    quieter the more wrong it was. What this asserts is that every rung of the ladder
    still gets the sentence, and that the number it is keyed to now -- the OTHER
    level's share -- is the one that is present in all of them.
    """
    x, duty = clogged(mean_blocked_ms=mean_blocked_ms)
    assert abs(duty - want_duty) < 0.03, f"fixture drifted: blocked {duty:.0%}"
    _, result = probed(x)

    # The estimate is on the blockade at every rung -- this is the defect, not a
    # fixture that stopped reproducing it.
    assert abs(result.median_pA - BLOCKED_PA) < 5.0, result.median_pA
    h = result.health
    # ...and the tracked level's own margin has sailed past the certification line on
    # three of the four rungs, which is exactly why it cannot be the trigger.
    if duty > 0.85:
        assert h.level_occupancy > B.LEVEL_OCCUPANCY_MIN, (
            "the fixture no longer demonstrates the quieting; re-read this test")

    assert h.alt_level_occupancy >= B.ALT_LEVEL_OCCUPANCY_MIN, (
        f"blocked {duty:.0%}: the open-pore returns hold "
        f"{h.alt_level_occupancy:.4f} of the recording and the block did not read "
        "them as a second level")
    assert h.alt_level_offset_pA == pytest.approx(OPEN_PA - BLOCKED_PA, abs=2.0)
    said = [w for w in result.warnings if "NOT CERTIFIED" in w]
    assert said, (
        f"blocked {duty:.0%}, baseline {result.median_pA:.1f} pA on the blockade, "
        f"and the result says nothing: {result.warnings!r}")


def test_the_row_fails_on_a_baseline_that_tracks_the_blocked_level():
    """The charter's invariant, stated as the one thing the previous fix could not do:
    a HEALTH CHECK that fails on a baseline sitting on the blockade.

    It cannot be done by condemning an inversion -- nothing in the samples separates
    the two readings. It can be done where the estimator's own rule is what fails: at
    56%-67% duty the tracker settles on the blocked level and the level it chose holds
    barely more of the recording than the one it rejected, so the majority rule ran
    without a majority worth the name. `LEVEL_DECISIVE_RATIO` is that line, and this is
    the regime ``overshoot``'s message names ("duty 0.60 -> 150 pA error with the sign
    flipped") and cannot see, because it is read through a frame this trace chose.
    """
    x, duty = clogged(mean_blocked_ms=30.0)
    assert 0.5 < duty < 0.7, f"fixture drifted: blocked {duty:.0%}"
    _, result = probed(x)
    assert abs(result.median_pA - BLOCKED_PA) < 5.0, (
        f"the tracker is not on the blockade here ({result.median_pA:.1f} pA); this "
        "test is about the case where it is")

    h = result.health
    assert h.level_occupancy > B.LEVEL_OCCUPANCY_MAJORITY, (
        "this must NOT be the old failing line -- the tracked level still holds a "
        f"majority ({h.level_occupancy:.4f})")
    row = h.check("level_occupancy")
    assert row is not None and not row.ok, (
        f"blocked {duty:.0%}, baseline on the blockade at {result.median_pA:.1f} pA, "
        f"level occupancy {h.level_occupancy:.4f} against the other level's "
        f"{h.alt_level_occupancy:.4f} -- and the row passes")
    assert not h.ok
    # The overshoot row, on the same estimate, is still green: the check whose own
    # message claims this regime cannot see it, which is why the failing line is here.
    assert h.check("overshoot").ok
    assert "majority worth the name" in row.message
    assert "overshoot" in row.message


def test_the_second_level_is_measured_in_no_frame_either():
    """``alt_level_occupancy`` has to be as frame-free as the row it sharpens, or the
    same flip that cancels ``overshoot`` cancels it too. Negating the trace negates the
    OFFSET and leaves the OCCUPANCY untouched."""
    x, _ = clogged()
    cfg = B.DEFAULT_BASELINE.replace(rough_s=2.0, base_s=0.5)
    seen = {}
    for pol in (-1, 1):
        b = B.estimate_baseline(x, FS, method="masked_median", polarity=pol,
                                rough_s=2.0, base_s=0.5)
        h = B.baseline_health(x, b, FS, config=cfg, polarity=pol)
        seen[pol] = (h.alt_level_occupancy, h.alt_level_offset_pA)
    assert seen[-1][0] == pytest.approx(seen[1][0], rel=0.02), seen

    # ...and directly on the function, where the claim is exact rather than approximate.
    rng = np.random.default_rng(11)
    d = rng.normal(0.0, 2.0, 200_000)
    d[::7] += 40.0
    off, occ = B.alternative_level(d, 2.0)
    noff, nocc = B.alternative_level(-d, 2.0)
    assert nocc == occ
    assert noff == pytest.approx(-off, rel=1e-9)


def test_the_second_level_reaches_provenance_and_the_restored_case_is_safe():
    """The GUI renders the block from ``to_dict``, and a health block restored from a
    project written before these existed carries neither -- the advisory must not raise
    on that, it must simply not fire on what it cannot see."""
    x, _ = clogged()
    _, res = probed(x)
    d = res.health.to_dict()
    assert d["alt_level_occupancy"] == pytest.approx(res.health.alt_level_occupancy)
    assert d["alt_level_offset_pA"] == pytest.approx(res.health.alt_level_offset_pA)

    import dataclasses
    old = dataclasses.replace(res.health, alt_level_occupancy=None,
                              alt_level_offset_pA=None)
    stale = dataclasses.replace(res, health=old)
    said = [w for w in stale.warnings if "NOT CERTIFIED" in w]
    # level_occupancy is 0.791 on this fixture, so the OTHER trigger still fires and
    # the sentence falls back to the form that names no currents.
    assert said and "TWO conductance levels" not in said[0]


def test_the_overshoot_row_no_longer_claims_a_coverage_it_does_not_have():
    """``overshoot``'s message named "occupancy past 50%" as something it catches. It
    catches it only when someone else settled the polarity, and the row now says which
    frame it was measured in and points at the row that is not read through one."""
    x, _ = clogged()
    _, result = probed(x)
    over = result.health.check("overshoot")
    assert over is not None and over.ok, "fixture no longer has overshoot passing"
    assert "canonical" in over.message.lower()
    assert "polarity" in over.message.lower()
    assert "occupancy" in over.message.lower() or "frame-free" in over.message


# ===========================================================================
# 2. WHY IT REPORTS INSTEAD OF CONDEMNING — the measurement behind the choice
# ===========================================================================
def test_the_inverted_estimate_and_a_valid_one_are_the_same_place():
    """THE LOAD-BEARING TEST of the design, not of the code.

    The 81%-blocked fixture (baseline on the blockade, every depth wrong) against a
    renewal fixture at 40 ms / 8 per s, which is a CORRECTLY measured 24%-occupancy
    recording that ``test_baseline_occupancy_reference`` requires to stay certified. If
    the inverted one ever looks WORSE than the valid one on a statistic in this block,
    a failing threshold becomes possible and this file's design argument is out of
    date -- so this test exists to fail on that day rather than to pass quietly.
    """
    bad, _ = clogged()
    _, bad_res = probed(bad)

    good, _ = renewal(8, dwell_s=0.040)
    good_res = B.estimate_baseline_result(
        good, FS, config=B.DEFAULT_BASELINE.replace(rough_s=2.0, base_s=0.5,
                                                    decim=None), polarity=-1)
    assert good_res.health.ok, (
        "the valid comparison estimate is no longer certified: "
        f"{[c.name for c in good_res.health.checks if not c.ok]}")

    def num(res, name):
        row = res.health.check(name)
        return None if row is None else row.value

    for name in ("mask_fraction", "level_occupancy"):
        b, g = num(bad_res, name), num(good_res, name)
        assert b is not None and g is not None
        # Occupancy: the INVERTED recording is the LESS occupied of the two, so no
        # occupancy threshold separates them in the direction a condemnation needs.
        if name == "mask_fraction":
            assert b < g, f"{name}: inverted {b:.4f} is no longer below valid {g:.4f}"
        else:
            assert b > g, f"{name}: inverted {b:.4f} is no longer above valid {g:.4f}"

    assert bad_res.health.overshoot < B.OVERSHOOT_FLAG
    assert good_res.health.overshoot < B.OVERSHOOT_FLAG


def test_the_advisory_is_not_a_verdict_on_the_measurement():
    """The renewal fixture above is measured CORRECTLY and still carries the advisory.
    That is the point: the sentence is about what is not known, and it must not read as
    a defect report on numbers that are right. What it must never do is take
    ``health.ok`` down with it, because the load-bearing sweep in
    ``test_baseline_occupancy_reference`` is what would then start failing."""
    x, _ = renewal(8, dwell_s=0.040)
    res = B.estimate_baseline_result(
        x, FS, config=B.DEFAULT_BASELINE.replace(rough_s=2.0, base_s=0.5,
                                                 decim=None), polarity=-1)
    assert res.health.level_occupancy < B.LEVEL_OCCUPANCY_MIN
    assert res.health.ok, "the advisory took health.ok down with it"
    assert res.health.check("level_occupancy").ok


# ===========================================================================
# 3. THE PROPERTY THE ROW EXISTS FOR — it is measured in no frame
# ===========================================================================
def test_the_row_is_the_same_number_at_either_polarity_and_overshoot_is_not():
    """The whole defect in one assertion. Every other check is read through a frame the
    trace itself chose; flip the polarity and ``overshoot`` swings by orders of
    magnitude while this row does not move at all."""
    x, _ = clogged()
    cfg = B.DEFAULT_BASELINE.replace(rough_s=2.0, base_s=0.5)
    occ, over = {}, {}
    for pol in (-1, 1):
        h = B.baseline_health(
            x, B.estimate_baseline(x, FS, method="masked_median", polarity=pol,
                                   **{k: v for k, v in
                                      (("rough_s", 2.0), ("base_s", 0.5))}),
            FS, config=cfg, polarity=pol)
        occ[pol], over[pol] = h.level_occupancy, h.overshoot
    assert occ[-1] == pytest.approx(occ[1], rel=0.02), occ
    assert max(over.values()) > 50 * max(min(over.values()), 1e-6), (
        f"overshoot no longer depends on the frame ({over}); if that is true the row "
        "this file adds may no longer be needed -- check before deleting it"
    )


def test_the_row_is_emitted_on_every_estimate_including_condemned_ones():
    """A margin the reader cannot see is a margin they will assume. The row is present
    on an event-free trace, on a >50%-duty inversion and on a non-masked method, all of
    which other checks already condemn or exempt."""
    rng = np.random.default_rng(3)
    n = 400_000
    empty = (-100.0 + rng.normal(0, 5.0, n)).astype(np.float32)
    busy = (-100.0 + rng.normal(0, 5.0, n)).astype(np.float32)
    for s in range(0, busy.size - 6_000, 10_000):
        busy[s:s + 6_000] -= 200.0

    for tag, trace in (("event-free", empty), ("60%-duty inversion", busy)):
        h = B.estimate_baseline_result(trace, FS, polarity=-1).health
        assert h.check("level_occupancy") is not None, tag
        assert h.level_occupancy is not None, tag

    block = B.estimate_baseline_result(
        empty, FS, config=B.BaselineConfig(method="block_median", window_s=0.01),
        polarity=-1).health
    assert block.check("level_occupancy") is not None, "non-masked method"
    assert block.mask_fraction is None


def test_the_row_can_fail_and_says_what_that_means():
    """The failing branch, through `baseline_health`'s own contract: it accepts any
    baseline array, and a baseline belonging to a DIFFERENT recording is the case
    nothing else in the block looks for."""
    rng = np.random.default_rng(1)
    x = (-100.0 + rng.normal(0, 5.0, 400_000)).astype(np.float32)
    stale = np.full(x.size, -300.0)          # measured on another pore
    h = B.baseline_health(x, stale, FS, polarity=-1)
    row = h.check("level_occupancy")
    assert row is not None and not row.ok, h.level_occupancy
    assert h.level_occupancy < B.LEVEL_OCCUPANCY_MAJORITY
    assert not h.ok
    assert "different recording" in row.message.lower()
    assert "never visits" in row.message


def test_the_number_is_carried_into_provenance():
    """The GUI renders the block from ``to_dict``; a row that does not survive the
    serialisation is a row a restored result cannot show."""
    x, _ = clogged()
    _, res = probed(x)
    d = res.health.to_dict()
    assert d["level_occupancy"] == pytest.approx(res.health.level_occupancy)
    assert any(c["name"] == "level_occupancy" for c in d["checks"])
    assert any("NOT CERTIFIED" in w for w in res.warnings)


# ===========================================================================
# 4. THE BOUND — the owner's own recording, no new API in the assertions
# ===========================================================================
@pytest.mark.skipif(not APO_600.exists(), reason="the owner's -600 mV file is absent")
def test_the_apo_600_recording_is_untouched():
    """CONTROL, and the two assertions that matter are written with NO new API so they
    run identically against a package without this change: the owner's -600 mV file
    still gets the 2.0 / 0.5 proposal and still gets its certificate. Nothing about the
    measurement moved -- the numbers in the table, the event count and the 521 ms clog
    are all downstream of these two and are pinned by the detection suite.

    WHAT DID CHANGE, DELIBERATELY: this file DOES now carry the advisory. Its second
    level (-120.7 pA, 5% of the recording) is the same structure the inverted synthetic
    has, and `alternative_level` lists the four discriminators that were measured
    against this very recording and died on it. So there is no honest rule that speaks
    on the synthetic and stays silent here, and a rule that stayed silent here would be
    claiming a discrimination this module does not have. What the block must not do is
    withdraw the CERTIFICATE, and it does not.
    """
    from nanoscope.io import base as iobase

    reader = iobase.select_reader(APO_600)(APO_600)
    doc = reader.load()
    channel = next(c.index for c in doc.meta.channels
                   if str(c.kind).endswith("CURRENT") and not c.is_dead)
    x = np.asarray(doc.channels[channel], dtype=np.float32)
    fs = float(doc.meta.sample_rate_hz)

    proposal, result = probed(x, fs)
    assert (proposal.config.rough_s, proposal.config.base_s) == (2.0, 0.5)
    assert result.health.ok, [c.name for c in result.health.checks if not c.ok]
    assert result.health.level_occupancy > 0.85
    assert result.health.check("level_occupancy").ok
    # The advisory that does fire says what it is about, and it is not a defect report:
    # the ratio here is 0.06x, four times clear of the failing line.
    said = [w for w in result.warnings if "NOT CERTIFIED" in w]
    assert said and "only chosen" in said[0]
    assert (result.health.alt_level_occupancy
            < 0.25 * B.LEVEL_DECISIVE_RATIO * result.health.level_occupancy)
