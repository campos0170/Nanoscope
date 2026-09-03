"""DET3-03, round two -- the ninth check now FAILS on a pore blocked ~80% of the time.

WHAT WAS STILL WRONG AFTER ROUND ONE
====================================
Round one gave the health block its frame-free row (``level_occupancy`` +
`alternative_level`) and an advisory on the result, and that was a real repair: the app
stopped being silent. But on the dossier's own fixture -- an open pore at 100 pA
blocked to 60 pA for 81% of a 30 s recording -- ``BaselineHealth.ok`` still read True
with ZERO failing checks. The baseline still settled at 60.04 pA, the events were still
the OPEN-PORE RETURNS reported as 'enhancement', dI/I0 still read 0.665 against a true
0.400 -- now with a note beside it instead of nothing. A note is not a check. What this
file pins is the line that stops the certificate.

WHY IT HAD TO BE A CONJUNCTION, MEASURED
========================================
Round one's own design argument is correct and this file does not overturn it: no
statistic in this block separates an inverted estimate from a valid one, and on the two
occupancies the INVERTED estimate looks HEALTHIER than valid estimates the suite
requires to stay certified::

    estimate                                     level occ   other level
    81% blocked, probed 2.0/0.5   (INVERTED)       0.791        0.191
    renewal 8/s 40 ms @ 2.0/0.5   (valid, pinned)  0.752        0.238
    renewal 8/s 40 ms @ 0.5/0.125 (valid, pinned)  0.770        0.202

Any line on either number that condemns the first condemns the other two. So the third
failing line asks a SECOND question, about the recording rather than about the margin:
below `LEVEL_OCCUPANCY_MIN` -- the floor every recording the owner has on disk clears --
does the pore ever REST at the level being tracked for as long as the window that
measured it? The inverted estimate's longest stretches at the blocked level (p99 394 ms)
are shorter than its own 500 ms window; the two valid ones hold theirs for 1.06 and 3.53
windows. A level the recording never holds for one window was voted for across many
visits to two levels, not measured on one of them.

WHAT THIS DOES NOT REACH, AND THE TESTS THAT SAY SO
===================================================
It is not an inversion detector and conjunction does not make it one.
:func:`test_the_line_is_a_conjunction_and_neither_half_would_do` pins that both halves
are needed. :func:`test_the_reach_is_one_rung_of_the_clogged_family` pins the honest
scope: across a family of clogged pores at each one's own probed window, most rungs stay
certified -- seven because the tracked level's occupancy RISES past the floor as the
inversion gets worse (round one's subject), and the rest because their probed window is
short enough that the open returns do hold it. The advisory is what covers the regime;
this is the part of it that can be checked. See `LEVEL_HOLD_MIN_WINDOWS` for the
measurement tables behind every number quoted here.

COULD THE PRE-EXISTING EIGHT HAVE CAUGHT IT? No, and that is pinned in
:func:`test_the_eight_blind_checks_still_certify_this_estimate`: every one of them reads
the trace through a canonical frame chosen by a polarity derived from the same trace, so
on a mostly-blocked pore they flip together and agree, in green, about a baseline sitting
on the blockade. ``overshoot`` -- the check whose own message names this exact regime --
reads 0.000 and passes.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P

FS = 50_000.0
OPEN_PA, BLOCKED_PA = 100.0, 60.0
APO_600 = pathlib.Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")


def clogged(mean_open_ms=20.0, mean_blocked_ms=80.0, *, seed=0, dur=30.0):
    """The dossier's own fixture: a pore that spends most of its life blocked.

    Exponential open and blocked dwells, so the blockades are a renewal process and not
    a period a window could align to. Returns ``(trace, blocked duty)``.
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
            noise=4.84):
    """``test_baseline_occupancy_reference``'s fixture, reproduced here so the
    comparison is against the trace THAT file certifies rather than a lookalike."""
    rng = np.random.default_rng(seed)
    n = int(round(dur * FS))
    x = np.full(n, float(i0))
    spans, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        d = rng.exponential(dwell_s)
        s, e = int(round(t * FS)), int(round((t + d) * FS))
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


def cfg(rough, base):
    return B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)


def probed(x, fs=FS):
    """The Preprocess panel's own chain: probe -> propose -> characterise WITH health,
    at the polarity the probe measured -- which is the polarity the GUI hands the health
    block, and the whole reason the other eight rows cancel."""
    proposal = P.probe_and_propose(x, fs)
    result = B.characterisation_baseline(
        x, fs, config=proposal.config, health=True,
        polarity=proposal.probe.polarity)
    return proposal, result


# ===========================================================================
# 1. THE BAR -- the certificate stops
# ===========================================================================
def test_the_mostly_blocked_pore_is_no_longer_certified():
    """THE LOAD-BEARING TEST. On the dossier's fixture the block must not read ok, and
    the row that fails must say WHY in the reader's own units."""
    x, duty = clogged()
    assert duty > 0.75, f"fixture drifted: blocked {duty:.0%}"
    proposal, result = probed(x)
    assert (proposal.config.rough_s, proposal.config.base_s) == (2.0, 0.5)
    # The estimate itself is unchanged: this stops the CERTIFICATE, it does not move
    # the baseline. Nothing in these samples says which level is the open pore.
    assert abs(result.median_pA - BLOCKED_PA) < 5.0, result.median_pA

    h = result.health
    assert not h.ok, (
        "the block still certifies a baseline on the blocked level of a pore blocked "
        f"{duty:.0%} of the time: {[c.name for c in h.checks if not c.ok]}")
    failing = [c.name for c in h.checks if not c.ok]
    assert failing == ["level_occupancy"], (
        f"the wrong row fired -- this must be the frame-free one: {failing}")

    row = h.check("level_occupancy")
    assert row.value < B.LEVEL_OCCUPANCY_MIN, row.value
    assert h.level_hold_windows < B.LEVEL_HOLD_MIN_WINDOWS, h.level_hold_windows
    # The message has to name both halves and both readings; a reader who cannot see
    # WHICH other current is on the table cannot act on this.
    m = row.message
    assert "never RESTS" in m and "window that measured it" in m, m
    assert "SETTLED ON THE BLOCKADE" in m, m
    assert "100.0 pA" in m and "60.0 pA" in m, m
    assert "NOT a proof" in m, (
        "the row must not claim to have proved the inversion -- nothing here can", m)


def test_the_eight_blind_checks_still_certify_this_estimate():
    """The hole was always there, and it is not closed by any of the eight: they are all
    read through a canonical frame this trace itself chose. ``overshoot`` -- whose own
    message names 'occupancy past 50%' -- reads ~0 and passes."""
    x, _ = clogged()
    _, result = probed(x)
    h = result.health
    for name in ("length_tracking", "length_rough", "mask_fraction", "mask_reference",
                 "base_vs_state", "baseline_intrusion", "rough_sigma_ratio",
                 "overshoot"):
        row = h.check(name)
        assert row is not None and row.ok, (
            f"{name} now fires on this fixture; if that is real, this file's argument "
            "about WHY the ninth row had to exist is out of date -- read it before "
            "deleting anything")
    assert h.overshoot < B.OVERSHOOT_FLAG, h.overshoot


# ===========================================================================
# 2. THE BOUND -- what must not start failing
# ===========================================================================
@pytest.mark.parametrize("rough, base, want_hold", [(2.0, 0.5, 1.056), (0.5, 0.125, 4.197)])
def test_the_valid_renewal_estimates_stay_certified(rough, base, want_hold):
    """The two correctly-measured estimates that ``test_baseline_occupancy_reference``
    and ``test_s3_detcore_level_occupancy`` require to stay certified. Both sit BELOW
    `LEVEL_OCCUPANCY_MIN`, so the reason they survive is the hold, and it is asserted as
    a number rather than left implicit -- the 2.0/0.5 row clears the line by only 6%
    and a reader changing either constant has to see that."""
    x, _ = renewal(8, dwell_s=0.040)
    h = B.estimate_baseline_result(x, FS, config=cfg(rough, base), polarity=-1).health
    assert h.level_occupancy < B.LEVEL_OCCUPANCY_MIN, h.level_occupancy
    assert h.level_hold_windows == pytest.approx(want_hold, rel=0.02), (
        h.level_hold_windows)
    assert h.level_hold_windows >= B.LEVEL_HOLD_MIN_WINDOWS
    assert h.ok, [c.name for c in h.checks if not c.ok]


def test_an_ordinary_good_recording_is_untouched():
    """A 9%-occupancy renewal fixture at a window that measures it: the row is green on
    both halves, so the new line cannot be reached from here at all."""
    x, _ = renewal(5, dwell_s=0.018)
    h = B.estimate_baseline_result(x, FS, config=cfg(1.0, 0.3), polarity=-1).health
    assert h.level_occupancy > B.LEVEL_OCCUPANCY_MIN, h.level_occupancy
    assert h.ok, [c.name for c in h.checks if not c.ok]
    assert h.check("level_occupancy").ok


@pytest.mark.corpus
def test_the_owners_apo_recording_still_certifies_at_every_window_in_evidence():
    """CONTROL, and the tightest real margin. His -600 mV file at its own proposed
    2 s / 500 ms window holds the open pore for only 0.88 of a window -- BELOW the new
    line -- and is spared entirely by the occupancy half (0.899 against a 0.80 floor).
    That is why this is a conjunction and not a hold check: a line on the hold alone
    condemns the owner's own recording."""
    if not APO_600.exists():
        pytest.skip(f"recording absent: {APO_600}")
    from nanoscope.io.base import open_document
    doc = open_document(APO_600)
    x = np.asarray(doc.channels[2], np.float64)
    fs = float(doc.meta.sample_rate_hz)

    proposal = P.probe_and_propose(x, fs, polarity=-1)
    assert (proposal.config.rough_s, proposal.config.base_s) == (2.0, 0.5)

    for rough, base in ((2.0, 0.5), (1.0, 0.3), (0.5, 0.1)):
        h = B.estimate_baseline_result(x, fs, config=B.DEFAULT_BASELINE.replace(
            rough_s=rough, base_s=base), polarity=-1).health
        assert h.ok and not h.warnings, (rough, base, h.warnings)
        assert h.level_occupancy > B.LEVEL_OCCUPANCY_MIN, (rough, h.level_occupancy)

    h = B.estimate_baseline_result(x, fs, config=B.DEFAULT_BASELINE.replace(
        rough_s=2.0, base_s=0.5), polarity=-1).health
    assert h.level_hold_windows < B.LEVEL_HOLD_MIN_WINDOWS, (
        "his file no longer sits under the hold line; the claim that the occupancy half "
        f"is what spares it is out of date ({h.level_hold_windows:.3f})")


# ===========================================================================
# 3. THE PROPERTIES OF THE LINE ITSELF
# ===========================================================================
def test_the_line_is_a_conjunction_and_neither_half_would_do():
    """Both halves, measured against each other on the two fixtures. This is the test
    that fails on the day someone simplifies the condition to one of them."""
    bad, _ = clogged()
    _, bad_res = probed(bad)
    good, _ = renewal(8, dwell_s=0.040)
    good_res = B.estimate_baseline_result(good, FS, config=cfg(2.0, 0.5), polarity=-1)

    bh, gh = bad_res.health, good_res.health
    # Occupancy alone: the INVERTED estimate is the more occupied of the two, so a line
    # on it that condemns the first condemns the second as well.
    assert bh.level_occupancy > gh.level_occupancy, (
        f"inverted {bh.level_occupancy:.4f} is no longer above valid "
        f"{gh.level_occupancy:.4f}; a failing occupancy threshold may now be possible "
        "and this file's design argument needs re-reading")
    assert gh.level_occupancy < B.LEVEL_OCCUPANCY_MIN
    # Hold alone: it would condemn the owner's own -600 mV file (0.88 windows), which
    # test_the_owners_apo_recording_still_certifies_at_every_window_in_evidence pins.
    assert bh.level_hold_windows < B.LEVEL_HOLD_MIN_WINDOWS <= gh.level_hold_windows
    assert not bh.ok and gh.ok


def test_the_hold_is_measured_in_no_frame():
    """Like the occupancy it conjoins with, the hold is read off ``|current - baseline|``
    -- so the polarity flip that cancels ``overshoot`` leaves it bit for bit alone. If
    it were frame-dependent the new line would cancel on exactly the recordings it is
    for."""
    x, _ = clogged()
    seen = {}
    for pol in (-1, 1):
        b = B.estimate_baseline(x, FS, method="masked_median", polarity=pol,
                                rough_s=2.0, base_s=0.5)
        h = B.baseline_health(x, b, FS, config=cfg(2.0, 0.5), polarity=pol)
        seen[pol] = h.level_hold_windows
    assert seen[-1] == pytest.approx(seen[1], rel=0.02), seen


def test_the_reach_is_one_rung_of_the_clogged_family():
    """THE HONEST SCOPE, pinned so it cannot be read as wider than it is.

    Across the clogged family at each rung's own probed window, this line condemns the
    dossier's case and leaves rungs whose tracked level has climbed back over the floor
    exactly where round one left them -- reported, advised, and not certified as
    verified. A guard that is believed to cover the regime is worse than one that is
    known not to.
    """
    caught, missed = [], []
    for mean_open_ms, mean_blocked_ms in ((20.0, 80.0), (20.0, 200.0), (20.0, 400.0)):
        x, duty = clogged(mean_open_ms, mean_blocked_ms)
        _, res = probed(x)
        assert abs(res.median_pA - BLOCKED_PA) < 5.0, (
            f"blocked {duty:.0%}: the tracker is not on the blockade "
            f"({res.median_pA:.1f} pA); this test is about the case where it is")
        (caught if not res.health.ok else missed).append(round(duty, 2))
        if res.health.ok:
            # The ones it does not reach are not silent -- that is round one's advisory,
            # and it is the reason this line is allowed to be narrow.
            assert [w for w in res.warnings if "NOT CERTIFIED" in w], (
                f"blocked {duty:.0%}: neither condemned nor advised")
    assert caught == [0.81], (caught, missed)
    assert missed, "the family no longer demonstrates the reach; re-read the docstring"


# ===========================================================================
# 4. THE SEAM -- provenance and restored projects
# ===========================================================================
def test_the_hold_reaches_provenance_and_a_restored_block_is_safe():
    """The GUI renders the block from ``to_dict``; a health block restored from a
    project written before this field existed carries None, and nothing may raise on
    it."""
    x, _ = clogged()
    _, res = probed(x)
    d = res.health.to_dict()
    assert d["level_hold_windows"] == pytest.approx(res.health.level_hold_windows)
    assert d["ok"] is False
    assert [c for c in d["checks"] if c["name"] == "level_occupancy"
            and c["level"] == "warn"], d["checks"]
    assert any("never RESTS" in m for m in d["warnings"]), d["warnings"]

    old = dataclasses.replace(res.health, level_hold_windows=None)
    stale = dataclasses.replace(res, health=old)
    # The advisory is keyed to the occupancies, not to the hold, so it still fires.
    assert [w for w in stale.warnings if "NOT CERTIFIED" in w]
    assert old.to_dict()["level_hold_windows"] is None
