"""Measure a recording's own timescales, and PROPOSE baseline parameters from them.

WHY THIS MODULE EXISTS
======================
`baseline.DEFAULT_BASELINE` (rough 0.5 s / base 0.1 s) is a defensible floor and a
provably wrong answer for at least two of the four recordings in evidence. The apo
``_0006_`` file has a **521 ms partial-clog state at t = 37.13 s**, and a 500 ms rough
window sitting inside a 521 ms state *is* that state -- the rough residual there is
~0, nothing gets masked, and the baseline lands 22 pA off (-56.6 against a true
-34.5). It needs ``rough_s = 2.0``. ``ferritin_data3`` is a gating pore stepping
296 pA with a ~35 ms level dwell, where a 2 s window cannot follow the steps at all
and the notebook uses ``rough_s = 0.03`` -- 60x shorter. **No single global default
can exist**, so the app measures each recording and proposes.

PROPOSE, NEVER DECIDE (spec 2.6)
================================
Nothing here is ever auto-applied. `propose_baseline` returns a `BaselineProposal`
for a human to confirm, and the two reasons are measured, not stylistic:

1. **The obvious sigma estimator is wrong on this data.** A delta-based sigma reads
   7x low on these already-low-passed exports (see `baseline.upper_mad_sigma`), and
   every quantity built on it -- mask fraction, dwell percentiles, the gating flag --
   came out garbage. The substitute used here is defensible but *not* trustworthy
   enough to act on unseen: it over-reads 3-4x on a gating pore.
2. **The reference's own QC triple is a degenerate objective.** Optimising
   ``qc_open_pore_level`` / ``drift`` / ``mad_noise`` monotonically selects the
   SHORTEST possible window -- a degenerate ``0.01/0.002/1`` set beats the shipped
   parameters on all three for every corpus file. Auto-tuning on them would
   reproduce the exact failure this whole change removes. So the defence is
   architectural: a human confirms.

THE RULE (spec 2.5), AND WHAT IT REPRODUCES
===========================================
::

    not gating:  rough_s = snap(ROUGH_MULTIPLE x longest_state)
                 base_s  = snap(rough_s / ROUGH_BASE_RATIO)
                 base_s  = max(base_s, snap_up(BASE_STATE_RATIO_MIN x state_dwell))
                 rough_s = max(rough_s, snap_up(ROUGH_BASE_RATIO x base_s))  # if floored
    gating:      base_s  = snap(level_dwell / LEVEL_BASE_DIVISOR)
                 rough_s = snap(ROUGH_MULTIPLE x base_s)
    always:      decim   = derived (`baseline.derive_decim`), never proposed
    not gating:  ...then RANKED -- see `rank_windows`

AND THEN THE RANKING, BECAUSE A BOUND IS NOT A CHOICE
=====================================================
Everything above is arithmetic on two measured timescales bounded from below by a
third. It ends on the SHORTEST window `baseline.baseline_health` tolerates, which is a
choice made by legality: the floor is a lower bound and the rule never looks above it.
`rank_windows` looks above it, on the app's own ``baseline_intrusion`` statistic --
the direct measurement of how far the tracked baseline sits inside the states -- read
TWICE per candidate: the MEDIAN over the states, and the WORST single state, because
the median is blind to one catastrophic state (the -600 mV file's 521 ms clog, ridden
into at a median of exactly 0.0 -- see `WORST_STATE_INTRUSION_MAX`). A window that
rides into a state loses to one that stays out; then least median intrusion, shortest
first on a tie. On the -700 mV apo recording that is the difference between a stamped
open pore of -48.857 pA (1.33% off the app's own independent estimate) and -48.218
(0.0005% off). Read `rank_windows` for the ladder, the ceiling, the tie-break and
what it moves.

**This is NOT a search over the reference's QC triple**, which is the thing point 2
below forbids and which would select the shortest window there is. It is a search over
a statistic that is bounded, two-sided in its failure modes and MONOTONE THE OTHER WAY
(a wider window intrudes less), so the bound that stops it is a measured zero rather
than a taste. The proposal it lands on is still a proposal: a human confirms it.

THE FLOOR, AND WHY IT IS NOT A FOURTH TIMESCALE
===============================================
The first two lines size ``base_s`` off ``rough_s``, which is sized off the p99.9 of
the dwells -- the LONGEST state the estimator has to sail over. Nothing in that chain
ever looked at the state a tracking median actually MEETS, and `baseline.
baseline_health`'s ``base_vs_state`` check does exactly that: it divides ``base_s`` by
the MEDIAN state dwell (`baseline.state_dwell_s` over the reference mask) and condemns
the run below `baseline.BASE_STATE_RATIO_MIN`. The two statistics differ by 1.5x on
the -700 mV apo recording and 6.4x on ferritin_data2, so the proposer could -- and
deterministically did -- offer a window its own health block then condemned:
0.1 / 0.03 s at **1.38x a floor of 3.0**, twice in a row, for ever.

`base_floor_s` closes it by dividing by the SAME number the check divides by
(`baseline.reference_state_dwell_s`, verified bit-identical to the check's divisor on
all six recordings in evidence) and snapping UP, so the shortest ``base_s`` this
module can propose is the shortest one on its grid that the check certifies.

WHERE THAT IS A GUARANTEE, AND WHERE IT IS ONLY AN IMPROVEMENT
--------------------------------------------------------------
It is a guarantee whenever ``base_vs_state`` reads the REFERENCE mask, which is
whenever the proposed ``rough_s`` is below `baseline.reference_rough_s` -- every
recording in evidence, and everything longer than ~20 s at any window this rule
proposes. It is not one when the proposed ``rough_s`` reaches or passes that window:
`baseline.reference_states` then returns ``None`` and the check measures the state
dwell on the CONFIGURED pass, a different mask and a different number, while the floor
was built on the reference one. Measured on a 3 s synthetic with 120 ms states
(reference window 0.75 s): the floor lifts ``base_s`` 0.1 -> 0.3 s and ``rough_s`` to
1.0, the reference pass drops out, the configured pass reads a 109 ms state against
the reference pass's 77 ms, and ``base_vs_state`` still warns -- at 2.75x rather than
the 1.29x it read without the floor. Pinned by
``tests/analysis/test_fup_proposer_floor.py``, not left as folklore, and
``base_vs_state``'s own remedy detects the case and stops naming the probe there.

The floor is skipped on a credibly gating pore, which is not an exception to the
contract but the same clause of it: ``base_vs_state`` writes
``short = (not gating) and ...``, and its ``gating`` is the predicate
`BaselineProbe.gating_credible` already spells. See `base_floor_s` for what applying
it there would have cost ferritin_data3.

Measured effect on the corpus -- exactly one proposal moves, which is the one the
check was condemning:

===============  ==============  ==============  ================  ==============
recording        median state    floor           BEFORE            AFTER
===============  ==============  ==============  ================  ==============
-500 mV apo      24.3 ms         0.1 s           1.0 / 0.3         **unchanged**
-600 mV apo      20.7 ms         0.1 s           2.0 / 0.5         **unchanged**
-700 mV apo      21.7 ms         0.1 s           0.1 / 0.03        **0.5 / 0.1**
ferritin_data1   6.85 ms         0.03 s          0.5 / 0.1         **unchanged**
ferritin_data2   2.76 ms         0.01 s          0.5 / 0.1         **unchanged**
ferritin_data3   3.34 ms         (gating: none)  0.03 / 0.01       **unchanged**
===============  ==============  ==============  ================  ==============

The -700 mV row is chartered to move. At 0.1 / 0.03 that recording's ``base_vs_state``
reads 1.3819 and ``baseline_intrusion`` 0.0416 (health NOT ok, two checks condemning);
at 0.5 / 0.1 it reads 4.6062 and 0.0043 with all eight checks green, and its stamped
open pore moves from -51.661 pA to -48.857 against the app's own independent
event-excluded estimate of -48.218 -- 7.14% out to 1.33% out.

"gating" here means `BaselineProbe.gating_credible`, not the bare score: the gating
statistic reads the same on a stepping pore and on a pore blocked a quarter of the
time by events long against its 20 ms block, and the rule INVERTS between them, so a
false positive does not degrade the proposal -- it replaces it with one measured
against the wrong quantity. Measured: 27% occupancy at 18 ms dwells flips the proposal
from 0.3/0.1 to 0.03/0.01 and the median measured depth from 110.0 pA to 46.1.

Measured on the four recordings in evidence, from the data alone:

===============  ==========  =========  =========  ==============  ================
recording        longest     gating     level      PROPOSED        known-good
===============  ==========  =========  =========  ==============  ================
ferritin_data1   194 ms      no (1.7)   --         0.5 / 0.1       0.5 (see below)
ferritin_data2   163 ms      no (4.7)   --         **0.5 / 0.1**   0.5 / 0.1
ferritin_data3   (gating)    yes (13.8) 28.5 ms    **0.03 / 0.01** 0.03 / 0.01
apo ``_0006_``   496 ms      no (0.5)   --         **2.0 / 0.5**   2.0 / 0.5
===============  ==========  =========  =========  ==============  ================

Three exact reproductions. ``ferritin_data1`` is the honest partial: the rule's
``rough_s`` lands on its shipped 0.5 s, but the notebook ran the SINGLE-PASS
``robust_median`` there, and no measurement in this probe chooses between the one-
and two-pass estimator -- the spec gives no rule for it either, and its own table
leaves data1's row unticked. The probe reports the timescales; the method stays the
user's.

WHY ``ROUGH_MULTIPLE`` IS 3 AND NOT THE SPEC'S 4
================================================
The spec's "``rough_s`` >= ~4x the longest dwell" is paired with ITS dwell numbers
(data1 185 ms, data2 100 ms, apo 521 ms), which came from three different threshold
and closing settings. `longest_state_s` here is ONE statistic applied to all four,
and it reads ~1.3x higher than the spec's per-file numbers on the two files where
they can be compared (data2 163 vs 100, data1 194 vs 185). 4 / 1.33 = 3. The
constant and the statistic are a matched pair; neither can be changed alone.

WHAT THIS MODULE MAY NOT DO
===========================
It never constructs a `BaselineConfig`. Every config it returns is
``DEFAULT_BASELINE.replace(...)`` touching ONLY ``rough_s`` / ``base_s`` / ``decim``,
which is not a style choice: it makes it structurally impossible for a probe to
invent a ``mask_sigma``, a ``dilate_ms`` or a ``despike_size``, and it keeps
`baseline.DEFAULT_BASELINE` the single baseline-parameter literal in the app that
``tests/analysis/test_baseline_seam.py`` guards. An undecidable probe returns
``DEFAULT_BASELINE`` itself, by identity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .baseline import (
    BASE_STATE_RATIO_MIN,
    BASELINE_INTRUSION_MAX,
    BASELINE_PARAM_KEYS,
    DEFAULT_BASELINE,
    GATING_BLOCK_S,
    GATING_THRESHOLD,
    MASK_FRACTION_CEILING,
    REFERENCE_MIN_ROUGH_S,
    REFERENCE_ROUGH_S,
    REFERENCE_WINDOWS_MIN,
    GATING_BLOCKS_MIN,
    ROUGH_BASE_RATIO,
    SIGMA_STRIDE,
    SIGMA_STRIDE_MIN_N,
    BaselineConfig,
    block_medians,
    config_intrusion,
    derive_decim,
    gating_spread,
    measured_polarity,
    reference_rough_s,
    reference_state_dwell_s,
    reference_states,
    require_finite,
    upper_mad_sigma,
)
from .cancel import check_cancelled

__all__ = [
    "BaselineProbe",
    "BaselineProposal",
    "BaselineCandidate",
    "probe_timescales",
    "propose_baseline",
    "probe_and_propose",
    "proposal_summary",
    "snap_to_grid",
    "snap_up_to_grid",
    "base_floor_s",
    "candidate_windows",
    "intrusion_cost",
    "rank_windows",
    "RANK_MAX_CANDIDATES",
    "WORST_STATE_INTRUSION_MAX",
    "BASE_STATE_RATIO_MIN",
    "PROBE_ROUGH_S",
    "PROBE_MIN_ROUGH_S",
    "PROBE_MASK_SIGMA",
    "PROBE_CLOSE_GAP_S",
    "LONGEST_STATE_Q",
    "ROUGH_MULTIPLE",
    "ROUGH_BASE_RATIO",
    "GATING_BLOCK_S",
    "GATING_THRESHOLD",
    "LEVEL_BLOCK_S",
    "LEVEL_STEP_SIGMA",
    "LEVEL_BASE_DIVISOR",
    "GRID_MANTISSAS",
    "GRID_MIN_S",
    "GRID_MAX_S",
]

# ---------------------------------------------------------------------------
# constants -- every one of these is a measurement, and says which
# ---------------------------------------------------------------------------

#: Rough window of the PROBE's own baseline, seconds, and the two constants that size
#: it down on a short recording. **They are `baseline.REFERENCE_ROUGH_S` and its
#: pair, not copies of them.** `baseline.reference_mask_fraction` measures the same
#: occupancy for the health block, the two numbers are compared against each other
#: (`baseline.MASK_REFERENCE_MAX_RATIO`), and two occupancies measured at two window
#: scales would be a comparison of the two window choices. See those constants for
#: the measurements behind 5.0 / 0.5 / 4.0 and `reference_rough_s` for the sizing.
PROBE_ROUGH_S = REFERENCE_ROUGH_S
PROBE_MIN_ROUGH_S = REFERENCE_MIN_ROUGH_S
PROBE_WINDOWS_MIN = REFERENCE_WINDOWS_MIN

#: Mask threshold for the probe's own dwell measurement, in `upper_mad_sigma` units.
#: The estimator's own default, deliberately: the states the probe reports are the
#: states the estimator will mask.
PROBE_MASK_SIGMA = 3.5

#: Gaps up to this long are closed before the dwell distribution is read (2.6c's
#: "morphological closing"; done on the run boundaries, which is the same operation
#: at O(#runs) instead of O(n) -- 16 ms rather than 297 ms on the 8.6 M file).
#:
#: **Without it the apo file's 521 ms clog is invisible.** That state is shallow --
#: 83% of its samples cross 3.5 sigma, 17% do not -- so at gap 0 it fragments into
#: runs of at most 26.16 ms, p99.9 reads 24.6 ms and ``rough_s`` comes out 20x
#: under-sized. Measured: gap 0 -> 24.6 ms, 0.4 ms -> 339, 1 ms -> 497, 2 ms -> 500.
#: The ceiling comes from the other direction: ferritin_data2's DISTINCT deep events
#: sit ~1-2 ms apart, and closing them together takes its p99.9 from 101 ms (gap 0)
#: to 147 (0.4 ms), 168 (1 ms), 227 (2 ms), 258 (4 ms). 1 ms is where the apo clog is
#: whole and data2 still proposes its shipped 0.5 s.
PROBE_CLOSE_GAP_S = 0.001

#: Percentile of the closed dwell distribution reported as ``longest_state_s``. The
#: max is too fragile (one cluster of data2 events with sub-ms gaps reads 271-443 ms
#: depending on decim, and would propose 1.0 s there); p99 misses the apo clog
#: entirely (59.8 ms), since it is ONE state in a 60 s recording.
LONGEST_STATE_Q = 99.9

#: ``rough_s = ROUGH_MULTIPLE * longest_state_s``. See the module docstring for why
#: this is 3 rather than the spec's 4 -- it is matched to `LONGEST_STATE_Q` and
#: `PROBE_CLOSE_GAP_S` and cannot be changed alone. Margins on the log grid at 3:
#: data1 0.583 s (snap boundary 0.707, 21%), data2 0.490-0.528 (31-44%),
#: apo 1.47-1.49 (boundary 1.414, 4.7% -- the tight one, but consistent across all
#: twelve probe settings measured).
ROUGH_MULTIPLE = 3.0

#: `GATING_BLOCK_S` and `GATING_THRESHOLD` are IMPORTED from `baseline`, not defined
#: here: `baseline_health` asks the same question ("are the long states events, or is
#: the open level itself stepping?") to decide whether to judge ``base_s`` against the
#: state dwell at all, and the proposal and the verdict must not answer it differently.

#: Block length for the conductance-LEVEL analysis on a gating pore. 2 ms resolves a
#: ~35 ms level dwell; the 20 ms gating block cannot.
LEVEL_BLOCK_S = 0.002

#: A jump between successive `LEVEL_BLOCK_S` medians beyond this many robust sigmas
#: of the jump distribution is a conductance STEP. Measured on ferritin_data3 the
#: mean step-to-step interval is 24.3 ms at 4 sigma, 28.5 at 6, 33.2 at 8, against
#: the 34.7 ms the spec quotes -- all three propose the same ``base_s``.
LEVEL_STEP_SIGMA = 6.0

#: ``base_s = level_dwell_s / LEVEL_BASE_DIVISOR`` on a gating pore -- "base_s shorter
#: than the shortest genuine level dwell", with the mean standing in for the short
#: tail. **The spec's "11.5 ms (p10)" is not reproducible**: a percentile of a
#: step-to-step interval distribution is quantised at the block length, so p10 reads
#: exactly one block on all four recordings and carries no information. The MEAN is
#: recoverable (28.5 ms measured against the spec's 34.7) and /3 reproduces
#: ferritin_data3's shipped 0.01 across every step threshold measured.
LEVEL_BASE_DIVISOR = 3.0

#: Proposals snap to a 1-2-3-5 decade grid, log-nearest. Not cosmetic: it is what
#: turns 0.505 into the notebook's 0.5 and 0.0095 into its 0.01, and a 1-2-5 grid
#: cannot express ferritin_data3's 0.03 at all.
GRID_MANTISSAS = (1.0, 2.0, 3.0, 5.0)
GRID_MIN_S = 0.002
GRID_MAX_S = 10.0

#: Most windows `rank_windows` will ESTIMATE, including the proposal itself. A cost
#: bound, and it is the only one of the FOUR bounds on the search that is a number
#: rather than a measurement -- the others are `baseline.reference_rough_s` (the
#: ceiling), `base_floor_s` (the bound on the way down; see `candidate_windows` for
#: both) and the stop rules in `rank_windows`.
#:
#: **It bounds the ESTIMATES, not the ladder.** `candidate_windows` returns every rung
#: between the two measured bounds and this number is spent by `rank_windows`; a cap
#: applied to a two-sided ladder before the walk would delete rungs the search needs
#: rather than rungs it was never going to reach.
#:
#: Measured per candidate on the owner's own recordings, conditioned exactly as the
#: app conditions them (estimate + detrend + the intrusion loop): **1.5 s** at 12.5 M
#: samples (-500 mV), **0.9-1.5 s** at 9.1 M (-700 mV), **0.26 s** at 3.0 M (-600 mV).
#: 4 is therefore a worst case of ~6 s on his longest file and a typical case of TWO
#: candidates, because the two stop rules are mutually exclusive in what they spend:
#: the whole corpus measures 2, 2, 2, 2, 3 and 0 candidates (-500, -600, -700,
#: ferritin_data1, ferritin_data2, ferritin_data3 -- the last exempt; -600's descent
#: stops on its first rung, which rides, and data2's phase 1 walks through its riding
#: rung to the clean one above), against 1, 1, 2, 2, 2 and 0 for the one-sided
#: ladder. The cap exists for the recording that never reaches a clean rung, where
#: the walk would otherwise run to the ceiling.
RANK_MAX_CANDIDATES = 4

#: A candidate whose tracked baseline followed MORE than this share of any SINGLE
#: state's own excursion has ridden INTO that state, and the ranking may not offer it
#: over a window that stays out of every state -- whatever the two windows' MEDIAN
#: intrusions read.
#:
#: WHY THE MEDIAN NEEDS THIS BESIDE IT. ``baseline_intrusion`` is a median over the
#: states of one recording (239 on the -600 mV apo file, 982 on -700 mV), and one
#: catastrophic state cannot move a median over hundreds -- structurally, not
#: unluckily. Measured on the owner's -600 mV recording at 1.0 / 0.3 s: the median
#: reads exactly 0.0 while the tracked baseline rides into the 521 ms partial clog at
#: t = 37.13 s to -50 to -51 pA against a -34.5 pA open pore. Run through the app's
#: own detection, that window does not detect the clog AT ALL (at 2.0 / 0.5 s it is
#: ONE 519.5 ms event, depth 28.9 pA) and instead invents ~30 shallow events on the
#: collapsed local baseline inside 36.5-38.5 s -- and all eight health checks stay
#: green both ways, because the intrusion check reports the same blind median.
#:
#: WHY 0.5, AND NOT ANOTHER NUMBER. The rough pass is a rolling MEDIAN, and a median
#: sits inside a state exactly when the state fills more than half of its window; a
#: baseline that followed more than half of a state's own excursion has crossed onto
#: the state's side of it. The corpus separates at exactly that line, measured over
#: the shared reference mask (`baseline.tracked_intrusion`'s ``worst_fraction``,
#: taken over the states longer than each candidate's own ``base_s`` -- see there for
#: why the short states' readings are noise and are excluded) on every rung of the
#: ladder on the three apo recordings, ferritin_data2, and the three clogged
#: synthetics of ``tests/analysis/test_rp_two_sided_ranking_core.py``:
#:
#: * every window in evidence whose baseline ENTERS a long state reads **>= 0.556**
#:   (-600 mV at 1.0/0.3: 0.556 on a 473 ms state; -600 at 0.5/0.1: 0.987; -500 at
#:   0.3/0.1: 0.973; ferritin_data2 at 1.0/0.3: 0.782 on a real 395 ms deep state at
#:   t = 99.3 s -- a SECOND recording carrying the same failure, found by this
#:   criterion rather than fitted to it; the clogged synthetics: 0.936-1.007);
#: * every window that STAYS OUT reads **<= 0.061** (the clogged synthetics'
#:   2.0/0.5 rungs: 0.009-0.061; -600 at 2.0/0.5: -0.112 on the 521 ms clog it sails
#:   over), or has NO state longer than its tracking window at all (both -700 mV
#:   rungs, the -500 and data1/data2 winners), which reads as ``None`` and cannot
#:   ride.
#:
#: 0.5 sits in the 9x gap between the two populations. It is a RANKING
#: criterion, not a ninth health check: the health block reports what a chosen
#: configuration measured (`baseline.BaselineHealth`, "reports never objectives"),
#: while this is the objective the proposer chooses BY -- the same layering that
#: keeps `BASELINE_INTRUSION_MAX` a tolerance there and `intrusion_cost` a cost here.
WORST_STATE_INTRUSION_MAX = 0.5

#: Stride for the two robust-sigma statistics on a long trace. A median of every 4th
#: sample of a 3 M-sample recording is the same number to ~0.1% and four times
#: cheaper; the DWELL measurement is never subsampled, because it must resolve the
#: run boundaries. `baseline` owns them because `baseline.gating_spread` -- which is
#: one of the two statistics -- is shared with the health block.
_SIGMA_STRIDE = SIGMA_STRIDE
_SIGMA_STRIDE_MIN_N = SIGMA_STRIDE_MIN_N


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BaselineProbe:
    """What a ~200 ms look at one recording measured. Reports only; decides nothing.

    Attributes:
        fs: Sample rate, Hz.
        n: Samples examined (the whole channel -- never a zoom rectangle; the apo
            file's decisive state is at t = 37 s of 60).
        polarity: The frame everything below was measured in, ``{-1, +1}``.
        sigma_pA: One-sided upper MAD of the probe residual -- the scale the mask
            threshold is in. See `baseline.upper_mad_sigma` for why it is not a
            two-sided MAD and emphatically not a delta-based sigma.
        noise_pA: Within-block spread of the `GATING_BLOCK_S` blocks. A second,
            gating-IMMUNE noise floor: a conductance step lasts far longer than one
            block, so it moves the block MEDIAN rather than the within-block spread.
            On ferritin_data3 this reads 19.5 pA against ``sigma_pA``'s 53.5 and a
            true 13.8.
        longest_state_s: `LONGEST_STATE_Q` of the gap-closed masked dwells -- the
            longest state the estimator would have to sail over.
        n_states: How many closed runs that percentile came from.
        state_dwell_s: `baseline.reference_state_dwell_s` -- the MEDIAN state dwell
            over the reference window, which is the number ``base_vs_state`` divides
            ``base_s`` by. **Not `longest_state_s`, and not a second estimate of it**:
            ``rough_s`` is sized against the longest state it must sail over (p99.9)
            while ``base_s`` is sized against the state a tracking median meets over
            and over (median). The two read 31 ms and 21.7 ms on the -700 mV apo
            recording and 163 ms against 2.8 ms on ferritin_data2, and it was the
            proposer having only the first of them that let it offer a window the
            checks condemn. ``None`` on an unusable probe; ``nan`` when the recording
            is too short for a reference window.
        mask_fraction: Fraction of samples in the probe's mask, undilated.
        gating: ``gating_score > GATING_THRESHOLD``.
        gating_score: ``IQR(block medians) / noise_pA``.
        level_dwell_s: Mean interval between conductance steps; ``None`` unless
            ``gating`` (it is meaningless, and dominated by events, otherwise).
        probe_rough_s: The probe's own rough window, which shrinks on a short
            recording.
        usable: False when the probe measured NO timescale at all. The proposal is
            then `baseline.DEFAULT_BASELINE`, said out loud. A partial answer (a
            stepping pore whose step rate could not be resolved) stays usable -- the
            dwell-based branch is still a better answer than the global default, and
            `propose_baseline` says which branch it took.
        notes: Caveats, in the user's words -- the reason for a refusal, or the part
            of the measurement that did not come out.
        elapsed_s: Wall time of the PROBE STAGE ONLY (`probe_timescales`) -- not of
            `propose_baseline`, not of `rank_windows`, and not of the GUI's "Measure
            this recording" button, which runs conditioning + this + the ranking + a
            health second pass. Re-measured 2026-09-01 on the owner's apoferritin
            recordings, three presses each in one process, medians: 140 ms (-1000 mV
            excerpt, 0.75 M samples), 343 ms (-600 mV, 3.0 M), 1,073 ms (-700 mV,
            9.1 M) -- ~115-190 ms per million samples, and the FIRST press of a file is
            colder (585 ms and 1,942 ms on the two larger files). An earlier table
            here ("233 ms at 3.0 M, ~70 ms per million") did not survive
            re-measurement (537 ms was seen on the same 3.0 M file cold), so treat
            these as this bench on this date. The whole button costs ~2 s at 3 M and
            ~6.6 s at 9 M, which is why the GUI runs it off-thread rather than in
            front of the user.
    """

    fs: float
    n: int
    polarity: int
    sigma_pA: float
    noise_pA: float
    longest_state_s: float
    n_states: int
    mask_fraction: float
    gating: bool
    gating_score: float
    level_dwell_s: float | None
    probe_rough_s: float
    usable: bool
    notes: tuple[str, ...] = ()
    elapsed_s: float = 0.0
    #: LAST, and with a default, deliberately. Every construction of this class in
    #: the tree is by keyword, and appending rather than inserting keeps it that way
    #: for anything outside it that builds a probe positionally.
    state_dwell_s: float | None = None

    @property
    def gating_credible(self) -> bool:
        """``gating``, believed -- the score is high AND events cannot explain it.

        `gating_score` is the spread of 20 ms block medians over the within-block
        noise, and **it cannot tell a pore stepping between conductance levels from a
        pore that is simply blocked a lot of the time by events long compared with
        20 ms**: past ~25% occupancy the block median itself lands inside a blockade
        often enough to look like a second level. `tests/analysis/test_baseline_probe`
        has asserted that limit since the flag was written; what was missing is that
        the rule INVERTS on the answer, so a false positive there does not degrade the
        proposal, it replaces it with one measured against the wrong quantity.
        Measured (renewal process, 110 pA blockades, 18 ms mean dwell): at 27%
        occupancy the score jumps 0.84 -> 17.2, the proposal flips 0.3/0.1 -> 0.03/0.01
        and the median measured depth goes 110.0 -> 46.1 pA.

        The second measurement that separates them is one the probe already takes.
        A stepping pore distributes its time over levels, so the ONE-SIDED mask
        against a wide baseline stays modest -- ferritin_data3, the genuinely gating
        recording in evidence, masks 15.4% while scoring 13.8. A pore this blocked
        masks a quarter of the trace or more: the misfires above measure 0.271, 0.279,
        0.308, 0.328, 0.330 and 0.354. `baseline.MASK_FRACTION_CEILING` (0.25) is
        already the app's line for "this mask is too much of the trace to be events",
        and it sits 1.6x above data3 and 1.08x below the nearest misfire.

        NOT folded into `gating` itself, deliberately: that field is DEFINED as
        ``gating_score > GATING_THRESHOLD``, it is displayed as the statistic it is,
        and a measurement that quietly answered a different question than its name
        would be the same defect one layer down. The rule reads this; the report
        reports that.
        """
        return bool(self.gating and self.mask_fraction <= MASK_FRACTION_CEILING)

    def to_dict(self) -> dict[str, Any]:
        """Flat and JSON-safe, for a log line or a provenance side-note."""
        return {
            "probe_fs": float(self.fs),
            "probe_n": int(self.n),
            "probe_polarity": int(self.polarity),
            "probe_sigma_pA": float(self.sigma_pA),
            "probe_noise_pA": float(self.noise_pA),
            "probe_longest_state_s": float(self.longest_state_s),
            "probe_n_states": int(self.n_states),
            "probe_state_dwell_s": (None if self.state_dwell_s is None
                                    else float(self.state_dwell_s)),
            "probe_base_floor_s": base_floor_s(self),
            "probe_mask_fraction": float(self.mask_fraction),
            "probe_gating": bool(self.gating),
            "probe_gating_credible": bool(self.gating_credible),
            "probe_gating_score": float(self.gating_score),
            "probe_level_dwell_s": (None if self.level_dwell_s is None
                                    else float(self.level_dwell_s)),
            "probe_rough_s": float(self.probe_rough_s),
            "probe_usable": bool(self.usable),
            "probe_notes": list(self.notes),
        }


@dataclass(frozen=True)
class BaselineCandidate:
    """One window the ranking ESTIMATED, and what `baseline_intrusion` read on it.

    A row of evidence, not a decision: `rank_windows` returns the whole ladder it
    measured -- including the candidates that lost and the one that could not be
    measured at all -- so the sentence the user reads is the table the app ranked on
    rather than a summary of it.

    Attributes:
        rough_s: The pass-1 window.
        base_s: The pass-2 window (already floored by `base_floor_s` where it binds).
        fraction: `baseline.config_intrusion`'s fraction -- the share of each state's
            own excursion this window's baseline followed, SIGNED, and bit-identical
            to what `baseline.baseline_health` will report for the same window.
            ``None`` when no state had a measurable depth.
        n_states: How many states that median came from; ``None`` with ``fraction``.
        cost: `intrusion_cost` of ``fraction`` -- the MEDIAN half of what is ranked
            on. ``None`` when ``fraction`` is.
        chosen: This is the window the proposal now carries.
        worst_fraction: The single WORST state's share -- the largest per-state
            ``intrusion/depth`` among the visited states LONGER than this
            candidate's own ``base_s``, signed like ``fraction``. The half of the
            objective the median cannot carry: one state ridden fully into does not
            move a median over hundreds (see `WORST_STATE_INTRUSION_MAX` for the
            -600 mV numbers, and `baseline.tracked_intrusion` for why the short
            states are excluded -- their per-state ratio is noise, measured within
            1% of the threshold on ferritin_data2). ``None`` when ``fraction`` is,
            AND when no visited state is that long -- a recording with no state a
            tracking median could dwell inside has nothing for the criterion to
            read.
        worst_dwell_s: How long that worst state is, seconds. ``None`` with
            ``worst_fraction``.
        worst_t_s: Where that worst state starts, seconds into the recording, so the
            user can go look at it. ``None`` with ``worst_fraction``.
    """

    rough_s: float
    base_s: float
    fraction: float | None
    n_states: int | None
    cost: float | None
    chosen: bool = False
    #: LAST, and with defaults, for the same reason `BaselineProbe.state_dwell_s` is
    #: -- every construction in the tree is by keyword, and appending rather than
    #: inserting keeps a positional builder outside it working.
    worst_fraction: float | None = None
    worst_dwell_s: float | None = None
    worst_t_s: float | None = None

    @property
    def measured(self) -> bool:
        """True when this candidate produced a number the ranking could use."""
        return self.cost is not None

    @property
    def rides(self) -> bool:
        """True when this window's baseline rode INTO at least one state.

        ``worst_fraction > WORST_STATE_INTRUSION_MAX`` -- the baseline followed more
        than half of one state's own excursion, i.e. it crossed onto the state's side
        of it. This is the criterion that makes the -600 mV clog visible to the
        ranking; a candidate for which it is True loses to any candidate for which it
        is False, before the median is even compared. ``None`` reads as False here
        because an unmeasurable candidate already cannot win (`measured` is False);
        it is "not shown to ride", never "shown not to".
        """
        return (self.worst_fraction is not None
                and self.worst_fraction > WORST_STATE_INTRUSION_MAX)

    def to_dict(self) -> dict[str, Any]:
        """Flat and JSON-safe, for a log line or a provenance side-note."""
        return {
            "rough_s": float(self.rough_s),
            "base_s": float(self.base_s),
            "intrusion_fraction": (None if self.fraction is None
                                   else float(self.fraction)),
            "intrusion_n_states": (None if self.n_states is None
                                   else int(self.n_states)),
            "cost": None if self.cost is None else float(self.cost),
            "chosen": bool(self.chosen),
            "worst_fraction": (None if self.worst_fraction is None
                               else float(self.worst_fraction)),
            "worst_dwell_s": (None if self.worst_dwell_s is None
                              else float(self.worst_dwell_s)),
            "worst_t_s": (None if self.worst_t_s is None
                          else float(self.worst_t_s)),
            "rides": bool(self.rides),
        }


@dataclass(frozen=True)
class BaselineProposal:
    """A `BaselineConfig` the app is OFFERING, with the sentence that justifies it.

    ``param_source`` is ``'proposed'`` when the probe decided and ``'default'`` when
    it fell back, so a run stamps honestly which of the two it used -- and
    `reason` is written to be shown verbatim next to the numbers, because a
    proposal a user cannot interrogate is an auto-apply with an extra click.
    """

    config: BaselineConfig
    param_source: str
    reason: str
    probe: BaselineProbe | None = None
    #: LAST, and with a default, for the same reason `BaselineProbe.state_dwell_s` is
    #: -- every construction in the tree is by keyword, and appending rather than
    #: inserting keeps a positional builder outside it working. Empty means the
    #: windows were NOT ranked; `rank_windows` says in `reason` why not.
    ranking: tuple[BaselineCandidate, ...] = ()

    @property
    def confident(self) -> bool:
        """True when the numbers came from this recording rather than the fallback."""
        return self.param_source == "proposed"

    def params(self) -> dict[str, Any]:
        """The ``baseline_*`` params mapping `baseline.baseline_config_from_params`
        reads back, carrying the source so it is not silently re-labelled ``'user'``.

        Exactly `BASELINE_PARAM_KEYS`, and NOT the whole ``to_provenance`` block:
        that also carries ``baseline_fingerprint``, which is DERIVED from the
        parameters and from ``fs``. A derived value inside a params dict is a value
        free to disagree with the parameters beside it, and the estimator stamps the
        real one at run time anyway.
        """
        prov = self.config.to_provenance()
        out = {k: prov[k] for k in BASELINE_PARAM_KEYS if k in prov}
        out["baseline_param_source"] = self.param_source
        return out


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
def snap_to_grid(value: float) -> float:
    """Nearest 1-2-3-5 decade value to ``value``, in LOG space, clipped to the grid.

    Log-nearest rather than linear because these are timescales spanning three
    decades: linear-nearest would put 0.033 s at 0.05 (1.5x away) rather than 0.03
    (1.1x), and ferritin_data3's shipped ``rough_s`` is 0.03.

    The grid point is built as the DECIMAL LITERAL ``f"{m}e{decade}"``, not as
    ``m * 10.0 ** decade``, and that is load-bearing rather than fussy. The two
    differ at exactly one point on this grid -- ``3 * 10.0**-1`` is
    ``0.30000000000000004``, one ULP above ``0.3`` -- and a
    `~nanoscope.analysis.baseline.BaselineConfig` fingerprint is a hash of the
    fields' ``repr``. So the probe proposing "0.3 s" and the Baseline panel's
    4-decimal spin box handing back "0.3 s" produced two DIFFERENT
    ``baseline_fingerprint``s for the same window, and anything that stamps the raw
    proposal (a headless driver, a batch run) would make two identically-windowed
    recordings read NOT COMPARABLE in ``Analysis ▸ Plots`` and in the
    voltage-dependence window. Measured: it was the whole of that discrepancy on the
    owner's −500 mV file (``sha256:af17c2c64914`` from the proposal against
    ``sha256:b64a75ebdf29`` from the panel, same 1.0/0.30 s, same 20 columns).

    Nothing computed changes. The windows are used as ``int(win_s * fs / decim)``
    and through `~nanoscope.analysis.baseline.derive_decim`; over every grid point
    and every sample rate from 1 kHz to 1 MHz both spellings give the same ``decim``
    and the same ``w`` (0.3 s: decim 200, w 75 at 50 kHz). Only the identity changes,
    and it changes onto the value the GUI has always stamped.
    """
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        return GRID_MIN_S
    v = min(max(v, GRID_MIN_S), GRID_MAX_S)
    best, err = GRID_MIN_S, np.inf
    for decade in range(-4, 3):
        for m in GRID_MANTISSAS:
            g = float(f"{m:g}e{decade}")
            if not (GRID_MIN_S - 1e-12 <= g <= GRID_MAX_S + 1e-12):
                continue
            e = abs(np.log(g / v))
            if e < err:
                best, err = g, e
    return float(best)


def snap_up_to_grid(value: float) -> float:
    """SMALLEST grid point at or above ``value`` -- `snap_to_grid` for a FLOOR.

    A bound snapped to the log-NEAREST grid point is not a bound. Measured, and this
    is the whole reason this function exists rather than a call to `snap_to_grid`:
    `base_floor_s` on the -700 mV apo recording wants
    ``BASE_STATE_RATIO_MIN * 21.71 ms = 65.13 ms``; log-nearest puts that at
    **0.05 s**, where ``base_vs_state`` reads **2.303x against a floor of 3.0 and
    still condemns the run** (measured through the estimator: 0.2/0.05 s scores
    ``base_vs_state`` 2.3030861903313187 and ``baseline_intrusion`` 0.0272, health
    NOT ok). Snapping up puts it at 0.1 s, which reads 4.606x, intrusion 0.0043 and
    all eight checks green. A floor that lands below the thing it is flooring would
    have reproduced the exact defect this change removes, one grid point along.

    Same grid, same decimal-literal spelling and the same reason for it as
    `snap_to_grid` -- see there: ``3 * 10.0**-1`` is one ULP above ``0.3`` and a
    `~nanoscope.analysis.baseline.BaselineConfig` fingerprint is a hash of the
    fields' ``repr``, so the two spellings would make two identically-windowed
    recordings read NOT COMPARABLE.

    A value that IS a grid point stays put (the comparison carries a 1e-12 relative
    tolerance, so ``5 * 10.0**-2`` does not get pushed to 0.1). Above `GRID_MAX_S`
    there is no grid point to return and the clip is the honest answer: the caller is
    asking for a tracking window longer than the longest this app will propose.
    """
    v = float(value)
    if not np.isfinite(v) or v <= GRID_MIN_S:
        return GRID_MIN_S
    if v >= GRID_MAX_S:
        return GRID_MAX_S
    best = None
    for decade in range(-4, 3):
        for m in GRID_MANTISSAS:
            g = float(f"{m:g}e{decade}")
            if not (GRID_MIN_S - 1e-12 <= g <= GRID_MAX_S + 1e-12):
                continue
            if g >= v * (1.0 - 1e-12) and (best is None or g < best):
                best = g
    return float(best) if best is not None else GRID_MAX_S


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
def _closed_dwells(mask: np.ndarray, fs: float, gap_s: float) -> np.ndarray:
    """Run lengths (seconds) of ``mask`` after gaps of ``gap_s`` or less are closed.

    Morphological closing by a flat structuring element, done on the RUN BOUNDARIES:
    identical to ``binary_closing`` in the interior and 18x cheaper (16 ms against
    297 ms on 8.6 M samples). It differs only at the array ends, where
    ``scipy.ndimage``'s ``border_value=0`` erodes a run that touches the edge -- and
    a state clipped by the start or end of the recording has an unknown length
    anyway, so keeping it whole is the more honest of the two.
    """
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)
    gap = int(round(float(gap_s) * float(fs)))
    keep = np.concatenate(([True], (starts[1:] - ends[:-1]) > gap))
    s = starts[keep]
    e = np.concatenate((ends[:-1][keep[1:]], ends[-1:]))
    return (e - s) / float(fs)


def _unusable(fs, n, pol, note, rough_s, t0) -> BaselineProbe:
    return BaselineProbe(
        fs=float(fs), n=int(n), polarity=int(pol), sigma_pA=float("nan"),
        noise_pA=float("nan"), longest_state_s=0.0, n_states=0, mask_fraction=0.0,
        gating=False, gating_score=float("nan"), level_dwell_s=None,
        probe_rough_s=float(rough_s), usable=False, notes=(note,),
        elapsed_s=time.perf_counter() - t0,
    )


def probe_timescales(
    signal: np.ndarray,
    fs: float,
    *,
    polarity: int | None = None,
) -> BaselineProbe:
    """Measure this recording's timescales. Proposes nothing; see `propose_baseline`.

    Three measurements, in one pass over the trace:

    1. **A robust sigma**, `baseline.upper_mad_sigma` of the residual against the
       probe's own wide masked baseline.
    2. **A dwell distribution**, from that residual masked at `PROBE_MASK_SIGMA` with
       gaps up to `PROBE_CLOSE_GAP_S` closed -- without the closing the apo file's
       shallow 521 ms clog fragments into runs of <= 26 ms and is missed entirely.
    3. **A gating flag**, the IQR of `GATING_BLOCK_S` block medians over the
       within-block noise, plus (only when gating) the mean interval between
       conductance steps.

    Args:
        signal: The trace the estimator will run on -- WHOLE CHANNEL, conditioned
            exactly as detection will condition it. Not a zoom rectangle: the apo
            file's decisive 521 ms state is one state at t = 37 s in 60 s, and a
            probe that misses it under-sizes ``rough_s`` twentyfold.
        fs: Sample rate, Hz.
        polarity: ``{-1, +1}``, or ``None`` to measure it with
            `baseline.measured_polarity`. Everything here is one-sided, so a wrong
            polarity measures the wrong population.

    Raises:
        ValueError: non-finite input (checked FIRST, per 2.7 -- a NaN reaching the
            kernel makes its sigma NaN, which makes the threshold all-False, which
            silently disables the entire event mask), or a non-positive ``fs``.

    Returns:
        A `BaselineProbe`. Check ``.usable`` before believing the timescales;
        `propose_baseline` already does.
    """
    t0 = time.perf_counter()
    x = require_finite(signal, stage="probe a recording's baseline timescales")
    x = np.asarray(x, dtype=np.float64)
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive number of Hz, got {fs!r}")
    n = int(x.size)
    duration = n / fs

    rough_s = reference_rough_s(n, fs)
    pol = int(polarity) if polarity is not None else measured_polarity(x, fs)
    if rough_s < PROBE_MIN_ROUGH_S:
        return _unusable(
            fs, n, pol,
            f"the recording is {duration:.3g} s long, and the probe needs at least "
            f"{PROBE_WINDOWS_MIN * PROBE_MIN_ROUGH_S:g} s to see over a state worth "
            f"sizing a window against",
            rough_s, t0)

    # -- 1/2. the probe baseline, its sigma, and the dwell distribution -----
    # Through THE DOOR, at the estimator's own defaults but wide and coarse, so the
    # states the probe reports are the states the estimator would mask. Never
    # cached (a probe config is not a measurement anyone stamps) and never
    # health-checked (health re-runs pass 1; the probe is a budget, not a report).
    from .baseline import characterisation_baseline  # local: avoids an import cycle

    check_cancelled()
    probe_cfg = DEFAULT_BASELINE.replace(
        rough_s=rough_s, base_s=rough_s / ROUGH_BASE_RATIO, decim=None)
    res = characterisation_baseline(
        x, fs, config=probe_cfg, polarity=pol, health=False, cache=False)
    if res.config.method != probe_cfg.method:      # a length fallback stood in
        return _unusable(fs, n, pol,
                         "the recording is too short for the probe's own estimator",
                         rough_s, t0)
    check_cancelled()

    resid = (x - res.baseline) if pol == -1 else (res.baseline - x)
    stride = _SIGMA_STRIDE if n >= _SIGMA_STRIDE_MIN_N else 1
    sigma = upper_mad_sigma(resid[::stride])
    if not (np.isfinite(sigma) and sigma > 0):
        return _unusable(fs, n, pol,
                         "the trace has no measurable noise (it is constant, or one "
                         "level fills more than half of it)", rough_s, t0)
    mask = resid < -PROBE_MASK_SIGMA * sigma
    check_cancelled()
    dwells = _closed_dwells(mask, fs, PROBE_CLOSE_GAP_S)
    longest = float(np.percentile(dwells, LONGEST_STATE_Q)) if dwells.size else 0.0

    # -- 2b. the MEDIAN state, measured the way `base_vs_state` measures it -----
    # Not `np.median(dwells)`, which is the same array `longest` came from and is a
    # DIFFERENT statistic from the check's: this mask is one-sided, undilated, and
    # gap-closed, and its median reads 17.9 ms on the -700 mV recording against the
    # check's 21.7 -- 0.28 ms against 2.76 on ferritin_data2, a factor of ten. A
    # floor built on a proxy is a floor that can land under the bound it is
    # enforcing, which is the defect. `baseline.reference_state_dwell_s` is the
    # check's own measurement, verified bit-identical to its divisor on all six
    # recordings in evidence. Cost: one more decimated median at the reference
    # window -- 0.088 s (data1) to 0.525 s (data2); see that function.
    check_cancelled()
    state_dwell = reference_state_dwell_s(x if pol == -1 else -x, fs)

    # -- 3. gating -----------------------------------------------------------
    check_cancelled()
    score, noise, n_blocks = gating_spread(x, fs)
    if n_blocks < GATING_BLOCKS_MIN:
        return _unusable(fs, n, pol,
                         "too few whole blocks to tell a stepping pore from a "
                         "steady one", rough_s, t0)
    if not np.isfinite(score):
        score = float("inf")
    gating = bool(score > GATING_THRESHOLD)

    level: float | None = None
    notes: tuple[str, ...] = ()
    if gating:
        check_cancelled()
        bm2, nb2 = block_medians(x, fs, LEVEL_BLOCK_S)
        steps = np.diff(bm2)
        step_sigma = 1.4826 * np.median(np.abs(steps - np.median(steps)))
        idx = (np.flatnonzero(np.abs(steps) > LEVEL_STEP_SIGMA * step_sigma)
               if step_sigma > 0 else np.empty(0, dtype=int))
        if idx.size >= 3:
            level = float(np.mean(np.diff(idx)) * nb2 / fs)
        else:
            notes = ("the pore steps between levels, but too few steps were resolved "
                     "to measure how fast",)

    return BaselineProbe(
        fs=fs, n=n, polarity=pol, sigma_pA=float(sigma), noise_pA=float(noise),
        longest_state_s=longest, n_states=int(dwells.size),
        state_dwell_s=float(state_dwell),
        mask_fraction=float(mask.mean()), gating=gating, gating_score=score,
        level_dwell_s=level, probe_rough_s=float(rough_s),
        # A dwell distribution WAS measured even when the level analysis failed, and
        # it is a better answer than the global default; `propose_baseline` takes the
        # dwell branch and says in the reason line that it is doing so.
        usable=True, notes=notes, elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
def _fallback(probe: BaselineProbe | None, why: str) -> BaselineProposal:
    """`baseline.DEFAULT_BASELINE` ITSELF -- by identity, not by re-spelling it.

    An undecidable probe must land on the app's one baseline literal, so that a
    fallback proposal and the app's default can never drift apart, and so that the
    test for this reads ``proposal.config is DEFAULT_BASELINE``.
    """
    return BaselineProposal(
        config=DEFAULT_BASELINE,
        param_source="default",
        reason=(f"Could not measure this recording ({why}). Falling back to the "
                f"default {DEFAULT_BASELINE.rough_s:g} s / "
                f"{DEFAULT_BASELINE.base_s:g} s -- a defensible floor, not an answer "
                f"for this file."),
        probe=probe,
    )


def base_floor_s(probe: BaselineProbe | None) -> float | None:
    """The shortest ``base_s`` ``base_vs_state`` will certify here -- or ``None``.

    THE HALF OF THE CONTRACT THE PROPOSER WAS MISSING. `baseline.baseline_health`
    condemns a run whose ``base_s`` is below `baseline.BASE_STATE_RATIO_MIN` times the
    median state dwell; this returns that bound, snapped UP onto the proposal grid
    (`snap_up_to_grid` -- log-nearest is not a floor, and on the -700 mV recording it
    lands on a window the check still condemns).

    ONE function because two callers need the same answer and a second spelling of it
    would be free to disagree: `propose_baseline` applies it, and `proposal_summary`
    tells the user it was applied.

    ``None`` in the three cases where the bound does not exist, each of which mirrors
    something the check itself does:

    * **no probe, or an unusable one** -- the proposal is `baseline.DEFAULT_BASELINE`
      by identity and nothing is being sized against anything;
    * **a credibly gating pore** (`BaselineProbe.gating_credible` with a level dwell,
      i.e. exactly when `propose_baseline` takes the LEVEL branch) -- ``base_vs_state``
      exempts the same case with ``short = (not gating) and ...``, spelling ``gating``
      as the same "scores above `GATING_THRESHOLD` and masks no more than
      `baseline.MASK_FRACTION_CEILING`" predicate. On a stepping pore the long states
      ARE baseline levels and ``base_s`` is SUPPOSED to be shorter than they are.
      **Measured, and this is why the exemption is load-bearing rather than tidy**:
      ferritin_data3's reference median state is 3.34 ms, so a floor applied there
      would be ``snap_up(3 x 3.34 ms)`` = 0.02 s and would move that recording off its
      known-good 0.03 / 0.01 s -- the shipped configuration this app's corpus contract
      reproduces -- to 0.1 / 0.02;
    * **an unmeasurable state dwell** (``nan``: a recording too short for a reference
      window, or a wide pass that found no run longer than one sample once dilated) --
      the check reports COULD NOT BE MEASURED there rather than a ratio, and a floor
      invented where the bound cannot be measured is a number with nothing behind it.

    TWO THINGS THIS BOUND IS NOT, both measured:

    * It is **not a guarantee once the proposed ``rough_s`` reaches
      `baseline.reference_rough_s`** -- see the module docstring's "WHERE THAT IS A
      GUARANTEE" for the 3 s recording where the check switches to the configured
      mask and the floor lands 2.75x rather than above 3.0.
    * On a trace with NO events it measures the dilated NOISE runs, exactly as
      `baseline.state_dwell_s` does for the check -- 2.73 ms on 60 s of pure Gaussian
      noise, giving a 0.01 s floor built on nothing. The two agree, which is the
      contract here, and what says the trace is empty is ``mask_fraction_floor``,
      which fires on that case. (It is still an improvement: without the floor that
      trace is proposed 0.002 / 0.002 s -- a rough window EQUAL to its base window.)
    """
    if probe is None or not probe.usable:
        return None
    if probe.gating_credible and probe.level_dwell_s:
        return None
    d = probe.state_dwell_s
    if d is None:
        return None
    d = float(d)
    if not np.isfinite(d) or d <= 0:
        return None
    return snap_up_to_grid(BASE_STATE_RATIO_MIN * d)


def propose_baseline(probe: BaselineProbe | None) -> BaselineProposal:
    """Turn a `BaselineProbe` into the parameters the app OFFERS, and the reason why.

    The rule is in the module docstring. It takes the LEVEL branch only when the
    gating flag is credible (`BaselineProbe.gating_credible`) -- a pore masking more
    than `baseline.MASK_FRACTION_CEILING` of its trace is one whose long states the
    same statistic would score as levels either way, and the dwell branch is the
    measurement that survives being wrong about which: on the six misfires measured it
    proposes windows that recover the true depth to 0.4%, where the level branch
    recovers 1-40% of it. The reason line says which branch was declined and why.

    Two things it deliberately does not do:

    * It never proposes ``mask_sigma``, ``dilate_ms``, ``window_s`` or
      ``despike_size``. Those stay at `baseline.DEFAULT_BASELINE`'s values --
      structurally, via ``replace``, not by convention.
    * It never optimises anything. The reference's QC triple is monotone in window
      length (a degenerate 0.01/0.002/1 set beats the shipped parameters on all
      three scalars for every corpus file), so a search over it selects the shortest
      window there is -- the exact failure this change removes. The rule is
      arithmetic on two measured timescales, and a human confirms it.
    """
    if probe is None:
        return _fallback(None, "no probe was run")
    if not probe.usable:
        return _fallback(probe, probe.notes[0] if probe.notes else "the probe failed")

    if probe.gating_credible and probe.level_dwell_s:
        base = snap_to_grid(probe.level_dwell_s / LEVEL_BASE_DIVISOR)
        rough = snap_to_grid(ROUGH_MULTIPLE * base)
        reason = (
            f"Gating pore: the open-pore level itself steps "
            f"({probe.gating_score:.1f}x the noise, threshold "
            f"{GATING_THRESHOLD:g}), so its long states are conductance LEVELS to "
            f"follow, not events to mask. Mean level dwell "
            f"{probe.level_dwell_s * 1e3:.0f} ms -> base {base * 1e3:g} ms (short "
            f"enough to follow a step); rough {rough * 1e3:g} ms."
        )
    elif probe.longest_state_s > 0:
        rough = snap_to_grid(ROUGH_MULTIPLE * probe.longest_state_s)
        base = snap_to_grid(rough / ROUGH_BASE_RATIO)

        # THE FLOOR. `rough_s` is sized against the LONGEST state (p99.9) it must sail
        # over; `base_s` fell out of it by ROUGH_BASE_RATIO and was never checked
        # against the state a tracking median actually meets -- the MEDIAN, which is
        # what `base_vs_state` divides by and condemns below BASE_STATE_RATIO_MIN.
        # Applied only where it BINDS, and `rough_s` carried up with it only then:
        # recomputing rough_s unconditionally from the floored base_s would have moved
        # the -500 mV proposal from 1.0 s to 2.0 s (snap_up(4 x 0.3) = 2.0) on a
        # recording whose eight checks are all green, which is the "fix" moving a
        # correct answer. `max` rather than assignment so the ROUGH_MULTIPLE x
        # longest_state property the line above established can only ever be exceeded.
        floor = base_floor_s(probe)
        floored = floor is not None and floor > base
        if floored:
            was_base, was_rough = base, rough
            base = float(floor)
            # `snap_to_grid`, NOT `snap_up_to_grid`. The floor is a bound and must
            # snap up; ``rough_s`` is a RATIO to it and must stay inside the 2.5-5.5x
            # band the spec requires (`tests/analysis/test_baseline_probe.py`
            # asserts it). Measured: snapping up put a 0.03 s base at rough 0.2
            # (0.12 -> 0.2), a ratio of 6.67 that fails that assertion, where
            # log-nearest puts it at 0.1 and 3.33x. Over every grid point the
            # log-nearest carry-up lands in 3.33-5.0x.
            rough = max(rough, snap_to_grid(ROUGH_BASE_RATIO * base))

        gate = ("no gating detected" if not probe.gating else
                "the block medians spread like a stepping pore, but with this much of "
                "the trace masked, events this long move a block median the same way, "
                "so the LEVEL branch is declined and these are read as events"
                if not probe.gating_credible else
                "pore steps, but its level dwell could not be measured")
        # "THE RULE'S", not "->". These two numbers are the RULE's answer, and on the
        # -700 mV recording they are not the app's: `rank_windows` supersedes them one
        # paragraph later, and this sentence used to read as the offer -- three
        # declarations of 0.5 s / 100 ms in front of a proposal of 1 s / 300 ms. The
        # supersession is announced by `_ranking_reason`, which is the only function
        # that knows whether there was one; what belongs HERE is a label that stays
        # true either way, including for the callers of `propose_baseline` that never
        # rank at all.
        reason = (
            f"Longest state {probe.longest_state_s * 1e3:.0f} ms "
            f"({probe.n_states} states, {probe.mask_fraction * 100:.1f}% of the "
            f"trace) -> the rule's rough window {rough:g} s, so no single state fills "
            f"more than ~{100 / ROUGH_MULTIPLE:.0f}% of the window; {gate} "
            f"({probe.gating_score:.1f}); shortest level dwell n/a -> the rule's base "
            f"window {base * 1e3:g} ms."
        )
        if floored:
            reason += (
                f" The {ROUGH_BASE_RATIO:g}:1 rule alone would have proposed "
                f"{was_rough:g} s / {was_base * 1e3:g} ms, and Baseline health "
                f"condemns that: the MEDIAN state here is "
                f"{float(probe.state_dwell_s) * 1e3:.1f} ms, so a "
                f"{was_base * 1e3:g} ms tracking window is only "
                f"{was_base / float(probe.state_dwell_s):.1f}x the states it exists "
                f"to ignore and rides into them, and depths measured from it read "
                f"shallow. base_s is floored at {BASE_STATE_RATIO_MIN:g}x that "
                f"median ({BASE_STATE_RATIO_MIN * float(probe.state_dwell_s):.3g} s) "
                f"-> {base * 1e3:g} ms, the shortest BASE window on the grid that "
                f"clears it ({base / float(probe.state_dwell_s):.1f}x), with the "
                f"rule's rough window carried up to {rough:g} s. That is a floor the "
                f"rule cannot go below, not the window the app has settled on."
            )
    else:
        return _fallback(probe, "no state crossed the mask threshold anywhere")

    # ONLY these three fields. See the module docstring: a probe that could set
    # `mask_sigma` would be a second baseline default, and the seam's AST guards
    # exist to make a second baseline default impossible to write.
    cfg = DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)
    return BaselineProposal(config=cfg, param_source="proposed", reason=reason,
                            probe=probe)


# ---------------------------------------------------------------------------
# the ranking -- choose by MEASUREMENT, not by legality
# ---------------------------------------------------------------------------
def _grid_points() -> tuple[float, ...]:
    """The 1-2-3-5 decade grid, ascending, spelled as `snap_to_grid` spells it.

    The DECIMAL LITERAL, not ``m * 10.0 ** decade``. That is load-bearing for exactly
    the reason `snap_to_grid`'s docstring gives -- ``3 * 10.0**-1`` is one ULP above
    ``0.3``, a `~nanoscope.analysis.baseline.BaselineConfig` fingerprint is a hash of
    the fields' ``repr``, and a candidate window built the other way would give the
    same window two identities and make two identically-windowed recordings read NOT
    COMPARABLE. The ladder's ``rough_s`` comes from HERE and its ``base_s`` from
    `snap_to_grid`; both spellings have to be the one the panel's spin box hands back.
    """
    out = []
    for decade in range(-4, 3):
        for m in GRID_MANTISSAS:
            g = float(f"{m:g}e{decade}")
            if GRID_MIN_S - 1e-12 <= g <= GRID_MAX_S + 1e-12:
                out.append(g)
    return tuple(sorted(out))


def candidate_windows(
    proposal: BaselineProposal, n: int, fs: float
) -> tuple[tuple[float, float], ...]:
    """The ladder `rank_windows` walks: ``((rough_s, base_s), ...)``, shortest first.

    IT CONTAINS THE PROPOSAL ITSELF, verbatim -- so the ranking always includes the
    answer the rule would have given, and "ranked" can never mean "replaced without
    comparison". Every other rung is a grid ``rough_s`` with the rule's own ``base_s``
    beside it (``snap_to_grid(rough_s / ROUGH_BASE_RATIO)``, lifted by `base_floor_s`
    wherever that binds ABOVE the proposal), which is exactly what the proposer would
    offer if the recording's longest state had been that much longer or shorter. The
    ladder is therefore made of windows the rule can already produce, not of new ones.

    **IT IS TWO-SIDED, AND IT WAS NOT.** The old ladder began at the proposal and
    walked strictly upward (``if rough <= first[0]: continue``), so a SHORTER window
    was never estimated -- while `rank_windows`' documented objective is "lowest cost,
    ties broken by the shortest window". On the -600 mV apo recording those two things
    disagree outright, measured over the one shared reference mask:

    ==============  =====================  ===================================
    -600 mV window  ``baseline_intrusion``  in the old ladder?
    ==============  =====================  ===================================
    0.3 / 100 ms    0.002888               no -- shorter than the proposal
    0.5 / 100 ms    0.003155               no -- shorter than the proposal
    **1 / 300 ms**  **0.0**                **no -- shorter, and it WINS**
    2 / 500 ms      0.0  (the proposal)    yes, and it was chosen
    3 / 1 s         0.0                    yes (never reached: stop-at-zero)
    ==============  =====================  ===================================

    1 s / 300 ms ties on the MEDIAN cost, is shorter, and wins the tie-break -- and
    the screen used to say "Nothing can beat a baseline that follows none of the
    states, so no wider window was estimated", whose second clause was true and whose
    first was false under that objective, because the thing that beats it is not
    wider. The ladder therefore holds rungs on both sides. **That tie is itself a
    tie on a blind statistic**: 1 s / 300 ms rides into the recording's 521 ms clog
    (worst state 0.556 against `WORST_STATE_INTRUSION_MAX`'s 0.5) and the worst-state
    criterion in `rank_windows` is what now separates the two rungs the median
    cannot. The ladder's job is unchanged either way -- hold the rungs so the search
    can measure them.

    THE FLOOR IS THE BOUND ON THE WAY DOWN
    ======================================
    Downward rungs stop where `base_floor_s` BINDS -- i.e. a rung is admitted below
    the proposal only while ``snap_to_grid(rough_s / ROUGH_BASE_RATIO)`` is already at
    or above the floor, never by lifting ``base_s`` onto it. Two reasons, and they are
    the same reason twice:

    * a lifted short rung is **not a window the rule can produce**. `propose_baseline`
      carries ``rough_s`` UP whenever the floor lifts ``base_s``
      (``rough = max(rough, snap_to_grid(ROUGH_BASE_RATIO * base))``), so the rule can
      never offer 0.2 s / 100 ms; it would offer 0.5 s / 100 ms. Ranking a window the
      rule cannot reach would make the ladder a search over new windows rather than
      over the rule's own;
    * those rungs are the ones ``base_vs_state`` condemns, at ratios outside the
      2.5-5.5x band `propose_baseline` maintains -- 0.1 s / 100 ms is a rough window
      EQUAL to its base window.

    Measured, that bound lands one rung below the shipped answer on all three apo
    recordings (floor 100 ms -> shortest admissible rung 0.3 s / 100 ms) and three
    rungs below on ferritin_data2 (floor 10 ms -> 0.03 s / 10 ms). The rungs it
    excludes are also the ones that read WORST: -600 mV reads 0.003288 at 0.1 s /
    100 ms and 0.003108 at 0.2 s / 100 ms, against 0.0 at the winner.

    THE CEILING, AND WHY IT IS `baseline.reference_rough_s` AND NOT THE RECORDING
    ============================================================================
    An unbounded search up the grid ends at the whole recording, and the reason it
    must not is not taste -- it is that above `baseline.reference_rough_s` the
    comparison stops being a comparison. Every candidate's intrusion is measured over
    the ONE wide-pass mask `baseline.reference_states` returns, and that function
    returns ``None`` the moment the configured ``rough_s`` reaches the reference
    window. Past the ceiling, `baseline.baseline_health` switches to the CONFIGURED
    pass -- a different mask, a different state set, a different number -- so a
    candidate measured there could not be put in the same column as one measured
    below it. It is also precisely where `base_floor_s`'s guarantee ends (see the
    module docstring's "WHERE THAT IS A GUARANTEE"), so the ranking and the floor stop
    at the same line rather than at two.

    On the owner's recordings the ceiling is `baseline.REFERENCE_ROUGH_S` = 5 s, so a
    0.5 s proposal has ``1 / 2 / 3`` above it and ``0.3`` below it (five rungs
    including itself).

    **`RANK_MAX_CANDIDATES` NO LONGER TRUNCATES THIS TUPLE.** It is a bound on how
    many windows are ESTIMATED, and truncating the ladder from the top is a different
    thing: on a two-sided ladder walked from the short end, a cap applied here would
    delete the winner rather than the windows the search was never going to reach.
    `rank_windows` spends the budget; this function reports what there is to spend it
    on. Building the whole tuple is free -- it is arithmetic on the grid, and the
    ladder never exceeds a couple of dozen rungs between the two bounds.

    Returns a one-element tuple (the proposal alone) when nothing fits between the
    floor and the ceiling; `rank_windows` reads that as "there is nothing to rank
    against".
    """
    cfg = proposal.config
    first = (float(cfg.rough_s), float(cfg.base_s))
    # Keyed by ``rough_s`` so the PROPOSAL always wins its own rung: a floored
    # proposal's ``base_s`` need not equal ``snap_to_grid(rough_s / ROUGH_BASE_RATIO)``
    # (base 200 ms carries rough up to 1 s, where the ratio would say 300 ms), and the
    # rung the user is being offered has to be the one the ladder measures.
    out: dict[float, tuple[float, float]] = {first[0]: first}
    ceiling = reference_rough_s(int(n), float(fs))
    if not np.isfinite(ceiling) or ceiling < REFERENCE_MIN_ROUGH_S:
        return (first,)
    floor = base_floor_s(proposal.probe)
    for rough in _grid_points():
        if rough >= ceiling or rough in out:
            continue
        derived = rough / ROUGH_BASE_RATIO
        base = snap_to_grid(derived)
        if rough < first[0]:
            # DOWNWARD: only windows the rule can produce UNAIDED. See "THE FLOOR IS
            # THE BOUND ON THE WAY DOWN" above.
            if derived < GRID_MIN_S:
                # ``base_s`` would be the grid's own lower CLAMP rather than the
                # rule's ratio -- 0.002 s / 0.002 s is a rough window equal to its
                # base window, which the rule cannot reach and the 2.5-5.5x band
                # `propose_baseline` maintains forbids.
                continue
            if floor is not None and floor > base:
                continue                  # the floor binds: the rule carries rough up
        elif floor is not None and floor > base:
            base = float(floor)
        if base > rough:                      # BaselineConfig refuses it, rightly
            continue
        out[rough] = (rough, base)
    return tuple(out[k] for k in sorted(out))


def intrusion_cost(fraction: float | None) -> float | None:
    """What a candidate is RANKED on: ``max(fraction, 0)``, or ``None``.

    ONE-SIDED, BECAUSE THE CHECK IT IS A SEARCH OVER IS ONE-SIDED.
    `baseline.baseline_health`'s ``baseline_intrusion`` row condemns on
    ``fraction > BASELINE_INTRUSION_MAX`` and on nothing else: a baseline that sits
    slightly ABOVE the local open pore reads negative and the check does not
    distinguish that from zero. Ranking on the signed value would prefer a window
    reading -0.0004 over one reading exactly 0.0 on the strength of a sign the check
    ignores; ranking on ``abs`` would prefer the opposite, on the same non-evidence.
    Clipping at zero says what is actually being asked -- **how much of each state's
    own depth did this window's baseline eat** -- and says it in the check's own
    terms. Measured, the whole corpus reads between -0.0004 and +0.0053, so nothing
    in evidence is more than half a percent either side.

    IT IS HALF OF THE OBJECTIVE, NOT ALL OF IT. This cost is a MEDIAN over the
    states, and a median over hundreds is structurally blind to ONE catastrophic
    state -- the -600 mV recording reads exactly 0.0 here at 1.0 / 0.3 s while that
    window's baseline rides into the 521 ms clog. `rank_windows` therefore compares
    `BaselineCandidate.rides` (the worst single state against
    `WORST_STATE_INTRUSION_MAX`) BEFORE this cost. The blindness lives in the median
    and the guard lives beside it rather than inside it, because folding the worst
    state into this number would make it stop being the health block's own statistic
    -- the property the whole ranking rests on.

    TWO CONSEQUENCES, both deliberate and both worth knowing:

    * The objective is BOUNDED BELOW BY ZERO, which is what makes `rank_windows`'
      stop rule exact rather than a heuristic: a candidate reading zero AND riding
      into nothing cannot be beaten, and the tie-break is the shortest window, so the
      first such rung on an ascending ladder IS the answer and nothing above it needs
      estimating.
    * A strongly NEGATIVE intrusion is not penalised. That is a real limit: a window
      reading -0.02 biases depths ~2% DEEP and would rank equal with a clean one.
      Nothing on this corpus comes within a factor of forty of that, and the check
      would not object either -- but if that regime is ever met, this is the function
      that has to learn about it, not the tie-break.

    ``None`` in, ``None`` out, and ``None`` for a non-finite fraction: a candidate
    with no measurable intrusion is NOT a candidate with no intrusion.
    """
    if fraction is None:
        return None
    f = float(fraction)
    if not np.isfinite(f):
        return None
    # ``max(-0.0, 0.0)`` is ``-0.0`` in Python (0.0 > -0.0 is False, so the first
    # argument stands), and a negative zero would travel into `BaselineCandidate`,
    # into ``to_dict`` and into any provenance built from it. It compares equal to 0.0
    # everywhere it is used and is still noise in a record someone reads.
    return 0.0 if f <= 0.0 else f


def _rank_declined(proposal: BaselineProposal, n: int, fs: float) -> str | None:
    """Why these windows must not be ranked, in the user's words -- or ``None``.

    ``""`` (empty, falsy-but-not-None) means DECLINE SILENTLY: the fallback proposal
    already says at length that nothing could be measured here, and appending "and
    they were not ranked either" to it would add a second negation with no new fact
    in it -- and the sentence it would append ("these are the windows the rule derives
    from this recording's timescales") is not even true of the global default.
    """
    p = proposal.probe
    if p is None or not p.usable or not proposal.confident:
        return ""
    if p.gating_credible and p.level_dwell_s:
        return ("the open-pore level itself steps, and the intrusion check declines "
                "to judge a stepping pore at all -- on one, the long states ARE "
                "baseline levels, so the share of them the baseline follows is "
                "supposed to be large and ranking on it would rank on nothing")
    if proposal.config.method != "masked_median":
        return "this estimator has no wide pass to measure the states against"
    ceiling = reference_rough_s(int(n), float(fs))
    if not np.isfinite(ceiling) or ceiling < REFERENCE_MIN_ROUGH_S:
        return (f"{int(n) / float(fs):.3g} s of signal cannot hold a reference window "
                "wide enough to see over the states")
    if float(proposal.config.rough_s) >= ceiling:
        # NOT "two candidates could not be measured against the same states" -- the
        # ladder is two-sided now, and two candidates BELOW the ceiling could be. What
        # cannot be measured against those states is THE PROPOSAL ITSELF, and a
        # ranking that never estimates the window it is replacing is exactly the
        # "replaced without comparison" this whole ladder exists to make impossible.
        return (f"the proposed {float(proposal.config.rough_s):g} s rough window is "
                f"already at or above the {ceiling:g} s reference window, so the "
                "window being replaced could not be measured against the same states "
                "as the ones that would replace it")
    return None


def rank_windows(
    signal: np.ndarray, fs: float, proposal: BaselineProposal
) -> BaselineProposal:
    """CHOOSE the window by measured intrusion instead of taking the shortest legal one.

    THE DEFECT THIS CLOSES
    ======================
    `base_floor_s` is a LOWER BOUND and `propose_baseline` never looks above it: it
    takes the shortest ``base_s`` on the grid that ``base_vs_state`` tolerates and
    stops. That is choosing by LEGALITY. The app already computes the statistic that
    ranks these windows -- ``baseline_intrusion``, the direct measurement of how far
    the tracked baseline sits inside the states -- and running it over the ladder
    costs one estimate per rung. Measured on the -700 mV apo recording, where the
    floor's answer is 0.5 s / 100 ms:

    ==============  ===================  ================================
    window          ``baseline_intrusion``  stamped open pore vs -48.218 pA
    ==============  ===================  ================================
    0.1 / 0.03      0.0416 (condemned)   -51.661   7.14% out
    0.5 / 0.1       0.00428              -48.857   1.33% out
    **1.0 / 0.3**   **0.0**              **-48.218**  **0.0005% out**
    2.0 / 0.5       0.0                  -48.240   0.045% out
    ==============  ===================  ================================

    The floor stops at row 2. This function measures rows 2 and 3 and takes row 3.

    THE SECOND DEFECT, WHICH IS THE SAME DEFECT LOOKING THE OTHER WAY
    =================================================================
    The search only ever looked UP, and then the screen claimed an optimality it had
    not tested: "Nothing can beat a baseline that follows none of the states, so no
    wider window was estimated." The second clause is true. The first is false under
    this function's own objective, because the thing that beats a clean window is not
    a wider one -- it is a SHORTER one that is equally clean and wins the tie-break.
    On the -600 mV apo recording that window exists (1 s / 300 ms against the shipped
    2 s / 500 ms; see `candidate_windows` for the table), and the app never estimated
    it. So the ladder is two-sided and the walk below is two-phase.

    THE THIRD DEFECT: THE OBJECTIVE ITSELF WAS BLIND TO ONE CATASTROPHIC STATE
    ==========================================================================
    The two-sided ladder then chose 1 s / 300 ms on -600 mV, and that window is
    WRONG in a way the median cannot show: its tracked baseline rides into the 521 ms
    partial clog at t = 37.13 s (-50 to -51 pA through a -57.7 pA state against a
    -34.5 pA open pore), the clog stops being detected at all -- at 2 s / 500 ms it
    is ONE 519.5 ms event, depth 28.9 pA -- and ~30 shallow events are invented on
    the collapsed local baseline, with all eight health checks green either way.
    ``baseline_intrusion`` read exactly 0.0 there, because it is a MEDIAN over 239
    states and one state ridden fully into does not move it. So the objective now has
    two criteria: the median, and the WORST single state among those longer than the
    candidate's own tracking window (`tracked_intrusion`'s ``worst_fraction`` against
    `WORST_STATE_INTRUSION_MAX`, which carries the measured separation -- ridden
    windows read >= 0.556, windows that stay out read <= 0.061 or have no state that
    long at all). A candidate that rides into a state loses to any candidate that
    stays out, before the medians are compared. The same criterion found the same
    failure on a SECOND recording: ferritin_data2's 1.0 / 0.3 offer rode a real
    395 ms deep state at t = 99.3 s to 78%, at a median of exactly 0.0.

    THE RULE
    ========
    Walk `candidate_windows` UP from the proposal, taking `baseline.config_intrusion`
    for each rung (the health block's own number, over ONE shared reference mask) and
    scoring it with `intrusion_cost`; **stop at the first rung that is CLEAN on both
    criteria** -- median cost zero AND no single state ridden into. Then, IF the
    proposal itself is clean that way, walk DOWN from it as well, one rung at a time,
    **stopping at the first rung that is not**. The winner: a rung that rides into a
    state loses to one that stays out; then lowest cost; ties broken by the SHORTEST
    window -- over everything estimated.

    * **Phase 1's stop rule is still exact.** A clean rung's key is (no ride, zero
      cost, its length): the first two terms are at their joint minimum and every
      rung above it is longer, so nothing above a clean rung can win, whatever it
      reads.
    * **Phase 2's stop rule is NOT exact, and inheriting phase 1's argument for it
      would have been wrong.** Descending, the objective is not bounded from the side
      the walk is moving towards: a rung reading 0.003 says nothing, in arithmetic,
      about the rung below it. What it rests on is a measured, physical claim --
      **intrusion grows, and riding deepens, as the tracking window shortens**,
      because a shorter rolling median rides further into the states it exists to
      ignore. Measured over the whole two-sided ladder on all six recordings in
      evidence (0.1 s to 3 s, 41 estimates), the cost is > 0 at EVERY rung below the
      first clean one on every one of them and no rung ever returns to zero on the
      way down; and on every recording and fixture with a rung that RIDES (-600 mV,
      -500 mV at 0.3 s, the three clogged synthetics), every rung below it rides
      deeper (0.556 -> 0.987 on -600 mV; 0.966 -> 1.000 on the CLOGGED fixture). It
      is a bound with evidence, not a proof, and it is stated as one:
      `_ranking_reason` tells the user how many shorter windows were not estimated
      and why, rather than claiming they were beaten.
    * **Phase 2 runs only when a tie is possible at all**, which is what keeps it
      cheap: it is gated on the PROPOSAL being clean, and when the proposal is clean
      phase 1 stops after one estimate. The two phases therefore never both
      spend: measured on the corpus the search estimates 2 (-500 mV), 2 (-600 mV --
      the descent's first rung, 1.0 / 0.3, rides and stops it), 2 (-700 mV),
      2 (ferritin_data1), 3 (ferritin_data2 -- phase 1 walks THROUGH the riding
      1.0 / 0.3 to the clean 2.0 / 0.5) and 0 (ferritin_data3, exempt) windows,
      against 1, 1, 2, 2, 2 and 0 before the ladder was two-sided.
    * **The tie-break is SHORTEST, and it is not "the first one tried".** 1.0 / 0.3
      and 2.0 / 0.5 both read exactly 0.0 on the -700 mV file and something has to
      separate them. A longer window is not free: it tracks slower drift less well,
      which is what ``length_tracking`` and ``rough_sigma_ratio`` exist to measure,
      and the corpus shows the cost directly -- on the -500 mV recording, the one that
      is genuinely non-stationary, ``overshoot`` goes 0.000227 at 1.0 / 0.3 to
      0.002446 at 2.0 / 0.5, a factor of ten, while the intrusion does not improve at
      all. So among windows that measure the same, the app takes the least smoothing
      it can get away with.
    * **The ceiling** is `baseline.reference_rough_s`; see `candidate_windows`.

    WHAT IT MOVES ON THE CORPUS, MEASURED
    =====================================
    The one-sided ladder moved three of the six (-700 mV, ferritin_data1 and data2,
    all three from 0.5 / 0.1 to 1.0 / 0.3, on intrusions of 0.00428 / 0.00509 /
    0.00531 -> 0.0). The two-sided ladder briefly moved -600 mV to 1.0 / 0.3 on the
    median tie-break alone; the worst-state criterion moves it BACK, because that
    window rides into the recording's 521 ms clog and the tie it claimed was a tie on
    a statistic blind to it:

    ================  ==============  ================  =========================
    recording         ONE-SIDED       NOW               why
    ================  ==============  ================  =========================
    -500 mV apo       1.0 / 0.3       **unchanged**     0.5 / 0.1 reads 0.006467
    -600 mV apo       2.0 / 0.5       **unchanged**     1.0 / 0.3 RIDES (0.556)
    -700 mV apo       1.0 / 0.3       **unchanged**     no descent (proposal not clean)
    ferritin_data1    1.0 / 0.3       **unchanged**     no descent (proposal not clean)
    ferritin_data2    1.0 / 0.3       **2.0 / 0.5**     1.0 / 0.3 RIDES a real 395 ms
                                                        deep state (0.782, t = 99.3 s)
    ferritin_data3    0.03 / 0.01     **unchanged**     (gating: not ranked)
    ================  ==============  ================  =========================

    **-500 mV is not held by slack.** Its proposal reads -0.00040 (a clean zero to
    `intrusion_cost`) and has no state longer than its 300 ms tracking window at all
    (``worst_fraction`` ``None``), so phase 2 DOES run there and does estimate
    0.5 s / 100 ms; that window reads 0.006467 and loses. The window below it,
    0.3 s / 100 ms, reads 0.006349 -- and rides (0.973, on a 239 ms state longer
    than ITS 100 ms window) -- and is never reached. The recording stays where it is
    because it was measured, not because it was skipped.

    **-600 mV is not held by slack either.** Its descent DOES run (the 2 s / 500 ms
    proposal is clean: median 0.0, worst qualifying state -0.112 -- the 521 ms clog
    itself, sailed clean over) and DOES estimate 1.0 / 0.3, which reads a median of
    exactly 0.0 -- and a worst state of 0.556, the 473 ms ridden-into run of the
    clog region. Detection at the two windows, through the app: 2.0 / 0.5 finds the
    clog as ONE 519.5 ms event; 1.0 / 0.3 does not find the clog at all and stamps
    ~30 shallow events there on a local baseline below -45 pA. The estimator still
    reproduces the published 2.0 / 0.5 and 0.1 / 0.03 windows bit-for-bit when they
    are typed in -- what changes is which windows the app OFFERS.

    The gating branch is excluded, and it is the same clause of the contract that
    exempts it from the floor: ``baseline_intrusion`` renders NOT APPLICABLE on a
    credibly gating pore, so there is no number there to rank on. See
    `_rank_declined`.

    Args:
        signal: The trace the probe measured, WHOLE CHANNEL, same array and same
            conditioning -- a ranking measured on a different array than the probe is
            a ranking of a different recording.
        fs: Sample rate, Hz.
        proposal: What `propose_baseline` offered.

    Returns:
        A `BaselineProposal`. Either the ranked winner, carrying the whole ladder in
        ``ranking`` and the comparison in ``reason``, or ``proposal`` itself with a
        sentence saying why it was not ranked.
    """
    why = _rank_declined(proposal, int(np.asarray(signal).size), fs)
    if why is not None:
        return _unranked(proposal, why)

    probe = proposal.probe
    x = np.asarray(signal)
    n, fs = int(x.size), float(fs)
    pol = int(probe.polarity)
    ladder = candidate_windows(proposal, n, fs)
    if len(ladder) < 2:
        return _unranked(
            proposal,
            f"there is no other window on the grid between the shortest base window "
            f"Baseline health certifies here and the "
            f"{reference_rough_s(n, fs):g} s reference window to compare against")

    # THE ONE WIDE PASS. Every candidate's intrusion is read over this mask -- see
    # `baseline.config_intrusion` for why it is measured once and passed in.
    check_cancelled()
    ref = reference_states(
        x if pol == -1 else -x, fs,
        config=DEFAULT_BASELINE.replace(rough_s=reference_rough_s(n, fs) / 2.0))
    if ref is None:                                # unreachable: w / 2 < w
        return _unranked(proposal, "the reference pass found no states to rank against")
    ref_mask, _ = ref

    # ``i0`` is where the PROPOSAL sits on the two-sided ladder, and it is found by
    # value rather than assumed to be rung 1: `candidate_windows` seeds its dict with
    # exactly this tuple, so the equality is an identity of values and not a float
    # comparison that could miss.
    proposed = (float(proposal.config.rough_s), float(proposal.config.base_s))
    i0 = ladder.index(proposed)

    measured: dict[int, BaselineCandidate] = {}

    def estimate(i: int) -> BaselineCandidate:
        check_cancelled()
        rough, base = ladder[i]
        candidate = DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)
        intr = config_intrusion(
            x, fs, config=candidate, polarity=pol, state_mask=ref_mask)
        # ``+ 0.0`` normalises IEEE negative zero, which `tracked_intrusion` does
        # produce (measured on a 30 ms/20% synthetic at 1.0 / 0.3 s). It compares
        # equal to 0.0 everywhere it is used and is still noise in a rendered row and
        # in anything built from ``to_dict``.
        frac = None if intr is None else float(intr["fraction"]) + 0.0
        # ``worst_*`` can be None INSIDE a measured dict: no state on this recording
        # is longer than this candidate's tracking window, so there is nothing its
        # median could dwell inside and nothing for the criterion to read. That is
        # "not shown to ride", and `BaselineCandidate.rides` reads it as False.
        wf = None if intr is None else intr["worst_fraction"]
        c = BaselineCandidate(
            rough_s=float(rough), base_s=float(base), fraction=frac,
            n_states=None if intr is None else int(intr["n_states"]),
            cost=intrusion_cost(frac),
            worst_fraction=None if wf is None else float(wf) + 0.0,
            worst_dwell_s=(None if wf is None
                           else float(intr["worst_dwell_s"])),
            worst_t_s=None if wf is None else float(intr["worst_t_s"]))
        measured[i] = c
        return c

    def clean(c: BaselineCandidate | None) -> bool:
        # BOTH criteria, not one: a rung whose MEDIAN reads zero while its baseline
        # has ridden into one long state (`BaselineCandidate.rides`) is exactly the
        # candidate the -600 mV recording produces at 1.0 / 0.3 s, and treating it as
        # clean is the blindness this guard removes -- it would stop the walk on, and
        # then offer, a window that misses a 521 ms clog and invents shallow events.
        return (c is not None and c.cost is not None and c.cost <= 0.0
                and not c.rides)

    # PHASE 1 -- UP from the proposal, stop at the first rung CLEAN on both criteria
    # (median cost zero, nothing ridden into). EXACT: a clean rung's first two key
    # terms are at their joint minimum and the tie-break prefers the shorter window,
    # so no rung above it can win whatever it reads.
    for i in range(i0, len(ladder)):
        if len(measured) >= RANK_MAX_CANDIDATES:
            break
        if clean(estimate(i)):
            break

    # PHASE 2 -- DOWN, and ONLY from a proposal that is itself clean on both
    # criteria. That gate is what makes the descent both cheap and meaningful: a
    # proposal that intrudes or rides has already been beaten by something wider, and
    # every rung below it is shorter than the one that lost. The stop is the mirror of
    # phase 1's and is NOT exact -- see this function's docstring for the measurement
    # it rests on (intrusion grows AND riding deepens on the way down, on everything
    # in evidence) and for why `_ranking_reason` reports the windows it never tried
    # instead of claiming they were beaten. On -600 mV this is the stop that fires:
    # the first rung down, 1.0 / 0.3, rides into the clog and ends the descent.
    if clean(measured.get(i0)):
        for i in range(i0 - 1, -1, -1):
            if len(measured) >= RANK_MAX_CANDIDATES:
                break
            if not clean(estimate(i)):
                break

    # ASCENDING BY WINDOW, whatever order the two phases estimated in. The tie-break
    # below reads the ROUGH WINDOW rather than the position, so this ordering does not
    # decide anything -- but `_ranking_reason` and `proposal_summary` render these
    # rows in order, and a ladder printed out of order is a table the reader cannot
    # scan for the shortest window that reaches zero.
    ordered = [measured[i] for i in sorted(measured)]
    scored = [c for c in ordered if c.measured]
    if not scored:
        return _unranked(
            proposal,
            "no candidate window had a state deep enough to measure an intrusion "
            "against, so there was nothing to rank")

    # THE RULE, IN ONE LINE: a window that rides into a state loses to one that stays
    # out; then lowest cost; then ties to the SHORTEST window. The FIRST term is the
    # worst-state guard -- lexicographic, not weighted, because riding into a long
    # blocked state is not a quantity to trade against a median (on -600 mV it is the
    # difference between detecting a 519.5 ms clog and inventing ~30 shallow events
    # in its place). The second and third terms are unchanged, so on any ladder where
    # nothing rides the ranking is bit-for-bit what it was.
    winner = min(scored,
                 key=lambda c: (c.rides, float(c.cost), float(c.rough_s)))
    ranking = tuple(
        BaselineCandidate(rough_s=c.rough_s, base_s=c.base_s, fraction=c.fraction,
                          n_states=c.n_states, cost=c.cost, chosen=c is winner,
                          worst_fraction=c.worst_fraction,
                          worst_dwell_s=c.worst_dwell_s, worst_t_s=c.worst_t_s)
        for c in ordered)
    # ...and the object the message is written about must be one OF ``ranking``, not
    # the pre-copy original: `_ranking_reason` separates the winner from the beaten
    # candidates, and an identity comparison against the original matches nothing in
    # the tuple, so the winner would have been listed among the windows it beat.
    best = next(c for c in ranking if c.chosen)
    chosen_cfg = DEFAULT_BASELINE.replace(
        rough_s=best.rough_s, base_s=best.base_s, decim=None)
    return BaselineProposal(
        config=chosen_cfg, param_source=proposal.param_source,
        reason=proposal.reason + _ranking_reason(ranking, best, proposed, ladder),
        probe=probe, ranking=ranking)


def _unranked(proposal: BaselineProposal, why: str) -> BaselineProposal:
    """``proposal`` unchanged, saying on screen that its windows were not ranked.

    NOT SILENCE. The proposal that comes back from a declined ranking is the shortest
    window the floor certifies, which is a different claim from "the least-intruding
    window on the grid", and a user reading the same two numbers either way has no
    way to tell which one he is being offered.

    ``why=""`` is the one case that IS silent -- see `_rank_declined`.
    """
    if not why:
        return proposal
    return BaselineProposal(
        config=proposal.config, param_source=proposal.param_source,
        reason=(proposal.reason
                + f" These windows were NOT ranked by measured intrusion ({why}), so "
                  "they are the ones the rule derives from this recording's "
                  "timescales rather than the ones the intrusion ranking -- median "
                  "and worst state -- would have chosen from the grid."),
        probe=proposal.probe, ranking=())


def _fmt_window(rough: float, base: float) -> str:
    return f"{rough:g} s / {base * 1e3:g} ms"


def _candidate_phrase(c: BaselineCandidate) -> str:
    """One candidate, as the user reads it. Signed, and it says what a sign MEANS.

    "follows -0.040% of each state's depth" is a true number and an unreadable
    sentence, and the -500 mV recording produces exactly it. `intrusion_cost` treats
    anything at or below zero as "followed none of it" -- because the check it is a
    search over does -- so the phrase has to say the same thing, or the winning row
    reads as a negative quantity next to a reason line claiming zero.

    BOTH CRITERIA, because the ranking decides on both and a row that showed only the
    median would be evidence for a different objective than the one used. On the
    -600 mV recording the losing 1 s / 300 ms rung reads "follows NONE" on the median
    while its baseline sits 56% of the way into a 473 ms state -- the row has to
    carry the number that made it lose, or the table shows a shorter clean-looking
    window losing to a longer one for no visible reason.
    """
    w = _fmt_window(c.rough_s, c.base_s)
    if c.fraction is None:
        return f"{w} could not be measured"
    if c.rides:
        # NOT the "sits at or just outside them" gloss: that gloss is a claim about
        # where the baseline SITS, and this window's baseline is inside one state.
        # One sentence must not say both.
        med = (f"{c.fraction * 100:.3f}%" if c.fraction > 0
               else f"NONE ({c.fraction * 100:.3f}%)")
        return (f"{w} follows {med} of each state's own depth at the MEDIAN, but "
                f"RIDES INTO its worst single state -- "
                f"{c.worst_fraction * 100:.1f}% of a {c.worst_dwell_s * 1e3:.0f} ms "
                f"state's own excursion followed, at t = {c.worst_t_s:.1f} s -- "
                f"which one state among {c.n_states} cannot move a median to show")
    if c.fraction > 0:
        head = f"{w} follows {c.fraction * 100:.3f}% of each state's own depth"
    else:
        head = (f"{w} follows NONE of each state's own depth "
                f"({c.fraction * 100:.3f}%, i.e. its baseline sits at or just "
                "outside them)")
    if c.worst_fraction is not None:
        # The sign convention `intrusion_cost` and the median phrase already carry:
        # at or below zero the baseline sits outside even its worst state.
        if c.worst_fraction > 0:
            return head + f" (worst single state {c.worst_fraction * 100:.1f}%)"
        return head + (f" (worst single state NONE of its own excursion, "
                       f"{c.worst_fraction * 100:.1f}%)")
    # The worst-state reading was TAKEN and found nothing to read: no state here is
    # longer than this window's own tracking window. Said, because the header
    # promises the measurement and a row that omitted it would look unmeasured.
    return head + " (no state longer than its tracking window to ride into)"


def _ranking_reason(
    ranking: tuple[BaselineCandidate, ...],
    best: BaselineCandidate,
    proposed: tuple[float, float],
    ladder: tuple[tuple[float, float], ...],
) -> str:
    """The paragraph the user reads under the proposal. Every number in it measured.

    It names EVERY candidate estimated, not just the winner, because the claim being
    made is a comparison and a comparison with one row in it is an assertion. It says
    where the search stopped and why -- "we did not try 2 s" and "2 s was worse" are
    different statements, and only one of them is ever true. And it does NOT claim the
    winner is clean when it is not: a ladder on which every rung intrudes still has a
    least-intruding rung, and the sentence for that case has to say what the chosen
    window still costs rather than reuse the sentence for the case that reaches zero.

    TWO SENTENCES THIS FUNCTION USED TO GET WRONG, AND BOTH WERE READ ON A REAL SCREEN
    ==================================================================================
    * **"Nothing can beat a baseline that follows none of the states, so no wider
      window was estimated."** The second clause was true and the first was FALSE
      under this module's own objective: what beats a clean window is a SHORTER clean
      one, on the tie-break, and the search had never looked there. On the -600 mV apo
      recording that window existed. The clean branch below now says which shorter
      windows were estimated, what they read, and how many were not estimated at all.
    * **The whole paragraph appended silently to a derivation that had already named
      different numbers.** On -700 mV the sentences above this one declare the rule's
      0.5 s / 100 ms three times and then this paragraph takes 1 s / 300 ms, with
      nothing between them saying the first two numbers had been superseded. ``proposed``
      is passed in for exactly that: when the ranking moves the window, it says so in
      the first sentence, before any of the evidence.

    ``ladder`` is here so the counts are of REAL rungs. "3 shorter windows were not
    estimated" has to be three windows that exist between the floor and this one, not
    an inference from the grid.

    TWO MORE SENTENCES THE REPLACEMENT ITSELF GOT WRONG, and the rule they broke
    ===========================================================================
    The RP rewrite replaced the false sentence above with two new ones, both printed
    on the -600 mV screen under an offer of the window that rides into that
    recording's 521 ms clog:

    * **"nothing WIDER can beat a baseline that follows none of the states, and a
      wider window that tied would lose the tie-break anyway"** -- false as a claim
      about baselines: on -600 mV the wider 2 s / 500 ms window IS better, decisively
      (it detects the clog; the shorter one's baseline had ridden into it), and
      "follows none of the states" was only ever the MEDIAN. What is true is a claim
      about the OBJECTIVE over the ladder walked, and that is what is printed now.
    * **"(N wider windows on the grid were not estimated at all: they could at best
      tie, and would lose the tie-break.)"** -- same defect: "could at best tie" is
      arithmetic about the old objective dressed as a fact about the windows.

    THE RULE THIS FUNCTION NOW FOLLOWS: every optimality claim is scoped to the
    ladder walked and to the objective used -- median intrusion, the worst-state
    criterion (`WORST_STATE_INTRUSION_MAX`, named in the paragraph), shortest on a
    tie -- and a sentence about a window never estimated may describe the stop rule,
    never that window.
    """
    clean = best.cost is not None and best.cost <= 0.0 and not best.rides
    out = ""
    if (best.rough_s, best.base_s) != tuple(proposed):
        out += (
            f" THE {_fmt_window(*proposed)} DERIVED ABOVE IS THE RULE'S WINDOW AND NOT "
            f"WHAT THE APP OFFERS: every sentence up to here explains how the rule "
            f"reaches it, and the ranking that follows SUPERSEDES it with "
            f"{_fmt_window(best.rough_s, best.base_s)}."
        )
    out += (
        f" RANKED BY MEASURED INTRUSION, on two criteria. The MEDIAN share of each "
        f"state's own depth the tracked baseline follows is the same statistic "
        f"Baseline health condemns above {BASELINE_INTRUSION_MAX:.0%} of the state "
        f"depth, from the same estimator; the WORST single state is measured beside "
        f"it because one state ridden into cannot move a median over hundreds, and a "
        f"window whose baseline follows more than "
        f"{WORST_STATE_INTRUSION_MAX:.0%} of any one state's excursion has ridden "
        f"INTO that state and loses to any window that stays out. Measured here: "
        + "; ".join(_candidate_phrase(c) for c in ranking) + ". "
    )
    # ``not c.chosen``, not ``c is not best``: `rank_windows` copies every candidate to
    # stamp the flag, so an identity test against anything outside ``ranking`` matches
    # every row -- including the winner -- and the sentence below would name the chosen
    # window as one of the ones it beat.
    beaten = [c for c in ranking
              if not c.chosen and c.cost is not None and c.cost > 0.0]
    if clean and beaten:
        costliest = max(beaten, key=lambda c: float(c.cost))
        out += (
            f"The median depth measured at "
            f"{_fmt_window(costliest.rough_s, costliest.base_s)} therefore reads "
            f"about {costliest.fraction * 100:.1f}% shallow, and at "
            f"{_fmt_window(best.rough_s, best.base_s)} it does not. "
        )
    # A RIDDEN candidate's harm is not "reads a little shallow" and must not be
    # summarised as if it were: inside the one state it entered, depths are measured
    # from within the state itself. Say the measured share and the measured state.
    riders = [c for c in ranking if not c.chosen and c.rides]
    if clean and riders:
        deepest = max(riders, key=lambda c: float(c.worst_fraction))
        out += (
            f"Inside the {deepest.worst_dwell_s * 1e3:.0f} ms state that "
            f"{_fmt_window(deepest.rough_s, deepest.base_s)} rides into, depths "
            f"would be measured against a baseline that has followed "
            f"{deepest.worst_fraction * 100:.0f}% of the state itself -- a blockade "
            f"that long can stop being detectable at all"
            + (f" -- while at {_fmt_window(best.rough_s, best.base_s)} no single "
               f"state is followed more than "
               f"{max(best.worst_fraction, 0.0) * 100:.1f}% of the way"
               if best.worst_fraction is not None else
               f" -- while at {_fmt_window(best.rough_s, best.base_s)} no state is "
               "even as long as the tracking window, so there is nothing its "
               "median could dwell inside")
            + ". ")
    # WHAT WAS NEVER ESTIMATED, counted off the ladder rather than inferred. The
    # search is two-sided and both of its stops have to be reported: an unestimated
    # window supports no claim at all, in either direction.
    lo = min(c.rough_s for c in ranking)
    hi = max(c.rough_s for c in ranking)
    unseen_below = [w for w in ladder if w[0] < lo]
    unseen_above = [w for w in ladder if w[0] > hi]
    shorter_est = [c for c in ranking if c.rough_s < best.rough_s]
    wider_est = [c for c in ranking if c.rough_s > best.rough_s]
    # ...and separately from ``unseen_below``, which is relative to the ESTIMATED
    # range. "There is no shorter window at all" is a claim about the LADDER relative
    # to the WINNER, and the two differ whenever the descent ran past the winner and
    # stopped on the ladder's own floor: there ``unseen_below`` is empty while a
    # shorter window plainly exists, was estimated, and lost.
    ladder_below_best = [w for w in ladder if w[0] < best.rough_s]

    out += f"{_fmt_window(best.rough_s, best.base_s)} is taken: "
    if len(ranking) == 1:
        out += ("the only window the search estimated -- see below for what it did "
                "not try and why. ")
    else:
        out += ("it is the SHORTEST window ESTIMATED that scores best on both "
                "criteria -- riding into no state, then the lowest measured median "
                "intrusion. Shortest, because a longer window is not free -- it "
                "tracks slower drift less well -- so among windows that measure the "
                "same on both criteria the app takes the least smoothing it can get "
                "away with. ")
    if clean:
        # THE SENTENCE THAT WAS FALSE, TWICE. The first rewrite of this branch said
        # "nothing WIDER can beat a baseline that follows none of the states", which
        # the -600 mV recording refutes in fact: the wider 2 s / 500 ms window IS
        # better there, because "follows none" was only the MEDIAN and the shorter
        # window had ridden into the clog. The claim printed now is scoped to the
        # objective over the rungs walked, and says so.
        # ...and whether wider rungs TIE is checked on BOTH criteria rather than
        # assumed: "they tie" is a claim about a measurement, and this function has
        # been wrong about exactly that kind of claim twice.
        tied = [c for c in wider_est
                if c.cost is not None and c.cost <= 0.0 and not c.rides]
        if wider_est and len(tied) == len(wider_est):
            many = len(wider_est) > 1
            out += (
                f"The {len(wider_est)} wider window{'s' if many else ''} estimated "
                f"above it "
                f"({'; '.join(_fmt_window(c.rough_s, c.base_s) for c in wider_est)}) "
                f"{'follow' if many else 'follows'} none of the states and "
                f"{'ride' if many else 'rides'} into none either, so "
                f"{'they tie' if many else 'it ties'} on both criteria and the tie "
                "goes to the shorter window; ")
        elif wider_est:
            out += (
                f"The {len(wider_est)} wider window"
                f"{'s' if len(wider_est) > 1 else ''} estimated above it "
                f"({'; '.join(_fmt_window(c.rough_s, c.base_s) for c in wider_est)}) "
                "did not beat it; ")
        else:
            out += "No wider window was estimated: "
        out += ("under this ranking's objective -- no state ridden into, then median "
                "intrusion, ties to the shortest window -- nothing WIDER can beat a "
                "window already clean on both criteria: it could at best match them "
                "and then lose the tie-break. That is a claim about the objective on "
                "the ladder walked, not a measurement of any window left "
                "unestimated. ")
        if shorter_est:
            nearest = max(shorter_est, key=lambda c: float(c.rough_s))
            if nearest.rides:
                nearest_read = (
                    f"RIDES INTO a {nearest.worst_dwell_s * 1e3:.0f} ms state "
                    f"({nearest.worst_fraction * 100:.1f}% of its own excursion "
                    "followed)")
            elif (nearest.measured and nearest.fraction is not None
                    and nearest.fraction > 0):
                nearest_read = (f"follows {nearest.fraction * 100:.3f}% of each "
                                "state's depth")
            else:
                nearest_read = "could not be measured"
            out += (
                "What could have beaten it is a SHORTER window that is equally clean "
                "on both criteria, and the search looked: the nearest one, "
                f"{_fmt_window(nearest.rough_s, nearest.base_s)}, was estimated and "
                + nearest_read + ", so nothing shorter was taken. ")
        if unseen_below:
            out += (
                f"{len(unseen_below)} shorter window"
                f"{'s' if len(unseen_below) > 1 else ''} on the grid "
                f"({'; '.join(_fmt_window(*w) for w in unseen_below)}) "
                f"{'were' if len(unseen_below) > 1 else 'was'} NOT estimated -- "
                "intrusion grows, and riding deepens, as the tracking window "
                "shortens, which is measured and not proved, so this is where the "
                "search stops rather than a claim that they are worse.")
        elif not ladder_below_best:
            out += ("There is no shorter window on the grid whose tracking window "
                    "Baseline health would certify, so this is the shortest the "
                    "search could offer.")
        else:
            # The descent ran off the bottom of the ladder and the LAST rung is the
            # one that stopped it. Everything shorter than the winner was estimated,
            # so there is nothing left to disclose -- and saying "there is no shorter
            # window" here would be false about windows named two sentences earlier.
            out += ("Every shorter window on the grid was estimated, and each of them "
                    "follows the states.")
        if unseen_above:
            many = len(unseen_above) > 1
            out += (f" ({len(unseen_above)} wider window{'s' if many else ''} on the "
                    f"grid {'were' if many else 'was'} not estimated at all: under "
                    f"the objective above {'they' if many else 'it'} could at best "
                    "match this window on both criteria and then lose the tie-break "
                    "-- a stop rule, not a measurement of "
                    f"{'them' if many else 'it'}.)")
    else:
        # COUNTED SEPARATELY, because "no better" is a claim and an unmeasurable
        # window supports no claim at all. Saying "the 1 wider window above it was no
        # better" about a candidate that produced no number is the same defect this
        # whole change is about, one sentence along.
        beat = [c for c in wider_est if c.measured]
        blind = [c for c in wider_est if not c.measured]
        if best.cost is not None and best.cost > 0.0:
            out += (
                f"It still follows {best.fraction * 100:.3f}% of each state's depth "
                "at the median, so depths measured with it read about that much "
                "shallow. ")
        if best.rides:
            # Reachable only when EVERY measured rung rides (a non-rider beats any
            # rider outright), so this window is the least-ridden of a set of bad
            # ones -- say that, never something gentler.
            out += (
                f"It RIDES INTO its worst single state "
                f"({best.worst_fraction * 100:.1f}% of a "
                f"{best.worst_dwell_s * 1e3:.0f} ms state's own excursion followed): "
                "every window estimated here does, so this is the least-ridden of "
                "them, and depths inside that state cannot be trusted at any of "
                "these windows. ")
        out += ("This is the best window ESTIMATED under this ranking's objective "
                "(no state ridden into where possible, then median intrusion), NOT "
                "a clean one")
        if beat:
            out += (f", and the {len(beat)} wider window"
                    f"{'s' if len(beat) > 1 else ''} estimated above it "
                    f"{'were' if len(beat) > 1 else 'was'} no better under that "
                    "objective")
        if blind:
            out += (f"; {len(blind)} wider window"
                    f"{'s' if len(blind) > 1 else ''} could not be measured at all")
        if not beat and not blind:
            out += (" -- and the search hit its bound before finding one clean on "
                    "both criteria, so a wider window than any tried here may "
                    "still be better")
        out += "."
        # THE SHORT SIDE, on a ladder where the winner is not clean. The descent is
        # gated on a clean proposal precisely because every rung below an intruding
        # window is shorter than one that already intrudes -- and that is a reason to
        # stop, not evidence about those windows. Say which it is.
        if shorter_est:
            nearest = max(shorter_est, key=lambda c: float(c.rough_s))
            # "was worse" is a claim, and a candidate that produced no number does not
            # support one -- an unmeasurable rung does not stop the upward walk, so it
            # can sit below the winner and reach this sentence.
            out += (f" {_fmt_window(nearest.rough_s, nearest.base_s)} was estimated "
                    + ("below it and lost under the same objective."
                       if nearest.measured else
                       "below it and could not be measured."))
        if unseen_below:
            out += (
                f" {len(unseen_below)} shorter window"
                f"{'s' if len(unseen_below) > 1 else ''} on the grid "
                f"({'; '.join(_fmt_window(*w) for w in unseen_below)}) "
                f"{'were' if len(unseen_below) > 1 else 'was'} NOT estimated: the "
                "search does not descend below a window that already follows the "
                "states or rides into one, so nothing here is a claim that they are "
                "worse.")
    return out


def probe_and_propose(
    signal: np.ndarray, fs: float, *, polarity: int | None = None
) -> BaselineProposal:
    """`probe_timescales`, then `propose_baseline`, then `rank_windows`.

    Off the GUI thread, please -- re-measured 2026-09-01 on the owner's recordings:
    the probe stage alone is 343 ms at 3.0 M samples and 1,073 ms at 9.1 M (medians
    of three; see `ProbeResult.elapsed_s`), and this whole function -- probe +
    proposal + `rank_windows`, which adds one wide pass plus one estimate per
    candidate -- is 0.4 s on the 0.75 M -1000 mV excerpt, 1.0 s on the 3.0 M -600 mV
    file and 3.4 s on the 9.1 M -700 mV one (an earlier 2.5 s was measured on the
    12.5 M -500 mV file). See `RANK_MAX_CANDIDATES` for the bound.

    THE RANKING IS HERE AND NOT IN `propose_baseline`, and that is a layering rather
    than an oversight: `propose_baseline` turns a `BaselineProbe` into windows and has
    no trace to measure anything on, while ranking needs the trace. This is the app's
    single entry point (``gui.analysis_panel`` calls exactly this), so nothing in the
    app can get the unranked answer by accident -- and a caller that genuinely wants
    the rule without the measurement still has `propose_baseline` by name.
    """
    return rank_windows(
        signal, fs, propose_baseline(probe_timescales(signal, fs, polarity=polarity)))


def proposal_summary(
    proposal: BaselineProposal, fs: float | None = None
) -> tuple[tuple[str, str], ...]:
    """``((label, value), ...)`` -- what a dialog shows above the reason line.

    Lives here, not in the widget, so the sentence the user reads and the numbers
    the estimator runs are produced by one function from one object.
    """
    p = proposal.probe
    cfg = proposal.config
    decim = cfg.decim if cfg.decim is not None else (
        derive_decim(cfg.base_s, fs) if fs else None)
    rows: list[tuple[str, str]] = []
    if p is not None:
        rows.append(("Noise (robust sigma)",
                     f"{p.sigma_pA:.2f} pA" if np.isfinite(p.sigma_pA) else "n/a"))
        rows.append(("Longest state",
                     f"{p.longest_state_s * 1e3:.0f} ms  ({p.n_states} states)"
                     if p.longest_state_s > 0 else "none found"))
        # The number `base_vs_state` divides by, shown beside the one `rough_s` is
        # sized against, because the two differ by 1.5x here and 6.4x on
        # ferritin_data2 and a reader who sees only "Longest state" cannot check the
        # base window against anything. `floor` is `base_floor_s`, not a re-derivation
        # of it -- the rule that applied it and the line that reports it are one call.
        floor = base_floor_s(p)
        if p.state_dwell_s is not None and np.isfinite(p.state_dwell_s):
            rows.append(("Median state (base_s floor)",
                         f"{p.state_dwell_s * 1e3:.1f} ms"
                         + (f"  -> base_s >= {floor * 1e3:g} ms"
                            if floor is not None else
                            "  -- not applied (the pore steps between levels)")))
        rows.append(("Pore stepping (gating)",
                     (f"yes -- {p.gating_score:.1f}x noise" if p.gating_credible else
                      f"{p.gating_score:.1f}x noise, but {p.mask_fraction:.0%} of the "
                      "trace is masked -- events, not levels")
                     if p.gating
                     else f"no -- {p.gating_score:.1f}x noise"
                     if np.isfinite(p.gating_score) else "not measured"))
        if p.level_dwell_s:
            rows.append(("Level dwell", f"{p.level_dwell_s * 1e3:.0f} ms"))
    rows.append(("Proposed rough window", f"{cfg.rough_s:g} s"))
    rows.append(("Proposed base window", f"{cfg.base_s * 1e3:g} ms"))
    rows.append(("Decimation (derived)", "auto" if decim is None else str(decim)))
    # THE EVIDENCE FOR THE TWO NUMBERS ABOVE, as a table rather than as a claim.
    # One row per window ESTIMATED, labelled by the window itself so the labels are
    # unique -- callers read these rows back through ``dict(proposal_summary(...))``
    # and a repeated label would silently drop a candidate out of the evidence.
    # Placed AFTER the proposal rather than before it because it explains a number the
    # reader has already found; `reason` carries the same comparison in a sentence.
    for c in proposal.ranking:
        # NO ANGLE BRACKET IN A VALUE THAT LANDS IN RICH TEXT. `BaselinePanel` renders
        # this table as HTML; a "<- chosen" marker is a malformed tag to Qt's parser
        # and it eats the marker and everything after it (see
        # `BaselinePanel._check_html`, where exactly that deleted a criterion on its
        # way to the screen). The panel escapes these now; the marker still does not
        # rely on it.
        mark = "  — CHOSEN" if c.chosen else ""
        # BOTH criteria on every measured row, same reason `_candidate_phrase` gives:
        # the ranking decides on the worst single state as well as the median, and a
        # table showing only the median would show the -600 mV screen a shorter
        # clean-looking window losing for no visible reason.
        if c.rides:
            worst_note = ""
        elif c.worst_fraction is None:
            # Measured, and found nothing to measure: no state on this recording is
            # longer than this window's own tracking window. Said in the row for the
            # same reason `_candidate_phrase` says it -- the criterion was applied.
            worst_note = "; no state long enough to ride into"
        elif c.worst_fraction > 0:
            worst_note = f"; worst single state {c.worst_fraction * 100:.1f}%"
        else:
            # at or below zero: outside even the worst state, same sign convention
            # as the NONE branch below.
            worst_note = (f"; worst single state NONE "
                          f"({c.worst_fraction * 100:.1f}%)")
        if c.fraction is None:
            value = f"not measurable here{mark}"
        elif c.rides:
            med = (f"{c.fraction * 100:.3f}%" if c.fraction > 0
                   else f"NONE ({c.fraction * 100:.3f}%)")
            value = (f"baseline follows {med} of the state depth at the median "
                     f"({c.n_states} states) but RIDES INTO a "
                     f"{c.worst_dwell_s * 1e3:.0f} ms state at "
                     f"t = {c.worst_t_s:.1f} s, "
                     f"{c.worst_fraction * 100:.1f}% of its excursion followed"
                     f"{mark}")
        elif c.fraction > 0:
            value = (f"baseline follows {c.fraction * 100:.3f}% of the state depth "
                     f"({c.n_states} states{worst_note}){mark}")
        else:
            # Same reason `_candidate_phrase` does this: a negative reading is
            # "followed none of it" to the check and to `intrusion_cost`, and a bare
            # "-0.040%" beside a CHOSEN mark reads as a defect rather than as a pass.
            value = (f"baseline follows NONE of the state depth "
                     f"({c.fraction * 100:.3f}%, {c.n_states} states{worst_note})"
                     f"{mark}")
        rows.append((f"Ranked: {c.rough_s:g} s / {c.base_s * 1e3:g} ms", value))
    return tuple(rows)


