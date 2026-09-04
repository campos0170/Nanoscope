"""Open-pore baseline (I0) tracking (signal-first port).

Baseline drift is endemic in nanopore recordings: I0 wanders with temperature,
electrode polarisation, and pore wetting, often by more than a shallow event's
depth over tens of seconds. A single global threshold is therefore hopeless --
the baseline must be tracked as a function of time.

THE CENTRAL HAZARD -- A BASELINE THAT FOLLOWS EVENTS EATS THEM
--------------------------------------------------------------
Every estimator here is a smoother. If its window is short compared to an event,
it will dive into the event, the local baseline will sit at the blocked level,
and the event will silently vanish from detection -- or survive with an
under-reported depth. There is no error message for this; the numbers just come
out wrong. **window_s must be much longer than the longest dwell you care
about** (>= 20x is the guard enforced here, >= 50x is comfortable).

Use `baseline_intrusion()` to quantify it after the fact: it measures how far the
baseline moved toward the events, as a fraction of event depth. Anything above
~0.1 means the baseline is eating the science.

Sign convention (from `nanoscope.analysis.events`): dev = polarity * (current -
baseline) is positive during an event, for blockades and enhancement alike. So
`polarity` names the side the excursions go, and the open-pore level is the
envelope on the *opposite* side. That is what steers 'percentile' and 'asls'.

Re-seamed onto the target neo-based data model: every entry point takes explicit
``(current, fs, polarity, ...)`` numpy args rather than a ``Recording``. The
target stores no polarity, so `derive_polarity()` measures it from the trace
(detected, never assumed; enhancement +1 is real physics). Pass ``polarity=None``
to `estimate_baseline` to let it derive.

All currents are pA, all times seconds. Baselines come back float64 and
full-length, one value per sample.

THE EVENT-MASKED TWO-PASS ESTIMATOR (`method='masked_median'`)
-------------------------------------------------------------
The block median above is only robust while event occupancy inside a block stays
under 50%. On the owner's own bursty recordings it does not: on
``500 pM Apo ferritin_..._0006_Process_-600 mV.abf`` the 10 ms block median stamps
80.6% of events with a baseline more than 10 pA from the true open pore, and
``depth_pA`` comes out 4.7x too small (measured; the spec's 14.7x does not reproduce). Widening the window is not a general fix --
on a gating pore (``ferritin_data3``) a 200 ms window measures 5.7x worse than a
30 ms one.

The fix is the estimator the owner already uses in his own analysis
(``FerritinModeler/dataLoader.ipynb::baseline_remove_masked_median``): a rough
long-window median exposes the events, a one-sided MAD threshold masks them, the
SIGNAL is linearly interpolated across the mask, and a SHORT window median of the
gap-filled trace becomes the baseline. Two windows, decoupled: ``rough_s`` decides
what counts as an event, ``base_s`` decides how fast the baseline may track.

**The kernel is IMPORTED from `nanoscope.analysis.ml_preprocess`, never
reimplemented here.** That module's `baseline_masked_median` /
`baseline_robust_median` / `_decimated_median` are bit-exact against the owner's
persisted training corpus (``CSVData/ferritin_data{1,2,3}.csv.gz``, max|diff| = 0)
and are already hardened against the non-deterministic 1-D
``scipy.ndimage.median_filter``. A second copy of a 20-line function with five
silent bit-exactness traps is the single largest risk in this change; there is
exactly one definition of `_decimated_median` in this tree and it must stay that
way. See `decimated_median` below, which re-exports it under a public name.

PARAMETERS ARE PER-RECORDING. THE APP PROPOSES; THE USER CONFIRMS.
------------------------------------------------------------------
There is no defensible global default: ``rough_s`` spans 0.03 s (a gating pore) to
2.0 s (a file with a 521 ms partial clog) across four real recordings -- a factor
of 67. `DEFAULT_BASELINE` is the notebook's own fallback (0.5 / 0.1), a floor
rather than an answer. `baseline_health` measures the failure modes the reference
algorithm cannot see; render them, do not auto-tune against them (the published
QC triple is monotonically improved by shrinking the window, so optimising it
selects the shortest possible window -- exactly the failure being fixed).

AND ``base_s`` IS SIZED AGAINST THE EVENTS, NOT AGAINST THE RECORDING
--------------------------------------------------------------------
A per-recording proposal is not by itself a guard, because the proposal can be
wrong. On the -700 mV apo recording the probe measured a 31 ms longest state and
proposed a 30 ms ``base_s``; a tracking median whose window is the size of the
states rides into them, and that file's depths came out ~3% shallow with
`BaselineHealth.ok` True and every check green. `BASE_STATE_RATIO_MIN` and
`BASELINE_INTRUSION_MAX` are the two checks that can see it -- the window against
the state timescale, and the finished baseline against the open pore beside each
state.

THE PROPOSER IS NOW TAUGHT THAT BOUND, AT THE POINT OF PROPOSAL
---------------------------------------------------------------
The two halves of that contract used to be asymmetric.
`baseline_probe.propose_baseline` sized ``base_s`` as
``ROUGH_MULTIPLE * longest_state_s / ROUGH_BASE_RATIO`` -- off the **p99.9** of the
dwell distribution -- and had never been told `BASE_STATE_RATIO_MIN`, which is
measured against the **median** (`state_dwell_s`). The two statistics differ by 1.5x
on the -700 mV apo recording and by 6.4x on ferritin_data2, so the proposer was free
to offer a window this check then condemned, and it was deterministic: two
consecutive probes of the -700 mV file both returned 0.1 / 0.03 s, at 1.38x a floor
of 3.0, for ever.

`reference_state_dwell_s` is the seam that closed it: the proposer now divides by the
SAME number this check divides by (bit-identical on all six recordings in evidence)
and floors ``base_s`` at `BASE_STATE_RATIO_MIN` times it, snapped UP onto the
proposal grid so the floor really is one -- log-NEAREST would have put the -700 mV
file at 0.05 s and 2.30x, still condemned.

**A configuration this check condemns for being too short to track is therefore no
longer proposable -- WHENEVER THE CHECK IS READING THE REFERENCE MASK.** That
qualifier is measured, not hedging. When the CONFIGURED ``rough_s`` is at or above
`reference_rough_s`, `reference_states` returns ``None`` and the block above measures
the state dwell on the configured pass instead, which is a different mask and a
different number; the floor was built on the reference one. Measured on a 3 s
synthetic with 120 ms states (reference window 0.75 s): the floor lifts ``base_s``
from 0.1 to 0.3 s and ``rough_s`` to 1.0, at which point the reference pass drops out,
the configured pass reads a 109 ms state rather than the reference pass's 77 ms, and
``base_vs_state`` still warns at 2.75. The floor took it from 1.29x to 2.75x and did
not clear it. The remedy below detects that case and says so rather than sending the
user to a probe that cannot fix it.

The two compensations therefore remain, as backstops rather than as workflow steps
(no recording in the owner's corpus reaches either any more): the remedy below
still names widening ``base_s`` by hand (the user may reach a condemned window that
way, or restore one from a project file), and `nanoscope.gui.baseline_panel` still
refuses the first Apply click on a proposal whose health block is not ``ok``.

The floor is NOT applied on a credibly gating pore, and that is the same exemption
this check makes: ``base_vs_state`` computes ``short = (not gating) and ...`` with
``gating`` spelled as "scores above `GATING_THRESHOLD` AND masks no more than
`MASK_FRACTION_CEILING` of the trace", and `baseline_probe.BaselineProbe.
gating_credible` is that same predicate on the probe's own two measurements. On a
stepping pore the long states ARE baseline levels and ``base_s`` is SUPPOSED to be
shorter than they are; flooring there would have moved ferritin_data3 off its
known-good 0.03 / 0.01 s to 0.1 / 0.02, which is the corpus contract this module
holds.
"""

from __future__ import annotations

import hashlib
import math
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Iterable, Mapping

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_dilation, maximum_filter1d
from scipy.signal import savgol_filter

from .cancel import check_cancelled

# THE KERNEL. Imported, never copied -- see the module docstring and
# ``tests/analysis/test_masked_baseline.py::test_exactly_one_decimated_median_in_the_tree``.
# ``_decimated_median`` is private in its own module only because nothing outside it
# needed it; re-exporting it here under a public name gives it a supported spelling
# without touching that file (its md5 is a frozen contract).
from .ml_preprocess import _decimated_median as decimated_median
from .ml_preprocess import baseline_masked_median as _masked_median_kernel
from .ml_preprocess import baseline_robust_median as _robust_median_kernel
from .ml_preprocess import despike_median

__all__ = [
    "derive_polarity",
    "require_polarity",
    "estimate_baseline",
    "estimate_baseline_result",
    "characterisation_baseline",
    "baseline_config_from_params",
    "BATCH_SOURCE",
    "BATCH_PROVENANCE_CLAUSE",
    "BATCH_SCOPE_KEY",
    "resolve_param_source",
    "batch_copied_warning",
    "clear_baseline_cache",
    "repair_nonfinite",
    "BASELINE_PARAM_KEYS",
    "PARAM_SOURCES",
    "FALLBACK_TOO_SHORT",
    "FALLBACK_CLAMPED",
    "adapt_to_length",
    "open_pore_current",
    "baseline_track",
    "baseline_subtracted",
    "decimate_track",
    "BASELINE_TRACK_MAX_POINTS",
    "remove_baseline_wander",
    "estimated_wander",
    "baseline_intrusion",
    "baseline_health",
    "require_finite",
    "measured_polarity",
    "upper_mad_sigma",
    "derive_decim",
    "decimated_median",
    "despike_median",
    "BaselineConfig",
    "BaselineCheck",
    "BaselineHealth",
    "BaselineResult",
    "BaselineWarning",
    "ConstantBaselineWarning",
    "AmplifiedWanderWarning",
    "collect_baseline_warnings",
    "baseline_progress_scope",
    "DEFAULT_BASELINE",
    "UNPROBED_DEFAULT_WARNING",
    "level_occupancy_warning",
    "alternative_level",
    "ALT_LEVEL_OCCUPANCY_MIN",
    "LEVEL_DECISIVE_RATIO",
    "LEVEL_HOLD_MIN_WINDOWS",
    "DEFAULT_DETREND_WINDOW_S",
    "METHODS",
    "MASKED_METHODS",
    "SAVGOL_WINDOW_S",
    "SAVGOL_POLY",
    "SAVGOL_DECIM",
    "TAPS_TARGET",
    "MIN_TAPS",
    "MASK_FRACTION_FLOOR",
    "MASK_FRACTION_CEILING",
    "MASK_REFERENCE_MAX_RATIO",
    "REFERENCE_ROUGH_S",
    "REFERENCE_MIN_ROUGH_S",
    "REFERENCE_WINDOWS_MIN",
    "ROUGH_BASE_RATIO",
    "reference_rough_s",
    "reference_mask_fraction",
    "reference_states",
    "reference_state_dwell_s",
    "state_dwell_s",
    "state_runs",
    "tracked_intrusion",
    "config_intrusion",
    "gating_spread",
    "block_medians",
    "stride_for",
    "BASE_STATE_RATIO_MIN",
    "BASELINE_INTRUSION_MAX",
    "INTRUSION_FLANK_S",
    "INTRUSION_PAD_S",
    "INTRUSION_MIN_DEPTH_SIGMA",
    "INTRUSION_MAX_STATES",
    "GATING_BLOCK_S",
    "GATING_THRESHOLD",
    "GATING_BLOCKS_MIN",
    "SIGMA_STRIDE",
    "SIGMA_STRIDE_MIN_N",
    "ROUGH_SIGMA_RATIO_MAX",
    "ROUGH_SIGMA_RATIO_MIN",
    "OVERSHOOT_FLAG",
]

#: The two-pass event-masked estimator and its single-pass sibling, both of which
#: come from `ml_preprocess`. Kept separate from the block-based methods because
#: they take a different parameter set (``rough_s``/``base_s``/``decim``, not
#: ``window_s``) and do their own two-pass masking (``two_pass=True`` is refused).
MASKED_METHODS = ("masked_median", "robust_median")

#: Every estimator `estimate_baseline` accepts. ADDITIVE ONLY -- the first four are
#: frozen (every stored result and every existing test depends on them being
#: reachable and bit-identical), then the two import-the-kernel paths, then
#: ``'savgol'``. Order is part of the contract: `test_masked_baseline` asserts the
#: first four by position, and a new estimator therefore goes on the END.
METHODS = ("block_median", "percentile", "asls", "constant", *MASKED_METHODS, "savgol")

#: The ``'savgol'`` baseline's shape, from the Cu2-insulin recipe -- the ONLY
#: parameter set in evidence for it, and it is identical in all four written
#: ``CSVData/cu_*_metadata.json`` (``sg_window_s`` 0.2, ``sg_decim`` 50,
#: ``sg_poly`` 3, ``masked_median_params`` null).
#:
#: THE 0.2 s WINDOW IS EVALUATED ON THE DECIMATED GRID: at 50 kHz / decim 50 that is
#: ``int(0.2 * 50000 / 50)`` = 200 -> forced odd -> **201 taps** over a 1 kHz series,
#: not 10 001 taps over the raw one. Decimation is not an optimisation here, it is
#: what makes the stage runnable at all: ``savgol_filter`` convolves directly, so an
#: undecimated window is O(n*w) and a 15 M-sample recording takes hours.
#:
#: NOT the same numbers as `ml_preprocess.baseline_savgol`, which is the APO chain's
#: SavGol: undecimated, ``w = int(window_s * fs)``, scipy's default ``mode='interp'``.
#: Same filter family, different estimator -- see :func:`_savgol_baseline`.
SAVGOL_WINDOW_S = 0.2

#: ``'savgol'`` polynomial order. Degree 3 in every Cu metadata file.
SAVGOL_POLY = 3

#: ``'savgol'`` decimation stride. 50 in every Cu metadata file; at 50 kHz that puts
#: the fit on a 1 kHz grid, three decades above the sub-Hz quantity being estimated.
SAVGOL_DECIM = 50

# Below this ratio of window_s to the shortest dwell we intend to keep, the
# smoother starts tracking events rather than the open pore.
MIN_WINDOW_TO_DWELL = 20.0

# Default smoothing window for baseline-WANDER removal (detrending), a deliberately
# far wider default than estimate_baseline's 10 ms I0-tracking window. Drift is a
# seconds-scale phenomenon (electrode polarisation, temperature, slow clogging), so
# the detrend window is chosen wide enough to sail over ms-scale events untouched --
# it should only subtract the slow envelope, never the events themselves.
DEFAULT_DETREND_WINDOW_S = 0.5

# Fraction of a full block a trailing remainder must reach to earn its own knot.
# A 3-sample "block median" is noise; better to let Pchip cover the tail.
_MIN_TAIL_FRAC = 0.25

# The window_s the block-based methods have always defaulted to. Kept as a private
# constant rather than a signature default so ``window_s=None`` can mean "whatever
# this method's own default is" -- 10 ms for the block methods (frozen: every
# existing caller and test depends on it), 0.5 s for 'robust_median', whose 10 ms
# rolling median would be 11 taps at decim 50 and would track the events.
_LEGACY_WINDOW_S = 0.01

#: Target tap count for the derived decimation (2.4). Cost of a decimated rolling
#: median goes as ``n * win_s * fs / decim**2`` -- decim is QUADRATIC -- so it is the
#: throttle, and it is derived from the window rather than exposed as a free knob.
#: 75 taps is ~1.8-6x faster than the corpus's decim 50 with visually identical
#: results on the real apo recording (median stamped error 0.91 pA vs 0.80 pA).
TAPS_TARGET = 75

#: Fewest taps a decimated median may run on. Below 3 the "median" is one or two
#: samples and the estimator is inoperative.
MIN_TAPS = 3

#: Mask-fraction floor, below which the mask found NO EVENTS (2.7).
#:
#: On an event-free trace the mask fraction is NOT zero: the +-``dilate_ms``
#: dilation multiplies the 3.5-sigma false-positive rate by ``2k+1`` = 101 at 1 ms /
#: 50 kHz, so pure noise masks ``Phi(-3.5) * 101`` = 2.35% by construction (measured
#: 2.26-2.60%). **A mask fraction near 2.3% therefore means "no events found", not
#: "events found".** Verified: 3.5 sigma / 0.2 ms -> 0.0052 (predicted 0.0049);
#: 5.0 sigma / 1.0 ms -> 0.0000.
MASK_FRACTION_FLOOR = 0.025

#: Mask-fraction ceiling, above which whole regions are being bridged rather than
#: masked. Measured on a synthetic gating pore: the mask grows monotonically with
#: ``rough_s`` (0.1145 -> 0.1998 -> 0.2984 -> 0.4006) while the fraction of samples
#: whose baseline error exceeds 100 pA goes 2.2% -> 14.0% -> 25.7% -> 38.2%.
#:
#: **It is an ABSOLUTE bound and it cannot see the failure `MASK_REFERENCE_MAX_RATIO`
#: is about**: a run whose mask fraction sits comfortably inside this band can still
#: be missing three quarters of the events, because the number it is being compared
#: against is a constant rather than this recording's own occupancy.
MASK_FRACTION_CEILING = 0.25

#: How much MORE of the trace a rough window wide enough to see over the states may
#: mask than the configured one does, before the estimate stops being certifiable.
#:
#: THE INVARIANT THIS EXISTS FOR. `MASK_FRACTION_FLOOR` and `MASK_FRACTION_CEILING`
#: are absolute; between them the app was free to report `BaselineHealth.ok` True on
#: an estimate whose per-event depth was 0.6% of the truth. Measured on a renewal
#: process at the owner's own parameters (I0 -34.8 pA, 110 pA blockades, 4.84 pA rms,
#: 60 s at 50 kHz) with only the arrival rate varied: at 18 ms dwells and 35%
#: occupancy the shipped probe proposes ``rough_s`` 0.03 s, the run masks 9.8% of the
#: trace, `reference_mask_fraction` masks 39.1%, and the median measured depth is
#: **7.8 pA against a true 110** -- with every check ok and no warning. The two
#: numbers were both in the app and were compared nowhere.
#:
#: 2.0 -- "the configured window found less than HALF of what a wide one finds" --
#: from the joint measurement of ratio against depth error over 16 (dwell, rate)
#: pairs x 7 rough windows, 2 ms..80 ms dwells and 5.6%..35% occupancy:
#:
#: ====================  ========================================================
#: ratio                 worst per-event depth seen at that ratio
#: ====================  ========================================================
#: <= 1.4 (86 runs)      0.990x  -- and the corpus and the apo file all live here
#: 1.5 .. 2.0 (7 runs)   0.986x
#: 2.0 .. 2.7 (3 runs)   0.950x  (the 5% cases; warned, deliberately)
#: >= 2.9 (12 runs)      **0.0062x** -- every one of them wrong, none of them warned
#: ====================  ========================================================
#:
#: The gap between the worst benign run (2.693 -> 0.972x) and the least-wrong bad one
#: (2.920 -> 0.013x) is where the constant sits. Margin on the OWNER's four
#: recordings, at the parameters he actually uses: ferritin_data2 0.98,
#: ferritin_data3 0.08, the apo ``_0006_`` file 0.99 -- 2.0x clear of the threshold on
#: the tightest of them, and ferritin_data1 has no pass 1 at all.
MASK_REFERENCE_MAX_RATIO = 2.0

#: Rough window of the REFERENCE measurement, seconds -- one number, used by
#: `reference_mask_fraction` here and by `baseline_probe.probe_timescales` (which
#: re-exports it as ``PROBE_ROUGH_S``).
#:
#: It is deliberately ONE constant and not two: the probe's occupancy and the health
#: block's occupancy are the pair whose disagreement is the whole point of
#: `MASK_REFERENCE_MAX_RATIO`, and two numbers that are meant to be comparable must
#: be measured at one scale or the comparison is measuring the difference between two
#: window choices. Wide on purpose: a reference has to see OVER the longest state it
#: is trying to measure, and 5 s clears the apo file's 521 ms clog with 10x to spare.
#: Verified stable -- the apo proposal is 1.96-2.00 s (i.e. snaps to 2.0) across probe
#: decim 500/1000/2000/4000 and closing gaps 0.5/1/2 ms, twelve combinations.
REFERENCE_ROUGH_S = 5.0

#: Shortest reference window worth measuring. Below this the reference cannot see
#: over a state long enough to matter and its answer would be the recording's length
#: rather than the pore's timescale, so `probe_timescales` reports ``usable=False``
#: and the health block reports the cross-check as NOT MEASURED rather than guessing.
REFERENCE_MIN_ROUGH_S = 0.5

#: A recording must be at least this many reference windows long. 4 -- the same
#: 25%-occupancy reasoning the proposal rule itself uses, applied to the reference.
REFERENCE_WINDOWS_MIN = 4.0

#: ``base_s = rough_s / this``: the ratio at which the app considers the two windows a
#: matched pair. The spec's ``rough_s / base_s ~ 3-5``, taken at the middle; it
#: reproduces 0.1 from 0.5 and 0.5 from 2.0 exactly.
#:
#: Lives here rather than in `baseline_probe` -- which re-exports it and is where the
#: proposal RULE that uses it is written -- because `reference_mask_fraction` needs
#: the same ratio to build its reference config, and two spellings of one ratio is
#: how the pair stops being matched.
ROUGH_BASE_RATIO = 4.0

#: Bounds on ``rough_sigma_pA / noise_pA`` -- the rough pass's robust sigma against
#: the finished detrended trace's. Both are the same quantity measured before and
#: after the estimator does its job, so on a healthy recording they agree.
#:
#: Measured (ratio): clean synthetic 0.99, 10%-duty events 0.98, 35% duty 0.96,
#: ferritin_data2 at its shipped parameters 1.18, ferritin_data3 1.34, apo 1.02,
#: 300 pA drift at a 2 s period (still fine) 2.37.
#: Sick: 300 pA drift at a **0.2 s** period 9.65-14.24 (the rough pass is measuring
#: the drift, the 3.5-sigma threshold becomes unreachable and **the mask collapses to
#: exactly 0.0000** while the median baseline error hits 30.6 pA); synthetic gating at
#: ``rough_s`` 0.5 s -> 29.13. And in the OTHER direction, gating at ``rough_s`` 0.2 s
#: -> 0.18 and at 2.0 s -> 0.31, where the base pass cannot follow the conductance
#: steps and the residual is 100-400 pA wide. Hence a two-sided, log-symmetric band.
ROUGH_SIGMA_RATIO_MAX = 3.0
ROUGH_SIGMA_RATIO_MIN = 1.0 / 3.0

#: ``overshoot`` = (samples above +4 sigma) / (samples below -4 sigma) on the
#: canonicalised detrended trace. The ONE published QC scalar that resists a
#: too-short window (a degenerate 0.01/0.002/1 set beats the shipped parameters on
#: level, drift AND noise for every corpus file, but loses here: 0.093 -> 0.178 on
#: data2, 0.099 -> 0.139 on data3). Healthy corpus values are ~0.09-0.11.
#:
#: WHAT IT CANNOT SEE, and the reason `LEVEL_OCCUPANCY_MIN` exists. ``overshoot`` is
#: computed on the CANONICAL frame, and the canonical frame is decided by a polarity
#: that -- everywhere in the app -- is DERIVED from the same trace (`derive_polarity`
#: from a global-median centre; the GUI hands the health block the probe's
#: ``measured_polarity``). Past ~50% occupancy the global median moves onto the
#: BLOCKED level, so the direction that is measured is the direction of the open-pore
#: RETURNS, the canonical frame flips with it, and the flip this scalar exists to
#: detect is applied to the scalar's own frame first. Measured on a pore blocked 81%
#: of a 30 s recording (open 100 pA, blocked 60 pA): the tracked baseline settles at
#: 60.04 pA and ``overshoot`` reads 0.0000 -- a pass. The repo's own 95%-duty
#: inversion fixture reads 2941.7 with ``polarity=-1`` pinned and 0.0003 with the
#: polarity derived, which is what the app does. So this constant guards the regime
#: only when someone else has already settled the direction.
OVERSHOOT_FLAG = 0.3

#: Least of the recording the TRACKED LEVEL may hold before this block stops
#: CERTIFYING that the level it tracked is the open pore -- and says so on the result.
#:
#: THE INVARIANT THIS EXISTS FOR. Every other check reads the trace through the
#: canonical frame, and the frame is chosen by a polarity derived from the same trace
#: (`OVERSHOOT_FLAG` above). On a pore blocked for most of the recording they all flip
#: together and then agree, in green, about a baseline sitting on the BLOCKADE:
#: measured on an 81%-blocked synthetic (open 100 pA, blocked 60 pA, 277 blockades),
#: mask fraction 0.2089 -- inside its band, but it is masking the OPEN RETURNS --
#: ``mask_reference`` 1.00x (both windows agree about the same inverted picture),
#: ``baseline_intrusion`` 1.5e-05, ``overshoot`` 0.0000; eight green rows, a baseline
#: of 60.04 pA, and every event depth reading dI/I0 0.665 against a true 0.400.
#:
#: WHY THIS IS AN ADVISORY ON ITS OWN, AND A FAILING LINE ONLY IN CONJUNCTION --
#: measured, not assumed. A
#: recording that is 21% off-level is arithmetically identical to one that is 79%
#: blocked with its open-pore returns masked: the two differ only in which level you
#: call open, and that is not in the trace. Not nearly-identical -- the SAME PLACE in
#: every statistic this module holds. Side by side, the 81%-blocked synthetic against
#: ``test_baseline_occupancy_reference``'s own renewal fixture at 40 ms / 8 per s (a
#: VALID 24%-occupancy recording whose measured depth is 1.0006x truth, and which that
#: file's load-bearing test requires to stay certified):
#:
#: ========================  ===================  ==================
#: statistic                 81% blocked (wrong)  renewal (right)
#: ========================  ===================  ==================
#: mask fraction             0.2089               0.2483
#: mask vs wide-window mask  1.00x                1.00x
#: state dwell               17.3 ms              40.1 ms
#: base window vs state      29.0x                12.5x
#: baseline intrusion        1.5e-05              -1.0e-07
#: gating spread             0.49                 1.16
#: rough sigma ratio         0.996                1.008
#: overshoot                 0.000                0.000
#: **level occupancy**       **0.791**            **0.752**
#: ========================  ===================  ==================
#:
#: The INVERTED one is the less occupied of the two on every one of them, this
#: statistic included -- so no threshold on any of them condemns the inverted estimate
#: without also condemning the valid one, and a threshold tuned to split them would be
#: fitted to two fixtures rather than to the physics. So this number is not sold as an
#: inversion detector. It is the margin the identification rests on; it is reported on
#: every estimate; and below this constant the RESULT carries
#: `level_occupancy_warning` -- a sentence naming BOTH readings, in the idiom
#: ``mask_fraction_floor`` already uses for the same ambiguity ("TWO readings and they
#: are opposite, so look at the trace before acting"). What FAILS instead is
#: `LEVEL_OCCUPANCY_MAJORITY`, below, `LEVEL_DECISIVE_RATIO` beside it, and -- for the
#: 81%-blocked case in the table above, which clears all three -- this constant
#: CONJOINED with `LEVEL_HOLD_MIN_WINDOWS`: below the floor the block additionally asks
#: whether the pore ever rests at the tracked level for one tracking window, because a
#: level that is only ever voted for is not a level this window measured. That
#: conjunction is what separates the 0.791 row of this table from the 0.752 renewal row
#: under it; neither number does, in either direction.
#:
#: AND THIS CONSTANT IS NOT THE ONLY TRIGGER FOR THAT ADVISORY, because on its own it
#: had the failure mode an advisory must not have. The margin measured here RISES as an
#: inverted estimate gets worse -- 0.791 at 81% blocked, 0.897 at 90%, 0.942 at 95% --
#: so a line drawn here alone caught the shallowest case by 0.9 percentage points and
#: went silent on the three worse ones. `ALT_LEVEL_OCCUPANCY_MIN` is the second trigger
#: and the one that covers the regime; this one still stands because a thin majority is
#: its own reason to say so, and it fires on recordings that have no second level at
#: all (a baseline drifting off a quiet trace).
#:
#: 0.80 -- the same 4:1 reading `REFERENCE_WINDOWS_MIN` uses -- measured at
#: ``INTRUSION_MIN_DEPTH_SIGMA`` (a state has to be deep enough to be a state) and
#: dilated by ``dilate_ms``. Every recording the owner has on disk clears it, and the
#: busiest of them sets the margin:
#:
#: =========================  ==============  =========================================
#: recording                  level occupancy  advisory?
#: =========================  ==============  =========================================
#: apo -500 mV (1.0 / 0.3)    0.906           no
#: apo -600 mV (2.0 / 0.5)    0.899           no
#: apo -700 mV (1.0 / 0.3)    0.883           no
#: apo -1000 mV 0004          0.869           no
#: apo -1000 mV 0000          0.858           no -- the tightest real margin
#: **81% blocked synthetic**  **0.791**       **yes**
#: renewal 40 ms / 8 per s    0.752           yes, on a correctly measured estimate:
#:                                            the advisory is about what is not KNOWN,
#:                                            not about what was measured
#: =========================  ==============  =========================================
LEVEL_OCCUPANCY_MIN = 0.80

#: Level occupancy below which the ``level_occupancy`` check FAILS rather than merely
#: reporting: below half, the rule the estimator applies -- "the open pore is the level
#: the pore is at most of the time" -- has no majority left to apply, and the baseline
#: is not a level of this recording at all. Unlike `LEVEL_OCCUPANCY_MIN` it needs no
#: corpus margin, because it is not a judgement about a pore.
#:
#: WHAT IT DOES AND DOES NOT REACH, measured, because a threshold that cannot fire is
#: the defect this whole row exists to answer. The scale is the QC MAD noise of
#: ``current - baseline``, and that noise is measured on the SAME residual, so a
#: baseline whose error is a SPREAD inflates its own yardstick and can never trip this:
#: a 50/50 two-level pore judged with a 1 s tracking window puts the baseline squarely
#: between its levels, 100 pA from every sample, and still reads 1.0000 here (the MAD
#: is 68.6 pA). What it does reach is a baseline that is systematically OFFSET from a
#: quiet trace -- the case `baseline_health`'s signature allows and nothing else in the
#: block looks for, since it accepts any baseline array: one restored from, or carried
#: over from, a different recording. A -300 pA baseline over a -100 pA trace reads
#: 0.0000 and warns. So: reachable through this function's own contract, not through
#: any output of the estimator measured here -- and `LEVEL_OCCUPANCY_MIN`'s docstring
#: is where the reason a tighter failing line would be wrong is written out.
LEVEL_OCCUPANCY_MAJORITY = 0.50

#: Least of the recording the OTHER level of the amplitude histogram has to hold
#: before `alternative_level` calls it a LEVEL -- and therefore before the result says
#: the open-pore identification was a CHOICE between two of them.
#:
#: WHY A SECOND OCCUPANCY HAD TO EXIST. `LEVEL_OCCUPANCY_MIN` keys its advisory to how
#: much of the recording the TRACKED level holds, and on an inverted estimate that
#: number RISES as the inversion gets worse. Measured on one fixture (open pore 100 pA,
#: blocked 60 pA, exponential dwells) with only the mean BLOCKED dwell moved:
#:
#: ==================  ==========  =================  ==============  ==============
#: mean blocked dwell  pore is     tracked baseline   tracked level   OTHER level
#: ==================  ==========  =================  ==============  ==============
#: 80 ms               81% blocked 60.04 pA (wrong)   0.791           0.191
#: 200 ms              90% blocked 60.02 pA (wrong)   0.897           0.094
#: 400 ms              95% blocked 60.0 pA  (wrong)   0.942           0.054
#: 900 ms              97% blocked 60.0 pA  (wrong)   0.970           0.028
#: ==================  ==========  =================  ==============  ==============
#:
#: Only the first row is under `LEVEL_OCCUPANCY_MIN`, and it clears it by 0.9
#: percentage points. A line drawn on the tracked level alone therefore goes SILENT
#: exactly as the estimate gets worse, which is the opposite of what an advisory is
#: for. The other level is there in all four rows, and this is the reading that says
#: so.
#:
#: WHAT THIS DOES **NOT** CLAIM. It is not an inversion detector and nothing here
#: pretends it is -- see `LEVEL_OCCUPANCY_MIN` for the measurement that killed that
#: hope, and `alternative_level` for the four candidate discriminators that were tried
#: against the owner's own recordings and died. A second level is what makes the
#: question ASKABLE, not what answers it. So the advisory fires on the owner's four
#: ABFs too (their other level holds 0.054 / 0.099 / 0.087 / 0.077 of the recording),
#: and that is the honest outcome rather than a false alarm: on every one of them the
#: block picked the open pore by majority occupancy and verified nothing, exactly as it
#: did on the inverted fixture. An advisory that fired only on the synthetic would be
#: claiming a discrimination this module does not have.
#:
#: 0.025 is `MASK_FRACTION_FLOOR`'s reading of the same question ("did this find a
#: population, or a dilated noise skirt?") applied to the other level instead of to the
#: mask, and it is used for that reason rather than tuned: below it the alternative is
#: a handful of excursions, not a level a pore could sit in. It does put a floor under
#: the regime -- a pore blocked 99% of the time with 20 ms returns holds 0.010 there
#: and says nothing -- which is stated rather than hidden, because at that point the
#: only thing left in the trace is the operator's knowledge of the open pore.
ALT_LEVEL_OCCUPANCY_MIN = 0.025

#: Occupancy of the OTHER level, as a fraction of the tracked level's, at or above
#: which the ``level_occupancy`` check FAILS: the estimator's majority rule ran, but
#: not on a majority worth the name.
#:
#: THE SECOND HONEST FAILING LINE, and the one that reaches the regime
#: `OVERSHOOT_FLAG`'s own message names ("occupancy past 50%, where the estimator
#: inverts rather than degrades"). `LEVEL_OCCUPANCY_MAJORITY` fails only where the
#: baseline is at NO level of the recording; this one fails where the recording has two
#: and the rule that chose between them barely preferred one. Because the two bands are
#: disjoint by construction, ``alt >= 0.5 * tracked`` can only happen when the tracked
#: level holds less than two thirds of the recording, so the whole of this line lives
#: between 0.50 and 0.667 of tracked occupancy -- a strip just above
#: `LEVEL_OCCUPANCY_MAJORITY`, not a re-litigation of the certified band.
#:
#: MEASURED, both sides. It fails on a pore blocked 41% of the time (tracked 0.605,
#: other 0.349, ratio 0.58) and on the two-level 50/50 pore that reads 0.596 tracked
#: and passed everything before this line existed. It cannot reach the estimates the
#: rest of the suite requires to stay certified: over
#: ``test_baseline_occupancy_reference``'s own 7 x 4 sweep the certified rows run
#: 0.119 to 0.317, and the owner's four ABFs run 0.060 to 0.114 -- a margin of 0.18 to
#: the nearest one. It is a ratio rather than a second absolute line because what it is
#: about is the DECISIVENESS of a comparison between two numbers, and that is not a
#: property either of them has alone.
LEVEL_DECISIVE_RATIO = 0.50

#: How long the recording has to HOLD the tracked level, in tracking windows, before a
#: majority under `LEVEL_OCCUPANCY_MIN` may still be certified. Measured as the p99 of
#: the unbroken runs the trace spends within `INTRUSION_MIN_DEPTH_SIGMA` of the tracked
#: baseline -- the same p99-of-runs reading `_mask_run_p99_s` uses for the other side --
#: divided by ``config.tracking_window_s``.
#:
#: THE THIRD FAILING LINE, AND WHY IT IS A CONJUNCTION. `LEVEL_OCCUPANCY_MIN` is the
#: floor every recording in evidence clears, and on its own it CANNOT be a failing line:
#: the 81%-blocked synthetic reads 0.791 there against 0.752 and 0.770 for two
#: correctly-measured renewal estimates that ``test_baseline_occupancy_reference`` and
#: ``test_s3_detcore_level_occupancy`` require to stay certified. The inverted estimate
#: is the MORE occupied of the three. So below the floor the block asks a second
#: question, about the recording rather than about the margin: does the pore ever REST
#: at the level being tracked for as long as the window that measured it? A level the
#: trace never holds for one window was voted for across many visits to two levels; a
#: level it does hold was, for at least part of the recording, measured on its own.
#:
#: ==============================================  ======  =======  ==========
#: estimate                                        level   p99 hold  certified?
#: ==============================================  ======  =======  ==========
#: 81% blocked, probed 2.0/0.5 (INVERTED)          0.791   0.79 w   **no**
#: renewal 8/s 40 ms @ 2.0/0.5 (valid, pinned)     0.752   1.06 w   yes
#: renewal 8/s 40 ms @ 0.5/0.125 (valid, pinned)   0.770   3.53 w   yes
#: apo -600 mV @ 2.0/0.5 (the owner's own file)    0.899   0.88 w   yes -- above the floor
#: apo -1000 mV 0004 @ 2.0/0.5                     0.869   0.93 w   yes -- above the floor
#: ferritin_data1 / data2 / data3                  0.849+  18-26 w  yes
#: ==============================================  ======  =======  ==========
#:
#: WHAT THIS DOES NOT REACH, stated because a guard read as wider than it is, is worse
#: than none. It is NOT an inversion detector and it does not become one by conjunction:
#: `alternative_level` lists the four discriminators that were measured against the
#: owner's recordings and died, and the two occupancies still put the inverted estimate
#: and a valid one in the same place. Swept over a FAMILY of clogged pores (mean open
#: 15-30 ms, mean blocked 60-120 ms, two seeds) at each one's own probed window, 13 of
#: 18 rungs are certified today and this line condemns ONE of them -- the dossier's own
#: 81% case. The other 12 escape for two reasons that are not defects in this line:
#: seven sit ABOVE the floor (the tracked level's occupancy RISES with the blockade
#: fraction, which is `ALT_LEVEL_OCCUPANCY_MIN`'s subject), and five got a SHORTER
#: probed window, which their open returns do hold. The same recording measured with a
#: narrower ``base_s`` is therefore not condemned here, and that is honest rather than
#: hidden: this line is about what the window could see, and a narrower window sees the
#: level. The advisory (`level_occupancy_warning`) is what covers the regime; this is
#: only the part of it that can be checked.
#:
#: WHAT IT COSTS, on the other side of the same sweep. Over a family of VALID renewal
#: recordings (rate 3-12/s, dwell 18-80 ms, two seeds, each at its own probed window)
#: 18 of 24 are certified today and this line takes the certificate off ONE of them: a
#: pore off-level 46% of the time whose tracked level holds 0.656 and which never rests
#: there for a window. That is a correctly measured estimate and it loses ``ok`` -- the
#: intended trade, not an oversight. At that occupancy the block is choosing between two
#: levels the recording spends comparable time at, and it has no way to check which is
#: the open pore; the row says exactly that. Every recording the owner has on disk, and
#: both valid estimates the suite pins, are above the floor or hold their level, so no
#: ordinary good recording is reached.
#:
#: 1.0 is not tuned -- it is one window, the same "recording vs its own window" reading
#: `length_tracking` and `_require_length` are built on. The margins it carries are
#: unequal and the tight one is named above: the pinned valid estimate clears it by 6%
#: (1.056), and the inverted one fails it by 21% (0.788). A quantile other than p99
#: would not separate them at all (p95 condemns both, p99.9 neither), so this line is
#: measured at ONE point of a run-length distribution and nothing wider should be read
#: into it.
LEVEL_HOLD_MIN_WINDOWS = 1.0

#: Block length for the conductance-LEVEL spread statistic, seconds (spec 2.5's
#: "20 ms block medians").
#:
#: Lives HERE rather than in `baseline_probe` -- which re-exports it and owns the
#: proposal rule built on it -- for the same reason `REFERENCE_ROUGH_S` does: the
#: proposal decides whether the long states are events or levels, and so must
#: `baseline_health`, or the app proposes a window on one reading of the recording
#: and certifies it on another.
GATING_BLOCK_S = 0.020

#: `gating_spread` above this means the pore steps between conductance states, and
#: the meaning of every long state INVERTS: they are BASELINE LEVELS the tracking
#: window is *supposed* to follow, not events it must sail over. `baseline_probe`
#: inverts its proposal rule on this; `state_dwell` and `baseline_intrusion` decline
#: to judge on it, because on a stepping pore both would be measuring the estimator
#: doing its job.
#:
#: Measured: ferritin_data1 1.69, apo 0.52, ferritin_data2 4.73, **ferritin_data3
#: 13.79**. 8.0 sits 1.7x above the highest non-gating file and 1.7x below data3 --
#: the widest available margin, and inside the spec's own "any threshold in 8-15".
GATING_THRESHOLD = 8.0

#: Subsampling stride for the two robust-sigma measurements that run over whole
#: traces (`baseline_probe`'s residual sigma and `gating_spread`'s within-block
#: noise). A MAD over 3 M samples and over 12 M agree to well past the digit either
#: is thresholded at; the stride is why the probe finishes in seconds on the 12.5 M
#: -500 mV recording.
SIGMA_STRIDE = 4
SIGMA_STRIDE_MIN_N = 2_000_000

#: Fewest whole `GATING_BLOCK_S` blocks a gating spread can be measured from. Below
#: this an IQR is being taken over three numbers and it is not a distribution.
GATING_BLOCKS_MIN = 4

#: ``base_s`` must be at least this many times the median state dwell measured over
#: the REFERENCE window (`state_dwell_s`). Below it the tracking median rides into
#: the very states it exists to ignore.
#:
#: THE FAILURE THIS EXISTS FOR. On the -700 mV apo recording the shipped probe
#: measured a longest state of 31 ms and proposed ``base_s`` = 0.03 s -- a window the
#: same size as the states -- and **all six of the checks above passed in green**.
#: They compare the window against the RECORDING length (6049x, 1815x -- both vast),
#: against mask-fraction constants, against a wide-window mask, and against the rough
#: pass's own sigma. Not one of them looks at the EVENT timescale, so a window 1.4x
#: the dwell certified exactly as cleanly as one 24x it, and the file's median dI/I0
#: came out 1.90 against 2.36 and 2.57 for its two siblings -- an "anomaly" that was
#: the window.
#:
#: 3.0, from the response measured on his three recordings. ``base_s`` alone moves;
#: ``rough_s`` is held at 2.0 s so the mask, the spans and the reference depth are
#: identical in every row:
#:
#: ==========  =====  =====  =====  =====  =====  =====
#: ``base_s``  -700 mV       -600 mV       -500 mV
#:             ratio  depth  ratio  depth  ratio  depth
#: ==========  =====  =====  =====  =====  =====  =====
#: 0.03 s      1.38   .973   1.45   .977   1.24   .961
#: 0.05 s      2.30   .981   2.41   .978   2.06   .964
#: 0.07 s      3.22   .987   3.38   .987   2.89   .972
#: 0.10 s      4.61   .997   4.83   .995   4.12   .980
#: 0.30 s      13.8   1.001  --     --     12.4   .995
#: ==========  =====  =====  =====  =====  =====  =====
#:
#: ("depth" is the median per-state depth as a fraction of the same recording's depth
#: at ``base_s`` = 0.5 s.) The knee is between 2.4x and 4.1x and 3.0 sits in it. The
#: bound is two-sided and both sides are real recordings, not headroom: it must fire
#: at 1.38x, and it must stay silent at 4.61x and 4.83x -- his -700 mV and -600 mV
#: files at 0.5/0.1, which measure depths to 0.3-0.5% -- and at the 4.5x of
#: ``test_masked_baseline.py``'s healthy fixture. **The report's suggested ">= 10x" is
#: not supportable**: it would warn on all three of those, and a check that cries wolf
#: on correct settings is one the owner learns to click past.
#:
#: ferritin_data3 reads **2.99x** at its known-good 0.03/0.01 and is held out by the
#: gating judgement rather than by this constant, which is 0.3% away. That is not
#: slack to be tuned away -- on a stepping pore a short window is CORRECT -- but it
#: does make `GATING_THRESHOLD` load-bearing here, which is why
#: ``test_baseline_state_timescale.py`` pins it on data3 specifically.
BASE_STATE_RATIO_MIN = 3.0

#: How far the tracked baseline may sit inside the states, as a fraction of their
#: own depth, before the estimate stops being certifiable -- the DIRECT form of what
#: `BASE_STATE_RATIO_MIN` tests on a proxy, and the measurement that actually caught
#: the -700 mV file (report SUMMARY.md 4.1: a stamped -51.661 pA against an
#: independently reconstructed open pore of -48.218).
#:
#: It is a TOLERANCE, not a discriminator: `tracked_intrusion` measures the fraction
#: of each state's excursion the baseline followed, and that is very nearly the
#: fraction by which the depth reads shallow -- on the -700 mV recording, ``base_s``
#: swept with ``rough_s`` held at 2.0 s, an intrusion of 0.040 goes with a depth 2.7%
#: low, 0.028 with 1.9%, 0.016 with 1.3% and 0.004 with 0.3%. So 0.03 says "certify
#: nothing whose depths are more than about three percent eaten", and what has to be
#: checked is only that no CORRECT configuration reads that high. None comes close:
#: at the windows in evidence his apo recordings read 0.004 / 0.003 / -0.000 / 0.000 and
#: ferritin_data2 reads 0.004, a factor of 7 clear, against 0.040 for the proposed
#: -700 mV window and 0.484 for a 0.01 s window on -600 mV.
#:
#: Kept as a SEPARATE check from the ratio rather than folded into it, because each
#: catches cases the other does not, both measured: ferritin_data2 at ``base_s``
#: 0.02 s scores a comfortable 7.7x on the ratio and is intruded 0.064 (its states
#: have a long tail the median does not see); the -700 mV file at 0.05 s scores 2.3x
#: on the ratio and only 0.028 here. The ratio is also the one a user can ACT on --
#: it says how far to widen -- while this one says what it is costing him.
BASELINE_INTRUSION_MAX = 0.03

#: Half-width of the flanking windows the intrusion reference is taken from, and the
#: gap left between the state and its flanks. 50 ms / 5 ms follows the report's own
#: reconstruction (the median of the event-free samples 50-10 ms before each event).
#: Taken from the BASELINE, never from the trace: the mask is ONE-SIDED, so the
#: median of the surviving samples carries a truncation bias -- measured at +4.3 pA
#: on ferritin_data2, which would have read as four times the intrusion actually
#: there.
INTRUSION_FLANK_S = 0.05
INTRUSION_PAD_S = 0.005

#: A state must be at least this many robust sigmas deep to contribute to the
#: intrusion fraction. The quantity is a RATIO of two measured depths, so a state
#: whose own depth is a couple of sigma contributes a ratio of two numbers that are
#: mostly noise -- and the mask, being one-sided and dilated, manufactures those
#: continuously (a fifth of the runs on the -700 mV recording are shallower than
#: 6 pA against a 94 pA population). Measured effect of the filter on the reading:
#: ferritin_data2 **0.0219 -> 0.0040**, the -700 mV proposal **0.0436 -> 0.0416**.
#: It cuts the quiet side by 5x and the loud side by 5%, which is what a filter that
#: removes noise rather than signal looks like -- and it is the difference between a
#: 1.4x margin on `BASELINE_INTRUSION_MAX` and a 7x one.
INTRUSION_MIN_DEPTH_SIGMA = 5.0

#: Most states the intrusion measurement will visit, evenly spaced through the
#: recording. It is a Python loop over states (a median of ~2500 samples x3 each), so
#: the cost is set by the state COUNT, not the sample count: 1193 states on the 9 M
#: -700 mV recording cost 31 ms. A pathological mask can produce hundreds of
#: thousands of runs, hence the cap; the quantity is a median over states, and a
#: median over 2000 of them is the same number.
INTRUSION_MAX_STATES = 2000

#: How far the tracked baseline may sit from the trace's own level across the FIRST
#: or LAST tracking window of a recording, in units of the detrended noise, before
#: the ``edge_tracking`` row says anything at all. Below this the boundary is
#: indistinguishable from the interior and there is nothing to report.
#:
#: THE HOLE THIS ROW EXISTS FOR. Every other check reads the recording as a whole,
#: and the one place the estimator has no data to read is its own two ends: pass 1
#: filters with ``mode='reflect'`` (`ml_preprocess._decimated_median`), so at a
#: boundary its median is biased by the trace's own SLOPE over ~``rough_s``/4. When
#: that bias reaches ``mask_sigma`` x the rough sigma and points the blockade way,
#: the one-sided `_rough_pass` masks the whole head as one giant state, pass 2
#: BRIDGES it, and the baseline is left pinned at a level the trace only reaches
#: later. Measured on an event-free trace of noise + open-pore wander (4 s, 100 kHz,
#: 5 pA noise, 1000 - 100t + 250 sin(pi t)) at the app default 0.5 / 0.1 s: the
#: baseline is **+69.5 pA wrong over the first 10 ms** (13.9x the noise floor) and
#: +39.0 pA over the first 100 ms, against 1.23 pA mid-trace -- and `detect_events`
#: shipped **18 events, 17 of them accepted**, every one inside the first 83 ms, on a
#: trace with none. All nine health checks read ``ok``. The same samples with 2 s of
#: the same wander in front of them are 0.67 pA out with no events there at all, so
#: the artefact belongs to the BOUNDARY, not to the samples.
#:
#: 3.0 is where the corpus sits: measured over the tracking window at the parameters
#: each file is analysed with, the owner's recordings read 0.18 sigma (-700 mV),
#: 0.28 sigma (-500 mV), 0.28 sigma (-600 mV at its proposed 2.0 / 0.5) and 0.11
#: sigma (the -1000 mV cut), against the 7.15 sigma of the fixture above.
EDGE_OFFSET_SIGMA = 3.0

#: The other half of the same row, and the half that keeps it off honest recordings:
#: a boundary offset is only reported when it is no larger than the drift THIS
#: baseline demonstrably follows over one tracking window elsewhere in the recording
#: (p95 of ``|b(t + base_s) - b(t)|`` over the interior). Larger than that and the
#: boundary holds a STEP the drift cannot explain -- a real blockade, a settling
#: transient, a voltage step -- which is signal the estimator is right to bridge.
#:
#: The two populations separate by ~9x at this line, measured:
#:
#: ==========================================  ========  =========  ======
#: trace (at the parameters it is analysed at)   offset   drift p95   ratio
#: ==========================================  ========  =========  ======
#: event-free wander, 0.5 / 0.1 s (the defect)  38.6 pA    87.8 pA    0.44
#: -600 mV at the DEFAULT 0.5 / 0.1 s           24.4 pA     3.1 pA    8.00
#: flat synthetic + a real 90 ms state at t=0  101.1 pA     2.0 pA   50.65
#: wander + a real 300 pA state at t=0         334.5 pA    87.0 pA    3.84
#: ==========================================  ========  =========  ======
#:
#: The second row is the one that matters: the owner's -600 mV file really does open
#: ~23 pA below its open pore for its first 70 ms (raw 10 ms medians -54, -56, -55,
#: -56, -53, -51, -54, then -32 pA), the baseline bridges that correctly, and this
#: row must stay green there -- it does, by 8x. What the check cannot do is tell a
#: genuine state at the boundary from unfollowed drift when the two are the same
#: size; that is why the drift comparison IS the criterion and why the outcome is a
#: warning that names both readings rather than a correction.
EDGE_DRIFT_RATIO = 1.0


class BaselineWarning(UserWarning):
    """Raised when a baseline estimate is at risk of tracking the events."""


class ConstantBaselineWarning(BaselineWarning):
    """The requested window degenerates to a single block: the estimate is effectively
    CONSTANT and cannot track drift, so a detrend built on it is a no-op.

    A SUBCLASS of `BaselineWarning`, not a sibling, so every existing
    ``pytest.warns(BaselineWarning)`` and filter rule keeps matching. The subtype
    exists for callers that must tell "this stage did nothing" apart from the softer
    at-risk advisories WITHOUT parsing prose — `modules.DetectSchmitt.run` keys
    ``detrend_effective`` in provenance on it."""


class AmplifiedWanderWarning(BaselineWarning):
    """`remove_baseline_wander`'s own output carries MORE baseline wander than its
    input: the stage made the very thing it exists to remove WORSE.

    The mechanism is the detrend's implied response, ``H(f) = 1 - M(f)`` where ``M``
    is the moving block-median. For wander whose period sits between roughly half the
    window and the window, ``M(f)`` is phase-INVERTED (the same sign flip a moving
    average's sinc response has on its first side lobe), so subtracting the estimate
    ADDS wander instead of removing it — measured, 80 pA of 3 Hz wander became a
    110.6 pA peak (+38%) under the default 0.5 s window, with the trace's events and
    depths otherwise untouched. Nothing downstream can see this from the data (the
    trace simply wanders more than it did), provenance stamps ``detrend_enabled=True``
    and ``detrend_effective=True``, and the docstring hazard covered only the
    too-SHORT-window failure (events subtracted) — so the kernel measures its own
    output against its own input and says so here, or nowhere.

    A SUBCLASS of `BaselineWarning` for the same reason `ConstantBaselineWarning` is:
    filters and ``pytest.warns`` keep matching, and `modules.DetectSchmitt.run` can
    stamp ``detrend_amplified`` in provenance without parsing prose."""


#: Thread-local carrier for `collect_baseline_warnings`. ``warnings.catch_warnings``
#: could capture the same messages, but it save/mutates the PROCESS-global filter
#: list — the exact cross-thread race ``stats.quiet_runtime_warnings`` exists to
#: fence — and the GUI recomputes the filter preview on its thread while a detection
#: worker may be inside this module on another. A thread-local list races nothing.
_warn_sink = threading.local()


@contextmanager
def collect_baseline_warnings():
    """Collect the `BaselineWarning` instances this module emits on THIS thread.

    Yields a list that receives every warning raised through `_warn_baseline` inside
    the block (the ``warnings.warn`` call still happens too, so filters and
    ``pytest.warns`` see them unchanged). Callers use it to route a kernel's own
    advisory onto a user surface — the preview error line, ``res.warnings`` — which
    the stdlib warning machinery cannot reach: a warning that dies in a filter while
    the stage silently does nothing is precisely the failure this exists to prevent.

    Re-entrant: an inner block captures its own slice, and on exit those warnings are
    appended to the outer block's list as well, so nesting never swallows one."""
    outer = getattr(_warn_sink, "sink", None)
    sink: list[BaselineWarning] = []
    _warn_sink.sink = sink
    try:
        yield sink
    finally:
        _warn_sink.sink = outer
        if outer is not None:
            outer.extend(sink)


def _warn_baseline(
    message: str,
    *,
    category: type[BaselineWarning] = BaselineWarning,
    stacklevel: int = 2,
) -> None:
    """``warnings.warn`` plus the thread-local sink — the ONE way this module warns.

    ``stacklevel`` is counted from the CALLER's frame (as if the caller had called
    ``warnings.warn`` itself); the +1 compensates for this wrapper."""
    sink = getattr(_warn_sink, "sink", None)
    if sink is not None:
        sink.append(category(message))
    warnings.warn(message, category, stacklevel=stacklevel + 1)


#: Thread-local carrier for `baseline_progress_scope` — the same pattern as
#: `_warn_sink`, and for the same reason: the GUI's detection worker is on another
#: thread than anything else touching this module, and a process-global would race.
_progress_sink = threading.local()


@contextmanager
def baseline_progress_scope(callback):
    """Route this module's coarse progress to ``callback(fraction, message)``.

    THE LIVENESS CHANNEL FOR THE ONE SLOW THING THIS MODULE DOES. On a whole-channel
    detection of the owner's 606 MB ATP recording, `characterisation_baseline` is
    ~86% of a ~30 s `detect_events` call [MEASURED on a 50 M-sample trace: door
    20.5 s of run 23.9 s -- polarity ladder 5.3 s, kernel estimate 4.6 s, health
    block 7.8 s], and none of it could say it was alive: `modules.DetectSchmitt.run`
    reported progress only at 0.0 and 1.0, so the status-bar progress bar sat at 0%
    for the entire run -- indistinguishable from a stall (GA5-01).

    The cancel machinery already solved the reach problem this has: an AMBIENT,
    thread-local carrier that the worker installs and the kernels poll, so no
    signature between `DetectSchmitt` and the door has to grow a callback parameter
    it would then have to forward through `detect_events` (which this module does not
    own). Progress rides the SAME stage boundaries the cancel poll already walks:
    every `_report_baseline_progress` call sits beside a `check_cancelled`.

    ``fraction`` is monotone non-decreasing in [0, 1] over the scope's lifetime (the
    carrier clamps and drops out-of-order reports, so a conditional stage that never
    runs leaves a gap, never a backwards jump). ``message`` names the stage in words
    fit for a status bar. With no scope installed every report is a no-op returning
    immediately -- the exact discipline `check_cancelled` follows -- so non-worker
    callers (row rebuilds, probes, tests) pay nothing.

    Reports are delivered on the calling thread; a GUI callback must therefore be
    thread-safe (the analysis worker's ``report_progress`` emits a queued Qt signal,
    which is).
    """
    prev = getattr(_progress_sink, "state", None)
    _progress_sink.state = {"cb": callback, "lo": 0.0, "hi": 1.0, "last": -1.0}
    try:
        yield
    finally:
        _progress_sink.state = prev


@contextmanager
def _progress_span(lo: float, hi: float):
    """Rescale nested `_report_baseline_progress` calls into ``[lo, hi]`` of the
    CURRENT span, so each stage reports its own local 0..1 and only its caller knows
    what share of the whole it is. The shares are the measured cost split (see
    `baseline_progress_scope`), not guesses. No-op when no scope is installed."""
    st = getattr(_progress_sink, "state", None)
    if st is None:
        yield
        return
    outer_lo, outer_hi = st["lo"], st["hi"]
    width = outer_hi - outer_lo
    st["lo"] = outer_lo + float(lo) * width
    st["hi"] = outer_lo + float(hi) * width
    try:
        yield
    finally:
        st["lo"], st["hi"] = outer_lo, outer_hi


def _report_baseline_progress(fraction: float, message: str) -> None:
    """Report ``fraction`` (local to the current span) if a scope is installed.

    Monotone by construction: a report that would move the delivered fraction
    backwards is dropped, so callers state their ladder positions honestly and a
    skipped conditional stage cannot make the bar retreat."""
    st = getattr(_progress_sink, "state", None)
    if st is None:
        return
    f = st["lo"] + min(max(float(fraction), 0.0), 1.0) * (st["hi"] - st["lo"])
    if f <= st["last"]:
        return
    st["last"] = f
    st["cb"](f, message)


def _nonfinite_reason(finite: np.ndarray) -> str:
    """Describe a failed finiteness mask -- dead channel vs localised taint.

    Both guards in this module count from a mask they have already computed, so this
    costs nothing on the path that matters (the clean one, which never calls it) and
    the two refusals cannot drift apart in wording. The distinction itself lives in
    ``core.types.nonfinite_reason``, which every non-finite refusal in the app shares;
    imported HERE rather than at module scope so this signal-first module keeps a
    zero-cost import on the happy path.
    """
    from ..core.types import nonfinite_reason

    n_bad = int(finite.size - np.count_nonzero(finite))
    bad = np.flatnonzero(~finite)
    return nonfinite_reason(n_bad, int(finite.size), int(bad[0]) if bad.size else -1)


def require_finite(signal: np.ndarray, *, stage: str = "estimate a baseline") -> np.ndarray:
    """Refuse a non-finite trace, naming the first bad sample. **Call this BEFORE the notch.**

    A NaN never reaches a baseline estimator as a *localised* defect in the real
    chain: ``filtfilt`` propagates it over the whole array, so one NaN in 200 000
    samples makes all 200 000 non-finite and the refusal happens for the right
    reason. It is when a NaN *does* reach the masked-median kernel -- from an
    unfiltered path, or a notch that was switched off -- that the failure turns
    silent: ``noise`` becomes NaN, ``rough < median(rough) - 3.5*noise`` is then
    all-False, **the entire event mask is disabled**, and the estimator returns a
    perfectly finite baseline that is 12 pA wrong. Nothing raises and nothing warns.

    (+-inf is harmless by comparison -- it sorts correctly inside a median, and the
    measured mask fraction moves from 0.0251 to 0.0256 -- but it is refused too,
    because a baseline fitted through an infinity is not a number anyone should act
    on.)

    Returns ``signal`` unchanged so this can be used inline.
    """
    x = np.asarray(signal)
    if x.size == 0:
        raise ValueError(f"cannot {stage} from an empty recording")
    finite = np.isfinite(x)
    if not finite.all():
        raise ValueError(
            f"cannot {stage}: {_nonfinite_reason(finite)} A baseline fitted through "
            "NaN/Inf is meaningless -- and a NaN reaching the masked-median kernel "
            "silently DISABLES its event mask and returns a finite, wrong baseline."
        )
    return x


# ---------------------------------------------------------------------------
# the unified estimator configuration -- ONE owner for the default
# ---------------------------------------------------------------------------


def derive_decim(base_s: float, fs: float, *, taps_target: int = TAPS_TARGET) -> int:
    """Decimation stride for a ``base_s``-second window, from a tap-count target.

    ``decim`` is not a scientific knob, it is a cost throttle: the decimated rolling
    median costs ``O(n * base_s * fs / decim**2)``, so doubling it is a 4x speedup,
    and the answer barely moves (decim 200 vs 50 on the real apo recording: median
    stamped baseline error 0.91 pA vs 0.80 pA, fraction of events off by > 10 pA
    0.0000 either way). So the app exposes the two WINDOWS in seconds and derives
    this, clamped so the filter never runs on fewer than `MIN_TAPS` taps.

    A pinned ``BaselineConfig.decim`` always wins over this -- that is what keeps the
    corpus sets (50, 50, 10) verbatim and the bit-exactness contract intact.
    """
    base_s = float(base_s)
    fs = float(fs)
    if not (np.isfinite(base_s) and base_s > 0):
        raise ValueError(f"base_s must be a positive number of seconds, got {base_s!r}")
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError(f"fs must be a positive number of Hz, got {fs!r}")
    hi = int(math.floor(base_s * fs / MIN_TAPS))
    if hi < 1:
        raise ValueError(
            f"a {base_s * 1e3:g} ms window at {fs:g} Hz is only {base_s * fs:.1f} samples; "
            f"a decimated median needs at least {MIN_TAPS} taps. Widen the window."
        )
    return int(min(max(int(math.floor(base_s * fs / int(taps_target))), 1), hi))


def _validate_masked_params(
    *, method: str, rough_s: float, base_s: float, decim: int | None,
    mask_sigma: float, dilate_ms: float, despike_size: int,
) -> None:
    """The masked/robust parameter contract, in ONE place.

    Both entry points -- `BaselineConfig` construction and a bare
    `estimate_baseline(method='masked_median', ...)` call -- pass through here, so a
    caller cannot reach the kernel with a parameter set that only one of them
    polices. (The reference implementation validates none of this and returns a
    silently wrong answer for every one of these.)
    """
    if not (np.isfinite(rough_s) and rough_s > 0):
        raise ValueError(f"rough_s must be a positive number of seconds, got {rough_s!r}")
    if not (np.isfinite(base_s) and base_s > 0):
        raise ValueError(f"base_s must be a positive number of seconds, got {base_s!r}")
    if method == "masked_median" and base_s > rough_s:
        raise ValueError(
            f"base_s={base_s:g}s must not exceed rough_s={rough_s:g}s: the rough pass "
            "exists to expose events that the short pass must not follow, so a short "
            "pass at least as wide as the rough one masks nothing and the two-pass "
            "structure is inoperative."
        )
    # The fs-dependent half of the decim contract (>= MIN_TAPS taps) lives in
    # ``BaselineConfig.resolved_decim``, which is the only place that knows the rate.
    if decim is not None and int(decim) < 1:
        raise ValueError(f"decim must be >= 1, got {decim!r}")
    if not (np.isfinite(mask_sigma) and mask_sigma > 0):
        raise ValueError(
            f"mask_sigma must be a positive number of robust sigmas, got {mask_sigma!r}; "
            "at <= 0 the mask swallows at least half the trace."
        )
    if mask_sigma > 10.0:
        raise ValueError(
            f"mask_sigma={mask_sigma:g} is beyond any real event threshold (> 10 sigma); "
            "nothing would ever be masked and this silently degrades to a plain "
            "short-window median."
        )
    if not (np.isfinite(dilate_ms) and dilate_ms >= 0):
        raise ValueError(f"dilate_ms must be >= 0 ms, got {dilate_ms!r}")
    size = int(despike_size)
    if size and (size < 2 or size > 9):
        raise ValueError(
            f"despike_size must be 0 (off) or 2-9 samples, got {despike_size!r}; "
            "anything wider starts eating the shortest real events."
        )


@dataclass(frozen=True)
class BaselineConfig:
    """Every knob of the open-pore (I0) estimator, as one auditable value.

    THIS IS THE SINGLE OWNER OF THE DEFAULT. The defect this class exists to remove
    is that the app measured three different baselines for one recording -- a 10 ms
    block median in the Schmitt detector, a 200 ms one in the ML characteriser, and
    a +-3 ms flank median in re-characterisation -- so the same event reported a
    different ``depth_pA`` depending on which detector found it (population median
    23.23 pA vs 111.31 pA). One config object flows from the GUI through detection,
    characterisation, ML characterisation and re-characterisation; no other module
    may carry a baseline window constant.

    Fields:
        method: One of `METHODS`. ``'masked_median'`` is the estimator this app now
            characterises with; the block-based methods remain reachable and
            bit-identical for stored results and the detrending stage.
        rough_s: Pass-1 window, seconds. Sets **which excursions become "events"**.
            Must be >= ~4x the longest state you want rejected -- the longest state
            has to occupy <~25% of it, or the rough median moves *into* that state
            and nothing gets masked. (Measured on the apo file, whose 521 ms partial
            clog is covered 15.1% at ``rough_s=0.5`` and 99.96% at 2.0.)
        base_s: Pass-2 window, seconds. Sets **how fast the baseline may track**.
            Must be SHORTER than the shortest genuine conductance level dwell, or a
            gating pore's steps are smoothed into the baseline.
        decim: Shared stride for both passes. ``None`` derives it from ``base_s``
            via `derive_decim`; an int pins it, which is what the corpus parameter
            sets do (50, 50, 10) and what keeps them bit-exact.
        mask_sigma: Mask threshold in robust sigmas below ``median(rough)``.
        dilate_ms: Mask growth on each side, milliseconds.
        window_s: The window for ``'robust_median'`` / ``'block_median'`` /
            ``'percentile'`` / ``'asls'``. NOT used by ``'masked_median'``.
        despike_size: Post-detrend median filter, samples; 0 = off. Applied by the
            preprocessing chain to the DETRENDED trace, never to the baseline --
            carried here so one object describes the whole estimator.

    ``rough_s`` and ``base_s`` are windows in SECONDS, converted on the decimated
    grid as ``w = int(win_s * fs / decim)`` and then forced odd. They are not tap
    counts and not block sizes.

    Deliberately NOT here: ``percentile``, the four ``asls_*`` knobs, and
    ``sg_poly``/``sg_decim``. This object owns the I0-tracking SHAPE, which is what
    has to be identical across the four call sites; those three methods' own tuning
    stays on `estimate_baseline` and in ``modules.DETREND_METHOD_PARAMS`` where the
    detrending stage already exposes it. `estimate_baseline_result` therefore runs
    them at their documented defaults -- for ``'savgol'`` that is `SAVGOL_POLY` /
    `SAVGOL_DECIM`, the Cu recipe's own values, with ``window_s`` taken from the
    config as for every other window-based method.
    """

    method: str = "masked_median"
    rough_s: float = 0.5
    base_s: float = 0.1
    decim: int | None = None
    mask_sigma: float = 3.5
    dilate_ms: float = 1.0
    window_s: float = 0.5
    despike_size: int = 0

    #: The order the fingerprint hashes, and the order `to_provenance` writes.
    _FIELDS = ("method", "rough_s", "base_s", "decim", "mask_sigma",
               "dilate_ms", "window_s", "despike_size")

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; expected one of {METHODS}")
        if not (np.isfinite(self.window_s) and self.window_s > 0):
            raise ValueError(
                f"window_s must be a positive number of seconds, got {self.window_s!r}"
            )
        _validate_masked_params(
            method=self.method, rough_s=self.rough_s, base_s=self.base_s,
            decim=self.decim, mask_sigma=self.mask_sigma, dilate_ms=self.dilate_ms,
            despike_size=self.despike_size,
        )
        object.__setattr__(self, "despike_size", int(self.despike_size))
        if self.decim is not None:
            object.__setattr__(self, "decim", int(self.decim))

    # -- resolution ---------------------------------------------------------
    @property
    def is_masked(self) -> bool:
        """True for the two `MASKED_METHODS`, which take rough_s/base_s, not window_s."""
        return self.method in MASKED_METHODS

    @property
    def tracking_window_s(self) -> float:
        """The window that decides how fast the baseline may follow the trace.

        ``base_s`` for the two-pass estimator, ``window_s`` for everything else. This
        is what the length checks and the "is it too short for the events" reasoning
        are about, and having one name for it is what lets `baseline_health` treat
        all five methods alike.
        """
        return float(self.base_s) if self.method == "masked_median" else float(self.window_s)

    def resolved_decim(self, fs: float) -> int:
        """``self.decim`` if pinned, else derived from the tracking window at ``fs``.

        Raises when a pinned ``decim`` leaves the filter fewer than `MIN_TAPS` taps --
        that is the regime where the reference's "baseline" is literally one sample
        broadcast over the whole recording, with no error and no warning.
        """
        win = self.tracking_window_s
        if self.decim is None:
            return derive_decim(win, fs)
        decim = int(self.decim)
        taps = int(win * float(fs) / decim)
        if taps < MIN_TAPS:
            raise ValueError(
                f"decim={decim} leaves only {taps} tap(s) for a {win * 1e3:g} ms window "
                f"at {fs:g} Hz (need >= {MIN_TAPS}). The 'baseline' would be one or two "
                f"samples broadcast over the whole recording. Use decim <= "
                f"{max(1, int(win * float(fs) / MIN_TAPS))}."
            )
        return decim

    def kernel_kwargs(self, fs: float) -> dict[str, Any]:
        """The exact keyword set for the imported `ml_preprocess` kernel."""
        decim = self.resolved_decim(fs)
        if self.method == "masked_median":
            return dict(rough_s=float(self.rough_s), base_s=float(self.base_s),
                        decim=decim, mask_sigma=float(self.mask_sigma),
                        dilate_ms=float(self.dilate_ms))
        if self.method == "robust_median":
            return dict(window_s=float(self.window_s), decim=decim)
        raise ValueError(f"{self.method!r} is not one of the kernel methods {MASKED_METHODS}")

    def resolve(self, fs: float) -> "BaselineConfig":
        """A copy with ``decim`` pinned to what it resolves to at ``fs``.

        What travels into provenance, so a restored project reproduces the numbers
        even if `TAPS_TARGET` or `derive_decim` ever changes.
        """
        if not self.is_masked:
            return self
        return _dc_replace(self, decim=self.resolved_decim(fs))

    def replace(self, **kw: Any) -> "BaselineConfig":
        """A validated copy with fields overridden."""
        return _dc_replace(self, **kw)

    # -- identity / persistence --------------------------------------------
    def fingerprint(self, fs: float | None = None) -> str:
        """Short digest identifying the estimator that produced a set of rows.

        Follows the `core.analysis.AnalysisResult.calibration_fingerprint` precedent
        exactly: rows carry the stamp, a stored result whose stamp is absent or
        differs from the live config is marked STALE, and the user is offered a
        re-run. Nothing is ever silently re-measured -- ``depth_pA`` moving 15x
        between two openings of the same project is indistinguishable from data
        corruption. Rows produced before this change have no fingerprint at all,
        which is itself the signal.

        ``fs`` resolves a ``decim`` of ``None`` first, so two recordings at different
        rates do not share a fingerprint while running different strides.
        """
        cfg = self.resolve(fs) if fs is not None else self
        parts = [f"{name}={getattr(cfg, name)!r}" for name in self._FIELDS]
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return f"sha256:{digest[:12]}"

    def to_provenance(self, fs: float | None = None) -> dict[str, Any]:
        """The flat, JSON-serialisable ``baseline_*`` provenance keys.

        Prefixed consistently so `modules.preprocess_params_from_provenance` needs no
        translation-table entry: the param name and the provenance name are the same
        string. NEVER a new EventSet column -- the 20-column SCHEMA is frozen.
        """
        cfg = self.resolve(fs) if fs is not None else self
        return {
            "baseline_method": cfg.method,
            "baseline_rough_s": float(cfg.rough_s),
            "baseline_base_s": float(cfg.base_s),
            "baseline_decim": None if cfg.decim is None else int(cfg.decim),
            "baseline_mask_sigma": float(cfg.mask_sigma),
            "baseline_dilate_ms": float(cfg.dilate_ms),
            "baseline_window_s": float(cfg.window_s),
            "baseline_despike_size": int(cfg.despike_size),
            "baseline_fingerprint": cfg.fingerprint(),
        }

    @classmethod
    def from_provenance(cls, prov: Mapping[str, Any] | None) -> "BaselineConfig":
        """Rebuild from `to_provenance` output (or from a partial/legacy dict).

        Missing keys fall back to `DEFAULT_BASELINE`'s values, so a provenance dict
        written before a field existed still loads. A row with no
        ``baseline_method`` at all is a PRE-CHANGE row: it is reconstructed as the
        block median at its old 10 ms window, which is the honest description of
        what measured it.
        """
        if not prov:
            return DEFAULT_BASELINE
        method = prov.get("baseline_method")
        if method not in METHODS:
            # Absent, or a marker rather than an estimator: ``'provided'`` is what
            # `detect_events` stamps when the CALLER handed it a baseline array, and
            # every row written before this change has no key at all. Neither names
            # a reproducible estimator, so both get the honest description of the
            # pre-change default. Callers that must NOT guess -- the rebuild path --
            # ask `reproducible_from` instead, which returns None here.
            return cls(method="block_median", window_s=_LEGACY_WINDOW_S)
        d = dict(prov)
        decim = d.get("baseline_decim", None)
        return cls(
            method=str(d.get("baseline_method", DEFAULT_BASELINE.method)),
            rough_s=float(d.get("baseline_rough_s", DEFAULT_BASELINE.rough_s)),
            base_s=float(d.get("baseline_base_s", DEFAULT_BASELINE.base_s)),
            decim=None if decim is None else int(decim),
            mask_sigma=float(d.get("baseline_mask_sigma", DEFAULT_BASELINE.mask_sigma)),
            dilate_ms=float(d.get("baseline_dilate_ms", DEFAULT_BASELINE.dilate_ms)),
            window_s=float(d.get("baseline_window_s", DEFAULT_BASELINE.window_s)),
            despike_size=int(d.get("baseline_despike_size", DEFAULT_BASELINE.despike_size)),
        )

    @classmethod
    def reproducible_from(
        cls, prov: Mapping[str, Any] | None
    ) -> "BaselineConfig | None":
        """The config that MEASURED these rows, or ``None`` when it is not knowable.

        The one backward resolver. `from_provenance` always answers, because a
        config object is what its callers need; this one refuses, because a REBUILD
        that guesses is worse than a rebuild that falls back to the local flank
        median it has always used. It returns ``None`` when the provenance carries
        no ``baseline_fingerprint`` -- i.e. a row measured before this change, or
        one whose detector was handed a baseline array by its caller
        (``baseline_method == 'provided'``) and so cannot name an estimator at all.

        The rebuilt row is only guaranteed to land on the detector's own baseline
        when this returns a config; that is exactly the condition
        `characterize.recharacterize_span` branches on, and it records which.
        """
        if not prov:
            return None
        if not prov.get("baseline_fingerprint"):
            return None
        if prov.get("baseline_method") not in METHODS:
            return None
        return cls.from_provenance(prov)

    def describe(self) -> str:
        """One line for a dialog or a log."""
        if self.method == "masked_median":
            d = "auto" if self.decim is None else str(self.decim)
            return (f"masked median, rough {self.rough_s * 1e3:g} ms / base "
                    f"{self.base_s * 1e3:g} ms, decim {d}, mask {self.mask_sigma:g} sigma, "
                    f"dilate {self.dilate_ms:g} ms")
        if self.method == "robust_median":
            d = "auto" if self.decim is None else str(self.decim)
            return f"robust median, window {self.window_s * 1e3:g} ms, decim {d}"
        return f"{self.method}, window {self.window_s * 1e3:g} ms"


#: THE default. The only baseline-parameter literal in the app.
#:
#: These are the notebook's own ``MASKED_MEDIAN_DEFAULTS`` -- the one parameter set
#: in evidence that is a *default* rather than a per-recording override. On the real
#: apo recording it already gets the median stamped-baseline error to 1.56 pA and
#: the fraction of events stamped > 10 pA off to 0.97%, against the 10 ms block
#: median's 80.6%. It is a defensible FLOOR, not an answer: that same file measures
#: best at ``rough_s = 2.0``, and ``ferritin_data3`` at 0.03. Propose per recording,
#: confirm with the user, persist with the analysis.
DEFAULT_BASELINE = BaselineConfig()

#: What a run at `DEFAULT_BASELINE` says about itself. See `BaselineResult.warnings`
#: for the measurement that makes it necessary: on the owner's gating pore the
#: default is 3.3x WORSE than the block median this change replaces, and every
#: health check reads ok.
UNPROBED_DEFAULT_WARNING = (
    "These depths were measured with the app's DEFAULT open-pore window "
    f"({DEFAULT_BASELINE.rough_s:g} s / {DEFAULT_BASELINE.base_s:g} s); this "
    "recording was never measured. The correct window is a property of the "
    "recording, not of the app -- on the owner's own four files it ranges from "
    "0.03 s (a gating pore) to 2.0 s (a 521 ms partial clog), and on the gating "
    "pore the default is 3.3x worse than the estimator it replaced with every "
    "health check reading ok. Open the Preprocess tab and press \u201cMeasure this "
    "recording\u201d before trusting a depth from this table."
)


def level_occupancy_warning(
    level_occupancy: float,
    *,
    alt_occupancy: float | None = None,
    alt_offset_pA: float | None = None,
    tracked_pA: float | None = None,
) -> str:
    """The advisory a baseline carries when its open-pore identification was made and
    not verified -- which is whenever this recording has a SECOND level
    (`ALT_LEVEL_OCCUPANCY_MIN`), or the tracked level's own margin is under
    `LEVEL_OCCUPANCY_MIN`.

    It is a WARNING on the result rather than a failing check, and the two constants'
    docstrings carry the measurements that forced that choice: an inverted estimate and
    a valid one sit on the same side of every statistic this module has -- occupancy,
    second-mode occupancy, mode sharpness, per-excursion agreement, side of zero -- so a
    check that condemned the first would condemn the owner's own four recordings with
    it. What is honest is to name both readings and hand the decision to the one party
    that can make it, the same thing ``mask_fraction_floor`` does when its number has
    two opposite meanings.

    WHY IT IS NOT KEYED TO THE MARGIN ALONE, which is what it used to be. On an
    inverted estimate the tracked level's occupancy RISES with the blockade fraction
    (0.791 at 81% blocked, 0.897 at 90%, 0.942 at 95%, 0.970 at 97%), so a line at
    `LEVEL_OCCUPANCY_MIN` fired on the shallowest case and went silent on the three
    worse ones -- the app got quieter the more wrong it was. The presence of a second
    level does not move that way.

    ``alt_offset_pA`` and ``tracked_pA`` are optional because a health block restored
    from a project written before they existed does not carry them; with them the
    sentence can name both candidate currents and both readings of dI/I0, which is the
    form in which the ambiguity is actually actionable.

    WHAT THE SENTENCE MAY CLAIM, and the one it used to overclaim. ``alt_offset_pA`` is
    the SEPARATION between the two levels, measured by this probe on the residual
    histogram. It is not a depth from the event table -- the table's ``depth_pA`` is
    measured per event -- and the sentence used to say *"Every depth in the table is the
    same D pA either way"*. Against the app's own exported CSVs that was wrong on all
    three of the owner's recordings at once (-700 mV: 91.9 pA claimed over a column
    running 13.43-167.41 pA, 20.1% of rows within 5%; -500 mV: 55.8 pA over
    10.99-126.04, 13.0%; -600 mV: 86.1 pA over 12.66-162.66, 32.2%), and the dI/I0 it
    quoted -- ``separation / |I0|`` -- was off the table's own median ``frac_depth`` by
    5.80% / 5.77% / 1.50%. It reaches the reader in the voltage-dependence DETECTION
    CAVEATS block directly above a fitted "Median depth (pA)" trend, which is the worst
    possible place for a sentence that sounds like it settled the amplitude question.
    So the claim is scoped to what was measured: the separation, named as such, and its
    dI/I0 against each candidate open pore. It stops there deliberately -- under the
    other reading the baseline every row is measured from moves too, so both halves of
    every ``frac_depth`` change and nothing here can say by how much for a given row.
    """
    alt_occ = 0.0 if alt_occupancy is None else float(alt_occupancy)
    if alt_offset_pA is not None and tracked_pA is not None:
        other = float(tracked_pA) + float(alt_offset_pA)
        depth = abs(float(alt_offset_pA))
        here = (f"{depth / abs(tracked_pA):.3f}"
                if tracked_pA else "undefined")
        there = f"{depth / abs(other):.3f}" if other else "undefined"
        # THE SENTENCE MUST NAME ITS OWN BASIS. ``depth`` is |alt_offset_pA|, the
        # SEPARATION between the two levels this probe found -- not a depth from the
        # event table, and the two are not the same number: measured against the app's
        # own exported CSVs, the -700 mV recording's advisory quoted 91.9 pA over a
        # depth column running 13.43-167.41 pA (12x) with a median of 96.13 pA and only
        # 20.1% of rows within 5% of the quoted value (-500 mV: 55.8 pA over
        # 10.99-126.04, 13.0% within 5%; -600 mV: 86.1 pA over 12.66-162.66, 32.2%).
        # The dI/I0 it quotes is likewise ``separation / |I0|``, off the table's own
        # median frac_depth by 5.77% / 1.50% / 5.80% on those three -- and the sentence
        # is reprinted verbatim in the voltage-dependence DETECTION CAVEATS block,
        # directly above a fitted "Median depth (pA)" trend. What IS invariant is the
        # separation, and what the sentence may say about dI/I0 is the reading of THAT
        # separation against each candidate open pore. It must not go on to predict the
        # table: under the other reading the baseline every row is measured from moves
        # too, so both the numerator and the denominator change and this block cannot
        # say by how much for any given row.
        levels = (
            f"This recording has TWO conductance levels: the tracked one at "
            f"{float(tracked_pA):.1f} pA, which holds {level_occupancy:.0%} of the "
            f"recording, and another at {other:.1f} pA holding {alt_occ:.0%}. The "
            "estimator called the first the open pore because the pore is there most "
            "of the time -- that is its whole rule, and no check in the Baseline "
            "health block reads the physics that would confirm it. Read the other way "
            f"up, this is a pore BLOCKED {level_occupancy:.0%} of the time whose "
            f"returns to a {other:.1f} pA open pore are what got masked as events. "
            f"The {depth:.1f} pA SEPARATION between the two levels is the same "
            "whichever one you call the open pore -- that is this probe's own "
            "measurement, and it is NOT a depth from the event table, whose depth_pA "
            "is measured per event and varies from event to event. What the choice "
            f"changes is what a {depth:.1f} pA excursion is a fraction OF: dI/I0 reads "
            f"{here} against this baseline and {there} against the other one. Every "
            f"depth and every dI/I0 in the table is referenced to {float(tracked_pA):.1f} "
            "pA, so if the other level is the open pore the whole column is measured "
            "from the wrong current -- and by how much, row by row, is not something "
            "this block can tell you."
        )
    else:
        levels = (
            f"The tracked baseline is the level this recording spends "
            f"{level_occupancy:.0%} of its time at, and this recording also visits "
            f"another level for {alt_occ:.0%} of it. The estimator called the first "
            "the open pore because the pore is there most of the time -- that is its "
            "whole rule, and no check in the Baseline health block reads the physics "
            f"that would confirm it. Read the other way up, this is a pore BLOCKED "
            f"{level_occupancy:.0%} of the time whose brief returns to the open pore "
            "are what got masked as events."
        )
    return (
        "The open pore here is NOT CERTIFIED, only chosen. " + levels + " The two "
        "readings are the same samples, so nothing in this block separates them: on a "
        "synthetic pore blocked 81% of the time (open 100 pA, blocked 60 pA) all eight "
        "checks read ok, overshoot included, over a baseline of 60.04 pA and a median "
        "dI/I0 of 0.665 against a true 0.400, with the events reported as "
        "'enhancement'; at 90% and 95% blocked the same fixture reads the same way. "
        "COMPARE THE BASELINE with the open-pore current you know for this pore -- the "
        "current just after the voltage step, or an unclogged stretch -- and analyse "
        "that stretch instead if the two disagree."
    )


# ---------------------------------------------------------------------------
# polarity derivation (target stores no polarity -- measure it)
# ---------------------------------------------------------------------------


def _count_coherent_runs(mask: np.ndarray, min_run: int = 2) -> int:
    """Count runs of consecutive True samples of length >= min_run.

    A single spurious sample beyond threshold is noise, not an event; requiring
    coherence of >= 2 samples is the same guard the detector uses.
    """
    if mask.size == 0 or not mask.any():
        return 0
    m = mask.view(np.int8)
    d = np.diff(np.concatenate(([np.int8(0)], m, [np.int8(0)])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts
    return int(np.count_nonzero(lengths >= min_run))


def require_polarity(polarity: Any) -> int:
    """``polarity`` as an int, or a ValueError naming why it cannot be one.

    THE ONE GATE for a value that is a DIRECTION rather than a number. Every consumer
    in the app multiplies by it -- ``dev = polarity * (current - baseline)`` in the
    detector, the canonical-frame negation here -- against thresholds derived from the
    UNSCALED noise, so a value outside ``{-1, +1}`` does not fail: it silently
    rescales, and then it is STAMPED (``provenance['polarity']``,
    ``baseline_polarity``) as if it had been measured. Measured through
    `detect.detect_events` on 20 blockades of 100 pA against a 1000 pA open pore:
    ``polarity=0`` returned an empty table with no warning, 7 returned 56 rows at a
    median depth of 36.02 pA and 100 returned 403 rows at 375.83 pA.

    It is a function rather than four inline checks because the value arrives at four
    doors -- `detect.detect_events`, `estimate_baseline_result`,
    `characterisation_baseline` and `estimate_baseline` -- from the same places
    (``ctx.params`` forwarded by `modules._detect_kwargs`, a restored provenance
    mapping, a batch config), and a rule enforced at three of four doors is not a rule.
    ``None`` is not accepted here: "measure it" is a decision each door makes for
    itself, and each one already spells it.
    """
    try:
        exact = int(polarity) == polarity
    except (TypeError, ValueError):
        exact = False
    if not exact or int(polarity) not in (-1, 1):
        raise ValueError(
            f"polarity must be +1 or -1, got {polarity!r}. It is the FACTOR that makes "
            "dev = polarity*(current - baseline) positive inside events -- a direction, "
            "not a gain -- and the thresholds it is compared against come from the "
            "unscaled noise, so any other value rescales the measurement and is then "
            "stamped into provenance as if it were one. Pass None to measure the "
            "direction from the data, or -1 for blockades and +1 for enhancements in "
            "the SIGNED current."
        )
    return int(polarity)


def derive_polarity(data: np.ndarray, *, k: float = 5.0) -> int:
    """Measure which way events point: the factor making dev positive-in-event.

    Returns +1 when coherent excursions run *above* a robust centre (a blockade
    at negative bias, or an enhancement at positive bias), -1 when they run
    below. Never assumes "blockade = down": at negative bias the open-pore
    current is negative and an ordinary blockade is an *upward* excursion, so the
    direction can only be measured, not guessed. Enhancement (+1) is real
    physics, not an error to be rectified away.

    Detected from a global-median centre (no baseline needed -- this is the
    chicken-and-egg breaker) by counting coherent runs each way beyond
    ``k * MAD-noise``. A flat or event-free trace has nothing to measure and
    falls back to +1.

    Raises:
        ValueError: on an empty trace, or one carrying NaN/Inf. A single NaN makes
            the median NaN, hence ``sigma`` NaN, hence ``sigma > 0`` False -- so the
            function used to take the *flat trace* exit and return ``+1`` for a
            recording whose direction it had not measured at all. ``+1`` is the
            wrong sign for every negative-bias blockade in this app, and it turns
            every depth into the negative of itself. Callers that have a defensible
            fallback (``merge._polarity``, ``characterize._resolve_polarity``)
            already catch this; ``estimate_baseline`` refuses a non-finite trace
            before it ever gets here.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot derive polarity from an empty trace")
    # ``all()`` on a bool array short-circuits, so the clean case (every real
    # recording) is one pass and no second reduction: measured 0.50 ms on the
    # owner's 2,993,798-sample -600 mV file, against the 23 ms this function then
    # spends on its first median and the 0.36 s of the detection around it.
    finite = np.isfinite(x)
    if not finite.all():
        raise ValueError(f"cannot derive polarity: {_nonfinite_reason(finite)} The median "
                         "and the MAD are NaN over them, so the polarity would silently "
                         "fall back to +1 instead of being measured.")
    resid = x - np.median(x)
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    if not (sigma > 0):
        return 1
    thr = k * sigma
    n_up = _count_coherent_runs(resid > thr)
    n_down = _count_coherent_runs(resid < -thr)
    if n_up == n_down:
        # No dominant direction (event-free or symmetric): fall back to the
        # heavier tail, then to +1.
        up_mass = float(resid[resid > thr].sum()) if n_up else 0.0
        down_mass = float(-resid[resid < -thr].sum()) if n_down else 0.0
        return 1 if up_mass >= down_mass else -1
    return 1 if n_up > n_down else -1


# ---------------------------------------------------------------------------
# block machinery
# ---------------------------------------------------------------------------


def _block_stat(x: np.ndarray, nb: int, q: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-block median (q is None) or percentile, plus the block centre indices.

    Vectorised via reshape -- the whole point of this module's performance. A
    rolling median with a 100k-sample kernel is O(n*k) and takes minutes on 5M
    samples; this is O(n log nb) and takes tenths of a second.
    """
    n = x.size
    n_full = n // nb
    vals: list[np.ndarray] = []
    centres: list[np.ndarray] = []

    if n_full:
        head = x[: n_full * nb].reshape(n_full, nb)
        v = np.median(head, axis=1) if q is None else np.percentile(head, q, axis=1)
        vals.append(np.asarray(v, dtype=np.float64))
        centres.append(np.arange(n_full, dtype=np.float64) * nb + (nb - 1) / 2.0)

    tail = n - n_full * nb
    if tail >= max(2, int(nb * _MIN_TAIL_FRAC)):
        t = x[n_full * nb :]
        v = np.median(t) if q is None else np.percentile(t, q)
        vals.append(np.asarray([v], dtype=np.float64))
        centres.append(np.asarray([n_full * nb + (tail - 1) / 2.0]))

    return np.concatenate(vals), np.concatenate(centres)


def _expand(centres: np.ndarray, knots: np.ndarray, n: int) -> np.ndarray:
    """Interpolate block knots back to full sample rate.

    PchipInterpolator, not CubicSpline: Pchip is shape-preserving, so it cannot
    overshoot. A natural cubic spline rings at a drift step, and the overshoot
    lobe is a phantom deviation that detection reports as an event.
    """
    if knots.size < 2:
        return np.full(n, float(knots[0]), dtype=np.float64)
    return PchipInterpolator(centres, knots, extrapolate=True)(
        np.arange(n, dtype=np.float64)
    )


def _asls_knots(y: np.ndarray, polarity: int, lam: float, p: float, niter: int) -> np.ndarray:
    """Asymmetric least squares (Eilers & Boelens 2005) on the decimated series.

    Runs on block knots, not raw samples: the baseline is slow by construction,
    and a banded solve on 5M unknowns would cost far more for nothing. `lam` is
    therefore a smoothness in *block* units -- it rescales with the asls knot
    spacing.

    Asymmetry follows the sign convention: residuals on the +polarity side are
    event-like and get weight p (~ignored); the open-pore side gets 1-p. Unlike
    the block median, asls does NOT need most knots to fall between events -- the
    asymmetry pulls the fit toward the open-pore envelope even where a knot lands
    inside an event. That is why it survives high occupancy where a plain median
    is already eaten; but it must still decimate fine enough that *some* knots
    see the open pore, hence `asls_decim`.
    """
    m = y.size
    if m < 3:
        return y.copy()
    d = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(m - 2, m), format="csc")
    dtd = lam * (d.T @ d)
    w = np.ones(m)
    z = y.copy()
    for _ in range(niter):
        wd = sp.diags(w, 0, format="csc")
        z = spla.spsolve(wd + dtd, w * y)
        w = np.where(polarity * (y - z) > 0, p, 1.0 - p)
    return z


def _savgol_baseline(
    x: np.ndarray, *, fs: float, window_s: float, poly: int, decim: int
) -> np.ndarray:
    """The Cu2-insulin ``baseline_remove_sg`` recipe, returning the BASELINE.

    Decimate by ``decim``, fit a degree-``poly`` Savitzky-Golay over a ``window_s``
    window ON THAT DECIMATED GRID with ``mode='mirror'``, then ``np.interp`` the
    result back to full length. Transcribed from
    ``Cu2Insulin_DataPreprocessing.ipynb`` Block 2::

        n  = len(sig)
        xs = np.asarray(sig[::decim], np.float64)
        w  = int(window_s * fs / decim)
        if w % 2 == 0: w += 1
        w  = max(w, poly + 2 + ((poly + 2) % 2 == 0))
        if w > len(xs): w = len(xs) - ((len(xs) + 1) % 2)
        b  = savgol_filter(xs, window_length=w, polyorder=poly, mode='mirror')
        baseline = np.interp(np.arange(n), np.arange(0, n, decim)[:len(b)], b)
                     .astype(np.float32)
        return (sig - baseline).astype(np.float32), baseline

    TWO TRAPS LIVE IN THAT NAME AND BOTH ARE WHY THIS FUNCTION IS HERE.

    First, the app already has a ``savgol_enabled`` stage in `filters.savgol` and it
    is the OPPOSITE OPERATION: a SMOOTHER, which RETURNS the smoothed signal as a
    low-pass. This is a baseline ESTIMATOR -- the smoothed signal is what gets
    SUBTRACTED. The two share a filter and a name and nothing else, and running one
    where the other was meant leaves a trace that looks plausible and is inverted in
    what it kept.

    Second, `ml_preprocess.baseline_savgol` already exists and is a DIFFERENT
    ESTIMATOR: the APO chain's, undecimated, ``w = int(window_s * fs)``, scipy's
    default ``mode='interp'``. On the Cu recipe's own parameters (0.2 s, poly 3,
    50 kHz) that is a 10 001-tap fit on the raw series against this one's 201-tap fit
    on a 1 kHz series -- close in intent, not the same numbers, and O(n*w) rather
    than O(n*w/decim**2). It is frozen against the apo notebooks and must not be
    retargeted, so the Cu recipe gets its own function rather than a fourth keyword
    on that one.

    THE float32 CAST LANDS ON THE BASELINE, BEFORE THE SUBTRACTION, exactly as
    `_kernel_baseline` documents for the masked kernels: the notebook's
    ``_interp_back`` casts, and ``(signal - float64_view_of_float32).astype(float32)``
    is then byte-identical to the notebook's ``current_detr_pA``. Returning float64
    holding those float32 values is `estimate_baseline`'s standing contract.

    THIS ESTIMATOR DOES NOT MASK, AND THAT IS THE POINT ON THE Cu FILES. The owner's
    Aug-2026 sweep chose it over ``masked_median`` for all four recordings, decisively
    on ``cu_10h``: a 3-sigma mask treats that file's baseline-unstable stretches
    (t < 20 s, t > 370 s) as events and interpolates ACROSS them, so the instability
    re-emerges in the detrended trace as enormous false events (measured attenuation
    -341% to -723%, 107-381 "deep" events against a true 28). Plain SavGol tracks
    straight through. The cost is stated rather than hidden: SavGol raises
    ``qc_overshoot`` (cu_30h 0.13 -> 0.37, cu_10h 0.18 -> 0.64) because the polynomial
    bows around a deep excursion -- which is exactly why `ml_preprocess.baseline_savgol`
    warns the ferritin models off it. Cu events are DEEP AND RARE (~105 in the whole
    four-file dataset), so the bow is paid ~26 times per recording; on a
    shallow-and-abundant recording like ``zn_30h`` (13 355 events) it would not be.
    """
    n = x.size
    decim = int(decim)
    poly = int(poly)
    if decim < 1:
        raise ValueError(f"sg_decim must be >= 1, got {decim}")
    if poly < 1:
        raise ValueError(f"sg_poly must be a positive integer, got {poly}")

    xs = np.ascontiguousarray(x[::decim], dtype=np.float64)
    w = int(float(window_s) * float(fs) / decim)
    if w % 2 == 0:
        w += 1
    # A window must host more points than the polynomial has coefficients, and be ODD
    # (the fit is centred on the sample it estimates). The notebook writes this as
    # ``max(w, poly + 2 + ((poly + 2) % 2 == 0))``; poly=3 -> 5.
    floor_w = poly + 2
    if floor_w % 2 == 0:
        floor_w += 1
    w = max(w, floor_w)
    if w > xs.size:
        # The notebook's clamp: the largest ODD length that fits the decimated series.
        w = xs.size - ((xs.size + 1) % 2)
        _warn_baseline(
            f"the {window_s:g} s SavGol baseline window is {int(float(window_s) * float(fs) / decim)} "
            f"taps on the decim-{decim} grid but the recording has only {xs.size} "
            f"decimated sample(s); the window was CLAMPED to {w}. A clamped window is "
            "a different estimator from the one you asked for -- it fits one polynomial "
            "over (nearly) the whole recording, so it tracks no drift inside it.",
            stacklevel=4,
        )
    if w <= poly:
        raise ValueError(
            f"recording is {n} sample(s) ({xs.size} at decim {decim}); a degree-{poly} "
            f"SavGol baseline needs a window of more than {poly} decimated samples and "
            f"this one clamps to {w}. Lower sg_decim, lower sg_poly, or analyse more "
            "signal."
        )
    b = savgol_filter(xs, window_length=w, polyorder=poly, mode="mirror")
    xp = np.arange(0, n, decim)[: b.size]
    baseline = np.interp(np.arange(n), xp, b).astype(np.float32)
    return np.asarray(baseline, dtype=np.float64)


def _single_pass(
    x: np.ndarray,
    *,
    fs: float,
    polarity: int,
    window_s: float,
    method: str,
    percentile: float,
    asls_lam: float,
    asls_p: float,
    asls_niter: int,
    asls_decim: float,
    sg_poly: int = SAVGOL_POLY,
    sg_decim: int = SAVGOL_DECIM,
) -> np.ndarray:
    n = x.size
    if method == "constant":
        return np.full(n, float(np.median(x)), dtype=np.float64)

    # SavGol has no blocks and no knots -- it is a direct convolution on a decimated
    # copy -- so it leaves before the block arithmetic below rather than pretending to
    # share it. It is POLARITY-BLIND on purpose: unlike 'percentile'/'asls' it fits the
    # signal symmetrically, which is what lets it track through the baseline
    # instability a one-sided estimator would mistake for events.
    if method == "savgol":
        return _savgol_baseline(
            x, fs=fs, window_s=window_s, poly=sg_poly, decim=sg_decim
        )

    # asls decimates finer than the smoothing window so some knots land in the
    # gaps between dense events; the DtD penalty (not the block size) sets the
    # final smoothness.
    win = window_s * asls_decim if method == "asls" else window_s
    nb = max(1, int(round(win * fs)))
    if nb > n:
        raise ValueError(
            f"window_s={window_s:g}s is longer than the recording ({n / fs:g}s); "
            "no block can be formed. Use method='constant' if that is what you meant."
        )

    if method == "percentile":
        # The open-pore level is the envelope opposite the excursions: events run
        # toward +polarity, so I0 is the high percentile when polarity=-1 and the
        # low one when polarity=+1. Getting this backwards traces the events.
        q = percentile if polarity == -1 else 100.0 - percentile
        knots, centres = _block_stat(x, nb, q)
    else:
        knots, centres = _block_stat(x, nb, None)

    if knots.size < 2:
        # ConstantBaselineWarning (not the base class): this is the DEGENERATE case —
        # one block means the estimate is a flat level and a detrend built on it does
        # nothing — and callers (modules.DetectSchmitt.run) must be able to recognise
        # the no-op structurally to keep provenance honest about it.
        _warn_baseline(
            f"window_s={window_s:g}s yields {knots.size} block(s) over {n / fs:g}s: "
            "the baseline is effectively constant and cannot track drift.",
            category=ConstantBaselineWarning,
            stacklevel=3,
        )

    if method == "asls":
        knots = _asls_knots(knots, polarity, asls_lam, asls_p, asls_niter)

    return _expand(centres, knots, n)


# ---------------------------------------------------------------------------
# the imported kernel, met by this app's detected polarity
# ---------------------------------------------------------------------------


def _kernel_baseline(
    x: np.ndarray, fs: float, *, polarity: int, config: BaselineConfig
) -> np.ndarray:
    """Run the `ml_preprocess` kernel and return the baseline, float64, full length.

    POLARITY BY MIRRORING, NOT BY BRANCHING INSIDE THE KERNEL. The reference mask is
    ONE-SIDED and DOWNWARD; it has no polarity parameter, because every recording in
    the owner's corpus has events running below the open-pore level. This app
    *detects* polarity (`derive_polarity`, never assumed) and enhancement events --
    which run UP -- are real physics below ~400 mM salt. So::

        pol == -1  ->  call the kernel with the signal untouched
        pol == +1  ->  call the kernel with -x, and negate the baseline back

    At ``pol == -1`` there is no arithmetic in this wrapper at all, so **the corpus
    contract holds by construction rather than by assertion** -- the bytes that come
    back are the kernel's own. At ``pol == +1`` it is the exact mirror algorithm:
    every step involved (odd-window order statistics, medians, ``np.interp``,
    the float32 cast) is exactly sign-symmetric in IEEE754, so mirroring loses
    nothing. Both claims are pinned by tests rather than argued.

    ``robust_median`` needs no mirror -- it has no mask, and a rolling median of a
    negated signal is the negation of the rolling median exactly -- but it is
    mirrored anyway so there is ONE polarity rule in this module instead of two.

    The kernel returns float32; this returns float64 holding those float32 values,
    which is what `estimate_baseline` has always returned. Per the notebook's own
    dtype trajectory the cast lands on the BASELINE before the subtraction, so
    ``(signal - float64_view).astype(float32)`` is byte-identical to the notebook's
    ``current_detr_pA``.
    """
    kw = config.kernel_kwargs(fs)
    kernel = _masked_median_kernel if config.method == "masked_median" else _robust_median_kernel
    check_cancelled()
    if int(polarity) == -1:
        _, base = kernel(x, fs, **kw)
    else:
        _, base = kernel(-x, fs, **kw)
        base = -base
    check_cancelled()
    return np.asarray(base, dtype=np.float64)


def upper_mad_sigma(residual: np.ndarray, *, median: float | None = None) -> float:
    """One-sided UPPER MAD of a residual, in the residual's own units.

    ``1.4826 * median(r[r > median(r)] - median(r))``: a robust sigma read from
    only the half of the distribution DOWNWARD events cannot reach, so it survives
    any event duty cycle -- which is the whole reason it is here rather than a
    two-sided MAD (inflated once occupancy passes ~10%) or the obvious delta-based
    estimator.

    **The delta estimator reads SEVEN TIMES LOW on this data.**
    ``1.4826*median|diff(x)|/sqrt(2)`` gives 0.97 pA on ferritin_data1 against a
    true 6.75, because the owner's "process" ABF exports are already low-passed and
    consecutive samples are strongly correlated. Every quantity built on it -- mask
    fraction, dwell percentiles, the gating flag -- came out garbage. Measured
    against the final QC noise: data1 6.20 (true 6.75), data2 17.5 (14.7),
    apo 4.50 (4.84). It OVER-reads on a gating pore (39-62 against 13.8 on
    ferritin_data3) because the upper conductance level contaminates the upper
    half, which is exactly why `baseline_probe` pairs it with a gating check and
    normalises the gating statistic by a short-window noise instead.

    ONE definition, three callers: `_rough_pass` (the health block's
    ``upper_mad_sigma_pA``), the probe's mask threshold, and the probe's noise
    floor. ``median`` skips recomputing a median the caller already has.

    Returns ``nan`` for an empty upper half (a constant trace), which is what every
    caller wants: a threshold built on it is then never crossed and the caller's
    own finite check fires.
    """
    r = np.asarray(residual)
    med = float(np.median(r)) if median is None else float(median)
    upper = r[r > med]
    if upper.size == 0:
        return float("nan")
    return float(1.4826 * np.median(upper - med))


def _rough_pass(
    x: np.ndarray, fs: float, *, config: BaselineConfig
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Re-run the kernel's PASS 1 for diagnostics: ``(rough, mask, sigma, upper_mad)``.

    The kernel returns only ``(detrended, baseline)``, so the mask -- the single most
    diagnostic thing the estimator computes -- is not observable from its output.
    Rather than fork the kernel to return it (a second implementation is the highest
    risk in this change), the rough pass is recomputed here from the SAME imported
    `decimated_median`, using the same five lines of threshold arithmetic.

    That those five lines really are the kernel's is not taken on trust: the
    reconstruction test feeds this mask through the kernel's own fill + pass 2 and
    asserts the result is byte-identical to `baseline_masked_median`'s output on the
    real corpus. If the kernel's masking ever changes, that test fails.

    Cost: one extra pass-1 median -- ~80% of the two-pass total at the default
    parameters, because ``rough_s`` is 5x ``base_s`` and cost is linear in the window.
    It is why the health block is opt-in.

    ``x`` must already be in the CANONICAL frame (events running DOWN), and must be
    handed over with its dtype UNTOUCHED: the kernel computes ``rough`` in whatever
    precision its input carries (float64 for the notebook's ``filtfilt`` output,
    float32 for a raw app trace), and coercing it here would move the mask.
    """
    check_cancelled()
    decim = config.resolved_decim(fs)
    x = np.asarray(x)
    rough = x - decimated_median(x, fs, config.rough_s, decim)
    check_cancelled()
    med = np.median(rough)
    sigma = 1.4826 * np.median(np.abs(rough - med))
    upper_mad = upper_mad_sigma(rough, median=med)
    mask = binary_dilation(
        rough < med - float(config.mask_sigma) * sigma,
        iterations=max(1, int(float(config.dilate_ms) * 1e-3 * float(fs))),
    )
    check_cancelled()
    return rough, mask, float(sigma), upper_mad


def reference_rough_s(n: int, fs: float) -> float:
    """The widest rough window ``n`` samples at ``fs`` can support, seconds.

    `REFERENCE_ROUGH_S`, shrunk to ``duration / REFERENCE_WINDOWS_MIN`` on a short
    recording. That shrink is a MEASUREMENT SCALE adapting to what is measurable, not
    an estimator window shrinking to score better on a degenerate objective: nothing
    the estimator itself runs is ever narrowed by this function.

    Compare the result against `REFERENCE_MIN_ROUGH_S` before using it -- below that
    the window is too narrow to see over a state worth sizing anything against, and
    both callers refuse rather than report a number.

    One function because `reference_mask_fraction` and
    `baseline_probe.probe_timescales` must measure at the SAME scale: their two
    occupancies are compared against each other, and a comparison between two
    different window choices measures the window choice.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        return 0.0
    return float(min(REFERENCE_ROUGH_S, int(n) / fs / REFERENCE_WINDOWS_MIN))


def reference_mask_fraction(
    x_canonical: np.ndarray, fs: float, *, config: BaselineConfig
) -> tuple[float, float] | None:
    """``(mask fraction, rough window)`` for THIS config measured over a WIDE window.

    The second occupancy number. `baseline_health`'s ``mask_fraction`` is what the
    configured ``rough_s`` found; this is what the same mask threshold finds when the
    rough window is `reference_rough_s` instead -- wide enough to sail over the states
    rather than to track them. **Only the mask's window differs**: ``mask_sigma``,
    ``dilate_ms`` and ``despike_size`` are the caller's own and the mask comes from
    the same `_rough_pass`, so a disagreement between the two fractions cannot be
    explained by anything except the width of the window that exposed the events.

    ``base_s`` and ``decim`` move WITH the rough window, at `ROUGH_BASE_RATIO` and
    `derive_decim` -- the same matched pair a user would type -- rather than being
    carried over from a config whose windows were 200x narrower. That is not
    housekeeping in either direction:

    * carrying the run's ``decim`` costs 25 000 taps on ferritin_data3's 10 ms
      ``base_s`` (measured at >120 s against 62 ms re-derived), and
    * the mask fraction on a STEPPING pore genuinely depends on it: ferritin_data3's
      reference reads 0.0077 / 0.0094 / 0.0499 / 0.0409 across a 64x decim range,
      because a 5 s median taken at 75 taps is a staircase and at 300 a curve. The
      RATIO this function exists to feed is nonetheless stable there (0.06..0.40 on
      data3, 0.965..0.985 on data2, 0.98..1.00 on the apo file, 3.72..3.97 on the
      35%-occupied synthetic) -- every one of them the same verdict. Deriving the
      throttle from the reference window is what keeps it that way.

    NOT `baseline_probe.BaselineProbe.mask_fraction` itself, though that number is
    measured over the same window and was the obvious candidate. The probe's fraction
    comes from a ONE-SIDED `upper_mad_sigma` on an UNDILATED threshold, so it carries
    a systematic offset against anything `_rough_pass` produces -- 0.0931 against
    0.1062 on the apo recording, 0.0382 against 0.0549 on ferritin_data2 -- and a
    threshold on that ratio would be absorbing the offset as well as the signal.
    Re-measuring costs one decimated median (62 ms on ferritin_data3, 262 ms on the
    8.6 M-sample ferritin_data2) and makes the two numbers the same measurement.
    It also means the check does not depend on a probe having been run at all: a
    config typed by hand or restored from a project file gets it identically.

    Returns ``None`` when the recording cannot support a reference window wider than
    the one already configured -- which includes every trace shorter than
    ``REFERENCE_WINDOWS_MIN * REFERENCE_MIN_ROUGH_S`` = 2 s. A caller that gets
    ``None`` must report the cross-check as not measured; it must not read it as
    agreement.

    Args:
        x_canonical: The trace in the CANONICAL frame (events running DOWN), dtype
            untouched -- the same contract `_rough_pass` is written to.
        fs: Sample rate, Hz.
        config: The configuration under test. Its ``rough_s`` is the thing being
            replaced; everything else is preserved.

    Returns:
        ``(fraction, rough_s)``, or ``None``.
    """
    got = reference_states(x_canonical, fs, config=config)
    if got is None:
        return None
    mask, w = got
    return float(np.count_nonzero(mask)) / max(mask.size, 1), w


def reference_states(
    x_canonical: np.ndarray, fs: float, *, config: BaselineConfig
) -> tuple[np.ndarray, float] | None:
    """``(mask, rough window)`` -- the ONE wide-window pass 1 every reference reads.

    `reference_mask_fraction` counts this mask, `state_dwell_s` measures its runs and
    `tracked_intrusion` uses those runs as spans. All three are readings of the same
    measurement and they are taken from one `_rough_pass`, because it is the
    expensive part (one decimated median over the whole trace) and because three
    reference masks measured three times could disagree with each other.

    See `reference_mask_fraction` for what the reference window is and why only
    ``rough_s`` moves. ``None`` under the same conditions: a non-masked method, a
    recording too short for a reference window, or a config already running at or
    above it.
    """
    if config.method != "masked_median":
        return None
    x = np.asarray(x_canonical)
    fs = float(fs)
    w = reference_rough_s(x.size, fs)
    if w < REFERENCE_MIN_ROUGH_S or w <= float(config.rough_s):
        return None
    ref_cfg = config.replace(rough_s=w, base_s=w / ROUGH_BASE_RATIO, decim=None)
    _, mask, _, _ = _rough_pass(x, fs, config=ref_cfg)
    return mask, w


def reference_state_dwell_s(x_canonical: np.ndarray, fs: float) -> float:
    """The median state dwell at the REFERENCE window -- ``base_vs_state``'s divisor,
    measured WITHOUT a configuration.

    THE NUMBER THAT MAKES THE PROPOSER AND THE CHECKS ONE CONTRACT
    ==============================================================
    `baseline_health` divides ``base_s`` by ``state_dwell_s`` of `reference_states`'
    mask and condemns the run below `BASE_STATE_RATIO_MIN`. Until this function
    existed `baseline_probe` had no way to ask for that number before choosing
    ``base_s`` -- it sized the window off the **p99.9** of its own dwell array
    instead, which reads 1.5x higher on the -700 mV apo recording and 6.4x higher on
    ferritin_data2 -- so the proposer was free to offer a window the checks then
    condemned, deterministically and for ever. It is the same measurement here as
    there, not a second estimate of it: `reference_states` is the one implementation
    and this is `state_dwell_s` over its mask.

    Measured, this function against the ``base_vs_state`` divisor the health block
    computes at each recording's own proposed config -- **bit-identical on all six**
    recordings in evidence (-500 0.024259999421596526 s, -600 0.020719999505996702,
    -700 0.021709999482393265, ferritin_data1 0.00685, data2 0.00276, data3 0.00334),
    and ``base_s / result`` reproduces the check's rendered ratio to the last bit
    (12.366034919726197, 24.131274706608558, 1.381851714198791, 14.598540145985401,
    36.231884057971016, 2.9940119760479043).

    ``x_canonical`` must be in the CANONICAL frame (events running DOWN), like every
    other reference measurement here.

    THE CONFIG HANDED TO `reference_states` IS A TEMPLATE, and ``rough_s`` on it is
    load-bearing only in that it must sit BELOW the reference window: that function
    replaces ``rough_s`` with `reference_rough_s` before measuring anything, and its
    ``None`` guard asks "is the CONFIGURED window already the wide one?" -- a question
    about a run, which a probe that has no configuration yet cannot be asked. Passing
    `DEFAULT_BASELINE` unchanged would answer it wrongly on exactly one recording
    length (``REFERENCE_WINDOWS_MIN * DEFAULT_BASELINE.rough_s`` = 2.0 s, where the
    reference window and the app default coincide at 0.5 s), and a floor that
    silently vanishes on one duration is the fail-open this whole change removes.
    Everything else on the template -- ``mask_sigma``, ``dilate_ms``,
    ``despike_size``, ``method`` -- is `DEFAULT_BASELINE`'s, which is not a
    convenience: `baseline_probe.propose_baseline` may only ever ``replace``
    ``rough_s`` / ``base_s`` / ``decim``, so those four fields are the ones the
    proposed config will carry, and they are the ones the health block will measure
    its own reference mask with.

    Returns ``nan`` when the recording is too short for a reference window at all
    (`REFERENCE_MIN_ROUGH_S`) -- a caller must not read that as zero. Cost: one
    decimated median over the whole trace at the reference window; measured 0.088 s
    (ferritin_data1, 1.2 M samples), 0.122 (data3, 2.3 M), 0.125 (-600, 3.0 M),
    0.376 (-700, 9.1 M) and 0.525 (data2, 8.6 M).
    """
    x = np.asarray(x_canonical)
    fs = float(fs)
    w = reference_rough_s(x.size, fs)
    if not np.isfinite(w) or w < REFERENCE_MIN_ROUGH_S:
        return float("nan")
    ref = reference_states(x, fs, config=DEFAULT_BASELINE.replace(rough_s=w / 2.0))
    if ref is None:                                   # unreachable: w / 2 < w
        return float("nan")
    mask, _ = ref
    return state_dwell_s(mask, fs, dilate_ms=DEFAULT_BASELINE.dilate_ms)


def block_medians(x: np.ndarray, fs: float, block_s: float) -> tuple[np.ndarray, int]:
    """``(median of each whole block, block length in samples)``; tail dropped."""
    nb = max(1, int(round(float(block_s) * float(fs))))
    usable = (np.asarray(x).size // nb) * nb
    if usable == 0:
        return np.empty(0), nb
    return np.median(np.asarray(x)[:usable].reshape(-1, nb), axis=1), nb


def stride_for(n: int) -> int:
    """`SIGMA_STRIDE` on a long recording, 1 on a short one. One rule, one spelling."""
    return SIGMA_STRIDE if int(n) >= SIGMA_STRIDE_MIN_N else 1


def gating_spread(x: np.ndarray, fs: float) -> tuple[float, float, int]:
    """``(score, within-block noise pA, n blocks)`` -- is the OPEN LEVEL itself
    stepping?

    ``score`` is ``IQR(GATING_BLOCK_S block medians) / within-block noise``: how far
    the pore's own level moves between blocks, in units of how far it moves inside
    one. On a pore that sits at one level and is briefly blocked, a 20 ms block
    median is the open pore whatever the events do, so the IQR stays near the noise;
    on a pore stepping between conductance states the block median IS the state and
    the IQR is the step size.

    Normalised by the WITHIN-block spread rather than by a one-sided residual sigma:
    that sigma over-reads 3-4x on exactly the gating file it is being asked to detect
    (53.5 against a true 13.8 on ferritin_data3), which would drag data3's score
    below ferritin_data2's and inverting the discrimination.

    Returns ``(nan, nan, n)`` when fewer than `GATING_BLOCKS_MIN` whole blocks exist
    or the trace has no measurable within-block noise -- callers must treat that as
    NOT MEASURED, never as "not gating".

    Polarity-free: an IQR and a MAD are both unchanged by negating the trace, so this
    may be handed the raw trace or the canonical one.
    """
    x = np.asarray(x)
    fs = float(fs)
    bm, nb = block_medians(x, fs, GATING_BLOCK_S)
    if bm.size < GATING_BLOCKS_MIN:
        return float("nan"), float("nan"), int(bm.size)
    stride = stride_for(x.size)
    blocks = x[: bm.size * nb].reshape(-1, nb)
    noise = upper_mad_sigma((blocks[::stride] - bm[::stride, None]).ravel())
    iqr = float(np.subtract(*np.percentile(bm, [75, 25])))
    if not (np.isfinite(noise) and noise > 0):
        return float("nan"), float(noise), int(bm.size)
    return float(iqr / noise), float(noise), int(bm.size)


def _mask_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, ends)`` of every True run in ``mask``, as sample indices."""
    m = np.asarray(mask).view(np.int8)
    d = np.diff(np.concatenate(([np.int8(0)], m, [np.int8(0)])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def state_runs(
    mask: np.ndarray, fs: float, *, dilate_ms: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, ends)`` of the runs of ``mask`` that carry a DWELL.

    Runs no longer than the dilation kernel are dropped, and dropping them is
    load-bearing rather than tidying. The mask is a one-sided threshold GROWN by
    ``+-dilate_ms``, so a single sample of Gaussian noise past 3.5 sigma becomes a run
    of exactly ``2k+1`` samples -- 2.02 ms at the shipped 1 ms / 50 kHz. Those runs
    hold a negligible share of the masked SAMPLES and yet dominate the run COUNT
    whenever a recording has fewer events than it has noise crossings: measured on a
    5%-duty synthetic with 20 ms states, 34 real states share the mask with 64 noise
    runs, and the median run length reads **2.02 ms for a 20 ms state**. A dwell
    measured that way is a reading of ``mask_sigma`` and the trace length, not of the
    pore, and it made the state-timescale check miss a configuration whose depths
    were 2.5% of the truth (`tests/analysis/test_baseline_state_timescale.py`).

    A genuine state of duration ``d`` masks ``d + 2k`` samples, so the filter costs
    only states shorter than one sample -- which are not states.
    """
    s, e = _mask_runs(mask)
    if s.size == 0:
        return s, e
    k = max(0, int(float(dilate_ms) * 1e-3 * float(fs)))
    keep = (e - s) > (2 * k + 1)
    return s[keep], e[keep]


def state_dwell_s(mask: np.ndarray, fs: float, *, dilate_ms: float = 0.0) -> float:
    """MEDIAN masked-run length in seconds -- the state timescale of this recording.

    The number `BASE_STATE_RATIO_MIN` sizes ``base_s`` against, and it is deliberately
    the median rather than `_mask_run_p99_s`'s p99. The p99 is a proposal input ("the
    longest state ``rough_s`` must sail over") and it is dominated by the rare long
    one; what decides whether a TRACKING median rides into the states is the state it
    meets over and over. Measured on his three apo recordings the two differ by 1.5x
    (21.2 ms median against 30.8 ms p99 at -700 mV) and by 6.4x on ferritin_data2,
    whose 2.6 ms states have a 16.7 ms tail -- sizing on the tail there would demand a
    640 ms window for a recording that measures correctly at 100 ms.

    Pass the config's own ``dilate_ms`` -- see `state_runs` for the noise runs that
    otherwise set this number instead of the pore.

    ``nan`` when the mask has no runs at all; a caller must not read that as zero.

    Measure it on the REFERENCE mask (`reference_states`), not on the configured one:
    a rough window shorter than the states tracks them instead of exposing them, so
    the configured mask's runs are a reading of ``rough_s`` rather than of the pore.
    """
    s, e = state_runs(mask, fs, dilate_ms=dilate_ms)
    if s.size == 0:
        return float("nan")
    return float(np.median(e - s) / float(fs))


def tracked_intrusion(
    x_canonical: np.ndarray,
    baseline_canonical: np.ndarray,
    mask: np.ndarray,
    fs: float,
    *,
    flank_s: float = INTRUSION_FLANK_S,
    pad_s: float = INTRUSION_PAD_S,
    max_states: int = INTRUSION_MAX_STATES,
    min_depth_pA: float = 0.0,
    dilate_ms: float = 0.0,
    worst_min_dwell_s: float = 0.0,
) -> dict[str, float] | None:
    """How far the tracked baseline sits INSIDE the states, per state, directly.

    The measurement from the -700 mV report, generalised: for every run of ``mask``,
    a local open-pore reference is taken from flanking windows on either side (which
    straddle the state, so drift cancels to first order), and the baseline inside the
    state is compared against it. The headline is the median over states of

        ``(reference - baseline_inside) / (reference - trace_inside)``

    -- the fraction of the state's own excursion that the baseline followed, which is
    the fraction by which every depth measured from it reads shallow.

    **The reference is taken from the BASELINE, not from the trace.** Reading it off
    the unmasked samples of the flanks is the obvious spelling and it is biased: the
    mask is one-sided, so the survivors' median sits above the level by a truncation
    bias that is a good fraction of the noise -- measured at +4.3 pA on ferritin_data2
    against a true intrusion of 0.9 pA, i.e. it would have reported four times the
    harm actually present, on a corpus file.

    Both arrays must be in the CANONICAL frame (states running DOWN); at ``polarity
    == +1`` negate both before calling, which is exact in IEEE754.

    ``min_depth_pA`` drops states shallower than that from the median -- see
    `INTRUSION_MIN_DEPTH_SIGMA` for why a ratio of two depths needs it.

    **THE WORST STATE IS REPORTED BESIDE THE MEDIAN**, because the median is
    structurally blind to one catastrophic state and that case is in evidence. On the
    owner's -600 mV recording at 1.0 / 0.3 s the tracked baseline rides into the
    521 ms partial clog at t = 37.13 s -- ~56% of that one state's own excursion
    followed -- while the median over the other ~239 states reads exactly 0.0: one
    state, however fully entered, cannot move a median over hundreds. ``worst_*`` is
    that one state: the largest per-state ``intrusion/depth`` ratio among the states
    visited, with the dwell and start time of the state that produced it, so a caller
    ranking windows (`baseline_probe.rank_windows`) can see the failure mode the
    headline cannot. The median keys are untouched -- same arrays, same statistics,
    bit-identical to what they were.

    ``worst_min_dwell_s`` restricts WHICH states the worst is taken over, and it is
    load-bearing rather than a nicety: the per-state fraction is a ratio of two
    median differences, and on a SHORT state it is mostly noise -- ferritin_data2 at
    2.0 / 0.5 s reads 0.495 on a 2.7 ms state whose depth barely clears the filter,
    within 1% of a threshold the real failure clears by 11%. The states a tracking
    median can actually dwell inside are the ones longer than its own window, so
    `config_intrusion` passes the config's ``base_s`` and the noise readings never
    reach the criterion. Measured with the gate, the two populations separate by 9x:
    every ridden-into window in evidence reads >= 0.556 and every window that stays
    out reads <= 0.061 (or has no state long enough to qualify, reported as ``None``).
    At the default 0.0 every good state qualifies.

    Returns ``None`` when the mask has no run with a measurable depth. Otherwise a
    dict of ``fraction`` / ``intrusion_pA`` / ``depth_pA`` / ``n_states`` (medians
    over the states visited) plus ``worst_fraction`` / ``worst_dwell_s`` /
    ``worst_t_s`` (the single worst state at least ``worst_min_dwell_s`` long --
    all three ``None`` when no visited state is that long).
    """
    x = np.asarray(x_canonical)
    b = np.asarray(baseline_canonical)
    fs = float(fs)
    s, e = state_runs(mask, fs, dilate_ms=dilate_ms)
    if s.size == 0:
        return None
    if s.size > int(max_states):
        take = np.linspace(0, s.size - 1, int(max_states)).astype(np.int64)
        s, e = s[take], e[take]
    pad = max(1, int(round(float(pad_s) * fs)))
    flank = max(2, int(round(float(flank_s) * fs)))
    n = x.size

    intr = np.full(s.size, np.nan)
    depth = np.full(s.size, np.nan)
    # One short Python loop over STATES (hundreds), not over samples -- the spans are
    # ragged. 1193 states on the 9 M-sample -700 mV recording cost 31 ms.
    for i in range(s.size):
        a, c = int(s[i]), int(e[i])
        parts = []
        for q0, q1 in ((max(0, a - pad - flank), max(0, a - pad)),
                       (min(n, c + pad), min(n, c + pad + flank))):
            if q1 - q0 > 2:
                parts.append(float(np.median(b[q0:q1])))
        if not parts or c <= a:
            continue
        ref = float(np.mean(parts))
        intr[i] = ref - float(np.median(b[a:c]))
        depth[i] = ref - float(np.median(x[a:c]))

    good = np.isfinite(depth) & (depth > max(0.0, float(min_depth_pA)))
    if not good.any():
        return None
    frac = intr[good] / depth[good]
    gs, ge = s[good], e[good]
    out = {
        "fraction": float(np.median(frac)),
        "intrusion_pA": float(np.median(intr[good])),
        "depth_pA": float(np.median(depth[good])),
        "n_states": int(good.sum()),
        "worst_fraction": None,
        "worst_dwell_s": None,
        "worst_t_s": None,
    }
    # The single worst state, by the same ratio the median is taken over, among the
    # states at least ``worst_min_dwell_s`` long -- see the docstring for why it
    # travels beside the median and why the short states are excluded from it.
    long_enough = (ge - gs) / fs > float(worst_min_dwell_s)
    if long_enough.any():
        idx = np.flatnonzero(long_enough)
        worst = int(idx[np.argmax(frac[idx])])
        out["worst_fraction"] = float(frac[worst])
        out["worst_dwell_s"] = float((int(ge[worst]) - int(gs[worst])) / fs)
        out["worst_t_s"] = float(int(gs[worst]) / fs)
    return out


def _mask_events(
    x: np.ndarray, baseline: np.ndarray, *, fs: float, polarity: int, k: float, pad_s: float
) -> np.ndarray:
    """Boolean mask of rough event samples, dilated by a guard band.

    Deliberately self-contained rather than calling the detector: detection asks
    *this* module for its baseline, so importing it here would be circular. This
    is a crude first-pass threshold, not a substitute for detection.
    """
    dev = polarity * (x - baseline)
    # MAD survives up to 50% event occupancy; std would be inflated by the very
    # events we are trying to find and would mask nothing.
    sigma = 1.4826 * np.median(np.abs(dev - np.median(dev)))
    if not (sigma > 0):
        return np.zeros(x.size, dtype=bool)
    mask = dev > k * sigma
    pad = max(1, int(round(pad_s * fs)))
    return maximum_filter1d(mask.view(np.uint8), size=2 * pad + 1).view(bool)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def estimate_baseline(
    current: np.ndarray,
    fs: float,
    *,
    polarity: int | None = None,
    window_s: float | None = None,
    method: str = "block_median",
    two_pass: bool = False,
    min_dwell_s: float = 2e-5,
    percentile: float = 80.0,
    asls_lam: float = 6e4,
    asls_p: float = 1e-3,
    asls_niter: int = 10,
    asls_decim: float = 0.1,
    two_pass_k: float = 4.0,
    two_pass_pad_s: float = 1e-4,
    sg_poly: int = SAVGOL_POLY,
    sg_decim: int = SAVGOL_DECIM,
    rough_s: float = 0.5,
    base_s: float = 0.1,
    decim: int | None = None,
    mask_sigma: float = 3.5,
    dilate_ms: float = 1.0,
) -> np.ndarray:
    """Estimate the time-varying open-pore current I0(t), one value per sample.

    Args:
        current: The current trace (1-D array, pA). float32 in is fine; it is
            upcast to float64 internally.
        fs: Sample rate in Hz. Required for every method except 'constant'
            (dwell and window widths are meaningless without a rate); a clear
            error is raised when it is ``None``.
        polarity: The factor making ``dev = polarity*(current-baseline)``
            positive-in-event, in ``{-1, +1}``. Decides which side I0 sits on.
            ``None`` measures it via `derive_polarity` (never assumed).
        window_s: Smoothing window in seconds. **Must be >> the longest dwell**
            (>= 20 * min_dwell_s is enforced with a warning). Too short and the
            baseline dives into every blockade and eats it; too long and real
            drift is not tracked. Unused by 'masked_median', which has its own
            two windows.

            ``None`` (the default) means "this method's own default": 10 ms for
            the block-based methods -- frozen, it is what every existing caller and
            stored result was measured with -- `DEFAULT_BASELINE.window_s`
            (0.5 s) for 'robust_median', whose 10 ms rolling median would be ELEVEN
            taps at decim 50 and would track the events rather than reject them, and
            `SAVGOL_WINDOW_S` (0.2 s) for 'savgol', which is the Cu recipe's own
            measured value. 10 ms there would be four taps at decim 50.
        method: One of:
            'block_median' -- median per non-overlapping window_s block, then
                Pchip back to full rate. Fast, no overshoot at drift steps -- and
                blind once event occupancy inside a block passes 50%, which on the
                owner's own recordings it routinely does. Kept as the default of
                this function (every stored result and every existing test depends
                on it) but NOT the default of `BaselineConfig`, which is what the
                app now characterises with.
            'masked_median' -- the event-masked two-pass estimator (see the module
                docstring): rough pass exposes the events, they are masked and
                interpolated across, a short pass tracks the drift. Takes
                ``rough_s``/``base_s``/``decim``/``mask_sigma``/``dilate_ms``;
                ignores ``window_s``. Imported verbatim from `ml_preprocess`.
            'robust_median' -- the single-pass sibling: one decimated rolling
                median at ``window_s``, events rejected only by the median's own
                breakdown point. One window has to be simultaneously long enough to
                reject events and short enough to track drift, which is exactly the
                compromise 'masked_median' removes.
            'percentile'   -- as above but a high percentile on the open-pore
                side of the signal (see `percentile`). Use when event occupancy
                is high enough (>~30%) to drag the median toward the events.
            'asls'         -- asymmetric least squares on the block knots; a
                smooth global baseline that ignores one-sided excursions. Costs
                more; `asls_lam` is in block units. Note it is deliberately
                biased a little toward the open-pore side by construction.
            'savgol'       -- Savitzky-Golay polynomial baseline fitted on a
                DECIMATED copy (stride ``sg_decim``) and interpolated back, the Cu2
                -insulin recipe. Takes ``window_s``/``sg_poly``/``sg_decim``; ignores
                the percentile/asls knobs. It does NOT mask, which is why it is the
                one that survives a recording whose baseline is unstable rather than
                event-rich: a mask treats instability as events and interpolates
                across it, and the instability then reappears as huge false events.
                It bows around deep excursions in exchange (measured qc_overshoot
                0.13 -> 0.37 on cu_30h), so it suits DEEP AND RARE events, not the
                shallow-and-abundant kind. NOTE the name collision with
                `filters.savgol`, which is the SMOOTHER (it returns the smoothed
                signal); this is the ESTIMATOR (the smoothed signal is subtracted).
            'constant'     -- global median. Only honest if there is no drift.
        two_pass: Event-immune mode. Pass 1 estimates a baseline and roughly
            thresholds events off it; pass 2 re-estimates with those samples
            linearly interpolated across, so the events cannot pull on it. Cheap
            insurance when events are long, deep, or dense. It cannot rescue a
            window_s that is far too short -- pass 1 has to see the events first.
        min_dwell_s: Shortest dwell you intend to keep. Only used to police
            window_s. Keep it consistent with what you pass to the detector.
        percentile: For method='percentile', the percentile *toward the open-pore
            side*, already sign-corrected for polarity. 80 means "the level 80%
            of the way toward open pore". Not a raw numpy percentile.
        asls_lam, asls_p, asls_niter: Eilers-Boelens smoothness, asymmetry, and
            iteration count. `asls_p` is small (heavily one-sided) on purpose so
            the fit ignores events even at high occupancy.
        asls_decim: asls knot spacing as a fraction of window_s. Finer than 1 so
            some knots fall between dense events; smoothness comes from asls_lam,
            not this.
        two_pass_k: Pass-1 threshold in robust sigmas.
        two_pass_pad_s: Guard band grown around each masked event, so the event's
            filtered shoulders are excluded too.
        sg_poly: 'savgol' polynomial degree (`SAVGOL_POLY`). Only used by 'savgol'.
        sg_decim: 'savgol' decimation stride (`SAVGOL_DECIM`). The window is measured
            in seconds and converted ON THE DECIMATED GRID, so this changes the tap
            count, not the width: 0.2 s at 50 kHz is 201 taps at stride 50. Only used
            by 'savgol'.
        rough_s: 'masked_median' pass-1 window, seconds -- sets which excursions
            become "events". See `BaselineConfig`.
        base_s: 'masked_median' pass-2 window, seconds -- sets how fast the
            baseline may track.
        decim: Shared stride for the two median methods; ``None`` derives it from
            the tracking window via `derive_decim`. Pin it (50 / 50 / 10) to
            reproduce the ferritin corpus.
        mask_sigma: 'masked_median' mask threshold in robust sigmas.
        dilate_ms: 'masked_median' mask growth on each side, milliseconds.

    Returns:
        float64 array, `len == current.size`, in pA.

    Raises:
        ValueError: on a non-finite trace, a missing/invalid rate, a non-positive
            or over-long window, a recording shorter than the tracking window, or
            an unknown method. A silently wrong baseline is worse than a crash.

    Note:
        'masked_median' and 'robust_median' refuse ``two_pass=True``: the first IS
        a two-pass estimator with two decoupled windows, and the legacy ``two_pass``
        flag is a *different* algorithm (block median + Pchip, one window for both
        passes, k = 4.0, 0.1 ms pad). Silently running one inside the other would
        produce a baseline neither of them describes.
    """
    x = require_finite(current)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method in MASKED_METHODS:
        if two_pass:
            raise ValueError(
                f"method={method!r} does not take two_pass=True -- 'masked_median' is "
                "already a two-pass estimator with two decoupled windows, and the legacy "
                "two_pass flag is a different algorithm (block median + Pchip, one window "
                "for both passes). Pick one."
            )
        cfg = BaselineConfig(
            method=method,
            rough_s=float(rough_s),
            base_s=float(base_s),
            decim=decim,
            mask_sigma=float(mask_sigma),
            dilate_ms=float(dilate_ms),
            window_s=DEFAULT_BASELINE.window_s if window_s is None else float(window_s),
        )
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"fs must be a positive number of Hz, got {fs!r}")
        _require_length(x.size, float(fs), cfg)
        pol = derive_polarity(x) if polarity is None else require_polarity(polarity)
        return _kernel_baseline(x, float(fs), polarity=pol, config=cfg)

    if window_s is None:
        # 'savgol' is NEW, so no stored result depends on it inheriting the frozen
        # 10 ms block-median default -- and 10 ms would be four taps at decim 50,
        # an estimator that tracks the events. It gets the Cu recipe's own 0.2 s.
        window_s = SAVGOL_WINDOW_S if method == "savgol" else _LEGACY_WINDOW_S
    else:
        window_s = float(window_s)
    if method != "constant":
        if fs is None:
            raise ValueError(
                "estimate_baseline requires a real sample rate (fs); the window and "
                "dwell widths are undefined without one. This trace has no time base "
                "(sample_rate_hz is None). Use method='constant' if that is what you meant."
            )
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"fs must be a positive number of Hz, got {fs!r}")
        if not np.isfinite(window_s) or window_s <= 0:
            raise ValueError(f"window_s must be a positive number of seconds, got {window_s!r}")
        if min_dwell_s <= 0:
            raise ValueError(f"min_dwell_s must be positive, got {min_dwell_s!r}")
        if window_s < MIN_WINDOW_TO_DWELL * min_dwell_s:
            _warn_baseline(
                f"window_s={window_s * 1e3:g} ms is only "
                f"{window_s / min_dwell_s:.1f}x min_dwell_s={min_dwell_s * 1e6:g} us "
                f"(want >= {MIN_WINDOW_TO_DWELL:g}x). The baseline will track the events "
                "and partly cancel them: depths come out shallow and short events "
                "disappear entirely. Widen window_s, or check "
                "baseline_intrusion() before believing any of this.",
                stacklevel=2,
            )
    if not (0.0 < percentile < 100.0):
        raise ValueError(f"percentile must be in (0, 100), got {percentile!r}")

    pol = derive_polarity(x) if polarity is None else require_polarity(polarity)

    kw = dict(
        fs=float(fs) if fs is not None else float("nan"),
        polarity=pol,
        window_s=float(window_s),
        method=method,
        percentile=float(percentile),
        asls_lam=float(asls_lam),
        asls_p=float(asls_p),
        asls_niter=int(asls_niter),
        asls_decim=float(asls_decim),
        sg_poly=int(sg_poly),
        sg_decim=int(sg_decim),
    )

    check_cancelled()
    b = _single_pass(x, **kw)
    if not two_pass:
        return b

    check_cancelled()
    mask = _mask_events(x, b, fs=float(fs), polarity=pol, k=two_pass_k, pad_s=two_pass_pad_s)
    good = ~mask
    if good.sum() < 2:
        raise ValueError(
            "two_pass masked essentially the whole trace as events; the pass-1 baseline "
            "or two_pass_k is wrong, or this is not a nanopore recording."
        )
    if not mask.any():
        return b
    idx = np.arange(x.size)
    clean = x.astype(np.float64, copy=True)
    clean[mask] = np.interp(idx[mask], idx[good], clean[good])
    check_cancelled()
    return _single_pass(clean, **kw)


# ---------------------------------------------------------------------------
# health -- the checks the reference algorithm does not perform
# ---------------------------------------------------------------------------


def _require_length(n: int, fs: float, config: BaselineConfig) -> None:
    """REFUSE a trace shorter than the tracking window. Nothing else does.

    The reference returns silent garbage here and never raises: at n <= 50 with
    decim 50 the "baseline" is literally ONE SAMPLE broadcast over the recording
    (n = 2/3/5 -> one decimated sample, mask 0.0, a constant); at n = 11 the mask
    comes back 1.0000 and the interpolation is skipped entirely; at n = 51/101 the
    constant is 3.7 pA off. Running THESE PARAMETERS on a trace this short is
    therefore a refusal, not a warning.

    It is a refusal about the PARAMETERS, not about the recording. `adapt_to_length`
    is the measured answer to "what can this many samples support?", and
    `characterisation_baseline` -- the app's door -- runs it before ever reaching
    here, so no legal document is refused because of its length. This function is
    the estimator's own strict contract: the config it was handed, unaltered, or an
    error. Nothing is silently re-tuned behind a caller who asked for a window.

    ``'constant'`` is exempt. It is a global median; it has no window, tracks
    nothing, and ``window_s`` is carried on the config only so the five methods
    share one object. Refusing a 200-sample trace because a window it never reads
    does not fit was a plain defect -- `estimate_baseline` has always answered that
    call, so the two entry points disagreed on the same method.
    """
    if config.method == "constant":
        return
    win = config.tracking_window_s
    need = int(math.ceil(win * float(fs)))
    if n < need:
        fitted, _ = adapt_to_length(config, n, float(fs))
        raise ValueError(
            f"recording is {n} samples ({n / float(fs):g} s); the "
            f"{'base_s' if config.method == 'masked_median' else 'window_s'} tracking "
            f"window is {win:g} s = {need} samples. Below one full window the estimator "
            "degenerates to a constant (the reference returns one sample broadcast over "
            f"the whole trace, with no error). What {n} samples CAN support is "
            f"{fitted.describe()}; `characterisation_baseline` runs that automatically "
            "and stamps the adaptation into provenance. Analyse more signal, or pass "
            "that config explicitly if you meant to measure this short a trace."
        )


@dataclass(frozen=True)
class BaselineCheck:
    """One health check, rendered as a row rather than logged as a line.

    ``level`` is ``'ok'`` | ``'warn'`` | ``'refuse'``. A ``'refuse'`` check never
    reaches here as data -- the corresponding condition raises before a baseline
    exists -- it is in the vocabulary so the GUI can describe the whole check set,
    including the ones that would have stopped the run.
    """

    name: str
    level: str
    label: str
    value: float | None
    threshold: str
    message: str

    @property
    def ok(self) -> bool:
        return self.level == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "level": self.level, "label": self.label,
            "value": None if self.value is None else float(self.value),
            "threshold": self.threshold, "message": self.message,
        }


@dataclass(frozen=True)
class BaselineHealth:
    """Structured verdict on a baseline estimate -- for the GUI to render, not to log.

    Two families of number live here and they must not be confused:

    **The published QC block** (``noise_pA``, ``open_pore_level_pA``,
    ``open_pore_drift_pA``, ``overshoot``) is the notebook's own, reproduced
    verbatim so it matches the stored ``*_metadata.json`` exactly. Three of the four
    are **reports, never objectives**: a degenerate ``rough_s=0.01 / base_s=0.002 /
    decim=1`` beats the shipped parameters on level, drift AND noise for every corpus
    file, because shrinking the window monotonically improves all three -- optimising
    them selects the shortest possible window, which is precisely the failure this
    estimator exists to fix. Only ``overshoot`` resists.

    **The health checks** are the ones the reference does not perform, and they
    penalise the real failure modes: a mask that found nothing, a mask that swallowed
    everything, a rough pass that is measuring drift instead of events, and a
    recording too short for its own window.

    All of those compare a number against a CONSTANT, and three checks exist because
    constants could not see the failures they are for:

    * ``mask_reference`` -- an estimate whose mask fraction is squarely inside its
      band while three quarters of the events are missing from it. It compares this
      recording's occupancy against ITSELF (the same mask, over a window wide enough
      to sail over the states) and is the only check that can tell a per-event depth
      of 7.8 pA from one of 110. See `MASK_REFERENCE_MAX_RATIO`.
    * ``base_vs_state`` -- the six checks above ``mask_reference`` are all about
      ``rough_s``, which decides *which excursions are events*; this one is about
      ``base_s``, which decides *how fast the baseline may follow*, and nothing else
      here looks at it against the EVENT timescale. See `BASE_STATE_RATIO_MIN`.
    * ``baseline_intrusion`` -- the same question asked of the answer rather than of
      the parameters: how far the tracked baseline actually sits inside the states,
      per state, against the open pore beside each one. See `BASELINE_INTRUSION_MAX`.
    * ``level_occupancy`` -- and this one is not about a constant, nor about a second
      measurement of the same thing; it is about the FRAME. Every check above reads the
      trace canonicalised by a polarity derived from the same trace, so a recording
      that is mostly BLOCKED flips all of them together and they then agree, in green,
      about a baseline sitting on the blockade. This row is ``|current - baseline|``
      and a histogram of the residual, two-sided and frame-free. It REPORTS the margin
      the open-pore identification rests on instead of claiming to have verified it --
      the two readings are the same trace, measured -- and fails where the estimator's
      own majority rule has no majority (`LEVEL_OCCUPANCY_MAJORITY`), had no decisive
      one (`LEVEL_DECISIVE_RATIO`), or ran under `LEVEL_OCCUPANCY_MIN` on a level the
      recording never HOLDS for one tracking window (`LEVEL_HOLD_MIN_WINDOWS`) -- the
      last being the only line in this block that reaches a baseline sitting on the
      blockade of a pore that is blocked ~80% of the time, and it reaches it as a
      conjunction because neither half separates that estimate from a valid one.
      Whenever the recording has a SECOND level at all, or the margin is under
      `LEVEL_OCCUPANCY_MIN`, the RESULT says so in `level_occupancy_warning`.

    The last two decline to judge on a pore whose OPEN LEVEL steps (`gating_spread`
    above `GATING_THRESHOLD`), where a tracking window shorter than the long states
    is the point rather than the defect.

    They do NOT decline by disappearing. Both need a pass wide enough to sail over
    the states, and when that pass finds nothing they are emitted as ``warn`` saying
    so in as many words -- because a check missing from the block is not a check that
    passed, and a reader counting green bullets cannot tell the two apart. Measured:
    a pore that steps +-150 pA every 80 ms and ALSO has 20 ms translocations gets a
    proposal of 0.03/0.01 from the app's own probe, and used to render six green rows
    and ``ok`` True on a configuration whose per-event depth read 5.6 pA against a
    true 100.0.

    ``mask_*`` and ``rough_sigma_*`` are ``None`` for the non-masked methods, which
    have no pass 1.
    """

    checks: tuple[BaselineCheck, ...] = ()
    method: str = ""
    n_samples: int = 0
    fs: float = float("nan")
    polarity: int = -1
    # -- masked-median diagnostics (None for the block/robust methods) --------
    mask_fraction: float | None = None
    mask_fraction_event_free: float | None = None
    mask_fraction_reference: float | None = None
    mask_reference_rough_s: float | None = None
    mask_reference_ratio: float | None = None
    mask_run_p99_s: float | None = None
    rough_sigma_pA: float | None = None
    upper_mad_sigma_pA: float | None = None
    rough_sigma_ratio: float | None = None
    # -- the state timescale, and the window measured against it ---------------
    state_dwell_s: float | None = None
    base_state_ratio: float | None = None
    intrusion_fraction: float | None = None
    intrusion_pA: float | None = None
    intrusion_depth_pA: float | None = None
    intrusion_n_states: int | None = None
    gating_score: float | None = None
    gating_states: bool | None = None
    #: Fraction of the recording within `INTRUSION_MIN_DEPTH_SIGMA` of the tracked
    #: baseline -- the ``level_occupancy`` check's number, and the one quantity here
    #: measured in no frame at all. Every estimate carries it; ``None`` only on a
    #: health block restored from a project written before it existed, which is why
    #: `BaselineResult.warnings` tests it for None rather than assuming it.
    #: See `LEVEL_OCCUPANCY_MIN`.
    level_occupancy: float | None = None
    #: The OTHER level of the recording's own amplitude histogram: its signed offset
    #: from the tracked baseline, and the fraction of the recording it holds. Also
    #: measured in no frame (`alternative_level`). ``None`` / ``0.0`` when the trace has
    #: no second level, and ``None`` on a health block restored from a project written
    #: before these existed -- which is why `BaselineResult.warnings` tests for None
    #: rather than assuming. See `ALT_LEVEL_OCCUPANCY_MIN`.
    alt_level_offset_pA: float | None = None
    alt_level_occupancy: float | None = None
    #: The longest 1% of the unbroken stretches the trace spends AT the tracked level,
    #: expressed in tracking windows (p99 of those run lengths / ``tracking_window_s``).
    #: Frame-free like the two above -- it is read off ``|current - baseline|`` -- and
    #: it is the second half of the ``level_occupancy`` row's third failing line: below
    #: `LEVEL_OCCUPANCY_MIN` a level the recording never holds for one window was voted
    #: for rather than measured. ``None`` on a health block restored from a project
    #: written before it existed. See `LEVEL_HOLD_MIN_WINDOWS`.
    level_hold_windows: float | None = None
    # -- the published QC block ----------------------------------------------
    noise_pA: float = float("nan")
    open_pore_level_pA: float = float("nan")
    open_pore_drift_pA: float = float("nan")
    open_pore_drift_sigma: float = float("nan")
    overshoot: float = float("nan")

    @property
    def ok(self) -> bool:
        """True when no check is warning or refusing."""
        return all(c.ok for c in self.checks)

    @property
    def warnings(self) -> tuple[str, ...]:
        """The messages of every non-ok check, in check order."""
        return tuple(c.message for c in self.checks if not c.ok)

    def check(self, name: str) -> BaselineCheck | None:
        for c in self.checks:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-serialisable. Goes into provenance under ``baseline_health``."""
        return {
            "ok": self.ok,
            "method": self.method,
            "n_samples": int(self.n_samples),
            "fs": float(self.fs),
            "polarity": int(self.polarity),
            "mask_fraction": self.mask_fraction,
            "mask_fraction_ok": None if self.mask_fraction is None else bool(
                MASK_FRACTION_FLOOR <= self.mask_fraction <= MASK_FRACTION_CEILING
            ),
            "mask_fraction_event_free": self.mask_fraction_event_free,
            "mask_fraction_reference": self.mask_fraction_reference,
            "mask_reference_rough_s": self.mask_reference_rough_s,
            "mask_reference_ratio": self.mask_reference_ratio,
            "mask_run_p99_s": self.mask_run_p99_s,
            "state_dwell_s": self.state_dwell_s,
            "base_state_ratio": self.base_state_ratio,
            "intrusion_fraction": self.intrusion_fraction,
            "intrusion_pA": self.intrusion_pA,
            "intrusion_depth_pA": self.intrusion_depth_pA,
            "intrusion_n_states": self.intrusion_n_states,
            "gating_score": self.gating_score,
            "gating_states": self.gating_states,
            "level_occupancy": self.level_occupancy,
            "level_hold_windows": self.level_hold_windows,
            "alt_level_offset_pA": self.alt_level_offset_pA,
            "alt_level_occupancy": self.alt_level_occupancy,
            "rough_sigma_pA": self.rough_sigma_pA,
            "upper_mad_sigma_pA": self.upper_mad_sigma_pA,
            "rough_sigma_ratio": self.rough_sigma_ratio,
            "qc_mad_noise_pA": float(self.noise_pA),
            "qc_open_pore_level_pA": float(self.open_pore_level_pA),
            "qc_open_pore_drift_pA": float(self.open_pore_drift_pA),
            "qc_open_pore_drift_sigma": float(self.open_pore_drift_sigma),
            "qc_overshoot": float(self.overshoot),
            "warnings": list(self.warnings),
            "checks": [c.to_dict() for c in self.checks],
        }


def _mask_run_p99_s(mask: np.ndarray, fs: float) -> float:
    """p99 of the masked RUN lengths, seconds -- the longest state the mask sees.

    Feeds the proposal rule ("``rough_s`` >= ~4x the longest dwell you want treated
    as an event"), and it is the honest way to measure a dwell distribution without
    a detector. Two caveats, both measured, both load-bearing:

    * it is computed on the DILATED mask, which closes gaps up to ``2*dilate_ms``
      and no further. A shallow long state at only ~4.5 sigma depth fragments into
      runs of <= 26 ms (83% of its samples cross the threshold, 17% do not), so on a
      recording like that this UNDERSTATES the longest state by more than 20x. It is
      a metric with no threshold attached for exactly that reason.
    * it is circular by construction: measuring dwell needs a baseline, which needs
      the dwell. Iterate from a long probe if you want to converge it.
    """
    starts, ends = _mask_runs(mask)
    if starts.size == 0:
        return 0.0
    return float(np.percentile(ends - starts, 99) / float(fs))


def _qc_block(
    detr: np.ndarray, fs: float
) -> tuple[float, float, float, float, float, int, int]:
    """``dataLoader.ipynb``'s own QC block, verbatim. ``detr`` must be CANONICAL (events down).

    Returns ``(noise_pA, level_pA, drift_pA, drift_sigma, overshoot, n_up, n_down)``
    -- the last two being the raw +-4 sigma counts ``overshoot`` is the ratio of,
    carried out so a warning can say whether it is describing a real event
    population or forty samples of noise tail.

    The 3.5 sigma / 1.0 ms / 0.5 s / 4 sigma constants here are the PUBLISHED QC
    definition and are deliberately NOT taken from the config: these four numbers
    have to keep matching the stored ``*_metadata.json`` to every printed digit, and
    they would stop doing so the moment they started tracking a user's mask_sigma.

    Units trap: ``open_pore_drift`` is reported in pA here; the notebook PRINTS it in
    sigma. data3's 0.410 pA and its "0.03 sigma" are the same number.

    Dtype trap, and it is not cosmetic: the notebook runs this on the **float32**
    detrended column, so every median, every reshape-median and ``bm.std()``
    accumulate in float32. Widening ``detr`` to float64 first moves
    ``qc_mad_noise_pA`` on ferritin_data1 from the stored 6.752663746261597 to
    6.752663725549914 and ``qc_open_pore_drift_pA`` from 0.9654889106750488 to
    0.965488935935769 -- immaterial physically, but it is the difference between
    matching the stored metadata exactly and not. ``detr`` is therefore used with
    its dtype untouched.
    """
    d = np.asarray(detr)
    noise = float(1.4826 * np.median(np.abs(d - np.median(d))))
    evm = binary_dilation(d < np.median(d) - 3.5 * noise,
                          iterations=max(1, int(1e-3 * float(fs))))
    op = d[~evm]
    w = int(0.5 * float(fs))
    m = (len(op) // w * w) if w else 0
    bm = np.median(op[:m].reshape(-1, w), axis=1) if m else np.array([0.0])
    n_up = int((d > 4 * noise).sum())
    n_down = int((d < -4 * noise).sum())
    over = float(n_up / max(n_down, 1))
    drift = float(bm.std())
    return (noise, float(np.median(op)) if op.size else float("nan"),
            drift, drift / noise if noise else float("nan"), over, n_up, n_down)


def alternative_level(
    detr: np.ndarray,
    noise: float,
    *,
    min_depth_sigma: float = INTRUSION_MIN_DEPTH_SIGMA,
) -> tuple[float | None, float]:
    """The OTHER level of the recording's own amplitude histogram, measured in NO frame.

    ``detr`` is ``current - baseline`` -- the residual, signed, exactly as
    `baseline_health` already has it -- and ``noise`` its MAD sigma. Returns
    ``(offset_pA, occupancy)``:

    * ``offset_pA`` -- the densest amplitude among the samples that are further than
      ``min_depth_sigma`` sigma from the tracked baseline, i.e. the second mode of the
      amplitude histogram, as a signed offset FROM the tracked level. ``None`` when
      nothing is that far from the baseline at all.
    * ``occupancy`` -- the fraction of the WHOLE recording inside a band around that
      mode. The band is half-width ``min(min_depth_sigma * noise, |offset| / 2)``, so
      it can never reach the tracked level's own band and count baseline samples as
      members of the other level: on the owner's -1000 mV 0000 file the second mode is
      only 7 sigma out, and a fixed 5-sigma band there reads 0.948 (it is counting the
      baseline) against 0.087 for this one.

    NO POLARITY ENTERS THIS FUNCTION, and that is the whole point of it. Every other
    statistic in the health block is read through the canonical frame, the frame is
    chosen by a polarity derived from the same trace, and on a mostly-blocked pore they
    therefore flip together and agree in green about a baseline sitting on the blockade
    (`OVERSHOOT_FLAG`). A signed histogram of the residual has nothing for a flip to
    cancel: negating ``detr`` negates ``offset_pA`` and leaves ``occupancy`` bit for
    bit unchanged.

    WHAT THIS IS NOT. It does not say which of the two levels is the open pore, and no
    version of it can. Four candidates for that were measured against the owner's own
    recordings and each one died on them, which is why this function reports a level
    instead of a verdict:

    * **which level is further from zero** (the pore physics: a blockade lowers |I|).
      On all four of the owner's ABFs the excursions run AWAY from zero -- baseline
      -34.6 pA with a second level at -120.7 pA on the -600 mV file, and the same on
      -700 and both -1000 mV files -- so this criterion condemns his entire corpus.
    * **the sharpness of the second mode** -- 0.913 of the off-level samples inside one
      5-sigma band on the inverted fixture against 0.912 on
      ``test_baseline_occupancy_reference``'s own VALID renewal fixture. It separates
      constant-depth event populations from varied-depth ones, not inverted estimates
      from valid ones.
    * **the agreement of the per-excursion depths** (are the excursions one depth or a
      spread?) -- 0.988 on the inverted fixture, 0.971 and 0.906 on the owner's -1000
      mV 0004 and -700 mV files. No line separates them.
    * **the occupancy of the second level itself** -- 0.191 / 0.094 / 0.054 / 0.028
      across the inversion ladder against 0.054 / 0.099 / 0.087 / 0.077 on the corpus:
      the same numbers.

    The two readings of a two-level trace are the same trace up to which level you call
    open, and that is not in the samples. `ALT_LEVEL_OCCUPANCY_MIN` is where what this
    function CAN honestly support is written down.
    """
    d = np.asarray(detr)
    n = int(d.size)
    if n == 0 or not np.isfinite(noise) or noise <= 0.0:
        return None, 0.0
    cut = float(min_depth_sigma) * float(noise)
    deep = np.abs(d) > cut
    if not deep.any():
        return None, 0.0
    dv = d[deep]
    lo, hi = float(np.min(dv)), float(np.max(dv))
    if not (hi > lo):
        offset = lo
    else:
        # One bin per sigma: the resolution at which two levels are distinguishable at
        # all, and the same scale every other row in the block is measured on.
        bins = int(np.clip(np.ceil((hi - lo) / noise), 8, 8192))
        counts, edges = np.histogram(dv, bins=bins, range=(lo, hi))
        j = int(np.argmax(counts))
        centre = 0.5 * (float(edges[j]) + float(edges[j + 1]))
        # Refine off the samples in that neighbourhood rather than trusting the bin
        # centre: a median is what the rest of this module reads a level with.
        near = np.abs(dv - centre) <= cut
        offset = float(np.median(dv[near])) if near.any() else float(centre)
    if not np.isfinite(offset) or offset == 0.0:
        return None, 0.0
    half = min(cut, 0.5 * abs(offset))
    occupancy = float(np.count_nonzero(np.abs(d - offset) <= half)) / n
    return float(offset), occupancy


def baseline_health(
    current: np.ndarray,
    baseline: np.ndarray,
    fs: float,
    *,
    config: BaselineConfig = DEFAULT_BASELINE,
    polarity: int | None = None,
    mask: np.ndarray | None = None,
    rough_sigma_pA: float | None = None,
    upper_mad_sigma_pA: float | None = None,
) -> BaselineHealth:
    """Everything the reference algorithm cannot tell you about its own answer.

    The reference has no guards at all. It returns a baseline of -299.6 pA for a
    -100 pA open pore at 95% blockade duty; it returns a finite, 12 pA-wrong answer
    with the event mask silently disabled if a NaN reaches it; it returns a mask
    fraction of exactly 0.0000 under 1 Hz drift while the median baseline error hits
    30.6 pA. Every one of those is caught here.

    Pass ``mask``/``rough_sigma_pA``/``upper_mad_sigma_pA`` from `_rough_pass` when
    you already have them (`estimate_baseline_result` does); otherwise pass 1 is
    recomputed, which roughly doubles the cost of the whole estimate.

    Returns a `BaselineHealth`, never raises for a *warning* condition -- the two
    refusal conditions (non-finite input, recording shorter than the tracking
    window) have already raised before a baseline could exist.
    """
    x = np.asarray(current)
    b = np.asarray(baseline, dtype=np.float64)
    if b.shape != x.shape:
        raise ValueError(f"baseline shape {b.shape} != current shape {x.shape}")
    fs = float(fs)
    pol = derive_polarity(x) if polarity is None else require_polarity(polarity)
    n = x.size
    # The notebook's own chain: detrend -> float32 -> [despike] -> QC. The float32
    # cast lands on the BASELINE before the subtraction there; subtracting a float64
    # array holding those same float32 values and rounding once is byte-identical
    # (the exact difference of two float32 values is representable in float64), so
    # this reproduces ``current_detr_pA`` and therefore the stored QC scalars.
    detr = (x - b).astype(np.float32)
    if config.despike_size:
        detr = despike_median(detr, size=config.despike_size)
    # CANONICAL frame: events run DOWN, which is the frame the kernel, the mask and
    # the QC block are all written in. At pol == -1 this is a no-op; at +1 the
    # negation is exact in IEEE754 and the despike commutes with it (an odd-window
    # median is an order statistic).
    detr_canon = detr if pol == -1 else -detr

    checks: list[BaselineCheck] = []

    # -- length ------------------------------------------------------------
    win = config.tracking_window_s
    checks.append(BaselineCheck(
        "length_tracking", "ok", "recording vs tracking window",
        float(n) / (win * fs), f">= 1.0 x {win:g} s",
        f"{n / fs:g} s of signal against a {win:g} s tracking window.",
    ))
    if config.method == "masked_median":
        ratio = float(n) / (float(config.rough_s) * fs)
        level = "ok" if ratio >= 1.0 else "warn"
        checks.append(BaselineCheck(
            "length_rough", level, "recording vs rough window", ratio,
            f">= 1.0 x rough_s = {config.rough_s:g} s",
            f"{n / fs:g} s of signal against a {config.rough_s:g} s rough window."
            if level == "ok" else
            f"the recording ({n / fs:g} s) is SHORTER than rough_s "
            f"({config.rough_s:g} s), so pass 1 sees one reflected window and the mask "
            "is arbitrary: measured mask fractions of 1.0000, 0.4520 and 0.0902 for "
            "rough windows of 0.001, 0.01 and 0.1 s on the same trace, with no error "
            "at any length. Shorten rough_s or analyse more signal.",
        ))

    # -- the published QC block -------------------------------------------
    check_cancelled()
    # Liveness beside the cancel polls (see `baseline_progress_scope`): local
    # fractions ordered by this block's measured cost split at the 50 M-sample scale
    # (QC 2.8 s, wide reference pass 2.5 s, edge/level rows the remainder of 7.8 s).
    _report_baseline_progress(0.0, "baseline health (QC block)")
    noise, level_pA, drift_pA, drift_sigma, over, n_up, n_down = _qc_block(detr_canon, fs)

    # -- mask diagnostics (masked_median only) -----------------------------
    mask_frac = mask_free = run_p99 = rough_ratio = None
    ref_frac = ref_rough = ref_ratio = None
    dwell = base_ratio = g_score = None
    gating = intr = None
    if config.method == "masked_median":
        if mask is None or rough_sigma_pA is None or upper_mad_sigma_pA is None:
            _, mask, rough_sigma_pA, upper_mad_sigma_pA = _rough_pass(
                x if pol == -1 else -x, fs, config=config
            )
        mask_frac = float(np.count_nonzero(mask)) / max(n, 1)
        run_p99 = _mask_run_p99_s(mask, fs)
        k = max(1, int(float(config.dilate_ms) * 1e-3 * fs))
        # Phi(-mask_sigma) * (2k + 1): the fraction a DILATED threshold masks on pure
        # Gaussian noise with no events in it at all.
        mask_free = float(
            0.5 * math.erfc(float(config.mask_sigma) / math.sqrt(2.0)) * (2 * k + 1)
        )
        rough_ratio = float(rough_sigma_pA / noise) if noise else float("nan")

        if mask_frac < MASK_FRACTION_FLOOR:
            checks.append(BaselineCheck(
                "mask_fraction_floor", "warn", "mask fraction (floor)", mask_frac,
                f">= {MASK_FRACTION_FLOOR:.3f}",
                f"only {mask_frac:.2%} of samples were masked. The +-{config.dilate_ms:g} ms "
                f"dilation multiplies the {config.mask_sigma:g}-sigma false-positive rate by "
                f"{2 * k + 1}x, so an EVENT-FREE trace masks {mask_free:.2%} by construction. "
                "A fraction at or below that has TWO readings and they are opposite, so "
                "look at the trace before acting: either there really are no events (the "
                "baseline is then a plain short-window median and is fine), or the "
                "estimator has INVERTED -- past ~50% occupancy the mask lands on the open "
                "pore instead of the events, and the baseline sits at the blockade level "
                "(measured: duty 0.60 on a -100 pA pore gives -295 pA, and the reported "
                "depths still look perfect). The other two causes are a rough window that "
                "is tracking the events instead of exposing them, and a trace drifting "
                "faster than rough_s can follow. Compare the baseline with the trace's "
                "modal current; the checks cannot tell these apart for you.",
            ))
        elif mask_frac > MASK_FRACTION_CEILING:
            checks.append(BaselineCheck(
                "mask_fraction_ceiling", "warn", "mask fraction (ceiling)", mask_frac,
                f"<= {MASK_FRACTION_CEILING:.2f}",
                f"{mask_frac:.2%} of samples were masked and interpolated across. Above "
                f"{MASK_FRACTION_CEILING:.0%} whole regions are being BRIDGED rather than "
                "masked: on a synthetic gating pore the mask grows monotonically with "
                "rough_s (11% -> 20% -> 30% -> 40%) while the fraction of samples whose "
                "baseline is more than 100 pA wrong goes 2% -> 14% -> 26% -> 38%. Shorten "
                "rough_s.",
            ))
        else:
            checks.append(BaselineCheck(
                "mask_fraction", "ok", "mask fraction", mask_frac,
                f"{MASK_FRACTION_FLOOR:.3f} .. {MASK_FRACTION_CEILING:.2f}",
                f"{mask_frac:.2%} of samples masked "
                f"({mask_free:.2%} would be masked on an event-free trace).",
            ))

        # -- the SECOND occupancy measurement, and the comparison ------------
        # THE POINT OF THIS CHECK. Everything above compares the mask against
        # CONSTANTS. A run can sit inside that band and still be missing three
        # quarters of the events, because a rough window shorter than the events
        # tracks them instead of exposing them and the mask then has nothing to
        # find. The only number that can see it is the SAME mask measured over a
        # window wide enough to sail over the states -- and the app already
        # computed one of those in `baseline_probe` and compared it to nothing.
        #
        # Skipped when the floor already fired: below the floor the mask found
        # nothing at all, a ratio against ~0 is arithmetic rather than evidence,
        # and the floor's message says considerably more about what to do.
        _report_baseline_progress(0.4, "baseline health (wide-window reference)")
        ref_states = (None if mask_frac < MASK_FRACTION_FLOOR
                      else reference_states(x if pol == -1 else -x, fs, config=config))
        if ref_states is not None:
            ref_mask, ref_rough = ref_states
            ref_frac = float(np.count_nonzero(ref_mask)) / max(n, 1)
            ref_ratio = float(ref_frac / mask_frac) if mask_frac else float("inf")
            disagrees = ref_ratio > MASK_REFERENCE_MAX_RATIO
            checks.append(BaselineCheck(
                "mask_reference", "warn" if disagrees else "ok",
                "mask vs a wide-window mask", ref_ratio,
                f"<= {MASK_REFERENCE_MAX_RATIO:g}x",
                f"the same {config.mask_sigma:g}-sigma mask finds {ref_frac:.2%} of the "
                f"trace with a {ref_rough:g} s rough window against {mask_frac:.2%} at "
                f"the configured {config.rough_s:g} s -- {ref_ratio:.2f}x, i.e. the two "
                "windows agree about how occupied this recording is."
                if not disagrees else
                f"a {ref_rough:g} s rough window masks {ref_frac:.2%} of this trace where "
                f"the configured {config.rough_s:g} s masks {mask_frac:.2%} -- "
                f"{ref_ratio:.2f}x more. The mask fraction is inside its band, so nothing "
                "else here objects, but the two numbers are measurements of the SAME "
                "occupancy and they disagree: a rough window shorter than the events is "
                "FOLLOWING them rather than exposing them, so the residual inside a "
                "blockade goes to zero and there is nothing left to mask. The baseline "
                "then sits inside the events and every depth is measured from it. "
                "Measured on a 35%-occupied 18 ms-dwell recording: at rough_s 0.03 s "
                "this ratio is 3.98 and the median event depth reads 7.8 pA against a "
                "true 110 (0.07x), while every other check passes; at 0.5 s the ratio "
                "is 1.04 and the depth is 109.9 pA. The baseline itself is only ~1.2 pA "
                "from the open pore either way -- two thirds of the samples still ARE "
                "open pore -- so no baseline-error metric can see this and only the "
                f"per-event depth can. WIDEN rough_s toward {ref_rough:g} s.",
            ))

        # -- THE STATE TIMESCALE, and the tracking window against it -------
        # Everything above is about rough_s -- which excursions become events.
        # These two are about base_s -- how fast the baseline may follow -- and
        # NOTHING above could see it: on the -700 mV apo recording all six of the
        # checks above passed in green on a base_s of 30 ms against a 21 ms state,
        # and the depth read 3% shallow.
        #
        # THE WIDE PASS THEY READ, and why this block is no longer nested inside
        # `mask_reference`. Both measure the pore against a window wide enough to
        # sail over its states. Usually that is `reference_states`; when the
        # CONFIGURED rough window is already at or above the reference window that
        # returns None -- and the configured pass is then the wider of the two and
        # is the right mask to read. Nested, both checks VANISHED there and
        # `health.ok` still read True over five rendered rows. A check that is
        # absent is not a check that passed, and the block below never omits one.
        state_mask = state_w = state_frac = None
        if ref_states is not None:
            state_mask, state_w, state_frac = ref_mask, ref_rough, ref_frac
        elif mask is not None and mask_frac >= MASK_FRACTION_FLOOR:
            w_ref = reference_rough_s(n, fs)
            if w_ref >= REFERENCE_MIN_ROUGH_S and float(config.rough_s) >= w_ref:
                state_mask, state_w, state_frac = mask, float(config.rough_s), mask_frac

        if state_mask is not None:
            check_cancelled()
            _report_baseline_progress(0.7, "baseline health (state checks)")
            # `state_runs` keeps every run longer than one sample once dilated, so
            # ZERO runs means the pass found nothing at all -- not that the states
            # were short (a 200 us event dilated by +-1 ms survives it; measured,
            # 4729 of 4750 raw runs kept). Counting the configured pass as well
            # separates a recording with no states at all -- nothing for a tracking
            # median to ride into -- from one whose WIDE pass cannot see the states
            # the configured pass does. The second is the case that certified a
            # per-event depth of 5.6 pA against a true 100.0.
            n_wide = int(state_runs(state_mask, fs, dilate_ms=config.dilate_ms)[0].size)
            n_cfg = (n_wide if state_mask is mask else
                     int(state_runs(mask, fs, dilate_ms=config.dilate_ms)[0].size))
            dwell = state_dwell_s(state_mask, fs, dilate_ms=config.dilate_ms)
            g_score, _, _ = gating_spread(x, fs)
            gating = bool(np.isfinite(g_score) and g_score > GATING_THRESHOLD
                          and state_frac <= MASK_FRACTION_CEILING)
            gate_note = (
                f"NOT APPLICABLE: the open level itself steps ({g_score:.1f}x the "
                f"within-block noise against a threshold of {GATING_THRESHOLD:g}, and "
                f"only {state_frac:.1%} of the trace is below it, so events cannot "
                "explain the spread). On a gating pore the long states ARE baseline "
                "levels and base_s is SUPPOSED to be shorter than they are -- that is "
                "what ferritin_data3's 0.03/0.01 is for. Judged as an event pore this "
                "estimate would read as badly intruded; it is not."
            )
            # The wide pass saw nothing the configured one did. Said once, used by
            # both checks, because both are unmeasurable for the same reason and a
            # reader must not take two silences for two passes.
            blind_note = (
                f"COULD NOT BE MEASURED -- which is not a pass. The {state_w:g} s "
                f"reference pass finds {state_frac:.2%} of this trace below its own "
                f"threshold while the configured {config.rough_s:g} s pass masks "
                f"{mask_frac:.2%} in {n_cfg} runs, so there is no state timescale to "
                "size base_s against and nothing here can tell you whether the "
                "tracking median rides into the states. The cause is a trace whose "
                "own LEVEL moves as far as its events do -- a stepping open pore, a "
                "conductance step, or drift -- which inflates the wide pass's noise "
                "estimate until its threshold is out of reach"
                + (f" (this trace scores {g_score:.1f}x on the gating statistic, "
                   f"threshold {GATING_THRESHOLD:g})" if gating else "")
                + ". Measured on a synthetic pore that steps +-150 pA every 80 ms "
                "and ALSO has 20 ms translocations: the probe proposes 0.03/0.01, "
                "six checks rendered green, and the per-event depth read 5.6 pA "
                "against a true 100.0. Do not read a depth or a dI/I0 off this "
                "table. Widen rough_s until both windows mask something, or measure "
                "a stationary stretch of the recording."
            )
            # Neither pass found a state. There is nothing for base_s to be too
            # short for, and saying so is a different statement from the one above.
            quiet_note = (
                f"no state to size base_s against: neither the configured "
                f"{config.rough_s:g} s pass nor the {state_w:g} s reference pass "
                "finds a masked run longer than a single sample once dilated, so "
                "this recording has no states for a tracking median to ride into "
                f"({mask_frac:.2%} of samples masked, against the {mask_free:.2%} an "
                f"event-free trace masks by itself at {config.mask_sigma:g} sigma)."
            )
            base_s = float(config.tracking_window_s)
            if np.isfinite(dwell) and dwell > 0:
                base_ratio = base_s / dwell
                want = BASE_STATE_RATIO_MIN * dwell
                short = (not gating) and base_ratio < BASE_STATE_RATIO_MIN
                checks.append(BaselineCheck(
                    "base_vs_state", "warn" if short else "ok",
                    "base window vs state dwell", base_ratio,
                    f">= {BASE_STATE_RATIO_MIN:g} x the {dwell * 1e3:.1f} ms state dwell",
                    gate_note if gating else
                    f"base_s {base_s:g} s is {base_ratio:.1f}x the {dwell * 1e3:.1f} ms median "
                    f"state this recording spends below its open pore ({state_frac:.1%} of "
                    f"the trace, measured over a {state_w:g} s window wide enough to see "
                    "over them)."
                    if not short else
                    f"base_s is {base_s:g} s and the median state this pore spends below "
                    f"its open pore is {dwell * 1e3:.1f} ms -- a tracking window only "
                    f"{base_ratio:.1f}x the states it exists to ignore. A rolling median that "
                    "narrow RIDES INTO them: the baseline sits below the true open pore at "
                    "exactly the samples every depth is measured from, so depths read "
                    "shallow while the baseline's own level, drift, noise and overshoot "
                    "stay perfect and nothing else here objects. Measured on the -700 mV "
                    "apo recording, whose probe proposed exactly this window: at 1.4x the "
                    "baseline sits 4.2 pA inside the states -- 4% of their depth -- and the "
                    "median depth reads 2.7% low; at 4.6x it reads 0.3% low; at 14x, 0.0%. "
                    "WIDEN base_s "
                    f"to at least {want:.3g} s BY HAND (and rough_s with it, at about "
                    f"{ROUGH_BASE_RATIO:g}x base_s) -- do not read a depth or a dI/I0 "
                    "off this table until you have. "
                    + ("OR PRESS PROBE AGAIN, WHICH NOW DOES IT FOR YOU: the "
                       "proposer divides by this same median and floors base_s at "
                       f"{BASE_STATE_RATIO_MIN:g}x it, so it offers the shortest "
                       f"window on its grid at or above {want:.3g} s instead of the "
                       "one condemned here, so a freshly proposed window should not "
                       "normally reach this message: these were most likely typed, "
                       "restored, or carried over from another recording. (Before "
                       "that floor existed the proposer sized "
                       "base_s off the p99.9 of the dwells, 1.5x this median on the "
                       "-700 mV recording, and it was deterministic: two consecutive "
                       "probes there both re-offered the condemned 0.1 / 0.03 s.)"
                       if ref_states is not None else
                       "The proposer floors base_s at "
                       f"{BASE_STATE_RATIO_MIN:g}x this median measured over the "
                       "REFERENCE window, and this ratio was measured over the "
                       f"configured {config.rough_s:g} s pass instead (which is "
                       "already at or above it), so pressing Probe again is not "
                       "guaranteed to clear this one -- widen both windows by "
                       "hand."),
                ))

            else:
                checks.append(BaselineCheck(
                    "base_vs_state", "warn" if n_cfg else "ok",
                    "base window vs state dwell", None,
                    f">= {BASE_STATE_RATIO_MIN:g} x the state dwell",
                    blind_note if n_cfg else quiet_note,
                ))

            check_cancelled()
            # ``worst_min_dwell_s = base_s`` for the same reason `config_intrusion`
            # passes it: the check consumes only the median half today, but the two
            # callers are one measurement and must not quietly diverge on the other.
            intr = tracked_intrusion(
                x if pol == -1 else -x, b if pol == -1 else -b, state_mask, fs,
                min_depth_pA=INTRUSION_MIN_DEPTH_SIGMA * noise,
                dilate_ms=config.dilate_ms,
                worst_min_dwell_s=float(config.base_s),
            )
            if intr is not None:
                frac = intr["fraction"]
                deep = (not gating) and frac > BASELINE_INTRUSION_MAX
                checks.append(BaselineCheck(
                    "baseline_intrusion", "warn" if deep else "ok",
                    "baseline vs the open pore beside each state", frac,
                    f"<= {BASELINE_INTRUSION_MAX:.0%} of the state depth",
                    gate_note if gating else
                    f"across {intr['n_states']} states the tracked baseline sits "
                    f"{intr['intrusion_pA']:.2f} pA inside them -- {frac:.1%} of their "
                    f"{intr['depth_pA']:.1f} pA excursion, taken against the open pore "
                    f"{INTRUSION_PAD_S * 1e3:g}-{(INTRUSION_PAD_S + INTRUSION_FLANK_S) * 1e3:g} "
                    "ms either side of each one."
                    if not deep else
                    f"the tracked baseline sits {intr['intrusion_pA']:.2f} pA INSIDE the "
                    f"states, measured against the open pore "
                    f"{INTRUSION_PAD_S * 1e3:g}-{(INTRUSION_PAD_S + INTRUSION_FLANK_S) * 1e3:g} "
                    f"ms either side of each of {intr['n_states']} of them. That is "
                    f"{frac:.1%} of their own {intr['depth_pA']:.1f} pA excursion, and it "
                    "comes off every depth this baseline measures: depth_pA and frac_depth "
                    f"read about {frac:.0%} low, and a voltage series measured this way "
                    "puts one recording out of line with the rest. This is the same "
                    "comparison that found the -700 mV file (a -51.7 pA stamped baseline "
                    "against a -48.2 pA open pore reconstructed from the samples before "
                    "each event). WIDEN base_s -- see the base-window check for how far -- "
                    "and re-measure.",
                ))

            elif n_wide:
                checks.append(BaselineCheck(
                    "baseline_intrusion", "ok",
                    "baseline vs the open pore beside each state", None,
                    f"<= {BASELINE_INTRUSION_MAX:.0%} of the state depth",
                    f"not measurable, and benign: every one of the {n_wide} states "
                    f"the {state_w:g} s pass found is shallower than "
                    f"{INTRUSION_MIN_DEPTH_SIGMA:g} sigma ({INTRUSION_MIN_DEPTH_SIGMA * noise:.1f} "
                    "pA), and a ratio of two depths measured that shallow is a "
                    "reading of the noise. There is no excursion here deep enough "
                    "for a baseline to sit inside.",
                ))
            else:
                checks.append(BaselineCheck(
                    "baseline_intrusion", "warn" if n_cfg else "ok",
                    "baseline vs the open pore beside each state", None,
                    f"<= {BASELINE_INTRUSION_MAX:.0%} of the state depth",
                    blind_note if n_cfg else quiet_note,
                ))

        # A RATIO THAT IS NOT A NUMBER CANNOT CERTIFY. ``rough_ratio`` is NaN in
        # exactly one reachable way: the final detrended noise is 0.00 pA (the
        # assignment above writes NaN rather than dividing by zero; ``noise`` is
        # `_qc_block`'s MAD, which reads 0 when more than half the residual sits at
        # exactly one current -- a railed amplifier, a saturated stretch cut into a
        # segment, quantised synthetic data). This branch used to map NaN to 'ok',
        # so the Baseline panel rendered "rough-pass sigma 0.00 pA against a final
        # noise of 0.00 pA (ratio nan)" as a green PASS row -- a certificate from a
        # check that had measured nothing. A check that cannot compute its own
        # statistic must SAY so; it is a 'warn' rather than an ok-with-prose
        # (`edge_tracking`'s choice when THIS SAME noise fails it) because there the
        # length rows already condemn the recording, whereas nothing above this row
        # necessarily condemns a zero-noise trace -- the mask-fraction band can pass
        # on quantised-flat data -- and a trace with no measurable noise is one
        # `detect_events` itself refuses to threshold ("noise RMS is 0.0").
        if not np.isfinite(rough_ratio):
            checks.append(BaselineCheck(
                "rough_sigma_ratio", "warn", "rough-pass sigma vs final noise",
                rough_ratio,
                f"{ROUGH_SIGMA_RATIO_MIN:.2f} .. {ROUGH_SIGMA_RATIO_MAX:.2f}",
                f"COULD NOT BE MEASURED: the final detrended noise is {noise:.2f} pA "
                f"(rough-pass sigma {rough_sigma_pA:.2f} pA), so the two-pass ratio "
                "this check exists to judge is not a number. More than half of this "
                "trace's residual sits at exactly one current -- a railed/saturated "
                "amplifier or quantised data, not a measured open pore -- and no "
                "window comparison can be certified on it. Detection refuses a trace "
                "with no measurable noise for the same reason; analyse a stretch "
                "where the current actually fluctuates.",
            ))
        else:
            level = ("warn" if rough_ratio > ROUGH_SIGMA_RATIO_MAX
                     or rough_ratio < ROUGH_SIGMA_RATIO_MIN else "ok")
            # ``upper_mad_sigma_pA`` can be NaN on its own (an empty upper half in
            # PASS 1's residual while the FINAL residual still has noise): never
            # print "nan pA" inside a PASS row -- say what it means instead.
            upper_says = (f"one-sided upper MAD {upper_mad_sigma_pA:.2f} pA"
                          if np.isfinite(upper_mad_sigma_pA) else
                          "one-sided upper MAD not measurable (no pass-1 residual "
                          "above its median)")
            checks.append(BaselineCheck(
                "rough_sigma_ratio", level, "rough-pass sigma vs final noise",
                rough_ratio,
                f"{ROUGH_SIGMA_RATIO_MIN:.2f} .. {ROUGH_SIGMA_RATIO_MAX:.2f}",
                f"rough-pass sigma {rough_sigma_pA:.2f} pA against a final noise of "
                f"{noise:.2f} pA (ratio {rough_ratio:.2f}); {upper_says}."
                if level == "ok" else
                (f"the rough pass measures {rough_sigma_pA:.2f} pA of scatter against a final "
                 f"detrended noise of {noise:.2f} pA (ratio {rough_ratio:.2f}). Far above 1 means "
                 "pass 1 is measuring DRIFT, not events -- a 300 pA sinusoid at a 0.2 s period "
                 "takes rough-sigma from 8.9 to 241 pA, puts the 3.5-sigma threshold out of "
                 "reach and collapses the mask to exactly zero while the median baseline error "
                 "hits 30.6 pA. Far below 1 means the SHORT pass cannot follow the trace: on a "
                 "gating pore the detrended residual stays 100-400 pA wide. Either way the two "
                 "windows do not match this recording."),
            ))

    # -- the two ENDS of the recording, where pass 1 has only reflected data --
    # THE ONE ROW ABOUT THE BOUNDARY, and it is emitted for every method. Everything
    # above judges the recording as a whole -- a mask fraction, a sigma ratio, a
    # median over states, an occupancy -- and every one of those statistics is
    # dominated by the interior, which is where the estimator has data on both sides
    # of its kernel. At the first and last sample it does not: the filters pad by
    # REFLECTION, so the level they report at a boundary is biased by the trace's own
    # slope, and on a drifting recording that bias is the size of the drift. See
    # `EDGE_OFFSET_SIGMA` for the fixture where this leaves the baseline 69.5 pA
    # (14 sigma) wrong across the head with all nine other checks green and 17
    # accepted events on an event-free trace.
    #
    # Measured on the ANSWER, not on the parameters: the median residual over the
    # first (last) tracking window against the median residual over the interior,
    # which is the same "compare this recording against ITSELF" construction
    # `mask_reference` uses. `EDGE_DRIFT_RATIO` carries the second half -- why a real
    # blockade sitting at t=0 does not light this up.
    check_cancelled()
    _report_baseline_progress(0.85, "baseline health (edge and level rows)")
    w_edge = max(1, int(round(float(config.tracking_window_s) * fs)))
    if n < 4 * w_edge or not (np.isfinite(noise) and noise > 0.0):
        # Emitted anyway, because a check missing from the block is not a check that
        # passed -- the same rule `base_vs_state` and `baseline_intrusion` follow when
        # their wide pass finds nothing. It is 'ok' rather than 'warn' because the
        # length rows above (and `tracking_window_fits` on a fallback) already say the
        # condemning thing about a recording this short, and one fact gets one sentence.
        checks.append(BaselineCheck(
            "edge_tracking", "ok", "baseline vs the trace at the recording's ends",
            None,
            f"needs >= 4 x the {w_edge / fs:g} s tracking window",
            f"COULD NOT BE MEASURED: {n / fs:g} s of signal is under 4x the "
            f"{w_edge / fs:g} s tracking window, so this recording has no interior to "
            "compare its two ends against. The length rows above are what speak to a "
            "recording this short."
            if n < 4 * w_edge else
            "COULD NOT BE MEASURED: the detrended noise is not a positive, finite "
            "number, so there is no scale to judge a boundary offset against.",
        ))
    else:
        # ``-detr_canon`` is (baseline - trace) in the canonical frame: POSITIVE
        # where the baseline sits ABOVE the trace, i.e. where the trace reads as
        # blocked. Reusing the block's own detrended, despiked residual keeps this
        # row on exactly the samples every other row above was measured on.
        above = -np.asarray(detr_canon, dtype=np.float64)
        interior = float(np.median(above[w_edge:n - w_edge]))
        head_off = float(np.median(above[:w_edge])) - interior
        tail_off = float(np.median(above[n - w_edge:])) - interior
        edge_which, edge_off = (("first", head_off) if abs(head_off) >= abs(tail_off)
                                else ("last", tail_off))
        # How far this baseline ITSELF moves over one tracking window inside the
        # recording -- the drift it demonstrably follows, read off its own answer
        # rather than off a constant. Strided by w/10: the quantity is a p95 over
        # displacements, and 10 samples per window is already thousands of them.
        b_canon = b if pol == -1 else -b
        step = max(1, w_edge // 10)
        i0 = np.arange(w_edge, n - 2 * w_edge, step)
        edge_drift = (float(np.percentile(np.abs(b_canon[i0 + w_edge] - b_canon[i0]), 95))
                      if i0.size else 0.0)
        visible = abs(edge_off) >= EDGE_OFFSET_SIGMA * noise
        drift_sized = abs(edge_off) <= EDGE_DRIFT_RATIO * edge_drift
        untracked = visible and drift_sized
        # A baseline that does not move at all over one window (a genuinely flat
        # recording) makes the ratio infinite rather than undefined -- an offset there
        # is a step by construction -- and the sentence has to say that in words
        # instead of dividing by zero.
        ratio_says = (f"{abs(edge_off) / edge_drift:.2f}x the {edge_drift:.2f} pA"
                      if edge_drift > 0.0 else
                      "beyond anything the drift can account for: 0.00 pA is all")
        checks.append(BaselineCheck(
            "edge_tracking", "warn" if untracked else "ok",
            "baseline vs the trace at the recording's ends", float(edge_off),
            f"within {EDGE_OFFSET_SIGMA:g} sigma of the interior, or a step larger "
            f"than {EDGE_DRIFT_RATIO:g}x the drift over one {w_edge / fs:g} s window",
            f"across the {edge_which} {w_edge / fs:g} s the tracked baseline sits "
            f"{edge_off:+.2f} pA from this recording's own level there "
            f"({abs(edge_off) / noise:.2f} sigma), and the interior is the reference: "
            "the estimate tracks the trace at both ends as well as it does in the "
            "middle."
            if not visible else
            f"the {edge_which} {w_edge / fs:g} s of this recording sits "
            f"{edge_off:+.2f} pA ({abs(edge_off) / noise:.1f} sigma) off the tracked "
            f"baseline, but that is {ratio_says} this baseline moves per "
            f"{w_edge / fs:g} s elsewhere "
            "in the recording, so it is a STEP the drift cannot explain -- a real "
            "blockade, a settling transient or a voltage step at the boundary, which "
            "the estimator is right to bridge rather than follow. Read as signal, not "
            "as an estimator failure. (The owner's -600 mV file opens like this: 23 pA "
            "below its open pore for 70 ms, ratio 8.0x.)"
            if not untracked else
            f"the tracked baseline sits {edge_off:+.2f} pA from this recording's own "
            f"level across its {edge_which} {w_edge / fs:g} s "
            f"({abs(edge_off) / noise:.1f} sigma, against the interior), and that is "
            f"only {ratio_says} this same "
            f"baseline moves per {w_edge / fs:g} s elsewhere -- i.e. the boundary is "
            "off by no more than the drift the estimator follows everywhere else, "
            "which is what a baseline that never TRACKED the trace at its own edge "
            "looks like. Pass 1 pads by REFLECTION, so at a boundary its median is "
            "biased by the trace's own slope; when that bias reaches the mask "
            "threshold the whole end is masked as one state, pass 2 bridges it, and "
            "the baseline is pinned at a level the trace only reaches later. The "
            "residual under it is then reported as EVENTS: measured on an event-free "
            "trace drifting 250 pA at 0.5 Hz, the baseline reads 69.5 pA (14 sigma) "
            "wrong over the first 10 ms and detection returned 18 events, 17 accepted, "
            "all inside the first 83 ms -- while the SAME samples with 2 s of the same "
            "drift in front of them are 0.67 pA out with no events there. Nothing here "
            "can separate that from a genuine state at the boundary of the same size, "
            "which is why this is reported rather than corrected. DO NOT read a depth "
            f"or a dI/I0 from the {edge_which} {w_edge / fs:g} s of this table: analyse "
            "a stretch that starts before the region you care about, or shorten the "
            "windows until pass 1 follows the drift at the edge (on the fixture above, "
            "0.05 / 0.01 s brings the head error from +69.5 pA to -2.4 pA).",
        ))

    # -- the tracked LEVEL against the recording's own amplitude histogram ----
    # THE ONE ROW WITH NO FRAME. Everything above reads the trace canonicalised by a
    # polarity derived from the same trace, so a recording that is mostly BLOCKED flips
    # all of them together and they then agree, in green, about a baseline sitting on
    # the blockade (see `LEVEL_OCCUPANCY_MIN` for the eight-green measurement). This is
    # |detr| and a histogram of detr: two-sided, polarity-free, nothing for a flip to
    # cancel.
    #
    # TWO numbers, because one of them was not enough. The tracked level's occupancy is
    # the margin the identification rests on -- but on an inverted estimate it RISES
    # with the blockade fraction (0.791 -> 0.897 -> 0.942 -> 0.970 across the ladder in
    # `ALT_LEVEL_OCCUPANCY_MIN`), so a line drawn on it alone goes quiet exactly as the
    # estimate gets worse. `alternative_level` is the other half: the second mode of
    # this recording's own amplitude histogram, which is present in every one of those
    # rows.
    #
    # The row is emitted on EVERY estimate, including the ones another check has already
    # condemned -- a margin the reader cannot see is a margin they will assume. It fails
    # on the three things a comparison of two occupancies can honestly fail on: a
    # baseline that is at NO level of this recording (`LEVEL_OCCUPANCY_MAJORITY`), a
    # majority rule that ran without a decisive majority (`LEVEL_DECISIVE_RATIO`), and a
    # majority under the corpus floor (`LEVEL_OCCUPANCY_MIN`) for a level the recording
    # never HOLDS for one tracking window (`LEVEL_HOLD_MIN_WINDOWS`) -- the last being
    # the one that reaches a baseline sitting on a pore blocked ~80% of the time, and it
    # is a conjunction because neither of its halves separates that estimate from a
    # valid one in either direction (the numbers are in `LEVEL_HOLD_MIN_WINDOWS`).
    #
    # None of the three fails on an INVERSION as such, because no statistic in this
    # module can tell one from a valid estimate -- `alternative_level` lists the four
    # that were tried and how each died on the owner's own files -- and everything that
    # is known but not verified is said as an advisory on the result
    # (`level_occupancy_warning`), which is the surface that can say "I do not know
    # which of these two levels is the open pore" without pretending a check measured it.
    check_cancelled()
    k_off = max(1, int(float(config.dilate_ms) * 1e-3 * fs))
    off = binary_dilation(
        np.abs(detr) > INTRUSION_MIN_DEPTH_SIGMA * noise, iterations=k_off)
    off_frac = float(np.count_nonzero(off)) / max(n, 1)
    level_occ = 1.0 - off_frac
    check_cancelled()
    alt_offset, alt_occ = alternative_level(detr, noise)
    alt_is_level = alt_offset is not None and alt_occ >= ALT_LEVEL_OCCUPANCY_MIN
    no_majority = level_occ < LEVEL_OCCUPANCY_MAJORITY
    indecisive = (not no_majority) and level_occ > 0.0 and (
        alt_occ >= LEVEL_DECISIVE_RATIO * level_occ)
    # How long the pore RESTS at the level being tracked, in tracking windows: the p99
    # of the unbroken ON-level runs, which is `_mask_run_p99_s`'s reading applied to the
    # complement of the same dilated off-level flag. Frame-free for the same reason the
    # occupancy above is -- ``off`` is |detr|, two-sided -- so a polarity measured from
    # an inverted trace cannot cancel it either. ``hold_windows`` is +inf when there is
    # no meaningful window to compare against, which is "nothing here to condemn" rather
    # than "condemn it"; a trace that never leaves the level is one long run and reads
    # large by measurement, and one that never reaches it has no runs and reads 0 --
    # where `LEVEL_OCCUPANCY_MAJORITY` has already fired with the better reason.
    on_starts, on_ends = _mask_runs(~off)
    hold_p99_s = (float(np.percentile(on_ends - on_starts, 99)) / fs
                  if on_starts.size else 0.0)
    win_s = float(config.tracking_window_s)
    hold_windows = (hold_p99_s / win_s if np.isfinite(win_s) and win_s > 0.0
                    else float("inf"))
    unheld = (
        (not no_majority) and (not indecisive) and alt_is_level
        and level_occ < LEVEL_OCCUPANCY_MIN
        and hold_windows < LEVEL_HOLD_MIN_WINDOWS
    )
    # The sentence the two failing branches and the advisory all need: which two levels
    # this recording has, in the reader's own units. ``level_pA`` from the QC block is
    # the DETRENDED level (the notebook's number, ~0 by construction); the current the
    # reader sees on the axis is the median of the tracked baseline itself.
    tracked_pA = float(np.median(b))
    alt_says = (
        "" if alt_offset is None else
        f"the other level this recording visits is {alt_offset:+.1f} pA from the "
        f"tracked one ({(tracked_pA + alt_offset):.1f} pA against {tracked_pA:.1f} pA) "
        f"and holds {alt_occ:.1%} of the recording"
    )
    # The branches in the order they are ASKED, not in the order they read: a baseline
    # at no level of the recording is a different report from a thin majority, and a
    # thin majority is a different report from one that is merely never held. Written as
    # a ladder rather than a chained conditional expression because there are now four
    # of them and the reader has to be able to see which one they got.
    level_msg = (
        f"the tracked baseline is more than {INTRUSION_MIN_DEPTH_SIGMA:g} sigma "
        f"({INTRUSION_MIN_DEPTH_SIGMA * noise:.1f} pA) away from the trace for "
        f"{off_frac:.1%} of this recording -- the level every depth here is measured "
        f"from holds only {level_occ:.1%} of the recording it was measured on. The "
        "estimator's rule is that the open pore is the level the pore is at MOST of "
        "the time; below half there is no majority for it to have found. This is what "
        "a baseline belonging to a DIFFERENT recording reads like -- one restored, or "
        "carried over, onto a trace it was not measured on (a -300 pA baseline over a "
        "-100 pA trace reads 0.0000 here). Every depth and dI/I0 in the table is then "
        "measured from a current this trace never visits. Re-measure the baseline on "
        "THIS recording before reading anything off it."
        if no_majority else
        f"the level every depth here is measured from holds {level_occ:.1%} of the "
        f"recording and {alt_says}. The estimator's rule is that the open pore is the "
        "level the pore is at MOST of the time, and at this margin the rule ran "
        f"without a majority worth the name -- the second level holds "
        f"{(alt_occ / level_occ if level_occ else float('nan')):.2f}x the first, "
        f"against the {LEVEL_DECISIVE_RATIO:.2f}x this block will certify a choice "
        "between two levels at (the owner's own recordings run 0.06-0.11x). This is "
        "the regime the overshoot check's own message names -- occupancy past 50%, "
        "where the estimator inverts rather than degrades -- and it is measured here "
        "instead of there because overshoot is read through a frame this trace chose. "
        "Which of the two levels is the open pore is not in these samples: analyse a "
        "stretch where the pore is open, or set the baseline from the open-pore "
        "current you know for this pore."
        if indecisive else
        # THE THIRD LINE. Both halves are needed and neither is sufficient: see
        # `LEVEL_HOLD_MIN_WINDOWS` for the three-row table where the inverted estimate
        # is the MORE occupied of the three and the valid ones are the ones that hold.
        f"the level every depth here is measured from holds {level_occ:.1%} of the "
        f"recording -- under the {LEVEL_OCCUPANCY_MIN:.0%} every recording in evidence "
        f"clears -- and {alt_says}. Worse, the pore never RESTS at the tracked level: "
        f"its longest stretches there (p99 {hold_p99_s * 1e3:.0f} ms) are shorter than "
        f"the {win_s * 1e3:g} ms window that measured it, so no window of this "
        "recording was ever free of the other level and the level reported is the "
        "outcome of a vote across many visits to both, not a measurement of one of "
        "them. THIS IS WHAT A BASELINE THAT HAS SETTLED ON THE BLOCKADE READS LIKE: a "
        f"pore blocked {level_occ:.0%} of the time, its brief returns to the open pore "
        f"masked as the events, and every depth referenced to {tracked_pA:.1f} pA "
        f"instead of {(tracked_pA + alt_offset if alt_offset is not None else float('nan')):.1f} "
        "pA. It is NOT a proof of that reading -- the same numbers are what a pore "
        f"genuinely off-level {off_frac:.0%} of the time gives -- which is exactly why "
        "this is not certified either way. Narrow the tracking window until the pore "
        "holds the level across it, analyse a stretch where the pore is open, or set "
        "the baseline from the open-pore current you know for this pore."
        if unheld else
        f"{level_occ:.1%} of this recording sits within {INTRUSION_MIN_DEPTH_SIGMA:g} "
        f"sigma ({INTRUSION_MIN_DEPTH_SIGMA * noise:.1f} pA) of the tracked baseline "
        f"and {alt_says} -- a decisive majority for the level that was tracked, which "
        "is what this row can check, and NOT a verification that THAT level is the "
        "open pore. Nothing here can be: the same trace read the other way up is a "
        f"pore BLOCKED {level_occ:.1%} of the time whose returns to the open pore are "
        "what got masked as events, and the two readings are the same samples. The "
        "result carries the advisory that spells both of them out; read it before "
        "reading a depth."
        if alt_is_level else
        f"{level_occ:.1%} of this recording sits within {INTRUSION_MIN_DEPTH_SIGMA:g} "
        f"sigma ({INTRUSION_MIN_DEPTH_SIGMA * noise:.1f} pA) of the tracked baseline "
        "and nothing else in the trace holds enough of it to be a second level "
        f"(the deepest mode outside the baseline holds {alt_occ:.1%}, under the "
        f"{ALT_LEVEL_OCCUPANCY_MIN:.1%} this block reads as a level). So the "
        "open-pore identification here is not a choice between two candidates. It is "
        "still not a VERIFICATION of one: the estimator calls the level the pore is at "
        "most of the time the open pore, and no check in this block reads the physics "
        "that would confirm it. Measured in no frame -- |current - baseline| and a "
        "histogram of the residual -- so unlike every other row above, a polarity "
        "measured from an inverted trace cannot cancel this one."
    )
    checks.append(BaselineCheck(
        "level_occupancy", "warn" if (no_majority or indecisive or unheld) else "ok",
        "tracked level vs the recording's own histogram", level_occ,
        f">= {LEVEL_OCCUPANCY_MAJORITY:.0%} of the recording, and >= "
        f"{1 / LEVEL_DECISIVE_RATIO:g}x the next level's share; under "
        f"{LEVEL_OCCUPANCY_MIN:.0%}, also held for >= {LEVEL_HOLD_MIN_WINDOWS:g} "
        "tracking window",
        level_msg,
    ))

    checks.append(BaselineCheck(
        "overshoot", "warn" if (np.isfinite(over) and over > OVERSHOOT_FLAG) else "ok",
        "overshoot", over, f"<= {OVERSHOOT_FLAG:g}",
        f"overshoot {over:.3f} ({n_up} samples above +4 sigma, {n_down} below; "
        "healthy corpus values 0.09-0.11). Measured in the CANONICAL frame, which is "
        f"chosen by the polarity ({pol:+d}) -- so on a recording whose own polarity "
        "measurement has inverted, this scalar inverts with it and passes; the "
        "occupancy margin above is the row that is not read through that frame."
        if not (np.isfinite(over) and over > OVERSHOOT_FLAG) else
        f"overshoot is {over:.3f}: {n_up} samples above +4 sigma against {n_down} below. "
        "If those counts are a meaningful fraction of the trace, the baseline is sitting "
        "INSIDE the events, so what should be "
        "a blockade reads as an upward excursion -- the signature of a window that is too "
        "short, or of occupancy past 50%, where the estimator inverts rather than degrades "
        "(measured: duty 0.35 -> 3.7 pA error, duty 0.60 -> 150 pA error with the sign "
        "flipped). It is the ONE published QC scalar that resists a too-short window; the "
        "other three are monotonically improved by shrinking it, so never tune on them. "
        "It catches the occupancy case only when the POLARITY is right: it is computed in "
        f"the canonical frame, which this run took from polarity {pol:+d}, and a polarity "
        "derived from a trace that is already mostly blocked flips the frame with the "
        "recording (the 95%-duty inversion reads 2941.7 at a pinned polarity and 0.0003 "
        "at a derived one). The occupancy row is the one that is frame-free.",
    ))

    return BaselineHealth(
        checks=tuple(checks),
        method=config.method,
        n_samples=int(n),
        fs=fs,
        polarity=pol,
        mask_fraction=mask_frac,
        mask_fraction_event_free=mask_free,
        mask_fraction_reference=ref_frac,
        mask_reference_rough_s=ref_rough,
        mask_reference_ratio=ref_ratio,
        mask_run_p99_s=run_p99,
        state_dwell_s=(None if dwell is None or not np.isfinite(dwell)
                       else float(dwell)),
        base_state_ratio=None if base_ratio is None else float(base_ratio),
        intrusion_fraction=None if intr is None else float(intr["fraction"]),
        intrusion_pA=None if intr is None else float(intr["intrusion_pA"]),
        intrusion_depth_pA=None if intr is None else float(intr["depth_pA"]),
        intrusion_n_states=None if intr is None else int(intr["n_states"]),
        gating_score=(None if g_score is None or not np.isfinite(g_score)
                      else float(g_score)),
        level_occupancy=None if level_occ is None else float(level_occ),
        alt_level_offset_pA=None if alt_offset is None else float(alt_offset),
        alt_level_occupancy=float(alt_occ),
        level_hold_windows=float(hold_windows),
        gating_states=None if gating is None else bool(gating),
        rough_sigma_pA=None if rough_sigma_pA is None else float(rough_sigma_pA),
        upper_mad_sigma_pA=None if upper_mad_sigma_pA is None else float(upper_mad_sigma_pA),
        rough_sigma_ratio=rough_ratio,
        noise_pA=noise,
        open_pore_level_pA=level_pA,
        open_pore_drift_pA=drift_pA,
        open_pore_drift_sigma=drift_sigma,
        overshoot=over,
    )


#: Reason token stamped as ``baseline_fallback`` when the trace cannot support a
#: tracking window at all and a global median (``'constant'``) stood in for it.
#: Defined HERE rather than beside the rest of the seam because `BaselineResult`
#: below carries it as a field default -- see `adapt_to_length` for what the two
#: tokens mean and the measurements behind the choice between them.
FALLBACK_TOO_SHORT = "trace_shorter_than_tracking_window"

#: Reason token stamped as ``baseline_fallback`` when the CONFIGURED ESTIMATOR ran,
#: with its window(s) clamped to the only widths this many samples can host.
FALLBACK_CLAMPED = "windows_clamped_to_recording"


@dataclass(frozen=True)
class BaselineResult:
    """A baseline plus everything needed to describe, stamp and judge it.

    THE API THE SEAM SHOULD USE. ``detect.py``, ``ml_modules.py``, ``truth.py`` and
    ``characterize.recharacterize_span`` all take a ``BaselineConfig`` and all stamp
    `provenance`, so a Schmitt row and an ML row measured on the same recording carry
    the same ``baseline_fingerprint`` and are directly comparable -- which is the
    whole point of this change.
    """

    baseline: np.ndarray
    config: BaselineConfig
    fs: float
    polarity: int
    health: BaselineHealth | None = None
    #: How the parameters in force were chosen (`PARAM_SOURCES`). Set by
    #: `characterisation_baseline`; the bare estimator has no opinion, so it is
    #: ``'default'`` until the seam says otherwise.
    param_source: str = "default"
    #: Non-None when the requested estimator could NOT run on this trace as
    #: configured and `adapt_to_length` stood something else up. The text is the
    #: reason, and it is surfaced as a warning rather than swallowed.
    fallback: str | None = None
    #: WHICH rung of `adapt_to_length` was taken -- `FALLBACK_CLAMPED` (the
    #: configured estimator at clamped windows) or `FALLBACK_TOO_SHORT` (a global
    #: median). Carried rather than assumed: ``provenance`` used to stamp
    #: ``FALLBACK_TOO_SHORT`` unconditionally, so every fallback row claimed the same
    #: reason no matter what actually happened, and no consumer could tell a clamped
    #: measurement from an untracked one.
    fallback_token: str = FALLBACK_TOO_SHORT

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint(self.fs)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Every soft advisory this baseline carries, for `AnalysisResult.warnings`.

        The last one is unconditional on ``param_source == 'default'``, and it is
        there because the health checks provably do not cover the case. Measured on
        the owner's own gating pore ``ferritin_data3``, against the ``baseline_pA``
        column his notebook produced for it:

            block median, 10 ms (the estimator this change replaces)   3.71 pA
            masked median at the APP DEFAULT, 0.5 / 0.1 s             12.36 pA
            masked median at HIS parameters, 0.03 / 0.01 s             2.69 pA

        At the correct polarity the default's health block reads ``ok`` on all five
        checks and `HEALTH.warnings` is empty, so on that recording an un-probed run
        is 3.3x WORSE than before this change and says nothing. The checks catch
        catastrophe (a 200 pA inversion); a 12 pA error on a trace whose noise floor
        is 4.6 pA is an 18% depth error that looks entirely publishable.

        That is not a defect in the default -- §0.3 established that no global
        default can be correct, which is why the app proposes and the user confirms.
        It is a defect in staying SILENT about a value nobody chose, and this is the
        one place that knows nobody chose it.
        """
        out: list[str] = []
        if self.fallback:
            out.append(self.fallback)
        if self.health is not None:
            out.extend(self.health.warnings)
        # Gated on the SOURCE and only the source. The length clamp used to overwrite
        # ``param_source`` with ``'fallback'`` on the way here, which silenced this
        # sentence on every short segment -- the weakest estimate of all (S7-DET3);
        # the clamp is now recorded in ``fallback`` / ``fallback_token`` and leaves
        # the source alone, so an un-probed default that was ALSO clamped says both.
        if self.config.is_masked and self.param_source == "default":
            out.append(UNPROBED_DEFAULT_WARNING)
        # ...and its sibling for windows the batch copied from another recording's
        # form: equally unmeasured HERE, and until sweep 7 stamped ``'proposed'``,
        # which switched this whole block off (S7-DET1).
        if self.config.is_masked and self.param_source == BATCH_SOURCE:
            out.append(batch_copied_warning(self.config))
        # The second unconditional-by-measurement advisory, and it is here for the
        # same reason as the one above: the checks provably do not cover the case.
        # `LEVEL_OCCUPANCY_MIN` carries the proof -- an inverted estimate and a valid
        # one land on the same side of every check in the block -- so what the health
        # block can honestly do is report the margin, and what this can honestly do is
        # say, once, that the open pore was identified and not verified.
        occ = None if self.health is None else self.health.level_occupancy
        alt_occ = None if self.health is None else self.health.alt_level_occupancy
        alt_off = None if self.health is None else self.health.alt_level_offset_pA
        # TWO triggers, and the second is the one that made this reachable across the
        # regime it is about. A thin majority is its own reason (`LEVEL_OCCUPANCY_MIN`).
        # A SECOND LEVEL is the other: on an inverted estimate the majority gets FATTER
        # as the inversion gets worse, so the first trigger alone caught a pore blocked
        # 81% of the time and said nothing at 90%, 95% or 97%. The second level is there
        # in all four. See `ALT_LEVEL_OCCUPANCY_MIN` for why it fires on the owner's own
        # recordings too, and why that is the honest outcome rather than a false alarm.
        if occ is not None and (
            occ < LEVEL_OCCUPANCY_MIN
            or (alt_occ is not None and alt_occ >= ALT_LEVEL_OCCUPANCY_MIN)
        ):
            out.append(level_occupancy_warning(
                float(occ),
                alt_occupancy=None if alt_occ is None else float(alt_occ),
                alt_offset_pA=None if alt_off is None else float(alt_off),
                tracked_pA=self.median_pA,
            ))
        # ONE FACT, ONE SENTENCE. The fallback above and `tracking_window_fits` in the
        # health block are the SAME sentence written twice: `_with_fallback` stamps the
        # fallback text onto a check so the block renders a row for it, and this
        # property then prepends the very same string. Every short or hand-cut segment
        # -- a legal document; `core.segment.MIN_SEGMENT_SAMPLES` is 2 -- therefore
        # handed `AnalysisResult.warnings` the identical ~300-character paragraph twice,
        # and `panels.StatsPanel.set_warnings` renders the list it is given WHOLE and
        # unfiltered (its docstring says so, deliberately: it must not know which
        # warning it is showing). Measured through the live widget: 2 bullets under
        # "About these numbers", 1 distinct. The de-duplication belongs HERE, at the one
        # place that assembles the list, and it is order-preserving so the fallback
        # still leads. It is written against the whole list rather than against the
        # fallback alone because the same collision can arrive from any two sources --
        # a health check and an advisory that quote each other -- and the panel is the
        # wrong place to discover it.
        seen: set[str] = set()
        out = [w for w in out if not (w in seen or seen.add(w))]
        return tuple(out)

    @property
    def median_pA(self) -> float:
        """The single-number open-pore current, the median of the baseline."""
        return float(np.median(self.baseline))

    def provenance(self, *, param_source: str | None = None) -> dict[str, Any]:
        """The flat ``baseline_*`` provenance block (3.5), ready to merge.

        ``param_source`` is one of `PARAM_SOURCES` -- how these parameters were
        chosen, which the staleness story needs and no measurement can recover
        afterwards. ``None`` uses the one the seam already recorded.

        THIS IS THE ONLY WRITER of the ``baseline_*`` keys. Four call sites merge
        the dict it returns; none of them assembles the keys itself, which is what
        makes a Schmitt row and an ML row on one recording carry the same
        ``baseline_fingerprint`` and be directly comparable.
        """
        prov = self.config.to_provenance(self.fs)
        prov["baseline_param_source"] = str(
            self.param_source if param_source is None else param_source
        )
        prov["baseline_median_pA"] = self.median_pA
        prov["baseline_polarity"] = int(self.polarity)
        if self.fallback:
            prov["baseline_fallback"] = str(self.fallback_token)
            prov["baseline_fallback_reason"] = str(self.fallback)
        if self.health is not None:
            prov["baseline_mask_fraction"] = self.health.mask_fraction
            prov["baseline_health"] = self.health.to_dict()
        return prov


def estimate_baseline_result(
    current: np.ndarray,
    fs: float,
    *,
    config: BaselineConfig | None = None,
    polarity: int | None = None,
    health: bool = True,
) -> BaselineResult:
    """Estimate I0(t) and judge it. One config in, one auditable value out.

    This is the entry point every call site should move to. `estimate_baseline`
    stays exactly as it was (a bare float64 array, backward compatible) for the
    callers and tests that depend on it.

    Args:
        current: 1-D current trace, pA. **Over the whole channel**, not a zoom
            rectangle: the mask threshold comes from a GLOBAL MAD, so a baseline
            estimated on a sub-range genuinely differs from the same window cut out
            of a full-trace estimate (measured penalty for slicing afterwards:
            16.95 pA, more than 1 sigma). The notebook truncates the RAW signal
            before filtering for exactly this reason.
        fs: Sample rate, Hz.
        config: The estimator configuration; ``None`` uses `DEFAULT_BASELINE`.
        polarity: ``{-1, +1}``, or ``None`` to measure it with `derive_polarity`.

            **PASS IT EXPLICITLY FOR ``masked_median``.** `derive_polarity` measures
            from a GLOBAL-MEDIAN centre, so on a drifting or gating RAW trace the
            coherent excursions it counts are the drift, not the events: measured on
            the owner's corpus it returns +1 for ferritin_data2 and data3 (and -1 on
            the same recordings once they are detrended, which is the frame the
            reference algorithm assumes). Pointing the one-sided mask the wrong way
            moves the baseline by up to 312 pA on data2 and 282 pA on data3.
            `derive_polarity` itself is frozen and must not be "fixed" for this --
            the seam should hand over the polarity the detector already established
            (``polarity_source`` provenance), or one measured on a detrended trace.
        health: Compute the `BaselineHealth` block. Costs one extra pass-1 median
            for ``'masked_median'`` (~80% of the estimate again at the default
            parameters, because rough_s is 5x base_s); the QC block alone is cheap.

    Raises:
        ValueError: non-finite input, a recording shorter than the tracking window,
            or an invalid parameter -- each with its own message.
    """
    cfg = DEFAULT_BASELINE if config is None else config
    x = require_finite(current)
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive number of Hz, got {fs!r}")
    _require_length(x.size, fs, cfg)
    cfg = cfg.resolve(fs)
    pol = derive_polarity(x) if polarity is None else require_polarity(polarity)

    # Liveness ladder (see `baseline_progress_scope`): the local shares are this
    # function's measured split at the 50 M-sample scale -- kernel estimate 4.6 s,
    # health rough pass ~1.7 s, health block 7.8 s of a 15.1 s total. On the no-health
    # path the estimate is the whole job, so it spans everything.
    _report_baseline_progress(0.0, "estimating baseline (tracking median)")
    if cfg.is_masked:
        base = _kernel_baseline(x, fs, polarity=pol, config=cfg)
    else:
        base = estimate_baseline(
            x, fs, polarity=pol, method=cfg.method, window_s=cfg.window_s
        )

    if not health:
        return BaselineResult(baseline=base, config=cfg, fs=fs, polarity=pol)

    _report_baseline_progress(0.35, "judging the baseline (pass-1 mask)")
    mask = rough_sigma = upper_mad = None
    if cfg.method == "masked_median":
        _, mask, rough_sigma, upper_mad = _rough_pass(x if pol == -1 else -x, fs, config=cfg)
    with _progress_span(0.5, 1.0):
        h = baseline_health(
            x, base, fs, config=cfg, polarity=pol, mask=mask,
            rough_sigma_pA=rough_sigma, upper_mad_sigma_pA=upper_mad,
        )
    return BaselineResult(baseline=base, config=cfg, fs=fs, polarity=pol, health=h)


# ===========================================================================
# THE SEAM -- one door, so a fifth call site cannot pick its own default
# ===========================================================================
# The defect this whole change removes was NOT that three call sites held three
# window constants. It was that nothing stopped them: `estimate_baseline` is a
# public function with a `window_s=` keyword and a silent default, so every site
# that hit a baseline problem fixed it locally (10 ms in `detect`, 200 ms in
# `ml_modules` after a real zoom-dependence bug, a +-3 ms flank median in
# `recharacterize_span`) and the app reported three depths for one event.
#
# Giving each site a `baseline_config=None` parameter fixes THOSE FOUR SITES.
# It does not fix the class: the fifth site still writes
# ``estimate_baseline(x, fs, window_s=0.2)`` and nothing objects. So the
# guarantee is moved to where both must pass through:
#
#   * `characterisation_baseline` is the ONLY door to a characterisation
#     baseline. It is the only function outside this module's own internals that
#     any app module may call for one.
#   * `baseline_config_from_params` is the ONLY forward resolver (GUI params ->
#     config), and `BaselineConfig.reproducible_from` the ONLY backward one
#     (stored rows -> the config that measured them). No call site parses a
#     ``baseline_*`` key itself.
#   * `DEFAULT_BASELINE` is the only baseline-parameter literal in the app, and
#     ``tests/analysis/test_baseline_seam.py`` holds AST guards that fail the
#     build if another module calls `estimate_baseline`, defines a baseline
#     window constant, or passes a literal window to the door.
#
# A fifth call site therefore cannot express its own default without deleting a
# test that says, in words, why it may not.


#: The provenance/param keys `BaselineConfig` round-trips. Param name IS provenance
#: name, so `modules.preprocess_params_from_provenance` needs no translation entry.
BASELINE_PARAM_KEYS: tuple[str, ...] = tuple(
    f"baseline_{name}" for name in BaselineConfig._FIELDS
)

#: How the parameters in force were chosen. Stamped as ``baseline_param_source``;
#: no measurement can recover it afterwards, and the staleness story needs it.
#:
#: ``'batch'`` -- the windows were COPIED from the batch form, which was filled in
#: with a DIFFERENT RECORDING selected, and this recording was never measured. That
#: is all the stamp knows: `resolve_param_source` gives it to every non-default form
#: source alike -- a confirmed proposal, a hand edit, a restored analysis -- and
#: keeps none of them, so the sentences that explain it must not name one (E4; see
#: `BATCH_PROVENANCE_CLAUSE`). It is a claim about the windows' provenance, not
#: about whether they are wrong: applying
#: one parameter set to every open recording is what Batch detect is for. What was
#: wrong was the CLAIM the batch used to stamp -- ``'proposed'`` ("measured on this
#: recording, and confirmed") on files nobody had probed -- which also silenced the
#: unconditional never-measured advisory exactly where it was least entitled to;
#: measured on the owner's -1000 mV file: 98 events at a median 95.11 pA under the
#: -600 mV file's 2.0 s / 0.5 s, against 115 at 90.33 pA at the honest default
#: (S7-DET1). See `resolve_param_source`.
#:
#: ``'fallback'`` is LEGACY: a project written before sweep 7 can carry it, because
#: the length clamp used to OVERWRITE the caller's source with it -- which turned
#: the never-measured advisory off on every short segment (S7-DET3). The clamp is
#: now recorded where it always also was, in ``baseline_fallback`` /
#: ``baseline_fallback_reason``, and the source stays what the caller said.
PARAM_SOURCES = ("default", "proposed", "user", "restored", "fallback", "batch")

#: The `PARAM_SOURCES` value for windows copied from the batch form (see above).
BATCH_SOURCE = "batch"

#: What a ``'batch'`` stamp can say about the windows' provenance, and no more: the
#: ONE clause the surfaces that explain the stamp are built from -- the result
#: advisory (`batch_copied_warning`) here, and the BaselinePanel's source badge for
#: a ``'batch'`` result -- so they cannot drift apart or outrun the stamp again.
#: Subject-less on purpose, so it reads after either prefix ("...COPIED FROM THE
#: BATCH FORM (1.5 s / 0.5 s): " and "These values: COPIED from the batch form: ").
#: What IS known is `BATCH_SCOPE_KEY` -- which recording was selected when the form
#: was filled in -- and that this recording was never probed; whether that form held
#: a confirmed proposal, a hand edit or a restored analysis is not recorded, and a
#: clause that names one of them ("proposed or confirmed", as sweep 7 first wrote)
#: was measured false on the owner's own files the first time a hand-typed form was
#: batch-detected: 1.5 s / 0.5 s typed into the -600 mV form ("set by hand"), Batch
#: detect, and the -1000 mV result's advisory and badge both asserted a proposal
#: and a confirmation that never happened (E4).
BATCH_PROVENANCE_CLAUSE = (
    "set for a DIFFERENT recording -- the one selected when the batch form was "
    "filled in -- and this recording was never measured"
)

#: Params key naming the recording the form's ``baseline_param_source`` is ABOUT --
#: the ``(doc_id, channel)`` that was selected when the BaselinePanel's windows were
#: proposed and confirmed, typed by hand, or restored (the panel's per-recording
#: target, whatever the source). Stamped by ``AnalysisPanel.batch_detection_params``
#: and read back by
#: `resolve_param_source` inside each detector run, which is the only place that
#: knows which recording it is measuring. It is in the ``baseline_`` namespace so
#: `modules._detect_kwargs` never forwards it to a detector function; it is NOT
#: written into the result's ``params`` (the detector records the RESOLVED source
#: there), so a per-load ``doc_id`` never reaches a project file.
BATCH_SCOPE_KEY = "baseline_param_scope"


def resolve_param_source(
    params: Mapping[str, Any] | None, target: Any, source: str
) -> str:
    """The ``baseline_param_source`` to stamp on a run over ``target``.

    ``source`` is what `baseline_config_from_params` resolved from the form; it is
    the truth for the recording the form was filled in for, and for every recording
    when the form is at the app default (nobody measured anything; the advisory
    fires everywhere, as it should). When the params carry `BATCH_SCOPE_KEY` -- the
    recording that was selected when the form was filled in -- a run over ANY OTHER
    target is stamped `BATCH_SOURCE`, so its provenance, its exported ``.meta.json``
    and its project entry say "copied from the batch form" rather than "a human chose
    these for this file". The stamp is the same for every non-default ``source``
    (``'proposed'``, ``'user'``, ``'restored'``, a legacy ``'fallback'``, or a form
    seeded from an earlier ``'batch'`` result) and the form's source is NOT kept
    beside it, which is why `batch_copied_warning` says only what the stamp knows
    (E4). Without the key, or at ``'default'``, ``source`` is returned as is, so the
    interactive Detect button is untouched.
    """
    if source == "default" or not params:
        return source
    scope = params.get(BATCH_SCOPE_KEY)
    if scope is None:
        return source
    try:
        same = tuple(scope) == tuple(target)
    except TypeError:
        same = scope == target
    return source if same else BATCH_SOURCE


def batch_copied_warning(config: "BaselineConfig") -> str:
    """The never-measured advisory for a `BATCH_SOURCE` run, naming the copied windows.

    The sibling of `UNPROBED_DEFAULT_WARNING`, and unconditional for the same reason:
    the health checks provably do not cover a window nobody measured on THIS
    recording, and the one thing this path knows is that the windows came from
    somewhere else.

    Built on `BATCH_PROVENANCE_CLAUSE`, so it claims nothing the stamp cannot back.
    The first sweep-7 version read "they were proposed or confirmed for a DIFFERENT
    recording" -- true of a batch sent from a confirmed proposal, false of one sent
    from a hand-typed form, and `resolve_param_source` stamps the two identically.
    Measured on the owner's own files (1.5 s / 0.5 s typed into the -600 mV form,
    Batch detect, the -1000 mV result): the advisory asserted a proposal and a
    confirmation that never happened (E4). The "never measured" half was always
    true; it is the provenance half that is now worded for what is known.
    """
    return (
        "These depths were measured with open-pore windows COPIED FROM THE BATCH FORM "
        f"({config.rough_s:g} s / {config.base_s:g} s): {BATCH_PROVENANCE_CLAUSE}. "
        "The correct window is a property of the recording, not of the run -- on the "
        "owner's own four files it ranges from 0.03 s (a gating pore) to 2.0 s (a "
        "521 ms partial clog). Select this recording, open the Preprocess tab and "
        "press “Measure this recording” before trusting a depth from this table."
    )

#: Fewest samples the clamped two-pass estimator will run on: below this even the
#: decimated median of the clamped ``base_s`` has fewer than `MIN_TAPS` taps, which
#: is the regime where the reference's "baseline" is one sample broadcast over the
#: whole trace. Derived, not chosen -- see `adapt_to_length`.
def _min_clamped_samples(config: "BaselineConfig") -> int:
    ratio = (float(config.base_s) / float(config.rough_s)
             if config.method == "masked_median" else 1.0)
    return int(math.ceil(MIN_TAPS / max(ratio, 1e-12)))


def adapt_to_length(
    config: "BaselineConfig", n: int, fs: float
) -> tuple["BaselineConfig", str | None]:
    """``(config that ``n`` samples can actually host, reason token or None)``.

    CLAMPING IS NOT AUTO-TUNING, AND THE DIFFERENCE IS THE WHOLE ARGUMENT.
    This module refuses (R5) to *search* for a window, because three of the four
    published QC scalars improve monotonically as the window shrinks, so any search
    against them selects the shortest possible window -- the exact failure the
    masked estimator exists to fix. There is no search here. There is one width the
    data can host, ``n / fs``, it is not chosen from a set of candidates, no metric
    is consulted, and the result is stamped, fingerprinted differently and warned
    about. The ratio between the two windows -- what actually decides the two-pass
    structure -- is the user's own and is preserved exactly.

    WHY THIS REPLACED THE BLOCK-MEDIAN FALLBACK. The previous answer here stood the
    pre-change 10 ms block median up in place of the configured estimator. Measured
    on the owner's ``..._0006_Process_-600 mV.abf``, over 27 real events cut into
    segments centred on them, reporting depth as a fraction of the depth the
    whole-recording masked median gives the same event:

        segment length      1 000    1 536    3 072    5 000   10 000   50 000
        block median 10 ms  0.020    0.187    0.011    0.014    0.014    0.014
        clamped masked      0.013    0.014    0.958    0.976    0.990    0.998

    The fallback that existed to avoid a degenerate answer WAS the degenerate
    answer: at every length it reported ~1.4% of the depth -- a 70x understatement,
    with the baseline sitting inside the event -- which is precisely the defect the
    two-pass estimator was brought in to remove. Clamping reproduces the
    whole-recording answer to within 1-4% from ~2 000 samples up.

    Below that it collapses too (0.013 at 1 000 samples), for the documented reason:
    ``rough_s`` must be >= ~3-4x the longest state it is meant to expose, and these
    events are ~20 ms dwell, so a 20-30 ms rough window IS the event. THAT REGIME IS
    NOT CLAIMED TO BE SOUND, and the honest comparison is how often each estimator is
    wrong WITHOUT SAYING SO. Over all 236 usable events, counting segments whose depth
    came out below half the whole-recording answer, and how many of those returned
    ``health.ok``:

        segment length     1 536             2 048            3 072           5 000
        clamped masked     222 wrong, 13 silent   50 / 0      0 / 0           1 / 0
        block median       236 wrong, 207 silent  236 / 235   236 / 19        237 / 16

    So the block median was wrong on EVERY segment at EVERY length and silent about
    it up to 99.6% of the time; the clamp is sound from ~2 000 samples up and, where
    it is not, is silent 5.9% of the time instead of 87.7%. That residual is real and
    is not designed away: the 13 have a mask fraction (0.07-0.24) and a rough-sigma
    ratio (2.1) squarely inside the healthy band, while the SOUND segments at that
    length sit ABOVE `MASK_FRACTION_CEILING` (0.515) -- on a segment this short a real
    event legitimately fills a quarter of it, so that ceiling stops discriminating.
    Retuning it against one recording is the auto-tune this module refuses; the floor
    on this rung is instead that a shorter segment is a worse measurement, said out
    loud in the warning `characterisation_baseline` attaches.

    The ladder, in order, each rung named in provenance:

    1. ``None`` -- the configured estimator fits; nothing is touched. **This is the
       only rung a normal-length recording can ever reach**, so no shipped number
       moves.
    2. `FALLBACK_CLAMPED` -- masked methods, ``rough_s`` clamped to the whole
       recording (the widest, most event-rejecting window that exists here) and
       ``base_s`` scaled to hold the config's own ``base_s/rough_s`` ratio.
       ``decim`` is unpinned so `derive_decim` keeps `TAPS_TARGET` taps at the new
       width -- a stride pinned for a 0.1 s window is meaningless at 0.02 s.
    3. `FALLBACK_TOO_SHORT` -- ``'constant'``, the global median. Reached when the
       clamped ``base_s`` cannot host `MIN_TAPS` taps even at stride 1 (15 samples
       at the default 5:1 ratio), and for the block/percentile/asls methods, whose
       clamped answer IS a single global statistic -- ``'constant'`` is that answer
       under its true name, and the one method with no window to misreport.
    """
    n = int(n)
    fs = float(fs)
    if config.method == "constant":
        return config, None
    if not (np.isfinite(fs) and fs > 0) or n <= 0:
        return config, None
    if n >= int(math.ceil(config.tracking_window_s * fs)):
        return config, None
    if config.is_masked and n >= _min_clamped_samples(config):
        if config.method == "robust_median":
            # One window, and it IS the tracking window; there is no second pass whose
            # ratio has to be held.
            fitted = config.replace(window_s=n / fs, decim=None)
        else:
            rough = n / fs
            base = rough * (float(config.base_s) / float(config.rough_s))
            fitted = config.replace(rough_s=rough, base_s=base, decim=None)
        # `_min_clamped_samples` is the floor in exact arithmetic; `derive_decim` is
        # the floor the FLOATS actually reach (n = 15 at the default ratio gives
        # base_s * fs = 2.9999999999999996, which is not 3 taps). Asking the same
        # function the estimator will ask is the only way the two cannot disagree.
        try:
            fitted.resolved_decim(fs)
        except ValueError:
            pass
        else:
            return fitted, FALLBACK_CLAMPED
    return BaselineConfig(method="constant", window_s=_LEGACY_WINDOW_S), FALLBACK_TOO_SHORT


def baseline_config_from_params(
    params: Mapping[str, Any] | None,
) -> tuple["BaselineConfig", str]:
    """``(config, param_source)`` for a FRESH measurement, from a module params map.

    The one forward resolver. A params mapping carrying no ``baseline_*`` key gets
    `DEFAULT_BASELINE` and ``'default'``; one carrying any of them gets that config
    and ``'user'``, unless it names its own ``baseline_param_source`` (the proposal
    dialog stamps ``'proposed'``; a restored project stamps ``'restored'``).

    Deliberately tolerant of extra keys -- it is handed the whole ``ctx.params``.
    """
    if not params:
        return DEFAULT_BASELINE, "default"
    explicit = params.get("baseline_config")
    if isinstance(explicit, BaselineConfig):
        source = str(params.get("baseline_param_source", "user"))
        return explicit, source if source in PARAM_SOURCES else "user"
    present = [k for k in BASELINE_PARAM_KEYS if k in params]
    if not present:
        return DEFAULT_BASELINE, "default"
    cfg = BaselineConfig.from_provenance(
        {k: params[k] for k in present} | {"baseline_method": params.get(
            "baseline_method", DEFAULT_BASELINE.method)}
    )
    source = str(params.get("baseline_param_source", "user"))
    return cfg, source if source in PARAM_SOURCES else "user"


def repair_nonfinite(signal: np.ndarray) -> tuple[np.ndarray, int]:
    """``(trace, n_repaired)`` with non-finite samples linearly interpolated away.

    `require_finite` REFUSES a non-finite trace, and that is the right contract for a
    measurement someone is about to act on: a NaN reaching the masked-median kernel
    makes ``noise`` NaN, which makes ``rough < median - 3.5*noise`` all-False, which
    **silently disables the entire event mask** and returns a perfectly finite
    baseline that is 12 pA wrong.

    A ROW REBUILD is the one caller that cannot accept that refusal, for a reason
    this codebase has already paid for once: one bad sample must cost the rows that
    READ it and no others. Rebuilds re-derive the whole-recording baseline so a
    rebuilt row lands where its neighbours did -- and a whole-recording estimate that
    refuses turns every row in the table into a fallback measurement, which is the
    blast radius going from "the pad" to "the table". Measured: one ``inf`` at sample
    1.9 M moved row 0's ``baseline_pA`` from -27.16 to -31.13 pA.

    Interpolating first keeps it local *because the estimator is local*: a rolling
    median of window ``w`` can only be moved by an input within ``w/2``, so every row
    further away than half the tracking window stays byte-identical. The rows near
    the repaired sample are still poisoned -- by
    `characterize.recharacterize_span`'s own flank rule, which reads their padded
    window directly and blanks the local baseline. Nothing is laundered at the row
    level; only the DISTANT rows are protected.

    An all-non-finite trace is returned unchanged with its count, so the caller's
    refusal path still fires rather than this inventing a signal.
    """
    x = np.asarray(signal)
    finite = np.isfinite(x)
    n_bad = int(finite.size - np.count_nonzero(finite))
    if n_bad == 0 or n_bad == finite.size:
        return x, n_bad
    idx = np.arange(x.size)
    out = np.asarray(x, dtype=np.float64).copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out, n_bad


#: How much coarser than `derive_decim` the polarity probe's own detrend may run.
#: Only the SIGN of the excursions is wanted, not a baseline, so 4x the stride --
#: 16x cheaper, cost being quadratic in decim -- is ample.
_POLARITY_DECIM_COARSEN = 4

#: The threshold `derive_polarity` counts runs against. Passed EXPLICITLY below so
#: the mirror probe and the vote are the same measurement by construction rather
#: than by two literals happening to agree.
_POLARITY_K = 5.0

#: The detrend scales the direction is measured at, seconds. Half-decade steps.
#:
#: This ladder is a property of the RECORDING, not of anybody's baseline parameters,
#: and that is the whole point -- see `_polarity_ladder`. The ends are measured, not
#: chosen: 0.01 s is the narrowest pass-2 window this app ships (``ferritin_data3``'s
#: ``base_s``), below which a detrend starts eating the events themselves; 1.0 s is
#: the widest scale at which the owner's gating pore still leaves anything to measure
#: (at 2 s and 5 s its residual MAD reaches 73 and 91 pA and NOTHING crosses 5 sigma,
#: so `derive_polarity` takes its documented flat-trace exit and returns +1 -- a
#: default reported as a measurement, which is exactly the defect this ladder removes).
_POLARITY_LADDER_S = (0.01, 0.03, 0.1, 0.3, 1.0)

#: A ladder rung must fit this many times into the recording, or it is not a detrend
#: of that recording -- it is a constant, and the residual is the whole trace.
_POLARITY_LADDER_WINDOWS = 4.0


def _polarity_measured(resid: np.ndarray) -> tuple[int, bool]:
    """``(derive_polarity(resid), did it actually measure anything)``.

    The second element is obtained by asking the FROZEN function itself, using only
    its own documented antisymmetry: counting coherent runs beyond ``k`` sigma either
    side of a robust centre is exactly sign-antisymmetric, so a *measured* answer must
    flip when the residual is negated. Both of `derive_polarity`'s no-evidence exits
    -- a flat trace (``sigma`` not > 0) and a trace where nothing crosses the
    threshold either way (``n_up == n_down == 0``, whose mass tie-break is
    ``0.0 >= 0.0``) -- return ``+1`` for a residual AND for its negative, so they are
    detected here without re-implementing a single line of the statistic.

    That matters because those exits are reachable on real data at a wide detrend
    window: on ``ferritin_data3`` at a 5 s rough window the residual MAD is 91 pA and
    no run reaches 5 x 91, so the ``+1`` handed back is the documented default. The
    baseline probe's own frame on that recording IS 5 s.

    BOTH no-evidence exits return ``+1``, so a ``-1`` cannot be one of them and needs
    no mirror. That is not an approximation, it is the exit conditions read off: the
    flat-trace branch is a bare ``return 1``, and the ``n_up == n_down == 0`` branch
    compares ``0.0 >= 0.0``. Skipping the mirror on a ``-1`` halves the cost of this
    function on every recording the owner has (all four vote ``-1`` on 19 of 20 rungs)
    and costs nothing in fidelity.
    """
    r = np.asarray(resid, dtype=np.float64)
    up = int(derive_polarity(r, k=_POLARITY_K))
    if up < 0:
        return up, True
    # Only the +1 branch pays for the mirror, and it copies rather than negating in
    # place: ``resid`` may be a read-only view, and a probe must never write to the
    # array it was handed.
    return up, up != int(derive_polarity(-r, k=_POLARITY_K))


def _polarity_rung_windows(n: int, fs: float) -> tuple[tuple[float, int], ...]:
    """``(window_s, decim)`` for the rungs of `_POLARITY_LADDER_S` this trace supports.

    Cheap -- no arithmetic over the trace -- so the vote below knows how many rungs
    are still to come and can stop as soon as the remainder cannot overturn it.
    """
    fs_f = float(fs)
    if not (np.isfinite(fs_f) and fs_f > 0) or n <= 0:
        return ()
    duration = n / fs_f
    out: list[tuple[float, int]] = []
    for win in _POLARITY_LADDER_S:
        if win * _POLARITY_LADDER_WINDOWS > duration:
            continue
        try:
            decim = derive_decim(win, fs_f) * _POLARITY_DECIM_COARSEN
        except ValueError:
            continue
        if int(win * fs_f / max(decim, 1)) < MIN_TAPS:
            continue
        out.append((float(win), int(decim)))
    return tuple(out)


def _polarity_rungs(
    x: np.ndarray, fs: float
) -> "Iterable[tuple[float, int, bool]]":
    """Yield ``(window_s, vote, measured)`` rung by rung, narrowest first."""
    xf = np.asarray(require_finite(x), dtype=np.float64)
    fs_f = float(fs)
    rungs = _polarity_rung_windows(xf.size, fs_f)
    n_rungs = max(len(rungs), 1)
    for i, (win, decim) in enumerate(rungs):
        # Liveness rides the rung loop the cancel poll already walks: each rung is
        # one decimated median over the whole trace, seconds each at the 50 M-sample
        # scale (see `baseline_progress_scope`). The fraction is position in the
        # ladder, reported BEFORE the rung's work so the bar moves when it starts.
        _report_baseline_progress(i / n_rungs,
                                  f"measuring polarity ({i + 1}/{n_rungs})")
        check_cancelled()
        resid = xf - np.asarray(
            decimated_median(xf, fs_f, win, decim), dtype=np.float64
        )
        check_cancelled()
        vote, measured = _polarity_measured(resid)
        yield float(win), vote, bool(measured)


def _polarity_ladder(
    x: np.ndarray, fs: float
) -> tuple[tuple[float, int, bool], ...]:
    """``(window_s, vote, measured)`` at EVERY usable rung of `_POLARITY_LADDER_S`.

    The whole ladder, with no short-circuit -- exposed (privately) so a test and a
    diagnostic can read the evidence rather than only the verdict. `_polarity_for`
    consumes the same rungs lazily and stops early, which is why the two are written
    against one generator instead of one calling the other.
    """
    return tuple(_polarity_rungs(x, fs))


class _PolarityMemo:
    """Memoise the mask DIRECTION for one trace object. Same contract as
    `_BaselineCache`, one rung down, and sound for the same reason the ladder exists:
    the direction depends on ``(samples, fs)`` and on NOTHING else, so measuring it
    once per trace is not an optimisation of a decision, it IS the decision.

    WHY. `characterisation_baseline` has to know the polarity BEFORE it can build its
    cache key -- the key contains it -- so the ladder ran on every call, hit or miss.
    Measured on the owner's -600 mV apo recording (2,993,798 samples), five detections
    over ONE array with the baseline already memoised, interleaved, min of three:
    **1.508 s with the single-window derivation, 2.283 s with the ladder** (+155 ms
    per detection, x1.51). `sweep_threshold` is 32 such detections and is already the
    slowest thing in the app at 18.9 s, so the ladder was charging it ~5 s for an
    answer it had worked out 31 times already.

    THE KEY IS THE CALLER'S OBJECT, not the float64 copy this module works in.
    `characterisation_baseline` converts a float32 recording on every call -- and the
    app's conditioned trace IS float32 -- so keying on the converted array would miss
    every time and memoise nothing. Hence ``_polarity_for(..., key=...)``: the caller
    names the object whose identity stands for these samples, exactly as it already
    does for `_BaselineCache`. A float32 -> float64 widening is exact, so the two
    objects carry the same numbers.

    Validated through a `weakref` like `_BaselineCache`, for the same reasons: a
    recycled ``id()`` cannot produce a hit, and a different array with identical
    contents simply misses and re-measures -- correct, just slower. IN-PLACE MUTATION
    OF A CACHED TRACE WOULD RETURN A STALE DIRECTION, which is safe on exactly the
    convention the baseline cache already relies on: `modules.preprocess_for_detection`
    returns a fresh array whenever preprocessing changes, so nobody mutates one.

    The memo is consulted AFTER `require_finite`, never before, so a trace carrying a
    NaN still raises on the ``on_nonfinite='refuse'`` path instead of being answered
    from an entry that an ``'interpolate'`` call had left behind.
    """

    __slots__ = ("_entries", "_max")

    def __init__(self, maxsize: int = 8) -> None:
        self._entries: dict[tuple[Any, ...], tuple[Any, int]] = {}
        self._max = int(maxsize)

    @staticmethod
    def _key(signal: np.ndarray, fs: float) -> tuple[Any, ...]:
        return (id(signal), int(signal.size), signal.dtype.str, float(fs))

    def get(self, key: tuple[Any, ...], signal: np.ndarray) -> "int | None":
        hit = self._entries.get(key)
        if hit is None:
            return None
        ref, polarity = hit
        if ref() is not signal:
            self._entries.pop(key, None)
            return None
        return polarity

    def put(self, key: tuple[Any, ...], signal: np.ndarray, polarity: int) -> None:
        import weakref

        entries = self._entries

        def discard(dead_ref, *, cache_key=key) -> None:
            # The callback can run after an id has been recycled and a new entry has
            # replaced this key. Remove only the entry that still owns THIS weakref.
            hit = entries.get(cache_key)
            if hit is not None and hit[0] is dead_ref:
                entries.pop(cache_key, None)

        try:
            ref = weakref.ref(signal, discard)
        except TypeError:            # not weak-referenceable -> simply do not cache
            return
        if len(self._entries) >= self._max:
            self._entries.pop(next(iter(self._entries)), None)
        self._entries[key] = (ref, int(polarity))

    def clear(self) -> None:
        self._entries.clear()


_POLARITY_MEMO = _PolarityMemo()


def _polarity_for(
    x: np.ndarray,
    fs: float,
    cfg: "BaselineConfig | None" = None,
    *,
    key: "np.ndarray | None" = None,
) -> int:
    """Which way the events run -- a property of the RECORDING, not of the parameters.

    `derive_polarity` is frozen and correct for what it does: it counts coherent runs
    beyond 5 sigma either side of the GLOBAL MEDIAN. On a drifting or gating RAW trace
    those runs are the DRIFT, not the events -- measured on the owner's own corpus it
    returns **+1 for ferritin_data2 and ferritin_data3** (and -1 on the same two once
    detrended, which is the frame the reference algorithm assumes). The masked
    estimator's mask is ONE-SIDED, so pointing it the wrong way moves the baseline by
    up to **312 pA** on data2 and 282 pA on data3. So the answer is taken on a
    detrended copy, and `derive_polarity` is simply handed the frame it assumes.

    WHAT CHANGED, AND WHY. This used to detrend at ONE window -- the configured
    estimator's own ``rough_s``, at a stride derived from its ``base_s``. Measured on
    ``ferritin_data3``, that made the mask direction an artefact of parameters nobody
    had confirmed:

    * at a 0.5 s rough window, stride 264 gives -1 and stride 333 gives +1. The stride
      is derived from ``base_s``, a PASS-2 parameter with nothing to do with polarity,
      times `_POLARITY_DECIM_COARSEN`, a cost constant. Change either and the mask
      flips end for end.
    * at the baseline probe's own frame (5 s on that recording) nothing crosses
      5 sigma at all, so the +1 handed back is `derive_polarity`'s documented
      flat-trace default -- reported as though it had been measured. Handed +1, the
      probe reads the apo file's decisive 496 ms clog as 27.8 ms and proposes
      ``rough_s = 0.1`` where the known-good answer is 2.0 -- a 20x under-sized window,
      with ``confident=True``.

    So the direction is now measured across a fixed ladder of detrend scales
    (`_POLARITY_LADDER_S`); rungs that measured NOTHING are dropped rather than voted
    (`_polarity_measured`); and the surviving rungs vote with HARMONIC weights,
    ``1, 1/2, 1/3, 1/4, 1/5``, narrowest first.

    Why weighted and not a plain majority. A rung's authority falls with its window,
    because the wider the detrend the more of the recording's own wander survives into
    the residual to be counted as an excursion -- on a gating pore the mid-ladder rungs
    count the CONDUCTANCE STEPS, and a step is one run up and one run down, so those
    rungs sit near a tie and their sign is noise. Harmonic weights make the narrowest
    rung the single strongest voice while still letting the rest of the ladder overturn
    it (1 against 1.283), so no one window decides and no window is selected by a
    parameter. A plain majority was measured getting a fast-gating fixture (290 pA
    steps every 20 ms, 120 pA blockades) wrong 1 case in 9; the weighted vote gets all
    9 right, along with the four corpus recordings and the owner's five readable ABFs.

    ``cfg`` is accepted and IGNORED. It selected the window, and that WAS the defect;
    it stays in the signature because both call sites have one to hand, and because a
    reader who reaches for it should find this paragraph. `derive_polarity` itself is
    untouched.

    A caller that KNOWS the polarity (the detector's ``polarity_source``) should still
    pass it; this is the fallback, not the preferred path.

    ``key`` names the object whose identity stands for these samples in
    `_PolarityMemo` -- pass it when ``x`` is a per-call conversion of a longer-lived
    array, which is what `characterisation_baseline` hands over. It changes no answer;
    it only decides whether the ladder is re-run for a trace already measured.
    """
    xf = np.asarray(require_finite(x), dtype=np.float64)
    # AFTER require_finite: see `_PolarityMemo`. The identity that counts is the
    # caller's, because `xf` may be a fresh float64 copy made for this call alone.
    memo_of = np.asarray(x if key is None else key)
    memo_key = _PolarityMemo._key(memo_of, fs)
    hit = _POLARITY_MEMO.get(memo_key, memo_of)
    if hit is not None:
        return hit
    pol = _polarity_vote(xf, fs)
    _POLARITY_MEMO.put(memo_key, memo_of, pol)
    return pol


def _polarity_vote(xf: np.ndarray, fs: float) -> int:
    """The ladder vote itself, unmemoised. `_polarity_for` is this plus the memo."""
    weights = [1.0 / (i + 1) for i in range(len(_polarity_rung_windows(xf.size, fs)))]
    total = 0.0
    first_measured: int | None = None
    for i, (_, vote, measured) in enumerate(_polarity_rungs(xf, fs)):
        if measured:
            if first_measured is None:
                first_measured = vote
            total += vote * weights[i]
        # Every rung still to come is worth at most its own weight, so once the lead
        # exceeds their sum the outcome is settled and the rest is arithmetic nobody
        # has to pay for. On a recording whose ladder agrees this stops at the second
        # rung: 1 + 1/2 = 1.5 already beats 1/3 + 1/4 + 1/5 = 0.783.
        if abs(total) > sum(weights[i + 1:]):
            break
    if first_measured is None:
        # No scale this recording supports measured anything -- a trace too short for
        # the narrowest rung, or one with no coherent excursion at any scale. Fall
        # back to the frozen function on the trace as given, which is what this
        # returned before there was a ladder.
        return int(derive_polarity(xf, k=_POLARITY_K))
    if total != 0.0:
        return 1 if total > 0.0 else -1
    # An exact cancellation. The NARROWEST rung that measured something wins: it is the
    # frame in which the least drift can have survived the detrend, and therefore the
    # one whose runs are least likely to be the drift rather than the events.
    return int(first_measured)


def measured_polarity(
    signal: np.ndarray, fs: float, *, config: "BaselineConfig | None" = None
) -> int:
    """``{-1, +1}`` -- which way the events run, measured the way THE DOOR measures it.

    The public face of `_polarity_for`, and the only correct way for a module outside
    this one to ask the question. `derive_polarity` is the wrong function to call
    directly on a raw recording: it centres on the GLOBAL MEDIAN, so on a drifting or
    gating trace it counts the drift rather than the events and returns ``+1`` on
    ferritin_data2 and ferritin_data3. Pointing the one-sided mask the wrong way moves
    the baseline by up to 312 pA.

    ``config`` is accepted and IGNORED -- the direction is a property of the recording
    and no longer depends on anybody's window (see `_polarity_for`). It stays in the
    signature because `remeasure` has a config to hand and passing it is harmless; an
    invariant test asserts that every shipped configuration gives the same answer.
    """
    sig = np.asarray(signal)
    return _polarity_for(sig, float(fs), config, key=sig)


class _BaselineCache:
    """Memoise the whole-recording baseline for ONE trace object.

    `recharacterize_span` must land on the SAME baseline the detector used -- that
    is the entire point of unifying the four sites -- and the only way to guarantee
    it without a 24 MB array riding in a JSON provenance dict is to recompute the
    whole-recording estimate from the stored config. At 3 M samples that is ~0.3 s,
    which is fine once per detection and unusable once per merged run or once per
    mask drag in the Event Inspector.

    The key is the trace's OBJECT IDENTITY, validated through a `weakref` so a
    recycled ``id()`` can never produce a hit: if the original array has been
    collected the ref reads ``None`` and the entry is dropped. A different array
    with identical contents simply misses and recomputes -- correct, just slower.

    IN-PLACE MUTATION OF A CACHED TRACE WOULD RETURN A STALE BASELINE. The app's
    conditioned trace is rebuilt (a new array) whenever preprocessing changes --
    `modules.preprocess_for_detection` returns a fresh array every call -- so no
    caller mutates one; that convention is what this relies on.
    """

    __slots__ = ("_entries", "_max")

    def __init__(self, maxsize: int = 4) -> None:
        self._entries: dict[tuple[Any, ...], tuple[Any, "BaselineResult"]] = {}
        self._max = int(maxsize)

    @staticmethod
    def _key(signal: np.ndarray, fs: float, cfg: "BaselineConfig",
             polarity: int, health: bool, on_nonfinite: str = "refuse"
             ) -> tuple[Any, ...]:
        return (id(signal), int(signal.size), signal.dtype.str, float(fs),
                cfg.fingerprint(fs), int(polarity), bool(health), str(on_nonfinite))

    def get(self, key: tuple[Any, ...], signal: np.ndarray) -> "BaselineResult | None":
        hit = self._entries.get(key)
        if hit is None:
            return None
        ref, result = hit
        if ref() is not signal:
            self._entries.pop(key, None)
            return None
        return result

    def put(self, key: tuple[Any, ...], signal: np.ndarray,
            result: "BaselineResult") -> None:
        import weakref

        entries = self._entries

        def discard(dead_ref, *, cache_key=key) -> None:
            # Do not wait for a later lookup or four-entry eviction to release a
            # whole-recording float64 baseline. Dual-signal detection intentionally
            # creates short-lived candidate/measurement arrays sequentially; without
            # this callback their baselines survived the arrays and could retain
            # hundreds of MiB each. Identity-check the weakref so a recycled id cannot
            # evict a newer entry that happens to have the same tuple key.
            hit = entries.get(cache_key)
            if hit is not None and hit[0] is dead_ref:
                entries.pop(cache_key, None)

        try:
            ref = weakref.ref(signal, discard)
        except TypeError:            # not weak-referenceable -> simply do not cache
            return
        if len(self._entries) >= self._max:
            self._entries.pop(next(iter(self._entries)), None)
        self._entries[key] = (ref, result)

    def clear(self) -> None:
        self._entries.clear()


_CACHE = _BaselineCache()


def clear_baseline_cache() -> None:
    """Drop every memoised whole-recording baseline AND every memoised mask direction.

    Both are keyed on a trace's object identity and both are per-trace measurements,
    so a caller who wants the door to genuinely re-measure has to lose both -- one
    door, one reset. For tests and for a reload.
    """
    _CACHE.clear()
    _POLARITY_MEMO.clear()


def characterisation_baseline(
    signal: np.ndarray,
    fs: float,
    *,
    config: "BaselineConfig | None" = None,
    polarity: int | None = None,
    param_source: str = "default",
    health: bool = True,
    cache: bool = True,
    on_nonfinite: str = "refuse",
) -> "BaselineResult":
    """THE door. Every characterisation baseline in this app comes from here.

    `estimate_baseline_result` is the estimator's own strict contract: it refuses a
    trace shorter than the tracking window, it never memoises, and it knows nothing
    about how the parameters were chosen. This adds the three things the APP needs
    on top of it, in one place rather than four:

    1. **A length adaptation** (`adapt_to_length`). A recording shorter than
       ``base_s`` cannot be measured at the configured windows, and a hand-cut
       segment is a fully legal document (``core.segment.MIN_SEGMENT_SAMPLES`` is
       2), so an exception here is a document the app let the user create and then
       refuses to analyse. The configured estimator instead runs with its windows
       clamped to the recording -- *named*, separately fingerprinted, reported
       through `BaselineResult.health`, and surfaced as a warning. Read
       `adapt_to_length` for the measurements: clamping reproduces the
       whole-recording depth to within 1-4% from ~2 000 samples up, where the block
       median that used to stand in here reported 1.4% of it.
    2. **Memoisation** keyed on the trace object, so `recharacterize_span` can
       reproduce the detector's whole-recording baseline per event without paying
       for it per event.
    3. **``param_source``**, carried into provenance.

    Args:
        signal: The trace the rows are measured on, WHOLE CHANNEL. Never a zoom
            rectangle -- the mask threshold is a global MAD, so a baseline estimated
            on a sub-range genuinely differs from the same window cut out of a
            full-trace estimate (16.95 pA on the owner's file, more than 1 sigma).
        fs: Sample rate, Hz.
        config: ``None`` -> `DEFAULT_BASELINE`.
        polarity: PASS IT. `derive_polarity` measures from a global-median centre
            and returns +1 on the owner's own raw drifting traces, which points the
            one-sided mask the wrong way and moves the baseline by up to 312 pA.
            The detector's measured polarity is the right source.
        param_source: One of `PARAM_SOURCES`.
        health: Compute the `BaselineHealth` block (roughly doubles the cost).
            True once per analysis; False for a per-event rebuild.
        cache: Consult and populate the whole-recording memo.
        on_nonfinite: ``'refuse'`` (default) raises on a non-finite trace, which is
            what a measurement someone is about to act on must do.
            ``'interpolate'`` calls `repair_nonfinite` first -- ONLY for a row
            rebuild, and only because a whole-recording estimate that refuses turns
            one bad sample into a table-wide change of measurement. Read that
            function's docstring before using it anywhere else.

    Returns:
        A `BaselineResult` whose ``.provenance(param_source=...)`` is the flat
        ``baseline_*`` block every call site stamps.
    """
    cfg = DEFAULT_BASELINE if config is None else config
    src = str(param_source)
    # FLOAT64, ALWAYS. The kernel computes its rough pass in whatever precision its
    # input carries, so the same recording handed over as float32 by one caller and
    # float64 by another produces baselines that differ in the last bits -- and the
    # rebuilt row then disagrees with the detected one, which is the entire defect
    # this seam removes. `detect_events` has always upcast to float64 and
    # `estimate_baseline` has always returned float64; the door makes that the rule
    # for every caller rather than an accident of who held the array.
    #
    # (`estimate_baseline_result` is deliberately NOT changed: the corpus contract
    # runs through it with the notebook's own dtype trajectory, and casting there
    # would move bytes that must not move.)
    if on_nonfinite not in ("refuse", "interpolate"):
        raise ValueError(
            f"on_nonfinite must be 'refuse' or 'interpolate', got {on_nonfinite!r}"
        )
    orig = np.asarray(signal)          # the OBJECT the memo is keyed on
    n = int(orig.size)
    fs_f = float(fs)
    x = orig if orig.dtype == np.float64 else orig.astype(np.float64)
    if on_nonfinite == "interpolate":
        x, _ = repair_nonfinite(x)

    fallback = None
    token = FALLBACK_TOO_SHORT
    if np.isfinite(fs_f) and fs_f > 0:
        need = int(math.ceil(cfg.tracking_window_s * fs_f))
        fitted, token_or_none = adapt_to_length(cfg, n, fs_f)
        if token_or_none is not None:
            token = token_or_none
            asked = cfg
            cfg = fitted
            # ``src`` is deliberately LEFT ALONE. It used to become ``'fallback'``
            # here, which answered a different question from the one the key asks:
            # ``param_source`` says who CHOSE the windows, and the clamp chose
            # nothing -- it ran the caller's windows at the only widths this trace
            # can host. Overwriting it turned the unconditional never-measured
            # advisory (gated on ``'default'``) OFF on every short segment; measured
            # on a 4 000-sample cut of the owner's -600 mV file: source 'fallback',
            # windows 80 / 16 ms, advisory absent, the panel still showing the form's
            # 0.5 / 0.1 s (S7-DET3). The clamp is recorded in ``fallback`` /
            # ``fallback_token`` (-> ``baseline_fallback`` in provenance), where a
            # reader can see both facts at once.
            if token == FALLBACK_CLAMPED:
                fallback = (
                    f"{n} samples ({n / fs_f:g} s) is shorter than the "
                    f"{asked.tracking_window_s:g} s tracking window ({need} samples), so "
                    f"{asked.method!r} ran with its window(s) CLAMPED to this recording: "
                    f"{cfg.describe()}. The ratio between the two passes is unchanged; "
                    "only the widths are, and only because no wider window exists here. "
                    "Depths are NOT comparable with ones measured at the configured "
                    "window, and if the health block reports a mask fraction at the "
                    "floor or a rough-sigma ratio out of band, the clamped rough window "
                    "is as long as the events themselves and this estimate is "
                    "inoperative. The clamp is only sound while the rough window stays "
                    "several times the longest dwell -- on the owner's own file that is "
                    "from ~2 000 samples up, and below it the health block still misses "
                    "about one collapse in twenty. A longer cut is a better measurement."
                )
            else:
                why = (
                    f"even clamped to this recording its short pass would run on fewer "
                    f"than {MIN_TAPS} taps"
                    if asked.is_masked else
                    f"a {asked.method!r} window clamped to this recording is a single "
                    "block, i.e. one global statistic"
                )
                fallback = (
                    f"{n} samples ({n / fs_f:g} s) cannot support a tracking window at "
                    f"all ({asked.method!r} asked for {asked.tracking_window_s:g} s = "
                    f"{need} samples, and {why}). Measured as a GLOBAL MEDIAN, which "
                    "tracks no drift whatsoever. Depths are NOT comparable with ones "
                    "measured by the configured estimator."
                )

    pol = polarity if polarity is None else require_polarity(polarity)
    if pol is None:
        # ``key=orig``: ``x`` is a float32 -> float64 conversion made for THIS call
        # (the app's conditioned trace is float32), so its identity is worthless as a
        # memo key while ``orig`` is the very object `_BaselineCache` is keyed on.
        # See `_PolarityMemo` -- without this the ladder re-runs on every cache hit,
        # +155 ms per detection on the owner's -600 mV recording.
        #
        # The span shares here and below are the door's MEASURED cost split at the
        # 50 M-sample scale (`baseline_progress_scope`: ladder 5.3 s, estimate +
        # health 15.1 s of a 20.5 s door), so the bar a worker drives through this
        # scope moves at roughly the speed of the work. A memo hit skips the span --
        # the monotone carrier simply continues from wherever the last report was.
        with _progress_span(0.0, 0.25):
            pol = _polarity_for(x, fs_f, cfg, key=orig)

    key = None
    if cache and pol is not None:
        key = _BaselineCache._key(orig, fs_f, cfg, int(pol), bool(health),
                                  on_nonfinite)
        hit = _CACHE.get(key, orig)
        if hit is not None:
            return _decorated(hit, fallback, src, token)

    with _progress_span(0.25, 1.0):
        res = estimate_baseline_result(x, fs_f, config=cfg, polarity=pol,
                                       health=health)
    # THE UNDECORATED MEASUREMENT IS WHAT IS CACHED, and the decoration is applied on the
    # way out -- to a hit and to a miss, through the same function. Storing the decorated
    # object made `_CACHE` a memo of "the measurement AND whoever happened to compute it
    # first", which is exactly what its own docstring forbids, and it went wrong in both
    # directions: a repeat fallback call came back with `tracking_window_fits` stamped
    # TWICE (6 checks -> 7, 2 warnings -> 3, stable thereafter), and -- worse -- a config
    # that fits the trace exactly came back carrying `baseline_fallback =
    # 'windows_clamped_to_recording'` and *"Depths are NOT comparable ... this estimate is
    # inoperative"*, with `health.ok` flipping True -> False on a bit-identical
    # measurement, purely because a default-config call had run on the same array first.
    # (The key is built from the FITTED config, so `modules.baseline_config_from_params`
    # handing a provenance-derived config straight back to `detect_events` collides by
    # construction; measured through the shipped detector, the provenance keys that
    # differed run-to-run were exactly ['baseline_health', 'baseline_warnings'].)
    # `baseline_fallback` is in `scoring.BASELINE_CONTRACT_KEYS`, so it rode into every
    # export sidecar and every project file.
    if key is not None:
        _CACHE.put(key, orig, res)
    return _decorated(res, fallback, src, token)


def _decorated(
    res: "BaselineResult", fallback: str | None, src: str, token: str
) -> "BaselineResult":
    """``res`` stamped with THIS call's parameter provenance — the one door for both the
    cache hit and the fresh measurement.

    How the parameters were chosen is not part of the measurement and must never be
    inherited; the only way to guarantee that is for the memo to hold the bare measurement
    and for every path out of it to pass through here. Two call sites deciding this
    separately is what produced the stamp leak.
    """
    if fallback is not None:
        return _with_fallback(res, fallback, src, token)
    if res.param_source != src:
        return _dc_replace(res, param_source=src)
    return res


def _with_fallback(
    res: "BaselineResult", message: str, src: str, token: str = FALLBACK_TOO_SHORT
) -> "BaselineResult":
    """``res`` with the length adaptation recorded on its health block and source.

    The check's LEVEL distinguishes the two rungs, because they are not the same
    event: `FALLBACK_CLAMPED` means the configured estimator ran (measured within
    1-4% of the whole-recording answer from ~2 000 samples up), which is a ``'warn'``;
    `FALLBACK_TOO_SHORT` means nothing tracked anything, which is a ``'refuse'``
    reported rather than raised. Both leave `BaselineHealth.ok` False, so a caller
    that only reads ``ok`` is no worse off than before.
    """
    check = BaselineCheck(
        name="tracking_window_fits",
        level="warn" if token == FALLBACK_CLAMPED else "refuse",
        label="trace vs tracking window",
        value=float(res.baseline.size),
        threshold=f">= 1.0 x the tracking window ({token})",
        message=message,
    )
    health = res.health
    if health is not None:
        health = _dc_replace(health, checks=(*health.checks, check))
    return _dc_replace(
        res, health=health, param_source=src, fallback=message,
        fallback_token=str(token),
    )


def open_pore_current(
    current: np.ndarray, fs: float, baseline: np.ndarray | None = None
) -> float:
    """Robust single-number open-pore current I0 in pA (median of the baseline).

    Signed, not a magnitude: at -1000 mV this is negative. Take abs() yourself if
    you want |I0|. Reported as the median so events and drift excursions cannot
    drag it; on a drifting recording no single number is right -- use the
    full-length baseline for anything per-event.
    """
    b = (
        estimate_baseline(current, fs)
        if baseline is None
        else np.asarray(baseline, dtype=np.float64)
    )
    n = np.asarray(current).size
    if b.size != n:
        raise ValueError(f"baseline length {b.size} != recording length {n}")
    if b.size == 0:
        raise ValueError("cannot take the open-pore current of an empty recording")
    return float(np.median(b))


# ---------------------------------------------------------------------------
# baseline-wander removal (detrending) -- flatten the slow drift, keep events
# ---------------------------------------------------------------------------


def estimated_wander(
    signal: np.ndarray,
    fs: float,
    *,
    method: str = "block_median",
    window_s: float = DEFAULT_DETREND_WINDOW_S,
    polarity: int | None = None,
    **method_kwargs: Any,
) -> np.ndarray:
    """The slow baseline wander `remove_baseline_wander` will subtract, full-length.

    This is exactly the open-pore envelope `estimate_baseline` tracks, returned so a
    caller can OVERLAY it on the raw trace ("here is the drift I am about to remove")
    or diff two windows. It is the same estimate the removal uses, so what you preview
    is what you get. Returned as ``float32`` (the signal's discipline), not the
    ``float64`` `estimate_baseline` yields, because its only consumers are plotting and
    subtraction.

    Args mirror `remove_baseline_wander`; see `estimate_baseline` for the meaning of
    ``method`` / ``window_s`` / ``polarity`` and the extra ``method_kwargs``.
    """
    base = estimate_baseline(
        signal, fs, method=method, window_s=window_s, polarity=polarity, **method_kwargs
    )
    return np.ascontiguousarray(base, dtype=np.float32)


#: When `remove_baseline_wander`'s output peak wander exceeds its input's by this
#: factor, the stage warns that it AMPLIFIED the wander (`AmplifiedWanderWarning`).
#: Chosen against measured separations, with margin on both sides: the 3 Hz /
#: 0.5 s-window phase-inversion case reads 1.35-1.37 (with or without 5 pA noise and
#: real 80 pA events on top), a detrend doing its legitimate job reads 0.58-0.60, and
#: a flat trace where the stage has nothing to do reads 0.95. The RMS corroboration
#: (`_block_median_wander`'s first number must ALSO have grown) is what keeps a
#: single event-owned block — identical in input and output — from tripping the peak
#: alone.
DETREND_AMPLIFIED_PEAK_RATIO = 1.15

#: Blocks per detrend window for the self-measure: W/8 blocks resolve wander down to
#: ~a quarter of the window — comfortably covering the half-to-one-window band where
#: the block median phase-inverts — while each block still holds thousands of samples
#: at any real rate, so its median shrugs off noise and sub-block events.
_WANDER_BLOCKS_PER_WINDOW = 8

#: Fewest wander blocks the self-measure will judge. Below this there is no series to
#: read a peak from — and a window that coarse over the trace is already the regime
#: `ConstantBaselineWarning` reports (one block, constant estimate).
_WANDER_MIN_BLOCKS = 4


def _block_median_wander(x: np.ndarray, fs: float, window_s: float) -> tuple[float, float] | None:
    """``(rms, peak)`` of the block-median wander series of ``x``, in pA.

    The cheap self-measure `remove_baseline_wander` judges its own output with:
    medians of ``window_s / _WANDER_BLOCKS_PER_WINDOW`` blocks, centred on their own
    median. Medians, not means, for the usual reason — an event occupying less than
    half a block cannot move its block's number — and the SAME grid on input and
    output, so an event that does own a block contributes identically to both and
    cancels in the comparison. Returns ``None`` when fewer than `_WANDER_MIN_BLOCKS`
    blocks fit (nothing to measure; the degenerate-window case warns elsewhere).
    """
    a = np.asarray(x, dtype=np.float64)
    w = max(1, int(round(float(window_s) / _WANDER_BLOCKS_PER_WINDOW * float(fs))))
    n_blocks = a.size // w
    if n_blocks < _WANDER_MIN_BLOCKS:
        return None
    bm = np.median(a[: n_blocks * w].reshape(n_blocks, w), axis=1)
    bm = bm - np.median(bm)
    return float(np.sqrt(np.mean(bm * bm))), float(np.abs(bm).max())


def remove_baseline_wander(
    signal: np.ndarray,
    fs: float,
    *,
    method: str = "block_median",
    window_s: float = DEFAULT_DETREND_WINDOW_S,
    polarity: int | None = None,
    **method_kwargs: Any,
) -> np.ndarray:
    """Flatten slow baseline drift while PRESERVING event shapes and depths.

    Tracks the time-varying open-pore level ``base(t)`` with `estimate_baseline`
    (which excludes events by construction, given a sane ``window_s``), then subtracts
    it and adds back a single representative level so the trace is not re-zeroed:

        base      = estimate_baseline(signal, fs, method, window_s, polarity)
        reference = median(base)            # the representative open-pore current
        out       = signal - base + reference

    Because ``base`` is flat across each event (it must be -- that is the whole point
    of the event-aware estimator and its ``window_s >> dwell`` guard), subtracting it
    removes the drift *between* events without touching the excursion *within* an
    event: a blockade that was 40 pA below its local open pore is still 40 pA below
    ``reference`` afterwards. Adding ``reference`` back keeps the open-pore level and
    the absolute event depths where downstream detection/characterisation expect them,
    rather than collapsing everything onto zero.

    THE HAZARD IS INHERITED, NOT ESCAPED. If ``window_s`` is not comfortably larger
    than the longest event, ``base`` dives into the events and this SUBTRACTS them --
    depths come out shallow and short events vanish. `estimate_baseline` raises a
    `BaselineWarning` in that regime (window_s < 20x min_dwell_s); heed it. The default
    ``window_s`` (`DEFAULT_DETREND_WINDOW_S`) is wide for exactly this reason.

    THE FLIP SIDE OF THAT HAZARD IS MEASURED HERE, NOT ASSUMED AWAY. Wander FASTER
    than the window is not merely left in place: for wander periods between roughly
    half the window and the window, the block median is phase-INVERTED against the
    wander it sampled (the sign flip of a moving average's first sinc side lobe), so
    the subtraction ADDS wander -- measured, 80 pA of 3 Hz wander came out as a
    110.6 pA peak (+38%) under the default 0.5 s window. Nothing downstream can see
    that from the data, so this function compares the block-median wander of its own
    output against its input (`_block_median_wander`, two O(n) passes) and raises an
    `AmplifiedWanderWarning` when the output is worse by more than
    `DETREND_AMPLIFIED_PEAK_RATIO`. Silent amplification is not an option: the
    warning reaches ``res.warnings`` through the same `collect_baseline_warnings`
    sink the no-op detrend already reports through.

    Args:
        signal: Current trace (1-D, pA). ``float32`` in is fine.
        fs: Sample rate in Hz. Required for every method except 'constant'.
        method: Estimator for the open-pore envelope -- 'block_median' (default),
            'percentile', 'asls', or 'constant'. See `estimate_baseline`. 'constant'
            subtracts a flat median and adds it straight back, i.e. a no-op; it exists
            only so the panel's method list maps 1:1.
        window_s: Smoothing window in seconds. **Must be >> the longest dwell.** The
            drift-removal default is far wider than the I0-tracking default.
        polarity: ``dev = polarity*(current-baseline)`` sign factor; ``None`` measures
            it. Only matters for the one-sided 'percentile'/'asls' methods.
        **method_kwargs: Forwarded verbatim to `estimate_baseline` (``two_pass``,
            ``min_dwell_s``, ``percentile``, ``asls_*`` ...).

    Returns:
        ``float32`` array, ``len == signal.size``: the drift-flattened trace, still
        centred on the open-pore level.

    Raises:
        ValueError: propagated from `estimate_baseline` (empty/non-finite trace,
            missing/invalid rate, non-positive or over-long ``window_s``, unknown
            method). A silently wrong detrend is worse than a crash.
    """
    base = estimate_baseline(
        signal, fs, method=method, window_s=window_s, polarity=polarity, **method_kwargs
    )
    reference = float(np.median(base))  # keep the representative open-pore level
    out = np.asarray(signal, dtype=np.float64) - base + reference

    # SELF-JUDGEMENT: is the output's wander worse than the input's? 'constant' is a
    # documented no-op (out == signal to the bit) with nothing to measure, and a rate
    # of None never reaches here (estimate_baseline refuses it first for every other
    # method). The peak trips the warning; the RMS must corroborate the direction so
    # one event-owned block -- identical on both sides of the subtraction by
    # construction of the shared grid -- cannot fake an amplification on its own.
    if method != "constant":
        got_in = _block_median_wander(signal, fs, window_s)
        got_out = _block_median_wander(out, fs, window_s)
        if got_in is not None and got_out is not None:
            rms_in, peak_in = got_in
            rms_out, peak_out = got_out
            if (np.isfinite(peak_in) and peak_in > 0.0
                    and peak_out > DETREND_AMPLIFIED_PEAK_RATIO * peak_in
                    and rms_out > rms_in):
                _warn_baseline(
                    f"the detrend AMPLIFIED the baseline wander it exists to remove: "
                    f"peak wander {peak_in:.1f} pA in -> {peak_out:.1f} pA out "
                    f"({100.0 * (peak_out / peak_in - 1.0):+.0f}%), RMS "
                    f"{rms_in:.1f} -> {rms_out:.1f} pA, measured on "
                    f"{window_s / _WANDER_BLOCKS_PER_WINDOW:g} s block medians. "
                    f"Wander with a period between about half the {window_s:g} s "
                    "window and the window itself is phase-inverted by the block "
                    "median, so subtracting the estimate ADDS it. A detrend cannot "
                    "remove wander this fast: strip it with the high-pass stage "
                    "instead (a detrend window short enough to follow it would eat "
                    "the events), or disable the detrend.",
                    category=AmplifiedWanderWarning,
                    stacklevel=3,
                )
    return np.ascontiguousarray(out, dtype=np.float32)


def config_intrusion(
    signal: np.ndarray,
    fs: float,
    *,
    config: BaselineConfig,
    polarity: int,
    state_mask: np.ndarray,
) -> dict[str, float] | None:
    """``baseline_intrusion``'s OWN number, for a window nobody has run yet.

    WHY THIS EXISTS
    ===============
    `baseline_health` measures how far the tracked baseline sits inside the states and
    condemns the run above `BASELINE_INTRUSION_MAX`. It can only do that for a
    configuration that has ALREADY been estimated, and `baseline_probe` has to CHOOSE
    a configuration before one exists. This is the same measurement, callable ahead of
    the choice, so a proposer can rank candidate windows on the statistic its own
    health block will report rather than on the shortest window a bound tolerates.

    IT IS THE SAME MEASUREMENT, NOT A SECOND ESTIMATE OF IT. Every input is spelled
    exactly as `baseline_health` spells it:

    * the estimate comes from `characterisation_baseline` (the door), uncached and
      without its own health block;
    * ``noise`` is the first line of `_qc_block` -- ``1.4826 x MAD`` of the
      float32 detrended, optionally despiked, CANONICAL trace -- and feeds
      ``min_depth_pA = INTRUSION_MIN_DEPTH_SIGMA * noise``, the filter without which
      the reading is a ratio of two depths that are mostly noise;
    * ``dilate_ms`` is the config's own, so `state_runs` drops the same noise runs;
    * ``state_mask`` is the caller's, and must be the WIDE pass (`reference_states`)
      -- see below.

    Verified against the health block on the owner's three apo recordings at every
    grid window between the proposal and the reference window (nine configurations,
    ``0.5/0.1`` through ``3.0/1.0``): the fraction returned here is **bit-identical**
    to ``BaselineHealth.intrusion_fraction`` in all nine, and so is ``n_states``. All
    FOUR returned scalars are asserted equal (not ``approx``) in
    ``tests/analysis/test_rk_intrusion_ranking_core.py`` -- ``approx`` there passed
    over a float32/float64 mismatch that moved ``depth_pA`` in its last bits while
    leaving the other three exact.

    ONE MASK FOR EVERY CANDIDATE, AND WHY THE CALLER SUPPLIES IT
    -----------------------------------------------------------
    The mask is what says WHERE the states are, and `reference_states` measures it at
    `reference_rough_s` -- a window that does not depend on the candidate at all. So a
    caller ranking N windows takes ONE wide pass (0.16-0.68 s on these recordings) and
    reuses it, which is both the cheap thing and the only correct one: a mask
    re-measured per candidate would make the comparison between two candidates a
    comparison of two masks. It is passed in rather than measured here for exactly
    that reason -- a function that took its own would silently invite N of them.

    Args:
        signal: The trace, WHOLE CHANNEL, in the caller's own frame (not canonical).
        fs: Sample rate, Hz.
        config: The candidate. Its ``rough_s`` / ``base_s`` / ``decim`` /
            ``despike_size`` are all used.
        polarity: ``{-1, +1}``. PASS IT -- the same reason the door says so.
        state_mask: The wide-pass mask from `reference_states`, full length.

    Returns:
        `tracked_intrusion`'s dict (``fraction`` / ``intrusion_pA`` / ``depth_pA`` /
        ``n_states``), or ``None`` when no state in the mask has a measurable depth --
        which a caller must read as NOT MEASURED, never as "no intrusion".
    """
    orig = np.asarray(signal)
    fs = float(fs)
    pol = int(polarity)
    check_cancelled()
    res = characterisation_baseline(
        orig, fs, config=config, polarity=pol, health=False, cache=False)
    b = np.asarray(res.baseline, dtype=np.float64)
    # FLOAT64, THE WAY THE DOOR HANDS IT ON, AND THIS IS NOT HOUSEKEEPING.
    # `characterisation_baseline` upcasts to float64 before it calls
    # `estimate_baseline_result`, which passes THAT array to `baseline_health`, which
    # passes it to `tracked_intrusion`. Measuring here on the caller's float32 instead
    # reproduced ``fraction``, ``n_states`` and ``intrusion_pA`` exactly -- the
    # baseline is float64 either way -- and moved ``depth_pA`` in its last bits
    # (109.1936583518982 against 109.19366216659546 on a synthetic), because a median
    # of a float32 slice is not a median of the same values widened. A ranking that
    # agrees with the health block on three numbers out of four is a ranking that will
    # disagree on the fourth exactly when it matters.
    x = orig if orig.dtype == np.float64 else orig.astype(np.float64)
    # `baseline_health`'s own chain, in its own order and its own dtypes: detrend ->
    # float32 -> [despike] -> canonical. The float32 cast is not cosmetic there (see
    # `_qc_block`) and copying the order is what makes ``noise`` the same number.
    detr = (x - b).astype(np.float32)
    if config.despike_size:
        detr = despike_median(detr, size=config.despike_size)
    detr_canon = detr if pol == -1 else -detr
    noise = float(1.4826 * np.median(np.abs(detr_canon - np.median(detr_canon))))
    check_cancelled()
    # ``worst_min_dwell_s = base_s``: the worst-state reading is only taken over the
    # states this candidate's own tracking window could dwell inside. A shorter
    # state's per-state ratio is noise (see `tracked_intrusion`'s docstring for the
    # ferritin_data2 measurement), and the mechanism the reading exists to catch --
    # the rolling median entering a state -- needs a state comparable to the window.
    return tracked_intrusion(
        x if pol == -1 else -x, b if pol == -1 else -b, state_mask, fs,
        min_depth_pA=INTRUSION_MIN_DEPTH_SIGMA * noise,
        dilate_ms=config.dilate_ms,
        worst_min_dwell_s=float(config.base_s),
    )


def _spans(events, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) from an EventSet, a DataFrame, or a (starts, ends) pair."""
    df = getattr(events, "df", events)
    if hasattr(df, "columns"):
        for a, b in (("start_idx", "end_idx"),):
            if a in df.columns and b in df.columns:
                return (
                    np.asarray(df[a], dtype=np.int64),
                    np.asarray(df[b], dtype=np.int64),
                )
        raise ValueError(
            f"event frame needs 'start_idx'/'end_idx' columns, got {list(df.columns)}"
        )
    try:
        s, e = events
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "events must be an EventSet, a DataFrame with start_idx/end_idx, "
            "or a (starts, ends) pair"
        ) from exc
    s = np.asarray(s, dtype=np.int64)
    e = np.asarray(e, dtype=np.int64)
    if s.shape != e.shape:
        raise ValueError(f"starts/ends length mismatch: {s.shape} vs {e.shape}")
    if s.size and (s.min() < 0 or e.max() > n or np.any(e <= s)):
        raise ValueError("event spans are out of range or non-positive length")
    return s, e


def baseline_intrusion(
    current: np.ndarray,
    baseline: np.ndarray,
    events,
    fs: float,
    polarity: int,
    *,
    flank_s: float = 5e-3,
    pad_s: float = 2e-4,
) -> dict:
    """Measure how far the baseline dipped *into* the events. The safety check.

    For each event the baseline inside it is compared against a reference taken
    from flanking windows on either side (which straddle the event, so drift
    cancels to first order). The intrusion is the part of the excursion the
    baseline followed, expressed as a fraction of the event's depth:

        0.0 -> the baseline ignored the event entirely (what you want)
        0.5 -> the baseline ate half the depth; every depth_pA is ~50% low
        1.0 -> the baseline sat at the blocked level; the event is invisible

    Args:
        current: source current trace (1-D, pA).
        baseline: full-length baseline, as returned by `estimate_baseline`.
        events: EventSet, DataFrame with start_idx/end_idx, or (starts, ends).
        fs: sample rate in Hz (for the flank/pad widths).
        polarity: sign convention factor, in ``{-1, +1}``.
        flank_s: half-width of each reference window outside the event. Should be
            comparable to the baseline window_s. Neighbouring events falling in a
            flank bias the reference toward them, i.e. make this optimistic.
        pad_s: gap left between the event and its flanks, to skip filter
            shoulders.

    Returns:
        dict with 'intrusion_score' (median fraction over events, the headline
        number), 'p90', 'mean', 'per_event' (float array, NaN where undefined),
        'median_intrusion_pA', 'n_events', 'severe' (bool, score > 0.1), and
        'verdict' (a sentence for the user). Score is NaN with no events --
        absence of detections is not evidence of a clean baseline.
    """
    x = np.asarray(current, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if b.shape != x.shape:
        raise ValueError(f"baseline shape {b.shape} != current shape {x.shape}")
    starts, ends = _spans(events, x.size)
    pol = int(polarity)
    n = x.size

    if starts.size == 0:
        return {
            "intrusion_score": float("nan"),
            "p90": float("nan"),
            "mean": float("nan"),
            "per_event": np.empty(0),
            "median_intrusion_pA": float("nan"),
            "n_events": 0,
            "severe": False,
            "verdict": "no events supplied; baseline intrusion is undefined.",
        }

    pad = max(1, int(round(pad_s * fs)))
    flank = max(2, int(round(flank_s * fs)))

    frac = np.full(starts.size, np.nan)
    intr = np.full(starts.size, np.nan)
    # One short Python loop over EVENTS (hundreds), not over samples. The spans
    # are ragged, so this is not a vectorisation failure.
    for i, (s, e) in enumerate(zip(starts, ends)):
        ls, le = max(0, s - pad - flank), max(0, s - pad)
        rs, re = min(n, e + pad), min(n, e + pad + flank)
        ref_parts = [b[ls:le], b[rs:re]]
        ref_parts = [p for p in ref_parts if p.size]
        if not ref_parts or e <= s:
            continue
        ref = float(np.mean([np.median(p) for p in ref_parts]))
        b_in = float(np.mean(b[s:e]))
        depth = pol * (float(np.mean(x[s:e])) - ref)
        intr[i] = pol * (b_in - ref)
        if depth > 0:
            frac[i] = intr[i] / depth

    ok = np.isfinite(frac)
    score = float(np.median(frac[ok])) if ok.any() else float("nan")
    p90 = float(np.percentile(frac[ok], 90)) if ok.any() else float("nan")
    severe = bool(np.isfinite(score) and score > 0.1)
    if not np.isfinite(score):
        verdict = "intrusion undefined: no event had a measurable depth."
    elif score > 0.3:
        verdict = (
            f"SEVERE: the baseline follows {score:.0%} of each event. Depths are that "
            "much too shallow and short events are being erased. Widen window_s or use "
            "two_pass=True."
        )
    elif severe:
        verdict = (
            f"the baseline follows {score:.0%} of each event; depths are biased shallow "
            "by about that much. Widen window_s."
        )
    else:
        verdict = f"baseline is event-immune ({score:.1%} of depth followed)."

    return {
        "intrusion_score": score,
        "p90": p90,
        "mean": float(np.mean(frac[ok])) if ok.any() else float("nan"),
        "per_event": frac,
        "median_intrusion_pA": float(np.median(intr[np.isfinite(intr)]))
        if np.isfinite(intr).any()
        else float("nan"),
        "n_events": int(starts.size),
        "severe": severe,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# THE PICTURE -- the estimator's own answer, in a form a plot can draw
# ---------------------------------------------------------------------------
# Two display kernels, and they are DISPLAY kernels: nothing here is stamped into
# provenance, measured against, or exported. They exist because the estimator that
# decides every ``depth_pA`` had, until now, no on-screen representation at all --
# the user confirmed two windows in seconds and never saw what they produced.
#
# THE ESTIMATOR THEY DRAW IS THE OPEN-PORE ONE (`characterisation_baseline`), not
# the detrend stage's `estimated_wander`. The two are different measurements that
# share a word: the first sets the depth of every row, the second flattens the trace
# the detector sees and touches no column. A user who ticks "show the baseline" to
# check "is my baseline right?" is asking about the first.


#: Knots a baseline track is decimated to before it reaches a plot.
#:
#: The trace itself must be PEAK-decimated (a one-sample blockade has to survive,
#: which is why `ChannelPane.make_curve` passes ``method="peak"`` explicitly); the
#: baseline must not be, because it carries no such spike -- it is a smoothed
#: envelope, so straight lines between dense enough samples of it are faithful and a
#: min/max pair per bin would only draw a band where there is a line.
#:
#: DENSE ENOUGH IS A MEASUREMENT, NOT A GUESS, and the first value tried was wrong.
#: The error of a uniform decimation is bounded by the baseline's own slope times the
#: knot spacing, and this baseline is not as smooth as "smoothed envelope" suggests:
#: on the owner's 15 s -1000 mV apo recording (749,651 samples at 50 kHz, ch2, the
#: app-default 0.5/0.1 s windows) it moves up to 0.032 pA per SAMPLE where the event
#: mask hands the short pass a step. Measured worst-case error against the
#: full-length baseline it stands for:
#:
#:     2 000 knots   1.45 pA        10 000 knots  0.68 pA
#:     4 000 knots   1.07 pA        20 000 knots  0.29 pA (99.9th pct 0.11 pA)
#:
#: against a pane that fits that recording's 233 pA span into ~500 px, i.e. 0.47 pA
#: per device pixel: 4 000 knots is a 2.3 px lie about where the baseline is, and
#: 20 000 is 0.6 px. Twenty thousand points is also well inside what one
#: ``PlotDataItem`` draws without help. Pass ``max_points=0`` for the exact
#: full-length track when a picture has to be right to the sample.
BASELINE_TRACK_MAX_POINTS = 20_000


def decimate_track(
    baseline: np.ndarray, max_points: int = BASELINE_TRACK_MAX_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """``(idx, value)`` knots of a full-length baseline, at most ``max_points`` of them.

    ``idx`` is int64 SAMPLE INDICES -- the vehicle
    :class:`~nanoscope.core.analysis.OverlayHint` carries, so the view maps them onto
    the same x axis the curve beside them is drawn against and an aligned layer's
    track cannot land ``time_offset_s`` away from the trace it annotates.

    The first and last samples are always knots (``np.linspace`` endpoints), so the
    drawn track spans exactly the recording rather than stopping short of it.
    A baseline already shorter than ``max_points`` is returned whole.
    """
    b = np.asarray(baseline, dtype=np.float64)
    n = int(b.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    m = int(max_points)
    if m <= 0 or n <= m:
        return np.arange(n, dtype=np.int64), np.ascontiguousarray(b)
    idx = np.unique(np.rint(np.linspace(0.0, n - 1.0, m)).astype(np.int64))
    return idx, np.ascontiguousarray(b[idx])


def baseline_track(
    signal: np.ndarray,
    fs: float,
    *,
    config: "BaselineConfig | None" = None,
    polarity: int | None = None,
    param_source: str = "default",
    max_points: int = BASELINE_TRACK_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """The characterisation baseline as decimated ``(idx, value)`` knots to draw.

    "Here is the open-pore level every depth in your table is measured against."
    Same door as the detection (`characterisation_baseline`, hence the same memo and
    the same clamping on a short segment), so the line on screen is the line the
    numbers came from and cannot drift from it.

    PASS THE POLARITY. It is the same argument, for the same reason, as the door's:
    `derive_polarity` reads +1 on the owner's own drifting raw traces, which points
    the one-sided mask the wrong way and moves the baseline by up to 312 pA -- and a
    drawn baseline that is wrong by 312 pA is worse than no baseline drawn, because
    it looks like an answer. The detection form's measured polarity is the source.

    ``health=False``: the block is roughly the cost again and nothing on the trace
    reads it -- the panel that owns the checkbox already shows the health block from
    the probe. Measured cost of the whole call on the owner's 15 s -1000 mV apo
    recording (749,651 samples at 50 kHz, ch2, cold cache): 70 ms, of which the
    decimation is under 1 ms. That is why this can be drawn on a tick rather than
    behind a "compute" button -- and why the tick does not need a worker thread,
    while the probe beside it (seconds) does.

    Returns:
        ``(idx int64, values float64)``, ready for
        ``OverlayHint(kind=OverlayKind.HLINE, starts=idx, values=values)``.
    """
    x = np.asarray(signal)
    if x.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    result = characterisation_baseline(
        x, float(fs), config=config, polarity=polarity,
        param_source=param_source, health=False,
    )
    return decimate_track(result.baseline, max_points)


def baseline_subtracted(signal: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """``signal - baseline`` as float32: the residual, centred on zero, FOR DISPLAY.

    THE ONE-LINE DIFFERENCE FROM `remove_baseline_wander`, and it is deliberate.
    That function subtracts the wander and then ADDS BACK ``median(base)``, because
    its output is fed to detection and characterisation, which measure absolute
    currents -- re-zeroing it there would move every ``depth_pA``. This one is drawn,
    never measured, so it keeps the zero: a trace at zero is what lets a -566 pA
    recording and a -1000 pA one be read on ONE y axis, which is the entire reason a
    subtracted view is worth having. Depths are unchanged by construction (a constant
    offset per sample cancels in ``open - blocked``); only the axis moves.

    Sign convention: the OPEN PORE sits at 0 and a blockade departs from it in the
    trace's own direction -- at negative bias a blockade still reads positive-going
    here, exactly as it does on the raw trace. Nothing is rectified; ``polarity``
    belongs to the estimator that produced ``baseline``, not to this subtraction.

    Raises:
        ValueError: if the two lengths differ -- a residual built from a baseline of
            another recording is a picture of nothing, and NumPy would broadcast a
            length-1 baseline silently rather than say so.
    """
    x = np.asarray(signal, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if x.shape != b.shape:
        raise ValueError(
            f"baseline shape {b.shape} != signal shape {x.shape}; a residual can only "
            f"be formed against the baseline of the same trace")
    return np.ascontiguousarray(x - b, dtype=np.float32)
