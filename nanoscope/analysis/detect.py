"""Event detection: find translocations in a current trace.

Two ideas carry this module.

**Measure the polarity, never assume it.** Most tools hardcode "blockades go
down" and silently return zero events on enhancement data. Enhancement is
physically real: below roughly 400 mM salt, counterion condensation around a
highly charged analyte can add more mobile charge than the analyte's volume
excludes, so |I| *rises* during a translocation. `detect_polarity` counts
coherent excursion runs in each direction and lets the data decide. A run
vote inside the ambiguous band (confidence below `_RUN_VOTE_DECISIVE`) is
NOT taken as a decision on its own: a symmetric smoothing filter's edge
lobes contribute ~2 brief counter-runs per square event and can outvote the
events themselves in run count, so the ambiguous band is arbitrated by where
the coherent sample mass lies -- the statistic those lobes cannot flip.

**Two thresholds, not one.** A single threshold chatters while the signal
loiters at the crossing, shattering one real event into a burst of fragments:
counts inflate, dwell times collapse. `detect_events` uses a Schmitt trigger --
enter at `k_in`, stay in until `dev` falls below `k_out` (`k_out < k_in`).

Sign convention (from `events.py`, obeyed everywhere)::

    dev = polarity * (current - baseline)   # positive inside an event

so `depth_pA` and `frac_depth` are always positive magnitudes and the direction
lives in `polarity`.

A caution on what `polarity` means here. `polarity` is the factor that makes
`dev` positive inside an event, i.e. it names the direction of the excursion in
the **signed** current: -1 = current moves below baseline, +1 = above. That is
not always the same thing as "blockade vs enhancement", because nanopore
recordings at negative bias have a negative open-pore current: with I0 = -1000
pA, an event that reduces |I| to 900 pA moves the signed current *up*, so
polarity is +1 even though it is an ordinary blockade. The physical reading
depends on the sign of the baseline, and `detect_polarity` reports it separately
under the `physical` key of its evidence dict rather than conflating the two.

Signal-first (PORT_PLAN §3.5): the engine takes an explicit ``(signal, fs,
...)`` rather than a Recording. ``polarity=None`` measures the polarity from the
data (verdict ``'measured'``); a provided ``polarity`` is used verbatim. There
is no stored polarity to cross-check against, so the SOURCE's contradiction
raise is gone -- passing the wrong polarity simply detects ~nothing, as physics
demands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from .baseline import BaselineConfig, characterisation_baseline, require_polarity
from .cancel import CancelFlag, check_cancelled, propagate_cancel
from .characterize import characterize
from .events import empty_df, n_clipped_boundary, span_provenance
from .filters import lowpass, measure_rise_time, noise_rms
from .measurement_quality import build_measurement_quality

try:  # pragma: no cover - packaging detail
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("nanoscope")
except Exception:  # pragma: no cover
    _VERSION = "unknown"

# A lone sample beyond 5 sigma is a spike, not a translocation. Requiring a run
# of >= 2 samples is the cheapest possible coherence test.
_MIN_RUN_SAMPLES = 2

# Below this confidence a vote sits in the AMBIGUOUS BAND: the winner holds less
# than three times the loser's count. The module's own tests draw the bands
# empirically -- a real measurement lands above 0.9
# (test_polarity_measured_both_ways), noise-only ambiguity below 0.5
# (test_polarity_undetermined_on_flat_noise) -- and a symmetric smoothing
# filter's edge lobes land INSIDE the band by construction: one brief
# opposite-sign lobe flanks each edge of a square event, so N real events yield
# ~2N counter-runs and a run-vote confidence of ~1/3. Used twice in
# `detect_polarity`: a run vote below it is not a decision on its own, and the
# coherent-sample majority must clear the same bar to arbitrate.
_RUN_VOTE_DECISIVE = 0.5

__all__ = [
    "detect_polarity",
    "detect_events",
    "gap_window_samples",
    "group_starts_by_gap",
    # The occupancy floor lives here, next to the gap rule it brakes, because BOTH
    # merge paths need it and the import only runs one way: `analysis.merge` imports
    # this module, so nothing here may import `analysis.merge`. They are re-exported
    # from there, so `from nanoscope.analysis.merge import MIN_MERGE_OCCUPANCY` -- the
    # spelling the GUI and the tests already use -- still resolves.
    "MIN_MERGE_OCCUPANCY",
    "MARGINAL_MERGE_OCCUPANCY",
    "LOW_OCCUPANCY_TOKEN",
    "span_occupancy",
    "split_low_occupancy_groups",
    # The conventional detector kernels. Each returns the SAME ``(starts, ends)``
    # interval form `detect_events` hands to `characterize`, measured against the
    # SAME baseline/polarity/noise machinery -- see the section header below.
    "NOISE_METHODS",
    "noise_level",
    "DetectionBasis",
    "detection_basis",
    "detect_fixed_threshold",
    "detect_peaks",
    "detect_derivative",
    "tkeo_energy",
    "detect_tkeo",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_baseline(baseline: np.ndarray | float, n: int) -> np.ndarray:
    """Broadcast a scalar or validate an array baseline to length n (float64)."""
    b = np.asarray(baseline, dtype=np.float64)
    if b.ndim == 0:
        return np.full(n, float(b))
    if b.ndim != 1:
        raise ValueError(f"baseline must be scalar or 1-D, got shape {b.shape}")
    if b.size != n:
        raise ValueError(f"baseline length {b.size} != signal length {n}")
    return b


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open [start, end) index pairs of every True run in a 1-D bool mask.

    Vectorised: one padded diff, no Python loop over samples.
    """
    padded = np.empty(mask.size + 2, dtype=bool)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = mask
    d = np.diff(padded.view(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _run_maxima(
    dev_padded: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Max of `dev` within each [start, end) run.

    `dev_padded` must be `dev` with one extra trailing element, so that a run
    ending at the final sample still yields an in-bounds reduceat index.
    """
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)
    bounds = np.empty(2 * starts.size, dtype=np.intp)
    bounds[0::2] = starts
    bounds[1::2] = ends
    # reduceat over interleaved run/gap boundaries; even slots are the runs.
    return np.maximum.reduceat(dev_padded, bounds)[0::2]


def _hysteresis_runs(
    x_padded: np.ndarray, n: int, enter: float, exit_: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """THE Schmitt rule, once: runs past `exit_` whose peak also passed `enter`.

    Returns ``(starts, ends, n_candidates)`` -- the surviving half-open runs and how
    many runs crossed the exit threshold before the peak test, which is the term the
    audit trail's ``n_rejected_below_enter`` is computed from.

    Extracted from `detect_events` (numbers unchanged; the pre/post hash of the 20
    frozen columns on a 40-event synthetic is 0abf310277c34d73 either way) because
    `detect_fixed_threshold` needs the identical rule in absolute units. Two copies
    of a two-threshold trigger is exactly how a fixed-threshold detector and a sigma
    detector come to disagree about where an event ended.

    `x_padded` must be the signal with one extra trailing element (see `_run_maxima`);
    `n` is the true length. Comparisons are STRICT, so a sample exactly at a
    threshold is outside.
    """
    starts, ends = _runs(x_padded[:n] > exit_)
    n_candidates = int(starts.size)
    if n_candidates:
        keep = _run_maxima(x_padded, starts, ends) > enter
        starts, ends = starts[keep], ends[keep]
    return starts, ends, n_candidates


def _count_coherent_runs(mask: np.ndarray) -> tuple[int, int]:
    """(number of runs >= _MIN_RUN_SAMPLES, total samples in those runs)."""
    s, e = _runs(mask)
    lengths = e - s
    keep = lengths >= _MIN_RUN_SAMPLES
    return int(np.count_nonzero(keep)), int(lengths[keep].sum())


def _check_finite(x: np.ndarray, what: str) -> None:
    if not np.isfinite(x).all():
        n_bad = int((~np.isfinite(x)).sum())
        raise ValueError(
            f"{what} contains {n_bad} non-finite sample(s) (NaN/inf). "
            "Detection thresholds are undefined against NaN; clean or clip the "
            "trace first."
        )


# ---------------------------------------------------------------------------
# polarity
# ---------------------------------------------------------------------------
def detect_polarity(
    current: np.ndarray,
    baseline: np.ndarray | float,
    *,
    k: float = 5.0,
) -> tuple[int, dict[str, Any]]:
    """Decide which way events point by counting coherent excursions each way.

    Args:
        current: Current trace in pA (1-D).
        baseline: Local open-pore current in pA, scalar or same length as
            `current`.
        k: Threshold in units of the MAD-based noise RMS of the residual.

    Returns:
        `(polarity, evidence)`. `polarity` is the factor that makes
        `dev = polarity * (current - baseline)` positive inside events: -1 when
        excursions run below the baseline, +1 when they run above it.

        `evidence` carries at least `n_down`, `n_up`, `ratio` (n_down/n_up),
        `confidence` (0..1) and `reported_vote` -- and those four are ONE
        measurement, always: `confidence` is `|n_down-n_up|/(n_down+n_up)` of
        the very counts reported beside it, and `reported_vote` names what
        those counts count ('coherent_runs' -- runs of >= 2 samples beyond
        +-k*noise -- or 'coherent_samples', the samples in them). See Gotcha 3
        for when the second one is reported; the raw run counts are never lost
        (`n_down_runs`/`n_up_runs`).

    Gotcha 1: `polarity` describes the signed current, not the physics. At
    negative bias the open-pore current is negative, so an ordinary blockade
    (|I| falling) is an *upward* excursion and yields polarity=+1. The physical
    reading, inferred from the sign of the baseline, is reported separately as
    `evidence['physical']` ('blockade'/'enhancement'/'unknown').

    Gotcha 2: when neither direction dominates -- a flat trace, or a trace with
    no events -- there is nothing to measure. This returns -1 with
    `confidence == 0.0` and `verdict == 'undetermined'`. Check the confidence;
    do not read the -1 as a measurement.

    Gotcha 3: a run vote inside the ambiguous band (confidence below
    `_RUN_VOTE_DECISIVE`) is arbitrated by the coherent SAMPLE mass rather than
    accepted as-is, because run counting has one known failure mode: a
    symmetric smoothing filter (Savitzky-Golay, running mean) leaves a brief
    opposite-sign lobe flanking each edge of a square event, so N real events
    contribute ~2N counter-runs -- enough to outvote the events themselves
    while every lobe is short and shallow. When the arbitration ran, the
    evidence additionally carries `decided_by` ('runs'/'coherent_samples'),
    `run_confidence`, `sample_confidence`, `n_down_runs`/`n_up_runs`,
    `reported_vote` and (when the weak run vote stood because the sample mass
    agreed with it) `corroborated_by_samples`; a decisive run vote leaves the
    evidence dict exactly as it always was.

    Gotcha 4, and it is what `reported_vote` exists for: whenever the sample
    mass turned out to be the decisive statistic, the reported `n_down`/`n_up`
    /`ratio`/`confidence` are ITS numbers, not the run vote's. They have to be,
    because these four are rendered as one sentence -- the Detection tab prints
    "physical (direction): n_down down / n_up up (confidence C, verdict)" -- and
    a confidence lifted from one statistic next to counts from another is a line
    no reader can check or reconcile: on the owner's -600 mV recording at a
    20 ms Savitzky-Golay window it read "enhancement (down): 247 down / 402 up
    (confidence 0.73)", where the counts say UP won 402:247, the direction says
    down, and 0.73 is neither |247-402|/649 = 0.24 nor anything else derivable
    from the line. The run counts are still on the record under
    `n_down_runs`/`n_up_runs` -- `detect_events`' arbitration warning quotes
    them, in as many words, as the run vote that was overruled.
    """
    cur = np.asarray(current, dtype=np.float64)
    if cur.ndim != 1:
        raise ValueError(f"current must be 1-D, got shape {cur.shape}")
    if cur.size == 0:
        raise ValueError("current is empty")
    if not (k > 0):
        raise ValueError(f"k must be positive, got {k}")
    _check_finite(cur, "current")

    base = _as_baseline(baseline, cur.size)
    _check_finite(base, "baseline")

    resid = cur - base
    noise = float(noise_rms(resid))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError(
            f"noise RMS of the baseline-subtracted trace is {noise!r}; a trace "
            "with no noise has no threshold scale. Is the signal constant, or "
            "already event-free?"
        )

    thr = k * noise
    n_down_runs, samp_down = _count_coherent_runs(resid < -thr)
    n_up_runs, samp_up = _count_coherent_runs(resid > thr)
    # The REPORTED counts. They start as the run counts and stay there unless the
    # arbitration below finds that the run vote was not the statistic that
    # decided -- see Gotcha 4: counts and confidence are reported as one vote or
    # the sentence built from them cannot be checked by the person reading it.
    n_down, n_up = n_down_runs, n_up_runs
    reported_vote = "coherent_runs"

    total = n_down + n_up
    if total == 0:
        polarity, confidence, verdict = -1, 0.0, "undetermined"
    elif n_down == n_up:
        polarity, confidence, verdict = -1, 0.0, "tie"
    else:
        polarity = -1 if n_down > n_up else 1
        confidence = abs(n_down - n_up) / total
        verdict = "measured"

    # -- arbitration: an ambiguous run vote is not a decision -----------------
    # The run COUNT is the fragile statistic in exactly one known regime: a
    # symmetric smoothing filter's edge lobes (one brief opposite-sign lobe
    # flanking each edge of a square event -- the response plot's own note)
    # outvote the events that produced them, at confidence ~1/3. The statistic
    # those lobes cannot flip is the coherent SAMPLE mass, already counted
    # above: [MEASURED] on the 5-blockade / savgol-at-defaults regression the
    # run vote was 5 down vs 10 up while the sample mass was 1734 down vs 177
    # up. So inside the ambiguous band the sample mass arbitrates -- and only
    # there: a decisive run vote is used verbatim and its evidence dict is
    # unchanged, byte for byte, which is also what keeps this guard clear of
    # the opposite failure mode (one long clog outvoting many short events by
    # sheer sample count) -- there the run vote is decisive and stands.
    arbitration: dict[str, Any] = {}
    if total > 0 and confidence < _RUN_VOTE_DECISIVE:
        run_confidence = confidence
        samp_total = samp_down + samp_up  # >= 2*total: every run has >= 2 samples
        samp_confidence = abs(samp_down - samp_up) / samp_total
        samp_polarity = -1 if samp_down > samp_up else 1
        decided_by = "runs"
        if samp_confidence >= _RUN_VOTE_DECISIVE:
            if samp_polarity != polarity or verdict == "tie":
                # The sample mass decisively contradicts the run vote (or breaks
                # a tie): direction and verdict come from the statistic that
                # actually decided. `detect_events` turns this stamp into a
                # polarity warning naming the likely cause, so the overruled run
                # vote is never silent.
                polarity, verdict = samp_polarity, "measured"
                decided_by = "coherent_samples"
            else:
                # Same direction: the weak run vote stands, corroborated. The
                # stamp lets `detect_events` skip the low-confidence warning --
                # two independent statistics agreeing is not a weak measurement.
                arbitration["corroborated_by_samples"] = True
            # ONE VOTE PER SENTENCE. Whichever of the two branches above ran, the
            # sample mass is the statistic that settled this measurement -- it
            # either overruled the run vote or is the reason the weak one was
            # allowed to stand -- so it is the vote that gets REPORTED: its
            # counts, its ratio, its confidence, together. Reporting its
            # confidence beside the run counts is what made the readout
            # unreadable (Gotcha 4), and reporting the run confidence instead
            # would understate a measurement the app itself then declines to warn
            # about (a corroborated vote read "25 down / 22 up (confidence 0.06)"
            # for a direction two independent statistics agreed on at 0.88).
            # `run_confidence` and `n_*_runs` below keep the overruled vote on
            # the record; nothing is replaced silently.
            confidence = samp_confidence
            n_down, n_up = samp_down, samp_up
            reported_vote = "coherent_samples"
        arbitration["decided_by"] = decided_by
        arbitration["run_confidence"] = run_confidence
        arbitration["sample_confidence"] = samp_confidence
        arbitration["n_down_runs"] = n_down_runs
        arbitration["n_up_runs"] = n_up_runs
        # Named, not inferred: a reader of this dict must never have to work out
        # which statistic `confidence` belongs to from the presence of another key.
        arbitration["reported_vote"] = reported_vote

    # Of the REPORTED counts, so `ratio`, `confidence` and the two counts stay one
    # arithmetic: a ratio of the run counts under a sample-mass confidence would
    # reintroduce the same mismatch one field further down the dict.
    if n_up == 0:
        ratio = float("inf") if n_down > 0 else float("nan")
    else:
        ratio = n_down / n_up

    med_base = float(np.median(base))
    excursion_sign = -1 if polarity == -1 else 1
    if verdict in ("undetermined", "tie") or med_base == 0.0:
        physical = "unknown"
    else:
        # |I| falls when the current moves toward zero, i.e. when the excursion
        # opposes the sign of the open-pore current.
        physical = (
            "blockade" if excursion_sign * np.sign(med_base) < 0 else "enhancement"
        )

    evidence: dict[str, Any] = {
        "n_down": n_down,
        "n_up": n_up,
        "ratio": ratio,
        "confidence": confidence,
        "verdict": verdict,
        "polarity": polarity,
        "direction": "down" if polarity == -1 else "up",
        "physical": physical,
        "n_down_samples": samp_down,
        "n_up_samples": samp_up,
        "noise_rms_pA": noise,
        "median_baseline_pA": med_base,
        "k": float(k),
        "threshold_pA": float(thr),
        "min_run_samples": _MIN_RUN_SAMPLES,
        # Present ONLY when the ambiguous band was entered (see Gotcha 3): a
        # decisive run vote stamps nothing, so its evidence stays byte-identical
        # to what it always was.
        **arbitration,
    }
    return polarity, evidence


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def _exclusion_samples(
    exclude_regions: Sequence[tuple[float, float]] | None, n: int, fs: float
) -> list[tuple[int, int]]:
    """Validate and convert [(t0,t1),...] seconds -> clipped [i0,i1) samples."""
    if not exclude_regions:
        return []
    out: list[tuple[int, int]] = []
    for region in exclude_regions:
        t0, t1 = region
        if not (np.isfinite(t0) and np.isfinite(t1)):
            raise ValueError(f"exclude_regions entry {region!r} is not finite")
        if t1 <= t0:
            raise ValueError(
                f"exclude_regions entry {region!r} is empty or reversed (t1 <= t0)"
            )
        i0 = max(0, int(round(t0 * fs)))
        i1 = min(n, int(round(t1 * fs)))
        if i1 > i0:
            out.append((i0, i1))
    return _merge_intervals(out)


# ---------------------------------------------------------------------------
# the merge rule (SHARED with analysis.merge so detect-time and post-hoc merges
# group events by ONE definition, never two subtly-different ones)
# ---------------------------------------------------------------------------
def gap_window_samples(max_gap_s: float, fs: float) -> int:
    """The merge window in samples: ``round(max_gap_s * fs)``.

    The single conversion both the detect-time ``merge_gap_s`` and the post-hoc
    :func:`nanoscope.analysis.merge.merge_by_gap` go through, so a window given in
    seconds means the same number of samples on both paths.
    """
    return int(round(float(max_gap_s) * float(fs)))


def group_starts_by_gap(
    starts: np.ndarray, ends: np.ndarray, max_gap_samples: int
) -> np.ndarray:
    """Boolean mask over start-sorted event spans: ``True`` where a new (un-merged)
    group begins.

    An event joins the group being built -- and a run of them collapses to one event
    spanning ``[first.start_idx, max end_idx of the run]`` -- exactly when

        gap = next.start_idx - (running max end_idx of the group) <= max_gap_samples

    (transitively: a chain each within the window is one group). A gap **wider** than
    the window starts a fresh group. This is the ONE rule the detector's ``merge_gap_s``
    and ``analysis.merge`` both obey; do not re-derive it anywhere else.

    THE COMPARISON IS AGAINST THE GROUP, NOT THE PREDECESSOR. This used to be the
    vectorised ``gaps = s[1:] - e[:-1]``, i.e. each event against its IMMEDIATE
    predecessor's end -- which is only the same rule while the spans are also
    non-overlapping. A curated table need not be: the Event Inspector's mask resize and
    the hand-added-event path can both leave a long event with a short one nested
    inside it. Reproduced on starts ``[10000, 10250, 19990]`` / ends
    ``[20000, 10400, 20100]`` at a 50-sample window: the third event, which starts 10
    samples BEFORE the first one ends, was compared against the nested short event's
    end 10400, read a gap of 9590 and split off into its own group -- so a merge left
    two rows overlapping each other by 10 samples. ``np.maximum.accumulate(e)`` is not
    the fix: it never resets, so one long early event would bind every later group to
    itself for the rest of the recording. The running maximum has to RESET at each new
    group start, which is a sequential scan.
    """
    s = np.asarray(starts)
    e = np.asarray(ends)
    n = int(s.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    is_start = np.ones(n, dtype=bool)
    if n > 1:
        gap = int(max_gap_samples)
        # ``.tolist()`` once: the scan is O(n) either way, but indexing Python ints is
        # several times cheaper than indexing a numpy array element-by-element, and this
        # runs over every candidate run of a multi-million-sample detection.
        s_i = s.tolist()
        e_i = e.tolist()
        cur_end = e_i[0]
        for i in range(1, n):
            if s_i[i] - cur_end > gap:
                is_start[i] = True
                cur_end = e_i[i]            # new group: the running maximum resets here
            else:
                is_start[i] = False
                if e_i[i] > cur_end:
                    cur_end = e_i[i]
    return is_start


# ---------------------------------------------------------------------------
# the brake on the merge rule (also SHARED -- see the note in __all__ for why it
# lives in this module and not in analysis.merge)
# ---------------------------------------------------------------------------
#: The fraction of a merged span that its source events must actually cover for the
#: merge to be allowed.
#:
#: THE PHYSICS. A translocation is one continuous residence of one molecule in the pore.
#: A run of fragments is the SAME residence only if the fragments are what the span is
#: mostly made of -- the recrossings the merge exists to stitch over are brief dips back
#: through the exit threshold, so a genuinely shattered event is >90 % occupied (the
#: bench's own three-fragment fixtures sit at 0.92-0.96). A span that is mostly open pore
#: between its fragments is not one translocation being repaired, it is several
#: translocations being glued to the open pore between them, and every derived column
#: then lies in the same direction: dwell becomes the whole span, and depth -- a median
#: over that span -- is dragged to the baseline by the open-pore majority (40 clean
#: 0.4 ms / 300 pA events spaced 8 ms apart, merged at a 10 ms window, became ONE row of
#: dwell 312.40 ms and depth 0.70 pA, accepted, with a blank flag).
#:
#: 0.5 is deliberately permissive: it is the point where the open pore becomes the
#: MAJORITY of the row, which is the weakest claim that is still unarguable. It is not
#: tuned to a dataset, and it is a floor rather than a threshold on the number of
#: fragments or on the window, because occupancy is the quantity that stays meaningful
#: across sample rates, translocation speeds and merge windows.
MIN_MERGE_OCCUPANCY = 0.5

#: Above the floor but below this, the merge is KEPT and recorded. A span that is a
#: third open pore is defensible (slow, multi-level translocations really do look like
#: that) but it is also the shape a runaway merge takes on its way down, so it must be
#: auditable after the fact rather than indistinguishable from a clean stitch.
MARGINAL_MERGE_OCCUPANCY = 0.8

#: Key under ``provenance['merged']`` listing the kept-but-marginal merges (and, under
#: ``'refused'``, the ones that were declined). The token is stable: exports and any
#: later audit look it up by name.
LOW_OCCUPANCY_TOKEN = "low_occupancy_merge"


def span_occupancy(starts: np.ndarray, ends: np.ndarray) -> float:
    """Fraction of ``[min start, max end)`` that the given spans actually cover.

    ``1.0`` for a single event (or a degenerate zero-length span).  Coverage is the
    interval union: a curated table may contain overlapping or nested events, and their
    shared samples must be counted once rather than manufacturing occupancy.
    """
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    if s.size == 0:
        return 1.0
    span = int(e.max()) - int(s.min())
    if span <= 0:
        return 1.0
    intervals = [(int(a), int(b)) for a, b in zip(s, e) if b > a]
    covered = sum(b - a for a, b in _merge_intervals(intervals))
    return covered / float(span)


def split_low_occupancy_groups(
    starts: np.ndarray, ends: np.ndarray, is_group_start: np.ndarray
) -> tuple[np.ndarray, list[float], list[tuple[int, int, float]]]:
    """Split every proposed group that is mostly open pore back into singletons.

    Takes the group boundaries :func:`group_starts_by_gap` proposed and returns
    ``(is_group_start, refused_occupancies, marginal_spans)``: a copy of the mask with
    each refused group's members marked as their own group start, the occupancy of each
    refusal (for a status line or a provenance counter), and ``(start, end, occupancy)``
    for each kept-but-marginal merge (for the provenance note).

    The gap rule itself is NOT weakened by this: grouping still comes from the one
    shared :func:`group_starts_by_gap`, and occupancy is a second, physical question
    asked of the group that rule proposed. Like the gap rule, it is defined ONCE --
    ``detect_events``' ``merge_gap_s`` and :mod:`nanoscope.analysis.merge` both call
    this function, so the detect-time and post-hoc merges cannot drift apart.

    This runs BEFORE any re-characterization on purpose. Refusing afterwards would mean
    measuring spans only to throw them away -- the expensive half of a merge -- and,
    worse, it would mean the refusal decision was taken on a row that had already been
    measured over the wrong span.
    """
    out = np.asarray(is_group_start).copy()
    bounds = np.flatnonzero(out).tolist()
    edges = bounds + [int(out.size)]

    refused: list[float] = []
    marginal: list[tuple[int, int, float]] = []
    for lo, hi in zip(bounds, edges[1:]):
        if hi - lo < 2:
            continue                       # a singleton is not a merge; nothing to judge
        occupancy = span_occupancy(starts[lo:hi], ends[lo:hi])
        if occupancy < MIN_MERGE_OCCUPANCY:
            out[lo:hi] = True              # every member becomes its own group again
            refused.append(occupancy)
        elif occupancy < MARGINAL_MERGE_OCCUPANCY:
            marginal.append((int(starts[lo]), int(ends[lo:hi].max()), occupancy))
    return out, refused, marginal


def _merge_intervals(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of possibly overlapping [i0,i1) intervals; overlaps must not be
    double-counted when they are subtracted from the analysed duration."""
    if not iv:
        return []
    iv = sorted(iv)
    merged = [iv[0]]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _validate_detection_timing(
    fs: float, min_dwell_s: float, max_dwell_s: float | None,
    merge_gap_s: float | None,
) -> None:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}")
    if not np.isfinite(min_dwell_s) or min_dwell_s < 0:
        raise ValueError(f"min_dwell_s must be finite and >= 0, got {min_dwell_s!r}")
    if max_dwell_s is not None and (
        not np.isfinite(max_dwell_s) or max_dwell_s <= min_dwell_s
    ):
        raise ValueError(
            f"max_dwell_s ({max_dwell_s}) must be finite and exceed min_dwell_s ({min_dwell_s})"
        )
    if merge_gap_s is not None and (not np.isfinite(merge_gap_s) or merge_gap_s < 0):
        raise ValueError(f"merge_gap_s must be finite and >= 0, got {merge_gap_s!r}")


def _unmeasurable_noise_message(noise: float, label: str, *, supplied_baseline: bool) -> str:
    if supplied_baseline:
        return (
            f"{label} after subtracting the supplied open-pore reference/baseline is "
            f"{noise!r}; cannot resolve detection noise. Check the reference current, "
            "units and numeric precision: a reference far from this trace can erase "
            "its noise during subtraction."
        )
    return f"{label} is {noise!r}; cannot set thresholds on a trace with no measurable noise."


@propagate_cancel
def detect_events(
    signal: np.ndarray,
    fs: float,
    *,
    polarity: int | None = None,
    k_in: float = 5.0,
    k_out: float = 2.0,
    min_dwell_s: float = 2e-5,
    max_dwell_s: float | None = None,
    baseline: np.ndarray | float | None = None,
    baseline_config: BaselineConfig | None = None,
    baseline_param_source: str = "default",
    filter_cutoff_hz: float | None = None,
    filter_order: int = 8,
    filter_kind: str = "bessel",
    merge_gap_s: float | None = None,
    exclude_regions: Sequence[tuple[float, float]] | None = None,
    pore_id: str = "",
    recording: str = "",
    units: str = "pA",
    cancel: CancelFlag | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find translocation events with a Schmitt trigger and characterise them.

    Args:
        signal: Current trace samples (1-D). float32 in is fine; it is upcast to
            float64 internally.
        fs: Sample rate in Hz. Required -- dwell is undefined without a rate.
        polarity: +1 or -1 making `dev = polarity*(current-baseline)` positive
            in events. `None` measures it from the data via `detect_polarity`
            (verdict ``'measured'``) and uses the measured value. Anything else
            is REFUSED (`baseline.require_polarity`): it is a direction, not a
            gain, and any other value rescales `dev` against thresholds taken
            from the unscaled noise -- 0 returned an empty table with no warning,
            7 returned 56 rows against 20 real events.
        k_in: Enter threshold, in units of the noise RMS.
        k_out: Exit threshold, in units of the noise RMS. Must be < `k_in`;
            the gap between them is what stops noise chatter at the crossing
            from shattering one event into many.
        min_dwell_s: Events shorter than this are dropped as sub-resolution.
        max_dwell_s: Events longer than this are kept but marked
            `accepted=False` with `flag='dwell>max_dwell_s'` -- a clog is real
            signal the user must see, not something to disappear. NOTE that it
            also RIGHT-TRUNCATES the accepted dwell sample: fitting those events
            with a left-truncation-only likelihood biases tau short (-3.7% when
            the limit sits at 5 tau, -24% at 2.5 tau). The limit is stamped into
            provenance, and `analysis.modules.dwell_fit_inputs` hands it to the
            fit as `t_max` so the likelihood corrects for it.
        baseline: Precomputed baseline (scalar or full length). If None, the
            characterisation baseline is measured by
            `baseline.characterisation_baseline` -- the app's ONE door to an
            estimator -- on the (optionally filtered) trace. Passing an array
            bypasses the estimator entirely, so no ``baseline_fingerprint`` is
            stamped: the rows cannot name what measured them, and a later rebuild
            correctly declines to guess.
        baseline_config: The estimator configuration, or None for
            `baseline.DEFAULT_BASELINE`. THE SAME OBJECT flows through the ML
            characteriser, `truth` and `characterize.recharacterize_span`, so one
            recording yields one baseline no matter which path measured it. This
            detector holds no window constant of its own; it used to call
            `estimate_baseline` at its silent 10 ms default while the ML path used
            200 ms, and the same event reported 32.04 pA or 67.69 pA depending on
            which detector found it.
        baseline_param_source: How those parameters were chosen -- one of
            `baseline.PARAM_SOURCES`. Recorded, because no measurement can recover
            it afterwards.
        filter_cutoff_hz: If given, the trace is low-passed at this cutoff
            before detection *and* characterisation, and events with
            `dwell_s < 2*rise_s` are marked `filter_limited`.
        filter_order: Order of that low-pass. Ignored by `filter_kind='gaussian'`
            (a symmetric kernel has no order). Defaults to the electrophysiology
            convention, 8.
        filter_kind: `'bessel'` (default; flat group delay, no ringing on a step),
            `'butter'` (sharper roll-off, overshoots) or `'gaussian'` (no overshoot
            at all). Every kind runs zero-phase, so the applied magnitude is the
            SQUARE of the designed one -- see `filters.lowpass`. The rise time
            stamped into provenance is measured through whichever one you pick, so
            `filter_limited` stays correct for all three.
        merge_gap_s: Merge events separated by less than this gap.
        exclude_regions: `[(t0,t1),...]` in seconds to ignore (clogs, bubbles,
            voltage changes). Excluded time is subtracted from
            `analyzed_duration_s` and events overlapping it are removed.
        pore_id: Physical pore identifier stamped into every event row.
        recording: Source recording name stamped into every event row.
        units: Signal units label carried into provenance/labels.
        cancel: Zero-arg predicate polled between the pipeline's stages and,
            through `characterize`, inside its per-event loops. `None` uses the
            AMBIENT token (`analysis.cancel.cancel_scope`), which is what makes a
            detection started from the GUI worker stoppable without any caller --
            `modules.DetectSchmitt`, `truth.sweep_threshold`, the ML detector --
            having to forward it. Raises `analysis.cancel.OperationCancelled`; a
            cancelled detection returns nothing, never a half-built table.

    Returns:
        `(df, provenance)`. `df` is the 20-column events DataFrame (see
        `events.SCHEMA`); `provenance` records every parameter, the noise
        estimate, the polarity and its evidence, and `analyzed_duration_s`.

    Gotcha: `analyzed_duration_s` -- the denominator of every event-rate number
    -- accounts for `exclude_regions`. It is the recorded duration only when
    nothing was excluded.
    """
    if fs is None:
        raise ValueError("fs is required; dwell is undefined without a sample rate")
    fs = float(fs)
    _validate_detection_timing(fs, min_dwell_s, max_dwell_s, merge_gap_s)
    if not np.isfinite(k_in) or not np.isfinite(k_out):
        raise ValueError(f"k_in and k_out must be finite, got {k_in!r} and {k_out!r}")
    if not (k_out < k_in):
        raise ValueError(
            f"k_out ({k_out}) must be < k_in ({k_in}). Equal thresholds are a "
            "single-threshold detector, which chatters at the crossing and "
            "shatters events; that is the bug hysteresis exists to prevent."
        )
    if not (k_out > 0):
        raise ValueError(f"k_out must be positive, got {k_out}")
    # POLARITY IS A DIRECTION, NOT A GAIN, and it was the one knob on this call with
    # nothing standing behind it. It multiplies the deviation while the thresholds are
    # derived from the UNSCALED noise, so any other value silently rescales the
    # detector: measured on 20 blockades of 100 pA against a 1000 pA open pore with
    # 5 pA noise, polarity=-1 finds the 20 at 99.79 pA, polarity=0 zeroes dev and
    # returns a clean EMPTY table with no warning at all, polarity=7 returns 56 rows at
    # a median depth of 36.02 pA and polarity=100 returns 403 rows at 375.83 pA -- a
    # blockade a third of the open pore deep that is nowhere in the data. Each value
    # was stamped verbatim into ``provenance['polarity']``, which is what an export
    # writes and a row rebuild reads back, so the wrong number outlived the run. The
    # reachable door is not the GUI combo (auto/+1/-1) but the public API and
    # `modules._detect_kwargs`, which forwards arbitrary ``ctx.params`` keys straight
    # through -- a restored, replayed or batch params mapping lands here.
    # `baseline.require_polarity` is the gate, not a copy of it here: the same value
    # reaches the estimator doors from the same places, and it now refuses at all of
    # them with one sentence. ``None`` still means "measure it" -- that is this call's
    # own decision and it is made below, against `detect_polarity`.
    if polarity is not None:
        require_polarity(polarity)

    # Cancellation is polled at every stage boundary below. The stages are the
    # granularity a Cancel can achieve here: each one is a single vectorised call
    # into numpy/scipy that cannot be interrupted from Python. Measured on the
    # owner's 2.99 M-sample -600 mV recording the whole of detect_events is 0.315 s
    # (baseline 0.040 s, characterize 0.087 s), so the worst-case latency a stage
    # boundary imposes is well under a tenth of a second -- and the per-event loops
    # inside characterize poll far more finely than that.
    check_cancelled(cancel)

    # -- optional pre-filter -------------------------------------------------
    if getattr(baseline_config, "open_pore_reference", None) is not None:
        from .reference_context import validate_open_pore_context
        validate_open_pore_context({"baseline_config": baseline_config},
                                  units=units, module_id="detect.schmitt")
        if baseline is not None:
            raise ValueError("Choose a reproducible open-pore reference or a supplied baseline, not both.")
    filter_rise_s: float | None = None
    work = np.asarray(signal)
    if filter_cutoff_hz is not None:
        if not (0 < filter_cutoff_hz < fs / 2):
            raise ValueError(
                f"filter_cutoff_hz ({filter_cutoff_hz}) must be in (0, fs/2) = "
                f"(0, {fs / 2})"
            )
        # The rise time MUST be measured through the same kind/order that touched the
        # data, or `filter_limited` (and the resolution floor the GUI prints) is a
        # statement about a filter that was never applied.
        work = lowpass(
            signal, fs, filter_cutoff_hz, order=filter_order, kind=filter_kind
        )
        filter_rise_s = float(
            measure_rise_time(
                fs, filter_cutoff_hz, order=filter_order, kind=filter_kind
            )
        )

    cur = np.asarray(work, dtype=np.float64)
    if cur.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {cur.shape}")
    if cur.size == 0:
        raise ValueError("recording is empty")
    _check_finite(cur, "recording current")
    n = cur.size

    # -- polarity request bookkeeping ---------------------------------------
    polarity_requested = polarity
    est_polarity = None if polarity is None else int(polarity)

    # -- baseline ------------------------------------------------------------
    check_cancelled(cancel)
    baseline_provided = baseline is not None
    baseline_prov: dict[str, Any] = {}
    baseline_warnings: tuple[str, ...] = ()
    if baseline is None:
        # `work`, NOT `cur`. Both carry the same numbers — the door upcasts to float64 itself,
        # with the same `.astype(np.float64)` that produced `cur` — but `cur` is a FRESH OBJECT
        # ON EVERY CALL, and the door's whole-recording memo (and the polarity memo one rung
        # below it) are keyed on the identity of the array they were handed. The app's
        # conditioned trace is float32 (`modules.preprocess_for_detection` returns contiguous
        # float32), so every detection was converting it anew and missing both caches.
        # [MEASURED] on the owner's -600 mV recording, three detections over one array:
        # float32 in -> 1.093 / 1.102 / 1.099 s; the same array as float64 -> 1.089 / 0.185 /
        # 0.185 s. `sweep_threshold` is 32 detections over ONE trace and was paying the full
        # baseline 32 times. The pre-P1 tree behaves the same way (0.961 / 0.959 / 0.959), so
        # this is older than the polarity ladder — the ladder only made each miss dearer.
        result = characterisation_baseline(
            work, fs, config=baseline_config, polarity=est_polarity,
            param_source=baseline_param_source,
        )
        base = _as_baseline(result.baseline, n)
        baseline_method = result.config.method
        baseline_prov = result.provenance()
        baseline_warnings = result.warnings
    else:
        base = _as_baseline(baseline, n)
        baseline_method = "provided"
    _check_finite(base, "baseline")

    from .preview_capture import publish_detection_trace
    publish_detection_trace(cur, base)

    resid = cur - base
    noise = float(noise_rms(resid))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError(_unmeasurable_noise_message(
            noise, "noise RMS", supplied_baseline=(
                baseline_provided or bool(baseline_prov.get("baseline_reference_applied"))
            ) and bool(np.any(cur != cur[0])),
        ))

    # -- polarity: measure it (always, for the evidence record) --------------
    check_cancelled(cancel)
    measured_polarity, evidence = detect_polarity(cur, base, k=k_in)
    if polarity is None:
        polarity = measured_polarity
        polarity_source = "measured"
    else:
        polarity = int(polarity)
        polarity_source = "provided"

    # There is no stored polarity to contradict in the signal-first model; the
    # soft notes below are the honesty channel for a measurement that was
    # absent, arbitrated, or weak. They gate on the EVIDENCE, not only on the
    # verdict: the savgol-at-defaults regression (CF2-4) was a confidence-0.33
    # flip carrying verdict 'measured', invisible to a gate that only knew
    # 'undetermined' -- detection searched the wrong direction and reported a
    # clean-looking zero. The arbitrated/weak notes apply only when the
    # measurement is what detection actually used; a provided polarity is the
    # user's own instruction and needs no advisory about the vote it overrode.
    polarity_warning = None
    if evidence["verdict"] == "undetermined":
        polarity_warning = (
            "polarity is undetermined (no coherent excursions above threshold); "
            f"proceeding with polarity={polarity:+d}."
        )
    elif (
        polarity_source == "measured"
        and evidence.get("decided_by") == "coherent_samples"
    ):
        # The overruled vote is quoted from `n_*_runs`, NOT from the reported
        # `n_down`/`n_up`: those now carry the sample mass (detect_polarity's
        # Gotcha 4), and quoting them here would print the sample counts under
        # the words "coherent runs".
        polarity_warning = (
            f"polarity run vote was ambiguous ({evidence['n_down_runs']} down / "
            f"{evidence['n_up_runs']} up coherent runs) and the direction was taken "
            f"from the coherent signal mass instead ({evidence['n_down_samples']} "
            f"down vs {evidence['n_up_samples']} up samples, confidence "
            f"{evidence['confidence']:.2f}). Many brief counter-runs flanking "
            "larger excursions are the signature of a smoothing filter's edge "
            "lobes (e.g. Savitzky-Golay); if the direction is wrong, set the "
            "polarity override or disable the filter."
        )
    elif (
        polarity_source == "measured"
        and "decided_by" in evidence
        and not evidence.get("corroborated_by_samples", False)
    ):
        polarity_warning = (
            f"polarity was measured at low confidence "
            f"({evidence['confidence']:.2f}: {evidence['n_down_runs']} down / "
            f"{evidence['n_up_runs']} up coherent runs, with no decisive sample-mass "
            f"majority either); proceeding with polarity={polarity:+d}. If the "
            "event count looks wrong, set the polarity override -- and if a "
            "smoothing filter is enabled, its edge lobes may be the cause."
        )

    # -- dev, with excluded regions detected-through -------------------------
    # One spare trailing slot so a run touching the last sample stays in bounds
    # for the reduceat in _run_maxima.
    dev_padded = np.empty(n + 1, dtype=np.float64)
    np.multiply(resid, polarity, out=dev_padded[:n])
    dev_padded[n] = -np.inf
    dev = dev_padded[:n]

    # Excluded regions are NOT zeroed out of `dev`. Zeroing would truncate an
    # event that straddles a boundary into a spurious short event ending exactly
    # at the edge; instead we detect on the full trace and drop any event that
    # overlaps an exclusion (below), which removes straddlers and internals
    # alike. Excluded time is still subtracted from the analysed duration.
    ex = _exclusion_samples(exclude_regions, n, fs)
    excluded_samples = sum(i1 - i0 for i0, i1 in ex)
    analyzed_duration_s = (n - excluded_samples) / fs
    if analyzed_duration_s <= 0:
        raise ValueError(
            "exclude_regions removes the entire recording; nothing left to "
            "analyse."
        )

    # -- Schmitt trigger, vectorised ----------------------------------------
    check_cancelled(cancel)
    thr_in, thr_out = k_in * noise, k_out * noise
    starts, ends, n_candidates = _hysteresis_runs(dev_padded, n, thr_in, thr_out)
    # The BIGGEST rejection the detector makes, and until now the only one with no
    # counter: every run that crossed the EXIT threshold (k_out·noise) but whose peak
    # never reached the ENTER threshold (k_in·noise) is noise, and is dropped here.
    # On the owner's -600 mV apoferritin recording that is 18,201 of 18,477 candidate
    # runs -- 98.5 % of everything the trigger saw -- which left the provenance audit
    # trail unable to close: 18477 candidates, 276 events, and every other counter
    # zero. Counted here, at the test that does the rejecting.
    n_rejected_below_enter = n_candidates - int(starts.size)

    # -- merge close neighbours ---------------------------------------------
    # Same rule as the post-hoc analysis.merge: group runs whose gap
    # (next.start - cur.end) is <= the window, then keep each group's
    # [first.start, last.end]. group_starts_by_gap is the shared definition.
    #
    # ...AND THE SAME BRAKE. The gap rule is transitive and transitivity has no stop:
    # a regularly spaced train of clean, well-separated events collapses into ONE row
    # the moment the window exceeds the spacing. Reproduced HERE, at detection time,
    # not only post-hoc: 40 isolated 0.4 ms / 300 pA events spaced 8 ms apart, detected
    # with merge_gap_s = 10 ms, became one event of dwell 312.40 ms and depth 0.70 pA
    # (a median over a span that is 96 % open pore, so the true 300 pA blockade is
    # out-voted), accepted, with a blank flag. The occupancy floor was added to
    # analysis.merge for exactly that failure and guarded only the post-hoc path, which
    # left the identical pathology one Detection-tab spinbox away. Same shared floor,
    # applied to the groups BEFORE their ends are taken: a group whose fragments cover
    # less than MIN_MERGE_OCCUPANCY of the span they would make is split back into
    # singletons, so the group ENDS below are taken from the floor's mask and not from
    # the gap rule's.
    n_before_merge = int(starts.size)
    n_merges_refused = 0
    n_merges_marginal = 0
    refused_merge_occupancy: list[float] = []
    if merge_gap_s is not None and starts.size > 1:
        is_group_start = group_starts_by_gap(
            starts, ends, gap_window_samples(merge_gap_s, fs)
        )
        is_group_start, refused, marginal = split_low_occupancy_groups(
            starts, ends, is_group_start
        )
        n_merges_refused = len(refused)
        n_merges_marginal = len(marginal)
        refused_merge_occupancy = [round(float(o), 4) for o in refused]
        is_group_end = np.concatenate((is_group_start[1:], [True]))
        starts, ends = starts[is_group_start], ends[is_group_end]
    n_merged = n_before_merge - int(starts.size)

    # -- duration gates ------------------------------------------------------
    dwell = (ends - starts) / fs
    long_enough = dwell >= min_dwell_s
    n_rejected_short = int((~long_enough).sum())
    starts, ends = starts[long_enough], ends[long_enough]

    # -- drop anything touching an excluded region ---------------------------
    n_rejected_excluded = 0
    if ex and starts.size:
        drop = np.zeros(starts.size, dtype=bool)
        for i0, i1 in ex:  # loop over regions (a handful), never over samples
            drop |= (starts < i1) & (ends > i0)
        n_rejected_excluded = int(drop.sum())
        starts, ends = starts[~drop], ends[~drop]

    # -- events cut off by the EDGE of the analysed span ---------------------
    # The same problem an exclusion straddler poses, and the same answer. A run that
    # is still above threshold at sample 0 (or at sample n-1) did not necessarily
    # start there (or end there): the rest of it is outside the array. Its dwell is a
    # lower bound, its depth is the median of whatever fraction survived, and its
    # baseline was read from the one flank that exists -- none of the three is a
    # measurement of the event.
    #
    # Measured on the user's -600 mV recording, cutting a real 2.400 ms / 43.7 pA /
    # 0.318 dI/I0 event in half:
    #
    #   tail half (end_idx == n): dwell 3.220 ms (+34%), depth 64.1 pA (+47%)
    #   head half (start_idx == 0): depth 125.6 pA, frac_depth 3.148
    #
    # -- a 315% blockade, i.e. more current removed than the pore was carrying, and
    # every one of those rows came back flag='', accepted=True. Two such rows per
    # segment, and the segmentation feature exists to make many segments.
    #
    # Unlike an exclusion straddler these rows are KEPT: the excluded case drops
    # because the user declared that stretch untrustworthy, whereas here the event is
    # real and the user needs to see that their cut landed on one. Kept, de-accepted
    # and flagged is what ``max_dwell_s`` already does for the same kind of "real row,
    # not a measurement" case.
    #
    # The RULE itself is not here. ``characterize`` applies it, against the
    # ``clip_bounds`` it is given, so the ML detector and every row-rebuild path get
    # the identical determination instead of a second copy of this line that drifts.
    # This pre-count exists only because provenance is assembled before the
    # characterisation; it is overwritten below from the rows that actually carry the
    # flag, which is what stops provenance claiming a clipped event no row admits to.
    n_clipped = int(((starts <= 0) | (ends >= n)).sum()) if starts.size else 0

    provenance: dict[str, Any] = {
        "nanoscope_version": _VERSION,
        "detector": "schmitt_hysteresis",
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "recording": recording,
        "source": recording or None,
        "pore_id": pore_id,
        "units": units,
        "fs": fs,
        "n_samples": n,
        "duration_s": n / fs,
        "analyzed_duration_s": analyzed_duration_s,
        "exclude_regions": [tuple(map(float, r)) for r in (exclude_regions or [])],
        "excluded_duration_s": excluded_samples / fs,
        "k_in": float(k_in),
        "k_out": float(k_out),
        "min_dwell_s": float(min_dwell_s),
        "max_dwell_s": None if max_dwell_s is None else float(max_dwell_s),
        "merge_gap_s": None if merge_gap_s is None else float(merge_gap_s),
        "noise_rms_pA": noise,
        "noise_method": "MAD (filters.noise_rms)",
        "threshold_in_pA": float(thr_in),
        "threshold_out_pA": float(thr_out),
        "polarity": polarity,
        "polarity_requested": polarity_requested,
        "polarity_source": polarity_source,
        "polarity_measured": measured_polarity,
        "polarity_evidence": evidence,
        "polarity_warning": polarity_warning,
        "baseline_method": baseline_method,
        "baseline_provided": baseline_provided,
        "baseline_median_pA": float(np.median(base)),
        # The estimator's own block, written by the ONE writer of these keys
        # (`BaselineResult.provenance`) so a Schmitt row and an ML row measured on
        # the same recording carry the SAME ``baseline_fingerprint`` and are
        # directly comparable. Empty when the caller handed us a baseline array:
        # there is then no estimator to name, and the absent fingerprint is the
        # honest signal a rebuild reads.
        **baseline_prov,
        "baseline_warnings": list(baseline_warnings),
        "filter_cutoff_hz": (
            None if filter_cutoff_hz is None else float(filter_cutoff_hz)
        ),
        # The DESIGN of the filter, not just its corner. Without these an exported
        # EventSet cannot be reproduced from its own provenance: a 10 kHz 8-pole
        # Bessel and a 10 kHz 2-pole Butterworth are different measurements, and
        # zero-phase (filtfilt) squares whichever one it is. `filters` documents
        # that the CALLER stamps these; this is that caller.
        "filter_kind": None if filter_cutoff_hz is None else str(filter_kind),
        "filter_order": (
            None
            if filter_cutoff_hz is None or filter_kind == "gaussian"
            else int(filter_order)
        ),
        "filter_zero_phase": None if filter_cutoff_hz is None else True,
        "filter_rise_s": filter_rise_s,
        # THE AUDIT TRAIL, and it BALANCES:
        #
        #   n_candidate_runs
        #     - n_rejected_below_enter    (peak never reached k_in·noise)
        #     - n_merged                  (folded into a neighbour by merge_gap_s)
        #     - n_rejected_short          (shorter than min_dwell_s)
        #     - n_rejected_excluded       (overlapped an excluded region)
        #     == n_events
        #
        # in that order, because that is the order the gates are applied above and
        # each one counts what reached it. ``n_clipped_boundary`` is NOT a term:
        # those rows are kept in the table (flagged and de-accepted), so they are
        # already inside ``n_events``. Neither is ``n_merges_refused``, for the same
        # reason -- a refused merge drops nothing, it only declines to fold rows
        # together -- see the note on that key below.
        # `tests/analysis/test_detect_audit_trail.py` pins the identity so a new gate
        # cannot be added without a counter for it.
        "n_candidate_runs": n_candidates,
        "n_rejected_below_enter": n_rejected_below_enter,
        "n_merged": n_merged,
        "n_rejected_short": n_rejected_short,
        "n_rejected_excluded": n_rejected_excluded,
        #: Groups the gap rule proposed and the OCCUPANCY FLOOR declined, with the
        #: occupancy of each (see `split_low_occupancy_groups`).
        #:
        #: DELIBERATELY NOT A TERM OF THE IDENTITY ABOVE. The detector is documented as
        #: needing a counter for every rejection it makes, and this is not a rejection:
        #: nothing is dropped by a refused merge. The group's events all survive, as
        #: singletons, and go on to the dwell and exclusion gates like any other row --
        #: so they are already inside ``n_events``, and subtracting them would break the
        #: balance in the same direction ``n_clipped_boundary`` would. What the refusal
        #: changes is ``n_merged``, which counts rows FOLDED AWAY and therefore already
        #: excludes them: the identity closes with or without this key, and it is here
        #: because "the window you chose proposed 12 merges and 12 were declined" is
        #: otherwise indistinguishable from "nothing was close enough to merge" -- both
        #: report ``n_merged == 0``.
        "n_merges_refused": n_merges_refused,
        "refused_merge_occupancy": refused_merge_occupancy,
        #: Merges KEPT between the floor and ``MARGINAL_MERGE_OCCUPANCY``. Counted
        #: for the reason that constant gives: a third-open-pore span is defensible
        #: (slow, multi-level translocations look like that) but it is also what a
        #: runaway merge looks like on its way down, so it must not be indistinguishable
        #: from a clean stitch afterwards. The post-hoc merge records the same fact
        #: per-span under ``provenance['merged'][LOW_OCCUPANCY_TOKEN]``; a detection has
        #: no such note, and a count is what this path can honestly carry.
        "n_merges_marginal": n_merges_marginal,
        #: Events still above threshold at the first or last analysed sample, so cut
        #: off by the edge of the span rather than by the pore. Reported, kept in the
        #: table, and NOT accepted -- see ``flag == 'clipped_by_boundary'``.
        "n_clipped_boundary": n_clipped,
        # The span these rows were judged against, so a MERGE or a mask resize can
        # re-judge them against the same edges instead of against whatever array it
        # happens to hold. This detector is always handed the whole analysed array
        # -- a segment document included, whose indices are its own -- so the span
        # is the array. Written through the one writer both detectors share.
        **span_provenance(0, n),
    }

    if starts.size == 0:
        provenance["n_events"] = 0
        provenance["n_filter_limited"] = 0
        df = empty_df()
        provenance["measurement_quality"] = build_measurement_quality(cur, base, df, fs, provenance)
        return df, provenance

    df = characterize(
        cur,
        base,
        starts,
        ends,
        fs,
        polarity,
        noise_rms=noise,
        filter_rise_s=filter_rise_s,
        # Both rejection rules are characterize's, not this function's. They used to
        # be applied to the returned frame here, which is precisely why they never
        # reached the ML detector, the merge or the Event Inspector.
        max_dwell_s=max_dwell_s,
        clip_bounds=(0, n),
        pore_id=pore_id,
        recording=recording,
        units=units,
        cancel=cancel,
    )

    if filter_rise_s is not None:
        # Stated by the contract, so assert it here rather than trusting the
        # upstream default; OR-ed so a stricter call from characterize survives.
        limited = df["dwell_s"].to_numpy() < 2.0 * filter_rise_s
        df["filter_limited"] = df["filter_limited"].to_numpy(dtype=bool) | limited

    # Report what the ROWS say, not what a parallel computation predicted: the two
    # agreed here, but a provenance count that cannot disagree with the table is one
    # fewer way for "n_clipped_boundary=1 with zero flagged rows" to happen again.
    provenance["n_clipped_boundary"] = n_clipped_boundary(df)

    provenance["n_events"] = int(len(df))
    provenance["n_filter_limited"] = int(df["filter_limited"].sum())
    provenance["measurement_quality"] = build_measurement_quality(cur, base, df, fs, provenance)
    return df, provenance


# ---------------------------------------------------------------------------
# conventional detector kernels: fixed threshold, peaks, derivative, TKEO
# ---------------------------------------------------------------------------
# WHICH ONE TO REACH FOR. A menu of five detectors with no guidance is worse than
# one detector, so the rule of thumb is here, once, and each kernel's docstring
# repeats the half of it that is about itself:
#
#   drifting baseline, ordinary translocations (>= 2 samples)  -> `detect_events`
#       the Schmitt trigger in units of sigma. THE DEFAULT. None of the kernels
#       below replaces it; they cover the cases where a level-and-dwell test is
#       the wrong test.
#   you know the open-pore level in the channel's own units    -> `detect_fixed_threshold`
#       and the trace does not drift. "An event is any sample below 900 pA" is a
#       statement a sigma threshold cannot make, because sigma moves with the noise.
#   spikes of 1-5 samples, isolated, amplitude is the signal   -> `detect_peaks`
#       prominence/height/distance, then a width at half height for the interval.
#   the LEVEL is buried under drift but the EDGES are sharp    -> `detect_derivative`
#       |dI/dt| finds the two edges and pairs them into an interval.
#   very short bursts, too short to clear a level threshold    -> `detect_tkeo`
#       psi[n] = x[n]^2 - x[n-1]*x[n+1] answers to amplitude AND rate of change.
#
# FOUR OF THE FIVE MEASURE THE SAME WAY -- and the exception is deliberate.
# `detect_peaks`, `detect_derivative` and `detect_tkeo` take baseline, polarity and
# noise from `detection_basis`, which calls the app's one baseline door
# (`baseline.characterisation_baseline`), the one polarity rule (`detect_polarity`)
# and `noise_level` (MAD by default, i.e. `filters.noise_rms`). `detect_events`
# measures the same three quantities through the same three doors on its own path.
# `detect_fixed_threshold` takes NONE of them: an absolute level in pA is the one
# statement that must not be re-referenced to an estimated baseline, which is why it
# is the kernel you can still run when the baseline estimator is the thing you
# distrust -- and why it is the one that drifts (see its docstring). So: one baseline,
# one polarity and one noise definition wherever a detector needs them at all.
#
# THE FOUR KERNELS BELOW RETURN ``(starts, ends)``, half-open sample index arrays of
# dtype int64 -- exactly what `detect_events` and `detect_cusum.detect_events_cusum`
# hand to `characterize` internally. (`detect_events` itself returns the finished
# ``(DataFrame, provenance)``; it is the whole pipeline, not a kernel.) Nothing about
# the 20 frozen EventSet columns changes; the kernels only decide WHICH spans get
# measured.
#
# [MEASURED] AGAINST THE APP'S OWN ANSWER, on the owner's 15 s apo-ferritin recording
# (``..._0004_process _ -1000 mV_ 385-400 s time trace.abf``, 749,651 samples,
# 50 kHz). Reference = `detect_events` at ITS defaults: 115 events, median dwell
# 20.1 ms (max 33.9), median depth 90.3 pA, median SNR 15.8, and a median rise time
# of 1.01 ms over the 88 of the 115 events where `rise_s` is defined at all.
# Recall = reference events overlapped; precision = returned spans overlapping a
# reference event; each kernel run from the SAME `DetectionBasis`:
#
#   kernel / settings                  spans  recall  prec  median span  longest
#   detect_tkeo link_gap_s=2e-3          108   0.826  0.796     19.6 ms    33.3 ms
#   detect_peaks rel_height=0.9           91   1.000  1.000     24.9 ms    37.8 ms
#   detect_fixed_threshold 5s/2s         158   0.896  0.627     15.5 ms    33.5 ms
#   detect_peaks DEFAULTS                278   0.852  1.000      0.44 ms   28.6 ms
#   detect_tkeo DEFAULTS              14,006   0.817  0.998      0.02 ms    0.2 ms
#   detect_derivative DEFAULTS           146   0.678  1.000      4.49 ms  412.9 ms
#
# TWO OF THEM NEED TUNING BEFORE THEY ARE HONEST ON A RECORDING LIKE THIS ONE, and
# their docstrings carry the recipe and its cost:
#   * `detect_derivative` -- its shipped defaults return a 412.9 ms span (one missed
#     falling edge) and reach only 0.678 recall here. NO setting found reaches both
#     high recall and a bounded span on this file; it is not a detector to select
#     blind on 20 ms events with 1 ms edges. See its "ON THE OWNER'S OWN RECORDING".
#   * `detect_peaks` -- correct but NOT dwell-comparable at its defaults: a width at
#     half height is not a threshold crossing, so it reports 0.44 ms where the app
#     reports 20.1 ms, and 2.42 spans per reference event. `rel_height` is the knob.
# `detect_events` remains the default detector; nothing here displaces it.

#: Noise bases `noise_level` will accept. MAD is first because it is what the rest
#: of the app uses and what every existing threshold in this module is quoted in.
NOISE_METHODS: tuple[str, ...] = ("mad", "std", "iqr")

#: 1/(2*0.6744897501960817) -- scales the interquartile range of a Gaussian to sigma,
#: the same job 1.4826 does for the MAD.
_IQR_TO_SIGMA = 0.7412689708268266


def noise_level(x: np.ndarray, method: str = "mad") -> float:
    """Robust scale of `x` in the units of `x`, by an EXPLICITLY CHOSEN basis.

    The rest of this module hard-codes one basis -- ``filters.noise_rms``, i.e.
    1.4826*MAD -- and for detection that is the right default: events are one-sided
    contamination of the very trace the threshold is derived from, and the MAD is the
    only one of the three below that survives them. It is not the right answer every
    time, which is why this is a parameter now rather than a constant:

    * ``'mad'`` (default) -- ``filters.noise_rms``, IMPORTED not copied, so a kernel's
      threshold and `detect_events`' threshold mean the same thing. 50% breakdown:
      the estimate is intact until events occupy half the trace. Measured on 12 pA
      Gaussian noise contaminated with 20-sigma blockades it reads +1.3% at 1%
      occupancy, +6.1% at 5%, +14.6% at 10% (see `filters.noise_rms`).
    * ``'std'`` -- the textbook sigma. 0% breakdown: on that same contaminated trace
      it reads +123%/+347%/+508%, and a threshold five times too high finds nothing.
      Use it ONLY on a stretch you have already established is event-free (a Working
      Region over open pore), where it is the most efficient of the three.
    * ``'iqr'`` -- 0.74127*(p75-p25). 25% breakdown, slightly more efficient than the
      MAD on clean Gaussian noise. In practice it tracks the MAD closely and is here as
      a cross-check: when the two disagree by more than a few percent, the trace is not
      the symmetric-noise-plus-sparse-events you think it is.

    [MEASURED] on the owner's 15 s apo recording (``..._0004_process _ -1000 mV_
    385-400 s time trace.abf``, 749,651 samples, channel IN15, real events in it):
    MAD 5.7311 pA, IQR 5.7891 pA, std 32.5321 pA. The std is 5.7x the MAD because the
    events are in the trace, so a 5-sigma threshold taken from it sits at 163 pA --
    above nearly every event the Schmitt trigger finds at 28.7 pA. On the clean 40-event
    synthetic used throughout this section: MAD 5.1472, IQR 5.1480, std 14.8946 against
    a true sigma of 5.0.

    All three are scaled to the Gaussian sigma, so `k=5` means the same number of
    sigma whichever you pick -- what changes is how the sigma is estimated, not the
    units of the threshold.

    Raises:
        ValueError: on an unknown method, an empty or non-1-D array, or non-finite
            samples (a threshold derived from a NaN is not a threshold).
    """
    m = str(method).lower()
    if m not in NOISE_METHODS:
        raise ValueError(
            f"unknown noise method {method!r}; expected one of {NOISE_METHODS}"
        )
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError("cannot estimate noise from an empty array")
    a = a.astype(np.float64, copy=False)
    if m == "mad":
        return float(noise_rms(a))          # the one existing definition
    _check_finite(a, "noise sample")
    if m == "std":
        return float(np.std(a))
    q75, q25 = np.percentile(a, [75.0, 25.0])
    return float(_IQR_TO_SIGMA * (q75 - q25))


@dataclass(frozen=True)
class DetectionBasis:
    """What every kernel needs before it can threshold anything, measured ONCE.

    Built by `detection_basis`. Carrying it as an object is what lets a caller run
    two kernels over one trace (a comparison, a sweep, a preview) without paying for
    -- or, worse, silently disagreeing about -- the baseline, the polarity and the
    noise. `dev` obeys the module's sign convention: ``dev = polarity*(current -
    baseline)``, positive inside an event, blockade and enhancement alike.
    """

    dev: np.ndarray
    """polarity*(current - baseline), float64, full length. Positive inside events."""

    current: np.ndarray
    """The (optionally filtered) trace the baseline was measured on, float64."""

    baseline: np.ndarray
    """Open-pore level per sample, float64, full length."""

    polarity: int
    """+1 or -1; the factor that makes `dev` positive inside an event."""

    noise: float
    """Scale of the residual in the channel's units, by `noise_method`."""

    noise_method: str
    """Which basis measured `noise` -- one of `NOISE_METHODS`."""

    polarity_source: str
    """``'provided'`` or ``'measured'``."""

    polarity_evidence: dict[str, Any]
    """`detect_polarity`'s evidence dict, so a caller can show the vote."""

    baseline_method: str
    """``'provided'``, or the estimator method name from the baseline door."""


def detection_basis(
    signal: np.ndarray,
    fs: float,
    *,
    polarity: int | None = None,
    baseline: np.ndarray | float | None = None,
    baseline_config: BaselineConfig | None = None,
    baseline_param_source: str = "default",
    noise_method: str = "mad",
) -> DetectionBasis:
    """Measure baseline, polarity and noise for a kernel -- through the app's doors.

    This is the seam that keeps the five conventional detectors comparable. It calls
    `baseline.characterisation_baseline` (the ONE estimator door, memoised on the
    trace object, so two kernels over one array pay for the baseline once) and
    `detect_polarity` (the ONE polarity rule), and nothing else in this section is
    allowed to compute either.

    Args:
        signal: Current trace, 1-D. float32 is fine; upcast to float64 here. NOTE
            the memo in the baseline door is keyed on the array OBJECT, so hand the
            SAME array to both kernels of a comparison, and hand it float64 if you
            want the memo to hit (`detect_events` documents the 1.09 s -> 0.19 s
            measurement).
        fs: Sample rate, Hz.
        polarity: ``None`` measures it; +1/-1 is used verbatim; anything else is
            refused by `baseline.require_polarity` -- it is a direction, not a gain.
        baseline: Precomputed baseline (scalar or full length) to bypass the
            estimator. The kernels expose this for the same reason `detect_events`
            does: a Working Region re-detect should not re-measure the baseline.
        baseline_config: Estimator configuration, or ``None`` for
            `baseline.DEFAULT_BASELINE`.
        baseline_param_source: How those parameters were chosen (`baseline.PARAM_SOURCES`).
        noise_method: One of `NOISE_METHODS`; see `noise_level`.

    Returns:
        `DetectionBasis`.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if polarity is not None:
        require_polarity(polarity)
    work = np.asarray(signal)
    cur = np.asarray(work, dtype=np.float64)
    if cur.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {cur.shape}")
    if cur.size == 0:
        raise ValueError("recording is empty")
    _check_finite(cur, "recording current")
    n = cur.size

    if baseline is None:
        result = characterisation_baseline(
            work, fs, config=baseline_config,
            polarity=None if polarity is None else int(polarity),
            param_source=baseline_param_source,
        )
        base = _as_baseline(result.baseline, n)
        baseline_method = result.config.method
    else:
        base = _as_baseline(baseline, n)
        baseline_method = "provided"
    _check_finite(base, "baseline")

    resid = cur - base
    noise = noise_level(resid, noise_method)
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError(_unmeasurable_noise_message(
            noise, f"noise ({noise_method})", supplied_baseline=(
                baseline is not None or getattr(baseline_config, "open_pore_reference", None) is not None
            ) and bool(np.any(cur != cur[0])),
        ))

    measured, evidence = detect_polarity(cur, base, k=5.0)
    if polarity is None:
        pol, source = measured, "measured"
    else:
        pol, source = int(polarity), "provided"

    return DetectionBasis(
        dev=resid * pol,
        current=cur,
        baseline=base,
        polarity=pol,
        noise=noise,
        noise_method=str(noise_method).lower(),
        polarity_source=source,
        polarity_evidence=evidence,
        baseline_method=baseline_method,
    )


def _padded(dev: np.ndarray) -> np.ndarray:
    """`dev` with one -inf trailing slot, the layout `_run_maxima` requires."""
    out = np.empty(dev.size + 1, dtype=np.float64)
    out[: dev.size] = dev
    out[dev.size] = -np.inf
    return out


def _gate_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    fs: float,
    *,
    min_dwell_s: float = 0.0,
    merge_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge-then-dwell, by the SAME rules and in the SAME order as `detect_events`.

    `group_starts_by_gap` + `split_low_occupancy_groups` (the shared gap rule and its
    occupancy brake) and then the ``min_dwell_s`` floor -- in that order, because a
    merge can only lengthen a span and a floor applied first would drop fragments the
    merge was about to rescue. Every kernel below routes through here, so a
    ``merge_gap_s`` of 1 ms means one thing across all of them.

    Returns int64 arrays, always, including when empty.
    """
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    if merge_gap_s is not None:
        if merge_gap_s < 0:
            raise ValueError(f"merge_gap_s must be >= 0, got {merge_gap_s}")
        if s.size > 1:
            is_group_start = group_starts_by_gap(s, e, gap_window_samples(merge_gap_s, fs))
            is_group_start, _refused, _marginal = split_low_occupancy_groups(
                s, e, is_group_start
            )
            is_group_end = np.concatenate((is_group_start[1:], [True]))
            s, e = s[is_group_start], e[is_group_end]
    if min_dwell_s:
        if min_dwell_s < 0:
            raise ValueError(f"min_dwell_s must be >= 0, got {min_dwell_s}")
        keep = (e - s) / float(fs) >= float(min_dwell_s)
        s, e = s[keep], e[keep]
    return s.astype(np.int64, copy=False), e.astype(np.int64, copy=False)


def detect_fixed_threshold(
    signal: np.ndarray,
    fs: float,
    *,
    level: float,
    exit_level: float | None = None,
    direction: str | None = None,
    min_dwell_s: float = 0.0,
    merge_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Events as excursions past an ABSOLUTE level, in the channel's own units.

    REACH FOR THIS when you know your open-pore current and want to say so: "this
    pore runs at 1000 pA and anything below 900 pA is an event" is a statement about
    the physics of your pore, and it is exactly the statement a threshold in units of
    sigma cannot make -- sigma is re-measured on every trace, so the same k finds a
    different level on a noisier file, and a level you validated on Monday is not the
    level you get on Tuesday. It is also the only detector here that needs no
    baseline estimate at all, which makes it the honest choice on a trace whose
    baseline estimator you do not trust (a heavily gating pore, a partial clog).

    DO NOT reach for it on a DRIFTING trace. It is the one kernel that does not
    subtract a baseline, so drift walks the whole trace through the level.
    [MEASURED] on the standard 40-event synthetic (2 s, 40 x 1 ms, 100 pA deep, 5 pA
    noise) with a -40 pA/s ramp added and the level 50 pA below the STARTING open
    pore: recall 0.400, and the 16 spans it returns include one 1302.8 ms "event"
    that is really the tail of the recording. The same ramp upward hides events
    instead: recall 0.875. `detect_events` (sigma against a tracked baseline) is the
    detector for a drifting trace; on both ramps it stays at 1.000/1.000.

    RECALL/PRECISION, measured on the standard synthetic (level 950, exit 990,
    open pore 1000): 1.000 / 1.000, 40 spans for 40 events -- and on a LITERALLY
    noiseless square pulse (constant 1000 pA, samples [500, 600) at 900 pA) it
    returns exactly ``(500, 600)``. It is the only kernel here that can be run on
    such a trace at all: the other three go through `detection_basis`, whose noise
    estimate is 0 on a noiseless trace, and it refuses rather than divide by it.

    [MEASURED] ON THE OWNER'S OWN RECORDING (15 s apo, 749,651 samples; reference =
    `detect_events` at its defaults, 115 events, median dwell 20.1 ms), with the
    level set to the app's own bar expressed absolutely -- baseline median plus
    5 sigma to enter, 2 sigma to exit: 158 spans, recall 0.896, PRECISION 0.627,
    median span 15.5 ms. The precision is where the absolute level costs you: with
    no baseline to subtract, the pore's slow wander walks stretches of open pore
    past a level that was correct at the start of the file. Tune it by taking
    `level` from the open-pore stretch you actually care about rather than from the
    whole-file median, or use `detect_events` if the trace drifts at all.

    Args:
        signal: Current trace, 1-D, in the channel's units.
        fs: Sample rate, Hz.
        level: The ENTER level, absolute, same units as `signal`. A sample exactly
            AT the level is outside the event (the comparison is strict), which
            keeps a flat trace sitting on the level from producing a single
            recording-long event.
        exit_level: Optional hysteresis level, on the OPEN-PORE side of `level`
            (i.e. between `level` and the open pore). The event ends when the trace
            comes back past THIS level, not past `level`. Strongly recommended:
            with a single level the trace chatters while it loiters at the crossing
            and one event shatters into a burst of fragments -- counts inflate,
            dwells collapse. Measured on a 1 ms/100 pA event with 5 pA noise and the
            level parked 2 pA above the blocked level: 1 event with hysteresis
            (exit 990), 11 fragments without. ``None`` means a single-level detector, which this
            kernel permits (a plain absolute threshold is a thing a user explicitly
            asks for) but does not recommend.
        direction: ``'below'`` (events go below `level`, the ordinary blockade at
            positive bias), ``'above'``, or ``None`` to decide from the data --
            `level` below the trace median means ``'below'``. Auto is a convenience,
            not a measurement: at negative bias the open pore is negative and a
            blockade moves the signed current UP, so pass it explicitly if you care.
        min_dwell_s: Drop events shorter than this (0 = no floor).
        merge_gap_s: Merge events closer than this, through the app's shared gap rule
            and its occupancy brake (`_gate_intervals`).

    Returns:
        ``(starts, ends)``: half-open sample indices, int64, sorted, non-overlapping.
    """
    fs = float(fs)
    cur = np.asarray(signal, dtype=np.float64)
    if cur.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {cur.shape}")
    if cur.size == 0:
        raise ValueError("recording is empty")
    _check_finite(cur, "recording current")
    level = float(level)
    if not np.isfinite(level):
        raise ValueError(f"level must be finite, got {level}")

    if direction is None:
        direction = "below" if level < float(np.median(cur)) else "above"
    if direction not in ("below", "above"):
        raise ValueError(
            f"direction must be 'below', 'above' or None, got {direction!r}"
        )
    sign = -1.0 if direction == "below" else 1.0

    enter = sign * level
    if exit_level is None:
        exit_ = enter
    else:
        exit_ = sign * float(exit_level)
        if not (exit_ < enter):
            raise ValueError(
                f"exit_level ({exit_level}) must lie between level ({level}) and the "
                f"open pore for direction={direction!r}: the exit has to be the "
                "EASIER test, or the trigger has no hysteresis and chatters at the "
                "crossing -- the bug hysteresis exists to prevent."
            )

    x = sign * cur
    starts, ends = _hysteresis_runs(_padded(x), x.size, enter, exit_)[:2]
    return _gate_intervals(
        starts, ends, fs, min_dwell_s=min_dwell_s, merge_gap_s=merge_gap_s
    )


def detect_peaks(
    signal: np.ndarray,
    fs: float,
    *,
    basis: DetectionBasis | None = None,
    polarity: int | None = None,
    baseline: np.ndarray | float | None = None,
    baseline_config: BaselineConfig | None = None,
    noise_method: str = "mad",
    prominence_sigma: float = 5.0,
    height_sigma: float | None = 5.0,
    distance_s: float = 0.0,
    rel_height: float = 0.5,
    min_dwell_s: float = 0.0,
    merge_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Peaks in the deviation from baseline, widened into intervals at half height.

    REACH FOR THIS for SPIKES: excursions a few samples long where the amplitude is
    the measurement and the dwell barely exists. A level-and-dwell detector is built
    to reject those -- the exit threshold has to be crossed, the minimum dwell has to
    be cleared -- and a spike train is exactly the signal for which "how tall" is the
    question and "how long" is instrument bandwidth. Prominence is also the right
    test when events sit ON something: prominence measures a peak against its own
    surroundings, so a spike riding on a slow partial block is found at its true
    height above the local structure, where an absolute level would miss it.

    DO NOT reach for it for ordinary translocations of any length. A peak is a
    SINGLE SAMPLE and this kernel has to invent an interval around it; the interval
    comes from `scipy.signal.peak_widths` at `rel_height`, which for a square event
    is close to right (measured below) and for a multi-level event is not -- a
    two-level translocation has two peaks and this kernel will find two events unless
    `distance_s` or `merge_gap_s` says otherwise. It also cannot see a flat-topped
    event that never comes back down inside the trace.

    PRECISION/RECALL, measured (fs = 50 kHz, 5 pA noise, 40 events of 100 pA):
    at the defaults (prominence 5 sigma AND height 5 sigma, rel_height 0.5) it is
    1.000 / 1.000 on all three shapes -- 1 ms squares, 3-sample spikes and
    SINGLE-SAMPLE spikes -- and on a square pulse occupying [500, 600) in 0.01 pA of
    noise it returns (499, 600), i.e. the half-height bounds land within one sample
    of the true edges. (0.01 pA, not zero: this kernel measures its noise through
    `detection_basis`, which refuses a trace whose MAD is exactly 0 -- there is no
    threshold to set. `detect_fixed_threshold` is the one that runs on a literally
    noiseless trace.)

    [MEASURED] ON THE OWNER'S OWN RECORDING -- READ THIS BEFORE COMPARING DWELLS.
    15 s apo, 749,651 samples; reference = `detect_events` at its defaults, 115
    events, median dwell 20.1 ms. At the defaults: 278 spans, recall 0.852,
    precision 1.000 -- correct about WHERE, and 2.42 spans per reference event,
    with a median span of 0.44 ms against the app's 20.1 ms. That 46x is not an
    error, it is the definition: `rel_height=0.5` returns a width at HALF HEIGHT,
    and these events are 20 ms long with ~1 ms edges and internal structure, so
    each one lights up as several narrow half-height slices. DO NOT put this
    kernel's dwell in the same histogram as `detect_events`' without saying so.
    THE KNOB IS `rel_height`: at 0.9 the same recording gives 91 spans, recall
    1.000, precision 1.000, median span 24.9 ms against the app's 20.1 ms -- the
    only setting measured here that is both complete and dwell-comparable. It is
    NOT the default because it is the wrong answer for the shape this kernel exists
    for: on single-sample spikes rel_height 0.9 returns 11-sample spans where 0.5
    returns 2, an 11x dwell error on a 20 us event. Pick by the shape you have.

    THE DEFAULTS ARE NOT DECORATION. With ``height_sigma=None``, prominence alone
    admits 397 spans for those same 40 events (precision 0.101), because a noise
    peak's prominence is measured down to the deepest minimum it can reach before a
    higher sample -- and beside a real 20-sigma event that search runs for tens of
    milliseconds, so an ordinary 3-sigma noise bump inherits a 7-sigma prominence.
    Prominence discriminates SHAPE; only the height test knows how far from the
    baseline the trace actually went. With ``rel_height=1.0`` on the same trace,
    recall collapses to 0.025: the full-prominence base of the first peak runs to
    those same far minima and swallows the whole trace as one span.

    Args:
        signal: Current trace, 1-D.
        fs: Sample rate, Hz.
        basis: A `DetectionBasis` already measured on this trace; ``None`` measures
            one (see `detection_basis`). Pass it to compare two kernels over one
            trace without re-measuring the baseline.
        polarity, baseline, baseline_config, noise_method: forwarded to
            `detection_basis` when `basis` is None.
        prominence_sigma: Minimum prominence, in units of the noise. This is the
            main knob: 5 sigma is the same statistical bar `detect_events` sets with
            ``k_in=5``, but it is asked of the peak against its surroundings rather
            than against the baseline.
        height_sigma: Height bar in units of the noise, measured on ``dev`` (i.e.
            from the baseline), defaulting to the same 5 sigma `detect_events` uses.
            NOT optional in practice -- see the precision measurement above; ``None``
            disables it and costs an order of magnitude in precision on a trace with
            real events in it. Prominence answers "is this a peak?", height answers
            "did the current actually go anywhere?", and a peak detector on a
            nanopore trace needs both.
        distance_s: Minimum spacing between accepted peaks, seconds. 0 = no spacing
            test. Set it to just under your shortest expected inter-event gap to stop
            one ringing event from being counted several times.
        rel_height: Where the interval is cut, as a fraction of the peak's prominence
            measured down from the peak: 0.5 = full width at half maximum (the
            default, and for a square event within a sample or two of the true
            edges), 1.0 = the full prominence base (the whole event, but it runs into
            neighbouring structure and can swallow the gap between two close events).
        min_dwell_s: Drop events shorter than this. NOTE the peak widths are already
            narrow by construction; a floor here removes the spikes this kernel is
            for. Default 0.
        merge_gap_s: Merge events closer than this (shared gap rule + occupancy brake).

    Returns:
        ``(starts, ends)``: half-open sample indices, int64. Overlapping peak
        intervals are unioned before they are returned -- two peaks whose half-height
        spans overlap are one event, and an EventSet has no concept of a duplicate
        row, so emitting both would double-count one translocation in every rate,
        histogram and fit downstream.
    """
    fs = float(fs)
    b = basis if basis is not None else detection_basis(
        signal, fs, polarity=polarity, baseline=baseline,
        baseline_config=baseline_config, noise_method=noise_method,
    )
    dev = b.dev
    n = dev.size
    if not (prominence_sigma > 0):
        raise ValueError(f"prominence_sigma must be > 0, got {prominence_sigma}")
    if not (0.0 < rel_height <= 1.0):
        raise ValueError(f"rel_height must be in (0, 1], got {rel_height}")
    if distance_s < 0:
        raise ValueError(f"distance_s must be >= 0, got {distance_s}")

    distance = max(1, int(round(distance_s * fs))) if distance_s > 0 else None
    peaks, _props = find_peaks(
        dev,
        height=None if height_sigma is None else height_sigma * b.noise,
        prominence=prominence_sigma * b.noise,
        distance=distance,
    )
    if peaks.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    widths, _wh, left_ips, right_ips = peak_widths(dev, peaks, rel_height=rel_height)
    starts = np.floor(left_ips).astype(np.int64)
    ends = np.ceil(right_ips).astype(np.int64)
    # A peak whose width rounds to nothing is still one sample of event: the peak
    # itself. Without this a 1-sample spike -- the very thing this kernel is for --
    # comes back as an empty [i, i) span that characterize would measure over no
    # samples at all.
    ends = np.maximum(ends, starts + 1)
    np.clip(starts, 0, n - 1, out=starts)
    np.clip(ends, 1, n, out=ends)
    merged = _merge_intervals([(int(s), int(e)) for s, e in zip(starts, ends)])
    starts = np.array([s for s, _ in merged], dtype=np.int64)
    ends = np.array([e for _, e in merged], dtype=np.int64)
    return _gate_intervals(
        starts, ends, fs, min_dwell_s=min_dwell_s, merge_gap_s=merge_gap_s
    )


def detect_derivative(
    signal: np.ndarray,
    fs: float,
    *,
    basis: DetectionBasis | None = None,
    polarity: int | None = None,
    baseline: np.ndarray | float | None = None,
    baseline_config: BaselineConfig | None = None,
    noise_method: str = "mad",
    k: float = 5.0,
    threshold_per_s: float | None = None,
    max_pair_dwell_s: float | None = None,
    min_dwell_s: float = 0.0,
    merge_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Events as an EDGE PAIR in |dI/dt|: one rising edge, then its falling edge.

    REACH FOR THIS when the level is untrustworthy but the edges are not. Slow
    baseline wander, a pore that changes conductance mid-file, an unmatched
    electrode offset -- all of them move the LEVEL and none of them moves the
    DERIVATIVE, because differencing kills anything slower than the differencing
    interval. It is also the detector that answers "where did the current change?"
    rather than "where was the current low?", which is the right question when
    events are long compared to the baseline window (the case where the baseline
    estimator itself starts to ride into the event).

    DO NOT reach for it on a SHALLOW event in a WIDE-BAND trace, and note that this
    is the opposite way round from the intuition. The test is on the derivative's own
    noise, which for white noise is sqrt(2)*sigma*fs (measured: 352,973 pA/s against a
    predicted 363,965 for sigma = 5.147 pA at 50 kHz), while a one-sample step of depth
    D produces D*fs -- so the edge is only D/(sqrt(2)*sigma) sigma tall no matter how
    long the event is. A 6-sigma-deep event is a 4.2-sigma EDGE and falls under k = 5.
    [MEASURED] 40 events of 30 pA (6 sigma) in a 5 pA-noise trace: recall 0.225
    unfiltered, and 1.000 through a 4-pole 5 kHz Bessel -- LOW-PASSING RESCUES THIS
    DETECTOR, because it removes the noise's high-frequency power faster than it softens
    the edge. At 20 pA (4 sigma) it is 0.000 unfiltered, 0.750 at 5 kHz, 1.000 at 2 kHz.
    The Schmitt trigger found 40/40 in every one of those cases; when the LEVEL is
    trustworthy, use the level. Also: an event that starts before the trace or ends
    after it has one edge, and an unpaired edge is dropped, not truncated.

    PRECISION/RECALL, measured (fs = 50 kHz, 5 pA noise, 40 events of 100 pA):
    1.000 / 1.000 on 1 ms squares AND on 3-sample spikes; unchanged at 1.000 / 1.000
    when a 40 pA/s ramp is added to the same trace (where `detect_fixed_threshold`
    drops to 0.875 and, on a downward ramp, to 0.400). On a square pulse occupying
    [500, 600) in 0.01 pA of noise it returns exactly ``(500, 600)``. (0.01 pA, not
    zero: the threshold is k times the derivative's own noise scale, and
    `detection_basis` refuses a trace whose MAD is exactly 0 before this kernel ever
    gets to measure one.)

    [MEASURED] ON THE OWNER'S OWN RECORDING -- THIS KERNEL NEEDS TUNING AND MAY NOT
    REACH A USABLE POINT. 15 s apo, 749,651 samples; reference = `detect_events` at
    its defaults, 115 events, median dwell 20.1 ms, median depth 90.3 pA, and a
    MEDIAN RISE TIME OF 1.01 ms (over the 88 events where `rise_s` is defined). At the shipped defaults: 146 spans, recall 0.678, precision
    1.000, median span 4.49 ms -- and one span of 412.9 ms starting at t = 12.725 s,
    a rising edge that found no falling edge for 20,643 samples. The cause is the
    failure mode two paragraphs up, on real data: a 90 pA step spread over a 1.01 ms
    rise is ~64,000 pA/s against a derivative noise scale of ~405,000 pA/s, so the
    EVENT edges are far under k = 5 and what clears the bar is the fast sub-structure
    INSIDE the events (which is why precision is 1.000 while recall is not).
    What tuning buys, all measured on that file:
      * ``max_pair_dwell_s=50e-3`` (above the longest real dwell): longest span
        28.7 ms, but recall falls 0.678 -> 0.557 -- refusing the over-long pair
        deletes the only span covering the events it swallowed.
      * low-pass first (`filters.lowpass`), the rescue that works on the synthetic:
        2 kHz gives recall 0.791 at precision 0.998, and still a 206.9 ms span.
      * ``k=3`` with the cap and ``merge_gap_s=5e-3``: recall 0.930, PRECISION 0.362.
    No combination tried reached high recall AND a bounded longest span here. On a
    recording of long events with soft edges, use `detect_events`; reach for this
    kernel when the edges are genuinely sharp and the level is genuinely untrustworthy.

    Args:
        signal: Current trace, 1-D.
        fs: Sample rate, Hz.
        basis: An already-measured `DetectionBasis`, or ``None`` to measure one.
        polarity, baseline, baseline_config, noise_method: forwarded to
            `detection_basis` when `basis` is None. Polarity matters here: the
            derivative of ``dev`` is thresholded SIGNED, so a rising edge means
            "into an event" and a falling edge "out of one". Get the polarity
            backwards and every event is paired inside-out.
        k: Threshold in units of the derivative's own noise scale, measured on
            ``diff(dev)*fs`` by `noise_method`. 5 is the same bar `detect_events`
            sets on the level.
        threshold_per_s: Absolute edge threshold in units/second, overriding `k`.
            Use it when you know your slew limit (e.g. a 10 kHz filter cannot move
            faster than ~2.2e5 pA/s for a 10 pA step).
        max_pair_dwell_s: Refuse a pairing longer than this. This is NOT
            `detect_events`' ``max_dwell_s`` (which KEEPS the event and flags it):
            here an over-long pair is not a long event, it is a MISSED falling edge
            that has glued one event to the exit of a later one, so the honest
            outcome is to emit nothing rather than a span that is mostly open pore.
            ``None`` = no limit.
        min_dwell_s: Drop events shorter than this (0 = no floor).
        merge_gap_s: Merge events closer than this (shared gap rule + occupancy brake).

    Returns:
        ``(starts, ends)``: half-open sample indices, int64. The bounds are the
        edges themselves: a rising edge occupying difference samples [a, b) means
        the trace had arrived at the event level by sample b, so ``start = b``;
        a falling edge starting at difference sample c means sample c is the last
        one inside, so ``end = c + 1``. On a clean square pulse (0.01 pA of noise)
        those are EXACT -- the test suite pins that -- which is what makes this
        kernel's dwell comparable with the Schmitt trigger's when it pairs the
        edges correctly. See the owner's-recording block above for what happens
        when it does not.
    """
    fs = float(fs)
    b = basis if basis is not None else detection_basis(
        signal, fs, polarity=polarity, baseline=baseline,
        baseline_config=baseline_config, noise_method=noise_method,
    )
    dev = b.dev
    n = dev.size
    if n < 3:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if threshold_per_s is None and not (k > 0):
        raise ValueError(f"k must be > 0, got {k}")

    d = np.diff(dev) * fs
    if threshold_per_s is not None:
        thr = float(threshold_per_s)
        if not (thr > 0):
            raise ValueError(f"threshold_per_s must be > 0, got {threshold_per_s}")
    else:
        scale = noise_level(d, b.noise_method)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "derivative noise scale is zero; a perfectly smooth trace has no "
                "edges to threshold."
            )
        thr = float(k) * scale

    rise_s, rise_e = _runs(d > thr)
    fall_s, _fall_e = _runs(d < -thr)
    if rise_s.size == 0 or fall_s.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    starts = rise_e.astype(np.int64)            # first sample at the event level
    ends_all = (fall_s + 1).astype(np.int64)    # one past the last sample inside

    # Pair each rising edge with the FIRST falling edge after it. Several rising
    # edges before one falling edge (a noisy or multi-step entry) collapse to the
    # earliest, which is the conservative reading: the event began at the first
    # unambiguous move away from baseline.
    j = np.searchsorted(ends_all, starts, side="right")
    ok = j < ends_all.size
    starts, j = starts[ok], j[ok]
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    _uniq, first = np.unique(j, return_index=True)
    starts, ends = starts[first], ends_all[_uniq]

    if max_pair_dwell_s is not None:
        keep = (ends - starts) / fs <= float(max_pair_dwell_s)
        starts, ends = starts[keep], ends[keep]
    return _gate_intervals(
        starts, ends, fs, min_dwell_s=min_dwell_s, merge_gap_s=merge_gap_s
    )


def tkeo_energy(x: np.ndarray) -> np.ndarray:
    """Teager-Kaiser energy ``psi[n] = x[n]^2 - x[n-1]*x[n+1]``, same length as `x`.

    For a single sinusoid this is A^2*sin^2(w) -- amplitude AND frequency at once,
    from three samples and two multiplies, with no window and therefore no delay.
    That is the whole reason it is here: every other operator in this module needs a
    run of samples before it can say anything, and psi answers on sample three.

    The two edge samples have no neighbours and are set to 0 rather than being
    dropped, so the returned array is index-aligned with `x` and a run found in psi
    indexes straight back into the trace.

    psi is NOT positive-definite on real data (it goes negative where the trace
    curves the other way), and it is in the SQUARE of the input's units -- a
    threshold on psi is not a threshold in pA and must not be shown as one.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {a.shape}")
    psi = np.zeros(a.size, dtype=np.float64)
    if a.size >= 3:
        psi[1:-1] = a[1:-1] * a[1:-1] - a[:-2] * a[2:]
    return psi


def detect_tkeo(
    signal: np.ndarray,
    fs: float,
    *,
    basis: DetectionBasis | None = None,
    polarity: int | None = None,
    baseline: np.ndarray | float | None = None,
    baseline_config: BaselineConfig | None = None,
    noise_method: str = "mad",
    k: float = 20.0,
    link_gap_s: float | None = None,
    polarity_gate: bool = True,
    min_dwell_s: float = 0.0,
    merge_gap_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Events as bursts of Teager-Kaiser energy -- amplitude AND rate of change.

    REACH FOR THIS for the SHORTEST events: bursts of one to a few samples, the ones
    a level detector loses because its exit threshold and its two-sample coherence
    test both need a run of samples that a 40 us event at 50 kHz does not have. psi
    needs three samples, total, and it answers to how FAST the trace moved as well as
    how far, so a fast small event outscores a slow large wander -- which is the
    discrimination a level threshold cannot make at all.

    DO NOT reach for it for ordinary square events without setting `link_gap_s`.
    psi is an EDGE operator on anything flat-topped: a 1 ms square event lights up at
    its entry and again at its exit and is flat in between, so it comes back as at
    least TWO detections per event unless `link_gap_s` bridges them. [MEASURED] on the
    standard 40-event synthetic at the default k = 20: 626 spans, precision 0.064,
    recall 1.000; with ``link_gap_s=2e-3`` (just above the 1 ms dwell) exactly 40 spans
    at 1.000 / 1.000. Even a 3-SAMPLE event has two edges: 69 spans / precision 0.580
    unlinked, 40 spans / 1.000 with ``link_gap_s=1e-4``.
    And note psi(-x) == psi(x) EXACTLY -- squaring is sign-blind -- so TKEO cannot by
    itself tell a blockade from an enhancement or from a positive-going artifact;
    `polarity_gate` is what re-imposes the app's polarity rule on its output.

    PRECISION/RECALL, measured (fs = 50 kHz, 5 pA noise, 40 events of 100 pA):
    SINGLE-SAMPLE spikes 1.000 / 1.000 at the default k = 20 -- this is the shape the
    kernel is for, and the one where it needs no linking at all. 3-sample spikes
    1.000 / 1.000 with ``link_gap_s=1e-4``; 1 ms squares 1.000 / 1.000 with
    ``link_gap_s=2e-3``. The default k = 20 is measured, not conventional: at k = 5
    (the bar a LEVEL detector uses) the single-sample case falls to precision 0.053
    -- 750 spans for 40 events -- because psi's distribution is heavy-tailed even for
    Gaussian noise. Everything from k = 20 to k = 200 gives 1.000 / 1.000 there, so
    20 is the loose end of a wide plateau rather than a tuned number.

    [MEASURED] ON THE OWNER'S OWN RECORDING, and it is the flat-topped warning above
    playing out on real data. 15 s apo, 749,651 samples; reference = `detect_events`
    at its defaults, 115 events, median dwell 20.1 ms. At the defaults: 14,006 spans
    -- 149 per reference event -- recall 0.817, precision 0.998, median span 0.02 ms
    (one sample). Every one of those is a real edge; none of them is an event. With
    ``link_gap_s=2e-3``: 108 spans, recall 0.826, precision 0.796, median span
    19.6 ms against the app's 20.1 ms, which IS dwell-comparable. SET `link_gap_s`
    OR DO NOT USE THIS KERNEL ON EVENTS THIS LONG. Note also that `polarity_gate`
    removes nothing at all on this file (14,006 spans either way): psi's threshold
    already selects the fast moves, and on this recording they all lie on the
    blockade side. The gate is insurance, and on clean data it is free.

    Args:
        signal: Current trace, 1-D.
        fs: Sample rate, Hz.
        basis: An already-measured `DetectionBasis`, or ``None`` to measure one.
        polarity, baseline, baseline_config, noise_method: forwarded to
            `detection_basis` when `basis` is None.
        k: Threshold in units of psi's own robust scale, above psi's median:
            ``thr = median(psi) + k*scale(psi)``. The median is not zero -- noise has
            energy -- so the offset is part of the threshold, not a nicety. psi's
            distribution is heavy-tailed even for Gaussian noise, so k here is NOT
            interchangeable with the k of a level detector: 5 on psi is a much looser
            test than 5 sigma on the level, and the measurements above are the guide.
        link_gap_s: Bridge bursts separated by less than this into one event, for the
            flat-topped case above. Set it a little ABOVE your longest expected dwell
            and well BELOW your shortest inter-event gap. This deliberately does NOT
            go through the occupancy brake that `merge_gap_s` uses: two 3-sample edge
            bursts spanning a 1 ms event occupy 6/50 = 12% of the span, so the brake
            would refuse every correct link. That is also the hazard -- an oversized
            `link_gap_s` glues real neighbours together with nothing to stop it, which
            is exactly the pathology the brake exists for. ``None`` = no linking.
        polarity_gate: Keep only bursts whose mean ``dev`` is positive, i.e. on the
            side of the baseline the app's polarity rule says events are. It is the
            only thing standing between a sign-blind operator and a table of
            wrong-direction artifacts reported as events. [MEASURED] on the 3-sample
            spike trace contaminated with 20 enhancement-direction spikes of the same
            amplitude: 100 spans and precision 0.400 with the gate off, 69 spans and
            precision 0.580 with it on -- it removes 31 of the 40 wrong-sign spans and
            costs no recall. Turn it off only when you are deliberately hunting
            excursions in both directions.
        min_dwell_s: Drop events shorter than this. Default 0, and think twice: the
            events this kernel exists to find are the ones a dwell floor removes.
        merge_gap_s: Merge events closer than this (shared gap rule + occupancy brake).

    Returns:
        ``(starts, ends)``: half-open sample indices, int64.
    """
    fs = float(fs)
    b = basis if basis is not None else detection_basis(
        signal, fs, polarity=polarity, baseline=baseline,
        baseline_config=baseline_config, noise_method=noise_method,
    )
    dev = b.dev
    n = dev.size
    if n < 3:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if not (k > 0):
        raise ValueError(f"k must be > 0, got {k}")

    psi = tkeo_energy(dev)
    core = psi[1:-1]
    scale = noise_level(core, b.noise_method)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Teager-Kaiser energy has no measurable scale; the trace is constant."
        )
    thr = float(np.median(core)) + float(k) * scale
    starts, ends = _runs(psi > thr)
    if starts.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = starts.astype(np.int64)
    ends = ends.astype(np.int64)

    if link_gap_s is not None and starts.size > 1:
        if link_gap_s < 0:
            raise ValueError(f"link_gap_s must be >= 0, got {link_gap_s}")
        is_group_start = group_starts_by_gap(
            starts, ends, gap_window_samples(link_gap_s, fs)
        )
        is_group_end = np.concatenate((is_group_start[1:], [True]))
        starts, ends = starts[is_group_start], ends[is_group_end]

    if polarity_gate and starts.size:
        # np.add.reduceat over the interleaved run bounds: ONE pass over `dev`, no
        # Python loop per event. The bounds interleave strictly increasing (a run has
        # at least one sample, and _runs leaves at least one sub-threshold sample
        # between runs, a property `link_gap_s` grouping preserves because it only
        # drops interior bounds), which is what reduceat requires; `dev` is padded by
        # one slot so a final ``end == n`` is a legal index. Even elements are the
        # in-run sums, odd elements the gaps between them, so [::2] is the answer.
        # [MEASURED] at the default k on the owner's 15 s apo recording (749,651
        # samples) the gate faces 14,006 runs: the list comprehension this replaced
        # took 25.8 ms, the reduceat takes 0.51 ms, and the whole `detect_tkeo` call
        # goes 58.2 ms -> 33.2 ms (best of 5). The spans returned are bit-identical.
        bounds = np.empty(2 * starts.size, dtype=np.intp)
        bounds[0::2] = starts
        bounds[1::2] = ends
        padded = np.empty(dev.size + 1, dtype=np.float64)
        padded[: dev.size] = dev
        padded[dev.size] = 0.0
        sums = np.add.reduceat(padded, bounds)[::2]
        keep = sums > 0.0                      # mean > 0 <=> sum > 0, run length > 0
        starts, ends = starts[keep], ends[keep]

    return _gate_intervals(
        starts, ends, fs, min_dwell_s=min_dwell_s, merge_gap_s=merge_gap_s
    )
