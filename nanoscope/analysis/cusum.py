"""CUSUM level fitting: resolve the sub-states INSIDE an already-detected event.

WHAT THIS IS FOR. ``characterize`` reports ``n_levels``, an integer from a BIC-stopped
binary segmentation, and its own docstring tells you to read it as "go look at this
event", not as a measurement. That is the honest limit of a count: a folded-vs-unfolded
protein, or a molecule that pauses partway through the pore, produces a STEPPED
blockade whose interpretation lives in the amplitude and duration of each step -- and a
count throws exactly that away. This module produces the steps.

WHY CUSUM AND NOT MORE BINARY SEGMENTATION. ``characterize._segment_count`` stops on
BIC, and BIC assumes independent samples. A nanopore trace is low-passed before
detection (the app's own detector takes ``filter_cutoff_hz``), so its samples are
autocorrelated and the BIC term is computed against an effective N it does not have.
The result over-splits -- flagged in that function's docstring, and the reason its cap
``_MAX_LEVELS`` sits at 4.

CUSUM CARRIES THAT ASSUMPTION TOO -- this module claimed otherwise, and was wrong. Page's
statistic normalises by ``sigma``, the spread of ONE INDEPENDENT sample, so on a
low-passed trace it over-fires for the same reason BIC over-splits. Uncorrected it
reported 11.27 levels per event on 200 PERFECTLY RECTANGULAR single-level events. What
makes the method usable here is not the statistic but the correction: every significance
test is scaled by ``sqrt(tau)`` for the autocorrelation the detector's own low-pass
imposes (:func:`filter_autocorr_time`), and WHERE a boundary sits is settled by the batch
passes rather than sequentially. What survives of the advantage over BIC is that
sensitivity is an explicit (step size, threshold) pair in units of the noise you can see,
rather than a penalty computed against an effective N the data does not have.

It is also the method the reference implementation in this field uses (Raillon et al.,
Nanoscale 4:4916, 2012 -- OpenNanopore).

THE COUNT IS NOT REPLACED. ``events.SCHEMA['n_levels']`` keeps meaning what it has
always meant, measured the way it has always been measured. A CUSUM fit is a SEPARATE,
on-demand result with its own parameters and its own per-event level count
(``LEVEL_SCHEMA['n_levels']``), because silently re-measuring a column under a user
who has already curated against it is the one thing this codebase refuses to do
everywhere else (see ``AnalysisResult.baseline_fingerprint``). Where the two disagree,
that IS information: BIC over-splitting on a filtered trace is the expected direction.

SIGN CONVENTION. Identical to ``characterize`` and ``events``: the fit runs on
``dev = polarity * (current - baseline)``, which is POSITIVE inside an event. So a
deeper sub-state has a LARGER ``depth_pA``, whichever way the physical current went.

NOISE SCALE. ``sigma`` is the detector's own ``provenance['noise_rms_pA']`` when the
caller has it -- the same number every ``snr`` in the event table was divided by, so a
threshold expressed in sigma here means what it means there. See
``characterize.span_noise_rms`` for why that identity matters more than a locally
better estimate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import blake2s
from typing import Any

import numpy as np
import pandas as pd

from .cancel import CancelFlag, CancelPoller
from .filters import noise_rms as _noise_rms

__all__ = [
    "CUSUM_VERSION",
    "CusumConfig",
    "DEFAULT_CUSUM",
    "LEVEL_SCHEMA",
    "empty_levels",
    "cusum_changepoints",
    "fit_levels_1d",
    "fit_event_levels",
    "reconcile_levels",
    "integrated_autocorr_time",
    "open_pore_autocorr_time",
    "filter_autocorr_time",
]

#: Bumped when a change would move a fitted level. Lands in the result's params so a
#: stored fit can name the code that produced it.
#:
#: 2 -- WHERE a boundary goes is no longer decided sequentially, and the significance
#: scale is autocorrelation-corrected. v1 was CUSUM plus a per-boundary prune; v2 adds
#: the rescue / refine / drop-unearned batch passes, filter-ramp trimming, and the
#: ``sqrt(tau)`` correction (see :func:`filter_autocorr_time`). Every one of those moves
#: fitted levels, and by a lot: on 200 PERFECTLY RECTANGULAR single-level events v1
#: reported 11.27 levels each; v2 reports 1.00. A stored v1 level table is therefore not
#: comparable with a v2 one and must not be pooled with it.
CUSUM_VERSION = "2"

#: An event shorter than this has no resolvable internal structure; the same floor
#: ``characterize`` applies before it will attempt a sub-state count at all.
#:
#: PUBLIC, because a caller has to be able to ask BEFORE calling. The single-event re-fit
#: behind a mask edit (``MainWindow._refit_levels_for_event``) declines a span below this
#: rather than fitting it: the fit does not fail on a short span, it returns one "level"
#: spanning the whole event, and splicing that into the table would put a row into the
#: export that says nothing and that a re-run of the whole fit would not produce.
MIN_SAMPLES_FOR_LEVELS = 10
#: The name this module used before the floor had a caller outside it. Kept so the two
#: cannot drift; new code should read the public one.
_MIN_SAMPLES_FOR_LEVELS = MIN_SAMPLES_FOR_LEVELS


@dataclass(frozen=True)
class CusumConfig:
    """The three numbers that decide what counts as a level, plus two guards.

    ``step_sigma`` and ``threshold_sigma`` are both in units of the baseline noise RMS,
    deliberately: that is the unit ``detect_events`` already states its ``k_in`` /
    ``k_out`` thresholds in, so a user who has tuned one has tuned the other.
    """

    #: Smallest step the fit is asked to resolve, in noise RMS. This is CUSUM's
    #: ``delta``: the statistic is biased by ``step_sigma/2`` per sample, so a run of
    #: samples shifted by less than this drifts the accumulator DOWN and never fires.
    #: Below ~1 the fit starts resolving the noise; the default is deliberately not
    #: heroic.
    step_sigma: float = 2.0

    #: Decision threshold ``h`` on the accumulated statistic. Higher = fewer, more
    #: certain change points.
    #:
    #: 6.0 is measured, not taken from the literature. 3000 trials of a 400-sample
    #: pure-noise segment (which must NOT be split) against 1000 trials each of a
    #: single real step, at ``step_sigma=2`` and the default warm-up:
    #:
    #: ====  ===========  ============  ============
    #:  h    false split  find 2-sigma  find 3-sigma
    #: ====  ===========  ============  ============
    #: 4.0        0.85%          99.7%         99.6%
    #: 5.0        0.00%         100.0%        100.0%
    #: 6.0        0.00%         100.0%        100.0%
    #: 8.0        0.00%         100.0%        100.0%
    #: ====  ===========  ============  ============
    #:
    #: This threshold no longer carries the false-positive burden on its own -- it also
    #: serves as ``z_min`` for the three batch passes, so a change point it lets through
    #: must still be supported by :func:`_prune_boundaries` and its segment must still be
    #: earned against :func:`_drop_unearned_segments`. 6 is one step past where the
    #: false-split column reaches zero, and detection has not begun to cost anything.
    threshold_sigma: float = 6.0

    #: A level shorter than this is not reported as a level. Guards both against
    #: CUSUM's own detection delay producing a sliver at a transition and against
    #: reading the filter's rise time as a state.
    min_level_samples: int = 3

    #: How many samples seed a new segment's mean before the detector is armed.
    #: See :func:`cusum_changepoints` for why this is a median over a window rather
    #: than a single sample. The size trades two failures against each other: too
    #: SHORT and the window straddles a transition, seeding mu between two levels and
    #: emitting a sliver; too LONG and the detector cannot arm inside a genuinely short
    #: level. This is an UPPER BOUND -- :func:`cusum_changepoints` caps it at a third of
    #: the samples left in the event -- which is what keeps the second failure from
    #: mattering.
    #:
    #: THIS CONSTANT NO LONGER DECIDES ANYTHING, and that is the point. It used to: a
    #: flat window blinded the sequential pass to any level shorter than itself, so the
    #: reported multilevel fraction on the owner's -600 mV apo-ferritin recording swung
    #: 30.1% -> 21.0% across 5..16 purely on this number. Now that the batch passes
    #: (rescue / refine / drop-unearned) decide WHERE boundaries go, the same sweep is
    #: flat: 8.8% at every one of 5, 8, 12, 16, 24 (RE-MEASURED after the autocorrelation
    #: correction -- it read 31.9% before that landed, which is why the level COUNT is
    #: still tau's to answer for and not this constant's). The synthetic sweep is flat
    #: too: 0.0% wrong count and 0 median boundary error at every value from 3 to 16,
    #: over 400 trials per cell.
    #:
    #: So 12 is now only a seeding detail: it is how many samples the median gets to
    #: reject a straddling sample with, and it is chosen to be comfortably robust while
    #: staying below the length of a typical event. A reviewer looking for the constant
    #: that determines this module's answers will not find it here -- it is
    #: ``step_sigma`` and ``threshold_sigma``.
    warmup_samples: int = 12

    #: Runaway guard, not a modelling statement. Unlike ``characterize._MAX_LEVELS``
    #: (which caps at 4 because BIC over-splits) this is set high enough not to be
    #: the thing that decides your answer.
    max_levels: int = 32

    def __post_init__(self) -> None:
        if not (np.isfinite(self.step_sigma) and self.step_sigma > 0):
            raise ValueError(f"step_sigma must be finite and > 0, got {self.step_sigma!r}")
        if not (np.isfinite(self.threshold_sigma) and self.threshold_sigma > 0):
            raise ValueError(
                f"threshold_sigma must be finite and > 0, got {self.threshold_sigma!r}")
        if self.min_level_samples < 1:
            raise ValueError(
                f"min_level_samples must be >= 1, got {self.min_level_samples!r}")
        if self.warmup_samples < 1:
            raise ValueError(
                f"warmup_samples must be >= 1, got {self.warmup_samples!r}")
        if self.max_levels < 1:
            raise ValueError(f"max_levels must be >= 1, got {self.max_levels!r}")

    def fingerprint(self) -> str:
        """Digest over every field that can move a fitted level.

        Same role as ``BaselineConfig.fingerprint``: a stored level table can be asked
        whether the parameters now in the UI are the ones that produced it, without
        keeping the parameters themselves in every row.
        """
        raw = (f"{CUSUM_VERSION}|{self.step_sigma!r}|{self.threshold_sigma!r}|"
               f"{self.min_level_samples!r}|{self.warmup_samples!r}|{self.max_levels!r}")
        return blake2s(raw.encode(), digest_size=8).hexdigest()


DEFAULT_CUSUM = CusumConfig()


#: Column -> (dtype, unit, description). One row per (event, level).
#:
#: LONG, not wide. The obvious alternative -- widening ``events.SCHEMA`` with an
#: (amplitude, dwell) pair per sub-state -- is closed off on purpose: that schema is
#: frozen at 20 columns (``write_events_export`` promises "no new column, no
#: reordering" to every consumer outside this app, and ``core.history._MEASURED``
#: hardcodes the list), and a per-substate widening makes the column COUNT depend on
#: the data, which every one of those consumers would have to learn. Long form also
#: makes the question this feature exists to answer a groupby rather than a reshape:
#: "the second level of every 3-level event".
LEVEL_SCHEMA: dict[str, tuple[str, str, str]] = {
    "event_id":   ("int64",   "-",      "event this level belongs to; joins events.SCHEMA"),
    "recording":  ("string",  "-",      "name of the source Recording"),
    "pore_id":    ("string",  "-",      "which physical pore produced this; never pool across pores"),
    "level_idx":  ("int64",   "-",      "0-based position of this level within its event"),
    "n_levels":   ("int64",   "-",      "levels this fit resolved in the parent event"),
    "start_idx":  ("int64",   "sample", "first sample of the level (absolute, as events.SCHEMA)"),
    "end_idx":    ("int64",   "sample", "one past the last sample"),
    "t_start_s":  ("float64", "s",      "level start, seconds from recording start"),
    "dwell_s":    ("float64", "s",      "level duration, (end_idx-start_idx)/fs"),
    "level_pA":   ("float64", "pA",     "mean ABSOLUTE current of the level"),
    "depth_pA":   ("float64", "pA",     "median excursion from the event baseline, positive"),
    "frac_depth": ("float64", "-",      "depth_pA / |baseline_pA| of the parent event"),
    "step_pA":    ("float64", "pA",     "signed depth change from the previous level; NaN at level_idx 0"),
}

_LEVEL_FLOAT_COLUMNS = tuple(c for c, (d, _, _) in LEVEL_SCHEMA.items() if d == "float64")

# ------------------------------------------------------------------------------------------------
# What the fit RAN WITH, carried out on ``levels.attrs``
# ------------------------------------------------------------------------------------------------
# The two numbers that decide where a boundary lands -- the noise scale and the
# autocorrelation time -- are RESOLVED inside :func:`fit_event_levels` from a ladder of
# fallbacks (provenance, then a measurement of the trace), so a caller that passed
# ``sigma=None, tau=None`` does not know what it got. That is fine for one fit and fatal
# for a SECOND one: re-fitting a single edited event over its new span has to use the same
# estimator as the fit it is splicing into, or two neighbouring events in one table are
# segmented at two different significance scales and nothing on screen says so.
#
# ``tau`` is the expensive half. Measured on the owner's 2M-sample recording, the
# segmentation of one 600-sample event is ~0.33 ms while the whole-recording
# autocorrelation estimate behind it is a fixed ~ per-call cost over every event's
# residual -- which is why re-fitting one event pays for the whole table unless the number
# is handed back in.
#
# They travel on ``.attrs`` rather than in a return tuple because the return type is the
# frozen LEVEL_SCHEMA frame and 35 call sites unpack it as one value; ``recharacterize_span``
# already stamps ``df.attrs['baseline_source']`` for the same reason. Set on the object
# actually returned, after every astype/sort, so pandas' ``__finalize__`` cannot lose them.

#: ``levels.attrs`` key: the noise RMS (in the channel's units) the significance tests used.
SIGMA_ATTR = "cusum_sigma_pA"
#: ``levels.attrs`` key: the autocorrelation time, in SAMPLES, the z-tests were discounted
#: by. ``nan`` when the fit returned before resolving one (an empty event set).
TAU_ATTR = "cusum_tau_samples"


def _stamp_estimators(df: pd.DataFrame, sigma: float, tau: float | None) -> pd.DataFrame:
    """Record what the fit ran with on ``df.attrs``. See the block above."""
    df.attrs[SIGMA_ATTR] = float(sigma)
    df.attrs[TAU_ATTR] = float("nan") if tau is None else float(tau)
    return df


def empty_levels() -> pd.DataFrame:
    """An empty level table with the right columns and dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=d) for c, (d, _, _) in LEVEL_SCHEMA.items()})


def integrated_autocorr_time(
    residual: np.ndarray, *, n_scale: int = 64, max_lag: int = 512
) -> float:
    """Variance-inflation factor for a MEAN over ``n_scale`` correlated samples.

    ``tau = 1 + 2*sum_k (1 - k/n_scale) * rho_k``, the Bartlett-weighted form, so that
    ``Var(mean) = tau * sigma^2 / n``. Every significance test in this module divides a
    mean difference by ``sigma*sqrt(1/nl + 1/nr)``, which assumes independent samples;
    the trace it is handed is LOW-PASSED by construction (``modules.detection_signal``
    rebuilds what ``characterize`` measured on), so that assumption is false and every z
    was inflated. Uncorrected, 200 PERFECTLY RECTANGULAR single-level events came back
    frac_multilevel 1.0, 11.27 levels each.

    WHY WEIGHTED AND TRUNCATED AT ``n_scale``, not summed to the first non-positive rho.
    On the owner's real recordings the ACF never becomes non-positive at all -- measured
    rho = 0.977 at lag 1 and still 0.479 at lag 40 (0.8 ms) -- because the residual
    carries slow DRIFT the baseline estimator did not remove. An unweighted sum then has
    no natural stopping point and tau simply tracks whatever horizon it is given: 10, 18,
    31, 53, 85, 141 at max_lag 5, 10, 20, 40, 80, 200. A number set by its own cutoff is
    not a measurement.
    ``n_scale`` fixes that by asking the right question. The tests here compare two
    segments INSIDE one event, so correlation at lags longer than those segments is a
    shared offset that largely cancels in their difference -- which is exactly what the
    ``(1 - k/n)`` weight encodes. Pass the scale you are actually inferring at.

    Returns 1.0 (no correction) when the input is too short or degenerate, never
    over-claiming.
    """
    r = np.asarray(residual, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 32:
        return 1.0
    r = r - r.mean()
    var = float(r.dot(r) / n)
    if not (var > 0):
        return 1.0
    horizon = int(max(1, min(int(n_scale) - 1, max_lag, n // 4)))
    total = 0.0
    for k in range(1, horizon + 1):
        rho = float(r[:-k].dot(r[k:]) / ((n - k) * var))
        total += (1.0 - k / float(n_scale)) * rho
    return max(1.0, 1.0 + 2.0 * total)


def filter_autocorr_time(
    fs: float, cutoff_hz: float, *, order: int = 8, kind: str = "bessel",
    n_scale: int = 64, n: int = 20_000, seed: int = 0,
) -> float:
    """``tau`` of the DETECTOR'S OWN LOW-PASS, measured on white noise pushed through it.

    This is the honest source for the correction, and the previous two were not.
    Estimating tau from ``signal - baseline`` measures whatever happens to be in that
    residual, and on a real recording that is the filter PLUS drift PLUS the baseline
    estimator's own tracking error. Excluding events helped and was still not enough: on a
    CLEAN SYNTHETIC trace whose gaps contain nothing but filtered noise the residual
    estimate read 67 where the filter alone justifies ~6, and at 67 a genuine 3-level
    event collapses to 1.

    The correlation the z-tests must correct for is the one the FILTER imposes -- that is
    the entire reason neighbouring samples are not independent. So measure exactly that,
    and nothing else: white noise through the same ``filters.lowpass`` the detection ran,
    with the kind, order and cutoff taken from its own provenance. Drift cannot enter,
    because there is none; baseline error cannot enter, because no baseline is involved.

    One 20k-sample filter pass, and deterministic given ``seed``, so the same detection
    always produces the same tau. Returns 1.0 (no correction) for a cutoff that is not a
    usable fraction of Nyquist, which is also the "no low-pass ran" case.
    """
    from .filters import lowpass

    if not (np.isfinite(fs) and fs > 0 and np.isfinite(cutoff_hz) and 0 < cutoff_hz < fs / 2):
        return 1.0
    x = np.random.default_rng(seed).standard_normal(int(n))
    try:
        y = np.asarray(lowpass(x, float(fs), float(cutoff_hz),
                               order=int(order), kind=str(kind)), dtype=np.float64)
    except Exception:      # pragma: no cover - a bad filter spec must not break the fit
        return 1.0
    return integrated_autocorr_time(y, n_scale=n_scale)


def open_pore_autocorr_time(
    residual: np.ndarray, starts: np.ndarray, ends: np.ndarray, *,
    n_scale: int = 64, pad: int = 0,
) -> float:
    """``tau`` measured on OPEN PORE only -- the gaps between detected events.

    Measured over the whole residual, tau is contaminated by the events themselves: a
    blockade is a large, sustained excursion, so it raises the ACF at every lag up to its
    own duration. That is not the noise the significance tests are asking about -- they
    ask whether two segments INSIDE one event differ, given the noise -- and including
    events inflates tau, which suppresses real sub-states. Measured on the owner's two apo
    files, the whole-residual estimate came out 8x and 12x the high-passed open-pore one
    (90.5 vs 11.3 and 57.1 vs 4.8), and that gap is the whole reason a genuine 3-level
    event was reported as 1.

    PER GAP, then a MEDIAN -- never one ACF over the concatenation. Splicing disjoint
    stretches end-to-end manufactures a discontinuity at every join, and an ACF cannot
    tell that from real structure. The median across gaps is also robust to a gap that
    quietly contains an event the detector missed, which a mean is not.

    Falls back to the whole-residual estimate when no gap is long enough to measure --
    honest degradation rather than a fabricated number.
    """
    r = np.asarray(residual, dtype=np.float64)
    n = r.size
    a = np.asarray(starts, dtype=np.int64)
    b = np.asarray(ends, dtype=np.int64)
    # HIGH-PASS at the inference scale before measuring. Every test in this module compares
    # two segments INSIDE one event, so noise slower than an event shifts BOTH halves
    # together and cancels in their difference -- but it dominates the raw ACF, which is why
    # the residual estimate read 90 and 57 on the owner's two apo files where the same
    # traces high-passed read 11.3 and 4.8. Subtracting a moving average of the inference
    # length removes exactly the components that cannot affect the comparison.
    if n > 8:
        w = max(3, int(n_scale) | 1)
        if w < n:
            r = r - np.convolve(r, np.ones(w) / w, mode="same")
    if a.size == 0:
        return integrated_autocorr_time(r, n_scale=n_scale)
    order = np.argsort(a)
    a, b = a[order], b[order]

    # An ACF needs many multiples of the scale it is estimating; below that the estimate
    # is noise and would drag the median around.
    need = max(64, 4 * int(n_scale))
    taus: list[float] = []
    edges = [0, *((int(x) for pair in zip(b + pad, a[1:] - pad) for x in pair)), n]
    for lo, hi in zip(edges[0::2], edges[1::2]):
        lo, hi = max(int(lo), 0), min(int(hi), n)
        if hi - lo >= need:
            taus.append(integrated_autocorr_time(r[lo:hi], n_scale=n_scale))
    if not taus:
        return integrated_autocorr_time(r, n_scale=n_scale)
    return float(np.median(taus))


# ================================================================================================
# The algorithm, on one 1-D array. No app types.
# ================================================================================================
def cusum_changepoints(
    x: np.ndarray,
    *,
    sigma: float,
    config: CusumConfig = DEFAULT_CUSUM,
    tau: float = 1.0,
) -> np.ndarray:
    """Indices where the mean of ``x`` steps, as a sorted int64 array.

    Two-sided Page's CUSUM with a recursively estimated segment mean. For each sample
    the normalised residual ``s = (x[i] - mu) / sigma`` feeds two accumulators::

        g+ = max(0, g+ + s - step_sigma/2)      # fires on an UPWARD step
        g- = max(0, g- - s - step_sigma/2)      # fires on a DOWNWARD step

    Either crossing ``threshold_sigma`` is a change. The ``step_sigma/2`` subtraction is
    what makes this a detector rather than a random walk: with no step present the
    residual averages 0, so both accumulators are dragged to their ``max(0, ...)`` floor
    and stay there indefinitely. Only a sustained shift outruns the drag.

    The reported index is NOT where the accumulator crossed. CUSUM fires ``~h /
    (step_sigma/2)`` samples late by construction -- it has to accumulate evidence
    before it is entitled to an opinion -- so the change point is recovered by
    backtracking to the last sample at which the firing accumulator was still zero.
    That is the standard estimator and it is worth the bookkeeping: with the correction
    the median boundary error on a clean 2-sigma step is 0 samples, while the raw
    crossing index would sit ~6 samples late every time.

    ``mu`` is the running mean of the CURRENT segment, reset at every change point and
    SEEDED from a median (see the nested ``_arm``). It adapts fast when a segment is
    young (1/k) and slowly once established, which is what stops a single step from
    firing twice on its own settling.

    So the returned array is the output of three stages, not one: accumulate and fire,
    backtrack the index, then :func:`_prune_boundaries` drops the change points the
    segments either side do not support. The prune is not cosmetic -- it is what takes
    the false-split rate on pure noise to 0.00% over 3000 trials -- so a caller reading
    only the raw firings would get a materially different answer.

    Args:
        x: The signal to segment. Must be finite -- see :func:`fit_levels_1d`, which
            is where non-finite events are refused; a NaN here would silently poison
            the running mean for every later sample.
        sigma: Noise scale in the units of ``x``. Must be finite and > 0.
        config: Thresholds and guards.

    Returns:
        Change-point indices in ``(0, x.size)``, ascending, each the FIRST sample of a
        new level. Empty means one level.
    """
    x = np.asarray(x, dtype=np.float64)
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma must be finite and > 0, got {sigma!r}")
    n = x.size
    min_len = int(config.min_level_samples)
    if n < 2 * min_len:
        return np.empty(0, dtype=np.int64)

    # RAW sigma governs MAGNITUDE (is this step big enough to be a state?) and the
    # accumulator, so `step_sigma` keeps meaning what it says: a multiple of the noise
    # you can see. EFFECTIVE sigma governs SIGNIFICANCE (could this difference be noise?)
    # and carries the autocorrelation correction -- on a low-passed trace a mean is worth
    # far fewer independent samples than it has. Conflating the two is what produced 11
    # levels on a rectangular event.
    tau_eff = max(1.0, float(tau))
    root_tau = float(np.sqrt(tau_eff))
    sigma_z = float(sigma) * root_tau

    drift = 0.5 * float(config.step_sigma)
    # The accumulator random-walks sqrt(tau) further per sample on correlated noise, so
    # the DECISION threshold scales with it while the drift term (in raw noise units)
    # does not. Scaling sigma inside `s` instead would silently redefine step_sigma.
    h = float(config.threshold_sigma) * root_tau
    max_cuts = int(config.max_levels) - 1
    # ``max_levels=1`` means "do not split", and CusumConfig sanctions it. Returning here
    # rather than relying on the in-loop guard is the whole fix: that guard runs AFTER the
    # append, so at max_cuts=0 it broke out having already recorded one cut, and a config
    # that forbade splitting produced two levels.
    if max_cuts <= 0:
        return np.empty(0, dtype=np.int64)

    cps: list[int] = []
    warm = max(int(config.warmup_samples), min_len)

    def _arm(lo: int) -> tuple[float, int, int]:
        """Seed a fresh segment: (mu, k, first armed index).

        The window is CAPPED at a third of what is left to look at. ``warmup_samples``
        is an upper bound, not a fixed cost: on the owner's -600 mV apo-ferritin
        recording the median event is 20 samples long and 59% are shorter than 24, so a
        flat 12-sample warm-up spends most of a typical event unarmed and the multilevel
        fraction swings 30% -> 21% purely on this constant. Scaling it means a long
        event gets the full robust window and a short one still gets to be looked at.

        mu is the MEDIAN of the first ``warm`` samples, not their mean, and not the
        single sample ``x[lo]``. The change-point estimate is allowed to land a sample
        early -- a noise excursion in the last sample of the old level defers the
        accumulator's zero by one -- which puts one stale old-level sample at the head
        of the new segment. Seeded with a mean (or with ``x[lo]``) that one sample drags
        mu toward the OLD level, the very next samples read as a fresh step, and the fit
        emits a 3-sample sliver at every transition: a clean 3-level staircase came back
        as change points [149, 152, 300] instead of [150, 300]. A median over the
        warm-up window rejects that single contaminating sample outright.
        """
        w = max(min_len, min(warm, (n - lo) // 3))
        hi = min(lo + w, n)
        return float(np.median(x[lo:hi])), hi - lo, hi

    mu, k, i = _arm(0)
    gp = gm = 0.0
    # Last sample at which each accumulator was at its floor: the change-point estimate.
    seg_start = 0
    zp = zm = max(i - 1, 0)

    while i < n:
        s = (x[i] - mu) / sigma
        gp = max(0.0, gp + s - drift)
        gm = max(0.0, gm - s - drift)
        if gp <= 0.0:
            zp = i
        if gm <= 0.0:
            zm = i

        fired = gp > h or gm > h
        # A change is only accepted if BOTH the level it closes and the shortest
        # possible level it opens can clear min_level_samples. Rejecting here rather
        # than merging afterwards keeps the running mean from being reset by a
        # transition that will not survive.
        if fired:
            cp = (zp if gp > h else zm) + 1
            if cp - seg_start >= min_len and n - cp >= min_len:
                cps.append(cp)
                if len(cps) >= max_cuts:
                    break
                seg_start = cp
                mu, k, i = _arm(cp)
                gp = gm = 0.0
                zp = zm = max(i - 1, cp)
                continue
            # Not a usable change point: drop the evidence and keep going, otherwise
            # the accumulator stays above h and re-fires on every subsequent sample.
            gp = gm = 0.0
            zp = zm = i

        k += 1
        mu += (x[i] - mu) / k
        i += 1

    # THE WARM-UP BLIND SPOT, and why two more passes are not optional.
    #
    # ``_arm`` consumes the first ``w`` samples of every segment to seed mu, and the
    # backtrack anchors start at ``hi - 1`` and only move forward, so no accepted change
    # point can be closer than ``w`` to the segment start. That silently overrode
    # ``min_level_samples``: measured on a 300-sample event whose true first level was 6
    # samples with a 6-sigma step, the step WAS detected (495/500) but the boundary was
    # pinned at 12 in every single trial. Four LEVEL_SCHEMA columns then carry wrong
    # numbers at once -- dwell_s 2x too long, depth_pA a 50/50 mixture of the two real
    # states, frac_depth likewise, and the next level's step_pA half the true step --
    # which is precisely the quantity this feature exists to measure. Below ~w/2 the
    # median seed lands on the SUCCESSOR level, no residual accumulates, and the step
    # disappears with no trace at all (detection 8% at 5 samples, 1% at 4).
    #
    # CUSUM is kept for what it is good at -- deciding THAT a level changed, sequentially
    # and cheaply -- and the two passes below fix WHERE, which is a different question and
    # is not answerable sequentially:
    #   _rescue_splits  recovers a step the warm-up sat on top of;
    #   _refine_boundaries moves each boundary to its least-squares-optimal position.
    # Both gate on the same z-test as _prune_boundaries, so neither can manufacture a
    # level the data does not support.
    cps = _rescue_splits(x, cps, sigma=sigma_z, z_min=float(config.threshold_sigma),
                         min_len=min_len, max_cuts=max_cuts,
                         step_sigma=config.step_sigma, sigma_raw=float(sigma))
    cps = _refine_boundaries(x, cps, min_len=min_len)
    cps = _drop_unearned_segments(x, cps, sigma=sigma_z,
                                  z_min=float(config.threshold_sigma), min_len=min_len)
    return _prune_boundaries(x, cps, sigma=sigma_z,
                             z_min=float(config.threshold_sigma))


def _rss(x: np.ndarray, a: int, b: int) -> float:
    if b - a <= 1:
        return 0.0
    seg = x[a:b]
    return float(((seg - seg.mean()) ** 2).sum())


def _drop_unearned_segments(
    x: np.ndarray, cps: list[int], *, sigma: float, z_min: float, min_len: int
) -> list[int]:
    """Remove whole segments that do not pay for themselves. THE sliver killer.

    Why :func:`_prune_boundaries` cannot do this. A transition sliver -- the 3-sample
    straddle CUSUM leaves when its boundary lands early -- has a mean BETWEEN its two
    neighbours, so each of its boundaries is individually significant and the per-boundary
    z-test keeps both. Measured on a clean 3-level staircase with a 15-sigma step, the
    sliver ``[149, 152, 300]`` scores z = 17.6 and z = 7.8 on its two boundaries against a
    threshold of 6. No per-boundary test can reject that, because per-boundary is the
    wrong question: the sliver's two boundaries really do separate different means. What
    is false is the SEGMENT.

    So ask about the segment instead: compare the two-boundary model (left, segment,
    right) against the one-boundary model over the same span, with that single boundary
    placed optimally. If collapsing the segment costs little residual sum of squares, the
    segment was never a state -- it was one transition described twice. ``dRSS / sigma^2``
    is a chi-square on the two parameters given up (a mean and a boundary); ``z_min^2``
    keeps it on the one significance scale this module uses throughout.

    Least-unearned-first and iterative: collapsing one segment re-forms its neighbours.
    """
    if len(cps) < 2:
        return cps
    cps = sorted(cps)
    thresh = float(z_min) * float(z_min)
    while len(cps) >= 2:
        bounds = [0, *cps, x.size]
        best: tuple[float, int, int] | None = None
        for i in range(len(cps) - 1):
            a, c = bounds[i], bounds[i + 3]
            rss3 = _rss(x, a, cps[i]) + _rss(x, cps[i], cps[i + 1]) + _rss(x, cps[i + 1], c)
            found = _best_split(x, a, c, min_len)
            if found is None:
                continue
            m, gain = found
            rss2 = _rss(x, a, c) - gain
            delta = (rss2 - rss3) / (sigma * sigma)
            if best is None or delta < best[0]:
                best = (delta, i, m)
        if best is None or best[0] >= thresh:
            break
        _delta, i, m = best
        cps = sorted([*cps[:i], m, *cps[i + 2:]])
    return cps


def _best_split(x: np.ndarray, a: int, b: int, min_len: int) -> tuple[int, float] | None:
    """Least-squares-optimal single split of ``x[a:b]`` as ``(absolute index, RSS drop)``.

    Vectorised over every candidate cut via prefix sums, so refining a boundary costs the
    same order as reading the segment. Mirrors ``characterize._best_split``; kept local
    because that one is bound to the BIC caller's ``_MIN_LEVEL_SAMPLES`` constant while
    this one takes the CUSUM config's.
    """
    n = b - a
    if n < 2 * min_len:
        return None
    seg = x[a:b]
    cs = np.cumsum(seg)
    cq = np.cumsum(seg * seg)
    m = np.arange(min_len, n - min_len + 1, dtype=np.int64)
    if m.size == 0:
        return None
    sl, ql = cs[m - 1], cq[m - 1]
    sr, qr = cs[-1] - sl, cq[-1] - ql
    nr = n - m
    rss_split = (ql - sl * sl / m) + (qr - sr * sr / nr)
    whole = float(cq[-1] - cs[-1] * cs[-1] / n)
    k = int(np.argmin(rss_split))
    return a + int(m[k]), whole - float(rss_split[k])


def _z_of_split(x: np.ndarray, a: int, m: int, b: int, sigma: float) -> float:
    """Two-sample z on the means either side of ``m``. The one significance test used
    everywhere in this module, so a boundary added by rescue must clear exactly what a
    boundary kept by the prune must clear."""
    nl, nr = m - a, b - m
    if nl < 1 or nr < 1:
        return 0.0
    se = sigma * np.sqrt(1.0 / nl + 1.0 / nr)
    if not (se > 0):
        return float("inf")
    return float(abs(x[a:m].mean() - x[m:b].mean()) / se)


def _step_magnitude(x: np.ndarray, a: int, m: int, b: int) -> float:
    """|mean(x[a:m]) - mean(x[m:b])|, the size of the step a boundary claims."""
    if m - a < 1 or b - m < 1:
        return 0.0
    return float(abs(x[a:m].mean() - x[m:b].mean()))


#: A finished level may scatter this many sigma about its own mean before it stops being
#: describable as one constant state. Only consulted for a segment that CANNOT be split
#: any further, so a mid-search segment still holding two levels is never judged by it.
#: 3.0 is loose on purpose: within-level scatter is ~1 sigma by construction, inflated
#: somewhat by the low-pass autocorrelation, and this exists to catch mixtures (16 sigma
#: in the case that motivated it), not to police noise.
_MAX_LEVEL_RMS_SIGMA = 3.0


def _is_describable_as_one_level(
    x: np.ndarray, a: int, b: int, *, sigma: float, min_len: int
) -> bool:
    """True if ``x[a:b]`` is either flat enough to BE a level, or still splittable.

    The second clause is what makes this usable mid-search: the first split of a 3-level
    staircase necessarily leaves one side holding two levels, and rejecting that would
    stop the rescue before it started. Such a side is long enough to be split again, so
    it passes here and is judged on its next pass.
    """
    n = b - a
    if n <= 1:
        return True
    if np.sqrt(_rss(x, a, b) / n) <= _MAX_LEVEL_RMS_SIGMA * sigma:
        return True
    return _best_split(x, a, b, min_len) is not None


def _rescue_splits(
    x: np.ndarray, cps: list[int], *, sigma: float, z_min: float, min_len: int, max_cuts: int,
    step_sigma: float, sigma_raw: float | None = None,
) -> list[int]:
    """Add boundaries CUSUM could not see, inside its own warm-up windows.

    Sequential detection cannot fire on a step it has already spent its warm-up
    straddling. A least-squares split search has no such blind spot -- it looks at the
    whole segment at once -- so one optimal-split proposal per segment, gated on the same
    z-test the prune uses, recovers exactly the levels the warm-up ate without opening the
    door to the over-splitting that made BIC unusable here (the gate, not the search, is
    what holds the false-split rate down).

    TWO gates, not one, and the second is the one that keeps this honest. A z-test alone
    asks "is this step real?", and over enough samples the answer is yes for a step far
    too small to be a sub-state: a 0.5-sigma step across 600 samples scores z = 6.1 and
    would be reported as a level even though ``step_sigma`` is 2.0. Worse, a feature
    SHORTER than ``min_level_samples`` cannot be excluded by length alone -- with
    ``min_level_samples=50``, a 10-sample spike smeared over the nearest legal 50-sample
    window scores z = 6.6 on a diluted mean of 1.0, and the fit would report a 50-sample
    "level" that exists nowhere in the data. So a rescued split must ALSO clear
    ``step_sigma * sigma`` in raw magnitude, which is precisely what that parameter
    promises: the smallest step this fit is asked to resolve.

    Iterates because splitting a segment creates two that may each hide another step.
    """
    if max_cuts <= 0:
        return cps
    # MAGNITUDE gate in RAW noise units -- "is the step big enough to be a state?" is a
    # question about the signal, not about how many independent samples measured it. Only
    # the z-test below carries the autocorrelation correction.
    raw = float(sigma if sigma_raw is None else sigma_raw)
    min_step = float(step_sigma) * raw
    cps = sorted(cps)
    while len(cps) < max_cuts:
        bounds = [0, *cps, x.size]
        best: tuple[float, int] | None = None
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            found = _best_split(x, a, b, min_len)
            if found is None:
                continue
            m, _gain = found
            if _step_magnitude(x, a, m, b) < min_step:
                continue
            # A rescued split must leave two describable levels behind. With
            # ``min_level_samples=50`` and a 10-sample spike at the end of the event,
            # the best legal split smears that spike over the nearest 50-sample window
            # and reports a "level" of mean 8 whose samples are 40 zeros and 10 forties
            # -- a 16-sigma mixture describing a state that exists nowhere in the trace.
            # The spike is genuinely shorter than the length the caller declared
            # resolvable, so the honest answer is no split, not an averaged one.
            # Also RAW: within-segment scatter is a marginal quantity, and correlation
            # changes how many independent samples you have, not the spread itself.
            if not (_is_describable_as_one_level(x, a, m, sigma=raw, min_len=min_len)
                    and _is_describable_as_one_level(x, m, b, sigma=raw, min_len=min_len)):
                continue
            z = _z_of_split(x, a, m, b, sigma)
            if z >= z_min and (best is None or z > best[0]):
                best = (z, m)
        if best is None:
            break
        cps = sorted([*cps, best[1]])
    return cps


def _refine_boundaries(x: np.ndarray, cps: list[int], *, min_len: int) -> list[int]:
    """Move each boundary to the least-squares-optimal position between its neighbours.

    CUSUM reports where the EVIDENCE crossed, corrected by a backtrack to the last zero.
    That estimator is unbiased on a clean step but it is still sequential, so it cannot
    place a boundary earlier than the warm-up window that preceded it. Re-solving each
    boundary against the two segments it actually separates is a batch question with an
    exact answer, and it is what takes the pinned-at-12 boundary back to the true 6.

    One sweep, left to right, each boundary free to move only within its neighbours'
    span so the pass can never reorder or collapse the partition.
    """
    if not cps:
        return cps
    cps = sorted(cps)
    for i, _cp in enumerate(cps):
        a = cps[i - 1] if i > 0 else 0
        b = cps[i + 1] if i + 1 < len(cps) else x.size
        found = _best_split(x, a, b, min_len)
        if found is not None:
            cps[i] = found[0]
    return cps


def _prune_boundaries(
    x: np.ndarray, cps: list[int], *, sigma: float, z_min: float
) -> np.ndarray:
    """Drop change points the segments either side do not actually support.

    THE SLIVER PROBLEM. A change point is allowed to land a few samples early, because
    a noise excursion in the tail of the old level defers the accumulator's zero. The
    warm-up median absorbs most of that, but when the boundary lands 2-3 samples early
    the warm-up window itself straddles the step, mu is seeded between the two levels,
    and the detector fires a second time as soon as the true level asserts itself. The
    result is a sliver exactly ``min_level_samples`` wide sitting on the transition:
    a clean 3-level staircase coming back as ``[150, 298, 301]``. Warm-up alone leaves
    ~2-3% of transitions like this at any window short enough to still resolve a
    12-sample level.

    A sliver is not merely short -- it is UNSUPPORTED. Its mean sits between its
    neighbours and it has 3 samples to prove the difference with, so the standard error
    of that difference is enormous. So the test is a two-sample z on the segment means,

        z = |mu_left - mu_right| / (sigma * sqrt(1/n_left + 1/n_right))

    and the weakest boundary is removed until every survivor clears ``z_min``. Taking
    ``z_min`` to be the detector's own ``threshold_sigma`` says something coherent: a
    boundary must be supported to the same confidence the detector demanded to fire.
    Long, well-separated levels are unaffected (their n makes z huge); a 3-sample
    intermediate straddle is not, which is the asymmetry being exploited.

    Iterative and least-significant-first, because removing one boundary changes the
    segments -- and therefore the z -- of its neighbours.
    """
    cps = list(cps)
    if not cps or not (np.isfinite(sigma) and sigma > 0):
        return np.asarray(cps, dtype=np.int64)
    n = x.size
    while cps:
        bounds = [0, *cps, n]
        means = np.array([x[bounds[i]:bounds[i + 1]].mean() for i in range(len(bounds) - 1)])
        sizes = np.array([bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)],
                         dtype=np.float64)
        se = sigma * np.sqrt(1.0 / sizes[:-1] + 1.0 / sizes[1:])
        z = np.where(se > 0, np.abs(np.diff(means)) / np.where(se > 0, se, 1.0), np.inf)
        k = int(np.argmin(z))
        if z[k] >= z_min:
            break
        cps.pop(k)
    return np.asarray(cps, dtype=np.int64)


def fit_levels_1d(
    x: np.ndarray,
    *,
    sigma: float,
    config: CusumConfig = DEFAULT_CUSUM,
    tau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment ``x`` into piecewise-constant levels.

    Returns ``(starts, stops)``, both int64 and relative to ``x``, with
    ``starts[0] == 0`` and ``stops[-1] == x.size``. A non-finite or too-short ``x``
    yields the single level ``[0, x.size)`` -- the same "one level" a flat event gets,
    because refusing to fit is not the same as fitting one state and must not look
    different in the table.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    whole = (np.array([0], dtype=np.int64), np.array([n], dtype=np.int64))
    if n < _MIN_SAMPLES_FOR_LEVELS or not np.isfinite(x).all():
        return whole
    cps = cusum_changepoints(x, sigma=sigma, config=config, tau=tau)
    if cps.size == 0:
        return whole
    starts = np.concatenate(([0], cps)).astype(np.int64)
    stops = np.concatenate((cps, [n])).astype(np.int64)
    return starts, stops


# ================================================================================================
# Over an event set -> the level table.
# ================================================================================================
def fit_event_levels(
    signal: np.ndarray,
    fs: float,
    events: pd.DataFrame,
    *,
    baseline: np.ndarray | float,
    polarity: int,
    sigma: float | None = None,
    tau: float | None = None,
    provenance: Mapping[str, Any] | None = None,
    config: CusumConfig = DEFAULT_CUSUM,
    accepted_only: bool = True,
    cancel: CancelFlag | None = None,
) -> pd.DataFrame:
    """Fit sub-state levels for every event in ``events``; return a LEVEL_SCHEMA frame.

    Args:
        signal: The current trace the events were detected in, in pA. This must be the
            SAME trace ``characterize`` measured -- if detection ran on a low-passed
            copy, pass that copy, or the fitted levels describe a signal whose events
            have different edges from the ones in the table.
        fs: Sample rate in Hz.
        events: An ``events.SCHEMA`` frame. ``start_idx`` / ``end_idx`` /
            ``baseline_pA`` / ``event_id`` are read; everything else is carried
            through or ignored.
        baseline: The baseline track (full length) or a scalar, as handed to
            ``characterize``. Used to form ``dev``.
        polarity: +1 or -1, the same value ``characterize`` was given.
        sigma: Noise RMS in pA. ``None`` takes ``provenance['noise_rms_pA']``, and
            failing that measures the detrended trace -- in that order and for the
            reason ``characterize.span_noise_rms`` gives: matching the number the
            table's ``snr`` already used matters more than a locally better estimate.
        tau: The autocorrelation time in SAMPLES that the z-tests are discounted by.
            ``None`` measures it from the WHOLE RECORDING (open-pore residual, floored
            by the low-pass filter's own tau -- see the block below), which is the right
            answer for a fit over a whole table and the wrong PRICE for a fit over ONE
            event: the segmentation of a 600-sample event is ~0.33 ms while the estimate
            behind it walks every event's residual in the recording. Pass the value a
            previous fit resolved (``levels.attrs[TAU_ATTR]``, and
            ``AnalysisResult.params['tau_samples']`` after ``modules.fit_levels``) to
            re-fit one edited event with provably the same estimator as its neighbours.
            Must be finite and > 0; ``1.0`` means "uncorrelated".
        provenance: The detection provenance, for ``noise_rms_pA``.
        accepted_only: Fit only curated-in events. The default matches every export
            path in this app: a rejected event is not part of the result.
        cancel: Cooperative cancellation, polled per event.

    There is deliberately NO ``t0_s`` offset. ``characterize`` writes
    ``t_start_s = starts / fs`` with no recording-start term, so a level carrying one
    would sit on a clock the parent event does not use -- and ``RecordingMeta.t0_s`` is
    a real non-zero number on this owner's files (a windowed ABF reports 61.303 s), so
    the obvious wiring would have silently put every level 61 s away from its own event.
    Levels are on the event table's clock, by construction, with no way to opt out.

    Raises:
        ValueError: if ``fs``/``polarity``/``baseline`` are impossible, or if any event
            lies outside ``signal``. The bounds are VALIDATED, not clipped: an event
            reaching past the end of the trace means the caller paired an event table
            with the wrong signal (a slice, a different channel, a stale detection), and
            silently truncating it produces a level table that looks fine and describes
            nothing.

    Returns:
        One row per (event, level), ordered by ``event_id`` then ``level_idx``.
        An event that could not be fitted (too short, non-finite samples) still
        contributes exactly one row spanning it, with ``n_levels == 1``.

        ``.attrs[SIGMA_ATTR]`` / ``.attrs[TAU_ATTR]`` carry the noise scale and the
        autocorrelation time this call actually resolved, so a later single-event
        re-fit can run the SAME estimator instead of a fresh one.
    """
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError(f"fs must be finite and > 0, got {fs!r}")
    if int(polarity) not in (-1, 1):
        raise ValueError(f"polarity must be +1 or -1, got {polarity!r}")
    if tau is not None:
        tau = float(tau)
        # VALIDATED, not clipped. A tau of 0 divides by zero inside the z-tests and a
        # negative one inverts the discount, so a caller handing back a number it got from
        # somewhere else (a restored project's params, a hand-edited config) must be told,
        # not silently given a different segmentation from the fit it is splicing into.
        if not (np.isfinite(tau) and tau > 0):
            raise ValueError(f"tau must be finite and > 0, got {tau!r}")

    signal = np.asarray(signal, dtype=np.float64)
    pol = int(polarity)
    base = (np.full(signal.size, float(baseline), dtype=np.float64)
            if np.isscalar(baseline) or np.asarray(baseline).ndim == 0
            else np.asarray(baseline, dtype=np.float64))
    if base.size != signal.size:
        raise ValueError(
            f"baseline has {base.size} samples but signal has {signal.size}")

    sig = _resolve_sigma(sigma, provenance, signal, base, pol)

    # THE AUTOCORRELATION CORRECTION, measured ONCE per recording. Estimating it per
    # event is not an option: the median event here is 20 samples and an ACF needs
    # hundreds. The residual is dominated by open pore (events are a small duty cycle),
    # which is exactly the noise the z-tests are asking about.
    # THE FILTER RAMP. Detection bounds enclose the low-pass rise and fall, so each event
    # opens and closes with a ramp between two real levels. That ramp is not a state, but
    # it is a genuine sustained change in mean, so every significance test in this module
    # is right to flag it -- which is why 74-79% of PERFECTLY RECTANGULAR events came back
    # multilevel at the app default. The fix is to not offer it: fit the interior only,
    # then hand the outer levels back their event's true edges so levels still tile it.
    rise_s = None
    if provenance is not None:
        try:
            rise_s = float(provenance.get("filter_rise_s"))
        except (TypeError, ValueError):
            rise_s = None
    n_edge = 0
    if rise_s is not None and np.isfinite(rise_s) and rise_s > 0:
        n_edge = int(np.ceil(rise_s * float(fs)))

    rows = events
    if accepted_only and "accepted" in rows.columns:
        rows = rows[rows["accepted"].astype(bool)]
    if rows.empty:
        # No tau was resolved (it is measured FROM the rows), and saying so is the point of
        # the NaN: a caller that reuses this stamp must fall back to measuring, not to 0.
        return _stamp_estimators(empty_levels(), sig, tau)

    # event_id comes from the COLUMN, or failing that the frame's index -- never from a
    # running count of output rows. The old ``getattr(row, 'event_id', len(out))``
    # fallback numbered by LEVEL: a 2-level event followed by a 1-level event came back
    # as ids [0, 1, 2], so the second level of event 0 became its own "event" and
    # levels_summary reported three events where there were two.
    if "event_id" in rows.columns:
        ids = pd.to_numeric(rows["event_id"], errors="coerce")
        if ids.isna().any():
            raise ValueError("events['event_id'] contains non-numeric or missing ids")
        event_ids = ids.astype("int64").tolist()
    else:
        event_ids = [int(v) for v in rows.index]

    starts_all = rows["start_idx"].astype("int64").to_numpy()
    ends_all = rows["end_idx"].astype("int64").to_numpy()
    if starts_all.size:
        lo_bad, hi_bad = int(starts_all.min()), int(ends_all.max())
        if lo_bad < 0 or hi_bad > signal.size:
            raise ValueError(
                f"events span samples [{lo_bad}, {hi_bad}) but signal has {signal.size} "
                f"samples: the event table and the trace do not describe the same data "
                f"(a slice, a different channel, or a stale detection)")

    # THE AUTOCORRELATION CORRECTION, measured ONCE per recording (an ACF needs hundreds
    # of samples; the median event here is 20). The residual is dominated by open pore,
    # which is exactly the noise the significance tests ask about. n_scale is the length
    # the inference actually happens over -- a typical event -- so the Bartlett weight
    # discounts correlation longer than the segments being compared.
    #
    # SKIPPED ENTIRELY when the caller handed one in. "Measured once per recording" is the
    # whole design, and a re-fit of ONE edited event is still that recording: paying for it
    # again would make the cheap half (~0.33 ms of segmentation for a 600-sample event) ride
    # on a whole-recording walk, AND -- worse -- would let the re-fitted event answer to a
    # slightly different significance scale from the neighbours it is spliced beside, because
    # the estimate moves with whatever the edit did to the event set. Reusing the number is
    # what makes the two fits the SAME fit.
    if tau is None:
        _lens = (rows["end_idx"].astype("int64") - rows["start_idx"].astype("int64")).to_numpy()
        _ns = int(max(8, np.median(_lens))) if _lens.size else 64
        # PREFER the filter's own tau. It is the only estimate that cannot be contaminated by
        # drift or by the baseline estimator's tracking error -- see filter_autocorr_time. The
        # open-pore residual estimate remains the fallback for a detection that ran with no
        # low-pass at all, where there is no filter to ask.
        _cut = None
        if provenance is not None:
            try:
                _cut = float(provenance.get("filter_cutoff_hz"))
            except (TypeError, ValueError):
                _cut = None
        tau_open = open_pore_autocorr_time(
            signal - base,
            rows["start_idx"].astype("int64").to_numpy(),
            rows["end_idx"].astype("int64").to_numpy(),
            n_scale=_ns, pad=n_edge,
        )
        # The FILTER's tau is a FLOOR, not an alternative. That correlation is definitely
        # present -- the detector imposed it -- so no measurement may claim less. Above the
        # floor the open-pore estimate carries what the filter model cannot know about: these
        # traces are not white, they are 1/f (the PSD panel measures alpha ~= 1.34), and that
        # excess IS real correlated noise the significance tests must answer for.
        #
        # Taking either one alone fails in a measurable direction. Filter-only ignores the 1/f
        # and over-splits: 85.6% and 89.8% multilevel with 9.5 and 12.5 levels per event on the
        # owner's two apo files. Residual-only is swamped by drift and under-splits, which is
        # what suppressed a genuine 3-level event to 1.
        tau = tau_open
        if _cut is not None and np.isfinite(_cut) and _cut > 0:
            tau = max(tau_open, filter_autocorr_time(
                float(fs), _cut,
                order=int(provenance.get("filter_order") or 8),
                kind=str(provenance.get("filter_kind") or "bessel"),
                n_scale=_ns))

    poll = CancelPoller(cancel)
    out: list[dict[str, Any]] = []
    for ev_id, row in zip(event_ids, rows.itertuples(index=False)):
        poll()
        a, b = int(row.start_idx), int(row.end_idx)
        if b <= a:
            continue

        dev = pol * (signal[a:b] - base[a:b])
        # Trim the ramps, fit the interior, then restore the edges. Skipped when the
        # event is too short to survive it -- a trim that leaves nothing would turn a
        # real event into no levels at all.
        trim = n_edge if (b - a) > 2 * n_edge + _MIN_SAMPLES_FOR_LEVELS else 0
        core = dev[trim:len(dev) - trim] if trim else dev
        starts, stops = fit_levels_1d(core, sigma=sig, config=config, tau=tau)
        if trim:
            starts = starts + trim
            stops = stops + trim
            starts[0] = 0                 # the first level owns the entry ramp...
            stops[-1] = dev.size          # ...and the last owns the exit ramp
        n_lv = int(starts.size)
        base_pA = float(getattr(row, "baseline_pA", np.nan))
        abs_base = abs(base_pA)
        prev_depth = np.nan

        for j in range(n_lv):
            lo, hi = int(starts[j]), int(stops[j])
            seg_dev = dev[lo:hi]
            # MEDIAN, matching events.SCHEMA['depth_pA'] (characterize takes a grouped
            # median, not a mean) so a level depth and an event depth are the same kind
            # of number and can be compared without a footnote.
            depth = float(np.median(seg_dev)) if seg_dev.size else np.nan
            out.append({
                "event_id":   int(ev_id),
                "recording":  str(getattr(row, "recording", "") or ""),
                "pore_id":    str(getattr(row, "pore_id", "") or ""),
                "level_idx":  j,
                "n_levels":   n_lv,
                "start_idx":  a + lo,
                "end_idx":    a + hi,
                "t_start_s":  (a + lo) / fs,
                "dwell_s":    (hi - lo) / fs,
                "level_pA":   float(np.mean(signal[a + lo : a + hi])),
                "depth_pA":   depth,
                "frac_depth": depth / abs_base if abs_base > 0 else np.nan,
                "step_pA":    np.nan if j == 0 else depth - prev_depth,
            })
            prev_depth = depth

    if not out:
        return _stamp_estimators(empty_levels(), sig, tau)
    df = pd.DataFrame(out)
    df = df.sort_values(["event_id", "level_idx"], kind="stable").reset_index(drop=True)
    # Same rule the event schema states: no +/-Inf ever reaches a float column; NaN is
    # this app's one "not measured" sentinel (see events.demote_infinities).
    for col in _LEVEL_FLOAT_COLUMNS:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return _stamp_estimators(
        df.astype({c: d for c, (d, _, _) in LEVEL_SCHEMA.items()})[list(LEVEL_SCHEMA)],
        sig, tau)


def _resolve_sigma(
    sigma: float | None,
    provenance: Mapping[str, Any] | None,
    signal: np.ndarray,
    base: np.ndarray,
    pol: int,
) -> float:
    """The noise scale, by the same ladder ``span_noise_rms`` uses. Always finite and > 0."""
    if sigma is not None:
        s = float(sigma)
        if not (np.isfinite(s) and s > 0):
            raise ValueError(f"sigma must be finite and > 0, got {sigma!r}")
        return s
    if provenance is not None:
        try:
            pn = float(provenance.get("noise_rms_pA"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pn = float("nan")
        if np.isfinite(pn) and pn > 0:
            return pn
    resid = signal - base
    resid = resid[np.isfinite(resid)]
    if resid.size:
        try:
            nr = _noise_rms(resid)
        except ValueError:
            nr = float("nan")
        if np.isfinite(nr) and nr > 0:
            return float(nr)
    return 1e-9


def levels_summary(levels: pd.DataFrame) -> dict[str, float]:
    """Scalars worth putting on an AnalysisResult: how much structure the fit found."""
    if levels.empty:
        return {"n_events_fitted": 0.0, "n_levels_total": 0.0,
                "frac_multilevel": float("nan"), "mean_levels_per_event": float("nan")}
    per_event = levels.groupby("event_id")["n_levels"].first()
    return {
        "n_events_fitted": float(per_event.size),
        "n_levels_total": float(len(levels)),
        "frac_multilevel": float((per_event > 1).mean()),
        "mean_levels_per_event": float(per_event.mean()),
    }


# ================================================================================================
# reconciliation: a level table describes SPANS, and spans move
# ================================================================================================
#
# THE INVARIANT THIS RESTS ON, and it is a property of :func:`fit_event_levels` rather
# than a convention: the levels of one event TILE that event exactly. The fit slices
# ``dev = signal[a:b]`` for the parent event's ``[start_idx, end_idx)``, and after the
# edge-trim restoration writes ``starts[0] = 0`` and ``stops[-1] = dev.size`` before
# offsetting every boundary by ``a``. So for every event present in a level table,
#
#     levels.start_idx.min() == event.start_idx   and   levels.end_idx.max() == event.end_idx
#
# holds by construction, and the moment it stops holding the levels are describing a
# span the event no longer has. That makes staleness DERIVABLE from the two tables --
# no call site has to announce a mask edit, a merge, an undo or a restore for the level
# table to notice that it no longer describes anything.
#
# Which is the whole point. The measured defect (C-5) was that the Event Inspector
# dropped the stale rows from its own private copy while the window that OWNS the fit,
# the result store and therefore the project file all kept them: a 4.0 ms event
# exporting 6.0 ms of sub-state, with the UI drawing no CUSUM line and the CSV writing
# one. Routing one more announcement would have fixed the one path that announces;
# deriving it here fixes every path, including the ones written next month.

def reconcile_levels(
    levels: pd.DataFrame | None, events: pd.DataFrame | None
) -> tuple[pd.DataFrame | None, list[int]]:
    """Drop every level group that no longer describes its parent event's span.

    Returns ``(kept, dropped_event_ids)``. ``kept`` is ``None`` when nothing survives --
    absence is how both the readout and the export say "no level fit", and a level table
    that describes spans that no longer exist is worse than no level table at all.

    A group is dropped when its ``event_id`` is gone from ``events``, or when the span
    it tiles is not the span the event now has. Rows with a non-finite / unusable
    ``event_id`` are dropped too: they cannot be matched to anything, so they cannot be
    certified as current.

    *** ``kept is levels`` IS THE "NOTHING MOVED" SIGNAL, AND IT IS THE ONLY ONE. ***
    Callers must branch on that identity, never on ``dropped`` being empty. There are
    three inputs this function REFUSES to certify -- a levels table with no
    ``start_idx``/``end_idx``, an ``event_id`` column that will not become an integer,
    and (below) nothing to work with at all -- and on those it can return ``kept=None``
    while naming no ids, because it cannot read the ids to name them. Measured: with the
    two callers branching on ``dropped``, the Inspector took ``kept=None`` and drew no
    CUSUM line while MainWindow saw ``dropped == []``, kept the uncertified table and
    left Export enabled -- which is precisely the C-5 symptom this function exists to
    end, reopened at the one input the shared rule declines to answer. ``dropped`` is
    for the sentence shown to the user; ``kept`` is the answer.

    Pure and total -- ``events=None`` means "no event table", which invalidates the
    whole fit, and either frame may be empty.
    """
    if levels is None or getattr(levels, "empty", True):
        # Nothing to reconcile is not a refusal: hand back exactly what came in, so
        # `kept is levels` stays true whenever nothing was dropped.
        return levels, []
    if events is None or getattr(events, "empty", True) \
            or "event_id" not in getattr(events, "columns", ()):
        return None, [int(v) for v in pd.unique(levels["event_id"])]
    if not {"event_id", "start_idx", "end_idx"} <= set(levels.columns):
        # A table that cannot state which span it describes cannot be certified.
        return None, []

    if not {"start_idx", "end_idx"} <= set(events.columns):
        return None, [int(v) for v in pd.unique(levels["event_id"])]
    try:
        spans = events[["start_idx", "end_idx"]].set_index(
            events["event_id"].astype("int64"))
        ids = levels["event_id"].astype("int64")
    except (TypeError, ValueError):
        # An event_id column that will not become an integer cannot be joined on, so
        # nothing in this table can be certified as still describing a live span.
        return None, []
    spans = spans[~spans.index.duplicated(keep="first")]
    grouped = levels.groupby(ids)
    lo = grouped["start_idx"].min()
    hi = grouped["end_idx"].max()
    want_lo = spans["start_idx"].reindex(lo.index)
    want_hi = spans["end_idx"].reindex(hi.index)
    ok = (lo.to_numpy() == want_lo.to_numpy()) & (hi.to_numpy() == want_hi.to_numpy())
    good = set(int(v) for v in lo.index[ok])
    dropped = sorted(int(v) for v in lo.index if int(v) not in good)
    if not dropped:
        return levels, []
    kept = levels[ids.isin(good)]
    return (None if kept.empty else kept), dropped
