"""Per-event feature extraction: turn (start, end) index pairs into the EventSet SCHEMA.

Detection says *where* the events are. This module says *what they are*: how deep,
how long, how much charge, how many sub-states.

THE DEPTH DEFINITION -- READ THIS
---------------------------------
`depth_pA` here is the **median** of the in-event deviation, not the max.

This is a scientific choice, not a style choice, and it changes published numbers:

  * The max of N samples drawn from a level plus noise is an *extreme order
    statistic*. Its expectation grows like sigma*sqrt(2*ln(N)), so a max-based
    depth is biased high by roughly +2 to +3 sigma for a typical 100-2000 sample
    event -- and the bias *grows with dwell time*, which manufactures a spurious
    dwell-vs-depth correlation in exactly the scatter plot everyone publishes.
  * The median is a consistent estimator of the blockade level. Its bias is ~0
    and its scatter shrinks like 1/sqrt(N). It is also robust to the rise/fall
    transitions bracketing the event and to a spike part-way through.

`max_dev_pA` is still reported, because it is the honest thing to look at for
spikes and for filter-limited events. It is just not the depth.

Reviewers ask about this. The answer is: median of in-event dev, transitions
included, no trimming; see `tests/analysis/test_characterize.py::test_median_depth_unbiased_max_is_not`
for the measured bias.

Sign convention (from `nanoscope.analysis.events`): dev = polarity * (current - baseline)
is positive during an event for blockades and enhancement alike. Every magnitude
here is therefore positive and direction lives only in `polarity`.

Signal-first port (PORT_PLAN §3.3): the source consumed a `Recording`; this takes
the current array, the local baseline, the event edges, `fs`, and `polarity` as
explicit primitives. The algorithm and every load-bearing invariant are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .cancel import CancelFlag, CancelPoller
from .events import (
    DERIVED_FLAGS,
    FLAG_CLIPPED,
    FLAG_MAX_DWELL,
    FLAG_NON_FINITE,
    FLAG_SATURATED,
    FLAG_ZERO_BASELINE,
    SCHEMA,
    analysed_span,
    combine_flags,
    empty_df,
    inherited_curation,
)
from .filters import noise_rms as _noise_rms

# Below this many samples an event has no resolvable internal structure: a
# 2-level fit would just be fitting the noise. Hard floor, not a tunable.
_MIN_SAMPLES_FOR_LEVELS = 10

# Cap on sub-states. Beyond ~4 the piecewise-constant model is over-fitting a
# filtered, autocorrelated trace rather than resolving real protein conformers.
_MAX_LEVELS = 4

# A candidate sub-state shorter than this is noise, not a conformational state.
_MIN_LEVEL_SAMPLES = 3

# Rise/fall crossings are searched a little outside [start, end): the detector
# triggers at k*sigma, which for a shallow event is already above the 10% level.
# The pad is ``max(_EDGE_PAD_SAMPLES, _EDGE_PAD_S * fs)`` -- see ``_edge_pad_samples``.
_EDGE_PAD_SAMPLES = 64

#: The pad's floor in time: 64 samples at the owner's 50 kHz. A fixed sample count is a
#: shrinking window as the rate climbs (64 samples are 0.26 ms at 250 kHz), and an edge
#: whose 10 % point lies further out than that comes back NaN -- which a hand-placed or
#: step-fitted start, sitting MID-RAMP, makes common: a 0.6 ms edge at 250 kHz has its
#: 10 % point 0.3 ms before the fitted start, 75 samples, outside the old 64.
_EDGE_PAD_S = 1.28e-3


def _edge_pad_samples(fs: float) -> int:
    """The rise/fall search pad, in samples: ``max(64, 1.28 ms)``.

    BIT-IDENTICAL to the old fixed 64 at 50 kHz (64 samples ARE 1.28 ms) and at every lower
    rate; only a rate above ~50.4 kHz changes. ``round``, not ``ceil``: ``1.28e-3 * 50000.0``
    is not exactly 64.0 in floating point and a ceiling would turn it into 65, moving
    every row of the owner's recordings. A rate that is not a finite positive number keeps
    the old 64.
    """
    try:
        n = _EDGE_PAD_S * float(fs)
    except (TypeError, ValueError):
        return _EDGE_PAD_SAMPLES
    if not np.isfinite(n) or n <= _EDGE_PAD_SAMPLES:
        return _EDGE_PAD_SAMPLES
    return int(round(n))

#: A row whose depth was measured off a FLAT TOP -- the samples ran into an
#: amplifier or ADC rail, so what was recorded is where the hardware stopped, not
#: how deep the blockade went. ``depth_pA``, ``frac_depth``, ``ecd_pA_s`` and
#: ``max_dev_pA`` are then LOWER BOUNDS, exactly as ``clipped_by_boundary`` says of
#: ``dwell_s`` when the analysed span cuts an event -- and until this token existed
#: the app had no way to say that about the amplitude axis. Measured: 20 blockades
#: of true depth 300 pA on a 1000 pA pore, clipped at an 800 pA rail, came back
#: ``depth_pA`` 200.30 / ``frac_depth`` 0.2002 (-33%) with ``flag=''`` and
#: ``accepted=True`` on every row.
#:
#: Like ``zero_baseline`` it is a CAVEAT, not a rejection: the event is real, its
#: dwell is a real measurement, and the depth is a real floor -- so the row stays
#: accepted and the flag says what the number is worth. The token is defined in
#: :mod:`nanoscope.analysis.events` and imported above for compatibility.

#: How many CONSECUTIVE samples must sit on exactly the same extreme value before
#: the event is called saturated. Ionic current plus thermal noise does not repeat
#: a float value three times in a row at the extreme of the whole analysed array;
#: an amplifier sitting on its rail does nothing else. Checked on the owner's four
#: real recordings (-500 / -600 / -700 / -1000 mV, 2.7 M - 37 M samples): zero rows
#: flagged, so the rule costs an ordinary recording nothing.
_SATURATION_RUN = 3


def span_noise_rms(
    left: np.ndarray,
    right: np.ndarray,
    window: np.ndarray,
    provenance: Mapping[str, Any] | None = None,
) -> float:
    """The SNR denominator for a span re-measured after detection. Finite and > 0.

    Every row the DETECTOR produced divides by one number: ``noise_rms(current -
    baseline_track)`` over the whole recording, stamped as ``provenance['noise_rms_pA']``.
    ``snr`` is therefore only meaningful as a comparison BETWEEN rows of one table, and
    a row re-measured afterwards (a merge, an Event Inspector mask resize, a hand-added
    missed event) has to land on the same scale or it is not comparable with its
    neighbours.

    It did not. All three re-measurement paths took ``1.4826 * MAD`` of the RAW flank
    current -- and ``filters.noise_rms``'s own docstring says MAD is taken about the
    median, so a drift or a level step inflates it. The flanks around a MERGED span are
    the worst case: the pad is as wide as the merged event, so the left and right flanks
    can sit at visibly different levels. Measured on the user's -600 mV recording over
    349 merged spans, against a detector noise of 3.68 pA:

    ======================================  ===========  ==============
    estimator                                median      x detector
    ======================================  ===========  ==============
    raw pooled flank MAD (what it did)        43.32 pA        11.8
    per-flank median removed first             6.34 pA         1.7
    ======================================  ===========  ==============

    So the merged rows' SNR came out ~5-7x too small -- median SNR 0.85 on merged rows
    against 6.27 on unmerged ones in the same table -- and any SNR-based curation threw
    the merged events away for being noisy when nothing about them was.

    The order below follows from that:

    1. **The detector's own ``noise_rms_pA``**, when provenance carries it. It is a
       detrended-residual MAD by construction and it is literally the number every other
       row in the table used, so the re-measured row is comparable by identity rather
       than by luck.
    2. Otherwise a **detrended** flank MAD: each flank has its own median removed before
       they are pooled, which is what kills the level step (MAD is shift-invariant, so
       removing a single POOLED median -- as the baseline does -- would change nothing).
    3. Otherwise the whole window's MAD, which at least has the event in it.
    4. Otherwise a tiny epsilon, because :func:`characterize` refuses a non-positive
       ``noise_rms``.

    Args:
        left: Samples immediately before the span (may be empty).
        right: Samples immediately after the span (may be empty).
        window: The whole padded neighbourhood, as a last resort.
        provenance: The detection's provenance, or None.
    """
    if provenance is not None:
        pn = provenance.get("noise_rms_pA")
        try:
            pn = float(pn)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pn = float("nan")
        if np.isfinite(pn) and pn > 0:
            return float(pn)

    # A flank carrying NaN/Inf is dropped rather than detrended: its median is itself
    # non-finite, ``f - median`` is then ``inf - inf`` = NaN, and every rebuilt row over
    # a poisoned neighbourhood emits a numpy RuntimeWarning out of a GUI worker (up to
    # one per row of a merge) before ``_noise_rms`` refuses the result anyway. Dropping
    # it costs nothing and can only improve the estimate -- a clean flank beside a
    # poisoned one now yields a real scale instead of the 1e-9 epsilon at the bottom of
    # the ladder. The ROW is unaffected either way: `recharacterize_span` has already
    # blanked its baseline, so it is flagged ``non_finite`` and its snr is NaN.
    parts = [
        np.asarray(f, dtype=np.float64) - np.median(np.asarray(f, dtype=np.float64))
        for f in (np.asarray(x, dtype=np.float64) for x in (left, right))
        if f.size and np.isfinite(f).all()
    ]
    candidates = []
    if parts:
        candidates.append(np.concatenate(parts))
    if np.asarray(window).size:
        candidates.append(np.asarray(window, dtype=np.float64))
    for cand in candidates:
        try:
            nr = _noise_rms(cand)
        except ValueError:
            nr = float("nan")
        if np.isfinite(nr) and nr > 0:
            return float(nr)
    return 1e-9


def _validate_fs(fs: Any) -> float:
    """The one ``fs`` gate both `characterize` and `recharacterize_span` pass
    through -- matching `detect.detect_events`'s contract so a bad rate is refused
    the same way everywhere: ``ValueError`` naming ``fs``, never a bare
    ``TypeError``/``ZeroDivisionError``/``OverflowError`` from an un-vetted
    arithmetic use downstream. ``None`` is refused with `detect_events`'s own
    wording; anything else that will not convert, plus non-positive or
    non-finite rates, are refused with `_validate_detection_timing`'s wording.
    """
    if fs is None:
        raise ValueError("fs is required; dwell is undefined without a sample rate")
    try:
        fs = float(fs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fs must be positive and finite, got {fs!r}") from exc
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs!r}")
    return fs


def characterize(
    current: np.ndarray,
    baseline: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    fs: float,
    polarity: int,
    *,
    noise_rms: float,
    filter_rise_s: float | None = None,
    max_dwell_s: float | None = None,
    clip_bounds: tuple[int, int] | None = None,
    inherit_flag: Any = None,
    inherit_accepted: Any = None,
    pore_id: str = "",
    recording: str = "",
    units: str = "pA",
    cancel: CancelFlag | None = None,
) -> pd.DataFrame:
    """Measure every detected event and return rows matching `events.SCHEMA`.

    THIS FUNCTION OWNS ``flag`` AND ``accepted``
    -------------------------------------------
    Every row in the app is built here -- the Schmitt detector, the ML detector,
    a gap merge, an Event Inspector mask resize and a hand-added missed event all
    come through this one call. So the four automatic rejection rules live here
    and nowhere else:

    ======================  ===================================================
    ``zero_baseline``        ``|baseline_pA| == 0``; ``frac_depth`` is undefined
    ``dwell>max_dwell_s``    longer than the declared limit (a clog)
    ``clipped_by_boundary``  OVERLAPS ``clip_bounds`` and touches an edge of it
                             (a row wholly outside the analysed span was cut by
                             nothing, so the token would be a false reason)
    ``non_finite``           a sample it was measured from, or a magnitude
                             computed out of one, is NaN or +/-Inf
    ======================  ===================================================

    Plus two CAVEATS that flag a row without rejecting it, because the event is real
    and one of its numbers is not a measurement:

    ======================  ===================================================
    ``zero_baseline``        (above) ``frac_depth`` is undefined
    ``saturated``            the depth was read off a flat top: 3+ consecutive
                             samples on exactly the extreme value of the
                             analysed trace, i.e. an amplifier/ADC rail, so
                             depth/frac_depth/ecd are LOWER BOUNDS
                             (:data:`FLAG_SATURATED`)
    ======================  ===================================================

    They were previously applied by the CALLER, one caller at a time, which is why
    the boundary-clip rule reached ``detect_events`` and none of the other three
    paths, and why a merge could rebuild a rejected row as an accepted one with its
    reason erased. A rule applied here cannot be forgotten by a new caller.

    WHERE A NaN/Inf HAS TO BE BEFORE IT TAINTS A ROW
    ------------------------------------------------
    The rule is: **a row is tainted by a non-finite sample exactly when one of the
    numbers it REPORTS was computed from that sample.** Not "near the event" -- one
    bad sample in a 15 M-sample file must cost the handful of rows that read it, not
    the table. Measured, region by region (a 600-sample event, one bad sample):

    ==========================  ==============================================  ========
    where the bad sample is      what it does to the row                         taint?
    ==========================  ==============================================  ========
    inside ``[start, end)``      ``ecd``/``mean_dev`` = +/-Inf while the median   YES
                                 depth reads through it (20.135567 against a
                                 clean 20.140960) -- a plausible wrong number
    the local baseline /         NaN: baseline -> NaN, every magnitude with it.   YES
    noise flanks, i.e. the       Inf: the flank MEDIAN is robust and survives,
    padded window of             but the noise ladder collapses to its 1e-9
    :func:`recharacterize_span`  epsilon and ``snr`` lands at 2.0e10, finite
                                 and unflagged. Handled by blanking the local
                                 baseline there, so BOTH reach this rule
    the rise/fall search pad     A non-finite sample compares TRUE against one    no --
    (``_EDGE_PAD_SAMPLES``       of the two levels and becomes a false crossing:  but the
    each side of the event)      ``+Inf >= 0.9 d`` puts the 90 % point before     SIDE
                                 the event and ``rise_s`` comes back **0.0**;     that
                                 ``-Inf <= 0.1 d`` becomes a false 10 % point     READ
                                 after it and shortens ``fall_s``                 it -> NaN
    anywhere else in the trace   nothing: no reduction this row reports reads     NO
                                 it
    ==========================  ==============================================  ========

    The pad is the one asymmetric case and it is deliberate: ``rise_s``/``fall_s``
    already document NaN as "the event never reached 90 % inside the search window",
    so refusing to measure them is the honest answer, while ``depth``, ``ecd`` and the
    baseline are untouched by anything outside ``[start, end)`` -- rejecting the whole
    row for it would be over-reach. Only the crossing time whose OWN 10-90 % interval
    read the bad sample is refused: the other side, and the neighbouring event whose
    pad overlaps the same samples, keep their numbers. See ``rise_fall_over_set``.

    Neither detector can reach this rule with a poisoned RECORDING (``detect_events``,
    ``estimate_baseline`` and ``filters.noise_rms`` each refuse one outright); what
    does reach it is a poisoned CALIBRATION -- a float32 overflow or a bad gain applied
    after load -- and every rebuild path, which reads its own local reference.

    Args:
        current: Signal samples in `units` (pA by convention). `polarity` sets the
            sign convention.
        baseline: Full-length local open-pore current in the same units (same length
            as `current`), e.g. from `nanoscope.analysis.baseline.estimate_baseline`.
        starts: First sample index of each event (int, ascending, non-overlapping).
        ends: One-past-last sample index of each event. Must satisfy ends > starts.
        fs: Sample rate in Hz.
        polarity: +1 or -1; makes dev = polarity*(current - baseline) positive in event.
        noise_rms: Baseline noise RMS in the signal units, used for `snr`. Must be
            finite and > 0.
        filter_rise_s: 10-90% rise time of the anti-alias/analysis filter, in
            seconds. If given, events with `dwell_s < 2 * filter_rise_s` are marked
            `filter_limited` -- their depth is attenuated by the filter and their
            dwell is biased long, so they must not be fit as if they were resolved.
        max_dwell_s: Events longer than this are KEPT but flagged
            ``dwell>max_dwell_s`` and not accepted -- a clog is real signal the user
            must see. `None` means no limit was declared, so the rule cannot be
            applied (and, on a rebuild, its token must be inherited instead of
            re-derived; see :func:`~nanoscope.analysis.events.inherited_curation`).
        clip_bounds: ``(lo, hi)`` -- the edges of the ANALYSED span, in the same
            coordinates as `starts`/`ends`. An event that OVERLAPS the span and
            has ``start <= lo`` or ``end >= hi`` was cut off by that edge rather
            than by the pore: its dwell is a lower bound, its depth is the median
            of whatever fraction survived, and its baseline was read from the one
            flank that exists. Such rows are flagged ``clipped_by_boundary`` and
            not accepted. An event lying wholly outside the span (``end <= lo`` or
            ``start >= hi``) is NOT clipped -- the span never touched it, so it is
            a complete measurement; see the ``outside`` clause below.
            Defaults to ``(0, len(current))``, which is right for any caller that
            hands over the whole analysed array (both detectors). A caller that
            characterizes inside a PADDED WINDOW cut out of a longer recording
            must translate: ``(0 - lo_window, n_recording - lo_window)``, so a
            window edge is not mistaken for a recording edge --
            :func:`recharacterize_span` does exactly that.
        inherit_flag: Flag token(s) carried over from the row(s) this one replaces
            (scalar, or one per event). Unioned with the flags derived here.
        inherit_accepted: The matching curation decision(s) to carry over
            (scalar, or one per event). Combined with AND: a rebuilt row is
            accepted only if the rows it replaces were AND the new measurement
            passes. Both come from :func:`~nanoscope.analysis.events.inherited_curation`.
        pore_id: Physical pore identifier. Falls back to `recording`. Never pool
            events across pores.
        recording: Source recording name, stamped into the `recording` column and
            used as the `pore_id` fallback.
        units: Signal units label (informational; the SCHEMA column names keep their
            `_pA` suffix as the frozen cross-module contract regardless).
        cancel: Zero-arg predicate polled inside the two per-event loops
            (`rise_fall_over_set`, `_count_levels`) -- the pure-Python part of this module,
            and the only part where a cancel can land. `None` uses the AMBIENT token
            (`analysis.cancel.cancel_scope`), which is how a run started from the GUI
            worker becomes cancellable without this argument being passed anywhere.
            Raises `analysis.cancel.OperationCancelled`, so a cancelled run returns no
            partial table at all.

    Returns:
        DataFrame with exactly the SCHEMA columns and dtypes, one row per event.

    Gotchas:
        * `depth_pA` is the MEDIAN in-event deviation, not the max -- see the module
          docstring. This materially changes reported dI/I0.
        * `rise_s`/`fall_s` are NaN when the event never reaches 90% of its own
          level inside the search window (typical for filter-limited events). NaN
          means "not measurable here", which is the truth; it is not zero.
        * `ecd_pA_s` integrates the deviation over [start, end) only. It therefore
          inherits whatever the detector chose for the event edges, and shrinks if
          you tighten `k_out`.
    """
    fs = _validate_fs(fs)
    current = np.asarray(current, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64).ravel()
    ends = np.asarray(ends, dtype=np.int64).ravel()

    if baseline.shape != current.shape:
        raise ValueError(
            f"baseline length {baseline.shape} != current length {current.shape}; "
            "estimate_baseline() returns a full-length array."
        )
    if starts.size != ends.size:
        raise ValueError(f"starts ({starts.size}) and ends ({ends.size}) differ in length")
    if not np.isfinite(noise_rms) or noise_rms <= 0:
        raise ValueError(f"noise_rms must be finite and > 0, got {noise_rms!r}; snr would be meaningless")

    resolved_pore = pore_id or recording

    if starts.size == 0:
        return empty_df()

    if np.any(ends <= starts):
        bad = int(np.argmax(ends <= starts))
        raise ValueError(f"event {bad} has end ({ends[bad]}) <= start ({starts[bad]}): zero-length event")
    if starts[0] < 0 or ends[-1] > current.size:
        raise ValueError(
            f"events span [{starts.min()}, {ends.max()}) which falls outside the trace [0, {current.size})"
        )
    if np.any(np.diff(starts) < 0):
        raise ValueError("starts must be ascending")
    if np.any(starts[1:] < ends[:-1]):
        bad = int(np.argmax(starts[1:] < ends[:-1]))
        raise ValueError(
            f"events {bad} and {bad + 1} overlap ([{starts[bad]}, {ends[bad]}) vs "
            f"[{starts[bad + 1]}, {ends[bad + 1]})); per-event features would double-count samples"
        )

    pol = int(polarity)
    n_ev = starts.size
    lengths = (ends - starts).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))

    # Gather every in-event sample into one flat array so the per-event reductions
    # are reduceat calls rather than a Python loop over samples.
    total = int(lengths.sum())
    # Position of each gathered sample WITHIN its own event, kept because the
    # half-max width below needs it as well as the gather does.
    pos_in_event = np.arange(total, dtype=np.int64) - np.repeat(offsets, lengths)
    flat_idx = pos_in_event + np.repeat(starts, lengths)
    cur_all = current[flat_idx]
    dev_all = pol * (cur_all - baseline[flat_idx])
    base_all = baseline[flat_idx]

    sum_dev = np.add.reduceat(dev_all, offsets)
    mean_dev = sum_dev / lengths
    max_dev = np.maximum.reduceat(dev_all, offsets)
    baseline_pA = np.add.reduceat(base_all, offsets) / lengths

    depth_pA = _grouped_median(dev_all, offsets, lengths)

    dwell_s = lengths / fs
    ecd_pA_s = sum_dev / fs

    abs_base = np.abs(baseline_pA)
    frac_depth = np.divide(
        depth_pA, abs_base, out=np.full(n_ev, np.nan), where=abs_base > 0
    )

    snr = depth_pA / float(noise_rms)

    # Which events were measured from a NaN/Inf sample -- half of the non-finite
    # rule below, computed here because the sub-state fit must SKIP those events:
    # its prefix sums over an Inf are ``inf - inf``, i.e. a NaN split gain and a
    # numpy RuntimeWarning per event, and there is no meaningful number of
    # conformational sub-states in a poisoned window anyway.
    finite_sample = np.isfinite(dev_all) & np.isfinite(base_all)
    bad_samples = np.add.reduceat(finite_sample.astype(np.int64), offsets) < lengths

    half_max_width_s = _half_max_width_s(
        dev_all, max_dev, pos_in_event, offsets, lengths, dwell_s, fs
    )

    rise_s, fall_s = rise_fall_over_set(current, baseline, pol, starts, ends, depth_pA, fs, cancel=cancel)
    n_levels = _count_levels(dev_all, offsets, lengths, cancel=cancel, skip=bad_samples)

    if filter_rise_s is None:
        filter_limited = np.zeros(n_ev, dtype=bool)
    else:
        if not np.isfinite(filter_rise_s) or filter_rise_s < 0:
            raise ValueError(f"filter_rise_s must be finite and >= 0, got {filter_rise_s!r}")
        # AGAINST THE WIDTH THE FILTER DID NOT STRETCH, not against `dwell_s`.
        # `dwell_s` is measured on the FILTERED trace, and this module's own
        # documentation says that quantity is corrupted by the very filter being
        # tested for (`filters.lowpass` Gotcha 1: "dwell times of short events are
        # biased LONG"). Testing it against 2*rise let the filter carry an event
        # over the line precisely when the filter was destroying it -- measured,
        # 20 square 200 us events at 100 kHz, depth 100 pA, noise 10 pA:
        #
        #   cutoff  rise   2*rise   true dwell   MEASURED dwell   depth   old flag
        #    2 kHz  241 us  481 us    200 us         545 us      35.7 pA   0 / 20
        #    5 kHz   97 us  193 us    200 us         320 us      68.6 pA   0 / 20
        #
        # -- at 2 kHz the true dwell is 0.42x the boundary and NOTHING was flagged,
        # so a row 64% short in depth went to the fits as resolved (both
        # `dwell_fit_inputs` and the Analysis panel drop filter-limited rows by
        # default, so these are exactly the rows a fit keeps). The inflation is not
        # a fixed offset either: it grows with SNR, so at 2 pA noise the same events
        # measured 630 us and at 20 pA noise 795 us -- which is why "subtract a rise
        # time" cannot be the fix and the flag has to be judged on a different
        # quantity.
        #
        # `half_max_width_s` is that quantity (see the helper): the width at half the
        # event's OWN peak, which a symmetric zero-phase filter leaves at the true
        # duration for a resolved event and floors at its own kernel width for an
        # unresolved one. Same sweep, same rows: 20/20 flagged at 2 kHz, 0/20 at
        # 5 kHz (where the noiseless plateau does reach 99.8 of 100 pA, i.e. the
        # event really is resolved) -- and the 5 ms events in the same fixture stay
        # unflagged at every cutoff. It is never LONGER than `dwell_s`, so every row
        # the old rule flagged is still flagged.
        filter_limited = half_max_width_s < 2.0 * float(filter_rise_s)

    # -- the four automatic rejection rules, applied HERE and only here -------
    # ``> 0`` is False for NaN as well as for 0, and a NaN baseline is not a ZERO
    # baseline: saying so would put a false reason on the row (and, before the
    # non-finite rule below, was the ONLY thing said about it). The finiteness
    # clause keeps this token meaning exactly what it says.
    zero_baseline = np.isfinite(baseline_pA) & ~(abs_base > 0)

    lo_edge, hi_edge = (0, int(current.size)) if clip_bounds is None else (
        int(clip_bounds[0]), int(clip_bounds[1])
    )
    # ``clipped_by_boundary`` is a relation between a row and the span the detector
    # LOOKED AT: the event was still above threshold at the first or last analysed
    # sample, so its dwell is a lower bound set by where the span was drawn. That is
    # only a statement about a row the span could have cut, i.e. one that OVERLAPS it.
    #
    # A row that lies wholly outside the span was not cut by anything -- it was
    # measured, end to end, from samples the span never touched. The case is real:
    # run the ML detector on a zoom rectangle, then drag the measure cursors over a
    # translocation somewhere else in the recording and press "Mark selection as
    # missed event". Before this second clause that row came back
    # ``accepted=False, flag='clipped_by_boundary'`` (measured: a clean 4.00 ms,
    # 298 pA event at [5000, 5200) against a span of [10000, 20000)) -- the user's
    # explicit "this IS an event" overridden by a reason that was not true of it,
    # and counted in ``n_clipped_boundary``.
    outside = (ends <= lo_edge) | (starts >= hi_edge)
    clipped = ((starts <= lo_edge) | (ends >= hi_edge)) & ~outside

    # A row measured from NaN/Inf is not a measurement. The rule is HERE with the
    # other three, so the merge, the Event Inspector resize, the hand-added event
    # and both detectors all inherit it -- the load boundary already reports a
    # non-finite FILE (``TraceDocument.__post_init__``), but a poisoned sample can
    # also arrive from a calibration overflow, a detrend, or a flank read outside
    # the event, and none of those cross that boundary.
    #
    # Two questions, because either alone is blind to half the cases: are the
    # SAMPLES this row was measured from finite (``bad_samples``, computed above --
    # a mid-event Inf whose effect the median then hides: measured depth 20.135567
    # against a clean 20.140960, while ecd and mean_dev were -inf), and are the
    # MAGNITUDES finite (a baseline read from a poisoned flank, which puts no bad
    # sample inside the event at all). ``frac_depth`` is excluded: NaN there is the
    # *documented* answer for ``zero_baseline``, which has its own token.
    # ``rise_s``/``fall_s`` are excluded for the same reason -- NaN means "the event
    # never reached 90 % in the search window", which is the truth, not a defect.
    non_finite = bad_samples | ~(
        np.isfinite(baseline_pA) & np.isfinite(depth_pA) & np.isfinite(ecd_pA_s)
        & np.isfinite(max_dev) & np.isfinite(mean_dev) & np.isfinite(snr)
    )

    if max_dwell_s is None:
        too_long = np.zeros(n_ev, dtype=bool)
    else:
        # A clipped event's dwell is not a measurement of anything, so
        # 'dwell>max_dwell_s' would be a FALSE explanation of why the row was
        # rejected; the boundary reason is the true and complete one.
        too_long = (dwell_s > float(max_dwell_s)) & ~clipped

    # Built so the only Python-level work is over the rows that are actually FLAGGED
    # (a handful) and, for the union, only when a caller passes something to inherit --
    # which neither detector does. Measured on the owner's 1738-event -600 mV
    # detection, all four rules armed: characterize 83.21 -> 81.82 ms (within the
    # run-to-run spread; the non-finite rule is two reduceat passes over the flat
    # in-event array, which the medians above already dominate).
    # A depth read off a flat top is a floor, not a measurement. The row is KEPT and
    # accepted -- the event happened, its dwell is a real number -- but it says so,
    # the way ``clipped_by_boundary`` says it for the time axis. See FLAG_SATURATED.
    saturated = _saturated_events(cur_all, current, pol, pos_in_event, offsets, lengths)

    flag = np.where(zero_baseline, FLAG_ZERO_BASELINE, "").astype(object)
    if saturated.any():
        flag[saturated] = [combine_flags(f, FLAG_SATURATED) for f in flag[saturated]]
    if too_long.any():
        flag[too_long] = [combine_flags(f, FLAG_MAX_DWELL) for f in flag[too_long]]
    if clipped.any():
        flag[clipped] = [combine_flags(f, FLAG_CLIPPED) for f in flag[clipped]]
    if non_finite.any():
        flag[non_finite] = [combine_flags(f, FLAG_NON_FINITE) for f in flag[non_finite]]
    accepted = ~(too_long | clipped | non_finite)

    if inherit_flag is not None:
        carried = _broadcast(inherit_flag, n_ev, "", "inherit_flag")
        flag = np.array(
            [combine_flags(flag[i], carried[i]) for i in range(n_ev)], dtype=object
        )
    if inherit_accepted is not None:
        keep = _broadcast(inherit_accepted, n_ev, True, "inherit_accepted")
        accepted = accepted & np.array([bool(v) for v in keep], dtype=bool)

    df = pd.DataFrame(
        {
            "event_id": np.arange(n_ev, dtype=np.int64),
            "recording": recording,
            "pore_id": str(resolved_pore),
            "start_idx": starts,
            "end_idx": ends,
            "t_start_s": starts / fs,
            "dwell_s": dwell_s,
            "baseline_pA": baseline_pA,
            "depth_pA": depth_pA,
            "frac_depth": frac_depth,
            "ecd_pA_s": ecd_pA_s,
            "max_dev_pA": max_dev,
            "mean_dev_pA": mean_dev,
            "rise_s": rise_s,
            "fall_s": fall_s,
            "snr": snr,
            "n_levels": n_levels,
            "filter_limited": filter_limited,
            "accepted": accepted,
            "flag": flag,
        }
    )
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})[list(SCHEMA)]


def _broadcast(value: Any, n: int, default: Any, name: str) -> list[Any]:
    """One value per event, from a scalar, a sequence, or None."""
    if value is None:
        return [default] * n
    if isinstance(value, (str, bool, np.bool_)):
        return [value] * n
    try:
        seq = list(value)
    except TypeError:
        return [value] * n
    if len(seq) == 1:
        return seq * n
    if len(seq) != n:
        raise ValueError(f"{name} has {len(seq)} entries for {n} event(s)")
    return seq


# ===========================================================================
# THE ONE ROW-REBUILD PATH
# ===========================================================================
# A gap merge, an Event Inspector mask resize and a hand-added missed event all
# do the same thing: characterize one span against a LOCAL open-pore baseline,
# read out of a padded window, and put the result where a detected row used to
# be. That was written three times -- `merge._characterize_span`,
# `event_inspector._recharacterize_row`, `analysis_panel._characterize_missed` --
# with each copy preserving something slightly different, which is exactly how
# C1 (merge OR-ed `accepted` and blanked `flag`) and C21 (the resize blanked
# `flag` but kept `accepted=False`) were the SAME defect wearing two costumes.
#
# It is written once here. A fourth rebuild path gets the guarantee by calling
# this; it cannot get it wrong by forgetting an argument, because the arguments
# it would have to forget do not exist.

#: A few ms of open-pore context each side of a rebuilt span, so the local
#: baseline / noise have event-free flanks to read and rise/fall have room.
RECHAR_PAD_MS = 3.0


def recharacterize_span(
    signal: np.ndarray,
    fs: float,
    start: int,
    end: int,
    *,
    provenance: Mapping[str, Any] | None = None,
    prior: Any = None,
    polarity: int | None = None,
    baseline: np.ndarray | float | None = None,
    baseline_config: Any = None,
    recording: str = "",
    pore_id: str = "",
    units: str = "pA",
    pad_ms: float = RECHAR_PAD_MS,
) -> pd.DataFrame:
    """Rebuild ONE schema row for the span ``[start, end)``, curation included.

    A window padded around the span stands in for the whole trace: the noise scale
    comes from :func:`span_noise_rms`, and :func:`characterize` produces the exact
    row the detector would -- including the four automatic rejection rules, judged
    against the ANALYSED SPAN's edges rather than the padded window's.

    THE BASELINE COMES FROM THE DETECTION, NOT FROM THE WINDOW
    ----------------------------------------------------------
    This was the fourth of four places that estimated a characterisation baseline,
    at a fourth setting: the robust median of the event-free flanks, over
    ``+-3 ms``. That is *better* than a 10 ms block median inside an event -- the
    flanks are event-free by construction -- and still wrong for the same reason
    all four were wrong: it is not the same measurement its neighbours got. One
    event on the owner's -600 mV recording read **32.04 pA** through the Schmitt
    detector, **37.54 pA** through this function, and **67.69 pA** through the ML
    characteriser. A merge or a mask drag therefore moved a depth by 17% with no
    edit to the data at all, and on a file whose 521 ms partial clog swallows a
    3 ms flank the error is far larger.

    Resolution order, most exact first, recorded in ``df.attrs['baseline_source']``:

    1. ``'provided'`` -- a full-length ``baseline`` array was passed; it is sliced.
    2. ``'recording'`` -- ``provenance`` names a reproducible estimator
       (:meth:`~nanoscope.analysis.baseline.BaselineConfig.reproducible_from`), so
       the whole-recording baseline is re-derived from the rows' OWN stored
       configuration and sliced. Memoised on the trace object by
       :func:`~nanoscope.analysis.baseline.characterisation_baseline`, so a merge
       of hundreds of runs pays for one estimate, not hundreds.
    3. ``'flanks'`` -- the pre-change local median. Reached when no baseline
       travels with the rows and none can be named: a row detected before this
       change, or one whose detector was handed a baseline array by its caller.
       Guessing an estimator for those rows would be worse than measuring locally.
    4. ``'non_finite'`` -- the pad contains a NaN/Inf, so the row is deliberately
       poisoned (see the long comment below). This outranks 1-3: a rebuild whose
       own measurement window is corrupt must say so, not read around it.

    Note the asymmetry in (2): the config comes from the ROW's provenance, not from
    the live configuration. A rebuild must reproduce what measured the row; if the
    user has since changed the estimator, that is a STALENESS question the
    ``baseline_fingerprint`` answers, not something a merge should silently apply.

    THE SPAN COMES OUT OF PROVENANCE, NOT OUT OF AN ARGUMENT
    --------------------------------------------------------
    ``clipped_by_boundary`` is a relation between a row and the span the detector
    looked at, so a rebuild that does not know the span cannot re-derive it. This
    function used to take an ``n_total`` argument for exactly that and **no caller
    anywhere passed it**, so every rebuild judged the row against the whole
    recording: a row the ML detector had correctly flagged as cut by the user's
    zoom rectangle found itself nowhere near an edge and -- the token being
    DERIVED -- dropped it and un-rejected itself, on a plain gap merge, with no
    edit at all.

    The span now travels WITH THE ROWS, in the EventSet's own provenance
    (:func:`~nanoscope.analysis.events.span_provenance` writes it, both detectors
    stamp it, :func:`~nanoscope.analysis.events.analysed_span` reads it). Every
    rebuild path already hands this function that provenance -- it is where
    ``noise_rms_pA``, ``max_dwell_s``, ``filter_rise_s`` and the polarity already
    come from -- so a fourth rebuild path gets the boundary judgement right
    without passing anything, and cannot get it wrong by forgetting an argument
    that no longer exists. What that buys is the property the defect broke: a
    row's boundary judgement is STABLE under any number of rebuilds.

    What ``prior`` buys you is the half that used to be lost. The rows this one
    replaces are put through
    :func:`~nanoscope.analysis.events.inherited_curation`: a curator's rejection
    and a curator's own flag tokens carry over (``accepted`` combined with AND,
    never OR), while an automatic rejection is re-judged from the new span. Pass
    ``prior=None`` only when the row genuinely replaces nothing -- a hand-added
    missed event.

    Args:
        signal: The trace the detection measured on, FULL length (the indices are
            absolute positions in it).
        fs: Sample rate in Hz.
        start, end: The new span, half-open, in absolute samples.
        provenance: The detection's provenance. Supplies ``noise_rms_pA`` (so the
            rebuilt row's SNR is on the same scale as its neighbours'),
            ``filter_rise_s``, ``max_dwell_s``, THE ANALYSED SPAN, and -- unless
            ``polarity`` is given -- the polarity. Omitting it is what a rebuild
            path must never do; ``tests/analysis/test_flag_invariant.py`` has an
            AST guard that says so.
        prior: The row(s) being replaced -- a DataFrame, a Series, a mapping, or
            None. Only ``flag`` and ``accepted`` are read.
        polarity: Overrides the provenance polarity. ``None`` falls back to
            provenance and then to one measured from the window.
        baseline: The FULL-LENGTH baseline the detection already computed (or a
            scalar). Given, it is simply SLICED -- see the section above. This is
            the cheapest and most exact way for a caller that already holds it
            (``merge`` computes it once for a whole table) to guarantee the
            rebuilt row lands on the detector's own numbers.
        baseline_config: Override the estimator read out of ``provenance``. Rarely
            wanted; the point of reading it from provenance is that the rebuild
            reproduces what MEASURED the row, not what is configured now.
        recording, pore_id, units: Stamped into the row.
        pad_ms: Flank width, in ms, either side of the span.

    Returns:
        A 1-row SCHEMA DataFrame whose ``start_idx`` / ``end_idx`` / ``t_start_s``
        are ABSOLUTE (characterize measured them inside the local window).

        EXACTLY THE FROZEN 20, AND NOT ONE COLUMN MORE. This function ends inside
        :func:`characterize`, which ends with ``df[list(SCHEMA)]``; every column the
        app declares OUTSIDE ``SCHEMA`` -- ``scoring.ADDED_COLUMN``,
        `events.ORIGINAL_BOUNDS_SCHEMA`, `events.SOURCE_STATUS_SCHEMA` -- is therefore
        absent from what comes back, and that is deliberate rather than incidental. It
        is the same mechanism that lets a project saved before any of those columns
        existed still load, and it is why the CALLER re-applies them:

        * `merge._merged_row` combines them across the run it is collapsing (a merged
          row has more than one author, and `events.combine_sources` is what that
          means);
        * `remeasure.remeasure_events` copies them from the row being replaced, since
          a re-measure rewrites amplitudes and cannot change who found the event;
        * the Event Inspector's mask drag writes the MEASURED columns back BY NAME
          onto the shared frame, so an untouched column survives the edit;
        * the hand-added-event path passes ``prior=None`` -- the row replaces nothing
          -- and stamps ``manual`` itself.

        Carrying the pair HERE instead would look tidier and would be wrong for the
        bounds: ``orig_start_idx`` must never be written by a rebuild, because a value
        written after an edit records the edit under the detector's name. One rule for
        all four non-schema columns is cheaper to keep true than one rule per column,
        and `events.carry_source_status` is the named door for any future caller that
        wants the re-application in one call rather than a loop.
    """
    fs = _validate_fs(fs)
    sig = np.asarray(signal)
    n = int(sig.size)
    # The span the DETECTOR analysed, in the rows' own coordinates. Read from the
    # provenance that already travels with them; ``(0, n)`` only when nothing
    # carries a span, which is the right answer for every whole-recording
    # detection and is what this did before the span existed.
    # There is deliberately NO ``span=`` override parameter. One existed briefly and
    # no caller ever passed it, which is exactly the hazard this design removes: an
    # override reintroduces the argument a fourth rebuild path can pass wrongly (or
    # forget), and it bypasses ``analysed_span``'s refusal of an empty or reversed
    # pair -- ``span=(0, 0)`` was used verbatim and flagged every row.
    span_lo, span_hi = analysed_span(provenance, n)
    start = int(start)
    end = int(end)

    ev_len = max(end - start, 1)
    few = int(round(float(fs) * float(pad_ms) / 1000.0)) if fs else 200
    pad = max(int(ev_len), int(few), 1)
    lo = int(np.clip(start - pad, 0, n))
    hi = int(np.clip(end + pad, 0, n))

    window = np.asarray(sig[lo:hi], dtype=np.float64)
    left = np.asarray(sig[lo:start], dtype=np.float64)
    right = np.asarray(sig[end:hi], dtype=np.float64)
    flanks = np.concatenate([left, right]) if (left.size or right.size) else window

    # THE PADDED WINDOW IS PART OF WHAT THIS ROW WAS MEASURED FROM.
    # Unlike a detector -- which gets its baseline from `estimate_baseline` and its
    # noise from `filters.noise_rms`, both of which REFUSE a non-finite trace outright
    # -- a rebuild reads its open-pore reference out of these flanks and nothing else.
    # So a non-finite sample in the pad is not "near" the row, it is INSIDE its
    # measurement, and it has to taint it. Today's behaviour splits on which
    # non-finite value it is, which is the tell:
    #
    #   flank NaN  -> median NaN -> baseline/depth/ecd NaN -> already flagged
    #   flank +Inf -> median is an ORDER STATISTIC and reads straight through it, so
    #                 the baseline is fine; but `filters.noise_rms` refuses the same
    #                 window, `span_noise_rms` walks its whole ladder and returns the
    #                 1e-9 epsilon, and the row lands with snr = -2.0e10 -- finite,
    #                 accepted, unflagged. (Measured: one +inf in the pad, provenance
    #                 without `noise_rms_pA`. This is the "snr = 19.8e9" report §2.3
    #                 could not reproduce; it needs provenance to be MISSING the
    #                 detector noise, which is the hand-added-event and
    #                 legacy-EventSet case.)
    #
    # Blanking the local baseline is how that reaches the rule: `characterize` then
    # sees a non-finite `baseline` array over the event's own samples and applies the
    # SAME `non_finite` token by the SAME clause as every other poisoned row. No new
    # argument, no second rule, and NaN and Inf stop disagreeing. A sample outside this
    # window is untouched -- one bad sample must cost the rows around it, not the table.
    poisoned = flanks.size and not bool(np.isfinite(flanks).all())

    noise = span_noise_rms(left, right, window, provenance)
    pol = _resolve_polarity(polarity, provenance, window)
    baseline_win, baseline_source = _rebuild_baseline(
        sig, window, flanks, lo, hi, fs=fs, polarity=pol, poisoned=bool(poisoned),
        baseline=baseline, baseline_config=baseline_config, provenance=provenance,
    )
    filter_rise_s = _provenance_float(provenance, "filter_rise_s")
    max_dwell_s = _provenance_float(provenance, "max_dwell_s")

    # ``max_dwell_s`` is only RE-DERIVABLE when the limit is known. When it is not,
    # its token has to be inherited or the reason for a rejection is destroyed.
    #
    derived = DERIVED_FLAGS if max_dwell_s is not None else (DERIVED_FLAGS - {FLAG_MAX_DWELL})
    inherit_flag, inherit_accepted = _prior_curation(prior, derived)

    out = characterize(
        window,
        baseline_win,
        np.array([start - lo], dtype=np.int64),
        np.array([end - lo], dtype=np.int64),
        float(fs),
        int(pol),
        noise_rms=noise,
        filter_rise_s=filter_rise_s,
        max_dwell_s=max_dwell_s,
        # The ANALYSED SPAN's edges, expressed in this window's coordinates.
        # Without the translation every padded window would look clipped at both
        # ends; without the span coming from provenance, a row cut by a zoom
        # rectangle would be re-judged against the whole recording and quietly
        # stop saying so.
        clip_bounds=(span_lo - lo, span_hi - lo),
        inherit_flag=inherit_flag,
        inherit_accepted=inherit_accepted,
        recording=recording,
        pore_id=pore_id,
        units=units,
    )
    idx = out.index[0]
    out.at[idx, "start_idx"] = int(start)
    out.at[idx, "end_idx"] = int(end)
    out.at[idx, "t_start_s"] = float(start) / float(fs)
    # WHICH open-pore reference measured this row. Not a column -- the 20-column
    # SCHEMA is frozen -- and not a warning, because on the ``'recording'`` path
    # there is nothing to warn about. It is here so a caller can tell a rebuilt row
    # that landed on the detector's own baseline from one that had to fall back.
    out.attrs["baseline_source"] = baseline_source
    if (provenance or {}).get("fractional_blockage_reference_unavailable") is True:
        out["frac_depth"] = np.nan
    return out


#: The four open-pore references a rebuild can end up on, most exact first.
REBUILD_BASELINE_SOURCES = ("provided", "recording", "flanks", "non_finite")


def _rebuild_baseline(
    sig: np.ndarray,
    window: np.ndarray,
    flanks: np.ndarray,
    lo: int,
    hi: int,
    *,
    fs: float,
    polarity: int,
    poisoned: bool,
    baseline: Any,
    baseline_config: Any,
    provenance: Mapping[str, Any] | None,
) -> tuple[np.ndarray, str]:
    """The padded window's baseline, and which of `REBUILD_BASELINE_SOURCES` it is.

    Written as one function, because "which baseline does a rebuilt row get" is
    exactly the question three rebuild paths used to answer three ways.

    ``poisoned`` wins outright. A non-finite sample inside the pad is INSIDE this
    row's measurement, and blanking the local baseline is how that reaches
    :func:`characterize`'s ``non_finite`` rule -- the same rule, by the same clause,
    as every other poisoned row. Reading a clean whole-recording baseline instead
    would launder a corrupt window into a finite, accepted, unflagged row, which is
    the precise failure the poisoning was written to stop.

    Everything else degrades rather than raises: a rebuild is a UI action on one
    event, and a distant bad sample or a recording too short for the configured
    estimator must cost that row its exactness, never the edit.
    """
    n_win = int(window.size)
    if poisoned:
        return np.full(n_win, float("nan"), dtype=np.float64), "non_finite"

    if baseline is not None:
        arr = np.asarray(baseline, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(n_win, float(arr), dtype=np.float64), "provided"
        if arr.size == int(np.asarray(sig).size):
            return np.asarray(arr[lo:hi], dtype=np.float64), "provided"
        if arr.size == n_win:
            return arr, "provided"
        raise ValueError(
            f"baseline has {arr.size} samples; expected a scalar, the window's "
            f"{n_win}, or the trace's {np.asarray(sig).size}."
        )

    from .baseline import BaselineConfig, characterisation_baseline

    cfg = baseline_config
    if cfg is None:
        cfg = BaselineConfig.reproducible_from(provenance)
    if cfg is not None:
        try:
            if getattr(cfg, "open_pore_reference", None) is not None:
                from .reference_context import validate_open_pore_context
                recorded = dict(provenance or {})
                validate_open_pore_context(
                    {**recorded, "baseline_config": cfg},
                    units=recorded.get("units"), module_id="detect.schmitt")
            res = characterisation_baseline(
                sig, float(fs), config=cfg, polarity=int(polarity), health=False,
                # A whole-recording estimate that REFUSES on one bad sample anywhere
                # turns every rebuilt row in the table into a fallback measurement --
                # the blast radius going from "the pad" to "the table", which is the
                # one thing this module's non-finite rule exists to prevent.
                # `repair_nonfinite` interpolates the defect away for the ESTIMATE
                # only; rows whose own pad contains it are still poisoned above, by
                # the flank rule, and rows further than half the tracking window away
                # are byte-identical because a rolling median is local.
                on_nonfinite="interpolate",
            )
        except (ValueError, TypeError):
            if getattr(cfg, "open_pore_reference", None) is not None:
                # A user reference is a measurement instruction, not a hint to
                # replace silently with a different local reference.
                raise
            # The trace these rows were detected on is not the trace we were handed
            # (a shorter conditioned slice, a non-finite sample far outside the pad,
            # a rate the config cannot run at). Fall through to the local median --
            # honestly labelled -- rather than refusing to rebuild the row.
            pass
        else:
            return np.asarray(res.baseline[lo:hi], dtype=np.float64), "recording"

    scalar = float(np.median(flanks)) if flanks.size else 0.0
    return np.full(n_win, scalar, dtype=np.float64), "flanks"


def _prior_curation(prior: Any, derived: frozenset[str]) -> tuple[str, bool]:
    """``(flag, accepted)`` to carry over from the row(s) a rebuild replaces."""
    if prior is None:
        return "", True
    if isinstance(prior, pd.DataFrame):
        if not len(prior):
            return "", True
        return inherited_curation(
            prior["flag"].tolist(), prior["accepted"].tolist(), derived=derived
        )
    if isinstance(prior, pd.Series):
        return inherited_curation(prior.get("flag", ""), prior.get("accepted", True),
                                  derived=derived)
    if isinstance(prior, Mapping):
        return inherited_curation(prior.get("flag", ""), prior.get("accepted", True),
                                  derived=derived)
    raise TypeError(f"prior must be a DataFrame, Series, Mapping or None, got {type(prior)!r}")


def _resolve_polarity(
    polarity: int | None, provenance: Mapping[str, Any] | None, window: np.ndarray
) -> int:
    """Explicit polarity, else the detector's, else one measured from the window."""
    if polarity in (-1, 1):
        return int(polarity)  # type: ignore[arg-type]
    if provenance is not None:
        p = provenance.get("polarity")
        if p in (-1, 1):
            return int(p)  # type: ignore[arg-type]
    from .baseline import derive_polarity  # local: baseline is a heavier import

    try:
        return int(derive_polarity(np.asarray(window, dtype=np.float64)))
    except ValueError:
        return -1


def _provenance_float(provenance: Mapping[str, Any] | None, key: str) -> float | None:
    """A finite float from provenance, or None when absent/unusable."""
    if provenance is None:
        return None
    value = provenance.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def saturated_warning(n_saturated: int, n_events: int) -> str:
    """The one sentence both detectors print about flat-topped (railed) events.

    Written here, beside the rule, for the same reason
    :func:`~nanoscope.analysis.events.clipped_boundary_warning` is written beside its
    own: one wording, one place, so the Stats panel and any other surface cannot
    describe the same rows differently.
    """
    return (
        f"{n_saturated} of {n_events} event(s) run into a flat rail: "
        f"{_SATURATION_RUN}+ consecutive samples sit on exactly the extreme value of "
        "the analysed trace, so the amplifier or ADC clipped them. Their depth, "
        f"frac_depth and ecd are LOWER BOUNDS, not measurements -- flagged "
        f"'{FLAG_SATURATED}'. Lower the gain or the bias and record again if those "
        "depths matter."
    )


def _saturated_events(
    cur_all: np.ndarray,
    current: np.ndarray,
    pol: int,
    pos_in_event: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Which events were measured off a flat top -- see :data:`FLAG_SATURATED`.

    The evidence is already in the samples the characteriser walks: a run of
    :data:`_SATURATION_RUN` consecutive in-event samples holding EXACTLY the extreme
    value of the analysed array, on the side events move toward (``pol``). Exact
    equality is the whole point -- a rail is the one thing in a noisy ionic trace that
    produces a repeated bit pattern -- and tying it to the extreme of the whole array
    rather than to the event's own minimum is what keeps a deep, well-measured event
    from calling itself clipped.

    Vectorised on the same flat in-event array as the medians: a windowed sum over the
    at-rail mask, then one ``reduceat``. The window is required to lie inside a single
    event (``pos_in_event >= run - 1``), so two events touching end-to-start cannot
    lend each other a run.
    """
    n_ev = lengths.size
    total = cur_all.size
    run = _SATURATION_RUN
    if total < run or current.size == 0:
        return np.zeros(n_ev, dtype=bool)
    finite = current[np.isfinite(current)]
    if finite.size == 0:
        return np.zeros(n_ev, dtype=bool)
    # ``dev = pol * (current - baseline)`` is positive inside an event, so the events
    # move in the direction of ``pol``: a blockade on a positive open pore measures
    # ``pol = -1`` and rails against the MINIMUM of the trace.
    rail = float(finite.min()) if pol < 0 else float(finite.max())

    # ...BUT ONLY WHERE A REPEATED VALUE IS EVIDENCE OF ANYTHING. The whole rule rests
    # on the sentence in :data:`_SATURATION_RUN`: ionic current plus thermal noise does
    # not land on the same float three times running. On a PIECEWISE-CONSTANT trace that
    # premise is simply false -- a synthetic square, an idealised fixture, a trace someone
    # simulated to check their analysis -- and every flat top then reads as a rail. The
    # symptom: a noiseless 300 pA square, the deepest event in its own trace and nowhere
    # near an amplifier limit, came back flagged `saturated`, which would have told a user
    # modelling their pore that their simulator had clipped. A rail is a claim about the
    # INSTRUMENT, so it is only made about a trace that looks like one came out of it:
    # if most adjacent samples are bit-identical there is no noise here to be interrupted,
    # and this check has nothing to say. Real recordings are nowhere near the bound (the
    # owner's four apo ABFs run 0.00-0.02 by this measure), so it costs them nothing.
    step = max(1, finite.size // 65536)          # bounded work on a 37 M-sample trace
    probe = finite[::step]
    if probe.size >= 2:
        flat_frac = float(np.mean(probe[1:] == probe[:-1]))
        if flat_frac > 0.5:
            return np.zeros(n_ev, dtype=bool)

    at_rail = cur_all == rail
    counts = np.concatenate(([0], np.cumsum(at_rail, dtype=np.int64)))
    stop = np.arange(run, total + 1, dtype=np.int64)
    full = np.zeros(total, dtype=bool)
    # A window of `run` samples ending at index i-1, entirely at the rail and entirely
    # inside this event.
    full[run - 1 :] = (counts[stop] - counts[stop - run]) == run
    full &= pos_in_event >= (run - 1)
    return np.maximum.reduceat(full.astype(np.int8), offsets) > 0


def _half_max_width_s(
    dev_all: np.ndarray,
    max_dev: np.ndarray,
    pos_in_event: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    dwell_s: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Each event's full width at half its OWN peak deviation, in seconds.

    THE DWELL THE ANALYSIS FILTER HAS NOT ALREADY STRETCHED, and the only reason
    it exists. ``dwell_s`` is the distance between the detector's exit-threshold
    crossings on the FILTERED trace, so a low-pass inflates it twice over: it
    smears the event's edges outward, and it crushes the white noise, which lowers
    the exit threshold ``k_out * noise`` and keeps the smeared tails "inside" the
    event for longer still. Both push the measured dwell up, by an amount that
    depends on the SNR rather than on the filter alone.

    A width measured at half the event's own peak does not move that way. For a
    symmetric zero-phase kernel convolved with a rectangle, the half-maximum points
    stay on the true edges while the pulse is still resolved, and once it is not,
    the width simply floors at the kernel's own width -- which is what
    ``filter_limited`` is asking about. Measured against the true 200 us / 400 us
    square events of the two fixtures in :func:`characterize`::

        true dwell   cutoff   measured dwell   half-max width
          200 us      1 kHz       935 us           480 us
          200 us      2 kHz       545 us           270 us
          200 us      5 kHz       320 us           200 us   <- resolved, and reads it
          400 us    1.5 kHz       808 us           408 us
         5000 us      1 kHz      5710 us          4990 us

    Vectorised on the same flat in-event array the medians use: one comparison,
    two ``reduceat`` passes. Never longer than the event itself, so it can only
    make :data:`filter_limited` stricter, never weaker.

    Falls back to ``dwell_s`` when the half-max level is not usable -- a peak that
    is non-finite (the row is flagged ``non_finite`` and rejected anyway) or not
    positive (a hand-drawn span containing no excursion). Refusing to guess a width
    there leaves the pre-existing rule in charge rather than inventing a number.
    """
    level = np.repeat(0.5 * max_dev, lengths)
    above = dev_all >= level
    # Sentinel values that lose their reduction when an event has no sample above
    # the level: `lengths` is larger than any in-event position, -1 smaller.
    first = np.minimum.reduceat(
        np.where(above, pos_in_event, np.repeat(lengths, lengths)), offsets
    )
    last = np.maximum.reduceat(np.where(above, pos_in_event, -1), offsets)
    width_samples = (last - first + 1).astype(np.float64)
    unusable = ~np.isfinite(max_dev) | (max_dev <= 0.0) | (width_samples <= 0.0)
    return np.where(unusable, dwell_s, width_samples / fs)


def _grouped_median(values: np.ndarray, offsets: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Median of each contiguous group in a flat array, without a Python loop.

    One lexsort orders samples by (group, value); the middle element(s) of each
    group are then a fancy-index away.
    """
    groups = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    order = np.lexsort((values, groups))
    sorted_vals = values[order]

    half = lengths // 2
    upper = sorted_vals[offsets + half]
    lower = sorted_vals[offsets + np.maximum(half - 1, 0)]
    even = (lengths % 2) == 0
    return np.where(even, 0.5 * (lower + upper), upper)


def rise_fall_over_set(
    current: np.ndarray,
    baseline: np.ndarray,
    pol: int,
    starts: np.ndarray,
    ends: np.ndarray,
    depth: np.ndarray,
    fs: float,
    *,
    cancel: CancelFlag | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """10-90% entry and 90-10% exit times, in seconds, by linear interpolation.

    Levels are 10% and 90% of that event's own depth (the median level), so a
    partially-blocked event is not measured against the wrong reference.
    Search windows are padded outside the event -- the detector triggers at
    k*sigma, which can already be above the 10% level -- but clipped to the
    neighbouring events so one event's tail never becomes another's rise.
    The pad is ``max(64 samples, 1.28 ms)`` (``_edge_pad_samples``): 64 samples
    at 50 kHz and below, so every row of a recording at or under that rate is
    unchanged, and 1.28 ms above it, where 64 samples would be a window too
    short to hold the 10 % point of an edge fitted mid-ramp.
    NaN where the event never reaches 90% inside the window.

    RISE AND FALL ARE PROPERTIES OF THE EVENT SET, NOT OF ONE EVENT
    ---------------------------------------------------------------
    ``lo_bound``/``hi_bound`` below are where that becomes load-bearing: the
    search window is clipped to the NEIGHBOURING events, so a row's crossing
    times depend on rows that are not it. A caller that rebuilds one row at a
    time therefore cannot get them right -- it presents a single event to this
    function, the neighbour bound degenerates to the window edges, and the pad
    walks into whatever sits within ``_EDGE_PAD_SAMPLES``.

    That is not hypothetical. Re-measuring the owner's 1641-event apo detection
    against its OWN stored estimator -- ``baseline_pA``, ``depth_pA``,
    ``ecd_pA_s``, ``snr`` and ``dwell_s`` all bit-identical, i.e. nothing was
    measured differently -- moved ``rise_s`` on **812** rows and ``fall_s`` on
    **920**, with 331 and 341 going from a finite value to NaN, purely from
    losing the bound. Splitting the same rows by neighbour distance is decisive:
    of the 1399 rows with a neighbour inside the pad, 812 moved; of the 83
    isolated rows (neighbour further than 4x the pad), **0** moved. Recomputing
    the whole table through this function instead restores all 20 columns to
    bit-identity.

    So this function is PUBLIC, and any path that rebuilds a whole table calls it
    once with every span -- see
    :func:`~nanoscope.analysis.remeasure.remeasure_events`. It is the only shape
    in which the answer is the one the detector would have given.
    """
    n_ev = starts.size
    rise = np.full(n_ev, np.nan)
    fall = np.full(n_ev, np.nan)

    pad = _edge_pad_samples(fs)
    lo_bound = np.concatenate(([0], ends[:-1]))
    hi_bound = np.concatenate((starts[1:], [current.size]))
    w0 = np.maximum(starts - pad, lo_bound)
    w1 = np.minimum(ends + pad, hi_bound)

    # Loop over EVENTS (thousands), not samples (millions): each iteration works
    # on a short window and the crossing search is inherently sequential. That
    # makes it one of the two places a long detection actually spends its
    # pure-Python time, so it is one of the two places a cancel must be able to
    # land -- ticked on a stride so the poll costs nothing measurable.
    poll = CancelPoller(cancel)
    for i in range(n_ev):
        poll()
        d = depth[i]
        if not np.isfinite(d) or d <= 0:
            continue
        a, b = int(w0[i]), int(w1[i])
        w = pol * (current[a:b] - baseline[a:b])
        # This window reaches OUTSIDE the event (_EDGE_PAD_SAMPLES each side), so it can
        # be poisoned by a sample the row's other magnitudes never touched -- and the two
        # non-finite values behave differently here, which is what makes it a defect
        # rather than a curiosity. Unmasked, ``inf >= hi`` is TRUE, so an Inf becomes a
        # "90 % crossing": on the rise side that gives ``rise_s = 0.0`` (measured: 0.0
        # against a clean 2.39e-4 s, with the Inf 40 samples BEFORE the event) -- an
        # instantaneous transition, physically impossible, finite, accepted, unflagged --
        # and on the fall side a ``-inf`` is likewise TRUE for ``<= lo`` and becomes a
        # false 10 % crossing that shortens ``fall_s``. ``masked`` is how that stops: a
        # non-finite sample is neither above nor below a level, so it can never BE a
        # crossing, and the crossings that remain are the real ones.
        #
        # What is then refused is the crossing time whose OWN search interval read a
        # non-finite sample -- the interval each time is measured over, and nothing
        # wider. Refusing the whole window instead (the first shape of this guard) cost
        # four numbers where one was wrong: measured on the owner's recording with one
        # Inf 20 samples before event 100, it blanked ev100's rise AND fall AND the
        # neighbouring ev99's rise and fall, where the pre-fix code had all four right,
        # because the two searches share one array and the pads of adjacent events
        # overlap. NaN is the documented value for "not measurable in this window", and
        # for the side that read the bad sample that is exactly what this is. The row is
        # NOT flagged: nothing outside [start, end) enters depth, ecd or the baseline, so
        # rejecting it would be over-reach -- see the boundary table in `characterize`.
        # One `isfinite` pass, and a clean window then pays nothing else: `finite` is
        # None on the fast path and every `&` and every interval check below drops out.
        # (Measured on the owner's 1738-event detection: characterize 83.19 -> 83.42 ms,
        # against 86.90 ms for the version that masked unconditionally.)
        finite = np.isfinite(w)
        if bool(finite.all()):
            finite = None
        hi, lo = 0.9 * d, 0.1 * d

        pick = (w >= hi) if finite is None else (finite & (w >= hi))
        above = np.flatnonzero(pick)
        if above.size == 0:
            continue
        i90, j90 = int(above[0]), int(above[-1])

        pre = (w[:i90] <= lo) if finite is None else (finite[:i90] & (w[:i90] <= lo))
        below_before = np.flatnonzero(pre)
        if below_before.size and i90 > 0:
            i10 = int(below_before[-1])
            if finite is None or finite[i10 : i90 + 1].all():
                t10 = _cross(w, i10, lo)
                t90 = _cross(w, i90 - 1, hi)
                rise[i] = max(t90 - t10, 0.0) / fs

        tail = w[j90 + 1 :]
        post = (tail <= lo) if finite is None else (finite[j90 + 1 :] & (tail <= lo))
        below_after = np.flatnonzero(post)
        if below_after.size and j90 + 1 < w.size:
            j10 = int(below_after[0]) + j90 + 1
            if finite is None or finite[j90 : j10 + 1].all():
                t90f = _cross(w, j90, hi)
                t10f = _cross(w, j10 - 1, lo)
                fall[i] = max(t10f - t90f, 0.0) / fs

    return rise, fall


def _cross(w: np.ndarray, a: int, level: float) -> float:
    """Fractional index where the segment w[a]->w[a+1] crosses `level`."""
    y0, y1 = w[a], w[a + 1]
    denom = y1 - y0
    if denom == 0:
        return float(a)
    return a + float((level - y0) / denom)


def _count_levels(
    values: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    cancel: CancelFlag | None = None,
    skip: np.ndarray | None = None,
) -> np.ndarray:
    """Number of resolved sub-states per event (1 = simple square blockade).

    Recursive binary segmentation with a BIC stop, capped at `_MAX_LEVELS`.
    A folded vs unfolded protein traverses the pore as a stepped blockade; this
    is the cheap, honest v1 detector for that.

    Gotcha: BIC assumes independent samples. A low-pass-filtered trace is
    autocorrelated, so on heavily filtered data this over-splits. Treat
    `n_levels > 1` as a flag to go look at the event, not as a measurement.

    ``skip`` marks events not to fit at all (the non-finite ones): the prefix-sum
    split search over an Inf is ``inf - inf`` per candidate cut, which is a NaN
    gain and a numpy RuntimeWarning per event. They keep the default 1, which is
    what an unfittable event has always reported.
    """
    out = np.ones(lengths.size, dtype=np.int64)
    poll = CancelPoller(cancel)   # the second per-event Python loop; see rise_fall_over_set
    for i in range(lengths.size):
        poll()
        n = int(lengths[i])
        if n < _MIN_SAMPLES_FOR_LEVELS or (skip is not None and skip[i]):
            continue
        o = int(offsets[i])
        out[i] = _segment_count(values[o : o + n])
    return out


def _segment_count(x: np.ndarray) -> int:
    """Greedy binary segmentation of a 1-D signal into piecewise-constant blocks."""
    n = x.size
    segments = [(0, n)]
    rss = _seg_rss(x, 0, n)
    bic = _bic(rss, n, 1)

    while len(segments) < _MAX_LEVELS:
        best_gain, best = -np.inf, None
        for si, (a, b) in enumerate(segments):
            cut = _best_split(x, a, b)
            if cut is None:
                continue
            m, gain = cut
            if gain > best_gain:
                best_gain, best = gain, (si, m)
        if best is None or best_gain <= 0:
            break

        si, m = best
        a, b = segments[si]
        new_rss = rss - best_gain
        new_bic = _bic(new_rss, n, len(segments) + 1)
        if new_bic >= bic:
            break
        segments = segments[:si] + [(a, m), (m, b)] + segments[si + 1 :]
        rss, bic = new_rss, new_bic

    return len(segments)


def _seg_rss(x: np.ndarray, a: int, b: int) -> float:
    seg = x[a:b]
    return float(((seg - seg.mean()) ** 2).sum())


def _best_split(x: np.ndarray, a: int, b: int) -> tuple[int, float] | None:
    """Best single split of x[a:b], as (absolute index, RSS reduction).

    Vectorized over all candidate split points via prefix sums.
    """
    n = b - a
    if n < 2 * _MIN_LEVEL_SAMPLES:
        return None
    seg = x[a:b]
    cs = np.cumsum(seg)
    cq = np.cumsum(seg * seg)

    m = np.arange(_MIN_LEVEL_SAMPLES, n - _MIN_LEVEL_SAMPLES + 1, dtype=np.int64)
    if m.size == 0:
        return None
    sl, ql = cs[m - 1], cq[m - 1]
    sr, qr = cs[-1] - sl, cq[-1] - ql
    nr = n - m
    rss_split = (ql - sl * sl / m) + (qr - sr * sr / nr)

    whole = float(cq[-1] - cs[-1] * cs[-1] / n)
    k = int(np.argmin(rss_split))
    gain = whole - float(rss_split[k])
    return a + int(m[k]), gain


def _bic(rss: float, n: int, k: int) -> float:
    """BIC of a k-segment constant model. Params: k means + (k-1) breakpoints + variance."""
    rss = max(rss, 1e-300)
    p = 2 * k
    return n * np.log(rss / n) + p * np.log(n)
