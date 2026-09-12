"""CUSUM EVENT DETECTION: find translocations with a sequential changepoint test.

READ THIS FIRST -- THIS IS NOT ``analysis.cusum``. :mod:`nanoscope.analysis.cusum`
is a LEVEL FITTER: it runs *inside* an event a detector already found and resolves
its sub-states. This module is a DETECTOR: it runs on the whole trace and decides
where the events are in the first place. Same statistic, opposite ends of the
pipeline; nothing here imports from there and nothing there imports from here.

WHAT IT IS. A one-sided sequential CUSUM (Page's test) for downward blockades, with
a state-dependent recovery test:

* **onset** -- a CUSUM of ``z - k`` against the open pore (reference level 0),
  firing when the statistic reaches ``h``. The onset SAMPLE is Page's changepoint
  estimate (the last time the statistic sat at zero), not the crossing sample, so a
  60-sample detection delay still returns the true first sample of the step.
* **recovery** -- a second CUSUM, of ``(event_level - z) - k``, i.e. the mirror test
  asking "has the trace come back to the open pore?" measured against the level the
  event was actually observed at. That level is estimated ONCE, over the detection
  delay window, and FROZEN for the event's whole life. Being referenced to the
  observed level rather than to an absolute threshold is what lets ONE ``(k, h)``
  close a 3-sigma and a 30-sigma event equally well, and it keeps a stepped event
  whole as long as its steps stay within ``k`` of the frozen level. What it does NOT
  do is track: a step that drops more than ``k`` below the frozen level reads as a
  recovery and ends the event -- see the note in `cusum_onset_recovery_regions`.
* then **fixed temporal merging** and **minimum-duration filtering**, in that order.

Provenance stamps ``detector = "cusum_sequential"``. The Schmitt path stamps
``"schmitt_hysteresis"``; the two are directly comparable because everything below
the region-finding -- baseline, noise scale, merge rule, occupancy brake,
characterisation, flags -- is the SAME shared code, not a second copy of it.

WHERE THE ALGORITHM CAME FROM. Cell "23D. CUSUM EVENT SEGMENTER" of the owner's
``Paper1_ApoFerritin_SyntheticBenchmark_v8`` notebook (byte-identical in the v4.1,
v5 and v6 variants), whose own freeze record describes it as "one-sided sequential
CUSUM for downward blockade onset with state-dependent recovery CUSUM". The
defaults here, ``k = 0.75`` / ``h = 28``, are that notebook's exact-label calibrated
operating point on apo-ferritin at -600 mV (precision 0.854, recall 0.993, F1 0.918
at IoU >= 0.30, transferred unchanged to -500/-700 mV) -- the only point the
owner's own write-up reports as performing well, and the only one selected against
exact ground truth rather than against weak labels. See ``PARAMETER_NOTES``.

THREE THINGS THE PORT DELIBERATELY CHANGED, and why each one is not optional:

1. **Sign.** The notebook hardcodes ``z = (median(x) - x)/sigma`` -- "blockades go
   down in the SIGNED current". That is false for the owner's own recordings: at
   -600 mV the open pore sits near -1000 pA, so a blockade that reduces |I| moves
   the SIGNED current UP. The notebook detector run verbatim on his own files finds
   ZERO events. This module runs on ``z = dev / noise`` with
   ``dev = polarity * (current - baseline)``, the app's universal in-event
   direction, so it works at either bias sign and on enhancement data for free.
2. **Baseline and noise scale.** The notebook takes a scalar median and its own
   1.4826*MAD, computed inside the detector, over the whole array. This module
   takes the shared ``baseline.characterisation_baseline`` (which may be a full
   length array) and ``filters.noise_rms`` on the residual -- the same two numbers
   the Schmitt and ML paths use. Without that, the same event reports a different
   depth depending on which detector found it; that defect is documented at
   ``ml_modules.py`` and cost 67.69 pA vs 32.04 pA on one real event. It also means
   ``k = 0.75 sigma`` is the SAME sigma the table's ``snr`` column is divided by.
3. **NaN.** The notebook's guard catches a NaN only if it happens to poison the
   median; otherwise NaN propagates into the statistic, every ``>= h`` and ``<= 0``
   comparison goes False, and the state machine STALLS SILENTLY -- it stops
   detecting from that sample to the end of the trace and reports nothing wrong.
   This module refuses at the door through ``detect._check_finite``, exactly as
   ``detect_events`` does.

WHAT HAPPENS TO A TRACE THAT BEGINS MID-EVENT, because the received answer ("the
onset test sees a change, so it cannot see an event that has already started") is
only half right and the half it gets wrong matters. The statistic starts at zero and
``z`` is ALREADY positive at sample 0, so it accumulates from the first sample and
fires with Page's candidate at 0. [MEASURED] a 4001-sample leading event in a 20,000
sample trace comes back as exactly ``(0, 4000)``, flagged ``clipped_by_boundary`` and
NOT accepted -- which is the right answer, because its true onset is outside the
array and its dwell is only a lower bound. The blind spot opens ONLY when the
depressed level is a big enough fraction of the trace to move the BASELINE (in the
notebook, the whole-array median it takes internally); ``z`` is then near zero
throughout and there is genuinely nothing to see. That is a property of the baseline
estimate, not of the test -- which is one more reason this port uses the shared
characterisation baseline instead of a median computed inside the detector.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, NamedTuple, Sequence

import numpy as np
import pandas as pd

from .baseline import BaselineConfig, characterisation_baseline, require_polarity
from .cancel import CancelFlag, check_cancelled
from .characterize import characterize
from .detect import (
    MARGINAL_MERGE_OCCUPANCY,
    MIN_MERGE_OCCUPANCY,
    _as_baseline,
    _check_finite,
    _exclusion_samples,
    detect_polarity,
    gap_window_samples,
    group_starts_by_gap,
    split_low_occupancy_groups,
)
from .events import empty_df, n_clipped_boundary, span_provenance
from .filters import lowpass, measure_rise_time, noise_rms
from .measurement_quality import build_measurement_quality

try:  # pragma: no cover - packaging detail
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("nanoscope")
except Exception:  # pragma: no cover
    _VERSION = "unknown"

__all__ = [
    "detect_events_cusum",
    "cusum_onset_recovery_regions",
    "CusumRegions",
    "CUSUM_LEVEL_FLOOR_OFFSET",
    "DEFAULT_CUSUM_K",
    "DEFAULT_CUSUM_H",
    "DEFAULT_CUSUM_MERGE_GAP_S",
    "DEFAULT_CUSUM_MIN_DWELL_S",
    "PARAMETER_NOTES",
]


# ---------------------------------------------------------------------------
# the calibrated operating point
# ---------------------------------------------------------------------------
#: Drift/slack of the sequential test, in units of the MAD noise sigma. NOT pA, and
#: not a multiple of anything measured at runtime: it is a fixed standardized
#: constant chosen by grid search. An excursion shallower than ``k`` sigma can never
#: grow the statistic, so ``k`` is the depth floor of the whole detector.
DEFAULT_CUSUM_K = 0.75

#: Decision threshold of the sequential test, also in sigma-units of accumulated
#: evidence. The detection delay for an event of depth ``d`` sigma is roughly
#: ``h / (d - k)`` samples -- 6 samples for a 5-sigma event at these defaults -- and
#: the onset ESTIMATE is corrected for it, so a large ``h`` costs latency and
#: closing accuracy, not onset accuracy.
#:
#: Chosen by grid-search F1, NOT by an average-run-length design. There is no ARL
#: target anywhere in the source notebooks; do not describe it as one.
DEFAULT_CUSUM_H = 28.0

#: 0.50 ms at the owner's 50 kHz -- ``CUSUM_MERGE_GAP_MS`` in the notebook. An order
#: of magnitude wider than the 0.06 ms his Schmitt/hysteresis weak-labeller uses on
#: the host-guest and Zn/Cu series, because apo-ferritin events are 10-30 ms and
#: recross far more.
DEFAULT_CUSUM_MERGE_GAP_S = 5e-4

#: 1.00 ms -- ``CUSUM_MAIN_MIN_EVENT_MS``. The notebook also carries a 0.10 ms
#: "native" variant, but only as a sensitivity REPORT; the primary analysis is 1 ms.
DEFAULT_CUSUM_MIN_DWELL_S = 1e-3

#: The event-level floor: the frozen recovery reference is never allowed below
#: ``k + CUSUM_LEVEL_FLOOR_OFFSET``.
#:
#: LOAD-BEARING, and the one magic number in the algorithm. The recovery statistic
#: accumulates ``(event_level - z) - k`` per sample, so once the trace is back at the
#: open pore (``z ~ 0``) its drift is ``event_level - k`` and the event needs
#: ``h / (event_level - k)`` samples to close.
#:
#: WHEN THE FLOOR BINDS, exactly. The onset statistic is zero at ``start-1`` and never
#: resets before it crosses, so ``count * (event_level - k) = g_crossing >= h`` -- i.e.
#: ``event_level >= k + h/count``, and the raw level is STRICTLY above ``k`` however
#: bad the estimate is. So the floor binds precisely when the detection delay exceeds
#: ``h / 0.25 = 4h`` samples, which is the shallow-event regime (an event only ``k +
#: epsilon`` deep takes forever to trigger and is then estimated at ``k + epsilon``).
#: There the recovery drift ``h/count`` is arbitrarily small and the closing latency
#: ``h / (h/count) = count`` samples is unbounded: a 60-sample-delay event needs
#: another 60 samples to close, a 600-sample-delay event another 600. The floor caps
#: that latency at ``h / 0.25 = 4h`` samples -- 24 at ``h = 6``, 112 at ``h = 28``.
#:
#: The failure it prevents is therefore not "the drift is negative" (it cannot be) but
#: "the trace runs out before the recovery arrives": measured, a 10,000-sample /
#: 0.6-sigma event at ``k = 0.5, h = 6`` closes at its true last sample with the floor
#: and RUNS TO THE END OF THE TRACE without it, whenever fewer than 60 samples remain.
#:
#: It is a bare literal in the source notebook with no comment. What it DOES is the
#: above; WHY 0.25 rather than 0.1 or 0.5 is not recorded anywhere and is not
#: reconstructible -- see ``PARAMETER_NOTES``.
CUSUM_LEVEL_FLOOR_OFFSET = 0.25

#: Threshold handed to `detect_polarity` for the evidence record. The polarity
#: measurement is a property of the recording, not of the detector, so it uses the
#: app's usual 5-sigma coherent-excursion test rather than this detector's ``k``
#: (which is 0.75 sigma and would count noise as excursions). Same constant and same
#: reasoning as the ML path's ``_POLARITY_K``.
_POLARITY_K = 5.0

#: What is known -- and what is NOT known -- about the numbers above. Kept next to
#: them because a later re-tune must not silently inherit an operating point whose
#: provenance is this thin.
PARAMETER_NOTES: dict[str, str] = {
    "source": (
        "Exact-label calibration on apo-ferritin at -600 mV, 480 labelled events "
        "(precision 0.854, recall 0.993, F1 0.918 at IoU >= 0.30); transferred "
        "unchanged to -500/-700 mV."
    ),
    "alternatives": (
        "Two weak-label operating points also exist and are both WORSE: "
        "k=2.5/h=24 (F1 0.074, recall 0.039) and k=2.5/h=6 (F1 0.397). They were "
        "selected against Teacher weak labels, not ground truth."
    ),
    "not_recovered": (
        "The benchmark's own frozen CUSUM_FREEZE.json is on a Colab Drive path "
        "that is not on this machine, and every notebook has its CUSUM-section "
        "outputs cleared, so the printed 'Selected k / Selected h' is gone. These "
        "are the manuscript's numbers, not the freeze file's."
    ),
    "no_per_project_values": (
        "The host-guest, Zn2-insulin and Cu2-insulin projects NEVER ran this "
        "detector -- they use a Schmitt/hysteresis weak labeller. There are no "
        "per-project (k, h) values for them and none may be invented."
    ),
    "level_floor": (
        "The 0.25 in the event-level floor is uncommented in the source. Its "
        "effect is known (>= 0.25 sigma of recovery drift); the choice of "
        "magnitude is not recorded."
    ),
}


# ---------------------------------------------------------------------------
# the sequential core
# ---------------------------------------------------------------------------
#: Chunk size the vectorised scan starts at, and the ceiling it doubles up to.
#:
#: The recurrence is sequential per sample, so it is evaluated CHUNK AT A TIME with
#: the closed form ``g_i = S_i - min(S_0..S_i)`` (see `_scan_to_threshold`), which is
#: three numpy passes per chunk and no Python per-sample loop. The chunk starts small
#: and doubles because a scan usually stops early: restarting every scan at a 1 M
#: sample chunk would compute a megasample of statistic to find a crossing 40 samples
#: away, once per event. Doubling bounds the wasted work at ~2x the distance actually
#: scanned while keeping the number of Python iterations logarithmic.
#:
#: [MEASURED] on an 11.3 M-sample trace at 50 kHz (226 s, 2258 planted events, 15 pA
#: noise): the core is 0.23 s and the whole of `detect_events_cusum` is 1.09 s, the
#: rest of which is `characterize`. A per-sample Python loop over the same array is
#: three orders of magnitude slower and was never an option.
_SCAN_CHUNK_START = 1024
_SCAN_CHUNK_MAX = 1 << 20


class CusumRegions(NamedTuple):
    """What the sequential core found.

    Attributes:
        regions: ``(start, end)`` sample pairs, **INCLUSIVE at both ends** -- the
            notebook's convention, kept here so this function can be checked against
            it sample for sample. `detect_events_cusum` converts to the app's
            exclusive ends (``end + 1``) before anything else touches them; getting
            that wrong shortens every dwell by one sample.
        n_open_at_end: how many regions were closed by the END OF THE TRACE rather
            than by the recovery test. At most 1, since such a region terminates the
            scan. Reported, never dropped: `characterize` flags it
            ``clipped_by_boundary`` and de-accepts it, which is strictly better than
            the notebook (which keeps it silently, unmarked).
    """

    regions: list[tuple[int, int]]
    n_open_at_end: int


def _scan_to_threshold(
    z: np.ndarray,
    start: int,
    h: float,
    sign: float,
    offset: float,
    cancel: CancelFlag | None,
) -> tuple[int, int]:
    """First crossing of a one-sided CUSUM over ``sign*z + offset``, from ``start``.

    Runs the recurrence

        g = max(0, g + sign*z[i] + offset),   g = 0 at ``start``

    and returns ``(crossing, candidate)``: the first ``i`` with ``g >= h``, and Page's
    changepoint estimate for it -- ``1 + (last index whose statistic was 0)``, which
    is the sample the excursion is estimated to have begun at. ``(-1, -1)`` if the
    statistic never reaches ``h`` before the end of the array.

    Both the onset and the recovery test are this same function; only the increment
    differs (``+z - k`` vs ``-z + level - k``), which is why it takes ``sign`` and
    ``offset`` rather than a pre-built increment array -- building one would mean an
    extra full-length temporary per event.

    VECTORISED, not a per-sample loop. With ``S_j`` the cumulative increment and
    ``S`` extended by a leading zero, the reset recurrence has the closed form
    ``g_i = S_i - min(S_0 .. S_i)``, and the candidate is the LAST index attaining
    that running minimum: both are ``np.minimum.accumulate`` /
    ``np.maximum.accumulate``. The running minimum is re-zeroed at each chunk
    boundary (only differences matter) so ``S`` cannot drift to a magnitude where
    float64 resolution approaches ``h`` on a multi-million-sample trace.
    """
    n = int(z.size)
    run_min = 0.0            # running min of the extended cumulative sum, renormalised to 0
    run_arg = int(start)     # ABSOLUTE sample index the running min was last attained at
    s_off = 0.0              # cumulative sum carried into this chunk, same renormalisation
    pos = int(start)
    width = _SCAN_CHUNK_START

    while pos < n:
        stop = min(n, pos + width)
        chunk = z[pos:stop]
        m = stop - pos

        # S[j] is the cumulative increment through absolute sample pos+j. The constant
        # part of the increment is accumulated analytically rather than materialised.
        s = np.cumsum(chunk)
        if sign < 0:
            np.negative(s, out=s)
        s += offset * np.arange(1, m + 1, dtype=np.float64)
        s += s_off

        run_mins = np.minimum.accumulate(s)
        np.minimum(run_mins, run_min, out=run_mins)
        g = s - run_mins

        # Positions attaining the running minimum are exactly where the statistic is 0,
        # i.e. where a candidate onset is (re)armed. Equality is exact: `run_mins` is
        # built from these very floats by minimum.accumulate, so no tolerance is needed.
        # The index written is the candidate itself (one PAST the zero sample), and the
        # filler is the carried candidate, which is <= every index in this chunk's
        # accumulate -- so maximum.accumulate yields "the most recent candidate".
        armed = np.where(s == run_mins, np.arange(pos + 1, stop + 1, dtype=np.int64), run_arg)
        last_arg = np.maximum.accumulate(armed)

        hits = np.flatnonzero(g >= h)
        if hits.size:
            j = int(hits[0])
            return pos + j, int(last_arg[j])

        base = float(run_mins[-1])
        s_off = float(s[-1]) - base
        run_min = 0.0
        run_arg = int(last_arg[-1])
        pos = stop
        width = min(width * 2, _SCAN_CHUNK_MAX)
        # One poll per chunk. A chunk is a handful of vectorised numpy calls that
        # cannot be interrupted from Python, so this is the finest granularity a
        # Cancel can achieve here -- and at the 1 M ceiling it is ~10 ms of work.
        check_cancelled(cancel)

    return -1, -1


def cusum_onset_recovery_regions(
    dev_sigma: np.ndarray,
    k: float,
    h: float,
    *,
    level_floor_offset: float = CUSUM_LEVEL_FLOOR_OFFSET,
    cancel: CancelFlag | None = None,
) -> CusumRegions:
    """The two-state sequential core: standardized deviation in, INCLUSIVE spans out.

    Args:
        dev_sigma: ``dev / noise`` -- the deviation in units of the noise sigma,
            POSITIVE INSIDE AN EVENT (the app's universal convention). This is the
            only place the notebook's sign flip lives; callers hand it ``dev``, not
            ``median - current``.
        k: Drift constant, in sigma. An excursion shallower than this is invisible.
        h: Decision threshold, in accumulated sigma.
        level_floor_offset: The event level is floored at ``k + this``; see
            :data:`CUSUM_LEVEL_FLOOR_OFFSET` for why removing it is fatal. Exposed
            so a test can remove it and watch the event run to the end of the trace,
            not because it is a knob anyone should turn.
        cancel: Polled once per scan chunk.

    Returns:
        :class:`CusumRegions` -- inclusive ``(start, end)`` pairs, sorted and
        non-overlapping by construction (the machine cannot be in two events at once,
        and a fresh onset candidate is armed only after the previous event closes).

    THE STATE DEPENDENCE, precisely, because it is the part a well-meaning
    "improvement" breaks. At the instant onset fires -- and ONLY then -- the recovery
    reference is set to the MEAN of ``dev_sigma`` over ``[onset_estimate, crossing]``,
    i.e. over the detection delay, which for a deep event is a handful of samples. It
    is then FROZEN for the event's whole life; it is not a median, it is not
    re-estimated, and it does NOT track the trace.

    WHAT FREEZING BUYS, and what it costs, both measured rather than asserted --
    because the intuitive story ("a frozen deep reference keeps a shallow second level
    inside the same event") is the wrong way round and it is worth saying so here
    where someone will otherwise re-derive it.

    It buys two things. The criterion is DEPTH-RELATIVE, so one ``(k, h)`` closes a
    3-sigma and a 30-sigma event equally well where an absolute recovery threshold
    would need tuning per depth. And the whole closing rule is FIXED AT ONSET: the
    recovery latency ``h / (event_level - k)`` is known the moment the event starts,
    and the emitted boundary does not depend on what the event does afterwards.

    It costs multi-level events. The recovery statistic grows whenever the trace comes
    back to within ``k`` of the frozen level, so a step MORE than ``k`` shallower reads
    as a recovery and ends the event. [MEASURED] 50 samples at 6 sigma stepping to 5.0
    sigma for 400, at k=0.75/h=28: frozen returns ``[(500, 549), (662, 949)]`` while a
    running-mean variant returns ``[(500, 949)]`` -- the running mean decays toward the
    shallow level, its drift dies, and it keeps the event whole. Both answers are
    pinned in ``test_detect_cusum.py::test_t8_the_frozen_level_does_not_track_the_trace``.
    The frozen one is the owner's algorithm and ``merge_gap_s`` is the repair; do NOT
    "improve" it into a running mean, which is a different detector with a different
    calibration.

    THE PRICE OF FREEZING IT, measured, because it is what ``merge_gap_s`` is for. The
    level is a mean of ``count = ceil(h/(depth-k))`` samples, so its standard error is
    ``1/sqrt(count)`` sigma -- and for a DEEP event that count is tiny (2 samples at
    depth 20 sigma with the default ``h = 28``). The recovery drift inside the event is
    then ``level_error - k``, which turns positive whenever the level is overestimated
    by more than ``k``, and the event closes early and is emitted as two fragments.
    On 15 pA Gaussian noise: 4 % of 20-sigma/1.5 ms events split, 3 % of 8-sigma and
    5-sigma/10 ms events, 0 % of 3-sigma/10 ms events (whose 12-sample window estimates
    the level well). This is a property of the algorithm as the owner specified it --
    the spec is explicit that there is no minimum-sample requirement and no
    re-estimation -- and the 0.5 ms ``merge_gap_s`` default is what stitches the
    fragments back together. Do not "fix" it by widening the window here; that changes
    the detector.
    """
    z = np.asarray(dev_sigma, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError(f"dev_sigma must be 1-D, got shape {z.shape}")
    k = float(k)
    h = float(h)
    if not np.isfinite(k) or k <= 0:
        raise ValueError(
            f"cusum_k must be finite and positive, got {k!r}. It is the drift of a "
            "one-sided test; at k <= 0 the statistic grows on pure noise and the "
            "detector fires everywhere."
        )
    if not np.isfinite(h) or h <= 0:
        raise ValueError(
            f"cusum_h must be finite and positive, got {h!r}. It is the evidence "
            "required to declare an event; at h <= 0 every sample declares one."
        )

    n = int(z.size)
    regions: list[tuple[int, int]] = []
    n_open_at_end = 0
    level_floor = k + float(level_floor_offset)

    i = 0
    while i < n:
        # -- onset: CUSUM of (z - k) against the open pore ------------------
        cross, candidate = _scan_to_threshold(z, i, h, +1.0, -k, cancel)
        if cross < 0:
            break
        start = min(max(candidate, 0), cross)

        # -- freeze the recovery reference over the detection delay ---------
        level = float(z[start:cross + 1].mean())
        level = max(level, level_floor)

        # -- recovery: CUSUM of ((level - z) - k), from the sample after the
        #    crossing (the crossing sample itself is consumed by the firing) --
        cross_off, candidate_off = _scan_to_threshold(z, cross + 1, h, -1.0, level - k, cancel)
        if cross_off < 0:
            # No recovery before the array ends. The notebook emits the open event
            # ending at the last sample and so do we -- the event is real signal and
            # dropping it would hide a clog. Downstream, `characterize`'s clip_bounds
            # rule flags it clipped_by_boundary and de-accepts it.
            regions.append((start, n - 1))
            n_open_at_end += 1
            break

        end = max(start, candidate_off - 1)      # INCLUSIVE
        regions.append((start, end))
        i = cross_off + 1

    return CusumRegions(regions, n_open_at_end)


# ---------------------------------------------------------------------------
# the app-facing detector
# ---------------------------------------------------------------------------
def detect_events_cusum(
    signal: np.ndarray,
    fs: float,
    *,
    polarity: int | None = None,
    cusum_k: float = DEFAULT_CUSUM_K,
    cusum_h: float = DEFAULT_CUSUM_H,
    min_dwell_s: float = DEFAULT_CUSUM_MIN_DWELL_S,
    max_dwell_s: float | None = None,
    baseline: np.ndarray | float | None = None,
    baseline_config: BaselineConfig | None = None,
    baseline_param_source: str = "default",
    filter_cutoff_hz: float | None = None,
    filter_order: int = 8,
    filter_kind: str = "bessel",
    merge_gap_s: float | None = DEFAULT_CUSUM_MERGE_GAP_S,
    exclude_regions: Sequence[tuple[float, float]] | None = None,
    pore_id: str = "",
    recording: str = "",
    units: str = "pA",
    cancel: CancelFlag | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find translocation events with a sequential CUSUM test and characterise them.

    Same contract as `detect.detect_events` -- ``(20-column events.SCHEMA frame,
    provenance dict)`` -- and the same baseline, noise, polarity, merge, exclusion,
    characterisation and flag machinery. Only the region-finding differs. See the
    module docstring for what the algorithm is and for the three deliberate
    departures from the source notebook.

    Args:
        signal: Current trace samples (1-D); upcast to float64 internally.
        fs: Sample rate in Hz.
        polarity: +1 or -1 making ``dev = polarity*(current-baseline)`` positive in
            events. ``None`` measures it from the data. Anything else is REFUSED --
            it is a direction, not a gain (`baseline.require_polarity`). This matters
            more here than on the Schmitt path: the notebook this came from assumed a
            direction, and the assumption is wrong for the owner's own recordings.
        cusum_k: Drift, in units of the noise sigma. See :data:`DEFAULT_CUSUM_K`.
        cusum_h: Decision threshold. See :data:`DEFAULT_CUSUM_H`.
        min_dwell_s: Events shorter than this are dropped, AFTER the merge.
        max_dwell_s: Events longer than this are kept, flagged and de-accepted by
            `characterize` -- never dropped. This is the app's answer to the
            notebook's total absence of a maximum event length; a pore that clogs
            produces one enormous region there, silently.
        baseline: Precomputed baseline, or ``None`` to measure it through the shared
            `baseline.characterisation_baseline`. Passing an array bypasses the
            estimator, so no ``baseline_fingerprint`` is stamped.
        baseline_config: The shared estimator configuration -- the SAME object the
            Schmitt and ML paths take, so one recording yields one baseline whichever
            detector found the event.
        baseline_param_source: How those parameters were chosen (`baseline.PARAM_SOURCES`).
        filter_cutoff_hz: Optional pre-filter, applied before detection AND
            characterisation; events with ``dwell_s < 2*rise_s`` are marked
            ``filter_limited``.
        filter_order: Order of that low-pass (ignored by ``'gaussian'``).
        filter_kind: ``'bessel'`` | ``'butter'`` | ``'gaussian'``; all zero-phase.
        merge_gap_s: Merge events separated by less than this gap, through the SHARED
            gap rule and its occupancy brake. ``None`` disables merging.
        exclude_regions: ``[(t0,t1),...]`` seconds to ignore; the time is subtracted
            from ``analyzed_duration_s`` and overlapping events are removed.
        pore_id: Physical pore identifier stamped into every row.
        recording: Source recording name stamped into every row.
        units: Signal units label.
        cancel: Zero-arg predicate; ``None`` uses the ambient token, which is what
            makes a GUI-worker detection stoppable. Raises `cancel.OperationCancelled`.

    Returns:
        ``(df, provenance)``.

    THE AUDIT TRAIL BALANCES::

        n_cusum_firings
          - n_merged             (folded into a neighbour by merge_gap_s)
          - n_rejected_short     (shorter than min_dwell_s, applied AFTER the merge)
          - n_rejected_excluded  (overlapped an excluded region)
          == n_events

    in the order the gates are applied. Note there are only THREE rejection terms
    where the Schmitt path has four: this detector has no equivalent of
    ``n_rejected_below_enter``, because a region is only ever emitted once the
    statistic has already crossed ``h`` -- there is no such thing as a candidate that
    failed the test. ``n_clipped_boundary``, ``n_open_at_end``, ``n_merges_refused``
    and ``n_merges_marginal`` are deliberately NOT terms: none of them drops a row.
    `tests/analysis/test_cusum_detect_audit_trail.py` pins the identity.

    Gotcha: ``analyzed_duration_s`` -- the denominator of every rate -- accounts for
    ``exclude_regions``, so it is the recorded duration only when nothing was excluded.
    """
    if fs is None:
        raise ValueError("fs is required; dwell is undefined without a sample rate")
    fs = float(fs)
    if not (min_dwell_s >= 0):
        raise ValueError(f"min_dwell_s must be >= 0, got {min_dwell_s}")
    if max_dwell_s is not None and max_dwell_s <= min_dwell_s:
        raise ValueError(
            f"max_dwell_s ({max_dwell_s}) must exceed min_dwell_s ({min_dwell_s})"
        )
    if merge_gap_s is not None and merge_gap_s < 0:
        raise ValueError(f"merge_gap_s must be >= 0, got {merge_gap_s}")
    # k and h are validated inside the core, where the recurrence they break lives.
    # Same gate as the Schmitt path for polarity, and for the same reason.
    if polarity is not None:
        require_polarity(polarity)

    check_cancelled(cancel)

    # -- optional pre-filter -------------------------------------------------
    if getattr(baseline_config, "open_pore_reference", None) is not None:
        from .reference_context import validate_open_pore_context
        validate_open_pore_context({"baseline_config": baseline_config},
                                  units=units, module_id="detect.cusum")
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
        work = lowpass(signal, fs, filter_cutoff_hz, order=filter_order, kind=filter_kind)
        filter_rise_s = float(
            measure_rise_time(fs, filter_cutoff_hz, order=filter_order, kind=filter_kind)
        )

    cur = np.asarray(work, dtype=np.float64)
    if cur.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {cur.shape}")
    if cur.size == 0:
        raise ValueError("recording is empty")
    # THE DOOR. One NaN is enough to stall the sequential recurrence permanently and
    # silently (module docstring, departure 3), so it is refused here rather than
    # detected-around. Same helper as `detect_events`, not a copy of it.
    _check_finite(cur, "recording current")
    n = cur.size

    polarity_requested = polarity
    est_polarity = None if polarity is None else int(polarity)

    # -- baseline: the SHARED estimator, never a local median ----------------
    check_cancelled(cancel)
    baseline_provided = baseline is not None
    baseline_prov: dict[str, Any] = {}
    baseline_warnings: tuple[str, ...] = ()
    if baseline is None:
        # `work`, not `cur`: the door's whole-recording memo is keyed on array
        # identity and `cur` is a fresh object on every call. See the long note at
        # the same point in `detect.detect_events`.
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
        raise ValueError(
            f"noise RMS is {noise!r}; cannot set thresholds on a trace with no "
            "measurable noise."
        )

    # -- polarity ------------------------------------------------------------
    check_cancelled(cancel)
    measured_polarity, evidence = detect_polarity(cur, base, k=_POLARITY_K)
    if polarity is None:
        polarity = measured_polarity
        polarity_source = "measured"
    else:
        polarity = int(polarity)
        polarity_source = "provided"

    polarity_warning = None
    if evidence["verdict"] == "undetermined":
        polarity_warning = (
            "polarity is undetermined (no coherent excursions above threshold); "
            f"proceeding with polarity={polarity:+d}."
        )
    elif polarity_source == "measured" and evidence.get("decided_by") == "coherent_samples":
        polarity_warning = (
            f"polarity run vote was ambiguous ({evidence['n_down_runs']} down / "
            f"{evidence['n_up_runs']} up coherent runs) and the direction was taken "
            f"from the coherent signal mass instead (confidence "
            f"{evidence['confidence']:.2f}). If the direction is wrong, set the "
            "polarity override or disable the smoothing filter."
        )
    elif (
        polarity_source == "measured"
        and "decided_by" in evidence
        and not evidence.get("corroborated_by_samples", False)
    ):
        polarity_warning = (
            f"polarity was measured at low confidence ({evidence['confidence']:.2f}); "
            f"proceeding with polarity={polarity:+d}. If the event count looks wrong, "
            "set the polarity override."
        )

    # -- the standardized deviation the sequential test runs on --------------
    # THIS LINE IS THE PORT. The notebook computes (median - current)/MAD inside the
    # detector and thereby hardcodes "down is an event"; here the direction comes from
    # `polarity` and the scale is the app's own noise_rms -- so `cusum_k = 0.75` is
    # 0.75 of the very sigma the table's `snr` column is divided by, and the detector
    # works at either bias sign.
    # One temporary, not two: on an 11.3 M-sample float64 trace each full-length
    # intermediate is 90 MB, and `dev` on its own is never wanted here.
    z = resid * (polarity / noise)

    # Excluded regions are detected THROUGH, then overlapping events dropped -- the
    # Schmitt path's reasoning applies unchanged: zeroing them would truncate a
    # straddling event into a spurious short one ending exactly at the edge.
    ex = _exclusion_samples(exclude_regions, n, fs)
    excluded_samples = sum(i1 - i0 for i0, i1 in ex)
    analyzed_duration_s = (n - excluded_samples) / fs
    if analyzed_duration_s <= 0:
        raise ValueError(
            "exclude_regions removes the entire recording; nothing left to analyse."
        )

    # -- the sequential core -------------------------------------------------
    check_cancelled(cancel)
    found = cusum_onset_recovery_regions(z, cusum_k, cusum_h, cancel=cancel)
    n_cusum_firings = len(found.regions)

    if n_cusum_firings:
        starts = np.asarray([s for s, _ in found.regions], dtype=np.int64)
        # INCLUSIVE -> EXCLUSIVE, once, here, at the boundary between the notebook's
        # convention and the app's (events.SCHEMA: end_idx is "one past the last
        # sample"). Everything below this line is in app coordinates.
        ends = np.asarray([e + 1 for _, e in found.regions], dtype=np.int64)
    else:
        starts = np.zeros(0, dtype=np.int64)
        ends = np.zeros(0, dtype=np.int64)

    # -- STEP 1: MERGE (before the duration gate -- the owner's stated order) --
    # The SHARED rule, not a second copy: `group_starts_by_gap` compares against the
    # group's running maximum end, and `split_low_occupancy_groups` refuses a group
    # that is mostly open pore. The gap ARITHMETIC is identical to the notebook's
    # (`next.start - prev.end_exclusive` is the same number as its
    # `s - e_inclusive - 1`), so this is the notebook's merge plus a brake it does
    # not have. The brake is kept rather than skipped because the pathology it exists
    # for is one spinbox away here too: 40 clean events spaced 8 ms apart merged at a
    # 10 ms window became ONE row of dwell 312.40 ms and depth 0.70 pA on the Schmitt
    # path, and nothing about a CUSUM region makes it immune.
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

    # -- STEP 2: MINIMUM DURATION, in SAMPLES ---------------------------------
    # The notebook's own rule: `minimum_samples = max(1, round(min_event_ms*fs/1000))`
    # and keep iff the inclusive length is >= that. It is a rounded SAMPLE count,
    # where `detect_events` compares seconds directly (`dwell >= min_dwell_s`); the
    # two differ only when `min_dwell_s * fs` is not an integer, and this path keeps
    # the owner's arithmetic so a run here reproduces his notebook sample for sample.
    # The `max(1, ...)` means min_dwell_s = 0 still requires a non-empty event, which
    # every region already is.
    min_samples = max(1, int(round(float(min_dwell_s) * fs)))
    long_enough = (ends - starts) >= min_samples
    n_rejected_short = int((~long_enough).sum())
    starts, ends = starts[long_enough], ends[long_enough]

    # -- STEP 3: drop anything touching an excluded region --------------------
    n_rejected_excluded = 0
    if ex and starts.size:
        drop = np.zeros(starts.size, dtype=bool)
        for i0, i1 in ex:  # a handful of regions, never a loop over samples
            drop |= (starts < i1) & (ends > i0)
        n_rejected_excluded = int(drop.sum())
        starts, ends = starts[~drop], ends[~drop]

    # Overwritten below from the rows that actually carry the flag; this pre-count
    # exists only because provenance is assembled before characterisation.
    n_clipped = int(((starts <= 0) | (ends >= n)).sum()) if starts.size else 0

    provenance: dict[str, Any] = {
        "nanoscope_version": _VERSION,
        # NOT "cusum": `analysis.cusum` is the sub-state level fitter and a bare
        # "cusum" in an exported provenance record would be unreadable a year from
        # now. The name says which stage of the pipeline this was.
        "detector": "cusum_sequential",
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
        "cusum_k": float(cusum_k),
        "cusum_h": float(cusum_h),
        #: The frozen recovery reference's floor, in sigma. Stamped because it is the
        #: difference between an event that closes and an event that swallows the rest
        #: of the recording, and it is invisible in (k, h) alone.
        "cusum_level_floor": float(cusum_k) + CUSUM_LEVEL_FLOOR_OFFSET,
        #: WHICH sigma `cusum_k` and `cusum_h` are counted in. The notebook's is a
        #: MAD over the raw array computed inside the detector; this is the app's
        #: residual noise, the same number `noise_rms_pA` reports and every `snr` is
        #: divided by. A reader comparing thresholds across the two needs to know.
        "cusum_sigma_source": "MAD (filters.noise_rms) of current - baseline",
        "min_dwell_s": float(min_dwell_s),
        "min_dwell_samples": int(min_samples),
        "max_dwell_s": None if max_dwell_s is None else float(max_dwell_s),
        "merge_gap_s": None if merge_gap_s is None else float(merge_gap_s),
        "noise_rms_pA": noise,
        "noise_method": "MAD (filters.noise_rms)",
        "polarity": polarity,
        "polarity_requested": polarity_requested,
        "polarity_source": polarity_source,
        "polarity_measured": measured_polarity,
        "polarity_evidence": evidence,
        "polarity_warning": polarity_warning,
        "baseline_method": baseline_method,
        "baseline_provided": baseline_provided,
        "baseline_median_pA": float(np.median(base)),
        # The estimator's own block, written by its ONE writer, so a CUSUM row and a
        # Schmitt row on the same recording carry the same baseline_fingerprint.
        **baseline_prov,
        "baseline_warnings": list(baseline_warnings),
        "filter_cutoff_hz": None if filter_cutoff_hz is None else float(filter_cutoff_hz),
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
        #   n_cusum_firings
        #     - n_merged             (folded into a neighbour by merge_gap_s)
        #     - n_rejected_short     (shorter than min_dwell_s, AFTER the merge)
        #     - n_rejected_excluded  (overlapped an excluded region)
        #     == n_events
        #
        # in that order, because that is the order the gates are applied above. There
        # is no `n_rejected_below_enter` term and there cannot be one: the Schmitt
        # trigger emits a candidate run on the EXIT threshold and then tests its peak
        # against the ENTER threshold (98.5 % of its candidates die there), whereas a
        # CUSUM region does not exist until the test has already passed.
        "n_cusum_firings": n_cusum_firings,
        "n_merged": n_merged,
        "n_rejected_short": n_rejected_short,
        "n_rejected_excluded": n_rejected_excluded,
        #: Regions closed by the END OF THE TRACE rather than by the recovery test --
        #: the recovery statistic never reached `h` before the array ran out.
        #:
        #: NOT A TERM of the identity, for the same reason `n_clipped_boundary` is
        #: not: the row is KEPT. It is the honest name for a specific failure the
        #: notebook has no word for at all -- a pore that clogs, or an event level so
        #: badly underestimated that recovery became unreachable -- and it is counted
        #: separately from `n_clipped_boundary` because they are different claims: this
        #: one is about the DETECTOR running out of trace, that one is about the ROW
        #: touching the edge of the analysed span (a merge or a min-dwell rejection can
        #: leave one true and the other false).
        "n_open_at_end": int(found.n_open_at_end),
        #: Merges the gap rule proposed and the occupancy floor declined. NOT a term:
        #: a refused merge drops nothing, it only declines to fold rows together, so
        #: its events are already inside `n_events`. Counted because "12 proposed, 12
        #: declined" is otherwise indistinguishable from "nothing was close enough".
        "n_merges_refused": n_merges_refused,
        "refused_merge_occupancy": refused_merge_occupancy,
        #: Merges KEPT between MIN_MERGE_OCCUPANCY and MARGINAL_MERGE_OCCUPANCY.
        "n_merges_marginal": n_merges_marginal,
        "min_merge_occupancy": MIN_MERGE_OCCUPANCY,
        "marginal_merge_occupancy": MARGINAL_MERGE_OCCUPANCY,
        #: Rows cut off by the edge of the analysed span; kept, flagged and NOT
        #: accepted. Re-derived from the rows below.
        "n_clipped_boundary": n_clipped,
        **span_provenance(0, n),
    }

    if starts.size == 0:
        provenance["n_events"] = 0
        provenance["n_filter_limited"] = 0
        df = empty_df()
        provenance["measurement_quality"] = build_measurement_quality(cur, base, df, fs, provenance)
        return df, provenance

    # `characterize` is the app's SINGLE writer of every event row and the sole owner
    # of `flag` and `accepted`. This detector computes no depth, no dwell, no ECD --
    # exactly like the notebook, which also returns only regions, and exactly like the
    # ML path, which hands its regions here the same way.
    df = characterize(
        cur,
        base,
        starts,
        ends,
        fs,
        polarity,
        noise_rms=noise,
        filter_rise_s=filter_rise_s,
        max_dwell_s=max_dwell_s,
        clip_bounds=(0, n),
        pore_id=pore_id,
        recording=recording,
        units=units,
        cancel=cancel,
    )

    if filter_rise_s is not None:
        limited = df["dwell_s"].to_numpy() < 2.0 * filter_rise_s
        df["filter_limited"] = df["filter_limited"].to_numpy(dtype=bool) | limited

    # Report what the ROWS say, not what a parallel computation predicted.
    provenance["n_clipped_boundary"] = n_clipped_boundary(df)
    provenance["n_events"] = int(len(df))
    provenance["n_filter_limited"] = int(df["filter_limited"].sum())
    provenance["measurement_quality"] = build_measurement_quality(cur, base, df, fs, provenance)
    return df, provenance
