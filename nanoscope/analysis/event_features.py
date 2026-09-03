"""The Zn2-insulin per-event waveform features and the >=4-sigma SUSTAINED gate.

This is a **port**, not a design. Every number here is fixed by
``Zn2Insulin_StudentInference.ipynb`` (cells 15 and 51) and checked against the
owner's own output, ``CSVData/event_clustering_features.csv`` -- 1488 rows x 29
columns, produced by his run. Where this module and a prose description of the
pipeline disagree, the notebook wins; where the notebook and its own comments
disagree (cell 15's comment says 5 sigma, its constant says 4.0), the code wins.

NOTHING HERE IS STORED. The thirteen features are DERIVED on demand from
``(signal, fs, EventSet)`` exactly the way :mod:`nanoscope.analysis.event_shape`
derives the charge deficit and the ensemble: the frozen 20-column
``events.SCHEMA`` is a cross-module contract with a parquet round-trip riding on
it and it gains no columns for this. An event's ``[start_idx, end_idx)`` window
plus three per-recording scalars is all the state a feature needs.

THE TWO-BASELINE TRAP -- READ THIS BEFORE CHANGING ANYTHING
-----------------------------------------------------------
The gate and the features run on the same detrended trace and **do not share a
baseline or a noise estimate**. This looks like a bug. It is the oracle, and
"fixing" it silently moves every downstream number:

===============  =========================================  ==========================================
                 gate (cell 15, :func:`gate_scalars`)       features (cell 51, :func:`feature_scalars`)
===============  =========================================  ==========================================
baseline         median of the EVENT-EXCLUDED samples       median of the WHOLE trace
noise            MAD of the event-excluded samples          MAD over the whole trace
I0               a 5 s rolling per-event local median       ``abs(median(raw))``, ONE global scalar
===============  =========================================  ==========================================

Measured on zn_30h: excluded MAD 2.795 pA vs global MAD 2.907 pA, and the
event-excluded baseline sits 0.0764 pA away from the whole-trace one, so the
catalog's ``depth_sust_pA`` is offset by exactly that constant from the feature
table's. Two consequences worth naming:

* ``snr`` here is ``depth_sust_pA / global MAD``. The catalog's ``sust_sigma`` is
  ``depth_sust_pA / excluded MAD``. They are different numbers; do not compute
  one and store it as the other.
* The notebook's per-event ``local_I0`` machinery (the rolling median, the drift
  report, the catalog's ``I0_local_pA``/``f_peak``/``fracblock`` columns) has **no
  effect on the thirteen features**. Cell 51 never calls it. ``f_sust`` here
  divides by the single global I0. It IS ported -- :func:`local_open_pore_pA` --
  because the GATE half of this module needs it: the Cu project's
  ``min_fracblock`` clause is a dI/I0 floor against the time-local I0, and that is
  the row above. Do not reach for it from the feature half; ``f_sust`` divides by
  ``FeatureScalars.i0_pA`` and changing that breaks the oracle.

WHAT "SUSTAINED" MEANS
----------------------
The median of the deviation over the entire event window, inclusive. No
sub-window, no plateau search, no trimming of onset/offset samples, and it is
read on the DETRENDED trace, never the raw. Cell 15 explains the choice: ``min()``
is the most spike-sensitive statistic available -- a one-sample noise excursion
clears a min-based gate where a real event HOLDS its level -- and the measured
median/min ratio was 0.74-0.82 and flat from 3 samples out past 1 ms, so the
median discriminates event-ness rather than duration. The peak-depth alternative
is computed by :func:`sustained_gate` for reporting but **does not gate**.

The gate does nearly all the selection work: 11004 fired regions on zn_30h become
1076 gated events (9.8 %), and 1488 of the four recordings' fired regions survive
in total. A small error in the gate therefore changes every downstream cluster,
fraction and confidence interval while still looking entirely plausible. That is
why it lives in one function with one constant.

INDEX ALIGNMENT
---------------
:func:`event_features` returns a frame **indexed by the caller's own row labels**,
carrying only the rows it could measure. The notebook does
``CLUST.loc[_sub.index, _col] = _hl`` -- a positional array assigned into a
label-indexed slice, correct only because the original ``RangeIndex`` happened to
survive; any filter, sort or ``reset_index`` in between misaligns every label
silently. Returning a label-aligned frame is how a port avoids inheriting that.

FLOAT FIDELITY
--------------
The reference trace is ``float32`` on disk and the notebook leaves it that way:
``median`` and ``MAD`` of a ``float32`` array are computed in ``float32``, while a
segment's deviations are explicitly promoted (``med - seg.astype(float)``) and
every per-event statistic is ``float64``. This module reproduces that split
exactly -- it does **not** widen the input trace -- because widening moves the
per-recording scalars in their last bits and that alone costs bit-exactness
against the oracle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _sps
from scipy.ndimage import median_filter as _median_filter

from .cancel import CancelFlag, CancelPoller
from .events import EventSet

__all__ = [
    "FEATURE_COLUMNS",
    "CLUST_FEATS",
    "MIN_DEPTH_SIGMA",
    "MIN_DEPTH_PA",
    "MIN_FRACBLOCK",
    "I0_WINDOW_S",
    "ENTROPY_BINS",
    "MIN_EVENT_SAMPLES",
    "SKEW_KURT_MIN_SAMPLES",
    "MAD_SCALE",
    "FeatureScalars",
    "GateScalars",
    "GateResult",
    "AmplitudeGateResult",
    "shannon_entropy",
    "feature_scalars",
    "gate_scalars",
    "sustained_gate",
    "local_open_pore_pA",
    "amplitude_gate",
    "event_features",
    "time_half",
]

#: The feature table's column order, as ``_event_features`` emits it (dict order,
#: matching ``event_clustering_features.csv``). ``dwell_ms`` rides along un-logged;
#: the thirteen clustering features are the rest.
FEATURE_COLUMNS: tuple[str, ...] = (
    "dwell_ms",
    "log_dwell",
    "depth_max_pA",
    "depth_sust_pA",
    "mean_pA",
    "std_pA",
    "min_dev_pA",
    "skewness",
    "kurtosis",
    "area_pA_ms",
    "t_peak_frac",
    "entropy",
    "f_sust",
    "snr",
)

#: The thirteen features **in the order the scaling and PCA consume them**. This is
#: NOT ``FEATURE_COLUMNS`` minus ``dwell_ms``: ``CLUST_FEATS`` puts ``entropy``
#: LAST where the CSV puts it eleventh. The loading tables and the per-scope PCA
#: bases are stated in this order, so a downstream module that uses the CSV order
#: instead produces a differently-ordered loading table for the same fit.
CLUST_FEATS: tuple[str, ...] = (
    "log_dwell",
    "depth_max_pA",
    "depth_sust_pA",
    "mean_pA",
    "std_pA",
    "min_dev_pA",
    "skewness",
    "kurtosis",
    "area_pA_ms",
    "t_peak_frac",
    "f_sust",
    "snr",
    "entropy",
)

#: The sustained-depth gate, in units of the event-excluded MAD. Cell 1's
#: ``MIN_DEPTH_SIGMA``. A stale comment in cell 15 says 5; the notebook flags its
#: own comment as stale, and the measurement agrees -- the smallest ``sust_sigma``
#: among the owner's gated rows is 4.0011 (zn_30h) and 4.0109 (zn_2h).
MIN_DEPTH_SIGMA: float = 4.0

#: The ABSOLUTE sustained-depth floor in pA, from ``Cu2Insulin_StudentInference``
#: cell 1 (added 2026-08-26). Calibrated to sit just above the typical 4-sigma gate
#: (7-10 pA on the Cu pores) and far below the real events (20-140 pA). Only the Cu
#: project ships it non-zero; ``0.0`` is off.
MIN_DEPTH_PA: float = 10.0

#: The ABSOLUTE sustained dI/I0 floor, same source and date. 0.05 sits below every
#: Cu recording's event core (per-tag median dI/I0 0.24 / 0.11 / 0.28 / 0.62 for
#: cu_2h / cu_10h / cu_18h / cu_30h) and matches ``F_MIN`` in
#: ``Cu2Insulin_IntraEventAnalysis.ipynb``, so the two notebooks agree. ``0.0`` is off.
MIN_FRACBLOCK: float = 0.05

#: The rolling window for the time-local open-pore current, in seconds. Cell 1's
#: ``I0_WINDOW_S``: long against events (< 10 ms), short against drift (tens of s).
I0_WINDOW_S: float = 5.0

#: Fixed bin count for the within-event amplitude entropy. **Fixed, not
#: Freedman-Diaconis**: the bin *edges* adapt to each event's own min/max, the bin
#: *count* never does, and the normaliser is ``log(8)``.
ENTROPY_BINS: int = 8

#: ``if _e - _s < 3: continue`` -- a window shorter than this is not measured at
#: all. Never fires on the reference data (``min_event_ms=0.1`` at 50 kHz is
#: exactly 5 samples), but it is contract.
MIN_EVENT_SAMPLES: int = 3

#: Below this, ``skewness`` and ``kurtosis`` fall back to ``0.0`` -- a
#: near-symmetric prior, since a 4-sample histogram cannot support a shape
#: statistic. Also the floor for :func:`shannon_entropy`.
SKEW_KURT_MIN_SAMPLES: int = 5

#: MAD -> sigma for a Gaussian.
MAD_SCALE: float = 1.4826


# ===========================================================================
# per-recording scalars
# ===========================================================================


@dataclass(frozen=True)
class FeatureScalars:
    """The three per-recording numbers :func:`event_features` needs.

    Attributes:
        baseline_pA: ``median`` of the WHOLE detrended trace (the notebook's
            ``_medS``). Deviations are ``baseline_pA - sample``, so a downward
            blockade reads positive.
        noise_pA: ``1.4826 * median(|trace - baseline_pA|)`` over the whole trace.
            The denominator of ``snr``. NOT the gate's event-excluded MAD.
        i0_pA: ``abs(median(raw current))`` over the whole recording -- one
            positive scalar. The denominator of ``f_sust``.
    """

    baseline_pA: float
    noise_pA: float
    i0_pA: float


@dataclass(frozen=True)
class GateScalars:
    """The event-excluded baseline and noise the sustained gate measures against.

    Attributes:
        baseline_pA: median of the quiet (event-excluded) samples.
        noise_pA: ``1.4826 * MAD`` of the quiet samples. The gate threshold is
            ``MIN_DEPTH_SIGMA * noise_pA``.
        quiet: boolean mask, True where the sample survived the two-pass
            sigma-clip. Kept because the galleries want the same local baseline.
        duty: ``1 - quiet.mean()``, the fraction of samples the clip called
            event-like. Reported by cell 15 and worth showing: it is *why* the
            baseline must exclude events at all, since it differs per recording
            (zn_30h fires ~36 events/s where zn_2h fires ~3/s, and an uncorrected
            median is therefore dragged toward the event level by a DIFFERENT
            amount in each recording being compared).
    """

    baseline_pA: float
    noise_pA: float
    quiet: np.ndarray
    duty: float


@dataclass(frozen=True)
class GateResult:
    """Per-region gate decision plus the depths it decided on.

    All arrays are length ``n_regions`` and positionally aligned with the windows
    that were passed in.

    Attributes:
        keep: ``depth_sust_pA >= min_depth_sigma * scalars.noise_pA``. THE gate.
        depth_sust_pA: ``baseline - median(segment)``, the sustained depth.
        depth_peak_pA: ``baseline - min(segment)``, the deepest single sample.
            Computed for reporting only -- **it does not gate**.
        sust_sigma: ``depth_sust_pA / scalars.noise_pA``.
        peak_sigma: ``depth_peak_pA / scalars.noise_pA``.
        scalars: the :class:`GateScalars` used.
        min_depth_sigma: the threshold actually applied.
        index: the caller's own row labels, so a decision can be carried back by
            LABEL (``events.df.loc[result.kept_index]``) rather than by position.
    """

    keep: np.ndarray
    depth_sust_pA: np.ndarray
    depth_peak_pA: np.ndarray
    sust_sigma: np.ndarray
    peak_sigma: np.ndarray
    scalars: GateScalars
    min_depth_sigma: float
    index: pd.Index

    @property
    def n_kept(self) -> int:
        """How many regions passed."""
        return int(np.count_nonzero(self.keep))

    @property
    def kept_index(self) -> pd.Index:
        """The labels of the regions that passed, in their original order."""
        return self.index[self.keep]


@dataclass(frozen=True)
class AmplitudeGateResult:
    """The three sustained-depth gates' verdicts, and each one's separately.

    The per-clause masks are not decoration: cell 15 PRINTS the attrition of each
    gate on its own ("sigma>=4 keeps N | pA>=10 keeps N | dI/I0>=0.05 keeps N |
    ALL THREE keep N") precisely so it is visible WHICH gate is doing the work and
    the thresholds can be tuned against a real recording instead of taken on faith.
    Collapsing them to one boolean throws that away.

    Attributes:
        keep: ``keep_sigma & keep_pa & keep_fracblock``. THE decision.
        keep_sigma: the relative, per-recording clause. All-True when off.
        keep_pa: the absolute picoamp clause. All-True when off.
        keep_fracblock: the absolute dI/I0 clause. All-True when off.
        sustained: the :class:`GateResult` every clause was measured from -- one
            measurement of ``depth_sust_pA``, shared. **Read ``keep`` from HERE, not
            from ``sustained.keep``**: when the sigma clause is off it was called
            with a threshold of 0, so ``sustained.keep`` then means "depth >= 0",
            which is a real test and not the same as "no test".
        i0_local_pA: ``|I0|`` at each event's ONSET, from the 5 s rolling
            event-excluded median of the RAW trace. NaN when the fracblock gate is
            off (it is the only consumer).
        fracblock: ``depth_sust_pA / i0_local_pA``. NaN when the gate is off.
        i0_drift: peak-to-peak swing of the local-I0 curve as a fraction of its own
            median. Above ~0.15 the notebook prints a warning: a global I0 would be
            materially wrong for that recording.
        min_depth_sigma / min_depth_pa / min_fracblock: the thresholds applied,
            ``0.0`` meaning the clause did not run.
    """

    keep: np.ndarray
    keep_sigma: np.ndarray
    keep_pa: np.ndarray
    keep_fracblock: np.ndarray
    sustained: GateResult
    i0_local_pA: np.ndarray
    fracblock: np.ndarray
    i0_drift: float
    min_depth_sigma: float
    min_depth_pa: float
    min_fracblock: float

    @property
    def n_kept(self) -> int:
        """How many regions passed all three."""
        return int(np.count_nonzero(self.keep))

    @property
    def depth_sust_pA(self) -> np.ndarray:
        """The one sustained depth every clause decided on."""
        return self.sustained.depth_sust_pA

    def attrition(self) -> dict[str, int]:
        """Per-gate survivor counts, the way cell 15 prints them."""
        return {
            "fired": int(self.keep.size),
            "sigma": int(np.count_nonzero(self.keep_sigma)),
            "pA": int(np.count_nonzero(self.keep_pa)),
            "fracblock": int(np.count_nonzero(self.keep_fracblock)),
            "all": self.n_kept,
        }


# ===========================================================================
# input normalisation
# ===========================================================================


def _as_trace(signal, name: str = "signal") -> np.ndarray:
    """A 1-D view of ``signal`` **without widening its dtype** (see module docs)."""
    arr = np.asarray(signal)
    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty; per-recording scalars need samples")
    if not np.issubdtype(arr.dtype, np.floating):
        # An integer or object trace has no float32/float64 distinction to
        # preserve, so promoting is free of the fidelity concern above.
        arr = arr.astype(np.float64)
    return arr


def _require_finite(arr: np.ndarray, name: str) -> None:
    """Refuse a trace with NaN/Inf rather than return a NaN scalar for it.

    A single non-finite sample makes ``median`` NaN, which makes the baseline NaN,
    which makes every depth NaN and every gate decision False -- a silent
    "detected nothing" that looks exactly like a quiet recording.
    """
    if not np.all(np.isfinite(arr)):
        bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(
            f"{name} holds {bad} non-finite sample(s) of {arr.size}; the median and MAD "
            "would both be NaN and every event would silently fail the gate. Clean or "
            "trim the trace first."
        )


def _oriented(sig: np.ndarray, polarity: int) -> np.ndarray:
    """``polarity * sig``, a bit-exact no-op at ``+1``.

    Polarity is handled by MIRRORING the trace, which is exact (detecting an
    upward excursion in ``x`` is detecting a downward one in ``-x``) and leaves the
    ``+1`` path -- every Zn2 recording -- untouched, so the oracle holds by
    construction rather than by assertion.
    """
    pol = int(polarity)
    if pol not in (1, -1):
        raise ValueError(f"polarity must be +1 or -1, got {polarity!r}")
    return sig if pol == 1 else (pol * sig)


def _windows(events, n: int) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """``(starts, ends, index)`` from an EventSet, a frame, or an ``(m, 2)`` array.

    Half-open ``[start, end)``, clipped into ``[0, n]`` so an event past the end of
    a re-loaded trace yields an empty window rather than wrapping a negative index.
    The returned ``index`` is the caller's own row labels, which is what makes the
    result joinable without the notebook's positional-into-labelled hazard.
    """
    if isinstance(events, EventSet):
        df = events.df
    elif isinstance(events, pd.DataFrame):
        df = events
    else:
        arr = np.asarray(events)
        if arr.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                pd.RangeIndex(0),
            )
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"windows must be an (m, 2) array of [start, end) samples, got shape {arr.shape}"
            )
        starts = np.clip(arr[:, 0].astype(np.int64), 0, n)
        ends = np.clip(arr[:, 1].astype(np.int64), 0, n)
        return starts, ends, pd.RangeIndex(len(starts))

    missing = [c for c in ("start_idx", "end_idx") if c not in df.columns]
    if missing:
        raise ValueError(
            f"event table is missing {missing}; event boundaries are the only input "
            "these features take from the EventSet"
        )
    starts = np.clip(df["start_idx"].to_numpy(dtype=np.int64), 0, n)
    ends = np.clip(df["end_idx"].to_numpy(dtype=np.int64), 0, n)
    return starts, ends, df.index


# ===========================================================================
# the scalars
# ===========================================================================


def feature_scalars(
    detrended,
    raw=None,
    *,
    i0_pA: float | None = None,
    polarity: int = 1,
) -> FeatureScalars:
    """The whole-trace baseline, whole-trace MAD and global I0 (cell 51).

    Args:
        detrended: the recording's detrended current in pA (``current_detr_pA``),
            the trace the events were detected on. Its dtype is preserved.
        raw: the recording's raw current in pA (``current_raw_pA``), used only for
            ``i0_pA = abs(median(raw))``. Pass ``None`` and give ``i0_pA``
            directly when the raw trace is not to hand.
        i0_pA: an explicit open-pore magnitude, overriding ``raw``.
        polarity: ``+1`` (default) or ``-1``; see :func:`_oriented`.

    Returns:
        :class:`FeatureScalars`.

    Raises:
        ValueError: on an empty or non-finite trace, if neither ``raw`` nor
            ``i0_pA`` is given, or if ``i0_pA`` is not finite and positive.
    """
    sig = _oriented(_as_trace(detrended, "detrended"), polarity)
    _require_finite(sig, "detrended")

    # float32 in, float32 median and MAD -- see FLOAT FIDELITY in the module docs.
    med = float(np.median(sig))
    mad = MAD_SCALE * float(np.median(np.abs(sig - med)))

    if i0_pA is None:
        if raw is None:
            raise ValueError(
                "f_sust is depth/I0, so feature_scalars needs either the raw trace "
                "or an explicit i0_pA; got neither"
            )
        rawv = _as_trace(raw, "raw")
        _require_finite(rawv, "raw")
        i0 = abs(float(np.median(rawv)))
    else:
        i0 = abs(float(i0_pA))
    if not np.isfinite(i0) or i0 <= 0.0:
        raise ValueError(f"i0_pA must be finite and non-zero, got {i0!r}")

    return FeatureScalars(baseline_pA=med, noise_pA=mad, i0_pA=i0)


def gate_scalars(detrended, *, polarity: int = 1) -> GateScalars:
    """The EVENT-EXCLUDED baseline and noise the gate measures against (cell 15).

    A two-pass sigma-clip on the detrended trace: clip at 5 sigma of the raw
    median/MAD, re-measure on the survivors, then keep everything within 3 sigma of
    *that* as "quiet". The baseline and MAD are the median and MAD of the quiet
    samples.

    Args:
        detrended: the detrended current in pA. Dtype preserved.
        polarity: ``+1`` (default) or ``-1``.

    Returns:
        :class:`GateScalars`.

    Raises:
        ValueError: on an empty or non-finite trace, or if the clip leaves no quiet
            samples at all (a trace with no measurable open-pore level -- refused
            rather than answered with a NaN baseline).
    """
    sig = _oriented(_as_trace(detrended, "detrended"), polarity)
    _require_finite(sig, "detrended")

    m0 = float(np.median(sig))
    s0 = MAD_SCALE * float(np.median(np.abs(sig - m0)))
    k0 = np.abs(sig - m0) < 5 * s0
    if not k0.any():
        raise ValueError(
            "the 5-sigma pass excluded every sample (MAD is 0 -- a constant trace?); "
            "there is no open-pore level to measure a depth against"
        )
    m1 = float(np.median(sig[k0]))
    s1 = MAD_SCALE * float(np.median(np.abs(sig[k0] - m1)))
    quiet = np.abs(sig - m1) < 3 * s1
    if not quiet.any():
        raise ValueError(
            "the event-exclusion mask kept no samples; the trace has no quiet "
            "stretch to define a baseline from"
        )
    med = float(np.median(sig[quiet]))
    mad = MAD_SCALE * float(np.median(np.abs(sig[quiet] - med)))
    return GateScalars(
        baseline_pA=med,
        noise_pA=mad,
        quiet=quiet,
        duty=float(1.0 - quiet.mean()),
    )


# ===========================================================================
# the gate
# ===========================================================================


def sustained_gate(
    detrended,
    events,
    *,
    min_depth_sigma: float = MIN_DEPTH_SIGMA,
    scalars: GateScalars | None = None,
    polarity: int = 1,
    cancel: CancelFlag | None = None,
) -> GateResult:
    """The >=4-sigma SUSTAINED-depth gate: which detected regions are events.

    For each region, ``depth_sust = baseline - median(segment)`` on the DETRENDED
    trace, kept when ``depth_sust >= min_depth_sigma * noise``. The peak depth
    ``baseline - min(segment)`` is measured alongside for reporting and plays no
    part in the decision -- see WHAT "SUSTAINED" MEANS in the module docstring.

    Args:
        detrended: the detrended current in pA.
        events: an :class:`~nanoscope.analysis.events.EventSet`, a frame with
            ``start_idx``/``end_idx``, or an ``(m, 2)`` array of half-open windows.
        min_depth_sigma: threshold in event-excluded MADs. Defaults to
            :data:`MIN_DEPTH_SIGMA` (4.0). Changing it changes the population.
        scalars: precomputed :class:`GateScalars`; recomputed from ``detrended``
            when omitted. Pass them when gating the same trace more than once.
        polarity: ``+1`` (default) or ``-1``. Applied to the trace before
            measuring, and to ``scalars`` computation when they are derived here.
        cancel: cancellation token; the ambient one is used when ``None``.

    Returns:
        :class:`GateResult`, positionally aligned with ``events``.

    Raises:
        ValueError: on a bad trace (see :func:`gate_scalars`) or a non-finite
            ``min_depth_sigma``.
    """
    thr_sigma = float(min_depth_sigma)
    if not np.isfinite(thr_sigma):
        raise ValueError(f"min_depth_sigma must be finite, got {min_depth_sigma!r}")

    sig = _oriented(_as_trace(detrended, "detrended"), polarity)
    if scalars is None:
        scalars = gate_scalars(sig, polarity=1)  # already oriented
    med = scalars.baseline_pA
    mad = scalars.noise_pA

    starts, ends, index = _windows(events, sig.size)
    m = len(starts)
    depth_sust = np.full(m, np.nan, dtype=np.float64)
    depth_peak = np.full(m, np.nan, dtype=np.float64)

    poll = CancelPoller(cancel)
    for i in range(m):
        poll()
        seg = sig[starts[i] : ends[i]]
        if seg.size == 0:
            continue
        depth_peak[i] = med - float(seg.min())
        depth_sust[i] = med - float(np.median(seg))

    with np.errstate(invalid="ignore", divide="ignore"):
        sust_sigma = depth_sust / mad if mad else np.full(m, np.nan)
        peak_sigma = depth_peak / mad if mad else np.full(m, np.nan)
        keep = depth_sust >= thr_sigma * mad
    keep &= np.isfinite(depth_sust)

    return GateResult(
        keep=keep,
        depth_sust_pA=depth_sust,
        depth_peak_pA=depth_peak,
        sust_sigma=np.asarray(sust_sigma, dtype=np.float64),
        peak_sigma=np.asarray(peak_sigma, dtype=np.float64),
        scalars=scalars,
        min_depth_sigma=thr_sigma,
        index=index,
    )


# ===========================================================================
# the two ABSOLUTE gates (Cu2Insulin_StudentInference.ipynb cell 1, 2026-08-26)
# ===========================================================================


def local_open_pore_pA(
    raw,
    quiet,
    fs: float,
    *,
    at=None,
    win_s: float = I0_WINDOW_S,
) -> tuple[np.ndarray, float, float]:
    """The notebook's ``local_I0`` -- event-excluded rolling-median open-pore current.

    Ported from cell 1 of all three StudentInference notebooks, where it is the
    denominator of ``fracblock``. ``I0`` is NOT constant within a recording: the
    open-pore level drifts by tens of percent over a few minutes (measured spans:
    -162 to -255 pA across the four Zn pores, -125 to -316 pA across the four Cu
    ones). Detrending flattens the *trace*, so the noise is stable, but a GLOBAL
    ``I0`` still makes every dI/I0 depend on WHEN in the recording its event
    happened. So each event is normalised by the open-pore current local to it.

    The implementation follows the notebook's own second version, and for the
    reason its docstring gives: the first materialised a full-length array plus a
    float64 ``arange`` -- ~230 MB for a 390 s recording -- and killed the kernel.
    Block medians are a few thousand points, so the rolling median is computed on
    those and interpolated at the requested indices instead.

    Args:
        raw: the RAW (un-detrended) current in pA. Sign is preserved here; callers
            that want a magnitude take ``abs`` of the result, as the notebook does.
        quiet: boolean mask, True on event-excluded samples -- normally
            :attr:`GateScalars.quiet`. Blocks with no quiet sample fall back to the
            whole block.
        fs: sample rate in Hz.
        at: sample indices to evaluate at (the event onsets). ``None`` returns the
            whole block-median curve instead.
        win_s: rolling-median window. Default 5 s -- long against events (< 10 ms),
            short against drift (tens of seconds).

    Returns:
        ``(values, global_median_pA, drift)``. ``values`` is the local ``I0`` at
        each index in ``at`` (or the block curve when ``at`` is ``None``);
        ``drift`` is the peak-to-peak swing of the smoothed curve as a fraction of
        its own median, which is what the notebook prints when it exceeds 0.15.

    Raises:
        ValueError: on an empty trace or a non-positive ``fs``.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive, finite sample rate, got {fs!r}")
    arr = _as_trace(np.asarray(raw, dtype=np.float32), "raw")
    q = np.asarray(quiet, dtype=bool).ravel()
    if q.size != arr.size:
        raise ValueError(
            f"quiet mask has {q.size} samples but raw has {arr.size}; the mask must "
            "describe the same trace the I0 curve is measured on"
        )

    blk = max(int(0.1 * fs), 1)
    n = arr.size
    nb = max(n // blk, 1)
    bm = np.empty(nb, dtype=np.float64)
    for i in range(nb):
        sl = slice(i * blk, (i + 1) * blk)
        qs = q[sl]
        seg = arr[sl][qs] if qs.any() else arr[sl]
        bm[i] = np.median(seg) if seg.size else np.nan

    good = np.isfinite(bm)
    if not good.any():
        g = float(np.median(arr))
        n_at = len(np.asarray(at)) if at is not None else 1
        return np.full(n_at, g, dtype=np.float32), g, 0.0
    bm = np.interp(np.arange(nb), np.flatnonzero(good), bm[good])

    w = max(int(round(win_s * fs / blk)), 1)
    if w % 2 == 0:
        w += 1
    # THE (1, w) FORM IS NOT A STYLE CHOICE. scipy.ndimage.median_filter 1.15 is
    # NON-DETERMINISTIC on a 1-D input once the footprint passes ~261 taps -- this repo
    # measured six different answers in forty runs on a real recording, spanning 6.4 pA,
    # with a fixed 5-sigma decode counting anywhere from 337 to 469 events. The 2-D form
    # takes a different path inside ndimage and is stable (0 wrong in 600 calls). A 5 s
    # I0 window over block medians can exceed that footprint, and this baseline feeds
    # `min_fracblock`, so the 1-D form would have made Cu's dI/I0 gate irreproducible.
    # Pinned structurally by tests/analysis/test_masked_baseline.py.
    sm = _median_filter(bm[None, :], size=(1, w), mode="nearest")[0]
    centers = (np.arange(nb) + 0.5) * blk
    drift = float(
        (np.abs(sm).max() - np.abs(sm).min()) / max(abs(float(np.median(sm))), 1e-9)
    )
    if at is None:
        return sm, float(np.median(sm)), drift
    vals = np.interp(np.asarray(at, dtype=float), centers, sm).astype(np.float32)
    return vals, float(np.median(sm)), drift


def amplitude_gate(
    detrended,
    events,
    *,
    min_depth_sigma: float = MIN_DEPTH_SIGMA,
    min_depth_pa: float = 0.0,
    min_fracblock: float = 0.0,
    raw=None,
    fs: float | None = None,
    scalars: GateScalars | None = None,
    polarity: int = 1,
    cancel: CancelFlag | None = None,
) -> AmplitudeGateResult:
    """All three sustained-depth gates, AND-ed, on ONE measurement of the depth.

    This is the gate cell 15 of ``Cu2Insulin_StudentInference.ipynb`` applies:
    every clause reads the SAME ``depth_sust_pA`` that :func:`sustained_gate`
    already computes, so there is exactly one place that decides what "how deep is
    this event" means::

        keep = (ds >= min_depth_sigma * noise)      # relative, per-recording
               and (ds >= min_depth_pa)             # absolute pA
               and (ds / |I0_local| >= min_fracblock)   # absolute dI/I0

    WHY THE TWO ABSOLUTE GATES EXIST (the notebook's own rationale, 2026-08-26).
    ``min_depth_sigma`` scales with each recording's OWN noise, so wherever the
    baseline wanders or the noise band widens, the gate widens with it and a
    baseline fluctuation still clears it. The other two do not move when the
    baseline misbehaves:

    * ``min_depth_pa`` is blunt and pore-independent -- a floor in picoamps.
    * ``min_fracblock`` is the principled one: a sustained dI/I0 floor against the
      TIME-LOCAL ``I0``, which is what partially normalises pore size, and it bites
      hardest exactly where the sigma gate is weakest. Measured: cu_10h has
      ``I0 = 316 pA``, so its 4-sigma gate is only dI/I0 ~ 0.023 -- low enough to
      admit baseline wander wholesale.

    ``0.0`` disables a clause entirely; it is NOT the same as a threshold of zero,
    which would still reject an upward (negative-depth) excursion.

    Args:
        detrended: the detrended current in pA -- what the depths are measured on.
        events: an EventSet, a frame with ``start_idx``/``end_idx``, or an
            ``(m, 2)`` array of HALF-OPEN windows (see :func:`sustained_gate`).
        min_depth_sigma: threshold in event-excluded MADs. ``0.0`` = off.
        min_depth_pa: absolute sustained-depth floor in pA. ``0.0`` = off.
        min_fracblock: absolute sustained dI/I0 floor. ``0.0`` = off. Needs
            ``raw`` and ``fs``.
        raw: the RAW current in pA, for the time-local ``I0``. Required only when
            ``min_fracblock`` is on.
        fs: sample rate, required only when ``min_fracblock`` is on.
        scalars: precomputed :class:`GateScalars`; recomputed when omitted.
        polarity: ``+1`` (default) or ``-1``, applied to both traces.
        cancel: cancellation token.

    Returns:
        :class:`AmplitudeGateResult`.

    Raises:
        ValueError: if ``min_fracblock`` is on but ``raw``/``fs`` were not given --
            LOUDLY, rather than silently skipping a gate the caller asked for.
    """
    thr_sigma = float(min_depth_sigma)
    thr_pa = float(min_depth_pa)
    thr_frac = float(min_fracblock)
    for _name, _v in (("min_depth_pa", thr_pa), ("min_fracblock", thr_frac)):
        if not np.isfinite(_v) or _v < 0:
            raise ValueError(f"{_name} must be finite and >= 0, got {_v!r}")

    # The sigma clause is measured by the EXISTING gate, always -- even when it is
    # switched off -- because that is where ``depth_sust_pA`` is defined, and the
    # other two clauses read the same array. A second depth measurement here is
    # precisely the drift this module exists to prevent.
    sustained = sustained_gate(
        detrended,
        events,
        min_depth_sigma=(thr_sigma if thr_sigma > 0 else 0.0),
        scalars=scalars,
        polarity=polarity,
        cancel=cancel,
    )
    ds = sustained.depth_sust_pA
    m = ds.size
    finite = np.isfinite(ds)

    keep_sigma = (
        sustained.keep if thr_sigma > 0 else np.ones(m, dtype=bool)
    )
    keep_pa = (ds >= thr_pa) & finite if thr_pa > 0 else np.ones(m, dtype=bool)

    i0_local = np.full(m, np.nan, dtype=np.float64)
    fracblock = np.full(m, np.nan, dtype=np.float64)
    drift = 0.0
    if thr_frac > 0:
        if raw is None or fs is None:
            raise ValueError(
                "min_fracblock is on but no raw trace / sample rate was supplied. "
                "dI/I0 is measured against the TIME-LOCAL open-pore current, which "
                "only the RAW (un-detrended) signal carries -- the detrended trace "
                "has had exactly that level removed. Pass raw= and fs=, or set "
                "min_fracblock=0.0; approximating I0 here would silently change "
                "which events are kept."
            )
        rawt = _oriented(_as_trace(np.asarray(raw), "raw"), polarity)
        starts, _ends, _idx = _windows(events, rawt.size)
        i0v, _i0_med, drift = local_open_pore_pA(
            rawt, sustained.scalars.quiet, float(fs), at=starts
        )
        i0_local = np.abs(np.asarray(i0v, dtype=np.float64))
        with np.errstate(invalid="ignore", divide="ignore"):
            fracblock = ds / np.maximum(i0_local, 1e-9)
        keep_frac = (fracblock >= thr_frac) & finite
    else:
        keep_frac = np.ones(m, dtype=bool)

    return AmplitudeGateResult(
        keep=keep_sigma & keep_pa & keep_frac,
        keep_sigma=keep_sigma,
        keep_pa=keep_pa,
        keep_fracblock=keep_frac,
        sustained=sustained,
        i0_local_pA=i0_local,
        fracblock=fracblock,
        i0_drift=float(drift),
        min_depth_sigma=thr_sigma,
        min_depth_pa=thr_pa,
        min_fracblock=thr_frac,
    )


# ===========================================================================
# the thirteen features
# ===========================================================================


def shannon_entropy(d, bins: int = ENTROPY_BINS) -> float:
    """Normalised Shannon entropy (0-1) of an event's amplitude distribution.

    ~0 means the samples sit at one level (a flat plateau); ~1 means they spread
    evenly across levels (a multi-level, substate-rich interior).

    **Fixed bin count, natural log, normalised by ``log(bins)``.** The bin edges
    are ``np.histogram``'s default over the segment's own ``[min, max]``, so the
    binning adapts in *width* per event but never in *count* -- this is not a
    Freedman-Diaconis rule, and swapping one in changes every value. Empty bins are
    dropped before normalising. Returns ``0.0`` for fewer than
    :data:`SKEW_KURT_MIN_SAMPLES` samples (the histogram would be too crude) or for
    a perfectly flat segment.
    """
    d = np.asarray(d, float)
    if len(d) < SKEW_KURT_MIN_SAMPLES or float(d.max() - d.min()) <= 0:
        return 0.0
    p, _ = np.histogram(d, bins=bins)
    p = p[p > 0].astype(float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum() / np.log(bins))


def _one_event(d: np.ndarray, i0: float, noise: float, fs: float) -> dict[str, float]:
    """The thirteen features (plus ``dwell_ms``) for one already-deviated segment.

    ``d = baseline - segment``, so positive is deeper. Kept as a private helper so
    the loop body and the documented formulas are the same code.
    """
    n = len(d)
    dwell_ms = n / fs * 1000.0
    dmax = float(d.max())
    dsust = float(np.median(d))
    return {
        "dwell_ms": dwell_ms,
        "log_dwell": float(np.log10(max(dwell_ms, 1e-3))),
        "depth_max_pA": dmax,
        "depth_sust_pA": dsust,
        "mean_pA": float(d.mean()),
        "std_pA": float(d.std()),  # population, ddof=0
        "min_dev_pA": float(d.min()),  # the SHALLOWEST sample; goes negative
        "skewness": float(_sps.skew(d)) if n >= SKEW_KURT_MIN_SAMPLES else 0.0,
        "kurtosis": float(_sps.kurtosis(d)) if n >= SKEW_KURT_MIN_SAMPLES else 0.0,
        "area_pA_ms": float(d.sum() / fs * 1000.0),
        "t_peak_frac": float(np.argmax(d) / max(n - 1, 1)),
        "entropy": shannon_entropy(d),
        "f_sust": (dsust / i0 if i0 and np.isfinite(i0) else np.nan),
        "snr": (dsust / noise if noise else np.nan),
    }


def event_features(
    detrended,
    fs: float,
    events,
    *,
    scalars: FeatureScalars,
    polarity: int = 1,
    min_samples: int = MIN_EVENT_SAMPLES,
    cancel: CancelFlag | None = None,
) -> pd.DataFrame:
    """The thirteen clustering features (plus ``dwell_ms``), one row per event.

    Every feature is a statistic of the per-event deviation
    ``d = scalars.baseline_pA - segment`` over the event's ``[start_idx, end_idx)``
    window of the DETRENDED trace:

    ==================  ===========================================================
    ``dwell_ms``        ``n / fs * 1000``
    ``log_dwell``       ``log10(max(dwell_ms, 1e-3))``, base 10
    ``depth_max_pA``    ``max(d)`` -- the deepest single sample
    ``depth_sust_pA``   ``median(d)`` -- the SUSTAINED depth, the same statistic
                        the gate uses (against a different baseline; see the
                        module docstring)
    ``mean_pA``         ``mean(d)``
    ``std_pA``          ``std(d)``, population (ddof=0)
    ``min_dev_pA``      ``min(d)`` -- the SHALLOWEST sample, negative when a sample
                        rises above baseline. It is not a magnitude.
    ``skewness``        ``scipy.stats.skew`` at defaults (bias=True)
    ``kurtosis``        ``scipy.stats.kurtosis`` at defaults (Fisher/excess,
                        bias=True)
    ``area_pA_ms``      ``sum(d) / fs * 1000``, signed
    ``t_peak_frac``     ``argmax(d) / max(n-1, 1)`` -- where in the event the
                        deepest sample fell, 0 at onset and 1 at offset. First
                        maximum on ties.
    ``entropy``         :func:`shannon_entropy`, 8 fixed bins
    ``f_sust``          ``median(d) / scalars.i0_pA`` -- dI/I0 from the SUSTAINED
                        depth and the GLOBAL I0
    ``snr``             ``median(d) / scalars.noise_pA`` -- against the WHOLE-trace
                        MAD, not the gate's event-excluded one
    ==================  ===========================================================

    Args:
        detrended: the detrended current in pA. Dtype preserved (see FLOAT
            FIDELITY in the module docstring).
        fs: sample rate in Hz; must be finite and > 0 (three features carry a time
            unit).
        events: an :class:`~nanoscope.analysis.events.EventSet`, a frame with
            ``start_idx``/``end_idx``, or an ``(m, 2)`` array of half-open windows.
            The gate is NOT applied here -- pass the already-gated rows (see
            :func:`sustained_gate`) if that is what you want measured.
        scalars: the recording's :class:`FeatureScalars`. Required, and deliberately
            not defaulted: the whole-trace baseline/MAD/I0 are a property of the
            recording, not of this event subset, and recomputing them from a slice
            would quietly produce a different feature table.
        polarity: ``+1`` (default) or ``-1``. Cell 51 has no polarity handling
            because every Zn2 recording is ``+1``; ``-1`` mirrors the trace, which
            is the exact generalisation and a no-op at ``+1``.
        min_samples: windows shorter than this are not measured and are ABSENT from
            the result. Defaults to :data:`MIN_EVENT_SAMPLES` (3); floored at 1,
            since an empty window has no statistics at all.
        cancel: cancellation token; the ambient one is used when ``None``.

    Returns:
        A ``float64`` frame with :data:`FEATURE_COLUMNS`, **indexed by the row
        labels of the events that were measured**. Rows dropped for
        ``min_samples`` are missing from the index, so
        ``events.df.loc[result.index]`` recovers the matching metadata and no
        positional assignment is ever needed. An empty input yields an empty frame
        with the right columns and dtypes.

    Raises:
        ValueError: on a bad ``fs``, a bad trace, or an event table without
            ``start_idx``/``end_idx``.
    """
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be finite and > 0, got {fs!r}; dwell_ms needs a time base")

    sig = _oriented(_as_trace(detrended, "detrended"), polarity)
    starts, ends, index = _windows(events, sig.size)
    m = len(starts)

    med = scalars.baseline_pA
    i0 = scalars.i0_pA
    noise = scalars.noise_pA
    # A zero or negative floor would let an empty window through to ``d.max()``.
    floor = max(int(min_samples), 1)

    rows: list[dict[str, float]] = []
    keep_pos: list[int] = []
    poll = CancelPoller(cancel)
    for i in range(m):
        poll()
        s, e = int(starts[i]), int(ends[i])
        if e - s < floor:
            continue
        # `med - seg.astype(float)` -- the explicit promotion is the notebook's,
        # and it is what makes every per-event statistic float64 off a float32
        # trace.
        d = med - sig[s:e].astype(float)
        rows.append(_one_event(d, i0, noise, fs))
        keep_pos.append(i)

    if not rows:
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in FEATURE_COLUMNS},
            index=index[:0],
        )
    out = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS), index=index[keep_pos])
    return out.astype("float64")


# ===========================================================================
# early / late split
# ===========================================================================


def _one_span(value, what: str) -> tuple[float, float]:
    """Coerce ``value`` to a ``(t0, t1)`` clock window, in seconds.

    A bare number is a DURATION and means ``(0.0, value)`` -- the ordinary case,
    where onsets are measured from the start of the recording.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        t0, t1 = 0.0, float(arr)
    elif arr.shape == (2,):
        t0, t1 = float(arr[0]), float(arr[1])
    else:
        raise ValueError(
            f"{what} must be a duration in seconds or a (t0, t1) pair, got "
            f"{value!r}"
        )
    if not (np.isfinite(t0) and np.isfinite(t1)):
        raise ValueError(f"{what} is not finite: ({t0}, {t1})")
    if not (t1 > t0):
        raise ValueError(
            f"{what} must span a positive interval, got (t0={t0}, t1={t1}). A "
            "recording with no duration has no halves, and falling back to the "
            "events' own median here is exactly the substitution this argument "
            "exists to prevent."
        )
    return (t0, t1)


def _resolve_spans(span, keys, *, group_given: bool):
    """``None``, or ``{group key: (t0, t1)}`` covering every key in ``keys``."""
    if span is None:
        return None
    if isinstance(span, Mapping):
        if not group_given:
            raise ValueError(
                "span is a per-group mapping but no `group` was given; without "
                "groups there is nothing to key it by"
            )
        out = {}
        for k in keys:
            if k not in span:
                raise ValueError(
                    f"span has no clock window for group {k!r}; a group whose "
                    f"recording length is unknown cannot be split by the clock, "
                    f"and guessing it from that group's own events is the defect "
                    f"this argument exists to prevent. Keys given: "
                    f"{sorted(map(repr, span))}"
                )
            out[k] = _one_span(span[k], f"span[{k!r}]")
        if not out:
            # An empty recording has no keys to look up, so nothing above ran.
            # Validate what was handed in anyway: a caller with a broken clock
            # should hear about it here rather than on its next recording.
            for k, v in span.items():
                _one_span(v, f"span[{k!r}]")
        return out
    one = _one_span(span, "span")
    return {k: one for k in keys}


def time_half(onset_s, group=None, *, span=None) -> np.ndarray:
    """``"early"``/``"late"`` per event, split at the RECORDING'S CLOCK midpoint.

    THE INVARIANT: **a split defined by the sample cannot measure the sample's
    own shape.** This function used to split at each group's own MEDIAN onset,
    and a median halves its own sample by definition -- so the pooled first-half
    fraction computed from it was 0.50 on every recording, for ever, however
    lopsided the event supply actually was. Everything downstream that reads
    ``early_frac`` as "the fraction that fell in the first half of the
    recording" (the population table's caption, and
    :func:`~nanoscope.analysis.subpop_gmm._stability_warnings`' recording-level
    non-stationarity caveat) was therefore reading a constant. The app looked
    like it checked for drift and could not.

    Pass ``span`` -- the recording's own clock -- and the half means what the
    caption says. Two consequences worth stating, because both are the point
    rather than a side effect:

    * **A scope whose events do not span the recording lands entirely in one
      half.** That is the correct answer, not a degenerate one: a class living
      in the first tenth of a recording IS concentrated in its first half, and
      the median split reported it as a comfortable 50/50.
    * **``group`` no longer decides where the boundary is** -- only which clock
      applies. Within one recording every scope shares one boundary, so scopes
      become comparable with each other; a table pooled across recordings passes
      a mapping and each recording keeps its own clock.

    ``span=None`` keeps the old median split. It is retained for ONE purpose --
    reproducing the notebook's ``time_half`` column bit for bit (the reference
    recordings split exactly 54/54, 55/55, 97/97 and 538/538, which is the
    signature of a median and not of a clock) -- and it must not be used to
    measure non-stationarity. Nothing in the application takes that path:
    :func:`~nanoscope.analysis.subpop_report.feature_table` always supplies a
    clock -- the span the rows were DETECTED in
    (:func:`~nanoscope.analysis.subpop_report.analysed_span_s`), which is the
    whole trace when the detector was pointed at the whole trace and the
    sub-range when it was not. "The recording's clock" below means that span:
    the length of the FILE is the wrong one whenever the two differ, and reading
    it split a 22 s clock under events detected in 8.8 s.

    Args:
        onset_s: per-event onset in seconds, on the same clock as ``span``.
        group: per-event grouping key (the recording tag or incubation hours).
            ``None`` treats every event as one group.
        span: the recording's clock. A duration in seconds (meaning
            ``(0.0, duration)``), an explicit ``(t0, t1)`` pair, or a mapping
            from group key to either of those. ``None`` falls back to the
            per-group median split -- see above for why that is not a
            measurement of the recording.

    Returns:
        An object array of ``"early"``/``"late"``, positionally aligned with
        ``onset_s``. The boundary is ``<=`` -> ``"early"``, unchanged.

    Raises:
        ValueError: if ``group`` does not align with ``onset_s``, if ``span``
            does not span a positive interval, or if a per-group mapping is
            missing a key.
    """
    t = np.asarray(onset_s, dtype=np.float64).ravel()
    out = np.empty(t.shape, dtype=object)
    if group is None:
        keys = np.zeros(t.shape, dtype=np.int8)
    else:
        keys = np.asarray(group).ravel()
        if keys.shape != t.shape:
            raise ValueError(
                f"group has {keys.shape} entries but onset_s has {t.shape}; they are "
                "per-event and must align"
            )
    if t.size == 0:
        # Still validate `span`, so a caller that hands in a broken clock hears
        # about it on the empty recording too rather than only on the next one.
        _resolve_spans(span, [], group_given=group is not None)
        return out
    uniq = pd.unique(pd.Series(keys))
    spans = _resolve_spans(span, uniq, group_given=group is not None)
    for k in uniq:
        mm = keys == k
        if spans is None:
            mid = float(np.median(t[mm]))
        else:
            t0, t1 = spans[k]
            mid = 0.5 * (t0 + t1)
        out[mm] = np.where(t[mm] <= mid, "early", "late")
    return out
