"""The all-points amplitude histogram — how a pore's LEVELS are read before any detector.

A pure ``(signal,) -> numpy`` kernel with no Qt and no
:class:`~nanoscope.core.session.Session`; the view around it lives in
:mod:`nanoscope.gui.psd_panel` beside the noise spectrum, because the two are the same
kind of tool: whole-trace characterisation that needs **no detection, no provenance and
no schema**. Nothing here can touch the frozen 20-column ``events.SCHEMA`` — it never
sees an event.

WHAT IT IS FOR
--------------
Histogram the sample VALUES, not the time axis. Every sample of the span falls in one
bin, so a recording that spends most of its time at the open-pore current and a little
of it blocked shows a tall peak at the open level and a smaller one at each blocked
level, with the depth ``ΔI`` readable straight off the current axis as the distance
between them. That is the classic single-channel / nanopore amplitude histogram, and it
answers the three questions a user has BEFORE choosing a detector:

* where is the open-pore level, and is it one level or two (a flickering pore)?
* how deep are the blockades, in the same units as the trace?
* is the baseline drifting (a peak smeared into a plateau) or steady (a narrow peak)?

THE COUNT AXIS IS LOGARITHMIC, AND THAT IS THE WHOLE POINT
----------------------------------------------------------
Translocations are RARE. On the owner's 15 s apo-ferritin run the events occupy well
under 1% of the samples, so on a linear count axis the blocked level is a flat line on
the floor next to a peak 10³ times taller and the view says nothing that the mean did
not already say. On a log axis the same 0.1%-occupancy level is a clearly separated
bump. This module therefore reports counts as exact integers and leaves the axis to the
view, but :func:`level_peaks` — which is what turns the picture into numbers — is built
to find the small peak next to the big one rather than only the mode.

UNITS
-----
Unit-agnostic, exactly as :mod:`nanoscope.analysis.psd`: the bin edges and every level
come back in whatever units the calibrated trace carried (pA in, pA out). Nothing here
converts, infers or labels a unit — the caller owns the label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AllPointsHistogram",
    "all_points_histogram",
    "level_peaks",
    "DEFAULT_BINS",
    "QUANTILE_SUBSAMPLE_CAP",
]

#: Bin count the GUI opens with. 512 bins over a typical ±10% blockade range puts ~20
#: bins across an open-pore peak of realistic width, which resolves a split baseline
#: without turning Poisson counting noise into visible grass at the 1-count level.
DEFAULT_BINS = 512

#: Above this many finite samples, a quantile-clipped display span is estimated from a
#: strided subsample rather than from every sample. This affects ONLY where the drawn
#: window starts and stops — never a count. ``np.quantile`` of 8 M float64 costs a full
#: 64 MB copy plus a partition (~0.2 s) to move a display edge by less than one bin, and
#: :class:`AllPointsHistogram` reports ``n_below``/``n_above`` exactly either way, so the
#: reader is never told a clipped sample was counted.
QUANTILE_SUBSAMPLE_CAP = 1_000_000


@dataclass(frozen=True)
class AllPointsHistogram:
    """Counts per amplitude bin, plus every sample the drawn window does NOT hold.

    ``counts`` and ``edges`` follow :func:`numpy.histogram` (``edges.size ==
    counts.size + 1``). The four tallies are what keep the picture honest: a histogram
    drawn over a clipped span silently DISCARDS the samples outside it, and a spike that
    was clipped away is exactly the kind of thing a user needs told rather than hidden.
    """
    counts: np.ndarray          # int64, one per bin
    edges: np.ndarray           # float64, counts.size + 1
    lo: float                   # low edge of the histogrammed span
    hi: float                   # high edge
    n_total: int                # samples handed in
    n_finite: int               # of those, finite ones
    n_nonfinite: int            # NaN/inf: counted, never binned, always reported
    n_below: int                # finite samples below `lo` (clipped out)
    n_above: int                # finite samples above `hi` (clipped out)
    clipped: bool               # True == the span is narrower than the data range

    @property
    def centers(self) -> np.ndarray:
        """Bin centres — the x the view plots the counts against."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def bin_width(self) -> float:
        """One bin's width in signal units: the resolution any level below is read to."""
        return float(self.edges[1] - self.edges[0]) if self.edges.size > 1 else 0.0

    @property
    def n_binned(self) -> int:
        """Samples actually in the picture. ``n_finite - n_below - n_above``."""
        return int(self.counts.sum())

    @property
    def mode_center(self) -> float:
        """The most populated bin's centre — the open-pore level on a quiet trace."""
        if not self.counts.size or not self.counts.any():
            return float("nan")
        return float(self.centers[int(np.argmax(self.counts))])

    @property
    def mode_count(self) -> int:
        return int(self.counts.max()) if self.counts.size else 0


def all_points_histogram(
    signal: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    span: tuple[float, float] | None = None,
    clip_quantile: float = 0.0,
) -> AllPointsHistogram:
    """Histogram the sample values of one span. Counts are exact; the window is a choice.

    Args:
        signal: The 1-D span (already sliced by the caller — this function is never
            handed more of a recording than the user asked to look at).
        bins: Number of equal-width bins, ``>= 1``.
        span: Explicit ``(lo, hi)`` window. ``None`` derives one (see ``clip_quantile``).
        clip_quantile: Ignored when ``span`` is given. ``0.0`` (default) uses the full
            finite ``[min, max]`` so every sample is in the picture. A small value such
            as ``1e-4`` windows on ``[q, 1-q]`` instead, which is what makes a trace with
            one saturating spike readable: without it the spike sets the range and the
            entire pore signal collapses into a single bin. The discarded samples are
            counted into ``n_below``/``n_above``, never dropped silently.

    Returns:
        :class:`AllPointsHistogram`.

    Raises:
        ValueError: on an empty signal, ``bins < 1``, an inverted ``span``, or a span
            that no finite sample could fall in.

    Non-finite samples are EXCLUDED from the histogram and reported in ``n_nonfinite``
    rather than being coerced to an edge bin: a NaN has no amplitude, and piling NaNs
    onto the lowest bin would invent a level that the pore never sat at.
    """
    x = np.asarray(signal)
    if x.ndim != 1:
        x = x.ravel()
    n_total = int(x.size)
    if n_total == 0:
        raise ValueError("all_points_histogram needs at least one sample; got an empty span")
    if int(bins) < 1:
        raise ValueError(f"all_points_histogram needs bins >= 1; got {bins!r}")

    finite_mask = np.isfinite(x)
    n_finite = int(np.count_nonzero(finite_mask))
    n_nonfinite = n_total - n_finite
    if n_finite == 0:
        raise ValueError(
            f"all_points_histogram found no finite samples in {n_total:,}: every sample is "
            f"NaN or inf, so the recording has no amplitude to histogram.")
    # float64 FOR THE BINNING, always. numpy takes the bin edges' dtype from the DATA, so a
    # float32 recording gets float32 edges — and a channel sitting at 42.0 with a range of
    # 8e-8 then produced 512 edges that were all exactly 42.0 and a bin_width of 0.0.
    # float32 carries ~7 digits, so any span narrower than ~1e-6 of the offset (a quiet pore
    # on a large pedestal, a zoomed span) collapses the same way. The cast costs ~40 ms at
    # 8 M samples, measured, against a whole class of silently-degenerate pictures.
    xf = np.asarray(x[finite_mask] if n_nonfinite else x, dtype=np.float64)

    data_lo, data_hi = float(np.min(xf)), float(np.max(xf))
    if span is not None:
        lo, hi = float(span[0]), float(span[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            raise ValueError(f"all_points_histogram needs a finite lo < hi span; got {span!r}")
    elif clip_quantile > 0.0:
        lo, hi = _quantile_span(xf, float(clip_quantile))
    else:
        lo, hi = data_lo, data_hi

    if hi <= lo:
        # A constant span (every sample identical) is a real, common answer — a dead
        # channel, or a saturated one. Give it a bin of nonzero width centred on the
        # value so the view has something to draw and the level still reads correctly.
        pad = max(abs(lo) * 1e-9, 1e-12)
        lo, hi = lo - pad, hi + pad

    counts, edges = np.histogram(xf, bins=int(bins), range=(lo, hi))
    n_below = int(np.count_nonzero(xf < lo))
    n_above = int(np.count_nonzero(xf > hi))
    return AllPointsHistogram(
        counts=counts.astype(np.int64, copy=False),
        edges=np.asarray(edges, dtype=np.float64),
        lo=float(lo), hi=float(hi),
        n_total=n_total, n_finite=n_finite, n_nonfinite=n_nonfinite,
        n_below=n_below, n_above=n_above,
        clipped=bool(n_below or n_above),
    )


def _quantile_span(xf: np.ndarray, q: float) -> tuple[float, float]:
    """``[q, 1-q]`` quantiles of the finite samples, subsampled above the cap.

    See :data:`QUANTILE_SUBSAMPLE_CAP` for why the estimate is allowed to be approximate:
    it moves a DISPLAY edge, and every sample it puts outside the window is counted and
    reported by the caller.
    """
    q = float(np.clip(q, 0.0, 0.49))
    sample = xf
    if xf.size > QUANTILE_SUBSAMPLE_CAP:
        stride = int(np.ceil(xf.size / QUANTILE_SUBSAMPLE_CAP))
        sample = xf[::stride]
    lo, hi = np.quantile(np.asarray(sample, dtype=np.float64), [q, 1.0 - q])
    return float(lo), float(hi)


def level_peaks(
    hist: AllPointsHistogram,
    *,
    max_peaks: int = 4,
    smooth_bins: int = 11,
    min_separation_frac: float = 0.02,
    min_count_frac: float = 1e-4,
    prominence_ratio: float = 3.0,
) -> list[dict]:
    """The most populated amplitude LEVELS, largest first — the numbers behind the picture.

    A greedy peak pick, deliberately not :func:`scipy.signal.find_peaks`: the thing that
    matters here is that a blockade level holding **0.1% of the samples** is found next to
    an open-pore peak holding 99%, which is a question about the VALLEY between them, not
    about height in absolute counts. Measured on the two synthetics the tests pin: a
    N(100, 1) trace with 10% of its samples at N(60, 1) returns exactly the two levels,
    and one with only 0.1% at N(60, 1) — 1 000 samples of a million, a peak bin 950x
    shorter than the mode — still returns both.

    Args:
        hist: The histogram to read.
        max_peaks: At most this many levels come back.
        smooth_bins: Width of the moving average applied before the maximum test, in
            bins. Counting noise makes a smooth peak's top jitter by ``±sqrt(count)``, so
            an unsmoothed "greater than both neighbours" test finds dozens of spurious
            maxima on the shoulder of ONE real peak — measured: a single clean N(100, 1)
            level reported four "levels" spanning 99.69–100.38 at ``smooth_bins=3``, and
            one at 11. ``1`` disables smoothing.
        min_separation_frac: Minimum distance between two reported levels as a fraction
            of the histogram's span. Peaks closer than this are one level seen twice.
        min_count_frac: A level's SMOOTHED count must be at least this fraction of the
            tallest bin's count, and at least 1.0 (one sample per bin averaged over the
            smoothing window) whichever is larger. Set low on purpose — it exists only to
            kill the 1-count grass a log axis makes so visible, NOT to rank levels;
            ``prominence_ratio`` is what rejects a shoulder.
        prominence_ratio: A candidate must rise this many times above the LOWEST smoothed
            count between it and the nearest taller level already accepted. This is the
            test that separates a real second level (with an empty valley between, so the
            ratio is unbounded) from a bump on the tail of the first (whose valley is
            almost as tall as the bump). Measured: the 3.2-sigma tail bumps of a single
            Gaussian level sit at ratios of 1.1–1.4 and are rejected; a genuine blockade
            level's valley is empty and it is kept.

    Returns:
        A list of ``{"center", "count", "fraction", "index"}`` dicts, count-descending.
        ``fraction`` is of the BINNED samples (``hist.n_binned``), so the fractions of
        all levels sum to at most 1 and a clipped sample is not counted as occupancy.
    """
    counts = np.asarray(hist.counts, dtype=np.float64)
    if counts.size == 0 or not counts.any():
        return []
    k = max(int(smooth_bins), 1)
    if k > 1 and counts.size >= k:
        kernel = np.ones(k, dtype=np.float64) / k
        smooth = np.convolve(counts, kernel, mode="same")
    else:
        smooth = counts

    # Local maxima including the two ends: a level pinned against the edge of the drawn
    # window (a clipped range, or a saturating channel) is still a level.
    is_peak = np.ones(smooth.size, dtype=bool)
    is_peak[1:] &= smooth[1:] >= smooth[:-1]
    is_peak[:-1] &= smooth[:-1] >= smooth[1:]
    idx = np.flatnonzero(is_peak & (counts > 0))
    if idx.size == 0:
        return []

    order = idx[np.argsort(-smooth[idx], kind="stable")]
    centers = hist.centers
    span = float(hist.hi - hist.lo)
    min_sep = abs(span) * float(min_separation_frac)
    # A level must clear BOTH floors: a fraction of the tallest bin (kills grass under a
    # huge mode) and one sample per bin averaged over the smoothing window (kills the
    # single stray sample in an empty tail, which the fractional floor alone cannot —
    # measured: a clean single N(100,1) level reported two extra "levels" of ONE count
    # each, five sigma out, because 1 > 1e-4 x 1440 and the valley beyond the tail is
    # empty so nothing else rejected them).
    floor = max(float(counts.max()) * float(min_count_frac), 1.0)
    binned = max(hist.n_binned, 1)

    chosen: list[int] = []
    for i in order:
        if smooth[i] < floor:
            continue
        if any(abs(centers[i] - centers[j]) < min_sep for j in chosen):
            continue
        # Prominence against every TALLER level already accepted (the scan is
        # height-ordered, so `chosen` holds exactly those). The valley is the lowest
        # smoothed count strictly between the two bins; an empty valley (a real second
        # level) gives an unbounded ratio and always passes.
        keep = True
        for j in chosen:
            lo_i, hi_i = (i, j) if i < j else (j, i)
            valley = float(smooth[lo_i + 1:hi_i].min()) if hi_i - lo_i > 1 else 0.0
            if smooth[i] < float(prominence_ratio) * valley:
                keep = False
                break
        if not keep:
            continue
        chosen.append(int(i))
        if len(chosen) >= int(max_peaks):
            break
    chosen.sort(key=lambda i: -counts[i])
    return [{"center": float(centers[i]), "count": int(hist.counts[i]),
             "fraction": float(hist.counts[i]) / binned, "index": int(i)}
            for i in chosen]
