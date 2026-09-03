"""Signature translocation-plot maths: ECD, dwell-vs-blockage density, event ensemble.

These are the three shapes every translocation paper leads with, computed here as
pure ``(signal, fs, EventSet) -> numpy`` kernels with no Qt, no Session and — the
load-bearing constraint — **no new EventSet columns**. The frozen 20-column
``events.SCHEMA`` is a cross-module contract with a parquet/project round-trip
riding on it, so the equivalent-charge-deficit and the ensemble traces are
reconstructed ON DEMAND from what is already persisted: each event's stored
scalar ``baseline_pA`` plus its ``[start_idx, end_idx)`` window into the
recording's calibrated current. Nothing here is stored back.

THE ECD DEFINITION -- READ THIS
-------------------------------
:func:`charge_deficit` returns, per event, the **signed** integral

    ECD_i = integral over [start_i, end_i) of (baseline_pA_i - current(t)) dt

with ``dt = 1/fs`` (a left Riemann sum over the event's samples, matching the
``ecd_pA_s`` convention in :mod:`nanoscope.analysis.characterize`). The sign is
literal and deliberate: a **downward** blockage (current below the local baseline)
gives a **positive** deficit, an **upward** enhancement gives a negative one. This
is the classic "equivalent charge deficit" -- the area between the open-pore level
and the current trace during the event -- in ``pA*s``. Unlike the stored
``ecd_pA_s`` (which is polarity-rectified to always be positive and integrates the
detector's full-length baseline array), this is signed and uses each event's own
scalar ``baseline_pA``, so it is reproducible from the persisted EventSet alone.

Sign note for negative-I0 recordings: the definition is the literal
``baseline - current``. On a recording whose open pore sits at I0 < 0, a blockade
moves the current UP (toward zero), so ``baseline - current`` is negative there --
that is the honest consequence of the literal formula, and the caller who wants a
magnitude takes ``abs`` (or histograms ``|ECD|``). The formula is not rectified
here on purpose: keeping the sign is what lets a density plot separate blockages
from enhancements.
"""

from __future__ import annotations

import numpy as np

from .events import EventSet

__all__ = ["charge_deficit", "density_2d", "event_ensemble", "PA_S_TO_FC"]

# 1 pA*s = 1e-12 C = 1e3 fC. The canonical ``gain`` for charge_deficit(...) when
# the caller wants femtocoulombs out of a pA-units recording.
PA_S_TO_FC: float = 1.0e3


def charge_deficit(
    signal: np.ndarray,
    fs: float,
    eventset: EventSet,
    *,
    gain: float | None = None,
) -> np.ndarray:
    """Equivalent charge deficit (ECD) for every event in ``eventset``.

    For each row ``i`` this integrates ``baseline_pA_i - current(t)`` over the
    event's ``[start_idx, end_idx)`` window as a left Riemann sum with ``dt=1/fs``::

        ECD_i = (baseline_pA_i * n_i - sum(signal[start_i:end_i])) / fs

    which is exactly ``depth * dwell`` for an ideal square blockade. The value is
    SIGNED (a downward blockage is positive; see the module docstring).

    Args:
        signal: The recording's calibrated current, one array for the whole
            channel (the same trace the events were detected on). Indexed by the
            EventSet's absolute ``start_idx``/``end_idx``.
        fs: Sample rate in Hz. Must be finite and > 0 (ECD has a ``*s`` in its
            units, so it is undefined without a time base).
        eventset: The :class:`~nanoscope.analysis.events.EventSet` to measure. One
            value is returned per row of ``eventset.df`` (the caller filters to
            ``eventset.accepted`` first if it wants accepted-only, mirroring how
            the plots window sources its rows). Uses each row's own stored
            ``baseline_pA`` and ``start_idx``/``end_idx``.
        gain: Optional multiplicative conversion applied to the result. ``None``
            returns the ECD in ``pA*s``; pass :data:`PA_S_TO_FC` (``1e3``) to get
            femtocoulombs from a pA-units recording, or any other pA*s->charge
            factor. It only scales the returned array; it never changes the sign.

    Returns:
        ``float64`` array of length ``len(eventset.df)`` in ``pA*s`` (or the
        ``gain``-scaled unit). Events whose window is entirely off the end of
        ``signal`` yield ``NaN``; a partially-overlapping window is integrated
        over its in-bounds portion. An empty EventSet yields an empty array.

    Robustness:
        ``start_idx``/``end_idx`` are clipped to ``[0, len(signal)]`` before
        integrating, so an event at the very edge (or a signal reloaded over a
        narrower sample range than the events were detected on) never indexes out
        of bounds.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = signal.size
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be finite and > 0, got {fs!r}; ECD (pA*s) needs a time base")

    df = eventset.df
    m = len(df)
    if m == 0:
        return np.empty(0, dtype=np.float64)

    starts = np.clip(df["start_idx"].to_numpy(dtype=np.int64), 0, n)
    ends = np.clip(df["end_idx"].to_numpy(dtype=np.int64), 0, n)
    baseline = df["baseline_pA"].to_numpy(dtype=np.float64)
    lengths = (ends - starts).astype(np.float64)

    # One prefix sum over the whole trace turns every per-event window sum into an
    # O(1) difference -- O(n + m) total, no Python loop over events or samples.
    csum = np.concatenate(([0.0], np.cumsum(signal)))
    seg_sum = csum[ends] - csum[starts]

    out = (baseline * lengths - seg_sum) / fs
    out[lengths <= 0.0] = np.nan
    if gain is not None:
        out = out * float(gain)
    return out


def density_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xbins: int,
    ybins: int,
    xlog: bool = False,
    ylog: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D histogram for the dwell-vs-blockage density map.

    Points where either coordinate is non-finite are dropped; with ``xlog`` /
    ``ylog`` the corresponding axis is binned in log space and its non-positive
    values are dropped too (they have no place on a log axis). A point is kept
    only if BOTH coordinates survive, so it can actually be placed.

    Args:
        x: X metric, one value per event (e.g. ``dwell_s``).
        y: Y metric, one value per event (e.g. ``depth_pA``).
        xbins: Number of bins along X (>= 1).
        ybins: Number of bins along Y (>= 1).
        xlog: Bin X in log10 space (edges are geometric).
        ylog: Bin Y in log10 space.

    Returns:
        ``(counts2d, xedges, yedges)`` where ``counts2d`` has shape
        ``(xbins, ybins)`` (``counts2d[i, j]`` = events in X-bin ``i``, Y-bin
        ``j``) and the edge arrays have length ``bins + 1``. Edges are
        deterministic: the data min/max span the range (geometric when logged),
        and a degenerate or empty axis falls back to a fixed unit span so the
        result is always a valid, reproducible grid.
    """
    xbins = int(xbins)
    ybins = int(ybins)
    if xbins < 1 or ybins < 1:
        raise ValueError(f"xbins and ybins must be >= 1, got {xbins} and {ybins}")

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"x ({x.size}) and y ({y.size}) must have the same length")

    mask = np.isfinite(x) & np.isfinite(y)
    if xlog:
        mask &= x > 0
    if ylog:
        mask &= y > 0
    x, y = x[mask], y[mask]

    xedges = _bin_edges(x, xbins, xlog)
    yedges = _bin_edges(y, ybins, ylog)
    counts, xe, ye = np.histogram2d(x, y, bins=(xedges, yedges))
    return counts, xe, ye


def _bin_edges(v: np.ndarray, nbins: int, log: bool) -> np.ndarray:
    """Deterministic ``nbins + 1`` edges spanning ``v`` (geometric when ``log``).

    ``v`` is assumed already finite (and positive when ``log``). Empty and
    single-valued inputs fall back to a fixed span so the edges are always
    strictly increasing -- ``np.histogram2d`` rejects non-monotonic edges.
    """
    if v.size:
        lo, hi = float(np.min(v)), float(np.max(v))
    else:
        lo, hi = (1.0, 10.0) if log else (0.0, 1.0)
    if log:
        if lo <= 0.0:
            lo = 1e-12
        if hi <= lo:
            hi = lo * 10.0
        edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    else:
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, nbins + 1)
    # Guard the log10->10** round-trip: an outer edge that lands a hair INSIDE the
    # data range would drop the extreme point. Clamp the outer edges to the true
    # min/max so every in-range point is always binned.
    if v.size:
        edges[0] = min(edges[0], lo)
        edges[-1] = max(edges[-1], hi)
    return edges


def event_ensemble(
    signal: np.ndarray,
    fs: float,
    eventset: EventSet,
    *,
    n_points: int,
    pad_frac: float = 0.0,
    align: str = "start",
    normalize_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample every accepted event onto a common axis for overlay/averaging.

    The signature nanopore "event ensemble": each accepted event's current window
    is interpolated onto ``n_points`` samples so the events can be drawn on top of
    one another and averaged into a mean/median waveform.

    Two axis conventions:

    * ``normalize_time=True`` (default) -- each event is DURATION-normalized: its
      ``[start, end)`` window (extended by ``pad_frac`` of the dwell on each side)
      maps onto ``T in [-pad_frac, 1+pad_frac]``, so events of different dwell
      overlay by shape. ``align`` is immaterial here (each event fills the same
      normalized span).
    * ``normalize_time=False`` -- real-time alignment on a COMMON sample window of
      width ``max_dwell * (1 + 2*pad_frac)``: ``align="start"`` puts every event's
      start at ``T=0``; ``align="center"`` puts every event's centre at ``T=0``.
      ``T`` is in seconds (relative to that reference).

    Args:
        signal: The recording's calibrated current (whole channel).
        fs: Sample rate in Hz (used for the seconds axis when
            ``normalize_time=False``).
        eventset: Source events; only the ACCEPTED rows are used.
        n_points: Length of the common axis every event is resampled to (>= 2).
        pad_frac: Fraction of each event's dwell to include as pre/post padding
            (0 = the event span only).
        align: ``"start"`` or ``"center"`` (only affects ``normalize_time=False``).
        normalize_time: Duration-normalize each event (True) or align on a common
            real-time window (False).

    Returns:
        ``(T, matrix, mean_trace, median_trace)``:

        * ``T`` -- the common axis, length ``n_points`` (unitless normalized time,
          or seconds).
        * ``matrix`` -- shape ``(n_kept, n_points)``, one resampled current trace
          per KEPT event. Events whose (padded) window falls off either end of the
          signal are skipped; ``n_kept = n_accepted - n_skipped``, so the skip
          count is ``len(eventset.accepted.df) - matrix.shape[0]``.
        * ``mean_trace`` / ``median_trace`` -- shape ``(n_points,)``, reduced over
          the kept events (all-``NaN`` when nothing was kept).
    """
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if align not in ("start", "center"):
        raise ValueError(f"align must be 'start' or 'center', got {align!r}")
    pad_frac = float(pad_frac)
    if pad_frac < 0:
        raise ValueError(f"pad_frac must be >= 0, got {pad_frac}")

    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = signal.size

    acc = eventset.accepted.df
    starts = acc["start_idx"].to_numpy(dtype=np.int64)
    ends = acc["end_idx"].to_numpy(dtype=np.int64)

    if normalize_time:
        T = np.linspace(-pad_frac, 1.0 + pad_frac, n_points, endpoint=False)
        rows = _ensemble_normalized(signal, n, starts, ends, n_points, pad_frac)
    else:
        T, rows = _ensemble_realtime(
            signal, n, fs, starts, ends, n_points, pad_frac, align)

    if rows:
        matrix = np.vstack(rows)
        mean_trace = matrix.mean(axis=0)
        median_trace = np.median(matrix, axis=0)
    else:
        matrix = np.empty((0, n_points), dtype=np.float64)
        mean_trace = np.full(n_points, np.nan)
        median_trace = np.full(n_points, np.nan)
    return T, matrix, mean_trace, median_trace


def _ensemble_normalized(
    signal: np.ndarray, n: int, starts: np.ndarray, ends: np.ndarray,
    n_points: int, pad_frac: float,
) -> list[np.ndarray]:
    """Duration-normalized rows: each event's padded window -> ``n_points`` samples."""
    grid = np.arange(n, dtype=np.float64)
    rows: list[np.ndarray] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        length = e - s
        if length <= 0:
            continue
        pad = pad_frac * length
        a, b = s - pad, e + pad
        if a < 0.0 or b > n:          # padded window runs off the trace edge -> skip
            continue
        targets = np.linspace(a, b, n_points, endpoint=False)
        rows.append(np.interp(targets, grid, signal))
    return rows


def _ensemble_realtime(
    signal: np.ndarray, n: int, fs: float, starts: np.ndarray, ends: np.ndarray,
    n_points: int, pad_frac: float, align: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Real-time rows on a common sample window, aligned at the event start/centre."""
    lengths = ends - starts
    valid = lengths > 0
    if not np.any(valid):
        return np.zeros(n_points, dtype=np.float64), []

    width = int(lengths[valid].max())
    wpad = int(round(pad_frac * width))
    total = width + 2 * wpad                       # common window length in samples

    if align == "center":
        mids = (starts + ends) / 2.0
        a_all = np.round(mids - total / 2.0).astype(np.int64)
        ref = total / 2.0                          # centre sits at this local offset
    else:  # start
        a_all = starts - wpad
        ref = float(wpad)                          # start sits at this local offset

    src = np.arange(total, dtype=np.float64)
    tgt = np.linspace(0.0, total - 1, n_points)
    denom = float(fs) if (fs and np.isfinite(fs) and fs > 0) else 1.0
    T = (tgt - ref) / denom

    rows: list[np.ndarray] = []
    for a, ok in zip(a_all.tolist(), valid.tolist()):
        if not ok:
            continue
        b = a + total
        if a < 0 or b > n:                         # window runs off the trace edge -> skip
            continue
        rows.append(np.interp(tgt, src, signal[a:b]))
    return T, rows
