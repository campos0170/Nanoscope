"""Fit a hand-placed event's edges to the step they sit on, and find the event under a pointer.

Pure numpy, no GUI. Two public entry points and one helper, built to
``docs/CURSOR_ERGONOMICS_DESIGN.md`` (draft 3), sections 3.2, 5.1, 5.2 and 5.5:

* :func:`fit_edges`    -- move a placed start/end onto the step it sits on, or say why not.
* :func:`catch_event`  -- the ``C`` key: find the event under a pointer, then fit both edges.
* :func:`estimate_open_pore_tau` -- the autocorrelation time ``fit_edges`` needs, measured
  on open pore only.

Sign convention (``events.py``, obeyed everywhere): ``dev = polarity * (signal - baseline)``
is positive inside an event, so ``polarity=+1`` means the current moves ABOVE the baseline
(an ordinary blockade at a negative bias, whose open-pore current is negative) and ``-1``
means it moves below. The polarity is the target table's (``baseline.measured_polarity``
for a Manual table), never read off the window, which flips on drift.

THE STEP FIT (``method="step"``) is ``cusum._best_split``'s least-squares split, made
DIRECTED and OUTERMOST, with six rules that each answer a failure measured on
``truth.synth_trace``'s default noise (white plus 30% 1/f). In the order the code applies
them:

1. DIRECTED. Only splits whose step has the right sign in ``dev`` are candidates (up for a
   start, down for an end). Undirected, a sub-state step that runs the same way as the real
   edge wins: the end landed on the inner step in 185 to 200 of 200 trials.
2. CUT AT AN OPPOSITE STEP. Before the outer side is tested, the strongest opposite-sign
   step beyond the split is looked for; if it is real (local z and a 3-sigma size, judged
   only on the samples between it and the split) the window is cut there and split again,
   up to three times, and every later statistic is clipped at the cut. Without it a deeper
   neighbour that the table does not hold fills the outer half of the end window and the end
   is fitted in only 6 to 51% of trials.
3. OUTER SIDE AT BASELINE. Skip a guard of one rise past the split, average
   ``max(32, rise)`` samples, stopping at any cut; the mean must lie within
   ``max(3 sigma sqrt(tau/n), 1.5 sigma)`` of ``b_out``, the median of ``max(64, W)``
   samples beyond the search window (the contract's baseline track only when the window
   was cut). If it fails the outer segment is split again, outward; if it still fails the
   window is grown outward once. Testing right next to the split, or against the
   whole-recording track, failed on ramp tails, on 1/f drift and on close neighbours.
4. SIGNIFICANCE IS LOCAL. ``z`` is taken over at most ``L = max(64, 4 rise)`` samples each
   side of the split, clipped at a cut and at the other edge, divided by ``sqrt(tau)``,
   and must reach 6. A z over the whole window grows with the window on correlated noise
   (it passed on pure pink noise in 81% of 2,500-sample windows).
5. SIZE. The local step must be at least ``max(3 sigma, 0.15 * interior median dev)``. A
   0.3 factor rejected the outer step of every sub-state shallower than 30% of the depth.
6. BORDER. A split within ``max(min_len + 1, rise)`` of the OUTER window edge is retried
   once with the window grown outward; the test is skipped when the window already reaches
   a neighbour.

An edge that fails keeps its placed position and says why (``EdgeResult.reason``). A fit on
both edges is all or nothing at the ROW level (:attr:`EdgeFit.complete`): the caller decides
what to do with one good edge and one that failed; this module never mixes the two silently.

WHAT THE CALLER SUPPLIES
* ``signal``: the array the add door measures on (``detection_signal``; raw for a Manual table),
  indexed absolutely; ``baseline``: the contract's per-sample track for the SAME array.
* ``polarity``: the target table's, +1 or -1. ``noise_rms``: the table's ``noise_rms_pA``.
* ``tau``: :func:`estimate_open_pore_tau`, floored at ``cusum.filter_autocorr_time`` when the
  provenance has a low-pass (the open-pore number alone cannot see the filter's own short-range
  correlation: see its docstring).
* ``rise_s``: the table's median ``rise_s`` when it has rows, else None (a first-pass estimate);
  ``filter_rise_s``: the filter's 10-90% rise, the floor under every rise and the length of C's
  smoothing.
* ``px_samples``: samples per screen pixel at the current zoom; ``lo`` / ``hi``: the neighbouring
  rows' bounds.

KNOWN LIMITS, each pinned by a test
* The rise estimate (``rise_s=None``) works at 50 kHz and not at 250 kHz on a soft multi-step
  edge, which is left unfitted (design 5.1).
* An edge placed further out than the window ``W`` is usually left alone, but a split can hug the
  INNER window border when the real step lies just beyond it, because rule 4's statistics read up
  to ``L`` samples past the window. The measured rates are in the test file.
* ``b_out`` is a median of samples beyond the window; a real edge within 64 samples beyond it makes
  that median the event level, and the outer test then passes on the event's own interior.
* An event longer than the baseline estimator's rough window is partly absorbed into the CONTRACT's
  track, and the outer test then declines it.

Numbers are in ``tests/analysis/test_edge_fit.py``, which rebuilds the design review's
measurements as floors.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "EdgeResult",
    "EdgeFit",
    "Catch",
    "fit_edges",
    "catch_event",
    "estimate_open_pore_tau",
    "METHOD",
    "VERSION",
    "GATES",
    "RISE_FLOOR_SAMPLES",
    # the sentences a refusal or an unfitted edge carries (shown as they stand)
    "REASON_VIEW",
    "REASON_NO_EVENT",
    "REASON_ON_EVENT",
    "REASON_OUTSIDE",
    "REASON_NO_STEP",
    "REASON_OUTER",
    "REASON_BORDER",
    "REASON_Z",
    "REASON_SIZE",
    "REASON_SHORT",
    "REASON_WINDOW",
    "REASON_NONFINITE",
    "REASON_NOT_REQUESTED",
    "REASON_NO_BASELINE",
    "REASON_CROSS",
]

METHOD = "step"
#: Bumped whenever a rule or a threshold below changes what a fit returns; written to
#: ``fit_json`` so a stored row says which rule set produced it.
VERSION = 1

# ------------------------------------------------------------------------------------------------
# The thresholds. Named constants for the code, and ``GATES`` -- a read-only record of the same
# numbers -- written to ``fit_json`` so a stored row says what gated it.
# ------------------------------------------------------------------------------------------------
_Z_MIN = 6.0               # local two-sample z, already divided by sqrt(tau)
_SIZE_SIGMA = 3.0          # the step must be at least this many sigma ...
_SIZE_FRAC = 0.15          # ... and this fraction of the interior median dev
_OUTER_Z = 3.0             # outer mean within _OUTER_Z * sigma * sqrt(tau / n) of b_out ...
_OUTER_FLOOR_SIGMA = 1.5   # ... but never tighter than this many sigma
_GUARD_RISES = 1.0         # the outer test starts this many rises past the split
_OUTER_LEN_MIN = 32        # the outer test averages max(32 samples, 1 rise)
_OUTER_LEN_RISES = 1.0
_FLANK_MIN = 64            # b_out is the median of max(64, W) samples beyond the window
_LOCAL_MIN = 64            # local z over L = max(64, 4 rises) samples each side
_LOCAL_RISES = 4.0
_WINDOW_PX = 12.0          # W = max(12 px of the current zoom, 3 rises) ...
_WINDOW_RISES = 3.0
_SPAN_CAP = 0.45           # ... capped at 0.45 * span, and the cap wins
_MAX_CUTS = 3              # opposite-step cuts per edge
_MAX_RESPLIT = 8           # outward re-splits of a segment whose outer side is not baseline
_CATCH_TRIGGER_SIGMA = 4.0  # C: smoothed dev over the pointer's pixel +-1 px must reach this
_CATCH_FOOT_SIGMA = 2.0    # C: the foot, where the fit window is widened to
_CATCH_HALF = 0.5          # C: placed edges at this fraction of the local peak
_CATCH_SMOOTH_RISES = 3.0  # C: boxcar of this many filter rise times

GATES: Mapping[str, float] = MappingProxyType({
    "z_min": _Z_MIN, "size_sigma": _SIZE_SIGMA, "size_frac": _SIZE_FRAC,
    "outer_z": _OUTER_Z, "outer_floor_sigma": _OUTER_FLOOR_SIGMA,
    "guard_rises": _GUARD_RISES, "outer_len_min": _OUTER_LEN_MIN, "outer_len_rises": _OUTER_LEN_RISES,
    "flank_min": _FLANK_MIN, "local_min": _LOCAL_MIN, "local_rises": _LOCAL_RISES,
    "window_px": _WINDOW_PX, "window_rises": _WINDOW_RISES, "span_cap": _SPAN_CAP,
    "max_cuts": _MAX_CUTS, "max_resplit": _MAX_RESPLIT,
    "catch_trigger_sigma": _CATCH_TRIGGER_SIGMA, "catch_foot_sigma": _CATCH_FOOT_SIGMA,
    "catch_half": _CATCH_HALF, "catch_smooth_rises": _CATCH_SMOOTH_RISES,
})

#: Nothing is sharper than two samples once sampled: the floor under every rise.
RISE_FLOOR_SAMPLES = 2.0
#: The interior median is read from at most this many samples (a strided subset beyond).
_MEDIAN_CAP = 4096

# Refusals and per-edge reasons. Plain sentences, because the GUI shows them as they stand.
REASON_VIEW = "The event runs past the view: zoom out"
REASON_NO_EVENT = "No event under the pointer"
REASON_ON_EVENT = "The pointer is on an event already in the table"
REASON_OUTSIDE = "The pointer is outside the visible trace"
REASON_NO_STEP = "no step of the right sign in the window"
REASON_OUTER = "the outer side is not at baseline"
REASON_BORDER = "the step sits at the edge of the search window"
REASON_Z = "the step is not significant against the noise"
REASON_SIZE = "the step is too small"
REASON_SHORT = "span too short to fit"
REASON_WINDOW = "no room to search: the neighbours leave no window"
REASON_NONFINITE = "non-finite samples in the window"
REASON_NOT_REQUESTED = "not requested"
REASON_NO_BASELINE = "no open-pore samples to take a baseline from"
REASON_CROSS = "the fitted edges cross"


# ================================================================================================
# Results
# ================================================================================================
@dataclass(frozen=True)
class EdgeResult:
    """One edge. ``position`` is an absolute sample index into the signal (half-open for an
    end, like the repo's events). When ``fitted`` is False it is the PLACED position,
    unchanged, and ``reason`` says why; ``z`` and ``step`` then carry whatever was measured
    before the gate that refused (0.0 if nothing was)."""

    position: int
    fitted: bool
    z: float = 0.0
    step: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class EdgeFit:
    """Both edges, and the parameters that made them (``params``; see :meth:`to_json`)."""

    start: EdgeResult
    end: EdgeResult
    params: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Both edges fitted: the row-level all-or-nothing the design asks for."""
        return bool(self.start.fitted and self.end.fitted)

    def to_json(self) -> str:
        """A compact, deterministic JSON string for the ``fit_json`` column."""
        body: dict[str, Any] = dict(self.params)
        body["edges"] = {
            "start": _edge_dict(self.start),
            "end": _edge_dict(self.end),
        }
        return json.dumps(_clean(body), separators=(",", ":"), sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class Catch:
    """What ``C`` found. ``ok`` False means a refusal and ``reason`` is the sentence to show;
    the other fields are then None / NaN. ``start`` and ``end`` are the PLACED half-peak edges
    (``end`` half-open), ``foot_start``/``foot_end`` the 2-sigma extent, ``peak_dev`` the local
    peak of the smoothed dev in signal units, and ``fit`` the two-edge fit (whose edges may
    individually be unfitted; an event too short to fit still returns ``ok=True``)."""

    ok: bool
    reason: str = ""
    start: int | None = None
    end: int | None = None
    foot_start: int | None = None
    foot_end: int | None = None
    peak_dev: float = float("nan")
    fit: EdgeFit | None = None


def _edge_dict(e: EdgeResult) -> dict[str, Any]:
    return {"pos": e.position, "fitted": e.fitted, "z": e.z, "step": e.step, "reason": e.reason}


def _clean(obj: Any) -> Any:
    """JSON-safe, compact: numpy scalars to Python, floats to 6 significant digits, NaN and
    infinities to null."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return float(f"{f:.6g}") if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


# ================================================================================================
# Reading dev
# ================================================================================================
class _Dev:
    """``dev = polarity * (signal - baseline) - ref`` over absolute sample ranges, always as a
    fresh float64 copy so no caller can write through to the user's array. ``ref`` is the
    scalar zero level used when no baseline track was supplied (see ``_estimate_ref``)."""

    __slots__ = ("x", "track", "scalar", "pol", "ref", "n", "baseline_kind")

    def __init__(self, x: np.ndarray, track: np.ndarray | None, scalar: float | None, pol: int) -> None:
        self.x = x
        self.track = track
        self.scalar = scalar
        self.pol = pol
        self.ref = 0.0
        self.n = int(x.size)
        # "track": a per-sample baseline; "scalar": one number; "flank": none was given and the
        # zero level comes from the samples beyond the windows (``ref``)
        self.baseline_kind = "track" if track is not None else ("scalar" if scalar is not None else "flank")

    def block(self, a: int, b: int, step: int = 1) -> np.ndarray:
        out = np.array(self.x[a:b:step], dtype=np.float64)
        if self.track is not None:
            out -= self.track[a:b:step]
        elif self.scalar:
            out -= self.scalar
        if self.pol < 0:
            np.negative(out, out=out)
        if self.ref:
            out -= self.ref
        return out


def _make_dev(signal: Any, baseline: Any, polarity: int) -> tuple[_Dev | None, str]:
    """(dev reader, "") or (None, why not)."""
    try:
        x = np.asarray(signal)
    except Exception:           # pragma: no cover - exotic input
        return None, "the signal is not an array"
    if x.ndim != 1 or x.size == 0 or x.dtype.kind not in "fiu":
        return None, "the signal must be a non-empty 1-D numeric array"
    track = None
    scalar = None
    if baseline is not None:
        try:
            nd = np.ndim(baseline)
        except Exception:       # pragma: no cover
            return None, "the baseline is not numeric"
        if nd == 0:
            try:
                scalar = float(baseline)
            except (TypeError, ValueError):
                return None, "the baseline is not numeric"
            if not math.isfinite(scalar):
                return None, "the baseline is not finite"
        else:
            track = np.asarray(baseline)
            if track.ndim != 1 or track.size != x.size or track.dtype.kind not in "fiu":
                return None, "the baseline track must match the signal's length"
    return _Dev(x, track, scalar, polarity), ""


# ================================================================================================
# Numeric kernels (all on a local float64 block; indices are block-local)
# ================================================================================================
def _prefix(d: np.ndarray) -> np.ndarray:
    """Prefix sums with a leading zero: ``P[y] - P[x]`` is ``d[x:y].sum()``. One cumsum serves
    every split and every local mean of an edge."""
    P = np.empty(d.size + 1)
    P[0] = 0.0
    np.cumsum(d, out=P[1:])
    return P


def _mean(P: np.ndarray, x: int, y: int) -> float:
    """Mean of ``d[x:y]`` from its prefix sums (``y > x``)."""
    return float(P[y] - P[x]) / (y - x)


def _median(a: np.ndarray) -> float:
    """Median of a finite, non-empty float array without ``np.median``'s per-call overhead."""
    n = a.size
    h = n >> 1
    if n & 1:
        return float(np.partition(a, h)[h])
    p = np.partition(a, (h - 1, h))
    return 0.5 * (float(p[h - 1]) + float(p[h]))


_COUNT = np.arange(1 << 14, dtype=np.float64)    # 0, 1, 2, ...: split counts without a fresh arange


def _directed_split(P: np.ndarray, a: int, b: int, edge: str, min_len: int) -> int | None:
    """``cusum._best_split`` over the window [a, b) restricted to splits whose step has the
    right sign: the mean after the split above the mean before it for a start, below it for an
    end. The gain is the split's residual-sum-of-squares drop, so with the sign filter off
    this is ``_best_split``'s argmin exactly. ``P`` is :func:`_prefix` of the block."""
    n = b - a
    if n < 2 * min_len or a < 0:
        return None
    hi = n - min_len + 1
    m = _COUNT[min_len:hi] if hi <= _COUNT.size else np.arange(min_len, hi, dtype=np.float64)
    left = P[a + min_len:a + hi]                       # P at each candidate split
    ml = (left - P[a]) / m
    nr = n - m
    mr = (P[b] - left) / nr
    diff = mr - ml
    gain = (m * nr / n) * diff * diff
    ok = diff > 0.0 if edge == "start" else diff < 0.0
    gain = np.where(ok, gain, -1.0)
    k = int(gain.argmax())
    if gain[k] < 0.0:
        return None
    return a + min_len + k


def _window_half_width(span: int, px: float, rise: float, min_len: int) -> int:
    """W = max(12 px of the current zoom in samples, 3 rises), capped at 0.45 span; the cap wins."""
    w = max(_WINDOW_PX * px, _WINDOW_RISES * rise)
    w = min(w, _SPAN_CAP * span)
    return max(int(w), min_len)


def _boxcar(d: np.ndarray, a: int, b: int, k: int) -> np.ndarray:
    """Centred moving average of ``d`` over the block range [a, b) with window ``k``, reading
    up to ``k`` samples outside it, clipped at the array ends."""
    k = max(1, int(k))
    h = k // 2
    aa, bb = max(0, a - h), min(d.size, b + k)
    seg = d[aa:bb]
    cs = np.concatenate(([0.0], np.cumsum(seg)))
    i = np.arange(a, b) - aa
    left = np.clip(i - h, 0, seg.size)
    right = np.clip(i - h + k, 0, seg.size)
    return (cs[right] - cs[left]) / np.maximum(right - left, 1)


def _edge_rise_estimate(d: np.ndarray, m: int, edge: str, inner_level: float, outer_level: float,
                        rise0: float, region: tuple[int, int]) -> float | None:
    """10-90% width of the edge around the split ``m``, on a lightly smoothed dev: the nearest
    10% crossing before (start) or after (end) the split, and the nearest 90% crossing on the
    other side. None when either crossing is not in ``region``."""
    a, b = region
    if b - a < 4:
        return None
    k = max(1, int(round(rise0)))
    sm = _boxcar(d, a, b, k)
    mi = m - a
    if not 0 < mi < sm.size:
        return None
    span = inner_level - outer_level
    l10, l90 = outer_level + 0.1 * span, outer_level + 0.9 * span
    if edge == "start":
        lo_i = np.flatnonzero(sm[:mi] <= l10)
        hi_i = np.flatnonzero(sm[mi:] >= l90)
        if lo_i.size == 0 or hi_i.size == 0:
            return None
        return float((mi + hi_i[0]) - lo_i[-1])
    lo_i = np.flatnonzero(sm[mi:] <= l10)
    hi_i = np.flatnonzero(sm[:mi] >= l90)
    if lo_i.size == 0 or hi_i.size == 0:
        return None
    return float((mi + lo_i[0]) - hi_i[-1])


# ================================================================================================
# One edge
# ================================================================================================
@dataclass(frozen=True)
class _EdgeGeom:
    """Everything one edge's search needs, in BLOCK-LOCAL indices (``off`` is the absolute
    index of ``d[0]``). ``outer_lim``/``inner_lim`` are the neighbour bound on the outside and
    the other (placed) edge on the inside."""

    edge: str
    c: int
    inner_lim: int
    outer_lim: int
    W: int
    Wo0: int
    rise: float


def _fit_edge(d: np.ndarray, g: _EdgeGeom, *, sigma: float, tau: float, interior_med: float,
              min_len: int) -> tuple[int, bool, str, float, float]:
    """The six rules on one edge. Returns (block-local position, fitted, reason, z, step)."""
    edge, c, other, olim0 = g.edge, g.c, g.inner_lim, g.outer_lim
    start = edge == "start"
    opp = "end" if start else "start"
    W, rise = g.W, g.rise
    sq_tau = math.sqrt(max(tau, 1.0))
    L = int(max(_LOCAL_MIN, _LOCAL_RISES * rise))
    gd = int(round(_GUARD_RISES * rise))
    Lout = int(max(_OUTER_LEN_MIN, _OUTER_LEN_RISES * rise))
    fc = int(max(_FLANK_MIN, W))
    bt = max(min_len + 1, rise)
    size_gate = max(_SIZE_SIGMA * sigma, _SIZE_FRAC * interior_med)
    z_min = _Z_MIN
    size_cut = _SIZE_SIGMA * sigma
    outer_z = _OUTER_Z * sigma * math.sqrt(max(tau, 1.0))
    outer_floor = _OUTER_FLOOR_SIGMA * sigma
    nmax = d.size
    P = _prefix(d)
    grown = 0
    while True:
        Wo = g.Wo0 * (1 + grown)
        if start:
            a, b = max(olim0, c - Wo, 0), min(c + W, other, nmax)
        else:
            a, b = max(c - W, other, 0), min(olim0, c + Wo, nmax)
        if b - a < 2 * min_len:
            return c, False, REASON_WINDOW, 0.0, 0.0
        m = _directed_split(P, a, b, edge, min_len)
        olim = olim0
        clipped = False
        # rule 2: cut at an opposite step, up to three times
        for _ in range(_MAX_CUTS):
            if m is None:
                u = _directed_split(P, a, b, opp, min_len)
            elif start:
                u = _directed_split(P, a, m, opp, min_len)
            else:
                u = _directed_split(P, m, b, opp, min_len)
            if u is None:
                break
            # the opposite step is judged on the samples between it and the split (and beyond it,
            # up to the window's end): a high side and a low side of at most L samples
            if m is None:
                x1, y1, x2, y2 = u, min(b, u + L), max(a, u - L), u
            elif start:
                x1, y1, x2, y2 = max(a, u - L), u, u, min(m, u + L)
            else:
                x1, y1, x2, y2 = u, min(b, u + L), max(m, u - L), u
            n1, n2 = y1 - x1, y2 - x2
            if n1 < min_len or n2 < min_len:
                break
            gap = abs(_mean(P, x1, y1) - _mean(P, x2, y2))
            if gap < size_cut or gap / (sigma * math.sqrt(1.0 / n1 + 1.0 / n2)) / sq_tau < z_min:
                break
            if start:
                a = u
            else:
                b = u
            clipped = True
            olim = u
            m = _directed_split(P, a, b, edge, min_len)
        if m is None:
            return c, False, REASON_NO_STEP, 0.0, 0.0
        # rule 3: the outer side must be at baseline, else split it again, outward
        b_off = 0.0
        if not clipped:
            if start:
                x, y = max(olim0, a - fc, 0), a
            else:
                x, y = b, min(olim0, b + fc, nmax)
            if y - x >= min_len:
                b_off = _median(d[x:y])
        outer_ok = False
        for _ in range(_MAX_RESPLIT):
            if start:
                y = max(m - gd, 0)
                x = max(olim, y - Lout, 0)
            else:
                x = min(m + gd, nmax)
                y = min(olim, x + Lout, nmax)
            n_out = y - x
            if n_out >= min_len and abs(_mean(P, x, y) - b_off) < max(outer_z / math.sqrt(n_out), outer_floor):
                outer_ok = True
                break
            m2 = _directed_split(P, a, m, edge, min_len) if start else _directed_split(P, m, b, edge, min_len)
            if m2 is None:
                break
            m = m2
        if not outer_ok:
            if grown == 0:
                grown = 1
                continue
            return c, False, REASON_OUTER, 0.0, 0.0
        # rule 6: a split hugging the OUTER window edge is retried once with the window grown
        near = ((m - a) < bt) if start else ((b - m) < bt)
        near = near and ((a > olim0) if start else (b < olim0))
        if near:
            if grown == 0:
                grown = 1
                continue
            return c, False, REASON_BORDER, 0.0, 0.0
        # rules 4 and 5: local z, then size
        if start:
            xo, yo, xi, yi = max(olim, m - L, 0), m, m, min(other, m + L, nmax)
        else:
            xi, yi, xo, yo = max(other, m - L, 0), m, m, min(olim, m + L, nmax)
        n_in, n_out = yi - xi, yo - xo
        if n_in < 1 or n_out < 1:
            return c, False, REASON_WINDOW, 0.0, 0.0
        step = _mean(P, xi, yi) - _mean(P, xo, yo)
        z = abs(step) / (sigma * math.sqrt(1.0 / n_in + 1.0 / n_out)) / sq_tau
        if z < z_min:
            return c, False, REASON_Z, z, step
        if step < size_gate:
            return c, False, REASON_SIZE, z, step
        return m, True, "", z, step


# ================================================================================================
# fit_edges
# ================================================================================================
def fit_edges(
    signal: np.ndarray,
    fs: float,
    start: int,
    end: int,
    *,
    polarity: int,
    noise_rms: float,
    tau: float,
    lo: int,
    hi: int,
    baseline: np.ndarray | float | None = None,
    edges: Sequence[str] = ("start", "end"),
    method: str = "step",
    rule: Any = None,
    px_samples: float | None = None,
    rise_s: float | None = None,
    min_len: int = 3,
    filter_rise_s: float | None = None,
) -> EdgeFit:
    """Move a placed ``start`` and ``end`` onto the steps they sit on.

    Args:
        signal: the array the add door measures on (``detection_signal``; raw for a Manual
            table), absolute indexing. Only the windows around the edges are read, so a
            15 M-sample array costs nothing extra. Never modified.
        fs: sample rate, Hz.
        start, end: the placed bounds, absolute sample indices, ``end`` half-open.
        polarity: +1 or -1, the target table's, such that ``polarity * (x - baseline)`` is
            positive inside an event.
        noise_rms: the table's ``noise_rms_pA`` (sigma of ``signal - baseline``).
        tau: the open-pore autocorrelation time in samples (:func:`estimate_open_pore_tau`),
            floored at the filter's own when the provenance has a low-pass. < 1 reads as 1.
        lo, hi: the nearest neighbouring events' bounds (``lo`` is the previous event's end,
            ``hi`` the next one's start), or 0 / ``len(signal)``. No window crosses them.
        baseline: the contract's per-sample baseline track (same length as ``signal``), a
            scalar, or None. None takes one scalar zero level from the medians of samples
            beyond the windows; it is weaker than the track and meant for stand-alone use.
        edges: which edges to fit; the other comes back unfitted with reason "not requested".
        method: "step" is the only one.
        rule: reserved for the phase-2 table rule; anything but None raises
            ``NotImplementedError``.
        px_samples: samples per screen pixel at the current zoom (None: 1). Sets the window
            ``W = max(12 px, 3 rise)``.
        rise_s: the target table's median ``rise_s`` in seconds. None: a first-pass 10-90%
            estimate around a first split, per edge, floored at the filter rise time.
        min_len: the shortest segment either side of a split (3, ``cusum``'s default).
        filter_rise_s: the filter's 10-90% rise time in seconds, the floor under the rise
            (and under the first-pass estimate). Not in the design's signature; optional.

    Never raises on bad DATA (non-finite samples in a window, a span too short to fit, zero
    noise, a window the neighbours squeeze to nothing): the affected edges come back unfitted
    at their placed positions with a ``reason``. It raises only for API misuse (``rule``,
    an unknown ``method`` or ``edges`` name).
    """
    if rule is not None:
        raise NotImplementedError("the table rule is phase 2; only method='step' exists")
    return _fit_edges_impl(signal, fs, start, end, polarity=polarity, noise_rms=noise_rms,
                           tau=tau, lo=lo, hi=hi, baseline=baseline, edges=edges, method=method,
                           px_samples=px_samples, rise_s=rise_s, min_len=min_len,
                           filter_rise_s=filter_rise_s, foot=None, extra=None)


def _num(v: Any, default: float | None = None) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _polarity(v: Any) -> int | None:
    """+1 or -1, or None."""
    f = _num(v)
    return int(f) if f is not None and f in (1.0, -1.0) else None


def _unfitted(start: int, end: int, params: dict, why: str, only: Sequence[str] | None = None) -> EdgeFit:
    keep = ("start", "end") if only is None else tuple(only)
    return EdgeFit(
        EdgeResult(start, False, reason=why if "start" in keep else REASON_NOT_REQUESTED),
        EdgeResult(end, False, reason=why if "end" in keep else REASON_NOT_REQUESTED),
        params,
    )


def _fit_edges_impl(signal, fs, start, end, *, polarity, noise_rms, tau, lo, hi, baseline, edges,
                    method, px_samples, rise_s, min_len, filter_rise_s, foot, extra) -> EdgeFit:
    """``fit_edges`` plus two internal inputs: ``foot`` (how far past the placed start and end
    the outer window must reach, in samples, from ``catch_event``) and ``extra`` (a dict
    merged into ``params``)."""
    if method != METHOD:
        raise ValueError(f"unknown method {method!r}; only {METHOD!r} exists")
    want = tuple(edges)
    if not want or any(e not in ("start", "end") for e in want):
        raise ValueError(f"edges must be a non-empty subset of ('start', 'end'), got {edges!r}")
    s0, e0 = _int(start), _int(end)
    params: dict[str, Any] = {"method": METHOD, "version": VERSION, "gates": dict(GATES),
                              "placed": [s0, e0], "edges_requested": list(want)}
    if extra:
        params.update(extra)
    fs_ = _num(fs)
    sigma = _num(noise_rms)
    tau_ = _num(tau, 1.0)
    params.update(fs=fs_, noise=sigma, tau=max(tau_, 1.0))
    if s0 is None or e0 is None:
        return _unfitted(s0 or 0, e0 or 0, params, "the placed bounds are not integers", want)
    pol = _polarity(polarity)
    if pol is None:
        return _unfitted(s0, e0, params, "polarity must be +1 or -1", want)
    if fs_ is None or fs_ <= 0:
        return _unfitted(s0, e0, params, "fs must be positive and finite", want)
    if sigma is None or sigma <= 0:
        return _unfitted(s0, e0, params, "noise_rms must be positive and finite", want)
    dev, why = _make_dev(signal, baseline, pol)
    if dev is None:
        return _unfitted(s0, e0, params, why, want)
    n = dev.n
    min_len = max(2, _int(min_len) or 3)
    params["min_len"] = min_len
    lo_ = min(max(_int(lo, 0), 0), n)
    hi_ = max(min(_int(hi, n), n), 0)
    params.update(lo=lo_, hi=hi_)
    span = e0 - s0
    if span < 2 * min_len + 2:
        return _unfitted(s0, e0, params, REASON_SHORT, want)
    if s0 < 0 or e0 > n:
        return _unfitted(s0, e0, params, "the span is outside the signal", want)
    px = _num(px_samples, 1.0)
    px = 1.0 if px is None or px < 1.0 else px
    params["px_samples"] = px

    # the zero level when no baseline track was given: medians of samples beyond the windows
    if dev.baseline_kind == "flank":
        ref, why = _estimate_ref(dev, s0, e0, lo_, hi_, px, min_len)
        if ref is None:
            return _unfitted(s0, e0, params, why, want)
        dev.ref = ref
    params["baseline"] = dev.baseline_kind

    # the interior median dev (for the size gate), from at most _MEDIAN_CAP samples; a
    # non-finite sample inside the event is left out here (the edges' own windows refuse theirs)
    seg = dev.block(s0, e0, max(1, span // _MEDIAN_CAP))
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return _unfitted(s0, e0, params, REASON_NONFINITE, want)
    interior_med = _median(seg)

    # the rise: the table's, or a per-edge first-pass estimate; either floored
    floor = max(RISE_FLOOR_SAMPLES, (_num(filter_rise_s, 0.0) or 0.0) * fs_)
    table_rise = _num(rise_s)
    table_rise = table_rise * fs_ if table_rise is not None and table_rise > 0 else None
    cap = _SPAN_CAP * span
    params["rise_floor_samples"] = floor
    results: dict[str, EdgeResult] = {}
    rises: dict[str, Any] = {}
    widths: dict[str, int] = {}
    for edge in ("start", "end"):
        c = s0 if edge == "start" else e0
        if edge not in want:
            results[edge] = EdgeResult(c, False, reason=REASON_NOT_REQUESTED)
            continue
        reach = 0 if foot is None else int(foot[0 if edge == "start" else 1])
        # -- the rise for this edge
        if table_rise is not None:
            r_e, src = max(floor, min(table_rise, cap)), "table"
        else:
            r_e, src = _first_pass_rise(dev, edge, s0, e0, lo_, hi_, span, px, floor, reach,
                                        sigma, tau_, min_len)
        rises[edge] = {"samples": r_e, "s": r_e / fs_, "source": src}
        W = _window_half_width(span, px, r_e, min_len)
        # the outer half-window reaches the foot plus three rises when C supplied one
        Wo0 = W if foot is None else max(W, reach + int(round(_WINDOW_RISES * r_e)))
        widths[edge] = W
        # -- the block: every sample this edge's rules may read
        L = int(max(_LOCAL_MIN, _LOCAL_RISES * r_e))
        gd = int(round(_GUARD_RISES * r_e))
        Lout = int(max(_OUTER_LEN_MIN, _OUTER_LEN_RISES * r_e))
        fc = int(max(_FLANK_MIN, W))
        need_out = 2 * Wo0 + max(fc, L, gd + Lout) + 2
        need_in = W + L + 2
        if edge == "start":
            b0, b1 = max(lo_, c - need_out, 0), min(e0, c + need_in, n)
        else:
            b0, b1 = max(s0, c - need_in, 0), min(hi_, c + need_out, n)
        if b1 - b0 < 2 * min_len:
            results[edge] = EdgeResult(c, False, reason=REASON_WINDOW)
            continue
        d = dev.block(b0, b1)
        if not np.isfinite(d).all():
            results[edge] = EdgeResult(c, False, reason=REASON_NONFINITE)
            continue
        geom = _EdgeGeom(
            edge=edge, c=c - b0,
            inner_lim=(e0 - b0) if edge == "start" else (s0 - b0),
            outer_lim=(lo_ - b0) if edge == "start" else (hi_ - b0),
            W=W, Wo0=Wo0, rise=r_e)
        m, ok, reason, z, step = _fit_edge(d, geom, sigma=sigma, tau=tau_, interior_med=interior_med,
                                           min_len=min_len)
        results[edge] = EdgeResult(int(m + b0) if ok else c, bool(ok), float(z), float(step), reason)
    s_res, e_res = results["start"], results["end"]
    if s_res.fitted and e_res.fitted and s_res.position >= e_res.position:
        # unreachable at the default min_len (W is capped at 0.45 of the span); a large min_len can
        results["start"] = EdgeResult(s0, False, s_res.z, s_res.step, REASON_CROSS)
        results["end"] = EdgeResult(e0, False, e_res.z, e_res.step, REASON_CROSS)
    params["W"] = widths
    params["rise"] = rises
    params["interior_median_dev"] = interior_med
    if table_rise is not None:
        params["rise_s"] = table_rise / fs_
        params["rise_source"] = "table"
    else:
        params["rise_s"] = None
        params["rise_source"] = "estimate"
    return EdgeFit(results["start"], results["end"], params)


def _int(v: Any, default: int | None = None) -> int | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return int(math.floor(f)) if f != int(f) else int(f)


def _estimate_ref(dev: _Dev, s0: int, e0: int, lo: int, hi: int, px: float, min_len: int) -> tuple[float | None, str]:
    """The scalar open-pore level (in ``polarity * signal`` units) when no baseline track was
    given: the median of up to 256 samples either side of the event, beyond the nominal
    window and inside the neighbours."""
    span = e0 - s0
    gap = int(max(_WINDOW_PX * px, _WINDOW_RISES * RISE_FLOOR_SAMPLES))
    gap = min(gap, int(_SPAN_CAP * span))
    reach = 256
    parts = []
    a0, a1 = max(lo, s0 - gap - reach), max(lo, s0 - gap)
    if a1 - a0 >= min_len:
        parts.append(dev.block(a0, a1))
    b0, b1 = min(hi, e0 + gap), min(hi, e0 + gap + reach)
    if b1 - b0 >= min_len:
        parts.append(dev.block(b0, b1))
    if not parts:
        # no room beyond the nominal window: whatever open pore the neighbours leave
        a0, a1 = lo, s0
        b0, b1 = e0, hi
        if a1 - a0 >= min_len:
            parts.append(dev.block(a0, a1))
        if b1 - b0 >= min_len:
            parts.append(dev.block(b0, b1))
    if not parts:
        return None, REASON_NO_BASELINE
    cat = np.concatenate(parts)
    if not np.isfinite(cat).all():
        return None, REASON_NONFINITE
    return _median(cat), ""


def _first_pass_rise(dev: _Dev, edge: str, s0: int, e0: int, lo: int, hi: int, span: int, px: float,
                     floor: float, reach: int, sigma: float, tau: float, min_len: int) -> tuple[float, str]:
    """The edge's 10-90% rise, estimated around a first split found with the floor rise; the
    floor when nothing can be measured. Returns (samples, source)."""
    n = dev.n
    c = s0 if edge == "start" else e0
    other = e0 if edge == "start" else s0
    W0 = _window_half_width(span, px, floor, min_len)
    Wo = max(W0, reach)
    L = int(max(_LOCAL_MIN, _LOCAL_RISES * floor))
    cap = int(max(64, _SPAN_CAP * span))
    r_est = int(min(cap, max(2048, 8 * W0)))
    k = max(1, int(round(floor)))
    if edge == "start":
        b0, b1 = max(lo, c - max(Wo + L, r_est) - k - 2, 0), min(other, c + max(W0 + L, r_est) + k + 2, n)
    else:
        b0, b1 = max(other, c - max(W0 + L, r_est) - k - 2, 0), min(hi, c + max(Wo + L, r_est) + k + 2, n)
    if b1 - b0 < 2 * min_len + 2:
        return floor, "floor"
    d = dev.block(b0, b1)
    if not np.isfinite(d).all():
        return floor, "floor"
    cl, ot = c - b0, other - b0
    if edge == "start":
        a, b = max(lo - b0, cl - Wo, 0), min(cl + W0, ot, d.size)
    else:
        a, b = max(cl - W0, ot, 0), min(hi - b0, cl + Wo, d.size)
    m = _directed_split(_prefix(d), a, b, edge, min_len)
    if m is None:
        return floor, "floor"
    if edge == "start":
        inn = d[m:min(ot, m + L)]
        out = d[max(lo - b0, m - L, 0):m]
    else:
        inn = d[max(ot, m - L, 0):m]
        out = d[m:min(hi - b0, m + L, d.size)]
    if inn.size < min_len or out.size < min_len:
        return floor, "floor"
    ra = max(lo - b0 if edge == "start" else ot, m - r_est, 0)
    rb = min(ot if edge == "start" else hi - b0, m + r_est, d.size)
    est = _edge_rise_estimate(d, m, edge, _median(inn), _median(out), floor, (ra, rb))
    if est is None:
        return floor, "floor"
    r = max(floor, min(est, _SPAN_CAP * span))
    return r, ("estimate" if r > floor else "floor")


# ================================================================================================
# catch_event
# ================================================================================================
def _refusal(reason: str) -> Catch:
    return Catch(ok=False, reason=reason)


def catch_event(
    signal: np.ndarray,
    fs: float,
    pointer: float,
    *,
    polarity: int,
    noise_rms: float,
    baseline: np.ndarray | float | None,
    lo: int,
    hi: int,
    view_lo: int,
    view_hi: int,
    px_samples: float,
    tau: float,
    rise_s: float | None = None,
    min_len: int = 3,
    filter_rise_s: float | None = None,
) -> Catch:
    """``C``: find the event under ``pointer`` and fit both of its edges (design 3.2, steps 2-6).

    ``dev = polarity * (signal - baseline)`` is smoothed by a boxcar of three FILTER rise
    times (``filter_rise_s``; without it, six samples), outward from the pointer in blocks,
    never over the whole signal. The trigger is the largest smoothed dev over the pointer's
    pixel plus or minus one pixel; it must reach 4 * ``noise_rms`` or the answer is "No event
    under the pointer". The growth then runs from the local peak (the largest smoothed dev
    within three pixels of the pointer, on the trigger's own half-height run) to where dev
    falls below half that peak (the PLACED edges) and to where it falls below 2 sigma (the
    FOOT); each edge's outer fit window is widened to its foot plus three rise times, because
    a sub-state shallower than half the peak lies outside the half-peak edge.

    ``lo``/``hi`` are the nearest neighbouring events' bounds (0 / ``len(signal)`` when
    there are none): growth stops at them. ``view_lo``/``view_hi`` are the visible sample
    range: growth that reaches one of them refuses, "The event runs past the view: zoom out".
    A neighbour or the recording's end that is as close as the view edge is not a refusal.

    The returned :class:`Catch` has ``ok=True`` whenever an event was found and bounded, even
    when one or both edges then fail their fit (``fit.start.fitted``/``fit.end.fitted``): the
    caller shows an unfitted edge as "not fitted" and keeps the good one as a preview.
    """
    pol = _polarity(polarity)
    if pol is None:
        return _refusal("polarity must be +1 or -1")
    fs_ = _num(fs)
    sigma = _num(noise_rms)
    p_f = _num(pointer)
    if fs_ is None or fs_ <= 0 or sigma is None or sigma <= 0 or p_f is None:
        return _refusal("bad input: fs, noise_rms and the pointer must be finite and positive")
    dev, why = _make_dev(signal, baseline, pol)
    if dev is None:
        return _refusal(why)
    n = dev.n
    p = int(math.floor(p_f))
    px = _num(px_samples, 1.0)
    px = 1 if px is None or px < 1.0 else int(round(px))
    min_len = max(2, _int(min_len) or 3)
    lo_ = min(max(_int(lo, 0), 0), n)
    hi_ = max(min(_int(hi, n), n), 0)
    v0, v1 = _int(view_lo, 0), _int(view_hi, n)
    if v0 is None or v1 is None or v1 <= v0:
        return _refusal("bad input: the view must be a non-empty sample range")
    g0, g1 = max(0, lo_, v0), min(n, hi_, v1)
    if not 0 <= p < n or p < v0 or p >= v1:
        return _refusal(REASON_OUTSIDE)
    if p < lo_ or p >= hi_:
        return _refusal(REASON_ON_EVENT)
    if g1 <= g0:
        return _refusal(REASON_NO_EVENT)

    # smoothing length: three filter rise times, at least three samples; with no filter rise
    # given, three times the two-sample floor
    fr = (_num(filter_rise_s, 0.0) or 0.0) * fs_
    k = max(3, int(round(_CATCH_SMOOTH_RISES * (fr if fr > 0 else RISE_FLOOR_SAMPLES))))
    h = k // 2

    if dev.baseline_kind == "flank":
        # no track: one scalar zero level from the median of the domain (strided, bounded)
        step = max(1, (g1 - g0) // 20000)
        zero = dev.block(g0, g1, step)
        zero = zero[np.isfinite(zero)]
        dev.ref = _median(zero) if zero.size else 0.0

    def sm(a: int, b: int) -> np.ndarray:
        """Smoothed dev over [a, b) within [g0, g1): boxcar of k, windows clipped at the
        domain, non-finite samples read as zero dev (the fit then refuses them)."""
        r0, r1 = max(g0, a - h), min(g1, b - h + k)
        seg = dev.block(r0, r1)
        np.nan_to_num(seg, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cs = np.concatenate(([0.0], np.cumsum(seg)))
        i = np.arange(a, b)
        left = np.clip(i - h, g0, g1) - r0
        right = np.clip(i - h + k, g0, g1) - r0
        return (cs[right] - cs[left]) / np.maximum(right - left, 1)

    # the trigger: pixel max of the smoothed dev over the pointer's pixel +-1 px, i.e. every
    # sample within one pixel of the pointer (the whole of the pointer's pixel, plus up to a
    # pixel of margin)
    r_trig = px
    r_peak = 3 * px
    a0, b0 = max(g0, p - r_peak - 1), min(g1, p + r_peak + 2)
    s0 = sm(a0, b0)
    t0, t1 = max(a0, p - r_trig), min(b0, p + r_trig + 1)
    if t1 <= t0:
        return _refusal(REASON_NO_EVENT)
    pi = t0 - a0 + int(np.argmax(s0[t0 - a0:t1 - a0]))
    v_trig = float(s0[pi])
    trig = _CATCH_TRIGGER_SIGMA * sigma
    if not v_trig >= trig:
        return _refusal(REASON_NO_EVENT)
    # the local peak: largest smoothed dev within 3 px of the pointer, on the trigger's own
    # run above max(2 sigma, half the trigger value), so a neighbour behind a dip is not it
    run_thr = max(_CATCH_FOOT_SIGMA * sigma, _CATCH_HALF * v_trig)
    below = np.flatnonzero(s0 <= run_thr)
    left_b = below[below < pi]
    right_b = below[below > pi]
    r_lo = int(left_b[-1]) + 1 if left_b.size else 0
    r_hi = int(right_b[0]) if right_b.size else s0.size
    q0, q1 = max(r_lo, p - r_peak - a0), min(r_hi, p + r_peak + 1 - a0)
    if q1 <= q0:
        q0, q1 = pi, pi + 1
    seed = q0 + int(np.argmax(s0[q0:q1]))
    peak = float(s0[seed])
    seed += a0
    thr_half = max(_CATCH_FOOT_SIGMA * sigma, _CATCH_HALF * peak)
    thr_foot = _CATCH_FOOT_SIGMA * sigma

    blk0 = max(512, 8 * px)

    def first_le(a: int, stop: int, thr: float) -> int | None:
        blk = blk0
        while a < stop:
            b = min(stop, a + blk)
            idx = np.flatnonzero(sm(a, b) <= thr)
            if idx.size:
                return a + int(idx[0])
            a, blk = b, blk * 2
        return None

    def last_le(stop: int, floor_: int, thr: float) -> int | None:
        blk = blk0
        b = stop
        while b > floor_:
            a = max(floor_, b - blk)
            idx = np.flatnonzero(sm(a, b) <= thr)
            if idx.size:
                return a + int(idx[-1])
            b, blk = a, blk * 2
        return None

    j = first_le(seed + 1, g1, thr_half)
    end = g1 if j is None else j
    j = last_le(seed, g0, thr_half)
    start = g0 if j is None else j + 1
    hit_left, hit_right = j is None, end == g1
    # a bound is the VIEW only when the view is strictly tighter than the neighbour / the trace
    if hit_left and v0 > max(lo_, 0):
        return _refusal(REASON_VIEW)
    if hit_right and v1 < min(hi_, n):
        return _refusal(REASON_VIEW)
    j = first_le(end, g1, thr_foot) if not hit_right else None
    foot_end = g1 if j is None else j
    j = last_le(start, g0, thr_foot) if not hit_left else None
    foot_start = g0 if j is None else j + 1

    extra = {"catch": {"peak_dev": peak, "foot": [foot_start, foot_end], "smooth_samples": k,
                       "trigger_sigma": _CATCH_TRIGGER_SIGMA, "pointer": p,
                       "view": [v0, v1]}}
    fit = _fit_edges_impl(signal, fs, start, end, polarity=pol, noise_rms=sigma, tau=tau,
                          lo=lo_, hi=hi_, baseline=baseline, edges=("start", "end"),
                          method=METHOD, px_samples=px, rise_s=rise_s, min_len=min_len,
                          filter_rise_s=filter_rise_s,
                          foot=(start - foot_start, foot_end - end), extra=extra)
    return Catch(ok=True, start=start, end=end, foot_start=foot_start, foot_end=foot_end,
                 peak_dev=peak, fit=fit)


# ================================================================================================
# tau
# ================================================================================================
_GAP_CHUNK = 1 << 17      # the longest open-pore stretch measured in one piece (samples)
_GAP_BUDGET = 1 << 22     # most samples measured in all: a median needs a few hundred pieces, not 15 M samples
_GAP_MAX_PIECES = 2048    # ... and most separate pieces


def estimate_open_pore_tau(
    residual: np.ndarray,
    spans: Sequence[tuple[int, int]] | np.ndarray | None,
    *,
    noise_rms: float,
    fs: float,
    L: int = 64,
    filter_autocorr: float | None = None,
) -> float:
    """The open-pore autocorrelation time, in samples, that :func:`fit_edges` divides its z by.

    The median over the open-pore gaps of ``cusum.integrated_autocorr_time(., n_scale=L)`` of
    the residual high-passed at the inference scale (a ``L``-sample moving average
    removed), each gap clear of every span by ``L`` samples: the quantity
    ``cusum.open_pore_autocorr_time(residual, starts, ends, n_scale=L, pad=L)`` documents,
    floored at ``filter_autocorr`` when given (the low-pass's own tau, as CUSUM's caller does).
    ``residual`` is the signal minus the contract's baseline; ``spans`` are ``(start, end)``
    pairs, the rows of the target table or of the recording's selected detector table.

    It does NOT call ``open_pore_autocorr_time``, whose pairing of event bounds is off by one:
    it measures the stretches [0, b0 + pad), [a1 - pad, b1 + pad), ... that CONTAIN the events
    and skips the gaps between them (white noise with 90 boxes of 40 sigma, true tau 1.0: that
    function returns 4.5, the gaps 1.0). That is a defect in ``cusum.py``, reported with the
    fix; this module measures the gaps themselves.

    With no spans it does NOT fall back to the whole residual, which counts the events as open
    pore and inflates tau (the design review measured 90.5 and 57.1 against 11.3 and 4.8 on the
    owner's apo files). It builds pseudo-events instead: every run where
    ``|residual| > 4 * noise_rms``, non-finite samples included, each padded by ``L`` like a
    real row.

    Cost is bounded: at most about 4 M samples in at most 2,048 pieces are measured, so 15 M
    samples take well under a second.

    The high-pass is the definition, and it has a consequence the caller must know: correlation
    SHORTER than ``L`` samples -- the low-pass filter's own -- is removed with the slow drift, so an
    AR(1) process with rho 0.8 reads 1.2 here though its raw integrated time is 8.3. What survives is
    structure around the inference scale (1/f wander, ringing). That is why the design floors this
    number at ``cusum.filter_autocorr_time`` whenever the provenance has a low-pass.

    ``fs`` is accepted for symmetry with the other estimators; the answer is in samples and does not
    depend on it. Returns at least 1.0 (no correction), and exactly ``max(1.0, filter_autocorr)`` for
    input too short or too degenerate to measure. A contract-worker computation, never a key-press one.
    """
    floor = max(1.0, _num(filter_autocorr, 1.0) or 1.0)
    try:
        r = np.asarray(residual)
        if r.ndim != 1 or r.size < 32 or r.dtype.kind not in "fiu":
            return floor
        n = r.size
        L = max(3, int(L))
        sigma = _num(noise_rms)
        r64 = np.asarray(r, dtype=np.float64)
        bad = ~np.isfinite(r64)
        have_bad = bool(bad.any())
        if spans is not None and len(spans) > 0:
            arr = np.asarray(spans, dtype=np.float64).reshape(-1, 2)
            arr = arr[np.isfinite(arr).all(axis=1)]
            starts = np.clip(arr[:, 0], 0, n).astype(np.int64)
            ends = np.clip(arr[:, 1], 0, n).astype(np.int64)
            if have_bad:
                bs, be = _mask_runs(bad)
                starts, ends = np.concatenate((starts, bs)), np.concatenate((ends, be))
        else:
            if sigma is not None and sigma > 0:
                thr = 4.0 * sigma
                mask = r64 > thr
                mask |= r64 < -thr
            else:
                mask = np.zeros(n, dtype=bool)
            if have_bad:
                mask |= bad
            starts, ends = _mask_runs(mask)
        tau = _gap_tau(r64, starts, ends, L)
    except Exception:           # pragma: no cover - an estimator must not take the caller down
        return floor
    tau = _num(tau, 1.0) or 1.0
    return float(max(tau, floor, 1.0))


def _mask_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open [start, end) of every True run in a bool mask."""
    if not mask.any():
        e = np.empty(0, dtype=np.int64)
        return e, e
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2].astype(np.int64), edges[1::2].astype(np.int64)


def _open_gaps(starts: np.ndarray, ends: np.ndarray, n: int, pad: int) -> tuple[np.ndarray, np.ndarray]:
    """The sorted, disjoint stretches [lo, hi) of a length-``n`` trace that are clear of every
    span ``[start, end)`` by ``pad`` samples."""
    s_ = np.asarray(starts, dtype=np.int64)
    e_ = np.asarray(ends, dtype=np.int64)
    live = e_ > s_                                   # an empty or inverted span excludes nothing
    a, b = s_[live] - pad, e_[live] + pad
    if a.size == 0:
        return np.array([0], dtype=np.int64), np.array([n], dtype=np.int64)
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    run_end = np.maximum.accumulate(b)
    first = np.empty(a.size, dtype=bool)
    first[0] = True
    first[1:] = a[1:] > run_end[:-1]                 # a padded span that starts a new cluster
    last = np.empty(a.size, dtype=bool)
    last[-1] = True
    last[:-1] = first[1:]
    lo = np.concatenate(([0], run_end[last]))
    hi = np.concatenate((a[first], [n]))
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    keep = hi > lo
    return lo[keep], hi[keep]


def _piece_tau(seg: np.ndarray, w: int, L: int) -> float | None:
    """tau of one open-pore stretch: high-passed by an ``w``-sample moving average (the
    inference-scale filter of ``cusum.open_pore_autocorr_time``), the ``w // 2`` samples at each
    end dropped because their average is incomplete."""
    from .cusum import integrated_autocorr_time

    h = w // 2
    if seg.size < w + 32 or not np.isfinite(seg).all():
        return None
    cs = np.empty(seg.size + 1)
    cs[0] = 0.0
    np.cumsum(seg, out=cs[1:])
    hp = seg[h:seg.size - h] - (cs[w:] - cs[:-w]) / w
    return float(integrated_autocorr_time(hp, n_scale=L))


def _gap_tau(r: np.ndarray, starts: np.ndarray, ends: np.ndarray, L: int) -> float:
    """Median of the per-stretch taus over the open-pore stretches; the whole trace, in the
    same bounded pieces, when no stretch is long enough (honest degradation, like CUSUM's)."""
    n = r.size
    w = max(3, int(L) | 1)
    need = max(64, 4 * int(L)) + 2 * (w // 2)
    lo, hi = _open_gaps(starts, ends, n, pad=int(L))
    pieces: list[tuple[int, int]] = []
    for a, b in zip(lo.tolist(), hi.tolist()):
        if b - a < need:
            continue
        k = max(1, (b - a) // _GAP_CHUNK)
        cuts = np.linspace(a, b, k + 1).astype(np.int64).tolist()
        pieces.extend(zip(cuts[:-1], cuts[1:]))
    if not pieces:
        k = max(1, n // _GAP_CHUNK)
        cuts = np.linspace(0, n, k + 1).astype(np.int64).tolist()
        pieces = list(zip(cuts[:-1], cuts[1:]))
    mean_len = sum(b - a for a, b in pieces) / len(pieces)
    keep = int(min(len(pieces), _GAP_MAX_PIECES, max(1, _GAP_BUDGET // max(mean_len, 1.0))))
    if keep < len(pieces):
        pieces = [pieces[i] for i in np.unique(np.linspace(0, len(pieces) - 1, keep).astype(np.int64))]
    taus = [t for t in (_piece_tau(r[a:b], w, L) for a, b in pieces) if t is not None]
    return float(np.median(taus)) if taus else 1.0
