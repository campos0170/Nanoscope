"""Tests for the step fit and ``C`` (analysis/edge_fit.py).

SYNTHETIC DATA ONLY. No recording on disk is read, so nothing here can write a calibration
sidecar next to a real file. Every trace is ``truth.synth_trace``'s default noise (white plus
30% 1/f, 6 pA) with erf-edged events added, a baseline estimated by the same
``characterisation_baseline`` door the app uses, and ``noise_rms`` / ``tau`` measured the way
the contract measures them.

The rates asserted are FLOORS taken from the second design review
(``docs/CURSOR_ERGONOMICS_DESIGN.md`` sections 3.2 and 5.2), which measured 100 events per
case; these use fewer (seeded) trials, so each floor sits below the review's number by a
margin that keeps the file stable. ``tests/analysis`` keeps the whole file under about a
minute. The full-scale numbers are in the report that came with this module, not here.

Both polarities are built from ONE trace by negating signal and baseline, so ``dev`` is
bit-for-bit the same and the polarity twin of every case must give the same answer.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from scipy.special import erf

from nanoscope.analysis import edge_fit as ef
from nanoscope.analysis.baseline import characterisation_baseline
from nanoscope.analysis.filters import lowpass, noise_rms
from nanoscope.analysis.truth import synth_trace

SIGMA = 6.0                       # pA: the review's noise
FILTER_RISE_S = 0.034e-3          # a 10 kHz low-pass, the review's "filter rise"
FS_LIST = (50_000.0, 250_000.0)
PX_LIST = (1, 10, 100)


# ================================================================================================
# Synthetic rigs
# ================================================================================================
def erf_event(n: int, s: int, e: int, depth: float, rise: float) -> np.ndarray:
    """An erf-edged box on [s, e) in a length-``n`` slot; ``rise`` is the 10-90% time in samples."""
    t = np.arange(n, dtype=np.float64)
    sd = rise / 2.563
    if sd <= 0:
        return depth * ((t >= s).astype(np.float64) - (t >= e).astype(np.float64))
    k = np.sqrt(2.0) * sd
    return depth * (0.5 * (1 + erf((t - s) / k)) - 0.5 * (1 + erf((t - e) / k)))


@dataclass
class Rig:
    """A synthetic trace as the app would hold it: the signal in the table's own sign, the
    contract's baseline track, the table's ``noise_rms`` and the open-pore ``tau``."""

    fs: float
    polarity: int
    x: np.ndarray
    base: np.ndarray
    noise: float
    tau: float
    slots: list          # [(a, b, pad)] absolute slot bounds: the neighbours' bounds
    truths: list         # per slot, the list of true (start, end) of every event in it
    _twin: "Rig | None" = None

    @property
    def dev(self) -> np.ndarray:
        return self.polarity * (self.x - self.base)

    def twin(self) -> "Rig":
        """The polarity -1 twin: the same dev, written the other way up."""
        if self._twin is None:
            self._twin = Rig(self.fs, -self.polarity, -self.x, -self.base, self.noise, self.tau,
                             self.slots, self.truths)
        return self._twin


Builder = Callable[[int, int], "tuple[np.ndarray, list]"]


def build_rig(fs: float, specs: "list[tuple[int, Builder]]", *, seed: int, pad_s: float = 0.1,
              hardware_hz: float | None = None, drift: Callable[[np.ndarray], np.ndarray] | None = None,
              measure_tau: bool = True, true_baseline: bool = False) -> Rig:
    """One slot per spec: ``pad`` of open pore, the event(s), ``pad`` more, where ``pad`` is
    ``max(pad_s, 2 * length)``. ``hardware_hz`` pushes the noise through a 4-pole low-pass of
    that cutoff (rescaled to the same sigma); ``drift(t)`` adds a slow baseline wander."""
    slots, pos = [], 0
    for length, _b in specs:
        pad = int(max(pad_s * fs, 2 * length))
        slots.append((pos, pos + length + 2 * pad, pad))
        pos += length + 2 * pad
    n = pos
    rec, _ = synth_trace(fs=fs, duration_s=n / fs, n_events=0, noise_pA=SIGMA, seed=seed)
    x = np.asarray(rec.data, dtype=np.float64)[:n]
    if x.size < n:
        x = np.concatenate([x, np.full(n - x.size, x[-1])])
    if hardware_hz:
        y = np.asarray(lowpass(x, fs, hardware_hz, order=4), dtype=np.float64)
        x = -1000.0 + (y - y.mean()) * (SIGMA / np.std(y))
    if drift is not None:
        x = x + drift(np.arange(n) / fs)
    truths = []
    for (a, b, pad), (_length, builder) in zip(slots, specs):
        add, tr = builder(b - a, pad)
        x[a:b] += add
        truths.append([(a + s, a + e) for s, e in tr])
    if true_baseline:
        base = np.full(n, -1000.0) if drift is None else -1000.0 + drift(np.arange(n) / fs)
    else:
        base = np.asarray(characterisation_baseline(x, fs, polarity=1, health=False, cache=False).baseline,
                          dtype=np.float64)
    resid = x - base
    noise = noise_rms(resid)
    tau = 1.0
    if measure_tau and truths and truths[0]:
        spans = [(t[0][0], t[-1][1]) for t in truths]
        tau = ef.estimate_open_pore_tau(resid, spans, noise_rms=noise, fs=fs, L=64)
    return Rig(fs, 1, x, base, noise, tau, slots, truths)


def box(length: int, depth: float, rise: float) -> "tuple[int, Builder]":
    def b(n: int, s: int):
        return erf_event(n, s, s + length, depth, rise), [(s, s + length)]
    return length, b


def scenario(fs: float, kind: str, n_events: int):
    """(specs, true edge rise in samples). The review's cases at ``fs``: 20 ms events, 15 sigma
    (90 pA) unless the case says otherwise."""
    ms = fs / 1e3
    length = int(20 * ms)
    r1 = int(1 * ms)                              # a 1 ms edge
    sharp = max(2, int(round(0.06 * ms)))         # 60 us: 3 samples at 50 kHz, 15 at 250 kHz
    if kind == "plain 1ms":
        return [box(length, 15 * SIGMA, r1)] * n_events, r1
    if kind == "plain sharp":
        return [box(length, 15 * SIGMA, sharp)] * n_events, sharp
    if kind == "deeper nbr":
        gap = int(1.5 * ms)

        def b(n, s):
            # the event (6 sigma) with a DEEPER (15 sigma) event the table does not hold 1.5 ms after it
            return (erf_event(n, s, s + length, 6 * SIGMA, sharp)
                    + erf_event(n, s + length + gap, s + 2 * length + gap, 15 * SIGMA, sharp)), [(s, s + length)]
        return [(length, b)] * n_events, sharp
    edge = sharp if kind.endswith(" sharp") else r1            # "substate end 2ms sharp": 60 us edges
    if kind.startswith("substate end"):
        tail = int(float(kind.split()[2][:-2]) * ms)

        def b(n, s):
            return (erf_event(n, s, s + length - tail, 60.0, edge) + erf_event(n, s, s + length, 30.0, edge)), [(s, s + length)]
        return [(length, b)] * n_events, edge
    if kind.startswith("substate start"):
        shoulder = int(float(kind.split()[2][:-2]) * ms)

        def b(n, s):
            return (erf_event(n, s, s + length, 30.0, edge) + erf_event(n, s + shoulder, s + length, 60.0, edge)), [(s, s + length)]
        return [(length, b)] * n_events, edge
    raise KeyError(kind)


_RIGS: dict = {}


def rig_for(fs: float, kind: str, n_events: int, seed_offset: int = 0, **kw) -> Rig:
    """Cached: building a rig costs more than fitting on it."""
    key = (fs, kind, n_events, seed_offset, tuple(sorted(kw.items(), key=lambda kv: kv[0])))
    if key not in _RIGS:
        specs, _r = scenario(fs, kind, n_events)
        seed = (int(fs / 1e3) * 7 + sum(map(ord, kind)) + seed_offset) % 997
        _RIGS[key] = build_rig(fs, specs, seed=seed, **kw)
    return _RIGS[key]


# ================================================================================================
# Scoring
# ================================================================================================
def place_edges(rng, s: int, e: int, px: int, mode: str) -> "tuple[int, int]":
    """Where a hand puts the bounds: ``px`` is within +-3 px of the truth; ``draft`` within +-10%
    of the span; ``in30`` / ``out30`` are 30% of the span inside / outside."""
    span = e - s
    if mode == "px":
        return s + int(round(rng.uniform(-3, 3) * px)), e + int(round(rng.uniform(-3, 3) * px))
    if mode == "draft":
        return s + int(rng.uniform(-0.1, 0.1) * span), e + int(rng.uniform(-0.1, 0.1) * span)
    if mode == "in30":
        return s + int(0.3 * span), e - int(0.3 * span)
    if mode == "out30":
        return s - int(0.3 * span), e + int(0.3 * span)
    raise KeyError(mode)


def score_fits(rig: Rig, rise_true: float, px: int, seed: int, *, place: str = "px",
               rise_known: bool = True, pick: Callable | None = None) -> Counter:
    """Fit every slot's event from a hand-placed span and count what came back."""
    rng = np.random.default_rng(seed)
    tol = max(3.0, rise_true / 2)
    c: Counter = Counter()
    for (a, b, _), tr in zip(rig.slots, rig.truths):
        s, e = tr[0] if pick is None else pick(tr)
        a0, b0 = place_edges(rng, s, e, px, place)
        if b0 - a0 < 8 or a0 < a or b0 > b:
            continue
        f = ef.fit_edges(rig.x, rig.fs, a0, b0, polarity=rig.polarity, noise_rms=rig.noise, tau=rig.tau,
                         lo=a, hi=b, baseline=rig.base, px_samples=px,
                         rise_s=(rise_true / rig.fs) if rise_known else None, filter_rise_s=FILTER_RISE_S)
        c["n"] += 1
        for name, edge, truth in (("start", f.start, s), ("end", f.end, e)):
            if edge.fitted:
                c["fitted"] += 1
                if abs(edge.position - truth) <= tol:
                    c[name + " right"] += 1
                else:
                    c["wrong"] += 1
            else:
                c["unfitted"] += 1
                assert edge.position == (a0 if name == "start" else b0)     # left exactly as placed
                assert edge.reason
        if f.complete:
            c["row fitted"] += 1
            if abs(f.start.position - s) <= tol and abs(f.end.position - e) <= tol:
                c["row right"] += 1
    return c


def quantise(p: int, px: int) -> int:
    """The sample under the centre of the pixel the pointer is in."""
    return (p // px) * px + px // 2 if px > 1 else p


def catch_at(rig: Rig, p: int, px: int, lo: int, hi: int, *, view_px: int = 1200, rise_true: float | None = None,
             rise_known: bool = True, view: "tuple[int, int] | None" = None) -> ef.Catch:
    """``C`` with the pointer at sample ``p``; the view is ``view_px`` pixels wide with the pointer
    somewhere between 20% and 80% across it (deterministic in ``p``)."""
    if view is None:
        u = np.random.default_rng(p).uniform(0.2, 0.8)
        v0 = int(p - u * view_px * px)
        view = (v0, v0 + view_px * px)
    return ef.catch_event(rig.x, rig.fs, p, polarity=rig.polarity, noise_rms=rig.noise, baseline=rig.base,
                          lo=lo, hi=hi, view_lo=view[0], view_hi=view[1], px_samples=px, tau=rig.tau,
                          rise_s=(rise_true / rig.fs) if (rise_known and rise_true) else None,
                          filter_rise_s=FILTER_RISE_S)


def score_catches(rig: Rig, rise_true: float, px_of: Callable[[int], int], seed: int, *, n_ptr: int = 2,
                  rise_known: bool = True, view_px: int = 1200) -> Counter:
    """A pointer at random interior samples of every event; what did ``C`` return?"""
    rng = np.random.default_rng(seed)
    tol = max(3.0, rise_true / 2)
    c: Counter = Counter()
    for (a, b, _), tr in zip(rig.slots, rig.truths):
        s, e = tr[0][0], tr[-1][1]
        px = px_of(e - s)
        if (e - s) > 0.6 * view_px * px:
            continue                                  # the event is wider than the view
        for _ in range(n_ptr):
            p = quantise(int(rng.integers(s, e)), px)
            ct = catch_at(rig, p, px, a, b, view_px=view_px, rise_true=rise_true, rise_known=rise_known)
            c["n"] += 1
            if not ct.ok:
                c["no trigger" if ct.reason == ef.REASON_NO_EVENT else "refused"] += 1
                continue
            f = ct.fit
            if f.complete:
                if abs(f.start.position - s) <= tol and abs(f.end.position - e) <= tol:
                    c["right"] += 1
                else:
                    c["wrong"] += 1
            else:
                c["unfitted"] += 1
                if f.start.fitted != f.end.fitted:
                    c["one edge"] += 1
    return c


@pytest.fixture(scope="module", autouse=True)
def _release_rigs():
    """The rigs are cached for speed and are tens of megabytes; let go of them when the module is done."""
    yield
    _RIGS.clear()
    _NOISE.clear()


# ================================================================================================
# fit_edges: the review's measured rates as floors (design 5.2)
# ================================================================================================
N_FIT = 60                        # events per case (the review used 100)


def _fit_all_zooms(fs: float, kind: str, *, rise_known: bool = True, place: str = "px"):
    specs, rise = scenario(fs, kind, N_FIT)
    rig = rig_for(fs, kind, N_FIT)
    out = {}
    for px in PX_LIST:
        c = score_fits(rig, rise, px, seed=11 + px, place=place, rise_known=rise_known)
        twin = score_fits(rig.twin(), rise, px, seed=11 + px, place=place, rise_known=rise_known)
        assert twin == c, f"polarity -1 must be the same fit as +1: {kind} {fs} px{px}"
        out[px] = c
    return out


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("kind, floor", [
    ("plain 1ms", 0.93),      # review: 99 to 100%
    ("plain sharp", 0.93),    # 100%
    ("deeper nbr", 0.93),     # 100%: a deeper event 1.5 ms away that the table does not hold
])
def test_edges_placed_within_three_pixels_are_fitted_at_every_zoom(fs, kind, floor):
    for px, c in _fit_all_zooms(fs, kind).items():
        assert c["row right"] / c["n"] >= floor, (kind, fs, px, dict(c))
        assert c["wrong"] <= 0.04 * 2 * c["n"], (kind, fs, px, dict(c))


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("kind", ["substate end 2ms", "substate end 4ms", "substate start 2ms",
                                  "substate end 2ms sharp", "substate start 2ms sharp"])
def test_substate_onsets_and_ends_are_fitted(fs, kind):
    """A 20 ms event with a 30 pA tail (or shoulder) under a 90 pA body: the row is right when BOTH
    edges are at the outer steps. Review: ends 95 to 99%, starts 95 to 98%."""
    for px, c in _fit_all_zooms(fs, kind).items():
        assert c["row right"] / c["n"] >= 0.85, (kind, fs, px, dict(c))
        assert c["wrong"] <= 0.08 * 2 * c["n"], (kind, fs, px, dict(c))


def test_substate_edges_land_on_the_outer_step_not_the_inner_one():
    """Rule 1. Undirected, the least-squares split of the end window takes the LARGEST step of either
    sign, which is the 90 -> 30 pA step 2 ms inside the end (185 to 200 of 200 trials in the review).
    The directed, outermost fit lands on the 30 -> 0 pA step at the real end."""
    from nanoscope.analysis.cusum import _best_split

    fs, kind, px = 50_000.0, "substate end 2ms", 10
    specs, rise = scenario(fs, kind, N_FIT)
    rig = rig_for(fs, kind, N_FIT)
    dev = rig.dev
    tail = int(2 * fs / 1e3)
    rng = np.random.default_rng(3)
    undirected_inner = directed_outer = directed_inner = n = 0
    for (a, b, _), tr in zip(rig.slots, rig.truths):
        s, e = tr[0]
        a0, b0 = place_edges(rng, s, e, px, "px")
        n += 1
        f = ef.fit_edges(rig.x, fs, a0, b0, polarity=1, noise_rms=rig.noise, tau=rig.tau, lo=a, hi=b,
                         baseline=rig.base, px_samples=px, rise_s=rise / fs)
        W = f.params["W"]["end"]
        m, _gain = _best_split(dev, max(b0 - W, a0), min(b0 + W, b), 3)
        undirected_inner += abs(m - (e - tail)) <= 40
        directed_outer += f.end.fitted and abs(f.end.position - e) <= rise / 2 + 3
        directed_inner += f.end.fitted and abs(f.end.position - (e - tail)) <= 40
    assert undirected_inner >= 0.6 * n          # most trials (the review's wider windows: 185 to 200 of 200)
    assert directed_outer >= 0.85 * n           # review: 95 to 99%
    assert directed_inner <= 0.12 * n
    assert directed_inner <= undirected_inner / 5


def test_a_soft_edge_is_fitted_mid_ramp_within_half_a_rise():
    fs = 50_000.0
    specs, rise = scenario(fs, "plain 1ms", N_FIT)
    rig = rig_for(fs, "plain 1ms", N_FIT)
    rng = np.random.default_rng(1)
    errs = []
    for (a, b, _), tr in zip(rig.slots, rig.truths):
        s, e = tr[0]
        a0, b0 = place_edges(rng, s, e, 10, "px")
        f = ef.fit_edges(rig.x, fs, a0, b0, polarity=1, noise_rms=rig.noise, tau=rig.tau, lo=a, hi=b,
                         baseline=rig.base, px_samples=10, rise_s=rise / fs)
        assert f.complete
        errs += [f.start.position - s, f.end.position - e]
    assert np.abs(errs).max() <= rise / 2
    assert abs(np.mean(errs)) <= 0.1 * rise          # not biased to either side of mid-ramp


@pytest.mark.parametrize("fs", FS_LIST)
def test_sub_state_with_the_rise_unknown(fs):
    """The edge rise is the table's median, or a first-pass 10-90% estimate (design 5.1). Unknown, the
    estimate works at 50 kHz and NOT at 250 kHz on a multi-step edge: a soft sub-state edge there is
    left unfitted rather than misplaced, and supplying the table's rise restores it.

    This pins the DOCUMENTED limit; if the estimator is ever improved, flip the 250 kHz assertion."""
    kind, px = "substate end 2ms", 10
    specs, rise = scenario(fs, kind, N_FIT)
    rig = rig_for(fs, kind, N_FIT)
    known = score_fits(rig, rise, px, seed=21, rise_known=True)
    unknown = score_fits(rig, rise, px, seed=21, rise_known=False)
    assert known["row right"] / known["n"] >= 0.85, dict(known)
    assert unknown["wrong"] <= 0.12 * 2 * unknown["n"], dict(unknown)             # declines; does not misplace
    if fs < 100_000:
        assert unknown["row right"] / unknown["n"] >= 0.8, dict(unknown)
    else:
        assert unknown["row right"] / unknown["n"] <= 0.4, dict(unknown)
        # and the fit says why it did not fit: the sub-state edge is too small against a rise it under-estimated
        f = ef.fit_edges(rig.x, fs, *place_edges(np.random.default_rng(2), *rig.truths[0][0], px, "px"),
                         polarity=1, noise_rms=rig.noise, tau=rig.tau, lo=rig.slots[0][0], hi=rig.slots[0][1],
                         baseline=rig.base, px_samples=px, filter_rise_s=FILTER_RISE_S)
        assert f.params["rise_source"] == "estimate" and f.params["rise"]["end"]["samples"] < rise / 2


def test_fit_edges_without_a_filter_rise_still_estimates_the_rise():
    """filter_rise_s is optional; the floor is then two samples."""
    fs = 50_000.0
    specs, rise = scenario(fs, "plain 1ms", N_FIT)
    rig = rig_for(fs, "plain 1ms", N_FIT)
    (a, b, _), tr = rig.slots[3], rig.truths[3]
    s, e = tr[0]
    f = ef.fit_edges(rig.x, fs, s + 5, e - 5, polarity=1, noise_rms=rig.noise, tau=rig.tau, lo=a, hi=b, baseline=rig.base)
    assert f.complete and f.params["rise_floor_samples"] == 2.0 and f.params["px_samples"] == 1.0


# ---- hand-drawn spans that are off by a lot -------------------------------------------------------
@pytest.mark.parametrize("fs", [50_000.0])
def test_spans_drawn_30_percent_outside_are_fitted_when_the_window_reaches(fs):
    """W is capped at 0.45 of the span, and the cap wins: at scan zoom (100 samples per pixel) a span
    drawn 30% too wide still has the true edges inside its windows. (At 250 kHz the same 20 ms event is
    5,000 samples and 30% of the span is beyond the 12-pixel window, so it is not a case for this rule.)"""
    for kind in ("plain 1ms", "plain sharp", "deeper nbr", "substate end 2ms"):
        specs, rise = scenario(fs, kind, N_FIT)
        rig = rig_for(fs, kind, N_FIT)
        c = score_fits(rig, rise, 100, seed=5, place="out30")
        assert c["row right"] / c["n"] >= 0.9, (kind, dict(c))


@pytest.mark.parametrize("fs", FS_LIST)
def test_spans_drawn_30_percent_inside_are_not_misplaced_at_scan_zoom(fs):
    """A span 30% too narrow: the true edges are outside W (0.45 of the NARROWED span). The edges are
    left where they were put, or found by the one outward growth, and not fitted wrongly. (At 1 and 10
    samples per pixel the window is far shorter than the miss, and a split can hug its inner border;
    see the report: those are not asserted.)"""
    for kind in ("plain 1ms", "plain sharp", "substate end 2ms"):
        specs, rise = scenario(fs, kind, N_FIT)
        rig = rig_for(fs, kind, N_FIT)
        c = score_fits(rig, rise, 100, seed=5, place="in30")       # also asserts unfitted => exactly as placed
        assert c["wrong"] <= 2, (kind, fs, dict(c))
        assert c["unfitted"] + c["fitted"] == 2 * c["n"]


def test_an_edge_within_one_growth_of_the_window_is_found_and_further_out_is_left_alone():
    """Rule 6: a split near the OUTER window edge is retried once with the window grown outward."""
    rng = np.random.default_rng(4)
    n, s0, e0 = 4_000, 1_000, 2_500
    x = rng.normal(0, 1, n)
    x[s0:e0] += 20.0

    def fit(shift):
        return ef.fit_edges(x, 50_000.0, s0 + shift, e0 - shift, polarity=1, noise_rms=1.0, tau=1.0, lo=0, hi=n,
                            baseline=0.0, px_samples=1, rise_s=2 / 50_000.0)

    ok = fit(10)               # step 10 samples outside a 12-sample half window, so near the border
    assert ok.complete and ok.start.position == pytest.approx(s0, abs=1) and ok.end.position == pytest.approx(e0, abs=1)
    far = fit(22)              # still near the border after the one growth
    assert not far.start.fitted and far.start.reason == ef.REASON_BORDER and far.start.position == s0 + 22
    assert not far.complete


@pytest.mark.parametrize("fs", [50_000.0])
@pytest.mark.parametrize("dur_ms", [300, 400, 500])
def test_long_events_with_a_true_baseline(fs, dur_ms):
    """300 to 500 ms events fitted at scan zoom. The baseline here is the true one: an event longer
    than the estimator's rough window (0.5 s) is partly absorbed into the CONTRACT's baseline, which is
    a known limit of the track, not of the fit."""
    ms = fs / 1e3
    rig = build_rig(fs, [box(int(dur_ms * ms), 15 * SIGMA, ms)] * 6, seed=dur_ms, pad_s=0.1, true_baseline=True)
    for px in (100, 625):
        for known in (True, False):
            c = score_fits(rig, ms, px, seed=5, rise_known=known)
            assert c["row right"] / c["n"] >= 0.8, (dur_ms, px, known, dict(c))
            assert c["wrong"] <= 1


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("amp_k, period", [(10, 1.0), (30, 1.0)])
def test_a_drift_case_fits_against_the_estimated_track_and_against_a_wrong_constant(fs, amp_k, period):
    """10 and 30 sigma of 1 s sinusoidal drift. Against the contract's estimated track the fit holds at
    every zoom. The outer side is judged against the median of samples beyond the window, not the
    track, so even a constant (WRONG) baseline fits at the zooms where the window is short. (At scan
    zoom the window is wide enough for the drift across it to fail the outer test and the fit declines,
    sometimes misplacing: a wrong baseline is not asserted there.)"""
    ms = fs / 1e3
    drift = lambda t: amp_k * SIGMA * np.sin(2 * np.pi * t / period)           # noqa: E731
    rig = build_rig(fs, [box(int(20 * ms), 15 * SIGMA, ms)] * N_FIT, seed=3, pad_s=0.1, drift=drift)
    flat = Rig(rig.fs, 1, rig.x, np.full_like(rig.base, float(np.median(rig.x))), rig.noise, rig.tau, rig.slots, rig.truths)
    for px in PX_LIST:
        c = score_fits(rig, ms, px, seed=5)
        assert c["row right"] / c["n"] >= 0.9, ("track", fs, amp_k, px, dict(c))
        assert c["wrong"] <= 2
        if px <= 10:
            c = score_fits(flat, ms, px, seed=5)
            assert c["row right"] / c["n"] >= 0.9 and c["wrong"] <= 2, ("constant baseline", fs, amp_k, px, dict(c))


@pytest.mark.parametrize("fs", FS_LIST)
def test_two_to_ten_sample_events_are_left_as_placed_or_fitted_right(fs):
    """A span under 2 * min_len + 2 = 8 samples cannot be fitted and is returned exactly as placed;
    8 to 10 samples may be fitted, and when they are they are right."""
    rng = np.random.default_rng(2)
    for length in range(2, 11):
        rig = build_rig(fs, [box(length, 20 * SIGMA, 1)] * 20, seed=length, pad_s=0.05)
        for (a, b, _), tr in zip(rig.slots, rig.truths):
            s, e = tr[0]
            a0, b0 = s + int(rng.integers(-1, 2)), e + int(rng.integers(-1, 2))
            f = ef.fit_edges(rig.x, fs, a0, b0, polarity=1, noise_rms=rig.noise, tau=rig.tau, lo=a, hi=b,
                             baseline=rig.base, px_samples=1, filter_rise_s=FILTER_RISE_S)
            if b0 - a0 < 8:
                assert (f.start.position, f.end.position) == (a0, b0)
                assert not f.start.fitted and not f.end.fitted
                assert f.start.reason == f.end.reason == ef.REASON_SHORT
            for edge, placed, truth in ((f.start, a0, s), (f.end, b0, e)):
                if edge.fitted:
                    assert abs(edge.position - truth) <= 2, (length, edge, truth)
                else:
                    assert edge.position == placed and edge.reason


# ---- nothing to fit -------------------------------------------------------------------------------
_NOISE: dict = {}


def _noise_trace(kind: str, fs: float):
    """Pure noise, no events: ``synth_trace`` default (white + 30% 1/f), or the same pushed through a
    4-pole 10 kHz low-pass (hardware-like), with the contract's baseline, noise_rms and a tau measured
    by ``estimate_open_pore_tau`` with no rows."""
    key = (kind, fs)
    if key not in _NOISE:
        dur = 12.0 if fs < 100_000 else 4.0
        rec, _ = synth_trace(fs=fs, duration_s=dur, n_events=0, noise_pA=SIGMA, seed=4)
        x = np.asarray(rec.data, dtype=np.float64)
        if kind == "hardware":
            y = np.asarray(lowpass(x, fs, 10e3, order=4), dtype=np.float64)
            x = -1000.0 + (y - y.mean()) * (SIGMA / np.std(y))
        base = np.asarray(characterisation_baseline(x, fs, polarity=1, health=False, cache=False).baseline, dtype=np.float64)
        sig = noise_rms(x - base)
        tau = ef.estimate_open_pore_tau(x - base, [], noise_rms=sig, fs=fs)
        _NOISE[key] = (x, base, sig, tau)
    return _NOISE[key]


@pytest.mark.parametrize("kind", ["pink", "hardware"])
@pytest.mark.parametrize("fs", FS_LIST)
def test_no_fit_on_pure_noise(kind, fs):
    """Spans of 2, 20 and 300 ms drawn on pure noise, at every zoom, both polarities, rise known and
    unknown: not one edge is fitted. Tau is the one ``estimate_open_pore_tau`` measures on the noise
    (1.0 for pink, about 3 for the 250 kHz hardware-like trace); for pink noise, where it is 1, the
    review's own tau = 1 is run too. (An UNDER-stated tau on correlated noise is not a case: it is the
    caller's job to pass the filter's own, which the contract floors it at.) Review: no false fit in
    1,800 spans."""
    x, base, sig, tau = _noise_trace(kind, fs)
    rng = np.random.default_rng(17)
    margin = 50_000
    fitted = rows = spans = 0
    modes = ((tau, FILTER_RISE_S), (tau, None)) + (((1.0, None),) if kind == "pink" else ())
    for pol in (1, -1):
        xs, bs = (x, base) if pol == 1 else (-x, -base)
        for span_s in (2e-3, 20e-3, 300e-3):
            L = int(span_s * fs)
            for px in PX_LIST:
                for t, rise_s in modes:
                    for _ in range(25):
                        s0 = int(rng.integers(L + margin, x.size - 2 * L - margin))
                        f = ef.fit_edges(xs, fs, s0, s0 + L, polarity=pol, noise_rms=sig, tau=t, lo=s0 - margin,
                                         hi=s0 + L + margin, baseline=bs, px_samples=px, rise_s=rise_s,
                                         filter_rise_s=FILTER_RISE_S)
                        spans += 1
                        fitted += f.start.fitted + f.end.fitted
                        rows += f.complete
    assert spans >= 800
    assert rows == 0, (kind, fs, rows, spans)                    # a row is both edges: never
    if kind == "pink":
        assert fitted == 0, (kind, fs, fitted, spans)
    else:
        # Hardware-like noise can give a LONE edge, and at 250 kHz it depends on the realization: 1 in
        # 28,800 spans over eight of them, but 0 to 3 in the 900 spans here over twenty (they are random
        # spans of one trace, so a single odd stretch is hit again and again). A lone edge leaves the
        # row unfitted; the bound is 1%, far from that and far from a regression.
        assert fitted <= max(1, 0.01 * spans), (kind, fs, fitted, spans)


# ================================================================================================
# fit_edges: bad data, API, determinism
# ================================================================================================
def _clean_event(n=6_000, s=2_000, e=3_500, depth=20.0, seed=0, rise=2.0):
    """White-noise (sigma 1) trace with one erf-edged event of ``depth`` sigma: dev == x, baseline 0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n) + erf_event(n, s, e, depth, rise)
    return x, s, e


def _fit_clean(x, s, e, **kw):
    args = dict(polarity=1, noise_rms=1.0, tau=1.0, lo=0, hi=x.size, baseline=0.0, px_samples=1, rise_s=2 / 50_000.0)
    args.update(kw)
    return ef.fit_edges(x, 50_000.0, s, e, **args)


def test_clean_event_is_fitted_to_the_sample():
    x, s, e = _clean_event()
    f = _fit_clean(x, s + 5, e - 4)
    assert f.complete
    assert abs(f.start.position - s) <= 1 and abs(f.end.position - e) <= 1
    assert f.start.z > 6 and f.start.step > 15 and f.end.step > 15 and f.start.reason == f.end.reason == ""


def test_nan_in_one_edges_window_leaves_that_edge_as_placed_and_the_other_fitted():
    x, s, e = _clean_event()
    x[e + 30] = np.nan                                      # inside the end window, 30 samples past the end
    f = _fit_clean(x, s + 5, e - 4)
    assert f.start.fitted and abs(f.start.position - s) <= 1
    assert not f.end.fitted and f.end.position == e - 4 and f.end.reason == ef.REASON_NONFINITE
    assert not f.complete


def test_nan_far_from_both_windows_changes_nothing():
    x, s, e = _clean_event()
    clean = _fit_clean(x, s + 5, e - 4)
    x2 = x.copy()
    x2[100] = np.nan                                        # 1,900 samples before the start window
    x2[5_900] = np.inf
    x2[(s + e) // 2] = np.nan                               # the middle of the event, in neither window
    same = _fit_clean(x2, s + 5, e - 4)
    assert (same.start.position, same.end.position) == (clean.start.position, clean.end.position)
    assert same.complete


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_everywhere_never_raises_and_fits_nothing(bad):
    x = np.full(5_000, bad)
    f = _fit_clean(x, 1_000, 2_000)
    assert not f.start.fitted and not f.end.fitted
    assert (f.start.position, f.end.position) == (1_000, 2_000)
    assert f.start.reason and f.end.reason


def test_a_non_finite_baseline_sample_in_a_window_refuses_that_edge():
    x, s, e = _clean_event()
    base = np.zeros(x.size)
    base[s - 20] = np.nan
    f = _fit_clean(x, s + 5, e - 4, baseline=base)
    assert not f.start.fitted and f.start.reason == ef.REASON_NONFINITE
    assert f.end.fitted


@pytest.mark.parametrize("kwargs, reason_part", [
    (dict(polarity=0), "polarity"),
    (dict(polarity=2), "polarity"),
    (dict(polarity=None), "polarity"),
    (dict(noise_rms=0.0), "noise_rms"),
    (dict(noise_rms=-1.0), "noise_rms"),
    (dict(noise_rms=float("nan")), "noise_rms"),
    (dict(noise_rms=None), "noise_rms"),
])
def test_bad_parameters_return_unfitted_edges_with_a_reason(kwargs, reason_part):
    x, s, e = _clean_event()
    f = _fit_clean(x, s + 5, e - 4, **kwargs)
    assert not f.start.fitted and not f.end.fitted
    assert reason_part in f.start.reason and reason_part in f.end.reason
    assert (f.start.position, f.end.position) == (s + 5, e - 4)


def test_bad_fs_signal_baseline_and_bounds_never_raise():
    x, s, e = _clean_event()
    cases = [
        dict(fs=float("nan")), dict(fs=0.0), dict(fs=-1.0), dict(fs=None),
        dict(signal=np.zeros((4, 4))), dict(signal=np.array([])), dict(signal=None), dict(signal="text"),
        dict(signal=np.array(["a", "b"] * 2000)),
        dict(baseline=np.zeros(10)),                                   # wrong length
        dict(baseline="x"), dict(baseline=np.zeros((x.size, 2))),
        dict(start=-5), dict(end=x.size + 10), dict(start=e, end=s), dict(start=s, end=s),
        dict(start=float("nan")), dict(end=None), dict(start="a"),
        dict(lo=float("nan")), dict(hi=None), dict(lo=e, hi=s), dict(tau=float("nan")), dict(tau=-3.0),
        dict(px_samples=float("nan")), dict(px_samples=-4), dict(rise_s=float("nan")), dict(rise_s=-1.0), dict(rise_s=0.0),
        dict(min_len=0), dict(min_len=None), dict(filter_rise_s=float("nan")),
    ]
    for c in cases:
        args = dict(signal=x, fs=50_000.0, start=s + 5, end=e - 4, polarity=1, noise_rms=1.0, tau=1.0, lo=0, hi=x.size,
                    baseline=0.0, px_samples=1, rise_s=2 / 50_000.0)
        args.update(c)
        signal, fs, start, end = args.pop("signal"), args.pop("fs"), args.pop("start"), args.pop("end")
        f = ef.fit_edges(signal, fs, start, end, **args)                    # must not raise
        assert isinstance(f, ef.EdgeFit)
        assert isinstance(f.start.position, int) and isinstance(f.end.position, int)
        json.loads(f.to_json())                                             # and must still serialise
        for edge in (f.start, f.end):
            assert edge.fitted or edge.reason


def test_the_table_rule_is_a_placeholder_and_unknown_names_raise():
    x, s, e = _clean_event()
    with pytest.raises(NotImplementedError):
        _fit_clean(x, s, e, rule="table")
    with pytest.raises(ValueError):
        _fit_clean(x, s, e, method="cusum")
    with pytest.raises(ValueError):
        _fit_clean(x, s, e, edges=("middle",))
    with pytest.raises(ValueError):
        _fit_clean(x, s, e, edges=())


def test_one_edge_only():
    x, s, e = _clean_event()
    f = _fit_clean(x, s + 5, e - 4, edges=("start",))
    assert f.start.fitted and not f.end.fitted
    assert f.end.position == e - 4 and f.end.reason == ef.REASON_NOT_REQUESTED
    g = _fit_clean(x, s + 5, e - 4, edges=("end",))
    assert g.end.fitted and not g.start.fitted and g.start.reason == ef.REASON_NOT_REQUESTED
    both = _fit_clean(x, s + 5, e - 4)
    assert (f.start, g.end) == (both.start, both.end)                      # requesting one edge is the same fit


def test_fit_is_all_or_nothing_per_row_through_complete():
    x, s, e = _clean_event()
    x[e + 25] = np.nan
    f = _fit_clean(x, s + 5, e - 4)
    assert f.start.fitted and not f.end.fitted and f.complete is False     # one good edge, one not: not a row
    good = _fit_clean(*_clean_event()[:1], s + 5, e - 4)
    assert good.complete


@pytest.mark.parametrize("make, reason", [
    # nothing in the window steps the right way
    (lambda x, s, e: (np.zeros_like(x), dict()), ef.REASON_NO_STEP),
    # a 2-sigma step is significant and under the 3-sigma size floor
    (lambda x, s, e: (x * 0 + erf_event(x.size, s, e, 2.0, 2.0), dict()), ef.REASON_SIZE),
    # a 4-sigma step is big enough and not significant once tau is 30
    (lambda x, s, e: (x * 0 + erf_event(x.size, s, e, 4.0, 2.0), dict(tau=30.0)), ef.REASON_Z),
    # the neighbour leaves 3 samples of window
    (lambda x, s, e: (x, dict(hi=e - 10, lo=0)), ef.REASON_WINDOW),
])
def test_each_failure_names_its_reason(make, reason):
    x, s, e = _clean_event()
    x = x - erf_event(x.size, s, e, 20.0, 2.0)                           # the white noise alone
    sig, extra = make(x, s, e)
    if reason in (ef.REASON_SIZE, ef.REASON_Z):
        sig = sig + x                                                    # noise + the weak step
    if reason == ef.REASON_NO_STEP:
        sig = np.zeros_like(x)
    f = _fit_clean(sig, s + 3, e - 3, **extra)
    got = {f.start.reason, f.end.reason}
    assert reason in got, (reason, f.start, f.end)
    for edge in (f.start, f.end):
        assert not edge.fitted and edge.position in (s + 3, e - 3)       # as placed


def test_a_step_on_a_steep_ramp_is_refused_because_its_outer_side_is_not_baseline():
    """The outer side must sit at the level of the samples beyond the window. Here every outer sample is
    on a ramp climbing 0.09 sigma per sample: the mean of the 32 samples next to the split is 2 sigma away
    from the median of the 64 beyond the window, and no outward re-split or growth finds a baseline."""
    n = 6_000
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, n)
    x[2_000:3_500] += 20.0
    x[:2_000] += np.linspace(0.0, 180.0, 2_000)
    x[3_500:] += np.linspace(180.0, 0.0, n - 3_500)
    f = _fit_clean(x, 2_003, 3_497)
    assert not f.complete
    assert f.start.reason == f.end.reason == ef.REASON_OUTER
    assert (f.start.position, f.end.position) == (2_003, 3_497)


# ---- determinism, inputs untouched, dtypes -------------------------------------------------------
def test_fit_is_deterministic_and_leaves_its_inputs_alone():
    fs = 50_000.0
    specs, rise = scenario(fs, "substate end 2ms", 6)
    rig = rig_for(fs, "substate end 2ms", N_FIT)
    (a, b, _), tr = rig.slots[2], rig.truths[2]
    s, e = tr[0]
    x, base = rig.x.copy(), rig.base.copy()
    x.flags.writeable = False                                             # a read-only array must be accepted
    base.flags.writeable = False
    h = (x.tobytes(), base.tobytes())
    args = dict(polarity=1, noise_rms=rig.noise, tau=rig.tau, lo=a, hi=b, baseline=base, px_samples=10, rise_s=rise / fs)
    f1 = ef.fit_edges(x, fs, s + 12, e - 9, **args)
    f2 = ef.fit_edges(x, fs, s + 12, e - 9, **args)
    assert f1 == f2 and f1.to_json() == f2.to_json()
    assert (x.tobytes(), base.tobytes()) == h
    # a second, unrelated call in between does not change the answer (no hidden state)
    ef.fit_edges(x, fs, s + 50, e - 50, **dict(args, px_samples=100))
    assert ef.fit_edges(x, fs, s + 12, e - 9, **args) == f1


def test_float32_int_and_float64_signals_give_the_same_fit():
    fs = 50_000.0
    rig = rig_for(fs, "plain 1ms", N_FIT)
    (a, b, _), tr = rig.slots[1], rig.truths[1]
    s, e = tr[0]
    x32 = rig.x.astype(np.float32)
    x64 = x32.astype(np.float64)
    args = dict(polarity=1, noise_rms=rig.noise, tau=rig.tau, lo=a, hi=b, px_samples=10, rise_s=1e-3)
    f32 = ef.fit_edges(x32, fs, s + 9, e - 7, baseline=rig.base.astype(np.float32), **args)
    f64 = ef.fit_edges(x64, fs, s + 9, e - 7, baseline=rig.base.astype(np.float32).astype(np.float64), **args)
    assert f32 == f64 and f32.complete
    # raw 16-bit ADC counts: scale to the same shape, offset baseline as a scalar
    adc = np.round(x64 * 10.0).astype(np.int16)
    fi = ef.fit_edges(adc, fs, s + 9, e - 7, polarity=1, noise_rms=rig.noise * 10, tau=rig.tau, lo=a, hi=b,
                      baseline=float(np.median(rig.base) * 10), px_samples=10, rise_s=1e-3)
    assert fi.complete and abs(fi.start.position - f32.start.position) <= 2


def test_baseline_as_track_scalar_or_none():
    """A constant baseline given as a track, a scalar, or not at all (taken from beyond the windows)
    fits the same event to within a sample."""
    x, s, e = _clean_event(seed=2)
    x = x - 500.0
    kw = dict(polarity=1, noise_rms=1.0, tau=1.0, lo=0, hi=x.size, px_samples=1, rise_s=2 / 50_000.0)
    t = ef.fit_edges(x, 50_000.0, s + 5, e - 4, baseline=np.full(x.size, -500.0), **kw)
    c = ef.fit_edges(x, 50_000.0, s + 5, e - 4, baseline=-500.0, **kw)
    n = ef.fit_edges(x, 50_000.0, s + 5, e - 4, baseline=None, **kw)
    assert t.complete and c.complete and n.complete
    assert t == c or (t.start.position, t.end.position) == (c.start.position, c.end.position)
    assert abs(n.start.position - t.start.position) <= 1 and abs(n.end.position - t.end.position) <= 1
    assert t.params["baseline"] == "track" and c.params["baseline"] == "scalar" and n.params["baseline"] == "flank"


def test_polarity_follows_the_sign_convention():
    """polarity +1: dev = x - baseline is positive inside the event (current ABOVE baseline); -1: below.
    The wrong polarity finds a step of the wrong sign in the event's own windows and fits nothing."""
    x, s, e = _clean_event(seed=5)
    up = _fit_clean(x, s + 5, e - 4, polarity=1)
    wrong = _fit_clean(x, s + 5, e - 4, polarity=-1)
    down = _fit_clean(-x, s + 5, e - 4, polarity=-1)
    assert up.complete and down.complete
    assert (up.start.position, up.end.position) == (down.start.position, down.end.position)
    assert not wrong.complete


def test_fit_json_is_compact_deterministic_and_says_how_the_edges_were_made():
    fs = 50_000.0
    rig = rig_for(fs, "substate end 2ms", N_FIT)
    (a, b, _), tr = rig.slots[4], rig.truths[4]
    s, e = tr[0]
    kw = dict(polarity=1, noise_rms=rig.noise, tau=rig.tau, lo=a, hi=b, baseline=rig.base, px_samples=10)
    f = ef.fit_edges(rig.x, fs, s + 4, e - 4, rise_s=1e-3, **kw)
    j = f.to_json()
    d = json.loads(j)
    assert j == json.dumps(d, separators=(",", ":"), sort_keys=True)          # compact and key-sorted
    assert j == ef.fit_edges(rig.x, fs, s + 4, e - 4, rise_s=1e-3, **kw).to_json()
    assert d["method"] == "step" and d["version"] == ef.VERSION
    assert d["fs"] == fs and d["px_samples"] == 10 and d["baseline"] == "track" and d["min_len"] == 3
    assert d["noise"] == pytest.approx(rig.noise, rel=1e-5) and d["tau"] == pytest.approx(max(rig.tau, 1.0), rel=1e-5)
    assert d["rise_source"] == "table" and d["rise_s"] == pytest.approx(1e-3)
    assert set(d["W"]) == {"start", "end"} and d["W"]["start"] >= 12 * 10 * 0 + 12
    assert d["rise"]["start"]["source"] == "table" and d["rise"]["start"]["s"] == pytest.approx(1e-3)
    assert d["gates"]["z_min"] == 6.0 and d["gates"]["size_frac"] == 0.15 and d["gates"]["max_cuts"] == 3
    assert d["placed"] == [s + 4, e - 4] and d["lo"] == a and d["hi"] == b
    for side, edge in (("start", f.start), ("end", f.end)):
        assert d["edges"][side] == {"pos": edge.position, "fitted": edge.fitted, "z": pytest.approx(edge.z, rel=1e-5),
                                    "step": pytest.approx(edge.step, rel=1e-5), "reason": edge.reason}
    # rise unknown: the estimate and its source are recorded per edge
    g = ef.fit_edges(rig.x, fs, s + 4, e - 4, rise_s=None, filter_rise_s=FILTER_RISE_S, **kw)
    dg = json.loads(g.to_json())
    assert dg["rise_source"] == "estimate" and dg["rise_s"] is None
    assert {v["source"] for v in dg["rise"].values()} <= {"estimate", "floor"}
    # an unfitted edge serialises too, with its reason; nothing is NaN
    bad = _fit_clean(np.full(5_000, np.nan), 1_000, 2_000)
    assert "NaN" not in bad.to_json() and json.loads(bad.to_json())["edges"]["start"]["fitted"] is False


def test_the_fit_reads_only_the_windows_not_the_signal():
    """Cost does not scale with the recording: the samples read are a few thousand out of millions."""
    n = 3_000_000
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, n).astype(np.float32)
    x[1_000_000:1_001_000] += 20.0
    read = []
    orig = ef._Dev.block

    def spy(self, a, b, step=1):
        read.append(len(range(a, b, step)))
        return orig(self, a, b, step)

    ef._Dev.block = spy
    try:
        f = ef.fit_edges(x, 50_000.0, 1_000_004, 1_000_996, polarity=1, noise_rms=1.0, tau=1.0, lo=0, hi=n,
                         baseline=0.0, px_samples=10, rise_s=2 / 50_000.0)
    finally:
        ef._Dev.block = orig
    assert f.complete
    assert sum(read) < 0.01 * n, read


# ================================================================================================
# catch_event: C (design 3.2)
# ================================================================================================
def view_for(p: int, px: int, view_px: int = 1200) -> "tuple[int, int]":
    u = np.random.default_rng(p).uniform(0.2, 0.8)
    v0 = int(p - u * view_px * px)
    return v0, v0 + view_px * px


def _catch_rig(fs: float, dur_ms: float, k_sigma: float, n: int) -> "tuple[Rig, int]":
    ms = fs / 1e3
    rise = max(2, int(round(FILTER_RISE_S * fs)))
    key = ("catch", fs, dur_ms, k_sigma, n)
    if key not in _RIGS:
        _RIGS[key] = build_rig(fs, [box(int(dur_ms * ms), k_sigma * SIGMA, rise)] * n, seed=int(dur_ms * 10 + k_sigma), pad_s=0.1)
    return _RIGS[key], rise


def _smooth_samples(fs: float) -> int:
    return max(3, int(round(3 * FILTER_RISE_S * fs)))


def judge_catches(rig: Rig, rise: float, px_of: Callable[[int], int], seed: int, *, n_ptr: int = 2,
                  rise_known: bool = True) -> Counter:
    """score_catches, plus: a view refusal is only legitimate when the true event reaches the view edge."""
    rng = np.random.default_rng(seed)
    c = score_catches(rig, rise, px_of, seed, n_ptr=n_ptr, rise_known=rise_known)
    # re-walk the same pointers to check the refusals
    rng = np.random.default_rng(seed)
    margin_of = lambda px: 2 * (_smooth_samples(rig.fs) + px)            # noqa: E731
    for (a, b, _), tr in zip(rig.slots, rig.truths):
        s, e = tr[0][0], tr[-1][1]
        px = px_of(e - s)
        if (e - s) > 0.6 * 1200 * px:
            continue
        for _ in range(n_ptr):
            p = quantise(int(rng.integers(s, e)), px)
            ct = catch_at(rig, p, px, a, b, rise_true=rise, rise_known=rise_known)
            if not ct.ok and ct.reason == ef.REASON_VIEW:
                v0, v1 = view_for(p, px)
                m = margin_of(px)
                if s - m > v0 and e + m < v1:
                    c["bad refusal"] += 1
    return c


EVENT_ZOOM = lambda n: max(1, int(round(n / 240)))                        # noqa: E731  an event is a fifth of the view


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("dur_ms", [0.2, 2, 20])
def test_c_catches_the_event_under_the_pointer(fs, dur_ms):
    """Pointer at a random interior sample, 10 sigma events, at the zoom where the event is a fifth of
    the view and at 10 and 100 samples per pixel. Review: 97 to 100% (0.2 ms events: 82 to 100%)."""
    rig, rise = _catch_rig(fs, dur_ms, 10, 24 if fs > 100_000 else 30)
    for name, px_of in (("event-zoom", EVENT_ZOOM), ("px10", lambda n: 10), ("px100", lambda n: 100)):
        c = judge_catches(rig, rise, px_of, seed=3)
        if c["n"] == 0:
            continue
        asked = c["n"] - c["refused"]
        assert c["bad refusal"] == 0, (name, dict(c))
        assert asked >= 0.5 * c["n"]
        assert c["right"] / asked >= 0.93, (fs, dur_ms, name, dict(c))
        # a fitted-but-misplaced catch is rare (a noise or baseline-track bump ahead of an event, which the
        # outermost rule takes for a shoulder); none at the default seed, 4% at the worst of 40 others
        assert c["wrong"] <= 0.06 * c["n"], (fs, dur_ms, name, dict(c))


def test_c_catches_a_weak_event_less_often_but_never_wrongly():
    """5 sigma, 0.2 ms events at event zoom: caught about 80% of the time (review 82%); the rest is 'no
    event under the pointer', not a misplaced one."""
    rig, rise = _catch_rig(50_000.0, 0.2, 5, 60)
    c = judge_catches(rig, rise, EVENT_ZOOM, seed=3)
    assert c["right"] / c["n"] >= 0.70, dict(c)
    assert c["wrong"] <= 0.06 * c["n"] and c["no trigger"] + c["right"] + c["unfitted"] + c["wrong"] == c["n"], dict(c)


def test_c_reports_the_placed_edges_the_foot_the_peak_and_the_fit():
    fs = 50_000.0
    rig, rise = _catch_rig(fs, 20, 10, 30)
    (a, b, _), tr = rig.slots[5], rig.truths[5]
    s, e = tr[0]
    p = (s + e) // 2
    ct = catch_at(rig, p, 10, a, b, rise_true=rise)
    assert ct.ok and ct.reason == ""
    assert abs(ct.start - s) <= 3 and abs(ct.end - e) <= 3                      # half-peak edges of a sharp event
    assert ct.foot_start <= ct.start < ct.end <= ct.foot_end and ct.foot_end - ct.foot_start < 0.03 * fs
    assert ct.peak_dev == pytest.approx(10 * SIGMA, rel=0.25)
    assert ct.fit.complete and abs(ct.fit.start.position - s) <= 3 and abs(ct.fit.end.position - e) <= 3
    d = json.loads(ct.fit.to_json())
    assert d["catch"]["foot"] == [ct.foot_start, ct.foot_end] and d["catch"]["pointer"] == p
    assert d["catch"]["smooth_samples"] == _smooth_samples(fs) and d["catch"]["trigger_sigma"] == 4.0
    twin = catch_at(rig.twin(), p, 10, a, b, rise_true=rise)
    assert twin == ct                                                              # polarity -1: the same catch


def test_c_widens_the_fit_window_to_the_foot_so_a_substate_tail_is_not_cut_off():
    """A 90 pA body with a 2 ms 30 pA tail: half the peak is 45 pA, so the placed end is the INNER step;
    the foot (2 sigma) reaches the real end and the fit window follows it there."""
    fs, ms = 50_000.0, 50.0
    L, tail, rise = int(20 * ms), int(2 * ms), 3

    def b(n, s):
        return (erf_event(n, s, s + L - tail, 60.0, rise) + erf_event(n, s, s + L, 30.0, rise)), [(s, s + L)]
    rig = build_rig(fs, [(L, b)] * 20, seed=5, pad_s=0.1)
    hits = inner = 0
    for (a, bb, _), tr in zip(rig.slots, rig.truths):
        s, e = tr[0]
        ct = catch_at(rig, (s + e) // 2, 10, a, bb, rise_true=rise)
        assert ct.ok
        assert e - tail - 6 <= ct.end <= e - tail + 6                              # placed at the inner (half-peak) step ...
        assert ct.foot_end >= e - 6                                                # ... the foot reaches the outer end
        hits += ct.fit.end.fitted and abs(ct.fit.end.position - e) <= 4            # ... and the fit finds it
        inner += ct.fit.end.fitted and abs(ct.fit.end.position - (e - tail)) <= 10
    assert hits >= 0.9 * len(rig.slots) and inner == 0


@pytest.mark.parametrize("fs", FS_LIST)
def test_c_finds_the_whole_sub_state_event(fs):
    """Tail or shoulder, sharp and 1 ms edges, at event zoom. Review: 88 to 94% (sharp), 97 to 99% (1 ms)."""
    ms = fs / 1e3
    sharp = max(2, int(round(FILTER_RISE_S * fs)))
    for edges, floor in ((sharp, 0.85), (int(ms), 0.88)):
        for kind in ("tail", "shoulder"):
            L, t2 = int(20 * ms), int(2 * ms)

            def b(n, s, edges=edges, kind=kind):
                if kind == "tail":
                    return erf_event(n, s, s + L - t2, 60.0, edges) + erf_event(n, s, s + L, 30.0, edges), [(s, s + L)]
                return erf_event(n, s, s + L, 30.0, edges) + erf_event(n, s + t2, s + L, 60.0, edges), [(s, s + L)]
            key = ("substate catch", fs, edges, kind)
            if key not in _RIGS:
                _RIGS[key] = build_rig(fs, [(L, b)] * 24, seed=5 + len(kind), pad_s=0.1)
            c = judge_catches(_RIGS[key], edges, EVENT_ZOOM, seed=7)
            asked = c["n"] - c["refused"]
            assert c["right"] / asked >= floor, (fs, edges, kind, dict(c))
            assert c["wrong"] <= 0.06 * c["n"] and c["bad refusal"] == 0, (fs, edges, kind, dict(c))


def _pair_rig(fs: float, gap_ms: float, rise: float, n: int = 30) -> Rig:
    ms = fs / 1e3
    La, g = int(2 * ms), int(gap_ms * ms)
    key = ("pair", fs, gap_ms, rise, n)
    if key not in _RIGS:
        def b(m, s):
            return (erf_event(m, s, s + La, 10 * SIGMA, rise) + erf_event(m, s + La + g, s + 2 * La + g, 10 * SIGMA, rise),
                    [(s, s + La), (s + La + g, s + 2 * La + g)])
        _RIGS[key] = build_rig(fs, [(2 * La + g, b)] * n, seed=int(gap_ms * 10), pad_s=0.1)
    return _RIGS[key]


@pytest.mark.parametrize("fs", FS_LIST)
@pytest.mark.parametrize("gap_ms", [0.3, 1.0])
def test_two_close_events_stay_two(fs, gap_ms):
    """Two missed 2 ms events with sharp edges, 0.3 and 1 ms apart, neither in the table: C on either
    one returns that one (review: 93 to 100% at 1 to 10 samples per pixel)."""
    rise = max(2, int(round(FILTER_RISE_S * fs)))
    rig = _pair_rig(fs, gap_ms, rise)
    tol = max(3.0, rise / 2)
    rng = np.random.default_rng(8)
    px = 10
    c: Counter = Counter()
    for (a, b, _), tr in zip(rig.slots, rig.truths):
        for which in (0, 1):
            s, e = tr[which]
            p = quantise(int(rng.integers(s, e)), px)
            ct = catch_at(rig, p, px, a, b, rise_true=rise)
            c["n"] += 1
            if not ct.ok:
                c["refused"] += 1
                continue
            f = ct.fit
            if f.complete and abs(f.start.position - s) <= tol and abs(f.end.position - e) <= tol:
                c["separate"] += 1
            elif ct.start <= tr[0][0] + 3 and ct.end >= tr[1][1] - 3:
                c["merged"] += 1
            else:
                c["other"] += 1
    assert c["separate"] / c["n"] >= 0.93, dict(c)
    assert c["merged"] == 0, dict(c)


def test_c_respects_the_neighbouring_events_of_the_table():
    """Two events with 1 ms edges 0.3 ms apart never return to baseline between them, so C on one of
    them runs into the other (a documented limit) -- unless the table holds the neighbour: lo / hi stop
    the growth and the fit windows."""
    fs, ms = 50_000.0, 50.0
    soft = int(ms)
    rig = _pair_rig(fs, 0.3, soft)
    rng = np.random.default_rng(9)
    merged = bounded = n = 0
    for (a, b, _), tr in zip(rig.slots, rig.truths):
        (s0, e0), (s1, e1) = tr
        p = quantise(int(rng.integers(s0, e0)), 10)
        free = catch_at(rig, p, 10, a, b, rise_true=soft)                        # neighbour not in the table
        held = catch_at(rig, p, 10, a, s1, rise_true=soft)                       # neighbour starts at hi = s1
        n += 1
        merged += free.ok and free.end > e0 + 5
        if held.ok:
            bounded += held.end <= s1 and held.foot_end <= s1 and held.start >= a
            for edge in (held.fit.start, held.fit.end):
                assert a <= edge.position <= s1
    assert merged >= 0.8 * n
    assert bounded >= 0.9 * n


def test_c_on_either_side_of_a_gap_keeps_to_its_own_bounds():
    """lo / hi that cut straight through a long event confine C to that side: the event is clipped at
    the neighbour, never grown past it."""
    fs = 50_000.0
    rig, rise = _catch_rig(fs, 20, 10, 30)
    (a, b, _), tr = rig.slots[0], rig.truths[0]
    s, e = tr[0]
    mid = (s + e) // 2
    left = catch_at(rig, mid - 100, 10, a, mid, rise_true=rise)
    right = catch_at(rig, mid + 100, 10, mid, b, rise_true=rise)
    assert left.ok and right.ok
    assert abs(left.start - s) <= 3 and left.end == mid and left.foot_end == mid
    assert right.start == mid and abs(right.end - e) <= 3


@pytest.mark.parametrize("kind", ["pink", "hardware"])
@pytest.mark.parametrize("fs", FS_LIST)
def test_c_refuses_on_baseline_with_no_false_catch(kind, fs):
    """600 pointers on pure noise per case, both polarities and three zooms: 'No event under the pointer'
    every time (review: no false catch in 400 trials)."""
    x, base, sig, tau = _noise_trace(kind, fs)
    rng = np.random.default_rng(31)
    n = false = 0
    for pol in (1, -1):
        xs, bs = (x, base) if pol == 1 else (-x, -base)
        for px in PX_LIST:
            for _ in range(100):
                p = quantise(int(rng.integers(50_000, x.size - 50_000)), px)
                ct = ef.catch_event(xs, fs, p, polarity=pol, noise_rms=sig, baseline=bs, lo=0, hi=x.size,
                                    view_lo=p - 600 * px, view_hi=p + 600 * px, px_samples=px, tau=tau,
                                    filter_rise_s=FILTER_RISE_S)
                n += 1
                if ct.ok or ct.reason != ef.REASON_NO_EVENT:
                    false += 1
                    assert not (ct.ok and ct.fit.complete)             # and never as a fitted row
                else:
                    assert ct.start is None and ct.fit is None and math.isnan(ct.peak_dev)
    assert n == 600
    # none at the default seed and none on pink noise at any seed tried; on other realizations of the
    # hardware-like 250 kHz trace 1 to 2 pointers in 600 trigger (4.1 sigma, at 100 samples per pixel),
    # and none of those is ever fitted as a row (asserted above). The bound is 1%.
    assert false <= (0 if kind == "pink" else 6), (kind, fs, false)


def test_c_on_baseline_under_strong_drift_triggers_rarely_and_never_fits():
    """30 sigma of 1 s drift: the review saw at most 0.7% triggers on baseline, none fitted."""
    fs, ms = 50_000.0, 50.0
    drift = lambda t: 30 * SIGMA * np.sin(2 * np.pi * t / 1.0)                    # noqa: E731
    rig = build_rig(fs, [box(int(20 * ms), 10 * SIGMA, ms)] * 40, seed=11, pad_s=0.1, drift=drift)
    rng = np.random.default_rng(5)
    n = trig = fitted = 0
    for _ in range(600):
        i = int(rng.integers(len(rig.slots)))
        a, b, _pad = rig.slots[i]
        s, e = rig.truths[i][0][0], rig.truths[i][-1][1]
        gap_l, gap_r = s - a - int(0.02 * fs), b - e - int(0.02 * fs)
        p = int(a + rng.integers(0, max(1, gap_l))) if rng.random() < 0.5 else int(e + 0.02 * fs + rng.integers(0, max(1, gap_r)))
        p = quantise(p, 10)
        ct = catch_at(rig, p, 10, a, b, rise_true=ms)
        n += 1
        trig += ct.ok or ct.reason != ef.REASON_NO_EVENT
        fitted += bool(ct.ok and ct.fit.complete)
    assert trig <= 0.02 * n and fitted == 0, (trig, fitted, n)


def test_c_catches_in_drift():
    fs, ms = 50_000.0, 50.0
    drift = lambda t: 10 * SIGMA * np.sin(2 * np.pi * t / 1.0)                    # noqa: E731
    rig = build_rig(fs, [box(int(20 * ms), 10 * SIGMA, 3)] * 30, seed=12, pad_s=0.1, drift=drift)
    c = judge_catches(rig, 3, EVENT_ZOOM, seed=4)
    asked = c["n"] - c["refused"]
    assert c["right"] / asked >= 0.93 and c["wrong"] <= 0.06 * c["n"], dict(c)


def test_c_refuses_at_the_view_edge_with_the_sentence():
    """If growth reaches the edge of the view, refuse: 'The event runs past the view: zoom out'."""
    assert ef.REASON_VIEW == "The event runs past the view: zoom out"
    assert ef.REASON_NO_EVENT == "No event under the pointer"
    fs = 50_000.0
    rig, rise = _catch_rig(fs, 20, 10, 30)
    (a, b, _), tr = rig.slots[2], rig.truths[2]
    s, e = tr[0]
    p = (s + e) // 2
    inside = catch_at(rig, p, 10, a, b, rise_true=rise, view=(s - 400, e + 400))
    cut_left = catch_at(rig, p, 10, a, b, rise_true=rise, view=(s + 300, e + 400))
    cut_right = catch_at(rig, p, 10, a, b, rise_true=rise, view=(s - 400, e - 300))
    assert inside.ok
    for ct in (cut_left, cut_right):
        assert not ct.ok and ct.reason == ef.REASON_VIEW
        assert ct.start is None and ct.end is None and ct.fit is None
    # a neighbour at the same place as the view edge is not a refusal: growth stopped at a table row
    held = catch_at(rig, p, 10, s + 300, b, rise_true=rise, view=(s + 300, e + 400))
    assert held.ok and held.start == s + 300
    # the recording's own end is not "past the view" either
    first = catch_at(rig, p, 10, 0, b, rise_true=rise, view=(0, e + 400))
    assert first.ok


def test_c_other_refusals_and_bad_input_never_raise():
    fs = 50_000.0
    rig, rise = _catch_rig(fs, 20, 10, 30)
    (a, b, _), tr = rig.slots[2], rig.truths[2]
    s, e = tr[0]
    p = (s + e) // 2
    args = dict(polarity=1, noise_rms=rig.noise, baseline=rig.base, lo=a, hi=b, view_lo=s - 400, view_hi=e + 400,
                px_samples=10, tau=rig.tau, filter_rise_s=FILTER_RISE_S)
    assert catch_at(rig, p, 10, a, b, view=(s - 400, e + 400)).ok
    off_view = ef.catch_event(rig.x, fs, e + 1_000, **args)
    assert not off_view.ok and off_view.reason == ef.REASON_OUTSIDE
    on_row = ef.catch_event(rig.x, fs, p, **dict(args, lo=p + 5))
    assert not on_row.ok and on_row.reason == ef.REASON_ON_EVENT
    for bad in (dict(pointer=float("nan")), dict(pointer=-5), dict(pointer=10 ** 9), dict(polarity=0), dict(noise_rms=0.0),
                dict(noise_rms=float("nan")), dict(fs=float("nan")), dict(fs=0.0), dict(view_lo=10, view_hi=5), dict(lo=b, hi=a),
                dict(px_samples=float("nan")), dict(px_samples=-3), dict(baseline=np.zeros(5)), dict(baseline="x"),
                dict(signal=np.zeros(0)), dict(signal=np.zeros((3, 3))), dict(signal=None), dict(tau=float("nan")),
                dict(rise_s=float("nan")), dict(min_len=0), dict(view_lo=None), dict(lo=None), dict(filter_rise_s=-1.0)):
        kw = dict(args, pointer=p, signal=rig.x, fs=fs)
        kw.update(bad)
        signal, fs_, pointer = kw.pop("signal"), kw.pop("fs"), kw.pop("pointer")
        ct = ef.catch_event(signal, fs_, pointer, **kw)                           # must not raise
        assert isinstance(ct, ef.Catch)
        assert ct.ok or ct.reason


def test_c_on_a_tiny_event_is_ok_and_leaves_the_fit_unfitted_or_right():
    fs = 50_000.0
    rig = build_rig(fs, [box(4, 25 * SIGMA, 1)] * 20, seed=9, pad_s=0.05)
    oks = 0
    for (a, b, _), tr in zip(rig.slots, rig.truths):
        s, e = tr[0]
        ct = catch_at(rig, s + 2, 1, a, b, rise_true=2, view=(s - 600, s + 600))
        if ct.ok:
            oks += 1
            for edge in (ct.fit.start, ct.fit.end):
                assert edge.fitted or edge.reason
    assert oks >= 15


def test_c_smooths_outward_from_the_pointer_not_over_the_signal():
    """The smoothing runs in blocks outward from the pointer: for a 2 ms event in a 3 M-sample signal
    only a few thousand samples are ever read."""
    fs, n = 50_000.0, 3_000_000
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, n)
    x[1_500_000:1_500_100] += 30.0
    read = []
    orig = ef._Dev.block

    def spy(self, a, b, step=1):
        read.append(len(range(a, b, step)))
        return orig(self, a, b, step)

    ef._Dev.block = spy
    try:
        ct = ef.catch_event(x, fs, 1_500_050, polarity=1, noise_rms=1.0, baseline=0.0, lo=0, hi=n,
                            view_lo=1_500_050 - 6_000, view_hi=1_500_050 + 6_000, px_samples=10, tau=1.0,
                            filter_rise_s=FILTER_RISE_S)
    finally:
        ef._Dev.block = orig
    assert ct.ok and abs(ct.start - 1_500_000) <= 3 and abs(ct.end - 1_500_100) <= 3
    assert sum(read) < 0.02 * n and max(read) < 0.01 * n, (sum(read), max(read))


# ================================================================================================
# estimate_open_pore_tau
# ================================================================================================
def _reference_gap_tau(r, spans, L=64):
    """The definition, written out the slow way: high-pass each open-pore gap (clear of every span by L)
    by a (2*(L//2)+1)-sample moving average, drop the incomplete ends, Bartlett-weighted integrated
    autocorrelation time, median over gaps."""
    from nanoscope.analysis.cusum import integrated_autocorr_time

    w = max(3, int(L) | 1)
    h = w // 2
    edges = [0]
    for s, e in sorted(spans):
        edges += [s - L, e + L]
    edges += [r.size]
    taus = []
    for lo, hi in zip(edges[0::2], edges[1::2]):
        lo, hi = max(lo, 0), min(hi, r.size)
        if hi - lo >= max(64, 4 * L) + 2 * h:
            seg = r[lo:hi]
            hp = seg - np.convolve(seg, np.ones(w) / w, mode="same")
            taus.append(integrated_autocorr_time(hp[h:hi - lo - h], n_scale=L))
    return float(np.median(taus))


def _correlated_with_events(seed=11, n=300_000, n_events=20, amp=15.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=n) + 0.4 * np.cumsum(rng.normal(size=n))
    r = r - np.convolve(r, np.ones(2001) / 2001, mode="same")
    starts = 4_000 + 14_000 * np.arange(n_events)
    spans = [(int(s), int(s) + 3_000) for s in starts]
    for s, e in spans:
        r[s:e] += amp
    return r, spans


def test_tau_with_spans_is_the_median_over_the_open_pore_gaps():
    r, spans = _correlated_with_events()
    tau = ef.estimate_open_pore_tau(r, spans, noise_rms=1.0, fs=50_000.0)
    ref = _reference_gap_tau(r, spans)
    assert ref > 1.5                                           # a non-trivial number, not the clamp at 1
    assert tau == pytest.approx(ref, rel=1e-9)
    assert ef.estimate_open_pore_tau(r, np.array(spans), noise_rms=1.0, fs=50_000.0) == tau      # arrays are fine too
    assert ef.estimate_open_pore_tau(r, spans[::-1], noise_rms=1.0, fs=50_000.0) == tau          # and so is any order
    other_scale = ef.estimate_open_pore_tau(r, spans, noise_rms=1.0, fs=50_000.0, L=32)
    assert other_scale == pytest.approx(_reference_gap_tau(r, spans, L=32), rel=1e-9) and other_scale != tau


def test_tau_with_no_spans_uses_pseudo_events_not_the_whole_residual():
    """White noise (tau 1) with 100 boxes of 40 sigma: the whole residual reads 4.5 because the events are
    counted as open pore; with no rows the 4-sigma runs are the events, and the answer is the noise's."""
    from nanoscope.analysis.cusum import open_pore_autocorr_time

    rng = np.random.default_rng(7)
    n, per, length = 204_000, 2_000, 1_000
    r = rng.normal(size=n)
    spans = [(1_000 + per * i, 1_000 + per * i + length) for i in range(100)]
    for s, e in spans:
        r[s:e] += 40.0
    empty = np.array([], dtype=np.int64)
    whole = open_pore_autocorr_time(r, empty, empty, n_scale=64, pad=64)           # events counted as open pore
    known = ef.estimate_open_pore_tau(r, spans, noise_rms=1.0, fs=50_000.0)
    pseudo = ef.estimate_open_pore_tau(r, [], noise_rms=1.0, fs=50_000.0)
    pseudo_none = ef.estimate_open_pore_tau(r, None, noise_rms=1.0, fs=50_000.0)
    assert whole > 3.5
    assert known == pytest.approx(1.0, abs=0.1) and pseudo == pytest.approx(1.0, abs=0.1)
    assert pseudo_none == pseudo


def test_tau_pseudo_events_agree_with_the_rows_on_correlated_noise():
    r, spans = _correlated_with_events()
    sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    known = ef.estimate_open_pore_tau(r, spans, noise_rms=sigma, fs=50_000.0)
    pseudo = ef.estimate_open_pore_tau(r, [], noise_rms=sigma, fs=50_000.0)
    assert pseudo == pytest.approx(known, rel=0.25)
    # the same residual with the events removed: the open-pore value both are estimating
    quiet = r.copy()
    for s, e in spans:
        quiet[s:e] -= 15.0
    assert ef.estimate_open_pore_tau(quiet, [], noise_rms=sigma, fs=50_000.0) == pytest.approx(known, rel=0.25)


def test_tau_is_floored_at_the_filter_autocorrelation_time():
    r, spans = _correlated_with_events()
    base = ef.estimate_open_pore_tau(r, spans, noise_rms=1.0, fs=50_000.0)
    assert ef.estimate_open_pore_tau(r, spans, noise_rms=1.0, fs=50_000.0, filter_autocorr=base + 5) == pytest.approx(base + 5)
    assert ef.estimate_open_pore_tau(r, [], noise_rms=1.0, fs=50_000.0, filter_autocorr=40.0) == 40.0
    assert ef.estimate_open_pore_tau(r, spans, noise_rms=1.0, fs=50_000.0, filter_autocorr=0.2) == base
    assert ef.estimate_open_pore_tau(r, spans, noise_rms=1.0, fs=50_000.0, filter_autocorr=float("nan")) == base


def test_tau_is_never_below_one_and_never_raises():
    rng = np.random.default_rng(0)
    white = rng.normal(size=50_000)
    assert ef.estimate_open_pore_tau(white, [], noise_rms=1.0, fs=1e3) == pytest.approx(1.0, abs=0.05)
    assert ef.estimate_open_pore_tau(white, [], noise_rms=1.0, fs=1e3) >= 1.0
    withbad = white.copy()
    withbad[500] = np.nan
    withbad[7_000:7_100] = np.inf
    for spans in (None, [], [(20_000, 21_000)]):
        t = ef.estimate_open_pore_tau(withbad, spans, noise_rms=1.0, fs=1e3)
        assert np.isfinite(t) and t >= 1.0
    for r in (np.zeros(10), np.array([]), np.zeros((4, 100)), np.full(5_000, np.nan), np.zeros(5_000), "text", None):
        assert ef.estimate_open_pore_tau(r, [], noise_rms=1.0, fs=1e3) == 1.0
    assert ef.estimate_open_pore_tau(np.zeros(10), [], noise_rms=1.0, fs=1e3, filter_autocorr=3.5) == 3.5
    for noise in (0.0, -1.0, float("nan"), None):
        assert ef.estimate_open_pore_tau(white, [], noise_rms=noise, fs=1e3) >= 1.0
    for spans in ([(5, 3)], [(float("nan"), 10)], [(-100, 10 ** 9)], [(10, 10)], [(0, 50_000)]):
        assert ef.estimate_open_pore_tau(white, spans, noise_rms=1.0, fs=1e3) >= 1.0


def test_tau_cost_is_bounded_whatever_the_recording_holds():
    """One huge gap, or twelve thousand tiny ones: the measurement is capped at about 4 M samples in at
    most 2,048 pieces, so it stays cheap on 15 M-sample recordings."""
    calls = []
    orig = ef._piece_tau
    ef._piece_tau = lambda seg, w, L: (calls.append(seg.size), orig(seg, w, L))[1]
    try:
        rng = np.random.default_rng(1)
        r = rng.normal(size=5_000_000)
        t0 = time.perf_counter()
        assert ef.estimate_open_pore_tau(r, [], noise_rms=1.0, fs=1e3) == pytest.approx(1.0, abs=0.05)
        assert time.perf_counter() - t0 < 10.0
        assert sum(calls) <= ef._GAP_BUDGET + 2 * ef._GAP_CHUNK and len(calls) <= ef._GAP_MAX_PIECES
        calls.clear()
        starts = np.arange(1_000, 3_000_000 - 2_000, 500)               # 6,000 rows with 272-sample gaps ... all too short
        r = rng.normal(size=3_000_000)
        spans = [(int(s), int(s) + 100) for s in starts]
        assert ef.estimate_open_pore_tau(r, spans, noise_rms=1.0, fs=1e3) == pytest.approx(1.0, abs=0.05)
        assert len(calls) <= ef._GAP_MAX_PIECES and sum(calls) <= ef._GAP_BUDGET + 2 * ef._GAP_CHUNK
    finally:
        ef._piece_tau = orig


@pytest.mark.xfail(strict=False, reason=(
    "cusum.open_pore_autocorr_time builds its gap list as [0, b0 + pad, a1 - pad, b1 + pad, ...] and pairs "
    "(even, odd), i.e. it measures [0, b0 + pad), [a1 - pad, b1 + pad), ... -- the stretches that CONTAIN the events -- "
    "and never the gaps between them. estimate_open_pore_tau measures the gaps itself. Fix: start the list "
    "[0, a0 - pad, b0 + pad, a1 - pad, ...] and end it [..., a_last - pad, b_last + pad, n]."))
def test_cusum_open_pore_autocorr_time_measures_the_gaps_between_events():
    from nanoscope.analysis.cusum import open_pore_autocorr_time

    rng = np.random.default_rng(1)
    r = rng.normal(size=300_000)                                         # white: tau is 1
    starts = 2_000 + 3_000 * np.arange(90)
    ends = starts + 1_000
    for s, e in zip(starts, ends):
        r[s:e] += 40.0
    assert open_pore_autocorr_time(r, starts, ends, n_scale=64, pad=64) == pytest.approx(1.0, abs=0.1)


# ================================================================================================
# A fuzz: hostile and plausible input, numpy warnings turned into errors
# ================================================================================================
def test_fuzz_hostile_input_never_raises():
    """Garbage in every argument of all three functions: an answer, never an exception, never a numpy warning."""
    import warnings

    rng = np.random.default_rng(0)
    weird = [0, 1, -1, 3, 8, 64, 10 ** 6, 10 ** 12, -10 ** 12, 0.5, 1e-9, 1e9, float("nan"), float("inf"), -float("inf"), None, "x", True]
    pick = lambda: weird[int(rng.integers(len(weird)))]                         # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(1500):
            n = int(rng.choice([0, 1, 5, 50, 200, 1000, 5000]))
            x = rng.normal(size=n)
            kind = int(rng.integers(5))
            if kind == 1 and n > 20:
                x[n // 3: 2 * n // 3] += rng.uniform(-30, 30)
            if kind == 2 and n:
                x[int(rng.integers(n))] = float(rng.choice([np.nan, np.inf, -np.inf]))
            if kind == 3:
                x = x.astype(np.float32)
            if kind == 4:
                x = (x * 1000).astype(np.int16)
            s = int(rng.integers(-10, max(n, 1) + 10))
            e = s + int(rng.integers(-5, max(n, 1) + 10))
            kw = dict(polarity=int(rng.choice([1, -1, 0, 2])), noise_rms=float(rng.choice([1.0, 0.0, 1e-12, 1e12])),
                      tau=float(rng.choice([1.0, 0.5, 30.0, 1e9])), lo=int(rng.integers(-5, max(n, 1) + 5)),
                      hi=int(rng.integers(-5, max(n, 1) + 5)), px_samples=[None, 1, 10.5, 1e6, 0][int(rng.integers(5))],
                      rise_s=[None, 1e-4, 1.0, 0.0][int(rng.integers(4))], filter_rise_s=[None, 1e-5, 1e3][int(rng.integers(3))],
                      min_len=int(rng.choice([1, 2, 3, 100])),
                      baseline=[None, 0.0, 5.0, rng.normal(size=n), rng.normal(size=n + 1)][int(rng.integers(5))])
            if rng.random() < 0.15:
                kw[list(kw)[int(rng.integers(len(kw)))]] = pick()
            fs = float(rng.choice([1.0, 1e3, 5e4, 2.5e5, 0.0, -1.0]))
            sx, ex = (s, e) if rng.random() < 0.9 else (pick(), pick())
            f = ef.fit_edges(x, fs, sx, ex, **kw)
            assert isinstance(f.start.position, int) and isinstance(f.end.position, int)
            assert all(edge.fitted or edge.reason for edge in (f.start, f.end))
            json.loads(f.to_json())
            ckw = {k: kw[k] for k in ("polarity", "noise_rms", "baseline", "lo", "hi", "tau", "rise_s", "min_len", "filter_rise_s")}
            ckw.update(view_lo=int(rng.integers(-10, max(n, 1))), view_hi=int(rng.integers(0, max(n, 1) + 10)),
                       px_samples=kw["px_samples"] if kw["px_samples"] is not None else 1)
            ptr = float(rng.integers(-5, max(n, 1) + 5)) if rng.random() < 0.9 else pick()
            c = ef.catch_event(x, fs, ptr, **ckw)
            assert isinstance(c, ef.Catch) and (c.ok or c.reason)
            spans = [None, [], [(s, e)], [(pick(), pick())]][int(rng.integers(4))]
            t = ef.estimate_open_pore_tau(x, spans, noise_rms=kw["noise_rms"], fs=fs, L=int(rng.choice([1, 3, 64, 10 ** 6])),
                                          filter_autocorr=[None, 2.0, float("nan")][int(rng.integers(3))])
            assert np.isfinite(t) and t >= 1.0


def test_fuzz_plausible_traces_keep_the_invariants():
    """Fits and catches on random events and random placements: positions inside the neighbours and the
    view, the edges in order, an unfitted edge exactly as placed, and the polarity twin identical."""
    import warnings

    rng = np.random.default_rng(2)
    complete = oks = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(1500):
            n = int(rng.choice([300, 1500, 6000]))
            fs = float(rng.choice([2e4, 5e4, 2.5e5]))
            x = rng.normal(size=n) * float(rng.choice([0.5, 1.0, 4.0])) + float(rng.choice([0.0, -1000.0]))
            pol = int(rng.choice([1, -1]))
            events = []
            for _k in range(int(rng.integers(0, 4))):
                length = int(rng.integers(3, max(4, n // 3)))
                start = int(rng.integers(0, n - length))
                x[start:start + length] += float(rng.uniform(1, 60)) * pol
                events.append((start, start + length))
            base = np.full(n, float(np.median(x))) if rng.random() < 0.5 else np.median(x) + np.cumsum(rng.normal(size=n)) * 0.02
            sig = float(np.std(x - base)) or 1.0
            if events and rng.random() < 0.85:
                es, ee = events[int(rng.integers(len(events)))]
                jit = max(2, int(rng.choice([0.02, 0.1, 0.3]) * (ee - es)))
                s0, e0 = es + int(rng.integers(-jit, jit + 1)), ee + int(rng.integers(-jit, jit + 1))
            else:
                s0 = int(rng.integers(-2, n))
                e0 = s0 + int(rng.integers(-3, max(n // 2, 4)))
            px = float(rng.choice([1, 3, 10, 100, 700, 5000]))
            lo = int(rng.integers(0, max(1, s0))) if s0 > 0 and rng.random() < 0.5 else 0
            hi = int(rng.integers(min(n, e0 + 1), n + 1)) if e0 < n and rng.random() < 0.5 else n
            kw = dict(polarity=pol, noise_rms=sig, tau=float(rng.choice([1.0, 3.0, 20.0])), lo=lo, hi=hi,
                      baseline=base if rng.random() < 0.7 else None, px_samples=px,
                      rise_s=[None, 2 / fs, 1e-3, 5e-3][int(rng.integers(4))], filter_rise_s=[None, 3.4e-5][int(rng.integers(2))])
            f = ef.fit_edges(x, fs, s0, e0, **kw)
            for edge, placed in ((f.start, s0), (f.end, e0)):
                if edge.fitted:
                    assert max(0, lo) <= edge.position <= min(n, hi)
                else:
                    assert edge.position == placed and edge.reason
            if f.complete:
                complete += 1
                assert f.start.position < f.end.position
            twin = ef.fit_edges(-x, fs, s0, e0, **dict(kw, polarity=-pol, baseline=None if kw["baseline"] is None else -kw["baseline"]))
            assert (twin.start, twin.end) == (f.start, f.end)
            p = float(rng.integers(events[0][0], events[0][1])) if events and rng.random() < 0.8 else float(rng.integers(0, n))
            v0 = int(p - rng.uniform(0.1, 0.9) * px * 1200)
            c = ef.catch_event(x, fs, p, polarity=pol, noise_rms=sig, baseline=kw["baseline"], lo=lo if lo < p else 0,
                               hi=hi if hi > p else n, view_lo=v0, view_hi=v0 + int(px * 1200), px_samples=px, tau=kw["tau"],
                               rise_s=kw["rise_s"], filter_rise_s=kw["filter_rise_s"])
            if c.ok:
                oks += 1
                assert 0 <= c.foot_start <= c.start < c.end <= c.foot_end <= n
                assert max(v0, 0, lo if lo < p else 0) <= c.foot_start and c.foot_end <= min(v0 + int(px * 1200), n, hi if hi > p else n)
                for edge, placed in ((c.fit.start, c.start), (c.fit.end, c.end)):
                    assert edge.fitted or edge.position == placed
    assert complete >= 30 and oks >= 60                      # the fuzz does reach the fitting code
