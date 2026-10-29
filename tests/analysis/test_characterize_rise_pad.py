"""The rise/fall search pad is ``max(64 samples, 1.28 ms)`` (design 5.6).

``characterize.rise_fall_over_set`` looks for the 10% and 90% points of an edge inside a pad
outside the event. The pad was a fixed 64 samples: 1.28 ms at the owner's 50 kHz, but 0.26 ms
at 250 kHz, shorter than the 10% point of an edge whose START was fitted mid-ramp. The new pad
is the same 64 samples at 50 kHz and below, so no row of a 50 kHz recording moves, and 1.28 ms
above it.

"Before" below is the same code with ``_edge_pad_samples`` patched back to a constant 64, so
the contrast is the one change and nothing else. Synthetic data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

from nanoscope.analysis import characterize as ch
from nanoscope.analysis.characterize import characterize, rise_fall_over_set


def _old_pad(monkeypatch) -> None:
    """The fixed 64-sample pad this change replaced."""
    monkeypatch.setattr(ch, "_edge_pad_samples", lambda fs: 64)


def _erf_ramp_event(n, s, e, depth, rise_samples):
    """dev (positive in the event) of an erf-edged box, mid-ramp at ``s`` and ``e``; the
    10-90% time of both edges is ``rise_samples``."""
    t = np.arange(n, dtype=np.float64)
    k = np.sqrt(2.0) * rise_samples / 2.563
    return depth * (0.5 * (1 + erf((t - s) / k)) - 0.5 * (1 + erf((t - e) / k)))


# ------------------------------------------------------------------------------------------------
# The pad itself
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fs", [1_000.0, 10_000.0, 25_000.0, 49_999.0, 50_000.0, 50_390.0])
def test_pad_is_the_old_64_samples_at_50khz_and_below(fs):
    assert ch._edge_pad_samples(fs) == 64


@pytest.mark.parametrize("fs, expect", [(60_000.0, 77), (100_000.0, 128), (250_000.0, 320), (1_000_000.0, 1280)])
def test_pad_is_1p28_ms_above_50khz(fs, expect):
    assert ch._edge_pad_samples(fs) == expect
    assert ch._edge_pad_samples(fs) / fs == pytest.approx(1.28e-3, abs=0.6 / fs)


@pytest.mark.parametrize("fs", [float("nan"), float("inf"), 0.0, -5.0, "not a rate", None])
def test_pad_keeps_the_old_value_for_a_rate_that_is_not_a_number(fs):
    assert ch._edge_pad_samples(fs) == 64


def test_pad_at_exactly_50khz_is_not_65():
    """1.28e-3 * 50000.0 is not exactly 64.0 in every float environment, and a ceiling would
    turn the owner's rate into 65 samples and move every one of their rows."""
    assert ch._edge_pad_samples(50_000.0) == 64
    assert ch._edge_pad_samples(np.float64(50_000.0)) == 64


# ------------------------------------------------------------------------------------------------
# The boundary is exactly 64 samples at 50 kHz and below
# ------------------------------------------------------------------------------------------------
def _linear_edges_event(n, s, e, depth, up, down):
    """dev of a box whose 0-100 % ramps are ``up`` samples before ``s`` and ``down`` after ``e``."""
    dev = np.zeros(n)
    dev[s - up:s] = depth * np.arange(up) / up
    dev[s:e] = depth
    dev[e:e + down] = depth * (1 - np.arange(down) / down)
    return dev


@pytest.mark.parametrize("fs", [10_000.0, 25_000.0, 50_000.0])
def test_the_10_percent_point_one_sample_beyond_64_is_still_out_of_reach_at_50khz_and_below(fs, monkeypatch):
    """A ramp whose 10 % point lies 65 samples before the start (and 64 after the end) is outside
    a 64-sample pad and inside a 65-sample one. At 50 kHz and below it must stay outside, i.e.
    NaN exactly as before: the identity is exact at the boundary, not just on easy cases."""
    n, s, e, depth = 4_000, 1_000, 2_000, 100.0
    cur = _linear_edges_event(n, s, e, depth, up=72, down=71)
    base = np.zeros(n)
    args = (cur, base, 1, np.array([s]), np.array([e]), np.array([depth]), fs)
    rise, fall = rise_fall_over_set(*args)
    assert np.isnan(rise[0]) and np.isnan(fall[0])
    monkeypatch.setattr(ch, "_edge_pad_samples", lambda f: 65)
    rise65, fall65 = rise_fall_over_set(*args)
    assert np.isfinite(rise65[0]) and np.isfinite(fall65[0])         # the probe does see a one-sample change
    monkeypatch.undo()
    rise250, fall250 = rise_fall_over_set(cur, base, 1, np.array([s]), np.array([e]), np.array([depth]), 250_000.0)
    assert np.isfinite(rise250[0]) and np.isfinite(fall250[0])       # and the wider pad reaches it


# ------------------------------------------------------------------------------------------------
# Bit-identical at 50 kHz and below
# ------------------------------------------------------------------------------------------------
def _soft_events_with_neighbours(fs, seed=3):
    """Events with soft (1 ms) and sharp edges, several within 64 samples of a neighbour so the
    neighbour clip is exercised, and one touching the start of the record."""
    rng = np.random.default_rng(seed)
    n = int(0.4 * fs)
    ms = fs / 1e3
    base = np.full(n, -1000.0)
    dev = np.zeros(n)
    spans = []
    pos = 5
    for i, (length, gap, depth, rise) in enumerate([
        (3.0, 0.6, 80.0, 1.0), (2.0, 0.4, 60.0, 0.1), (1.0, 0.05, 90.0, 0.05),
        (5.0, 4.0, 40.0, 1.0), (2.0, 1.0, 120.0, 0.3), (6.0, 0.02, 70.0, 0.5),
        (2.0, 5.0, 50.0, 0.8),
    ]):
        s = int(pos)
        e = s + int(length * ms)
        spans.append((s, e))
        lo = max(0, s - int(10 * ms))
        hi = min(n, e + int(10 * ms))
        t = np.arange(lo, hi, dtype=np.float64)
        k = np.sqrt(2.0) * max(rise * ms, 0.5) / 2.563
        dev[lo:hi] += depth * (0.5 * (1 + erf((t - s) / k)) - 0.5 * (1 + erf((t - e) / k)))
        pos = e + gap * ms
    cur = base + dev + rng.normal(0.0, 2.0, n)           # polarity +1: dev = cur - base
    starts = np.array([a for a, _ in spans], dtype=np.int64)
    ends = np.array([b for _, b in spans], dtype=np.int64)
    depth = np.array([np.median((cur - base)[a:b]) for a, b in spans])
    return cur, base, starts, ends, depth


@pytest.mark.parametrize("fs", [10_000.0, 25_000.0, 50_000.0])
def test_rise_and_fall_are_bit_identical_to_the_fixed_pad_at_50khz_and_below(fs, monkeypatch):
    cur, base, starts, ends, depth = _soft_events_with_neighbours(fs)
    rise_new, fall_new = rise_fall_over_set(cur, base, 1, starts, ends, depth, fs)
    _old_pad(monkeypatch)
    rise_old, fall_old = rise_fall_over_set(cur, base, 1, starts, ends, depth, fs)
    assert np.isfinite(rise_new).sum() >= 2 and np.isfinite(fall_new).sum() >= 2     # the case measures something
    assert np.array_equal(rise_new, rise_old, equal_nan=True)
    assert np.array_equal(fall_new, fall_old, equal_nan=True)


def test_whole_characterized_table_is_unchanged_at_50khz(monkeypatch):
    """Not just the two arrays: every column of the row table, at the owner's rate."""
    fs = 50_000.0
    cur, base, starts, ends, _depth = _soft_events_with_neighbours(fs)
    new = characterize(cur.astype(np.float32), base, starts, ends, fs, 1, noise_rms=2.0)
    _old_pad(monkeypatch)
    old = characterize(cur.astype(np.float32), base, starts, ends, fs, 1, noise_rms=2.0)
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def test_rates_above_50khz_do_change_where_the_old_pad_was_too_short():
    """The contrast that makes the identity test above meaningful: at 250 kHz the same kind of
    case does move."""
    fs = 250_000.0
    cur, base, starts, ends, depth = _soft_events_with_neighbours(fs)
    new = rise_fall_over_set(cur, base, 1, starts, ends, depth, fs)
    old = None
    mp = pytest.MonkeyPatch()
    try:
        _old_pad(mp)
        old = rise_fall_over_set(cur, base, 1, starts, ends, depth, fs)
    finally:
        mp.undo()
    assert not (np.array_equal(new[0], old[0], equal_nan=True) and np.array_equal(new[1], old[1], equal_nan=True))


# ------------------------------------------------------------------------------------------------
# A fitted 0.6 ms edge at 250 kHz
# ------------------------------------------------------------------------------------------------
FS_HI = 250_000.0
RISE_S = 0.6e-3


def _slow_event(noise_pA, seed=0):
    """A 10 ms, 100 pA event with 0.6 ms edges at 250 kHz, polarity -1 (current falls)."""
    n = int(0.05 * FS_HI)
    s, e = 5_000, 7_500                                  # the MID-RAMP samples, as a fit puts them
    dev = _erf_ramp_event(n, s, e, 100.0, RISE_S * FS_HI)
    base = np.full(n, 1000.0)
    rng = np.random.default_rng(seed)
    cur = base - dev + (rng.normal(0.0, noise_pA, n) if noise_pA else 0.0)
    return cur, base, s, e


def test_slow_edge_at_250khz_was_nan_and_is_finite_now(monkeypatch):
    cur, base, s, e = _slow_event(noise_pA=0.0)
    starts, ends = np.array([s]), np.array([e])
    depth = np.array([100.0])
    _old_pad(monkeypatch)
    rise_old, fall_old = rise_fall_over_set(cur, base, -1, starts, ends, depth, FS_HI)
    assert np.isnan(rise_old[0]) and np.isnan(fall_old[0])        # before: 64 samples cannot hold the 10 % point
    monkeypatch.undo()
    rise_new, fall_new = rise_fall_over_set(cur, base, -1, starts, ends, depth, FS_HI)
    assert np.isfinite(rise_new[0]) and np.isfinite(fall_new[0])  # after
    assert rise_new[0] == pytest.approx(RISE_S, rel=0.03)
    assert fall_new[0] == pytest.approx(RISE_S, rel=0.03)


def test_fitted_0p6ms_edge_at_250khz_has_a_finite_rise_s_end_to_end(monkeypatch):
    """The whole path of the design's test: place the bounds by hand, let the step fit put them
    on the steps (mid-ramp), characterize."""
    from nanoscope.analysis.edge_fit import fit_edges

    cur, base, s, e = _slow_event(noise_pA=2.0, seed=4)
    fit = fit_edges(cur, FS_HI, s + 40, e - 30, polarity=-1, noise_rms=2.0, tau=1.0, lo=0, hi=cur.size,
                    baseline=base, px_samples=10, rise_s=RISE_S, filter_rise_s=0.034e-3)
    assert fit.complete
    assert abs(fit.start.position - s) <= 40 and abs(fit.end.position - e) <= 40     # mid-ramp, within half a rise
    starts, ends = np.array([fit.start.position]), np.array([fit.end.position])
    after = characterize(cur.astype(np.float32), base, starts, ends, FS_HI, -1, noise_rms=2.0)
    assert np.isfinite(after["rise_s"].iloc[0]) and np.isfinite(after["fall_s"].iloc[0])
    assert after["rise_s"].iloc[0] == pytest.approx(RISE_S, rel=0.15)
    assert after["fall_s"].iloc[0] == pytest.approx(RISE_S, rel=0.15)
    _old_pad(monkeypatch)
    before = characterize(cur.astype(np.float32), base, starts, ends, FS_HI, -1, noise_rms=2.0)
    # before: NaN, or a value taken from a noise dip at the window's edge, never the edge's own rise
    r_before = before["rise_s"].iloc[0]
    assert np.isnan(r_before) or abs(r_before - RISE_S) > 0.03 * RISE_S


def test_slow_edge_at_50khz_is_unchanged_by_the_new_pad(monkeypatch):
    """The owner's rate: a 1 ms edge (50 samples) fitted mid-ramp keeps a finite rise_s, the
    same one as before."""
    fs = 50_000.0
    n = int(0.2 * fs)
    s, e = 2_000, 5_000
    dev = _erf_ramp_event(n, s, e, 100.0, 1e-3 * fs)
    base = np.full(n, 1000.0)
    cur = base - dev
    starts, ends, depth = np.array([s]), np.array([e]), np.array([100.0])
    new = rise_fall_over_set(cur, base, -1, starts, ends, depth, fs)
    _old_pad(monkeypatch)
    old = rise_fall_over_set(cur, base, -1, starts, ends, depth, fs)
    assert np.isfinite(new[0][0]) and new[0][0] == pytest.approx(1e-3, rel=0.03)
    assert np.array_equal(new[0], old[0]) and np.array_equal(new[1], old[1])


# ------------------------------------------------------------------------------------------------
# The neighbour clip survives a wider pad
# ------------------------------------------------------------------------------------------------
def test_a_wider_pad_is_still_clipped_to_the_neighbours():
    """Two identical 0.2 ms-edged events 100 samples apart at 250 kHz: the new pad (320) reaches
    past the neighbour, so each window must stop at it. The second event's rise is its own, the
    same as the first's, not the first event's tail."""
    fs = FS_HI
    n = 6_000
    r = 0.2e-3 * fs
    dev = _erf_ramp_event(n, 1_000, 2_000, 100.0, r) + _erf_ramp_event(n, 2_100, 3_100, 100.0, r)
    base = np.full(n, 1000.0)
    cur = base - dev
    starts, ends = np.array([1_000, 2_100]), np.array([2_000, 3_100])
    depth = np.array([100.0, 100.0])
    rise, fall = rise_fall_over_set(cur, base, -1, starts, ends, depth, fs)
    assert ch._edge_pad_samples(fs) == 320 > 100
    assert np.all(np.isfinite(rise)) and np.all(np.isfinite(fall))
    assert rise[1] == pytest.approx(rise[0], rel=0.02)
    assert fall[0] == pytest.approx(fall[1], rel=0.02)
    assert rise[0] == pytest.approx(0.2e-3, rel=0.05)
