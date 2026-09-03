"""Tests for analysis.event_shape: ground-truth waveforms in, known numbers out.

Three signature translocation quantities, each checked against an analytic truth:

* :func:`charge_deficit` -- a square blockade of depth ``D`` and width ``W`` has
  ECD ``D*W``; a triangular (ramp) blockade has ``D*W/2``; an upward enhancement
  flips the sign. The ``gain`` conversion scales the result.
* :func:`density_2d` -- known points land in known bins, log binning bins in log
  space and drops non-positive values, and the edges are deterministic.
* :func:`event_ensemble` -- identical events average to the single event shape;
  start vs centre alignment differ on unequal-length events; events off the trace
  edge are skipped (and thus counted out of the kept rows).

None of this touches the frozen EventSet schema: the EventSets are hand-built at
the exact SCHEMA and the quantities are computed on demand from ``(signal, fs)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.event_shape import (
    PA_S_TO_FC,
    charge_deficit,
    density_2d,
    event_ensemble,
)
from nanoscope.analysis.events import SCHEMA, EventSet, empty_df

FS = 100_000.0
BASE = 1000.0


# ================================================================================================
# builders
# ================================================================================================

def _eventset(specs: list[dict], *, recording: str = "rec") -> EventSet:
    """An EventSet at the exact SCHEMA from ``[{start,end,baseline,accepted}, ...]``.

    Only ``start_idx``/``end_idx``/``baseline_pA``/``accepted`` are load-bearing for
    event_shape; the rest are filled with plausible placeholders so the frame is a
    valid 20-column SCHEMA frame.
    """
    df = empty_df()
    for i, s in enumerate(specs):
        start, end = int(s["start"]), int(s["end"])
        df.loc[i] = {
            "event_id": i, "recording": recording, "pore_id": "P",
            "start_idx": start, "end_idx": end,
            "t_start_s": start / FS, "dwell_s": (end - start) / FS,
            "baseline_pA": float(s.get("baseline", BASE)),
            "depth_pA": float(s.get("depth", 50.0)),
            "frac_depth": 0.05, "ecd_pA_s": 0.0,
            "max_dev_pA": 60.0, "mean_dev_pA": 50.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 10.0,
            "n_levels": 1, "filter_limited": False,
            "accepted": bool(s.get("accepted", True)), "flag": "",
        }
    df = df.astype({c: d for c, (d, _, _) in SCHEMA.items()})
    return EventSet(df, {"analyzed_duration_s": 1.0, "recording": recording})


def _flat_signal(n: int, baseline: float = BASE) -> np.ndarray:
    return np.full(n, float(baseline), dtype=np.float64)


# ================================================================================================
# (a) charge_deficit -- square == D*W, triangle == D*W/2, sign, gain, robustness
# ================================================================================================

def test_square_blockade_ecd_is_depth_times_width():
    D, s, e = 120.0, 1_000, 1_500          # width W = 500 samples
    W = (e - s) / FS
    sig = _flat_signal(4_000)
    sig[s:e] = BASE - D                      # downward blockade
    es = _eventset([{"start": s, "end": e, "baseline": BASE}])

    ecd = charge_deficit(sig, FS, es)
    assert ecd.shape == (1,)
    assert ecd[0] == pytest.approx(D * W, rel=1e-9)       # 120 * 5e-3 = 0.6 pA*s
    assert ecd[0] > 0                                      # a downward blockade is positive


def test_triangular_blockade_ecd_is_half_depth_times_width():
    D, s, e = 200.0, 2_000, 2_600
    N = e - s
    W = N / FS
    sig = _flat_signal(6_000)
    sig[s:e] = BASE - np.linspace(0.0, D, N)  # ramp dev 0->D : a right-triangle of area D*N/2
    es = _eventset([{"start": s, "end": e, "baseline": BASE}])

    ecd = charge_deficit(sig, FS, es)
    assert ecd[0] == pytest.approx(D * W / 2.0, rel=1e-9)


def test_sign_is_correct_for_up_vs_down_events():
    D = 90.0
    sig = _flat_signal(8_000)
    sig[1_000:1_500] = BASE - D               # down (blockade)
    sig[3_000:3_500] = BASE + D               # up (enhancement), same |dev| and width
    es = _eventset([
        {"start": 1_000, "end": 1_500, "baseline": BASE},
        {"start": 3_000, "end": 3_500, "baseline": BASE},
    ])
    ecd = charge_deficit(sig, FS, es)
    assert ecd[0] > 0 and ecd[1] < 0
    assert ecd[0] == pytest.approx(-ecd[1], rel=1e-9)      # equal magnitude, opposite sign


def test_gain_scales_to_femtocoulombs():
    D, s, e = 100.0, 500, 1_000
    W = (e - s) / FS
    sig = _flat_signal(2_000)
    sig[s:e] = BASE - D
    es = _eventset([{"start": s, "end": e, "baseline": BASE}])

    pa_s = charge_deficit(sig, FS, es)
    fc = charge_deficit(sig, FS, es, gain=PA_S_TO_FC)
    assert pa_s[0] == pytest.approx(D * W, rel=1e-9)
    assert fc[0] == pytest.approx(D * W * 1e3, rel=1e-9)   # 1 pA*s = 1e3 fC


def test_value_per_event_includes_rejected_rows():
    """One value per row of the passed set -- charge_deficit does not filter curation."""
    sig = _flat_signal(4_000)
    sig[500:700] = BASE - 40.0
    sig[1_500:1_700] = BASE - 40.0
    es = _eventset([
        {"start": 500, "end": 700, "accepted": True},
        {"start": 1_500, "end": 1_700, "accepted": False},
    ])
    ecd = charge_deficit(sig, FS, es)
    assert ecd.size == 2
    assert np.all(np.isfinite(ecd))


def test_empty_eventset_returns_empty_array():
    ecd = charge_deficit(_flat_signal(100), FS, _eventset([]))
    assert isinstance(ecd, np.ndarray)
    assert ecd.size == 0


def test_event_off_the_edge_is_nan_partial_window_is_finite():
    n = 2_000
    sig = _flat_signal(n)
    sig[1_800:2_000] = BASE - 50.0
    es = _eventset([
        {"start": 1_900, "end": 2_100, "baseline": BASE},   # partly past the end -> clipped
        {"start": 5_000, "end": 5_200, "baseline": BASE},   # entirely past the end -> NaN
    ])
    ecd = charge_deficit(sig, FS, es)
    assert np.isfinite(ecd[0])                               # integrated over its in-bounds part
    assert np.isnan(ecd[1])


def test_zero_fs_is_rejected():
    with pytest.raises(ValueError):
        charge_deficit(_flat_signal(100), 0.0, _eventset([{"start": 1, "end": 5}]))


# ================================================================================================
# (b) density_2d -- bin placement, log binning, deterministic edges
# ================================================================================================

def test_known_points_land_in_the_right_bins():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    counts, xe, ye = density_2d(x, y, xbins=2, ybins=2)
    assert counts.shape == (2, 2)
    # xedges [1, 2.5, 4], yedges [10, 25, 40]: (1,10),(2,20) -> (0,0); (3,30),(4,40) -> (1,1)
    np.testing.assert_array_equal(xe, np.array([1.0, 2.5, 4.0]))
    np.testing.assert_array_equal(ye, np.array([10.0, 25.0, 40.0]))
    assert counts[0, 0] == 2
    assert counts[1, 1] == 2
    assert counts[0, 1] == 0 and counts[1, 0] == 0
    assert counts.sum() == 4


def test_log_binning_bins_in_log_space():
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.zeros(4)
    counts, xe, ye = density_2d(x, y, xbins=3, ybins=1, xlog=True)
    # geometric edges [1, 10, 100, 1000]; bins [1,10),[10,100),[100,1000]
    np.testing.assert_allclose(xe, np.array([1.0, 10.0, 100.0, 1000.0]), rtol=1e-12)
    x_marginal = counts.sum(axis=1)
    np.testing.assert_array_equal(x_marginal, np.array([1.0, 1.0, 2.0]))   # 1000 lands in last bin


def test_log_binning_drops_non_positive():
    x = np.array([-5.0, 0.0, 1.0, 10.0, 100.0])
    y = np.ones(5)
    counts, xe, ye = density_2d(x, y, xbins=2, ybins=1, xlog=True)
    assert counts.sum() == 3                                 # -5 and 0 dropped for the log axis


def test_edges_are_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    c1, xe1, ye1 = density_2d(x, y, xbins=7, ybins=5)
    c2, xe2, ye2 = density_2d(x, y, xbins=7, ybins=5)
    np.testing.assert_array_equal(xe1, xe2)
    np.testing.assert_array_equal(ye1, ye2)
    np.testing.assert_array_equal(c1, c2)
    assert xe1.size == 8 and ye1.size == 6


def test_non_finite_points_are_dropped():
    x = np.array([1.0, 2.0, np.nan, 3.0])
    y = np.array([1.0, np.inf, 2.0, 3.0])
    counts, _, _ = density_2d(x, y, xbins=3, ybins=3)
    assert counts.sum() == 2                                 # only (1,1) and (3,3) survive


# ================================================================================================
# (c) event_ensemble -- mean of identical events, alignment, edge skipping
# ================================================================================================

def _ramp_event(sig: np.ndarray, s: int, N: int, depth: float, baseline: float = BASE) -> None:
    """Write a downward ramp blockade (dev 0->depth) into ``sig[s:s+N]``."""
    sig[s:s + N] = baseline - np.linspace(0.0, depth, N)


def test_identical_events_mean_equals_single_event_shape():
    n, N, D, npts = 20_000, 400, 150.0, 64
    sig = _flat_signal(n)
    positions = [2_000, 8_000, 14_000]
    for p in positions:
        _ramp_event(sig, p, N, D)
    es = _eventset([{"start": p, "end": p + N, "baseline": BASE} for p in positions])

    T, matrix, mean, median = event_ensemble(
        sig, FS, es, n_points=npts, pad_frac=0.0, align="start", normalize_time=True)

    assert matrix.shape == (len(positions), npts)
    assert T.shape == (npts,) and mean.shape == (npts,) and median.shape == (npts,)
    # every row is the same shape -> mean and median ARE that shape
    single = np.interp(
        np.linspace(positions[0], positions[0] + N, npts, endpoint=False),
        np.arange(n, dtype=float), sig)
    np.testing.assert_allclose(mean, single, rtol=1e-9)
    np.testing.assert_allclose(median, single, rtol=1e-9)
    for r in range(matrix.shape[0]):
        np.testing.assert_allclose(matrix[r], single, rtol=1e-9)


def test_alignment_start_vs_center_differ_on_unequal_events():
    n, npts = 20_000, 80
    sig = _flat_signal(n)
    # a long event and a short one; the long one's length sets the common window
    _ramp_event(sig, 5_000, 400, 120.0)      # long: N = 400
    sig[10_000:10_100] = BASE - 120.0        # short: N = 100 (square, so its position shows)
    es = _eventset([
        {"start": 5_000, "end": 5_400, "baseline": BASE},
        {"start": 10_000, "end": 10_100, "baseline": BASE},
    ])

    _, mat_start, _, _ = event_ensemble(
        sig, FS, es, n_points=npts, align="start", normalize_time=False)
    _, mat_center, _, _ = event_ensemble(
        sig, FS, es, n_points=npts, align="center", normalize_time=False)

    assert mat_start.shape == (2, npts) and mat_center.shape == (2, npts)
    # the short event sits at a different place in the common window under each alignment
    assert not np.allclose(mat_start, mat_center)


def test_only_accepted_events_are_ensembled():
    n, N, npts = 20_000, 300, 48
    sig = _flat_signal(n)
    for p in (3_000, 9_000, 15_000):
        _ramp_event(sig, p, N, 100.0)
    es = _eventset([
        {"start": 3_000, "end": 3_000 + N, "accepted": True},
        {"start": 9_000, "end": 9_000 + N, "accepted": False},   # rejected -> excluded
        {"start": 15_000, "end": 15_000 + N, "accepted": True},
    ])
    _, matrix, _, _ = event_ensemble(sig, FS, es, n_points=npts, normalize_time=True)
    assert matrix.shape[0] == 2                                  # the rejected event is not used


def test_edge_events_are_skipped_and_counted():
    n, N, npts = 20_000, 400, 50
    sig = _flat_signal(n)
    positions = [0, 8_000, 15_000]           # the s=0 event cannot take left padding
    for p in positions:
        _ramp_event(sig, p, N, 100.0)
    es = _eventset([{"start": p, "end": p + N, "baseline": BASE} for p in positions])

    _, matrix, mean, _ = event_ensemble(
        sig, FS, es, n_points=npts, pad_frac=0.5, align="start", normalize_time=True)

    n_accepted = len(es.accepted.df)
    assert n_accepted == 3
    assert matrix.shape[0] == 2                                  # the s=0 event was skipped
    skipped = n_accepted - matrix.shape[0]
    assert skipped == 1
    assert np.all(np.isfinite(mean))


def test_empty_ensemble_is_all_nan():
    T, matrix, mean, median = event_ensemble(
        _flat_signal(1_000), FS, _eventset([]), n_points=32)
    assert matrix.shape == (0, 32)
    assert T.shape == (32,)
    assert np.all(np.isnan(mean)) and np.all(np.isnan(median))


def test_normalized_axis_spans_padded_unit_interval():
    n, N, npts = 10_000, 500, 40
    sig = _flat_signal(n)
    _ramp_event(sig, 4_000, N, 80.0)
    es = _eventset([{"start": 4_000, "end": 4_500, "baseline": BASE}])
    T, _, _, _ = event_ensemble(
        sig, FS, es, n_points=npts, pad_frac=0.25, normalize_time=True)
    assert T[0] == pytest.approx(-0.25)                         # padding before the event start
    assert T[-1] < 1.25 and T[-1] > 1.0                        # up to (but excluding) 1 + pad
