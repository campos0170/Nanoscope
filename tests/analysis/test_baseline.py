"""Tests for nanoscope.analysis.baseline (signal-first port).

The load-bearing claims:
  1. block_median tracks a known drift to within noise.
  2. a too-short window demonstrably EATS events, and baseline_intrusion() says so.
  3. block_median is fast enough on 5M samples to sit behind a GUI.

Ported from the standalone ``tests/test_baseline.py``. The old ``Recording``
fixture is gone: the entry points now take explicit ``(current, fs, polarity)``
numpy args. ``_make`` returns ``(current, fs, truth, starts, ends)`` and every
``rec.*`` becomes the matching primitive (``rec.current`` -> ``current``,
``rec.fs`` -> ``fs``, ``rec.n`` -> ``current.size``, ``rec.polarity`` -> ``POL``).
Every numeric assertion is preserved unchanged.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

from nanoscope.analysis.baseline import (
    BaselineWarning,
    baseline_intrusion,
    derive_polarity,
    estimate_baseline,
    open_pore_current,
)

# Sign convention: events push current toward +polarity. I0 = -1000 pA and a
# blockade reduces |I| -> current rises toward -800 -> deviations are positive ->
# polarity = +1. This is the user's actual regime (negative voltages).
I0 = -1000.0
POL = 1


def _drift(t: np.ndarray, amp: float = 200.0, f: float = 0.5) -> np.ndarray:
    """Known sinusoidal drift, the thing the baseline must follow."""
    return amp * np.sin(2 * np.pi * f * t)


def _make(
    *,
    fs: float = 50_000.0,
    duration_s: float = 10.0,
    noise_pA: float = 15.0,
    n_events: int = 0,
    dwell_s: float = 2e-4,
    depth_frac: float = 0.3,
    drift_amp: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """(current, fs, true_baseline, starts, ends) -- square events on a drifting trace."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    truth = I0 + _drift(t, amp=drift_amp)
    x = truth + rng.normal(0, noise_pA, n)

    w = int(round(dwell_s * fs))
    if n_events:
        gap = n // n_events
        starts = (np.arange(n_events) * gap + gap // 3).astype(np.int64)
        ends = starts + w
        assert ends[-1] < n
        for s, e in zip(starts, ends):
            x[s:e] += POL * depth_frac * abs(I0)
    else:
        starts = np.empty(0, dtype=np.int64)
        ends = np.empty(0, dtype=np.int64)

    return x.astype(np.float32), fs, truth, starts, ends


# ---------------------------------------------------------------------------
# 0. polarity is measured, never assumed
# ---------------------------------------------------------------------------


def test_derive_polarity_measures_up_excursions():
    """At negative bias a blockade is an upward excursion -> polarity +1."""
    current, _, _, _, _ = _make(n_events=100, dwell_s=2e-4, depth_frac=0.3)
    assert derive_polarity(current) == 1


def test_derive_polarity_measures_down_excursions():
    """Flip the trace and the excursions point the other way -> polarity -1."""
    current, _, _, _, _ = _make(n_events=100, dwell_s=2e-4, depth_frac=0.3)
    assert derive_polarity(-current) == -1


def test_estimate_baseline_derives_polarity_when_none():
    """polarity=None must match the measured polarity, for a polarity-sensitive method."""
    current, fs, truth, _, _ = _make(n_events=200, dwell_s=3e-3, depth_frac=0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BaselineWarning)
        auto = estimate_baseline(current, fs, method="percentile", polarity=None, min_dwell_s=3e-3)
        given = estimate_baseline(current, fs, method="percentile", polarity=POL, min_dwell_s=3e-3)
    assert np.array_equal(auto, given)


# ---------------------------------------------------------------------------
# 1. tracking a known drift
# ---------------------------------------------------------------------------


def test_block_median_tracks_sinusoidal_drift():
    current, fs, truth, _, _ = _make(drift_amp=200.0, noise_pA=15.0)
    b = estimate_baseline(current, fs, window_s=0.01, polarity=POL)

    assert b.shape == (current.size,)
    assert b.dtype == np.float64
    assert np.isfinite(b).all()

    err = b - truth
    assert np.sqrt(np.mean(err**2)) < 2.0, "block_median failed to track a 200 pA sine"
    assert np.abs(err).max() < 8.0


def test_constant_baseline_cannot_track_drift():
    """The contrast that proves the previous test measures something."""
    current, fs, truth, _, _ = _make(drift_amp=200.0)
    b = estimate_baseline(current, fs, method="constant", polarity=POL)
    assert np.allclose(b, b[0])
    assert np.sqrt(np.mean((b - truth) ** 2)) > 100.0


def test_asls_tracks_drift():
    current, fs, truth, _, _ = _make(drift_amp=200.0)
    b = estimate_baseline(current, fs, window_s=0.01, method="asls", polarity=POL)
    assert np.sqrt(np.mean((b - truth) ** 2)) < 8.0


def test_percentile_tracks_drift():
    current, fs, truth, _, _ = _make(drift_amp=200.0)
    b = estimate_baseline(current, fs, window_s=0.01, method="percentile", polarity=POL)
    # Biased toward the open-pore side by construction; it must still follow the
    # shape, so judge it after removing the constant offset.
    err = (b - truth) - np.mean(b - truth)
    assert np.sqrt(np.mean(err**2)) < 3.0


# ---------------------------------------------------------------------------
# 2. the central hazard: a short window eats events
# ---------------------------------------------------------------------------


def test_short_window_eats_events_and_intrusion_quantifies_it():
    current, fs, _, starts, ends = _make(n_events=100, dwell_s=2e-3, depth_frac=0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BaselineWarning)
        b_short = estimate_baseline(current, fs, window_s=2e-4, min_dwell_s=2e-3, polarity=POL)
    b_long = estimate_baseline(current, fs, window_s=0.05, min_dwell_s=2e-3, polarity=POL)

    short = baseline_intrusion(current, b_short, (starts, ends), fs, POL)
    long = baseline_intrusion(current, b_long, (starts, ends), fs, POL)

    assert short["intrusion_score"] > 0.8, "short window should swallow the events whole"
    assert short["severe"] and "SEVERE" in short["verdict"]
    assert long["intrusion_score"] < 0.05
    assert not long["severe"]
    assert short["intrusion_score"] > 10 * long["intrusion_score"]

    # The consequence, stated in the units that matter: measured depth collapses.
    dev_short = POL * (current - b_short)
    dev_long = POL * (current - b_long)
    depth_short = np.mean([dev_short[s:e].mean() for s, e in zip(starts, ends)])
    depth_long = np.mean([dev_long[s:e].mean() for s, e in zip(starts, ends)])
    true_depth = 0.3 * abs(I0)
    assert depth_long == pytest.approx(true_depth, rel=0.05)
    assert depth_short < 0.2 * true_depth


def test_short_window_warns():
    current, fs, _, _, _ = _make(n_events=50, dwell_s=2e-3)
    with pytest.warns(BaselineWarning, match="min_dwell_s"):
        estimate_baseline(current, fs, window_s=1e-3, min_dwell_s=2e-3, polarity=POL)


def test_adequate_window_does_not_warn():
    current, fs, _, _, _ = _make(n_events=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimate_baseline(current, fs, window_s=0.01, min_dwell_s=2e-5, polarity=POL)


def test_baseline_survives_normal_events():
    current, fs, truth, starts, ends = _make(n_events=200, dwell_s=2e-4, drift_amp=100.0)
    b = estimate_baseline(current, fs, window_s=0.01, polarity=POL)
    score = baseline_intrusion(current, b, (starts, ends), fs, POL)["intrusion_score"]
    assert score < 0.1
    assert np.sqrt(np.mean((b - truth) ** 2)) < 5.0


# ---------------------------------------------------------------------------
# 3. two_pass and dense events
# ---------------------------------------------------------------------------


def test_two_pass_reduces_intrusion():
    # 30% occupancy: enough to drag a block median toward the events.
    current, fs, truth, starts, ends = _make(n_events=200, dwell_s=1.5e-3, depth_frac=0.3)
    one = estimate_baseline(current, fs, window_s=0.01, min_dwell_s=1.5e-3 / 20, polarity=POL)
    two = estimate_baseline(
        current, fs, window_s=0.01, two_pass=True, min_dwell_s=1.5e-3 / 20, polarity=POL
    )

    s1 = baseline_intrusion(current, one, (starts, ends), fs, POL)["intrusion_score"]
    s2 = baseline_intrusion(current, two, (starts, ends), fs, POL)["intrusion_score"]
    assert s2 < s1
    err1 = np.sqrt(np.mean((one - truth) ** 2))
    err2 = np.sqrt(np.mean((two - truth) ** 2))
    assert err2 < err1


def test_percentile_and_asls_beat_median_when_occupancy_exceeds_half():
    # 60% occupancy (2000 events x 3ms in 10s): the block median now sits INSIDE
    # the events. This is exactly the regime 'percentile' and 'asls' exist for.
    current, fs, truth, s, e = _make(n_events=2000, dwell_s=3e-3, depth_frac=0.3)
    assert (e - s).sum() / current.size == pytest.approx(0.6, abs=0.02)  # sanity on the setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BaselineWarning)
        med = estimate_baseline(current, fs, window_s=0.01, min_dwell_s=3e-3, polarity=POL)
        pct = estimate_baseline(
            current, fs, window_s=0.01, method="percentile", min_dwell_s=3e-3, polarity=POL
        )
        als = estimate_baseline(
            current, fs, window_s=0.01, method="asls", min_dwell_s=3e-3, polarity=POL
        )
    e_med = np.abs(np.median(med - truth))
    e_pct = np.abs(np.median(pct - truth))
    e_als = np.abs(np.median(als - truth))
    assert e_med > 200.0, "sanity: the median should have been eaten"
    assert e_pct < 20.0
    assert e_als < 20.0
    assert e_pct < 0.1 * e_med
    assert e_als < 0.1 * e_med


def test_percentile_respects_polarity():
    """With polarity flipped, the open-pore envelope is on the other side."""
    current, fs, truth, _, _ = _make(n_events=200, dwell_s=3e-3, depth_frac=0.3)
    flipped = (-current).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BaselineWarning)
        b = estimate_baseline(
            flipped, fs, window_s=0.01, method="percentile", min_dwell_s=3e-3, polarity=-1
        )
    assert np.abs(np.median(b - (-truth))) < 50.0


# ---------------------------------------------------------------------------
# 4. performance -- the decision that makes the GUI usable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["block_median", "percentile"])
def test_fast_on_5M_samples(method):
    fs = 250_000.0
    n = 5_000_000
    rng = np.random.default_rng(1)
    t = np.arange(n) / fs
    x = I0 + _drift(t, amp=150.0, f=0.2) + rng.normal(0, 15.0, n)
    current = x.astype(np.float32)

    t0 = time.perf_counter()
    b = estimate_baseline(current, fs, window_s=0.01, method=method, polarity=POL)
    dt = time.perf_counter() - t0

    assert b.shape == (n,)
    assert dt < 2.0, f"{method} took {dt:.2f}s on 5M samples; the GUI would stall"


# ---------------------------------------------------------------------------
# 5. open_pore_current + validation
# ---------------------------------------------------------------------------


def test_open_pore_current():
    current, fs, _, _, _ = _make(n_events=100)
    b = estimate_baseline(current, fs, window_s=0.01, polarity=POL)
    assert open_pore_current(current, fs, b) == pytest.approx(I0, abs=2.0)
    assert open_pore_current(current, fs) == pytest.approx(I0, abs=2.0)
    # Signed, not a magnitude.
    assert open_pore_current(current, fs, b) < 0


def test_open_pore_current_rejects_wrong_length():
    current, fs, _, _, _ = _make()
    with pytest.raises(ValueError, match="length"):
        open_pore_current(current, fs, np.zeros(10))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window_s": 0}, "positive"),
        ({"window_s": -1.0}, "positive"),
        ({"window_s": 1e6}, "longer than the recording"),
        ({"method": "magic"}, "unknown method"),
        ({"percentile": 100.0}, "percentile"),
    ],
)
def test_rejects_nonsense(kwargs, match):
    current, fs, _, _, _ = _make(duration_s=1.0)
    with pytest.raises(ValueError, match=match):
        estimate_baseline(current, fs, polarity=POL, **kwargs)


def test_rejects_nonfinite_current():
    current, fs, _, _, _ = _make(duration_s=1.0)
    bad = current.copy()
    bad[100] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        estimate_baseline(bad, fs, polarity=POL)


def test_rejects_missing_rate():
    current, _, _, _, _ = _make(duration_s=1.0)
    with pytest.raises(ValueError, match="sample rate"):
        estimate_baseline(current, None, polarity=POL)


def test_intrusion_rejects_shape_mismatch():
    current, fs, _, starts, ends = _make(duration_s=1.0, n_events=5)
    with pytest.raises(ValueError, match="shape"):
        baseline_intrusion(current, np.zeros(10), (starts, ends), fs, POL)


def test_intrusion_with_no_events_is_nan_not_zero():
    current, fs, _, _, _ = _make(duration_s=1.0)
    b = estimate_baseline(current, fs, polarity=POL)
    r = baseline_intrusion(current, b, (np.empty(0), np.empty(0)), fs, POL)
    assert np.isnan(r["intrusion_score"])
    assert r["n_events"] == 0


def test_intrusion_accepts_dataframe():
    pd = pytest.importorskip("pandas")
    current, fs, _, starts, ends = _make(n_events=50, dwell_s=2e-4)
    b = estimate_baseline(current, fs, window_s=0.01, polarity=POL)
    df = pd.DataFrame({"start_idx": starts, "end_idx": ends})
    from_df = baseline_intrusion(current, b, df, fs, POL)["intrusion_score"]
    from_pair = baseline_intrusion(current, b, (starts, ends), fs, POL)["intrusion_score"]
    assert from_df == pytest.approx(from_pair)


def test_baseline_does_not_overshoot_at_a_drift_step():
    """Pchip, not a cubic spline: a spline rings here and makes phantom events."""
    fs = 50_000.0
    n = 200_000
    rng = np.random.default_rng(3)
    x = np.full(n, I0) + rng.normal(0, 5.0, n)
    x[n // 2 :] += 300.0
    current = x.astype(np.float32)
    b = estimate_baseline(current, fs, window_s=0.01, polarity=POL)
    lo, hi = I0, I0 + 300.0
    assert b.min() > lo - 15.0
    assert b.max() < hi + 15.0


# ---------------------------------------------------------------------------
# 6. the real generator, once it exists
# ---------------------------------------------------------------------------


def test_with_synth_trace_if_available():
    truth_mod = pytest.importorskip(
        "nanoscope.analysis.truth", reason="analysis.truth not written yet (Phase 4)"
    )
    rec, gt = truth_mod.synth_trace(
        fs=50_000.0, duration_s=5.0, i0_pA=-1000.0, noise_pA=15.0, n_events=100, seed=0
    )
    b = estimate_baseline(rec.data, rec.sample_rate_hz, polarity=rec.polarity, window_s=0.01)
    assert b.shape == (rec.n,)
    assert open_pore_current(rec.data, rec.sample_rate_hz, b) == pytest.approx(-1000.0, abs=10.0)
    if {"start_idx", "end_idx"} <= set(gt.columns):
        assert (
            baseline_intrusion(rec.data, b, gt, rec.sample_rate_hz, rec.polarity)["intrusion_score"]
            < 0.15
        )
