"""Tests for baseline-wander removal (detrending): the pre-detection transform that
flattens slow open-pore drift while PRESERVING event shapes and depths.

The load-bearing claims (mirroring the estimator's own hazards):
  a. the open-pore (non-event) samples come out FLAT — their spread collapses by a
     large factor once the drift is subtracted;
  b. event DEPTHS survive — a blockade that was `depth` below its local open pore is
     still `depth` below the flattened open-pore level;
  c. a flat, event-free noise trace is ~unchanged (there is no drift to remove);
  d. a window shorter than the events is not silently accepted — the estimator warns
     it will track them, and the removal then subtracts them (depths collapse);
  e. `estimated_wander` IS the open-pore envelope (`estimate_baseline` cast to
     float32), for a preview/overlay;
  f. validation errors from `estimate_baseline` propagate unchanged.

`remove_baseline_wander`/`estimated_wander` live in `nanoscope.analysis.baseline`.

Sign convention (as in test_baseline.py): I0 = -1000 pA at negative bias, a blockade
reduces |I| so the current rises toward 0 -> excursions are positive -> polarity +1.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanoscope.analysis.baseline import (
    DEFAULT_DETREND_WINDOW_S,
    BaselineWarning,
    estimate_baseline,
    estimated_wander,
    remove_baseline_wander,
)

I0 = -1000.0
POL = 1


def _synth(
    *,
    fs: float = 50_000.0,
    duration_s: float = 20.0,
    noise_pA: float = 12.0,
    drift_lin: float = 250.0,
    drift_sin: float = 120.0,
    drift_f: float = 0.1,
    n_events: int = 8,
    dwell_s: float = 2e-3,
    depth: float = 300.0,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """A flat open-pore level + slow drift (linear ramp + slow sine) + square
    blockades of KNOWN depth. Returns
    ``(current, fs, truth_baseline, event_mask, starts, ends, depth)``."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    truth = I0 + drift_lin * (t / t[-1]) + drift_sin * np.sin(2 * np.pi * drift_f * t)
    x = truth + rng.normal(0.0, noise_pA, n)

    w = int(round(dwell_s * fs))
    gap = n // n_events
    starts = (np.arange(n_events) * gap + gap // 3).astype(np.int64)
    ends = starts + w
    assert ends[-1] < n
    ev = np.zeros(n, dtype=bool)
    for s, e in zip(starts, ends):
        x[s:e] += POL * depth  # blockade -> upward at negative bias
        ev[s:e] = True

    return x.astype(np.float32), fs, truth, ev, starts, ends, depth


def _event_depths(y: np.ndarray, ev: np.ndarray, starts, ends) -> np.ndarray:
    """Per-event median level minus the flattened open-pore reference (pA)."""
    ref = float(np.median(y[~ev]))
    return np.array([np.median(y[s:e]) - ref for s, e in zip(starts, ends)])


# ---------------------------------------------------------------------------
# a. the open pore is flattened
# ---------------------------------------------------------------------------


def test_removal_flattens_the_open_pore():
    current, fs, _truth, ev, _s, _e, _d = _synth()
    y = remove_baseline_wander(current, fs, window_s=0.5)

    assert y.dtype == np.float32
    assert y.shape == current.shape
    assert np.isfinite(y).all()

    before, after = current[~ev], y[~ev]
    # The drift dominated the open-pore spread; removing it collapses std and ptp.
    assert after.std() < before.std() / 5.0
    assert np.ptp(after) < np.ptp(before) / 3.0
    # What remains is essentially the ~12 pA noise floor, not a residual trend.
    assert after.std() < 20.0


# ---------------------------------------------------------------------------
# b. event depths are preserved
# ---------------------------------------------------------------------------


def test_event_depths_are_preserved():
    current, fs, _truth, ev, starts, ends, depth = _synth()
    y = remove_baseline_wander(current, fs, window_s=0.5)

    got = _event_depths(y, ev, starts, ends)
    # Median depth lands on the injected depth, and every event is within tolerance
    # (the whole point: the baseline sailed OVER the events, not INTO them).
    assert np.median(got) == pytest.approx(depth, rel=0.05)
    assert np.all(np.abs(got - depth) < 0.15 * depth)


def test_open_pore_level_is_kept_not_re_zeroed():
    """`signal - base + median(base)`: the flattened trace still sits at the
    REPRESENTATIVE open-pore current (the median of the tracked baseline), so absolute
    depths mean the same thing downstream — it is not collapsed onto zero."""
    current, fs, _truth, ev, _s, _e, _d = _synth()
    reference = float(np.median(estimate_baseline(current, fs, window_s=0.5)))
    y = remove_baseline_wander(current, fs, window_s=0.5)
    assert float(np.median(y[~ev])) == pytest.approx(reference, abs=5.0)
    # A large negative open-pore current, not re-zeroed to ~0.
    assert abs(reference) > 500.0


@pytest.mark.parametrize("method", ["block_median", "percentile"])
def test_median_and_percentile_flatten_and_keep_depths(method):
    current, fs, _truth, ev, starts, ends, depth = _synth()
    y = remove_baseline_wander(current, fs, method=method, window_s=0.5, polarity=POL)
    assert y[~ev].std() < current[~ev].std() / 4.0
    assert np.median(_event_depths(y, ev, starts, ends)) == pytest.approx(depth, rel=0.1)


def test_asls_method_runs_and_is_well_formed():
    """asls is the high-occupancy tool, not the drift tool (its default stiffness
    cannot chase a slow sine), but it must still run and return a finite, same-length
    float32 trace that does not INFLATE the open-pore spread."""
    current, fs, _truth, ev, _s, _e, _d = _synth()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BaselineWarning)
        y = remove_baseline_wander(current, fs, method="asls", window_s=0.5, polarity=POL)
    assert y.dtype == np.float32
    assert y.shape == current.shape
    assert np.isfinite(y).all()
    assert y[~ev].std() <= current[~ev].std() + 1e-3


# ---------------------------------------------------------------------------
# c. a flat, event-free trace is ~unchanged
# ---------------------------------------------------------------------------


def test_flat_noise_trace_is_essentially_unchanged():
    rng = np.random.default_rng(1)
    fs = 50_000.0
    x = (-500.0 + rng.normal(0.0, 10.0, int(fs * 5))).astype(np.float32)
    y = remove_baseline_wander(x, fs, window_s=0.5)
    # No drift to remove: base is ~constant, so y ~= x to well within the noise.
    assert np.max(np.abs(y - x)) < 1.0
    assert y.std() == pytest.approx(x.std(), rel=0.02)
    assert float(np.median(y)) == pytest.approx(float(np.median(x)), abs=1.0)


def test_constant_method_is_a_noop():
    current, fs, *_ = _synth(duration_s=2.0, n_events=2)
    y = remove_baseline_wander(current, fs, method="constant")
    # base = median(x) constant, reference = median(base) = median(x): exact identity.
    np.testing.assert_allclose(y, current, atol=1e-3)


# ---------------------------------------------------------------------------
# d. a too-short window is honest about eating events
# ---------------------------------------------------------------------------


def test_short_window_warns_and_eats_events():
    current, fs, _truth, ev, starts, ends, depth = _synth(dwell_s=4e-3)
    # A window shorter than the events is not silently accepted: the estimator warns
    # it will track them (window_s < 20x min_dwell_s), matching baseline.py's guard.
    with pytest.warns(BaselineWarning, match="min_dwell_s"):
        eaten = remove_baseline_wander(current, fs, window_s=1e-3, min_dwell_s=4e-3)
    # And the consequence is real: the removal subtracted the events too, so the
    # measured depth collapses toward zero.
    got = np.median(_event_depths(eaten, ev, starts, ends))
    assert got < 0.2 * depth


def test_wide_window_does_not_warn():
    current, fs, *_ = _synth(dwell_s=2e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any BaselineWarning becomes a failure
        remove_baseline_wander(current, fs, window_s=0.5, min_dwell_s=2e-3)


# ---------------------------------------------------------------------------
# e. estimated_wander is the base array (for preview/overlay)
# ---------------------------------------------------------------------------


def test_estimated_wander_is_the_baseline_cast_to_float32():
    current, fs, _truth, ev, _s, _e, _d = _synth()
    w = estimated_wander(current, fs, window_s=0.5)
    assert w.dtype == np.float32
    assert w.shape == current.shape
    # It is exactly the estimator's open-pore envelope, cast to float32.
    base = estimate_baseline(current, fs, window_s=0.5)
    np.testing.assert_array_equal(w, np.ascontiguousarray(base, dtype=np.float32))


def test_estimated_wander_tracks_the_true_drift():
    current, fs, truth, ev, _s, _e, _d = _synth()
    w = estimated_wander(current, fs, window_s=0.5).astype(np.float64)
    # In the open pore the wander follows the injected baseline within noise.
    assert np.sqrt(np.mean((w[~ev] - truth[~ev]) ** 2)) < 25.0


def test_removal_equals_signal_minus_wander_plus_reference():
    """The documented identity, so what `estimated_wander` previews is what
    `remove_baseline_wander` subtracts."""
    current, fs, *_ = _synth(duration_s=6.0, n_events=4)
    w = estimated_wander(current, fs, window_s=0.5).astype(np.float64)
    y = remove_baseline_wander(current, fs, window_s=0.5).astype(np.float64)
    ref = float(np.median(w))
    np.testing.assert_allclose(y, current.astype(np.float64) - w + ref, atol=0.5)


# ---------------------------------------------------------------------------
# f. defaults + validation propagation
# ---------------------------------------------------------------------------


def test_default_window_is_the_wide_drift_window():
    # The public default must be the wide detrend window, not the 10 ms I0 default.
    assert DEFAULT_DETREND_WINDOW_S == 0.5
    current, fs, _truth, ev, starts, ends, depth = _synth()
    y = remove_baseline_wander(current, fs)  # no window_s -> the default
    assert y[~ev].std() < current[~ev].std() / 4.0
    assert np.median(_event_depths(y, ev, starts, ends)) == pytest.approx(depth, rel=0.1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window_s": 0.0}, "positive"),
        ({"window_s": -1.0}, "positive"),
        ({"window_s": 1e6}, "longer than the recording"),
        ({"method": "magic"}, "unknown method"),
    ],
)
def test_propagates_estimate_baseline_validation(kwargs, match):
    current, fs, *_ = _synth(duration_s=2.0, n_events=2)
    with pytest.raises(ValueError, match=match):
        remove_baseline_wander(current, fs, **kwargs)


def test_rejects_missing_rate():
    current, _fs, *_ = _synth(duration_s=2.0, n_events=2)
    with pytest.raises(ValueError, match="sample rate"):
        remove_baseline_wander(current, None)


def test_rejects_nonfinite_signal():
    current, fs, *_ = _synth(duration_s=2.0, n_events=2)
    bad = current.copy()
    bad[123] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        remove_baseline_wander(bad, fs)
