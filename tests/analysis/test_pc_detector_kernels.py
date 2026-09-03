"""The conventional detector kernels: fixed threshold, peaks, derivative, TKEO.

Every test here hands a kernel a trace whose event positions are KNOWN and asks
whether it gets them back. A detector that cannot recover the events it was handed
is not a detector, and the recall/precision numbers quoted in each kernel's
docstring are the ones these tests pin.

Two invariants matter as much as the recall:

* the kernels return the SAME ``(starts, ends)`` form `detect_events` hands to
  `characterize`, so the 20 frozen EventSet columns are produced by the existing
  measurement path and nothing about them changes;
* the kernels measure baseline, polarity and noise through the app's existing doors,
  so two detectors run on one recording cannot report depths that disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.detect import (
    NOISE_METHODS,
    _hysteresis_runs,
    _run_maxima,
    _runs,
    detect_derivative,
    detect_events,
    detect_fixed_threshold,
    detect_peaks,
    detect_tkeo,
    detection_basis,
    noise_level,
    tkeo_energy,
)
from nanoscope.analysis.events import SCHEMA

FS = 50_000.0
I0 = 1000.0
DEPTH = 100.0
SIGMA = 5.0


def _trace(*, dwell=50, depth=DEPTH, n_events=40, spacing=2000, first=5000,
           n=100_000, sigma=SIGMA, seed=7, drift=0.0):
    """A trace whose events are where we put them. Returns (float32 trace, truth)."""
    rng = np.random.default_rng(seed)
    x = np.full(n, I0) + rng.normal(0.0, sigma, n)
    if drift:
        x += np.linspace(0.0, drift * n / FS, n)
    truth = []
    for i in range(n_events):
        s = first + i * spacing
        x[s : s + dwell] -= depth
        truth.append((s, s + dwell))
    return x.astype(np.float32), truth


def _recall_precision(truth, starts, ends):
    """Overlap matching, one detection per truth event and vice versa."""
    found = list(zip(starts.tolist(), ends.tolist()))
    used: set[int] = set()
    hits = 0
    for ts, te in truth:
        for j, (fs_, fe) in enumerate(found):
            if j not in used and fs_ < te and fe > ts:
                used.add(j)
                hits += 1
                break
    recall = hits / len(truth)
    precision = len(used) / len(found) if found else 0.0
    return recall, precision


@pytest.fixture(scope="module")
def square():
    """40 x 1 ms, 100 pA deep, 5 pA noise -- the ordinary translocation shape."""
    x, truth = _trace()
    return x, truth, detection_basis(np.asarray(x, dtype=np.float64), FS)


@pytest.fixture(scope="module")
def spikes():
    """40 x 3 samples (60 us) -- too short for a dwell-based detector to like."""
    x, truth = _trace(dwell=3)
    return x, truth, detection_basis(np.asarray(x, dtype=np.float64), FS)


@pytest.fixture(scope="module")
def single_sample_spikes():
    """40 x 1 sample (20 us) -- the shape TKEO exists for."""
    x, truth = _trace(dwell=1)
    return x, truth, detection_basis(np.asarray(x, dtype=np.float64), FS)


# ---------------------------------------------------------------------------
# each kernel recovers the events it was handed
# ---------------------------------------------------------------------------
def test_the_fixed_threshold_recovers_every_square_event_at_full_precision(square):
    x, truth, _ = square
    r, p = _recall_precision(truth, *detect_fixed_threshold(
        x, FS, level=I0 - DEPTH / 2, exit_level=I0 - 2 * SIGMA))
    assert (r, p) == (1.0, 1.0)


def test_peak_detection_recovers_squares_spikes_and_single_sample_spikes(
    square, spikes, single_sample_spikes
):
    for x, truth, basis in (square, spikes, single_sample_spikes):
        r, p = _recall_precision(truth, *detect_peaks(x, FS, basis=basis))
        assert (r, p) == (1.0, 1.0)


def test_the_derivative_detector_recovers_squares_and_spikes(square, spikes):
    for x, truth, basis in (square, spikes):
        r, p = _recall_precision(truth, *detect_derivative(x, FS, basis=basis))
        assert (r, p) == (1.0, 1.0)


def test_tkeo_recovers_single_sample_spikes_with_no_linking_at_all(
    single_sample_spikes,
):
    x, truth, basis = single_sample_spikes
    r, p = _recall_precision(truth, *detect_tkeo(x, FS, basis=basis))
    assert (r, p) == (1.0, 1.0)


def test_tkeo_recovers_flat_topped_events_once_link_gap_bridges_their_edges(square):
    x, truth, basis = square
    unlinked_s, _unlinked_e = detect_tkeo(x, FS, basis=basis)
    # An edge operator fires twice (or more) per flat-topped event: 626 spans for 40
    # events, which is the measurement the docstring's "DO NOT" is built on.
    assert unlinked_s.size > 10 * len(truth)
    r, p = _recall_precision(truth, *detect_tkeo(x, FS, basis=basis, link_gap_s=2e-3))
    assert (r, p) == (1.0, 1.0)


# ---------------------------------------------------------------------------
# bounds, not just counts
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clean_pulse():
    """One square pulse over samples [500, 600) with negligible (0.01 pA) noise."""
    rng = np.random.default_rng(0)
    x = np.full(2000, I0) + rng.normal(0.0, 0.01, 2000)
    x[500:600] -= DEPTH
    x = x.astype(np.float32)
    return x, detection_basis(np.asarray(x, dtype=np.float64), FS)


def test_the_fixed_threshold_returns_the_exact_edges_of_a_clean_pulse(clean_pulse):
    x, _basis = clean_pulse
    starts, ends = detect_fixed_threshold(x, FS, level=I0 - DEPTH / 2)
    assert (starts.tolist(), ends.tolist()) == ([500], [600])


def test_the_derivative_detector_returns_the_exact_edges_of_a_clean_pulse(clean_pulse):
    x, basis = clean_pulse
    starts, ends = detect_derivative(x, FS, basis=basis)
    assert (starts.tolist(), ends.tolist()) == ([500], [600])


def test_peak_widths_land_within_one_sample_of_the_edges_of_a_clean_pulse(clean_pulse):
    x, basis = clean_pulse
    starts, ends = detect_peaks(x, FS, basis=basis)
    assert starts.size == 1
    assert abs(int(starts[0]) - 500) <= 1 and abs(int(ends[0]) - 600) <= 1


# ---------------------------------------------------------------------------
# the guidance in the docstrings is the behaviour of the code
# ---------------------------------------------------------------------------
def test_a_single_absolute_level_chatters_and_the_exit_level_stops_it():
    x, _ = _trace(n_events=1, first=1000, n=5000)
    parked = I0 - DEPTH + 2.0          # 2 pA above the blocked level
    one_level, _ = detect_fixed_threshold(x, FS, level=parked)
    with_exit, _ = detect_fixed_threshold(
        x, FS, level=parked, exit_level=I0 - 2 * SIGMA)
    assert one_level.size > 5          # measured: 11 fragments of one event
    assert with_exit.size == 1


def test_an_exit_level_on_the_far_side_of_the_enter_level_is_refused():
    x, _ = _trace(n_events=1, first=1000, n=5000)
    with pytest.raises(ValueError, match="hysteresis"):
        detect_fixed_threshold(x, FS, level=950.0, exit_level=940.0)


def test_the_fixed_threshold_loses_events_to_drift_which_is_why_it_says_not_to():
    x, truth = _trace(drift=-40.0)
    starts, ends = detect_fixed_threshold(
        x, FS, level=I0 - DEPTH / 2, exit_level=I0 - 2 * SIGMA)
    recall, _p = _recall_precision(truth, starts, ends)
    assert recall < 0.5                                   # measured 0.400
    assert (ends - starts).max() / FS > 1.0               # a 1302.8 ms "event"
    # ...and the detector that tracks a baseline is untouched by the same ramp.
    df, _prov = detect_events(np.asarray(x, dtype=np.float64), FS)
    r2, p2 = _recall_precision(
        truth, df["start_idx"].to_numpy(), df["end_idx"].to_numpy())
    assert (r2, p2) == (1.0, 1.0)


def test_prominence_alone_admits_noise_and_the_height_test_is_what_excludes_it(square):
    x, truth, basis = square
    _r, p_no_height = _recall_precision(
        truth, *detect_peaks(x, FS, basis=basis, height_sigma=None))
    _r2, p_default = _recall_precision(truth, *detect_peaks(x, FS, basis=basis))
    assert p_no_height < 0.2           # measured 0.101, 397 spans for 40 events
    assert p_default == 1.0


def test_the_full_prominence_base_swallows_the_trace_and_recall_collapses(square):
    x, truth, basis = square
    r, _p = _recall_precision(
        truth, *detect_peaks(x, FS, basis=basis, rel_height=1.0))
    assert r < 0.1                     # measured 0.025 -- one span, not forty


def test_the_derivative_detector_needs_a_steep_edge_not_a_deep_event():
    """A 6-sigma-deep event is only a 4.2-sigma EDGE, and k=5 rejects it."""
    x, truth = _trace(depth=6 * SIGMA)
    basis = detection_basis(np.asarray(x, dtype=np.float64), FS)
    r_deriv, _p = _recall_precision(truth, *detect_derivative(x, FS, basis=basis))
    assert r_deriv < 0.5                                  # measured 0.225
    df, _prov = detect_events(np.asarray(x, dtype=np.float64), FS)
    r_schmitt, p_schmitt = _recall_precision(
        truth, df["start_idx"].to_numpy(), df["end_idx"].to_numpy())
    assert (r_schmitt, p_schmitt) == (1.0, 1.0)


def test_the_teager_kaiser_operator_is_exactly_sign_blind():
    x = np.array([0.0, 1.0, 4.0, 2.0, -3.0, 0.5, 0.0])
    assert np.array_equal(tkeo_energy(x), tkeo_energy(-x))
    # and the definition itself, on the one interior sample of a three-sample array
    assert tkeo_energy(np.array([2.0, 3.0, 4.0]))[1] == 3.0 * 3.0 - 2.0 * 4.0
    assert tkeo_energy(np.array([2.0, 3.0, 4.0]))[0] == 0.0


def test_the_polarity_gate_removes_wrong_direction_bursts_and_costs_no_recall():
    x, truth = _trace(dwell=3)
    for i in range(20):                       # 20 enhancement-direction artifacts
        s = 6000 + i * 4000
        x[s : s + 3] += DEPTH
    basis = detection_basis(np.asarray(x, dtype=np.float64), FS)
    r_off, p_off = _recall_precision(
        truth, *detect_tkeo(x, FS, basis=basis, polarity_gate=False))
    r_on, p_on = _recall_precision(
        truth, *detect_tkeo(x, FS, basis=basis, polarity_gate=True))
    assert r_off == r_on == 1.0
    assert p_on > p_off                       # measured 0.580 vs 0.400


# ---------------------------------------------------------------------------
# the noise basis is a choice now, and MAD is the one that survives the events
# ---------------------------------------------------------------------------
def test_the_noise_basis_is_selectable_and_only_the_robust_ones_survive_events(square):
    x, _truth, basis = square
    resid = basis.current - basis.baseline
    mad = noise_level(resid, "mad")
    iqr = noise_level(resid, "iqr")
    std = noise_level(resid, "std")
    assert set(NOISE_METHODS) == {"mad", "std", "iqr"}
    assert abs(mad - SIGMA) / SIGMA < 0.05        # measured 5.1472 against 5.0
    assert abs(iqr - SIGMA) / SIGMA < 0.05        # measured 5.1480
    assert std > 2 * SIGMA                        # measured 14.8946 -- the events
    with pytest.raises(ValueError, match="unknown noise method"):
        noise_level(resid, "rms")


def test_a_kernel_thresholds_on_the_basis_it_was_asked_for():
    """The events inflate std, std raises the bar, and the events fall under it.

    40 events of 4 ms and 30 pA in 5 pA noise (10% duty cycle): MAD reads 5.60 pA, so
    the 5-sigma height bar is 28 pA and all 40 clear it; std reads 9.55 pA because the
    events are IN the trace it is measured on, the same bar becomes 47.8 pA, and the
    detector returns 2 spans. This is the whole argument for MAD being the default,
    made reachable as a parameter.
    """
    x, truth = _trace(depth=30.0, dwell=200)
    on_mad = detect_peaks(x, FS, noise_method="mad")
    on_std = detect_peaks(x, FS, noise_method="std")
    assert _recall_precision(truth, *on_mad) == (1.0, 1.0)
    assert on_std[0].size < 0.25 * len(truth)


# ---------------------------------------------------------------------------
# one baseline, one polarity, one merge rule -- the whole point of the seam
# ---------------------------------------------------------------------------
def test_the_kernels_and_the_schmitt_detector_agree_on_polarity_and_baseline(square):
    x, _truth, basis = square
    df, prov = detect_events(np.asarray(x, dtype=np.float64), FS)
    assert basis.polarity == prov["polarity"] == -1
    assert basis.polarity_source == "measured"
    assert abs(float(np.median(basis.baseline)) - prov["baseline_median_pA"]) < 1e-9
    assert basis.baseline_method == prov["baseline_method"]


def test_a_polarity_that_is_a_gain_rather_than_a_direction_is_refused(square):
    x, _truth, _basis = square
    with pytest.raises(ValueError):
        detection_basis(np.asarray(x, dtype=np.float64), FS, polarity=7)


def test_the_kernels_obey_the_shared_occupancy_brake_on_the_merge_window():
    """40 events 8 ms apart, merged at 10 ms, must NOT become one 312 ms row."""
    x, truth = _trace(dwell=20, spacing=400, n=20_000, first=200, n_events=40)
    basis = detection_basis(np.asarray(x, dtype=np.float64), FS)
    starts, ends = detect_fixed_threshold(
        x, FS, level=I0 - DEPTH / 2, exit_level=I0 - 2 * SIGMA, merge_gap_s=10e-3)
    assert starts.size == len(truth)
    assert (ends - starts).max() < 10 * 20
    del basis


def test_a_min_dwell_floor_drops_the_short_spans_and_nothing_else(spikes):
    x, truth, basis = spikes
    kept_all, _e = detect_peaks(x, FS, basis=basis)
    kept_long, _e2 = detect_peaks(x, FS, basis=basis, min_dwell_s=1e-3)
    assert kept_all.size == len(truth)
    assert kept_long.size == 0


# ---------------------------------------------------------------------------
# the drop-in contract: these intervals go straight into the frozen pipeline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kernel",
    ["fixed", "peaks", "derivative", "tkeo"],
)
def test_every_kernel_returns_sorted_non_overlapping_int64_half_open_spans(
    square, kernel
):
    x, _truth, basis = square
    if kernel == "fixed":
        starts, ends = detect_fixed_threshold(x, FS, level=I0 - DEPTH / 2)
    elif kernel == "peaks":
        starts, ends = detect_peaks(x, FS, basis=basis)
    elif kernel == "derivative":
        starts, ends = detect_derivative(x, FS, basis=basis)
    else:
        starts, ends = detect_tkeo(x, FS, basis=basis, link_gap_s=2e-3)
    assert starts.dtype == np.int64 and ends.dtype == np.int64
    assert starts.size == ends.size and starts.size > 0
    assert np.all(ends > starts)
    assert np.all(np.diff(starts) > 0)
    assert np.all(starts[1:] >= ends[:-1])
    assert starts[0] >= 0 and ends[-1] <= x.size


def test_kernel_spans_characterise_into_the_frozen_twenty_column_schema(square):
    """The whole reason the kernels return intervals: `characterize` does the rest."""
    x, truth, basis = square
    starts, ends = detect_derivative(x, FS, basis=basis)
    df = characterize(
        basis.current, basis.baseline, starts, ends, FS, basis.polarity,
        noise_rms=basis.noise, clip_bounds=(0, x.size),
    )
    assert list(df.columns)[: len(SCHEMA)] == list(SCHEMA)
    assert len(df) == len(truth)
    assert abs(float(df["depth_pA"].median()) - DEPTH) < 2.0
    assert abs(float(df["dwell_s"].median()) - 1e-3) < 1e-5
    assert bool(df["accepted"].all())


# ---------------------------------------------------------------------------
# the extraction that made the hysteresis rule shared must not have moved it
# ---------------------------------------------------------------------------
def test_the_shared_hysteresis_rule_is_the_rule_the_schmitt_detector_used(square):
    """`_hysteresis_runs` reproduces the inline three lines it was extracted from."""
    x, _truth, basis = square
    dev = basis.dev
    padded = np.empty(dev.size + 1)
    padded[: dev.size] = dev
    padded[dev.size] = -np.inf
    thr_in, thr_out = 5.0 * basis.noise, 2.0 * basis.noise

    want_s, want_e = _runs(dev > thr_out)
    n_candidates = int(want_s.size)
    keep = _run_maxima(padded, want_s, want_e) > thr_in
    want_s, want_e = want_s[keep], want_e[keep]

    got_s, got_e, got_n = _hysteresis_runs(padded, dev.size, thr_in, thr_out)
    assert got_n == n_candidates
    assert np.array_equal(got_s, want_s) and np.array_equal(got_e, want_e)


def test_the_schmitt_detector_still_finds_exactly_the_forty_events_it_always_did(
    square,
):
    """No measured number moves: bounds, depth and the audit trail are unchanged."""
    x, truth, _basis = square
    df, prov = detect_events(np.asarray(x, dtype=np.float64), FS)
    assert len(df) == len(truth) == prov["n_events"]
    assert (
        prov["n_candidate_runs"]
        - prov["n_rejected_below_enter"]
        - prov["n_merged"]
        - prov["n_rejected_short"]
        - prov["n_rejected_excluded"]
    ) == prov["n_events"]
    r, p = _recall_precision(
        truth, df["start_idx"].to_numpy(), df["end_idx"].to_numpy())
    assert (r, p) == (1.0, 1.0)
