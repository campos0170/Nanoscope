"""Tests for analysis.characterize: ground-truth square events in, known numbers out.

Ported from the SOURCE ``tests/test_characterize.py`` (PORT_PLAN §3.3). The
Recording-based fixtures are refactored to the signal-first data model: the
helper returns ``(current, baseline, polarity)`` primitives and every call passes
them to ``characterize(current, baseline, starts, ends, fs, polarity, ...)``.
Every assertion and pinned number is preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.characterize import characterize, recharacterize_span
from nanoscope.analysis.events import SCHEMA

FS = 100_000.0


def build_trace(
    *,
    i0_pA: float = 1000.0,
    polarity: int = -1,
    n: int = 200_000,
    events: list[tuple[int, int, float]] = (),
    noise_pA: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """A flat trace with exact square deviations. `events` is [(start, end, dev_pA)].

    dev is applied through the sign convention: current = baseline + dev/polarity,
    so that polarity*(current - baseline) == dev exactly. Returns the current array
    (float32, as the neo readers deliver), the full-length baseline (float64), and
    the polarity.
    """
    rng = np.random.default_rng(seed)
    baseline = np.full(n, float(i0_pA))
    current = baseline.copy()
    for s, e, dev in events:
        current[s:e] += dev / polarity
    if noise_pA:
        current += rng.normal(0.0, noise_pA, n)
    return current.astype(np.float32), baseline, polarity


def test_schema_exact():
    """Columns and dtypes must match SCHEMA exactly -- downstream indexes by name."""
    cur, base, pol = build_trace(events=[(1000, 1500, 100.0)])
    df = characterize(cur, base, [1000], [1500], FS, pol, noise_rms=5.0)

    assert list(df.columns) == list(SCHEMA)
    for col, (dtype, _, _) in SCHEMA.items():
        assert df[col].dtype == pd.api.types.pandas_dtype(dtype), col


def test_empty_input_returns_empty_schema_frame():
    cur, base, pol = build_trace()
    df = characterize(
        cur, base, np.array([], dtype=int), np.array([], dtype=int), FS, pol, noise_rms=5.0
    )
    assert len(df) == 0
    assert list(df.columns) == list(SCHEMA)


def test_square_events_recover_depth_dwell_ecd_exactly():
    """Noiseless squares: every feature is analytic, so demand near-machine precision."""
    truth = [(1_000, 1_500, 120.0), (10_000, 10_050, 300.0), (50_000, 62_000, 45.5)]
    cur, base, pol = build_trace(events=truth)
    starts = np.array([t[0] for t in truth])
    ends = np.array([t[1] for t in truth])
    depths = np.array([t[2] for t in truth])

    df = characterize(cur, base, starts, ends, FS, pol, noise_rms=10.0)

    assert len(df) == 3
    np.testing.assert_allclose(df["dwell_s"], (ends - starts) / FS, rtol=1e-12)
    np.testing.assert_allclose(df["depth_pA"], depths, rtol=1e-4)
    np.testing.assert_allclose(df["max_dev_pA"], depths, rtol=1e-4)
    np.testing.assert_allclose(df["mean_dev_pA"], depths, rtol=1e-4)
    np.testing.assert_allclose(df["baseline_pA"], 1000.0, rtol=1e-9)
    np.testing.assert_allclose(df["frac_depth"], depths / 1000.0, rtol=1e-4)
    # ECD of a square = depth * dwell.
    np.testing.assert_allclose(df["ecd_pA_s"], depths * (ends - starts) / FS, rtol=1e-4)
    np.testing.assert_allclose(df["snr"], depths / 10.0, rtol=1e-4)
    np.testing.assert_array_equal(df["t_start_s"], starts / FS)
    np.testing.assert_array_equal(df["n_levels"], 1)
    assert df["accepted"].all()
    assert (df["flag"] == "").all()


def test_enhancement_polarity_gives_positive_depth():
    """polarity=+1: current goes UP, depth must still come back positive."""
    cur, base, pol = build_trace(polarity=+1, events=[(2_000, 2_800, 75.0)])
    assert cur[2_400] > cur[0]

    df = characterize(cur, base, [2_000], [2_800], FS, pol, noise_rms=5.0)
    assert df["depth_pA"].iloc[0] == pytest.approx(75.0, rel=1e-4)
    assert df["ecd_pA_s"].iloc[0] > 0


def test_negative_open_pore_current_blockade():
    """Real data sits at I0 < 0. Depth and frac_depth must stay positive magnitudes.

    Under the fixed convention dev = polarity*(current - baseline), a blockade on a
    negative I0 moves current UP (toward zero), so the sign that makes dev positive
    is +1. detect_polarity() owns choosing it; characterize() only applies it.
    """
    cur, base, pol = build_trace(i0_pA=-1000.0, polarity=+1, events=[(3_000, 3_600, 90.0)])
    # Blockade: |current| is reduced.
    assert abs(cur[3_300]) < abs(cur[0])

    df = characterize(cur, base, [3_000], [3_600], FS, pol, noise_rms=5.0)
    assert df["depth_pA"].iloc[0] == pytest.approx(90.0, rel=1e-4)
    assert df["baseline_pA"].iloc[0] == pytest.approx(-1000.0)
    assert df["frac_depth"].iloc[0] == pytest.approx(0.09, rel=1e-3)


def test_median_depth_unbiased_max_is_not():
    """THE justification for median-based depth: quantify the max's upward bias.

    max_dev over N noisy samples is an extreme order statistic with expectation
    ~ depth + sigma*sqrt(2*ln N). The median is a consistent estimate of the level.
    """
    sigma = 20.0
    depth = 200.0
    length = 500
    n_events = 300
    gap = 200

    starts = np.arange(n_events) * (length + gap) + 1_000
    ends = starts + length
    n = int(ends[-1] + 1_000)
    cur, base, pol = build_trace(
        n=n, events=[(s, e, depth) for s, e in zip(starts, ends)], noise_pA=sigma, seed=7
    )
    df = characterize(cur, base, starts, ends, FS, pol, noise_rms=sigma)

    median_bias = df["depth_pA"].mean() - depth
    max_bias = df["max_dev_pA"].mean() - depth

    # The median is unbiased: its standard error over 300 events x 500 samples is
    # tiny, so a fifth of a sigma is a loose but real bound.
    assert abs(median_bias) < 0.2 * sigma, f"median depth biased by {median_bias:.2f} pA"

    # The max is biased high by ~sigma*sqrt(2*ln(500)) ~ 3.5 sigma.
    expected_max_bias = sigma * np.sqrt(2 * np.log(length))
    assert max_bias > 2.5 * sigma, f"max bias only {max_bias:.2f} pA"
    assert max_bias == pytest.approx(expected_max_bias, rel=0.25)

    # And the bias is not a wash: it inflates every reported dI/I0 by ~35%.
    assert max_bias / median_bias > 10 if median_bias > 0 else True
    assert df["max_dev_pA"].mean() > df["depth_pA"].mean() + 2 * sigma


def test_max_bias_grows_with_dwell_median_does_not():
    """The killer consequence: a max-based depth manufactures a dwell-depth correlation."""
    sigma = 20.0
    depth = 200.0
    rng_lengths = [50, 5_000]
    results = {}
    for length in rng_lengths:
        n_events = 100
        gap = 200
        starts = np.arange(n_events) * (length + gap) + 500
        ends = starts + length
        n = int(ends[-1] + 500)
        cur, base, pol = build_trace(
            n=n, events=[(s, e, depth) for s, e in zip(starts, ends)], noise_pA=sigma, seed=11
        )
        df = characterize(cur, base, starts, ends, FS, pol, noise_rms=sigma)
        results[length] = (df["depth_pA"].mean(), df["max_dev_pA"].mean())

    med_short, max_short = results[50]
    med_long, max_long = results[5_000]

    # Median: same level whatever the dwell.
    assert abs(med_long - med_short) < 0.3 * sigma
    # Max: 100x longer event reads ~1.5 sigma deeper, purely from sampling.
    assert max_long - max_short > 1.0 * sigma


def test_two_level_event_yields_two_levels():
    """Folded/unfolded protein: a stepped blockade must report n_levels == 2."""
    s, mid, e = 5_000, 5_400, 5_800
    cur, base, pol = build_trace(
        events=[(s, mid, 100.0), (mid, e, 250.0)], noise_pA=8.0, seed=3
    )
    df = characterize(cur, base, [s], [e], FS, pol, noise_rms=8.0)
    assert int(df["n_levels"].iloc[0]) == 2


def test_three_level_event():
    s = 5_000
    bounds = [5_000, 5_300, 5_600, 5_900]
    devs = [80.0, 200.0, 120.0]
    cur, base, pol = build_trace(
        events=[(a, b, d) for (a, b), d in zip(zip(bounds[:-1], bounds[1:]), devs)],
        noise_pA=6.0,
        seed=5,
    )
    df = characterize(cur, base, [s], [bounds[-1]], FS, pol, noise_rms=6.0)
    assert int(df["n_levels"].iloc[0]) == 3


def test_square_events_stay_at_one_level_under_noise():
    """The stop rule must not manufacture sub-states out of white noise."""
    sigma = 15.0
    length, gap, n_events = 400, 200, 200
    starts = np.arange(n_events) * (length + gap) + 500
    ends = starts + length
    n = int(ends[-1] + 500)
    cur, base, pol = build_trace(
        n=n, events=[(s, e, 150.0) for s, e in zip(starts, ends)], noise_pA=sigma, seed=19
    )
    df = characterize(cur, base, starts, ends, FS, pol, noise_rms=sigma)
    frac_single = (df["n_levels"] == 1).mean()
    assert frac_single > 0.9, f"only {frac_single:.0%} of square events read as 1 level"


def test_n_levels_capped_and_short_events_skip():
    """Below ~10 samples there is nothing to resolve; say 1 rather than guess."""
    cur, base, pol = build_trace(events=[(1_000, 1_006, 100.0)], noise_pA=5.0, seed=2)
    df = characterize(cur, base, [1_000], [1_006], FS, pol, noise_rms=5.0)
    assert int(df["n_levels"].iloc[0]) == 1

    # A many-stepped staircase must saturate at the cap, not run away.
    bounds = np.arange(2_000, 2_000 + 9 * 200 + 1, 200)
    devs = [50.0 * (i + 1) for i in range(len(bounds) - 1)]
    cur2, base2, pol2 = build_trace(
        events=[(a, b, d) for (a, b), d in zip(zip(bounds[:-1], bounds[1:]), devs)],
        noise_pA=4.0,
        seed=4,
    )
    df2 = characterize(cur2, base2, [int(bounds[0])], [int(bounds[-1])], FS, pol2, noise_rms=4.0)
    assert int(df2["n_levels"].iloc[0]) == 4


def test_rise_fall_from_known_ramp():
    """Linear ramps of R samples give a 10-90% time of exactly 0.8*R/fs."""
    depth = 200.0
    ramp = 40
    s, e = 5_000, 6_000
    n = 12_000
    baseline = np.full(n, 1000.0)
    current = baseline.copy()
    dev = np.zeros(n)
    dev[s : s + ramp] = np.linspace(0, depth, ramp, endpoint=False)
    dev[s + ramp : e - ramp] = depth
    dev[e - ramp : e] = np.linspace(depth, 0, ramp, endpoint=False)
    current += dev / -1

    df = characterize(current.astype(np.float32), baseline, [s], [e], FS, -1, noise_rms=1.0)

    expected = 0.8 * ramp / FS
    assert df["rise_s"].iloc[0] == pytest.approx(expected, rel=0.06)
    assert df["fall_s"].iloc[0] == pytest.approx(expected, rel=0.06)


def test_rise_fall_nan_when_transition_is_not_in_the_record():
    """A NaN means 'not measurable here', which is the truth. It must not be 0.

    An event truncated by the end of the recording has no 90->10 exit, and one
    truncated by the start has no 10->90 entry. Reporting 0 s would tell the user
    the pore has infinite bandwidth.
    """
    n = 4_000
    baseline = np.full(n, 1000.0)
    current = baseline.copy()
    current[:500] -= 200.0   # already blocked when recording started
    current[3_500:] -= 200.0  # still blocked when recording stopped

    df = characterize(
        current.astype(np.float32), baseline, [0, 3_500], [500, n], FS, -1, noise_rms=1.0
    )
    assert np.isnan(df["rise_s"].iloc[0])   # no entry transition on record
    assert np.isfinite(df["fall_s"].iloc[0])
    assert np.isfinite(df["rise_s"].iloc[1])
    assert np.isnan(df["fall_s"].iloc[1])   # no exit transition on record


def test_filter_limited_flag():
    """dwell < 2*filter_rise_s: depth is attenuated, dwell biased long. Must be flagged."""
    truth = [(1_000, 1_010, 100.0), (5_000, 9_000, 100.0)]
    cur, base, pol = build_trace(events=truth)
    starts = np.array([t[0] for t in truth])
    ends = np.array([t[1] for t in truth])

    df = characterize(cur, base, starts, ends, FS, pol, noise_rms=5.0)
    assert not df["filter_limited"].any()

    # 10 samples = 100 us; a 100 us rise time makes the first event filter-limited.
    df2 = characterize(cur, base, starts, ends, FS, pol, noise_rms=5.0, filter_rise_s=1e-4)
    assert list(df2["filter_limited"]) == [True, False]


def test_pore_id_resolution():
    """pore_id defaults to the recording name and an explicit pore_id overrides it."""
    cur, base, pol = build_trace(events=[(1_000, 1_500, 100.0)])
    assert (
        characterize(cur, base, [1000], [1500], FS, pol, noise_rms=5.0, recording="synthetic")[
            "pore_id"
        ].iloc[0]
        == "synthetic"
    )
    assert (
        characterize(
            cur, base, [1000], [1500], FS, pol, noise_rms=5.0, recording="synthetic", pore_id="pore-A"
        )["pore_id"].iloc[0]
        == "pore-A"
    )
    # With neither, pore_id is the empty fallback.
    assert characterize(cur, base, [1000], [1500], FS, pol, noise_rms=5.0)["pore_id"].iloc[0] == ""


def test_recording_column_is_stamped():
    cur, base, pol = build_trace(events=[(1_000, 1_500, 100.0)])
    df = characterize(cur, base, [1000], [1500], FS, pol, noise_rms=5.0, recording="rec-42")
    assert df["recording"].iloc[0] == "rec-42"


def test_drifting_baseline_is_subtracted_not_ignored():
    """depth must be measured against the LOCAL baseline, not the global mean."""
    n = 40_000
    drift = np.linspace(1000.0, 1400.0, n)
    current = drift.copy()
    s, e = 20_000, 21_000
    current[s:e] -= 150.0
    df = characterize(current.astype(np.float32), drift, [s], [e], FS, -1, noise_rms=3.0)
    assert df["depth_pA"].iloc[0] == pytest.approx(150.0, rel=1e-3)
    # frac_depth uses the local I0 (~1200), not the trace start (1000).
    assert df["frac_depth"].iloc[0] == pytest.approx(150.0 / df["baseline_pA"].iloc[0], rel=1e-6)
    assert df["baseline_pA"].iloc[0] == pytest.approx(1205.0, abs=5.0)


@pytest.mark.parametrize(
    "kwargs, starts, ends, match",
    [
        ({}, [1000], [1000], "zero-length"),
        ({}, [1000], [500], "zero-length"),
        ({}, [-5], [100], "outside the trace"),
        ({}, [1000], [10**9], "outside the trace"),
        ({}, [1000, 1200], [1500, 1600], "overlap"),
        ({}, [1000, 500], [1500, 1600], "ascending"),
        ({"noise_rms": 0.0}, [1000], [1500], "noise_rms"),
        ({"noise_rms": -1.0}, [1000], [1500], "noise_rms"),
        ({"noise_rms": np.nan}, [1000], [1500], "noise_rms"),
        ({"filter_rise_s": -1.0}, [1000], [1500], "filter_rise_s"),
    ],
)
def test_fails_loudly_on_nonsense(kwargs, starts, ends, match):
    cur, base, pol = build_trace()
    call = {"noise_rms": 5.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        characterize(cur, base, starts, ends, FS, pol, **call)


def test_baseline_length_mismatch_raises():
    cur, _, pol = build_trace()
    with pytest.raises(ValueError, match="full-length"):
        characterize(cur, np.zeros(10), [1000], [1500], FS, pol, noise_rms=5.0)


def test_length_mismatch_raises():
    cur, base, pol = build_trace()
    with pytest.raises(ValueError, match="differ in length"):
        characterize(cur, base, [1000, 2000], [1500], FS, pol, noise_rms=5.0)


@pytest.mark.parametrize(
    "fs, match",
    [
        (None, "fs is required"),
        (0.0, "fs must be positive"),
        (-50_000.0, "fs must be positive"),
        (float("inf"), "fs must be positive"),
        (float("nan"), "fs must be positive"),
    ],
)
def test_bad_fs_raises_value_error_naming_fs(fs, match):
    """A bad fs must be refused the same way `detect.detect_events` refuses one:
    ValueError naming fs, never a bare TypeError/ZeroDivisionError/OverflowError
    leaking out of an un-vetted arithmetic use downstream (regression for B11)."""
    cur, base, pol = build_trace(events=[(1000, 1500, 100.0)])
    with pytest.raises(ValueError, match=match):
        characterize(cur, base, [1000], [1500], fs, pol, noise_rms=5.0)


@pytest.mark.parametrize(
    "fs, match",
    [
        (None, "fs is required"),
        (0.0, "fs must be positive"),
        (-50_000.0, "fs must be positive"),
        (float("inf"), "fs must be positive"),
    ],
)
def test_recharacterize_span_bad_fs_raises_value_error_naming_fs(fs, match):
    """`recharacterize_span` must refuse a bad fs exactly like `characterize` and
    `detect_events` -- not silently accept a negative rate (which came back as an
    ACCEPTED row with a negative dwell_s) and not raise
    ZeroDivisionError/OverflowError/TypeError for 0/inf/None (B11)."""
    signal = np.full(50_000, -500.0)
    signal[20_000:20_100] += 100.0
    with pytest.raises(ValueError, match=match):
        recharacterize_span(signal, fs, 20_000, 20_100)


def test_zero_baseline_flagged_not_silently_infinite():
    """frac_depth against a zero I0 is undefined -- NaN + flag, never inf."""
    n = 5_000
    baseline = np.zeros(n)
    current = np.zeros(n)
    current[1_000:1_500] = -50.0
    df = characterize(current.astype(np.float32), baseline, [1_000], [1_500], FS, -1, noise_rms=2.0)
    assert np.isnan(df["frac_depth"].iloc[0])
    assert df["flag"].iloc[0] == "zero_baseline"
    assert df["depth_pA"].iloc[0] == pytest.approx(50.0, rel=1e-3)


def test_median_is_robust_to_a_spike_inside_the_event():
    """One bad sample must not move the depth. It moves the max by design."""
    cur, base, pol = build_trace(events=[(1_000, 1_500, 100.0)])
    current = cur.copy()
    current[1_200] -= 800.0
    df = characterize(current, base, [1_000], [1_500], FS, pol, noise_rms=5.0)
    assert df["depth_pA"].iloc[0] == pytest.approx(100.0, rel=1e-3)
    assert df["max_dev_pA"].iloc[0] == pytest.approx(900.0, rel=1e-3)


def test_many_events_vectorized_path_matches_naive_loop():
    """Guard the reduceat/lexsort machinery against an obvious per-event reference."""
    rng = np.random.default_rng(42)
    n_events = 500
    starts = np.cumsum(rng.integers(50, 300, n_events)) + 100
    lengths = rng.integers(11, 400, n_events)
    ends = starts + lengths
    starts = np.concatenate([[starts[0]], starts[1:] + np.cumsum(lengths)[:-1]])
    ends = starts + lengths
    n = int(ends[-1] + 100)

    cur, base, pol = build_trace(
        n=n,
        events=[(int(s), int(e), 100.0 + 0.5 * i) for i, (s, e) in enumerate(zip(starts, ends))],
        noise_pA=10.0,
        seed=1,
    )
    df = characterize(cur, base, starts, ends, FS, pol, noise_rms=10.0)

    cur64 = np.asarray(cur, dtype=np.float64)
    for i in rng.choice(n_events, 25, replace=False):
        s, e = int(starts[i]), int(ends[i])
        dev = -1.0 * (cur64[s:e] - base[s:e])
        assert df["depth_pA"].iloc[i] == pytest.approx(float(np.median(dev)), rel=1e-9)
        assert df["mean_dev_pA"].iloc[i] == pytest.approx(float(dev.mean()), rel=1e-9)
        assert df["max_dev_pA"].iloc[i] == pytest.approx(float(dev.max()), rel=1e-9)
        assert df["ecd_pA_s"].iloc[i] == pytest.approx(float(dev.sum()) / FS, rel=1e-9)
