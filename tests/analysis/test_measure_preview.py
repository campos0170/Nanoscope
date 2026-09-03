"""Tests for the BOUNDED live preview of ``measure_selection`` (``max_samples``).

`measure_selection` is O(n) over the selection and the GUI calls it once per drag frame,
so on the 15 M-sample recordings this app exists for the unbounded version freezes the app
for ~1.1 s per frame. `max_samples` bounds that by measuring a strided subsample instead.
That is only acceptable if it is honest, so this file pins the whole contract:

  - **the exact path did not move** — the default (`max_samples=None`) output is asserted
    BIT-FOR-BIT against a table of hex floats captured from the pre-bound implementation,
    across every amplitude method, a tiny selection, the whole trace and the fs=None
    branch. Nothing below the bound may change by one ulp;
  - **a selection under the bound is not a preview** — same dict, same bits, `stride == 1`;
  - **above it the preview strides**, with the promised ratio;
  - **geometry stays exact** — n_samples/dwell_s/t_start_s/start_idx/end_idx describe the
    TRUE span, never the subsample;
  - **the reference stays exact** whenever the selection has flanks and fs is known;
  - **converging statistics land inside stated tolerances**, and
  - **statistics striding cannot estimate come back NaN**, listed in `suppressed`, rather
    than plausible-looking and wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.measure import (
    AMPLITUDE_METHODS,
    LIVE_PREVIEW_MAX_SAMPLES,
    MEASUREMENT_DIAGNOSTIC_KEYS,
    MEASUREMENT_SCHEMA,
    append_measurement,
    empty_measurements_df,
    measure_selection,
    preview_stride,
)

FS = 100_000.0

# Keys `measure_selection` returns that are pure geometry: the answer to "which samples did
# you mean", not "what do they say". A preview must reproduce every one of them exactly.
_GEOMETRY_KEYS = ("start_idx", "end_idx", "n_samples", "t_start_s", "dwell_s")


def golden_trace(seed: int = 7, n: int = 40_000) -> np.ndarray:
    """The fixture the golden table was captured on: 1000 pA open pore + 4 pA noise, one
    180 pA blockade over [12k, 19k), and a short 90 pA spike so min/max/kurtosis have teeth.
    Do NOT change it — the hex floats below were measured against exactly this array."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 1000.0) + rng.normal(0.0, 4.0, n)
    x[12_000:19_000] -= 180.0
    x[25_000:25_060] += 90.0
    return np.ascontiguousarray(x, dtype=np.float32)


def noisy_trace(n: int = 3_000_000, seed: int = 11) -> np.ndarray:
    """A long recording with many short blockades — big enough to force a real stride."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s in rng.choice(n - 400, size=n // 5_000, replace=False):
        x[s:s + 120] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def long_blockade_trace(n: int = 3_000_000, seed: int = 5) -> np.ndarray:
    """100 pA open pore with ONE 25 s, 30 pA-deep blockade over [250k, 2.75M).

    The fixture for the amplitude tolerances, because on `noisy_trace` the selection's level
    sits ON the baseline (short events, ~2 % duty), so `amplitude` is ~0 and ANY absolute
    error reads as an enormous relative one — a property of the fixture, not the estimator.
    Here the selected span IS the event, so amplitude is a solid 30 pA and a relative
    tolerance means something. A short deep spike inside it gives min/max/kurtosis teeth."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    x[250_000:2_750_000] -= 30.0
    x[1_000_000:1_000_050] -= 40.0
    return np.ascontiguousarray(x, dtype=np.float32)


def heavy_tailed_trace(n: int = 3_000_000, seed: int = 2) -> np.ndarray:
    """Thousands of short blockades: the "noisy multi-level mess" kurtosis exists to name."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s in rng.choice(n - 300, size=6_000, replace=False):
        x[s:s + 150] -= 25.0
    return np.ascontiguousarray(x, dtype=np.float32)


# The selection used by the tolerance tests: the whole blockade, flanked on both sides.
_SPAN = (250_000, 2_750_000)


# ================================================================================================
# INVARIANT: the exact path is bit-for-bit what it always was
# ================================================================================================
# Captured by running the PRE-CHANGE `measure_selection` over `golden_trace()`. Hex floats, not
# decimal literals, so the comparison is exact rather than "close enough to print the same".

_GOLDEN = {
    (12000, 19000, 'median', 100000.0): {
        "amplitude": float.fromhex("0x1.67b9740000000p+7"),
        "amplitude_method": 'median',
        "area_ecd": float.fromhex("0x1.92cf12c710cb3p+3"),
        "baseline": float.fromhex("0x1.f3eb8c0000000p+9"),
        "dwell_s": float.fromhex("0x1.1eb851eb851ecp-4"),
        "end_idx": 19000,
        "frac_amplitude": float.fromhex("0x1.706aaba8d4776p-3"),
        "kurtosis": float.fromhex("0x1.39b04353c3000p-8"),
        "level": float.fromhex("0x1.99fd2f0000000p+9"),
        "max": float.fromhex("0x1.a0a40e0000000p+9"),
        "mean": float.fromhex("0x1.9a01e585d9f74p+9"),
        "median": float.fromhex("0x1.99fd2f0000000p+9"),
        "min": float.fromhex("0x1.9303360000000p+9"),
        "n_samples": 7000,
        "ptp": float.fromhex("0x1.b41b000000000p+4"),
        "rms": float.fromhex("0x1.9a031a335934ap+9"),
        "skewness": float.fromhex("-0x1.7a4087ec28d8fp-6"),
        "snr": float.fromhex("0x1.4e72fe0017b71p+5"),
        "start_idx": 12000,
        "std": float.fromhex("0x1.f71cb0f639512p+1"),
        "t_start_s": float.fromhex("0x1.eb851eb851eb8p-4"),
        "units": 'pA',
        "var": float.fromhex("0x1.ee60e14f62b67p+3"),
    },
    (12000, 19000, 'mean', 100000.0): {
        "amplitude": float.fromhex("0x1.67a699e898230p+7"),
        "amplitude_method": 'mean',
        "area_ecd": float.fromhex("0x1.92cf12c710cb3p+3"),
        "baseline": float.fromhex("0x1.f3eb8c0000000p+9"),
        "dwell_s": float.fromhex("0x1.1eb851eb851ecp-4"),
        "end_idx": 19000,
        "frac_amplitude": float.fromhex("0x1.70575cf3bb1a2p-3"),
        "kurtosis": float.fromhex("0x1.39b04353c3000p-8"),
        "level": float.fromhex("0x1.9a01e585d9f74p+9"),
        "max": float.fromhex("0x1.a0a40e0000000p+9"),
        "mean": float.fromhex("0x1.9a01e585d9f74p+9"),
        "median": float.fromhex("0x1.99fd2f0000000p+9"),
        "min": float.fromhex("0x1.9303360000000p+9"),
        "n_samples": 7000,
        "ptp": float.fromhex("0x1.b41b000000000p+4"),
        "rms": float.fromhex("0x1.9a031a335934ap+9"),
        "skewness": float.fromhex("-0x1.7a4087ec28d8fp-6"),
        "snr": float.fromhex("0x1.4e617700e7689p+5"),
        "start_idx": 12000,
        "std": float.fromhex("0x1.f71cb0f639512p+1"),
        "t_start_s": float.fromhex("0x1.eb851eb851eb8p-4"),
        "units": 'pA',
        "var": float.fromhex("0x1.ee60e14f62b67p+3"),
    },
    (12000, 19000, 'min', 100000.0): {
        "amplitude": float.fromhex("0x1.83a1580000000p+7"),
        "amplitude_method": 'min',
        "area_ecd": float.fromhex("0x1.92cf12c710cb3p+3"),
        "baseline": float.fromhex("0x1.f3eb8c0000000p+9"),
        "dwell_s": float.fromhex("0x1.1eb851eb851ecp-4"),
        "end_idx": 19000,
        "frac_amplitude": float.fromhex("0x1.8cff2f03bd3bap-3"),
        "kurtosis": float.fromhex("0x1.39b04353c3000p-8"),
        "level": float.fromhex("0x1.9303360000000p+9"),
        "max": float.fromhex("0x1.a0a40e0000000p+9"),
        "mean": float.fromhex("0x1.9a01e585d9f74p+9"),
        "median": float.fromhex("0x1.99fd2f0000000p+9"),
        "min": float.fromhex("0x1.9303360000000p+9"),
        "n_samples": 7000,
        "ptp": float.fromhex("0x1.b41b000000000p+4"),
        "rms": float.fromhex("0x1.9a031a335934ap+9"),
        "skewness": float.fromhex("-0x1.7a4087ec28d8fp-6"),
        "snr": float.fromhex("0x1.6864eee65a359p+5"),
        "start_idx": 12000,
        "std": float.fromhex("0x1.f71cb0f639512p+1"),
        "t_start_s": float.fromhex("0x1.eb851eb851eb8p-4"),
        "units": 'pA',
        "var": float.fromhex("0x1.ee60e14f62b67p+3"),
    },
    (12000, 19000, 'max', 100000.0): {
        "amplitude": float.fromhex("0x1.4d1df80000000p+7"),
        "amplitude_method": 'max',
        "area_ecd": float.fromhex("0x1.92cf12c710cb3p+3"),
        "baseline": float.fromhex("0x1.f3eb8c0000000p+9"),
        "dwell_s": float.fromhex("0x1.1eb851eb851ecp-4"),
        "end_idx": 19000,
        "frac_amplitude": float.fromhex("0x1.552a9888d0cbfp-3"),
        "kurtosis": float.fromhex("0x1.39b04353c3000p-8"),
        "level": float.fromhex("0x1.a0a40e0000000p+9"),
        "max": float.fromhex("0x1.a0a40e0000000p+9"),
        "mean": float.fromhex("0x1.9a01e585d9f74p+9"),
        "median": float.fromhex("0x1.99fd2f0000000p+9"),
        "min": float.fromhex("0x1.9303360000000p+9"),
        "n_samples": 7000,
        "ptp": float.fromhex("0x1.b41b000000000p+4"),
        "rms": float.fromhex("0x1.9a031a335934ap+9"),
        "skewness": float.fromhex("-0x1.7a4087ec28d8fp-6"),
        "snr": float.fromhex("0x1.35b61a04f9408p+5"),
        "start_idx": 12000,
        "std": float.fromhex("0x1.f71cb0f639512p+1"),
        "t_start_s": float.fromhex("0x1.eb851eb851eb8p-4"),
        "units": 'pA',
        "var": float.fromhex("0x1.ee60e14f62b67p+3"),
    },
    (12000, 19000, 'mode', 100000.0): {
        "amplitude": float.fromhex("0x1.685fa80000000p+7"),
        "amplitude_method": 'mode',
        "area_ecd": float.fromhex("0x1.92cf12c710cb3p+3"),
        "baseline": float.fromhex("0x1.f3eb8c0000000p+9"),
        "dwell_s": float.fromhex("0x1.1eb851eb851ecp-4"),
        "end_idx": 19000,
        "frac_amplitude": float.fromhex("0x1.7114e3c64703dp-3"),
        "kurtosis": float.fromhex("0x1.39b04353c3000p-8"),
        "level": float.fromhex("0x1.99d3a20000000p+9"),
        "max": float.fromhex("0x1.a0a40e0000000p+9"),
        "mean": float.fromhex("0x1.9a01e585d9f74p+9"),
        "median": float.fromhex("0x1.99fd2f0000000p+9"),
        "min": float.fromhex("0x1.9303360000000p+9"),
        "n_samples": 7000,
        "ptp": float.fromhex("0x1.b41b000000000p+4"),
        "rms": float.fromhex("0x1.9a031a335934ap+9"),
        "skewness": float.fromhex("-0x1.7a4087ec28d8fp-6"),
        "snr": float.fromhex("0x1.4f0d8475a9bb0p+5"),
        "start_idx": 12000,
        "std": float.fromhex("0x1.f71cb0f639512p+1"),
        "t_start_s": float.fromhex("0x1.eb851eb851eb8p-4"),
        "units": 'pA',
        "var": float.fromhex("0x1.ee60e14f62b67p+3"),
    },
    (0, 40000, 'median', 100000.0): {
        "amplitude": float.fromhex("0x0.0p+0"),
        "amplitude_method": 'median',
        "area_ecd": float.fromhex("0x1.83e1c350b0f28p+3"),
        "baseline": float.fromhex("0x1.f3740e0000000p+9"),
        "dwell_s": float.fromhex("0x1.999999999999ap-2"),
        "end_idx": 40000,
        "frac_amplitude": float.fromhex("0x0.0p+0"),
        "kurtosis": float.fromhex("0x1.d3b84c0a38b60p-1"),
        "level": float.fromhex("0x1.f3740e0000000p+9"),
        "max": float.fromhex("0x1.12b3b60000000p+10"),
        "mean": float.fromhex("0x1.e44d3c5ed9168p+9"),
        "median": float.fromhex("0x1.f3740e0000000p+9"),
        "min": float.fromhex("0x1.9303360000000p+9"),
        "n_samples": 40000,
        "ptp": float.fromhex("0x1.24c86c0000000p+8"),
        "rms": float.fromhex("0x1.e584248875557p+9"),
        "skewness": float.fromhex("-0x1.b03e30ca6cf6fp+0"),
        "snr": float.fromhex("0x0.0p+0"),
        "start_idx": 0,
        "std": float.fromhex("0x1.128e3a3cb2cbap+6"),
        "t_start_s": float.fromhex("0x0.0p+0"),
        "units": 'pA',
        "var": float.fromhex("0x1.2674c3ae9751cp+12"),
    },
    (11990, 12003, 'median', 100000.0): {
        "amplitude": float.fromhex("0x1.697e800000000p+6"),
        "amplitude_method": 'median',
        "area_ecd": float.fromhex("0x1.aedb549f94856p-8"),
        "baseline": float.fromhex("0x1.c679f40000000p+9"),
        "dwell_s": float.fromhex("0x1.10a137f38c543p-13"),
        "end_idx": 12003,
        "frac_amplitude": float.fromhex("0x1.973fb71ae94f0p-4"),
        "kurtosis": float.fromhex("-0x1.6770242d982e0p-2"),
        "level": float.fromhex("0x1.f3a9c40000000p+9"),
        "max": float.fromhex("0x1.f7a0600000000p+9"),
        "mean": float.fromhex("0x1.dfc3296276276p+9"),
        "median": float.fromhex("0x1.f3a9c40000000p+9"),
        "min": float.fromhex("0x1.97388c0000000p+9"),
        "n_samples": 13,
        "ptp": float.fromhex("0x1.819f500000000p+7"),
        "rms": float.fromhex("0x1.e1479f802afd9p+9"),
        "skewness": float.fromhex("-0x1.46d24c18a47cfp+0"),
        "snr": float.fromhex("0x1.557841757b2b7p-1"),
        "start_idx": 11990,
        "std": float.fromhex("0x1.3180b54f852fap+6"),
        "t_start_s": float.fromhex("0x1.eb1c432ca57a8p-4"),
        "units": 'pA',
        "var": float.fromhex("0x1.6c93f0bd4b4a4p+12"),
    },
    (25000, 25060, 'mode', 50000.0): {
        "amplitude": float.fromhex("0x1.66007e0000000p+6"),
        "amplitude_method": 'mode',
        "area_ecd": float.fromhex("0x1.bddb33daf8df8p-4"),
        "baseline": float.fromhex("0x1.f393e80000000p+9"),
        "dwell_s": float.fromhex("0x1.3a92a30553261p-10"),
        "end_idx": 25060,
        "frac_amplitude": float.fromhex("0x1.6ee76042b6defp-4"),
        "kurtosis": float.fromhex("-0x1.7cbbd5f2adf84p-1"),
        "level": float.fromhex("0x1.1029fbe000000p+10"),
        "max": float.fromhex("0x1.12b3b60000000p+10"),
        "mean": float.fromhex("0x1.10775f8888889p+10"),
        "median": float.fromhex("0x1.1057360000000p+10"),
        "min": float.fromhex("0x1.0decb20000000p+10"),
        "n_samples": 60,
        "ptp": float.fromhex("0x1.31c1000000000p+4"),
        "rms": float.fromhex("0x1.107802d19a7e8p+10"),
        "skewness": float.fromhex("0x1.1c6ca99f840cap-8"),
        "snr": float.fromhex("0x1.90e267f54e944p+4"),
        "start_idx": 25000,
        "std": float.fromhex("0x1.2a4b8b17ad12cp+2"),
        "t_start_s": float.fromhex("0x1.0000000000000p-1"),
        "units": 'pA',
        "var": float.fromhex("0x1.5b93f61dea61dp+4"),
    },
    (12000, 19000, 'median', None): {
        "amplitude": float.fromhex("0x1.680b840000000p+7"),
        "amplitude_method": 'median',
        "area_ecd": float.fromhex("nan"),
        "baseline": float.fromhex("0x1.f400100000000p+9"),
        "dwell_s": float.fromhex("nan"),
        "end_idx": 19000,
        "frac_amplitude": float.fromhex("0x1.70af95fe7a2d2p-3"),
        "kurtosis": float.fromhex("0x1.39b04353c3000p-8"),
        "level": float.fromhex("0x1.99fd2f0000000p+9"),
        "max": float.fromhex("0x1.a0a40e0000000p+9"),
        "mean": float.fromhex("0x1.9a01e585d9f74p+9"),
        "median": float.fromhex("0x1.99fd2f0000000p+9"),
        "min": float.fromhex("0x1.9303360000000p+9"),
        "n_samples": 7000,
        "ptp": float.fromhex("0x1.b41b000000000p+4"),
        "rms": float.fromhex("0x1.9a031a335934ap+9"),
        "skewness": float.fromhex("-0x1.7a4087ec28d8fp-6"),
        "snr": float.fromhex("0x1.6ba4bd24f99d3p+5"),
        "start_idx": 12000,
        "std": float.fromhex("0x1.f71cb0f639512p+1"),
        "t_start_s": float.fromhex("nan"),
        "units": 'pA',
        "var": float.fromhex("0x1.ee60e14f62b67p+3"),
    },
}


@pytest.mark.parametrize("case", list(_GOLDEN), ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}-fs{c[3]}")
def test_default_path_is_bit_identical_to_the_pre_bound_implementation(case):
    """Adding `max_samples` must not have moved the exact answer by one ulp."""
    start, stop, method, fs = case
    out = measure_selection(golden_trace(), fs, start, stop,
                            amplitude_method=method, units="pA")
    for key, want in _GOLDEN[case].items():
        got = out[key]
        if isinstance(want, float) and np.isnan(want):
            assert np.isnan(got), key
        else:
            assert got == want, f"{key}: {got!r} != {want!r}"


def test_the_exact_path_reports_itself_as_exact():
    out = measure_selection(golden_trace(), FS, 12_000, 19_000)
    assert out["approximate"] is False
    assert out["stride"] == 1
    assert out["n_measured"] == out["n_samples"] == 7_000
    assert out["suppressed"] == ()


# ================================================================================================
# INVARIANT: a selection under the bound is not a preview
# ================================================================================================

@pytest.mark.parametrize("method", AMPLITUDE_METHODS)
def test_below_the_bound_output_is_bit_identical_to_the_unbounded_call(method):
    """The whole point of the threshold: nothing changes for a normal hand-measured event."""
    x = golden_trace()
    exact = measure_selection(x, FS, 12_000, 19_000, amplitude_method=method, units="pA")
    bounded = measure_selection(x, FS, 12_000, 19_000, amplitude_method=method, units="pA",
                                max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert set(bounded) == set(exact)
    for key, want in exact.items():
        got = bounded[key]
        if isinstance(want, float) and np.isnan(want):
            assert np.isnan(got), key
        else:
            assert got == want, key
    assert bounded["stride"] == 1 and bounded["approximate"] is False


def test_a_selection_exactly_at_the_bound_still_does_not_stride():
    x = noisy_trace(n=LIVE_PREVIEW_MAX_SAMPLES + 5_000)
    out = measure_selection(x, FS, 0, LIVE_PREVIEW_MAX_SAMPLES,
                            max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert out["stride"] == 1 and out["approximate"] is False
    assert out["n_measured"] == LIVE_PREVIEW_MAX_SAMPLES


def test_one_sample_over_the_bound_strides():
    x = noisy_trace(n=LIVE_PREVIEW_MAX_SAMPLES + 5_000)
    out = measure_selection(x, FS, 0, LIVE_PREVIEW_MAX_SAMPLES + 1,
                            max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert out["stride"] == 2 and out["approximate"] is True


# ================================================================================================
# the stride arithmetic
# ================================================================================================

@pytest.mark.parametrize("n,cap,want", [
    (100, None, 1),        # no bound at all
    (100, 200, 1),         # under the bound
    (200, 200, 1),         # exactly at it
    (201, 200, 2),
    (1_000, 200, 5),
    (1_001, 200, 6),       # ceil, so the subsample never EXCEEDS the bound
    (15_000_000, 200_000, 75),
])
def test_preview_stride_arithmetic(n, cap, want):
    assert preview_stride(n, cap) == want


@pytest.mark.parametrize("n,cap", [(1_000, 200), (1_001, 200), (15_000_000, 200_000)])
def test_the_subsample_never_exceeds_the_bound(n, cap):
    assert len(range(0, n, preview_stride(n, cap))) <= cap


@pytest.mark.parametrize("bad", [0, 1, -5])
def test_a_bound_that_cannot_hold_two_samples_raises(bad):
    with pytest.raises(ValueError, match="max_samples"):
        preview_stride(1_000, bad)
    with pytest.raises(ValueError, match="max_samples"):
        measure_selection(golden_trace(), FS, 12_000, 19_000, max_samples=bad)


def test_the_span_validation_is_unchanged_by_the_bound():
    """A bad span is still a bad span — the bound never rescues or masks one."""
    x = golden_trace()
    for kwargs in ({}, {"max_samples": 1_000}):
        with pytest.raises(ValueError, match="greater than"):
            measure_selection(x, FS, 500, 500, **kwargs)
        with pytest.raises(ValueError, match="outside the trace"):
            measure_selection(x, FS, 0, x.size + 1, **kwargs)
        with pytest.raises(ValueError, match="at least"):
            measure_selection(x, FS, 10, 11, **kwargs)


# ================================================================================================
# INVARIANT: geometry is exact regardless
# ================================================================================================

def test_preview_geometry_describes_the_true_span_not_the_subsample():
    x = noisy_trace()
    start, stop = 250_000, 2_750_000
    exact = measure_selection(x, FS, start, stop)
    preview = measure_selection(x, FS, start, stop, max_samples=LIVE_PREVIEW_MAX_SAMPLES)

    assert preview["approximate"] is True and preview["stride"] > 1
    for key in _GEOMETRY_KEYS:
        assert preview[key] == exact[key], key
    # n_samples is the SPAN; n_measured is what was read, and they differ by the stride.
    assert preview["n_samples"] == stop - start
    assert preview["n_measured"] < preview["n_samples"] <= LIVE_PREVIEW_MAX_SAMPLES * 100
    assert preview["dwell_s"] == pytest.approx((stop - start) / FS, rel=1e-12)


def test_preview_dwell_is_not_scaled_down_by_the_stride():
    """The obvious bug this invariant exists to catch: reporting the SUBSAMPLE's duration."""
    x = noisy_trace()
    preview = measure_selection(x, FS, 0, 2_000_000, max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert preview["dwell_s"] == pytest.approx(20.0, rel=1e-12)
    assert preview["n_measured"] != preview["n_samples"]


# ================================================================================================
# INVARIANT: the reference (baseline/noise/polarity) stays exact when it can
# ================================================================================================

def test_preview_baseline_and_noise_are_exact_when_the_selection_has_flanks():
    """The flanks are already capped to a few ms, so subsampling never touches them."""
    x = noisy_trace()
    start, stop = 250_000, 2_750_000
    exact = measure_selection(x, FS, start, stop)
    preview = measure_selection(x, FS, start, stop, max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert preview["baseline"] == exact["baseline"]          # bit-identical, not merely close
    # snr's denominator is the flank noise; recovering it from amplitude/snr proves it too.
    assert (exact["amplitude"] / exact["snr"]) == pytest.approx(
        preview["amplitude"] / preview["snr"], rel=1e-6)


def test_a_flankless_whole_trace_selection_is_still_bounded_and_close():
    """"Snap to view" on a fully zoomed-out recording: no flanks, so both references fall
    back to the whole trace. That fallback must be strided too, or the bound buys nothing."""
    x = noisy_trace()
    exact = measure_selection(x, FS, 0, x.size)
    preview = measure_selection(x, FS, 0, x.size, max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert preview["approximate"] is True
    assert preview["n_samples"] == x.size                      # geometry still exact
    assert preview["baseline"] == pytest.approx(exact["baseline"], rel=1e-3)


def test_fs_none_preview_clamps_the_flanks():
    """With no time base there is no 3 ms cap, so the flank windows track the selection and
    the preview must clamp them itself; times are NaN either way."""
    x = noisy_trace()
    preview = measure_selection(x, None, 1_000_000, 2_000_000,
                                max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert preview["approximate"] is True
    assert np.isnan(preview["dwell_s"]) and np.isnan(preview["area_ecd"])
    assert preview["n_samples"] == 1_000_000
    assert np.isfinite(preview["baseline"])


# ================================================================================================
# INVARIANT: converging statistics are within a stated tolerance
# ================================================================================================
# The tolerances below are the WORST relative error measured over ten seeds of
# `long_blockade_trace` at the real bound (stride 13, 2.5 M -> ~192 k samples), rounded up for
# headroom. They are claims about the ESTIMATOR, not fits to one seed:
#
#   mean / median / rms / level (median|mean)     1.5e-4      (~sigma/sqrt(m))
#   amplitude / frac_amplitude / snr / area_ecd   3.6e-4      (all are affine in the level)
#   std                                           4.0e-3      (~sqrt(2/m))
#   var                                           8.0e-3      (twice std's, being its square)
#   level / amplitude with the 'mode' estimator   2.0e-2 / 5.0e-2   (see its own test)
#
# `baseline` is absent because it is not approximated at all — see the reference tests above.
#
# The per-statistic tests below run ONE seed, which is a fixture check, not a claim about the
# estimator. `test_the_documented_tolerances_hold_over_ten_seeds` is the claim: it sweeps the
# same ten seeds the module docstring quotes and asserts the DOCUMENTED numbers, so the table
# in `analysis.measure`'s docstring cannot quietly drift away from what the code does.

# The module docstring's table, verbatim. Keep the two in sync — that is the point.
_DOC_TOLERANCE = {
    "mean": 1.5e-4, "median": 1.5e-4, "rms": 1.5e-4, "level": 1.5e-4,
    "amplitude": 3.6e-4, "frac_amplitude": 3.6e-4, "snr": 3.6e-4, "area_ecd": 3.6e-4,
    "std": 4.0e-3, "var": 8.0e-3,
    # `long_blockade_trace` carries a 50-sample spike, i.e. exactly the "tail from ONE rare
    # burst" case the docstring says costs up to ~20 % — not the <1 % bulk-structure case
    # (`heavy_tailed_trace`, pinned at 5e-2 by its own test above).
    "skewness": 2.0e-1, "kurtosis": 2.0e-1,
}
_DOC_TOLERANCE_MODE = {
    "level": 2.0e-2, "amplitude": 5.0e-2, "frac_amplitude": 5.0e-2, "snr": 5.0e-2,
}


@pytest.mark.parametrize("method,table", [("median", _DOC_TOLERANCE),
                                          ("mode", _DOC_TOLERANCE_MODE)])
def test_the_documented_tolerances_hold_over_ten_seeds(method, table):
    """Every number in the module docstring's error table, re-measured.

    One seed can be lucky; the docstring says "worst over ten seeds", so assert exactly that.
    `baseline` is checked for BIT-equality, not a tolerance — with a time base and flanks it
    is not approximated at all, and a preview that started estimating it would be a silent
    change of contract that no tolerance test would notice."""
    worst: dict[str, float] = {}
    for seed in range(10):
        x = long_blockade_trace(seed=seed)
        exact = measure_selection(x, FS, *_SPAN, amplitude_method=method)
        preview = measure_selection(x, FS, *_SPAN, amplitude_method=method,
                                    max_samples=LIVE_PREVIEW_MAX_SAMPLES)
        assert preview["stride"] > 1
        assert preview["baseline"] == exact["baseline"], f"seed {seed}: baseline moved"
        for key in table:
            got, want = float(preview[key]), float(exact[key])
            assert np.isfinite(got) and np.isfinite(want), key
            worst[key] = max(worst.get(key, 0.0), abs(got - want) / abs(want))
    over = {k: (v, table[k]) for k, v in worst.items() if v > table[k]}
    assert not over, f"{method}: outside the DOCUMENTED tolerance — {over}"

def test_preview_central_statistics_are_within_tolerance():
    x = long_blockade_trace()
    exact = measure_selection(x, FS, *_SPAN)
    preview = measure_selection(x, FS, *_SPAN, max_samples=LIVE_PREVIEW_MAX_SAMPLES)

    assert preview["stride"] > 1
    assert preview["mean"] == pytest.approx(exact["mean"], rel=1e-3)
    assert preview["median"] == pytest.approx(exact["median"], rel=1e-3)
    assert preview["rms"] == pytest.approx(exact["rms"], rel=1e-3)
    assert preview["level"] == pytest.approx(exact["level"], rel=1e-3)
    assert preview["std"] == pytest.approx(exact["std"], rel=2e-2)
    assert preview["var"] == pytest.approx(exact["var"], rel=4e-2)


def test_preview_amplitude_is_within_tolerance():
    """The headline number, on a selection where it is a real 30 pA and not ~0."""
    x = long_blockade_trace()
    exact = measure_selection(x, FS, *_SPAN)
    preview = measure_selection(x, FS, *_SPAN, max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert exact["amplitude"] == pytest.approx(30.0, abs=0.5)
    for key in ("amplitude", "frac_amplitude", "snr"):
        assert preview[key] == pytest.approx(exact[key], rel=2e-3), key


def test_preview_area_ecd_integrates_over_the_true_span():
    """area_ecd is an INTEGRAL: summing a 1-in-13 subsample would under-report it 13-fold.
    The estimator scales the subsample's mean by the TRUE n_samples instead."""
    x = long_blockade_trace()
    exact = measure_selection(x, FS, *_SPAN)
    preview = measure_selection(x, FS, *_SPAN, max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert preview["stride"] >= 10
    assert preview["area_ecd"] == pytest.approx(exact["area_ecd"], rel=2e-3)
    # ...and emphatically not off by a factor of the stride.
    assert abs(preview["area_ecd"]) > abs(exact["area_ecd"]) / 2


def test_preview_preserves_the_shape_verdict_kurtosis_is_asked_for():
    """Skew/kurtosis are moment estimators, not order statistics, so they converge — and on
    the case they exist to name (a heavy-tailed, multi-level selection) the preview agrees
    to well under a percent. They are noisier when the tail comes from a single rare burst
    of a few dozen samples: the subsample catches only a handful of them, so the VALUE can
    run ~15 % high even though the verdict ("very heavy-tailed") is unchanged. Reported,
    not blanked — unlike min/max they are unbiased, just noisy."""
    x = heavy_tailed_trace()
    exact = measure_selection(x, FS, *_SPAN)
    preview = measure_selection(x, FS, *_SPAN, max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert exact["kurtosis"] > 1.0 and exact["skewness"] < -1.0   # the fixture is what we think
    assert preview["skewness"] == pytest.approx(exact["skewness"], rel=5e-2)
    assert preview["kurtosis"] == pytest.approx(exact["kurtosis"], rel=5e-2)


@pytest.mark.parametrize("method", ["median", "mean"])
def test_central_level_estimators_survive_striding(method):
    x = long_blockade_trace()
    exact = measure_selection(x, FS, *_SPAN, amplitude_method=method)
    preview = measure_selection(x, FS, *_SPAN, amplitude_method=method,
                                max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert np.isfinite(preview["level"]) and np.isfinite(preview["amplitude"])
    assert preview["level"] == pytest.approx(exact["level"], rel=1e-3)


def test_the_mode_level_is_coarser_but_still_usable():
    """'mode' is quantised to a histogram bin whose edges come from the observed min/max, and
    those move under striding — so it drifts by a bin or two where median/mean drift by
    ~1e-4. Still a level, still shown; just an order of magnitude looser."""
    x = long_blockade_trace()
    exact = measure_selection(x, FS, *_SPAN, amplitude_method="mode")
    preview = measure_selection(x, FS, *_SPAN, amplitude_method="mode",
                                max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert np.isfinite(preview["level"]) and np.isfinite(preview["amplitude"])
    assert preview["level"] == pytest.approx(exact["level"], rel=2e-2)
    assert preview["amplitude"] == pytest.approx(exact["amplitude"], rel=5e-2)


# ================================================================================================
# INVARIANT: what striding cannot estimate is BLANKED, not guessed
# ================================================================================================

@pytest.mark.parametrize("method", ["median", "mean", "mode"])
def test_order_statistics_are_suppressed_in_a_preview(method):
    x = noisy_trace()
    preview = measure_selection(x, FS, 250_000, 2_750_000, amplitude_method=method,
                                max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert set(preview["suppressed"]) == {"min", "max", "ptp"}
    for key in ("min", "max", "ptp"):
        assert np.isnan(preview[key]), key


@pytest.mark.parametrize("method", ["min", "max"])
def test_an_extreme_level_estimator_suppresses_the_amplitude_too(method):
    """With method 'min'/'max' the LEVEL is the very order statistic striding misses, so the
    headline amplitude would be systematically shallow. Blank it rather than mislead."""
    x = noisy_trace()
    preview = measure_selection(x, FS, 250_000, 2_750_000, amplitude_method=method,
                                max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert set(preview["suppressed"]) == {
        "min", "max", "ptp", "level", "amplitude", "frac_amplitude", "snr"}
    for key in preview["suppressed"]:
        assert np.isnan(preview[key]), key
    # ...but the statistics that DO converge are still there.
    for key in ("mean", "median", "std", "var", "rms", "baseline", "area_ecd"):
        assert np.isfinite(preview[key]), key


def test_the_extreme_bias_is_real_which_is_why_it_is_suppressed():
    """Sanity-check the premise: an unsuppressed strided max really is biased inward."""
    x = noisy_trace()
    exact = measure_selection(x, FS, 250_000, 2_750_000)
    stride = preview_stride(2_500_000, LIVE_PREVIEW_MAX_SAMPLES)
    naive_max = float(np.max(x[250_000:2_750_000:stride]))
    assert naive_max < exact["max"]


def test_nothing_is_suppressed_below_the_bound():
    x = golden_trace()
    out = measure_selection(x, FS, 12_000, 19_000, amplitude_method="min",
                            max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert out["suppressed"] == ()
    for key in ("min", "max", "ptp", "level", "amplitude", "snr"):
        assert np.isfinite(out[key]), key


# ================================================================================================
# INVARIANT: the diagnostics never reach a saved row
# ================================================================================================

def test_diagnostic_keys_are_not_schema_columns():
    assert not set(MEASUREMENT_DIAGNOSTIC_KEYS) & set(MEASUREMENT_SCHEMA)


def test_append_measurement_drops_the_preview_diagnostics():
    """There is no saved column in which "this row was estimated" could be true, because a
    saved row never is. append_measurement builds from the schema, so they fall away."""
    x = noisy_trace()
    preview = measure_selection(x, FS, 250_000, 2_750_000,
                                max_samples=LIVE_PREVIEW_MAX_SAMPLES)
    assert preview["approximate"] is True
    df = append_measurement(empty_measurements_df(),
                            {**preview, "recording": "rec", "channel": "0"})
    assert list(df.columns) == list(MEASUREMENT_SCHEMA)
    for key in MEASUREMENT_DIAGNOSTIC_KEYS:
        assert key not in df.columns
