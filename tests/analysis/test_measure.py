"""Tests for analysis.measure: hand-picked squares in, known numbers out.

A manual selection is measured against a locally-computed open-pore reference, so
the fixtures build a flat baseline with a square blockade and assert the amplitude
(by every method), dwell, baseline recovery, dI/I0, the shape statistics
(skewness/kurtosis on a known distribution), the fs=None degenerate case, and a
CSV round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nanoscope.analysis.measure import (
    AMPLITUDE_METHODS,
    MEASUREMENT_META_COLUMNS,
    MEASUREMENT_SCHEMA,
    UNKNOWN_CHANNEL_INDEX,
    append_measurement,
    coerce_measurements,
    empty_measurements_df,
    measure_selection,
    schema_default,
    write_measurements_csv,
)

FS = 100_000.0


def build_square(
    *,
    i0: float = 1000.0,
    level: float = 900.0,
    start: int = 5_000,
    stop: int = 7_000,
    n: int = 20_000,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A flat trace at `i0` with a square set to `level` over [start, stop)."""
    rng = np.random.default_rng(seed)
    x = np.full(n, float(i0), dtype=np.float64)
    x[start:stop] = float(level)
    if noise:
        x += rng.normal(0.0, noise, n)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_empty_df_matches_schema():
    df = empty_measurements_df()
    assert list(df.columns) == list(MEASUREMENT_SCHEMA)
    for col, (dtype, _, _) in MEASUREMENT_SCHEMA.items():
        assert df[col].dtype == pd.api.types.pandas_dtype(dtype), col
    assert len(df) == 0


def test_measure_returns_every_stat_key():
    """Every schema column that is NOT caller-stamped metadata comes out of the measurement.

    Keyed off MEASUREMENT_META_COLUMNS rather than a hard-coded list, so an additive schema
    change (identity/annotation columns) does not silently stop checking the stat columns.
    """
    x = build_square()
    m = measure_selection(x, FS, 5_000, 7_000)
    for key in MEASUREMENT_SCHEMA:
        if key in MEASUREMENT_META_COLUMNS:
            continue
        assert key in m, key


# ---------------------------------------------------------------------------
# a clean square: amplitude/dwell/baseline are analytic
# ---------------------------------------------------------------------------


def test_clean_square_dwell_and_baseline():
    x = build_square(i0=1000.0, level=900.0, start=5_000, stop=7_000)
    m = measure_selection(x, FS, 5_000, 7_000)
    assert m["n_samples"] == 2_000
    assert m["dwell_s"] == pytest.approx(2_000 / FS, rel=1e-12)
    assert m["t_start_s"] == pytest.approx(5_000 / FS, rel=1e-12)
    assert m["baseline"] == pytest.approx(1000.0, abs=1e-6)


@pytest.mark.parametrize("method", AMPLITUDE_METHODS)
def test_clean_square_amplitude_by_every_method(method):
    """On a noiseless square every level estimator coincides: amplitude == depth."""
    x = build_square(i0=1000.0, level=900.0)
    m = measure_selection(x, FS, 5_000, 7_000, amplitude_method=method)
    assert m["level"] == pytest.approx(900.0, abs=1e-6)
    assert m["amplitude"] == pytest.approx(100.0, abs=1e-6)
    assert m["frac_amplitude"] == pytest.approx(0.1, rel=1e-6)
    # ECD of a square is depth * dwell, positive charge deficit.
    assert m["area_ecd"] == pytest.approx(100.0 * 2_000 / FS, rel=1e-9)


def test_noisy_square_median_mean_agree_min_is_the_peak():
    """median ~= mean ~= level; 'min' picks the deepest current -> a bigger amplitude."""
    x = build_square(i0=1000.0, level=900.0, noise=8.0, seed=3)
    med = measure_selection(x, FS, 5_000, 7_000, amplitude_method="median")
    mean = measure_selection(x, FS, 5_000, 7_000, amplitude_method="mean")
    mode = measure_selection(x, FS, 5_000, 7_000, amplitude_method="mode")
    mn = measure_selection(x, FS, 5_000, 7_000, amplitude_method="min")

    assert med["amplitude"] == pytest.approx(100.0, abs=3.0)
    assert mean["amplitude"] == pytest.approx(100.0, abs=3.0)
    assert mode["amplitude"] == pytest.approx(100.0, abs=6.0)
    # The deepest sample is well below the level, so the min-amplitude is larger.
    assert mn["amplitude"] > med["amplitude"] + 2.0 * 8.0  # deepest sample is >2 sigma below level


def test_negative_open_pore_blockade_gives_positive_amplitude():
    """Real data sits at I0 < 0; a blockade moves current toward zero. Amplitude > 0."""
    x = build_square(i0=-1000.0, level=-900.0)
    m = measure_selection(x, FS, 5_000, 7_000)
    assert m["baseline"] == pytest.approx(-1000.0, abs=1e-6)
    assert m["amplitude"] == pytest.approx(100.0, abs=1e-6)
    assert m["frac_amplitude"] == pytest.approx(0.1, rel=1e-6)
    assert m["area_ecd"] > 0  # charge deficit stays positive


def test_explicit_baseline_and_polarity_are_honoured():
    x = build_square(i0=1000.0, level=900.0)
    m = measure_selection(x, FS, 5_000, 7_000, baseline=1000.0, polarity=1, noise_rms=5.0)
    assert m["baseline"] == 1000.0
    assert m["amplitude"] == pytest.approx(100.0, abs=1e-6)
    assert m["snr"] == pytest.approx(100.0 / 5.0, rel=1e-9)


# ---------------------------------------------------------------------------
# shape statistics on a known distribution
# ---------------------------------------------------------------------------


def test_skewness_and_kurtosis_are_finite_and_correct_signed():
    """A right-skewed, heavy-tailed selection: skew > 0 and excess kurtosis > 0.

    Build the in-selection samples from an exponential (population skew 2, excess
    kurtosis 6) and check measure reproduces scipy exactly and with the right sign.
    """
    rng = np.random.default_rng(11)
    n = 20_000
    x = np.full(n, 1000.0, dtype=np.float64)
    sel = 900.0 + rng.exponential(scale=10.0, size=2_000)  # right-skewed levels
    x[5_000:7_000] = sel
    xf = x.astype(np.float32)

    m = measure_selection(xf, FS, 5_000, 7_000)
    sel64 = np.asarray(xf[5_000:7_000], dtype=np.float64)

    assert np.isfinite(m["skewness"]) and np.isfinite(m["kurtosis"])
    assert m["skewness"] == pytest.approx(float(stats.skew(sel64)), rel=1e-9)
    assert m["kurtosis"] == pytest.approx(float(stats.kurtosis(sel64, fisher=True)), rel=1e-9)
    assert m["skewness"] > 0.5   # exponential is strongly right-skewed
    assert m["kurtosis"] > 1.0   # and leptokurtic


def test_basic_moments_match_numpy():
    x = build_square(noise=5.0, seed=7)
    m = measure_selection(x, FS, 5_000, 7_000)
    sel = np.asarray(x[5_000:7_000], dtype=np.float64)
    assert m["mean"] == pytest.approx(float(sel.mean()), rel=1e-9)
    assert m["median"] == pytest.approx(float(np.median(sel)), rel=1e-9)
    assert m["std"] == pytest.approx(float(sel.std()), rel=1e-9)
    assert m["var"] == pytest.approx(float(sel.var()), rel=1e-9)
    assert m["min"] == pytest.approx(float(sel.min()), rel=1e-9)
    assert m["max"] == pytest.approx(float(sel.max()), rel=1e-9)
    assert m["ptp"] == pytest.approx(float(sel.max() - sel.min()), rel=1e-9)
    assert m["rms"] == pytest.approx(float(np.sqrt(np.mean(sel ** 2))), rel=1e-9)


# ---------------------------------------------------------------------------
# fs = None: dwell undefined, amplitude still valid
# ---------------------------------------------------------------------------


def test_fs_none_gives_undefined_dwell_but_valid_amplitude():
    x = build_square(i0=1000.0, level=900.0)
    m = measure_selection(x, None, 5_000, 7_000)
    assert np.isnan(m["dwell_s"])
    assert np.isnan(m["t_start_s"])
    assert np.isnan(m["area_ecd"])
    assert m["n_samples"] == 2_000
    assert m["amplitude"] == pytest.approx(100.0, abs=1e-6)
    assert m["frac_amplitude"] == pytest.approx(0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, stop, match",
    [
        (7_000, 5_000, "greater than start"),
        (5_000, 5_000, "greater than start"),
        (-5, 100, "outside the trace"),
        (5_000, 10 ** 9, "outside the trace"),
        (5_000, 5_001, "at least"),  # 1 sample < _MIN_SELECTION_SAMPLES
    ],
)
def test_invalid_selection_raises(start, stop, match):
    x = build_square()
    with pytest.raises(ValueError, match=match):
        measure_selection(x, FS, start, stop)


def test_unknown_amplitude_method_raises():
    x = build_square()
    with pytest.raises(ValueError, match="amplitude_method"):
        measure_selection(x, FS, 5_000, 7_000, amplitude_method="p2p")


def test_bad_fs_raises():
    x = build_square()
    with pytest.raises(ValueError, match="fs must be positive"):
        measure_selection(x, -1.0, 5_000, 7_000)


# ---------------------------------------------------------------------------
# the table + CSV round-trip
# ---------------------------------------------------------------------------


def test_append_grows_table_and_autoassigns_id():
    x = build_square()
    df = empty_measurements_df()
    for i in range(3):
        m = measure_selection(x, FS, 5_000 + i, 7_000 + i)
        df = append_measurement(df, {**m, "recording": "rec", "channel": "0"})
    assert len(df) == 3
    assert list(df["measurement_id"]) == [0, 1, 2]
    assert (df["recording"] == "rec").all()
    assert list(df.columns) == list(MEASUREMENT_SCHEMA)
    # dtypes preserved through concat
    for col, (dtype, _, _) in MEASUREMENT_SCHEMA.items():
        assert df[col].dtype == pd.api.types.pandas_dtype(dtype), col


def test_csv_round_trips(tmp_path):
    x = build_square(i0=1000.0, level=900.0, noise=4.0, seed=1)
    df = empty_measurements_df()
    for method in ("median", "min"):
        m = measure_selection(x, FS, 5_000, 7_000, amplitude_method=method)
        df = append_measurement(df, {**m, "recording": "apoferritin", "channel": "0"})

    path = write_measurements_csv(df, tmp_path / "measurements.csv")
    assert path.exists()

    back = pd.read_csv(path)
    assert list(back.columns) == list(MEASUREMENT_SCHEMA)
    assert len(back) == 2
    np.testing.assert_allclose(back["amplitude"], df["amplitude"], rtol=1e-9)
    np.testing.assert_allclose(back["dwell_s"], df["dwell_s"], rtol=1e-12)
    np.testing.assert_allclose(back["kurtosis"], df["kurtosis"], rtol=1e-9)
    assert list(back["amplitude_method"]) == ["median", "min"]
    assert list(back["recording"]) == ["apoferritin", "apoferritin"]
