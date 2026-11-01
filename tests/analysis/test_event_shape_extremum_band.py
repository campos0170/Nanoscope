"""Event ensemble: alignment on the extremum, and the variability band (pure numpy).

Synthetic traces whose deepest sample is placed by hand, so the expected alignment is
known exactly: a blockade that LOWERS the current (polarity -1, the usual positive-I0
pore) and one that RAISES it (polarity +1: a negative-I0 blockade, or an enhancement).
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import event_shape
from nanoscope.analysis.events import SCHEMA, EventSet, empty_df

FS = 50_000.0


def _eventset(spans, *, baseline=1000.0, provenance=None, accepted=None) -> EventSet:
    df = empty_df()
    for i, (s, e) in enumerate(spans):
        df.loc[i] = {
            "event_id": i, "recording": "synthetic", "pore_id": "P",
            "start_idx": s, "end_idx": e, "t_start_s": s / FS, "dwell_s": (e - s) / FS,
            "baseline_pA": baseline, "depth_pA": 50.0, "frac_depth": 0.05,
            "ecd_pA_s": 0.0, "max_dev_pA": 50.0, "mean_dev_pA": 25.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 10.0, "n_levels": 1,
            "filter_limited": False,
            "accepted": True if accepted is None else bool(accepted[i]), "flag": "",
        }
    df = df.astype({c: d for c, (d, _, _) in SCHEMA.items()})
    return EventSet(df, dict(provenance or {}))


def _trace_with_dips(sign: float):
    """Five events of different dwell, each with ONE deepest sample at a known offset.

    ``sign=-1`` dips the current below the 1000 baseline; ``+1`` lifts it above.
    """
    sig = np.full(20_000, 1000.0)
    spans, deepest = [], []
    for i, (length, at) in enumerate(((40, 3), (90, 80), (60, 30), (150, 149), (25, 0))):
        s = 1_000 + i * 3_000
        sig[s:s + length] += sign * 30.0          # the plateau
        sig[s + at] += sign * 80.0                # the deepest sample
        spans.append((s, s + length))
        deepest.append(s + at)
    return sig, spans, np.asarray(deepest)


@pytest.mark.parametrize("sign, polarity", [(-1.0, -1), (+1.0, +1)])
def test_extremum_indices_follow_the_table_polarity(sign, polarity):
    sig, spans, deepest = _trace_with_dips(sign)
    starts, ends = (np.asarray(v) for v in zip(*spans))
    np.testing.assert_array_equal(
        event_shape.extremum_indices(sig, starts, ends, polarity), deepest)
    # The opposite polarity reads the opposite side: never the injected deepest sample.
    wrong = event_shape.extremum_indices(sig, starts, ends, -polarity)
    assert not np.any(wrong == deepest)


@pytest.mark.parametrize("sign, polarity", [(-1.0, -1), (+1.0, +1)])
def test_extremum_alignment_puts_every_deepest_sample_at_t0(sign, polarity):
    sig, spans, deepest = _trace_with_dips(sign)
    es = _eventset(spans, provenance={"polarity": polarity})
    longest = max(e - s for s, e in spans)
    # Points = window length, so every resampled point is an original sample.
    n_points = 2 * longest
    T, matrix, mean, median = event_shape.event_ensemble(
        sig, FS, es, n_points=n_points, pad_frac=0.0, align="extremum",
        normalize_time=False)
    assert matrix.shape == (5, n_points)
    zero = int(np.flatnonzero(T == 0.0)[0])
    peak = 1000.0 + sign * 110.0
    np.testing.assert_allclose(matrix[:, zero], peak)
    np.testing.assert_allclose(mean[zero], peak)
    # T is seconds about the extremum: the window reaches a full longest dwell to both sides.
    assert T[0] == pytest.approx(-longest / FS)
    # Every event lies wholly inside its row: the plateau samples on both sides of each
    # deepest sample are all present (count the samples off the baseline per row).
    for row, (s, e) in zip(matrix, spans):
        assert np.count_nonzero(np.abs(row - 1000.0) > 1.0) == e - s


def test_extremum_without_a_recorded_polarity_measures_it_from_the_trace():
    rng = np.random.default_rng(3)
    sig, spans, deepest = _trace_with_dips(-1.0)
    sig = sig + rng.normal(0.0, 0.5, sig.size)
    es = _eventset(spans, provenance={})
    polarity, source = event_shape.ensemble_polarity(sig, es)
    assert polarity == -1 and "measured from the trace" in source
    recorded = event_shape.ensemble_polarity(sig, _eventset(spans, provenance={"polarity": 1}))
    assert recorded == (1, "the polarity the event table was detected with")


def test_extremum_skips_events_without_a_finite_sample_and_off_the_edge():
    sig, spans, _deepest = _trace_with_dips(-1.0)
    s, e = spans[2]
    sig[s:e] = np.nan                               # nothing to align this one on
    spans = spans + [(5, 30)]                        # window would run off the trace start
    es = _eventset(spans, provenance={"polarity": -1})
    _T, matrix, _mean, _median = event_shape.event_ensemble(
        sig, FS, es, n_points=64, pad_frac=0.1, align="extremum", normalize_time=False)
    assert matrix.shape[0] == len(spans) - 2


def test_start_and_center_are_unchanged_by_the_shared_window():
    sig, spans, _ = _trace_with_dips(-1.0)
    es = _eventset(spans, provenance={"polarity": -1})
    starts, ends = (np.asarray(v) for v in zip(*spans))
    a_all, total, ref, valid = event_shape.realtime_window(starts, ends, 0.2, "start")
    width = int((ends - starts).max())
    wpad = int(round(0.2 * width))
    assert total == width + 2 * wpad and ref == float(wpad)
    np.testing.assert_array_equal(a_all, starts - wpad)
    assert valid.all()
    T, matrix, _m, _md = event_shape.event_ensemble(
        sig, FS, es, n_points=50, pad_frac=0.2, align="center", normalize_time=False)
    assert matrix.shape == (5, 50) and T.shape == (50,)
    with pytest.raises(ValueError, match="align must be one of"):
        event_shape.event_ensemble(sig, FS, es, n_points=50, align="peak",
                                   normalize_time=False)


def _base_realtime_rows(signal, fs, starts, ends, n_points, pad_frac, align):
    """The 650403e kernel's real-time window, inline: start / centre must not move."""
    n = signal.size
    lengths = ends - starts
    valid = lengths > 0
    width = int(lengths[valid].max())
    wpad = int(round(pad_frac * width))
    total = width + 2 * wpad
    if align == "center":
        a_all = np.round((starts + ends) / 2.0 - total / 2.0).astype(np.int64)
        ref = total / 2.0
    else:
        a_all = starts - wpad
        ref = float(wpad)
    src = np.arange(total, dtype=np.float64)
    tgt = np.linspace(0.0, total - 1, n_points)
    T = (tgt - ref) / fs
    rows = [np.interp(tgt, src, signal[a:a + total])
            for a, ok in zip(a_all.tolist(), valid.tolist())
            if ok and a >= 0 and a + total <= n]
    return T, np.vstack(rows)


@pytest.mark.parametrize("align", ["start", "center"])
@pytest.mark.parametrize("pad_frac, n_points", [(0.0, 50), (0.2, 64), (0.35, 333)])
def test_start_and_center_match_the_base_kernel_exactly(align, pad_frac, n_points):
    rng = np.random.default_rng(11)
    sig, spans, _ = _trace_with_dips(-1.0)
    sig = sig + rng.normal(0.0, 2.0, sig.size)
    spans = spans + [(30, 90), (19_950, 19_990)]       # near the edges: some are skipped
    es = _eventset(spans, provenance={"polarity": -1})
    starts, ends = (np.asarray(v, dtype=np.int64) for v in zip(*spans))
    T, matrix, mean, median = event_shape.event_ensemble(
        sig, FS, es, n_points=n_points, pad_frac=pad_frac, align=align,
        normalize_time=False)
    T0, m0 = _base_realtime_rows(sig, FS, starts, ends, n_points, pad_frac, align)
    np.testing.assert_array_equal(T, T0)
    np.testing.assert_array_equal(matrix, m0)
    np.testing.assert_array_equal(mean, m0.mean(axis=0))
    np.testing.assert_array_equal(median, np.median(m0, axis=0))


def test_band_matches_its_curve_and_is_left_out_where_the_curve_is_undefined():
    m = np.array([
        [1.0, 2.0, 3.0, np.nan],
        [2.0, 4.0, np.nan, np.nan],
        [3.0, 6.0, 5.0, 7.0],
        [6.0, 8.0, 1.0, np.inf],
    ])
    mean, median = m.mean(axis=0), np.median(m, axis=0)   # event_ensemble's curves
    lower, upper, n = event_shape.ensemble_band(m, "sd")
    # n is the count behind the summary value: all rows where it exists, else 0.
    np.testing.assert_array_equal(n, [4, 4, 0, 0])
    np.testing.assert_array_equal(n, event_shape.ensemble_point_counts(m))
    for j in range(2):
        sd = np.std(m[:, j], ddof=1)
        assert lower[j] == pytest.approx(mean[j] - sd)
        assert upper[j] == pytest.approx(mean[j] + sd)
    # Where an event is not finite the mean and median are NaN, and so is the band:
    # never a spread of the remaining events shaded around a curve that is not drawn.
    assert np.isnan(mean[2:]).all() and np.isnan(median[2:]).all()
    assert np.isnan(lower[2:]).all() and np.isnan(upper[2:]).all()
    lo_q, hi_q, n_q = event_shape.ensemble_band(m, "iqr")
    np.testing.assert_array_equal(n_q, n)
    for j in range(2):
        assert lo_q[j] == pytest.approx(np.percentile(m[:, j], 25))
        assert hi_q[j] == pytest.approx(np.percentile(m[:, j], 75))
        assert lo_q[j] <= median[j] <= hi_q[j]
    assert np.isnan(lo_q[2:]).all() and np.isnan(hi_q[2:]).all()
    # One event has no spread anywhere.
    one_lo, one_hi, one_n = event_shape.ensemble_band(m[:1, :2], "sd")
    assert np.isnan(one_lo).all() and np.isnan(one_hi).all()
    np.testing.assert_array_equal(one_n, [1, 1])
    with pytest.raises(ValueError):
        event_shape.ensemble_band(m, "ci95")


def test_band_of_an_empty_ensemble_is_all_nan():
    lower, upper, n = event_shape.ensemble_band(np.empty((0, 6)), "sd")
    assert np.isnan(lower).all() and np.isnan(upper).all()
    np.testing.assert_array_equal(n, np.zeros(6))
