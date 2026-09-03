"""The Zn2-insulin sustained gate and the thirteen derived waveform features.

WHAT THIS FILE IS DEFENDING
---------------------------
This module is a **port** of ``Zn2Insulin_StudentInference.ipynb`` cells 15 and
51, and the owner's own run is the oracle:
``CSVData/event_clustering_features.csv``, 1488 rows x 29 columns. The gate does
nearly all the selection work -- 11004 fired regions on zn_30h become 1076 gated
events, 17004 fired become 1488 across the four recordings -- so an error of a few
events changes every cluster, fraction and confidence interval downstream while
still looking entirely plausible. Nothing in this file is checked against my own
reasoning about what the pipeline "should" do; the analytic tests pin the
FORMULAS, and the corpus tests pin them against HIS NUMBERS.

The claims, in order of how much damage their failure does:

  a. all fourteen columns reproduce all 1488 oracle rows at ``max|diff| == 0.0``
     -- bit-exact, not merely close (``test_features_reproduce_the_oracle_bit_exactly``);
  b. the gate reproduces the ``passes_gate`` flag on every one of the 17004 fired
     regions in the four catalogs, and its two depth columns bit-exactly;
  c. the gate is SUSTAINED, i.e. it rejects the one-sample spike that a
     peak-depth gate keeps -- the whole reason the median is the gating statistic;
  d. the TWO-BASELINE TRAP holds: the gate's event-excluded baseline/MAD and the
     features' whole-trace baseline/MAD/global-I0 are different numbers on the
     same trace, on purpose. A test pins the difference so that "fixing" it fails
     here rather than silently 500 lines downstream;
  e. the frozen 20-column ``events.SCHEMA`` gains nothing -- the features are
     derived on demand and the EventSet comes back untouched;
  f. results are label-aligned to the caller's index (the notebook's
     positional-array-into-labelled-slice hazard cannot be inherited), and the
     work polls the ambient cancel token.

A note on reading the oracle. ``pd.read_csv`` at its default ``float_precision``
is **not** round-trip exact -- it lands up to 1 ULP off the value that was
written, which is exactly the ~1e-14 "float round-off" a naive comparison
reports. Every read of an oracle CSV here passes
``float_precision="round_trip"``, and with it the agreement is exactly zero.

Pure numpy/scipy; the corpus-marked tests skip cleanly when the research data is
absent.
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import event_features as EF
from nanoscope.analysis import events as EV
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope
from nanoscope.analysis.events import EventSet

FS = 50_000.0
CORPUS = pathlib.Path("/Users/abrahamcampos/Desktop/Zn2IonCondition/CSVData")
TAGS = ("zn_2h", "zn_10h", "zn_18h", "zn_30h")
ORACLE = CORPUS / "event_clustering_features.csv"
#: fired / gated per recording, from the owner's catalogs (spec §1).
EXPECTED_COUNTS = {
    "zn_2h": (408, 108),
    "zn_10h": (3945, 110),
    "zn_18h": (1647, 194),
    "zn_30h": (11004, 1076),
}
_HAVE_CORPUS = ORACLE.exists() and all((CORPUS / f"{t}.csv.gz").exists() for t in TAGS)
corpus = pytest.mark.skipif(not _HAVE_CORPUS, reason="Zn2IonCondition research data not present")


# ================================================================================================
# builders
# ================================================================================================

def _scalars(baseline: float = 0.0, noise: float = 1.0, i0: float = 200.0) -> EF.FeatureScalars:
    return EF.FeatureScalars(baseline_pA=baseline, noise_pA=noise, i0_pA=i0)


def _one_event_features(seg, *, baseline=0.0, noise=1.0, i0=200.0, fs=FS, min_samples=1):
    """Features for a single hand-built segment, as a plain dict.

    The segment is planted in the middle of a flat trace so the window arithmetic
    is exercised too, rather than the segment being the whole trace.
    """
    seg = np.asarray(seg, dtype=np.float64)
    trace = np.full(seg.size + 200, baseline, dtype=np.float64)
    trace[100 : 100 + seg.size] = seg
    out = EF.event_features(
        trace,
        fs,
        np.array([[100, 100 + seg.size]]),
        scalars=_scalars(baseline, noise, i0),
        min_samples=min_samples,
    )
    assert len(out) == 1
    return out.iloc[0].to_dict()


def _eventset(windows, *, index=None, recording="rec") -> EventSet:
    """An EventSet at the exact SCHEMA from ``[(start, end), ...]``."""
    rows = []
    for i, (s, e) in enumerate(windows):
        rows.append(
            {
                "event_id": i,
                "recording": recording,
                "pore_id": "pore",
                "start_idx": int(s),
                "end_idx": int(e),
                "t_start_s": s / FS,
                "dwell_s": (e - s) / FS,
                "baseline_pA": 0.0,
                "depth_pA": 1.0,
                "frac_depth": 0.0,
                "ecd_pA_s": 0.0,
                "max_dev_pA": 1.0,
                "mean_dev_pA": 1.0,
                "rise_s": 0.0,
                "fall_s": 0.0,
                "snr": 1.0,
                "n_levels": 1,
                "filter_limited": False,
                "accepted": True,
                "flag": "",
            }
        )
    df = pd.DataFrame(rows) if rows else EV.empty_df()
    df = df.astype({c: d for c, (d, _, _) in EV.SCHEMA.items()})
    if index is not None:
        df.index = pd.Index(index)
    return EventSet(df=df)


def _trace_with_events(
    depths, *, n=200_000, width=50, gap=1000, noise_sd=1.0, baseline=0.0, seed=0
):
    """A flat noisy trace with square blockades of the given depths planted in it.

    Returns ``(trace, windows)``. Deterministic: the noise comes from a seeded
    Generator, so a test asserting an exact gate decision is asserting something
    stable.
    """
    rng = np.random.default_rng(seed)
    trace = baseline + rng.normal(0.0, noise_sd, n)
    windows = []
    at = gap
    for d in depths:
        w = width if np.isscalar(width) else width[len(windows)]
        trace[at : at + w] -= d
        windows.append((at, at + w))
        at += w + gap
    return trace, np.array(windows, dtype=np.int64)


# ================================================================================================
# (a) the formulas -- one hand-built event at a time
# ================================================================================================

def test_square_blockade_reports_its_own_depth_in_every_depth_feature():
    """A perfect square: max, sustained, mean and shallowest depth all equal D."""
    d, n = 20.0, 40
    f = _one_event_features(np.full(n, -d), noise=2.0, i0=200.0)
    assert f["depth_max_pA"] == d
    assert f["depth_sust_pA"] == d
    assert f["mean_pA"] == d
    assert f["min_dev_pA"] == d
    assert f["std_pA"] == 0.0
    assert f["dwell_ms"] == pytest.approx(n / FS * 1000.0)
    assert f["log_dwell"] == pytest.approx(math.log10(n / FS * 1000.0))
    assert f["area_pA_ms"] == pytest.approx(d * n / FS * 1000.0)
    assert f["f_sust"] == pytest.approx(d / 200.0)
    assert f["snr"] == pytest.approx(d / 2.0)
    # A flat interior has no amplitude spread at all -> zero entropy, and
    # argmax of a constant array is index 0 -> the peak sits "at onset".
    assert f["entropy"] == 0.0
    assert f["t_peak_frac"] == 0.0


def test_a_perfectly_flat_event_yields_nan_shape_statistics_not_zero():
    """Zero variance makes skew/kurtosis undefined; that reads NaN, never 0.0.

    ``0.0`` is the deliberate n<5 fallback (a near-symmetric prior for a segment
    too short to support a shape statistic). Returning it for a zero-variance
    segment as well would make an undefined value indistinguishable from a
    defaulted one -- and NaN is this app's "not measured", which
    ``populations.feature_matrix`` already drops row-wise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = _one_event_features(np.full(40, -20.0))
    assert math.isnan(f["skewness"])
    assert math.isnan(f["kurtosis"])


def test_ramp_blockade_integrates_to_half_the_square():
    """A triangular blockade of depth D over W samples has area D*W/2."""
    d, n = 30.0, 101
    ramp = -np.concatenate([np.linspace(0, d, 51), np.linspace(d, 0, 51)[1:]])
    f = _one_event_features(ramp)
    assert f["area_pA_ms"] == pytest.approx(d * n / 2 / FS * 1000.0, rel=1e-2)
    assert f["depth_max_pA"] == pytest.approx(d)
    assert f["t_peak_frac"] == pytest.approx(0.5, abs=1e-9)


def test_min_dev_is_the_shallowest_sample_and_goes_negative_above_baseline():
    """``min_dev_pA`` is ``min(d)``, a signed deviation -- not a magnitude.

    The oracle's minimum is -37.3 pA: a sample that rose ABOVE the baseline
    during an event. Rectifying it would erase exactly that.
    """
    seg = np.full(20, -10.0)
    seg[7] = +4.0  # rises above baseline
    f = _one_event_features(seg)
    assert f["min_dev_pA"] == pytest.approx(-4.0)
    assert f["depth_max_pA"] == pytest.approx(10.0)


def test_t_peak_frac_is_the_position_of_the_deepest_sample_first_on_ties():
    n = 11
    seg = np.full(n, -5.0)
    seg[3] = -50.0
    assert _one_event_features(seg)["t_peak_frac"] == pytest.approx(3 / (n - 1))
    # two equally deep samples -> np.argmax takes the FIRST
    seg[8] = -50.0
    assert _one_event_features(seg)["t_peak_frac"] == pytest.approx(3 / (n - 1))
    # a single-sample event has no span; the denominator floors at 1
    assert _one_event_features(np.array([-5.0]))["t_peak_frac"] == 0.0


def test_depth_sust_f_sust_and_snr_all_use_the_MEDIAN_not_the_peak():
    """One deep spike must not move the sustained depth (that is the point)."""
    seg = np.full(21, -10.0)
    seg[10] = -200.0
    f = _one_event_features(seg, noise=2.0, i0=100.0)
    assert f["depth_sust_pA"] == pytest.approx(10.0)
    assert f["depth_max_pA"] == pytest.approx(200.0)
    assert f["f_sust"] == pytest.approx(10.0 / 100.0)
    assert f["snr"] == pytest.approx(10.0 / 2.0)


def test_std_is_the_population_standard_deviation():
    seg = -np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    f = _one_event_features(seg)
    assert f["std_pA"] == pytest.approx(float(np.std(np.array([1.0, 2, 3, 4, 5]), ddof=0)))
    assert f["std_pA"] != pytest.approx(float(np.std(np.array([1.0, 2, 3, 4, 5]), ddof=1)))


def test_skew_and_kurtosis_are_scipy_defaults_biased_and_excess():
    rng = np.random.default_rng(7)
    seg = -np.abs(rng.normal(20.0, 3.0, 400))
    from scipy import stats as sps

    d = 0.0 - seg
    f = _one_event_features(seg)
    assert f["skewness"] == pytest.approx(float(sps.skew(d)))  # bias=True
    assert f["kurtosis"] == pytest.approx(float(sps.kurtosis(d)))  # Fisher/excess
    # excess kurtosis, so a near-normal sample sits near 0, not near 3
    assert abs(f["kurtosis"]) < 1.0


def test_log_dwell_floors_at_one_microsecond():
    """``log10(max(dwell_ms, 1e-3))`` -- a sub-microsecond event reads -3, not -inf."""
    f = _one_event_features(np.full(3, -5.0), fs=1e7, min_samples=1)
    assert f["dwell_ms"] == pytest.approx(3e-4)
    assert f["log_dwell"] == pytest.approx(-3.0)


def test_short_events_default_their_shape_statistics_to_zero():
    """Below five samples: skew, kurtosis and entropy are 0.0 by contract.

    Never fires on the reference data (``min_event_ms=0.1`` at 50 kHz is exactly
    five samples, and the notebook prints ``n_short = 0``), but it is contract and
    a coarser detector setting reaches it.
    """
    f = _one_event_features(np.array([-5.0, -9.0, -2.0, -30.0]), min_samples=1)
    assert f["skewness"] == 0.0
    assert f["kurtosis"] == 0.0
    assert f["entropy"] == 0.0
    assert f["depth_max_pA"] == pytest.approx(30.0)  # the rest still measure


# ================================================================================================
# entropy -- fixed 8 bins, natural log, normalised by log(8)
# ================================================================================================

def test_entropy_is_one_when_the_amplitudes_fill_all_eight_bins_evenly():
    seg = -np.arange(80, dtype=float)  # 10 samples per bin
    assert _one_event_features(seg)["entropy"] == pytest.approx(1.0)


def test_entropy_of_two_equally_populated_levels_is_log2_over_log8():
    """A clean two-level event: ``ln 2 / ln 8`` = exactly 1/3."""
    seg = np.concatenate([np.full(30, -10.0), np.full(30, -40.0)])
    assert _one_event_features(seg)["entropy"] == pytest.approx(1.0 / 3.0)


def test_entropy_uses_a_FIXED_bin_count_not_freedman_diaconis():
    """The bin count never adapts; only the edges do.

    Freedman-Diaconis on this segment asks for far more than eight bins and would
    report a visibly different number. Pinning the difference is what stops a
    later "improvement" from silently re-scaling every entropy in the table.
    """
    rng = np.random.default_rng(3)
    d = np.abs(rng.normal(20.0, 5.0, 2000))
    seg = -d

    def _entropy_with(bins):
        p, _ = np.histogram(d, bins=bins)
        p = p[p > 0].astype(float)
        p /= p.sum()
        return float(-(p * np.log(p)).sum() / np.log(len(np.histogram(d, bins=bins)[0])))

    iqr = np.subtract(*np.percentile(d, [75, 25]))
    fd_width = 2 * iqr / len(d) ** (1 / 3)
    fd_bins = int(np.ceil((d.max() - d.min()) / fd_width))
    assert fd_bins > 20, "the fixture must actually distinguish the two rules"

    got = _one_event_features(seg)["entropy"]
    assert got == pytest.approx(EF.shannon_entropy(d, bins=8))
    assert got != pytest.approx(_entropy_with(fd_bins), abs=1e-6)


def test_entropy_drops_empty_bins_before_normalising():
    """Two far-apart levels leave six bins empty; they must not enter the sum."""
    seg = np.concatenate([np.full(20, -1.0), np.full(20, -100.0)])
    assert _one_event_features(seg)["entropy"] == pytest.approx(math.log(2) / math.log(8))


def test_entropy_is_zero_for_a_flat_or_too_short_segment():
    assert EF.shannon_entropy(np.full(50, 3.0)) == 0.0
    assert EF.shannon_entropy(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


# ================================================================================================
# (e)+(f) frame contract: schema untouched, labels preserved, cancellable
# ================================================================================================

def test_the_frozen_eventset_schema_gains_nothing():
    """The features are DERIVED. The EventSet comes back exactly as it went in."""
    trace, windows = _trace_with_events([25.0, 30.0], n=20_000)
    es = _eventset(windows)
    before = list(es.df.columns)
    out = EF.event_features(trace, FS, es, scalars=_scalars(noise=1.0))
    assert list(es.df.columns) == before == EV.REQUIRED
    # and none of the derived names leaked back in, including the two that
    # collide by name with schema columns but not by definition
    assert not (set(EF.FEATURE_COLUMNS) - {"snr"}) & set(es.df.columns)
    assert len(out) == 2


def test_result_is_indexed_by_the_callers_own_labels():
    """The notebook's ``CLUST.loc[_sub.index, col] = positional_array`` hazard.

    That assignment is correct only while the original RangeIndex survives; any
    filter or sort in between misaligns every row silently. Returning a frame
    carrying the caller's labels means a downstream join can never be positional.
    """
    trace, windows = _trace_with_events([25.0, 30.0, 35.0], n=30_000)
    es = _eventset(windows, index=["e7", "e3", "e9"])
    out = EF.event_features(trace, FS, es, scalars=_scalars())
    assert list(out.index) == ["e7", "e3", "e9"]
    # a filtered, non-monotonic subset keeps its own labels
    sub = EventSet(df=es.df.iloc[[2, 0]])
    out2 = EF.event_features(trace, FS, sub, scalars=_scalars())
    assert list(out2.index) == ["e9", "e7"]
    assert out2.loc["e9"].equals(out.loc["e9"])


def test_the_gate_carries_the_callers_labels_too():
    """``kept_index`` lets a decision travel back by LABEL, never by position."""
    trace, windows = _trace_with_events([1.0, 20.0, 25.0], n=30_000, width=40)
    es = _eventset(windows, index=["a", "b", "c"])
    g = EF.sustained_gate(trace, es)
    assert list(g.index) == ["a", "b", "c"]
    assert list(g.kept_index) == ["b", "c"]
    assert list(es.df.loc[g.kept_index, "event_id"]) == [1, 2]


def test_windows_below_min_samples_are_absent_from_the_result():
    """``if _e - _s < 3: continue`` -- dropped, not filled with NaN.

    The dropped labels are recoverable by differencing the indices, which is the
    honest report: those events were not measured.
    """
    trace = np.zeros(1000)
    trace[100:140] -= 20.0
    trace[300:302] -= 20.0
    es = _eventset([(100, 140), (300, 302)], index=["long", "tiny"])
    out = EF.event_features(trace, FS, es, scalars=_scalars())
    assert list(out.index) == ["long"]
    assert set(es.df.index) - set(out.index) == {"tiny"}


def test_a_zero_length_window_is_never_measured():
    """``min_samples`` floors at 1: an empty window has no statistics at all."""
    trace = np.zeros(1000)
    out = EF.event_features(
        trace, FS, np.array([[100, 100], [200, 210]]), scalars=_scalars(), min_samples=0
    )
    assert list(out.index) == [1]


def test_empty_input_gives_an_empty_frame_with_the_right_columns():
    out = EF.event_features(np.zeros(100), FS, _eventset([]), scalars=_scalars())
    assert len(out) == 0
    assert list(out.columns) == list(EF.FEATURE_COLUMNS)
    assert all(str(t) == "float64" for t in out.dtypes)


def test_every_feature_column_is_float64():
    trace, windows = _trace_with_events([25.0], n=10_000)
    out = EF.event_features(trace, FS, windows, scalars=_scalars())
    assert list(out.columns) == list(EF.FEATURE_COLUMNS)
    assert all(str(t) == "float64" for t in out.dtypes)


def test_clust_feats_is_the_pca_order_with_entropy_LAST():
    """``CLUST_FEATS`` != ``FEATURE_COLUMNS`` minus dwell -- the tail differs.

    The CSV column order puts entropy eleventh; the PCA input order puts it last.
    Same thirteen features, different loading-table row order for the same fit.
    """
    assert set(EF.CLUST_FEATS) == set(EF.FEATURE_COLUMNS) - {"dwell_ms"}
    assert len(EF.CLUST_FEATS) == 13
    assert EF.CLUST_FEATS[-1] == "entropy"
    assert EF.CLUST_FEATS[-3:] == ("f_sust", "snr", "entropy")
    assert EF.FEATURE_COLUMNS[-3:] == ("entropy", "f_sust", "snr")


def test_bad_inputs_are_refused_with_a_reason():
    trace = np.zeros(1000)
    es = _eventset([(100, 140)])
    for bad_fs in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fs must be"):
            EF.event_features(trace, bad_fs, es, scalars=_scalars())
    with pytest.raises(ValueError, match="start_idx"):
        EF.event_features(trace, FS, pd.DataFrame({"a": [1]}), scalars=_scalars())
    with pytest.raises(ValueError, match="non-finite"):
        EF.feature_scalars(np.array([1.0, np.nan, 2.0]), i0_pA=100.0)
    with pytest.raises(ValueError, match="non-finite"):
        EF.gate_scalars(np.array([1.0, np.inf, 2.0]))
    with pytest.raises(ValueError, match="i0_pA"):
        EF.feature_scalars(np.zeros(100), i0_pA=0.0)
    with pytest.raises(ValueError, match="raw trace"):
        EF.feature_scalars(np.zeros(100))
    with pytest.raises(ValueError, match="polarity"):
        EF.gate_scalars(np.zeros(100), polarity=0)


def test_both_kernels_poll_the_ambient_cancel_token():
    """Contract 6: long work stops on request, without terminate()."""
    n_ev = 400  # comfortably past EVENT_POLL_STRIDE
    trace, windows = _trace_with_events([25.0] * n_ev, n=400_000, width=20, gap=800)
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            EF.event_features(trace, FS, windows, scalars=_scalars())
        with pytest.raises(OperationCancelled):
            EF.sustained_gate(trace, windows)


def test_running_twice_gives_identical_output():
    trace, windows = _trace_with_events([25.0, 12.0, 40.0], n=30_000)
    sc = _scalars()
    a = EF.event_features(trace, FS, windows, scalars=sc)
    b = EF.event_features(trace, FS, windows, scalars=sc)
    pd.testing.assert_frame_equal(a, b)
    g1, g2 = EF.sustained_gate(trace, windows), EF.sustained_gate(trace, windows)
    assert np.array_equal(g1.keep, g2.keep)
    assert np.array_equal(g1.depth_sust_pA, g2.depth_sust_pA, equal_nan=True)


# ================================================================================================
# (c) the gate is SUSTAINED
# ================================================================================================

def test_the_gate_rejects_the_spike_a_peak_gate_would_keep():
    """THE reason the median is the gating statistic.

    Two candidate regions of identical length: one holds 6 sigma for its whole
    span, the other is baseline apart from a single 30-sigma sample. A min-based
    gate keeps both. The sustained gate keeps only the first -- and reports the
    spike's peak depth, so the rejection is visible rather than silent.
    """
    trace, windows = _trace_with_events([6.0, 0.0], n=20_000, width=40, noise_sd=1.0)
    s, e = windows[1]
    trace[s + 20] -= 30.0  # one deep sample inside the second region
    g = EF.sustained_gate(trace, windows)
    assert list(g.keep) == [True, False]
    assert g.depth_peak_pA[1] > 4.0 * g.scalars.noise_pA  # a peak gate WOULD keep it
    assert g.depth_sust_pA[1] < 4.0 * g.scalars.noise_pA
    assert g.n_kept == 1


def test_the_gate_threshold_is_four_excluded_mads_and_the_boundary_is_inclusive():
    trace, windows = _trace_with_events([0.0, 0.0], n=20_000, width=40, noise_sd=1.0)
    sc = EF.gate_scalars(trace)
    # plant one event exactly AT the threshold and one a hair below
    for i, factor in enumerate((4.0, 4.0 - 1e-6)):
        s, e = windows[i]
        trace[s:e] = sc.baseline_pA - factor * sc.noise_pA
    g = EF.sustained_gate(trace, windows, scalars=sc)
    assert g.sust_sigma[0] == pytest.approx(4.0, abs=1e-9)
    assert list(g.keep) == [True, False]  # `>=`, not `>`
    assert EF.MIN_DEPTH_SIGMA == 4.0


def test_a_shallower_threshold_keeps_strictly_more():
    trace, windows = _trace_with_events(
        [3.0, 5.0, 8.0, 12.0], n=40_000, width=40, noise_sd=1.0
    )
    sc = EF.gate_scalars(trace)
    keeps = [
        EF.sustained_gate(trace, windows, scalars=sc, min_depth_sigma=k).n_kept
        for k in (2.0, 4.0, 6.0, 10.0)
    ]
    assert keeps == sorted(keeps, reverse=True)
    assert keeps[0] > keeps[-1]


def test_the_gate_baseline_EXCLUDES_events():
    """A high duty cycle drags the whole-trace median; the gate's baseline holds.

    This is why cell 15 sigma-clips at all: zn_30h fires ~36 events/s where zn_2h
    fires ~3/s, so an uncorrected median biases depth by a DIFFERENT amount in
    each recording being compared.
    """
    rng = np.random.default_rng(11)
    trace = rng.normal(0.0, 1.0, 100_000)
    for at in range(0, 100_000, 100):  # 30 % duty
        trace[at : at + 30] -= 25.0
    plain = float(np.median(trace))
    sc = EF.gate_scalars(trace)
    assert abs(sc.baseline_pA) < 0.05  # the excluded baseline sits on the open pore
    assert abs(plain) > 0.5  # the plain median has been dragged toward the events
    assert abs(plain) > 10 * abs(sc.baseline_pA)
    assert sc.duty == pytest.approx(0.30, abs=0.05)


def test_peak_depth_is_reported_but_never_gates():
    trace, windows = _trace_with_events([6.0], n=20_000, width=40)
    g = EF.sustained_gate(trace, windows)
    assert np.isfinite(g.depth_peak_pA[0]) and np.isfinite(g.peak_sigma[0])
    assert g.depth_peak_pA[0] >= g.depth_sust_pA[0]
    assert g.peak_sigma[0] == pytest.approx(g.depth_peak_pA[0] / g.scalars.noise_pA)
    # the decision is a pure function of the sustained depth
    assert np.array_equal(
        g.keep, g.depth_sust_pA >= g.min_depth_sigma * g.scalars.noise_pA
    )


def test_gate_scalars_refuses_a_trace_with_no_open_pore_level():
    with pytest.raises(ValueError, match="constant trace|no open-pore"):
        EF.gate_scalars(np.full(1000, 3.0))


def test_polarity_is_an_exact_mirror():
    """At -1 the trace is mirrored, so the +1 path -- every Zn2 recording -- is a
    bit-exact no-op and the oracle holds by construction."""
    trace, windows = _trace_with_events([6.0, 20.0], n=20_000, width=40)
    up = EF.sustained_gate(trace, windows)
    down = EF.sustained_gate(-trace, windows, polarity=-1)
    assert np.array_equal(up.keep, down.keep)
    assert np.allclose(up.depth_sust_pA, down.depth_sust_pA, rtol=0, atol=0)
    fu = EF.event_features(trace, FS, windows, scalars=_scalars())
    fd = EF.event_features(-trace, FS, windows, scalars=_scalars(), polarity=-1)
    pd.testing.assert_frame_equal(fu, fd)


def test_windows_are_clipped_into_the_trace_rather_than_wrapping():
    trace = np.zeros(1000)
    trace[900:1000] -= 20.0
    out = EF.event_features(trace, FS, np.array([[900, 5000], [-10, 20]]), scalars=_scalars())
    assert len(out) == 2
    assert out.iloc[0]["dwell_ms"] == pytest.approx(100 / FS * 1000.0)
    assert out.iloc[1]["depth_max_pA"] == 0.0  # clipped to [0, 20), not wrapped


# ================================================================================================
# (d) the two-baseline trap
# ================================================================================================

def test_the_gate_and_the_features_deliberately_use_DIFFERENT_baselines():
    """Pinning the trap, so "fixing" it fails here and not 500 lines downstream.

    Same trace, two baselines and two noise estimates: the gate's are measured on
    event-excluded samples, the features' on the whole trace. Measured on zn_30h
    the two MADs are 2.795 and 2.907 pA and the baselines are 0.0764 pA apart, so
    the catalog's depth column is offset by a constant from the feature table's.
    """
    rng = np.random.default_rng(5)
    trace = rng.normal(0.0, 2.0, 200_000)
    for at in range(0, 200_000, 500):  # 8 % duty -- enough to move the median
        trace[at : at + 40] -= 30.0
    gate = EF.gate_scalars(trace)
    feat = EF.feature_scalars(trace, i0_pA=250.0)
    assert gate.baseline_pA != feat.baseline_pA
    assert gate.noise_pA != feat.noise_pA
    assert feat.noise_pA > gate.noise_pA  # events widen the whole-trace MAD

    windows = np.array([[at, at + 40] for at in range(0, 200_000, 500)])
    g = EF.sustained_gate(trace, windows, scalars=gate)
    f = EF.event_features(trace, FS, windows, scalars=feat)
    offset = g.depth_sust_pA - f["depth_sust_pA"].to_numpy()
    assert np.allclose(offset, offset[0], rtol=0, atol=1e-9)  # a CONSTANT offset
    assert abs(offset[0]) > 1e-6
    # and `snr` is against the whole-trace MAD, NOT the gate's excluded one
    assert np.allclose(f["snr"].to_numpy(), f["depth_sust_pA"].to_numpy() / feat.noise_pA)
    assert not np.allclose(f["snr"].to_numpy(), g.sust_sigma)


def test_f_sust_divides_by_the_single_global_I0():
    """Not a per-event local I0 -- cell 51 never calls the rolling-median machinery."""
    trace, windows = _trace_with_events([10.0, 20.0, 30.0], n=30_000, width=40)
    sc = _scalars(i0=250.0)
    out = EF.event_features(trace, FS, windows, scalars=sc)
    assert np.allclose(out["f_sust"], out["depth_sust_pA"] / 250.0)


# ================================================================================================
# time_half
# ================================================================================================

def test_time_half_splits_each_group_at_its_own_median_with_early_inclusive():
    onset = np.array([0.0, 1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    grp = np.array(["a", "a", "a", "a", "b", "b", "b"])
    half = EF.time_half(onset, grp)
    assert list(half[:4]) == ["early", "early", "late", "late"]
    # odd group: the median sample itself is "early" (`<=`)
    assert list(half[4:]) == ["early", "early", "late"]
    assert list(EF.time_half(np.array([1.0, 2.0, 3.0]))) == ["early", "early", "late"]
    assert EF.time_half(np.array([])).size == 0


def test_time_half_refuses_a_mismatched_group_length():
    with pytest.raises(ValueError, match="per-event"):
        EF.time_half(np.arange(5.0), np.array(["a", "b"]))


# ================================================================================================
# (a)+(b) THE ORACLE -- the owner's own 1488 rows and 17004 fired regions
# ================================================================================================

@pytest.fixture(scope="module")
def zn_traces():
    """``tag -> (detrended float32, raw float64, fs, catalog)``, read once.

    Dtypes mirror the notebook exactly: ``current_detr_pA`` is explicitly cast to
    float32 (so its median and MAD are float32 arithmetic), ``current_raw_pA`` is
    left as pandas read it. Widening either moves the per-recording scalars in
    their last bits, which is enough to lose bit-exactness.
    """
    if not _HAVE_CORPUS:
        pytest.skip("Zn2IonCondition research data not present")
    out = {}
    for t in TAGS:
        df = pd.read_csv(
            CORPUS / f"{t}.csv.gz",
            compression="gzip",
            usecols=["current_raw_pA", "current_detr_pA"],
        )
        meta = json.loads((CORPUS / f"{t}_metadata.json").read_text())
        cat = pd.read_csv(CORPUS / f"events_catalog_{t}.csv", float_precision="round_trip")
        out[t] = (
            df["current_detr_pA"].values.astype(np.float32),
            df["current_raw_pA"].values,
            float(meta["fs_hz"]),
            cat,
        )
        del df
    return out


def _catalog_windows(cat: pd.DataFrame, fs: float) -> np.ndarray:
    s = np.round(cat["onset_s"].to_numpy() * fs).astype(np.int64)
    e = np.round(cat["end_s"].to_numpy() * fs).astype(np.int64)
    return np.column_stack([s, e])


@corpus
def test_sustained_gate_reproduces_every_catalog_decision(zn_traces):
    """17004 fired regions in, the owner's own ``passes_gate`` flag out.

    Not "close": every single region agrees, and the counts land on his
    408/3945/1647/11004 fired and 108/110/194/1076 gated. The gate keeps 9.8 % of
    zn_30h, so this is the test that would catch a port whose downstream numbers
    are all plausible and all wrong.
    """
    total_fired = total_kept = 0
    for tag in TAGS:
        sig, _raw, fs, cat = zn_traces[tag]
        g = EF.sustained_gate(sig, _catalog_windows(cat, fs))
        fired, gated = EXPECTED_COUNTS[tag]
        assert len(cat) == fired
        assert int(cat["passes_gate"].sum()) == gated
        assert np.array_equal(g.keep, cat["passes_gate"].to_numpy())
        assert g.n_kept == gated
        total_fired += fired
        total_kept += gated
    assert (total_fired, total_kept) == (17004, 1488)


@corpus
def test_gate_depth_columns_reproduce_the_catalog_bit_exactly(zn_traces):
    """``depth_sust_pA``/``depth_peak_pA``/``sust_sigma``/``peak_sigma``, diff 0.0."""
    for tag in TAGS:
        sig, _raw, fs, cat = zn_traces[tag]
        g = EF.sustained_gate(sig, _catalog_windows(cat, fs))
        for col, got in (
            ("depth_sust_pA", g.depth_sust_pA),
            ("depth_peak_pA", g.depth_peak_pA),
            ("sust_sigma", g.sust_sigma),
            ("peak_sigma", g.peak_sigma),
        ):
            diff = np.nanmax(np.abs(got - cat[col].to_numpy()))
            assert diff == 0.0, f"{tag}.{col} max|diff| = {diff:.3e}"


@corpus
def test_features_reproduce_the_oracle_bit_exactly(zn_traces):
    """All fourteen columns, all 1488 rows, ``max|diff| == 0.0``.

    Read the oracle with ``float_precision="round_trip"``: pandas' default parser
    lands up to 1 ULP off what was written, which is precisely the ~1e-14 that a
    naive comparison reports and calls "float round-off".
    """
    oracle = pd.read_csv(ORACLE, float_precision="round_trip")
    assert oracle.shape == (1488, 29)

    worst = dict.fromkeys(EF.FEATURE_COLUMNS, 0.0)
    for tag in TAGS:
        sig, raw, fs, cat = zn_traces[tag]
        sub = cat[cat["passes_gate"]].reset_index(drop=True)
        sc = EF.feature_scalars(sig, raw)
        got = EF.event_features(sig, fs, _catalog_windows(sub, fs), scalars=sc)
        exp = oracle[oracle["tag"] == tag].reset_index(drop=True)
        assert len(got) == len(exp) == EXPECTED_COUNTS[tag][1]
        # boundaries line up row for row before any value is compared
        assert np.array_equal(sub["onset_s"].to_numpy(), exp["onset_s"].to_numpy())
        for c in EF.FEATURE_COLUMNS:
            d = float(np.nanmax(np.abs(got[c].to_numpy() - exp[c].to_numpy())))
            worst[c] = max(worst[c], d)
    bad = {c: d for c, d in worst.items() if d != 0.0}
    assert not bad, f"not bit-exact: {bad}"


@corpus
def test_the_two_baseline_offset_is_real_and_constant_per_recording(zn_traces):
    """His catalog and his feature table disagree by a constant. On purpose.

    zn_30h: excluded MAD 2.795 pA vs whole-trace 2.907 pA, and every catalog
    ``depth_sust_pA`` sits exactly 0.0764 pA above the oracle's. A port that
    "corrects" one baseline to the other fails right here.
    """
    oracle = pd.read_csv(ORACLE, float_precision="round_trip")
    # tag -> (event-excluded MAD, whole-trace MAD, catalog-minus-oracle depth offset)
    expect = {"zn_30h": (2.795, 2.907, 0.0764), "zn_2h": (2.176, 2.234, 0.0166)}
    for tag, (gate_mad, feat_mad, offset) in expect.items():
        sig, raw, fs, cat = zn_traces[tag]
        gs = EF.gate_scalars(sig)
        fs_ = EF.feature_scalars(sig, raw)
        assert gs.noise_pA == pytest.approx(gate_mad, abs=5e-3)
        assert fs_.noise_pA == pytest.approx(feat_mad, abs=5e-3)
        assert gs.noise_pA != fs_.noise_pA

        sub = cat[cat["passes_gate"]].reset_index(drop=True)
        exp = oracle[oracle["tag"] == tag].reset_index(drop=True)
        d = sub["depth_sust_pA"].to_numpy() - exp["depth_sust_pA"].to_numpy()
        assert np.allclose(d, offset, atol=5e-4)
        # The offset IS the difference of the two baselines, constant to within a
        # float32 ULP: the catalog takes `median(float32 segment)` while the
        # feature table medians the promoted float64 deviations, so an
        # even-length segment averages its two middle samples in different
        # precisions. Measured spread: 1.9e-6 (2^-19) on zn_2h, 7.6e-6 (2^-17) on
        # zn_30h -- exactly one float32 step at those depths, and nothing else.
        assert np.allclose(d, gs.baseline_pA - fs_.baseline_pA, atol=1e-5)
        assert np.ptp(d) < 1e-5


@corpus
def test_snr_is_not_the_catalogs_sust_sigma(zn_traces):
    """Same numerator, different denominator. Two columns, not one."""
    oracle = pd.read_csv(ORACLE, float_precision="round_trip")
    sig, raw, fs, cat = zn_traces["zn_30h"]
    sub = cat[cat["passes_gate"]].reset_index(drop=True)
    exp = oracle[oracle["tag"] == "zn_30h"].reset_index(drop=True)
    assert not np.allclose(exp["snr"].to_numpy(), sub["sust_sigma"].to_numpy())
    fsc = EF.feature_scalars(sig, raw)
    assert np.allclose(exp["snr"].to_numpy(), exp["depth_sust_pA"].to_numpy() / fsc.noise_pA)


@corpus
def test_time_half_reproduces_the_owner_split(zn_traces):
    """54/54, 55/55, 97/97, 538/538 -- his own early/late column, exactly."""
    oracle = pd.read_csv(ORACLE, float_precision="round_trip")
    got = EF.time_half(oracle["onset_s"].to_numpy(), oracle["tag"].to_numpy())
    assert np.array_equal(got, oracle["time_half"].to_numpy())
    counts = {t: dict(pd.Series(got[oracle["tag"].to_numpy() == t]).value_counts()) for t in TAGS}
    assert counts["zn_2h"] == {"early": 54, "late": 54}
    assert counts["zn_10h"] == {"early": 55, "late": 55}
    assert counts["zn_18h"] == {"early": 97, "late": 97}
    assert counts["zn_30h"] == {"early": 538, "late": 538}


@corpus
def test_the_per_recording_scalars_match_the_extraction(zn_traces):
    """The three numbers a downstream module must not recompute from a subset."""
    expect = {  # tag -> (feature baseline, whole-trace MAD, global I0)
        "zn_2h": (+0.030268, 2.23, 197.0),
        "zn_10h": (-0.038005, 2.85, 162.0),
        "zn_18h": (-0.052270, 1.98, 169.0),
        "zn_30h": (-0.132807, 2.91, 253.0),
    }
    for tag, (med, mad, i0) in expect.items():
        sig, raw, _fs, _cat = zn_traces[tag]
        sc = EF.feature_scalars(sig, raw)
        assert sc.baseline_pA == pytest.approx(med, abs=1e-6)
        assert sc.noise_pA == pytest.approx(mad, abs=5e-3)
        assert round(sc.i0_pA) == i0
