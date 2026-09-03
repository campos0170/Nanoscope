"""Tests for nanoscope.analysis.truth.

Ported from the SOURCE ``tests/test_truth.py`` (PORT_PLAN §3.7) to the target's
signal-first data model. The SOURCE installed a *reference* threshold detector
into ``sys.modules`` only when the real detector was absent; the target now ships
a real signal-first detector (``analysis.detect.detect_events``), so the port
points ``_detect()`` straight at it through a thin ``SynthRecording``-consuming
adapter. Every assertion and pinned value is preserved unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from nanoscope.analysis.events import EventSet
from nanoscope.analysis.truth import (
    aaft_surrogate,
    null_control,
    recovery_map,
    sweep_threshold,
    synth_trace,
)


# ---------------------------------------------------------------------------
# detector adapter: wrap the signal-first detect_events into a rec-consuming
# EventSet-returning callable, matching the SOURCE detect(rec) call shape.
# ---------------------------------------------------------------------------
def _detect():
    from nanoscope.analysis.detect import detect_events

    def detect(rec, **kwargs) -> EventSet:
        # There is no stored polarity to cross-check in the signal-first model,
        # so the SOURCE's `check_polarity` flag is a no-op here.
        kwargs.pop("check_polarity", None)
        df, prov = detect_events(
            rec.data,
            rec.sample_rate_hz,
            polarity=rec.polarity,
            recording=rec.name,
            **kwargs,
        )
        return EventSet(df, prov)

    return detect


# ---------------------------------------------------------------------------
# synth_trace -- the foundation, so check it hard
# ---------------------------------------------------------------------------


def test_synth_trace_shape_and_metadata():
    rec, gt = synth_trace(fs=50_000.0, duration_s=2.0, n_events=50, seed=1)
    assert rec.n == 100_000
    assert rec.fs == 50_000.0
    assert rec.current.dtype == np.float32
    assert len(gt) == 50
    assert {"t_start_s", "dwell_s", "depth_pA", "frac_depth"} <= set(gt.columns)


def test_synth_trace_is_deterministic():
    a, ga = synth_trace(duration_s=1.0, n_events=20, seed=7)
    b, gb = synth_trace(duration_s=1.0, n_events=20, seed=7)
    assert np.array_equal(a.current, b.current)
    pd.testing.assert_frame_equal(ga, gb)
    c, _ = synth_trace(duration_s=1.0, n_events=20, seed=8)
    assert not np.array_equal(a.current, c.current)


def test_synth_trace_events_do_not_overlap_and_are_sorted():
    _, gt = synth_trace(duration_s=5.0, n_events=300, seed=3)
    assert gt["t_start_s"].is_monotonic_increasing
    assert (gt["start_idx"].to_numpy()[1:] >= gt["end_idx"].to_numpy()[:-1]).all()


def test_synth_trace_sign_convention_dev_is_positive_during_events():
    """polarity * (current - baseline) must be positive during events, for both signs of I0."""
    for i0 in (-1000.0, +1000.0):
        rec, gt = synth_trace(
            duration_s=2.0, i0_pA=i0, noise_pA=1.0, n_events=30,
            depth_frac_range=(0.2, 0.4), dwell_range_s=(1e-3, 2e-3), seed=2,
        )
        assert rec.polarity == int(-np.sign(i0))
        idx = np.concatenate(
            [np.arange(s, e) for s, e in zip(gt["start_idx"], gt["end_idx"])]
        )
        dev = rec.polarity * (rec.current[idx].astype(np.float64) - i0)
        assert dev.mean() > 0
        assert np.median(dev) > 0
        assert (gt["depth_pA"] > 0).all()
        assert (gt["frac_depth"] > 0).all()
        # occlusion reduces |I|
        assert np.abs(rec.current[idx]).mean() < abs(i0)


def test_synth_trace_ground_truth_depth_matches_the_trace():
    rec, gt = synth_trace(
        fs=50_000.0, duration_s=4.0, i0_pA=-1000.0, noise_pA=0.0, n_events=40,
        dwell_range_s=(1e-3, 2e-3), depth_frac_range=(0.1, 0.4), seed=5,
    )
    measured = np.array(
        [
            rec.polarity * (rec.current[s:e].astype(np.float64) - (-1000.0)).mean()
            for s, e in zip(gt["start_idx"], gt["end_idx"])
        ]
    )
    np.testing.assert_allclose(measured, gt["depth_pA"], rtol=1e-3)


def test_synth_trace_noise_rms_matches_request():
    rec, _ = synth_trace(duration_s=4.0, noise_pA=12.0, n_events=0, seed=4)
    assert np.isclose(rec.current.astype(np.float64).std(), 12.0, rtol=0.02)


def test_synth_trace_noise_is_pink_not_white():
    """1/f must be present: low-frequency power well above the high-frequency floor."""
    rec, _ = synth_trace(duration_s=8.0, noise_pA=15.0, n_events=0, pink_frac=0.5, seed=6)
    f, p = signal.welch(rec.current.astype(np.float64), fs=rec.fs, nperseg=8192)
    low = p[(f > 5) & (f < 50)].mean()
    high = p[(f > 10_000) & (f < 20_000)].mean()
    assert low > 10 * high


def test_synth_trace_zero_events_returns_empty_truth():
    rec, gt = synth_trace(duration_s=1.0, n_events=0, seed=0)
    assert len(gt) == 0
    assert {"t_start_s", "dwell_s", "depth_pA", "frac_depth"} <= set(gt.columns)


def test_synth_trace_drift_and_line_appear():
    rec, _ = synth_trace(duration_s=4.0, noise_pA=1.0, n_events=0, drift_pA=50.0, seed=0)
    half = rec.n // 2
    assert abs(rec.current[:half].mean() - rec.current[half:].mean()) > 20
    rec2, _ = synth_trace(duration_s=4.0, noise_pA=1.0, n_events=0, line_pA=20.0, seed=0)
    f, p = signal.welch(rec2.current.astype(np.float64), fs=rec2.fs, nperseg=int(rec2.fs))
    assert p[np.argmin(np.abs(f - 60.0))] > 100 * np.median(p)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fs": 0},
        {"duration_s": 0},
        {"i0_pA": 0.0},
        {"noise_pA": -1.0},
        {"n_events": -5},
        {"depth_frac_range": (0.1, 1.5)},
        {"dwell_range_s": (2e-3, 1e-4)},
        {"pink_frac": 1.5},
        {"dwell_range_s": (1e-9, 1e-8)},  # under one sample
    ],
)
def test_synth_trace_rejects_nonsense(kwargs):
    base = {"duration_s": 1.0, "n_events": 5}
    base.update(kwargs)
    with pytest.raises(ValueError):
        synth_trace(**base)


def test_synth_trace_raises_when_events_cannot_fit():
    with pytest.raises(ValueError, match="cannot place"):
        synth_trace(duration_s=0.1, n_events=500, dwell_range_s=(1e-3, 2e-3), seed=0)


# ---------------------------------------------------------------------------
# AAFT
# ---------------------------------------------------------------------------


def test_aaft_preserves_amplitude_distribution_exactly():
    rec, _ = synth_trace(duration_s=2.0, n_events=100, seed=11)
    s = aaft_surrogate(rec.current, rng=0)
    assert s.shape == rec.current.shape
    assert s.dtype == rec.current.dtype
    np.testing.assert_array_equal(np.sort(s), np.sort(rec.current))


def test_aaft_preserves_the_power_spectrum():
    """Welch PSDs must agree band-by-band -- 1/f, line and roll-off all survive.

    READ THE FIXTURE BEFORE YOU TRUST THIS TEST. It is Gaussian (`synth_trace`'s noise
    is a white/pink mix, kurtosis 3.000), and on Gaussian noise AAFT's rank-remap has
    nothing to bias. On the owner's own recordings the same measurement fails badly:
    band by band, the surrogate's power exceeds the original's by up to 2.2x on the apo
    -600 mV file, 6.4x on ferritin_data2 and 14.2x on ferritin_data3. This test passing
    is therefore not evidence that the property holds for a real recording -- which is
    why `null_control` now checks whether the surrogate it built is a null of the trace
    it came from, instead of assuming it. See
    ``test_the_null_refuses_to_report_a_rate_when_the_surrogate_is_not_one``.
    """
    rec, _ = synth_trace(
        duration_s=8.0, noise_pA=15.0, n_events=0, pink_frac=0.5, line_pA=10.0, seed=12
    )
    x = rec.current.astype(np.float64)
    s = aaft_surrogate(x, rng=1).astype(np.float64)

    f, px = signal.welch(x, fs=rec.fs, nperseg=8192)
    _, ps = signal.welch(s, fs=rec.fs, nperseg=8192)
    m = f > 0
    edges = np.logspace(np.log10(f[m].min()), np.log10(f[m].max()), 13)
    b = np.digitize(f[m], edges)
    for i in range(1, len(edges)):
        sel = b == i
        if sel.sum() < 3:
            continue
        a, c = px[m][sel].mean(), ps[m][sel].mean()
        assert abs(np.log(c / a)) < 0.35, f"band {edges[i - 1]:.0f}-{edges[i]:.0f} Hz off by {c / a:.2f}x"


def test_aaft_destroys_events():
    """A surrogate of an event-rich trace must yield ~no detections."""
    rec, gt = synth_trace(
        duration_s=10.0, n_events=200, noise_pA=15.0,
        depth_frac_range=(0.2, 0.4), dwell_range_s=(2e-4, 2e-3), seed=13,
    )
    detect = _detect()
    # min_dwell above one sample: AAFT preserves the amplitude distribution exactly,
    # so the deep event-level samples survive as scattered single-sample spikes; a
    # realistic minimum dwell (here 7-8 samples) rejects them as sub-resolution, which
    # is what any researcher does. Without it a permissive detector's own false-
    # positive rate -- not AAFT -- would dominate the count.
    #
    # 5 samples (1e-4) used to be enough and is not any more. The raised floor is
    # CORRECT, and `test_the_raised_dwell_floor_is_the_price_of_not_eating_events`
    # below pins the reason rather than the number -- the earlier note here guessed at
    # it ("the block median dived into the surrogate's excursions") and the guess was
    # only half right. Measured, same surrogate, k_in=5, both estimators:
    #
    #   threshold crossings   6641 (old 10 ms block median) vs 6592 (event-masked)
    #   detection threshold   70.5 pA                       vs 74.1 pA
    #   runs >= 3 samples      152                          vs  352
    #   runs >= 5 samples       12                          vs   36
    #
    # So it is not more crossings (0.7 % FEWER) and not a lower bar (the new one is
    # 5 % HIGHER). It is longer runs: the 10 ms window tracks the residual's own
    # low-frequency wander and cuts each crossing short, and the 0.5/0.1 s masked
    # estimator deliberately does not -- which is the same property that stops it
    # diving into real events. A minimum dwell is a coherence requirement against the
    # residual's autocorrelation, and the residual's coherence time is longer under an
    # estimator that refuses to follow it. One more sample of required coherence
    # removes all 36 and none of the 200 real ones.
    md = 1.5e-4
    n_real = len(detect(rec, k_in=5.0, min_dwell_s=md).df)
    assert n_real > 0.8 * len(gt)

    # A surrogate has no events, so its measured polarity is noise-driven; silence
    # the detector's polarity guard, which exists to catch a mislabelled real trace.
    surr = rec.with_current(aaft_surrogate(rec.current, rng=2))
    n_surr = len(detect(surr, k_in=5.0, min_dwell_s=md, check_polarity=False).df)
    assert n_surr < 0.1 * n_real, f"AAFT surrogate kept {n_surr} of {n_real} detections"


def test_aaft_is_reproducible_and_varies_with_seed():
    x = synth_trace(duration_s=1.0, n_events=10, seed=14)[0].current
    assert np.array_equal(aaft_surrogate(x, rng=3), aaft_surrogate(x, rng=3))
    assert not np.array_equal(aaft_surrogate(x, rng=3), aaft_surrogate(x, rng=4))


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.zeros(3)])
def test_aaft_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        aaft_surrogate(bad)


# ---------------------------------------------------------------------------
# null_control
# ---------------------------------------------------------------------------


def test_null_control_reports_low_fdr_on_clean_events():
    rec, _ = synth_trace(
        duration_s=6.0, n_events=120, noise_pA=15.0,
        depth_frac_range=(0.2, 0.4), dwell_range_s=(2e-4, 2e-3), seed=15,
    )
    out = null_control(rec, detect_kwargs={"k_in": 5.0, "min_dwell_s": 1e-4}, n_surrogates=3, rng=0)
    assert set(out) == {
        "n_real", "n_event_free", "n_surrogate_mean", "n_surrogate_std",
        "estimated_fdr", "per_surrogate", "notes",
    }
    assert out["n_real"] > 90
    assert len(out["per_surrogate"]) == 3
    assert out["estimated_fdr"] < 0.1
    # On a Gaussian fixture the surrogate IS a null of the trace, so the control has
    # nothing to say and reports a rate. See test_the_null_refuses_... below.
    assert out["notes"] == ()
    assert out["n_event_free"] == 0


def test_null_control_low_fdr_under_default_detect_settings():
    """Regression: the surrogate must be built from event-*removed* noise.

    AAFT preserves the amplitude distribution exactly, so surrogating the raw trace
    keeps the deep event samples and the detector re-fires on them. With the default
    min_dwell_s (which does not gate the resulting short spurious runs) that pinned
    estimated_fdr to 1.0 for any trace with real events -- a confidently-wrong control.
    """
    rec, _ = synth_trace(
        duration_s=6.0, n_events=120, noise_pA=15.0,
        depth_frac_range=(0.2, 0.4), dwell_range_s=(2e-4, 2e-3), seed=15,
    )
    out = null_control(rec, n_surrogates=3, rng=0)  # default detect settings
    assert out["n_real"] > 90
    assert out["estimated_fdr"] < 0.1


def test_null_control_fdr_is_nan_when_nothing_is_real():
    """No real detections means the false-positive *rate* is undefined, not zero."""
    rec, _ = synth_trace(duration_s=2.0, n_events=0, noise_pA=15.0, seed=16)
    out = null_control(rec, detect_kwargs={"k_in": 12.0}, n_surrogates=2, rng=0)
    assert out["n_real"] == 0
    assert np.isnan(out["estimated_fdr"])


def test_null_control_rejects_nonsense():
    rec, _ = synth_trace(duration_s=1.0, n_events=5, seed=0)
    with pytest.raises(ValueError):
        null_control(rec, n_surrogates=0)


# ---------------------------------------------------------------------------
# sweep_threshold
# ---------------------------------------------------------------------------


def test_sweep_threshold_finds_a_plateau_on_clean_data():
    rec, gt = synth_trace(
        duration_s=10.0, n_events=100, noise_pA=15.0,
        depth_frac_range=(0.2, 0.4), dwell_range_s=(3e-4, 2e-3), seed=17,
    )
    df = sweep_threshold(
        rec, k_values=np.arange(2.5, 10.01, 0.5),
        # See test_aaft_destroys_events: the surrogate's false positives all sit at
        # the dwell floor, and the block median used to hide them by eating them.
        detect_kwargs={"min_dwell_s": 1.5e-4}, rng=0
    )

    assert list(df.columns) == [
        "k", "n_events", "median_dwell_s", "median_frac_depth",
        "n_surrogate", "dlogn_dk", "in_plateau",
    ]
    assert df["in_plateau"].any(), "no plateau found on data engineered to have one"

    k_rec = df.attrs["k_recommended"]
    assert np.isfinite(k_rec)
    assert 3.0 <= k_rec <= 10.0

    # inside the plateau the count is right and insensitive to k
    p = df[df["in_plateau"]]
    assert len(p) >= 4
    assert p["n_events"].max() / max(p["n_events"].min(), 1) < 1.2
    assert abs(p["n_events"].median() - len(gt)) < 0.15 * len(gt)

    # the surrogate finds nothing where the real trace plateaus
    assert p["n_surrogate"].max() < 0.1 * p["n_events"].median()


def test_sweep_threshold_count_falls_monotonically_ish_with_k():
    rec, _ = synth_trace(
        duration_s=6.0, n_events=100, noise_pA=15.0,
        depth_frac_range=(0.05, 0.4), dwell_range_s=(2e-4, 2e-3), seed=18,
    )
    df = sweep_threshold(rec, k_values=np.arange(3.0, 9.01, 1.0), rng=0)
    assert df["n_events"].iloc[0] >= df["n_events"].iloc[-1]


def test_sweep_threshold_no_plateau_in_pure_noise():
    """Pure noise has no detection regime; k_recommended must be nan, not a guess."""
    rec, _ = synth_trace(duration_s=6.0, n_events=0, noise_pA=15.0, seed=19)
    df = sweep_threshold(rec, k_values=np.arange(2.5, 6.01, 0.5), rng=0, plateau_tol=0.15)
    assert not df["in_plateau"].any()
    assert np.isnan(df.attrs["k_recommended"])


@pytest.mark.parametrize("k_values", [np.array([5.0]), np.array([5.0, 4.0, 6.0])])
def test_sweep_threshold_rejects_nonsense(k_values):
    rec, _ = synth_trace(duration_s=1.0, n_events=5, seed=0)
    with pytest.raises(ValueError):
        sweep_threshold(rec, k_values=k_values)


# ---------------------------------------------------------------------------
# recovery_map
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _rec_for_recovery():
    return synth_trace(
        duration_s=10.0, fs=50_000.0, i0_pA=-1000.0, noise_pA=15.0, n_events=100,
        depth_frac_range=(0.2, 0.4), dwell_range_s=(3e-4, 2e-3), seed=20,
    )[0]


def test_recovery_map_has_the_expected_shape(_rec_for_recovery):
    """Efficiency ~1 for long+deep events, ~0 for short+shallow ones."""
    df = recovery_map(
        _rec_for_recovery,
        detect_kwargs={"k_in": 5.0, "filter_cutoff_hz": 10_000.0},
        dwell_grid=np.array([2e-5, 5e-3]),
        depth_grid=np.array([0.02, 0.40]),
        n_per_cell=20,
        seed=0,
    )
    assert len(df) == 4
    assert {"dwell_s", "frac_depth", "n_injected", "n_recovered", "efficiency",
            "depth_bias"} <= set(df.columns)
    assert (df["n_injected"] == 20).all()
    assert df["efficiency"].between(0, 1).all()

    g = df.set_index([df["dwell_s"].round(9), df["frac_depth"].round(4)])["efficiency"]
    long_deep = g.loc[(round(5e-3, 9), 0.40)]
    short_shallow = g.loc[(round(2e-5, 9), 0.02)]
    assert long_deep > 0.9, f"long+deep events must be recovered, got {long_deep:.2f}"
    assert short_shallow < 0.2, f"short+shallow events must be missed, got {short_shallow:.2f}"
    assert long_deep > short_shallow


def test_recovery_map_efficiency_is_monotone_in_depth(_rec_for_recovery):
    df = recovery_map(
        _rec_for_recovery,
        detect_kwargs={"k_in": 5.0, "filter_cutoff_hz": 10_000.0},
        dwell_grid=np.array([5e-4, 2e-3]),
        depth_grid=np.array([0.02, 0.10, 0.40]),
        n_per_cell=15,
        seed=1,
    )
    for _, sub in df.groupby("dwell_s"):
        e = sub.sort_values("frac_depth")["efficiency"].to_numpy()
        assert e[0] <= e[-1] + 1e-9
        assert e[-1] > 0.8


def test_recovery_map_measures_filter_attenuation(_rec_for_recovery):
    """depth_bias < 1 for events shorter than the filter rise time."""
    df = recovery_map(
        _rec_for_recovery,
        detect_kwargs={"k_in": 4.0, "filter_cutoff_hz": 10_000.0},
        dwell_grid=np.array([4e-5, 4e-3]),
        depth_grid=np.array([0.20, 0.40]),
        n_per_cell=15,
        seed=2,
    )
    short = df[df["dwell_s"] < 1e-4]["depth_bias"].median()
    long = df[df["dwell_s"] > 1e-3]["depth_bias"].median()
    assert short < long, f"short depth_bias {short:.2f} should be below long {long:.2f}"
    assert short < 0.95


def test_recovery_map_output_is_consumable_by_stats(_rec_for_recovery):
    """attrs must carry an efficiency callable that fit_dwell_exponential can weight with."""
    df = recovery_map(
        _rec_for_recovery,
        detect_kwargs={"k_in": 5.0, "filter_cutoff_hz": 10_000.0},
        dwell_grid=np.array([2e-5, 5e-4, 5e-3]),
        depth_grid=np.array([0.05, 0.40]),
        n_per_cell=10,
        seed=3,
    )
    for key in ("efficiency_fn", "dwell_grid", "depth_grid", "efficiency_grid"):
        assert key in df.attrs
    assert df.attrs["efficiency_grid"].shape == (3, 2)

    fn = df.attrs["efficiency_fn"]
    e = fn(np.array([3e-5, 1e-3]), frac_depth=0.40)
    assert e.shape == (2,)
    assert ((0.0 <= e) & (e <= 1.0)).all()
    assert e[1] >= e[0]

    # marginalised over the depth grid
    em = fn(np.array([1e-3]))
    assert em.shape == (1,)
    assert 0.0 <= em[0] <= 1.0

    # extrapolation must still be a probability
    assert ((0.0 <= fn(np.array([1e-9, 1.0]), frac_depth=0.9)) <= 1.0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dwell_grid": np.array([1e-3])},
        {"depth_grid": np.array([0.1])},
        {"dwell_grid": np.array([2e-3, 1e-3])},
        {"n_per_cell": 0},
        {"dwell_grid": np.array([1e-9, 1e-8])},  # under one sample
    ],
)
def test_recovery_map_rejects_nonsense(_rec_for_recovery, kwargs):
    base = {"n_per_cell": 2}
    base.update(kwargs)
    with pytest.raises(ValueError):
        recovery_map(_rec_for_recovery, **base)


# ---------------------------------------------------------------------------
# the surrogate false-positive floor: WHY it moved, and what the null is worth
# ---------------------------------------------------------------------------
def _lowpass_drifting(*, drift_pA: float, n: int = 600_000, fs: float = 50_000.0,
                      cut_hz: float = 5_000.0, noise_pA: float = 12.0,
                      depth_pA: float = 250.0, seed: int = 4):
    """Band-limited GAUSSIAN noise, square events, and an optional slow drift.

    Deliberately the friendliest possible case for the AAFT null -- the noise is
    Gaussian, so `aaft_surrogate`'s rank-remap has nothing to bias -- with ONE knob
    that makes a real recording a real recording: the open pore wanders. Every
    instrument has a low-pass, so the noise is band-limited too.
    """
    from nanoscope.analysis.truth import SynthRecording

    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    b, a = signal.butter(4, cut_hz / (fs / 2.0), "low")
    y = signal.filtfilt(b, a, rng.standard_normal(n))
    y = noise_pA * y / y.std()
    ev = np.zeros(n)
    for s in range(5_000, n - 5_000, 12_000):
        ev[s:s + 150] = -depth_pA
    x = -1000.0 + drift_pA * np.sin(2.0 * np.pi * 0.7 * t) + y + ev
    return SynthRecording(data=x.astype(np.float32), sample_rate_hz=fs,
                          polarity=-1, name="lowpass_drift")


def test_the_raised_dwell_floor_is_the_price_of_not_eating_events():
    """WHY the surrogate floor tripled (12 -> 36) when the default estimator changed.

    Three hypotheses were on the table: the new estimator makes more false positives,
    the old estimator's null was flattered by its own diving baseline, or it is a
    threshold interaction. Measured, it is none of the first and all of the second, by
    a mechanism the earlier note guessed wrong: the crossings and the threshold are
    essentially unchanged -- the new estimator's bar is HIGHER -- and what grows is the
    LENGTH of each crossing. A 10 ms window follows the residual's low-frequency wander
    and cuts a crossing short; the 0.5/0.1 s masked estimator refuses to follow it,
    which is exactly the property that stops it diving into real events.

    So the floor is a coherence requirement against the residual's autocorrelation, and
    a floor calibrated against a fast-tracking estimator is too permissive under a slow
    one. This test pins the mechanism, not the number 36.
    """
    from nanoscope.analysis.baseline import DEFAULT_BASELINE, BaselineConfig
    from nanoscope.analysis.detect import detect_events

    rec, _ = synth_trace(
        duration_s=10.0, n_events=200, noise_pA=15.0,
        depth_frac_range=(0.2, 0.4), dwell_range_s=(2e-4, 2e-3), seed=13,
    )
    surr = rec.with_current(aaft_surrogate(rec.current, rng=2))
    old = BaselineConfig(method="block_median", window_s=0.01)

    stats = {}
    for name, cfg in (("old", old), ("new", DEFAULT_BASELINE)):
        df, prov = detect_events(surr.data, surr.fs, polarity=surr.polarity,
                                 recording="s", baseline_config=cfg,
                                 k_in=5.0, min_dwell_s=0.0)
        runs = np.round(df["dwell_s"].to_numpy() * surr.fs).astype(int)
        stats[name] = {
            "crossings": int(runs.size),
            "threshold": 5.0 * float(prov["noise_rms_pA"]),
            "ge3": int((runs >= 3).sum()),
            "ge5": int((runs >= 5).sum()),
        }

    a, b = stats["old"], stats["new"]
    # 1. Not more crossings: the same threshold excursions, within a couple of percent.
    assert abs(b["crossings"] - a["crossings"]) < 0.05 * a["crossings"], stats
    # 2. Not a lower bar: the event-masked estimator's noise estimate is HIGHER.
    assert b["threshold"] > a["threshold"], stats
    # 3. Longer runs. That, and only that, is where the extra false positives come from.
    assert b["ge3"] > 1.8 * a["ge3"], stats
    assert b["ge5"] > 1.8 * a["ge5"], stats


def test_the_null_refuses_to_report_a_rate_when_the_surrogate_is_not_one():
    """THE CONTROL CHECKS ITS OWN PREMISE.

    ``null_control`` returned ``estimated_fdr = 1.00`` on all three of the owner's real
    recordings -- apo -600 mV, ferritin_data2, ferritin_data3 -- and the panel printed
    "estimated false-positive rate: 100.0%". It was not measuring a false-positive
    rate. The surrogates were producing 4040 / 8225 / 6059 detections where the very
    traces they were built from produce 56 / 213 / 556.

    The fixture here is the friendliest case there is: Gaussian, band-limited noise
    with clean square events. With no drift the AAFT surrogate IS a null of it and the
    control reports a rate. Add the one thing every real pore does -- a slow wander --
    and the same surrogate manufactures hundreds of detections out of noise that yields
    none, because phase randomisation is a null for a stationary spectrum plus Gaussian
    noise and a drifting recording is not that.
    """
    clean = null_control(_lowpass_drifting(drift_pA=0.0),
                         detect_kwargs={"k_in": 5.0, "min_dwell_s": 1e-4},
                         n_surrogates=3, rng=0)
    assert clean["notes"] == ()
    assert clean["estimated_fdr"] < 0.1
    assert clean["n_event_free"] <= 2

    drifting = null_control(_lowpass_drifting(drift_pA=200.0),
                            detect_kwargs={"k_in": 5.0, "min_dwell_s": 1e-4},
                            n_surrogates=3, rng=0)
    assert drifting["n_real"] > 40
    assert drifting["n_surrogate_mean"] > 100 * max(drifting["n_event_free"], 1)
    assert np.isnan(drifting["estimated_fdr"]), (
        "a rate read off a surrogate that is not a null of the recording is worse "
        "than no rate at all"
    )
    note = " ".join(drifting["notes"])
    assert "not the same noise" in note
    assert str(drifting["n_event_free"]) in note
    assert "n_event_free" in note


def test_the_null_reports_the_direct_false_positive_count_either_way():
    """``n_event_free`` is the detector run on this recording's own noise with its
    events excised and refilled from its quiet stretches -- no surrogate anywhere. It
    is the number to look at when the surrogate is refused, and it costs one extra
    detection out of ``n_surrogates + 2``."""
    rec, _ = synth_trace(duration_s=6.0, n_events=120, noise_pA=15.0,
                         depth_frac_range=(0.2, 0.4), dwell_range_s=(2e-4, 2e-3),
                         seed=15)
    out = null_control(rec, detect_kwargs={"k_in": 5.0, "min_dwell_s": 1e-4},
                       n_surrogates=2, rng=0)
    assert out["n_event_free"] == 0
    assert out["estimated_fdr"] == 0.0

    empty = null_control(synth_trace(duration_s=2.0, n_events=0, noise_pA=15.0,
                                     seed=16)[0],
                         detect_kwargs={"k_in": 12.0}, n_surrogates=2, rng=0)
    assert empty["n_real"] == 0
    assert np.isnan(empty["estimated_fdr"])
    assert any("undefined" in n for n in empty["notes"])


def test_the_faithfulness_rule_is_loose_enough_not_to_fire_on_poisson_scatter():
    """It is a check for a BROKEN premise, not a significance test: two counts of the
    same rare thing must not trip it just by scattering."""
    from nanoscope.analysis.truth import _surrogate_is_faithful

    assert _surrogate_is_faithful(0, 0.0)
    assert _surrogate_is_faithful(0, 9.0)          # small counts get an allowance
    assert not _surrogate_is_faithful(0, 11.0)
    assert _surrogate_is_faithful(100, 300.0)      # 3x is still tolerated
    assert not _surrogate_is_faithful(100, 400.0)
    assert not _surrogate_is_faithful(56, 4040.0)  # the apo file, as measured


def test_the_sweep_says_whether_its_own_surrogate_column_means_anything():
    """Same question as `null_control`'s, promoted from a test assertion to a runtime
    one. ``test_sweep_threshold_finds_a_plateau_on_clean_data`` asserts the surrogate
    finds almost nothing where the real trace plateaus; when instead it OUTNUMBERS the
    real detections, phase randomisation manufactured events rather than destroying
    them, and ``n_surrogate`` is an artefact. Measured on the owner's -600 mV apo file:
    ~21 000 surrogate detections at every k against 240-652 real ones, and flat in k.
    """
    rec, _ = synth_trace(duration_s=10.0, n_events=100, noise_pA=15.0,
                         depth_frac_range=(0.2, 0.4), dwell_range_s=(3e-4, 2e-3),
                         seed=17)
    ok = sweep_threshold(rec, k_values=np.arange(2.5, 10.01, 0.5),
                         detect_kwargs={"min_dwell_s": 1.5e-4}, rng=0)
    assert ok.attrs["surrogate_is_a_null"] is True
    assert ok.attrs["surrogate_note"] == ""

    drifting = sweep_threshold(_lowpass_drifting(drift_pA=200.0),
                               k_values=np.arange(4.0, 8.01, 1.0),
                               detect_kwargs={"min_dwell_s": 1e-4}, rng=0)
    assert drifting.attrs["surrogate_is_a_null"] is False
    assert "not a false-positive count" in drifting.attrs["surrogate_note"]
    # the plateau is read off n_events alone and is not disturbed by any of this
    assert set(drifting.columns) >= {"k", "n_events", "n_surrogate", "in_plateau"}
