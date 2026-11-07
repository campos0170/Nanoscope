"""Ground-truth and invariance checks for nonstationary-noise detection.

Seeds here differ from the original stress reproducer. Threshold policy is fixed
across all cases; no per-trace tuning or ground-truth event mask is supplied.
"""
import json
import warnings
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from nanoscope.analysis.baseline import BaselineConfig, baseline_wander_result, collect_baseline_warnings
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.detection_noise import detection_noise_profile
from nanoscope.analysis.filters import lowpass

FS = 2000.
N = 24000
T = np.arange(N) / FS
CFG = BaselineConfig(rough_s=2., base_s=.1)


def signal(seed, kind="white", shape="step", events=False, polarity=-1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=N)
    if kind == "ar":
        z = lfilter([1.], [1., -.9], z)
    elif kind == "filtered":
        z = lowpass(z, FS, 100., kind="bessel")
    elif kind == "spikes":
        z = rng.standard_t(3, N) / np.sqrt(3)
        z[rng.choice(N, 48, replace=False)] += rng.choice([-25., 25.], 48)
    scale = np.ones(N)
    if shape == "step":
        scale[(T >= 4.037) & (T < 8.039)] = 8.
    elif shape == "ramp":
        scale = 1. + 7. * np.sin(np.pi * T / 12.) ** 2
    elif shape == "burst":
        scale[(T >= 5.03) & (T < 5.43)] = 8.
    z *= scale
    z -= z.mean()
    z /= z.std()
    x = 200. + 2. * z
    truth = []
    if events:
        for start in (1., 2., 4.5, 5.5, 6.5, 7.5, 9., 10.):
            a, b = int(start * FS), int((start + .04) * FS)
            x[a:b] -= 40.
            truth.append((a, b))
    return -polarity * x, truth


def run(x, policy="auto", polarity=-1, supplied=False):
    with warnings.catch_warnings(), collect_baseline_warnings():
        warnings.simplefilter("ignore")
        if not supplied:
            x = baseline_wander_result(x, FS, method="masked_median", polarity=polarity,
                                       rough_s=2., base_s=.1).corrected
        return detect_events(x, FS, polarity=polarity, k_in=5., k_out=2.,
                             min_dwell_s=.002, baseline_config=CFG,
                             baseline=-polarity*200. if supplied else None,
                             noise_policy=policy)


def matches(df, truth):
    # Separated event ground truth: one predicted interval must overlap at least
    # half the union. Extra predicted fragments count against precision.
    found = set()
    for a, b in truth:
        for idx, row in df.iterrows():
            lo, hi = int(row.start_idx), int(row.end_idx)
            inter = max(0, min(b, hi) - max(a, lo))
            union = b - a + hi - lo - inter
            if inter / union >= .5 and idx not in found:
                found.add(idx)
                break
    return len(found)


@pytest.mark.parametrize("seed", [9001, 21073, 52591, 69107])
@pytest.mark.parametrize("shape", ["step", "ramp", "burst"])
@pytest.mark.parametrize("polarity", [-1, 1])
def test_white_noise_only_false_detections_eliminated(seed, shape, polarity):
    x, _ = signal(seed, shape=shape, polarity=polarity)
    before = x.copy()
    global_df, _ = run(x, "global", polarity)
    new, p = run(x, "auto", polarity)
    # Both paths now share the repaired baseline mask. It can remove all false
    # positives even before local detection is enabled, so global > 0 is not a
    # valid requirement here. The independent exact-baseline reproducer below
    # isolates and locks down the original detector-only failure.
    assert len(new) <= len(global_df)
    assert len(new) == 0
    assert p["noise_policy_effective"] == "local"
    assert p["threshold_in_pA"] is None
    assert p["threshold_in_range_pA"][1] > p["threshold_in_range_pA"][0]
    assert np.array_equal(x, before)


@pytest.mark.parametrize("seed", [9001, 21073, 52591])
@pytest.mark.parametrize("kind", ["white", "ar", "filtered", "spikes"])
def test_stationary_auto_is_exact_legacy_path(seed, kind):
    x, _ = signal(seed, kind=kind, shape="stationary", events=True)
    old, op = run(x, "global")
    new, np_ = run(x, "auto")
    assert np_["noise_policy_effective"] == "global"
    pd.testing.assert_frame_equal(new, old)
    assert np_["threshold_in_pA"] == op["threshold_in_pA"]
    assert np_["n_candidate_runs"] == op["n_candidate_runs"]


@pytest.mark.parametrize("seed", [9001, 21073, 52591])
@pytest.mark.parametrize("kind", ["white", "ar", "filtered"])
@pytest.mark.parametrize("shape", ["step", "ramp"])
def test_high_snr_events_recovered_with_local_noise(seed, kind, shape):
    x, truth = signal(seed, kind=kind, shape=shape, events=True)
    df, p = run(x)
    assert matches(df, truth) == len(truth)
    assert len(df) <= len(truth) + (0 if kind == "white" else 1)
    assert p["noise_policy_effective"] == "local"
    assert np.median(np.abs(df.depth_pA.to_numpy() - 40.)) / 40. < .1
    assert p["n_candidate_runs"] - p["n_rejected_below_enter"] - p["n_merged"] - p["n_rejected_short"] - p["n_rejected_excluded"] == len(df)


@pytest.mark.parametrize("seed", [9001, 21073, 52591])
@pytest.mark.parametrize("kind", ["ar", "filtered"])
@pytest.mark.parametrize("shape", ["step", "ramp", "burst"])
def test_correlated_noise_false_positive_rate_strongly_reduced(seed, kind, shape):
    x, _ = signal(seed, kind=kind, shape=shape)
    old, _ = run(x, "global")
    new, p = run(x, "auto")
    assert p["noise_policy_effective"] == "local"
    assert len(new) <= 2
    assert len(new) <= max(1, .1 * len(old))


def test_original_reproducer_exact_known_baseline_and_subtracted():
    for seed in (1729, 1730):
        z = np.random.default_rng(seed).normal(size=N) * np.where((T >= 4) & (T < 8), 3., .35)
        z -= z.mean()
        z *= 2. / z.std()
        for supplied in (False, True):
            df, p = run(200. + z, supplied=supplied)
            assert len(df) == 0
            assert p["noise_policy_effective"] == "local"
            if supplied:
                legacy, _ = run(200. + z, "global", supplied=True)
                assert len(legacy) == {1729: 80, 1730: 90}[seed]


def test_short_burst_with_exact_baseline_is_not_a_noise_scale_failure():
    x, _ = signal(52591, shape="burst")
    df, p = run(x, supplied=True)
    assert len(df) == 0
    assert p["noise_policy_effective"] == "local"


@pytest.mark.parametrize("seed", [777, 1153, 19319])
@pytest.mark.parametrize("amplitude_ratio", [1., 8.])
def test_changing_noise_correlation_does_not_confuse_variance(seed, amplitude_ratio):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=N)
    correlated = lfilter([1.], [1., -.98], rng.normal(size=8000))
    correlated /= correlated.std()
    noise[8000:16000] = amplitude_ratio * correlated
    with warnings.catch_warnings(), collect_baseline_warnings():
        warnings.simplefilter("ignore")
        old, _ = detect_events(200. + noise, FS, polarity=-1, baseline=200.,
                               noise_policy="global", min_dwell_s=.002)
        new, p = detect_events(200. + noise, FS, polarity=-1, baseline=200.,
                               noise_policy="auto", min_dwell_s=.002)
    if amplitude_ratio == 1:
        assert len(new) == len(old) == 0
    else:
        assert len(old) >= 10
        assert len(new) <= 1
        assert p["noise_policy_effective"] == "local"
    assert p["noise_local_scale_min_pA"] >= p["noise_rms_pA"]


@pytest.mark.parametrize("seed", [52591, 21073])
def test_true_events_inside_short_noise_burst_survive_subtraction(seed):
    x, _ = signal(seed, shape="burst")
    truth = [(1000, 1080), (10200, 10280), (15000, 15800)]
    for a, b in truth:
        x[a:b] -= 80.
    df, p = run(x)
    assert matches(df, truth) == len(truth)
    assert len(df) == len(truth)
    assert np.median(np.abs(df.depth_pA.to_numpy() - 80.)) / 80. < .1


@pytest.mark.parametrize("polarity", [-1, 1])
def test_long_and_boundary_events_survive_variable_noise_with_known_baseline(polarity):
    x, _ = signal(73019, shape="step", polarity=polarity)
    truth = [(0, 80), (2000, 2020), (8000, 8800), (15980, 16080), (N-80, N)]
    for a, b in truth:
        x[a:b] += polarity * 40.
    df, p = run(x, polarity=polarity, supplied=True)
    assert matches(df, truth) == len(truth)
    assert len(df) == len(truth)
    assert p["n_clipped_boundary"] == 2
    assert np.count_nonzero(~df.accepted.to_numpy()) == 2
    assert np.median(np.abs(df.depth_pA.to_numpy() - 40.)) / 40. < .1


def test_constant_noise_windows_are_disclosed_and_finite():
    x, _ = signal(39011)
    residual = x - 200.
    residual[:1000] = 0.
    track, p = detection_noise_profile(residual, FS, policy="local")
    assert np.isfinite(track).all() and np.all(track > 0)
    assert p["noise_warnings"]
    short, q = detection_noise_profile(np.arange(200.), FS, policy="local", global_noise=1.)
    assert short is None
    assert q["noise_policy_effective"] == "global"
    assert q["noise_warnings"]


def test_profile_reconstructs_exactly_and_mirrors_without_input_mutation():
    x, _ = signal(71821)
    residual = x - 200.
    before = residual.copy()
    a, p = detection_noise_profile(residual, FS)
    b, q = detection_noise_profile(-residual, FS)
    assert a is not None
    assert np.array_equal(a, b)
    assert np.array_equal(residual, before)
    assert p == q
    edges = np.r_[0, p["noise_profile_tile_ends"]]
    restored = np.repeat(p["noise_profile_scales_pA"], np.diff(edges))
    assert np.array_equal(a, restored)
    assert json.loads(json.dumps(p)) == p


def test_tiles_and_serialized_metadata_are_bounded():
    x = np.random.default_rng(31337).normal(size=600000)
    x[200000:400000] *= 10
    a, p = detection_noise_profile(x, 1., policy="local", window_s=.001)
    assert a.shape == x.shape
    assert len(p["noise_profile_tile_ends"]) <= 4096
    assert len(json.dumps(p)) < 160000
    assert np.isfinite(a).all() and np.all(a > 0)


@pytest.mark.parametrize("n", [1, 2, 127, 128, 255, 256, 383, 511])
def test_short_and_partial_windows_use_safe_fallback(n):
    x = np.random.default_rng(21073).normal(size=n)
    a, p = detection_noise_profile(x, FS, global_noise=1.)
    assert a is None
    assert p["noise_policy_effective"] == "global"
    assert p["noise_warnings"]


@pytest.mark.parametrize("kwargs", [{"policy":"bad"}, {"window_s":0}, {"window_s":-1}, {"window_s":float("nan")}, {"window_s":float("inf")}])
def test_invalid_recipe_is_refused(kwargs):
    with pytest.raises(ValueError):
        detection_noise_profile(np.arange(1000.), FS, **kwargs)


def test_cancellation_inside_noise_loop_returns_no_partial_profile():
    x, _ = signal(23293)
    count = [0]
    def cancel():
        count[0] += 1
        return count[0] >= 8
    with pytest.raises(OperationCancelled):
        detection_noise_profile(x - 200., FS, cancel=cancel)
    assert count[0] == 8


def test_profile_cache_reuses_only_identical_inputs_and_never_shares_mutable_outputs(monkeypatch):
    from nanoscope.analysis import detection_noise as module
    x, _ = signal(60623)
    x -= 200.
    module._PROFILE_CACHE.clear()
    original = module._compute_detection_noise_profile
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, '_compute_detection_noise_profile', counted)
    track, info = module.detection_noise_profile(x, FS)
    expected = track.copy()
    original_scale = info['noise_profile_scales_pA'][0]
    track[:] = -1
    info['noise_profile_scales_pA'][0] = -1
    info['noise_warnings'].append('caller mutation')
    repeated, record = module.detection_noise_profile(x.copy(), FS)
    assert len(calls) == 1
    np.testing.assert_array_equal(repeated, expected)
    assert record['noise_profile_scales_pA'][0] == original_scale
    assert 'caller mutation' not in record['noise_warnings']
    x[1000] += .25
    module.detection_noise_profile(x, FS)
    assert len(calls) == 2
    module.detection_noise_profile(x, FS, window_s=.125)
    assert len(calls) == 3
    module.detection_noise_profile(x, FS, policy='local')
    assert len(calls) == 4
    module.detection_noise_profile(x, FS, global_noise=2.)
    assert len(calls) == 5
    assert len(module._PROFILE_CACHE) <= 4
    assert all(not isinstance(value, np.ndarray)
               for metadata in module._PROFILE_CACHE.values() for value in metadata.values())
