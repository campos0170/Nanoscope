"""Scientific regressions for rate round-off, kernel parity and saved identities."""
import math

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import savgol_filter

from nanoscope.analysis import baseline as B
from nanoscope.analysis import ml_preprocess as ML
from nanoscope.analysis.sample_count import floor_sample_count, ceil_sample_count
from nanoscope.analysis.baseline_mask import fill_masked_background


FS = 2000.0
INFERRED = 1999.9999999999982
ROUND_OFF_RATES = [INFERRED, np.nextafter(FS, 0), np.nextafter(FS, np.inf), FS]


def trace():
    t = np.arange(24000) / FS
    y = 100 + 2 * np.sin(2 * np.pi * t / 16)
    for a, b in ((1, 1.03), (3, 3.2), (5, 5.007), (8, 8.4), (10, 10.05)):
        y[int(a * FS):int(b * FS)] -= 20
    y += np.random.default_rng(471).normal(0, 20 / 32, len(y))
    return (y.astype(np.float32) * 2.4 + 14).astype(np.float32)


@pytest.mark.parametrize('value,expected', [
    (99.99999999999991, 100), (100.00000000000009, 100),
    ((.3 - .2) * 1000, 100), ((.1 + .2) * 1000, 300),
    (2.9999999999999996, 3), (np.nextafter(1., 0.), 1),
    (np.nextafter(1., np.inf), 1),
])
def test_near_integer_floor_and_ceil_share_one_boundary(value, expected):
    assert floor_sample_count(value) == ceil_sample_count(value) == expected


@pytest.mark.parametrize('value', [99.999999, 100.000001, .125, .5, 1e-100,
                                  1e12 + .125, 1e15 + .25, -100.001, -99.999])
def test_real_fractional_counts_keep_original_floor_and_ceil(value):
    assert floor_sample_count(value) == math.floor(value)
    assert ceil_sample_count(value) == math.ceil(value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_sample_counts_refuse(value):
    with pytest.raises(ValueError, match='finite'):
        floor_sample_count(value)
    with pytest.raises(ValueError, match='finite'):
        ceil_sample_count(value)


def test_tolerance_is_narrow_and_does_not_redefine_real_fractional_rates():
    boundary = 100.
    near = boundary - 16 * math.ulp(boundary)
    outside = boundary - 17 * math.ulp(boundary)
    assert floor_sample_count(near) == 100
    assert floor_sample_count(outside) == 99
    assert floor_sample_count(.5 * 1999.99 / 2) == 499
    assert floor_sample_count(.5 * 2000.01 / 2) == 500
    assert ceil_sample_count(np.nextafter(0., 1.)) == 1


@pytest.mark.parametrize('method', ['masked_median', 'robust_median', 'savgol'])
@pytest.mark.parametrize('fs', ROUND_OFF_RATES)
def test_same_samples_and_settings_produce_identical_baselines_near_rate_boundary(method, fs):
    y = trace(); original = y.copy()
    cfg = B.BaselineConfig(method=method)
    expected = B.characterisation_baseline(y, FS, config=cfg, polarity=-1, health=False, cache=False)
    actual = B.characterisation_baseline(y, fs, config=cfg, polarity=-1, health=False, cache=False)
    np.testing.assert_array_equal(actual.baseline, expected.baseline)
    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize('fs', ROUND_OFF_RATES)
def test_health_mask_reconstructs_exact_same_masked_kernel(fs):
    y = trace(); cfg = B.BaselineConfig()
    rough, mask, sigma, upper = B._rough_pass(y, fs, config=cfg)
    nominal = B._rough_pass(y, FS, config=cfg)
    np.testing.assert_array_equal(rough, nominal[0]); np.testing.assert_array_equal(mask, nominal[1])
    assert sigma == nominal[2] and upper == nominal[3]
    filled = fill_masked_background(y, mask, fs, cfg.base_s, robust_flanks=True)
    reconstructed = B.decimated_median(filled, fs, cfg.base_s, cfg.resolved_decim(fs))
    _, kernel = ML.baseline_masked_median(y, fs, **cfg.kernel_kwargs(fs))
    np.testing.assert_array_equal(reconstructed, kernel)


@pytest.mark.parametrize('fs', ROUND_OFF_RATES)
def test_decimation_dilation_and_minimum_length_consistently_keep_supported_windows(fs):
    cfg = B.BaselineConfig(base_s=.075, rough_s=.375)
    assert cfg.resolved_decim(fs) == 2
    assert B.BaselineConfig(base_s=.003, rough_s=.015, decim=2).resolved_decim(fs) == 2
    full = B.BaselineConfig(method='robust_median', window_s=.1)
    adapted, token = B.adapt_to_length(full, 200, fs)
    assert adapted == full and token is None
    #15samples at the default5:1 ratio support exactly3tracking samples.
    adapted, token = B.adapt_to_length(B.BaselineConfig(), 15, fs)
    assert token == B.FALLBACK_CLAMPED and adapted.method == 'masked_median'
    assert adapted.resolved_decim(fs) == 1


def test_genuine_fractional_rate_still_changes_median_window():
    y = trace()
    normal = B.characterisation_baseline(y, FS, polarity=-1, health=False, cache=False).baseline
    actual = B.characterisation_baseline(y, 1999.99, polarity=-1, health=False, cache=False).baseline
    assert np.max(np.abs(normal - actual)) > 0.01


@pytest.mark.parametrize('method', B.METHODS)
def test_algorithm_revision_versions_windowed_estimators_and_refuses_exact_legacy_replay(method, monkeypatch):
    cfg = B.BaselineConfig(method=method).resolve(FS)
    current = cfg.to_provenance(FS)
    with monkeypatch.context() as old_code:
        old_code.setattr(B, 'SAMPLE_COUNT_POLICY', None)
        old = cfg.to_provenance(FS)
    if method == 'constant':
        assert current == old
        assert B.BaselineConfig.reproducible_from(old) == cfg
    else:
        assert current['baseline_sample_count_policy'] == B.SAMPLE_COUNT_POLICY
        assert current['baseline_fingerprint'] != old['baseline_fingerprint']
        assert B.BaselineConfig.from_provenance(old) == cfg
        with pytest.raises(B.BaselinePolicyChangedError, match='Re-run detection'):
            B.BaselineConfig.reproducible_from(old)
        with pytest.raises(B.BaselinePolicyChangedError, match='unsupported'):
            B.BaselineConfig.from_provenance(current | {'baseline_sample_count_policy': 'future-v9'})
    assert B.BaselineConfig.reproducible_from(current) == cfg


def legacy_median(y, fs, window, decim):
    w = int(window * fs / decim); w += w % 2 == 0
    b = median_filter(np.ascontiguousarray(y[::decim], dtype=float)[None, :], size=(1, w), mode='reflect')[0]
    return np.interp(np.arange(len(y)), np.arange(0, len(y), decim)[:len(b)], b).astype(np.float32)


@pytest.mark.parametrize('method', ['robust_median', 'masked_median', 'savgol'])
def test_fixed_model_input_contract_keeps_legacy_arithmetic_and_stamps_it(method):
    y = trace(); fs = INFERRED
    cfg = ML.MLPreprocessConfig(notch_enabled=False, baseline_method=method, baseline_decim=2,
                               baseline_window_s=.5, normalize=False, despike_size=0)
    if method == 'robust_median':
        expected = y - legacy_median(y, fs, .5, 2)
    elif method == 'masked_median':
        rough = y - legacy_median(y, fs, .5, 2)
        sigma = 1.4826 * np.median(np.abs(rough - np.median(rough)))
        mask = binary_dilation(rough < np.median(rough) - 3.5 * sigma, iterations=max(1, int(.001 * fs)))
        filled = y.copy(); idx = np.arange(len(y))
        filled[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
        expected = y - legacy_median(filled, fs, .1, 2)
    else:
        w = int(.5 * fs); w += w % 2 == 0
        expected = y - savgol_filter(y, w, 3, mode='interp')
    np.testing.assert_array_equal(ML.apply_ml_preprocess(y, fs, cfg), expected.astype(np.float32))
    stored = cfg.to_dict()
    assert stored['baseline_sample_count_policy'] == ML.ML_SAMPLE_COUNT_POLICY
    if method == 'masked_median':
        assert stored['baseline_mask_noise_policy'] == ML.ML_MASK_NOISE_POLICY
        with pytest.raises(ValueError, match='fixed notebook'):
            ML.MLPreprocessConfig.from_dict(stored | {'baseline_mask_noise_policy': 'future-v9'})
    assert ML.MLPreprocessConfig.from_dict(stored) == cfg
    del stored['baseline_sample_count_policy']
    assert ML.MLPreprocessConfig.from_dict(stored) == cfg
    with pytest.raises(ValueError, match='fixed notebook'):
        ML.MLPreprocessConfig.from_dict(stored | {'baseline_sample_count_policy': 'future-v9'})


def test_processed_export_stamps_actual_sample_policy_without_accepting_editable_claim():
    from nanoscope.analysis.conditioning import processed_trace_provenance, conditioning_params
    params = {'detrend_enabled': True, 'detrend_method': 'masked_median',
              'detrend_sample_count_policy': 'fabricated'}
    exported = processed_trace_provenance(params=params, fs=FS, n_samples=1000, start_idx=0, units='pA')
    assert exported['detrend_sample_count_policy'] == B.SAMPLE_COUNT_POLICY
    assert 'detrend_sample_count_policy' not in conditioning_params(params)
    assert 'detrend_sample_count_policy' not in exported['params']
    assert params['detrend_sample_count_policy'] == 'fabricated'


def test_algorithm_revision_invalidates_whole_recording_baseline_cache(monkeypatch):
    y = trace(); cfg = B.BaselineConfig(); calls = []
    original = B.estimate_baseline_result
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(B, 'estimate_baseline_result', counted)
    B.clear_baseline_cache()
    B.characterisation_baseline(y, FS, config=cfg, polarity=-1, health=False)
    B.characterisation_baseline(y, FS, config=cfg, polarity=-1, health=False)
    assert len(calls) == 1
    monkeypatch.setattr(B, 'SAMPLE_COUNT_POLICY', 'future-test-policy')
    B.characterisation_baseline(y, FS, config=cfg, polarity=-1, health=False)
    assert len(calls) == 2
    B.clear_baseline_cache()


@pytest.mark.parametrize('method', ['masked_median', 'robust_median', 'savgol', 'block_median'])
def test_runtime_policy_is_generated_and_old_exact_replay_refuses_before_cache_lookup(method):
    from nanoscope.analysis.modules import stamp_preprocess_provenance, detection_signal_key, detection_signal
    params = {'detrend_enabled': True, 'detrend_method': method}
    provenance = {'detrend_sample_count_policy': 'fabricated'}
    stamp_preprocess_provenance(provenance, params, FS)
    assert provenance['detrend_sample_count_policy'] == B.SAMPLE_COUNT_POLICY
    assert detection_signal_key(provenance)
    old = dict(provenance); del old['detrend_sample_count_policy']
    with pytest.raises(B.BaselinePolicyChangedError, match='Re-run detection'):
        detection_signal_key(old)
    with pytest.raises(B.BaselinePolicyChangedError, match='Re-run detection'):
        detection_signal(trace(), FS, old)
    stamp_preprocess_provenance(provenance, {'detrend_enabled': False}, FS)
    assert 'detrend_sample_count_policy' not in provenance
