"""Ground-truth tests for masking events near strongly nonstationary noise."""
import warnings
import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from nanoscope.analysis import baseline as B
from nanoscope.analysis import ml_preprocess as ML
from nanoscope.analysis.baseline_mask import baseline_mask_noise_scale, fill_masked_background

FS = 2000.
CFG = B.BaselineConfig(rough_s=2., base_s=.1)
EVENTS = ((2., 2.04), (5.17, 5.21), (5.46, 5.5), (8., 8.04))


def scene(seed, *, events=False, polarity=-1, drift=False):
    t = np.arange(24000) / FS
    burst = (t >= 5.03) & (t < 5.43)
    noise = np.random.default_rng(seed).normal(size=len(t)) * np.where(burst, 8., 1.)
    noise -= noise.mean(); noise *= 2. / noise.std()
    baseline = np.full(t.size, 200.)
    if drift:
        baseline += 3. * np.sin(2 * np.pi * t / 12.)
        baseline[t >= 7.] += 8.
    raw = baseline + noise
    if events:
        for start, end in EVENTS:
            raw[int(start * FS):int(end * FS)] -= 40.
    return -polarity * raw, -polarity * baseline, burst


def twice(x, polarity):
    with warnings.catch_warnings(), B.collect_baseline_warnings():
        warnings.simplefilter('ignore')
        corrected = B.baseline_wander_result(x, FS, method='masked_median', polarity=polarity,
                                             rough_s=2., base_s=.1).corrected
        ref = B.characterisation_baseline(corrected, FS, config=CFG, polarity=polarity,
                                           health=False, cache=False).baseline
    return corrected - ref


@pytest.mark.parametrize('seed', [52591, 64721, 72103, 91273])
@pytest.mark.parametrize('polarity', [-1, 1])
def test_noise_only_burst_does_not_create_a_one_sided_baseline_offset(seed, polarity):
    raw, baseline, burst = scene(seed, polarity=polarity)
    before = raw.copy()
    corrected = twice(raw, polarity)
    local_sigma = np.std((raw - baseline)[burst])
    assert abs(np.mean(corrected[burst])) < .2 * local_sigma
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize('seed', [52591, 64721, 72103, 91273])
@pytest.mark.parametrize('polarity', [-1, 1])
def test_nearby_events_cannot_inflate_mask_noise_and_become_baseline_anchors(seed, polarity):
    raw, baseline, _ = scene(seed, events=True, polarity=polarity)
    result = twice(raw, polarity)
    for start, end in EVENTS:
        a, b = int(start * FS), int(end * FS)
        oracle = polarity * np.median((raw - baseline)[a:b])
        measured = polarity * np.median(result[a:b])
        assert abs(measured - oracle) <= max(4., .1 * oracle), (seed, start, measured, oracle)


def test_nonstationary_health_mask_matches_actual_kernel(monkeypatch):
    raw, _, _ = scene(52591, events=True)
    expected_mask = B._rough_pass(raw, FS, config=CFG)[1]
    seen = []
    original = ML.fill_masked_background
    def record_mask(signal, mask, *args, **kwargs):
        seen.append((mask.copy(), kwargs.get('robust_flanks')))
        return original(signal, mask, *args, **kwargs)
    monkeypatch.setattr(ML, 'fill_masked_background', record_mask)
    ML.baseline_masked_median(raw, FS, **CFG.kernel_kwargs(FS))
    assert len(seen) == 1 and seen[0][1] is True
    np.testing.assert_array_equal(seen[0][0], expected_mask)


def test_stationary_mask_keeps_global_threshold_and_robust_fill_preserves_depth():
    raw = np.random.default_rng(81917).normal(200, 2, 24000)
    raw[7000:7080] -= 40
    rough = raw - B.decimated_median(raw, FS, CFG.rough_s, CFG.resolved_decim(FS))
    sigma = 1.4826 * np.median(abs(rough - np.median(rough)))
    assert np.isscalar(baseline_mask_noise_scale(rough, FS, sigma))
    expected = binary_dilation(rough < np.median(rough) - CFG.mask_sigma * sigma, iterations=2)
    np.testing.assert_array_equal(B._rough_pass(raw, FS, config=CFG)[1], expected)
    before = raw.copy()
    _, new = ML.baseline_masked_median(raw, FS, **CFG.kernel_kwargs(FS))
    oracle = 200. - np.median(raw[7000:7080])
    measured = np.median(new[7000:7080] - raw[7000:7080])
    assert abs(measured - oracle) < 2.
    np.testing.assert_array_equal(raw, before)


def test_mask_policy_versions_only_automatic_masked_estimator_and_refuses_old_exact_replay(monkeypatch):
    cfg = CFG.resolve(FS); current = cfg.to_provenance(FS)
    with monkeypatch.context() as historic:
        historic.setattr(B, 'MASK_NOISE_POLICY', None)
        old = cfg.to_provenance(FS)
    assert old['baseline_fingerprint'] != current['baseline_fingerprint']
    assert current['baseline_mask_noise_policy'] == B.MASK_NOISE_POLICY
    assert B.BaselineConfig.from_provenance(old) == cfg
    with pytest.raises(B.BaselinePolicyChangedError, match='Re-run detection'):
        B.BaselineConfig.reproducible_from(old)
    assert B.BaselineConfig.reproducible_from(current) == cfg


@pytest.mark.parametrize('fs', [1999.9999999999982, np.nextafter(FS, 0.), np.nextafter(FS, np.inf)])
def test_adaptive_mask_remains_identical_across_roundoff_rate_changes(fs):
    raw, _, _ = scene(64721, events=True)
    expected = B.characterisation_baseline(raw, FS, config=CFG, polarity=-1, health=False, cache=False)
    actual = B.characterisation_baseline(raw, fs, config=CFG, polarity=-1, health=False, cache=False)
    np.testing.assert_array_equal(actual.baseline, expected.baseline)


@pytest.mark.parametrize('polarity', [-1, 1])
def test_adaptive_flanks_ignore_noisy_endpoints_without_changing_observations(polarity):
    truth = np.full(2000, 200.)
    raw = truth.copy(); mask = np.zeros(raw.size, dtype=bool); mask[900:980] = True
    raw[mask] -= 40; raw[899] -= 18; raw[980] += 22
    raw *= -polarity; truth *= -polarity
    actual = fill_masked_background(raw, mask, FS, .1, robust_flanks=True)
    np.testing.assert_array_equal(actual[mask], truth[mask])
    np.testing.assert_array_equal(actual[~mask], raw[~mask])
    legacy = fill_masked_background(raw, mask, FS, .1, robust_flanks=False)
    assert np.max(np.abs(legacy[mask] - truth[mask])) > 10


@pytest.mark.parametrize('polarity', [-1, 1])
def test_adaptive_flank_anchors_reproduce_linear_drift_across_neighboring_events(polarity):
    truth = -polarity * (200. + np.arange(2000) * .01)
    mask = np.zeros(truth.size, dtype=bool)
    mask[500:700] = True; mask[750:950] = True
    raw = truth.copy(); raw[mask] += polarity * 40.
    actual = fill_masked_background(raw, mask, FS, .1, robust_flanks=True)
    np.testing.assert_allclose(actual[mask], truth[mask], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual[~mask], raw[~mask])


@pytest.mark.parametrize('step_at', [880, 1000])
@pytest.mark.parametrize('step_height', [-8., 8.])
def test_adaptive_flanks_stop_at_resolved_baseline_steps(step_at, step_height):
    truth = np.full(2000, 200.); truth[step_at:] += step_height
    mask = np.zeros(truth.size, dtype=bool); mask[900:980] = True
    raw = truth.copy(); raw[mask] -= 40
    actual = fill_masked_background(raw, mask, FS, .1, robust_flanks=True)
    np.testing.assert_array_equal(actual[mask], truth[mask])
    np.testing.assert_array_equal(actual[~mask], raw[~mask])


@pytest.mark.parametrize('start,stop', [(0, 80), (1920, 2000), (1, 1999)])
def test_missing_flank_support_retains_defined_legacy_boundary_behavior(start, stop):
    raw = np.linspace(190., 210., 2000); mask = np.zeros(raw.size, dtype=bool); mask[start:stop] = True
    expected = fill_masked_background(raw, mask, FS, .1, robust_flanks=False)
    actual = fill_masked_background(raw, mask, FS, .1, robust_flanks=True)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('polarity', [-1, 1])
def test_stationary_single_sample_anchor_regression_preserves_raw_oracle_depth(polarity):
    noise = np.random.default_rng(52591).normal(size=24000)
    noise -= noise.mean(); noise *= 2. / noise.std()
    raw = 200. + noise
    for start, dwell in [(1., .01), (2., .05), (4.5, .4), (5.12, .04), (7.5, .01), (9., .4)]:
        raw[int(round(start * FS)):int(round((start + dwell) * FS))] -= 32.
    raw *= -polarity; before = raw.copy()
    result = twice(raw, polarity)
    oracle = polarity * np.median(raw[4000:4100] + polarity * 200.)
    measured = polarity * np.median(result[4000:4100])
    assert abs(measured - oracle) < .1 * oracle
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize('polarity', [-1, 1])
@pytest.mark.parametrize('seed', [105319, 710371, 21073])
def test_dense_event_trains_cannot_be_reinterpreted_as_increased_noise(seed, polarity):
    """A local two-level distribution is not evidence of increased random noise."""
    raw = np.random.default_rng(seed).normal(200., 2., 24000)
    spans = [(round(t * FS), round((t + .04) * FS))
             for t in np.arange(4.8, 6.4, .08)]
    for a, b in spans:
        raw[a:b] -= 24.
    raw *= -polarity
    result = twice(raw, polarity)
    for a, b in spans:
        oracle = polarity * np.median(raw[a:b] + polarity * 200.)
        measured = polarity * np.median(result[a:b])
        assert abs(measured - oracle) < .1 * oracle, (seed, a, measured, oracle)


@pytest.mark.corpus
def test_real_ferritin_dense_state_retains_depth_against_independent_flanks():
    """The 168.46 s state has clear open-pore observations on both sides.

    These fixed spans were inspected in the raw trace; neither a current mask nor
    a stored baseline defines the reference. Local level-MAD contamination once
    raised sigma from 17 to 81 pA and removed most of this ~300 pA event.
    """
    from pathlib import Path
    import pandas as pd
    path = Path('/Users/abrahamcampos/Desktop/Research-Data/FerritinModeler/CSVData/ferritin_data2.csv.gz')
    if not path.exists():
        pytest.skip('the ferritin corpus is not on this machine')
    raw = np.asarray(pd.read_csv(path, usecols=['current_raw_pA'])['current_raw_pA'], np.float32)
    filtered = ML.notch_sequential(raw, 50000.)
    del raw
    left = float(np.median(filtered[8422500:8423000]))
    right = float(np.median(filtered[8425400:8425900]))
    state = slice(8423200, 8425200)
    reference = .5 * (left + right)
    oracle = reference - float(np.median(filtered[state]))
    assert oracle > 200. and abs(left - right) < 50., (left, right, oracle)
    res = B.characterisation_baseline(filtered, 50000.,
        config=B.BaselineConfig(rough_s=.5, base_s=.1, decim=50, despike_size=3),
        polarity=-1, health=False, cache=False)
    measured = float(np.median(res.baseline[state] - filtered[state]))
    assert abs(measured - oracle) < .1 * oracle, (measured, oracle)


@pytest.mark.parametrize('polarity', [-1, 1])
@pytest.mark.parametrize('shape', ['linear', 'quadratic'])
def test_large_physical_drift_is_not_counted_as_random_mask_noise(shape, polarity):
    """High-SNR events stay measurable even as the physical baseline accelerates."""
    t = np.arange(24000) / FS
    truth = 200. + (120. * t if shape == 'linear' else 20. * t * t)
    raw = truth + np.random.default_rng(814903).normal(0., 2., t.size)
    spans = [(round(z * FS), round((z + .08) * FS)) for z in [2., 5., 8., 10.]]
    for a, b in spans:
        raw[a:b] -= 100.
    raw *= -polarity; truth *= -polarity
    before = raw.copy()
    result = B.characterisation_baseline(raw, FS, config=CFG, polarity=polarity,
                                         health=False, cache=False)
    for a, b in spans:
        oracle = polarity * np.median((raw - truth)[a:b])
        measured = polarity * np.median((raw - result.baseline)[a:b])
        assert abs(measured - oracle) < .1 * oracle, (shape, a, measured, oracle)
    np.testing.assert_array_equal(raw, before)
