"""Scientific regressions for dose-domain uncertainty and response-origin invariance."""
import json
import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import norm, t
from nanoscope.analysis.curve_fitting import CurveFitResult, fit_curve


@pytest.mark.parametrize('model', ['logistic4', 'logistic5', 'hill'])
@pytest.mark.parametrize('weighted', [False, True])
def test_wide_dose_intervals_match_analytic_delta_method_and_reload(model, weighted):
    x = np.geomspace(1e-8, 1e3, 400)
    midpoint = 'midpoint_scale' if model == 'logistic5' else 'ec50'
    asymmetry = 1.3 if model == 'logistic5' else 1.
    sigma = .001 + np.linspace(0, .002, len(x))
    y = 2 + 10*expit(1.7*np.log(x/2e-5))**asymmetry + sigma*np.sin(np.arange(len(x)))
    result = fit_curve(x, y, {'model': model, 'initial': {midpoint: 2e-5},
                             'weighting': 'sigma' if weighted else 'uniform'},
                       y_sigma=sigma if weighted else None)
    assert result.success
    params = result.parameters
    names = list(params)
    asymmetry = params.get('asymmetry', 1.)

    def analytic_jacobian(coords):
        fraction = expit(params['hill']*np.log(coords/params[midpoint]))
        response = fraction**asymmetry
        common = (params['top']-params['bottom'])*asymmetry*response*(1-fraction)
        columns = {'bottom': 1-response, 'top': response,
                   midpoint: -params['hill']*common/params[midpoint],
                   'hill': common*np.log(coords/params[midpoint]),
                   'asymmetry': (params['top']-params['bottom'])*response*np.log(fraction)}
        return np.column_stack([columns[name] for name in names])

    jac = analytic_jacobian(x)/(sigma[:, None] if weighted else 1.)
    _, singular, vectors = np.linalg.svd(jac, full_matrices=False)
    noise = np.dot(result.residuals, result.residuals)/(len(x)-len(names))
    covariance = (vectors.T/singular**2)@vectors*(1. if weighted else noise)
    np.testing.assert_allclose([result.standard_errors[name] for name in names],
                               np.sqrt(np.diag(covariance)), rtol=1e-5)
    gradient = analytic_jacobian(result.x_curve)
    variance = np.einsum('ij,jk,ik->i', gradient, covariance, gradient)
    quantile = norm.ppf(.975) if weighted else t.ppf(.975, len(x)-len(names))
    expected_confidence = quantile*np.sqrt(variance)
    observation_variance = np.interp(result.x_curve, x, sigma)**2 if weighted else noise
    expected_prediction = quantile*np.sqrt(variance+observation_variance)
    np.testing.assert_allclose(result.confidence_upper-result.y_curve, expected_confidence, rtol=1e-5)
    np.testing.assert_allclose(result.prediction_upper-result.y_curve, expected_prediction, rtol=1e-5)
    restored = CurveFitResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(restored.confidence_lower, result.confidence_lower)
    np.testing.assert_array_equal(restored.prediction_upper, result.prediction_upper)


@pytest.mark.parametrize('baseline', [0., 1e9, 1e12, -1e12])
def test_response_offset_keeps_resolved_r_squared(baseline):
    x = np.linspace(-5, 5, 120)
    y = baseline+2*x+.1*np.sin(x)
    result = fit_curve(x, y, {'model': 'linear'})
    centered = (y-y[0])-(y-y[0]).mean()
    expected = 1-np.dot(result.residuals, result.residuals)/np.dot(centered, centered)
    assert result.metrics['r_squared'] == pytest.approx(expected, abs=1e-12)
    assert result.metrics['r_squared'] == pytest.approx(.9998517424, abs=3e-6)
    assert not any('constant response' in warning for warning in result.warnings)


@pytest.mark.parametrize('baseline', [0., 1e9, -1e12])
def test_truly_constant_response_has_no_r_squared(baseline):
    x = np.linspace(-5, 5, 120)
    result = fit_curve(x, np.full(len(x), baseline), {'model': 'linear'})
    assert result.metrics['r_squared'] is None
    assert any('constant response' in warning for warning in result.warnings)


@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_r_squared_keeps_resolved_response_units(scale):
    x = np.linspace(-5, 5, 120)
    result = fit_curve(x, scale*(2*x+.1*np.sin(x)), {'model': 'linear'})
    assert result.metrics['r_squared'] == pytest.approx(.9998517424, abs=1e-10)


def test_unresolvable_rounding_jitter_does_not_claim_response_variance():
    x = np.linspace(-5, 5, 120)
    y = np.full(len(x), 1e9)
    y[::2] = np.nextafter(1e9, np.inf)
    result = fit_curve(x, y, {'model': 'linear', 'fixed': {'intercept': 1e9, 'slope': 0.}})
    assert result.metrics['r_squared'] is None


def test_undefined_custom_sensitivity_withholds_intervals_but_keeps_valid_fit():
    x = np.linspace(0, 10, 100)
    result = fit_curve(x, np.sqrt(x), {'model': 'custom', 'custom_expression': 'sqrt(x+a)',
                                     'custom_parameters': ['a'], 'initial': {'a': 0.}})
    assert result.success and np.all(np.isfinite(result.y_curve))
    assert result.confidence_lower is None and result.prediction_lower is None
    assert result.standard_errors['a'] is None and result.parameter_ci['a'] is None
    assert result.metrics['interval_method'] == 'unavailable'
    assert any('uncertainty' in warning and 'omitted' in warning for warning in result.warnings)
    restored = CurveFitResult.from_dict(result.to_dict())
    assert restored.parameters == result.parameters


def test_overflowing_covariance_is_explicitly_withheld_and_remains_reloadable():
    x = np.linspace(1, 2, 30)
    result = fit_curve(x, 1e-170*x, {'model': 'custom', 'custom_expression': 'a*x*1e-170',
                                  'custom_parameters': ['a'], 'initial': {'a': 1.},
                                  'weighting': 'sigma'}, y_sigma=np.full(len(x), 1e-10))
    assert result.success and np.all(np.isfinite(result.y_fit))
    assert result.standard_errors['a'] is None
    assert result.confidence_lower is None and result.prediction_upper is None
    assert result.metrics['interval_method'] == 'unavailable'
    assert any('non-finite' in warning and 'omitted' in warning for warning in result.warnings)
    assert CurveFitResult.from_dict(result.to_dict()).parameters == result.parameters


@pytest.mark.parametrize('heterogeneous', [False, True])
def test_repeated_x_noise_policy_is_row_order_invariant(heterogeneous):
    x = np.repeat(np.linspace(0, 10, 30), 2)
    sigma = np.tile([.01, 1. if heterogeneous else .01], 30)
    y = 2+3*x+sigma*.1*np.sin(np.arange(len(x)))
    permutation = np.arange(len(x)).reshape(-1, 2)[:, ::-1].ravel()
    results = [fit_curve(x[order], y[order], {'model': 'linear', 'weighting': 'sigma'},
                         y_sigma=sigma[order])
               for order in (np.arange(len(x)), permutation)]
    design = np.column_stack((np.ones(len(x)), x))/sigma[:, None]
    covariance = np.linalg.inv(design.T@design)
    for result in results:
        assert result.success
        np.testing.assert_allclose(list(result.standard_errors.values()),
                                   np.sqrt(np.diag(covariance)), rtol=1e-6)
        assert result.confidence_lower is not None
        if heterogeneous:
            assert result.prediction_lower is None and result.prediction_upper is None
            assert any('future-observation noise policy' in warning for warning in result.warnings)
        else:
            confidence_variance = ((result.confidence_upper-result.y_curve)/norm.ppf(.975))**2
            prediction_variance = ((result.prediction_upper-result.y_curve)/norm.ppf(.975))**2
            np.testing.assert_allclose(prediction_variance-confidence_variance, .01**2, rtol=1e-10)
        restored = CurveFitResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
        assert (restored.prediction_lower is None) == heterogeneous
    np.testing.assert_allclose(results[0].confidence_upper, results[1].confidence_upper, rtol=1e-9)
    if not heterogeneous:
        np.testing.assert_allclose(results[0].prediction_upper, results[1].prediction_upper, rtol=1e-9)


@pytest.mark.parametrize('frequency', [.2, 1e3+.2, 1e6+.2])
def test_periodic_custom_sensitivity_matches_analytic_covariance(frequency):
    x = np.sort(np.random.default_rng(841).uniform(.2, 3, 120))
    y = np.sin(frequency*x)+.001*np.sin(np.arange(len(x)))
    result = fit_curve(x, y, {'model': 'custom', 'custom_expression': 'sin(a*x)',
                             'custom_parameters': ['a'], 'initial': {'a': frequency},
                             'weighting': 'sigma'}, y_sigma=np.full(len(x), .01))
    assert result.success
    sensitivity = x*np.cos(result.parameters['a']*x)/.01
    analytic_se = 1/np.sqrt(sensitivity@sensitivity)
    assert result.standard_errors['a'] == pytest.approx(analytic_se, rel=1e-5)
    gradient = result.x_curve*np.cos(result.parameters['a']*result.x_curve)
    expected_halfwidth = norm.ppf(.975)*np.abs(gradient)*analytic_se
    np.testing.assert_allclose(result.confidence_upper-result.y_curve, expected_halfwidth,
                               rtol=2e-5, atol=1e-9)


@pytest.mark.parametrize('origin', [1e14, 1e15])
def test_unresolvable_coordinate_origin_withholds_misleading_uncertainty(origin):
    x = origin+np.linspace(-6, 6, 200)
    y = 2+8*np.exp(-.5*((x-(origin+.5))/1.1)**2)+.02*np.sin(np.arange(len(x)))
    result = fit_curve(x, y, {'model': 'gaussian', 'weighting': 'sigma',
                             'initial': {'baseline': 2, 'amplitude': 8, 'center': origin+.5, 'sigma': 1.1}},
                       y_sigma=np.full(len(x), .02))
    assert result.success and result.metrics['r_squared'] > .999
    assert result.standard_errors['center'] is None and result.parameter_ci['center'] is None
    assert result.confidence_lower is None and result.prediction_lower is None
    assert any('Floating-point resolution' in warning for warning in result.warnings)
    assert CurveFitResult.from_dict(result.to_dict()).parameters == result.parameters
    # Recentring the same represented observations restores independently checked uncertainty.
    centered = fit_curve(x-origin, y, {'model': 'gaussian', 'weighting': 'sigma'},
                         y_sigma=np.full(len(x), .02))
    p = centered.parameters
    z = ((x-origin)-p['center'])/p['sigma']
    peak = np.exp(-.5*z*z)
    design = np.column_stack((np.ones(len(x)), peak,
                              p['amplitude']*peak*z/p['sigma'],
                              p['amplitude']*peak*z*z/p['sigma']))/.02
    analytic_errors = np.sqrt(np.diag(np.linalg.inv(design.T@design)))
    np.testing.assert_allclose(list(centered.standard_errors.values()), analytic_errors, rtol=1e-5)


@pytest.mark.parametrize('section,key,value', [
    ('metrics', 'r_squared', 'not-a-number'), ('metrics', 'r2', 'not-a-number'), ('metrics', 'rmse', True),
    ('metrics', 'n', '60'), ('metrics', 'dof', 2.5), ('metrics', 'nfev', -1),
    ('metrics', 'jacobian_rank', 100), ('metrics', 'objective', 3),
    ('metrics', 'n_input', 1), ('metrics', 'n_excluded_range', 100),
    ('config', 'confidence', '.95'), ('config', 'confidence', 1.),
    ('config', 'confidence', True), ('config', 'loss', 'bogus'),
    ('config', 'weighting', []), ('config', 'x_min', '0'),
    ('config', 'max_nfev', 1.5), ('config', 'max_nfev', True),
    ('config', 'f_scale', -1), ('config', 'initial', {'slope': '1'}),
    ('config', 'fixed', {'slope': False}), ('config', 'bounds', {'slope': [2, 1]}),
    ('config', 'bounds', {'slope': ['0', None]}),
])
def test_saved_known_metrics_and_config_require_correct_types(section, key, value):
    x = np.linspace(0, 10, 60)
    payload = fit_curve(x, 2+3*x+.01*np.sin(x), {'model': 'linear'}).to_dict()
    payload[section][key] = value
    with pytest.raises(ValueError):
        CurveFitResult.from_dict(payload)


@pytest.mark.parametrize('options', [
    {'x_min': '1', 'x_max': '9'}, {'f_scale': '.1', 'loss': 'soft_l1'},
    {'max_nfev': 12.5}, {'max_nfev': -1}, {'max_nfev': '100000'},
    {'initial': {'intercept': '2'}, 'fixed': {'slope': '3'}},
    {'initial': None, 'fixed': None, 'bounds': None},
])
def test_executed_fit_options_are_canonical_and_self_reloadable(options):
    x = np.linspace(0, 10, 60)
    result = fit_curve(x, 2+3*x+.01*np.sin(x), {'model': 'linear', **options})
    restored = CurveFitResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    assert restored.parameters == result.parameters
    for group in ('initial', 'fixed', 'bounds'):
        assert isinstance(restored.config[group], dict)
    for key in ('x_min', 'x_max', 'f_scale'):
        if key in options:
            assert restored.config[key] == float(options[key])
    if 'max_nfev' in options:
        assert restored.config['max_nfev'] == min(max(int(options['max_nfev']), 10), 50000)


def test_saved_null_optional_parameter_groups_normalize_to_mappings():
    x = np.linspace(0, 10, 60)
    payload = fit_curve(x, 2+3*x+.01*np.sin(x), {'model': 'linear'}).to_dict()
    for key in ('initial', 'fixed', 'bounds'):
        payload['config'][key] = None
    restored = CurveFitResult.from_dict(payload)
    assert all(restored.config[key] == {} for key in ('initial', 'fixed', 'bounds'))
