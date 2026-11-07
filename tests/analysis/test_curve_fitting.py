"""Independent truth, uncertainty and failure checks for nonlinear regression."""
import json

import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.special import expit

from nanoscope.analysis.curve_fitting import (
    CurveFitResult, FitCancelled, compile_custom_expression, evaluate_curve,
    fit_curve, get_model, list_models,
)

TRUTH = {
    'linear': dict(intercept=2,slope=-3),
    'gaussian': dict(baseline=2, amplitude=10, center=.7, sigma=1.2),
    'exp1': dict(baseline=2, amplitude=8, rate=-.8),
    'exp2': dict(baseline=2, amplitude1=8, rate1=-.6, amplitude2=4, rate2=-2.3),
    'power_law': dict(baseline=2, amplitude=3, exponent=1.3),
    'logistic4': dict(bottom=2, top=12, ec50=3, hill=1.7),
    'logistic5': dict(bottom=2, top=12, midpoint_scale=3, hill=1.7, asymmetry=1.5),
    'hill': dict(bottom=2, top=12, ec50=3, hill=1.7),
    'boltzmann': dict(bottom=2, top=12, v_half=-.2, slope=.4),
    'stretched_exp': dict(baseline=2, amplitude=8, tau=2, beta=.7),
    'lorentzian': dict(baseline=2, amplitude=10, center=.7, gamma=1.2),
    'voigt': dict(baseline=2, amplitude=10, center=.7, sigma=.8, gamma=.5),
}

@pytest.mark.parametrize('model', list(TRUTH))
def test_builtin_default_guesses_recover_truth(model):
    x = (np.geomspace(.001,1000,300) if model in ('logistic4','logistic5','hill') else
         np.linspace(.01,10,300) if get_model(model).positive_x or model.startswith('exp') else np.linspace(-6,6,300))
    y = evaluate_curve(model,x,TRUTH[model])
    r = fit_curve(x,y,{'model':model})
    assert r.success
    assert r.metrics['sse'] < 1e-12
    for p,v in TRUTH[model].items():
        assert r.parameters[p] == pytest.approx(v, rel=1e-6, abs=1e-6)


def test_weighted_gaussian_matches_independent_scipy_and_covariance():
    rng = np.random.default_rng(71)
    x = np.linspace(-5,5,130)
    sigma = .05+.02*(x+5)
    def independent(x,b,a,c,s): return b+a*np.exp(-.5*((x-c)/s)**2)
    y = independent(x,2,8,.4,1.1)+rng.normal(0,sigma)
    expected,cov = curve_fit(independent,x,y,p0=[2,8,.4,1.1],sigma=sigma,absolute_sigma=True)
    r = fit_curve(x,y,{'model':'gaussian','weighting':'sigma'},y_sigma=sigma)
    actual = [r.parameters[k] for k in ('baseline','amplitude','center','sigma')]
    actual_se = [r.standard_errors[k] for k in ('baseline','amplitude','center','sigma')]
    np.testing.assert_allclose(actual,expected,rtol=1e-6)
    np.testing.assert_allclose(actual_se,np.sqrt(np.diag(cov)),rtol=2e-5)
    assert np.all(r.prediction_upper-r.prediction_lower >= r.confidence_upper-r.confidence_lower)
    assert r.metrics['objective'] == 'original_y'


def test_logistic_is_independent_formula_and_scale_is_not_5pl_ec50():
    x=np.geomspace(.001,1e3,300)
    y=2+10*x**1.7/(3**1.7+x**1.7)
    r=fit_curve(x,y,{'model':'logistic4'})
    assert r.parameters['ec50']==pytest.approx(3)
    y5=evaluate_curve('logistic5',[3],TRUTH['logistic5'])[0]
    assert y5 != pytest.approx(7)
    assert 'NOT EC50' in get_model('logistic5').description
    assert evaluate_curve('logistic4',[0],TRUTH['logistic4'])[0]==2


def test_signed_voltage_boltzmann_and_three_point_refusal():
    x=np.linspace(-.9,.2,100)
    y=2+10*expit((x+.3)/.07)
    r=fit_curve(x,y,{'model':'boltzmann'})
    assert r.parameters['v_half']==pytest.approx(-.3)
    with pytest.raises(ValueError,match='free parameters'):
        fit_curve([.1,.2,.3],[2,3,4],{'model':'logistic4'})
    with pytest.raises(ValueError,match='non-negative'):
        fit_curve(x,y,{'model':'logistic4'})


def test_fixed_parameters_and_bounds_reduce_degrees_of_freedom():
    x=np.array([.1,.5,1.])
    y=2+8*np.exp(-.8*x)
    r=fit_curve(x,y,{'model':'exp1','fixed':{'baseline':2},'bounds':{'rate':[-2,-.01]}})
    assert r.metrics['n_parameters']==2
    assert r.parameters['baseline']==2
    assert r.parameter_ci['baseline']==(2,2)
    assert r.parameters['rate']==pytest.approx(-.8)
    with pytest.raises(ValueError,match='outside'):
        fit_curve(x,y,{'model':'exp1','initial':{'rate':2},'bounds':{'rate':[-2,-.01]}})


def test_custom_equation_and_safe_json_roundtrip():
    x=np.linspace(0,8,90);y=4+3*x+.02*np.cos(x)
    cfg={'model':'custom','custom_expression':'a*x+b','custom_parameters':['a','b']}
    r=fit_curve(x,y,cfg)
    assert r.parameters['a']==pytest.approx(3,rel=.001)
    payload=json.loads(json.dumps(r.to_dict(),allow_nan=False))
    restored=CurveFitResult.from_dict(payload)
    np.testing.assert_array_equal(r.residuals,restored.residuals)
    np.testing.assert_array_equal(r.confidence_lower,restored.confidence_lower)
    assert restored.parameter_ci==r.parameter_ci
    assert restored.config['bounds']=={'a':[None,None],'b':[None,None]}
    r2=fit_curve(x,y,restored.config)
    assert r2.parameters==pytest.approx(r.parameters)

@pytest.mark.parametrize('expression', [
    '__import__("os").system("whoami")','x.__class__','x[0]+a','[a]*1000000',
    '(lambda z: z)(x)','open("file")','np.exp(a*x)','sum(x)+a',
])
def test_custom_rejects_execution_and_access(expression):
    with pytest.raises(ValueError): compile_custom_expression(expression,['a'])


def test_custom_rejects_unused_and_invalid_parameters():
    with pytest.raises(ValueError,match='Unused'): compile_custom_expression('a*x',['a','b'])
    with pytest.raises(ValueError,match='identifiers'): compile_custom_expression('x+exp',['exp'])
    with pytest.raises(ValueError,match='finite'): compile_custom_expression('a*x+1e999',['a'])


def test_robust_outlier_resistance_and_honest_uncertainty():
    x=np.linspace(-5,5,160); y=2+8*np.exp(-.5*((x-.5)/1.1)**2)
    y[15]+=40
    ordinary=fit_curve(x,y,{'model':'gaussian'})
    robust=fit_curve(x,y,{'model':'gaussian','loss':'soft_l1','f_scale':.1})
    assert abs(robust.parameters['center']-.5)<abs(ordinary.parameters['center']-.5)
    assert robust.confidence_lower is None
    assert robust.parameter_ci['center'] is None
    assert any('Robust loss' in w for w in robust.warnings)


def test_cancellation_and_input_validation():
    x=np.arange(10.);y=x*x
    with pytest.raises(FitCancelled): fit_curve(x,y,{'model':'gaussian'},cancel=lambda:True)
    with pytest.raises(ValueError,match='finite'): fit_curve(x,np.where(x==0,np.nan,y),{'model':'gaussian'})
    with pytest.raises(ValueError,match='distinct'): fit_curve(np.ones(10),y,{'model':'gaussian'})
    with pytest.raises(ValueError,match='positive'): fit_curve(x,y,{'model':'gaussian','weighting':'sigma'},y_sigma=np.zeros(10))
    with pytest.raises(ValueError,match='requires observation'): fit_curve(x,y,{'model':'gaussian','weighting':'sigma'})
    with pytest.raises(ValueError,match='Lower x'): fit_curve(x,y,{'model':'gaussian','x_min':7,'x_max':2})


def test_range_preserved_and_intervals_omitted_for_unidentifiable_model():
    x=np.linspace(0,10,80);y=3*x+2
    r=fit_curve(x,y,{'model':'custom','custom_expression':'a*x+b*x+c',
                   'custom_parameters':['a','b','c'],'x_min':2,'x_max':8})
    assert np.min(r.x)>=2 and np.max(r.x)<=8
    assert any('identifiable' in w for w in r.warnings)
    assert r.confidence_lower is None


@pytest.mark.parametrize('scale',[1e-12,1.,1e12])
def test_fitting_stable_across_response_units(scale):
    x=np.linspace(-5,5,140)
    y=scale*(2+8*np.exp(-.5*((x-.4)/1.1)**2))
    r=fit_curve(x,y,{'model':'gaussian'})
    assert r.parameters['center']==pytest.approx(.4,abs=1e-7)
    assert r.parameters['sigma']==pytest.approx(1.1,abs=1e-7)
    assert r.parameters['amplitude']/scale==pytest.approx(8,rel=1e-7)
    assert r.metrics['r_squared']==pytest.approx(1)


def test_registry_exposes_all_requested_models():
    assert {m.id for m in list_models()}==set(TRUTH)|{'custom'}

@pytest.mark.parametrize('x_start,x_end,rate',[(0,5,.6),(0,5,3),(-1,0,.6),(-1,0,-3)])
def test_exponential_guesses_handle_growth_decay_and_signed_voltage(x_start,x_end,rate):
    x=np.linspace(x_start,x_end,150);y=2+3*np.exp(rate*x)
    r=fit_curve(x,y,{'model':'exp1'})
    assert r.metrics['r_squared']==pytest.approx(1,abs=1e-10)
    assert r.parameters['rate']==pytest.approx(rate,rel=1e-5)
    assert r.parameters['baseline']==pytest.approx(2,rel=1e-4)


def test_voigt_matches_independent_complex_error_function():
    from scipy.special import wofz
    x=np.linspace(-4,5,160);sigma=.8;gamma=.5;center=.7
    profile=wofz(((x-center)+1j*gamma)/(sigma*np.sqrt(2))).real/(sigma*np.sqrt(2*np.pi))
    peak=wofz(1j*gamma/(sigma*np.sqrt(2))).real/(sigma*np.sqrt(2*np.pi))
    independent=2+10*profile/peak
    np.testing.assert_allclose(evaluate_curve('voigt',x,TRUTH['voigt']),independent,rtol=1e-13)
    r=fit_curve(x,independent,{'model':'voigt'})
    assert r.parameters['sigma']==pytest.approx(sigma,rel=1e-6)
    assert r.parameters['gamma']==pytest.approx(gamma,rel=1e-6)


def test_logistic5_matches_independent_power_formula():
    x=np.geomspace(.001,1000,200)
    independent=2+10/(1+(3/x)**1.7)**1.5
    r=fit_curve(x,independent,{'model':'logistic5'})
    for p,v in TRUTH['logistic5'].items(): assert r.parameters[p]==pytest.approx(v,rel=1e-6)


def test_active_bounds_suppress_invalid_symmetric_intervals():
    x=np.linspace(-5,5,150);y=2+8*np.exp(-.5*((x-.4)/1.1)**2)
    r=fit_curve(x,y,{'model':'gaussian','initial':{'sigma':1.3},'bounds':{'sigma':[1.3,3]}})
    assert r.parameters['sigma']==pytest.approx(1.3,rel=1e-6)
    assert r.confidence_lower is None
    assert any('near bounds' in w for w in r.warnings)
    with pytest.raises(ValueError,match='positive'): fit_curve(x,y,{'model':'gaussian','bounds':{'sigma':[-1,3]}})


def test_range_accounting_and_known_sigma_prediction_variance():
    from scipy.stats import norm
    x=np.linspace(0,10,80);sigma=.1+.02*x;y=3*x+2
    r=fit_curve(x,y,{'model':'linear','weighting':'sigma','x_min':2,'x_max':8},y_sigma=sigma)
    assert r.metrics['n_input']==80
    assert r.metrics['n_excluded_range']==80-r.metrics['n']
    q=norm.ppf(.975)
    confidence_var=((r.confidence_upper-r.y_curve)/q)**2
    prediction_var=((r.prediction_upper-r.y_curve)/q)**2
    np.testing.assert_allclose(prediction_var-confidence_var,np.interp(r.x_curve,x,sigma)**2,rtol=1e-12,atol=1e-12)
    assert r.metrics['prediction_noise']=='linearly interpolated supplied sigma'


def test_cancellation_during_optimizer():
    calls=0
    def cancelled():
        nonlocal calls
        calls+=1
        return calls>7
    x=np.linspace(-5,5,150);y=2+8*np.exp(-.5*((x-.4)/1.1)**2)
    with pytest.raises(FitCancelled): fit_curve(x,y,{'model':'gaussian'},cancelled=cancelled)
    assert calls==8

@pytest.fixture
def saved_gaussian():
    x=np.linspace(-5,5,50);y=2+8*np.exp(-.5*((x-.4)/1.1)**2)+.01*np.sin(x)
    return fit_curve(x,y,{'model':'gaussian'}).to_dict()

@pytest.mark.parametrize('field,value',[
    ('x', [[1,2],[3,4]]),('y',[1,2]),('y_fit',[float('inf')]*50),
    ('residuals',[float('nan')]*50),('residuals',[10.]*50),('x_curve',[]),
    ('y_curve',[1,2]),('confidence_lower',[0,1]),('confidence_upper',None),
    ('y_sigma',[0.]*50),('schema_version',2),('schema_version',True),
    ('warnings','warning'),('success','yes'),('message',{}),
    ('standard_errors',{'sigma':-1}),('parameter_ci',{'sigma':[2,1]}),
    ('parameter_ci',{'sigma':[None,1]}),('parameter_ci',{'sigma':[[1],[2]]}),
    ('parameter_ci',{'unknown':[1,2]}),('metrics',{'n':49}),('metrics',{'sse':float('nan')}),
])
def test_restore_rejects_malformed_diagnostics(saved_gaussian,field,value):
    saved_gaussian[field]=value
    with pytest.raises(ValueError): CurveFitResult.from_dict(saved_gaussian)


def test_restore_rejects_unknown_model_and_inconsistent_parameters(saved_gaussian):
    import copy
    for model in ('unknown',None,[]):
        d=copy.deepcopy(saved_gaussian);d['config']['model']=model
        with pytest.raises(ValueError): CurveFitResult.from_dict(d)
    for parameter,value in [('sigma',float('nan')),('center',True)]:
        d=copy.deepcopy(saved_gaussian);d['parameters'][parameter]=value
        with pytest.raises(ValueError): CurveFitResult.from_dict(d)
    saved_gaussian['parameters']['unexpected']=1
    with pytest.raises(ValueError): CurveFitResult.from_dict(saved_gaussian)


def test_restore_rejects_reversed_curve_bands_and_missing_sigma(saved_gaussian):
    import copy
    d=copy.deepcopy(saved_gaussian)
    d['confidence_lower']=[value+1 for value in d['confidence_upper']]
    with pytest.raises(ValueError,match='exceeds'): CurveFitResult.from_dict(d)
    saved_gaussian['config']['weighting']='sigma'
    with pytest.raises(ValueError,match='missing observation'): CurveFitResult.from_dict(saved_gaussian)


def test_restore_limits_size_before_allocating(saved_gaussian):
    class TooLong:
        def __len__(self): return 5_000_001
        def __iter__(self): raise AssertionError('must reject size before conversion')
    saved_gaussian['x']=TooLong()
    with pytest.raises(ValueError,match='point limit'): CurveFitResult.from_dict(saved_gaussian)


def test_restore_accepts_omitted_optional_defaults_without_computation(saved_gaussian,monkeypatch):
    import nanoscope.analysis.curve_fitting as module
    for key in ('standard_errors','parameter_ci','confidence_lower','confidence_upper',
                'prediction_lower','prediction_upper','y_sigma','metrics','warnings',
                'success','message','schema_version'):
        saved_gaussian.pop(key,None)
    def forbidden(*args,**kwargs): raise AssertionError('restore must not compute a fit or curve')
    monkeypatch.setattr(module,'fit_curve',forbidden)
    monkeypatch.setattr(module,'evaluate_curve',forbidden)
    restored=CurveFitResult.from_dict(saved_gaussian)
    assert restored.success and restored.schema_version==1
    assert restored.confidence_lower is None and restored.warnings==[]
    assert all(v is None for v in restored.parameter_ci.values())


def test_restore_validates_custom_ast_without_evaluating(monkeypatch):
    import nanoscope.analysis.curve_fitting as module
    x=np.linspace(0,5,20);y=2+3*x
    d=fit_curve(x,y,{'model':'custom','custom_expression':'a*x+b','custom_parameters':['a','b']}).to_dict()
    def forbidden(*args,**kwargs): raise AssertionError('restore must not evaluate equation')
    monkeypatch.setattr(module,'evaluate_curve',forbidden)
    monkeypatch.setattr(module,'fit_curve',forbidden)
    restored=CurveFitResult.from_dict(d)
    assert restored.parameters['a']==pytest.approx(3)
    d['config']['custom_expression']='__import__("os").system("command")'
    with pytest.raises(ValueError): CurveFitResult.from_dict(d)
