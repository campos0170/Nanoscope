"""Independent numerical contracts for fitting support and saved uncertainty."""
import json
import numpy as np
import pytest
from scipy import stats as scipy_stats
from nanoscope.analysis import stats


def test_full_support_gof_rejects_density_that_hides_mass_outside_data():
    sample=.001+np.random.default_rng(44).exponential(.01,2000)
    # A valid normalized mixture, with 99% mass in a remote component. Conditioning
    # the diagnostic on the observed min/max wrongly endorses this model.
    params={'tau_1':.01,'tau_2':100.,'w_1':.01,'w_2':.99}
    meta={'family':'exp-mixture','t_min':.001,'t_max':None}
    fit=stats.FitResult(params,{},0,0,0,len(sample),'deliberately wrong mixture',meta,
        stats.density_from('exp-mixture',params,meta))
    result=stats.goodness_of_fit(fit,sample)
    assert result['describes_data'] is False
    assert result['ks']>.98
    assert result['grid_mass']<.02
    assert result['calibrated'] is False
    assert 'far beyond chance' not in result['note']


def test_full_support_exponential_ks_matches_independent_analytic_cdf():
    x=.001+np.random.default_rng(7).exponential(.01,1200)
    fit=stats.fit_dwell_exponential(x,t_min=.001,n_boot=0)
    reference=scipy_stats.kstest(x,lambda t:1-np.exp(-(t-.001)/fit.params['tau_1'])).statistic
    assert fit.meta['gof']['ks']==pytest.approx(reference,abs=1e-13)


@pytest.mark.parametrize('family,params,distribution',[
    ('gamma',{'k':3.,'theta':.01},scipy_stats.gamma(a=3.,scale=.01)),
    ('inverse-gaussian',{'mu':.03,'lambda':.12},scipy_stats.invgauss(mu=.25,scale=.12)),
])
def test_peaked_cdf_and_pdf_obey_complete_truncation_support(family,params,distribution):
    lo,hi=.008,.06;meta={'t_min':lo,'t_max':hi}
    x=np.linspace(0,.08,200)
    reference=np.clip((distribution.cdf(np.clip(x,lo,hi))-distribution.cdf(lo))/
                      (distribution.cdf(hi)-distribution.cdf(lo)),0,1)
    np.testing.assert_allclose(stats.cdf_from(family,params,meta)(x),reference,atol=2e-14)
    pdf=stats.density_from(family,params,meta)
    assert np.all(pdf(np.array([0.,lo/2,hi*2]))==0.)
    grid=np.linspace(lo,hi,20000)
    assert np.trapz(pdf(grid),grid)==pytest.approx(1.,abs=1e-7)


def test_efficiency_cdf_uses_the_same_selection_normalizer():
    params={'tau_1':.01};meta={'t_min':.001,'t_max':.05,
        'eta_grid':[.001,.003,.008,.02],'eta_vals':[.1,.4,.8,.95]}
    x=np.linspace(.001,.05,10000)
    pdf=stats.density_from('exp-mixture',params,meta)(x)
    cdf=stats.cdf_from('exp-mixture',params,meta)(x)
    numerical=np.r_[0.,np.cumsum((pdf[1:]+pdf[:-1])*np.diff(x)*.5)]
    np.testing.assert_allclose(cdf,numerical,atol=5e-7)
    assert cdf[0]==0. and cdf[-1]==pytest.approx(1.)


def test_mixture_support_is_observable_not_preselection_population(monkeypatch):
    taus=np.array([.0001,.01]);weights=np.array([.99,.01])
    # Isolate reporting from optimizer variation; this is the analytic conditional
    # population counterexample in the audit, not a fitted-data recovery test.
    monkeypatch.setattr(stats,'_fit_mix',lambda *a,**k:(taus,weights,0.,np.zeros(3)))
    x=.001+np.random.default_rng(9).exponential(.01,2000)
    fit=stats.fit_dwell_exponential(x,t_min=.001,n_components=2,n_boot=0)
    observed=weights*np.exp(-.001/taus);observed/=observed.sum()
    np.testing.assert_allclose(fit.meta['observed_component_weights'],observed)
    assert fit.meta['observed_component_support'][0]<10
    assert any('observed share' in note and '~10 events' in note for note in fit.meta['warnings'])


def test_bootstrap_band_survives_json_roundtrip_exactly():
    x=.001+np.random.default_rng(10).exponential(.01,500)
    fit=stats.fit_dwell_exponential(x,t_min=.001,n_boot=60)
    grid=np.linspace(.001,.04,90)
    original=stats.ci_band(fit,grid)
    blob=json.loads(json.dumps(stats.fit_to_json(fit)))
    assert blob['uncertainty_schema_version']==1
    restored=stats.ci_band(stats.fit_from_json(blob),grid)
    np.testing.assert_array_equal(original['lo'],restored['lo'])
    np.testing.assert_array_equal(original['hi'],restored['hi'])
    assert original['n_replicates']==restored['n_replicates']==60
    # Backward compatibility: old saved fits retain estimates, without inventing a band.
    blob['meta'].pop('boot');blob['meta'].pop('boot_params')
    old=stats.fit_from_json(blob)
    assert old.params==fit.params and old.ci==fit.ci
    assert stats.ci_band(old,grid) is None


def test_truncated_gaussian_recovers_population_and_preserves_density():
    raw=np.random.default_rng(401).normal(50,20,10000)
    x=raw[(raw>=45)&(raw<=90)]
    fit=stats.fit_depth_gaussians(x,lower=45,upper=90,n_boot=0,quantity='depth_pA',unit='pA')
    assert fit.params['mu_1']==pytest.approx(50,abs=1)
    assert fit.params['sigma_1']==pytest.approx(20,abs=1)
    assert fit.meta['converged'] and not fit.meta['numerical_bound_reached']
    grid=np.linspace(45,90,10000)
    assert np.trapz(fit.predict(grid),grid)==pytest.approx(1.,abs=1e-7)
    assert np.all(fit.predict([0,44,91])==0.)
    restored=stats.fit_from_json(json.loads(json.dumps(stats.fit_to_json(fit))))
    np.testing.assert_array_equal(fit.predict(grid),restored.predict(grid))


def test_truncated_gaussian_mixture_recovers_separated_components():
    rng=np.random.default_rng(17)
    raw=np.r_[rng.normal(30,4,2500),rng.normal(65,6,2500)]
    x=raw[(raw>=28)&(raw<=68)]
    fit=stats.fit_depth_gaussians(x,n_components=2,lower=28,upper=68,n_boot=0)
    assert fit.params['mu_1']==pytest.approx(30,abs=1.0)
    assert fit.params['mu_2']==pytest.approx(65,abs=1.0)
    assert fit.params['sigma_1']==pytest.approx(4,abs=.8)
    assert fit.params['sigma_2']==pytest.approx(6,abs=.8)
    assert fit.params['w_1']==pytest.approx(.5,abs=.06)


@pytest.mark.parametrize('bounds',[{'lower':70,'upper':40},{'lower':np.nan},{'upper':45}])
def test_truncated_gaussian_refuses_inconsistent_support(bounds):
    x=np.linspace(40,60,100)
    with pytest.raises(ValueError):stats.fit_depth_gaussians(x,n_boot=0,**bounds)


def test_truncated_gaussian_deferred_uncertainty_matches_sync():
    x=np.random.default_rng(6).normal(50,9,250);x=x[(x>44)&(x<65)]
    expected=stats.fit_depth_gaussians(x,lower=44,upper=65,n_boot=20,seed=12)
    with stats.deferred_ci() as jobs:
        actual=stats.fit_depth_gaussians(x,lower=44,upper=65,n_boot=20,seed=12)
    assert len(jobs)==1 and actual.meta['ci_pending']
    stats.apply_ci(actual,jobs[0]())
    assert actual.ci==expected.ci
    np.testing.assert_array_equal(actual.meta['boot'],expected.meta['boot'])
    assert actual.meta['n_boot_ok']>=20


def test_exp_gamma_optional_deferred_bootstrap_produces_persisted_bands():
    rng=np.random.default_rng(18)
    x=np.r_[rng.exponential(.001,60),rng.gamma(15,.0015,140)]
    x=x[x>=.0001]
    with stats.deferred_ci() as jobs:
        fit=stats.fit_dwell_exp_gamma(x,t_min=.0001,n_boot=20,seed=7)
    assert len(jobs)==1 and fit.meta['ci_pending']
    stats.apply_ci(fit,jobs[0]())
    assert fit.meta['n_boot_ok']==20 and len(fit.ci)==4
    grid=np.linspace(.0001,.05,100)
    band=stats.ci_band(fit,grid)
    restored=stats.fit_from_json(json.loads(json.dumps(stats.fit_to_json(fit))))
    np.testing.assert_array_equal(band['lo'],stats.ci_band(restored,grid)['lo'])


@pytest.mark.parametrize('kwargs',[{'t_min':-.1},{'t_max':.001},{'t_max':np.nan},{'t_min':.02,'t_max':.01}])
def test_exp_gamma_refuses_inconsistent_truncation(kwargs):
    x=np.linspace(.002,.012,50)
    with pytest.raises(ValueError):stats.fit_dwell_exp_gamma(x,**kwargs)
