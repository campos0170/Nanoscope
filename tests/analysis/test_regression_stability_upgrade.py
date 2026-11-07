import numpy as np
import pytest
from scipy.stats import linregress, t
from nanoscope.analysis.populations import linear_fit
from nanoscope.analysis.iv import iv_fit
from nanoscope.analysis.voltage_dep import vdep_fit


def test_large_time_origin_does_not_destroy_slope():
    x = 1e15 + np.arange(30)
    y = 2.5 * np.arange(30) + 3
    result = linear_fit(x, y)
    assert result.slope == pytest.approx(2.5)
    assert result.r2 == pytest.approx(1)
    np.testing.assert_allclose(result.predict(x), y, atol=1e-10)


def test_ols_uncertainty_matches_scipy_and_converts_conductance():
    rng = np.random.default_rng(24)
    x = np.linspace(-1, 1, 40)
    y = 100*x + 3 + rng.normal(0, 4, x.size)
    ref = linregress(x, y)
    result = linear_fit(x, y)
    assert result.meta['standard_errors']['slope'] == pytest.approx(ref.stderr)
    assert result.meta['standard_errors']['intercept'] == pytest.approx(ref.intercept_stderr)
    assert result.meta['ci95']['slope'] == pytest.approx((ref.slope-t.ppf(.975,38)*ref.stderr,ref.slope+t.ppf(.975,38)*ref.stderr))
    iv = iv_fit(x,y,current_units='pA')
    assert iv.conductance_ci95_nS == pytest.approx(np.array(result.meta['ci95']['slope'])*.001)
    assert iv_fit(x[:2],y[:2],current_units='pA').slope_ci95 is None


@pytest.mark.parametrize('constant',[1.,2.,10.,1e-12,1e12])
def test_constant_voltage_response_has_no_identifiable_exponential_scale(constant):
    with pytest.raises(ValueError, match='does not vary'):
        vdep_fit(np.linspace(-1,1,10),np.full(10,constant),model='exponential')


def test_voltage_fit_reports_objective_and_rejects_shape_mismatch():
    x = np.linspace(-1,1,10)
    result = vdep_fit(x,3*np.exp(2*x),model='exponential')
    assert result['r2_space']=='ln(response)'
    assert result['params']['V0']==pytest.approx(.5)
    with pytest.raises(ValueError, match='same length'):
        vdep_fit(x,[1,2])
