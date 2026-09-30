"""Hard assignments must not acquire confidence intervals for soft weights."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from nanoscope.analysis import incubation_study as study
from nanoscope.plot import incubation_figures as figures


def _overlapping_components():
    # Known overlapping normals, evaluated without fitting or random draws.
    # Four observations favor component 0, one favors component 1; the fitted
    # density weights remain 0.6/0.4. Every posterior is below 0.65 confidence.
    model = GaussianMixture(n_components=2, covariance_type="full")
    model.weights_ = np.array([.6, .4])
    model.means_ = np.array([[-.5], [.5]])
    model.covariances_ = np.ones((2, 1, 1))
    model.precisions_cholesky_ = np.ones((2, 1, 1))
    samples = np.array([[0.], [.1], [.2], [.3], [.8]])
    posteriors = model.predict_proba(samples)
    labels = pd.Series(model.predict(samples))
    fit = SimpleNamespace(k=2, weights=model.weights_, labels=labels,
                          confidence=posteriors.max(axis=1),
                          sizes={k: int((labels == k).sum()) for k in range(2)})
    # Deliberately exclude each fitted point from its interval. Percentile
    # intervals can do this; display code must preserve both actual endpoints.
    ci = SimpleNamespace(lo=np.array([.61111, .30001]), hi=np.array([.70001, .38999]))
    features = pd.DataFrame(dict(dwell_ms=np.ones(len(samples))))
    table = study.SG.population_table(features, fit, duration_s=10., ci=ci)
    return SimpleNamespace(fit=fit, ci=ci, table=table, n_events=len(samples))


def test_population_export_separates_overlapping_component_estimands():
    quant = _overlapping_components()
    assert quant.fit.confidence.max() < .65
    assert quant.fit.sizes == {0: 4, 1: 1}
    before = quant.table.copy(deep=True)
    table = study._appendix_population_table(quant, 10.)
    np.testing.assert_array_equal(table.assigned_event_fraction, [.8, .2])
    np.testing.assert_array_equal(table.assigned_event_rate_per_s, [.4, .1])
    np.testing.assert_array_equal(table.mixture_weight, [.6, .4])
    np.testing.assert_array_equal(table.mixture_weight_lo95, quant.ci.lo)
    np.testing.assert_array_equal(table.mixture_weight_hi95, quant.ci.hi)
    assert not {'fraction', 'rate_per_s', 'frac_lo95', 'frac_hi95'}.intersection(table.columns)
    pd.testing.assert_frame_equal(quant.table, before)


def test_weight_plot_preserves_intervals_excluding_points_and_distinguishes_counts():
    quant = _overlapping_components()
    fig = figures._figure(width=5, height=4)
    axis = fig.subplots()
    figures._plot_population_fractions(axis, quant)
    lines = {line.get_label(): line for line in axis.lines}
    np.testing.assert_array_equal(lines['Fitted mixture weight'].get_ydata(), [.6, .4])
    np.testing.assert_array_equal(lines['Assigned event fraction'].get_ydata(), [.8, .2])
    segments = axis.collections[0].get_segments()
    np.testing.assert_array_equal(segments[0][:, 1], [quant.ci.lo[0], quant.ci.hi[0]])
    np.testing.assert_array_equal(segments[1][:, 1], [quant.ci.lo[1], quant.ci.hi[1]])
    fig.canvas.draw()
    fig.clear()


def test_interval_diagnostics_use_weights_and_retain_other_bootstrap_warnings():
    quant = _overlapping_components()
    quant.ci.lo, quant.ci.hi = np.array([.05, .05]), np.array([.95, .95])
    old = study.SG._ci_warnings(np.array([.8, .2]), quant.ci.lo, quant.ci.hi)
    assert 'subpop 0 (0.800,' in old[0]
    failed_refit = 'BOOTSTRAP REPLICATES LOST: 1 of 20 refits did not converge and was dropped.'
    independent_calibration = 'Independent bootstrap calibration caveat.'
    quant.ci.warnings = (failed_refit, independent_calibration, *old)
    quant.warnings = ('Unchanged separation warning.', *quant.ci.warnings)
    study._appendix_weight_warnings(quant)
    for warnings in (quant.ci.warnings, quant.warnings):
        text = '\n'.join(warnings)
        assert 'subpop 0 (0.600,' in text and 'subpop 1 (0.400,' in text
        assert 'subpop 0 (0.800,' not in text and 'subpop 1 (0.200,' not in text
        assert 'mixture weight' in text and 'fraction' not in text
        assert 'NOT calibrated' in text  # updated switching diagnostic retained
        assert failed_refit in warnings and independent_calibration in warnings
    assert quant.warnings[0] == 'Unchanged separation warning.'
