"""Tests for analysis.stats.

The load-bearing test is `test_naive_uncensored_fit_is_biased_long`: it measures
the error you would make by fitting the survivors as if the sample were complete.
If that test ever goes green for the wrong reason, the module has no purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from nanoscope.analysis.stats import (
    FitResult,
    fit_depth_gaussians,
    fit_dwell_exponential,
    fit_dwell_peaked,
)


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


def sample_truncated_exp(tau, t_min, n, rng):
    """Exact left-truncated exponential: memorylessness makes this t_min + Exp(tau)."""
    return t_min + rng.exponential(tau, size=n)


def sample_truncated_mixture(taus, weights, t_min, n, rng):
    """Left-truncated mixture by rejection -- i.e. literally conditioning on survival."""
    out = []
    while sum(len(o) for o in out) < n:
        m = 4 * n
        j = rng.choice(len(taus), size=m, p=weights)
        t = rng.exponential(np.asarray(taus)[j])
        out.append(t[t >= t_min])
    return np.concatenate(out)[:n]


# ---------------------------------------------------------------------------
# the core claim: censored MLE recovers tau, naive does not
# ---------------------------------------------------------------------------


def test_censored_mle_recovers_tau_within_bootstrap_ci():
    rng = np.random.default_rng(11)
    tau, t_min, n = 1.0e-3, 2.0e-4, 3000
    t = sample_truncated_exp(tau, t_min, n, rng)

    fit = fit_dwell_exponential(t, t_min=t_min, n_boot=1000, seed=3)

    lo, hi = fit.ci["tau_1"]
    assert lo < tau < hi, f"true tau={tau:g} outside CI [{lo:g}, {hi:g}]"
    assert abs(fit.params["tau_1"] - tau) / tau < 0.06
    assert fit.n == n
    assert np.isfinite(fit.loglik) and np.isfinite(fit.aic) and np.isfinite(fit.bic)


def test_numerical_mle_matches_closed_form():
    rng = np.random.default_rng(4)
    tau, t_min = 3.3e-4, 7.0e-5
    t = sample_truncated_exp(tau, t_min, 2000, rng)

    fit = fit_dwell_exponential(t, t_min=t_min, n_boot=0)

    # fit_dwell_exponential raises internally if its numerical MLE disagrees with
    # mean(t) - t_min; this pins the returned value to the closed form as well.
    assert fit.params["tau_1"] == pytest.approx(t.mean() - t_min, rel=1e-12)
    assert fit.meta["closed_form"] is True


def test_naive_uncensored_fit_is_biased_long():
    """Quantify the error the module exists to prevent.

    Truncating at t_min = tau/2 makes the naive estimator overshoot by ~50%: it
    estimates E[t] = tau + t_min instead of tau. The naive fit's *shape* still
    matches the histogram perfectly, which is why this bias survives review.
    """
    rng = np.random.default_rng(7)
    tau, t_min, n = 1.0e-3, 5.0e-4, 20000
    t = sample_truncated_exp(tau, t_min, n, rng)

    censored = fit_dwell_exponential(t, t_min=t_min, n_boot=0).params["tau_1"]

    # what a nanopore paper typically does instead:
    _, naive_scipy = scipy.stats.expon.fit(t, floc=0)
    naive_mean = t.mean()

    err_censored = abs(censored - tau) / tau
    err_naive = abs(naive_scipy - tau) / tau

    assert naive_scipy == pytest.approx(naive_mean, rel=1e-6)
    assert err_censored < 0.02, f"censored fit off by {err_censored:.1%}"
    assert err_naive > 0.45, f"expected ~50% naive bias, got {err_naive:.1%}"
    assert naive_scipy > tau  # biased *long*, always, never short
    assert err_naive > 20 * err_censored
    # the bias is exactly t_min in expectation, not a vague inflation
    assert (naive_scipy - censored) == pytest.approx(t_min, rel=1e-6)


@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 2.0])
def test_bias_scales_with_t_min_over_tau(ratio):
    """The naive relative bias is t_min/tau. Show it across the regime."""
    rng = np.random.default_rng(int(100 * ratio))
    tau, n = 5.0e-4, 40000
    t_min = ratio * tau
    t = sample_truncated_exp(tau, t_min, n, rng)

    censored = fit_dwell_exponential(t, t_min=t_min, n_boot=0).params["tau_1"]
    naive = t.mean()

    assert abs(censored - tau) / tau < 0.03
    assert (naive - tau) / tau == pytest.approx(ratio, rel=0.05)


# ---------------------------------------------------------------------------
# mixtures
# ---------------------------------------------------------------------------


def test_two_component_mixture_recovered_and_bic_prefers_two():
    rng = np.random.default_rng(21)
    taus, w, t_min, n = (2.0e-4, 3.0e-3), (0.6, 0.4), 5.0e-5, 6000
    t = sample_truncated_mixture(taus, w, t_min, n, rng)

    f1 = fit_dwell_exponential(t, t_min=t_min, n_components=1, n_boot=0)
    f2 = fit_dwell_exponential(t, t_min=t_min, n_components=2, n_boot=0)

    assert f2.bic < f1.bic, "BIC failed to prefer the true 2-component model"
    assert f2.aic < f1.aic
    assert f2.loglik > f1.loglik

    assert f2.params["tau_1"] == pytest.approx(taus[0], rel=0.20)
    assert f2.params["tau_2"] == pytest.approx(taus[1], rel=0.20)
    assert f2.params["w_1"] == pytest.approx(w[0], abs=0.08)
    assert f2.params["w_1"] + f2.params["w_2"] == pytest.approx(1.0)
    assert f2.params["tau_1"] < f2.params["tau_2"]  # sorted, so CIs are not label-switched


def test_bic_does_not_prefer_two_components_on_single_exponential_data():
    rng = np.random.default_rng(33)
    t = sample_truncated_exp(8.0e-4, 1.0e-4, 4000, rng)

    f1 = fit_dwell_exponential(t, t_min=1.0e-4, n_components=1, n_boot=0)
    f2 = fit_dwell_exponential(t, t_min=1.0e-4, n_components=2, n_boot=0)

    assert f1.bic < f2.bic, "BIC bought a spurious second component"


def test_mixture_bootstrap_ci_brackets_truth():
    rng = np.random.default_rng(5)
    taus, w, t_min = (1.5e-4, 4.0e-3), (0.5, 0.5), 5.0e-5
    t = sample_truncated_mixture(taus, w, t_min, 4000, rng)

    fit = fit_dwell_exponential(t, t_min=t_min, n_components=2, n_boot=150, seed=1)

    for key, truth in (("tau_1", taus[0]), ("tau_2", taus[1])):
        lo, hi = fit.ci[key]
        assert lo < truth < hi, f"{key}: true {truth:g} outside CI [{lo:g}, {hi:g}]"


# ---------------------------------------------------------------------------
# efficiency weighting
# ---------------------------------------------------------------------------


def _eta_logistic(t, t50=5.0e-4, width=1.5e-4):
    return 1.0 / (1.0 + np.exp(-(np.asarray(t) - t50) / width))


def test_efficiency_weighting_corrects_a_distorted_sample():
    """A soft detection roll-off biases tau long even with an honest t_min.

    Here events are kept with probability eta(t), so nothing is missing below a
    hard cutoff -- the sample is thinned smoothly. A truncation-only fit cannot
    see that and reports tau too long. Feeding the measured eta(t) (the shape
    truth.recovery_map returns) recovers the truth.
    """
    rng = np.random.default_rng(19)
    tau = 1.0e-3
    t_all = rng.exponential(tau, size=200_000)
    keep = rng.random(t_all.size) < _eta_logistic(t_all)
    t = t_all[keep]
    assert t.size > 5000

    # the logistic floors at ~0.03 rather than 0, so a handful of very short
    # events do survive; the stated floor has to be honest about that
    t_min = 1e-9
    assert t.min() > t_min

    eta_grid = np.geomspace(1e-9, 1e-1, 300)
    recovery = pd.DataFrame({"dwell_s": eta_grid, "efficiency": _eta_logistic(eta_grid)})

    naive = fit_dwell_exponential(t, t_min=t_min, n_boot=0).params["tau_1"]
    corrected = fit_dwell_exponential(
        t, t_min=t_min, efficiency=recovery, n_boot=0
    ).params["tau_1"]

    err_naive = abs(naive - tau) / tau
    err_corr = abs(corrected - tau) / tau

    assert err_naive > 0.15, f"distortion too weak to be a test ({err_naive:.1%})"
    assert err_corr < 0.05, f"efficiency-weighted fit still off by {err_corr:.1%}"
    assert err_corr < err_naive / 3


def test_efficiency_accepts_callable_and_array_forms():
    rng = np.random.default_rng(23)
    tau = 8.0e-4
    t_all = rng.exponential(tau, size=80_000)
    t = t_all[rng.random(t_all.size) < _eta_logistic(t_all, t50=3e-4, width=1e-4)]

    grid = np.geomspace(1e-9, 1e-1, 300)
    vals = _eta_logistic(grid, t50=3e-4, width=1e-4)

    a = fit_dwell_exponential(t, t_min=1e-9, efficiency=(grid, vals), n_boot=0)
    b = fit_dwell_exponential(
        t, t_min=1e-9, efficiency=lambda x: _eta_logistic(x, t50=3e-4, width=1e-4), n_boot=0
    )
    c = fit_dwell_exponential(
        t, t_min=1e-9, efficiency=pd.DataFrame({"dwell": grid, "recovery": vals}), n_boot=0
    )
    for f in (a, b, c):
        assert f.params["tau_1"] == pytest.approx(tau, rel=0.06)


def test_efficiency_from_detected_injected_counts():
    """recovery_map may report raw counts rather than a ratio."""
    rng = np.random.default_rng(2)
    t = sample_truncated_exp(1e-3, 1e-4, 500, rng)
    grid = np.geomspace(1e-5, 1e-1, 40)
    df = pd.DataFrame(
        {"dwell_s": grid, "n_injected": 100, "n_detected": np.round(100 * _eta_logistic(grid, 1e-4, 3e-5))}
    )
    fit = fit_dwell_exponential(t, t_min=1e-4, efficiency=df, n_boot=0)
    assert np.isfinite(fit.params["tau_1"]) and fit.params["tau_1"] > 0


def test_flat_unit_efficiency_reduces_to_hard_truncation():
    rng = np.random.default_rng(9)
    t = sample_truncated_exp(6e-4, 1e-4, 1500, rng)
    grid = np.geomspace(1e-4, 1.0, 100)
    flat = pd.DataFrame({"dwell_s": grid, "efficiency": np.ones_like(grid)})

    hard = fit_dwell_exponential(t, t_min=1e-4, n_boot=0)
    soft = fit_dwell_exponential(t, t_min=1e-4, efficiency=flat, n_boot=0)

    assert soft.params["tau_1"] == pytest.approx(hard.params["tau_1"], rel=1e-3)


def test_efficiency_map_over_dwell_and_depth_is_marginalised_with_a_warning():
    rng = np.random.default_rng(31)
    t = sample_truncated_exp(1e-3, 1e-4, 300, rng)
    grid = np.geomspace(1e-5, 1e-1, 30)
    df = pd.DataFrame(
        {
            "dwell_s": np.repeat(grid, 3),
            "frac_depth": np.tile([0.1, 0.2, 0.3], grid.size),
            "efficiency": np.repeat(_eta_logistic(grid, 1e-4, 3e-5), 3),
        }
    )
    fit = fit_dwell_exponential(t, t_min=1e-4, efficiency=df, n_boot=0)
    assert any("depth" in w for w in fit.meta["warnings"])


def test_efficiency_grid_above_t_min_is_flagged():
    rng = np.random.default_rng(13)
    t = sample_truncated_exp(1e-3, 1e-4, 400, rng)
    grid = np.geomspace(1e-3, 1e-1, 50)
    df = pd.DataFrame({"dwell_s": grid, "efficiency": np.linspace(0.5, 1.0, 50)})
    fit = fit_dwell_exponential(t, t_min=1e-4, efficiency=df, n_boot=0)
    assert any("extrapolation" in w for w in fit.meta["warnings"])


# ---------------------------------------------------------------------------
# fail loudly
# ---------------------------------------------------------------------------


def test_too_few_events_raises():
    with pytest.raises(ValueError, match="fewer than 20"):
        fit_dwell_exponential(np.linspace(1e-4, 1e-3, 19), t_min=1e-5)


def test_identical_values_raise():
    with pytest.raises(ValueError, match="identical"):
        fit_dwell_exponential(np.full(50, 3e-4), t_min=1e-5)
    with pytest.raises(ValueError, match="identical"):
        fit_depth_gaussians(np.full(50, 0.2))


@pytest.mark.parametrize("bad", [0.0, -1e-4, -5.0])
def test_nonpositive_t_min_raises(bad):
    rng = np.random.default_rng(0)
    t = sample_truncated_exp(1e-3, 1e-4, 100, rng)
    with pytest.raises(ValueError, match="t_min must be a positive"):
        fit_dwell_exponential(t, t_min=bad)


def test_t_min_above_shortest_event_raises():
    """Events below your own stated detection limit mean the model is a lie."""
    rng = np.random.default_rng(0)
    t = sample_truncated_exp(1e-3, 1e-4, 100, rng)
    with pytest.raises(ValueError, match="shorter than the stated detection limit"):
        fit_dwell_exponential(t, t_min=5e-4)


def test_t_min_exactly_at_min_dwell_is_accepted():
    """dwell = n_samples/fs round-trips; a float-equal t_min must not trip the guard."""
    rng = np.random.default_rng(0)
    fs = 50_000.0
    n_samp = rng.geometric(0.02, size=200) + 1
    t = n_samp / fs
    fit = fit_dwell_exponential(t, t_min=t.min(), n_boot=0)
    assert fit.params["tau_1"] > 0


def test_non_finite_and_nonpositive_dwell_raise():
    rng = np.random.default_rng(0)
    t = sample_truncated_exp(1e-3, 1e-4, 100, rng)
    with pytest.raises(ValueError, match="non-finite"):
        fit_dwell_exponential(np.r_[t, np.nan], t_min=1e-4)
    with pytest.raises(ValueError, match="non-positive"):
        fit_dwell_exponential(np.r_[t, -1e-4], t_min=1e-4)


def test_negative_depth_raises_sign_convention():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0.2, 0.02, 100), [-0.05]]
    with pytest.raises(ValueError, match="sign convention"):
        fit_depth_gaussians(x)


def test_heavy_truncation_tau_below_t_min_still_fits():
    """tau below the detection limit: most mass piles up just above t_min.

    A realistic hard case (t_min = 5*tau). tau = mean(t) - t_min is much smaller
    than the shortest observed dwell, so the numerical optimiser's tau bounds must
    track tau's scale, not min(dwell); the closed-form cross-check inside the fit
    would raise if they didn't.
    """
    rng = np.random.default_rng(0)
    tau, t_min = 2.0e-5, 1.0e-4
    t = sample_truncated_exp(tau, t_min, 4000, rng)
    fit = fit_dwell_exponential(t, t_min=t_min, n_boot=0)
    assert fit.params["tau_1"] == pytest.approx(t.mean() - t_min, rel=1e-9)
    assert fit.params["tau_1"] == pytest.approx(tau, rel=0.05)


def test_bad_efficiency_object_raises():
    rng = np.random.default_rng(0)
    t = sample_truncated_exp(1e-3, 1e-4, 100, rng)
    with pytest.raises(ValueError, match="no recognisable dwell column"):
        fit_dwell_exponential(t, t_min=1e-4, efficiency=pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_dwell_exponential(t, t_min=1e-4, efficiency=(np.array([1e-4, 1e-2]), np.array([0.5, 1.7])))
    with pytest.raises(TypeError):
        fit_dwell_exponential(t, t_min=1e-4, efficiency="the recovery map")


def test_too_few_events_for_requested_components():
    rng = np.random.default_rng(0)
    t = sample_truncated_exp(1e-3, 1e-4, 30, rng)
    with pytest.raises(ValueError, match="too few for a 2-component"):
        fit_dwell_exponential(t, t_min=1e-4, n_components=2)


# ---------------------------------------------------------------------------
# depth mixtures
# ---------------------------------------------------------------------------


def test_depth_single_gaussian_recovered():
    rng = np.random.default_rng(0)
    x = rng.normal(0.25, 0.03, 2000)
    fit = fit_depth_gaussians(x, n_boot=200)
    assert fit.params["mu_1"] == pytest.approx(0.25, abs=0.005)
    assert fit.params["sigma_1"] == pytest.approx(0.03, rel=0.08)
    lo, hi = fit.ci["mu_1"]
    assert lo < 0.25 < hi


def test_depth_two_populations_recovered_and_bic_prefers_two():
    rng = np.random.default_rng(17)
    n = 3000
    z = rng.random(n) < 0.65
    x = np.where(z, rng.normal(0.12, 0.015, n), rng.normal(0.34, 0.03, n))

    f1 = fit_depth_gaussians(x, n_components=1, n_boot=0)
    f2 = fit_depth_gaussians(x, n_components=2, n_boot=100)

    assert f2.bic < f1.bic, "BIC failed to prefer the true 2-population model"
    assert f2.params["mu_1"] == pytest.approx(0.12, abs=0.01)
    assert f2.params["mu_2"] == pytest.approx(0.34, abs=0.015)
    assert f2.params["sigma_1"] == pytest.approx(0.015, rel=0.15)
    assert f2.params["w_1"] == pytest.approx(0.65, abs=0.05)
    assert f2.params["w_1"] + f2.params["w_2"] == pytest.approx(1.0)


def test_depth_bic_does_not_prefer_two_on_unimodal_data():
    rng = np.random.default_rng(6)
    x = rng.normal(0.2, 0.02, 2000)
    f1 = fit_depth_gaussians(x, n_components=1, n_boot=0)
    f2 = fit_depth_gaussians(x, n_components=2, n_boot=0)
    assert f1.bic < f2.bic


# ---------------------------------------------------------------------------
# FitResult surface
# ---------------------------------------------------------------------------


def test_predict_is_a_normalised_density_over_the_observed_support():
    rng = np.random.default_rng(0)
    tau, t_min = 1e-3, 2e-4
    fit = fit_dwell_exponential(sample_truncated_exp(tau, t_min, 2000, rng), t_min=t_min, n_boot=0)

    x = np.linspace(t_min, t_min + 40 * tau, 200_000)
    assert np.trapz(fit.predict(x), x) == pytest.approx(1.0, rel=1e-3)
    assert fit.predict(np.array([t_min * 0.5]))[0] == 0.0  # nothing below the limit
    assert np.all(fit.predict(x) >= 0)


def test_predict_density_matches_the_observed_histogram():
    rng = np.random.default_rng(8)
    tau, t_min = 1e-3, 3e-4
    t = sample_truncated_exp(tau, t_min, 50_000, rng)
    fit = fit_dwell_exponential(t, t_min=t_min, n_boot=0)

    dens, edges = np.histogram(t, bins=40, range=(t_min, t_min + 3 * tau), density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    pred = fit.predict(centres)
    assert np.max(np.abs(pred - dens)) / dens.max() < 0.05


def test_depth_predict_integrates_to_one():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0.1, 0.01, 1000), rng.normal(0.3, 0.02, 1000)]
    fit = fit_depth_gaussians(x, n_components=2, n_boot=0)
    g = np.linspace(-0.2, 0.8, 200_000)
    assert np.trapz(fit.predict(g), g) == pytest.approx(1.0, rel=1e-3)


def test_summary_is_readable_and_mentions_censoring():
    rng = np.random.default_rng(0)
    fit = fit_dwell_exponential(sample_truncated_exp(1e-3, 1e-4, 500, rng), t_min=1e-4, n_boot=200)
    s = fit.summary()
    assert "tau_1" in s and "95% CI" in s and "BIC" in s
    assert "truncation" in s
    assert isinstance(fit, FitResult) and len(s.splitlines()) >= 4


def test_aic_bic_relationship():
    rng = np.random.default_rng(0)
    fit = fit_dwell_exponential(sample_truncated_exp(1e-3, 1e-4, 500, rng), t_min=1e-4, n_boot=0)
    assert fit.aic == pytest.approx(2 * 1 - 2 * fit.loglik)
    assert fit.bic == pytest.approx(1 * np.log(500) - 2 * fit.loglik)


def test_accepts_pandas_series_from_an_eventset():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"dwell_s": sample_truncated_exp(1e-3, 1e-4, 300, rng)})
    fit = fit_dwell_exponential(df["dwell_s"], t_min=1e-4, n_boot=0)
    assert fit.params["tau_1"] == pytest.approx(df["dwell_s"].mean() - 1e-4, rel=1e-9)


def test_the_fitted_density_actually_carries_the_efficiency_weighting():
    """``predict()`` must return eta(t)*f(t)/Z, not the bare exponential at the corrected tau.

    This is checked against an INDEPENDENT reimplementation of the formula rather than
    against another call into stats, and that is the whole point of the test. Since v6 the
    fitters build their own density through :func:`density_from`, reading eta out of
    ``meta['eta_grid'/'eta_vals']`` — so a round-trip identity test cannot catch eta going
    missing from ``meta``: both sides would then be identically wrong and agree perfectly.
    Deleting that key was measured to leave all 138 fit tests green while the plotted curve
    silently lost its eta factor (and with it up to 30% of its shape near t_min).

    The parameters and the likelihood are unaffected either way — only the drawn curve is,
    which is exactly the kind of defect that survives a green suite.
    """
    from nanoscope.analysis._compat import trapezoid
    from nanoscope.analysis.stats import _eta_at, _mix_pdf

    rng = np.random.default_rng(11)
    t = sample_truncated_exp(6e-4, 1e-4, 1500, rng)
    grid = np.geomspace(1e-4, 1e-2, 24)
    vals = np.clip(1.0 - np.exp(-grid / 5e-4), 0.0, 1.0)   # steeply rolls off at short t
    fit = fit_dwell_exponential(t, t_min=1e-4, efficiency=(grid, vals), n_boot=0)

    x = np.geomspace(1.05e-4, 8e-3, 60)
    tau = np.array([fit.params["tau_1"]])
    w = np.array([1.0])
    raw = _mix_pdf(x, tau, w) * _eta_at(x, grid, vals)
    expected = raw / trapezoid(
        _mix_pdf(_q := np.geomspace(1e-4, 1.0, 20_000), tau, w) * _eta_at(_q, grid, vals), _q)

    assert np.allclose(fit.predict(x), expected, rtol=2e-3), (
        "the fitted density does not include eta(t); it is the plain exponential")

    # And the shape claim that number encodes: eta suppresses short dwells relative to the
    # SAME-tau unweighted exponential. With eta dropped this ratio is flat.
    plain = _mix_pdf(x, tau, w)
    ratio = fit.predict(x) / plain
    assert ratio[0] < 0.5 * ratio[-1], "eta is not shaping the curve at short dwells"


@pytest.mark.parametrize("build, lo, hi, label", [
    (lambda t: fit_dwell_exponential(t, t_min=1e-4, n_boot=0), 1e-4, None, "left-truncated"),
    (lambda t: fit_dwell_exponential(t[t <= 5e-3], t_min=1e-4, t_max=5e-3, n_boot=0),
     1e-4, 5e-3, "doubly truncated"),
    (lambda t: fit_dwell_exponential(t, t_min=1e-4, n_components=2, n_boot=0),
     1e-4, None, "k=2 mixture"),
    (lambda t: fit_dwell_exponential(
        t, t_min=1e-4, n_boot=0,
        efficiency=(np.geomspace(1e-4, 1e-2, 24),
                    np.clip(1 - np.exp(-np.geomspace(1e-4, 1e-2, 24) / 5e-4), 0, 1))),
     1e-4, None, "eta-weighted"),
    (lambda t: fit_dwell_peaked(t, model="gamma", t_min=1e-4), 1e-4, None, "gamma"),
    (lambda t: fit_dwell_peaked(t, model="inverse-gaussian", t_min=1e-4),
     1e-4, None, "inverse-gaussian"),
])
def test_every_fitted_density_integrates_to_one_over_its_observable_window(build, lo, hi, label):
    """predict() claims to be a DENSITY of the observed data. Check that it is one.

    This is the property a round-trip identity test structurally cannot see. Since v6 the
    fitter and the project restore share density_from(), so both sides of a round trip
    compute the same normalisation -- and would agree perfectly while both being wrong. A
    dropped or doubled Z, a truncation mass taken over the wrong interval, or an eta factor
    applied without renormalising are all invisible to identity and all caught here.

    Nothing else in the suite pins this: it is asserted against the integral itself, not
    against another call into stats.
    """
    from nanoscope.analysis._compat import trapezoid

    rng = np.random.default_rng(3)
    t = sample_truncated_exp(6e-4, 1e-4, 2000, rng)
    fit = build(t)

    top = hi if hi is not None else 1.0          # far past any fitted tau
    x = np.geomspace(lo, top, 200_000)
    area = float(trapezoid(fit.predict(x), x))
    assert area == pytest.approx(1.0, rel=2e-3), f"{label}: density integrates to {area:.6g}, not 1"


def test_the_depth_mixture_density_integrates_to_one():
    """The untruncated sibling of the test above."""
    from nanoscope.analysis._compat import trapezoid

    rng = np.random.default_rng(3)
    d = np.r_[rng.normal(0.2, 0.03, 500), rng.normal(0.55, 0.05, 400)]
    fit = fit_depth_gaussians(d, n_components=2, n_boot=0)
    x = np.linspace(-1.0, 2.0, 200_000)
    assert float(trapezoid(fit.predict(x), x)) == pytest.approx(1.0, rel=2e-3)


# ================================================================================================
# exponential + gamma: the family a bimodal dwell distribution with a peaked slow lobe needs
# ================================================================================================

def _bimodal_dwell(n_fast=200, n_slow=600, seed=5):
    """A fast escape plus a sharply peaked slow passage — the shape measured on the owner's
    apo-ferritin recordings (25% fast, 75% gamma with k ≈ 70)."""
    rng = np.random.default_rng(seed)
    return np.r_[rng.exponential(2e-3, n_fast), rng.gamma(70.0, 18.4e-3 / 70.0, n_slow)]


def test_exp_gamma_beats_both_families_it_was_added_to_replace():
    """The justification, asserted rather than asserted-in-a-docstring.

    An exponential mixture is monotone in linear density, so it can lean toward a peak but
    never make a sharp one; a single gamma forced to explain a fast population as well
    collapses to k ≈ 1, which IS an exponential. On the owner's -600 mV file the three came
    out at BIC -2062 / -2040 / -2682.
    """
    from nanoscope.analysis.stats import fit_dwell_exp_gamma

    t = _bimodal_dwell()
    t = t[t > 2e-5]
    eg = fit_dwell_exp_gamma(t, t_min=2e-5)
    ex = fit_dwell_exponential(t, t_min=2e-5, n_components=2, n_boot=0)
    ga = fit_dwell_peaked(t, model="gamma", t_min=2e-5)

    assert eg.bic < ex.bic - 50, f"exp+gamma BIC {eg.bic:.1f} vs 2-exp {ex.bic:.1f}"
    assert eg.bic < ga.bic - 50, f"exp+gamma BIC {eg.bic:.1f} vs gamma {ga.bic:.1f}"
    assert ga.params["k"] < 2.0, "the single gamma should collapse toward an exponential here"


def test_exp_gamma_recovers_the_parameters_it_was_generated_from():
    from nanoscope.analysis.stats import fit_dwell_exp_gamma

    t = _bimodal_dwell(n_fast=250, n_slow=750, seed=11)
    t = t[t > 2e-5]
    f = fit_dwell_exp_gamma(t, t_min=2e-5)
    assert f.params["w_exp"] == pytest.approx(0.25, abs=0.08)
    assert f.params["tau"] == pytest.approx(2e-3, rel=0.5)
    assert f.params["k"] > 20, f"k = {f.params['k']:.3g}; the slow lobe should be sharply peaked"
    assert f.params["k"] * f.params["theta"] == pytest.approx(18.4e-3, rel=0.15)
    assert f.meta["gamma_rel_sd"] < 0.3


def test_exp_gamma_density_is_a_density_and_survives_a_round_trip():
    """It must integrate to 1 over the observable window (identity tests cannot see a
    normalisation error — both sides share density_from), and it must persist."""
    from nanoscope.analysis._compat import trapezoid
    from nanoscope.analysis.stats import fit_dwell_exp_gamma, fit_from_json, fit_to_json

    t = _bimodal_dwell()
    t = t[t > 2e-5]
    f = fit_dwell_exp_gamma(t, t_min=2e-5)
    x = np.geomspace(2e-5, 2.0, 300_000)
    assert float(trapezoid(f.predict(x), x)) == pytest.approx(1.0, rel=2e-3)

    g = fit_from_json(fit_to_json(f))
    xs = np.geomspace(3e-5, 0.2, 400)
    assert np.array_equal(f.predict(xs), g.predict(xs))


def test_exp_gamma_says_so_when_the_peaked_component_was_not_needed():
    """A k that comes back ≈ 1 means the data did not want this family; reporting a
    two-mechanism interpretation off that would be the whole point of the family, wrongly."""
    from nanoscope.analysis.stats import fit_dwell_exp_gamma

    rng = np.random.default_rng(3)
    t = rng.exponential(5e-3, 800)          # one plain exponential population
    t = t[t > 2e-5]
    f = fit_dwell_exp_gamma(t, t_min=2e-5)
    if f.params["k"] <= 1.05:
        assert any("essentially an exponential" in w for w in f.meta["warnings"])


def test_exp_gamma_states_that_it_has_no_bootstrap_CIs():
    """The other fitters here produce CIs, so a blank column reads as "the bootstrap failed"
    rather than "this family does not offer one"."""
    from nanoscope.analysis.stats import fit_dwell_exp_gamma

    f = fit_dwell_exp_gamma(_bimodal_dwell()[_bimodal_dwell() > 2e-5], t_min=2e-5)
    assert f.ci == {}
    assert any("no bootstrap CIs" in w for w in f.meta["warnings"])


# ================================================================================================
# WHERE THE FIT MISSES: binned residuals, the reduced chi-square, and the CI band
# ================================================================================================


def _exp_fit(n=1200, tau=5e-4, t_min=1e-4, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    t = sample_truncated_exp(tau, t_min, n, rng)
    return fit_dwell_exponential(t, t_min=t_min, n_boot=n_boot), t


def test_the_chi_square_is_exactly_the_squares_of_the_residuals_that_are_drawn():
    """THE WHOLE POINT of returning both from one call: a reduced chi-square computed over
    a different pooling than the strip shows is a number nobody can check by eye."""
    from nanoscope.analysis.stats import binned_residuals

    fit, t = _exp_fit()
    edges = np.geomspace(t.min(), t.max(), 31)
    r = binned_residuals(fit, t, edges)

    assert r["chi2"] == pytest.approx(float(np.sum(r["residual"] ** 2)), rel=1e-12)
    assert r["residual"].size == r["n_bins"] == r["edges"].size - 1 == r["centers"].size
    assert r["reduced_chi2"] == pytest.approx(r["chi2"] / r["dof"])


def test_low_count_bins_are_POOLED_not_dropped():
    """Dropping them would discard real events and flatter the fit in exactly the tail
    where families differ most; the Cochran floor is what makes each term chi-squared."""
    from nanoscope.analysis.stats import MIN_EXPECTED_PER_BIN, binned_residuals

    fit, t = _exp_fit()
    # 200 log bins over a 1200-event sample guarantees a long tail below the floor.
    edges = np.geomspace(t.min(), t.max(), 201)
    r = binned_residuals(fit, t, edges)

    assert r["pooled"] and r["n_bins"] < r["n_bins_raw"] == 200
    below = r["expected"][:-1]      # the last group absorbs the leftover tail by design
    assert np.all(below >= MIN_EXPECTED_PER_BIN - 1e-9), below.min()
    # No event is lost by pooling: the groups still partition the same edges.
    assert r["observed"].sum() == pytest.approx(float(np.histogram(t, bins=edges)[0].sum()))
    assert r["edges"][0] == edges[0] and r["edges"][-1] == edges[-1]

    unpooled = binned_residuals(fit, t, edges, min_expected=0)
    assert unpooled["n_bins"] == 200, "min_expected=0 must leave the drawn bins alone"


def test_the_reduced_chi_square_is_about_one_on_a_correct_fit_and_large_on_a_wrong_one():
    """The only claim made for the number: a scale, in units of counting noise."""
    from nanoscope.analysis.stats import binned_residuals

    fit, t = _exp_fit(n=4000)
    edges = np.geomspace(t.min(), t.max(), 41)
    good = binned_residuals(fit, t, edges)
    assert 0.4 < good["reduced_chi2"] < 2.5, good["reduced_chi2"]

    # The SAME fitted curve against a sample it does not describe.
    rng = np.random.default_rng(7)
    wrong = 1e-4 + rng.gamma(6.0, 1.2e-4, 4000)
    w_edges = np.geomspace(wrong.min(), wrong.max(), 41)
    bad = binned_residuals(fit, wrong, w_edges)
    assert bad["reduced_chi2"] > 10 * good["reduced_chi2"], (bad["reduced_chi2"],
                                                             good["reduced_chi2"])


def test_the_note_always_says_the_dof_is_approximate_and_never_offers_a_p_value():
    """An unbinned truncated MLE did not minimise this chi-square, so the null sits
    between two chi-square distributions — the same objection goodness_of_fit states
    about its own KS p-value."""
    from nanoscope.analysis.stats import binned_residuals

    fit, t = _exp_fit()
    note = binned_residuals(fit, t, np.geomspace(t.min(), t.max(), 31))["note"]
    assert "APPROXIMATE" in note and "no p-value" in note
    assert "p =" not in note and "p-value =" not in note


def test_a_mixture_is_not_charged_for_a_weight_that_is_not_free():
    """k weights, k-1 of them free. Counting `len(params)` would spend a dof that was
    never estimated and quietly shrink every reduced chi-square of a mixture."""
    from nanoscope.analysis.stats import _n_free_params

    two = FitResult(params={"tau_1": 1e-3, "tau_2": 5e-3, "w_1": 0.4, "w_2": 0.6},
                    ci={}, loglik=0.0, aic=0.0, bic=0.0, n=100, model="x")
    one = FitResult(params={"tau_1": 1e-3}, ci={}, loglik=0.0, aic=0.0, bic=0.0,
                    n=100, model="x")
    assert _n_free_params(two) == 3 and _n_free_params(one) == 1


def test_too_few_pooled_bins_declines_the_number_instead_of_printing_arithmetic():
    """With no residual degrees of freedom a reduced chi-square is arithmetic, not
    evidence — exactly what r2_dof_caveat says of an R² through two points."""
    from nanoscope.analysis.stats import binned_residuals

    fit, t = _exp_fit()
    r = binned_residuals(fit, t, np.geomspace(t.min(), t.max(), 4), min_expected=1e9)
    assert r["n_bins"] == 1 and r["dof"] is None and r["reduced_chi2"] is None
    assert "unavailable" in r["note"] and "arithmetic, not evidence" in r["note"]
    assert r["residual"].size == 1, "the residuals are still drawable"


def test_binned_residuals_refuses_rather_than_inventing():
    from nanoscope.analysis.stats import binned_residuals

    fit, t = _exp_fit()
    with pytest.raises(ValueError, match="ascending"):
        binned_residuals(fit, t, [1e-4, 5e-4, 2e-4])
    with pytest.raises(ValueError, match=">= 2 finite"):
        binned_residuals(fit, [np.nan], np.geomspace(1e-4, 1e-2, 5))
    with pytest.raises(ValueError, match="fall inside"):
        binned_residuals(fit, t, np.linspace(10.0, 20.0, 5))


# -- the band ------------------------------------------------------------------------------------


def test_the_band_is_the_replicate_curves_not_the_corners_of_the_parameter_box():
    """A band drawn at (tau_lo) and (tau_hi) is a corner of the box the data rule OUT.
    The band must bracket the fitted curve and be strictly narrower than that corner
    construction nowhere claims to be."""
    from nanoscope.analysis.stats import ci_band

    fit, t = _exp_fit()
    x = np.geomspace(t.min(), t.max(), 60)
    band = ci_band(fit, x)

    assert band is not None and band["n_replicates"] >= 20
    curve = fit.predict(x)
    assert np.all(band["lo"] <= curve + 1e-12) and np.all(band["hi"] >= curve - 1e-12)
    assert np.all(band["hi"] >= band["lo"])
    assert "POINTWISE" in band["note"]


def test_the_band_is_declined_when_the_replicates_are_not_there():
    """None, never an exception and never a fabricated band: a restored fit, a point fit
    and a family with no bootstrap must all draw nothing rather than something wrong."""
    from nanoscope.analysis.stats import (ci_band, fit_from_json, fit_to_json,
                                          point_fits_only)

    fit, t = _exp_fit()
    x = np.geomspace(t.min(), t.max(), 40)
    assert ci_band(fit_from_json(fit_to_json(fit)), x) is None, "a reopened project"
    with point_fits_only():
        assert ci_band(fit_dwell_exponential(t, t_min=1e-4), x) is None


def test_the_replicates_stay_out_of_the_saved_project():
    """megabytes of resampled parameter noise per project that nothing reads back."""
    from nanoscope.analysis.stats import BOOT_KEY, BOOT_PARAMS_KEY, fit_to_json

    fit, _ = _exp_fit()
    assert fit.meta[BOOT_KEY].shape == (fit.meta["n_boot_ok"], 1)
    blob = fit_to_json(fit)
    assert BOOT_KEY not in blob["meta"] and BOOT_PARAMS_KEY not in blob["meta"]
    assert blob["ci"], "the INTERVALS still persist; only the band's raw material does not"


def test_a_deferred_bootstrap_can_still_produce_a_band():
    """Otherwise the deferred path — the one the GUI actually uses — is the single route
    that can never draw one."""
    from nanoscope.analysis.stats import apply_ci, ci_band, deferred_ci

    rng = np.random.default_rng(4)
    t = 1e-4 + rng.gamma(2.0, 3e-4, 900)
    with deferred_ci() as jobs:
        fit = fit_dwell_peaked(t, model="gamma", t_min=1e-4, n_boot=40)
    assert fit.ci == {} and jobs
    apply_ci(fit, jobs[0]())
    x = np.geomspace(t.min(), t.max(), 40)
    assert fit.ci and ci_band(fit, x) is not None


def test_the_depth_mixture_retains_its_replicates_too():
    from nanoscope.analysis.stats import BOOT_PARAMS_KEY, ci_band

    rng = np.random.default_rng(11)
    d = np.concatenate([rng.normal(0.2, 0.02, 400), rng.normal(0.45, 0.03, 400)])
    fit = fit_depth_gaussians(d, n_components=2, n_boot=60)
    assert list(fit.params) == fit.meta[BOOT_PARAMS_KEY]
    assert ci_band(fit, np.linspace(0.1, 0.6, 50)) is not None
