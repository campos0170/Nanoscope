"""FIT HONESTY: every fit says what it measured, in what unit, and whether it is WRONG.

Area B of the apo-ferritin end-to-end report
(``nanoscope-reports/apo-run-2026-08-19/SUMMARY.md`` §7). Every number asserted below was
measured on the REAL merged event sets from that run -- the three apo-ferritin recordings
at -500 / -600 / -700 mV, 500 pM in 1x PBS through a 22 nm pore, detected at the app's
defaults (min_dwell_s 20 us, fs 50 kHz) and merged at a 10 ms gap. The CSVs live beside
the report; :data:`APO` loads them, and the tests skip rather than lie if they are gone.

Standing facts these tests must respect (report, front matter):

* **The gain is uncalibrated.** Open pore reads -27 to -52 pA where a 22 nm pore should be
  ~1000x larger. Every pA here is an instrument number; ratios and BIC differences are
  valid, absolute picoamps are not physical. Nothing below reports a pA as a measurement
  -- they are fixtures for a labelling and identifiability contract.
* **frac_depth is 1.81-2.57 on these recordings and that is REAL**: the current MAGNITUDE
  grows during an event. It is not a blockade fraction. Which is exactly why a mixture fit
  to ``depth_pA`` must never come back captioned "dI/I0" (High-4).

The defects, and the hunk of ``revert_areaB`` that puts each one back:

    High-3   no goodness of fit, no cross-family comparison  no-gof-no-comparison
    High-4   the depth mixture was hardcoded to "dI/I0"      depth-quantity-label
                                                             fitresult-unit-of
    Medium-11 components reported without identifiability     identifiability-notes
    Medium-12 EM local optima, BIC non-monotonic in k         em-split-ladder
    Medium-18 the peaked families carried no error bars       peaked-no-ci
    Low-19   inverse-gaussian returned BIC = +inf             truncated-nll-density-space
    Low-20   degenerate exponential mixtures went unflagged   degenerate-pairs

Run this file with ``-p revert_areaB`` and 14 of the 22 tests must fail. The eight that
pass BOTH ways are deliberately not pins: five are SILENCE controls (a flag that can only
fire says nothing — ``..._is_not_flagged``, ``..._draws_no_identifiability_warnings``,
``..._is_still_a_local_optimum_says_so``, ``..._only_ever_improves_the_likelihood``,
``test_frac_depth_keeps_the_dimensionless_reading_by_default``), one is the cost contract
(``..._obeys_point_fits_only``), one is a guard tested by injection
(``..._refuses_instead_of_returning_inf``, whose real case is pinned by
``..._no_longer_returns_an_infinite_bic``), and one is the fixture guard.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import stats

# ================================================================================================
# the real recordings
# ================================================================================================

APO = Path("/Users/abrahamcampos/Desktop/nanoscope-reports/apo-run-2026-08-19/verification")

#: Detector settings the run used on all three files (report §1): app defaults, 50 kHz.
APO_PROV = {"min_dwell_s": 20e-6, "fs": 50000.0}
APO_TMIN = 20e-6


def _apo(run: str) -> pd.DataFrame:
    path = APO / run / "events_merge.csv"
    if not path.exists():
        pytest.skip(f"apo-ferritin merged event set not available: {path}")
    return pd.read_csv(path)


def _accepted(df: pd.DataFrame) -> pd.DataFrame:
    """The sample every fit in the app sees: accepted, filter-limited excluded."""
    return df[df["accepted"].astype(bool) & ~df["filter_limited"].astype(bool)]


@pytest.fixture(scope="module")
def apo_700_dwell() -> np.ndarray:
    """986 merged dwell times, -700 mV. Median 19.36 ms, mean 19.829 ms, min 0.10 ms."""
    return _accepted(_apo("apo_-700mV"))["dwell_s"].to_numpy(float)


@pytest.fixture(scope="module")
def apo_500_depth() -> np.ndarray:
    """863 merged depths in UNCALIBRATED pA, -500 mV. Median 62.87, sd 17.13."""
    return np.abs(_accepted(_apo("apo_-500mV"))["depth_pA"].to_numpy(float))


@pytest.fixture(scope="module")
def apo_700_depth() -> np.ndarray:
    """986 merged depths in UNCALIBRATED pA, -700 mV."""
    return np.abs(_accepted(_apo("apo_-700mV"))["depth_pA"].to_numpy(float))


# ================================================================================================
# High-4 + Low-28 -- THE FIT SAYS WHAT IT MEASURED, AND IN WHAT
# ================================================================================================

def test_the_depth_mixture_is_labelled_with_the_column_it_was_actually_given(apo_700_depth):
    """MEASURED (report High-4): fitting ``depth_pA`` returned a FitResult whose model
    read ``Gaussian mixture on dI/I0 (k=3, EM)`` and whose params were 85.46, 99.69,
    105.8 -- picoamps under a dimensionless caption, on the one dataset where dI/I0 > 1
    is the fact that must not be confused with anything else.

    Reverted by: ``depth-quantity-label``.
    """
    fit = stats.fit_depth_gaussians(apo_700_depth, n_components=3, n_boot=0,
                                    quantity="depth_pA", unit="pA")

    assert "depth_pA" in fit.model, fit.model
    assert "dI/I0" not in fit.model, fit.model
    assert fit.meta["quantity"] == "depth_pA"
    # The values that used to be printed bare, now each carrying its unit.
    mus = [fit.params[f"mu_{j}"] for j in (1, 2, 3)]
    assert mus == pytest.approx([85.46, 99.69, 105.78], abs=0.02), mus
    for j in (1, 2, 3):
        assert fit.unit_of(f"mu_{j}") == "pA"
        assert fit.unit_of(f"sigma_{j}") == "pA"
        assert fit.unit_of(f"w_{j}") == "", "a mixing weight is a probability, not a current"


def test_frac_depth_keeps_the_dimensionless_reading_by_default(apo_700_depth):
    """The default is not "no unit", it is the dimensionless ratio the function was
    written for -- so a caller that says nothing gets the pre-existing behaviour."""
    fit = stats.fit_depth_gaussians(apo_700_depth / 100.0, n_components=1, n_boot=0)
    assert fit.model == "Gaussian mixture on dI/I0 (k=1, EM)"
    assert fit.unit_of("mu_1") == ""


def test_every_fitted_parameter_carries_its_own_unit(apo_700_dwell):
    """A fit is not all one unit: gamma returns a dimensionless shape beside a scale in
    seconds. Captioning the whole FitResult with one unit is guaranteed to mislabel
    something, which is how picoamps ended up under "dI/I0" (High-4).

    Reverted by: ``fitresult-unit-of``.
    """
    exp = stats.fit_dwell_exponential(apo_700_dwell, t_min=APO_TMIN, n_components=2,
                                      n_boot=0)
    assert exp.unit_of("tau_1") == "s" and exp.unit_of("tau_2") == "s"
    assert exp.unit_of("w_1") == "" and exp.unit_of("w_2") == ""

    gam = stats.fit_dwell_peaked(apo_700_dwell, model="gamma", t_min=APO_TMIN, n_boot=0)
    assert gam.unit_of("k") == "", "the gamma shape is dimensionless"
    assert gam.unit_of("theta") == "s"

    ig = stats.fit_dwell_peaked(apo_700_dwell, model="inverse-gaussian", t_min=APO_TMIN,
                                n_boot=0)
    assert ig.unit_of("mu") == "s" and ig.unit_of("lambda") == "s"

    eg = stats.fit_dwell_exp_gamma(apo_700_dwell, t_min=APO_TMIN)
    assert eg.unit_of("tau") == "s" and eg.unit_of("theta") == "s"
    assert eg.unit_of("w_exp") == "" and eg.unit_of("k") == ""

    # And a parameter nobody declared answers "" instead of raising -- a caption is not
    # worth an exception.
    assert exp.unit_of("not_a_param") == ""


def test_the_summary_prints_the_unit_beside_the_estimate(apo_700_dwell):
    gam = stats.fit_dwell_peaked(apo_700_dwell, model="gamma", t_min=APO_TMIN, n_boot=0)
    lines = {ln.split()[0]: ln for ln in gam.summary().splitlines() if ln.startswith("  ")}
    assert lines["theta"].rstrip().endswith(" s"), lines["theta"]
    assert not lines["k"].rstrip().endswith(" s"), lines["k"]


# ================================================================================================
# High-3 -- A FIT THAT DOES NOT DESCRIBE THE DATA MUST BE ABLE TO SAY SO
# ================================================================================================

def test_the_exponential_is_rejected_on_every_one_of_the_three_recordings():
    """MEASURED (report High-3): the panel printed "tau = 0.01981 (n = 986)" with a
    bootstrap CI of [0.019589, 0.020032] and NO goodness-of-fit statistic, while the drawn
    curve visibly missed the histogram. These dwell distributions are a narrow ~16-28 ms
    mode with a hard ceiling; tau is arithmetically ``mean(t) - t_min`` (observed mean
    19.829 ms against tau 19.809 ms at -700 mV).

    Reverted by: ``no-gof-no-comparison`` (the function did not exist).
    """
    expected = {"apo_-500mV": 0.4577, "apo_-600mV": 0.4478, "apo_-700mV": 0.6395}
    for run, d_expected in expected.items():
        t = _accepted(_apo(run))["dwell_s"].to_numpy(float)
        fit = stats.fit_dwell_exponential(t, t_min=APO_TMIN, n_boot=0)
        gof = stats.goodness_of_fit(fit, t)
        assert gof["ks"] == pytest.approx(d_expected, abs=5e-4), (run, gof["ks"])
        assert gof["ks"] > gof["ks_critical_001"] * 4, run
        assert gof["describes_data"] is False, run
        assert "DOES NOT DESCRIBE" in gof["note"]


def test_a_sample_that_really_is_exponential_passes_and_agrees_with_scipy():
    """The other half of the contract: the verdict has to be capable of saying yes, and
    its statistic has to be the KS statistic and not something merely KS-shaped.

    Cross-checked against ``scipy.stats.kstest`` with the analytic left-truncated
    exponential CDF: 0.015456 here, 0.015518 there on the same 4000 draws.
    """
    from scipy import stats as sps

    rng = np.random.default_rng(7)
    t_min, tau = 5e-5, 2e-3
    x = t_min + rng.exponential(tau, 4000)
    fit = stats.fit_dwell_exponential(x, t_min=t_min, n_boot=0)
    gof = stats.goodness_of_fit(fit, x)

    assert gof["describes_data"] is True
    assert "DOES NOT DESCRIBE" not in gof["note"]
    analytic = sps.kstest(
        x, lambda v: 1.0 - np.exp(-(np.asarray(v) - t_min) / fit.params["tau_1"])).statistic
    assert gof["ks"] == pytest.approx(analytic, abs=2e-3), (gof["ks"], analytic)


def test_the_cross_family_comparison_ranks_every_model_the_menu_offers(apo_700_dwell):
    """MEASURED (report §3 and High-3): on -700 mV the exponential is the WORST model in
    the app's own menu. gamma beats it by 2111 BIC and exponential+gamma by 2729. The app
    could already fit all four families and never compared them.

    Reverted by: ``no-gof-no-comparison``.
    """
    rows = stats.compare_dwell_models(apo_700_dwell, t_min=APO_TMIN)
    by_label = {r["label"]: r for r in rows}

    assert rows[0]["label"] == "exponential + gamma (bimodal)"
    assert [r["label"] for r in rows] == sorted(
        by_label, key=lambda k: by_label[k]["bic"]), "rows must be BIC-sorted"

    assert by_label["exponential + gamma (bimodal)"]["bic"] == pytest.approx(-8483.15, abs=0.5)
    assert by_label["gamma (peaked)"]["bic"] == pytest.approx(-7865.21, abs=0.5)
    assert by_label["exponential"]["bic"] == pytest.approx(-5754.50, abs=0.5)

    delta = by_label["exponential"]["bic"] - by_label["exponential + gamma (bimodal)"]["bic"]
    assert delta == pytest.approx(2728.66, abs=1.0), delta
    assert all(r["n"] == 986 for r in rows), "every family must see the SAME sample"


def test_a_family_that_refuses_stays_in_the_comparison_with_its_reason():
    """Dropping a failed family silently is how an infinite BIC survived a whole run
    (Low-19). Twenty identical values: the exponential can be fit, nothing peaked can."""
    x = np.full(40, 1e-3)
    x[::2] = 1.1e-3                       # not all identical, but pathologically discrete
    rows = stats.compare_dwell_models(x, t_min=1e-4)
    assert len(rows) == len(stats.DWELL_MODEL_LABELS)
    for r in rows:
        assert (r["bic"] is None) == (r["error"] is not None)
    assert all(r["bic"] is None or np.isfinite(r["bic"]) for r in rows)


def test_comparing_an_unknown_family_is_an_error_not_a_shrug(apo_700_dwell):
    with pytest.raises(ValueError, match="unknown dwell family"):
        stats.compare_dwell_models(apo_700_dwell, t_min=APO_TMIN, families=["gamma"])


# ================================================================================================
# Low-19 -- AN INFINITE BIC IS NOT A FIT
# ================================================================================================

def test_the_inverse_gaussian_no_longer_returns_an_infinite_bic(apo_700_dwell):
    """MEASURED (report Low-19): ``inverse-gaussian (first passage)`` returned
    **BIC = +inf** on -700 mV (n = 986, mu 19.83 ms, lambda 0.636) -- silently unusable,
    still offered in the menu and still drawn.

    The cause was arithmetic, not statistics. The likelihood summed ``log(pdf(t))`` and
    refused the whole sample the moment any density underflowed; the shortest merged event
    is 0.10 ms, which drives the inverse-Gaussian exponent to about -7.6e3, so ``exp()``
    returned exactly 0.0. Summing LOG densities gives that event its true finite penalty.

    The new value is independently confirmed against ``scipy.stats.invgauss``:
    loglik 2982.267 at mu = 19.8294 ms, lambda = 0.052148 s, i.e. BIC -5950.7468 -- and a
    grid scan of the same likelihood finds its maximum at the same place. The OLD
    parameters were not even a maximum: they score loglik -1302.19 (BIC +2618).

    Reverted by: ``truncated-nll-density-space``.
    """
    from scipy import stats as sps

    fit = stats.fit_dwell_peaked(apo_700_dwell, model="inverse-gaussian",
                                 t_min=APO_TMIN, n_boot=0)
    assert np.isfinite(fit.bic), fit.bic
    assert fit.bic == pytest.approx(-5950.75, abs=0.5)
    assert fit.params["mu"] == pytest.approx(0.0198294, rel=1e-4)
    assert fit.params["lambda"] == pytest.approx(0.052148, rel=1e-3)

    # Independent re-derivation of that likelihood, through scipy's own inverse-Gaussian.
    mu, lam = fit.params["mu"], fit.params["lambda"]
    d = sps.invgauss(mu / lam, scale=lam)
    ll = float(np.sum(d.logpdf(apo_700_dwell))
               - apo_700_dwell.size * np.log(1.0 - d.cdf(APO_TMIN)))
    assert ll == pytest.approx(fit.loglik, abs=0.05), (ll, fit.loglik)
    assert 2 * np.log(986) - 2 * ll == pytest.approx(fit.bic, abs=0.1)


def test_a_peaked_fit_with_no_finite_optimum_refuses_instead_of_returning_inf():
    """The belt to the log-space braces: if a family still cannot be maximised, it raises
    rather than handing back a number no comparison can use."""
    from nanoscope.analysis import stats as st

    real = st._truncated_nll
    try:
        st._truncated_nll = lambda *a, **k: np.inf
        with pytest.raises(RuntimeError, match="not finite"):
            st.fit_dwell_peaked(np.linspace(1e-3, 5e-3, 50), model="gamma", t_min=1e-4)
    finally:
        st._truncated_nll = real


# ================================================================================================
# Medium-18 -- THE MODELS THAT ACTUALLY FIT MUST CARRY ERROR BARS TOO
# ================================================================================================

def test_the_peaked_families_report_confidence_intervals(apo_700_dwell):
    """MEASURED (report Medium-18): only the exponential family returned CIs.
    ``gamma (peaked)`` and ``inverse-gaussian (first passage)`` -- the two that can have a
    mode, i.e. the two that describe these data -- returned ``ci = {}`` and printed no
    uncertainty at all. The fits with error bars were the ones the data reject.

    Reverted by: ``peaked-no-ci``.
    """
    for model, params in (("gamma", ("k", "theta")),
                          ("inverse-gaussian", ("mu", "lambda"))):
        fit = stats.fit_dwell_peaked(apo_700_dwell, model=model, t_min=APO_TMIN,
                                     n_boot=60, seed=0)
        assert set(fit.ci) == set(params), (model, fit.ci)
        assert fit.meta["n_boot_ok"] >= 20
        for p in params:
            lo, hi = fit.ci[p]
            assert np.isfinite(lo) and np.isfinite(hi)
            assert lo < hi
            assert lo <= fit.params[p] <= hi, (model, p, fit.ci[p], fit.params[p])


def test_the_peaked_bootstrap_obeys_point_fits_only(apo_700_dwell):
    """The rival scans read ``bic`` and nothing else; a CI they discard must not be paid
    for. Same contract as every other bootstrap in the module, and the reason the
    cross-family comparison costs ~200 ms rather than minutes."""
    with stats.point_fits_only():
        fit = stats.fit_dwell_peaked(apo_700_dwell, model="gamma", t_min=APO_TMIN)
    assert fit.ci == {}
    assert fit.meta["n_boot_ok"] == 0
    # ...and the point estimate is bit-identical either way.
    full = stats.fit_dwell_peaked(apo_700_dwell, model="gamma", t_min=APO_TMIN, n_boot=20)
    assert fit.params == full.params
    assert fit.bic == full.bic


# ================================================================================================
# Low-20 -- A MIXTURE THAT FOUND ONE COMPONENT TWICE MUST SAY SO
# ================================================================================================

def test_the_degenerate_exponential_mixture_is_flagged(apo_700_dwell):
    """MEASURED (report Low-20): -700 mV k=2 returned tau_1 = tau_2 = 19.809 ms --
    identical to 8 significant figures -- with weights 0.88 / 0.12, printed as two time
    constants with nothing saying they are one.

    Reverted by: ``degenerate-pairs``.
    """
    fit = stats.fit_dwell_exponential(apo_700_dwell, t_min=APO_TMIN, n_components=2,
                                      n_boot=0)
    t1, t2 = fit.params["tau_1"], fit.params["tau_2"]
    assert abs(t2 - t1) / t1 < 1e-6, "the degeneracy itself must still be reproduced"

    assert fit.meta["degenerate"] is True
    assert fit.meta["degenerate_pairs"] == [[1, 2]]
    warned = [w for w in fit.meta["warnings"] if "DEGENERATE MIXTURE" in w]
    assert warned, fit.meta["warnings"]
    assert "tau_1" in warned[0] and "tau_2" in warned[0]
    assert "k=1 solution" in warned[0]


def test_a_genuinely_separated_mixture_is_not_flagged(apo_700_dwell):
    """The flag has to be able to stay quiet, or it says nothing. -500 mV k=2 separates
    cleanly: tau = 23.4 ms and 180.1 ms."""
    t = _accepted(_apo("apo_-500mV"))["dwell_s"].to_numpy(float)
    fit = stats.fit_dwell_exponential(t, t_min=APO_TMIN, n_components=2, n_boot=0)
    assert fit.params["tau_2"] / fit.params["tau_1"] > 5
    assert fit.meta["degenerate"] is False
    assert not [w for w in fit.meta["warnings"] if "DEGENERATE" in w]


# ================================================================================================
# Medium-12 -- EM MUST NOT REPORT A LOCAL OPTIMUM AS THE ANSWER
# ================================================================================================

def test_bic_is_monotone_in_k_on_the_500mV_depths(apo_500_depth):
    """MEASURED (report Medium-12): -500 mV ``depth_pA`` gave BIC 7365.41 (k=1) ->
    **7370.71 (k=2)** -> 7198.43 (k=3) -> 7170.43 (k=4). Two Gaussians scoring WORSE than
    one is impossible for a true maximum -- a k-component mixture contains every
    (k-1)-component one -- so the k=2 answer was a local optimum, and its means (65.2 /
    81.6 pA) sat away from the histogram's own peaks.

    A deterministic split-the-widest-component start reaches loglik -3615.90 against the
    random starts' -3668.45: **52.5 higher**, which is proof rather than preference. BIC
    becomes 7265.59 and the ladder is monotone.

    Reverted by: ``em-split-ladder``.
    """
    fits = {k: stats.fit_depth_gaussians(apo_500_depth, n_components=k, n_boot=0,
                                         quantity="depth_pA", unit="pA")
            for k in (1, 2, 3, 4)}
    lls = {k: f.loglik for k, f in fits.items()}
    bics = {k: f.bic for k, f in fits.items()}

    # The nested-model inequality, which is what "local optimum" violates.
    for k in (2, 3, 4):
        assert lls[k] >= lls[k - 1] - 1e-9, (k, lls)

    assert bics[1] == pytest.approx(7365.41, abs=0.05)
    assert bics[2] == pytest.approx(7265.59, abs=0.05), "the k=2 local optimum is back"
    assert bics[3] == pytest.approx(7198.43, abs=0.05)
    assert bics[4] == pytest.approx(7170.43, abs=0.05)
    assert bics[2] < bics[1], "two Gaussians cannot fit worse than one"
    assert fits[2].meta["split_ladder_won"] is True
    assert fits[2].params["mu_1"] == pytest.approx(55.12, abs=0.05)


def test_a_fit_that_is_still_a_local_optimum_says_so():
    """The ladder is a better start, not a proof. If EM still lands below the k=1 fit the
    result is reported as where the optimiser stopped, not as the model."""
    rng = np.random.default_rng(1)
    x = np.abs(rng.normal(50, 8, 400))
    fit = stats.fit_depth_gaussians(x, n_components=2, n_boot=0)
    assert fit.meta["local_optimum"] is False
    assert fit.loglik >= fit.meta["loglik_k1"] - 1e-9
    assert not [w for w in fit.meta["warnings"] if "LOCAL OPTIMUM" in w]


def test_the_split_ladder_only_ever_improves_the_likelihood(apo_700_depth):
    """It is added as an EXTRA start after the ten random ones, on a private generator, so
    it cannot make any existing fit worse -- and cannot move the numbers of a fit the
    random starts already won."""
    for k in (2, 3, 4):
        fit = stats.fit_depth_gaussians(apo_700_depth, n_components=k, n_boot=0)
        assert np.isfinite(fit.loglik)
        assert fit.loglik >= fit.meta["loglik_k1"] - 1e-9, k


# ================================================================================================
# Medium-11 -- COMPONENTS THAT WERE NOT RESOLVED MUST NOT PRINT AS POPULATIONS
# ================================================================================================

def test_the_700mV_three_component_depth_fit_reports_every_way_it_failed(apo_700_depth):
    """MEASURED (report Medium-11): -700 mV ``depth_pA`` at k=3 returns

        mu_1 = 85.46 pA with a 95% CI of [17.6, 87.6]   -- five times sigma_1 = 7.8
        w_2  = 0.027 (2.7%) with sigma_2 = 41 pA and a weight CI of [0.014, 0.671]

    and printed "mu = 85.46, 99.69, 105.8" with nothing beside it. Anyone reading that
    list would report three populations that are not there.

    Reverted by: ``identifiability-notes``.
    """
    fit = stats.fit_depth_gaussians(apo_700_depth, n_components=3, n_boot=400, seed=0,
                                    quantity="depth_pA", unit="pA")
    notes = "\n".join(fit.meta["warnings"])

    # The measurements the report made, reproduced.
    assert fit.ci["mu_1"][0] == pytest.approx(17.64, abs=0.5)
    assert fit.ci["mu_1"][1] == pytest.approx(87.59, abs=0.5)
    assert fit.params["w_2"] == pytest.approx(0.0269, abs=0.002)
    assert fit.ci["w_2"] == pytest.approx((0.0136, 0.6708), abs=0.02)

    # ...and each of the three ways this fit is unidentified, said out loud.
    assert "UNIDENTIFIED COMPONENT(S)" in notes, notes
    assert "mu_1" in notes and "2*sigma_1" in notes
    assert "UNRESOLVED WEIGHT(S)" in notes, notes
    assert "w_2" in notes
    assert "COMPONENTS NOT SEPARATED" in notes, notes
    assert "pA" in notes, "the numbers in a warning need units as much as the params do"


def test_a_cleanly_separated_mixture_draws_no_identifiability_warnings():
    """The flags have to be capable of silence. Two well-separated, well-populated
    Gaussians: nothing to complain about."""
    rng = np.random.default_rng(11)
    x = np.r_[rng.normal(20, 2, 600), rng.normal(60, 3, 600)]
    fit = stats.fit_depth_gaussians(np.abs(x), n_components=2, n_boot=200, seed=0,
                                    quantity="depth_pA", unit="pA")
    notes = "\n".join(fit.meta["warnings"])
    assert "UNIDENTIFIED" not in notes, notes
    assert "UNRESOLVED WEIGHT" not in notes, notes
    assert "NOT SEPARATED" not in notes, notes


def test_a_component_carrying_fewer_events_than_the_module_would_fit_is_called_out():
    """``_clean_1d`` refuses a whole SAMPLE below 20 events. A component supported by
    fewer than that is being estimated from a sample the module would not accept."""
    rng = np.random.default_rng(5)
    x = np.r_[rng.normal(20, 2, 800), rng.normal(90, 1, 8)]
    fit = stats.fit_depth_gaussians(np.abs(x), n_components=2, n_boot=0)
    notes = "\n".join(fit.meta["warnings"])
    assert "fewer than 20 events" in notes, notes


# ================================================================================================
# the sample under test is the sample the app fits
# ================================================================================================

def test_the_fixtures_match_the_report(apo_700_dwell, apo_500_depth):
    """A guard on the fixtures themselves: if these CSVs are ever swapped, every number
    above becomes a statement about different data and this test says so first."""
    assert apo_700_dwell.size == 986
    assert np.median(apo_700_dwell) == pytest.approx(0.01936, abs=1e-6)
    assert apo_700_dwell.mean() == pytest.approx(0.0198294, abs=1e-6)
    assert apo_500_depth.size == 863
    assert np.median(apo_500_depth) == pytest.approx(62.87, abs=0.02)


# ================================================================================================
# High-3, THE SECOND SURFACE — the verdict has to live in the FIT, not in one panel
# ================================================================================================
# The first pass put `goodness_of_fit` in `stats`, correct and family-agnostic, and then
# called it from exactly ONE place: `gui/plots_panel.py`. The app has two fit surfaces.
# The other — the Inspector's Fits box, `AnalysisPanel._on_fit_requested` ->
# `modules.fit_dwell` -> `StatsPanel.set_fit_text(FitResult.summary())` — went on printing
# the report's own quoted read-out on the same -700 mV data,
#
#     tau_1  0.0198094  [0.01959, 0.02003]  s
#
# with no statistic, no rival and no warning, while the OTHER window was calling the same
# fit unusable. Medium-11 and Low-20 avoided this by putting their notes in the fitter;
# High-3's verdict went into the panel. `_stamp_gof` moves it to the seam.

def test_every_fit_this_module_returns_carries_its_own_verdict(apo_700_dwell):
    """All four families, one sample. A fit that cannot say whether it is wrong is the
    defect, whichever family it belongs to."""
    fits = {
        "exponential": stats.fit_dwell_exponential(apo_700_dwell, n_components=1,
                                                   t_min=APO_TMIN),
        "gamma": stats.fit_dwell_peaked(apo_700_dwell, model="gamma", t_min=APO_TMIN),
        "inverse-gaussian": stats.fit_dwell_peaked(apo_700_dwell, model="inverse-gaussian",
                                                   t_min=APO_TMIN),
        "exp+gamma": stats.fit_dwell_exp_gamma(apo_700_dwell, t_min=APO_TMIN),
    }
    for name, fit in fits.items():
        gof = fit.meta.get("gof")
        assert isinstance(gof, dict), f"{name} carries no goodness-of-fit verdict"
        assert set(gof) >= {"ks", "ks_critical_001", "describes_data", "note"}, name
        assert gof["n"] == fit.n, name


def test_the_amplitude_mixture_carries_one_too(apo_500_depth):
    """Not a dwell-only rule: the depth mixture is a drawn curve over a histogram too."""
    fit = stats.fit_depth_gaussians(apo_500_depth, n_components=2)
    assert isinstance(fit.meta.get("gof"), dict)


def test_the_verdict_is_printed_by_the_summary_the_inspector_shows(apo_700_dwell):
    """THE ORIGINAL READ-OUT, verbatim, plus what was missing from it.

    `FitResult.summary()` is the whole of what the Inspector's Fits box displays. On the
    -700 mV merged set it printed tau_1 = 0.0198094 s with a CI of [0.01959, 0.02003] and
    nothing else — the report's quoted lines. It must still print exactly those numbers
    (this batch may not move a fit), and it must now also print the verdict.
    """
    fit = stats.fit_dwell_exponential(apo_700_dwell, n_components=1, t_min=APO_TMIN)
    text = fit.summary()

    assert "0.0198094" in text, text                    # tau has NOT moved
    assert "goodness of fit:" in text, text
    assert "KS D = 0.6395" in text, text
    assert "DOES NOT DESCRIBE THESE DATA" in text, text


def test_a_fit_that_does_describe_its_data_says_that_instead():
    """The verdict discriminates. A blanket "this is wrong" on every fit would be worth
    nothing; drawn from the model it fits, the same code path passes."""
    rng = np.random.default_rng(11)
    t = rng.exponential(0.02, 4000)
    t = t[t > APO_TMIN]
    fit = stats.fit_dwell_exponential(t, n_components=1, t_min=APO_TMIN)
    gof = fit.meta["gof"]
    assert gof["describes_data"] is True, gof["note"]
    assert "consistent with these data" in fit.summary()


def test_a_verdict_that_cannot_be_computed_does_not_break_the_fit():
    """`_stamp_gof` never raises into a fit.

    A FitResult carrying no density — every one from `fit_from_json`, i.e. every fit a
    reopened project restores — cannot be KS-tested. It must come back unstamped and
    intact, not as an exception out of a fitter.
    """
    bare = stats.FitResult(params={"tau_1": 0.02}, ci={}, loglik=1.0, aic=0.0, bic=0.0,
                           n=200, model="carries no density", meta={})
    out = stats._stamp_gof(bare, np.linspace(0.01, 0.05, 200))
    assert out is bare
    assert "gof" not in out.meta
    assert "goodness of fit:" not in out.summary()


def test_goodness_of_fit_is_reachable_from_the_fitter_not_only_from_a_panel():
    """The structural half, and the actual lesson. Grepping the package for callers of
    `goodness_of_fit` used to return one GUI module. The seam both fit surfaces cross is
    `stats`, so that is where the call belongs."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(stats))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_stamp_gof" in called, "no fitter routes its result through _stamp_gof"

    # ...and every public fitter does, not just the one that was measured.
    src = inspect.getsource(stats)
    assert src.count("_stamp_gof(FitResult(") == 4, (
        "a FitResult is being returned without a verdict; there are four fitters")
