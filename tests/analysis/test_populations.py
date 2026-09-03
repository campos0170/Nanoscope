"""Tests for analysis.populations: the pure clustering / PCA / fit-passthrough core.

These pin the invariants the interactive Plots window leans on: GMM recovers
well-separated blobs and auto-selects k by BIC, k-means recovers them too, PCA
concentrates variance from a correlated cloud onto PC1, ``feature_matrix`` drops
non-finite rows while keeping the index aligned, and every guard raises a clear
``ValueError``. Everything is deterministic (fixed ``random_state`` / seeded RNG).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import populations
from nanoscope.analysis.populations import (
    DROPPED_LABEL,
    ClusterResult,
    LinearFit,
    PCAResult,
    cluster,
    feature_matrix,
    fit_exponential,
    fit_gaussian_mixture,
    line_points,
    linear_fit,
    pca,
)
from nanoscope.analysis.stats import FitResult


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _two_blobs(n: int = 150, sep: float = 12.0, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Two well-separated 2-D Gaussian blobs; return the frame and true labels."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(0.0, 0.0), scale=0.6, size=(n, 2))
    b = rng.normal(loc=(sep, sep), scale=0.6, size=(n, 2))
    X = np.vstack([a, b])
    truth = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    order = rng.permutation(X.shape[0])
    X, truth = X[order], truth[order]
    df = pd.DataFrame({"amp": X[:, 0], "dwell": X[:, 1]})
    return df, truth


def _agreement(pred: np.ndarray, truth: np.ndarray) -> float:
    """Best label-permutation agreement for a 2-cluster labelling."""
    same = np.mean(pred == truth)
    return max(same, 1.0 - same)


# ---------------------------------------------------------------------------
# feature_matrix
# ---------------------------------------------------------------------------


def test_feature_matrix_drops_nonfinite_and_realigns() -> None:
    df = pd.DataFrame(
        {"a": [1.0, 2.0, np.nan, 4.0, np.inf], "b": [10.0, 20.0, 30.0, 40.0, 50.0]},
        index=[100, 101, 102, 103, 104],
    )
    X, kept = feature_matrix(df, ["a", "b"])
    assert X.shape == (3, 2)
    assert list(kept) == [100, 101, 103]  # rows with NaN / inf gone, index preserved
    assert X.dtype == np.float64
    np.testing.assert_array_equal(X[:, 0], [1.0, 2.0, 4.0])


def test_feature_matrix_coerces_non_numeric_to_dropped() -> None:
    df = pd.DataFrame({"a": ["1.0", "oops", "3.0"], "b": [1.0, 2.0, 3.0]})
    X, kept = feature_matrix(df, ["a", "b"])
    assert list(kept) == [0, 2]
    np.testing.assert_array_equal(X[:, 0], [1.0, 3.0])


def test_feature_matrix_rejects_empty_columns() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least one column"):
        feature_matrix(df, [])


def test_feature_matrix_rejects_unknown_column() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not in the DataFrame"):
        feature_matrix(df, ["a", "nope"])


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def test_gmm_autoselects_k2_and_recovers_blobs() -> None:
    df, truth = _two_blobs()
    res = cluster(df, ["amp", "dwell"], method="gmm", k=None, k_max=6)
    assert isinstance(res, ClusterResult)
    assert res.method == "gmm"
    assert res.chosen_k == 2
    assert res.k == 2
    assert res.bic is not None and min(res.bic, key=res.bic.get) == 2
    assert res.n_kept == len(df)
    pred = res.labels.to_numpy()
    assert not np.any(pred == DROPPED_LABEL)
    assert _agreement(pred, truth) > 0.95


def test_kmeans_k2_recovers_blobs() -> None:
    df, truth = _two_blobs()
    res = cluster(df, ["amp", "dwell"], method="kmeans", k=2)
    assert res.method == "kmeans"
    assert res.k == 2
    assert res.bic is None
    assert _agreement(res.labels.to_numpy(), truth) > 0.95


def test_cluster_labels_realign_with_dropped_rows() -> None:
    df, _ = _two_blobs(n=60)
    df = df.copy()
    df.loc[df.index[0], "amp"] = np.nan  # force one dropped row
    res = cluster(df, ["amp", "dwell"], method="gmm", k=2)
    assert res.labels.iloc[0] == DROPPED_LABEL
    assert res.n_kept == len(df) - 1
    # sizes + means cover exactly the fitted clusters
    assert set(res.sizes) == set(res.means.index)
    assert sum(res.sizes.values()) == res.n_kept
    assert list(res.means.columns) == ["amp", "dwell"]


def test_cluster_means_are_in_original_units() -> None:
    df, truth = _two_blobs(sep=12.0)
    res = cluster(df, ["amp", "dwell"], method="kmeans", k=2)
    # one cluster near (0,0), one near (12,12) -- standardisation must not leak out
    mins = res.means.min()
    maxs = res.means.max()
    assert mins["amp"] < 3.0 and maxs["amp"] > 9.0


def test_cluster_unknown_method_raises() -> None:
    df, _ = _two_blobs(n=30)
    with pytest.raises(ValueError, match="unknown method"):
        cluster(df, ["amp", "dwell"], method="dbscan")


def test_cluster_k_gt_n_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="exceeds the number of valid rows"):
        cluster(df, ["a", "b"], method="gmm", k=5)


def test_cluster_too_few_rows_raises() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="need >= 2"):
        cluster(df, ["a", "b"], method="kmeans", k=2)


def test_cluster_bad_k_raises() -> None:
    df, _ = _two_blobs(n=30)
    with pytest.raises(ValueError, match="k must be >= 1"):
        cluster(df, ["amp", "dwell"], method="gmm", k=0)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def test_pca_concentrates_variance_on_pc1() -> None:
    rng = np.random.default_rng(1)
    t = rng.normal(0, 1, 400)
    # a correlated 3-D cloud: mostly one direction plus small noise
    df = pd.DataFrame(
        {
            "x": t + rng.normal(0, 0.05, 400),
            "y": 2 * t + rng.normal(0, 0.05, 400),
            "z": -t + rng.normal(0, 0.05, 400),
        }
    )
    res = pca(df, ["x", "y", "z"], n_components=2)
    assert isinstance(res, PCAResult)
    assert res.scores.shape == (400, 2)
    assert res.components.shape == (2, 3)
    assert res.explained_variance_ratio[0] > 0.95
    assert res.explained_variance_ratio[0] > res.explained_variance_ratio[1]
    assert len(res.kept_index) == 400


def test_pca_scores_align_to_kept_index() -> None:
    df = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan, 4.0, 5.0], "y": [1.0, 4.0, 9.0, 16.0, 25.0]},
        index=[10, 11, 12, 13, 14],
    )
    res = pca(df, ["x", "y"], n_components=1)
    assert list(res.kept_index) == [10, 11, 13, 14]
    assert res.scores.shape == (4, 1)
    assert res.n_kept == 4


def test_pca_too_many_components_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    with pytest.raises(ValueError, match="exceeds min"):
        pca(df, ["a", "b"], n_components=3)


def test_pca_bad_components_raises() -> None:
    df, _ = _two_blobs(n=30)
    with pytest.raises(ValueError, match="n_components must be >= 1"):
        pca(df, ["amp", "dwell"], n_components=0)


# ---------------------------------------------------------------------------
# fit pass-throughs
# ---------------------------------------------------------------------------


def test_fit_exponential_delegates_and_infers_t_min() -> None:
    rng = np.random.default_rng(2)
    t_min = 5e-5
    dwell = t_min + rng.exponential(2e-4, size=500)  # left-truncated-ish sample
    with pytest.warns(RuntimeWarning, match="min_dwell_s"):
        res = fit_exponential(dwell)  # t_min inferred from min(dwell)
    assert isinstance(res, FitResult)
    assert "tau_1" in res.params
    assert res.meta["t_min"] == pytest.approx(float(dwell.min()))


def test_fit_exponential_is_never_silent_about_a_guessed_t_min() -> None:
    """B1: substituting min(dwell) for the detector's floor biases tau long by ~t_min.

    That substitution used to happen with no warning anywhere, which is how the
    Plots window and the Fits panel came to print two different taus for the same
    events. It must be visible in the warning stream AND on the result itself, so
    a caller that forgot to pass t_min still shows the user what happened."""
    rng = np.random.default_rng(21)
    dwell = 5e-5 + rng.exponential(2e-4, size=400)

    with pytest.warns(RuntimeWarning, match="shortest observed dwell"):
        res = fit_exponential(dwell)

    assert res.meta["t_min_source"] == "observed_min"
    note = res.meta["warnings"][0]
    assert "provenance has no 'min_dwell_s'" in note
    assert "pass an EventSet from detect() to get it right" in note
    assert note in res.summary()


def test_fit_exponential_forwards_explicit_t_min() -> None:
    rng = np.random.default_rng(3)
    dwell = 1e-4 + rng.exponential(3e-4, size=300)
    res = fit_exponential(dwell, t_min=1e-4)
    assert res.meta["t_min"] == pytest.approx(1e-4)
    assert res.meta["t_min_source"] == "explicit"
    assert res.meta["warnings"] == []   # nothing was guessed, so nothing is claimed


def test_fit_gaussian_mixture_delegates() -> None:
    rng = np.random.default_rng(4)
    vals = np.concatenate(
        [rng.normal(0.2, 0.02, 200), rng.normal(0.6, 0.03, 200)]
    )
    res = fit_gaussian_mixture(vals, k=2)
    assert isinstance(res, FitResult)
    assert {"mu_1", "mu_2", "sigma_1", "sigma_2"}.issubset(res.params)
    # two well-separated modes recovered, ordered by mean
    assert res.params["mu_1"] < res.params["mu_2"]


# ---------------------------------------------------------------------------
# linear / power-law best-fit line
# ---------------------------------------------------------------------------


def test_linear_fit_recovers_known_slope_and_intercept() -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = 2.5 * x - 3.0                       # exact line, no noise
    fit = linear_fit(x, y, xlabel="x", ylabel="y")
    assert isinstance(fit, LinearFit)
    assert fit.log_log is False
    assert fit.slope == pytest.approx(2.5, abs=1e-9)
    assert fit.intercept == pytest.approx(-3.0, abs=1e-9)
    assert fit.r2 == pytest.approx(1.0, abs=1e-12)
    assert fit.n == 50
    assert fit.xlabel == "x" and fit.ylabel == "y"
    # predict evaluates the line in data units.
    np.testing.assert_allclose(fit.predict(x), y, atol=1e-9)


def test_linear_fit_drops_nonfinite_pairs() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([3.0, 5.0, 7.0, np.inf, 11.0])   # y = 2x+1 on the finite pairs
    fit = linear_fit(x, y)
    assert fit.n == 3                              # rows 0, 1, 4 survive
    assert fit.slope == pytest.approx(2.0, abs=1e-9)
    assert fit.intercept == pytest.approx(1.0, abs=1e-9)


def test_linear_fit_noisy_data_has_r2_between_0_and_1() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 200)
    y = 1.5 * x + 2.0 + rng.normal(0.0, 3.0, x.size)
    fit = linear_fit(x, y)
    assert 0.0 < fit.r2 < 1.0
    # the slope is still recovered roughly
    assert fit.slope == pytest.approx(1.5, abs=0.3)


def test_linear_fit_flat_response_r2_is_nan() -> None:
    x = np.linspace(0.0, 5.0, 20)
    y = np.full_like(x, 4.0)                # zero variance -> SStot == 0
    fit = linear_fit(x, y)
    assert np.isnan(fit.r2)
    assert fit.slope == pytest.approx(0.0, abs=1e-9)


def test_linear_fit_power_law_recovered_in_log_log() -> None:
    x = np.linspace(1.0, 20.0, 60)
    y = 3.0 * x ** 2                        # a pure power law
    fit = linear_fit(x, y, log_log=True)
    assert fit.log_log is True
    assert fit.meta["exponent"] == pytest.approx(2.0, abs=1e-6)
    assert fit.meta["coefficient"] == pytest.approx(3.0, abs=1e-6)
    assert fit.slope == pytest.approx(2.0, abs=1e-6)
    assert fit.r2 == pytest.approx(1.0, abs=1e-9)
    # predict follows the power law in data units.
    np.testing.assert_allclose(fit.predict(x), y, rtol=1e-6)


def test_linear_fit_log_log_drops_nonpositive() -> None:
    x = np.array([-1.0, 0.0, 1.0, 2.0, 4.0])
    y = np.array([5.0, 5.0, 3.0, 12.0, 48.0])   # only x>0 & y>0 usable: last three
    fit = linear_fit(x, y, log_log=True)
    assert fit.n == 3                            # the -1 and 0 x-values are gone


def test_linear_fit_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match=">= 2 valid points"):
        linear_fit([1.0], [2.0])


def test_linear_fit_constant_x_raises() -> None:
    with pytest.raises(ValueError, match="degenerate x"):
        linear_fit([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])


def test_linear_fit_no_positive_for_log_log_raises() -> None:
    with pytest.raises(ValueError, match="valid points"):
        linear_fit([-1.0, -2.0, 0.0], [1.0, 2.0, 3.0], log_log=True)


def test_line_points_linear_spans_range() -> None:
    fit = linear_fit(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0]))
    lx, ly = line_points(fit, 0.0, 2.0, n=100)
    assert lx.shape == ly.shape == (100,)
    assert lx[0] == pytest.approx(0.0) and lx[-1] == pytest.approx(2.0)
    np.testing.assert_allclose(ly, 2.0 * lx + 1.0, atol=1e-9)


def test_line_points_power_law_is_log_spaced_and_positive_only() -> None:
    x = np.linspace(1.0, 10.0, 30)
    fit = linear_fit(x, 3.0 * x ** 2, log_log=True)
    lx, ly = line_points(fit, 1.0, 10.0, n=50)
    np.testing.assert_allclose(ly, 3.0 * lx ** 2, rtol=1e-6)
    with pytest.raises(ValueError, match="xmin, xmax > 0"):
        line_points(fit, 0.0, 10.0)


# ================================================================================================
# clustering must qualify its own answer, like every other analysis here
# ================================================================================================

def test_a_degenerate_cluster_is_reported_not_left_for_the_reader_to_notice():
    """MEASURED: on the owner's three apo-ferritin recordings of ONE sample, auto-BIC
    returned k = 7, 5 and 3 with clusters of 1, 3 and 4 events — three mutually
    contradictory answers, every one of them printed as a bare "sizes: 0:189, 1:1".

    A cluster of one event is not a population; it is a component spent on an outlier. Every
    other analysis in this package qualifies its numbers (the depth floor, the right-censoring
    cut, the guessed t_min, the unprobed baseline). Clustering was the one surface that did not.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "dwell_s": np.r_[rng.normal(0.018, 0.002, 240), [0.23]],
        "frac_depth": np.r_[rng.normal(2.5, 0.2, 240), [0.48]],
    })
    res = populations.cluster(df, ["dwell_s", "frac_depth"], k=2)
    assert min(res.sizes.values()) == 1
    assert res.warnings, "a cluster of one event was reported with no caveat"
    joined = " ".join(res.warnings)
    assert "DEGENERATE" in joined and "cluster 1 (n=1)" in joined


def test_a_healthy_clustering_says_nothing():
    """The mirror. A warning that fires on every run is noise, and this codebase has paid
    for that before — a caveat nobody reads is a caveat that is not there."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "dwell_s": np.r_[rng.normal(0.005, 0.0004, 300), rng.normal(0.030, 0.002, 300)],
        "frac_depth": np.r_[rng.normal(0.6, 0.05, 300), rng.normal(2.6, 0.15, 300)],
    })
    res = populations.cluster(df, ["dwell_s", "frac_depth"], k=2)
    assert min(res.sizes.values()) > 50
    assert res.warnings == (), f"a clean 2-cluster split warned anyway: {res.warnings}"


def test_an_auto_k_that_BIC_barely_preferred_is_called_unstable():
    """Auto-k reports a preference; when the margin is inside a few BIC units that
    preference is not a result. Reading k as "the number of populations" is exactly the
    mistake this note exists to block."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400)})
    res = populations.cluster(df, ["a", "b"], k=None, k_max=4)
    if res.bic and len(res.bic) > 1:
        ordered = sorted(res.bic.values())
        if abs(ordered[1] - ordered[0]) < 10.0:
            assert any("UNSTABLE k" in w for w in res.warnings), (
                f"BIC margin {abs(ordered[1]-ordered[0]):.2f} went unflagged")


# ---------------------------------------------------------------------------
# robust_scale — the winsorised median/IQR scaler shared with subpop_gmm
# ---------------------------------------------------------------------------


def test_robust_scale_centres_on_the_median_and_divides_by_the_IQR():
    x = np.arange(101.0)  # median 50, IQR 50
    Z = populations.robust_scale(x)
    assert Z.shape == (101, 1)
    assert Z[50, 0] == pytest.approx(0.0)
    assert Z[75, 0] - Z[25, 0] == pytest.approx(1.0)


def test_robust_scale_is_not_moved_by_an_outlier_that_would_wreck_a_z_score():
    """The whole reason this exists instead of StandardScaler.

    One event three orders of magnitude off the bulk. A z-score's SD absorbs it
    and collapses every other event toward zero; the median and the IQR do not
    move at all.
    """
    from sklearn.preprocessing import StandardScaler

    x = np.arange(200.0)
    y = x.copy()
    y[-1] = 1e6
    clean = populations.robust_scale(x.reshape(-1, 1))
    dirty = populations.robust_scale(y.reshape(-1, 1))
    assert np.allclose(clean[:190], dirty[:190])

    zc = StandardScaler().fit_transform(x.reshape(-1, 1))
    zd = StandardScaler().fit_transform(y.reshape(-1, 1))
    assert not np.allclose(zc[:190], zd[:190], atol=0.05)


def test_robust_scale_winsorises_before_it_scales():
    """The clip is visible in the OUTPUT: the extreme tail comes back flattened."""
    x = np.arange(200.0)
    x[-1] = 1e6
    x[-2] = 5e5
    Z = populations.robust_scale(x.reshape(-1, 1))
    assert Z[-1, 0] == pytest.approx(Z[-2, 0])
    assert np.isfinite(Z).all()


def test_robust_scale_maps_nonfinite_to_the_scaled_median_keeping_the_row():
    """Unlike feature_matrix, which drops the row whole. An event dropped here
    would be an event silently missing from a population FRACTION."""
    x = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 7.0])
    Z = populations.robust_scale(x.reshape(-1, 1))
    assert Z.shape == (7, 1)
    assert Z[2, 0] == 0.0 and Z[4, 0] == 0.0 and Z[5, 0] == 0.0


def test_robust_scale_survives_a_constant_and_an_all_nan_column():
    X = np.column_stack([np.full(30, 5.0), np.full(30, np.nan), np.arange(30.0)])
    Z = populations.robust_scale(X)
    assert np.isfinite(Z).all()
    assert np.all(Z[:, 0] == 0.0) and np.all(Z[:, 1] == 0.0)
    assert Z[:, 2].std() > 0


def test_robust_scale_treats_columns_independently():
    X = np.column_stack([np.arange(101.0), np.arange(101.0) * 1000.0])
    Z = populations.robust_scale(X)
    assert np.allclose(Z[:, 0], Z[:, 1])


@pytest.mark.parametrize(
    "X, limits, msg",
    [
        (np.arange(10.0), (99.0, 1.0), "low < high"),
        (np.arange(10.0), (-1.0, 99.0), "low < high"),
        (np.empty((0, 3)), (1.0, 99.0), "empty feature matrix"),
        (np.zeros((2, 2, 2)), (1.0, 99.0), "expected a 2-D"),
    ],
)
def test_robust_scale_guards(X, limits, msg):
    with pytest.raises(ValueError, match=msg):
        populations.robust_scale(X, limits=limits)


# ---------------------------------------------------------------------------
# cluster(n_init=...) — EM restarts on the BIC ladder
# ---------------------------------------------------------------------------


def test_cluster_n_init_default_is_unchanged_sklearn_behaviour():
    """The new keyword must not move any existing number."""
    from sklearn.mixture import GaussianMixture

    df, _ = _two_blobs(n=120)
    res = populations.cluster(df, ["amp", "dwell"], k=2, standardize=False)
    direct = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    assert np.array_equal(
        res.labels.to_numpy(), direct.fit_predict(df[["amp", "dwell"]].to_numpy(float))
    )
    assert res.meta["n_init"] == 1


def test_cluster_n_init_reaches_the_bic_ladder():
    from sklearn.mixture import GaussianMixture

    df, _ = _two_blobs(n=100)
    X = df[["amp", "dwell"]].to_numpy(float)
    res = populations.cluster(df, ["amp", "dwell"], k=None, k_max=3, standardize=False, n_init=3)
    for k in (1, 2, 3):
        want = GaussianMixture(
            n_components=k, covariance_type="full", n_init=3, random_state=0
        ).fit(X).bic(X)
        assert res.bic[k] == pytest.approx(float(want), rel=1e-12, abs=1e-9)


def test_cluster_rejects_a_non_positive_n_init():
    df, _ = _two_blobs(n=60)
    with pytest.raises(ValueError, match="n_init must be"):
        populations.cluster(df, ["amp", "dwell"], k=2, n_init=0)
