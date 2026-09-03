"""Tests for analysis.subpop_gmm: the zn_30h subpopulation quantification.

Two layers. The synthetic layer pins the mechanics on data whose answer is
known by construction -- that components come back numbered largest-first, that
the PPCA residual term is the density it claims to be, that the bootstrap's
Hungarian matching survives a permuted reference, that every guard raises rather
than returning a plausible wrong number.

The oracle layer (``@pytest.mark.corpus``) is the one that matters: the owner
already ran this analysis, and his own outputs -- ``subpop_zn_30h_table.csv``
and the GMM x HDBSCAN crosstab -- are the specification. ``n_PCs = 11``,
``K = 4``, BIC overfitting to ``K = 8`` and 98.61 % of events above a 0.9
posterior are HIS numbers, not targets to tune toward: a port that lands
elsewhere has diverged upstream, and the test says so by failing rather than by
being relaxed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import subpop_gmm as SG
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope
from nanoscope.analysis.stats import point_fits_only
from nanoscope.analysis.subpop_gmm import (
    CONFIDENCE_FLOOR,
    SUBPOP_FEATURES,
    FractionCI,
    PCSelection,
    SubpopFit,
    SubpopQuantification,
    bic_ladder,
    bootstrap_fractions,
    fit_subpopulations,
    label_crosstab,
    population_table,
    quantify_subpopulations,
    scale_features,
    select_pcs_and_k,
)

ZN_DIR = Path("/Users/abrahamcampos/Desktop/Zn2IonCondition/CSVData")
ORACLE_FEATURES = ZN_DIR / "event_clustering_features.csv"
ORACLE_TABLE = ZN_DIR / "subpop_zn_30h_table.csv"
ORACLE_META = ZN_DIR / "zn_30h_metadata.json"


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------


def _blobs(sizes=(120, 70, 40), d: int = 6, sep: float = 14.0, seed: int = 0):
    """Well-separated Gaussian blobs of KNOWN, distinct sizes, largest first.

    Centres march along the diagonal so every column carries the same spread.
    Putting each blob on its own axis instead would give the rarest blob's axis
    an IQR set by noise alone, and :func:`scale_features` would then blow that
    one feature up by an order of magnitude — a real property of the scaler, but
    not the thing these tests are pinning.
    """
    rng = np.random.default_rng(seed)
    parts, truth = [], []
    for i, n in enumerate(sizes):
        centre = np.full(d, sep * i, dtype=float)
        parts.append(rng.normal(centre, 1.0, size=(n, d)))
        truth.append(np.full(n, i))
    X = np.vstack(parts)
    y = np.concatenate(truth)
    order = rng.permutation(len(X))  # so blob order is not row order
    return X[order], y[order]


def _event_frame(X: np.ndarray, y: np.ndarray, *, seed: int = 1) -> pd.DataFrame:
    """A per-event table carrying the blob features plus the reporting columns."""
    rng = np.random.default_rng(seed)
    n = len(X)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["dwell_ms"] = np.exp(y * 1.5) + rng.random(n) * 0.01
    df["depth_max_pA"] = 20.0 + 10.0 * y
    df["depth_sust_pA"] = 10.0 + 5.0 * y
    df["f_sust"] = 0.05 + 0.01 * y
    df["entropy"] = 0.7 + 0.05 * y
    df["kurtosis"] = -1.0 + y
    df["onset_s"] = np.linspace(0.0, 100.0, n)
    df["time_half"] = np.where(df["onset_s"] <= df["onset_s"].median(), "early", "late")
    df.index = pd.Index([f"e{i:04d}" for i in range(n)], name="event")
    return df


@pytest.fixture(scope="module")
def blob_frame():
    X, y = _blobs()
    return _event_frame(X, y), y


# ---------------------------------------------------------------------------
# scale_features -- the winsorise-then-median/IQR front end
# ---------------------------------------------------------------------------


def test_scale_features_clips_before_it_scales():
    """The two largest values in a column come back EQUAL.

    That equality is the whole observable consequence of winsorising first: both
    outliers were clipped to the 99th percentile before the median/IQR were
    applied, so they land on the same scaled value. Scale first and clip after
    and they stay distinct.
    """
    col = np.arange(200.0)
    col[-1] = 1e6
    col[-2] = 5e5
    df = pd.DataFrame({"a": col, "b": np.arange(200.0)})
    Z = scale_features(df, ["a", "b"])
    assert Z[-1, 0] == pytest.approx(Z[-2, 0])
    # ... and the untouched column is not clipped anywhere in its interior
    assert len(np.unique(Z[:, 1])) > 190


def test_scale_features_puts_nonfinite_at_the_scaled_median_not_out_of_the_table():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, np.inf, 6.0]})
    Z = scale_features(df, ["a"])
    assert Z.shape == (6, 1)  # nothing dropped
    assert np.all(np.isfinite(Z))
    assert Z[2, 0] == 0.0 and Z[4, 0] == 0.0


def test_scale_features_constant_column_is_zero_not_infinite():
    df = pd.DataFrame({"a": np.full(50, 7.0), "b": np.arange(50.0)})
    Z = scale_features(df, ["a", "b"])
    assert np.all(Z[:, 0] == 0.0)
    assert np.all(np.isfinite(Z))


def test_scale_features_respects_the_requested_column_order():
    df = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * -3.0})
    ab = scale_features(df, ["a", "b"])
    ba = scale_features(df, ["b", "a"])
    assert np.allclose(ab, ba[:, ::-1])


@pytest.mark.parametrize(
    "cols, frame, msg",
    [
        ([], pd.DataFrame({"a": [1.0]}), "at least one"),
        (["zz"], pd.DataFrame({"a": [1.0]}), "not in the table"),
        (["a"], pd.DataFrame({"a": []}), "empty feature matrix"),
    ],
)
def test_scale_features_guards(cols, frame, msg):
    """The scaler is imported from subpop_cluster; its guards must still bite here."""
    with pytest.raises(ValueError, match=msg):
        scale_features(frame, cols)


def test_subpop_features_is_the_clustering_stage_list_not_a_copy():
    """One list, shared. Two literals would be two things to forget to update."""
    from nanoscope.analysis.subpop_cluster import CLUST_FEATS

    assert SUBPOP_FEATURES is CLUST_FEATS


def test_subpop_feature_order_is_the_specified_one():
    """entropy LAST, not in the middle where the feature CSV puts it."""
    assert SUBPOP_FEATURES[-1] == "entropy"
    assert SUBPOP_FEATURES[:2] == ("log_dwell", "depth_max_pA")
    assert len(SUBPOP_FEATURES) == 13
    assert len(set(SUBPOP_FEATURES)) == 13


# ---------------------------------------------------------------------------
# select_pcs_and_k
# ---------------------------------------------------------------------------


def test_residual_loglik_is_the_isotropic_gaussian_it_claims_to_be():
    """Hand-computed against the normal log-density on the discarded axes."""
    T = np.array([[1.0, 2.0, 3.0, 4.0]])
    ev = np.array([10.0, 5.0, 2.0, 1.0])
    got = SG._residual_loglik(T, ev, 2)
    s2 = (2.0 + 1.0) / 2.0
    want = sum(-0.5 * (v ** 2 / s2 + np.log(2 * np.pi * s2)) for v in (3.0, 4.0))
    assert got[0] == pytest.approx(want)


def test_residual_loglik_is_zero_when_nothing_is_discarded():
    T = np.ones((5, 3))
    assert np.all(SG._residual_loglik(T, np.array([3.0, 2.0, 1.0]), 3) == 0.0)
    assert np.all(SG._residual_loglik(T, np.array([3.0, 2.0, 1.0]), 9) == 0.0)


def test_select_recovers_the_true_component_count(blob_frame):
    df, y = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    sel = select_pcs_and_k(Xs, p_grid=(2, 3, 4), k_grid=(1, 2, 3, 4), n_splits=3)
    assert isinstance(sel, PCSelection)
    assert sel.k == 3  # three blobs were planted
    assert sel.grid.shape == (3, 4)
    assert sel.n_pcs in (2, 3, 4)
    # the argmax cell is the reported one
    row = sel.p_grid.index(sel.n_pcs)
    col = sel.k_grid.index(sel.k)
    assert sel.loglik == pytest.approx(float(sel.grid[row, col]))
    assert sel.loglik == pytest.approx(float(np.nanmax(sel.grid)))


def test_select_is_deterministic(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    kw = dict(p_grid=(2, 3), k_grid=(1, 2, 3), n_splits=3)
    a = select_pcs_and_k(Xs, **kw)
    b = select_pcs_and_k(Xs, **kw)
    assert np.array_equal(a.grid, b.grid, equal_nan=True)
    assert (a.n_pcs, a.k) == (b.n_pcs, b.k)


def test_select_profiles_slice_the_grid_the_way_they_say(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    sel = select_pcs_and_k(Xs, p_grid=(2, 3), k_grid=(1, 2, 3), n_splits=3)
    kp = sel.k_profile()
    assert list(kp) == list(sel.k_grid)
    row = sel.p_grid.index(sel.n_pcs)
    assert kp[sel.k] == pytest.approx(float(sel.grid[row, sel.k_grid.index(sel.k)]))
    pp = sel.p_profile()
    assert list(pp) == list(sel.p_grid)


def test_select_warns_when_k_lands_on_the_grid_edge(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    sel = select_pcs_and_k(Xs, p_grid=(3,), k_grid=(1, 2, 3), n_splits=3)
    assert sel.k == 3
    assert any("GRID EDGE" in w and "K=" in w for w in sel.warnings)


def test_select_warns_on_a_flat_optimum():
    """One blob: every K above 1 fits it equally well, so the choice is noise."""
    rng = np.random.default_rng(3)
    Xs = rng.normal(size=(150, 4))
    sel = select_pcs_and_k(Xs, p_grid=(2, 3), k_grid=(1, 2, 3), n_splits=3)
    assert any("FLAT OPTIMUM" in w for w in sel.warnings)


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        (dict(n_splits=1), "n_splits must be"),
        (dict(n_init=0), "n_init must be"),
        (dict(p_grid=()), "empty search grid"),
        (dict(k_grid=()), "empty search grid"),
        (dict(p_grid=(99,)), "must lie in 1.."),
        (dict(k_grid=(0,)), "must be >= 1"),
    ],
)
def test_select_guards(kwargs, msg):
    Xs = np.random.default_rng(0).normal(size=(40, 4))
    base = dict(p_grid=(2,), k_grid=(1,), n_splits=3)
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        select_pcs_and_k(Xs, **base)


def test_select_rejects_a_nonfinite_matrix_rather_than_fitting_it():
    Xs = np.ones((40, 3))
    Xs[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        select_pcs_and_k(Xs, p_grid=(2,), k_grid=(1,), n_splits=3)


def test_select_rejects_a_1d_matrix():
    with pytest.raises(ValueError, match="must be 2-D"):
        select_pcs_and_k(np.ones(30), p_grid=(1,), k_grid=(1,))


def test_select_polls_the_ambient_cancel_token():
    Xs = np.random.default_rng(0).normal(size=(200, 5))
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            select_pcs_and_k(Xs, p_grid=(2, 3), k_grid=(1, 2, 3), n_splits=3)


# ---------------------------------------------------------------------------
# fit_subpopulations
# ---------------------------------------------------------------------------


def test_fit_numbers_components_largest_first(blob_frame):
    df, y = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index)
    sizes = [fit.sizes[i] for i in range(3)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes == [120, 70, 40]  # the planted sizes, largest first


def test_fit_recovers_the_planted_partition(blob_frame):
    df, y = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index)
    # blob 0 was the largest, so it must be subpop 0; agreement is exact here
    assert np.array_equal(fit.labels.to_numpy(), y)


def test_fit_posteriors_are_reordered_with_the_labels(blob_frame):
    """The single most portable-looking way to break this port.

    ``posteriors`` is reordered by the same permutation as ``labels``. If only
    one of them is permuted the labels stay plausible and every confidence is
    silently wrong, so pin the agreement rather than the permutation.
    """
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index)
    assert np.array_equal(fit.posteriors.argmax(axis=1), fit.labels.to_numpy())
    assert np.allclose(fit.confidence, fit.posteriors.max(axis=1))
    assert np.allclose(fit.posteriors.sum(axis=1), 1.0)
    assert fit.conf_frac == pytest.approx((fit.confidence > CONFIDENCE_FLOOR).mean())
    assert fit.conf_median == pytest.approx(float(np.median(fit.confidence)))


def test_fit_means_and_weights_follow_the_same_order(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index)
    # weights are the model's fractions; with well-separated blobs they track
    # the hard-assignment sizes, and must be in the SAME (descending) order
    assert list(np.argsort(-fit.weights)) == [0, 1, 2]
    assert fit.means.shape == (3, 3)
    assert fit.weights.shape == (3,)


def test_fit_labels_carry_the_source_index(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index)
    assert fit.labels.index.equals(df.index)
    assert fit.labels.name == "subpop"


def test_fit_defaults_to_a_range_index(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3)
    assert isinstance(fit.labels.index, pd.RangeIndex)


def test_fit_is_deterministic(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    a = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index)
    b = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index)
    assert a.labels.equals(b.labels)
    assert np.allclose(a.posteriors, b.posteriors)
    assert a.bic == b.bic


def test_fit_warns_when_bic_disagrees_with_the_requested_k(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=2, bic_k_grid=(1, 2, 3))
    assert fit.bic_k == 3
    assert any("CRITERIA DISAGREE" in w for w in fit.warnings)


def test_fit_warns_on_a_degenerate_component():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 1, size=(300, 3)), rng.normal(60, 0.1, size=(3, 3))])
    fit = fit_subpopulations(X, n_pcs=2, k=2, bic_check=False)
    assert any("DEGENERATE COMPONENT" in w for w in fit.warnings)


def test_fit_warns_when_assignment_confidence_is_low():
    """One cloud split in two: every event sits on the shoulder."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 3))
    fit = fit_subpopulations(X, n_pcs=2, k=2, bic_check=False)
    assert fit.conf_frac < 0.8
    assert any("LOW ASSIGNMENT CONFIDENCE" in w for w in fit.warnings)


def test_fit_can_skip_the_bic_ladder(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, bic_check=False)
    assert fit.bic is None and fit.bic_k is None


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        (dict(n_pcs=0, k=2), "must lie in 1"),
        (dict(n_pcs=99, k=2), "must lie in 1"),
        (dict(n_pcs=2, k=0), "must lie in 1"),
        (dict(n_pcs=2, k=10_000), "must lie in 1"),
        (dict(n_pcs=2, k=2, n_init=0), "n_init must be"),
    ],
)
def test_fit_guards(kwargs, msg):
    X = np.random.default_rng(0).normal(size=(60, 4))
    with pytest.raises(ValueError, match=msg):
        fit_subpopulations(X, bic_check=False, **kwargs)


def test_fit_rejects_a_mis_sized_index():
    X = np.random.default_rng(0).normal(size=(60, 4))
    with pytest.raises(ValueError, match="would misalign"):
        fit_subpopulations(X, n_pcs=2, k=2, index=pd.RangeIndex(59), bic_check=False)


def test_fit_polls_the_ambient_cancel_token():
    X = np.random.default_rng(0).normal(size=(200, 4))
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            fit_subpopulations(X, n_pcs=2, k=2)


# ---------------------------------------------------------------------------
# bic_ladder -- reuse of populations.cluster
# ---------------------------------------------------------------------------


def test_bic_ladder_matches_a_direct_sklearn_ladder(blob_frame):
    """The delegation to populations.cluster must not change the numbers."""
    from sklearn.mixture import GaussianMixture

    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, bic_check=False)
    Z = fit.scores
    bic, argmin, _ = bic_ladder(Z, k_grid=(1, 2, 3, 4), n_init=3)
    for k in (1, 2, 3, 4):
        direct = GaussianMixture(
            n_components=k, covariance_type="full", n_init=3, random_state=0
        ).fit(Z).bic(Z)
        assert bic[k] == pytest.approx(float(direct), rel=1e-12, abs=1e-9)
    assert argmin == min(bic, key=bic.get)


def test_bic_ladder_rejects_a_non_contiguous_grid():
    Z = np.random.default_rng(0).normal(size=(60, 3))
    with pytest.raises(ValueError, match="contiguous k_grid"):
        bic_ladder(Z, k_grid=(2, 3, 4))


# ---------------------------------------------------------------------------
# bootstrap_fractions
# ---------------------------------------------------------------------------


def test_bootstrap_matches_components_through_a_permuted_reference(blob_frame):
    """Hungarian matching is what makes the intervals mean anything.

    Permute the reference means and the intervals must permute with them. Drop
    the matching and each column would average over arbitrary EM numberings, so
    every interval would come out the same width.
    """
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, bic_check=False)
    a = bootstrap_fractions(fit.scores, fit.means, n_boot=12)
    perm = np.array([2, 0, 1])
    b = bootstrap_fractions(fit.scores, fit.means[perm], n_boot=12)
    assert np.allclose(a.lo[perm], b.lo)
    assert np.allclose(a.hi[perm], b.hi)


def test_bootstrap_brackets_the_true_fractions_on_separated_blobs(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, bic_check=False)
    ci = bootstrap_fractions(fit.scores, fit.means, n_boot=40)
    truth = np.array([120, 70, 40]) / 230.0
    assert np.all(ci.lo <= truth + 1e-9)
    assert np.all(ci.hi >= truth - 1e-9)
    assert ci.computed and ci.n_ok == 40


def test_bootstrap_is_reproducible_and_seed_sensitive(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, bic_check=False)
    a = bootstrap_fractions(fit.scores, fit.means, n_boot=15, seed=0)
    b = bootstrap_fractions(fit.scores, fit.means, n_boot=15, seed=0)
    c = bootstrap_fractions(fit.scores, fit.means, n_boot=15, seed=1)
    assert np.array_equal(a.samples, b.samples, equal_nan=True)
    assert not np.array_equal(a.samples, c.samples, equal_nan=True)


def test_bootstrap_skipped_by_point_fits_only(blob_frame):
    """The app's standing rule: a CI nobody reads must not cost 200 fits."""
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, bic_check=False)
    with point_fits_only():
        ci = bootstrap_fractions(fit.scores, fit.means, n_boot=200)
    assert not ci.computed
    assert ci.n_ok == 0
    assert np.all(np.isnan(ci.lo)) and np.all(np.isnan(ci.hi))


def test_bootstrap_zero_resamples_is_a_skip_not_a_crash(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, bic_check=False)
    ci = bootstrap_fractions(fit.scores, fit.means, n_boot=0)
    assert not ci.computed and ci.samples.shape == (0, 3)


def test_bootstrap_flags_the_label_switching_signature():
    """The caveat the owner's own zn_30h intervals earn.

    Constructed here rather than measured: an interval that swallows another
    component's point estimate is the signature, and the warning must fire on it.
    """
    notes = SG._ci_warnings(
        np.array([0.36, 0.274, 0.25, 0.116]),
        np.array([0.182, 0.152, 0.111, 0.014]),
        np.array([0.638, 0.660, 0.268, 0.118]),
    )
    joined = " ".join(notes)
    assert "LABEL SWITCHING" in joined
    assert "UNINFORMATIVE INTERVAL" in joined
    assert "NOT calibrated" in joined


def test_ci_warnings_stay_quiet_on_a_tight_well_separated_bootstrap():
    notes = SG._ci_warnings(
        np.array([0.60, 0.30, 0.10]),
        np.array([0.57, 0.27, 0.08]),
        np.array([0.63, 0.33, 0.12]),
    )
    assert notes == ()


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        (dict(n_boot=-1), "n_boot must be"),
        (dict(n_init=0), "n_init must be"),
    ],
)
def test_bootstrap_guards(kwargs, msg):
    Z = np.random.default_rng(0).normal(size=(50, 3))
    with pytest.raises(ValueError, match=msg):
        bootstrap_fractions(Z, np.zeros((2, 3)), **kwargs)


def test_bootstrap_rejects_means_of_the_wrong_width():
    Z = np.random.default_rng(0).normal(size=(50, 3))
    with pytest.raises(ValueError, match="reference_means must be"):
        bootstrap_fractions(Z, np.zeros((2, 5)), n_boot=2)


def test_bootstrap_polls_the_ambient_cancel_token():
    Z = np.random.default_rng(0).normal(size=(80, 3))
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            bootstrap_fractions(Z, np.zeros((2, 3)), n_boot=50)


# ---------------------------------------------------------------------------
# population_table
# ---------------------------------------------------------------------------


def test_population_table_columns_and_order(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    tab = population_table(df, fit, duration_s=100.0)
    assert list(tab.columns) == [
        "subpop", "n", "fraction", "frac_lo95", "frac_hi95", "rate_per_s",
        "dwell_p50_ms", "dwell_p90_ms", "depth_max_pA", "depth_sust_pA",
        "f_sust", "entropy", "kurtosis", "conf_med", "early_frac",
    ]
    assert list(tab["subpop"]) == [0, 1, 2]
    assert list(tab["n"]) == [120, 70, 40]
    assert tab["fraction"].sum() == pytest.approx(1.0, abs=1e-3)
    assert tab["rate_per_s"].tolist() == [1.2, 0.7, 0.4]


def test_population_table_profiles_are_medians(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    tab = population_table(df, fit, duration_s=100.0, decimals=None)
    for k in range(3):
        sub = df.loc[fit.labels == k]
        assert tab.loc[k, "entropy"] == pytest.approx(sub["entropy"].median())
        assert tab.loc[k, "dwell_p50_ms"] == pytest.approx(sub["dwell_ms"].median())
        assert tab.loc[k, "dwell_p90_ms"] == pytest.approx(sub["dwell_ms"].quantile(0.9))


def test_population_table_early_frac_reads_time_half(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    tab = population_table(df, fit, duration_s=100.0, decimals=None)
    for k in range(3):
        sub = df.loc[fit.labels == k]
        assert tab.loc[k, "early_frac"] == pytest.approx(
            (sub["time_half"] == "early").mean()
        )


def test_population_table_without_time_half_is_nan_not_a_guess(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    tab = population_table(df.drop(columns=["time_half"]), fit, duration_s=100.0)
    assert tab["early_frac"].isna().all()


def test_population_table_without_duration_leaves_rate_nan(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    tab = population_table(df, fit)
    assert tab["rate_per_s"].isna().all()


def test_population_table_rejects_a_reindexed_frame(blob_frame):
    """The index-alignment hazard, made loud.

    A positional label array assigned into a label-indexed frame misaligns every
    row and still produces a table that looks fine. This must raise.
    """
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    with pytest.raises(ValueError, match="misaligns every row"):
        population_table(df.reset_index(drop=True), fit, duration_s=100.0)


def test_population_table_rejects_a_filtered_frame(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    with pytest.raises(ValueError, match="labelled"):
        population_table(df.iloc[:100], fit, duration_s=100.0)


def test_population_table_rejects_a_non_positive_duration(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    with pytest.raises(ValueError, match="duration_s must be"):
        population_table(df, fit, duration_s=0.0)


def test_population_table_decimals_none_is_unrounded(blob_frame):
    df, _ = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    rounded = population_table(df, fit, duration_s=100.0)
    raw = population_table(df, fit, duration_s=100.0, decimals=None)
    assert rounded["fraction"].tolist() == [
        round(v, 3) for v in raw["fraction"].tolist()
    ]
    # 70/230 = 0.30434... — rounding really did throw information away, so the
    # two calls are genuinely different rather than accidentally equal.
    assert raw.loc[1, "fraction"] != rounded.loc[1, "fraction"]
    assert raw.loc[1, "fraction"] == pytest.approx(70 / 230)


# ---------------------------------------------------------------------------
# label_crosstab
# ---------------------------------------------------------------------------


def test_crosstab_counts_agreement(blob_frame):
    df, y = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    other = pd.Series(y, index=df.index)
    xt = label_crosstab(fit, other, name="truth")
    assert xt.index.name == "subpop" and xt.columns.name == "truth"
    assert np.array_equal(xt.to_numpy(), np.diag([120, 70, 40]))


def test_crosstab_aligns_a_series_by_index_not_by_position(blob_frame):
    df, y = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    shuffled = pd.Series(y, index=df.index).sample(frac=1.0, random_state=0)
    assert not shuffled.index.equals(df.index)
    xt = label_crosstab(fit, shuffled, name="truth")
    assert np.array_equal(xt.to_numpy(), np.diag([120, 70, 40]))


def test_crosstab_rejects_a_series_missing_events(blob_frame):
    df, y = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    with pytest.raises(ValueError, match="no entry in"):
        label_crosstab(fit, pd.Series(y[:100], index=df.index[:100]))


def test_crosstab_takes_a_sequence_positionally(blob_frame):
    df, y = blob_frame
    Xs = scale_features(df, [f"f{i}" for i in range(6)])
    fit = fit_subpopulations(Xs, n_pcs=3, k=3, index=df.index, bic_check=False)
    xt = label_crosstab(fit, list(y))
    assert np.array_equal(xt.to_numpy(), np.diag([120, 70, 40]))


# ---------------------------------------------------------------------------
# quantify_subpopulations -- the whole thing
# ---------------------------------------------------------------------------


def test_quantify_end_to_end(blob_frame):
    df, y = blob_frame
    q = quantify_subpopulations(
        df,
        duration_s=100.0,
        features=[f"f{i}" for i in range(6)],
        exploratory_labels=pd.Series(y, index=df.index),
        exploratory_name="truth",
        p_grid=(2, 3, 4),
        k_grid=(1, 2, 3, 4),
        n_splits=3,
        n_boot=20,
    )
    assert isinstance(q, SubpopQuantification)
    assert isinstance(q.fit, SubpopFit) and isinstance(q.ci, FractionCI)
    assert q.n_events == 230
    assert q.fit.k == q.selection.k == 3
    assert list(q.table["n"]) == [120, 70, 40]
    assert q.crosstab is not None
    assert np.array_equal(q.crosstab.to_numpy(), np.diag([120, 70, 40]))
    assert q.features == tuple(f"f{i}" for i in range(6))


def test_quantify_is_deterministic(blob_frame):
    df, _ = blob_frame
    kw = dict(
        duration_s=100.0,
        features=[f"f{i}" for i in range(6)],
        p_grid=(2, 3),
        k_grid=(1, 2, 3),
        n_splits=3,
        n_boot=10,
    )
    a = quantify_subpopulations(df, **kw)
    b = quantify_subpopulations(df, **kw)
    assert a.fit.labels.equals(b.fit.labels)
    assert np.array_equal(a.ci.samples, b.ci.samples, equal_nan=True)
    pd.testing.assert_frame_equal(a.table, b.table)


def test_quantify_collects_and_dedups_every_warning(blob_frame):
    df, _ = blob_frame
    q = quantify_subpopulations(
        df.drop(columns=["time_half"]),
        features=[f"f{i}" for i in range(6)],
        p_grid=(2, 3),
        k_grid=(1, 2, 3),
        n_splits=3,
        n_boot=0,
    )
    joined = " ".join(q.warnings)
    assert "NO TIME-STABILITY CHECK" in joined
    assert "NO EVENT RATES" in joined
    assert len(q.warnings) == len(set(q.warnings))


def test_quantify_without_exploratory_labels_has_no_crosstab(blob_frame):
    df, _ = blob_frame
    q = quantify_subpopulations(
        df,
        duration_s=100.0,
        features=[f"f{i}" for i in range(6)],
        p_grid=(2, 3),
        k_grid=(1, 2),
        n_splits=3,
        n_boot=0,
    )
    assert q.crosstab is None


# ===========================================================================
# THE ORACLE: the owner's own zn_30h run.
# ===========================================================================


pytestmark_corpus = pytest.mark.corpus


@pytest.fixture(scope="module")
def zn30h():
    """The owner's zn_30h slice of his own clustering-feature table."""
    if not ORACLE_FEATURES.exists():
        pytest.skip(f"owner's feature table absent: {ORACLE_FEATURES}")
    clust = pd.read_csv(ORACLE_FEATURES)
    sp = clust[clust.tag == "zn_30h"].copy()
    return clust, sp


@pytest.fixture(scope="module")
def zn30h_quantified(zn30h):
    """ONE run of the full pipeline, shared by every oracle assertion.

    Module-scoped because the cross-validated search is 480 mixture fits plus a
    200-resample bootstrap; re-running it per assertion would add minutes to the
    suite for no extra information.
    """
    clust, sp = zn30h
    if not ORACLE_META.exists():
        pytest.skip(f"owner's metadata absent: {ORACLE_META}")
    duration = json.loads(ORACLE_META.read_text())["duration_s"]
    return quantify_subpopulations(
        sp,
        duration_s=duration,
        exploratory_labels=clust.loc[sp.index, "hdbscan_30"],
        exploratory_name="hdbscan_30",
    )


@pytest.mark.corpus
def test_oracle_gate_kept_1076_events(zn30h):
    """The gate does most of the work: 11,004 fired, 1,076 survived."""
    _, sp = zn30h
    assert len(sp) == 1076


@pytest.mark.corpus
def test_oracle_selects_11_pcs_and_4_components(zn30h_quantified):
    """HIS numbers. A port that lands elsewhere diverged upstream."""
    sel = zn30h_quantified.selection
    assert (sel.n_pcs, sel.k) == (11, 4)
    assert sel.loglik == pytest.approx(16.41, abs=0.01)


@pytest.mark.corpus
def test_oracle_k_profile(zn30h_quantified):
    want = {1: 7.42, 2: 13.73, 3: 15.55, 4: 16.41, 5: 15.33, 6: 15.64, 7: 15.69, 8: 14.66}
    got = zn30h_quantified.selection.k_profile()
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, abs=0.01)


@pytest.mark.corpus
def test_oracle_bic_overfits_to_eight(zn30h_quantified):
    """BIC is the cross-check, and it disagrees -- loudly and on the record."""
    fit = zn30h_quantified.fit
    assert fit.bic_k == 8
    assert any("CRITERIA DISAGREE" in w for w in fit.warnings)


@pytest.mark.corpus
def test_oracle_assignment_confidence(zn30h_quantified):
    """The owner's prose says "99 %"; his ``:.0%`` format rounded 98.61 %."""
    fit = zn30h_quantified.fit
    assert fit.conf_frac == pytest.approx(0.9861, abs=5e-4)
    assert fit.conf_median == pytest.approx(1.0, abs=1e-6)


@pytest.mark.corpus
def test_oracle_population_table_is_reproduced_exactly(zn30h_quantified):
    if not ORACLE_TABLE.exists():
        pytest.skip(f"owner's table absent: {ORACLE_TABLE}")
    want = pd.read_csv(ORACLE_TABLE)
    got = zn30h_quantified.table
    assert list(got.columns) == list(want.columns)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        want.reset_index(drop=True),
        check_dtype=False,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.corpus
def test_oracle_crosstab_against_hdbscan_30(zn30h_quantified):
    want = np.array([[2, 385, 0], [44, 251, 0], [70, 156, 43], [113, 12, 0]])
    assert np.array_equal(zn30h_quantified.crosstab.to_numpy(), want)


@pytest.mark.corpus
def test_oracle_sizes_and_fractions(zn30h_quantified):
    assert [zn30h_quantified.fit.sizes[i] for i in range(4)] == [387, 295, 269, 125]


@pytest.mark.corpus
def test_oracle_bootstrap_intervals_are_flagged_uncalibrated(zn30h_quantified):
    """His CIs really are this wide, and the port must SAY so, not smooth it.

    subpop 0's interval (0.182-0.638) swallows subpop 1's and subpop 2's point
    estimates whole. That is label switching surviving the Hungarian match, not
    3x uncertainty in a population fraction.
    """
    ci = zn30h_quantified.ci
    assert ci.computed and ci.n_boot == 200
    assert ci.lo[0] == pytest.approx(0.182, abs=5e-4)
    assert ci.hi[0] == pytest.approx(0.638, abs=5e-4)
    joined = " ".join(ci.warnings)
    assert "LABEL SWITCHING" in joined
    assert "NOT calibrated" in joined
    assert "LABEL SWITCHING" in " ".join(zn30h_quantified.warnings)


@pytest.mark.corpus
def test_oracle_the_95_percent_variance_rule_would_have_blurred_the_partition(zn30h):
    """Why the PC count is cross-validated instead of read off a variance curve.

    The measurement behind the module docstring's first claim. On zn_30h the
    95 %-variance rule stops at 4 PCs; the same K=4 mixture fit there lands on a
    different partition and assigns only 79 % of events confidently against the
    98.6 % the CV-selected 11 PCs manage. The separating structure is in the PCs
    the variance rule discarded.
    """
    from sklearn.decomposition import PCA

    _, sp = zn30h
    Xs = scale_features(sp)
    evr = np.cumsum(PCA(random_state=0).fit(Xs).explained_variance_ratio_)
    nk = int(np.searchsorted(evr, 0.95)) + 1
    assert nk == 4

    thin = fit_subpopulations(Xs, n_pcs=nk, k=4, index=sp.index, bic_check=False)
    full = fit_subpopulations(Xs, n_pcs=11, k=4, index=sp.index, bic_check=False)
    assert [thin.sizes[i] for i in range(4)] == [444, 307, 240, 85]
    assert [full.sizes[i] for i in range(4)] == [387, 295, 269, 125]
    assert thin.conf_frac == pytest.approx(0.79, abs=5e-3)
    assert full.conf_frac == pytest.approx(0.9861, abs=5e-4)
    assert any("LOW ASSIGNMENT CONFIDENCE" in w for w in thin.warnings)


@pytest.mark.corpus
def test_oracle_run_is_reproducible(zn30h):
    """Two runs of the seeded pipeline give identical labels.

    Cheap enough to repeat: the selection is skipped (both runs are pinned at
    the oracle's own (11, 4)) and the bootstrap is short.
    """
    _, sp = zn30h
    Xs = scale_features(sp)
    a = fit_subpopulations(Xs, n_pcs=11, k=4, index=sp.index, bic_check=False)
    b = fit_subpopulations(Xs, n_pcs=11, k=4, index=sp.index, bic_check=False)
    assert a.labels.equals(b.labels)
    assert [a.sizes[i] for i in range(4)] == [387, 295, 269, 125]
    ca = bootstrap_fractions(a.scores, a.means, n_boot=8)
    cb = bootstrap_fractions(b.scores, b.means, n_boot=8)
    assert np.array_equal(ca.samples, cb.samples, equal_nan=True)


# ================================================================================================
# the BIC cross-check's warnings belong to the BIC cross-check
# ================================================================================================

def test_bic_ladder_warnings_name_the_fit_they_came_from():
    """``bic_ladder`` warns about ITS OWN argmin fit, at a K the module does not
    report and with a component numbering that is not the reported one.

    Merged flat into the report's caveats those read as complaints about the
    table above them: on the owner's zn_30h the BIC argmin is K=8, so the caveat
    said "DEGENERATE CLUSTER(S): cluster 5 (n=10)" directly under a table listing
    subpops 0-3. A reader looks for cluster 5, does not find it, and concludes
    either that the table is wrong or that a subpopulation was dropped. No number
    was wrong; the attribution was.

    Forced degeneracy: well-separated blobs with a few starved ones, so the
    ladder's argmin fit HAS a component small enough to complain about.
    """
    rng = np.random.default_rng(3)
    Z = np.vstack([
        rng.normal(c * 12.0, 0.35, size=(n, 3))
        for c, n in zip(range(6), (300, 300, 300, 300, 20, 8))
    ])
    _bic, k, notes = bic_ladder(Z, k_grid=(1, 2, 3, 4, 5, 6), n_init=2)

    assert notes, "no warning to attribute -- pick a harsher blob configuration"
    for note in notes:
        assert note.startswith("BIC CROSS-CHECK"), note
        assert f"K={k}" in note, note
        assert "NOT the reported model" in note, note
