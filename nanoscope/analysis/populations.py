"""Population-level exploration of per-event tables: clustering, PCA, and fits.

This is the pure, Qt-free core behind the interactive Plots window. It takes the
same per-event ``DataFrame`` that the rest of the app already speaks -- a
detected :class:`~nanoscope.analysis.events.EventSet` ``.df`` or the manual
:class:`~nanoscope.gui.measure_panel.MeasurePanel` measurements -- and answers
three unsupervised questions about it:

* **Are there subpopulations?** :func:`cluster` fits a Gaussian mixture (or
  k-means) over a user-chosen feature set and, for GMM with ``k`` left open,
  picks the number of components by minimum BIC. No labels, no training set --
  the structure is discovered, not taught.
* **What varies together?** :func:`pca` projects many per-event parameters down
  to a couple of components so a 2-D scatter can stand in for a high-dimensional
  cloud.
* **What distribution does one axis follow?** :func:`fit_exponential` and
  :func:`fit_gaussian_mixture` are thin pass-throughs to the censored MLE fits
  in :mod:`nanoscope.analysis.stats`, exposed here so the plots layer has a
  single import surface. They are *not* reimplemented -- the truncation physics
  lives in ``stats`` and must stay there.

The one invariant everything leans on: labels and scores map back to the
*original* ``df`` rows. Rows with a NaN/inf in any chosen feature are dropped
before fitting (you cannot cluster on a missing value), but the returned
``labels`` is realigned to ``df.index`` with ``-1`` marking the dropped rows, so
a caller can colour its scatter without bookkeeping.

sklearn/numpy only -- no supervised classification, no deep learning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as _SKLearnPCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from nanoscope.analysis.modules import NO_TMIN_WARNING
from nanoscope.analysis.stats import (
    FitResult,
    fit_depth_gaussians,
    fit_dwell_exponential,
)

__all__ = [
    "ClusterResult",
    "PCAResult",
    "LinearFit",
    "feature_matrix",
    "robust_scale",
    "cluster",
    "pca",
    "linear_fit",
    "line_points",
    "fit_exponential",
    "fit_gaussian_mixture",
]

DROPPED_LABEL = -1

#: Winsorisation limits for :func:`robust_scale`, in percent. The 1st/99th
#: percentile is where the Zn2-insulin waveform-clustering pipeline clips, and
#: the choice is load-bearing: ``kurtosis`` and ``area_pA_ms`` both carry a
#: handful of events three orders of magnitude off the median, and an IQR
#: computed on the unclipped column is a fine scale for a column whose extremes
#: then dominate every distance in the space.
WINSOR_LIMITS = (1.0, 99.0)


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """A fitted clustering, with labels realigned to the source ``DataFrame``.

    Attributes:
        labels: Integer cluster id per source row, indexed by the original
            ``df.index``. Rows dropped for a non-finite feature carry
            :data:`DROPPED_LABEL` (``-1``).
        k: Number of clusters actually fit.
        method: ``"gmm"`` or ``"kmeans"``.
        columns: The feature columns used, in order.
        sizes: ``{cluster_id: n_rows}`` over the kept rows.
        means: Per-cluster feature means in the **original** (un-standardised)
            units, a ``DataFrame`` indexed by cluster id with one column per
            feature.
        bic: For GMM, ``{k: BIC}`` over every candidate ``k`` considered (a
            single entry when ``k`` was pinned); ``None`` for k-means.
        chosen_k: The ``k`` selected -- the min-BIC candidate for an
            auto-selected GMM, otherwise just ``k``.
        n_kept: Number of rows that survived the finite-feature filter.
        warnings: Caveats about THIS clustering -- degenerate clusters, and an
            auto-selected ``k`` that BIC barely preferred. Every other analysis in this
            package qualifies its own numbers (the depth floor, the right-censoring cut,
            the guessed ``t_min``, the unprobed baseline); clustering was the one surface
            that returned a bare answer. Measured on the owner's three apo-ferritin
            recordings of ONE sample, auto-BIC returned k = 7, 5 and 3 with clusters of
            1, 3 and 4 events -- three mutually contradictory answers, none of them
            flagged. A cluster of one event is not a population, and a caller cannot tell
            that from ``sizes`` alone without knowing to look.
        meta: Diagnostics (standardisation flag, random_state, ...).
    """

    labels: pd.Series
    k: int
    method: str
    columns: list[str]
    sizes: dict[int, int]
    means: pd.DataFrame
    bic: dict[int, float] | None
    chosen_k: int
    n_kept: int
    warnings: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<ClusterResult {self.method} k={self.k} "
            f"n_kept={self.n_kept} cols={self.columns}>"
        )


@dataclass
class PCAResult:
    """A PCA projection, with scores realigned to the kept source rows.

    Attributes:
        scores: ``(n_kept, n_components)`` projection of the (optionally
            standardised) features onto the principal axes.
        kept_index: The ``df.index`` labels of the rows in ``scores``, in row
            order, so a caller can join scores back onto ``df``.
        explained_variance_ratio: Fraction of total variance on each component.
        components: ``(n_components, n_features)`` loadings -- the principal
            axes in feature space.
        columns: The feature columns used, in order.
        n_kept: Number of rows that survived the finite-feature filter.
        meta: Diagnostics (standardisation flag, ...).
    """

    scores: np.ndarray
    kept_index: pd.Index
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    columns: list[str]
    n_kept: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        evr = ", ".join(f"{r:.3f}" for r in self.explained_variance_ratio)
        return f"<PCAResult n_kept={self.n_kept} evr=[{evr}] cols={self.columns}>"


@dataclass
class LinearFit:
    """A straight-line least-squares fit, optionally in log-log space.

    The plain fit is ``y = slope*x + intercept``. A ``log_log`` fit is the same
    line drawn through ``(log10 x, log10 y)`` -- which is a *power law* in data
    units, ``y = coefficient * x**exponent`` with ``exponent = slope`` and
    ``coefficient = 10**intercept``. Both forms of the power-law parameters are
    surfaced in :attr:`meta` (``exponent``/``coefficient``) so the caller can
    print either the line or the power law without re-deriving anything.

    Attributes:
        slope: Line slope. For a log-log fit this is the power-law exponent.
        intercept: Line intercept. For a log-log fit this is ``log10`` of the
            power-law coefficient.
        r2: Coefficient of determination in the fitted space (``log10`` space for
            a log-log fit); ``nan`` when the response has zero variance.
        n: Number of finite (and, for log-log, positive) points actually fit.
        xlabel: Name of the x column, for titles/details (``""`` if unknown).
        ylabel: Name of the y column, for titles/details (``""`` if unknown).
        log_log: Whether the fit was done on ``log10`` of both axes.
        meta: Diagnostics; for a log-log fit also carries ``exponent`` and
            ``coefficient`` (the power-law form).
    """

    slope: float
    intercept: float
    r2: float
    n: int
    xlabel: str = ""
    ylabel: str = ""
    log_log: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted y over ``x`` in DATA units.

        For a plain fit this is ``slope*x + intercept``; for a log-log fit it is
        the power law ``coefficient * x**exponent`` (which needs ``x > 0``).
        """
        x = np.asarray(x, dtype=np.float64)
        if self.log_log:
            coef = 10.0 ** self.intercept
            return coef * np.power(x, self.slope)
        return self.slope * x + self.intercept

    def __repr__(self) -> str:
        if self.log_log:
            coef = 10.0 ** self.intercept
            return (
                f"<LinearFit log-log y={coef:.4g}*x^{self.slope:.4g} "
                f"r2={self.r2:.3f} n={self.n}>"
            )
        return (
            f"<LinearFit y={self.slope:.4g}*x+{self.intercept:.4g} "
            f"r2={self.r2:.3f} n={self.n}>"
        )


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------


def feature_matrix(
    df: pd.DataFrame, columns: Sequence[str]
) -> tuple[np.ndarray, pd.Index]:
    """Pull ``columns`` from ``df`` as a clean float matrix, dropping bad rows.

    Non-numeric entries are coerced to NaN, then any row with a NaN or inf in
    any selected column is dropped whole -- you cannot cluster or project on a
    partially-missing feature vector without silently imputing, and imputing
    behind the caller's back is worse than dropping.

    Args:
        df: The per-event table.
        columns: Feature columns to extract, in order.

    Returns:
        ``(X, kept_index)`` where ``X`` is a ``float64`` array of shape
        ``(n_kept, len(columns))`` and ``kept_index`` is the ``df.index`` labels
        of the surviving rows, in the same row order -- so labels/scores derived
        from ``X`` map straight back onto ``df``.

    Raises:
        ValueError: if ``columns`` is empty or names a column absent from ``df``.
    """
    cols = list(columns)
    if len(cols) < 1:
        raise ValueError("feature_matrix needs at least one column; got none")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) {missing} not in the DataFrame; available: {list(df.columns)}"
        )
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    X = sub.to_numpy(dtype=np.float64)
    if X.ndim == 1:  # single row edge case -> keep 2-D
        X = X.reshape(-1, len(cols))
    good = np.all(np.isfinite(X), axis=1)
    return X[good], df.index[good]


def robust_scale(
    X,
    *,
    limits: tuple[float, float] = WINSOR_LIMITS,
) -> np.ndarray:
    """Winsorise each column, then centre on its median and divide by its IQR.

    THE scaler for per-event *waveform* features, and deliberately not
    :class:`~sklearn.preprocessing.StandardScaler`. A mean and a standard
    deviation are both moved by the outliers they are supposed to be measuring:
    on the owner's Zn2-insulin recordings a single 30 ms event drags the
    ``area_pA_ms`` mean off the bulk of the sample and inflates its SD until
    every *other* event's scaled area collapses toward zero. Median and IQR do
    not move, so the scaled space keeps the structure the clustering is looking
    for.

    Order matters and is not the obvious one: **winsorise first, then compute
    the median and the IQR on the CLIPPED matrix.** Clipping after scaling would
    leave the scale itself set by the tails; clipping before it means the
    location and spread are measured on the same values the model will see.

    Non-finite handling is by design *lenient*, and differs from
    :func:`feature_matrix`, which drops a row whole. Here a NaN/inf entry is
    excluded from its own column's percentiles (via the ``nan``-aware
    reductions) and the resulting non-finite scaled entry lands at ``0.0`` --
    the scaled median, i.e. "no information from this feature for this row".
    That keeps every event in the table: a waveform feature that degenerates on
    one short event (``skewness`` of a three-sample event, say) must not delete
    the event from the population it belongs to. A caller that would rather drop
    such rows should filter with :func:`feature_matrix` first.

    A column with a zero (or non-finite) IQR is divided by ``1.0`` rather than
    by zero -- it is constant, so it contributes nothing but must not poison the
    whole matrix with infinities.

    Args:
        X: ``(n_samples, n_features)`` array-like of features, in column order.
            Coerced to ``float64``; a 1-D input is read as a single column.
        limits: ``(low_pct, high_pct)`` winsorisation percentiles, per column.

    Returns:
        A new ``float64`` array of the same shape, finite everywhere.

    Raises:
        ValueError: if ``X`` is not 2-D-able, is empty, or ``limits`` is not a
            strictly increasing pair inside ``[0, 100]``.
    """
    lo_pct, hi_pct = (float(limits[0]), float(limits[1]))
    if not (0.0 <= lo_pct < hi_pct <= 100.0):
        raise ValueError(
            f"limits must be (low, high) with 0 <= low < high <= 100, got {limits!r}"
        )
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {A.shape}")
    if A.size == 0:
        raise ValueError(f"empty feature matrix, shape {A.shape}")

    A = np.where(np.isfinite(A), A, np.nan)
    with warnings.catch_warnings():
        # An all-NaN column makes every nan-reduction below warn; the guards
        # that follow already give it the right answer (centre 0, scale 1).
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(A, [lo_pct, hi_pct], axis=0)
        A = np.clip(A, lo, hi)
        centre = np.nanmedian(A, axis=0)
        spread = np.nanpercentile(A, 75, axis=0) - np.nanpercentile(A, 25, axis=0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    Z = (A - centre) / spread
    return np.where(np.isfinite(Z), Z, 0.0)


def _prepare(
    df: pd.DataFrame, columns: Sequence[str], *, standardize: bool
) -> tuple[np.ndarray, np.ndarray, pd.Index, list[str]]:
    """Shared front end: extract, guard, and (optionally) standardise features."""
    cols = list(columns)
    X_raw, kept = feature_matrix(df, cols)
    n, p = X_raw.shape
    if p < 1:
        raise ValueError("need at least one feature column")
    if n < 2:
        raise ValueError(
            f"only {n} row(s) have finite values across {cols}; need >= 2 to fit"
        )
    if standardize:
        X = StandardScaler().fit_transform(X_raw)
    else:
        X = X_raw.astype(np.float64, copy=True)
    return X_raw, X, kept, cols


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def _summarise(
    df_index: pd.Index,
    kept: pd.Index,
    labels_kept: np.ndarray,
    X_raw: np.ndarray,
    cols: list[str],
) -> tuple[pd.Series, dict[int, int], pd.DataFrame]:
    """Build the realigned label series, cluster sizes, and per-cluster means."""
    labels = pd.Series(DROPPED_LABEL, index=df_index, dtype="int64")
    labels.loc[kept] = labels_kept.astype("int64")

    uniq = np.unique(labels_kept)
    sizes = {int(c): int(np.sum(labels_kept == c)) for c in uniq}
    means = pd.DataFrame(
        {col: [X_raw[labels_kept == c, j].mean() for c in uniq] for j, col in enumerate(cols)},
        index=pd.Index([int(c) for c in uniq], name="cluster"),
    )
    return labels, sizes, means


#: A cluster smaller than this is reported as degenerate: too small to be a population,
#: and a sign the model is describing outliers rather than structure. Both a floor and a
#: fraction, because "5 of 40" and "5 of 4000" are different statements.
_MIN_CLUSTER_N = 5
_MIN_CLUSTER_FRAC = 0.01

#: How close a runner-up BIC has to be before an auto-selected k is called unstable. BIC
#: differences below ~2 are conventionally "not worth a mention"; k is a structural claim,
#: so anything inside 10 is reported as a choice the data did not really make.
_BIC_MARGIN = 10.0


def _cluster_warnings(sizes: dict[int, int], n_kept: int, bic: dict[int, float] | None,
                      chosen_k: int, requested_k: int | None) -> tuple[str, ...]:
    """Caveats a caller must see before treating these labels as populations."""
    out: list[str] = []
    floor = max(_MIN_CLUSTER_N, int(round(_MIN_CLUSTER_FRAC * max(n_kept, 1))))
    tiny = {c: n for c, n in sorted(sizes.items()) if n < floor}
    if tiny:
        named = ", ".join(f"cluster {c} (n={n})" for c, n in tiny.items())
        out.append(
            f"DEGENERATE CLUSTER(S): {named} — below {floor} events "
            f"({_MIN_CLUSTER_FRAC:.0%} of {n_kept}, floor {_MIN_CLUSTER_N}). A cluster this "
            f"small is not a population: it is the model spending a component on outliers "
            f"or on the tail of a skewed feature. Treat these labels as unassigned, pin k "
            f"to the structure you can see, or cluster on a transformed feature."
        )
    if requested_k is None and bic and len(bic) > 1:
        ordered = sorted(bic.items(), key=lambda kv: kv[1])
        (k1, b1), (k2, b2) = ordered[0], ordered[1]
        if abs(b2 - b1) < _BIC_MARGIN:
            out.append(
                f"UNSTABLE k: BIC chose k={k1} over k={k2} by {abs(b2 - b1):.1f}, inside the "
                f"{_BIC_MARGIN:g} margin at which this choice is not meaningful. Auto-k is "
                f"reporting a preference the data does not support; pin k instead of reading "
                f"the number of clusters as a result."
            )
    return tuple(out)


def cluster(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    method: str = "gmm",
    k: int | None = None,
    k_max: int = 8,
    standardize: bool = True,
    random_state: int = 0,
    n_init: int = 1,
) -> ClusterResult:
    """Cluster events over a chosen feature set to expose subpopulations.

    Args:
        df: The per-event table.
        columns: Numeric feature columns to cluster on.
        method: ``"gmm"`` (:class:`sklearn.mixture.GaussianMixture`, full
            covariance) or ``"kmeans"`` (:class:`sklearn.cluster.KMeans`).
        k: Number of clusters. For GMM, ``None`` means *auto-select* by minimum
            BIC over ``1..k_max``. For k-means a value is required; ``None``
            defaults to ``2``.
        k_max: Upper bound on the GMM BIC search (clamped to the sample size).
        standardize: Z-score the features before fitting (default on -- without
            it a large-magnitude column dominates the distance/covariance).
        random_state: Seed for the estimator; fixed so results are reproducible.
        n_init: **GMM only** -- how many EM restarts each candidate ``k`` gets,
            the best kept by training log-likelihood. The default ``1`` is
            sklearn's, and it is what a single-shot exploratory clustering wants.
            Raise it when the BIC ladder itself is the answer: EM on a mixture
            with a heavy-tailed component lands in a different local optimum
            depending on where k-means seeded it, and a ladder built from
            one-restart fits can put its minimum at a ``k`` that a second restart
            would have beaten. (The Zn2-insulin subpopulation port drives this at
            ``n_init=3`` for exactly that reason -- see
            :func:`nanoscope.analysis.subpop_gmm.bic_ladder`.) k-means keeps its
            own fixed ``n_init=10`` and ignores this.

    Returns:
        A :class:`ClusterResult` whose ``labels`` are realigned to ``df.index``
        (``-1`` for rows dropped on a non-finite feature).

    Raises:
        ValueError: on an unknown ``method``, fewer than 2 valid rows, no
            feature columns, a non-positive ``k``, ``k`` greater than the number
            of valid rows, or a non-positive ``n_init``.
    """
    method = str(method).lower()
    if method not in ("gmm", "kmeans"):
        raise ValueError(f"unknown method {method!r}; expected 'gmm' or 'kmeans'")
    n_init = int(n_init)
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")

    X_raw, X, kept, cols = _prepare(df, columns, standardize=standardize)
    n = X.shape[0]

    if k is not None:
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > n:
            raise ValueError(
                f"k={k} exceeds the number of valid rows ({n}); "
                f"cannot fit more clusters than points"
            )

    bic: dict[int, float] | None = None

    if method == "gmm":
        if k is None:
            k_hi = int(min(max(int(k_max), 1), n))
            bic = {}
            best_k, best_bic = 1, np.inf
            for cand in range(1, k_hi + 1):
                gm = GaussianMixture(
                    n_components=cand,
                    covariance_type="full",
                    n_init=n_init,
                    random_state=random_state,
                )
                gm.fit(X)
                b = float(gm.bic(X))
                bic[cand] = b
                if b < best_bic:
                    best_k, best_bic = cand, b
            chosen_k = best_k
        else:
            chosen_k = k
        model = GaussianMixture(
            n_components=chosen_k,
            covariance_type="full",
            n_init=n_init,
            random_state=random_state,
        )
        labels_kept = model.fit_predict(X)
        if bic is None:
            bic = {chosen_k: float(model.bic(X))}
    else:  # kmeans
        chosen_k = 2 if k is None else k
        if chosen_k > n:
            raise ValueError(
                f"k={chosen_k} exceeds the number of valid rows ({n})"
            )
        model = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=10)
        labels_kept = model.fit_predict(X)

    labels, sizes, means = _summarise(df.index, kept, labels_kept, X_raw, cols)
    actual_k = len(sizes)
    notes = _cluster_warnings(sizes, int(n), bic, int(chosen_k), None if k is None else int(k))

    return ClusterResult(
        labels=labels,
        k=actual_k,
        method=method,
        columns=cols,
        sizes=sizes,
        means=means,
        bic=bic,
        chosen_k=int(chosen_k),
        n_kept=int(n),
        warnings=notes,
        meta={
            "standardize": bool(standardize),
            "random_state": int(random_state),
            "requested_k": None if k is None else int(k),
            "k_max": int(k_max),
            "n_init": int(n_init),
        },
    )


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def pca(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    n_components: int = 2,
    standardize: bool = True,
    random_state: int | None = None,
) -> PCAResult:
    """Project a per-event feature set to a few principal components.

    Args:
        df: The per-event table.
        columns: Numeric feature columns to project.
        n_components: Number of principal components to keep.
        standardize: Z-score the features first (default on -- PCA on raw units
            is dominated by whichever column has the largest variance, which is
            usually just a unit choice).
        random_state: Seed for sklearn's PCA. ``None`` (the default) keeps the
            previous behaviour exactly. It matters only when sklearn picks the
            *randomized* SVD solver, which its ``svd_solver="auto"`` rule does
            for tall-but-not-tall-enough matrices; on those inputs an unseeded
            PCA is not reproducible run to run. Callers that must reproduce a
            stored projection should pass a seed --
            :mod:`nanoscope.analysis.subpop_cluster` passes ``0``.

    Returns:
        A :class:`PCAResult` with ``scores`` aligned to ``kept_index``.

    Raises:
        ValueError: on fewer than 2 valid rows, no feature columns, a
            non-positive ``n_components``, ``n_components`` greater than
            ``min(n_kept, n_features)``, or feature columns with **no variance at
            all** -- see below.

    **A refusal is not a number.** Every selected column being constant leaves
    zero total variance, and ``explained_variance_ratio_`` is
    ``explained_variance_ / total_var`` -- a division by zero that sklearn hands
    back as ``[nan, nan]`` with a ``RuntimeWarning``. The Plots window printed
    that verbatim into its plot title, both axis labels and Details
    ("Explained variance: PC1 nan%, PC2 nan%  (cumulative nan%)") and exported it
    into a PNG. So this function refuses instead, in the same words and for the
    same reason :func:`nanoscope.analysis.stats.fit_dwell_exponential` refuses a
    zero-range column: a quantised, saturated or stuck upstream stage is a bug to
    be reported, not a statistic to be formatted. :func:`linear_fit` ("degenerate
    x") and :func:`robust_scale` (a constant column divided by 1.0, explicitly)
    already draw the same line; ``pca`` was the one function on this path with no
    such guard.

    A column that is constant while OTHERS vary is not fatal -- it simply
    contributes nothing to any component -- but it is not silent either: its name
    lands in ``meta['constant_columns']`` so the surface that offered the user the
    feature list can say which of their choices did no work.
    """
    X_raw, X, kept, cols = _prepare(df, columns, standardize=standardize)
    n, p = X.shape
    nc = int(n_components)
    if nc < 1:
        raise ValueError(f"n_components must be >= 1, got {nc}")
    limit = min(n, p)
    if nc > limit:
        raise ValueError(
            f"n_components={nc} exceeds min(n_rows={n}, n_features={p})={limit}"
        )

    # Measured on the RAW features, not on the standardised matrix: after
    # StandardScaler a constant column is a column of zeros whose ptp is zero for
    # the same reason, but the number worth naming in the message is the value the
    # user's column actually holds.
    const = [j for j in range(p) if not float(np.ptp(X_raw[:, j])) > 0.0]
    if len(const) == p:
        named = "; ".join(f"{cols[j]} = {X_raw[0, j]:g}" for j in const)
        raise ValueError(
            f"all {p} selected feature(s) are constant over the {n} rows with finite "
            f"values ({named}); there is no variance for PCA to explain, so no "
            f"principal component exists and no percentage can be reported. This "
            f"usually means a quantisation or detector bug upstream, or a curation "
            f"filter that left one clipped population."
        )

    model = _SKLearnPCA(n_components=nc, random_state=random_state)
    scores = model.fit_transform(X)

    return PCAResult(
        scores=np.asarray(scores, dtype=np.float64),
        kept_index=kept,
        explained_variance_ratio=np.asarray(
            model.explained_variance_ratio_, dtype=np.float64
        ),
        components=np.asarray(model.components_, dtype=np.float64),
        columns=cols,
        n_kept=int(n),
        meta={
            "standardize": bool(standardize),
            "n_components": nc,
            "random_state": random_state,
            # WHAT WAS DROPPED, COUNTED WHERE IT WAS DROPPED. `feature_matrix`
            # discards any row with a non-finite entry in any selected column, and
            # `n_kept` alone cannot tell a caller whether that happened -- the Plots
            # window printed a bare shrunken "n = 1960 kept" while its own clustering
            # pane, over the identical drop, said "1960 kept, 40 dropped (non-finite
            # features)". One voice: the count travels with the result.
            "n_input": int(df.shape[0]),
            "n_dropped": int(df.shape[0] - n),
            # Selected features that did no work (see the docstring).
            "constant_columns": [cols[j] for j in const],
            # The rank centred data of this shape can have, which is what makes a
            # "cumulative 100%" at n <= nc + 1 arithmetic rather than a finding. See
            # the Plots window's PCA caveat, and stats.r2_dof_caveat for the same
            # rule applied to a 2-point R^2 = 1.
            "max_rank": int(min(n - 1, p)),
        },
    )


# ---------------------------------------------------------------------------
# linear / power-law best-fit line (for the scatter view)
# ---------------------------------------------------------------------------


def linear_fit(
    x,
    y,
    *,
    log_log: bool = False,
    xlabel: str = "",
    ylabel: str = "",
) -> LinearFit:
    """Least-squares best-fit line through a scatter, plain or in log-log space.

    Two forms, one for each axis convention the scatter view offers:

    * **Plain** (``log_log=False``): fit ``y = slope*x + intercept`` directly.
    * **Power law** (``log_log=True``): fit ``log10(y) = slope*log10(x) +
      intercept``, i.e. a straight line in log-log space, which is the power law
      ``y = coefficient * x**exponent`` with ``exponent = slope`` and
      ``coefficient = 10**intercept``.

    Non-finite ``(x, y)`` pairs are dropped first; a log-log fit additionally
    drops any pair with a non-positive x or y (``log10`` needs both > 0). The fit
    itself is a degree-1 least squares via the normal equations -- numpy only, no
    scipy. ``R²`` is ``1 - SSres/SStot`` in the fitted space, ``nan`` when the
    response has no variance (a horizontal cloud, ``SStot == 0``).

    Args:
        x: The x values (any array-like).
        y: The y values (any array-like, same length as ``x``).
        log_log: Fit in ``log10``-``log10`` space (a power law) instead of the
            raw axes.
        xlabel: Optional x column name, carried onto the result for titles.
        ylabel: Optional y column name, carried onto the result for titles.

    Returns:
        A :class:`LinearFit`; for a log-log fit ``meta`` also carries the
        power-law ``exponent`` and ``coefficient``.

    Raises:
        ValueError: on fewer than 2 usable points, or a degenerate x (all equal,
            so the slope is undefined), with a message naming the cause.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must be the same length; got {x.size} and {y.size}"
        )

    good = np.isfinite(x) & np.isfinite(y)
    if log_log:
        # A log fit needs both axes strictly positive; a <=0 point has no log10.
        good &= (x > 0) & (y > 0)
    x, y = x[good], y[good]
    n = int(x.size)
    if n < 2:
        why = (
            "no positive (x, y) pairs for a log-log fit"
            if log_log else "fewer than 2 finite (x, y) pairs"
        )
        raise ValueError(f"linear_fit needs >= 2 valid points; {why} (n = {n})")

    xf = np.log10(x) if log_log else x
    yf = np.log10(y) if log_log else y

    if float(np.ptp(xf)) == 0.0:
        raise ValueError(
            "degenerate x: all x values are equal, so the slope is undefined"
        )

    # Degree-1 least squares via the normal equations (no scipy needed).
    slope, intercept = np.polyfit(xf, yf, 1)
    slope, intercept = float(slope), float(intercept)

    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    ss_res = float(np.sum((yf - (slope * xf + intercept)) ** 2))
    r2 = np.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    meta: dict[str, Any] = {"log_log": bool(log_log)}
    if log_log:
        meta["exponent"] = slope
        meta["coefficient"] = 10.0 ** intercept

    return LinearFit(
        slope=slope,
        intercept=intercept,
        r2=float(r2),
        n=n,
        xlabel=str(xlabel),
        ylabel=str(ylabel),
        log_log=bool(log_log),
        meta=meta,
    )


def line_points(
    fit: LinearFit, xmin: float, xmax: float, n: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the fitted curve over ``[xmin, xmax]`` in DATA units.

    For a plain fit this is the straight line ``slope*x + intercept`` over a
    linear grid; for a log-log fit it is the power law ``coefficient *
    x**exponent`` over a log-spaced grid (so it draws smoothly across decades).
    The output is always in raw data coordinates -- pyqtgraph's ``setLogMode``
    takes the log itself, so the same coords render correctly on log or linear
    axes.

    Args:
        fit: The :class:`LinearFit` to evaluate.
        xmin: Left edge of the x range (must be > 0 for a log-log fit).
        xmax: Right edge of the x range.
        n: Number of samples.

    Returns:
        ``(x, y)`` arrays of length ``n`` in data units.

    Raises:
        ValueError: on a non-positive ``xmin`` for a log-log fit, or ``n < 2``.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"line_points needs n >= 2; got {n}")
    xmin, xmax = float(xmin), float(xmax)
    if fit.log_log:
        if xmin <= 0 or xmax <= 0:
            raise ValueError(
                f"log-log line_points needs xmin, xmax > 0; got ({xmin}, {xmax})"
            )
        xs = np.geomspace(xmin, xmax, n)
    else:
        xs = np.linspace(xmin, xmax, n)
    return xs, fit.predict(xs)


# ---------------------------------------------------------------------------
# fit pass-throughs (single import surface for the plots layer)
# ---------------------------------------------------------------------------


def fit_exponential(
    dwell_s,
    *,
    t_min: float | None = None,
    n_components: int = 1,
    **kwargs: Any,
) -> FitResult:
    """Fit a (left-truncated) exponential to a dwell-like column.

    A thin delegate to :func:`nanoscope.analysis.stats.fit_dwell_exponential`;
    the censored-MLE machinery and its truncation correction are *not* copied
    here. The only convenience is ``t_min``: left as ``None`` it defaults to the
    smallest observed dwell, which the underlying fit requires as a positive
    duration.

    **Pass ``t_min``.** The default is a guess, not a measurement: the real
    detection floor is ``provenance['min_dwell_s']``, and substituting
    ``min(dwell_s)`` for it biases tau **SHORT**, because the estimator is
    ``tau_hat = mean(t) - t_min`` and the shortest OBSERVED dwell can only ever
    sit at or above the detector's own gate -- so the substitute floor is too
    high and over-subtracts. Measured -21.3% on a detection whose observed
    minimum was 3x its ``min_dwell_s``. (The opposite mistake, ignoring
    truncation altogether, is what biases tau long by ``+t_min``; do not confuse
    the two.) Either way it is invisible on the plot, because the histogram is
    missing exactly the same events. Because that substitution
    is silent-by-construction, this function refuses to be quiet about it: it
    raises a ``RuntimeWarning`` AND stamps the same sentence into
    ``FitResult.meta['warnings']``, so a caller that forgot still shows the user
    what happened. :func:`nanoscope.analysis.modules.dwell_fit_inputs` resolves
    ``t_min`` (and the right truncation, and filter-limited exclusion) properly
    from an EventSet; prefer it.

    Args:
        dwell_s: Observed dwell times in seconds (positive).
        t_min: Detection limit in seconds; defaults to ``min(dwell_s)`` with a
            warning.
        n_components: Number of exponential components.
        **kwargs: Forwarded to ``fit_dwell_exponential`` (``t_max``,
            ``efficiency``, ``n_boot``, ``seed``).

    Returns:
        The :class:`~nanoscope.analysis.stats.FitResult`; ``.predict`` overlays a
        density histogram, ``.params`` carry the untruncated ``tau_j``.
    """
    guessed = None
    if t_min is None:
        x = np.asarray(dwell_s, dtype=np.float64).ravel()
        finite_pos = x[np.isfinite(x) & (x > 0)]
        if finite_pos.size == 0:
            raise ValueError("no positive, finite dwell values to infer t_min from")
        t_min = float(finite_pos.min())
        guessed = NO_TMIN_WARNING.format(t_min=t_min)
        warnings.warn(guessed, RuntimeWarning, stacklevel=2)
    fit = fit_dwell_exponential(
        dwell_s, t_min=float(t_min), n_components=int(n_components), **kwargs
    )
    if guessed is not None:
        fit.meta["t_min_source"] = "observed_min"
        fit.meta["warnings"] = [guessed, *fit.meta.get("warnings", [])]
    else:
        fit.meta.setdefault("t_min_source", "explicit")
    return fit


def fit_gaussian_mixture(values, k: int = 1, **kwargs: Any) -> FitResult:
    """Fit a ``k``-component Gaussian mixture to an amplitude/depth-like column.

    A thin delegate to :func:`nanoscope.analysis.stats.fit_depth_gaussians`
    (hand-rolled EM); not reimplemented here. Note that the underlying fit
    expects non-negative magnitudes (the depth sign convention), so pass
    magnitudes for a signed amplitude column.

    Pass ``depth_floor`` (``threshold_out_pA``, divided for a fractional column by
    the largest per-event ``|baseline_pA|`` in the sample -- which
    :func:`nanoscope.analysis.modules.depth_floor_from_provenance` computes) if
    the column came from the detector: the mixture is untruncated, and a
    population sitting on the trigger's own floor comes back with its mean
    biased high and its sigma biased low unless the fit is told to say so.

    Args:
        values: The column to fit (non-negative).
        k: Number of Gaussian components.
        **kwargs: Forwarded to ``fit_depth_gaussians`` (``depth_floor``,
            ``n_boot``, ``seed``).

    Returns:
        The :class:`~nanoscope.analysis.stats.FitResult` with ``mu_j``,
        ``sigma_j``, ``w_j`` per component; ``.predict`` gives the mixture
        density.
    """
    return fit_depth_gaussians(values, n_components=int(k), **kwargs)
