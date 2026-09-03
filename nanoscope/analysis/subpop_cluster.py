"""Per-scope PCA and exploratory clustering of per-event waveform features.

This is a **port**, not a design. Every number here is pinned to the owner's own
run of ``Zn2Insulin_StudentInference.ipynb`` (cells 51-56), whose stored output
``CSVData/event_clustering_features.csv`` (1488 x 29) carries the cluster labels
themselves -- ``hdbscan``/``hdbscan_{2,10,18,30}`` and
``kmeans3``/``kmeans3_{2,10,18,30}`` -- so "does this port work?" is a decidable
question rather than a matter of taste. It does: all ten label columns reproduce
at **exact equality, 1488/1488 rows**, and the per-scope loading tables
reproduce his ``pca_loadings_*.csv`` to the three decimals he rounded them to.
See ``tests/analysis/test_subpop_cluster.py``.

What the stage does, in order
-----------------------------
1. **Winsorise, then scale** -- :func:`nanoscope.analysis.populations.robust_scale`,
   imported rather than reimplemented (see below).
2. **PCA within the scope.** ``min(n_features, 10)`` components; the number kept
   is the first count whose cumulative explained variance reaches 95 %.
3. **Cluster the kept PCs** with HDBSCAN (density-based; free to return no
   clusters at all) and with k-means at a fixed k=3 (a partition; always returns
   three). Running both is the design: where they disagree, the scope is a
   continuum that k-means is slicing anyway.

"Scope" is the unit of re-fitting: the pooled table, then each recording alone.
Within each scope the winsorisation percentiles, the median, the IQR *and* the
PCA basis are re-fit from that scope's rows only -- nothing is fit on the pooled
data and applied to a subset. The notebook's own comment gives the reason: the
pooled PCA is dominated by cross-recording variance and buries exactly the
within-recording modes the subpopulation analysis is hunting for.

Four things that look like bugs and are not
-------------------------------------------
* **HDBSCAN must come from sklearn.** The notebook's ``_make_hdbscan`` prefers
  ``sklearn.cluster.HDBSCAN`` and falls back to the standalone ``hdbscan``
  package. sklearn was installed, so the sklearn branch is what produced the
  oracle -- and the two packages do *not* agree. Measured on the owner's own
  scaled matrices, sklearn reproduces all five scopes exactly (adjusted Rand
  1.0000 *and* label-for-label identity), while the standalone package gives:

  =======  ================  =============  ==========================
  scope    exact agreement   adjusted Rand  noise points (vs oracle)
  =======  ================  =============  ==========================
  ALL      0.9819            0.9382         294 vs 319
  zn_30h   0.2138            0.9772         241 vs 229
  =======  ================  =============  ==========================

  Read those two columns together, because they say different things. On
  ``zn_30h`` the *partition* is nearly the oracle's (Rand 0.977); exact
  agreement collapses to 0.21 because clusters 0 and 1 come back with their
  names swapped. On the pooled scope the disagreement is real membership --
  25 events change between noise and cluster. Either way the stored label
  columns are not reproduced, which is what a port has to do. So
  :func:`hdbscan_labels` **raises rather than falling back**: silently swapping
  in an estimator that gives different answers is precisely the failure this
  project has already paid for once.
* **Three of the thirteen features are perfectly collinear inside any one
  recording.** ``depth_sust_pA``, ``f_sust`` and ``snr`` are one quantity over
  three per-recording constants (1, ``I0``, and the global MAD), so within a
  scope their correlation is exactly 1.0 and their PCA loading rows are
  identical -- visible in the owner's own ``pca_loadings_30.csv``. Effective
  within-scope dimensionality is 11, not 13. Do not "fix" it by dropping
  features: it is *why* the downstream cross-validation lands on ``n_PCs = 11``.
* **Every within-recording scope is 100 % HDBSCAN noise.** ``hdbscan_2``,
  ``hdbscan_10`` and ``hdbscan_18`` are all ``-1``. The notebook calls this a
  real finding -- those scopes are continua with no density modes -- and not a
  knob to turn down until clusters appear.
* **``min_samples=10`` is not the default.** HDBSCAN defaults it to
  ``min_cluster_size``; the notebook records that doing so labelled 100 % of a
  real zn_30h event set as noise.

The alignment hazard this module exists to remove
--------------------------------------------------
The notebook accumulates labels with ``CLUST.loc[_sub.index, _col] = _hl`` -- a
*positional* numpy array assigned into a *label*-indexed slice. It is correct
only because the subset happens to preserve the parent's original ``RangeIndex``;
any filter, sort or ``reset_index`` in between silently misaligns every label
while still producing a perfectly plausible-looking picture. Here
:meth:`ScopeClustering.labels` carries its own ``pd.Index`` and
:meth:`SubpopClustering.label_table` joins on index labels, so correctness stops
depending on an accident of row order.

Reuse, and the one thing that is genuinely new
-----------------------------------------------
PCA, k-means and the robust scaler all live in
:mod:`nanoscope.analysis.populations` and are *called* here, not copied. What
this module adds is the part the app did not have: the scope loop, the
cumulative-variance PC count, the info-share loading table, and the HDBSCAN
binding.

The thirteen features are **derived** -- computed on demand from trace +
EventSet the way :mod:`nanoscope.analysis.event_shape` does. Nothing here
touches the frozen 20-column EventSet schema, and note that the schema's ``snr``
is a different quantity from this table's ``snr`` (schema: depth over baseline
noise RMS; here: sustained depth over the recording's global MAD). Same word,
different number; do not join them.

sklearn/numpy/pandas only. t-SNE and UMAP are figure aids with no stored oracle;
:func:`tsne_embedding` and :func:`umap_embedding` say what that costs you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from nanoscope.analysis import populations
from nanoscope.analysis.cancel import CancelFlag, check_cancelled
from nanoscope.analysis.event_features import CLUST_FEATS as _CLUST_FEATS

__all__ = [
    "CLUST_FEATS",
    "SCOPE_ALL",
    "SCOPE_COLUMN",
    "MIN_SCOPE_N",
    "MAX_PCA_COMPONENTS",
    "VAR_TARGET",
    "KMEANS_K",
    "HDBSCAN_MIN_SAMPLES",
    "HDBSCAN_MIN_CLUSTER_FLOOR",
    "HDBSCAN_NOISE_LABEL",
    "RANDOM_STATE",
    "ScopePCA",
    "ScopeClustering",
    "SubpopClustering",
    "feature_values",
    "scale_features",
    "scale_and_pca",
    "loading_table",
    "LOADING_DECIMALS",
    "iter_scopes",
    "hdbscan_min_cluster_size",
    "hdbscan_labels",
    "kmeans_labels",
    "cluster_scopes",
    "tsne_perplexity",
    "umap_neighbors",
    "EMBEDDING_MIN_COMPONENTS",
    "embedding_refusal",
    "tsne_embedding",
    "umap_embedding",
]


# ---------------------------------------------------------------------------
# the pinned contract
# ---------------------------------------------------------------------------

#: The 13 waveform features, **in the order the notebook's ``CLUST_FEATS`` lists
#: them**. This is NOT the column order of ``event_clustering_features.csv``:
#: the CSV inherits the ``_event_features`` dict order, which puts ``entropy``
#: 11th, while the PCA input puts it last. Feature order permutes the rows of
#: the loading table and, through the SVD's sign convention, can flip the sign
#: of a component -- so the two orders are not interchangeable. Use this one.
#:
#: RE-EXPORTED FROM :mod:`nanoscope.analysis.event_features`, not restated here.
#: It used to be a second literal with the same thirteen names, which is the
#: shape of seam that fails silently: the module that *produces* the columns and
#: the module that *consumes* them each held their own copy, so reordering one
#: (the sign-flip hazard the paragraph above warns about) would have left the
#: other agreeing with the notebook and the two disagreeing with each other,
#: with no exception anywhere -- the PCA would simply read the columns in a
#: different order and return a differently-signed basis. One object now, so
#: ``subpop_cluster.CLUST_FEATS is event_features.CLUST_FEATS``, and the chain
#: on to ``subpop_gmm.SUBPOP_FEATURES`` is single-sourced end to end.
CLUST_FEATS: tuple[str, ...] = _CLUST_FEATS

#: Name of the pooled scope. Its label columns carry no suffix (``hdbscan``,
#: ``kmeans3``); every other scope is suffixed (``hdbscan_30``).
SCOPE_ALL = "ALL"

#: Column the scopes are cut on. Compared **as strings**, which is what makes
#: the scope order lexicographic (see :func:`iter_scopes`).
SCOPE_COLUMN = "incubation_h"

#: Scopes smaller than this are skipped. An HDBSCAN floor: ``min_cluster_size``
#: is at least 20, and asking for density modes among 40 points is not a
#: question with an answer.
MIN_SCOPE_N = 60

#: PCA is capped here even though there are 13 features. The cap is the
#: notebook's, and it interacts with :data:`VAR_TARGET` -- see
#: :func:`scale_and_pca` on the count that can overrun it.
MAX_PCA_COMPONENTS = 10

#: Cumulative explained-variance target that decides how many PCs are kept.
VAR_TARGET = 0.95

#: k for the fixed-k reference partition. Imposed, not discovered: k-means
#: partitions a continuum as happily as it partitions clusters, which is
#: exactly why it is run beside HDBSCAN rather than instead of it.
KMEANS_K = 3

#: Deliberately not the sklearn default (which is ``min_cluster_size``).
HDBSCAN_MIN_SAMPLES = 10

#: ``min_cluster_size`` is ``max(this, n // 100)`` -- integer floor division.
HDBSCAN_MIN_CLUSTER_FLOOR = 20

#: HDBSCAN's own label for "in no cluster". It collides numerically with
#: :data:`nanoscope.analysis.populations.DROPPED_LABEL`, which makes the
#: entirely different claim "this row had a non-finite feature and was dropped".
#: HDBSCAN labels therefore never travel through ``populations``' realignment.
HDBSCAN_NOISE_LABEL = -1

#: **The** seed. Every seeded step in this module takes its seed from this one
#: default, threaded through :func:`cluster_scopes`; two runs give identical
#: output. (HDBSCAN takes no seed; it is deterministic given its input.)
RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# 2. feature extraction + robust scaling (the scaler itself is imported)
# ---------------------------------------------------------------------------


def feature_values(
    df: pd.DataFrame,
    columns: Sequence[str] = CLUST_FEATS,
) -> np.ndarray:
    """Pull the feature columns as a float matrix, **keeping every row**.

    This is the one place this module deliberately does not use
    :func:`nanoscope.analysis.populations.feature_matrix`. That function drops
    any row with a non-finite feature, which is the right policy when a fit
    would otherwise be undefined -- but here the row count is part of the
    contract. A scope's ``n`` sets HDBSCAN's ``min_cluster_size``, and the label
    table has to cover every event that went in; a quietly shorter matrix would
    change the clustering itself, not merely the bookkeeping. Non-finite entries
    are handed to :func:`scale_features`, which lands them on the scaled median.

    Args:
        df: The per-event table.
        columns: Feature columns to extract, in order.

    Returns:
        An ``(len(df), len(columns))`` ``float64`` array; non-numeric entries
        become NaN.

    Raises:
        ValueError: if ``columns`` is empty or names a column absent from ``df``.
    """
    cols = [str(c) for c in columns]
    if not cols:
        raise ValueError("need at least one feature column; got none")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature column(s) {missing} not in the table; "
            f"available: {list(df.columns)}"
        )
    values = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return values.to_numpy(dtype=np.float64).reshape(len(df), len(cols))


def scale_features(
    df: pd.DataFrame,
    columns: Sequence[str] = CLUST_FEATS,
) -> np.ndarray:
    """Extract the features and robust-scale them: winsorise 1-99, median/IQR.

    A two-line composition of :func:`feature_values` and
    :func:`nanoscope.analysis.populations.robust_scale`, and it exists to make
    that second call the *only* scaler in the pipeline. The arithmetic --
    clip to the per-column 1st/99th percentile, then centre on the median and
    divide by the IQR **of the clipped column** -- lives there and is verified
    bit-identical to the notebook's inline expression on all five scopes and on
    the degenerate columns the owner's data never produces (all-NaN, constant,
    infinite, zero-IQR-after-clipping).

    One correction to the folklore around this step, measured rather than
    assumed: at 1-99 % the clip-then-measure order is **not** what makes the
    port reproduce. Winsorising 1 % of each tail cannot move a quartile or a
    median, so on the owner's feature matrix clipping first and clipping last
    agree to 1.3e-15 -- and the median and IQR come out bitwise identical. The
    orders only diverge once the clip reaches inside the IQR: measured on the
    same matrix, they are identical at 1-99, 5-95, 10-90 and 20-80, and first
    differ at 26-74 (max 8.6e-02). So preserve the order because it is the
    notebook's and because it is the right convention at any limit -- but if a
    port disagrees with the oracle, this is not where to look.

    The downstream GMM stage needs the same scaling over all 13 columns with no
    PCA truncation; it should call this, not grow a third copy.

    Args:
        df: The per-event table for one scope.
        columns: Feature columns, in order.

    Returns:
        An ``(n, n_features)`` ``float64`` array, finite everywhere.
    """
    return populations.robust_scale(feature_values(df, columns))


# ---------------------------------------------------------------------------
# 3. per-scope PCA
# ---------------------------------------------------------------------------


@dataclass
class ScopePCA:
    """A scope's robust-scaled PCA: the basis, the variance profile, the scores.

    Attributes:
        scope: Scope name (:data:`SCOPE_ALL` or a stringified scope value).
        index: ``df.index`` labels of the rows, in row order. Carrying this is
            what lets labels be joined back by *label* rather than by position.
        columns: Feature names, in PCA input order.
        explained_variance_ratio: Variance fraction on each fitted component.
        components: ``(n_components, n_features)`` eigenvectors, unit-norm rows.
        n_kept: Number of leading PCs that reach :data:`VAR_TARGET`.
        scores: ``(n, n_kept)`` projection -- **the matrix the clusterers see**.
        scores_full: ``(n, n_components)`` projection, kept for diagnostics.
        meta: The parameters actually used, including ``n_kept_unclamped``.
    """

    scope: str
    index: pd.Index
    columns: tuple[str, ...]
    explained_variance_ratio: np.ndarray
    components: np.ndarray
    n_kept: int
    scores: np.ndarray
    scores_full: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of events in the scope."""
        return int(self.scores.shape[0])

    @property
    def cumulative_variance(self) -> np.ndarray:
        """Running total of :attr:`explained_variance_ratio`."""
        return np.cumsum(self.explained_variance_ratio)

    @property
    def variance_captured(self) -> float:
        """Fraction of variance the kept PCs actually capture."""
        return float(self.cumulative_variance[self.n_kept - 1])

    def __repr__(self) -> str:
        return (
            f"<ScopePCA {self.scope!r} n={self.n} kept={self.n_kept}"
            f"/{len(self.explained_variance_ratio)} var={self.variance_captured:.3f}>"
        )


def scale_and_pca(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = CLUST_FEATS,
    scope: str = SCOPE_ALL,
    var_target: float = VAR_TARGET,
    max_components: int = MAX_PCA_COMPONENTS,
    random_state: int | None = RANDOM_STATE,
) -> ScopePCA:
    """Robust-scale a scope's features and project them onto its own PCs.

    Fit and transform happen on the same rows by design -- this is a
    *description* of one scope, not a model to be carried elsewhere.

    The number of PCs kept is ``searchsorted(cumsum(evr), var_target) + 1``: the
    first count whose cumulative variance reaches the target. Two edges are
    worth knowing about.

    **The count can overrun the components that exist.** PCA is capped at
    :data:`MAX_PCA_COMPONENTS` (10) while there are 13 features, so on a noisy
    recording the cumulative variance can top out below ``var_target``;
    ``searchsorted`` then returns one past the end, and the notebook's
    ``Z[:, :nk]`` silently hands back fewer columns than ``nk`` claims -- a
    latent misreport that its own data never triggers. Here ``n_kept`` is
    clamped to the number of components actually fitted, so it is always the
    true width of :attr:`ScopePCA.scores`. The clamp cannot move the owner's
    numbers (his five scopes reach 99.88-99.99 % cumulative variance by PC10 and
    keep 4-6 PCs, nowhere near the cap) and ``meta["n_kept_unclamped"]`` records
    what the unguarded expression would have said.

    **The seed is threaded even though it is currently a no-op.** sklearn's
    ``svd_solver="auto"`` picks a deterministic solver for matrices this shape,
    so ``random_state`` changes nothing on the owner's data -- but it stops
    being a no-op the moment a scope is large enough for the randomized solver,
    and an unseeded PCA there would not reproduce run to run.

    Args:
        df: The per-event feature table for **one scope**.
        columns: Feature columns, in order. See :data:`CLUST_FEATS`.
        scope: Name to record on the result.
        var_target: Cumulative explained-variance target, in [0, 1].
        max_components: Cap on the number of components fitted.
        random_state: Seed for sklearn's PCA.

    Returns:
        A :class:`ScopePCA`.

    Raises:
        ValueError: on an empty feature list, a missing column, fewer than 2
            rows, or a ``var_target`` outside [0, 1].
    """
    cols = tuple(str(c) for c in columns)
    if not 0.0 <= float(var_target) <= 1.0:
        raise ValueError(f"var_target must be in [0, 1], got {var_target}")
    if len(df) < 2:
        raise ValueError(f"scope {scope!r} has {len(df)} row(s); PCA needs at least 2")

    Xs = scale_features(df, cols)
    n_comp = int(min(len(cols), int(max_components), Xs.shape[0]))

    scaled = pd.DataFrame(Xs, index=df.index, columns=list(cols))
    res = populations.pca(
        scaled,
        cols,
        n_components=n_comp,
        standardize=False,
        random_state=random_state,
    )

    evr = np.asarray(res.explained_variance_ratio, dtype=np.float64)
    unclamped = int(np.searchsorted(np.cumsum(evr), float(var_target)) + 1)
    n_kept = max(1, min(unclamped, n_comp))

    return ScopePCA(
        scope=str(scope),
        index=res.kept_index,
        columns=cols,
        explained_variance_ratio=evr,
        components=np.asarray(res.components, dtype=np.float64),
        n_kept=n_kept,
        scores=np.asarray(res.scores[:, :n_kept], dtype=np.float64),
        scores_full=np.asarray(res.scores, dtype=np.float64),
        meta={
            "var_target": float(var_target),
            "n_components": n_comp,
            "random_state": random_state,
            "n_kept_unclamped": unclamped,
        },
    )


#: Decimal places the owner's ``pca_loadings_*.csv`` are written to:
#: ``(loadings, info_share)``. Not merely cosmetic -- see :func:`loading_table`.
LOADING_DECIMALS = (3, 4)


def loading_table(
    result: ScopePCA,
    *,
    decimals: tuple[int, int] | None = LOADING_DECIMALS,
) -> pd.DataFrame:
    """Feature x PC loading table with the info-share ranking, sorted.

    ``info_share`` for a feature is ``sum_i evr_i * loading_i(feature)**2`` over
    the kept PCs. Because each PC's eigenvector is unit-norm, the shares sum to
    exactly the variance the kept PCs capture -- so the column reads as "how
    much of the retained information does this feature carry", and sorting by it
    ranks the features by that.

    The loadings are the **eigenvector entries** (sklearn's ``components_``),
    not correlation loadings (``components_ * sqrt(explained_variance_)``). The
    two rank features differently, and the owner's stored tables are the former.

    **Rounding happens before sorting, and it changes the answer.** The notebook
    builds this table from ``load.round(3)`` and ``info.round(4)`` and sorts the
    rounded column, so ``decimals`` defaults to reproducing that. It is not a
    display detail:

    * In the pooled scope ``entropy`` (0.0060028) genuinely outranks
      ``t_peak_frac`` (0.0059575), but both round to 0.0060 and the owner's
      table lists ``t_peak_frac`` first. Sorting unrounded would silently
      reorder his table.
    * Inside a single recording ``depth_sust_pA``, ``f_sust`` and ``snr`` are
      collinear, so their info shares agree -- but only to about 1e-15, *not*
      bitwise (measured: they differ in the last two or three ulps on every
      scope). Unrounded, their relative order is decided by floating-point
      noise, which is no ordering at all. Rounding makes the tie exact and the
      tie-break below makes it meaningful.

    Ties are then broken by :data:`CLUST_FEATS` position, explicitly, via
    ``lexsort`` -- rather than by inheriting whichever tie behaviour the
    installed pandas' default sort happens to have.

    Pass ``decimals=None`` for full-precision values. Be aware that this
    reintroduces the noise-ordered collinear rows described above; the numbers
    are better and the ranking among near-tied features is worse.

    Args:
        result: A fitted :class:`ScopePCA`.
        decimals: ``(loading_places, info_share_places)``, or ``None`` to skip
            rounding.

    Returns:
        A ``DataFrame`` indexed by feature name, with one ``PC{i}`` column per
        kept PC plus ``info_share``, sorted by ``info_share`` descending.
    """
    nk = int(result.n_kept)
    load = pd.DataFrame(
        result.components[:nk].T,
        index=list(result.columns),
        columns=[f"PC{i + 1}" for i in range(nk)],
    )
    info = (load**2 * result.explained_variance_ratio[:nk]).sum(axis=1)
    if decimals is not None:
        load = load.round(int(decimals[0]))
        info = info.round(int(decimals[1]))

    table = load.copy()
    table["info_share"] = info
    # Primary key last: descending info_share, ties by CLUST_FEATS position.
    order = np.lexsort((np.arange(len(table)), -info.to_numpy(dtype=np.float64)))
    return table.iloc[order]


# ---------------------------------------------------------------------------
# the scope iterator
# ---------------------------------------------------------------------------


def iter_scopes(
    df: pd.DataFrame,
    *,
    scope_column: str = SCOPE_COLUMN,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``("ALL", df)``, then one ``(scope, subset)`` per distinct value.

    Scope values are compared and sorted **as strings**, so the order is
    lexicographic rather than numeric: for the Zn2 series that is
    ``ALL, "10", "18", "2", "30"`` -- exactly the order the oracle's
    ``hdbscan_10, hdbscan_18, hdbscan_2, hdbscan_30`` suffixes appear in.
    Sorting numerically yields the same columns in a different order, which
    breaks any positional comparison against the stored table.

    Each subset keeps the parent's index labels, which is what makes the join in
    :meth:`SubpopClustering.label_table` correct by construction rather than by
    coincidence.

    Args:
        df: The per-event feature table.
        scope_column: Column whose distinct values define the scopes.

    Yields:
        ``(scope_name, subset)`` pairs.

    Raises:
        ValueError: if ``scope_column`` is not a column of ``df``.
    """
    if scope_column not in df.columns:
        raise ValueError(
            f"scope column {scope_column!r} not in the table; "
            f"available: {list(df.columns)}"
        )
    yield SCOPE_ALL, df
    as_str = df[scope_column].astype(str)
    for value in sorted(as_str.unique()):
        yield str(value), df[as_str == str(value)]


# ---------------------------------------------------------------------------
# 4-5. the two clusterers
# ---------------------------------------------------------------------------


def hdbscan_min_cluster_size(n: int) -> int:
    """``max(20, n // 100)`` -- integer floor division, not a true fraction.

    Invisible on the owner's data (all five scopes land on the floor of 20) and
    visible everywhere else: ``min_cluster_size`` is an integer parameter, so at
    n = 2599 the floor form gives 25 where an ``n/100`` form gives 25.99.
    """
    return int(max(HDBSCAN_MIN_CLUSTER_FLOOR, int(n) // 100))


def _hdbscan_estimator(n: int, *, min_samples: int = HDBSCAN_MIN_SAMPLES):
    """Build the sklearn HDBSCAN estimator, or explain why we will not guess.

    Deliberately **no fallback** to the standalone ``hdbscan`` package -- see
    the module docstring. The two label differently, the oracle came from
    sklearn, and a fallback that quietly changes the answer is worse than an
    import error that says so out loud.
    """
    try:
        from sklearn.cluster import HDBSCAN as _HDBSCAN
    except ImportError as exc:  # pragma: no cover - scikit-learn>=1.3 is a hard dep
        raise ImportError(
            "sklearn.cluster.HDBSCAN is required (scikit-learn >= 1.3). The "
            "standalone `hdbscan` package is NOT an acceptable substitute: on "
            "the owner's zn_30h scope it reproduces only 21% of the stored "
            "labels, with a different noise set as well as a relabelling."
        ) from exc
    return _HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size(n),
        min_samples=int(min_samples),
    )


def hdbscan_labels(
    Z: np.ndarray,
    *,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """Density-cluster the kept PCs. ``-1`` means noise, not "dropped".

    ``min_cluster_size`` is derived from the number of rows in ``Z``; it is not
    a free parameter. HDBSCAN takes no seed and is deterministic given input.

    Args:
        Z: ``(n, n_kept)`` scores from :func:`scale_and_pca`.
        min_samples: Conservatism of the density estimate.

    Returns:
        ``(n,)`` int labels, :data:`HDBSCAN_NOISE_LABEL` for noise.

    Raises:
        ValueError: if ``Z`` is not 2-D.
        ImportError: if sklearn's HDBSCAN is unavailable.
    """
    Za = np.asarray(Z, dtype=np.float64)
    if Za.ndim != 2:
        raise ValueError(f"expected a 2-D score matrix, got shape {Za.shape}")
    est = _hdbscan_estimator(Za.shape[0], min_samples=min_samples)
    return np.asarray(est.fit_predict(Za), dtype=np.int64)


def kmeans_labels(
    Z: np.ndarray,
    *,
    k: int = KMEANS_K,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Fixed-k reference partition, delegated to :func:`populations.cluster`.

    That function already runs ``KMeans(n_clusters=k, n_init=10,
    random_state=...)``, which is the notebook's call exactly -- so this is a
    binding, not a second implementation.

    ``Z`` is finite by construction (``robust_scale`` guarantees it), so
    ``populations.cluster`` drops no rows and its realigned label series is a
    straight positional array here. The guard below asserts that rather than
    trusting it, because a ``DROPPED_LABEL`` of ``-1`` slipping through would be
    indistinguishable from a cluster id downstream.

    Args:
        Z: ``(n, n_kept)`` scores from :func:`scale_and_pca`.
        k: Number of clusters.
        random_state: Seed.

    Returns:
        ``(n,)`` int labels in ``0..k-1``.

    Raises:
        ValueError: if ``Z`` is not 2-D.
    """
    Za = np.asarray(Z, dtype=np.float64)
    if Za.ndim != 2:
        raise ValueError(f"expected a 2-D score matrix, got shape {Za.shape}")
    cols = [f"pc{i + 1}" for i in range(Za.shape[1])]
    frame = pd.DataFrame(Za, columns=cols)
    res = populations.cluster(
        frame,
        cols,
        method="kmeans",
        k=int(k),
        standardize=False,
        random_state=int(random_state),
    )
    labels = res.labels.to_numpy(dtype=np.int64)
    if np.any(labels == populations.DROPPED_LABEL):
        raise RuntimeError(
            "k-means dropped rows from a matrix that should be finite; "
            "robust_scale's finiteness guarantee has been broken"
        )
    return labels


def _silhouette(Z: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette score, or ``None`` where it is undefined rather than a fake 0."""
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(labels) <= len(uniq):
        return None
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(Z, labels))


# ---------------------------------------------------------------------------
# the per-scope driver
# ---------------------------------------------------------------------------


@dataclass
class ScopeClustering:
    """Everything computed for one scope: the projection and both clusterings.

    Attributes:
        scope: Scope name.
        pca: The scope's own :class:`ScopePCA`.
        loadings: The scope's :func:`loading_table`.
        hdbscan: ``(n,)`` HDBSCAN labels, ``-1`` = noise.
        kmeans3: ``(n,)`` k-means labels.
        hdbscan_silhouette: Silhouette over the non-noise points, or ``None``
            when HDBSCAN found fewer than two clusters -- which is the *normal*
            outcome for a within-recording scope here, not a failure.
        kmeans3_silhouette: Silhouette over all points.
        meta: The HDBSCAN parameters and backend actually used.
    """

    scope: str
    pca: ScopePCA
    loadings: pd.DataFrame
    hdbscan: np.ndarray
    kmeans3: np.ndarray
    hdbscan_silhouette: float | None = None
    kmeans3_silhouette: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def index(self) -> pd.Index:
        """Index labels of this scope's rows, in row order."""
        return self.pca.index

    @property
    def n(self) -> int:
        """Number of events in the scope."""
        return self.pca.n

    @property
    def suffix(self) -> str:
        """Label-column suffix: empty for the pooled scope, ``_<scope>`` else."""
        return "" if self.scope == SCOPE_ALL else f"_{self.scope}"

    @property
    def n_hdbscan_clusters(self) -> int:
        """Number of non-noise HDBSCAN clusters."""
        return int(len(set(self.hdbscan[self.hdbscan >= 0].tolist())))

    @property
    def hdbscan_noise_fraction(self) -> float:
        """Fraction of events HDBSCAN called noise. 1.0 reads as "a continuum"."""
        return float(np.mean(self.hdbscan < 0)) if self.n else float("nan")

    def labels(self, method: str) -> pd.Series:
        """Labels for ``"hdbscan"`` or ``"kmeans3"``, indexed by event label.

        Raises:
            ValueError: on an unknown ``method``.
        """
        if method == "hdbscan":
            values = self.hdbscan
        elif method == "kmeans3":
            values = self.kmeans3
        else:
            raise ValueError(
                f"unknown method {method!r}; expected 'hdbscan' or 'kmeans3'"
            )
        return pd.Series(values, index=self.index, name=method + self.suffix)

    def feature_medians(self, df: pd.DataFrame, method: str = "hdbscan") -> pd.DataFrame:
        """Per-cluster median of each feature -- how a cluster is read physically.

        Args:
            df: The feature table this scope was cut from (any superset of the
                scope's rows; it is joined by index label, not by position).
            method: ``"hdbscan"`` or ``"kmeans3"``.

        Returns:
            A ``DataFrame`` indexed by cluster id, one column per feature.
        """
        lab = self.labels(method)
        sub = df.loc[lab.index, list(self.pca.columns)]
        return sub.groupby(lab.to_numpy()).median()

    def __repr__(self) -> str:
        return (
            f"<ScopeClustering {self.scope!r} n={self.n} "
            f"hdbscan={self.n_hdbscan_clusters}cl/"
            f"{self.hdbscan_noise_fraction:.0%}noise kmeans3={KMEANS_K}>"
        )


@dataclass
class SubpopClustering:
    """The whole per-scope sweep: one :class:`ScopeClustering` per scope.

    Attributes:
        scopes: Scope name -> result, in :func:`iter_scopes` order.
        skipped: Scope name -> row count, for scopes below :data:`MIN_SCOPE_N`.
            Recorded rather than dropped silently: "this recording had too few
            events to cluster" is a finding, not an absence.
        index: Index of the table the sweep ran on.
        meta: The seeds and parameters actually used.
    """

    scopes: dict[str, ScopeClustering]
    skipped: dict[str, int] = field(default_factory=dict)
    # `pd.Index` cannot be the factory itself: `pd.Index()` with no argument
    # raises TypeError rather than building an empty index.
    index: pd.Index = field(default_factory=lambda: pd.Index([]))
    meta: dict[str, Any] = field(default_factory=dict)

    def label_table(self) -> pd.DataFrame:
        """Assemble every scope's labels into one table, joined **by index label**.

        Column order is ``hdbscan``, then the per-scope HDBSCAN columns in scope
        order, then the same for ``kmeans3`` -- matching the oracle's layout.

        Dtype is float, and the two kinds of "no cluster" stay distinguishable:
        ``NaN`` means *this event is not in this scope*, ``-1.0`` means *this
        event is in this scope and HDBSCAN called it noise*. Collapsing them
        (with a ``fillna(-1)``, say) would silently convert every out-of-scope
        row into a noise point and roughly quadruple the apparent noise rate.

        Returns:
            A ``DataFrame`` indexed like the source table, one float column per
            (method, scope).
        """
        out = pd.DataFrame(index=self.index)
        for method in ("hdbscan", "kmeans3"):
            for res in self.scopes.values():
                series = res.labels(method)
                out[str(series.name)] = pd.Series(
                    series.to_numpy(dtype=np.float64), index=series.index
                )
        return out

    def __repr__(self) -> str:
        return (
            f"<SubpopClustering scopes={list(self.scopes)} "
            f"skipped={list(self.skipped)} n={len(self.index)}>"
        )


def cluster_scopes(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = CLUST_FEATS,
    scope_column: str = SCOPE_COLUMN,
    min_scope_n: int = MIN_SCOPE_N,
    var_target: float = VAR_TARGET,
    max_components: int = MAX_PCA_COMPONENTS,
    k: int = KMEANS_K,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    random_state: int = RANDOM_STATE,
    cancel: CancelFlag | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> SubpopClustering:
    """Run the whole exploratory sweep: scale, project and cluster, per scope.

    One projection per scope. The notebook re-derives the identical projection
    in each of its five analysis cells (PCA overview, HDBSCAN, k-means, t-SNE,
    UMAP); computed once and shared, it is the same answer for a fifth of the
    work.

    Cancellation is polled between stages within each scope, which is the honest
    granularity: the cost here is one PCA, one HDBSCAN and one k-means fit per
    scope, and none of the three exposes a progress hook to poll inside.

    ROW ORDER IS PART OF THE INPUT. Every seeded step here is deterministic given
    the same rows in the same order, and the owner's ten label columns reproduce
    exactly -- but the integer labels are NOT invariant to a permutation of the
    rows. Measured on his 1488-event table, permuting the rows and mapping back:
    every k-means partition is identical (adjusted Rand 1.000000) while the
    numbering changes (``kmeans3`` agrees on 2% of rows), and ``hdbscan_30``
    moves ONE event across the noise boundary (ARI 0.996942). k-means++ draws its
    seeds by row position, and HDBSCAN's mutual-reachability tree breaks exact
    distance ties by position.

    So: the SCIENCE is order-invariant, the LABELS are not. Reproducing his
    label columns needs his row ordering -- ``(tag, onset)``, which is what
    :func:`~nanoscope.analysis.subpop_report.feature_table` emits. Sort or filter
    the table upstream and expect the same clusters under different names.

    Args:
        df: The per-event feature table, carrying the feature columns and
            ``scope_column``.
        columns: Feature columns, in PCA input order.
        scope_column: Column defining the scopes.
        min_scope_n: Scopes smaller than this are recorded and skipped.
        var_target: Cumulative-variance target for the PC count.
        max_components: Cap on components fitted per scope.
        k: k for the fixed-k partition.
        min_samples: HDBSCAN's ``min_samples``.
        random_state: The single seed for every seeded step.
        cancel: Cancellation token; ``None`` uses the ambient one.
        progress: Optional ``(fraction, message)`` sink, called once per scope
            before that scope's work. Purely a report channel: it is never read
            back, so passing one cannot change a single number here.

    Returns:
        A :class:`SubpopClustering`.

    Raises:
        ValueError: on a missing column or an empty table.
        OperationCancelled: if cancellation is requested mid-sweep.
    """
    cols = tuple(str(c) for c in columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature column(s) {missing} not in the table; "
            f"available: {list(df.columns)}"
        )
    if len(df) < 1:
        raise ValueError("cluster_scopes needs a non-empty table")

    scopes: dict[str, ScopeClustering] = {}
    skipped: dict[str, int] = {}

    # Materialised so the progress fraction has a denominator. `iter_scopes` is a
    # generator over `df` and holds no copies -- each item is a view-shaped subset
    # the loop was going to hold anyway.
    todo = list(iter_scopes(df, scope_column=scope_column))
    for i, (scope, sub) in enumerate(todo):
        check_cancelled(cancel)
        if progress is not None:
            progress(i / len(todo), f"Clustering scope {scope} ({len(sub)} events)")
        if len(sub) < int(min_scope_n):
            skipped[scope] = int(len(sub))
            continue

        proj = scale_and_pca(
            sub,
            columns=cols,
            scope=scope,
            var_target=var_target,
            max_components=max_components,
            random_state=random_state,
        )
        Z = proj.scores

        check_cancelled(cancel)
        hl = hdbscan_labels(Z, min_samples=min_samples)

        check_cancelled(cancel)
        kl = kmeans_labels(Z, k=k, random_state=random_state)

        keep = hl >= 0
        scopes[scope] = ScopeClustering(
            scope=scope,
            pca=proj,
            loadings=loading_table(proj),
            hdbscan=hl,
            kmeans3=kl,
            hdbscan_silhouette=(_silhouette(Z[keep], hl[keep]) if keep.any() else None),
            kmeans3_silhouette=_silhouette(Z, kl),
            meta={
                "min_cluster_size": hdbscan_min_cluster_size(len(sub)),
                "min_samples": int(min_samples),
                "hdbscan_backend": "sklearn.cluster.HDBSCAN",
            },
        )

    return SubpopClustering(
        scopes=scopes,
        skipped=skipped,
        index=df.index,
        meta={
            "columns": list(cols),
            "scope_column": str(scope_column),
            "min_scope_n": int(min_scope_n),
            "var_target": float(var_target),
            "max_components": int(max_components),
            "k": int(k),
            "min_samples": int(min_samples),
            "random_state": int(random_state),
        },
    )


# ---------------------------------------------------------------------------
# 6-7. the two embeddings -- figure aids, and only that
# ---------------------------------------------------------------------------


def tsne_perplexity(n: int) -> int:
    """``min(30, max(5, n // 50))`` -- the notebook's n-dependent perplexity."""
    return int(min(30, max(5, int(n) // 50)))


def umap_neighbors(n: int) -> int:
    """``min(30, max(10, n // 60))`` -- the notebook's n-dependent neighbourhood."""
    return int(min(30, max(10, int(n) // 60)))


#: Columns a score matrix must have before a 2-D embedding can be fitted to it.
#: Both embeddings place each event somewhere in a PICTURE, so they need at
#: least two axes to place it *from*: t-SNE's ``init="pca"`` runs a 2-component
#: PCA over the scores it is handed, and a one-column matrix makes that
#: impossible. A scope can legitimately arrive one column wide --
#: :func:`scale_and_pca` keeps the components that reach :data:`VAR_TARGET`, and
#: one axis can carry 95 % of the scaled variance on its own -- so this is a
#: property of the data, not a caller error, and it has to be *said* rather than
#: left to scikit-learn, whose refusal for it ("n_components=2 must be between 1
#: and min(n_samples, n_features)=1") describes a solver, not a recording.
EMBEDDING_MIN_COMPONENTS = 2


def embedding_refusal(
    Z: np.ndarray,
    *,
    kind: str,
    perplexity: int | None = None,
) -> str | None:
    """Why a 2-D embedding cannot be fitted to ``Z``, or ``None`` if it can.

    The one precondition check both :func:`tsne_embedding` and
    :func:`umap_embedding` ask before handing a score matrix to a solver, so
    that a matrix no embedding can consume is refused **in this module's own
    words** -- naming the shape of the data that made the fit impossible --
    rather than surfacing whichever internal sentence the installed
    scikit-learn or umap-learn happens to raise. Callers that degrade to a
    figure-less report (see
    :func:`~nanoscope.analysis.subpop_report.scope_embeddings`) can then put
    the sentence straight into their notes.

    The returned text is a clause, not a sentence: it begins lowercase and
    describes the matrix, so a caller may prefix it with the scope it belongs
    to.

    Args:
        Z: The ``(n, n_kept)`` score matrix that would be embedded. A non-2-D
            ``Z`` is answered here too, but the callers raise for it first, as a
            caller error rather than a property of the data.
        kind: What is being fitted, for the message -- ``"t-SNE"``/``"UMAP"``.
        perplexity: The perplexity t-SNE would run with, when that applies.
            scikit-learn requires it to be strictly below the point count.

    Returns:
        A clause explaining the refusal, or ``None`` when ``Z`` can be embedded.
    """
    Za = np.asarray(Z, dtype=np.float64)
    if Za.ndim != 2:
        # The callers raise for this first, as a CALLER error rather than a property of
        # the data -- but this function is public and must be total, so it answers rather
        # than raising an IndexError of its own on ``Za.shape[1]``.
        return f"a 2-D {kind} needs a 2-D score matrix; this one has shape {Za.shape}"
    n, width = int(Za.shape[0]), int(Za.shape[1])
    if width < EMBEDDING_MIN_COMPONENTS:
        return (
            f"a 2-D {kind} needs a score space at least "
            f"{EMBEDDING_MIN_COMPONENTS} components wide to place events from, "
            f"and this one is {width} ({n} event(s) x {width} component(s)). "
            f"PCA keeps the components that reach the "
            f"{VAR_TARGET:.0%} variance target, so a single kept component "
            f"means one axis already carries nearly all of the scaled variance"
        )
    if n < EMBEDDING_MIN_COMPONENTS:
        return (
            f"a 2-D {kind} needs at least {EMBEDDING_MIN_COMPONENTS} events to "
            f"place, and this scope has {n}"
        )
    if perplexity is not None and int(perplexity) >= n:
        return (
            f"{kind} runs at perplexity {int(perplexity)}, which has to stay "
            f"below the point count, and this scope has {n} event(s)"
        )
    if not np.isfinite(Za).all():
        bad = int((~np.isfinite(Za)).any(axis=1).sum())
        return (
            f"{bad} of {n} event(s) carry a non-finite PCA score, which no "
            f"{kind} solver can place"
        )
    return None


def tsne_embedding(
    Z: np.ndarray,
    *,
    perplexity: int | None = None,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """2-D t-SNE of a scope's PCA scores, for looking at.

    Consumes the scope's own PCA space and **never** any cluster labels --
    labels are a colouring applied afterwards, so the embedding cannot be
    accused of drawing the very clusters it is used to display.

    Seeded, but do not gate anything on the coordinates: Barnes-Hut t-SNE is
    sensitive to the sklearn version, and nothing stored in the owner's run
    depends on these numbers. They exist to be plotted.

    Args:
        Z: ``(n, n_kept)`` scores.
        perplexity: Defaults to :func:`tsne_perplexity` of ``n``.
        random_state: Seed.

    Returns:
        ``(n, 2)`` embedding.

    Raises:
        ValueError: if ``Z`` is not 2-D, or if :func:`embedding_refusal` finds
            the score matrix cannot be embedded at all -- a one-component scope
            being the case that happens on real data. The message describes the
            matrix, not the solver; a caller that treats the embedding as the
            optional figure aid it is should catch this and say so in its notes
            rather than lose the run around it.
    """
    from sklearn.manifold import TSNE

    Za = np.asarray(Z, dtype=np.float64)
    if Za.ndim != 2:
        raise ValueError(f"expected a 2-D score matrix, got shape {Za.shape}")
    perp = tsne_perplexity(Za.shape[0]) if perplexity is None else int(perplexity)
    refusal = embedding_refusal(Za, kind="t-SNE", perplexity=perp)
    if refusal is not None:
        raise ValueError(refusal)
    model = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=int(random_state),
    )
    return np.asarray(model.fit_transform(Za), dtype=np.float64)


def umap_embedding(
    Z: np.ndarray,
    *,
    n_neighbors: int | None = None,
    min_dist: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """2-D UMAP of a scope's PCA scores, for looking at.

    Same independence from the cluster labels as :func:`tsne_embedding`, and the
    same warning only more so: ``random_state`` forces UMAP single-threaded, but
    the result is still **not** bit-reproducible across umap-learn, numba and
    platform versions. Never assert on these coordinates.

    Args:
        Z: ``(n, n_kept)`` scores.
        n_neighbors: Defaults to :func:`umap_neighbors` of ``n``.
        min_dist: UMAP's minimum embedded distance.
        random_state: Seed.

    Returns:
        ``(n, 2)`` embedding.

    Raises:
        ValueError: if ``Z`` is not 2-D, or if :func:`embedding_refusal` finds
            the score matrix cannot be embedded at all -- the same
            data-shaped refusal :func:`tsne_embedding` makes, in the same words.
        ImportError: if ``umap-learn`` is not installed. Unlike HDBSCAN there is
            no correctness question here, so a caller may reasonably catch this
            and skip the figure.
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "UMAP embedding needs the `umap-learn` package; the rest of the "
            "clustering stage does not depend on it."
        ) from exc

    Za = np.asarray(Z, dtype=np.float64)
    if Za.ndim != 2:
        raise ValueError(f"expected a 2-D score matrix, got shape {Za.shape}")
    refusal = embedding_refusal(Za, kind="UMAP")
    if refusal is not None:
        raise ValueError(refusal)
    nn = umap_neighbors(Za.shape[0]) if n_neighbors is None else int(n_neighbors)
    model = umap.UMAP(
        n_components=2,
        n_neighbors=nn,
        min_dist=float(min_dist),
        random_state=int(random_state),
    )
    return np.asarray(model.fit_transform(Za), dtype=np.float64)
