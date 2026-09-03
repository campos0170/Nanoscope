"""The per-scope scaling/PCA/clustering stage, against the owner's own labels.

WHAT THIS FILE IS DEFENDING
---------------------------
This stage is a **port**. The owner already ran it, and
``CSVData/event_clustering_features.csv`` stores the answers -- ten columns of
cluster labels, ``hdbscan``/``hdbscan_{2,10,18,30}`` and
``kmeans3``/``kmeans3_{2,10,18,30}``, over 1488 events -- alongside five
``pca_loadings_<scope>.csv`` tables. So "is the port right?" is decidable, and
the corpus tests here decide it by exact equality rather than by tolerance.

The claims, in order of how much damage their failure does:

  a. **All ten label columns reproduce exactly**, 1488/1488, NaN pattern
     included. Anything less is not a port. The tests report agreement *and*
     adjusted Rand, so a future regression says whether it broke the partition
     or merely renamed the clusters -- a distinction that decides whether the
     physics changed.
  b. **HDBSCAN comes from sklearn and never silently falls back.** The
     standalone ``hdbscan`` package is installed in this environment and gives
     a *different answer* on the largest scope (0.21 agreement, and a different
     noise set, not just a relabelling). ``test_standalone_hdbscan_disagrees``
     measures that rather than asserting it from the docs, so the day the two
     packages converge the test says so instead of quietly passing.
  c. **The scaler is imported, not copied.** ``populations.robust_scale`` is
     the only winsorise-then-median/IQR kernel this module may use, enforced by
     AST inspection -- a second implementation of a shared idea is this
     project's defining failure mode.
  d. **Labels join by index label, never by position.** The notebook's
     ``CLUST.loc[_sub.index, _col] = _hl`` is correct only while the frame
     keeps its original ``RangeIndex``; ``test_label_table_survives_a_shuffled_index``
     reorders the index and checks every event still gets its own label, which
     is the failure that would otherwise produce plausible-but-wrong clusters.
  e. Winsorisation precedes the median/IQR, the scope order is lexicographic,
     ``min_cluster_size`` floors, and the PC count is clamped to the components
     that exist -- each pinned separately, because each is a place a
     reasonable-looking rewrite would land somewhere else.
  f. Determinism: two runs, identical output, every seeded step included.

The corpus-marked tests skip cleanly when the research data is absent; the rest
run anywhere and cover the edges the owner's four recordings never exercise.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import populations, subpop_cluster as sc
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope

# ---------------------------------------------------------------------------
# the owner's data
# ---------------------------------------------------------------------------

ZN2_DIR = Path("/Users/abrahamcampos/Desktop/Zn2IonCondition/CSVData")
ORACLE_CSV = ZN2_DIR / "event_clustering_features.csv"

#: Every label column in the oracle, in the order it stores them. The order is
#: itself a claim: it follows the lexicographic scope order (10, 18, 2, 30).
ORACLE_LABEL_COLUMNS = [
    "hdbscan",
    "hdbscan_10",
    "hdbscan_18",
    "hdbscan_2",
    "hdbscan_30",
    "kmeans3",
    "kmeans3_10",
    "kmeans3_18",
    "kmeans3_2",
    "kmeans3_30",
]

#: Per-scope ``(n_events, n_PCs_kept)`` read off the owner's stored tables.
ORACLE_SCOPES = {"ALL": (1488, 5), "10": (110, 5), "18": (194, 5), "2": (108, 6), "30": (1076, 4)}

requires_oracle = pytest.mark.skipif(
    not ORACLE_CSV.is_file(), reason=f"oracle table absent: {ORACLE_CSV}"
)


@pytest.fixture(scope="module")
def oracle() -> pd.DataFrame:
    """The owner's stored feature+label table, 1488 x 29."""
    if not ORACLE_CSV.is_file():
        pytest.skip(f"oracle table absent: {ORACLE_CSV}")
    return pd.read_csv(ORACLE_CSV)


@pytest.fixture(scope="module")
def swept(oracle: pd.DataFrame) -> sc.SubpopClustering:
    """One full sweep over the oracle's features, shared by the corpus tests."""
    return sc.cluster_scopes(oracle)


# ---------------------------------------------------------------------------
# synthetic data, for everything the owner's four recordings cannot reach
# ---------------------------------------------------------------------------


def _synthetic(n_per: int = 80, seed: int = 0) -> pd.DataFrame:
    """Three well-separated blobs across two scopes, with the real column names.

    Deliberately *easy*: these tests are about plumbing -- alignment, ordering,
    determinism, cancellation -- and a hard clustering problem would make them
    fail for reasons that have nothing to do with what they check.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for scope_value, offset in ((2, 0.0), (30, 12.0)):
        for centre in (0.0, 8.0, 16.0):
            block = rng.normal(centre + offset, 0.35, size=(n_per, len(sc.CLUST_FEATS)))
            frame = pd.DataFrame(block, columns=list(sc.CLUST_FEATS))
            frame[sc.SCOPE_COLUMN] = scope_value
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ===========================================================================
# (a) the oracle: exact label reproduction
# ===========================================================================


@pytest.mark.corpus
@requires_oracle
def test_every_oracle_label_column_reproduces_exactly(oracle, swept):
    """All ten stored label columns, event for event. This is the port's pass mark."""
    from sklearn.metrics import adjusted_rand_score

    table = swept.label_table()
    assert list(table.columns) == ORACLE_LABEL_COLUMNS

    failures = []
    for column in ORACLE_LABEL_COLUMNS:
        got, want = table[column], oracle[column]
        # NaN means "not in this scope" and must land on exactly the same rows.
        assert (got.isna() == want.isna()).all(), f"{column}: NaN pattern differs"
        in_scope = want.notna()
        agree = float((got[in_scope].to_numpy() == want[in_scope].to_numpy()).mean())
        if agree != 1.0:
            # Report WHICH kind of failure: a renamed partition still scores
            # ARI 1.0, a genuinely different partition does not.
            ari = adjusted_rand_score(
                want[in_scope].to_numpy(), got[in_scope].to_numpy()
            )
            failures.append(f"{column}: agreement {agree:.4f}, adjusted Rand {ari:.4f}")
    assert not failures, "label columns diverged from the oracle:\n  " + "\n  ".join(failures)


@pytest.mark.corpus
@requires_oracle
def test_scope_partition_matches_the_oracle(swept):
    """Five scopes, in lexicographic order, with the owner's event counts."""
    assert list(swept.scopes) == ["ALL", "10", "18", "2", "30"]
    assert swept.skipped == {}
    assert {s: r.n for s, r in swept.scopes.items()} == {
        s: n for s, (n, _) in ORACLE_SCOPES.items()
    }


@pytest.mark.corpus
@requires_oracle
def test_loading_tables_reproduce_the_stored_csvs(swept):
    """Values *and* row order of every ``pca_loadings_<scope>.csv``.

    Row order is checked because it is the part a plausible rewrite gets wrong:
    the notebook sorts the ROUNDED info_share, which reorders near-ties. See
    :func:`nanoscope.analysis.subpop_cluster.loading_table`.
    """
    for scope, result in swept.scopes.items():
        path = ZN2_DIR / f"pca_loadings_{scope}.csv"
        if not path.is_file():
            pytest.skip(f"stored loading table absent: {path}")
        want = pd.read_csv(path, index_col=0)
        got = sc.loading_table(result.pca)

        assert list(got.index) == list(want.index), f"[{scope}] row order differs"
        assert list(got.columns) == list(want.columns), f"[{scope}] columns differ"
        # Exact equality: both sides are rounded to the same decimals.
        np.testing.assert_array_equal(
            got.reindex(want.index).to_numpy(), want.to_numpy(),
            err_msg=f"[{scope}] loading values differ",
        )


@pytest.mark.corpus
@requires_oracle
def test_kept_pc_counts_match_the_owners_tables(swept):
    """4-6 PCs per scope, and never the clamped path on this data."""
    for scope, (_, n_kept) in ORACLE_SCOPES.items():
        result = swept.scopes[scope].pca
        assert result.n_kept == n_kept, f"[{scope}] kept {result.n_kept} PCs, oracle {n_kept}"
        # The clamp is inert here -- the owner's scopes reach ~99.9% by PC10.
        assert result.meta["n_kept_unclamped"] == n_kept
        assert result.variance_captured >= sc.VAR_TARGET
        assert result.scores.shape == (result.n, n_kept)


@pytest.mark.corpus
@requires_oracle
def test_within_recording_scopes_are_pure_hdbscan_noise(swept):
    """A finding, not a failure: those scopes are continua with no density modes.

    Pinned so that a future change which "improves" them into clusters has to
    argue with the owner's stored labels rather than slip through.
    """
    for scope in ("2", "10", "18"):
        result = swept.scopes[scope]
        assert result.n_hdbscan_clusters == 0
        assert result.hdbscan_noise_fraction == 1.0
        assert result.hdbscan_silhouette is None  # undefined, not a fake 0.0
    for scope in ("ALL", "30"):
        assert swept.scopes[scope].n_hdbscan_clusters == 2


# ===========================================================================
# (b) HDBSCAN: the right backend, and no silent fallback
# ===========================================================================


def test_hdbscan_uses_sklearn_and_says_so():
    """The backend is recorded, so a stored result can be audited after the fact."""
    frame = _synthetic()
    swept = sc.cluster_scopes(frame)
    for result in swept.scopes.values():
        assert result.meta["hdbscan_backend"] == "sklearn.cluster.HDBSCAN"
        assert result.meta["min_samples"] == 10


def test_no_fallback_to_the_standalone_hdbscan_package():
    """AST guard: this module must not import the standalone ``hdbscan``.

    The notebook's ``_make_hdbscan`` falls back to it on ImportError. Inheriting
    that fallback would mean a machine without sklearn's HDBSCAN silently
    produces different clusters instead of failing -- and on the owner's largest
    scope "different" means 21% agreement.
    """
    source = Path(inspect.getfile(sc)).read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "hdbscan" not in imported, (
        "subpop_cluster imports the standalone `hdbscan` package; it must use "
        "sklearn.cluster.HDBSCAN exclusively"
    )
    assert "sklearn" in imported


@pytest.mark.corpus
@requires_oracle
def test_standalone_hdbscan_disagrees_on_the_largest_scope(oracle, swept):
    """MEASURE the disagreement that makes the backend choice load-bearing.

    Not a documentation claim: if the two packages ever converge, this test
    fails and tells us the constraint has relaxed, rather than leaving a stale
    warning in a docstring forever.

    It also separates the two ways a clustering can differ, because they are
    not equally serious and the raw agreement number conflates them. On this
    scope the standalone package's *partition* is close to the oracle's
    (adjusted Rand ~0.977) -- what destroys exact agreement is that clusters 0
    and 1 come back renamed. The noise set differs too, so no permutation
    repairs it, and the stored label columns are not reproduced either way.
    """
    from sklearn.metrics import adjusted_rand_score

    standalone = pytest.importorskip(
        "hdbscan", reason="standalone hdbscan not installed; nothing to compare"
    )
    scope = swept.scopes["30"]
    Z = scope.pca.scores
    other = standalone.HDBSCAN(
        min_cluster_size=sc.hdbscan_min_cluster_size(len(Z)),
        min_samples=sc.HDBSCAN_MIN_SAMPLES,
    ).fit_predict(Z)
    want = oracle["hdbscan_30"].dropna().to_numpy()

    # sklearn: identical labels, not merely an equivalent partition.
    assert float((scope.hdbscan == want).mean()) == 1.0
    assert adjusted_rand_score(want, scope.hdbscan) == pytest.approx(1.0)

    agreement = float((other == want).mean())
    assert agreement < 0.5, (
        f"the standalone hdbscan package now agrees at {agreement:.4f} on zn_30h "
        "(it was 0.21); the no-fallback rule may be re-examinable"
    )
    # A relabelling is most of the gap...
    assert adjusted_rand_score(want, other) > 0.9
    # ...but not all of it: the noise SET differs, so no permutation fixes it.
    assert int((other < 0).sum()) != int((want < 0).sum())


def test_min_cluster_size_floors_rather_than_dividing():
    """``max(20, n // 100)``. Integer floor -- invisible on the oracle, not elsewhere."""
    assert sc.hdbscan_min_cluster_size(1488) == 20   # 14 -> floor
    assert sc.hdbscan_min_cluster_size(1076) == 20   # 10 -> floor
    assert sc.hdbscan_min_cluster_size(2000) == 20
    assert sc.hdbscan_min_cluster_size(2599) == 25   # not 25.99
    assert sc.hdbscan_min_cluster_size(10_000) == 100


# ===========================================================================
# (c) the scaler is imported, not reimplemented
# ===========================================================================


def test_the_module_calls_the_shared_scaler_and_defines_no_second_one():
    """AST guard against a private copy of the winsorise/median/IQR kernel.

    ``populations.robust_scale`` is the app's one such kernel and is verified
    bit-identical to the notebook's inline expression. This module must call it
    and must not grow percentile arithmetic of its own.
    """
    tree = ast.parse(Path(inspect.getfile(sc)).read_text())
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "robust_scale" in called, "subpop_cluster must call populations.robust_scale"
    for banned in ("nanpercentile", "percentile", "nanmedian"):
        assert banned not in called, (
            f"subpop_cluster calls np.{banned} -- that is the shared scaler's job; "
            "use populations.robust_scale instead of a second implementation"
        )
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "robust_scale" not in defined


def test_scaling_is_the_notebooks_expression_verbatim():
    """Pin the whole scaling convention by transcribing cell 51 and comparing.

    This is the honest way to pin the winsorise-then-median/IQR order. The
    tempting alternative -- swap the order and assert the answer changes --
    does NOT work at these limits, and believing it does would leave the
    convention untested: clipping 1 % of each tail cannot move a quartile or a
    median, so the two orders agree to floating-point noise. Measured on the
    owner's own feature matrix they are identical at 1-99, 5-95, 10-90 and
    20-80, and first diverge at 26-74 (max 8.6e-02) -- i.e. only once the clip
    reaches inside the IQR. See :func:`test_the_two_clip_orders_commute_at_these_limits`.

    So the order is pinned the only way that actually bites: against the source
    expression, character for character.
    """
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(300, len(sc.CLUST_FEATS))) * rng.uniform(
        0.1, 500.0, size=len(sc.CLUST_FEATS)
    )
    raw[0, 0] = 1e6          # a dominating outlier
    raw[1, 3] = np.nan       # a missing feature
    raw[2, 7] = np.inf
    raw[:, 5] = 4.0          # a constant (zero-IQR) column
    frame = pd.DataFrame(raw, columns=list(sc.CLUST_FEATS))

    # --- cell 51 `_clust_scale_pca`, transcribed -------------------------
    X = frame[list(sc.CLUST_FEATS)].to_numpy(float)
    X = np.where(np.isfinite(X), X, np.nan)
    lo, hi = np.nanpercentile(X, [1, 99], axis=0)
    X = np.clip(X, lo, hi)
    med = np.nanmedian(X, axis=0)
    iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    expected = np.where(
        np.isfinite((X - med) / np.where(iqr > 0, iqr, 1.0)),
        (X - med) / np.where(iqr > 0, iqr, 1.0),
        0.0,
    )
    # ---------------------------------------------------------------------

    np.testing.assert_array_equal(sc.scale_features(frame), expected)
    assert np.isfinite(sc.scale_features(frame)).all()


def test_the_two_clip_orders_commute_at_these_limits():
    """Record WHY the order cannot be pinned by a differential test.

    Documents the boundary as an executable fact rather than a claim in prose:
    identical while the clip stays outside the quartiles, different once it does
    not. If a future change moves the winsorisation limits inside the IQR, the
    order stops being free and this test is where that shows up.
    """
    rng = np.random.default_rng(2)
    column = np.sort(rng.normal(size=400))

    def clip_first(lo_pct, hi_pct):
        lo, hi = np.percentile(column, [lo_pct, hi_pct])
        clipped = np.clip(column, lo, hi)
        spread = np.percentile(clipped, 75) - np.percentile(clipped, 25)
        return (clipped - np.median(clipped)) / spread

    def scale_first(lo_pct, hi_pct):
        spread = np.percentile(column, 75) - np.percentile(column, 25)
        z = (column - np.median(column)) / spread
        lo, hi = np.percentile(z, [lo_pct, hi_pct])
        return np.clip(z, lo, hi)

    for lo_pct, hi_pct in [(1, 99), (5, 95), (10, 90), (20, 80)]:
        np.testing.assert_allclose(
            clip_first(lo_pct, hi_pct), scale_first(lo_pct, hi_pct), atol=1e-12,
            err_msg=f"orders should commute at {lo_pct}-{hi_pct}",
        )
    # Inside the IQR they genuinely part company.
    assert not np.allclose(clip_first(30, 70), scale_first(30, 70), atol=1e-6)


def test_winsorisation_bounds_the_extremes():
    """What the clip is for: one outlier must not own the whole axis.

    At a realistic scope size the taming is complete -- the 99th percentile
    lands among the bulk, so the outlier is pulled all the way back to it.
    """
    column = np.r_[np.linspace(0.0, 1.0, 999), 1e6]
    frame = pd.DataFrame({c: column.copy() for c in sc.CLUST_FEATS})
    scaled = sc.scale_features(frame)

    unclipped = (column - np.median(column)) / (
        np.percentile(column, 75) - np.percentile(column, 25)
    )
    assert np.abs(unclipped).max() > 1e6
    assert np.abs(scaled).max() < 5.0
    assert np.isfinite(scaled).all()


def test_winsorisation_is_weaker_than_it_looks_on_a_small_scope():
    """A trap worth pinning: at n=100 the 99th percentile straddles the outlier.

    ``np.percentile`` interpolates, so with a single extreme value in 100 the
    upper clip bound sits ~1 % of the way from the bulk to the outlier -- which
    is enormous when the outlier is. The clip still cuts the excursion by ~50x,
    but it does NOT pull the point back to the bulk the way it does at n=1000.
    Scopes here run 108-1488 events, so the small end of that range is closer
    to this regime than to the clean one above.
    """
    column = np.r_[np.linspace(0.0, 1.0, 99), 1e6]
    frame = pd.DataFrame({c: column.copy() for c in sc.CLUST_FEATS})
    scaled = np.abs(sc.scale_features(frame)).max()

    unclipped = np.abs(
        (column - np.median(column))
        / (np.percentile(column, 75) - np.percentile(column, 25))
    ).max()
    assert scaled < unclipped / 20.0, "clipping should still cut the excursion"
    assert scaled > 1e3, (
        "at n=100 the clip bound is interpolated onto the outlier; if this now "
        "passes, the winsorisation semantics have changed"
    )


def test_non_finite_features_keep_their_rows(caplog):
    """A NaN feature lands on the scaled median; it must not delete the event.

    ``populations.feature_matrix`` drops such rows whole, which is right for a
    fit that would otherwise be undefined and wrong here: the row count sets
    HDBSCAN's ``min_cluster_size`` and the label table has to cover every event.
    """
    frame = _synthetic(n_per=40)
    frame.loc[0, "skewness"] = np.nan
    frame.loc[1, "kurtosis"] = np.inf

    values = sc.feature_values(frame)
    assert values.shape == (len(frame), len(sc.CLUST_FEATS))
    assert not np.isfinite(values[0]).all()

    scaled = sc.scale_features(frame)
    assert scaled.shape == (len(frame), len(sc.CLUST_FEATS))
    assert np.isfinite(scaled).all()
    assert scaled[0, list(sc.CLUST_FEATS).index("skewness")] == 0.0

    swept = sc.cluster_scopes(frame)
    assert swept.scopes["ALL"].n == len(frame)
    assert len(swept.label_table()) == len(frame)


# ===========================================================================
# (d) labels join by index label, never by position
# ===========================================================================


def test_label_table_survives_a_hostile_index():
    """THE regression test for the notebook's alignment hazard.

    ``CLUST.loc[_sub.index, _col] = _hl`` assigns a positional array into a
    label-indexed slice; it is right only while the frame keeps its original
    ``RangeIndex``. Give the frame a non-monotonic, non-integer index and a
    positional implementation attaches every label to the wrong event -- while
    still producing clusters that look entirely reasonable.

    The **row order is deliberately unchanged** here, so every estimator sees a
    bit-identical matrix and the only thing that can differ is the join. (A
    permuted row order is a separate question, and a confounded one: k-means++
    seeds from the data in order, so permuting rows legitimately renumbers the
    clusters. That is tested separately below.)
    """
    frame = _synthetic(n_per=40)
    baseline = sc.cluster_scopes(frame).label_table()

    hostile = frame.copy()
    rng = np.random.default_rng(7)
    hostile.index = [f"evt-{i:04d}" for i in rng.permutation(len(frame))]

    got = sc.cluster_scopes(hostile).label_table()

    assert got.index.equals(hostile.index), "label table must keep the source index"
    np.testing.assert_array_equal(got.to_numpy(), baseline.to_numpy())
    assert list(got.columns) == list(baseline.columns)


def test_labels_follow_the_event_through_a_row_permutation():
    """Permute the rows: each event must still receive its own cluster's label.

    Cluster *ids* may legitimately renumber under a permutation (k-means++
    draws its seeds in data order), so identity of the labels is the wrong
    assertion -- identity of the **partition over events** is the right one, and
    a misaligned join breaks it. Adjusted Rand of exactly 1.0 says every event
    kept the same companions; anything less means labels moved between events.
    """
    from sklearn.metrics import adjusted_rand_score

    frame = _synthetic(n_per=40)
    frame["event_id"] = np.arange(len(frame))

    baseline = sc.cluster_scopes(frame).label_table()
    baseline.index = frame["event_id"].to_numpy()

    rng = np.random.default_rng(13)
    order = rng.permutation(len(frame))
    shuffled = frame.iloc[order].copy()
    shuffled.index = [f"evt-{i:04d}" for i in order]

    got = sc.cluster_scopes(shuffled).label_table()
    got.index = shuffled["event_id"].to_numpy()
    aligned = got.reindex(baseline.index)

    for column in baseline.columns:
        in_scope = baseline[column].notna()
        assert (aligned[column].notna() == in_scope).all(), f"{column}: scope membership moved"
        assert adjusted_rand_score(
            baseline.loc[in_scope, column].to_numpy(),
            aligned.loc[in_scope, column].to_numpy(),
        ) == pytest.approx(1.0), f"{column}: the partition over events changed"


def test_label_table_keeps_noise_and_out_of_scope_distinct():
    """``-1`` (HDBSCAN noise) and ``NaN`` (not in this scope) are different claims.

    Collapsing them -- a ``fillna(-1)`` is the obvious slip -- would convert
    every out-of-scope row into a noise point and multiply the apparent noise
    rate by the number of scopes.
    """
    frame = _synthetic(n_per=40)
    table = sc.cluster_scopes(frame).label_table()

    for scope in ("2", "30"):
        column = table[f"hdbscan_{scope}"]
        in_scope = frame[sc.SCOPE_COLUMN].astype(str) == scope
        assert column[in_scope].notna().all()
        assert column[~in_scope].isna().all()

    assert table["hdbscan"].notna().all()  # the pooled scope covers everything
    assert sc.HDBSCAN_NOISE_LABEL == populations.DROPPED_LABEL  # the collision is real
    # ...which is exactly why HDBSCAN labels never go through populations'
    # realignment; k-means labels, which do, must never carry the sentinel.
    assert (table[[c for c in table.columns if c.startswith("kmeans3")]].fillna(0) >= 0).all().all()


# ===========================================================================
# (e) the remaining pinned conventions
# ===========================================================================


def test_scope_order_is_lexicographic_on_strings():
    """``ALL, "10", "18", "2", "30"`` -- not numeric order.

    Numeric sorting yields the same columns in a different order and breaks any
    positional comparison against the owner's stored table.
    """
    frame = pd.DataFrame(
        {c: np.linspace(0, 1, 8) for c in sc.CLUST_FEATS}
        | {sc.SCOPE_COLUMN: [2, 2, 10, 10, 18, 18, 30, 30]}
    )
    assert [name for name, _ in sc.iter_scopes(frame)] == ["ALL", "10", "18", "2", "30"]


def test_iter_scopes_subsets_keep_parent_index_labels():
    """The property the whole alignment story rests on."""
    frame = _synthetic(n_per=20)
    frame.index = [f"e{i}" for i in range(len(frame))]
    for name, subset in sc.iter_scopes(frame):
        assert subset.index.isin(frame.index).all()
        if name != sc.SCOPE_ALL:
            assert (subset[sc.SCOPE_COLUMN].astype(str) == name).all()


def test_small_scopes_are_recorded_not_silently_dropped():
    """"Too few events to cluster" is a finding; it must be reported."""
    frame = _synthetic(n_per=40)  # 120 events per scope
    tiny = frame[frame[sc.SCOPE_COLUMN] == 2].head(10).copy()
    tiny[sc.SCOPE_COLUMN] = 99
    combined = pd.concat([frame, tiny], ignore_index=True)

    swept = sc.cluster_scopes(combined)
    assert "99" in swept.skipped and swept.skipped["99"] == 10
    assert "99" not in swept.scopes
    # A skipped scope contributes no column at all -- not a column of NaN.
    assert "hdbscan_99" not in swept.label_table().columns


def test_kept_pc_count_is_clamped_to_the_components_that_exist():
    """The notebook's latent ``nk`` bug: a count that can exceed the array width.

    ``searchsorted(cumsum(evr), target) + 1`` returns one past the end when the
    capped components never reach the target; ``Z[:, :nk]`` then silently
    returns fewer columns than ``nk`` claims. The owner's data never triggers it
    (99.9% cumulative by PC10). Isotropic noise does.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        rng.normal(size=(300, len(sc.CLUST_FEATS))), columns=list(sc.CLUST_FEATS)
    )
    result = sc.scale_and_pca(frame, max_components=3, var_target=0.99)

    assert result.meta["n_kept_unclamped"] > 3, "expected the unclamped count to overrun"
    assert result.n_kept == 3
    assert result.scores.shape[1] == result.n_kept, "n_kept must be the true width"


def test_loading_table_rounds_before_sorting():
    """Rounding reorders near-ties, and the owner's tables are the rounded ones."""
    frame = _synthetic(n_per=40)
    result = sc.scale_and_pca(frame)

    rounded = sc.loading_table(result)
    assert (rounded["info_share"] == rounded["info_share"].round(4)).all()
    assert rounded["info_share"].is_monotonic_decreasing

    full = sc.loading_table(result, decimals=None)
    assert full["info_share"].is_monotonic_decreasing
    assert set(full.index) == set(rounded.index)
    # Full precision really is finer than the stored table's 4 places.
    assert not np.array_equal(full["info_share"].to_numpy(), rounded["info_share"].to_numpy())


def test_info_share_sums_to_the_variance_the_kept_pcs_capture():
    """The property that makes ``info_share`` readable as a share at all."""
    frame = _synthetic(n_per=40)
    result = sc.scale_and_pca(frame)
    table = sc.loading_table(result, decimals=None)
    assert table["info_share"].sum() == pytest.approx(result.variance_captured, abs=1e-12)


def test_pca_and_scaling_are_refit_inside_each_scope():
    """Within-scope re-fit, not a pooled fit applied to subsets.

    Shift one scope far away in feature space: if anything were fit on the
    pooled table, the shifted scope's own projection would be dominated by the
    offset rather than by its internal structure.
    """
    frame = _synthetic(n_per=60)
    pooled = sc.cluster_scopes(frame)

    scope_30 = frame[frame[sc.SCOPE_COLUMN] == 30]
    standalone = sc.scale_and_pca(scope_30, scope="30")

    np.testing.assert_allclose(
        pooled.scopes["30"].pca.scores, standalone.scores, rtol=0, atol=0
    )
    # And the pooled scope genuinely differs from any single scope's basis.
    assert not np.allclose(
        pooled.scopes["ALL"].pca.explained_variance_ratio,
        standalone.explained_variance_ratio,
    )


def test_kmeans_is_delegated_to_populations():
    """The binding must produce exactly what ``populations.cluster`` produces."""
    rng = np.random.default_rng(11)
    Z = np.r_[rng.normal(0, 0.3, (60, 4)), rng.normal(6, 0.3, (60, 4)), rng.normal(12, 0.3, (60, 4))]
    mine = sc.kmeans_labels(Z, k=3, random_state=sc.RANDOM_STATE)

    cols = [f"pc{i + 1}" for i in range(Z.shape[1])]
    theirs = populations.cluster(
        pd.DataFrame(Z, columns=cols), cols,
        method="kmeans", k=3, standardize=False, random_state=sc.RANDOM_STATE,
    ).labels.to_numpy()
    np.testing.assert_array_equal(mine, theirs)


# ===========================================================================
# (f) determinism, cancellation, and the embeddings
# ===========================================================================


def test_two_sweeps_are_bit_identical():
    """Every seeded step comes from one seed; running twice must not drift."""
    frame = _synthetic(n_per=50)
    first, second = sc.cluster_scopes(frame), sc.cluster_scopes(frame)

    pd.testing.assert_frame_equal(first.label_table(), second.label_table())
    for scope in first.scopes:
        a, b = first.scopes[scope], second.scopes[scope]
        np.testing.assert_array_equal(a.pca.scores, b.pca.scores)
        np.testing.assert_array_equal(a.hdbscan, b.hdbscan)
        np.testing.assert_array_equal(a.kmeans3, b.kmeans3)
        pd.testing.assert_frame_equal(a.loadings, b.loadings)


def test_the_seed_actually_reaches_sklearns_pca():
    """``populations.pca`` grew a ``random_state``; the default must not have moved."""
    signature = inspect.signature(populations.pca)
    assert signature.parameters["random_state"].default is None, (
        "the new parameter must default to None so existing callers are unchanged"
    )
    frame = pd.DataFrame(
        np.random.default_rng(0).normal(size=(50, 4)), columns=list("abcd")
    )
    assert populations.pca(frame, list("abcd")).meta["random_state"] is None
    assert populations.pca(frame, list("abcd"), random_state=0).meta["random_state"] == 0


def test_the_sweep_polls_the_ambient_cancel_token():
    """Long work must stop on request rather than run to completion first."""
    frame = _synthetic(n_per=60)
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            sc.cluster_scopes(frame)


def test_tsne_consumes_only_the_scores_and_is_reproducible():
    """Seeded, label-free, right shape. The coordinates themselves are not pinned.

    Barnes-Hut t-SNE is sklearn-version sensitive and nothing stored in the
    owner's run depends on these numbers, so asserting on them would buy a
    brittle test and no protection.
    """
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(120, 4))
    first = sc.tsne_embedding(Z, random_state=0)
    second = sc.tsne_embedding(Z, random_state=0)

    assert first.shape == (120, 2)
    np.testing.assert_array_equal(first, second)
    assert sc.tsne_perplexity(120) == 5
    assert sc.tsne_perplexity(1488) == 29
    assert sc.tsne_perplexity(100_000) == 30


def test_umap_neighbour_rule():
    """Pinned arithmetic; the embedding itself is not bit-reproducible anywhere."""
    assert sc.umap_neighbors(120) == 10
    assert sc.umap_neighbors(1488) == 24
    assert sc.umap_neighbors(100_000) == 30


# ===========================================================================
# input guards
# ===========================================================================


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"columns": []}, "at least one feature column"),
        ({"columns": ["nope"]}, "not in the table"),
    ],
)
def test_missing_or_empty_features_are_refused(kwargs, message):
    frame = _synthetic(n_per=20)
    with pytest.raises(ValueError, match=message):
        sc.cluster_scopes(frame, **kwargs)


def test_result_containers_construct_with_their_defaults():
    """``pd.Index`` cannot be a ``default_factory`` -- ``pd.Index()`` raises."""
    empty = sc.SubpopClustering(scopes={})
    assert len(empty.index) == 0
    assert empty.skipped == {}
    assert empty.label_table().empty
    assert "SubpopClustering" in repr(empty)


def test_public_names_all_exist():
    """``__all__`` must not promise a name the module does not define."""
    missing = [name for name in sc.__all__ if not hasattr(sc, name)]
    assert not missing, f"__all__ names missing from the module: {missing}"


def test_missing_scope_column_is_refused():
    frame = pd.DataFrame({c: np.linspace(0, 1, 10) for c in sc.CLUST_FEATS})
    with pytest.raises(ValueError, match="scope column"):
        sc.cluster_scopes(frame)


def test_score_matrices_must_be_two_dimensional():
    for fn in (sc.hdbscan_labels, sc.kmeans_labels):
        with pytest.raises(ValueError, match="2-D score matrix"):
            fn(np.zeros(10))


def test_a_scope_too_small_for_pca_is_refused():
    frame = pd.DataFrame({c: [1.0] for c in sc.CLUST_FEATS})
    with pytest.raises(ValueError, match="PCA needs at least 2"):
        sc.scale_and_pca(frame)


def test_var_target_outside_zero_one_is_refused():
    frame = _synthetic(n_per=20)
    with pytest.raises(ValueError, match=r"var_target must be in \[0, 1\]"):
        sc.scale_and_pca(frame, var_target=1.5)


# ================================================================================================
# row order is part of the input
# ================================================================================================

def test_the_partition_survives_a_row_permutation_even_though_the_numbering_does_not():
    """Documented in ``cluster_scopes``, and the reason the owner's labels reproduce.

    Anyone re-deriving his table from a differently-ordered source gets the same
    science under different names -- k-means++ draws its seeds by row position.
    Asserting the invariant that DOES hold (the partition) pins the docstring's
    claim, so a future change that quietly makes labels order-dependent in a
    stronger way than this is caught.
    """
    from sklearn.metrics import adjusted_rand_score

    frame = _synthetic(n_per=60)
    base = sc.cluster_scopes(frame).label_table()

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(frame))
    shuffled = frame.iloc[perm].reset_index(drop=True)
    back = sc.cluster_scopes(shuffled).label_table().iloc[np.argsort(perm)].reset_index(drop=True)

    for col in base.columns:
        a, b = base[col].to_numpy(), back[col].to_numpy()
        keep = ~(pd.isna(a) | pd.isna(b))
        # The NaN pattern is scope membership, which cannot depend on order at all.
        assert (pd.isna(a) == pd.isna(b)).all(), col
        if keep.sum() < 2:
            continue
        assert adjusted_rand_score(a[keep].astype(int), b[keep].astype(int)) == pytest.approx(
            1.0, abs=0.02
        ), f"{col}: the PARTITION moved, not just the numbering"


def test_the_docstring_warns_that_labels_are_not_order_invariant():
    """A hazard this quiet has to be written down where the caller reads it: the
    failure mode is a plausible-looking table with every integer changed."""
    doc = inspect.getdoc(sc.cluster_scopes) or ""
    assert "ROW ORDER IS PART OF THE INPUT" in doc
    assert "not invariant" in doc.lower()
