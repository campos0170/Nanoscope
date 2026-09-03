"""S4 / SA01 + SA02 + SA04 — what ``populations.pca`` must say about a degenerate input.

THE RULE. A refusal is not a number, and a count counts what it says.

``explained_variance_ratio_`` is ``explained_variance_ / total_var``. With every selected
feature constant that denominator is zero, sklearn warns and returns ``[nan, nan]``, and
the Plots window printed it verbatim into its plot title, BOTH axis labels, the Details
pane and an exported PNG:

    Explained variance: PC1 nan%, PC2 nan%  (cumulative nan%)

``pca`` was the one function on that path with no degeneracy guard: :func:`linear_fit`
raises on a "degenerate x", :func:`robust_scale` divides a constant column by 1.0 on
purpose, and :func:`nanoscope.analysis.stats.fit_dwell_exponential` refuses a zero-range
column outright with the sentence this one now borrows.

The other two facts this file pins are the ones the WINDOW needs and could not get from a
``PCAResult``: how many rows the finite-feature filter dropped (so the PCA pane can say
what the clustering pane already says), and the rank centred data of this shape can have
(so "cumulative 100.0%" at n <= 3 can be marked exact by construction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import populations

FEATS = ["dwell_s", "amplitude", "snr", "rms"]


def _constant(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({"dwell_s": np.full(n, 2.5e-4), "amplitude": np.full(n, 30.0),
                         "snr": np.full(n, 9.0), "rms": np.full(n, 1.4)})


def _varying(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(size=n) for c in FEATS})


def test_all_constant_features_are_refused_with_the_cause_named(recwarn):
    with pytest.raises(ValueError) as exc:
        populations.pca(_constant(), FEATS, n_components=2)
    msg = str(exc.value)
    assert "all 4 selected feature(s) are constant" in msg
    assert "no variance for PCA to explain" in msg
    # ...and the sklearn divide-by-zero never happens, so no RuntimeWarning escapes.
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_the_refusal_names_the_constant_values_so_the_upstream_bug_is_findable():
    with pytest.raises(ValueError, match=r"dwell_s = 0.00025"):
        populations.pca(_constant(), FEATS, n_components=2)


def test_one_varying_column_among_constants_is_not_refused_but_is_named():
    df = _constant()
    df["snr"] = np.linspace(1.0, 9.0, len(df))
    res = populations.pca(df, FEATS, n_components=2)
    assert np.all(np.isfinite(res.explained_variance_ratio))
    assert set(res.meta["constant_columns"]) == {"dwell_s", "amplitude", "rms"}


def test_a_healthy_feature_set_reports_no_constant_columns():
    res = populations.pca(_varying(), FEATS, n_components=2)
    assert res.meta["constant_columns"] == []
    assert np.all(np.isfinite(res.explained_variance_ratio))


def test_the_dropped_row_count_travels_with_the_result():
    df = _varying(200, seed=3)
    df.loc[df.index[:17], "snr"] = np.nan
    df.loc[df.index[100:103], "rms"] = np.inf
    res = populations.pca(df, FEATS, n_components=2)
    assert res.meta["n_input"] == 200
    assert res.n_kept == 180
    assert res.meta["n_dropped"] == 20


def test_a_clean_table_reports_no_dropped_rows():
    res = populations.pca(_varying(50), FEATS, n_components=2)
    assert res.meta["n_dropped"] == 0
    assert res.meta["n_input"] == res.n_kept == 50


@pytest.mark.parametrize("n,expect_rank", [(2, 1), (3, 2), (5, 4), (30, 4)])
def test_max_rank_is_min_of_n_minus_one_and_p(n, expect_rank):
    """Centred data of n rows has rank <= n-1; p features cap it at p. This is what
    makes "cumulative 100.0%" arithmetic rather than a finding at n <= 3."""
    res = populations.pca(_varying(n, seed=n), FEATS, n_components=2)
    assert res.meta["max_rank"] == expect_rank


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pure_noise_at_two_or_three_rows_really_does_explain_everything(n, seed):
    """The fixture behind the caveat: ANY n <= 3 rows report cumulative 100%."""
    res = populations.pca(_varying(n, seed=seed), FEATS, n_components=2)
    assert float(np.sum(res.explained_variance_ratio[:2])) == pytest.approx(1.0, abs=1e-9)
    assert res.meta["max_rank"] <= 2
