"""CS4-03 — a figure aid may cost the reader the picture, never the run.

``scope_embeddings`` called ``SC.tsne_embedding(res.pca.scores)`` with no guard on the
WIDTH of the score matrix. A scope whose PCA kept a single component (PC1 alone reaching
the 95 % variance target, which one heavy-tailed event is enough to do — the robust
scaler divides ``kurtosis`` by the IQR) made t-SNE's ``init="pca"`` ask for two
components out of one, and scikit-learn's

    ValueError: n_components=2 must be between 1 and min(n_samples, n_features)=1
    with svd_solver='randomized'

escaped the entire pipeline: "Analysis failed" with a solver sentence in it, no report,
no subpopulation assignment, the whole feature + gate + clustering stage discarded — for
an embedding the module itself calls "EXPORTED and never printed" and that nothing else
in the report depends on.

The rule these tests pin: the two embeddings are asked one shared precondition
(:func:`subpop_cluster.embedding_refusal`) which describes the DATA in this package's own
words, and the report boundary turns any embedding failure into a note that names the
scope, the reason and what still stands — while a cancelled run stays cancelled.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import subpop_cluster as SC
from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.analysis.events import REQUIRED, EventSet

FS = 100_000.0


# ================================================================================================
# fixtures — the shape that actually happens on a real recording
# ================================================================================================

def _events(n_events: int, *, seed: int = 3, spike_pA: float = 0.0,
            spike_samples: int = 1, n_spiky: int = 1):
    """A flat recording plus its EventSet; ``spike_pA`` makes ONE event leptokurtic.

    One single-sample spike inside one otherwise ordinary translocation is the commonest
    artefact in a solid-state recording, and it is enough to send PC1 past the 0.95
    variance target on its own — which is the whole condition under test. Measured on
    this fixture: n_kept = 1, PC1 = 0.987, loading 0.99 on ``kurtosis``.
    """
    rng = np.random.default_rng(seed)
    n = 60_000 + n_events * 4_000
    x = np.full(n, 100.0) + rng.normal(0.0, 1.2, n)
    rows, pos = [], 20_000
    for i in range(n_events):
        w = int(rng.integers(200, 600))
        depth = float(rng.uniform(30.0, 60.0))
        seg = np.full(w, depth) + rng.normal(0.0, 2.0, w)
        if i < n_spiky and spike_pA:
            seg[w // 2:w // 2 + spike_samples] += float(spike_pA)
        x[pos:pos + w] -= seg
        rows.append({
            "event_id": i, "recording": "rec", "pore_id": "p1",
            "start_idx": pos, "end_idx": pos + w,
            "t_start_s": pos / FS, "dwell_s": w / FS,
            "baseline_pA": 100.0, "depth_pA": depth, "frac_depth": depth / 100.0,
            "ecd_pA_s": depth * w / FS, "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / 1.2, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
        pos += 4_000
    return (np.ascontiguousarray(x, dtype=np.float32),
            EventSet(pd.DataFrame(rows)[REQUIRED]))


def _cfg(**kw) -> SR.SubpopConfig:
    base = {"k_grid": (1, 2), "n_splits": 2, "n_boot": 0, "with_tsne": True}
    base.update(kw)
    return SR.SubpopConfig(**base)


def _run(sig, es, cfg):
    return SR.run_subpopulations(
        sig, sig.astype(np.float64), FS, es,
        recording="rec", duration_s=len(sig) / FS, config=cfg)


# ================================================================================================
# 1. the precondition, in the package's own words
# ================================================================================================

def test_a_one_component_score_space_is_refused_by_this_package_not_by_sklearn():
    """The refusal names the matrix, not a solver: no ``svd_solver`` in the sentence."""
    Z = np.linspace(-1.0, 1.0, 80).reshape(80, 1)
    why = SC.embedding_refusal(Z, kind="t-SNE", perplexity=5)
    assert why is not None
    assert "at least 2 components wide" in why
    assert "80 event(s) x 1 component(s)" in why
    assert "95%" in why                       # says WHY one component can be all there is
    assert "svd_solver" not in why and "n_components=2" not in why

    with pytest.raises(ValueError) as exc:
        SC.tsne_embedding(Z, random_state=0)
    assert "at least 2 components wide" in str(exc.value)
    assert "svd_solver" not in str(exc.value)


def test_umap_refuses_the_same_matrix_in_the_same_words_before_it_fits():
    """One precondition for both embeddings — and it is asked BEFORE the solver runs.

    The fake ``umap`` module explodes if anything constructs a model, so a passing test
    proves the refusal happens ahead of the fit rather than being a caught solver error.
    """
    def _boom(*a, **k):  # pragma: no cover - fires only on a regression
        raise AssertionError("UMAP was fitted on a matrix that cannot be embedded")

    fake = types.ModuleType("umap")
    fake.UMAP = _boom
    saved = sys.modules.get("umap")
    sys.modules["umap"] = fake
    try:
        with pytest.raises(ValueError) as exc:
            SC.umap_embedding(np.zeros((60, 1)), random_state=0)
    finally:
        if saved is None:
            del sys.modules["umap"]
        else:
            sys.modules["umap"] = saved
    assert "at least 2 components wide" in str(exc.value)
    assert "UMAP" in str(exc.value)


def test_a_scope_with_too_few_points_for_its_perplexity_says_that_instead():
    """Each impossible shape gets its own reason — never one catch-all sentence."""
    why = SC.embedding_refusal(np.zeros((4, 3)), kind="t-SNE", perplexity=5)
    assert why is not None and "perplexity 5" in why and "4 event(s)" in why


def test_a_non_finite_score_is_named_and_counted():
    Z = np.zeros((10, 3))
    Z[2, 1] = np.nan
    why = SC.embedding_refusal(Z, kind="t-SNE", perplexity=5)
    assert why is not None and "1 of 10 event(s)" in why and "non-finite" in why


def test_a_healthy_score_matrix_is_not_refused():
    rng = np.random.default_rng(0)
    assert SC.embedding_refusal(rng.normal(size=(120, 4)), kind="t-SNE",
                                perplexity=5) is None
    assert SC.embedding_refusal(rng.normal(size=(120, 4)), kind="UMAP") is None


# ================================================================================================
# 2. the pipeline: the picture is lost, the run is not
# ================================================================================================

def test_a_rank_one_scope_costs_the_picture_and_not_the_report():
    """The defect, end to end through the library entry point the GUI calls."""
    sig, es = _events(75, spike_pA=90.0)
    rep = _run(sig, es, _cfg())                      # used to raise sklearn's ValueError

    assert rep.embeddings is None
    said = [w for w in rep.warnings if w.startswith("NO T-SNE:")]
    assert len(said) == 1, rep.warnings
    note = said[0]
    assert "'ALL'" in note                           # which scope
    assert "at least 2 components wide" in note      # why, in our words
    assert "PC1 alone carries" in note and "kurtosis" in note   # what about the data
    assert "nothing else in this report depends on them" in note
    assert "svd_solver" not in note
    # and the run really did produce its findings
    assert rep.clustering is not None and rep.clustering.scopes["ALL"].pca.n_kept == 1
    assert len(rep.event_table()) == 75
    assert rep.text()


def test_the_same_scope_without_the_spike_still_gets_its_coordinates():
    """The guard is not a blanket skip: an ordinary scope is embedded as before."""
    sig, es = _events(75, spike_pA=0.0)
    rep = _run(sig, es, _cfg())
    assert rep.clustering.scopes["ALL"].pca.n_kept >= 2
    assert rep.embeddings is not None
    assert {"tsne_x", "tsne_y"} <= set(rep.embeddings.columns)
    assert not [w for w in rep.warnings if w.startswith("NO T-SNE:")]


def test_an_unforeseen_solver_failure_becomes_a_quoted_note_not_a_lost_run(monkeypatch):
    """A failure we cannot explain is quoted verbatim and named as the solver's."""
    sig, es = _events(75, spike_pA=0.0)

    def _blow_up(*a, **k):
        raise RuntimeError("BLAS died on this platform")

    monkeypatch.setattr(SC, "tsne_embedding", _blow_up)
    rep = _run(sig, es, _cfg())
    note = next(w for w in rep.warnings if w.startswith("NO T-SNE:"))
    assert "failed inside its solver" in note
    assert "RuntimeError: BLAS died on this platform" in note
    assert rep.embeddings is None
    assert len(rep.event_table()) == 75


def test_a_cancelled_embedding_still_cancels_the_run(monkeypatch):
    """A cancelled run is not a report with a missing figure."""
    sig, es = _events(75, spike_pA=0.0)

    def _cancelled(*a, **k):
        raise OperationCancelled("stopped")

    monkeypatch.setattr(SC, "tsne_embedding", _cancelled)
    with pytest.raises(OperationCancelled):
        _run(sig, es, _cfg())


def test_the_note_reaches_the_report_text_the_reader_scrolls_to():
    """The caveat is not merely on the object: it is in the document body."""
    sig, es = _events(75, spike_pA=90.0)
    rep = _run(sig, es, _cfg())
    assert "NO T-SNE:" in rep.text()
