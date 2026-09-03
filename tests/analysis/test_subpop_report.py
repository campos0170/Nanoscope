"""The composition of the three subpopulation modules, and the seams between them.

Two layers, and the second is the one that matters.

**Synthetic** — the seams, the guards and the caveats: that ``CLUST_FEATS`` is one
object and not three copies, that the app's polarity is NEGATED on the way in,
that a single scope collapses onto the pooled scope instead of corroborating
itself, that the floors refuse rather than return an argmax computed from
singular covariances, that progress is monotonic and cancel unwinds, and that two
runs are identical down to the last table.

**Corpus** (``@pytest.mark.corpus``) — the whole chain from float32 samples to the
reported fractions, against the owner's own run. Nobody had ever run the three
modules INTO one another: the clustering was validated by feeding it his stored
feature CSV and the mixture by feeding it the same CSV, so the composition itself
was unverified until this file. It reproduces, exactly:

* all 14 feature columns bit-exact (``max|diff| == 0.0``) over 1488 rows;
* all ten stored label columns exact, NaN pattern included;
* ``n_PCs = 11``, ``K = 4``, BIC argmin ``K = 8``, confidence 0.9861;
* the population table equal to ``subpop_zn_30h_table.csv``.

Reading the oracle uses ``float_precision="round_trip"`` throughout. Pandas'
default reader is not round-trip exact and lands up to 1 ULP off what pandas
wrote, which is enough to make a genuine 1-ULP bug indistinguishable from the
parser.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import event_features as EF
from nanoscope.analysis import subpop_cluster as SC
from nanoscope.analysis import subpop_gmm as SG
from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope
from nanoscope.analysis.events import REQUIRED, EventSet

CORPUS = pathlib.Path("/Users/abrahamcampos/Desktop/Zn2IonCondition/CSVData")
ORACLE = CORPUS / "event_clustering_features.csv"
STORED_TABLE = CORPUS / "subpop_zn_30h_table.csv"
TAGS = ("zn_2h", "zn_10h", "zn_18h", "zn_30h")
HOURS = {"zn_2h": 2, "zn_10h": 10, "zn_18h": 18, "zn_30h": 30}
_HAVE_CORPUS = ORACLE.exists()
corpus = pytest.mark.corpus

FS = 50_000.0


# ================================================================================================
# builders
# ================================================================================================

def _synth(
    n_events: int = 300,
    *,
    n: int = 400_000,
    seed: int = 0,
    depths: tuple[float, float] = (30.0, 60.0),
    sigma: float = 2.0,
    baseline: float = 200.0,
) -> tuple[np.ndarray, EventSet]:
    """A flat recording with two obvious event classes, and the EventSet naming them.

    Two depths and two dwell distributions, so the mixture has something real to
    find; the point of these tests is the plumbing, not the physics.
    """
    rng = np.random.default_rng(seed)
    sig = (baseline + rng.normal(0.0, sigma, n)).astype(np.float32)
    rows = []
    pos = 500
    for i in range(n_events):
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        depth = depths[1] if deep else depths[0]
        sig[pos:pos + dwell] -= depth
        rows.append({
            "event_id": i, "recording": "synth", "pore_id": "p1",
            "start_idx": pos, "end_idx": pos + dwell,
            "t_start_s": pos / FS, "dwell_s": dwell / FS,
            "baseline_pA": baseline, "depth_pA": depth, "frac_depth": depth / baseline,
            "ecd_pA_s": depth * dwell / FS, "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / sigma, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
        pos += int(rng.integers(500, 1100))
        if pos + 200 >= n:
            break
    return sig, EventSet(pd.DataFrame(rows)[REQUIRED])


def _cheap(**kw) -> SR.SubpopConfig:
    """A config small enough for a unit test: 2 folds, K in 1..3, no bootstrap.

    ``with_tsne`` off by default HERE (not in the product default): it adds a
    second or two per call, and this file calls the pipeline three dozen times.
    The embedding stage has its own tests below, which turn it back on.
    """
    base = {"k_grid": (1, 2, 3), "n_splits": 2, "n_boot": 0, "with_tsne": False}
    base.update(kw)
    return SR.SubpopConfig(**base)


def _run(sig, es, *, config=None, **kw) -> SR.SubpopReport:
    return SR.run_subpopulations(
        sig, sig.astype(np.float64), FS, es,
        recording="synth", duration_s=len(sig) / FS,
        config=config if config is not None else _cheap(), **kw,
    )


# ================================================================================================
# (a) THE SEAM -- one feature order, not three copies of it
# ================================================================================================

def test_clust_feats_is_a_single_object_across_all_three_modules():
    """``is``, not ``==``.

    They were three separate thirteen-name literals. Equality would have passed on
    the day they diverged too: the PCA reads its columns BY NAME, so reordering one
    literal produces a differently-ordered loading table and -- through the SVD's
    sign convention -- a differently-signed basis, and nothing raises.
    """
    assert SC.CLUST_FEATS is EF.CLUST_FEATS
    assert SG.SUBPOP_FEATURES is EF.CLUST_FEATS


def test_clust_feats_is_not_the_csv_column_order():
    """``entropy`` is last in the PCA order and eleventh in the table order."""
    assert EF.CLUST_FEATS[-1] == "entropy"
    assert EF.FEATURE_COLUMNS.index("entropy") == 11
    assert set(EF.CLUST_FEATS) == set(EF.FEATURE_COLUMNS) - {"dwell_ms"}
    assert tuple(EF.CLUST_FEATS) != tuple(c for c in EF.FEATURE_COLUMNS if c != "dwell_ms")


# ================================================================================================
# (b) THE OTHER SEAM -- the two polarity conventions are mirror images
# ================================================================================================

@pytest.mark.parametrize("app_pol,expected", [(1, -1), (-1, 1)])
def test_polarity_is_negated_between_the_two_conventions(app_pol, expected):
    """``dev = polarity*(current - baseline)`` vs ``d = baseline - sample``."""
    assert SR.polarity_for_features(app_pol) == expected


@pytest.mark.parametrize("bad", [None, "", "up", float("nan")])
def test_unknown_polarity_reads_as_a_downward_blockade(bad):
    """The whole Zn2 corpus is +1 in ``event_features`` terms; absence defaults there."""
    assert SR.polarity_for_features(bad) == 1


def test_the_wrong_polarity_empties_the_gate_rather_than_biasing_it():
    """The mirrored trace has no downward excursion at all, so nothing passes.

    Worth pinning because it is the *loud* failure. On a recording whose real
    polarity is the other one, the same mistake instead reports the baseline
    noise -- a report that looks entirely plausible.
    """
    sig, es = _synth(120)
    right = SR.feature_table(sig, sig.astype(np.float64), FS, es, recording="s")[1]
    wrong = SR.feature_table(
        sig, sig.astype(np.float64), FS, es, recording="s",
        config=SR.SubpopConfig(polarity=-1),
    )[1]
    assert right.n_kept == len(es.df)
    assert wrong.n_kept == 0


# ================================================================================================
# (c) the gated feature table
# ================================================================================================

def test_feature_table_carries_the_features_and_the_metadata():
    sig, es = _synth(120)
    feats, gate = SR.feature_table(sig, sig.astype(np.float64), FS, es, recording="rec")
    for col in EF.FEATURE_COLUMNS:
        assert col in feats.columns
    for col in SR.META_COLUMNS:
        assert col in feats.columns
    assert len(feats) == gate.n_kept
    assert (feats["recording"] == "rec").all()
    assert set(np.unique(feats["time_half"].to_numpy())) <= {"early", "late"}


def test_feature_table_keeps_the_callers_own_row_labels():
    """The notebook's positional-into-labelled hazard cannot be inherited."""
    sig, es = _synth(120)
    df = es.df.copy()
    df.index = [f"e{i}" for i in range(len(df))]
    feats, gate = SR.feature_table(sig, sig.astype(np.float64), FS, EventSet(df))
    assert list(feats.index) == list(gate.kept_index)
    assert all(str(i).startswith("e") for i in feats.index)


def test_rejected_regions_are_counted_not_hidden():
    """A gate that keeps a tenth of the regions is the finding, not an omission."""
    sig, es = _synth(200)
    # A shallow class the 4-sigma gate should throw out entirely.
    df = es.df.copy()
    sig = sig.copy()
    for row in df.itertuples():
        if row.event_id % 2:
            sig[row.start_idx:row.end_idx] += 60.0 - 1.0   # undo the depth, leave 1 pA
    rep = _run(sig, EventSet(df))
    assert rep.n_fired == len(df)
    assert rep.n_kept < rep.n_fired
    assert any(w.startswith("GATE:") for w in rep.warnings)


def test_a_per_event_scope_of_the_wrong_length_is_refused():
    sig, es = _synth(80)
    with pytest.raises(ValueError, match="BEFORE gating"):
        SR.feature_table(sig, sig.astype(np.float64), FS, es, scope=["a", "b"])


def test_a_per_event_scope_is_aligned_before_the_gate():
    """The caller labels the REGIONS; it does not have to know which survived."""
    sig, es = _synth(160)
    tags = ["odd" if i % 2 else "even" for i in range(len(es.df))]
    feats, gate = SR.feature_table(sig, sig.astype(np.float64), FS, es, scope=tags)
    expected = pd.Series(tags, index=es.df.index).loc[gate.kept_index].to_numpy()
    assert np.array_equal(feats["scope"].to_numpy(), expected)


# ================================================================================================
# (d) one scope is the pooled scope, and says so
# ================================================================================================

def test_a_single_scope_collapses_onto_the_pooled_scope():
    """``ALL`` and ``zn_30h`` over the same 1076 rows are one finding, not two.

    ``iter_scopes`` yields the pooled scope and then one per distinct value, so a
    single-recording table cut on its own name produces two entries with identical
    membership and identical numbers -- which reads as corroboration.
    """
    sig, es = _synth(200)
    rep = _run(sig, es)
    assert list(rep.clustering.scopes) == [SC.SCOPE_ALL]
    assert list(rep.labels.columns) == ["hdbscan", "kmeans3"]
    assert (rep.features["scope"] == SC.SCOPE_ALL).all()


def test_several_scopes_are_kept_apart():
    sig, es = _synth(400, n=700_000)
    tags = ["a" if i < len(es.df) // 2 else "b" for i in range(len(es.df))]
    rep = SR.run_subpopulations(
        sig, sig.astype(np.float64), FS, es, recording="synth",
        duration_s=len(sig) / FS, config=_cheap(), scope=tags,
    )
    assert set(rep.clustering.scopes) == {SC.SCOPE_ALL, "a", "b"}
    assert "hdbscan_a" in rep.labels.columns and "hdbscan_b" in rep.labels.columns


def test_a_scope_below_the_floor_is_recorded_rather_than_dropped_silently():
    sig, es = _synth(200)
    tags = ["tiny" if i < 10 else "main" for i in range(len(es.df))]
    rep = SR.run_subpopulations(
        sig, sig.astype(np.float64), FS, es, recording="synth",
        duration_s=len(sig) / FS, config=_cheap(), scope=tags,
    )
    assert "tiny" in rep.clustering.skipped
    assert any("NOT CLUSTERED" in w for w in rep.warnings)


# ================================================================================================
# (e) the floors -- refusing beats a plausible number
# ================================================================================================

def test_the_mixture_is_refused_below_the_event_floor():
    """A K<=8 full-covariance search on 40 events returns an argmax from singular
    covariances, and nothing about the number it returns would look wrong."""
    sig, es = _synth(40, n=80_000)
    rep = _run(sig, es)
    assert rep.quantification is None
    assert any(w.startswith("NO MIXTURE:") for w in rep.warnings)
    assert SR.MIN_GMM_EVENTS == SC.MIN_SCOPE_N


def test_a_thin_fit_is_flagged_with_its_parameter_count():
    sig, es = _synth(200)
    rep = _run(sig, es)
    assert rep.quantification is not None
    thin = [w for w in rep.warnings if w.startswith("THIN FIT:")]
    assert thin, rep.warnings
    assert "events per parameter" in thin[0]


def test_a_roomier_fit_is_not_flagged_thin():
    sig, es = _synth(SR.SOFT_GMM_EVENTS + 60, n=900_000)
    rep = _run(sig, es, config=_cheap(k_grid=(1, 2)))
    assert len(rep.features) >= SR.SOFT_GMM_EVENTS
    assert not [w for w in rep.warnings if w.startswith("THIN FIT:")]


def test_every_stage_caveat_reaches_the_top_level_list():
    """The sub-results each see only their own; :attr:`warnings` is the union."""
    sig, es = _synth(200)
    rep = _run(sig, es)
    for w in rep.quantification.warnings:
        assert w in rep.warnings
    assert len(set(rep.warnings)) == len(rep.warnings)   # de-duplicated


# ================================================================================================
# (f) determinism, progress, cancel
# ================================================================================================

def test_two_runs_are_identical_down_to_the_last_table():
    sig, es = _synth(200)
    a, b = _run(sig, es), _run(sig, es)
    assert a.text() == b.text()
    ta, tb = a.tables(), b.tables()
    assert list(ta) == list(tb)
    for name in ta:
        pd.testing.assert_frame_equal(ta[name], tb[name], check_exact=True)
    assert np.array_equal(a.gate.keep, b.gate.keep)
    assert a.warnings == b.warnings


def test_the_bootstrap_seed_is_honoured_end_to_end():
    sig, es = _synth(200)
    a = _run(sig, es, config=_cheap(n_boot=8, bootstrap_seed=0))
    b = _run(sig, es, config=_cheap(n_boot=8, bootstrap_seed=0))
    c = _run(sig, es, config=_cheap(n_boot=8, bootstrap_seed=7))
    assert np.array_equal(a.quantification.ci.samples, b.quantification.ci.samples)
    assert not np.array_equal(a.quantification.ci.samples, c.quantification.ci.samples)


def test_progress_is_monotonic_reaches_one_and_never_reports_an_empty_message():
    sig, es = _synth(200)
    seen: list[tuple[float, str]] = []
    _run(sig, es, progress=lambda f, m: seen.append((f, m)))
    assert seen, "a minute-long pipeline that reports nothing is a hang"
    fracs = [f for f, _ in seen]
    assert fracs[0] == pytest.approx(0.0)
    assert fracs[-1] == pytest.approx(1.0)
    assert fracs == sorted(fracs), "a bar that goes backwards is worse than no bar"
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert all(m.strip() for _, m in seen)


def test_progress_moves_during_the_cross_validation_not_only_between_stages():
    """The 480-fit search is the minute that matters; a bar frozen through it is
    indistinguishable from a hang."""
    sig, es = _synth(200)
    seen: list[float] = []
    _run(sig, es, progress=lambda f, m: seen.append(f))
    during = [f for f in seen if SR._P_FEATURES < f < SR._P_CLUSTER + 0.5]
    assert len(during) >= 4, seen


def test_the_ambient_cancel_token_unwinds_the_run():
    sig, es = _synth(200)
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            _run(sig, es)


def test_an_explicit_cancel_token_unwinds_the_run():
    sig, es = _synth(200)
    with pytest.raises(OperationCancelled):
        _run(sig, es, cancel=lambda: True)


# ================================================================================================
# (g) the report and its export
# ================================================================================================

def test_the_report_names_every_item_it_computed():
    sig, es = _synth(200)
    text = _run(sig, es).text()
    for marker in ("ITEMS 1-2", "ITEMS 3-5", "ITEMS 6-7", "ITEMS 8-10",
                   "ITEM 11", "ITEM 12", "ITEM 13", "CAVEATS"):
        assert marker in text, marker


def test_the_report_says_out_loud_what_item_13_does_and_does_not_export():
    """The caveat has to describe what these FILES carry, and it once said more than that.

    While the galleries and the trace overlays did not exist, the honest sentence was "not
    in this pass". They exist now — the gallery filters by the ``subpop`` column and the
    trace spans are coloured by it — so that sentence became a false caveat, which is worse
    than none: a reader who trusts the warnings would have gone looking for a surface the
    app already has. What is still true, and is what it says now, is that both are VIEWS:
    no picture of either leaves in the export, and the ``subpop`` column is what a reader
    rebuilds them from elsewhere.
    """
    sig, es = _synth(200)
    rep = _run(sig, es)
    note = next(w for w in rep.warnings if "ITEM 13 IS PARTIAL" in w)
    assert "subpop" in note, "the caveat must name the column both views are drawn from"
    assert "not in this pass" not in note, (
        "the galleries and the trace overlays exist; a caveat that still says they do not "
        "is a false caveat")

    # THE HOLE THIS CLOSES. The three assertions above read the WARNING only, and
    # `assert "galleries" in rep.text()` passed happily while the ITEM 13 body of the
    # very same document ended "The raw-event GALLERIES and the rendered trace
    # overlays (full map + zoom) are NOT in this pass -- see the caveats." So the
    # report contradicted itself: the body said the views do not exist, the caveat
    # beside it said they do, and two enabled buttons on the window proved the
    # caveat right. Case-insensitive, because the body shouted the NOT.
    body = rep.text()
    assert "not in this pass" not in body.lower(), (
        "the report BODY still tells the reader the galleries and trace overlays do not "
        "exist, while the caveat in the same document says they do")
    assert "galleries" in body
    assert "Colour the trace" in body, (
        "the body should name the button that opens the view it is describing")


def test_the_report_states_the_gate_that_produced_its_population():
    sig, es = _synth(200)
    text = _run(sig, es).text()
    assert "sustained gate" in text
    assert "event-excluded" in text


def test_the_event_table_carries_the_subpopulation_and_its_posterior():
    sig, es = _synth(200)
    rep = _run(sig, es)
    tbl = rep.event_table()
    assert "subpop" in tbl.columns and "subpop_confidence" in tbl.columns
    assert "hdbscan" in tbl.columns and "kmeans3" in tbl.columns
    assert tbl.index.equals(rep.features.index)
    assert tbl["subpop_confidence"].between(0.0, 1.0).all()


def test_write_report_writes_the_text_the_sidecar_and_every_table(tmp_path):
    sig, es = _synth(200)
    rep = _run(sig, es)
    written = SR.write_report(rep, tmp_path / "out")
    assert written["text"].read_text().startswith("=")
    meta = json.loads(written["meta"].read_text())
    assert meta["warnings"] == list(rep.warnings)
    assert meta["config"]["random_state"] == rep.meta["config"]["random_state"]
    for name in rep.tables():
        assert written[name].exists() and written[name].stat().st_size > 0
    assert set(SR.report_paths(tmp_path / "out", tuple(rep.tables()))) == set(written.values())


def test_the_sidecar_carries_the_caveats_so_an_exported_csv_is_never_stripped_of_them(tmp_path):
    """A CSV of fractions without the label-switching warning beside it is exactly
    the artefact this pipeline must not emit."""
    sig, es = _synth(200)
    rep = _run(sig, es, config=_cheap(n_boot=8))
    written = SR.write_report(rep, tmp_path / "out")
    meta = json.loads(written["meta"].read_text())
    assert meta["warnings"]
    assert written["populations"].exists()


def test_an_export_that_fails_part_way_leaves_nothing_behind(tmp_path, monkeypatch):
    """Eight files under one stem: they land together or not at all.

    A folder holding today's ``populations.csv`` beside last week's ``events.csv``
    is worse than no export -- it is indistinguishable from a consistent one, and
    the disagreement between the two is silent.
    """
    sig, es = _synth(200)
    rep = _run(sig, es)
    calls = {"n": 0}
    real = pd.DataFrame.to_csv

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full, halfway through")
        return real(self, *a, **kw)

    monkeypatch.setattr(pd.DataFrame, "to_csv", flaky)
    with pytest.raises(OSError):
        SR.write_report(rep, tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


def test_an_export_over_an_existing_one_replaces_every_member(tmp_path):
    """The second half of the same promise: a re-export is not a merge."""
    sig, es = _synth(200)
    rep = _run(sig, es)
    stale = tmp_path / "out_events.csv"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("this,is,last,weeks,run\n")
    written = SR.write_report(rep, tmp_path / "out")
    assert "last,weeks,run" not in written["events"].read_text()


def test_report_paths_without_names_still_lists_the_two_that_are_always_written(tmp_path):
    paths = SR.report_paths(tmp_path / "out")
    assert [p.name for p in paths] == ["out.txt", "out.meta.json"]


def test_exported_csvs_carry_an_index_only_where_it_is_a_value(tmp_path):
    """An unnamed integer column in an exported CSV is one more thing to mistake
    for data; a feature name or a subpopulation id is not."""
    sig, es = _synth(200)
    rep = _run(sig, es)
    written = SR.write_report(rep, tmp_path / "out")
    events = pd.read_csv(written["events"])
    assert list(events.columns)[0] != "Unnamed: 0"
    assert "event_id" in events.columns and "subpop" in events.columns
    loadings = pd.read_csv(written["loadings_ALL"], index_col=0)
    assert set(loadings.index) == set(EF.CLUST_FEATS)
    grid = pd.read_csv(written["cv_grid"], index_col=0)
    assert grid.index.name == "n_pcs"


# ================================================================================================
# (h) item 13's half -- the median waveforms
# ================================================================================================

def test_waveforms_give_one_median_trace_per_subpopulation():
    sig, es = _synth(200)
    rep = _run(sig, es)
    w = rep.waveforms
    assert len(w) == SR.WAVEFORM_POINTS
    cols = [c for c in w.columns if c != "T"]
    assert len(cols) == rep.quantification.fit.k
    assert sum(w.attrs["n_events"].values()) == len(rep.features)
    assert w["T"].iloc[0] < 0.0 < 1.0 < w["T"].iloc[-1]   # padding either side


def test_waveforms_refuse_labels_for_events_they_do_not_have():
    sig, es = _synth(80)
    labels = pd.Series([0] * 3, index=["nope", "nope2", "nope3"])
    with pytest.raises(ValueError, match="not in the event set"):
        SR.subpopulation_waveforms(sig, FS, es, labels)


def test_waveforms_can_be_switched_off():
    sig, es = _synth(200)
    rep = _run(sig, es, config=_cheap(with_waveforms=False))
    assert rep.waveforms is None
    assert "Not computed" in rep.text()


# ================================================================================================
# (h2) items 6-7 -- exported, never printed, never gated on
# ================================================================================================

def test_tsne_is_on_by_default_and_lands_in_the_events_table():
    sig, es = _synth(200)
    rep = _run(sig, es, config=_cheap(with_tsne=True))
    assert rep.embedding_names() == ("t-SNE",)
    assert list(rep.embeddings.columns) == ["tsne_x", "tsne_y"]
    assert rep.embeddings.index.equals(rep.features.index)
    tbl = rep.event_table()
    assert "tsne_x" in tbl.columns and "tsne_y" in tbl.columns


def test_the_report_prints_no_embedding_coordinate():
    """A coordinate on a page invites exactly the reading these cannot support."""
    sig, es = _synth(200)
    rep = _run(sig, es, config=_cheap(with_tsne=True))
    text = rep.text()
    assert "ITEMS 6-7" in text and "t-SNE" in text
    for value in rep.embeddings["tsne_x"].head(5):
        assert f"{value:.4f}" not in text


def test_umap_is_off_by_default_and_the_report_says_why():
    sig, es = _synth(200)
    rep = _run(sig, es, config=_cheap(with_tsne=True))
    assert "UMAP" not in rep.embedding_names()
    assert "not bit-reproducible" in rep.text()
    assert SR.SubpopConfig().with_umap is False
    assert SR.SubpopConfig().with_tsne is True


def test_a_scope_above_the_embedding_ceiling_is_skipped_with_a_reason():
    """A figure aid must not become the reason a report takes ten minutes."""
    sig, es = _synth(200)
    rep = _run(sig, es, config=_cheap(with_tsne=True))
    table, notes = SR.scope_embeddings(rep.clustering, tsne=True, max_events=10)
    assert table is None
    assert notes and "ceiling" in notes[0]


def test_embedding_an_absent_scope_says_so_rather_than_returning_nothing():
    sig, es = _synth(200)
    rep = _run(sig, es)
    table, notes = SR.scope_embeddings(rep.clustering, scope="nope", tsne=True)
    assert table is None
    assert notes and "was not clustered" in notes[0]


def test_asking_for_no_embedding_is_silent():
    sig, es = _synth(200)
    rep = _run(sig, es)
    assert SR.scope_embeddings(rep.clustering, tsne=False, umap=False) == (None, ())


# ================================================================================================
# (i) the frozen contracts
# ================================================================================================

def test_the_event_schema_is_untouched_by_a_whole_run():
    """The thirteen features are DERIVED. Nothing is written back."""
    sig, es = _synth(200)
    before = list(es.df.columns)
    snapshot = es.df.copy(deep=True)
    rep = _run(sig, es)
    assert list(es.df.columns) == before == REQUIRED
    pd.testing.assert_frame_equal(es.df, snapshot)
    for name in EF.FEATURE_COLUMNS:
        if name not in REQUIRED:
            assert name not in es.df.columns
    assert rep.n_kept > 0


def test_the_report_module_imports_no_gui():
    import ast

    tree = ast.parse(pathlib.Path(SR.__file__).read_text())
    for node in ast.walk(tree):
        mod = getattr(node, "module", None) or ""
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [mod] + [a.name for a in getattr(node, "names", [])]
            assert not any("gui" in n.split(".") or "PySide6" in n for n in names if n)


# ================================================================================================
# (j) THE ORACLE -- the whole chain, on the owner's own recordings
# ================================================================================================

@pytest.fixture(scope="module")
def zn_composed():
    """The three modules run INTO one another over all four Zn2 recordings.

    Module-scoped: the four traces are ~1.5 M samples each and the mixture stage
    is 480 fits plus a 200-resample bootstrap. Everything below reads this one run.
    """
    if not _HAVE_CORPUS:
        pytest.skip("Zn2IonCondition research data not present")
    frames = []
    for tag in TAGS:
        raw_df = pd.read_csv(
            CORPUS / f"{tag}.csv.gz", compression="gzip",
            usecols=["current_raw_pA", "current_detr_pA"],
        )
        meta = json.loads((CORPUS / f"{tag}_metadata.json").read_text())
        fs = float(meta["fs_hz"])
        # float32 for the detrended trace, exactly as the notebook: widening it moves
        # the per-recording median and MAD in their last bits.
        detr = raw_df["current_detr_pA"].values.astype(np.float32)
        raw = raw_df["current_raw_pA"].values
        del raw_df
        cat = pd.read_csv(
            CORPUS / f"events_catalog_{tag}.csv", float_precision="round_trip")
        win = pd.DataFrame(
            {"start_idx": np.round(cat["onset_s"].to_numpy() * fs).astype(np.int64),
             "end_idx": np.round(cat["end_s"].to_numpy() * fs).astype(np.int64)},
            index=cat.index,
        )
        feats, gate = SR.feature_table(
            detr, raw, fs, win, recording=tag, scope=str(HOURS[tag]))
        # `feature_table` is the composable primitive and keeps the caller's own
        # scope name; only `run_subpopulations` collapses a lone scope onto ALL.
        assert (feats["scope"] == str(HOURS[tag])).all()
        feats["incubation_h"] = HOURS[tag]
        feats["duration_s"] = float(meta["duration_s"])
        feats["onset_s"] = cat.loc[gate.kept_index, "onset_s"].to_numpy()
        feats["time_half"] = EF.time_half(feats["onset_s"].to_numpy())
        frames.append(feats.reset_index(drop=True))
    table = pd.concat(frames, ignore_index=True)
    clustering = SC.cluster_scopes(table, scope_column="scope")
    labels = clustering.label_table()
    sub = table[table["scope"] == "30"]
    quant = SG.quantify_subpopulations(
        sub,
        duration_s=float(sub["duration_s"].iloc[0]),
        exploratory_labels=labels.loc[sub.index, "hdbscan_30"],
        exploratory_name="hdbscan_30",
    )
    return table, labels, quant


@corpus
def test_oracle_the_composed_features_are_bit_exact(zn_composed):
    """Not "close": ``max|diff| == 0.0`` on every one of 14 columns, 1488 rows."""
    table, _labels, _q = zn_composed
    oracle = pd.read_csv(ORACLE, float_precision="round_trip")
    assert len(table) == len(oracle) == 1488
    for col in EF.FEATURE_COLUMNS:
        diff = np.max(np.abs(table[col].to_numpy() - oracle[col].to_numpy()))
        assert diff == 0.0, f"{col}: max|diff| = {diff!r}"
    assert np.array_equal(table["time_half"].to_numpy(), oracle["time_half"].to_numpy())
    assert np.array_equal(table["onset_s"].to_numpy(), oracle["onset_s"].to_numpy())


@corpus
def test_oracle_the_composed_labels_reproduce_all_ten_stored_columns(zn_composed):
    """Label for label, NaN pattern included -- NaN is 'not in this scope' and
    ``-1`` is 'in this scope, and HDBSCAN called it noise'."""
    _table, labels, _q = zn_composed
    oracle = pd.read_csv(ORACLE, float_precision="round_trip")
    expected = ["hdbscan", "hdbscan_10", "hdbscan_18", "hdbscan_2", "hdbscan_30",
                "kmeans3", "kmeans3_10", "kmeans3_18", "kmeans3_2", "kmeans3_30"]
    assert list(labels.columns) == expected
    for col in expected:
        got = labels[col].to_numpy(dtype=np.float64)
        exp = oracle[col].to_numpy(dtype=np.float64)
        assert ((got == exp) | (np.isnan(got) & np.isnan(exp))).all(), col


@corpus
def test_oracle_the_composed_gate_keeps_his_counts(zn_composed):
    """108 / 110 / 194 / 1076 out of 408 / 3945 / 1647 / 11004."""
    table, _labels, _q = zn_composed
    counts = table.groupby("recording").size().to_dict()
    assert counts == {"zn_2h": 108, "zn_10h": 110, "zn_18h": 194, "zn_30h": 1076}


@corpus
def test_oracle_the_composed_mixture_selects_eleven_pcs_and_four_components(zn_composed):
    _table, _labels, q = zn_composed
    assert (q.selection.n_pcs, q.selection.k) == (11, 4)
    assert q.selection.loglik == pytest.approx(16.4090, abs=5e-4)
    assert q.fit.bic_k == 8, "the BIC cross-check overfits here, and must keep saying so"
    assert q.fit.conf_frac == pytest.approx(0.9861, abs=5e-4)
    assert q.fit.sizes == {0: 387, 1: 295, 2: 269, 3: 125}


@corpus
def test_oracle_the_composed_population_table_equals_his_stored_one(zn_composed):
    if not STORED_TABLE.exists():
        pytest.skip(f"owner's population table absent: {STORED_TABLE}")
    _table, _labels, q = zn_composed
    stored = pd.read_csv(STORED_TABLE, float_precision="round_trip")
    pd.testing.assert_frame_equal(q.table, stored, atol=0, rtol=0)


@corpus
def test_oracle_the_composed_crosstab_equals_his(zn_composed):
    _table, _labels, q = zn_composed
    assert q.crosstab.to_numpy().tolist() == [
        [2, 385, 0], [44, 251, 0], [70, 156, 43], [113, 12, 0],
    ]


@corpus
def test_oracle_the_bootstrap_intervals_are_still_flagged_uncalibrated(zn_composed):
    """His own run carries the label-switching signature. Reproducing the pipeline
    means reproducing the warning, not smoothing it away."""
    _table, _labels, q = zn_composed
    assert any("LABEL SWITCHING" in w for w in q.warnings)


# ================================================================================================
# the composable primitives take the SAME two forms at every stage
# ================================================================================================

def test_waveforms_take_a_bare_frame_like_feature_table_does():
    """The documented first stage accepts a frame, so the last one must too.

    ``feature_table`` does ``getattr(events, "df", events)`` and says so in its
    docstring. ``subpopulation_waveforms`` used to reach straight for
    ``events.df``, so a script composing the public primitives ran the gate, the
    13 features, the per-scope clustering, the cross-validation and the bootstrap
    -- and then lost all of it to an ``AttributeError`` naming neither the
    argument nor the fix. Same object, both ends, or the primitives do not compose.
    """
    sig, es = _synth(60)
    labels = pd.Series((es.df.index % 2).astype(int), index=es.df.index)

    from_set = SR.subpopulation_waveforms(sig, FS, es, labels, n_points=32)
    from_frame = SR.subpopulation_waveforms(sig, FS, es.df, labels, n_points=32)

    pd.testing.assert_frame_equal(from_frame, from_set, check_exact=True)
    assert list(from_frame.columns) == ["T", "subpop_0", "subpop_1"]
    assert from_frame.attrs["n_events"] == from_set.attrs["n_events"]


def test_waveforms_on_a_bare_frame_still_reject_labels_naming_absent_events():
    """Losing the EventSet must not also lose the alignment check -- that check is
    the one thing standing between a mislabelled index and a silently wrong plot."""
    sig, es = _synth(40)
    labels = pd.Series([0, 1], index=[es.df.index[0], 10_000])
    with pytest.raises(ValueError, match="not in the event set"):
        SR.subpopulation_waveforms(sig, FS, es.df, labels, n_points=16)


def test_a_dataframe_survives_the_whole_pipeline_end_to_end():
    """The regression proper: a frame in at the top, a full report out the bottom,
    waveforms included. Cheap config, because the point is that it FINISHES."""
    sig, es = _synth(120)
    rep = SR.run_subpopulations(
        sig, sig.astype(np.float64), FS, es.df,
        recording="synth", duration_s=len(sig) / FS, config=_cheap(),
    )
    assert rep.waveforms is not None and len(rep.waveforms) > 0


# ================================================================================================
# the cost ceiling
# ================================================================================================

def test_slow_gmm_note_is_silent_at_the_owners_own_event_count():
    """His zn_30h is 1076 events and takes about two minutes. Warning there would
    train him to click through the warning, which is worse than not having one."""
    assert SR.slow_gmm_note(1076) is None
    assert SR.slow_gmm_note(SR.SLOW_GMM_EVENTS) is None
    assert SR.SLOW_GMM_EVENTS > 1076, "the owner's own run must never be flagged"
    # 2000 events measured at 7 minutes quiet, 42 under load; the threshold has to
    # sit below the first size that behaves like that.
    assert SR.SLOW_GMM_EVENTS < 2000


def test_slow_gmm_note_fires_above_the_ceiling_and_says_cancelling_is_safe():
    note = SR.slow_gmm_note(SR.SLOW_GMM_EVENTS + 1)
    assert note is not None
    assert str(SR.SLOW_GMM_EVENTS + 1) in note
    # The two things a reader needs: that nothing is silently subsampled, and
    # that they can get out. Without the first, the numbers look negotiable.
    assert "Nothing is subsampled" in note
    assert "cancelling" in note.lower()
    # The measured anchors, so the message cannot drift away from the constant.
    assert "1076" in note and "11000" in note


def test_the_ceiling_sits_above_the_floor():
    """A ceiling below the thin-fit floor would warn both ways at once."""
    assert SR.SOFT_GMM_EVENTS < SR.SLOW_GMM_EVENTS


def test_a_slow_run_warns_in_the_report_and_on_the_progress_sink(monkeypatch):
    """The warning has to reach the caller BEFORE the 480 fits, not only after."""
    monkeypatch.setattr(SR, "SLOW_GMM_EVENTS", 10)
    sig, es = _synth(120)
    seen: list[tuple[float, str]] = []
    rep = _run(sig, es, progress=lambda f, m: seen.append((f, m)))

    assert any("SLOW:" in w for w in rep.warnings)
    said = [m for _f, m in seen if "many minutes" in m]
    assert said, f"no pre-warning on the progress sink; got {[m for _f, m in seen]}"


# ================================================================================================
# caveats must describe THIS run
# ================================================================================================

def test_item13_caveat_does_not_promise_waveforms_that_were_not_computed():
    """With ``with_waveforms`` off there is no waveforms table in ``tables()`` and
    none in the export, so a caveat saying they are "here (and exported)" sends a
    reader looking for a file that was never written."""
    sig, es = _synth(120)
    rep = _run(sig, es, config=_cheap(with_waveforms=False))

    assert rep.waveforms is None
    item13 = [w for w in rep.warnings if "ITEM 13" in w]
    assert len(item13) == 1, item13
    assert "ABSENT" in item13[0]
    assert "MEDIAN WAVEFORMS are here" not in item13[0]


def test_item13_caveat_still_claims_the_waveforms_when_they_are_there():
    sig, es = _synth(120)
    rep = _run(sig, es, config=_cheap(with_waveforms=True))

    assert rep.waveforms is not None
    item13 = [w for w in rep.warnings if "ITEM 13" in w]
    assert len(item13) == 1 and "PARTIAL" in item13[0]
