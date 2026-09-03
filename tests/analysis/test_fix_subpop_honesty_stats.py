"""H4 + M8 — the subpopulation pipeline may not state a class count without its evidence.

WHAT WAS WRONG, measured through the real GUI on the owner's own recording
(500 pM apoferritin, 1x PBS, 22 nm pore, -700 mV, 985 gated events):

    header : "7 subpopulation(s) at n_PCs=11 (84.4 % confidently assigned) · 5 caveats"
    status : "... 7 subpopulation(s) at n_PCs=11; 5 caveat(s)."

and the same partition, measured in the eleven-dimensional space it was fitted
in: mean silhouette **0.098**, 27.0 % of events with a NEGATIVE silhouette,
31.5 % sitting nearer another class's centre, subpop 6 (n=19) at -0.220, HDBSCAN
finding **0 clusters and 100 % noise** on the identical cloud, and plain k-means
at K=3 on the same coordinates scoring 0.197 -- twice the reported model. None
of those numbers appeared anywhere. The window printed a silhouette for the
exploratory k-means and **none for the model it led with**, and the closest
thing to a warning ("HDBSCAN found no cluster at all ... that reads as a
continuum") ended with the words "not a failure" as caveat 2 of 6 inside a
9,238-character document.

The invariant these tests defend is not "K=7 is wrong". The pipeline must stay
able to say yes -- on Zn2-insulin it reports K=4 usefully -- and it must stay
able to say no, which it already does: on -600 mV, given one population, it
chose K=1. The invariant is:

    **A REPORT'S HEADLINE MAY NOT BE MORE CONFIDENT THAN ITS OWN CAVEATS.**

So: the mixture's own separation is measured (it never was), the class count is
not formattable anywhere without it, the density method's continuum verdict is
promoted to where the count is stated, and ``early_frac`` -- the one statistic
in the app that can see this recording's non-stationarity, printed under a
caption saying what an extreme value means and then compared against nothing --
is finally read.

M8 specifically: ``subpop_gmm``'s only ``NO TIME-STABILITY CHECK`` raise fired
when the ``time_half`` COLUMN WAS ABSENT and never when the VALUE was extreme.
On -500 mV subpop 0 read 0.08 and subpop 4 read 0.05 -- together 341 of 848
events -- and subpop 2 read 0.99. Every one of them was printed and left unread.

NOTHING HERE ASSERTS A NUMBER MOVED. The last test in this file pins the
opposite: the mixture's labels, K, n_PCs and held-out likelihood are bit-equal
to what the same seed produces without any of this machinery, because all of it
reads a finished fit and refits nothing.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

import nanoscope
from nanoscope.analysis import subpop_gmm as SG
from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import REQUIRED, EventSet

FS = 50_000.0


# ===========================================================================
# 0. WHERE DID `nanoscope` COME FROM?
# ===========================================================================
#
# This file is also run against an out-of-tree REVERT of the H4 hunks, where
# every behavioural test below must fail. A revert stub with the wrong signature
# once manufactured four fake failures in this project and was believed, so this
# test exists to be the one that PASSES in both trees: it proves the failures
# came from a real, whole, correctly-resolved package rather than from an
# import that did not happen.


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    for mod in (SG, SR):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents, f"{mod.__name__} at {p} is not under {root}"
        assert p.stat().st_size > 20_000, f"{p} is {p.stat().st_size} B — a stub?"
    print(f"\n[H4] nanoscope resolved from {root}")


# ===========================================================================
# fixtures — real fits, never hand-written stubs
# ===========================================================================

_FEATS = list(SG.SUBPOP_FEATURES)


def _frame(X: np.ndarray, early: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=_FEATS)
    df["dwell_ms"] = np.abs(df[_FEATS[0]]) * 5.0 + 10.0
    if early is not None:
        df["time_half"] = np.where(np.asarray(early, dtype=bool), "early", "late")
    return df


@pytest.fixture(scope="module")
def one_cloud_cut_into_four():
    """ONE Gaussian, forced to K=4. The exact situation the report could not see.

    Not a contrived pathology: it is what a mixture does to a continuum, and
    every criterion the pipeline had before this pass is satisfied by it.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(600, len(_FEATS)))
    Xs = SG.scale_features(_frame(X), tuple(_FEATS))
    fit = SG.fit_subpopulations(
        Xs, n_pcs=6, k=4, index=pd.RangeIndex(600), random_state=0,
        n_init=3, bic_check=False,
    )
    return fit


@pytest.fixture(scope="module")
def four_real_blobs():
    """Four genuinely separated blobs — the pipeline must NOT caveat these."""
    rng = np.random.default_rng(11)
    centres = rng.normal(size=(4, len(_FEATS))) * 12.0
    lab = np.repeat(np.arange(4), 150)
    X = centres[lab] + rng.normal(size=(600, len(_FEATS))) * 0.4
    Xs = SG.scale_features(_frame(X), tuple(_FEATS))
    fit = SG.fit_subpopulations(
        Xs, n_pcs=6, k=4, index=pd.RangeIndex(600), random_state=0,
        n_init=3, bic_check=False,
    )
    return fit


# ===========================================================================
# 1. the statistic that did not exist
# ===========================================================================


def test_separation_is_measured_in_the_space_the_mixture_was_fitted_in(
        one_cloud_cut_into_four):
    """Not in the two coordinates the scatter draws.

    A claim about seven classes in eleven dimensions is answerable with "the
    picture is only a projection" if it is checked in the picture. So the
    silhouette is over ``fit.scores`` exactly — asserted by recomputing it.
    """
    from sklearn.metrics import silhouette_score

    fit = one_cloud_cut_into_four
    sep = SG.measure_separation(fit, random_state=0)
    assert sep is not None
    assert sep.n_pcs == fit.n_pcs == 6
    assert not sep.subsampled
    expected = float(silhouette_score(fit.scores, fit.labels.to_numpy()))
    assert sep.silhouette == pytest.approx(expected, abs=1e-12)
    # and it is emphatically NOT the drawn plane's number
    drawn = float(silhouette_score(fit.scores[:, :2], fit.labels.to_numpy()))
    assert sep.silhouette != pytest.approx(drawn, abs=1e-9)


def test_nearer_other_centre_counts_every_event_and_is_not_the_posterior(
        one_cloud_cut_into_four):
    """A wide component can win the posterior on an event nearer another centre.

    So this is a genuinely different number from ``conf_frac`` and is computed
    over ALL events (it is O(n*k), so there is nothing to subsample).
    """
    fit = one_cloud_cut_into_four
    sep = SG.measure_separation(fit, random_state=0)
    d = np.linalg.norm(fit.scores[:, None, :] - fit.means[None, :, :], axis=2)
    expected = float((d.argmin(axis=1) != fit.labels.to_numpy()).mean())
    assert sep.nearer_other_fraction == pytest.approx(expected, abs=1e-12)
    assert sep.nearer_other_fraction > 0.0


def test_a_reference_partition_is_scored_on_the_same_coordinates(
        one_cloud_cut_into_four):
    """The yardstick for "how separated can this cloud be made at all".

    The report already printed a silhouette for the exploratory k-means — in a
    DIFFERENT space (the scope's own 95 %-variance PCA), so it could not be
    compared with anything. This one is in the mixture's coordinates.
    """
    sep = SG.measure_separation(one_cloud_cut_into_four, random_state=0)
    assert sep.reference_k == SG.SEPARATION_REFERENCE_K == 3
    assert sep.reference_silhouette is not None
    assert -1.0 <= sep.reference_silhouette <= 1.0


def test_separation_is_none_at_k1_rather_than_a_fabricated_zero():
    rng = np.random.default_rng(3)
    Xs = SG.scale_features(_frame(rng.normal(size=(200, len(_FEATS)))), tuple(_FEATS))
    fit = SG.fit_subpopulations(
        Xs, n_pcs=4, k=1, index=pd.RangeIndex(200), random_state=0,
        n_init=2, bic_check=False)
    assert SG.measure_separation(fit, random_state=0) is None


def test_above_the_cap_the_silhouette_subsamples_deterministically_and_says_so(
        one_cloud_cut_into_four):
    fit = one_cloud_cut_into_four
    a = SG.measure_separation(fit, random_state=0, max_n=200)
    b = SG.measure_separation(fit, random_state=0, max_n=200)
    assert a.subsampled and a.n_used == 200 and a.n_events == 600
    assert a.silhouette == b.silhouette          # seeded, so a re-run agrees
    assert set(a.per_subpop) == set(range(fit.k))  # no class lost by the subsample
    # the nearest-centre number is still over every event
    assert a.nearer_other_fraction == pytest.approx(
        SG.measure_separation(fit, random_state=0).nearer_other_fraction, abs=1e-12)


# ===========================================================================
# 2. the caveats that must fire before a count is quoted
# ===========================================================================


def test_one_cloud_cut_into_four_is_reported_as_not_separated(one_cloud_cut_into_four):
    fit = one_cloud_cut_into_four
    sep = SG.measure_separation(fit, random_state=0)
    assert sep.silhouette <= SG.SEPARATION_FLOOR
    assert sep.verdict == "NO SUBSTANTIAL STRUCTURE"
    assert not sep.substantial
    notes = SG._separation_warnings(sep, fit.k)
    assert any(n.startswith("NOT SEPARATED:") for n in notes), notes
    text = "\n".join(notes)
    assert f"{sep.silhouette:.3f}" in text
    # it must say WHY the other criteria do not cover this
    assert "BIC" in text and "posterior" in text


def test_four_real_blobs_raise_no_separation_caveat_at_all(four_real_blobs):
    """The pipeline must not become a machine that refuses everything."""
    sep = SG.measure_separation(four_real_blobs, random_state=0)
    assert sep.substantial and sep.silhouette > 0.5
    assert SG._separation_warnings(sep, four_real_blobs.k) == ()


def test_a_class_that_lives_inside_another_class_is_named(one_cloud_cut_into_four):
    """Subpop 6 on the owner's -700 mV run scored -0.220 with n=19 (n=53, -0.234
    after the baseline change that landed alongside this pass)."""
    fit = one_cloud_cut_into_four
    sep = SG.measure_separation(fit, random_state=0)
    negatives = [c for c, (_, s, _) in sep.per_subpop.items() if s < 0]
    assert negatives, "fixture guard: cutting one cloud must produce a negative class"
    notes = "\n".join(SG._separation_warnings(sep, fit.k))
    assert "CLASS(ES) INSIDE ANOTHER CLASS" in notes
    for c in negatives:
        assert f"subpop {c} " in notes


def _sep(sil: float, ref: float | None, *, k: int = 4, neg: float = 0.2) -> SG.Separation:
    """A hand-built :class:`Separation`, so the GATES are tested without sklearn.

    The measurement is pinned against sklearn above; the thresholds are decisions
    and must be testable at exactly their edges rather than at whatever a
    synthetic cloud happens to score this week.
    """
    return SG.Separation(
        silhouette=sil, neg_fraction=neg, nearer_other_fraction=0.3,
        per_subpop={i: (50, sil, 0.3) for i in range(k)},
        reference_k=3, reference_silhouette=ref,
        n_used=200, n_events=200, n_pcs=11,
    )


def test_a_less_separated_mixture_than_plain_kmeans_says_so():
    notes = "\n".join(SG._separation_warnings(_sep(0.05, 0.20), 7))
    assert "NOT THE BEST-SEPARATED PARTITION ON OFFER" in notes
    assert "0.200" in notes and "0.050" in notes
    # ...and it must refuse to be read as a recommendation to report K=3
    assert "NOT a recommendation" in notes


def test_the_rival_comparison_stays_quiet_on_a_partition_that_is_actually_good():
    """A caveat list that fires on good results is wallpaper.

    Two good partitions of a real structure (0.62 vs 0.68) is not a finding, and
    the comparison must not fire there — that is why it is gated on the band and
    not on the sign of the difference alone.
    """
    notes = "\n".join(SG._separation_warnings(_sep(0.62, 0.68), 4))
    assert notes == "" or "NOT THE BEST-SEPARATED" not in notes


def test_the_bands_are_read_off_the_cited_scale_not_invented():
    assert SG.SEPARATION_FLOOR == 0.25 and SG.SEPARATION_WEAK_CEILING == 0.50
    assert _sep(0.25, None).verdict == "NO SUBSTANTIAL STRUCTURE"
    assert _sep(0.26, None).verdict == "WEAK, COULD BE ARTIFICIAL"
    assert _sep(0.51, None).verdict == "REASONABLY SEPARATED"
    assert _sep(0.80, None).verdict == "STRONGLY SEPARATED"
    assert "WEAKLY SEPARATED" in "\n".join(SG._separation_warnings(_sep(0.30, None), 4))
    assert "NOT SEPARATED" in "\n".join(SG._separation_warnings(_sep(0.25, None), 4))


# ===========================================================================
# 3. M8 — early_frac is finally READ, not only printed
# ===========================================================================


def _table(early_fracs, ns):
    return pd.DataFrame({
        "subpop": list(range(len(ns))),
        "n": list(ns),
        "early_frac": list(early_fracs),
    })


def test_m8_an_extreme_early_frac_fires_not_only_a_missing_column():
    """THE M8 DEFECT. -500 mV: subpop 0 = 0.08, subpop 4 = 0.05, subpop 2 = 0.99.

    Before this pass the ONLY ``NO TIME-STABILITY CHECK`` raise in the module
    fired when the ``time_half`` column was ABSENT. Present-and-extreme was
    rounded to two decimals, printed under a caption explaining what it means,
    and compared against nothing.
    """
    notes = SG._stability_warnings(_table([0.08, 0.52, 0.99, 0.48, 0.05],
                                          [204, 190, 165, 152, 137]))
    text = "\n".join(notes)
    assert "TIME-UNSTABLE CLASS(ES)" in text, notes
    for sp in (0, 2, 4):
        assert f"subpop {sp} " in text
    assert "subpop 1 " not in text and "subpop 3 " not in text
    assert "0.08" in text and "0.99" in text and "0.05" in text


def test_m8_a_class_lopsided_the_other_way_fires_too():
    notes = "\n".join(SG._stability_warnings(_table([0.19, 0.53], [129, 856])))
    assert "TIME-UNSTABLE CLASS(ES)" in notes and "subpop 0 " in notes


def test_m8_stable_classes_fire_nothing():
    assert SG._stability_warnings(_table([0.48, 0.53, 0.57, 0.37], [236, 190, 165, 117])) == ()


def test_m8_a_non_stationary_recording_is_called_out_separately():
    """A lopsided RECORDING and a lopsided CLASS want different responses.

    If the event rate itself changed, no class is "present throughout" and every
    early_frac must be read against the pooled value rather than against 0.50.
    """
    notes = "\n".join(SG._stability_warnings(_table([0.10, 0.12, 0.09], [300, 300, 300])))
    assert "THE RECORDING IS NOT STATIONARY" in notes
    assert "0.10" in notes  # the pooled value the reader must compare against


def test_m8_the_missing_column_check_still_fires_and_the_new_one_stays_quiet():
    """No regression: absent column keeps its own caveat and gains no false one."""
    rng = np.random.default_rng(5)
    df = _frame(rng.normal(size=(220, len(_FEATS))))     # no time_half
    q = SG.quantify_subpopulations(
        df, duration_s=10.0, k_grid=(1, 2, 3), n_splits=3, n_boot=0,
        cv_n_init=1, fit_n_init=2, bic_n_init=1)
    text = "\n".join(q.warnings)
    assert "NO TIME-STABILITY CHECK" in text
    assert "TIME-UNSTABLE CLASS(ES)" not in text
    assert "THE RECORDING IS NOT STATIONARY" not in text


# ===========================================================================
# 4. the count is not formattable without the evidence
# ===========================================================================


@pytest.fixture(scope="module")
def continuum_quant():
    """A real ``quantify_subpopulations`` run that CHOOSES K > 1 on a weak cloud.

    Three centres a couple of sigma apart: enough offset that the held-out
    likelihood buys components (K=1 must not win, or the test below is vacuous),
    nowhere near enough that the classes are apart. That combination is the
    apoferritin situation in miniature, and it is the one the pipeline could not
    previously see.
    """
    rng = np.random.default_rng(19)
    n = 450
    lab = np.repeat(np.arange(3), 150)
    centres = rng.normal(size=(3, len(_FEATS))) * 1.6
    X = centres[lab] + rng.normal(size=(n, len(_FEATS)))
    early = rng.random(n) < 0.5
    early[lab == 0] = rng.random((lab == 0).sum()) < 0.05   # a drifting class
    q = SG.quantify_subpopulations(
        _frame(X, early), duration_s=20.0, k_grid=(1, 2, 3, 4), n_splits=3,
        n_boot=0, cv_n_init=1, fit_n_init=3, bic_n_init=1)
    assert q.fit.k > 1, "fixture guard: the tests below are vacuous at K=1"
    return q


def test_the_quantification_carries_its_own_separation(continuum_quant):
    q = continuum_quant
    assert q.separation is not None
    assert q.separation.n_events == q.n_events


def test_claim_welds_the_class_count_to_the_evidence(continuum_quant):
    """The one string a caller may print. It cannot be made to drop the qualifier."""
    q = continuum_quant
    claim = q.claim()
    assert f"{q.fit.k} subpopulation(s)" in claim
    assert f"n_PCs={q.fit.n_pcs}" in claim
    assert f"{q.separation.silhouette:.3f}" in claim
    assert q.separation.verdict in claim
    assert "nearer another class" in claim


def test_a_k1_claim_says_one_population_rather_than_an_unqualified_count():
    """The pipeline must stay able to say no; -600 mV chose K=1 and must still."""
    rng = np.random.default_rng(23)
    q = SG.quantify_subpopulations(
        _frame(rng.normal(size=(150, len(_FEATS))) * 0.2, rng.random(150) < 0.5),
        duration_s=10.0, k_grid=(1, 2), n_splits=3, n_boot=0,
        cv_n_init=1, fit_n_init=2, bic_n_init=1)
    assert q.fit.k == 1 and q.separation is None
    assert "one population" in q.claim()
    assert "confidently assigned" not in q.claim()


def test_separation_caveats_come_before_the_bootstrap_caveats():
    """Whether the classes EXIST is a prior question to how precisely they are sized.

    Run WITH a bootstrap on purpose: the fixture above uses ``n_boot=0``, and an
    ordering test against an empty list is a test that asserts nothing.
    """
    rng = np.random.default_rng(5)
    lab = np.repeat(np.arange(3), 150)
    centres = rng.normal(size=(3, len(_FEATS))) * 1.2
    X = centres[lab] + rng.normal(size=(450, len(_FEATS)))
    q = SG.quantify_subpopulations(
        _frame(X, rng.random(450) < 0.5), duration_s=20.0,
        k_grid=(1, 2, 3, 4, 5, 6), n_splits=3, n_boot=60, cv_n_init=1,
        fit_n_init=3, bic_n_init=1, boot_n_init=1)
    ws = list(q.warnings)
    sep_at = [i for i, w in enumerate(ws)
              if w.startswith(("NOT SEPARATED", "WEAKLY SEPARATED"))]
    boot_at = [i for i, w in enumerate(ws)
               if w.startswith(("UNINFORMATIVE INTERVAL", "LABEL SWITCHING"))]
    assert sep_at, f"guard: no separation caveat to order — {ws}"
    assert boot_at, f"guard: no bootstrap caveat to order against — {ws}"
    assert min(sep_at) < min(boot_at)


# ===========================================================================
# 5. the report — the document and the sidecar
# ===========================================================================

N_SAMPLES = 400_000
N_EVENTS = 220


def _synthetic_recording():
    """Real trace, real events, real pipeline — the report is never a stub here."""
    rng = np.random.default_rng(0)
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, 2.0, N_SAMPLES)
    spans, pos = [], 500
    for i in range(N_EVENTS):
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        if pos + dwell + 400 >= N_SAMPLES:
            break
        x[pos:pos + dwell] -= 60.0 if deep else 30.0
        spans.append((pos, pos + dwell))
        pos += int(rng.integers(500, 1100))
    rows = []
    for i, (s, e) in enumerate(spans):
        depth = 60.0 if i % 2 else 30.0
        rows.append({
            "event_id": i, "recording": "r", "pore_id": "p1",
            "start_idx": s, "end_idx": e, "t_start_s": s / FS, "dwell_s": (e - s) / FS,
            "baseline_pA": 200.0, "depth_pA": depth, "frac_depth": depth / 200.0,
            "ecd_pA_s": depth * (e - s) / FS, "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / 2.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
    return (np.ascontiguousarray(x, dtype=np.float32),
            EventSet(pd.DataFrame(rows)[REQUIRED],
                     {"analyzed_duration_s": N_SAMPLES / FS, "polarity": -1}))


@pytest.fixture(scope="module")
def real_report():
    arr, events = _synthetic_recording()
    cfg = SR.SubpopConfig(k_grid=(1, 2, 3), n_splits=2, n_boot=0)
    rep = SR.run_subpopulations(arr, arr, FS, events, recording="r",
                                duration_s=N_SAMPLES / FS, config=cfg)
    assert rep.quantification is not None and rep.clustering is not None
    return rep


def test_the_document_leads_with_what_was_found_and_how_well_it_holds_up(real_report):
    text = real_report.text()
    assert "WHAT THIS RUN FOUND" in text
    head = text.index("WHAT THIS RUN FOUND")
    # ...at the TOP, before the tables, not as caveat 2 of 6 at the bottom
    assert head < text.index("ITEMS 1-2  GATE AND FEATURES")
    assert head < text.index("ITEM 11  POPULATION TABLE")
    # every clause of the headline is in the block, wrapped but not paraphrased
    block = text[head:text.index("ITEMS 1-2  GATE AND FEATURES")]
    flat = " ".join(block.split())
    for bit in real_report.headline().split("·"):
        assert bit.strip() in flat, bit


def test_the_document_prints_a_separation_for_the_model_it_reports(real_report):
    """It printed one for the exploratory k-means and none for the mixture."""
    text = real_report.text()
    assert "separation       :" in text
    q = real_report.quantification
    assert f"{q.separation.silhouette:.4f}" in text
    assert "space the mixture was FITTED in" in text
    assert f"k-means K={q.separation.reference_k} on the same coordinates" in text
    assert "per-class        :" in text
    # and it now says what a posterior is NOT evidence of
    assert "it is not evidence the components are apart" in text


def test_the_document_prints_the_margin_the_k_was_chosen_by(real_report):
    """K=7 beat K=6 by 0.11 nats/event; the profile line stated that only to a
    reader who did the subtraction."""
    text = real_report.text()
    assert "margin over K=" in text
    assert "nats/event" in text


def test_the_headline_states_count_evidence_and_caveat_count(real_report):
    head = real_report.headline()
    q = real_report.quantification
    assert f"{real_report.n_kept} of {real_report.n_fired}" in head
    assert q.claim() in head
    assert "caveat" in head


def test_the_continuum_verdict_is_promoted_to_where_the_count_is_stated(real_report):
    """HDBSCAN finding 0 clusters was caveat 2 of 6 and ended "not a failure"."""
    res = real_report.clustering.scopes["ALL"]
    verdict = real_report.continuum_verdict()
    if res.n_hdbscan_clusters == 0 and real_report.quantification.fit.k > 1:
        assert "ONE CONTINUUM" in verdict
        assert verdict in real_report.headline()
        joined = "\n".join(real_report.warnings)
        assert "DENSITY SAYS CONTINUUM, MIXTURE SAYS" in joined
    else:
        assert verdict == ""


def test_the_exported_sidecar_carries_the_claim_and_the_statistic(real_report):
    """The sidecar outlives the session; a script reading it must not be able to
    pick the class count out without the evidence beside it."""
    with tempfile.TemporaryDirectory() as d:
        written = SR.write_report(real_report, pathlib.Path(d) / "rep")
        meta = json.loads(pathlib.Path(written["meta"]).read_text())
    assert meta["headline"] == real_report.headline()
    sep = real_report.quantification.separation
    if sep is not None:
        assert meta["separation"]["silhouette"] == pytest.approx(sep.silhouette)
        assert meta["separation"]["verdict"] == sep.verdict
        assert set(meta["separation"]["per_subpop"]) == {
            str(c) for c in sep.per_subpop}


# ===========================================================================
# 6. AND NOTHING THE MIXTURE PRODUCED MOVED
# ===========================================================================


def test_measuring_separation_moves_no_number_the_mixture_produced():
    """The whole value of this build is that the scientific numbers reproduce.

    ``measure_separation`` reads a finished fit; it must not perturb the fit, and
    the ``quantify_subpopulations`` it now runs inside must produce exactly the
    fit ``fit_subpopulations`` produces on its own with the same seed.
    """
    rng = np.random.default_rng(31)
    df = _frame(rng.normal(size=(300, len(_FEATS))), rng.random(300) < 0.5)
    q = SG.quantify_subpopulations(
        df, duration_s=10.0, k_grid=(1, 2, 3), n_splits=3, n_boot=0,
        cv_n_init=1, fit_n_init=3, bic_n_init=1)

    Xs = SG.scale_features(df, tuple(SG.SUBPOP_FEATURES))
    ref = SG.fit_subpopulations(
        Xs, n_pcs=q.fit.n_pcs, k=q.fit.k, index=df.index, random_state=0,
        n_init=3, bic_check=True, bic_k_grid=(1, 2, 3), bic_n_init=1)

    assert ref.k == q.fit.k and ref.n_pcs == q.fit.n_pcs
    np.testing.assert_array_equal(ref.labels.to_numpy(), q.fit.labels.to_numpy())
    np.testing.assert_array_equal(ref.scores, q.fit.scores)
    np.testing.assert_array_equal(ref.means, q.fit.means)
    np.testing.assert_array_equal(ref.posteriors, q.fit.posteriors)
    assert ref.conf_frac == q.fit.conf_frac
    assert ref.sizes == q.fit.sizes
    assert ref.bic == q.fit.bic and ref.bic_k == q.fit.bic_k

    # and the measurement itself does not touch what it measures
    before = (q.fit.scores.copy(), q.fit.means.copy(), q.fit.labels.copy())
    SG.measure_separation(q.fit, random_state=0)
    np.testing.assert_array_equal(before[0], q.fit.scores)
    np.testing.assert_array_equal(before[1], q.fit.means)
    np.testing.assert_array_equal(before[2].to_numpy(), q.fit.labels.to_numpy())


def test_two_runs_of_the_same_quantification_agree_on_the_separation():
    rng = np.random.default_rng(37)
    df = _frame(rng.normal(size=(260, len(_FEATS))), rng.random(260) < 0.5)
    kw = dict(duration_s=10.0, k_grid=(1, 2, 3), n_splits=3, n_boot=0,
              cv_n_init=1, fit_n_init=2, bic_n_init=1)
    a = SG.quantify_subpopulations(df, **kw)
    b = SG.quantify_subpopulations(df, **kw)
    assert a.claim() == b.claim()
    if a.separation is not None:
        assert a.separation.silhouette == b.separation.silhouette
        assert a.separation.reference_silhouette == b.separation.reference_silhouette
    assert a.warnings == b.warnings
