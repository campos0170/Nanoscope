"""H4b — the three subpopulation messages that were wrong on real inputs.

H4 shipped the silhouette, which did not exist in this codebase, and it gates the
subpopulation claim correctly. Three of the sentences it prints did not survive
contact with real inputs, and every one of them is a sentence a READER carries
away rather than a number the pipeline computes:

1. **The wrong half.** ``early_frac`` is the FIRST-half fraction. On a
   back-loaded recording (pooled 0.10) the caveat printed

       "THE RECORDING IS NOT STATIONARY: 10 % of all gated events fall in the
        SECOND half of it."

   which is the first-half number in front of the second half's name. Ten per
   cent of the events fall in the FIRST half; ninety fall in the second. The
   front-loaded branch was right, so exactly one branch was wrong — and the test
   covering it asserted ``"0.10" in notes``, which the wrong string satisfies.

2. **It contradicted itself on the next line.** The same caveat ends "every
   early_frac must be read against this 0.90, not against 0.50" — and the class
   gate immediately underneath it tested every class against 0.25-0.75 anyway.
   On a 90 %-front-loaded recording whose classes are each ~0.90 (i.e. perfectly
   stable *relative to the recording*), all of them were condemned. The
   function's own docstring documents the pooled comparison; the gate did not
   make it.

3. **A caveat that fires on everything.** ``NOT THE BEST-SEPARATED PARTITION ON
   OFFER`` compared this mixture's silhouette against ONE k-means at ANOTHER K.
   Two measured facts break that comparison. The mean silhouette falls with K on
   a continuum, so a K=3 k-means out-scoring a K=7 mixture is arithmetic; and
   k-means minimises exactly the within-class scatter a silhouette penalises, so
   it out-scores a full-covariance mixture at the SAME K too — apo -700 mV K=7:
   0.166 vs 0.052; the Zn2+ positive control K=4: 0.385 vs 0.110. Under either
   rule the caveat fires on every real dataset in this project, including the
   positive control whose K=4 is the pipeline's validated output.

THE INVARIANT, and it is the one H4 was built on, one turn further:

    A report's headline may not be more confident than its own caveats — and a
    caveat that cannot distinguish a good result from a bad one is noise wearing
    a warning's clothes.

So the ceiling replaces the rival: an EXISTENCE claim about the cloud ("no
partition of these coordinates is separated at any searched K"), which needs no
comparability across K and is unaffected by k-means' advantage because k-means
is the one measuring the bound. It fires on the apo continuum and is silent on
the Zn2+ control — see ``test_the_ceiling_gate_discriminates_*`` below, which is
the test the fix has to pass to be measuring anything at all.

The user-visible half of this file's subject is in
``tests/gui/test_fup_subpop_messages_wired.py``, through a real ``MainWindow``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import nanoscope
from nanoscope.analysis import subpop_gmm as SG
from nanoscope.analysis import subpop_report as SR

_ZN = pathlib.Path(
    "/Users/abrahamcampos/Desktop/Zn2IonCondition/CSVData/event_clustering_features.csv"
)
requires_zn = pytest.mark.skipif(
    not _ZN.exists(), reason="Zn2IonCondition research data not present"
)


# ===========================================================================
# 0. WHERE DID `nanoscope` COME FROM?
# ===========================================================================
#
# This file is also run against an out-of-tree REVERT of the H4b hunks, where
# every behavioural test below must fail. A revert stub with a wrong signature
# once manufactured four fake failures in this project and was believed, so this
# test exists to be the one that PASSES in both trees: it proves the failures
# came from a real, whole, correctly-resolved package.


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    for mod in (SG, SR):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents, f"{mod.__name__} at {p} is not under {root}"
        assert p.stat().st_size > 20_000, f"{p} is {p.stat().st_size} B — a stub?"
    print(f"\n[H4b] nanoscope resolved from {root}")


# ===========================================================================
# 1. DEFECT 1 — the half the sentence names must be the half the number is
# ===========================================================================


def _table(early_fracs, ns):
    return pd.DataFrame({
        "subpop": list(range(len(ns))),
        "n": list(ns),
        "early_frac": list(early_fracs),
    })


def _stationarity_note(notes) -> str:
    for s in notes:
        if s.startswith("THE RECORDING IS NOT STATIONARY"):
            return s
    return ""


def test_a_back_loaded_recording_names_the_second_half_with_the_second_half_share():
    """THE DEFECT. `early_frac` 0.10 means 90 % of the events are LATE, not 10 %."""
    note = _stationarity_note(SG._stability_warnings(_table([0.10, 0.12, 0.09],
                                                            [300, 300, 300])))
    assert note, "the recording-level caveat must still fire"
    assert "90 % of all gated events fall in the second half" in note
    # the exact sentence the -700-shaped input used to print, unconstructable now
    assert "10 % of all gated events fall in the second half" not in note


def test_the_front_loaded_branch_is_unchanged():
    """Only one branch was wrong; the other must not be 'fixed' into being wrong."""
    note = _stationarity_note(SG._stability_warnings(_table([0.88, 0.90, 0.92],
                                                            [300, 300, 300])))
    assert "90 % of all gated events fall in the first half" in note


@pytest.mark.parametrize("pooled", [0.02, 0.10, 0.24, 0.76, 0.90, 0.98])
def test_the_percentage_and_the_half_always_agree(pooled):
    """The invariant behind defect 1, at both ends and on both sides.

    Whatever half the sentence names, the percentage in front of it is THAT
    half's share — never the other one's. Read off the printed string, so a
    future rewrite of the sentence is checked rather than trusted.
    """
    note = _stationarity_note(SG._stability_warnings(_table([pooled] * 3, [300] * 3)))
    assert note, f"pooled={pooled} is outside the band and must fire"
    early = round(100 * pooled)
    late = 100 - early
    if "first half" in note:
        assert f"{early} % of all gated events fall in the first half" in note
    else:
        assert f"{late} % of all gated events fall in the second half" in note
    # ...and `early_frac` is named as a first-half fraction where the pooled
    # value is quoted, which is the thing that made the half readable backwards.
    assert "FIRST-half fraction" in note
    assert f"read against this {pooled:.2f}" in note


# ===========================================================================
# 2. DEFECT 2 — the class gate makes the comparison the sentence promises
# ===========================================================================


def test_the_band_reduces_to_the_captions_band_on_a_stationary_recording():
    """`STABILITY_BAND` is the pooled-0.50 special case, not a second rule."""
    assert SG.stability_band_for(0.5) == SG.STABILITY_BAND
    assert SG.STABILITY_ODDS_RATIO == pytest.approx(3.0)


@pytest.mark.parametrize("pooled", [0.05, 0.2, 0.5, 0.8, 0.95])
def test_the_band_is_the_same_odds_ratio_wherever_it_sits(pooled):
    lo, hi = SG.stability_band_for(pooled)
    p_odds = pooled / (1 - pooled)
    assert (hi / (1 - hi)) / p_odds == pytest.approx(SG.STABILITY_ODDS_RATIO)
    assert p_odds / (lo / (1 - lo)) == pytest.approx(SG.STABILITY_ODDS_RATIO)
    assert 0.0 <= lo < pooled < hi <= 1.0


def test_a_lopsided_recording_does_not_condemn_classes_shaped_like_itself():
    """THE DEFECT. 90 % front-loaded, every class exactly proportional to it.

    Each class is 0.88-0.92 against a pooled 0.90 — i.e. every one of them is
    distributed exactly like the recording's own event supply, which is the
    definition of "nothing to report about this class". The old fixed band
    condemned all four, 100.0 % of the events, one line under the sentence
    saying to read them against 0.90 rather than 0.50.
    """
    notes = SG._stability_warnings(_table([0.88, 0.90, 0.92, 0.90],
                                          [250, 250, 250, 250]))
    text = "\n".join(notes)
    assert "THE RECORDING IS NOT STATIONARY" in text
    assert "TIME-UNSTABLE CLASS(ES)" not in text
    assert "100.0 % of the events" not in text


def test_the_lopsided_recording_can_still_say_yes():
    """The gate must not have been switched off — only re-centred.

    On the SAME 90 %-front-loaded recording, a class that is NOT shaped like it
    (half its events late, when the recording delivers one in ten late) is a
    real finding and must be named.
    """
    notes = SG._stability_warnings(_table([0.92, 0.92, 0.50], [450, 450, 100]))
    text = "\n".join(notes)
    assert "TIME-UNSTABLE CLASS(ES)" in text
    assert "subpop 2 " in text
    assert "subpop 0 " not in text and "subpop 1 " not in text


def test_the_caveat_and_its_own_gate_cannot_disagree():
    """The self-consistency invariant, stated once and checked over a sweep.

    If the recording-level caveat tells the reader to compare each class against
    the pooled fraction, then no class INSIDE the band around that pooled
    fraction may be named by the class-level caveat. This is the property the
    pair of sentences violated; it is asserted here from the printed pooled
    value, not from the input.
    """
    rng = np.random.default_rng(0)
    for pooled in (0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95):
        ef = np.clip(pooled + rng.normal(0.0, 0.02, 6), 0.0, 1.0)
        ns = [200] * 6
        notes = SG._stability_warnings(_table(ef.tolist(), ns))
        text = "\n".join(notes)
        note = _stationarity_note(notes)
        if note:
            printed = float(note.split("read against this ")[1].split(",")[0])
            lo, hi = SG.stability_band_for(printed)
            assert all(lo <= v <= hi for v in ef), (printed, ef)
        assert "TIME-UNSTABLE CLASS(ES)" not in text, (pooled, text)


def test_a_stationary_recording_keeps_the_captions_own_edges():
    """No regression: at pooled ~0.50 the printed band is still 0.25-0.75."""
    notes = SG._stability_warnings(_table([0.24, 0.50, 0.50, 0.76], [100, 300, 300, 100]))
    text = "\n".join(notes)
    assert "THE RECORDING IS NOT STATIONARY" not in text
    assert "TIME-UNSTABLE CLASS(ES)" in text
    assert "subpop 0 " in text and "subpop 3 " in text
    assert "subpop 1 " not in text and "subpop 2 " not in text
    assert "outside 0.25-0.75" in text


@pytest.mark.parametrize("pooled", [0.0, 1.0, float("nan")])
def test_the_band_survives_a_degenerate_pooled_fraction(pooled):
    """Every event in one half — or none measurable — must not raise or flag."""
    lo, hi = SG.stability_band_for(pooled)
    assert 0.0 <= lo <= hi <= 1.0
    if np.isfinite(pooled):
        notes = "\n".join(SG._stability_warnings(_table([pooled] * 3, [100] * 3)))
        assert "TIME-UNSTABLE CLASS(ES)" not in notes


# ===========================================================================
# 3. DEFECT 3 — the rival becomes a ceiling, and the ceiling discriminates
# ===========================================================================


def _sep(sil: float, profile: dict[int, float] | None, *, k: int = 4) -> SG.Separation:
    ref = None if not profile else profile.get(3)
    return SG.Separation(
        silhouette=sil, neg_fraction=0.2, nearer_other_fraction=0.3,
        per_subpop={i: (50, sil, 0.3) for i in range(k)},
        reference_k=3, reference_silhouette=ref,
        n_used=200, n_events=200, n_pcs=11,
        reference_profile=dict(profile or {}),
    )


def test_the_ceiling_is_the_max_of_the_reference_row_not_one_k():
    sep = _sep(0.05, {2: 0.223, 3: 0.202, 4: 0.192, 7: 0.166})
    assert sep.ceiling_silhouette == pytest.approx(0.223)
    assert sep.ceiling_k == 2
    assert sep.ceiling_substantial is False


def test_the_ceiling_caveat_fires_when_no_searched_k_is_separated():
    notes = "\n".join(SG._separation_warnings(
        _sep(0.052, {2: 0.223, 3: 0.202, 7: 0.166}, k=7), 7))
    assert "NOT THE BEST-SEPARATED PARTITION ON OFFER" in notes
    assert "AND NEITHER IS ANY OTHER" in notes
    assert "at any K" in notes
    assert "0.223" in notes and "0.052" in notes
    assert "NOT a recommendation" in notes
    # ...and it must not smuggle back the causal reading it replaced
    assert "signature of a density being subdivided" not in notes


def test_the_ceiling_caveat_is_silent_where_a_separated_partition_exists():
    """The Zn2+ shape: the mixture is weak but the cloud is not structureless.

    Firing here is the failure the whole change exists to remove — it condemned
    a K=4 result that reproduces bit-exactly and is the pipeline's validated
    output, on a comparison that was going to fire whatever the mixture did.
    """
    notes = "\n".join(SG._separation_warnings(
        _sep(0.110, {2: 0.678, 3: 0.667, 4: 0.385, 7: 0.386}, k=4), 4))
    assert "BEST-SEPARATED PARTITION ON OFFER" not in notes
    # the honest caveat about the mixture itself is untouched and still fires
    assert "NOT SEPARATED" in notes


def test_a_good_partition_still_fires_nothing_about_rivals():
    notes = "\n".join(SG._separation_warnings(_sep(0.62, {2: 0.68, 3: 0.64}), 4))
    assert "BEST-SEPARATED PARTITION ON OFFER" not in notes


def test_the_caveat_names_the_ks_it_actually_measured():
    """The existence claim is only ever made about the row in hand.

    One measured K is a LOWER bound on the true ceiling, so a caveat that said
    "any K" off a single measurement would be claiming more than it has. It
    names the searched range instead, and the range is the profile's own keys.
    """
    contiguous = "\n".join(SG._separation_warnings(
        _sep(0.052, {2: 0.223, 3: 0.202, 4: 0.19, 5: 0.19, 6: 0.18, 7: 0.17, 8: 0.17},
             k=7), 7))
    assert "(K=2-8)" in contiguous
    gappy = "\n".join(SG._separation_warnings(_sep(0.052, {2: 0.223, 3: 0.202, 7: 0.17},
                                                   k=7), 7))
    assert "(K=2,3,7)" in gappy
    single = "\n".join(SG._separation_warnings(_sep(0.05, None, k=7), 7))
    assert single == "" or "BEST-SEPARATED" not in single
    one_ref = "\n".join(SG._separation_warnings(_sep(0.05, {3: 0.20}, k=7), 7))
    assert "(K=3)" in one_ref, "a single measured K may not be reported as 'any K'"


def test_the_ceiling_grid_follows_the_callers_own_k_grid():
    """A caller that searched 1..4 must not be answered about 1..8."""
    df = _synthetic_features(n=300)
    q = SG.quantify_subpopulations(df, k_grid=(1, 2, 3, 4), n_splits=2, n_boot=0,
                                   cv_n_init=1, fit_n_init=2, bic_n_init=1,
                                   random_state=0)
    if q.separation is None:
        pytest.skip("K = 1 on this fixture")
    assert set(q.separation.reference_profile) <= {2, 3, 4} | {q.fit.k}


def test_one_measured_k_is_used_as_a_lower_bound_never_as_a_rival():
    """A `Separation` with no reference row (hand-built, or from before the row
    existed) falls back to its single reference, and the message says so."""
    assert _sep(0.05, {3: 0.60}).ceiling_silhouette == pytest.approx(0.60)
    assert _sep(0.05, None).ceiling_silhouette is None
    assert _sep(0.05, None).ceiling_substantial is None
    assert SG._separation_warnings(_sep(0.05, None), 4) != ()   # NOT SEPARATED still
    assert "BEST-SEPARATED" not in "\n".join(SG._separation_warnings(_sep(0.05, None), 4))


# ---------------------------------------------------------------------------
# ...and the same gate, on the two real clouds it has to tell apart.
# ---------------------------------------------------------------------------


def _fit_scores(df: pd.DataFrame):
    q = SG.quantify_subpopulations(df, random_state=0, n_boot=0)
    return q


def _one_cloud_cut_into_wedges(n=900, k=7, seed=4):
    """The apo -700 mV situation, built rather than recorded.

    ONE anisotropic Gaussian, and a mixture told to report ``k`` classes over it.
    That is precisely what the silhouette exists to detect and what every other
    statistic in this module is blind to: the wedges predict held-out events, the
    posteriors are near 1 inside each wedge, and each wedge earns its parameters.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 13)) @ np.diag(np.linspace(3.0, 0.5, 13))
    df = pd.DataFrame(X, columns=list(SG.SUBPOP_FEATURES))
    fit = SG.fit_subpopulations(SG.scale_features(df), n_pcs=11, k=k, random_state=0)
    return fit, SG.measure_separation(fit, random_state=0)


def test_the_ceiling_gate_fires_on_one_cloud_cut_into_wedges():
    fit, sep = _one_cloud_cut_into_wedges()
    assert sep is not None
    notes = "\n".join(SG._separation_warnings(sep, fit.k))
    print(f"\n[H4b] wedges K={fit.k} sil={sep.silhouette:.3f} "
          f"ceiling={sep.ceiling_silhouette:.3f} at K={sep.ceiling_k}")
    assert sep.ceiling_substantial is False
    assert "BEST-SEPARATED PARTITION ON OFFER" in notes
    assert "AND NEITHER IS ANY OTHER" in notes


@requires_zn
def test_the_ceiling_gate_discriminates_the_zn_control_from_an_apo_continuum():
    """THE TEST THE FIX EXISTS TO PASS.

    A caveat that fires on every dataset carries no information. The Zn2+
    positive control (``zn_30h``: n_PCs=11, K=4, the pipeline's own validated
    output, reproducing bit-exactly) is measured here beside a cloud with the apo
    recordings' shape, and the caveat must come out DIFFERENTLY on the two — the
    old comparison fired on both, and on all three apo recordings besides.
    """
    zn = pd.read_csv(_ZN)
    zn = zn[zn["tag"] == "zn_30h"].reset_index(drop=True)
    q_zn = _fit_scores(zn)
    assert q_zn.fit.k >= 2 and q_zn.separation is not None
    zn_notes = "\n".join(SG._separation_warnings(q_zn.separation, q_zn.fit.k))

    fit_c, sep_c = _one_cloud_cut_into_wedges(k=q_zn.fit.k)
    assert sep_c is not None
    cont_notes = "\n".join(SG._separation_warnings(sep_c, fit_c.k))

    print(f"\n[H4b] zn_30h  K={q_zn.fit.k} sil={q_zn.separation.silhouette:.3f} "
          f"ceiling={q_zn.separation.ceiling_silhouette:.3f} "
          f"at K={q_zn.separation.ceiling_k}")
    print(f"[H4b] wedges  K={fit_c.k} sil={sep_c.silhouette:.3f} "
          f"ceiling={sep_c.ceiling_silhouette:.3f} at K={sep_c.ceiling_k}")

    fires_zn = "BEST-SEPARATED PARTITION ON OFFER" in zn_notes
    fires_cont = "BEST-SEPARATED PARTITION ON OFFER" in cont_notes
    assert fires_cont and not fires_zn, (
        f"the gate must separate these two: zn={fires_zn} continuum={fires_cont}"
    )
    # ...at the SAME K, so the difference cannot be an artefact of K.
    assert fit_c.k == q_zn.fit.k


@requires_zn
def test_the_reference_row_is_measured_on_the_same_events_as_the_silhouette():
    """The ceiling is only a bound on the mixture's own number if both were
    measured on the same (possibly subsampled) events, in the same coordinates."""
    zn = pd.read_csv(_ZN)
    zn = zn[zn["tag"] == "zn_30h"].reset_index(drop=True)
    q = _fit_scores(zn)
    sep = q.separation
    assert sep is not None and sep.reference_profile
    assert set(sep.reference_profile) >= {q.fit.k, sep.reference_k}
    assert sep.n_pcs == q.fit.n_pcs
    assert sep.reference_silhouette == pytest.approx(
        sep.reference_profile[sep.reference_k])


# ===========================================================================
# 4. nothing the mixture computed may move
# ===========================================================================


def _synthetic_features(n=320, seed=11):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, size=(n // 2, 13))
    b = rng.normal(2.5, 1.0, size=(n - n // 2, 13))
    return pd.DataFrame(np.vstack([a, b]), columns=list(SG.SUBPOP_FEATURES))


def test_measuring_separation_cannot_move_a_single_number_the_mixture_produced():
    df = _synthetic_features()
    fit = SG.fit_subpopulations(SG.scale_features(df), n_pcs=4, k=3, random_state=0)
    before = (int(fit.k), fit.labels.to_numpy().copy(), fit.scores.copy(),
              float(fit.conf_frac))
    sep = SG.measure_separation(fit, random_state=0)
    assert sep is not None
    assert int(fit.k) == before[0]
    np.testing.assert_array_equal(fit.labels.to_numpy(), before[1])
    np.testing.assert_array_equal(fit.scores, before[2])
    assert float(fit.conf_frac) == before[3]


def test_the_reference_row_is_deterministic():
    df = _synthetic_features()
    fit = SG.fit_subpopulations(SG.scale_features(df), n_pcs=4, k=3, random_state=0)
    a = SG.measure_separation(fit, random_state=0)
    b = SG.measure_separation(fit, random_state=0)
    assert a is not None and b is not None
    assert a.reference_profile == b.reference_profile
    assert a.ceiling_silhouette == b.ceiling_silhouette
    assert a.silhouette == b.silhouette


def test_the_ceiling_reaches_the_exported_sidecar():
    """The sidecar outlives the session; the claim may not escape without it."""
    df = _synthetic_features()
    fit = SG.fit_subpopulations(SG.scale_features(df), n_pcs=4, k=3, random_state=0)
    sep = SG.measure_separation(fit, random_state=0)
    assert sep is not None and sep.reference_profile
    # the report's own writer is exercised in the GUI file; here, the shape the
    # writer reads must exist and be JSON-able
    import json
    payload = {
        "reference_profile": {str(k): v for k, v in sep.reference_profile.items()},
        "ceiling_k": sep.ceiling_k,
        "ceiling_silhouette": sep.ceiling_silhouette,
        "ceiling_substantial": sep.ceiling_substantial,
    }
    assert json.loads(json.dumps(payload))["ceiling_k"] == sep.ceiling_k
