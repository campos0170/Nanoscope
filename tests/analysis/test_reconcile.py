"""Core contract for overlap-aware detector reconciliation."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from nanoscope.analysis.events import (
    KIND_MIXED,
    REVIEW_COL,
    REVIEW_DONE,
    REVIEW_NONE,
    SCHEMA,
    SOURCE_COL,
    SOURCE_MANUAL,
    SOURCE_MIXED,
    EventSet,
    source_kind,
    stamp_source_status,
)
from nanoscope.analysis.labels import parse_labels
from nanoscope.analysis.reconcile import (
    CONSENSUS_CONTRIBUTORS_COL,
    CONSENSUS_CANONICAL_SOURCE_COL,
    CONSENSUS_CANONICAL_MODULE_COL,
    CONSENSUS_CURATION_CONFLICT_COL,
    CONSENSUS_LABEL_CONFLICTS_COL,
    CONSENSUS_MEMBER_COUNT_COL,
    CONSENSUS_MODULE_ID,
    CONSENSUS_SUPPORT_COL,
    MAX_MATCH_EDGES,
    MAX_PAIR_COMPARISONS,
    ReconcileConfig,
    reconciliation_source_fingerprint,
    reconcile_detections,
)
from nanoscope.analysis.scoring import (
    ADDED_COLUMN,
    TRAINING_SELECTION_REVIEWED,
    build_training_frame,
    ensure_added_column,
    score_detection,
)
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.sources import DETECTION_MODULE_IDS, detector_name


FS = 50_000.0


def _frame(module_id: str, rows: list[dict]) -> pd.DataFrame:
    built = []
    for event_id, spec in enumerate(rows):
        start, stop = int(spec["span"][0]), int(spec["span"][1])
        built.append({
            "event_id": event_id,
            "recording": "apo.abf",
            "pore_id": "pore-a",
            "start_idx": start,
            "end_idx": stop,
            "t_start_s": start / FS,
            "dwell_s": (stop - start) / FS,
            "baseline_pA": -100.0,
            "depth_pA": float(spec.get("depth", 20.0)),
            "frac_depth": float(spec.get("depth", 20.0)) / 100.0,
            "ecd_pA_s": 0.1,
            "max_dev_pA": 21.0,
            "mean_dev_pA": 19.0,
            "rise_s": 0.0001,
            "fall_s": 0.0001,
            "snr": 5.0,
            "n_levels": 1,
            "filter_limited": False,
            "accepted": bool(spec.get("accepted", True)),
            "flag": str(spec.get("flag", "")),
        })
    if built:
        df = pd.DataFrame(built)
        for column, (dtype, _unit, _desc) in SCHEMA.items():
            df[column] = df[column].astype(dtype)
        df = df[list(SCHEMA)]
    else:
        from nanoscope.analysis.events import empty_df
        df = empty_df()
    df = stamp_source_status(df, module_id)
    ensure_added_column(df)
    for pos, spec in enumerate(rows):
        if spec.get("added", False):
            df.loc[pos, ADDED_COLUMN] = True
            df.loc[pos, SOURCE_COL] = SOURCE_MANUAL
        if "review" in spec:
            df.loc[pos, REVIEW_COL] = spec["review"]
    df[SOURCE_COL] = pd.array(df[SOURCE_COL], dtype="string")
    df[REVIEW_COL] = pd.array(df[REVIEW_COL], dtype="string")
    return df


def _result(
    module_id: str,
    rows: list[dict],
    *,
    doc_id: str = "doc-1",
    channel: int = 0,
    calibration=(1.0, 0.0),
    baseline: str | None = "baseline-a",
    polarity: int | None = -1,
) -> AnalysisResult:
    provenance = {
        "fs": FS,
        "units": "pA",
        "source": "apo.abf",
        "baseline_fingerprint": baseline,
        "polarity": polarity,
        "analysed_start_idx": 0,
        "analysed_stop_idx": 100_000,
        "analyzed_duration_s": 2.0,
    }
    return AnalysisResult(
        module_id=module_id,
        module_version="1",
        doc_id=doc_id,
        channel=channel,
        params={},
        calibration_fingerprint=calibration,
        payload=EventSet(_frame(module_id, rows), provenance),
        baseline_fingerprint=baseline,
    )


def test_overlap_union_is_deterministic_and_does_not_mutate_inputs():
    schmitt = _result("detect.schmitt", [
        {"span": (100, 200), "depth": 20},
        {"span": (500, 600), "depth": 30},
    ])
    keras = _result("detect.keras", [
        {"span": (110, 190), "depth": 22},
        {"span": (800, 900), "depth": 40},
    ])
    before = [r.payload.df.copy(deep=True) for r in (schmitt, keras)]

    forward = reconcile_detections([schmitt, keras])
    reverse = reconcile_detections([keras, schmitt])

    pd.testing.assert_frame_equal(forward.result.payload.df, reverse.result.payload.df)
    assert forward.input_modules == ("detect.keras", "detect.schmitt")
    assert (forward.n_input_rows, forward.n_clusters, forward.n_output_rows,
            forward.n_collapsed) == (4, 3, 3, 1)
    out = forward.result.payload.df
    assert out[CONSENSUS_SUPPORT_COL].tolist() == [2, 1, 1]
    assert out[CONSENSUS_MEMBER_COUNT_COL].tolist() == [2, 1, 1]
    assert out.loc[0, CONSENSUS_CONTRIBUTORS_COL] == "detect.keras|detect.schmitt"
    assert forward.result.module_id == CONSENSUS_MODULE_ID
    assert forward.result.overlays[0].starts.tolist() == out["start_idx"].tolist()
    for original, snapshot in zip((schmitt, keras), before):
        pd.testing.assert_frame_equal(original.payload.df, snapshot)


def test_manual_duplicate_converges_with_detector_and_labels_from_both_survive():
    result = _result("detect.schmitt", [
        {"span": (100, 210), "flag": "molecule=apo;quality=clean"},
        {"span": (120, 190), "added": True, "review": REVIEW_DONE,
         "flag": "molecule=holo;operator=alice"},
    ])
    outcome = reconcile_detections([result])
    out = outcome.result.payload.df

    assert len(out) == 1 and outcome.n_collapsed == 1
    # A hand boundary is the authoritative complete row; measured fields stay coherent.
    assert (int(out.loc[0, "start_idx"]), int(out.loc[0, "end_idx"])) == (120, 190)
    # The reconciled detector did not miss it: an automatic contributor also found it.
    assert bool(out.loc[0, ADDED_COLUMN]) is False
    assert int(out.loc[0, CONSENSUS_SUPPORT_COL]) == 1
    assert out.loc[0, SOURCE_COL] == SOURCE_MIXED
    assert out.loc[0, CONSENSUS_CANONICAL_SOURCE_COL] == SOURCE_MANUAL
    assert out.loc[0, CONSENSUS_CANONICAL_MODULE_COL] == "detect.schmitt"
    labels = parse_labels(out.loc[0, "flag"])
    assert labels == {
        "molecule": "apo|holo",
        "quality": "clean",
        "operator": "alice",
    }
    assert out.loc[0, CONSENSUS_LABEL_CONFLICTS_COL] == "molecule"
    assert outcome.n_label_conflicts == 1
    assert outcome.result.payload.provenance["reconciliation"]["n_label_conflicts"] == 1
    assert any("conflicting user labels" in warning for warning in outcome.result.warnings)
    # The explicit human ruling belongs to the physical event; an unreviewed detector
    # duplicate cannot erase it from reviewed-only supervised-learning exports.
    assert out.loc[0, REVIEW_COL] == REVIEW_DONE
    score = score_detection(out)
    assert score.fn == 0


def test_a_hand_rejection_wins_but_an_automatic_rejection_is_not_a_vote():
    schmitt = _result("detect.schmitt", [
        {"span": (100, 200), "accepted": False, "flag": "artifact=capacitive",
         "review": REVIEW_DONE},
        # Deliberately shorter than the viable match: without the validity-first
        # canonical rank this row wins the medoid/duration tie and donates a clipped
        # measurement to an output later marked accepted.
        {"span": (520, 580), "accepted": False, "flag": "clipped_by_boundary"},
    ])
    keras = _result("detect.keras", [
        {"span": (105, 195), "accepted": True},
        {"span": (500, 600), "accepted": True},
    ])
    out = reconcile_detections([schmitt, keras]).result.payload.df

    assert bool(out.loc[0, "accepted"]) is False       # person's rejection is conservative
    assert parse_labels(out.loc[0, "flag"])["artifact"] == "capacitive"
    assert bool(out.loc[1, "accepted"]) is True        # detector self-reject did not veto
    # A derived flag from the non-canonical rejected row must not contaminate a good span.
    assert "clipped_by_boundary" not in out.loc[1, "flag"]
    assert (int(out.loc[1, "start_idx"]), int(out.loc[1, "end_idx"])) == (500, 600)
    assert out.loc[1, CONSENSUS_CANONICAL_SOURCE_COL] == "detect.keras"
    assert out.loc[1, CONSENSUS_CANONICAL_MODULE_COL] == "detect.keras"


def test_contradictory_human_truth_stays_unresolved_until_explicitly_curated():
    source = _result("detect.schmitt", [
        {"span": (100, 220), "accepted": False, "review": REVIEW_DONE},
        {"span": (120, 200), "accepted": True, "added": True,
         "review": REVIEW_DONE},
    ])

    outcome = reconcile_detections([source])
    out = outcome.result.payload.df

    assert len(out) == 1
    assert (int(out.loc[0, "start_idx"]), int(out.loc[0, "end_idx"])) == (120, 200)
    assert bool(out.loc[0, "accepted"]) is False
    assert out.loc[0, REVIEW_COL] == REVIEW_DONE
    assert out.loc[0, CONSENSUS_CURATION_CONFLICT_COL] == "accepted|rejected"
    assert outcome.n_curation_conflicts == 1
    # A contradictory ruling is reviewed activity but not a resolved target label.
    assert build_training_frame(
        out, selection=TRAINING_SELECTION_REVIEWED).empty

    # GUI accept/reject actions clear this marker. Model that explicit resolution here
    # to pin the training-selection contract independently of Qt.
    out.loc[0, "accepted"] = True
    out.loc[0, CONSENSUS_CURATION_CONFLICT_COL] = ""
    trained = build_training_frame(out, selection=TRAINING_SELECTION_REVIEWED)
    assert len(trained) == 1 and bool(trained.loc[0, "accepted"]) is True


def test_support_filter_is_an_agreement_rule_but_can_keep_manual_truth():
    schmitt = _result("detect.schmitt", [
        {"span": (100, 200)},
        {"span": (500, 600), "added": True, "review": REVIEW_DONE},
    ])
    keras = _result("detect.keras", [{"span": (110, 190)}])
    cusum = _result("detect.cusum", [{"span": (115, 205)}])
    cfg = ReconcileConfig(min_detector_support=3, keep_manual=True)
    outcome = reconcile_detections([schmitt, keras, cusum], config=cfg)

    assert outcome.n_clusters == 2
    assert outcome.n_output_rows == 2
    assert outcome.n_manual_kept == 1
    out = outcome.result.payload.df
    assert out[CONSENSUS_SUPPORT_COL].tolist() == [3, 0]
    assert bool(out.loc[1, ADDED_COLUMN]) is True

    dropped = reconcile_detections(
        [schmitt, keras, cusum],
        config=ReconcileConfig(min_detector_support=3, keep_manual=False),
    )
    assert dropped.n_output_rows == 1 and dropped.n_filtered == 1


def test_a_broad_candidate_cannot_swallow_two_events_from_one_detector():
    broad = _result("detect.keras", [{"span": (100, 400)}])
    narrow = _result("detect.schmitt", [
        {"span": (110, 180)},
        {"span": (300, 370)},
    ])
    outcome = reconcile_detections([broad, narrow])

    assert outcome.n_clusters == 2
    assert outcome.n_collapsed == 1
    assert sorted(outcome.result.payload.df[CONSENSUS_MEMBER_COUNT_COL]) == [1, 2]


def test_a_broad_span_cannot_transitively_bridge_nonoverlapping_detectors():
    left = _result("detect.schmitt", [{"span": (0, 10)}])
    bridge = _result("detect.keras", [{"span": (0, 20)}])
    right = _result("detect.cusum", [{"span": (10, 20)}])

    forward = reconcile_detections([left, bridge, right])
    reverse = reconcile_detections([right, bridge, left])

    assert forward.n_clusters == 2
    assert sorted(forward.result.payload.df[CONSENSUS_SUPPORT_COL]) == [1, 2]
    assert max(forward.result.payload.df[CONSENSUS_MEMBER_COUNT_COL]) == 2
    pd.testing.assert_frame_equal(forward.result.payload.df, reverse.result.payload.df)
    rec = forward.result.payload.provenance["reconciliation"]
    assert rec["cluster_coherence"].startswith("complete linkage")


def test_pathological_overlap_is_bounded_with_actionable_error(monkeypatch):
    # Lower the production ceilings so this proves both guards without building the
    # million-edge pathology they protect the interactive Preview from.
    import nanoscope.analysis.reconcile as module

    crowded_a = _result("detect.schmitt", [
        {"span": (100 + i, 10_000 + i)} for i in range(8)
    ])
    crowded_b = _result("detect.keras", [
        {"span": (200 + i, 10_100 + i)} for i in range(8)
    ])

    monkeypatch.setattr(module, "MAX_PAIR_COMPARISONS", 10)
    with pytest.raises(ValueError, match="simultaneous-span comparisons.*shorter segment"):
        reconcile_detections([crowded_a, crowded_b])

    monkeypatch.setattr(module, "MAX_PAIR_COMPARISONS", MAX_PAIR_COMPARISONS)
    monkeypatch.setattr(module, "MAX_MATCH_EDGES", 5)
    with pytest.raises(ValueError, match="eligible overlap links.*minimum temporal overlap"):
        reconcile_detections([crowded_a, crowded_b])
    assert MAX_MATCH_EDGES >= 3 * 30_000, "production edge budget lost the 30k-row design margin"


def test_membership_and_input_provenance_make_every_output_auditable():
    a = replace(
        _result("detect.schmitt", [{"span": (100, 200)}]),
        module_version="2.4",
        params={"threshold_sigma": 5.5},
        warnings=("source advisory",),
    )
    b = _result("detect.keras", [{"span": (110, 190)}])
    outcome = reconcile_detections([a, b])
    prov = outcome.result.payload.provenance
    rec = prov["reconciliation"]

    assert rec["algorithm"] == "constrained-overlap-coefficient"
    assert rec["canonical_rule"].startswith("viable (not automatically rejected), then manual")
    assert rec["input_modules"] == ["detect.keras", "detect.schmitt"]
    assert set(rec["input_provenance"]) == {"detect.keras", "detect.schmitt"}
    assert rec["input_results"]["detect.schmitt"] == {
        "module_version": "2.4",
        "params": {"threshold_sigma": 5.5},
        "calibration_fingerprint": [1.0, 0.0],
        "baseline_fingerprint": "baseline-a",
        "warnings": ["source advisory"],
    }
    assert rec["input_result_fingerprints"]["detect.schmitt"] == (
        reconciliation_source_fingerprint(a)
    )
    members = rec["memberships"][0]["members"]
    assert {(m["module_id"], m["event_id"]) for m in members} == {
        ("detect.keras", 0), ("detect.schmitt", 0)
    }
    assert {m["row_position"] for m in members} == {0}
    assert sum(bool(m["canonical"]) for m in members) == 1
    assert prov["analyzed_duration_s"] == pytest.approx(2.0)
    assert outcome.result.payload.rate_hz() == pytest.approx(0.5)
    contract = rec["measurement_contract"]
    assert contract["polarity_fully_verified"] is True
    assert contract["common_known_polarity"] == -1
    assert prov["polarity"] == -1


def test_source_fingerprint_detects_curation_geometry_and_replacement_changes():
    source = _result("detect.schmitt", [{"span": (100, 200)}])
    original = reconciliation_source_fingerprint(source)

    curated_df = source.payload.df.copy()
    curated_df.loc[0, "accepted"] = False
    curated = replace(source, payload=EventSet(curated_df, source.payload.provenance))
    assert reconciliation_source_fingerprint(curated) != original

    moved_df = source.payload.df.copy()
    moved_df.loc[0, "end_idx"] = 201
    moved = replace(source, payload=EventSet(moved_df, source.payload.provenance))
    assert reconciliation_source_fingerprint(moved) != original

    replaced = replace(source, module_version="new")
    assert reconciliation_source_fingerprint(replaced) != original


@pytest.mark.parametrize(
    "cfg,match",
    [
        (lambda: ReconcileConfig(min_overlap_fraction=0), "min_overlap_fraction"),
        (lambda: ReconcileConfig(min_overlap_fraction=float("nan")), "min_overlap_fraction"),
        (lambda: ReconcileConfig(min_detector_support=True), "min_detector_support"),
        (lambda: ReconcileConfig(min_detector_support=0), "min_detector_support"),
        (lambda: ReconcileConfig(keep_manual="yes"), "keep_manual"),
    ],
)
def test_config_rejects_ambiguous_values(cfg, match):
    with pytest.raises(ValueError, match=match):
        cfg()


def test_mismatched_identity_and_measurement_contracts_are_refused():
    a = _result("detect.schmitt", [{"span": (100, 200)}])
    base = _result("detect.keras", [{"span": (110, 190)}])

    with pytest.raises(ValueError, match="same document and channel"):
        reconcile_detections([a, replace(base, doc_id="other")])
    with pytest.raises(ValueError, match="different calibrations"):
        reconcile_detections([a, replace(base, calibration_fingerprint=(2.0, 0.0))])
    other_baseline_prov = dict(
        base.payload.provenance, baseline_fingerprint="baseline-b"
    )
    other_baseline = replace(
        base,
        baseline_fingerprint="baseline-b",
        payload=EventSet(base.payload.df, other_baseline_prov),
    )
    with pytest.raises(ValueError, match="different baselines"):
        reconcile_detections([a, other_baseline])
    with pytest.raises(ValueError, match="cannot exceed"):
        reconcile_detections([a, base], config=ReconcileConfig(min_detector_support=3))

    other_range_prov = dict(base.payload.provenance, analysed_start_idx=10_000)
    other_range = replace(base, payload=EventSet(base.payload.df, other_range_prov))
    # A union can honestly combine coverage; lack of a candidate outside one detector's
    # view is not interpreted as a vote.
    assert reconcile_detections([a, other_range]).n_output_rows == 1
    with pytest.raises(ValueError, match="identical recorded analysed spans.*support 1"):
        reconcile_detections(
            [a, other_range], config=ReconcileConfig(min_detector_support=2)
        )

    malformed = base.payload.df.copy()
    malformed.loc[0, "end_idx"] = malformed.loc[0, "start_idx"]
    bad = replace(base, payload=EventSet(malformed, base.payload.provenance))
    with pytest.raises(ValueError, match="invalid half-open span"):
        reconcile_detections([a, bad])


def test_provenance_validation_and_promotion_use_the_same_normalised_values():
    a = _result("detect.schmitt", [{"span": (100, 200)}])
    b = _result("detect.keras", [{"span": (110, 190)}])
    a_prov = dict(a.payload.provenance, fs="50000", units=" pA ")
    b_prov = dict(b.payload.provenance, fs=50_000.0, units="pA")
    a = replace(a, payload=EventSet(a.payload.df, a_prov))
    b = replace(b, payload=EventSet(b.payload.df, b_prov))

    prov = reconcile_detections([a, b]).result.payload.provenance
    assert prov["fs"] == 50_000.0
    assert prov["units"] == "pA"

    for invalid_value in ("not-a-rate", float("nan"), True):
        bad_prov = dict(a.payload.provenance, fs=invalid_value)
        bad = replace(a, payload=EventSet(a.payload.df, bad_prov))
        with pytest.raises(ValueError, match="(invalid|non-finite) 'fs'"):
            reconcile_detections([bad])

    negative_prov = dict(a.payload.provenance, fs=-50_000)
    negative = replace(a, payload=EventSet(a.payload.df, negative_prov))
    with pytest.raises(ValueError, match="non-positive 'fs'"):
        reconcile_detections([negative])


def test_baseline_must_be_consistent_inside_each_result_and_known_for_every_input():
    known = _result("detect.schmitt", [{"span": (100, 200)}], baseline="baseline-a")
    unknown = _result("detect.keras", [{"span": (110, 190)}], baseline=None)

    derived = reconcile_detections([known, unknown]).result
    assert derived.baseline_fingerprint is None
    assert "baseline_fingerprint" not in derived.payload.provenance
    assert derived.payload.provenance["reconciliation"]["measurement_contract"][
        "common_baseline_fingerprint"] is None

    contradictory_prov = dict(
        known.payload.provenance, baseline_fingerprint="baseline-b"
    )
    contradictory = replace(
        known, payload=EventSet(known.payload.df, contradictory_prov)
    )
    with pytest.raises(ValueError, match="inconsistent baseline fingerprints"):
        reconcile_detections([contradictory])


def test_unknown_or_out_of_span_coverage_never_produces_a_confident_rate():
    explicit = _result("detect.schmitt", [{"span": (100, 200)}])
    legacy = _result("detect.keras", [{"span": (450_000, 450_100)}])
    legacy_prov = dict(legacy.payload.provenance)
    legacy_prov.pop("analysed_start_idx")
    legacy_prov.pop("analysed_stop_idx")
    legacy_prov["analyzed_duration_s"] = 10.0
    legacy = replace(legacy, payload=EventSet(legacy.payload.df, legacy_prov))

    derived = reconcile_detections([explicit, legacy]).result
    assert "analyzed_duration_s" not in derived.payload.provenance
    assert "rate_hz" not in derived.scalars
    assert any("coverage cannot be computed" in warning for warning in derived.warnings)

    # A missed event can be drawn outside the detector window. That gesture proves an
    # event, not how much extra time the curator searched, so the old detector duration
    # is not a valid denominator for the combined table.
    outside = _result("detect.schmitt", [
        {"span": (100, 200)},
        {"span": (150_000, 150_100), "added": True, "review": REVIEW_DONE},
    ])
    outside_result = reconcile_detections([outside]).result
    assert "analyzed_duration_s" not in outside_result.payload.provenance
    assert "rate_hz" not in outside_result.scalars
    assert any("outside its source detector" in warning
               for warning in outside_result.warnings)


def test_opposite_or_invalid_detector_polarities_are_refused():
    down = _result("detect.schmitt", [{"span": (100, 200)}], polarity=-1)
    up = _result("detect.keras", [{"span": (110, 190)}], polarity=1)
    with pytest.raises(ValueError, match="opposite detector polarities"):
        reconcile_detections([down, up])

    invalid = _result("detect.keras", [{"span": (110, 190)}], polarity=0)
    with pytest.raises(ValueError, match=r"invalid detector polarity.*expected -1, \+1"):
        reconcile_detections([down, invalid])


def test_missing_polarity_is_auditable_but_does_not_block_legacy_input():
    known = _result("detect.schmitt", [{"span": (100, 200)}], polarity=-1)
    legacy = _result("detect.keras", [{"span": (110, 190)}], polarity=None)

    outcome = reconcile_detections([known, legacy])
    prov = outcome.result.payload.provenance
    contract = prov["reconciliation"]["measurement_contract"]
    assert contract["input_polarities"] == {
        "detect.keras": None,
        "detect.schmitt": -1,
    }
    assert contract["polarity_fully_verified"] is False
    assert "polarity" not in prov
    assert any("no recorded detector polarity" in warning for warning in outcome.result.warnings)


def test_mixed_preprocessing_chains_are_explicit_instead_of_claiming_one_contract():
    schmitt = _result("detect.schmitt", [{"span": (100, 200)}])
    keras = _result("detect.keras", [{"span": (110, 190)}])
    changed_prov = dict(keras.payload.provenance, filter_cutoff_hz=5_000.0)
    keras = replace(keras, payload=EventSet(keras.payload.df, changed_prov))

    outcome = reconcile_detections([schmitt, keras])
    prov = outcome.result.payload.provenance
    contract = prov["reconciliation"]["measurement_contract"]
    assert contract["mixed_detection_signal_chains"] is True
    assert contract["common_detection_signal_key"] is None
    assert "filter_cutoff_hz" not in prov
    assert any("different detector preprocessing chains" in w for w in outcome.result.warnings)


def test_reconciled_is_a_discoverable_mixed_detection_source():
    assert CONSENSUS_MODULE_ID in DETECTION_MODULE_IDS
    assert detector_name(CONSENSUS_MODULE_ID) == "Reconciled"
    assert source_kind(CONSENSUS_MODULE_ID) == KIND_MIXED
