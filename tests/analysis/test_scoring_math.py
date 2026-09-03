"""Pure tests for :mod:`nanoscope.analysis.scoring` — the detector-vs-curation score
and the training-export table.

The scoring model (module docstring): with the detector's output being every row it
emitted (``added == False``) and the truth being every row the user kept
(``accepted == True``):

    TP matched  = accepted & ~added
    FP spurious = ~accepted & ~added
    FN missed   = added   (hand-added by the user; accepted by construction)

    precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R), each None if
    its denominator is zero.

These tests pin the arithmetic on hand-built frames (so the numbers are exact and
independent of any detector) and cover every divide-by-zero edge, then check the
training-export table honours accepted-only-by-default and the ``added`` label.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import events as event_model
from nanoscope.analysis.scoring import (
    ADDED_COLUMN,
    TRAINING_COLUMNS,
    TRAINING_SELECTION_ACCEPTED,
    TRAINING_SELECTION_ALL,
    TRAINING_SELECTION_REVIEWED,
    added_mask,
    build_training_frame,
    ensure_added_column,
    score_detection,
    training_provenance,
    write_training_export,
)


# ================================================================================================
# builders
# ================================================================================================

def _frame(accepted: list[bool], added: list[bool] | None = None,
           ids: list[int] | None = None) -> pd.DataFrame:
    """A minimal detection frame carrying just the columns scoring reads."""
    n = len(accepted)
    data = {
        "event_id": ids if ids is not None else list(range(n)),
        "recording": ["rec"] * n,
        "pore_id": ["p1"] * n,
        "start_idx": np.arange(n, dtype=np.int64) * 100,
        "end_idx": np.arange(n, dtype=np.int64) * 100 + 50,
        "t_start_s": np.arange(n, dtype=np.float64) * 1e-3,
        "dwell_s": np.full(n, 5e-4),
        "depth_pA": np.full(n, 30.0),
        "accepted": np.array(accepted, dtype=bool),
    }
    if added is not None:
        data[ADDED_COLUMN] = np.array(added, dtype=bool)
    return pd.DataFrame(data)


# ================================================================================================
# the arithmetic — exact P/R/F1 on a well-formed frame
# ================================================================================================

def test_scores_a_typical_curation_exactly():
    # 5 detected: 3 kept (TP), 2 rejected (FP); plus 1 hand-added (FN).
    df = _frame(
        accepted=[True, True, True, False, False, True],
        added=[False, False, False, False, False, True],
        ids=[0, 1, 2, 3, 4, 5],
    )
    s = score_detection(df)

    assert (s.tp, s.fp, s.fn) == (3, 2, 1)
    assert s.n_detected == 5          # everything with added == False
    assert s.n_truth == 4             # everything accepted (3 kept + 1 added)
    assert s.precision == pytest.approx(3 / 5)
    assert s.recall == pytest.approx(3 / 4)
    assert s.f1 == pytest.approx(2 * (3 / 5) * (3 / 4) / ((3 / 5) + (3 / 4)))

    assert s.matched_ids == [0, 1, 2]
    assert s.spurious_ids == [3, 4]
    assert s.missed_ids == [5]


def test_perfect_detection_scores_one():
    df = _frame(accepted=[True, True, True], added=[False, False, False])
    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (3, 0, 0)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0


# ================================================================================================
# a missing 'added' column is all-False
# ================================================================================================

def test_missing_added_column_is_all_false():
    # No 'added' column at all: nothing hand-added, so FN == 0 and recall is defined.
    df = _frame(accepted=[True, True, False])
    assert ADDED_COLUMN not in df.columns
    assert not added_mask(df).any()

    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (2, 1, 0)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == 1.0            # no misses possible without an 'added' column
    assert s.missed_ids == []


# ================================================================================================
# divide-by-zero edges -> None (n/a), never 0.0 or a raise
# ================================================================================================

def test_no_detections_precision_na():
    # Every row hand-added (the detector emitted nothing): precision undefined.
    df = _frame(accepted=[True, True], added=[True, True])
    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (0, 0, 2)
    assert s.n_detected == 0
    assert s.precision is None        # TP+FP == 0
    assert s.recall == 0.0            # 0 / (0 + 2)
    assert s.f1 is None               # precision is None


def test_all_rejected_precision_zero_recall_na():
    # The detector fired but the user rejected everything, and added nothing:
    # precision = 0/ (0+2) = 0; recall = 0/0 = None.
    df = _frame(accepted=[False, False], added=[False, False])
    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (0, 2, 0)
    assert s.precision == 0.0
    assert s.recall is None           # TP+FN == 0
    assert s.f1 is None


def test_all_missed_recall_zero_precision_na():
    # The user hand-added real events the detector never saw: recall 0, precision n/a.
    df = _frame(accepted=[True, True, True], added=[True, True, True])
    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (0, 0, 3)
    assert s.recall == 0.0
    assert s.precision is None
    assert s.f1 is None


def test_empty_frame_all_none():
    s = score_detection(_frame(accepted=[]))
    assert (s.n_detected, s.n_truth, s.tp, s.fp, s.fn) == (0, 0, 0, 0, 0)
    assert s.precision is None and s.recall is None and s.f1 is None
    assert s.matched_ids == [] and s.missed_ids == [] and s.spurious_ids == []


def test_f1_none_when_precision_and_recall_both_zero():
    # 1 spurious (FP) + 1 missed (FN), no matches: P=0, R=0, so F1 = 0/0 -> None.
    df = _frame(accepted=[False, True], added=[False, True])
    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)
    assert s.precision == 0.0 and s.recall == 0.0
    assert s.f1 is None


# ================================================================================================
# ensure_added_column
# ================================================================================================

def test_ensure_added_column_adds_default_false_in_place():
    df = _frame(accepted=[True, False])
    same = ensure_added_column(df)
    assert same is df                          # mutated in place, same object
    assert ADDED_COLUMN in df.columns
    assert df[ADDED_COLUMN].dtype == bool
    assert not df[ADDED_COLUMN].any()


def test_ensure_added_column_leaves_existing_untouched():
    df = _frame(accepted=[True, True], added=[False, True])
    ensure_added_column(df)
    assert list(df[ADDED_COLUMN]) == [False, True]


# ================================================================================================
# training export table
# ================================================================================================

def test_build_training_frame_defaults_to_accepted_including_added():
    df = _frame(
        accepted=[True, False, True, True],
        added=[False, False, False, True],
        ids=[0, 1, 2, 3],
    )
    out = build_training_frame(df)                 # accepted-only
    assert list(out.columns) == TRAINING_COLUMNS
    # Rejected event 1 is dropped; the accepted set (incl. the hand-added #3) survives.
    assert list(out["accepted"]) == [True, True, True]
    assert list(out["added"]) == [False, False, True]
    assert out["added"].sum() == 1


def test_build_training_frame_include_all_keeps_rejected():
    df = _frame(accepted=[True, False, True], added=[False, False, False])
    out = build_training_frame(df, include_all=True)
    assert len(out) == 3
    assert list(out["accepted"]) == [True, False, True]


def test_training_selection_explicit_spellings_preserve_the_legacy_modes():
    df = _frame(accepted=[True, False, True], added=[False, False, True])

    pd.testing.assert_frame_equal(
        build_training_frame(df),
        build_training_frame(df, selection=TRAINING_SELECTION_ACCEPTED),
    )
    pd.testing.assert_frame_equal(
        build_training_frame(df, include_all=True),
        build_training_frame(df, selection=TRAINING_SELECTION_ALL),
    )


def test_reviewed_training_selection_keeps_human_decisions_of_both_signs_only():
    df = _frame(
        accepted=[True, False, False, True, False, True, False],
        added=[False] * 7,
    )
    df["flag"] = pd.array([
        "",                         # reviewed positive
        "",                         # reviewed negative
        "",                         # legacy human negative: verdict proves review
        "",                         # untouched detector candidate
        "clipped_by_boundary",      # reviewed token, but detector-owned rejection
        "",                         # legacy row: review state unknown
        "non_finite",               # legacy detector-owned rejection
    ], dtype="string")
    df[event_model.REVIEW_COL] = pd.array([
        event_model.REVIEW_DONE,
        event_model.REVIEW_DONE,
        event_model.REVIEW_NONE,
        event_model.REVIEW_NONE,
        event_model.REVIEW_DONE,
        pd.NA,
        pd.NA,
    ], dtype="string")

    out = build_training_frame(df, selection=TRAINING_SELECTION_REVIEWED)

    # Positive and negative HUMAN verdicts are supervised labels. Untouched/unknown
    # candidates and automatic detector rejections are not.
    assert list(out["start_idx"]) == [0, 100, 200]
    assert list(out["accepted"]) == [True, False, False]


def test_reviewed_training_selection_excludes_unresolved_curation_conflicts():
    df = _frame(accepted=[True, False, False], added=[False] * 3)
    df[event_model.REVIEW_COL] = pd.array(
        [event_model.REVIEW_DONE] * 3, dtype="string"
    )
    df["consensus_curation_conflict"] = pd.array(
        ["", "accepted|rejected", pd.NA], dtype="string"
    )

    out = build_training_frame(df, selection=TRAINING_SELECTION_REVIEWED)

    assert list(out["start_idx"]) == [0, 200]
    assert list(out["accepted"]) == [True, False]


def test_reviewed_training_selection_fails_closed_without_an_accepted_label():
    df = _frame(accepted=[True, False]).drop(columns="accepted")

    out = build_training_frame(df, selection=TRAINING_SELECTION_REVIEWED)

    assert out.empty
    assert list(out.columns) == TRAINING_COLUMNS


@pytest.mark.parametrize("selection", ["kept", "", "accepted-and-reviewed"])
def test_training_selection_rejects_unknown_values(selection):
    df = _frame(accepted=[True])

    with pytest.raises(ValueError, match="training selection must be one of"):
        build_training_frame(df, selection=selection)


def test_training_selection_rejects_conflicting_legacy_and_explicit_arguments():
    df = _frame(accepted=[True])

    with pytest.raises(ValueError, match="include_all=True conflicts"):
        build_training_frame(
            df, include_all=True, selection=TRAINING_SELECTION_REVIEWED)


def test_build_training_frame_materialises_missing_added():
    df = _frame(accepted=[True, True])              # no 'added' column
    out = build_training_frame(df)
    assert "added" in out.columns
    assert not out["added"].any()


def test_training_frame_regenerates_the_reserved_label_namespace_from_flags():
    df = _frame(accepted=[True, True])
    df["flag"] = pd.array([
        "type=collision",
        "molecule=apo-ferritin;type=translocation",
    ], dtype="string")
    # Simulate a frame cached before its canonical flag tokens were edited.
    df["label_type"] = ["stale-a", "stale-b"]
    df["label_orphan"] = ["must-not", "survive"]

    out = build_training_frame(df)

    assert list(out.columns) == TRAINING_COLUMNS + ["label_type", "label_molecule"]
    assert list(out["label_type"]) == ["collision", "translocation"]
    assert list(out["label_molecule"]) == ["", "apo-ferritin"]
    assert "label_orphan" not in out.columns


def test_training_labels_are_discovered_only_from_rows_that_are_exported():
    df = _frame(accepted=[True, False])
    df["flag"] = pd.array(["molecule=apo", "private=withheld"], dtype="string")

    out = build_training_frame(df)

    assert "label_molecule" in out.columns
    assert "label_private" not in out.columns


def test_write_training_export_csv_and_sidecar(tmp_path):
    df = _frame(
        accepted=[True, False, True, True],
        added=[False, False, False, True],
        ids=[10, 11, 12, 13],
    )
    prov = {"fs": 100_000.0, "polarity": -1, "min_dwell_s": 5e-4,
            "detrend_enabled": True, "detrend_method": "block_median",
            "notch_enabled": False}
    out = write_training_export(
        df, tmp_path / "train.csv", provenance=prov,
        source_file="/data/pore.abf", detector="detect.keras",
    )
    # The data file holds exactly the accepted rows (incl. the added one), in order.
    loaded = pd.read_csv(out["data"])
    assert list(loaded.columns) == TRAINING_COLUMNS
    assert list(loaded["accepted"]) == [True, True, True]
    assert list(loaded["added"]) == [False, False, True]

    # The sidecar records the reproducibility contract.
    meta = json.loads(out["meta"].read_text())
    assert meta["source_file"] == "/data/pore.abf"
    assert meta["detector"] == "detect.keras"
    assert meta["fs"] == 100_000.0
    assert meta["selection"] == "accepted"
    assert meta["n_rows"] == 3
    assert meta["detrend_contract"]["detrend_method"] == "block_median"


def test_write_training_export_reviewed_selection_and_sidecar_agree(tmp_path):
    df = _frame(
        accepted=[True, False, True, False],
        added=[False] * 4,
        ids=[41, 7, 99, 12],
    )
    df["flag"] = pd.array(["", "", "", "clipped_by_boundary"], dtype="string")
    df[event_model.REVIEW_COL] = pd.array([
        event_model.REVIEW_DONE,
        event_model.REVIEW_DONE,
        event_model.REVIEW_NONE,
        event_model.REVIEW_DONE,
    ], dtype="string")

    paths = write_training_export(
        df, tmp_path / "reviewed.csv", selection=TRAINING_SELECTION_REVIEWED)
    written = pd.read_csv(paths["data"])
    sidecar = json.loads(paths["meta"].read_text())

    assert list(written["accepted"]) == [True, False]
    assert sidecar["selection"] == TRAINING_SELECTION_REVIEWED
    assert sidecar["n_rows"] == len(written) == 2
    assert sidecar["exported_event_ids"] == [41, 7]
    assert "event_id" not in written.columns
    note = sidecar["curation"]["note"]
    assert "resolved human-reviewed decisions" in note
    assert "accepted positives and rejected negatives" in note
    assert "accepted rows only" not in note


def test_training_sidecar_carries_json_safe_reconciliation_provenance(tmp_path):
    df = _frame(accepted=[True])
    reconciliation = {
        "version": np.int64(1),
        "algorithm": "constrained-overlap-coefficient",
        "input_modules": ["detect.schmitt", "detect.keras"],
        "memberships": [{"event_id": np.int64(0), "source_rows": np.array([2, 7])}],
    }

    paths = write_training_export(
        df,
        tmp_path / "reconciled.csv",
        provenance={"fs": np.float64(50_000), "reconciliation": reconciliation},
    )
    written = pd.read_csv(paths["data"])
    sidecar = json.loads(paths["meta"].read_text())

    assert sidecar["reconciliation"] == {
        "version": 1,
        "algorithm": "constrained-overlap-coefficient",
        "input_modules": ["detect.schmitt", "detect.keras"],
        "memberships": [{"event_id": 0, "source_rows": [2, 7]}],
    }
    assert "reconciliation" not in written.columns


def test_training_sidecar_carries_label_vocabulary_as_metadata_only(tmp_path):
    df = _frame(accepted=[True])
    vocabulary = [{
        "key": "molecule",
        "value": "apo-ferritin",
        "hotkey": "c",
        "note": "500 pM standard",
    }]

    paths = write_training_export(
        df,
        tmp_path / "labelled.csv",
        provenance={"fs": 50_000.0, "label_vocabulary": vocabulary},
    )
    written = pd.read_csv(paths["data"])
    sidecar = json.loads(paths["meta"].read_text())

    assert sidecar["label_vocabulary"] == vocabulary
    assert "label_vocabulary" not in written.columns


def test_write_training_export_parquet(tmp_path):
    df = _frame(accepted=[True, True], added=[False, True])
    out = write_training_export(df, tmp_path / "train.parquet")
    assert out["data"].suffix == ".parquet"
    loaded = pd.read_parquet(out["data"])
    assert list(loaded.columns) == TRAINING_COLUMNS
    assert len(loaded) == 2


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
@pytest.mark.parametrize("labeled", [False, True])
def test_training_sidecar_columns_exactly_match_written_table(tmp_path, suffix, labeled):
    df = _frame(accepted=[True, True], added=[False, True])
    if labeled:
        df["flag"] = ["molecule=apo-ferritin;type=collision", "molecule=holo-ferritin"]

    out = write_training_export(df, tmp_path / f"train{suffix}")
    loaded = (pd.read_parquet(out["data"]) if suffix == ".parquet"
              else pd.read_csv(out["data"]))
    meta = json.loads(out["meta"].read_text())
    expected = TRAINING_COLUMNS + (["label_molecule", "label_type"] if labeled else [])

    assert list(loaded.columns) == expected
    assert meta["columns"] == list(loaded.columns)


def test_training_provenance_is_json_safe():
    prov = {"fs": np.float64(50_000.0), "polarity": np.int64(-1),
            "filter_cutoff_hz": np.float32(10_000.0), "detrend_enabled": np.bool_(True)}
    meta = training_provenance(prov, source_file="x.abf", detector="detect.schmitt")
    # Must round-trip through json without a numpy type sneaking in.
    text = json.dumps(meta)
    back = json.loads(text)
    assert back["fs"] == 50_000.0
    assert back["detrend_contract"]["filter_cutoff_hz"] == pytest.approx(10_000.0)
