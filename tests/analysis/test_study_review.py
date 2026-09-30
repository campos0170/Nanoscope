"""Review approval follows the measured event snapshot, not the visible screen."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.incubation_detection import PreparedRecording, RecordingAnalysis
from nanoscope.analysis.manual_detection import recording_identity
from nanoscope.analysis.manual_workspace import load_workspace, save_workspace
from nanoscope.analysis.study_review import (
    REVIEW_KEY, confirm_review, require_reviewed, review_fingerprint, review_is_current,
    EVENT_REVIEW_COL, apply_event_decision, event_reviews, retain_event_decisions,
)


@pytest.fixture
def row():
    raw = np.full(100, -100., dtype=np.float64)
    signal = np.zeros(100, dtype=np.float32)
    signal[10:13] = -10
    signal[30:34] = -20
    prepared = PreparedRecording("zn_30h", raw, signal, 1000., .1,
        {"event_polarity": 1, "retained_ranges": [[0, 100, 0, 100]]},
        {"preparation_config": {"baseline_window_s": .5}, "raw_sha256": "source-one"})
    events = pd.DataFrame({"event_id": ["zn_30h:0", "zn_30h:1"],
        "start_idx": [10, 30], "end_idx": [12, 33], "accepted": [True, False],
        "passes_gate": [True, False], "depth_sust_pA": [10., 20.],
        "optional_measurement": [np.nan, 3.]})
    analysis = RecordingAnalysis("zn_30h", raw, signal, 1000., .1, events,
        {"recording_identity": recording_identity(prepared),
         "measurement_settings": {"min_depth_sigma": 4.}, "recipe_id": "fixture"})
    return {"tag": "zn_30h", "recording_id": "stable-id", "prepared": prepared,
            "analysis": analysis, "source_path": Path("/raw/example.abf"),
            "decode_settings": {"enter": .4}, "measurement_settings": {"min_depth_sigma": 4.},
            "preparation_settings": {"baseline_window_s": .5}, "incubation_h": 30.}


def approve(row):
    row[REVIEW_KEY] = confirm_review(row)


def test_explicit_receipt_is_detached_and_does_not_change_decisions(row):
    before = row["analysis"].catalog.copy(deep=True)
    assert not review_is_current(row)
    receipt = confirm_review(row)
    assert REVIEW_KEY not in row
    assert receipt["event_count"] == 2 and receipt["accepted_count"] == 1
    pd.testing.assert_frame_equal(row["analysis"].catalog, before)
    row[REVIEW_KEY] = receipt
    assert review_is_current(row)
    require_reviewed([row])
    row["analysis"].catalog.loc[1, "accepted"] = True
    assert receipt["accepted_count"] == 1
    assert not review_is_current(row)


@pytest.mark.parametrize("change", [
    lambda row: row["analysis"].catalog.loc.__setitem__((slice(None), "accepted"), [False, False]),
    lambda row: row["analysis"].catalog.loc.__setitem__((1, "end_idx"), 34),
    lambda row: row["analysis"].catalog.loc.__setitem__((1, "event_id"), "zn_30h:new"),
    lambda row: row["analysis"].catalog.loc.__setitem__((1, "depth_sust_pA"), 19.),
    lambda row: row["analysis"].provenance["measurement_settings"].update(min_depth_sigma=5.),
    lambda row: row["analysis"].provenance["recording_identity"].update(detrended_sha256="new"),
    lambda row: row["decode_settings"].update(enter=.5),
    lambda row: row["preparation_settings"].update(baseline_window_s=1.),
    lambda row: row["prepared"].provenance.update(raw_sha256="source-two"),
    lambda row: row["prepared"].metadata.update(retained_ranges=[[0, 100, 100, 200]]),
    lambda row: row.update(incubation_h=31.),
    lambda row: row.update(recording_id="other-recording"),
])
def test_scientific_changes_make_approval_stale_including_rejected_events(row, change):
    approve(row)
    change(row)
    assert not review_is_current(row)
    with pytest.raises(ValueError, match="zn_30h"):
        require_reviewed([row])


def test_drafts_and_styling_are_not_scientific_approval_evidence(row):
    approve(row)
    row.update(drafts={"decode": {"enter": .9}}, style={"palette": "viridis"},
               figures=[Path("/figure.png")], signal_nulls=pd.DataFrame({"count": [3]}))
    # The GUI blocks unapplied drafts separately, without destroying receipts.
    assert review_is_current(row)
    row["analysis"].catalog.index = pd.Index([99, 101], name="display_index")
    assert review_is_current(row)


def test_saved_workspace_receipt_survives_restoration_and_cannot_approve_legacy(row, tmp_path):
    approve(row)
    save_workspace({"records": [row]}, tmp_path)
    restored = load_workspace(tmp_path)["records"][0]
    assert review_fingerprint(restored) == review_fingerprint(row)
    assert review_is_current(restored)
    restored.pop(REVIEW_KEY)
    assert not review_is_current(restored)
    # Existing analysis remains intact and can still be inspected.
    pd.testing.assert_frame_equal(restored["analysis"].catalog, row["analysis"].catalog)


def test_pending_error_names_only_unapproved_inputs(row):
    approved = deepcopy(row)
    approve(approved)
    second = deepcopy(row)
    second["tag"] = second["analysis"].tag = "cu_30h"
    with pytest.raises(ValueError, match="cu_30h") as error:
        require_reviewed([approved, second])
    assert "zn_30h" not in str(error.value)
    assert "Use reviewed events for analysis" in str(error.value)


@pytest.mark.parametrize("column,values,match", [
    ("accepted", ["true", "false"], "booleans"),
    ("accepted", [True, pd.NA], "booleans"),
    ("accepted", [1, 0], "booleans"),
    ("start_idx", [10., 30.], "integer sample indices"),
    ("start_idx", [-1, 30], "integer sample indices"),
    ("end_idx", [12, 100], "inclusive intervals"),
    ("end_idx", [35, 33], "inclusive intervals"),
    ("end_idx", [9, 33], "inclusive intervals"),
    ("event_id", ["duplicate", "duplicate"], "identities"),
    ("event_id", ["", "second"], "identities"),
])
def test_invalid_event_contract_cannot_be_approved(row, column, values, match):
    row["analysis"].catalog[column] = values
    with pytest.raises(ValueError, match=match):
        confirm_review(row)
    assert not review_is_current(row)


def test_changed_prepared_geometry_cannot_approve_old_measurements(row):
    row["prepared"].fs_hz = 500.
    row["prepared"].duration_s = .2
    with pytest.raises(ValueError, match="Prepared recording changed"):
        confirm_review(row)


def test_zero_candidates_can_be_explicitly_reviewed(row):
    row["analysis"].catalog = row["analysis"].catalog.iloc[:0].copy()
    approve(row)
    assert row[REVIEW_KEY]["event_count"] == row[REVIEW_KEY]["accepted_count"] == 0
    assert review_is_current(row)


def test_malformed_receipts_fail_closed(row):
    for receipt in (None, [], {}, {"version": "future"}):
        row[REVIEW_KEY] = receipt
        assert not review_is_current(row)
    approve(row)
    row[REVIEW_KEY]["accepted_count"] = True
    assert not review_is_current(row)


def test_explicit_noop_attention_preserves_science_and_approval(row, tmp_path):
    approve(row)
    original = row['analysis']
    fingerprint = review_fingerprint(row)
    row['analysis'] = apply_event_decision(original, ['zn_30h:0'], True)
    assert event_reviews(row['analysis']).tolist() == ['reviewed', 'unreviewed']
    pd.testing.assert_frame_equal(row['analysis'].catalog.drop(columns=EVENT_REVIEW_COL), original.catalog)
    assert row['analysis'].provenance is original.provenance
    assert review_fingerprint(row) == fingerprint and review_is_current(row)
    save_workspace({'records': [row]}, tmp_path)
    restored = load_workspace(tmp_path)['records'][0]
    assert event_reviews(restored['analysis']).tolist() == ['reviewed', 'unreviewed']
    assert review_is_current(restored)


def test_fresh_accept_reject_accept_remains_reviewed_and_reset_returns_to_gate(row):
    original = row['analysis']
    updated = apply_event_decision(original, ['zn_30h:0'], False)
    updated = apply_event_decision(updated, ['zn_30h:0'], True)
    assert updated.catalog.accepted.tolist() == [True, False]
    assert event_reviews(updated).tolist() == ['reviewed', 'unreviewed']
    updated.catalog['source_review'] = ['reviewed', 'reviewed']
    reset = apply_event_decision(updated, [], None)
    assert reset.catalog.accepted.tolist() == original.catalog.passes_gate.tolist()
    assert event_reviews(reset).tolist() == ['unreviewed', 'unreviewed']
    assert reset.catalog.source_review.tolist() == ['reviewed', 'reviewed']
    assert updated.catalog[EVENT_REVIEW_COL].iloc[0] == 'reviewed'


def test_imported_automatic_gate_disagreement_does_not_invent_review(row):
    analysis = row['analysis']
    analysis.catalog['source_event_id'] = [4, 9]
    analysis.catalog['source_accepted'] = analysis.catalog.accepted
    analysis.catalog['source_review'] = ['unreviewed', 'reviewed']
    analysis.catalog['passes_gate'] = [False, True]
    assert event_reviews(analysis).tolist() == ['unreviewed', 'reviewed']
    # Older saved snapshots can still prove a later changed decision.
    analysis.catalog.loc[0, 'accepted'] = False
    assert event_reviews(analysis).tolist() == ['reviewed', 'reviewed']


def test_remeasurement_retains_review_by_source_id_and_exact_geometry(row):
    before = row['analysis']
    before.catalog['source_event_id'] = [7, 42]
    before.catalog['source_review'] = ['unreviewed', 'unreviewed']
    before = apply_event_decision(before, ['zn_30h:1'], True)
    measured = replace(before, catalog=before.catalog.drop(columns=EVENT_REVIEW_COL).iloc[::-1].reset_index(drop=True))
    measured.catalog['accepted'] = [False, True]
    measured.catalog['depth_sust_pA'] = [21., 11.]
    output = retain_event_decisions(before, measured)
    assert output.catalog.source_event_id.tolist() == [42, 7]
    assert output.catalog.accepted.tolist() == [True, True]
    assert output.catalog[EVENT_REVIEW_COL].tolist() == ['reviewed', 'unreviewed']
    assert output.catalog.depth_sust_pA.tolist() == [21., 11.]
    assert measured.catalog.accepted.tolist() == [False, True]
    changed = replace(measured, catalog=measured.catalog.copy(deep=True))
    changed.catalog.loc[0, 'end_idx'] += 1
    with pytest.raises(ValueError, match='bounds changed'):
        retain_event_decisions(before, changed)
    changed = replace(measured, provenance={**measured.provenance, 'recording_identity': {'foreign': True}})
    with pytest.raises(ValueError, match='different recording'):
        retain_event_decisions(before, changed)


@pytest.mark.parametrize('ids', [[True], ['missing'], []])
def test_invalid_individual_review_does_not_change_original(row, ids):
    original = row['analysis'].catalog.copy(deep=True)
    with pytest.raises(ValueError):
        apply_event_decision(row['analysis'], ids, True)
    pd.testing.assert_frame_equal(row['analysis'].catalog, original)


def test_legacy_review_inference_uses_positions_when_display_index_is_duplicated(row):
    frame = row['analysis'].catalog
    frame['source_event_id'] = [10, 11]
    frame['source_accepted'] = frame.accepted
    frame['source_review'] = ['unreviewed', 'unreviewed']
    frame['source_flag'] = ''
    frame.index = [7, 7]
    assert event_reviews(row['analysis']).tolist() == ['unreviewed', 'reviewed']
