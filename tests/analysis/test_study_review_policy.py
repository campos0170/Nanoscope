"""Explicit study review changes no native/legacy rejection inference."""
import numpy as np
import pandas as pd

from nanoscope.analysis.events import (event_status, status_of, FLAG_CLIPPED, FLAG_NON_FINITE,
    STUDY_REVIEW_POLICY_COL, STUDY_REVIEW_POLICY)
from nanoscope.analysis.scoring import build_training_frame


def test_scoped_status_and_training_selection_agree_for_every_review_state():
    frame = pd.DataFrame({'event_id': list(range(7)), 'start_idx': list(range(7)), 'accepted': [False] * 5 + [True, True],
        'review': ['unreviewed', 'reviewed', pd.NA, 'reviewed', 'reviewed', 'unreviewed', 'reviewed'],
        'flag': ['', '', '', FLAG_CLIPPED, FLAG_NON_FINITE, '', ''],
        STUDY_REVIEW_POLICY_COL: [STUDY_REVIEW_POLICY] * 7})
    expected = ['auto-rejected', 'rejected', 'unknown', 'auto-rejected', 'auto-rejected', 'unreviewed', 'accepted']
    assert event_status(frame).tolist() == expected
    assert [status_of(row) for _, row in frame.iterrows()] == expected
    trained = build_training_frame(frame, selection='reviewed')
    assert trained.start_idx.tolist() == [1, 6]
    assert trained.accepted.tolist() == [False, True]
    # Absence or an unrecognized policy retains the native legacy contract.
    legacy = frame.drop(columns=STUDY_REVIEW_POLICY_COL)
    assert event_status(legacy).tolist()[:3] == ['rejected'] * 3
    legacy[STUDY_REVIEW_POLICY_COL] = pd.NA
    assert event_status(legacy).tolist()[:3] == ['rejected'] * 3
    legacy[STUDY_REVIEW_POLICY_COL] = 'unknown-future-policy'
    assert [status_of(row) for _, row in legacy.iterrows()][:3] == ['rejected'] * 3


def test_mixed_policy_rows_do_not_change_each_others_status():
    frame = pd.DataFrame({'event_id': [0, 1, 2], 'start_idx': [0, 1, 2], 'accepted': [False] * 3,
        'review': ['unreviewed'] * 3, 'flag': [''] * 3,
        STUDY_REVIEW_POLICY_COL: [STUDY_REVIEW_POLICY, pd.NA, np.nan]})
    assert event_status(frame).tolist() == ['auto-rejected', 'rejected', 'rejected']
    assert build_training_frame(frame, selection='reviewed').start_idx.tolist() == [1, 2]
