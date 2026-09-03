"""S7-DET2 -- the detector's own automatic rejections are not the user's false positives.

``score_detection`` read ``fp = ~accepted & ~added``, so a row the DETECTOR de-accepted
(``clipped_by_boundary``, ``dwell>max_dwell_s``, ``non_finite``) counted as a curation
gesture. Measured on the owner's -600 mV file with a 5 ms max dwell: "truth (kept) 77,
spurious (FP) 240, precision 0.243, F1 0.391" with zero gestures made, and the
tautology caption -- which exists to say those ratios mean nothing yet -- hidden.

Now a row the detector itself rejected is neither a detection to score nor a truth to
keep: it is counted apart as ``n_auto_rejected``, ``fp`` is the USER's rejections only,
and ``n_user_gestures`` (``fp + fn``) is the fact the readout's guard asks for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import (
    FLAG_CLIPPED,
    FLAG_CURATED,
    FLAG_MAX_DWELL,
    FLAG_NON_FINITE,
    FLAG_ZERO_BASELINE,
)
from nanoscope.analysis import scoring as S
from nanoscope.analysis.scoring import ADDED_COLUMN, ScoreResult, score_detection


def _frame(accepted, flags=None, added=None) -> pd.DataFrame:
    n = len(accepted)
    data = {
        "event_id": list(range(n)),
        "accepted": np.array(accepted, dtype=bool),
    }
    if flags is not None:
        data["flag"] = list(flags)
    if added is not None:
        data[ADDED_COLUMN] = np.array(added, dtype=bool)
    return pd.DataFrame(data)


def test_a_boundary_clipped_row_the_detector_rejected_is_not_a_user_false_positive():
    df = _frame([True, True, False], flags=["", "", FLAG_CLIPPED])
    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (2, 0, 0)
    assert s.n_auto_rejected == 1 and s.auto_rejected_ids == [2]
    assert s.n_detected == 2, "a row the detector disowned is not a detection to score"
    assert s.n_truth == 2
    assert s.spurious_ids == []
    assert s.n_user_gestures == 0
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0


def test_two_hundred_and_thirty_nine_max_dwell_rejections_are_zero_gestures():
    n = 317
    flags = [FLAG_MAX_DWELL] * 239 + [FLAG_CLIPPED] + [""] * (n - 240)
    accepted = [False] * 240 + [True] * (n - 240)
    s = score_detection(_frame(accepted, flags=flags))
    assert s.n_auto_rejected == 240
    assert (s.n_detected, s.n_truth, s.tp, s.fp, s.fn) == (77, 77, 77, 0, 0)
    assert s.n_user_gestures == 0
    assert s.precision == 1.0, "not 0.243: nobody curated this table"


@pytest.mark.parametrize("flag", ["", None, FLAG_CURATED, FLAG_ZERO_BASELINE,
                                  f"{FLAG_ZERO_BASELINE};molecule=apo"])
def test_a_rejection_with_no_automatic_reason_is_the_users_and_counts_as_spurious(flag):
    df = _frame([True, False], flags=["", flag])
    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)
    assert s.n_auto_rejected == 0
    assert s.spurious_ids == [1]
    assert s.n_user_gestures == 1
    assert s.precision == pytest.approx(0.5)


def test_an_automatic_token_beside_a_caveat_is_still_automatic():
    df = _frame([True, False, False],
                flags=["", f"{FLAG_ZERO_BASELINE};{FLAG_CLIPPED}", FLAG_NON_FINITE])
    s = score_detection(df)
    assert s.n_auto_rejected == 2 and s.fp == 0


def test_a_frame_without_a_flag_column_scores_exactly_as_before():
    # Every rejection is a person's when there is no reason column at all.
    df = _frame([True, True, False, False], added=[False, False, False, False])
    s = score_detection(df)
    assert (s.tp, s.fp, s.fn) == (2, 2, 0) and s.n_auto_rejected == 0
    assert not S.auto_rejected_mask(df).any()


def test_a_hand_added_row_is_never_auto_rejected_and_the_output_shape_is_unchanged():
    df = _frame([True, True], flags=["", FLAG_CLIPPED], added=[False, True])
    s = score_detection(df)
    assert s.n_auto_rejected == 0 and s.fn == 1 and s.n_user_gestures == 1
    # The positional constructor every existing caller uses still works, and the new
    # fields default: the training export's readers see the same eleven fields first.
    legacy = ScoreResult(5, 5, 5, 0, 0, 1.0, 1.0, 1.0, [], [], [])
    assert legacy.n_auto_rejected == 0 and legacy.n_user_gestures == 0
    d = s.as_dict()
    assert list(d)[:11] == ["n_detected", "n_truth", "tp", "fp", "fn", "precision",
                            "recall", "f1", "matched_ids", "missed_ids", "spurious_ids"]
    assert d["n_auto_rejected"] == 0


def test_auto_rejected_mask_names_exactly_the_detectors_rows():
    df = _frame([True, False, False, False],
                flags=["", FLAG_MAX_DWELL, "", FLAG_CURATED])
    assert S.auto_rejected_mask(df).tolist() == [False, True, False, False]
