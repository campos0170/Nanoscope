"""S7-DET2 (widget half) -- the ScorePanel withholds the ratios until a PERSON has
curated, and says so; the detector's own automatic rejections do not defeat the guard.

Measured before the fix on the owner's -600 mV file: one boundary-clipped row ->
precision 0.997 / F1 0.998 rendered with the caption hidden; a 5 ms max dwell ->
"truth (kept) 77, spurious (FP) 240, precision 0.243, F1 0.391", zero gestures.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import FLAG_CLIPPED, FLAG_MAX_DWELL
from nanoscope.analysis.scoring import ADDED_COLUMN, ScoreResult, score_detection
from nanoscope.gui.panels import ScorePanel


def _frame(accepted, flags, added=None) -> pd.DataFrame:
    n = len(accepted)
    df = pd.DataFrame({
        "event_id": list(range(n)),
        "accepted": np.array(accepted, dtype=bool),
        "flag": list(flags),
    })
    df[ADDED_COLUMN] = np.array(added if added is not None else [False] * n, dtype=bool)
    return df


@pytest.fixture
def panel(qapp, qtbot):
    p = ScorePanel()
    qtbot.addWidget(p)
    return p


def test_a_single_boundary_clipped_row_no_longer_turns_the_ratios_on(panel):
    df = _frame([True] * 316 + [False], [""] * 316 + [FLAG_CLIPPED])
    panel.set_score(score_detection(df))
    for key in ("precision", "recall", "f1"):
        assert panel.value_text(key) == "not scored yet"
    assert panel.tautology_label.isVisibleTo(panel)
    assert "by construction" in panel.tautology_label.text()
    assert "1 row(s) the detector rejected on its own" in panel.tautology_label.text()
    assert panel.value_text("n_detected") == "316"
    assert panel.value_text("fp") == "0"
    assert panel.value_text("n_auto_rejected") == "1"


def test_a_max_dwell_cut_is_reported_as_auto_rejected_not_as_two_hundred_forty_false_positives(panel):
    flags = [FLAG_MAX_DWELL] * 239 + [FLAG_CLIPPED] + [""] * 77
    df = _frame([False] * 240 + [True] * 77, flags)
    panel.set_score(score_detection(df))
    assert panel.value_text("n_truth") == "77"
    assert panel.value_text("fp") == "0"
    assert panel.value_text("n_auto_rejected") == "240"
    assert panel.value_text("precision") == "not scored yet"
    assert panel.tautology_label.isVisibleTo(panel)
    assert "240 row(s)" in panel.tautology_label.text()


def test_one_hand_rejection_turns_the_ratios_on_and_hides_the_caption(panel):
    df = _frame([True, True, True, False, False], ["", "", "", "", FLAG_CLIPPED])
    panel.set_score(score_detection(df))
    assert panel.value_text("fp") == "1"
    assert panel.value_text("n_auto_rejected") == "1"
    assert panel.value_text("precision") == "0.750"
    assert not panel.tautology_label.isVisibleTo(panel)
    assert panel.tautology_label.text() == ""


def test_a_table_that_is_all_auto_rejections_is_still_uncurated(panel):
    df = _frame([False, False], [FLAG_MAX_DWELL, FLAG_MAX_DWELL])
    s = score_detection(df)
    assert ScorePanel.is_tautological(s)
    panel.set_score(s)
    assert panel.value_text("precision") == "not scored yet"
    assert panel.tautology_label.isVisibleTo(panel)


def test_the_guard_asks_for_user_gestures_and_the_legacy_constructor_still_works():
    fresh = ScoreResult(5, 5, 5, 0, 0, 1.0, 1.0, 1.0, [], [], [])
    assert ScorePanel.is_tautological(fresh)
    rejected = ScoreResult(5, 3, 3, 2, 0, 0.6, 1.0, 0.75, [], [], [])
    assert not ScorePanel.is_tautological(rejected)
    added = ScoreResult(5, 6, 5, 0, 1, 1.0, 0.833, 0.909, [], [], [])
    assert not ScorePanel.is_tautological(added)
    empty = ScoreResult(0, 0, 0, 0, 0, None, None, None, [], [], [])
    assert not ScorePanel.is_tautological(empty)
    only_auto = ScoreResult(0, 0, 0, 0, 0, None, None, None, [], [], [],
                            n_auto_rejected=3)
    assert ScorePanel.is_tautological(only_auto)
