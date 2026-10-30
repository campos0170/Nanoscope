"""S9-06 — A re-run that finds one event where two hand-made events were drawn keeps both notes.

MEASURED DEFECT (sweep 9, a synthetic recording with a real event at samples 20,000-20,300 that the
first Schmitt run was not shown). Two old measurements on its halves, labelled "first half: slow
entry" (20,000-20,120) and "second half: fast exit" (20,160-20,300), were converted into events
(Analysis ▸ Convert old measurements to events…, where a label becomes the event's note) and copied
into Schmitt (Analysis ▸ Copy hand-made events to…). Detection was then re-run with settings that
find 20,000-20,300 as ONE event. ``manual_events.carry_hand_rows`` moved each hand row onto that
detected row (overlap coefficient 1.0 for both) with ``new_df.iat[best, note] = str(note)``, so the
second note overwrote the first: the table kept "second half: fast exit" only, while the status
line said "Kept 2 hand-made events; 2 are now detected (1, 1), their labels moved onto them". The
labels two lines above were merged (``combine_flags``); the notes were not. The design keeps notes
"even when the new run detects the same event" (MANUAL_EVENT_TABLE_DESIGN section 2, point 4).

THE FIX. A note moved onto a detected row is joined with every note earlier hand rows moved there
(``manual_events.join_notes``, handed the whole list each time): in time order, since the hand rows
are visited by start, with ``NOTE_JOINER`` ("; "), and a blank or repeated note adds nothing, even
when a third row repeats a note that is already part of a joined one. The report and the status
line are unchanged: both events were kept, and both are now that one detection. (The merges in
this build keep the first fragment's note, so there was no joiner to share; ``NOTE_JOINER`` is the
one the design's merge rule in section 3 can use.)

These tests drive a real MainWindow through the path a note really takes into a table: the old
measurement list as a project restore leaves it, Analysis ▸ Convert old measurements to events…,
Analysis ▸ Copy hand-made events to… (each question answered through its seam), and Analysis ▸ Run
detection on the worker. The last two pin the order and the de-duplication on the pure function.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import manual_events as me
from nanoscope.analysis.baseline import DEFAULT_BASELINE
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import MANUAL_MODULE_ID, EventSet

from .test_manual_event_door import (  # noqa: F401  (``win`` is a fixture)
    FS, I0, _open, _seed_measurements, _table, win)

SCHMITT = "detect.schmitt"
FIRST = (20_000, 20_120)
SECOND = (20_160, 20_300)


@pytest.fixture
def modals(monkeypatch):
    """Every modal the gestures raise, recorded: a prompt nobody expected fails the test."""
    seen: list[str] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: seen.append(self.text()) or 0)
    return seen


def _convert_and_copy(win, doc_id, monkeypatch, labels) -> None:
    """Two labelled old measurements on the halves of the event, converted into the Manual
    table, then copied into Schmitt: the way a hand-made row comes to carry a note."""
    _seed_measurements(win, doc_id, [(*FIRST, labels[0]), (*SECOND, labels[1])])
    win._sync_export_action()                 # what a restore does once the list has landed
    monkeypatch.setattr(win, "_choose_measurements_to_convert", lambda plan: [0, 1])
    assert win._act_convert_measurements.isEnabled(), "the menu item would swallow the click"
    win._act_convert_measurements.trigger()
    QtWidgets.QApplication.processEvents()
    assert win._analysis_panel.current_detection().module_id == MANUAL_MODULE_ID
    monkeypatch.setattr(win, "_choose_copy_target", lambda: SCHMITT)
    assert win._act_copy_hand_made.isEnabled(), "the menu item would swallow the click"
    win._act_copy_hand_made.trigger()
    QtWidgets.QApplication.processEvents()
    schmitt = _table(win, doc_id, SCHMITT)
    assert me.hand_made_mask(schmitt).sum() == 2, "precondition: both halves are in Schmitt"


def _rerun(win, qtbot) -> list[str]:
    """Analysis ▸ Run detection over the whole trace, which holds the event whole; the status
    lines the panel said on the way."""
    panel = win._analysis_panel
    said: list[str] = []
    panel.statusMessage.connect(said.append)
    panel._detection.min_dwell_s.setValue(1e-4)
    assert win._act_run_detection.isEnabled(), "the menu item would swallow the click"
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        win._act_run_detection.trigger()
    qtbot.wait(30)
    return said


def _detected_row(win, doc_id):
    df = _table(win, doc_id, SCHMITT)
    hit = df.loc[(df["start_idx"] < SECOND[1]) & (df["end_idx"] > FIRST[0])]
    assert len(hit) == 1, f"the re-run finds the event whole: {hit[['start_idx', 'end_idx']]}"
    return hit.iloc[0]


# ================================================================================================
# the measured case, through the real path
# ================================================================================================
def test_a_rerun_that_finds_both_halves_as_one_event_keeps_both_notes_in_time_order(
        win, qtbot, tmp_path, monkeypatch, modals):
    doc_id = _open(win, tmp_path)
    _convert_and_copy(win, doc_id, monkeypatch,
                      ("first half: slow entry", "second half: fast exit"))

    said = _rerun(win, qtbot)

    report = win._analysis_panel.last_carry_report
    assert report is not None and len(report.moved) == 2 and not report.kept
    row = _detected_row(win, doc_id)
    assert {det for _old, det in report.moved} == {int(row["event_id"])}
    assert row["note"] == "first half: slow entry; second half: fast exit"
    assert not bool(row.get("added", False)), "still the detector's own row"
    # The report is as it was: both events kept, both now this one detection.
    d = int(row["event_id"])
    assert any(f"Kept 2 hand-made events; 2 are now detected ({d}, {d})" in m for m in said), said
    assert modals == [], modals


def test_two_halves_with_the_same_note_keep_it_once(win, qtbot, tmp_path, monkeypatch, modals):
    doc_id = _open(win, tmp_path)
    _convert_and_copy(win, doc_id, monkeypatch, ("slow", "slow"))

    _rerun(win, qtbot)

    assert _detected_row(win, doc_id)["note"] == "slow"


# ================================================================================================
# the pure rule: order, blanks and repeats
# ================================================================================================
def _carried(rows):
    """Hand rows ``[(span, note), ...]``, stored in that order, carried into a fresh run that
    finds 20,000-20,300 whole: the carried EventSet and its report."""
    rng = np.random.default_rng(3)
    sig = (np.full(60_000, I0) + rng.normal(0.0, 8.0, 60_000)).astype(np.float32)
    sig[20_000:20_300] += 300.0
    contract = me.build_contract(sig, FS, "pA", config=DEFAULT_BASELINE, recording="r.dat",
                                 doc_id="d", channel=0)
    old = me.empty_manual_events(contract).df
    for event_id, (span, note) in enumerate(rows):
        row = me.measure_hand_row(sig, FS, *span, provenance=contract.provenance, units="pA",
                                  recording="r.dat", pore_id="p", event_id=event_id, note=note)
        old = pd.concat([old, row], ignore_index=True)       # append order, not time order
    det, prov = detect_events(sig, FS, min_dwell_s=1e-4)
    return me.carry_hand_rows(EventSet(det, prov), old, sig, fs=FS, units="pA",
                              old_module_id=SCHMITT)


def test_the_carry_over_joins_notes_in_time_order_whatever_order_the_rows_are_in():
    """Three hand rows on one detected event, stored out of time order, two with a note and
    one blank: the detected row reads the two notes in the order the halves lie on the trace."""
    out, report = _carried([((20_200, 20_300), "late"), ((20_000, 20_090), "early"),
                            ((20_100, 20_190), None)])

    assert len(report.moved) == 3
    assert out.df["note"].iloc[0] == "early; late"
    assert me.NOTE_JOINER == "; "
    assert me.join_notes(pd.NA, "a", "", None, "a", "b") == "a; b"
    assert me.join_notes(pd.NA, None) is pd.NA


def test_a_third_hand_row_repeating_the_first_note_adds_nothing():
    """Three thirds of one event noted "slow", "fast" and "slow" again: the detected row reads
    each note once, though the third lands on a note already joined from the first two."""
    out, report = _carried([((20_000, 20_090), "slow"), ((20_100, 20_190), "fast"),
                            ((20_200, 20_300), "slow")])

    assert len(report.moved) == 3 and len({det for _old, det in report.moved}) == 1
    assert out.df["note"].iloc[0] == "slow; fast"
