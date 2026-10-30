"""A merge keeps every fragment's note, origin and legacy measurement ids.

MEASURED DEFECT (left open by the cross-group review of the sweep-9 fixes, 2026-10-29; a synthetic
recording with one event split into four fragments 20 samples apart). `merge._merged_row` carried
every non-boolean extra column as the run's first value (``series.iloc[0]``), so Merge all, Merge
events in selection and the Event inspector's Merge with next each turned four hand-made fragments
noted "slow entry", nothing, "slow entry" and "fast exit", copied from rows 7, 8, none and 9, and
converted from measurements 12, 15, none and "20,21", into one event that said "slow entry",
"detect.manual:7" and "12". The design asks a merge to join the notes and concatenate the other two
(MANUAL_EVENT_TABLE_DESIGN section 3). Both id columns are read back, so the loss showed elsewhere:
four converted measurements merged in the Manual table kept the id "0" alone, so the old list
offered measurements 1-3 as unconverted again, and four copies merged in Schmitt kept the origin
"detect.manual:0" alone, so copying again called one row already copied and three overlaps.

THE FIX. ``note`` is joined by ``manual_events.join_notes``, the rule a re-run's carry-over uses:
every fragment's note at once, in time order (the run is start-sorted), with ``NOTE_JOINER``
("; "), a blank or repeated note adding nothing. ``origin`` and ``legacy_measurement_ids`` are
listed by the new ``manual_events.join_ids``: every id once, in time order, joined by ``ID_JOINER``
(","), the separator the conversion's reader splits on and the design writes (``"12,15"``). A cell
that already lists several (a row merged before) is split first, so no id is listed twice. Copy's
"already copied" check now reads every id of an origin. Undo brings each fragment back as it was.

The first test drives each merge through its own control in a real MainWindow, then Edit ▸ Undo;
the next two make these cells the way the app does, through the panel doors behind Analysis ▸
Convert old measurements to events… and Analysis ▸ Copy hand-made events to…; the last pins the
rule on the pure function.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import manual_events as me
from nanoscope.analysis.baseline import DEFAULT_BASELINE
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import MANUAL_MODULE_ID, EventSet
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.core.analysis import AnalysisResult

from .test_manual_event_door import (  # noqa: F401  (``win`` is a fixture)
    FS, I0, _doc, _seed_measurements, _signal, _table, win)

SCHMITT = "detect.schmitt"
#: One event the detector split into four (20-sample gaps, which a 1 ms window bridges), and an
#: event far from it that every merge must leave alone.
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_400), (10_420, 10_700), (10_720, 11_000)]
_FAR = (30_000, 30_300)
WINDOW_S = 1e-3
#: What each fragment says, in time order: its note, the row it was copied from, and the old
#: measurements it was converted from. The second has no note, the third repeats the first's note
#: and names no row, and the fourth was merged before, so it already lists two measurements.
_CELLS = [
    ("slow entry", "detect.manual:7", "12"),
    (None, "detect.manual:8", "15"),
    ("slow entry", None, None),
    ("fast exit", "detect.manual:9", "20,21"),
]
_NOTE = "slow entry; fast exit"
_ORIGIN = "detect.manual:7,detect.manual:8,detect.manual:9"
_LEGACY = "12,15,20,21"
#: The labels of the old measurements on the fragments, for the conversion path.
_LABELS = ("slow entry", "", "slow entry", "fast exit")
_KEPT = ["start_idx", "end_idx", "note", "origin", "legacy_measurement_ids"]


@pytest.fixture
def modals(monkeypatch):
    """Every modal the gestures raise, recorded: a prompt nobody expected fails the test."""
    seen: list[str] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: seen.append(self.text()) or 0)
    return seen


def _open(win, tmp_path, *, hand_rows=True):
    """The recording on screen with its Schmitt table: the far event as the detector found it (it
    was not given the shattered one) and, with ``hand_rows``, the four fragments as hand-made rows
    that say what :data:`_CELLS` says."""
    signal = _signal(_FRAGMENTS + [_FAR])
    doc_id = win._session.add_document(_doc(tmp_path, signal, name="merge.dat"))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(_signal([_FAR], seed=1), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == 1, "precondition: the detector holds the far event only"
    if hand_rows:
        recording, pore_id, n0 = str(df["recording"].iloc[0]), str(df["pore_id"].iloc[0]), len(df)
        for k, ((s, e), (note, origin, ids)) in enumerate(zip(_FRAGMENTS, _CELLS)):
            row = me.measure_hand_row(
                signal, FS, s, e, provenance=prov, units="pA", recording=recording,
                pore_id=pore_id, event_id=n0 + k, note=note, origin=origin, legacy_ids=ids)
            df = me.insert_hand_row(df, row)
    win._session.add_result(AnalysisResult(
        module_id=SCHMITT, module_version="1.0", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    return doc_id


def _show(panel, module_id) -> None:
    sel = panel._result_selector
    sel.setCurrentIndex(sel.findData(module_id))
    assert panel.current_detection().module_id == module_id


def _merged(df) -> pd.Series:
    """The one row the four fragments became."""
    hit = df.loc[(df["start_idx"] < _FRAGMENTS[-1][1]) & (df["end_idx"] > _FRAGMENTS[0][0])]
    assert len(hit) == 1, f"the fragments did not merge: {hit[['start_idx', 'end_idx']]}"
    return hit.iloc[0]


# ------------------------------------------------------------------------------------------------
# the three merges, each through its own control
# ------------------------------------------------------------------------------------------------
def _merge_all(win, qtbot) -> None:
    """Results ▸ Merge… ▸ Merge all, with a window that bridges the fragments and not the far
    event. It runs on the worker, so its outcome arrives on ``mergeFinished``."""
    controls = win._analysis_panel.merge_controls
    controls.set_window_s(WINDOW_S)
    with qtbot.waitSignal(win._analysis_panel.mergeFinished, timeout=30_000) as blocker:
        controls.merge_all_btn.click()
    qtbot.wait(20)
    assert blocker.args[0].status == "ok", blocker.args[0]


def _merge_selection(win, qtbot) -> None:
    """Results ▸ Merge… ▸ Merge events in selection, over the four fragments only."""
    win._analysis_panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region((_FRAGMENTS[0][0] - 500) / FS,
                                       (_FRAGMENTS[-1][1] + 500) / FS)
    assert win._trace_view.measure_bounds() is not None
    win._analysis_panel.merge_controls.merge_region_btn.click()


def _merge_with_next(win, qtbot) -> None:
    """The Event inspector's Merge with next, on the first fragment."""
    win._open_event_inspector()
    insp = win._event_inspector
    df = win._analysis_panel.current_detection().payload.df
    insp.show_event(int(df.loc[df["start_idx"] == _FRAGMENTS[0][0], "event_id"].iloc[0]))
    win._analysis_panel.merge_controls.set_window_s(WINDOW_S)
    insp._merge_next_btn.click()


@pytest.mark.parametrize("merge", [_merge_all, _merge_selection, _merge_with_next],
                         ids=["merge_all", "merge_events_in_selection", "merge_with_next"])
def test_a_merge_keeps_every_fragments_note_origin_and_ids_and_undo_brings_each_back(
        win, qtbot, tmp_path, modals, merge):
    doc_id = _open(win, tmp_path)
    before = _table(win, doc_id, SCHMITT)[_KEPT].copy()
    assert len(before) == 5

    merge(win, qtbot)

    df = _table(win, doc_id, SCHMITT)
    assert len(df) == 2, "the four fragments became one; the far event stayed apart"
    row = _merged(df)
    assert (int(row["start_idx"]), int(row["end_idx"])) == (_FRAGMENTS[0][0], _FRAGMENTS[-1][1])
    assert row["note"] == _NOTE
    assert row["origin"] == _ORIGIN
    assert row["legacy_measurement_ids"] == _LEGACY
    far = df.loc[df["start_idx"] >= _FAR[0]].iloc[0]
    assert all(pd.isna(far[c]) for c in ("note", "origin", "legacy_measurement_ids"))

    win._act_undo.trigger()
    QtWidgets.QApplication.processEvents()

    restored = _table(win, doc_id, SCHMITT)[_KEPT].reset_index(drop=True)
    pd.testing.assert_frame_equal(restored, before.reset_index(drop=True))
    assert modals == [], modals


# ------------------------------------------------------------------------------------------------
# what the ids are for: the old measurement list and Copy hand-made events to…
# ------------------------------------------------------------------------------------------------
def _convert(win, doc_id) -> None:
    """Four labelled old measurements on the fragments, converted into the Manual table (a label
    becomes the event's note, and the measurement's id its ``legacy_measurement_ids``)."""
    panel = win._analysis_panel
    _seed_measurements(win, doc_id, [(*span, label) for span, label in zip(_FRAGMENTS, _LABELS)])
    report = panel.convert_measurements([0, 1, 2, 3])
    assert len(report.converted) == 4, report
    assert panel.current_detection().module_id == MANUAL_MODULE_ID


def test_merged_converted_events_leave_every_old_measurement_converted(
        win, qtbot, tmp_path, modals):
    doc_id = _open(win, tmp_path, hand_rows=False)
    _convert(win, doc_id)
    panel = win._analysis_panel

    _merge_all(win, qtbot)

    row = _merged(_table(win, doc_id, MANUAL_MODULE_ID))
    assert row["note"] == _NOTE
    assert row["legacy_measurement_ids"] == "0,1,2,3"
    # The ids are how a converted original is told from an unconverted one: none is offered again.
    assert panel.converted_measurement_ids() == {0, 1, 2, 3}
    assert [r.already_converted for r in panel.conversion_plan()] == [True] * 4
    assert modals == [], modals


def test_copying_again_onto_merged_copies_copies_nothing_twice(win, qtbot, tmp_path, modals):
    doc_id = _open(win, tmp_path, hand_rows=False)
    _convert(win, doc_id)
    panel = win._analysis_panel
    first = panel.copy_hand_made_events(SCHMITT)
    assert first is not None and len(first.copied) == 4, first
    manual_ids = sorted(int(i) for i in _table(win, doc_id, MANUAL_MODULE_ID)["event_id"])
    _show(panel, SCHMITT)

    _merge_all(win, qtbot)

    row = _merged(_table(win, doc_id, SCHMITT))
    assert row["origin"] == ",".join(f"{MANUAL_MODULE_ID}:{i}" for i in manual_ids)
    assert row["note"] == _NOTE
    _show(panel, MANUAL_MODULE_ID)
    again = panel.copy_hand_made_events(SCHMITT)
    assert again is not None and again.copied == () and again.overlapping == (), again
    assert sorted(again.already_copied) == manual_ids
    assert len(_table(win, doc_id, SCHMITT)) == 2
    assert modals == [], modals


# ------------------------------------------------------------------------------------------------
# the rule, on the pure function
# ------------------------------------------------------------------------------------------------
def test_a_merge_lists_each_id_once_and_the_notes_in_time_order_however_the_rows_are_stored():
    """A Manual table stored out of time order, one fragment merged before (two ids), another
    naming one of those ids again, a blank note and a repeated one."""
    sig = _signal(_FRAGMENTS + [_FAR])
    contract = me.build_contract(sig, FS, "pA", config=DEFAULT_BASELINE, recording="merge.dat",
                                 doc_id="d", channel=0)
    cells = [("slow entry", "detect.schmitt:1", "12,15"), ("", None, "15"),
             ("slow entry", "detect.schmitt:1", None), ("fast exit", "detect.schmitt:4", "20")]
    df = me.empty_manual_events(contract).df
    for k, ((s, e), (note, origin, ids)) in enumerate(zip(_FRAGMENTS, cells)):
        row = me.measure_hand_row(sig, FS, s, e, provenance=contract.provenance, units="pA",
                                  recording="merge.dat", pore_id="p", event_id=k, note=note,
                                  origin=origin, legacy_ids=ids)
        df = me.insert_hand_row(df, row)
    df = df.iloc[[2, 0, 3, 1]].reset_index(drop=True)          # append order, not time order

    out, n_merged = merge_by_gap(EventSet(df, contract.events_provenance()), sig, FS,
                                 max_gap_s=WINDOW_S)

    assert n_merged == 3
    row = _merged(out.df)
    assert row["note"] == "slow entry; fast exit"
    assert row["origin"] == "detect.schmitt:1,detect.schmitt:4"
    assert row["legacy_measurement_ids"] == "12,15,20"
    assert me.ID_JOINER == ","
    assert me.join_ids(pd.NA, None, " ", "12, 15", "15", "20") == "12,15,20"
    assert me.join_ids(pd.NA, None) is pd.NA
