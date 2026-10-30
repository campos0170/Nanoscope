"""S9-01 — Ctrl+M files a hand-made event into the table on screen, and Ctrl+Z takes that add back.

MEASURED DEFECT (sweep 9, a synthetic 60,000-sample recording with a Schmitt and a CUSUM table on
one channel). ``AnalysisPanel.target_module_id`` answered with the channel's remembered choice
BEFORE the table on screen, and only the Table selector and the Detection tab's detector combo
wrote that choice. Choose Schmitt in the Table selector, then Analysis ▸ Reconcile: the selector
reads "Reconciled — 2 candidates", the curator rejects one Reconciled event, brackets a missed
event and presses Ctrl+M, and the status bar said "Added event 2 to Schmitt. ⌘Z undoes it." while
the Edit menu read "Undo Curate event". Pressing it re-accepted the rejected Reconciled event and
left the hand row in Schmitt, out of view, where it had also made Reconciled stale. No Table
selector use was needed: switching the detector combo to CUSUM to run it, which every
two-detector reconcile starts with, recorded the choice too, so Ctrl+M never gave the refusal the
design promises for a derived table ("Reconciled is derived. Show Schmitt or CUSUM and add the
event there", MANUAL_EVENT_TABLE_DESIGN 4.2; 4.4: "the chosen table, which is the table on
screen"). A newly installed run took the screen without the choice following, so a project saved
then reopened on whichever table came first.

THE FIX. On the channel shown, ``target_module_id`` answers with the table on screen
(``_stored_detection``), the table Edit ▸ Undo is keyed to, so an add's "Ctrl+Z undoes it" is true
and Reconciled on screen refuses with the existing sentence, writing nothing. The remembered
choice is written with ``_active_detection`` in ``_set_active_detection``, the one writer behind
every door that puts a table on screen (the Table selector, a selection, a newly installed run,
Reconcile; the detector combo writes it as before), so the choice and the screen never disagree:
a save writes the table on screen (``event_table_choices_v1``) and a reopen shows it. Choosing
"Manual — no events yet" still files into Manual.

These tests drive a real MainWindow: the Table selector, the Detection tab's detector combo,
Analysis ▸ Reconcile (its preview dialog answered OK through the window's own factory seam), a
rejection through the panel method the table and inspector call, the Measure region and the
Ctrl+M action, Edit ▸ Undo, and Save / reopen of a project.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import MANUAL_MODULE_ID, EventSet
from nanoscope.analysis.manual_events import hand_made_mask
from nanoscope.analysis.reconcile import CONSENSUS_MODULE_ID, reconcile_detections
from nanoscope.core.analysis import AnalysisResult
from nanoscope.gui.main_window import MainWindow

from .test_manual_event_door import (  # noqa: F401  (``win`` is a fixture)
    FS, I0, _DETECTED, _MISSED, _doc, _open, _signal, _table, win)

SCHMITT = "detect.schmitt"
CUSUM = "detect.cusum"
REFUSAL = "Reconciled is derived"


class _PreviewAnsweredOK:
    """The Reconcile preview, answered OK: what the window's factory seam hands back."""

    def __init__(self, sources) -> None:
        self.outcome = reconcile_detections(sources)

    def exec(self) -> int:
        return QtWidgets.QDialog.Accepted


def _settle() -> None:
    QtWidgets.QApplication.processEvents()


def _cusum_run_lands(win, doc_id) -> None:
    """A second detector's table on the channel, installed the way a run's result is."""
    df, prov = detect_events(_signal(_DETECTED, seed=5), FS, baseline=I0, min_dwell_s=1e-4)
    win._session.add_result(AnalysisResult(
        module_id=CUSUM, module_version="1.0", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    _settle()


def _choose(win, module_id) -> None:
    """Pick ``module_id`` in the Results dock's Table selector."""
    sel = win._analysis_panel._result_selector
    sel.setCurrentIndex(sel.findData(module_id))
    _settle()


def _reconcile(win, monkeypatch) -> None:
    """Analysis ▸ Reconcile detector results…, preview answered OK."""
    monkeypatch.setattr(win, "_make_reconcile_dialog", _PreviewAnsweredOK)
    assert win._act_reconcile.isEnabled(), "the menu item would swallow the click"
    win._act_reconcile.trigger()
    _settle()


def _ctrl_m(win, span) -> str:
    """Measure on, the region bracketing ``span``, Ctrl+M; the status line after it."""
    win._analysis_panel.measure_panel.set_active(True)
    win._trace_view.set_measure_region(span[0] / FS, span[1] / FS)
    _settle()
    win._act_add_event.trigger()
    _settle()
    return win.statusBar().currentMessage()


def _shown(win) -> str:
    return win._analysis_panel.current_detection().module_id


def _choice(win, doc_id) -> str | None:
    return win._analysis_panel.chosen_tables().get((doc_id, 0))


def _hand_rows(win, doc_id, module_id) -> int:
    df = _table(win, doc_id, module_id)
    return 0 if df is None else int(hand_made_mask(df).sum())


def _ruling(win, doc_id, event_id) -> bool:
    rec = _table(win, doc_id, CONSENSUS_MODULE_ID)
    return bool(rec.loc[rec["event_id"] == event_id, "accepted"].iloc[0])


def _reconciled_with_one_rejection(win, doc_id, monkeypatch) -> int:
    """Reconciled on screen and the curator's first ruling on it; the ruled event's id."""
    _reconcile(win, monkeypatch)
    assert _shown(win) == CONSENSUS_MODULE_ID, "precondition: Reconcile puts its table on screen"
    ruled = int(_table(win, doc_id, CONSENSUS_MODULE_ID)["event_id"].iloc[0])
    win._analysis_panel.set_event_accepted(ruled, False)
    _settle()
    assert _ruling(win, doc_id, ruled) is False
    return ruled


# ================================================================================================
# the measured case, and the combo variant that needs no Table selector at all
# ================================================================================================
def test_with_reconciled_on_screen_ctrl_m_is_refused_and_writes_nothing(win, tmp_path,
                                                                        monkeypatch):
    """The finding's scenario: Schmitt chosen in the selector, then Reconcile, one rejection,
    Ctrl+M over a missed event. The add is refused with the design's sentence and writes
    nothing: no source table gains a row behind the curator's back (which would also have made
    Reconciled stale), no Manual table appears, and the ruling stands."""
    doc_id = _open(win, tmp_path)
    _cusum_run_lands(win, doc_id)
    _choose(win, SCHMITT)
    assert _shown(win) == SCHMITT
    ruled = _reconciled_with_one_rejection(win, doc_id, monkeypatch)

    said = _ctrl_m(win, _MISSED)

    assert REFUSAL in said, said
    assert _hand_rows(win, doc_id, SCHMITT) == 0 and _hand_rows(win, doc_id, CUSUM) == 0
    assert win._session.results.get(doc_id, 0, MANUAL_MODULE_ID) is None
    assert _shown(win) == CONSENSUS_MODULE_ID and _choice(win, doc_id) == CONSENSUS_MODULE_ID
    assert win._analysis_panel.target_table_name(doc_id, 0) == "Reconciled"
    assert _ruling(win, doc_id, ruled) is False


def test_the_detector_combo_then_reconcile_refuses_too(win, tmp_path, monkeypatch):
    """No Table selector use: the person switches the Detection tab's detector to CUSUM to run
    it, CUSUM's run lands, they reconcile and press Ctrl+M. That used to file into CUSUM."""
    doc_id = _open(win, tmp_path)
    combo = win._analysis_panel._detection.detector_combo
    combo.setCurrentIndex(combo.findData(CUSUM))
    _settle()
    _cusum_run_lands(win, doc_id)
    assert _shown(win) == CUSUM
    _reconcile(win, monkeypatch)

    said = _ctrl_m(win, _MISSED)

    assert REFUSAL in said, said
    assert _hand_rows(win, doc_id, CUSUM) == 0 and _hand_rows(win, doc_id, SCHMITT) == 0


# ================================================================================================
# the way forward the refusal names, and Ctrl+Z after an add
# ================================================================================================
def test_showing_a_source_table_files_there_and_ctrl_z_takes_back_exactly_that_add(
        win, tmp_path, monkeypatch):
    """The refusal's own advice, followed: show Schmitt, add it there. The message and the
    Edit menu name the add, and Ctrl+Z removes the row it added and leaves the Reconciled
    ruling the curator made before alone."""
    doc_id = _open(win, tmp_path)
    _cusum_run_lands(win, doc_id)
    _choose(win, SCHMITT)
    ruled = _reconciled_with_one_rejection(win, doc_id, monkeypatch)
    _choose(win, SCHMITT)
    assert _choice(win, doc_id) == SCHMITT

    said = _ctrl_m(win, _MISSED)

    assert "to Schmitt" in said and "undoes it" in said, said
    assert _hand_rows(win, doc_id, SCHMITT) == 1
    assert "Add event" in win._act_undo.text(), win._act_undo.text()
    win._act_undo.trigger()
    _settle()
    assert _hand_rows(win, doc_id, SCHMITT) == 0
    assert _ruling(win, doc_id, ruled) is False


def test_choosing_manual_with_reconciled_on_screen_files_into_manual_and_undoes(
        win, tmp_path, monkeypatch):
    """"Manual — no events yet" in the selector over Reconciled: the add goes to Manual, the
    table now on screen, and Ctrl+Z takes it back to an empty Manual table."""
    doc_id = _open(win, tmp_path)
    _cusum_run_lands(win, doc_id)
    ruled = _reconciled_with_one_rejection(win, doc_id, monkeypatch)
    _choose(win, MANUAL_MODULE_ID)
    assert _shown(win) == MANUAL_MODULE_ID and _choice(win, doc_id) == MANUAL_MODULE_ID

    said = _ctrl_m(win, _MISSED)

    assert "to Manual" in said, said
    assert len(_table(win, doc_id, MANUAL_MODULE_ID)) == 1
    win._act_undo.trigger()
    _settle()
    assert len(_table(win, doc_id, MANUAL_MODULE_ID)) == 0
    assert _hand_rows(win, doc_id, SCHMITT) == 0
    assert _ruling(win, doc_id, ruled) is False


# ================================================================================================
# the choice follows every door that changes the table on screen, and survives a reopen
# ================================================================================================
def test_the_remembered_table_follows_the_table_on_screen(win, tmp_path, monkeypatch):
    """Each door that puts a table on screen moves the channel's remembered table with it, so
    the trace label, the add and Edit ▸ Undo all name one table."""
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    seen = []

    def step(label):
        seen.append((label, _shown(win), _choice(win, doc_id),
                     panel.target_module_id(doc_id, 0)))

    step("a run's table on screen")
    _cusum_run_lands(win, doc_id)
    step("a newly installed run")
    _choose(win, SCHMITT)
    step("the Table selector")
    combo = panel._detection.detector_combo
    combo.setCurrentIndex(combo.findData(CUSUM))
    _settle()
    step("the detector combo")
    _reconcile(win, monkeypatch)
    step("Reconcile")
    _choose(win, MANUAL_MODULE_ID)
    step("Manual — no events yet")

    assert [(s, c, t) for _label, s, c, t in seen] == [
        (SCHMITT, SCHMITT, SCHMITT),
        (CUSUM, CUSUM, CUSUM),
        (SCHMITT, SCHMITT, SCHMITT),
        (CUSUM, CUSUM, CUSUM),
        (CONSENSUS_MODULE_ID, CONSENSUS_MODULE_ID, CONSENSUS_MODULE_ID),
        (MANUAL_MODULE_ID, MANUAL_MODULE_ID, MANUAL_MODULE_ID),
    ], seen


def test_a_reopened_project_shows_and_files_into_the_table_that_was_on_screen(qapp, qtbot,
                                                                              tmp_path):
    """CUSUM's run took the screen with no selector use. The project saved with it on screen
    reopens on CUSUM, and Ctrl+M there files into CUSUM, not into whichever table came first."""
    from nanoscope.core.project import load_project, save_project

    win = MainWindow()
    qtbot.addWidget(win)
    doc_id = _open(win, tmp_path)
    _cusum_run_lands(win, doc_id)
    assert _shown(win) == CUSUM

    path = tmp_path / "p.nanoscope"
    save_project(path, win._gather_project_state())
    state = load_project(path)
    saved = state.workspace.extra.get("event_table_choices_v1")
    assert isinstance(saved, list) and [r["module_id"] for r in saved] == [CUSUM], saved

    win2 = MainWindow()
    qtbot.addWidget(win2)
    live_id = win2._session.add_document(_doc(tmp_path, _signal(_DETECTED + [_MISSED])))
    win2._create_default_layers(win2._session.document(live_id))
    win2._restore_state = state
    win2._apply_restore_for_doc(live_id)
    win2._on_selection(live_id, 0)
    _settle()

    assert _shown(win2) == CUSUM
    said = _ctrl_m(win2, _MISSED)
    assert "to CUSUM" in said, said
    assert _hand_rows(win2, live_id, CUSUM) == 1 and _hand_rows(win2, live_id, SCHMITT) == 0
