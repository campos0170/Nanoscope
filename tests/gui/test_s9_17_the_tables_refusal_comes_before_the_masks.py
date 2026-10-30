"""S9-17 — With a table on screen that takes no hand-made row, the first press gives the table's
refusal, as the Measure tab's preview does.

MEASURED DEFECT (found while integrating the sweep-9 fixes). S9-11's hold (an edge still unset)
and the second press for an unfitted edge, which S9-05 gave Ctrl+M and the Add button, sat in
``TraceKeyController.add`` before the add door's refusals of the TABLE. With Reconciled on screen
(S9-01 makes it the add's target) or a Manual table on an old calibration, the preview read "No
event can be stored here: Reconciled is derived. ..." while the first D or Ctrl+M said "Nothing was
added: the finish edge is not set yet. Point at it and press F ...", or "Press again to add it as
placed": the person was sent to set an edge, or press again, for a table that cannot take the row.
F, as advised, was then refused as Reconciled.

THE FIX. ``AnalysisPanel.refuse_add_for_the_table`` asks the door's span-independent refusals (a
channel that takes no events; Reconciled; a stale table; a detector with no stored table) with the
door's own sentences, and ``add`` asks it before anything about the mask, keeping the mask.
``add_manual_event`` asks the same two helpers at the points it always did, so its order and its
sentences do not change.

Reconciled is put on screen without a Table selector choice, as
``test_manual_event_door::test_reconciled_refuses_a_hand_add`` does, so it is the add's target on
e2c92cb too and the two trees compare press for press. Synthetic recordings only.
"""


from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.events import MANUAL_MODULE_ID, EventSet
from nanoscope.analysis.manual_events import hand_made_mask
from nanoscope.analysis.reconcile import CONSENSUS_MODULE_ID
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration

from .test_manual_event_door import (  # noqa: F401  (``win`` is a fixture)
    _MISSED, _open, _table, win)

SCHMITT = "detect.schmitt"
REFUSAL = "Reconciled is derived"
BASELINE = (45_000, 45_300)            # no step anywhere near: a fit there leaves the edge placed


def _settle() -> None:
    QtWidgets.QApplication.processEvents()


def _point(win, doc_id, sample, px=10.0) -> None:
    win._trace_view.pointer_sample = lambda: (doc_id, 0, float(sample), float(px))


def _wait_label(win, qtbot) -> None:
    label = win._trace_keys._label_timer
    qtbot.wait(10)
    qtbot.waitUntil(lambda: not label.isActive(), timeout=2_000)
    _settle()


def _show_reconciled(win, doc_id) -> None:
    base = win._session.results.get(doc_id, 0, SCHMITT)
    win._session.add_result(AnalysisResult(
        module_id=CONSENSUS_MODULE_ID, module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=base.calibration_fingerprint,
        payload=EventSet(base.payload.df.copy(), dict(base.payload.provenance))))
    _settle()
    panel = win._analysis_panel
    assert panel.current_detection().module_id == CONSENSUS_MODULE_ID
    # The add's target on BOTH trees: no table was chosen, so e2c92cb files nothing elsewhere.
    assert panel.target_module_id(doc_id, 0) == CONSENSUS_MODULE_ID


def _hand_rows(win, doc_id) -> int:
    n = 0
    for mid in (SCHMITT, MANUAL_MODULE_ID, CONSENSUS_MODULE_ID):
        df = _table(win, doc_id, mid)
        n += 0 if df is None else int(hand_made_mask(df).sum())
    return n


def _press(win, how: str) -> str:
    if how == "D":
        win._trace_keys.add()
    elif how == "Ctrl+M":
        assert win._act_add_event.isEnabled()
        win._act_add_event.trigger()
    else:
        button = win._analysis_panel.measure_panel._add_button
        assert button.isEnabled(), "the Add button is greyed"
        button.click()
    _settle()
    return win.statusBar().currentMessage()


def _preview(win, qtbot) -> str:
    _wait_label(win, qtbot)
    return win._analysis_panel.measure_panel.event_preview_text()


# -- (a) the half-built mask Shift+S alone leaves ---------------------------------------------------
@pytest.mark.parametrize("how", ["D", "Ctrl+M"])
def test_on_reconciled_the_first_press_on_a_half_built_mask_says_reconciled(win, tmp_path, qtbot, how):
    doc_id = _open(win, tmp_path)
    _show_reconciled(win, doc_id)
    _point(win, doc_id, _MISSED[0])
    win._trace_keys.place_edge("start", fit=False)              # Shift+S goes down on Reconciled
    assert win._trace_view.measure_bounds() is not None
    preview = _preview(win, qtbot)
    said = _press(win, how)
    assert _hand_rows(win, doc_id) == 0
    assert REFUSAL in said, f"preview: {preview!r} / first {how}: {said!r}"


def test_on_reconciled_no_press_sends_the_person_to_set_an_edge_first(win, tmp_path, qtbot):
    doc_id = _open(win, tmp_path)
    _show_reconciled(win, doc_id)
    keys = win._trace_keys
    _point(win, doc_id, _MISSED[0])
    keys.place_edge("start", fit=False)                          # Shift+S
    shift_s = win.statusBar().currentMessage()
    keys.add()                                                   # D
    first = win.statusBar().currentMessage()
    _point(win, doc_id, _MISSED[1])
    keys.place_edge("end", fit=True)                             # F
    f_said = win.statusBar().currentMessage()
    keys.place_edge("end", fit=False)                            # Shift+F
    keys.add()                                                   # D
    last = win.statusBar().currentMessage()
    assert _hand_rows(win, doc_id) == 0
    assert REFUSAL in last
    assert REFUSAL in first, (f"Shift+S: {shift_s!r} / D: {first!r} / F: {f_said!r} / "
                              f"Shift+F then D: {last!r}")


# -- (b) two placed (unfitted) edges, made while Schmitt was on screen ------------------------------
@pytest.mark.parametrize("how", ["D", "Ctrl+M", "Add button"])
def test_on_reconciled_the_first_press_on_unfitted_edges_asks_no_second_press(win, tmp_path, qtbot, how):
    doc_id = _open(win, tmp_path)
    keys = win._trace_keys
    _point(win, doc_id, BASELINE[0])
    keys.place_edge("start", fit=True)
    _point(win, doc_id, BASELINE[1])
    keys.place_edge("end", fit=True)
    assert keys._start.source == "placed" and keys._end.source == "placed"
    _show_reconciled(win, doc_id)
    assert win._trace_view.measure_bounds() is not None, "the mask survived the switch"
    preview = _preview(win, qtbot)
    said = _press(win, how)
    assert _hand_rows(win, doc_id) == 0
    second = _press(win, how)
    assert _hand_rows(win, doc_id) == 0
    assert REFUSAL in said, f"preview: {preview!r} / first {how}: {said!r} / second: {second!r}"


# -- control: a mask with both edges set by hand ------------------------------------------------------
@pytest.mark.parametrize("how", ["D", "Ctrl+M"])
def test_control_two_exact_edges_on_reconciled_first_press(win, tmp_path, qtbot, how):
    doc_id = _open(win, tmp_path)
    _show_reconciled(win, doc_id)
    keys = win._trace_keys
    _point(win, doc_id, _MISSED[0])
    keys.place_edge("start", fit=False)
    _point(win, doc_id, _MISSED[1])
    keys.place_edge("end", fit=False)
    preview = _preview(win, qtbot)
    said = _press(win, how)
    assert _hand_rows(win, doc_id) == 0
    assert REFUSAL in said, f"preview: {preview!r} / first {how}: {said!r}"


# -- the other table refusal: a Manual table on an old calibration -----------------------------------
@pytest.mark.parametrize("how", ["D", "Ctrl+M"])
def test_on_a_calibration_stale_manual_table_the_first_press_says_so(win, tmp_path, qtbot, how):
    doc_id = _open(win, tmp_path, detect=False)
    panel = win._analysis_panel
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok          # Manual made, and on screen
    _settle()
    assert panel.current_detection().module_id == MANUAL_MODULE_ID
    win._session.set_calibration(doc_id, 0, ChannelCalibration(gain=2.0, target_units="pA"),
                                 write_sidecar=False)
    _settle()
    before = int(hand_made_mask(_table(win, doc_id, MANUAL_MODULE_ID)).sum())
    _point(win, doc_id, BASELINE[0])
    win._trace_keys.place_edge("start", fit=False)                # Shift+S
    assert win._trace_view.measure_bounds() is not None
    preview = _preview(win, qtbot)
    said = _press(win, how)
    assert int(hand_made_mask(_table(win, doc_id, MANUAL_MODULE_ID)).sum()) == before
    assert "calibration changed" in said.lower(), f"preview: {preview!r} / first {how}: {said!r}"
