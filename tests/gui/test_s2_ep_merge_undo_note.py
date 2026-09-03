"""EP2-02 — UNDOING A MERGE THAT CONSUMED A REFUSED ROW BRINGS ITS REFUSAL BACK.

BL-2's remap (``MainWindow._remap_refusals``) retires a refusal whose anchored
``start_idx`` vanished — right for the merge (the merged-away row no longer exists) and,
before this fix, irreversible: the retire was a one-way ``ledger.forget``. Edit ▸ Undo
installs the pre-merge frame under the SAME lineage token, so the frame again holds all
three rows the accept-all was refused over — and the disclosure on the gallery, the
events table and the inspector kept saying "2 … kept rejected" indefinitely. That
contradicts the ledger's own contract ("a count of currently-refused rows, not a latch
… goes back UP when it is rejected again") and USER_GUIDE 9b ("An undo restores the
table exactly").

THE FIX THIS PINS. ``_remap_refusals`` returns what it retired, as
``start_idx -> bucket``; the lineage cache keeps those anchors beside the ledger, and
``_restore_retired_refusals`` re-records each one on the first re-fed frame that holds
its start again. A redo of the merge simply retires them again — pinned below too.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore

import nanoscope.gui.event_gallery as gal_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_N = 220_000
#: The D2 geometry: a refused (over-dwell) row R with measurable neighbours A and B
#: close enough that "merge A with next (6 ms window)" swallows R — the merge that
#: retires R's refusal. Plus a second over-dwell event and two clean far ones, so the
#: detector's refusals number three (with the clipped opening).
_A = (50_000, 50_300)
_R = (50_700, 60_700)          # 100 ms dwell -> refused (over the 5 ms limit)
_B = (61_100, 61_400)
_LONG1 = (10_000, 20_000)
_CLEAN_FAR = [(90_000, 90_300), (110_000, 110_250)]
_MAX_DWELL_S = 5e-3


def _trace() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(_N, 100.0) + rng.normal(0.0, 1.5, _N)
    x[0:300] -= 30.0
    for s, e in [_LONG1, _A, _R, _B] + _CLEAN_FAR:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_s2d2.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _drain(qtbot, ms: int = 150) -> None:
    qtbot.wait(ms)
    QtCore.QCoreApplication.instance().processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)


def _refused_and_rejected(df) -> list[int]:
    return [i for i in range(len(df))
            if not bool(df["accepted"].iloc[i])
            and not gal_mod._derived_accepted(df["flag"].iloc[i])]


@pytest.fixture
def merged_after_note(qapp, qtbot, tmp_path):
    """A wired MainWindow taken to the moment AFTER the merge: 3 refused rows, note = 3
    everywhere, then a merge across R (note correctly recounts to 2)."""
    win = MainWindow()
    qtbot.addWidget(win)
    doc = _doc(tmp_path)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    panel._detection.max_dwell_s.setValue(_MAX_DWELL_S)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    _drain(qtbot)
    df = panel.current_detection().payload.df
    unmeasurable = [i for i in range(len(df))
                    if not gal_mod._derived_accepted(df["flag"].iloc[i])]
    assert len(unmeasurable) == 3, f"fixture wants 3 refusals, got {df['flag'].tolist()}"

    win._act_event_gallery.trigger()
    _drain(qtbot)
    gal = win._event_gallery
    win._act_event_inspector.trigger()
    _drain(qtbot)
    insp = win._event_inspector
    # Reject-all first so the accept-all WRITES rows and every surface repaints its
    # note through the ordinary data signals — the pre-merge state is not under test.
    gal.reject_all_btn.click()
    _drain(qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()
    tbl = win._analysis_panel._table
    assert "3 events kept rejected" in tbl.refusal_label.text(), tbl.refusal_label.text()

    # Merge A with its forward run: the 6 ms window reaches across R and B, so the
    # refused row is CONSUMED and its ledger entry retires (correct — pinned below).
    starts = df["start_idx"].to_numpy()
    a_id = int(df["event_id"].to_numpy()[int(np.argmin(np.abs(starts - _A[0])))])
    report = win._analysis_panel.merge_event_with_next(a_id, 0.006)
    _drain(qtbot)
    assert report.status == "ok", f"the merge did not run: {report}"
    assert "2 tiles kept rejected" in gal.status.text(), (
        "sanity (the merge leg, pinned by BL-2): the note must recount to 2 while the "
        "merged-away refusal is off the frame:\n" + gal.status.text())

    try:
        yield win, gal, insp, tbl, len(df)
    finally:
        win._analysis_controller.abort()


def test_undo_of_the_merge_restores_the_retired_refusal_on_every_surface(
        merged_after_note, qtbot):
    win, gal, insp, tbl, n_before = merged_after_note

    win._act_undo.trigger()
    _drain(qtbot)

    restored = win._analysis_panel.current_detection().payload.df
    assert len(restored) == n_before, "the undo did not restore the pre-merge frame"
    assert len(_refused_and_rejected(restored)) == 3, \
        "ground truth: the restored frame must hold 3 refused-and-rejected rows"
    assert "3 tiles kept rejected" in gal.status.text(), (
        "the gallery still undercounts the standing refusals after the undo:\n"
        + gal.status.text())
    assert "3 events kept rejected" in tbl.refusal_label.text(), (
        "the events table still undercounts after the undo:\n"
        + repr(tbl.refusal_label.text()))
    assert "3 events kept rejected" in insp._batch_note.text(), (
        "the inspector still undercounts after the undo:\n"
        + repr(insp._batch_note.text()))


def test_redo_of_the_merge_retires_the_refusal_again(merged_after_note, qtbot):
    """The restore must not turn the retire into a latch the other way: redoing the
    merge takes the row off the frame again, and the note goes back to 2."""
    win, gal, insp, tbl, n_before = merged_after_note

    win._act_undo.trigger()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()

    win._act_redo.trigger()
    _drain(qtbot)
    redone = win._analysis_panel.current_detection().payload.df
    assert len(redone) == n_before - 2, "the redo did not reinstall the merged frame"
    assert "2 tiles kept rejected" in gal.status.text(), (
        "after the redo the merged-away refusal must retire again:\n"
        + gal.status.text())
    assert "2 events kept rejected" in tbl.refusal_label.text(), \
        repr(tbl.refusal_label.text())
