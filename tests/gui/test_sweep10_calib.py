"""Sweep 10, S10-DATA1 (with S10-CAL1): a calibration commit that moves no number must not
leave the sidecar somewhere Edit ▸ Undo cannot reach.

An Apply or Reset that moved no number in the session still rewrote ``<datafile>.npcal.json``
and made no step, so the bytes it wrote were recorded nowhere. Three routes followed:

1. the user's version-1 file re-encoded as version 2, same numbers, after which a redo kept
   for another channel found the file "changed since this step" and refused to put it back;
2. the Apply that line advises ("Apply it in the Calibration panel to store it") stored
   values that every later undo then refused to take out of the file, so the recording
   reopened with calibrations the user had undone;
3. after a project restore, an Apply with nothing typed wrote the project's gain over the
   user's own, note and all, with no Undo.

Now a file that already opens as the write would leave it is not rewritten, and a commit
that does change the file is a step whose Undo puts the user's exact bytes back. Each test
ends with a fresh ``Session`` opening the same file, which must agree with the window's.

Synthetic recordings only, each under ``tmp_path``; the panel's confirmations are answered
through their seams.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

from nanoscope.core.calibration import sidecar_path
from nanoscope.core.document import TraceDocument
from nanoscope.core.segment import SegmentDocument
from nanoscope.core.session import Session
from nanoscope.gui.main_window import MainWindow

from .test_dataloss_regressions import _doc as _csv_document, _event_trace, _write_csv
from .test_editing_settings_undo import (NOTE, _apply, _open, _panel, _redo,
                                         _status, _undo, _write_sidecar)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _gains(w: MainWindow, doc_id: str) -> tuple[float, ...]:
    doc = w._session.document(doc_id)
    return tuple(w._session.calibration(doc_id, info.index).gain for info in doc.meta.channels)


def _reopened(w: MainWindow, doc_id: str) -> tuple[float, ...]:
    """The gains a fresh session gives the same recording, from the file alone."""
    doc = w._session.document(doc_id)
    fresh = Session()
    did = fresh.add_document(TraceDocument(meta=doc.meta, channels=doc.channels))
    return tuple(fresh.calibration(did, info.index).gain for info in doc.meta.channels)


def _file_gains(w: MainWindow, doc_id: str) -> dict | None:
    sc = sidecar_path(w._session.document(doc_id).meta.path)
    if not sc.exists():
        return None
    return {k: v["gain"] for k, v in json.loads(sc.read_text())["channels"].items()}


# ================================================================================================
# route 1: a no-change Apply or Reset on the user's version-1 file strands a redo
# ================================================================================================

@pytest.mark.parametrize("press", ["apply", "reset"])
def test_a_no_change_press_leaves_the_users_file_and_the_redo_puts_its_step_back(
        window, qtbot, tmp_path, press):
    sc = _write_sidecar(tmp_path / "pore.dat", (2.0, "pA", "note zero"), (3.0, "pA", "note one"))
    theirs = sc.read_bytes()
    did = _open(window, tmp_path, channels=3)
    panel, _asked = _panel(window)
    assert _gains(window, did) == (2.0, 3.0, 1.0)

    window._on_selection(did, 1)
    _apply(panel, "0.5")
    redone = sc.read_bytes()
    _undo(window)
    assert sc.read_bytes() == theirs and _gains(window, did) == (2.0, 3.0, 1.0)

    # Apply on ch0 with nothing typed, or Reset on the untouched ch2: no number moves.
    window._on_selection(did, 0 if press == "apply" else 2)
    (panel._apply if press == "apply" else panel._reset)()
    assert sc.read_bytes() == theirs, "a press that moved no number rewrote the user's file"
    assert window._act_redo.text() == "Redo Apply calibration"

    _redo(window)
    assert _gains(window, did) == (2.0, 0.5, 1.0)
    assert sc.read_bytes() == redone, _status(window)
    assert "left as it is" not in _status(window)
    assert _reopened(window, did) == _gains(window, did)


def test_a_re_indented_version_two_file_is_left_as_it_is_too(window, tmp_path):
    """Not version 1 as such: any file not in this build's own byte layout was rewritten."""
    data = tmp_path / "pore.dat"
    sc = sidecar_path(data)
    sc.write_text(json.dumps({"version": 2, "channels": {"0": {
        "gain": 2.0, "offset": 0.0, "target_units": "pA", "note": NOTE,
        "name": "current", "kind": "current"}}}, indent=4))
    theirs = sc.read_bytes()
    did = _open(window, tmp_path)
    panel, asked = _panel(window)
    panel._apply()
    assert sc.read_bytes() == theirs and asked == []
    assert not window._act_undo.isEnabled()
    assert panel.status_text() == f"Applied · saved to {sc.name}"


# ================================================================================================
# route 2: the KEPT line's advice no longer leaves undone calibrations in the file
# ================================================================================================

def test_following_the_kept_advice_is_a_step_and_undoing_everything_empties_the_file(
        window, tmp_path):
    did = _open(window, tmp_path, "two.dat", channels=2)
    panel, _asked = _panel(window)

    window._on_selection(did, 0)
    _apply(panel, "2")                                      # ch0 1 -> 2
    _undo(window)
    window._on_selection(did, 1)
    _apply(panel, "3")                                      # ch1 1 -> 3
    _redo(window)                                           # ch0's redo: the file is ch1's now
    assert "left as it is" in _status(window)
    assert "Apply it in the Calibration panel" in _status(window)
    assert _gains(window, did) == (2.0, 3.0) and _file_gains(window, did) == {"1": 3.0}

    window._on_selection(did, 0)
    panel._apply()                                          # as advised, nothing typed
    assert _file_gains(window, did) == {"0": 2.0, "1": 3.0}
    assert window._act_undo.text() == "Undo Apply calibration"
    assert _reopened(window, did) == (2.0, 3.0)

    _undo(window)                                           # the advised Apply
    assert _file_gains(window, did) == {"1": 3.0}
    assert "left as it is" not in _status(window)
    _undo(window)                                           # ch0's redone step
    assert _gains(window, did) == (1.0, 3.0) and _file_gains(window, did) == {"1": 3.0}
    assert _reopened(window, did) == _gains(window, did)
    _undo(window)                                           # ch1's step
    assert _gains(window, did) == (1.0, 1.0)
    assert _file_gains(window, did) is None, "calibrations the user undid stayed in the file"
    assert _reopened(window, did) == (1.0, 1.0)

    _redo(window)
    _redo(window)
    _redo(window)
    assert _gains(window, did) == (2.0, 3.0) and _file_gains(window, did) == {"0": 2.0, "1": 3.0}
    assert _reopened(window, did) == _gains(window, did)


# ================================================================================================
# route 3: after a project restore, an untyped Apply over the user's own value is undoable
# ================================================================================================

def test_an_untyped_apply_after_a_restore_is_a_step_whose_undo_gives_back_the_users_file(
        window, qtbot, tmp_path):
    csv = tmp_path / "pore.csv"
    trace = _event_trace()
    _write_csv(csv, trace)
    did = window._session.add_document(_csv_document(csv, trace))
    window._create_default_layers(window._session.document(did))
    window._on_selection(did, 0)
    panel, asked = _panel(window)
    _apply(panel, "0.5")
    project = tmp_path / "pore.nanoscope"
    assert window._write_project(str(project))
    # Between sessions the user edits their own file: the project says 0.5, the file 0.75.
    sc = _write_sidecar(csv, (0.75, "pA", NOTE))
    theirs = sc.read_bytes()

    with qtbot.waitSignal(window.projectRestored, timeout=30_000):
        window._open_project(str(project))
    qtbot.wait(50)
    restored = next(d.doc_id for d in window._session.documents()
                    if not isinstance(d, SegmentDocument))
    window._on_selection(restored, 0)
    asked.clear()
    assert window._session.calibration(restored, 0).gain == 0.5

    panel._apply()                                          # nothing typed
    assert asked == [("note", "Apply")]
    assert json.loads(sc.read_text())["channels"]["0"]["gain"] == 0.5
    assert window._act_undo.isEnabled()
    assert window._act_undo.text() == "Undo Apply calibration"
    assert _reopened(window, restored) == (0.5,)

    _undo(window)
    assert sc.read_bytes() == theirs, "the user's own value and note did not come back"
    assert _status(window) == f"Undone: Apply calibration · saved to {sc.name}."
    assert window._session.calibration(restored, 0).gain == 0.5    # the project's, as before
    assert _reopened(window, restored) == (0.75,)

    _redo(window)
    assert json.loads(sc.read_text())["channels"]["0"]["gain"] == 0.5
    assert _reopened(window, restored) == _gains(window, restored)


def test_an_apply_that_moves_neither_a_number_nor_the_file_is_still_no_step(window, tmp_path):
    """Design 3 / 13.2's rule stands: a file this build wrote, Applied again unchanged."""
    did = _open(window, tmp_path)
    panel, _asked = _panel(window)
    _apply(panel, "0.5")
    sc = sidecar_path(tmp_path / "pore.dat")
    written = sc.read_bytes()
    _undo(window)
    _redo(window)
    assert window._settings_history.undo_depth(("calibration", did, 0)) == 1
    panel._apply()
    assert sc.read_bytes() == written
    assert window._settings_history.undo_depth(("calibration", did, 0)) == 1
