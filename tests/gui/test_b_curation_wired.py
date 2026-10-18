"""Change B through a real MainWindow: undo / redo keep the curation windows' place,
the inspector's ⌘Z reaches the one undo stack, and one "Advance after decision" choice
shows on both windows.

Measured before: Edit ▸ Undo after a reject in the inspector reset it to All / the first
event, and with the gallery open the gallery's re-feed put its cursor on tile 0 and
announced it, dragging the inspector, the table and the trace to that event.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw_mod
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 200_000
_SPANS = [(5_000 + 15_000 * k, 5_400 + 15_000 * k) for k in range(12)]
#: The same, with event 2 split by a 50-sample gap: something for a merge to join.
_SPLIT = sorted(_SPANS[:2] + [(35_000, 35_200), (35_250, 35_400)] + _SPANS[3:])
_SETTINGS_KEY = "curation/advance_after_decision"


@pytest.fixture(autouse=True)
def _advance_setting_is_clean():
    QtCore.QSettings().remove(_SETTINGS_KEY)
    yield
    QtCore.QSettings().remove(_SETTINGS_KEY)


def _signal(spans=_SPANS) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(N, -1000.0) + rng.normal(0.0, 8.0, N)
    for s, e in spans:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _wired(window, tmp_path, *, stamped: bool = True, spans=_SPANS):
    signal = _signal(spans)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "b.dat", format_name="synthetic",
                         n_samples=int(signal.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(signal,))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    df, prov = detect_events(signal, FS, baseline=-1000.0, min_dwell_s=1e-4)
    assert len(df) == len(spans)
    if not stamped:
        # The ML / template payloads carry no run stamp: the case the ledger cannot
        # tell lineage for, and the one the keep-place hint must not depend on.
        prov = {k: v for k, v in prov.items() if k != "detected_at"}
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    window._open_event_gallery()
    window._open_event_inspector()
    return window._event_inspector, window._event_gallery


def _ids(window) -> list[int]:
    return [int(i) for i in window._analysis_panel.current_detection().payload.df["event_id"]]


@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "no-run-stamp"])
def test_edit_undo_keeps_the_inspector_and_gallery_where_they_were(window, tmp_path, qtbot,
                                                                    stamped):
    insp, gal = _wired(window, tmp_path, stamped=stamped)
    ids = _ids(window)
    insp.set_filter("Accepted only")
    insp.show_event(ids[5])
    gal.view.selectionModel().setCurrentIndex(gal._model.index(8, 0),
                                              QItemSelectionModel.ClearAndSelect)
    qtbot.wait(10)
    gal_cur = gal.current_event_id()
    insp.show_event(ids[5])
    insp.set_accepted_current(False)
    assert window._act_undo.isEnabled()
    gal_emits: list[int] = []
    gal.eventSelected.connect(gal_emits.append)
    window._act_undo.trigger()
    qtbot.wait(10)
    df = window._analysis_panel.current_detection().payload.df
    assert bool(df["accepted"].iloc[5]), "the undo did not restore the event"
    assert (insp.current_filter(), insp.current_event_id()) == ("Accepted only", ids[5])
    assert gal_emits == [], "the gallery's re-feed announced a selection"
    assert gal.current_event_id() in (gal_cur, ids[5])
    assert window._keep_curation_place is False


def test_the_inspectors_ctrl_z_undoes_on_the_one_stack_and_returns_to_the_decision(
        window, tmp_path, qtbot):
    insp, _gal = _wired(window, tmp_path)
    ids = _ids(window)
    insp.set_advance_after_decision(True)
    insp.show_event(ids[4])
    insp.decide_accept(False)
    assert insp.current_event_id() == ids[5]
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier)
    assert insp._handle_own_key(ev)
    qtbot.wait(10)
    df = window._analysis_panel.current_detection().payload.df
    assert bool(df["accepted"].iloc[4])
    assert insp.current_event_id() == ids[4]
    # ⇧⌘Z redoes it, and the window keeps its place.
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
    assert insp._handle_own_key(ev)
    qtbot.wait(10)
    df = window._analysis_panel.current_detection().payload.df
    assert not bool(df["accepted"].iloc[4])
    assert insp.current_event_id() == ids[4]


def test_a_kept_place_undo_and_redo_keep_the_panels_row_and_trace_marker(window, tmp_path,
                                                                          qtbot):
    insp, gal = _wired(window, tmp_path)
    ids = _ids(window)
    panel = window._analysis_panel
    insp.set_filter("Accepted only")
    insp.show_event(ids[6])
    assert panel.selected_event_id() == ids[6]
    insp.set_accepted_current(False)
    for act in (window._act_undo, window._act_redo):
        moved: list[int] = []
        insp.eventSelected.connect(moved.append)
        gal.eventSelected.connect(moved.append)
        act.trigger()
        qtbot.wait(10)
        assert panel.selected_event_id() == ids[6], act.text()
        assert insp.current_event_id() == gal.current_event_id() == ids[6]
        assert moved == [], "the quiet re-select echoed into the curation windows"
        insp.eventSelected.disconnect(moved.append)
        gal.eventSelected.disconnect(moved.append)


def test_ctrl_z_after_a_no_op_decision_undoes_the_other_event_and_stays(window, tmp_path,
                                                                        qtbot):
    insp, gal = _wired(window, tmp_path)
    ids = _ids(window)
    insp.set_advance_after_decision(True)
    for eid in (ids[2], ids[7]):                         # two real writes; 7 is the last
        insp.show_event(eid)
        insp.decide_accept(False)
    insp.show_event(ids[2])
    insp.decide_accept(False)                            # no-op on 2: no checkpoint
    assert insp.current_event_id() == ids[3]
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier)
    assert insp._handle_own_key(ev)
    qtbot.wait(10)
    df = window._analysis_panel.current_detection().payload.df
    assert bool(df["accepted"].iloc[7]) and not bool(df["accepted"].iloc[2])
    assert insp.current_event_id() == gal.current_event_id() == ids[3]
    assert window._analysis_panel.selected_event_id() == ids[3]


def test_the_inspectors_cmd_keys_never_extend_the_gallery_selection(window, tmp_path, qtbot):
    """⌘U / ⌘Z run while ⌘ is down; the gallery's mirror of the move must replace its
    selection, or one R in the gallery then rejects two events."""
    insp, gal = _wired(window, tmp_path)
    ids = _ids(window)
    gal.show()
    qtbot.wait(20)
    gal.view.selectionModel().setCurrentIndex(gal._model.index(7, 0),
                                              QItemSelectionModel.ClearAndSelect)
    QTest.keyPress(gal.view, Qt.Key_Control, Qt.ControlModifier)
    try:
        assert QtGui.QGuiApplication.keyboardModifiers() & Qt.ControlModifier
        assert insp._handle_own_key(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_U, Qt.ControlModifier))
        assert insp.current_event_id() == ids[0]
        assert gal.selected_event_ids() == [ids[0]]
        insp.set_advance_after_decision(True)
        insp.decide_accept(False)                        # 0 -> 1, the gallery follows
        assert insp._handle_own_key(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier))
        qtbot.wait(10)
        assert insp.current_event_id() == ids[0]
        assert gal.selected_event_ids() == [ids[0]]
    finally:
        QTest.keyRelease(gal.view, Qt.Key_Control)


def test_undoing_a_merge_never_keeps_a_renumbered_id(window, tmp_path, qtbot):
    insp, gal = _wired(window, tmp_path, spans=_SPLIT)
    ids = _ids(window)
    with qtbot.waitSignal(window._analysis_panel.mergeFinished, timeout=30_000):
        window._on_merge_all_requested(0.005)
    qtbot.wait(20)
    merged = _ids(window)
    assert len(merged) == len(ids) - 1
    insp.show_event(merged[-1])                    # an id that names another event after undo
    gal.view.selectionModel().setCurrentIndex(gal._model.index(6, 0),
                                              QItemSelectionModel.ClearAndSelect)
    window._act_undo.trigger()
    qtbot.wait(10)
    assert _ids(window) == ids
    # Renumbered frames get today's reset, not a kept id pointing at a different event.
    assert insp.current_filter() == "All" and insp.current_event_id() == ids[0]


def test_both_windows_count_the_same_progress_for_one_frame(window, tmp_path, qtbot):
    insp, gal = _wired(window, tmp_path)
    ids = _ids(window)

    def counts(w) -> str:
        return w.progress_text().split(" · ", 1)[-1] if w.progress_text().startswith("End") \
            else w.progress_text()

    insp.show_event(ids[2])
    insp.set_accepted_current(False)
    qtbot.wait(5)
    assert counts(insp) == counts(gal) == "Reviewed 1 / 12 · accepted 11"
    gal.view.selectionModel().setCurrentIndex(gal._model.index(7, 0),
                                              QItemSelectionModel.ClearAndSelect)
    gal.set_accepted_selected(False)
    qtbot.wait(5)
    assert counts(insp) == counts(gal) == "Reviewed 2 / 12 · accepted 10"


def test_one_advance_choice_shows_on_both_windows(window, tmp_path):
    insp, gal = _wired(window, tmp_path)
    gal.set_advance_after_decision(True)
    assert insp._advance_check.isChecked() and gal._advance_check.isChecked()
    insp._advance_check.setChecked(False)
    assert not gal._advance_check.isChecked()


def test_the_new_signals_are_classified_and_wired_by_main_window_only():
    for key in ("event_inspector.undoRequested", "event_inspector.redoRequested",
                "event_inspector.advanceAfterDecisionChanged",
                "event_gallery.advanceAfterDecisionChanged"):
        assert key in mw_mod.SURFACE_NEUTRAL_SIGNALS, key
    gui = pathlib.Path(mw_mod.__file__).parent
    for path in gui.glob("*.py"):
        if path.name == "main_window.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == "connect"
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr in ("redoRequested", "advanceAfterDecisionChanged")):
                raise AssertionError(f"{path.name} wires {node.value.attr}")
