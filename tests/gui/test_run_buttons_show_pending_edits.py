"""Run buttons show the edits they have not run; Ctrl+Return runs the panel you are in.

The owner's two rules (USER_GUIDE §1): SETTINGS take effect on the panel's run button, and a
● on that button means there are edits it has not run; CURATION is immediate and undoable.

What is pinned here, through a real MainWindow:

* Detect reads "Detect ●" exactly while the chain judge -- the verdict behind the event
  table's "⚠ Chain changed" / "⚠ Settings changed" chip -- says the form differs from the run
  of the table on screen, and plain "Detect" with no table, with an ML or Manual table, and
  while the run gate is closed. Each route that moves the chip moves the mark: a form edit,
  Revert, a run landing, Edit ▸ Undo, a recording switch. The Preprocess workspace's
  "Run detection…" follows the same verdict.
* Calibration's Apply reads "Apply ●" while Apply would change the channel's committed
  calibration, and plain after Apply, after Reset, after Undo and while the typed value is
  refused.
* Ctrl/⌘+Return or Ctrl/⌘+keypad Enter presses Apply in the Calibration tab and Detect in
  the Preprocess / Detection tabs and the Preprocess workspace, after committing what is
  typed in the focused field (a table's cell editor included); nowhere else; never
  ambiguous; never while the button is greyed out, when the key goes on to the focused
  widget as before; a plain keypad Enter presses nothing new; the Explorer and the
  workspace's spectral-lines list keep their own.

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``. Keys are real key
presses sent through the window's QWindow, after the window is active.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
from dataclasses import replace

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectSchmitt, register_all
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.inspector_widgets import RUN_KEYS
from nanoscope.gui.main_window import MainWindow

from . import p1_hosting as H

FS = 50_000.0
I0 = -1_000.0
N = 60_000
_SPANS = [(5_000, 5_400), (15_000, 15_500), (25_000, 25_600), (40_000, 40_500)]
DOT = "Detect ●"
DETECT_TIP = "The form differs from the table on screen; Detect runs it."
CTRL = Qt.KeyboardModifier.ControlModifier
KEYPAD = Qt.KeyboardModifier.KeypadModifier


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    register_all()
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 875)
    w.show()
    qtbot.waitExposed(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def qt_messages():
    seen: list[str] = []

    def handler(_mode, _context, message):
        seen.append(str(message))

    old = QtCore.qInstallMessageHandler(handler)
    try:
        yield seen
    finally:
        QtCore.qInstallMessageHandler(old)


def _open(w: MainWindow, tmp_path, name: str = "pore.dat", *, detect: bool = True,
          seed: int = 0) -> str:
    """A synthetic one-channel recording on screen, with a Schmitt table when ``detect``."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(seed),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    if detect:
        session = w._session
        session.add_result(DetectSchmitt().run(SimpleAnalysisContext(
            session.calibrated(doc_id, 0), session.document(doc_id).meta,
            w._analysis_panel._form_params())))
    _settle(w)
    return doc_id


def _settle(w: MainWindow):
    """Judge now: the judge runs on a zero-delay timer after each edit."""
    QtWidgets.QApplication.processEvents()
    status = w._analysis_panel.refresh_chain_status()
    QtWidgets.QApplication.processEvents()
    return status


def _det(w: MainWindow):
    return w._analysis_panel.detection_panel


def _detect(w: MainWindow) -> str:
    return _det(w).detect_button.text()


def _chip(w: MainWindow) -> str:
    return w._analysis_panel._table.notice_text()


def _agrees(w: MainWindow) -> None:
    """The mark and the chip say the same thing (the gate is open in these tests)."""
    assert (_detect(w) == DOT) == bool(_chip(w)), (_detect(w), _chip(w))


def _bump_k_in(w: MainWindow) -> None:
    det = _det(w)
    det.k_in.setValue(det.k_in.value() + 1.0)


def _show_table(w: MainWindow, module_id: str) -> None:
    selector = w._analysis_panel._result_selector
    selector.setCurrentIndex(selector.findData(module_id))
    QtWidgets.QApplication.processEvents()
    assert w._analysis_panel.current_detection().module_id == module_id


def _recorder(w: MainWindow) -> list[dict]:
    """Every run the Detect path would start, recorded instead of run."""
    runs: list[dict] = []
    w._analysis_panel._run_detection = lambda params: runs.append(dict(params)) or True
    return runs


def _clicks(button) -> list[bool]:
    got: list[bool] = []
    button.clicked.connect(lambda *_: got.append(True))
    return got


def _focus(qtbot, widget) -> None:
    win = widget.window()
    win.raise_()
    win.activateWindow()
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is win, timeout=3000)
    widget.setFocus(Qt.OtherFocusReason)
    qtbot.waitUntil(lambda: QtWidgets.QApplication.focusWidget() is widget, timeout=3000)


def _key(widget, key=Qt.Key.Key_Return, modifiers=CTRL) -> None:
    """A real key press through the QWindow of ``widget``'s window, as a user's would be."""
    QTest.keyClick(H.window_handle(widget), key, modifiers)
    QtWidgets.QApplication.processEvents()


def _type(qtbot, field, text: str) -> None:
    _focus(qtbot, field)
    field.selectAll()
    QTest.keyClicks(field, text)
    QtWidgets.QApplication.processEvents()


def _run_key_shortcuts(w: MainWindow) -> list:
    out = list(_det(w).run_key_shortcuts) + list(w._cal_run_key_shortcuts)
    if w._signal_conditioning is not None:
        out += list(w._signal_conditioning.run_key_shortcuts)
    return out


# ================================================================================================
# A. Detect shows when the form has un-run edits
# ================================================================================================

def test_detect_is_plain_with_no_table_whatever_the_form_says(window, tmp_path):
    _open(window, tmp_path, detect=False)
    assert window._analysis_panel.chain_status().kind == "none"
    assert _detect(window) == "Detect"
    _bump_k_in(window)
    _settle(window)
    assert _detect(window) == "Detect" and _chip(window) == ""
    assert _det(window).detect_button.toolTip() == ""


def test_an_edit_marks_detect_and_revert_takes_the_mark_away(window, tmp_path):
    _open(window, tmp_path)
    ap, det = window._analysis_panel, _det(window)
    assert ap.chain_status().kind == "match"
    assert _detect(window) == "Detect" and det.detect_button.toolTip() == ""

    _bump_k_in(window)
    _settle(window)
    assert _chip(window) == "⚠ Settings changed"
    assert _detect(window) == DOT
    assert det.detect_button.toolTip() == DETECT_TIP
    assert det.detect_button.accessibleName() == "Detect"

    assert ap.revert_chain_to_shown_detection()
    _settle(window)
    assert _chip(window) == "" and _detect(window) == "Detect"
    assert det.detect_button.toolTip() == ""

    # A chain edit marks it too ("⚠ Chain changed").
    det.set_params({"filter_cutoff_hz": 5_000.0})
    _settle(window)
    assert _chip(window) == "⚠ Chain changed" and _detect(window) == DOT


def test_the_mark_waits_for_the_judge_and_never_runs_ahead_of_the_chip(window, tmp_path):
    """The mark is pushed with the chip from the one verdict; the event loop moves both."""
    _open(window, tmp_path)
    _bump_k_in(window)
    QtWidgets.QApplication.processEvents()          # the judge's zero-delay timer, no more
    _agrees(window)
    assert _detect(window) == DOT


def test_a_closed_gate_shows_plain_detect_and_reopening_it_restores_the_mark(window, tmp_path):
    _open(window, tmp_path)
    det = _det(window)
    _bump_k_in(window)
    _settle(window)
    assert _detect(window) == DOT
    det.set_runnable(False, "Select a CURRENT channel to run detection.")
    assert not det.detect_button.isEnabled()
    assert _detect(window) == "Detect" and det.detect_button.toolTip() == ""
    det.set_runnable(True)
    assert _detect(window) == DOT and det.detect_button.toolTip() == DETECT_TIP


def test_a_run_landing_takes_the_mark_away(window, tmp_path, qtbot):
    _open(window, tmp_path)
    _bump_k_in(window)
    _settle(window)
    assert _detect(window) == DOT
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        _det(window).detect_button.click()
    qtbot.waitUntil(lambda: _detect(window) == "Detect", timeout=5_000)
    _settle(window)
    assert window._analysis_panel.chain_status().kind == "match"
    assert _chip(window) == "" and _detect(window) == "Detect"


def test_undo_of_the_edit_takes_the_mark_away_and_redo_puts_it_back(window, tmp_path):
    _open(window, tmp_path)
    _bump_k_in(window)
    window._on_edit_menu_about_to_show()             # commits the edit as one step
    _settle(window)
    assert _detect(window) == DOT
    assert window._act_undo.text().startswith("Undo Change k_in")
    window._act_undo.trigger()
    _settle(window)
    assert _chip(window) == "" and _detect(window) == "Detect"
    window._act_redo.trigger()
    _settle(window)
    assert _chip(window) == "⚠ Settings changed" and _detect(window) == DOT


@pytest.mark.parametrize("module_id", ["detect.keras", "detect.manual"])
def test_an_ml_or_manual_table_on_screen_keeps_detect_plain(window, tmp_path, module_id):
    doc_id = _open(window, tmp_path)
    ap = window._analysis_panel
    schmitt = ap.current_detection()
    window._session.add_result(replace(schmitt, module_id=module_id, params={}))
    QtWidgets.QApplication.processEvents()
    _show_table(window, module_id)
    _bump_k_in(window)
    status = _settle(window)
    assert status.kind == "not_classical", status.sentence
    assert _chip(window) == "" and _detect(window) == "Detect"
    # Back on the Schmitt table the mark follows the verdict again.
    _show_table(window, "detect.schmitt")
    _bump_k_in(window)
    _settle(window)
    assert _detect(window) == DOT
    _agrees(window)
    assert doc_id


def test_a_recording_whose_form_matches_its_table_shows_plain_detect(window, tmp_path):
    a = _open(window, tmp_path, "a.dat", seed=0)
    _bump_k_in(window)
    _settle(window)
    assert _detect(window) == DOT
    b = _open(window, tmp_path, "b.dat", seed=1)
    assert window._analysis_panel.chain_status().kind == "match"
    assert _detect(window) == "Detect" and _chip(window) == ""
    window._on_selection(a, 0)
    _settle(window)
    _agrees(window)
    window._on_selection(b, 0)
    _settle(window)
    assert _detect(window) == "Detect"


def test_the_preprocess_workspace_run_button_follows_the_same_verdict(window, tmp_path):
    _open(window, tmp_path)
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    _settle(window)
    assert ws.run_button.isEnabled()
    assert ws.run_button.text() == "Run detection…"
    width = ws.run_button.width()
    _bump_k_in(window)
    _settle(window)
    assert ws.run_button.text() == "Run detection… ●"
    assert ws.run_button.toolTip().startswith(
        "The form differs from the table on screen; Run detection runs it.\n\n"
        "Analysis ▸ Run detection with the form's chain")
    assert ws.run_button.width() == width, "the mark never moves the status row"
    ws.set_run_enabled(False)
    assert ws.run_button.text() == "Run detection…"
    window._sync_detection_action()
    assert ws.run_button.text() == "Run detection… ●"
    assert window._analysis_panel.revert_chain_to_shown_detection()
    _settle(window)
    assert ws.run_button.text() == "Run detection…"
    assert not ws.run_button.toolTip().startswith("The form differs")


# ================================================================================================
# B. Calibration Apply shows when there is an edit to apply
# ================================================================================================

def _calibration(w: MainWindow):
    panel = w._calibration_panel
    panel._confirm_sidecar_removal = lambda *a, **k: True
    panel._confirm_note_loss = lambda *a, **k: True
    return panel


def test_apply_is_marked_while_an_edit_waits_for_it(window, tmp_path):
    doc_id = _open(window, tmp_path, detect=False)
    panel = _calibration(window)
    apply = panel._btn_apply
    assert apply.isEnabled(), "an unchanged committable value: Apply is enabled..."
    assert apply.text() == "Apply" and apply.toolTip() == "", "...and has nothing to apply"
    assert apply.accessibleName() == "Apply"

    panel._gain.setText("2")
    assert apply.text() == "Apply ●" and apply.toolTip() == "There is an edit to apply."
    panel._gain.setText("1.0")                       # the committed value, spelled otherwise
    assert apply.text() == "Apply"
    panel._gain.setText("abc")                       # refused: Apply is disabled
    assert not apply.isEnabled() and apply.text() == "Apply" and apply.toolTip() == ""

    panel._gain.setText("2")
    assert apply.text() == "Apply ●"
    apply.click()
    assert window._session.calibration(doc_id, 0).gain == 2.0
    assert apply.isEnabled() and apply.text() == "Apply" and apply.toolTip() == ""

    panel._offset.setText("5")
    assert apply.text() == "Apply ●"
    panel._btn_reset.click()
    assert window._session.calibration(doc_id, 0).gain == 1.0
    assert apply.text() == "Apply"

    # "Treat this channel as" waits for Apply like a typed number does.
    index = panel._treat_as.findData(ChannelKind.VOLTAGE)
    panel._treat_as.setCurrentIndex(index)
    assert apply.text() == "Apply ●"
    panel._treat_as.setCurrentIndex(0)
    assert apply.text() == "Apply"

    # Undo puts the committed calibration back and the form with it: nothing to apply.
    panel._gain.setText("3")
    apply.click()
    assert window._session.calibration(doc_id, 0).gain == 3.0
    panel._gain.setText("4")
    assert apply.text() == "Apply ●"
    window._act_undo.trigger()
    QtWidgets.QApplication.processEvents()
    assert window._session.calibration(doc_id, 0).gain == 1.0
    assert panel._gain.text() == "1"
    assert apply.text() == "Apply" and apply.toolTip() == ""


def test_apply_is_marked_by_what_apply_would_change_not_by_the_rounded_text(window, tmp_path):
    """The mark compares what Apply would build from the fields with the channel's committed
    calibration, not with the fields' rounded text."""
    doc_id = _open(window, tmp_path, detect=False)
    panel = _calibration(window)
    apply = panel._btn_apply
    session = window._session

    def commit(**kw):
        session.set_calibration(doc_id, 0, ChannelCalibration(
            provenance=Provenance.UNKNOWN, **kw))
        QtWidgets.QApplication.processEvents()

    # A committed 1/3 shows as "0.333333": Apply would store 0.333333, so it is pending...
    commit(gain=1.0 / 3.0)
    assert panel._gain.text() == "0.333333"
    assert apply.isEnabled() and apply.text() == "Apply ●"
    assert apply.toolTip() == "There is an edit to apply."
    # ...and the exact committed value typed back is not.
    panel._gain.setText(repr(1.0 / 3.0))
    assert apply.text() == "Apply" and apply.toolTip() == ""

    # Units are compared stripped, as Apply stores them: committed "nA " is pending (Apply
    # would store "nA")...
    commit(gain=2.0, target_units="nA ")
    assert panel._units.text() == "nA "
    assert apply.text() == "Apply ●"
    # ...and trailing spaces typed after a committed "nA" are not.
    commit(gain=2.0, target_units="nA")
    assert apply.text() == "Apply"
    panel._units.setText("nA  ")
    assert apply.text() == "Apply"
    panel._units.setText("pA")
    assert apply.text() == "Apply ●"

    # A refused value: Apply is disabled and plain.
    panel._gain.setText("0")
    assert not apply.isEnabled() and apply.text() == "Apply" and apply.toolTip() == ""


# ================================================================================================
# C. Ctrl+Return presses the run button of the panel you are in
# ================================================================================================

def test_ctrl_return_in_a_calibration_field_applies_what_is_typed(window, tmp_path, qtbot):
    doc_id = _open(window, tmp_path, detect=False)
    panel = _calibration(window)
    window._inspector_tabs.setCurrentWidget(window._cal_scroll)
    commits: list = []
    panel.commitStarting.connect(lambda *a: commits.append(a))
    runs = _recorder(window)
    amb: list = []
    for shortcut in _run_key_shortcuts(window):
        shortcut.activatedAmbiguously.connect(lambda: amb.append(True))

    fired: list = []
    for shortcut in window._cal_run_key_shortcuts:
        shortcut.activated.connect(lambda: fired.append(True))

    _type(qtbot, panel._gain, "2.5")
    _key(panel._gain)
    assert window._session.calibration(doc_id, 0).gain == 2.5
    assert fired == [True], "the Calibration tab's run key took it"
    assert len(commits) == 1, "one Apply, not the key and the field's Return both"
    assert QtWidgets.QApplication.focusWidget() is panel._gain, "the field keeps the keyboard"
    assert panel._btn_apply.text() == "Apply"

    _type(qtbot, panel._offset, "4")
    _key(panel._offset, Qt.Key.Key_Enter, CTRL | Qt.KeyboardModifier.KeypadModifier)
    assert window._session.calibration(doc_id, 0).offset == 4.0
    assert len(commits) == 2

    # A refused value (an empty gain): Apply is disabled and the key does nothing.
    _focus(qtbot, panel._gain)
    panel._gain.selectAll()
    QTest.keyClick(panel._gain, Qt.Key.Key_Backspace)
    assert panel._gain.text() == "" and not panel._btn_apply.isEnabled()
    _key(panel._gain)
    assert len(commits) == 2 and window._session.calibration(doc_id, 0).gain == 2.5
    assert runs == [] and amb == []


def test_ctrl_return_in_a_detection_field_runs_detect_with_the_typed_value(window, tmp_path,
                                                                          qtbot, qt_messages):
    _open(window, tmp_path, detect=False)
    det = _det(window)
    window._reveal_detection_panel()
    det.show_detection_tab()
    det.set_detector("detect.cusum")
    QtWidgets.QApplication.processEvents()
    runs = _recorder(window)
    clicks = _clicks(det.detect_button)
    amb: list = []
    for shortcut in _run_key_shortcuts(window):
        shortcut.activatedAmbiguously.connect(lambda: amb.append(True))

    before = det.cusum_h.value()
    _type(qtbot, det.cusum_h, "12")
    assert det.cusum_h.value() == before, "typed, not yet committed"
    _key(det.cusum_h)
    assert len(runs) == 1 and len(clicks) == 1
    assert runs[0]["detector"] == "detect.cusum"
    assert runs[0]["cusum_h"] == pytest.approx(12.0), "the run used what the box showed"
    assert det.cusum_h.value() == pytest.approx(12.0)
    assert QtWidgets.QApplication.focusWidget() is det.cusum_h

    # The keypad's Enter, from a button on the same tab.
    null_runs: list = []
    det.nullControlRequested.connect(lambda *_: null_runs.append(True))
    _focus(qtbot, det.null_button)
    _key(det.null_button, Qt.Key.Key_Enter, CTRL | Qt.KeyboardModifier.KeypadModifier)
    assert len(runs) == 2 and null_runs == [], "Detect, not the focused button"

    # The same path as a click: the curation question is asked, and a No runs nothing.
    asked: list = []
    window._analysis_panel.confirm_replacing_curation = lambda module_id: (
        asked.append(module_id), False)[1]
    _key(det.cusum_h)
    assert asked == ["detect.cusum"] and len(runs) == 2

    # A closed gate: the key does nothing at all.
    det.set_runnable(False, "Select a CURRENT channel to run detection.")
    _key(det.cusum_h)
    assert len(asked) == 1 and len(runs) == 2
    assert amb == []
    assert not [m for m in qt_messages if "mbiguous" in m], qt_messages


def test_ctrl_return_in_the_preprocess_tab_and_workspace_presses_detect(window, tmp_path,
                                                                        qtbot):
    _open(window, tmp_path, detect=False)
    det = _det(window)
    runs = _recorder(window)
    window._reveal_detection_panel()
    det.show_preprocess_tab()
    QtWidgets.QApplication.processEvents()
    route = det.preprocess_route_buttons["filters"]
    _focus(qtbot, route)
    _key(route)
    assert len(runs) == 1

    window._open_signal_conditioning()
    ws = window._signal_conditioning
    det.filter_enabled.setChecked(True)
    det.select_stage("lowpass")
    QtWidgets.QApplication.processEvents()
    assert ws.isAncestorOf(det.filter_cutoff_hz), "the chain editor lives in the window"
    _type(qtbot, det.filter_cutoff_hz, "5000")
    _key(det.filter_cutoff_hz)
    assert len(runs) == 2
    assert runs[-1]["filter_cutoff_hz"] == pytest.approx(5_000.0)


def test_ctrl_return_does_nothing_in_the_ml_tab_the_event_table_or_the_trace(window, tmp_path,
                                                                            qtbot):
    doc_id = _open(window, tmp_path)
    ap, det = window._analysis_panel, _det(window)
    runs = _recorder(window)
    spies = [_clicks(det.detect_button), _clicks(det.ml_detect_button),
             _clicks(window._calibration_panel._btn_apply)]
    rows = len(ap.current_detection().payload.df)
    window._reveal_detection_panel()

    det.show_ml_tab()
    QtWidgets.QApplication.processEvents()
    _focus(qtbot, det.ml_range)
    assert QtWidgets.QApplication.focusWidget() is det.ml_range
    _key(det.ml_range)
    _key(det.ml_range, Qt.Key.Key_Enter, CTRL | Qt.KeyboardModifier.KeypadModifier)

    view = ap._table.view
    _focus(qtbot, view)
    assert QtWidgets.QApplication.focusWidget() is view
    _key(view)

    trace = window._trace_view
    _focus(qtbot, trace)
    assert QtWidgets.QApplication.focusWidget() is trace
    _key(trace)

    assert runs == [] and all(spy == [] for spy in spies)
    assert len(ap.current_detection().payload.df) == rows
    assert window._session.calibration(doc_id, 0).gain == 1.0


def test_the_run_keys_are_focus_scoped_and_never_ambiguous(window, tmp_path, qtbot):
    _open(window, tmp_path)
    window._open_signal_conditioning()
    shortcuts = _run_key_shortcuts(window)
    assert len(shortcuts) == 4 * len(RUN_KEYS)
    for shortcut in shortcuts:
        assert shortcut.context() == Qt.WidgetWithChildrenShortcut
        assert shortcut.key().toString() in RUN_KEYS
    # No window- or application-wide owner of the key anywhere in the main window or the
    # Preprocess workspace: those would match alongside a scoped one and Qt would fire
    # neither.
    for top in (window, window._signal_conditioning):
        for action in top.findChildren(QtGui.QAction):
            assert not [k for k in action.shortcuts() if k.toString() in RUN_KEYS], action
        for shortcut in top.findChildren(QtGui.QShortcut):
            if shortcut.key().toString() in RUN_KEYS:
                assert shortcut in shortcuts, shortcut
    # The scopes are disjoint: none contains another.
    scopes = [s.parent() for s in shortcuts]
    for a in scopes:
        for b in scopes:
            assert a is b or not a.isAncestorOf(b), (a, b)


def test_the_explorers_ctrl_return_still_updates_its_plot(window, tmp_path, qtbot):
    _open(window, tmp_path, detect=False)
    runs = _recorder(window)
    page = H.SyntheticPlotPage("iv")
    explorer = H.hosted(qtbot, page, "iv")
    H.key(explorer, Qt.Key.Key_Return, H.CTRL)
    assert page.updates == 1 and runs == []

    det = _det(window)
    window._reveal_detection_panel()
    det.show_detection_tab()
    QtWidgets.QApplication.processEvents()
    _focus(qtbot, det.min_dwell_s)
    _key(det.min_dwell_s)
    assert len(runs) == 1 and page.updates == 1

    H.key(explorer, Qt.Key.Key_Enter, H.CTRL)
    assert page.updates == 2 and len(runs) == 1


# ================================================================================================
# D. The key around table cells, greyed buttons, blank space, the lines list and keypad Enter
# ================================================================================================

def _interval_table(w: MainWindow, *, valid: bool):
    """The open-pore reference's interval table with one row, complete when ``valid``."""
    ap, det = w._analysis_panel, _det(w)
    ref = ap._baseline.reference_editor
    ref.mode.setCurrentIndex(ref.mode.findData("intervals"))
    det.show_open_pore_baseline()
    QtWidgets.QApplication.processEvents()
    table = ref.table
    ref.add_button.click()
    QtWidgets.QApplication.processEvents()
    assert table.isVisible() and table.rowCount() == 1
    if valid:
        table.item(0, 0).setText("0")
        table.item(0, 1).setText("50")
        QtWidgets.QApplication.processEvents()
    return table


def _edit_cell(qtbot, table, col: int, text: str):
    """Open the cell editor of row 0, ``col`` and type ``text`` into it."""
    _focus(qtbot, table)
    table.setCurrentCell(0, col)
    table.editItem(table.item(0, col))
    QtWidgets.QApplication.processEvents()
    editor = QtWidgets.QApplication.focusWidget()
    assert isinstance(editor, QtWidgets.QLineEdit), editor
    assert table.viewport().isAncestorOf(editor), "a cell editor of the table"
    editor.selectAll()
    QTest.keyClicks(editor, text)
    QtWidgets.QApplication.processEvents()
    return editor


def _drain():
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    QtWidgets.QApplication.processEvents()


def test_ctrl_return_in_a_table_cell_commits_it_runs_detect_and_the_table_keeps_the_keys(
        window, tmp_path, qtbot):
    _open(window, tmp_path, detect=False)
    runs = _recorder(window)
    table = _interval_table(window, valid=True)
    assert _det(window).detect_button.isEnabled()
    editor = _edit_cell(qtbot, table, 1, "80")
    _key(editor)
    _drain()
    assert table.item(0, 1).text() == "80", "the cell committed"
    assert len(runs) == 1, "and Detect ran"
    assert np.allclose(runs[0]["baseline_open_pore_reference"]["intervals_s"], [[0.0, 0.08]])
    assert QtWidgets.QApplication.focusWidget() is table, "the table keeps the keyboard"
    QTest.keyClick(H.window_handle(table), Qt.Key.Key_Left)
    QtWidgets.QApplication.processEvents()
    assert (table.currentRow(), table.currentColumn()) == (0, 0)


def test_a_greyed_run_button_never_swallows_the_key(window, tmp_path, qtbot):
    """With Detect disabled its run keys are disabled too, and Ctrl+Return reaches the
    focused cell editor, which commits on it as it did before there was a run key."""
    _open(window, tmp_path, detect=False)
    runs = _recorder(window)
    det = _det(window)
    clicks = _clicks(det.detect_button)
    table = _interval_table(window, valid=False)       # an incomplete interval: gate closed
    ws = window._signal_conditioning
    detect_keys = list(det.run_key_shortcuts) + list(ws.run_key_shortcuts)
    gc.collect()                                       # the follower outlives a collection
    assert not det.detect_button.isEnabled()
    assert detect_keys and not any(s.isEnabled() for s in detect_keys)

    editor = _edit_cell(qtbot, table, 0, "80")
    _key(editor)
    _drain()
    assert table.item(0, 0).text() == "80", "the cell committed, as before"
    assert runs == [] and clicks == []
    assert QtWidgets.QApplication.focusWidget() is table

    # The keys follow the button both ways.
    table.item(0, 1).setText("100")
    QtWidgets.QApplication.processEvents()
    assert det.detect_button.isEnabled()
    assert all(s.isEnabled() for s in detect_keys)
    det.set_runnable(False, "Select a CURRENT channel to run detection.")
    assert not any(s.isEnabled() for s in detect_keys)
    det.set_runnable(True)
    assert all(s.isEnabled() for s in detect_keys)
    # And the Calibration tab's follows Apply.
    panel = _calibration(window)
    cal_keys = list(window._cal_run_key_shortcuts)
    assert panel._btn_apply.isEnabled() and all(s.isEnabled() for s in cal_keys)
    panel._gain.setText("abc")
    assert not panel._btn_apply.isEnabled() and not any(s.isEnabled() for s in cal_keys)
    panel._gain.setText("1")
    assert all(s.isEnabled() for s in cal_keys)


def test_ctrl_return_after_a_click_on_blank_space_in_the_calibration_tab_applies(
        window, tmp_path, qtbot):
    doc_id = _open(window, tmp_path, detect=False)
    panel = _calibration(window)
    scroll = window._cal_scroll
    window._inspector_tabs.setCurrentWidget(scroll)
    QtWidgets.QApplication.processEvents()

    # The tab has exactly one run key, on the tab page itself; none inside it.
    inside = [s for s in window.findChildren(QtGui.QShortcut)
              if s.key().toString() in RUN_KEYS and isinstance(s.parent(), QtWidgets.QWidget)
              and (s.parent() is scroll or scroll.isAncestorOf(s.parent()))]
    assert len(inside) == len(RUN_KEYS) == len(window._cal_run_key_shortcuts)
    assert all(s in window._cal_run_key_shortcuts and s.parent() is scroll for s in inside)

    commits: list = []
    panel.commitStarting.connect(lambda *a: commits.append(a))
    _type(qtbot, panel._gain, "3")
    assert panel._btn_apply.text() == "Apply ●"
    blank = QtCore.QPoint(panel.width() - 5, panel._title.geometry().bottom() + 2)
    assert panel.childAt(blank) is None, "blank space in the tab"
    H.click(panel, pos=blank)
    assert QtWidgets.QApplication.focusWidget() is scroll, QtWidgets.QApplication.focusWidget()
    assert window._session.calibration(doc_id, 0).gain == 1.0, "the click applied nothing"
    _key(scroll)
    assert window._session.calibration(doc_id, 0).gain == 3.0
    assert len(commits) == 1 and panel._btn_apply.text() == "Apply"


def test_the_spectral_lines_list_keeps_its_own_ctrl_return(window, tmp_path, qtbot):
    _open(window, tmp_path, detect=False)
    runs = _recorder(window)
    det = _det(window)
    clicks = _clicks(det.detect_button)
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    lines = ws.lines_list
    for i in range(ws.preview_tabs.count()):
        if ws.preview_tabs.widget(i).isAncestorOf(lines):
            ws.preview_tabs.setCurrentIndex(i)
    QtWidgets.QApplication.processEvents()
    lines.item(0).setHidden(False)
    lines.item(0).setText("50 Hz")
    lines.setCurrentRow(0)
    assert lines.isVisible() and det.detect_button.isEnabled()

    # The list claims the override for Ctrl+Return and Ctrl+keypad Enter, and only those.
    for key, mods, claimed in ((Qt.Key.Key_Return, CTRL, True),
                               (Qt.Key.Key_Enter, CTRL | KEYPAD, True),
                               (Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier, False),
                               (Qt.Key.Key_Return, CTRL | Qt.KeyboardModifier.ShiftModifier,
                                False)):
        override = QtGui.QKeyEvent(QtCore.QEvent.ShortcutOverride, key, mods)
        override.ignore()
        QtWidgets.QApplication.sendEvent(lines, override)
        assert override.isAccepted() is claimed, (key, mods)

    _focus(qtbot, lines)
    assert QtWidgets.QApplication.focusWidget() is lines
    _key(lines)
    _key(lines, Qt.Key.Key_Enter, CTRL | KEYPAD)
    assert runs == [] and clicks == [], "Detect is not pressed from the list"

    # Elsewhere in the window the key still presses Detect.
    _focus(qtbot, ws.stage_list)
    _key(ws.stage_list)
    assert len(runs) == 1


def test_a_plain_keypad_enter_presses_nothing_new(window, tmp_path, qtbot):
    doc_id = _open(window, tmp_path)
    runs = _recorder(window)
    det = _det(window)
    clicks = _clicks(det.detect_button)
    window._reveal_detection_panel()
    det.show_detection_tab()
    QtWidgets.QApplication.processEvents()
    assert det.detect_button.isEnabled()
    for widget in (det.k_in, det.null_button):
        _focus(qtbot, widget)
        assert QtWidgets.QApplication.focusWidget() is widget
        _key(widget, Qt.Key.Key_Enter, KEYPAD)
    assert runs == [] and clicks == []

    panel = _calibration(window)
    window._inspector_tabs.setCurrentWidget(window._cal_scroll)
    QtWidgets.QApplication.processEvents()
    commits: list = []
    panel.commitStarting.connect(lambda *a: commits.append(a))
    panel._gain.setText("2")
    _focus(qtbot, panel._treat_as)
    assert QtWidgets.QApplication.focusWidget() is panel._treat_as
    _key(panel._treat_as, Qt.Key.Key_Enter, KEYPAD)
    assert commits == [] and window._session.calibration(doc_id, 0).gain == 1.0
    # With Ctrl it is the run key.
    _key(panel._treat_as, Qt.Key.Key_Enter, CTRL | KEYPAD)
    assert len(commits) == 1 and window._session.calibration(doc_id, 0).gain == 2.0
