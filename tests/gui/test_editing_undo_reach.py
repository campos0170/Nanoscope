"""Edit ▸ Undo reaches every window: design §3.7, defects 1, 2 and 3 of §1.2.

* **Defect 1: ⇧⌘Z did nothing in the main window on a Mac.** Edit ▸ Redo registered the
  standard Redo key AND "Ctrl+Shift+Z". On the cocoa platform the standard key resolves to
  Ctrl+Shift+Z (measured 2026-10-30), so one key was registered twice, and Qt answers that
  with "Ambiguous shortcut overload" and triggers nothing. Offscreen, Redo is Ctrl+Y, so no
  test saw it. The Mac's resolution is simulated here by resolving ``REDO_KEYS`` twice to
  Ctrl+Shift+Z, and offscreen a key registered twice is exactly as dead (measured: 0 of 1
  presses fired; registered once, 1 of 1).
* **Defect 2: after a Table-selector switch, Undo kept the previous table's state.**
  ``_on_result_switched`` never refreshed it, so a table WITH history came back under a
  disabled Undo, and ⌘Z on a disabled action does nothing.
* **Defect 3: ⌘Z did nothing in the Signal Conditioning window**, where the chain is edited.
  It forwarded ⌘S and ⇧⌘S to the main window but not Undo and Redo.
* **The Event gallery had an Undo and no Redo**, so an undo taken there had no way forward
  from where it was taken.

Real key presses through ``QTest`` on the active window, a real MainWindow, and a synthetic
recording whose ``meta.path`` is under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import key_sequences
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1_000.0
S, C = "detect.schmitt", "detect.cusum"
#: Ten events, alternately short (8-10 ms) and long (16-20 ms): a 10 ms minimum dwell keeps
#: the long five, so the two tables differ.
_SPANS = []
_pos = 6_000
for _i in range(10):
    _SPANS.append((_pos, _pos + (400 + 10 * _i if _i % 2 == 0 else 800 + 20 * _i)))
    _pos += 10_000
N = _SPANS[-1][1] + 6_000


def _signal() -> np.ndarray:
    rng = np.random.default_rng(8)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 7.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def qt_messages():
    seen: list[str] = []
    old = QtCore.qInstallMessageHandler(lambda _mode, _ctx, msg: seen.append(str(msg)))
    try:
        yield seen
    finally:
        QtCore.qInstallMessageHandler(old)


@pytest.fixture
def make_window(qapp, qtbot, tmp_path):
    """A shown MainWindow on one recording with a Schmitt table (and a CUSUM table of the
    long five, when asked), built AFTER the test has had its chance to patch the keys:
    MainWindow resolves them once, at construction."""
    made: list[MainWindow] = []

    def make(*, cusum: bool = False) -> tuple[MainWindow, str]:
        w = MainWindow()
        qtbot.addWidget(w)
        made.append(w)
        w.resize(1512, 900)
        w.show()
        data = _signal()
        info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
        meta = RecordingMeta(path=tmp_path / "reach.dat", format_name="synthetic",
                             n_samples=N, sample_rate_hz=FS,
                             rate_provenance=Provenance.FILE_META, channels=(info,))
        doc = TraceDocument(meta=meta, channels=(data,))
        doc_id = w._session.add_document(doc)
        w._create_default_layers(doc)
        w._on_selection(doc_id, 0)
        cal = w._session.calibration(doc_id, 0).fingerprint()
        for module_id, min_dwell in [(S, 1e-4)] + ([(C, 1e-2)] if cusum else []):
            frame, prov = detect_events(data, FS, baseline=I0, min_dwell_s=min_dwell)
            w._session.add_result(AnalysisResult(
                module_id=module_id, module_version="1.0", doc_id=doc_id, channel=0,
                params={"min_dwell_s": min_dwell}, calibration_fingerprint=cal,
                payload=EventSet(frame, prov)))
            qtbot.wait(10)
        if cusum:
            _show_table(w, C)
        _show_table(w, S)
        return w, doc_id

    yield make
    for w in made:
        w._analysis_controller.abort()


def _show_table(w: MainWindow, module_id: str) -> None:
    """What the user does: pick the table in the Results dock's Table selector."""
    selector = w._analysis_panel._result_selector
    selector.setCurrentIndex(selector.findData(module_id))
    QTest.qWait(10)
    assert w._analysis_panel.current_detection().module_id == module_id


def _frame(w: MainWindow):
    return w._analysis_panel.current_detection().payload.df


def _accepted(w: MainWindow, event_id: int) -> bool:
    df = _frame(w)
    return bool(df.loc[df["event_id"] == event_id, "accepted"].iloc[0])


def _reject(w: MainWindow, event_id: int) -> None:
    """Through the events table, the real curation path."""
    assert w._analysis_panel._table.set_accepted_by_id(event_id, False)
    QTest.qWait(5)
    assert not _accepted(w, event_id)


def _press(window: QtWidgets.QWidget, target: QtWidgets.QWidget, key, mods) -> None:
    """A real key press on ``target`` with ``window`` the active window: Qt's shortcut map
    decides who answers it, as it does for a user."""
    window.activateWindow()
    QTest.qWait(10)
    assert QtWidgets.QApplication.activeWindow() is window, "the window never became active"
    target.setFocus(Qt.OtherFocusReason)
    QTest.qWait(5)
    QTest.keyClick(target, key, mods)
    QTest.qWait(20)


_CMD = Qt.ControlModifier                       # ⌘ on a Mac: Qt maps it to Control
_SHIFT_CMD = Qt.ControlModifier | Qt.ShiftModifier


# ================================================================================================
# defect 1 -- each resolved key once
# ================================================================================================

def test_the_shared_helper_registers_each_resolved_key_once(qapp):
    """The one rule, fed the Mac's resolution: the standard Redo and "Ctrl+Shift+Z" are then
    ONE key, and a list of keys keeps it once, in order; an unbound key is dropped."""
    mac_redo = QtGui.QKeySequence("Ctrl+Shift+Z")
    assert [k.toString() for k in key_sequences.unique_sequences(mac_redo, "Ctrl+Shift+Z")] \
        == ["Ctrl+Shift+Z"]
    assert [k.toString() for k in key_sequences.unique_sequences(
        "Ctrl+Y", mac_redo, "Ctrl+Shift+Z", QtGui.QKeySequence())] == ["Ctrl+Y", "Ctrl+Shift+Z"]
    for keys in (key_sequences.undo_sequences(), key_sequences.redo_sequences()):
        assert keys and len({k.toString() for k in keys}) == len(keys), keys


def test_a_redo_key_that_resolves_twice_is_registered_once_and_shift_cmd_z_redoes(
        make_window, monkeypatch, qt_messages):
    """THE DEFECT, on the Mac's resolution: Redo's keys resolve twice to Ctrl+Shift+Z. The
    main window registers it once, and a real ⇧⌘Z on the active main window redoes."""
    monkeypatch.setattr(key_sequences, "REDO_KEYS",
                        (QtGui.QKeySequence("Ctrl+Shift+Z"), "Ctrl+Shift+Z"))
    w, _doc_id = make_window()
    assert [k.toString() for k in w._act_redo.shortcuts()] == ["Ctrl+Shift+Z"], (
        "Edit ▸ Redo registered one key twice: Qt calls that ambiguous and fires nothing")

    event_id = int(_frame(w)["event_id"].iloc[2])
    _reject(w, event_id)
    w._act_undo.trigger()
    assert _accepted(w, event_id)
    assert w._act_redo.isEnabled(), w._act_redo.text()

    _press(w, w._trace_view, Qt.Key_Z, _SHIFT_CMD)
    assert not _accepted(w, event_id), "⇧⌘Z on the main window did not redo the rejection"
    assert not [m for m in qt_messages if "Ambiguous" in m], qt_messages


def test_cmd_z_on_the_main_window_still_undoes(make_window, qt_messages):
    """CONTROL: the Undo key, through the same helper, on the platform's own resolution."""
    w, _doc_id = make_window()
    event_id = int(_frame(w)["event_id"].iloc[1])
    _reject(w, event_id)
    _press(w, w._trace_view, Qt.Key_Z, _CMD)
    assert _accepted(w, event_id)
    assert not [m for m in qt_messages if "Ambiguous" in m], qt_messages


# ================================================================================================
# defect 2 -- Undo follows the Table selector
# ================================================================================================

def test_undo_follows_a_table_selector_switch_and_cmd_z_works_on_return(make_window):
    """Reject on Schmitt, look at CUSUM, come back. Undo must name CUSUM's state while CUSUM
    is shown, and Schmitt's again on the way back -- without the Edit menu being opened in
    between, which is the only thing that used to refresh it."""
    w, _doc_id = make_window(cusum=True)
    event_id = int(_frame(w)["event_id"].iloc[4])
    _reject(w, event_id)
    assert w._act_undo.isEnabled()

    _show_table(w, C)
    assert not w._act_undo.isEnabled(), (
        f"CUSUM has no history, yet Undo still offers Schmitt's step: {w._act_undo.text()!r}")
    assert w._act_undo.text() == (
        "Undo — nothing on this table; the threshold detector (1 step) on this channel")

    _show_table(w, S)
    assert w._act_undo.isEnabled(), (
        f"back on the table with history, Undo reads {w._act_undo.text()!r}, disabled")
    assert w._act_undo.text() == "Undo Curate event"
    _press(w, w._trace_view, Qt.Key_Z, _CMD)
    assert _accepted(w, event_id), "⌘Z on the table with history did nothing"


# ================================================================================================
# defect 3 -- ⌘Z and ⇧⌘Z in the Signal Conditioning window
# ================================================================================================

def test_cmd_z_and_shift_cmd_z_in_the_signal_conditioning_window_undo_and_redo(
        make_window, qt_messages):
    w, _doc_id = make_window()
    event_id = int(_frame(w)["event_id"].iloc[3])
    _reject(w, event_id)
    w._open_signal_conditioning()
    ws = w._signal_conditioning
    assert ws.isWindow() and ws.window() is not w

    _press(ws, ws.stage_list, Qt.Key_Z, _CMD)
    assert _accepted(w, event_id), "⌘Z in the Signal Conditioning window undid nothing"
    _press(ws, ws.stage_list, Qt.Key_Z, _SHIFT_CMD)
    assert not _accepted(w, event_id), "⇧⌘Z in the Signal Conditioning window redid nothing"
    assert not [m for m in qt_messages if "Ambiguous" in m], qt_messages


def test_a_focused_spin_box_in_the_signal_conditioning_window_keeps_cmd_z(make_window):
    """CONTROL: the text field the user is typing in keeps ⌘Z for its typing, as in the main
    window -- Qt asks it first -- and the table is not touched."""
    w, _doc_id = make_window()
    event_id = int(_frame(w)["event_id"].iloc[3])
    _reject(w, event_id)
    w._open_signal_conditioning()
    ws = w._signal_conditioning
    spins = [s for s in ws.findChildren(QtWidgets.QAbstractSpinBox)
             if s.isVisible() and s.isEnabled()]
    assert spins, "a chain setting to type into"
    spin = spins[0]
    before = spin.text()
    ws.activateWindow()
    QTest.qWait(10)
    spin.setFocus(Qt.OtherFocusReason)
    spin.selectAll()
    QTest.keyClicks(spin, "7")
    assert spin.text() != before
    QTest.keyClick(spin, Qt.Key_Z, _CMD)
    QTest.qWait(20)
    assert spin.text() == before, "⌘Z in the spin box did not take back the typing"
    assert not _accepted(w, event_id), "⌘Z typed in a spin box undid the table instead"


# ================================================================================================
# the Event gallery's Redo
# ================================================================================================

def _open_gallery(w: MainWindow):
    w._act_event_gallery.trigger()
    QTest.qWait(20)
    gal = w._event_gallery
    gal.set_filter("All")
    gal.activateWindow()
    QTest.qWait(20)
    return gal


def _layout_holding(layout: QtWidgets.QLayout, widget: QtWidgets.QWidget):
    if layout.indexOf(widget) >= 0:
        return layout
    for i in range(layout.count()):
        child = layout.itemAt(i).layout()
        found = _layout_holding(child, widget) if child is not None else None
        if found is not None:
            return found
    return None


def test_the_gallery_has_a_redo_beside_its_undo_that_mirrors_edit_redo(make_window):
    w, _doc_id = make_window()
    gal = _open_gallery(w)
    foot = _layout_holding(gal.undo_btn.parentWidget().layout(), gal.undo_btn)
    assert foot is not None and foot.indexOf(gal.redo_btn) == foot.indexOf(gal.undo_btn) + 1, (
        "Redo is not beside Undo in the gallery's footer")
    assert not gal.redo_btn.isEnabled() and not gal.act_redo.isEnabled()
    assert any(k == QtGui.QKeySequence("Ctrl+Shift+Z") for k in gal.act_redo.shortcuts())

    event_id = int(_frame(w)["event_id"].iloc[2])
    _reject(w, event_id)
    w._act_undo.trigger()
    assert _accepted(w, event_id)
    assert gal.redo_btn.isEnabled() and gal.act_redo.isEnabled()
    assert gal.redo_btn.text() == "Redo", "the footer button's text must never change"
    assert w._act_redo.text() in gal.redo_btn.toolTip(), gal.redo_btn.toolTip()

    QTest.mouseClick(gal.redo_btn, Qt.LeftButton)
    QTest.qWait(20)
    assert not _accepted(w, event_id), "the gallery's Redo button redid nothing"
    assert not gal.redo_btn.isEnabled(), "Redo stayed enabled with nothing left to redo"


def test_shift_cmd_z_in_the_gallery_redoes_what_cmd_z_there_undid(make_window, qt_messages):
    w, _doc_id = make_window()
    gal = _open_gallery(w)
    event_id = int(_frame(w)["event_id"].iloc[0])
    _reject(w, event_id)

    _press(gal, gal.view, Qt.Key_Z, _CMD)
    assert _accepted(w, event_id)
    _press(gal, gal.view, Qt.Key_Z, _SHIFT_CMD)
    assert not _accepted(w, event_id), "⇧⌘Z in the gallery redid nothing"
    assert not [m for m in qt_messages if "Ambiguous" in m], qt_messages
