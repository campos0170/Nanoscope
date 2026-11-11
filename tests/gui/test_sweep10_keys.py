"""Sweep 10, group A (keys): ⌘Z after stepping a value, and the size spin boxes' first step.

* **S10-PLOT1.** The line edit inside every spin box and editable combo claims Undo and Redo
  (Qt's ``ShortcutOverride``) even with nothing of its own to undo. Stepping a spin box or
  picking a font leaves the control focused and its line edit's history empty, so ⌘Z and ⇧⌘Z
  reached no one: not a plot window's appearance history, not Edit ▸ Undo, not Signal
  Conditioning's forwarded Undo. The documented exception is only a field the user is TYPING
  in (USER_GUIDE §9b, design §3.7), and that stays pinned here as the control.
* **S10-PLOT2.** The size spin boxes ran 0 (Default), 1, 2 ... 5 pt, which ``from_mapping``
  drops, so the first steps showed a size while the figure and the history did nothing.

Keys are sent with ``QTest.keyClick`` to the FOCUSED control, so the real override path runs;
the existing history tests press the key on the window with nothing focused, which is why they
stayed green. Synthetic data only; every recording's ``meta.path`` is under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.gui.key_sequences import redo_sequences, undo_sequences
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget

from .test_editing_undo_reach import _accepted, _frame, _reject, make_window  # noqa: F401
from .test_plot_appearance_history import PAUSE_MS, _activate, _build


def _key(sequence) -> None:
    """One press of ``sequence`` on whatever holds the keyboard, as a user's press arrives."""
    combo = QtGui.QKeySequence(sequence)[0]
    QTest.keyClick(QtWidgets.QApplication.focusWidget(), combo.key(), combo.keyboardModifiers())
    QTest.qWait(20)


def _focus(qtbot, widget) -> None:
    widget.setFocus(Qt.OtherFocusReason)
    qtbot.waitUntil(widget.hasFocus, timeout=2000)


def _plot_window(qtbot, kind):
    window = _build(kind)
    qtbot.addWidget(window)
    window.resize(1200, 850)
    _activate(qtbot, window)
    if hasattr(window, "_workflow"):
        window._workflow.set_section("Appearance")
    QtWidgets.QApplication.processEvents()
    return window, window._appearance_widget


# ================================================================================================
# S10-PLOT1 -- a stepped spin box or a picked font lets ⌘Z / ⇧⌘Z through
# ================================================================================================

def test_the_guard_is_installed_by_the_apps_own_factory(qapp):
    """Every window the suite and the app build shares the one QApplication ``create_app``
    makes, and the guard is installed there, once."""
    from nanoscope.gui.app import _UndoKeyGuard, create_app

    guard = qapp._nanoscope_undo_key_guard
    assert isinstance(guard, _UndoKeyGuard)
    assert create_app([]) is qapp and qapp._nanoscope_undo_key_guard is guard


@pytest.mark.parametrize("kind", ["iv", "psd", "vdep", "event"])
def test_a_stepped_size_spin_box_lets_cmd_z_undo_and_redo_the_appearance(qtbot, kind):
    window, w = _plot_window(qtbot, kind)
    # 2026-11-11: a Noise or Voltage-dependence window records the axis quantity it presents in
    # its appearance state (8661d35, the quantity-display witness), so "nothing chosen yet" is
    # that witness alone, no longer {}. The steps are compared with the state before the step,
    # which also pins that undo and redo leave the witness exactly where it was.
    base = window.appearance_state()
    assert set(base) <= {"quantity_display"}, "no presentation choice has been made yet"
    spin = w.title_size
    _focus(qtbot, spin)
    QTest.keyClick(spin, Qt.Key_PageUp)               # one page: ten single steps
    qtbot.wait(PAUSE_MS)                              # the run closes: one step
    assert window.appearance_state() == {**base, "title_size": 10.0}
    assert QtWidgets.QApplication.focusWidget() is spin

    _key(undo_sequences()[0])
    assert window.appearance_state() == base, "⌘Z on a stepped spin box did not undo the step"
    assert spin.text() == "Default"
    for key in redo_sequences():                      # every Redo key this platform resolves
        assert QtWidgets.QApplication.focusWidget() is spin
        _key(key)
        assert window.appearance_state() == {**base, "title_size": 10.0}, key.toString()
        _key(undo_sequences()[0])
        assert window.appearance_state() == base


def test_the_spin_boxs_own_arrow_button_lets_cmd_z_through(qtbot):
    window, w = _plot_window(qtbot, "iv")
    spin = w.line_width
    opt = QtWidgets.QStyleOptionSpinBox()
    spin.initStyleOption(opt)
    up = spin.style().subControlRect(QtWidgets.QStyle.CC_SpinBox, opt,
                                     QtWidgets.QStyle.SC_SpinBoxUp, spin)
    for _ in range(2):
        QTest.mouseClick(spin, Qt.LeftButton, pos=up.center())
    qtbot.wait(PAUSE_MS)
    assert window.appearance_state() == {"line_width": 1.0}
    assert QtWidgets.QApplication.focusWidget() is spin
    _key(undo_sequences()[0])
    assert window.appearance_state() == {}


def test_a_font_picked_from_the_list_lets_cmd_z_through(qtbot):
    window, w = _plot_window(qtbot, "iv")
    combo = w.font_family
    _focus(qtbot, combo)
    combo.setCurrentIndex(3)                          # as a pick from the list does
    qtbot.wait(PAUSE_MS)
    picked = window.appearance_state()
    assert set(picked) == {"font_family"}
    _key(undo_sequences()[0])
    assert window.appearance_state() == {}, "⌘Z on a picked font did not undo the pick"
    assert combo.currentText() == "Arial (default)"


def test_a_spin_box_the_user_typed_in_keeps_cmd_z_for_its_typing(qtbot):
    """CONTROL: the documented exception. Typing leaves the line edit something to undo, so
    ⌘Z takes back the typing and the appearance history is not touched."""
    window, w = _plot_window(qtbot, "iv")
    spin = w.title_size
    _focus(qtbot, spin)
    spin.selectAll()
    QTest.keyClicks(spin, "12")
    qtbot.wait(PAUSE_MS)
    assert window.appearance_state() == {"title_size": 12.0}
    depth = w._history.depth()
    assert spin.lineEdit().isUndoAvailable()

    _key(undo_sequences()[0])
    assert spin.text() == "Default", "⌘Z did not take back what was typed"
    assert w._history.depth() == depth, "⌘Z typed in a spin box moved the appearance history"


def test_a_font_chosen_in_the_other_font_list_is_one_step_for_cmd_z(qtbot, monkeypatch):
    """The font box is a list now, not an editable line (P0 G2): a family outside the short
    list is chosen in the Other font… dialog, and that is ONE appearance step -- ⌘Z takes it
    back whole, and the box shows the short list's first row again."""
    from nanoscope.gui import font_picker

    window, w = _plot_window(qtbot, "iv")
    combo = w.font_family
    _focus(qtbot, combo)

    def pick_menlo(dialog):
        assert dialog.select_family("Menlo")
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(font_picker.FontListDialog, "exec", pick_menlo)
    combo.setCurrentIndex(combo.count() - 1)          # the Other font… row
    qtbot.waitUntil(lambda: window.appearance_state().get("font_family") == "Menlo")
    qtbot.wait(PAUSE_MS)
    depth = w._history.depth()
    assert depth[0] == 1 and combo.currentText() == "Menlo"
    _key(undo_sequences()[0])
    assert window.appearance_state() == {}, "⌘Z on an Other font… pick did not undo it"
    assert combo.currentText() == "Arial (default)"


def test_a_stepped_detection_spin_box_lets_cmd_z_reach_edit_undo(make_window):
    """The main window: step a Detection form setting with the arrow key, wait out the pause,
    press ⌘Z on the still-focused spin box. Edit ▸ Undo must fire and put the value back."""
    w, _doc_id = make_window()
    w.activateWindow()
    QTest.qWait(20)
    fired: list[int] = []
    w._act_undo.triggered.connect(lambda *_: fired.append(1))
    # 2026-11-11: the Analysis page opens on Preprocess, which since 8661d35 is the workspace
    # overview and holds no spin box. Raise Detection and step the named field.
    panel = w._analysis_panel.detection_panel
    panel.show_detection_tab()
    QTest.qWait(20)
    spin = panel.k_in
    assert spin.isVisible() and spin.isEnabled(), "a Detection setting to step"
    spin.setFocus(Qt.OtherFocusReason)
    QTest.qWait(10)
    before = spin.value()
    QTest.keyClick(spin, Qt.Key_Up)
    QTest.qWait(900)                                  # past the settings pause: one step
    assert spin.value() != before
    assert w._act_undo.isEnabled(), w._act_undo.text()
    assert QtWidgets.QApplication.focusWidget() is spin

    _key(undo_sequences()[0])
    assert fired, "⌘Z on a stepped Detection spin box never reached Edit ▸ Undo"
    assert spin.value() == before


def test_a_detection_spin_box_the_user_typed_in_keeps_cmd_z(make_window):
    """CONTROL in the main window: typing in the same box keeps ⌘Z for the typing."""
    w, _doc_id = make_window()
    w.activateWindow()
    QTest.qWait(20)
    fired: list[int] = []
    w._act_undo.triggered.connect(lambda *_: fired.append(1))
    event_id = int(_frame(w)["event_id"].iloc[1])
    _reject(w, event_id)                              # Edit ▸ Undo has a step to take
    # 2026-11-11: as above, the field is named; Preprocess holds no spin box since 8661d35.
    panel = w._analysis_panel.detection_panel
    panel.show_detection_tab()
    QTest.qWait(20)
    spin = panel.k_in
    assert spin.isVisible() and spin.isEnabled(), "a Detection setting to type in"
    spin.setFocus(Qt.OtherFocusReason)
    QTest.qWait(10)
    before = spin.text()
    spin.selectAll()
    QTest.keyClicks(spin, "7")
    assert spin.text() != before
    _key(undo_sequences()[0])
    assert spin.text() == before, "⌘Z did not take back what was typed"
    assert not fired and not _accepted(w, event_id), "⌘Z typed in a spin box undid the table"


def test_a_stepped_spin_box_in_signal_conditioning_lets_its_cmd_z_through(make_window):
    """Signal Conditioning forwards ⌘Z to the main window's Undo; a stepped spin box there
    must not keep it."""
    w, _doc_id = make_window()
    w._open_signal_conditioning()
    ws = w._signal_conditioning
    asked: list[str] = []
    ws.undoRequested.connect(lambda: asked.append("undo"))
    ws.activateWindow()
    QTest.qWait(20)
    spins = [s for s in ws.findChildren(QtWidgets.QAbstractSpinBox)
             if s.isVisible() and s.isEnabled()]
    assert spins, "a chain setting to step"
    spin = spins[0]
    spin.setFocus(Qt.OtherFocusReason)
    QTest.qWait(10)
    QTest.keyClick(spin, Qt.Key_Up)
    assert QtWidgets.QApplication.focusWidget() is spin
    _key(undo_sequences()[0])
    assert asked == ["undo"], "⌘Z on a stepped spin box in Signal Conditioning was kept"


# ================================================================================================
# S10-PLOT2 -- the first step up from Default is the smallest size a figure draws
# ================================================================================================

_FLOORS = {"title_size": 6.0, "axis_size": 6.0, "tick_size": 6.0, "legend_size": 6.0,
           "line_width": 0.5, "marker_size": 1.0}


@pytest.mark.parametrize("key", sorted(_FLOORS))
def test_one_step_up_from_default_is_a_size_the_figure_takes(qtbot, key):
    w = PlotAppearanceWidget()
    qtbot.addWidget(w)
    w.show()
    spin = getattr(w, key)
    _focus(qtbot, spin)
    QTest.keyClick(spin, Qt.Key_Up)
    assert spin.value() == _FLOORS[key], spin.text()
    assert getattr(w.state(), key) == _FLOORS[key], "the size shown is not the size applied"
    QTest.keyClick(spin, Qt.Key_Down)
    assert spin.text() == "Default" and getattr(w.state(), key) is None


def test_a_typed_size_below_the_floor_is_not_taken(qtbot):
    w = PlotAppearanceWidget()
    qtbot.addWidget(w)
    w.show()
    spin = w.tick_size
    _focus(qtbot, spin)
    spin.selectAll()
    QTest.keyClicks(spin, "3")
    assert w.state() == PlotAppearance()
    QTest.keyClick(spin, Qt.Key_Return)
    assert spin.text() == "Default", "a size the figure ignores stayed on screen"
    spin.selectAll()
    QTest.keyClicks(spin, "12")                       # "1" on the way is not a size either
    assert w.state() == PlotAppearance(tick_size=12)


def test_old_saved_sizes_below_the_floor_still_read_as_default(qtbot):
    assert PlotAppearance.from_mapping(
        {"title_size": 3, "tick_size": 5.5, "legend_size": 1, "marker_size": .5,
         "line_width": .1, "axis_size": 6}).to_dict() == {"axis_size": 6.0}
    w = PlotAppearanceWidget()
    qtbot.addWidget(w)
    w.set_state({"title_size": 3, "axis_size": 6, "legend_size": 30})
    assert w.title_size.text() == "Default"
    assert w.axis_size.value() == 6 and w.legend_size.value() == 30
    assert w.state() == PlotAppearance(axis_size=6, legend_size=30)
