"""Change C -- every key reaches exactly one owner, in both of the inspector's hosts.

Docked, the inspector is inside the main window, so its Ctrl+S and the main window's
Ctrl+S would both match and Qt would fire NEITHER (an ambiguous pair). Docked, it gives
those keys (Save, Save As, Undo, Redo) to the main window's own actions; every curation
key stays its own, in both hosts, with any of its own widgets focused -- except the one key
a focused button uses itself: Space presses it (S8-KEYS2; see
test_s8_keys2_curation_keys_yield_to_the_focused_control). A click in the
docked inspector takes the keyboard, a click elsewhere never sends keys to it.
"""

from __future__ import annotations

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    accepted, build_window, close_window, docked, private_settings, settle)

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.labels import LabelVocabulary
from nanoscope.gui.main_window import MainWindow


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


def _ids(window) -> list[int]:
    return [int(i) for i in window._analysis_panel.current_detection().payload.df["event_id"]]


def _flag(window, event_id: int) -> str:
    df = window._analysis_panel.current_detection().payload.df
    return str(df.loc[df["event_id"] == event_id, "flag"].iloc[0])


def _open(qtbot, host: str):
    window = build_window(qtbot)
    vocab = LabelVocabulary()
    vocab.add("type", "translocation")
    window._label_vocab = vocab
    window._push_vocabulary()
    if host == "dock":
        insp, _dock = docked(window)
    else:
        window._open_event_inspector()
        insp = window._event_inspector
        settle()
    return window, insp


def _press(insp, target, key, mods=Qt.NoModifier) -> None:
    (insp if insp.isWindow() else insp.window()).activateWindow()
    target.setFocus(Qt.OtherFocusReason)
    settle(5)
    QTest.keyClick(target, key, mods)
    settle(5)


@pytest.mark.parametrize("host", ["window", "dock"])
def test_every_curation_key_reaches_the_inspector_whatever_of_its_own_has_focus(
        private_settings, qtbot, qt_messages, host):
    window, insp = _open(qtbot, host)
    ids = _ids(window)
    traced: list[int] = []
    insp.traceRequested.connect(traced.append)
    for target in (insp._plot_widget, insp._filter_combo, insp._context_combo, insp._next_btn):
        name = target.objectName() or type(target).__name__
        insp.show_event(ids[2])
        _press(insp, target, Qt.Key_R)
        assert not accepted(window, ids[2]), name
        _press(insp, target, Qt.Key_A)
        assert accepted(window, ids[2]), name
        _press(insp, target, Qt.Key_Space)
        if isinstance(target, QtWidgets.QAbstractButton):
            # A focused button is pressed by Space: "next" moves on and the decision stays.
            assert accepted(window, ids[2]) and insp.current_event_id() == ids[3], name
            insp.show_event(ids[2])
        else:
            assert not accepted(window, ids[2]), name
        _press(insp, target, Qt.Key_1)
        assert "type=translocation" in _flag(window, ids[2]), name
        _press(insp, target, Qt.Key_T)
        assert traced[-1:] == [ids[2]], name
        _press(insp, target, Qt.Key_Right)
        assert insp.current_event_id() == ids[3], name
        _press(insp, target, Qt.Key_Left)
        assert insp.current_event_id() == ids[2], name
        _press(insp, target, Qt.Key_U, Qt.ControlModifier)
        assert insp.current_event_id() == ids[0], name
    assert not [m for m in qt_messages if "Ambiguous" in m], qt_messages
    close_window(window)


def test_save_and_undo_in_the_docked_inspector_reach_the_main_windows_actions_once(
        private_settings, qtbot, qt_messages, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(MainWindow, "_save_project",
                        lambda self: calls.append("save") or True)
    monkeypatch.setattr(MainWindow, "_save_project_as",
                        lambda self: calls.append("save-as") or True)
    window, insp = _open(qtbot, "dock")
    ids = _ids(window)
    for target in (insp._plot_widget, insp._filter_combo):
        calls.clear()
        _press(insp, target, Qt.Key_S, Qt.ControlModifier)
        assert calls == ["save"], (target, calls)
        _press(insp, target, Qt.Key_S, Qt.ControlModifier | Qt.ShiftModifier)
        assert calls == ["save", "save-as"], (target, calls)
        insp.show_event(ids[1])
        _press(insp, target, Qt.Key_R)
        assert not accepted(window, ids[1])
        _press(insp, target, Qt.Key_Z, Qt.ControlModifier)
        assert accepted(window, ids[1]), "one Ctrl+Z, one undo"
        _press(insp, target, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
        assert not accepted(window, ids[1]), "one Ctrl+Shift+Z, one redo"
        _press(insp, target, Qt.Key_Z, Qt.ControlModifier)
        assert accepted(window, ids[1])
    # The pointer route stays: the Save button is the inspector's own action.
    calls.clear()
    insp.act_save_project.trigger()
    assert calls == ["save"]
    assert not [m for m in qt_messages if "Ambiguous" in m], qt_messages
    close_window(window)


def test_back_in_its_own_window_it_answers_its_own_save_and_undo_again(private_settings, qtbot,
                                                                       monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(MainWindow, "_save_project",
                        lambda self: calls.append("save") or True)
    window, insp = _open(qtbot, "dock")
    assert insp.act_save_project.shortcut().isEmpty()
    window.set_curation_host("event_inspector", "window")
    settle()
    assert insp.act_save_project.shortcut() == QtGui.QKeySequence(QtGui.QKeySequence.Save)
    assert insp._undo_shortcut.key() == QtGui.QKeySequence("Ctrl+Z")
    _press(insp, insp._plot_widget, Qt.Key_S, Qt.ControlModifier)
    assert calls == ["save"]
    close_window(window)


def _keys(obj) -> list[QtGui.QKeySequence]:
    if isinstance(obj, QtGui.QAction):
        return [k for k in obj.shortcuts() if not k.isEmpty()] if obj.isEnabled() else []
    return [obj.key()] if obj.isEnabled() and not obj.key().isEmpty() else []


def test_no_shortcut_in_the_docked_inspector_shadows_one_of_the_main_window(private_settings,
                                                                           qtbot):
    window, insp = _open(qtbot, "dock")
    main: set[str] = set()
    for obj in window.findChildren(QtGui.QAction) + window.findChildren(QtGui.QShortcut):
        parent = obj.parent()
        if isinstance(parent, QtWidgets.QWidget) and (parent is insp or insp.isAncestorOf(parent)):
            continue
        context = (obj.shortcutContext() if isinstance(obj, QtGui.QAction) else obj.context())
        if context in (Qt.WindowShortcut, Qt.ApplicationShortcut):
            main.update(k.toString() for k in _keys(obj))
    own = {k.toString()
           for obj in insp.findChildren(QtGui.QAction) + insp.findChildren(QtGui.QShortcut)
           for k in _keys(obj)}
    assert own and main
    assert not (own & main), sorted(own & main)
    close_window(window)


def test_a_click_in_the_docked_inspector_takes_the_keyboard(private_settings, qtbot):
    window, insp = _open(qtbot, "dock")
    ids = _ids(window)
    spins = [s for s in window._analysis_panel.findChildren(QtWidgets.QAbstractSpinBox)
             if s.isVisible() and s.isEnabled()]
    assert spins, "a detection parameter to type into"
    spin = spins[0]
    before = spin.text()
    # A focusable button, two no-focus buttons, the selectable readout, and a plain label
    # whose press propagates up to the window.
    for click in (insp._next_btn, insp._label_apply_btn, insp._details_btn, insp._readout,
                  insp._progress):
        window.activateWindow()
        spin.setFocus(Qt.OtherFocusReason)
        settle(5)
        assert QtWidgets.QApplication.focusWidget() is spin
        QTest.mouseClick(click, Qt.LeftButton, Qt.NoModifier, QtCore.QPoint(4, 4))
        settle(5)
        focus = QtWidgets.QApplication.focusWidget()
        assert focus is insp or insp.isAncestorOf(focus), (click, focus)
    target = insp.current_event_id()
    QTest.keyClick(QtWidgets.QApplication.focusWidget(), Qt.Key_R)
    settle()
    assert not accepted(window, target)
    assert sum(not accepted(window, i) for i in ids) == 1, "only the inspector's event"
    assert spin.text() == before, "no detection parameter changed"
    close_window(window)


def _dock_tab(window, title: str):
    """The main window's dock tab bar holding ``title``, and that tab's index."""
    for bar in window.findChildren(QtWidgets.QTabBar):
        if bar.parent() is window:
            for i in range(bar.count()):
                if bar.tabText(i) == title:
                    return bar, i
    raise AssertionError(f"no dock tab {title!r}")


def test_clicking_the_inspectors_tab_takes_the_keyboard_from_the_panel_behind_it(
        private_settings, qtbot):
    """Review C-R1: the keyboard stayed in a detection spin box hidden behind the tab just
    clicked, and '1' made it 11.000 s while R curated nothing on screen."""
    window, insp = _open(qtbot, "dock")
    window.activateWindow()
    window._inspector.raise_()
    settle()
    spin = next(s for s in window._analysis_panel.findChildren(QtWidgets.QDoubleSpinBox)
                if s.isEnabled() and not s.visibleRegion().isEmpty())
    spin.setFocus(Qt.MouseFocusReason)
    settle()
    before = spin.value()
    bar, index = _dock_tab(window, "Event inspector")
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, bar.tabRect(index).center())
    settle(60)
    assert spin.visibleRegion().isEmpty() and not insp.visibleRegion().isEmpty()
    focus = QtWidgets.QApplication.focusWidget()
    assert focus is insp or insp.isAncestorOf(focus), focus
    target = insp.current_event_id()
    for key in (Qt.Key_Up, Qt.Key_1, Qt.Key_R):
        QTest.keyClick(QtWidgets.QApplication.focusWidget(), key)
        settle(5)
    assert spin.value() == before, "no hidden detection parameter changed"
    assert not accepted(window, target) and "type=translocation" in _flag(window, target)
    close_window(window)


@pytest.mark.parametrize("key", [Qt.Key_Tab, Qt.Key_Backtab])
def test_tab_walks_past_the_panel_behind_the_docked_inspector(private_settings, qtbot, key):
    """Review C-R5: the 41st Tab put the keyboard in the Inspector panel's widgets while
    that panel was behind the Event inspector's tab."""
    window, insp = _open(qtbot, "dock")
    window.activateWindow()
    insp._index_spin.setFocus(Qt.TabFocusReason)
    settle()
    panel = window._inspector
    assert panel.visibleRegion().isEmpty()
    reached_results = False
    for n in range(150):
        QTest.keyClick(QtWidgets.QApplication.focusWidget(), key)
        settle(2)
        focus = QtWidgets.QApplication.focusWidget()
        assert focus is not None and not panel.isAncestorOf(focus), (n, focus)
        reached_results |= window._results_dock.isAncestorOf(focus)
    assert reached_results, "the walk went on past the hidden panel, not back to the start"
    close_window(window)


def test_keys_on_the_trace_never_reach_the_docked_inspector(private_settings, qtbot):
    window, insp = _open(qtbot, "dock")
    df = window._analysis_panel.current_detection().payload.df
    before = (list(df["accepted"]), list(df["flag"]))
    window.activateWindow()
    for key in (Qt.Key_R, Qt.Key_A, Qt.Key_1, Qt.Key_Space):
        window._trace_view.setFocus(Qt.OtherFocusReason)
        settle(5)
        QTest.keyClick(window._trace_view, key)
        settle(5)
    df = window._analysis_panel.current_detection().payload.df
    assert (list(df["accepted"]), list(df["flag"])) == before
    close_window(window)


def test_cmd_z_in_the_docked_inspector_walks_an_advance_back(private_settings, qtbot):
    window, insp = _open(qtbot, "dock")
    ids = _ids(window)
    insp.set_advance_after_decision(True)
    try:
        insp.show_event(ids[1])
        _press(insp, insp._plot_widget, Qt.Key_R)
        assert insp.current_event_id() == ids[2] and not accepted(window, ids[1])
        _press(insp, insp._plot_widget, Qt.Key_Z, Qt.ControlModifier)
        assert accepted(window, ids[1])
        assert insp.current_event_id() == ids[1], "back on the event the advance left"
    finally:
        insp.set_advance_after_decision(False)
    close_window(window)


def test_an_editor_inside_the_docked_inspector_keeps_its_own_keys(private_settings, qtbot):
    window, insp = _open(qtbot, "dock")
    ids = _ids(window)
    flags = [_flag(window, i) for i in ids]
    _press(insp, insp._index_spin, Qt.Key_A, Qt.ControlModifier)   # select the number
    QTest.keyClick(insp._index_spin, Qt.Key_3)
    QTest.keyClick(insp._index_spin, Qt.Key_Return)
    settle()
    assert insp.current_event_id() == ids[2], "the digit typed into the Event box"
    assert [_flag(window, i) for i in ids] == flags, "and labelled nothing"
    close_window(window)
