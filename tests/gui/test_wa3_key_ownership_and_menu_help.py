"""WA3 — the three residues the wave-A fix-up round left, closed by the coordinator.

1. WB3's second half. The Event *inspector* still registered A, R, Space, T, ←, → and 1–9 as
   ``Qt.WindowShortcut``, which asks only "is my window active?". The gallery had already been
   narrowed for this; the inspector had not. Measured at 1512×900 with the inspector open and
   active while six rows were shift-selected in the events table: an R aimed at the TABLE was
   answered by the inspector — the six highlighted rows were untouched and event 7, which the
   user had selected nowhere, was silently rejected. A curation write to a row nothing on
   screen shows as chosen is the worst shape in this app.

2. The new menu-help line pushed Qt's OWN fallback tooltip into the status bar. Qt derives an
   unwritten ``toolTip()`` from the label by stripping the mnemonic *and* a trailing ellipsis,
   so the guard's comparison against ``action.text()`` (ellipsis intact) never matched for
   ``&Open…`` or ``Save Project &As…``: hovering either wiped whatever report was standing
   there and replaced it with the bare word "Open".

3. Batch detect counted RECORDINGS and never said that some of those tables hold events the
   user rejected or added by hand — the one sentence the four single-run doors all say.

The three are here together because they are one idea: **a control must answer for the thing
the user aimed at, and say what it is about to destroy.**
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.core.session import Session
from nanoscope.gui.event_inspector import EventInspectorWindow
from nanoscope.gui.main_window import MainWindow


@pytest.fixture
def session(qapp) -> Session:
    """An empty Session is enough: every assertion here is about which window a KEY belongs
    to, which is decided before any data is involved."""
    return Session()


@pytest.fixture
def main_window(qapp, qtbot) -> MainWindow:
    """A real MainWindow at the owner's display size — the size every WA measurement used."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 900)
    w.show()
    qapp.processEvents()
    return w


# ---------------------------------------------------------------------------------------------
# 1. Key ownership
# ---------------------------------------------------------------------------------------------

def test_the_inspectors_keys_ask_where_the_focus_is_not_which_window_is_active(qapp, session):
    """``WidgetWithChildrenShortcut``, never ``WindowShortcut`` — the whole fix in one assert."""
    win = EventInspectorWindow(session)
    try:
        assert win._shortcuts, "the inspector registered no shortcuts at all"
        contexts = {sc.context() for sc in win._shortcuts}
        assert contexts == {Qt.WidgetWithChildrenShortcut}, (
            "an inspector shortcut still answers for the whole window: "
            f"{sorted(str(c) for c in contexts)}")
    finally:
        win.deleteLater()


def test_the_inspector_keeps_its_own_keys_when_qts_map_cannot_reach_them(qapp, session):
    """The price of narrowing the context is that Qt wants the focus inside this window, and a
    window shown but never activated has no focus widget at all. The fallback pays it."""
    win = EventInspectorWindow(session)
    try:
        assert hasattr(win, "_handle_own_key"), "no fallback: the narrowing costs the inspector its keyboard"
        fired: list[str] = []
        for sc in win._shortcuts:
            sc.activated.connect(lambda k=sc.key().toString(): fired.append(k))
        ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_R, Qt.NoModifier)
        assert win._handle_own_key(ev) is True
        assert fired == ["R"], f"expected the inspector's own R, got {fired}"
    finally:
        win.deleteLater()


def test_a_disabled_inspector_shortcut_is_as_dead_in_the_fallback_as_in_the_map(qapp, session):
    win = EventInspectorWindow(session)
    try:
        for sc in win._shortcuts:
            sc.setEnabled(False)
        ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_R, Qt.NoModifier)
        assert win._handle_own_key(ev) is False
    finally:
        win.deleteLater()


def test_the_fallback_reads_the_shortcuts_it_installed_rather_than_a_second_list(qapp, session):
    """A second list of keys is how a shortcut and its fallback drift into disagreeing about
    what R means. Adding one shortcut must change what the fallback answers, with no edit."""
    win = EventInspectorWindow(session)
    try:
        ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_F7, Qt.NoModifier)
        assert win._handle_own_key(ev) is False
        seen: list[str] = []
        sc = QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_F7), win)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(lambda: seen.append("F7"))
        win._shortcuts.append(sc)
        assert win._handle_own_key(ev) is True and seen == ["F7"]
    finally:
        win.deleteLater()


# ---------------------------------------------------------------------------------------------
# 2. Menu help
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["&Open…", "Save Project &As…", "&Quit", "Zoom &In"])
def test_a_label_echo_never_reaches_the_status_bar(qapp, main_window, label):
    """Qt's auto-derived tooltip is the label without its mnemonic and without its trailing
    ellipsis. Whatever the ellipsis rule is in this Qt, asking a fresh QAction the same
    question is exact — which is why the guard asks rather than reconstructs."""
    act = QtGui.QAction(label)
    assert main_window._menu_action_help(act) == "", (
        f"{label!r} pushed its own label into the status bar as if it were a sentence")


def test_a_written_sentence_still_reaches_the_status_bar(qapp, main_window):
    act = QtGui.QAction("&Open…")
    act.setToolTip("Open a recording from disk.")
    assert main_window._menu_action_help(act) == "Open a recording from disk."


def test_a_status_tip_still_wins_over_a_tooltip(qapp, main_window):
    act = QtGui.QAction("&Open…")
    act.setToolTip("the tooltip")
    act.setStatusTip("the line written for this bar")
    assert main_window._menu_action_help(act) == "the line written for this bar"


def test_the_menus_that_ship_carry_real_sentences_and_no_echoes(qapp, main_window):
    """The census, on the real menu bar: every sentence the app shows must be one somebody
    wrote. This is what stops the guard being tightened into uselessness."""
    echoes, real = [], 0
    for menu in main_window.menuBar().findChildren(QtWidgets.QMenu):
        for act in menu.actions():
            if act.isSeparator() or not act.text():
                continue
            help_text = main_window._menu_action_help(act)
            if not help_text:
                continue
            real += 1
            if help_text == QtGui.QAction(act.text()).toolTip():
                echoes.append(act.text())
    assert echoes == [], f"actions pushing their own label into the status bar: {echoes}"
    assert real > 20, f"only {real} actions carry a sentence — the guard is now too tight"
