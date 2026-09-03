"""IG-05 — the suite's modal guard must reach the call it says it reaches: ``menu.exec``.

``tests/conftest.py`` closes "the SEAM every one of them must pass through" and names
``QMenu.exec`` as one of two — "the other blocking ``exec`` (the File Manager's context
menu)" — under a header that spells out the stake: a live modal "is NOT curable by a more
aggressive wait … presents as 'the suite hangs' instead of 'test X failed'".

Only the QDialog half held. Under PySide6 6.11 ``QMenu.exec`` sits in the type dict as a
``staticmethod`` (it has a static overload) and Shiboken answers ``menu.exec`` from the C++
method table instead, so:

    QtWidgets.QMenu.exec  -> <function _menu_exec>      is_modal_guard -> True
    QtWidgets.QMenu().exec-> <built-in method exec>     is_modal_guard -> False

i.e. the assertion in ``tests/gui/test_modal_guard.py`` that "QMenu.exec is guarded" passed
while every real menu still opened a real nested event loop. A probe that right-clicked the
File Manager sat there until a 500 s watchdog killed it.

These tests are written against the INSTANCE and against a real widget's real gesture,
because the class-level lookup is precisely the thing that lied. They also pin the two
seams the fix installs (per-instance stub at construction; an application event filter that
closes any menu that nevertheless becomes visible) and the guard's answers — cancel by
default, and the entry a test names through ``menu_answer``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import pytest
from PySide6 import QtCore, QtWidgets

from conftest import is_modal_guard

#: Generous next to the ~0 ms a guarded exec takes, and far under any real hang: a live
#: nested loop never returns at all, so anything that finishes has not hung.
_BUDGET_S = 5.0


def test_a_qmenu_instance_and_not_only_the_class_carries_the_guard(qapp):
    """THE DEFECT, stated as the property that was false.

    Both lookups are asserted, and in this order, so a future PySide that fixes (or breaks)
    one of them tells you WHICH one changed rather than leaving the suite to hang."""
    menu = QtWidgets.QMenu()
    menu.addAction("Show only this recording")

    assert is_modal_guard(QtWidgets.QMenu.exec), "the class-level stub is not installed"
    assert is_modal_guard(menu.exec), (
        f"the CLASS says guarded and the INSTANCE resolves {menu.exec!r} — this is the hole: "
        f"every real menu.exec() opens a live nested event loop and hangs the run")
    assert is_modal_guard(menu.exec_), "the PyQt-era exec_ alias is not guarded on the instance"


def test_a_subclass_of_qmenu_is_guarded_too(qapp):
    """The stub is installed at construction, so anything that runs ``QMenu.__init__``
    inherits it — a future ``class RecordingMenu(QMenu)`` does not reopen the hole."""

    class _Sub(QtWidgets.QMenu):
        pass

    sub = _Sub()
    sub.addAction("anything")
    assert is_modal_guard(sub.exec), sub.exec


def test_exec_returns_at_once_and_says_what_the_menu_was_going_to_offer(qapp, blocked_modals):
    """It FIRES, and it is recorded. A guard nobody has watched fire is ``None == None``."""
    menu = QtWidgets.QMenu()
    menu.addAction("Show only this recording")
    menu.addSeparator()
    closed = menu.addAction("Close recording")
    closed.setEnabled(False)

    t0 = time.perf_counter()
    chosen = menu.exec(QtCore.QPoint(0, 0))
    elapsed = time.perf_counter() - t0

    assert chosen is None, "the cancelling answer is 'nothing was picked'"
    assert elapsed < _BUDGET_S, f"menu.exec blocked for {elapsed:.1f} s — the loop is live"
    assert len(blocked_modals) == 1, blocked_modals
    rec = blocked_modals[0]
    assert rec.kind == "QMenu", rec
    # The entries, because a context menu has no title and the entries are all the user sees.
    assert "Show only this recording" in rec.text, rec
    assert "Close recording (disabled)" in rec.text, rec


def test_the_menu_never_actually_opens(qapp):
    """The point of answering instead of driving: no popup is ever put on the screen, so a
    test cannot be left with a grabbed pointer or a stray top-level window."""
    menu = QtWidgets.QMenu()
    menu.addAction("Show only this recording")
    menu.exec(QtCore.QPoint(0, 0))
    assert not menu.isVisible()
    assert QtWidgets.QApplication.activePopupWidget() is None


def test_menu_answer_clicks_the_entry_a_test_names(qapp, menu_answer, blocked_modals):
    """The guard can also answer AS A USER PICKING SOMETHING — which is what makes the
    context menu's own branches testable at all (see the File Manager tests)."""
    menu = QtWidgets.QMenu()
    only = menu.addAction("Show only this recording")
    fired: list[str] = []
    only.triggered.connect(lambda: fired.append("only"))
    menu.addAction("Add to view (compare)")

    menu_answer("Show only this recording")
    chosen = menu.exec(QtCore.QPoint(0, 0))

    assert chosen is only, chosen
    assert fired == ["only"], "Qt's own exec triggers the action it returns; so does the guard"
    assert blocked_modals and "Show only this recording" in blocked_modals[0].text


def test_menu_answer_refuses_an_entry_the_user_could_not_have_clicked(qapp, menu_answer):
    """A disabled entry is not a gesture. Naming one is a broken test, and it says so
    rather than quietly cancelling and letting the assertions downstream go green."""
    menu = QtWidgets.QMenu()
    act = menu.addAction("Add to view (compare)")
    act.setEnabled(False)

    menu_answer("Add to view (compare)")
    with pytest.raises(AssertionError, match="DISABLED"):
        menu.exec(QtCore.QPoint(0, 0))


def test_menu_answer_refuses_an_entry_that_is_not_there(qapp, menu_answer):
    menu = QtWidgets.QMenu()
    menu.addAction("Close recording")
    menu_answer("Show only this recording")
    with pytest.raises(AssertionError, match="no such entry"):
        menu.exec(QtCore.QPoint(0, 0))


def test_a_menu_that_slips_past_the_instance_stub_is_still_closed_when_it_shows(qapp,
                                                                               blocked_modals):
    """THE NET, exercised. Menus Qt builds in C++ (``QMenuBar.addMenu``,
    ``createStandardContextMenu``) never run the patched ``__init__``, and ``popup()`` plus
    a hand-rolled loop is a blocking popup with no ``exec`` in it to patch. The instance
    stub is deleted here to stand in for both: what is left is the real C++ ``exec``, i.e.
    exactly the call that hung the run before.
    """
    menu = QtWidgets.QMenu()
    menu.addAction("Raw C++ entry")
    del menu.exec                      # back to <built-in method exec> — the blocking one
    assert not is_modal_guard(menu.exec)

    t0 = time.perf_counter()
    menu.exec(QtCore.QPoint(0, 0))     # opens a REAL nested event loop
    elapsed = time.perf_counter() - t0

    assert elapsed < _BUDGET_S, f"the visible-popup net did not close it ({elapsed:.1f} s)"
    assert any(m.kind == "QMenu" and "Raw C++ entry" in m.text for m in blocked_modals), \
        blocked_modals
    assert not menu.isVisible()
