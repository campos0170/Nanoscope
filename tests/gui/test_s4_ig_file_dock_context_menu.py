"""IG-06 — the File Manager's context menu is transient again, and IG-05's guard is what
lets this file exist at all.

``FileDock._show_context_menu`` built its popup as ``QtWidgets.QMenu(self)`` and let it go.
A Qt parent is an OWNER, so every right-click left the menu and its five QActions alive
under the dock for the lifetime of the window. Measured on the shipped widget: 400
right-clicks on one recording row took the dock from 0 QMenus / 1 QAction to 400 / 2001 —
exactly +1 and +5 per click, strictly monotone across every 100-click checkpoint, no
plateau, ~84 MB of RSS per 1000 clicks. Right-clicking the list is the dock's own advertised
way to say "show only this recording" (its hint label points at it), so a long comparison
session climbs the whole time.

Everything here is driven through the real widget with a real ``QContextMenuEvent`` on the
tree's viewport — the event Qt delivers on a right-click, which the dock has asked for by
``setContextMenuPolicy(CustomContextMenu)``. That gesture used to HANG the suite (IG-05:
conftest's ``QMenu.exec`` stub never reached a menu instance), which is why no test had
ever right-clicked this dock; ``tests/gui/test_s4_ig_modal_guard_reaches_qmenu.py`` pins
that half.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import time

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

import nanoscope.gui.file_dock as fd
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 20_000

#: The gesture is cheap; 400 is the count the defect was measured at. A leak of +1 per click
#: shows up in the first hundred, so this stays well inside a couple of seconds.
_CLICKS = 400


def _doc(tmp_path, name: str) -> TraceDocument:
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    x = np.full(N, -1000.0, dtype=np.float64) + rng.normal(0.0, 5.0, N)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(np.ascontiguousarray(x, dtype=np.float32),))


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    """A real MainWindow with two recordings open — the state the compare entries are for."""
    w = MainWindow()
    qtbot.addWidget(w)
    ids = [w._session.add_document(_doc(tmp_path, n)) for n in ("run_a.dat", "run_b.dat")]
    for doc_id in ids:
        w._create_default_layers(w._session.document(doc_id))
    w._on_selection(ids[0], 0)
    w.show()
    qtbot.wait(20)
    dock = w._file_dock
    dock._tree.expandAll()
    qtbot.wait(10)
    yield w, dock, ids
    w._analysis_controller.abort()


def _doc_rows(dock) -> list[QtWidgets.QTreeWidgetItem]:
    """Every "this is a recording" row of the tree, in view order."""
    out = []
    it = QtWidgets.QTreeWidgetItemIterator(dock._tree)
    while it.value():
        item = it.value()
        payload = item.data(0, fd._ROLE)
        if payload and payload[0] == "doc":
            out.append(item)
        it += 1
    return out


def _right_click(dock, item) -> None:
    """The real right-click: a QContextMenuEvent on the tree's viewport, which is what Qt
    delivers and what ``CustomContextMenu`` turns into ``customContextMenuRequested``."""
    tree = dock._tree
    rect = tree.visualItemRect(item)
    pos = QtCore.QPoint(rect.left() + 4, rect.center().y())
    ev = QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Reason.Mouse, pos,
                                 tree.viewport().mapToGlobal(pos))
    QtWidgets.QApplication.sendEvent(tree.viewport(), ev)


def _counts(dock) -> tuple[int, int]:
    gc.collect()
    return (len(dock.findChildren(QtWidgets.QMenu)),
            len(dock.findChildren(QtGui.QAction)))


# ================================================================================================
# The gesture works at all — and does not hang
# ================================================================================================
def test_right_clicking_a_recording_offers_the_compare_entries(wired, qtbot, blocked_modals):
    w, dock, ids = wired
    t0 = time.perf_counter()
    _right_click(dock, _doc_rows(dock)[0])
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"the right-click blocked for {elapsed:.1f} s"
    assert len(blocked_modals) == 1, blocked_modals
    offered = blocked_modals[0].text
    for entry in ("Show only this recording", "Add to view (compare)", "Close recording"):
        assert entry in offered, f"{entry!r} missing from {offered!r}"


def test_picking_show_only_from_the_real_menu_still_shows_only_that_recording(
        wired, qtbot, menu_answer):
    """THE FREEING MUST NOT BREAK THE CHOOSING. ``exec`` returns a QAction the menu owns and
    the dock compares it by identity; a menu freed a moment too early would leave that
    comparison reading a dead object, and the gesture would quietly do nothing."""
    w, dock, ids = wired
    rows = _doc_rows(dock)
    assert len(rows) == 2
    w._session.show_only(ids[0])
    qtbot.wait(10)
    assert w._session.shown_doc_ids() == frozenset({ids[0]})

    menu_answer("Show only this recording")
    _right_click(dock, rows[1])
    qtbot.wait(30)

    assert w._session.shown_doc_ids() == frozenset({ids[1]}), (
        f"picking 'Show only this recording' left {sorted(w._session.shown_doc_ids())} shown")


def test_picking_add_to_view_puts_the_other_recording_back(wired, qtbot, menu_answer):
    w, dock, ids = wired
    w._session.show_only(ids[0])
    qtbot.wait(10)
    assert w._session.shown_doc_ids() == frozenset({ids[0]})

    menu_answer("Add to view (compare)")
    _right_click(dock, _doc_rows(dock)[1])
    qtbot.wait(30)

    assert w._session.shown_doc_ids() == frozenset(ids), (
        f"'Add to view (compare)' left {sorted(w._session.shown_doc_ids())} shown")


# ================================================================================================
# IG-06 — the popup is transient
# ================================================================================================
def test_four_hundred_right_clicks_leave_no_menu_behind(wired, qtbot):
    """THE DEFECT, at the count it was measured at: 0 -> 400 menus, 1 -> 2001 actions."""
    w, dock, ids = wired
    row = _doc_rows(dock)[0]
    menus0, actions0 = _counts(dock)

    checkpoints = []
    for i in range(_CLICKS):
        _right_click(dock, row)
        if (i + 1) % 100 == 0:
            # deleteLater lands on the next turn of the loop — which in the shipped app is
            # the very next thing that happens after a menu closes.
            qtbot.wait(1)
            checkpoints.append(_counts(dock))

    qtbot.wait(5)
    menus1, actions1 = _counts(dock)
    assert menus1 == menus0, (
        f"{_CLICKS} right-clicks left {menus1 - menus0} QMenu widgets alive under the File "
        f"Manager (checkpoints: {checkpoints})")
    assert actions1 == actions0, (
        f"{_CLICKS} right-clicks left {actions1 - actions0} QActions alive under the File "
        f"Manager (checkpoints: {checkpoints})")
    assert all(c == (menus0, actions0) for c in checkpoints), (
        f"the count moved mid-run and came back: {checkpoints}")


def test_the_menu_is_gone_even_when_the_gesture_ends_in_a_signal(wired, qtbot, menu_answer):
    """The freeing is in a ``finally``, so it survives the branch that emits — the path a
    ``deleteLater()`` written after the last ``elif`` would take, but a ``return`` or a
    raising slot would skip."""
    w, dock, ids = wired
    w._session.show_only(ids[0])
    qtbot.wait(10)
    menus0, actions0 = _counts(dock)

    # "Add to view (compare)" stays enabled once both are drawn (the dock greys an entry out
    # only when it would say nothing), so all twenty gestures really do reach the emit.
    menu_answer("Add to view (compare)")
    for _ in range(20):
        _right_click(dock, _doc_rows(dock)[1])
        qtbot.wait(1)
    assert w._session.shown_doc_ids() == frozenset(ids)

    qtbot.wait(5)
    assert _counts(dock) == (menus0, actions0), _counts(dock)


def test_a_right_click_on_a_group_row_opens_nothing_and_leaks_nothing(wired, qtbot,
                                                                     blocked_modals):
    """The early return above the menu is still an early return: no popup, and nothing
    built to be freed."""
    w, dock, ids = wired
    tree = dock._tree
    group = None
    it = QtWidgets.QTreeWidgetItemIterator(tree)
    while it.value():
        payload = it.value().data(0, fd._ROLE)
        if payload and payload[0] == "group":
            group = it.value()
            break
        it += 1
    if group is None:
        pytest.skip("this tree has no group rows to right-click")

    menus0, actions0 = _counts(dock)
    _right_click(dock, group)
    qtbot.wait(5)
    assert blocked_modals == [], "a group row raised a popup menu"
    assert _counts(dock) == (menus0, actions0)
