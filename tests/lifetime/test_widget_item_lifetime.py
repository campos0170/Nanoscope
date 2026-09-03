"""Does a slot destroy the very item Qt is still holding? Driven with REAL mouse events.

Each test here reproduces (or forbids) the shape of the 2026-08-31 SIGSEGV: a checkable
QListWidget whose item signal leads, however indirectly, back to a ``clear()`` of that same
widget. See ``conftest.py`` for why this directory runs the collector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow


def _frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    h = n // 2
    return pd.DataFrame({
        "dwell_s": np.abs(np.r_[rng.normal(1e-4, 1.5e-5, h), rng.normal(6e-4, 6e-5, n - h)]),
        "amplitude": np.abs(np.r_[rng.normal(20.0, 2.0, h), rng.normal(48.0, 3.0, n - h)]),
        "frac_depth": np.abs(rng.normal(0.25, 0.05, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "rms": np.abs(rng.normal(1.4, 0.3, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _click_checkbox(view: QtWidgets.QAbstractItemView, row: int) -> None:
    """A REAL mouse press+release on a row's checkbox, through the delegate.

    ``setCheckState()`` would not do: it never enters ``QStyledItemDelegate::editorEvent``,
    which is the frame the crash report shows and the only path on which Qt is still holding
    the item when the slot runs.
    """
    idx = view.model().index(row, 0)
    rect = view.visualRect(idx)
    pt = QtCore.QPointF(rect.left() + 9, rect.center().y())
    for kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease):
        QtWidgets.QApplication.sendEvent(view.viewport(), QtGui.QMouseEvent(
            kind, pt, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        QtWidgets.QApplication.processEvents()


@pytest.fixture
def loaded(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"run A": _frame(seed=0), "run B": _frame(seed=1)})
    w.resize(1200, 800)
    w.show()
    for _ in range(6):
        QtWidgets.QApplication.processEvents()
    return w


# ORDER MATTERS IN THIS FILE. The two "survives repeated clicks" tests below reproduce the
# fault by DYING -- a SIGSEGV takes the whole pytest process with it, so anything after them
# never runs and never reports. The ordering assertion therefore goes FIRST: it fails loudly,
# with a sentence, on exactly the condition that causes the crash.


def test_the_legend_list_is_not_rebuilt_from_inside_its_own_slot(loaded, monkeypatch):
    """The ORDERING invariant, asserted rather than survived.

    "Does not segfault" is a useless assertion in a suite that mostly runs with the collector
    off -- the fault needs the collector to recycle the memory. THIS is the durable statement:
    while a slot connected to the widget's own item signal is on the stack, nothing may
    ``clear()`` that widget. Deferring the rebuild (QTimer.singleShot) satisfies it; calling
    it inline does not.
    """
    entries = loaded._legend_entries
    if entries.count() == 0:
        pytest.skip("no legend entries offered by this draw")

    state = {"in_slot": False, "violations": []}
    real_clear = QtWidgets.QListWidget.clear

    def watched_clear(self):
        if self is entries and state["in_slot"]:
            state["violations"].append("cleared while its own item slot was on the stack")
        return real_clear(self)

    real_slot = type(loaded)._on_legend_entry_toggled

    def watched_slot(self, item):
        state["in_slot"] = True
        try:
            return real_slot(self, item)
        finally:
            state["in_slot"] = False

    monkeypatch.setattr(QtWidgets.QListWidget, "clear", watched_clear, raising=True)
    monkeypatch.setattr(type(loaded), "_on_legend_entry_toggled", watched_slot, raising=True)

    _click_checkbox(entries, 0)
    QtWidgets.QApplication.processEvents()

    assert not state["violations"], (
        "the Legend-entries list was rebuilt from inside its own itemChanged slot -- that "
        "destroys the QListWidgetItem Qt is still holding on its editorEvent stack frame, "
        f"which is the 2026-08-31 SIGSEGV. Violations: {state['violations']}")


def test_unticking_a_legend_entry_repeatedly_does_not_destroy_the_item_under_the_mouse(loaded):
    """THE OWNER'S CRASH. He said: "working on the Plots panel, unchecking the legend on a
    plot." Reproduced at exit code 139 (SIGSEGV) on the SECOND click -- the first survived on
    memory the collector had not yet reused.

    A segfault kills the process, so this test cannot assert its way out of one; surviving
    the loop IS the assertion. The ordering it depends on is pinned separately by
    ``test_the_legend_list_is_not_rebuilt_from_inside_its_own_slot``, which fails loudly
    instead of dying.
    """
    entries = loaded._legend_entries
    if entries.count() == 0:
        pytest.skip("no legend entries offered by this draw")
    for i in range(8):
        if entries.count() == 0:
            break
        _click_checkbox(entries, i % entries.count())
    assert True   # reaching here is the result


def test_ticking_sources_repeatedly_does_not_destroy_the_item_under_the_mouse(loaded):
    """The same shape one widget over: ``set_data`` clears and refills ``_source_list``, and
    the tick handler can reach it. Also the path an ASYNCHRONOUS detection result takes when
    it lands while the user is mid-click."""
    sources = loaded._source_list
    assert sources.count() >= 2
    for i in range(8):
        _click_checkbox(sources, i % sources.count())
    assert True
