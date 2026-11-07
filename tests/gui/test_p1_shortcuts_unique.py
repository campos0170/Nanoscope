"""Phase 1: no key is registered twice among the live shortcuts under the Explorer.

Qt fires NEITHER of two shortcuts on one key in one window, and a shortcut whose parent is
hidden does not fire. So for each of the eight pages in front, on each inspector tab, every
enabled ``QShortcut`` whose parent is visible and every visible enabled ``QAction`` shortcut
is collected, and no resolved key may appear twice. The old Ctrl+1 to 4 of the pages' bars
and Distributions' Ctrl+Return are the cases that would have collided. Synthetic; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections import Counter

import pytest
from PySide6 import QtGui, QtWidgets

from . import p1_hosting as H

PAGE_IDS = ("distributions", "distribution_summaries", "comparison_figures", "event_waveforms",
            "noise", "iv", "voltage_series", "subpopulations")


def live_keys(explorer) -> Counter:
    keys: Counter = Counter()
    for shortcut in explorer.findChildren(QtGui.QShortcut):
        parent = shortcut.parent()
        if shortcut.isEnabled() and isinstance(parent, QtWidgets.QWidget) and parent.isVisible():
            keys[shortcut.key().toString()] += 1
    for action in explorer.findChildren(QtGui.QAction):
        if not (action.isEnabled() and action.shortcuts()):
            continue
        owners = [w for w in action.associatedObjects()
                  if isinstance(w, QtWidgets.QWidget) and w.isVisible()]
        if action.isVisible() and owners:
            for sequence in action.shortcuts():
                keys[sequence.toString()] += 1
    return keys


@pytest.fixture(scope="module")
def builders():
    return H.page_builders()


@pytest.mark.parametrize("key", PAGE_IDS)
def test_no_live_key_appears_twice_with_the_page_in_front(qtbot, builders, key):
    page = builders[key]()
    explorer = H.hosted(qtbot, page, key)
    other = builders["iv" if key != "iv" else "noise"]()
    explorer.add_page("iv" if key != "iv" else "noise", other)       # a page behind
    explorer.raise_page(key)
    for tab in (0, 1):
        explorer.inspector().set_tab(tab)
        QtWidgets.QApplication.processEvents()
        keys = live_keys(explorer)
        doubled = {k: n for k, n in keys.items() if n > 1}
        assert not doubled, (key, tab, doubled)
        assert keys["Ctrl+1"] == 1 and keys["Ctrl+2"] == 1             # the shell's, once
        assert keys["Ctrl+Return"] == 1 and keys["Ctrl+Enter"] == 1    # Update plot's, once


def test_the_detector_sees_a_doubled_key(qtbot):
    """The control: two shortcuts on one key in one window ARE reported."""
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)
    for _ in range(2):
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+K"), window)
    action = QtGui.QAction("x", window)
    action.setShortcut("Ctrl+K")
    button = QtWidgets.QToolButton()
    window.setCentralWidget(button)
    button.addAction(action)
    window.show()
    assert live_keys(window)["Ctrl+K"] == 3
