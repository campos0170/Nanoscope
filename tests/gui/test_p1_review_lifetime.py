"""Phase 1 Foundation, review round: the pane lives as long as its page.

``QStackedWidget.removeWidget`` hands the pane to Python; ``InspectorHost.retire`` has to take
it back, or the pane (and every control of the page that was moved into it) is deleted the
moment ``AnalysisExplorer.remove_page`` returns, while the page is still alive. No test here
keeps a reference to a pane: the real route does not.

Synthetic only; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import shiboken6
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui.analysis_inspector import InspectorHost

from . import p1_hosting as H

KEYS = ("distributions", "iv", "noise")


@pytest.fixture(scope="module")
def builders():
    return H.page_builders()


def _widgets(page) -> dict[str, QtWidgets.QWidget]:
    return {name: value for name, value in vars(page).items()
            if isinstance(value, QtWidgets.QWidget) and shiboken6.isValid(value)}


def _drain():
    for _ in range(3):
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        QtWidgets.QApplication.processEvents()


@pytest.mark.parametrize("key", KEYS)
def test_remove_page_leaves_every_control_the_page_holds_alive(qtbot, builders, key):
    page = builders[key]()
    explorer = H.hosted(qtbot, page, key)
    before = _widgets(page)
    assert len(before) > 5
    assert explorer.remove_page(key) is page
    dead = sorted(name for name, widget in before.items() if not shiboken6.isValid(widget))
    assert dead == []
    for widget in before.values():                  # reading them is what a page's closeEvent does
        widget.objectName()
    page.close()                                    # the page's own teardown reads its controls
    page.deleteLater()
    _drain()
    assert not shiboken6.isValid(page)


def test_replacing_a_page_under_one_id_closes_the_old_one_without_raising(qtbot, builders):
    old = builders["distributions"]()
    explorer = H.hosted(qtbot, old, "distributions")
    new = builders["distributions"]()
    explorer.add_page("distributions", new)         # remove_page(old), old.close(), deleteLater
    _drain()
    assert not shiboken6.isValid(old)
    explorer.show_page("distributions", new)
    assert explorer.inspector().pane("distributions") is not None


@pytest.mark.parametrize("key", KEYS)
def test_a_removed_page_can_be_added_again_and_is_hosted_once_more(qtbot, builders, key):
    page = builders[key]()
    explorer = H.hosted(qtbot, page, key)
    host = explorer.inspector()
    explorer.remove_page(key)
    assert host.pane(key) is None
    explorer.show_page(key, page)                   # the same widget, adopted again
    QtWidgets.QApplication.processEvents()
    assert host.pane(key) is not None and host.current_key() == key
    host.set_tab(0)
    first = host.pane(key).data_pane.group_keys()[0]
    frame = host.pane(key).data_pane.group(first)
    assert frame.body.findChildren(QtWidgets.QWidget)           # the controls came back
    assert host.isVisible()


def test_adopting_a_key_twice_retires_the_first_pane_without_killing_its_controls(qtbot):
    page = H.SyntheticPlotPage()
    qtbot.addWidget(page)
    host = InspectorHost()
    qtbot.addWidget(host)
    host.adopt("synthetic", page.inspector_adapter())
    host.adopt("synthetic", page.inspector_adapter())           # the key already holds a pane
    host.show_page("synthetic")
    assert shiboken6.isValid(page._estimator)
    assert host.pane("synthetic").data_pane.group("estimator").isAncestorOf(page._estimator)


def test_retiring_a_pane_invalidates_its_stack_item_and_keeps_controls_until_page_death(qtbot):
    page, other = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("noise")
    # This test explicitly destroys page; pytest-qt must not close it a second time.
    qtbot.addWidget(other)
    host = InspectorHost()
    qtbot.addWidget(host)
    try:
        adapter = page.inspector_adapter()
        pane = host.adopt("iv", adapter)
        remaining = host.adopt("noise", other.inspector_adapter())
        host.show_page("iv")
        layout = host._stack.layout()
        item = layout.itemAt(layout.indexOf(pane))
        other_item = layout.itemAt(layout.indexOf(remaining))
        controls = [widget for group in adapter.groups for widget in group.widgets]
        assert item.widget() is pane and controls
        switched = QtTest.QSignalSpy(host._stack.currentChanged)

        host.retire("iv")
        assert not shiboken6.isValid(item)
        assert shiboken6.isValid(other_item) and other_item.widget() is remaining
        assert host._stack.count() == 1 and host._stack.currentWidget() is remaining
        assert switched.count() == 1
        assert pane.parentWidget() is host._stack and shiboken6.isValid(pane)
        assert all(shiboken6.isValid(widget) for widget in controls)
        assert pane.data_pane.isAncestorOf(page._estimator)
        page._estimator.objectName()             # the removed page may still read its controls

        page.deleteLater()
        _drain()
        assert not shiboken6.isValid(page) and not shiboken6.isValid(pane)
        assert all(not shiboken6.isValid(widget) for widget in controls)
        assert shiboken6.isValid(other) and shiboken6.isValid(remaining)
    finally:
        if shiboken6.isValid(page):
            page.deleteLater()
            _drain()
