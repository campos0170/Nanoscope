"""The shared column checklist (``gui/column_checklist.py``): many columns in one gesture.

One widget answers every column chooser: the Results table's Columns… popup, the
Comparison pair matrix and (G3) the Plots features. These pin its contract — a click on a
row's TEXT toggles it, ``changed`` fires once per gesture, the filter matches label and
key and scopes the bulk buttons, the bounds hold and say why — and the popup's: it stays
up through clicks, Esc closes it, it flips above a button with no room below.
"""
from __future__ import annotations

import ast
import os
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from nanoscope.gui import column_checklist
from nanoscope.gui.column_checklist import ColumnChecklist, ColumnChooserPopup


def _entries(n):
    return [(f"key_{i}", f"Label {i}", f"tip {i}") for i in range(n)]


def _row(checklist, key):
    proxy = checklist.proxy
    for r in range(proxy.rowCount()):
        index = proxy.index(r, 0)
        if index.data(Qt.UserRole) == key:
            return index
    raise AssertionError(f"no row {key!r}; rows: {checklist.shown_keys()}")


def click_text(checklist, key):
    """A left click on the row's text, well right of the check indicator."""
    rect = checklist.view.visualRect(_row(checklist, key))
    QTest.mouseClick(checklist.view.viewport(), Qt.LeftButton, Qt.NoModifier,
                     QPoint(rect.left() + min(90, rect.width() - 4), rect.center().y()))


@pytest.fixture
def make(qapp, qtbot):
    def _make(**kwargs):
        widget = ColumnChecklist(**kwargs)
        qtbot.addWidget(widget)
        widget.resize(320, 420)
        widget.show()
        qtbot.waitExposed(widget)
        return widget
    return _make


def test_a_click_on_the_row_text_toggles_it_and_emits_once(make):
    checklist = make()
    checklist.set_entries(_entries(5), checked=["key_0", "key_1"], defaults=[])
    seen = []
    checklist.changed.connect(seen.append)
    click_text(checklist, "key_3")
    assert checklist.checked() == ["key_0", "key_1", "key_3"]
    click_text(checklist, "key_0")
    assert checklist.checked() == ["key_1", "key_3"]
    assert seen == [["key_0", "key_1", "key_3"], ["key_1", "key_3"]], "one changed per gesture"


def test_down_and_space_toggle_the_next_row(make):
    checklist = make()
    checklist.set_entries(_entries(4), checked=["key_0", "key_1", "key_2", "key_3"], defaults=[])
    seen = []
    checklist.changed.connect(seen.append)
    checklist.view.setFocus()
    checklist.view.setCurrentIndex(_row(checklist, "key_0"))
    QTest.keyClick(checklist.view, Qt.Key_Down)
    QTest.keyClick(checklist.view, Qt.Key_Space)
    assert checklist.view.currentIndex().data(Qt.UserRole) == "key_1"
    assert checklist.checked() == ["key_0", "key_2", "key_3"]
    assert len(seen) == 1


def test_set_checked_and_set_entries_are_silent(make):
    checklist = make()
    seen = []
    checklist.changed.connect(seen.append)
    checklist.set_entries(_entries(4), checked=["key_2"], defaults=[])
    checklist.set_checked(["key_1", "key_3"])
    assert checklist.checked() == ["key_1", "key_3"]
    assert seen == []


def test_the_filter_appears_from_twelve_matches_label_and_key_and_scopes_the_buttons(make):
    short = make()
    short.set_entries(_entries(11), checked=[], defaults=[])
    assert short.filter.isHidden()
    checklist = make()
    entries = [*_entries(9), ("max_dev_pA", "max_dev_pA", ""),
               ("depth_pA", "median depth (depth_pA)", ""), ("snr", "snr", "")]
    keys = [key for key, _label, _tip in entries]
    checklist.set_entries(entries, checked=keys, defaults=keys)
    assert not checklist.filter.isHidden()
    QTest.keyClicks(checklist.filter, "median")           # the label
    assert checklist.shown_keys() == ["depth_pA"]
    checklist.filter.clear()
    QTest.keyClicks(checklist.filter, "DEPTH_PA")          # the key, any case
    assert checklist.shown_keys() == ["depth_pA"]
    checklist.filter.clear()
    QTest.keyClicks(checklist.filter, "_pA")
    assert checklist.shown_keys() == ["max_dev_pA", "depth_pA"]
    assert (checklist.all_btn.text(), checklist.none_btn.text()) == ("Show these", "Hide these")
    seen = []
    checklist.changed.connect(seen.append)
    QTest.mouseClick(checklist.none_btn, Qt.LeftButton)
    assert set(keys) - set(checklist.checked()) == {"max_dev_pA", "depth_pA"}
    assert len(seen) == 1, "a bulk button is one gesture"
    checklist.filter.clear()
    assert (checklist.all_btn.text(), checklist.none_btn.text()) == ("Show all", "Hide all")
    assert checklist.count.text() == "10 of 12 columns shown"


def test_hide_all_keeps_the_keep_key_and_the_last_row_is_held(make):
    checklist = make(minimum=1, keep=("key_3",))
    checklist.set_entries(_entries(5), checked=["key_0", "key_1"], defaults=[])
    QTest.mouseClick(checklist.none_btn, Qt.LeftButton)
    assert checklist.checked() == ["key_3"]
    last = checklist.model.item(3)
    assert not (last.flags() & Qt.ItemIsEnabled)
    assert last.toolTip() == "At least one column stays visible"
    assert checklist.count.text() == "1 of 5 columns shown · at least one column stays visible"
    click_text(checklist, "key_3")
    assert checklist.checked() == ["key_3"], "the last visible row cannot be unticked"
    click_text(checklist, "key_4")
    assert checklist.checked() == ["key_3", "key_4"]
    assert checklist.model.item(3).flags() & Qt.ItemIsEnabled
    assert checklist.model.item(3).toolTip() == "tip 3", "its own tooltip comes back"


def test_a_bounded_pick_refuses_the_one_too_many_and_says_why(make):
    checklist = make(noun="features", minimum=2, maximum=3)
    checklist.set_entries(_entries(5), checked=["key_0", "key_1"], defaults=["key_0", "key_1"])
    assert checklist.count.text() == "2 of 5 selected · pick 2–3"
    assert checklist.all_btn.isHidden(), "Select all past the maximum could only be refused"
    click_text(checklist, "key_4")
    assert checklist.checked() == ["key_0", "key_1", "key_4"]
    assert "untick one to choose another" in checklist.count.text()
    click_text(checklist, "key_2")
    assert checklist.checked() == ["key_0", "key_1", "key_4"]
    held = checklist.model.item(2)
    assert not (held.flags() & Qt.ItemIsEnabled)
    assert held.toolTip() == "At most 3 features: untick one to choose another"
    QTest.mouseClick(checklist.default_btn, Qt.LeftButton)
    assert checklist.checked() == ["key_0", "key_1"]


def test_a_separator_is_not_a_key_and_hides_under_a_filter(make):
    checklist = make(noun="features")
    checklist.set_entries(_entries(13), checked=["key_0"], defaults=[])
    checklist.set_separator_before("key_11", "Identifiers")
    texts = [checklist.model.item(r).text() for r in range(checklist.model.rowCount())]
    assert texts[-3:] == ["Identifiers", "Label 11", "Label 12"]
    assert "Identifiers" not in checklist.checked() and len(checklist.keys()) == 13
    assert checklist.checked() == ["key_0"], "the checked state survives the separator"
    QTest.keyClicks(checklist.filter, "Label 1")
    assert "Identifiers" not in [checklist.proxy.index(r, 0).data()
                                 for r in range(checklist.proxy.rowCount())]


def test_the_popup_stays_open_through_clicks_and_esc_closes_it(qapp, qtbot):
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    layout = QtWidgets.QVBoxLayout(host)
    button = QtWidgets.QPushButton("Columns…")
    layout.addWidget(button)
    layout.addStretch(1)
    host.resize(400, 600)
    host.show()
    qtbot.waitExposed(host)
    popup = ColumnChooserPopup(ColumnChecklist(), host)
    popup.aboutToShow.connect(lambda: popup.checklist.set_entries(
        _entries(6), checked=[k for k, _l, _t in _entries(6)], defaults=[]))
    popup.show_below(button)
    qtbot.waitUntil(popup.isVisible, timeout=2000)
    assert popup.windowFlags() & Qt.Popup
    below = button.mapToGlobal(QPoint(0, button.height()))
    assert popup.pos().y() >= below.y() - 1, "opens below its button"
    for key in ("key_1", "key_3", "key_4"):
        click_text(popup.checklist, key)
        qtbot.wait(20)
        assert popup.isVisible(), f"the popup closed on the click on {key}"
    assert popup.checklist.checked() == ["key_0", "key_2", "key_5"]
    QTest.keyClick(popup.checklist.view, Qt.Key_Escape)
    qtbot.waitUntil(lambda: not popup.isVisible(), timeout=2000)
    assert host.focusWidget() is button, "closing hands the focus back to the button"


def test_the_popup_flips_above_a_button_with_no_room_below(qapp, qtbot):
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    button = QtWidgets.QPushButton("Columns…", host)
    host.show()
    qtbot.waitExposed(host)
    screen = button.screen().availableGeometry()
    host.move(screen.left() + 20, screen.bottom() - host.height() - 5)
    qtbot.wait(20)
    popup = ColumnChooserPopup(None, host)
    popup.checklist.set_entries(_entries(10), checked=[], defaults=[])
    popup.show_below(button)
    qtbot.waitUntil(popup.isVisible, timeout=2000)
    assert popup.geometry().bottom() <= button.mapToGlobal(QPoint(0, 0)).y()
    popup.hide()


def test_the_module_is_a_qt_only_leaf():
    """``panels.py`` may import it only because it imports no sibling gui module."""
    tree = ast.parse(pathlib.Path(column_checklist.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import: from {'.' * node.level}{node.module}"
            assert (node.module or "").split(".")[0] in {"__future__", "PySide6", "collections"}
        elif isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= {"weakref", "PySide6"}
