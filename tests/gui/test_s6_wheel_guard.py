"""Scrolling the page never edits a value widget — clicked or not (user notes #1 and #5).

A two-finger scroll passing over a spinbox or combo used to change its value, and CLICKING
one first used to make that worse rather than better: the old guard exempted the focused
widget, so "click, then scroll the dock" — the exact gesture of the second report ("the
dropdown menus change by scrolling over them. The user needs to click first, then select")
— was the one sequence that could still silently rewrite a merge window or a filter kind.

The app-level ``_WheelGuard`` now refuses the wheel on every QAbstractSpinBox and
QComboBox regardless of focus, and REDIRECTS it to the nearest enclosing scroll area's
viewport so the analysis dock (~140 such widgets) still scrolls under the cursor.
QAbstractSlider is not guarded at all: every one in this app is a QScrollBar.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.gui.app import _WheelGuard
from nanoscope.gui.panels import DetectionPanel


def _wheel(widget):
    ev = QtGui.QWheelEvent(
        QtCore.QPointF(widget.width() / 2, widget.height() / 2),
        widget.mapToGlobal(QtCore.QPoint(widget.width() // 2, widget.height() // 2)),
        QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase, False)
    QtWidgets.QApplication.instance().sendEvent(widget, ev)


def _click_focus(widget, qtbot):
    """Give ``widget`` focus the way a click does, and wait until Qt agrees.

    Qt.MouseFocusReason — not setFocus() with the default reason — because the behaviour
    under test is specifically "the user clicked this control, THEN scrolled".
    """
    widget.window().activateWindow()
    widget.setFocus(QtCore.Qt.MouseFocusReason)
    qtbot.waitUntil(lambda: QtWidgets.QApplication.focusWidget() is widget, timeout=2000)


def _scrollable_panel(qtbot, tab=1):
    """A real DetectionPanel held 200 px tall, so its page scrollers have something to scroll.

    The panel is the app's densest stack of guarded widgets. Since the Inspector
    restructure (c86b119, 2026-09-04) it ships unwrapped: each mode page's settings
    scroll inside their own bounded ``InspectorScrollArea`` (``panel.page_scrolls[page]``),
    and that page scroller, not any area around the panel, is the nearest enclosing
    scroll area of a control like ``k_in``. The outer QScrollArea here only bounds the
    panel's height (the panel's minimum is below 200 px, so the outer area itself gets
    no range).

    ``tab`` raises the page whose widget the caller is about to click: a QTabWidget hides
    the pages it is not showing, and a hidden widget cannot take focus — so "the user
    clicked it" only means something on the visible page. Default 1 = Detection.
    """
    panel = DetectionPanel()
    panel.mode_tabs.setCurrentIndex(tab)
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(panel)
    area.resize(420, 200)
    qtbot.addWidget(area)
    area.show()
    qtbot.waitExposed(area)
    return area, panel


# ================================================================================================
# the guard refuses the wheel — focused or not
# ================================================================================================

def test_unfocused_spinbox_ignores_the_wheel(qapp, qtbot):
    host = QtWidgets.QWidget(); qtbot.addWidget(host)
    lay = QtWidgets.QVBoxLayout(host)
    spin = QtWidgets.QDoubleSpinBox(); spin.setRange(0, 100); spin.setValue(15.0)
    other = QtWidgets.QLineEdit()
    lay.addWidget(spin); lay.addWidget(other)
    host.show(); other.setFocus()
    assert not spin.hasFocus()
    _wheel(spin)
    assert spin.value() == 15.0, "an unfocused spinbox was edited by a passing scroll"


def test_unfocused_combo_ignores_the_wheel(qapp, qtbot):
    host = QtWidgets.QWidget(); qtbot.addWidget(host)
    lay = QtWidgets.QVBoxLayout(host)
    combo = QtWidgets.QComboBox(); combo.addItems(["a", "b", "c"])
    other = QtWidgets.QLineEdit()
    lay.addWidget(combo); lay.addWidget(other)
    host.show(); other.setFocus()
    _wheel(combo)
    assert combo.currentIndex() == 0, "an unfocused combo was changed by a passing scroll"


def test_a_clicked_detection_spinbox_still_refuses_the_wheel(qapp, qtbot):
    """The regression the user reported TWICE.

    This replaces an earlier test that asserted the opposite ("a focused spinbox must keep
    its wheel gesture"). Click-then-scroll is the reported gesture, so focus cannot be the
    permission: on a real DetectionPanel spinbox the threshold must read the same after the
    wheel as before it.
    """
    area, panel = _scrollable_panel(qtbot)
    spin = panel.k_in
    before = spin.value()
    _click_focus(spin, qtbot)
    _wheel(spin)
    assert spin.value() == before, (
        "clicking a spinbox armed the silent edit — a scroll changed the detection threshold")


def test_a_clicked_detection_combo_still_refuses_the_wheel(qapp, qtbot):
    """Same for a dropdown: "the user needs to click first, then select" — click then
    SELECT, from the popup; a scroll after the click must not pick for them."""
    area, panel = _scrollable_panel(qtbot, tab=0)   # the filter combos live on Preprocess
    combo = panel.filter_kind
    before = combo.currentIndex()
    _click_focus(combo, qtbot)
    _wheel(combo)
    assert combo.currentIndex() == before, (
        "clicking a combo armed the silent edit — a scroll changed the filter kind")


# ================================================================================================
# ...and the page scrolls anyway
# ================================================================================================

def test_the_wheel_over_a_spinbox_scrolls_the_enclosing_scroll_area(qapp, qtbot):
    """Refusing the edit must not cost the scroll.

    Each Inspector page's settings scroll in their own bounded scroller, packed with
    guarded spinboxes and combos; a guard that merely swallowed the event would make the
    page unscrollable wherever the pointer happened to land. The event is redirected to
    the nearest enclosing scroll area's viewport instead — for ``k_in`` the Detection
    page's own scroller — so its bar really moves, and the spinbox keeps its value.
    """
    area, panel = _scrollable_panel(qtbot)
    bar = panel.page_scrolls[panel._detection_page].verticalScrollBar()
    qtbot.waitUntil(lambda: bar.maximum() > 0, timeout=2000)
    bar.setValue(0)
    before = panel.k_in.value()
    _click_focus(panel.k_in, qtbot)
    _wheel(panel.k_in)
    assert bar.value() > 0, "the wheel was swallowed — the page stopped scrolling over a spinbox"
    assert panel.k_in.value() == before, "the wheel scrolled the page AND edited the spinbox"


def test_a_wheel_with_no_enclosing_scroll_area_is_simply_refused(qapp, qtbot):
    """The no-scroll-area case: nothing to redirect to, so the event just dies. What must
    not happen is the spinbox seeing it."""
    host = QtWidgets.QWidget(); qtbot.addWidget(host)
    lay = QtWidgets.QVBoxLayout(host)
    spin = QtWidgets.QSpinBox(); spin.setRange(0, 100); spin.setValue(7)
    lay.addWidget(spin)
    host.show(); qtbot.waitExposed(host)
    _click_focus(spin, qtbot)
    _wheel(spin)
    assert spin.value() == 7


# ================================================================================================
# sliders and scrollbars are untouched
# ================================================================================================

def test_sliders_are_not_guarded_at_all(qapp, qtbot):
    """A QScrollBar IS a QAbstractSlider, and it is the only kind this app builds (the
    scroll areas' own bars and the pan bar under the trace). Guarding it — even on focus,
    which a scrollbar never has — is what would break the docks."""
    assert QtWidgets.QAbstractSlider not in _WheelGuard._GUARDED

    bar = QtWidgets.QScrollBar(QtCore.Qt.Vertical); qtbot.addWidget(bar)
    bar.setRange(0, 100); bar.setValue(0)
    bar.show(); qtbot.waitExposed(bar)
    _wheel(bar)
    assert bar.value() > 0, "a scrollbar stopped responding to the wheel"


def test_the_guard_is_installed_once_on_the_application(qapp):
    """One application-level filter is the whole coverage story: QCoreApplication delivers
    every event to it, so widgets Qt builds in C++ are covered too and no panel has to
    remember to opt in."""
    assert isinstance(getattr(qapp, "_nanoscope_wheel_guard", None), _WheelGuard)
