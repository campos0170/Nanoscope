"""Focus follows the pointer onto the trace, the cursor says whether the trace keys are live, and a
caption sits in the target pane.

The rule that matters is who the trace may NOT take focus from. A pointer crossing the trace has
to leave alone anything that owns bare keys: an item view (the Results table's j/k/a/r/f, the Files
tree's arrows), a text or spin field, and the docked Event inspector (Left/Right/A/R/Space and the
label hotkeys), which registers itself with ``TraceView.add_key_owner``. Taking focus from those
would silently kill their keys, and a label hotkey on ``d`` would become an add after the crossing.
It does take focus from nothing, a button, a scroll bar, a closed combo box, a dock.
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.core.layer import ViewMode

from .test_trace_rig import (  # noqa: F401  (fixtures are used by name)
    CURRENT, VOLTAGE, Qt, build_rig, free_mouse, hover, point_at, zoom_middle)


class _Window:
    """A window holding the trace beside every kind of focus holder."""

    def __init__(self, qapp) -> None:
        self.qapp = qapp
        self.window = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.window)
        self.rig = build_rig(qapp, parent=self.window, show=False)
        lay.addWidget(self.rig.view, 1)
        self.button = QtWidgets.QPushButton("a button")
        self.line = QtWidgets.QLineEdit()
        self.spin = QtWidgets.QSpinBox()
        self.table = QtWidgets.QTableWidget(3, 2)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["a", "b"])
        self.editable = QtWidgets.QComboBox()
        self.editable.setEditable(True)
        self.scroll = QtWidgets.QScrollBar()
        self.scroll.setFocusPolicy(Qt.StrongFocus)
        # A stand-in for the docked Event inspector: a plain container whose children are buttons
        # (which the trace WOULD take focus from) -- protected only by being registered.
        self.inspector = QtWidgets.QWidget()
        inner = QtWidgets.QVBoxLayout(self.inspector)
        self.inspector_button = QtWidgets.QPushButton("accept")
        inner.addWidget(self.inspector_button)
        for w in (self.button, self.line, self.spin, self.table, self.combo, self.editable,
                  self.scroll, self.inspector):
            lay.addWidget(w)
        self.window.resize(1000, 900)
        self.window.show()
        self.window.activateWindow()
        qapp.processEvents()

    @property
    def view(self):
        return self.rig.view

    def arm(self) -> None:
        self.rig.panel.set_active(True, seed=False)
        zoom_middle(self.rig, 0.5)
        self.window.activateWindow()
        self.qapp.processEvents()

    def focus(self, widget) -> None:
        widget.setFocus(Qt.OtherFocusReason)
        self.qapp.processEvents()
        assert self.qapp.focusWidget() is not None
        assert not self.view.hasFocus()

    def cross(self, into=None) -> None:
        """Leave the target pane for the other one, then come back (or go to ``into``)."""
        c = sum(self.rig.x_range()) / 2
        hover(self.view, point_at(self.view, self.rig.other_vb, c))
        hover(self.view, point_at(self.view, into or self.rig.target_vb, c))

    def focus_widget(self):
        return self.qapp.focusWidget()


@pytest.fixture
def win(qapp, free_mouse):
    w = _Window(qapp)
    yield w
    w.window.hide()


# ================================================================================================
# who gives up focus
# ================================================================================================
def test_the_pointer_takes_focus_from_nothing(win):
    win.arm()
    win.view.clearFocus()
    win.qapp.processEvents()
    win.cross()
    assert win.view.hasFocus()
    assert win.view.last_focus_reason() == Qt.MouseFocusReason


@pytest.mark.parametrize("holder", ["button", "scroll", "combo"])
def test_the_pointer_takes_focus_from_holders_with_no_keys_of_their_own(win, holder):
    win.arm()
    win.focus(getattr(win, holder))
    win.cross()
    assert win.view.hasFocus(), holder
    assert win.view.last_focus_reason() == Qt.MouseFocusReason


@pytest.mark.parametrize("holder", ["table", "line", "spin", "editable"])
def test_the_pointer_never_takes_focus_from_an_item_view_or_a_text_or_spin_field(win, holder):
    win.arm()
    widget = getattr(win, holder)
    win.focus(widget)
    owner = win.focus_widget()
    win.cross()
    assert not win.view.hasFocus(), holder
    assert win.focus_widget() is owner


def test_the_pointer_never_takes_focus_from_anything_inside_a_registered_key_owner(win):
    win.arm()
    win.view.add_key_owner(win.inspector)
    win.focus(win.inspector_button)
    win.cross()
    assert not win.view.hasFocus() and win.focus_widget() is win.inspector_button
    # Unregistered, the same button is just a button.
    win.view.remove_key_owner(win.inspector)
    win.cross()
    assert win.view.hasFocus()


def test_a_closed_combo_box_gives_way_but_an_open_one_does_not(win):
    win.arm()
    win.focus(win.combo)
    win.combo.showPopup()
    win.qapp.processEvents()
    try:
        assert win.view._focus_holder_is_free(win.combo) is False
    finally:
        win.combo.hidePopup()
    win.qapp.processEvents()
    assert win.view._focus_holder_is_free(win.combo) is True


def test_a_custom_clickable_frame_gives_way_like_the_button_it_is(win):
    """The window starts with focus on a workflow chip: a QFrame with StrongFocus that acts on
    Return/Space. A rule that left unknown widgets alone would not engage until the first click."""
    class Chip(QtWidgets.QFrame):
        def keyPressEvent(self, ev):
            ev.accept()

    chip = Chip()
    chip.setFocusPolicy(Qt.StrongFocus)
    win.window.layout().addWidget(chip)
    win.arm()
    chip.show()
    win.focus(chip)
    win.cross()
    assert win.view.hasFocus()


def test_a_slider_keeps_its_arrow_keys(win):
    slider = QtWidgets.QSlider(Qt.Horizontal)
    win.window.layout().addWidget(slider)
    win.arm()
    slider.show()
    win.focus(slider)
    win.cross()
    assert not win.view.hasFocus()


# ================================================================================================
# when the rule applies at all
# ================================================================================================
def test_nothing_happens_while_the_window_is_not_active(win):
    win.arm()
    win.view.clearFocus()
    other = QtWidgets.QWidget()
    other.show()
    other.activateWindow()
    win.qapp.processEvents()
    if win.window.isActiveWindow():
        pytest.skip("this platform does not deactivate a window when another one is activated")
    win.cross()
    assert not win.view.hasFocus()
    other.hide()


def test_only_the_target_pane_takes_focus(win):
    win.arm()
    win.view.clearFocus()
    c = sum(win.rig.x_range()) / 2
    hover(win.view, point_at(win.view, win.rig.target_vb, c))      # (already inside the target)
    win.view.clearFocus()
    # Cross into the OTHER pane only.
    hover(win.view, QtCore.QPointF(5, 5))
    hover(win.view, point_at(win.view, win.rig.other_vb, c))
    assert not win.view.hasFocus()


def test_measure_off_means_no_target_and_no_focus_but_an_annotation_target_stands_in(win):
    win.view.clearFocus()
    zoom_middle(win.rig, 0.5)
    win.window.activateWindow()
    win.cross()
    assert not win.view.hasFocus()                                  # no Measure, no target
    win.view.set_annotation_target(win.rig.doc_id, CURRENT)
    win.cross()
    assert win.view.hasFocus()
    assert win.view.annotation_target() == (win.rig.doc_id, CURRENT)
    win.view.set_annotation_target(None)
    assert win.view.annotation_target() is None


def test_focus_follows_the_pointer_can_be_switched_off(win):
    win.arm()
    win.view.set_focus_follows_pointer(False)
    win.view.clearFocus()
    win.cross()
    assert not win.view.hasFocus() and win.view.focus_follows_pointer() is False


def test_the_hook_is_re_armed_by_a_rebuild(win):
    win.arm()
    win.view.rebuild()                                              # brand new ViewBoxes
    win.rig.session.set_view_mode(ViewMode.STACKED)
    win.qapp.processEvents()
    win.view.clearFocus()
    win.cross()
    assert win.view.hasFocus()


# ================================================================================================
# the recorded reason
# ================================================================================================
def test_a_click_on_the_trace_records_mouse_focus(win):
    win.arm()
    win.focus(win.button)
    c = sum(win.rig.x_range()) / 2
    p = point_at(win.view, win.rig.other_vb, c).toPoint()           # clicking the OTHER pane too
    # QTest, not sendEvent: Qt hands focus to a clicked widget only for a SPONTANEOUS press.
    QtTest.QTest.mouseClick(win.view.viewport(), Qt.LeftButton, Qt.NoModifier, p)
    win.qapp.processEvents()
    assert win.view.hasFocus() and win.view.last_focus_reason() == Qt.MouseFocusReason


def test_parking_focus_on_the_trace_is_not_a_mouse_focus(win):
    win.arm()
    win.focus(win.button)
    win.view.setFocus()                                             # what MainWindow does when a dock hides
    assert win.view.hasFocus() and win.view.last_focus_reason() == Qt.OtherFocusReason


def test_losing_and_regaining_the_window_does_not_overwrite_a_mouse_focus(win):
    win.arm()
    win.view.clearFocus()
    win.view.setFocus(Qt.MouseFocusReason)
    assert win.view.last_focus_reason() == Qt.MouseFocusReason
    out = QtGui.QFocusEvent(QtCore.QEvent.FocusOut, Qt.ActiveWindowFocusReason)
    QtWidgets.QApplication.sendEvent(win.view, out)
    back = QtGui.QFocusEvent(QtCore.QEvent.FocusIn, Qt.ActiveWindowFocusReason)
    QtWidgets.QApplication.sendEvent(win.view, back)
    assert win.view.last_focus_reason() == Qt.MouseFocusReason
    # ...but the SAME window event after focus had been parked is a real change of reason.
    win.view.clearFocus()
    win.view.setFocus(Qt.TabFocusReason)
    assert win.view.last_focus_reason() == Qt.TabFocusReason


def test_the_reason_starts_out_unknown(qapp, free_mouse):
    rig = build_rig(qapp, show=False)                               # never shown: never focused
    assert rig.view.last_focus_reason() is None


# ================================================================================================
# the cursor
# ================================================================================================
def _vb_cursor(vb):
    return vb.cursor().shape() if vb.hasCursor() else None


def test_the_target_pane_shows_a_cross_while_the_trace_has_focus_and_an_arrow_otherwise(win):
    win.arm()
    win.view.setFocus(Qt.MouseFocusReason)
    win.qapp.processEvents()
    assert _vb_cursor(win.rig.target_vb) == Qt.CrossCursor
    assert _vb_cursor(win.rig.other_vb) is None                      # the other pane keeps its own
    win.focus(win.line)
    assert _vb_cursor(win.rig.target_vb) == Qt.ArrowCursor
    assert _vb_cursor(win.rig.other_vb) is None
    win.view.setFocus(Qt.MouseFocusReason)
    assert _vb_cursor(win.rig.target_vb) == Qt.CrossCursor


def test_the_cursor_is_re_applied_after_a_rebuild(win):
    win.arm()
    win.view.setFocus(Qt.MouseFocusReason)
    win.rig.session.set_view_mode(ViewMode.STACKED)
    win.qapp.processEvents()
    assert _vb_cursor(win.rig.target_vb) == Qt.CrossCursor
    win.view.rebuild()
    assert _vb_cursor(win.rig.target_vb) == Qt.CrossCursor
    assert _vb_cursor(win.rig.other_vb) is None


def test_there_is_no_cursor_change_without_a_target(win):
    win.view.setFocus(Qt.MouseFocusReason)
    win.qapp.processEvents()
    assert all(_vb_cursor(p.plot.vb) is None for p in win.view.panes())
    win.arm()
    assert _vb_cursor(win.rig.target_vb) == Qt.CrossCursor
    win.rig.panel.set_active(False)
    assert all(_vb_cursor(p.plot.vb) is None for p in win.view.panes())


def test_the_viewport_cursor_is_the_cross_over_the_target_pane_only(win):
    win.arm()
    win.view.setFocus(Qt.MouseFocusReason)
    win.qapp.processEvents()
    c = sum(win.rig.x_range()) / 2
    hover(win.view, point_at(win.view, win.rig.target_vb, c))
    hover(win.view, point_at(win.view, win.rig.target_vb, c) + QtCore.QPointF(3, 2))
    assert win.view.viewport().cursor().shape() == Qt.CrossCursor
    hover(win.view, point_at(win.view, win.rig.other_vb, c))
    hover(win.view, point_at(win.view, win.rig.other_vb, c) + QtCore.QPointF(3, 2))
    assert win.view.viewport().cursor().shape() == Qt.ArrowCursor


# ================================================================================================
# the caption
# ================================================================================================
def _label(view):
    return view._target_label


def test_the_caption_sits_bottom_left_in_the_target_pane(win):
    win.arm()
    win.view.set_target_label("→ Manual · 37 hand-made · 2 in view")
    win.qapp.processEvents()
    label = _label(win.view)
    vb = win.rig.target_vb
    assert label is not None and label.text() == "→ Manual · 37 hand-made · 2 in view"
    assert label.parentItem() is vb and label.isVisible()
    r = label.mapRectToItem(vb, label.boundingRect())
    assert 0 <= r.left() <= 16 and r.bottom() <= vb.height() and vb.height() - r.bottom() <= 16
    assert r.top() > vb.height() / 2                                  # the top edge is the crosshair's
    assert label not in vb.addedItems                                  # never part of the autorange
    assert win.view.target_label_text() == "→ Manual · 37 hand-made · 2 in view"
    assert win.rig.other_vb.childItems().count(label) == 0


def test_an_empty_caption_hides_it_and_a_new_text_updates_it_in_place(win):
    win.arm()
    win.view.set_target_label("→ Manual")
    first = _label(win.view)
    win.view.set_target_label("→ Schmitt · 4")
    assert _label(win.view) is first and first.text() == "→ Schmitt · 4"
    win.view.set_target_label("")
    assert _label(win.view) is None and first.scene() is None
    win.view.set_target_label("→ Manual")
    assert _label(win.view) is not None


def test_the_caption_survives_a_rebuild_and_follows_the_target(win):
    win.arm()
    win.view.set_target_label("→ Manual")
    win.rig.session.set_view_mode(ViewMode.STACKED)
    win.qapp.processEvents()
    label = _label(win.view)
    assert label is not None and label.parentItem() is win.rig.target_vb and label.isVisible()
    # Re-targeted to the other pane, it goes with it.
    win.rig.view.set_measure_active(win.rig.doc_id, VOLTAGE, True, seed=False)
    label = _label(win.view)
    assert label.parentItem() is win.rig.other_vb
    win.rig.panel.set_active(False)
    assert _label(win.view) is None                                    # no target, no caption
    win.rig.panel.set_active(True, seed=False)


def test_the_caption_is_transparent_to_the_mouse(win):
    win.arm()
    win.view.set_target_label("→ Manual")
    label = _label(win.view)
    assert label.acceptedMouseButtons() == Qt.NoButton
    assert label.shape().isEmpty()
    assert not label.acceptHoverEvents()


def test_the_caption_actually_paints(win):
    """Rendering must not raise inside ``paint`` (Qt would swallow it into stderr and the test
    would never know): grab the view with and without the caption and compare its rectangle."""
    win.arm()
    win.qapp.processEvents()
    vb = win.rig.target_vb

    def grab_rect(with_label: bool):
        win.view.set_target_label("→ Manual · 37 hand-made" if with_label else "")
        win.qapp.processEvents()
        label = _label(win.view)
        if label is None:                                       # (the empty case has no item)
            r = vb.mapRectToScene(QtCore.QRectF(8, vb.height() - 30, 150, 22))
        else:
            r = label.mapRectToScene(label.boundingRect())
        box = win.view.mapFromScene(r).boundingRect()
        return win.view.grab().toImage().copy(box), box

    without, box = grab_rect(False)
    with_label, box2 = grab_rect(True)
    assert box2.width() > 20 and box2.height() > 8
    w = min(without.width(), with_label.width(), box2.width())
    h = min(without.height(), with_label.height(), box2.height())
    differing = sum(1 for x in range(w) for y in range(h) if without.pixel(x, y) != with_label.pixel(x, y))
    assert differing > 0.25 * w * h                              # a pill with text, not nothing
