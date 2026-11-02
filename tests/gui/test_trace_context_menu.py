"""The trace menu: a right click on the target pane offers Add event here, Add the selection,
Discard, Delete this hand-made event and the Annotate switch, each with its key.

A real, shown MainWindow on a synthetic recording (``test_trace_annotate_mode``'s). The menu is
shown with ``popup``, which the tests replace with a recorder: a visible popup would be closed by
the suite's modal net, and the items are triggered directly, as a click on one would.
"""

from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from .test_trace_annotate_mode import (  # noqa: F401  (rig and free_mouse are fixtures)
    EVENT, SECOND, Qt, armed, at, drag, drag_move, free_mouse, hover, press, release, rig, rows,
    settle, table, target_vb, zoom)
from .test_trace_rig import double_click


def _items(menu) -> dict[str, QtWidgets.QAction]:
    return {a.objectName(): a for a in menu.actions() if a.objectName()}


def _right_click(view, pos) -> None:
    hover(view, pos)
    press(view, pos, button=Qt.RightButton)
    release(view, pos, button=Qt.RightButton)


class _Popups:
    def __init__(self, menu) -> None:
        self.calls: list[QtCore.QPoint] = []
        self.menu = menu

    def __call__(self, pos, *_args) -> None:
        self.calls.append(pos)


def test_a_right_click_on_the_target_pane_opens_the_trace_menu(rig, qtbot, monkeypatch):
    win, doc_id = rig
    menu = win._trace_keys._menu
    popups = _Popups(menu)
    monkeypatch.setattr(menu, "popup", popups)
    zoom(win, doc_id, 19_000, 21_500)
    _right_click(win._trace_view, at(win, doc_id, 20_200))
    assert len(popups.calls) == 1
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts == ["Add event here\tC", "Add the selection as an event\tD",
                     "Discard the selection\tX", "Delete this hand-made event",
                     "Annotate: drag adds an event\tE"]
    items = _items(menu)
    # No mask yet: only what can act is enabled.
    assert items["traceMenuAddHere"].isEnabled()
    assert not items["traceMenuAddSelection"].isEnabled()
    assert not items["traceMenuDiscard"].isEnabled()
    assert not items["traceMenuDeleteHandMade"].isEnabled()
    assert not win._analysis_panel.measure_panel.is_active()     # a menu changes nothing


def _on_axis(win, doc_id, side: str, sample: int) -> QtCore.QPointF:
    """A viewport point on the target pane's ``side`` axis: the middle of the left axis, or the
    bottom axis right under ``sample``. Asserted to be on the axis and outside the plot area."""
    tv = win._trace_view
    axis = tv.pane_for_layer(f"{doc_id}#0").plot.getAxis(side)
    rect = axis.mapRectToScene(axis.rect())
    if side == "left":
        point = rect.center()
    else:
        sx = target_vb(win, doc_id).mapViewToScene(
            QtCore.QPointF(tv.x_for_index(doc_id, 0, sample), 0.0)).x()
        point = QtCore.QPointF(sx, rect.center().y())
    assert axis.isVisible() and axis in tv.scene().items(point)
    assert not target_vb(win, doc_id).sceneBoundingRect().contains(point)
    return QtCore.QPointF(tv.mapFromScene(point))


@pytest.mark.parametrize("side", ["left", "bottom"])
def test_a_right_click_on_an_axis_opens_no_menu(rig, monkeypatch, side):
    """pyqtgraph's axis hands its clicks to the plot's view box, at a point off the trace: on the
    left axis "Add event here" refused and turned Measure on, and "Delete this hand-made event"
    acted on an event the pointer was not over. An axis right-click does nothing, as before the
    menu; the plot area right above it still opens the menu."""
    win, doc_id = rig
    menu = win._trace_keys._menu
    popups = _Popups(menu)
    monkeypatch.setattr(menu, "popup", popups)
    zoom(win, doc_id, 19_000, 21_500)
    middle = (EVENT[0] + EVENT[1]) // 2
    _right_click(win._trace_view, _on_axis(win, doc_id, side, middle))
    assert popups.calls == []
    assert rows(win, doc_id) == 0
    assert not win._analysis_panel.measure_panel.is_active()
    _right_click(win._trace_view, at(win, doc_id, middle))
    assert len(popups.calls) == 1


@pytest.mark.parametrize("side", ["left", "bottom"])
def test_a_right_click_on_an_axis_still_cancels_a_draw(rig, monkeypatch, side):
    """Mid-draw the scene gives every click to the view box that owns the drag, wherever it
    lands: the cancel is not limited to the plot area."""
    win, doc_id = rig
    menu = win._trace_keys._menu
    popups = _Popups(menu)
    monkeypatch.setattr(menu, "popup", popups)
    tv = win._trace_view
    zoom(win, doc_id, 19_000, 21_500)
    win._trace_keys.set_annotate(True)
    p0, p1 = at(win, doc_id, EVENT[0] - 60), at(win, doc_id, EVENT[1] + 70)
    drag(tv, p0, p1, release_it=False)
    assert tv.draw_in_progress()
    on_axis = _on_axis(win, doc_id, side, EVENT[1] + 70)
    press(tv, on_axis, button=Qt.RightButton)
    release(tv, on_axis, button=Qt.RightButton)
    assert not tv.draw_in_progress()
    release(tv, p1)
    assert popups.calls == [] and rows(win, doc_id) == 0


@pytest.mark.parametrize("side", ["left", "bottom"])
def test_a_double_click_on_an_axis_adds_nothing_in_annotate_mode(rig, qtbot, side):
    """The axis hands its double-click to the view box too: in Annotate mode a double-click on
    the bottom axis under an event caught and added it, and one on the left axis caught at a
    sample just left of the view. Only a double-click on the trace catches; on the trace right
    above it still adds the event."""
    win, doc_id = rig
    armed(win, doc_id, 19_000, 21_500)
    middle = (EVENT[0] + EVENT[1]) // 2
    double_click(win._trace_view, _on_axis(win, doc_id, side, middle))
    settle(qtbot)
    assert rows(win, doc_id) == 0
    assert win._trace_view.measure_bounds() is None
    double_click(win._trace_view, at(win, doc_id, middle))
    settle(qtbot)
    assert rows(win, doc_id) == 1


def test_add_event_here_adds_one_row_with_measure_off(rig, qtbot):
    win, doc_id = rig
    assert not win._analysis_panel.measure_panel.is_active()
    menu = win._trace_keys.context_menu_for(doc_id, 0, 20_200)
    _items(menu)["traceMenuAddHere"].trigger()
    df = table(win, doc_id)
    assert df is not None and len(df) == 1
    assert abs(int(df.iloc[0]["start_idx"]) - EVENT[0]) <= 3
    assert abs(int(df.iloc[0]["end_idx"]) - EVENT[1]) <= 3
    assert df.iloc[0]["bounds_source"] == "start=fit:step;end=fit:step"
    win._act_undo.trigger()
    settle(qtbot)
    assert rows(win, doc_id) == 0


def test_add_event_here_on_baseline_says_no_event_and_adds_nothing(rig):
    win, doc_id = rig
    menu = win._trace_keys.context_menu_for(doc_id, 0, 40_000)
    _items(menu)["traceMenuAddHere"].trigger()
    assert rows(win, doc_id) == 0
    assert "No event under the pointer" in win.statusBar().currentMessage()


def test_delete_this_hand_made_event_deletes_one_row_in_one_undo_step(rig, qtbot):
    win, doc_id = rig
    keys = win._trace_keys
    _items(keys.context_menu_for(doc_id, 0, 20_200))["traceMenuAddHere"].trigger()
    _items(keys.context_menu_for(doc_id, 0, 60_100))["traceMenuAddHere"].trigger()
    assert rows(win, doc_id) == 2
    # Over baseline the item is there but disabled; over a hand-made row it deletes that row.
    assert not _items(keys.context_menu_for(doc_id, 0, 40_000))["traceMenuDeleteHandMade"].isEnabled()
    delete = _items(keys.context_menu_for(doc_id, 0, 20_200))["traceMenuDeleteHandMade"]
    assert delete.isEnabled()
    delete.trigger()
    settle(qtbot)
    df = table(win, doc_id)
    assert len(df) == 1 and abs(int(df.iloc[0]["start_idx"]) - SECOND[0]) <= 3
    win._act_undo.trigger()                                      # one step brings it back
    settle(qtbot)
    assert rows(win, doc_id) == 2


def test_delete_is_disabled_over_a_detected_event(rig, qtbot):
    win, doc_id = rig
    keys = win._trace_keys
    _items(keys.context_menu_for(doc_id, 0, 20_200))["traceMenuAddHere"].trigger()
    df = table(win, doc_id)
    assert keys._hand_made_at(doc_id, 0, 20_200) == int(df.iloc[0]["event_id"])
    # The same row, called detected: the lookup asks the one hand-made rule, not the span.
    df.loc[:, "source"] = "detected"
    if "added" in df.columns:
        df.loc[:, "added"] = False
    assert keys._hand_made_at(doc_id, 0, 20_200) is None
    assert not _items(keys.context_menu_for(doc_id, 0, 20_200))["traceMenuDeleteHandMade"].isEnabled()


def test_the_selection_items_follow_the_mask(rig):
    win, doc_id = rig
    keys = win._trace_keys
    win._trace_view.pointer_sample = lambda: (doc_id, 0, 20_200.0, 10.0)
    keys.catch()
    items = _items(keys.context_menu_for(doc_id, 0, 30_000))
    assert items["traceMenuAddSelection"].isEnabled() and items["traceMenuDiscard"].isEnabled()
    items["traceMenuDiscard"].trigger()
    assert win._trace_view.measure_empty() and rows(win, doc_id) == 0
    keys.catch()
    _items(keys.context_menu_for(doc_id, 0, 30_000))["traceMenuAddSelection"].trigger()
    assert rows(win, doc_id) == 1


def test_the_menu_annotate_item_checks_the_scan_strip_button(rig):
    win, doc_id = rig
    item = _items(win._trace_keys.context_menu_for(doc_id, 0, 30_000))["traceMenuAnnotate"]
    assert item.isCheckable() and not item.isChecked()
    item.trigger()
    assert win._trace_keys.annotate() and win._annotate_button.isChecked()
    assert _items(win._trace_keys.context_menu_for(doc_id, 0, 30_000))["traceMenuAnnotate"].isChecked()


def test_a_right_drag_still_scales_and_opens_no_menu(rig, monkeypatch):
    win, doc_id = rig
    menu = win._trace_keys._menu
    popups = _Popups(menu)
    monkeypatch.setattr(menu, "popup", popups)
    tv = win._trace_view
    zoom(win, doc_id, 19_000, 21_500)
    before = tuple(tv.anchor_plot().vb.viewRange()[0])
    p0, p1 = at(win, doc_id, 20_000), at(win, doc_id, 20_800)
    hover(tv, p0)
    press(tv, p0, button=Qt.RightButton)
    for k in range(1, 9):
        drag_move(tv, QtCore.QPointF(p0.x() + (p1.x() - p0.x()) * k / 8, p0.y()),
                  buttons=Qt.RightButton)
    release(tv, p1, button=Qt.RightButton)
    assert popups.calls == []
    assert tuple(tv.anchor_plot().vb.viewRange()[0]) != before


def test_a_right_click_during_a_draw_still_cancels_it(rig, monkeypatch):
    win, doc_id = rig
    menu = win._trace_keys._menu
    popups = _Popups(menu)
    monkeypatch.setattr(menu, "popup", popups)
    tv = win._trace_view
    zoom(win, doc_id, 19_000, 21_500)
    win._trace_keys.set_annotate(True)
    p0, p1 = at(win, doc_id, EVENT[0] - 60), at(win, doc_id, EVENT[1] + 70)
    drag(tv, p0, p1, release_it=False)
    assert tv.draw_in_progress()
    press(tv, p1, button=Qt.RightButton, )
    release(tv, p1, button=Qt.RightButton)
    assert not tv.draw_in_progress()
    release(tv, p1)
    assert popups.calls == [] and rows(win, doc_id) == 0
