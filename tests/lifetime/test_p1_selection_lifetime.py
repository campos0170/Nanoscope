"""Selection and the Format tab under the CYCLIC GC, driven with REAL mouse events.

The selection touches objects that are created and destroyed around it: the element list's rows
(rebuilt when the set of elements changes), the outline and the mini toolbar (children of a view's
viewport, built lazily, retired with ``hide()`` then ``deleteLater()``), the legend rows a page
redraws, and the page itself (closed, removed, deleted while something is selected). A slot that
destroys what Qt still holds is the shape of every crash this suite exists for (see
``conftest.py``): so the list is never rebuilt inside its own selection slot, nothing keeps a row
or an item between rebuilds, and every later use of a retired widget is guarded.

Real clicks through the QWINDOW of the Explorer; the list is clicked on its viewport. Synthetic
data only; offscreen.
"""
from __future__ import annotations

import gc

import pytest
from PySide6 import QtCore, QtWidgets

from gui import p1_format_rig as F
from gui import p1_hosting as H
from nanoscope.gui import figure_text
from nanoscope.gui.format_inspector import ElementRef


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _settle(turns: int = 3) -> None:
    for _ in range(turns):
        QtWidgets.QApplication.processEvents()
    gc.collect()


def _rig(qtbot):
    page, explorer = F.hosted_rich(qtbot)
    explorer.inspector().set_tab(1)
    _settle()
    return page, explorer, explorer.inspector().pane("iv")


KEYS = ["title", "axis.bottom.label", "axis.left.ticks", "legend.entry.second", "annotation.fit",
        F.CAPTION, F.PANEL]


def test_selecting_and_clearing_by_real_clicks_under_redraws_raises_nothing(qtbot):
    page, explorer, pane = _rig(qtbot)
    plot, widget = page._plot, page._appearance_widget
    vb = plot.getViewBox().sceneBoundingRect()
    for turn in range(24):
        figure_text.GATE.reset()
        F.click_scene(plot, F.point_of(plot, KEYS[turn % len(KEYS)]))
        assert pane.hub.current() is not None
        if turn % 3 == 0:                                   # a redraw with the selection standing
            widget.apply_appearance(widget.state().__class__.from_mapping(
                {**widget.state().to_dict(), "title_size": 10 + turn % 5}))
        if turn % 4 == 3:
            F.click_scene(plot, vb.center() + QtCore.QPointF(0, 40))
            assert pane.hub.current() is None
        _settle(2)


def test_picking_rows_by_real_clicks_while_the_list_is_rebuilt_raises_nothing(qtbot):
    page, explorer, pane = _rig(qtbot)
    fp = pane.format_pane
    legend = page._plot.legend
    for turn in range(16):
        refs = [ref for ref in fp.elements() if ref.key != "figure"]
        F.pick_row(fp.list, refs[turn % len(refs)])
        if turn % 4 == 1:                                   # entries come and go behind the list
            legend.removeItem("second")
            page._plot.removeItem(page._second)
            fp.refresh()
        if turn % 4 == 3:
            page._second = page._plot.plot([0, 1, 2], [1.0, 2.0, 1.0], name="second")
            legend.addItem(page._second, "second")
            page.controller.refresh()
            fp.refresh()
        _settle()
    assert fp.list.ref_of(fp.list.currentItem()) == pane.hub.current()


def test_removing_the_page_with_a_selection_a_toolbar_and_an_outline_raises_nothing(qtbot):
    page, explorer, pane = _rig(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    controller = page.controller
    assert controller._nanoscope_minibar.isVisible()
    explorer.remove_page("iv")
    _settle()
    page.close()
    page.deleteLater()
    _settle(4)
    assert explorer.inspector().pane("iv") is None


def test_retiring_the_overlays_by_hand_leaves_later_selections_working(qtbot):
    page, explorer, pane = _rig(qtbot)
    fp = pane.format_pane
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    old = page.controller._nanoscope_minibar
    fp.retire_overlays()                                    # hide(), then deleteLater()
    _settle(4)
    assert not figure_text._alive(old) or not old.isVisible()
    figure_text.GATE.reset()
    F.click_scene(page._plot, F.point_of(page._plot, "axis.bottom.label"))
    assert pane.hub.current() == ElementRef("", "axis.bottom.label")
    assert page.controller._nanoscope_minibar is not old and page.controller._nanoscope_minibar.isVisible()


def test_two_pages_selected_in_turn_through_tab_switches_raise_nothing(qtbot):
    first, explorer = F.hosted_rich(qtbot, "iv")
    second = F.RichPage("voltage_series")
    explorer.show_page("voltage_series", second)
    second.type_words()
    for turn in range(10):
        page, key = (first, "iv") if turn % 2 == 0 else (second, "voltage_series")
        explorer.raise_page(key)
        _settle(2)
        figure_text.GATE.reset()
        F.click_scene(page._plot, F.point_of(page._plot, KEYS[turn % 3]))
        assert explorer.inspector().selection_hub(key).current() is not None
    explorer.remove_page("voltage_series")
    second.deleteLater()
    _settle(4)
    H.click(explorer.toolbar().reset_figure_button)
