"""Phase 1, Format group: selecting an element of the figure, from the canvas and from the list.

Every gesture is a real one (clicks through the QWINDOW of the view's top-level window, a pan with
the button held, a click on a row of the element list); the page is hosted by a real
``AnalysisExplorer``. What is asserted is what is SELECTED, SHOWN and DRAWN. Synthetic data only;
offscreen.
"""
from __future__ import annotations

import ast
import os
import pathlib
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets
from pyqtgraph.exporters import ImageExporter

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import figure_text_for
from nanoscope.gui.format_inspector import ElementRef
from nanoscope.gui.format_toolbar import MiniTextToolbar, SelectionOutline

from . import p1_format_rig as F
from . import p1_hosting as H


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _rig(qtbot, **options):
    page, explorer = F.hosted_rich(qtbot, **options)
    explorer.inspector().set_tab(0)
    QtWidgets.QApplication.processEvents()
    pane = explorer.inspector().pane("iv")
    return page, explorer, pane, pane.format_pane


def _outline(page):
    return getattr(page.controller, "_nanoscope_outline", None)


def _bar(page):
    return getattr(page.controller, "_nanoscope_minibar", None)


KINDS = [("title", "text"), ("axis.bottom.label", "text"), ("axis.left.label", "text"),
         ("axis.bottom.ticks", "text"), ("axis.left.ticks", "text"),
         ("legend.entry.series", "text"), ("legend", "legend"), ("annotation.fit", "fit_label"),
         (F.CAPTION, "caption"), (F.PANEL, "text")]


# -- canvas to list ---------------------------------------------------------------------------------

@pytest.mark.parametrize("key,kind", KINDS)
def test_a_click_on_an_element_selects_it_shows_its_panel_and_the_format_tab(qtbot, key, kind):
    page, explorer, pane, fp = _rig(qtbot)
    assert explorer.inspector().current_tab() == 0 and pane.hub.current() is None
    F.click_scene(page._plot, F.point_of(page._plot, key))
    ref = ElementRef("", key)
    assert pane.hub.current() == ref
    assert explorer.inspector().current_tab() == 1                   # "Format follows what you click"
    assert fp.list.ref_of(fp.list.currentItem()) == ref              # the list follows
    assert fp.current_panel().kind == kind and fp.current_panel().isVisible()
    assert _outline(page) is not None and _outline(page).isVisible()
    assert _outline(page).key() == key
    if key != "legend":                                              # the frame has no text of its own
        assert _bar(page) is not None and _bar(page).isVisible() and _bar(page).key() == key


def test_a_right_click_selects_and_still_raises_the_same_menu(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    controller = page.controller
    controller.last_menu = None
    F.click_scene(page._plot, F.point_of(page._plot, "title"), button=F.R)
    assert pane.hub.current() == ElementRef("", "title")
    menu = controller.last_menu
    assert menu is not None
    names = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert names[:2] == ["Edit title…", "Text style…"], names
    menu.close()


def test_a_double_click_still_opens_the_inline_editor(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    controller = page.controller
    opened = []
    controller.opened.connect(lambda kind, key: opened.append((kind, key)))
    F.click_scene(page._plot, F.point_of(page._plot, "title"), double=True)
    assert opened == [("inline", "title")]
    assert pane.hub.current() == ElementRef("", "title")             # the first click selected it
    assert figure_text._alive(controller.editor) and controller.editor.isVisible()
    assert not _bar(page).isVisible()                                # the toolbar steps aside
    controller.close_editors()


def test_a_double_click_on_the_numbers_still_opens_axis_properties(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    controller = page.controller
    opened = []
    controller.opened.connect(lambda kind, key: opened.append((kind, key)))
    F.click_scene(page._plot, F.point_of(page._plot, "axis.left.ticks"), double=True)
    assert opened == [("axis", "axis.left.ticks")]
    assert pane.hub.current() == ElementRef("", "axis.left.ticks")
    controller.close_editors()


def test_a_legend_sample_click_still_hides_the_series_and_is_not_a_selection(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    assert pane.hub.current() is not None
    item = page._plot.listDataItems()[0]
    sample, _label = page._plot.legend.items[0]
    mark = page._appearance_widget.history_mark()
    assert item.isVisible()
    legend = page.controller.anchor_rect("legend")
    assert legend is not None and not legend.isEmpty()
    assert not _bar(page).geometry().intersects(legend)
    F.click_scene(page._plot, sample.sceneBoundingRect().center())
    assert not item.isVisible()                                      # pyqtgraph's own behaviour
    assert page._appearance_widget.history_mark() == mark            # outside every history
    assert pane.hub.current() is None                                # nobody claimed that click


def test_a_click_on_data_never_selects_a_series(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    vb = page._plot.getViewBox()
    on_line = vb.mapViewToScene(QtCore.QPointF(1.0, 2.0))
    F.click_scene(page._plot, on_line)
    assert pane.hub.current() is None                                # series come from the list


def test_dragging_the_fit_label_selects_it_and_the_outline_follows_it_there(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    plot = page._plot
    start = F.point_of(plot, "annotation.fit")
    F.drag_scene(plot, start, delta=(90, 60))
    assert page._appearance_widget.state().fit_label_state().placement == "custom"   # it moved
    assert pane.hub.current() == ElementRef("", "annotation.fit")
    wanted = page.controller.anchor_rect("annotation.fit")
    room = F.view_of(plot).viewport().rect()
    assert _outline(page).geometry() == wanted.adjusted(-2, -2, 2, 2).intersected(room)


# -- clearing ---------------------------------------------------------------------------------------

def test_a_click_on_empty_plot_clears_the_selection_the_outline_and_the_toolbar(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    assert pane.hub.current() is not None and _outline(page).isVisible() and _bar(page).isVisible()
    vb = page._plot.getViewBox().sceneBoundingRect()
    F.click_scene(page._plot, vb.center() + QtCore.QPointF(0, 40))
    assert pane.hub.current() is None
    assert not _outline(page).isVisible() and not _bar(page).isVisible()
    assert fp.list.currentItem() is None
    assert explorer.inspector().current_tab() == 1                   # nothing switches back
    assert fp.current_panel().kind == "figure"


def test_a_pan_keeps_the_selection(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, "axis.left.label"))
    before = page._plot.viewRange()
    F.drag_scene(page._plot, page._plot.getViewBox().sceneBoundingRect().center())
    assert page._plot.viewRange() != before                          # it really panned
    assert pane.hub.current() == ElementRef("", "axis.left.label")
    assert _outline(page).isVisible()


def test_a_click_that_one_element_claimed_is_not_a_clear(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    assert _bar(page).isVisible()
    for key in ("axis.bottom.label", "legend", F.PANEL):
        anchor = page.controller.anchor_rect(key)
        assert anchor is not None and not _bar(page).geometry().intersects(anchor)
    F.click_scene(page._plot, F.point_of(page._plot, "axis.bottom.label"))
    assert pane.hub.current() == ElementRef("", "axis.bottom.label")
    F.click_scene(page._plot, F.point_of(page._plot, F.PANEL))      # an EditableLabelItem
    assert pane.hub.current() == ElementRef("", F.PANEL)


# -- list to canvas ---------------------------------------------------------------------------------

def test_a_list_pick_outlines_the_element_shows_the_panel_and_leaves_the_tab_alone(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    explorer.inspector().set_tab(1)
    QtWidgets.QApplication.processEvents()
    F.pick_row(fp.list, ElementRef("", "axis.bottom.label"))        # a real click on the row
    assert pane.hub.current() == ElementRef("", "axis.bottom.label")
    assert fp.current_panel().kind == "text" and explorer.inspector().current_tab() == 1
    outline = _outline(page)
    assert outline is not None and outline.isVisible() and outline.key() == "axis.bottom.label"
    wanted = page.controller.anchor_rect("axis.bottom.label")
    room = F.view_of(page._plot).viewport().rect()
    assert outline.geometry() == wanted.adjusted(-2, -2, 2, 2).intersected(room)


def test_picking_an_element_from_code_while_the_data_tab_shows_does_not_switch_tabs(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    fp.show_element("axis.left")
    assert explorer.inspector().current_tab() == 0                   # a list pick never changes tabs
    assert pane.hub.current() == ElementRef("", "axis.left")


def test_the_whole_axis_and_the_legend_frame_are_outlined(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    explorer.inspector().set_tab(1)
    for key in ("axis.left", "legend"):
        fp.show_element(key)
        QtWidgets.QApplication.processEvents()
        outline = _outline(page)
        assert outline.isVisible() and outline.key() == key
    assert _bar(page) is None                    # an axis and the frame have no text to format here
    fp.show_element("series.data")
    assert not _outline(page).isVisible()        # nothing to outline


def test_the_list_holds_every_element_and_lists_the_series_by_name(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    assert [ref.key for ref in fp.elements()] == [
        "figure", "title", "axis.bottom", "axis.bottom.label", "axis.bottom.ticks", "axis.left",
        "axis.left.label", "axis.left.ticks", "legend", "legend.entry.series",
        "legend.entry.second", "series.data", "annotation.fit", F.CAPTION, F.PANEL]
    assert pane.hub.refs() == []         # not built yet: the Format tab was never shown
    explorer.inspector().set_tab(1)
    assert [r.key for r in pane.hub.refs()] == [r.key for r in fp.elements()]


# -- the outline and the toolbar are not part of the figure ------------------------------------------

def test_the_outline_and_the_toolbar_are_in_no_export_but_are_on_screen(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    plot = page._plot
    plain = ImageExporter(plot)
    plain.parameters()["width"] = 600
    before = plain.export(toBytes=True)
    screen_before = F.view_of(plot).viewport().grab().toImage()
    F.click_scene(plot, F.point_of(plot, "title"))
    assert _outline(page).isVisible() and _bar(page).isVisible()
    exporter = ImageExporter(plot)
    exporter.parameters()["width"] = 600
    after = exporter.export(toBytes=True)
    assert after == before                                           # not one pixel of either
    assert F.view_of(plot).viewport().grab().toImage() != screen_before   # but they ARE drawn
    assert not any(isinstance(i, (SelectionOutline, MiniTextToolbar)) for i in plot.scene().items()
                   if hasattr(i, "isWidget"))


def test_a_selection_and_the_toolbar_do_not_change_what_live_composition_captures(qtbot):
    from dataclasses import replace

    from nanoscope.gui import figure_composition as C

    window = F.bare(qtbot)
    plot = window.plots[0]
    window.widget.apply_appearance(replace(window.state, title="A title"))
    QtWidgets.QApplication.processEvents()
    before = C.capture_cartesian_model(plot)
    assert before is not None                                        # a plot that is composable
    F.click_scene(plot, F.point_of(plot, "title"))
    controller = window.controllers[0]
    assert controller._nanoscope_minibar.isVisible() and controller._nanoscope_outline.isVisible()
    assert window.hub.current() == ElementRef("", "title")
    assert C.capture_cartesian_model(plot) == before


# -- after redraws, and when elements go -------------------------------------------------------------

def test_the_selection_survives_a_redraw_and_follows_the_element(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    widget = page._appearance_widget
    widget.apply_appearance(page._appearance.__class__.from_mapping(
        {**widget.state().to_dict(), "title_size": 20}))
    QtWidgets.QApplication.processEvents()
    assert pane.hub.current() == ElementRef("", "title")
    wanted = page.controller.anchor_rect("title")
    assert _outline(page).isVisible()
    assert _outline(page).geometry().contains(wanted.adjusted(1, 1, -1, -1))


def test_a_legend_entry_that_vanishes_falls_back_to_the_legend_then_the_figure(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    explorer.inspector().set_tab(1)
    F.click_scene(page._plot, F.point_of(page._plot, "legend.entry.second"))
    assert pane.hub.current() == ElementRef("", "legend.entry.second")
    page._plot.legend.removeItem("second")
    page._plot.removeItem(page._second)
    pane.format_pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    assert pane.hub.current() == ElementRef("", "legend")
    page._plot.legend.removeItem("series")
    page._plot.removeItem(page._plot.listDataItems()[0])
    pane.format_pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    assert pane.hub.current() in (ElementRef("", "legend"), ElementRef("", "figure"))
    assert fp.list.ref_of(fp.list.currentItem()) == pane.hub.current()


def test_the_selection_clears_when_the_page_is_removed_and_nothing_is_left_on_screen(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    outline, bar = _outline(page), _bar(page)
    explorer.remove_page("iv")
    QtWidgets.QApplication.processEvents()
    assert explorer.inspector().selection_hub("iv") is None
    assert not outline.isVisible() and not bar.isVisible()


# -- no new routes of input ---------------------------------------------------------------------------

def _tree(name: str) -> ast.AST:
    path = pathlib.Path(__file__).resolve().parents[2] / "nanoscope" / "gui" / name
    return ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("module", ["format_inspector.py", "format_toolbar.py", "figure_text.py",
                                    "fit_label.py", "font_picker.py"])
def test_no_event_filter_no_key_override_and_no_escape_or_tab_shortcut(module):
    tree = _tree(module)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "installEventFilter", f"{module}:{node.lineno}"
            if module != "figure_text.py":            # the inline editor's own Esc is figure_text's
                assert node.attr not in ("Key_Escape", "Key_Tab"), f"{module}:{node.lineno}"
        if isinstance(node, ast.FunctionDef):
            assert node.name != "eventFilter", f"{module}:{node.lineno}"
            if module in ("format_inspector.py", "format_toolbar.py"):
                assert node.name != "keyPressEvent", f"{module}:{node.lineno}"
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert name not in ("QShortcut", "setShortcut"), f"{module}:{node.lineno}"


# -- several plots ---------------------------------------------------------------------------------------

def test_an_empty_click_on_another_plot_clears_the_selection(qtbot):
    window = F.bare(qtbot, views=("a", "b"))
    window.shown = list(window.controllers)
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    F.click_scene(window.plots[0], F.point_of(window.plots[0], "axis.left.label"))
    assert window.hub.current() == ElementRef("a", "axis.left.label")
    vb = window.plots[1].getViewBox().sceneBoundingRect()
    F.click_scene(window.plots[1], vb.center() + QtCore.QPointF(0, 30))
    assert window.hub.current() is None


def test_the_other_plots_of_one_scene_do_not_clear_what_one_click_just_claimed(qtbot):
    """Two controllers can share a scene: both see the one click, one claims it, the other
    says "nothing of mine" in the same turn of the loop. That must not undo the selection."""
    window = F.bare(qtbot, views=("a", "b"))
    window.shown = list(window.controllers)
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    first, second = window.controllers
    first.claimed.emit("title")
    second.cleared.emit()
    assert window.hub.current() == ElementRef("a", "title")
    QtWidgets.QApplication.processEvents()                   # the click is over
    second.cleared.emit()
    assert window.hub.current() is None


def test_a_controller_says_claimed_for_a_right_click_too_and_cleared_for_a_left_miss(qtbot):
    window = F.bare(qtbot)
    controller, plot = window.controllers[0], window.plots[0]
    seen = []
    controller.claimed.connect(lambda key: seen.append(("claimed", key)))
    controller.cleared.connect(lambda: seen.append(("cleared",)))
    window.widget.apply_appearance(replace(window.state, title="T"))
    F.click_scene(plot, F.point_of(plot, "title"), button=F.R)
    controller.last_menu.close()
    vb = plot.getViewBox().sceneBoundingRect()
    F.click_scene(plot, vb.center() + QtCore.QPointF(0, 30))
    assert seen == [("claimed", "title"), ("cleared",)]


def test_the_first_empty_click_after_dragging_an_annotation_clears_selection(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    F.drag_scene(page._plot, F.point_of(page._plot, "annotation.fit"), delta=(70, 40))
    assert pane.hub.current() == ElementRef("", "annotation.fit")
    vb = page._plot.getViewBox().sceneBoundingRect()
    F.click_scene(page._plot, vb.center() + QtCore.QPointF(0, 50))
    assert pane.hub.current() is None


def test_a_retired_pane_stops_listening_to_its_live_page(qtbot):
    page, explorer, pane, fp = _rig(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    fp.retire()
    fp.retire()                              # idempotent during teardown
    pane.hub.clear()
    page.controller.claimed.emit("axis.left.label")
    page._appearance_widget.apply_appearance(replace(page._appearance_widget.state(), grid=False))
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    assert pane.hub.current() is None
    assert page.controller._nanoscope_minibar is None
