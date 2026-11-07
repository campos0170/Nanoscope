"""Phase 1, Format group: the mini toolbar over a selected text, and the construction budget.

The toolbar is a child of the view's VIEWPORT (never a scene item), built once per controller at
the first selection of a text, and every control writes through the controller: one gesture is
one Undo step, a run of size clicks is one step. Real clicks through the QWINDOW; what is
asserted is what is stored and DRAWN. Synthetic data only; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import TextStyle
from nanoscope.gui.format_inspector import ElementRef
from nanoscope.gui.format_toolbar import MiniTextToolbar, SelectionOutline

from . import p1_format_rig as F
from . import p1_hosting as H
from .test_p0_reset_on_quantity_change import noise, events, _type_event_words
from .test_plot_window_routes import _edit_double_click
from .test_p0_pins_clear_with_the_words import _pin_inside


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _rig(qtbot, key="title"):
    page, explorer = F.hosted_rich(qtbot)
    F.click_scene(page._plot, F.point_of(page._plot, key))
    bar = page.controller._nanoscope_minibar
    assert bar.isVisible() and bar.key() == key
    return page, explorer, bar


def _up_arrow(spin):
    opt = QtWidgets.QStyleOptionSpinBox()
    opt.initFrom(spin)
    opt.subControls = QtWidgets.QStyle.SubControl.SC_All
    rect = spin.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_SpinBox, opt,
                                       QtWidgets.QStyle.SubControl.SC_SpinBoxUp, spin)
    return rect.center()


def _style(page, key="title") -> TextStyle:
    return page._appearance_widget.state().text_style(key)


def _marks(page) -> int:
    return page._appearance_widget.history_mark()


# -- what it is --------------------------------------------------------------------------------------

def test_it_is_a_child_of_the_viewport_built_once_and_never_a_scene_item(qtbot):
    page, explorer, bar = _rig(qtbot)
    viewport = F.view_of(page._plot).viewport()
    assert isinstance(bar, MiniTextToolbar) and bar.parentWidget() is viewport
    assert isinstance(page.controller._nanoscope_outline, SelectionOutline)
    assert page.controller._nanoscope_outline.parentWidget() is viewport
    for key in ("axis.left.label", "legend.entry.series", F.CAPTION, "title"):
        F.click_scene(page._plot, F.point_of(page._plot, key))
    assert page.controller._nanoscope_minibar is bar
    assert len(viewport.findChildren(MiniTextToolbar)) == 1          # one per controller
    assert not bar.testAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)


def test_it_is_not_built_until_a_text_is_selected(qtbot):
    page, explorer = F.hosted_rich(qtbot)
    explorer.inspector().set_tab(1)
    QtWidgets.QApplication.processEvents()
    assert getattr(page.controller, "_nanoscope_minibar", None) is None
    explorer.inspector().pane("iv").format_pane.show_element("legend")     # not a text
    assert getattr(page.controller, "_nanoscope_minibar", None) is None


# -- placement ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["title", "axis.bottom.label", "axis.left.label",
                                 "axis.left.ticks", "legend.entry.second", "annotation.fit",
                                 F.CAPTION, F.PANEL])
def test_it_floats_beside_the_element_inside_the_view_and_never_over_it(qtbot, key):
    page, explorer, bar = _rig(qtbot, key)
    room = F.view_of(page._plot).viewport().rect()
    anchor = page.controller.anchor_rect(key)
    assert room.contains(bar.geometry()), (bar.geometry(), room)
    assert not bar.geometry().intersects(anchor), (bar.geometry(), anchor)


def test_toolbar_wraps_existing_controls_inside_a_narrow_canvas(qtbot):
    page, explorer, bar = _rig(qtbot)
    controls = tuple(bar.fields._compact_controls)
    before = H.count_widgets(explorer)
    explorer.resize(840, 560)
    QtWidgets.QApplication.processEvents()
    bar.reposition()
    room = F.view_of(page._plot).viewport().rect()
    assert room.contains(bar.geometry()), (bar.geometry(), room)
    assert not bar.geometry().intersects(page.controller.anchor_rect("title"))
    assert tuple(bar.fields._compact_controls) == controls
    assert H.count_widgets(explorer) == before
    explorer.resize(1280, 820)
    QtWidgets.QApplication.processEvents()
    bar.reposition()
    assert tuple(bar.fields._compact_controls) == controls
    assert all(control.isVisibleTo(bar) for control in controls)


def test_title_toolbar_keeps_the_nearby_noise_fit_label_reachable(noise):
    page, _explorer = noise
    controller = figure_text.figure_text_for(page._plot)
    _edit_double_click(page._plot, page._plot.titleLabel, "PSD title")
    bar = controller._nanoscope_minibar
    fit_rect = controller.anchor_rect("annotation.fit")
    assert bar.isVisible() and fit_rect is not None and not fit_rect.isEmpty()
    assert not bar.geometry().intersects(fit_rect)
    _edit_double_click(page._plot, page._fit_label, "Flicker: {equation}")
    assert page.fit_label_text().startswith("Flicker:")


def test_title_toolbar_keeps_the_ensemble_axis_menu_reachable(events):
    page, _explorer = events
    tabs = page._tabs.tabBar()
    H.click(tabs, pos=tabs.tabRect(1).center())
    plot = page._ensemble_plot
    _type_event_words(page, plot=plot)
    controller = figure_text.figure_text_for(plot)
    bar = controller._nanoscope_minibar
    assert bar.isVisible()
    assert not bar.geometry().intersects(controller.anchor_rect("axis.bottom.label"))
    _pin_inside(plot, "bottom")                 # actual right-click, menu, and axis dialog


# -- what each control writes ------------------------------------------------------------------------

def test_the_font_is_picked_live_one_step_and_drawn(qtbot):
    page, explorer, bar = _rig(qtbot)
    before = _marks(page)
    drawn = dict(page._plot.titleLabel.opts)
    picker = bar.fields.family
    picker.setCurrentIndex(picker.findData("Times New Roman"))
    QtWidgets.QApplication.processEvents()
    assert _style(page).family == "Times New Roman" and _style(page).text == "A typed title"
    assert _marks(page) == before + 1
    assert page._plot.titleLabel.opts.get("family") != drawn.get("family")
    picker.setCurrentIndex(0)                                        # "Figure font": follow the figure
    QtWidgets.QApplication.processEvents()
    assert _style(page).family is None


def test_a_run_of_size_clicks_is_one_undo_step_and_one_undo_takes_it_back(qtbot):
    page, explorer, bar = _rig(qtbot)
    before = _marks(page)
    for _ in range(5):                                               # five real clicks on the arrow
        H.click(bar.fields.size, pos=_up_arrow(bar.fields.size))
    assert _style(page).size_pt == 8.0                               # 4 (the floor), then 5, 6, 7, 8
    assert page._plot.titleLabel.opts["size"] == "8pt"
    assert _marks(page) == before + 1                                # ONE step for the run
    assert page._appearance_widget.undo()
    assert _style(page).size_pt is None and _marks(page) == before
    assert page._plot.titleLabel.opts.get("size") != "8pt"


def test_two_runs_apart_are_two_steps(qtbot):
    page, explorer, bar = _rig(qtbot)
    before = _marks(page)
    H.click(bar.fields.size, pos=_up_arrow(bar.fields.size))
    page._appearance_widget._close_burst()                           # the 700 ms pause
    H.click(bar.fields.size, pos=_up_arrow(bar.fields.size))
    assert _marks(page) == before + 2


def test_bold_italic_colour_and_alignment_are_one_step_each(qtbot):
    page, explorer, bar = _rig(qtbot)
    before = _marks(page)
    H.click(bar.fields.bold)
    assert _style(page).bold is True and _marks(page) == before + 1
    page._appearance_widget._close_burst()
    H.click(bar.fields.italic)
    assert _style(page).italic is True and _marks(page) == before + 2
    page._appearance_widget._close_burst()
    button = bar.fields.color
    button.showMenu()
    QtWidgets.QApplication.processEvents()
    menu = button.menu()
    action = next(a for a in menu.actions() if a.text() == "#d55e00")
    QtTest.QTest.mouseClick(menu, H.L, H.NOMOD, menu.actionGeometry(action).center())
    QtWidgets.QApplication.processEvents()
    assert _style(page).color == "#d55e00" and _marks(page) == before + 3
    assert page._plot.titleLabel.opts["color"] == "#d55e00"
    assert bar.fields.align.isVisibleTo(bar)                         # the title can align
    bar.fields.align.setCurrentIndex(bar.fields.align.findData("left"))
    QtWidgets.QApplication.processEvents()
    assert _style(page).align == "left" and _marks(page) == before + 4
    # nothing the toolbar wrote touched the words
    assert _style(page).text == "A typed title"


def test_alignment_is_offered_for_the_title_only(qtbot):
    page, explorer, bar = _rig(qtbot, "axis.left.label")
    assert not bar.fields.align.isVisibleTo(bar)


def test_a_text_that_cannot_be_edited_offers_no_edit_button(qtbot):
    page, explorer, bar = _rig(qtbot, "axis.left.ticks")
    assert not bar.edit_button.isVisibleTo(bar) and bar.fields.isVisibleTo(bar)
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    assert bar.edit_button.isVisibleTo(bar)


def test_it_writes_the_fit_labels_size_without_losing_its_wording(qtbot):
    """The text trap: a bare TextStyle handed to the fit label cleared its template."""
    page, explorer, bar = _rig(qtbot, "annotation.fit")
    widget = page._appearance_widget
    page.controller.set_style_property("annotation.fit", "text", "{equation} fitted")
    bar.fields.size.setValue(12.0)                                   # a typed value, one write
    QtWidgets.QApplication.processEvents()
    state = widget.state()
    assert state.fit_label_state().template == "{equation} fitted"
    assert state.fit_label_state().size_pt == 12.0


def test_the_edit_button_opens_the_inline_editor_and_the_toolbar_steps_aside(qtbot):
    page, explorer, bar = _rig(qtbot)
    opened = []
    page.controller.opened.connect(lambda kind, key: opened.append((kind, key)))
    H.click(bar.edit_button)
    assert opened == [("inline", "title")]
    editor = page.controller.editor
    assert figure_text._alive(editor) and editor.isVisible()
    assert not bar.isVisible()                                       # not over its own editor
    H.key(editor, QtCore.Qt.Key.Key_Escape)                          # the editor's own Esc
    QtWidgets.QApplication.processEvents()
    assert bar.isVisible() and page.controller.editor is None or not figure_text._alive(
        page.controller.editor)


def test_committing_the_editor_brings_the_toolbar_back(qtbot):
    page, explorer, bar = _rig(qtbot)
    H.click(bar.edit_button)
    editor = page.controller.editor
    editor.setText("Retyped")
    H.key(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert _style(page).text == "Retyped" and bar.isVisible()


def test_it_goes_when_the_selection_clears_and_is_not_rebuilt_for_the_next_one(qtbot):
    page, explorer, bar = _rig(qtbot)
    vb = page._plot.getViewBox().sceneBoundingRect()
    F.click_scene(page._plot, vb.center() + QtCore.QPointF(0, 40))
    assert not bar.isVisible()
    figure_text.GATE.reset()          # two quick clicks on one title are a double click, as ever
    F.click_scene(page._plot, F.point_of(page._plot, "title"))
    assert bar.isVisible() and page.controller._nanoscope_minibar is bar


# -- construction: a selection builds nothing ----------------------------------------------------------

def test_no_widget_is_constructed_by_twenty_selections(qtbot):
    page, explorer = F.hosted_rich(qtbot)
    pane = explorer.inspector().pane("iv")
    fp = pane.format_pane
    keys = ["title", "axis.bottom.label", "axis.left.ticks", "legend", "legend.entry.second",
            "annotation.fit", F.CAPTION, F.PANEL, "axis.left", "series.data", "figure"]
    for key in keys:                                                 # builds every panel once
        fp.show_element(key)
    for key in ("title", "axis.left.label", "legend.entry.series", "annotation.fit"):
        F.click_scene(page._plot, F.point_of(page._plot, key))      # and the overlays
    QtWidgets.QApplication.processEvents()
    mark = H.count_widgets(explorer)
    for turn in range(20):
        key = keys[turn % len(keys)]
        if turn % 2:
            fp.show_element(key)                                     # the list's route
        elif key not in ("legend", "axis.left", "series.data", "figure"):
            F.click_scene(page._plot, F.point_of(page._plot, key))   # the canvas's route
        QtWidgets.QApplication.processEvents()
    assert H.count_widgets(explorer) == mark
    page._appearance_widget.apply_appearance(
        page._appearance_widget.state().__class__.from_mapping({"tick_size": 11}))
    QtWidgets.QApplication.processEvents()
    assert H.count_widgets(explorer) == mark                         # nor does a redraw


def test_the_first_selection_of_a_kind_builds_its_panel_once(qtbot):
    page, explorer = F.hosted_rich(qtbot)
    fp = explorer.inspector().pane("iv").format_pane
    fp.show_element("title")
    built = dict(fp._panels)
    fp.show_element("axis.bottom.label")
    fp.show_element("title")
    assert fp._panels["text"] is built["text"]                       # retargeted, not rebuilt
    assert "axis" not in fp._panels and "fit_label" not in fp._panels   # built on first use only
