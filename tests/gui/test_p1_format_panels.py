"""Phase 1, Format group: the element list, the panels, and where the appearance widget's rows went.

The page's ``PlotAppearanceWidget`` stays the state owner and the single history; its rows are
MOVED into the panels (same instances, labels with them), a control it had hidden stays hidden,
and a panel is retargeted, never rebuilt. Real keys and clicks; what is asserted is what is
stored and what is DRAWN. Synthetic data only; offscreen.
"""
from __future__ import annotations

import ast
import os
import pathlib
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import AxisPin, TextStyle
from nanoscope.gui.fit_label import FitLabelItem
from nanoscope.gui.format_inspector import (NO_CONTROLLERS, ElementList, ElementRef, FormatPane,
                                            SelectionHub)
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plot_workflow import FormatSpec

from . import p1_format_rig as F
from . import p1_hosting as H


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _panel(window, key):
    panel = window.pane.panel_for(key)
    window.pane.show_element(key)
    QtWidgets.QApplication.processEvents()
    return panel


# -- where the rows went -------------------------------------------------------------------------------

PLACES = {"figure": ("font_family", "background", "grid", "title_size", "axis_size", "tick_size",
                     "legend_size"),
          "legend": ("legend_visible",),
          "series.data": ("line_width", "marker_size", "data_color", "fit_color"),
          "title": ("title",), "axis.bottom.label": ("x_label",), "axis.left.label": ("y_label",)}


@pytest.mark.parametrize("key", list(PLACES))
def test_each_row_is_moved_into_its_panel_with_its_label(qtbot, key):
    window = F.bare(qtbot)
    panel = _panel(window, key)
    for name in PLACES[key]:
        control = getattr(window.widget, name)
        label = control._nanoscope_row_label
        assert panel.isAncestorOf(control) and panel.isAncestorOf(label), (key, name)
        assert control.isVisible() and label.isVisible(), (key, name)
    assert not window.widget.isVisible()                    # the widget itself is a hidden husk


def test_the_fit_label_group_is_moved_whole_and_its_duplicate_rows_are_not_shown(qtbot):
    window = F.bare(qtbot)
    window.widget.set_fit_label_available(True)
    window.controllers[0].host.register_fit_label(FitLabelItem(window.plots[0].getViewBox()))
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    panel = _panel(window, "annotation.fit")
    group = window.widget.fit_label_group
    assert panel.isAncestorOf(group) and group.isVisible()
    # Text, Size and Colour are the style rows above the group: the same stores
    for name in ("fit_label_template", "fit_label_size", "fit_label_color"):
        assert not getattr(window.widget, name).isVisibleTo(panel), name
    for name in ("fit_label_show", "fit_label_position", "fit_label_in_legend"):
        assert getattr(window.widget, name).isVisibleTo(panel), name


def test_a_control_the_widget_hid_stays_hidden_in_its_panel(qtbot):
    widget = PlotAppearanceWidget(PlotAppearance(), include_colors=False, include_markers=False,
                                  include_legend=False, include_labels=False)
    window = F.bare(qtbot, appearance=widget)
    series = _panel(window, "series.data")
    for name in ("data_color", "fit_color", "marker_size"):
        assert not getattr(widget, name).isVisibleTo(series), name
    assert widget.line_width.isVisibleTo(series)
    legend = _panel(window, "legend")
    assert not widget.legend_visible.isVisibleTo(legend)
    assert not widget.legend_size.isVisible()
    window.pane.show_element("title")
    text = window.pane.panel_for("title")
    assert not widget.title.isVisibleTo(text)               # the page showed no label boxes
    assert text.words.isVisibleTo(text)                     # its words are the generic box


def test_a_moved_control_still_does_its_job_one_step_of_the_widgets_own_history(qtbot):
    window = F.bare(qtbot)
    figure = _panel(window, "figure")
    before = window.widget.history_mark()
    combo = window.widget.background
    combo.setCurrentIndex(combo.findData("light"))
    assert window.state.background == "light" and window.widget.history_mark() == before + 1
    assert figure.isAncestorOf(combo)
    assert window.widget.undo() and window.state.background is None


def test_typing_in_the_titles_moved_box_is_the_figures_title(qtbot):
    window = F.bare(qtbot)
    panel = _panel(window, "title")
    box = window.widget.title
    assert panel.isAncestorOf(box)
    box.setFocus()
    QtTest.QTest.keyClicks(box, "Hello")
    QtWidgets.QApplication.processEvents()
    assert window.state.title == "Hello"
    assert "Hello" in window.plots[0].titleLabel.text
    assert not window.widget.x_label.isVisibleTo(panel)     # one words row at a time


@pytest.mark.parametrize("annotation", ["annotation.fit", F.CAPTION])
@pytest.mark.parametrize("annotation_first", [False, True])
def test_annotation_panels_never_take_the_shared_title_and_axis_wording_rows(
        qtbot, annotation, annotation_first):
    page, explorer = F.hosted_rich(qtbot)
    explorer.inspector().set_tab(1)
    pane = explorer.inspector().pane("iv").format_pane
    widget = page._appearance_widget
    rows = [(key, getattr(widget, attr), getattr(widget, attr)._nanoscope_row_label)
            for key, attr in (("title", "title"), ("axis.bottom.label", "x_label"),
                              ("axis.left.label", "y_label"))]
    words = [control.text() for _key, control, _label in rows]
    if annotation_first:
        pane.show_element(annotation)
    pane.show_element("title")
    text_panel = pane.current_panel()
    parents = [(control.parentWidget(), label.parentWidget()) for _key, control, label in rows]

    pane.show_element(annotation)
    annotation_panel = pane.current_panel()
    assert annotation_panel is not text_panel and annotation_panel.words.isVisibleTo(annotation_panel)
    for (key, control, label), (control_parent, label_parent), word in zip(rows, parents, words):
        pane.show_element(key)
        assert pane.current_panel() is text_panel
        assert control.parentWidget() is control_parent and label.parentWidget() is label_parent
        assert text_panel.isAncestorOf(control) and text_panel.isAncestorOf(label)
        assert control.isVisibleTo(text_panel) and label.isVisibleTo(text_panel)
        assert control.text() == word

    pane.show_element("title")
    widget.title.setFocus()
    widget.title.setCursorPosition(len(widget.title.text()))
    QtTest.QTest.keyClicks(widget.title, " amended")
    QtWidgets.QApplication.processEvents()
    assert widget.state().title == words[0] + " amended"


def test_the_other_texts_have_one_words_box_that_renames_them(qtbot):
    window = F.bare(qtbot)
    panel = _panel(window, "legend.entry.s")
    assert panel.words.isVisibleTo(panel) and not window.widget.title.isVisibleTo(panel)
    assert panel.words.placeholderText() == "Automatic: s"
    before = window.widget.history_mark()
    panel.words.setFocus()
    QtTest.QTest.keyClicks(panel.words, "Renamed")
    QtWidgets.QApplication.processEvents()
    assert window.state.text_style("legend.entry.s").text == "Renamed"
    assert window.plots[0].legend.items[0][1].text == "Renamed"
    assert window.widget.history_mark() == before + 1       # a run of keystrokes is one step


# -- the list ----------------------------------------------------------------------------------------

def test_the_element_list_is_a_plain_tree_that_uses_current_item_changed_only(qtbot):
    window = F.bare(qtbot)
    listing = window.pane.list
    assert isinstance(listing, ElementList) and listing.isHeaderHidden()
    assert listing.accessibleName() == "Figure elements"
    assert listing.editTriggers() == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    assert listing.maximumHeight() <= 160                    # about six rows
    tree = ast.parse((pathlib.Path(__file__).resolve().parents[2] / "nanoscope" / "gui"
                      / "format_inspector.py").read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "itemChanged" not in names and "currentItemChanged" in names


def test_the_rows_are_the_elements_on_screen_and_hold_keys_not_items(qtbot):
    window = F.bare(qtbot)
    keys = [ref.key for ref in window.pane.elements()]
    assert keys == ["figure", "title", "axis.bottom", "axis.bottom.label", "axis.bottom.ticks",
                    "axis.left", "axis.left.label", "axis.left.ticks", "legend",
                    "legend.entry.s", "series.data"]
    assert window.hub.refs() == list(window.pane.elements())
    item = window.pane.list.find(ElementRef("", "axis.left.label"))
    assert item.text(0) == "Label" and item.parent().text(0) == "Y axis"


def test_a_hidden_legend_still_has_rows_and_says_so(qtbot):
    window = F.bare(qtbot)
    window.plots[0].legend.hide()
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    item = window.pane.list.find(ElementRef("", "legend"))
    assert item.text(0) == "Legend (hidden)" and item.childCount() == 1


def test_only_the_view_on_screen_is_listed_and_a_gone_view_falls_back_to_the_figure(qtbot):
    window = F.bare(qtbot, views=("a", "b"))
    window.shown = [window.controllers[0]]
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    viewed = lambda: {ref.plot for ref in window.pane.elements()           # noqa: E731
                      if ref.key not in ("figure", "series.data")}
    assert viewed() == {"a"}
    window.pane.show_element(ElementRef("a", "title"))
    assert window.hub.current() == ElementRef("a", "title")
    window.shown = [window.controllers[1]]                   # the other view comes on screen
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    assert viewed() == {"b"}
    assert window.hub.current() == ElementRef("", "figure")


def test_a_click_on_a_view_selects_that_views_element(qtbot):
    window = F.bare(qtbot, views=("a", "b"))
    window.shown = list(window.controllers)
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    F.click_scene(window.plots[1], F.point_of(window.plots[1], "axis.left.label"))
    assert window.hub.current() == ElementRef("b", "axis.left.label")
    F.click_scene(window.plots[0], F.point_of(window.plots[0], "axis.bottom.ticks"))
    assert window.hub.current() == ElementRef("a", "axis.bottom.ticks")


def test_a_figure_nothing_of_which_can_be_clicked_says_so(qtbot):
    window = F.bare(qtbot, views=())
    assert [ref.key for ref in window.pane.elements()] == [
        "figure", "title", "axis.bottom.label", "axis.left.label", "legend", "series.data"]
    assert window.pane.note.text() == NO_CONTROLLERS and not window.pane.note.isHidden()
    # The lack of click-to-select does not hide the page's ordinary wording controls.
    for key, field in (("title", "title"), ("axis.bottom.label", "x_label"),
                       ("axis.left.label", "y_label")):
        window.pane.show_element(key)
        control = getattr(window.widget, field)
        panel = window.pane.current_panel()
        assert control.isEnabled() and control.isVisibleTo(panel)
        assert panel.fields.isHidden()       # do not promise per-element matplotlib styles


# -- the axis panel ----------------------------------------------------------------------------------

def _axis(qtbot, **options):
    window = F.bare(qtbot, **options)
    panel = _panel(window, "axis.bottom")
    return window, panel


def test_a_range_typed_in_the_axis_panel_is_a_pin_one_step_and_drawn(qtbot):
    window, panel = _axis(qtbot)
    fields = panel.fields
    assert fields.auto.isChecked() and not fields.lo.isEnabled()
    before = window.widget.history_mark()
    H.click(fields.auto)                                      # untick Auto: the shown range is pinned
    assert not fields.auto.isChecked() and fields.lo.isEnabled()
    for box, text in ((fields.lo, "1"), (fields.hi, "5")):
        box.setFocus()
        box.selectAll()
        QtTest.QTest.keyClicks(box, text)
        H.key(box, QtCore.Qt.Key.Key_Return)                  # editing finished: one write
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
    pin = window.state.axis_pin("bottom")
    assert (pin.min, pin.max) == (1.0, 5.0)
    assert window.plots[0].viewRange()[0] == pytest.approx([1.0, 5.0])
    assert panel.range_dot.is_set() and panel.range_dot.text() == "● set here"
    assert window.widget.history_mark() == before + 3        # Auto off, then each edit finished


def test_a_bad_range_says_why_and_changes_nothing(qtbot):
    window, panel = _axis(qtbot)
    fields = panel.fields
    H.click(fields.auto)
    before, state = window.widget.history_mark(), window.state
    fields.lo.setText("9")
    fields.hi.setText("2")
    fields.hi.setFocus()
    H.key(fields.hi, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert fields.error.isVisibleTo(panel) and "minimum is below its maximum" in fields.error.text()
    assert window.state == state and window.widget.history_mark() == before


def test_the_dot_beside_the_range_puts_it_back_to_auto(qtbot):
    window, panel = _axis(qtbot)
    window.widget.apply_appearance(window.state.with_axis_pin("bottom", AxisPin(min=0.5, max=1.5)))
    QtWidgets.QApplication.processEvents()
    assert panel.range_dot.is_set()
    before = window.widget.history_mark()
    H.click(panel.range_dot)
    assert window.state.axis_pin("bottom") == AxisPin()
    assert window.widget.history_mark() == before + 1
    assert panel.fields.auto.isChecked() and not panel.range_dot.is_set()


def test_the_major_tick_spacing_is_the_same_store_as_the_popover(qtbot):
    window, panel = _axis(qtbot)
    panel.fields.major.setValue(0.5)
    QtWidgets.QApplication.processEvents()
    assert window.state.axis_pin("bottom").tick_major == 0.5 and panel.major_dot.is_set()


def test_log_is_a_read_only_mirror_of_the_pages_own_box(qtbot):
    window, panel = _axis(qtbot)
    assert not panel.fields.log.isEnabled()
    assert "no log scale" in panel.fields.log.toolTip()
    assert not panel.log_link.isVisibleTo(panel)             # this page has no log control


def test_the_axis_panel_serves_both_axes_and_each_views_pins_are_its_own(qtbot):
    window = F.bare(qtbot, views=("a", "b"))
    window.shown = [window.controllers[0]]
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    window.pane.show_element(ElementRef("a", "axis.left"))
    panel = window.pane.panel_for("axis.left")
    panel.fields.auto.setChecked(False)
    panel.fields.lo.setText("0")
    panel.fields.hi.setText("2")
    panel._edited()
    QtWidgets.QApplication.processEvents()
    assert dict(window.state.axes) == {"a.left": AxisPin(min=0.0, max=2.0)}
    assert window.plots[0].viewRange()[1] == pytest.approx([0.0, 2.0])
    assert window.plots[1].viewRange()[1] != pytest.approx([0.0, 2.0])     # another view is not framed


def test_a_shared_label_says_why_it_is_not_typed_on_the_figure(qtbot):
    window = F.bare(qtbot, shared_labels=figure_text.SHARED_LABELS_REASON)
    panel = _panel(window, "axis.bottom.label")
    assert panel._locked.isVisibleTo(panel) and "share one X and one Y label" in panel._locked.text()
    window.pane.show_element("title")
    assert not window.pane.panel_for("title")._locked.isVisibleTo(window.pane.panel_for("title"))


# -- a page the Foundation still hosts by its fallback ----------------------------------------------------

def test_a_page_whose_format_widgets_are_the_whole_appearance_section_keeps_the_foundations_column(qtbot):
    widget = PlotAppearanceWidget(PlotAppearance())
    section = QtWidgets.QScrollArea()
    section.setWidget(widget)
    pane = FormatPane(FormatSpec(widget, lambda: (), extras={"figure": [section]}), SelectionHub())
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    assert pane.is_legacy() and pane.elements() == ()
    assert pane.panel_for("figure") is pane._body and pane._body.isAncestorOf(widget)
    pane.show_element("title")
    pane.refresh()                                           # nothing to do, nothing raised


# -- the delegation of the popovers ---------------------------------------------------------------------

def test_the_popovers_keep_their_attribute_names_by_delegation(qtbot):
    popover = figure_text.TextStylePopover("title", TextStyle(text="T", bold=True), "auto",
                                           align=True)
    qtbot.addWidget(popover)
    for name in ("family", "size", "bold", "italic", "color", "use_family", "align"):
        assert getattr(popover, name) is getattr(popover.fields, name), name
    assert popover.text is not None and popover.objectName() == "plotTextStylePopover"
    assert popover.value() == TextStyle(text="T", bold=True)
    axis = figure_text.AxisPropertiesPopover("bottom", figure_text.AxisState(shown_range=(0, 2)))
    qtbot.addWidget(axis)
    for name in ("label", "units", "lo", "hi", "auto", "log", "tick_size", "major", "error"):
        assert getattr(axis, name) is getattr(axis.fields, name), name
    assert axis.value().range is None


def test_a_refresh_never_overwrites_a_number_being_typed(qtbot):
    window, panel = _axis(qtbot)
    H.click(panel.fields.auto)
    window.activateWindow()
    panel.fields.lo.setFocus()
    QtWidgets.QApplication.processEvents()
    panel.fields.lo.setText("0.")                             # half a number, not yet committed
    window.pane.refresh()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    assert panel.fields.lo.hasFocus() and panel.fields.lo.text() == "0."


def test_the_hub_says_format_changed_once_per_turn_and_elements_changed_only_when_they_did(qtbot):
    hub = SelectionHub()
    formats, elements = [], []
    hub.formatChanged.connect(lambda: formats.append(1))
    hub.elementsChanged.connect(lambda: elements.append(1))
    for _ in range(5):
        hub.request_format_changed()
    assert formats == []                                      # not before the loop turns
    QtWidgets.QApplication.processEvents()
    assert formats == [1]
    hub.request_format_changed()
    QtWidgets.QApplication.processEvents()
    assert formats == [1, 1]
    refs = [ElementRef("", "figure"), ElementRef("", "title")]
    hub.set_refs(refs)
    hub.set_refs(list(refs))
    assert elements == [1] and hub.refs() == refs


def test_the_tick_numbers_are_the_axis_less_its_label(qtbot):
    window = F.bare(qtbot)
    controller = window.controllers[0]
    for side in ("left", "bottom"):
        ticks = controller.anchor_rect(f"axis.{side}.ticks")
        label = controller.anchor_rect(f"axis.{side}.label")
        whole = controller.anchor_rect(f"axis.{side}")
        assert not ticks.intersects(label) and whole.contains(ticks), (side, ticks, label, whole)


def test_format_control_transfers_invalidate_old_items_and_keep_controls_alive(qtbot):
    import shiboken6

    window = F.Bare()
    qtbot.addWidget(window)
    widget = window.widget
    widget.set_fit_label_available(True)
    controls = [widget.font_family, widget.font_family._nanoscope_row_label,
                widget.title, widget.title._nanoscope_row_label, widget.fit_label_group]
    old_items = []
    for control in controls:
        layout = control.parentWidget().layout()
        old_items.append(layout.itemAt(layout.indexOf(control)))
    for key in ("figure", "title", "annotation.fit"):
        window.pane.panel_for(key)
    assert all(not shiboken6.isValid(item) for item in old_items)
    assert all(shiboken6.isValid(control) for control in controls)
    widget.font_family.setCurrentIndex(widget.font_family.findData("Times New Roman"))
    assert widget.state().font_family == "Times New Roman"


def test_colour_map_element_follows_page_visibility_and_retains_its_controls(qtbot):
    extra = QtWidgets.QComboBox()
    extra.addItems(["A", "B"])
    active = [True]
    extra._nanoscope_format_visible = lambda: active[0]
    window = F.bare(qtbot, extras={"colour_map": [extra]})
    ref = ElementRef("", "colour_map")
    window.pane.show_element(ref)
    panel = window.pane.current_panel()
    assert ref in window.pane.elements() and panel.isAncestorOf(extra)
    extra.setCurrentIndex(1)
    active[0] = False
    window.pane.refresh()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    assert ref not in window.pane.elements()
    assert window.hub.current() == ElementRef("", "figure")
    active[0] = True
    window.pane.refresh()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    window.pane.show_element(ref)
    assert window.pane.current_panel() is panel and extra.currentIndex() == 1


def test_a_page_without_canvas_controllers_keeps_its_existing_wording_controls(qtbot):
    window = F.bare(qtbot)
    window.shown = []                         # matplotlib pages have no canvas controller
    window.pane.refresh()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    for key, name in (("title", "title"), ("axis.bottom.label", "x_label"),
                      ("axis.left.label", "y_label")):
        ref = ElementRef("", key)
        assert ref in window.pane.elements()
        window.pane.show_element(ref)
        panel = window.pane.current_panel()
        control = getattr(window.widget, name)
        assert panel.isAncestorOf(control) and control.isVisibleTo(panel)
        assert not panel.fields.isVisibleTo(panel)    # no promise of editable matplotlib yet
    assert ElementRef("", "title") in window.pane.elements()  # hiding other rows removes none


def test_rehosting_comparison_keeps_all_wording_rows_after_a_title_selection(qtbot):
    key = "comparison_figures"
    page = H.page_builders()[key]()
    explorer = H.hosted(qtbot, page, key)
    explorer.inspector().set_tab(1)
    original = explorer.inspector().pane(key).format_pane
    original.show_element("title")
    widget = page._appearance_widget
    assert widget.x_label.isHidden() and widget.y_label.isHidden()
    assert explorer.remove_page(key) is page
    explorer.show_page(key, page)
    explorer.inspector().set_tab(1)
    pane = explorer.inspector().pane(key).format_pane
    assert pane is not original

    for element, name in (("title", "title"), ("axis.bottom.label", "x_label"),
                          ("axis.left.label", "y_label")):
        ref = ElementRef("", element)
        assert ref in pane.elements()
        pane.show_element(ref)
        panel = pane.current_panel()
        control = getattr(widget, name)
        label = control._nanoscope_row_label
        assert panel.isAncestorOf(control) and panel.isAncestorOf(label)
        assert control.isVisibleTo(panel) and label.isVisibleTo(panel)
        assert control.isEnabled() and not control.isReadOnly()
        control.setFocus()
        control.selectAll()
        QtTest.QTest.keyClicks(control, f"Rehosted {name}")
        QtWidgets.QApplication.processEvents()
        assert getattr(widget.state(), name) == f"Rehosted {name}"
    assert all(ElementRef("", element) in pane.elements()
               for element in ("title", "axis.bottom.label", "axis.left.label"))


def test_rehosting_does_not_enable_wording_or_legend_controls_the_page_omitted(qtbot):
    widget = PlotAppearanceWidget(PlotAppearance(), include_labels=False, include_legend=False)
    first = FormatPane(FormatSpec(widget, lambda: ()), SelectionHub())
    qtbot.addWidget(first)
    first.panel_for("title")
    first.retire()
    second = FormatPane(FormatSpec(widget, lambda: ()), SelectionHub())
    qtbot.addWidget(second)
    panel = second.panel_for("title")
    second.show_element("title")
    for element in ("title", "axis.bottom.label", "axis.left.label", "legend"):
        assert ElementRef("", element) not in second.elements()
    assert panel._moved == {}
    for control in (widget.title, widget.x_label, widget.y_label, widget.legend_visible):
        assert control.isHidden()
