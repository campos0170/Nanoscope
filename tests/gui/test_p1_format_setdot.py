"""Phase 1, Format group: the "● set here" marker and where a value comes from.

``resolve(state, defaults, ref, prop)`` says whether a property is the element's own (``element``,
marked "● set here" and clickable), the figure's (``figure``, "from figure (Ticks 10 pt)") or the
page's default (``default``). A click on the marker clears that one property, as one Undo step.
Distributions reports seven concrete values at their defaults, so the page's own ``defaults()``
decides what "set" means. Real clicks; what is asserted is what is stored and shown. Synthetic
data only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import TextStyle, TextStyleFields
from nanoscope.gui.fit_label import FitLabelItem, FitLabelState
from nanoscope.gui.format_inspector import ElementRef, resolve
from nanoscope.gui.plot_appearance import PlotAppearance

from . import p1_format_rig as F
from . import p1_hosting as H


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _sources(panel) -> dict[str, str]:
    return {prop: dot.source() for prop, dot in panel.fields.dots.items()}


def _settle():
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()


# -- resolve -----------------------------------------------------------------------------------------

def test_resolve_names_the_three_sources_for_a_text():
    title, ticks = ElementRef("", "title"), ElementRef("", "axis.left.ticks")
    state = PlotAppearance(tick_size=10.0, font_family="Times New Roman").with_text_style(
        "title", TextStyle(bold=True, size_pt=18.0))
    assert resolve(state, None, title, "bold") == (True, "element")
    assert resolve(state, None, title, "size_pt") == (18.0, "element")
    assert resolve(state, None, title, "italic") == (None, "default")
    assert resolve(state, None, ticks, "size_pt") == (10.0, "figure")        # the role's field
    assert resolve(state, None, title, "size_pt") == (18.0, "element")       # the element wins
    assert resolve(state, None, title, "family") == ("Times New Roman", "figure")
    assert resolve(state, PlotAppearance(font_family="Times New Roman"), title,
                   "family") == ("Times New Roman", "default")               # the page's own default


def test_resolve_follows_the_role_of_each_element():
    state = PlotAppearance(title_size=20.0, axis_size=12.0, tick_size=9.0, legend_size=8.0)
    for key, size in (("title", 20.0), ("axis.bottom.label", 12.0), ("axis.left.ticks", 9.0),
                      ("legend.entry.x", 8.0)):
        assert resolve(state, None, ElementRef("", key), "size_pt") == (size, "figure"), key
    assert resolve(state, None, ElementRef("", "annotation.fit"), "size_pt") == (None, "default")


def test_resolve_reads_the_fit_labels_own_size_and_colour():
    state = PlotAppearance(fit_label=FitLabelState(size_pt=11.0, color="#d55e00"))
    ref = ElementRef("", "annotation.fit")
    assert resolve(state, None, ref, "size_pt") == (11.0, "element")
    assert resolve(state, None, ref, "color") == ("#d55e00", "element")


def test_resolve_for_the_figure_compares_with_the_pages_defaults():
    defaults = PlotAppearance(grid=True, background="dark", line_width=1.0)
    figure = ElementRef("", "figure")
    state = PlotAppearance(grid=True, background="dark", line_width=2.0)
    assert resolve(state, defaults, figure, "grid") == (True, "default")      # a default value
    assert resolve(state, defaults, figure, "background") == ("dark", "default")
    assert resolve(state, defaults, figure, "line_width") == (2.0, "element")
    assert resolve(state, None, figure, "grid") == (True, "element")          # "not None" alone would mark it
    assert resolve(PlotAppearance(), defaults, figure, "marker_size") == (None, "default")


# -- the text panel --------------------------------------------------------------------------------------

def test_a_fresh_element_shows_default_everywhere_and_a_set_property_shows_set_here(qtbot):
    window = F.bare(qtbot)
    panel = window.pane.panel_for("title")
    window.pane.show_element("title")
    assert set(_sources(panel).values()) == {"default"}
    assert all(not dot.isEnabled() for dot in panel.fields.dots.values())
    H.click(panel.fields.bold)
    _settle()
    sources = _sources(panel)
    assert sources["bold"] == "element" and panel.fields.dots["bold"].text() == "● set here"
    assert {v for k, v in sources.items() if k != "bold"} == {"default"}      # exactly the overridden one
    assert panel.fields.dots["bold"].isEnabled()


def test_an_inherited_size_shows_default_with_the_figures_value_in_grey_words(qtbot):
    window = F.bare(qtbot)
    window.widget.apply_appearance(replace(window.state, tick_size=10.0, title_size=18.0))
    panel = window.pane.panel_for("axis.bottom.ticks")
    window.pane.show_element("axis.bottom.ticks")
    _settle()
    assert panel.fields.size.value() == 0 and panel.fields.size.text().startswith("Default")
    dot = panel.fields.dots["size_pt"]
    assert dot.source() == "figure" and dot.text() == "from figure (Ticks 10 pt)"
    assert panel.figure_link.isVisibleTo(panel)
    window.pane.show_element("title")
    assert panel.fields.dots["size_pt"].text() == "from figure (Title 18 pt)"
    panel.fields.size.setValue(14.0)                                          # the element's own
    _settle()
    assert panel.fields.dots["size_pt"].source() == "element"
    assert window.state.text_style("title").size_pt == 14.0


def test_the_link_back_to_the_figures_sizes_selects_the_figure(qtbot):
    window = F.bare(qtbot)
    window.widget.apply_appearance(replace(window.state, tick_size=10.0))
    panel = window.pane.panel_for("axis.left.ticks")
    window.pane.show_element("axis.left.ticks")
    _settle()
    H.click(panel.figure_link)
    assert window.hub.current() == ElementRef("", "figure")


def test_a_click_on_the_marker_clears_that_one_property_in_one_undo_step(qtbot):
    window = F.bare(qtbot)
    window.widget.apply_appearance(replace(window.state, title="Kept words"))
    window.widget.apply_appearance(window.state.with_text_style(
        "title", TextStyle(text="Kept words", bold=True, italic=True, size_pt=16.0, color="#d55e00")))
    panel = window.pane.panel_for("title")
    window.pane.show_element("title")
    _settle()
    assert set(_sources(panel).values()) == {"element", "default"}
    assert _sources(panel)["bold"] == "element" and _sources(panel)["size_pt"] == "element"
    before = window.widget.history_mark()
    H.click(panel.fields.dots["bold"])
    _settle()
    style = window.state.text_style("title")
    assert style == TextStyle(text="Kept words", italic=True, size_pt=16.0, color="#d55e00")
    assert window.widget.history_mark() == before + 1
    assert _sources(panel)["bold"] == "default" and _sources(panel)["italic"] == "element"
    assert window.widget.undo() and window.state.text_style("title").bold is True


def test_clearing_the_size_marker_goes_back_to_the_figures_size_not_to_nothing(qtbot):
    window = F.bare(qtbot)
    window.widget.apply_appearance(replace(window.state, tick_size=10.0).with_text_style(
        "axis.left.ticks", TextStyle(size_pt=14.0)))
    panel = window.pane.panel_for("axis.left.ticks")
    window.pane.show_element("axis.left.ticks")
    _settle()
    assert panel.fields.dots["size_pt"].source() == "element"
    H.click(panel.fields.dots["size_pt"])
    _settle()
    assert window.state.text_style("axis.left.ticks").size_pt is None and window.state.tick_size == 10.0
    assert panel.fields.dots["size_pt"].source() == "figure"


def test_the_fit_label_marks_its_size_and_colour_and_clearing_one_keeps_the_wording(qtbot):
    window = F.bare(qtbot)
    window.widget.set_fit_label_available(True)
    window.controllers[0].host.register_fit_label(FitLabelItem(window.plots[0].getViewBox()))
    window.pane.refresh()
    _settle()
    window.widget.apply_appearance(replace(window.state, fit_label=FitLabelState(
        template="{equation} fitted", size_pt=12.0, color="#009e73")))
    panel = window.pane.panel_for("annotation.fit")
    window.pane.show_element("annotation.fit")
    _settle()
    assert _sources(panel)["size_pt"] == "element" and _sources(panel)["color"] == "element"
    H.click(panel.fields.dots["color"])
    _settle()
    label = window.state.fit_label_state()
    assert (label.template, label.size_pt, label.color) == ("{equation} fitted", 12.0, None)


# -- Distributions' defaults, and the figure's own fields ------------------------------------------------

DISTRIBUTIONS = PlotAppearance(grid=True, background="dark", data_color="#0072B2",
                               fit_color="#D55E00", line_width=1.0, marker_size=7.0,
                               legend_visible=True)


def test_values_a_page_reports_at_their_defaults_are_not_marked_as_set(qtbot):
    window = F.bare(qtbot, defaults=lambda: DISTRIBUTIONS)
    window.widget.set_state(DISTRIBUTIONS)             # what Distributions installs: seven concrete values
    for key in ("figure", "series.data", "legend"):
        panel = window.pane.panel_for(key)
        window.pane.show_element(key)
        _settle()
        assert {dot.source() for dot in panel._dots.values()} == {"default"}, key
    window.widget.apply_appearance(replace(window.state, grid=False, line_width=2.0))
    _settle()
    figure = window.pane.panel_for("figure")
    window.pane.show_element("figure")
    assert figure._dots["grid"].source() == "element" and figure._dots["background"].source() == "default"
    series = window.pane.panel_for("series.data")
    window.pane.show_element("series.data")
    assert series._dots["line_width"].source() == "element" and series._dots["marker_size"].source() == "default"


def test_without_a_defaults_provider_every_non_none_field_is_marked(qtbot):
    window = F.bare(qtbot)                              # defaults are PlotAppearance(): all None
    window.widget.set_state(DISTRIBUTIONS)
    figure = window.pane.panel_for("figure")
    window.pane.show_element("figure")
    _settle()
    assert figure._dots["grid"].source() == "element" and figure._dots["background"].source() == "element"


def test_a_click_on_a_figure_marker_clears_that_field_in_one_step(qtbot):
    window = F.bare(qtbot)
    window.widget.apply_appearance(replace(window.state, font_family="Times New Roman",
                                           title="Kept"))
    figure = window.pane.panel_for("figure")
    window.pane.show_element("figure")
    _settle()
    dot = figure._dots["font_family"]
    assert dot.source() == "element"
    before = window.widget.history_mark()
    H.click(dot)
    _settle()
    assert window.state.font_family is None and window.state.title == "Kept"
    assert window.widget.history_mark() == before + 1 and dot.source() == "default"


def test_the_text_style_fields_report_each_change_as_one_property(qtbot):
    fields = TextStyleFields(live=True, inherit_label="Figure font", dots=True)
    qtbot.addWidget(fields)
    seen = []
    fields.changed.connect(lambda prop, value: seen.append((prop, value)))
    fields.set_style(TextStyle(bold=True, size_pt=12.0))                       # silent
    assert seen == []
    fields.bold.setChecked(False)
    fields.size.setValue(0)
    fields.dots["italic"].cleared.emit()
    assert seen == [("bold", None), ("size_pt", None), ("italic", None)]
    assert fields.style() == TextStyle()


def test_typing_a_size_applies_it_once_when_finished_not_a_digit_at_a_time(qtbot):
    from PySide6 import QtCore, QtTest

    fields = TextStyleFields(live=True, inherit_label="Figure font")
    qtbot.addWidget(fields)
    fields.show()
    qtbot.waitExposed(fields)
    seen = []
    fields.changed.connect(lambda prop, value: seen.append((prop, value)))
    fields.size.setFocus()
    fields.size.selectAll()
    QtTest.QTest.keyClicks(fields.size, "40")           # "4" is a size too: it must not be applied on the way
    assert seen == []
    QtTest.QTest.keyClick(fields.size, QtCore.Qt.Key.Key_Return)
    assert seen == [("size_pt", 40.0)]


def test_the_size_box_steps_from_default_to_the_smallest_size_a_text_draws(qtbot):
    fields = TextStyleFields(live=True)
    qtbot.addWidget(fields)
    seen = []
    fields.changed.connect(lambda prop, value: seen.append((prop, value)))
    fields.size.stepBy(1)
    fields.size.stepBy(1)
    assert seen == [("size_pt", 4.0), ("size_pt", 5.0)]            # never 1 pt, which a style would drop
    fields.size.stepBy(-2)
    assert seen[-1] == ("size_pt", None) and fields.size.value() == 0
