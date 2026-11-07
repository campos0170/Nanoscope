"""Phase 1, Format group: Reset element, Reset look, Copy format and Paste format.

All of them use the existing override model and the existing writers, each is ONE Undo step, and
none ever touches the words: a typed title, a y label, a fit label's wording, a legend name or a
caption survive every one (the text trap: ``PlotAppearance.with_text_style`` stores a style's text
into the label, the fit label's template and so on, so every writer starts from the target's
current style). Real clicks on the footer's buttons; what is asserted is what is stored. Synthetic
data only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import AxisPin, TextStyle
from nanoscope.gui.fit_label import FitLabelState
from nanoscope.gui.format_inspector import ElementRef, FormatClipboard
from nanoscope.gui.plot_appearance import PlotAppearance

from . import p1_format_rig as F
from . import p1_hosting as H

STYLED = TextStyle(bold=True, italic=True, size_pt=15.0, color="#009e73",
                   family="Times New Roman")


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _settle():
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()


def _rig(qtbot):
    page, explorer = F.hosted_rich(qtbot)
    explorer.inspector().set_tab(1)
    _settle()
    return page, explorer, explorer.inspector().pane("iv").format_pane


def _style(page, key) -> TextStyle:
    return page._appearance_widget.state().text_style(key)


def _marks(page) -> int:
    return page._appearance_widget.history_mark()


def _style_it(page, key, style=STYLED):
    """Give ``key`` a hand-set style, keeping its words (the writer the panels use)."""
    page.controller.apply_style(key, replace(page.controller.target(key).style,
                                             **{k: v for k, v in style.to_dict().items()
                                                if k != "text"}))
    page._appearance_widget._close_burst()


def _words(page):
    """The five words a paste must never move."""
    state = page._appearance_widget.state()
    return {"title": state.title, "y": state.y_label, "fit": state.fit_label_state().template,
            "legend": state.text_style("legend.entry.second").text,
            "caption": state.text_style(F.CAPTION).text}


def _type_every_word(page):
    page.controller.set_style_property("annotation.fit", "text", "{equation} fitted")
    page.controller.set_style_property("legend.entry.second", "text", "Renamed")
    page.controller.set_style_property(F.CAPTION, "text", "Typed caption")
    page._appearance_widget._close_burst()


# -- Reset element ---------------------------------------------------------------------------------------

def test_reset_element_clears_the_style_of_a_text_and_keeps_its_words(qtbot):
    page, explorer, fp = _rig(qtbot)
    _style_it(page, "title")
    assert _style(page, "title").bold is True
    fp.show_element("title")
    before = _marks(page)
    H.click(fp.reset_button)
    assert _style(page, "title") == TextStyle(text="A typed title")       # the look went, the words stayed
    assert _marks(page) == before + 1
    assert page._appearance_widget.undo() and _style(page, "title").bold is True


def test_reset_element_on_the_fit_label_resets_its_size_and_colour_and_keeps_the_wording(qtbot):
    page, explorer, fp = _rig(qtbot)
    _type_every_word(page)
    page._appearance_widget.apply_appearance(replace(
        page._appearance_widget.state(),
        fit_label=replace(page._appearance_widget.state().fit_label_state(), size_pt=12.0,
                          color="#d55e00")))
    _style_it(page, "annotation.fit", TextStyle(bold=True))
    fp.show_element("annotation.fit")
    before = _marks(page)
    H.click(fp.reset_button)
    label = page._appearance_widget.state().fit_label_state()
    assert (label.template, label.size_pt, label.color) == ("{equation} fitted", None, None)
    assert _style(page, "annotation.fit").bold is None and _marks(page) == before + 1


def test_reset_element_on_an_axis_is_one_step_for_both_styles_and_the_range(qtbot):
    page, explorer, fp = _rig(qtbot)
    widget = page._appearance_widget
    _style_it(page, "axis.left.label", TextStyle(italic=True))
    _style_it(page, "axis.left.ticks", TextStyle(size_pt=14.0))
    widget.apply_appearance(widget.state().with_axis_pin("left", AxisPin(min=0.0, max=2.0)))
    assert page._plot.viewRange()[1] == pytest.approx([0.0, 2.0])
    fp.show_element("axis.left")
    before = _marks(page)
    H.click(fp.reset_button)
    state = widget.state()
    assert state.text_style("axis.left.label") == TextStyle(text="Current (pA)")
    assert state.text_style("axis.left.ticks") == TextStyle()
    assert state.y_label == "Current (pA)" and state.axis_pin("left") == AxisPin()
    assert _marks(page) == before + 1                      # one step, not three
    assert widget.undo()
    assert widget.state().text_style("axis.left.label").italic is True
    assert widget.state().axis_pin("left") == AxisPin(min=0.0, max=2.0)


def test_reset_element_on_the_legend_and_the_series_clears_their_figure_wide_fields(qtbot):
    page, explorer, fp = _rig(qtbot)
    widget = page._appearance_widget
    widget.apply_appearance(replace(widget.state(), legend_visible=False, legend_size=14.0,
                                    line_width=3.0, marker_size=9.0, data_color="#009e73",
                                    fit_color="#cc79a7", grid=False))
    fp.show_element("legend")
    H.click(fp.reset_button)
    assert (widget.state().legend_visible, widget.state().legend_size) == (None, None)
    assert widget.state().line_width == 3.0 and widget.state().grid is False
    fp.show_element("series.data")
    H.click(fp.reset_button)
    state = widget.state()
    assert (state.line_width, state.marker_size, state.data_color, state.fit_color) == (None,) * 4
    assert state.grid is False                             # not the series' to reset


def test_reset_look_returns_the_look_and_keeps_words_pins_places_and_legend_styles(qtbot):
    page, explorer, fp = _rig(qtbot)
    widget = page._appearance_widget
    _type_every_word(page)
    _style_it(page, "title")
    _style_it(page, "legend.entry.second", TextStyle(italic=True))
    state = widget.state()
    widget.apply_appearance(replace(
        state, font_family="Times New Roman", background="light", grid=False, title_size=20.0,
        line_width=2.0).with_axis_pin("bottom", AxisPin(min=0.0, max=2.0)).with_annotation_offset(
        F.CAPTION, (40.0, 50.0)))
    fp.show_element("figure")
    before = _marks(page)
    H.click(fp.panel_for("figure").reset_look)
    after = widget.state()
    assert (after.font_family, after.background, after.grid, after.title_size, after.line_width) == (None,) * 5
    assert after.text_style("title") == TextStyle(text="A typed title") and after.title == "A typed title"
    assert after.text_style("legend.entry.second") == TextStyle(text="Renamed", italic=True)
    assert after.axis_pin("bottom") == AxisPin(min=0.0, max=2.0)
    assert after.annotation_offset(F.CAPTION) == (40.0, 50.0)
    assert after.fit_label_state().template == "{equation} fitted"
    assert _marks(page) == before + 1


def test_reset_element_on_the_figure_is_reset_look(qtbot):
    page, explorer, fp = _rig(qtbot)
    widget = page._appearance_widget
    widget.apply_appearance(replace(widget.state(), grid=False, title_size=20.0))
    fp.show_element("figure")
    H.click(fp.reset_button)
    assert widget.state().grid is None and widget.state().title == "A typed title"


# -- Copy format and Paste format: the text trap, one test per kind of text ---------------------------------

#: (target key, how the target's words are read back)
TARGETS = {
    "title": lambda w: w["title"], "axis.left.label": lambda w: w["y"],
    "annotation.fit": lambda w: w["fit"], "legend.entry.second": lambda w: w["legend"],
    F.CAPTION: lambda w: w["caption"]}


@pytest.mark.parametrize("target", list(TARGETS))
def test_paste_onto_each_kind_of_text_keeps_its_words(qtbot, target):
    page, explorer, fp = _rig(qtbot)
    widget = page._appearance_widget
    _type_every_word(page)
    source = "axis.bottom.label" if target == "title" else "title"
    _style_it(page, source, replace(STYLED, align="left") if source == "title" else STYLED)
    words = _words(page)
    fp.show_element(source)
    H.click(fp.copy_button)
    fp.show_element(target)
    assert fp.paste_button.isEnabled()
    before = _marks(page)
    H.click(fp.paste_button)
    after = _style(page, target)
    assert (after.bold, after.italic, after.family) == (True, True, "Times New Roman"), after
    assert after.size_pt == 15.0 and after.color == "#009e73"
    assert TARGETS[target](_words(page)) == TARGETS[target](words)          # the words did not move
    assert _words(page) == words
    if target == "annotation.fit":      # its size and colour live in the fit label's own state
        assert widget.state().fit_label_state().size_pt == 15.0
        assert widget.state().fit_label_state().template == "{equation} fitted"
    assert after.align is None                                               # only the title aligns
    assert _marks(page) == before + 1                                       # one Undo step
    assert widget.undo() and _style(page, target).bold is None
    assert _words(page) == words


def test_a_style_copied_from_an_axis_label_pastes_onto_the_title_without_moving_its_words(qtbot):
    page, explorer, fp = _rig(qtbot)
    _style_it(page, "axis.bottom.label", STYLED)
    fp.show_element("axis.bottom.label")
    H.click(fp.copy_button)
    fp.show_element("title")
    H.click(fp.paste_button)
    assert _style(page, "title").bold is True and page._appearance_widget.state().title == "A typed title"


def test_an_axis_copies_both_its_styles_and_never_its_range_or_words(qtbot):
    page, explorer, fp = _rig(qtbot)
    widget = page._appearance_widget
    _style_it(page, "axis.bottom.label", TextStyle(italic=True, size_pt=13.0))
    _style_it(page, "axis.bottom.ticks", TextStyle(bold=True))
    widget.apply_appearance(widget.state().with_axis_pin("bottom", AxisPin(min=0.0, max=2.0)))
    fp.show_element("axis.bottom")
    H.click(fp.copy_button)
    fp.show_element("axis.left")
    before = _marks(page)
    H.click(fp.paste_button)
    state = widget.state()
    assert state.text_style("axis.left.label") == TextStyle(text="Current (pA)", italic=True, size_pt=13.0)
    assert state.text_style("axis.left.ticks") == TextStyle(bold=True)
    assert state.axis_pin("left") == AxisPin()                               # no range travels
    assert state.x_label == "Voltage (V)" and _marks(page) == before + 1


def test_the_figures_look_is_copied_without_its_words_and_pasted_as_one_step(qtbot):
    page, explorer, fp = _rig(qtbot)
    widget = page._appearance_widget
    _type_every_word(page)
    widget.apply_appearance(replace(widget.state(), font_family="Times New Roman", grid=False,
                                    background="light").with_axis_pin("bottom", AxisPin(min=0.0, max=2.0)))
    fp.show_element("figure")
    H.click(fp.copy_button)
    held = fp._clipboard.get()
    assert held.look.title is None and held.look.axes is None and held.look.y_label is None
    widget.apply_appearance(replace(widget.state(), font_family=None, grid=None, background=None,
                                    title="Another title"))
    before = _marks(page)
    H.click(fp.paste_button)
    state = widget.state()
    assert (state.font_family, state.grid, state.background) == ("Times New Roman", False, "light")
    assert state.title == "Another title" and state.axis_pin("bottom") == AxisPin(min=0.0, max=2.0)
    assert state.fit_label_state().template == "{equation} fitted"
    assert _marks(page) == before + 1


def test_figure_paste_keeps_the_destination_fit_labels_switches_and_style(qtbot):
    page, explorer, fp = _rig(qtbot)
    widget = page._appearance_widget
    widget.apply_appearance(replace(widget.state(), fit_label=FitLabelState(visible=False)))
    fp.show_element("figure")
    H.click(fp.copy_button)
    target = replace(widget.state(), fit_label=FitLabelState(
        visible=True, template="destination {equation}", size_pt=13.0, color="#009e73"))
    target = target.with_text_style("annotation.fit", replace(
        target.text_style("annotation.fit"), bold=True))
    widget.apply_appearance(target)
    H.click(fp.paste_button)
    assert widget.state().fit_label == target.fit_label
    assert widget.state().text_style("annotation.fit") == target.text_style("annotation.fit")


# -- the buttons' own rules ------------------------------------------------------------------------------

def test_paste_waits_for_something_copied_that_fits_the_selection(qtbot):
    page, explorer, fp = _rig(qtbot)
    fp.show_element("title")
    assert fp.copy_button.isEnabled() and not fp.paste_button.isEnabled()    # nothing copied yet
    H.click(fp.copy_button)
    assert fp.paste_button.isEnabled()                                       # a text onto a text
    fp.show_element("axis.left")
    assert not fp.paste_button.isEnabled()                                   # a text onto an axis
    fp.show_element("figure")
    assert not fp.paste_button.isEnabled()                                   # a text onto the figure
    fp.show_element("axis.bottom.ticks")
    assert fp.paste_button.isEnabled()
    fp.show_element("series.data")
    assert not fp.copy_button.isEnabled() and not fp.paste_button.isEnabled()
    assert fp.reset_button.isEnabled()
    fp.show_element("legend.entry.series")
    assert fp.paste_button.isEnabled()


def test_the_clipboard_is_in_memory_one_per_explorer_and_never_saved(qtbot):
    page, explorer, fp = _rig(qtbot)
    _style_it(page, "title")
    fp.show_element("title")
    system = QtGui.QGuiApplication.clipboard().text()
    keys = set(QtCore.QSettings().allKeys())
    saved = page.appearance_state()
    H.click(fp.copy_button)
    assert isinstance(fp._clipboard, FormatClipboard) and fp._clipboard.get() is not None
    assert QtGui.QGuiApplication.clipboard().text() == system
    assert set(QtCore.QSettings().allKeys()) == keys
    assert page.appearance_state() == saved                                  # nothing in the figure
    other, other_explorer = F.hosted_rich(qtbot, "voltage_series")
    other_explorer.inspector().set_tab(1)
    other_fp = other_explorer.inspector().pane("voltage_series").format_pane
    other_fp.show_element("title")
    assert other_fp._clipboard is not fp._clipboard and other_fp._clipboard.get() is None
    assert not other_fp.paste_button.isEnabled()
