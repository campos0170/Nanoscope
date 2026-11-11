"""Phase 0 follow-up, R1: what a recalled template promises, and when the promise is off.

A template recalled into Distribution summaries, Comparison figures or Voltage dependence arrives
with its words (and, in Distribution summaries, the ranges pinned on its value axes) and with its
quantity, which those windows only commit when Generate or Apply fit is pressed. The commit that
draws the template's own quantity keeps the template's words: they were written for it.

That is a promise about the words and pins the template INSTALLED, so it holds only while the
figure still has exactly those. The **Undo** of the recall (which brings back the words and pins
of the figure that is still on screen), a word or range changed after it, or a project saved and
reopened in between, leave on the figure what was typed for the quantity it shows now: the commit
is then an ordinary change of quantity and the rule runs (one Undo step, the sentence). Before, a
flag that nothing but the commit ever cleared kept the OLD figure's words and ranges over the new
quantity, with no step and no sentence.

Every gesture is a real one: the template is picked in its combo where the page shows one, the
Undo is the key, words are typed on the figure or in the Appearance boxes, ranges are pinned in
Axis properties, Generate and Apply fit are the buttons, and a save is a real ``_write_project``
reopened by ``_open_project``. Synthetic data only; offscreen.

2026-11-11: a page of the main window's Figures workspace shows no Templates combo (8661d35: saved
templates live under New figure, and the Explorer drops the Recipes group of every figure). The
Summaries tests, which run on such a page, therefore reach the recall through the page's template
bar (``TemplateBar.recall_template``, the operation the combo calls): no person can do that on that
page. ``_recall`` asserts the combo is not on screen, so the day it comes back those tests say so.
Whether this code (``_words_for``, ``promise_stands``) stays is an owner decision that is still
pending; until it is made, the Summaries tests pin the mechanism the guide documents, not a gesture
of the app. The pages in a bare Explorer (Voltage dependence, Comparison figures here) still show
the combo, and their tests pick in it.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import AxisPin

from .test_p0_pins_clear_with_the_words import _press_redo
from .test_p0_reset_on_quantity_change import (  # noqa: F401 -- the fixtures build the pages
    _boxes, _clear_status, _click_button, _drawn_title, _in_explorer, _label, _pick, _said,
    _section, _type_comparison_words, _type_in_box, _user_picks, comparison, summaries, vdep)
from .test_plot_window_routes import (
    _edit_double_click, _pin_through_axis_properties, _press_undo, _shown)


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


TEMPLATE_KINDS = ("distribution-summaries", "comparison-figures", "voltage-dependence")


@pytest.fixture(autouse=True)
def _no_templates():
    """Templates live in the (redirected) settings store, which outlives a test: start and end with
    none, so a name saved by one test is free for the next."""
    from nanoscope.gui import plot_templates as pt

    for kind in TEMPLATE_KINDS:
        pt.store_window_templates(kind, {})
    yield
    for kind in TEMPLATE_KINDS:
        pt.store_window_templates(kind, {})


def _generate(page):
    button = page._workflow.primary_actions[0]
    assert button.text().startswith("Generate")
    _click_button(button)


def _recall(page, name):
    """Pick ``name`` in the Templates combo of the Export section: the template's door.

    A page of the Figures workspace has no such door (saved templates are recalled with New figure
    there). With the Export section open, where a person would look for it, the combo is asserted
    off screen, and the template bar's own operation, which the combo calls, recalls the template.
    If the door ever comes back to those pages this fails, and the helper goes back to picking in
    the combo for them too."""
    _section(page, "Export")
    if getattr(page, "_nanoscope_figure_managed", False):
        assert not page._templates.combo.isVisible()
        assert page._templates.recall_template(name)
        QtWidgets.QApplication.processEvents()
        return
    _pick(page._templates.combo, text=name)


# =================================================================================================
# Distribution summaries: words and the value-axis ranges
# =================================================================================================

DEPTH_BOX_PIN = (10.0, 30.0)           # pA, on the box plot's value axis (the categories view's Y)
DWELL_ECDF_PIN = (1.5, 3.0)            # s, on the ECDF's value axis (the curves view's X)


def _value_pins(page):
    """The ranges pinned on the value axis of each pair: (ECDF / Survival's X, Box / Violin's Y)."""
    state = page._appearance_widget.state()
    return (state.for_view("curves", primary="curves").axis_pin("bottom").range(),
            state.for_view("categories", primary="curves").axis_pin("left").range())


def _summaries_with_a_depth_box_template(page):
    """Save a template for the depth box plot (its own title and its own value range), then show the
    dwell ECDF with a title and a value range of its own: the state the tests start from."""
    plot, widget = page._plot, page._appearance_widget
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _section(page, "Analysis")
    _pick(page._view_combo, data="box")
    _generate(page)
    _edit_double_click(plot, plot.titleLabel, "Depth box (template)")
    _pin_through_axis_properties(plot, "left", *DEPTH_BOX_PIN)
    assert page._templates.save_template("depth box")

    _section(page, "Data")
    _pick(page._metric_combo, data="dwell_s")
    _section(page, "Analysis")
    _pick(page._view_combo, data="ecdf")
    _generate(page)
    _edit_double_click(plot, plot.titleLabel, "Dwell ECDF (mine)")
    _pin_through_axis_properties(plot, "bottom", *DWELL_ECDF_PIN)
    assert (page._applied_config["metric"], page._applied_config["kind"]) == ("dwell_s", "ecdf")
    assert widget.state().title == "Dwell ECDF (mine)"
    assert _value_pins(page)[0] == pytest.approx(DWELL_ECDF_PIN)
    _clear_status(page)
    return plot, widget


def test_summaries_generate_keeps_a_recalled_templates_words_and_pins(summaries):
    """The documented route, unchanged: recall, then Generate. The words and the value range the
    template carries were written for its quantity, so committing it resets nothing."""
    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    assert page._stale and widget.state().title == "Depth box (template)"
    assert _value_pins(page)[1] == pytest.approx(DEPTH_BOX_PIN)
    steps = widget._history.depth()

    _generate(page)

    assert (page._applied_config["metric"], page._applied_config["kind"]) == ("depth_pA", "box")
    assert widget.state().title == "Depth box (template)"
    assert _value_pins(page)[1] == pytest.approx(DEPTH_BOX_PIN)
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_summaries_undo_of_the_recall_makes_the_next_generate_an_ordinary_change(summaries):
    """The Undo of a recall puts the old figure's title and range back; the controls stay on the
    template's quantity. Generate then draws a depth box over words and a range typed for a dwell
    ECDF: the rule has to run, as it does for a pick."""
    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    _press_undo(page)
    assert widget.state().title == "Dwell ECDF (mine)"
    assert _value_pins(page)[0] == pytest.approx(DWELL_ECDF_PIN)
    assert (page._metric_combo.currentData(), page._view_combo.currentData()) == ("depth_pA", "box")
    steps = widget._history.depth()

    _generate(page)

    assert (page._applied_config["metric"], page._applied_config["kind"]) == ("depth_pA", "box")
    assert widget.state().title is None and _drawn_title(plot)[0] != "Dwell ECDF (mine)"
    assert _value_pins(page) == (None, None)
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer) == (
        "Title and value range reset: the plot now shows a box plot of median depth.", True)
    assert widget.undo() and widget.state().title == "Dwell ECDF (mine)"


def test_summaries_a_word_typed_after_the_recall_is_a_word_typed_on_the_old_figure(summaries):
    """The figure on screen still shows the dwell ECDF until Generate, so a title typed now is a
    title typed on it: the promise is off and the commit resets it, as it does after a pick."""
    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    _section(page, "Appearance")
    _type_in_box(widget.title, "Typed after the recall")
    QtTest.QTest.qWait(800)
    assert widget.state().title == "Typed after the recall"
    steps = widget._history.depth()

    _generate(page)

    assert widget.state().title is None and _value_pins(page) == (None, None)
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer)[0].startswith("Title and value range reset: the plot now shows")


def test_summaries_a_range_pinned_after_the_recall_is_a_range_pinned_on_the_old_figure(summaries):
    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    _pin_through_axis_properties(plot, "bottom", 2.0, 2.5)       # the ECDF on screen has no pin yet
    assert _value_pins(page)[0] == pytest.approx((2.0, 2.5))
    steps = widget._history.depth()

    _generate(page)

    assert widget.state().title is None and _value_pins(page) == (None, None)
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer)[0].startswith("Title and value range reset: the plot now shows")


def test_summaries_a_change_of_look_after_the_recall_keeps_the_promise(summaries):
    """Size, colour, font: the look is not the template's words or ranges, and a Generate that
    commits the template's own quantity still keeps them."""
    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    _section(page, "Appearance")
    widget.title_size.setValue(15.0)
    QtTest.QTest.qWait(800)
    steps = widget._history.depth()

    _generate(page)

    assert (page._applied_config["metric"], page._applied_config["kind"]) == ("depth_pA", "box")
    assert widget.state().title == "Depth box (template)" and widget.state().title_size == 15.0
    assert _value_pins(page)[1] == pytest.approx(DEPTH_BOX_PIN)
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_summaries_undo_and_redo_of_the_recall_bring_the_promise_back_with_the_words(summaries):
    """Redo puts the template's words and range back exactly as it installed them, so the Generate
    after it commits the template's quantity and keeps them; the Generate after the Undo does not."""
    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    _press_undo(page)
    assert widget.state().title == "Dwell ECDF (mine)"
    _press_redo(page)
    assert widget.state().title == "Depth box (template)"
    steps = widget._history.depth()

    _generate(page)

    assert widget.state().title == "Depth box (template)"
    assert _value_pins(page)[1] == pytest.approx(DEPTH_BOX_PIN)
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_summaries_a_project_saved_between_the_recall_and_the_generate_carries_no_promise(
        summaries, qtbot, tmp_path):
    """The promise lives beside the draft and is not saved: the project holds the words and the
    range as they stand, and the Generate that follows the reopening is an ordinary change of
    quantity over them (the safe direction: words go to automatic, never to a wrong label)."""
    from .test_plot_window_routes import _main_window

    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    project = tmp_path / "recalled.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_distribution_summaries()
    page = reopened._distribution_summary_window
    from .test_p0_reset_on_quantity_change import _explorer_of

    explorer = _explorer_of(reopened)
    page.activateWindow()
    widget = page._appearance_widget
    assert widget.state().title == "Depth box (template)"
    assert (page._metric_combo.currentData(), page._view_combo.currentData()) == ("depth_pA", "box")
    assert page._stale and page._words_for is None
    steps = widget._history.depth()

    _generate(page)

    assert (page._applied_config["metric"], page._applied_config["kind"]) == ("depth_pA", "box")
    assert widget.state().title is None and _value_pins(page) == (None, None)
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer)[0].startswith("Title and value range reset: the plot now shows")
    reopened._dirty = False


def test_summaries_a_pick_after_the_recall_makes_the_commit_an_ordinary_change(summaries):
    """The promise is for the template's quantity: with the words and the range untouched, a
    Generate of ANOTHER measurement is not the commit they were written for."""
    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    _section(page, "Data")
    _pick(page._metric_combo, data="frac_depth")
    assert widget.state().title == "Depth box (template)"
    steps = widget._history.depth()

    _generate(page)

    assert page._applied_config["metric"] == "frac_depth"
    assert widget.state().title is None and _value_pins(page) == (None, None)
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer)[0].startswith("Title and value range reset: the plot now shows")


# =================================================================================================
# Voltage dependence: the fit label's wording
# =================================================================================================

def _vdep_with_an_exponential_template(page):
    """Save a template that carries the exponential trend and the wording typed for it, then show
    the linear trend with wording of its own."""
    plot, widget = page.plot_item(), page._appearance_widget
    page.set_fit("exponential")
    _edit_double_click(plot, page._fit_label, "Exponential trend: {equation}")
    assert page._templates.save_template("exp trend")
    page.set_fit("linear")                      # the rule takes the exponential wording back
    _edit_double_click(plot, page._fit_label, "Mine for the linear fit: {equation}")
    QtTest.QTest.qWait(800)
    assert page._applied_fit == "linear"
    assert widget.state().fit_label_state().template == "Mine for the linear fit: {equation}"
    _clear_status(page)
    return plot, widget


def _apply_fit(page):
    _section(page, "Analysis")
    _click_button(page._fit_apply_btn)


def test_voltage_dependence_apply_fit_keeps_the_wording_a_recalled_template_brought(vdep):
    """The documented route, unchanged: recall, then Apply fit."""
    page, explorer = vdep
    plot, widget = _vdep_with_an_exponential_template(page)
    page.activateWindow()
    _recall(page, "exp trend")
    assert page._fit_combo.currentText() == "exponential" and page._applied_fit == "linear"
    assert widget.state().fit_label_state().template == "Exponential trend: {equation}"
    steps = widget._history.depth()

    _apply_fit(page)

    assert page._applied_fit == "exponential"
    assert widget.state().fit_label_state().template == "Exponential trend: {equation}"
    assert page.fit_label_text().startswith("Exponential trend:")
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_voltage_dependence_undo_of_the_recall_makes_the_next_apply_fit_an_ordinary_change(vdep):
    """Undo brings back wording typed for the LINEAR fit; the combo stays on the exponential one.
    Apply fit would have drawn that wording over the exponential trend with no step and no
    sentence."""
    page, explorer = vdep
    plot, widget = _vdep_with_an_exponential_template(page)
    page.activateWindow()
    _recall(page, "exp trend")
    _press_undo(page)
    assert widget.state().fit_label_state().template == "Mine for the linear fit: {equation}"
    assert page._fit_combo.currentText() == "exponential" and page._applied_fit == "linear"
    steps = widget._history.depth()

    _apply_fit(page)

    assert page._applied_fit == "exponential"
    assert widget.state().fit_label_state().template == ""
    assert page.fit_label_text() and not page.fit_label_text().startswith("Mine for")
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer) == (
        "Fit label wording reset: the plot now shows the exponential fit.", True)
    assert widget.undo() and page.fit_label_text().startswith("Mine for the linear fit:")


def test_voltage_dependence_wording_typed_after_the_recall_is_wording_for_the_old_trend(vdep):
    page, explorer = vdep
    plot, widget = _vdep_with_an_exponential_template(page)
    page.activateWindow()
    _recall(page, "exp trend")
    _edit_double_click(plot, page._fit_label, "Typed after the recall: {equation}")
    QtTest.QTest.qWait(800)
    assert widget.state().fit_label_state().template == "Typed after the recall: {equation}"
    steps = widget._history.depth()

    _apply_fit(page)

    assert page._applied_fit == "exponential" and widget.state().fit_label_state().template == ""
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer)[0] == "Fit label wording reset: the plot now shows the exponential fit."


def test_voltage_dependence_a_change_of_look_after_the_recall_keeps_the_promise(vdep):
    page, explorer = vdep
    plot, widget = _vdep_with_an_exponential_template(page)
    page.activateWindow()
    _recall(page, "exp trend")
    widget.fit_label_size.setValue(9.0)                        # the label's size is its look
    QtTest.QTest.qWait(800)
    steps = widget._history.depth()

    _apply_fit(page)

    assert widget.state().fit_label_state().template == "Exponential trend: {equation}"
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_voltage_dependence_undo_and_redo_of_the_recall_bring_the_promise_back(vdep):
    page, explorer = vdep
    plot, widget = _vdep_with_an_exponential_template(page)
    page.activateWindow()
    _recall(page, "exp trend")
    _press_undo(page)
    _press_redo(page)
    assert widget.state().fit_label_state().template == "Exponential trend: {equation}"
    steps = widget._history.depth()

    _apply_fit(page)

    assert widget.state().fit_label_state().template == "Exponential trend: {equation}"
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_voltage_dependence_another_trend_than_the_templates_is_an_ordinary_change(vdep):
    """The wording came with the exponential trend; applying no fit at all is not that trend."""
    page, explorer = vdep
    plot, widget = _vdep_with_an_exponential_template(page)
    page.activateWindow()
    _recall(page, "exp trend")
    assert widget.state().fit_label_state().template == "Exponential trend: {equation}"
    steps = widget._history.depth()
    _section(page, "Analysis")
    _user_picks(page._fit_combo, "none")

    _click_button(page._fit_apply_btn)

    assert page._applied_fit == "none" and widget.state().fit_label_state().template == ""
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer)[0] == "Fit label wording reset: the plot now shows no fit."


# =================================================================================================
# Comparison figures: the three boxes
# =================================================================================================

def _comparison_with_an_events_template(page, drawn):
    """Save a template for the individual-event figure (with its own title and labels), then show the
    first figure with boxes of its own."""
    widget = page._appearance_widget
    first = page._view_combo.currentData()
    _section(page, "Data")
    _pick(page._view_combo, data="events")
    _generate(page)
    _section(page, "Appearance")
    _type_in_box(widget.title, "Template title")
    _type_in_box(widget.x_label, "Tx")
    _type_in_box(widget.y_label, "Ty")
    QtTest.QTest.qWait(800)
    assert page._templates.save_template("events")
    _section(page, "Data")
    _pick(page._view_combo, data=first)
    _generate(page)
    _type_comparison_words(page)
    assert (drawn[-1]["title"], drawn[-1]["x"], drawn[-1]["y"]) == ("Mine", ["Dw"], ["Dp"])
    _clear_status(page)
    return widget


def test_comparison_generate_keeps_the_boxes_a_recalled_template_brought(comparison):
    page, explorer, drawn = comparison
    widget = _comparison_with_an_events_template(page, drawn)
    page.activateWindow()
    _recall(page, "events")
    assert page._stale and _boxes(widget) == ("Template title", "Tx", "Ty")
    steps = widget._history.depth()

    _generate(page)

    assert drawn[-1]["title"] == "Template title" and drawn[-1]["x"] == ["Tx"]
    assert widget._history.depth() == steps and not page._status.text().startswith("Title")


def test_comparison_undo_of_the_recall_makes_the_next_generate_an_ordinary_change(comparison):
    page, explorer, drawn = comparison
    widget = _comparison_with_an_events_template(page, drawn)
    page.activateWindow()
    _recall(page, "events")
    _section(page, "Appearance")
    _press_undo(page)
    assert _boxes(widget) == ("Mine", "Dw", "Dp")
    assert page._view_combo.currentData() == "events"
    steps = widget._history.depth()

    _generate(page)

    last = drawn[-1]
    assert last["title"] == "Individual-event comparison"
    assert "Dw" not in last["x"] and "Dp" not in last["y"]
    assert _boxes(widget) == ("", "", "")
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert page._status.text().startswith("Title, X label and Y label reset: the plot now shows")


def test_comparison_a_word_typed_after_the_recall_is_reset_by_the_commit(comparison):
    page, explorer, drawn = comparison
    widget = _comparison_with_an_events_template(page, drawn)
    page.activateWindow()
    _recall(page, "events")
    _section(page, "Appearance")
    _type_in_box(widget.title, "Typed after the recall")
    QtTest.QTest.qWait(800)
    steps = widget._history.depth()

    _generate(page)

    assert drawn[-1]["title"] == "Individual-event comparison" and _boxes(widget) == ("", "", "")
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert page._status.text().startswith("Title, X label and Y label reset: the plot now shows")


def test_comparison_a_change_of_look_after_the_recall_keeps_the_promise(comparison):
    page, explorer, drawn = comparison
    widget = _comparison_with_an_events_template(page, drawn)
    page.activateWindow()
    _recall(page, "events")
    _section(page, "Appearance")
    widget.title_size.setValue(13.0)
    QtTest.QTest.qWait(800)
    steps = widget._history.depth()

    _generate(page)

    assert drawn[-1]["title"] == "Template title" and drawn[-1]["size"] == 13.0
    assert widget._history.depth() == steps and not page._status.text().startswith("Title")


def test_comparison_another_measurement_than_the_templates_is_an_ordinary_change(comparison):
    page, explorer, drawn = comparison
    widget = _comparison_with_an_events_template(page, drawn)
    page.activateWindow()
    _recall(page, "events")
    assert _boxes(widget) == ("Template title", "Tx", "Ty")
    _section(page, "Analysis")
    current = page._metric.currentData()
    other = next(page._metric.itemData(i) for i in range(page._metric.count())
                 if page._metric.itemData(i) != current)
    _pick(page._metric, data=other)
    steps = widget._history.depth()

    _generate(page)

    assert _boxes(widget) == ("", "", "") and drawn[-1]["title"] != "Template title"
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert page._status.text().startswith("Title, X label and Y label reset: the plot now shows")
