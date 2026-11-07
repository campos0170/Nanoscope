"""Phase 0, group G6B: the review's fixes that touch the reset rule (``gui.quantity_reset``).

1. A fit label whose wording the rule took back is the label nobody touched again, so Reset
   figure pressed next has nothing to do (it used to record a step that changed nothing).
2. The Undo and Redo tooltips of the step the rule records name what it took back
   ("reset title and Y label"), so it no longer passes for a real Reset figure.
3. The sentence on the status line stands only while what it names stays automatic: the Undo of
   the reset, a word typed since, a range pinned again take it down; Reset figure and its Undo, and
   a change of look, do not.
4. Distributions' Normalize (count, density, probability per bin) is what its y axis counts: a
   pick of it takes the words and the y range, in the step of the pick; the swing to density that
   ticking a second source brings by itself takes nothing.
5. Event waveforms' Align sets what t = 0 is on the ensemble's x axis: it takes the words and that
   axis's range, like Normalize time (and nothing while time is normalised).
6. A range Axis properties hands back as it opened, with Log scale changed beside it, is carried
   across to the units the Event-count map's axis is about to have.

Every gesture is a real one, as in ``test_p0_reset_on_quantity_change`` and
``test_p0_pins_clear_with_the_words`` (whose rigs this reuses): picks through the combo's popup,
words typed on the figure, ranges pinned through the Axis properties dialog by right-click on the
axis, Undo and Redo as the keys on the page's window, a click on the real Reset figure button,
and a save with ``_write_project`` reopened by ``_open_project``. What is asserted is what is
drawn, not only the state behind it. Synthetic data only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import AxisPin
from nanoscope.gui.fit_label import FitLabelState
from nanoscope.gui.plot_appearance import PlotAppearance, _describe_step

from .test_p0_pins_clear_with_the_words import (  # noqa: F401  (fixtures are used by name)
    _axes_of, _axis, _depth, _is_automatic, _pin, _pin_inside, _press_redo, _stays_pinned, _tab)
from .test_p0_reset_on_quantity_change import (  # noqa: F401
    _boxes, _clear_status, _click_button, _dist_words, _drawn_title, _explorer_of, _generate,
    _label, _pick, _said, _section, _series_window, _standing_line, _type_comparison_words,
    _type_dist_words, _type_event_words, _type_in_box, _type_noise_words, _type_subpop_words,
    _type_summary_words, _user_picks, comparison, dist, events, noise, subpop, subpop_report,
    summaries, vdep)
from .test_plot_window_routes import (
    _add_title, _edit_double_click, _main_window, _press_undo)

L = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _reset_figure(page):
    """Click the real Reset figure button of the page."""
    button = page._appearance_widget.reset_button
    _click_button(button)


# =================================================================================================
# 1. WINDOWS F1: a fit label whose wording went is the label nobody touched
# =================================================================================================

def test_the_rule_leaves_the_untouched_fit_label_as_none_and_keeps_one_that_was_set():
    from nanoscope.gui.quantity_reset import cleared_words

    typed = PlotAppearance(title="T", fit_label=FitLabelState(template="Linear: {equation}"))
    cleared, names = cleared_words(typed)
    assert names == ("Title", "Fit label wording")
    assert cleared.fit_label is None and cleared == PlotAppearance()
    # The wording alone (Voltage dependence's trend) does the same.
    cleared, names = cleared_words(typed, only=("fit_label",))
    assert cleared == PlotAppearance(title="T") and names == ("Fit label wording",)
    # A label whose numbers, place, size or colour were set keeps them: that is the look.
    for look in (dict(size_pt=9.0), dict(color="#009e73"), dict(fields=("equation", "bic")),
                 dict(visible=False), dict(placement="bottom-right"), dict(in_legend=True)):
        label = FitLabelState(template="Linear: {equation}", **look)
        cleared, _ = cleared_words(PlotAppearance(fit_label=label))
        assert cleared.fit_label == replace(label, template="") != FitLabelState(), look


def _vdep_wording(vdep):
    page, explorer = vdep
    plot = page.plot_item()
    _section(page, "Analysis")
    _edit_double_click(plot, page._fit_label, "Linear fit: {equation}")
    return page, explorer, plot, page._appearance_widget


def test_voltage_dependence_after_the_rule_reset_figure_has_nothing_to_do(vdep):
    """Before: the state held ``FitLabelState(template='')``, which is not ``PlotAppearance()``,
    so Reset figure recorded a step ('Undo: fit label style') that changed nothing visible."""
    page, explorer, plot, widget = _vdep_wording(vdep)
    assert page.fit_label_text().startswith("Linear fit:")

    _pick(page._metric_combo, text="Median depth (pA)")        # the rule takes the wording back

    assert not page.fit_label_text().startswith("Linear fit:")
    assert widget.state() == PlotAppearance() and widget.state().fit_label is None
    assert page.appearance_state() == {}
    steps = _depth(widget)
    _reset_figure(page)
    assert _depth(widget) == steps                              # nothing to reset: no step
    assert not widget.undo_button.toolTip().endswith("fit label style")
    _press_undo(page)                                           # the first Undo is the rule's
    assert page.fit_label_text().startswith("Linear fit:")
    assert widget.state().fit_label_state().template == "Linear fit: {equation}"


def test_voltage_dependence_a_fit_label_with_a_size_set_keeps_it_and_reset_figure_is_one_real_step(
        vdep):
    page, explorer, plot, widget = _vdep_wording(vdep)
    widget.fit_label_size.setValue(9.0)
    QtTest.QTest.qWait(800)
    _pick(page._metric_combo, text="Median depth (pA)")
    assert widget.state().fit_label == FitLabelState(size_pt=9.0)
    steps = _depth(widget)

    _reset_figure(page)

    assert _depth(widget) == (steps[0] + 1, 0)                  # a real step: the size goes
    assert widget.state() == PlotAppearance()
    assert widget.undo_button.toolTip() == "Undo: fit label size automatic"


def test_noise_after_the_rule_reset_figure_has_nothing_to_do(noise):
    page, explorer = noise
    widget = page._appearance_widget
    _edit_double_click(page._plot, page._fit_label, "Flicker: {equation}")
    assert page.fit_label_text().startswith("Flicker:")
    _section(page, "Data")

    _pick(page._view_combo, data="spectrogram")

    assert widget.state() == PlotAppearance() and page.appearance_state() == {}
    steps = _depth(widget)
    _reset_figure(page)
    assert _depth(widget) == steps


def test_distributions_after_the_rule_the_fit_label_is_empty_and_reset_figure_has_nothing_to_do(
        dist):
    page, explorer = dist
    _edit_double_click(page._plot, page._fit_label_item, "Single exponential: {equation}")
    assert page.config().style.fit_label_state().template.startswith("Single")

    _user_picks(page._hist_col, "frac_depth")                   # a column: the wording goes

    style = page.config().style
    assert not style.fit_label and style.fit_label_state() == FitLabelState()
    page._history_touch()
    steps = page._history.depth()
    _reset_figure(page)
    assert page._history.depth() == steps

    # A label with a size set keeps it, and the Reset is then one real step.
    _edit_double_click(page._plot, page._fit_label_item, "Again: {equation}")
    page._appearance_widget.fit_label_size.setValue(9.0)
    page._history_touch()
    _user_picks(page._hist_col, "depth_pA")
    assert page.config().style.fit_label_state() == FitLabelState(size_pt=9.0)
    steps = page._history.depth()
    _reset_figure(page)
    assert page._history.depth() == (steps[0] + 1, 0)
    assert not page.config().style.fit_label


# =================================================================================================
# 2. WINDOWS F4: the Undo and Redo tooltips of the rule's step say what it took back
# =================================================================================================

def test_describe_step_names_a_step_that_only_took_words_and_ranges_back():
    def pin(*sides, **ranges):
        look = PlotAppearance()
        for side in sides:
            look = look.with_axis_pin(side, AxisPin(min=0.0, max=5.0, **ranges))
        return look

    words = PlotAppearance(title="Capture vs voltage", y_label="Rate (Hz)")
    assert _describe_step(words, PlotAppearance()) == "reset title and Y label"
    # The look the figure has is kept by the step and does not change its name.
    look = PlotAppearance(title_size=14.0, background="light")
    assert _describe_step(replace(look, title="T", x_label="X", y_label="Y"), look) == (
        "reset title, X label and Y label")
    # The fit label's wording is a word; its numbers are not (a size kept is still a size).
    worded = PlotAppearance(title="T", fit_label=FitLabelState(template="Lin: {equation}",
                                                               size_pt=9.0))
    assert _describe_step(worded, PlotAppearance(title=None, fit_label=FitLabelState(size_pt=9.0))
                          ) == "reset title and fit label wording"
    # Words and ranges together, in the order the status line gives them.
    both = replace(pin("left", tick_major=1.0), title="T", y_label="Y")
    assert _describe_step(both, PlotAppearance()) == (
        "reset title, Y label, Y range and Y major tick")
    assert _describe_step(replace(pin("left"), title="T", y_label="Y"), PlotAppearance()
                          ) == "reset title, Y label and Y range"
    # Ranges alone: one axis keeps its older name, two axes are named.
    assert _describe_step(pin("left"), PlotAppearance()) == "y axis range automatic"
    assert _describe_step(pin("bottom", "left"), PlotAppearance()) == "reset X range and Y range"
    assert _describe_step(pin("embedding.bottom", "embedding.left"), PlotAppearance()
                          ) == "reset X range and Y range"
    assert _describe_step(pin("bottom", "left"), pin("bottom")) == "y axis range automatic"
    # Two views' pins on one side, taken together, are named once.
    assert _describe_step(pin("curves.bottom", "survival.bottom"), PlotAppearance()) == (
        "reset X range")
    # A single word keeps its own name.
    assert _describe_step(words, replace(words, title=None)) == (
        "title ‘Capture vs voltage’ → automatic")


def test_describe_step_leaves_every_other_step_as_it_was_named():
    words = PlotAppearance(title="T", y_label="Y", title_size=14.0)
    # A step that also moves the look is a Reset figure (or counted), never "reset <words>".
    assert _describe_step(words, PlotAppearance()) == "reset figure"
    assert _describe_step(replace(words, background="dark"),
                          replace(words, title=None, y_label=None)) == "3 appearance changes"
    assert _describe_step(PlotAppearance(), words) == "3 appearance changes"
    # Typing two labels is not taking them back.
    assert _describe_step(PlotAppearance(), PlotAppearance(title="T", y_label="Y")) == (
        "2 appearance changes")
    # Words changed to other words, a pin added, a pin moved: not a reset.
    assert _describe_step(PlotAppearance(title="T", y_label="Y"),
                          PlotAppearance(title="U", y_label=None)) == "2 appearance changes"
    one = PlotAppearance().with_axis_pin("left", AxisPin(min=0.0, max=5.0))
    assert _describe_step(PlotAppearance(title="T"), one) == "2 appearance changes"
    # A pin whose range moves to another range is not taken back either.
    other = PlotAppearance().with_axis_pin("left", AxisPin(min=1.0, max=6.0))
    assert _describe_step(replace(one, title="T"), replace(other, title=None)) == (
        "2 appearance changes")
    # A shipped preset is still named after itself.
    from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS
    name, preset = next(iter(_BUILTIN_PRESETS.items()))
    assert _describe_step(PlotAppearance(), preset) == f"{name} preset"


def test_voltage_dependence_the_rules_step_reads_as_what_it_took_back_not_as_a_reset_figure(vdep):
    """Before: 'Undo: reset figure' (the look is the default) or 'Undo: 2 appearance changes' (a
    look is set), the text of a real Reset figure, though no one pressed it."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    _section(page, "Analysis")

    _pick(page._metric_combo, text="Median depth (pA)")

    assert widget.state() == PlotAppearance() and page.appearance_state() == {}
    assert widget.undo_button.toolTip() == "Undo: reset title and Y label"
    _press_undo(page)
    assert widget.redo_button.toolTip() == "Redo: reset title and Y label"
    assert widget.undo_button.toolTip() == "Undo: y label automatic → ‘Rate (Hz)’"  # the typing
    _press_redo(page)
    assert widget.undo_button.toolTip() == "Undo: reset title and Y label"


def test_voltage_dependence_the_rules_step_keeps_its_name_when_a_look_or_a_range_is_set(vdep):
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    widget.title_size.setValue(14.0)
    widget.axis_size.setValue(12.0)
    QtTest.QTest.qWait(800)
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    _pin(plot, "left", 0.0, 5.0)
    _section(page, "Analysis")

    _pick(page._metric_combo, text="Median depth (pA)")

    assert (widget.state().title_size, widget.state().axis_size) == (14.0, 12.0)
    assert widget.undo_button.toolTip() == "Undo: reset title, Y label and Y range"
    _press_undo(page)
    assert widget.redo_button.toolTip() == "Redo: reset title, Y label and Y range"
    # A real Reset figure on a figure with a look keeps its own name.
    _press_redo(page)
    _reset_figure(page)
    assert widget.undo_button.toolTip() == "Undo: reset figure"
    _press_undo(page)
    assert widget.redo_button.toolTip() == "Redo: reset figure"


def test_a_reset_figure_on_a_figure_with_only_words_reads_like_the_rules_step(vdep):
    """Two steps with the same pair of values cannot be told apart by a history that holds only
    appearances: on a figure whose only customisation is typed words, Reset figure takes back
    exactly what the rule would, and the tooltip says so (it is accurate; 'reset figure' names a
    step that also moves the look)."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")

    _reset_figure(page)

    assert widget.state() == PlotAppearance()
    assert widget.undo_button.toolTip() == "Undo: reset title and Y label"


def test_distributions_has_no_appearance_undo_its_configuration_undo_names_no_step(dist):
    """The tooltips name the step on the pages with an appearance Undo (the guide says so).
    Distributions' Undo and Redo are its Configuration buttons, whose tooltips count steps: a step
    the rule records reads like any other change of the configuration."""
    import re

    page, explorer = dist
    _type_dist_words(page)
    page._history_touch()
    undo, redo = page._undo_btn, page._redo_btn

    _pick(page._plot_type_combo, text="Scatter")                # the rule takes the words back

    assert _dist_words(page)[0] == "" and page._plot_type_combo.currentText() == "Scatter"
    assert re.fullmatch(r"Undo the last change to this plot's configuration \(\d+ to step back\)\.",
                        undo.toolTip()), undo.toolTip()
    _press_undo(page)                                           # the words and the type come back
    assert page._plot_type_combo.currentText() == "Histogram" and _dist_words(page)[0] != ""
    assert redo.toolTip() == "Redo (1 to step forward).", redo.toolTip()


def test_subpopulations_a_pin_only_step_names_both_axes_it_took(subpop):
    page, explorer = subpop
    scatter = page.embedding_view()
    plot, widget = scatter._plot.getPlotItem(), page._appearance_widget
    _pin_inside(plot, "bottom")
    _pin_inside(plot, "left")
    _section(page, "Analysis")

    _pick(scatter._embed_combo, text="UMAP")                    # nothing typed: the two pins alone

    assert _axes_of(page) == {}
    assert widget.undo_button.toolTip() == "Undo: reset X range and Y range"
    assert _standing_line(page, explorer)[0] == (
        "X range and Y range reset: the plot now shows the UMAP embedding.")


# =================================================================================================
# 3. WINDOWS F3: the sentence stands only while what it names stays automatic
# =================================================================================================

SENTENCE = "Title and Y label reset: the plot now shows median depth."


def _vdep_after_the_rule(vdep, *, look=False, pin=False):
    """Voltage dependence with a title and a Y label typed (and a look, and a Y range, if asked),
    then the metric picked: the rule has put the words back and the sentence is up."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    if look:
        widget.title_size.setValue(14.0)
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    if pin:
        _pin(plot, "left", 0.0, 5.0)
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    assert _said(page, explorer) == (
        "Title, Y label and Y range reset: the plot now shows median depth." if pin else SENTENCE,
        True)
    return page, explorer, plot, widget


def test_voltage_dependence_the_undo_of_the_reset_takes_the_sentence_down(vdep):
    """Before: 'Title and Y label reset: the plot now shows median depth.' stayed on the bar for
    ten seconds over the title and label that Undo had just brought back."""
    page, explorer, plot, widget = _vdep_after_the_rule(vdep)

    _press_undo(page)

    assert _drawn_title(plot)[0] == "Capture vs voltage" and _label(plot, "left") == "Rate (Hz)"
    assert _said(page, explorer)[0] == ""
    _press_redo(page)                         # the words go again; the sentence is not brought back
    assert _drawn_title(plot)[0] == "" and _said(page, explorer)[0] == ""


@pytest.mark.parametrize("how", ["a title on the figure", "a label in the box"])
def test_voltage_dependence_a_title_or_label_typed_since_takes_the_sentence_down(vdep, how):
    page, explorer, plot, widget = _vdep_after_the_rule(vdep)
    if how.startswith("a title"):
        _add_title(plot, "Depth vs voltage")
    else:
        _section(page, "Appearance")
        _type_in_box(widget.y_label, "Depth")
    assert _said(page, explorer)[0] == "", how


def test_voltage_dependence_a_look_edit_reset_figure_and_their_undo_and_redo_leave_the_sentence(
        vdep):
    page, explorer, plot, widget = _vdep_after_the_rule(vdep, look=True)
    sentence = _said(page, explorer)[0]

    widget.axis_size.setValue(12.0)                            # a change of look
    assert _said(page, explorer)[0] == sentence
    QtTest.QTest.qWait(800)
    _reset_figure(page)                                        # nothing is typed after it either
    assert widget.state() == PlotAppearance() and _said(page, explorer)[0] == sentence
    _press_undo(page)                                          # Reset figure's own Undo
    assert widget.state().axis_size == 12.0 and _said(page, explorer)[0] == sentence
    _press_redo(page)
    assert widget.state() == PlotAppearance() and _said(page, explorer)[0] == sentence
    _press_undo(page)
    _press_undo(page)                                          # the axis size, still look only
    assert widget.state().title_size == 14.0 and widget.state().axis_size is None
    assert _said(page, explorer)[0] == sentence

    _press_undo(page)                                          # the rule's own step: words are back
    assert _drawn_title(plot)[0] == "Capture vs voltage"
    assert _said(page, explorer)[0] == ""


def test_voltage_dependence_a_range_pinned_again_takes_the_sentence_down(vdep):
    """The sentence named the Y range: pinning one again ends it; a range on an axis it did not
    name does not."""
    page, explorer, plot, widget = _vdep_after_the_rule(vdep, pin=True)
    sentence = _said(page, explorer)[0]

    _pin(plot, "bottom", -1100.0, -100.0)                      # the X axis: not what it named
    assert _said(page, explorer)[0] == sentence
    _pin_inside(plot, "left")                                  # the Y axis again
    assert _said(page, explorer)[0] == ""


def test_voltage_dependence_the_undo_that_brings_a_range_back_takes_the_sentence_down(vdep):
    page, explorer, plot, widget = _vdep_after_the_rule(vdep, pin=True)

    _press_undo(page)

    assert _axis(plot, "left").range == pytest.approx((0.0, 5.0))      # the pin is back ...
    assert _said(page, explorer)[0] == ""                              # ... and so is the silence


def test_voltage_dependence_a_trend_that_took_only_the_wording_keeps_its_sentence_over_a_typed_title(
        vdep):
    """The rule does not always take every word: a new trend takes the fit label's wording and
    leaves the title typed. The sentence names the wording alone, so a look edit leaves it even
    though a title is typed, and only the wording typed again ends it."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, page._fit_label, "Linear fit: {equation}")
    _section(page, "Analysis")
    _user_picks(page._fit_combo, "exponential")
    _click_button(page._fit_apply_btn)
    sentence = "Fit label wording reset: the plot now shows the exponential fit."
    assert _said(page, explorer) == (sentence, True) and widget.state().title is not None

    widget.title_size.setValue(14.0)                            # a look edit: title still typed
    QtTest.QTest.qWait(800)
    assert _said(page, explorer)[0] == sentence
    _edit_double_click(plot, plot.titleLabel, "Another title")  # a word it did not name
    assert _said(page, explorer)[0] == sentence
    _edit_double_click(plot, page._fit_label, "Exponential: {equation}")   # the wording again
    assert _said(page, explorer)[0] == ""


def test_event_waveforms_the_undo_of_the_reset_takes_the_sentence_down(events):
    page, explorer = events
    plot = page._density_main
    _type_event_words(page)
    _section(page, "Analysis")
    _pick(page._x_combo, data="frac_depth")
    assert _said(page, explorer)[0].startswith("Title, X label and Y label reset:")
    assert page._appearance_widget.undo_button.toolTip() == (       # the page has an appearance Undo
        "Undo: reset title, X label and Y label")

    _press_undo(page)

    assert _drawn_title(plot)[0] == "Dwell vs depth" and _label(plot, "bottom") == "Dwell"
    assert _said(page, explorer)[0] == ""


def test_event_waveforms_a_title_typed_since_takes_the_sentence_down(events):
    page, explorer = events
    _type_event_words(page)
    _section(page, "Analysis")
    _pick(page._x_combo, data="frac_depth")
    assert _said(page, explorer)[0] != ""

    _edit_double_click(page._density_main, page._density_main.titleLabel, "Mine")

    assert _drawn_title(page._density_main)[0] == "Mine" and _said(page, explorer)[0] == ""


def test_noise_the_undo_of_the_reset_takes_the_sentence_down(noise):
    page, explorer = noise
    _type_noise_words(page)
    _section(page, "Data")
    _pick(page._view_combo, data="spectrogram")
    assert _said(page, explorer)[0].endswith("the plot now shows the spectrogram.")
    assert page._appearance_widget.undo_button.toolTip() == (
        "Undo: reset title, X label, Y label and fit label wording")

    _press_undo(page)

    assert _drawn_title(page._spec_plot)[0] == "PSD, pore A" and page.fit_label_text()
    assert _said(page, explorer)[0] == ""


def test_noise_a_range_pinned_again_takes_the_sentence_down(noise):
    page, explorer = noise
    _pin_inside(page._plot, "left")
    _section(page, "Analysis")
    _click_button(page._asd_check)                              # PSD -> ASD: the PSD's Y range goes
    assert _said(page, explorer) == (
        "Y range reset: the plot now shows the amplitude spectral density.", True)

    _pin_inside(page._plot, "left")

    assert _said(page, explorer)[0] == ""


def test_summaries_the_undo_of_the_reset_takes_the_sentence_down(summaries):
    window, page, explorer = summaries
    _type_summary_words(page)
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)
    assert _said(page, explorer)[0].startswith("Title, X label and Y label reset:")
    assert page._appearance_widget.undo_button.toolTip() == "Undo: reset title, X label and Y label"

    _press_undo(page)

    assert _drawn_title(page._plot)[0] == "Dwell ECDF" and _said(page, explorer)[0] == ""


def test_comparison_the_undo_of_the_reset_a_typed_word_and_a_pinned_nothing_take_the_line_down(
        comparison):
    page, explorer, drawn = comparison
    widget = page._appearance_widget
    sentence = "Title, X label and Y label reset:"

    def generate_a_reset():
        _type_comparison_words(page)
        _section(page, "Analysis")
        _pick(page._y_combo, data="amplitude_pA" if page._applied_config["y"] != "amplitude_pA"
              else "snr")
        _generate(page)
        assert page._status.text().startswith(sentence), page._status.text()
        assert widget.undo_button.toolTip() == "Undo: reset title, X label and Y label"

    generate_a_reset()
    widget.title_size.setValue(16.0)                            # a change of look: the line stays
    assert page._status.text().startswith(sentence)
    QtTest.QTest.qWait(800)
    _reset_figure(page)                                         # Reset figure and its Undo: stays
    assert page._status.text().startswith(sentence)
    _section(page, "Appearance")
    _press_undo(page)
    assert page._status.text().startswith(sentence)
    _press_undo(page)                                           # the size
    assert page._status.text().startswith(sentence)
    _press_undo(page)                                           # the reset itself: words are back
    assert (drawn[-1]["title"], drawn[-1]["x"], drawn[-1]["y"]) == ("Mine", ["Dw"], ["Dp"])
    assert not page._status.text().startswith("Title") and page._status.text().startswith(
        "Generated result")

    generate_a_reset()                                          # a word typed since
    _type_in_box(widget.title, "Typed after")
    assert page._status.text().startswith("Generated result")


def test_comparison_generating_the_same_figure_again_does_not_announce_the_reset_again(
        comparison):
    """The status line leads with the sentence of the Generate that committed another figure; the
    next Generate of the same figure resets nothing, so it says nothing of the reset."""
    page, explorer, drawn = comparison
    _type_comparison_words(page)
    _section(page, "Analysis")
    _pick(page._y_combo, data="amplitude_pA" if page._applied_config["y"] != "amplitude_pA"
          else "snr")
    _generate(page)
    assert page._status.text().startswith("Title, X label and Y label reset:")

    _generate(page)                                             # the figure on screen, again

    assert page._status.text().startswith("Generated result"), page._status.text()
    assert "reset:" not in page._status.text()


def test_subpopulations_the_undo_of_the_reset_a_typed_word_and_a_look_edit(subpop):
    page, explorer = subpop
    scatter = page.embedding_view()
    plot, widget = scatter._plot.getPlotItem(), page._appearance_widget
    _type_subpop_words(page)
    _section(page, "Analysis")
    _pick(scatter._embed_combo, text="UMAP")
    sentence = _standing_line(page, explorer)[0]
    assert sentence == "Title, X label and Y label reset: the plot now shows the UMAP embedding."

    widget.title_size.setValue(18.0)                            # a look edit leaves it
    assert _standing_line(page, explorer)[0] == sentence
    QtTest.QTest.qWait(800)
    _reset_figure(page)                                         # and Reset figure, and its Undo
    assert _standing_line(page, explorer)[0] == sentence
    _press_undo(page)
    assert _standing_line(page, explorer)[0] == sentence
    _press_undo(page)
    _press_undo(page)                                           # the rule's step: words back
    assert _drawn_title(plot)[0] == "Pore A in t-SNE"
    assert _standing_line(page, explorer) == ("", True)

    _pick(scatter._embed_combo, text="t-SNE")                   # the words are typed: reset again
    assert _standing_line(page, explorer)[0] == (
        "Title, X label and Y label reset: the plot now shows the t-SNE embedding.")
    _add_title(plot, "Pore A in t-SNE again")                   # a word typed since
    assert _standing_line(page, explorer) == ("", True)


def test_distributions_the_undo_of_the_reset_a_typed_word_and_a_range_take_the_sentence_down(dist):
    page, explorer = dist
    _type_dist_words(page)
    _pin_inside(page._plot, "left")
    page._history_touch()
    _pick(page._plot_type_combo, text="Scatter")
    sentence = _said(page, explorer)[0]
    assert sentence == ("Title, X label, Y label, fit label wording and Y range reset: the plot "
                        "is now set to show the scatter of ycol against xcol.")

    page._appearance_widget.title_size.setValue(18.0)           # a look edit leaves it
    _reset_figure(page)                                         # so does Reset figure
    assert _said(page, explorer)[0] == sentence

    _press_undo(page)                                           # Reset figure's own Undo
    assert _said(page, explorer)[0] == sentence
    _press_undo(page)                                           # the size
    assert _said(page, explorer)[0] == sentence
    _press_undo(page)                                           # the plot type, with the words
    assert page._plot_type_combo.currentText() == "Histogram"
    assert _dist_words(page)[0] == "Dwell, pore A"
    assert _said(page, explorer)[0] == ""


def test_distributions_a_title_typed_since_takes_the_sentence_down(dist):
    page, explorer = dist
    _type_dist_words(page)
    _pick(page._plot_type_combo, text="Scatter")
    assert _said(page, explorer)[0] != ""

    _add_title(page._plot, "Mine")

    assert _dist_words(page)[0] == "Mine" and _said(page, explorer)[0] == ""


def test_distributions_a_range_pinned_again_takes_the_sentence_down(dist):
    page, explorer = dist
    _pin_inside(dist[0]._plot, "left")
    page._history_touch()
    _pick(page._plot_type_combo, text="Scatter")
    assert _said(page, explorer)[0].startswith("Y range reset: the plot is now set to show")
    _click_button(page._plot_btn)

    _pin_inside(page._plot, "bottom")                           # the X axis: not what it named
    assert _said(page, explorer)[0].startswith("Y range reset:")
    _pin_inside(page._plot, "left")                             # the Y axis again

    assert _said(page, explorer)[0] == ""


# =================================================================================================
# 4. CRITIC gap A: Distributions' Normalize is what the y axis counts
# =================================================================================================

WORDY = ("Dwell, pore A", "Dwell time (ms)", "Events per bin",
         "Single exponential: {equation}, {n}")


def _dist_typed_and_pinned(dist):
    """Words typed on the figure, a 16 pt bold orange title style, and both axes pinned."""
    page, explorer = dist
    plot = page._plot
    _type_dist_words(page)
    pins = {"bottom": _pin_inside(plot, "bottom"), "left": _pin_inside(plot, "left")}
    page._history_touch()
    assert page.config().axes.ranges() == ("X range", "Y range")
    return page, explorer, plot, pins


def test_distributions_a_normalize_pick_takes_the_words_and_the_y_range_in_one_undo_step(dist):
    """Before: 'Events per bin' stayed over a density axis, and a Y range pinned for counts
    framed densities (per second) with no step and no sentence."""
    page, explorer, plot, pins = _dist_typed_and_pinned(dist)
    assert _dist_words(page) == WORDY
    depth = page._history.depth()[0]
    _clear_status(page)

    _pick(page._norm_combo, text="density")

    assert page.normalization() == "density"
    assert page._history.depth()[0] == depth + 1                # the pick and the reset: ONE step
    assert _dist_words(page) == ("", "", "", "")
    assert page.config().axes.ranges() == ("X range",)         # the y range goes, the x range stays
    assert _said(page, explorer) == (
        "Title, X label, Y label, fit label wording and Y range reset: the plot is now set to "
        "show the histogram of dwell_s as density, fitted with exponential.", True)
    # What is drawn: the axis now says what it counts, the typed words are gone, the look stays.
    assert _label(plot, "left").startswith("density") and _label(plot, "bottom") != WORDY[1]
    assert _drawn_title(plot)[0] == ""
    assert page.config().style.text_style("title").size_pt == 16.0
    _is_automatic(plot, "left", was=pins["left"], auto_ranging=False)
    _stays_pinned(plot, "bottom", pins["bottom"])

    # A pick of this combo is followed by its `activated` signal, which flips `norm_user_set` after
    # the step was recorded, so the first Undo takes that flag back and shows nothing (it did before
    # this change); the next one takes back the whole configuration: the pick and the reset.
    assert page.undo_config() and page.normalization() == "density"
    assert page.undo_config()
    assert page.normalization() == "count"
    assert _dist_words(page) == WORDY and page.config().axes.ranges() == ("X range", "Y range")
    assert _drawn_title(plot)[0] == WORDY[0] and _label(plot, "left") == WORDY[2]
    _stays_pinned(plot, "left", pins["left"])
    assert page.redo_config() and page.redo_config() and _dist_words(page) == ("", "", "", "")
    assert page.normalization() == "density"


def test_distributions_every_normalize_mode_is_a_move_of_the_y_quantity(dist):
    page, explorer, plot, pins = _dist_typed_and_pinned(dist)

    _pick(page._norm_combo, text="probability per bin")

    assert _dist_words(page) == ("", "", "", "")
    assert _said(page, explorer)[0].endswith(
        "the histogram of dwell_s as probability per bin, fitted with exponential.")
    _add_title(plot, "Probability")
    _pick(page._norm_combo, text="count")                       # and back to counts
    assert _dist_words(page)[0] == "" and _said(page, explorer)[0] == (
        "Title reset: the plot is now set to show the histogram of dwell_s as counts, fitted "
        "with exponential.")


def test_distributions_the_swing_to_density_with_a_second_source_takes_nothing(dist):
    """Ticking another pore moves Normalize by itself (decision 2: the words survive it), and a
    Plot afterwards measures from the quantity then, not from before."""
    page, explorer, plot, pins = _dist_typed_and_pinned(dist)
    words, axes = _dist_words(page), page.config().axes
    steps = page._history.depth()[0]
    _clear_status(page)
    other = next(i for i in range(page._source_list.count())
                 if page._source_list.item(i).checkState() != QtCore.Qt.CheckState.Checked)

    page._source_list.item(other).setCheckState(QtCore.Qt.CheckState.Checked)
    _click_button(page._plot_btn)

    assert page.normalization() == "density" and not page._norm_user_set       # it swung by itself
    assert _dist_words(page) == words and page.config().axes == axes
    assert _said(page, explorer)[0] == ""
    assert _drawn_title(plot)[0] == WORDY[0]
    page._source_list.item(other).setCheckState(QtCore.Qt.CheckState.Unchecked)
    _click_button(page._plot_btn)
    assert page.normalization() == "count"                                    # and back by itself
    assert _dist_words(page) == words and page.config().axes == axes
    assert _said(page, explorer)[0] == "" and page._history.depth()[0] >= steps

    # A person's pick after it does reset.
    _pick(page._norm_combo, text="density")
    assert _dist_words(page) == ("", "", "", "") and page.config().axes.ranges() == ("X range",)


def test_distributions_the_public_setter_is_the_combo(dist):
    page, explorer, plot, pins = _dist_typed_and_pinned(dist)

    page.set_normalization("density")

    assert _dist_words(page) == ("", "", "", "") and page.config().axes.ranges() == ("X range",)
    assert page._norm_user_set


def test_distributions_a_normalize_set_on_a_page_that_is_not_the_histogram_takes_nothing(dist):
    """The y axis of a scatter is a column, not what Normalize counts."""
    page, explorer = dist
    page.set_plot_type("Scatter")
    _add_title(page._plot, "Calibration")
    _clear_status(page)
    steps = page._history.depth()[0]

    page.set_normalization("density")

    assert _dist_words(page)[0] == "Calibration" and _said(page, explorer)[0] == ""


def test_distributions_a_normalize_pick_with_nothing_typed_says_nothing_and_takes_down_the_old_line(
        dist):
    page, explorer = dist
    _type_dist_words(page)
    _pick(page._plot_type_combo, text="Scatter")
    assert _said(page, explorer)[0].startswith("Title, X label, Y label and fit label wording")
    _pick(page._plot_type_combo, text="Histogram")
    assert _said(page, explorer)[0] == ""
    _add_title(page._plot, "Mine")
    page._history_touch()
    _pick(page._norm_combo, text="density")
    assert _said(page, explorer)[0].startswith("Title reset:")
    _pick(page._norm_combo, text="count")                       # nothing typed any more
    assert _said(page, explorer)[0] == ""


def test_distributions_words_typed_over_density_are_saved_and_the_next_normalize_resets_them(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save, reopened by ``_open_project``: the project opens with the density axis and
    the words typed over it, and only the NEXT Normalize pick takes them back."""
    window = _series_window(qtbot, tmp_path, monkeypatch)
    window._open_plots()
    page = window._plots_window
    explorer = _explorer_of(window)
    page.activateWindow()
    assert page.sources()
    _click_button(page._plot_btn)
    _add_title(page._plot, "Counts")
    page._history_touch()
    _pick(page._norm_combo, text="density")                     # the words go
    assert _dist_words(page)[0] == ""
    _add_title(page._plot, "Density, three pores")
    _edit_double_click(page._plot, page._plot.getAxis("left").label, "Per second")
    page._history_touch()
    project = tmp_path / "normalize.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_plots()
    page = reopened._plots_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    assert page.normalization() == "density"
    assert _dist_words(page)[:3] == ("Density, three pores", "", "Per second")
    assert _drawn_title(page._plot)[0] == "Density, three pores"
    assert _said(page, explorer)[0] == ""
    _pick(page._norm_combo, text="probability per bin")
    assert _dist_words(page) == ("", "", "", "")
    assert _said(page, explorer)[0].startswith(
        "Title and Y label reset: the plot is now set to show")
    assert reopened._dirty
    reopened._dirty = False


# =================================================================================================
# 5. CRITIC gap B: Event waveforms' Align is what t = 0 means on the ensemble's x axis
# =================================================================================================

from .test_p0_pins_clear_with_the_words import _pin_the_event_figures  # noqa: E402

ENSEMBLE_WORDS = ("Dwell vs depth", "Dwell", "Depth")


def _real_time_ensemble(events):
    """The Event waveforms page on its Ensemble tab, in real time (so Align is enabled), with the
    map's and the ensemble's ranges pinned, words typed, and then the ensemble's x range pinned
    again over the seconds axis."""
    page, explorer = events
    pins = _pin_the_event_figures(page)
    ens, widget = page._ensemble_plot, page._appearance_widget
    _tab(page, 1)
    _section(page, "Analysis")
    _click_button(page._normalize)                              # normalised time -> seconds
    assert not page._normalize.isChecked() and page._align_combo.isEnabled()
    assert page._align_combo.currentText() == "start"
    _type_event_words(page, plot=ens)
    pins["ensemble.bottom"] = _pin_inside(ens, "bottom")
    _section(page, "Analysis")
    _clear_status(page)
    return page, explorer, ens, widget, pins


@pytest.mark.parametrize("align, phrase", [("extremum", "deepest sample"), ("center", "center")])
def test_event_waveforms_a_new_alignment_takes_the_words_and_the_ensembles_x_range(
        events, align, phrase):
    """Before: a typed 'Time from event start (s)' stayed over an axis whose t = 0 had moved to the
    deepest sample, and a time range pinned for 'start' framed another part of the events."""
    page, explorer, ens, widget, pins = _real_time_ensemble(events)
    mp = page._density_main
    assert _drawn_title(ens)[0] == ENSEMBLE_WORDS[0] and _label(ens, "bottom") == ENSEMBLE_WORDS[1]
    before, steps = _axes_of(page), _depth(widget)

    _pick(page._align_combo, text=align)

    assert _depth(widget) == (steps[0] + 1, 0)                  # ONE step
    assert _boxes(widget) == ("", "", "") and widget.state().title is None
    assert _label(ens, "bottom") == "Time (s)" and _drawn_title(ens)[0] != ENSEMBLE_WORDS[0]
    assert set(_axes_of(page)) == set(before) - {"ensemble.bottom"}
    _is_automatic(ens, "bottom", was=pins["ensemble.bottom"])
    _stays_pinned(ens, "left", pins["ensemble.left"])           # current stays current
    assert _said(page, explorer) == (
        "Title, X label, Y label and X range reset: the plot now shows the ensemble with t = 0 at "
        f"each event's {phrase}.", True)
    _tab(page, 0)                                               # the map's pins are another view's
    _stays_pinned(mp, "bottom", pins["map.bottom"])
    _stays_pinned(mp, "left", pins["map.left"])
    _tab(page, 1)
    _press_undo(page)                                           # one Undo brings all of it back
    assert _depth(widget) == (steps[0], 1) and _axes_of(page) == before
    assert (_drawn_title(ens)[0], _label(ens, "bottom"), _label(ens, "left")) == (
        ENSEMBLE_WORDS[0], ENSEMBLE_WORDS[1], ENSEMBLE_WORDS[2])
    _stays_pinned(ens, "bottom", pins["ensemble.bottom"])
    _press_redo(page)
    _is_automatic(ens, "bottom", was=pins["ensemble.bottom"])


def test_event_waveforms_each_alignment_is_measured_from_the_one_on_screen(events):
    page, explorer, ens, widget, pins = _real_time_ensemble(events)
    _pick(page._align_combo, text="extremum")
    assert widget.state().title is None
    _edit_double_click(ens, ens.titleLabel, "Mean event")
    _pick(page._align_combo, text="start")                      # typed again, so it resets again
    assert widget.state().title is None and _said(page, explorer)[0] == (
        "Title reset: the plot now shows the ensemble with t = 0 at each event's start.")
    _pick(page._align_combo, text="center")                     # nothing typed any more
    assert _said(page, explorer)[0] == ""


def test_event_waveforms_nothing_typed_and_no_range_on_the_axis_records_no_step(events):
    page, explorer = events
    ens, widget = page._ensemble_plot, page._appearance_widget
    _tab(page, 1)
    _section(page, "Analysis")
    _click_button(page._normalize)
    _clear_status(page)
    steps = _depth(widget)

    _pick(page._align_combo, text="extremum")

    assert _depth(widget) == steps and _said(page, explorer)[0] == ""
    assert _label(ens, "bottom") == "Time (s)"


def test_event_waveforms_while_time_is_normalised_the_alignment_is_immaterial(events):
    """The combo is greyed out there and a setter can still move it: no alignment is drawn, so
    nothing t = 0 means has changed."""
    page, explorer = events
    ens, widget = page._ensemble_plot, page._appearance_widget
    _tab(page, 1)
    _type_event_words(page, plot=ens)
    ens_x = _pin_inside(ens, "bottom")
    words, steps = widget.state(), _depth(widget)
    assert page._normalize.isChecked() and not page._align_combo.isEnabled()
    _clear_status(page)

    page.set_align("extremum")

    assert page._align_combo.currentText() == "extremum"
    assert widget.state() == words and _depth(widget) == steps and _said(page, explorer)[0] == ""
    _stays_pinned(ens, "bottom", ens_x)
    # ... and the first real-time draw measures from the Normalize time that made it real.
    _section(page, "Analysis")
    _click_button(page._normalize)
    assert widget.state().title is None and _said(page, explorer)[0] == (
        "Title, X label, Y label and X range reset: the plot now shows the ensemble against "
        "time in seconds.")


def test_event_waveforms_a_template_and_a_restore_move_the_alignment_without_taking_anything(
        events):
    from nanoscope.core import analysis_workspace as workspace

    page, explorer, ens, widget, pins = _real_time_ensemble(events)
    words, before, steps = widget.state(), _axes_of(page), _depth(widget)
    settings, appearance = page.template_settings()
    assert settings["align"] == "start" and settings["normalize_time"] is False

    page.apply_template_settings({**settings, "align": "extremum"}, page.appearance_state())

    assert page._align_combo.currentText() == "extremum"        # the template's own settings win
    assert widget.state() == words and _axes_of(page) == before and _depth(widget) == steps
    assert _said(page, explorer)[0] == ""
    artifact = workspace.capture_event_plots(page)
    page.apply_template_settings({**settings, "align": "center"}, page.appearance_state())
    assert workspace.restore_event_plots(page, artifact).restored
    assert page._align_combo.currentText() == "extremum"
    assert widget.state() == words and _axes_of(page) == before and _said(page, explorer)[0] == ""
    # What the restore drew is where the next alignment starts: "extremum" -> "start" is a change.
    _section(page, "Analysis")
    _pick(page._align_combo, text="start")
    assert widget.state().title is None and "ensemble.bottom" not in _axes_of(page)
    assert _said(page, explorer)[0].endswith("with t = 0 at each event's start.")


def test_event_waveforms_an_alignment_that_could_not_be_drawn_shows_no_quantity_to_leave(
        events, monkeypatch):
    """A message on the ensemble (here: no event can be aligned on its deepest sample) shows no
    alignment, so the first thing drawn after it has nothing stale to reset."""
    page, explorer, ens, widget, pins = _real_time_ensemble(events)
    real = page._ensemble_events
    monkeypatch.setattr(page, "_ensemble_events",
                        lambda: None if page._align_combo.currentText() == "extremum" else real())
    steps = _depth(widget)

    _pick(page._align_combo, text="extremum")                   # nothing to draw: a message

    assert page.ensemble_traces() is None and widget.state().title == ENSEMBLE_WORDS[0]
    assert _depth(widget) == steps and _said(page, explorer)[0] == ""
    _pick(page._align_combo, text="center")                     # the first draw since: not stale
    assert page.ensemble_traces() is not None
    assert widget.state().title == ENSEMBLE_WORDS[0] and _depth(widget) == steps
    assert _said(page, explorer)[0] == ""
    _pick(page._align_combo, text="start")                      # one is on screen now: it resets
    assert widget.state().title is None and _depth(widget) == (steps[0] + 1, 0)


def test_event_waveforms_an_empty_window_has_no_alignment_to_leave(events):
    page, explorer, ens, widget, pins = _real_time_ensemble(events)
    page.set_source(None, None, None)
    assert page.ensemble_traces() is None and widget.state().title == ENSEMBLE_WORDS[0]
    steps = _depth(widget)
    _clear_status(page)

    _pick(page._align_combo, text="extremum")

    assert _depth(widget) == steps and _said(page, explorer)[0] == ""
    assert widget.state().title == ENSEMBLE_WORDS[0]


@pytest.mark.parametrize("control", ["align", "normalize"])
def test_event_waveforms_a_restore_that_draws_no_ensemble_leaves_no_quantity_to_reset(
        events, control):
    """A saved ensemble that is missing draws a message ('No saved event ensemble is available'),
    which shows no alignment and no time axis: what is drawn next has nothing stale to reset, as
    after any other message. Before the snapshot let go of what it had drawn, the next Align (or
    Normalize time) was measured against the ensemble that the restore had just replaced."""
    from dataclasses import replace as replaced

    from nanoscope.core import analysis_workspace as workspace

    page, explorer, ens, widget, pins = _real_time_ensemble(events)
    artifact = workspace.capture_event_plots(page)
    assert workspace.restore_event_plots(
        page, replaced(artifact, payload={**artifact.payload, "ensemble": None})).restored
    assert page.ensemble_traces() is None and not ens.listDataItems()   # a message, no ensemble
    words, steps = widget.state(), _depth(widget)
    assert words.title == ENSEMBLE_WORDS[0]
    _clear_status(page)
    _section(page, "Analysis")

    if control == "align":
        _pick(page._align_combo, text="extremum")
    else:
        _click_button(page._normalize)                          # time normalised

    assert page.ensemble_traces() is not None                   # the first ensemble since
    assert widget.state() == words and _depth(widget) == steps  # nothing was on screen to leave
    assert _drawn_title(ens)[0] == ENSEMBLE_WORDS[0] and _said(page, explorer)[0] == ""


def test_event_waveforms_words_typed_over_an_alignment_are_saved_and_the_next_one_resets_them(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save, reopened by ``_open_project``: the ensemble opens in the alignment it was
    saved in, with its words, and the next alignment resets them."""
    window = _series_window(qtbot, tmp_path, monkeypatch)
    first = next(iter(window._session.documents())).doc_id
    window._on_selection(first, 0)
    window._open_event_plots()
    page = window._event_plot_window
    explorer = _explorer_of(window)
    page.activateWindow()
    assert page.density_counts() is not None
    _tab(page, 1)
    _section(page, "Analysis")
    _click_button(page._normalize)
    _pick(page._align_combo, text="center")
    _type_event_words(page, plot=page._ensemble_plot)
    pin = _pin_inside(page._ensemble_plot, "bottom")
    project = tmp_path / "align.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_event_plots()
    page = reopened._event_plot_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    widget, ens = page._appearance_widget, page._ensemble_plot
    assert page._align_combo.currentText() == "center" and not page._normalize.isChecked()
    _tab(page, 1)
    assert (widget.state().title, widget.state().x_label) == ENSEMBLE_WORDS[:2]
    assert _drawn_title(ens)[0] == ENSEMBLE_WORDS[0]
    _stays_pinned(ens, "bottom", pin)
    assert _said(page, explorer)[0] == ""
    _section(page, "Analysis")
    _pick(page._align_combo, text="extremum")
    assert widget.state().title is None and "ensemble.bottom" not in _axes_of(page)
    assert _said(page, explorer)[0] == (
        "Title, X label, Y label and X range reset: the plot now shows the ensemble with t = 0 "
        "at each event's deepest sample.")
    assert reopened._dirty
    reopened._dirty = False


# =================================================================================================
# 6. G5 residual (c): Log scale changed in Axis properties over a range that is already pinned
# =================================================================================================

def _axis_dialog(plot, side, *, log=None, untick_auto=False, tick=None):
    """Right-click the axis label, **Axis properties…**, optionally untick Auto (the boxes then
    hold the range the view shows) and tick or untick Log scale, optionally type a major tick
    spacing, and press OK WITHOUT typing a range: the range handed back is the one the dialog
    opened with. (``_pin`` types one.)"""
    from nanoscope.gui.figure_text import TextTarget, figure_text_for

    from .test_plot_window_routes import _choose, _click, _names

    controller = figure_text_for(plot)
    view = plot.scene().views()[0]
    window = view.window()
    home = window.pos()
    rect = controller._anchor_rect(TextTarget(item=plot.getAxis(side)))
    below = view.mapToGlobal(rect.bottomLeft())
    from PySide6 import QtGui
    screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    window.move(home.x() - max(0, below.x() + 520 - screen.right()),
                home.y() - max(0, below.y() + 280 - screen.bottom()))
    QtWidgets.QApplication.processEvents()
    try:
        controller.last_menu = None
        _click(plot, plot.getAxis(side).label.sceneBoundingRect().center())
        menu = controller.last_menu
        assert menu is not None and "Axis properties…" in _names(menu)
        _choose(menu, "Axis properties…")
        popover = controller.popover
        assert popover is not None and popover.side == side
        if untick_auto and popover.auto.isChecked():
            QtTest.QTest.mouseClick(popover.auto, L)
        if log is not None and popover.log.isChecked() != log:
            indicator = QtCore.QPoint(8, popover.log.height() // 2)
            QtTest.QTest.mouseClick(popover.log, L, NOMOD, indicator)
            assert popover.log.isChecked() == log
        if tick is not None:
            popover.major.selectAll()
            QtTest.QTest.keyClicks(popover.major, f"{tick:g}")
        ok = popover.findChild(QtWidgets.QDialogButtonBox).button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok)
        QtTest.QTest.mouseClick(ok, L)
        QtWidgets.QApplication.processEvents()
        assert not popover.isVisible(), "OK was refused: " + popover.error.text()
    finally:
        window.move(home)
        QtWidgets.QApplication.processEvents()


def _map_has_data_in_view(plot, side):
    """Whether the drawn map (the image) overlaps what the axis shows: a blank map does not."""
    index = 0 if side == "bottom" else 1
    lo, hi = plot.getViewBox().viewRange()[index]
    a, b = plot.getViewBox().childrenBounds()[index]
    return a < hi and b > lo


def test_across_log_carries_a_range_to_the_units_the_axis_is_about_to_have():
    import math

    from nanoscope.gui.event_plot_panel import _across_log

    assert _across_log((0.001, 0.1), True) == pytest.approx((-3.0, -1.0))
    assert _across_log((-3.0, -1.0), False) == pytest.approx((0.001, 0.1))
    lo, hi = _across_log(_across_log((20.0, 65.0), True), False)
    assert (lo, hi) == pytest.approx((20.0, 65.0), rel=1e-12)
    # A minimum at or below zero cannot be logged; a power that overflows has no image either.
    assert _across_log((0.0, 5.0), True) is None and _across_log((-1.0, 5.0), True) is None
    assert _across_log((-400.0, 400.0), False) is None and _across_log((1.0, 800.0), False) is None
    assert math.isfinite(_across_log((1e-300, 1e300), True)[0])
    # Two ends that land on the same number (here 300.0, or both underflowing to 0.0) are not a
    # range either.
    assert _across_log((1e300, 1e300 * (1 + 1e-15)), True) is None
    assert _across_log((-400.0, -399.0), False) is None


def test_event_waveforms_log_scale_over_a_pinned_range_carries_the_range_across_and_back(events):
    """Before: the range stayed in the old units over the new numbers (a Y range in pA over
    log10 of pA), so the Event-count map went blank with no step and no sentence."""
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    _section(page, "Analysis")
    assert not page._logy.isChecked()
    lo, hi = _pin_inside(mp, "left", major=20.0)
    assert _map_has_data_in_view(mp, "left")
    steps, _ = _depth(widget)
    _clear_status(page)

    _axis_dialog(mp, "left", log=True)                          # Log scale ticked, range untouched

    assert page._logy.isChecked() and page._density_logs == (True, True)
    pin = _axes_of(page)["map.left"]
    assert (pin["min"], pin["max"]) == pytest.approx((math.log10(lo), math.log10(hi)), rel=1e-5)
    assert "tick_major" not in pin                              # a spacing does not carry across
    _stays_pinned(mp, "left", (pin["min"], pin["max"]))
    assert _map_has_data_in_view(mp, "left")                    # the map is not blank
    assert _depth(widget) == (steps + 1, 0)                     # one step, the dialog's
    assert _said(page, explorer)[0] == ""                       # nothing was lost to announce
    _press_undo(page)                                           # (Undo does not move the box)
    assert _depth(widget) == (steps, 1)

    _press_redo(page)
    _axis_dialog(mp, "left", log=False)                         # and back
    assert not page._logy.isChecked()
    pin = _axes_of(page)["map.left"]
    assert (pin["min"], pin["max"]) == pytest.approx((lo, hi), rel=1e-4)
    assert _map_has_data_in_view(mp, "left")


import math  # noqa: E402


def test_event_waveforms_log_scale_on_the_x_axis_carries_the_range_typed_in_seconds(events):
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    _pin(mp, "bottom", 0.0012, 0.003, log=False)               # the route that opens the X axis
    assert not page._logx.isChecked()
    _stays_pinned(mp, "bottom", (0.0012, 0.003))
    _clear_status(page)

    _axis_dialog(mp, "bottom", log=True)

    assert page._logx.isChecked()
    pin = _axes_of(page)["map.bottom"]
    assert (pin["min"], pin["max"]) == pytest.approx((math.log10(0.0012), math.log10(0.003)))
    assert _map_has_data_in_view(mp, "bottom") and _said(page, explorer)[0] == ""


def test_event_waveforms_log_scale_beside_a_range_typed_in_the_dialog_keeps_it_as_typed(events):
    page, explorer = events
    mp = page._density_main
    _section(page, "Analysis")
    _pin_inside(mp, "left")
    _pin(mp, "left", 30.0, 60.0, log=True)                      # typed beside the change
    assert page._logy.isChecked()
    pin = _axes_of(page)["map.left"]
    assert (pin["min"], pin["max"]) == (30.0, 60.0)


def test_event_waveforms_log_scale_beside_a_major_tick_typed_in_the_dialog_keeps_the_tick(events):
    """The spacing the dialog opened with is in the units the axis is leaving and goes with the
    range it was pinned beside; one typed in the same dialog is for the axis as it is about to be
    drawn. Before, both went, and the spacing the person typed was lost without a word."""
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    _section(page, "Analysis")
    lo, hi = _pin_inside(mp, "left", major=20.0)
    steps, _ = _depth(widget)

    _axis_dialog(mp, "left", log=True, tick=0.1)                # range as it opened, spacing typed

    pin = _axes_of(page)["map.left"]
    assert (pin["min"], pin["max"]) == pytest.approx((math.log10(lo), math.log10(hi)), rel=1e-5)
    assert pin["tick_major"] == 0.1                             # not the 20 it opened, not lost
    assert _axis(mp, "left").tick_major == 0.1 and mp.getAxis("left").major_tick == 0.1
    assert _map_has_data_in_view(mp, "left") and _depth(widget) == (steps + 1, 0)

    _axis_dialog(mp, "left", log=False, tick=5.0)               # and back, with a spacing in pA
    pin = _axes_of(page)["map.left"]
    assert (pin["min"], pin["max"]) == pytest.approx((lo, hi), rel=1e-4)
    assert pin["tick_major"] == 5.0 and mp.getAxis("left").major_tick == 5.0
    assert _map_has_data_in_view(mp, "left")


def test_event_waveforms_log_scale_over_a_tick_with_more_digits_than_the_dialog_shows_drops_it(
        events):
    """The dialog shows a major tick to six decimals, so a spacing set another way (a template, an
    older project) that has more is still the one the dialog opened with when it is handed back."""
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    _section(page, "Analysis")
    lo, hi = _pin_inside(mp, "left", major=20.0)
    widget.apply_appearance(widget.state().with_axis_pin(
        "map.left", AxisPin(min=lo, max=hi, tick_major=20.1234567)))
    assert _axis(mp, "left").tick_major == 20.1234567

    _axis_dialog(mp, "left", log=True)

    pin = _axes_of(page)["map.left"]
    assert (pin["min"], pin["max"]) == pytest.approx((math.log10(lo), math.log10(hi)), rel=1e-5)
    assert "tick_major" not in pin and getattr(mp.getAxis("left"), "major_tick", None) is None


def test_event_waveforms_log_scale_with_auto_unticked_beside_a_typed_tick_keeps_the_tick(events):
    page, explorer = events
    mp = page._density_main
    _section(page, "Analysis")
    shown = _axis(mp, "left").shown_range
    assert "map.left" not in _axes_of(page)

    _axis_dialog(mp, "left", log=True, untick_auto=True, tick=0.1)

    pin = _axes_of(page)["map.left"]
    assert (pin["min"], pin["max"]) == pytest.approx(
        (math.log10(float(f"{shown[0]:.6g}")), math.log10(float(f"{shown[1]:.6g}"))))
    assert pin["tick_major"] == 0.1 and mp.getAxis("left").major_tick == 0.1
    assert _map_has_data_in_view(mp, "left")


def test_event_waveforms_log_scale_with_auto_unticked_carries_the_range_the_dialog_showed(events):
    """Unticking Auto fills the boxes with the range the view shows; leaving them as they are is
    asking for that frame, in the units the axis is about to have."""
    page, explorer = events
    mp = page._density_main
    _section(page, "Analysis")
    shown = _axis(mp, "left").shown_range
    assert shown[0] > 0 and "map.left" not in _axes_of(page)

    _axis_dialog(mp, "left", log=True, untick_auto=True)

    pin = _axes_of(page)["map.left"]
    assert (pin["min"], pin["max"]) == pytest.approx(
        (math.log10(float(f"{shown[0]:.6g}")), math.log10(float(f"{shown[1]:.6g}"))))
    assert _map_has_data_in_view(mp, "left")


def test_event_waveforms_axis_properties_with_no_log_change_leaves_the_range_alone(events):
    """Only a change of Log scale carries a range across: pressing OK on a dialog that changed
    nothing, or only the tick labels, leaves the pin as it was."""
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    _section(page, "Analysis")
    lo, hi = _pin_inside(mp, "left", major=20.0)
    before, steps = _axes_of(page), _depth(widget)

    _axis_dialog(mp, "left")                                    # OK, nothing changed

    assert _axes_of(page) == before and _depth(widget) == steps
    _stays_pinned(mp, "left", (lo, hi))
    assert not page._logy.isChecked()


def test_event_waveforms_log_scale_over_auto_leaves_the_axis_on_auto(events):
    page, explorer = events
    mp, widget = page._density_main, page._appearance_widget
    steps = _depth(widget)

    _axis_dialog(mp, "left", log=True)

    assert page._logy.isChecked() and _axes_of(page) == {}
    _is_automatic(mp, "left", auto_ranging=False)               # the page frames its own data
    assert _depth(widget) == steps


# =================================================================================================
# 7. G5 residual (a): the G1 helper that picks from a popup does not spin on an item it cannot reach
# =================================================================================================

def test_pick_fails_on_an_item_the_popup_cannot_reach_instead_of_spinning(events, monkeypatch):
    """The Event waveforms metric lists 'Selected amplitude' disabled when the table has none.
    The arrow keys skip a disabled row, so the walk used to repeat for ever."""
    page, explorer = events
    combo = page._x_combo
    _section(page, "Analysis")
    index = combo.findData("amplitude_pA")
    assert index >= 0 and not combo.model().item(index).isEnabled()
    before, presses = combo.currentData(), []
    real = QtTest.QTest.keyClick

    def counting(*args, **kwargs):
        presses.append(args[1])
        if len(presses) > 100:
            raise RuntimeError("the popup walk is not bounded")
        return real(*args, **kwargs)

    monkeypatch.setattr(QtTest.QTest, "keyClick", counting)

    with pytest.raises(AssertionError, match="cannot reach item"):
        _pick(combo, data="amplitude_pA")

    assert len(presses) <= combo.count() + 1                    # a key per row, then it gives up
    assert combo.currentData() == before and not combo.view().isVisible()
    # An item it can reach is still picked.
    monkeypatch.setattr(QtTest.QTest, "keyClick", real)
    _pick(combo, data="frac_depth")
    assert combo.currentData() == "frac_depth"


# =================================================================================================
# 8. the guide and the inventory say what the code does
# =================================================================================================

GUIDE_SENTENCES = (
    # G5 residual (d): the tick spacing is named apart (the documentation pass wrote the same fact
    # first, in these words, and test_p0_pins_clear_with_the_words pins them too).
    "The sentence names a major tick spacing apart from the range (*Y range and Y major tick "
    "reset: …*, or *Y major tick reset: …* when only a spacing was typed).",
    # WINDOWS F3: the sentence stands only while it is true.
    "The sentence also stands only while it is true. It says that what it names is automatic, so "
    "it comes off the status line (or, in Subpopulations and Comparison figures, off the line it "
    "stands on) as soon as one of those words or ranges is typed or pinned again: the **Undo** of "
    "the reset, which brings them back, a title or label typed since, a range pinned again.",
    "A change of look, **Reset figure** and its **Undo** and **Redo** leave it, since nothing it "
    "names is typed or pinned then; **Redo** takes the words and ranges away again without "
    "bringing the sentence back.",
    "The sentence also comes off when a word or range it names is typed or set again, as by the "
    "**Undo** that brings them back.",
    "or until a word or range it names is typed or pinned again (the **Undo** of the reset); "
    "Reset figure and a change of look leave it.",
    "or until a title or label is typed again, as by the **Undo** of the reset",
    # WINDOWS F4 and F1.
    "On the pages with an appearance Undo the Undo and Redo tooltips name the step by what it "
    "took back (*Undo: reset title and Y label*, *Undo: reset title, Y label and Y range*), "
    "where a real **Reset figure** reads *Undo: reset figure* because it also takes the look "
    "back; a figure whose only customisation is typed words and ranges has nothing else for "
    "Reset figure to take, so there the two read alike. Distributions' **Configuration** Undo "
    "names no step of any kind: its tooltip only counts them (*Undo the last change to this "
    "plot's configuration (3 to step back).*).",
    "**Reset figure** pressed straight after a change of quantity, on a figure with nothing else "
    "customised, finds nothing to take back and records no step.",
    # CRITIC gap A: Distributions' Normalize.
    "The histogram's **Normalize** (count, density, probability per bin) is what its Y axis "
    "counts, so picking another takes the words and the Y range back, in the step of the pick, "
    "and the sentence says what the histogram is now set to show (*…the histogram of dwell_s as "
    "density, fitted with exponential.*). The switch to density that ticking a second source "
    "makes by itself takes nothing.",
    "Picking another **Normalize** (count, density, probability per bin) is a change of what the "
    "Y axis counts: it does the same, with the Y range, in the step of the pick (the density "
    "that ticking a second source switches to by itself keeps the words and the range).",
    "of plot type, column, Normalize or applied fit that finds nothing typed and no range set "
    "takes that sentence off the status line",
    # CRITIC gap B: Event waveforms' Align.
    "Ensemble tabs, **Normalize time**, and **Align**.",
    "**Align** (start, center, extremum) is what t = 0 means on the ensemble's X axis once time is "
    "in seconds, so another alignment takes the words and that X range too (*…the plot now shows "
    "the ensemble with t = 0 at each event's deepest sample.*); while time is normalised the "
    "alignment is immaterial and the combo is greyed out, and setting it then takes nothing.",
    "toggling **Normalize time**, or changing **Align** (once time is in seconds, when it is what "
    "t = 0 means on the ensemble's X axis) puts the typed Title, X label and Y label",
    "**Align** takes the ensemble's X range too.",
    "Bins, the colormap, the band, the subset and the number of points keep both the words and "
    "the ranges.",
    # G5 residual (c): Log scale in Axis properties.
    "A range the dialog hands back as it opened (the pinned one, or the one the view shows when "
    "you untick Auto and leave the boxes alone) is carried across to the units the axis is about "
    "to have, log10 of it when Log scale is ticked and ten to the power when it is unticked, so "
    "the map keeps its frame instead of going blank; the major tick spacing the dialog opened "
    "with, which does not carry across, goes (one you type in the same dialog is kept, as a "
    "range is), and so does a range that cannot be (a minimum at or below zero cannot be "
    "logged).",
    "A range left as the dialog opened with it is carried across to the new units instead (log10 "
    "of it, or ten to the power).",
)


def _flat_guide():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    return root, " ".join((root / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8").split())


def test_the_guide_says_what_g6b_changed():
    root, flat = _flat_guide()
    for sentence in GUIDE_SENTENCES:
        assert sentence in flat, sentence
    # The sentence the alignment used to be in is gone.
    assert "the subset and the alignment keep both the words and the ranges" not in flat
    # The tooltips are named on the pages that have an appearance Undo, not on Distributions.
    assert "The Undo and Redo tooltips name the step by what it took back" not in flat


INVENTORY_PIECES = (
    # (page, list, index, key or None, piece)
    ("Normalize", "Picking another value yourself (not the automatic switch) is a change of what "
                  "the Y axis counts"),
    ("Align", "Changing it while time is in seconds takes the typed Title, X label and Y label "
              "and the range pinned on the ensemble's X axis back to automatic"),
    ("Log x", "A range left as the dialog opened with it is carried across to the new units"),
    ("Log y", "In Axis properties a range left as the dialog opened with it is carried across to "
              "the new units"),
)
GOTCHA_PIECES = (
    "the Normalize you pick (the same words and the Y range, not the automatic switch to "
    "density that a second source brings)",
    "The sentence comes off the status line when a word or range it names is typed or pinned "
    "again (the Undo of the reset does that), and on the pages with an appearance Undo the Undo "
    "and Redo tooltips name the step ('reset title and Y label'; Distributions' Configuration "
    "Undo only counts its steps)",
    "Normalize time and Align once time is in seconds (the ensemble's X range)",
    "(and carries a range left as the dialog opened with it across to the new units)",
    "the Undo and Redo tooltips name the step ('reset title, Y label and Y range')",
)


def test_the_inventory_says_what_g6b_changed():
    import json

    root, _flat = _flat_guide()
    inventory = json.loads((root / "docs" / "inventory.json").read_text(encoding="utf-8"))
    notes = {c["label"]: c.get("notes", "") for area in inventory for c in area["controls"]
             if c.get("group") in ("Plot ▸ Histogram", "Event waveforms ▸ Ensemble",
                                   "Event waveforms ▸ Event-count heatmap")}
    for label, piece in INVENTORY_PIECES:
        assert piece in notes[label], (label, piece)
    said = [g for area in inventory for g in area.get("gotchas", ())
            if g.startswith("Changing what a figure shows puts the words you typed back")]
    assert len(said) == 2
    for piece in GOTCHA_PIECES:
        assert any(piece in g for g in said), piece
    assert not any("and the Undo and Redo tooltips name the step ('reset title and Y label')" in g
                   for g in said)
