"""Phase 0, group G1: the reset rule. When a figure's quantity changes, the TYPED title, x
label, y label and fit-label wording go back to automatic; the look stays; one Undo step; a
status line that names only what was cleared (``gui.quantity_reset``).

Every gesture is a real one. Words are typed on the figure (double-click the label, right-click
the plot for Add title…, the Text style… popover) or in the Appearance boxes with the keyboard,
the quantity is changed with the real control (the combo's popup, the tab bar, the Generate
button), a save is a real ``_write_project`` reopened by ``_open_project``, and the status line is
read where the user sees it: the page's status bar inside a real Analysis Explorer, with its
geometry checked against the Explorer's. Synthetic data only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.analysis_explorer import (
    PAGE_COMPARISON_FIGURES, PAGE_DISTRIBUTIONS, PAGE_EVENT_WAVEFORMS, PAGE_NOISE,
    PAGE_SUBPOPULATIONS, PAGE_VOLTAGE_SERIES, AnalysisExplorer)
from nanoscope.gui.figure_text import TextStyle, figure_text_for
from nanoscope.gui.plot_appearance import PlotAppearance

from .test_plot_window_routes import (
    _add_title, _choose, _click, _edit_double_click, _install_series, _main_window,
    _pin_through_axis_properties, _press_undo, _restored_main_window, _shown)

L = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
ORANGE = "#d55e00"


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


# -- real gestures --------------------------------------------------------------------------------

def _pick(combo, *, text=None, data=None):
    """Click the combo, walk the popup with the arrow keys and press Return: a person's pick."""
    assert combo.isVisible(), "the control must be on screen to be picked from"
    combo.setFocus()
    QtTest.QTest.mouseClick(combo, L)
    QtWidgets.QApplication.processEvents()
    if data is not None:
        target = next(i for i in range(combo.count()) if combo.itemData(i) == data)
    else:
        target = next(i for i in range(combo.count()) if combo.itemText(i) == text)
    popup = combo.view()
    while popup.currentIndex().row() != target:
        down = popup.currentIndex().row() < target
        QtTest.QTest.keyClick(popup, QtCore.Qt.Key.Key_Down if down else QtCore.Qt.Key.Key_Up)
    QtTest.QTest.keyClick(popup, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert combo.currentIndex() == target


def _user_picks(combo, text):
    """Choose ``text`` the way the Plots window tells a person from code: with the dropdown
    open (its run gate arms Apply for a person and commits at once for code)."""
    combo.showPopup()
    QtWidgets.QApplication.processEvents()
    combo.setCurrentText(text)
    QtWidgets.QApplication.processEvents()
    combo.hidePopup()
    QtWidgets.QApplication.processEvents()


def _click_button(button):
    assert button.isVisible() and button.isEnabled(), button.text()
    QtTest.QTest.mouseClick(button, L)
    QtWidgets.QApplication.processEvents()


def _section(page, name):
    page._workflow.set_section(name)
    QtWidgets.QApplication.processEvents()


def _type_in_box(box, text):
    """Type into an Appearance box with the keyboard (replacing what it holds)."""
    box.setFocus()
    box.selectAll()
    QtTest.QTest.keyClicks(box, text)
    QtTest.QTest.keyClick(box, QtCore.Qt.Key.Key_Tab)      # focus leaves: the run is closed
    QtWidgets.QApplication.processEvents()


def _style_title(plot, *, size=16.0, color=ORANGE):
    """Right-click the title, Text style…, size, B, a colour, OK: a 16 pt bold orange title."""
    controller = figure_text_for(plot)
    controller.last_menu = None
    _click(plot, plot.titleLabel.sceneBoundingRect().center())
    _choose(controller.last_menu, "Text style…")
    popover = controller.popover
    assert popover is not None and popover.key == "title"
    popover.size.setValue(size)
    QtTest.QTest.mouseClick(popover.bold, L)
    popover.color.hex.clear()
    QtTest.QTest.keyClicks(popover.color.hex, color)
    ok = popover.findChild(QtWidgets.QDialogButtonBox).button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok)
    QtTest.QTest.mouseClick(ok, L)
    QtWidgets.QApplication.processEvents()


def _drawn_title(plot):
    """What the title DRAWS: (text, point size, bold, colour), read off the text item."""
    label = plot.titleLabel
    if not label.isVisible() or not label.item.toPlainText().strip():
        return ("", None, None, None)
    fmt = label.item.document().begin().begin().fragment().charFormat()
    # Bold is in the span's CSS for the pyqtgraph windows, and on the item's QFont in Distributions.
    bold = fmt.fontWeight() >= 600 or label.item.font().bold()
    return (label.item.toPlainText().strip(), fmt.fontPointSize(), bold,
            fmt.foreground().color().name())


def _squash(text):
    """A title as Qt's rich text draws it: runs of white space collapsed."""
    return " ".join(text.split())


def _label(plot, side):
    """The axis label as drawn (plain text; empty when the label draws nothing)."""
    axis = plot.getAxis(side)
    if not (axis.isVisible() and axis.label.isVisible()):
        return ""
    return figure_text.plain_text(axis.labelText)


def _boxes(widget):
    """The Title / X label / Y label boxes of the Appearance section, as typed."""
    return (widget.title.text(), widget.x_label.text(), widget.y_label.text())


def _in_explorer(qtbot, page, page_id):
    """Embed ``page`` in a real Analysis Explorer, as the app does (``show_page``)."""
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 880)
    explorer.show_page(page_id, page)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    return explorer


def _said(page, explorer):
    """The page's status line as the user sees it: ``(message, on screen)`` -- the bar is
    visible and lies inside the Explorer's rectangle."""
    bar = page.statusBar()
    top_left = bar.mapTo(explorer, QtCore.QPoint(0, 0))
    inside = explorer.rect().contains(QtCore.QRect(top_left, bar.size()))
    return bar.currentMessage(), bool(bar.isVisible() and bar.height() > 0 and inside)


def _clear_status(page):
    page.statusBar().clearMessage()


# -- the pure rule ----------------------------------------------------------------------------------

def test_the_rule_clears_the_words_and_nothing_else():
    from nanoscope.gui.fit_label import FitLabelState
    from nanoscope.gui.quantity_reset import cleared_words, reset_sentence, words_cleared

    look = PlotAppearance(
        title="Capture vs voltage", x_label="V", y_label="Rate (Hz)", font_family="Arial",
        title_size=14.0, background="light", grid=False, data_color="#0072b2",
        text=(("title", TextStyle(size_pt=16.0, bold=True, color=ORANGE)),),
        fit_label=FitLabelState(template="Single exponential: {equation}",
                                fields=("equation", "n"), placement="custom",
                                offset=(40.0, 30.0), size_pt=9.0, color="#009e73"),
        axes=PlotAppearance().with_axis_pin("left", figure_text.AxisPin(min=0.0, max=5.0)).axes)
    after, names = cleared_words(look)
    assert names == ("Title", "X label", "Y label", "Fit label wording")
    assert (after.title, after.x_label, after.y_label) == (None, None, None)
    assert after.fit_label == replace(look.fit_label, template="")
    # Everything else is the look, and stays exactly as it was.
    assert replace(after, title=look.title, x_label=look.x_label, y_label=look.y_label,
                   fit_label=look.fit_label) == look
    assert words_cleared(look, after) == names
    # Nothing typed: the same object back, no names, no sentence.
    assert cleared_words(after) == (after, ())
    assert reset_sentence((), "median depth") == ""
    # The caption's text is not one of the words.
    caption = look.with_text_style("annotation.provenance", TextStyle(text="Pooled by hand"))
    assert cleared_words(caption)[0].text_style("annotation.provenance").text == "Pooled by hand"
    # A label hidden with a single space is the same field, and goes with the others.
    hidden = PlotAppearance(title=" ")
    assert cleared_words(hidden) == (PlotAppearance(), ("Title",))
    # The wording alone (Voltage dependence's trend), two labels alone, and none at all.
    from nanoscope.gui.quantity_reset import typed_words

    wording, names = cleared_words(look, only=("fit_label",))
    assert names == ("Fit label wording",) == typed_words(look, only=("fit_label",))
    assert wording == replace(look, fit_label=replace(look.fit_label, template=""))
    labels, names = cleared_words(look, only=("title", "y_label"))
    assert names == ("Title", "Y label")
    assert (labels.title, labels.x_label, labels.y_label) == (None, look.x_label, None)
    assert labels.fit_label == look.fit_label
    assert cleared_words(look, only=()) == (look, ()) and typed_words(look, only=()) == ()


def test_announce_shows_the_sentence_and_takes_down_only_its_own():
    from nanoscope.gui.quantity_reset import announce

    class Bar:
        """What announce uses of a QStatusBar, by name."""
        def __init__(self):
            self.text, self.shown = "", []

        def showMessage(self, text, msecs):
            self.text = text
            self.shown.append((text, msecs))

        def currentMessage(self):
            return self.text

        def clearMessage(self):
            self.text = ""

    bar = Bar()
    first = announce(bar, "Title reset: the plot now shows a.", "")
    assert first == bar.text == "Title reset: the plot now shows a." and bar.shown == [(first, 10000)]
    assert announce(bar, "", first) == "" and bar.text == ""         # nothing to say: it goes
    bar.text = "Another note"
    assert announce(bar, "", first) == "" and bar.text == "Another note"     # not ours: it stays
    assert announce(bar, "", "") == "" and bar.text == "Another note"
    second = announce(bar, "Title reset: the plot now shows b.", first)      # a new sentence replaces
    assert second == bar.text == "Title reset: the plot now shows b."


def test_the_sentence_names_only_what_was_cleared():
    from nanoscope.gui.quantity_reset import reset_sentence

    assert reset_sentence(("Title",), "median depth") == (
        "Title reset: the plot now shows median depth.")
    assert reset_sentence(("Title", "Y label"), "median depth") == (
        "Title and Y label reset: the plot now shows median depth.")
    assert reset_sentence(("Title", "X label", "Y label"), "the spectrogram") == (
        "Title, X label and Y label reset: the plot now shows the spectrogram.")
    assert reset_sentence(("X label", "Y label", "Fit label wording"), "x") == (
        "X label, Y label and fit label wording reset: the plot now shows x.")
    assert reset_sentence(("Fit label wording",), "x", lead="the plot is now set to show") == (
        "Fit label wording reset: the plot is now set to show x.")


# -- Voltage dependence ---------------------------------------------------------------------------

@pytest.fixture
def vdep(qapp, qtbot, tmp_path):
    """A Voltage dependence page with a linear fit applied, in a real Explorer."""
    from nanoscope.gui.voltage_dep_panel import VDepWindow

    from .test_voltage_dep import _ladder_session

    page = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(page)
    page.set_fit("linear")
    explorer = _in_explorer(qtbot, page, PAGE_VOLTAGE_SERIES)
    return page, explorer


def _type_vdep_words(page):
    """Title (Add title…), both axis labels and the fit label's wording, all on the figure,
    and a 16 pt bold orange title style."""
    plot = page.plot_item()
    page._appearance_widget.fit_label_size.setValue(9.0)       # the fit label's own look
    _edit_double_click(plot, page._caption, "Pooled by hand")  # a caption: not one of the words
    _pin_through_axis_properties(plot, "bottom", -1100.0, -100.0)   # a typed range: the look
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    _edit_double_click(plot, plot.getAxis("bottom").label, "Bias (mV)")
    _edit_double_click(plot, page._fit_label, "Linear fit: {equation}")
    _style_title(plot)


def test_voltage_dependence_metric_pick_resets_the_words_and_keeps_the_look(vdep):
    """Before: a typed 'Rate (Hz)' stayed over median depth in pA, a typed title over another
    metric's points, and the fit label's wording over another metric's fit."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    assert page._metric_combo.currentText() == "Capture rate (Hz)"
    _type_vdep_words(page)
    assert _drawn_title(plot) == ("Capture vs voltage", 16.0, True, ORANGE)
    assert (_label(plot, "left"), _label(plot, "bottom")) == ("Rate (Hz)", "Bias (mV)")
    assert page.fit_label_text().startswith("Linear fit:")
    before = widget.state()
    steps = widget._history.depth()[0]
    _section(page, "Analysis")

    _pick(page._metric_combo, text="Median depth (pA)")

    # ONE step in this window's own history, and it takes the words back, not the metric.
    assert widget._history.depth() == (steps + 1, 0)
    # What is DRAWN: the automatic labels of the new metric, the typed words gone ...
    assert _drawn_title(plot) == ("", None, None, None)
    assert _label(plot, "left") == "Median depth (pA)"
    assert _label(plot, "bottom") == "Applied voltage (mV)"
    assert page.fit_label_text() and not page.fit_label_text().startswith("Linear fit:")
    assert _boxes(widget) == ("", "", "")
    assert widget.fit_label_template.text() == ""
    assert widget.title.placeholderText() == "Automatic"
    assert widget.x_label.placeholderText() == "Automatic: Applied voltage (mV)"
    assert widget.y_label.placeholderText() == "Automatic: Median depth (pA)"
    # ... the look stays: the title style, and the fit label's numbers, place and size ...
    after = widget.state()
    assert after.text_style("title") == TextStyle(size_pt=16.0, bold=True, color=ORANGE)
    assert after.fit_label == replace(before.fit_label, template="")
    assert after.fit_label_state().size_pt == 9.0
    assert after.text_style("annotation.provenance").text == "Pooled by hand"
    assert page.caption_text() == "Pooled by hand"
    assert after.axis_pin("bottom").range() == (-1100.0, -100.0)
    assert _shown(plot) == pytest.approx((-1100.0, -100.0))
    _add_title(plot, "Depth vs voltage")
    assert _drawn_title(plot) == ("Depth vs voltage", 16.0, True, ORANGE)
    assert widget.undo()
    # ... and the status line, where the user sees it, names exactly what went.
    assert _said(page, explorer) == (
        "Title, X label, Y label and fit label wording reset: the plot now shows median depth.",
        True)
    assert widget._history.depth()[0] == steps + 1       # the title typed after the reset: undone
    _press_undo(page)
    assert _drawn_title(plot) == ("Capture vs voltage", 16.0, True, ORANGE)
    assert (_label(plot, "left"), _label(plot, "bottom")) == ("Rate (Hz)", "Bias (mV)")
    assert page.fit_label_text().startswith("Linear fit:")
    assert widget._history.depth()[0] == steps
    assert page._metric_combo.currentText() == "Median depth (pA)"
    assert widget.redo()
    assert _drawn_title(plot) == ("", None, None, None)
    assert _label(plot, "left") == "Median depth (pA)"


def test_voltage_dependence_with_nothing_typed_records_no_step_and_says_nothing(vdep):
    page, explorer = vdep
    widget = page._appearance_widget
    _add_title(page.plot_item(), "Capture vs voltage")
    # Taken back to automatic by hand: nothing is typed any more.
    assert widget.undo()
    steps = widget._history.depth()
    _section(page, "Analysis")
    _clear_status(page)
    _pick(page._metric_combo, text="Median dwell (s)")
    assert widget._history.depth() == steps
    assert _said(page, explorer)[0] == ""
    assert _label(page.plot_item(), "left") == "Median dwell (s)"


def test_voltage_dependence_same_quantity_routes_keep_the_words(vdep, tmp_path):
    """A refresh, the same metric picked again, |V|, log y, accepted-only, a template that
    changes only the look, Undo and Redo: none of them is a change of quantity."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    steps = widget._history.depth()
    words = widget.state()
    _section(page, "Analysis")
    _clear_status(page)

    _pick(page._metric_combo, text="Capture rate (Hz)")          # the same metric again
    _click_button(page._workflow.primary_actions[0])             # Analyze / refresh
    explorer.refresh_current()                                    # the Explorer's Refresh current
    for box in (page._abs_v, page._log_y):                        # drawn the same quantity
        _click_button(box)
        _click_button(box)
    _section(page, "Data")
    _click_button(page._accepted_only)
    _click_button(page._accepted_only)
    _section(page, "Analysis")
    settings, appearance = page.template_settings()
    page.apply_template_settings(settings, appearance)            # a template, same metric
    assert widget.state() == words and widget._history.depth() == steps
    widget._apply_preset(1)                                       # a preset keeps the wording
    assert widget.state().title == "Capture vs voltage" and widget.state().y_label == "Rate (Hz)"
    assert widget.undo()
    assert widget.state() == words
    assert (_label(plot, "left"), _drawn_title(plot)[0]) == ("Rate (Hz)", "Capture vs voltage")
    assert _said(page, explorer)[0] == ""
    # A template that DOES change the metric moves it with the signal blocked, as a project
    # restore does: the rule does not run, and the words are the template's own (here the
    # same ones; a template without words installs none, see the template test below).
    settings = {**settings, "metric": "median_depth"}
    page.apply_template_settings(settings, appearance)
    assert page._metric_combo.currentText() == "Median depth (pA)"
    assert (_label(plot, "left"), _drawn_title(plot)[0]) == ("Rate (Hz)", "Capture vs voltage")
    assert widget.state().title == "Capture vs voltage"
    assert _said(page, explorer)[0] == ""
    # The words stay through Undo and Redo of anything else.
    _edit_double_click(plot, plot.getAxis("bottom").label, "Bias (mV)")
    assert widget.undo() and widget.redo()
    assert widget.state().title == "Capture vs voltage"
    assert _said(page, explorer)[0] == ""


def test_voltage_dependence_a_setter_is_the_combo_but_a_restore_is_not(vdep):
    """``set_metric`` moves the combo and its signal does the rest, as a pick does; a project
    restore (``restore_voltage_dependence``) blocks that signal and installs the saved state."""
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Capture vs voltage")
    artifact = workspace.capture_voltage_dependence(page)
    page.set_metric("median_depth")
    assert widget.state().title is None and _label(plot, "left") == "Median depth (pA)"
    assert widget.undo()
    assert widget.state().title == "Capture vs voltage"
    # The saved state (capture rate, with its title) goes back as it was saved, words included.
    assert workspace.restore_voltage_dependence(page, artifact).restored
    assert page._metric_combo.currentText() == "Capture rate (Hz)"
    assert widget.state().title == "Capture vs voltage"
    assert _label(plot, "left") == "Capture rate (Hz)"
    # The next change after a restore is measured from what the restore drew.
    page.set_metric("median_depth")
    assert widget.state().title is None


def test_voltage_dependence_words_saved_over_another_metric_open_unchanged(
        qapp, qtbot, tmp_path, monkeypatch):
    """A saved project with a typed title over another metric opens as it was saved; the rule
    acts on the NEXT change. One real save, reopened by ``_open_project``."""
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    _install_series(window, ())
    window._open_vdep()
    page = window._vdep_window
    explorer = window._analysis_explorer
    explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    plot, widget = page.plot_item(), page._appearance_widget
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    _add_title(plot, "Depth vs voltage")                     # typed over the NEW metric
    _edit_double_click(plot, plot.getAxis("left").label, "Depth (pA)")
    _click_button(page._abs_v)                               # the same quantity, drawn against |V|
    assert page._metric_combo.currentText() == "Median depth (pA)"
    project = tmp_path / "words.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_vdep()
    QtWidgets.QApplication.processEvents()
    page = reopened._vdep_window
    explorer = reopened._analysis_explorer
    explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    plot, widget = page.plot_item(), page._appearance_widget
    assert page._metric_combo.currentText() == "Median depth (pA)"
    assert (_drawn_title(plot)[0], _label(plot, "left")) == ("Depth vs voltage", "Depth (pA)")
    assert (widget.state().title, widget.state().y_label) == ("Depth vs voltage", "Depth (pA)")
    assert _said(page, explorer)[0] == ""
    # The boxes name the labels the restored figure would draw automatically.
    assert (widget.x_label.placeholderText(), widget.y_label.placeholderText()) == (
        "Automatic: Applied |voltage| (mV)", "Automatic: Median depth (pA)")
    # Opening drew the saved metric: choosing it again changes nothing ...
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    assert widget.state().title == "Depth vs voltage"
    # ... and the next metric does.
    _pick(page._metric_combo, text="Event count")
    assert _drawn_title(plot)[0] == "" and _label(plot, "left") == "Event count"
    assert _said(page, explorer) == (
        "Title and Y label reset: the plot now shows event count.", True)
    assert reopened._dirty, "the reset is an edit of the project"
    reopened._dirty = False


# -- Distribution summaries -------------------------------------------------------------------------

def _series_window(qtbot, tmp_path, monkeypatch):
    """A main window with three synthetic recordings, each with a detection."""
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    _install_series(window, ())
    return window


def _explorer_of(window):
    explorer = window._analysis_explorer
    explorer.resize(1280, 880)
    QtWidgets.QApplication.processEvents()
    return explorer


def _generate(page):
    button = page._workflow.primary_actions[0]
    assert button.text().startswith("Generate")
    _click_button(button)


@pytest.fixture
def summaries(qapp, qtbot, tmp_path, monkeypatch):
    window = _series_window(qtbot, tmp_path, monkeypatch)
    window._open_distribution_summaries()
    page = window._distribution_summary_window
    explorer = _explorer_of(window)
    page.activateWindow()
    _generate(page)
    assert page.has_data() and not page._stale
    yield window, page, explorer
    window._dirty = False


def _type_summary_words(page):
    plot = page._plot
    _edit_double_click(plot, plot.titleLabel, "Dwell ECDF")
    _edit_double_click(plot, plot.getAxis("bottom").label, "Dwell time")
    _edit_double_click(plot, plot.getAxis("left").label, "Fraction")
    _style_title(plot)


def test_summaries_words_go_when_generate_commits_another_measurement_or_summary(summaries):
    """Before: 'Dwell ECDF' stayed over a depth ECDF. The control only edits a draft, so the
    figure on screen still shows the old measurement until Generate commits the new one."""
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    _type_summary_words(page)
    assert _drawn_title(plot) == ("Dwell ECDF", 16.0, True, ORANGE)
    steps = widget._history.depth()
    _section(page, "Data")

    _pick(page._metric_combo, data="depth_pA")           # a draft: the figure is the old one
    assert page._stale and page._workflow.primary_actions[0].text() == "Generate ●"
    assert _drawn_title(plot)[0] == "Dwell ECDF" and _label(plot, "bottom") == "Dwell time"
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""

    _generate(page)                                       # the commit: now the quantity moved
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _drawn_title(plot) == ("ECDF", 16.0, True, ORANGE)     # the automatic title, in the look
    assert _label(plot, "bottom") == page._snapshot["x_label"] != "Dwell time"
    assert _label(plot, "left") == "Empirical F(x) = P(X ≤ x)"
    assert _boxes(widget) == ("", "", "")
    assert widget.title.placeholderText() == "Automatic: ECDF"
    assert widget.x_label.placeholderText() == f"Automatic: {page._snapshot['x_label']}"
    # The look stays: the title style survives, and the automatic title is drawn in it.
    assert widget.state().text_style("title") == TextStyle(size_pt=16.0, bold=True, color=ORANGE)
    assert _said(page, explorer) == (
        "Title, X label and Y label reset: the plot now shows the ECDF of median depth.", True)
    assert widget.undo()                                  # one step brings all three back
    assert (_drawn_title(plot)[0], _label(plot, "bottom"), _label(plot, "left")) == (
        "Dwell ECDF", "Dwell time", "Fraction")
    assert widget.redo()

    # A different summary of the same measurement is a different quantity too.
    _edit_double_click(plot, plot.titleLabel, "Depth ECDF")
    _section(page, "Analysis")
    _pick(page._view_combo, data="box")
    assert _drawn_title(plot)[0] == "Depth ECDF"
    _generate(page)
    assert plot.getAxis("bottom").labelText == "Source"
    assert widget.state().title is None
    assert _said(page, explorer) == (
        "Title reset: the plot now shows a box plot of median depth.", True)


def test_summaries_regenerating_the_same_quantity_keeps_the_words(summaries):
    """Generate again, the Explorer's Refresh current, ticking another source, a recalled
    template and Undo/Redo are not changes of quantity (and the recording is not one)."""
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    _type_summary_words(page)
    words = widget.state()
    steps = widget._history.depth()

    _generate(page)
    explorer.refresh_current()                            # the Explorer's Refresh current
    _section(page, "Data")
    other = next(i for i in range(page._source_list.count())
                 if page._source_list.item(i).checkState() != QtCore.Qt.CheckState.Checked)
    page._source_list.item(other).setCheckState(QtCore.Qt.CheckState.Checked)   # another pore
    _generate(page)
    assert len(page.selected_sources()) == 2
    assert widget.state() == words and widget._history.depth() == steps
    assert _drawn_title(plot) == ("Dwell ECDF", 16.0, True, ORANGE)
    assert (_label(plot, "bottom"), _label(plot, "left")) == ("Dwell time", "Fraction")
    assert _said(page, explorer)[0] == ""

    # A template that carries another measurement and the words: the words arrive with it,
    # and the Generate that commits it keeps them.
    settings, appearance = page.template_settings()
    page.apply_template_settings({**settings, "metric": "frac_depth"}, page.appearance_state())
    assert page._stale and widget.state() == words
    _generate(page)
    assert page._applied_config["metric"] == "frac_depth"
    assert widget.state() == words and widget._history.depth() == steps
    assert _said(page, explorer)[0] == ""
    # Only the NEXT change of quantity resets them.
    _section(page, "Data")
    _pick(page._metric_combo, data="dwell_s")
    _generate(page)
    assert widget.state().title is None
    assert _said(page, explorer)[0].startswith("Title, X label and Y label reset:")
    assert widget.undo() and widget.redo()
    assert widget.state().title is None


def test_summaries_with_nothing_typed_record_no_step_and_say_nothing(summaries):
    window, page, explorer = summaries
    widget = page._appearance_widget
    steps = widget._history.depth()
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_summaries_words_saved_over_another_measurement_open_unchanged(
        summaries, qtbot, tmp_path):
    """One real save, reopened by ``_open_project``: the saved figure opens with its words, a
    Generate of the same quantity keeps them, and the next measurement resets them."""
    window, page, explorer = summaries
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)
    _type_summary_words(page)
    project = tmp_path / "summaries.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_distribution_summaries()
    page = reopened._distribution_summary_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    plot, widget = page._plot, page._appearance_widget
    assert page.has_data() and page._applied_config["metric"] == "depth_pA"
    assert _drawn_title(plot) == ("Dwell ECDF", 16.0, True, ORANGE)
    assert (_label(plot, "bottom"), _label(plot, "left")) == ("Dwell time", "Fraction")
    assert _said(page, explorer)[0] == ""
    _generate(page)                                       # the same quantity, drawn again
    assert widget.state().title == "Dwell ECDF" and _said(page, explorer)[0] == ""
    _section(page, "Data")
    _pick(page._metric_combo, data="dwell_s")
    _generate(page)
    assert widget.state().title is None and _label(plot, "bottom") == page._snapshot["x_label"]
    assert _said(page, explorer) == (
        "Title, X label and Y label reset: the plot now shows the ECDF of dwell time.", True)
    reopened._dirty = False


# -- Comparison figures -----------------------------------------------------------------------------

def spy_on_comparison_renders(monkeypatch) -> list[dict]:
    """Record what every render of Comparison figures' matplotlib figure draws -- its title, the
    title's size and the axis labels -- in the list returned, one entry per render."""
    from nanoscope.plot import comparison_figures as renderer

    drawn: list[dict] = []
    real = renderer.render_comparison

    def spy(*args, **kwargs):
        figure = real(*args, **kwargs)
        shown = [ax for ax in figure.axes if ax.get_visible()]
        drawn.append({"title": figure.get_suptitle(), "size": figure._suptitle.get_fontsize(),
                      "x": sorted({ax.get_xlabel() for ax in shown}),
                      "y": sorted({ax.get_ylabel() for ax in shown})})
        return figure

    monkeypatch.setattr(renderer, "render_comparison", spy)
    return drawn


@pytest.fixture
def comparison(qapp, qtbot, monkeypatch):
    """A Comparison figures page with two sources, in a real Explorer; ``drawn`` records what
    every render of the matplotlib figure drew (its title, axis labels and title size)."""
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow

    from .test_comparison_figures_panel import source

    drawn = spy_on_comparison_renders(monkeypatch)
    page = ComparisonFiguresWindow()
    qtbot.addWidget(page)
    explorer = _in_explorer(qtbot, page, PAGE_COMPARISON_FIGURES)
    page.set_sources([source(), source("second", "Recording B")], selected_keys=["first", "second"])
    page.activateWindow()
    _generate(page)
    assert page.has_data() and drawn[-1]["title"] == "Recording comparisons"
    return page, explorer, drawn


def _type_comparison_words(page):
    _section(page, "Appearance")
    widget = page._appearance_widget
    widget.title_size.setValue(14.0)                          # the look
    _type_in_box(widget.title, "Mine")
    _type_in_box(widget.x_label, "Dw")
    _type_in_box(widget.y_label, "Dp")
    QtTest.QTest.qWait(800)         # a pause longer than the run's window: the typing is one step


def test_comparison_words_go_when_generate_commits_another_figure(comparison):
    """The figure type and the measurements are drafts until Generate, which commits them."""
    page, explorer, drawn = comparison
    widget = page._appearance_widget
    _type_comparison_words(page)
    assert (drawn[-1]["title"], drawn[-1]["x"], drawn[-1]["y"]) == ("Mine", ["Dw"], ["Dp"])
    renders, steps = len(drawn), widget._history.depth()

    _section(page, "Analysis")
    _pick(page._y_combo, data="amplitude_pA")                  # a draft
    assert page._stale and len(drawn) == renders
    assert drawn[-1]["title"] == "Mine" and widget._history.depth() == steps
    assert not page._status.text().startswith("Title")
    _generate(page)                                            # the commit

    assert widget._history.depth() == (steps[0] + 1, 0)
    last = drawn[-1]
    assert last["title"] == "Recording comparisons" and last["size"] == 14.0   # look stays
    assert last["x"] != ["Dw"] and last["y"] != ["Dp"] and "Amplitude" not in last["x"][0]
    assert _boxes(widget) == ("", "", "")
    text = page._status.text()
    assert text.startswith("Title, X label and Y label reset: the plot now shows selected "
                           "amplitude against dwell time, one panel per source.")
    assert page._status.isVisible() and explorer.rect().contains(
        QtCore.QRect(page._status.mapTo(explorer, QtCore.QPoint(0, 0)), page._status.size()))
    assert widget.undo()                                       # the three words, one step
    assert (drawn[-1]["title"], drawn[-1]["x"], drawn[-1]["y"]) == ("Mine", ["Dw"], ["Dp"])
    assert widget.redo()
    assert drawn[-1]["title"] == "Recording comparisons"

    # Another figure type is another quantity.
    _type_comparison_words(page)
    _section(page, "Data")
    _pick(page._view_combo, data="events")
    _generate(page)
    assert drawn[-1]["title"] == "Individual-event comparison" and widget.state().title is None
    assert page._status.text().startswith(
        "Title, X label and Y label reset: the plot now shows individual events of median depth.")


def test_comparison_a_pair_matrix_is_the_quantity_of_its_features(comparison):
    page, explorer, drawn = comparison
    widget = page._appearance_widget
    _section(page, "Data")
    _pick(page._view_combo, data="pairs")
    _type_comparison_words(page)                       # the matrix keeps each feature's own labels
    _generate(page)                                    # facets -> pairs: another figure
    assert widget.state().title is None
    assert page._status.text().startswith(
        "Title, X label and Y label reset: the plot now shows a pair matrix of 4 measurements.")
    _type_comparison_words(page)
    _generate(page)                                    # the same four features: nothing
    assert widget.state().title == "Mine" and not page._status.text().startswith("Title")
    checklist = page._features                         # untick Event area: a click on its row
    row = next(r for r in range(checklist.model.rowCount())
               if checklist.model.item(r).data(QtCore.Qt.ItemDataRole.UserRole) == "ecd_pA_s")
    index = checklist.proxy.mapFromSource(checklist.model.index(row, 0))
    QtTest.QTest.mouseClick(checklist.view.viewport(), L, NOMOD,
                            checklist.view.visualRect(index).center())
    assert page._features.checked() == ["dwell_s", "depth_pA", "snr"]
    assert page._stale and widget.state().title == "Mine"      # a draft: the matrix is the old one
    _generate(page)
    assert widget.state().title is None
    assert page._status.text().startswith(          # a matrix has no X or Y label box to type in
        "Title reset: the plot now shows a pair matrix of 3 measurements.")


def test_comparison_same_quantity_template_and_other_sources_keep_the_words(comparison):
    page, explorer, drawn = comparison
    widget = page._appearance_widget
    _type_comparison_words(page)
    words, steps = widget.state(), widget._history.depth()

    _generate(page)                                            # the same figure again
    page.set_sources([page._sources[key] for key in page._sources],
                     selected_keys=["second"])                 # another recording, same figure
    _generate(page)
    explorer.refresh_current()
    assert widget.state() == words and widget._history.depth() == steps
    assert drawn[-1]["title"] == "Mine" and not page._status.text().startswith("Title")

    # A template with another measurement carries the words; its Generate keeps them.
    settings, _appearance = page.template_settings()
    page.apply_template_settings({**settings, "y": "snr"}, page.appearance_state())
    assert page._stale and widget.state() == words
    _generate(page)
    assert page._applied_config["y"] == "snr" and widget.state() == words
    assert drawn[-1]["title"] == "Mine" and not page._status.text().startswith("Title")
    # The next change resets.
    _section(page, "Analysis")
    _pick(page._y_combo, data="depth_pA")
    _generate(page)
    assert widget.state().title is None and drawn[-1]["title"] == "Recording comparisons"


def test_comparison_with_nothing_typed_records_no_step(comparison):
    page, explorer, drawn = comparison
    widget = page._appearance_widget
    steps = widget._history.depth()
    _section(page, "Analysis")
    _pick(page._y_combo, data="snr")
    _generate(page)
    assert widget._history.depth() == steps and not page._status.text().startswith("Title")
    assert page._status.text().startswith("Generated snapshot")


# -- I-V: one quantity, so no rule to wire ------------------------------------------------------------

def test_iv_has_no_quantity_control_and_a_refresh_keeps_the_words(qapp, qtbot, tmp_path):
    """I-V always shows open-pore current against voltage with one fit: nothing to choose, so
    nothing to reset. If a selector is ever added, this fails until the rule is wired to it."""
    from nanoscope.gui.analysis_explorer import PAGE_IV
    from nanoscope.gui.iv_panel import IVWindow

    from .test_iv import _ohmic_session

    page = IVWindow(_ohmic_session(tmp_path))
    qtbot.addWidget(page)
    explorer = _in_explorer(qtbot, page, PAGE_IV)
    page.activateWindow()
    widget = page._appearance_widget
    chooser = [c for c in page.findChildren(QtWidgets.QComboBox)
               if c is not page._workflow.sections and not widget.isAncestorOf(c)]
    assert chooser == [], [c.accessibleName() for c in chooser]
    plot = page.plot_item()
    _add_title(plot, "I-V of pore A")
    _edit_double_click(plot, plot.getAxis("left").label, "I0 (pA)")
    words = widget.state()
    _click_button(page._workflow.primary_actions[0])                  # Analyze / refresh
    explorer.refresh_current()
    assert widget.state() == words and _said(page, explorer)[0] == ""
    assert (_drawn_title(plot)[0], _label(plot, "left")) == ("I-V of pore A", "I0 (pA)")


# -- Event waveforms --------------------------------------------------------------------------------

@pytest.fixture
def events(qapp, qtbot, tmp_path):
    """An Event waveforms page fed one recording's detection, in a real Explorer."""
    from .test_event_plot import _fed_window

    page = _fed_window(qtbot, tmp_path)
    explorer = _in_explorer(qtbot, page, PAGE_EVENT_WAVEFORMS)
    page.activateWindow()
    assert page.current_view() == "Density" and page.density_counts() is not None
    return page, explorer


def _type_event_words(page, plot=None):
    """The shared title typed on the figure, both axis labels in the Appearance boxes (the views
    share them, so they are renamed there), and a 16 pt bold orange title."""
    plot = plot or page._density_main
    _edit_double_click(plot, plot.titleLabel, "Dwell vs depth")
    _style_title(plot)
    _section(page, "Appearance")
    box = page._appearance_widget
    _type_in_box(box.x_label, "Dwell")
    _type_in_box(box.y_label, "Depth")
    QtTest.QTest.qWait(800)                    # the run of typing closes: one step


def test_event_waveforms_x_metric_pick_resets_the_shared_words_and_keeps_the_look(events):
    page, explorer = events
    plot, widget = page._density_main, page._appearance_widget
    _type_event_words(page)
    assert _drawn_title(plot) == ("Dwell vs depth", 16.0, True, ORANGE)
    assert (_label(plot, "bottom"), _label(plot, "left")) == ("Dwell", "Depth")
    steps = widget._history.depth()
    _section(page, "Analysis")

    _pick(page._x_combo, data="frac_depth")

    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _drawn_title(plot) == (_squash(page._density_auto_title), 16.0, True, ORANGE)
    assert page._density_auto_title.startswith("Median depth (pA) vs dI/I₀")
    assert (_label(plot, "bottom"), _label(plot, "left")) == (
        "dI/I₀ (log10)", "Median depth (pA)")
    assert _boxes(widget) == ("", "", "")
    assert _said(page, explorer) == (
        "Title, X label and Y label reset: the plot now shows median depth against dI/I₀.", True)
    assert widget.undo()
    assert (_drawn_title(plot)[0], _label(plot, "bottom"), _label(plot, "left")) == (
        "Dwell vs depth", "Dwell", "Depth")
    assert widget.redo()

    # The Y metric is the other axis of the same quantity pair.
    _type_event_words(page)
    _section(page, "Analysis")
    _pick(page._y_combo, data="dwell_s")
    assert widget.state().title is None and _said(page, explorer) == (
        "Title, X label and Y label reset: the plot now shows dwell against dI/I₀.", True)


def test_event_waveforms_view_switch_and_normalize_time_reset_the_shared_words(events):
    page, explorer = events
    widget = page._appearance_widget
    _type_event_words(page)
    bar = page._tabs.tabBar()

    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())     # the Ensemble tab
    assert page.current_view() == "Ensemble"
    assert widget.state().title is None and _boxes(widget) == ("", "", "")
    assert _said(page, explorer) == (
        "Title, X label and Y label reset: the plot now shows the ensemble.", True)
    plot = page._ensemble_plot
    assert _drawn_title(plot)[0] == _squash(page._ensemble_auto_title)
    assert _label(plot, "bottom") == "Normalized time (dwell)"
    assert widget.undo()                                                  # all three, one step
    assert _drawn_title(plot)[0] == "Dwell vs depth" and _label(plot, "bottom") == "Dwell"
    assert widget.redo()

    # Normalize time changes what the ensemble's x axis is.
    _edit_double_click(plot, plot.titleLabel, "Mean event")
    _clear_status(page)
    _section(page, "Analysis")
    _click_button(page._normalize)
    assert page._normalize.isChecked() is False and widget.state().title is None
    assert _label(plot, "bottom") == "Time (s)"
    assert _said(page, explorer) == (
        "Title reset: the plot now shows the ensemble against time in seconds.", True)

    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())     # and back: nothing typed
    steps = widget._history.depth()
    _clear_status(page)
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_event_waveforms_presentation_recordings_templates_and_restores_keep_the_words(
        events, tmp_path):
    """Log axes, bins, the colormap, the band, the alignment, another recording, a template
    carrying another metric, and a workspace restore are not changes of quantity."""
    from nanoscope.core import analysis_workspace as workspace

    from .test_event_plot import _doc_and_events, _install_detection

    page, explorer = events
    widget = page._appearance_widget
    _type_event_words(page)
    words, steps = widget.state(), widget._history.depth()
    _section(page, "Analysis")
    for box in (page._logx, page._logy):
        _click_button(box)
        _click_button(box)
    page.set_bins(page._bins.value() + 4)
    page.set_colormap("magma")
    page.set_band("sd")
    page.set_align("center")
    page.set_n_points(page._npoints.value() + 4)
    # Another recording selected: its events replace the bound source (what MainWindow does).
    doc, ev = _doc_and_events(tmp_path, name="other.dat", n_events=12)
    doc_id = page._session.add_document(doc)
    _install_detection(page._session, doc_id, ev)
    page.set_source(doc_id, 0, ev)
    assert page.density_counts().sum() == 12
    assert widget.state() == words and widget._history.depth() == steps
    assert _said(page, explorer)[0] == ""

    # A template carrying another metric (its appearance, here with the same words) and a
    # restore of the saved workspace move the controls with the signal blocked: the rule does
    # not run, and the words are the template's and the workspace's own.
    settings, _appearance = page.template_settings()
    page.apply_template_settings({**settings, "x_metric": "frac_depth"}, page.appearance_state())
    assert page._x_combo.currentData() == "frac_depth" and widget.state() == words
    artifact = workspace.capture_event_plots(page)
    assert workspace.restore_event_plots(page, artifact).restored
    assert widget.state() == words and _said(page, explorer)[0] == ""
    assert _drawn_title(page._density_main)[0] == "Dwell vs depth"
    # The next metric measured from what the restore drew.
    _section(page, "Analysis")
    _pick(page._x_combo, data="dwell_s")
    assert widget.state().title is None


def test_event_waveforms_with_nothing_typed_records_no_step(events):
    page, explorer = events
    widget = page._appearance_widget
    steps = widget._history.depth()
    _section(page, "Analysis")
    _pick(page._x_combo, data="ecd")
    bar = page._tabs.tabBar()
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_event_waveforms_words_saved_over_another_metric_open_unchanged(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save, reopened by ``_open_project``, through the main window's Explorer."""
    window = _series_window(qtbot, tmp_path, monkeypatch)
    first = next(iter(window._session.documents())).doc_id
    window._on_selection(first, 0)
    window._open_event_plots()
    page = window._event_plot_window
    explorer = _explorer_of(window)
    page.activateWindow()
    assert page.density_counts() is not None
    _section(page, "Analysis")
    _pick(page._y_combo, data="frac_depth")
    _type_event_words(page)
    project = tmp_path / "events.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_event_plots()
    page = reopened._event_plot_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    widget = page._appearance_widget
    assert page._y_combo.currentData() == "frac_depth"
    assert (widget.state().title, widget.state().x_label) == ("Dwell vs depth", "Dwell")
    assert _drawn_title(page._density_main)[0] == "Dwell vs depth"
    assert _said(page, explorer)[0] == ""
    _section(page, "Analysis")
    _pick(page._y_combo, data="frac_depth")                 # the same metric: nothing
    assert widget.state().title == "Dwell vs depth"
    _pick(page._y_combo, data="depth_pA")                   # the next change resets
    assert widget.state().title is None and _said(page, explorer)[0].startswith(
        "Title, X label and Y label reset:")
    reopened._dirty = False


# -- Noise ------------------------------------------------------------------------------------------

@pytest.fixture
def noise(qapp, qtbot, tmp_path):
    """A Noise page with one recording's Welch PSD and its 1/f fit label, in a real Explorer."""
    from .test_psd import _fed_window

    page, _doc_id = _fed_window(qtbot, tmp_path)
    explorer = _in_explorer(qtbot, page, PAGE_NOISE)
    page.activateWindow()
    assert page.has_data() and page._fit_item is not None and page.fit_label_text()
    return page, explorer


def _type_noise_words(page):
    """The shared title and the fit label's wording on the figure, both axis labels in the
    Appearance boxes (the four views share them), and a 16 pt bold orange title."""
    plot = page._plot
    _edit_double_click(plot, plot.titleLabel, "PSD, pore A")
    _edit_double_click(plot, page._fit_label, "Flicker: {equation}")
    _style_title(plot)
    _section(page, "Appearance")
    box = page._appearance_widget
    _type_in_box(box.x_label, "f")
    _type_in_box(box.y_label, "S")
    QtTest.QTest.qWait(800)


def test_noise_view_pick_resets_the_shared_words_and_keeps_the_look(noise):
    """Before: a PSD title stayed over the spectrogram (one title shared by all views)."""
    page, explorer = noise
    plot, widget = page._plot, page._appearance_widget
    _type_noise_words(page)
    assert _drawn_title(plot) == ("PSD, pore A", 16.0, True, ORANGE)
    assert page.fit_label_text().startswith("Flicker:")
    assert (_label(plot, "bottom"), _label(plot, "left")) == ("f", "S")
    steps = widget._history.depth()
    _section(page, "Data")

    _pick(page._view_combo, data="spectrogram")

    assert page.current_view() == "spectrogram"
    assert widget._history.depth() == (steps[0] + 1, 0)
    spec = page._spec_plot
    assert _drawn_title(spec)[0].startswith("Spectrogram — ") and _drawn_title(spec)[1:] == (
        16.0, True, ORANGE)                                             # the look, on the new view
    assert _label(spec, "bottom") == "Time (s from the start of this recording)"
    assert _label(spec, "left") == "Frequency (Hz)"
    assert _boxes(widget) == ("", "", "")
    assert _said(page, explorer) == (
        "Title, X label, Y label and fit label wording reset: the plot now shows the "
        "spectrogram.", True)
    assert widget.undo()
    assert _drawn_title(spec)[0] == "PSD, pore A" and _label(spec, "left") == "S"
    assert widget.redo()
    # Back to the PSD: the automatic labels of the spectrum, and the fit label in its own words
    # (nothing typed since, so nothing to say).
    _clear_status(page)
    _pick(page._view_combo, data="psd")
    assert _drawn_title(page._plot)[0].startswith("PSD — ")
    assert page.fit_label_text() and not page.fit_label_text().startswith("Flicker:")
    assert _said(page, explorer)[0] == ""


def test_noise_asd_switch_resets_the_words_but_log_axes_and_welch_settings_do_not(noise):
    page, explorer = noise
    plot, widget = page._plot, page._appearance_widget
    _type_noise_words(page)
    words, steps = widget.state(), widget._history.depth()
    _section(page, "Analysis")
    _click_button(page._loglog_check)                       # the same quantity on other axes
    _click_button(page._loglog_check)
    page.set_nperseg(2048)
    page.set_overlap_percent(25)
    page.set_window("flattop")
    assert widget.state() == words and widget._history.depth() == steps
    assert _drawn_title(plot)[0] == "PSD, pore A" and _said(page, explorer)[0] == ""

    _click_button(page._asd_check)                          # PSD -> ASD: another quantity
    assert page.is_asd()
    assert widget.state().title is None and _label(plot, "left").startswith("ASD (")
    assert _drawn_title(plot)[0].startswith("ASD — ")
    assert _said(page, explorer) == (
        "Title, X label, Y label and fit label wording reset: the plot now shows the "
        "amplitude spectral density.", True)
    assert widget.undo() and _drawn_title(plot)[0] == "PSD, pore A"


def test_noise_another_recording_a_comparison_a_template_and_a_restore_keep_the_words(
        noise, tmp_path):
    from nanoscope.core import analysis_workspace as workspace

    from .test_psd import _current_doc

    page, explorer = noise
    plot, widget = page._plot, page._appearance_widget
    _type_noise_words(page)
    words, steps = widget.state(), widget._history.depth()
    _section(page, "Data")
    page._session.add_document(_current_doc(tmp_path, name="second.dat"))
    page.refresh()
    assert page._rec_combo.count() == 2
    _pick(page._rec_combo, text=page._rec_combo.itemText(1))   # another recording, picked
    assert widget.state() == words and widget._history.depth() == steps
    assert _drawn_title(plot)[0] == "PSD, pore A" and _said(page, explorer)[0] == ""
    item = page._compare_list.item(0)                          # ... and one to compare with
    item.setCheckState(QtCore.Qt.CheckState.Checked)
    assert widget.state() == words and _said(page, explorer)[0] == ""

    # A template and a workspace restore move the view with the signal down: the rule does
    # not run, and the words are the template's and the workspace's own (here the same ones).
    settings, _appearance = page.template_settings()
    page.apply_template_settings({**settings, "view": "fft"}, page.appearance_state())
    assert page.current_view() == "fft" and widget.state() == words
    artifact = workspace.capture_psd(page)
    assert workspace.restore_psd(page, artifact).restored
    assert page.current_view() == "fft" and widget.state() == words
    assert _said(page, explorer)[0] == ""
    # The next view is measured from what the restore showed.
    _section(page, "Data")
    _pick(page._view_combo, data="allpoints")
    assert widget.state().title is None


def test_noise_with_nothing_typed_records_no_step(noise):
    page, explorer = noise
    widget = page._appearance_widget
    steps = widget._history.depth()
    _section(page, "Data")
    _pick(page._view_combo, data="fft")
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""


def test_noise_words_saved_over_another_view_open_unchanged(qapp, qtbot, tmp_path, monkeypatch):
    """One real save, reopened by ``_open_project``: the page opens on the view it was saved
    on, with its words, and the next view resets them."""
    from nanoscope.core.project import DocumentRef, ProjectState
    from nanoscope.gui.main_window import MainWindow

    from .test_plot_window_routes import _noisy_csv

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    recording = tmp_path / "noise.csv"
    _noisy_csv(recording)
    window = _main_window(qtbot)
    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._begin_restore(ProjectState(documents=[DocumentRef(str(recording))]))
    window._open_psd()
    page = window._psd_window
    explorer = _explorer_of(window)
    page.activateWindow()
    assert page.has_data()
    _section(page, "Data")
    _pick(page._view_combo, data="fft")
    _edit_double_click(page._fft_plot, page._fft_plot.titleLabel, "FFT of pore A")
    project = tmp_path / "noise.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_psd()
    page = reopened._psd_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    widget = page._appearance_widget
    assert page.current_view() == "fft" and widget.state().title == "FFT of pore A"
    assert _drawn_title(page._fft_plot)[0] == "FFT of pore A"
    assert _said(page, explorer)[0] == ""
    _section(page, "Data")
    _pick(page._view_combo, data="fft")                      # the same view: nothing
    assert widget.state().title == "FFT of pore A"
    _pick(page._view_combo, data="allpoints")
    assert widget.state().title is None
    assert _said(page, explorer) == (
        "Title reset: the plot now shows the all-points histogram.", True)
    reopened._dirty = False


# -- Subpopulations ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def subpop_report():
    """One real report (about a second); its embedding table also carries UMAP columns,
    exactly as ``test_subpop_views`` does (UMAP is an opt-in that is never run in tests)."""
    from nanoscope.analysis import subpop_report as SR

    from .test_subpop_views import FS as SUB_FS
    from .test_subpop_views import N_SAMPLES as SUB_N
    from .test_subpop_views import _synthetic_events, _with_umap

    arr, ev = _synthetic_events()
    config = SR.SubpopConfig(k_grid=(1, 2, 3), n_splits=2, n_boot=0)
    report = SR.run_subpopulations(arr, arr, SUB_FS, ev, recording="r",
                                   duration_s=SUB_N / SUB_FS, config=config)
    return _with_umap(report)


@pytest.fixture
def subpop(qapp, qtbot, subpop_report):
    from nanoscope.gui.subpop_window import SubpopReportWindow

    page = SubpopReportWindow()
    qtbot.addWidget(page)
    explorer = _in_explorer(qtbot, page, PAGE_SUBPOPULATIONS)
    page.set_report(subpop_report, subtitle="synthetic")
    page.activateWindow()
    assert page.embedding_view().embedding() == "t-SNE"
    return page, explorer


def _type_subpop_words(page):
    plot = page.embedding_view()._plot.getPlotItem()
    _add_title(plot, "Pore A in t-SNE")
    _style_title(plot)
    _section(page, "Appearance")
    box = page._appearance_widget
    _type_in_box(box.x_label, "u")
    _type_in_box(box.y_label, "v")
    QtTest.QTest.qWait(800)


def _standing_line(page, explorer):
    """The window's standing line under the document, as drawn, and whether it is on screen."""
    label = page._live_note
    inside = explorer.rect().contains(QtCore.QRect(label.mapTo(explorer, QtCore.QPoint(0, 0)),
                                                   label.size()))
    return page.live_note(), bool(label.isVisible() and inside)


def test_subpopulations_embedding_pick_resets_the_shared_words_and_keeps_the_look(subpop):
    page, explorer = subpop
    scatter = page.embedding_view()
    plot, widget = scatter._plot.getPlotItem(), page._appearance_widget
    _type_subpop_words(page)
    assert _drawn_title(plot) == ("Pore A in t-SNE", 16.0, True, ORANGE)
    assert (_label(plot, "bottom"), _label(plot, "left")) == ("u", "v")
    steps = widget._history.depth()
    _section(page, "Analysis")

    _pick(scatter._embed_combo, text="UMAP")

    assert scatter.embedding() == "UMAP" and widget._history.depth() == (steps[0] + 1, 0)
    assert _drawn_title(plot) == ("", None, None, None)
    assert widget.state().text_style("title") == TextStyle(size_pt=16.0, bold=True, color=ORANGE)
    assert (_label(plot, "bottom"), _label(plot, "left")) == (
        "UMAP 1 (arb. units)", "UMAP 2 (arb. units)")
    assert _boxes(widget) == ("", "", "")
    assert _standing_line(page, explorer) == (
        "Title, X label and Y label reset: the plot now shows the UMAP embedding.", True)
    assert widget.undo()
    assert (_drawn_title(plot)[0], _label(plot, "bottom")) == ("Pore A in t-SNE", "u")
    assert widget.redo()
    _add_title(plot, "Pore A in UMAP")
    assert _drawn_title(plot) == ("Pore A in UMAP", 16.0, True, ORANGE)
    # ... and back: the next pick is measured from the embedding on screen.
    _pick(scatter._embed_combo, text="t-SNE")
    assert widget.state().title is None and _standing_line(page, explorer) == (
        "Title reset: the plot now shows the t-SNE embedding.", True)


def test_subpopulations_colouring_another_report_and_nothing_typed_keep_the_words(
        subpop, subpop_report):
    page, explorer = subpop
    scatter = page.embedding_view()
    widget = page._appearance_widget
    steps = widget._history.depth()
    _section(page, "Analysis")
    _pick(scatter._embed_combo, text="UMAP")                 # nothing typed: no step, no sentence
    assert widget._history.depth() == steps and _standing_line(page, explorer)[0] == ""
    _pick(scatter._embed_combo, text="t-SNE")
    _type_subpop_words(page)
    words, steps = widget.state(), widget._history.depth()
    _section(page, "Analysis")
    _pick(scatter._colour_combo, text=scatter.colour_modes()[1])   # colouring is how it is drawn
    page.set_report(subpop_report, subtitle="a second run")        # a re-run, same embedding
    assert widget.state() == words and widget._history.depth() == steps
    assert _standing_line(page, explorer)[0] == ""
    # Undo and Redo of the reset itself do not run it again.
    _pick(scatter._embed_combo, text="UMAP")
    assert widget.state().title is None
    assert widget.undo() and widget.state() == words
    assert widget.redo() and widget.state().title is None
    assert widget._history.depth() == (steps[0] + 1, 0)


# -- Distributions ----------------------------------------------------------------------------------

def _wordy_config(**changes):
    """A Distributions configuration with typed words, a 16 pt bold title style, and a fit
    label whose wording, size and colour are set."""
    from nanoscope.gui.fit_label import FitLabelState
    from nanoscope.gui.plots_panel import (
        AxesConfig, FitConfig, HistogramConfig, PlotConfig, ScatterConfig, StyleSettings)

    style = StyleSettings(title="Dwell, pore A", x_label="Dwell (ms)", y_label="Events").with_text_style(
        "title", TextStyle(text="Dwell, pore A", size_pt=16.0, bold=True, color=ORANGE))
    style = style.with_fit_label(FitLabelState(template="Single exponential: {equation}",
                                               size_pt=9.0, color="#009e73"))
    return replace(PlotConfig(
        plot_type="Histogram", histogram=HistogramConfig(column="dwell_s"),
        scatter=ScatterConfig(x="dwell_s", y="depth_pA"),
        fit=FitConfig(applied="exponential", applied_k=1), axes=AxesConfig(time_unit="ms"),
        style=style), **changes)


def test_the_distributions_rule_as_a_value():
    """``PlotConfig.rebased``: the wording goes with the plot type, a column, or the applied
    model; the caption goes with the first two only; the look stays in every case."""
    from nanoscope.gui.plots_panel import FitConfig, HistogramConfig

    before = _wordy_config()
    look = replace(before.style.fit_label_state(), template="")

    def words(cfg):
        style = cfg.style
        return (style.title, style.x_label, style.y_label, style.fit_label_state().template)

    assert words(before) == ("Dwell, pore A", "Dwell (ms)", "Events", "Single exponential: {equation}")
    # The plot type: all four words, and none of the look.
    after = replace(before, plot_type="Scatter", fit=FitConfig(applied_scatter="linear")
                    ).rebased(before.identity)
    assert words(after) == ("", "", "", "")
    assert after.style.fit_label_state() == look
    assert after.style.text_style("title").size_pt == 16.0 and after.style.text_style("title").bold
    # A column: the same.
    after = replace(before, histogram=HistogramConfig(column="frac_depth")).rebased(before.identity)
    assert words(after) == ("", "", "", "") and after.style.fit_label_state() == look
    # The applied model alone: the wording only.
    for fit in (FitConfig(applied="gamma (peaked)"), FitConfig(applied="exponential", applied_k=2),
                FitConfig(applied="none")):
        after = replace(before, fit=fit).rebased(before.identity)
        assert words(after) == ("Dwell, pore A", "Dwell (ms)", "Events", ""), fit
        assert after.style.fit_label_state() == look
    # A scatter's applied line is its model too.
    scatter = _wordy_config(plot_type="Scatter", fit=FitConfig(applied_scatter="linear"))
    after = replace(scatter, fit=FitConfig(applied_scatter="power law (log-log)")
                    ).rebased(scatter.identity)
    assert words(after)[3] == "" and words(after)[0] == "Dwell, pore A"
    # What is NOT a change: the selected (not applied) model, a mixture whose count the draw
    # writes back, the time unit, a first fit, and a repeat.
    for fit in (FitConfig(kind="gamma (peaked)", applied="exponential", applied_k=1),):
        assert replace(before, fit=fit).rebased(before.identity) == replace(before, fit=fit)
    auto = _wordy_config(fit=FitConfig(applied="exponential", applied_k=2, applied_k_auto=True))
    drawn = replace(auto, fit=FitConfig(applied="exponential", applied_k=3, applied_k_auto=True))
    assert words(drawn.rebased(auto.identity)) == words(auto)
    assert words(replace(before, axes=replace(before.axes, time_unit="µs")
                         ).rebased(before.identity)) == words(before)
    ahead = _wordy_config(fit=FitConfig())                                  # words typed, no fit yet
    assert words(replace(ahead, fit=FitConfig(applied="exponential", applied_k=1)
                         ).rebased(ahead.identity)) == words(ahead)
    assert before.rebased(before.identity) is before
    assert before.identity.fit_model == "exponential:1" and ahead.identity.fit_model == ""
    assert _wordy_config(plot_type="Scatter", fit=FitConfig(applied_scatter="linear")
                         ).identity.fit_model == "linear"



def _dist_frame(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 5, n)
    return pd.DataFrame({
        "xcol": x, "ycol": 2 * x + 5 + rng.normal(0, .3, n),
        "dwell_s": rng.exponential(0.002, n) + 1e-4, "depth_pA": rng.normal(50, 5, n),
        "frac_depth": rng.uniform(0.15, 0.45, n),
        "accepted": np.ones(n, dtype=bool), "filter_limited": np.zeros(n, dtype=bool)})


@pytest.fixture
def dist(qapp, qtbot):
    """Distributions with a fitted dwell histogram, in a real Explorer."""
    from nanoscope.core.session import Session
    from nanoscope.gui.plots_panel import PlotsWindow

    page = PlotsWindow(Session())
    qtbot.addWidget(page)
    explorer = _in_explorer(qtbot, page, PAGE_DISTRIBUTIONS)
    page.set_data({"probe": _dist_frame(), "other": _dist_frame(1)})
    page.set_sources(["probe"])
    page.set_scatter_columns("xcol", "ycol")
    page.set_plot_type("Histogram")
    page.set_histogram_column("dwell_s")
    page.set_fit("exponential")
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    assert page._fit_label_item.isVisible() and page._applied_fit == "exponential"
    return page, explorer


def _type_dist_words(page):
    plot = page._plot
    _add_title(plot, "Dwell, pore A")
    _edit_double_click(plot, plot.getAxis("bottom").label, "Dwell time (ms)")
    _edit_double_click(plot, plot.getAxis("left").label, "Events per bin")
    _edit_double_click(plot, page._fit_label_item, "Single exponential: {equation}, {n}")
    _style_title(plot)


def _dist_words(page):
    style = page.config().style
    return (style.title, style.x_label, style.y_label, style.fit_label_state().template)


def test_distributions_plot_type_resets_the_fit_wording_with_the_caption_in_one_undo_step(dist):
    """Before: the wording typed for an exponential stayed over a linear fit
    ('Single exponential: y = 1.992·x + 5.027, n = 300'); the caption already went."""
    page, explorer = dist
    plot = page._plot
    _type_dist_words(page)
    assert page._fit_label_item.plain_text().startswith("Single exponential:")
    assert _drawn_title(plot) == ("Dwell, pore A", 16.0, True, ORANGE)
    look_before = TextStyle(size_pt=16.0, bold=True, color=ORANGE)
    page._history_touch()                                    # close the run: the typing is settled
    depth = page._history.depth()[0]

    _pick(page._plot_type_combo, text="Scatter")

    assert _dist_words(page) == ("", "", "", "")
    assert page._history.depth()[0] == depth + 1
    assert _said(page, explorer) == (
        "Title, X label, Y label and fit label wording reset: the plot is now set to show the "
        "scatter of ycol against xcol.", True)
    _user_picks(page._scatter_fit, "linear")
    _click_button(page._plot_btn)                            # Plot: the picture follows
    assert _drawn_title(plot)[0] == ""
    assert (_label(plot, "bottom"), _label(plot, "left")) == ("xcol", "ycol")
    text = page._fit_label_item.plain_text()
    assert text.startswith("y = ") and "Single exponential" not in text
    assert replace(page.config().style.text_style("title"), text=None) == look_before   # look stays
    # Distributions' own Undo: the plot type and the words come back together.
    assert page.undo_config()
    while page._plot_type_combo.currentText() != "Histogram" and page.undo_config():
        pass
    assert page._plot_type_combo.currentText() == "Histogram"
    assert _dist_words(page) == ("Dwell, pore A", "Dwell time (ms)", "Events per bin",
                                 "Single exponential: {equation}, {n}")


def test_distributions_column_change_clears_the_caption_and_the_wording(dist):
    page, explorer = dist
    _type_dist_words(page)
    _user_picks(page._hist_col, "frac_depth")
    assert _dist_words(page) == ("", "", "", "")
    assert _said(page, explorer)[0] == (
        "Title, X label, Y label and fit label wording reset: the plot is now set to show the "
        "histogram of frac_depth, fitted with exponential.")


def test_distributions_a_new_fit_model_clears_only_the_wording(dist):
    """The fit label's words describe one fit: another model, or another component count,
    takes them back to automatic when it is APPLIED (the picture only follows Apply fit)."""
    page, explorer = dist
    plot = page._plot
    _type_dist_words(page)
    page._appearance_widget.fit_label_size.setValue(9.0)       # the label's own look
    page._history_touch()
    _user_picks(page._fit_combo, "gamma (peaked)")             # a proposal: the picture is the old fit
    assert _dist_words(page)[3].startswith("Single exponential")
    assert page._fit_apply_btn.text() == "Apply fit ●" and _said(page, explorer)[0] == ""
    depth = page._history.depth()[0]

    _click_button(page._fit_apply_btn)

    assert _dist_words(page) == ("Dwell, pore A", "Dwell time (ms)", "Events per bin", "")
    assert page._history.depth()[0] == depth + 1               # the apply and the reset: one step
    assert _said(page, explorer) == (
        "Fit label wording reset: the plot is now set to show the histogram of dwell_s, fitted "
        "with gamma (peaked).", True)
    assert _drawn_title(plot) == ("Dwell, pore A", 16.0, True, ORANGE)
    assert page.config().style.fit_label_state().size_pt == 9.0           # the look stays
    assert not page._fit_label_item.plain_text().startswith("Single")
    assert page.undo_config()
    assert page._applied_fit == "exponential" and _dist_words(page)[3].startswith("Single")

    # The component count of an exponential mixture is part of the model.
    assert page.redo_config() and page._applied_fit == "gamma (peaked)"
    _user_picks(page._fit_combo, "exponential")
    _click_button(page._fit_apply_btn)
    assert page._applied_fit == "exponential" and page._applied_fit_k == 1
    _edit_double_click(plot, page._fit_label_item, "One state: {equation}")
    assert _dist_words(page)[3] == "One state: {equation}"
    page._fit_k.setValue(2)                                  # a setter commits at once
    assert page._applied_fit_k == 2
    assert _dist_words(page) == ("Dwell, pore A", "Dwell time (ms)", "Events per bin", "")
    _edit_double_click(plot, page._fit_label_item, "Two states: {equation}")
    page._fit_k.setValue(2)                                  # the same model again: nothing
    assert _dist_words(page)[3] == "Two states: {equation}"


def test_distributions_a_scatter_line_is_a_model_too(dist):
    page, explorer = dist
    plot = page._plot
    page.set_plot_type("Scatter")
    _user_picks(page._scatter_fit, "linear")
    _click_button(page._scatter_fit_apply_btn)
    assert page._applied_scatter_fit == "linear" and page._fit_label_item.isVisible()
    _add_title(plot, "Calibration line")
    _edit_double_click(plot, page._fit_label_item, "Straight line: {equation}")
    assert _dist_words(page)[0] == "Calibration line"
    assert _dist_words(page)[3] == "Straight line: {equation}"
    _user_picks(page._scatter_fit, "power law (log-log)")
    assert _dist_words(page)[3] == "Straight line: {equation}"          # a proposal changes nothing
    _click_button(page._scatter_fit_apply_btn)
    assert page._applied_scatter_fit == "power law (log-log)"
    assert _dist_words(page)[0] == "Calibration line" and _dist_words(page)[3] == ""
    assert _said(page, explorer) == (
        "Fit label wording reset: the plot is now set to show the scatter of ycol against xcol, "
        "fitted with power law (log-log).", True)


def test_distributions_a_first_fit_keeps_wording_typed_ahead_of_it(dist):
    page, explorer = dist
    _user_picks(page._fit_combo, "none")
    _click_button(page._fit_apply_btn)
    assert page._applied_fit == "none"
    box = page._appearance_widget.fit_label_template
    _type_in_box(box, "My fit: {equation}")
    assert _dist_words(page)[3] == "My fit: {equation}"
    _user_picks(page._fit_combo, "exponential")
    _click_button(page._fit_apply_btn)                          # none -> a model
    assert page._applied_fit == "exponential"
    assert _dist_words(page)[3] == "My fit: {equation}"
    assert page._fit_label_item.plain_text().startswith("My fit:")
    assert _said(page, explorer)[0] == ""


def test_distributions_time_unit_another_source_and_a_recalled_config_keep_the_words(dist):
    page, explorer = dist
    _type_dist_words(page)
    words = _dist_words(page)
    _user_picks(page._time_unit, "µs")                          # rescales limits, keeps words
    other = next(i for i in range(page._source_list.count())
                 if page._source_list.item(i).checkState() != QtCore.Qt.CheckState.Checked)
    page._source_list.item(other).setCheckState(QtCore.Qt.CheckState.Checked)   # another pore
    _click_button(page._plot_btn)
    assert _dist_words(page) == words and _said(page, explorer)[0] == ""
    recorded = page.config()
    _user_picks(page._plot_type_combo, "Scatter")
    assert _dist_words(page) == ("", "", "", "")
    page.apply_config(recorded)                                  # a template / Undo-like recall
    assert page._plot_type_combo.currentText() == "Histogram" and _dist_words(page) == words
    page._history_touch()
    page.statusBar().clearMessage()
    _click_button(page._plot_btn)                                # a re-plot of the same quantity
    assert _dist_words(page) == words and _said(page, explorer)[0] == ""


def test_distributions_a_refresh_keeps_the_words_unless_the_plotted_column_is_gone(dist):
    """The guide's sentence: a refresh keeps the words 'unless a refreshed source no longer has
    the plotted column'. ``set_data`` is what the main window pushes on a refresh."""
    page, explorer = dist
    _type_dist_words(page)
    words = _dist_words(page)
    page.statusBar().clearMessage()
    page.set_data({"probe": _dist_frame(7), "other": _dist_frame(1)})       # same columns, new rows
    assert _dist_words(page) == words and _said(page, explorer)[0] == ""
    page.set_data({"probe": _dist_frame(7).drop(columns=["dwell_s"]),
                   "other": _dist_frame(1).drop(columns=["dwell_s"])})      # the plotted column left
    assert page._hist_col.currentText() != "dwell_s"
    assert _dist_words(page) == ("", "", "", "")
    assert _said(page, explorer)[0].startswith("Title, X label, Y label and fit label wording reset")


def test_distributions_ticking_a_source_without_the_plotted_column_moves_it_like_a_pick(dist):
    """The guide's and the CHANGELOG's exception for a tick: the column boxes offer only the
    columns every ticked source has, so ticking a source that lacks the plotted column moves the
    column on its own and the rule runs (one Undo step, its sentence on the status line), while a
    source that has the column changes nothing. ``set_data`` is what the main window pushes when a
    source appears; the tick is the checkbox itself."""
    page, explorer = dist
    _type_dist_words(page)
    words = _dist_words(page)
    page.set_data({"probe": _dist_frame(), "other": _dist_frame(1),
                   "third": _dist_frame(2).drop(columns=["dwell_s"])})
    page.statusBar().clearMessage()

    def tick(key):
        row = next(i for i in range(page._source_list.count())
                   if page._source_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == key)
        page._source_list.item(row).setCheckState(QtCore.Qt.CheckState.Checked)

    tick("other")                                               # it has dwell_s: nothing moves
    _click_button(page._plot_btn)
    assert page._hist_col.currentText() == "dwell_s"
    assert _dist_words(page) == words and _said(page, explorer)[0] == ""

    depth = page._history.depth()[0]
    tick("third")                                               # it lacks dwell_s: the column moves
    columns = [page._hist_col.itemText(i) for i in range(page._hist_col.count())]
    assert "dwell_s" not in columns and page._hist_col.currentText() == "xcol"
    assert _dist_words(page) == ("", "", "", "")
    assert _said(page, explorer) == (
        "Title, X label, Y label and fit label wording reset: the plot is now set to show the "
        "histogram of xcol, fitted with exponential.", True)
    assert page._history.depth()[0] == depth + 1                # the tick and the reset: one step
    _click_button(page._plot_btn)
    assert _drawn_title(page._plot)[0] == ""


def test_distributions_with_nothing_typed_says_nothing(dist):
    page, explorer = dist
    page.statusBar().clearMessage()
    _user_picks(page._plot_type_combo, "Scatter")
    assert _said(page, explorer)[0] == ""


def test_distributions_words_saved_over_another_plot_open_unchanged(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save, reopened by ``_open_project``: the next change of plot type resets."""
    window = _series_window(qtbot, tmp_path, monkeypatch)
    window._open_plots()
    page = window._plots_window
    explorer = _explorer_of(window)
    page.activateWindow()
    assert page.sources(), "the main window offers the detections as Distributions' sources"
    _click_button(page._plot_btn)
    _user_picks(page._plot_type_combo, "Scatter")
    _click_button(page._plot_btn)
    _add_title(page._plot, "Depth against dwell")
    _edit_double_click(page._plot, page._plot.getAxis("bottom").label, "Dwell")
    assert _dist_words(page)[:3] == ("Depth against dwell", "Dwell", "")
    project = tmp_path / "distributions.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_plots()
    page = reopened._plots_window
    explorer = _explorer_of(reopened)
    page.activateWindow()
    assert page._plot_type_combo.currentText() == "Scatter"
    assert _dist_words(page)[:3] == ("Depth against dwell", "Dwell", "")
    assert _drawn_title(page._plot)[0] == "Depth against dwell"
    assert _said(page, explorer)[0] == ""
    _user_picks(page._plot_type_combo, "Scatter")          # the same plot: nothing
    assert _dist_words(page)[:3] == ("Depth against dwell", "Dwell", "")
    _user_picks(page._plot_type_combo, "Histogram")        # the next one resets
    assert _dist_words(page) == ("", "", "", "")
    assert _said(page, explorer)[0].startswith("Title and X label reset: the plot is now set to show")
    reopened._dirty = False


# -- the module itself ------------------------------------------------------------------------------

def test_quantity_reset_stays_a_leaf_of_pure_functions():
    """The reset rule's module imports no gui module and no Qt: windows may import it freely
    (the report window's import guard allows it for that reason)."""
    import ast
    import pathlib

    import nanoscope.gui.quantity_reset as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    imported = {n.module.split(".")[0] if n.module else "" for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.level == 0}
    imported |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                 for a in n.names}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level],\
        "no sibling import"
    assert imported <= {"__future__", "dataclasses", "typing"}, imported


# -- the public setters are the control's route; the first draw is no change ----------------------

def test_the_public_setters_reset_like_the_controls_that_they_move(
        events, noise, subpop, dist):
    """A setter moves the control and the control's signal does the rest (a restore and a
    template block that signal, so they reset nothing; the tests above)."""
    page, _explorer = events
    widget = page._appearance_widget
    _type_event_words(page)
    page.set_x_metric("frac_depth")
    assert widget.state().title is None and page._x_combo.currentData() == "frac_depth"
    _type_event_words(page)
    page.set_view("Ensemble")
    assert widget.state().title is None and page.current_view() == "Ensemble"

    page, _explorer = noise
    widget = page._appearance_widget
    _type_noise_words(page)
    page.set_view("spectrogram")
    assert widget.state().title is None and page.current_view() == "spectrogram"

    page, _explorer = subpop
    widget = page._appearance_widget
    _type_subpop_words(page)
    page.embedding_view().set_embedding("UMAP")
    assert widget.state().title is None and page.embedding_view().embedding() == "UMAP"

    page, _explorer = dist
    _type_dist_words(page)
    page.set_plot_type("Scatter")
    assert _dist_words(page) == ("", "", "", "")
    page.set_plot_type("Histogram")
    page.set_histogram_column("dwell_s")
    page.set_fit("exponential")
    _edit_double_click(page._plot, page._fit_label_item, "Exp: {equation}")
    page.set_fit("gamma (peaked)")                         # a setter commits at once
    assert _dist_words(page)[3] == ""


def test_the_first_draw_keeps_words_typed_before_it(qapp, qtbot, tmp_path, monkeypatch):
    """A figure that has not been drawn shows no quantity yet, so nothing it holds is stale."""
    from nanoscope.core.session import Session
    from nanoscope.gui.event_plot_panel import EventPlotWindow
    from nanoscope.gui.voltage_dep_panel import VDepWindow

    from .test_event_plot import _doc_and_events, _install_detection

    # Event waveforms: words typed on an empty window survive the events that draw it.
    session = Session()
    page = EventPlotWindow(session)
    qtbot.addWidget(page)
    explorer = _in_explorer(qtbot, page, PAGE_EVENT_WAVEFORMS)
    widget = page._appearance_widget
    widget.apply_appearance(PlotAppearance(title="Before any events", x_label="Dwell"))
    doc, ev = _doc_and_events(tmp_path, name="first.dat")
    doc_id = session.add_document(doc)
    _install_detection(session, doc_id, ev)
    page.set_source(doc_id, 0, ev)                         # the first draw
    assert page.density_counts() is not None
    assert (widget.state().title, widget.state().x_label) == ("Before any events", "Dwell")
    assert _said(page, explorer)[0] == ""

    # Voltage dependence: no recordings, so no points; picking a metric draws nothing either.
    empty = VDepWindow(Session())
    qtbot.addWidget(empty)
    explorer = _in_explorer(qtbot, empty, PAGE_VOLTAGE_SERIES)
    empty._appearance_widget.apply_appearance(PlotAppearance(title="Nothing yet"))
    empty.set_metric("median_depth")
    assert empty._appearance_widget.state().title == "Nothing yet"
    assert _said(empty, explorer)[0] == ""


def test_summaries_words_typed_before_the_first_generate_survive_it(
        qapp, qtbot, tmp_path, monkeypatch):
    window = _series_window(qtbot, tmp_path, monkeypatch)
    window._open_distribution_summaries()
    page = window._distribution_summary_window
    explorer = _explorer_of(window)
    page.activateWindow()
    assert not page.has_data()
    _section(page, "Appearance")
    _type_in_box(page._appearance_widget.title, "My title")
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)                                        # the first draw
    assert page.has_data() and page._appearance_widget.state().title == "My title"
    assert _drawn_title(page._plot)[0] == "My title" and _said(page, explorer)[0] == ""
    window._dirty = False


# -- an empty plot shows no quantity; the placeholders follow what would be drawn ------------------

def test_voltage_dependence_an_empty_plot_shows_no_quantity_and_a_hidden_title_counts(vdep):
    """Once the points are gone the plot shows a message, not a metric: words typed over it are
    not stale. A title hidden with a single space is the same field and goes with the others."""
    page, explorer = vdep
    widget = page._appearance_widget
    widget.apply_appearance(replace(widget.state(), title=" ", y_label="Rate"))   # hidden + typed
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    assert (widget.state().title, widget.state().y_label) == (None, None)
    assert _said(page, explorer)[0] == "Title and Y label reset: the plot now shows median depth."
    # No recordings left: the plot says so, and shows no metric.
    for doc in list(page._session.documents()):
        page._session.close_document(doc.doc_id)
    page.refresh()
    assert page.points() == [] and page._message_item is not None
    widget.apply_appearance(replace(widget.state(), title="Typed over the message"))
    _clear_status(page)
    _pick(page._metric_combo, text="Event count")
    assert widget.state().title == "Typed over the message" and _said(page, explorer)[0] == ""


def test_voltage_dependence_placeholders_name_the_axis_labels_that_would_be_drawn(vdep):
    page, _explorer = vdep
    widget = page._appearance_widget
    assert widget.x_label.placeholderText() == "Automatic: Applied voltage (mV)"
    assert widget.y_label.placeholderText() == "Automatic: Capture rate (Hz)"
    _section(page, "Analysis")
    _click_button(page._abs_v)
    _click_button(page._log_y)
    assert widget.x_label.placeholderText() == "Automatic: Applied |voltage| (mV)"
    assert widget.y_label.placeholderText() == "Automatic: Capture rate (Hz) (log)"
    plot = page.plot_item()
    assert _label(plot, "bottom") == "Applied |voltage| (mV)"           # what the boxes promise
    assert _label(plot, "left") == "Capture rate (Hz) (log)"


# -- a restore that shows another view is what the next switch is measured from ------------------

def test_event_waveforms_a_restore_onto_another_tab_is_where_the_next_switch_starts(events):
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = events
    widget = page._appearance_widget
    bar = page._tabs.tabBar()
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())     # the Ensemble tab
    _type_event_words(page, plot=page._ensemble_plot)
    artifact = workspace.capture_event_plots(page)                       # saved on the ensemble
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())     # back on the heatmap
    assert widget.state().title is None
    _clear_status(page)
    assert workspace.restore_event_plots(page, artifact).restored       # the tab is set, blocked
    assert page.current_view() == "Ensemble" and widget.state().title == "Dwell vs depth"
    assert _said(page, explorer)[0] == ""                                # the restore reset nothing
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())     # off the restored tab
    assert page.current_view() == "Density" and widget.state().title is None
    assert _said(page, explorer)[0].endswith("the plot now shows the event-count heatmap.")


def test_noise_a_restore_onto_another_view_is_where_the_next_pick_starts(noise):
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = noise
    widget = page._appearance_widget
    _section(page, "Data")
    _pick(page._view_combo, data="spectrogram")
    _edit_double_click(page._spec_plot, page._spec_plot.titleLabel, "Spectrogram of pore A")
    artifact = workspace.capture_psd(page)                               # saved on the spectrogram
    _pick(page._view_combo, data="psd")
    assert widget.state().title is None
    _clear_status(page)
    assert workspace.restore_psd(page, artifact).restored               # the view is set, silently
    assert page.current_view() == "spectrogram" and widget.state().title == "Spectrogram of pore A"
    assert _said(page, explorer)[0] == ""                                # the restore reset nothing
    _pick(page._view_combo, data="psd")                                  # off the restored view
    assert widget.state().title is None
    assert _said(page, explorer) == ("Title reset: the plot now shows the Welch PSD.", True)


# -- Comparison figures: a restored snapshot is the quantity the next Generate is measured from -----

def test_comparison_a_restored_snapshot_is_where_the_next_generate_starts(comparison, qtbot):
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow

    from .test_comparison_figures_panel import source

    page, _explorer, drawn = comparison
    _type_comparison_words(page)
    payload, tables = page.snapshot_state()
    fresh = ComparisonFiguresWindow()
    qtbot.addWidget(fresh)
    _in_explorer(qtbot, fresh, PAGE_COMPARISON_FIGURES)
    fresh.set_sources([source(), source("second", "Recording B")],
                      selected_keys=["first", "second"])
    fresh.restore_state(payload, tables)
    fresh.activateWindow()
    widget = fresh._appearance_widget
    assert (drawn[-1]["title"], drawn[-1]["x"], drawn[-1]["y"]) == ("Mine", ["Dw"], ["Dp"])
    assert widget.state().title == "Mine" and not fresh._status.text().startswith("Title")
    _generate(fresh)                                       # the quantity the snapshot shows
    assert widget.state().title == "Mine" and drawn[-1]["title"] == "Mine"
    _section(fresh, "Analysis")
    _pick(fresh._y_combo, data="snr")
    _generate(fresh)
    assert widget.state().title is None and drawn[-1]["title"] == "Recording comparisons"
    assert fresh._status.text().startswith("Title, X label and Y label reset:")


# -- an empty screen shows no quantity, and a restore shows exactly what it restored ---------------

def test_event_waveforms_an_empty_screen_has_no_quantity_to_leave(events):
    """With no events to draw the window says so and shows no metric and no ensemble, so words
    typed over it are not stale when the next X, Y or Normalize time comes. (A recording that
    lost its detection is another recording: the words stay, decision 2.)"""
    page, explorer = events
    widget = page._appearance_widget
    bar = page._tabs.tabBar()
    doc_id, channel, eventset = page._doc_id, page._channel, page._eventset
    _type_event_words(page)                                              # the heatmap
    page.set_source(None, None, None)
    assert page.density_counts() is None and widget.state().title == "Dwell vs depth"
    steps = widget._history.depth()
    _section(page, "Analysis")
    _clear_status(page)
    _pick(page._x_combo, data="frac_depth")
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""
    assert widget.state().title == "Dwell vs depth" and _boxes(widget)[1:] == ("Dwell", "Depth")

    page.set_source(doc_id, channel, eventset)                           # the ensemble
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())
    _type_event_words(page, plot=page._ensemble_plot)
    page.set_source(None, None, None)
    assert page.ensemble_traces() is None and widget.state().title == "Dwell vs depth"
    steps = widget._history.depth()
    _section(page, "Analysis")
    _clear_status(page)
    _click_button(page._normalize)
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""
    assert widget.state().title == "Dwell vs depth"


def test_event_waveforms_a_restored_snapshot_is_what_the_next_change_is_measured_from(events):
    """A restore draws saved arrays and says so to the trackers: the ensemble it shows is the
    one Normalize time is measured against, and a snapshot with nothing in it leaves no
    quantity on screen (a stale one would wipe the saved words at the next pick)."""
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = events
    widget = page._appearance_widget
    bar = page._tabs.tabBar()
    doc_id, channel, eventset = page._doc_id, page._channel, page._eventset
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())     # the Ensemble tab
    _type_event_words(page, plot=page._ensemble_plot)
    artifact = workspace.capture_event_plots(page)
    assert workspace.restore_event_plots(page, artifact).restored       # drawn from the snapshot
    assert page.current_view() == "Ensemble" and widget.state().title == "Dwell vs depth"
    _section(page, "Analysis")
    _clear_status(page)
    _click_button(page._normalize)                                      # off the restored picture
    assert widget.state().title is None and _said(page, explorer)[0].endswith(
        "the plot now shows the ensemble against time in seconds.")

    # A snapshot saved with nothing to show, restored over a window that shows an ensemble ...
    _type_event_words(page, plot=page._ensemble_plot)
    page.set_source(None, None, None)
    empty_ensemble = workspace.capture_event_plots(page)
    page.set_source(doc_id, channel, eventset)
    assert page.ensemble_traces() is not None
    assert workspace.restore_event_plots(page, empty_ensemble).restored
    assert page.ensemble_traces() is None and widget.state().title == "Dwell vs depth"
    steps = widget._history.depth()
    _section(page, "Analysis")
    _clear_status(page)
    _click_button(page._normalize)
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""
    assert widget.state().title == "Dwell vs depth"

    # ... and over a window that shows a heatmap.
    page.set_source(doc_id, channel, eventset)
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())
    _type_event_words(page)
    page.set_source(None, None, None)
    empty_heatmap = workspace.capture_event_plots(page)
    page.set_source(doc_id, channel, eventset)
    assert page.density_counts() is not None
    assert workspace.restore_event_plots(page, empty_heatmap).restored
    assert page.density_counts() is None and widget.state().title == "Dwell vs depth"
    steps = widget._history.depth()
    _section(page, "Analysis")
    _clear_status(page)
    _pick(page._x_combo, data="frac_depth")
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""
    assert widget.state().title == "Dwell vs depth"


def test_noise_an_empty_screen_has_no_quantity_to_leave(noise):
    page, explorer = noise
    widget = page._appearance_widget
    _type_noise_words(page)
    for doc in list(page._session.documents()):
        page._session.close_document(doc.doc_id)
    page.refresh()
    assert not page.has_data()
    steps = widget._history.depth()
    _section(page, "Analysis")
    _clear_status(page)

    _click_button(page._asd_check)                                      # no spectrum to leave

    assert page.is_asd() and widget._history.depth() == steps and _said(page, explorer)[0] == ""
    assert widget.state().title == "PSD, pore A"


def test_noise_a_window_nobody_has_fed_resets_for_a_view_and_says_nothing_without_words(
        qapp, qtbot):
    from nanoscope.core.session import Session
    from nanoscope.gui.psd_panel import PSDWindow

    page = PSDWindow(Session())
    qtbot.addWidget(page)
    explorer = _in_explorer(qtbot, page, PAGE_NOISE)
    page.activateWindow()
    assert not page.has_data() and page.current_view() == "psd"
    _section(page, "Data")
    _pick(page._view_combo, data="spectrogram")
    assert page.current_view() == "spectrogram" and _said(page, explorer)[0] == ""   # nothing typed
    _pick(page._view_combo, data="psd")
    page.set_asd(True)
    assert page.is_asd() and _said(page, explorer)[0] == ""


def test_windows_without_data_show_no_quantity_and_carry_no_note(qapp, qtbot):
    """Before a report or a recording there is nothing on screen to leave: the trackers say so
    (and the standing line has no sentence to keep)."""
    from nanoscope.core.session import Session
    from nanoscope.gui.subpop_window import SubpopReportWindow
    from nanoscope.gui.voltage_dep_panel import VDepWindow

    report = SubpopReportWindow()
    qtbot.addWidget(report)
    assert report._embedding_shown == "" and report._words_note_text == ""
    assert report.embedding_view().embedding() == ""
    vdep = VDepWindow(Session())
    qtbot.addWidget(vdep)
    assert vdep._drawn_metric is None and vdep.points() == []


# -- the sentence on the status line belongs to the plot it describes -------------------------------

def test_voltage_dependence_a_later_change_that_resets_nothing_takes_the_sentence_down(vdep):
    """The sentence says what the plot now shows. When the metric changes again with nothing
    typed it would name the wrong metric for the rest of its ten seconds, so it goes; a message
    that is not the sentence is never touched."""
    page, explorer = vdep
    plot = page.plot_item()
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    assert _said(page, explorer) == (
        "Title and Y label reset: the plot now shows median depth.", True)

    _pick(page._metric_combo, text="Event count")              # nothing typed any more
    assert _label(plot, "left") == "Event count"
    assert _said(page, explorer)[0] == ""

    _add_title(plot, "Counts")
    _pick(page._metric_combo, text="Median dwell (s)")
    assert _said(page, explorer)[0] == "Title reset: the plot now shows median dwell."
    page.statusBar().showMessage("Voltage dependence has a log scale on the y axis only.", 5000)
    _pick(page._metric_combo, text="Median dI/I₀")              # not our message: it stays
    assert _said(page, explorer)[0] == "Voltage dependence has a log scale on the y axis only."


def test_event_waveforms_a_later_change_that_resets_nothing_takes_the_sentence_down(events):
    page, explorer = events
    _type_event_words(page)
    _section(page, "Analysis")
    _pick(page._x_combo, data="frac_depth")
    assert _said(page, explorer)[0].startswith("Title, X label and Y label reset:")
    _pick(page._y_combo, data="dwell_s")                       # nothing typed any more
    assert (_label(page._density_main, "bottom"), _label(page._density_main, "left")) == (
        "dI/I₀ (log10)", "Dwell (s)")
    assert _said(page, explorer)[0] == ""
    # The tabs and Normalize time are quantities too.
    bar = page._tabs.tabBar()
    _edit_double_click(page._density_main, page._density_main.titleLabel, "Dwell vs depth")
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())
    assert _said(page, explorer)[0] == "Title reset: the plot now shows the ensemble."
    _section(page, "Analysis")
    _click_button(page._normalize)
    assert _said(page, explorer)[0] == ""
    _edit_double_click(page._ensemble_plot, page._ensemble_plot.titleLabel, "Mean event")
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())
    assert _said(page, explorer)[0].endswith("the plot now shows the event-count heatmap.")
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())     # nothing typed any more
    assert _said(page, explorer)[0] == ""


def test_noise_a_later_change_that_resets_nothing_takes_the_sentence_down(noise):
    page, explorer = noise
    _type_noise_words(page)
    _section(page, "Data")
    _pick(page._view_combo, data="spectrogram")
    assert _said(page, explorer)[0].endswith("the plot now shows the spectrogram.")
    _pick(page._view_combo, data="fft")                        # nothing typed any more
    assert page.current_view() == "fft"
    assert _said(page, explorer)[0] == ""
    _pick(page._view_combo, data="psd")
    _edit_double_click(page._plot, page._plot.titleLabel, "PSD again")
    _section(page, "Analysis")
    _click_button(page._asd_check)
    assert _said(page, explorer)[0].endswith("the plot now shows the amplitude spectral density.")
    _click_button(page._asd_check)                             # nothing typed any more
    assert _said(page, explorer)[0] == ""


def test_summaries_a_later_commit_that_resets_nothing_takes_the_sentence_down(summaries):
    window, page, explorer = summaries
    _type_summary_words(page)
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)
    sentence = _said(page, explorer)[0]
    assert sentence.endswith("the plot now shows the ECDF of median depth.")
    _pick(page._metric_combo, data="frac_depth")               # a draft: the figure is still depth
    assert _said(page, explorer)[0] == sentence
    _generate(page)                                            # the commit; nothing typed any more
    assert page._applied_config["metric"] == "frac_depth"
    assert _said(page, explorer)[0] == ""


def test_distributions_a_later_change_that_resets_nothing_takes_the_sentence_down(dist):
    page, explorer = dist
    _type_dist_words(page)
    _pick(page._plot_type_combo, text="Scatter")
    sentence = _said(page, explorer)[0]
    assert sentence.endswith("the plot is now set to show the scatter of ycol against xcol.")
    _click_button(page._plot_btn)               # the picture catches up: the sentence is still true
    assert _said(page, explorer)[0] == sentence
    _pick(page._plot_type_combo, text="Histogram")             # nothing typed any more
    assert _dist_words(page) == ("", "", "", "")
    assert _said(page, explorer)[0] == ""
    # Another message of the window's is not ours to take down.
    _type_dist_words(page)
    _pick(page._plot_type_combo, text="Scatter")
    assert _said(page, explorer)[0] == sentence
    page.statusBar().showMessage("Lasso selection cancelled; selection is unchanged.", 6000)
    _pick(page._plot_type_combo, text="Histogram")
    assert _said(page, explorer)[0] == "Lasso selection cancelled; selection is unchanged."

    # A column and the applied fit are quantities too; the first fit after none is not a change.
    _type_dist_words(page)
    _user_picks(page._hist_col, "frac_depth")
    assert _said(page, explorer)[0].endswith("the histogram of frac_depth, fitted with exponential.")
    _user_picks(page._hist_col, "depth_pA")                    # nothing typed any more
    assert _said(page, explorer)[0] == ""
    _type_dist_words(page)
    _pick(page._plot_type_combo, text="Scatter")
    sentence = _said(page, explorer)[0]
    assert sentence.endswith("scatter of ycol against xcol.")
    _user_picks(page._scatter_fit, "linear")
    _click_button(page._scatter_fit_apply_btn)                 # none -> linear: resets nothing
    assert page._applied_scatter_fit == "linear" and _said(page, explorer)[0] == sentence
    _user_picks(page._scatter_fit, "power law (log-log)")
    _click_button(page._scatter_fit_apply_btn)                 # linear -> power law: nothing typed
    assert page._applied_scatter_fit == "power law (log-log)" and _said(page, explorer)[0] == ""


def test_distributions_a_typed_range_that_goes_does_not_keep_the_sentence(dist):
    """The rule may find a typed range to clear and no words: the earlier sentence goes all the
    same, and the one that replaces it names the range (G5; it used to leave the line empty)."""
    page, explorer = dist
    _type_dist_words(page)
    _pick(page._plot_type_combo, text="Scatter")
    assert _said(page, explorer)[0].endswith("scatter of ycol against xcol.")
    _pin_through_axis_properties(page._plot, "bottom", 1.0, 3.0)
    assert page.config().axes.x_min == "1"

    _pick(page._plot_type_combo, text="Histogram")

    assert page.config().axes.x_min == "" and _dist_words(page) == ("", "", "", "")
    assert _said(page, explorer)[0] == (
        "X range reset: the plot is now set to show the histogram of dwell_s, fitted with "
        "exponential.")


def test_comparison_and_subpopulations_replace_their_line_on_every_change(comparison, subpop):
    page, explorer, drawn = comparison
    _type_comparison_words(page)
    _section(page, "Analysis")
    _pick(page._y_combo, data="amplitude_pA")
    _generate(page)
    assert page._status.text().startswith("Title, X label and Y label reset:")
    _pick(page._y_combo, data="snr")
    _generate(page)                                            # nothing typed any more
    assert page._status.text().startswith("Generated snapshot")

    page, explorer = subpop
    scatter = page.embedding_view()
    _type_subpop_words(page)
    _section(page, "Analysis")
    _pick(scatter._embed_combo, text="UMAP")
    assert _standing_line(page, explorer)[0].startswith("Title, X label and Y label reset:")
    _pick(scatter._embed_combo, text="t-SNE")                  # nothing typed any more
    assert _standing_line(page, explorer)[0] == ""


# -- getting back, templates and hidden labels: what the guide says is what happens ------------------

def _vdep_reset_to_depth(vdep):
    """Words typed over capture rate, then the metric picked: the reset has happened."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    assert widget.state().title is None
    return page, explorer, plot, widget


def test_voltage_dependence_to_go_back_pick_the_earlier_metric_and_then_undo(vdep):
    """Undo brings the words back over the metric that is on screen, not the metric. Picking the
    earlier metric first records no step (nothing is typed any more) and takes the old sentence
    down; the Undo after it puts the words back over that metric."""
    page, explorer, plot, widget = _vdep_reset_to_depth(vdep)
    steps = widget._history.depth()

    _pick(page._metric_combo, text="Capture rate (Hz)")

    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""
    assert widget.undo()
    assert page._metric_combo.currentText() == "Capture rate (Hz)"
    assert (_drawn_title(plot)[0], _label(plot, "left")) == ("Capture vs voltage", "Rate (Hz)")


def test_voltage_dependence_undo_first_and_the_pick_after_it_clears_the_words_again(vdep):
    """The other order is a trap: Undo puts the words over the depth plot, and picking the
    capture rate is then a change of metric, which takes them back to automatic (a new step)."""
    page, explorer, plot, widget = _vdep_reset_to_depth(vdep)

    assert widget.undo() and widget.state().title == "Capture vs voltage"
    assert page._metric_combo.currentText() == "Median depth (pA)"
    assert _label(plot, "left") == "Rate (Hz)"                 # the words, over the depth plot
    steps = widget._history.depth()
    _pick(page._metric_combo, text="Capture rate (Hz)")

    assert widget.state().title is None and _label(plot, "left") == "Capture rate (Hz)"
    assert widget._history.depth() == (steps[0] + 1, 0)        # a new step: the redo is gone
    assert _said(page, explorer)[0] == "Title and Y label reset: the plot now shows capture rate."
    assert widget.undo() and widget.state().title == "Capture vs voltage"


def test_voltage_dependence_a_template_installs_its_own_words_and_the_rule_does_not_run(vdep):
    """A recalled template is one step of the plot's appearance Undo that installs the title
    and labels it was saved with -- none, if it was saved without -- replacing the typed ones."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Typed before the recall")
    _edit_double_click(plot, plot.getAxis("left").label, "Mine (Hz)")
    settings, _appearance = page.template_settings()
    steps = widget._history.depth()
    _clear_status(page)

    plain = replace(widget.state(), title=None, x_label=None, y_label=None).to_dict()
    page.apply_template_settings({**settings, "metric": "median_depth"}, plain)
    assert page._metric_combo.currentText() == "Median depth (pA)"
    assert widget.state().title is None and _drawn_title(plot)[0] == ""
    assert _label(plot, "left") == "Median depth (pA)"
    assert widget._history.depth() == (steps[0] + 1, 0) and _said(page, explorer)[0] == ""
    assert widget.undo() and widget.state().title == "Typed before the recall"

    worded = replace(widget.state(), title="From the template", y_label="Depth (pA)").to_dict()
    page.apply_template_settings(settings, worded)
    assert _drawn_title(plot)[0] == "From the template" and _label(plot, "left") == "Depth (pA)"
    assert _said(page, explorer)[0] == ""


def test_distributions_a_template_installs_its_own_words_and_the_rule_does_not_run(dist):
    page, explorer = dist
    plain = page.config()                                      # a template saved with no words
    _type_dist_words(page)
    worded = page.config()
    page.statusBar().clearMessage()

    page.apply_template(plain)
    assert _dist_words(page) == ("", "", "", "") and _said(page, explorer)[0] == ""
    page.apply_template(worded)
    assert _dist_words(page) == ("Dwell, pore A", "Dwell time (ms)", "Events per bin",
                                 "Single exponential: {equation}, {n}")
    assert _said(page, explorer)[0] == ""


def _hide_title(plot):
    """Right-click the title, Hide title."""
    controller = figure_text_for(plot)
    controller.last_menu = None
    _click(plot, plot.titleLabel.sceneBoundingRect().center())
    _choose(controller.last_menu, "Hide title")


def test_a_hidden_title_counts_as_typed_and_comes_back_where_the_window_draws_one(events):
    """Hide title is a single space in the Title box: the same field, so a change of quantity
    puts it back to automatic, with a step and a sentence, as for any typed title."""
    page, explorer = events
    plot, widget = page._density_main, page._appearance_widget
    _edit_double_click(plot, plot.titleLabel, "Dwell vs depth")
    _hide_title(plot)
    assert widget.state().title == " " and _drawn_title(plot)[0] == ""
    steps = widget._history.depth()
    _section(page, "Analysis")
    _clear_status(page)

    _pick(page._y_combo, data="frac_depth")

    assert widget.state().title is None and widget._history.depth() == (steps[0] + 1, 0)
    assert _drawn_title(plot)[0] == _squash(page._density_auto_title) != ""
    assert _said(page, explorer) == (
        "Title reset: the plot now shows dI/I₀ against dwell.", True)
    assert widget.undo() and widget.state().title == " " and _drawn_title(plot)[0] == ""


def test_a_hidden_title_in_a_window_with_no_automatic_title_changes_nothing_you_can_see(vdep):
    """Voltage dependence draws no automatic title, so a hidden one coming back shows nothing;
    a hidden axis label does come back, as the automatic label of the new metric."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _section(page, "Appearance")
    _type_in_box(widget.title, " ")
    QtTest.QTest.qWait(800)                                    # the typing is settled as one step
    assert widget.state().title == " " and _drawn_title(plot)[0] == ""
    steps = widget._history.depth()
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    assert widget.state().title is None and _drawn_title(plot)[0] == ""
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer)[0] == "Title reset: the plot now shows median depth."

    _section(page, "Appearance")
    _type_in_box(widget.y_label, " ")
    QtTest.QTest.qWait(800)
    assert widget.state().y_label == " " and _label(plot, "left") == ""
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Event count")
    assert widget.state().y_label is None and _label(plot, "left") == "Event count"
    assert _said(page, explorer)[0] == "Y label reset: the plot now shows event count."


def test_subpopulations_a_hidden_title_is_reset_with_nothing_new_to_see(subpop):
    """Subpopulations draws no automatic title on either figure: the step and the sentence are
    the only sign that a hidden title was put back to automatic."""
    page, explorer = subpop
    scatter, waves = page.embedding_view(), page.waveform_view()
    plot, widget = scatter._plot.getPlotItem(), page._appearance_widget
    _section(page, "Appearance")
    _type_in_box(widget.title, " ")
    QtTest.QTest.qWait(800)
    assert widget.state().title == " "
    steps = widget._history.depth()
    _section(page, "Analysis")
    _pick(scatter._embed_combo, text="UMAP")
    assert widget.state().title is None and widget._history.depth() == (steps[0] + 1, 0)
    assert _drawn_title(plot)[0] == "" and _drawn_title(waves._plot.getPlotItem())[0] == ""
    assert _standing_line(page, explorer)[0] == "Title reset: the plot now shows the UMAP embedding."


def test_summaries_to_go_back_pick_the_earlier_measurement_generate_and_then_undo(summaries):
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    _type_summary_words(page)
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)
    assert widget.state().title is None
    steps = widget._history.depth()

    _pick(page._metric_combo, data="dwell_s")                  # the earlier measurement
    _generate(page)

    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""
    assert widget.undo()
    assert page._applied_config["metric"] == "dwell_s"
    assert _drawn_title(plot)[0] == "Dwell ECDF" and _label(plot, "bottom") == "Dwell time"


def test_comparison_to_go_back_pick_the_earlier_measurement_generate_and_then_undo(comparison):
    page, explorer, drawn = comparison
    widget = page._appearance_widget
    _type_comparison_words(page)
    _section(page, "Analysis")
    _pick(page._y_combo, data="amplitude_pA")
    _generate(page)
    assert widget.state().title is None
    steps = widget._history.depth()
    earlier = "depth_pA" if page._applied_config["y"] != "depth_pA" else "snr"

    _pick(page._y_combo, data=earlier)
    _generate(page)

    assert widget._history.depth() == steps and not page._status.text().startswith("Title")
    assert widget.undo()
    assert (drawn[-1]["title"], drawn[-1]["x"], drawn[-1]["y"]) == ("Mine", ["Dw"], ["Dp"])


def test_event_waveforms_to_go_back_click_the_earlier_tab_and_then_undo(events):
    page, explorer = events
    widget = page._appearance_widget
    bar = page._tabs.tabBar()
    _type_event_words(page)
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())     # the Ensemble tab
    assert widget.state().title is None
    steps = widget._history.depth()

    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())     # back to the heatmap
    assert page.current_view() == "Density"
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""
    assert widget.undo()
    assert _drawn_title(page._density_main)[0] == "Dwell vs depth"
    assert _label(page._density_main, "bottom") == "Dwell"


def test_event_waveforms_a_template_without_words_replaces_the_typed_ones(events):
    page, explorer = events
    widget = page._appearance_widget
    _type_event_words(page)
    settings, _appearance = page.template_settings()
    steps = widget._history.depth()
    _clear_status(page)

    plain = replace(widget.state(), title=None, x_label=None, y_label=None).to_dict()
    page.apply_template_settings(settings, plain)

    assert _boxes(widget) == ("", "", "") and widget._history.depth() == (steps[0] + 1, 0)
    assert _drawn_title(page._density_main)[0] == _squash(page._density_auto_title)
    assert _said(page, explorer)[0] == ""


# -- Voltage dependence: the applied trend ------------------------------------------------------------

def test_voltage_dependence_applying_another_trend_resets_only_the_fit_wording(vdep):
    """Distributions takes the fit label's wording back when another model is applied; so does
    Voltage dependence's own trend. Before, wording typed for the linear fit stayed over the
    exponential one. The title, the labels and the fit label's look stay."""
    page, explorer = vdep                                      # the linear trend is applied
    plot, widget = page.plot_item(), page._appearance_widget
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, page._fit_label, "Linear fit: {equation}")
    widget.fit_label_size.setValue(9.0)
    QtTest.QTest.qWait(800)
    assert page.fit_label_text().startswith("Linear fit:")
    steps = widget._history.depth()
    _section(page, "Analysis")
    _user_picks(page._fit_combo, "exponential")                # a proposal: nothing redraws
    assert page._fit_apply_btn.text() == "Apply fit ●"
    assert page.fit_label_text().startswith("Linear fit:") and widget._history.depth() == steps
    assert _said(page, explorer)[0] == ""

    _click_button(page._fit_apply_btn)

    assert page._applied_fit == "exponential"
    assert widget.state().fit_label_state().template == "" and widget.fit_label_template.text() == ""
    assert page.fit_label_text() and not page.fit_label_text().startswith("Linear fit:")
    assert widget.state().title == "Capture vs voltage"
    assert widget.state().fit_label_state().size_pt == 9.0
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert _said(page, explorer) == (
        "Fit label wording reset: the plot now shows the exponential fit.", True)
    assert widget.undo()                                       # the wording, over the new trend
    assert page.fit_label_text().startswith("Linear fit:") and page._applied_fit == "exponential"
    assert widget.redo()

    # Taking the fit away takes the wording with it ...
    _edit_double_click(plot, page._fit_label, "Exponential: {equation}")
    _user_picks(page._fit_combo, "none")
    _click_button(page._fit_apply_btn)
    assert page._applied_fit == "none" and widget.state().fit_label_state().template == ""
    assert _said(page, explorer)[0] == "Fit label wording reset: the plot now shows no fit."
    # ... and wording typed while there is none stays through the first fit, as in Distributions.
    _section(page, "Appearance")
    _type_in_box(widget.fit_label_template, "Typed ahead: {equation}")
    _section(page, "Analysis")
    _clear_status(page)
    depth = widget._history.depth()
    page.set_fit("linear")
    assert widget.state().fit_label_state().template == "Typed ahead: {equation}"
    assert page.fit_label_text().startswith("Typed ahead:")
    assert widget._history.depth() == depth and _said(page, explorer)[0] == ""
    page.set_fit("linear")                                     # the same trend again: nothing
    assert widget.state().fit_label_state().template == "Typed ahead: {equation}"


def test_voltage_dependence_a_recalled_template_keeps_the_wording_that_came_with_its_fit(vdep):
    """The template's fit arrives selected and waits for Apply fit; the Apply that commits it is
    not a change of the trend the wording was written for. A restore sets the applied trend
    directly and installs the saved wording."""
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _edit_double_click(plot, page._fit_label, "Exponential trend: {equation}")
    settings, _appearance = page.template_settings()
    page.apply_template_settings({**settings, "selected_fit": "exponential"},
                                 page.appearance_state())
    assert page._fit_combo.currentText() == "exponential" and page._applied_fit == "linear"
    steps = widget._history.depth()
    _section(page, "Analysis")
    _click_button(page._fit_apply_btn)
    assert page._applied_fit == "exponential"
    assert widget.state().fit_label_state().template == "Exponential trend: {equation}"
    assert page.fit_label_text().startswith("Exponential trend:")
    assert widget._history.depth() == steps and _said(page, explorer)[0] == ""
    page.set_fit("linear")                                     # only the NEXT trend resets
    assert widget.state().fit_label_state().template == ""

    _edit_double_click(plot, page._fit_label, "Linear trend: {equation}")
    artifact = workspace.capture_voltage_dependence(page)
    page.set_fit("exponential")
    assert widget.state().fit_label_state().template == ""
    _clear_status(page)
    assert workspace.restore_voltage_dependence(page, artifact).restored
    assert page._applied_fit == "linear" and page.fit_label_text().startswith("Linear trend:")
    assert widget.state().fit_label_state().template == "Linear trend: {equation}"
    assert _said(page, explorer)[0] == ""
    # The restore also drops what a recalled template promised: the saved wording was written
    # for the saved trend, so the next trend resets it.
    page.apply_template_settings({**settings, "selected_fit": "exponential"},
                                 page.appearance_state())
    assert workspace.restore_voltage_dependence(page, artifact).restored
    page.set_fit("exponential")
    assert widget.state().fit_label_state().template == ""


def test_voltage_dependence_fit_wording_saved_with_a_trend_opens_unchanged(
        qapp, qtbot, tmp_path, monkeypatch):
    """One real save, reopened by ``_open_project``: the applied trend comes back with the
    wording typed for it, and the next trend resets the wording."""
    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    _install_series(window, ())
    window._open_vdep()
    page = window._vdep_window
    window._analysis_explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    page.set_fit("linear")
    _edit_double_click(page.plot_item(), page._fit_label, "Linear fit: {equation}")
    project = tmp_path / "trend.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_vdep()
    QtWidgets.QApplication.processEvents()
    page = reopened._vdep_window
    explorer = reopened._analysis_explorer
    explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    widget = page._appearance_widget
    assert page._applied_fit == "linear" and page.fit_label_text().startswith("Linear fit:")
    assert widget.state().fit_label_state().template == "Linear fit: {equation}"
    assert _said(page, explorer)[0] == ""
    page.set_fit("linear")                                     # the same trend: nothing
    assert widget.state().fit_label_state().template == "Linear fit: {equation}"
    page.set_fit("exponential")                                # the next one resets the wording
    assert widget.state().fit_label_state().template == ""
    assert _said(page, explorer) == (
        "Fit label wording reset: the plot now shows the exponential fit.", True)
    assert reopened._dirty, "the reset is an edit of the project"
    reopened._dirty = False


# -- Subpopulations: a restore resets nothing, and says so ---------------------------------------------

def test_subpopulations_a_restore_over_another_embedding_leaves_no_reset_sentence(subpop):
    from nanoscope.core import analysis_workspace as workspace

    page, explorer = subpop
    scatter = page.embedding_view()
    plot, widget = scatter._plot.getPlotItem(), page._appearance_widget
    _section(page, "Analysis")
    _pick(scatter._embed_combo, text="UMAP")
    _add_title(plot, "UMAP words")
    artifact = workspace.capture_subpopulation(page)
    _pick(scatter._embed_combo, text="t-SNE")
    _add_title(plot, "t-SNE words")

    assert workspace.restore_subpopulation(page, artifact).restored

    assert scatter.embedding() == "UMAP" and widget.state().title == "UMAP words"
    assert _drawn_title(plot)[0] == "UMAP words"
    assert _standing_line(page, explorer) == ("", True)
    assert widget._history.depth() == (0, 0)
    # The next embedding is measured from what the restore showed.
    _pick(scatter._embed_combo, text="t-SNE")
    assert widget.state().title is None
    assert _standing_line(page, explorer)[0] == "Title reset: the plot now shows the t-SNE embedding."


# -- pins for the details of the rule -----------------------------------------------------------------

def test_event_waveforms_a_template_that_flips_the_tab_or_normalize_time_resets_nothing(events):
    page, explorer = events
    widget = page._appearance_widget
    _type_event_words(page)
    words, steps = widget.state(), widget._history.depth()
    settings, _appearance = page.template_settings()
    assert settings["view"] == "Density" and settings["normalize_time"] is True
    _clear_status(page)

    page.apply_template_settings({**settings, "view": "Ensemble", "normalize_time": False},
                                 page.appearance_state())

    assert page.current_view() == "Ensemble" and not page._normalize.isChecked()
    assert widget.state() == words and widget._history.depth() == steps
    assert _said(page, explorer)[0] == ""
    # What the recall drew is where the next change starts from.
    bar = page._tabs.tabBar()
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(0).center())
    assert widget.state().title is None
    assert _said(page, explorer)[0].endswith("the plot now shows the event-count heatmap.")


def test_subpopulations_a_new_report_takes_the_reset_sentence_off_the_line(subpop, subpop_report):
    page, explorer = subpop
    _type_subpop_words(page)
    _section(page, "Analysis")
    _pick(page.embedding_view()._embed_combo, text="UMAP")
    assert _standing_line(page, explorer)[0].startswith("Title, X label and Y label reset:")
    page.set_report(subpop_report, subtitle="a second run")
    assert _standing_line(page, explorer)[0] == ""


def test_a_change_that_clears_nothing_leaves_an_open_run_of_edits_open(vdep):
    """Nothing typed: the pick does not touch the history at all, not even to close the run of
    edits a person has open (a size typed digit by digit stays one step)."""
    page, explorer = vdep
    widget = page._appearance_widget
    widget._burst.setInterval(60_000)                          # open for as long as this test needs
    steps = widget._history.depth()
    widget.title_size.setValue(15.0)
    assert widget._burst.isActive() and widget._history.depth() == steps
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    assert widget._burst.isActive() and widget._history.depth() == steps
    widget.title_size.setValue(16.0)
    widget.undo()                                              # closes the run: ONE step, 15 and 16
    assert widget.state().title_size != 16.0 and widget._history.depth() == (steps[0], 1)


def test_voltage_dependence_built_for_a_restore_has_drawn_no_metric_yet(qapp, qtbot):
    """A window built with its first refresh deferred (the project-restore path) shows nothing;
    a metric picked before anything is drawn has nothing to leave."""
    from nanoscope.core.session import Session
    from nanoscope.gui.voltage_dep_panel import VDepWindow

    page = VDepWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(page)
    assert page._drawn_metric is None
    page._appearance_widget.apply_appearance(PlotAppearance(title="Typed before any draw"))
    page._metric_combo.setCurrentText("Event count")           # the combo's own signal
    assert page._appearance_widget.state().title == "Typed before any draw"
    assert page._drawn_metric is None


# -- the guide -------------------------------------------------------------------------------------

def test_the_guide_states_the_rule_where_each_window_s_appearance_is_described():
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[2] / "docs" / "USER_GUIDE.md").read_text(
        encoding="utf-8")
    assert "**When a figure's quantity changes, the words you typed go back to automatic.**" in text
    for window in ("**Voltage dependence:** the **Metric**", "**Event waveforms:**", "**Noise:**",
                   "**Subpopulations:** the **Embedding**", "**I-V** has no such control",
                   "**Distributions:** the plot type, a column and the applied fit"):
        assert window in text, window
    assert "**What a change puts back to automatic**" in text                      # Distributions
    assert "**Generate** puts the typed\n   Title, X label and Y label back to automatic" in text
    assert "In Comparison figures, **Generate** puts the typed Title, X label and Y label" in text
    assert "Switching **Show** to another view" in text                          # Noise
    assert "Changing the heatmap's **X** or **Y**" in text                        # Event waveforms


def test_the_guide_says_what_the_rule_does_with_the_sentence_hidden_labels_undo_and_templates():
    """Each sentence is pinned by a behaviour test above: the sentence taken down, a hidden title
    counted as typed, the order of the pick and Undo, a template installing its own words, a
    project opening with the saved ones, and Voltage dependence's trend."""
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[2] / "docs" / "USER_GUIDE.md").read_text(
        encoding="utf-8")
    flat = " ".join(text.split())
    for sentence in (
            "A figure with no typed words records no step and says nothing",
            "that sentence is removed, because the figure has moved on since and it may name a "
            "quantity the plot no longer shows (any other message on the status bar stays)",
            "A title or label you hid, with **Hide title** or a single space in its box, counts "
            "as typed and goes back to automatic too: the window's automatic title or label "
            "comes back if it draws one",
            "Voltage dependence and Subpopulations draw no automatic title",
            "the applied fit, when **Apply fit** replaces one trend with another or takes the fit "
            "away, which puts back only the fit label's wording",
            "To return to the earlier quantity with its words, pick it first",
            "**Undo** first and the pick after is a new change, which clears the words again",
            "Opening a project and recalling a template do not run the rule",
            "a template installs the title, labels and wording it was saved with (none, if it was "
            "saved without) in place of the ones typed",
            "the **Apply fit** that commits it keeps the wording that came with it",
            "Recalling a template installs the title, labels and fit wording it was saved with",
            "takes that sentence off the status line; the first fit applied after *none* resets "
            "nothing and leaves it",
            "until a new report is shown, the embedding changes again or a saved figure is "
            "restored"):
        assert sentence in flat, sentence
    # What the first version of these paragraphs got wrong is gone.
    for gone in ("pick the metric, view or embedding again to return to it",
                 "recalling a template, applying a preset",
                 "Nothing typed means no step and no message",
                 "and recalling a template keep the words"):
        assert gone not in flat, gone
