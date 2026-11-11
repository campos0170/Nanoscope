"""Phase 0 follow-up: the seams between G6B and the delta review's code fixes.

G6B (Normalize and Align join the reset rule; the sentence stands only while it is true; the Undo
tooltips name the step; Log scale carries a range across) was built on G5 without the fixes below;
the fixes were tested on trees without G6B. Each test drives one place where the two meet, by the
routes a person uses (the template's combo, the Undo key, words typed on the figure, ranges pinned
in Axis properties, the Generate and Apply fit buttons, Export current figure…, a project written
and opened again), and asserts what is drawn, written or said, not only the state behind it.

* R1 x G6B: after a template is recalled and undone, the commit that R1 now lets reset the words
  records the step G6B names (``reset title and X range``), on three windows.
* G6B x G5 x a restored project: a project saved with a density histogram (words and a Y range)
  and a real-time ensemble (words and an X range) opens as saved, and the first Normalize and the
  first Align reset exactly those, in one step each.
* FE1 x the rule: an SVG written after a metric change on a computer that has only the Liberation
  fonts names the stand-in and then Times New Roman's own list, on the new labels as well.
* R4 x G6B: opening a second project after a reset leaves nothing of the first one's words, ranges
  or standing sentence on the Subpopulations page, and a first report there draws none.
* R3 x G5 x G6B: another PCA feature takes the words and both ranges in the tick's own step, the
  sentence names them, and it goes with the Undo that brings them back.

Synthetic data only; offscreen.
"""
from __future__ import annotations

import dataclasses
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtTest, QtWidgets

from nanoscope.core.project import ProjectState, save_project
from nanoscope.gui import figure_text, font_picker
from nanoscope.plot import figure_fonts as ff

from .test_column_checklist import click_text
from .test_p0_integration import _export_through_the_menu, _svg_families
from .test_p0_r3_pca_feature_set import _run, dist, pca  # noqa: F401 -- fixtures, by name
from .test_p0_pins_clear_with_the_words import (
    _axes_of, _depth, _is_automatic, _pin_inside, _press_redo, _stays_pinned, _tab)
from .test_p0_reset_on_quantity_change import (  # noqa: F401 -- the fixtures build the pages
    _boxes, _clear_status, _click_button, _dist_words, _drawn_title, _explorer_of, _generate,
    _label, _pick, _said, _section, _series_window, _standing_line, _type_event_words,
    _type_subpop_words, _user_picks, comparison, subpop, subpop_report, summaries, vdep)
from .test_p0_template_promise import (  # noqa: F401 -- autouse fixtures of that file, by name
    _apply_fit, _comparison_with_an_events_template, _gate, _no_templates, _recall,
    _summaries_with_a_depth_box_template, _vdep_with_an_exponential_template)
from .test_plot_window_routes import (
    _add_title, _click_tab, _edit_double_click, _main_window, _pin_through_axis_properties,
    _press_undo)

L = QtTest.QTest


# =================================================================================================
# R1 x G6B: the step a template's Undo lets the commit record is the one G6B names
# =================================================================================================

def test_summaries_after_the_undo_of_a_recall_the_commit_records_a_step_that_names_what_it_took(
        summaries):
    """R1 x G6B. 2026-11-11: the page of the project window shows no Templates combo, so ``_recall``
    reaches the recall through the page's template bar (see the module docstring of
    ``test_p0_template_promise``)."""
    window, page, explorer = summaries
    plot, widget = _summaries_with_a_depth_box_template(page)
    page.activateWindow()
    _recall(page, "depth box")
    _press_undo(page)
    assert widget.undo_button.toolTip() == "Undo: x axis range 1.5–3"

    _generate(page)

    assert widget.state().title is None
    assert widget.undo_button.toolTip() == "Undo: reset title and X range"
    assert widget.redo_button.toolTip() == "Nothing to redo."
    assert _said(page, explorer)[0].startswith("Title and value range reset: the plot now shows")
    _press_undo(page)                       # the key takes the reset back, and names it to redo
    assert widget.state().title == "Dwell ECDF (mine)"
    assert widget.redo_button.toolTip() == "Redo: reset title and X range"
    assert _said(page, explorer)[0] == "", "the sentence goes with the words it said were automatic"
    _press_redo(page)
    assert widget.state().title is None
    assert widget.undo_button.toolTip() == "Undo: reset title and X range"
    assert _said(page, explorer)[0] == "", "a Redo does not bring the sentence back"


def test_voltage_dependence_after_the_undo_of_a_recall_apply_fit_records_the_wording_step(vdep):
    page, explorer = vdep
    plot, widget = _vdep_with_an_exponential_template(page)
    page.activateWindow()
    _recall(page, "exp trend")
    _press_undo(page)

    _apply_fit(page)

    assert widget.state().fit_label_state().template == ""
    assert widget.undo_button.toolTip() == "Undo: fit label text automatic"
    assert _said(page, explorer)[0] == "Fit label wording reset: the plot now shows the exponential fit."
    _press_undo(page)
    assert widget.redo_button.toolTip() == "Redo: fit label text automatic"
    assert page.fit_label_text().startswith("Mine for the linear fit:")
    assert _said(page, explorer)[0] == ""


def test_comparison_after_the_undo_of_a_recall_the_commit_records_a_step_that_names_the_boxes(
        comparison):
    page, explorer, drawn = comparison
    widget = _comparison_with_an_events_template(page, drawn)
    page.activateWindow()
    _recall(page, "events")
    _section(page, "Appearance")
    _press_undo(page)

    _generate(page)

    assert _boxes(widget) == ("", "", "")
    assert widget.undo_button.toolTip() == "Undo: reset title, X label and Y label"
    assert page._status.text().startswith("Title, X label and Y label reset: the plot now shows")
    _press_undo(page)
    assert _boxes(widget) == ("Mine", "Dw", "Dp")
    assert widget.redo_button.toolTip() == "Redo: reset title, X label and Y label"
    assert not page._status.text().startswith("Title"), "the sentence goes with the words it said"


# =================================================================================================
# G6B x G5 x a restored project: Normalize and Align on pages that opened with their pins
# =================================================================================================

def test_a_project_saved_with_pins_on_a_density_histogram_and_an_ensemble_resets_on_the_first_normalize_and_align(
        qapp, qtbot, tmp_path, monkeypatch):
    window = _series_window(qtbot, tmp_path, monkeypatch)
    window._open_plots()
    plots = window._plots_window
    _explorer_of(window)
    plots.activateWindow()
    assert plots.sources()
    _click_button(plots._plot_btn)
    _pick(plots._norm_combo, text="density")
    _add_title(plots._plot, "Density, three pores")
    _edit_double_click(plots._plot, plots._plot.getAxis("left").label, "Per second")
    y_pin = _pin_inside(plots._plot, "left")
    plots._history_touch()
    assert plots.config().axes.ranges() == ("Y range",)

    first = next(iter(window._session.documents())).doc_id
    window._on_selection(first, 0)
    window._open_event_plots()
    events = window._event_plot_window
    events.activateWindow()
    # 2026-11-11: a page in the project window computes nothing until Update plot, and Normalize
    # time and Align are drafts until it too (the figure workflow, 8661d35).
    bar = window._analysis_explorer.toolbar()
    _click_button(bar.update_button)
    assert events.density_counts() is not None
    _tab(events, 1)
    _section(events, "Analysis")
    _click_button(events._normalize)                       # seconds, so Align is enabled
    _pick(events._align_combo, text="center")
    _click_button(bar.update_button)
    assert events.inspector_adapter().pending().count == 0, "the saved figure is the centred ensemble"
    _type_event_words(events, plot=events._ensemble_plot)
    x_pin = _pin_inside(events._ensemble_plot, "bottom")
    project = tmp_path / "pinned.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_plots()
    reopened._open_event_plots()
    explorer = _explorer_of(reopened)
    plots, events = reopened._plots_window, reopened._event_plot_window

    # Distributions opens as saved: density, its words, its Y range, and no sentence.
    _click_tab(explorer, plots)
    plots.activateWindow()
    assert plots.normalization() == "density"
    assert _dist_words(plots)[:3] == ("Density, three pores", "", "Per second")
    assert plots.config().axes.ranges() == ("Y range",)
    _stays_pinned(plots._plot, "left", y_pin)
    assert _said(plots, explorer)[0] == ""
    steps = plots._history.depth()[0]

    _pick(plots._norm_combo, text="probability per bin")

    assert plots._history.depth()[0] == steps + 1          # the pick and its reset: one step
    assert _dist_words(plots) == ("", "", "", "") and plots.config().axes.ranges() == ()
    assert _said(plots, explorer)[0].startswith("Title, Y label and Y range reset: the plot is now set to show")
    _is_automatic(plots._plot, "left", was=y_pin, auto_ranging=False)
    assert plots.undo_config()                             # a reopened page: one Undo, pick and reset
    assert plots.normalization() == "density"
    assert _dist_words(plots)[:3] == ("Density, three pores", "", "Per second")
    _stays_pinned(plots._plot, "left", y_pin)
    assert _said(plots, explorer)[0] == "", "Undo brought them back: the sentence is down"

    # Event waveforms opens as saved: seconds, 'center', its words and its X range.
    _click_tab(explorer, events)
    events.activateWindow()
    widget, ens = events._appearance_widget, events._ensemble_plot
    _tab(events, 1)
    assert events._align_combo.currentText() == "center" and not events._normalize.isChecked()
    assert (widget.state().title, widget.state().x_label) == ("Dwell vs depth", "Dwell")
    _stays_pinned(ens, "bottom", x_pin)
    assert _said(events, explorer)[0] == ""
    before, depth = _axes_of(events), _depth(widget)
    _section(events, "Analysis")

    _pick(events._align_combo, text="extremum")             # a draft: nothing moves until Update plot
    assert _depth(widget) == depth and widget.state().title == "Dwell vs depth"
    assert _said(events, explorer)[0] == ""
    _click_button(explorer.toolbar().update_button)

    assert _depth(widget) == (depth[0] + 1, 0)
    assert widget.state().title is None and "ensemble.bottom" not in _axes_of(events)
    assert _said(events, explorer)[0] == (
        "Title, X label, Y label and X range reset: the plot now shows the updated event-count map "
        "and ensemble.")
    assert widget.undo_button.toolTip() == "Undo: reset title, X label, Y label and X range"
    _press_undo(events)
    assert _axes_of(events) == before and widget.state().title == "Dwell vs depth"
    _stays_pinned(ens, "bottom", x_pin)
    assert _said(events, explorer)[0] == ""
    reopened._dirty = False


# =================================================================================================
# FE1 x the rule: the SVG of a figure drawn in a stand-in font, after a metric change
# =================================================================================================

LIBERATION = ["Liberation Sans", "Liberation Serif", "Liberation Mono", "DejaVu Sans"]
SERIF_STAND_IN = ("'Liberation Serif', 'Times New Roman', Times, Tinos, 'TeX Gyre Termes', "
                  "'Nimbus Roman', serif")
SANS_STAND_IN = ("'Liberation Sans', Arial, Helvetica, Arimo, 'TeX Gyre Heros', 'Nimbus Sans', "
                 "sans-serif")


def test_an_svg_after_a_metric_change_on_a_computer_with_only_the_liberation_fonts_keeps_the_serif_list(
        vdep, monkeypatch, tmp_path):
    page, explorer = vdep
    monkeypatch.setattr(font_picker, "installed_families", lambda: list(LIBERATION))
    plot, widget = page.plot_item(), page._appearance_widget
    widget.apply_appearance(dataclasses.replace(widget.state(), font_family="Times New Roman"))
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    _pin_through_axis_properties(plot, "left", 20.0, 400.0)
    _section(page, "Analysis")

    _pick(page._metric_combo, text="Median depth (pA)")      # the rule: the words and the Y range go

    assert widget.state().title is None and widget.state().axis_pin("left").range() is None
    assert widget.state().font_family == "Times New Roman", "the figure keeps the name it was given"
    svg = _export_through_the_menu(page, monkeypatch, tmp_path / "depth.svg", ".svg")
    families = _svg_families(svg)
    assert families and set(families) == {SERIF_STAND_IN}, set(families)
    assert SERIF_STAND_IN == ff.css_font_family("Liberation Serif")
    assert "Median depth (pA)" in svg.read_text(encoding="utf-8"), "the new metric's own label"

    _click_button(page._workflow.reset_button)               # Reset figure: back to Arial's list
    again = _export_through_the_menu(page, monkeypatch, tmp_path / "reset.svg", ".svg")
    assert set(_svg_families(again)) == {SANS_STAND_IN}


# =================================================================================================
# R4 x G6B: opening a second project after a reset
# =================================================================================================

def test_opening_a_second_project_after_a_reset_leaves_nothing_of_the_first_on_subpopulations(
        qapp, qtbot, monkeypatch, tmp_path, subpop_report):
    """2026-11-11: figures belong to the project since 8661d35. Opening another project does not
    keep and blank the one Subpopulations page, as it once did: it drops the outgoing project's
    figures and the page alias with them, and the next report gets a NEW page. The promise is the
    same: nothing of the outgoing project's words, ranges, look or standing sentence reaches the
    next project's first report. A new page starts from the look the app gives every figure
    (``fresh``, a style binding and its sizes), not from ``{}``."""
    other = tmp_path / "other.nanoscope"
    save_project(str(other), ProjectState())                 # a project that holds no report
    window = _main_window(qtbot, monkeypatch)

    def open_report():
        window._open_subpop_window(subpop_report, source_result=None, source_recipe=None,
                                   accepted_event_ids=None)
        QtWidgets.QApplication.processEvents()
        page = window._subpop_window
        explorer = _explorer_of(window)
        page.activateWindow()
        scatter = page.embedding_view()
        return page, explorer, scatter, page._appearance_widget

    page, explorer, scatter, widget = open_report()
    fresh = page.appearance_state()
    assert not {"title", "x_label", "y_label", "axes", "text"} & set(fresh), fresh
    plot = scatter._plot.getPlotItem()
    _type_subpop_words(page)
    _pin_inside(plot, "bottom")
    _section(page, "Analysis")
    _pick(scatter._embed_combo, text="UMAP")                 # the rule: words and the range go
    assert widget.state().title is None
    sentence = ("Title, X label, Y label and X range reset: the plot now shows the UMAP "
                "embedding.")
    note, on_screen = _standing_line(page, explorer)
    assert on_screen and note.endswith(sentence), note
    window._dirty = False                                    # the Discard answer

    # (1) The sentence is standing when the next project is opened: it must not outlive the page.
    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._open_project(str(other))
    assert window._subpop_window is None, "the outgoing project's page is not kept"
    assert not list(window._figure_workspace.records())
    second, explorer, scatter, widget = open_report()
    assert second is not page
    assert second.appearance_state() == fresh and second.report() is not None
    assert "reset:" not in second.live_note() and "reset:" not in _standing_line(second, explorer)[0]
    assert scatter.embedding() == "t-SNE"
    assert not widget.undo_button.isEnabled() and not widget.redo_button.isEnabled()

    # (2) Words and a range typed on the second project's first report do not reach a third.
    plot = scatter._plot.getPlotItem()
    _add_title(plot, "Pore B in t-SNE")
    _pin_through_axis_properties(plot, "bottom", -3.0, 3.0)
    assert second.appearance_state().get("title") == "Pore B in t-SNE"
    assert "reset:" not in _standing_line(second, explorer)[0]
    window._dirty = False
    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._open_project(str(other))
    assert window._subpop_window is None
    third, explorer, scatter, widget = open_report()
    assert third is not second and third is not page
    assert third.appearance_state() == fresh and widget.title.text() == ""
    assert not widget.undo_button.isEnabled() and not widget.redo_button.isEnabled()
    assert _drawn_title(scatter._plot.getPlotItem())[0] == ""
    lo, hi = scatter._plot.getPlotItem().getViewBox().viewRange()[0]
    assert (round(lo, 3), round(hi, 3)) != (-3.0, 3.0)
    assert scatter.embedding() == "t-SNE"
    window._dirty = False


# =================================================================================================
# R3 x G5 x G6B: another PCA feature
# =================================================================================================

def test_another_pca_feature_takes_the_words_and_both_ranges_and_the_sentence_goes_with_the_undo(
        pca):
    """R3 makes the feature set a quantity; G5 makes the pinned score ranges part of what it takes
    (the sentence names them); G6B ends the sentence when the Undo brings them back, and a Redo
    does not bring it back."""
    page, explorer, (x_pin, y_pin) = pca
    plot = page._plot
    steps = page._history.depth()[0]

    click_text(page._features, "frac_depth")                  # one click on a row

    assert page._history.depth()[0] == steps + 1              # the tick's own step, no more
    assert _dist_words(page) == ("", "", "", "") and page.config().axes.ranges() == ()
    assert _said(page, explorer) == (
        "Title, X label, Y label, X range and Y range reset: the plot is now set to show the "
        "PCA scores.", True)
    _run(page)
    _is_automatic(plot, "bottom", was=x_pin, auto_ranging=False)
    _is_automatic(plot, "left", was=y_pin, auto_ranging=False)
    assert _said(page, explorer)[0].startswith("Title, X label, Y label, X range and Y range")

    assert page.undo_config()                                  # the feature, words and pins together
    assert set(page.feature_selection()) == {"dwell_s", "depth_pA"}
    assert _dist_words(page)[0] == "PCA of dwell and depth"
    assert page.config().axes.manual_x == pytest.approx(x_pin)
    assert page.config().axes.manual_y == pytest.approx(y_pin)
    assert _said(page, explorer)[0] == "", "the Undo brought them back: the sentence is down"
    assert page.redo_config()
    assert _dist_words(page) == ("", "", "", "") and page.config().axes.ranges() == ()
    assert _said(page, explorer)[0] == "", "a Redo does not bring the sentence back"
