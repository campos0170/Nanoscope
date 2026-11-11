"""Phase 0, the seams: what crosses the four groups and none of them owned.

The reset rule (G1), the font list (G2), Distributions' switches and Reset figure (G3) and the
export fonts (G4) were each built and verified alone. This file drives the places where two or
more of them meet, through the routes a person uses:

* a font picked in the new picker, then a quantity change (the reset rule keeps the look and the
  font), then **Reset figure** (the font goes back to Arial, as one Undo step), in a page that
  keeps its appearance in the shared widget and in Distributions, which keeps it in its own style,
  and in the two matplotlib windows the picker also drives (Comparison figures, the study figure
  editor);
* the status line: the rule's sentence and **Reset figure** share it without either erasing the
  other, and a later change with nothing typed takes the sentence down after a Reset figure too;
* Distributions: an old-format project (``fit_values_on_plot``) still shows BIC and KS and the
  picker's *Other font…* value, and a plot-type change takes the fit wording back while the
  numbers the fold put in the label stay, all through a real ``_write_project`` and
  ``_open_project``;
* an SVG exported through the page's own **Export current figure…** menu after a font was picked
  through *Other font…*: that family first, the stack after it; and what it names for a family
  this computer lacks;
* every Analysis Explorer page opens, draws and exports an SVG and a PDF with the defaults;
* nothing in the repository still names a removed control or the old button text;
* the cheap items the second-round verifiers left (each test names its item).

Synthetic data only; every recording is an in-memory document or a CSV written to ``tmp_path``.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib as mpl
import pandas as pd
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.core.figure_quantity_basis import read_quantity_bases
from nanoscope.core.session import Session
from nanoscope.gui import figure_text, plot_export
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.analysis_explorer import PAGE_IV
from nanoscope.gui.fit_label import FitLabelState
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plot_export_options import resolve_export_options
from nanoscope.plot import figure_fonts as ff
from nanoscope.plot.style import font_family_list

from .test_p0_distributions_switches_and_reset_figure import (
    OLD_KEYS, _df, _label, _manifest_text, _old_style, _with_old_keys_added)
from .test_p0_figure_font_picker import (  # noqa: F401 -- ``arial`` is a fixture
    OTHER, arial, drawn_families, in_family, pdf_font_names, picking, settle, tick_family)
from .test_p0_pins_clear_with_the_words import _press_redo
from .test_p0_reset_on_quantity_change import (  # noqa: F401 -- the fixtures build the pages
    _boxes, _click_button, _in_explorer, _pick, _said, _type_dist_words,
    _type_in_box, _type_vdep_words, _user_picks, comparison, dist, events, noise, subpop,
    subpop_report, summaries, vdep)
from .test_plot_window_routes import _edit_double_click, _press_undo
from .test_plots_reopen_figure import _main, _recordings, _settle

L = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
TIMES = "Times New Roman"
ARIAL_STACK = ff.css_font_family(None)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


# -- helpers ---------------------------------------------------------------------------------------

def _pick_font(combo, text: str) -> None:
    """Choose a row the way a person does -- open the drop-down, click the row -- as G2's ``pick``
    does, but sure that the drop-down is up first. Qt ignores a request to open a popup that
    follows within a moment of another popup closing (measured here: the first ``showPopup`` of a
    test that comes right after one that picked from a combo does nothing, and the click then
    lands on a list that is not there), so the request is repeated until the list is showing."""
    row = next(i for i in range(combo.count()) if combo.itemText(i) == text)
    for _ in range(10):
        combo.showPopup()
        QtWidgets.QApplication.processEvents()
        if combo.view().isVisible():
            break
        QtTest.QTest.qWait(60)
    assert combo.view().isVisible(), f"the font list did not open for {text!r}"
    view = combo.view()
    rect = view.visualRect(view.model().index(row, 0))
    QtTest.QTest.mouseClick(view.viewport(), L, NOMOD, rect.center())
    QtWidgets.QApplication.processEvents()


def _show_appearance(page) -> None:
    """The Appearance controls on screen, whichever way the page folds its sections away."""
    if hasattr(page, "_show_control_section"):         # Distributions
        page._show_control_section("Appearance")
    else:
        page._workflow.set_section("Appearance")
    QtWidgets.QApplication.processEvents()


def _text_places(plot) -> dict[str, set[str]]:
    """Where each piece of a figure's text is drawn, read off what Qt lays out: the families of
    the glyph runs of both axis labels (and the title, when it shows) and the font of the tick
    numbers."""
    out: dict[str, set[str]] = {}
    for side in ("left", "bottom"):
        axis = plot.getAxis(side)
        out[f"{side} label"] = drawn_families(axis.label)
        out[f"{side} ticks"] = {tick_family(axis)}
    if plot.titleLabel.isVisible() and plot.titleLabel.text:
        out["title"] = drawn_families(plot.titleLabel)
    return out


def _all_drawn_in(plot, family: str) -> None:
    for place, families in _text_places(plot).items():
        assert in_family(families, family), (family, place, sorted(families))


def _svg_families(path: Path) -> list[str]:
    """The ``font-family`` of every ``<text>`` of an SVG file."""
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    return [e.get("font-family", "") for e in root.iter("{http://www.w3.org/2000/svg}text")]


def _pdf_is_clean(path: Path) -> list[str]:
    """The PDF's font names, with the rule every export keeps: no Type 3 drawings and no
    private (dot-prefixed) system face."""
    raw = path.read_bytes()
    assert not re.search(rb"/Subtype\s*/Type3\b", raw), f"{path.name}: Type 3 fonts"
    names = sorted(pdf_font_names(path))
    assert not [n for n in names if n.split("+")[-1].startswith(".")], (path.name, names)
    return names


def _export_through_the_menu(page, monkeypatch, target: Path, extension: str = ".svg") -> Path:
    """Page's own Export ▾ > Export current figure…, with the two modal dialogs it opens
    answered as a person would (this format, native size, this file)."""
    monkeypatch.setattr(plot_export, "choose_export_options",
                        lambda parent, native_size, **kw: resolve_export_options(
                            native_size, extension=extension))
    monkeypatch.setattr(plot_export, "ask_export_path", lambda *a, **k: str(target))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    menu = page._export_menu if hasattr(page, "_export_menu") else page._workflow.export_menu
    action = next(a for a in menu.actions() if a.text() == "Export current figure…")
    action.trigger()
    QtWidgets.QApplication.processEvents()
    assert target.is_file() and target.stat().st_size > 0, f"{target.name} was not written"
    return target


def _bases(appearance: PlotAppearance) -> dict[str, tuple[str, str]]:
    """The quantity each axis shows, as the appearance's display witness records it:
    ``{axis: (quantity key, display unit)}``. 2026-11-11: a page that presents units carries this
    witness since 8661d35 (quantity display); it is not look, and Reset figure keeps it."""
    return {axis: (basis.quantity.key, basis.quantity.display_unit)
            for axis, basis in read_quantity_bases(appearance.quantity_display).items()}


# ===============================================================================================
# (a) a font picked in the picker x the reset rule x Reset figure
# ===============================================================================================

def test_a_picked_font_survives_a_quantity_change_and_reset_figure_puts_arial_back(vdep, arial):
    """Voltage dependence, whose appearance is the shared widget's. The font is look: the rule
    that takes the typed words back leaves it (and the new metric's automatic labels are drawn in
    it), and Reset figure takes it back to Arial (default) in ONE Undo step that Undo returns."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _show_appearance(page)
    _pick_font(widget.font_family, TIMES)
    assert widget.state().font_family == TIMES and widget.font_family.currentText() == TIMES
    _all_drawn_in(plot, TIMES)
    _type_vdep_words(page)
    page._workflow.set_section("Analysis")
    QtWidgets.QApplication.processEvents()

    _pick(page._metric_combo, text="Median depth (pA)")

    # the words went (and the sentence says so) ...
    assert _boxes(widget) == ("", "", "")
    assert _said(page, explorer) == (
        "Title, X label, Y label and fit label wording reset: the plot now shows median depth.",
        True)
    # ... the look stayed: the picked font, and what is drawn is in it, new labels included.
    assert widget.state().font_family == TIMES and widget.font_family.value() == TIMES
    assert widget.font_family.currentText() == TIMES
    assert widget.state().text_style("title").bold is True
    _all_drawn_in(plot, TIMES)
    assert in_family(drawn_families(page._fit_label), TIMES)

    steps = widget._history.depth()[0]
    _click_button(page._workflow.reset_button)                      # Reset figure, in the toolbar

    assert widget._history.depth() == (steps + 1, 0), "one gesture, one step"
    # 2026-11-11: everything but the witness of the quantity the axis shows (8661d35) is back to a
    # figure's defaults.
    assert replace(widget.state(), quantity_display=None) == PlotAppearance()
    assert _bases(widget.state()) == {"left": ("median_depth", "")}, "Reset figure keeps what is drawn"
    assert widget.font_family.value() == "" and widget.font_family.currentText() == "Arial (default)"
    _all_drawn_in(plot, "Arial")
    assert in_family(drawn_families(page._fit_label), "Arial")

    _press_undo(page)                                              # one Undo gives it all back
    assert widget._history.depth() == (steps, 1)
    assert widget.state().font_family == TIMES and widget.font_family.currentText() == TIMES
    assert widget.state().text_style("title").bold is True
    _all_drawn_in(plot, TIMES)
    assert widget.redo()
    assert widget.state().font_family is None
    _all_drawn_in(plot, "Arial")


def test_a_picked_font_survives_a_plot_type_change_in_distributions_and_reset_figure_undoes_in_a_step(
        dist, arial):
    """Distributions keeps the font in its own style and its own (Configuration) Undo; the
    picker is the shared widget's. Same seams, same expectations."""
    page, explorer = dist
    widget = page._appearance_widget
    _show_appearance(page)
    _pick_font(widget.font_family, TIMES)
    assert page.config().style.font_family == TIMES and widget.font_family.currentText() == TIMES
    _all_drawn_in(page._plot, TIMES)
    _type_dist_words(page)
    page._history_touch()
    page._show_control_section("Data")

    _pick(page._plot_type_combo, text="Scatter")
    _user_picks(page._scatter_fit, "linear")
    _click_button(page._plot_btn)

    style = page.config().style
    assert (style.title, style.x_label, style.y_label) == ("", "", "")
    assert style.fit_label_state().template == ""
    assert style.font_family == TIMES and widget.font_family.currentText() == TIMES
    assert style.text_style("title").bold is True, "the look stays"
    _all_drawn_in(page._plot, TIMES)
    assert in_family(drawn_families(page._fit_label_item), TIMES)

    steps = page._history.depth()[0]
    _click_button(widget.reset_button)                             # Reset figure, in the toolbar

    assert page._history.depth()[0] == steps + 1, "one gesture, one step"
    assert page.config().style.font_family == ""
    assert widget.font_family.value() == "" and widget.font_family.currentText() == "Arial (default)"
    _all_drawn_in(page._plot, "Arial")
    assert in_family(drawn_families(page._fit_label_item), "Arial")
    assert page.undo_config()
    assert page._history.depth()[0] == steps
    assert page.config().style.font_family == TIMES
    assert widget.font_family.currentText() == TIMES
    _all_drawn_in(page._plot, TIMES)


def test_the_picker_and_reset_figure_drive_the_matplotlib_comparison_figure(
        comparison, arial, monkeypatch):
    """Comparison figures is drawn by matplotlib (G4) from the shared Appearance widget (G2's
    picker, G3's Reset figure): a picked font reaches the figure's text as the family LIST, and
    Reset figure puts Arial's back."""
    from nanoscope.plot import comparison_figures as renderer

    page, explorer, _drawn = comparison
    seen: list[list[str]] = []
    real = renderer.render_comparison

    def spy(*args, **kwargs):
        figure = real(*args, **kwargs)
        seen.append(list(figure._suptitle.get_fontfamily()))
        return figure

    monkeypatch.setattr(renderer, "render_comparison", spy)
    widget = page._appearance_widget
    _show_appearance(page)
    _pick_font(widget.font_family, TIMES)
    assert seen and seen[-1][0] == TIMES, seen
    assert seen[-1][-1] == "DejaVu Sans" and "serif" in seen[-1], seen[-1]
    _click_button(page._workflow.reset_button)
    assert widget.font_family.currentText() == "Arial (default)"
    assert seen[-1] == font_family_list(None), seen[-1]


def test_the_study_editors_picker_and_buttons_drive_its_matplotlib_figure(qtbot, arial):
    """The study figure editor, through its real widgets: pick Times New Roman, press Apply
    appearance, then Reset all figure styling. The figure is Arial before, Times New Roman after
    the first, and Arial (not the study's own DejaVu Sans) after the second -- the real buttons
    that G4's own tests call as slots.

    2026-11-11: the editor has two inspector tabs since 8661d35: Format, in front, where the figure
    font picker lives and a pick is applied at once, and Study details, which holds the two
    buttons. So the pick is made in Format, where the Undo and Redo KEYS also answer (the window
    binds them, see test_p0_final_seams), and the Study details tab is clicked before the buttons
    are pressed."""
    from matplotlib.figure import Figure

    from nanoscope.gui.study_figure_window import StudyFigureWindow

    figure = Figure(figsize=(3, 2.4))
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1])
    axes.set_xlabel("Dwell (ms)")
    axes.set_title("Study")
    frame = pd.DataFrame({"event_id": ["a", "b"], "dwell_ms": [1., 2.], "accepted": [True, True]})
    page = StudyFigureWindow(figure, frame, {}, name="seam", provenance={"study": "seam"})
    qtbot.addWidget(page)
    page.show()
    qtbot.waitExposed(page)

    def family_of_the_drawn_text() -> str:
        return page._figure.axes[0].xaxis.label.get_fontfamily()[0]

    def button(text: str) -> QtWidgets.QPushButton:
        return next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == text)

    widget = page._appearance_widget
    (tabs,) = [t for t in page.findChildren(QtWidgets.QTabWidget)
               if [t.tabText(i) for i in range(t.count())] == ["Format", "Study details"]]
    assert tabs.currentIndex() == 0 and widget.font_family.isVisible()
    assert family_of_the_drawn_text() == "Arial"
    _pick_font(widget.font_family, TIMES)
    assert widget.state().font_family == TIMES
    assert family_of_the_drawn_text() == TIMES, "a pick in Format is applied at once"
    assert page.style()["appearance"]["font_family"] == TIMES
    _press_undo(page)                                       # the keys answer with Format in front
    assert family_of_the_drawn_text() == "Arial" and "font_family" not in page.style()["appearance"]
    _press_redo(page)
    assert family_of_the_drawn_text() == TIMES and page.style()["appearance"]["font_family"] == TIMES

    bar = tabs.tabBar()
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(1).center())     # Study details
    QtWidgets.QApplication.processEvents()
    assert tabs.currentIndex() == 1
    assert button("Apply appearance").isVisible() and button("Reset all figure styling").isVisible()
    _click_button(button("Apply appearance"))
    assert family_of_the_drawn_text() == TIMES
    assert page.style()["appearance"]["font_family"] == TIMES
    _click_button(button("Reset all figure styling"))
    assert page.style()["appearance"] == {}
    assert widget.font_family.currentText() == "Arial (default)"
    assert family_of_the_drawn_text() == "Arial"
    if page._figure is not None:
        page._figure.clear()


# ===============================================================================================
# (b) the reset rule's sentence and Reset figure share the status line
# ===============================================================================================

def test_reset_figure_and_the_rules_sentence_share_the_status_line_in_voltage_dependence(vdep):
    """Reset figure writes no status text of its own. It must not erase the rule's sentence (it
    is still true), the rule's next run must not leave that sentence behind when nothing is
    typed, and a sentence of an unrelated kind is never touched."""
    page, explorer = vdep
    widget = page._appearance_widget
    sentence = "Title, X label, Y label and fit label wording reset: the plot now shows median depth."
    _type_vdep_words(page)
    page._workflow.set_section("Analysis")
    QtWidgets.QApplication.processEvents()
    _pick(page._metric_combo, text="Median depth (pA)")
    assert _said(page, explorer) == (sentence, True)

    _click_button(page._workflow.reset_button)                     # Reset figure
    assert _said(page, explorer) == (sentence, True), "Reset figure said something or erased it"
    _press_undo(page)                                              # ... and its Undo
    assert _said(page, explorer) == (sentence, True)
    assert widget.redo()

    depth = widget._history.depth()
    # 2026-11-11: this metric reads "(ms)", not "(s)", since 8661d35 shows dwell in milliseconds.
    _pick(page._metric_combo, text="Median dwell (ms)")            # nothing is typed any more
    assert widget._history.depth() == depth, "no step"
    assert _said(page, explorer)[0] == "", "the old sentence outlived the next change"

    # Typed again, the next change names only what it cleared; Reset figure leaves it alone.
    page._workflow.set_section("Appearance")
    _type_in_box(widget.title, "Mine")
    QtTest.QTest.qWait(800)
    page._workflow.set_section("Analysis")
    _pick(page._metric_combo, text="Event count")
    assert _said(page, explorer) == ("Title reset: the plot now shows event count.", True)
    _click_button(page._workflow.reset_button)
    assert _said(page, explorer)[0] == "Title reset: the plot now shows event count."
    # a message that is not the rule's stays through the next change of the quantity
    page.statusBar().showMessage("Voltage dependence has a log scale on the y axis only.", 5000)
    _pick(page._metric_combo, text="Median depth (pA)")
    assert _said(page, explorer)[0] == "Voltage dependence has a log scale on the y axis only."


def test_reset_figure_and_the_rules_sentence_share_the_status_line_in_distributions(dist):
    page, explorer = dist
    _type_dist_words(page)
    page._history_touch()
    page._show_control_section("Data")
    _pick(page._plot_type_combo, text="Scatter")
    sentence = ("Title, X label, Y label and fit label wording reset: the plot is now set to show "
                "the scatter of ycol against xcol.")
    assert _said(page, explorer) == (sentence, True)

    steps = page._history.depth()[0]
    _click_button(page._appearance_widget.reset_button)            # Reset figure, in the toolbar
    assert page._history.depth()[0] == steps + 1
    assert _said(page, explorer) == (sentence, True), "Reset figure said something or erased it"

    _pick(page._plot_type_combo, text="Histogram")                 # nothing typed: the line clears
    assert _said(page, explorer)[0] == ""


# ===============================================================================================
# (c) Distributions: an old-format project, the picker's Other font…, and the reset rule
# ===============================================================================================

def test_an_old_format_project_with_a_picked_other_font_reopens_and_the_rule_keeps_its_numbers(
        qapp, qtbot, tmp_path, monkeypatch, arial):
    """A project the old build saved (Fit values on plot ticked) whose figure is also in Menlo,
    picked through Other font…. It reopens with BIC and KS on the label (G3's fold) and Menlo in
    the picker and on the figure (G2). Wording typed over the label goes back to automatic with
    a plot-type change (G1) and the numbers the fold put in the label stay; the project saved
    again carries none of the old keys and reopens to the same figure."""
    window, plots = _recordings(qtbot, tmp_path)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    _settle()
    plots._show_control_section("Appearance")
    picker = plots._appearance_widget.font_family
    picking(monkeypatch, OTHER)
    _pick_font(picker, "Other font…")
    settle(qtbot, picker)
    assert plots.config().style.font_family == OTHER and picker.has_extra_row()
    assert "BIC" not in _label(plots)
    saved = tmp_path / "now.nanoscope"
    assert window._write_project(str(saved))
    window._dirty = False
    older = tmp_path / "older.nanoscope"
    assert _with_old_keys_added(saved, older, fit_values_on_plot=True) == 1

    again = _main(qtbot)
    with qtbot.waitSignal(again.projectRestored, timeout=20000):
        again._open_project(str(older))
    again._open_plots()
    _settle()
    page = again._plots_window
    assert "BIC = " in _label(page) and "KS = " in _label(page), _label(page)
    assert page.config().style.font_family == OTHER
    reopened = page._appearance_widget.font_family
    assert reopened.value() == OTHER and reopened.currentText() == OTHER and reopened.has_extra_row()
    assert page._font_combo.value() == OTHER, "the Style group's own picker shows it too"
    _all_drawn_in(page._plot, OTHER)
    assert in_family(drawn_families(page._fit_label_item), OTHER)

    # G1 over G3's numbers: the wording is typed over the label, then the plot type moves.
    _edit_double_click(page._plot, page._fit_label_item, "Mine: {equation} | {bic}")
    assert _label(page).startswith("Mine: ")
    fields = page.config().style.fit_label_state().fields
    assert {"bic", "ks"} <= set(fields)
    page._history_touch()
    page._show_control_section("Data")
    _pick(page._plot_type_combo, text="Scatter")
    _user_picks(page._scatter_fit, "linear")
    _click_button(page._plot_btn)
    label = page.config().style.fit_label_state()
    assert label.template == "" and label.fields == fields, "the words went, the numbers stayed"
    assert "Mine" not in _label(page)
    _pick(page._plot_type_combo, text="Histogram")
    page.set_fit("exponential")
    _click_button(page._plot_btn)
    _settle()
    assert "BIC = " in _label(page) and "KS = " in _label(page) and "Mine" not in _label(page)
    assert page.config().style.font_family == OTHER

    # Saved again: none of the five old keys, the same figure, the same font, on a third window.
    resaved = tmp_path / "resaved.nanoscope"
    assert again._write_project(str(resaved))
    again._dirty = False
    text = _manifest_text(resaved)
    for key in OLD_KEYS:
        assert f'"{key}"' not in text, key
    third = _main(qtbot)
    with qtbot.waitSignal(third.projectRestored, timeout=20000):
        third._open_project(str(resaved))
    third._open_plots()
    _settle()
    last = third._plots_window
    assert "BIC = " in _label(last) and "KS = " in _label(last)
    assert last.config().style.font_family == OTHER
    assert last._appearance_widget.font_family.value() == OTHER
    window._dirty = again._dirty = third._dirty = False


# ===============================================================================================
# (d) an SVG exported after a font was picked through Other font…
# ===============================================================================================

def test_an_svg_exported_after_other_font_names_that_family_first_and_the_stack_after(
        dist, arial, monkeypatch, qtbot, tmp_path):
    """Distributions, through the page's own Export ▾ menu: the family picked in Other font…
    is the first name of every text element, then Arial's stand-ins; back on Arial (default) it
    is Arial's stack; and a family this computer lacks (a project from another computer) is
    written as the substitute the figure was drawn in -- Arial -- not as the missing name."""
    page, explorer = dist
    widget = page._appearance_widget
    _show_appearance(page)
    picker = widget.font_family
    picking(monkeypatch, OTHER)
    _pick_font(picker, "Other font…")
    settle(qtbot, picker)
    assert page.config().style.font_family == OTHER

    other = _export_through_the_menu(page, monkeypatch, tmp_path / "other.svg")
    families = _svg_families(other)
    assert families and set(families) == {ff.css_font_family(OTHER)}, set(families)
    assert families[0].split(",")[0].strip("'\" ") == OTHER
    assert ARIAL_STACK in families[0] or families[0].endswith(ARIAL_STACK.split(", ", 1)[1])

    _pick_font(picker, "Arial (default)")
    plain = _export_through_the_menu(page, monkeypatch, tmp_path / "plain.svg")
    assert set(_svg_families(plain)) == {ARIAL_STACK}

    page.set_font_family("Calibri")                  # not installed here: drawn in the substitute
    missing = _export_through_the_menu(page, monkeypatch, tmp_path / "missing.svg")
    assert set(_svg_families(missing)) == {ARIAL_STACK}, "the SVG names what the figure was drawn in"
    assert page.config().style.font_family == "Calibri", "...and the figure still keeps the name"


def test_an_svg_exported_from_a_page_that_keeps_the_shared_widget_names_the_picked_family_first(
        vdep, arial, monkeypatch, qtbot, tmp_path):
    page, explorer = vdep
    _show_appearance(page)
    picker = page._appearance_widget.font_family
    picking(monkeypatch, OTHER)
    _pick_font(picker, "Other font…")
    settle(qtbot, picker)
    exported = _export_through_the_menu(page, monkeypatch, tmp_path / "vdep.svg")
    families = _svg_families(exported)
    assert families and set(families) == {ff.css_font_family(OTHER)}, set(families)
    pdf = _export_through_the_menu(page, monkeypatch, tmp_path / "vdep.pdf", ".pdf")
    names = _pdf_is_clean(pdf)
    assert any(OTHER in n for n in names), names


# ===============================================================================================
# (e) every Explorer page opens, draws and exports with the defaults
# ===============================================================================================

PAGES = ("distributions", "summaries", "voltage", "iv", "events", "noise", "subpopulations",
         "comparison")


@pytest.fixture
def iv_in_the_explorer(qapp, qtbot, tmp_path):
    from nanoscope.gui.iv_panel import IVWindow

    from .test_iv import _ohmic_session

    page = IVWindow(_ohmic_session(tmp_path))
    qtbot.addWidget(page)
    explorer = _in_explorer(qtbot, page, PAGE_IV)
    page.activateWindow()
    return page, explorer


@pytest.mark.parametrize("kind", PAGES)
def test_every_explorer_page_opens_draws_and_exports_svg_and_pdf_with_the_defaults(
        kind, request, tmp_path):
    """The eight pages of the Analysis Explorer, each in a real Explorer, untouched: it draws, it
    exports an SVG whose every text element names Arial first and the stack after it, and a PDF
    that embeds Arial with no Type 3 drawing and no private system face.

    2026-11-11: Comparison figures were the exception, because their SVG held outlines and had no
    fonts to name. Since 8661d35 their SVG keeps literal text with the fallback families
    (``svg.fonttype`` is ``none``, then ``editable_font_stacks``) and their PDF embeds the font, so
    they are held to the same rule as the other seven pages."""
    if kind == "distributions":
        page, _explorer = request.getfixturevalue("dist")
        writer = page.save_image
    elif kind == "summaries":
        _window, page, _explorer = request.getfixturevalue("summaries")
        writer = page.save_image
    elif kind == "voltage":
        page, _explorer = request.getfixturevalue("vdep")
        writer = page.save_image
    elif kind == "iv":
        page, _explorer = request.getfixturevalue("iv_in_the_explorer")
        writer = page.save_image
    elif kind == "events":
        page, _explorer = request.getfixturevalue("events")
        writer = page.save_image
    elif kind == "noise":
        page, _explorer = request.getfixturevalue("noise")
        writer = page.save_image
    elif kind == "subpopulations":
        page, _explorer = request.getfixturevalue("subpop")
        scatter = page.embedding_view()._plot.getPlotItem()

        def writer(path):                       # the Explorer's generic figure writers
            return plot_export.write_live_plot_image(path, scatter)
    else:
        page, _explorer, _drawn = request.getfixturevalue("comparison")
        writer = page.save_image
    assert page.isVisibleTo(page.window()) or page.parent() is not None

    svg, pdf = tmp_path / f"{kind}.svg", tmp_path / f"{kind}.pdf"
    for target in (svg, pdf):
        writer(str(target))
        assert target.is_file() and target.stat().st_size > 500, f"{target.name} was not written"
    families = _svg_families(svg)
    assert families and set(families) == {ARIAL_STACK}, (kind, set(families))
    assert "AppleSystemUIFont" not in svg.read_text(encoding="utf-8")
    names = _pdf_is_clean(pdf)
    assert any("Arial" in n for n in names), (kind, names)


# ===============================================================================================
# (f) nothing names a removed control or the old button text
# ===============================================================================================

REPO = Path(__file__).resolve().parents[2]

#: A removed control's name or the old button text, and the files that may still say it, each
#: for a stated reason. Anything else that names one is a leftover.
REMOVED = {
    "Reset appearance": {
        "CHANGELOG.md": "names the rename"},
    "Fit values on plot": {
        "CHANGELOG.md": "names the removal",
        "docs/USER_GUIDE.md": "Files from earlier builds",
        "docs/inventory.json": "Fit label · Numbers note",
        "nanoscope/gui/plot_appearance.py": "comment on the Numbers grid",
        "nanoscope/gui/plots_panel.py": "the fold of an old file, and comments on it"},
    "R² on figure": {
        "CHANGELOG.md": "names the removal",
        "docs/USER_GUIDE.md": "Files from earlier builds",
        "docs/inventory.json": "Fit label · Numbers note",
        "nanoscope/gui/plots_panel.py": "the fold of an old file, and comments on it",
        "tests/gui/test_plots_fit_label.py": "a legacy payload test"},
    "KS statistic on figure": {
        "CHANGELOG.md": "names the removal",
        "docs/USER_GUIDE.md": "Files from earlier builds",
        "docs/inventory.json": "Fit label · Numbers note",
        "nanoscope/gui/plots_panel.py": "the fold of an old file, and comments on it"},
    "Title weight": {
        "CHANGELOG.md": "names the removal",
        "docs/USER_GUIDE.md": "Files from earlier builds, and why there is no such box",
        "nanoscope/gui/plots_panel.py": "the fold of an old file",
        "tests/gui/test_plots_presentation_controls.py": "a comment in a rewritten test"},
}
#: The old widgets' attribute names and the old font list helpers: no file may use them.
REMOVED_NAMES = ("_fit_values_check", "_r2_check", "_gof_check", "_title_bold", "_title_italic",
                 "_on_style_fit_values", "_on_style_show_r2", "_on_style_show_gof",
                 "_on_style_title_bold", "_on_style_title_italic", "_font_row_tooltip",
                 "_add_font_row")
#: The settings keys of the old switches may be spelled only where an old file is read or built.
LEGACY_KEYS = ("fit_values_on_plot", "show_r2", "show_gof", "title_bold", "title_italic")
LEGACY_KEY_FILES = {
    "nanoscope/gui/plots_panel.py",          # _LEGACY_STYLE_KEYS and _fold_legacy_style
    "nanoscope/gui/fit_label.py",            # FitLabelState.from_mapping(legacy=...), no caller
    "tests/gui/test_fit_label.py", "tests/gui/test_plots_fit_label.py",   # old payloads, written out
    "tests/gui/fixtures/plot_config_v1_pre_rename.json",                   # a file an old build wrote
    # 2026-11-11: project format 13 (8661d35) reads the older files at its restore boundary and
    # folds these keys away; its fixtures and tests write the old files out and check they are gone.
    "nanoscope/core/figure_codecs.py",       # the v12 restore boundary: _fold_v12_legacy_style
    "tools/make_phase2_legacy_fixture.py",   # builds a file as an older build wrote it
    "tests/core/test_p2_figure_codecs.py", "tests/core/test_p2_historical_dialects.py",  # old payloads
    "tests/core/fixtures/phase2/legacy/d1-provenance.json",    # provenance of old-format fixtures
    "tests/core/fixtures/phase2/legacy/d2-provenance.json",
    "tests/core/fixtures/phase2/legacy/v11-provenance.json",
}
SKIPPED = ("docs/sweep", "docs/RELEASE_", "docs/MANUAL_REVISION", ".git/", "__pycache__")


def _repo_files():
    for top in ("nanoscope", "docs", "tests", "tools", "packaging", "README.md", "PORT_PLAN.md",
                "CHANGELOG.md"):
        root = REPO / top
        paths = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in paths:
            relative = path.relative_to(REPO).as_posix()
            if not path.is_file() or any(skip in relative for skip in SKIPPED):
                continue
            if path.suffix not in {".py", ".md", ".json", ".txt", ".rst", ".toml", ".spec", ".html"}:
                continue
            if Path(relative).name.startswith("test_p0_"):
                continue                      # these four files assert the names are gone
            yield relative, path.read_text(encoding="utf-8", errors="replace")


def _unwrapped(text: str) -> str:
    """``text`` with each line break (and the indent, and a ``#`` that opens a wrapped comment line)
    read as one space, so a name that a document or a comment wraps across two lines is still
    found: ``Reset`` at the end of one line and ``appearance`` at the start of the next."""
    return re.sub(r"\s*\n\s*(?:#+\s*)?", " ", text)


def test_nothing_still_names_a_removed_control_or_the_old_button_text():
    leftovers: list[str] = []
    for relative, text in _repo_files():
        flat = _unwrapped(text)
        for name, allowed in REMOVED.items():
            if name in flat and relative not in allowed:
                leftovers.append(f"{relative}: {name!r}")
        for name in REMOVED_NAMES:
            if name in text:
                leftovers.append(f"{relative}: {name}")
        if re.search(r"\bQFontComboBox\s*\(", text):
            leftovers.append(f"{relative}: builds a QFontComboBox (the font controls are FontPicker)")
        if "Application default" in text:
            leftovers.append(f"{relative}: the font row's old name, 'Application default'")
        if relative not in LEGACY_KEY_FILES:
            for key in LEGACY_KEYS:
                if re.search(rf"\b{key}\b", text):
                    leftovers.append(f"{relative}: the old key {key}")
    assert not leftovers, "\n".join(leftovers)


def test_the_documented_allowances_are_still_true():
    """The allowlist above is not a place to forget things in: every file it names still says
    what it is allowed for, so an entry whose reason has gone is removed."""
    texts = dict(_repo_files())
    for name, allowed in REMOVED.items():
        for relative in allowed:
            assert name in texts[relative], f"{relative} no longer names {name!r}: drop the allowance"
    for relative in LEGACY_KEY_FILES:
        if relative.endswith(".json") and "fixtures" in relative:
            continue
        assert any(re.search(rf"\b{key}\b", texts[relative]) for key in LEGACY_KEYS), relative


def test_the_user_guide_has_no_sentence_that_two_groups_wrote_against_each_other():
    """The corrections the integration made to the four groups' paragraphs, pinned: the Numbers
    row is under Format ▸ Fit label; Reset figure leaves two colormap rows; the exports follow the
    picker's fallback; Distributions' Undo is the exception to 'pick first, then Undo'."""
    guide = (REPO / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    flat = " ".join(guide.split())
    assert "not in the Style group" not in flat
    assert "element style controls have no switch for them, except that **Residual strip**" in flat
    assert "Event waveforms keeps its **Event-count heatmap colormap**" in flat
    assert "Noise spectrum its **Spectrogram colormap** and **colour limits**" in flat
    assert "Distributions is the exception: its **Configuration** Undo takes the plot type" in flat
    assert "Auto (BIC) switched on or off" in flat
    assert "**Exports follow the figure.**" in flat
    assert "an SVG exported here names Arial first for a figure set in a Calibri this computer lacks" in flat
    assert "every text element names the font the figure is drawn in" in flat
    # the Figure fonts section is where the picker's fallback is written, and the export
    # paragraph sends the reader there
    assert "(**Figure fonts**, §10)" in flat


# ===============================================================================================
# the cheap items the second-round verifiers left
# ===============================================================================================

def test_g4_save_figure_saves_under_the_narrow_export_style_and_not_the_whole_style():
    """Verifier item (G4): nothing pinned the choice to save under the font keys only. Under the
    whole ``nanoscope_style`` the margins move (``savefig.pad_inches`` 0.1 -> 0.02) and every
    export is laid out differently. The rc values in force while ``savefig`` runs."""
    import matplotlib
    from matplotlib.figure import Figure

    from nanoscope.plot import figures
    from nanoscope.plot.style import _EXPORT_KEYS, rc_params

    matplotlib.use("Agg")
    seen: list[dict] = []
    figure = Figure(figsize=(2, 1.6))
    figure.subplots().plot([0, 1], [0, 1])
    real = figure.savefig

    def spy(*args, **kwargs):
        seen.append({key: mpl.rcParams[key] for key in rc_params()})
        return real(*args, **kwargs)

    figure.savefig = spy
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        figures.save_figure(figure, Path(folder) / "narrow")
    assert seen, "save_figure did not save through the figure"
    style = rc_params()
    for key, value in seen[0].items():
        if key in _EXPORT_KEYS:
            assert value == style[key], key
        else:
            assert value == mpl.rcParamsDefault[key], (key, value, "leaked from the whole style")
    assert style["savefig.pad_inches"] != mpl.rcParamsDefault["savefig.pad_inches"], (
        "the premise: the whole style would move the margins")
    assert seen[0]["pdf.fonttype"] == 42


def test_g3_every_r2_and_ks_token_of_a_typed_wording_becomes_the_dash():
    """Verifier item (G3): mutants that replaced only the FIRST ``{r2}`` or ``{ks}`` survived.
    The fold turns every one into the dash the old build drew."""
    words = "{equation} {r2}/{r2} | {ks}/{ks} {ks}"
    style = pp.StyleSettings().with_fit_label(FitLabelState(template=words))
    saved = {"show_r2": False, "show_gof": False, "fit_label": {"template": words}}
    folded = pp._fold_legacy_style(style, saved).fit_label_state()
    assert folded.template == "{equation} —/— | —/— —"
    assert "r2" not in folded.fields and "ks" not in folded.fields
    only_ks = pp._fold_legacy_style(style, {"show_gof": False, "fit_label": {"template": words}})
    assert only_ks.fit_label_state().template == "{equation} {r2}/{r2} | —/— —"


@pytest.mark.parametrize("flat", [False, True], ids=["a line", "a flat y"])
def test_g3_an_old_file_without_r2_on_figure_keeps_r2_in_details_and_drops_the_tag_from_the_label(
        qtbot, tmp_path, flat):
    """Verifier item (G3), named differences (a) and (b), through an old payload: with R² on
    figure unticked the label makes no R² claim (so a flat y shows no refusal tag on it), and
    Details' summary line carries R² (or the refusal) whatever the file said."""
    import json

    frame = _df()
    if flat:
        frame["ycol"] = 7.0
    page = pp.PlotsWindow(Session())
    qtbot.addWidget(page)
    page.resize(1200, 800)
    page.show()
    qtbot.waitExposed(page)
    page.set_data({"probe": frame})
    page.set_plot_type("Scatter")
    page.set_scatter_columns("xcol", "ycol")
    page.set_scatter_fit("linear")
    _settle()
    sidecar = tmp_path / "older.json"
    assert page.export_config_json(sidecar) is not None
    old = json.loads(sidecar.read_text(encoding="utf-8"))
    old["style"].update(_old_style(show_r2=False))
    sidecar.write_text(json.dumps(old), encoding="utf-8")
    assert page.import_config_json(sidecar)
    page.set_scatter_fit("linear")
    _settle()
    assert "R²" not in _label(page)
    details = page.details_text()
    assert ("R² undefined (y does not vary)" if flat else "R² = ") in details, details


def test_g1_a_linear_trend_applied_over_an_exponential_names_it_on_the_status_line(vdep):
    """Verifier item (G1): the phrase for the linear trend was never asserted."""
    page, explorer = vdep
    page.set_fit("exponential")
    _edit_double_click(page.plot_item(), page._fit_label, "Exponential: {equation}")
    assert page.fit_label_text().startswith("Exponential:")
    page.set_fit("linear")
    assert page.fit_label_text() and not page.fit_label_text().startswith("Exponential:")
    assert _said(page, explorer) == (
        "Fit label wording reset: the plot now shows the linear fit.", True)


def test_g1_switching_auto_bic_on_or_off_is_a_change_of_fit_and_takes_the_wording_back(dist):
    """Verifier item (G1): the guide's list of fit changes that take the wording back left out
    the one where the component count goes from a fixed number to Auto (BIC) or back -- a
    different fit identity -- which the guide and the changelog now name."""
    page, explorer = dist
    plot = page._plot
    page._show_control_section("Analysis")
    assert page._applied_fit == "exponential" and not page._fit_k_auto.isChecked()
    _edit_double_click(plot, page._fit_label_item, "One rate: {equation}")
    assert page.config().style.fit_label_state().template == "One rate: {equation}"
    _click_button(page._fit_k_auto)                       # Auto (BIC) on
    _click_button(page._fit_apply_btn)
    assert page._applied_fit_k_auto is True
    assert page.config().style.fit_label_state().template == ""
    assert _said(page, explorer)[0].startswith("Fit label wording reset: the plot is now set to show")
    _edit_double_click(plot, page._fit_label_item, "Chosen by BIC: {equation}")
    _click_button(page._fit_k_auto)                       # ... and off again
    _click_button(page._fit_apply_btn)
    assert page._applied_fit_k_auto is False
    assert page.config().style.fit_label_state().template == ""


def test_g3_reset_figure_in_event_waveforms_and_noise_leaves_their_colormap_rows_and_says_so(
        events, noise):
    """Verifier item (G3): the shared tip said the look goes back to its defaults, and these two
    pages keep a colormap row (Event waveforms' heatmap, Noise's spectrogram and its colour limits)
    that the shared look does not hold. Their tips say so, and the rows do stay."""
    page, explorer = events
    _show_appearance(page)
    page._cmap_combo.setCurrentText("magma")
    button = page._workflow.reset_button
    assert button.toolTip().endswith(" Here it also keeps the Event-count heatmap colormap.")
    _click_button(button)
    assert page._cmap_combo.currentText() == "magma"

    page, explorer = noise
    _show_appearance(page)
    page._spec_cmap_combo.setCurrentText("inferno")
    button = page._workflow.reset_button
    assert button.toolTip().endswith(" Here it also keeps the Spectrogram colormap and colour limits.")
    _click_button(button)
    assert page._spec_cmap_combo.currentText() == "inferno"


def test_g2_the_font_helpers_that_only_build_text_do_not_need_an_application():
    """Verifier item (G2): ``html_font_family`` (and so ``css_args`` and ``label_args`` with a
    family) asked the font database whether or not there was an application, and Qt aborts the
    process then. In a process with none they return the stack."""
    code = (
        "import nanoscope\n"
        "from nanoscope.gui.figure_text import TextStyle, css_args, label_args\n"
        "from nanoscope.gui.font_picker import html_font_family\n"
        "print(html_font_family('Times New Roman'))\n"
        "print(css_args(TextStyle(family='Menlo'))['font-family'])\n"
        "print(label_args(TextStyle(family='Menlo'))['family'])\n")
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=str(REPO), timeout=120)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()
    assert lines[0] == '"Times New Roman", Times, "Liberation Serif", Tinos, "TeX Gyre Termes", "Nimbus Roman", serif'
    assert lines[1].startswith("Menlo, Arial, Helvetica") and lines[2] == lines[1]
