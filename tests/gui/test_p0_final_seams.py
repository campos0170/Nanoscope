"""Phase 0, the final seams: what crosses G5 (a pinned range clears with the words) and G6A (the
independent review fixes), and neither group saw.

G5 was built on the integrated tree without G6A, and G6A without G5. Each was verified alone. This
file drives the places where they meet, through the routes a person uses, and reads what is DRAWN,
SAVED or WRITTEN, not a state field:

* a composition with shared axes made AFTER a quantity change took the words and a range
  (Distribution summaries, a new Measurement): the clones keep the figure's font in every text,
  the range that stayed is the range they share, the range that went does not come back, and the
  composed PDF embeds the chosen face and no private one (G6A F1 against G5);
* a Distributions project restored, then a plot-type change that takes the words and both pinned
  ranges, then Undo, Redo and a save (G6A R1, the history that starts at the saved state, against
  G5, a pin that is part of that state);
* Comparison figures: Reset figure clicked in one Section, the reset rule at Generate, and the
  Undo and Redo KEYS pressed from the other Sections (G6A F2 against the rule);
* an SVG and a PDF exported after Voltage dependence's metric took the words and the Y range: they
  still name the figure's font, a face of the main list or the application's own (G6A F3 and the
  export fonts against G5);
* the documents say one thing about ranges and about the Undo keys (the sentences G5 and G6A each
  wrote against the other).

Synthetic data only; every recording is an in-memory document or a CSV written to ``tmp_path``.
"""
from __future__ import annotations

import os
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.gui import figure_composition as fc
from nanoscope.gui import figure_text
from nanoscope.gui.analysis_explorer import PAGE_DISTRIBUTION_SUMMARIES
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.main_window import MainWindow
from nanoscope.plot import figure_fonts as ff

from .test_p0_figure_font_picker import (  # noqa: F401 -- ``arial`` is a fixture
    OTHER, arial, in_family, picking, settle)
from .test_p0_g6a_review_fixes import (  # noqa: F401 -- ``application_font`` is a fixture
    _click_through_window, _composed_pdf_fonts, _texts_drawn, _tick_families, application_font)
from .test_p0_integration import (
    _export_through_the_menu, _pdf_is_clean, _pick_font, _show_appearance, _svg_families)
from .test_p0_pins_clear_with_the_words import (
    _axes_of, _depth, _is_automatic, _pin_inside, _press_redo, _stays_pinned)
from .test_p0_reset_on_quantity_change import (  # noqa: F401  (the fixtures build the pages)
    _generate, _pick, _said, _section, _type_comparison_words, _type_summary_words,
    _type_vdep_words, _user_picks, comparison, summaries, vdep)
from .test_plot_window_routes import (
    _add_title, _edit_double_click, _install_series, _main_window, _press_undo,
    _restored_main_window)
from .test_plots_reopen_figure import _main, _recordings, _settle

REPO = Path(__file__).resolve().parents[2]
TIMES = "Times New Roman"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


# ================================================================================================
# 1. a composition made after a quantity change keeps the font, and shares the range that stayed
# ================================================================================================

@pytest.mark.parametrize("shared_legend", [False, True])
def test_a_shared_axes_composition_made_after_a_quantity_reset_keeps_the_font_and_the_ranges_that_stayed(
        summaries, arial, shared_legend, qtbot, tmp_path):
    """Distribution summaries in Times New Roman, with words typed and BOTH axes pinned. A new
    Measurement, committed by Generate, takes the words and the value range (G5) and leaves the
    font and the fraction range. The figure is saved to the Figure Library twice (the app's own
    route) and composed with Share X / Share Y in the real Compose dialog: each panel is re-drawn
    from its stored model (G6A), and the model is the figure AS IT IS NOW -- its title (automatic
    again), legend rows, axis labels and ticks in Times New Roman, the fraction axis on the pin
    that stayed, the value axis on the new data and not on the range that went."""
    window, page, explorer = summaries
    plot, widget = page._plot, page._appearance_widget
    _show_appearance(page)
    _pick_font(widget.font_family, TIMES)
    _type_summary_words(page)
    value = _pin_inside(plot, "bottom")
    fraction = _pin_inside(plot, "left")
    _section(page, "Data")
    _pick(page._metric_combo, data="depth_pA")
    _generate(page)

    sentence, on_screen = _said(page, explorer)
    assert on_screen and sentence == (
        "Title, X label, Y label and value range reset: the plot now shows the ECDF of median depth.")
    assert widget.state().font_family == TIMES, "the font is look: the rule leaves it"
    assert set(_axes_of(page)) == {"curves.left"}, "the value pin went, the fraction pin stayed"
    _is_automatic(plot, "bottom", was=value)
    _stays_pinned(plot, "left", fraction)
    live = _texts_drawn(plot.scene())
    assert {"ECDF"} <= set(live), live
    assert all(in_family(fams, TIMES) for fams in live.values()), live

    try:
        entries = [window._save_analysis_figure(PAGE_DISTRIBUTION_SUMMARIES, name)
                   for name in ("ECDF A", "ECDF B")]
        assert all(entries)
        for entry in entries:
            model = fc._model(FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs))
            assert model["ranges"][1] == pytest.approx(fraction, rel=1e-3), "the pin that stayed"
            assert model["ranges"][0] != pytest.approx(value, rel=1e-3), "the pin that went"
            clone = fc._clone_model(model, model["canvas"], model["ranges"])
            qtbot.addWidget(clone)
            clone.show()
            QtWidgets.QApplication.processEvents()
            drawn = _texts_drawn(clone.scene())
            assert set(drawn) == set(live), (sorted(drawn), sorted(live))
            for text, fams in drawn.items():
                assert in_family(fams, TIMES), f"{text!r} is drawn in {sorted(fams)} in the clone"
                assert fams == live[text], (text, fams, live[text])
            assert _tick_families(clone.getPlotItem()) == {TIMES}
            assert clone.getPlotItem().getViewBox().viewRange()[1] == pytest.approx(fraction, rel=1e-3)

        names = _composed_pdf_fonts(entries, qtbot, tmp_path, f"after-reset-{shared_legend}",
                                    shared_legend=shared_legend)
        assert names and not [n for n in names if n.split("+")[-1].startswith(".")], names
        assert any("TimesNewRoman" in n for n in names), names
        assert not any(n.split("+")[-1] in ("ArialMT", "Helvetica") for n in names), names
    finally:
        if window._figure_library is not None:
            window._figure_library.close()


# ================================================================================================
# 2. Distributions: restored, a change that takes the words and the pins, then Undo / Redo / save
# ================================================================================================

def _pinned_distributions_project(qtbot, tmp_path):
    """A project saved with a Distributions histogram fit styled the way a lab would (a typed
    title, Times New Roman) and BOTH axes pinned in Axis properties, opened in a fresh main
    window. Returns ``(writer window, opened window, its page, x pin, y pin)``."""
    window, plots = _recordings(qtbot, tmp_path)
    explorer = window._analysis_explorer
    explorer.resize(1280, 880)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    plots.set_title("Dwell, pore A")
    plots.set_font_family(TIMES)
    QtWidgets.QApplication.processEvents()
    plots.activateWindow()
    x_pin = _pin_inside(plots._plot, "bottom")
    y_pin = _pin_inside(plots._plot, "left")
    plots._history_touch()
    _settle()
    saved = tmp_path / "first.nanoscope"
    assert window._write_project(str(saved))
    window._dirty = False
    again = _main(qtbot)
    with qtbot.waitSignal(again.projectRestored, timeout=20000):
        again._open_project(str(saved))
    again._open_plots()
    _settle()
    again._analysis_explorer.resize(1280, 880)
    page = again._plots_window
    page.activateWindow()
    QtWidgets.QApplication.processEvents()
    return window, again, page, x_pin, y_pin


def _as_drawn(page) -> dict:
    """The figure as it is: the plot type on the combo, the title text, the font and the two
    typed ranges of the configuration."""
    config = page.config()
    return {"plot": page._plot_type_combo.currentText(),
            "title": page._plot.titleLabel.item.toPlainText().strip(),
            "font": config.style.font_family,
            "ranges": config.axes.ranges()}


def test_a_restored_distributions_project_undoes_and_redoes_a_change_that_took_the_words_and_the_pins(
        qapp, qtbot, tmp_path):
    """The history of an opened project starts at the state it was saved in (G6A R1); a pin is
    part of that state (G5). So the first change after opening -- here a new plot type, which
    takes the title and both pinned ranges and says so -- is one step that Undo takes back to
    the saved title, font AND ranges (drawn pinned again), Redo takes again, and a project saved
    after an Undo holds what is on screen, pins included."""
    window, again, page, x_pin, y_pin = _pinned_distributions_project(qtbot, tmp_path)
    explorer = again._analysis_explorer
    plot = page._plot
    saved = _as_drawn(page)
    assert saved == {"plot": "Histogram", "title": "Dwell, pore A", "font": TIMES,
                     "ranges": ("X range", "Y range")}
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)
    assert page._history.depth() == (0, 0) and not page._undo_btn.isEnabled()

    _user_picks(page._plot_type_combo, "Scatter")                 # the rule: words and both ranges
    assert page._history.depth() == (1, 0), "one step"
    assert page.config().style.title == "" and page.config().axes.ranges() == ()
    assert page.config().style.font_family == TIMES, "the font is look"
    sentence, on_screen = _said(page, explorer)
    assert on_screen and sentence.startswith(
        "Title, X range and Y range reset: the plot is now set to show the scatter of"), sentence
    page._plot_btn.click()                                        # the picture follows at Plot
    QtWidgets.QApplication.processEvents()
    _is_automatic(plot, "bottom", was=x_pin, auto_ranging=False)
    _is_automatic(plot, "left", was=y_pin, auto_ranging=False)

    _press_undo(page)                                             # Undo alone, the Undo key
    _settle()
    assert _as_drawn(page) == saved, "the saved title, the font and BOTH pins come back"
    assert page._history.depth() == (0, 1)
    _stays_pinned(plot, "bottom", x_pin)
    _stays_pinned(plot, "left", y_pin)

    _press_redo(page)
    _settle()
    assert page._history.depth() == (1, 0)
    assert page._plot_type_combo.currentText() == "Scatter"
    assert page.config().style.title == "" and page.config().axes.ranges() == ()
    assert page.config().style.font_family == TIMES

    _press_undo(page)                                             # and save with the Undo in effect
    _settle()
    assert again._dirty, "an Undo changes the project"
    second = tmp_path / "second.nanoscope"
    assert again._write_project(str(second))
    again._dirty = False
    reopened = _main(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20000):
        reopened._open_project(str(second))
    reopened._open_plots()
    _settle()
    reopened._analysis_explorer.resize(1280, 880)
    again_page = reopened._plots_window
    again_page.activateWindow()
    QtWidgets.QApplication.processEvents()
    assert _as_drawn(again_page) == saved, "the file holds what was on screen, pins included"
    _stays_pinned(again_page._plot, "bottom", x_pin)
    _stays_pinned(again_page._plot, "left", y_pin)
    assert again_page._history.depth() == (0, 0), "and it opens with a history of its own"
    window._dirty = again._dirty = reopened._dirty = False


def test_voltage_dependence_a_reopened_project_undoes_a_metric_change_to_the_saved_words_and_range(
        qapp, qtbot, tmp_path, monkeypatch):
    """The other pages keep their appearance in the shared widget, whose history a restore clears
    and re-seeds at the saved state (G6A's R1 was Distributions; nobody had driven this one). The
    saved state holds a typed title, a Y label and a pinned Y range; the first metric change
    after opening takes all three in one step (G5), Undo gives all three back and draws the
    range, Redo takes them again, and the file saved after an Undo holds them."""
    from .test_p0_reset_on_quantity_change import _label

    window = _restored_main_window(qtbot, tmp_path, monkeypatch)
    _install_series(window, ())
    window._open_vdep()
    page = window._vdep_window
    window._analysis_explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    page.activateWindow()
    plot = page.plot_item()
    _add_title(plot, "Capture vs voltage")
    _edit_double_click(plot, plot.getAxis("left").label, "Rate (Hz)")
    y_pin = _pin_inside(plot, "left")
    project = tmp_path / "vdep.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    def reopen(path):
        again = _main_window(qtbot)
        with qtbot.waitSignal(again.projectRestored, timeout=20_000):
            again._open_project(str(path))
        again._open_vdep()
        QtWidgets.QApplication.processEvents()
        again._analysis_explorer.resize(1280, 860)
        QtWidgets.QApplication.processEvents()
        again._vdep_window.activateWindow()
        return again, again._vdep_window

    reopened, page = reopen(project)
    plot, widget = page.plot_item(), page._appearance_widget

    def as_saved():
        QtWidgets.QApplication.processEvents()
        return (widget.state().title, widget.state().y_label, _label(plot, "left"),
                tuple(_axes_of(page)))

    saved = as_saved()
    assert saved == ("Capture vs voltage", "Rate (Hz)", "Rate (Hz)", ("left",))
    _stays_pinned(plot, "left", y_pin)
    assert _depth(widget) == (0, 0)
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")
    assert _depth(widget) == (1, 0), "one step"
    assert widget.state().title is None and not _axes_of(page)
    assert _said(page, reopened._analysis_explorer)[0] == (
        "Title, Y label and Y range reset: the plot now shows median depth.")
    _is_automatic(plot, "left", was=y_pin)

    _press_undo(page)
    assert as_saved() == saved and _depth(widget) == (0, 1)
    _stays_pinned(plot, "left", y_pin)
    _press_redo(page)
    assert _depth(widget) == (1, 0) and widget.state().title is None and not _axes_of(page)
    _press_undo(page)
    assert reopened._dirty, "an Undo changes the project"
    second = tmp_path / "vdep-second.nanoscope"
    assert reopened._write_project(str(second))
    reopened._dirty = False
    last, page = reopen(second)
    plot, widget = page.plot_item(), page._appearance_widget
    assert as_saved() == saved, "the file holds what was on screen"
    _stays_pinned(plot, "left", y_pin)
    last._dirty = reopened._dirty = window._dirty = False


# ================================================================================================
# 3. Comparison figures: the Undo and Redo keys from every Section
# ================================================================================================

def _spy_on_title_family(monkeypatch) -> list:
    """The font family list the last render gave the matplotlib figure's title, one per render.
    (``comparison`` already spies on the words; this wraps it.)"""
    from nanoscope.plot import comparison_figures as renderer

    seen: list = []
    inner = renderer.render_comparison

    def spy(*args, **kwargs):
        figure = inner(*args, **kwargs)
        seen.append(list(figure._suptitle.get_fontfamily()))
        return figure

    monkeypatch.setattr(renderer, "render_comparison", spy)
    return seen


def _drawn_state(drawn, families) -> tuple:
    """What the last render drew: the suptitle, its size and font first-name, the axis labels."""
    last = drawn[-1]
    return (last["title"], last["size"], families[-1][0], last["x"], last["y"])


def test_comparison_keys_undo_and_redo_a_reset_figure_and_the_rules_reset_from_other_sections(
        comparison, arial, monkeypatch):
    """Reset figure is in the toolbar of every Section, and the rule resets at Generate; the Undo
    and Redo keys now answer from every Section too (G6A F2). Words and a look are typed in
    Appearance, then: Reset figure from Data, Undo from Export, Redo from Analysis; the rule's
    reset at Generate from Analysis, Reset figure from Export, and the three steps taken back and
    forward from three other Sections. What the matplotlib figure was asked to draw decides."""
    page, explorer, drawn = comparison
    families = _spy_on_title_family(monkeypatch)
    widget = page._appearance_widget
    _section(page, "Appearance")
    _pick_font(widget.font_family, TIMES)
    _type_comparison_words(page)
    typed = _drawn_state(drawn, families)
    assert typed == ("Mine", 14.0, TIMES, ["Dw"], ["Dp"])
    base = widget._history.depth()[0]

    def at(section, act):
        _section(page, section)
        assert not widget.isVisible() or section == "Appearance", f"premise: {section} hides Appearance"
        act()

    at("Data", lambda: _click_through_window(page._workflow.reset_button))
    assert widget._history.depth() == (base + 1, 0)
    reset = _drawn_state(drawn, families)
    assert reset[:3] == ("Recording comparisons", reset[1], "Arial") and reset[1] != 14.0, reset
    assert reset[3] != ["Dw"] and reset[4] != ["Dp"]

    at("Export", lambda: _press_undo(page))
    assert _drawn_state(drawn, families) == typed, "Undo from Export"
    assert widget._history.depth() == (base, 1)
    at("Analysis", lambda: _press_redo(page))
    assert _drawn_state(drawn, families) == reset, "Redo from Analysis"
    assert widget._history.depth() == (base + 1, 0)
    at("Data", lambda: _press_undo(page))
    assert _drawn_state(drawn, families) == typed

    # The rule's reset at Generate (words go, the look stays), then Reset figure on what is left.
    _section(page, "Analysis")
    _pick(page._y_combo, data="amplitude_pA")
    _generate(page)
    ruled = _drawn_state(drawn, families)
    assert ruled[:3] == ("Recording comparisons", 14.0, TIMES), "the look stays"
    assert ruled[3] != ["Dw"] and ruled[4] != ["Dp"]
    assert widget._history.depth() == (base + 1, 0)
    at("Export", lambda: _click_through_window(page._workflow.reset_button))
    assert widget._history.depth() == (base + 2, 0)
    assert _drawn_state(drawn, families)[1:3] != (14.0, TIMES), "the look went"

    at("Data", lambda: _press_undo(page))                         # the look comes back ...
    assert _drawn_state(drawn, families) == ruled
    at("Appearance", lambda: _press_undo(page))                   # ... and then the words
    assert _drawn_state(drawn, families) == typed
    assert widget._history.depth() == (base, 2)
    at("Export", lambda: _press_redo(page))
    assert _drawn_state(drawn, families) == ruled
    assert widget._history.depth() == (base + 1, 1)


# ================================================================================================
# 4. an SVG and a PDF exported after the words and a range went still name the figure's font
# ================================================================================================

@pytest.mark.parametrize("how", ["a face of the main list", "the application's own font"])
def test_an_svg_and_a_pdf_exported_after_a_reset_of_words_and_a_range_name_the_figures_font(
        how, vdep, arial, application_font, monkeypatch, qtbot, tmp_path):
    """Voltage dependence in Times New Roman (a row of the list) or in Menlo, which is also the
    application's own font here (the case of Windows' Segoe UI: a family someone picked is named
    first, G6A F3). Words are typed, both axes are pinned; the Metric takes the words and the Y
    range (G5). The SVG then names the face on EVERY text -- the labels the new metric draws
    included -- and the PDF embeds it; Reset figure puts Arial's stack back, whatever the
    application's font is."""
    page, explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    if how == "the application's own font":
        application_font(OTHER)
    _show_appearance(page)
    picker = widget.font_family
    if how == "a face of the main list":
        family, face = TIMES, "TimesNewRoman"
        _pick_font(picker, TIMES)
    else:
        family, face = OTHER, OTHER
        picking(monkeypatch, OTHER)
        _pick_font(picker, "Other font…")
        settle(qtbot, picker)
    assert widget.state().font_family == family
    stack = ff.css_font_family(family)

    _type_vdep_words(page)                                        # also pins the X range
    y_pin = _pin_inside(plot, "left")
    x_before = _axes_of(page)["bottom"]
    steps = _depth(widget)
    _section(page, "Analysis")
    _pick(page._metric_combo, text="Median depth (pA)")

    assert _depth(widget) == (steps[0] + 1, 0)
    sentence, on_screen = _said(page, explorer)
    assert on_screen and sentence.startswith("Title, X label, Y label") and "Y range" in sentence, sentence
    assert _axes_of(page) == {"bottom": x_before}, "the Y pin went, the X pin (the voltage) stayed"
    _is_automatic(plot, "left", was=y_pin)
    assert widget.state().font_family == family

    svg = _export_through_the_menu(page, monkeypatch, tmp_path / "after.svg")
    families = _svg_families(svg)
    assert families and set(families) == {stack}, set(families)
    pdf = _export_through_the_menu(page, monkeypatch, tmp_path / "after.pdf", ".pdf")
    assert any(face in name for name in _pdf_is_clean(pdf)), "the PDF is in the same font"

    _click_through_window(page._workflow.reset_button)            # Reset figure: Arial again
    assert widget.state().font_family is None
    plain = _export_through_the_menu(page, monkeypatch, tmp_path / "plain.svg")
    assert set(_svg_families(plain)) == {ff.css_font_family(None)}
    plain_pdf = _export_through_the_menu(page, monkeypatch, tmp_path / "plain.pdf", ".pdf")
    assert any("Arial" in name for name in _pdf_is_clean(plain_pdf))


# ================================================================================================
# 5. the documents say one thing about ranges and about the Undo keys
# ================================================================================================

def _flat(relative: str) -> str:
    return " ".join((REPO / relative).read_text(encoding="utf-8").split())


def test_the_documents_say_one_thing_about_pinned_ranges_and_the_undo_keys():
    """G5 wrote that a pinned range goes with the words when its axis changes quantity; G6A, on a
    tree without it, wrote that axis ranges stay and that Distributions alone clears one, and
    that the Undo keys of every plot window answer 'while the Appearance controls are on screen'
    (which stopped being true of Comparison figures with its F2). The merged documents keep the
    later rule and the true keys."""
    guide, changelog = _flat("docs/USER_GUIDE.md"), _flat("CHANGELOG.md")
    inventory = _flat("docs/inventory.json")

    # Ranges: the look-stays sentence names only the ranges of axes that did not move.
    old_list = "text styles, axis ranges typed in Axis properties, the legend"
    assert old_list not in guide + changelog + inventory
    assert ("text styles, the legend, the ranges pinned on axes whose quantity did not move"
            in guide)
    assert ("text styles, the legend, the ranges pinned on axes whose quantity did not move"
            in changelog)
    assert "Distributions clears the typed range of an axis whose quantity moved, as it always has" \
        not in changelog
    assert "Distributions also clears the typed range of an axis whose quantity moved" not in guide
    assert ("an axis whose column moved loses its typed range (a change of plot type moves both "
            "axes)") in guide
    assert "an axis whose quantity moved loses its typed range" not in guide
    # ... and a range typed in Axis properties does not stay for ever.
    assert "until the axis comes to show another quantity" in guide
    assert "until the axis comes to show another quantity" in changelog

    # The Undo keys: every plot window but the study figure editor answers from every Section.
    assert "with ⌘Z / ⇧⌘Z while the Appearance controls are on screen" not in changelog
    assert "In Comparison figures and the study figure editor the keys answer" not in guide
    assert "(the study figure editor, which has no Sections, answers while its Appearance " \
        "controls are on screen)" in changelog
    assert "Comparison figures: ⌘Z answers from every Section" in changelog

    # The compose sentence: true where the application font is a private face, and says what
    # happens where it is an ordinary family.
    for text, other in ((guide, "**Other font…**"), (changelog, "Other font…")):
        assert "on a Mac, where the application font is a private face, the list has no row for it" in text
        assert f"where it is an ordinary family such as Segoe UI, {other} lists it" in text

    # Reset figure: the Undo tip names a step from what it took back (a lone title reads as that
    # title), so the CHANGELOG gives two examples and does not promise "reset figure" for every
    # reset.
    assert "button reads *Undo: reset figure*" not in changelog
    assert ("button names the step from what it took back: *Undo: reset figure* when a title, a "
            "label and a look went back together, *Undo: title ‘Capture vs voltage’ → automatic* "
            "when only the title did") in changelog


def test_the_documents_say_the_same_about_ticking_a_source_that_lacks_the_plotted_column():
    """Distributions' column boxes offer only the columns every ticked source has, so ticking a
    source without the plotted column moves it and the rule runs (tests/gui/
    test_p0_reset_on_quantity_change.py). The guide, the CHANGELOG, the inventory and the rule's
    own docstring each say so where they say that ticking another source keeps the words."""
    from nanoscope.gui import quantity_reset

    guide, changelog = _flat("docs/USER_GUIDE.md"), _flat("CHANGELOG.md")
    inventory = _flat("docs/inventory.json")
    docstring = " ".join(quantity_reset.__doc__.split())
    assert "unless a ticked source then lacks the plotted column" in guide
    assert "ticking another source that has the plotted column, applying a preset" in guide
    assert "A column also moves when you tick a source that lacks it" in guide
    assert "ticking another source that has the plotted column, applying a preset" in changelog
    assert "A column also moves on its own when you tick a source that lacks it" in changelog
    assert "ticking another source that has the plotted column, keeps the words" in docstring
    # The Data sources note, the clustering and PCA gotcha, and the X min note.
    assert inventory.count("unless the new row set no longer carries the plotted column") == 3
    for text in (guide, changelog):          # the unqualified sentence of the first draft is gone
        assert "ticking another source, applying a preset" not in text
        assert "Ticking another source and changing the Time unit keep the words" not in text


def test_the_changelog_carries_each_phase_0_change_once_and_the_inventory_agrees_on_sources():
    """The groups each wrote their own entries; the merged set has one entry per change: G5's
    pinned-range entry lives inside the reset rule's, G6A's row, caption and spacing entries
    inside the font list's and the Reset figure entry, and what two groups both said (the SVG of
    a picked application font, the Text style font box's tool tip) is said where it belongs."""
    changelog, inventory = _flat("CHANGELOG.md"), _flat("docs/inventory.json")
    once = (
        "**Words you typed, and ranges you pinned, go back to automatic when a figure's quantity "
        "changes",
        "**The Title, X label and Y label boxes say what a typed title covers**",
        "**Distributions: Undo on a project you opened goes back to what the project was saved with.**",
        "**A composition with shared axes keeps the figure's font in the title and the legend.**",
        "**Comparison figures: ⌘Z answers from every Section.**",
        "(Voltage dependence's caption, the words on Distributions' detection-limit line)",
        "It leaves out the faces of symbols (Symbol, Webdings, Wingdings, Wingdings 2 and 3, Zapf "
        "Dingbats, Apple Symbols)",
        "A source you renamed on its legend row (Distributions, Distribution summaries) keeps its name",
        "in a wide strip such as Voltage dependence's or I-V's they keep that spacing",
        "**A project saved by an older build** whose appearance holds a range with no view",
        "the ranges pinned on axes whose quantity did not move",
        "None of them takes a range off either, except that |V|",
        # The report figures' font list is told once (Changed), not again in the Fixed entry about
        # the face their text is printed in.
        "**Matplotlib figures draw in a list that starts with Arial.**",
        "**The report figures' text is printed in the face it was laid out in.**",
        "Liberation Sans, Arimo, TeX Gyre Heros and Nimbus Sans",
    )
    for sentence in once:
        assert changelog.count(sentence) == 1, (changelog.count(sentence), sentence)
    gone = (      # the titles of the fragments the groups handed over, now folded in
        "A range pinned on an axis goes back to automatic with the words when that axis changes quantity",
        "A font row says what it is drawn in when this computer lacks the font (plotting redesign",
        "Reset figure's tool tip, the guide and the control list name the caption",
        "The six Numbers boxes of the Fit label keep their natural spacing",
        "An SVG names a font you picked even when it is the application's own",
        "Text style…'s font box no longer calls its explicit Arial the figure default",
        "**Which axis moves, window by window.**",
    )
    for sentence in gone:
        assert sentence not in changelog, sentence
    # The inventory's three sentences about ticking a source said it clears a pinned range (two
    # in those words, the X min note in others); it does not (the code's own comment: 'NO
    # BLANKET UNPIN HERE ANY MORE'), and the rule's decision 2 is that another source is not
    # another quantity.
    assert "AND clears any pinned Axis range" not in inventory
    assert "and clears any pinned Axis range, because the row set changed underneath them" not in inventory
    assert ("keeps a pinned Axis range and the words you typed (another source is not another "
            "quantity)") in inventory
    assert ("a pinned Axis range and the words you typed stay (another source is not another "
            "quantity)") in inventory
    # ... and the X min note, a third copy of the same claim in other words.
    assert "change which sources are ticked" not in inventory
    assert ("ticking another source keeps them (another source is not another quantity) unless "
            "the new row set no longer carries the plotted column") in inventory
    assert "that persist across redraws, until the plot type or the axis's column changes" in _flat(
        "docs/USER_GUIDE.md")


def test_every_explorer_page_binds_the_undo_keys_to_the_whole_page_and_the_study_editor_does_not():
    """The CHANGELOG and the guide say the Undo and Redo keys of a plot window answer whatever the
    page's Section shows, except in the study figure editor, which has no Sections and answers
    while its Appearance controls are on screen. That is what the pages are built with: seven of
    the Explorer's eight pages hand their own root to the widget as ``shortcut_parent`` (G6A
    added the seventh, Comparison figures); the eighth, Distributions, has no widget history (its
    Configuration Undo is the page's); the study figure editor, which is not an Explorer page,
    hands nothing. That makes nine ``PlotAppearanceWidget`` calls in all."""
    import ast

    def keywords(module: str) -> list[dict]:
        tree = ast.parse((REPO / "nanoscope" / "gui" / f"{module}.py").read_text(encoding="utf-8"))
        return [{k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", ""))
                == "PlotAppearanceWidget"]

    for module in ("comparison_figures_panel", "distribution_summary_panel", "event_plot_panel",
                   "iv_panel", "psd_panel", "subpop_window", "voltage_dep_panel"):
        built = keywords(module)
        assert len(built) == 1 and built[0].get("shortcut_parent") == "self", (module, built)
    (study,) = keywords("study_figure_window")
    assert "shortcut_parent" not in study, study
    (plots,) = keywords("plots_panel")
    assert plots.get("history") == "False", plots
