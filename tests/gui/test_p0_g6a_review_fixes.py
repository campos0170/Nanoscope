"""Phase 0, group G6A: the independent fixes from the cross-group review.

Each fix is driven the way a person reaches it and checked against what is DRAWN or SAVED, not a
state field:

* F1 (fonts) -- a shared-axes composition re-draws each panel from its stored model; the clone's
  title and legend rows keep the figure's font (they came back in the application font, a private
  face the composed PDF then embedded), and a saved model that carries an unsafe family is refused;
* R1 (Distributions) -- a page restored from a project starts its Undo history at the saved state,
  and an Undo or Redo is a durable edit (the file saved after it holds what is on screen);
* F2 (Comparison figures) -- the Undo key answers from every Section, so Reset figure (in the
  toolbar) and the rule's reset at Generate can be taken back from where they were made;
* F3 (SVG) -- the SVG names a picked family first even when it is the application's own font;
* F5 / F6 (the picker) -- a main-list row says what a family this computer lacks is drawn in, and
  the Text style… box's tool tip stops calling its explicit Arial the figure default;
* the Other font… list leaves the faces of symbols out (an older project's value still shows);
* D3 (wording) -- the Title caption and the Add title entry say what a title covers; D6 -- Reset
  figure's tool tip says it also takes back a caption; L1 / C1 -- the Numbers grid and where the
  widget keeps its status sentence;
* the two matplotlib windows' Reset figure redraws what is drawn, and a window handle that PySide
  wrapped as the wrong type is replaced before a gesture is sent to it.

Synthetic data only; offscreen.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.core.analysis_workspace import PLOTS_ARTIFACT_ID, capture_plots, restore_plots
from nanoscope.core.figure_library import make_figure
from nanoscope.core.project import WorkspaceArtifact
from nanoscope.gui import figure_composition as fc
from nanoscope.gui import figure_text, font_picker
from nanoscope.gui.figure_bundle import capture_figure_snapshot, write_frozen_figure_image
from nanoscope.gui.figure_composer import FigureComposerDialog
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_export_options import resolve_export_options
from nanoscope.plot import figure_fonts as ff

from .test_p0_distributions_switches_and_reset_figure import _distributions, _with_old_keys_added
from .test_p0_figure_font_picker import (  # noqa: F401 -- ``arial`` is a fixture
    OTHER, add_title, arial, drawn_families, in_family, pdf_font_names, picking, settle)
from .test_p0_integration import (
    _export_through_the_menu, _pdf_is_clean, _pick_font, _show_appearance, _svg_families)
from .test_p0_reset_on_quantity_change import (  # noqa: F401 -- the fixtures build the pages
    _generate, _pick, _section, _type_comparison_words, _user_picks, comparison, dist, noise,
    summaries, vdep)
from .test_plot_window_routes import (
    _build, _choose, _click, _edit_double_click, _names, _press_undo, window_handle)
from .test_plots_reopen_figure import _main, _recordings, _settle

TIMES = "Times New Roman"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


# ================================================================================================
# a window handle PySide wrapped as the wrong type
# ================================================================================================

def test_window_handle_replaces_a_wrapper_of_the_wrong_type_at_the_windows_address(qapp, qtbot):
    """The review saw ``QTest.mouseMove(QWidgetItem, QPoint)`` fail once: ``windowHandle()`` had
    handed back a stale wrapper of another type. The same thing is staged here -- a QLayoutItem
    wrapper registered at the QWindow's address -- and the helper the gestures use gets the
    QWindow, where the bare call would not."""
    top = QtWidgets.QWidget()
    qtbot.addWidget(top)
    top.resize(220, 120)
    top.show()
    qtbot.waitExposed(top)
    real = top.windowHandle()
    address = shiboken6.getCppPointer(real)[0]
    shiboken6.invalidate(real)
    stale = shiboken6.wrapInstance(address, QtWidgets.QLayoutItem)
    assert top.windowHandle() is stale and not isinstance(stale, QtGui.QWindow), "the premise"
    with pytest.raises(TypeError):
        QtTest.QTest.mouseMove(top.windowHandle(), QtCore.QPoint(5, 5))

    child = QtWidgets.QLabel("x", top)                 # any widget of the window will do
    handle = window_handle(child)
    assert isinstance(handle, QtGui.QWindow) and handle is not stale
    QtTest.QTest.mouseMove(handle, QtCore.QPoint(5, 5))      # and a gesture reaches it
    assert window_handle(child) is handle, "asked again, the same window"
    assert window_handle(QtWidgets.QWidget()) is None, "a window that was never shown has none"


# ================================================================================================
# F1 -- a shared-axes composition keeps the figure's font in the title and the legend rows
# ================================================================================================

def _texts_drawn(scene) -> dict[str, set[str]]:
    """``{text: families its glyph runs are laid out in}`` for every visible text item of a
    scene -- title, axis labels, legend rows -- read off what Qt lays out."""
    return {item.toPlainText().strip(): drawn_families(item)
            for item in scene.items()
            if isinstance(item, QtWidgets.QGraphicsTextItem) and item.isVisible()
            and item.toPlainText().strip()}


def _tick_families(plot_item) -> set[str]:
    out = set()
    for side in ("bottom", "left"):
        font = plot_item.getAxis(side).style.get("tickFont")
        out.add(QtGui.QFontInfo(font if font is not None else QtWidgets.QApplication.font()).family())
    return out


def _library_entry(page_id: str, name: str, plot_item):
    """A Figure Library entry for ``plot_item`` as ``MainWindow._capture_figure`` makes it: the
    frozen picture, with the Cartesian model that lets the composer share axes."""
    model = fc.capture_cartesian_model(plot_item)
    assert model is not None, "the figure has no replayable model (a premise of this test)"
    snapshot = capture_figure_snapshot(
        plot_item, data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
        config={"source": name}, provenance={"source": name},
        additional_resources={fc.CARTESIAN_RESOURCE: model})
    metadata, blobs = snapshot.to_storage()
    return make_figure(name=name, page_id=page_id, source_label=name,
                       analysis=WorkspaceArtifact(artifact_id=page_id, kind="test", payload={}),
                       snapshot_metadata=metadata, snapshot_blobs=blobs, live_state=None)


def _summaries_page(qtbot, tmp_path, family):
    """A Distribution summaries ECDF with a typed title and a legend, in ``family``."""
    page, plot, _edit, _shown = _build("summaries", qtbot, tmp_path)
    page.set_appearance_state({"font_family": family})
    QtWidgets.QApplication.processEvents()
    add_title(plot, "Dwell ECDF")
    return page, plot


@pytest.mark.parametrize("family", ["", TIMES])
def test_a_shared_axes_clone_draws_its_title_and_legend_in_the_figure_font(
        family, qapp, qtbot, tmp_path, arial):
    """A composition with Share X / Y re-draws each panel from the stored model. The live figure
    is Arial (or the picked family) in every text; the clone's title and legend rows came back in
    the application font, a private face on macOS, beside axis labels and ticks in the right one."""
    page, plot = _summaries_page(qtbot, tmp_path, family)
    want = family or "Arial"
    assert plot.legend is not None and plot.legend.items, "premise: the figure has a legend"
    live = _texts_drawn(plot.scene())
    assert {"Dwell ECDF"} <= set(live), live
    assert all(in_family(fams, want) for fams in live.values()), live

    model = json.loads(fc.capture_cartesian_model(plot))
    clone = fc._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    clone.show()
    QtWidgets.QApplication.processEvents()
    drawn = _texts_drawn(clone.scene())
    assert set(drawn) == set(live), (drawn, live)
    for text, fams in drawn.items():
        assert in_family(fams, want), f"{text!r} is drawn in {sorted(fams)} in the clone, not {want}"
        assert fams == live[text], (text, fams, live[text])
    assert _tick_families(clone.getPlotItem()) == _tick_families(plot) == {want}


def _composed_pdf_fonts(entries, qtbot, tmp_path, name, *, shared_legend=False) -> list[str]:
    """Compose ``entries`` in the real Compose dialog with Share X / Y (and, when asked, One
    shared legend), export the composition's PDF as the Library does, and return the font names
    the file embeds."""
    dialog = FigureComposerDialog(entries, current_id=entries[0].artifact_id)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog._share_x.setChecked(True)
    dialog._share_y.setChecked(True)
    dialog._shared_legend.setChecked(shared_legend)
    dialog.update_preview()
    snapshot = dialog.snapshot
    assert snapshot is not None, dialog._message.text()
    target = tmp_path / f"{name}.pdf"
    write_frozen_figure_image(target, snapshot, resolve_export_options(
        (snapshot.width, snapshot.height), extension=".pdf"))
    return sorted(pdf_font_names(target))


@pytest.mark.parametrize("shared_legend", [False, True])
@pytest.mark.parametrize("family", ["", TIMES])
def test_a_shared_axes_composition_embeds_no_private_face_and_the_figures_font(
        family, shared_legend, qapp, qtbot, tmp_path, arial):
    """The composed PDF of two Distribution summaries ECDFs with Share X / Y: the title and the
    legend are in the font of the panels. They were the application font, which the PDF
    embedded as ``.SFNS-Regular`` on macOS -- three formats of one figure that disagreed."""
    _page, plot = _summaries_page(qtbot, tmp_path, family)
    entries = [_library_entry("test", "ECDF A", plot), _library_entry("test", "ECDF B", plot)]
    names = _composed_pdf_fonts(entries, qtbot, tmp_path, f"shared-{family or 'arial'}-{shared_legend}",
                                shared_legend=shared_legend)
    assert names and not [n for n in names if n.split("+")[-1].startswith(".")], names
    wanted = "TimesNewRoman" if family else "Arial"
    assert any(wanted in n for n in names), (family, names)
    if family:
        assert not any(n.split("+")[-1] in ("ArialMT", "Helvetica") for n in names), \
            f"a panel text is not in {family}: {names}"


def test_a_distributions_scatter_legend_rows_keep_the_figures_font_in_the_clone(
        dist, qtbot, arial):
    """Distributions: a scatter of two sources. The title travels by the QFont route and was
    right; the legend rows (a family in their HTML, which the model did not keep) were not."""
    page, explorer = dist
    page.set_sources(["probe", "other"])
    page.set_plot_type("Scatter")
    page.set_scatter_columns("xcol", "ycol")
    page.set_scatter_fit("none")
    page.set_font_family(TIMES)
    QtWidgets.QApplication.processEvents()
    plot = page.plot_item()
    add_title(plot, "Depth against dwell")
    assert plot.legend is not None and len(plot.legend.items) == 2, "premise: two legend rows"
    live = _texts_drawn(plot.scene())
    rows = [label.item.toPlainText().strip() for _sample, label in plot.legend.items]
    assert all(in_family(live[row], TIMES) for row in rows), live

    model = json.loads(fc.capture_cartesian_model(plot))
    assert [entry["family"] for entry in model["legend"]["entries"]] == [
        label.opts["family"] for _sample, label in plot.legend.items], "the rows' family is kept"
    clone = fc._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    clone.show()
    QtWidgets.QApplication.processEvents()
    drawn = _texts_drawn(clone.scene())
    for text in (*rows, "Depth against dwell"):
        assert in_family(drawn[text], TIMES), (text, drawn[text])


def test_a_saved_model_with_an_unsafe_family_is_not_replayed_into_label_html(qapp, qtbot, tmp_path, arial):
    """The family goes back into HTML written as style='…'. A saved project is data, not
    code: a title family that would end the declaration is dropped, and a legend entry that
    carries one makes the saved model invalid (sharing axes is then refused; independent axes
    still compose the frozen pictures)."""
    _page, plot = _summaries_page(qtbot, tmp_path, "")
    model = json.loads(fc.capture_cartesian_model(plot))
    assert model["title"]["options"]["family"] == font_picker.html_font_family(None)
    model["title"]["options"]["family"] = "Arial; color: #ff0000"
    clone = fc._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    title = clone.getPlotItem().titleLabel
    assert "family" not in title.opts, title.opts
    assert "#ff0000" not in title.item.toHtml().lower() and "color:#ff0000" not in title.item.toHtml()
    # A legend entry's: checked where the saved model is read.
    bad = json.loads(fc.capture_cartesian_model(plot))
    bad["legend"]["entries"][0]["family"] = "Arial'; color: red"
    with pytest.raises(ValueError, match="legend entry"):
        fc._validate_legend(bad)
    tampered = capture_figure_snapshot(
        plot, data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
        additional_resources={fc.CARTESIAN_RESOURCE: json.dumps(bad).encode()})
    compatible, reason = fc.shared_axes_compatibility([tampered, tampered])
    assert not compatible and "Invalid saved legend entry" in reason, reason
    for family in (None, "Arial", 'Arial, Helvetica, "Liberation Sans"'):
        ok = json.loads(fc.capture_cartesian_model(plot))
        ok["legend"]["entries"][0]["family"] = family
        fc._validate_legend(ok)                       # raises nothing


def test_a_saved_family_that_is_empty_or_longer_than_400_characters_is_not_replayed(
        qapp, qtbot, tmp_path, arial):
    """The same guard for a value with no unsafe character in it: a family is a short list of
    names. 400 characters pass (the bound is inclusive); 401, and an empty string, are no family:
    the title drops one it was given and a legend entry that carries one makes the model invalid."""
    _page, plot = _summaries_page(qtbot, tmp_path, "")
    for family, replayed in (("A" * 400, True), ("A" * 401, False)):
        model = json.loads(fc.capture_cartesian_model(plot))
        model["title"]["options"]["family"] = family
        clone = fc._clone_model(model, model["canvas"], model["ranges"])
        qtbot.addWidget(clone)
        drawn = clone.getPlotItem().titleLabel.opts.get("family") == family
        assert drawn is replayed, f"a {len(family)}-character title family: replayed={drawn}"
    for family, valid in (("A" * 400, True), ("A" * 401, False), ("", False)):
        entry = json.loads(fc.capture_cartesian_model(plot))
        entry["legend"]["entries"][0]["family"] = family
        if valid:
            fc._validate_legend(entry)
        else:
            with pytest.raises(ValueError, match="legend entry"):
                fc._validate_legend(entry)


# ================================================================================================
# R1 -- Distributions' Undo on a page restored from a project
# ================================================================================================

def _click_through_window(button) -> None:
    """A real left click on ``button``, sent to the window it is in."""
    assert button.isVisible() and button.isEnabled(), button.text()
    QtTest.QTest.mouseClick(window_handle(button), QtCore.Qt.MouseButton.LeftButton,
                            QtCore.Qt.KeyboardModifier.NoModifier,
                            button.mapTo(button.window(), button.rect().center()))
    QtWidgets.QApplication.processEvents()


def _saved_plots(path) -> dict:
    """What the project file holds for Distributions: ``{plot_type, title, font_family}``."""
    with zipfile.ZipFile(path) as archive:
        from nanoscope.core import project as project_format

        manifest = json.loads(archive.read(project_format.MANIFEST_NAME))
    for artifact in manifest["workspace"]["artifacts"]:
        if artifact["artifact_id"] == PLOTS_ARTIFACT_ID:
            fields = artifact["payload"]["config"]["fields"]
            style = fields["style"]["fields"]
            return {"plot_type": fields["plot_type"], "title": style["title"],
                    "font_family": style.get("font_family", "")}
    raise AssertionError("the project holds no Distributions artifact")


def _opened_distributions(qtbot, tmp_path, *, old_format=False):
    """A project saved with a Distributions histogram fit styled the way a lab would (a typed
    title, Times New Roman, BIC and KS in the fit label), opened in a FRESH main window. With
    ``old_format`` the file is one an older build wrote: BIC and KS and the title's bold and
    italic come from its ``fit_values_on_plot``, ``title_bold`` and ``title_italic`` switches.
    Returns ``(writer window, opened window, its Distributions page)``."""
    window, plots = _recordings(qtbot, tmp_path)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    plots.set_title("Dwell, pore A")
    plots.set_font_family(TIMES)
    if not old_format:
        for field in ("bic", "ks"):
            plots._appearance_widget.fit_label_fields[field].click()
    plots._history_touch()
    _settle()
    saved = tmp_path / "first.nanoscope"
    assert window._write_project(str(saved))
    window._dirty = False
    if old_format:
        older = tmp_path / "older.nanoscope"
        assert _with_old_keys_added(saved, older, fit_values_on_plot=True, title_bold=True,
                                    title_italic=True) == 1
        saved = older
    again = _main(qtbot)
    with qtbot.waitSignal(again.projectRestored, timeout=20000):
        again._open_project(str(saved))
    again._open_plots()
    _settle()
    page = again._plots_window
    assert page.config().style.title == "Dwell, pore A" and page.config().style.font_family == TIMES
    return window, again, page


def _what_is_drawn(page) -> dict:
    """The figure as drawn: the title text and its weight, the fit label's text, the plot type on
    the combo, the font."""
    title = page.config().style.text_style("title")
    return {"plot": page._plot_type_combo.currentText(),
            "title": page._plot.titleLabel.item.toPlainText().strip(),
            "bold/italic": (title.bold, title.italic),
            "label": page._fit_label_item.plain_text() if page._fit_label_item.isVisible() else "",
            "font": page.config().style.font_family}


@pytest.mark.parametrize("old_format", [False, True])
def test_the_first_change_after_opening_a_project_is_undone_back_to_the_saved_state(
        old_format, qapp, qtbot, tmp_path):
    """The restored configuration was never on the Undo stack, so the first change after opening
    was undone to the page's built-in defaults: the typed title, the font and the fit label's
    numbers did not come back, and 'press Undo alone' (the guide) was not true there."""
    window, again, page = _opened_distributions(qtbot, tmp_path, old_format=old_format)
    saved = _what_is_drawn(page)
    assert saved["title"] == "Dwell, pore A" and "BIC = " in saved["label"] and "KS = " in saved["label"]
    assert page._history.depth() == (0, 0) and not page._undo_btn.isEnabled()
    _user_picks(page._plot_type_combo, "Scatter")           # the rule clears the typed words
    assert page.config().style.title == ""
    assert page._history.depth() == (1, 0)
    _press_undo(page)                                       # Undo alone, the Undo key
    _settle()
    assert _what_is_drawn(page) == saved, "title, font and numbers all come back"
    assert page._history.depth() == (0, 1)
    window._dirty = again._dirty = False


def test_restoring_a_saved_state_into_a_window_that_holds_work_starts_its_history_afresh(
        qapp, qtbot, tmp_path):
    """``restore_plots`` is what a project open ends in, after the route has blanked the window, so
    the window it restores into is empty. Called on one that still holds work, the saved state is
    all there is: nothing to undo or redo, no step waiting on the coalescing timer, and the Undo
    and Redo buttons and the template combo say so. Two kinds of work: a closed step with a
    template recalled from its combo, then a typed title the timer has not closed."""
    window, again, page = _opened_distributions(qtbot, tmp_path)
    saved = capture_plots(page)

    def restored():
        report = restore_plots(page, saved)
        _settle()
        assert report.skipped == (), report.warnings
        assert page.config().style.title == "Dwell, pore A"
        assert page._plot_type_combo.currentText() == "Histogram"
        assert page._history.depth() == (0, 0)
        assert not page._undo_btn.isEnabled() and not page._redo_btn.isEnabled()

    template = "Publication figure (light)"
    page._template_combo.setCurrentText(template)               # the real recall route
    QtWidgets.QApplication.processEvents()
    assert page._template_combo.currentText() == template and page._history.depth() == (1, 0)
    assert page._undo_btn.isEnabled(), "premise: a step to undo"
    restored()
    assert page._template_combo.currentText() == page._TPL_MANUAL, "the template no longer describes it"

    page.set_title("Half typed")                                # a typed run, not closed
    assert page._history_timer.isActive() and page._undo_btn.isEnabled(), "premise: typing in hand"
    restored()
    assert not page._history_timer.isActive(), "no step is left waiting for the old window's typing"
    window._dirty = again._dirty = False


def test_the_undo_key_on_a_restored_page_with_nothing_changed_does_nothing(qapp, qtbot, tmp_path):
    window, again, page = _opened_distributions(qtbot, tmp_path)
    saved, config = _what_is_drawn(page), page.config()
    again._dirty = False
    _press_undo(page)
    _settle()
    assert page.config() == config and _what_is_drawn(page) == saved
    assert page._history.depth() == (0, 0) and not again._dirty, "and the project stays clean"
    window._dirty = again._dirty = False


@pytest.mark.parametrize("old_format", [False, True])
def test_reset_figure_then_undo_then_save_keeps_the_look_of_a_restored_page(
        old_format, qapp, qtbot, tmp_path):
    """Reset figure works on a restored page (it did nothing on the old build). Undo put the
    screen right, but the file was saved from the checkpoint taken at the Reset: the title, the
    font, the label's numbers and the title's bold and italic were lost on the next save (the
    review saw it on projects an older build had written, which is the second case here)."""
    window, again, page = _opened_distributions(qtbot, tmp_path, old_format=old_format)
    saved = _what_is_drawn(page)
    assert "BIC = " in saved["label"] and "KS = " in saved["label"]
    if old_format:
        assert saved["bold/italic"] == (True, True), "the older build's title weight came across"
    _click_through_window(page._appearance_widget.reset_button)
    _settle()
    assert page.config().style.title == "" and page.config().style.font_family == ""
    assert page._history.depth() == (1, 0)
    _press_undo(page)
    _settle()
    assert _what_is_drawn(page) == saved
    again._dirty = False
    second = tmp_path / "second.nanoscope"
    assert again._write_project(str(second))
    assert _saved_plots(second) == {"plot_type": "Histogram", "title": "Dwell, pore A",
                                    "font_family": TIMES}, "the file holds what is on screen"
    reopened = _main(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20000):
        reopened._open_project(str(second))
    reopened._open_plots()
    _settle()
    assert _what_is_drawn(reopened._plots_window) == saved
    window._dirty = again._dirty = reopened._dirty = False


def test_a_project_saved_after_an_undo_holds_what_is_on_screen(qapp, qtbot, tmp_path):
    window, again, page = _opened_distributions(qtbot, tmp_path)
    page.set_title("B")
    page._history_touch()
    _settle()
    assert page._history.depth() == (1, 0)
    _press_undo(page)
    _settle()
    assert page.config().style.title == "Dwell, pore A"
    again._dirty = False
    second = tmp_path / "second.nanoscope"
    assert again._write_project(str(second))
    assert _saved_plots(second)["title"] == "Dwell, pore A", "not the 'B' of before the Undo"
    window._dirty = again._dirty = False


def test_undo_and_redo_mark_the_project_unsaved(qapp, qtbot, tmp_path):
    window, again, page = _opened_distributions(qtbot, tmp_path)
    page.set_title("B")
    page._history_touch()
    _settle()
    again._dirty = False
    _press_undo(page)
    _settle()
    assert again._dirty, "an Undo changes the project"
    again._dirty = False
    assert page.redo_config()
    _settle()
    assert again._dirty and page.config().style.title == "B", "and so does a Redo"
    window._dirty = again._dirty = False


# ================================================================================================
# F2 -- Comparison figures: the Undo key answers from every Section
# ================================================================================================

def _drawn_words(drawn) -> tuple:
    """What the last render of the matplotlib figure drew: suptitle, its size, x and y labels."""
    last = drawn[-1]
    return (last["title"], last["size"], last["x"], last["y"])


def test_comparison_reset_figure_and_the_rules_reset_are_undone_from_the_section_they_were_made_in(
        comparison):
    """Reset figure sits in the toolbar and the rule resets at Generate, both reachable from every
    Section; the Undo key answered only while Appearance showed, so a stray click could not be
    taken back from where it was made. Read off what the matplotlib figure was asked to draw."""
    page, explorer, drawn = comparison
    widget = page._appearance_widget
    _type_comparison_words(page)
    typed = _drawn_words(drawn)
    assert typed == ("Mine", 14.0, ["Dw"], ["Dp"])

    for section in ("Data", "Analysis", "Export"):
        _section(page, section)
        assert not widget.isVisible(), f"premise: the Appearance controls are folded away in {section}"
        steps = widget._history.depth()
        _click_through_window(page._workflow.reset_button)
        assert widget._history.depth() == (steps[0] + 1, 0)
        reset = _drawn_words(drawn)
        assert reset[0] == "Recording comparisons" and reset[1] != 14.0, reset
        assert reset[2] != ["Dw"] and reset[3] != ["Dp"], reset
        _press_undo(page)
        assert _drawn_words(drawn) == typed, f"Undo from {section}"
        assert widget._history.depth() == (steps[0], 1)

    # The rule's reset at Generate, started from the toolbar of the Analysis Section.
    _section(page, "Analysis")
    _pick(page._y_combo, data="amplitude_pA")
    steps = widget._history.depth()
    _generate(page)
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert drawn[-1]["title"] == "Recording comparisons" and drawn[-1]["y"] != ["Dp"]
    _press_undo(page)
    assert (drawn[-1]["title"], drawn[-1]["x"], drawn[-1]["y"]) == ("Mine", ["Dw"], ["Dp"])
    assert widget._history.depth() == (steps[0], 1)


# ================================================================================================
# F3 -- an SVG names the family that was picked, even when it is the application's own font
# ================================================================================================

@pytest.fixture
def application_font(qapp):
    """``use(family)`` sets the family Qt draws text in when it was given none, for one test."""
    saved = QtGui.QGuiApplication.font()

    def use(family: str) -> None:
        font = QtGui.QFont(saved)
        font.setFamily(family)
        QtGui.QGuiApplication.setFont(font)

    yield use
    QtGui.QGuiApplication.setFont(saved)


def test_an_svg_names_a_picked_family_first_even_when_it_is_the_applications_own(
        vdep, arial, application_font, monkeypatch, qtbot, tmp_path):
    """On Windows the application font is a family a person can pick in Other font… (Segoe UI):
    the screen, the PNG and the PDF were drawn in it and the SVG named Arial, because the family
    was the application's own. Menlo stands in for Segoe UI; the page is Voltage dependence, the
    export is its own Export ▾ ▸ Export current figure…."""
    page, explorer = vdep
    application_font(OTHER)
    _show_appearance(page)
    picker = page._appearance_widget.font_family
    picking(monkeypatch, OTHER)
    _pick_font(picker, "Other font…")
    settle(qtbot, picker)
    assert page._appearance_widget.state().font_family == OTHER
    svg = _export_through_the_menu(page, monkeypatch, tmp_path / "menlo.svg")
    families = _svg_families(svg)
    assert families and set(families) == {ff.css_font_family(OTHER)}, set(families)
    pdf = _export_through_the_menu(page, monkeypatch, tmp_path / "menlo.pdf", ".pdf")
    assert any(OTHER in name for name in _pdf_is_clean(pdf)), "the PDF is in the same font"
    # An untouched figure is still Arial's stack: the application's own font is not the figure's.
    _pick_font(picker, "Arial (default)")
    plain = _export_through_the_menu(page, monkeypatch, tmp_path / "plain.svg")
    assert set(_svg_families(plain)) == {ff.css_font_family(None)}


# ================================================================================================
# F5 / F6 -- what a main-list row says; the Text style… font box's tool tip
# ================================================================================================

WINDOWS_LIKE = ["Arial", "Times New Roman", "Courier New", "Calibri", "Segoe UI"]
LIBERATION_ONLY = ["Liberation Sans", "Liberation Serif", "Liberation Mono", "DejaVu Sans"]


def _rows(picker) -> list[str]:
    return [picker.itemText(i) for i in range(picker.count())]


def test_a_main_list_family_this_computer_lacks_says_what_it_is_drawn_in(qapp, qtbot, monkeypatch):
    """The approved list text: on Windows, Helvetica is shown as Helvetica, drawn in Arial on this
    computer, rather than hidden. The row's TEXT only: what it stores, reports and saves is the
    family, so a figure styled on a Mac opens unchanged."""
    monkeypatch.setattr(font_picker, "installed_families", lambda: list(WINDOWS_LIKE))
    picker = font_picker.FontPicker()
    qtbot.addWidget(picker)
    assert _rows(picker) == ["Arial (default)", "Helvetica (drawn in Arial on this computer)",
                             "Times New Roman", "Courier New", "", "Other font…"]
    assert [picker.itemData(i) for i in range(4)] == ["", "Helvetica", "Times New Roman", "Courier New"]
    assert picker.itemData(1, QtCore.Qt.ItemDataRole.ToolTipRole) == _rows(picker)[1]
    assert picker.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole) is None, "a plain row has none"
    heard = []
    picker.changed.connect(heard.append)
    picker.setCurrentIndex(1)                              # a pick stores the family, not the words
    assert heard == ["Helvetica"] and picker.value() == "Helvetica"
    picker.set_value("Times New Roman")
    picker.set_value("Helvetica")                          # showing a stored value says nothing
    assert heard == ["Helvetica"] and picker.currentIndex() == 1
    assert not picker.has_extra_row()


def test_a_computer_with_only_the_liberation_fonts_names_each_substitute(qapp, qtbot, monkeypatch):
    monkeypatch.setattr(font_picker, "installed_families", lambda: list(LIBERATION_ONLY))
    picker = font_picker.FontPicker()
    qtbot.addWidget(picker)
    assert _rows(picker)[:4] == [
        "Arial (default; drawn in Liberation Sans on this computer)",
        "Helvetica (drawn in Liberation Sans on this computer)",
        "Times New Roman (drawn in Liberation Serif on this computer)",
        "Courier New (drawn in Liberation Mono on this computer)"]
    # The Text style… box: its first row is an explicit Arial.
    boxed = font_picker.FontPicker(first_row=("Arial", "Arial"))
    qtbot.addWidget(boxed)
    assert boxed.itemText(0) == "Arial (drawn in Liberation Sans on this computer)"
    assert boxed.itemData(0) == "Arial"
    # Distributions' Style row is the same picker.
    from nanoscope.core.session import Session
    from nanoscope.gui.plots_panel import PlotsWindow

    window = PlotsWindow(Session())
    qtbot.addWidget(window)
    window.set_font_family("Helvetica")
    assert window._font_combo.currentText() == "Helvetica (drawn in Liberation Sans on this computer)"
    assert window.config().style.font_family == "Helvetica"


def test_a_computer_that_has_the_four_fonts_keeps_the_plain_rows(qapp, qtbot, monkeypatch):
    monkeypatch.setattr(font_picker, "installed_families", lambda: [
        "arial", "Helvetica", "TIMES NEW ROMAN", "Courier New", "Menlo"])
    picker = font_picker.FontPicker()
    qtbot.addWidget(picker)
    assert _rows(picker) == ["Arial (default)", "Helvetica", "Times New Roman", "Courier New", "",
                             "Other font…"], "found in any spelling: nothing to say"
    assert font_picker.drawn_in_note("Arial (default)", "Arial") == "Arial (default)"


def test_the_text_style_font_box_no_longer_calls_its_explicit_arial_the_figure_default(
        qapp, qtbot, tmp_path):
    """In Text style… the first row is an explicit Arial that counts only while Custom font is
    ticked; the picker's own sentence ("The first row is the figure default") was wrong there."""
    page, plot, _edit, _shown = _build("vdep", qtbot, tmp_path)
    add_title(plot, "Capture vs voltage")
    controller = figure_text.figure_text_for(plot)
    controller.last_menu = None
    _click(plot, plot.titleLabel.sceneBoundingRect().center())
    _choose(controller.last_menu, "Text style…")
    popover = controller.popover
    assert popover is not None and popover.key == "title"
    tip = popover.family.toolTip()
    assert "figure default" not in tip and "Custom font" in tip, tip
    assert "Arial itself" in tip and ff.OTHER_FONT_LABEL in tip
    # The Appearance picker's own sentence is untouched: there the first row IS the default.
    assert "figure default" in page._appearance_widget.font_family.toolTip()


# ================================================================================================
# The Other font… list leaves out the faces of symbols
# ================================================================================================

SYMBOL_FACES = ("Symbol", "Webdings", "Wingdings", "Wingdings 2", "Wingdings 3", "Zapf Dingbats",
                "Apple Symbols")


def test_the_other_font_list_leaves_out_the_faces_of_symbols_but_a_stored_one_still_shows(
        qapp, qtbot, monkeypatch):
    """The design: Symbol is never a figure font (Greek is Unicode Greek in the chosen family). The
    list behind Other font… no longer offers Symbol, Webdings, the Wingdings, Zapf Dingbats or
    Apple Symbols -- but a project that names one still shows it, on its own extra row."""
    monkeypatch.setattr(font_picker, "installed_families", lambda: [
        "Arial", "Menlo", "Avenir", *SYMBOL_FACES, "wingdings 3", ".AppleSystemUIFont"])
    assert font_picker.listable_families() == ["Arial", "Menlo", "Avenir"]
    dialog = font_picker.FontListDialog()
    qtbot.addWidget(dialog)
    assert dialog.families() == ["Arial", "Menlo", "Avenir"]
    assert not dialog.select_family("Wingdings") and dialog.select_family("Menlo")
    assert all(ff.is_symbol_family(f) for f in (*SYMBOL_FACES, "wingdings", " ZAPF dingbats "))
    assert not any(ff.is_symbol_family(f) for f in (None, "", "Arial", "Symbola", "Apple Color Emoji"))

    picker = font_picker.FontPicker()
    qtbot.addWidget(picker)
    heard = []
    picker.changed.connect(heard.append)
    picker.set_value("Wingdings")                         # what an older project stored
    assert picker.has_extra_row() and picker.currentText() == "Wingdings"
    assert picker.value() == "Wingdings" and heard == []
    # Other font… opened on it and cancelled keeps it: the list does not hold it, the picker does.
    picking(monkeypatch, None)
    picker.setCurrentIndex(picker.count() - 1)
    settle(qtbot, picker)
    assert picker.value() == "Wingdings" and heard == []


def test_this_computers_other_font_list_has_no_symbol_face(qapp, qtbot):
    dialog = font_picker.FontListDialog()
    qtbot.addWidget(dialog)
    shown = dialog.families()
    assert shown and not [f for f in shown if ff.is_symbol_family(f)], shown
    installed = {f.casefold() for f in font_picker.installed_families()}
    assert {"arial", "menlo"} <= {f.casefold() for f in shown}
    assert any(face.casefold() in installed for face in SYMBOL_FACES), (
        "premise: this computer has a symbol face to leave out")


def test_the_composers_label_font_list_has_no_row_for_a_private_application_font(
        qapp, qtbot, tmp_path, monkeypatch, arial):
    """The compose paragraph of the guide and the CHANGELOG: an older Library figure is drawn in
    the application font, and 'on a Mac, where the application font is a private face, the list has
    no row for it'. The panel-label font is the figure-font list; the Mac's face starts with a dot
    and is never listed, whereas an ordinary family (Segoe UI on Windows) is listed under Other
    font… and can be picked -- which is why the sentence is for a Mac and not for every computer."""
    _page, plot = _summaries_page(qtbot, tmp_path, "")
    entries = [_library_entry("test", "ECDF A", plot), _library_entry("test", "ECDF B", plot)]
    monkeypatch.setattr(font_picker, "installed_families", lambda: [
        ".AppleSystemUIFont", "Arial", "Helvetica", "Times New Roman", "Courier New", "Segoe UI"])
    dialog = FigureComposerDialog(entries, current_id=entries[0].artifact_id)
    qtbot.addWidget(dialog)
    label_font = dialog._label_font
    assert isinstance(label_font, font_picker.FontPicker) and label_font.value() == ""
    rows = [label_font.itemText(i) for i in range(label_font.count())]
    assert not [row for row in rows if "AppleSystemUIFont" in row], rows
    other = font_picker.FontListDialog()
    qtbot.addWidget(other)
    assert ".AppleSystemUIFont" not in other.families() and not other.select_family(".AppleSystemUIFont")
    assert "Segoe UI" in other.families(), "an ordinary application font is listed"
    picking(monkeypatch, "Segoe UI")
    label_font.setCurrentIndex(label_font.count() - 1)                 # Other font…
    settle(qtbot, label_font)
    assert label_font.value() == "Segoe UI" and label_font.has_extra_row()


# ================================================================================================
# D3 -- what the Title caption and the Add title entry say
# ================================================================================================

def _background_point(plot):
    vb = plot.getViewBox().sceneBoundingRect()
    return QtCore.QPointF(vb.left() + 0.55 * vb.width(), vb.top() + 0.45 * vb.height())


@pytest.mark.parametrize("kind, scope", [("noise", "this view"), ("events", "this view"),
                                         ("subpop", "both figures")])
def test_the_title_caption_and_the_add_title_entry_say_what_a_typed_title_covers(
        kind, scope, qapp, qtbot, tmp_path):
    """They said '(all views)': a promise that the title stays across the views, which the reset
    rule takes back on the first view switch. They now say what is true: this view (the figure or
    figures on screen), whose words go back to automatic when the figure shows something else."""
    page, plot, _edit, _shown = _build(kind, qtbot, tmp_path)
    widget = page._appearance_widget
    assert widget.title_scope == scope
    boxes = (widget.title, widget.x_label, widget.y_label)
    captions = sorted(label.text() for label in widget.findChildren(QtWidgets.QLabel)
                      if label.buddy() in boxes)
    assert captions == sorted([f"Title ({scope})", f"X label ({scope})", f"Y label ({scope})"])
    for box in boxes:
        assert "go back to automatic when the figure comes to show something else" in box.toolTip()
        assert "Undo brings them back" in box.toolTip()
    assert "all views" not in " ".join(captions) + " ".join(b.toolTip() for b in boxes)
    assert box.placeholderText() == "Automatic (each view's own)"
    # The reason an axis label is renamed in Appearance and not on a figure (also the read-only
    # tool tip in Axis properties) no longer calls the Appearance boxes "all views".
    assert "all views" not in figure_text.SHARED_LABELS_REASON
    assert "(Appearance ▸ X label / Y label)" in figure_text.SHARED_LABELS_REASON

    controller = figure_text.figure_text_for(plot)
    if plot.titleLabel.isVisible():
        controller.last_menu = None
        _click(plot, plot.titleLabel.sceneBoundingRect().center())
        _choose(controller.last_menu, "Hide title")
    controller.last_menu = None
    _click(plot, _background_point(plot))
    names = _names(controller.last_menu)
    controller.last_menu.close()
    assert names[0] == f"Add title ({scope})…", names
    assert "all views" not in " ".join(names)
    assert ff.OTHER_FONT_LABEL not in names    # (a menu of this figure, nothing else)


def test_a_window_with_one_figure_keeps_plain_captions_and_add_title(qapp, qtbot, tmp_path):
    page, plot, _edit, _shown = _build("vdep", qtbot, tmp_path)
    widget = page._appearance_widget
    assert widget.title_scope == ""
    captions = sorted(label.text() for label in widget.findChildren(QtWidgets.QLabel)
                      if label.buddy() in (widget.title, widget.x_label, widget.y_label))
    assert captions == ["Title", "X label", "Y label"]
    assert "come to show something else" not in widget.title.toolTip()
    controller = figure_text.figure_text_for(plot)
    controller.last_menu = None
    _click(plot, _background_point(plot))
    assert _names(controller.last_menu)[0] == "Add title…"
    controller.last_menu.close()


def test_comparison_figures_title_caption_says_this_view(comparison):
    page, explorer, drawn = comparison
    widget = page._appearance_widget
    assert widget.title_scope == "this view"
    assert sorted(label.text() for label in widget.findChildren(QtWidgets.QLabel)
                  if label.buddy() in (widget.title, widget.x_label, widget.y_label)) == [
        "Title (this view)", "X label (this view)", "Y label (this view)"]


def test_a_title_typed_with_the_entry_goes_back_to_automatic_when_the_view_changes(
        noise, qapp, qtbot):
    """What the new words promise, driven: type a title with 'Add title (this view)…', change the
    view, and it is gone (the status line says so, and one Undo brings it back)."""
    from .test_plot_window_routes import _add_title

    page, explorer = noise
    plot = page._plot
    controller = figure_text.figure_text_for(plot)
    controller.last_menu = None
    if plot.titleLabel.isVisible():
        _click(plot, plot.titleLabel.sceneBoundingRect().center())
        _choose(controller.last_menu, "Hide title")
    _add_title(plot, "My title")
    assert page._appearance_widget.state().title == "My title"
    steps = page._appearance_widget._history.depth()
    page.set_view("fft")
    QtWidgets.QApplication.processEvents()
    assert page._appearance_widget.state().title is None
    assert page.statusBar().currentMessage().startswith("Title reset: the plot now shows")
    assert page._appearance_widget._history.depth() == (steps[0] + 1, 0)


# ================================================================================================
# D6 -- Reset figure also takes back a typed caption; the four enumerations say so
# ================================================================================================

def test_reset_figure_takes_back_a_typed_caption_and_says_so(vdep):
    """Voltage dependence's caption is text you typed on the figure: Reset figure puts it back to
    its automatic words, in the same one Undo step as the rest. (The reset RULE leaves it alone.)"""
    from .test_plot_window_routes import _edit_double_click

    page, explorer = vdep
    plot = page.plot_item()
    automatic = page.caption_text()
    assert automatic
    _edit_double_click(plot, page._caption, "Pooled by hand")
    assert page.caption_text() == "Pooled by hand"
    widget = page._appearance_widget
    tip = page._workflow.reset_button.toolTip()
    assert "caption" in tip and "text you typed on the figure" in tip and "Undo" in tip and "defaults" in tip
    steps = widget._history.depth()
    _click_through_window(page._workflow.reset_button)
    assert page.caption_text() == automatic
    assert widget._history.depth() == (steps[0] + 1, 0)
    _press_undo(page)
    assert page.caption_text() == "Pooled by hand"


def test_reset_figure_in_distributions_takes_back_typed_annotation_words_but_not_a_renamed_source(
        qapp, qtbot):
    """What the guide and the CHANGELOG now say: the words on the detection-limit line (an
    annotation) and a typed title go back to automatic, and a source renamed on its legend row
    keeps its name -- 'any other text you typed on the figure' was not true of that one. All of
    it by the gestures a person makes on the figure; one Undo step takes back what went."""
    page = _distributions(qtbot)
    plot = page._plot
    automatic = page._limit_line.label.toPlainText()
    assert automatic == "below detection limit"
    _edit_double_click(plot, page._limit_line.label, "detector floor")
    row = next(label for _sample, label in plot.legend.items if label.item.toPlainText().startswith("probe"))
    _edit_double_click(plot, row, "Renamed source")
    add_title(plot, "Dwell, pore A")
    rows = lambda: [label.item.toPlainText() for _sample, label in plot.legend.items]      # noqa: E731
    assert page._limit_line.label.toPlainText() == "detector floor"
    assert plot.titleLabel.item.toPlainText() == "Dwell, pore A"
    assert any(text.startswith("Renamed source") for text in rows()), rows()
    assert page.config().style.label_overrides, "the rename is the source's name"

    steps = page._history.depth()
    _click_through_window(page._appearance_widget.reset_button)
    assert page._limit_line.label.toPlainText() == automatic, "the typed annotation words went"
    assert plot.titleLabel.item.toPlainText().strip() == "" and page.config().style.title == ""
    assert any(text.startswith("Renamed source") for text in rows()), (
        "a source renamed on its legend row keeps its name", rows())
    assert page._history.depth() == (steps[0] + 1, 0), "one Undo step"
    _press_undo(page)
    assert page._limit_line.label.toPlainText() == "detector floor"
    assert plot.titleLabel.item.toPlainText() == "Dwell, pore A"
    assert any(text.startswith("Renamed source") for text in rows()), rows()


# ================================================================================================
# L1 / C1 -- the Numbers grid in a wide strip; where the widget keeps its status sentence
# ================================================================================================

def test_the_numbers_boxes_keep_their_natural_spacing_in_a_wide_strip(vdep):
    """Three boxes to a line, but in a strip a thousand pixels wide they sat a third of the width
    apart; the spare width now goes to an empty fourth column."""
    page, explorer = vdep
    _section(page, "Appearance")
    widget = page._appearance_widget
    assert widget.fit_label_fields["equation"].isVisible()
    assert widget.width() > 700, "premise: a wide strip"
    x = {name: box.mapTo(widget, QtCore.QPoint(0, 0)).x() for name, box in widget.fit_label_fields.items()}
    right = {name: box.mapTo(widget, QtCore.QPoint(box.width(), 0)).x()
             for name, box in widget.fit_label_fields.items()}
    assert right["n"] < 300, f"the third box of the line ends at {right['n']} px of a {widget.width()} px strip"
    assert x["r2"] - x["equation"] < 150 and x["n"] - x["r2"] < 150, x
    assert (x["ci"], x["bic"], x["ks"]) == (x["equation"], x["r2"], x["n"]), "two lines of three"


def test_the_widget_keeps_its_status_sentence_on_the_instance(qapp, qtbot):
    from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget

    class Bar:
        def __init__(self):
            self.shown = ""

        def showMessage(self, text, _ms=0):
            self.shown = text

        def currentMessage(self):
            return self.shown

    assert "_reset_said" not in vars(PlotAppearanceWidget) and not hasattr(PlotAppearanceWidget, "_reset_said")
    first, second = PlotAppearanceWidget(), PlotAppearanceWidget()
    for widget in (first, second):
        qtbot.addWidget(widget)
        assert vars(widget)["_reset_said"] == ""
    first.apply_appearance(PlotAppearance(title="Mine"))
    bar = Bar()
    said = first.reset_typed_words("median depth", bar)
    assert said == "Title reset: the plot now shows median depth." and bar.shown == said
    assert vars(first)["_reset_said"] == said and vars(second)["_reset_said"] == ""


# ================================================================================================
# comments and documents that said what no longer is
# ================================================================================================

def test_no_comment_calls_the_fit_label_a_values_box():
    """The 'Fit values on plot' box is gone; the fit's numbers are in the fit label. A comment may
    still say what the OLD on-plot values box held; none may speak of one as if it were there."""
    pattern = re.compile(r"(\bold\s+)?(on-plot values box|fit-values box|the values box)")
    stale = []
    for path in sorted((REPO / "nanoscope").rglob("*.py")):
        flat = " ".join(path.read_text(encoding="utf-8").replace("#", " ").replace('"', " ").split())
        for match in pattern.finditer(flat):
            if not match.group(1) or match.group(2) != "on-plot values box":
                stale.append(f"{path.relative_to(REPO)}: {flat[max(0, match.start() - 40):match.end() + 20]}")
    assert not stale, "\n".join(stale)


def _flat(relative: str) -> str:
    return " ".join((REPO / relative).read_text(encoding="utf-8").split())


def test_the_documents_say_what_the_fixes_do():
    guide, changelog = _flat("docs/USER_GUIDE.md"), _flat("CHANGELOG.md")
    inventory = _flat("docs/inventory.json")
    manual = _flat("docs/MANUAL_ACID_WORKFLOW.md")

    # D1 / F3: an unchosen font is Arial because it IS Arial; only text with no font of its own is
    # the application's, and only a private face is rewritten.
    assert "— a figure whose font was never chosen — is written as Arial" not in guide
    assert "and also for a figure whose font was never chosen, which is drawn in Arial" in guide
    assert "A family you picked is named first even when it is the application's own font" in guide
    assert "Text drawn in the application's own font is written as Arial." not in changelog
    assert "a family someone picked is named first, the application's own font included" in changelog

    # D2: Distributions drops the typed range of the axis whose column moved (both axes on a new
    # plot type), and the documents say so. G5 made that every window's rule and supersedes the
    # first wording of this fix ("Distributions ALSO clears ... an axis whose quantity moved", said
    # while axis ranges stayed everywhere else): the documents carry G5's words, and the sentence
    # that listed ranges among what stays is gone (test_p0_final_seams pins that side).
    assert "Distributions clears the typed range of an axis whose quantity moved" not in changelog
    assert "Distributions also clears the typed range of an axis whose quantity moved" not in guide
    assert "which already took back the title, the axis labels and the typed range of an axis that moved" in changelog
    assert ("an axis whose column moved loses its typed range (a change of plot type moves both axes)"
            in guide)
    assert "an axis whose quantity moved loses its typed range" not in guide

    # D3: nothing promises the title stays across the views.
    for text in (guide, changelog):
        assert "Add title (all views)" not in text and "Appearance, all views" not in text
    assert "**Add title (this view)…** (**Add title (both figures)…** in Subpopulations" in guide
    assert "goes back to automatic when the view changes" in guide

    # D4: the Font is no longer an editable list.
    assert "or an editable list such as the plot windows' Font" not in changelog
    assert "the plot windows' Font was one, before the short font list replaced it" in changelog

    # D5: Reset all figure styling leaves the study figure in Arial.
    assert "restores the underlying scientific figure" not in manual
    assert "leaves the text in Arial, the editor's default font, not the font the study drew it in" in manual

    # D6: the four enumerations of what Reset figure takes back all name the caption -- and none
    # says it takes back "any other text you typed on the figure": a source renamed on its legend
    # row keeps its name (the guide says so of Distribution summaries and Distributions).
    assert ("any caption or other annotation text you typed (Voltage dependence's caption, for one)"
            in guide)
    assert ("any caption or other annotation text you typed (Voltage dependence's caption, the "
            "words on Distributions' detection-limit line) and the look back") in changelog
    assert "A source you renamed on its legend row (Distributions, Distribution summaries) keeps its name" \
        in changelog
    assert "any other text you typed on the figure" not in guide + changelog
    assert "Put the text you typed on the figure (title, axis labels, fit-label wording, caption)" in inventory
    assert "(Reset figure leaves the name alone)" in guide and "source names" in guide
    assert "(**Reset figure** takes it back)" in guide and "(Reset figure takes it back)" in changelog

    # F2: Comparison figures' keys answer in every Section; only the study editor is the exception.
    assert "In Comparison figures and the study figure editor the keys answer" not in guide
    assert "(In the study figure editor the keys answer while the Appearance controls are on screen.)" in guide
    assert "(the study figure editor: while the Appearance controls are on screen)" in guide
    assert "Comparison figures and the study figure editor: while the Appearance controls" not in guide

    # F4 / F2 (documents): the compose bullet is true for figures saved by this build; an old
    # entry keeps its font, and only where the application font is a private face (a Mac) does the
    # list have no row for it; a Mac's matplotlib reads one face of Helvetica.
    assert "so they match the panels of figures saved by this build" in changelog
    assert "A figure saved to the Library by an earlier version is a frozen picture" in guide
    for text in (guide, changelog):
        assert "on a Mac, where the application font is a private face, the list has no row for it" in text
        assert "the list has no row for the application font" not in text
    assert "a study figure's bold title, y tick labels and bar text" in guide
    assert "a study figure's bold title, y tick labels and bar text" in changelog

    # F5 and the symbol faces: what a picker row says, and what Other font… leaves out.
    assert "*Helvetica (drawn in Arial on this computer)*" in guide
    assert "so are the faces of symbols (Symbol, Webdings, Wingdings," in guide
    # ...and both lists are the code's seven names, none left out (the inventory's had five).
    assert tuple(ff.SYMBOL_FAMILIES) == SYMBOL_FACES
    listed = "(Symbol, Webdings, Wingdings, Wingdings 2 and 3, Zapf Dingbats, Apple Symbols)"
    assert f"the faces of symbols {listed}" in guide and f"the faces of symbols {listed}" in inventory
    assert "A row for a font this computer lacks says what it is drawn in" in inventory
    assert "Other font… leaves out private fonts and the faces of symbols" in inventory
    assert "its row says which (on Windows: Helvetica (drawn in Arial on this computer))" in inventory
    assert "Other font… leaves out the faces of symbols such as Wingdings" in inventory
    assert "where the first row is an explicit Arial that counts only while Custom font is ticked" in inventory

    # R1 and F1: the guide says where the Undo history starts and what a composition keeps.
    assert "It starts when a project is opened, at the state the project was saved in" in guide
    assert "it marks the project unsaved, and a project saved after it holds what is on screen" in guide
    assert "A panel re-drawn with shared axes keeps the font of the figure it was saved from" in guide
