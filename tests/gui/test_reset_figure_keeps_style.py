"""S1: Reset figure returns a figure in a project to what a NEW figure of its kind looks like there.

A figure made in a project is bound to the project's default figure style: it is light, has no
grid, and its Style ▾ button names the style. Reset figure used to rebuild the appearance from
nothing, which dropped that binding with the look, so the same page came back dark, gridded and
bound to no style, a look no new figure has. It now keeps the figure's style as Reset look does
and still takes off what it always took off: the words, the pinned ranges, the element styles, the
dragged annotations, the fit-label wording and the look the Appearance controls had overridden.

A figure with no style (New figure > From recipe makes one, and so did the old Reset figure in a
project saved by it) is bound to the project's CURRENT default style by Reset figure, with the
look a new figure is given there (``FigureWorkspace.new_figure_look``, which ``create`` uses too),
and the Undo and Redo tooltips name that step "reset figure" while a style chosen in Style ▾ never
reads as one, whether the figure is awake or asleep when the style reaches it. A window that
belongs to no project (no workspace to give it a style) keeps the plain reset.

Distributions draws its lines and markers in pixels and keeps them in boxes of three decimals, so
a style whose sizes are not whole pixels (the 8 pt and 9 pt journal styles: 1 pt is 1.3333 px) is
applied exactly and leaves the figure no override of its own, by Reset figure or by Style ▾.

Every figure here is made with ``FigureWorkspace.create`` in a real ``MainWindow`` and reset with
the Explorer toolbar's own button. What a Reset figure always kept (the drawn-quantity record,
source names, Distributions' Font size (body), with a style or without one) is still kept.
Synthetic data only.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from PySide6 import QtWidgets

from nanoscope.core.figure_history_state import HistoryCapsule
from nanoscope.core.figure_spec import thaw_json
from nanoscope.core.figure_style import builtin_styles
from nanoscope.gui.analysis_catalog import PAGE_IDS
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.figure_text import AxisPin, TextStyle
from nanoscope.gui.fit_label import FitLabelState
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_appearance import (
    APPEARANCE_HISTORY_DEPTH, PlotAppearance, PlotAppearanceWidget, _AppearanceHistory,
    _describe_step, _fingerprint)

from .test_plot_appearance_history import _study_window

#: What the project's default figure style is called, as a new figure's Style ▾ button says it
#: (a name over 24 characters is cut to 23 and an ellipsis; a ● follows a revision with overrides).
DEFAULT_STYLE_BUTTON = "7 pt Arial · Nature, El… · r1 ▾"


@pytest.fixture
def project(qapp, qtbot):
    """``(window, workspace, explorer)`` of a real project with no figure yet."""
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    return window, owner, window._ensure_analysis_explorer()


def _new(owner, kind):
    """A new figure of ``kind`` and its page: made as the Explorer's New figure makes it."""
    key = owner.create(kind)
    return key, owner.page(key)


def _depth(kind, page):
    """``(steps back, steps forward)`` of the history that holds the page's appearance:
    Distributions' configuration history, the appearance widget's own elsewhere."""
    return (page._history if kind == "distributions" else page._appearance_widget._history).depth()


def _bound(style=None, **fields) -> PlotAppearance:
    style = style or builtin_styles()[0]
    return replace(PlotAppearance.from_mapping(bind_native_style({}, style)), **fields)


def _overrides_taken_off(state: PlotAppearance) -> PlotAppearance:
    """``state`` without the overrides its binding carries (Distributions' Font size (body) is one),
    to compare everything else of two bound figures."""
    mapping = state.to_dict()
    mapping["style_binding"] = {**mapping["style_binding"], "overrides": {}}
    return PlotAppearance.from_mapping(mapping)


def _customise(kind, page):
    """What a person does to a figure before pressing Reset figure: types words, sets sizes, a
    font, the background, the grid and the legend, styles the title, pins a range, drags the
    caption, words the fit label. Each is one gesture of the Appearance controls (one step)."""
    widget = page._appearance_widget
    widget.apply_appearance(replace(
        widget.state(), title="Capture vs voltage", x_label="Voltage (mV)", y_label="Rate (Hz)",
        title_size=20.0, axis_size=14.0, background="dark", grid=True,
        font_family="Times New Roman", legend_visible=False))
    bold = replace(widget.state().text_style("title"), bold=True)    # the title's text rides along
    widget.apply_appearance(widget.state().with_text_style("title", bold))
    state = replace(widget.state(), fit_label=FitLabelState(template="Fit: {equation}", size_pt=11.0))
    if kind != "distributions":     # it keeps its ranges and caption place in its own config
        state = (state.with_axis_pin("bottom", AxisPin(min=0.0, max=5.0, tick_major=1.0))
                      .with_annotation_offset("annotation.provenance", (12.0, 18.0)))
    widget.apply_appearance(state)
    QtWidgets.QApplication.processEvents()


def _unbind(page, **look):
    """Take the figure's style away and give it a look of its own, as a figure made from a recipe
    has it and as the old Reset figure left one in a saved project: bound to no style."""
    page.set_appearance_state({"title_size": 13.0, "axis_size": 12.0, "tick_size": 11.0,
                               "background": "light", "grid": False, "line_width": 2.0, **look})
    assert page._appearance_widget.state().style_binding is None


# ==================================================================================================
# 1. the defect, on every kind of figure that has the button
# ==================================================================================================

@pytest.mark.parametrize("kind", PAGE_IDS)
def test_reset_figure_leaves_a_project_figure_looking_as_a_new_one_does(project, kind):
    _window, owner, explorer = project
    _fresh_key, fresh = _new(owner, kind)        # a new figure of the kind, never touched
    key, page = _new(owner, kind)                # the one that is customised and reset
    widget, adapter, bar = page._appearance_widget, explorer.adapter(key), explorer.toolbar()
    style = owner.default_style()
    assert widget.state() == fresh._appearance_widget.state()
    assert bar.style_button.text() == DEFAULT_STYLE_BUTTON

    _customise(kind, page)
    edited_state, edited_saved = widget.state(), page.appearance_state()
    assert edited_state != fresh._appearance_widget.state()
    assert edited_state.title == "Capture vs voltage" and edited_state.background == "dark"
    assert bar.style_button.text() == DEFAULT_STYLE_BUTTON.replace(" ▾", " · ● ▾")
    steps = _depth(kind, page)

    bar.reset_figure_button.click()

    # The figure looks as a new one does: the same appearance, and it is bound to the style the
    # project gives new figures (light, no grid, the style's fonts), with nothing on top of it.
    assert widget.state() == fresh._appearance_widget.state()
    assert page.appearance_state() == fresh.appearance_state()
    state = widget.state()
    assert state.style_binding, "Reset figure dropped the figure's style"
    assert state.style_binding["ref"] == style.ref.to_dict()
    assert dict(state.style_binding["overrides"]) == {}
    assert (state.background, state.grid, state.font_family) == ("light", False, "Arial")
    assert (state.title, state.x_label, state.y_label, state.text, state.fit_label, state.axes,
            state.annotations, state.series_styles) == (None,) * 8
    assert bar.style_button.text() == DEFAULT_STYLE_BUTTON

    # One gesture, one Undo step, named as a Reset figure, and the one Undo brings everything back.
    assert _depth(kind, page) == (steps[0] + 1, 0)
    if kind != "distributions":              # its configuration Undo counts steps and names none
        assert widget.undo_button.toolTip() == "Undo: reset figure"
        assert bar.undo_action.toolTip() == "Undo: reset figure"
    adapter.undo()
    assert widget.state() == edited_state and page.appearance_state() == edited_saved
    assert bar.style_button.text() == DEFAULT_STYLE_BUTTON.replace(" ▾", " · ● ▾")
    assert _depth(kind, page) == (steps[0], 1)
    if kind != "distributions":
        assert widget.redo_button.toolTip() == "Redo: reset figure"
    adapter.redo()
    assert widget.state() == fresh._appearance_widget.state()
    assert _depth(kind, page) == (steps[0] + 1, 0)


@pytest.mark.parametrize("kind", PAGE_IDS)
def test_reset_figure_binds_a_figure_with_no_style_to_the_projects_default(project, kind):
    """A figure of a project that is bound to no style (New figure > From recipe makes one, and so
    did the old Reset figure) goes back to what a NEW figure of its kind looks like there: bound to
    the project's default style, light, no grid, with the words and pins gone. One Undo brings back
    the figure with no style, its look, its words and its pins; Redo takes them off again."""
    _window, owner, explorer = project
    _fresh_key, fresh = _new(owner, kind)
    key, page = _new(owner, kind)
    widget, adapter, bar = page._appearance_widget, explorer.adapter(key), explorer.toolbar()
    _unbind(page)
    _customise(kind, page)
    edited_state, edited_saved = widget.state(), page.appearance_state()
    assert edited_state.style_binding is None and "style_binding" not in edited_saved
    assert (edited_state.title, edited_state.background, edited_state.grid) == (
        "Capture vs voltage", "dark", True)
    steps = _depth(kind, page)

    bar.reset_figure_button.click()

    # The same appearance as the new figure that was never touched, and so the same binding.
    assert widget.state() == fresh._appearance_widget.state()
    assert page.appearance_state() == fresh.appearance_state()
    state = widget.state()
    assert state.style_binding, "Reset figure left the figure with no style"
    assert state.style_binding["ref"] == owner.default_style().ref.to_dict()
    assert dict(state.style_binding["overrides"]) == {}
    assert (state.background, state.grid, state.font_family) == ("light", False, "Arial")
    assert (state.title, state.x_label, state.y_label, state.text, state.fit_label, state.axes,
            state.annotations, state.series_styles) == (None,) * 8
    assert bar.style_button.text() == DEFAULT_STYLE_BUTTON

    # One gesture, one step named as a Reset figure; Undo, Redo and Undo again walk it.
    assert _depth(kind, page) == (steps[0] + 1, 0)
    if kind != "distributions":
        assert widget.undo_button.toolTip() == "Undo: reset figure"
        assert bar.undo_action.toolTip() == "Undo: reset figure"
    adapter.undo()
    assert widget.state() == edited_state and page.appearance_state() == edited_saved
    assert widget.state().style_binding is None and widget.state().title == "Capture vs voltage"
    assert _depth(kind, page) == (steps[0], 1)
    if kind != "distributions":
        assert widget.redo_button.toolTip() == "Redo: reset figure"
        assert bar.redo_action.toolTip() == "Redo: reset figure"
    adapter.redo()
    assert widget.state() == fresh._appearance_widget.state()
    assert widget.state().style_binding["ref"] == owner.default_style().ref.to_dict()
    assert _depth(kind, page) == (steps[0] + 1, 0)
    adapter.undo()
    assert widget.state() == edited_state, "and the Undo after a Redo is still the whole step"


@pytest.mark.parametrize("kind", PAGE_IDS)
def test_reset_figure_binds_the_default_style_the_project_has_now(project, kind):
    """The project's CURRENT default: a figure with no style is bound to the one the project gives
    new figures today, not to whatever it was when the figure was made. A figure that has a style
    keeps that one, as before."""
    _window, owner, explorer = project
    key, page = _new(owner, kind)
    keeps_key, keeps = _new(owner, kind)
    # (The slides style: every size far from the project's, and a line and a marker of a whole
    # number of pixels, which Distributions, drawing in pixels, can show exactly.)
    slides = builtin_styles()[3]
    assert slides.ref != owner.default_style().ref
    _unbind(page)
    owner.set_default_style(slides.ref)
    _fresh_key, fresh = _new(owner, kind)
    assert fresh._appearance_widget.state().style_binding["ref"] == slides.ref.to_dict()
    assert owner.activate(key)

    explorer.toolbar().reset_figure_button.click()

    state = page._appearance_widget.state()
    if kind == "distributions":
        # Its Font size (body) is the one thing Reset figure leaves: the 7 pt the figure had, not
        # the slides style's 18. Everything else is what a new figure has.
        assert dict(state.style_binding["overrides"]) == {"body_size_pt": 7.0}
        assert page.style().font_size_pt == 7.0 and fresh.style().font_size_pt == 18.0
        assert _overrides_taken_off(state) == _overrides_taken_off(fresh._appearance_widget.state())
    else:
        assert state == fresh._appearance_widget.state()
    assert state.style_binding["ref"] == slides.ref.to_dict() and state.title_size == 18.0

    assert owner.activate(keeps_key)
    explorer.toolbar().reset_figure_button.click()
    assert keeps._appearance_widget.state().style_binding["ref"] == builtin_styles()[0].ref.to_dict()


@pytest.mark.parametrize("kind", PAGE_IDS)
def test_reset_figure_binds_what_a_new_figure_is_bound_to_in_a_project_with_an_older_default_look(
        project, kind):
    """The look a project from an older build gives its new figures is a sparse one (it names
    only what it sets, and Distributions' Font size (body) is made from its axis and title size).
    A new figure and a figure with no style that Reset figure binds are given it by the same code,
    so they are bound alike, Distributions' body size included."""
    _window, owner, explorer = project
    registry, ref = owner.style_registry().import_legacy_style(
        "Old project look", {"title_size": 12.0, "grid": True, "background": "light"},
        "plot_appearance")
    owner._replace_styles(registry.with_default(ref))
    _fresh_key, fresh = _new(owner, kind)
    _key, page = _new(owner, kind)
    _unbind(page)
    _customise(kind, page)
    assert fresh._appearance_widget.state().style_binding["ref"] == ref.to_dict()

    explorer.toolbar().reset_figure_button.click()

    state = page._appearance_widget.state()
    assert state.style_binding["ref"] == ref.to_dict() and state.title is None
    if kind == "distributions":
        # (This page keeps the values of the figure that the sparse look does not name, as it does
        # for every look applied to it, so only what the project's look names is compared.)
        assert dict(state.style_binding["overrides"])["body_size_pt"] == 12.0
        assert page.style().font_size_pt == fresh.style().font_size_pt == 12.0
        assert (state.title_size, state.grid, state.background) == (12.0, True, "light")
    else:
        assert state == fresh._appearance_widget.state()
        assert dict(state.style_binding["overrides"]) == {}


@pytest.mark.parametrize("kind", ["iv", "noise", "comparison_figures", "distribution_summaries"])
def test_a_style_chosen_in_style_never_reads_as_a_reset_figure(project, kind):
    """The Undo tooltip names the step by what it did. A style picked in Style ▾ binds a figure with
    no style to that style (the project's default included, which lands on the same values as a
    Reset figure of a figure with a look and nothing else), and gives a figure its own style's
    look back: neither is a Reset figure."""
    _window, owner, explorer = project
    key, page = _new(owner, kind)
    widget = page._appearance_widget
    vocabulary = "matplotlib_appearance" if kind == "comparison_figures" else "plot_appearance"

    def pick(style):
        return explorer.apply_look(PlotAppearance.from_mapping(
            bind_native_style({}, style, vocabulary)), scope="page", keep_hand=False)

    for style in (builtin_styles()[2], owner.default_style()):
        _unbind(page)                 # a look and nothing else: no word, no pin, no element style
        assert pick(style).applied == (key,)
        assert widget.state().style_binding["ref"] == style.ref.to_dict()
        assert widget.undo_button.toolTip() != "Undo: reset figure", style.name
        assert widget.undo_button.toolTip().endswith("appearance changes"), style.name
        assert widget.undo() and widget.state().style_binding is None
        assert widget.redo_button.toolTip() != "Redo: reset figure"

    # The same style again, on a figure that has it and has looks of its own set on top of it.
    widget.undo()
    pick(owner.default_style())
    widget.apply_appearance(replace(widget.state(), title_size=20.0, axis_size=14.0))
    assert widget.undo_button.toolTip() == "Undo: 2 appearance changes"
    assert pick(owner.default_style()).applied == (key,)
    assert dict(widget.state().style_binding["overrides"]) == {}
    assert widget.undo_button.toolTip() != "Undo: reset figure"
    assert widget.undo_button.toolTip().endswith("appearance changes")
    # ...which is not what the Reset figure button of that same figure reads.
    assert widget.undo() and widget.state().title_size == 20.0
    explorer.toolbar().reset_figure_button.click()
    assert widget.undo_button.toolTip() == "Undo: reset figure"


def test_reset_figure_of_a_figure_with_no_style_keeps_the_quantity_record_and_the_source_names(project):
    """What a Reset figure keeps it keeps whether or not the figure has a style."""
    _window, owner, explorer = project
    _key, page = _new(owner, "distribution_summaries")
    widget = page._appearance_widget
    names = {"measurements": {"label": "Apo"}}
    _unbind(page)
    widget.apply_appearance(replace(widget.state(), source_appearance=names, title="Mine",
                                    title_size=20.0))
    kept_basis = widget.state().quantity_display
    assert widget.state().style_binding is None

    explorer.toolbar().reset_figure_button.click()

    state = widget.state()
    assert state.style_binding and state.quantity_display == kept_basis
    assert page.source_appearance_state() == names and state.source_appearance == names
    assert (state.title, state.title_size) == (None, 7.0)
    assert widget.undo_button.toolTip() == "Undo: reset figure"


# ==================================================================================================
# 2. what each renderer draws
# ==================================================================================================

@pytest.mark.parametrize("styled", [True, False], ids=["a style", "no style"])
def test_reset_figure_draws_a_new_figures_look_on_a_pyqtgraph_page(project, styled):
    _window, owner, explorer = project
    _key, fresh = _new(owner, "iv")
    _key, page = _new(owner, "iv")

    def drawn(window):
        plot = window.plot_item()
        return (window._plot_widget.backgroundBrush().color().name(),
                bool(plot.getAxis("bottom").grid), bool(plot.getAxis("left").grid),
                plot.getAxis("bottom").style["tickFont"].family())

    assert drawn(page) == drawn(fresh)
    assert drawn(page)[:3] == ("#ffffff", False, False)
    if not styled:
        _unbind(page)               # a look of its own: the figure is drawn, and bound to nothing
    _customise("iv", page)
    assert drawn(page)[:3] == ("#151a21", True, True) and drawn(page) != drawn(fresh)
    explorer.toolbar().reset_figure_button.click()
    assert drawn(page) == drawn(fresh)


@pytest.mark.parametrize("styled", [True, False], ids=["a style", "no style"])
def test_reset_figure_draws_the_projects_style_on_distributions(project, styled):
    _window, owner, explorer = project
    _key, fresh = _new(owner, "distributions")
    _key, page = _new(owner, "distributions")
    if not styled:
        _unbind(page)
    _customise("distributions", page)
    style = page.style()
    assert (style.background, style.show_grid, style.title_font_size_pt) == ("dark", True, 20.0)

    explorer.toolbar().reset_figure_button.click()

    style = page.style()
    assert style == fresh.style()
    assert (style.background, style.show_grid, style.font_family, style.title_font_size_pt,
            style.title, style.x_label) == ("light", False, "Arial", 7.0, "", "")
    assert style.style_binding, "Reset figure dropped the figure's style"
    assert style.style_binding["ref"] == owner.default_style().ref.to_dict()
    assert page._plot_widget.backgroundBrush().color().name() == "#ffffff"
    assert not page._plot.getAxis("bottom").grid


@pytest.mark.parametrize("styled", [True, False], ids=["a style", "no style"])
def test_reset_figure_draws_the_projects_style_on_a_matplotlib_page(project, monkeypatch, styled):
    from nanoscope.plot import comparison_figures as renderer

    from .test_comparison_figures_panel import source

    drawn = []
    real = renderer.render_comparison

    def spy(*args, **kwargs):
        figure = real(*args, **kwargs)
        axis = next(ax for ax in figure.axes if ax.get_visible())
        drawn.append({"title": figure._suptitle.get_fontsize() if figure._suptitle else None,
                      "ticks": axis.get_xticklabels()[0].get_fontsize(),
                      "label": axis.xaxis.label.get_fontsize(),
                      "family": axis.xaxis.label.get_fontname(),
                      "face": figure.get_facecolor()})
        return figure

    monkeypatch.setattr(renderer, "render_comparison", spy)
    _window, owner, explorer = project
    _key, page = _new(owner, "comparison_figures")
    sources = [source(), source("second", "Recording B")]
    page.set_sources(sources, selected_keys=[s.key for s in sources])
    assert page.analyze() and page.has_data()
    new_look = drawn[-1]
    assert (new_look["title"], new_look["ticks"], new_look["label"]) == (7.0, 6.0, 7.0)
    if not styled:
        _unbind(page)
        assert drawn[-1] != new_look, "premise: with no style the figure is drawn in its own look"

    page._appearance_widget.apply_appearance(replace(
        page._appearance_widget.state(), title_size=20.0, axis_size=14.0, tick_size=12.0,
        background="dark"))
    assert (drawn[-1]["title"], drawn[-1]["ticks"], drawn[-1]["face"]) != (
        new_look["title"], new_look["ticks"], new_look["face"])

    explorer.toolbar().reset_figure_button.click()

    assert drawn[-1] == new_look, "the figure is drawn in the style it is bound to, not matplotlib's"


# ==================================================================================================
# 3. what Reset figure keeps, and the figures it does not change
# ==================================================================================================

@pytest.mark.parametrize("kind", ["iv", "distributions", "comparison_figures"])
def test_reset_figure_keeps_the_style_a_figure_was_given(project, kind):
    """A figure bound to another style (Style ▾) stays bound to it: Reset figure is Reset look
    plus the words and pins, not a return to the project's default."""
    _window, owner, explorer = project
    key, page = _new(owner, kind)
    widget, adapter = page._appearance_widget, explorer.adapter(key)
    ieee = builtin_styles()[2]
    vocabulary = "matplotlib_appearance" if kind == "comparison_figures" else "plot_appearance"
    assert adapter.apply_look(PlotAppearance.from_mapping(
        bind_native_style({}, ieee, vocabulary)), False)
    given = widget.state()
    assert (given.font_family, given.title_size) == ("Times New Roman", 9.0)
    _customise(kind, page)

    explorer.toolbar().reset_figure_button.click()

    state = widget.state()
    assert state.style_binding, "Reset figure dropped the figure's style"
    assert state.style_binding["ref"] == ieee.ref.to_dict()
    assert dict(state.style_binding["overrides"]) == {}
    # (Distributions rounds a line width to whole pixels, so its own overrides, which Reset takes
    # off, make the style's line width and marker size differ in the last digits from the ones the
    # window put on the figure when the style was applied. Every other look field is equal.)
    look = ("font_family", "title_size", "axis_size", "tick_size", "legend_size", "background",
            "grid", "data_color", "fit_color", "title", "x_label", "y_label")
    assert [getattr(state, name) for name in look] == [
        getattr(given, name) for name in look[:9]] + [None] * 3
    assert (state.font_family, state.title_size, state.background) == ("Times New Roman", 9.0, "light")
    assert explorer.toolbar().style_button.text() == "9 pt Times New Roman · … · r1 ▾"


def test_reset_figure_keeps_distributions_font_size_body_as_it_always_did(project):
    _window, owner, explorer = project
    _key, page = _new(owner, "distributions")
    page.set_font_size(13)
    _customise("distributions", page)
    assert dict(page.style().style_binding["overrides"])["body_size_pt"] == 13.0

    explorer.toolbar().reset_figure_button.click()

    style = page.style()
    assert style.font_size_pt == 13.0, "Font size (body) is not part of the shared look"
    assert dict(style.style_binding["overrides"]) == {"body_size_pt": 13.0}
    assert (style.background, style.show_grid, style.title_font_size_pt, style.title) == (
        "light", False, 7.0, "")


def test_reset_figure_keeps_the_drawn_quantity_and_the_source_names(project):
    _window, owner, explorer = project
    _key, page = _new(owner, "distribution_summaries")
    widget = page._appearance_widget
    names = {"measurements": {"label": "Apo"}}
    widget.apply_appearance(replace(widget.state(), source_appearance=names, title="Mine",
                                    title_size=20.0))
    kept_basis = widget.state().quantity_display
    steps = _depth("distribution_summaries", page)

    explorer.toolbar().reset_figure_button.click()

    state = widget.state()
    assert state.quantity_display == kept_basis
    assert page.source_appearance_state() == names and state.source_appearance == names
    assert (state.title, state.title_size) == (None, 7.0)
    assert _depth("distribution_summaries", page) == (steps[0] + 1, 0)
    assert widget.undo_button.toolTip() == "Undo: reset figure"


def test_reset_figure_pressed_again_has_nothing_to_do(project):
    _window, owner, explorer = project
    _key, page = _new(owner, "iv")
    _customise("iv", page)
    button = explorer.toolbar().reset_figure_button
    button.click()
    steps = _depth("iv", page)
    button.click()
    assert _depth("iv", page) == steps, "a figure already at its style records no step"
    _key, untouched = _new(owner, "iv")
    steps = _depth("iv", untouched)
    explorer.toolbar().reset_figure_button.click()
    assert _depth("iv", untouched) == steps == (0, 0), "nor does a figure nobody touched"


def test_a_figure_with_no_style_in_a_project_is_bound_to_the_projects_default_by_reset_figure(project):
    """A figure bound to no style (what the old Reset figure left in a saved project, and what New
    figure > From recipe makes: its Style button says "Style: custom") is bound to the project's
    default style by Reset figure, as a new figure is, and not put back to the old dark, gridded
    look that no new figure has."""
    _window, owner, explorer = project
    _fresh_key, fresh = _new(owner, "iv")
    _key, page = _new(owner, "iv")
    widget = page._appearance_widget
    page.set_appearance_state({"title": "Old project", "title_size": 14.0, "background": "dark"})
    assert widget.state().style_binding is None and widget.state().title == "Old project"
    old = widget.state()

    explorer.toolbar().reset_figure_button.click()

    state = widget.state()
    assert state == fresh._appearance_widget.state()
    assert state.style_binding and state.style_binding["ref"] == owner.default_style().ref.to_dict()
    assert dict(state.style_binding["overrides"]) == {}
    assert (state.background, state.grid, state.font_family, state.title) == ("light", False, "Arial", None)
    assert widget.undo_button.toolTip() == "Undo: reset figure"
    assert explorer.toolbar().style_button.text() == DEFAULT_STYLE_BUTTON
    assert widget.undo() and widget.state() == old, "one Undo brings the figure with no style back"


def test_a_bare_window_with_no_style_keeps_the_plain_reset():
    """A window that belongs to no project has no style to bind and no project default: its Reset
    figure goes back to the plain defaults, as it always did."""
    bare = PlotAppearanceWidget()
    bare.apply_appearance(PlotAppearance(title="T", title_size=14.0, background="dark"))
    bare.reset_button.click()
    assert bare.state() == PlotAppearance()
    assert bare.state().style_binding is None
    assert bare.undo_button.toolTip() == "Undo: reset figure"
    assert bare.undo() and bare.state().title == "T"
    assert bare.redo() and bare.redo_button.toolTip() == "Nothing to redo."


def test_a_project_whose_styles_cannot_be_read_resets_a_figure_with_no_style_plain(project):
    """New figures are refused there (a project's styles are what they are bound to), so there is
    no look of a new figure to return to: the figure goes back to the plain defaults, as before."""
    _window, owner, explorer = project
    _key, page = _new(owner, "iv")
    widget = page._appearance_widget
    page.set_appearance_state({"title": "Old project", "title_size": 14.0, "background": "dark"})
    owner._styles = None

    explorer.toolbar().reset_figure_button.click()

    assert widget.state().style_binding is None and widget.state().title is None
    assert not {k: v for k, v in widget.state().to_dict().items() if k != "quantity_display"}
    assert widget.undo_button.toolTip() == "Undo: reset figure"


def test_the_study_figure_editor_keeps_its_style_on_reset_figure(qapp, qtbot):
    """The study and report editors share the widget: styled with Style ▾, Reset figure returns
    the figure to that style and takes its words off."""
    editor = _study_window()
    qtbot.addWidget(editor)
    editor.show()
    widget = editor._appearance_widget
    ieee = builtin_styles()[2]
    assert editor.apply_style(ieee)
    widget.apply_appearance(replace(widget.state(), title="Study", title_size=16.0, background="dark"))
    apply_button = next(b for b in editor.findChildren(QtWidgets.QPushButton)
                        if b.text() == "Apply appearance")
    apply_button.click()
    assert editor._figure.get_suptitle() == "Study" and editor._figure.axes[0].title.get_fontsize() == 16.0

    widget.reset_button.click()
    apply_button.click()

    assert editor.style().get("style_binding"), "Reset figure dropped the figure's style"
    assert editor.style()["style_binding"]["ref"] == ieee.ref.to_dict()
    assert editor.style()["style_binding"]["overrides"] == {}
    assert editor._figure.get_suptitle() == ""
    assert editor._figure.axes[0].title.get_fontsize() == 9.0
    assert editor._figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    assert editor.undo_style() and editor._figure.get_suptitle() == "Study"
    editor.close()


# ==================================================================================================
# 4. a saved project
# ==================================================================================================

def test_a_project_saved_after_reset_figure_reopens_with_its_style(project):
    _window, owner, explorer = project
    key, page = _new(owner, "noise")
    _customise("noise", page)
    explorer.toolbar().reset_figure_button.click()
    assert owner.checkpoint(key)
    saved = owner.get(key).spec.presentation
    style = owner.default_style()
    assert saved.style_ref == style.ref.to_dict() and not saved.overrides

    owner.install_workspace(owner.capture_workspace())

    assert owner.get(key).spec.presentation == saved
    owner.activate(key)
    reopened = owner.page(key)._appearance_widget.state()
    assert reopened.style_binding and reopened.style_binding["ref"] == style.ref.to_dict()
    assert (reopened.background, reopened.grid, reopened.title) == ("light", False, None)


def test_a_figure_saved_without_a_style_by_the_old_reset_still_opens_and_reset_figure_binds_it(project):
    """What the old Reset figure left in a project: a figure bound to no style. It opens as it
    was saved, and Reset figure now gives it the project's default style, saved with the project."""
    _window, owner, explorer = project
    _fresh_key, fresh = _new(owner, "iv")
    key, page = _new(owner, "iv")
    page.set_appearance_state(PlotAppearance())       # what the old Reset figure left
    assert owner.checkpoint(key) and owner.get(key).spec.presentation.style_ref is None

    owner.install_workspace(owner.capture_workspace())
    owner.activate(key)
    reopened = owner.page(key)._appearance_widget

    assert reopened.state().style_binding is None and reopened.state().title is None
    reopened.apply_appearance(replace(reopened.state(), title="Again", title_size=15.0))
    explorer.toolbar().reset_figure_button.click()
    assert reopened.state() == fresh._appearance_widget.state()
    assert reopened.state().style_binding["ref"] == owner.default_style().ref.to_dict()
    assert owner.checkpoint(key)
    assert owner.get(key).spec.presentation.style_ref == owner.default_style().ref.to_dict()

    owner.install_workspace(owner.capture_workspace())
    owner.activate(key)
    assert owner.page(key)._appearance_widget.state() == fresh._appearance_widget.state()


def test_reset_figure_keeps_the_witness_the_source_names_and_the_legend_names_of_a_styled_figure(
        qtbot):
    witness = {"version": 1, "axes": {"left": {"quantity": {"key": "median_depth"}}}}
    names = {"first": {"label": "Apo"}}
    entry = TextStyle(text="Apo", bold=True)
    widget = PlotAppearanceWidget(
        _bound(title="T", title_size=20.0, quantity_display=witness, source_appearance=names
               ).with_text_style("legend.entry.first", entry).with_text_style(
            "axis.left.label", TextStyle(italic=True)),
        reset_keeps=lambda key: key.startswith("legend.entry."))
    qtbot.addWidget(widget)

    widget.reset_button.click()

    state = widget.state()
    assert state.style_binding and state.title is None and state.title_size == 7.0
    assert state.quantity_display == witness and state.source_appearance == names
    assert dict(state.text) == {"legend.entry.first": entry}, "the legend's names stay, the rest goes"


@pytest.mark.parametrize("styled", [True, False], ids=["a style", "no style"])
@pytest.mark.parametrize("kind", ["iv", "distributions", "comparison_figures"])
def test_a_reset_figure_put_to_sleep_wakes_with_its_reset_and_its_undo(project, kind, styled):
    """The workspace puts a figure to sleep when it needs the room, keeping its appearance and
    its Undo history as plain values. The step Reset figure makes is one of them, whether the
    figure had a style or Reset figure bound it to the project's default."""
    _window, owner, explorer = project
    key, page = _new(owner, kind)
    widget = page._appearance_widget
    if not styled:
        _unbind(page)
    widget.apply_appearance(replace(widget.state(), title="Capture", title_size=20.0,
                                    background="dark"))
    edited = widget.state()
    assert bool(edited.style_binding) is styled
    explorer.toolbar().reset_figure_button.click()
    reset = widget.state()
    assert reset.style_binding and reset.title is None

    _new(owner, kind)                            # another figure comes to the front
    assert owner.suspend(key) and owner.page(key) is None
    owner.activate(key)
    widget, adapter = owner.page(key)._appearance_widget, explorer.adapter(key)

    assert widget.state() == reset
    if kind != "distributions":
        assert adapter.history().undo_text == "Undo: reset figure"
    adapter.undo()
    assert widget.state() == edited
    if kind != "distributions":
        assert adapter.history().redo_text == "Redo: reset figure"
    adapter.redo()
    assert widget.state() == reset


def test_a_history_saved_after_reset_figure_restores_and_names_the_step(qtbot):
    """What a project keeps of a figure's Undo history holds the step Reset figure makes now (a
    bound figure to its style, not to no style): it restores losslessly, is named, and walks."""
    first, second = PlotAppearanceWidget(_bound()), PlotAppearanceWidget()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.apply_appearance(replace(first.state(), title="T", title_size=20.0, background="dark"))
    first.reset_button.click()
    assert first.state() == _bound()

    capsule = first.capture_history_capsule()
    second.set_state(first.state())
    second.restore_history_capsule(capsule)

    assert second.undo_button.toolTip() == "Undo: reset figure"
    assert second.undo() and (second.state().title, second.state().background) == ("T", "dark")
    assert second.redo() and second.state() == _bound()
    assert second.redo_button.toolTip() == "Nothing to redo."


# ==================================================================================================
# 5. the Undo and Redo tooltip
# ==================================================================================================

def test_describe_step_names_a_reset_of_a_styled_figure():
    plain = _bound()
    on_top = PlotAppearance.from_mapping({
        **plain.to_dict(), "title": "T", "title_size": 20.0,
        "style_binding": {**plain.to_dict()["style_binding"], "overrides": {"title_size_pt": 20.0}}})
    assert _describe_step(on_top, plain) == "reset figure"
    # The drawn-quantity record and the source names go through a Reset figure.
    witness = {"version": 1, "axes": {"left": {"quantity": {"key": "median_depth"}}}}
    assert _describe_step(replace(on_top, quantity_display=witness, source_appearance={}),
                          replace(plain, quantity_display=witness, source_appearance={})
                          ) == "reset figure"
    # A figure with no style is named as before; an empty record of source names (Distribution
    # summaries and Comparison figures hold one) is no override either.
    assert _describe_step(PlotAppearance(title="T", title_size=14.0, source_appearance={}),
                          PlotAppearance(source_appearance={})) == "reset figure"
    # A word alone keeps the name of that word, as for a figure with no style.
    assert _describe_step(replace(plain, title="T"), plain) == "title ‘T’ → automatic"


def test_describe_step_does_not_call_another_style_or_a_figure_with_its_own_look_a_reset():
    arial, times = builtin_styles()[0], builtin_styles()[2]
    # Binding another style moves the look, the binding and its name: that is not a reset.
    assert _describe_step(_bound(arial), _bound(times)).endswith("appearance changes")
    assert _describe_step(PlotAppearance(title_size=14.0, grid=True), _bound(arial)
                          ).endswith("appearance changes")
    # Still carrying a word, a pin, an element style or an override of the look: not plain.
    plain = _bound()
    on_top = [replace(plain, title="T"),
              plain.with_axis_pin("left", AxisPin(min=0.0, max=1.0)),
              plain.with_text_style("title", TextStyle(bold=True)),
              plain.with_annotation_offset("annotation.provenance", (1.0, 2.0)),
              replace(plain, fit_label=FitLabelState(template="Fit")),
              replace(plain, legend_visible=False)]
    for after in on_top:
        assert _describe_step(_bound(title="T", title_size=20.0, grid=True), after
                              ).endswith("appearance changes"), after
    overridden = PlotAppearance.from_mapping({
        **plain.to_dict(), "title_size": 20.0, "style_binding": {
            **plain.to_dict()["style_binding"], "overrides": {"title_size_pt": 20.0}}})
    assert _describe_step(replace(plain, title="T", grid=True), overridden
                          ).endswith("appearance changes")


def _hand_set(state, **look):
    """``state`` with ``look`` set by hand on top of its style, as one look control sets it: the
    field itself and the binding's record of that override (the two a single edit moves)."""
    from nanoscope.gui.figure_style_binding import NATIVE_LOOK_KEYS

    binding = state.to_dict()["style_binding"]
    overrides = {**binding["overrides"], **{NATIVE_LOOK_KEYS[name]: value for name, value in look.items()}}
    return PlotAppearance.from_mapping(
        {**state.to_dict(), **look, "style_binding": {**binding, "overrides": overrides}})


def test_describe_step_reads_a_look_control_of_a_styled_figure_as_that_control_both_ways():
    """One look control edited on a figure bound to a style moves two fields, the look and the
    binding's record of the override, while the style stays what it was. It is that control's step,
    in both directions: neither a count of two changes nor a reset nobody pressed."""
    plain = _bound()
    assert _describe_step(plain, _hand_set(plain, title_size=20.0)) == "title size 7 → 20 pt"
    assert _describe_step(_hand_set(plain, title_size=20.0), plain) == "title size 20 → 7 pt"
    assert _describe_step(plain, _hand_set(plain, background="dark")) == "background light → dark"
    assert _describe_step(_hand_set(plain, background="dark"), plain) == "background dark → light"
    assert _describe_step(plain, _hand_set(plain, grid=True)) == "grid hidden → shown"
    assert _describe_step(_hand_set(plain, grid=True), plain) == "grid shown → hidden"
    # Two controls are two changes; the record of them is no third.
    assert _describe_step(plain, _hand_set(plain, title_size=20.0, axis_size=14.0)
                          ) == "2 appearance changes"
    # A binding that moved without any control (an override the controls do not own) keeps its name.
    binding = plain.to_dict()["style_binding"]
    other = PlotAppearance.from_mapping({**plain.to_dict(), "style_binding": {
        **binding, "overrides": {"body_size_pt": 11.0}}})
    assert _describe_step(plain, other).startswith("figure style 7 pt Arial")
    # And a word edited on a styled figure is as it was.
    assert _describe_step(plain, replace(plain, title="T")) == "title automatic → ‘T’"


def test_describe_step_reads_how_a_step_that_binds_a_style_was_made():
    """A Reset figure binds the project's default style to a figure with no style, and a style
    chosen in Style ▾ can bind that same one to the same values: only the one who made the step can
    say which it was, and the values alone never call it a reset."""
    arial = builtin_styles()[0]
    unbound = PlotAppearance(title="T", title_size=14.0, background="dark", grid=True)
    assert _describe_step(unbound, _bound(arial), reset=True) == "reset figure"
    assert _describe_step(unbound, _bound(arial), reset=False).endswith("appearance changes")
    assert _describe_step(unbound, _bound(arial)).endswith("appearance changes")
    # A figure that has a style: Style ▾ giving it its own style back is not a reset either, though
    # the values alone read it as one (as a step restored from saved history is read).
    on_top = _hand_set(_hand_set(_bound(arial), title_size=20.0), axis_size=14.0)
    assert _describe_step(on_top, _bound(arial)) == "reset figure"
    assert _describe_step(on_top, _bound(arial), reset=True) == "reset figure"
    assert _describe_step(on_top, _bound(arial), reset=False) == "2 appearance changes"
    # The word does not make a step a reset: another style, words and overrides left, a single
    # control and a single word are what they were.
    times = builtin_styles()[2]
    assert _describe_step(_bound(arial), _bound(times), reset=True).endswith("appearance changes")
    assert _describe_step(unbound, replace(_bound(arial), title="T"), reset=True
                          ).endswith("appearance changes")
    assert _describe_step(_bound(arial), _hand_set(_bound(arial), title_size=20.0), reset=True
                          ) == "title size 7 → 20 pt"
    assert _describe_step(PlotAppearance(title="T"), PlotAppearance(), reset=True
                          ) == "title ‘T’ → automatic"


@pytest.mark.parametrize("kind", ["iv", "voltage_series", "comparison_figures"])
def test_one_look_control_typed_on_a_project_figure_and_back_reads_as_that_control(project, kind):
    """The report: on a figure of a project, typing a look control back to Default read "Undo: reset
    figure" (and typing it, "2 appearance changes"). Each reads as the one control it is, and Redo
    names the same step."""
    _window, owner, explorer = project
    _key, page = _new(owner, kind)
    widget = page._appearance_widget
    assert widget.state().style_binding and widget.state().title_size == 7.0

    widget.title_size.setValue(20.0)
    widget._close_burst()
    assert dict(widget.state().style_binding["overrides"]) == {"title_size_pt": 20.0}
    assert widget.undo_button.toolTip() == "Undo: title size 7 → 20 pt"

    widget.title_size.setValue(0)              # the control's Default
    widget._close_burst()
    assert dict(widget.state().style_binding["overrides"]) == {} and widget.state().title_size == 7.0
    assert widget.undo_button.toolTip() == "Undo: title size 20 → 7 pt"

    assert widget.undo()
    assert widget.redo_button.toolTip() == "Redo: title size 20 → 7 pt"
    assert widget.undo_button.toolTip() == "Undo: title size 7 → 20 pt"
    assert widget.redo() and widget.undo_button.toolTip() == "Undo: title size 20 → 7 pt"

    widget.grid.setCurrentIndex(widget.grid.findData(True))
    widget._close_burst()
    assert widget.undo_button.toolTip() == "Undo: grid hidden → shown"
    widget.title_size.setValue(18.0)
    widget.axis_size.setValue(12.0)
    widget._close_burst()
    assert widget.undo_button.toolTip() == "Undo: 2 appearance changes"
    # Only a real Reset figure, which takes several things off, reads as one.
    explorer.toolbar().reset_figure_button.click()
    assert widget.undo_button.toolTip() == "Undo: reset figure"


def test_a_widget_binds_what_its_page_says_a_new_figure_looks_like(qtbot):
    """The widget's side of Reset figure: its page (in a project) hands it a look to bind a figure
    with no style to, and it asks when the button is pressed, never before; a figure that has a
    style keeps its own, and no look (or no answer) is the plain reset."""
    arial, times = builtin_styles()[0], builtin_styles()[2]
    asked = []
    chosen = [arial]

    def look():
        asked.append(1)
        return None if chosen[0] is None else _bound(chosen[0])

    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    widget.set_new_figure_look(look)
    widget.apply_appearance(PlotAppearance(title="T", title_size=14.0, background="dark"))
    assert asked == [], "the page's answer is read at the press, not before"

    chosen[0] = times                    # the project's default changed since
    widget.reset_button.click()
    assert asked == [1] and widget.state() == _bound(times)
    assert widget.undo_button.toolTip() == "Undo: reset figure"
    assert widget.undo() and widget.state().style_binding is None
    assert widget.redo_button.toolTip() == "Redo: reset figure"

    widget.redo()
    widget.apply_appearance(replace(widget.state(), title="Again", title_size=20.0))
    chosen[0] = arial
    widget.reset_button.click()          # it has a style: that one is kept, nothing is asked
    assert asked == [1] and widget.state() == _bound(times)

    plain = PlotAppearanceWidget()
    qtbot.addWidget(plain)
    for answer in (None, "no answer"):
        plain.set_new_figure_look(None if answer == "no answer" else (lambda: None))
        plain.apply_appearance(PlotAppearance(title="T", title_size=14.0))
        plain.reset_button.click()
        assert plain.state() == PlotAppearance(), answer


def test_a_history_remembers_how_each_step_was_made_through_undo_redo_and_the_ceiling():
    plain, other = PlotAppearance(), PlotAppearance(title="T")
    history = _AppearanceHistory(plain)
    history.hint_next(True)
    assert history.note(other) and history.last_hint() is True
    history.hint_next(False)
    assert history.note(plain) and history.last_hint() is False
    assert history.note(replace(other, title="U")) and history.last_hint() is None, "said nothing"
    assert history.undo() and history.next_hint() is None and history.last_hint() is False
    assert history.undo() and history.next_hint() is False and history.last_hint() is True
    assert history.redo() and history.last_hint() is False
    assert history.redo() and history.last_hint() is None
    assert history.undo() and history.undo() and history.last_hint() is True, "after a Redo"
    assert history.redo() and history.redo() and history.last_hint() is None
    # A hint is for the next step that is recorded: used up by a call that records none.
    history.hint_next(True)
    assert not history.note(history.base)
    assert history.note(replace(other, title="V")) and history.last_hint() is None
    # What saved history says nothing of, it does not claim.
    values, hints = history.sequence()
    history.restore(values[-1], values[:-1], [], 0)
    assert history.last_hint() is None and history.sequence()[1] == [None] * len(values)
    history.restore(values[-1], values[:2], values[2:-1], 0, [True, False, None, None, None])
    assert history.sequence()[1] == [True, False, None, None, None]
    history.restore(values[-1], values[:2], values[2:-1], 0, [True] * 9)   # more than it holds
    assert history.sequence()[1] == [True] * 5 and history.depth() == (2, 2)
    history.restore(values[-1], values[:2], values[2:-1], 0, [False])    # fewer
    assert history.sequence()[1] == [False, None, None, None, None]
    # The ceiling drops the oldest steps with their hints, and keeps the rest in step.
    many = _AppearanceHistory(plain)
    for number in range(APPEARANCE_HISTORY_DEPTH + 5):
        many.hint_next(number % 2 == 0)
        many.note(PlotAppearance(title=str(number)))
    assert many.depth()[0] == APPEARANCE_HISTORY_DEPTH and many.dropped == 5
    assert many.last_hint() is ((APPEARANCE_HISTORY_DEPTH + 4) % 2 == 0)
    many.undo()
    assert many.last_hint() is ((APPEARANCE_HISTORY_DEPTH + 3) % 2 == 0)
    assert many.next_hint() is ((APPEARANCE_HISTORY_DEPTH + 4) % 2 == 0)


def test_a_history_forgets_the_hint_of_a_value_that_is_no_longer_the_one_the_step_arrived_at():
    plain, other = PlotAppearance(), PlotAppearance(title="T")
    witness = {"version": 1, "axes": {"left": {"quantity": {"key": "median_depth"}}}}
    history = _AppearanceHistory(plain)
    history.hint_next(True)
    history.note(other)
    history.rebase(replace(other, quantity_display=witness))     # the renderer's own record moves
    assert history.last_hint() is True
    history.rebase(replace(other, title="Typed elsewhere"))      # another value altogether
    assert history.last_hint() is None


@pytest.mark.parametrize("kind", ["iv", "comparison_figures"])
def test_a_style_step_put_to_sleep_is_still_not_a_reset_figure_when_it_wakes(project, kind):
    """The values alone read a style that gives a figure back its own style as a reset; the step
    remembers that it was a Style ▾, through the sleep of its figure."""
    _window, owner, explorer = project
    key, page = _new(owner, kind)
    widget = page._appearance_widget
    widget.apply_appearance(replace(widget.state(), title_size=20.0, axis_size=14.0))
    vocabulary = "matplotlib_appearance" if kind == "comparison_figures" else "plot_appearance"
    explorer.apply_look(PlotAppearance.from_mapping(
        bind_native_style({}, owner.default_style(), vocabulary)), scope="page", keep_hand=False)
    assert widget.undo_button.toolTip() == "Undo: 2 appearance changes"

    _new(owner, kind)
    assert owner.suspend(key) and owner.page(key) is None
    owner.activate(key)

    assert explorer.adapter(key).history().undo_text == "Undo: 2 appearance changes"


@pytest.mark.parametrize("styled", [True, False], ids=["a style", "no style"])
@pytest.mark.parametrize("kind", [kind for kind in PAGE_IDS if kind != "distributions"])
def test_a_style_picked_while_the_figure_sleeps_is_named_as_it_is_on_a_figure_that_is_awake(
        project, kind, styled):
    """Style ▾ reaches a figure that is asleep as a step on the history it keeps as plain values,
    and the step reads, once the figure wakes, as the same pick reads on a figure that is awake:
    never "reset figure", which the values alone call a style handed back to a figure that has it."""
    _window, owner, _explorer = project
    vocabulary = "matplotlib_appearance" if kind == "comparison_figures" else "plot_appearance"
    look = PlotAppearance.from_mapping(bind_native_style({}, owner.default_style(), vocabulary))

    def made():
        key, page = _new(owner, kind)
        if not styled:
            _unbind(page)
        widget = page._appearance_widget
        widget.apply_appearance(replace(widget.state(), title_size=20.0, axis_size=14.0))
        return key

    awake, asleep = made(), made()
    assert owner.suspend(asleep) and owner.page(asleep) is None
    assert owner.apply_look_to_figure(awake, look, keep_hand=False)
    assert owner.apply_look_to_figure(asleep, look, keep_hand=False) and owner.page(asleep) is None
    named = owner.history(awake).undo_text
    assert named.endswith("appearance changes"), "premise: a style is a step of several changes"

    assert owner.activate(asleep)

    assert owner.history(asleep).undo_text == named != "Undo: reset figure"
    assert owner.history(asleep).redo_text == "Nothing to redo."


def test_a_style_picked_for_a_sleeping_figure_does_not_take_the_name_of_a_reset_figure_that_was_undone(
        project):
    """Undone, the step Reset figure made waits on the redo side of the figure's history with its
    name. A style picked while the figure sleeps ends that branch, and lands on the very value the
    Reset had: the step the pick makes is a Style ▾, not the Reset figure that is gone."""
    _window, owner, explorer = project
    key, page = _new(owner, "iv")
    widget = page._appearance_widget
    _unbind(page)
    explorer.toolbar().reset_figure_button.click()
    reset = widget.state()
    assert widget.undo_button.toolTip() == "Undo: reset figure"
    assert widget.undo() and widget.redo_button.toolTip() == "Redo: reset figure"

    _new(owner, "iv")
    assert owner.suspend(key) and owner.page(key) is None
    look = PlotAppearance.from_mapping(bind_native_style({}, owner.default_style()))
    assert owner.apply_look_to_figure(key, look, keep_hand=False)
    assert owner.activate(key)

    widget = owner.page(key)._appearance_widget
    assert widget.state() == reset, "premise: the pick landed on the value the Reset figure had"
    assert widget.undo_button.toolTip() != "Undo: reset figure"
    assert widget.undo_button.toolTip().endswith("appearance changes")
    assert widget.redo_button.toolTip() == "Nothing to redo."
    assert widget.undo() and widget.state().style_binding is None


def test_a_dormant_step_carries_how_it_was_made_and_ends_the_hints_of_the_branch_it_cuts_off(qtbot):
    from nanoscope.gui.plot_appearance import note_dormant_step

    plain, other = PlotAppearance(), PlotAppearance(title="T")

    def capsule(base, past=(), future=(), hints=(), dropped=0):
        return HistoryCapsule(
            "nanoscope.appearance-history/1", base.to_dict(), tuple(s.to_dict() for s in past),
            tuple(s.to_dict() for s in future), dropped,
            {"step_hints": [{"at": at, "of": _fingerprint(state), "reset": reset}
                            for at, state, reset in hints]} if hints else {})

    def hints(held):
        return [(entry["at"], entry["of"], entry["reset"])
                for entry in thaw_json(held.extra.get("step_hints", []))]

    # A step noted after one that was undone: the redo step's hint goes with the step, and the new
    # step's own is its value's, though it is the same value.
    undone = capsule(plain, future=[other], hints=[(1, other, True)])
    noted = note_dormant_step(undone, other, reset=False)
    assert thaw_json(noted.base) == other.to_dict() and noted.future == ()
    assert thaw_json(noted.past) == [plain.to_dict()]
    assert hints(noted) == [(1, _fingerprint(other), False)]

    # Said nothing (reset=None), it records the step and leaves no hint for it; the hints of the
    # steps behind it stay where they were.
    kept = capsule(other, past=[plain], hints=[(0, plain, True), (1, other, True)])
    later = note_dormant_step(kept, PlotAppearance(title="U"), reset=None)
    assert hints(later) == [(0, _fingerprint(plain), True), (1, _fingerprint(other), True)]
    assert thaw_json(later.past) == [plain.to_dict(), other.to_dict()] and later.future == ()

    # A value that is already the base is no step: the capsule comes back as it was.
    assert note_dormant_step(kept, other, reset=False) is kept

    # At the ceiling the oldest step goes, and the hints move up with the values they belong to.
    values = [PlotAppearance(title=str(n)) for n in range(APPEARANCE_HISTORY_DEPTH + 1)]
    full = capsule(values[-1], past=values[:-1], hints=[
        (0, values[0], True), (1, values[1], True), (APPEARANCE_HISTORY_DEPTH, values[-1], True)])
    over = note_dormant_step(full, PlotAppearance(title="new"), reset=False)
    assert over.dropped == 1 and len(over.past) == APPEARANCE_HISTORY_DEPTH
    assert [(at, reset) for at, _of, reset in hints(over)] == [
        (0, True), (APPEARANCE_HISTORY_DEPTH - 1, True), (APPEARANCE_HISTORY_DEPTH, False)]
    # ...and a widget that takes the capsule reads each hint on the value it names.
    widget = PlotAppearanceWidget(PlotAppearance(title="new"))
    qtbot.addWidget(widget)
    widget.restore_history_capsule(over)
    assert widget._history.last_hint() is False
    assert widget.undo() and widget._history.last_hint() is True


def test_the_study_figure_editor_resets_a_figure_with_no_style_to_the_plain_look(qapp, qtbot):
    """The study and report editors make their figures in matplotlib's own look and bind a style
    only when one is chosen (Style ▾): a figure of theirs with none has no project default to be
    bound to, so Reset figure puts it back to that plain look, as a new editor figure is drawn."""
    editor = _study_window()
    qtbot.addWidget(editor)
    editor.show()
    widget = editor._appearance_widget
    assert not editor.style().get("style_binding")
    widget.apply_appearance(replace(widget.state(), title="Study", title_size=16.0, background="dark"))
    apply_button = next(b for b in editor.findChildren(QtWidgets.QPushButton)
                        if b.text() == "Apply appearance")
    apply_button.click()
    assert editor._figure.get_suptitle() == "Study" and editor._figure.axes[0].title.get_fontsize() == 16.0

    widget.reset_button.click()
    apply_button.click()

    assert not editor.style().get("style_binding") and editor.style()["appearance"] == {}
    assert editor._figure.get_suptitle() == ""
    assert editor.undo_style() and editor._figure.get_suptitle() == "Study"
    editor.close()


@pytest.mark.parametrize("body", [13.0, 7.0], ids=["its own body size", "the style's body size"])
def test_reset_figure_keeps_the_font_size_body_of_a_distributions_figure_with_no_style(project, body):
    """Font size (body) is the one part of its look that Distributions keeps through Reset figure
    (the button's tool tip and the guide say so), whether the figure has a style or Reset figure
    is what binds the project's default to it. The figure takes the style's look and goes on with
    the body size it had, which rides on the binding as an override, as it does on a figure that
    was made with the style. A body size the style gives anyway is nothing to keep: a new figure's."""
    _window, owner, explorer = project
    _key, fresh = _new(owner, "distributions")
    _key, page = _new(owner, "distributions")
    _unbind(page)
    page.set_font_size(body)
    assert page.style().font_size_pt == body and not page.style().style_binding
    unbound = page.style()

    explorer.toolbar().reset_figure_button.click()

    style = page.style()
    assert style.font_size_pt == body
    assert style.style_binding["ref"] == owner.default_style().ref.to_dict()
    # The rest is a new figure's: the style's look, and the words and pins gone.
    assert (style.background, style.show_grid, style.font_family, style.title_font_size_pt,
            style.axis_label_font_size_pt, style.tick_font_size_pt, style.title, style.x_label) == (
        "light", False, "Arial", 7.0, 7.0, 6.0, "", "")
    assert (_overrides_taken_off(page._appearance_widget.state())
            == _overrides_taken_off(fresh._appearance_widget.state()))
    if body == fresh.style().font_size_pt:
        assert dict(style.style_binding["overrides"]) == {}
        assert style == fresh.style() and page.appearance_state() == fresh.appearance_state()
    else:
        assert dict(style.style_binding["overrides"]) == {"body_size_pt": body}

    # One Undo brings back the figure with no style and its body size, and Redo the reset.
    assert page.undo_config()
    assert page.style() == unbound and not page.style().style_binding
    assert page.redo_config() and page.style() == style


def test_reset_figure_keeps_the_font_size_body_in_a_project_with_an_older_default_look(project):
    """The look a project from an older build gives its new figures is a sparse one, and a new
    Distributions figure's body size is made from its axis and title size (12 pt here). A figure
    with no style that has another body size keeps it through Reset figure; one that has the
    look's is bound as a new figure is."""
    _window, owner, explorer = project
    registry, ref = owner.style_registry().import_legacy_style(
        "Old project look", {"title_size": 12.0, "grid": True, "background": "light"},
        "plot_appearance")
    owner._replace_styles(registry.with_default(ref))
    _key, fresh = _new(owner, "distributions")
    assert dict(fresh._appearance_widget.state().style_binding["overrides"])["body_size_pt"] == 12.0

    for body in (13.0, 12.0):
        _key, page = _new(owner, "distributions")
        _unbind(page)
        page.set_font_size(body)

        explorer.toolbar().reset_figure_button.click()

        # (This page keeps the values of the figure that the sparse look does not name, as it does
        # for every look applied to it, so only what the look names, and the body size, are compared.)
        state = page._appearance_widget.state()
        assert state.style_binding["ref"] == ref.to_dict()
        assert dict(state.style_binding["overrides"])["body_size_pt"] == body, body
        assert page.style().font_size_pt == body
        assert (state.title_size, state.grid, state.background) == (12.0, True, "light")


def test_the_workspace_names_a_new_figures_look_with_the_body_size_a_distributions_figure_keeps(project):
    """``FigureWorkspace.new_figure_look`` is the one place a new figure's look is made (``create``
    and Reset figure both ask it). A Distributions figure's own Font size (body) rides on it as an
    override of the binding when the style gives another one, and on no other kind of figure."""
    _window, owner, _explorer = project
    def overrides(look):
        return dict(look.style_binding["overrides"])

    assert overrides(owner.new_figure_look("distributions")) == {}
    assert overrides(owner.new_figure_look("distributions", body_size_pt=7.0)) == {}, "the style's own"
    assert overrides(owner.new_figure_look("distributions", body_size_pt=13)) == {"body_size_pt": 13.0}
    assert owner.new_figure_look("distributions", body_size_pt=7.0) == owner.new_figure_look(
        "distributions", owner.default_style())
    for kind in PAGE_IDS:
        if kind != "distributions":
            assert overrides(owner.new_figure_look(kind, body_size_pt=13.0)) == {}, kind


def test_a_distributions_window_keeps_the_fractional_pixel_sizes_an_appearance_gives_it(project):
    """A style of 1 pt lines and 4 pt markers is 1.3333 and 5.3333 pixels, and the (hidden) boxes
    that carry the two sizes keep three decimals. Writing an appearance onto them must not write
    their rounding back as the figure's size."""
    _window, owner, _explorer = project
    _key, page = _new(owner, "distributions")
    _unbind(page)

    page.set_appearance_state({"line_width": 4 / 3, "marker_size": 16 / 3})

    assert (page.style().line_width, page.style().point_size) == (4 / 3, 16 / 3)


@pytest.mark.parametrize("which", [1, 2], ids=["8 pt style", "9 pt style"])
def test_reset_figure_gives_a_distributions_figure_with_no_style_a_new_figures_look_for_fractional_pixel_sizes(
        project, which):
    """The 8 pt and the 9 pt journal styles have 1 pt lines and 4 pt markers: 1.3333 and 5.3333
    pixels, which the window's boxes round to three decimals. The rounding is no override of the
    figure's own: Reset figure binds the style and the figure is a new one's, with no dot on its
    Style button."""
    _window, owner, explorer = project
    owner.set_default_style(builtin_styles()[which].ref)
    _key, fresh = _new(owner, "distributions")
    bar = explorer.toolbar()
    new_button = bar.style_button.text()
    _key, page = _new(owner, "distributions")
    _unbind(page)

    bar.reset_figure_button.click()

    assert dict(page.style().style_binding["overrides"]) == {}
    assert page.appearance_state() == fresh.appearance_state()
    assert bar.style_button.text() == new_button and "●" not in new_button


@pytest.mark.parametrize("which", [1, 2], ids=["8 pt style", "9 pt style"])
def test_a_style_of_fractional_pixel_sizes_leaves_a_distributions_figure_no_override_of_its_own(
        project, which):
    """Style ▾ gives a figure the style's sizes, and nothing of its own on top of them."""
    _window, owner, explorer = project
    key, page = _new(owner, "distributions")            # bound to the project's 7 pt style
    style = builtin_styles()[which]
    look = PlotAppearance.from_mapping(bind_native_style({}, style))

    assert explorer.apply_look(look, scope="page", keep_hand=False).applied == (key,)

    assert page.style().style_binding["ref"] == style.ref.to_dict()
    assert dict(page.style().style_binding["overrides"]) == {}
    assert "●" not in explorer.toolbar().style_button.text()


def test_a_style_chosen_in_style_still_gives_a_distributions_figure_the_styles_body_size(project):
    """Only Reset figure keeps Font size (body): Style ▾ applies a whole style, body size included,
    to a figure with no style as to one that has a style."""
    _window, owner, explorer = project
    key, page = _new(owner, "distributions")
    _unbind(page)
    page.set_font_size(13)
    look = PlotAppearance.from_mapping(bind_native_style({}, owner.default_style()))

    assert explorer.apply_look(look, scope="page", keep_hand=False).applied == (key,)

    assert page.style().font_size_pt == 7.0
    assert dict(page.style().style_binding["overrides"]) == {}


# ==================================================================================================
# 6. the guide
# ==================================================================================================

def test_the_guide_says_what_reset_figure_does_to_the_look_in_a_project():
    """The Reset figure paragraph of the guide: the look goes back to the style the figure is bound
    to, a figure with no style is bound to the project's default as a new one is, and Distributions
    keeps Font size (body) either way, as the button's tool tip says."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    guide = " ".join((root / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8").split())
    start = guide.index("**Reset figure** puts the title, axis labels")
    paragraph = guide[start: guide.index("**Recipes keep analytical settings", start)]

    assert ("In a project the look goes back to the style the figure is bound to (the project's "
            "default when the figure was made, or the one you chose in **Style ▾**), as "
            "**Reset look** returns it; a figure with no style (made from a recipe, or from an "
            "older project) is bound to the project's default style first, as a new figure is, "
            "and takes that style's look. Distributions keeps its **Font size (body)** either way "
            "(see below)") in paragraph
    assert "the figure's style (the one a new figure gets" not in paragraph
    assert "goes back to the plain defaults" not in paragraph, "a figure in a project is bound"
    assert "Reset figure leaves it, and leaves **Font size (body)**" in paragraph
