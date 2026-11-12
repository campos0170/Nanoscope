"""T1: a shipped template named for its look changes the look and nothing else.

The shipped "Publication figure (light)" promised a white canvas, no grid, print-sized text, a
heavier curve and a 1600 x 1200 export, and said that every analysis field was left blank. Only the
COLUMNS were: a blank column means "keep the window's own" (:meth:`PlotConfig.with_template`), while
the plot type, the bins, the normalisation, the log axes, the fit and the typed ranges have no blank,
so the template installed the dataclass defaults for all of them. Picked on a log-log scatter with a
power-law fit, or on a log-x density histogram with an exponential fit, it produced a 30-bin linear
count histogram with no fit (and cleared the typed title and the pinned range on the way).

The shipped look template now has a BLANK PLOT TYPE, which is how a template says it names no
analysis (:attr:`PlotConfig.carries_analysis`), and is stored with its style and no analysis
section; :meth:`PlotConfig.with_template` then takes the style settings it names and nothing else.
A template with a plot type (every saved one, and the two shipped analysis templates) is recalled as
it always was.

Every gesture here is a real one: the window is built the way the Distributions tests build it, the
fit is the one the setters apply, and the template is picked in its combo (the user's door) or
handed to ``apply_template`` (the script's). Synthetic data only; offscreen.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plots_panel import (
    FrozenMap,
    PlotConfig,
    PlotsWindow,
    StyleSettings,
    plot_config_from_json,
    plot_config_to_json,
)

LOOK = "Publication figure (light)"
DWELL = "Dwell-time histogram (log-x, exponential)"
SCATTER = "Dwell vs depth scatter (log-x)"

#: What the shipped look is meant to set, and so everything it may change: a white canvas, no
#: grid, text sized for a printed column, a heavier curve and a 1600 x 1200 export.
THE_LOOK = {
    "background": "light", "show_grid": False, "font_size_pt": 11.0, "title_font_size_pt": 13.0,
    "axis_label_font_size_pt": 12.0, "tick_font_size_pt": 11.0, "line_width": 2.0,
    "export_w": 1600, "export_h": 1200,
}


def _events(n: int = 300, *, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(2.94e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "frac_depth": np.abs(rng.normal(0.30, 0.05, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _source(key: str = "doc0:0", label: str = "apo -500 mV") -> PlotSource:
    return PlotSource(key=key, label=label, df=_events(),
                      provenance={"fs": 250_000.0, "min_dwell_s": 2e-5},
                      colour="#0072B2", doc_id=key.split(":")[0], channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


@pytest.fixture(autouse=True)
def _clean_template_store():
    """The suite redirects QSettings to a temp .ini, but the store is process-wide: a template
    saved by one test would be in the next one's combo."""
    def _clear():
        s = QtCore.QSettings()
        s.remove(PlotsWindow._TPL_SETTINGS_KEY)
        s.sync()
    _clear()
    yield
    _clear()


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source()])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_time_unit("ms")
    return w


def _recall(win: PlotsWindow, name: str, door: str) -> None:
    """The template, by the user's door (the combo) or the script's (``apply_template``)."""
    if door == "combo":
        win._template_combo.setCurrentText(name)
    else:
        win.apply_template(win._BUILTIN_TEMPLATES[name])


def _typed_look_and_words(win: PlotsWindow) -> None:
    """What a person has done to the figure by hand, none of which a look may take away."""
    win.set_background("dark")
    win.set_title("my typed title")
    win.set_data_color("#cc0000")
    win.set_point_size(9)


def _a_fitted_log_log_scatter(win: PlotsWindow) -> None:
    win.set_plot_type("Scatter")
    win.set_scatter_columns("dwell_s", "depth_pA")
    win.set_scatter_log(True, True)
    win.set_scatter_fit("power law")
    assert win._applied_scatter_fit == pp._SFIT_POWER, "premise: the fit is the one on screen"
    _typed_look_and_words(win)
    win.set_axis_range(xmin=1e-4, xmax=2e-3)
    assert win._manual_x is not None, "premise: a range is pinned"


def _a_fitted_log_x_density_histogram(win: PlotsWindow) -> None:
    win.set_plot_type("Histogram")
    win.set_histogram_column("dwell_s")
    win.set_log_x(True)
    win.set_normalization("density")
    win.set_bins(50)
    win.set_fit("exponential")
    assert win._applied_fit == pp._FIT_EXP, "premise: the fit is the one on screen"
    _typed_look_and_words(win)
    win.set_axis_range(xmin=1e-4, xmax=2e-3)
    assert win._manual_x is not None, "premise: a range is pinned"


def _a_kde_with_a_log_y_axis(win: PlotsWindow) -> None:
    win.set_plot_type(pp.KDE)
    win.set_kde_columns("dwell_s", "depth_pA")
    win.set_kde_log(False, True)
    _typed_look_and_words(win)


def _an_event_count_heatmap(win: PlotsWindow) -> None:
    win.set_plot_type(pp.HEATMAP)
    win.set_heatmap_columns("depth_pA", "dwell_s")
    win.set_heatmap_log(False, True)
    win.set_heatmap_bins(17)
    _typed_look_and_words(win)


FIGURES = {"fitted log-log scatter": _a_fitted_log_log_scatter,
           "fitted log-x density histogram": _a_fitted_log_x_density_histogram,
           "KDE with a log y axis": _a_kde_with_a_log_y_axis,
           "event-count heatmap": _an_event_count_heatmap}


def _nudge_the_analysis(win: PlotsWindow) -> None:
    """A change of analysis the page being shown really offers: a log axis, a bin count."""
    page = win._plot_type()
    if page == pp.SCATTER:
        win.set_scatter_log(True, False)
    elif page == pp.KDE:
        win.set_kde_log(True, True)
    elif page == pp.HEATMAP:
        win.set_heatmap_bins(23)
    else:
        win.set_bins(33)


def _look_of(style: StyleSettings) -> dict:
    return {name: getattr(style, name) for name in THE_LOOK}


# ================================================================================================
# THE DEFECT: the look wipes the analysis
# ================================================================================================

@pytest.mark.parametrize("door", ["combo", "script"])
@pytest.mark.parametrize("figure", sorted(FIGURES))
def test_the_light_look_leaves_the_analysis_exactly_as_it_was(win, figure, door):
    FIGURES[figure](win)
    before = win.config()

    _recall(win, LOOK, door)
    after = win.config()

    # EVERYTHING but the style: plot type, columns, bins, normalisation, log axes, the fit
    # proposal and the fit on screen, the Time unit, the typed range and its pin, the ticked
    # sources, Accepted only, and the memory of the user's own Normalize pick.
    assert replace(after, style=before.style) == before
    assert after.plot_type == before.plot_type
    assert (after.fit.applied, after.fit.applied_scatter) == (
        before.fit.applied, before.fit.applied_scatter)
    if "fitted" in figure:
        assert (after.fit.applied, after.fit.applied_scatter) != (pp._FIT_NONE, pp._SFIT_NONE), (
            "premise: a fit is on screen, and is still")
    # ...and the look did change, to what the template is meant to set.
    assert _look_of(after.style) == THE_LOOK
    assert _look_of(before.style) != THE_LOOK, "premise: there was a look to change"


@pytest.mark.parametrize("figure", sorted(FIGURES))
def test_the_light_look_changes_only_the_look_it_names(win, figure):
    """A look is not a clear-out: the title typed on the figure, and the data colour and marker
    size the person chose, are theirs and stay. (The value-level test below holds the same for
    the element text styles, the fit label, the legend memory and the rest of the style.)"""
    FIGURES[figure](win)
    before = win.config()

    win.apply_template(win._BUILTIN_TEMPLATES[LOOK])

    style = win.style()
    assert replace(style, **_look_of(before.style)) == before.style, (
        "the style differs from the one before in the nine settings the look names, and in "
        "nothing else")
    assert style.title == "my typed title"
    assert style.data_color == "#cc0000" and style.point_size == 9
    assert win._title_edit.text() == "my typed title", "and the controls say so"
    assert win._bg_combo.currentText() == "Light" and not win._grid_check.isChecked()


@pytest.mark.parametrize("figure", sorted(FIGURES))
def test_recalling_the_light_look_is_one_undo_step_that_brings_everything_back(win, figure):
    FIGURES[figure](win)
    before = win.config()
    back, _ = win._history.depth()

    win._template_combo.setCurrentText(LOOK)
    assert win._history.depth()[0] == back + 1, "one step, not one per control written"

    after = win.config()
    assert win.undo_config()
    assert win.config() == before, "analysis and look both come back"
    assert win.redo_config()
    assert win.config() == after, "and Redo puts the look back, and only the look"


def test_the_light_look_on_a_window_with_no_data_changes_the_look_and_nothing_else(qapp, qtbot):
    empty = PlotsWindow(Session())
    qtbot.addWidget(empty)
    before = empty.config()

    empty.apply_template(empty._BUILTIN_TEMPLATES[LOOK])

    assert replace(empty.config(), style=before.style) == before
    assert _look_of(empty.style()) == THE_LOOK


@pytest.mark.parametrize("figure", sorted(FIGURES))
def test_the_name_stays_while_the_window_is_the_look_and_goes_when_it_is_not(win, figure):
    """The combo reads "these settings ARE that template". The look names no analysis, so an
    analysis change does not take its name off; a change to what it does name does."""
    FIGURES[figure](win)
    win._template_combo.setCurrentText(LOOK)
    assert win._template_combo.currentText() == LOOK

    _nudge_the_analysis(win)
    assert win._template_combo.currentText() == LOOK, (
        "a log axis or a bin count is not part of a look")

    win.set_line_width(5)
    assert win._template_combo.currentText() == PlotsWindow._TPL_MANUAL, (
        "a line width is: the figure no longer has the template's look")


def test_the_light_look_recalled_twice_does_nothing_the_second_time(win):
    _a_fitted_log_log_scatter(win)
    win.apply_template(win._BUILTIN_TEMPLATES[LOOK])
    once = win.config()
    steps = win._history.depth()

    win.apply_template(win._BUILTIN_TEMPLATES[LOOK])

    assert win.config() == once and win._history.depth() == steps


def test_the_ticked_sources_still_stay_where_they_were(win):
    win.set_data([_source("doc0:0", "apo -500 mV"), _source("doc1:0", "apo -600 mV")])
    win.set_sources(["apo -500 mV", "apo -600 mV"])
    ticked = win.selected_source_keys()
    assert len(ticked) == 2

    win.apply_template(win._BUILTIN_TEMPLATES[LOOK])

    assert win.selected_source_keys() == ticked


def _every_section_moved_off_its_default() -> PlotConfig:
    """A config with EVERY analysis section (all but the style) away from its defaults, so that
    a look which reached into any one of them would show."""
    return PlotConfig(
        plot_type=pp.KDE, norm_user_set=True,
        selection=pp.SelectionConfig(source_keys=("doc0:0", "doc1:1"), accepted_only=False),
        histogram=pp.HistogramConfig(column="frac_depth", bins=64, normalization=pp._NORM_PROB,
                                     colour_by=pp._COLOUR_CLUSTER, log_x=True, log_y=True,
                                     include_filter_limited=True, kde_overlay=True),
        scatter=pp.ScatterConfig(x="dwell_s", y="snr", log_x=True, log_y=True,
                                 colour_by=pp._COLOUR_CLUSTER),
        kde=pp.KdeConfig(x="depth_pA", y="snr", log_x=False, log_y=True, levels=13),
        heatmap=pp.HeatmapConfig(x="snr", y="dwell_s", log_x=False, log_y=True, bins=17),
        cluster=pp.ClusterConfig(features=("dwell_s", "snr"), method=pp._METHOD_KMEANS,
                                 k_mode=pp._K_FIXED, k=5),
        fit=pp.FitConfig(kind=pp._FIT_GAMMA, k=3, k_auto=True, scatter_kind=pp._SFIT_POWER,
                         applied=pp._FIT_EXP, applied_k=2, applied_k_auto=True,
                         applied_scatter=pp._SFIT_LINEAR, range_lo="0.1", range_hi="2.0",
                         range_unit="s"),
        axes=pp.AxesConfig(time_unit="s", x_min="1.5", x_max="2.5", y_min="3.5", y_max="4.5",
                           manual_x=(1e-4, 2e-3), manual_y=(10.0, 80.0)),
        style=StyleSettings(title="figure 3b", x_label="dwell", y_label="events",
                            data_color="#ff0000", point_size=9, show_legend=False,
                            legend_pos=(0.31, 0.42), font_family="Helvetica",
                            text_styles=FrozenMap({"axis.left.ticks": {"size_pt": 11.0}}),
                            label_overrides=FrozenMap({"doc0:0": "pore A"}),
                            fit_label=FrozenMap({"template": "Fit: {equation}"})))


def test_the_look_reaches_into_no_analysis_section_of_any_config():
    """At the value level, with no window to hide a difference: every section a config has
    besides its style, and every setting of the style the look does not name."""
    mine = _every_section_moved_off_its_default()
    for section in pp._CONFIG_SECTIONS:
        if section != "style":
            assert getattr(mine, section) != getattr(PlotConfig(), section), (
                f"premise: {section} is away from its default")
    assert mine.plot_type != PlotConfig().plot_type and mine.norm_user_set

    got = mine.with_template(PlotsWindow._BUILTIN_TEMPLATES[LOOK])

    assert replace(got, style=mine.style) == mine
    assert replace(got.style, **_look_of(mine.style)) == mine.style
    assert _look_of(got.style) == THE_LOOK


def test_a_look_names_only_what_it_moves_off_the_defaults():
    """A sparse look: one setting named, one setting changed."""
    mine = _every_section_moved_off_its_default()
    only_the_canvas = PlotConfig(plot_type="", style=StyleSettings(background="light"))

    got = mine.with_template(only_the_canvas)

    assert got.style == replace(mine.style, background="light")
    assert replace(got, style=mine.style) == mine


# ================================================================================================
# WHAT DOES NOT CHANGE: a template with an analysis is recalled as it always was
# ================================================================================================

def test_a_users_own_template_keeps_its_meaning_even_when_it_looks_like_the_shipped_one(win):
    """A saved template is the window's whole configuration, and always names a plot type. One
    saved from a figure whose analysis was all defaults and whose look is exactly the shipped
    look is still a template WITH settings: recalling it puts those settings on the window, as
    it did before the shipped look stopped carrying any."""
    _a_fitted_log_log_scatter(win)
    mine = PlotConfig(style=StyleSettings(**THE_LOOK))      # a plot type, the default one
    assert mine.plot_type, "premise: a saved template always has one"
    PlotsWindow._store_templates({"my look": plot_config_to_json(mine)})
    win._load_templates_into_combo()

    win._template_combo.setCurrentText("my look")

    after = win.config()
    assert after.plot_type == pp.HISTOGRAM and after.histogram.bins == 30
    assert (after.histogram.log_x, after.scatter.log_x, after.scatter.log_y) == (False,) * 3
    assert after.fit.scatter_kind == pp._SFIT_NONE and after.fit.applied_scatter == pp._SFIT_NONE
    assert _look_of(after.style) == THE_LOOK
    assert after.style.title == "", "a saved template installs the words it was saved with"


def test_a_template_saved_from_a_window_still_recalls_the_whole_configuration(win):
    """The round trip a person makes: set the figure up, save it, change it, pick it."""
    _a_fitted_log_x_density_histogram(win)
    saved = win.config().as_template()
    PlotsWindow._store_templates({"dwell setup": plot_config_to_json(saved)})
    win._load_templates_into_combo()
    win.set_plot_type("Scatter")
    win.set_bins(12)

    win._template_combo.setCurrentText("dwell setup")

    got = win.config()
    assert (got.plot_type, got.histogram.bins, got.histogram.log_x,
            got.histogram.normalization, got.fit.kind) == (
        pp.HISTOGRAM, 50, True, pp._NORM_DENSITY, pp._FIT_EXP)
    assert got.style.title == "my typed title"


def test_the_shipped_analysis_templates_still_install_their_analysis(win):
    _a_fitted_log_log_scatter(win)

    win._template_combo.setCurrentText(DWELL)
    got = win.config()
    assert got.plot_type == pp.HISTOGRAM
    assert (got.histogram.column, got.histogram.bins, got.histogram.log_x,
            got.histogram.normalization) == ("dwell_s", 50, True, pp._NORM_DENSITY)
    assert got.fit.kind == pp._FIT_EXP and got.fit.applied == pp._FIT_NONE, (
        "the fit arrives selected, not claimed to be drawn")

    win._template_combo.setCurrentText(SCATTER)
    got = win.config()
    assert got.plot_type == pp.SCATTER
    assert (got.scatter.x, got.scatter.y, got.scatter.log_x, got.scatter.log_y) == (
        "dwell_s", "depth_pA", True, False)


def test_every_shipped_template_says_what_it_is_meant_to_set():
    """Each shipped template either names an analysis (a plot type) or is a look (none). A new
    one has to be put on one side or the other here, on purpose."""
    meant = {name: ("analysis" if tpl.plot_type else "look")
             for name, tpl in PlotsWindow._BUILTIN_TEMPLATES.items()}
    assert meant == {LOOK: "look", DWELL: "analysis", SCATTER: "analysis"}

    look = PlotsWindow._BUILTIN_TEMPLATES[LOOK]
    assert _look_of(look.style) == THE_LOOK, "and the look is the one the template advertises"
    assert replace(look.style, **_look_of(StyleSettings())) == StyleSettings(), (
        "the look names nothing but those nine settings")


# ================================================================================================
# STORED THINGS: the look travels as a look, and nothing stored before has to change
# ================================================================================================

def test_the_look_is_still_a_look_after_a_trip_through_json(win):
    """The recipes route and the project copy read the shipped templates as JSON."""
    shipped = PlotsWindow._BUILTIN_TEMPLATES[LOOK]
    stored = plot_config_to_json(shipped)
    assert stored["plot_type"] == "", "the stored form names no plot type"
    assert set(stored) == {"format", "version", "app_version", "plot_type", "style"}, (
        "and no analysis settings at all: the style is all it carries")
    assert plot_config_from_json(stored) == shipped, "the shipped value is its own round trip"

    _a_fitted_log_log_scatter(win)
    before = win.config()
    win.apply_template(plot_config_from_json(stored))
    assert replace(win.config(), style=before.style) == before
    assert _look_of(win.style()) == THE_LOOK


def test_a_look_that_was_given_an_analysis_section_keeps_it_when_stored():
    """The writer leaves out what a look leaves at the defaults, not what it was given."""
    odd = PlotConfig(plot_type="", histogram=pp.HistogramConfig(bins=99),
                     style=StyleSettings(background="light"))

    stored = plot_config_to_json(odd)

    assert set(stored) == {"format", "version", "app_version", "plot_type", "histogram", "style"}
    assert plot_config_from_json(stored) == odd
    assert plot_config_to_json(plot_config_from_json(stored)) == stored


def test_a_template_written_by_the_old_build_recalls_as_it_did(win):
    """A template written before this change is a full configuration in JSON and has a plot
    type, so recalling it installs its analysis. The shipped look's own old JSON was one (plot
    type Histogram, 30 bins, no fit): a copy of it anywhere still means that."""
    old = plot_config_to_json(PlotConfig(style=StyleSettings(**THE_LOOK)))
    assert old["plot_type"] == pp.HISTOGRAM and old["histogram"]["bins"] == 30
    _a_fitted_log_x_density_histogram(win)

    win.apply_template(plot_config_from_json(old))

    got = win.config()
    assert (got.histogram.bins, got.histogram.log_x, got.fit.kind, got.fit.applied) == (
        30, False, pp._FIT_NONE, pp._FIT_NONE), "a template with an analysis installs it"


def test_the_recipe_route_reads_the_look_as_a_look(win):
    """New figure > From recipe: the built-in entry's payload is the template's JSON, and the
    recipe handed to the new figure carries it on without a plot type."""
    from nanoscope.gui.figure_recipes import entries, initial_recipe

    entry = next(e for e in entries() if e.kind == "distributions" and e.name == LOOK)
    assert entry.builtin and entry.payload["plot_type"] == ""
    recipe = initial_recipe(entry)
    assert recipe["plot_template"]["plot_type"] == ""

    _a_fitted_log_x_density_histogram(win)
    before = win.config()
    win.apply_template(plot_config_from_json(recipe["plot_template"]))
    assert replace(win.config(), style=before.style) == before


@pytest.mark.parametrize("written", ["now", "before this change"])
def test_a_new_figure_made_from_the_look_recipe_is_an_ordinary_new_figure_with_that_look(
        qtbot, written):
    """The route a person really has to the shipped templates: New figure > From recipe. The
    figure it makes has the analysis of any new figure, and the look the recipe carries, whether
    the recipe's JSON is the one written now or the full one this build used to write for it
    (a plot type, and every analysis section at its default)."""
    from nanoscope.gui.figure_recipes import entries, initial_recipe
    from nanoscope.gui.main_window import MainWindow

    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    entry = next(e for e in entries() if e.kind == "distributions" and e.name == LOOK)
    if written == "before this change":
        old = plot_config_to_json(replace(PlotsWindow._BUILTIN_TEMPLATES[LOOK],
                                          plot_type=pp.HISTOGRAM))
        assert old["plot_type"] == pp.HISTOGRAM and "histogram" in old
        entry = replace(entry, payload=old)

    first = owner.page(owner.create("distributions"))
    assert first is not None
    plain = first.config()
    page = owner.page(owner.create("distributions", initial_recipe=initial_recipe(entry)))
    assert page is not None

    assert replace(page.config(), style=plain.style) == plain, "the analysis of a new figure"
    # All nine settings the look names: a white canvas, no grid, the four text sizes, the curve's
    # width and the export size.
    assert _look_of(page.config().style) == THE_LOOK


@pytest.mark.parametrize("written", ["now", "before this change"])
def test_reset_figure_on_the_figure_made_from_the_look_recipe_gives_a_new_figures_look_and_binding(
        qtbot, written):
    """A figure made by New figure > From recipe has the recipe's look and no style (its Style
    button says "Style: custom"). Reset figure binds it to the project's default style and puts its
    look where a new figure's is, so what it did before (the old dark, gridded look, still bound to
    nothing) is gone. What is not part of that look stays: the page's own export size, and the
    Font size (body) that Distributions keeps through every Reset figure (the recipe's 11 pt, which
    rides on the binding as an override of the figure's own, so the Style button shows its dot)."""
    from nanoscope.gui.figure_recipes import entries, initial_recipe
    from nanoscope.gui.main_window import MainWindow

    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    explorer = window._ensure_analysis_explorer()
    entry = next(e for e in entries() if e.kind == "distributions" and e.name == LOOK)
    if written == "before this change":
        old = plot_config_to_json(replace(PlotsWindow._BUILTIN_TEMPLATES[LOOK],
                                          plot_type=pp.HISTOGRAM))
        entry = replace(entry, payload=old)
    fresh = owner.page(owner.create("distributions"))
    key = owner.create("distributions", initial_recipe=initial_recipe(entry))
    page = owner.page(key)
    assert not page.style().style_binding, "premise: the recipe's figure is bound to no style"
    assert _look_of(page.config().style) == THE_LOOK
    before = page.config()
    steps = page._history.depth()[0]

    explorer.toolbar().reset_figure_button.click()

    style = page.style()
    assert style.style_binding["ref"] == owner.default_style().ref.to_dict()
    assert dict(style.style_binding["overrides"]) == {"body_size_pt": 11.0}
    state = page.appearance_state()
    assert {**state, "style_binding": {**state["style_binding"], "overrides": {}}} == (
        fresh.appearance_state()), "a new figure's look and binding, the Font size (body) aside"
    assert (style.background, style.show_grid, style.font_family) == ("light", False, "Arial")
    assert style.font_size_pt == 11.0 != fresh.style().font_size_pt
    assert (style.title_font_size_pt, style.axis_label_font_size_pt, style.tick_font_size_pt,
            style.line_width) == (
        fresh.style().title_font_size_pt, fresh.style().axis_label_font_size_pt,
        fresh.style().tick_font_size_pt, fresh.style().line_width)
    assert (style.export_w, style.export_h) == (1600, 1200), "the export size is not the look's"
    assert replace(page.config(), style=before.style) == before, "and the analysis is untouched"
    assert explorer.toolbar().style_button.text() == "7 pt Arial · Nature, El… · r1 · ● ▾"
    assert page._history.depth() == (steps + 1, 0)
    assert page.undo_config() and page.config() == before and not page.style().style_binding
    assert page.redo_config() and page.style() == style


# ================================================================================================
# A PAGE BOUND TO A SHARED STYLE: the binding owns the look, so the look is an override of it
# ================================================================================================

def _shared_styles() -> dict:
    """The kinds of style a figure of a project can be bound to: a journal's, the slides one
    (every size far from the look's), and a sparse one that leaves the rest to the renderer."""
    from nanoscope.core.figure_style import FigureStyle, builtin_styles

    styles = builtin_styles()
    return {"journal 8 pt": styles[1], "slides 18 pt": styles[3],
            "sparse, inheriting the native defaults": FigureStyle(
                "legacy.sparse", 1, "Sparse", look={"grid": True},
                provenance={"inherit_native_defaults": True})}


@pytest.mark.parametrize("shared", sorted(_shared_styles()))
def test_the_light_look_lands_on_a_page_bound_to_a_shared_style(win, shared):
    """A figure of a project is bound to the project's figure style, and a bound look is
    resolved from the binding, so a setting written under it would be put straight back. The
    look is applied the way every look gesture is: as an override the binding then resolves."""
    from nanoscope.gui.figure_style_binding import bind_native_style

    _a_fitted_log_x_density_histogram(win)
    native = bind_native_style(pp._encode_dc(win.style()), _shared_styles()[shared],
                               "style_settings")
    win._set_config(replace(win.config(), style=pp._project_style_settings(
        pp._decode_dc(StyleSettings, native))))
    assert win.style().style_binding, "premise: the page is bound"
    reference = win.style().style_binding["ref"]
    before = win.config()
    assert _look_of(before.style) != THE_LOOK, "premise: there is a look to change"

    win.apply_template(win._BUILTIN_TEMPLATES[LOOK])

    after = win.config()
    assert replace(after, style=before.style) == before, "the analysis is untouched"
    assert after.style.style_binding["ref"] == reference, "and it is still bound to that style"
    assert _look_of(after.style) == THE_LOOK, "while carrying the look the template names"
    assert after.style.title == "my typed title"


def test_the_light_look_on_a_figure_page_of_a_project_keeps_its_style_and_its_analysis(qtbot):
    """The same on the real thing: a Distributions figure of the workspace, which a new project
    binds to its default figure style. The look lands as the template's own, the figure is still
    bound to that style, the analysis is the figure's, and one Undo takes the whole step back."""
    from nanoscope.gui.main_window import MainWindow

    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    page = owner.page(owner.create("distributions"))
    assert page is not None
    page.set_plot_type("Scatter")
    page.set_scatter_log(True, True)
    assert page.style().style_binding, "premise: a new figure is bound to the project's style"
    before = page.config()
    steps = page._history.depth()[0]

    page.apply_template(PlotsWindow._BUILTIN_TEMPLATES[LOOK])

    after = page.config()
    assert replace(after, style=before.style) == before
    assert after.style.style_binding["ref"] == before.style.style_binding["ref"]
    assert _look_of(after.style) == THE_LOOK
    assert page._history.depth()[0] == steps + 1
    assert page.undo_config() and page.config() == before


def test_a_project_saved_after_the_light_look_reopens_with_the_same_figure(qtbot, tmp_path):
    """What the recall leaves behind is an ordinary saved state: the figure's plot, its log axes
    and its look (a bound style with the look's settings as overrides) survive a save and an
    open, in a fresh main window."""
    from nanoscope.gui.main_window import MainWindow

    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    key = owner.create("distributions")
    page = owner.page(key)
    page.set_plot_type("Scatter")
    page.set_scatter_log(True, True)
    page.apply_template(PlotsWindow._BUILTIN_TEMPLATES[LOOK])
    wanted = page.config()
    assert owner.checkpoint(key, changed=True)
    path = tmp_path / "after_the_look.nanoscope"
    assert window._write_project(str(path))
    window._dirty = False

    again = MainWindow(recovery_enabled=False)
    qtbot.addWidget(again)
    with qtbot.waitSignal(again.projectRestored, timeout=20000):
        again._open_project(str(path))
    reopened = again._ensure_figure_workspace()
    (figure,) = [k for k, e in reopened._entries.items() if e.record.spec.kind == "distributions"]
    page = reopened.page(figure)
    if page is None:
        reopened.activate(figure)
        page = reopened.page(figure)
    got = page.config()

    assert (got.plot_type, got.scatter.log_x, got.scatter.log_y) == (pp.SCATTER, True, True)
    assert _look_of(got.style) == THE_LOOK and got.style.style_binding
    assert got.style == wanted.style
    again._dirty = False
