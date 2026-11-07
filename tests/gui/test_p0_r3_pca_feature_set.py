"""Phase 0 final review R3: the PCA's feature set is the PCA's quantity.

PCA has no axis column, so ``AnalysisIdentity`` used to be the same for any feature set: ticking
other features and pressing Run PCA again drew other scores (PC1 and PC2 are other combinations of
other columns) under the title, the axis labels and the pinned ranges typed for the old ones, with
no step and no sentence. The feature set is now part of the identity on the PCA page, so the
words and both pins go with it like they go with a column: one Configuration Undo step (the tick's
own), one sentence naming what went. On every other plot type the set is a clustering setting and
moves nothing. Real gestures: the combo's popup, a click on the checklist row, the Run PCA button,
**Add title…**, a double-click on the label and **Axis properties…** for the pins. Synthetic data
only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.gui import figure_text

from .test_column_checklist import click_text
from .test_p0_pins_clear_with_the_words import _is_automatic, _pin_inside, _stays_pinned
from .test_p0_reset_on_quantity_change import (
    _clear_status, _click_button, _dist_frame, _dist_words, _drawn_title, _label,
    _pick, _reveal_control, _said, _user_picks, dist)
from .test_plot_window_routes import _add_title, _edit_double_click

PLOT_TYPE = "PCA"


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _analysis_section(page):
    """Reveal the real Groups inspector section containing the Features checklist."""
    _reveal_control(page._features.view)
    QtWidgets.QApplication.processEvents()
    assert page._features.view.isVisible()


def _features(page, *keys):
    """Leave exactly ``keys`` ticked, one click on a row at a time (a person's gestures)."""
    _analysis_section(page)
    for key in list(page.feature_selection()):
        if key not in keys:
            click_text(page._features, key)
    for key in keys:
        if key not in page.feature_selection():
            click_text(page._features, key)
    assert set(page.feature_selection()) == set(keys)


def _run(page):
    _click_button(page._pca_btn)
    assert page._pca_result is not None


@pytest.fixture
def pca(dist):
    """Distributions on PCA of dwell_s and depth_pA, run, with a title, both axis labels and both
    axes pinned: everything the rule has a say about."""
    page, explorer = dist
    plot = page._plot
    _pick(page._plot_type_combo, text=PLOT_TYPE)
    _features(page, "dwell_s", "depth_pA")
    _run(page)
    _add_title(plot, "PCA of dwell and depth")
    _edit_double_click(plot, plot.getAxis("bottom").label, "PC1 (dwell, depth)")
    _edit_double_click(plot, plot.getAxis("left").label, "PC2 (dwell, depth)")
    pins = (_pin_inside(plot, "bottom"), _pin_inside(plot, "left"))
    page._history_touch()
    _clear_status(page)
    assert _dist_words(page)[:3] == ("PCA of dwell and depth", "PC1 (dwell, depth)",
                                     "PC2 (dwell, depth)")
    return page, explorer, pins


def test_a_new_feature_takes_the_words_and_both_pins_in_one_undo_step(pca):
    """The reviewer's case: the typed words and the pinned score ranges stayed over other scores."""
    page, explorer, (x_pin, y_pin) = pca
    plot = page._plot
    steps = page._history.depth()[0]

    click_text(page._features, "frac_depth")                       # one click on a row

    assert page._history.depth()[0] == steps + 1                   # the tick's own step, no more
    assert _dist_words(page) == ("", "", "", "")
    assert page.config().axes.manual_x is None and page.config().axes.manual_y is None
    assert _said(page, explorer) == (
        "Title, X label, Y label, X range and Y range reset: the plot is now set to show the "
        "PCA scores.", True)
    _run(page)                                                     # the picture follows Run PCA
    assert _drawn_title(plot)[0] == ""
    assert (_label(plot, "bottom"), _label(plot, "left")) == (
        f"PC1 ({100 * page._pca_result.explained_variance_ratio[0]:.1f}%)",
        f"PC2 ({100 * page._pca_result.explained_variance_ratio[1]:.1f}%)")
    _is_automatic(plot, "bottom", was=x_pin, auto_ranging=False)
    _is_automatic(plot, "left", was=y_pin, auto_ranging=False)
    assert _said(page, explorer)[0].startswith("Title, X label, Y label, X range and Y range")

    # Distributions' Configuration Undo takes the feature and the words and pins back together.
    assert page.undo_config()
    assert set(page.feature_selection()) == {"dwell_s", "depth_pA"}
    assert _dist_words(page)[:3] == ("PCA of dwell and depth", "PC1 (dwell, depth)",
                                     "PC2 (dwell, depth)")
    assert page.config().axes.manual_x == pytest.approx(x_pin)
    assert page.config().axes.manual_y == pytest.approx(y_pin)
    assert page.redo_config()
    assert _dist_words(page) == ("", "", "", "") and page.config().axes.ranges() == ()


def test_the_public_setter_is_the_same_route_as_the_click(pca):
    page, explorer, _pins = pca
    page.set_feature_selection(["dwell_s", "frac_depth", "xcol"])      # the reviewer's call
    assert _dist_words(page) == ("", "", "", "") and page.config().axes.ranges() == ()
    assert _said(page, explorer)[0].startswith("Title, X label, Y label, X range and Y range reset")
    _run(page)
    assert page.config().style.title == ""


def test_other_features_with_nothing_typed_say_nothing_and_take_the_earlier_sentence_down(pca):
    page, explorer, _pins = pca
    click_text(page._features, "frac_depth")                       # resets and says so
    assert "reset" in _said(page, explorer)[0]
    steps = page._history.depth()[0]

    click_text(page._features, "xcol")                             # nothing left to reset

    assert page._history.depth()[0] == steps + 1                   # the tick only
    assert _said(page, explorer)[0] == ""                          # the old sentence is gone
    _run(page)
    assert _dist_words(page) == ("", "", "", "")


def test_words_typed_after_the_change_survive_run_pca(pca):
    """The reset happens when the set changes, not at Run PCA: what is typed after it is kept."""
    page, _explorer, _pins = pca
    click_text(page._features, "frac_depth")
    _run(page)
    plot = page._plot
    _add_title(plot, "Three features")
    _run(page)                                                     # the same set again
    assert page.config().style.title == "Three features"
    assert _drawn_title(plot)[0] == "Three features"


def test_a_feature_set_on_another_plot_type_is_a_clustering_setting_and_moves_nothing(dist):
    page, explorer = dist
    plot = page._plot
    _add_title(plot, "Dwell, pore A")
    pin = _pin_inside(plot, "bottom")
    page._history_touch()
    _clear_status(page)
    _analysis_section(page)
    for key in ("dwell_s", "depth_pA", "frac_depth", "xcol"):
        click_text(page._features, key)
    assert page.config().style.title == "Dwell, pore A"
    _stays_pinned(plot, "bottom", pin)
    assert _said(page, explorer)[0] == ""
    # ...and the next move that is measured from the anchor (a time unit, which rescales the
    # pin and keeps the words) finds the set where the ticks left it: it is no quantity here.
    _user_picks(page._time_unit, "µs")
    assert page.config().style.title == "Dwell, pore A" and page.config().axes.manual_x is not None
    assert _said(page, explorer)[0] == ""
    # ...and running PCA from the histogram keeps the caption as it always did: it is the
    # caption of the picture about to appear (`_rebase`); the pin of the axis that moved goes.
    page.run_pca()
    assert page.config().style.title == "Dwell, pore A" and page.config().axes.manual_x is None
    assert _said(page, explorer)[0] == "X range reset: the plot is now set to show the PCA scores."


def test_a_refresh_of_the_same_table_keeps_the_words_and_one_that_loses_a_feature_resets_them(pca):
    page, explorer, (x_pin, _y_pin) = pca
    page.set_data({"probe": _dist_frame(), "other": _dist_frame(1)})      # Refresh: same columns
    assert page.config().style.title == "PCA of dwell and depth"
    assert _said(page, explorer)[0] == ""
    _run(page)                                                            # a Refresh drops the result
    _stays_pinned(page._plot, "bottom", x_pin)

    page.set_data({"probe": _dist_frame().drop(columns=["depth_pA"]),
                   "other": _dist_frame(1).drop(columns=["depth_pA"])})  # a feature is gone
    assert page.feature_selection() == ["dwell_s"]
    assert _dist_words(page) == ("", "", "", "")                          # a move nobody typed
    assert page.config().axes.ranges() == ()


def test_a_config_naming_features_the_table_lacks_does_not_move_the_anchor(pca):
    """The stale-anchor hazard (`_installable`): ``set_checked`` ignores a feature the list does
    not offer, so the anchor must take the set from the list, not from the config. Otherwise the
    next rebase (here a time-unit change, which moves no quantity) would read that as a move."""
    page, _explorer, _pins = pca
    cfg = page.config()
    page.apply_config(replace(cfg, cluster=replace(
        cfg.cluster, features=("depth_pA", "dwell_s", "no_such_column"))))
    assert page.feature_selection() == ["dwell_s", "depth_pA"]
    assert page._identity.features == ("depth_pA", "dwell_s")
    _user_picks(page._time_unit, "µs")
    assert page.config().style.title == "PCA of dwell and depth"
    assert page.config().axes.manual_x is not None


def test_a_template_lists_the_features_in_its_own_order_without_that_being_a_move(pca):
    page, _explorer, _pins = pca
    worded = page.config()
    reordered = replace(worded, cluster=replace(
        worded.cluster, features=tuple(reversed(worded.cluster.features))))
    assert reordered.identity == worded.identity
    page.apply_template(reordered)
    _user_picks(page._time_unit, "µs")
    assert page.config().style.title == worded.style.title


def test_a_recalled_template_installs_its_own_features_and_words_and_the_rule_does_not_run(pca):
    page, explorer, _pins = pca
    worded = page.config()
    click_text(page._features, "frac_depth")                       # resets
    _clear_status(page)
    page.apply_template(worded)                                    # back to dwell_s, depth_pA
    assert set(page.feature_selection()) == {"dwell_s", "depth_pA"}
    assert page.config().style.title == "PCA of dwell and depth"
    assert _said(page, explorer)[0] == ""


def test_words_saved_over_a_feature_set_open_unchanged_and_the_next_tick_resets(pca, qtbot):
    """What a project open ends in (``restore_plots``, which installs the saved config with the
    window's own writer and the saved PCA result): the saved words, pins and set come back as they
    were, with no sentence, and the NEXT change of the set resets them."""
    from nanoscope.core.analysis_workspace import capture_plots, restore_plots
    from nanoscope.core.session import Session
    from nanoscope.gui.analysis_explorer import PAGE_DISTRIBUTIONS
    from nanoscope.gui.plots_panel import PlotsWindow

    from .test_p0_reset_on_quantity_change import _in_explorer

    page, _explorer, (x_pin, y_pin) = pca
    saved = capture_plots(page)

    fresh = PlotsWindow(Session())
    qtbot.addWidget(fresh)
    explorer = _in_explorer(qtbot, fresh, PAGE_DISTRIBUTIONS)
    fresh.set_data({"probe": _dist_frame(), "other": _dist_frame(1)})
    fresh.activateWindow()
    report = restore_plots(fresh, saved)
    QtWidgets.QApplication.processEvents()
    assert report.skipped == (), report.warnings
    assert fresh._plot_type_combo.currentText() == PLOT_TYPE and fresh._pca_result is not None
    assert set(fresh.feature_selection()) == {"dwell_s", "depth_pA"}
    assert _dist_words(fresh)[:3] == ("PCA of dwell and depth", "PC1 (dwell, depth)",
                                      "PC2 (dwell, depth)")
    assert fresh.config().axes.manual_x == pytest.approx(x_pin)
    assert fresh.config().axes.manual_y == pytest.approx(y_pin)
    assert _said(fresh, explorer)[0] == ""
    assert fresh._identity.features == ("depth_pA", "dwell_s")

    _analysis_section(fresh)
    click_text(fresh._features, "frac_depth")                      # the next change resets
    assert _dist_words(fresh) == ("", "", "", "") and fresh.config().axes.ranges() == ()
    assert _said(fresh, explorer)[0].startswith(
        "Title, X label, Y label, X range and Y range reset: the plot is now set to show")
