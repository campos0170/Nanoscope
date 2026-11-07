"""Phase 0 final review R2: a Y range pinned on a DENSITY histogram goes with a new column.

A histogram's y has no column of its own (it is a count), so a Y range pinned in Axis properties
rightly survives a new column while the Normalize is *count* or *probability per bin*. A *density*
is per the unit of the column it is a density of (``density (1/ms)`` over raw ``dwell_s``, plain
``density`` over ``frac_depth``), and it is the default as soon as two sources are ticked: the
range pinned for one column then frames nothing over the next (measured: 90.4 to 378 over bars
topping out at 6.88, a blank plot with the words reset and no sentence about the range).

Every gesture is a real one (Axis properties by right-click on the axis label, the column by the
combo, Plot by its button). What is asserted is what is drawn and what Axis properties would say.
Synthetic data only; offscreen.
"""
from __future__ import annotations

from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
import pytest
import numpy as np

from nanoscope.gui.plots_panel import (AxesConfig, HistogramConfig, PlotConfig, ScatterConfig,
    FrozenMap, migrate_plot_quantity, plot_config_to_json, plot_config_from_json)

from .test_p0_pins_clear_with_the_words import _axis, _pin_inside
from .test_p0_reset_on_quantity_change import (  # noqa: F401  (the fixture is used by name)
    _clear_status, _click_button, _said, _user_picks, dist)
from .test_plot_window_routes import _add_title

SAID = ("Y range reset: the plot is now set to show the histogram of frac_depth, "
        "fitted with exponential.")


def _hist(norm, column="dwell_s", unit="ms", **axes):
    return PlotConfig(histogram=HistogramConfig(column=column, normalization=norm),
                      axes=AxesConfig(time_unit=unit, **axes))


def _y_pinned(norm, **more):
    return _hist(norm, y_min="90.4", y_max="378", manual_y=(90.4, 378.0), **more)


def _drawn_top(plot):
    """The tallest thing drawn, in the data units of the y axis."""
    return plot.getViewBox().childrenBounds()[1][1]


def _frames_the_data(plot):
    lo, hi = _axis(plot, "left").shown_range
    axis = plot.getAxis("left")
    raw_top = _drawn_top(plot)
    displayed_top = (10. ** raw_top * axis.display_factor if axis.logMode else raw_top * axis.scale)
    return lo <= 0 <= hi and displayed_top <= hi


# -- the rule as a value ----------------------------------------------------------------------------

def test_the_rule_a_new_column_moves_the_y_axis_of_a_density_histogram():
    before = _y_pinned("density", x_min="0.1", x_max="2", manual_x=(1e-4, 2e-3))
    after = replace(before, histogram=replace(before.histogram,
                                              column="frac_depth")).rebased(before.identity)
    assert (after.axes.x_min, after.axes.x_max, after.axes.manual_x) == ("", "", None)
    assert (after.axes.y_min, after.axes.y_max, after.axes.manual_y) == ("", "", None)


@pytest.mark.parametrize("norm", ["count", "probability per bin"])
def test_the_rule_a_new_column_leaves_the_y_axis_of_a_count_or_a_probability_alone(norm):
    before = _y_pinned(norm)
    after = replace(before, histogram=replace(before.histogram,
                                              column="frac_depth")).rebased(before.identity)
    assert (after.axes.y_min, after.axes.y_max, after.axes.manual_y) == ("90.4", "378",
                                                                        (90.4, 378.0))


def test_the_rule_a_scatter_keeps_its_y_range_when_only_its_x_column_moves():
    """The density belongs to the HISTOGRAM: a scatter whose config remembers a density
    normalization (the combo was left there) has a y axis of its own column, which did not move."""
    before = PlotConfig(plot_type="Scatter", histogram=HistogramConfig(normalization="density"),
                        scatter=ScatterConfig(x="dwell_s", y="depth_pA"),
                        axes=AxesConfig(y_min="1", y_max="2", manual_y=(1.0, 2.0)))
    after = replace(before, scatter=replace(before.scatter, x="frac_depth")).rebased(before.identity)
    assert (after.axes.y_min, after.axes.y_max, after.axes.manual_y) == ("1", "2", (1.0, 2.0))
    assert after.axes.manual_x is None


@pytest.mark.parametrize("unit,factor", [("s", 1.), ("ms", 1000.), ("µs", 1e6)])
def test_legacy_unit_migration_carries_display_ranges_and_keeps_raw_density_pin(unit, factor):
    before = _y_pinned("density", unit=unit, x_min=str(.0001 * factor),
                       x_max=str(.002 * factor), manual_x=(.0001, .002))
    after = migrate_plot_quantity(before)
    assert (float(after.axes.x_min), float(after.axes.x_max)) == pytest.approx((.1, 2.))
    assert (float(after.axes.y_min), float(after.axes.y_max)) == pytest.approx((.0904, .378))
    assert after.axes.manual_x == before.axes.manual_x
    assert after.axes.manual_y == before.axes.manual_y
    assert migrate_plot_quantity(after) == after


def test_the_rule_the_same_column_is_no_move():
    before = _y_pinned("density")
    assert before.rebased(before.identity) is before


def test_the_rule_another_plot_type_still_takes_both_axes():
    before = _y_pinned("density", x_min="0.1", manual_x=(1e-4, 2e-3))
    after = replace(before, plot_type="Scatter").rebased(before.identity)
    assert after.axes.manual_x is None and after.axes.manual_y is None


# -- the real routes --------------------------------------------------------------------------------

def test_a_new_column_takes_the_y_range_of_a_density_histogram_in_one_undo_step(dist):
    page, explorer = dist
    plot = page._plot
    page.set_sources(["probe", "other"])
    _click_button(page._plot_btn)
    assert page.config().histogram.normalization == "density"
    pin = _pin_inside(plot, "left")
    page._history_touch()
    np.testing.assert_allclose(page.config().axes.manual_y, np.asarray(pin) / plot.getAxis("left").scale)
    assert _axis(plot, "left").range == pytest.approx(pin)
    _clear_status(page)
    depth = page._history.depth()[0]

    _user_picks(page._hist_col, "frac_depth")
    _click_button(page._plot_btn)

    assert page.config().axes.manual_y is None
    assert (page.config().axes.y_min, page.config().axes.y_max) == ("", "")
    assert _axis(plot, "left").range is None                       # Axis properties: Auto
    assert _frames_the_data(plot)                                  # and the plot is not blank
    assert _said(page, explorer) == (SAID, True)
    assert page._history.depth()[0] == depth + 1                   # one step for the whole move
    assert page.undo_config()                                      # Distributions' own Undo
    assert page._hist_col.currentText() == "dwell_s"
    np.testing.assert_allclose(page.config().axes.manual_y, np.asarray(pin) / plot.getAxis("left").scale)
    assert _axis(plot, "left").range == pytest.approx(pin)  # same displayed physical range


def test_a_new_column_names_the_range_beside_the_words_it_takes(dist):
    page, explorer = dist
    plot = page._plot
    page.set_sources(["probe", "other"])
    _click_button(page._plot_btn)
    _add_title(plot, "Dwell, pore A")
    _pin_inside(plot, "left")
    page._history_touch()
    _user_picks(page._hist_col, "frac_depth")
    assert _said(page, explorer) == (
        "Title and Y range reset: the plot is now set to show the histogram of frac_depth, "
        "fitted with exponential.", True)
    assert page.config().axes.manual_y is None


def test_a_new_column_leaves_the_y_range_of_a_count_histogram(dist):
    page, explorer = dist
    plot = page._plot
    _click_button(page._plot_btn)
    assert page.config().histogram.normalization == "count"
    pin = _pin_inside(plot, "left")
    page._history_touch()
    _clear_status(page)

    _user_picks(page._hist_col, "frac_depth")
    _click_button(page._plot_btn)

    np.testing.assert_allclose(page.config().axes.manual_y, np.asarray(pin) / plot.getAxis("left").scale)
    assert _axis(plot, "left").range == pytest.approx(pin)
    assert _said(page, explorer)[0] == ""


def test_a_plot_again_on_the_same_column_keeps_the_density_y_range(dist):
    page, explorer = dist
    plot = page._plot
    page.set_sources(["probe", "other"])
    _click_button(page._plot_btn)
    pin = _pin_inside(plot, "left")
    page._history_touch()
    _clear_status(page)

    _click_button(page._plot_btn)

    np.testing.assert_allclose(page.config().axes.manual_y, np.asarray(pin) / plot.getAxis("left").scale)
    assert _axis(plot, "left").range == pytest.approx(pin)
    assert _said(page, explorer)[0] == ""


def test_legacy_microsecond_recipe_keeps_the_physical_density_range(dist):
    page, explorer = dist
    plot = page._plot
    page.set_sources(["probe", "other"])
    _click_button(page._plot_btn)
    pin = _pin_inside(plot, "left")
    page._history_touch()
    before = page.config()
    raw_y = before.axes.manual_y
    # Version 3 used µs x text but per-second y text; raw manual pins were canonical.
    old = replace(before, axes=replace(before.axes, time_unit='µs',
        quantity_display=FrozenMap({}), x_min='', x_max='',
        y_min=str(raw_y[0]), y_max=str(raw_y[1])))
    payload = plot_config_to_json(old)
    payload['version'] = 3
    payload['axes'].pop('quantity_display', None)
    page.apply_config(plot_config_from_json(payload))
    assert page.config().axes.time_unit == 'ms'
    np.testing.assert_array_equal(page.config().axes.manual_y, raw_y)
    assert (float(page.config().axes.y_min), float(page.config().axes.y_max)) == pytest.approx(pin)
    assert _axis(plot, 'left').range == pytest.approx(pin)


def test_a_reopened_density_project_keeps_the_y_range_until_a_new_column_takes_it(
        qapp, qtbot, tmp_path, monkeypatch):
    """A project saved with two sources (Normalize: density), a column and a pinned Y range opens
    as saved (the rule does not run on an open), and the first new column after opening takes the
    range in one step that Undo gives back."""
    from nanoscope.gui import figure_text
    from nanoscope.gui.main_window import MainWindow

    from .test_p0_pins_clear_with_the_words import _is_automatic, _stays_pinned
    from .test_plot_window_routes import _press_undo
    from .test_plots_reopen_figure import _main, _recordings, _settle

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    figure_text.GATE.reset()
    window, plots = _recordings(qtbot, tmp_path, names=("a", "b"))
    window._analysis_explorer.resize(1280, 880)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    _settle()
    plots.activateWindow()
    plots._plot_btn.click()
    _settle()
    assert plots.config().histogram.normalization == "density"
    y_pin = _pin_inside(plots._plot, "left")
    plots._history_touch()
    _settle()
    saved = tmp_path / "density.nanoscope"
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
    _settle()
    assert page.config().histogram.normalization == "density"
    _stays_pinned(page._plot, "left", y_pin)                       # opens as it was saved
    assert page._history.depth() == (0, 0)

    _user_picks(page._hist_col, "frac_depth")
    assert page.config().axes.manual_y is None
    assert page._history.depth() == (1, 0), "one step"
    assert "Y range" in page.statusBar().currentMessage()
    page._plot_btn.click()
    _settle()
    _is_automatic(page._plot, "left", was=y_pin, auto_ranging=False)
    assert _frames_the_data(page._plot)

    _press_undo(page)
    _settle()
    assert page._hist_col.currentText() == "dwell_s"
    _stays_pinned(page._plot, "left", y_pin)
    window._dirty = again._dirty = False
    figure_text.GATE.reset()
