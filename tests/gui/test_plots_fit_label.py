"""Distributions: a fit's numbers go to the fit label, never to the title (plotux G3).

The owner's report: applying a Linear fit in Distributions ▸ Scatter made the equation the
plot's title, so a typed title took the result off the figure (and every export) and
clearing it put the equation back as the title. A fit now leaves the title, the Title box
and ``style.title`` exactly as they were; its numbers are drawn in the fit label, one line
per source in the source's colour, with the same honesty rules the title had: the R²
refusal is printed whatever the switches say.

Synthetic DataFrames only; offscreen.
"""
from __future__ import annotations

import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.plots_panel import (
    PlotsWindow, StyleSettings, plot_config_from_json, plot_config_to_json,
)
from nanoscope.plot.pg_svg import export_svg


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _df(seed: int = 0, n: int = 300, slope: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({
        "xcol": x, "ycol": slope * x + 5.0 + rng.normal(0.0, 0.3, n),
        "dwell_s": rng.exponential(0.002, n) + 1e-4,
        "depth_pA": rng.normal(50.0, 5.0, n),
        "accepted": np.ones(n, dtype=bool), "filter_limited": np.zeros(n, dtype=bool)})


def _scatter(qtbot, frames=None, fit="linear") -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data(frames or {"probe": _df()})
    w.set_plot_type("Scatter")
    w.set_scatter_columns("xcol", "ycol")
    w.set_scatter_fit(fit)
    return w


def _title_text(w) -> str:
    return w._plot.titleLabel.item.toPlainText() if w._plot.titleLabel.isVisible() else ""


def test_a_scatter_fit_leaves_the_title_alone_and_puts_its_numbers_in_the_fit_label(qapp, qtbot):
    w = _scatter(qtbot)
    assert _title_text(w) == "", "a fit must never become the title"
    assert w.config().style.title == ""
    label = w._fit_label_item
    assert label.isVisible()
    text = label.plain_text()
    assert re.search(r"y = [\d.]+·x \+ [\d.]+", text), text
    assert "R² = 0.9" in text and "n = 300" in text, text


def test_a_histogram_fit_puts_its_parameters_in_the_label_not_the_title(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"probe": _df()})
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    assert _title_text(w) == ""
    text = w._fit_label_item.plain_text()
    assert "τ" in text and "ms" in text and "n = 300" in text, text
    # The legend's fit row names the fit and carries no numbers.
    rows = [label.text for _sample, label in w._plot.legend.items]
    fit_rows = [r for r in rows if "fit" in r]
    assert fit_rows and not any(re.search(r"\d", r) for r in fit_rows), rows


def test_a_typed_title_and_the_fit_are_both_on_the_figure_and_in_the_svg(qapp, qtbot, tmp_path):
    w = _scatter(qtbot)
    w.set_title("Depth vs dwell")
    assert _title_text(w) == "Depth vs dwell"
    assert "R² = 0.9" in w._fit_label_item.plain_text()
    svg = export_svg(w._plot, tmp_path / "fig.svg").read_text()
    assert "Depth vs dwell" in svg
    assert "R² = 0.9" in svg and "n = 300" in svg


def test_two_sources_get_two_coloured_label_lines_and_number_free_legend_rows(qapp, qtbot):
    w = _scatter(qtbot, {"apo": _df(1), "holo": _df(2, slope=3.0)})
    w.set_sources(["apo", "holo"]) if hasattr(w, "set_sources") else None
    lines = w._fit_label_item.lines()
    assert len(lines) == 2, lines
    colours = {colour for _text, colour in lines}
    assert len(colours) == 2, lines
    rows = [label.text for _sample, label in w._plot.legend.items if "fit" in label.text]
    assert len(rows) == 2 and all("(linear)" in r for r in rows), rows
    assert not any(re.search(r"\d", r.split("(")[0]) or "=" in r for r in rows), rows
    # Each line is prefixed with the figure's own name for its source.
    plain = w._fit_label_item.plain_text()
    assert "apo:" in plain and "holo:" in plain, plain


def test_unticking_r2_drops_the_number_but_never_the_refusal(qapp, qtbot):
    flat = _df()
    flat["ycol"] = 7.0                    # y does not vary: R² is undefined
    w = _scatter(qtbot, {"probe": flat})
    assert "R² undefined" in w._fit_label_item.plain_text()
    w._r2_check.setChecked(False)
    assert "R² undefined" in w._fit_label_item.plain_text()
    w2 = _scatter(qtbot)
    w2._r2_check.setChecked(False)
    assert "R²" not in w2._fit_label_item.plain_text()
    assert "y = " in w2._fit_label_item.plain_text()


def test_a_template_keeps_the_users_words_and_takes_new_numbers(qapp, qtbot):
    w = _scatter(qtbot)
    host = figure_text.figure_text_for(w._plot).host
    host.apply_text("annotation.fit", figure_text.TextStyle(text="Slope: {equation} ({n})"))
    first = w._fit_label_item.plain_text()
    assert first.startswith("Slope: y = ") and "(n = 300)" in first, first
    w.set_data({"probe": _df(5, n=200, slope=4.0)})
    w.set_scatter_columns("xcol", "ycol")
    w.set_scatter_fit("linear")
    w.plot() if hasattr(w, "plot") and callable(w.plot) else None
    again = w._fit_label_item.plain_text()
    assert again.startswith("Slope: y = ") and "(n = 200)" in again, again


def test_dragging_the_label_is_one_undo_step_and_undo_puts_it_back(qapp, qtbot):
    w = _scatter(qtbot)
    vb = w._plot.getViewBox()
    before_range = [v for r in vb.viewRange() for v in r]
    back = w._history.depth()[0]
    w._fit_label_item.moved.emit((150.0, 100.0))
    state = w.config().style.fit_label_state()
    assert state.placement == "custom" and state.offset == (150.0, 100.0)
    assert w._history.depth()[0] == back + 1
    assert [v for r in vb.viewRange() for v in r] == pytest.approx(before_range)
    assert w._fit_label_item.offset() == pytest.approx((150.0, 100.0))
    assert w.undo_config()
    assert w.config().style.fit_label_state().placement == "top-left"
    # ...and the saved position survives the configuration file.
    w.redo_config() if hasattr(w, "redo_config") else None
    payload = plot_config_to_json(w.config())
    assert plot_config_from_json(payload).style.fit_label_state().offset == (150.0, 100.0)


def test_an_older_configuration_opens_with_the_label_shown_and_its_old_switches_mapped():
    payload = plot_config_to_json(_blank_config())
    payload["style"].pop("fit_label", None)
    payload["style"].pop("text_styles", None)
    payload["style"]["fit_values_on_plot"] = True
    state = plot_config_from_json(payload).style.fit_label_state()
    assert state.visible and state.fields == ("equation", "r2", "n", "bic", "ks")
    payload["style"]["fit_values_on_plot"] = False
    state = plot_config_from_json(payload).style.fit_label_state()
    assert state.visible and state.fields == ("equation", "r2", "n")
    # The shipped v1 fixture (written before the label existed) loads it visible too.
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "plot_config_v1_pre_rename.json")
    import json

    with open(fixture, encoding="utf-8") as handle:
        old = plot_config_from_json(json.load(handle))
    assert old.style.fit_label_state().visible


def _blank_config():
    from nanoscope.gui.plots_panel import PlotConfig

    return PlotConfig()


def test_a_style_holding_element_styles_and_a_label_stays_hashable_and_round_trips():
    style = StyleSettings().with_text_style(
        "axis.left.label", figure_text.TextStyle(text="Current", bold=True, color="#d55e00"))
    style = style.with_fit_label(style.fit_label_state().__class__(offset=(3.0, 4.0),
                                                                  placement="custom"))
    hash(style)
    from nanoscope.gui.plots_panel import PlotConfig

    cfg = PlotConfig(style=style)
    again = plot_config_from_json(plot_config_to_json(cfg))
    assert again == cfg
    assert again.style.y_label == "Current"
    assert again.style.text_style("axis.left.label").bold is True


def test_pca_title_no_longer_carries_the_explained_variance(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"probe": _df()})
    w.set_feature_selection(["dwell_s", "depth_pA", "xcol"])
    w.set_plot_type("PCA")
    w.run_pca()
    assert "Explained variance" not in _title_text(w)
    assert "Explained variance" in w.details_text()
