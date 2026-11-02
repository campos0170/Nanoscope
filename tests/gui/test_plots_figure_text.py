"""Distributions: in-figure text and axis editing, and literal text (plotux G3).

The Plots window's ``FigureTextHost`` writes every in-figure edit into its own
StyleSettings / AxesConfig as ONE step of its Undo, so an edit made on the figure and the
same edit made in the panel are the same change. Typed text is drawn literally, with
``_{…}``/``^{…}`` for sub/superscripts. Synthetic DataFrames only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore
from PySide6.QtTest import QTest

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.plots_panel import PlotsWindow, plot_config_from_json, plot_config_to_json


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _window(qtbot, frames=None) -> PlotsWindow:
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 5.0, 200)
    df = pd.DataFrame({"xcol": x, "ycol": 2 * x + rng.normal(0, 0.3, 200),
                       "accepted": True, "filter_limited": False})
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1100, 760)
    w.show()
    qtbot.waitExposed(w)
    w.set_data(frames or {"probe": df})
    if frames:
        w.set_sources(list(frames))
    w.set_plot_type("Scatter")
    w.set_scatter_columns("xcol", "ycol")
    return w


def test_a_typed_title_and_axis_label_are_drawn_literally(qapp, qtbot):
    w = _window(qtbot)
    w.set_title("I < 5 pA & τ > 1 ms x_{0}")
    item = w._plot.titleLabel.item
    assert item.toPlainText() == "I < 5 pA & τ > 1 ms x0"
    assert "<sub>0</sub>" in w._plot.titleLabel.text
    w._style = replace(w._style, x_label="a<b> c")
    w._presentation_changed()                       # the restyle path
    assert w._plot.getAxis("bottom").label.toPlainText().startswith("a<b> c")
    w._redraw()                                     # and a full draw
    assert w._plot.getAxis("bottom").label.toPlainText().startswith("a<b> c")


def test_double_click_on_the_title_edits_it_as_one_undo_step_that_survives_json(qapp, qtbot):
    w = _window(qtbot)
    w.set_title("Draft")
    w._history_touch()          # close the typed run, as a focus-out would
    back = w._history.depth()[0]
    view = w._plot_widget
    rect = w._plot.titleLabel.sceneBoundingRect()
    pos = view.mapFromScene(rect.center())
    handle = view.window().windowHandle()
    at = view.viewport().mapTo(view.window(), pos)
    QTest.mouseDClick(handle, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier(0), at)
    qapp.processEvents()
    controller = figure_text.figure_text_for(w._plot)
    editor = controller.editor
    assert editor is not None and editor.text() == "Draft"
    editor.selectAll()
    QTest.keyClicks(editor, "Fig 2")
    QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    assert w.config().style.title == "Fig 2"
    assert w._plot.titleLabel.item.toPlainText() == "Fig 2"
    assert w._history.depth()[0] == back + 1
    assert plot_config_from_json(plot_config_to_json(w.config())).style.title == "Fig 2"
    assert w.undo_config()
    assert w.config().style.title == "Draft"


def test_text_style_on_the_y_label_is_one_step_and_reaches_the_axis(qapp, qtbot):
    w = _window(qtbot)
    host = figure_text.figure_text_for(w._plot).host
    back = w._history.depth()[0]
    style = w.config().style.text_style("axis.left.label")
    host.apply_text("axis.left.label", replace(style, bold=True, color="#d55e00"))
    assert w._history.depth()[0] == back + 1
    css = w._plot.getAxis("left").labelStyle
    assert css.get("font-weight") == "bold" and css.get("color") == "#d55e00"
    again = plot_config_from_json(plot_config_to_json(w.config()))
    assert again.style.text_style("axis.left.label").bold is True


def test_axis_properties_pin_the_range_into_the_axis_range_boxes(qapp, qtbot):
    w = _window(qtbot)
    controller = figure_text.figure_text_for(w._plot)
    state = w._figure_axis_state("bottom")
    assert state.log_available
    back = w._history.depth()[0]
    controller._apply_axis("bottom", replace(state, range=(2.0, 4.0)))
    assert w.config().axes.manual_x == pytest.approx((2.0, 4.0))
    assert w._range_edits["xmin"].text() == "2" and w._range_edits["xmax"].text() == "4"
    assert w._plot.getViewBox().viewRange()[0] == pytest.approx([2.0, 4.0])
    assert w._history.depth()[0] == back + 1
    # Auto gives the axis back to the plot.
    controller._apply_axis("bottom", replace(w._figure_axis_state("bottom"), range=None))
    assert w.config().axes.manual_x is None


def test_log_x_from_axis_properties_goes_through_the_pages_own_log_box(qapp, qtbot):
    w = _window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column("xcol")
    controller = figure_text.figure_text_for(w._plot)
    controller._apply_axis("bottom", replace(w._figure_axis_state("bottom"), log=True))
    # The same control the panel has, so the same gate and the same Undo apply.
    assert w._hist_logx.isChecked()
    assert w._figure_axis_state("bottom").log
    w.set_plot_type("PCA")
    assert not w._figure_axis_state("bottom").log_available


def test_legend_rename_from_the_figure_writes_the_one_name_resolver_escaped(qapp, qtbot):
    rng = np.random.default_rng(1)
    frames = {}
    for name in ("apo", "holo"):
        x = rng.uniform(1.0, 5.0, 100)
        frames[name] = pd.DataFrame({"xcol": x, "ycol": x + rng.normal(0, 0.2, 100),
                                     "accepted": True, "filter_limited": False})
    w = _window(qtbot, frames)
    targets = figure_text.figure_text_for(w._plot).targets()
    keys = [k for k in targets if k.startswith("legend.entry.")]
    assert len(keys) == 2, list(targets)
    host = figure_text.figure_text_for(w._plot).host
    host.apply_text(keys[0], figure_text.TextStyle(text="Apo <1>"))
    src_key = keys[0][len("legend.entry."):]
    assert w.config().style.label_overrides.get(src_key) == "Apo <1>"
    rows = [label.item.toPlainText() for _s, label in w._plot.legend.items]
    assert any(r.startswith("Apo <1>") for r in rows), rows
