"""Phase 0 final review R6: the Reset figure tool tip says what the button does to a pinned range.

Reset figure puts the typed words and the look back, and on every page that has **Axis properties**
except Distributions it also takes the range and the major tick spacing pinned there (they are part
of the appearance). Only the guide and the inventory said so; the tip, the one statement beside the
button, listed words, caption and look and then data, fits and analysis settings, as if that were
all. The six pages with Axis properties now add one sentence to the shared tip; Distributions (its
range lives in the Axis range boxes, which Reset figure keeps, and its tip says so), Comparison
figures (a picture with no Axis properties) and the study figure editor are as they were.

Synthetic data only; offscreen. The behaviour the sentence states is checked on a real page: a range
pinned in Axis properties and the real button.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from nanoscope.gui import figure_text
from nanoscope.gui.plot_appearance import PlotAppearanceWidget

from .test_p0_reset_on_quantity_change import (  # noqa: F401 -- the fixtures build the pages
    _click_button, _in_explorer, comparison, dist, events, noise, subpop, subpop_report,
    summaries, vdep)
from .test_plot_window_routes import _pin_through_axis_properties

SENTENCE = "A range or tick spacing typed in Axis properties goes with the look."
KEPT = {
    "events": " Here it also keeps the Event-count heatmap colormap.",
    "noise": " Here it also keeps the Spectrogram colormap and colour limits.",
}


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _shared_tip(qtbot):
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    return widget.reset_button.toolTip()


def _page_of(made, name):
    return made[1] if name == "summaries" else made[0]


@pytest.mark.parametrize("name", ["vdep", "summaries", "subpop", "events", "noise"])
def test_the_tip_of_a_page_with_axis_properties_says_a_pinned_range_goes_with_the_look(
        name, request, qtbot):
    page = _page_of(request.getfixturevalue(name), name)
    shared = _shared_tip(qtbot)
    tip = page._appearance_widget.reset_button.toolTip()
    assert tip == f"{shared} {SENTENCE}{KEPT.get(name, '')}", tip
    assert tip.count(SENTENCE) == 1


def test_the_tip_of_the_iv_page_says_it_too(qapp, qtbot, tmp_path):
    from nanoscope.gui.analysis_explorer import PAGE_IV
    from nanoscope.gui.iv_panel import IVWindow

    from .test_iv import _ohmic_session

    page = IVWindow(_ohmic_session(tmp_path))
    qtbot.addWidget(page)
    _in_explorer(qtbot, page, PAGE_IV)
    assert page._appearance_widget.reset_button.toolTip() == f"{_shared_tip(qtbot)} {SENTENCE}"


def test_comparison_figures_has_no_axis_properties_and_its_tip_does_not_mention_them(
        comparison, qtbot):
    page, _explorer, _drawn = comparison
    assert page._appearance_widget.reset_button.toolTip() == _shared_tip(qtbot)


def test_distributions_keeps_its_axis_range_boxes_and_its_tip_does_not_say_a_range_goes(
        dist, qtbot):
    page, _explorer = dist
    tip = page._appearance_widget.reset_button.toolTip()
    assert SENTENCE not in tip and "Axis range boxes" in tip
    page.set_axis_range(xmin=0.5, xmax=8.0)
    kept = page.config().axes
    assert kept.manual_x is not None
    _click_button(page._appearance_widget.reset_button)
    assert page.config().axes == kept


def test_the_button_does_what_the_sentence_says_on_a_page_with_axis_properties(vdep):
    """Voltage dependence: a Y range pinned in Axis properties goes back to Auto with the button,
    in the one Undo step of the reset, and Undo brings it back."""
    page, _explorer = vdep
    plot, widget = page.plot_item(), page._appearance_widget
    _pin_through_axis_properties(plot, "left", 20.0, 400.0)
    pinned = widget.state().axis_pin("left").range()
    assert pinned is not None
    steps = widget._history.depth()

    _click_button(page._workflow.reset_button)

    assert widget.state().axis_pin("left").range() is None
    assert widget._history.depth() == (steps[0] + 1, 0)
    assert widget.undo() and widget.state().axis_pin("left").range() == pinned
