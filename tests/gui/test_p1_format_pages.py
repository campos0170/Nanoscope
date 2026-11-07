"""Phase 1, Format group: the Format tab on each of the eight pages.

Parametrized over the eight page ids; a page still hosted by the Foundation's fallback adapter is
SKIPPED (its adapter turns the panels on; the page group's own tests, and the integrator's run of
this file, say the rest). For a page with its own adapter: every listed element shows its panel,
a click on a text of the canvas selects it and shows the Format tab, one write is one Undo step
and keeps the words, and a run of five size clicks is one step. Real clicks through the QWINDOW;
synthetic data only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import figure_text_for
from nanoscope.gui.format_inspector import ElementRef

from . import p1_format_rig as F
from . import p1_hosting as H

PAGE_IDS = ("distributions", "distribution_summaries", "comparison_figures", "event_waveforms",
            "noise", "iv", "voltage_series", "subpopulations")


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


@pytest.fixture(scope="module")
def builders():
    return H.page_builders()


def _hosted(qtbot, builders, key):
    page = builders[key]()
    explorer = H.hosted(qtbot, page, key)
    if explorer.adapter(key).fallback:
        pytest.skip(f"{key} is still hosted by the fallback adapter")
    widget = page._appearance_widget                    # words, so each text has something to click
    widget.apply_appearance(replace(widget.state(), title="A title", x_label="Horizontal",
                                    y_label="Vertical"))
    widget.clear_history()
    explorer.inspector().set_tab(0)
    QtWidgets.QApplication.processEvents()
    pane = explorer.inspector().pane(key)
    return page, explorer, pane, pane.format_pane


def _clickable_text(page, fp):
    """A text of the canvas that is on screen and can take a click, or None."""
    for controller in fp._controllers():
        plot = controller.plot
        for key in ("axis.bottom.label", "axis.left.label", "title"):
            target = controller.target(key)
            shown = plot.titleLabel.isVisible() if key == "title" else target is not None and \
                target.item.label.isVisible()
            if target is not None and "style" in target.capabilities and shown:
                if controller.anchor_rect(key) and not controller.anchor_rect(key).isEmpty():
                    return plot, controller, key
    return None


def _mark(explorer, key) -> int:
    return explorer.adapter(key).history().mark


def test_distributions_external_undo_redo_refreshes_the_selected_format_panel(qtbot, builders):
    page, explorer, pane, fp = _hosted(qtbot, builders, "distributions")
    explorer.inspector().set_tab(1)
    fp.show_element("title")
    ref = pane.hub.current()
    fields = fp.current_panel().fields
    before = fp.target_for(ref).style.bold
    fp.write_text(ref, "bold", True)
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    assert fields.bold.isChecked()
    assert explorer.adapter("distributions").undo()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    assert fields.bold.isChecked() == bool(before)
    assert explorer.adapter("distributions").redo()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    assert fields.bold.isChecked()


def test_distributions_caption_can_be_enabled_from_format_while_not_drawn(qtbot, builders):
    page, explorer, pane, fp = _hosted(qtbot, builders, "distributions")
    explorer.inspector().set_tab(1)
    ref = ElementRef("", "annotation.provenance")
    assert not page.config().style.show_provenance
    assert ref in fp.elements()                       # the user can reach the disabled caption
    F.pick_row(fp.list, ref)
    panel = fp.current_panel()
    assert panel.kind == "caption" and panel.isAncestorOf(page._prov_check)
    assert page._prov_check.isVisibleTo(panel) and page._prov_check.isEnabled()
    assert fp.target_for(ref) is None                 # generated provenance has no text editor
    assert not panel.words.isVisibleTo(panel) and not panel.fields.isVisibleTo(panel)
    assert not fp.copy_button.isEnabled() and not fp.reset_button.isEnabled()
    page._history_commit()                         # finish the fixture's configuration changes
    before, data = page.config(), page._plot_data().copy(deep=True)
    mark = _mark(explorer, "distributions")
    option = QtWidgets.QStyleOptionButton()
    option.initFrom(page._prov_check)
    indicator = page._prov_check.style().subElementRect(
        QtWidgets.QStyle.SubElement.SE_CheckBoxIndicator, option, page._prov_check)
    H.click(page._prov_check, pos=indicator.center())
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    assert page.config().style.show_provenance
    qtbot.waitUntil(lambda: _mark(explorer, "distributions") == mark + 1, timeout=3000)
    assert _mark(explorer, "distributions") == mark + 1
    assert page._plot_data().equals(data)
    assert explorer.adapter("distributions").undo()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    assert page.config() == before and not page._prov_check.isChecked()
    assert fp.elements().count(ref) == 1 and pane.hub.current() == ref


@pytest.mark.parametrize("key", PAGE_IDS)
def test_every_listed_element_shows_a_panel_and_the_panels_fit_the_column(qtbot, builders, key):
    page, explorer, pane, fp = _hosted(qtbot, builders, key)
    explorer.inspector().set_tab(1)
    QtWidgets.QApplication.processEvents()
    assert not fp.is_legacy() and fp.elements()[0] == ElementRef("", "figure")
    for ref in fp.elements():
        fp.show_element(ref)
        panel = fp.current_panel()
        assert panel is not None and hub_is(pane, ref)
        assert panel.minimumSizeHint().width() <= 290, (key, ref, panel.minimumSizeHint())
    assert fp._scroll.horizontalScrollBar().maximum() == 0


def hub_is(pane, ref) -> bool:
    return pane.hub.current() == ref


@pytest.mark.parametrize("key", PAGE_IDS)
def test_a_click_on_a_text_of_the_canvas_selects_it_and_shows_the_format_tab(qtbot, builders, key):
    page, explorer, pane, fp = _hosted(qtbot, builders, key)
    found = _clickable_text(page, fp)
    if found is None:
        pytest.skip(f"{key} has no text on screen to click")
    plot, controller, element = found
    F.click_scene(plot, F.point_of(plot, element))
    assert pane.hub.current() == ElementRef(fp.plot_id(controller), element)
    assert explorer.inspector().current_tab() == 1 and fp.current_panel().kind == "text"


@pytest.mark.parametrize("key", PAGE_IDS)
def test_one_write_is_one_undo_step_and_five_size_clicks_are_one(qtbot, builders, key):
    page, explorer, pane, fp = _hosted(qtbot, builders, key)
    found = _clickable_text(page, fp)
    if found is None:
        pytest.skip(f"{key} has no text on screen to click")
    plot, controller, element = found
    F.click_scene(plot, F.point_of(plot, element))
    bar = controller._nanoscope_minibar
    words = controller.target(element).style.text
    before = _mark(explorer, key)
    H.click(bar.fields.bold)
    assert controller.target(element).style.bold is True and _mark(explorer, key) == before + 1
    assert controller.target(element).style.text == words                    # the words survived
    from .test_p1_mini_toolbar import _up_arrow
    for _ in range(5):
        H.click(bar.fields.size, pos=_up_arrow(bar.fields.size))
    assert controller.target(element).style.size_pt == 8.0
    assert _mark(explorer, key) == before + 2                                # bold, then the run
