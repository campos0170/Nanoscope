"""Matplotlib editing shares the real Format controller without regenerating analysis."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
from PySide6 import QtCore, QtTest

from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
from nanoscope.gui.format_inspector import ElementRef, FormatPane, SelectionHub
from nanoscope.gui.plot_workflow import FormatSpec
from .test_comparison_figures_panel import source, forbid_prepare


def plotted(qtbot):
    page = ComparisonFiguresWindow()
    qtbot.addWidget(page)
    page.resize(1100, 800)
    page.show()
    page.set_sources([source("a"), source("b", "Recording B")], selected_keys=["a", "b"])
    page.set_config({"kind": "facets"})
    assert page.analyze()
    page.fit_canvas()
    return page


def panel_controller(page):
    return next(c for c in page._text_controllers if c.panel != "figure")


def assert_tables(left, right):
    assert set(left) == set(right)
    for key in left:
        pd.testing.assert_frame_equal(left[key], right[key], check_exact=True)


def test_inline_panel_title_is_one_undo_and_redraw_uses_only_committed_tables(qtbot, monkeypatch):
    page = plotted(qtbot)
    before = deepcopy(page._tables)
    forbid_prepare(monkeypatch)
    controller = panel_controller(page)
    automatic = controller.target("title").auto
    original_global = page._appearance_widget.state().title
    page._view.zoom(1.25)
    zoom = page._view_state()
    mark = page._appearance_widget.history_mark()
    rect = controller.anchor_rect("title")
    QtTest.QTest.mouseDClick(page._view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=rect.center())
    assert controller.editor is not None and controller.editor.isVisible()
    controller.editor.setText("β $literal$")
    QtTest.QTest.keyClick(controller.editor, QtCore.Qt.Key.Key_Return)
    assert controller.target("title").style.text == "β $literal$"
    assert controller.target("title").auto == automatic
    assert page._appearance_widget.state().title == original_global
    assert page._appearance_widget.history_mark() == mark + 1
    assert page._view_state() == zoom
    assert_tables(page._tables, before)
    assert page._appearance_widget.undo()
    assert controller.target("title").style.text is None
    assert_tables(page._tables, before)


def test_format_selection_and_canvas_selection_share_viewport_overlays(qtbot, monkeypatch):
    page = plotted(qtbot)
    forbid_prepare(monkeypatch)
    controller = panel_controller(page)
    hub = SelectionHub(page)
    pane = FormatPane(FormatSpec(page._appearance_widget, lambda: page._text_controllers,
                                views_changed=page.formatViewsChanged), hub, page)
    pane.resize(300, 650)
    pane.show()
    ref = ElementRef(controller.host.pin_view, "title")
    before = page._figure_item.native_figure_svg()
    pane.select(ref)
    assert hub.current() == ref
    bar = controller._nanoscope_minibar
    assert bar.parentWidget() is page._view.viewport()
    assert controller._nanoscope_outline.parentWidget() is page._view.viewport()
    assert page._figure_item.native_figure_svg() == before  # overlays are not drawing items
    controller.set_style_property("title", "bold", True)
    assert controller.target("title").style.bold is True
    assert hub.current() == ref
    assert pane.controller_for(ref) is controller  # redrawing retains controller identity
    pane.retire()


def test_axis_format_pins_range_without_replacing_tick_numbers_or_scientific_data(qtbot, monkeypatch):
    page = plotted(qtbot)
    forbid_prepare(monkeypatch)
    controller = panel_controller(page)
    before = deepcopy(page._tables)
    target = controller.target("axis.bottom.ticks")
    assert target.capabilities == frozenset({"style"})
    axis = controller.host.axis_state("bottom")
    lo, hi = axis.shown_range
    from nanoscope.gui.plot_appearance import valid_pin_key
    assert valid_pin_key(controller.host.pin_view + ".bottom")
    controller.host.apply_axis("bottom", replace(axis, range=(lo, lo + .75 * (hi - lo)), tick_pt=8.5))
    assert controller.host.axis_state("bottom").range == (lo, lo + .75 * (hi - lo))
    assert controller.target("axis.bottom.ticks").style.size_pt == 8.5
    assert_tables(page._tables, before)
    payload, tables = page.snapshot_state()
    reopened = ComparisonFiguresWindow()
    qtbot.addWidget(reopened)
    reopened.restore_state(payload, tables)
    restored = next(c for c in reopened._text_controllers if c.panel == controller.panel)
    assert restored.host.axis_state("bottom").range == controller.host.axis_state("bottom").range
    assert_tables(reopened._tables, before)


def test_restoring_empty_comparison_retires_the_previous_targets(qtbot, monkeypatch):
    empty = ComparisonFiguresWindow()
    qtbot.addWidget(empty)
    payload, tables = empty.snapshot_state()
    page = plotted(qtbot)
    forbid_prepare(monkeypatch)
    previous = tuple(page._text_controllers)
    panel_controller(page).edit_inline("title")
    page.restore_state(payload, tables)
    assert page._text_controllers == []
    assert page._figure_item is None
    assert all(controller._retired and controller.editor is None for controller in previous)


def test_facet_words_follow_source_identity_when_sources_reorder(qtbot):
    page = plotted(qtbot)
    first = next(c for c in page._text_controllers if c.panel == 'facet:source:a')
    first.set_style_property('title', 'text', 'Recording A only')
    page.set_sources([source('b', 'Recording B'), source('a')], selected_keys=['b', 'a'])
    assert page.analyze()
    assert next(c for c in page._text_controllers if c.panel == first.panel) is first
    assert first.target('title').style.text == 'Recording A only'
    other = next(c for c in page._text_controllers if c.panel == 'facet:source:b')
    assert other.target('title').style.text is None


def test_quantity_reset_clears_panel_words_and_changed_axis_pin_in_one_undo(qtbot, monkeypatch):
    page = plotted(qtbot)
    controller = panel_controller(page)
    controller.set_style_property('title', 'text', 'Old measurement')
    controller.set_style_property('title', 'bold', True)
    for side in ('bottom', 'left'):
        state = controller.host.axis_state(side)
        lo, hi = state.shown_range
        controller.host.apply_axis(side, replace(state, range=(lo, lo + .8 * (hi - lo))))
    x_before = controller.host.axis_state('bottom').range
    y_before = controller.host.axis_state('left').range
    mark = page._appearance_widget.history_mark()
    page.set_config({'y': 'snr'})
    assert page.analyze()
    assert controller.target('title').style.text is None
    assert controller.target('title').style.bold is True
    assert controller.host.axis_state('bottom').range == x_before
    assert controller.host.axis_state('left').range is None
    assert page._appearance_widget.history_mark() == mark + 1
    forbid_prepare(monkeypatch)
    assert page._appearance_widget.undo()
    assert controller.target('title').style.text == 'Old measurement'
    assert controller.host.axis_state('left').range == y_before


def test_recalled_template_panel_words_keep_only_their_unmodified_promise(qtbot):
    page = plotted(qtbot)
    controller = panel_controller(page)
    state = page._appearance_widget.state().with_text_style(
        controller.host.storage_key('title'), replace(controller.target('title').style, text='Template measurement'))
    page.apply_template_settings({'y': 'snr'}, state.to_dict())
    assert page.analyze()
    assert controller.target('title').style.text == 'Template measurement'
    page.apply_template_settings({'y': 'amplitude_pA'}, state.to_dict())
    controller.set_style_property('title', 'text', 'Typed for current SNR')
    assert page.analyze()
    assert controller.target('title').style.text is None
