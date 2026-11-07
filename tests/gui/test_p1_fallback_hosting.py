"""Final Phase 1 native-adapter coverage retained from Foundation hosting checks.

Every real page now supplies its own adapter. Existing controls remain wired after moving;
only the page toolbar and empty section wrappers retire. Synthetic data only; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS, PlotAppearanceWidget

from . import p1_hosting as H

BAR_PAGES = ("distribution_summaries", "comparison_figures", "event_waveforms", "noise",
             "voltage_series", "subpopulations")
ALL = ("distributions", "iv", *BAR_PAGES)
RETURN = QtCore.Qt.Key.Key_Return


@pytest.fixture(scope="module")
def builders():
    return H.page_builders()


def _host(qtbot, builders, key, size=(1280, 820)):
    page = builders[key]()
    return H.hosted(qtbot, page, key, size=size), page


@pytest.mark.parametrize("key", ALL)
def test_the_appearance_controls_are_in_the_format_tab_once_and_nowhere_else(qtbot, builders, key):
    explorer, page = _host(qtbot, builders, key)
    host = explorer.inspector()
    widget = page._appearance_widget
    host.set_tab(0)
    assert not widget.isVisible()                      # not on the Data tab, not in the old page
    host.set_tab(1)
    QtWidgets.QApplication.processEvents()
    pane = host.pane(key).format_pane
    assert not pane.is_legacy() and not widget.isVisible()
    figure = pane.panel_for("figure")
    assert figure.isAncestorOf(widget.font_family) and widget.font_family.isVisible()
    # one place: the old Section's Appearance widgets are not also in a Data group
    for frame in (host.pane(key).data_pane.group(k) for k in host.pane(key).data_pane.group_keys()):
        assert not frame.isAncestorOf(widget)
    assert isinstance(widget, PlotAppearanceWidget)


@pytest.mark.parametrize("key", BAR_PAGES)
def test_a_section_name_reveals_the_inspector_instead_of_toggling_widgets(qtbot, builders, key):
    explorer, page = _host(qtbot, builders, key)
    host, bar = explorer.inspector(), page._workflow
    bar.set_section("Appearance")
    assert host.current_tab() == 1
    bar.set_section("Data")
    assert host.current_tab() == 0
    assert all(not s.isEnabled() for s in bar.section_shortcuts)
    names = host.pane(key).data_pane.group_keys()
    assert names == tuple(group.key for group in explorer.adapter(key).groups)
    target = explorer.adapter(key).reveal_map.get("Data", "sources")
    frame = host.pane(key).data_pane.group(target) or host.pane(key).data_pane.group(names[0])
    assert frame is not None and frame.is_expanded() and frame.isVisible()
    for section in ("Analysis", "Export"):
        bar.set_section(section)
    assert host.current_tab() == 0


@pytest.mark.parametrize("key", BAR_PAGES)
def test_the_pages_bar_is_hidden_and_its_objects_stay_wired(qtbot, builders, key):
    explorer, page = _host(qtbot, builders, key)
    bar = page._workflow
    assert not bar.isVisible() and bar in explorer.adapter(key).chrome
    if key == "subpopulations":
        assert page._readout.isVisible() and not bar.isAncestorOf(page._readout)
    clicks = []
    bar.primary_actions[0].clicked.connect(lambda: clicks.append(1))
    if key in ("event_waveforms", "noise", "voltage_series"):      # verbs with no dialog
        bar.primary_actions[0].click()                  # the compatibility object stays wired
        assert clicks == [1]


@pytest.mark.parametrize("key", BAR_PAGES)
def test_export_entries_are_the_bars_menu_and_the_toolbar_lists_them_in_order(qtbot, builders, key):
    explorer, page = _host(qtbot, builders, key)
    bar = page._workflow
    expected = [a.text() for a in bar.export_menu.actions() if not a.isSeparator()]
    menu = H.export_menu(explorer)
    shown = [a.text() for a in menu.actions() if not a.isSeparator()]
    menu.close()
    assert shown[:len(expected)] == expected
    assert shown[len(expected):] == ["Save to Figure Library…", "Save figure bundle…",
                                     "Open Figure Library"]
    enabled = {a.text(): a.isEnabled() for a in menu.actions() if a.text() in expected}
    entries = explorer.adapter(key).exports()
    assert all(enabled[entry.text] == (True if entry.enabled is None else bool(entry.enabled())) for entry in entries)


@pytest.mark.parametrize("key", ("distribution_summaries", "event_waveforms", "noise",
                                 "voltage_series", "distributions"))
def test_reset_view_moves_to_the_status_line_and_stays_one_button(qtbot, builders, key):
    explorer, page = _host(qtbot, builders, key)
    status = page.statusBar()
    on_line = status.findChildren(QtWidgets.QToolButton, "plotResetViewButton")
    assert len(on_line) == 1 and on_line[0].isVisible()
    assert len(page.findChildren(QtWidgets.QAbstractButton, "plotResetViewButton")) == 1


@pytest.mark.parametrize("key", ALL)
def test_no_control_root_is_owned_by_two_inspector_groups(qtbot, builders, key):
    explorer, page = _host(qtbot, builders, key)
    adapter = explorer.adapter(key)
    assert adapter is page.inspector_adapter() and not adapter.fallback
    roots = [widget for group in adapter.groups for widget in group.widgets]
    assert len({id(widget) for widget in roots}) == len(roots)
    for group in adapter.groups:
        frame = explorer.inspector().pane(key).data_pane.group(group.key)
        assert all(frame.body.isAncestorOf(widget) for widget in group.widgets)


@pytest.mark.parametrize("key", ALL)
def test_every_page_specific_format_control_reaches_its_named_panel(qtbot, builders, key):
    explorer, page = _host(qtbot, builders, key)
    if key == "noise":
        page.set_view("spectrogram")              # its colour-map controls belong to this view
    explorer.inspector().set_tab(1)
    QtWidgets.QApplication.processEvents()
    page_pane = explorer.inspector().pane(key)
    pane = page_pane.format_pane
    assert not pane.is_legacy()
    element_keys = {"figure": ("figure",), "legend": ("legend",),
                    "series": ("series.data",), "fit_label": ("annotation.fit",),
                    "caption": ("annotation.provenance", "annotation.caption"),
                    "colour_map": ("colour_map",)}
    for kind, widgets in explorer.adapter(key).format.extras.items():
        if key == "distributions" and kind == "colour_map":
            from nanoscope.gui.plots_panel import HEATMAP
            # A histogram has no colour scale. Offer its control only after the
            # actual heatmap workflow is selected and updated.
            assert not [ref for ref in pane.elements() if ref.key == "colour_map"]
            page.set_plot_type(HEATMAP)
            H.press_update(explorer)
            QtWidgets.QApplication.processEvents()
        offered = [ref for ref in pane.elements() if ref.key in element_keys[kind]]
        assert offered, (key, kind, "the control needs an offered element, not a hidden panel")
        pane.show_element(offered[0])              # selecting builds and shows its lazy panel
        QtWidgets.QApplication.processEvents()
        assert page_pane.hub.current() == offered[0]
        panel = pane.current_panel()
        assert panel is not None and panel.objectName() == f"formatPanel_{kind}"
        assert all(panel.isAncestorOf(widget) for widget in widgets), (key, kind)
    if key == "distributions":
        page.set_plot_type("Histogram")
        H.press_update(explorer)
        QtWidgets.QApplication.processEvents()
        assert not [ref for ref in pane.elements() if ref.key == "colour_map"]


def test_update_plot_applies_only_the_pending_verb_and_a_page_with_none_runs_its_first(
        qtbot, builders, monkeypatch):
    from .test_p0_reset_on_quantity_change import _user_picks
    explorer, page = _host(qtbot, builders, "voltage_series")
    refresh, apply_fit = page._workflow.primary_actions
    heard = []
    original_refresh, original_fit = page._refresh_from_user, page._apply_fit_from_user
    monkeypatch.setattr(page, "_refresh_from_user",
                        lambda: (heard.append("refresh"), original_refresh())[1])
    monkeypatch.setattr(page, "_apply_fit_from_user",
                        lambda: (heard.append("apply"), original_fit())[1])
    H.press_update(explorer)
    assert heard == ["refresh"]
    other = next(page._fit_combo.itemText(i) for i in range(page._fit_combo.count())
                 if page._fit_combo.itemText(i) != page._fit_combo.currentText())
    _user_picks(page._fit_combo, other)
    assert "●" in apply_fit.text()
    H.press_update(explorer)
    assert heard == ["refresh", "apply"]                 # the fit is applied, nothing re-gathered


def test_the_reflowed_voltage_controls_fit_without_a_sideways_scrollbar(qtbot, builders):
    explorer, page = _host(qtbot, builders, "voltage_series")       # its analysis strip is 518 px
    scroll = explorer.inspector().pane("voltage_series").data_pane
    explorer.inspector().reveal("plot")
    QtWidgets.QApplication.processEvents()
    content = scroll.widget().minimumSizeHint().width()
    assert content <= scroll.viewport().width()
    assert scroll.horizontalScrollBar().maximum() == 0


def test_noise_leaves_no_empty_controls_column_behind(qtbot, builders):
    page = builders["noise"]()
    column = page._recording_group.parentWidget()
    while column is not None and not isinstance(column, QtWidgets.QScrollArea):
        column = column.parentWidget()
    assert column is not None and column.isVisibleTo(page)        # the page's left column
    explorer = H.hosted(qtbot, page, "noise")
    assert column.isHidden()
    assert page._plot_widget.width() > 600


def test_a_pending_verb_marker_is_the_count_and_the_text_stays(qtbot, builders):
    explorer, page = _host(qtbot, builders, "voltage_series")
    bar = explorer.toolbar()
    apply_fit = next(b for b in page._workflow.primary_actions if b.text().startswith("Apply fit"))
    assert bar.update_button.text() == "Update plot"
    from .test_p0_reset_on_quantity_change import _user_picks
    other = next(page._fit_combo.itemText(i) for i in range(page._fit_combo.count())
                 if page._fit_combo.itemText(i) != page._fit_combo.currentText())
    _user_picks(page._fit_combo, other)                 # a person's pick arms Apply fit
    assert "●" in apply_fit.text()                       # the hidden button still says so
    bar.sync()
    assert bar.update_button.text() == "Update plot (1)"
    assert all(reason in bar.update_action.toolTip()
               for reason in explorer.adapter("voltage_series").pending().reasons)


# -- Distributions ----------------------------------------------------------------------------

def test_distributions_controls_move_out_of_its_column_and_the_plot_gets_the_room(qtbot, builders):
    explorer, page = _host(qtbot, builders, "distributions")
    host = explorer.inspector()
    assert page._section_area.isHidden() and not page._workflow_toolbar.isVisible()
    assert page._plot_widget.width() > 700
    frame = host.pane("distributions").data_pane.group("plot")
    assert frame.body.isAncestorOf(page._plot_group)
    assert page._axis_range_group.isHidden()
    assert host.pane("distributions").data_pane.group("sources").body.isAncestorOf(page._source_group)
    assert host.pane("distributions").data_pane.group("groups").body.isAncestorOf(page._features_group)
    assert host.pane("distributions").data_pane.group("recipes").body.isAncestorOf(
        page._template_combo)
    assert not host.pane("distributions").data_pane.group("recipes").is_expanded()


def test_distributions_section_helper_now_reveals_the_inspector(qtbot, builders):
    explorer, page = _host(qtbot, builders, "distributions")
    host = explorer.inspector()
    page._show_control_section("Appearance")
    assert host.current_tab() == 1 and page._section_combo.currentText() == "Appearance"
    page._show_control_section("Export")
    assert host.current_tab() == 0 and host.pane("distributions").data_pane.group("recipes").is_expanded()
    page._show_control_section(0)                          # an index, as the combo gives it
    assert page._section_combo.currentText() == "Data"
    assert page._source_group.isVisible() and page._plot_group.isVisible()


def test_distributions_command_return_is_update_plot_and_runs_the_plot_once(qtbot, builders, monkeypatch):
    explorer, page = _host(qtbot, builders, "distributions")
    draws = []
    real = page._armed_redraw
    monkeypatch.setattr(page, "_armed_redraw", lambda: (draws.append(1), real())[1])
    assert page._run_action.shortcut().isEmpty()           # the shell owns the key now
    H.key(explorer, RETURN, H.CTRL)
    assert draws == [1]
    H.press_update(explorer)
    assert draws == [1, 1]


def test_distributions_count_is_stale_reasons_plus_a_pending_fit(qtbot, builders):
    explorer, page = _host(qtbot, builders, "distributions")
    bar = explorer.toolbar()
    assert bar.update_button.text() == "Update plot" and page._plot_armed
    page._mark_stale("the plot type")                      # the page rewrites "Plot / Apply ●"
    assert bar.update_button.text() == "Update plot (1)"    # ... and the toolbar hears it
    assert "the plot type" in bar.update_action.toolTip()
    from .test_p0_reset_on_quantity_change import _user_picks
    other = next(page._fit_combo.itemText(i) for i in range(page._fit_combo.count())
                 if page._fit_combo.itemText(i) != page._fit_combo.currentText())
    _user_picks(page._fit_combo, other)
    assert bar.update_button.text() == "Update plot (2)"
    assert "fit model" in bar.update_action.toolTip()
    H.press_update(explorer)
    assert not page._stale and bar.update_button.text() == "Update plot"


def test_distributions_undo_redo_are_its_whole_config_history_and_the_toolbar_mirrors_it(qtbot, builders):
    explorer, page = _host(qtbot, builders, "distributions")
    bar = explorer.toolbar()
    explorer.refresh_header()
    start = page._appearance_widget.state()
    page._appearance_widget.apply_look(_BUILTIN_PRESETS["7 pt Arial · Nature, Elsevier"])
    bar.sync()
    assert bar.undo_action.isEnabled() and bar.undo_action.toolTip() == page._undo_btn.toolTip()
    H.press_undo(explorer)
    assert page._appearance_widget.state().title_size == start.title_size
    assert bar.redo_action.isEnabled()
    H.press_redo(explorer)
    assert bar.undo_action.isEnabled()


def test_distributions_reset_figure_is_one_config_step(qtbot, builders):
    explorer, page = _host(qtbot, builders, "distributions")
    page._appearance_widget.apply_look(_BUILTIN_PRESETS["Slides · 18 pt Arial"])
    before = page._history.depth()[0]
    H.press_reset_figure(explorer)
    assert page._history.depth()[0] == before + 1


def test_distributions_lasso_select_sits_on_the_status_line_beside_its_mode_label(qtbot, builders):
    explorer, page = _host(qtbot, builders, "distributions")
    status = page.statusBar()
    buttons = status.findChildren(QtWidgets.QToolButton, "plotCanvasActionButton")
    assert len(buttons) == 1 and buttons[0].defaultAction() is page._lasso_action
    assert buttons[0].isVisible() and page._selection_mode_label.isVisible()
    assert buttons[0].text() == "Lasso select"


def test_distributions_export_menu_lists_its_own_three_entries(qtbot, builders):
    explorer, page = _host(qtbot, builders, "distributions")
    menu = H.export_menu(explorer)
    shown = [a.text() for a in menu.actions() if not a.isSeparator()]
    menu.close()
    assert shown[:3] == ["Export current figure…", "Export plot data…", "Export configuration JSON…"]
