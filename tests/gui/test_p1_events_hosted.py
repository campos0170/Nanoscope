"""Phase 1, group G4: Event waveforms hosted in a real Analysis Explorer.

Clicks and keys go through the QWindow, combos are picked from their popups, menus are clicked,
and the project round trip is a real ``_write_project`` reopened by ``_open_project`` until
``projectRestored``. Every assertion is on what is drawn or shown. Synthetic data only; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import figure_text_for
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS
from nanoscope.gui.plot_interaction import app_menu_for

from . import p1_hosting as H
from .test_event_plot import _fed_window
from .test_p0_reset_on_quantity_change import _click_button, _pick, _series_window
from .test_plot_window_routes import _edit_double_click, _main_window, _press_undo

KEY = "event_waveforms"
L = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
RETURN = QtCore.Qt.Key.Key_Return


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


@pytest.fixture
def page(qapp, qtbot, tmp_path):
    return _fed_window(qtbot, tmp_path)


@pytest.fixture
def explorer(qtbot, page):
    return H.hosted(qtbot, page, KEY)


def _tab(page, index):
    """Click a tab of the page's own Density / Ensemble tab widget (not the Explorer's)."""
    bar = page._tabs.tabBar()
    QtTest.QTest.mouseClick(bar, L, NOMOD, bar.tabRect(index).center())
    QtWidgets.QApplication.processEvents()


def _data_pane(explorer):
    return explorer.inspector().pane(KEY).data_pane


def _body(explorer, group):
    return _data_pane(explorer).group(group).body


# -- the groups -------------------------------------------------------------------------------------

def test_the_groups_hold_the_pages_own_widgets_and_the_strips_are_husks(explorer, page):
    adapter = explorer.adapter(KEY)
    assert adapter is page.inspector_adapter() and adapter.fallback is False
    assert [g.key for g in adapter.groups] == ["sources", "plot_map", "plot_ensemble", "recipes"]
    assert [g.title for g in adapter.groups] == [
        "Sources", "Plot · Event-count map", "Plot · Ensemble", "Recipes"]
    for group, widgets in (
            ("sources", (page._data_section, page._subset_combo)),
            ("plot_map", (page._x_combo, page._y_combo, page._logx, page._logy, page._bins)),
            ("plot_ensemble", (page._normalize, page._align_combo, page._npoints, page._pad,
                               page._band_combo)),
            ("recipes", (page._templates,))):
        for widget in widgets:
            assert _body(explorer, group).isAncestorOf(widget), (group, widget)
    for husk in (page._density_analysis_controls, page._ensemble_analysis_controls):
        assert husk.isHidden() and not _data_pane(explorer).isAncestorOf(husk)
    assert page.property("nanoscopeInspectorHosted") is True
    assert page.inspector_adapter() is adapter                    # idempotent
    assert _data_pane(explorer).group("recipes").is_expanded() is False


def test_the_hidden_set_stays_hidden_and_wired(explorer, page):
    for name in ("_d_save", "_d_data", "_e_save", "_e_data", "_legacy_toolbar",
                 "_appearance_section", "_density_export_controls", "_ensemble_export_controls"):
        assert getattr(page, name).isHidden(), name
    assert not page._workflow.isVisible()
    heard = []
    page.refreshRequested.connect(lambda: heard.append(1))
    page._workflow.primary_actions[0].click()                     # the hidden verb still works
    assert heard == [1]
    seen = []
    original = page.save_image
    page.save_image = lambda *a, **k: seen.append("image")
    page._e_save.click()                                          # a hidden button is still wired
    page.save_image = original
    assert seen == ["image"]


def test_the_groups_follow_the_inner_tab_with_real_clicks(explorer, page):
    pane = _data_pane(explorer)
    assert page.current_view() == "Density"
    assert pane.group("plot_map").isVisible() and pane.group("plot_ensemble").isHidden()
    assert pane.group("sources").isVisible() and page._data_section.isVisible()
    assert not page._subset_combo.isVisible()                     # the heatmap uses every event
    _tab(page, 1)
    assert page.current_view() == "Ensemble"
    assert pane.group("plot_ensemble").isVisible() and pane.group("plot_map").isHidden()
    assert page._subset_combo.isVisible() and page._band_combo.isVisible()
    _tab(page, 0)
    assert pane.group("plot_map").isVisible() and pane.group("plot_ensemble").isHidden()
    assert not page._subset_combo.isVisible()
    page.set_view("Ensemble")                                     # a project restore / template
    assert pane.group("plot_ensemble").isVisible() and page._subset_combo.isVisible()


def test_the_old_section_names_reveal_the_controls_of_the_view_on_screen(explorer, page):
    pane = _data_pane(explorer)
    pane.group("plot_map").set_expanded(False)
    pane.group("plot_ensemble").set_expanded(False)
    page._workflow.set_section("Analysis")
    assert explorer.inspector().current_tab() == 0 and pane.group("plot_map").is_expanded()
    _tab(page, 1)
    page._workflow.set_section("Appearance")
    assert explorer.inspector().current_tab() == 1
    page._workflow.set_section("Analysis")
    assert explorer.inspector().current_tab() == 0 and pane.group("plot_ensemble").is_expanded()
    page._workflow.set_section("Export")
    assert pane.group("recipes").is_expanded()
    page._workflow.set_section("Data")
    assert pane.group("sources").is_expanded()


# -- every control still does its job --------------------------------------------------------------

def test_the_map_controls_redraw_the_heatmap_through_real_picks_and_clicks(explorer, page):
    before = page.density_counts().copy()
    edges = page.density_edges()[0].copy()
    _pick(page._x_combo, data="frac_depth")
    assert page._x_combo.currentData() == "frac_depth"
    assert page.density_edges()[0].tolist() != edges.tolist()     # a new quantity on x
    logs = page._density_log_state()
    _click_button(page._logy)
    assert page._density_log_state() == (logs[0], not logs[1])
    bins = page._bins.value()
    page._bins.setFocus()
    page._bins.selectAll()
    QtTest.QTest.keyClicks(page._bins, "12")
    QtTest.QTest.keyClick(page._bins, QtCore.Qt.Key.Key_Tab)
    assert page._bins.value() == 12 != bins
    assert page.density_counts().shape == (12, 12) and page.density_counts().shape != before.shape


def test_the_ensemble_controls_redraw_the_ensemble_through_real_picks_and_clicks(explorer, page):
    _tab(page, 1)
    assert page._align_combo.isVisible() and not page._align_combo.isEnabled()
    _click_button(page._normalize)                                # real time: Align answers
    assert page._normalize.isChecked() is False and page._align_combo.isEnabled()
    _pick(page._align_combo, text="center")
    assert page._align_combo.currentText() == "center"
    _pick(page._band_combo, data="sd")
    assert page.ensemble_band() is not None
    page._npoints.setFocus()
    page._npoints.selectAll()
    QtTest.QTest.keyClicks(page._npoints, "60")
    QtTest.QTest.keyClick(page._npoints, QtCore.Qt.Key.Key_Tab)
    assert page.ensemble_traces()[1].shape[1] == 60
    _pick(page._subset_combo, data="Accepted only")               # the one subset it always has
    assert page._ensemble_subset() == "Accepted only"


# -- the verb, the exports, the history ------------------------------------------------------------

def test_update_plot_is_the_pages_refresh_signal_once_per_press_and_key(explorer, page):
    heard = []
    page.refreshRequested.connect(lambda: heard.append(1))
    H.press_update(explorer)
    assert heard == [1]
    H.key(explorer, RETURN, H.CTRL)
    assert heard == [1, 1]
    bar = explorer.toolbar()
    assert bar.update_button.text() == "Update plot"              # nothing is ever pending here
    assert explorer.adapter(KEY).pending().count == 0
    assert "Re-snapshot" in bar.update_action.toolTip()


def test_export_menu_is_the_right_click_menu_and_runs_the_pages_handlers(explorer, page,
                                                                         monkeypatch):
    shell = {"Save to Figure Library…", "Save figure bundle…", "Open Figure Library"}
    menu = H.export_menu(explorer)
    entries = [a.text() for a in menu.actions() if not a.isSeparator() and a.text() not in shell]
    menu.close()
    right_click: list[str] = []
    for action in app_menu_for(page._density_main).actions():
        right_click = [] if action.isSeparator() else right_click + [action.text()]
    assert entries == right_click == ["Export current figure…", "Export data (CSV)…"]
    seen = []
    monkeypatch.setattr(page, "save_image", lambda *a, **k: seen.append("image"))
    monkeypatch.setattr(page, "export_data", lambda *a, **k: seen.append("data"))
    H.choose_export(explorer, "Export current figure…")
    H.choose_export(explorer, "Export data (CSV)…")
    assert seen == ["image", "data"]


@pytest.mark.parametrize("tab", (0, 1), ids=("data_tab", "format_tab"))
def test_the_undo_key_takes_back_an_edit_made_in_the_figure_from_either_inspector_tab(
        explorer, page, tab):
    explorer.inspector().set_tab(tab)
    plot = page._density_main
    _edit_double_click(plot, plot.titleLabel, "Dwell vs depth")
    assert page.appearance_state().get("title") == "Dwell vs depth"
    assert explorer.toolbar().undo_action.isEnabled()
    _press_undo(page)
    assert "title" not in page.appearance_state()
    assert explorer.toolbar().redo_action.isEnabled()


def test_reset_view_sits_on_the_status_line_and_acts_on_the_view_on_screen(explorer, page):
    status = page.statusBar()
    buttons = status.findChildren(QtWidgets.QToolButton, "plotResetViewButton")
    assert len(buttons) == 1 and buttons[0].isVisible()
    legacy = page.findChildren(QtWidgets.QAbstractButton, "plotResetViewButtonLegacy")
    assert len(legacy) == 1 and not legacy[0].isVisible()
    for index, plot in ((0, page._density_main), (1, page._ensemble_plot)):
        _tab(page, index)
        lo, hi = plot.viewRange()[0]
        plot.setXRange(lo + 0.45 * (hi - lo), lo + 0.5 * (hi - lo), padding=0)
        QtWidgets.QApplication.processEvents()
        narrow = plot.viewRange()[0][1] - plot.viewRange()[0][0]
        assert narrow < 0.1 * (hi - lo)
        H.click(buttons[0], pos=QtCore.QPoint(8, buttons[0].height() // 2))
        framed = plot.viewRange()[0][1] - plot.viewRange()[0][0]
        assert framed > 5 * narrow, index                        # its own framing is back


# -- Format ----------------------------------------------------------------------------------------

def test_the_format_spec_follows_the_view_and_names_the_colour_map(explorer, page):
    spec = explorer.adapter(KEY).format
    assert spec.appearance is page._appearance_widget
    assert [c for c in spec.controllers()] == [figure_text_for(page._density_main)]
    assert spec.extras == {"colour_map": [page._density_palette]}
    assert page._density_palette._nanoscope_format_visible()
    assert page._cmap_combo.parentWidget() is page._density_palette
    seen = []
    spec.views_changed.connect(lambda *_: seen.append(page.current_view()))
    _tab(page, 1)
    assert spec.controllers() == [figure_text_for(page._ensemble_plot)]
    assert not page._density_palette._nanoscope_format_visible()
    _tab(page, 0)
    assert seen == ["Ensemble", "Density"]
    assert spec.controllers() == [figure_text_for(page._density_main)]
    assert page._density_palette._nanoscope_format_visible()


def test_the_colormap_pick_still_recolours_without_recounting(explorer, page):
    counts = page.density_counts().copy()
    page._cmap_combo.setCurrentText("magma")                      # (its panel is the Format group's)
    assert page._cmap_combo.currentText() == "magma"
    assert page.density_counts().tolist() == counts.tolist()


# -- sizes ----------------------------------------------------------------------------------------

def test_the_page_fits_the_explorers_minimum_size(qtbot, page):
    explorer = H.hosted(qtbot, page, KEY, size=(840, 560))
    explorer.resize(840, 560)
    QtWidgets.QApplication.processEvents()
    assert page.minimumSizeHint().width() <= 500
    host, pane = explorer.inspector(), explorer.inspector().pane(KEY)
    assert host.minimumSizeHint().width() <= 330
    for tab in (0, 1):
        _tab(page, tab)
        for group in explorer.adapter(KEY).groups:
            assert pane.data_pane.group(group.key).minimumSizeHint().width() <= 290, group.key
        host.set_tab(0)
        assert pane.data_pane.horizontalScrollBar().maximum() == 0
    assert page._density_glw.width() >= 420 and page._density_glw.height() >= 300
    _tab(page, 1)
    assert page._ensemble_widget.width() >= 420 and page._ensemble_widget.height() >= 300
    assert explorer.width() == 840


def test_the_typography_toggle_does_not_bring_back_the_tall_standalone_minimum(explorer, page):
    assert page._density_glw.minimumSize() == QtCore.QSize(280, 300)
    page._appearance_action.setChecked(True)
    page._appearance_action.setChecked(False)
    assert page._density_glw.minimumHeight() == 240 and page._ensemble_widget.minimumHeight() == 240


# -- the project -----------------------------------------------------------------------------------

def test_a_full_save_and_reopen_restores_the_hosted_controls_and_the_view(
        qapp, qtbot, tmp_path, monkeypatch):
    source = _series_window(qtbot, tmp_path, monkeypatch)
    first = next(iter(source._session.documents())).doc_id
    source._on_selection(first, 0)
    source._open_event_plots()
    page = source._event_plot_window
    explorer = source._analysis_explorer
    assert page.property("nanoscopeInspectorHosted") is True
    # _tabs is the page's own Density / Ensemble widget, not the Explorer's
    assert isinstance(page._tabs, QtWidgets.QTabWidget) and page._tabs is not explorer._tabs
    assert [page._tabs.tabText(i) for i in range(page._tabs.count())] == [
        "Event-count heatmap", "Ensemble"]
    page._bins.setValue(17)
    _pick(page._y_combo, data="frac_depth")
    _tab(page, 1)
    _pick(page._band_combo, data="sd")
    _click_button(page._normalize)
    _pick(page._align_combo, text="center")
    page._npoints.setValue(64)
    page._appearance_widget.apply_look(_BUILTIN_PRESETS["7 pt Arial · Nature, Elsevier"])
    project = tmp_path / "events-hosted.nanoscope"
    assert source._write_project(str(project))
    source._dirty = False
    saved = page.appearance_state()
    assert saved["title_size"] == 7

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_event_plots()
    again = reopened._event_plot_window
    explorer = reopened._analysis_explorer
    assert again.property("nanoscopeInspectorHosted") is True
    assert again.current_view() == "Ensemble" and again._tabs is not explorer._tabs
    assert again._bins.value() == 17 and again._y_combo.currentData() == "frac_depth"
    assert again._band_combo.currentData() == "sd" and again._normalize.isChecked() is False
    assert again._align_combo.currentText() == "center" and again._npoints.value() == 64
    assert again.appearance_state() == saved
    pane = explorer.inspector().pane(reopened._figure_workspace.id_for_page(again)).data_pane
    assert pane.group("plot_ensemble").isVisible() and pane.group("plot_map").isHidden()
    assert again._subset_combo.isVisible()
    reopened._dirty = False
