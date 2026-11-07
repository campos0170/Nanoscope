"""Phase 1, group G4: Subpopulations hosted in a real Analysis Explorer.

Clicks and keys go through the QWindow, combos are picked from their popups, menus are clicked,
and the project round trip is a real ``_write_project`` reopened by ``_open_project`` until
``projectRestored``. Every assertion is on what is drawn or shown. Synthetic data only (the
report is the one ``test_p0_reset_on_quantity_change`` builds); offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import shiboken6
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import figure_text_for
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.subpop_window import (ASSIGNMENT_INSTALLED, ASSIGNMENT_NONE, SubpopReportWindow)

from . import p1_hosting as H
from .test_p0_reset_on_quantity_change import _pick, subpop_report  # noqa: F401  (a fixture)
from .test_plot_window_routes import (_add_title, _main_window, _press_undo, _window_pos,
                                      window_handle)

KEY = "subpopulations"
RETURN = QtCore.Qt.Key.Key_Return
NOTE = ("accepted now: 3 of 5; this run's fractions describe the 5 accepted at run time "
        "— re-run to update")


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


@pytest.fixture
def page(qapp, qtbot, subpop_report):
    w = SubpopReportWindow()
    qtbot.addWidget(w)
    w.set_report(subpop_report, subtitle="synthetic")
    return w


@pytest.fixture
def explorer(qtbot, page):
    return H.hosted(qtbot, page, KEY)


def _data_pane(explorer):
    return explorer.inspector().pane(KEY).data_pane


def _body(explorer, group):
    return _data_pane(explorer).group(group).body


# -- the groups -------------------------------------------------------------------------------------

def test_the_groups_hold_the_pages_own_widgets_and_the_selector_row_is_a_husk(explorer, page):
    adapter = explorer.adapter(KEY)
    assert adapter is page.inspector_adapter() and adapter.fallback is False
    assert [g.key for g in adapter.groups] == ["about", "view", "report"]
    assert [g.title for g in adapter.groups] == ["Sources", "View", "Report"]
    assert [g.expanded for g in adapter.groups] == [True, True, False]
    view = page.embedding_view()
    for group, widgets in (("about", (page._data_section,)),
                           ("view", (view._embed_combo, view._colour_combo)),
                           ("report", (page._gallery_btn, page._trace_btn))):
        for widget in widgets:
            assert _body(explorer, group).isAncestorOf(widget), (group, widget)
    assert view._selector_controls.isHidden()
    assert not _data_pane(explorer).isAncestorOf(view._selector_controls)
    assert page.property("nanoscopeInspectorHosted") is True
    assert page.inspector_adapter() is adapter


def test_the_report_and_both_figures_stay_in_the_page_and_the_report_shows(explorer, page):
    assert page._text.isVisible() and page.text().startswith(page.report().text()[:20])
    assert page._split.isAncestorOf(page._text) and page._split.isAncestorOf(page._figures)
    assert page._figures.isAncestorOf(page.embedding_view()) and \
        page._figures.isAncestorOf(page.waveform_view())
    assert page._caveat_summary.summary.isVisible() or page._standing_caveats.isVisible()


def test_the_hidden_set_stays_hidden_and_wired(explorer, page):
    for name in ("_copy_btn", "_export_btn", "_close_btn", "_appearance_section"):
        assert getattr(page, name).isHidden(), name
    assert not page._workflow.isVisible()
    heard = []
    page.analyzeRequested.connect(lambda: heard.append("analyze"))
    page.exportRequested.connect(lambda: heard.append("export"))
    page._workflow.primary_actions[0].click()                     # the hidden verb still works
    page._export_btn.click()
    assert heard == ["analyze", "export"]
    page._copy_btn.click()
    assert QtWidgets.QApplication.clipboard().text() == page.text()


def test_readout_move_invalidates_its_old_toolbar_item(qtbot, page):
    layout = page._workflow.layout()
    item = layout.itemAt(layout.indexOf(page._readout))
    readout = page._readout
    H.hosted(qtbot, page, KEY)
    assert not shiboken6.isValid(item)
    assert shiboken6.isValid(readout) and page._readout is readout
    assert readout.parentWidget() is page and page.layout().indexOf(readout) >= 0


def test_the_close_button_is_hidden_only_while_hosted(qapp, qtbot, subpop_report):
    alone = SubpopReportWindow()
    qtbot.addWidget(alone)
    alone.show()
    qtbot.waitExposed(alone)
    assert alone._close_btn.isVisible() and alone.minimumSize() == QtCore.QSize(900, 480)
    hosted = SubpopReportWindow()
    qtbot.addWidget(hosted)
    H.hosted(qtbot, hosted, KEY)
    assert not hosted._close_btn.isVisible() and hosted.minimumSize() == QtCore.QSize(520, 360)


# -- every control still does its job --------------------------------------------------------------

def test_the_selectors_redraw_the_scatter_through_real_picks(explorer, page):
    view = page.embedding_view()
    assert view.embedding() == "t-SNE"
    before = view.drawn()[0].copy()
    other = next(name for name in view.embeddings() if name != "t-SNE")
    _pick(view._embed_combo, text=other)
    assert view.embedding() == other and view.drawn()[0].tolist() != before.tolist()
    mode = next(name for name in view.colour_modes() if name != view.colour_mode())
    _pick(view._colour_combo, text=mode)
    assert view.colour_mode() == mode


def test_the_two_entry_buttons_are_gated_by_the_owners_word_and_click_for_real(explorer, page):
    assert not page._gallery_btn.isEnabled() and not page._trace_btn.isEnabled()
    page.set_assignment_state(ASSIGNMENT_INSTALLED)
    assert page.can_show_events() and page._trace_btn.isEnabled()
    heard = []
    page.galleryRequested.connect(lambda: heard.append("gallery"))
    page.traceRequested.connect(lambda: heard.append("trace"))
    explorer.reveal("report")
    H.click(page._gallery_btn)
    H.click(page._trace_btn)
    assert heard == ["gallery", "trace"]
    page.set_assignment_state(ASSIGNMENT_NONE, "retired")
    assert not page._gallery_btn.isEnabled()


# -- the verb, the count, the exports, the history -------------------------------------------------

def test_update_plot_requests_analysis_once_per_press_and_key(explorer, page):
    heard = []
    page.analyzeRequested.connect(lambda: heard.append(1))
    H.press_update(explorer)
    assert heard == [1]
    H.key(explorer, RETURN, H.CTRL)
    assert heard == [1, 1]
    assert "dialog opens first" not in explorer.toolbar().update_action.toolTip()
    assert "original detection" in explorer.toolbar().update_action.toolTip().lower()


def test_the_count_is_one_while_the_curation_sentence_says_the_set_moved(explorer, page):
    bar = explorer.toolbar()
    assert explorer.adapter(KEY).pending().count == 0 and bar.update_button.text() == "Update plot"
    page.set_curation_note(NOTE)                                  # no workspaceChanged, no refresh
    assert bar.update_button.text() == "Update plot (1)"
    assert NOTE in bar.update_action.toolTip()
    page.set_curation_note("")
    assert bar.update_button.text() == "Update plot"
    page.set_curation_note(NOTE)
    assert bar.update_button.text() == "Update plot (1)"
    page.set_report(page.report(), subtitle="again")              # a re-run clears the sentence
    assert bar.update_button.text() == "Update plot"


def test_export_menu_lists_the_report_entries_and_follows_the_report(explorer, page, qtbot):
    shell = {"Save to Figure Library…", "Save figure bundle…", "Open Figure Library"}
    menu = H.export_menu(explorer)
    entries = {a.text(): a.isEnabled() for a in menu.actions()
               if not a.isSeparator() and a.text() not in shell}
    menu.close()
    assert list(entries) == ["Export report and tables…", "Copy report"]
    assert all(entries.values())
    heard = []
    page.exportRequested.connect(lambda: heard.append("export"))
    H.choose_export(explorer, "Export report and tables…")
    assert heard == ["export"]
    QtWidgets.QApplication.clipboard().clear()
    H.choose_export(explorer, "Copy report")
    assert QtWidgets.QApplication.clipboard().text() == page.text()
    page.set_report(None)
    menu = H.export_menu(explorer)
    assert [a.isEnabled() for a in menu.actions()
            if not a.isSeparator() and a.text() not in shell] == [False, False]
    menu.close()


def test_the_undo_key_takes_back_an_edit_made_in_the_figure_from_either_inspector_tab(
        explorer, page):
    plot = page.embedding_view()._plot.getPlotItem()
    for tab, title in ((0, "Pore A"), (1, "Pore B")):
        explorer.inspector().set_tab(tab)
        _add_title(plot, title)
        assert page.appearance_state().get("title") == title
        assert explorer.toolbar().undo_action.isEnabled()
        _press_undo(page)
        assert "title" not in page.appearance_state()


# -- Format ----------------------------------------------------------------------------------------

def test_the_format_spec_has_a_controller_per_figure_and_the_series_note(explorer, page):
    spec = explorer.adapter(KEY).format
    assert spec.appearance is page._appearance_widget
    assert spec.controllers() == [figure_text_for(page.embedding_view()._plot.getPlotItem()),
                                  figure_text_for(page.waveform_view()._plot.getPlotItem())]
    assert spec.extras == {"series": [page._categorical_note]}
    assert spec.views_changed is page.formatViewsChanged
    assert callable(spec.series_provider)
    assert explorer.adapter(KEY).navigation() is None
    assert not page.findChildren(QtWidgets.QAbstractButton, "plotResetViewButton")


# -- the readout row and the splitters -------------------------------------------------------------

def test_the_readout_moves_to_a_row_of_its_own_and_reads_under_the_pointer(explorer, page):
    readout = page._readout
    assert readout.parentWidget() is page and readout.isVisible()
    assert not page._workflow.isAncestorOf(readout)
    rect = QtCore.QRect(readout.mapTo(explorer, QtCore.QPoint(0, 0)), readout.size())
    assert explorer.rect().contains(rect) and readout.width() > 150
    assert readout.mapTo(page, QtCore.QPoint(0, 0)).y() > page._embed_view.mapTo(
        page, QtCore.QPoint(0, 0)).y()                           # the bottom row
    plot = page.embedding_view()._plot.getPlotItem()
    x, y, _ = page.embedding_view().drawn()
    scene_pt = plot.getViewBox().mapViewToScene(QtCore.QPointF(float(x[0]), float(y[0])))
    handle = window_handle(page.embedding_view()._plot)
    QtTest.QTest.mouseMove(handle, _window_pos(plot, scene_pt))
    QtTest.QTest.qWait(60)
    QtTest.QTest.mouseMove(handle, _window_pos(plot, scene_pt) + QtCore.QPoint(3, 2))
    QtTest.QTest.qWait(60)
    assert readout.full_text()


def test_both_splitters_keep_their_shape(explorer, page):
    assert page.splitter().count() == 2 and page.figures_splitter().count() == 3
    assert page.splitter().widget(0) is page._text
    assert all(size > 0 for size in page.splitter().sizes())


# -- sizes ----------------------------------------------------------------------------------------

def test_the_page_fits_the_explorers_minimum_size(qtbot, page):
    explorer = H.hosted(qtbot, page, KEY, size=(840, 560))
    explorer.resize(840, 560)
    QtWidgets.QApplication.processEvents()
    assert page.minimumSizeHint().width() <= 500
    host, pane = explorer.inspector(), explorer.inspector().pane(KEY)
    assert host.minimumSizeHint().width() <= 330
    for group in explorer.adapter(KEY).groups:
        assert pane.data_pane.group(group.key).minimumSizeHint().width() <= 290, group.key
    assert pane.data_pane.horizontalScrollBar().maximum() == 0
    readout = page._readout
    rect = QtCore.QRect(readout.mapTo(explorer, QtCore.QPoint(0, 0)), readout.size())
    assert readout.isVisible() and explorer.rect().contains(rect)
    assert explorer.width() == 840 and page.width() <= 520


def test_the_old_section_names_reveal_the_matching_group(explorer, page):
    pane = _data_pane(explorer)
    for name in ("about", "view", "report"):
        pane.group(name).set_expanded(False)
    page._workflow.set_section("Data")
    assert pane.group("about").is_expanded() and explorer.inspector().current_tab() == 0
    page._workflow.set_section("Analysis")
    assert pane.group("view").is_expanded()
    page._workflow.set_section("Export")
    assert pane.group("report").is_expanded()
    page._workflow.set_section("Appearance")
    assert explorer.inspector().current_tab() == 1


# -- the project -----------------------------------------------------------------------------------

def test_a_full_save_and_reopen_keeps_the_splitters_the_view_and_the_hosting(
        qapp, qtbot, tmp_path, monkeypatch, subpop_report):
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    source = _main_window(qtbot)
    source._open_subpop_window(subpop_report)
    page = source._subpop_window
    explorer = source._analysis_explorer
    explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    assert page.property("nanoscopeInspectorHosted") is True and page._text.isVisible()
    view = page.embedding_view()
    other = next(name for name in view.embeddings() if name != "t-SNE")
    _pick(view._embed_combo, text=other)
    page.splitter().setSizes([300, page.splitter().width() - 300 - page.splitter().handleWidth()])
    page.figures_splitter().setSizes([260, 200, 90])
    splitter, figures = list(page.splitter().sizes()), list(page.figures_splitter().sizes())
    project = tmp_path / "subpop-hosted.nanoscope"
    assert source._write_project(str(project))
    source._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_subpopulations_view()
    again = reopened._subpop_window
    reopened._analysis_explorer.resize(1280, 860)
    QtWidgets.QApplication.processEvents()
    assert again.property("nanoscopeInspectorHosted") is True
    assert again.embedding_view().embedding() == other
    assert again.splitter().count() == 2 and again.figures_splitter().count() == 3
    assert again.splitter().sizes() == pytest.approx(splitter, abs=2)
    assert again.figures_splitter().sizes() == pytest.approx(figures, abs=2)
    live_key = reopened._figure_workspace.id_for_page(again)
    assert live_key is not None
    assert reopened._analysis_explorer.inspector().pane(live_key).data_pane.group(
        "view").body.isAncestorOf(again.embedding_view()._embed_combo)
    reopened._dirty = False
