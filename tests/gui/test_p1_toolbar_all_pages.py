"""Phase 1, shell group (G1): the one toolbar on each of the eight pages.

Each page is built on an empty Session (synthetic data only), hosted in a real Explorer and
driven through the toolbar by real clicks: Undo and Redo mirror and drive the page's own
history, Update plot shows a count read from state and runs the page's own verb, the drop-down
runs the registered refresher, Reset figure is one Undo step, Export ▾ lists the page's own
entries, Reset view is on the status line and Lasso select mirrors its action.

What each page's count MEANS belongs to the page groups; what the toolbar does with it is here.
Offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import pytest
from PySide6 import QtCore, QtWidgets

from . import p1_hosting as H

PAGES = ("distributions", "distribution_summaries", "comparison_figures", "event_waveforms",
         "noise", "iv", "voltage_series", "subpopulations")
STATUS_LINE_VIEW = ("distributions", "distribution_summaries", "event_waveforms", "noise", "iv",
                    "voltage_series")
SHELL_ITEMS = ["Save to Figure Library…", "Save figure bundle…", "Open Figure Library"]
#: Names a page might call to compute; none may run when the toolbar only READS state.
COMPUTE_NAMES = ("refresh", "_refresh_from_user", "_refresh_from_button", "_redraw", "_run_plot",
                 "analyze", "_generate_from_user", "_apply_fit_from_user", "_apply_fit",
                 "_apply_fit_from_button", "analyze_selected")


def _hosted(qtbot, key, *, refresh=None):
    page = H.page_builders()[key]()
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    explorer.show_page(key, page, refresh=refresh)
    explorer.show()
    explorer.activateWindow()
    qtbot.waitExposed(explorer)
    QtWidgets.QApplication.processEvents()
    return explorer, page, explorer.adapter(key)


def _title(adapter, text):
    widget = adapter.format.appearance
    widget.apply_appearance(replace(widget.state(), title=text))
    QtWidgets.QApplication.processEvents()


def _shown_title(adapter):
    return adapter.format.appearance.state().title


@pytest.mark.parametrize("key", PAGES)
def test_undo_and_redo_mirror_the_pages_own_history_and_a_real_click_drives_it(qtbot, key):
    explorer, page, adapter = _hosted(qtbot, key)
    bar = explorer.toolbar()
    assert not bar.undo_action.isEnabled() and not bar.redo_action.isEnabled()
    _title(adapter, "Pore 3")
    qtbot.waitUntil(lambda: adapter.history().can_undo, timeout=3000)
    if adapter.history_changed is None:               # the fallback's Distributions: no signal yet
        bar.sync()
    state = adapter.history()
    assert state.can_undo and not state.can_redo
    assert bar.undo_action.isEnabled() and bar.undo_action.toolTip() == state.undo_text
    assert "Undo" in state.undo_text
    H.press_undo(explorer)
    assert _shown_title(adapter) is None
    assert not bar.undo_action.isEnabled() and bar.redo_action.isEnabled()
    assert bar.redo_action.toolTip() == adapter.history().redo_text
    H.press_redo(explorer)
    assert _shown_title(adapter) == "Pore 3" and bar.undo_action.isEnabled()
    assert not bar.redo_action.isEnabled()


@pytest.mark.parametrize("key", PAGES)
def test_update_plot_shows_the_pages_count_without_calling_anything_that_computes(
        qtbot, monkeypatch, key):
    explorer, page, adapter = _hosted(qtbot, key)
    bar = explorer.toolbar()
    verb = adapter.update
    adapter.update = lambda: pytest.fail("reading the count ran the page's verb")
    with monkeypatch.context() as patch:              # undone before the page closes (it may compute)
        for name in COMPUTE_NAMES:
            if hasattr(page, name):
                patch.setattr(page, name, lambda *_a, **_k: pytest.fail(f"{name} computed"))
        for _ in range(2):                            # a tab round trip, the refresh route
            explorer.show_home()
            explorer.raise_page(key)
            explorer.refresh_header()
            bar.sync()
    adapter.update = verb
    count = adapter.pending().count
    assert bar.update_button.text() == (f"Update plot ({count})" if count else "Update plot")
    assert bar.update_button.property("pending") is bool(count)
    assert bar.update_action.isEnabled() == (not adapter.pending().busy)


@pytest.mark.parametrize("key", PAGES)
def test_a_real_click_on_update_plot_runs_the_pages_verb_once(qtbot, key):
    explorer, page, adapter = _hosted(qtbot, key)
    calls = []
    adapter.update = lambda: calls.append(key)
    H.press_update(explorer)
    assert calls == [key]
    assert explorer.toolbar().update_button.isEnabled()


@pytest.mark.parametrize("key", PAGES)
def test_the_drop_down_item_runs_the_registered_refresher_and_not_the_verb(qtbot, key):
    seen = []
    explorer, page, adapter = _hosted(qtbot, key, refresh=lambda: seen.append("refresh"))
    adapter.update = lambda: seen.append("update")
    bar = explorer.toolbar()
    bar.update_button.showMenu()
    menu = bar.update_button.menu()
    item = next(a for a in menu.actions() if a.text() == "Refresh from recordings")
    assert item.isEnabled()
    H.QtTest.QTest.mouseClick(menu, H.L, H.NOMOD, menu.actionGeometry(item).center())
    QtWidgets.QApplication.processEvents()
    assert seen == ["refresh"]


@pytest.mark.parametrize("key", PAGES)
def test_reset_figure_is_one_undo_step_and_takes_the_words_back(qtbot, key):
    explorer, page, adapter = _hosted(qtbot, key)
    _title(adapter, "Pore 3")
    qtbot.waitUntil(lambda: adapter.history().can_undo, timeout=3000)
    tip = explorer.toolbar().reset_figure_button.toolTip()
    assert "Undo" in tip and "defaults" in tip
    H.press_reset_figure(explorer)
    assert _shown_title(adapter) is None
    H.press_undo(explorer)                          # ONE step: a single Undo brings the words back
    assert _shown_title(adapter) == "Pore 3"
    H.press_undo(explorer)                          # and the stack under it is the edit alone
    assert _shown_title(adapter) is None


@pytest.mark.parametrize("key", PAGES)
def test_export_lists_the_pages_own_entries_in_kind_order_then_the_three_shell_items(qtbot, key):
    explorer, page, adapter = _hosted(qtbot, key)
    entries = list(adapter.exports())
    assert entries, "every page has at least its figure export"
    order = {"figure": 0, "data": 1, "recipe": 2, "page": 3}
    expected = [e.text for e in sorted(entries, key=lambda e: order.get(e.kind, 9))]
    menu = H.export_menu(explorer)
    shown = [(a.text(), a.isEnabled()) for a in menu.actions() if a.text()]
    menu.hide()
    assert [t for t, _ in shown[:-3]] == expected
    assert [t for t, _ in shown[-3:]] == SHELL_ITEMS
    by_text = {e.text: e for e in entries}
    for text, enabled in shown[:-3]:
        predicate = by_text[text].enabled
        assert enabled == (bool(predicate()) if predicate is not None else True), text
    assert len(set(t for t, _ in shown)) == len(shown), "no entry twice"


@pytest.mark.parametrize("key", STATUS_LINE_VIEW)
def test_reset_view_is_on_the_pages_status_line_and_a_real_click_resets_that_pages_view(
        qtbot, monkeypatch, key):
    """A real click on the visible button reaches ``reset_view`` of a navigation whose plot is
    drawn inside THIS page. (An empty page has nothing to frame, so the number a view shows
    cannot tell a reset from none; the call can.)"""
    from nanoscope.gui import plot_interaction as pi
    explorer, page, adapter = _hosted(qtbot, key)
    buttons = adapter.status_bar().findChildren(QtWidgets.QToolButton, "plotResetViewButton")
    assert len(buttons) == 1 and buttons[0].isVisible(), "one Reset view, on the status line"
    assert buttons[0].isEnabled()
    assert not any(b.isVisible() for b in page.findChildren(
        QtWidgets.QAbstractButton, "plotResetViewButton") if b is not buttons[0])
    reset = []
    original = pi.PlotNavigation.reset_view
    monkeypatch.setattr(pi.PlotNavigation, "reset_view",
                        lambda self: reset.append(self) or original(self))
    H.click(buttons[0], pos=QtCore.QPoint(6, buttons[0].height() // 2))
    assert len(reset) == 1
    views = reset[0].plot.getViewBox().scene().views()
    assert views and page.isAncestorOf(views[0]), "the navigation of a plot in this page"


def test_lasso_select_on_distributions_is_on_its_status_line_and_mirrors_the_action(qtbot):
    explorer, page, adapter = _hosted(qtbot, "distributions")
    action = page._lasso_action
    buttons = [b for b in adapter.status_bar().findChildren(QtWidgets.QToolButton)
               if b.defaultAction() is action]
    assert len(buttons) == 1 and buttons[0].isVisible()
    button = buttons[0]
    assert not any(b.isVisible() for b in page._workflow_toolbar.findChildren(QtWidgets.QToolButton)
                   if b.defaultAction() is action)
    assert button.isEnabled() == action.isEnabled()
    if action.isEnabled():
        assert action.isCheckable() and not action.isChecked()
        H.click(button)
        assert action.isChecked() and button.isChecked()
        action.setChecked(False)                      # the lasso ending on its own
        assert not button.isChecked()
    action.setEnabled(False)                          # a plot type the lasso does not suit
    assert not button.isEnabled()
    action.setEnabled(True)
    assert button.isEnabled()


@pytest.mark.parametrize("key", PAGES)
def test_removing_the_front_page_then_clicking_every_button_raises_nothing(qtbot, key):
    explorer, page, adapter = _hosted(qtbot, key)
    explorer.remove_page(key)
    page.close()
    page.deleteLater()
    QtWidgets.QApplication.processEvents()
    bar = explorer.toolbar()
    for button in (bar.update_button, bar.reset_figure_button, bar.style_button,
                   bar.inspector_toggle):
        button.click()
    bar.undo_action.trigger()
    bar.redo_action.trigger()
    bar.update_action.trigger()
    H.key(explorer, QtCore.Qt.Key.Key_Return, H.CTRL)
    H.key(explorer, QtCore.Qt.Key.Key_2, H.CTRL)
    QtWidgets.QApplication.processEvents()
    assert explorer.page_count() == 0
