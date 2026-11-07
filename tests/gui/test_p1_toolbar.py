"""Phase 1 Foundation, F4: the Explorer's one toolbar, driven by real clicks and keys.

On ``SyntheticPlotPage`` (the interface with no analysis) inside a real Explorer; the I-V
page has its own file. Every button is clicked through the QWindow; Export ▾ and Style ▾ are
real menus. Synthetic only; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui.analysis_toolbar import ExplorerToolbar, UpdateButton
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS
from nanoscope.gui.plot_workflow import InspectorGroup

from . import p1_hosting as H

KEY_RETURN, KEY_ENTER = QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter


def _explorer(qtbot, page=None, key="iv", **kw):
    page = page if page is not None else H.SyntheticPlotPage(key)
    explorer = H.hosted(qtbot, page, key, **kw)
    return explorer, page, explorer.toolbar()


def test_the_toolbar_is_one_row_in_the_documented_order(qtbot):
    explorer, page, bar = _explorer(qtbot)
    assert isinstance(bar, ExplorerToolbar) and bar.objectName() == "analysisToolbar"
    assert not bar.isMovable() and bar.leading_actions == []
    assert isinstance(bar.update_button, UpdateButton)
    assert bar.update_button.popupMode() == QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup
    widgets = [H._toolbar_button(explorer, bar.undo_action), H._toolbar_button(explorer, bar.redo_action),
               bar.update_button, bar.style_button, bar.reset_figure_button, bar.export_button,
               bar.inspector_toggle]
    xs = [w.mapTo(explorer, QtCore.QPoint(0, 0)).x() for w in widgets]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)
    ys = {w.mapTo(explorer, QtCore.QPoint(0, 0)).y() for w in widgets}
    assert max(ys) - min(ys) <= 6                          # one row
    assert bar.style_button.text() == "Style: custom ▾" and bar.export_button.text() == "Export ▾"
    assert bar.reset_figure_button.text() == "Reset figure"
    assert bar.update_action.shortcuts() and bar.undo_action.shortcuts() == [] \
        and bar.redo_action.shortcuts() == []             # NO key on Undo or Redo


def test_undo_and_redo_mirror_the_pages_own_history_and_a_real_click_drives_it(qtbot):
    explorer, page, bar = _explorer(qtbot)
    widget = page._appearance_widget
    assert not bar.undo_action.isEnabled() and not bar.redo_action.isEnabled()
    widget.title.textEdited.emit("Hello")                 # an edit in the page's own box
    widget._close_burst()
    assert bar.undo_action.isEnabled() and not bar.redo_action.isEnabled()
    assert bar.undo_action.toolTip() == widget.undo_button.toolTip() and "title" in bar.undo_action.toolTip()
    H.press_undo(explorer)
    assert widget.state().title is None
    assert not bar.undo_action.isEnabled() and bar.redo_action.isEnabled()
    assert bar.redo_action.toolTip() == widget.redo_button.toolTip()
    H.press_redo(explorer)
    assert widget.state().title == "Hello" and bar.undo_action.isEnabled()


def test_a_silent_change_is_picked_up_by_refresh_header_the_one_refresh_route(qtbot):
    explorer, page, bar = _explorer(qtbot)
    page.pending_count = 3                                # state moved, nothing emitted
    assert bar.update_button.text() == "Update plot"
    explorer.refresh_header()                             # MainWindow calls this after a restore
    assert bar.update_button.text() == "Update plot (3)"


def test_update_plot_shows_the_count_from_page_state_and_a_click_runs_the_pages_verb_once(qtbot):
    explorer, page, bar = _explorer(qtbot)
    button = bar.update_button
    assert button.text() == "Update plot" and button.property("pending") is False
    page.set_pending(2, "the metric", "a fit model")
    assert button.text() == "Update plot (2)" and button.property("pending") is True
    assert "the metric; a fit model" in bar.update_action.toolTip()
    assert "Count one update." in bar.update_action.toolTip() and "⌘Return" in bar.update_action.toolTip()
    assert page.updates == 0                               # reading the count ran nothing
    H.press_update(explorer)
    assert page.updates == 1
    page.set_pending(0)
    assert button.text() == "Update plot" and button.property("pending") is False
    H.press_update(explorer)                               # not gated on the count
    assert page.updates == 2


def test_command_return_and_enter_are_update_plot_and_busy_does_nothing(qtbot):
    explorer, page, bar = _explorer(qtbot)
    H.key(explorer, KEY_RETURN, H.CTRL)
    assert page.updates == 1
    H.key(explorer, KEY_ENTER, H.CTRL)                     # the keypad key is another Qt key
    assert page.updates == 2
    page.set_pending(0, busy=True)
    assert not bar.update_action.isEnabled() and not bar.update_button.isEnabled()
    H.key(explorer, KEY_RETURN, H.CTRL)
    H.press_update(explorer)
    assert page.updates == 2
    page.set_pending(0, busy=False)
    assert bar.update_button.isEnabled()
    H.key(explorer, KEY_RETURN, H.CTRL)
    assert page.updates == 3


def test_the_drop_down_item_runs_the_registered_refresher_and_nothing_else(qtbot):
    calls: list[str] = []
    page = H.SyntheticPlotPage("iv")
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    explorer.show_page("iv", page, refresh=lambda: calls.append("refresh"))
    explorer.show()
    qtbot.waitExposed(explorer)
    bar = explorer.toolbar()
    before = len(bar.update_button.actions())
    bar.update_button.showMenu()
    menu = bar.update_button.menu()
    assert [a.text() for a in menu.actions() if a.isVisible()] == ["Refresh from recordings"]
    QtTest.QTest.mouseClick(menu, H.L, H.NOMOD, menu.actionGeometry(menu.actions()[0]).center())
    QtWidgets.QApplication.processEvents()
    assert calls == ["refresh"] and page.updates == 0
    assert len(bar.update_button.actions()) == before


def test_command_return_commits_an_uncommitted_edit_in_the_inspector_first(qtbot):
    seen: list[str] = []
    page = H.SyntheticPlotPage("iv")
    edit = QtWidgets.QLineEdit()
    committed = {"text": ""}
    edit.editingFinished.connect(lambda: committed.__setitem__("text", edit.text()))
    adapter = page.inspector_adapter()
    adapter.groups[:] = [InspectorGroup("estimator", "Estimator", [edit])]
    adapter.update = lambda: seen.append(committed["text"])
    explorer = H.hosted(qtbot, page, "iv")
    edit.setFocus()
    qtbot.waitUntil(edit.hasFocus)
    QtTest.QTest.keyClicks(edit, "typed title")
    assert committed["text"] == ""                         # typed, not committed
    H.key(explorer, KEY_RETURN, H.CTRL)
    assert seen == ["typed title"]


def test_reset_figure_is_one_undo_step_and_says_so(qtbot):
    explorer, page, bar = _explorer(qtbot)
    widget = page._appearance_widget
    widget.apply_look(_BUILTIN_PRESETS["7 pt Arial · Nature, Elsevier"])
    steps = widget._history.depth()[0]
    assert "Undo" in bar.reset_figure_button.toolTip() and "defaults" in bar.reset_figure_button.toolTip()
    H.press_reset_figure(explorer)
    assert widget._history.depth()[0] == steps + 1 and widget.state().to_dict() == {}
    H.press_undo(explorer)
    assert widget.state().title_size == 7


def test_export_menu_lists_the_pages_entries_then_the_three_shell_items(qtbot):
    explorer, page, bar = _explorer(qtbot)
    menu = H.export_menu(explorer)
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts == ["Export current figure…", "Export data (CSV)…", "Save to Figure Library…",
                     "Save figure bundle…", "Open Figure Library"]
    assert sum(1 for a in menu.actions() if a.isSeparator()) == 1
    by_text = {a.text(): a for a in menu.actions()}
    assert by_text["Open Figure Library"].isEnabled()
    assert not by_text["Save to Figure Library…"].isEnabled()    # the header says "no data"
    menu.close()
    H.choose_export(explorer, "Export data (CSV)…")
    assert page.exported == ["data"]


def test_the_shell_export_items_are_enabled_exactly_as_the_old_buttons_were(qtbot):
    explorer, page, bar = _explorer(qtbot)
    explorer._status_provider = lambda _id, _page: ("current", "current")
    explorer.refresh_header()
    assert explorer._save_figure_btn.isEnabled() and explorer._save_bundle_btn.isEnabled()
    heard: list[tuple[str, str]] = []
    explorer.saveFigureRequested.connect(lambda key: heard.append(("figure", key)))
    explorer.saveBundleRequested.connect(lambda key: heard.append(("bundle", key)))
    explorer.figureLibraryRequested.connect(lambda: heard.append(("library", "")))
    for text in ("Save to Figure Library…", "Save figure bundle…", "Open Figure Library"):
        H.choose_export(explorer, text)
    assert heard == [("figure", "iv"), ("bundle", "iv"), ("library", "")]
    explorer._status_provider = lambda _id, _page: ("none", "empty")
    explorer.refresh_header()
    assert not explorer._save_figure_btn.isEnabled()
    menu = H.export_menu(explorer)
    by_text = {a.text(): a for a in menu.actions()}
    assert not by_text["Save to Figure Library…"].isEnabled()
    assert not by_text["Save figure bundle…"].isEnabled()
    assert by_text["Open Figure Library"].isEnabled()
    menu.close()


def test_the_inspector_toggle_shows_and_hides_the_column_and_remembers(qtbot):
    explorer, page, bar = _explorer(qtbot)
    host = explorer.inspector()
    assert host.isVisible() and bar.inspector_toggle.isChecked()
    H.click(bar.inspector_toggle)
    assert not host.isVisible() and host.settings().visible is False
    H.click(bar.inspector_toggle)
    assert host.isVisible() and host.settings().visible is True


def test_style_menu_applies_a_built_in_look_to_the_page_in_front_as_one_step(qtbot):
    explorer, page, bar = _explorer(qtbot)
    widget = page._appearance_widget
    steps = widget._history.depth()[0]
    bar.style_button.showMenu()
    menu = bar.style_button.menu()
    this_page = next(a for a in menu.actions() if a.text() == "This page")
    assert this_page.isEnabled()
    sub = this_page.menu()
    names = [a.text() for a in sub.actions() if not a.isSeparator()]
    assert names[:len(_BUILTIN_PRESETS)] == list(_BUILTIN_PRESETS)      # then the saved looks
    sub.popup(menu.mapToGlobal(menu.actionGeometry(this_page).topRight()))
    action = next(a for a in sub.actions() if a.text() == "9 pt Times New Roman · IEEE")
    QtTest.QTest.mouseClick(sub, H.L, H.NOMOD, sub.actionGeometry(action).center())
    QtWidgets.QApplication.processEvents()
    assert widget.state().font_family == "Times New Roman" and widget.state().title_size == 9
    assert widget._history.depth()[0] == steps + 1
    assert bar.undo_action.toolTip() == "Undo: 9 pt Times New Roman · IEEE preset"
    assert explorer.apply_look(_BUILTIN_PRESETS["Slides · 18 pt Arial"]).applied == ("iv",)
    assert widget.state().title_size == 18


def test_unbound_toolbar_has_placeholders_and_only_style_is_alive(qtbot):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.show()
    bar = explorer.toolbar()
    for w in (bar.update_button, bar.reset_figure_button, bar.export_button, bar.inspector_toggle):
        assert not w.isEnabled()
    assert not bar.undo_action.isEnabled() and not bar.redo_action.isEnabled()
    assert bar.style_button.isEnabled()
    assert bar.update_button.text() == "Update plot" and bar.undo_action.toolTip() == "Nothing to undo"
    assert not explorer.inspector().isVisible()


def test_only_the_bound_pages_signals_reach_the_toolbar(qtbot):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    a, b = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("voltage_series")
    explorer.show_page("iv", a)
    explorer.show_page("voltage_series", b)
    explorer.show()
    bar = explorer.toolbar()
    assert bar.bound_key() == "voltage_series"
    a.set_pending(5, "behind")                       # a page behind says something
    assert bar.update_button.text() == "Update plot"
    b.set_pending(2, "front")
    assert bar.update_button.text() == "Update plot (2)"
    explorer.raise_page("iv")                        # a tab switch reads the page that came forward
    assert bar.update_button.text() == "Update plot (5)"
    b.set_pending(9)                                 # and the one that left is not listened to
    assert bar.update_button.text() == "Update plot (5)"


def test_a_page_left_behind_is_no_longer_listened_to_at_all(qtbot):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    a, b = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("voltage_series")
    explorer.show_page("iv", a)
    explorer.show_page("voltage_series", b)
    explorer.show()
    reads = []
    adapter = explorer.adapter("voltage_series")
    real = adapter.pending
    adapter.pending = lambda: (reads.append(1), real())[1]
    a.set_pending(3)                       # a page that is not bound: the toolbar does not even look
    assert reads == []
    b.set_pending(1)
    assert reads != []


def test_forget_unbinds_only_the_named_page(qtbot):
    explorer, page, bar = _explorer(qtbot)
    bar.forget("somewhere_else")
    assert bar.bound_key() == "iv" and bar.update_button.isEnabled()
    bar.forget("iv")
    assert bar.bound_key() is None and not bar.update_button.isEnabled()
    assert bar.update_button.text() == "Update plot"


def test_removing_a_page_forgets_it_before_its_tab_goes(qtbot, monkeypatch):
    explorer, page, bar = _explorer(qtbot)
    seen = []
    real = bar.forget
    monkeypatch.setattr(bar, "forget", lambda key: (seen.append((key, explorer._tabs.indexOf(page))), real(key))[1])
    explorer.remove_page("iv")
    assert seen and seen[0][0] == "iv" and seen[0][1] >= 0       # the tab was still there


def test_reset_figure_wears_the_pages_own_tooltip(qtbot):
    explorer, page, bar = _explorer(qtbot)
    page._appearance_widget.reset_button.setToolTip("Reset: also keeps the colormap.")
    bar.sync()
    assert bar.reset_figure_button.toolTip() == "Reset: also keeps the colormap."


def test_the_update_button_is_amber_only_while_something_is_pending(qtbot):
    explorer, page, bar = _explorer(qtbot)
    assert bar.update_button.styleSheet() == ""
    page.set_pending(1, "x")
    assert "#ffca78" in bar.update_button.styleSheet()
    page.set_pending(0)
    assert bar.update_button.styleSheet() == ""


def test_export_entries_are_ordered_by_kind_and_gated_when_the_menu_opens(qtbot):
    from nanoscope.gui.plot_workflow import ExportEntry
    explorer, page, bar = _explorer(qtbot)
    allowed = {"on": False}
    ran = []
    page.inspector_adapter().exports = lambda: [
        ExportEntry("Copy report", lambda: ran.append("copy"), kind="page"),
        ExportEntry("Export configuration JSON…", lambda: ran.append("json"), kind="recipe"),
        ExportEntry("Export plot data…", lambda: ran.append("data"), kind="data",
                    enabled=lambda: allowed["on"]),
        ExportEntry("Export current figure…", lambda: ran.append("figure"), kind="figure")]
    menu = H.export_menu(explorer)
    texts = [a.text() for a in menu.actions() if not a.isSeparator()][:4]
    assert texts == ["Export current figure…", "Export plot data…", "Export configuration JSON…",
                     "Copy report"]
    assert not next(a for a in menu.actions() if a.text() == "Export plot data…").isEnabled()
    menu.close()
    allowed["on"] = True                              # evaluated each time it is about to show
    menu = H.export_menu(explorer)
    assert next(a for a in menu.actions() if a.text() == "Export plot data…").isEnabled()
    menu.close()
    H.choose_export(explorer, "Copy report")
    assert ran == ["copy"]


def test_run_update_does_nothing_with_nothing_bound_or_while_busy(qtbot):
    explorer, page, bar = _explorer(qtbot)
    page.set_pending(0, busy=True)
    bar.run_update()
    assert page.updates == 0
    page.set_pending(0, busy=False)
    bar.run_update()
    assert page.updates == 1
    bar.bind(None, None)
    bar.run_update()
    assert page.updates == 1


def test_a_page_that_dies_while_bound_leaves_the_toolbar_quiet(qtbot):
    """A toolbar bound by itself (no Explorer to drop the tab for it): the adapter outlives
    the page's C++ object, and every read of it is refused instead of raising."""
    bar = ExplorerToolbar()
    qtbot.addWidget(bar)
    page = H.SyntheticPlotPage("iv")
    bar.bind("iv", page.inspector_adapter())
    assert bar.adapter() is not None and bar.update_action.isEnabled()
    page.deleteLater()
    for _ in range(3):
        QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        QtWidgets.QApplication.processEvents()
    assert bar.adapter() is None
    bar.sync()                                       # must not raise
    bar.run_update()
    assert not bar.update_action.isEnabled()
