"""Phase 1, shell group (G1): Style ▾ in full.

``This page`` and ``Every open page`` are two submenus over ``PlotAppearanceWidget.apply_look``;
a look is ONE Undo step per page; "Undo style on N pages" answers for the batch only while
every page it styled is still at the step it left; "Keep text styles I set by hand" is
remembered; saved looks are listed, saved from the menu and managed. Every pick is a real
click on a real menu. Synthetic pages in a real Explorer; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.gui.analysis_inspector import InspectorSettings
from nanoscope.gui.figure_text import TextStyle
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS, PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plot_workflow import PendingState

from . import p1_hosting as H

NATURE = "7 pt Arial · Nature, Elsevier"
IEEE = "9 pt Times New Roman · IEEE"
SLIDES = "Slides · 18 pt Arial"
NOTE = "Applies to figures open now. A figure opened later starts plain."


@pytest.fixture(autouse=True)
def _no_saved_looks():
    """Saved looks live in QSettings outside ``windows/``: start and end every test empty."""
    QtCore.QSettings().remove(PlotAppearanceWidget._PRESETS_KEY)
    yield
    QtCore.QSettings().remove(PlotAppearanceWidget._PRESETS_KEY)


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    """A look that fails or is applied opens no dialog: any message box fails the test."""
    def refuse(*_a, **_k):
        pytest.fail("a message box opened")
    for name in ("exec", "warning", "critical", "information"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, refuse)


def _explorer(qtbot, keys=("iv", "voltage_series", "noise"), size=(1280, 820)):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(*size)
    pages = {}
    for key in keys:
        pages[key] = H.SyntheticPlotPage(key)
        explorer.add_page(key, pages[key])
    explorer.show()
    qtbot.waitExposed(explorer)
    if keys:
        explorer.raise_page(keys[0])
    QtWidgets.QApplication.processEvents()
    return explorer, pages


def _widget(pages, key):
    return pages[key]._appearance_widget


def _edit(widget, **fields):
    """An edit made in the page's own controls: one Undo step, announced to the page."""
    widget.apply_appearance(PlotAppearance.from_mapping({**widget.state().to_dict(), **fields}))


def _open(explorer):
    explorer.toolbar().style_button.showMenu()
    QtWidgets.QApplication.processEvents()
    return explorer.toolbar().style_button.menu()


def _action(menu, text):
    return next(a for a in menu.actions() if a.text() == text)


def _click_item(menu, action):
    assert action.isEnabled(), f"{action.text()!r} is disabled"
    QtTest.QTest.mouseClick(menu, H.L, H.NOMOD, menu.actionGeometry(action).center())
    QtWidgets.QApplication.processEvents()


def _scope_menu(explorer, scope):
    """The submenu of ``This page`` / ``Every open page``, from the owner that keeps it alive
    (a wrapper got from ``QAction.menu()`` and dropped takes the C++ menu with it)."""
    return explorer._style_menu.scope_menu({"This page": "page", "Every open page": "all"}[scope])


def _pick(explorer, scope, text):
    """Style ▾ ▸ ``scope`` ▸ ``text``, as two real clicks."""
    menu = _open(explorer)
    sub = _scope_menu(explorer, scope)
    sub.popup(menu.mapToGlobal(menu.actionGeometry(sub.menuAction()).topRight()))
    _click_item(sub, _action(sub, text))
    menu.hide()


def _pick_top(explorer, text):
    menu = _open(explorer)
    _click_item(menu, _action(menu, text))
    menu.hide()


def _front_status(explorer, key):
    return explorer.adapter(key).status_bar().currentMessage()


def _texts(menu):
    return [a.text() for a in menu.actions() if a.isVisible() and not a.isSeparator() and a.text()]


# -- the menu ----------------------------------------------------------------------------------

def test_the_menu_is_two_scopes_over_the_same_looks_and_the_fixed_items(qtbot):
    explorer, _pages = _explorer(qtbot)
    menu = _open(explorer)
    assert not explorer._style_menu.scope_menu("selected").menuAction().isVisible()
    assert not explorer._style_menu._default_menu.menuAction().isVisible()
    assert _texts(menu) == ["This page", "Every open page", "Keep text styles I set by hand",
                            "Save this page's look as…", "Manage saved looks…", "Undo style", NOTE]
    assert not _action(menu, NOTE).isEnabled() and not _action(menu, "Undo style").isEnabled()
    keep = _action(menu, "Keep text styles I set by hand")
    assert keep.isCheckable() and keep.isChecked(), "ticked by default"
    page, everywhere = _scope_menu(explorer, "This page"), _scope_menu(explorer, "Every open page")
    assert _texts(page) == _texts(everywhere) == list(_BUILTIN_PRESETS) + ["None saved yet"]
    assert [a.text() for a in page.actions() if a.isSeparator()] , "a heading over the built-in looks"
    for name in (NATURE, IEEE, SLIDES, "8 pt Arial · PLOS, Springer, AIP"):
        assert name in _texts(page)
    menu.hide()


def test_the_submenus_are_not_built_until_the_menu_is_first_shown(qtbot):
    explorer, _pages = _explorer(qtbot)
    submenu = explorer._style_menu.scope_menu("page")
    assert submenu.actions() == []
    _open(explorer).hide()
    assert submenu.actions()


# -- applying ------------------------------------------------------------------------------------

def test_every_open_page_through_the_menu_is_one_step_per_page_and_the_front_page_says_so(qtbot):
    explorer, pages = _explorer(qtbot)
    explorer.raise_page("voltage_series")                    # a page behind: iv, noise
    before = {k: explorer.adapter(k).history().mark for k in pages}
    _pick(explorer, "Every open page", NATURE)
    for key, page in pages.items():
        widget = page._appearance_widget
        assert widget.state().title_size == 7 and widget.state().font_family == "Arial"
        assert explorer.adapter(key).history().mark == before[key] + 1
        assert explorer.adapter(key).history().undo_text == f"Undo: {NATURE} preset"
    assert _front_status(explorer, "voltage_series") == (
        f"{NATURE} applied to 3 pages; Undo is per page.")
    # a second application of the same look moves nothing and says so
    _pick(explorer, "Every open page", NATURE)
    assert all(explorer.adapter(k).history().mark == before[k] + 1 for k in pages)
    assert "nothing to change" in _front_status(explorer, "voltage_series")


def test_this_page_styles_the_front_page_only_and_says_so(qtbot):
    explorer, pages = _explorer(qtbot)
    _pick(explorer, "This page", IEEE)
    assert _widget(pages, "iv").state().font_family == "Times New Roman"
    assert _widget(pages, "voltage_series").state().font_family != "Times New Roman"
    assert _widget(pages, "noise").history_mark() == 0
    assert _front_status(explorer, "iv") == f"{IEEE} applied to this page; Undo is per page."


def test_the_explorers_apply_look_names_a_shipped_look_and_calls_a_stranger_the_look(qtbot):
    explorer, _pages = _explorer(qtbot)
    explorer.apply_look(_BUILTIN_PRESETS[NATURE], scope="all")
    assert _front_status(explorer, "iv") == f"{NATURE} applied to 3 pages; Undo is per page."
    explorer.apply_look(PlotAppearance(line_width=3.3), scope="page")
    assert _front_status(explorer, "iv") == "The look applied to this page; Undo is per page."


def test_a_page_with_no_status_line_is_reported_in_the_explorers_own(qtbot):
    explorer, _pages = _explorer(qtbot)
    explorer.adapter("iv").status_bar = lambda: None
    _pick(explorer, "This page", NATURE)
    assert explorer.statusBar().currentMessage() == f"{NATURE} applied to this page; Undo is per page."


def test_a_look_carries_no_typed_word(qtbot):
    explorer, pages = _explorer(qtbot, keys=("iv", "voltage_series"))
    _edit(_widget(pages, "iv"), title="My own title")
    other = _widget(pages, "voltage_series")
    _edit(other, title="Pore 3", line_width=2.5)
    assert other.save_user_preset("Slimline")
    _pick(explorer, "Every open page", "Slimline")
    assert _widget(pages, "iv").state().title == "My own title", "the look did not stamp 'Pore 3'"
    assert _widget(pages, "iv").state().line_width == 2.5, "but it did carry the line width"


def test_every_page_kind_takes_a_look_while_it_is_behind_another_tab_and_records_one_step(qtbot):
    """The probe the design asked for, kept as a test: all eight real pages in one Explorer,
    seven of them hidden, one look to every open page. Each hidden page redraws with the
    look and has exactly one new step on its own history."""
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    pages = {}
    for key, build in H.page_builders().items():
        pages[key] = build()
        explorer.add_page(key, pages[key])
    explorer.show()
    qtbot.waitExposed(explorer)
    explorer.raise_page("distributions")
    QtWidgets.QApplication.processEvents()
    assert sorted(explorer.page_ids()) == sorted(pages)
    before = {k: explorer.adapter(k).history().mark for k in pages}
    result = explorer.apply_look(_BUILTIN_PRESETS[NATURE], scope="all")
    assert result.failed == () and sorted(result.applied) == sorted(pages)
    assert result.applied[0] == "distributions", "the front page first"
    for key in pages:
        adapter = explorer.adapter(key)
        assert adapter.history().mark == before[key] + 1, key
        assert adapter.format.appearance.state().title_size == 7, key
        assert adapter.history().can_undo, key
    assert sum(1 for p in pages.values() if p.isVisible()) == 1, "seven pages were behind"
    assert explorer._style.undo_last_batch() == 8
    assert all(explorer.adapter(k).format.appearance.state().title_size is None for k in pages)


def test_a_page_that_refuses_is_named_with_its_reason_takes_no_step_and_opens_no_dialog(qtbot):
    explorer, pages = _explorer(qtbot)

    def refuse(look, keep_hand):
        raise RuntimeError("the figure will not render")
    explorer.adapter("voltage_series").apply_look = refuse
    _pick(explorer, "Every open page", NATURE)
    assert _widget(pages, "voltage_series").history_mark() == 0
    assert _widget(pages, "iv").history_mark() == 1 and _widget(pages, "noise").history_mark() == 1
    said = _front_status(explorer, "iv")
    assert said == (f"{NATURE} applied to 2 pages; Undo is per page. "
                    "Not applied to Voltage dependence (the figure will not render).")
    # the batch holds the two pages that took the look, and only those
    assert [k for k, _ in explorer._style.last_batch().pages] == ["iv", "noise"]


def test_when_the_only_page_refuses_the_sentence_says_it_was_not_applied(qtbot):
    explorer, pages = _explorer(qtbot, keys=("iv",))

    def refuse(look, keep_hand):
        raise ValueError("no figure to draw")
    explorer.adapter("iv").apply_look = refuse
    _pick(explorer, "This page", NATURE)
    assert _front_status(explorer, "iv") == f"{NATURE} was not applied to I-V / conductance (no figure to draw)."
    assert explorer._style.last_batch() is None and _widget(pages, "iv").history_mark() == 0


def test_a_page_in_the_middle_of_a_gather_still_takes_the_look(qtbot):
    explorer, pages = _explorer(qtbot)
    explorer.adapter("iv").pending = lambda: PendingState(1, ("gathering",), True)
    explorer.toolbar().sync()
    assert not explorer.toolbar().update_action.isEnabled()
    _pick(explorer, "Every open page", NATURE)
    assert _widget(pages, "iv").history_mark() == 1 and _widget(pages, "iv").state().title_size == 7
    assert not explorer.toolbar().update_action.isEnabled(), "applying a look did not end the busy state"


def test_a_figure_opened_later_starts_plain(qtbot):
    explorer, pages = _explorer(qtbot, keys=("iv", "voltage_series"))
    _pick(explorer, "Every open page", NATURE)
    late = H.SyntheticPlotPage("noise")
    explorer.add_page("noise", late)
    assert late._appearance_widget.state().title_size is None
    assert late._appearance_widget.history_mark() == 0
    assert "noise" not in [k for k, _ in explorer._style.last_batch().pages]


def test_from_home_every_open_page_still_works_in_tab_order_and_this_page_is_off(qtbot):
    explorer, pages = _explorer(qtbot)
    explorer.show_home()
    menu = _open(explorer)
    assert not _action(menu, "This page").isEnabled()
    assert _action(menu, "Every open page").isEnabled()
    assert not _action(menu, "Save this page's look as…").isEnabled()
    menu.hide()
    _pick(explorer, "Every open page", NATURE)
    assert all(p._appearance_widget.history_mark() == 1 for p in pages.values())
    assert explorer.statusBar().currentMessage() == f"{NATURE} applied to 3 pages; Undo is per page."
    assert explorer._style.last_batch() is not None and \
        [k for k, _ in explorer._style.last_batch().pages] == ["iv", "voltage_series", "noise"]


def test_with_no_page_open_the_menu_is_safe_and_every_open_page_is_off(qtbot):
    explorer, _pages = _explorer(qtbot, keys=())
    menu = _open(explorer)
    assert not _action(menu, "This page").isEnabled()
    assert not _action(menu, "Every open page").isEnabled()
    assert not _action(menu, "Manage saved looks…").isEnabled()
    menu.hide()
    assert explorer.apply_look(_BUILTIN_PRESETS[NATURE], scope="all").applied == ()


# -- Undo style on N pages ---------------------------------------------------------------------

def test_undo_style_on_n_pages_reverses_exactly_those_steps_with_a_real_click(qtbot):
    explorer, pages = _explorer(qtbot)
    _pick(explorer, "Every open page", NATURE)
    menu = _open(explorer)
    assert _action(menu, "Undo style on 3 pages").isEnabled()
    menu.hide()
    _pick_top(explorer, "Undo style on 3 pages")
    for page in pages.values():
        assert page._appearance_widget.state().title_size is None
        assert explorer.adapter(page._key).history().can_redo
    menu = _open(explorer)
    assert _action(menu, "Undo style").isEnabled() is False, "the batch is used up"
    menu.hide()
    assert explorer._style.last_batch() is None


def test_the_undo_item_is_one_page_wide_when_one_page_took_the_look(qtbot):
    explorer, pages = _explorer(qtbot)
    _pick(explorer, "This page", NATURE)
    menu = _open(explorer)
    assert _action(menu, "Undo style on 1 page").isEnabled()
    menu.hide()


def test_undo_style_is_disabled_once_any_page_moves_on(qtbot):
    explorer, pages = _explorer(qtbot)
    _pick(explorer, "Every open page", NATURE)
    pages["noise"]._appearance_widget.title.textEdited.emit("moved on")
    pages["noise"]._appearance_widget._close_burst()
    menu = _open(explorer)
    assert _action(menu, "Undo style").isEnabled() is False
    menu.hide()
    assert explorer._style.last_batch() is None
    assert explorer._style.undo_last_batch() == 0
    assert _widget(pages, "iv").state().title_size == 7, "no page was undone behind the user's back"


@pytest.mark.parametrize("announces", [True, False], ids=["page announces history", "page is silent"])
def test_undo_then_a_different_edit_is_a_move_on_even_at_the_same_depth(qtbot, announces):
    """The mark alone cannot tell this: Undo puts the depth back, the new edit takes it
    forward again to the same number, and the batch is no longer what the page last did. A page
    that announces its history ends the batch at the Undo; a silent one (the fallback's
    Distributions until it has a signal) is told apart by the value of its history when the
    menu opens."""
    explorer, pages = _explorer(qtbot, keys=("iv", "voltage_series"))
    if not announces:
        explorer.adapter("iv").history_changed = None
    _pick(explorer, "Every open page", NATURE)
    widget = _widget(pages, "iv")
    marks = explorer.adapter("iv").history().mark
    H.press_undo(explorer)                                   # the front page's own Undo
    assert widget.state().title_size is None
    _edit(widget, title="Different")
    assert explorer.adapter("iv").history().mark == marks, "same depth as right after the batch"
    menu = _open(explorer)
    assert _action(menu, "Undo style").isEnabled() is False
    menu.hide()
    assert explorer._style.undo_last_batch() == 0
    assert _widget(pages, "voltage_series").state().title_size == 7


def test_a_second_look_replaces_the_batch_and_an_idempotent_one_keeps_it(qtbot):
    explorer, pages = _explorer(qtbot)
    _pick(explorer, "Every open page", NATURE)
    first = explorer._style.last_batch()
    _pick(explorer, "Every open page", NATURE)               # nothing moved
    assert explorer._style.last_batch() is first
    _pick(explorer, "Every open page", IEEE)
    assert explorer._style.last_batch().name == IEEE
    assert explorer._style.undo_last_batch() == 3
    assert all(p._appearance_widget.state().title_size == 7 for p in pages.values()), \
        "back to the Nature look, one step each"


def test_batch_undo_never_reaches_a_replacement_page_with_matching_history(qtbot):
    explorer, pages = _explorer(qtbot, keys=("iv", "noise"))
    _pick(explorer, "Every open page", NATURE)
    old_history = explorer.adapter("iv").history()
    replacement = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", replacement)           # retires the original page and pane
    adapter = explorer.adapter("iv")
    assert adapter.apply_look(_BUILTIN_PRESETS[NATURE], True)
    new_history = adapter.history()
    assert (new_history.mark, new_history.undo_text, new_history.can_redo) == (
        old_history.mark, old_history.undo_text, old_history.can_redo)
    before = {key: explorer.adapter(key).format.appearance.state() for key in explorer.page_ids()}
    menu = _open(explorer)
    assert not _action(menu, "Undo style").isEnabled()
    menu.hide()
    assert explorer._style.undo_last_batch() == 0
    assert {key: explorer.adapter(key).format.appearance.state()
            for key in explorer.page_ids()} == before
    assert replacement is not pages["iv"]


# -- keep text styles I set by hand ----------------------------------------------------------------

def test_keep_hand_is_on_by_default_and_the_checkbox_turns_it_off_and_is_remembered(qtbot):
    explorer, pages = _explorer(qtbot)
    widget = _widget(pages, "iv")
    widget.edit_text_style("title", TextStyle(bold=True, size_pt=16))
    widget._close_burst()
    _pick(explorer, "This page", NATURE)
    style = widget.state().text_style("title")
    assert (style.bold, style.size_pt) == (True, 16.0), "a hand-set title style survives"
    assert widget.state().axis_size == 7, "the figure-wide fields still change"
    assert widget.undo()
    widget.edit_text_style("title", TextStyle(bold=True, size_pt=16))
    widget._close_burst()
    _pick_top(explorer, "Keep text styles I set by hand")     # untick
    assert InspectorSettings().keep_hand is False
    _pick(explorer, "This page", NATURE)
    style = widget.state().text_style("title")
    assert (style.bold, style.size_pt) == (None, None), "unticked: today's replacing behaviour"
    menu = _open(explorer)
    assert not _action(menu, "Keep text styles I set by hand").isChecked()
    menu.hide()


# -- saved looks -------------------------------------------------------------------------------

def test_save_this_pages_look_from_the_menu_lists_it_in_both_scopes_and_applies_it(
        qtbot, monkeypatch):
    explorer, pages = _explorer(qtbot)
    source = _widget(pages, "iv")
    _edit(source, title_size=11.0, line_width=2.5, title="Typed")
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *_a, **_k: ("My look & more", True)))
    _pick_top(explorer, "Save this page's look as…")
    states = PlotAppearanceWidget.user_preset_states()
    assert list(states) == ["My look & more"]
    assert "title" not in states["My look & more"] or states["My look & more"]["title"] is None, \
        "the words stay with the figure they were typed on"
    assert states["My look & more"]["line_width"] == 2.5
    assert explorer.statusBar().currentMessage() == "Saved the look ‘My look & more’ on this computer."
    menu = _open(explorer)
    page, everywhere = _scope_menu(explorer, "This page"), _scope_menu(explorer, "Every open page")
    assert "My look && more" in _texts(page) and "My look && more" in _texts(everywhere)
    assert "None saved yet" not in _texts(page)
    menu.hide()
    _pick(explorer, "Every open page", "My look && more")
    assert _widget(pages, "noise").state().line_width == 2.5
    assert _widget(pages, "noise").state().title is None


def test_saving_over_an_existing_look_asks_first_and_no_replaces_nothing(qtbot, monkeypatch):
    explorer, pages = _explorer(qtbot)
    widget = _widget(pages, "iv")
    assert widget.save_user_preset("Mine")
    _edit(widget, line_width=4.0)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *_a, **_k: ("Mine", True)))
    asked = []
    answers = iter((QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.Yes))

    def question(*args, **_kw):
        asked.append(args[1])
        return next(answers)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(question))
    _pick_top(explorer, "Save this page's look as…")
    assert PlotAppearanceWidget.user_preset_states()["Mine"].get("line_width") != 4.0
    _pick_top(explorer, "Save this page's look as…")
    assert PlotAppearanceWidget.user_preset_states()["Mine"]["line_width"] == 4.0
    assert asked == ["Replace saved look?", "Replace saved look?"]


@pytest.mark.parametrize("scope", ["This page", "Every open page"])
def test_reopening_a_saved_look_menu_applies_its_overwritten_values(qtbot, monkeypatch, scope):
    explorer, pages = _explorer(qtbot, keys=("iv", "noise"))
    source = _widget(pages, "iv")
    _edit(source, line_width=1.5)
    assert source.save_user_preset("Mine")
    _open(explorer).hide()                         # both submenus now capture the old look
    _edit(source, line_width=4.0)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *_a, **_k: ("Mine", True)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Yes))
    _pick_top(explorer, "Save this page's look as…")
    assert PlotAppearanceWidget.user_preset_states()["Mine"]["line_width"] == 4.0
    explorer.raise_page("noise")
    _pick(explorer, scope, "Mine")                 # reopen, then actually choose the saved item
    assert _widget(pages, "noise").state().line_width == 4.0


def test_a_cancelled_or_empty_name_saves_nothing(qtbot, monkeypatch):
    explorer, _pages = _explorer(qtbot)
    for reply in (("x", False), ("   ", True)):
        monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                            staticmethod(lambda *_a, _r=reply, **_k: _r))
        _pick_top(explorer, "Save this page's look as…")
    assert PlotAppearanceWidget.user_preset_states() == {}


def test_manage_saved_looks_lists_them_and_delete_removes_one_after_a_confirmation(
        qtbot, monkeypatch):
    explorer, pages = _explorer(qtbot)
    widget = _widget(pages, "iv")
    assert widget.save_user_preset("Zebra") and widget.save_user_preset("apple")
    _pick_top(explorer, "Manage saved looks…")
    dialog = explorer.findChild(QtWidgets.QDialog, "analysisManageLooks")
    assert dialog is not None and dialog.isVisible()
    listing = dialog.findChild(QtWidgets.QListWidget, "analysisManageLooksList")
    assert [listing.item(i).text() for i in range(listing.count())] == ["apple", "Zebra"]
    delete = dialog.findChild(QtWidgets.QPushButton, "analysisManageLooksDelete")
    listing.setCurrentRow(1)
    asked = []

    def question(*args, **_kw):
        asked.append(args[1])
        return QtWidgets.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(question))
    H.click(delete)
    assert asked == ["Delete saved look?"]
    assert list(PlotAppearanceWidget.user_preset_states()) == ["apple"]
    assert [listing.item(i).text() for i in range(listing.count())] == ["apple"]
    menu = _open(explorer)
    assert "Zebra" not in _texts(_scope_menu(explorer, "This page"))
    menu.hide()
    dialog.hide()
    _pick_top(explorer, "Manage saved looks…")               # the same window comes back
    assert explorer.findChildren(QtWidgets.QDialog, "analysisManageLooks") == [dialog]
