"""Phase 1 Foundation: ``StyleService`` and the stub ``StyleMenu`` (Style ▾).

A look applied to a page is ONE Undo step on that page's own history; the loop over pages is
the service's. Synthetic pages in a real Explorer; the menu is clicked, not triggered.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtTest, QtWidgets

from nanoscope.gui.figure_text import TextStyle
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS, PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plot_styles import StyleMenu, StyleService

from . import p1_hosting as H

NATURE = "7 pt Arial · Nature, Elsevier"
IEEE = "9 pt Times New Roman · IEEE"


def _explorer(qtbot, keys=("iv", "voltage_series", "noise")):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    pages = {}
    for key in keys:
        pages[key] = H.SyntheticPlotPage(key)
        explorer.add_page(key, pages[key])
    explorer.show()
    qtbot.waitExposed(explorer)
    explorer.raise_page(keys[0])
    return explorer, pages


def test_built_in_and_saved_looks(qtbot):
    explorer, pages = _explorer(qtbot)
    service = explorer._style
    assert [n for n, _ in service.built_in_looks()] == list(_BUILTIN_PRESETS)
    assert service.saved_looks() == []
    widget = pages["iv"]._appearance_widget
    widget.apply_look(_BUILTIN_PRESETS[IEEE])
    assert widget.save_user_preset("Zebra") and widget.save_user_preset("apple")
    assert [n for n, _ in service.saved_looks()] == ["apple", "Zebra"]       # case-insensitive
    assert service.saved_looks()[0][1].font_family == "Times New Roman"


def test_scope_page_styles_the_front_page_and_one_step_each(qtbot):
    explorer, pages = _explorer(qtbot)
    look = _BUILTIN_PRESETS[NATURE]
    result = explorer._style.apply(NATURE, look, scope="page", keep_hand=True)
    assert result.applied == ("iv",) and result.failed == () and result.name == NATURE
    assert pages["iv"]._appearance_widget.state().title_size == 7
    assert pages["voltage_series"]._appearance_widget.state().title_size is None
    assert pages["iv"]._appearance_widget.history_mark() == 1


def test_scope_all_styles_every_page_front_first_one_step_each_and_a_second_run_is_idempotent(qtbot):
    explorer, pages = _explorer(qtbot)
    explorer.raise_page("voltage_series")
    look = _BUILTIN_PRESETS[NATURE]
    result = explorer.apply_look(look, scope="all")
    assert result.applied == ("voltage_series", "iv", "noise")      # front, then tab order
    for page in pages.values():
        assert page._appearance_widget.history_mark() == 1
        assert page._appearance_widget.state().title_size == 7
    again = explorer.apply_look(look, scope="all")
    assert again.applied == () and again.failed == ()               # already that look: no step
    assert all(p._appearance_widget.history_mark() == 1 for p in pages.values())
    explorer.show_home()                                            # from Home: tab order
    assert explorer.apply_look(_BUILTIN_PRESETS[IEEE], scope="all").applied == ("iv", "voltage_series", "noise")
    assert explorer.apply_look(_BUILTIN_PRESETS[IEEE], scope="page").applied == ()


def test_a_page_that_refuses_is_listed_and_the_others_still_get_the_look(qtbot):
    explorer, pages = _explorer(qtbot)

    def refuse(look, keep_hand):
        raise RuntimeError("the figure will not render")
    explorer.adapter("voltage_series").apply_look = refuse
    result = explorer.apply_look(_BUILTIN_PRESETS[NATURE], scope="all")
    assert result.applied == ("iv", "noise")
    assert result.failed == (("voltage_series", "the figure will not render"),)
    assert pages["voltage_series"]._appearance_widget.history_mark() == 0


def test_undo_last_batch_reverses_exactly_those_steps_and_only_while_intact(qtbot):
    explorer, pages = _explorer(qtbot)
    service = explorer._style
    explorer.apply_look(_BUILTIN_PRESETS[NATURE], scope="all")
    batch = service.last_batch()
    assert batch is not None and [k for k, _ in batch.pages] == ["iv", "voltage_series", "noise"]
    assert all(mark == 1 for _, mark in batch.pages)
    assert service.undo_last_batch() == 3
    assert all(p._appearance_widget.state().title_size is None for p in pages.values())
    assert service.last_batch() is None and service.undo_last_batch() == 0
    explorer.apply_look(_BUILTIN_PRESETS[NATURE], scope="all")
    pages["noise"]._appearance_widget.title.textEdited.emit("moved on")   # one page moves on
    assert service.undo_last_batch() == 0                                  # ...so none is undone
    assert pages["iv"]._appearance_widget.state().title_size == 7


def test_the_menu_applies_a_look_to_the_front_page_and_honours_keep_hand(qtbot):
    explorer, pages = _explorer(qtbot)
    widget = pages["iv"]._appearance_widget
    widget.edit_text_style("title", TextStyle(bold=True, size_pt=16))
    widget._close_burst()
    bar = explorer.toolbar()
    for keep in (True, False):
        explorer.inspector().settings().keep_hand = keep
        bar.style_button.showMenu()
        menu = bar.style_button.menu()
        this_page = next(a for a in menu.actions() if a.text() == "This page")
        sub = this_page.menu()
        sub.popup(menu.mapToGlobal(menu.actionGeometry(this_page).topRight()))
        action = next(a for a in sub.actions() if a.text() == NATURE)
        QtTest.QTest.mouseClick(sub, H.L, H.NOMOD, sub.actionGeometry(action).center())
        QtWidgets.QApplication.processEvents()
        style = widget.state().text_style("title")
        assert (style.bold, style.size_pt) == ((True, 16.0) if keep else (None, None))
        assert widget.undo()
        widget.edit_text_style("title", TextStyle(bold=True, size_pt=16))
        widget._close_burst()


def test_the_style_menu_offers_this_page_only_while_a_page_is_in_front(qtbot):
    explorer, pages = _explorer(qtbot)
    menu = explorer.toolbar().style_button.menu()
    menu.aboutToShow.emit()
    assert next(a for a in menu.actions() if a.text() == "This page").isEnabled()
    explorer.show_home()
    menu.aboutToShow.emit()
    assert not next(a for a in menu.actions() if a.text() == "This page").isEnabled()
    assert isinstance(explorer._style_menu, StyleMenu) and isinstance(explorer._style, StyleService)
