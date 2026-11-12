"""The Analysis menu's two grouping devices: the Export submenu and the plain lines inside
*Explore this recording* (pytest-qt, offscreen).

The menu was forty rows in five runs. Export was the longest run (fourteen rows) and none of
its commands has a shortcut, so it is now ONE row that opens a submenu, with three groups
divided by lines. *Explore this recording* (thirteen rows) is split by three plain lines. This
is grouping only: the forty actions keep their labels, shortcuts, tooltips and wiring
(``test_analysis_launcher.py`` pins those and that every action is still reachable). What is
pinned here is the SHAPE the user sees, and the four things that shape could silently lose:

* the row count and the position of the plot surfaces at the top of the menu;
* the Alt+X mnemonic of the submenu, which must not be ambiguous among the menu's own rows;
* the status-bar sentences written for the export commands: they are pushed there by a
  ``hovered`` connection made on every menu under the bar, and a submenu built later than
  that walk would be left out of it;
* the sentence of a row of the menu itself, which a submenu closing must not wipe when the
  pointer leaves the submenu for that row (the same holds for File ▸ Open Recent).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.main_window import MainWindow

# The thirteen rows of *Explore this recording*, in the four runs the plain lines make.
_EXPLORE_GROUPS = (
    ("_act_plots", "_act_distribution_summaries", "_act_comparison_figures", "_act_event_plots"),
    ("_act_event_gallery", "_act_reconcile", "_act_labels"),
    ("_act_signal_conditioning", "_act_psd", "_act_psd_selection"),
    ("_act_fit_levels", "_act_subpop", "_act_show_saved_subpop"),
)


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _rows(menu) -> list:
    return [a for a in menu.actions() if not a.isSeparator()]


def test_explore_this_recording_is_split_by_three_plain_lines_into_4_3_3_3(win):
    actions = win._analysis_menu.actions()
    start = next(i for i, a in enumerate(actions)
                 if a.isSeparator() and a.text() == "Explore this recording")
    end = next(i for i, a in enumerate(actions)
               if a.isSeparator() and a.text() == "Compare recordings")
    runs: list[list] = [[]]
    for act in actions[start + 1:end]:
        if act.isSeparator():
            assert not act.text(), "the lines inside Explore are plain, not headings"
            runs.append([])
        else:
            runs[-1].append(act)
    assert runs == [[getattr(win, a) for a in group] for group in _EXPLORE_GROUPS]


def test_the_menu_is_twenty_seven_rows_and_the_plot_surfaces_stay_near_the_top(win):
    """Forty rows became twenty-seven: the forty actions less the fourteen in Export, plus
    the one row that opens it. Distributions stays the ninth row (index 8), the submenu sits
    after the last Compare recordings row, and Analysis home follows it."""
    rows = _rows(win._analysis_menu)
    export_row = win._analysis_export_menu.menuAction()
    assert len(rows) == 27
    assert rows.index(win._act_plots) == 8
    assert rows.index(export_row) == rows.index(win._act_incubation_study) + 1
    assert rows[rows.index(export_row) + 1] is win._act_analysis_home


def test_export_takes_the_x_mnemonic_and_no_other_row_of_the_menu_does(win):
    export = win._analysis_export_menu
    assert export.title() == "E&xport"
    holders = [a for a in _rows(win._analysis_menu) if "&x" in a.text().lower()]
    assert holders == [export.menuAction()]


def test_the_export_submenu_is_wired_for_the_hover_sentences_like_every_other_menu(qapp, win):
    """``_install_menu_hover_help`` walks the finished menu bar, so the submenu must exist by
    then: it paints tooltips, and a hovered command's sentence reaches the status bar. The
    gesture is driven through ``hovered`` on the popped-up submenu; the arrow-key route is
    covered for the top menus by ``test_wa2_menu_help_reaches_the_status_bar.py``."""
    export = win._analysis_export_menu
    assert export.toolTipsVisible()
    command = win._act_include_hand_made          # live with nothing open, and has a sentence
    assert command in export.actions() and command.toolTip()
    export.popup(QtCore.QPoint(0, 0))
    qapp.processEvents()
    try:
        export.hovered.emit(command)
        assert win.statusBar().currentMessage() == " ".join(command.toolTip().split())
    finally:
        export.close()
        qapp.processEvents()


@pytest.fixture
def popups_stay_open(qapp):
    """``tests/conftest.py`` arms a net that closes any open popup menu within 25 ms of the next
    event-loop turn, so that a stray menu can never block the suite. The test below keeps a
    menu open across several turns on purpose (the withdrawal it checks is queued for the next
    one), so the net is stopped for its duration and started again after it."""
    import conftest
    assert conftest._ensure_popup_closer()
    timer = conftest._POPUP_CLOSER._timer
    timer.stop()
    yield
    timer.start()


def _analysis_export(win):
    """(menu, its submenu, a row inside the submenu, a row of the menu that has a sentence)."""
    return (win._analysis_menu, win._analysis_export_menu, win._act_figure_library,
            win._act_incubation_study)


def _file_open_recent(win):
    file_menu = next(m for m in win.menuBar().findChildren(QtWidgets.QMenu)
                     if m.title() == "&File")
    inside = next(a for a in win._recent_menu.actions() if not a.isSeparator())
    target = next(a for a in file_menu.actions() if a.text() == "Open &Project…")
    return file_menu, win._recent_menu, inside, target


@pytest.mark.parametrize("where", [_analysis_export, _file_open_recent],
                         ids=["analysis_export", "file_open_recent"])
def test_leaving_a_submenu_for_a_row_of_its_menu_keeps_that_rows_sentence(
        qapp, qtbot, win, popups_stay_open, where):
    """On macOS the status bar is the only place a menu row's sentence can be read. Moving the
    pointer from a submenu to a row of the menu that opened it shows that row's sentence at
    once (``hovered``) and only then hides the submenu, whose ``aboutToHide`` queues the
    withdrawal of the menu sentence: that queued call used to remove the sentence of the row
    the pointer was on, leaving the row highlighted and the bar empty until the pointer
    moved. The sentence must stay while the row is highlighted, and go when the menu closes."""
    top, sub, inside, target = where(win)
    sentence = " ".join(target.toolTip().split())
    assert sentence and win._menu_action_help(target) == sentence, (
        "the row this test moves to must carry a written sentence")
    bar = win.statusBar()
    bar.clearMessage()
    top.popup(QtCore.QPoint(0, 0))
    qapp.processEvents()
    try:
        top.setActiveAction(sub.menuAction())              # the submenu opens
        assert sub.isVisible()
        sub.setActiveAction(inside)                         # the pointer is on one of its rows
        top.setActiveAction(target)                         # ... and moves to a row of the menu
        assert not sub.isVisible() and top.isVisible()
        qtbot.wait(200)                                     # the queued withdrawal has run
        assert top.isVisible() and top.activeAction() is target
        assert bar.currentMessage() == sentence, (
            f"the row is highlighted but the status bar shows {bar.currentMessage()!r}")

        top.close()                                         # the menu really closes
        qtbot.wait(200)
        assert bar.currentMessage() == ""
    finally:
        top.close()
        qapp.processEvents()
