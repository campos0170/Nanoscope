"""Phase 1, shell group (G1): the three keys the shell owns, on every page, by real key events.

⌘1 and ⌘2 are the inspector's two tabs, ⌘Return (and the keypad ⌘Enter) is Update plot, ⌘3 and
⌘4 are unbound, and the page's own Ctrl+1 to 4 reach nobody. Each key is sent through the QWindow
of the Explorer, after the window is active, as a user's key would be. Synthetic pages;
offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from . import p1_hosting as H

K = QtCore.Qt.Key
PAGES = ("distributions", "distribution_summaries", "comparison_figures", "event_waveforms",
         "noise", "iv", "voltage_series", "subpopulations")


def _hosted(qtbot, key):
    page = H.page_builders()[key]()
    return H.hosted(qtbot, page, key), page


def _count_updates(explorer, key):
    calls = []
    adapter = explorer.adapter(key)
    adapter.update = lambda: calls.append(key)
    return calls


@pytest.mark.parametrize("key", PAGES)
def test_cmd_1_and_cmd_2_are_the_inspector_tabs_on_every_page(qtbot, key):
    explorer, page = _hosted(qtbot, key)
    host = explorer.inspector()
    host.set_tab(0)
    pane = host.pane(key)
    assert pane.currentIndex() == 0
    H.key(explorer, K.Key_2, H.CTRL)
    assert host.current_tab() == 1 and pane.currentIndex() == 1 and pane.currentWidget() is pane.format_pane
    H.key(explorer, K.Key_1, H.CTRL)
    assert host.current_tab() == 0 and pane.currentIndex() == 0 and pane.currentWidget() is pane.data_pane


@pytest.mark.parametrize("key", PAGES)
def test_cmd_3_and_cmd_4_are_unbound_and_the_pages_own_ctrl_digits_reach_nobody(qtbot, key):
    explorer, page = _hosted(qtbot, key)
    host = explorer.inspector()
    host.set_tab(0)
    bar = getattr(page, "_workflow", None)
    section = bar.sections.currentText() if bar is not None else None
    flags = {k: host.pane(key).data_pane.group(k).is_expanded()
             for k in host.pane(key).data_pane.group_keys()}
    calls = _count_updates(explorer, key)
    for digit in (K.Key_3, K.Key_4, K.Key_1, K.Key_2):
        host.set_tab(0)
        H.key(explorer, digit, H.CTRL)
        if digit in (K.Key_3, K.Key_4):
            assert host.current_tab() == 0, f"Ctrl+{digit} did something"
    host.set_tab(0)
    assert section is None or bar.sections.currentText() == section, "the hidden Section combo moved"
    assert {k: host.pane(key).data_pane.group(k).is_expanded()
            for k in host.pane(key).data_pane.group_keys()} == flags
    assert calls == [], "no digit key ran the update"


@pytest.mark.parametrize("key", PAGES)
def test_cmd_return_is_update_plot_once_and_the_keypad_key_too(qtbot, key):
    explorer, page = _hosted(qtbot, key)
    calls = _count_updates(explorer, key)
    H.key(explorer, K.Key_Return, H.CTRL)
    assert calls == [key]
    H.key(explorer, K.Key_Enter, H.CTRL)
    assert calls == [key, key]


def test_cmd_return_reaches_only_the_page_in_front(qtbot):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    a, b = H.SyntheticPlotPage("iv"), H.SyntheticPlotPage("voltage_series")
    explorer.add_page("iv", a)
    explorer.add_page("voltage_series", b)
    explorer.show()
    qtbot.waitExposed(explorer)
    explorer.raise_page("iv")
    H.key(explorer, K.Key_Return, H.CTRL)
    assert (a.updates, b.updates) == (1, 0)
    explorer.raise_page("voltage_series")
    H.key(explorer, K.Key_Return, H.CTRL)
    assert (a.updates, b.updates) == (1, 1)


def test_the_keys_do_nothing_on_home(qtbot):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    page = H.SyntheticPlotPage("iv")
    explorer.add_page("iv", page)
    explorer.show()
    qtbot.waitExposed(explorer)
    explorer.show_home()
    explorer.inspector().set_tab(0)
    H.key(explorer, K.Key_2, H.CTRL)
    H.key(explorer, K.Key_Return, H.CTRL)
    assert explorer.inspector().current_tab() == 0 and page.updates == 0
    assert not explorer.inspector().isVisible()


def test_cmd_return_while_the_page_is_busy_does_nothing(qtbot):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", page)
    explorer.show()
    qtbot.waitExposed(explorer)
    page.set_pending(1, "a gather", busy=True)
    H.key(explorer, K.Key_Return, H.CTRL)
    assert page.updates == 0
    page.set_pending(0, busy=False)
    H.key(explorer, K.Key_Return, H.CTRL)
    assert page.updates == 1


def test_cmd_return_with_a_modal_dialog_open_does_nothing(qtbot):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", page)
    explorer.show()
    qtbot.waitExposed(explorer)
    dialog = QtWidgets.QDialog(explorer)
    dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    dialog.show()
    qtbot.waitExposed(dialog)
    try:
        QtTest.QTest.keyClick(H.window_handle(dialog), K.Key_Return, H.CTRL)
        QtTest.QTest.keyClick(H.window_handle(explorer), K.Key_Return, H.CTRL)
        QtWidgets.QApplication.processEvents()
        assert page.updates == 0
    finally:
        dialog.hide()
        dialog.deleteLater()
    H.key(explorer, K.Key_Return, H.CTRL)
    assert page.updates == 1, "the key is back once the dialog is closed"


def test_cmd_return_commits_a_box_that_has_not_lost_focus_before_the_page_updates(qtbot):
    """Distributions' Fit range boxes commit on ``editingFinished``. A value typed and not
    yet committed must be in the plot ⌘Return draws, so the shell takes the focus off the box
    first and only then calls the page's verb."""
    explorer, page = _hosted(qtbot, "distributions")
    order = []
    page._fit_lo.editingFinished.connect(lambda: order.append(("commit", page._fit_lo.text())))
    explorer.adapter("distributions").update = lambda: order.append(("update", page._fit_lo.text()))
    explorer.reveal("plot")
    page._fit_lo.setFocus()
    qtbot.waitUntil(page._fit_lo.hasFocus)
    QtTest.QTest.keyClicks(page._fit_lo, "0.25")
    assert order == [], "typed, not committed"
    H.key(explorer, K.Key_Return, H.CTRL)
    assert order == [("commit", "0.25"), ("update", "0.25")]


def test_a_tab_switch_moves_the_keys_to_the_page_now_in_front(qtbot):
    explorer = H.AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1280, 820)
    pages = {k: H.SyntheticPlotPage(k) for k in ("iv", "noise", "voltage_series")}
    for key, page in pages.items():
        explorer.add_page(key, page)
    explorer.show()
    qtbot.waitExposed(explorer)
    tally = {k: 0 for k in pages}
    for key in pages:
        explorer.raise_page(key)
        H.key(explorer, K.Key_Return, H.CTRL)
        tally[key] += 1
        assert {k: p.updates for k, p in pages.items()} == {**{k: 1 for k in tally if tally[k]},
                                                            **{k: 0 for k in tally if not tally[k]}}
