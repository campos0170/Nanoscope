"""Phase 1 Foundation, F7: the I-V page, the reference adapter, hosted in a real Explorer.

Clicks and keys through the QWindow, real menus, a real save and reopen. Synthetic data only
(CSV traces written to ``tmp_path``, in-memory sessions); offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import iv
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS
from nanoscope.gui.plot_interaction import app_menu_for

from . import p1_hosting as H
from .test_beta_iv_first_open_async import _ohmic_session as _big_session
from .test_iv import _ohmic_session

NATURE = "7 pt Arial · Nature, Elsevier"
RETURN = QtCore.Qt.Key.Key_Return


@pytest.fixture
def page(qapp, qtbot, tmp_path):
    w = IVWindow(_ohmic_session(tmp_path, G_nS=100.0, leak_pA=-10.0))
    qtbot.addWidget(w)
    return w


@pytest.fixture
def explorer(qtbot, page):
    return H.hosted(qtbot, page, "iv")


def test_the_groups_hold_the_pages_own_sections_and_the_hidden_set_stays_wired(explorer, page):
    adapter = explorer.adapter("iv")
    assert adapter is page.inspector_adapter() and adapter.fallback is False
    assert [g.key for g in adapter.groups] == ["estimator"]
    frame = explorer.inspector().pane("iv").data_pane.group("estimator")
    assert page._data_section.parentWidget() is frame.body
    assert page._analysis_section.parentWidget() is frame.body
    assert page._data_section.isVisible() and page._analysis_section.isVisible()
    for name in ("_refresh_btn", "_save_btn", "_save_data_btn", "_export_section",
                 "_appearance_section"):
        assert not getattr(page, name).isVisible(), name       # hidden, objects kept
    assert not page._workflow.isVisible()
    heard = []
    page.workspaceChanged.connect(lambda: heard.append(1))
    page._workflow.primary_actions[0].click()                  # the hidden verb still works
    assert heard == [1]
    assert page.property("nanoscopeInspectorHosted") is True
    assert page.inspector_adapter() is adapter                 # idempotent


def test_update_plot_through_a_real_click_calls_the_pages_verb_once(explorer, page, monkeypatch):
    calls = []
    real = page.refresh
    monkeypatch.setattr(page, "refresh", lambda: (calls.append("refresh"), real())[1])
    heard = []
    page.workspaceChanged.connect(lambda: heard.append(1))
    H.press_update(explorer)
    assert calls == ["refresh"] and heard == [1]
    H.key(explorer, RETURN, H.CTRL)
    assert calls == ["refresh", "refresh"] and heard == [1, 1]
    assert explorer.toolbar().update_button.text() == "Update plot"      # count 0 on I-V


def test_hosting_binding_and_tab_switches_compute_nothing(explorer, page, monkeypatch):
    for name in ("refresh", "_refresh_from_user", "_redraw", "_refresh_maybe_async"):
        monkeypatch.setattr(page, name, lambda *a, **k: pytest.fail("a host action computed"))
    monkeypatch.setattr(iv, "iv_points", lambda *a, **k: pytest.fail("gathered"))
    explorer.refresh_header()
    explorer.toolbar().sync()
    explorer.inspector().set_tab(1)
    explorer.inspector().set_tab(0)
    explorer.show_home()
    explorer.raise_page("iv")
    H.key(explorer, QtCore.Qt.Key.Key_2, H.CTRL)
    H.key(explorer, QtCore.Qt.Key.Key_1, H.CTRL)


def test_a_slow_gather_greys_update_plot_and_command_return_does_nothing(
        qapp, qtbot, tmp_path, monkeypatch):
    release = threading.Event()
    real = iv.compute_channel_medians
    monkeypatch.setattr(iv, "compute_channel_medians",
                        lambda items: (release.wait(30), real(items))[1])
    iv.forget_channel_medians()
    try:
        page = IVWindow(_big_session(tmp_path))
        qtbot.addWidget(page)
        assert page._gather_worker is not None                 # the constructor started one
        explorer = H.hosted(qtbot, page, "iv")
        bar = explorer.toolbar()
        assert not bar.update_action.isEnabled() and not bar.update_button.isEnabled()
        assert "gather" in bar.update_action.toolTip()
        calls = []
        monkeypatch.setattr(page, "refresh", lambda: calls.append("refresh"))
        H.key(explorer, RETURN, H.CTRL)
        H.press_update(explorer)
        assert calls == []                                     # busy: nothing runs
        release.set()
        qtbot.waitUntil(lambda: page._gather_worker is None, timeout=20_000)
        assert bar.update_action.isEnabled() and bar.update_button.isEnabled()
        monkeypatch.undo()
    finally:
        release.set()
        iv.forget_channel_medians()


def _after_the_last_separator(menu) -> list[str]:
    """The export block of the app right-click menu: context entries, Reset view and the
    zoom items come first, each block closed by a separator."""
    texts: list[str] = []
    for action in menu.actions():
        if action.isSeparator():
            texts = []
        else:
            texts.append(action.text())
    return texts


def test_a_gather_that_starts_after_hosting_greys_the_verb_at_once(qapp, qtbot, tmp_path, monkeypatch):
    release = threading.Event()
    real = iv.compute_channel_medians
    monkeypatch.setattr(iv, "compute_channel_medians",
                        lambda items: (release.wait(30), real(items))[1])
    iv.forget_channel_medians()
    try:
        page = IVWindow(_big_session(tmp_path), defer_initial_refresh=True)
        qtbot.addWidget(page)
        explorer = H.hosted(qtbot, page, "iv")
        bar = explorer.toolbar()
        assert page._gather_worker is None and bar.update_action.isEnabled()
        page._refresh_maybe_async()                      # the page starts a gather by itself
        assert page._gather_worker is not None
        assert not bar.update_action.isEnabled()          # heard from pendingChanged, no refresh call
    finally:
        release.set()
        page.abort()
        iv.forget_channel_medians()


@pytest.mark.parametrize("section", ("Appearance", "Export"))
def test_a_page_left_on_another_section_is_hosted_with_the_replaced_controls_hidden(
        qapp, qtbot, tmp_path, section):
    page = IVWindow(_ohmic_session(tmp_path))
    qtbot.addWidget(page)
    page._workflow.set_section(section)                  # standalone: shows that Section's widgets
    explorer = H.hosted(qtbot, page, "iv")
    for name in ("_refresh_btn", "_save_btn", "_save_data_btn", "_export_section",
                 "_appearance_section"):
        assert getattr(page, name).isHidden(), name       # each hidden itself, not through a parent


def test_abort_during_a_gather_gives_the_verb_back(qapp, qtbot, tmp_path, monkeypatch):
    release = threading.Event()
    real = iv.compute_channel_medians
    monkeypatch.setattr(iv, "compute_channel_medians",
                        lambda items: (release.wait(30), real(items))[1])
    iv.forget_channel_medians()
    try:
        page = IVWindow(_big_session(tmp_path))
        qtbot.addWidget(page)
        explorer = H.hosted(qtbot, page, "iv")
        bar = explorer.toolbar()
        assert not bar.update_action.isEnabled()
        release.set()                       # let the thread finish, then abort what is left
        page.abort()
        assert page._gather_worker is None and bar.update_action.isEnabled()
    finally:
        release.set()
        iv.forget_channel_medians()


def test_export_menu_is_the_right_click_menus_export_entries(explorer, page):
    texts = [a.text() for a in H.export_menu(explorer).actions() if not a.isSeparator()]
    H.export_menu(explorer).close()
    page_entries = [t for t in texts if t not in ("Save to Figure Library…", "Save figure bundle…",
                                                  "Open Figure Library")]
    right_click = _after_the_last_separator(app_menu_for(page._plot))
    assert page_entries == right_click == ["Export current figure…", "Export data (CSV)…"]


def test_export_entries_run_the_pages_own_handlers(explorer, page, monkeypatch):
    seen = []
    monkeypatch.setattr(page, "save_image", lambda *a, **k: seen.append("image"))
    monkeypatch.setattr(page, "export_data", lambda *a, **k: seen.append("data"))
    H.choose_export(explorer, "Export current figure…")
    H.choose_export(explorer, "Export data (CSV)…")
    assert seen == ["image", "data"]


def test_undo_redo_and_reset_figure_follow_the_pages_own_history(explorer, page):
    widget, bar = page._appearance_widget, explorer.toolbar()
    assert not bar.undo_action.isEnabled()
    widget.apply_look(_BUILTIN_PRESETS[NATURE])
    assert bar.undo_action.isEnabled()                           # historyChanged, no refresh call
    assert bar.undo_action.toolTip() == f"Undo: {NATURE} preset"
    steps = widget._history.depth()[0]
    H.press_reset_figure(explorer)
    assert widget._history.depth()[0] == steps + 1 and widget.state().to_dict() == {}
    H.press_undo(explorer)
    assert widget.state().title_size == 7 and page.appearance_state()["title_size"] == 7
    H.press_undo(explorer)
    assert page.appearance_state() == {} and bar.redo_action.isEnabled()
    H.press_redo(explorer)
    assert widget.state().title_size == 7


def test_the_format_spec_names_the_pages_plot_and_adds_no_extras(explorer, page):
    from nanoscope.gui.figure_text import figure_text_for
    spec = explorer.adapter("iv").format
    assert spec.appearance is page._appearance_widget and not spec.extras
    assert spec.controllers() == [figure_text_for(page._plot)]


def test_the_undo_key_takes_back_an_edit_made_in_the_figure_from_either_tab(explorer, page):
    """The old FIG-4 test, ported: the keys sit on the page, so an edit made in the figure is
    taken back whichever inspector tab shows."""
    from .test_plot_window_routes import _edit_double_click, _press_undo
    plot = page.plot_item()
    axis = plot.getAxis("bottom")
    widget = page._appearance_widget
    for tab in (0, 1):
        explorer.inspector().set_tab(tab)
        QtWidgets.QApplication.processEvents()
        before, depth = axis.label.toPlainText().strip(), widget._history.depth()
        _edit_double_click(plot, axis.label, f"Pore {tab}")
        assert axis.label.toPlainText().strip() == f"Pore {tab}"
        assert widget._history.depth() == (depth[0] + 1, 0)
        _press_undo(page)
        assert axis.label.toPlainText().strip() == before
        assert widget._history.depth() == (depth[0], 1)


def test_the_page_fits_the_explorers_minimum_size(qtbot, page):
    explorer = H.hosted(qtbot, page, "iv", size=(840, 560))
    explorer.resize(840, 560)
    QtWidgets.QApplication.processEvents()
    assert page.minimumSizeHint().width() <= 500
    host = explorer.inspector()
    assert host.minimumSizeHint().width() <= 330
    pane = host.pane("iv")
    assert pane.data_pane.horizontalScrollBar().maximum() == 0
    # The Format tab is the Format group's element panels (they replaced the Foundation's
    # 483 px stub that scrolled sideways): no sideways scroll bar, and every panel fits.
    host.set_tab(1)
    QtWidgets.QApplication.processEvents()
    assert pane.format_pane._scroll.horizontalScrollBar().maximum() == 0
    assert page._plot_widget.width() >= 420 and page._plot_widget.height() >= 300
    assert explorer.width() == 840


def test_reset_view_sits_on_the_status_line_and_the_old_button_is_renamed(explorer, page):
    status = page.statusBar()
    visible = status.findChildren(QtWidgets.QToolButton, "plotResetViewButton")
    assert len(visible) == 1 and visible[0].isVisible()
    legacy = page.findChildren(QtWidgets.QAbstractButton, "plotResetViewButtonLegacy")
    assert len(legacy) == 1 and not legacy[0].isVisible()
    assert page.findChild(QtWidgets.QAbstractButton, "plotResetViewButton") is visible[0]
    page._plot.setXRange(0.2, 0.3, padding=0)
    QtWidgets.QApplication.processEvents()
    assert page._plot.viewRange()[0] == pytest.approx([0.2, 0.3])
    H.click(visible[0], pos=QtCore.QPoint(8, visible[0].height() // 2))
    lo, hi = page._plot.viewRange()[0]
    assert abs(lo - 0.2) > 1e-6 or abs(hi - 0.3) > 1e-6        # the page's own framing is back


def _csv(path, value, n=4000):
    x = np.full(n, value, dtype=np.float32)
    pd.DataFrame({"time": np.arange(n) / 50000.0, "current": x}).to_csv(path, index=False)


def _main_window(qtbot):
    from nanoscope.gui.main_window import MainWindow
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_a_full_save_and_reopen_restores_the_appearance_with_the_page_hosted(
        qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.core.project import DocumentRef, ProjectState
    from nanoscope.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    paths = []
    for mv in (-200, -500, -1000):
        path = tmp_path / f"pore_{mv}mV.csv"
        _csv(path, 0.1 * mv - 10.0)
        paths.append(path)
    source = _main_window(qtbot)
    with qtbot.waitSignal(source.projectRestored, timeout=20_000):
        source._begin_restore(ProjectState(documents=[DocumentRef(str(p)) for p in paths]))
    source._open_iv()
    QtWidgets.QApplication.processEvents()
    page = source._iv_window
    explorer = source._analysis_explorer
    live_key = source._figure_workspace.id_for_page(page)
    assert explorer.adapter(live_key).page is page and page.property("nanoscopeInspectorHosted")
    H.press_update(explorer)
    page._appearance_widget.apply_look(_BUILTIN_PRESETS[NATURE])
    page._appearance_widget.title.textEdited.emit("Pore 3")
    saved = page.appearance_state()
    assert saved["title"] == "Pore 3" and saved["title_size"] == 7
    project = tmp_path / "hosted.nanoscope"
    assert source._write_project(str(project))
    source._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_iv()
    QtWidgets.QApplication.processEvents()
    again = reopened._iv_window
    assert again.appearance_state() == saved
    assert again.property("nanoscopeInspectorHosted") is True
    assert reopened._analysis_explorer.toolbar().bound_key() == reopened._figure_workspace.id_for_page(again)
    reopened._dirty = False
