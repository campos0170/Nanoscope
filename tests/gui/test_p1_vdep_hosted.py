"""Phase 1, G3: Voltage dependence hosted in a real Analysis Explorer.

Clicks and keys go through the QWindow, menus are real, the project round trip is a real
``_write_project`` and ``_open_project`` until ``projectRestored``. Synthetic data only
(an in-memory ladder session, CSV traces written to ``tmp_path``); offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import voltage_dep
from nanoscope.gui.figure_text import figure_text_for
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS
from nanoscope.gui.plot_interaction import app_menu_for
from nanoscope.gui.voltage_dep_panel import VDepWindow

from . import p1_hosting as H
from .test_plot_window_routes import _edit_double_click, _press_undo
from .test_voltage_dep import _ladder_session

KEY = "voltage_series"
RETURN = QtCore.Qt.Key.Key_Return
DOWN = QtCore.Qt.Key.Key_Down
SHELL = {"Save to Figure Library…", "Save figure bundle…", "Open Figure Library"}


@pytest.fixture
def page(qapp, qtbot, tmp_path):
    w = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(w)
    return w


@pytest.fixture
def explorer(qtbot, page):
    return H.hosted(qtbot, page, KEY)


def _body(explorer, group):
    return explorer.inspector().pane(KEY).data_pane.group(group).body


def _count_gathers(monkeypatch, *, forbid=False):
    """Every gather of the (V, metric) points goes through ``voltage_dep.vdep_series``."""
    calls = []
    real = voltage_dep.vdep_series

    def counted(*args, **kwargs):
        if forbid:
            pytest.fail("the points were gathered again")
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(voltage_dep, "vdep_series", counted)
    return calls


def _pick_with_the_keyboard(combo, key=DOWN):
    """A person at the combo: focus it, press a key. The page tells that from code."""
    combo.setFocus()
    QtWidgets.QApplication.processEvents()
    assert combo.hasFocus()
    H.key(combo, key)


def test_the_groups_hold_the_pages_own_widgets_and_the_hidden_set_stays_wired(explorer, page):
    adapter = explorer.adapter(KEY)
    assert adapter is page.inspector_adapter() and adapter.fallback is False
    assert [g.key for g in adapter.groups] == ["sources", "plot", "fit", "recipes"]
    assert [g.expanded for g in adapter.groups] == [True, True, False, False]
    assert page._data_section.parentWidget() is _body(explorer, "sources")
    for name in ("_metric_combo", "_abs_v", "_log_y"):
        assert _body(explorer, "plot").isAncestorOf(getattr(page, name)), name
    assert _body(explorer, "fit").isAncestorOf(page._fit_combo)
    assert page._templates.parentWidget() is _body(explorer, "recipes")
    assert page._accepted_only.isVisible() and page._metric_combo.isVisible()
    assert not page._fit_combo.isVisible()
    assert page._abs_v.isVisible() and page._log_y.isVisible()
    explorer.reveal("fit")
    assert page._fit_combo.isVisible()
    for name in ("_refresh_btn", "_save_btn", "_save_data_btn", "_export_section",
                 "_appearance_section", "_analysis_section"):
        assert getattr(page, name).isHidden(), name
    assert not page._fit_apply_btn.isVisible() and not page._workflow.isVisible()
    for name in ("_headline", "_warning_scroll", "_plot_widget", "_details", "_table"):
        assert getattr(page, name).parentWidget() is not None
    assert page._plot_widget.isVisible() and page._table.isVisible() and page._details.isVisible()
    # hidden but still wired
    seen = []
    page.workspaceChanged.connect(lambda: seen.append(1))
    page._workflow.primary_actions[0].click()
    assert seen == [1]
    assert page.property("nanoscopeInspectorHosted") is True
    assert page.inspector_adapter() is adapter                  # idempotent
    assert "fit you have chosen" in explorer.toolbar().update_action.toolTip()
    spec = adapter.format
    assert spec.appearance is page._appearance_widget and not spec.extras
    assert spec.controllers() == [figure_text_for(page._plot)]


def test_the_controls_do_their_jobs_through_real_clicks_and_keys(explorer, page, monkeypatch):
    plot = page._plot
    # a real click on Log y re-draws on a log axis
    assert not plot.getAxis("left").logMode
    H.click(page._log_y, pos=QtCore.QPoint(8, page._log_y.height() // 2))
    assert page._log_y.isChecked() and plot.getAxis("left").logMode
    H.click(page._log_y, pos=QtCore.QPoint(8, page._log_y.height() // 2))
    assert not plot.getAxis("left").logMode
    # a real click on |V| flips the voltage axis label
    before = plot.getAxis("bottom").labelText
    H.click(page._abs_v, pos=QtCore.QPoint(8, page._abs_v.height() // 2))
    assert page._abs_v.isChecked() and plot.getAxis("bottom").labelText != before
    # a real key on the Metric combo re-gathers and re-draws for the other metric
    first = page._metric()
    _pick_with_the_keyboard(page._metric_combo)
    assert page._metric() != first
    assert page._drawn_metric == page._metric()
    # and a click on Accepted only re-gathers
    calls = _count_gathers(monkeypatch)
    H.click(page._accepted_only, pos=QtCore.QPoint(8, page._accepted_only.height() // 2))
    assert calls == [1] and not page._accepted_only.isChecked()


@pytest.mark.parametrize("name", ["_abs_v", "_log_y"])
def test_a_click_in_the_middle_of_a_moved_check_box_still_toggles_it(explorer, page, name):
    """The strip sized a check box to its words; in the form it must not fill the column, or a
    click in the middle of the row would find empty space (the Phase 0 tests click there)."""
    box = getattr(page, name)
    assert box.isVisible() and box.width() < 120
    was = box.isChecked()
    H.click(box)
    assert box.isChecked() is (not was)


def test_update_plot_counts_the_waiting_fit_and_applies_it_without_gathering(
        explorer, page, monkeypatch):
    bar = explorer.toolbar()
    assert page._applied_fit == "none" and bar.update_button.text() == "Update plot"
    heard = []
    page.pendingChanged.connect(lambda: heard.append(1))
    explorer.reveal("fit")
    _pick_with_the_keyboard(page._fit_combo)                    # a person chose "linear"
    assert page._fit_combo.currentText() != page._applied_fit
    assert heard and page._workflow.primary_actions[1].text() == "Apply fit ●"
    assert bar.update_button.text() == "Update plot (1)" and bar.update_button.property("pending")
    assert "fit model: Apply pending" in bar.update_action.toolTip()
    # choosing the drawn model again takes the mark away
    _pick_with_the_keyboard(page._fit_combo, QtCore.Qt.Key.Key_Up)
    assert bar.update_button.text() == "Update plot" and not bar.update_button.property("pending")
    _pick_with_the_keyboard(page._fit_combo)
    chosen = page._fit_combo.currentText()
    # Update plot (a real click) applies the fit and does NOT gather again
    _count_gathers(monkeypatch, forbid=True)
    seen = []
    page.workspaceChanged.connect(lambda: seen.append(1))
    H.press_update(explorer)
    assert page._applied_fit == chosen and page._fit_item is not None
    assert seen == [1]
    assert bar.update_button.text() == "Update plot" and not bar.update_button.property("pending")
    assert page._workflow.primary_actions[1].text() == "Apply fit"
    # nothing waiting: the same click and ⌘Return re-gather as Analyze / refresh did
    monkeypatch.undo()
    calls = _count_gathers(monkeypatch)
    H.press_update(explorer)
    H.key(explorer, RETURN, H.CTRL)
    assert calls == [1, 1]
    assert page._applied_fit == chosen


def test_a_fit_waiting_in_a_saved_page_counts_after_a_silent_restore(explorer, page):
    """The count follows the mark even when the page writes it silently: the Explorer's
    refresh route (``refresh_header``) is how a project open reaches the toolbar."""
    page._fit_combo.blockSignals(True)
    page._fit_combo.setCurrentText("exponential")
    page._fit_combo.blockSignals(False)
    page._sync_fit_apply_btn()                                  # what restore_voltage_dependence does
    assert explorer.toolbar().update_button.text() == "Update plot (1)"
    page._fit_combo.setCurrentText("none")                      # code applies at once
    assert page._applied_fit == "none"
    explorer.refresh_header()
    assert explorer.toolbar().update_button.text() == "Update plot"


def test_the_undo_key_takes_back_an_edit_made_in_the_figure_from_either_tab(explorer, page):
    plot = page._plot
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
    # the toolbar mirrors the page's own history and Reset figure is one step
    bar = explorer.toolbar()
    widget.apply_look(_BUILTIN_PRESETS["7 pt Arial · Nature, Elsevier"])
    assert bar.undo_action.isEnabled() and "Nature" in bar.undo_action.toolTip()
    steps = widget._history.depth()[0]
    H.press_reset_figure(explorer)
    assert widget._history.depth()[0] == steps + 1
    H.press_undo(explorer)
    assert widget.state().title_size == 7


def test_export_menu_is_the_right_click_menus_export_entries(explorer, page, monkeypatch):
    texts = [a.text() for a in H.export_menu(explorer).actions() if not a.isSeparator()]
    H.export_menu(explorer).close()
    entries = [t for t in texts if t not in SHELL]
    block: list[str] = []
    for action in app_menu_for(page._plot).actions():
        if action.isSeparator():
            block = []
        else:
            block.append(action.text())
    assert entries == block == ["Export current figure…", "Export data (CSV)…"]
    seen = []
    monkeypatch.setattr(page, "save_image", lambda *a, **k: seen.append("image"))
    monkeypatch.setattr(page, "export_data", lambda *a, **k: seen.append("data"))
    H.choose_export(explorer, "Export current figure…")
    H.choose_export(explorer, "Export data (CSV)…")
    assert seen == ["image", "data"]


def test_reset_view_sits_on_the_status_line_and_returns_the_pages_framing(explorer, page):
    status = page.statusBar()
    visible = status.findChildren(QtWidgets.QToolButton, "plotResetViewButton")
    assert len(visible) == 1 and visible[0].isVisible()
    assert page.findChild(QtWidgets.QAbstractButton, "plotResetViewButton") is visible[0]
    lo, hi = page._plot.viewRange()[0]
    page._plot.setXRange(lo + (hi - lo) * 0.4, lo + (hi - lo) * 0.6, padding=0)
    QtWidgets.QApplication.processEvents()
    zoomed = page._plot.viewRange()[0]
    H.click(visible[0], pos=QtCore.QPoint(8, visible[0].height() // 2))
    assert page._plot.viewRange()[0][1] - page._plot.viewRange()[0][0] > (zoomed[1] - zoomed[0]) * 1.5


@pytest.mark.parametrize("section", ("Data", "Analysis", "Appearance", "Export"))
def test_a_page_left_on_any_section_is_hosted_with_the_replaced_controls_hidden(
        qapp, qtbot, tmp_path, section):
    page = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(page)
    page._workflow.set_section(section)                 # standalone: shows that Section's widgets
    H.hosted(qtbot, page, KEY)
    for name in ("_refresh_btn", "_save_btn", "_save_data_btn", "_export_section",
                 "_appearance_section", "_analysis_section"):
        assert getattr(page, name).isHidden(), name     # each hid itself, not through a parent
    assert page._metric_combo.isVisible() and page._templates.parentWidget() is not None


def test_the_page_fits_the_explorers_minimum_size(qtbot, page):
    explorer = H.hosted(qtbot, page, KEY, size=(840, 560))
    explorer.resize(840, 560)
    host = explorer.inspector()
    for group in host.adapter(KEY).groups:
        host.reveal(group.key)
    QtWidgets.QApplication.processEvents()
    assert page.minimumSizeHint().width() <= 500
    pane = host.pane(KEY)
    for group in explorer.adapter(KEY).groups:
        assert pane.data_pane.group(group.key).minimumSizeHint().width() <= 290, group.key
    assert pane.data_pane.horizontalScrollBar().maximum() == 0
    assert page._plot_widget.width() >= 420 and page._plot_widget.height() >= 300
    assert explorer.width() == 840


def test_hosting_binding_and_tab_switches_compute_nothing(explorer, page, monkeypatch):
    for name in ("refresh", "_refresh_from_user", "_apply_fit", "_redraw"):
        monkeypatch.setattr(page, name, lambda *a, **k: pytest.fail("a host action computed"))
    explorer.refresh_header()
    explorer.toolbar().sync()
    explorer.inspector().set_tab(1)
    explorer.inspector().set_tab(0)
    explorer.show_home()
    explorer.raise_page(KEY)
    H.key(explorer, QtCore.Qt.Key.Key_2, H.CTRL)
    H.key(explorer, QtCore.Qt.Key.Key_1, H.CTRL)


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


def test_a_full_save_and_reopen_restores_settings_look_and_the_waiting_fit(
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
    source._open_vdep()
    QtWidgets.QApplication.processEvents()
    page = source._vdep_window
    explorer = source._analysis_explorer
    H.press_update(explorer)
    live_key = source._figure_workspace.id_for_page(page)
    assert explorer.adapter(live_key).page is page and page.property("nanoscopeInspectorHosted")
    page._abs_v.setChecked(True)
    page._log_y.setChecked(True)
    H.press_update(explorer)
    page._appearance_widget.apply_look(_BUILTIN_PRESETS["7 pt Arial · Nature, Elsevier"])
    page._appearance_widget.title.textEdited.emit("Ladder")
    page._fit_combo.blockSignals(True)
    page._fit_combo.setCurrentText("linear")        # chosen, not applied: a waiting fit
    page._fit_combo.blockSignals(False)
    page._sync_fit_apply_btn()
    saved = page.appearance_state()
    assert saved["title"] == "Ladder" and saved["title_size"] == 7
    project = tmp_path / "hosted.nanoscope"
    assert source._write_project(str(project))
    source._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_vdep()
    QtWidgets.QApplication.processEvents()
    again = reopened._vdep_window
    assert again.property("nanoscopeInspectorHosted") is True
    assert again.appearance_state() == saved
    assert again._abs_v.isChecked() and again._log_y.isChecked()
    assert again._fit_combo.currentText() == "linear" and again._applied_fit == "none"
    explorer = reopened._analysis_explorer
    explorer.refresh_header()
    assert explorer.toolbar().bound_key() == reopened._figure_workspace.id_for_page(again)
    assert explorer.toolbar().update_button.text() == "Update plot (1)"
    reopened._dirty = False
