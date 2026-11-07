"""Phase 1, G3: Distribution summaries hosted in a real Analysis Explorer.

Clicks and keys go through the QWindow, menus are real, the project round trip is a real
``_write_project`` and ``_open_project`` until ``projectRestored``. Synthetic data only
(in-memory event tables); offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.core.sources import PlotSource
from nanoscope.gui import distribution_summary_panel as M
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS
from nanoscope.gui.plot_interaction import app_menu_for

from . import p1_hosting as H
from .test_plot_window_routes import _edit_double_click, _press_undo

KEY = "distribution_summaries"
RETURN = QtCore.Qt.Key.Key_Return
DOWN = QtCore.Qt.Key.Key_Down
SHELL = {"Save to Figure Library…", "Save figure bundle…", "Open Figure Library"}


def _source(key="a", label="Run A", dwell=(.001, .002, .002, .005, .008)):
    n = len(dwell)
    return PlotSource(key, label, pd.DataFrame({
        "dwell_s": list(dwell), "depth_pA": [10. * (i + 1) for i in range(n)],
        "accepted": [True] * n}), kind="manual", doc_key=key, current=(key == "a"))


@pytest.fixture
def page(qapp, qtbot):
    w = M.DistributionSummaryWindow()
    qtbot.addWidget(w)
    w.set_sources([_source(), _source("b", "Run B", (.003, .004, .004, .006))])
    return w


@pytest.fixture
def explorer(qtbot, page):
    return H.hosted(qtbot, page, KEY)


def _body(explorer, group):
    return explorer.inspector().pane(KEY).data_pane.group(group).body


def _pick_with_the_keyboard(combo, key=DOWN):
    combo.setFocus()
    QtWidgets.QApplication.processEvents()
    assert combo.hasFocus()
    H.key(combo, key)


def _count(explorer) -> int:
    return explorer.adapter(KEY).pending().count


def _computes(monkeypatch):
    calls = []
    real = M.summarize_distribution
    monkeypatch.setattr(M, "summarize_distribution",
                        lambda *a, **k: (calls.append(k["kind"]), real(*a, **k))[1])
    return calls


def test_the_groups_hold_the_pages_own_widgets_and_the_hidden_set_stays_wired(explorer, page):
    adapter = explorer.adapter(KEY)
    assert adapter is page.inspector_adapter() and adapter.fallback is False
    assert [g.key for g in adapter.groups] == ["sources", "plot", "recipes"]
    assert [g.expanded for g in adapter.groups] == [True, True, False]
    for name in ("_source_list", "_accepted_only", "_data_note"):
        assert _body(explorer, "sources").isAncestorOf(getattr(page, name)), name
    for name in ("_metric_combo", "_view_combo", "_violin_points", "_method_note"):
        assert _body(explorer, "plot").isAncestorOf(getattr(page, name)), name
    assert page._templates.parentWidget() is _body(explorer, "recipes")
    for name in ("_data_section", "_analysis_section", "_appearance_section", "_export_section"):
        assert getattr(page, name).isHidden(), name
    assert not page._workflow.isVisible()
    assert page._source_list.isVisible() and page._metric_combo.isVisible()
    assert page._status.isVisible() and page._plot_widget.isVisible()
    assert page.property("nanoscopeInspectorHosted") is True
    assert page.inspector_adapter() is adapter                  # idempotent
    assert "Compute the summary" in explorer.toolbar().update_action.toolTip()
    spec = adapter.format
    assert spec.appearance is page._appearance_widget
    assert not spec.extras and page._source_appearance_editor.isHidden()
    assert spec.controllers() == [M.figure_text_for(page._plot)]
    # hidden but still wired: the old verb button works and keeps its text
    assert not page.has_data()
    page._workflow.primary_actions[0].click()
    assert page.has_data()


def test_a_click_in_the_middle_of_the_accepted_only_check_box_still_toggles_it(explorer, page):
    box = page._accepted_only
    assert box.isVisible() and box.width() < 220
    was = box.isChecked()
    H.click(box)
    assert box.isChecked() is (not was)


def test_the_violin_grid_row_follows_the_summary_type_through_real_keys(explorer, page):
    assert not page._violin_points.isEnabled()                  # ECDF
    _pick_with_the_keyboard(page._view_combo, QtCore.Qt.Key.Key_End)
    assert page._view_combo.currentData() == "violin" and page._violin_points.isEnabled()
    assert "Gaussian KDE" in page._method_note.text()


def test_update_plot_counts_the_draft_changes_and_computes_only_on_the_verb(
        explorer, page, monkeypatch):
    bar = explorer.toolbar()
    calls = _computes(monkeypatch)
    heard = []
    page.pendingChanged.connect(lambda: heard.append(1))
    # nothing generated yet, a source ticked
    assert bar.update_button.text() == "Update plot (1)"
    assert explorer.adapter(KEY).pending().reasons == ("not generated yet",)
    H.press_update(explorer)                                    # a real click
    assert calls == ["ecdf"] and page.has_data()
    assert bar.update_button.text() == "Update plot" and not bar.update_button.property("pending")
    assert _count(explorer) == 0 and heard
    # controls only edit a draft: they compute nothing and raise the count, key by key
    _pick_with_the_keyboard(page._metric_combo)
    assert calls == ["ecdf"] and _count(explorer) == 1
    assert explorer.adapter(KEY).pending().reasons == ("the measurement",)
    assert bar.update_button.text() == "Update plot (1)" and bar.update_button.property("pending")
    assert page._workflow.primary_actions[0].text() == "Generate ●"
    _pick_with_the_keyboard(page._view_combo)
    assert _count(explorer) == 2
    assert bar.update_button.text() == "Update plot (2)"
    assert "the summary type" in bar.update_action.toolTip()
    # moved back: one key still differs, then none differs but the figure stays marked stale
    _pick_with_the_keyboard(page._metric_combo, QtCore.Qt.Key.Key_Up)
    assert _count(explorer) == 1
    _pick_with_the_keyboard(page._view_combo, QtCore.Qt.Key.Key_Up)
    assert all(page.config()[k] == page._applied_config[k] for k in page.config())
    assert page._stale and _count(explorer) == 1                # the floor: never 0 beside STALE
    assert bar.update_button.text() == "Update plot (1)"
    H.key(explorer, RETURN, H.CTRL)                             # ⌘Return is the same verb
    assert calls == ["ecdf", "ecdf"] and _count(explorer) == 0 and not page._stale


def test_a_silent_restore_reaches_the_toolbar_without_the_refresh_route(explorer, page):
    """``restore_state`` emits nothing the Explorer listens to but ``pendingChanged``."""
    H.press_update(explorer)
    bar = explorer.toolbar()
    assert bar.update_button.text() == "Update plot"
    payload, tables = page.snapshot_state()
    payload["stale"] = True                                 # a project saved while out of date
    page.restore_state(payload, tables)
    assert page._stale and bar.update_button.text() == "Update plot (1)"
    payload["stale"] = False
    page.restore_state(payload, tables)
    assert bar.update_button.text() == "Update plot"
    # and the adapter's pending_changed IS that signal: a silent write, then the signal alone
    page._stale = True
    assert bar.update_button.text() == "Update plot"
    page.pendingChanged.emit()
    assert bar.update_button.text() == "Update plot (1)"


def test_no_ticked_source_is_nothing_to_update(explorer, page):
    page._source_list.item(0).setCheckState(QtCore.Qt.Unchecked)
    assert page._selected_keys == [] and not page.has_data()
    assert _count(explorer) == 0 and explorer.toolbar().update_button.text() == "Update plot"


def test_a_source_ticked_with_the_mouse_raises_the_count(explorer, page):
    H.press_update(explorer)
    assert _count(explorer) == 0
    item = page._source_list.item(1)
    rect = page._source_list.visualItemRect(item)
    H.click(page._source_list.viewport(), pos=QtCore.QPoint(8, rect.center().y()))
    assert page._selected_keys == ["a", "b"]
    assert explorer.adapter(KEY).pending().reasons == ("the ticked sources",)
    assert explorer.toolbar().update_button.text() == "Update plot (1)"


def test_a_failed_generate_keeps_the_old_figure_marked_stale(explorer, page, monkeypatch):
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2])))
    H.press_update(explorer)
    before = page._plot_data()
    page.set_sources([_source(dwell=[True, False])])
    assert page._stale and _count(explorer) >= 1
    H.press_update(explorer)
    assert warned and "real numbers" in warned[0]
    pd.testing.assert_frame_equal(page._plot_data(), before, check_exact=True)
    assert page._stale and _count(explorer) >= 1
    assert explorer.toolbar().update_button.property("pending")


def test_export_is_greyed_until_a_figure_exists_and_equals_the_right_click_menu(
        explorer, page, monkeypatch):
    menu = H.export_menu(explorer)
    entries = {a.text(): a for a in menu.actions() if not a.isSeparator() and a.text() not in SHELL}
    assert list(entries) == ["Export current figure…", "Export plotted coordinates CSV…"]
    assert not any(a.isEnabled() for a in entries.values())     # no figure yet
    menu.close()
    H.press_update(explorer)
    menu = H.export_menu(explorer)
    entries = {a.text(): a for a in menu.actions() if not a.isSeparator() and a.text() not in SHELL}
    assert all(a.isEnabled() for a in entries.values())
    menu.close()
    block: list[str] = []
    for action in app_menu_for(page._plot).actions():
        if action.isSeparator():
            block = []
        else:
            block.append(action.text())
    assert block == list(entries)
    seen = []
    monkeypatch.setattr(page, "save_image", lambda *a, **k: seen.append("image"))
    monkeypatch.setattr(page, "save_csv", lambda *a, **k: seen.append("csv"))
    H.choose_export(explorer, "Export current figure…")
    H.choose_export(explorer, "Export plotted coordinates CSV…")
    assert seen == ["image", "csv"]


def test_the_undo_key_takes_back_an_edit_made_in_the_figure_from_either_tab(explorer, page):
    H.press_update(explorer)
    plot, widget = page._plot, page._appearance_widget
    for tab in (0, 1):
        explorer.inspector().set_tab(tab)
        QtWidgets.QApplication.processEvents()
        before, depth = plot.titleLabel.text, widget._history.depth()
        _edit_double_click(plot, plot.titleLabel, f"Edit {tab}")
        assert plot.titleLabel.text == f"Edit {tab}"
        assert widget._history.depth() == (depth[0] + 1, 0)
        _press_undo(page)
        assert plot.titleLabel.text == before
        assert widget._history.depth() == (depth[0], 1)
    bar = explorer.toolbar()
    widget.apply_look(_BUILTIN_PRESETS["7 pt Arial · Nature, Elsevier"])
    assert bar.undo_action.isEnabled()
    steps = widget._history.depth()[0]
    H.press_reset_figure(explorer)
    assert widget._history.depth()[0] == steps + 1


@pytest.mark.parametrize("previous_name", [None, "Earlier name"])
def test_format_legend_typing_uses_source_names_and_one_undo_step(
        explorer, page, qtbot, previous_name):
    page._source_list.item(1).setCheckState(QtCore.Qt.CheckState.Checked)
    H.press_update(explorer)
    names = {"a": {"color": "#009e73"}, "b": {"label": "Other source"}}
    if previous_name is not None:
        names["a"]["label"] = previous_name
    page.set_source_appearance_state(names)
    before_data = page._plot_data()
    widget = page._appearance_widget
    widget.clear_history()
    before_mark = widget.history_mark()
    explorer.inspector().set_tab(1)
    pane = explorer.inspector().pane(KEY).format_pane
    pane.show_element("legend.entry.a")
    panel = pane.current_panel()
    assert panel.words.isVisibleTo(panel) and not panel.fields.isVisibleTo(panel)
    panel.words.setFocus()
    panel.words.selectAll()
    for chunk, expected in (("Apo", "Apo"), (" <1>", "Apo <1>")):
        QtTest.QTest.keyClicks(panel.words, chunk)
        for _ in range(3):
            QtWidgets.QApplication.processEvents()
        assert panel.words.text() == expected
        assert page.source_appearance_state()["a"]["label"] == expected
        assert page._source_appearance_editor.state()["a"]["label"] == expected
        drawn = [M.plain_text(label.text) for _sample, label in page._plot.legend.items]
        assert f"{expected} · n=5" in drawn and "Other source · n=4" in drawn
        assert widget.history_mark() == before_mark + 1
    pane.list.setFocus()                              # focus-out ends this typing run
    QtWidgets.QApplication.processEvents()
    assert widget._history.depth() == (1, 0)
    assert widget.state().text_style("legend.entry.a").is_default()
    assert widget.state().source_appearance['a']['label'] == "Apo <1>"
    assert page.source_appearance_state()["a"]["color"] == "#009e73"
    pd.testing.assert_frame_equal(page._plot_data(), before_data, check_exact=True)

    payload, tables = page.snapshot_state()
    assert payload["source_appearance"]["a"]["label"] == "Apo <1>"
    assert not any(key.startswith("legend.entry.")
                   for key in payload["appearance"].get("text", {}))
    assert widget.undo() and page.source_appearance_state() == names
    assert widget._history.depth() == (0, 1)
    assert widget.redo() and page.source_appearance_state()["a"]["label"] == "Apo <1>"

    restored = M.DistributionSummaryWindow()
    qtbot.addWidget(restored)
    restored.restore_state(payload, tables)
    assert restored.source_appearance_state() == payload["source_appearance"]
    assert "Apo <1> · n=5" in [M.plain_text(label.text)
                              for _sample, label in restored._plot.legend.items]
    pd.testing.assert_frame_equal(restored._plot_data(), before_data, check_exact=True)


def test_reset_view_sits_on_the_status_line_and_returns_the_pages_framing(explorer, page):
    H.press_update(explorer)
    status = page.statusBar()
    visible = status.findChildren(QtWidgets.QToolButton, "plotResetViewButton")
    assert len(visible) == 1 and visible[0].isVisible()
    framed = page._plot.viewRange()
    page._plot.setXRange(framed[0][0], framed[0][0] + (framed[0][1] - framed[0][0]) * 0.1, padding=0)
    QtWidgets.QApplication.processEvents()
    assert page._plot.viewRange()[0][1] < framed[0][1]
    H.click(visible[0], pos=QtCore.QPoint(8, visible[0].height() // 2))
    assert page._plot.viewRange()[0] == pytest.approx(framed[0], rel=1e-6)


@pytest.mark.parametrize("section", ("Data", "Analysis", "Appearance", "Export"))
def test_a_page_left_on_any_section_is_hosted_with_the_replaced_boxes_hidden(
        qapp, qtbot, section):
    page = M.DistributionSummaryWindow()
    page.set_sources([_source()])
    qtbot.addWidget(page)
    page._workflow.set_section(section)                 # standalone: shows that Section's widgets
    H.hosted(qtbot, page, KEY)
    for name in ("_data_section", "_analysis_section", "_appearance_section", "_export_section"):
        assert getattr(page, name).isHidden(), name     # each hid itself, not through a parent
    assert page._source_list.isVisible() and page._templates.parentWidget() is not None


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
    # the pages' explicit 650x480 minimum was budgeted for a page with its controls on it
    assert page.minimumWidth() <= 480 and page.minimumHeight() <= 360


def test_hosting_binding_and_tab_switches_compute_nothing(explorer, page, monkeypatch):
    calls = _computes(monkeypatch)
    explorer.refresh_header()
    explorer.toolbar().sync()
    explorer.inspector().set_tab(1)
    explorer.inspector().set_tab(0)
    explorer.show_home()
    explorer.raise_page(KEY)
    H.key(explorer, QtCore.Qt.Key.Key_2, H.CTRL)
    H.key(explorer, QtCore.Qt.Key.Key_1, H.CTRL)
    assert calls == [] and not page.has_data()


def _main_window(qtbot, monkeypatch, sources):
    from nanoscope.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.show()
    qtbot.waitExposed(window)
    monkeypatch.setattr(window, "_plots_sources", lambda: sources)
    return window


def test_a_full_save_and_reopen_restores_the_figure_look_and_the_stale_draft(
        qapp, qtbot, tmp_path, monkeypatch):
    sources = [_source(), _source("b", "Run B", (.003, .004, .004, .006))]
    source = _main_window(qtbot, monkeypatch, sources)
    source._open_distribution_summaries()
    QtWidgets.QApplication.processEvents()
    page = source._distribution_summary_window
    explorer = source._analysis_explorer
    figure_id = explorer.current_page_id()
    hosted = explorer.adapter(figure_id)
    assert hosted.page is page and hosted.kind == KEY and page.property("nanoscopeInspectorHosted")
    assert hosted.format.appearance is page._appearance_widget
    H.press_update(explorer)
    assert page.has_data() and explorer.toolbar().update_button.text() == "Update plot"
    page._appearance_widget.apply_look(_BUILTIN_PRESETS["7 pt Arial · Nature, Elsevier"])
    page._appearance_widget.title.textEdited.emit("Dwell ECDF")
    _pick_with_the_keyboard(page._metric_combo)                  # a draft the figure does not show
    assert explorer.toolbar().update_button.text() == "Update plot (1)"
    saved, applied, draft = page.appearance_state(), page.figure_applied_config(), page.config()
    assert saved["title"] == "Dwell ECDF" and saved["title_size"] == 7
    project = tmp_path / "hosted.nanoscope"
    assert source._write_project(str(project))
    source._dirty = False

    reopened = _main_window(qtbot, monkeypatch, sources)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    monkeypatch.setattr(M, "summarize_distribution",
                        lambda *a, **k: pytest.fail("restore computed a summary"))
    reopened._open_distribution_summaries()
    QtWidgets.QApplication.processEvents()
    again = reopened._distribution_summary_window
    assert again.property("nanoscopeInspectorHosted") is True
    assert again.appearance_state() == saved
    assert again.figure_applied_config() == applied and again.config() == draft
    explorer = reopened._analysis_explorer
    explorer.refresh_header()
    assert explorer.toolbar().bound_key() == figure_id
    assert explorer.toolbar().update_button.text() == "Update plot (1)"
    reopened._dirty = False
