"""Phase 1, G3: Comparison figures hosted in a real Analysis Explorer.

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
import shiboken6
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.core.sources import PlotSource
from nanoscope.gui import comparison_figures_panel as M
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS, PlotAppearance

from . import p1_hosting as H
from .test_plot_window_routes import _press_undo

KEY = "comparison_figures"
RETURN = QtCore.Qt.Key.Key_Return
DOWN = QtCore.Qt.Key.Key_Down
UP = QtCore.Qt.Key.Key_Up
SHELL = {"Save to Figure Library…", "Save figure bundle…", "Open Figure Library"}
NATURE = "7 pt Arial · Nature, Elsevier"
BOXES = {"facets": "_facet_controls", "pairs": "_pair_controls", "events": "_event_controls"}


def _source(key="first", label="Recording A", shift=0.0):
    frame = pd.DataFrame({"event_id": [5, 5, 9, 11, 14, 17],
        "dwell_s": [.001, .002, .003, .004, .005, .006],
        "depth_pA": [10. + shift, 20., 30., 40., 50., 60.],
        "amplitude_pA": [12., 25., 35., 42., 55., 64.],
        "ecd_pA_s": [.01, .04, .09, .16, .25, .36], "snr": [5., 8., 10., 12., 15., 16.],
        "accepted": [True, True, False, True, True, True]}, index=[2, 2, 4, 9, 10, 10])
    return PlotSource(key, label, frame, kind="manual", doc_key=key, current=(key == "first"))


@pytest.fixture
def page(qapp, qtbot):
    w = M.ComparisonFiguresWindow()
    qtbot.addWidget(w)
    w.set_sources([_source(), _source("second", "Recording B", 2.0)])
    return w


@pytest.fixture
def explorer(qtbot, page):
    return H.hosted(qtbot, page, KEY)


def _body(explorer, group):
    return explorer.inspector().pane(KEY).data_pane.group(group).body


def _pick_with_the_keyboard(widget, key=DOWN):
    widget.setFocus()
    QtWidgets.QApplication.processEvents()
    assert widget.hasFocus()
    H.key(widget, key)


def _settle():
    """A wrapping row needs a few layout rounds to find its height after a box is shown."""
    for _ in range(5):
        QtWidgets.QApplication.processEvents()


def _count(explorer) -> int:
    return explorer.adapter(KEY).pending().count


def _prepares(monkeypatch):
    from nanoscope.plot import comparison_figures
    calls = []
    real = comparison_figures.prepare_comparison
    monkeypatch.setattr(comparison_figures, "prepare_comparison",
                        lambda d, c: (calls.append(c["kind"]), real(d, c))[1])
    return calls


def test_the_groups_hold_the_pages_own_widgets_and_one_box_shows_per_type(explorer, page):
    adapter = explorer.adapter(KEY)
    assert adapter is page.inspector_adapter() and adapter.fallback is False
    assert [g.key for g in adapter.groups] == ["sources", "plot", "recipes"]
    assert [g.expanded for g in adapter.groups] == [True, True, False]
    for name in ("_source_list", "_accepted_only", "_data_note"):
        assert _body(explorer, "sources").isAncestorOf(getattr(page, name)), name
    plot = _body(explorer, "plot")
    for name in ("_view_combo", "_method_note", "_facet_controls", "_pair_controls",
                 "_event_controls"):
        assert plot.isAncestorOf(getattr(page, name)), name
    for box, names in (("_facet_controls", ("_x_combo", "_y_combo", "_columns", "_log_x", "_log_y")),
                       ("_pair_controls", ("_features", "_bins")),
                       ("_event_controls", ("_metric", "_points", "_show_box", "_event_log_y"))):
        for name in names:
            assert getattr(page, box).isAncestorOf(getattr(page, name)), (box, name)
    assert page._templates.parentWidget() is _body(explorer, "recipes")
    for name in ("_data_section", "_analysis_section", "_appearance_section", "_export_section"):
        assert getattr(page, name).isHidden(), name
    assert not page._workflow.isVisible()
    # _update_method_note still shows one box at a time, inside the group
    for kind, shown in BOXES.items():
        page._view_combo.setCurrentIndex(page._view_combo.findData(kind))
        QtWidgets.QApplication.processEvents()
        assert [name for name in BOXES.values() if getattr(page, name).isVisible()] == [shown]
    # the canvas bar stays in the page
    for name in ("_fit_button", "_zoom_in", "_zoom_out", "_warnings", "_status", "_view"):
        assert getattr(page, name).parentWidget() is not None
    assert page._status.isVisible() and page._view.isVisible()
    assert page.property("nanoscopeInspectorHosted") is True
    assert page.inspector_adapter() is adapter                  # idempotent
    assert "Prepare the figure" in explorer.toolbar().update_action.toolTip()
    spec = adapter.format
    assert spec.appearance is page._appearance_widget and spec.controllers() == []
    assert not spec.extras and page._source_appearance_editor.isHidden()
    # no navigation: no Reset view on the status line
    assert page.statusBar().findChildren(QtWidgets.QToolButton, "plotResetViewButton") == []
    # hidden but still wired
    page._workflow.primary_actions[0].click()
    assert page.has_data()


def test_reflow_invalidates_old_items_of_the_extra_rows_without_losing_controls(qtbot, page):
    """Rows outside form_rows still release their exact old layout items through PySide."""
    controls = [page._features, page._pair_note, page._log_x, page._log_y,
                page._show_box, page._event_log_y]
    held = []
    for widget in controls:
        outer = widget.parentWidget().layout()
        layouts = [outer, *outer.findChildren(QtWidgets.QLayout)]
        owners = [layout for layout in layouts if layout.indexOf(widget) >= 0]
        assert len(owners) == 1, widget.objectName()
        layout = owners[0]
        item = layout.itemAt(layout.indexOf(widget))
        assert item.widget() is widget
        held.append(item)
    explorer = H.hosted(qtbot, page, KEY)
    assert all(not shiboken6.isValid(item) for item in held)
    assert all(shiboken6.isValid(widget) and _body(explorer, "plot").isAncestorOf(widget)
               for widget in controls)


@pytest.mark.parametrize("name, kind", [("_accepted_only", "facets"), ("_log_x", "facets"),
                                        ("_log_y", "facets"), ("_event_log_y", "events")])
def test_a_click_in_the_middle_of_a_moved_check_box_still_toggles_it(explorer, page, name, kind):
    page._view_combo.setCurrentIndex(page._view_combo.findData(kind))
    _settle()
    box = getattr(page, name)
    assert box.isVisible() and box.width() < 220
    was = box.isChecked()
    H.click(box)
    assert box.isChecked() is (not was)


def test_the_event_box_check_box_wraps_when_the_column_is_narrow(explorer, page):
    box = page._show_box
    assert box.text() == "Overlay box summary (linear quartiles; 1.5 IQR)"      # the sentence is unchanged
    assert box.minimumSizeHint().width() < 150
    narrow, wide = box.heightForWidth(180), box.heightForWidth(600)
    assert narrow > wide >= box.sizeHint().height() - 4
    # drawn: the box and its words are painted (not blank), and a real click ticks it
    page._view_combo.setCurrentIndex(page._view_combo.findData("events"))
    _settle()
    image = box.grab().toImage()
    assert any(image.pixel(x, y) != image.pixel(0, 0)
               for x in range(0, image.width(), 3) for y in range(0, image.height(), 3))
    was = box.isChecked()
    H.click(box, pos=QtCore.QPoint(8, box.height() // 2))
    assert box.isChecked() is (not was)


def test_update_plot_counts_the_draft_changes_and_prepares_only_on_the_verb(
        explorer, page, monkeypatch):
    bar = explorer.toolbar()
    calls = _prepares(monkeypatch)
    assert bar.update_button.text() == "Update plot (1)"
    assert explorer.adapter(KEY).pending().reasons == ("not generated yet",)
    H.press_update(explorer)                                    # a real click
    assert calls == ["facets"] and page.has_data() and _count(explorer) == 0
    assert bar.update_button.text() == "Update plot" and not bar.update_button.property("pending")
    # controls only edit a draft: nothing is prepared; each differing key is one
    H.click(page._log_x, pos=QtCore.QPoint(8, page._log_x.height() // 2))
    assert page._log_x.isChecked() and calls == ["facets"] and _count(explorer) == 1
    assert explorer.adapter(KEY).pending().reasons == ("log x",)
    _pick_with_the_keyboard(page._columns, UP)
    assert _count(explorer) == 2 and bar.update_button.text() == "Update plot (2)"
    assert bar.update_button.property("pending") and "log x" in bar.update_action.toolTip()
    assert page._workflow.primary_actions[0].text() == "Generate ●"
    # a key that is not this figure type's own is still a differing draft key only if it moved
    H.click(page._log_x, pos=QtCore.QPoint(8, page._log_x.height() // 2))
    assert _count(explorer) == 1
    _pick_with_the_keyboard(page._columns, DOWN)
    assert all(page.config()[k] == page._applied_config[k] for k in page.config())
    assert page._stale and _count(explorer) == 1                # the floor: never 0 beside STALE
    assert bar.update_button.text() == "Update plot (1)"
    H.key(explorer, RETURN, H.CTRL)
    assert calls == ["facets", "facets"] and _count(explorer) == 0 and not page._stale


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


def test_a_failed_generate_keeps_the_old_figure_marked_stale(explorer, page, monkeypatch):
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2])))
    H.press_update(explorer)
    before = page._plot_data()
    page._source_list.item(0).setCheckState(QtCore.Qt.Unchecked)
    page._source_list.item(1).setCheckState(QtCore.Qt.Unchecked)      # nothing ticked: refused
    H.press_update(explorer)
    assert warned and "Select at least one event source" in warned[0]
    pd.testing.assert_frame_equal(page._plot_data(), before, check_exact=True)
    assert page._stale and _count(explorer) >= 1
    assert explorer.toolbar().update_button.property("pending")


def test_export_is_greyed_until_a_figure_exists_and_runs_the_pages_handlers(
        explorer, page, monkeypatch):
    menu = H.export_menu(explorer)
    entries = {a.text(): a for a in menu.actions() if not a.isSeparator() and a.text() not in SHELL}
    assert list(entries) == ["Export current figure…", "Export plotted values CSV…"]
    assert not any(a.isEnabled() for a in entries.values())
    menu.close()
    H.press_update(explorer)
    menu = H.export_menu(explorer)
    entries = {a.text(): a for a in menu.actions() if not a.isSeparator() and a.text() not in SHELL}
    assert all(a.isEnabled() for a in entries.values())
    menu.close()
    seen = []                       # the page's own handlers, up to the file dialog
    monkeypatch.setattr(M, "choose_live_plot_export", lambda *a, **k: seen.append("image"))
    monkeypatch.setattr(M, "ask_export_path", lambda *a, **k: seen.append("csv"))
    H.choose_export(explorer, "Export current figure…")
    H.choose_export(explorer, "Export plotted values CSV…")
    assert seen == ["image", "csv"]


def test_the_undo_key_works_with_the_data_tab_showing(explorer, page):
    """Comparison's keys sit on the page (``shortcut_parent=self``), not on the Appearance
    widget the inspector hides: an edit made on the Format tab is taken back from the Data tab."""
    H.press_update(explorer)
    widget = page._appearance_widget
    explorer.inspector().set_tab(1)
    QtWidgets.QApplication.processEvents()
    widget.title.setFocus()
    QtWidgets.QApplication.processEvents()
    QtTest.QTest.keyClicks(widget.title, "Pair figure")
    assert widget.state().title == "Pair figure" and page.appearance_state()["title"] == "Pair figure"
    depth = widget._history.depth()
    H.key(explorer, QtCore.Qt.Key.Key_1, H.CTRL)                # back to the Data & analysis tab
    assert explorer.inspector().current_tab() == 0
    assert not widget.isVisible()
    _press_undo(page)
    assert widget.state().title is None and "title" not in page.appearance_state()
    assert widget._history.depth() == (depth[0], 1)
    bar = explorer.toolbar()
    assert bar.redo_action.isEnabled()
    widget.apply_look(_BUILTIN_PRESETS[NATURE])
    assert bar.undo_action.isEnabled() and "Nature" in bar.undo_action.toolTip()
    steps = widget._history.depth()[0]
    H.press_reset_figure(explorer)
    assert widget._history.depth()[0] == steps + 1


def test_a_look_that_will_not_render_comes_back_failed_without_a_dialog_or_a_step(
        explorer, page, monkeypatch):
    H.press_update(explorer)
    look = _BUILTIN_PRESETS[NATURE]
    adapter, widget = explorer.adapter(KEY), page._appearance_widget
    # a look that renders: one step, nothing failed
    mark = adapter.history().mark
    ok = explorer.apply_look(look, scope="page")
    assert ok.applied == (KEY,) and ok.failed == () and adapter.history().mark == mark + 1
    assert page.appearance_state()["title_size"] == 7
    widget.undo()
    before_state, mark = page.appearance_state(), adapter.history().mark
    # the figure will not draw it: no modal opens, no step, the old figure stays
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2])))

    def refuse(*args, **kwargs):
        raise RuntimeError("renderer refused")
    monkeypatch.setattr(M.ComparisonFiguresWindow, "_make_figure_item", staticmethod(refuse))
    failed = explorer.apply_look(look, scope="all")
    assert warned == []
    assert failed.applied == () and failed.failed == ((KEY, "renderer refused"),)
    assert adapter.history().mark == mark and page.appearance_state() == before_state
    assert page._quiet is False and page._quiet_error == ""
    # a hand edit that will not draw is still reported in a dialog, as it always was
    widget.title.textEdited.emit("Typed")
    assert warned and "renderer refused" in warned[0]
    assert page.appearance_state() == before_state


def test_a_look_reaches_the_page_behind_another_tab_as_one_step(explorer, page):
    H.press_update(explorer)
    other = H.SyntheticPlotPage("iv")
    explorer.show_page("iv", other)
    explorer.raise_page("iv")
    QtWidgets.QApplication.processEvents()
    assert not page.isVisible() and explorer.inspector().current_key() == "iv"
    adapter, widget = explorer.adapter(KEY), page._appearance_widget
    behind = adapter.history().mark
    result = explorer.apply_look(_BUILTIN_PRESETS["Slides · 18 pt Arial"], scope="all")
    assert KEY in result.applied and not result.failed
    assert adapter.history().mark == behind + 1 and widget.state().title_size == 18
    assert page.appearance_state()["title_size"] == 18           # and the figure was redrawn with it
    explorer.raise_page(KEY)
    QtWidgets.QApplication.processEvents()
    assert explorer.toolbar().undo_action.isEnabled()


@pytest.mark.parametrize("section", ("Data", "Analysis", "Appearance", "Export"))
def test_a_page_left_on_any_section_is_hosted_with_the_replaced_boxes_hidden(
        qapp, qtbot, section):
    page = M.ComparisonFiguresWindow()
    page.set_sources([_source()])
    qtbot.addWidget(page)
    page._workflow.set_section(section)                 # standalone: shows that Section's widgets
    H.hosted(qtbot, page, KEY)
    for name in ("_data_section", "_analysis_section", "_appearance_section", "_export_section"):
        assert getattr(page, name).isHidden(), name     # each hid itself, not through a parent
    assert page._source_list.isVisible() and page._templates.parentWidget() is not None


def test_the_page_fits_the_explorers_minimum_size_for_every_figure_type(qtbot, page):
    explorer = H.hosted(qtbot, page, KEY, size=(840, 560))
    explorer.resize(840, 560)
    host = explorer.inspector()
    for group in host.adapter(KEY).groups:
        host.reveal(group.key)
    pane = host.pane(KEY)
    for kind in BOXES:
        page._view_combo.setCurrentIndex(page._view_combo.findData(kind))
        QtWidgets.QApplication.processEvents()
        assert page.minimumSizeHint().width() <= 500, kind
        for group in explorer.adapter(KEY).groups:
            assert pane.data_pane.group(group.key).minimumSizeHint().width() <= 290, (kind, group.key)
        assert pane.data_pane.horizontalScrollBar().maximum() == 0, kind
    H.press_update(explorer)
    # the page's own status line, canvas bar and warnings summary take 129 px of the 383 the
    # page has at 840x560: the canvas gets 254 (the design's 300 is not reachable here)
    assert page._view.width() >= 420 and page._view.height() >= 240
    assert explorer.width() == 840
    assert page.minimumWidth() <= 480 and page.minimumHeight() <= 360   # was 700x500


def test_hosting_binding_and_tab_switches_prepare_nothing(explorer, page, monkeypatch):
    calls = _prepares(monkeypatch)
    explorer.refresh_header()
    explorer.toolbar().sync()
    explorer.inspector().set_tab(1)
    explorer.inspector().set_tab(0)
    explorer.show_home()
    explorer.raise_page(KEY)
    H.key(explorer, QtCore.Qt.Key.Key_2, H.CTRL)
    H.key(explorer, QtCore.Qt.Key.Key_1, H.CTRL)
    assert calls == [] and not page.has_data()


def test_the_canvas_bar_still_works_in_the_page(explorer, page):
    H.press_update(explorer)
    H.click(page._zoom_in)
    scale = page._view.transform().m11()
    H.click(page._zoom_out)
    assert page._view.transform().m11() < scale
    H.click(page._fit_button)
    assert page._view._fitted


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
    sources = [_source(), _source("second", "Recording B", 2.0)]
    source = _main_window(qtbot, monkeypatch, sources)
    source._open_comparison_figures()
    QtWidgets.QApplication.processEvents()
    page = source._comparison_figures_window
    explorer = source._analysis_explorer
    key = source._figure_workspace.current_id()
    hosted = explorer.adapter(key)
    assert hosted.key == key and hosted.kind == KEY and hosted.page is page
    assert hosted.format is page.inspector_adapter().format
    assert page.property("nanoscopeInspectorHosted")
    _pick_with_the_keyboard(page._view_combo, DOWN)             # pairs
    H.press_update(explorer)
    assert page.has_data() and page.figure_applied_config()["kind"] == "pairs"
    page._appearance_widget.apply_look(_BUILTIN_PRESETS[NATURE])
    H.click(page._accepted_only, pos=QtCore.QPoint(8, page._accepted_only.height() // 2))
    assert explorer.toolbar().update_button.text() == "Update plot (1)"
    saved, applied, draft = page.appearance_state(), page.figure_applied_config(), page.config()
    assert saved["title_size"] == 7 and draft["accepted_only"] is False
    project = tmp_path / "hosted.nanoscope"
    assert source._write_project(str(project))
    source._dirty = False

    reopened = _main_window(qtbot, monkeypatch, sources)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    _prepares(monkeypatch)
    from nanoscope.plot import comparison_figures
    monkeypatch.setattr(comparison_figures, "prepare_comparison",
                        lambda *a, **k: pytest.fail("restore prepared a figure"))
    reopened._open_comparison_figures()
    QtWidgets.QApplication.processEvents()
    again = reopened._comparison_figures_window
    assert again.property("nanoscopeInspectorHosted") is True
    assert again.appearance_state() == saved
    assert again.figure_applied_config() == applied and again.config() == draft
    explorer = reopened._analysis_explorer
    explorer.refresh_header()
    assert reopened._figure_workspace.current_id() == key
    assert explorer.toolbar().bound_key() == key
    assert explorer.toolbar().update_button.text() == "Update plot (1)"
    reopened._dirty = False
