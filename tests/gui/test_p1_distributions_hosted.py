"""Phase 1, group G5: Distributions hosted in the Analysis Explorer's inspector.

``PlotsWindow.inspector_adapter()`` is the page's own adapter (not the Foundation's
fallback): its controls MOVE into the inspector's groups, once; the toolbar's Update plot is
``_run_plot`` with a count read from the gate's own flags; Undo and Redo are its whole
``PlotConfig`` history; Lasso select and its Mode label sit on the status line.

Real routes: a real Explorer, clicks and keys through ``window().windowHandle()``
(``p1_hosting``), real menus, a real save with ``_write_project`` and a FRESH MainWindow
opened with ``_open_project`` until ``projectRestored``. Tests assert what is DRAWN or SHOWN.
Standalone behaviour is the existing tests' (not edited). Synthetic data only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import TextStyle, figure_text_for
from nanoscope.gui.plot_appearance import _BUILTIN_PRESETS, PlotAppearance
from nanoscope.gui.plots_panel import PlotsWindow, StyleSettings, _LimitLabel

from . import p1_hosting as H
from .test_p0_reset_on_quantity_change import _user_picks
from .test_plots_legend_text import _depth as _legend_depth
from .test_plots_legend_text import _histogram, _right_click, _titles, _walk_to
from .test_plots_organization_and_selection import _source

KEY = "distributions"
RETURN = QtCore.Qt.Key.Key_Return
NATURE = "7 pt Arial · Nature, Elsevier"
SLIDES = "Slides · 18 pt Arial"


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _frame(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": np.abs(rng.normal(3e-4, 1e-4, n)) + 1e-5,
        "amplitude": np.abs(rng.normal(30, 5, n)), "frac_depth": np.abs(rng.normal(0.25, 0.05, n)),
        "snr": np.abs(rng.normal(9, 2, n)), "rms": np.abs(rng.normal(1.4, 0.3, n)),
        "accepted": np.ones(n, dtype=bool), "filter_limited": np.zeros(n, dtype=bool)})


def _page(*, armed: bool = True) -> PlotsWindow:
    page = PlotsWindow(Session())
    if not armed:
        page.auto_plot = False           # the window opens on the prompt, as a project does
    page.set_data({"run A": _frame(seed=1), "run B": _frame(seed=2)})
    return page


def _host(qtbot, *, armed: bool = True, size=(1280, 820)):
    page = _page(armed=armed)
    explorer = H.hosted(qtbot, page, KEY, size=size)
    return explorer, page


def _draws(page, monkeypatch) -> list:
    """Count the page's one recompute (``_armed_redraw``: every run gesture ends there)."""
    calls: list[int] = []
    real = page._armed_redraw
    monkeypatch.setattr(page, "_armed_redraw", lambda: (calls.append(1), real())[1])
    return calls


def _other_fit(page) -> str:
    return next(page._fit_combo.itemText(i) for i in range(page._fit_combo.count())
                if page._fit_combo.itemText(i) != page._fit_combo.currentText())


def _tick_click(page, row: int) -> None:
    """A real click on a source row's check box."""
    lst = page._source_list
    rect = lst.visualItemRect(lst.item(row))
    H.click(lst.viewport(), pos=QtCore.QPoint(rect.left() + 10, rect.center().y()))


def _frame_of(explorer, key: str):
    return explorer.inspector().pane(KEY).data_pane.group(key)


# ================================================================================================
# the adapter and the groups
# ================================================================================================

def test_the_page_has_its_own_adapter_and_the_groups_hold_its_own_widgets(qtbot):
    explorer, page = _host(qtbot)
    adapter = explorer.adapter(KEY)
    assert adapter is page.inspector_adapter() and not adapter.fallback
    assert [g.key for g in adapter.groups] == ["sources", "plot", "fit_display", "groups", "recipes"]
    assert [g.title for g in adapter.groups] == ["Sources", "Plot", "Fit display", "Groups", "Recipes"]
    holds = {"sources": (page._source_group, page._source_list, page._accepted_check),
             "plot": (page._plot_group, page._plot_type_combo, page._pages),
             "fit_display": (page._ci_band_check, page._residual_check),
             "groups": (page._features_group, page._cluster_group),
             "recipes": (page._template_combo, page._template_save_btn, page._template_rename_btn,
                         page._template_delete_btn, page._import_cfg_btn)}
    for key, widgets in holds.items():
        body = _frame_of(explorer, key).body
        for widget in widgets:
            assert body.isAncestorOf(widget), (key, widget)
    # the first two groups are open, the rest collapsed (7.1)
    assert [_frame_of(explorer, g.key).is_expanded() for g in adapter.groups] == \
        [True, True, False, False, False]


def test_fit_stays_inside_plot_and_the_fit_display_rows_left_the_style_group(qtbot):
    explorer, page = _host(qtbot)
    plot = _frame_of(explorer, "plot").body
    for fit_row in (page._fit_combo, page._fit_k, page._fit_k_auto, page._scatter_fit):
        assert plot.isAncestorOf(fit_row)
    assert _frame_of(explorer, "fit") is None
    display = _frame_of(explorer, "fit_display")
    assert display.body.isAncestorOf(page._ci_band_check) and display.body.isAncestorOf(
        page._residual_check)
    assert not page._style_group.isAncestorOf(page._ci_band_check)
    explorer.reveal("fit_display")
    QtWidgets.QApplication.processEvents()
    # the boxes still do their job from the group: the strip is drawn, then not
    assert not page._resid_widget.isVisible()
    H.click(page._residual_check, pos=QtCore.QPoint(8, page._residual_check.height() // 2))
    assert page._residual_check.isChecked() and page.config().style.show_residuals


def test_recipes_hold_the_template_rows_and_not_undo_redo_or_export(qtbot):
    explorer, page = _host(qtbot)
    explorer.reveal("recipes")
    QtWidgets.QApplication.processEvents()
    body = _frame_of(explorer, "recipes").body
    for shown in (page._template_combo, page._template_save_btn, page._template_rename_btn,
                  page._template_delete_btn, page._template_split_note, page._import_cfg_btn):
        assert body.isAncestorOf(shown) and shown.isVisible(), shown
    # Undo and Redo are the toolbar's, JSON export is in Export ▾ only
    for gone in (page._undo_btn, page._redo_btn, page._export_cfg_btn):
        assert not gone.isVisible() and not body.isAncestorOf(gone)
    # the ceiling sentence is hidden until a step was dropped (the host made every root
    # visible; the page puts it back), and shows in the group once one was
    assert not page._history_note.isVisible()
    page._history.dropped = 3
    page._sync_config_buttons()
    assert page._history_note.isVisible() and body.isAncestorOf(page._history_note)
    assert "3 oldest steps" in page._history_note.text()


def test_the_style_groups_other_widgets_are_named_for_the_format_tab_not_moved(qtbot):
    explorer, page = _host(qtbot)
    spec = explorer.adapter(KEY).format
    assert spec.appearance is page._appearance_widget
    assert spec.extras["figure"] == [page._font_size]
    assert not any(w is spec.appearance or w.isAncestorOf(spec.appearance)
                   for w in spec.extras["figure"])
    assert spec.extras["legend"] == [page._legend_hint, page._legend_reset_btn,
                                     page._legend_counts, page._legend_entries]
    assert spec.extras["caption"] == [page._prov_check]
    assert spec.extras["colour_map"] == [page._heatmap_cmap]
    # their labels travel with them (PlotAppearanceWidget's convention)
    assert page._font_size._nanoscope_row_label.text() == "Font size (body)"
    assert page._legend_entries._nanoscope_row_label.text() == "Legend entries"
    assert page._heatmap_cmap._nanoscope_row_label.text() == "Heatmap colormap"
    # the controller is the plot's, and "set here" is measured against the page's real defaults
    assert spec.controllers() == [figure_text_for(page._plot)]
    fresh = _page()
    assert spec.defaults() == PlotAppearance.from_mapping(fresh.appearance_state())
    assert spec.defaults() != PlotAppearance()          # grid on, dark background, ...


def test_the_hidden_set_is_hidden_and_still_wired(qtbot, monkeypatch):
    explorer, page = _host(qtbot)
    assert page._section_area.isHidden() and not page._workflow_toolbar.isVisible()
    for gone in (page._plot_btn, page._fit_apply_btn, page._scatter_fit_apply_btn,
                 page._save_btn, page._save_data_btn, page._axis_range_group):
        assert not gone.isVisible(), gone
    # the central splitter holds the plot alone, and it has the room
    assert page._plot_widget.width() > 700
    # ... and the objects are alive and wired: a hidden button's click and the hidden
    # action's trigger still run the plot, and the text still carries the ●
    draws = _draws(page, monkeypatch)
    page._plot_btn.click()
    page._run_action.trigger()
    assert draws == [1, 1]
    page._mark_stale("the plot type")
    assert page._plot_btn.text() == "Plot ●" and page._run_action.text() == "Plot / Apply ●"
    assert page._axis_range_group.findChildren(QtWidgets.QLineEdit)       # the store stays
    assert set(page._range_edits) == {"xmin", "xmax", "ymin", "ymax"}


def test_adoption_builds_little_and_the_groups_fit_the_column(qtbot):
    from nanoscope.gui.analysis_explorer import AnalysisExplorer

    page = _page()
    shell = AnalysisExplorer()
    qtbot.addWidget(shell)
    before = H.count_widgets(page) + H.count_widgets(shell.inspector())
    shell.add_page(KEY, page)
    added = H.count_widgets(page) + H.count_widgets(shell.inspector()) - before
    # five group frames and nothing else: no widget of the adapter's own, no Format panel
    assert added <= 36, added
    assert not shell.inspector().pane(KEY).format_pane._built
    explorer = H.hosted(qtbot, _page(), KEY, size=(840, 560))
    page = explorer.page(KEY)
    host = explorer.inspector()
    assert page.minimumSizeHint().width() <= 500
    assert page._plot_widget.width() >= 420 and page._plot_widget.height() >= 300
    for group in explorer.adapter(KEY).groups:
        host.reveal(group.key)
    QtWidgets.QApplication.processEvents()
    pane = host.pane(KEY)
    # every group open at the default width: nothing scrolls sideways
    assert pane.data_pane.horizontalScrollBar().maximum() == 0
    assert all(_frame_of(explorer, g.key).is_expanded() for g in explorer.adapter(KEY).groups)
    widths = {g.key: _frame_of(explorer, g.key).minimumSizeHint().width()
              for g in explorer.adapter(KEY).groups}
    for key in ("sources", "plot", "fit_display", "recipes"):
        assert widths[key] <= (296 if key == "plot" else 290), widths
    # (Groups holds ColumnChecklist's three-button row, 280 px wide on its own: 294 here, a
    # few pixels over the budget, which scrolls sideways only below the default width)
    assert widths["groups"] <= 296, widths
    # the moved group boxes are flat: the inspector's frame is their frame
    assert page._plot_group.isFlat() and page._source_group.isFlat()


def test_a_retained_page_scrubbed_by_reset_state_stays_hosted_and_a_removed_one_lets_go(qtbot):
    explorer, page = _host(qtbot)
    bar, host = explorer.toolbar(), explorer.inspector()
    page._ci_band_check.setChecked(True)
    assert bar.undo_action.isEnabled()
    page.reset_state()                                      # a project switch scrubs a retained page
    assert page.property("nanoscopeInspectorHosted") is True and page._hosted
    assert bar.update_button.text() == "Update plot (1)"    # nothing plotted: "not plotted yet"
    assert not bar.undo_action.isEnabled()                  # the stack went with the data
    assert _frame_of(explorer, "fit_display").body.isAncestorOf(page._ci_band_check)
    assert not page._ci_band_check.isChecked()
    # removing the page tells it: the inspector's reveal is dropped and the Section API is
    # its own again (it would otherwise call into an inspector that no longer lists it)
    removed = explorer.remove_page(KEY)
    assert removed is page and page._inspector_reveal is None and not page._hosted
    page._show_control_section("Analysis")                  # runs the standalone branch quietly
    assert not page._plot_group.isHidden() and page._source_group.isHidden()


# ================================================================================================
# the verb: Update plot, ⌘Return and the count
# ================================================================================================

def test_command_return_runs_the_plot_once_and_the_shell_owns_the_key(qtbot, monkeypatch):
    explorer, page = _host(qtbot)
    draws = _draws(page, monkeypatch)
    assert page._run_action.shortcut().isEmpty()
    H.key(explorer, RETURN, H.CTRL)
    assert draws == [1]
    H.key(explorer, QtCore.Qt.Key.Key_Enter, H.CTRL)
    assert draws == [1, 1]
    H.press_update(explorer)
    assert draws == [1, 1, 1]


def test_update_plot_before_the_first_plot_counts_one_and_a_press_plots(qtbot):
    explorer, page = _host(qtbot, armed=False)
    bar = explorer.toolbar()
    adapter = explorer.adapter(KEY)
    assert not page._plot_armed
    assert adapter.pending().count == 1 and adapter.pending().reasons == ("not plotted yet",)
    assert bar.update_button.text() == "Update plot (1)"
    assert "not plotted yet" in bar.update_action.toolTip()
    H.press_update(explorer)
    assert page._plot_armed and adapter.pending().count == 0
    assert bar.update_button.text() == "Update plot"
    assert page._hist_items, "the histogram is drawn"


def test_the_count_is_stale_reasons_plus_a_proposed_fit_and_a_press_commits_all(qtbot):
    explorer, page = _host(qtbot)
    bar, adapter = explorer.toolbar(), explorer.adapter(KEY)
    assert adapter.pending().count == 0
    _tick_click(page, 1)                                     # untick run B: the data selection
    assert page._stale and adapter.pending().reasons == ("the data selection",)
    assert bar.update_button.text() == "Update plot (1)"
    _user_picks(page._plot_type_combo, "Scatter")
    assert adapter.pending().reasons == ("the data selection", "the plot type")
    _user_picks(page._plot_type_combo, "Histogram")
    other = _other_fit(page)
    _user_picks(page._fit_combo, other)                      # a proposal, not applied
    assert page._applied_fit != other
    assert adapter.pending().count == 3 and adapter.pending().reasons[-1] == "fit model"
    assert bar.update_button.text() == "Update plot (3)"
    assert "fit model" in bar.update_action.toolTip()
    # the page's own stale banner names the verb the user can see
    assert page._plot_state.isVisible() or page._plot_state.text()
    assert "Update plot (⌘Return)" in page._plot_state.text()
    H.press_update(explorer)                                  # one draw commits all of it
    assert page._applied_fit == other and not page._stale
    assert adapter.pending().count == 0 and adapter.pending().reasons == ()
    assert bar.update_button.text() == "Update plot"
    assert page._fit_items, "the proposed fit is now drawn"


def test_the_toolbar_hears_the_gate_through_pending_changed_and_nothing_else(qtbot):
    """Update plot's count follows ``pendingChanged``; no ``workspaceChanged`` or
    ``refresh_header`` is in play (the state is moved, then the gate announces itself)."""
    explorer, page = _host(qtbot)
    bar = explorer.toolbar()
    assert bar.update_button.text() == "Update plot"
    page._stale, page._stale_reason = True, "the plot type"
    page._stale_reasons = ["the plot type"]
    page._sync_run_state()                      # the page's own gate sync, what every site calls
    assert bar.update_button.text() == "Update plot (1)"
    assert "the plot type" in bar.update_action.toolTip()
    page._stale, page._stale_reason, page._stale_reasons = False, "", []
    page._sync_run_state()
    assert bar.update_button.text() == "Update plot"


def test_the_stale_reasons_are_kept_without_repeats_and_cleared_at_every_site(qtbot):
    page = _page()
    assert page._stale_reasons == []
    page._mark_stale("the plot type")
    page._mark_stale("the plot type")
    page._mark_stale("an axis column")
    assert page._stale_reasons == ["the plot type", "an axis column"]
    assert page._stale_reason == "an axis column"
    page._armed_redraw()                                     # a run clears them
    assert page._stale_reasons == [] and page._stale_reason == ""
    page._mark_stale("the data selection")
    page._show_plot_prompt()                                  # the empty canvas clears them
    assert page._stale_reasons == []
    page._plot_armed = True
    page._mark_stale("the KDE overlay")
    page.reset_state()                                        # a blanked window clears them
    assert page._stale_reasons == [] and not page._plot_armed


def test_the_restored_snapshot_site_sets_one_reason_and_a_plain_one_clears(qtbot):
    """A restored picture that declined to estimate the KDE overlay is stale for that one
    reason (not for whatever was stale before the snapshot)."""
    page = _page()
    page._mark_stale("the plot type")
    page._kde_pending_after_snapshot = False
    page._render_workspace_snapshot()
    assert page._stale_reasons == [] and not page._stale
    page._mark_stale("the data selection")
    original = page._redraw

    def declined():
        original()
        page._kde_pending_after_snapshot = True
    page._redraw = declined
    page._render_workspace_snapshot()
    assert page._stale and page._stale_reasons == ["the KDE overlay"]


def test_pending_changed_and_history_changed_are_emitted(qtbot):
    page = _page()
    pending = QtTest.QSignalSpy(page.pendingChanged)
    history = QtTest.QSignalSpy(page.historyChanged)
    page._mark_stale("the plot type")
    assert pending.count() >= 1
    before = pending.count()
    page._armed_redraw()
    assert pending.count() > before
    # a step, an undo and a redo each announce (undo_config / redo_config emitted nothing before)
    page._ci_band_check.setChecked(True)
    n = history.count()
    assert n >= 1
    assert page.undo_config()
    assert history.count() > n
    n = history.count()
    assert page.redo_config()
    assert history.count() > n


# ================================================================================================
# history: the toolbar mirrors the page's own Undo, whole PlotConfig
# ================================================================================================

def test_undo_and_redo_on_the_toolbar_are_the_pages_config_history(qtbot):
    explorer, page = _host(qtbot)
    bar = explorer.toolbar()
    assert not bar.undo_action.isEnabled()
    explorer.reveal("fit_display")
    QtWidgets.QApplication.processEvents()
    H.click(page._residual_check, pos=QtCore.QPoint(8, page._residual_check.height() // 2))
    assert page.config().style.show_residuals
    assert bar.undo_action.isEnabled() and bar.undo_action.toolTip() == page._undo_btn.toolTip()
    H.press_undo(explorer)                                     # a real click on the toolbar
    assert not page.config().style.show_residuals and not page._residual_check.isChecked()
    assert bar.redo_action.isEnabled() and not bar.undo_action.isEnabled()
    H.press_redo(explorer)
    assert page.config().style.show_residuals and page._residual_check.isChecked()


def test_the_toolbar_hears_an_undo_that_the_page_did_itself(qtbot):
    """``undo_config`` emitted nothing; the toolbar (and the Format tab) read Undo's state
    from ``historyChanged`` now, with no ``refresh_header`` in between."""
    explorer, page = _host(qtbot)
    bar = explorer.toolbar()
    page._ci_band_check.setChecked(True)
    assert bar.undo_action.isEnabled()
    assert page.undo_config()
    assert not bar.undo_action.isEnabled() and bar.redo_action.isEnabled()
    assert page.redo_config()
    assert bar.undo_action.isEnabled() and not bar.redo_action.isEnabled()


def test_the_history_walk_reaches_a_control_of_each_group(qtbot):
    explorer, page = _host(qtbot)
    seen = []
    # Sources: a real click on a tick
    base = page._history.depth()[0]
    ticked = page._source_list.item(1).checkState()
    _tick_click(page, 1)
    seen.append(("sources", page._history.depth()[0] - base))
    assert page._source_list.item(1).checkState() != ticked
    assert page.undo_config() and page._source_list.item(1).checkState() == ticked
    # Plot: the bin count, by the keyboard
    explorer.reveal("plot")
    QtWidgets.QApplication.processEvents()
    bins = page._bins
    before = bins.value()
    bins.setFocus()
    QtWidgets.QApplication.processEvents()
    base = page._history.depth()[0]
    H.key(bins, QtCore.Qt.Key.Key_Up)
    seen.append(("plot", page._history.depth()[0] - base))
    assert bins.value() == before + 1
    assert page.undo_config() and bins.value() == before
    # Fit display: a real click
    explorer.reveal("fit_display")
    QtWidgets.QApplication.processEvents()
    base = page._history.depth()[0]
    H.click(page._ci_band_check, pos=QtCore.QPoint(8, page._ci_band_check.height() // 2))
    seen.append(("fit_display", page._history.depth()[0] - base))
    assert page._ci_band_check.isChecked()
    assert page.undo_config() and not page._ci_band_check.isChecked()
    # Groups: the clustering method
    explorer.reveal("groups")
    QtWidgets.QApplication.processEvents()
    method = page._method_combo
    other = next(method.itemText(i) for i in range(method.count())
                 if method.itemText(i) != method.currentText())
    was = method.currentText()
    base = page._history.depth()[0]
    _user_picks(method, other)
    seen.append(("groups", page._history.depth()[0] - base))
    assert page.undo_config() and method.currentText() == was
    # Format-tab extras (the Provenance caption): hooked on the widget, so moving changes nothing
    base = page._history.depth()[0]
    page._prov_check.click()
    seen.append(("caption", page._history.depth()[0] - base))
    assert page.undo_config() and not page._prov_check.isChecked()
    assert seen == [("sources", 1), ("plot", 1), ("fit_display", 1), ("groups", 1), ("caption", 1)]


def test_undo_reaches_an_edit_made_in_the_figure_from_either_inspector_tab(qtbot):
    explorer, page = _host(qtbot)
    host = explorer.inspector()
    controller = figure_text_for(page._plot)
    for tab in (0, 1):
        host.set_tab(tab)
        before = page.config().style.text_style("axis.left.label")
        base = page._history.depth()[0]
        controller.host.apply_text("axis.left.label", replace(before, bold=not before.bold))
        assert page._history.depth()[0] == base + 1
        H.key(explorer, QtCore.Qt.Key.Key_Z, H.CTRL)
        assert page._history.depth()[0] == base, f"tab {tab}"
        assert page.config().style.text_style("axis.left.label") == before


def test_history_mark_counts_steps_a_run_of_typing_and_goes_on_growing_at_the_ceiling(qtbot):
    page = _page()
    start = page.history_mark()
    page._ci_band_check.setChecked(True)
    assert page.history_mark() == start + 1
    page._residual_check.setChecked(True)
    assert page.history_mark() == start + 2
    assert page.undo_config() and page.history_mark() == start + 1
    assert page.redo_config() and page.history_mark() == start + 2
    # a run of typing still open counts as one (Undo is offered for it)
    page._title_edit.setText("Fig")
    assert page.history_mark() == start + 3
    page._history_commit()
    assert page.history_mark() == start + 3
    # at the ceiling the stack stops growing and the mark does not stop
    page._history._depth = page._history.depth()[0]
    marks = []
    for on in (False, True, False, True):
        page._ci_band_check.setChecked(on)
        marks.append(page.history_mark())
    assert marks == [start + 4, start + 5, start + 6, start + 7]


def test_the_adapters_history_state_is_the_pages_tooltips_and_mark(qtbot):
    explorer, page = _host(qtbot)
    adapter = explorer.adapter(KEY)
    assert adapter.history().undo_text == "Nothing to undo yet."
    page._ci_band_check.setChecked(True)
    state = adapter.history()
    assert state.can_undo and state.undo_text == page._undo_btn.toolTip()
    assert state.mark == page.history_mark() > 0
    assert "1 to step back" in state.undo_text


# ================================================================================================
# look, reset, style
# ================================================================================================

def test_a_built_in_look_and_reset_figure_are_one_config_step_each(qtbot):
    explorer, page = _host(qtbot)
    base = page._history.depth()[0]
    result = explorer.apply_look(_BUILTIN_PRESETS[NATURE])
    assert KEY in result.applied and not result.failed
    assert page._history.depth()[0] == base + 1
    style = page.config().style
    assert style.title_font_size_pt == 7 and style.axis_label_font_size_pt == 7
    assert explorer.toolbar().undo_action.isEnabled()
    H.press_reset_figure(explorer)
    assert page._history.depth()[0] == base + 2
    assert page.config().style.title_font_size_pt == 0           # back to "follow the body"
    H.press_undo(explorer)
    assert page.config().style.title_font_size_pt == 7
    H.press_undo(explorer)
    assert page._history.depth()[0] == base


def test_a_look_reaches_the_page_behind_another_tab_as_one_step(qtbot):
    explorer, page = _host(qtbot)
    explorer.show_page("iv", H.SyntheticPlotPage("iv"))
    explorer.raise_page("iv")
    QtWidgets.QApplication.processEvents()
    assert not page.isVisible()
    base = page._history.depth()[0]
    result = explorer.apply_look(_BUILTIN_PRESETS[SLIDES], scope="all")
    assert KEY in result.applied
    assert page._history.depth()[0] == base + 1
    assert page.config().style.title_font_size_pt == 18


# ================================================================================================
# Export, Reset view, Lasso and Mode on the status line
# ================================================================================================

def test_export_menu_is_the_right_click_menu_and_json_is_a_recipe_entry(qtbot, monkeypatch):
    from nanoscope.gui.plot_interaction import app_menu_for

    explorer, page = _host(qtbot)
    menu = H.export_menu(explorer)
    shown = [a.text() for a in menu.actions() if not a.isSeparator()]
    menu.close()
    assert shown[:3] == ["Export current figure…", "Export plot data…", "Export configuration JSON…"]
    block: list[str] = []
    for action in app_menu_for(page._plot).actions():
        block = [] if action.isSeparator() else block + [action.text()]
    assert block == shown[:3]
    ran = []
    monkeypatch.setattr(page, "export_config_json", lambda *a, **k: ran.append("json"))
    monkeypatch.setattr(page, "save_image", lambda *a, **k: ran.append("figure"))
    monkeypatch.setattr(page, "export_data", lambda *a, **k: ran.append("data"))
    for text in shown[:3]:
        H.choose_export(explorer, text)
    assert ran == ["figure", "data", "json"]
    kinds = [e.kind for e in explorer.adapter(KEY).exports()]
    assert kinds == ["figure", "data", "recipe"]


def test_reset_view_sits_on_the_status_line_and_the_old_button_is_renamed(qtbot):
    explorer, page = _host(qtbot)
    status = page.statusBar()
    visible = status.findChildren(QtWidgets.QToolButton, "plotResetViewButton")
    assert len(visible) == 1 and visible[0].isVisible()
    legacy = page.findChildren(QtWidgets.QAbstractButton, "plotResetViewButtonLegacy")
    assert len(legacy) == 1 and not legacy[0].isVisible()
    page._plot.setXRange(-0.5, 0.5, padding=0)
    QtWidgets.QApplication.processEvents()
    assert page._plot.viewRange()[0] == pytest.approx([-0.5, 0.5])
    H.click(visible[0], pos=QtCore.QPoint(8, visible[0].height() // 2))
    lo, hi = page._plot.viewRange()[0]
    assert (lo, hi) != pytest.approx((-0.5, 0.5))                 # the page's own framing is back


def _scatter_page(qtbot, size=(1280, 820)):
    frame = pd.DataFrame({
        "event_id": [17, 23, 41, 59, 71, 83], "dwell_s": [1., 10., 100., 5., 20., 1000.],
        "depth_pA": [10., 100., 1000., 100., 40., 1000.], "accepted": [True] * 6})
    page = PlotsWindow(Session())
    page.set_data([_source("A", frame)])
    page.set_plot_type("Scatter")
    page.set_scatter_columns("dwell_s", "depth_pA")
    page.set_scatter_log(True, True)
    explorer = H.hosted(qtbot, page, KEY, size=size)
    page._plot.setRange(xRange=(-.3, 3.3), yRange=(.7, 3.3), padding=0)
    QtWidgets.QApplication.processEvents()
    return explorer, page


def test_lasso_select_and_its_mode_label_sit_on_the_status_line_and_still_select(qtbot):
    explorer, page = _scatter_page(qtbot)
    status = page.statusBar()
    buttons = status.findChildren(QtWidgets.QToolButton, "plotCanvasActionButton")
    assert len(buttons) == 1 and buttons[0].defaultAction() is page._lasso_action
    assert buttons[0].isVisible() and page._selection_mode_label.isVisible()
    assert page._selection_mode_label.parentWidget() is status
    assert page._lasso_action.isEnabled() and "Pan/zoom" in page._selection_mode_label.text()
    selected = QtTest.QSignalSpy(page.eventRowsSelected)
    H.click(buttons[0])                                         # the real button
    assert page._lasso.enabled and "Lasso selection" in page._selection_mode_label.text()
    item = page._event_point_items["A"][0]
    spot = item.mapToScene(item.points()[1].pos())
    polygon = QtGui.QPolygonF([QtCore.QPointF(spot.x() + dx, spot.y() + dy)
                               for dx, dy in ((-12, -12), (12, -12), (12, 12), (-12, 12), (-12, -12))])
    viewport = page._plot_widget.viewport()
    points = [page._plot_widget.mapFromScene(p) for p in polygon]
    qtbot.mousePress(viewport, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, points[0])
    for point in points[1:]:
        qtbot.mouseMove(viewport, point)
    qtbot.mouseRelease(viewport, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, points[-1])
    assert selected.count() == 1 and selected.at(0) == ["A", (23,)]
    # a plot type with no events to pick disables the tool, and the button follows
    page.set_plot_type("Histogram")
    assert not page._lasso_action.isEnabled() and not buttons[0].isEnabled()


# ================================================================================================
# Section shim, axes, scrolling, in-figure text
# ================================================================================================

def test_a_section_name_reveals_the_inspector_instead_of_hiding_widgets(qtbot):
    explorer, page = _host(qtbot)
    host = explorer.inspector()
    page._show_control_section("Appearance")
    assert host.current_tab() == 1
    page._show_control_section("Export")                          # "Export" is Recipes now
    assert host.current_tab() == 0 and _frame_of(explorer, "recipes").is_expanded()
    page._show_control_section("Analysis")
    assert _frame_of(explorer, "plot").is_expanded()
    # nothing was shown or hidden by the call: every group's roots are as they were
    assert page._source_group.isVisible() and page._plot_group.isVisible()
    assert page._section_combo.currentText() == "Analysis"
    page._show_control_section("Nonsense")                        # asked for nothing before either
    assert host.current_tab() == 0
    page._show_control_section(2)                                  # an index, as the combo sends
    assert host.current_tab() == 1


def test_a_collapsed_plot_group_does_not_take_the_log_boxes_away_from_the_axis_panel(qtbot):
    explorer, page = _host(qtbot)
    controller = figure_text_for(page._plot)
    assert controller.host.axis_state("bottom").log_available
    _frame_of(explorer, "plot").set_expanded(False)
    QtWidgets.QApplication.processEvents()
    assert not page._hist_logx.isVisible()                         # collapsed: not on screen
    state = controller.host.axis_state("bottom")
    assert state.log_available                                     # ... but the plot has them
    controller._apply_axis("bottom", replace(state, log=True))
    assert page._hist_logx.isChecked()


def test_the_axis_panels_door_writes_the_range_boxes_as_one_step_and_a_column_change_obeys_the_reset_rule(
        qtbot):
    explorer, page = _host(qtbot)
    controller = figure_text_for(page._plot)
    base = page._history.depth()[0]
    state = controller.host.axis_state("bottom")
    controller._apply_axis("bottom", replace(state, range=(1.0, 4.0)))
    assert page._history.depth()[0] == base + 1
    assert page._range_edits["xmin"].text() == "1" and page._range_edits["xmax"].text() == "4"
    # the boxes are in the axis's display unit (ms); the pin is stored in seconds
    assert page.config().axes.manual_x == pytest.approx((0.001, 0.004))
    assert page._plot.getViewBox().viewRange()[0] == pytest.approx([0.001, 0.004])
    # the same pin on a standalone window and the same column change: the same result
    alone = _page()
    alone.show()
    figure_text_for(alone._plot).host.apply_axis(
        "bottom", replace(figure_text_for(alone._plot).host.axis_state("bottom"), range=(1.0, 4.0)))
    other = next(c for c in ("amplitude", "frac_depth") if c != page._hist_col.currentText())
    _user_picks(page._hist_col, other)
    _user_picks(alone._hist_col, other)
    assert page.config().axes == alone.config().axes
    assert page._reset_said == alone._reset_said


def test_no_gesture_scrolls_the_inspector(qtbot):
    explorer, page = _host(qtbot, size=(840, 560))
    host = explorer.inspector()
    host.reveal("plot")
    QtWidgets.QApplication.processEvents()
    bar = host.pane(KEY).data_pane.verticalScrollBar()
    assert bar.maximum() > 0
    parked = bar.maximum() // 2
    bar.setValue(parked)
    controller = figure_text_for(page._plot)
    explorer.apply_look(_BUILTIN_PRESETS[NATURE])
    H.press_update(explorer)
    H.press_undo(explorer)
    H.press_redo(explorer)
    H.press_reset_figure(explorer)
    controller.host.apply_text("axis.left.label", TextStyle(italic=True))
    H.key(explorer, RETURN, H.CTRL)
    H.key(explorer, QtCore.Qt.Key.Key_Z, H.CTRL)
    page._mark_stale("the plot type")
    page.undo_config()
    assert bar.value() == parked


def test_the_amber_banner_names_the_verb_of_the_page_it_is_on(qtbot):
    alone = _page()
    alone.show()
    alone._mark_stale("the plot type")
    assert alone._plot_state.text().endswith("Press Plot to redraw.")
    explorer, page = _host(qtbot)
    page._mark_stale("the plot type")
    assert page._plot_state.text().endswith("Press Update plot (⌘Return) to redraw.")
    assert page._plot_state.isVisibleTo(page)


def test_a_text_run_is_one_undo_step_where_a_single_edit_is_too(qtbot):
    page = _page()
    page.show()
    host = figure_text_for(page._plot).host
    key = "axis.left.ticks"
    before = page.config().style.text_style(key)
    base = page._history.depth()[0]
    for size in (7.0, 8.0, 9.0, 10.0, 11.0):                       # five clicks of a size spin
        host.apply_text_run(key, replace(page.config().style.text_style(key), size_pt=size))
    assert page.config().style.text_style(key).size_pt == 11.0
    assert page.undo_config()                                        # Undo closes the run first
    assert page._history.depth()[0] == base                          # ... and takes ALL of it back
    assert page.config().style.text_style(key) == before
    assert page.redo_config()
    assert page.config().style.text_style(key).size_pt == 11.0
    # a pause closes the step by itself: one step, then a single edit is its own
    page2 = _page()
    host2 = figure_text_for(page2._plot).host
    base2 = page2._history.depth()[0]
    for size in (7.0, 8.0, 9.0):
        host2.apply_text_run(key, replace(page2.config().style.text_style(key), size_pt=size))
    assert page2._history.depth()[0] == base2                       # still open
    qtbot.waitUntil(lambda: page2._history.depth()[0] == base2 + 1, timeout=3000)
    host2.apply_text(key, replace(page2.config().style.text_style(key), bold=True))
    assert page2._history.depth()[0] == base2 + 2                   # a single edit: at once


def test_a_text_run_changes_what_is_drawn_and_a_hidden_run_is_still_one_step(qtbot):
    page = _page()
    page.show()
    host = figure_text_for(page._plot).host
    key = "axis.bottom.label"
    for size in (12.0, 14.0, 16.0):
        host.apply_text_run(key, replace(page.config().style.text_style(key), size_pt=size))
    assert "16" in str(page._plot.getAxis("bottom").labelStyle.get("font-size", ""))


def test_format_axis_reset_preserves_words_and_other_axis_and_is_one_undo_step(qtbot):
    page = _page()
    qtbot.addWidget(page)
    host = figure_text_for(page._plot).host
    host.apply_axis("bottom", replace(host.axis_state("bottom"), range=(1.0, 4.0)))
    host.apply_axis("left", replace(host.axis_state("left"), range=(0.0, 50.0)))
    host.apply_text("axis.bottom.label", TextStyle(text="My dwell axis", bold=True))
    host.apply_text("axis.bottom.ticks", TextStyle(size_pt=14.0, italic=True))
    before = page.config()
    depth = page._history.depth()[0]
    seen = []
    page.historyChanged.connect(lambda: seen.append(page.config()))
    host.apply_format_styles({
        "axis.bottom.label": TextStyle(text="My dwell axis"),
        "axis.bottom.ticks": TextStyle(),
    }, auto_range_of="bottom")
    after = page.config()
    assert after.style.x_label == "My dwell axis"
    assert after.style.text_style("axis.bottom.label").bold is None
    assert after.style.text_style("axis.bottom.ticks").is_default()
    assert after.axes.manual_x is None and after.axes.x_min == after.axes.x_max == ""
    assert after.axes.manual_y == before.axes.manual_y
    assert after.axes.y_min == before.axes.y_min and after.axes.y_max == before.axes.y_max
    assert replace(after, style=before.style, axes=before.axes) == before
    assert page._history.depth()[0] == depth + 1
    assert seen and all(cfg == before or cfg == after for cfg in seen)
    assert page.undo_config() and page.config() == before
    assert page.redo_config() and page.config() == after


def test_format_paste_finishes_a_prior_text_run_without_merging_its_undo(qtbot):
    page = _page()
    qtbot.addWidget(page)
    host = figure_text_for(page._plot).host
    initial = page.config()
    depth = page._history.depth()[0]
    host.apply_text_run("title", TextStyle(text="My title", size_pt=14.0))
    typed = page.config()
    assert page._history.depth()[0] == depth
    host.apply_format_styles({"axis.left.ticks": TextStyle(size_pt=12.0)})
    assert page._history.depth()[0] == depth + 2
    assert page.undo_config() and page.config() == typed
    assert page.undo_config() and page.config() == initial


def test_spin_runs_on_two_elements_and_a_discrete_edit_are_separate_steps(qtbot):
    page = _page()
    qtbot.addWidget(page)
    host = figure_text_for(page._plot).host
    before = page.config()
    host.apply_text_run("axis.bottom.ticks", TextStyle(size_pt=12.0))
    first = page.config()
    host.apply_text_run("axis.left.ticks", TextStyle(size_pt=13.0))
    second = page.config()
    host.apply_text("axis.left.ticks", TextStyle(size_pt=13.0, bold=True))
    assert page.undo_config() and page.config() == second
    assert page.undo_config() and page.config() == first
    assert page.undo_config() and page.config() == before


def test_format_focus_out_commits_the_spin_run_without_waiting(qtbot):
    page = _page()
    qtbot.addWidget(page)
    host = figure_text_for(page._plot).host
    depth = page._history.depth()[0]
    host.apply_text_run("axis.bottom.ticks", TextStyle(size_pt=12.0))
    assert page._history_timer.isActive() and page._history.depth()[0] == depth
    host.commit_text_run()
    assert not page._history_timer.isActive() and page._history.depth()[0] == depth + 1


def test_style_application_leaves_a_pending_text_edit_as_its_own_undo(qtbot):
    explorer, page = _host(qtbot)
    host = figure_text_for(page._plot).host
    initial = page.config()
    host.apply_text_run("title", TextStyle(text="My pending title"))
    typed = page.config()
    assert KEY in explorer.apply_look(_BUILTIN_PRESETS[NATURE]).applied
    assert page.undo_config() and page.config() == typed
    assert page.undo_config() and page.config() == initial


def test_invalid_format_axis_is_rejected_before_any_state_is_changed(qtbot):
    page = _page()
    qtbot.addWidget(page)
    host = figure_text_for(page._plot).host
    before, depth = page.config(), page._history.depth()
    with pytest.raises(ValueError, match="Unknown axis"):
        host.apply_format_styles({"title": TextStyle(text="lost")}, auto_range_of="right")
    assert page.config() == before and page._history.depth() == depth


# ================================================================================================
# the detection-limit words
# ================================================================================================

def _click(w, item, button) -> None:
    view = w._plot_widget
    viewport = view.viewport()
    pos = view.mapFromScene(item.sceneBoundingRect().center())
    QtTest.QTest.mouseMove(viewport, pos)
    local, screen = QtCore.QPointF(pos), QtCore.QPointF(viewport.mapToGlobal(pos))
    for kind, held in ((QtCore.QEvent.Type.MouseButtonPress, button),
                       (QtCore.QEvent.Type.MouseButtonRelease, QtCore.Qt.MouseButton.NoButton)):
        QtWidgets.QApplication.sendEvent(
            viewport, QtGui.QMouseEvent(kind, local, local, screen, button, held,
                                        QtCore.Qt.KeyboardModifier.NoModifier))
        QtWidgets.QApplication.processEvents()


def test_the_detection_limit_label_announces_a_click_before_dispatching_it(qtbot):
    page = _histogram(qtbot)
    label = page._limit_line.label
    assert isinstance(label, _LimitLabel) and label.isVisible()
    selected, edits = [], []
    label.selectRequested.connect(lambda: selected.append(1))
    label.editRequested.connect(lambda: edits.append(1))
    _walk_to(page, label)
    _click(page, label, QtCore.Qt.MouseButton.LeftButton)
    assert selected == [1]                                         # a single click selects ...
    menu = _right_click(page, label)                               # ... a right click selects too
    assert selected == [1, 1]
    assert _titles(menu)[:2] == ["Edit detection-limit label…", "Text style…"]   # and still raises its menu
    menu.close()
    # nothing the click announced moved the history
    assert _legend_depth(page) == page._history.depth()[0]


# ================================================================================================
# a full save and reopen, with the page hosted
# ================================================================================================

def test_a_full_save_and_reopen_restores_the_page_hosted(qapp, qtbot, tmp_path, monkeypatch):
    from .test_plots_reopen_figure import _recordings, _reopen, _settle
    from nanoscope.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self, *a, **k: None)
    window, plots = _recordings(qtbot, tmp_path)
    explorer = window._analysis_explorer
    assert plots.property("nanoscopeInspectorHosted") is True
    live_key = window._figure_workspace.id_for_page(plots)
    assert explorer.adapter(live_key).page is plots and not explorer.adapter(live_key).fallback
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    plots.set_title("Pore A")
    explorer.apply_look(_BUILTIN_PRESETS[NATURE])
    _settle()
    saved_config = plots.config()
    saved_appearance = plots.appearance_state()
    assert saved_config.style.title == "Pore A" and saved_config.style.title_font_size_pt == 7

    again, reopened = _reopen(qtbot, window, tmp_path)
    assert reopened.config() == saved_config
    assert reopened.appearance_state() == saved_appearance
    assert reopened.property("nanoscopeInspectorHosted") is True
    second = again._analysis_explorer
    restored_key = again._figure_workspace.id_for_page(reopened)
    assert second.toolbar().bound_key() == restored_key and not second.adapter(restored_key).fallback
    # the hidden set is hidden again on the fresh page, and every group holds its instances
    assert reopened._section_area.isHidden() and not reopened._plot_btn.isVisible()
    pane = second.inspector().pane(restored_key)
    assert pane.data_pane.group("plot").body.isAncestorOf(reopened._plot_type_combo)
    assert pane.data_pane.group("recipes") is None
    assert {entry.text for entry in second.adapter(restored_key).exports()} >= {"Save as recipe…", "Manage recipes…"}
    # and it is a working page: Update plot draws, Undo takes a step back
    H.press_update(second)
    assert reopened._plot_armed
    again._dirty = False
    window._dirty = False
