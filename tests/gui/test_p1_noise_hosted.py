"""Phase 1, group G4: Noise spectrum hosted in a real Analysis Explorer.

Clicks and keys go through the QWindow, combos are picked from their popups, menus are clicked,
and the project round trip is a real ``_write_project`` reopened by ``_open_project`` until
``projectRestored``. Every assertion is on what is drawn or shown. Synthetic data only; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.core.project import DocumentRef, ProjectState
from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import figure_text_for
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_interaction import app_menu_for
from nanoscope.gui.psd_panel import VIEW_ALLPOINTS, VIEW_FFT, VIEW_PSD, VIEW_SPECTROGRAM

from . import p1_hosting as H
from .test_p0_pins_clear_with_the_words import _axes_of, _axis, _pin_inside
from .test_p0_reset_on_quantity_change import _click_button, _pick
from .test_plot_window_routes import _edit_double_click, _main_window, _noisy_csv, _press_undo
from .test_psd import _fed_window

KEY = "noise"
RETURN = QtCore.Qt.Key.Key_Return


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


@pytest.fixture
def page(qapp, qtbot, tmp_path):
    return _fed_window(qtbot, tmp_path)[0]


@pytest.fixture
def explorer(qtbot, page):
    return H.hosted(qtbot, page, KEY)


def _tick(box):
    """Click a check box where it takes a click (its indicator): the box spans its column."""
    QtTest.QTest.mouseClick(box, QtCore.Qt.MouseButton.LeftButton,
                            QtCore.Qt.KeyboardModifier.NoModifier,
                            QtCore.QPoint(8, box.height() // 2))
    QtWidgets.QApplication.processEvents()


def _data_pane(explorer):
    return explorer.inspector().pane(KEY).data_pane


def _body(explorer, group):
    return _data_pane(explorer).group(group).body


# -- the groups -------------------------------------------------------------------------------------

def test_the_groups_hold_the_pages_own_group_boxes_and_the_old_column_is_gone(explorer, page):
    adapter = explorer.adapter(KEY)
    assert adapter is page.inspector_adapter() and adapter.fallback is False
    assert [g.key for g in adapter.groups] == ["sources", "view", "estimator", "recipes"]
    assert [g.title for g in adapter.groups] == ["Sources", "View", "Estimator", "Recipes"]
    assert [g.expanded for g in adapter.groups] == [True, True, False, False]
    for group, widgets in (
            ("sources", (page._recording_group, page._rec_combo, page._compare_list,
                         page._segment_group, page._start_spin, page._end_spin, page._full_btn,
                         page._seg_label)),
            ("view", (page._view_group, page._view_combo, page._view_options,
                      page._display_group, page._loglog_check, page._asd_check)),
            ("estimator", (page._welch_group, page._nperseg_combo, page._window_combo,
                           page._overlap_spin, page._rms_group, page._rms_lo_spin,
                           page._rms_hi_spin)),
            ("recipes", (page._export_info, page._templates))):
        for widget in widgets:
            assert _body(explorer, group).isAncestorOf(widget), (group, widget)
    assert page._controls_scroll.isHidden()                       # nothing is left in it
    assert page._style_group.isHidden()                           # a husk of legacy rows
    for legacy in (page._title_edit, page._data_swatch, page._fit_swatch, page._bg_combo,
                   page._export_w, page._export_h):
        assert not legacy.isVisible()
    assert page.inspector_adapter() is adapter
    assert page.property("nanoscopeInspectorHosted") is True
    assert _data_pane(explorer).group("recipes").is_expanded() is False


def test_the_hidden_set_stays_hidden_and_wired(explorer, page, monkeypatch):
    for name in ("_save_btn", "_save_data_btn", "_refresh_btn"):
        assert getattr(page, name).isHidden(), name
    assert not page._workflow.isVisible()
    seen = []
    monkeypatch.setattr(page, "save_image", lambda *a, **k: seen.append("image"))
    monkeypatch.setattr(page, "export_data", lambda *a, **k: seen.append("data"))
    page._save_btn.click()
    page._save_data_btn.click()
    assert seen == ["image", "data"]
    heard = []
    page.workspaceChanged.connect(lambda: heard.append(1))
    page._workflow.primary_actions[0].click()                     # the hidden verb still works
    assert heard == [1]


def test_the_spectrogram_colour_rows_leave_the_style_husk_for_the_format_extras(explorer, page):
    extras = explorer.adapter(KEY).format.extras
    assert list(extras) == ["colour_map"] and len(extras["colour_map"]) == 1
    container = extras["colour_map"][0]
    for widget in (page._spec_cmap_combo, page._spec_levels_combo, page._spec_levels_row,
                   page._spec_level_lo, page._spec_level_hi, page._spec_logf_check):
        assert container.isAncestorOf(widget), widget
        assert not page._style_group.isAncestorOf(widget)
    assert not page._view_options.isAncestorOf(page._spec_logf_check)
    page.set_view(VIEW_SPECTROGRAM)
    assert page._spec_cmap_combo.isEnabled()                      # still gated by the view
    page._spec_logf_check.setChecked(True)                        # still wired to the redraw
    assert page._spec_log_edges is not None
    page._spec_levels_combo.setCurrentIndex(1)
    assert page._spec_level_lo.isEnabled() == (page._spec_levels_mode() == "manual")
    page.set_view(VIEW_PSD)
    assert not page._spec_cmap_combo.isEnabled()


def test_the_old_section_names_reveal_the_matching_group(explorer, page):
    pane = _data_pane(explorer)
    for name in ("sources", "view", "estimator", "recipes"):
        pane.group(name).set_expanded(False)
    page._workflow.set_section("Analysis")
    assert pane.group("estimator").is_expanded() and explorer.inspector().current_tab() == 0
    page._workflow.set_section("Export")
    assert pane.group("recipes").is_expanded()
    page._workflow.set_section("Data")
    assert pane.group("sources").is_expanded()
    page._workflow.set_section("Appearance")
    assert explorer.inspector().current_tab() == 1


# -- every control still does its job --------------------------------------------------------------

def test_the_estimator_and_display_controls_recompute_through_real_picks_and_clicks(explorer, page):
    explorer.reveal("estimator")
    before = page.psd_result()[1].copy()
    _pick(page._window_combo, text="boxcar")
    assert page._window_combo.currentText() == "boxcar"
    assert page.psd_result()[1].tolist() != before.tolist()      # a new estimate
    _pick(page._nperseg_combo, text="1024")
    assert page.segment_info()[0] == 1024
    explorer.reveal("view")
    assert not page.is_asd()
    _click_button(page._asd_check)
    assert page.is_asd()
    _tick(page._loglog_check)
    assert page._loglog_check.isChecked() is False
    explorer.reveal("estimator")
    page._overlap_spin.setFocus()
    page._overlap_spin.selectAll()
    QtTest.QTest.keyClicks(page._overlap_spin, "75")
    QtTest.QTest.keyClick(page._overlap_spin, QtCore.Qt.Key.Key_Tab)
    assert page._overlap_spin.value() == 75
    rms = page.rms_value()
    page._rms_hi_spin.setFocus()
    page._rms_hi_spin.selectAll()
    QtTest.QTest.keyClicks(page._rms_hi_spin, "2000")
    QtTest.QTest.keyClick(page._rms_hi_spin, QtCore.Qt.Key.Key_Tab)
    assert page.rms_value() != rms
    assert page.rms_readout_text()


def test_the_view_pick_switches_the_drawn_plot_and_the_format_controller(explorer, page):
    spec = explorer.adapter(KEY).format
    colours, = spec.extras["colour_map"]
    assert not colours._nanoscope_format_visible()
    seen = []
    spec.views_changed.connect(lambda *_: seen.append(page.current_view()))
    expected = {VIEW_PSD: page._plot, VIEW_FFT: page._fft_plot,
                VIEW_SPECTROGRAM: page._spec_plot, VIEW_ALLPOINTS: page._hist_plot}
    assert spec.controllers() == [figure_text_for(page._plot)]
    for view in (VIEW_FFT, VIEW_SPECTROGRAM, VIEW_ALLPOINTS, VIEW_PSD):
        _pick(page._view_combo, data=view)
        assert page.current_view() == view and expected[view].getViewBox().isVisible()
        assert spec.controllers() == [figure_text_for(expected[view])]
        assert colours._nanoscope_format_visible() is (view == VIEW_SPECTROGRAM)
    assert seen == [VIEW_FFT, VIEW_SPECTROGRAM, VIEW_ALLPOINTS, VIEW_PSD]


def test_a_pin_typed_on_one_views_y_axis_does_not_frame_another(explorer, page):
    pin = _pin_inside(page._plot, "left")
    assert set(_axes_of(page)) == {"psd.left"}
    _pick(page._view_combo, data=VIEW_FFT)
    assert set(_axes_of(page)) == {"psd.left"}
    state = _axis(page._fft_plot, "left")
    assert state.range is None                                    # Auto, not the PSD's pin
    assert state.shown_range != pytest.approx(pin, rel=1e-3)
    _pick(page._view_combo, data=VIEW_PSD)
    assert figure_text_for(page._plot).host.axis_state("left").range == pytest.approx(pin)


# -- the verb, the exports, the history ------------------------------------------------------------

def test_update_plot_is_analyze_refresh_once_per_press_and_key(explorer, page, monkeypatch):
    calls = []
    real = page.refresh
    monkeypatch.setattr(page, "refresh", lambda: (calls.append(1), real())[1])
    heard = []
    page.workspaceChanged.connect(lambda: heard.append(1))
    H.press_update(explorer)
    assert calls == [1] and heard == [1]
    H.key(explorer, RETURN, H.CTRL)
    assert calls == [1, 1] and heard == [1, 1]
    bar = explorer.toolbar()
    assert bar.update_button.text() == "Update plot" and explorer.adapter(KEY).pending().count == 0
    assert "Re-gather" in bar.update_action.toolTip()


def test_export_menu_is_the_right_click_menu_and_runs_the_pages_handlers(explorer, page,
                                                                         monkeypatch):
    shell = {"Save to Figure Library…", "Save figure bundle…", "Open Figure Library"}
    menu = H.export_menu(explorer)
    entries = [a.text() for a in menu.actions() if not a.isSeparator() and a.text() not in shell]
    menu.close()
    right_click: list[str] = []
    for action in app_menu_for(page._plot).actions():
        right_click = [] if action.isSeparator() else right_click + [action.text()]
    assert entries == right_click == ["Export current figure…", "Export data (CSV)…"]
    seen = []
    monkeypatch.setattr(page, "save_image", lambda *a, **k: seen.append("image"))
    monkeypatch.setattr(page, "export_data", lambda *a, **k: seen.append("data"))
    H.choose_export(explorer, "Export current figure…")
    H.choose_export(explorer, "Export data (CSV)…")
    assert seen == ["image", "data"]


@pytest.mark.parametrize("tab", (0, 1), ids=("data_tab", "format_tab"))
def test_the_undo_key_takes_back_an_edit_made_in_the_figure_from_either_inspector_tab(
        explorer, page, tab):
    explorer.inspector().set_tab(tab)
    _edit_double_click(page._plot, page._plot.titleLabel, "PSD, pore A")
    assert page.appearance_state().get("title") == "PSD, pore A"
    assert explorer.toolbar().undo_action.isEnabled()
    _press_undo(page)
    assert "title" not in page.appearance_state()
    assert explorer.toolbar().redo_action.isEnabled()


def test_reset_view_sits_on_the_status_line_and_acts_on_the_view_on_screen(explorer, page):
    buttons = page.statusBar().findChildren(QtWidgets.QToolButton, "plotResetViewButton")
    assert len(buttons) == 1 and buttons[0].isVisible()
    legacy = page.findChildren(QtWidgets.QAbstractButton, "plotResetViewButtonLegacy")
    assert len(legacy) == 1 and not legacy[0].isVisible()
    assert page.findChild(QtWidgets.QAbstractButton, "plotResetViewButton") is buttons[0]
    for view in (VIEW_PSD, VIEW_FFT):
        _pick(page._view_combo, data=view)
        plot = page.active_plot_item()
        lo, hi = plot.viewRange()[0]
        plot.setXRange(lo + 0.45 * (hi - lo), lo + 0.5 * (hi - lo), padding=0)
        QtWidgets.QApplication.processEvents()
        narrow = plot.viewRange()[0][1] - plot.viewRange()[0][0]
        assert narrow < 0.1 * (hi - lo)
        H.click(buttons[0], pos=QtCore.QPoint(8, buttons[0].height() // 2))
        assert plot.viewRange()[0][1] - plot.viewRange()[0][0] > 5 * narrow, view


# -- sizes ----------------------------------------------------------------------------------------

def test_the_page_fits_the_explorers_minimum_size(qtbot, page):
    explorer = H.hosted(qtbot, page, KEY, size=(840, 560))
    explorer.resize(840, 560)
    QtWidgets.QApplication.processEvents()
    assert page.minimumSizeHint().width() <= 500
    host, pane = explorer.inspector(), explorer.inspector().pane(KEY)
    assert host.minimumSizeHint().width() <= 330
    for group in explorer.adapter(KEY).groups:
        assert pane.data_pane.group(group.key).minimumSizeHint().width() <= 290, group.key
    assert pane.data_pane.horizontalScrollBar().maximum() == 0
    for view in (VIEW_PSD, VIEW_FFT, VIEW_SPECTROGRAM, VIEW_ALLPOINTS):
        page.set_view(view)
        QtWidgets.QApplication.processEvents()
        assert page._view_stack.currentWidget().width() >= 420, view
    page.set_view(VIEW_PSD)
    QtWidgets.QApplication.processEvents()
    assert page._plot_widget.width() >= 420 and page._plot_widget.height() >= 300
    assert explorer.width() == 840


# -- the project -----------------------------------------------------------------------------------

def test_a_full_save_and_reopen_restores_the_hosted_controls_and_the_view(
        qapp, qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    recording = tmp_path / "noise.csv"
    _noisy_csv(recording)
    source = _main_window(qtbot)
    with qtbot.waitSignal(source.projectRestored, timeout=20_000):
        source._begin_restore(ProjectState(documents=[DocumentRef(str(recording))]))
    source._open_psd()
    page = source._psd_window
    # New managed figures compute only when Update plot is pressed.
    H.press_update(source._analysis_explorer)
    assert page.has_data() and page.property("nanoscopeInspectorHosted") is True
    source._analysis_explorer.reveal("estimator")
    _pick(page._window_combo, text="boxcar")
    _pick(page._nperseg_combo, text="1024")
    source._analysis_explorer.reveal("view")
    _click_button(page._asd_check)
    page._overlap_spin.setValue(75)
    _pick(page._view_combo, data=VIEW_SPECTROGRAM)
    H.press_update(source._analysis_explorer)
    page._spec_cmap_combo.setCurrentText("magma")
    page._spec_logf_check.setChecked(True)
    assert page.current_view() == VIEW_SPECTROGRAM
    saved = page.appearance_state()
    project = tmp_path / "noise-hosted.nanoscope"
    assert source._write_project(str(project))
    source._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_psd()
    again = reopened._psd_window
    assert again.property("nanoscopeInspectorHosted") is True
    assert again._window_combo.currentText() == "boxcar" and again._nperseg_combo.currentText() == "1024"
    assert again.is_asd() and again._overlap_spin.value() == 75
    assert again.current_view() == VIEW_SPECTROGRAM
    assert again._spec_cmap_combo.currentText() == "magma" and again._spec_logf_check.isChecked()
    assert again.appearance_state() == saved
    explorer = reopened._analysis_explorer
    assert explorer.inspector().pane(reopened._figure_workspace.id_for_page(again)).data_pane.group("view").body.isAncestorOf(
        again._view_combo)
    reopened._dirty = False
