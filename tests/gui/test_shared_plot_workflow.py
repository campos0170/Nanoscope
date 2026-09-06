"""Workflow chrome and styling cannot change analytical results or the zoom."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.analysis import iv, voltage_dep
from nanoscope.core.session import Session
from nanoscope.gui.event_plot_panel import DENSITY, DENSITY_LABEL, ENSEMBLE
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.plot_workflow import PlotWorkflowBar, SECTIONS, WarningSummary
from nanoscope.gui.subpop_window import SubpopReportWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow
from nanoscope.gui.psd_panel import VIEWS, VIEW_SPECTROGRAM
from .test_event_plot import _fed_window as event_window
from .test_psd import _fed_window as noise_window
from .test_subpop_views import report  # real, module-scoped synthetic report fixture


@pytest.fixture(params=["iv", "voltage", "events", "noise", "subpopulations"])
def page(request, qtbot, tmp_path):
    kind = request.param
    if kind == "events":
        window = event_window(qtbot, tmp_path, n_events=5)
        plots = [window._density_main, window._ensemble_plot]
    elif kind == "noise":
        window, _ = noise_window(qtbot, tmp_path)
        plots = [window._plot]
    elif kind == "subpopulations":
        window = SubpopReportWindow()
        qtbot.addWidget(window)
        window.set_report(request.getfixturevalue("report"))
        plots = list(window.figure_export_items())
    elif kind == "iv":
        window = IVWindow(Session(), defer_initial_refresh=True)
        qtbot.addWidget(window)
        window._points = [iv.IVPoint(v, 100 * v + 2, "pA", "measured", str(i), f"run {i}")
                          for i, v in enumerate((-.2, -.5, -.8, -1.0))]
        window._redraw()
        plots = [window._plot]
    else:
        window = VDepWindow(Session(), defer_initial_refresh=True)
        qtbot.addWidget(window)
        window._points = [voltage_dep.VDepPoint(
            v, -100 * v + 2, f"run {i}", baseline_fingerprint="baseline-a",
            analyzed_duration_s=10, voltage_source="measured", doc_id=str(i))
                          for i, v in enumerate((-.2, -.5, -.8, -1.0))]
        window._series = voltage_dep.vdep_comparability(window._points, metric="capture_rate_hz")
        window._redraw()
        plots = [window._plot]
    window.resize(1280, 820)
    window.show()
    qtbot.wait(10)
    return kind, window, plots


def _forbid_analysis(monkeypatch, window):
    def forbidden(*_args, **_kwargs):
        pytest.fail("A section, style, or warning disclosure invoked analysis")
    for name in ("refresh", "_refresh_from_user", "_recompute", "_refresh_view",
                 "_redraw", "_redraw_analysis", "_redraw_density", "_redraw_ensemble"):
        if hasattr(window, name):
            monkeypatch.setattr(window, name, forbidden)
    if isinstance(window, SubpopReportWindow):
        monkeypatch.setattr(window._embed_view, "_redraw", forbidden)
        monkeypatch.setattr(window._wave_view, "_redraw", forbidden)


def test_every_page_exposes_same_sections_and_persistent_actions(page, qtbot, monkeypatch):
    _kind, window, plots = page
    assert tuple(window._workflow.sections.itemText(i) for i in range(4)) == SECTIONS
    before = window._plot_data().copy(deep=True)
    for plot in plots:
        plot.setRange(xRange=(.1, .4), yRange=(.2, .8), padding=0)
    ranges = [np.asarray(plot.viewRange()).copy() for plot in plots]
    _forbid_analysis(monkeypatch, window)
    for section in (*SECTIONS, "Data"):
        window._workflow.set_section(section)
        qtbot.wait(5)
        assert window._workflow.isVisible()
        assert window._workflow.export_button.isVisible()
        assert all(button.isVisible() for button in window._workflow.primary_actions)
        assert window._workflow.sections.currentText() == section
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)
    for plot, limits in zip(plots, ranges):
        np.testing.assert_allclose(plot.viewRange(), limits, atol=1e-12, rtol=0)


def test_shared_style_changes_preserve_values_fit_and_view(page, qtbot, monkeypatch):
    kind, window, plots = page
    before = window._plot_data().copy(deep=True)
    fit = getattr(window, "_fit", None)
    source = (copy.deepcopy(window.report()) if kind == "subpopulations" else None)
    for plot in plots:
        plot.setRange(xRange=(.1, .4), yRange=(.2, .8), padding=0)
    limits = [np.asarray(plot.viewRange()).copy() for plot in plots]
    _forbid_analysis(monkeypatch, window)
    window._workflow.set_section("Appearance")
    window._appearance_widget.title_size.setValue(17)
    window._appearance_widget.line_width.setValue(2.5)
    window._appearance_widget.title.setText("Publication view")
    window._appearance_widget.title.textEdited.emit("Publication view")
    qtbot.wait(10)
    assert window.appearance_state()["title"] == "Publication view"
    assert all("Publication view" in plot.titleLabel.text for plot in plots)
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)
    assert getattr(window, "_fit", None) is fit
    if source is not None:
        pd.testing.assert_frame_equal(window.report().embeddings, source.embeddings, check_exact=True)
        pd.testing.assert_frame_equal(window.report().waveforms, source.waveforms, check_exact=True)
    for plot, expected in zip(plots, limits):
        np.testing.assert_allclose(plot.viewRange(), expected, atol=1e-12, rtol=0)


def test_warning_summary_keeps_every_original_word_and_expands_without_actions(qtbot):
    widget = WarningSummary()
    qtbot.addWidget(widget)
    warnings = ("First important warning " * 25, "Second scientific caveat")
    widget.set_warnings(warnings)
    widget.show()
    assert not widget.isHidden()
    assert widget.details.isHidden()
    assert "(2)" in widget.summary.text()
    assert widget.full_text() == "\n\n".join(warnings)
    widget.toggle.setChecked(True)
    assert not widget.details.isHidden()
    assert widget.details.toPlainText() == widget.full_text()
    widget.toggle.setChecked(False)
    assert widget.full_text() == "\n\n".join(warnings)


def test_workflow_navigation_is_not_the_primary_analysis_action(qtbot):
    calls = []
    bar = PlotWorkflowBar(actions=(("Analyze", lambda: calls.append("analyze")),))
    qtbot.addWidget(bar)
    for section in SECTIONS:
        bar.set_section(section)
    assert calls == []
    bar.primary_actions[0].click()
    assert calls == ["analyze"]


def test_heatmap_has_precise_visible_label_and_legacy_workspace_key(qtbot, tmp_path):
    window = event_window(qtbot, tmp_path, n_events=3)
    assert window._tabs.tabText(0) == DENSITY_LABEL
    assert window.current_view() == DENSITY
    window.set_view(ENSEMBLE)
    window.set_view("Density")
    assert window._tabs.currentIndex() == 0
    window.set_view(DENSITY_LABEL)
    assert window.current_view() == "Density"
    assert "count" in window._colorbar.getAxis("left").labelText.lower()


def test_subpopulation_colours_and_caveats_survive_shared_preset(qtbot, report):
    window = SubpopReportWindow()
    qtbot.addWidget(window)
    window.set_report(report)
    before = window._plot_data().copy(deep=True)
    palette = window.embedding_view().legend_entries()
    caveats = window.caveats()
    window.set_appearance_state({"data_color": "#ff0000", "fit_color": "#00ff00", "background": "light"})
    assert "data_color" not in window.appearance_state()
    assert window.embedding_view().legend_entries() == palette
    window._caveat_summary.toggle.setChecked(True)
    assert window.caveats() == caveats
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)


@pytest.mark.parametrize("view", [name for name, _label in VIEWS])
def test_noise_shared_preset_and_export_target_follow_visible_view(
        qtbot, tmp_path, monkeypatch, view):
    window, _ = noise_window(qtbot, tmp_path)
    window.set_view(view)
    window.show()
    qtbot.wait(10)
    plot = window.active_plot_item()
    target = window.figure_export_target()
    assert target is (window._spec_widget.scene() if view == VIEW_SPECTROGRAM else plot)
    before = window._plot_data().copy(deep=True)
    plot.setRange(xRange=(1, 2), yRange=(1, 2), padding=0)
    limits = np.asarray(plot.viewRange()).copy()
    _forbid_analysis(monkeypatch, window)
    window._appearance_widget._apply_preset(1)
    window._appearance_widget.legend_visible.setCurrentIndex(1)
    qtbot.wait(5)
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)
    assert window.figure_export_target() is target
    np.testing.assert_allclose(plot.viewRange(), limits, atol=1e-12, rtol=0)
    if view == VIEW_SPECTROGRAM:
        assert not window._appearance_widget.data_color.isEnabled()
        assert "colormap" in window._appearance_widget.data_color.toolTip()


@pytest.mark.parametrize("view", [DENSITY, ENSEMBLE])
def test_event_shared_palette_changes_existing_items_not_counts_or_traces(
        qtbot, tmp_path, monkeypatch, view):
    window = event_window(qtbot, tmp_path, n_events=4)
    window.set_view(view)
    window.show()
    qtbot.wait(10)
    target = window.figure_export_target()
    assert target is (window._density_glw.ci if view == DENSITY else window._ensemble_plot)
    before = window._plot_data().copy(deep=True)
    plot = window._density_main if view == DENSITY else window._ensemble_plot
    plot.setRange(xRange=(.1, .2), yRange=(2, 3), padding=0)
    limits = np.asarray(plot.viewRange()).copy()
    _forbid_analysis(monkeypatch, window)
    window.set_appearance_state({"data_color": "#cc0066", "fit_color": "#229944", "line_width": 3})
    qtbot.wait(5)
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)
    np.testing.assert_allclose(plot.viewRange(), limits, atol=1e-12, rtol=0)
    items = ((window._top_marginal_item, window._right_marginal_item)
             if view == DENSITY else (window._events_item,))
    for item in items:
        assert item.opts["pen"].widthF() == 3
        assert item.opts["pen"].color().name() == "#cc0066"
    assert window._appearance_widget.fit_color.isEnabled() == (view == ENSEMBLE)


def test_subpopulation_export_never_assigns_undrawn_class_name_to_visible_curve(qtbot, report):
    altered = copy.deepcopy(report)
    columns = [column for column in altered.waveforms if column != "T"]
    assert len(columns) >= 2
    altered.waveforms[columns[0]] = np.nan
    window = SubpopReportWindow()
    qtbot.addWidget(window)
    window.set_report(altered)
    table = window._plot_data()
    curves = table[table.part == "waveform"]
    assert columns[0] not in set(curves.series)
    for column in columns[1:]:
        samples = curves[curves.series == column]
        np.testing.assert_array_equal(samples.x, altered.waveforms["T"])
        np.testing.assert_array_equal(samples.y, altered.waveforms[column])
    assert window.figure_export_items() == (
        window._embed_view._plot.getPlotItem(), window._wave_view._plot.getPlotItem())


def test_subpopulation_empty_report_disables_persistent_export(qtbot):
    window = SubpopReportWindow()
    qtbot.addWidget(window)
    assert not window._workflow.export_button.isEnabled()


def test_subpopulation_small_layout_never_overlaps_captions_and_keeps_full_context(qtbot, report):
    window = SubpopReportWindow()
    qtbot.addWidget(window)
    window.set_report(report)
    window.resize(900, 700)
    window.show()
    for section in SECTIONS:
        window._workflow.set_section(section)
        for expanded in (False, True):
            window._caveat_summary.toggle.setChecked(expanded)
            qtbot.wait(15)
            for view in (window._embed_view, window._wave_view):
                assert view._caption.y() > view._plot.geometry().bottom()
                assert view.caption() in window._caveat_summary.full_text()
            assert window._wave_view.y() > window._embed_view.geometry().bottom()
            assert window._caveat_summary.summary.isVisible()
            assert window._workflow.primary_actions[0].isVisible()
    assert window._embed_view._honesty.text() in window.caveats()
    assert window._embed_view._umap_note.text() in window.caveats()


@pytest.mark.parametrize("embedded,size", [(False, (900, 700)), (True, (1280, 820)),
                                            (True, (1000, 700))])
def test_subpopulation_details_scroll_without_growing_shell(
        qtbot, report, monkeypatch, embedded, size, tmp_path):
    from nanoscope.gui.analysis_explorer import AnalysisExplorer, PAGE_SUBPOPULATIONS
    page = SubpopReportWindow()
    page.set_report(report)
    if embedded:
        shell = AnalysisExplorer()
        shell.show_page(PAGE_SUBPOPULATIONS, page)
    else:
        shell = page
    qtbot.addWidget(shell)
    shell.resize(*size)
    shell.show()
    qtbot.wait(20)
    before = page._plot_data().copy(deep=True)
    _forbid_analysis(monkeypatch, page)
    for section in ("Analysis", "Appearance"):
        page._caveat_summary.toggle.setChecked(False)
        page._workflow.set_section(section)
        page._caveat_summary.toggle.setChecked(True)
        qtbot.wait(30)
        assert (shell.width(), shell.height()) == size
        for widget in (page._workflow, page._standing_caveats):
            corner = widget.mapTo(shell, QtCore.QPoint(0, 0))
            assert shell.rect().contains(QtCore.QRect(corner, widget.size()))
        close = next(button for button in page.findChildren(QtWidgets.QPushButton)
                     if button.text() == "Close")
        assert shell.rect().contains(QtCore.QRect(close.mapTo(shell, QtCore.QPoint(0, 0)), close.size()))
        assert page._caveat_summary.details.isVisible()
        viewport = page._body_scroll.viewport()
        detail_rect = QtCore.QRect(page._caveats.mapTo(viewport, QtCore.QPoint(0, 0)), page._caveats.size())
        assert viewport.rect().intersects(detail_rect), "Full details must be brought into view"
        assert page._caveats.toPlainText() == page.caveats()
        pd.testing.assert_frame_equal(page._plot_data(), before, check_exact=True)
    shell.grab().save(str(tmp_path / "subpopulation-responsive.png"))
