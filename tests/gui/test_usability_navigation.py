"""User-facing navigation, compact canvas and explicit update actions; offscreen only."""
from copy import deepcopy

import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.analysis_catalog import FIGURE_NAVIGATION_HELP, ENTRY_BY_ROUTE
from nanoscope.gui.analysis_explorer import AnalysisExplorer
from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow

from . import p1_hosting as H
from .p2_shell_rig import Controller, record
from .test_comparison_figures_panel import source


@pytest.fixture(autouse=True)
def isolated_figures_preference(qapp):
    settings = QtCore.QSettings()
    key = "analysisExplorer/figuresVisible"
    had, previous = settings.contains(key), settings.value(key)
    settings.remove(key)
    yield
    if had:
        settings.setValue(key, previous)
    else:
        settings.remove(key)


@pytest.mark.parametrize("label,kind", [
    ("Pair-plot matrix", "pairs"),
    ("Faceted comparisons", "facets"),
    ("Individual-event overlays", "events"),
])
def test_direct_new_comparison_recipe_opens_correct_empty_controls(qtbot, monkeypatch, label, kind):
    from nanoscope.gui.main_window import MainWindow

    computed = []
    monkeypatch.setattr(ComparisonFiguresWindow, "analyze", lambda self: computed.append(self))
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    explorer = window._ensure_analysis_explorer()
    action = next(action for action in explorer.figures_list().new_menu.actions() if action.text() == label)
    action.trigger()
    key = owner.current_id()
    page = owner.page(key)
    assert isinstance(page, ComparisonFiguresWindow)
    assert page.config()["kind"] == kind
    assert page._view_combo.currentText() == label
    assert owner.get(key).spec.result.status == "empty"
    assert not computed and not page.has_data()
    # A second New is a separate editable figure, not a revisit to the first.
    action.trigger()
    assert owner.current_id() != key and owner.page(key) is page
    assert not computed


def test_analysis_help_distinguishes_reopen_from_new_without_changing_routes(qtbot):
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    for kind in ("distributions", "distribution_summaries", "comparison_figures"):
        explorer._home.set_current_route(kind)
        assert FIGURE_NAVIGATION_HELP in explorer._home._description.text()
        assert explorer._home.current_route() == kind
        assert ENTRY_BY_ROUTE[kind].label in explorer._home._open.text()
    assert "most recently used" in explorer._home_btn.toolTip()


@pytest.mark.parametrize("factory,kind", [
    (ComparisonFiguresWindow, "comparison_figures"),
    (DistributionSummaryWindow, "distribution_summaries"),
])
def test_hosted_update_wording_and_pending_reason_match_visible_action(qtbot, factory, kind):
    page = factory()
    page.set_sources([source()])
    if kind == "comparison_figures":
        page.set_config({"kind": "pairs"})
    assert "Generate" in page._status.text()
    explorer = H.hosted(qtbot, page, kind)
    assert "Update plot" in page._status.text()
    assert "Generate" not in page._status.text()
    assert "press Generate" not in page._templates.note.text()
    assert "Generate to use" not in page._templates.note.text()
    if kind == "distribution_summaries":
        assert "Update plot to begin" in page._plot.titleLabel.text
    H.press_update(explorer)
    assert page.has_data() and not page._stale
    original = page._plot_data().copy(deep=True)
    snapshot = deepcopy(page._snapshot)
    if kind == "comparison_figures":
        page._bins.setValue(page._bins.value() + 3)
        expected = "histogram bins"
    else:
        page._metric_combo.setCurrentIndex(page._metric_combo.findData("depth_pA"))
        expected = "measurement"
    assert expected in explorer._pending_notice.text()
    assert explorer._pending_notice.isVisible()
    assert expected in page._status.toolTip() and "Update plot" in page._status.toolTip()
    assert page._snapshot == snapshot
    pd.testing.assert_frame_equal(page._plot_data(), original, check_exact=True)
    H.press_update(explorer)
    assert not page._stale and not explorer._pending_notice.isVisible()


@pytest.mark.parametrize("width", [840, 1024])
def test_figures_toggle_reclaims_canvas_preserving_selection_and_pending_edits(qtbot, width):
    values = [record("First"), record("Second")]
    owner = Controller(*values)
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(owner)
    page = H.SyntheticPlotPage("iv")
    explorer.show_page(values[0].spec.figure_id, page, kind="iv")
    explorer.resize(width, 700)
    explorer.show()
    qtbot.waitExposed(explorer)
    listing = explorer.figures_list()
    listing.restore_selection([value.spec.figure_id for value in values], values[0].spec.figure_id)
    page.set_pending(1, "the measurement")
    QtWidgets.QApplication.processEvents()
    selected, current = listing.selected_ids(), listing.current_id()
    shown_width = explorer._tabs.width()
    bar = explorer.toolbar()
    assert not listing.new_button.isVisible() and not listing.duplicate_button.isVisible()
    assert bar.new_figure_button.isVisible()
    assert bar.widgetForAction(bar.duplicate_action).isVisible()
    assert bar.export_button.mapTo(explorer, bar.export_button.rect().topRight()).x() < explorer.width()
    H.click(bar.widgetForAction(bar.figures_action))
    assert listing.isHidden() and not bar.figures_action.isChecked()
    # At 840 px some released width also restores the inspector's remembered width.
    assert explorer._tabs.width() >= shown_width + 100
    assert listing.selected_ids() == selected and listing.current_id() == current
    assert page.updates == 0 and explorer._pending_notice.isVisible()
    # The same command remains keyboard-accessible after the entire sidebar is hidden.
    H.key(explorer, QtCore.Qt.Key.Key_L, H.CTRL | QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert listing.isVisible() and bar.figures_action.isChecked()
    assert listing.selected_ids() == selected and listing.current_id() == current
    assert page.updates == 0


def test_figures_visibility_restores_between_explorers_without_mutating_projects(qtbot):
    first = AnalysisExplorer()
    qtbot.addWidget(first)
    first.bind_figure_workspace(Controller(record()))
    first.set_figures_visible(False)
    value = record()
    owner = Controller(value)
    second = AnalysisExplorer()
    qtbot.addWidget(second)
    second.bind_figure_workspace(owner)
    second.show()
    qtbot.waitExposed(second)
    assert second.figures_list().isHidden()
    assert not second.toolbar().figures_action.isChecked()
    assert second.figures_list().current_id() == value.spec.figure_id
    assert second.page_count() == 0 and not owner.calls
    second.set_figures_visible(True)
    assert second.figures_list().isVisible()
    second.bind_figure_workspace(None)
    assert second.figures_list().isHidden() and not second.toolbar().figures_action.isVisible()


@pytest.mark.parametrize("width", [840, 1024])
def test_real_pairplot_canvas_controls_fit_with_figures_and_inspector_open(qtbot, width):
    value = record("Pair plot", kind="comparison_figures")
    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    explorer.bind_figure_workspace(Controller(value))
    page = ComparisonFiguresWindow()
    page.set_sources([source()])
    page.set_config({"kind": "pairs", "features": ["dwell_s", "depth_pA", "snr"]})
    explorer.show_page(value.spec.figure_id, page, kind="comparison_figures")
    explorer.resize(width, 800)
    explorer.show()
    qtbot.waitExposed(explorer)
    H.press_update(explorer)
    page._bins.setValue(page._bins.value() + 3)
    page._fit_button.click()
    QtWidgets.QApplication.processEvents()
    assert explorer.figures_list().isVisible()
    assert page._warnings.isVisible()
    assert page._status.height() >= page._status.heightForWidth(page._status.width())
    for widget in (page._status, page._canvas_note, page._fit_button, page._zoom_out,
                   page._zoom_in, page._view, page._warnings.toggle):
        assert widget.isVisible()
        bounds = QtCore.QRect(widget.mapTo(explorer._tabs, QtCore.QPoint()), widget.size())
        assert explorer._tabs.rect().contains(bounds), (widget, bounds, explorer._tabs.rect())
    # Fit canvas remains a fit of the whole publication image, not a cropped viewport.
    fitted = page._view.mapFromScene(page._view.sceneRect()).boundingRect()
    assert page._view.viewport().rect().adjusted(-2, -2, 2, 2).contains(fitted)
    assert page._warnings.toggle.text() == "Details"
    page._warnings.toggle.click()
    QtWidgets.QApplication.processEvents()
    assert page._warnings.details.isVisible()
    assert page._warnings.details.toPlainText() == page.details_text()


def test_pending_explanation_clears_on_home_without_running_analysis(qtbot):
    page = H.SyntheticPlotPage("iv")
    explorer = H.hosted(qtbot, page, "iv")
    page.set_pending(2, "the measurement", "the source selection")
    assert explorer._pending_notice.isVisible()
    assert "the source selection" in explorer._pending_notice.text()
    explorer.show_home()
    assert not explorer._pending_notice.isVisible() and page.updates == 0


@pytest.mark.parametrize("kind,plot_names", [
    ("distribution_summaries", ("_plot_widget",)),
    ("noise", ("_plot_widget", "_fft_widget", "_spec_widget", "_hist_widget")),
    ("event_waveforms", ("_density_glw", "_ensemble_widget")),
    ("distributions", ("_plot_widget",)),
    ("iv", ("_plot_widget",)),
    ("voltage_series", ("_plot_widget",)),
])
def test_managed_plot_pages_fit_narrow_canvas_with_both_sidebars_open(qtbot, kind, plot_names):
    from nanoscope.gui.main_window import MainWindow

    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    key = owner.create(kind)
    page = owner.page(key)
    explorer = window._ensure_analysis_explorer()
    explorer.resize(840, 800)
    explorer.set_figures_visible(True)
    explorer.show()
    qtbot.waitExposed(explorer)
    QtWidgets.QApplication.processEvents()
    assert explorer.figures_list().isVisible() and explorer.inspector().isVisible()
    assert page.width() <= explorer._tabs.width()
    for name in plot_names:
        plot = getattr(page, name)
        assert plot.minimumWidth() <= 280
        if plot.isVisible():
            bounds = QtCore.QRect(plot.mapTo(explorer._tabs, QtCore.QPoint()), plot.size())
            assert bounds.left() >= 0 and bounds.right() < explorer._tabs.width(), (kind, name, bounds)
