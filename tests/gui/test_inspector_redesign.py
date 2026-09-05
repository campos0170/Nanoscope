"""Inspector layout changes must not edit analysis inputs or event data."""
from __future__ import annotations

import os
import nanoscope  # noqa: F401
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import EventTable


def settle():
    for _ in range(6):
        QtWidgets.QApplication.processEvents()


@pytest.fixture
def window(qapp, qtbot):
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName('NanoScopeTest')
    qapp.setApplicationName('InspectorRedesign')
    QtCore.QSettings().clear()
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1512, 900)
    win.show()
    settle()
    yield win
    win._analysis_controller.abort()
    win._teardown()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app)


def assert_button_reachable(button):
    assert button.isVisible(), button.text()
    center = button.rect().center()
    assert button.visibleRegion().contains(center), f'{button.text()} is clipped'
    assert button.visibleRegion().boundingRect().height() >= button.height() - 2, button.text()


@pytest.mark.parametrize('width', [320, 460, 560])
def test_navigation_and_primary_actions_stay_visible_while_scrolling(window, width):
    window.resizeDocks([window._inspector], [width], QtCore.Qt.Horizontal)
    panel = window._analysis_panel
    detection = panel.detection_panel
    for index, button in ((0, panel.baseline_workspace.refresh_button),
                          (1, detection.detect_button), (2, detection.ml_detect_button),
                          (3, panel._measure._toggle)):
        detection.mode_tabs.setCurrentIndex(index)
        settle()
        scroll = detection.page_scrolls[detection.mode_tabs.currentWidget()]
        before = button.mapTo(window, QtCore.QPoint())
        for position in (scroll.verticalScrollBar().maximum(), 0):
            scroll.verticalScrollBar().setValue(position)
            settle()
            assert button.mapTo(window, QtCore.QPoint()) == before
            assert_button_reachable(button)
            assert not detection.mode_tabs.tabBar().visibleRegion().isEmpty()


def test_layout_navigation_and_disclosures_do_not_change_scientific_inputs(window, qtbot):
    panel = window._analysis_panel
    detection = panel.detection_panel
    detection.filter_enabled.setChecked(True)
    detection.filter_cutoff_hz.setValue(5000)
    detection.max_dwell_s.setValue(.03)
    detection.merge_gap_ms.setValue(.2)
    before = panel.detection_params()
    ml_before = detection.ml_params()
    with qtbot.assertNotEmitted(detection.paramsChanged):
        for index in range(detection.mode_tabs.count()):
            detection.mode_tabs.setCurrentIndex(index)
            detection.detection_advanced.set_expanded(True)
            detection.detection_advanced.set_expanded(False)
            if hasattr(detection, 'ml_advanced'):
                detection.ml_advanced.set_expanded(True)
                detection.ml_advanced.set_expanded(False)
            panel.baseline_workspace.runtime_details.set_expanded(True)
            panel.baseline_workspace.runtime_details.set_expanded(False)
            settle()
    assert panel.detection_params() == before
    assert detection.ml_params() == ml_before


def test_inactive_detector_does_not_leave_a_gap_in_the_active_form(window):
    detection = window._analysis_panel.detection_panel
    detection.show_detection_tab()
    settle()
    below_thresholds = detection.k_out.mapTo(detection, detection.k_out.rect().bottomLeft()).y()
    dwell_top = detection.min_dwell_s.mapTo(detection, QtCore.QPoint()).y()
    assert dwell_top - below_thresholds < 2 * detection.fontMetrics().height()


def test_empty_results_are_compact_and_populated_results_have_direct_tabs(window):
    panel = window._analysis_panel
    empty_height = panel._results_host.height()
    assert empty_height < 100
    assert not panel.results_tabs.isVisible()
    panel._sync_results_presence(True, 10)
    settle()
    assert panel.results_tabs.isVisible()
    assert [panel.results_tabs.tabText(i) for i in range(panel.results_tabs.count())] == [
        'Events', 'Summary', 'Fits', 'Quality']
    for index in range(panel.results_tabs.count()):
        panel.results_tabs.setCurrentIndex(index)
        settle()
        assert panel.detection_panel.mode_tabs.isVisible()
    panel.results_toggle.setChecked(False)
    settle()
    assert panel._results_host.height() < 100
    panel.results_toggle.setChecked(True)
    settle()
    assert panel.results_tabs.isVisible()


def test_baseline_buttons_fit_without_scrollbar_occlusion(window):
    panel = window._analysis_panel
    detection = panel.detection_panel
    for width in (320, 460):
        window.resizeDocks([window._inspector], [width], QtCore.Qt.Horizontal)
        detection.show_open_pore_baseline()
        scroll = detection.page_scrolls[detection._preprocess_page]
        for button in (panel.baseline_panel.probe_button, panel.baseline_panel.apply_button,
                       panel.baseline_panel.reset_button, panel.baseline_workspace.export_button,
                       panel.baseline_workspace.expand_button):
            settle()
            scroll.ensureWidgetVisible(button)
            settle()
            assert_button_reachable(button)


def test_expanded_comparison_is_the_same_live_plot_and_preserves_inputs(window, qtbot):
    panel = window._analysis_panel
    workspace = panel.baseline_workspace
    x = np.linspace(0, 1, 101)
    raw, base = 100 + np.sin(x), np.full_like(x, 100)
    workspace.set_preview(x, raw, base, raw - base)
    before = panel.detection_params()
    original = workspace.preview
    with qtbot.assertNotEmitted(workspace.modeChanged), \
            qtbot.assertNotEmitted(workspace.runtimeEnabledRequested), \
            qtbot.assertNotEmitted(workspace.refreshRequested):
        workspace.expand_comparison()
        settle()
        assert workspace.preview is original
        assert workspace._expanded_dialog.isVisible()
        workspace.set_preview(x, raw + 1, base, raw + 1 - base)
        np.testing.assert_array_equal(workspace.raw_curve.getData()[1], raw + 1)
        workspace.mark_preview_stale('Recording settings changed')
        assert workspace._expanded_status.text() == workspace.preview_status.text()
        workspace._expanded_dialog.close()
        settle()
    assert workspace._expanded_dialog is None
    assert workspace.preview is original
    assert workspace.preview.parentWidget() is workspace._preview_layout.parentWidget()
    assert panel.detection_params() == before


def test_column_preferences_preserve_data_schema_and_restore_by_name(qapp, qtbot):
    settings = QtCore.QSettings()
    settings.remove(EventTable._COLUMN_SETTINGS_KEY)
    raw = 100 + np.random.default_rng(9).normal(0, 1, 10000)
    raw[2000:2250] -= 25
    raw[6000:6250] -= 30
    frame, _ = detect_events(raw.astype(np.float32), 50000, min_dwell_s=.0001)
    before = frame.copy(deep=True)
    table = EventTable()
    qtbot.addWidget(table)
    table.set_events(frame)
    header = table.view.horizontalHeader()
    columns = list(frame.columns)
    order = [columns[header.logicalIndex(i)] for i in range(header.count())]
    assert order[:6] == list(EventTable._SCIENTIFIC_ORDER[:6])
    table.set_column_visible('start_idx', True)
    table.set_column_visible('snr', False)
    header.moveSection(header.visualIndex(columns.index('depth_pA')), 0)
    table.view.sortByColumn(columns.index('depth_pA'), QtCore.Qt.DescendingOrder)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert list(table.dataframe.columns) == list(before.columns)
    restored = EventTable()
    qtbot.addWidget(restored)
    # Preference keys are column names, independent of source-frame ordering.
    restored.set_events(frame.loc[:, list(reversed(columns))].copy())
    new_columns = list(restored.dataframe.columns)
    assert new_columns[restored.view.horizontalHeader().logicalIndex(0)] == 'depth_pA'
    assert not restored.view.isColumnHidden(new_columns.index('start_idx'))
    assert restored.view.isColumnHidden(new_columns.index('snr'))
    settings.remove(EventTable._COLUMN_SETTINGS_KEY)
