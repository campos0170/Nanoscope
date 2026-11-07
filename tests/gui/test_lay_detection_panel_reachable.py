"""Reachability on the current Inspector, Results dock and Conditioning workspace.

The old fixture assumed one ancestor scroll containing DetectionPanel and Stats.
That topology was retired: parameters now scroll per page, commands are pinned,
Stats/Fits live in Results and the protocol editor lives in Signal Conditioning.
These checks retain full-rectangle geometry and actual Qt visibility assertions
at narrow widths instead of accepting a hidden or clipped control.
"""
import pytest
from PySide6 import QtCore, QtWidgets
from nanoscope.gui.panels import _ElidingCheckBox
from .inspector_layout_rig import build, settle, show_page, rectangle, assert_whole, reveal, close_window


@pytest.fixture(autouse=True)
def isolated_settings(qapp):
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName('NanoScopeTest'); qapp.setApplicationName('CurrentInspectorReachability')
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org); qapp.setApplicationName(app)


@pytest.fixture
def window(qtbot):
    window = build(qtbot, width=460)
    yield window
    close_window(window)


@pytest.mark.parametrize('width', [320, 460, 560])
@pytest.mark.parametrize('page', ['Preprocess', 'Detection', 'ML', 'Measure'])
def test_current_parameter_pages_have_no_horizontal_scrollbar(window, width, page):
    window.resizeDocks([window._inspector], [width], QtCore.Qt.Horizontal)
    area = show_page(window, page)
    assert area.viewport().width() > 0
    assert area.horizontalScrollBar().maximum() == 0
    assert not area.horizontalScrollBar().isVisible()
    assert area.widget().minimumSizeHint().width() <= area.viewport().width()


def test_protocol_verbs_are_reachable_in_the_real_conditioning_workspace(window):
    detection = window._analysis_panel.detection_panel
    show_page(window, 'Preprocess')
    assert_whole(detection.open_conditioning_button)
    detection.open_conditioning_button.click()
    workspace = window._signal_conditioning
    workspace.resize(workspace.minimumWidth(), workspace.minimumHeight())
    settle(workspace)
    for name in ('protocol_combo', 'protocol_save_btn', 'protocol_rename_btn', 'protocol_delete_btn'):
        assert_whole(getattr(detection, name), workspace)
    combo = rectangle(detection.protocol_combo, workspace)
    save = rectangle(detection.protocol_save_btn, workspace)
    assert save.top() > combo.bottom()


@pytest.mark.parametrize('width', [320, 460, 560])
def test_ml_preset_verbs_are_reachable_and_left_aligned(window, width):
    window.resizeDocks([window._inspector], [width], QtCore.Qt.Horizontal)
    show_page(window, 'ML')
    detection = window._analysis_panel.detection_panel
    for name in ('ml_preset_save_btn', 'ml_preset_delete_btn'):
        area = reveal(window, getattr(detection, name))
        assert rectangle(getattr(detection, name), area.viewport()).left() < area.viewport().width()
    assert rectangle(detection.ml_preset_save_btn, area.widget()).left() < area.viewport().width() / 2
    assert_whole(detection.ml_detect_button)


@pytest.mark.parametrize('width', [320, 460, 560])
def test_detection_noise_and_advanced_controls_remain_reachable(window, width):
    window.resizeDocks([window._inspector], [width], QtCore.Qt.Horizontal)
    show_page(window, 'Detection')
    detection = window._analysis_panel.detection_panel
    detection.detection_advanced.set_expanded(True)
    settle(window)
    for name in ('noise_policy', 'noise_window_s', 'k_in', 'k_out', 'min_dwell_s',
                 'polarity_override', 'max_dwell_s', 'merge_gap_ms'):
        reveal(window, getattr(detection, name))
    for name in ('detect_button', 'sweep_button', 'null_button', 'recovery_button'):
        assert_whole(getattr(detection, name), detection._detection_page)


def test_frequency_response_plot_is_reachable_in_its_workspace(window):
    window._open_signal_conditioning()
    workspace = window._signal_conditioning
    workspace.preview_tabs.setCurrentIndex(2)
    settle(workspace)
    plot = window._analysis_panel.detection_panel.response_plot
    assert workspace.isAncestorOf(plot)
    assert_whole(plot, workspace)


def test_stats_fit_controls_are_reachable_in_results_not_the_inspector(window):
    panel = window._analysis_panel
    panel.results_tabs.setCurrentIndex(2)
    settle(window)
    assert window._results_dock.isAncestorOf(panel._stats.fit_button)
    assert not window._inspector.isAncestorOf(panel._stats.fit_button)
    for name in ('use_efficiency', 'use_filter_limited', 'fit_button'):
        reveal(window, getattr(panel._stats, name))


def test_preset_combo_does_not_size_itself_to_its_longest_label(window):
    combo = window._analysis_panel.detection_panel.ml_preset
    assert max(len(combo.itemText(i)) for i in range(combo.count())) > 80
    assert combo.minimumSizeHint().width() <= 260


@pytest.mark.parametrize('name', ['ml_norm_method', 'normalize_mode', 'detector_combo',
                                   'detrend_method', 'noise_policy'])
def test_scientific_combos_do_not_floor_their_viewport(window, name):
    combo = getattr(window._analysis_panel.detection_panel, name)
    assert combo.minimumSizeHint().width() <= 260
    if name == 'noise_policy':
        assert combo.sizeAdjustPolicy() == QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon


def test_long_checkbox_labels_keep_their_words_without_forcing_a_wide_dock(window):
    detection = window._analysis_panel.detection_panel
    assert isinstance(detection.savgol_enabled, _ElidingCheckBox)
    assert detection.savgol_enabled.text() == 'enable (peak-preserving polynomial smooth — KEEPS the fit)'
    assert detection.savgol_enabled.toolTip()
    assert detection.savgol_enabled.minimumSizeHint().width() <= 200
    assert window._analysis_panel._stats.use_efficiency.minimumSizeHint().width() <= 200


def test_state_changes_keep_pinned_actions_fixed_and_parameter_pages_accessible(window):
    detection = window._analysis_panel.detection_panel
    area = show_page(window, 'Detection')
    before = rectangle(detection.detect_button, window)
    edits = [lambda: detection.ml_preset.setCurrentIndex(detection.ml_preset.count() - 1),
             lambda: detection.detector_combo.setCurrentIndex(detection.detector_combo.count() - 1),
             lambda: detection.detector_combo.setCurrentIndex(0),
             lambda: detection.noise_policy.setCurrentIndex(detection.noise_policy.findData('local')),
             lambda: detection.noise_window_s.setValue(.37),
             lambda: detection.savgol_enabled.setChecked(True),
             lambda: detection.set_ml_runnable(False, 'no model loaded'),
             lambda: detection.set_ml_runnable(True, '')]
    for edit in edits:
        edit(); settle(window)
        assert rectangle(detection.detect_button, window) == before
        assert_whole(detection.detect_button)
        assert area.horizontalScrollBar().maximum() == 0
