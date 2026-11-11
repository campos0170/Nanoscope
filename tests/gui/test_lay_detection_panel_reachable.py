"""Reachability on the current Inspector, Results dock and Conditioning workspace.

The old fixture assumed one ancestor scroll containing DetectionPanel and Stats.
That topology was retired: parameters now scroll per page, commands are pinned,
Stats/Fits live in Results and the protocol editor lives in Signal Conditioning.
These checks retain full-rectangle geometry and actual Qt visibility assertions
at narrow widths instead of accepting a hidden or clipped control. One test pins the look the
Preprocess overview's scroll area has to leave alone (it paints nothing of its own).
"""
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
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


@pytest.mark.parametrize('width', [320, 460, 560])
def test_the_preprocess_overview_scrolls_in_a_viewport_that_floors_nothing(window, width):
    """2026-11-11: the hosted Preprocess page is the workspace overview (8661d35). It sits in a
    bounded scroll area like every other page: its door (about 195 px wide) must not floor the
    Inspector, and at the usual widths the page needs no horizontal scroll. show_page hands back
    this same scroller for the hosted page, so the test above measures it too (it used to
    measure the hidden legacy one); what only this test says is that the area imposes no width
    floor of its own and holds the door, whole."""
    window.resizeDocks([window._inspector], [width], QtCore.Qt.Horizontal)
    detection = window._analysis_panel.detection_panel
    area = show_page(window, 'Preprocess')
    assert area is detection.preprocess_overview_scroll, (
        'the rig must hand back the scroller on screen')
    assert area.isVisible() and area.viewport().width() > 0
    assert area.isAncestorOf(detection.preprocess_open_button)
    assert area.minimumSizeHint().width() == 0, 'the viewport may not impose a width floor'
    assert area.horizontalScrollBar().maximum() == 0
    assert not area.horizontalScrollBar().isVisible()
    assert area.widget().minimumSizeHint().width() <= area.viewport().width()
    assert_whole(detection.preprocess_open_button, area.viewport())


def test_the_preprocess_overview_keeps_the_tab_panes_colour(window, qapp):
    """The scroll area around the overview paints nothing, so the page looks as released.

    QScrollArea.setWidget turns the overview's autofill on and its viewport fills with the
    window colour, so a bare wrap repaints the whole page. In the app's theme that was the
    window's near-black (#16181c) in place of the tab pane's #2f3238, with a 4 px frame of the
    pane left around it, and the unselected route buttons (their circles are 6 levels off that
    black) almost vanished. The two setAutoFillBackground(False) calls in
    set_preprocess_workspace_hosted keep the released page; this pins both, and the picture
    they keep: a pixel inside the overview is the pane's colour, the colour of the page's own
    margin. The suite runs without the app's theme, where pane and window colours are closer
    but still apart, which the last assertion requires so that the comparison can tell."""
    detection = window._analysis_panel.detection_panel
    area = show_page(window, 'Preprocess')
    overview, viewport = area.widget(), area.viewport()
    assert not overview.autoFillBackground()
    assert not viewport.autoFillBackground()
    spot = QtCore.QPoint(viewport.width() - 3, viewport.height() - 3)
    assert viewport.childAt(spot) is overview, 'the sampled spot must be the overview itself'
    tabs, page = detection.mode_tabs, detection._preprocess_page
    picture = tabs.grab().toImage()

    def colour(point):
        return picture.pixelColor(int(point.x() * picture.devicePixelRatio()),
                                  int(point.y() * picture.devicePixelRatio()))

    inside = colour(viewport.mapTo(tabs, spot))
    margin = colour(page.mapTo(tabs, QtCore.QPoint(1, 1)))
    assert inside == margin, (
        f'the overview paints {inside.name()} where the page shows the tab pane, {margin.name()}')
    assert margin != qapp.palette().color(QtGui.QPalette.Window), (
        'the palette does not tell the tab pane from the window colour, so this proves nothing')


def test_protocol_verbs_are_reachable_in_the_real_conditioning_workspace(window):
    detection = window._analysis_panel.detection_panel
    show_page(window, 'Preprocess')
    # 2026-11-11: the Preprocess page's door is preprocess_open_button since 8661d35 (the old
    # open_conditioning_button is a hidden legacy widget); the workspace is what it opens.
    assert_whole(detection.preprocess_open_button, detection._preprocess_page)
    detection.preprocess_open_button.click()
    settle(window)
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
