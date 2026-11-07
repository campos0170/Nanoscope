"""Actions stay reachable in the current Inspector, independently of page scroll.

The former whole-column scroll was replaced by page-owned scroll areas and pinned
commands. Test the same visible/clickable guarantees in that current hierarchy,
including the real Signal Conditioning door and the smallest supported window.
"""
import pytest
from PySide6 import QtCore, QtWidgets
from .inspector_layout_rig import build, settle, show_page, rectangle, assert_whole, reveal, close_window

_SCREENS = [(1512, 900), (1440, 900), (1280, 800)]
_ACTIONS = ('detect_button', 'sweep_button', 'null_button', 'recovery_button')
_LONG_REASON = ('Detection runs on CURRENT channels; this is a voltage channel '
                '(your Calibration-panel override; the file says current).')
_SHORT_REASON = 'Sample rate unknown — dwell time is undefined without a time base.'


@pytest.fixture(autouse=True)
def isolated_settings(qapp):
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName('NanoScopeTest'); qapp.setApplicationName('CurrentInspectorFold')
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org); qapp.setApplicationName(app)


@pytest.fixture
def make_window(qtbot):
    windows = []
    def make(size=(1512, 900), width=None):
        window = build(qtbot, size, width)
        windows.append(window)
        return window
    yield make
    for window in windows:
        close_window(window)


@pytest.mark.parametrize('size', _SCREENS, ids=lambda s: f'{s[0]}x{s[1]}')
@pytest.mark.parametrize('action', _ACTIONS)
def test_every_detection_action_is_on_screen_without_scrolling(make_window, size, action):
    window = make_window(size)
    area = show_page(window, 'Detection')
    detection = window._analysis_panel.detection_panel
    button = getattr(detection, action)
    assert area.verticalScrollBar().value() == 0
    assert_whole(button, detection._detection_page)
    before = rectangle(button, window)
    area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())
    settle(window)
    assert rectangle(button, window) == before
    assert_whole(button, detection._detection_page)


@pytest.mark.parametrize('size', _SCREENS)
def test_the_detection_actions_actually_paint(make_window, size):
    window = make_window(size)
    show_page(window, 'Detection')
    detection = window._analysis_panel.detection_panel
    for name in _ACTIONS:
        button = getattr(detection, name)
        assert_whole(button)
        # A full painted rectangle can still be under another widget; hit-test too.
        center = button.mapTo(window, button.rect().center())
        hit = window.childAt(center)
        assert hit is button or button.isAncestorOf(hit)


@pytest.mark.parametrize('size', _SCREENS)
def test_the_action_row_keeps_the_x_fix_it_was_given(make_window, size):
    window = make_window(size)
    show_page(window, 'Detection')
    detection = window._analysis_panel.detection_panel
    boxes = [rectangle(getattr(detection, name), detection._detection_page) for name in _ACTIONS]
    assert boxes[0].top() == boxes[1].top()
    assert boxes[2].top() == boxes[3].top() > boxes[0].bottom()
    assert boxes[0].right() < boxes[1].left()
    for name in _ACTIONS:
        assert_whole(getattr(detection, name), detection._detection_page)


@pytest.mark.parametrize('size', _SCREENS)
def test_pinning_the_actions_steals_no_room_from_the_parameters(make_window, size):
    window = make_window(size)
    area = show_page(window, 'Detection')
    detection = window._analysis_panel.detection_panel
    detection.detection_advanced.set_expanded(True)
    settle(window)
    assert area.viewport().height() > 80
    actions = detection.page_actions[detection._detection_page]
    assert not area.isAncestorOf(detection.detect_button)
    assert actions.isAncestorOf(detection.detect_button)
    assert rectangle(actions, detection._detection_page).bottom() < rectangle(area, detection._detection_page).top()
    for name in ('noise_policy', 'noise_window_s', 'k_in', 'k_out', 'min_dwell_s',
                 'polarity_override', 'max_dwell_s', 'merge_gap_ms'):
        reveal(window, getattr(detection, name))
    assert_whole(detection.detect_button)


@pytest.mark.parametrize('size', _SCREENS)
def test_response_design_has_a_visible_door_and_opens_the_actual_plot(make_window, size):
    window = make_window(size)
    show_page(window, 'Preprocess')
    detection = window._analysis_panel.detection_panel
    button = detection.open_conditioning_button
    assert 'Signal Conditioning' in button.text()
    assert_whole(button, detection._preprocess_page)
    before = detection.params()
    button.click()
    settle(window)
    workspace = window._signal_conditioning
    assert workspace is not None and workspace.isVisible()
    workspace.preview_tabs.setCurrentIndex(2)
    settle(workspace)
    assert_whole(detection.response_plot)
    assert detection.params() == before


def test_noise_controls_and_policy_text_do_not_floor_the_inspector(make_window):
    window = make_window(width=460)
    area = show_page(window, 'Detection')
    detection = window._analysis_panel.detection_panel
    assert detection.noise_policy.minimumSizeHint().width() <= 260
    assert area.horizontalScrollBar().maximum() == 0
    for policy in ('auto', 'local', 'global'):
        detection.noise_policy.setCurrentIndex(detection.noise_policy.findData(policy))
        reveal(window, detection.noise_policy)
        assert_whole(detection.detect_button)


def test_the_gate_banner_does_not_move_the_pinned_preprocess_actions(make_window):
    window = make_window()
    show_page(window, 'Preprocess')
    detection = window._analysis_panel.detection_panel
    names = ('preprocess_section_selector', 'open_conditioning_button', 'filter_preview')
    geometries = []
    for runnable, reason in ((False, _LONG_REASON), (True, ''), (False, _SHORT_REASON)):
        detection.set_runnable(runnable, reason)
        settle(window)
        geometries.append([rectangle(getattr(detection, n), window) for n in names])
        for name in names:
            assert_whole(getattr(detection, name))
    assert geometries[0] == geometries[1] == geometries[2]


def test_the_gate_banner_does_not_move_the_detection_actions(make_window):
    window = make_window()
    show_page(window, 'Detection')
    detection = window._analysis_panel.detection_panel
    positions = []
    for runnable, reason in ((False, _LONG_REASON), (True, ''), (False, _SHORT_REASON)):
        detection.set_runnable(runnable, reason)
        settle(window)
        positions.append([rectangle(getattr(detection, n), window) for n in _ACTIONS])
    assert positions[0] == positions[1] == positions[2]


def test_gate_keeps_the_exact_reason_and_run_state(make_window):
    window = make_window()
    show_page(window, 'Detection')
    detection = window._analysis_panel.detection_panel
    detection.set_runnable(False, _LONG_REASON)
    settle(window)
    assert not detection._gate_label.isHidden()
    assert detection._gate_label.text() == _LONG_REASON
    assert not detection.detect_button.isEnabled()
    assert_whole(detection.detect_button)
    detection.set_runnable(True, '')
    settle(window)
    assert detection._gate_label.isHidden() and detection.detect_button.isEnabled()
    assert_whole(detection.detect_button)
