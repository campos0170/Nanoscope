"""Hosted Preprocess routes retain authoritative widgets and explicit effect scope."""
from __future__ import annotations

import os
import nanoscope  # noqa: F401
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PySide6 import QtCore
from nanoscope.analysis.conditioning_preview import ConditioningSource
from nanoscope.gui.panels import DetectionPanel
from nanoscope.gui.baseline_panel import BaselinePanel
from nanoscope.gui.baseline_workspace import BaselineCorrectionWorkspace
from nanoscope.gui.signal_conditioning import SignalConditioningWindow


@pytest.fixture
def workspace(qapp, qtbot):
    det = DetectionPanel()
    reference = BaselinePanel()
    comparison = BaselineCorrectionWorkspace(reference,
        method_choices=(('block median', 'block_median'),))
    window = SignalConditioningWindow()
    qtbot.addWidget(det)
    qtbot.addWidget(window)
    window.install_editor(det.signal_chain_editor_parts())
    window.install_preprocess_pages(reference, comparison, det.filter_preview,
                                    det.preview_error_label)
    window.resize(1120, 900)
    window.show()
    yield window, det, reference, comparison
    window.abort()


def test_route_changes_preserve_authoritative_widgets_and_settings(workspace, qtbot):
    window, det, reference, comparison = workspace
    det.set_params({'detrend_enabled': True, 'detrend_method': 'block_median'})
    comparison.sync_runtime(det.params())
    before = det.params(), reference.params(), comparison.form_state()
    requests = []
    comparison.runtimeEnabledRequested.connect(requests.append)
    comparison.refreshRequested.connect(lambda: requests.append('calculate'))
    comparison.exportRequested.connect(lambda: requests.append('export'))
    for key in ('reference', 'comparison', 'filters', 'comparison'):
        assert window.show_workspace_section(key)
        assert window.current_workspace_section() == key
        assert (det.params(), reference.params(), comparison.form_state()) == before
        for control in (reference.show_track, reference.subtract_view,
                        det.filter_preview, comparison.runtime_enabled,
                        comparison.advanced_button):
            assert control.window() is window
            assert control.isVisibleTo(window)
        assert window.reference_scroll.widget().isAncestorOf(reference)
        assert comparison is window.workspace_tabs.widget(2)
    assert requests == []
    assert not window.show_workspace_section('unknown')
    assert window.current_workspace_section() == 'comparison'


def test_custom_comparison_never_hides_global_display_or_runtime(workspace):
    window, det, reference, comparison = workspace
    comparison.sync_runtime({'detrend_enabled': True, 'detrend_method': 'block_median'})
    reference.show_track.setChecked(True)
    reference.subtract_view.setChecked(True)
    window.show_workspace_section('comparison')
    for source in ('anchors', 'intervals', 'recording'):
        comparison.background.source.setCurrentIndex(comparison.background.source.findData(source))
        assert comparison.runtime_enabled.isChecked()
        for control in (comparison.runtime_enabled, comparison.advanced_button,
                        reference.show_track, reference.subtract_view):
            assert control.isEnabled()
            assert control.isVisibleTo(window)
        assert reference.show_track.isChecked() and reference.subtract_view.isChecked()
    # The same live toggle still emits its intent under a custom background.
    changed = []
    comparison.runtimeEnabledRequested.connect(changed.append)
    comparison.runtime_enabled.click()
    assert changed == [False]


def test_installation_is_idempotent_and_target_context_tracks_source(workspace):
    window, det, reference, comparison = workspace
    window.install_preprocess_pages(reference, comparison, det.filter_preview,
                                    det.preview_error_label)
    assert window.workspace_tabs.count() == 3
    assert [window.workspace_tabs.tabText(i) for i in range(3)] == [
        'Filters && drift', 'Event reference', 'Background comparison']
    window.show_workspace_section('reference')
    window.set_source_label('first.csv / current')
    window.set_source(ConditioningSource('first', 0, 'first.csv',
        np.zeros(100), 1000., 'pA', 0.))
    assert '1,000 Hz' in window.workspace_target.text()
    assert 'pA' in window.workspace_target.text()
    window.set_source_label('second.csv / current')
    window.set_source(ConditioningSource('second', 0, 'second.csv',
        np.zeros(100), 2000., 'nA', 0.))
    assert 'second.csv' in window.workspace_target.text()
    assert '2,000 Hz' in window.workspace_target.text()
    assert 'nA' in window.workspace_target.text()
    assert 'first.csv' not in window.workspace_target.text()
    window.set_source_label('')
    window.set_source(None)
    assert window.workspace_target.text() == 'Target: Select a current recording'
    assert window.windowTitle() == 'Preprocess workspace'


def test_hidden_chain_route_does_not_start_a_new_chain_preview(workspace):
    window, det, reference, comparison = workspace
    window.show_workspace_section('comparison')
    window._schedule(immediate=True)
    assert not window._debounce.isActive()
    assert window._dirty
    window.show_workspace_section('filters')
    assert window._debounce.isActive()



def test_display_refusals_remain_visible_on_all_routes_and_clear(workspace):
    window, det, reference, comparison = workspace
    before = det.params(), reference.params()
    processed_problem = "The filter cutoff is above Nyquist."
    baseline_problem = "The baseline could not be drawn here: insufficient samples."
    det.set_preview_error(processed_problem)
    reference.say_view_problem(baseline_problem)
    for route in ('filters', 'reference', 'comparison'):
        window.show_workspace_section(route)
        assert det.preview_error_label.isVisibleTo(window)
        assert window.workspace_view_status.isVisibleTo(window)
        assert processed_problem in det.preview_error_label.toolTip()
        assert baseline_problem in window.workspace_view_status.toolTip()
        assert (det.params(), reference.params()) == before
    det.set_preview_error('')
    reference.say_view_problem('')
    assert det.preview_error_label.isHidden()
    assert window.workspace_view_status.isHidden()
    assert window.workspace_view_status.text() == ''
    assert (det.params(), reference.params()) == before
