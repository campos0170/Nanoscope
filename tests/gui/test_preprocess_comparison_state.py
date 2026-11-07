"""Scope and freshness contracts for the shared Preprocess comparison workspace."""
import nanoscope  # Qt bootstrap before Qt import
import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.gui.baseline_panel import BaselinePanel
from nanoscope.gui.baseline_workspace import (
    BaselineCorrectionWorkspace, DISPLAY_ZERO, RUNTIME_RETAIN_OFFSET,
)


@pytest.fixture
def workspace(qtbot):
    reference = BaselinePanel()
    widget = BaselineCorrectionWorkspace(
        reference,
        method_choices=(("block median", "block_median"),
                        ("event-masked median", "masked_median")),
    )
    qtbot.addWidget(widget)
    return widget


def preview(widget):
    t = np.arange(100) / 1000.
    raw = np.linspace(100., 102., len(t))
    widget.set_preview(t, raw, raw - 1., np.ones(len(t)))


@pytest.mark.parametrize('source', ['anchors', 'intervals', 'recording'])
def test_custom_recipe_keeps_active_runtime_and_display_controls_operable(workspace, source, qtbot):
    workspace.show()
    workspace.sync_runtime({'detrend_enabled': True, 'detrend_method': 'masked_median'})
    workspace.baseline_panel.show_track.setChecked(True)
    workspace.baseline_panel.subtract_view.setChecked(True)
    workspace.background.source.setCurrentIndex(workspace.background.source.findData(source))
    assert workspace.runtime_enabled.isChecked()
    assert workspace.runtime_enabled.isEnabled()
    assert workspace.runtime_method.isEnabled()
    assert workspace.advanced_button.isEnabled()
    assert workspace.baseline_panel.display_group.isVisible()
    assert workspace.baseline_panel.subtract_view.isChecked()
    with qtbot.waitSignal(workspace.runtimeEnabledRequested) as signal:
        workspace.runtime_enabled.click()
    assert signal.args == [False]


def test_export_status_cannot_make_unchanged_stale_plot_appear_current(workspace):
    preview(workspace)
    plotted = workspace.corrected_curve.getData()[1].copy()
    workspace.mark_preview_stale('Reference windows changed')
    workspace.set_busy(True, 'Exporting current recipe')
    workspace.set_busy(False)
    workspace.set_status('Wrote new-settings.csv')
    assert workspace.is_preview_stale()
    assert 'Out of date' in workspace.freshness_label.text()
    assert workspace.freshness_label.toolTip() == 'Reference windows changed'
    assert workspace.preview_status.text() == 'Wrote new-settings.csv'
    np.testing.assert_array_equal(plotted, workspace.corrected_curve.getData()[1])
    preview(workspace)
    assert not workspace.is_preview_stale()
    assert 'Current comparison' in workspace.freshness_label.text()


def test_expanded_plot_keeps_freshness_and_returns_to_original_slot(workspace, qtbot):
    preview(workspace)
    workspace.set_workspace_hosted(True)
    workspace.expand_comparison()
    workspace.mark_preview_stale('Recipe changed')
    workspace.set_status('Export succeeded')
    assert 'Out of date' in workspace._expanded_freshness.text()
    assert workspace._expanded_status.text() == 'Export succeeded'
    workspace._expanded_dialog.close()
    qtbot.waitUntil(lambda: workspace._expanded_dialog is None)
    index = workspace._preview_layout.indexOf(workspace._expanded_placeholder)
    assert workspace._preview_layout.itemAt(index + 1).widget() is workspace.preview
    assert 'Out of date' in workspace.freshness_label.text()


def test_mode_labels_explain_estimators_without_changing_saved_tokens(workspace, qtbot):
    before = workspace.form_state()
    assert workspace.output_mode_combo.itemText(0) == 'Event-reference residual (zero)'
    assert workspace.output_mode_combo.itemText(1) == 'Drift-stage correction (offset retained)'
    assert 'Event reference Rough/Base' in workspace.contract_label.text()
    workspace.output_mode_combo.setCurrentIndex(1)
    assert workspace.output_mode() == RUNTIME_RETAIN_OFFSET
    assert 'Filters & drift parameters' in workspace.contract_label.text()
    workspace.background.source.setCurrentIndex(workspace.background.source.findData('anchors'))
    assert workspace.output_mode() == RUNTIME_RETAIN_OFFSET
    assert 'median(background)' in workspace.output_mode_combo.currentText()
    workspace.restore_form_state(before)
    assert workspace.output_mode() == DISPLAY_ZERO
    assert workspace.background.source_key() == 'automatic'


def test_hosted_source_switch_does_not_hide_reparented_reference(workspace, qtbot):
    workspace.set_workspace_hosted(True)
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    layout = QtWidgets.QVBoxLayout(host)
    layout.addWidget(workspace.baseline_panel)
    host.show()
    workspace.background.source.setCurrentIndex(workspace.background.source.findData('anchors'))
    assert workspace.baseline_panel.isVisible()
    assert workspace.background.parentWidget() is workspace._recipe_host
    assert workspace.preview_box.parentWidget() is workspace._comparison_splitter
    assert workspace.action_widget.parentWidget() is workspace
    assert workspace.export_button.parentWidget() is workspace.action_widget


def test_shared_reference_link_is_navigation_only(workspace, qtbot):
    workspace.set_workspace_hosted(True)
    workspace.sync_reference_summary('Known current: 100 pA')
    assert '100 pA' in workspace.reference_summary.text()
    before = workspace.form_state()
    with qtbot.waitSignal(workspace.referenceEditRequested):
        workspace.edit_reference_button.click()
    assert workspace.form_state() == before


def test_layout_roundtrip_does_not_change_parameters_or_preview(workspace):
    preview(workspace)
    before = workspace.form_state()
    plotted = workspace.corrected_curve.getData()[1].copy()
    workspace.set_workspace_hosted(True)
    workspace.set_workspace_hosted(True)
    workspace.set_workspace_hosted(False)
    workspace.set_workspace_hosted(True)
    assert workspace.form_state() == before
    np.testing.assert_array_equal(plotted, workspace.corrected_curve.getData()[1])


def _layout_item(widget):
    root = widget.parentWidget().layout()
    for layout in [root, *root.findChildren(QtWidgets.QLayout)]:
        index = layout.indexOf(widget)
        if index >= 0:
            return layout.itemAt(index)
    raise AssertionError('Widget has no layout item')


def test_hosting_retires_old_layout_items_including_inspector_action_holder(workspace, qtbot):
    import shiboken6
    from nanoscope.gui.widget_transfer import prepare_widget_move
    pinned = QtWidgets.QWidget()
    qtbot.addWidget(pinned)
    pinned_layout = QtWidgets.QVBoxLayout(pinned)
    prepare_widget_move(workspace.action_widget)
    pinned_layout.addWidget(workspace.action_widget)
    controls = (workspace.background, workspace.preview_box, workspace.action_widget,
                workspace.output_mode_combo, workspace.contract_label, workspace.export_button)
    items = [_layout_item(control) for control in controls]
    try:
        workspace.set_workspace_hosted(True)
        assert all(not shiboken6.isValid(item) for item in items)
        assert all(shiboken6.isValid(control) for control in controls)
        assert pinned_layout.count() == 0
        assert workspace.action_widget.parentWidget() is workspace
    finally:
        for item in items:
            if shiboken6.isValid(item):
                shiboken6.invalidate(item)


def test_expand_and_return_retire_preview_layout_items(workspace, qtbot):
    import shiboken6
    preview(workspace)
    held = _layout_item(workspace.preview)
    expanded_item = None
    try:
        workspace.expand_comparison()
        assert not shiboken6.isValid(held)
        expanded_item = _layout_item(workspace.preview)
        workspace._expanded_dialog.close()
        assert not shiboken6.isValid(expanded_item)
        assert shiboken6.isValid(workspace.preview)
    finally:
        for item in (held, expanded_item):
            if item is not None and shiboken6.isValid(item):
                shiboken6.invalidate(item)
