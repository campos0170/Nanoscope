"""Editor moves retire only obsolete native layout items, never live controls.

A valid Python wrapper for an item already deleted by native reparenting can
alias a later QEvent at the same address. Hold exact old items to test that
boundary directly instead of relying on an allocator-dependent callback crash.
"""
import nanoscope  # noqa: F401
import pytest
import shiboken6
from PySide6 import QtCore, QtWidgets
from nanoscope.gui.panels import DetectionPanel
from nanoscope.gui.signal_conditioning import SignalConditioningWindow


@pytest.fixture(autouse=True)
def isolated_settings(qapp):
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName('NanoScopeTest'); qapp.setApplicationName('EditorItemLifetime')
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org); qapp.setApplicationName(app)


def _item(widget):
    root = widget.parentWidget().layout()
    for layout in [root, *root.findChildren(QtWidgets.QLayout)]:
        index = layout.indexOf(widget)
        if index >= 0:
            return layout.itemAt(index)
    raise AssertionError('control has no original layout item')


@pytest.mark.parametrize('name', [
    '_protocol_label', 'protocol_combo', 'protocol_save_btn', 'protocol_rename_btn',
    'protocol_delete_btn', 'highpass_measurement_label', 'normalize_units_label',
    'detrend_refusal_label', 'response_chain_check', 'response_plot', 'response_note',
    'first_stage', 'last_stage',
])
def test_editor_transfer_invalidates_old_item_preserving_live_controls(qtbot, name):
    panel = DetectionPanel(); qtbot.addWidget(panel)
    panel.select_stage('notch')
    old_stack = panel.filter_stack
    pages = [old_stack.widget(i) for i in range(old_stack.count())]
    widget = pages[0] if name == 'first_stage' else pages[-1] if name == 'last_stage' else getattr(panel, name)
    held = _item(widget)
    before = panel.params()
    changed = []
    panel.paramsChanged.connect(changed.append)
    try:
        parts = panel.signal_chain_editor_parts()
        assert not shiboken6.isValid(held)
        assert shiboken6.isValid(widget)
        assert all(shiboken6.isValid(page) for page in pages)
        assert [parts.stage_stack.widget(i) for i in range(parts.stage_stack.count())] == pages
        assert panel.current_stage_key() == 'notch'
        assert panel.params() == before and not changed
        assert panel.signal_chain_editor_parts() is parts
        # Exercise a pre-existing scientific connection through the real control.
        panel.notch_enabled.click()
        assert changed and changed[-1]['notch_enabled'] != before['notch_enabled']
        assert panel.params()['notch_enabled'] == changed[-1]['notch_enabled']
        panel.select_stage('lowpass')
        assert panel.filter_stack.currentWidget() is pages[0]
        assert shiboken6.isValid(panel.response_curve)
    finally:
        # A failing regression must not leave a stale wrapper to poison another test.
        if shiboken6.isValid(held):
            shiboken6.invalidate(held)


def test_install_editor_retires_toggle_item_and_keeps_native_callback(qtbot, monkeypatch):
    panel = DetectionPanel(); qtbot.addWidget(panel)
    parts = panel.signal_chain_editor_parts()
    toggle = parts.chain_response_toggle
    held = _item(toggle)
    window = SignalConditioningWindow(); qtbot.addWidget(window)
    calls = []
    actual = window._on_view_changed
    def changed():
        calls.append(toggle.isChecked())
        return actual()
    monkeypatch.setattr(window, '_on_view_changed', changed)
    try:
        window.install_editor(parts)
        assert not shiboken6.isValid(held)
        assert shiboken6.isValid(toggle) and window.isAncestorOf(toggle)
        assert shiboken6.isValid(panel.response_plot)
        before = toggle.isChecked()
        toggle.click()
        assert calls == [not before]
        installed_item = _item(toggle)
        window.install_editor(parts)
        assert shiboken6.isValid(installed_item) and installed_item.widget() is toggle
        toggle.click()
        assert calls == [not before, before]
    finally:
        window.abort()
        if shiboken6.isValid(held):
            shiboken6.invalidate(held)
