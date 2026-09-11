"""Learning-oriented visibility must not become an analysis preset."""
from contextlib import ExitStack

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui import panels


@pytest.fixture
def panel(qtbot, monkeypatch, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "controls.ini"), QtCore.QSettings.IniFormat)
    monkeypatch.setattr(panels, "QSettings", lambda: settings)
    widget = panels.DetectionPanel()
    qtbot.addWidget(widget)
    widget.resize(420, 650)
    widget.show()
    return widget


def test_default_is_advanced_and_guide_choice_is_not_persisted(panel, qtbot):
    assert panel.control_level() == "advanced"
    panel.set_control_level("essential")
    second = panels.DetectionPanel()
    qtbot.addWidget(second)
    assert second.control_level() == "advanced"
    panel.set_control_level("essential", persist=True)
    third = panels.DetectionPanel()
    qtbot.addWidget(third)
    assert third.control_level() == "essential"
    with pytest.raises(ValueError):
        panel.set_control_level("invalid")
    assert panel.control_level() == "essential"


def test_control_levels_preserve_parameters_presets_flags_and_signals(panel, qtbot):
    panel.set_recording(50_000)
    panel.set_params({"filter_cutoff_hz": 5_000., "hp_enabled": True,
                      "notch_enabled": True, "savgol_enabled": True,
                      "normalize_enabled": True, "merge_gap_s": .002, "max_dwell_s": .07})
    before, ml_before = panel.params(), panel.ml_params()
    protocols = panel.protocol_combo.currentText()
    chosen = panel._ml_preset_chosen
    disabled = {name: widget.isEnabled() for name, widget in vars(panel).items()
                if isinstance(widget, (QtWidgets.QAbstractSpinBox, QtWidgets.QCheckBox, QtWidgets.QComboBox))}
    with ExitStack() as stack:
        for signal in (panel.paramsChanged, panel.detectRequested, panel.mlDetectRequested,
                       panel.mlModelChanged, panel.previewFilterToggled,
                       panel.sweepRequested, panel.nullControlRequested, panel.recoveryRequested):
            stack.enter_context(qtbot.assertNotEmitted(signal))
        for _ in range(3):
            panel.set_control_level("essential")
            panel.detection_advanced.set_expanded(True)
            panel.set_control_level("advanced")
    assert panel.params() == before and panel.ml_params() == ml_before
    assert panel.protocol_combo.currentText() == protocols
    assert panel._ml_preset_chosen == chosen
    assert {name: getattr(panel, name).isEnabled() for name in disabled} == disabled


def test_essential_keeps_active_filters_warnings_polarity_and_expert_summary(panel, qtbot):
    panel.set_params({"normalize_enabled": True, "hp_enabled": True,
                      "merge_gap_s": .002, "max_dwell_s": .07})
    panel.set_control_level("essential")
    panel.show_signal_chain()
    assert not panel.filter_selector.isHidden()
    assert not panel.active_filters_label.isHidden()
    assert not panel.normalize_units_label.isHidden()
    assert not panel.highpass_measurement_label.isHidden()
    assert panel.response_plot.isHidden()
    assert panel.filter_selector.count() == len(panel._filter_specs)
    assert "NORMALIZE ON" in panel.control_level_summary()
    assert "high-pass" in panel.control_level_summary()
    assert "merge 2 ms" in panel.control_level_summary()
    assert "max dwell 0.07 s" in panel.control_level_summary()
    panel.show_detection_tab()
    assert panel.polarity_override.isVisible()
    assert panel.detect_button.isVisible()
    assert all(button.isHidden() for button in (panel.sweep_button, panel.null_button, panel.recovery_button))
    assert panel.advanced_tools_button.isVisible()
    assert panel.detection_advanced_summary.isVisible()
    panel.detection_advanced.set_expanded(True)
    assert panel.max_dwell_s.isVisible() and panel.merge_gap_ms.isVisible()
    panel.advanced_tools_button.click()
    assert panel.control_level() == "advanced" and panel.sweep_button.isVisible()


def test_advanced_disclosure_preference_restores_without_touching_values(panel):
    panel.detection_advanced.set_expanded(True)
    panel.set_control_level("essential")
    assert not panel.detection_advanced.toggle.isChecked()
    panel.set_control_level("advanced")
    assert panel.detection_advanced.toggle.isChecked()


def test_customized_or_loaded_ml_is_disclosed_and_navigation_reveals_it(panel, monkeypatch):
    panel.set_control_level("essential")
    index = panel.mode_tabs.indexOf(panel._ml_page)
    assert not panel.mode_tabs.isTabVisible(index)
    panel.ml_norm_center.setChecked(not panel.ml_norm_center.isChecked())
    assert panel.mode_tabs.isTabVisible(index)
    assert "ML setup retained" in panel.control_level_summary()
    before = panel.ml_params()
    panel.show_ml_tab()
    assert panel.control_level() == "advanced"
    assert panel.mode_tabs.currentWidget() is panel._ml_page
    assert panel.ml_params() == before


@pytest.mark.parametrize("width", [360, 420])
def test_narrow_essential_actions_and_reference_link_remain_reachable(panel, qtbot, width):
    panel.resize(width, 600)
    panel.set_control_level("essential")
    panel.show_detection_tab()
    qtbot.wait(30)
    for button in (panel.detect_button, panel.advanced_tools_button, panel.baseline_reference_button):
        assert button.visibleRegion().contains(button.rect().center()), button.text()
    before = panel.params()
    panel.baseline_reference_button.click()
    assert panel.mode_tabs.currentWidget() is panel._preprocess_page
    assert panel.preprocess_section_selector.currentIndex() == 1
    assert panel.params() == before


def test_learning_view_does_not_touch_project_history_or_results(panel, qtbot):
    from nanoscope.gui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    detection = window._analysis_panel.detection_panel
    params = window._analysis_panel.detection_params()
    undo, redo = list(window._undo_journal), list(window._redo_journal)
    dirty = window._dirty
    try:
        with qtbot.assertNotEmitted(window._session.resultsChanged):
            with qtbot.assertNotEmitted(detection.paramsChanged):
                detection.set_control_level("essential")
                detection.baseline_reference_button.click()
                detection.set_control_level("advanced")
        assert window._analysis_panel.detection_params() == params
        assert list(window._undo_journal) == undo and list(window._redo_journal) == redo
        assert window._dirty == dirty
    finally:
        window._analysis_controller.abort()
        window._teardown()
