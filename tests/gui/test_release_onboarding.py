"""First-use visibility choices and keyboard navigation never run science."""
from contextlib import ExitStack

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.first_analysis_guide import (
    CONTROL_LEVEL_CHOICE_KEY, EXISTING_CONTROL_LEVEL_KEY, FirstAnalysisGuide, STEPS,
)
from nanoscope.gui.panels import DetectionPanel
from nanoscope.gui.workflow_rail import WorkflowStrip
from nanoscope.gui.workflow_status import Stage


@pytest.fixture(autouse=True)
def preferences():
    settings = QtCore.QSettings()
    keys = (CONTROL_LEVEL_CHOICE_KEY, EXISTING_CONTROL_LEVEL_KEY)
    saved = {key: settings.value(key) for key in keys if settings.contains(key)}
    for key in keys:
        settings.remove(key)
    yield
    for key in keys:
        settings.remove(key)
    for key, value in saved.items():
        settings.setValue(key, value)


def test_guide_opening_does_not_pick_a_control_level(qtbot):
    guide = FirstAnalysisGuide()
    qtbot.addWidget(guide)
    with qtbot.assertNotEmitted(guide.controlLevelRequested):
        guide.show()
        guide.refresh_context()
    assert not guide.first_use_choice.isHidden()
    assert not QtCore.QSettings().contains(CONTROL_LEVEL_CHOICE_KEY)


@pytest.mark.parametrize("level", ["essential", "advanced"])
def test_explicit_first_use_choice_is_persisted_without_scientific_edits(qtbot, level):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    before = (panel.params(), panel.ml_params())
    guide = FirstAnalysisGuide()
    qtbot.addWidget(guide)
    guide.controlLevelRequested.connect(lambda value: panel.set_control_level(value, persist=True))
    with qtbot.assertNotEmitted(panel.paramsChanged), qtbot.assertNotEmitted(panel.detectRequested):
        getattr(guide, level + "_choice").click()
    assert panel.control_level() == level
    assert QtCore.QSettings().value(CONTROL_LEVEL_CHOICE_KEY) == level
    assert (panel.params(), panel.ml_params()) == before
    assert guide.first_use_choice.isHidden()


def test_existing_expert_preference_is_not_overwritten(qtbot):
    settings = QtCore.QSettings()
    settings.setValue(EXISTING_CONTROL_LEVEL_KEY, "advanced")
    settings.setValue(CONTROL_LEVEL_CHOICE_KEY, "essential")
    panel = DetectionPanel()
    guide = FirstAnalysisGuide()
    qtbot.addWidget(panel)
    qtbot.addWidget(guide)
    assert panel.control_level() == "advanced"
    assert guide.first_use_choice.isHidden()


def test_first_use_preference_can_seed_a_later_inspector(qtbot):
    QtCore.QSettings().setValue(CONTROL_LEVEL_CHOICE_KEY, "essential")
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.control_level() == "essential"


@pytest.mark.parametrize("key", [QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space])
def test_workflow_stage_is_keyboard_activatable_even_when_locked(qtbot, key):
    strip = WorkflowStrip()
    qtbot.addWidget(strip)
    chip = strip._chips[Stage.DETECT]
    assert chip.focusPolicy() == QtCore.Qt.StrongFocus
    with qtbot.waitSignal(strip.stageActivated) as signal:
        qtbot.keyClick(chip, key)
    assert signal.args == [Stage.DETECT.value]


def test_guide_teaches_all_reference_modes_and_non_install_preview():
    text = STEPS[2].instructions
    for phrase in ("Automatic", "Known open-pore current", "Selected open-pore intervals",
                   "signed current", "recording start", "Preview reference and detection",
                   "without changing stored results", "fresh detection", "not independently verified"):
        assert phrase in text


def test_about_dialog_copy_is_explicit_and_local(qtbot, monkeypatch):
    from nanoscope.gui.build_info_dialog import BuildInfoDialog
    from types import SimpleNamespace
    copied = []
    monkeypatch.setattr(QtWidgets.QApplication, "clipboard",
                        lambda: SimpleNamespace(setText=copied.append))
    dialog = BuildInfoDialog()
    qtbot.addWidget(dialog)
    assert copied == []
    dialog.copy_button.click()
    assert copied == [dialog.details.toPlainText()]
    assert "No recording/project names" in copied[0]


@pytest.fixture
def window(qtbot):
    from nanoscope.gui.main_window import MainWindow
    widget = MainWindow(recovery_enabled=False)
    qtbot.addWidget(widget)
    yield widget
    widget._analysis_controller.abort()
    widget._teardown()


def scientific_state(window):
    return (window._analysis_panel.detection_params(),
            window._analysis_panel.detection_panel.ml_params(), window._dirty,
            list(window._undo_journal), list(window._redo_journal))


def no_scientific_signals(stack, window, qtbot):
    panel = window._analysis_panel.detection_panel
    for signal in (window._session.resultsChanged, panel.paramsChanged,
                   panel.detectRequested, panel.mlDetectRequested):
        stack.enter_context(qtbot.assertNotEmitted(signal))


@pytest.mark.parametrize("level", ["essential", "advanced"])
def test_main_window_first_use_choice_works_before_loading_recording(window, qtbot, level):
    assert window._current_selection is None
    before = scientific_state(window)
    panel = window._analysis_panel.detection_panel
    with ExitStack() as stack:
        no_scientific_signals(stack, window, qtbot)
        window._act_first_analysis_guide.trigger()
        guide = window._first_analysis_guide
        assert guide.isVisible() and not guide.first_use_choice.isHidden()
        assert not QtCore.QSettings().contains(CONTROL_LEVEL_CHOICE_KEY)
        getattr(guide, level + "_choice").click()
        assert panel.control_level() == level
        assert guide.first_use_choice.isHidden()
        guide.close()
        window._act_first_analysis_guide.trigger()
        assert window._first_analysis_guide is guide
        assert guide.first_use_choice.isHidden()
    settings = QtCore.QSettings()
    assert settings.value(CONTROL_LEVEL_CHOICE_KEY) == level
    assert settings.value(EXISTING_CONTROL_LEVEL_KEY) == level
    assert scientific_state(window) == before


def test_main_window_opening_guide_preserves_existing_expert_choice(window, qtbot):
    panel = window._analysis_panel.detection_panel
    panel.set_control_level("advanced", persist=True)
    before = scientific_state(window)
    with ExitStack() as stack:
        no_scientific_signals(stack, window, qtbot)
        window._act_first_analysis_guide.trigger()
        assert window._first_analysis_guide.first_use_choice.isHidden()
    assert panel.control_level() == "advanced"
    assert not QtCore.QSettings().contains(CONTROL_LEVEL_CHOICE_KEY)
    assert scientific_state(window) == before


def test_main_window_about_action_opens_owned_local_diagnostics_without_science(window, qtbot, monkeypatch):
    from types import SimpleNamespace
    from nanoscope.gui.build_info_dialog import BuildInfoDialog
    copied = []
    monkeypatch.setattr(QtWidgets.QApplication, "clipboard",
                        lambda: SimpleNamespace(setText=copied.append))
    before = scientific_state(window)
    with ExitStack() as stack:
        no_scientific_signals(stack, window, qtbot)
        assert window.findChildren(BuildInfoDialog) == []
        window._act_about.trigger()
        dialogs = window.findChildren(BuildInfoDialog)
        assert len(dialogs) == 1
        dialog = dialogs[0]
        assert dialog.parent() is window and dialog.isVisible()
        assert dialog.details.isReadOnly()
        from nanoscope import __version__
        assert dialog.details.toPlainText().startswith(f"NanoScope {__version__}\n")
        assert copied == []
        dialog.copy_button.click()
        assert copied == [dialog.details.toPlainText()]
        dialog.close()
    assert scientific_state(window) == before
