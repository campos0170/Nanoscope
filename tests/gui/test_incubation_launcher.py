"""Unified entry, exact checkpoint routing and independent study lifetimes."""
import pytest

from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.incubation_launcher import IncubationStudyLauncher
from nanoscope.analysis import manual_workspace as checkpoint


@pytest.fixture
def main(qtbot, monkeypatch):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    monkeypatch.setattr(window, '_confirm_close_choice', lambda: 'discard')
    yield window
    window._dirty = False
    window._teardown()


def test_single_menu_entry_and_explicit_reference_recipe(main):
    labels = [a.text() for a in main._analysis_menu.actions()]
    assert labels.count('Incubation study…') == 1
    assert 'Manual study workspace…' not in labels
    assert 'Acid incubation workspace…' not in labels
    launcher = main._open_incubation_study()
    assert isinstance(launcher, IncubationStudyLauncher)
    assert main._open_incubation_study() is launcher
    assert [launcher._profile.itemData(i) for i in range(launcher._profile.count())] == ['general', 'zn', 'cu', 'acid']
    assert main._incubation_reference_window is None
    assert main._open_incubation_reference() is main._incubation_reference_window


@pytest.mark.parametrize('profile', ['general', 'zn', 'cu', 'acid'])
def test_new_profile_checkpoint_and_shared_parent(main, tmp_path, profile):
    launcher = main._open_incubation_study()
    window = launcher.create_study(tmp_path/profile, profile=profile, name='My incubation series', sample='Test sample')
    assert window.parent() is main
    assert window.state['name'] == 'My incubation series'
    if profile != 'acid': assert window.state['model_path'] == ''
    else: assert 'settings' in window.state
    assert checkpoint.load_workspace(window.folder)['name'] == 'My incubation series'
    assert callable(window.save_main_curation)
    assert launcher.open_study(window.folder) is window
    assert len(launcher._windows) == 1
    assert not main._session.documents()


def test_open_saved_profile_restores_settings_and_keeps_other_study(main, qtbot, tmp_path):
    from nanoscope.gui.manual_study_window import ManualStudyWindow
    saved = ManualStudyWindow(main); qtbot.addWidget(saved)
    saved.new_workspace(tmp_path/'saved-cu', profile='cu', name='Copper series')
    saved.state['comparison_settings']['n_blocks'] = 37
    saved._revision = checkpoint.save_workspace(saved.state, saved.folder, expected_revision=saved._revision)
    launcher = main._open_incubation_study()
    active = launcher.create_study(tmp_path/'active-zn', profile='zn')
    active.state['user_draft_marker'] = 'keep this in memory'
    restored = launcher.open_study(saved.folder)
    qtbot.waitUntil(lambda: not restored.is_busy(), timeout=30000)
    assert restored in launcher._windows
    assert restored.state['comparison_settings']['n_blocks'] == 37
    assert active.state['user_draft_marker'] == 'keep this in memory'
    assert launcher.open_study(saved.folder) is restored
    # A later operation failure belongs to the study; the initial-open handlers
    # must not close or delete this successfully opened window.
    restored.operationFailed.emit('A later operation failed')
    assert restored.isVisible()
    assert restored in launcher._windows


def test_unknown_saved_kind_leaves_existing_study_untouched(main, tmp_path):
    launcher = main._open_incubation_study()
    active = launcher.create_study(tmp_path/'active')
    checkpoint.save_workspace(dict(workspace_kind='future_unrecognized_v9', records=[]), tmp_path/'unknown')
    with pytest.raises(ValueError): launcher.open_study(tmp_path/'unknown')
    assert launcher._windows == [active]
    assert main._manual_study_window is active


def test_invalid_loaded_checkpoint_does_not_replace_active_study(main, qtbot, tmp_path):
    launcher = main._open_incubation_study()
    active = launcher.create_study(tmp_path/'active')
    state = dict(active.state); state['comparison_settings'] = 'invalid settings'
    checkpoint.save_workspace(state, tmp_path/'broken')
    launcher.open_study(tmp_path/'broken')
    qtbot.waitUntil(lambda: not launcher._opening, timeout=30000)
    assert launcher._windows == [active]
    assert main._manual_study_window is active
    assert 'settings' in launcher._status.text()


def test_new_workspace_cannot_replace_existing_folder(main, tmp_path):
    launcher = main._open_incubation_study()
    active = launcher.create_study(tmp_path/'active', profile='zn')
    with pytest.raises(ValueError, match='already open'):
        launcher.create_study(active.folder, profile='cu')
    assert launcher._windows == [active]


def test_close_during_load_allows_reopening_same_study(main, qtbot, monkeypatch, tmp_path):
    import threading
    import time
    from nanoscope.analysis.cancel import check_cancelled
    from nanoscope.gui.manual_study_window import _empty_state
    folder = tmp_path/'saved'
    checkpoint.save_workspace(_empty_state(), folder)
    original = checkpoint.load_workspace
    entered = threading.Event()
    def slow(*args, **kwargs):
        entered.set()
        while True:
            check_cancelled(kwargs.get('cancel')); time.sleep(.005)
    monkeypatch.setattr(checkpoint, 'load_workspace', slow)
    launcher = main._open_incubation_study()
    loading = launcher.open_study(folder)
    qtbot.waitUntil(entered.is_set, timeout=3000)
    loading.close()
    assert folder.resolve() not in launcher._opening
    monkeypatch.setattr(checkpoint, 'load_workspace', original)
    reopened = launcher.open_study(folder)
    assert reopened is not loading
    qtbot.waitUntil(lambda: not reopened.is_busy(), timeout=30000)
    assert reopened.folder == folder
    assert reopened in launcher._windows


def test_legacy_schema_only_acid_uses_acid_window_reference(main, qtbot, tmp_path):
    from nanoscope.analysis.acid_incubation import new_state
    from nanoscope.gui.acid_study_window import AcidStudyWindow
    folder = tmp_path/'legacy-acid'
    checkpoint.save_workspace(new_state('Legacy acid'), folder)
    launcher = main._open_incubation_study()
    manual = launcher.create_study(tmp_path/'manual')
    acid = launcher.open_study(folder)
    qtbot.waitUntil(lambda: not acid.is_busy(), timeout=30000)
    assert isinstance(acid, AcidStudyWindow)
    assert main._acid_study_window is acid
    assert main._manual_study_window is manual
