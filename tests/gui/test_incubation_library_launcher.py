"""Saved studies reopen through the complete engine loader without a chooser."""
import threading
import time

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.analysis.cancel import check_cancelled
from nanoscope.gui import incubation_launcher as launcher_module
from nanoscope.gui.incubation_launcher import IncubationStudyLauncher
from nanoscope.gui.incubation_library import StudyLibrary, StudyLibraryError
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.manual_study_window import ManualStudyWindow, _empty_state


@pytest.fixture
def main(qtbot, monkeypatch):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    monkeypatch.setattr(window, '_confirm_close_choice', lambda: 'discard')
    yield window
    window._dirty = False
    window._teardown()


@pytest.fixture
def library(tmp_path):
    return StudyLibrary(tmp_path/'Library')


def saved_study(folder, **changes):
    state = _empty_state()
    state.update(changes)
    checkpoint.save_workspace(state, folder)
    return folder


def select(launcher, folder):
    key = str(folder.resolve())
    for index in range(launcher._saved.topLevelItemCount()):
        item = launcher._saved.topLevelItem(index)
        if item.data(0, QtCore.Qt.UserRole) == key:
            launcher._saved.setCurrentItem(item)
            return item
    raise AssertionError('Study was not listed: '+key)


def test_registered_aliases_survive_restart_and_open_without_chooser(main, library, tmp_path, qtbot, monkeypatch):
    zinc = saved_study(tmp_path/'legacy-zinc', name='Manual study')
    copper = saved_study(tmp_path/'legacy-copper', name='Manual study')
    library.remember(zinc, 'Zn-insulin incubation')
    library.remember(copper, 'Cu-insulin incubation')
    first = IncubationStudyLauncher(main, library=library); qtbot.addWidget(first)
    assert first._saved.topLevelItemCount() == 2
    first.close()
    launcher = IncubationStudyLauncher(main, library=StudyLibrary(library.root)); qtbot.addWidget(launcher)
    launcher._profile.setCurrentIndex(launcher._profile.findData('acid'))
    assert launcher._saved.topLevelItemCount() == 2
    item = select(launcher, copper)
    assert item.text(0) == 'Cu-insulin incubation'
    def no_chooser(*_args, **_kwargs):
        raise AssertionError('A listed study should open without a folder chooser')
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', no_chooser)
    qtbot.mouseClick(launcher._open_selected, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: not launcher._opening, timeout=30000)
    restored = launcher._windows[0]
    assert isinstance(restored, ManualStudyWindow)
    assert restored.folder == copper
    assert restored.state['name'] == 'Manual study'
    assert restored.state.get('study_profile', 'general') == 'general'
    assert restored.state['comparison_settings'] == checkpoint.load_workspace(copper)['comparison_settings']
    assert next(e for e in library.entries() if e.path == copper).label == 'Cu-insulin incubation'
    # A double-click reuses the loaded document, retaining unsaved state.
    restored.state['user_draft_marker'] = 'preserve me'
    launcher._saved.itemDoubleClicked.emit(select(launcher, copper), 0)
    assert launcher._windows == [restored]
    assert restored.state['user_draft_marker'] == 'preserve me'


def test_new_and_browse_choosers_start_in_library_and_last_saved_folder(main, library, tmp_path, qtbot, monkeypatch):
    folder = saved_study(tmp_path/'saved'); library.remember(folder)
    launcher = IncubationStudyLauncher(main, library=library); qtbot.addWidget(launcher)
    calls = []
    def choose(parent, title, directory):
        calls.append((title, directory)); return ''
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', choose)
    launcher._choose_new(); launcher._choose_open()
    assert calls[0][1] == str(library.root)
    assert calls[1][1] == str(folder)
    assert str(library.root) in launcher._library_location.text()


def test_explicit_open_of_already_loaded_legacy_window_adds_it_once(main, library, tmp_path, qtbot):
    window = ManualStudyWindow(main); qtbot.addWidget(window)
    window.new_workspace(tmp_path/'legacy-entry', profile='cu')
    main._manual_study_window = window
    launcher = IncubationStudyLauncher(main, library=library); qtbot.addWidget(launcher)
    launcher.refresh(); launcher.refresh()
    assert library.entries() == []
    window.state['user_draft_marker'] = 'keep in memory'
    assert launcher.open_study(window.folder) is window
    assert [entry.path for entry in library.entries()] == [window.folder]
    assert window.state['user_draft_marker'] == 'keep in memory'


def test_create_registers_success_and_failed_load_never_repoints_alias(main, library, tmp_path, qtbot):
    launcher = IncubationStudyLauncher(main, library=library); qtbot.addWidget(launcher)
    active = launcher.create_study(tmp_path/'active', profile='zn', name='My zinc study')
    assert [(entry.path, entry.label) for entry in library.entries()] == [(active.folder, 'My zinc study')]
    broken = saved_study(tmp_path/'broken', name='Manual study', comparison_settings='invalid settings')
    library.remember(broken, 'Cu-insulin incubation')
    registry = (library.root/'studies.json').read_bytes()
    launcher.refresh(); select(launcher, broken)
    launcher.open_selected_study()
    qtbot.waitUntil(lambda: not launcher._opening, timeout=30000)
    assert launcher._windows == [active]
    assert main._manual_study_window is active
    assert (library.root/'studies.json').read_bytes() == registry
    assert 'settings' in launcher._status.text()


def test_cancelled_open_is_not_remembered(main, library, tmp_path, qtbot, monkeypatch):
    folder = saved_study(tmp_path/'saved')
    entered = threading.Event()
    def slow(*args, **kwargs):
        entered.set()
        while True:
            check_cancelled(kwargs.get('cancel')); time.sleep(.005)
    monkeypatch.setattr(checkpoint, 'load_workspace', slow)
    launcher = IncubationStudyLauncher(main, library=library); qtbot.addWidget(launcher)
    loading = launcher.open_study(folder)
    qtbot.waitUntil(entered.is_set, timeout=3000)
    assert library.entries() == []
    loading.close()
    assert not launcher._opening
    assert library.entries() == []


def test_missing_study_is_visible_and_show_folder_requires_explicit_action(main, library, tmp_path, qtbot, monkeypatch):
    folder = saved_study(tmp_path/'saved'); library.remember(folder, 'Zn-insulin incubation')
    opened = []
    monkeypatch.setattr(QtGui.QDesktopServices, 'openUrl', lambda url: opened.append(url.toLocalFile()) or True)
    launcher = IncubationStudyLauncher(main, library=library); qtbot.addWidget(launcher)
    assert opened == []
    qtbot.mouseClick(launcher._show_folder, QtCore.Qt.LeftButton)
    assert opened == [str(folder)]
    folder.rename(tmp_path/'moved')
    launcher.refresh()
    assert launcher._saved.topLevelItem(0).text(0) == 'Zn-insulin incubation'
    assert launcher._saved.topLevelItem(0).text(2) == 'Unavailable'
    assert not launcher._open_selected.isEnabled()
    assert not launcher._show_folder.isEnabled()
    assert launcher._browse.isEnabled()
    assert 'missing' in launcher._library_note.text()


@pytest.mark.parametrize('action', ['open', 'create'])
def test_registry_write_error_keeps_successfully_opened_or_created_study(main, library, tmp_path, qtbot, monkeypatch, action):
    folder = tmp_path/action
    if action == 'open': saved_study(folder)
    def fail(*args, **kwargs): raise StudyLibraryError('Library is read-only')
    monkeypatch.setattr(library, 'remember', fail)
    launcher = IncubationStudyLauncher(main, library=library); qtbot.addWidget(launcher)
    window = launcher.open_study(folder) if action == 'open' else launcher.create_study(folder)
    qtbot.waitUntil(lambda: not launcher._opening, timeout=30000)
    assert launcher._windows == [window]
    assert window.folder == folder
    assert window.isVisible()
    assert main._manual_study_window is window
    assert 'read-only' in launcher._status.text()
    assert not library.entries()


def test_corrupt_registry_preserves_bytes_and_still_allows_browse(main, library, qtbot, monkeypatch):
    registry = library.root/'studies.json'; registry.write_text('{corrupt')
    launcher = IncubationStudyLauncher(main, library=library); qtbot.addWidget(launcher)
    assert 'has not been changed' in launcher._library_note.text()
    assert launcher._browse.isEnabled()
    calls = []
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', lambda *args: calls.append(args[2]) or '')
    launcher._choose_open()
    assert calls == [str(library.root)]
    assert registry.read_text() == '{corrupt'


def test_inaccessible_library_root_keeps_new_and_browse_available(main, tmp_path, qtbot, monkeypatch):
    def inaccessible(): raise StudyLibraryError('Cannot access the library root')
    monkeypatch.setattr(launcher_module, 'StudyLibrary', inaccessible)
    monkeypatch.setattr(launcher_module, 'default_library_root', lambda: tmp_path/'unavailable')
    launcher = IncubationStudyLauncher(main); qtbot.addWidget(launcher)
    assert launcher._browse.isEnabled()
    assert 'Cannot access' in launcher._library_note.text()
    window = launcher.create_study(tmp_path/'new')
    assert launcher._windows == [window]
    assert checkpoint.load_workspace(window.folder)['name'] == 'Incubation study'
    assert 'Could not add' in launcher._status.text()
