"""Persistent study indexing never mutates or reloads scientific checkpoints."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.gui import incubation_library as library_module
from nanoscope.gui.incubation_library import StudyLibrary, StudyLibraryError
from nanoscope.gui.incubation_profiles import ACID_SCHEMA, initialize_study_metadata


def save_study(folder, *, profile='general', name='Manual study', payload=False):
    if profile == 'acid':
        state = dict(schema=ACID_SCHEMA, name=name, records=[], settings={'seed': 17},
                     figures=[], history=[], analysis=None)
    else:
        state = dict(name=name, model_path='', records=[], comparison=None,
                     features=None, clustering=None, mixture=None, figures=[], history=[],
                     comparison_settings={'seed': 91}, cluster_settings={'n_clusters': 5},
                     mixture_settings={'min_events': 42})
    state = initialize_study_metadata(state, profile, name=name, sample='Saved sample')
    if payload:
        state['records'] = [{'tag': 'sample', 'signal': np.arange(16, dtype=np.float32)}]
    checkpoint.save_workspace(state, folder, expected_revision=None)
    return folder


def test_registry_persists_across_instances_and_preserves_alias_and_study_bytes(tmp_path):
    root = tmp_path/'library'
    folder = save_study(tmp_path/'elsewhere'/'study', profile='cu')
    before = (folder/'workspace.json').read_bytes()
    first = StudyLibrary(root)
    assert not (root/'studies.json').exists()
    remembered = first.remember(folder, label='Copper preparation A')
    assert remembered.label == 'Copper preparation A' and remembered.profile_id == 'cu'
    second = StudyLibrary(root)
    assert second.entries() == [remembered]
    assert second.remember(folder).label == 'Copper preparation A'
    assert len(json.loads((root/'studies.json').read_text())['entries']) == 1
    assert second.picker_directory() == str(folder.resolve())
    assert (folder/'workspace.json').read_bytes() == before


def test_discovery_is_immediate_metadata_only_and_never_infers_chemistry(tmp_path, monkeypatch):
    root = tmp_path/'library'
    general = save_study(root/'acid_zn_cu_file_name', payload=True)
    acid = save_study(root/'ordinary-name', profile='acid', name='Saved Acid name')
    save_study(root/'container'/'nested-study', profile='zn')
    (root/'unrelated').mkdir()
    manifest = json.loads((general/'workspace.json').read_text())
    for name in manifest['array_sha256']:
        (general/name).unlink()
    def refuse(*args, **kwargs):
        pytest.fail('Library indexing tried to load scientific payloads')
    monkeypatch.setattr(checkpoint, 'load_workspace', refuse)
    monkeypatch.setattr(np, 'load', refuse)
    library = StudyLibrary(root)
    entries = {x.path: x for x in library.entries()}
    assert set(entries) == {general.resolve(), acid.resolve()}
    assert entries[general.resolve()].profile_id == 'general'
    assert entries[acid.resolve()].label == 'Saved Acid name'
    assert all(x.available for x in entries.values())
    assert not (root/'studies.json').exists()
    assert library.picker_directory() == str(root.resolve())


def test_copy_is_distinct_but_symlink_and_registry_alias_are_deduplicated(tmp_path):
    root = tmp_path/'library'
    original = save_study(tmp_path/'original', profile='zn')
    root.mkdir()
    copied = root/'independent-copy'
    shutil.copytree(original, copied)
    alias = root/'friendly-link'
    alias.symlink_to(original, target_is_directory=True)
    library = StudyLibrary(root)
    library.remember(alias, label='Reviewed zinc study')
    library.remember(original)
    entries = library.entries()
    assert len(entries) == 2
    by_path = {x.path: x for x in entries}
    assert by_path[original.resolve()].label == 'Reviewed zinc study'
    assert by_path[copied.resolve()].label == 'Manual study'
    assert len(json.loads((root/'studies.json').read_text())['entries']) == 1


def test_stale_registered_path_remains_visible_and_picker_skips_it(tmp_path, monkeypatch):
    library = StudyLibrary(tmp_path/'library')
    first = save_study(tmp_path/'one', profile='cu')
    second = save_study(tmp_path/'two', profile='zn')
    clock = iter([10, 20])
    monkeypatch.setattr(library_module.time, 'time', lambda: next(clock))
    library.remember(first, label='Available')
    library.remember(second, label='Moved zinc')
    second.rename(tmp_path/'moved')
    missing, available = library.entries()
    assert missing.path == second and missing.label == 'Moved zinc'
    assert missing.profile_id == 'zn' and not missing.available and 'missing' in missing.error
    assert available.available
    assert library.picker_directory() == str(first)
    first.rename(tmp_path/'moved-too')
    assert library.picker_directory() == str(library.root)


def test_corrupt_and_unsupported_manifests_are_individual_unavailable_entries(tmp_path):
    library = StudyLibrary(tmp_path/'library')
    good = save_study(library.root/'good', profile='cu')
    bad = save_study(library.root/'bad', profile='zn')
    future = save_study(library.root/'future', profile='acid')
    library.remember(bad, label='Keep this label')
    (bad/'workspace.json').write_text('{broken')
    data = json.loads((future/'workspace.json').read_text())
    data['version'] = 100
    (future/'workspace.json').write_text(json.dumps(data))
    entries = {x.path: x for x in library.entries()}
    assert len(entries) == 3 and entries[good].available
    assert not entries[bad].available and entries[bad].error
    assert entries[bad].label == 'Keep this label'
    assert not entries[future].available and entries[future].error


@pytest.mark.parametrize('bad_bytes', [
    b'{broken', b'{"version":2,"entries":[]}', b'{"version":true,"entries":[]}',
    b'{"version":1,"version":1,"entries":[]}',
    b'{"version":1,"entries":[{"path":"relative","label":"Study"}]}',
    b'{"version":1,"entries":[{"path":"/tmp/a","label":"Study","last_opened":NaN}]}',
    b'\xff\xfe',
])
def test_damaged_registry_never_silently_discarded_or_rewritten(tmp_path, bad_bytes):
    library = StudyLibrary(tmp_path/'library')
    good = save_study(tmp_path/'valid-study')
    registry = library.root/'studies.json'
    registry.write_bytes(bad_bytes)
    for action in (library.entries, library.picker_directory, lambda: library.remember(good)):
        with pytest.raises(StudyLibraryError, match='Restore this file from a backup'):
            action()
        assert registry.read_bytes() == bad_bytes
    assert not (library.root/'studies.json.lock').exists()


def test_invalid_study_or_label_does_not_create_registry(tmp_path):
    library = StudyLibrary(tmp_path/'library')
    good = save_study(tmp_path/'valid-study')
    for folder, label in ((tmp_path/'missing', None), (good, ''), (good, 19)):
        with pytest.raises(StudyLibraryError):
            library.remember(folder, label=label)
    assert not (library.root/'studies.json').exists()


def test_atomic_replace_failure_retains_registry_and_removes_temporary_file(tmp_path, monkeypatch):
    library = StudyLibrary(tmp_path/'library')
    one = save_study(tmp_path/'one')
    two = save_study(tmp_path/'two')
    library.remember(one)
    before = (library.root/'studies.json').read_bytes()
    def fail_replace(*args):
        raise OSError('simulated disk failure')
    monkeypatch.setattr(library_module.os, 'replace', fail_replace)
    with pytest.raises(StudyLibraryError, match='simulated disk failure'):
        library.remember(two)
    assert (library.root/'studies.json').read_bytes() == before
    assert not list(library.root.glob('.studies-*'))
    assert not (library.root/'studies.json.lock').exists()


def test_two_instances_merge_latest_registry_instead_of_overwriting(tmp_path):
    first = StudyLibrary(tmp_path/'library')
    second = StudyLibrary(first.root)
    one = save_study(tmp_path/'one')
    two = save_study(tmp_path/'two')
    first.entries()
    second.entries()
    first.remember(one, label='First alias')
    second.remember(two, label='Second alias')
    first.remember(one)
    assert {x.label for x in second.entries()} == {'First alias', 'Second alias'}


def test_concurrent_processes_merge_under_registry_lock(tmp_path):
    root = tmp_path/'library'
    one = save_study(tmp_path/'one')
    two = save_study(tmp_path/'two')
    source = Path(__file__).resolve().parents[2]
    code = ('from nanoscope.gui.incubation_library import StudyLibrary; import sys; '
            'StudyLibrary(sys.argv[1]).remember(sys.argv[2], label=sys.argv[3])')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONPATH=str(source))
    processes = [subprocess.Popen([sys.executable, '-c', code, str(root), str(folder), label],
                 cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for folder, label in ((one, 'One'), (two, 'Two'))]
    for process in processes:
        out, error = process.communicate(timeout=30)
        assert process.returncode == 0, out+error
    assert {x.label for x in StudyLibrary(root).entries()} == {'One', 'Two'}


def test_lock_conflict_preserves_existing_registry(tmp_path, monkeypatch):
    library = StudyLibrary(tmp_path/'library')
    folder = save_study(tmp_path/'study')
    library.remember(folder)
    before = (library.root/'studies.json').read_bytes()
    class BusyLock:
        def __init__(self, *args): pass
        def tryLock(self, timeout): return False
    monkeypatch.setattr(library_module.QtCore, 'QLockFile', BusyLock)
    with pytest.raises(StudyLibraryError, match='busy or cannot be locked'):
        library.remember(folder, label='Not saved')
    assert (library.root/'studies.json').read_bytes() == before


def test_registry_and_study_read_methods_are_nonmutating(tmp_path):
    library = StudyLibrary(tmp_path/'library')
    folder = save_study(tmp_path/'study')
    library.remember(folder)
    registry = library.root/'studies.json'
    before = registry.read_bytes(), registry.stat().st_mtime_ns
    study_before = (folder/'workspace.json').read_bytes()
    library.entries(); library.picker_directory()
    assert (registry.read_bytes(), registry.stat().st_mtime_ns) == before
    assert (folder/'workspace.json').read_bytes() == study_before


def test_defaults_and_override_are_isolated_from_real_user_library(tmp_path, monkeypatch):
    monkeypatch.setenv('NANOSCOPE_INCUBATION_LIBRARY_ROOT', str(tmp_path/'environment'))
    assert StudyLibrary().root == tmp_path/'environment'
    assert StudyLibrary(tmp_path/'explicit').root == tmp_path/'explicit'
    monkeypatch.delenv('NANOSCOPE_INCUBATION_LIBRARY_ROOT')
    monkeypatch.setattr(library_module.QtCore.QStandardPaths, 'writableLocation', lambda *_: str(tmp_path/'Documents'))
    assert StudyLibrary().root == tmp_path/'Documents'/'NanoScope'/'Incubation Studies'
    monkeypatch.setattr(library_module.QtCore.QStandardPaths, 'writableLocation', lambda *_: '')
    monkeypatch.setattr(library_module.Path, 'home', lambda: tmp_path/'home')
    assert StudyLibrary().root == tmp_path/'home'/'Documents'/'NanoScope'/'Incubation Studies'


def test_registry_additional_fields_survive_a_successful_update(tmp_path):
    library = StudyLibrary(tmp_path/'library')
    folder = save_study(tmp_path/'study')
    library.remember(folder)
    path = library.root/'studies.json'
    data = json.loads(path.read_text())
    data['note'] = 'User-owned index metadata'
    data['entries'][0]['note'] = 'Keep this too'
    path.write_text(json.dumps(data))
    library.remember(folder, label='Updated label')
    saved = json.loads(path.read_text())
    assert saved['note'] == data['note']
    assert saved['entries'][0]['note'] == data['entries'][0]['note']


def test_default_root_helper_does_not_create_directories(tmp_path, monkeypatch):
    intended = tmp_path/'not-created'/'library'
    monkeypatch.setenv('NANOSCOPE_INCUBATION_LIBRARY_ROOT', str(intended))
    assert library_module.default_library_root() == intended
    assert not intended.exists()


def test_unusable_library_root_is_an_actionable_error(tmp_path):
    root = tmp_path/'file-not-directory'
    root.write_text('Retain this file')
    with pytest.raises(StudyLibraryError, match='Cannot access'):
        StudyLibrary(root)
    assert root.read_text() == 'Retain this file'


def test_broken_registry_link_is_not_treated_as_empty_registry(tmp_path):
    library = StudyLibrary(tmp_path/'library')
    study = save_study(tmp_path/'study')
    registry = library.root/'studies.json'
    target = tmp_path/'missing-registry.json'
    registry.symlink_to(target)
    with pytest.raises(StudyLibraryError, match='missing target'):
        library.remember(study)
    assert registry.is_symlink() and registry.readlink() == target
    assert not target.exists()


def test_broken_manifest_link_is_visible_as_an_unavailable_discovered_study(tmp_path):
    library = StudyLibrary(tmp_path/'library')
    folder = library.root/'workspace'
    folder.mkdir()
    (folder/'workspace.json').symlink_to(tmp_path/'missing-manifest.json')
    entry, = library.entries()
    assert entry.path == folder and not entry.available and entry.error


def test_identical_remember_with_unchanged_recency_does_not_rewrite(tmp_path, monkeypatch):
    library = StudyLibrary(tmp_path/'library')
    study = save_study(tmp_path/'study')
    monkeypatch.setattr(library_module.time, 'time', lambda: 100)
    library.remember(study, label='Alias')
    def refuse(*args): pytest.fail('Unchanged index was needlessly rewritten')
    monkeypatch.setattr(library, '_write_registry', refuse)
    assert library.remember(study).label == 'Alias'
