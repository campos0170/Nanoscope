"""Profiles name new defaults; saved workspaces retain their own scientific state."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.gui.incubation_profiles import (
    ACID_SCHEMA, ACID_WORKSPACE_KIND, MANUAL_WORKSPACE_KIND, PROFILES,
    get_profile, initialize_study_metadata, peek_workspace, recording_defaults, saved_profile,
)


def manual():
    return dict(name='Manual study', model_path='', records=[], comparison=None,
        features=None, clustering=None, mixture=None, figures=[], history=[],
        comparison_settings={'seed': 91}, cluster_settings={'n_clusters': 5},
        mixture_settings={'min_events': 42})


def acid():
    return dict(schema=ACID_SCHEMA, workspace_kind=ACID_WORKSPACE_KIND,
        name='Existing Acid study', records=[], settings={'SEED': 17, 'FAMILY_REFERENCE_TAG': 'reference_custom'},
        figures=[], history=[], analysis=None)


def save(state, folder):
    checkpoint.save_workspace(state, folder, expected_revision=None)
    return folder


def rewrite_manifest(folder, mutate):
    path = folder/'workspace.json'
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


def test_explicit_profiles_have_general_routes_and_only_established_edge_defaults():
    assert [(p.id, p.engine) for p in PROFILES] == [('general', 'manual'), ('zn', 'manual'), ('cu', 'manual'), ('acid', 'acid')]
    assert recording_defaults('general') == recording_defaults('zn') == {'median_engine': 'reflect'}
    assert recording_defaults('cu') == {'median_engine': 'shrinking'}
    assert recording_defaults('acid') == {}
    changed = recording_defaults('cu'); changed['median_engine'] = 'reflect'
    assert recording_defaults('cu')['median_engine'] == 'shrinking'
    with pytest.raises(FrozenInstanceError): get_profile('cu').label = 'Changed'
    for value in ('cu_30h.abf', 'anything_future', None, 1):
        with pytest.raises(ValueError): get_profile(value)


@pytest.mark.parametrize('profile', ['general', 'zn', 'cu', 'acid'])
def test_new_profile_metadata_preserves_engine_settings_and_roundtrips(profile, tmp_path):
    state = acid() if profile == 'acid' else manual()
    before = deepcopy(state)
    updated = initialize_study_metadata(state, profile, name='Custom time course', sample='Protein X', condition_label='pH 6.1')
    assert state == before
    assert updated['name'] == 'Custom time course' and updated['sample'] == 'Protein X'
    assert updated['condition_label'] == 'pH 6.1' and updated['study_profile'] == profile
    assert {k: v for k, v in updated.items() if k.endswith('settings')} == {k: v for k, v in before.items() if k.endswith('settings')}
    folder = save(updated, tmp_path/profile)
    route = peek_workspace(folder)
    assert route.profile_id == profile and route.engine == get_profile(profile).engine
    assert route.name == 'Custom time course' and route.path == folder.resolve()
    restored = checkpoint.load_workspace(folder)
    assert restored == updated and saved_profile(restored) == profile


def test_default_names_and_conditions_are_explicit_and_do_not_assume_sample():
    assert initialize_study_metadata(manual())['sample'] == ''
    assert initialize_study_metadata(manual(), 'cu')['condition_label'] == 'Cu'
    assert initialize_study_metadata(manual(), 'zn')['name'] == 'Zn incubation study'
    assert initialize_study_metadata(manual(), 'zn', condition_label='')['condition_label'] == ''
    assert initialize_study_metadata(acid(), 'acid')['sample'] == ''


@pytest.mark.parametrize('change', [
    {'records': [{'tag': 'cu_30h'}]}, {'history': [{'stage': 'prepare'}]},
    {'features': []}, {'analysis': {}}, {'figures': ['result.png']}, {'study_profile': 'general'},
])
def test_profile_initialization_never_changes_existing_scientific_state(change):
    state = manual(); state.update(change); before = deepcopy(state)
    with pytest.raises(ValueError, match='new empty study'):
        initialize_study_metadata(state, 'cu')
    assert state == before


def test_engine_switch_and_unknown_state_rejected():
    with pytest.raises(ValueError, match='Acid profile'):
        initialize_study_metadata(manual(), 'acid')
    with pytest.raises(ValueError, match='general workspace'):
        initialize_study_metadata(acid(), 'general')
    with pytest.raises(ValueError, match='not a recognized'):
        initialize_study_metadata({}, 'general')


def test_old_markerless_manual_is_general_even_with_acid_or_cu_filename(tmp_path):
    state = manual()
    state['name'] = 'Zn study'
    state['records'] = [{'tag': 'acid_30h', 'source_path': tmp_path/'cu_signal.abf', 'median_engine': 'shrinking'}]
    folder = save(state, tmp_path/'acid-incubation')
    before = (folder/'workspace.json').read_bytes()
    route = peek_workspace(folder)
    assert (route.engine, route.profile_id) == ('manual', 'general')
    assert saved_profile(state) == 'general'
    assert (folder/'workspace.json').read_bytes() == before
    assert 'study_profile' not in state


def test_legacy_acid_schema_routes_without_new_profile_or_workspace_kind(tmp_path):
    state = acid(); del state['workspace_kind']
    folder = save(state, tmp_path/'generic-folder-name')
    assert peek_workspace(folder).profile_id == 'acid'
    assert peek_workspace(folder/'workspace.json').engine == 'acid'


def test_peeking_routes_missing_scientific_payload_without_loading_or_trusting_it(tmp_path, monkeypatch):
    state = initialize_study_metadata(manual(), 'cu')
    state['records'] = [{'tag': 'sample', 'prepared': np.arange(8, dtype=np.float32)}]
    folder = save(state, tmp_path/'workspace')
    manifest = json.loads((folder/'workspace.json').read_text())
    payload = folder/next(iter(manifest['array_sha256']))
    payload.unlink()
    def refused(*args, **kwargs): pytest.fail('Routing tried to load a scientific payload')
    monkeypatch.setattr(checkpoint, 'load_workspace', refused)
    monkeypatch.setattr(np, 'load', refused)
    assert peek_workspace(folder).profile_id == 'cu'
    # The routing decision makes no integrity claim; actual engine open must
    # still use the normal verified loader and report the missing payload.


@pytest.mark.parametrize('changes', [
    {'workspace_kind': 'future_workspace_v17'}, {'schema': 'future-science-v17'},
    {'study_profile': 'future'}, {'study_profile': 'acid'}, {'workspace_kind': ACID_WORKSPACE_KIND},
])
def test_unknown_or_conflicting_saved_metadata_is_not_silently_generalized(tmp_path, changes):
    state = manual(); state.update(changes)
    folder = save(state, tmp_path/'workspace')
    with pytest.raises(ValueError): peek_workspace(folder)


def test_conflicting_acid_profile_is_rejected(tmp_path):
    state = acid(); state['study_profile'] = 'zn'
    with pytest.raises(ValueError, match='conflicts'):
        peek_workspace(save(state, tmp_path/'workspace'))


def test_arbitrary_codec_checkpoints_are_not_manual_studies(tmp_path):
    folder = save({'name': 'Analysis context', 'records': [], 'figures': [], 'history': []}, tmp_path/'workspace')
    with pytest.raises(ValueError, match='not a recognized'):
        peek_workspace(folder)


@pytest.mark.parametrize('mutation', [
    lambda m: m.update(format='other'), lambda m: m.update(version=2),
    lambda m: m.update(version=True), lambda m: m.update(state={'type': 'array', 'path': '/private/other.npy'}),
    lambda m: m['state']['items'].append(['name', 'Duplicate name']),
    lambda m: m['state']['items'].append(['study_profile', {'type': 'array', 'path': 'arrays/unknown.npy'}]),
])
def test_invalid_manifest_envelopes_and_nonscalar_route_fields_refused(tmp_path, mutation):
    folder = save(manual(), tmp_path/'workspace')
    rewrite_manifest(folder, mutation)
    with pytest.raises(ValueError): peek_workspace(folder)


def test_missing_and_reference_bundles_require_their_own_opener(tmp_path):
    for folder in (tmp_path/'missing', tmp_path/'reference'):
        folder.mkdir()
        if folder.name == 'reference':
            (folder/'metadata.json').write_text(json.dumps({'format': 'nanoscope-incubation-result'}))
        with pytest.raises(ValueError, match='workspace.json'):
            peek_workspace(folder)
