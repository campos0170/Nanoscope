"""Malformed project fields keep the documented error family."""
import json
import zipfile

import pandas as pd
import pytest

from nanoscope.core.project import (
    MANIFEST_NAME, DocumentRef, FitRec, ProjectError, ProjectState, ResultRec,
    load_project, save_project,
)
from nanoscope.core.identity import fingerprint_source


def rewrite(path, change, corrupt=None):
    with zipfile.ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(parts[MANIFEST_NAME])
    change(manifest)
    parts[MANIFEST_NAME] = json.dumps(manifest).encode()
    if corrupt:
        parts.update(corrupt)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)


@pytest.mark.parametrize('value', [[], {}, 42, True, ''])
def test_bad_measurement_member_is_project_error(tmp_path, value):
    path = save_project(tmp_path / 'bad.nanoscope', ProjectState())
    rewrite(path, lambda m: m.update(measurements=value))
    with pytest.raises(ProjectError, match='measurements'):
        load_project(path)


def test_bad_measurements_parquet_is_project_error(tmp_path):
    path = save_project(tmp_path / 'bad.nanoscope', ProjectState())
    rewrite(path, lambda m: m.update(measurements='bad.parquet'), {'bad.parquet': b'not parquet'})
    with pytest.raises(ProjectError, match='damaged|unreadable'):
        load_project(path)


def test_infinite_integer_field_is_project_error(tmp_path):
    path = save_project(tmp_path / 'bad.nanoscope', ProjectState())
    rewrite(path, lambda m: m.update(documents=[{'path': 'source.csv', 'loaded_range': [0, float('inf')]}]))
    with pytest.raises(ProjectError, match='documents'):
        load_project(path)


def test_result_and_fit_keep_their_own_source_identity(tmp_path):
    source = tmp_path / 'source.dat'
    source.write_bytes(b'original recording')
    original = fingerprint_source(source, 100)
    source.write_bytes(b'changed recording')
    current = fingerprint_source(source, 200)
    state = ProjectState(documents=[DocumentRef(str(source), source=original)], results=[
        ResultRec(str(source), 0, 'detect.schmitt', pd.DataFrame({'event_id': [0]}),
                  source_fingerprint=current)], fits=[
        FitRec(str(source), 0, 'fit.levels', table=pd.DataFrame({'level': [1]}),
               source_fingerprint=original)])
    saved = load_project(save_project(tmp_path / 'identity.nanoscope', state))
    assert saved.results[0].source_fingerprint == current
    assert saved.fits[0].source_fingerprint == original
