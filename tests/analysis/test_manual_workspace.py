"""Checkpoint fidelity, portable outputs and interrupted-save protection."""
from dataclasses import fields, is_dataclass
import json
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import manual_workspace as workspace, subpop_gmm as sg
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.analysis.incubation_detection import PreparedRecording, RecordingAnalysis
from nanoscope.analysis.manual_preparation import RawPreparedRecording
from nanoscope.analysis.manual_detection import InferenceResult, MeasurementSettings
from nanoscope.analysis.manual_comparisons import ComparisonResult, MixtureSettings


def equivalent(before, after):
    if is_dataclass(before):
        assert type(before) is type(after)
        for field in fields(before): equivalent(getattr(before,field.name),getattr(after,field.name))
    elif isinstance(before,pd.DataFrame):
        pd.testing.assert_frame_equal(before,after)
        equivalent(before.attrs,after.attrs)
    elif isinstance(before,pd.Series):
        pd.testing.assert_series_equal(before,after)
    elif isinstance(before,np.ndarray):
        assert before.dtype==after.dtype and before.shape==after.shape
        if before.dtype.hasobject:
            for a,b in zip(before.flat,after.flat): equivalent(a,b)
        else: np.testing.assert_array_equal(before,after)
    elif isinstance(before,dict):
        assert before.keys()==after.keys()
        for key in before: equivalent(before[key],after[key])
    elif isinstance(before,(list,tuple)):
        assert type(before) is type(after) and len(before)==len(after)
        for a,b in zip(before,after): equivalent(a,b)
    elif before is pd.NA or before is pd.NaT: assert before is after
    elif isinstance(before,float) and np.isnan(before): assert np.isnan(after)
    else: assert before==after


def quantification():
    selection=sg.PCSelection(2,2,-3.4,np.array([[-3.6,-3.4],[np.nan,-3.8]]),(2,4),(1,2),5,0,2,('grid warning',))
    fit=sg.SubpopFit(labels=pd.Series([0,1],index=pd.Index([13,15],name='event')),
        posteriors=np.array([[.8,.2],[.1,.9]]),confidence=np.array([.8,.9]),conf_frac=.5,conf_median=.85,
        scores=np.array([[1.,2.],[3.,4.]]),means=np.array([[1.,2.],[3.,4.]]),weights=np.array([.4,.6]),
        sizes={0:1,1:1},n_pcs=2,k=2,order=np.array([1,0]),explained_variance_ratio=np.array([.7,.2]),
        bic={1:4.2,2:3.1},bic_k=2,warnings=('fit warning',),meta={'seed':np.int64(0)})
    ci=sg.FractionCI(np.array([.1,.2]),np.array([.8,.9]),np.array([[.4,.6],[np.nan,np.nan]]),1,2,0,('lost fit',))
    separation=sg.Separation(.1,.3,.4,{0:(1,.1,.2),1:(1,.2,.3)},2,.3,2,2,2,False,{2:.3,3:.2})
    return sg.SubpopQuantification(selection,fit,ci,pd.DataFrame({'subpop':[0,1],'n':[1,1]}),
        pd.DataFrame([[1,0],[0,1]],index=pd.Index([0,1],name='subpop'),columns=pd.Index([-1,0],name='hdbscan')),
        ('log_dwell','depth_sust_pA'),2,separation,('calibration warning',))


def test_all_scientific_result_dataclasses_and_candidate_provenance_round_trip(tmp_path):
    values=np.arange(10,dtype=np.float32)
    catalog=pd.DataFrame({'start_idx':np.array([1,4],np.int64),'end_idx':np.array([2,6],np.int64),
                          'accepted':[True,True],'passes_gate':[True,True]})
    catalog.attrs={'manual_detection':{'candidate_geometry_sha256':'geometry','index_convention':'inclusive',
        'decode_settings':{'enter':.4,'smooth':3},'nested':{1:('a',np.array([3,4],dtype=np.int32))}}}
    basic=dict(tag='test',raw=values.astype(float),detrended=values,fs_hz=10.,duration_s=1.,
               metadata={'raw_path':'/external/missing.abf'},provenance={'recording_identity':{'sha256':'abc'}})
    prepared=PreparedRecording(**basic)
    raw=RawPreparedRecording(**basic,filtered=values+.2,baseline=values*.1)
    analysis=RecordingAnalysis('test',prepared.raw,values,10.,1.,catalog,{'measurement_settings':{'min_depth_sigma':4.}})
    state={'prepared':prepared,'raw':raw,'analysis':analysis,'candidates':catalog,
           'inference':InferenceResult(values,{'uncovered_tail_samples':2},{'probability_sha256':'hash'}),
           'comparison':ComparisonResult(catalog,pd.DataFrame({'n':[2]}),{1:catalog},{'methods':('a','b')}),
           'mixture':quantification(),'settings':(MeasurementSettings(),MixtureSettings())}
    workspace.save_workspace(state,tmp_path)
    restored=workspace.load_workspace(tmp_path)
    equivalent(state,restored)
    restored['candidates'].loc[0,'accepted']=False
    assert restored['candidates'].loc[0,'passes_gate']
    assert restored['candidates'].attrs['manual_detection']['candidate_geometry_sha256']=='geometry'


def test_ndarray_shapes_dtypes_structures_and_object_tuples(tmp_path):
    objects=np.empty((2,),dtype=object); objects[0]=(1,2); objects[1]={'label':'x','missing':pd.NA}
    scalar_object=np.empty((),dtype=object); scalar_object[()]=(1,2)
    state=[np.asarray(3.5,dtype=np.float32),np.empty((0,2),dtype=np.int16),np.ones((3,1,2),dtype='>f8'),
           np.arange(20).reshape(4,5)[:,::2],np.asfortranarray(np.ones((3,4))),objects,scalar_object,
           np.array([(1,2.)],dtype=[('a','<i4'),('b','<f8')]),np.array([(1,2.)],dtype=[('x','<i4'),('y','<f8')]),
           np.datetime64('2026-09-14','ns'),np.array(['2026-09-14'],dtype='datetime64[ns]')]
    workspace.save_workspace(state,tmp_path)
    equivalent(state,workspace.load_workspace(tmp_path))


def test_dataframe_extension_dtypes_categories_duplicate_labels_and_indexes(tmp_path):
    frame=pd.DataFrame({'integer':pd.Series([1,pd.NA,3],dtype='Int64'),
        'boolean':pd.Series([True,pd.NA,False],dtype='boolean'),
        'text':pd.Series(['a',pd.NA,'c'],dtype='string'),
        'category':pd.Categorical(['x','y','x'],categories=['x','y','unused'],ordered=True),
        'time':pd.date_range('2026-09-14',periods=3,tz='America/New_York')})
    frame.index=pd.MultiIndex.from_arrays([['first','first','second'],[1,2,1]],names=['source','row'])
    duplicate=pd.concat([pd.Series([1,2],dtype='int16'),pd.Series([3.5,4.5],dtype='float32')],axis=1)
    duplicate.columns=pd.Index(['same','same'],name='measure')
    dated=pd.DataFrame({'x':[1,2]},index=pd.date_range('2026-09-14',periods=2,freq='D',name='day'))
    named=pd.DataFrame(index=pd.RangeIndex(3,9,2,name='event'))
    state=[frame,duplicate,dated,named,pd.Series([],dtype='Float32',name=('named',1))]
    workspace.save_workspace(state,tmp_path)
    equivalent(state,workspace.load_workspace(tmp_path))


def test_tampered_array_rejected_on_load_and_repaired_from_memory_on_save(tmp_path):
    state={'raw':np.arange(31,dtype=np.float32)}
    workspace.save_workspace(state,tmp_path)
    manifest=json.loads((tmp_path/'workspace.json').read_text())
    relative=next(iter(manifest['array_sha256'])); path=tmp_path/relative
    expected=path.read_bytes(); path.write_bytes(b'corrupt file')
    with pytest.raises(ValueError,match='changed'): workspace.load_workspace(tmp_path)
    workspace.save_workspace(state,tmp_path)
    assert path.read_bytes()==expected
    equivalent(state,workspace.load_workspace(tmp_path))
    np.save(path,state['raw']+1,allow_pickle=False)
    workspace.save_workspace(state,tmp_path)
    assert path.read_bytes()==expected


def test_owned_outputs_travel_and_are_verified_external_inputs_are_not_read(tmp_path):
    original=tmp_path/'original'; workspace.save_workspace({},original)
    run=original/'runs'/'mixture'; run.mkdir(parents=True)
    (run/'table.csv').write_text('n\n2\n'); (run/'figure.png').write_bytes(b'illustration')
    state={'result_folder':run,'figure':run/'figure.png','source':tmp_path/'missing-source.abf'}
    workspace.save_workspace(state,original,owned_outputs=workspace.stage_output_files(run))
    manifest=json.loads((original/'workspace.json').read_text())
    assert set(manifest['owned_file_sha256'])=={'runs/mixture/table.csv','runs/mixture/figure.png'}
    moved=tmp_path/'moved'; shutil.copytree(original,moved); shutil.rmtree(original)
    restored=workspace.load_workspace(moved)
    assert restored['result_folder']==moved/'runs'/'mixture'
    assert restored['figure'].read_bytes()==b'illustration'
    assert restored['source']==state['source'] and not restored['source'].exists()
    (moved/'runs'/'mixture'/'table.csv').write_text('n\n999\n')
    before=(moved/'workspace.json').read_bytes()
    with pytest.raises(ValueError,match='changed'): workspace.load_workspace(moved)
    with pytest.raises(ValueError,match='changed'): workspace.save_workspace(restored,moved)
    assert (moved/'workspace.json').read_bytes()==before


def test_interrupted_manifest_publish_keeps_previous_checkpoint(tmp_path,monkeypatch):
    previous={'values':np.arange(5),'step':1}; workspace.save_workspace(previous,tmp_path)
    before=(tmp_path/'workspace.json').read_bytes(); replace=workspace.os.replace
    def fail_manifest(source,destination):
        if Path(destination).name=='workspace.json': raise OSError('simulated interruption')
        return replace(source,destination)
    monkeypatch.setattr(workspace.os,'replace',fail_manifest)
    with pytest.raises(OSError,match='interruption'):
        workspace.save_workspace({'values':np.arange(13),'step':2},tmp_path)
    assert (tmp_path/'workspace.json').read_bytes()==before
    equivalent(previous,workspace.load_workspace(tmp_path))
    assert not list(tmp_path.glob('.workspace-*')) and not list((tmp_path/'arrays').glob('.array-*'))


def test_cancellation_and_unverified_or_escaping_paths_are_refused(tmp_path):
    workspace.save_workspace({'values':np.arange(3)},tmp_path)
    before=(tmp_path/'workspace.json').read_bytes(); cancel=threading.Event(); cancel.set()
    with pytest.raises(OperationCancelled): workspace.save_workspace({'x':2},tmp_path,cancel=cancel)
    assert before==(tmp_path/'workspace.json').read_bytes()
    manifest=json.loads(before); manifest['state']={'type':'array','path':'arrays/unverified.npy'}
    (tmp_path/'workspace.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Unverified'): workspace.load_workspace(tmp_path)
    manifest['state']={'type':'workspace_path','value':'../../elsewhere'}
    (tmp_path/'workspace.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='escapes'): workspace.load_workspace(tmp_path)


def test_repeated_temporary_dataframe_buffers_do_not_reuse_object_ids(tmp_path):
    state=[pd.DataFrame({f'column_{i}':np.arange(4,dtype=np.float64)+i*10 for i in range(25)}) for _ in range(12)]
    workspace.save_workspace(state,tmp_path)
    equivalent(state,workspace.load_workspace(tmp_path))


def test_only_registered_outputs_are_owned_and_later_user_files_stay_unowned(tmp_path):
    workspace.save_workspace({}, tmp_path)
    run = tmp_path / 'runs' / 'complete'; run.mkdir(parents=True)
    output = run / 'table.csv'; output.write_text('n\n2\n')
    (tmp_path / 'notes.txt').write_text('user notes')
    (tmp_path / '.DS_Store').write_bytes(b'Finder A')
    (run / '.DS_Store').write_bytes(b'Finder A')
    (run / '._table.csv').write_bytes(b'AppleDouble A')
    state = {'folder': run}
    workspace.save_workspace(state, tmp_path, owned_outputs=workspace.stage_output_files(run))
    manifest = json.loads((tmp_path / 'workspace.json').read_text())
    assert set(manifest['owned_file_sha256']) == {'runs/complete/table.csv'}
    (run / 'my_annotations.txt').write_text('another unrelated file')
    (tmp_path / 'notes.txt').unlink()
    (tmp_path / '.DS_Store').write_bytes(b'Finder B')
    (run / '.DS_Store').unlink()
    (run / '._table.csv').write_bytes(b'AppleDouble B')
    workspace.save_workspace(state, tmp_path)
    (run / 'my_annotations.txt').write_text('edited notes')
    equivalent(state, workspace.load_workspace(tmp_path))
    assert set(json.loads((tmp_path / 'workspace.json').read_text())['owned_file_sha256']) == {'runs/complete/table.csv'}
    output.write_text('n\n999\n')
    with pytest.raises(ValueError, match='changed'):
        workspace.save_workspace(state, tmp_path)


def test_legacy_metadata_inventory_recovers_without_ignoring_scientific_tamper(tmp_path):
    workspace.save_workspace({}, tmp_path)
    run = tmp_path / 'runs' / 'complete'; run.mkdir(parents=True)
    table = run / 'table.csv'; table.write_text('n\n2\n')
    workspace.save_workspace({'table': table}, tmp_path)
    manifest_path = tmp_path / 'workspace.json'
    legacy = json.loads(manifest_path.read_text())
    for relative in ('.DS_Store', 'runs/complete/.DS_Store', 'runs/complete/._table.csv'):
        legacy['owned_file_sha256'][relative] = hashlib.sha256(b'outdated metadata').hexdigest()
    # This unknown legacy file remains protected: migration cannot guess whether
    # a user's old entry was a scientific result merely from its extension.
    unknown = tmp_path / 'old_notes.txt'; unknown.write_bytes(b'legacy')
    legacy['owned_file_sha256']['old_notes.txt'] = hashlib.sha256(b'legacy').hexdigest()
    manifest_path.write_text(json.dumps(legacy))
    (tmp_path / '.DS_Store').write_bytes(b'changed metadata')
    state, revision = workspace.load_workspace(tmp_path, with_revision=True)
    assert state['table'] == table
    workspace.save_workspace(state, tmp_path, expected_revision=revision)
    migrated = json.loads(manifest_path.read_text())
    assert set(migrated['owned_file_sha256']) == {'runs/complete/table.csv', 'old_notes.txt'}
    table.write_text('n\n9\n')
    with pytest.raises(ValueError, match='changed'):
        workspace.load_workspace(tmp_path)
    with pytest.raises(ValueError, match='changed'):
        workspace.save_workspace(state, tmp_path)


def test_local_file_references_are_protected_without_adopting_directory_contents(tmp_path):
    workspace.save_workspace({}, tmp_path)
    image = tmp_path / 'figure.png'; image.write_bytes(b'image')
    notes = tmp_path / 'notes.txt'; notes.write_text('not a result')
    state = {'figures': [image], 'folder': tmp_path}
    workspace.save_workspace(state, tmp_path)
    assert set(json.loads((tmp_path / 'workspace.json').read_text())['owned_file_sha256']) == {'figure.png'}
    notes.unlink()
    equivalent(state, workspace.load_workspace(tmp_path))
    image.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        workspace.load_workspace(tmp_path)


def test_stale_revision_cannot_overwrite_newer_review_or_publish_its_outputs(tmp_path):
    initial = workspace.save_workspace({'accepted': [True, True]}, tmp_path, expected_revision=None)
    state, loaded = workspace.load_workspace(tmp_path, with_revision=True)
    assert initial == loaded == workspace.workspace_revision(tmp_path)
    newer = workspace.save_workspace({'accepted': [False, True]}, tmp_path, expected_revision=loaded)
    assert newer != loaded
    before = (tmp_path / 'workspace.json').read_bytes()
    output = tmp_path / 'stale.csv'; output.write_text('stale output')
    with pytest.raises(workspace.WorkspaceConflictError, match='another session'):
        workspace.save_workspace(state, tmp_path, expected_revision=loaded, owned_outputs=[output])
    with pytest.raises(workspace.WorkspaceConflictError):
        workspace.save_workspace(state, tmp_path, expected_revision=None)
    assert (tmp_path / 'workspace.json').read_bytes() == before
    assert workspace.load_workspace(tmp_path)['accepted'] == [False, True]
    assert 'stale.csv' not in json.loads(before)['owned_file_sha256']


def _concurrent_writer(folder, revision, accepted, ready, start, results):
    ready.put(True)
    start.wait(15)
    try:
        result = workspace.save_workspace({'accepted': accepted}, folder, expected_revision=revision)
        results.put(('saved', accepted, result))
    except workspace.WorkspaceConflictError:
        results.put(('conflict', accepted, None))


def test_simultaneous_process_writers_publish_exactly_one_revision(tmp_path):
    revision = workspace.save_workspace({'accepted': [True, True]}, tmp_path)
    context = multiprocessing.get_context('spawn')
    ready, results = context.Queue(), context.Queue()
    start = context.Event()
    children = [context.Process(target=_concurrent_writer,
        args=(str(tmp_path), revision, accepted, ready, start, results))
        for accepted in ([False, True], [True, False])]
    try:
        for child in children: child.start()
        for _ in children: assert ready.get(timeout=20)
        start.set()
        outcomes = [results.get(timeout=20) for _ in children]
        for child in children:
            child.join(20)
            assert child.exitcode == 0
        assert sorted(outcome[0] for outcome in outcomes) == ['conflict', 'saved']
        winner = next(outcome for outcome in outcomes if outcome[0] == 'saved')
        state, actual = workspace.load_workspace(tmp_path, with_revision=True)
        assert state['accepted'] == winner[1] and actual == winner[2]
    finally:
        for child in children:
            if child.is_alive(): child.kill()
            child.join(5)
        ready.close(); results.close()


def _hold_writer_lock(folder, ready):
    with workspace._writer_lock(Path(folder), None):
        ready.set()
        time.sleep(30)


@pytest.mark.skipif(os.name == 'nt', reason='SIGKILL lifecycle probe; Windows lock path has separate platform semantics')
def test_killed_writer_releases_os_lock_and_waiter_can_cancel(tmp_path):
    revision = workspace.save_workspace({'step': 1}, tmp_path)
    before = (tmp_path / 'workspace.json').read_bytes()
    context = multiprocessing.get_context('spawn')
    ready = context.Event()
    child = context.Process(target=_hold_writer_lock, args=(str(tmp_path), ready))
    cancelled = threading.Event()
    timer = threading.Timer(.1, cancelled.set)
    try:
        child.start()
        assert ready.wait(20)
        timer.start()
        start = time.monotonic()
        with pytest.raises(OperationCancelled):
            workspace.save_workspace({'step': 2}, tmp_path, expected_revision=revision, cancel=cancelled)
        assert time.monotonic() - start < 2
        assert (tmp_path / 'workspace.json').read_bytes() == before
        child.kill(); child.join(5)
        assert child.exitcode != 0
        workspace.save_workspace({'step': 3}, tmp_path, expected_revision=revision)
        assert workspace.load_workspace(tmp_path) == {'step': 3}
    finally:
        timer.cancel()
        if child.is_alive(): child.kill()
        child.join(5)


def test_cancelled_stage_registration_never_blesses_unpublished_outputs(tmp_path, monkeypatch):
    revision = workspace.save_workspace({'step': 1}, tmp_path)
    before = (tmp_path / 'workspace.json').read_bytes()
    output = tmp_path / 'partial.csv'; output.write_text('unpublished')
    cancelled = threading.Event()
    original_sha = workspace._sha
    def cancel_after_hash(path, cancel=None):
        result = original_sha(path, cancel)
        if Path(path) == output: cancelled.set()
        return result
    monkeypatch.setattr(workspace, '_sha', cancel_after_hash)
    with pytest.raises(OperationCancelled):
        workspace.save_workspace({'step': 2}, tmp_path, expected_revision=revision,
                                 owned_outputs=[output], cancel=cancelled)
    assert (tmp_path / 'workspace.json').read_bytes() == before
    workspace.save_workspace({'step': 3}, tmp_path, expected_revision=revision)
    output.unlink()
    assert workspace.load_workspace(tmp_path) == {'step': 3}
    assert not json.loads((tmp_path / 'workspace.json').read_text())['owned_file_sha256']
