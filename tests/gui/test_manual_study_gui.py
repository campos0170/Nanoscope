"""Independent manual-stage controls, truthful drafts and owned worker lifetimes."""
from dataclasses import asdict
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtGui

from nanoscope.analysis import incubation_study
from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.analysis import manual_detection as detection
from nanoscope.analysis.cancel import check_cancelled
from nanoscope.analysis.incubation_detection import RecordingAnalysis
from nanoscope.analysis.manual_preparation import RawPreparedRecording
from nanoscope.gui import manual_study_window as gui
from nanoscope.gui.worker_thread import drain_detached


@pytest.fixture
def window(qtbot):
    window=gui.ManualStudyWindow(); qtbot.addWidget(window)
    window.test_failures=[];window.operationFailed.connect(window.test_failures.append)
    yield window
    window.abort();drain_detached(3000)


def complete(qtbot,window,action):
    previous=len(window.test_failures)
    with qtbot.waitSignal(window.operationFinished,timeout=5000):
        assert action()
    qtbot.waitUntil(lambda:not window.is_busy(),timeout=3000)
    assert window.test_failures[previous:]==[]


def add_sources(window,tmp_path,monkeypatch,qtbot,count=1):
    window.new_workspace(tmp_path/'workspace')
    paths=[]
    for i in range(count):
        p=tmp_path/f'source-{i}.abf';p.write_bytes(b'synthetic mock recording');paths.append(p)
    def inspect(path,**kwargs):
        return dict(path=str(path),channels=[dict(index=0,name='IN11',units='pA'),dict(index=1,name='IN14',units='mV')],
                    fs_hz=50000.,n_samples=1000,source=dict(path=str(path)),notes=[],issues=[])
    monkeypatch.setattr(gui.prep,'inspect_raw',inspect)
    complete(qtbot,window,lambda:window.add_raw(paths))
    return paths


def prepared(tag):
    raw=np.full(1000,-200.,np.float64);detr=np.zeros(1000,np.float32)
    detr[20:30]=-20;detr[60:70]=-1
    baseline=np.full(1000,-200.,np.float32)
    meta=dict(event_polarity=1,qc_mad_noise_pA=1.)
    return RawPreparedRecording(tag,raw,detr,50000.,.02,meta,
        dict(metadata=meta,event_polarity=1),raw+detr,baseline)


def catalog(tag):
    return pd.DataFrame(dict(event_id=[tag+':0',tag+':1'],start_idx=[20,60],end_idx=[29,69],
        onset_s=[20/50000,60/50000],end_s=[30/50000,70/50000],dwell_ms=[.2,.2],
        depth_peak_pA=[20.,1.],depth_sust_pA=[20.,1.],fracblock=[.1,.005],
        passes_gate=[True,False],accepted=[True,False]))


def measured(window):
    row=window.state['records'][0];p=prepared(row['tag']);row['prepared']=p
    row['inference']=detection.InferenceResult(np.zeros(1000,np.float32),{},dict(test_model=True))
    row['candidates']=catalog(row['tag']).drop(columns=['passes_gate','accepted'])
    row['analysis']=RecordingAnalysis(row['tag'],p.raw,p.detrended,p.fs_hz,p.duration_s,catalog(row['tag']),p.provenance)
    row['model_path']='model-A.keras';window.state['model_path']='model-A.keras'
    row.pop('drafts',None);window.state.pop('drafts',None);window.refresh()
    return row


def test_each_stage_is_explicit_and_never_calls_automated_recipe(window,tmp_path,monkeypatch,qtbot):
    calls=[];add_sources(window,tmp_path,monkeypatch,qtbot)
    monkeypatch.setattr(incubation_study,'run_study',lambda *a,**k:pytest.fail('manual controls called automated study'))
    def prepare(path,channel,tag,config,**kwargs):calls.append('prepare');return prepared(tag)
    def infer(recording,model,settings,**kwargs):
        calls.append('infer');return detection.InferenceResult(np.zeros(1000,np.float32),{},dict(model=model))
    def candidates(recording,inference,settings,**kwargs):calls.append('candidates');return catalog(recording.tag)
    def measure(recording,candidates,settings,**kwargs):
        calls.append('measure');return RecordingAnalysis(recording.tag,recording.raw,recording.detrended,
            recording.fs_hz,recording.duration_s,catalog(recording.tag),recording.provenance)
    monkeypatch.setattr(gui.prep,'prepare_raw',prepare);monkeypatch.setattr(gui.detect,'run_inference',infer)
    monkeypatch.setattr(gui.detect,'find_candidates',candidates);monkeypatch.setattr(gui.detect,'measure_candidates',measure)
    complete(qtbot,window,window.prepare_selected);assert calls==['prepare']
    window._model.setText(str(tmp_path/'student.keras'))
    complete(qtbot,window,window.run_inference);assert calls==['prepare','infer']
    complete(qtbot,window,window.find_candidates);assert calls==['prepare','infer','candidates']
    complete(qtbot,window,window.measure_candidates);assert calls==['prepare','infer','candidates','measure']
    saved=checkpoint.load_workspace(window.folder)
    assert len(saved['records'][0]['analysis'].catalog)==2
    assert saved['records'][0]['analysis'].catalog.accepted.sum()==1


def test_drafts_survive_selection_save_and_reopen(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot,count=2)
    window._prep_form.controls['masked_base_s'].setValue(.321)
    window._selector.setCurrentIndex(1)
    window._measure_form.controls['min_depth_sigma'].setValue(7)
    window._selector.setCurrentIndex(0)
    assert window._prep_form.controls['masked_base_s'].value()==.321
    assert 'preparation' in window._pending.text()
    complete(qtbot,window,window.save)
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window._prep_form.controls['masked_base_s'].value()==.321
    window._selector.setCurrentIndex(1)
    assert window._measure_form.controls['min_depth_sigma'].value()==7
    assert window.state['records'][0]['preparation_settings']['masked_base_s']!=.321


def test_pending_source_and_upstream_inference_cannot_run_downstream(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot);measured(window)
    monkeypatch.setattr(gui.prep,'prepare_raw',lambda *a,**k:pytest.fail('unapplied source settings started prep'))
    window._source_form.controls['channel'].setValue(2)
    assert not window.prepare_selected()
    assert 'Apply recording details' in window._status.text()
    window.restore_settings()
    window._inference_form.controls['clip_sigma'].setValue(31)
    assert not window.measure_candidates()
    assert 'Inference settings' in window._status.text()
    window.restore_settings()
    window._measure_form.controls['min_depth_sigma'].setValue(9)
    with pytest.raises(ValueError,match='changed recording stages'):window._analyses()


def test_selecting_record_restores_its_model_identity(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot,count=2);measured(window)
    window.state['records'][1]['model_path']='model-B.keras'
    window._selector.setCurrentIndex(1);assert window._model.text()=='model-B.keras'
    window._selector.setCurrentIndex(0);assert window._model.text()=='model-A.keras'


def test_review_and_figure_render_preserve_automatic_gate_decisions(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot);row=measured(window);tag=row['tag']
    complete(qtbot,window,lambda:window._review([tag+':1'],True))
    updated=window.state['records'][0]['analysis'].catalog
    assert updated.passes_gate.tolist()==[True,False]
    assert updated.accepted.tolist()==[True,True]
    observed=[]
    def render(analysis,accepted,folder,**kwargs):
        observed.append((analysis.catalog.passes_gate.tolist(),len(accepted)));return []
    monkeypatch.setattr(gui.figures,'recording_figures',render)
    complete(qtbot,window,window.render_figures)
    assert observed==[([True,False],2)]
    saved=checkpoint.load_workspace(window.folder)['records'][0]['analysis'].catalog
    assert saved.passes_gate.tolist()==[True,False] and saved.accepted.tolist()==[True,True]


def test_cancel_keeps_previous_checkpoint_and_ownership_until_acknowledged(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    entered=threading.Event();release=threading.Event();folder=window.folder
    def compute(progress,cancel):
        entered.set();release.wait(3);check_cancelled(cancel);return 'unexpected'
    try:
        assert window._stage('Slow operation',compute,lambda state,result:state.update(unexpected=result))
        qtbot.waitUntil(entered.is_set,timeout=2000)
        window.cancel();assert window.is_busy() and not window._toolbar.isEnabled()
        assert not window.open_workspace(folder)
        with qtbot.waitSignal(window.operationCancelled,timeout=4000):release.set()
        qtbot.waitUntil(lambda:not window.is_busy(),timeout=3000)
        assert 'unexpected' not in checkpoint.load_workspace(folder)
        complete(qtbot,window,lambda:window.open_workspace(folder))
        assert len(window.state['records'])==1
    finally:release.set()


def test_closing_discards_late_result_and_reopens_previous_checkpoint(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    entered=threading.Event();release=threading.Event();folder=window.folder
    def compute(progress,cancel):entered.set();release.wait(3);return 'late result'
    try:
        assert window._stage('Slow close',compute,lambda state,result:state.update(late=result))
        qtbot.waitUntil(entered.is_set,timeout=2000);window.close();release.set()
        qtbot.waitUntil(lambda:not window.is_busy(),timeout=4000)
        assert 'late' not in checkpoint.load_workspace(folder)
        complete(qtbot,window,lambda:window.open_workspace(folder))
        assert 'late' not in window.state
    finally:release.set()


def test_invalidated_figures_and_old_event_titles_are_cleared(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot);row=measured(window)
    image=QtGui.QImage(10,10,QtGui.QImage.Format_RGB32);image.fill(QtGui.QColor('red'))
    path=window.folder/'old-figure.png';assert image.save(str(path))
    window.state['figures']=[path];window.refresh();window.show_figure(0)
    assert len(window._figure.scene().items())==1
    window._trace.setTitle('previous event title');window.preview_trace()
    assert window._trace.plotItem.titleLabel.text!='previous event title'
    monkeypatch.setattr(gui.prep,'prepare_raw',lambda *a,**kwargs:prepared(row['tag']))
    complete(qtbot,window,window.prepare_selected)
    assert window.state['figures']==[] and window._figure.scene().items()==[]
    window.state['records']=[];window.refresh()
    assert window._trace.listDataItems()==[] and window._source_info.text()==''


def test_renderers_receive_committed_custom_settings_and_refuse_changed_drafts(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot);measured(window)
    comparison_settings=gui.compare.ComparisonSettings(n_blocks=7,n_boot=50)
    cluster_settings=gui.compare.ClusterSettings(kmeans_k=4,with_tsne=False)
    window.state['comparison']=gui.compare.ComparisonResult(pd.DataFrame(),pd.DataFrame(),{},dict(test=True))
    window.state['comparison_settings']=asdict(comparison_settings)
    window.state['clustering']=(pd.DataFrame({'PC1':[1.],'PC2':[2.],'kmeans4':[0]}),{})
    window.state['cluster_settings']=asdict(cluster_settings)
    window.state.pop('drafts',None);window.refresh()
    observed=[]
    def comparison(*args,settings,**kwargs):observed.append(('comparison',settings.n_blocks,settings.n_boot));return []
    def clustering(*args,settings,**kwargs):observed.append(('clustering',settings.kmeans_k));return []
    monkeypatch.setattr(gui.figures,'comparison_figures',comparison)
    monkeypatch.setattr(gui.figures,'clustering_figures',clustering)
    window._figure_group.setCurrentIndex(1);complete(qtbot,window,window.render_figures)
    window._figure_group.setCurrentIndex(2);complete(qtbot,window,window.render_figures)
    assert observed==[('comparison',7,50),('clustering',4)]
    window._cluster_form.controls['kmeans_k'].setValue(5)
    assert not window.render_figures()
    assert 'Clustering settings changed' in window._status.text()
    assert observed==[('comparison',7,50),('clustering',4)]


@pytest.mark.parametrize('failure_stage',['second_write','second_publication'])
def test_paired_event_export_rolls_back_on_second_output_failure(tmp_path,monkeypatch,failure_stage):
    output=tmp_path/'export';output.mkdir();frame=catalog('sample')
    if failure_stage=='second_write':
        real=pd.DataFrame.to_csv
        def fail(self,path,*args,**kwargs):
            if Path(path).name.endswith('_accepted.csv'):raise OSError('simulated second output write failure')
            return real(self,path,*args,**kwargs)
        monkeypatch.setattr(pd.DataFrame,'to_csv',fail)
    else:
        real=gui.os.link
        def fail(source,target,*args,**kwargs):
            if Path(target).name.endswith('_accepted.csv'):raise OSError('simulated second output publication failure')
            return real(source,target,*args,**kwargs)
        monkeypatch.setattr(gui.os,'link',fail)
    with pytest.raises(OSError,match='simulated second output'):
        gui._export_event_tables(frame,'sample',output)
    assert list(output.iterdir())==[]


@pytest.mark.parametrize('suffix',['candidates','accepted'])
def test_paired_event_export_preserves_existing_either_output(tmp_path,suffix):
    target=tmp_path/f'sample_{suffix}.csv';target.write_text('existing result')
    with pytest.raises(FileExistsError):gui._export_event_tables(catalog('sample'),'sample',tmp_path)
    assert target.read_text()=='existing result'
    assert list(tmp_path.iterdir())==[target]


def test_paired_event_export_preserves_racing_writer(tmp_path,monkeypatch):
    real=gui.os.link
    def racing(source,target,*args,**kwargs):
        if Path(target).name.endswith('_accepted.csv'):Path(target).write_text('independent writer')
        return real(source,target,*args,**kwargs)
    monkeypatch.setattr(gui.os,'link',racing)
    with pytest.raises(FileExistsError):gui._export_event_tables(catalog('sample'),'sample',tmp_path)
    assert not (tmp_path/'sample_candidates.csv').exists()
    assert (tmp_path/'sample_accepted.csv').read_text()=='independent writer'
    assert len(list(tmp_path.iterdir()))==1


def test_paired_event_export_cancel_between_publications_rolls_back(tmp_path,monkeypatch):
    from nanoscope.analysis.cancel import OperationCancelled
    cancelled=threading.Event();real=gui.os.link
    def first_link(source,target,*args,**kwargs):
        result=real(source,target,*args,**kwargs);cancelled.set();return result
    monkeypatch.setattr(gui.os,'link',first_link)
    with pytest.raises(OperationCancelled):
        gui._export_event_tables(catalog('sample'),'sample',tmp_path,cancel=cancelled.is_set)
    assert list(tmp_path.iterdir())==[]


def failed(qtbot,window,action):
    with qtbot.waitSignal(window.operationFailed,timeout=5000):
        assert action()
    qtbot.waitUntil(lambda:not window.is_busy(),timeout=3000)


def test_add_remove_add_allocates_unique_ids_and_focuses_selected_trace(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot,count=2)
    complete(qtbot,window,window.remove_recording)
    survivor_id=window.state['records'][0]['recording_id']
    extra=tmp_path/'extra.abf';extra.write_bytes(b'new synthetic source')
    complete(qtbot,window,lambda:window.add_raw([extra]))
    rows=window.state['records']
    assert len({r['tag'] for r in rows})==len({r['recording_id'] for r in rows})==2
    assert rows[0]['recording_id']==survivor_id
    for ordinal,row in enumerate(rows):
        row['prepared']=prepared(row['tag']);row['prepared'].detrended[20:30]=-20*(ordinal+1)
        row['candidates']=catalog(row['tag'])
    window.refresh();window._selector.setCurrentIndex(1);window.show_candidates()
    window._focus_event(window._table_model.index(0,0))
    assert window._trace.listDataItems()[0].yData.min()==-40
    complete(qtbot,window,window.save)
    assert len({r['tag'] for r in checkpoint.load_workspace(window.folder)['records']})==2


def test_ambiguous_event_identity_is_refused_and_old_trace_cleared(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot,count=2);measured(window)
    window.show_candidates();window._focus_event(window._table_model.index(0,0))
    assert window._trace.listDataItems()
    window.state['records'][1]['tag']=window.state['records'][0]['tag']
    assert window._focus_event(window._table_model.index(0,0)) is False
    assert 'Duplicate recording tag' in window._status.text()
    assert not window._trace.listDataItems() and window._trace.plotItem.titleLabel.text==''


def test_unchanged_source_apply_preserves_results_but_changed_conditions_invalidate(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot);row=measured(window)
    comparison=gui.compare.ComparisonResult(pd.DataFrame({'tag':[row['tag']]}),pd.DataFrame({'n':[1]}),{}, {})
    features=pd.DataFrame({'tag':[row['tag']],'depth_pA':[20.]})
    window.state.update(comparison=comparison,features=features,clustering=(features,{}),mixture=(features,None,{}),mixture_tag=row['tag'])
    image=QtGui.QImage(10,10,QtGui.QImage.Format_RGB32);image.fill(QtGui.QColor('red'))
    path=window.folder/'saved-figure.png';assert image.save(str(path));window.state['figures']=[path]
    window.refresh();record_id=row['recording_id'];analysis=row['analysis']
    complete(qtbot,window,window.apply_recording_details)
    assert window.state['comparison'] is comparison and window.state['features'] is features
    assert window.state['clustering'] is not None and window.state['mixture'] is not None and window.state['figures']==[path]
    saved=checkpoint.load_workspace(window.folder)
    assert saved['clustering'] is not None and saved['mixture'] is not None and saved['figures']==[path]
    window._source_form.controls['incubation_h'].setValue(5)
    complete(qtbot,window,window.apply_recording_details)
    assert all(window.state[k] is None for k in ('comparison','features','clustering','mixture'))
    assert window.state['figures']==[] and window.state['records'][0]['analysis'] is analysis
    assert window.state['records'][0]['recording_id']==record_id


@pytest.mark.parametrize('damage',['empty_state','duplicate_tag','invalid_source','display_failure'])
def test_failed_open_preserves_active_workspace_revision_and_unsaved_drafts(window,tmp_path,monkeypatch,qtbot,damage):
    add_sources(window,tmp_path,monkeypatch,qtbot,count=2);measured(window)
    window.show_candidates();window._prep_form.controls['masked_base_s'].setValue(.432)
    old_state,old_folder,old_revision=window.state,window.folder,window._revision
    incoming=tmp_path/'bad-workspace';state=gui._copy_state(window.state)
    if damage=='empty_state':state={}
    elif damage=='duplicate_tag':state['records'][1]['tag']=state['records'][0]['tag']
    elif damage=='invalid_source':state['records'][1]['source_info']=None
    checkpoint.save_workspace(state,incoming)
    if damage=='display_failure':
        real=window.refresh
        def refresh(**kwargs):
            if window.folder==incoming:raise RuntimeError('simulated display failure')
            return real(**kwargs)
        monkeypatch.setattr(window,'refresh',refresh)
    failed(qtbot,window,lambda:window.open_workspace(incoming))
    assert window.state is old_state and window.folder==old_folder and window._revision==old_revision
    assert window._prep_form.controls['masked_base_s'].value()==.432
    assert 'previous workspace' in window._status.text()
    assert not window._loading
    complete(qtbot,window,window.save)
    assert checkpoint.load_workspace(old_folder)['records'][0]['drafts']['preparation']['masked_base_s']==.432


def test_stale_session_preserves_newer_review_and_keeps_local_drafts(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot);row=measured(window)
    complete(qtbot,window,window.save)
    other=gui.ManualStudyWindow();qtbot.addWidget(other)
    other.test_failures=[];other.operationFailed.connect(other.test_failures.append)
    try:
        complete(qtbot,other,lambda:other.open_workspace(window.folder))
        revision=other._revision
        other._prep_form.controls['masked_base_s'].setValue(.456)
        complete(qtbot,window,lambda:window._review([row['tag']+':0'],False))
        current=(window.folder/'workspace.json').read_bytes()
        failed(qtbot,other,other.save)
        assert (window.folder/'workspace.json').read_bytes()==current
        assert other._revision==revision and other._prep_form.controls['masked_base_s'].value()==.456
        assert other.state['records'][0]['analysis'].catalog.accepted.tolist()==[True,False]
        assert checkpoint.load_workspace(window.folder)['records'][0]['analysis'].catalog.accepted.tolist()==[False,False]
        assert 'another session' in other._status.text() and 'drafts remain' in other._status.text()
        complete(qtbot,other,lambda:other.open_workspace(window.folder))
        assert other._revision==window._revision
        complete(qtbot,other,other.save)
    finally:other.abort();drain_detached(3000)


def test_saved_result_display_failure_keeps_published_revision_and_recovers_on_open(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    window._prep_form.controls['masked_base_s'].setValue(.482)
    previous=window._revision
    with monkeypatch.context() as patch:
        def refresh(**kwargs):
            window._loading=True
            raise RuntimeError('simulated rendering failure after publication')
        patch.setattr(window,'refresh',refresh)
        failed(qtbot,window,window.save)
    assert window._revision!=previous and window._revision==checkpoint.workspace_revision(window.folder)
    assert not window._loading and 'Reopen the workspace' in window._status.text()
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window._prep_form.controls['masked_base_s'].value()==.482
    complete(qtbot,window,window.save)


def _saved_quantification():
    from nanoscope.analysis.subpop_gmm import PCSelection,SubpopFit,FractionCI,SubpopQuantification
    selection=PCSelection(4,2,-1.,np.array([[-3.,-2.],[-2.,-1.],[-4.,-3.]]),(2,4,6),(1,2),2,0,1)
    fit=SubpopFit(pd.Series([0,1]),np.eye(2),np.ones(2),1.,1.,np.array([[0.],[1.]]),np.array([[0.],[1.]]),
        np.array([.6,.4]),{0:1,1:1},1,2,np.arange(2),np.ones(1),bic={1:7.,2:5.},bic_k=2)
    ci=FractionCI(np.array([.4,.2]),np.array([.8,.6]),np.array([[.6,.4],[.5,.5]]),2,2,0)
    table=pd.DataFrame(dict(subpop=[0,1],n=[1,1],fraction=[.5,.5],rate_per_s=[50.,50.],frac_lo95=[.4,.2],frac_hi95=[.8,.6]))
    return SubpopQuantification(selection,fit,ci,table,pd.DataFrame([[1,0],[0,1]]),('depth_pA',),2)


@pytest.mark.parametrize('view',['clustering','Assignments','Population quantification','Cross-validation grid','Mixture-weight bootstrap','BIC ladder','Exploratory crosstab'])
def test_saved_population_tables_redisplay_after_save_and_open_without_fitting(window,tmp_path,monkeypatch,qtbot,view):
    add_sources(window,tmp_path,monkeypatch,qtbot);row=measured(window)
    assignments=catalog(row['tag']).assign(tag=row['tag'],subpop=[0,1],kmeans3=[0,1])
    window.state.update(features=assignments,clustering=(assignments,{}),mixture=(assignments,_saved_quantification(),{}),mixture_tag=row['tag'])
    window.refresh()
    monkeypatch.setattr(gui.compare,'cluster_features',lambda *a,**k:pytest.fail('redisplay reran clustering'))
    monkeypatch.setattr(gui.compare,'fit_mixture',lambda *a,**k:pytest.fail('redisplay reran mixture'))
    def show():
        if view=='clustering':window.show_clustering()
        else:window._mixture_table.setCurrentText(view);window.show_mixture()
    show();expected=window._table_model.frame.copy()
    assert not expected.empty
    if view=='Population quantification':
        assert expected.mixture_weight.tolist()==[.6,.4]
        assert expected.assigned_event_fraction.tolist()==[.5,.5]
    elif view=='Cross-validation grid':assert expected.n_pcs.tolist()==[2,4,6]
    elif view=='Exploratory crosstab':assert expected.subpopulation.tolist()==[0,1]
    complete(qtbot,window,window.save)
    pd.testing.assert_frame_equal(window._table_model.frame,expected)
    assert window._results.currentWidget() is window._table
    complete(qtbot,window,lambda:window.open_workspace(window.folder));show()
    pd.testing.assert_frame_equal(window._table_model.frame,expected)


def test_unprepared_selection_clears_previous_trace_title(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot,count=2);measured(window)
    assert window._trace.plotItem.titleLabel.text
    window._selector.setCurrentIndex(1)
    assert window._trace.plotItem.titleLabel.text=='' and not window._trace.listDataItems()


def test_new_workspace_allows_finder_metadata_but_refuses_user_files(window,tmp_path):
    folder=tmp_path/'finder';folder.mkdir();(folder/'.DS_Store').write_bytes(b'Finder metadata')
    assert window.new_workspace(folder)
    (folder/'.DS_Store').write_bytes(b'Finder metadata changed')
    assert checkpoint.load_workspace(folder)['records']==[]
    other=tmp_path/'not-empty';other.mkdir();(other/'notes.txt').write_text('keep')
    with pytest.raises(ValueError,match='empty folder'):window.new_workspace(other)
    assert (other/'notes.txt').read_text()=='keep' and window.folder==folder


@pytest.mark.parametrize('polarity,version,invalidated',[(1,None,False),(-1,None,True),(-1,2,False)])
def test_open_old_upward_features_requires_reextraction_without_changing_disk(window,tmp_path,monkeypatch,qtbot,polarity,version,invalidated):
    add_sources(window,tmp_path,monkeypatch,qtbot);row=measured(window)
    row['event_polarity']=polarity;row['analysis'].provenance['event_polarity']=polarity
    row['prepared'].metadata['event_polarity']=polarity;row.pop('recording_id')
    assignments=pd.DataFrame({'tag':[row['tag']],'depth_pA':[20.]})
    if version is not None:assignments.attrs['methods']={'feature_orientation_version':version}
    window.state.update(features=assignments,clustering=(assignments,{}),mixture=(assignments,None,{}),mixture_tag=row['tag'])
    folder=tmp_path/'legacy';checkpoint.save_workspace(window.state,folder)
    before=(folder/'workspace.json').read_bytes()
    complete(qtbot,window,lambda:window.open_workspace(folder))
    assert (folder/'workspace.json').read_bytes()==before
    assert bool(window.state['records'][0]['recording_id'])
    assert (window.state['features'] is None)==invalidated
    assert (window.state['clustering'] is None)==invalidated and (window.state['mixture'] is None)==invalidated
    assert window.state['records'][0]['analysis'].catalog.accepted.tolist()==[True,False]
    if invalidated:assert 'Extract waveform features again' in window._pending.text()
