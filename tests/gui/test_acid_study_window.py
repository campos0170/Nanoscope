from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import acid_incubation as engine
from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.analysis.cancel import check_cancelled
from nanoscope.gui.acid_study_window import AcidStudyWindow, _SettingsForm, PREPARATION_KEYS, DETECTION_KEYS, ANALYSIS_KEYS
from nanoscope.gui.worker_thread import drain_detached


@pytest.fixture
def window(qtbot):
    w=AcidStudyWindow();qtbot.addWidget(w)
    yield w
    w.abort();drain_detached(5000);w.close()


def complete(qtbot,w,action):
    errors=[];w.operationFailed.connect(errors.append)
    assert action() is not False
    qtbot.waitUntil(lambda:not w.is_busy(),timeout=20000)
    assert not errors, errors
    w.operationFailed.disconnect(errors.append)


def seeded(w,tmp_path):
    w.new_workspace(tmp_path/'study')
    cat=pd.DataFrame({'event_id':['a','b','c','d'],'start_idx':[10,30,50,70],
        'end_idx':[14,35,55,77],'accepted':[True,True,False,True],
        'passes_gate':[True,True,False,True],'event_review':['unreviewed']*4})
    row=dict(tag='acid_30h_whole',source_path=tmp_path/'source.abf',source_info={'sha256':'raw','fs_hz':50000.,'channels':[{'units':'pA'}]},
        recording_id='r1',incubation_h=30.,pore_id='p30',pore_diameter_nm=None,
        concentration_pm=100.,voltage_mv=-300.,channel=0,comparison_only=False,
        catalog=cat,families=pd.DataFrame({'event_id':['a','b','c'],'family':[0,1,1]}),
        detection={'settings_sha256':'settings'})
    w.state['records']=[row];w._revision=checkpoint.save_workspace(w.state,w.folder,expected_revision=w._revision)
    w.refresh();return row


def test_current_engine_defaults_roundtrip_in_real_controls(qtbot):
    defaults=engine.default_settings()
    for keys in (PREPARATION_KEYS,DETECTION_KEYS,ANALYSIS_KEYS):
        form=_SettingsForm(defaults,keys);qtbot.addWidget(form)
        assert form.values()=={k:defaults[k] for k in keys if k in defaults}


def test_new_workspace_has_staged_controls_and_unknown_diameter(window,tmp_path):
    seeded(window,tmp_path);window._assert_no_drafts()
    assert window._stages.count()==6
    assert window._source_form.controls['pore_diameter_nm'].text()==''
    names={b.text() for b in window.findChildren(QtWidgets.QPushButton)}
    assert {'Prepare selected recording','Calibrate and detect selected recording','Accept selected events',
        'Reject selected events','Accept selected families','Reject selected families','Keep only selected families',
        'Build event families for review','Analyze confirmed event selections','Style selected figure…'}<=names


def test_rejected_candidates_can_be_accepted_and_roundtrip(window,tmp_path,qtbot):
    seeded(window,tmp_path)
    window.show_events();window._table.selectRow(2)
    complete(qtbot,window,lambda:window.review_events(True))
    assert window.state['records'][0]['catalog'].accepted.tolist()==[True,True,True,True]
    assert window.state['records'][0]['catalog'].event_review.iloc[2]=='reviewed'
    complete(qtbot,window,window.confirm_selection)
    assert engine.selection_is_current(window.state['records'][0])
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window.state['records'][0]['catalog'].accepted.all()
    assert window.state['records'][0]['review_receipt']['all_events_personally_inspected'] is False


def test_family_keep_only_rejects_unassigned_and_preserves_other_records(window,tmp_path,qtbot):
    row=seeded(window,tmp_path)
    other={**row,'tag':'other','recording_id':'r2','catalog':row['catalog'].copy(deep=True)}
    window.state['records'].append(other);window.refresh()
    window.show_families();window._table.selectRow(0)
    complete(qtbot,window,lambda:window.review_families(True,only=True))
    assert window.state['records'][0]['catalog'].accepted.tolist()==[True,False,False,False]
    assert window.state['records'][1]['catalog'].accepted.tolist()==[True,True,False,True]
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window.state['records'][0]['catalog'].accepted.tolist()==[True,False,False,False]


def test_unapplied_draft_blocks_science_but_is_saved(window,tmp_path,qtbot):
    seeded(window,tmp_path)
    window._prep_form.controls['BASELINE_WINDOW_S'].setValue(3.)
    with pytest.raises(ValueError,match='Apply changed preparation'):window.prepare_record()
    complete(qtbot,window,window.save)
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window._prep_form.controls['BASELINE_WINDOW_S'].value()==3.
    assert window.state['settings']['BASELINE_WINDOW_S']==2.


def test_concurrent_checkpoint_change_refuses_stale_commit(window,tmp_path,qtbot):
    seeded(window,tmp_path);before=window._revision
    newer=dict(window.state);newer['name']='Other session'
    checkpoint.save_workspace(newer,window.folder,expected_revision=before)
    errors=[];window.operationFailed.connect(errors.append)
    window.save();qtbot.waitUntil(lambda:not window.is_busy(),timeout=20000)
    assert any('changed in another session' in e for e in errors)
    assert window._revision==before
    assert checkpoint.load_workspace(window.folder)['name']=='Other session'


def test_cancelled_operation_cannot_publish(window,tmp_path,qtbot):
    seeded(window,tmp_path);before=checkpoint.workspace_revision(window.folder)
    def slow(state,folder,progress,cancel):
        (folder/'partial.dat').write_bytes(b'partial stage output')
        for i in range(500):
            check_cancelled(cancel);time.sleep(.005)
        state['name']='Wrong';return state
    window._stage('Long test',slow,fresh_output=True)
    qtbot.wait(30);window.cancel();qtbot.waitUntil(lambda:not window.is_busy(),timeout=5000)
    assert checkpoint.workspace_revision(window.folder)==before
    assert window.state['name']!='Wrong'
    assert not list((window.folder/'runs').glob('*'))


def test_source_metadata_edits_update_engine_labels_and_expire_review(window,tmp_path,qtbot):
    seeded(window,tmp_path)
    complete(qtbot,window,window.confirm_selection)
    window._source_form.controls['pore_diameter_nm'].setText('26')
    complete(qtbot,window,window.apply_recording_details)
    row=window.state['records'][0]
    assert row['pore_diameter_nm']==row['nominal_pore_nm']==26.
    assert row['pore_uncertain'] is False
    assert not engine.selection_is_current(row)
    window._source_form.controls['pore_diameter_nm'].setText('')
    complete(qtbot,window,window.apply_recording_details)
    row=window.state['records'][0]
    assert row['nominal_pore_nm'] is None and row['pore_uncertain'] is True


def test_appearance_save_preserves_science_and_numeric_recipe(window,tmp_path,qtbot):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from nanoscope.plot.acid_figure_recipe import capture_figure,reconstruct
    seeded(window,tmp_path)
    complete(qtbot,window,window.confirm_selection)
    before=window.state['records'][0]['catalog'].copy(deep=True)
    receipt=window.state['records'][0]['review_receipt'].copy()
    folder=window.folder/'runs'/'initial';folder.mkdir(parents=True)
    path=folder/'test_acid_plot.png'
    figure=Figure(figsize=(5,3));FigureCanvasAgg(figure)
    axis=figure.subplots();axis.plot([1,2,3],[4,8,6],color='#147d92',label='Accepted')
    axis.legend();figure.savefig(path);capture_figure(figure,path)
    window.state['records'][0]['prepared']={'figures':[path]}
    window.state['figures']=[]
    window._revision=checkpoint.save_workspace(window.state,window.folder,expected_revision=window._revision,owned_outputs=checkpoint.stage_output_files(folder))
    window.refresh()
    assert window._figure_selector.count()==1
    assert window.state.get('analysis') is None
    complete(qtbot,window,window.style_selected_figure)
    editor=window._style_windows[-1]
    editor._panel_edits['title'].textEdited.emit('User-styled events')
    assert editor.apply_appearance()
    complete(qtbot,window,editor.save_to_study)
    assert path.is_file()
    current=window.state['figures'][0]
    assert current!=path and current.with_suffix('.recipe.json').is_file()
    rebuilt,_,_=reconstruct(current)
    assert rebuilt.axes[0].get_title()=='User-styled events'
    np.testing.assert_array_equal(rebuilt.axes[0].lines[0].get_ydata(),[4,8,6])
    pd.testing.assert_frame_equal(window.state['records'][0]['catalog'],before)
    assert window.state['records'][0]['review_receipt']==receipt
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window.state['figure_styles']['study::test_acid_plot']['panels']['0']['title']=='User-styled events'
    editor.close()
    complete(qtbot,window,window.style_selected_figure)
    editor=window._style_windows[-1]
    editor.reset_style()
    assert editor._figure.axes[0].get_title()==''
    complete(qtbot,window,editor.save_to_study)
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    restored,_,_=engine.reconstruct_figure(window.state,window.state['figures'][0])
    assert restored.axes[0].get_title()==''
    np.testing.assert_array_equal(restored.axes[0].lines[0].get_ydata(),[4,8,6])
    editor.close()


def test_family_build_control_precedes_any_full_analysis(window,tmp_path,qtbot,monkeypatch):
    seeded(window,tmp_path);window.state['records'][0]['families']=None
    complete(qtbot,window,window.confirm_selection)
    calls=[]
    def build(state,folder,progress=None,cancel=None):
        assert state.get('analysis') is None
        assert engine.selection_is_current(state['records'][0])
        calls.append('family review')
        state['records'][0]['families']=pd.DataFrame({'event_id':['a','b','c'],'family':[0,1,1]})
        return state
    monkeypatch.setattr(engine,'build_families',build)
    monkeypatch.setattr(engine,'analyze_study',lambda *a,**k:pytest.fail('Full analysis ran during family review'))
    complete(qtbot,window,window.build_families)
    window.show_families();window._table.selectRow(1)
    complete(qtbot,window,lambda:window.review_families(False))
    assert calls==['family review']
    assert window.state.get('analysis') is None
    assert window.state['records'][0]['catalog'].accepted.tolist()==[True,False,False,True]
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window.state['records'][0]['catalog'].accepted.tolist()==[True,False,False,True]


def test_native_engine_metadata_diameter_and_summary_tables_display(window,tmp_path):
    row=seeded(window,tmp_path)
    row.pop('pore_diameter_nm');row['nominal_pore_nm']=18.;row['pore_uncertain']=False
    window.state['analysis']={'summaries':[{'tag':row['tag'],'n_accepted':3,'duration_s':1.,'rate_per_s':3.}]}
    window.refresh();window._assert_no_drafts()
    assert window._source_form.controls['pore_diameter_nm'].text()=='18.0'
    window.show_analysis_table()
    assert window._table_model.frame.iloc[0].n_accepted==3


@pytest.mark.parametrize('source_fs',[1000.,50000.])
def test_source_event_preview_preserves_inclusive_bounds(window,tmp_path,source_fs):
    import pyarrow as pa
    import pyarrow.parquet as pq
    row=seeded(window,tmp_path);row['source_info']['fs_hz']=source_fs
    signal=np.arange(100,dtype=np.float32)
    path=tmp_path/'trace.parquet';pq.write_table(pa.table({'current_detr_pA':signal,'current_raw_pA':signal-200}),path,row_group_size=16)
    row['prepared']={'parquet_path':path,'metadata':{'fs_hz':1000.,'n_samples':100}}
    window.show_events();window._table.selectRow(1);window.inspect_selected_event()
    curve=window._trace.listDataItems()[0];x,y=curve.getData()
    np.testing.assert_array_equal(y,np.arange(24,42,dtype=np.float32))
    np.testing.assert_allclose(x,np.arange(24,42)/1000.)
    regions=[i for i in window._trace.items() if type(i).__name__=='LinearRegionItem']
    assert regions[0].getRegion()==(.030,.036)


def _prepared_preview(window,tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    row=seeded(window,tmp_path)
    signal=np.sin(np.arange(1200)/10).astype(np.float32)
    signal[10:15]-=20.;signal[300:306]-=80.;signal[900:906]+=40.
    path=tmp_path/'preview.parquet'
    pq.write_table(pa.table({'current_detr_pA':signal,'current_raw_pA':signal-200}),path,row_group_size=128)
    row['source_info']['fs_hz']=1000.
    row['prepared']={'parquet_path':path,'metadata':{'fs_hz':1000.,'n_samples':len(signal)}}
    row['catalog'].loc[1,['start_idx','end_idx']]=[300,305]
    row['catalog'].loc[2,['start_idx','end_idx']]=[900,905]
    window.show_events()
    return row,signal


def _assert_curve_in_view(window):
    x,y=window._trace.listDataItems()[0].getData()
    xr,yr=window._trace.getViewBox().viewRange()
    assert xr[0]<=min(x)<=max(x)<=xr[1]
    assert yr[0]<=min(y)<=max(y)<=yr[1]


def test_event_navigation_reframes_both_axes_after_pan_and_one_shot_fit(window,tmp_path,qtbot):
    _prepared_preview(window,tmp_path)
    view=window._trace.getViewBox()
    for index in (0,1,2):
        # Panning/zooming disables automatic range updates in the live inspector.
        view.setRange(xRange=(.192,.206),yRange=(-15.,5.),padding=0)
        window._table.selectRow(index);window.inspect_selected_event()
        qtbot.wait(1)
        _assert_curve_in_view(window)
        # An explicit autoRange() call performs a one-shot fit. Navigation must
        # also work after it, without relying on auto-range remaining enabled.
        window._trace.autoRange()
        window._table.selectRow((index+1)%3);window.inspect_selected_event()
        qtbot.wait(1)
        _assert_curve_in_view(window)


def test_full_trace_preview_reframes_and_replaces_event_title(window,tmp_path,qtbot):
    row,signal=_prepared_preview(window,tmp_path)
    window._table.selectRow(1);window.inspect_selected_event()
    assert row['catalog'].iloc[1].event_id in window._trace.plotItem.titleLabel.text
    window._trace.getViewBox().setRange(xRange=(.3,.305),yRange=(-85.,-75.),padding=0)
    window.preview_trace();qtbot.wait(1)
    _assert_curve_in_view(window)
    np.testing.assert_array_equal(window._trace.listDataItems()[0].getData()[1],signal)
    assert window._trace.plotItem.titleLabel.text==f"{row['tag']} · Prepared recording"
    assert not any(type(item).__name__=='LinearRegionItem' for item in window._trace.items())


@pytest.mark.parametrize('sample_rate',[0.,-1000.,float('nan'),float('inf')])
def test_event_preview_rejects_invalid_prepared_clock(window,tmp_path,sample_rate):
    row,_=_prepared_preview(window,tmp_path)
    row['prepared']['metadata']['fs_hz']=sample_rate
    window._table.selectRow(0)
    with pytest.raises(ValueError,match='sample rate'):
        window.inspect_selected_event()
