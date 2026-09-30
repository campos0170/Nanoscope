"""General, Zn and Cu studies use the same native curation and save controls."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.analysis.cancel import check_cancelled
from nanoscope.gui import manual_study_window as gui
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.worker_thread import drain_detached
from .test_manual_study_gui import measured, prepared


def complete(qtbot,study,action):
    failures=[];study.operationFailed.connect(failures.append)
    try:
        assert action() is not False
        qtbot.waitUntil(lambda:not study.is_busy(),timeout=30000)
        assert not failures,failures
    finally:study.operationFailed.disconnect(failures.append)


@pytest.fixture
def windows(qtbot,tmp_path,monkeypatch):
    main=MainWindow(recovery_enabled=False);qtbot.addWidget(main)
    monkeypatch.setattr(main,'_confirm_close_choice',lambda:'discard')
    study=gui.ManualStudyWindow(main);qtbot.addWidget(study)
    main._manual_study_window=study
    def seed(profile='general',count=1):
        study.new_workspace(tmp_path/'study',profile=profile,name='Time series',sample='Test protein')
        paths=[]
        for i in range(count):
            path=tmp_path/f'source-{i}.abf';path.write_bytes(b'raw fixture');paths.append(path)
        monkeypatch.setattr(gui.prep,'inspect_raw',lambda path,**kw:dict(
            path=str(path),channels=[dict(index=0,name='IN11',units='pA')],
            fs_hz=50000.,n_samples=1000,notes=[],issues=[]))
        complete(qtbot,study,lambda:study.add_raw(paths))
        row=measured(study)
        row['prepared'].detrended+=np.random.default_rng(4).normal(0,.05,1000).astype(np.float32)
        row.pop('inference',None);row.pop('model_path',None);study.state['model_path']=''
        row.pop('drafts',None);study.state.pop('drafts',None);study.refresh()
        return row
    yield main,study,seed
    study.abort();drain_detached(3000);main._dirty=False;main._teardown()


@pytest.mark.parametrize('profile',['general','zn','cu'])
def test_profiles_share_actual_table_gallery_inspector_save_and_undo(qtbot,windows,profile):
    main,study,seed=windows;row=seed(profile)
    assert row['median_engine']==('shrinking' if profile=='cu' else 'reflect')
    complete(qtbot,study,study.open_main_analysis)
    result=main._analysis_panel.current_detection()
    assert main._event_gallery.frame() is result.payload.df
    assert main._study_curation_save.isEnabled()
    assert study._main_curation_button.text()=='Curate in main workspace…'
    main._analysis_panel.set_event_accepted(0,False)
    main._undo_events();assert main._analysis_panel.current_detection().payload.df.accepted.iloc[0]
    main._redo_events()
    main._analysis_panel.set_event_accepted(1,True)
    main._open_event_inspector();main._event_inspector.show_event(1)
    main._event_inspector._apply_edited_bounds(1,59,72,main._session.document(result.doc_id).raw(0))
    complete(qtbot,study,main._save_study_curation)
    current=study.state['records'][0]
    assert current['analysis'].catalog.accepted.tolist()==[False,True]
    assert current['analysis'].catalog.start_idx.tolist()==[20,59]
    assert current['analysis'].catalog.end_idx.tolist()==[29,71]
    assert not study._review_is_current(current)
    assert 'inference' not in current
    restored=checkpoint.load_workspace(study.folder)
    assert restored['study_profile']==profile and restored['sample']=='Test protein'
    assert restored['records'][0]['analysis'].catalog.accepted.tolist()==[False,True]
    main._analysis_panel.set_event_accepted(0,True)
    complete(qtbot,study,main._save_study_curation)
    assert study.state['records'][0]['analysis'].catalog.accepted.all()


def test_save_uses_captured_record_committed_settings_after_selection_changes(qtbot,windows,monkeypatch):
    main,study,seed=windows;row=seed('zn',count=2)
    row['measurement_settings']['min_depth_sigma']=7.
    row.pop('drafts',None);study.refresh()
    complete(qtbot,study,study.open_main_analysis)
    study._selector.setCurrentIndex(1)
    study._measure_form.controls['min_depth_sigma'].setValue(13.)
    observed=[];original=gui.detect.measure_imported_events
    def measure(recording,events,settings,**kw):
        observed.append((recording.tag,settings.min_depth_sigma))
        return original(recording,events,settings,**kw)
    monkeypatch.setattr(gui.detect,'measure_imported_events',measure)
    main._analysis_panel.set_event_accepted(0,False)
    complete(qtbot,study,main._save_study_curation)
    assert observed==[('recording_1',7.)]
    assert study.state['records'][0]['analysis'].catalog.accepted.tolist()==[False,False]
    assert 'analysis' not in study.state['records'][1]
    assert study.state['records'][1]['drafts']['measurement']['min_depth_sigma']==13.
    main._return_to_study()
    assert study._selector.currentIndex()==0 and study._stages.currentIndex()==2


def test_reentry_preserves_live_edits_and_study_review_expires_old_copy(qtbot,windows):
    main,study,seed=windows;seed()
    complete(qtbot,study,study.open_main_analysis)
    native=main._analysis_panel.current_detection()
    main._analysis_panel.set_event_accepted(0,False)
    main._return_to_study();assert 'unsaved changes' in study._status.text()
    assert study.open_main_analysis()
    assert len(main._session.documents())==1
    assert main._analysis_panel.current_detection() is native
    event_id=study.state['records'][0]['analysis'].catalog.event_id.iloc[1]
    complete(qtbot,study,lambda:study._review([event_id],True))
    assert not main._study_curation_save.isEnabled()
    with pytest.raises(ValueError,match='study changed'):main._save_study_curation()
    main._return_to_study();complete(qtbot,study,study.open_main_analysis)
    assert len(main._session.documents())==2
    assert main._analysis_panel.current_detection().payload.df.accepted.tolist()==[True,True]
    assert not native.payload.df.accepted.iloc[0]


def test_failed_and_cancelled_save_show_main_feedback_without_changing_checkpoint(qtbot,windows,monkeypatch):
    main,study,seed=windows;seed('cu')
    complete(qtbot,study,study.open_main_analysis)
    main._analysis_panel.set_event_accepted(0,False)
    before=study._revision
    monkeypatch.setattr(gui.detect,'measure_imported_events',lambda *a,**k:(_ for _ in ()).throw(ValueError('measurement failed')))
    with qtbot.waitSignal(study.operationFailed,timeout=10000):main._save_study_curation()
    qtbot.waitUntil(lambda:not study.is_busy(),timeout=5000)
    assert 'measurement failed' in main._study_curation_label.text()
    assert checkpoint.workspace_revision(study.folder)==before
    def slow(*args,cancel=None,**kw):
        while True:check_cancelled(cancel);time.sleep(.005)
    monkeypatch.setattr(gui.detect,'measure_imported_events',slow)
    main._save_study_curation();qtbot.wait(30);study.cancel()
    qtbot.waitUntil(lambda:not study.is_busy(),timeout=5000)
    assert 'canceled' in main._study_curation_label.text()
    assert checkpoint.workspace_revision(study.folder)==before
    assert study.state['records'][0]['analysis'].catalog.accepted.iloc[0]
    assert not main._analysis_panel.current_detection().payload.df.accepted.iloc[0]


def test_project_switch_during_export_never_installs_into_replacement(qtbot,windows,monkeypatch):
    from nanoscope.gui import study_event_bridge as bridge
    main,study,seed=windows;seed()
    entered=threading.Event();release=threading.Event();original=bridge.prepare_review_export
    def paused(*args,**kw):
        result=original(*args,**kw);entered.set();assert release.wait(10);return result
    monkeypatch.setattr(bridge,'prepare_review_export',paused)
    failures=[];study.operationFailed.connect(failures.append)
    study.open_main_analysis();qtbot.waitUntil(entered.is_set,timeout=30000)
    main._clear_session();release.set()
    qtbot.waitUntil(lambda:not study.is_busy(),timeout=30000)
    assert not main._session.documents()
    assert any('main project changed' in message for message in failures)


def test_metadata_edits_and_saved_profile_reopen_without_changing_science(qtbot,windows):
    main,study,seed=windows;row=seed('cu')
    before=row['analysis'];settings=deepcopy(row['preparation_settings'])
    study._study_details_form.controls['name'].setText('New material series')
    study._study_details_form.controls['sample'].setText('Another protein')
    study._study_details_form.controls['condition_label'].setText('Salt condition')
    complete(qtbot,study,study.apply_study_details)
    assert study.state['records'][0]['analysis'] is before
    complete(qtbot,study,lambda:study.open_workspace(study.folder))
    assert study.state['study_profile']=='cu'
    assert study.state['name']=='New material series'
    assert study.state['sample']=='Another protein'
    assert study.state['condition_label']=='Salt condition'
    assert study.state['records'][0]['preparation_settings']==settings
    assert study.state['records'][0]['median_engine']=='shrinking'
    assert 'New material series' in study.windowTitle()


def test_cu_save_preserves_retained_time_map_and_blocks_cross_gap_bounds(qtbot,windows):
    main,study,seed=windows;row=seed('cu')
    p=row['prepared'];p.metadata['retained_ranges']=[[0,50,100,150],[50,1000,300,1250]]
    row['schmitt_settings']=asdict(gui.signal.SchmittSettings())
    native=pd.DataFrame(dict(event_id=[0,1],start_idx=[20,60],end_idx=[30,70],accepted=[True,True]))
    imported=gui.detect.measure_imported_events(p,native,gui.detect.MeasurementSettings(**row['measurement_settings']),source_provenance={})
    row['analysis']=gui.signal.analyze(p,gui.signal.SchmittSettings(),imported=imported)
    row['event_input']='schmitt';row.pop('drafts',None);study.refresh()
    complete(qtbot,study,study.open_main_analysis)
    main._analysis_panel.set_event_accepted(1,True)
    complete(qtbot,study,main._save_study_curation)
    cat=study.state['records'][0]['analysis'].catalog
    np.testing.assert_allclose(cat.original_onset_s,[120/50000,310/50000])
    main._open_event_inspector();main._event_inspector.show_event(0)
    result=main._analysis_panel.current_detection()
    main._event_inspector._apply_edited_bounds(0,20,55,main._session.document(result.doc_id).raw(0))
    before=study._revision
    with qtbot.waitSignal(study.operationFailed,timeout=10000):main._save_study_curation()
    qtbot.waitUntil(lambda:not study.is_busy(),timeout=5000)
    assert study._revision==before
    assert 'excluded source interval' in main._study_curation_label.text()
