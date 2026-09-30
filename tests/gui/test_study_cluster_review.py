"""Cluster curation changes the canonical events and survives downstream invalidation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.analysis import study_cluster_review as review, study_review
from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.analysis.incubation_detection import RecordingAnalysis
from nanoscope.gui import manual_study_window as gui
from .test_manual_study_gui import add_sources, complete, measured, prepared, window


def clustered(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot,count=2)
    first=measured(window)
    template=first['analysis'].catalog.copy(deep=True)
    third=template.iloc[[1]].copy();third['start_idx']=100;third['end_idx']=109
    third['onset_s']=100/50000;third['end_s']=110/50000
    template=pd.concat([template,third],ignore_index=True)
    template['event_id']=[0,1,2]
    template['accepted']=[True,True,False]
    frames=[]
    for row in window.state['records']:
        p=prepared(row['tag']);row['prepared']=p
        row['inference']=first['inference'];row['model_path']='model-A.keras'
        row['analysis']=RecordingAnalysis(row['tag'],p.raw,p.detrended,p.fs_hz,p.duration_s,
            template.copy(deep=True),deepcopy(p.provenance))
        row['candidates']=row['analysis'].catalog.drop(columns=['passes_gate','accepted'])
        row.pop('drafts',None)
        row[study_review.REVIEW_KEY]=study_review.confirm_review(row)
        frame=row['analysis'].catalog.iloc[:2].copy();frame['tag']=row['tag']
        frame['kmeans2']=[0,1];frame['hdbscan']=[-1,0]
        frames.append(frame)
    assignments=pd.concat(frames,ignore_index=True)
    window.state.update(features=assignments.copy(),clustering=(assignments,{'kmeans_column':'kmeans2','status':'computed'}))
    mixture=assignments.copy();mixture['subpop']=[0.,1.,np.nan,np.nan]
    window.state['mixture']=(mixture,None,{'mixture_tag':first['tag']})
    window.state['mixture_tag']=first['tag']
    window.state['cluster_settings']['kmeans_k']=2
    window.state['figure_styles']={'example':{'appearance':{'line_width':2.5}}}
    window.state.pop('drafts',None);window.refresh()
    return assignments


def choose(window,labels,*,kind='pooled',method='kmeans2'):
    control=window._cluster_review
    control.source.setCurrentIndex(control.source.findData(kind))
    control.method.setCurrentIndex(control.method.findData(method))
    control.table.clearSelection()
    for i in range(control.table.rowCount()):
        if control.table.item(i,0).data(QtCore.Qt.UserRole) in labels:
            control.table.selectionModel().select(control.table.model().index(i,0),
                QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    assert set(control.selected_labels())==set(labels)


def test_bulk_review_keeps_memberships_and_individual_edits_make_mixed_groups(window,tmp_path,monkeypatch,qtbot):
    clustered(window,tmp_path,monkeypatch,qtbot)
    complete(qtbot,window,window.load_cluster_review)
    snapshot=window.state['cluster_reviews']['pooled'];fingerprint=snapshot['snapshot_id']
    gate=[r['analysis'].catalog.passes_gate.tolist() for r in window.state['records']]
    choose(window,[0])
    assert window.show_cluster_members() is not False
    assert window._table_model.frame.event_id.tolist()==[0,0]
    assert window._table_model.frame.tag.nunique()==2
    complete(qtbot,window,lambda:window.review_clusters(False))
    assert [r['analysis'].catalog.accepted.tolist() for r in window.state['records']]==[[False,True,False]]*2
    assert all(window.state[key] is None for key in ('comparison','features','clustering','mixture'))
    assert window.state['cluster_reviews']['pooled']['snapshot_id']==fingerprint
    assert review.current(snapshot,window.state['records'])
    assert all(not study_review.review_is_current(r) for r in window.state['records'])
    choose(window,[0]);complete(qtbot,window,lambda:window.review_clusters(True))
    window._selector.setCurrentIndex(0);window.show_candidates();window._table.selectRow(0)
    complete(qtbot,window,lambda:window.review_selected(False))
    summary=review.summaries(snapshot,window.state['records'],'kmeans2').set_index('label')
    assert summary.loc[0,'n_accepted']==1 and summary.loc[0,'n_rejected']==1
    assert [r['analysis'].catalog.passes_gate.tolist() for r in window.state['records']]==gate
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert review.current(window.state['cluster_reviews']['pooled'],window.state['records'])
    choose(window,[0]);complete(qtbot,window,lambda:window.review_clusters(True))
    assert [r['analysis'].catalog.accepted.tolist() for r in window.state['records']]==[[True,True,False]]*2
    assert window.state['figure_styles']['example']['appearance']['line_width']==2.5


def test_keep_only_selected_groups_controls_real_features_and_confirmation(window,tmp_path,monkeypatch,qtbot):
    clustered(window,tmp_path,monkeypatch,qtbot)
    complete(qtbot,window,window.load_cluster_review)
    choose(window,[1]);complete(qtbot,window,lambda:window.review_clusters(True,only=True))
    assert [r['analysis'].catalog.accepted.tolist() for r in window.state['records']]==[[False,True,False]]*2
    assert window.extract_features() is False
    complete(qtbot,window,window.confirm_cluster_review)
    assert all(study_review.review_is_current(r) for r in window.state['records'])
    complete(qtbot,window,window.extract_features)
    assert sorted(zip(window.state['features'].tag,window.state['features'].event_id))==[(r['tag'],1) for r in window.state['records']]
    choose(window,[review.UNASSIGNED]);complete(qtbot,window,lambda:window.review_clusters(True,only=True))
    assert [r['analysis'].catalog.accepted.tolist() for r in window.state['records']]==[[False,False,True]]*2


def test_mixture_review_scope_does_not_change_other_recordings(window,tmp_path,monkeypatch,qtbot):
    clustered(window,tmp_path,monkeypatch,qtbot)
    window._cluster_review.source.setCurrentIndex(1)
    complete(qtbot,window,window.load_cluster_review)
    other=deepcopy(window.state['records'][1][study_review.REVIEW_KEY])
    choose(window,[0],kind='mixture',method='subpop')
    complete(qtbot,window,lambda:window.review_clusters(True,only=True))
    assert window.state['records'][0]['analysis'].catalog.accepted.tolist()==[True,False,False]
    assert window.state['records'][1]['analysis'].catalog.accepted.tolist()==[True,True,False]
    assert window.state['records'][1][study_review.REVIEW_KEY]==other
    complete(qtbot,window,window.confirm_cluster_review)
    assert all(study_review.review_is_current(r) for r in window.state['records'])
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    choose(window,[1],kind='mixture',method='subpop')
    complete(qtbot,window,lambda:window.review_clusters(True))
    assert window.state['records'][0]['analysis'].catalog.accepted.tolist()==[True,True,False]


def test_new_measurement_retires_even_numerically_identical_cluster_membership(window,tmp_path,monkeypatch,qtbot):
    clustered(window,tmp_path,monkeypatch,qtbot)
    complete(qtbot,window,window.load_cluster_review)
    original=window.state['records'][0]['analysis']
    def apply(state,result):state['records'][0]['analysis']=replace(original,catalog=original.catalog.copy())
    complete(qtbot,window,lambda:window._stage('New measurement',lambda p,c:None,apply))
    assert not window.state['cluster_reviews']
    assert window.review_clusters(False) is False


def test_stale_memberships_refuse_bulk_changes_atomically_and_can_reopen(window,tmp_path,monkeypatch,qtbot):
    clustered(window,tmp_path,monkeypatch,qtbot)
    complete(qtbot,window,window.load_cluster_review)
    choose(window,[0])
    window.state['records'][1]['analysis'].catalog.loc[0,'depth_sust_pA']+=1
    before=[r['analysis'].catalog.accepted.tolist() for r in window.state['records']]
    revision=window._revision
    with qtbot.waitSignal(window.operationFailed,timeout=5000):
        assert window.review_clusters(False)
    qtbot.waitUntil(lambda:not window.is_busy())
    assert window._revision==revision
    assert [r['analysis'].catalog.accepted.tolist() for r in window.state['records']]==before
    complete(qtbot,window,window.save)
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert not any(button.isEnabled() for button in window._cluster_review.actions)


def test_new_clustering_run_replaces_review_membership_without_changing_acceptance(window,tmp_path,monkeypatch,qtbot):
    assignments=clustered(window,tmp_path,monkeypatch,qtbot)
    complete(qtbot,window,window.load_cluster_review)
    old=window.state['cluster_reviews']['pooled']['snapshot_id']
    choose(window,[0])
    changed=assignments.copy();changed['kmeans2']=1-changed.kmeans2
    monkeypatch.setattr(gui.compare,'cluster_features',lambda *a,**k:(changed,{'kmeans_column':'kmeans2','status':'computed'}))
    complete(qtbot,window,window.run_clustering)
    snapshot=window.state['cluster_reviews']['pooled']
    assert snapshot['snapshot_id']!=old
    assert window._cluster_review.selected_labels()==[]
    choose(window,[0]);complete(qtbot,window,lambda:window.review_clusters(False))
    assert [r['analysis'].catalog.accepted.tolist() for r in window.state['records']]==[[True,False,False]]*2


def test_malformed_snapshot_is_rejected_before_workspace_install(window,tmp_path,monkeypatch,qtbot):
    clustered(window,tmp_path,monkeypatch,qtbot)
    complete(qtbot,window,window.load_cluster_review)
    state=deepcopy(window.state);state['cluster_reviews']['pooled']['methods']=['bad']
    with pytest.raises(ValueError):gui._validated_workspace_state(state)


def test_accepting_an_already_accepted_cluster_preserves_completed_results(window,tmp_path,monkeypatch,qtbot):
    clustered(window,tmp_path,monkeypatch,qtbot)
    complete(qtbot,window,window.load_cluster_review)
    features=window.state['features'];clustering=window.state['clustering']
    receipts=[deepcopy(r[study_review.REVIEW_KEY]) for r in window.state['records']]
    choose(window,[0]);complete(qtbot,window,lambda:window.review_clusters(True))
    assert window.state['features'] is features and window.state['clustering'] is clustering
    assert [r[study_review.REVIEW_KEY] for r in window.state['records']]==receipts


def test_rejecting_all_clusters_keeps_zero_events_through_downstream_stages(window,tmp_path,monkeypatch,qtbot):
    clustered(window,tmp_path,monkeypatch,qtbot)
    for record in window.state['records']:
        analysis=record['analysis']
        record['analysis']=replace(analysis,provenance={**analysis.provenance,'noise_mad_pA':1.})
    complete(qtbot,window,window.load_cluster_review)
    choose(window,[0,1,review.UNASSIGNED]);complete(qtbot,window,lambda:window.review_clusters(False))
    complete(qtbot,window,window.confirm_cluster_review)
    complete(qtbot,window,window.run_comparisons)
    assert window.state['comparison'].summary.n_events.tolist()==[0,0]
    complete(qtbot,window,window.extract_features)
    assert window.state['features'].empty
    complete(qtbot,window,window.run_clustering)
    assert window.state['clustering'][1]['status']=='insufficient_events'
    assert all(not row['analysis'].catalog.accepted.any() for row in window.state['records'])


@pytest.mark.parametrize('operation,helper', [
    ('load_cluster_review','create_snapshot'),
    ('review_clusters','apply_decision'),
    ('confirm_cluster_review','confirm_review'),
])
def test_cluster_actions_leave_event_loop_responsive_through_worker_compute(
        window,tmp_path,monkeypatch,qtbot,operation,helper):
    import threading
    clustered(window,tmp_path,monkeypatch,qtbot)
    if operation!='load_cluster_review':
        complete(qtbot,window,window.load_cluster_review);choose(window,[0])
    entered=threading.Event();release=threading.Event();gui_thread=threading.get_ident()
    owner=study_review if helper=='confirm_review' else review
    original=getattr(owner,helper)
    def blocked(*args,**kwargs):
        assert threading.get_ident()!=gui_thread
        entered.set()
        assert release.wait(5)
        return original(*args,**kwargs)
    monkeypatch.setattr(owner,helper,blocked)
    beats=[];timer=QtCore.QTimer();timer.setInterval(5);timer.timeout.connect(lambda:beats.append(1));timer.start()
    try:
        assert (window.review_clusters(False) if operation=='review_clusters' else getattr(window,operation)())
        qtbot.waitUntil(entered.is_set,timeout=3000)
        qtbot.wait(40)
        assert len(beats)>=3 and window.is_busy()
        with qtbot.waitSignal(window.operationFinished,timeout=5000): release.set()
        qtbot.waitUntil(lambda:not window.is_busy())
        assert not window.test_failures
    finally:
        release.set();timer.stop()


def test_cluster_cancel_during_bulk_compute_preserves_checkpoint(window,tmp_path,monkeypatch,qtbot):
    import threading
    clustered(window,tmp_path,monkeypatch,qtbot)
    complete(qtbot,window,window.load_cluster_review);choose(window,[0])
    before=window.state;revision=window._revision
    entered=threading.Event();release=threading.Event();original=review.apply_decision
    def blocked(*args,**kwargs):
        entered.set();assert release.wait(5)
        return original(*args,**kwargs)
    monkeypatch.setattr(review,'apply_decision',blocked)
    try:
        assert window.review_clusters(False)
        qtbot.waitUntil(entered.is_set)
        window.cancel()
        with qtbot.waitSignal(window.operationCancelled,timeout=5000):release.set()
        qtbot.waitUntil(lambda:not window.is_busy())
        assert window.state is before and window._revision==revision
        saved=checkpoint.load_workspace(window.folder)
        assert [r['analysis'].catalog.accepted.tolist() for r in saved['records']]==[[True,True,False]]*2
    finally:release.set()


def test_refresh_and_selection_do_not_revalidate_full_catalog_on_gui_thread(window,tmp_path,monkeypatch,qtbot):
    import threading
    clustered(window,tmp_path,monkeypatch,qtbot)
    gui_thread=threading.get_ident();calls=[]
    for owner,name in ((review,'require_current'),(review,'summaries'),(study_review,'review_is_current')):
        original=getattr(owner,name)
        def checked(*args,_original=original,_name=name,**kwargs):
            assert threading.get_ident()!=gui_thread, _name+' traversed the catalog in the GUI thread'
            calls.append(_name);return _original(*args,**kwargs)
        monkeypatch.setattr(owner,name,checked)
    complete(qtbot,window,window.load_cluster_review)
    count=len(calls)
    choose(window,[0]);window.show_cluster_members();window.refresh(preserve_table=True)
    window._selector.setCurrentIndex(1);window._cluster_review.method.setCurrentIndex(1)
    assert len(calls)==count
    assert window.cluster_review_view('pooled')['current']


def test_late_real_queued_worker_signals_after_close_are_ignored(window,tmp_path,monkeypatch,qtbot):
    import threading
    from nanoscope.gui.worker_thread import drain_detached
    add_sources(window,tmp_path,monkeypatch,qtbot)
    entered=threading.Event();release=threading.Event();installed=[]
    def operation(progress,cancel):
        progress(.4,'Queued before close')
        entered.set();assert release.wait(3)
        progress(.8,'Queued while closing')
        return 'late result'
    assert window._start('Late callback probe',operation,installed.append)
    assert entered.wait(3)  # Do not deliver queued Qt callbacks before closing.
    release.set();window.close();drain_detached(3000)
    qtbot.wait(30)
    assert window._worker is None and not installed and not window.test_failures
    # Python can also deliver a retired sender as None. All terminal slots must
    # reject that case, rather than allowing None == the absent current worker.
    status=window._status.text();progress=window._progress.value()
    window._on_progress(.9,'Stale progress');window._on_finished('stale result')
    window._on_failed('stale failure');window._on_cancelled()
    assert window._status.text()==status and window._progress.value()==progress
    assert not window.test_failures
