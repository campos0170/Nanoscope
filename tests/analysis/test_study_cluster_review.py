"""Frozen cluster curation controls canonical decisions without losing membership."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.incubation_detection import PreparedRecording, RecordingAnalysis
from nanoscope.analysis.manual_detection import recording_identity
from nanoscope.analysis.manual_workspace import load_workspace, save_workspace
from nanoscope.analysis import study_cluster_review as clusters
from nanoscope.analysis.study_review import (REVIEW_KEY, confirm_review, review_is_current,
    EVENT_REVIEW_COL, event_reviews, apply_event_decision)


def record(tag):
    raw=np.full(100,-100.,dtype=np.float64);signal=np.zeros(100,np.float32)
    prepared=PreparedRecording(tag,raw,signal,1000.,.1,{}, {'event_polarity':1})
    catalog=pd.DataFrame({'event_id':[tag+':0',tag+':1',tag+':2'],
        'start_idx':[10,30,50],'end_idx':[15,35,55], 'accepted':[True,True,False],
        'passes_gate':[True,True,False],'depth_pA':[10.,20.,2.]})
    analysis=RecordingAnalysis(tag,raw,signal,1000.,.1,catalog,
        {'recording_identity':recording_identity(prepared),'event_polarity':1})
    row={'recording_id':'id-'+tag,'tag':tag,'prepared':prepared,'analysis':analysis,
         'preparation_settings':{'window':.5},'measurement_settings':{'sigma':4.}}
    row[REVIEW_KEY]=confirm_review(row)
    return row


@pytest.fixture
def records():
    return [record('zn'),record('cu')]


def assignments(records):
    frames=[]
    for row in records:
        frame=row['analysis'].catalog.iloc[:2][['event_id','start_idx','end_idx']].copy()
        if frame.empty:continue
        frame['tag']=row['tag'];frame['kmeans3']=[0,1];frame['hdbscan']=[-1,0]
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def snapshot(records):
    return clusters.create_snapshot(records,assignments(records),{'kmeans_column':'kmeans3'},
        kind='pooled',settings={'kmeans_k':3})


def test_frozen_membership_includes_unassigned_and_original_input_is_detached(records):
    table=assignments(records);metadata={'kmeans_column':'kmeans3'}
    result=clusters.create_snapshot(records,table,metadata,kind='pooled',settings={})
    assert result['methods']==['kmeans3','hdbscan']
    assert result['recording_ids']==['id-zn','id-cu']
    assert len(result['memberships'])==6
    assert sum(result['memberships'].kmeans3==clusters.UNASSIGNED)==2
    table.loc[0,'kmeans3']=99;metadata['kmeans_column']='changed'
    assert clusters.current(result,records)
    assert clusters.members(result,records,'kmeans3',[0])=={'id-cu':['cu:0'],'id-zn':['zn:0']}


def test_reject_reaccept_and_individual_override_leave_grouping_usable(records):
    frozen=snapshot(records);before=[r['analysis'].catalog.copy(deep=True) for r in records]
    updated,action=clusters.apply_decision(frozen,records,'kmeans3',[0],False)
    assert action['changed_count']==action['target_count']==2
    assert action['changes'][0]['before'] is True and action['changes'][0]['after'] is False
    assert clusters.current(frozen,updated)
    assert all(not review_is_current(r) for r in updated)
    for original,table,new in zip(records,before,updated):
        pd.testing.assert_frame_equal(original['analysis'].catalog,table)
        assert new['analysis'].catalog.accepted.tolist()==[False,True,False]
        assert new['analysis'].catalog.passes_gate.tolist()==[True,True,False]
    restored,_=clusters.apply_decision(frozen,updated,'kmeans3',[0],True)
    assert all(r['analysis'].catalog.accepted.tolist()==[True,True,False] for r in restored)
    restored[0]['analysis'].catalog.loc[0,'accepted']=False
    assert clusters.current(frozen,restored)
    summary=clusters.summaries(frozen,restored,'kmeans3').set_index('label')
    assert summary.loc[0,'status']=='mixed' and summary.loc[0,'n_accepted']==1
    assert summary.loc[clusters.UNASSIGNED,'n_total']==2


def test_keep_only_changes_whole_domain_and_can_include_unassigned(records):
    frozen=snapshot(records)
    updated,action=clusters.apply_decision(frozen,records,'hdbscan',[clusters.UNASSIGNED],True,only=True)
    assert action['changed_count']==6 and action['target_count']==6
    assert all(r['analysis'].catalog.accepted.tolist()==[False,False,True] for r in updated)
    assert all(r['analysis'].catalog.passes_gate.tolist()==[True,True,False] for r in updated)
    assert clusters.current(frozen,updated)


def test_mixture_domain_uses_target_only_and_preserves_other_recording(records):
    table=assignments(records);table['subpop']=[0.,1.,np.nan,np.nan]
    frozen=clusters.create_snapshot(records,table,{'mixture_tag':'zn'},kind='mixture',settings={})
    assert frozen['recording_ids']==['id-zn'] and len(frozen['memberships'])==3
    updated,action=clusters.apply_decision(frozen,records,'subpop',[1],True,only=True)
    assert updated[0]['analysis'].catalog.accepted.tolist()==[False,True,False]
    assert updated[1]['analysis'] is records[1]['analysis']
    assert review_is_current(updated[1])
    assert all(c['recording_id']=='id-zn' for c in action['changes'])
    table.loc[2,'subpop']=0.
    with pytest.raises(ValueError,match='outside'):
        clusters.create_snapshot(records,table,{'mixture_tag':'zn'},kind='mixture',settings={})


@pytest.mark.parametrize('change',[
    lambda r:r['analysis'].catalog.loc.__setitem__((0,'start_idx'),11),
    lambda r:r['analysis'].catalog.loc.__setitem__((0,'event_id'),'new-event'),
    lambda r:r['analysis'].catalog.loc.__setitem__((0,'depth_pA'),11.),
    lambda r:r['analysis'].provenance['recording_identity'].update(detrended_sha256='changed'),
    lambda r:r['preparation_settings'].update(window=1.),
    lambda r:r['measurement_settings'].update(sigma=5.),
])
def test_stale_member_in_one_record_refuses_entire_action_atomically(records,change):
    frozen=snapshot(records);change(records[1])
    before=[r['analysis'].catalog.copy(deep=True) for r in records]
    assert not clusters.current(frozen,records)
    with pytest.raises(ValueError,match='stale'):
        clusters.apply_decision(frozen,records,'kmeans3',[0],False)
    for row,table in zip(records,before):
        pd.testing.assert_frame_equal(row['analysis'].catalog,table)
    clusters.validate_snapshot(frozen)  # A stale source does not damage saved evidence.


def test_row_reordering_and_saved_rejected_group_remain_reversible(records,tmp_path):
    frozen=snapshot(records)
    updated,_=clusters.apply_decision(frozen,records,'kmeans3',[0],False)
    updated[0]['analysis']=replace(updated[0]['analysis'],catalog=updated[0]['analysis'].catalog.iloc[::-1].copy())
    assert clusters.current(frozen,updated[::-1])
    save_workspace({'records':updated,'cluster_reviews':{'pooled':frozen}},tmp_path)
    loaded=load_workspace(tmp_path);frozen=loaded['cluster_reviews']['pooled']
    assert clusters.current(frozen,loaded['records'])
    restored,_=clusters.apply_decision(frozen,loaded['records'],'kmeans3',[0],True)
    assert bool(restored[0]['analysis'].catalog.set_index('event_id').loc['zn:0','accepted'])


@pytest.mark.parametrize('label',[True,False,0.,'0',None,3.5,np.nan])
def test_user_label_choices_are_strict(records,label):
    frozen=snapshot(records)
    with pytest.raises(ValueError,match='labels'):
        clusters.members(frozen,records,'kmeans3',[label])


@pytest.mark.parametrize('damage',['duplicate','foreign_id','foreign_tag','float_bounds','fractional_label','bool_label'])
def test_invalid_assignment_contract_refused(records,damage):
    table=assignments(records)
    if damage=='duplicate':table=pd.concat([table,table.iloc[:1]],ignore_index=True)
    elif damage=='foreign_id':table.loc[0,'event_id']='unknown'
    elif damage=='foreign_tag':table.loc[0,'tag']='unknown'
    elif damage=='float_bounds':table['start_idx']=table.start_idx.astype(float)
    elif damage=='fractional_label':table['kmeans3']=table.kmeans3.astype(float);table.loc[0,'kmeans3']=.5
    else:table['kmeans3']=[True,False,True,False]
    with pytest.raises(ValueError):
        clusters.create_snapshot(records,table,{'kmeans_column':'kmeans3'},kind='pooled',settings={})


def test_zero_event_domain_record_is_retained_and_structural_tampering_refused(records):
    records[1]['analysis'].catalog=records[1]['analysis'].catalog.iloc[:0].copy()
    frozen=snapshot(records)
    assert 'id-cu' in frozen['recording_ids']
    assert clusters.current(frozen,records)
    altered=deepcopy(frozen);altered['memberships'].loc[0,'kmeans3']=99
    with pytest.raises(ValueError,match='changed'):
        clusters.validate_snapshot(altered)


def test_cluster_noop_records_attention_without_invalidating_approval_or_membership(records, tmp_path):
    frozen = snapshot(records)
    updated, action = clusters.apply_decision(frozen, records, 'kmeans3', [0], True)
    assert action['changed_count'] == 0 and action['target_count'] == 2
    for old, new in zip(records, updated):
        assert event_reviews(new['analysis']).tolist() == ['reviewed', 'unreviewed', 'unreviewed']
        assert new['analysis'].provenance is old['analysis'].provenance
        assert new[REVIEW_KEY] == old[REVIEW_KEY] and review_is_current(new)
        pd.testing.assert_frame_equal(new['analysis'].catalog.drop(columns=EVENT_REVIEW_COL), old['analysis'].catalog)
    assert clusters.current(frozen, updated)
    save_workspace({'records': updated, 'snapshot': frozen}, tmp_path)
    restored = load_workspace(tmp_path)
    assert clusters.current(restored['snapshot'], restored['records'])
    assert all(review_is_current(row) for row in restored['records'])
    restored['records'][0]['analysis'] = apply_event_decision(restored['records'][0]['analysis'], ['zn:1'], True)
    assert clusters.current(frozen, restored['records'])


def test_keep_only_marks_every_decided_row_and_leaves_external_domain_untouched(records):
    table = assignments(records)
    table['subpop'] = [0., 1., np.nan, np.nan]
    frozen = clusters.create_snapshot(records, table, {'mixture_tag': 'zn'}, kind='mixture', settings={})
    updated, _ = clusters.apply_decision(frozen, records, 'subpop', [0], True, only=True)
    assert event_reviews(updated[0]['analysis']).tolist() == ['reviewed'] * 3
    assert EVENT_REVIEW_COL not in updated[1]['analysis'].catalog
    assert updated[1]['analysis'] is records[1]['analysis']
    assert clusters.current(frozen, updated)


def test_cluster_gesture_resolves_only_targeted_native_label_conflicts(records):
    from nanoscope.analysis.study_review import EVENT_REVIEW_CONFLICT_COL
    for row in records:
        row['analysis'].catalog['source_consensus_curation_conflict'] = 'conflict'
        row[REVIEW_KEY] = confirm_review(row)
    frozen = snapshot(records)
    updated, action = clusters.apply_decision(frozen, records, 'kmeans3', [0], True)
    assert action['changed_count'] == 0
    for old, new in zip(records, updated):
        assert new['analysis'].catalog[EVENT_REVIEW_CONFLICT_COL].tolist() == ['', 'conflict', 'conflict']
        assert new['analysis'].catalog.source_consensus_curation_conflict.tolist() == ['conflict'] * 3
        assert new[REVIEW_KEY] == old[REVIEW_KEY] and review_is_current(new)
    assert clusters.current(frozen, updated)


def test_unchanged_bulk_action_preserves_existing_review_receipts(records):
    frozen=snapshot(records)
    updated,action=clusters.apply_decision(frozen,records,'kmeans3',[0],True)
    assert action['changed_count']==0
    assert all(review_is_current(r) for r in updated)
