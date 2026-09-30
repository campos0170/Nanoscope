"""User confirmation gates new study analysis while saved results stay viewable."""
from copy import deepcopy

import pandas as pd
import pytest

from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.analysis.incubation_detection import RecordingAnalysis
from nanoscope.analysis.study_review import REVIEW_KEY, review_is_current
from nanoscope.gui import manual_study_window as gui
from .test_manual_study_gui import add_sources, catalog, complete, measured, prepared, window


@pytest.mark.parametrize("action,engine", [
    ("run_comparisons", "compare_recordings"),
    ("extract_features", "extract_features"),
    ("run_clustering", "cluster_features"),
    ("run_mixture", "fit_mixture"),
])
def test_pending_review_blocks_each_downstream_engine(window,tmp_path,monkeypatch,qtbot,action,engine):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    row=measured(window)
    window.state['features']=pd.DataFrame({'tag':[row['tag']],'depth_pA':[20.]})
    monkeypatch.setattr(gui.compare,engine,lambda *a,**k:pytest.fail('Unreviewed events reached '+engine))
    assert getattr(window,action)() is False
    assert not window.is_busy()
    assert row['tag'] in window._status.text()
    assert 'Use reviewed events for analysis' in window._status.text()
    assert REVIEW_KEY not in window.state['records'][0]


def test_confirmation_preserves_decisions_and_real_feature_extraction_uses_them(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    row=measured(window);tag=row['tag']
    before=row['analysis'].catalog.copy(deep=True)
    complete(qtbot,window,window.confirm_event_review)
    reviewed=window.state['records'][0]
    receipt=deepcopy(reviewed[REVIEW_KEY])
    pd.testing.assert_frame_equal(reviewed['analysis'].catalog,before)
    assert review_is_current(reviewed)
    complete(qtbot,window,window.extract_features)
    assert window.state['features'].event_id.tolist()==[tag+':0']
    complete(qtbot,window,lambda:window._review([tag+':0'],False))
    complete(qtbot,window,lambda:window._review([tag+':1'],True))
    updated=window.state['records'][0]
    assert updated['analysis'].catalog.accepted.tolist()==[False,True]
    assert updated['analysis'].catalog.passes_gate.tolist()==[True,False]
    assert not review_is_current(updated)
    assert REVIEW_KEY not in updated
    assert receipt['accepted_count']==1
    assert window.state['features'] is None
    assert window.extract_features() is False
    complete(qtbot,window,window.confirm_event_review)
    assert window.state['records'][0][REVIEW_KEY]['fingerprint']!=receipt['fingerprint']
    complete(qtbot,window,window.extract_features)
    assert window.state['features'].event_id.tolist()==[tag+':1']
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert review_is_current(window.state['records'][0])
    assert window.state['features'].event_id.tolist()==[tag+':1']


def test_review_edit_saves_pending_status_and_reopen_does_not_approve_it(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    row=measured(window);tag=row['tag']
    complete(qtbot,window,window.confirm_event_review)
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert review_is_current(window.state['records'][0])
    complete(qtbot,window,lambda:window._review([tag+':1'],True))
    saved=checkpoint.load_workspace(window.folder)['records'][0]
    assert REVIEW_KEY not in saved and not review_is_current(saved)
    assert saved['analysis'].catalog.accepted.tolist()==[True,True]
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert not review_is_current(window.state['records'][0])
    assert window.extract_features() is False


def test_source_repreparation_requires_new_review_even_with_same_parameters(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    row=measured(window);tag=row['tag']
    complete(qtbot,window,window.confirm_event_review)
    monkeypatch.setattr(gui.prep,'prepare_raw',lambda *a,**kw:prepared(tag))
    complete(qtbot,window,window.prepare_selected)
    updated=window.state['records'][0]
    assert 'analysis' not in updated and REVIEW_KEY not in updated
    assert not review_is_current(updated)


def test_removing_earlier_recording_preserves_survivors_review(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot,count=2)
    survivor=window.state['records'][1];tag=survivor['tag'];data=prepared(tag)
    survivor['prepared']=data
    survivor['analysis']=RecordingAnalysis(tag,data.raw,data.detrended,data.fs_hz,
        data.duration_s,catalog(tag),data.provenance)
    window.refresh();window._selector.setCurrentIndex(1)
    complete(qtbot,window,window.confirm_event_review)
    receipt=deepcopy(window.state['records'][1][REVIEW_KEY])
    before=window.state['records'][1]['analysis'].catalog.copy(deep=True)
    window._selector.setCurrentIndex(0)
    complete(qtbot,window,window.remove_recording)
    assert len(window.state['records'])==1
    remaining=window.state['records'][0]
    assert remaining['tag']==tag and remaining[REVIEW_KEY]==receipt
    assert review_is_current(remaining)
    pd.testing.assert_frame_equal(remaining['analysis'].catalog,before)
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window.state['records'][0][REVIEW_KEY]==receipt
    assert review_is_current(window.state['records'][0])


def test_unapplied_scientific_settings_cannot_be_confirmed(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    measured(window)
    window._measure_form.controls['min_depth_sigma'].setValue(8)
    assert window.confirm_event_review() is False
    assert 'changed recording stages' in window._status.text()
    assert REVIEW_KEY not in window.state['records'][0]


def test_styling_and_save_reopen_preserve_review_and_existing_analysis(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    measured(window)
    complete(qtbot,window,window.confirm_event_review)
    complete(qtbot,window,window.extract_features)
    receipt=deepcopy(window.state['records'][0][REVIEW_KEY])
    features=window.state['features'].copy(deep=True)
    # This is the committed style store used by the shared figure editor.
    window.state['figure_styles']={'example':{'font_family':'Arial','font_size':14}}
    complete(qtbot,window,window.save)
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    assert window.state['figure_styles']['example']['font_size']==14
    assert window.state['records'][0][REVIEW_KEY]==receipt
    assert review_is_current(window.state['records'][0])
    pd.testing.assert_frame_equal(window.state['features'],features)


def test_filtered_review_uses_event_identity_and_retains_automatic_gate(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    row=measured(window);tag=row['tag']
    window.show_candidates()
    window._review_filter.setCurrentText('Rejected only')
    assert window._table_model.frame.event_id.tolist()==[tag+':1']
    window._table.selectRow(0)
    complete(qtbot,window,lambda:window.review_selected(True))
    updated=window.state['records'][0]['analysis'].catalog
    assert updated.accepted.tolist()==[True,True]
    assert updated.passes_gate.tolist()==[True,False]
    window.show_candidates()
    window._review_filter.setCurrentText('Accepted only')
    assert window._table_model.frame.event_id.tolist()==[tag+':0',tag+':1']
    window._table.selectRow(0)
    complete(qtbot,window,lambda:window.review_selected(False))
    assert window.state['records'][0]['analysis'].catalog.accepted.tolist()==[False,True]


def test_legacy_result_can_be_viewed_without_implicitly_approving_it(window,tmp_path,monkeypatch,qtbot):
    add_sources(window,tmp_path,monkeypatch,qtbot)
    row=measured(window)
    assignments=pd.DataFrame({'event_id':[row['tag']+':0'],'PC1':[1.],'PC2':[2.],'kmeans3':[0]})
    window.state['clustering']=(assignments,{'status':'computed'})
    complete(qtbot,window,window.save)
    complete(qtbot,window,lambda:window.open_workspace(window.folder))
    monkeypatch.setattr(gui.compare,'cluster_features',lambda *a,**k:pytest.fail('Displaying saved data ran clustering'))
    window.show_clustering()
    pd.testing.assert_frame_equal(window._table_model.frame,assignments)
    assert not review_is_current(window.state['records'][0])
    assert window.run_clustering() is False
