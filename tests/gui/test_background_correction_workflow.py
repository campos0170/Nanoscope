"""Source-safe UI routes for current full-resolution background comparison/export."""
import json
import threading
from pathlib import Path

import nanoscope  # Qt bootstrap before Qt imports
import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.segment import persistent_key
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.core.form_draft_state import background_state_readable
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.sample_rate_dialog import SampleRateDialog
from nanoscope.gui import analysis_panel as ap


def document(tmp, name='target', *, fs=1000., units='pA', value=100.):
    n = 12000
    t = np.arange(n) / fs
    raw = value + .3*np.sin(2*np.pi*t)
    meta = RecordingMeta(path=tmp/f'{name}.dat', format_name='synthetic', n_samples=n,
                         sample_rate_hz=fs, rate_provenance=Provenance.FILE_META,
                         channels=(ChannelInfo(index=0,name='current',raw_units=units,
                                               kind=ChannelKind.CURRENT),))
    return TraceDocument(meta=meta,channels=(raw.astype(np.float32),))


def install(win, doc, select=True):
    did = win._session.add_document(doc)
    win._create_default_layers(doc)
    if select: win._on_selection(did,0)
    return did


@pytest.fixture
def bench(qapp, qtbot, tmp_path):
    win = MainWindow(); qtbot.addWidget(win)
    did = install(win,document(tmp_path))
    yield win, win._analysis_panel, win._analysis_panel.baseline_workspace, did
    win._analysis_panel.abort_background_previews()
    win._analysis_controller.abort()


def wait(panel, ws, qtbot):
    qtbot.waitUntil(lambda: panel._pending is None and not ws.is_busy(),timeout=20000)


def anchors(ws, *, current='100', end='12000'):
    e = ws.background
    e.source.setCurrentIndex(e.source.findData('anchors'))
    e._append(e.regions,['A','0',end])
    e._append(e.anchors,['A','0',current])
    e._append(e.anchors,['A',end,current])
    return e


def test_comparison_choice_does_not_toggle_main_trace_or_runtime(bench):
    _win,panel,ws,_ = bench
    before=(panel.baseline_panel.show_baseline(),panel.baseline_panel.subtract_baseline(),
            panel.detection_panel.params()['detrend_enabled'])
    ws.output_mode_combo.setCurrentIndex(1)
    ws.output_mode_combo.setCurrentIndex(0)
    assert before == (panel.baseline_panel.show_baseline(),panel.baseline_panel.subtract_baseline(),
                      panel.detection_panel.params()['detrend_enabled'])


def test_anchor_controls_reachable_calculate_and_export_preserves_raw(bench,qtbot,tmp_path):
    win,panel,ws,did=bench; before=win._session.calibrated(did,0).data.copy()
    e=anchors(ws)
    # A comparison source cannot disable unrelated automatic processing controls.
    assert ws.runtime_box.isEnabled() is True
    assert ws.runtime_enabled.isEnabled()
    assert e.pages.currentIndex()==1
    assert background_state_readable(ws.form_state())
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert not ws.is_preview_stale()
    assert abs(np.median(ws.corrected_curve.getData()[1])) < .01
    path=tmp_path/'anchors.csv'
    assert panel.export_baseline_correction(path);wait(panel,ws,qtbot)
    meta=json.loads(path.with_suffix('.meta.json').read_text())
    assert meta['recipe']['source']=='anchors'
    assert meta['source']['metadata']['doc_key']==persistent_key(win._session.document(did))
    assert meta['raw_input_modified'] is False
    assert np.array_equal(before,win._session.calibrated(did,0).data)
    assert len(np.loadtxt(path,delimiter=',',skiprows=1))==len(before)


@pytest.mark.parametrize('bad',['','NaN','Infinity','oops'])
def test_invalid_anchor_draft_is_retained_and_refused(bench,qtbot,bad):
    _win,panel,ws,_=bench; e=anchors(ws,current=bad)
    state=ws.form_state()
    assert background_state_readable(state)
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert ws.is_preview_stale()
    assert 'finite number' in ws.preview_status.text()
    assert e.anchors.item(0,2).text()==bad
    assert ws.corrected_curve.getData()==(None,None)


def test_segment_jump_policy_is_explicit_and_refuses_gaps(bench,qtbot):
    _win,panel,ws,_=bench;e=anchors(ws,end='6000')
    e._append(e.regions,['B','6000','12000']);e._append(e.anchors,['B','6000','110']);e._append(e.anchors,['B','12000','110'])
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert not ws.is_preview_stale()
    y=ws.baseline_curve.getData()[1]
    assert y[0]==100 and y[-1]==110
    e.segment_policy.setCurrentIndex(e.segment_policy.findData('continuous'))
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert 'exactly one region' in ws.preview_status.text()
    e.segment_policy.setCurrentIndex(e.segment_policy.findData('separate'))
    e.regions.item(1,1).setText('6100')
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert ws.is_preview_stale()
    assert 'gap' in ws.preview_status.text().lower() or 'own region' in ws.preview_status.text()


def test_manual_open_pore_reference_has_nonmutating_comparison_export(bench,qtbot,tmp_path):
    _win,panel,ws,_=bench
    ref=panel.baseline_panel.reference_editor
    ref.mode.setCurrentIndex(ref.mode.findData('constant'));ref.current.setText('100')
    ws.background.source.setCurrentIndex(ws.background.source.findData('intervals'))
    before=panel.baseline_panel.params()
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert not ws.is_preview_stale()
    assert np.allclose(ws.baseline_curve.getData()[1],100)
    path=tmp_path/'manual.csv';assert panel.export_baseline_correction(path);wait(panel,ws,qtbot)
    assert json.loads(path.with_suffix('.meta.json').read_text())['recipe']['source']=='intervals'
    assert panel.baseline_panel.params()==before
    assert panel.baseline_panel.reference_mode()=='constant'


def select_reference(win,ws,doc):
    did=install(win,doc,select=False)
    e=ws.background;e.source.setCurrentIndex(e.source.findData('recording'))
    choice={'doc_key':persistent_key(win._session.document(did)),'channel':0}
    i=next(i for i in range(e.reference.count()) if e.reference.itemData(i)==choice)
    e.reference.setCurrentIndex(i)
    return did


def test_reference_recording_export_and_missing_source_refusal(bench,qtbot,tmp_path):
    win,panel,ws,_=bench
    rid=select_reference(win,ws,document(tmp_path,'reference',value=90.))
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert not ws.is_preview_stale()
    assert np.allclose(ws.corrected_curve.getData()[1],10,atol=1e-4)
    path=tmp_path/'reference.csv';assert panel.export_baseline_correction(path);wait(panel,ws,qtbot)
    data=json.loads(path.with_suffix('.meta.json').read_text())
    assert data['reference']['metadata']['doc_key']==persistent_key(win._session.document(rid))
    win._session.close_document(rid)
    assert not panel.export_baseline_correction(tmp_path/'missing.csv')
    assert 'missing' in ws.preview_status.text().lower()
    assert not (tmp_path/'missing.csv').exists()


@pytest.mark.parametrize('units,shift,expected',[('nA','0','units'),('pA','1','cover')])
def test_reference_units_and_uncovered_shift_refused(bench,qtbot,tmp_path,units,shift,expected):
    win,panel,ws,_=bench;select_reference(win,ws,document(tmp_path,'reference',units=units))
    ws.background.shift.setText(shift)
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert expected in ws.preview_status.text().lower()
    assert ws.corrected_curve.getData()==(None,None)


def test_anchor_binding_survives_rate_change_without_relabelling(bench,qtbot):
    win,panel,ws,did=bench;anchors(ws)
    old=ws.background.state()
    dlg=SampleRateDialog(win._session,did,win)
    dlg.mode.setCurrentIndex(dlg.mode.findData('sample_rate'));dlg.rate.setText('500');dlg.apply()
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert 'Confirm current source' in ws.preview_status.text()
    assert ws.background.state()['binding']==old['binding']
    assert ws.background.state()['anchors']==old['anchors']
    ws.background.confirm_source.click()
    assert ws.background.state()['binding']['sample_rate_hz']==500
    dlg.close()


def test_late_rate_worker_is_rejected_and_old_clock_never_becomes_current(bench,qtbot,monkeypatch):
    win,panel,ws,did=bench
    started=threading.Event();release=threading.Event();original=ap._baseline_workspace_arrays
    def delayed(*a,**kw):
        started.set()
        if not release.wait(10): raise RuntimeError('worker release timeout')
        return original(*a,**kw)
    monkeypatch.setattr(ap,'_baseline_workspace_arrays',delayed)
    ws.refresh_button.click();qtbot.waitUntil(started.is_set,timeout=10000)
    try:
        dlg=SampleRateDialog(win._session,did,win)
        dlg.mode.setCurrentIndex(dlg.mode.findData('sample_rate'));dlg.rate.setText('500');dlg.apply();dlg.close()
    finally: release.set()
    wait(panel,ws,qtbot)
    assert ws.is_preview_stale()
    assert 'sample rate' in ws.preview_status.text()
    assert ws.corrected_curve.getData()==(None,None)


def test_other_stage_change_invalidates_baseline_only_scope(bench,qtbot):
    _win,panel,ws,_=bench
    panel.detection_panel.set_params({'filter_cutoff_hz':None,'detrend_enabled':True,'hp_enabled':False})
    ws.output_mode_combo.setCurrentIndex(1);ws.refresh_button.click();wait(panel,ws,qtbot)
    assert not ws.is_preview_stale()
    panel.detection_panel.set_params({'hp_enabled':True})
    assert ws.is_preview_stale()
    ws.refresh_button.click();wait(panel,ws,qtbot)
    assert 'Other enabled Signal chain stages are not included' in ws.preview_status.text()


def test_background_form_draft_returns_with_its_source_and_invalid_text(bench,qtbot,tmp_path):
    win,panel,ws,did=bench;anchors(ws,current='incomplete')
    state=panel.form_state()['background']
    other=install(win,document(tmp_path,'other'))
    assert ws.background.source_key()=='automatic'
    win._on_selection(did,0)
    assert panel.form_state()['background']==state
    # An unrelated detection landing cannot erase a custom export recipe from save.
    panel._form_dirty=False
    assert panel.form_key() in panel.form_drafts()


def test_main_trace_baseline_fit_runs_off_thread_and_source_switch_discards_it(bench,qtbot,tmp_path,monkeypatch):
    win,panel,ws,did=bench
    from nanoscope.analysis import baseline
    original=baseline.characterisation_baseline; started=threading.Event();release=threading.Event();threads=[]
    def delayed(*a,**kw):
        threads.append(threading.get_ident());started.set()
        if not release.wait(10): raise RuntimeError('worker release timeout')
        return original(*a,**kw)
    monkeypatch.setattr(baseline,'characterisation_baseline',delayed)
    panel.baseline_panel.show_track.setChecked(True)
    qtbot.waitUntil(started.is_set,timeout=10000)
    try:
        other=install(win,document(tmp_path,'other'))
        assert panel.current_target()==(other,0)
        assert threading.get_ident() not in threads
    finally: release.set()
    qtbot.waitUntil(lambda:panel._baseline_view_identity is None,timeout=20000)
    assert panel.current_target()==(other,0)


def test_automatic_main_trace_baseline_recalculates_after_calibration(bench,qtbot):
    from nanoscope.core.calibration import ChannelCalibration
    win,panel,_ws,did=bench
    panel.baseline_panel.subtract_view.setChecked(True)
    qtbot.waitUntil(lambda: panel._trace_view.baseline_subtracted_for(did,0) is not None,
                    timeout=20000)
    old = panel._trace_view.baseline_subtracted_for(did,0).copy()
    win._session.set_calibration(did,0,ChannelCalibration(
        gain=2.,offset=5.,target_units='pA',provenance=Provenance.USER))
    # The old-calibration rail must disappear before a replacement can be accepted.
    assert panel._trace_view.baseline_subtracted_for(did,0) is None
    qtbot.waitUntil(lambda: panel._trace_view.baseline_subtracted_for(did,0) is not None,
                    timeout=20000)
    assert np.allclose(panel._trace_view.baseline_subtracted_for(did,0),old*2+5,
                       rtol=1e-5,atol=1e-3)


@pytest.mark.parametrize('policy_scope',['runtime','reference'])
def test_obsolete_endpoint_policy_requires_safe_rerun_even_with_matching_fingerprint(bench,monkeypatch,policy_scope):
    from dataclasses import replace
    from nanoscope.analysis.baseline import (
        BASELINE_PARAM_KEYS, BaselineConfig, baseline_endpoint_policy,
        baseline_sample_count_policy,
    )
    from nanoscope.analysis.detect import detect_events
    from nanoscope.analysis.events import EventSet
    from nanoscope.core.analysis import AnalysisResult
    win,panel,_ws,did=bench
    config = (BaselineConfig(method='block_median',window_s=.5) if policy_scope == 'reference'
              else panel.baseline_panel.config())
    raw = win._session.calibrated(did,0).data
    noisy = raw + np.random.default_rng(361).normal(0.,.1,len(raw))
    df,prov = detect_events(noisy,1000.,baseline_config=config)
    # A matching saved fingerprint cannot bless an unrecorded endpoint policy,
    # either in runtime drift correction or in the reference used for detection.
    if policy_scope == 'runtime':
        prov = dict(prov,detrend_enabled=True,detrend_method='block_median',
                    detrend_sample_count_policy=baseline_sample_count_policy('block_median'))
        policy_key = 'detrend_endpoint_policy'
    else:
        policy_key = 'baseline_endpoint_policy'
    prov.pop(policy_key,None)
    params = dict(panel._form_params())
    params.update({key:value for key,value in config.to_provenance().items() if key in BASELINE_PARAM_KEYS})
    result = AnalysisResult(module_id='detect.schmitt',module_version='1.0',doc_id=did,
        channel=0,params=params,
        calibration_fingerprint=win._session.calibration(did,0).fingerprint(),
        payload=EventSet(df,prov),baseline_fingerprint=prov['baseline_fingerprint'])
    win._session.add_result(result)
    assert not panel._reference_redetection_needed(result)
    for on_screen in (True,False):
        stale = panel._staleness(result,on_screen=on_screen)
        assert stale.kind == ap._STALE_PROCESSING_POLICY and stale.action_label == 'Re-run'
        assert 'depth_pA' in stale.columns and 'dwell_s' in stale.columns
    assert panel._stale_edit_refusal(result,'Nothing changed.')
    assert panel._statistics_staleness(result)
    assert panel._baseline_staleness(result)
    assert 'endpoint policy' in panel._stale_inputs_text(result)
    study_prov = dict(prov,study_fixed_baseline_fingerprint=result.baseline_fingerprint)
    study = replace(result,module_id='detect.study',payload=EventSet(df,study_prov))
    assert panel._baseline_staleness(study)
    assert panel._staleness(study,on_screen=False).kind == ap._STALE_PROCESSING_POLICY
    calls=[]
    monkeypatch.setattr(panel,'confirm_replacing_curation',lambda *_:False)
    monkeypatch.setattr(panel,'_on_rerun',lambda:calls.append('rerun'))
    monkeypatch.setattr(panel,'remeasure_active_detection',lambda:calls.append('remeasure'))
    monkeypatch.setattr(panel,'_on_detect_requested',lambda *_:calls.append('live-detect'))
    panel._on_stale_action(ap._STALE_PROCESSING_POLICY)
    assert calls == []
    monkeypatch.setattr(panel,'confirm_replacing_curation',lambda *_:True)
    panel._on_stale_action(ap._STALE_PROCESSING_POLICY)
    assert calls == ['rerun']
    current = replace(result,payload=EventSet(df,{**prov,
        policy_key:baseline_endpoint_policy('block_median')}))
    assert panel._detrend_policy_staleness(current) is None
