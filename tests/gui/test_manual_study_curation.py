"""User-driven curation snapshots flow into study stages without model execution."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.core.session import Session
from nanoscope.gui import manual_study_window as gui
from nanoscope.gui.incubation_window import IncubationStudyWindow
from nanoscope.gui.worker_thread import drain_detached
from .test_manual_study_gui import add_sources, complete as _complete, prepared


def complete(qtbot,window,action):
    try: _complete(qtbot,window,action)
    except Exception:
        pytest.fail(repr(window.test_failures)+': '+window._status.text())


class MainWorkspace(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._session=Session()
        self.events=pd.DataFrame(dict(event_id=[4,9],start_idx=[20,60],end_idx=[30,70],
            accepted=[False,True],label=['artifact','reviewed'],depth_pA=[19.,.8]))
        self.opened=[]

    def snapshot_study_events(self,record):
        return dict(events=self.events,provenance=dict(module_id='detect.cusum',full_recording=True))

    def open_study_review(self,payload,launch_doc_ids=None): self.opened.append(payload)


@pytest.fixture
def connected(qtbot,tmp_path,monkeypatch):
    parent=MainWorkspace(); qtbot.addWidget(parent)
    window=gui.ManualStudyWindow(parent); qtbot.addWidget(window)
    window.test_failures=[]; window.operationFailed.connect(window.test_failures.append)
    add_sources(window,tmp_path,monkeypatch,qtbot)
    row=window.state['records'][0]; row['prepared']=prepared(row['tag'])
    row['prepared'].detrended += np.random.default_rng(4).normal(0,.05,1000).astype(np.float32)
    row.pop('drafts',None); window.refresh()
    yield window,parent
    window.abort(); drain_detached(3000)


def test_curated_snapshot_needs_no_model_and_invalidates_downstream(connected,qtbot,monkeypatch):
    window,parent=connected
    monkeypatch.setattr(gui.detect,'run_inference',lambda *a,**kw:pytest.fail('Model ran during import'))
    window.state.update(comparison='old',features='old',clustering='old',mixture='old',figures=[])
    complete(qtbot,window,window.use_main_detections)
    row=window.state['records'][0]; catalog=row['analysis'].catalog
    assert row['event_input']=='main_workspace' and 'inference' not in row
    assert catalog.start_idx.tolist()==[20,60] and catalog.end_idx.tolist()==[29,69]
    assert catalog.accepted.tolist()==[False,True]
    assert catalog.source_label.tolist()==['artifact','reviewed']
    assert catalog.source_depth_pA.tolist()==[19.,.8]
    assert catalog.passes_gate.tolist()==[True,True]
    assert all(window.state[key] is None for key in ('comparison','features','clustering','mixture'))
    assert window._table_model.frame.accepted.tolist()==[False,True]
    analyses,_=window._analyses()
    assert len(analyses)==1 and int(analyses[0].catalog.accepted.sum())==1
    parent.events.loc[0,'accepted']=True
    assert row['imported_events'].accepted.tolist()==[False,True]
    assert row['analysis'].catalog.accepted.tolist()==[False,True]


def test_review_survives_remeasurement_and_workspace_reopen(connected,qtbot):
    window,parent=connected
    complete(qtbot,window,window.use_main_detections)
    row=window.state['records'][0]
    complete(qtbot,window,lambda:window._review([row['analysis'].catalog.event_id.iloc[1]],False))
    window._measure_form.controls['min_depth_sigma'].setValue(8)
    complete(qtbot,window,window.measure_candidates)
    assert not window.state['records'][0]['analysis'].catalog.accepted.any()
    folder=window.folder
    complete(qtbot,window,lambda:window.open_workspace(folder))
    assert not window.state['records'][0]['analysis'].catalog.accepted.any()
    assert 'saved main-workspace detections' in window._event_input_label.text()
    assert len(window._analyses()[0])==1
    complete(qtbot,window,window.use_main_detections)
    assert window.state['records'][0]['analysis'].catalog.accepted.tolist()==[False,True]


def test_model_drafts_do_not_block_curated_analysis_and_prepare_resets_input(connected,qtbot,monkeypatch):
    window,_=connected
    complete(qtbot,window,window.use_main_detections)
    window._inference_form.controls['clip_sigma'].setValue(31)
    window._decode_form.controls['enter'].setValue(.7)
    assert len(window._analyses()[0])==1
    old=window.state['records'][0]['prepared']
    monkeypatch.setattr(gui.prep,'prepare_raw',lambda *a,**kw:old)
    complete(qtbot,window,window.prepare_selected)
    row=window.state['records'][0]
    assert 'event_input' not in row and 'analysis' not in row and 'imported_events' not in row


def test_invalid_source_does_not_change_saved_study(connected,qtbot,monkeypatch):
    window,parent=connected
    complete(qtbot,window,window.use_main_detections)
    before=(window.folder/'workspace.json').read_bytes()
    def reject(record): raise ValueError('Select the matching full recording')
    monkeypatch.setattr(parent,'snapshot_study_events',reject)
    assert window.use_main_detections() is False
    assert 'matching full recording' in window._status.text()
    assert (window.folder/'workspace.json').read_bytes()==before


def test_curated_checkpoint_contains_independent_source_evidence(connected,qtbot):
    window,parent=connected
    complete(qtbot,window,window.use_main_detections)
    saved=checkpoint.load_workspace(window.folder)
    row=saved['records'][0]
    assert row['imported_events'].equals(parent.events)
    assert row['analysis'].provenance['imported_events']['source_provenance']['module_id']=='detect.cusum'
    assert row['analysis'].provenance['input_mode']=='imported_events'


def test_preparation_alone_opens_native_tools_and_mutable_sidecar_is_not_owned(connected,qtbot):
    from nanoscope.core.calibration import sidecar_path
    window,parent=connected
    complete(qtbot,window,window.open_main_analysis)
    assert len(parent.opened)==1
    payload=parent.opened[0]
    assert payload['result'].payload.df.empty
    sidecar=sidecar_path(payload['document'].meta.path)
    assert sidecar.is_file()
    sidecar.write_text('{"native_calibration_experiment":true}')
    saved=checkpoint.load_workspace(window.folder)
    assert saved['records'][0]['main_workspace_bridge']['path']==str(payload['document'].meta.path)
    payload['document'].meta.path.write_bytes(b'changed prepared samples')
    with pytest.raises(ValueError,match='hash|changed|integrity|mismatch'):
        checkpoint.load_workspace(window.folder)


def test_automated_recipe_has_reachable_manual_curation_entry(qtbot):
    class Owner(QtWidgets.QWidget):
        calls=0
        def _open_manual_study(self): self.calls+=1
    parent=Owner(); qtbot.addWidget(parent)
    window=IncubationStudyWindow(parent); qtbot.addWidget(window)
    window._manual_workflow_btn.click()
    assert parent.calls==1


def test_real_main_workspace_roundtrip_repeated_updates_and_downstream_features(qtbot,tmp_path,monkeypatch):
    from nanoscope.gui.main_window import MainWindow
    from .test_manual_study_gui import measured
    parent=MainWindow(recovery_enabled=False); qtbot.addWidget(parent)
    monkeypatch.setattr(parent,'_confirm_close_choice',lambda:'discard')
    window=gui.ManualStudyWindow(parent); qtbot.addWidget(window)
    window.test_failures=[]; window.operationFailed.connect(window.test_failures.append)
    try:
        add_sources(window,tmp_path,monkeypatch,qtbot)
        row=measured(window)
        row['prepared'].detrended += np.random.default_rng(4).normal(0,.05,1000).astype(np.float32)
        complete(qtbot,window,window.open_main_analysis)
        native=parent._analysis_panel.current_detection()
        assert native.module_id=='detect.study'
        parent._analysis_panel.set_event_accepted(0,False)
        parent._undo_events()
        assert bool(parent._analysis_panel.current_detection().payload.df.accepted.iloc[0])
        parent._redo_events()
        assert not bool(parent._analysis_panel.current_detection().payload.df.accepted.iloc[0])
        parent._analysis_panel.set_event_accepted(1,True)
        parent._open_event_inspector()
        parent._event_inspector.show_event(1)
        parent._event_inspector._apply_edited_bounds(1,59,72,parent._session.document(native.doc_id).raw(0))
        complete(qtbot,window,window.use_main_detections)
        imported=window.state['records'][0]['analysis'].catalog
        assert imported.accepted.tolist()==[False,True]
        assert imported.start_idx.tolist()==[20,59]
        assert imported.end_idx.tolist()==[29,71]
        complete(qtbot,window,window.confirm_event_review)
        complete(qtbot,window,window.extract_features)
        assert len(window.state['features'])==1
        parent._analysis_panel.set_event_accepted(0,True)
        complete(qtbot,window,window.use_main_detections)
        assert window.state['records'][0]['analysis'].catalog.accepted.tolist()==[True,True]
        assert window.state['features'] is None
        folder=window.folder
        complete(qtbot,window,lambda:window.open_workspace(folder))
        assert window.state['records'][0]['analysis'].catalog.accepted.tolist()==[True,True]
        assert window.open_main_analysis()
        assert parent._analysis_panel.current_detection().payload.df.accepted.tolist()==[True,True]
        assert len(parent._session.documents())==1
    finally:
        window.abort(); drain_detached(3000)
        parent._dirty=False; parent._teardown()
