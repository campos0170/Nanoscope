"""Incubation curation uses the real native event workspace and saved study."""
from copy import deepcopy
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanoscope.analysis import acid_incubation as engine
from nanoscope.analysis import manual_workspace as checkpoint
from nanoscope.analysis.cancel import check_cancelled
from nanoscope.analysis.incubation_detection import _fingerprint
from nanoscope.gui.acid_study_window import AcidStudyWindow
from nanoscope.gui.main_window import MainWindow


@pytest.fixture
def windows(qtbot, monkeypatch, tmp_path):
    main = MainWindow(recovery_enabled=False)
    qtbot.addWidget(main)
    monkeypatch.setattr(main, '_confirm_close_choice', lambda: 'discard')
    study = AcidStudyWindow(main)
    qtbot.addWidget(study)
    main._acid_study_window = study
    study.new_workspace(tmp_path/'study')
    source = tmp_path/'source.abf'; source.write_bytes(b'acid source fixture')
    source_hash = _fingerprint(source)['sha256']
    state = engine.new_state(records=[dict(tag='acid_30h', source_path=source,
        source_info=dict(sha256=source_hash, fs_hz=50000., channels=[dict(units='pA')]),
        channel=0, incubation_h=30., recording_id='acid-record-1', comparison_only=False)])
    signal = np.random.default_rng(43).normal(0, .2, 10000).astype(np.float32)
    signal[1000:1040] -= 10.; signal[2000:2070] -= 15.
    path = tmp_path/'prepared.parquet'
    pq.write_table(pa.table(dict(current_detr_pA=signal, current_raw_pA=signal-200.)), path)
    row = state['records'][0]
    row['prepared'] = dict(parquet_path=path, sha256=_fingerprint(path)['sha256'],
        source_sha256=source_hash,
        settings_sha256=engine._digest({key:state['settings'][key] for key in engine.PREPARE_KEYS}),
        metadata=dict(fs_hz=50000., n_samples=len(signal), event_polarity=1))
    row['detection'] = dict(primary_set='cusum', catalog_detector='cusum',
        settings_sha256='original', context_path=tmp_path/'detection-context')
    row['catalog'] = pd.DataFrame(dict(event_id=['acid:a', 'acid:b'],
        start_idx=[1000, 2000], end_idx=[1039, 2069], accepted=[True, True],
        event_review=['unreviewed', 'unreviewed'], polarity=[1, 1], passes_gate=[True, True]))
    state['workspace_kind'] = 'acid_incubation_v1'
    study.state = state
    study._revision = checkpoint.save_workspace(state, study.folder, expected_revision=study._revision)
    study.refresh()
    yield main, study
    study.abort()
    main._dirty = False
    main._teardown()


def complete(qtbot, study, action):
    failures = []
    study.operationFailed.connect(failures.append)
    try:
        assert action() is not False
        qtbot.waitUntil(lambda: not study.is_busy(), timeout=30000)
        assert not failures, failures
    finally:
        study.operationFailed.disconnect(failures.append)


def opened(qtbot, windows):
    main, study = windows
    complete(qtbot, study, study.open_main_curation)
    return main, study, main._analysis_panel.current_detection()


def test_primary_review_uses_shared_events_and_saves_acceptance(qtbot, windows):
    main, study, result = opened(qtbot, windows)
    assert study._main_curation_button.text() == 'Curate in main workspace…'
    assert main._event_gallery.frame() is result.payload.df
    assert not main._study_curation_bar.isHidden()
    assert main._study_curation_save.isEnabled()
    assert 'acid_30h' in main._study_curation_label.text()
    main._analysis_panel.set_event_accepted(0, False)
    assert not result.payload.df.accepted.iloc[0]
    assert study.state['records'][0]['catalog'].accepted.all()
    assert 'Unsaved' in main._study_curation_label.text()
    complete(qtbot, study, main._save_study_curation)
    row = study.state['records'][0]
    assert row['catalog'].accepted.tolist() == [False, True]
    assert not engine.selection_is_current(row)
    restored = checkpoint.load_workspace(study.folder)
    assert restored['records'][0]['catalog'].accepted.tolist() == [False, True]
    assert 'confirm the selection' in main._study_curation_label.text()
    main._analysis_panel.set_event_accepted(1, False)
    complete(qtbot, study, main._save_study_curation)
    assert not study.state['records'][0]['catalog'].accepted.any()


def test_return_and_reenter_preserve_live_edits_without_duplicate(qtbot, windows):
    main, study, result = opened(qtbot, windows)
    main._analysis_panel.set_event_accepted(0, False)
    main._return_to_study()
    assert study.isVisible()
    assert study._stages.currentIndex() == 3
    assert 'unsaved changes' in study._status.text()
    complete(qtbot, study, study.open_main_curation)
    assert len(main._session.documents()) == 1
    assert main._analysis_panel.current_detection() is result
    assert not result.payload.df.accepted.iloc[0]
    assert main._study_curation_save.isEnabled()


def test_save_targets_captured_study_record_after_study_selection_changes(qtbot, windows):
    main, study, result = opened(qtbot, windows)
    other = deepcopy(study.state['records'][0])
    other.update(tag='acid_other', recording_id='acid-record-2')
    other.pop('main_workspace_bridge', None)
    study.state['records'].append(other)
    study._active_tag = 'acid_other'; study.refresh()
    main._analysis_panel.set_event_accepted(0, False)
    complete(qtbot, study, main._save_study_curation)
    assert study.state['records'][0]['catalog'].accepted.tolist() == [False, True]
    assert study.state['records'][1]['catalog'].accepted.tolist() == [True, True]
    main._return_to_study()
    assert study._active_tag == 'acid_30h'


def test_recording_switch_blocks_wrong_save_and_show_events_refocuses(qtbot, windows, tmp_path):
    from nanoscope.gui.acid_event_bridge import prepare_review_export
    main, study, result = opened(qtbot, windows)
    main.open_study_review(prepare_review_export(study.state, 'acid_30h', tmp_path/'other-review'))
    assert not main._study_curation_save.isEnabled()
    assert not main._study_curation_show.isHidden()
    with pytest.raises(ValueError, match='Show this study'):
        main._save_study_curation()
    main.show_study_curation()
    assert main._analysis_panel.current_detection() is result
    assert main._study_curation_save.isEnabled()


def test_closed_or_replaced_study_cannot_receive_wrong_commit(qtbot, windows, tmp_path):
    main, study, result = opened(qtbot, windows)
    study.show(); study.close()
    assert not main._study_curation_save.isEnabled()
    main._return_to_study()
    assert main._study_curation_save.isEnabled()
    study.new_workspace(tmp_path/'another-study')
    with pytest.raises(ValueError, match='workspace changed'):
        main._save_study_curation()
    assert study.state['records'] == []


def test_cancelled_save_keeps_previous_checkpoint_and_native_edits(qtbot, windows, monkeypatch):
    from nanoscope.gui import acid_event_bridge as bridge
    main, study, result = opened(qtbot, windows)
    main._analysis_panel.set_event_accepted(0, False)
    before = study._revision
    runs = set((study.folder/'runs').iterdir())
    def slow(state, tag, snapshot, folder, progress=None, cancel=None):
        (folder/'incomplete').write_text('partial')
        while True:
            check_cancelled(cancel); time.sleep(.005)
    monkeypatch.setattr(bridge, 'apply_review', slow)
    main._save_study_curation(); qtbot.wait(30); study.cancel()
    qtbot.waitUntil(lambda: not study.is_busy(), timeout=10000)
    assert study._revision == before
    assert checkpoint.workspace_revision(study.folder) == before
    assert set((study.folder/'runs').iterdir()) == runs
    assert study.state['records'][0]['catalog'].accepted.all()
    assert not result.payload.df.accepted.iloc[0]
    assert main._study_curation_save.isEnabled()


def test_project_reset_during_open_does_not_install_late_review(qtbot, windows, monkeypatch):
    from nanoscope.gui import acid_event_bridge as bridge
    main, study = windows
    original = bridge.prepare_review_export
    started = threading.Event(); release = threading.Event()
    def paused(*args, **kwargs):
        payload = original(*args, **kwargs)
        started.set()
        assert release.wait(10)
        return payload
    monkeypatch.setattr(bridge, 'prepare_review_export', paused)
    failures = []; study.operationFailed.connect(failures.append)
    study.open_main_curation()
    qtbot.waitUntil(started.is_set, timeout=30000)
    main._clear_session(); release.set()
    qtbot.waitUntil(lambda: not study.is_busy(), timeout=30000)
    assert any('main project changed' in message for message in failures)
    assert not main._session.documents()
    assert main._study_curation_bar.isHidden()


def test_review_sidecar_is_mutable_without_breaking_study_checkpoint(qtbot, windows):
    main, study, result = opened(qtbot, windows)
    path = Path(study.state['records'][0]['main_workspace_bridge']['path'])
    sidecar = path.with_name(path.name+'.npcal.json')
    sidecar.write_text(sidecar.read_text()+'\n')
    restored = checkpoint.load_workspace(study.folder)
    assert restored['records'][0]['main_workspace_bridge']['path'] == str(path)


def test_later_background_edit_remains_pending_after_captured_save(qtbot, windows, tmp_path, monkeypatch):
    from nanoscope.gui import acid_event_bridge as bridge
    main, study, result = opened(qtbot, windows)
    main._analysis_panel.set_event_accepted(0, False)
    original = bridge.apply_review
    started = threading.Event(); release = threading.Event()
    def paused(*args, **kwargs):
        started.set(); assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(bridge, 'apply_review', paused)
    main._save_study_curation()
    qtbot.waitUntil(started.is_set, timeout=10000)
    # A background native result can change A after navigation to recording B.
    main.open_study_review(bridge.prepare_review_export(study.state, 'acid_30h', tmp_path/'other-review'))
    result.payload.df.loc[result.payload.df.event_id == 1, 'accepted'] = False
    main._checkpoint_everything()
    release.set()
    qtbot.waitUntil(lambda: not study.is_busy(), timeout=30000)
    assert study.state['records'][0]['catalog'].accepted.tolist() == [False, True]
    main.show_study_curation()
    assert main._study_curation_context['pending']
    assert 'Unsaved' in main._study_curation_label.text()
    complete(qtbot, study, main._save_study_curation)
    assert not study.state['records'][0]['catalog'].accepted.any()


def test_adopting_existing_project_review_discloses_pending_curation(qtbot, windows):
    main, study, result = opened(qtbot, windows)
    main._analysis_panel.set_event_accepted(0, False)
    main._study_curation_sessions.clear(); main._study_curation_context = None
    complete(qtbot, study, study.open_main_curation)
    assert len(main._session.documents()) == 1
    assert main._study_curation_context['pending']
    main._return_to_study()
    assert 'unsaved changes' in study._status.text()


def test_quick_study_review_expires_main_copy_and_opens_current_events(qtbot, windows):
    main, study, result = opened(qtbot, windows)
    complete(qtbot, study, lambda: study._stage('Quick acceptance',
        lambda state, folder, progress, cancel: engine.set_event_acceptance(state, 'acid_30h', ['acid:a'], False)))
    assert not main._study_curation_save.isEnabled()
    assert 'study changed' in main._study_curation_label.text()
    with pytest.raises(ValueError, match='study changed'):
        main._save_study_curation()
    main._return_to_study()
    assert study.isVisible()
    complete(qtbot, study, study.open_main_curation)
    assert len(main._session.documents()) == 2
    assert main._analysis_panel.current_detection() is not result
    assert main._analysis_panel.current_detection().payload.df.accepted.tolist() == [False, True]
    assert main._study_curation_save.isEnabled()
