"""Regression coverage for beta save lockout, source laundering and job replacement."""
from dataclasses import replace
import weakref

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA
from nanoscope.core.acquisition_state import WORKSPACE_KEY, UNREADABLE_KEY
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.identity import fingerprint_source
from nanoscope.core.project import (
    DocumentRef, ProjectState, ResultRec, load_project, save_project,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow


def make_window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def recording(path, n=1000):
    x = np.full(n, 100., dtype=np.float32)
    pd.DataFrame({'time': np.arange(n) / 50000., 'current': x}).to_csv(path, index=False)
    return TraceDocument(
        RecordingMeta(path=path, format_name='synthetic', n_samples=n,
                      sample_rate_hz=50000., rate_provenance=Provenance.FILE_META,
                      channels=(ChannelInfo(0, 'current', 'pA', ChannelKind.CURRENT),)),
        (x,))


@pytest.mark.parametrize('damaged', [None, {}, 'unknown', 3])
def test_unreadable_acquisition_section_keeps_evidence_and_allows_saving(qtbot, tmp_path, monkeypatch, damaged):
    window = make_window(qtbot)
    path = tmp_path / 'source.csv'
    recording(path)
    state = ProjectState(documents=[DocumentRef(str(path))])
    state.workspace.extra[WORKSPACE_KEY] = damaged
    state.workspace.extra[UNREADABLE_KEY] = {'older evidence': True}
    with qtbot.waitSignal(window.projectRestored, timeout=10000):
        window._begin_restore(state)
    assert len(window._session.documents()) == 1
    window._mark_dirty()
    snapshot = window._capture_recovery_snapshot(window._session_revision)
    assert snapshot is not None
    assert snapshot.state.workspace.extra[WORKSPACE_KEY] == []
    first = tmp_path / 'first.nanoscope'
    second = tmp_path / 'save-as.nanoscope'
    window._project_path = str(first)
    assert window._save_project()
    window._mark_dirty()
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *a, **k: (str(second), ''))
    assert window._save_project_as()
    recovered = window._gather_project_state()
    assert recovered.workspace.extra[WORKSPACE_KEY] == []
    for path in (first, second):
        saved = load_project(path)
        assert saved.workspace.extra[UNREADABLE_KEY] == {'older evidence': True}
        assert saved.workspace.extra[UNREADABLE_KEY + '_1'] == damaged
    window._dirty = False


def test_source_mismatch_survives_save_and_clears_only_when_results_replaced(qtbot, tmp_path):
    path = tmp_path / 'source.csv'
    recording(path)
    original = fingerprint_source(path, 1000)
    row = {name: (False if dtype == 'bool' else '' if dtype == 'string' else 0)
           for name, (dtype, _, _) in SCHEMA.items()}
    row.update(event_id=0, start_idx=20, end_idx=40, accepted=True,
               depth_pA=12., dwell_s=.0004)
    frame = pd.DataFrame([row]).astype({name: spec[0] for name, spec in SCHEMA.items()})
    state = ProjectState(documents=[DocumentRef(str(path), source=original)], results=[
        ResultRec(str(path), 0, 'detect.schmitt', frame)])
    changed = recording(path, 1500)
    current = fingerprint_source(path, 1500)
    window = make_window(qtbot)
    doc_id = window._session.add_document(changed)
    window._restore_state = state
    window._restore_sources[changed.source_key] = original
    window._apply_restore_for_doc(doc_id)
    result = window._session.results.get(doc_id, 0, 'detect.schmitt')
    assert result is not None
    assert result.source_fingerprint == original
    assert window._restore_notes
    saved_path = save_project(tmp_path / 'changed.nanoscope', window._gather_project_state())
    saved = load_project(saved_path)
    assert saved.documents[0].source == original
    assert saved.results[0].source_fingerprint == original
    pd.testing.assert_frame_equal(saved.results[0].events, result.payload.df)

    reopened = make_window(qtbot)
    reopened_id = reopened._session.add_document(recording(path, 1500))
    reopened._restore_state = saved
    reopened._restore_sources[str(path)] = saved.documents[0].source
    reopened._apply_restore_for_doc(reopened_id)
    assert any('not the file' in note for note in reopened._restore_notes)

    # A genuine new analysis result has no inherited source stamp. Replacing the
    # old result removes the last reason to retain that historical document stamp.
    window._session.add_result(replace(result, source_fingerprint=None, source_fingerprint_recorded=False))
    fresh = window._gather_project_state()
    assert fresh.documents[0].source.digest == current.digest
    assert fresh.documents[0].source.n_samples == 1500
    assert fresh.results[0].source_fingerprint.digest == current.digest
    for w in (window, reopened):
        w._restore_state = None
        w._dirty = False


def test_replacement_releases_subpopulation_input_and_clears_stale_fit_status(qtbot):
    window = make_window(qtbot)
    signal = np.zeros(10000)
    reference = weakref.ref(signal)
    window._subpop_signal_pending = signal
    window._subpop_target = object()
    window._subpop_accepted_ids_pending = frozenset({1})
    window._subpop_source_recipe_pending = {'source': 'old'}
    window._level_fit_pending = {'token': object()}
    window._active_analysis_label = 'fit.levels'
    window.statusBar().showMessage('Fitting sub-state levels', 0)
    del signal
    window._on_analysis_started('detect.schmitt')
    assert reference() is None
    assert not window._subpop_pending
    assert window._subpop_accepted_ids_pending is None
    assert window._subpop_source_recipe_pending is None
    assert window._level_fit_pending is None
    assert 'Replaced fit.levels with detect.schmitt' in window.statusBar().currentMessage()
    window._on_analysis_finished(None)
    assert window._active_analysis_label is None


def test_load_completion_failure_does_not_strand_queue(qtbot, tmp_path, monkeypatch):
    window = make_window(qtbot)
    doc = recording(tmp_path / 'recording.csv')
    window._active_open = doc.meta.path
    errors = []
    def failed(message):
        errors.append(message)
        window._finish_active()
    monkeypatch.setattr(window, '_on_failed', failed)
    def refuse(_doc):
        raise ValueError('damaged optional metadata')
    monkeypatch.setattr(window._session, 'add_document', refuse)
    window._on_loaded(doc)
    assert window._active_open is None
    assert 'damaged optional metadata' in errors[0]


def test_shared_cancel_discloses_both_jobs_and_progress_is_indeterminate(qtbot, tmp_path, monkeypatch):
    window = make_window(qtbot)
    window._active_open = tmp_path / 'second.csv'
    window._active_analysis_label = 'detect.schmitt'
    for controller in (window._controller, window._analysis_controller):
        monkeypatch.setattr(controller, 'is_busy', lambda: True)
        monkeypatch.setattr(controller, 'abort_stopped', lambda: True)
    window._on_progress(.5, 'Detecting candidates')
    assert window._progress.maximum() == 0
    assert 'second.csv' in window._progress.format()
    assert 'detect.schmitt' in window._progress.format()
    assert 'second.csv' in window._cancel_btn.toolTip()
    window._cancel_load()
    message = window.statusBar().currentMessage()
    assert 'second.csv' in message and 'detect.schmitt' in message
    assert 'Cancelled.' in message
    monkeypatch.setattr(window._controller, 'is_busy', lambda: False)
    window._on_progress(.5, 'Detecting candidates')
    assert window._progress.maximum() == 100
