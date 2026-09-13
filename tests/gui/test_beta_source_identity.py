"""Independent save/restore identity checks; synthetic inputs, bounded GUI seams."""
import numpy as np
import pandas as pd

from nanoscope.analysis.events import EventSet, SCHEMA
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.identity import fingerprint_source
from nanoscope.core.project import DocumentRef, ProjectState, ResultRec
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow


def recording(path, n=1000):
    signal = np.arange(n, dtype=np.float32)
    pd.DataFrame({'time': np.arange(n) / 50000., 'current': signal}).to_csv(path, index=False)
    return TraceDocument(RecordingMeta(path=path, format_name='synthetic', n_samples=n,
        sample_rate_hz=50000., rate_provenance=Provenance.FILE_META,
        channels=(ChannelInfo(0, 'current', 'pA', ChannelKind.CURRENT),)), (signal,))


def events():
    row = {name: (False if dtype == 'bool' else '' if dtype == 'string' else 0)
           for name, (dtype, _, _) in SCHEMA.items()}
    row.update(event_id=0, start_idx=20, end_idx=40, accepted=True, depth_pA=12., dwell_s=.0004)
    return pd.DataFrame([row]).astype({name: spec[0] for name, spec in SCHEMA.items()})


def window_for(qtbot, doc):
    window = MainWindow()
    qtbot.addWidget(window)
    return window, window._session.add_document(doc)


def test_unknown_restored_identity_does_not_become_known_on_save(qtbot, tmp_path):
    path = tmp_path / 'source.csv'
    doc = recording(path)
    state = ProjectState(documents=[DocumentRef(str(path))], results=[
        ResultRec(str(path), 0, 'detect.schmitt', events())])
    window, doc_id = window_for(qtbot, doc)
    window._restore_state = state
    window._restore_sources[doc.source_key] = None
    window._apply_restore_for_doc(doc_id)
    saved = window._gather_project_state()
    window._restore_state = None
    window._dirty = False
    assert saved.results[0].source_fingerprint is None or not saved.results[0].source_fingerprint.digest


def test_carried_missing_channel_retains_original_document_evidence(qtbot, tmp_path):
    path = tmp_path / 'source.csv'
    recording(path)
    original = fingerprint_source(path, 1000)
    state = ProjectState(documents=[DocumentRef(str(path), source=original)], results=[
        ResultRec(str(path), 3, 'detect.schmitt', events(),
                  channel_name='missing_current', channel_kind='current')])
    doc = recording(path, 1500)
    window, doc_id = window_for(qtbot, doc)
    window._restore_state = state
    window._restore_sources[doc.source_key] = original
    window._apply_restore_for_doc(doc_id)
    window._capture_unrestorable(state)
    assert window._carried_over['results']
    saved = window._gather_project_state()
    window._restore_state = None
    window._dirty = False
    assert saved.documents[0].source.digest == original.digest


def test_live_unchanged_result_cannot_be_restamped_after_disk_replacement(qtbot, tmp_path):
    path = tmp_path / 'source.csv'
    doc = recording(path)
    window, doc_id = window_for(qtbot, doc)
    window._session.add_result(AnalysisResult(
        module_id='detect.schmitt', module_version='test', doc_id=doc_id,
        channel=0, params={}, calibration_fingerprint=(1., 0.), payload=EventSet(events())))
    first = window._gather_project_state().results[0].source_fingerprint
    recording(path, 1500)
    second = window._gather_project_state().results[0].source_fingerprint
    window._dirty = False
    assert first is not None and second == first
