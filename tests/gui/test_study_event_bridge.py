"""The study uses the app's actual shared curation and persisted source contract."""
from copy import deepcopy
from dataclasses import replace
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.incubation_detection import PreparedRecording, RecordingAnalysis, _fingerprint
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.session import Session
from nanoscope.core.types import SampleRange
from nanoscope.core.sources import discover_plot_sources
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.study_event_bridge import MODULE_ID, prepare_review_export
from nanoscope.io.hdf5 import Hdf5Reader


@pytest.fixture
def window(qtbot, monkeypatch):
    win = MainWindow(recovery_enabled=False)
    qtbot.addWidget(win)
    monkeypatch.setattr(win, "_confirm_close_choice", lambda: "discard")
    yield win
    win._dirty = False
    win._teardown()


def record(tmp_path, *, events=True):
    signal = np.random.default_rng(73).normal(0, .2, 10000).astype(np.float32)
    signal[1000:1040] -= 10
    signal[2000:2070] -= 15
    raw = np.asarray(signal - 200, np.float64)
    source = tmp_path / 'source.abf'
    source.write_bytes(b'test original source')
    prepared = PreparedRecording('test', raw, signal, 50000., len(signal) / 50000.,
        dict(event_polarity=1), dict(raw_file=_fingerprint(source), event_polarity=1))
    row = dict(recording_id='record-1', tag='test', source_path=source,
        channel=0, pore_id='pore-1', prepared=prepared)
    if events:
        catalog = pd.DataFrame(dict(event_id=['test:0', 'test:1'],
            start_idx=[1000, 2000], end_idx=[1039, 2069],
            depth_peak_pA=[10., 15.], I0_local_pA=[200., 200.],
            passes_gate=[True, True], accepted=[True, False]))
        row['analysis'] = RecordingAnalysis('test', raw, signal, 50000., .2, catalog,
            dict(prepared.provenance))
    return row


def opened(window, tmp_path, *, events=True):
    row = record(tmp_path, events=events)
    payload = prepare_review_export(row, tmp_path / 'review')
    row['main_workspace_bridge'] = payload['descriptor']
    window.open_study_review(payload)
    return row, payload


def test_export_is_reopenable_exact_signal_and_units_without_original_sidecar(tmp_path):
    row = record(tmp_path)
    payload = prepare_review_export(row, tmp_path / 'review')
    doc = Hdf5Reader(payload['document'].meta.path).load()
    assert doc.meta.path != row['source_path']
    assert np.array_equal(doc.raw(0), row['prepared'].detrended)
    assert np.isclose(doc.meta.sample_rate_hz, row['prepared'].fs_hz, rtol=1e-12)
    session = Session(); session.add_document(doc)
    assert session.calibrated(doc.doc_id, 0).units == 'pA'
    assert session.calibration(doc.doc_id, 0).fingerprint() == (1., 0.)
    native = payload['result'].payload.df
    assert native.event_id.tolist() == [0, 1]
    assert native.end_idx.tolist() == [1040, 2070]
    assert native.accepted.tolist() == [True, False]
    assert abs(native.baseline_pA.iloc[0]) < .1
    assert native.study_I0_local_pA.tolist() == [200., 200.]
    assert native.frac_depth.isna().all()
    assert native.source.tolist() == [MODULE_ID, MODULE_ID]
    assert native.review.tolist() == ['unreviewed', 'reviewed']
    assert not row['source_path'].with_suffix('.abf.npcal.json').exists()


def test_native_table_gallery_inspector_edit_and_snapshot_are_shared(window, tmp_path):
    row, payload = opened(window, tmp_path)
    result = window._analysis_panel.current_detection()
    assert result.module_id == MODULE_ID
    assert window._event_gallery.frame() is result.payload.df
    window._analysis_panel.set_event_accepted(0, False)
    window._open_event_inspector()
    inspector = window._event_inspector
    inspector.show_event(1)
    inspector._apply_edited_bounds(1, 1995, 2075, payload['document'].raw(0))
    enabled, tip = inspector._revert_state(1)
    assert enabled and 'the imported study candidate table' in tip
    snapshot = window.snapshot_study_events(row)
    assert snapshot['events'].accepted.tolist() == [False, False]
    assert snapshot['events'].start_idx.tolist() == [1000, 1995]
    assert snapshot['events'].end_idx.tolist() == [1040, 2075]
    assert snapshot['events'].frac_depth.isna().all()
    assert snapshot['provenance']['source_module_id'] == MODULE_ID
    assert snapshot['provenance']['source_kind'] == 'prepared_review'
    assert row['analysis'].catalog.start_idx.tolist() == [1000, 2000]
    snapshot['events'].loc[0, 'accepted'] = True
    assert not result.payload.df.accepted.iloc[0]


def test_prepared_source_can_enter_main_tools_before_any_inference(window, tmp_path):
    row, _ = opened(window, tmp_path, events=False)
    assert len(window._analysis_panel.current_detection().payload.df) == 0
    assert window._act_run_detection.isEnabled()
    snapshot = window.snapshot_study_events(row)
    assert snapshot['events'].empty
    assert 'inference' not in row and 'analysis' not in row


def test_prepared_only_user_can_run_actual_schmitt_then_curate_and_return(window, tmp_path, qtbot):
    row, _ = opened(window, tmp_path, events=False)
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(.0002)
    refused = []; panel.runRefused.connect(refused.append)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30000):
        panel.trigger_detection()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=3000)
    result = panel.current_detection()
    assert not refused
    assert result.module_id == 'detect.schmitt'
    assert len(result.payload.df) >= 2
    event_id = int(result.payload.df.event_id.iloc[0])
    panel.set_event_accepted(event_id, False)
    snapshot = window.snapshot_study_events(row)
    assert snapshot['provenance']['source_module_id'] == 'detect.schmitt'
    assert not snapshot['events'].loc[snapshot['events'].event_id == event_id, 'accepted'].iloc[0]
    assert 'inference' not in row


@pytest.mark.parametrize('mutation, message', [
    ('source', 'study changed'), ('calibration', 'calibration'),
    ('exclusion', 'full exposure'), ('range', 'full exposure'),
    ('duration', 'full exposure'), ('busy', 'finish first'),
    ('malformed_range', 'full exposure'), ('rate', 'full exposure'),
    ('unknown_history', 'no recorded source evidence'),
    ('polarity', 'polarity'),
])
def test_refuses_wrong_source_calibration_exposure_and_inflight_edits(window, tmp_path, mutation, message):
    row, _ = opened(window, tmp_path)
    result = window._analysis_panel.current_detection()
    if mutation == 'source':
        row['prepared'].detrended = row['prepared'].detrended.copy()
        row['prepared'].detrended[100] += 1
    elif mutation == 'calibration':
        window._session.set_calibration(result.doc_id, 0, ChannelCalibration(gain=2, target_units='pA'), write_sidecar=False)
    elif mutation == 'exclusion': result.payload.provenance['exclude_regions'] = [(0, .01)]
    elif mutation == 'range': result.payload.provenance['analysed_start_idx'] = 2
    elif mutation == 'duration': result.payload.provenance['analyzed_duration_s'] = .1
    elif mutation == 'malformed_range': result.payload.provenance['analysed_start_idx'] = .5
    elif mutation == 'rate': result.payload.provenance['fs'] = float('nan')
    elif mutation == 'unknown_history':
        window._session.add_result(replace(result, source_fingerprint=None, source_fingerprint_recorded=True))
    elif mutation == 'polarity':
        # This fixture's record declares event_polarity=1 (downward native); the
        # native/imported table is always characterized events-down (polarity=-1),
        # so the two agree by default. Flip it to simulate importing a detection run
        # with the wrong polarity for this recording's declared orientation.
        result.payload.provenance['polarity'] = 1
    else: window._analysis_panel._merge_job = ('test',)
    try:
        with pytest.raises(ValueError, match=message): window.snapshot_study_events(row)
    finally:
        if mutation == 'busy': window._analysis_panel._merge_job = None


def test_rerun_never_reinterprets_imported_table_as_schmitt(window, tmp_path, monkeypatch):
    opened(window, tmp_path)
    panel = window._analysis_panel
    refused = []; panel.runRefused.connect(refused.append)
    monkeypatch.setattr(panel, '_on_detect_requested', lambda *a: pytest.fail('wrong detector ran'))
    panel._on_rerun()
    assert refused and 'imported' in refused[-1]


def test_independent_review_exports_preserve_existing_open_results(window, tmp_path):
    row, first = opened(window, tmp_path)
    window._analysis_panel.set_event_accepted(0, False)
    second = prepare_review_export(row, tmp_path / 'second-review')
    window.open_study_review(second)
    previous = window._session.results.get(first['document'].doc_id, 0, MODULE_ID)
    assert not previous.payload.df.accepted.iloc[0]
    assert window._session.has_document(first['document'].doc_id)
    assert len(window._session.documents()) == 2


def test_project_save_reopen_preserves_review_signal_labels_and_units_sidecar(window, tmp_path, qtbot):
    row, payload = opened(window, tmp_path)
    doc = payload['document']; sidecar = doc.meta.path.with_suffix('.h5.npcal.json')
    # Use the actual generated sidecar name rather than assuming a second codec.
    from nanoscope.core.calibration import sidecar_path
    sidecar = sidecar_path(doc.meta.path)
    before = sidecar.read_bytes()
    window._analysis_panel.set_event_accepted(0, False)
    path = tmp_path / 'curated.nanoscope'
    assert window._write_project(str(path))
    assert sidecar.read_bytes() == before
    other = MainWindow(recovery_enabled=False); qtbot.addWidget(other)
    other._confirm_close_choice = lambda: 'discard'
    try:
        other._open_project(str(path))
        qtbot.waitUntil(lambda: not other._restoring and len(other._session.documents()) == 1, timeout=10000)
        reopened = other._session.documents()[0]
        other._on_selection(reopened.doc_id, 0)
        assert reopened.doc_id != doc.doc_id
        assert np.array_equal(reopened.raw(0), row['prepared'].detrended)
        assert other._session.calibrated(reopened.doc_id, 0).units == 'pA'
        snapshot = other.snapshot_study_events(row)
        assert snapshot['events'].accepted.tolist() == [False, False]
        assert sidecar.read_bytes() == before
    finally:
        other._dirty = False; other._teardown()


def test_review_labels_and_original_native_ids_survive_full_return_export(window, tmp_path):
    from nanoscope.analysis.manual_detection import measure_imported_events, MeasurementSettings
    row, _ = opened(window, tmp_path)
    result = window._analysis_panel.current_detection()
    result.payload.df['event_id'] = [7, 42]
    result.payload.df.loc[0, 'flag'] = 'molecule=insulin;flagged'
    result.payload.df.loc[0, 'review'] = 'reviewed'
    result.payload.df.loc[0, 'source'] = 'detect.keras'
    result.payload.df.loc[1, 'source'] = 'manual'
    result.payload.df['added'] = [False, True]
    snapshot = window.snapshot_study_events(row)
    row['analysis'] = measure_imported_events(row['prepared'], snapshot['events'],
        MeasurementSettings(), source_provenance=snapshot['provenance'])
    exported = prepare_review_export(row, tmp_path / 'returned-review')['result'].payload.df
    assert exported.event_id.tolist() == [7, 42]
    assert exported.source.tolist() == ['detect.keras', 'manual']
    assert exported.review.tolist() == ['reviewed', 'reviewed']
    assert 'molecule=insulin' in exported.flag.iloc[0] and 'flagged' in exported.flag.iloc[0]
    assert exported.added.tolist() == [False, True]


def test_actual_cu_schmitt_rows_keep_conventional_origin_in_training_provenance(tmp_path):
    from nanoscope.analysis.study_signal import analyze, SchmittSettings
    from nanoscope.analysis.events import source_kind
    from nanoscope.analysis.scoring import curation_stamp
    from nanoscope.core.sources import detector_name
    from nanoscope.gui.study_event_bridge import SCHMITT_SOURCE_ID
    row = record(tmp_path)
    row['analysis'] = analyze(row['prepared'], SchmittSettings())
    assert len(row['analysis'].catalog) == 2
    native = prepare_review_export(row, tmp_path / 'cu-review')['result'].payload.df
    assert native.source.tolist() == [SCHMITT_SOURCE_ID] * 2
    assert native.source.map(source_kind).tolist() == ['conventional'] * 2
    assert detector_name(SCHMITT_SOURCE_ID) == 'Study Schmitt'
    stamp = curation_stamp(native, source_df=native)
    assert stamp['curation']['by_kind'] == {'conventional': 2}
    # A student record keeps the existing model-origin source token.
    student = prepare_review_export(record(tmp_path), tmp_path / 'student-review')['result'].payload.df
    assert student.source.map(source_kind).tolist() == ['ml'] * 2


def test_explicit_study_review_survives_import_remeasure_save_and_export(window, tmp_path):
    from nanoscope.analysis.manual_detection import measure_imported_events, MeasurementSettings
    from nanoscope.analysis.study_review import apply_event_decision, retain_event_decisions
    from nanoscope.analysis.manual_workspace import save_workspace, load_workspace
    from nanoscope.analysis.scoring import build_training_frame
    row = record(tmp_path)
    row['analysis'].catalog['accepted'] = True
    payload = prepare_review_export(row, tmp_path / 'review')
    row['main_workspace_bridge'] = payload['descriptor']
    window.open_study_review(payload)
    snapshot = window.snapshot_study_events(row)
    imported = measure_imported_events(row['prepared'], snapshot['events'],
        MeasurementSettings(), source_provenance=snapshot['provenance'])
    assert imported.catalog.source_review.tolist() == ['unreviewed', 'unreviewed']
    selected_id = imported.catalog.event_id.iloc[0]
    reviewed = apply_event_decision(imported, [selected_id], False)
    reviewed = apply_event_decision(reviewed, [selected_id], True)
    measured = measure_imported_events(row['prepared'], snapshot['events'],
        MeasurementSettings(min_depth_sigma=7), source_provenance=snapshot['provenance'])
    row['analysis'] = retain_event_decisions(reviewed, measured)
    save_workspace({'record': row}, tmp_path / 'checkpoint')
    restored = load_workspace(tmp_path / 'checkpoint')['record']
    native = prepare_review_export(restored, tmp_path / 'returned-review')['result'].payload.df
    assert native.accepted.tolist() == [True, True]
    assert native.review.tolist() == ['reviewed', 'unreviewed']
    trained = build_training_frame(native, selection='reviewed')
    assert trained.start_idx.tolist() == [int(native.start_idx.iloc[0])]
    assert trained.accepted.tolist() == [True]
    # Archive values remain provenance, not the current attention authority.
    assert restored['analysis'].catalog.source_review.tolist() == ['unreviewed'] * 2


def test_fresh_study_noop_accept_and_reset_do_not_manufacture_review(tmp_path):
    from nanoscope.analysis.study_review import apply_event_decision
    from nanoscope.analysis.scoring import build_training_frame
    row = record(tmp_path)
    row['analysis'].catalog['accepted'] = [True, False]
    row['analysis'].catalog['passes_gate'] = [True, False]
    eid = row['analysis'].catalog.event_id.iloc[0]
    row['analysis'] = apply_event_decision(row['analysis'], [eid], True)
    native = prepare_review_export(row, tmp_path / 'accepted')['result'].payload.df
    assert native.review.tolist() == ['reviewed', 'unreviewed']
    assert build_training_frame(native, selection='reviewed').start_idx.tolist() == [int(native.start_idx.iloc[0])]
    row['analysis'] = apply_event_decision(row['analysis'], [], None)
    reset = prepare_review_export(row, tmp_path / 'reset')['result'].payload.df
    assert reset.accepted.tolist() == [True, False]
    assert reset.review.tolist() == ['unreviewed', 'unreviewed']
    assert build_training_frame(reset, selection='reviewed').empty


def test_gate_negative_stays_automatic_through_native_remeasurement_and_return(tmp_path):
    from nanoscope.analysis.baseline import BaselineConfig
    from nanoscope.analysis.remeasure import remeasure_events
    from nanoscope.analysis.manual_detection import measure_imported_events, MeasurementSettings
    from nanoscope.analysis.study_review import apply_event_decision
    from nanoscope.analysis.events import event_status, STUDY_REVIEW_POLICY_COL, STUDY_REVIEW_POLICY
    from nanoscope.analysis.scoring import build_training_frame
    row = record(tmp_path)
    row['analysis'].catalog['accepted'] = [True, False]
    row['analysis'].catalog['passes_gate'] = [True, False]
    payload = prepare_review_export(row, tmp_path / 'automatic')
    native = payload['result'].payload
    assert event_status(native.df).tolist() == ['unreviewed', 'auto-rejected']
    rebuilt = remeasure_events(native, payload['document'].raw(0), payload['document'].meta.sample_rate_hz,
        config=BaselineConfig(method='constant'), polarity=-1).events
    assert rebuilt.df[STUDY_REVIEW_POLICY_COL].tolist() == [STUDY_REVIEW_POLICY] * 2
    assert rebuilt.df.accepted.tolist() == [True, False]
    assert build_training_frame(rebuilt.df, selection='reviewed').empty
    row['analysis'] = measure_imported_events(row['prepared'], rebuilt.df, MeasurementSettings(), source_provenance={})
    returned = prepare_review_export(row, tmp_path / 'untouched-return')['result'].payload.df
    assert event_status(returned).tolist() == ['unreviewed', 'auto-rejected']
    rejected_id = row['analysis'].catalog.event_id.iloc[1]
    row['analysis'] = apply_event_decision(row['analysis'], [rejected_id], False)
    explicit = prepare_review_export(row, tmp_path / 'explicit-rejection')['result'].payload.df
    assert event_status(explicit).tolist() == ['unreviewed', 'rejected']
    assert build_training_frame(explicit, selection='reviewed').start_idx.tolist() == [int(explicit.start_idx.iloc[1])]


def test_imported_legacy_human_negative_keeps_training_evidence(tmp_path):
    from nanoscope.analysis.manual_detection import measure_imported_events, MeasurementSettings
    from nanoscope.analysis.events import STUDY_REVIEW_POLICY_COL, event_status
    from nanoscope.analysis.scoring import build_training_frame
    row = record(tmp_path)
    native = prepare_review_export(row, tmp_path / 'source')['result'].payload.df
    native = native.drop(columns=STUDY_REVIEW_POLICY_COL)
    native['review'] = ['unreviewed', pd.NA]
    native['accepted'] = [True, False]
    native['flag'] = ''
    assert event_status(native).tolist() == ['unreviewed', 'rejected']
    row['analysis'] = measure_imported_events(row['prepared'], native, MeasurementSettings(), source_provenance={})
    returned = prepare_review_export(row, tmp_path / 'legacy-return')['result'].payload.df
    assert returned.review.tolist() == ['unreviewed', 'reviewed']
    assert build_training_frame(returned, selection='reviewed').start_idx.tolist() == [int(returned.start_idx.iloc[1])]


def test_unresolved_native_labels_survive_return_and_only_explicit_targets_resolve(tmp_path):
    from nanoscope.analysis.manual_detection import measure_imported_events, MeasurementSettings
    from nanoscope.analysis.study_review import apply_event_decision, retain_event_decisions
    from nanoscope.analysis.scoring import build_training_frame
    from nanoscope.analysis.manual_workspace import load_workspace, save_workspace
    row = record(tmp_path)
    native = prepare_review_export(row, tmp_path / 'first-conflict')['result'].payload.df
    native['review'] = 'reviewed'
    native['consensus_curation_conflict'] = 'contradictory accept/reject rulings'
    imported = measure_imported_events(row['prepared'], native, MeasurementSettings(), source_provenance={})
    row['analysis'] = imported
    unchanged = prepare_review_export(row, tmp_path / 'unchanged-conflict')['result'].payload.df
    assert build_training_frame(unchanged, selection='reviewed').empty
    assert unchanged.consensus_curation_conflict.tolist() == native.consensus_curation_conflict.tolist()
    selected_id = imported.catalog.event_id.iloc[0]
    reviewed = apply_event_decision(imported, [selected_id], True)
    # Even an acceptance no-op resolves only the explicit target's label.
    measured = measure_imported_events(row['prepared'], native, MeasurementSettings(min_depth_sigma=5), source_provenance={})
    row['analysis'] = retain_event_decisions(reviewed, measured)
    save_workspace({'record': row}, tmp_path / 'conflict-checkpoint')
    restored = load_workspace(tmp_path / 'conflict-checkpoint')['record']
    resolved = prepare_review_export(restored, tmp_path / 'resolved-one')['result'].payload.df
    assert resolved.consensus_curation_conflict.tolist() == ['', 'contradictory accept/reject rulings']
    assert build_training_frame(resolved, selection='reviewed').start_idx.tolist() == [int(native.start_idx.iloc[0])]
    restored['analysis'] = apply_event_decision(restored['analysis'], [], None)
    reset = prepare_review_export(restored, tmp_path / 'reset-conflicts')['result'].payload.df
    assert reset.consensus_curation_conflict.tolist() == native.consensus_curation_conflict.tolist()
    assert build_training_frame(reset, selection='reviewed').empty


def test_imported_sources_remain_distinct_after_study_review(tmp_path):
    from nanoscope.analysis.manual_detection import measure_imported_events, MeasurementSettings
    from nanoscope.analysis.study_review import apply_event_decision
    from nanoscope.analysis.events import source_kind
    row = record(tmp_path)
    native = prepare_review_export(row, tmp_path / 'first')['result'].payload.df
    native['source'] = ['detect.cusum', 'manual']
    imported = measure_imported_events(row['prepared'], native, MeasurementSettings(), source_provenance={})
    # A Cu measurement policy must not relabel the imported detector.
    imported.provenance['schmitt_settings'] = {'fixture': True}
    row['analysis'] = apply_event_decision(imported, [imported.catalog.event_id.iloc[0]], True)
    output = prepare_review_export(row, tmp_path / 'imported')['result'].payload.df
    assert output.source.tolist() == ['detect.cusum', 'manual']
    assert output.source.map(source_kind).tolist() == ['conventional', 'manual']


def test_native_undo_and_source_discovery_cover_study_candidates(window, tmp_path):
    row, _ = opened(window, tmp_path)
    window._analysis_panel.set_event_accepted(0, False)
    window._undo_events()
    assert window.snapshot_study_events(row)['events'].accepted.tolist() == [True, False]
    sources = discover_plot_sources(window._session.documents(), window._session.results.get)
    assert any(source.module_id == MODULE_ID for source in sources)


def test_object_cell_snapshot_is_detached_and_catalog_archives_do_not_grow(window, tmp_path):
    row, _ = opened(window, tmp_path)
    result = window._analysis_panel.current_detection()
    result.payload.df['annotation'] = [{'labels': ['a']}, {'labels': ['b']}]
    snapshot = window.snapshot_study_events(row)
    result.payload.df.annotation.iloc[0]['labels'].append('changed later')
    assert snapshot['events'].annotation.iloc[0] == {'labels': ['a']}
    row['analysis'].catalog['source_study_old_metric'] = [1., 2.]
    exported = prepare_review_export(row, tmp_path / 'fresh-review')['result'].payload.df
    assert 'study_source_study_old_metric' not in exported


def test_missing_physical_fraction_reference_survives_merge_and_remeasurement(window, tmp_path):
    from nanoscope.analysis.baseline import BaselineConfig
    from nanoscope.analysis.remeasure import remeasure_events
    row = record(tmp_path)
    row['analysis'].catalog.loc[1, ['start_idx', 'end_idx']] = [1045, 1084]
    row['prepared'].detrended[1045:1085] -= 12
    payload = prepare_review_export(row, tmp_path / 'review')
    row['main_workspace_bridge'] = payload['descriptor']
    window.open_study_review(payload)
    report = window._analysis_panel.merge_event_with_next(0, .0002)
    assert report.status == 'ok'
    result = window._analysis_panel.current_detection()
    assert len(result.payload.df) == 1 and result.payload.df.frac_depth.isna().all()
    updated = remeasure_events(result.payload, payload['document'].raw(0), payload['document'].meta.sample_rate_hz,
        config=BaselineConfig(method='constant'), polarity=-1)
    assert updated.events.df.frac_depth.isna().all()


def test_reconciled_snapshot_checks_each_contributing_baseline(window, tmp_path, monkeypatch):
    from nanoscope.analysis.reconcile import reconcile_detections
    from nanoscope.analysis.events import EventSet
    row, payload = opened(window, tmp_path)
    study = window._analysis_panel.current_detection()
    other = replace(study, module_id='detect.schmitt',
        payload=EventSet(study.payload.df.copy(deep=True), deepcopy(study.payload.provenance)))
    window._session.add_result(other)
    outcome = reconcile_detections([study, other])
    window._session.add_result(outcome.result)
    monkeypatch.setattr(window._analysis_panel, '_baseline_staleness',
        lambda result: 'baseline settings changed' if result.module_id == 'detect.schmitt' else None)
    with pytest.raises(ValueError, match='contributing detector'):
        window.snapshot_study_events(row)
    monkeypatch.setattr(window._analysis_panel, '_baseline_staleness', lambda result: None)
    snapshot = window.snapshot_study_events(row)
    assert snapshot['provenance']['source_module_id'] == 'detect.reconciled'
    assert snapshot['events'].frac_depth.isna().all()


@pytest.mark.parametrize('mutation', ['none', 'channel', 'rate', 'partial'])
def test_raw_source_import_uses_verified_file_channel_clock_not_reader_rounding(window, tmp_path, mutation):
    row, payload = opened(window, tmp_path)
    doc = payload['document']
    # Treat this ordinary on-disk recording as the raw input to a distinct study.
    # A one-ULP reader representation difference is intentionally legitimate.
    raw = np.nextafter(doc.raw(0), np.float32(np.inf)).astype(np.float64)
    prepared = replace(row['prepared'], raw=raw,
        provenance=dict(raw_file=_fingerprint(doc.meta.path), reader_scaling='notebook_float32'))
    raw_record = dict(row, prepared=prepared, source_path=doc.meta.path)
    raw_record.pop('main_workspace_bridge')
    if mutation == 'channel': raw_record['channel'] = 1
    elif mutation == 'rate':
        window._session._documents[doc.doc_id] = replace(doc, meta=replace(doc.meta, sample_rate_hz=40000.))
    elif mutation == 'partial':
        window._session._documents[doc.doc_id] = replace(doc, meta=replace(doc.meta, loaded_range=SampleRange(0, len(raw))))
    if mutation == 'none':
        snapshot = window.snapshot_study_events(raw_record)
        assert snapshot['provenance']['source_kind'] == 'raw_recording'
        assert snapshot['provenance']['study_reader_scaling'] == 'notebook_float32'
    else:
        with pytest.raises(ValueError): window.snapshot_study_events(raw_record)


def test_open_review_refuses_when_launch_project_no_longer_open(window, tmp_path):
    row = record(tmp_path)
    payload = prepare_review_export(row, tmp_path / 'review')
    launch_doc_ids = frozenset({'doc-from-a-project-that-was-since-closed'})
    with pytest.raises(ValueError, match='project changed'):
        window.open_study_review(payload, launch_doc_ids)
    assert window._session.doc_id_for_source(payload['document'].source_key) is None


def test_open_review_still_succeeds_when_launch_project_is_still_open(window, tmp_path):
    row, first = opened(window, tmp_path)
    launch_doc_ids = frozenset(doc.doc_id for doc in window._session.documents())
    second = prepare_review_export(row, tmp_path / 'second-review')
    window.open_study_review(second, launch_doc_ids)
    assert window._session.has_document(second['document'].doc_id)
    assert window._session.has_document(first['document'].doc_id)


def test_a_study_accepted_event_cut_off_by_the_trace_edge_arrives_rejected(tmp_path):
    """The study's own Schmitt keeps an event still in progress at the first or last
    sample of the prepared trace and accepts it; its gate has no boundary rule. The
    export wrote the study's ``accepted`` over characterize's verdict, so the two edge
    rows opened accepted beside ``clipped_by_boundary`` (measured: accepted
    [True, True, True, True]), the state the event table corrects back to rejected on the
    next Accept. A curator's acceptance of the edge row in the study also made it a
    reviewed training positive. The study's decision is now ANDed with characterize's
    rules, as on every rebuild, so the native rule wins and the study's own decision
    stays in ``study_accepted``."""
    from nanoscope.analysis.study_signal import analyze, SchmittSettings
    from nanoscope.analysis.study_review import apply_event_decision
    from nanoscope.analysis.events import FLAG_CLIPPED, event_status, has_flag
    from nanoscope.analysis.scoring import build_training_frame
    row = record(tmp_path, events=False)
    row['prepared'].detrended[:300] -= 15
    row['prepared'].detrended[-300:] -= 15
    row['analysis'] = analyze(row['prepared'], SchmittSettings())
    catalog = row['analysis'].catalog
    assert catalog.accepted.tolist() == [True] * 4
    assert catalog.start_idx.iloc[0] == 0
    assert catalog.end_idx.iloc[-1] == len(row['prepared'].detrended) - 1
    native = prepare_review_export(row, tmp_path / 'edges')['result'].payload.df
    assert has_flag(native.flag, FLAG_CLIPPED).tolist() == [True, False, False, True]
    assert native.accepted.tolist() == [False, True, True, False]
    assert event_status(native).tolist() == ['auto-rejected', 'unreviewed', 'unreviewed', 'auto-rejected']
    assert native.study_accepted.tolist() == [True] * 4
    row['analysis'] = apply_event_decision(row['analysis'], [catalog.event_id.iloc[0]], True)
    reviewed = prepare_review_export(row, tmp_path / 'reviewed-edge')['result'].payload.df
    assert reviewed.review.iloc[0] == 'reviewed' and not reviewed.accepted.iloc[0]
    assert event_status(reviewed).iloc[0] == 'auto-rejected'
    assert build_training_frame(reviewed, selection='reviewed').empty


def test_an_imported_max_dwell_rejection_keeps_its_reason_in_the_review_table(tmp_path):
    """A main-workspace detection that declared ``max_dwell_s`` rejects a clog as
    ``dwell>max_dwell_s``, and the study imports the row rejected with that token in
    ``source_flag``. The return trip stripped every derived token, but this export
    declares no ``max_dwell_s`` and cannot re-derive that one, so the row opened rejected
    with flag '' (measured): C21's rejected-with-no-reason, invisible to any filter on the
    reason. The token is now inherited, as recharacterize_span inherits it when the limit
    is unknown."""
    from nanoscope.analysis.manual_detection import measure_imported_events, MeasurementSettings
    from nanoscope.analysis.events import FLAG_MAX_DWELL, event_status
    row = record(tmp_path)
    main = pd.DataFrame(dict(event_id=np.array([0, 1], np.int64),
        start_idx=np.array([1000, 2000], np.int64), end_idx=np.array([1040, 2070], np.int64),
        accepted=[True, False], flag=pd.array(['', FLAG_MAX_DWELL], dtype='string'),
        review=['unreviewed', 'unreviewed']))
    row['analysis'] = measure_imported_events(row['prepared'], main, MeasurementSettings(),
        source_provenance=dict(max_dwell_s=1e-3, polarity=-1))
    native = prepare_review_export(row, tmp_path / 'returned')['result'].payload.df
    assert native.flag.tolist() == ['', FLAG_MAX_DWELL]
    assert native.accepted.tolist() == [True, False]
    assert event_status(native).tolist() == ['unreviewed', 'auto-rejected']


@pytest.mark.parametrize('decision', ['accepted_in_the_study', 'reset_to_the_automatic_gate'])
def test_a_study_acceptance_does_not_clear_an_inherited_max_dwell_rejection(tmp_path, decision):
    """The same imported clog, then accepted in the study: by the curator's Accept, or by
    "Reset review to automatic gate", whose gate passes a deep clog. The study catalog
    then reads accepted [True, True]. inherited_curation trusts a prior ``accepted`` to
    honour its own flags, which holds for a main-workspace row but not for a study
    decision, so inheriting the reason alone opened the clog accepted beside
    ``dwell>max_dwell_s`` (measured on a first version of the fix), the state the event
    table calls forbidden. An inherited rejecting reason now rejects the row whatever the
    study decided."""
    from nanoscope.analysis.manual_detection import measure_imported_events, MeasurementSettings
    from nanoscope.analysis.study_review import apply_event_decision
    from nanoscope.analysis.events import FLAG_MAX_DWELL, event_status
    row = record(tmp_path)
    main = pd.DataFrame(dict(event_id=np.array([0, 1], np.int64),
        start_idx=np.array([1000, 2000], np.int64), end_idx=np.array([1040, 2070], np.int64),
        accepted=[True, False], flag=pd.array(['', FLAG_MAX_DWELL], dtype='string'),
        review=['unreviewed', 'unreviewed']))
    imported = measure_imported_events(row['prepared'], main, MeasurementSettings(),
        source_provenance=dict(max_dwell_s=1e-3, polarity=-1))
    if decision == 'accepted_in_the_study':
        imported = apply_event_decision(imported, [imported.catalog.event_id.iloc[1]], True)
    else:
        imported = apply_event_decision(imported, [], None)
    assert imported.catalog.accepted.tolist() == [True, True]
    row['analysis'] = imported
    native = prepare_review_export(row, tmp_path / 'returned')['result'].payload.df
    assert native.flag.tolist() == ['', FLAG_MAX_DWELL]
    assert native.accepted.tolist() == [True, False]
    assert event_status(native).iloc[1] == 'auto-rejected'


@pytest.mark.parametrize('branch', ['raw_recording', 'prepared_review'])
def test_a_case_variant_spelling_of_the_open_file_is_the_same_source(window, tmp_path, branch):
    """Both source checks in snapshot_events compared ``Path.resolve()`` strings, and
    resolve() does not fold case. On a case-insensitive volume (macOS's default APFS) the
    study's own file under another case spelling (a Finder rename, a Recent entry) was
    refused with "Select this study recording's original raw channel or its exported
    prepared review copy." Identity is now asked of the filesystem through
    identity.same_source; a byte-identical copy at another path is another file and is
    still refused."""
    row, payload = opened(window, tmp_path)
    path = payload['document'].meta.path
    variant = path.with_name(path.name.upper())
    if not variant.exists() or not os.path.samefile(variant, path):
        pytest.skip('case-sensitive volume: no case variant of one file exists')
    copy = tmp_path / 'elsewhere' / path.name
    copy.parent.mkdir()
    shutil.copyfile(path, copy)
    if branch == 'raw_recording':
        prepared = replace(row['prepared'],
            provenance=dict(row['prepared'].provenance, raw_file=_fingerprint(path)))
        row = dict(row, prepared=prepared)
        row.pop('main_workspace_bridge')
        spelled = lambda spelling: dict(row, source_path=spelling)
    else:
        spelled = lambda spelling: dict(row,
            main_workspace_bridge=dict(row['main_workspace_bridge'], path=str(spelling)))
    assert window.snapshot_study_events(spelled(variant))['provenance']['source_kind'] == branch
    with pytest.raises(ValueError, match='original raw channel'):
        window.snapshot_study_events(spelled(copy))
