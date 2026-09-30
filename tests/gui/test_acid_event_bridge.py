"""Acid events use the shared editor while retaining calibrated study science."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import acid_incubation as acid
from nanoscope.analysis.incubation_detection import _fingerprint
from nanoscope.analysis.events import source_kind, KIND_CONVENTIONAL
from nanoscope.gui import acid_event_bridge as bridge
from .test_study_event_bridge import window


def state(tmp_path, *, acquisition_orientation=1, primary_polarity=1):
    signal = np.random.default_rng(73).normal(0, .2, 10000).astype(np.float32)
    signal[1000:1040] -= 10
    signal[2000:2070] -= 15
    source = tmp_path / 'source.abf'; source.write_bytes(b'original acid current')
    parquet = tmp_path / 'signal.parquet'
    pd.DataFrame({'current_detr_pA': signal * acquisition_orientation * primary_polarity,
        'current_raw_pA': signal - 200}).to_parquet(parquet, index=False)
    result = acid.new_state(records=[dict(tag='acid', source_path=source,
        source_info={'sha256': _fingerprint(source)['sha256']}, recording_id='acid-1')])
    row = result['records'][0]
    row['prepared'] = dict(parquet_path=parquet, sha256=_fingerprint(parquet)['sha256'],
        source_sha256=_fingerprint(source)['sha256'],
        settings_sha256=acid._digest({k: result['settings'][k] for k in acid.PREPARE_KEYS}),
        metadata=dict(fs_hz=50000., n_samples=len(signal), event_polarity=acquisition_orientation))
    detector = 'cusum' if primary_polarity == 1 else 'cusum_resistive'
    row['detection'] = dict(primary_set=detector, catalog_detector=detector,
        context_path=tmp_path/'context', settings_sha256='calibrated-settings')
    row['catalog'] = pd.DataFrame(dict(event_id=['C000000', 'C000001'],
        start_idx=[1000, 2000], end_idx=[1039, 2069], accepted=[True, False],
        passes_gate=[True, False], event_review=['unreviewed', 'unreviewed'],
        polarity=[primary_polarity]*2, peak_depth_pA=[10., 15.], local_I0_pA=[200., 200.]))
    row['families'] = pd.DataFrame(dict(event_id=['C000000', 'C000001'], family=[0, 1]))
    result['family_model_path'] = tmp_path/'frozen-family-model.json'
    result['family_model_sha256'] = 'frozen'
    return result, signal


def opened(window, tmp_path, **kwargs):
    study, signal = state(tmp_path, **kwargs)
    payload = bridge.prepare_review_export(study, 'acid', tmp_path/'review')
    study['records'][0]['main_workspace_bridge'] = payload['descriptor']
    window.open_study_review(payload)
    return study, payload, signal


@pytest.mark.parametrize('acquisition_orientation,primary_polarity', [(1, 1), (-1, 1), (1, -1), (-1, -1)])
def test_prepared_copy_is_events_down_and_preserves_acid_measurements(tmp_path, acquisition_orientation, primary_polarity):
    study, signal = state(tmp_path, acquisition_orientation=acquisition_orientation, primary_polarity=primary_polarity)
    payload = bridge.prepare_review_export(study, 'acid', tmp_path/'review')
    np.testing.assert_array_equal(payload['document'].raw(0), signal)
    frame = payload['result'].payload.df
    assert frame.event_id.tolist() == [0, 1]
    assert frame.study_event_id.tolist() == ['C000000', 'C000001']
    assert frame.end_idx.tolist() == [1040, 2070]
    assert frame.study_local_I0_pA.tolist() == [200., 200.]
    assert frame.frac_depth.isna().all()
    assert frame.review.tolist() == ['unreviewed', 'unreviewed']
    assert all(source_kind(v) == KIND_CONVENTIONAL for v in frame.source)
    assert payload['descriptor']['acid_review_signature'] == bridge.review_signature(study, 'acid')


def test_shared_review_undo_save_repeat_and_annotation_reopen(window, tmp_path):
    study, payload, _ = opened(window, tmp_path)
    old = study['records'][0]['catalog'].copy(deep=True)
    panel = window._analysis_panel
    panel.set_event_accepted(0, False)
    window._undo_events()
    assert bool(panel.current_detection().payload.df.accepted.iloc[0])
    window._redo_events()
    frame = panel.current_detection().payload.df
    frame.loc[0, 'flag'] = 'molecule=insulin'
    snapshot = bridge.snapshot_events(window, study, 'acid')
    updated = bridge.apply_review(study, 'acid', snapshot, tmp_path/'commit')
    pd.testing.assert_frame_equal(study['records'][0]['catalog'], old)
    row = updated['records'][0]
    assert row['catalog'].accepted.tolist() == [False, False]
    assert row['catalog'].event_review.tolist() == ['reviewed', 'unreviewed']
    assert row['catalog'].peak_depth_pA.tolist() == [10., 15.]
    pd.testing.assert_frame_equal(row['families'], study['records'][0]['families'])
    assert updated['family_model_sha256'] == 'frozen'
    assert not acid.selection_is_current(row)
    panel.set_event_accepted(1, True)
    again = bridge.apply_review(updated, 'acid', bridge.snapshot_events(window, updated, 'acid'), tmp_path/'commit2')
    assert again['records'][0]['catalog'].accepted.tolist() == [False, True]
    fresh = bridge.prepare_review_export(again, 'acid', tmp_path/'review2')
    assert 'molecule=insulin' in fresh['result'].payload.df.flag.iloc[0]
    assert fresh['result'].payload.df.review.tolist() == ['reviewed', 'reviewed']
    assert not any('source_source_' in c for c in again['records'][0]['catalog'])


def test_geometry_remeasures_invalidates_family_and_retains_native_decisions(window, tmp_path, monkeypatch):
    study, payload, _ = opened(window, tmp_path)
    window._open_event_inspector()
    window._event_inspector.show_event(1)
    window._event_inspector._apply_edited_bounds(1, 1995, 2075, payload['document'].raw(0))
    calls = []
    def measure(state, row, events, identities, folder, progress, cancel):
        calls.append((events.copy(), identities))
        result = row['catalog'].copy()
        result['start_idx'] = events.start_idx.to_numpy()
        result['end_idx'] = events.end_idx.to_numpy()-1
        result['peak_depth_pA'] = [10., 17.]
        return result, tmp_path/'new-context'
    monkeypatch.setattr(bridge, '_remeasure', measure)
    updated = bridge.apply_review(study, 'acid', bridge.snapshot_events(window, study, 'acid'), tmp_path/'commit')
    assert len(calls) == 1
    row = updated['records'][0]
    assert row['catalog'].start_idx.tolist() == [1000, 1995]
    assert row['catalog'].end_idx.tolist() == [1039, 2074]
    assert row['catalog'].peak_depth_pA.tolist() == [10., 17.]
    assert row['catalog'].accepted.tolist() == [True, False]
    assert row['detection']['context_path'] == tmp_path/'new-context'
    assert row['families'] is None and 'family_model_path' not in updated
    assert row['catalog'].event_id.tolist() == ['C000000', 'C000001']


@pytest.mark.parametrize('mutation', ['catalog', 'prepared', 'raw', 'calibration', 'partial', 'wrong_recording'])
def test_stale_or_mismatched_review_cannot_enter_study(window, tmp_path, mutation):
    study, payload, _ = opened(window, tmp_path)
    row = study['records'][0]
    if mutation == 'catalog': row['catalog'].loc[0, 'accepted'] = False
    elif mutation == 'prepared': row['prepared']['parquet_path'].write_bytes(b'changed')
    elif mutation == 'raw': row['source_path'].write_bytes(b'changed')
    elif mutation == 'calibration':
        from nanoscope.core.calibration import ChannelCalibration
        window._session.set_calibration(payload['document'].doc_id, 0, ChannelCalibration(gain=2., target_units='pA'))
    elif mutation == 'partial': window._analysis_panel.current_detection().payload.provenance['analysed_stop_idx'] = 100
    else: row['main_workspace_bridge']['path'] = str(tmp_path/'elsewhere.h5')
    with pytest.raises(ValueError): bridge.snapshot_events(window, study, 'acid')


def test_commit_refuses_replaced_review_file_after_snapshot(window, tmp_path):
    study, payload, _ = opened(window, tmp_path)
    snapshot = bridge.snapshot_events(window, study, 'acid')
    payload['document'].meta.path.write_bytes(b'changed between capture and commit')
    with pytest.raises(ValueError, match='review file changed'):
        bridge.apply_review(study, 'acid', snapshot, tmp_path/'commit')


def test_merge_renumbering_retains_untouched_acid_identity(tmp_path):
    study, _ = state(tmp_path)
    row = study['records'][0]
    frame = pd.DataFrame(dict(event_id=[0], study_event_id=['C000001'],
        start_idx=[2000], end_idx=[2070], accepted=[True]))
    assert bridge._acid_ids(row, frame) == ['C000001']
    assert not bridge._same_geometry(row['catalog'], frame, ['C000001'])
    frame['study_event_id'] = pd.NA
    assert bridge._acid_ids(row, frame)[0].startswith('acid-1:curated:')


@pytest.mark.parametrize('changes', [dict(event_id=[0, 0]), dict(start_idx=[1000, 1030]),
    dict(end_idx=[1000, 2070]), dict(accepted=[1, 0])])
def test_invalid_native_event_contract_refused_before_measurement(tmp_path, changes):
    frame = pd.DataFrame(dict(event_id=[0, 1], start_idx=[1000, 2000], end_idx=[1040, 2070], accepted=[True, False]))
    for key, values in changes.items(): frame[key] = values
    with pytest.raises(ValueError): bridge._validated_events(frame, 10000)


def _measure_changed_rows(state, row, events, identities, folder, progress, cancel):
    """Isolate native identity/annotation transfer from expensive Acid kernels."""
    catalog = pd.DataFrame(dict(event_id=identities,
        start_idx=events.start_idx.to_numpy(), end_idx=events.end_idx.to_numpy()-1,
        accepted=[False]*len(events), passes_gate=[False]*len(events),
        event_review=['unreviewed']*len(events), polarity=[1]*len(events),
        peak_depth_pA=[21.]*len(events), local_I0_pA=[200.]*len(events)))
    return catalog, Path(folder)/'context'


def test_real_manual_add_merge_repeated_commit_and_undo(window, tmp_path, monkeypatch):
    from nanoscope.analysis.events import SOURCE_MANUAL
    study, payload, _ = opened(window, tmp_path)
    monkeypatch.setattr(bridge, '_remeasure', _measure_changed_rows)
    doc_id = payload['document'].doc_id
    window._trace_view.set_measure_active(doc_id, 0, True)
    window._trace_view.set_measure_region(1045/50000., 1080/50000.)
    panel = window._analysis_panel
    added = panel.add_current_selection_as_event()
    assert added is not None
    frame = panel.current_detection().payload.df
    manual = frame.loc[frame.event_id == added].iloc[0]
    assert manual.source == SOURCE_MANUAL and manual.review == 'reviewed'
    assert pd.isna(manual.study_start_idx) and pd.isna(manual.study_passes_gate)
    panel.set_event_accepted(added, True)
    frame.loc[frame.event_id == added, 'flag'] = 'family=manual-test'
    first = bridge.apply_review(study, 'acid', bridge.snapshot_events(window, study, 'acid'), tmp_path/'add')
    catalog = first['records'][0]['catalog']
    assert catalog.event_id.is_unique and len(catalog) == 3
    manual_id = catalog.loc[catalog.source_event_id == added, 'event_id'].iloc[0]
    assert manual_id.startswith('acid-1:curated:')
    assert bool(catalog.loc[catalog.event_id == manual_id, 'accepted'].iloc[0])
    again = bridge.apply_review(first, 'acid', bridge.snapshot_events(window, first, 'acid'), tmp_path/'add-again')
    assert again['records'][0]['catalog'].event_id.tolist() == catalog.event_id.tolist()
    report = panel.merge_event_with_next(0, .001)
    assert report.status == 'ok' and len(panel.current_detection().payload.df) == 2
    merged = bridge.apply_review(again, 'acid', bridge.snapshot_events(window, again, 'acid'), tmp_path/'merge')
    current = merged['records'][0]['catalog']
    assert current.event_id.tolist() == ['C000000', 'C000001']
    assert current.end_idx.tolist() == [1079, 2069]
    assert 'family=manual-test' in current.source_flag.iloc[0]
    assert current.source_event_id.tolist() == [0, 1]
    repeated = bridge.apply_review(merged, 'acid', bridge.snapshot_events(window, merged, 'acid'), tmp_path/'merge-again')
    assert repeated['records'][0]['catalog'].event_id.tolist() == current.event_id.tolist()
    window._undo_events()
    restored = bridge.apply_review(repeated, 'acid', bridge.snapshot_events(window, repeated, 'acid'), tmp_path/'undo-merge')
    assert len(restored['records'][0]['catalog']) == 3
    assert set(restored['records'][0]['catalog'].event_id) == {'C000000', 'C000001', manual_id}


def test_empty_native_result_commits_empty_catalog(window, tmp_path, monkeypatch):
    from dataclasses import replace
    from nanoscope.analysis.events import EventSet
    study, _, _ = opened(window, tmp_path)
    monkeypatch.setattr(bridge, '_remeasure', _measure_changed_rows)
    result = window._analysis_panel.current_detection()
    empty = EventSet(result.payload.df.iloc[0:0].copy(), deepcopy(result.payload.provenance))
    window._session.add_result(replace(result, payload=empty))
    updated = bridge.apply_review(study, 'acid', bridge.snapshot_events(window, study, 'acid'), tmp_path/'empty')
    row = updated['records'][0]
    assert row['catalog'].empty and row['detection']['candidates'] == row['detection']['accepted'] == 0
    assert row['families'] is None
    fresh = bridge.prepare_review_export(updated, 'acid', tmp_path/'empty-reopen')
    assert fresh['result'].payload.df.empty


def test_canceled_measurement_leaves_catalog_receipt_and_checkpoint_unchanged(window, tmp_path, monkeypatch):
    import threading
    from nanoscope.analysis.cancel import OperationCancelled
    study, payload, _ = opened(window, tmp_path)
    study = acid.confirm_selection(study, 'acid')
    folder = tmp_path/'workspace'
    acid.checkpoint.save_workspace(study, folder, expected_revision=None)
    before = (folder/'workspace.json').read_bytes()
    old = study['records'][0]['catalog'].copy(deep=True)
    window._open_event_inspector(); window._event_inspector.show_event(1)
    window._event_inspector._apply_edited_bounds(1, 1995, 2075, payload['document'].raw(0))
    snapshot = bridge.snapshot_events(window, study, 'acid')
    cancelled = threading.Event()
    def measure(*args):
        result = _measure_changed_rows(*args)
        cancelled.set()
        return result
    monkeypatch.setattr(bridge, '_remeasure', measure)
    with pytest.raises(OperationCancelled):
        bridge.apply_review(study, 'acid', snapshot, tmp_path/'canceled', cancel=cancelled)
    pd.testing.assert_frame_equal(study['records'][0]['catalog'], old)
    assert acid.selection_is_current(study['records'][0])
    assert (folder/'workspace.json').read_bytes() == before


def test_empty_automatic_resistive_selection_keeps_resistive_calibration_context(tmp_path, monkeypatch):
    study, signal = state(tmp_path, primary_polarity=-1)
    row = study['records'][0]
    detector = 'cusum_resistive'
    ns = dict(signal=signal, cusum_info={'polarity': 1}, cusum_info_res={'polarity': -1},
        detector_catalogs={detector: row['catalog'].copy()}, analysis_sets={detector: pd.DataFrame()},
        primary_polarity=-1)
    class Runtime:
        def __init__(self, *args, **kwargs): self.namespace = ns
        def run(self, stage):
            if stage == 25:
                # This is stage25's empty primary fallback; human acceptance
                # can still select these rows after the advisory gate fails.
                ns['primary_polarity'] = 1
                ns['primary_cusum_info'] = ns['cusum_info']
    saved = []
    monkeypatch.setattr(acid, 'AcidRuntime', Runtime)
    monkeypatch.setattr(acid, '_restore_context', lambda *args: None)
    monkeypatch.setattr(acid, '_save_context', lambda context, path: saved.append(context.copy()) or path)
    events = pd.DataFrame(dict(event_id=[0, 1], start_idx=[999, 2000], end_idx=[1040, 2070], accepted=[True, True]))
    bridge._remeasure(study, row, events, ['C000000', 'C000001'], tmp_path/'commit', None, None)
    assert saved[0]['primary_polarity'] == -1
    assert saved[0]['primary_cusum_info'] == {'polarity': -1}
    np.testing.assert_array_equal(saved[0]['analysis_signal'], -signal)


def test_undo_committed_merge_restores_removed_original_acid_id(window, tmp_path, monkeypatch):
    study, _ = state(tmp_path)
    # Adjacent fragments meet the native occupancy requirement for a real merge.
    study['records'][0]['catalog'].loc[1, ['start_idx', 'end_idx']] = [1045, 1080]
    payload = bridge.prepare_review_export(study, 'acid', tmp_path/'review')
    study['records'][0]['main_workspace_bridge'] = payload['descriptor']
    window.open_study_review(payload)
    monkeypatch.setattr(bridge, '_remeasure', _measure_changed_rows)
    panel = window._analysis_panel
    assert panel.merge_event_with_next(0, .001).status == 'ok'
    merged = bridge.apply_review(study, 'acid', bridge.snapshot_events(window, study, 'acid'), tmp_path/'merged')
    assert merged['records'][0]['catalog'].event_id.tolist() == ['C000000']
    window._undo_events()
    restored = bridge.apply_review(merged, 'acid', bridge.snapshot_events(window, merged, 'acid'), tmp_path/'restored')
    assert restored['records'][0]['catalog'].event_id.tolist() == ['C000000', 'C000001']
