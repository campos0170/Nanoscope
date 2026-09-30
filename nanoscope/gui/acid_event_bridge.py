"""The acid study uses the application's shared event editor and undo history.

Native measurements describe a reproducible, events-down prepared trace. Acid
measurements remain the calibrated notebook measurements; a geometry edit is
remeasured by those same kernels before the study checkpoint can be committed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import acid_incubation as acid
from ..analysis.cancel import check_cancelled
from ..analysis.event_amplitude import AMPLITUDE_COLUMNS
from ..analysis.incubation_detection import PreparedRecording, RecordingAnalysis, _fingerprint
from ..analysis.manual_detection import _imported_integer_column
from ..analysis.events import REVIEW_NONE, REVIEW_VALUES
from . import study_event_bridge as native


def _row(state, tag):
    row = acid._record(state, tag)
    if not row.get('prepared') or not row.get('detection') or not isinstance(row.get('catalog'), pd.DataFrame):
        raise ValueError('Prepare and detect this acid recording before curating its events.')
    if not row.get('recording_id'):
        raise ValueError('The acid recording has no stable identity; reopen the study.')
    return row


def _polarity(row):
    detector = row['detection'].get('catalog_detector', row['detection']['primary_set'].removesuffix('_matched'))
    polarity = -1 if detector.endswith('_resistive') else 1
    catalog = row['catalog']
    if 'polarity' in catalog and len(catalog):
        values = catalog.polarity.drop_duplicates().tolist()
        if values != [polarity]:
            raise ValueError('The acid catalog has inconsistent event polarities.')
    return polarity


def review_signature(state, tag):
    """Cheap state identity for refocusing an existing review without reading traces."""
    row = _row(state, tag)
    prepared = row['prepared']
    return acid._digest(dict(recording_id=row['recording_id'], tag=row['tag'],
        source_path=str(row['source_path']), source_sha256=row['source_info'].get('sha256'),
        channel=row.get('channel', 0), pore_id=row.get('pore_id'),
        prepared_path=str(prepared['parquet_path']), prepared_sha256=prepared.get('sha256'),
        prepared_metadata=prepared['metadata'], prepared_settings=prepared.get('settings_sha256'),
        detection_settings=row['detection'].get('settings_sha256'),
        detection_context=str(row['detection'].get('context_path', '')),
        primary_set=row['detection']['primary_set'], polarity=_polarity(row),
        catalog=native.catalog_signature({'candidates': row['catalog']})))


def _verify_files(state, row, cancel=None):
    check_cancelled(cancel)
    prepared = row['prepared']
    expected = acid._digest({k: state['settings'][k] for k in acid.PREPARE_KEYS})
    if prepared.get('settings_sha256') != expected:
        raise ValueError('Acid preparation settings changed; prepare and detect the recording again.')
    for path, expected, label in (
            (prepared['parquet_path'], prepared.get('sha256'), 'Prepared acid data'),
            (row['source_path'], prepared.get('source_sha256', row['source_info'].get('sha256')), 'Raw acid source')):
        if not expected or _fingerprint(path, cancel)['sha256'] != expected:
            raise ValueError(label + ' changed; reopen or prepare the correct source before review.')


def _catalog_adapter(row):
    frame = row['catalog'].copy(deep=True).reset_index(drop=True)
    if not frame.event_id.is_unique:
        raise ValueError('Acid event identities must be unique.')
    # A previous commit remembers the CURRENT native IDs, including a merge's
    # renumbering. Acid IDs stay strings and are carried as study_event_id.
    if 'source_event_id' not in frame:
        frame['source_event_id'] = np.arange(len(frame), dtype=np.int64)
    ids = _imported_integer_column(frame.rename(columns={'event_id': '_acid_id', 'source_event_id': 'event_id'}), 'event_id')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Saved native event identities must be unique.')
    frame['source_event_id'] = ids
    if 'event_review' not in frame:
        frame['event_review'] = REVIEW_NONE
    if 'source_source' not in frame:
        detector = row['detection'].get('catalog_detector', row['detection']['primary_set'].removesuffix('_matched'))
        frame['source_source'] = 'detect.cusum' if detector.startswith('cusum') else 'detect.schmitt'
    return frame


def _adapter(state, tag, cancel=None):
    import pyarrow.parquet as pq

    row = _row(state, tag)
    _verify_files(state, row, cancel)
    prepared = row['prepared']; metadata = prepared['metadata']
    n, fs = int(metadata['n_samples']), float(metadata['fs_hz'])
    if n <= 0 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Acid review requires a complete, positive sample clock.')
    reader = pq.ParquetFile(prepared['parquet_path'])
    if reader.metadata.num_rows != n:
        raise ValueError('Prepared acid sample count no longer matches its metadata.')
    raw_column = 'current_raw_code' if 'current_raw_code' in reader.schema.names else 'current_raw_pA'
    signal, raw = np.empty(n, np.float32), np.empty(n, np.float32)
    offset = 0
    for group in range(reader.num_row_groups):
        check_cancelled(cancel)
        data = reader.read_row_group(group, columns=['current_detr_pA', raw_column])
        stop = offset + data.num_rows
        signal[offset:stop] = data.column('current_detr_pA').to_numpy()
        raw[offset:stop] = data.column(raw_column).to_numpy()
        offset = stop
    if raw_column == 'current_raw_code':
        raw *= np.float32(metadata['adc_step_pA'])
    orientation = metadata.get('event_polarity', 1)
    if orientation not in (-1, 1):
        raise ValueError('Prepared event polarity must be -1 or +1.')
    # _load_prepared applies the acquisition orientation once. The primary
    # resistive catalog is measured against -signal, so apply that once too.
    signal *= np.float32(orientation * _polarity(row))
    provenance = dict(event_polarity=1, raw_file=dict(path=str(row['source_path']),
        sha256=prepared.get('source_sha256', row['source_info'].get('sha256'))),
        reader_scaling='Acid prepared parquet; events-down primary polarity')
    recording = PreparedRecording(tag, raw, signal, fs, n/fs, deepcopy(metadata), provenance)
    catalog = _catalog_adapter(row)
    analysis = RecordingAnalysis(tag, raw, signal, fs, n/fs, catalog, provenance)
    return dict(recording_id=row['recording_id'], tag=tag, source_path=row['source_path'],
        channel=row.get('channel', 0), pore_id=row.get('pore_id', ''), prepared=recording,
        analysis=analysis, main_workspace_bridge=deepcopy(row.get('main_workspace_bridge')))


def prepare_review_export(state, tag, folder, *, cancel=None):
    adapter = _adapter(state, tag, cancel)
    payload = native.prepare_review_export(adapter, Path(folder), cancel=cancel)
    descriptor = payload['descriptor']
    descriptor.update(acid_review_signature=review_signature(state, tag), acid_bridge_version=1,
        acid_event_ids=adapter['analysis'].catalog.event_id.tolist())
    # Preserve native annotations without interpreting them as Acid measurements.
    frame = payload['result'].payload.df
    catalog = adapter['analysis'].catalog
    for name in ('label', 'consensus_label_conflicts', *AMPLITUDE_COLUMNS):
        if 'source_' + name in catalog:
            frame[name] = catalog['source_' + name].to_numpy(copy=True)
    # A hand-added row has no original Acid measurements. The native append
    # preserves dtypes, so study integer/bool columns must represent missing
    # values rather than fail conversion (or turn NaN into a True gate result).
    # Object bool archives also avoid the native merge's boolean OR rule: an
    # archived Acid gate belongs to one original event, not a merged span.
    for name in frame:
        if name.startswith('study_') and name != 'study_review_policy':
            if pd.api.types.is_bool_dtype(frame[name].dtype):
                frame[name] = frame[name].astype(object)
            elif pd.api.types.is_integer_dtype(frame[name].dtype):
                frame[name] = frame[name].astype('Int64')
    payload['result'].payload.provenance['acid_measurement_note'] = (
        'Native measurements use the prepared trace. Committing geometry changes '
        'recomputes acid measurements, gates and waveform features with the saved calibration. '
        'Explicit amplitude annotations are retained separately in source_ columns.')
    return payload


def snapshot_events(window, state, tag):
    row = _row(state, tag)
    descriptor = row.get('main_workspace_bridge')
    if not isinstance(descriptor, dict) or descriptor.get('acid_review_signature') != review_signature(state, tag):
        raise ValueError('The acid study changed since this review was opened. Open its current events again.')
    result = window._analysis_panel.current_detection()
    if result is None or not window._session.has_document(result.doc_id):
        raise ValueError('Select this acid recording in the main event editor first.')
    doc = window._session.document(result.doc_id)
    if doc.meta.path.resolve() != Path(descriptor['path']).resolve():
        raise ValueError('Select this acid recording’s prepared review copy before saving its edits.')
    if int((result.payload.provenance or {}).get('polarity', 0)) != -1:
        raise ValueError('Acid review uses events-down prepared current; select downward detections.')
    snapshot = native.snapshot_events(window, _adapter(state, tag))
    snapshot['provenance'].update(acid_review_signature=review_signature(state, tag),
        acid_recording_id=row['recording_id'], acid_primary_polarity=_polarity(row))
    return snapshot


def _validated_events(events, n):
    if not isinstance(events, pd.DataFrame) or not events.columns.is_unique:
        raise ValueError('Review events must be a table with unique columns.')
    if not {'event_id', 'start_idx', 'end_idx', 'accepted'} <= set(events):
        raise ValueError('Review events need identities, bounds and acceptance decisions.')
    frame = deepcopy(events)
    for name in ('event_id', 'start_idx', 'end_idx'):
        frame[name] = _imported_integer_column(frame, name)
    if not frame.event_id.is_unique:
        raise ValueError('Native event IDs must be unique.')
    if frame.accepted.isna().any() or any(not isinstance(x, (bool, np.bool_)) for x in frame.accepted):
        raise ValueError('Native acceptance decisions must be boolean.')
    frame = frame.sort_values('start_idx', kind='stable').reset_index(drop=True)
    if ((frame.end_idx <= frame.start_idx) | (frame.end_idx > n)).any():
        raise ValueError('Review event bounds must be nonempty and within the complete recording.')
    if len(frame) > 1 and (frame.start_idx.to_numpy()[1:] < frame.end_idx.to_numpy()[:-1]).any():
        raise ValueError('Review event bounds must not overlap.')
    return frame


def _acid_ids(row, events):
    """Native merges renumber ALL rows; the exported Acid identity survives them."""
    previous = row['catalog']
    existing = set(previous.event_id)
    # Native undo can restore a detector row removed by an ALREADY COMMITTED
    # merge. Its exported identity remains valid even while absent from the
    # current Acid catalog; never relabel that restoration as a new event.
    existing.update(row.get('main_workspace_bridge', {}).get('acid_event_ids', []))
    used = set(); result = []
    for _, event in events.iterrows():
        value = event.get('study_event_id')
        if not isinstance(value, (str, int, np.integer)) or value not in existing or value in used:
            # Native additions have no study annotation until the next export.
            # Preserve the last commit's identity when this same native row still
            # intersects it; a merge may legitimately retain its first identity.
            matches = previous.iloc[0:0]
            if 'source_event_id' in previous:
                matches = previous.loc[(previous.source_event_id == event.event_id)
                    & (previous.start_idx < event.end_idx) & (previous.end_idx >= event.start_idx)]
            value = matches.event_id.iloc[0] if len(matches) == 1 else None
            if value is None or value in used:
                token = acid._digest([row['recording_id'], int(event.start_idx), int(event.end_idx)])[:20]
                value = row['recording_id'] + ':curated:' + token
        if value in used:
            raise ValueError('Review events could not be assigned distinct study identities.')
        result.append(value); used.add(value)
    return result


def _same_geometry(previous, events, identities):
    if len(previous) != len(events) or set(previous.event_id) != set(identities):
        return False
    prior = previous.set_index('event_id').loc[identities]
    return (np.array_equal(prior.start_idx.to_numpy(), events.start_idx.to_numpy())
        and np.array_equal(prior.end_idx.to_numpy() + 1, events.end_idx.to_numpy()))


def _remeasure(state, row, events, identities, folder, progress, cancel):
    """Rerun measurement/feature/matching stages, never calibration or detection."""
    folder = Path(folder)
    with acid._math_session(state['settings']['SEED']):
        rt = acid.AcidRuntime(state['settings'], folder, row, state['records'], progress=progress, cancel=cancel)
        acid._restore_context(rt, row)
        ns = rt.namespace
        detector = row['detection'].get('catalog_detector', row['detection']['primary_set'].removesuffix('_matched'))
        region_name = {'cusum': 'cusum_candidate_regions', 'threshold': 'threshold_candidate_regions',
            'cusum_resistive': 'cusum_candidate_regions_res', 'threshold_resistive': 'threshold_candidate_regions_res'}[detector]
        ns[region_name] = list(zip(events.start_idx.astype(int), (events.end_idx-1).astype(int)))
        # The original level table can only describe exact detector bounds. Stage
        # 20 joins on BOTH bounds, so changed/new/merged rows receive missing level
        # metadata rather than the first fragment's cached shape measurements.
        if len(events):
            for stage in (20, 22, 24):
                rt.run(stage)
            catalog = ns['detector_catalogs'][detector]
            catalog['event_id'] = identities
            ns[detector + '_catalog'] = catalog
            rt.run(25)
            catalog = ns['detector_catalogs'][detector].copy(deep=True)
            selected = set(ns['analysis_sets'].get(row['detection']['primary_set'], pd.DataFrame()).get('event_id', []))
            catalog['automatic_accepted'] = catalog.event_id.isin(selected)
            catalog['passes_gate'] = catalog.accepted.astype(bool)
        else:
            catalog = row['catalog'].iloc[0:0].copy(deep=True)
            ns['detector_catalogs'][detector] = catalog
            ns[detector + '_catalog'] = catalog
            ns['analysis_sets'][row['detection']['primary_set']] = catalog
        # The checkpoint context owns updated candidate masks and diagnostics.
        # Curated acceptance is installed by apply_review, then _restore_context
        # selects it for all later analyses without replacing the automatic gate.
        ns['primary_polarity'] = _polarity(row)
        ns['primary_cusum_info'] = ns['cusum_info'] if ns['primary_polarity'] == 1 else ns['cusum_info_res']
        ns['analysis_signal'] = ns['signal'] if ns['primary_polarity'] == 1 else -ns['signal']
        context = acid._save_context(ns, folder / 'context')
    return catalog, context


def apply_review(state, tag, snapshot, folder, *, progress=None, cancel=None):
    """Return a detached study revision containing the exact native decisions."""
    row = _row(state, tag); descriptor = row.get('main_workspace_bridge')
    provenance = deepcopy(snapshot.get('provenance', {}))
    signature = review_signature(state, tag)
    if (not isinstance(descriptor, dict) or descriptor.get('acid_review_signature') != signature
            or provenance.get('acid_review_signature') != signature
            or provenance.get('acid_recording_id') != row['recording_id']
            or provenance.get('source_path') != descriptor.get('path')):
        raise ValueError('This review snapshot no longer matches the acid study. Open the current events again.')
    _verify_files(state, row, cancel)
    if _fingerprint(descriptor['path'], cancel)['sha256'] != descriptor['file_sha256']:
        raise ValueError('The prepared review file changed; open a fresh review copy.')
    events = _validated_events(snapshot['events'], int(row['prepared']['metadata']['n_samples']))
    identities = _acid_ids(row, events)
    same_geometry = _same_geometry(row['catalog'], events, identities)
    if progress: progress(.05, 'Saving event decisions…' if same_geometry else 'Remeasuring edited acid event bounds…')
    context = None
    if same_geometry:
        catalog = row['catalog'].set_index('event_id').loc[identities].reset_index().copy(deep=True)
    else:
        catalog, context = _remeasure(state, row, events, identities, folder, progress, cancel)
    catalog['accepted'] = events.accepted.to_numpy(bool)
    reviews = events.get('review', pd.Series(REVIEW_NONE, index=events.index)).astype('string')
    if reviews.isna().any() or not reviews.isin(REVIEW_VALUES).all():
        raise ValueError('Native event review status is invalid.')
    catalog['event_review'] = reviews.to_numpy()
    # Source columns archive native measurements and all annotations. Do not
    # accumulate another study_/source_ prefix on every round trip.
    catalog = catalog.drop(columns=[c for c in catalog if str(c).startswith('source_')])
    for name in events.columns:
        if not str(name).startswith('study_'):
            catalog['source_' + name] = events[name].to_numpy(copy=True)
    if 'consensus_curation_conflict' in events:
        catalog['event_review_conflict'] = events.consensus_curation_conflict.to_numpy(copy=True)
    updated = acid._copy(state); current = acid._record(updated, tag)
    current['catalog'] = catalog
    current.pop('review_receipt', None)
    current['native_review_provenance'] = provenance
    current['detection'] = {**row['detection'], 'candidates': len(catalog),
        'accepted': int(catalog.accepted.sum()), 'acceptance_method': 'Shared main event editor',
        'human_review': bool(reviews.eq('reviewed').any())}
    if context is not None:
        current['detection']['context_path'] = context
    acid._retire(updated, 'analysis')
    if not same_geometry:
        acid._invalidate_families(updated)
    acid._history(updated, 'Curate acid events in main editor', tag=tag,
        geometry_changed=not same_geometry, candidates=len(catalog), accepted=int(catalog.accepted.sum()),
        source=provenance)
    current['main_workspace_bridge'] = {**descriptor,
        'acid_event_ids': list(dict.fromkeys([*descriptor.get('acid_event_ids', []), *identities])),
        'catalog_signature': native.catalog_signature({'candidates': _catalog_adapter(current)}),
        'acid_review_signature': review_signature(updated, tag)}
    check_cancelled(cancel)
    return updated
