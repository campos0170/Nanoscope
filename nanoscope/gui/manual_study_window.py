"""Interactive raw-recording study operations with explicit, saved stage boundaries."""
from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..analysis import manual_preparation as prep
from ..analysis import manual_detection as detect
from ..analysis import manual_comparisons as compare
from ..analysis import study_signal as signal
from ..analysis import signal_comparisons
from ..analysis import study_review
from ..analysis import study_cluster_review
from ..plot import signal_study_figures
from ..analysis import manual_workspace as checkpoint
from ..analysis.incubation_study import StudyRecording, NOTEBOOK_FEATURES, _appendix_population_table
from ..analysis.incubation_detection import RecordingAnalysis
from ..analysis.ml_preprocess import MLPreprocessConfig
from ..analysis.cancel import OperationCancelled, cancel_scope, check_cancelled
from ..core.identity import same_source
from ..plot import incubation_figures as figures
from .incubation_window import _FrameModel, _FigureView
from .worker_thread import JOB_NONE, retire_worker_thread, start_worker_thread


def _empty_state():
    return dict(name='Manual study', model_path='', records=[], comparison=None,
                features=None, clustering=None, mixture=None, figures=[], history=[],
                comparison_settings=asdict(compare.ComparisonSettings()),
                cluster_settings=asdict(compare.ClusterSettings()),
                mixture_settings=asdict(compare.MixtureSettings()))


def _copy_state(state):
    copied = dict(state)
    copied['records'] = [dict(r) for r in state['records']]
    copied['history'] = list(state['history'])
    copied['figures'] = list(state['figures'])
    return copied


def _invalidate_global(state):
    for key in ('comparison', 'features', 'clustering', 'mixture'): state[key] = None
    state['figures'] = []


def _review_record_key(row):
    """Cheap identity of immutable, committed scientific objects; never an approval."""
    analysis=row.get('analysis')
    return (row.get('recording_id'),id(analysis),id(getattr(analysis,'catalog',None)),id(row.get('prepared')))


def _review_views(state,cancel=None):
    """Build display summaries in the worker; mutation actions still validate afresh.

    This cache is not saved and cannot authorize acceptance changes. The GUI only
    renders these results for the exact committed objects that produced them.
    """
    output={'records':{},'clusters':{}}
    for row in state['records']:
        check_cancelled(cancel)
        output['records'][row.get('recording_id')]=(_review_record_key(row),study_review.review_is_current(row))
    for kind,snapshot in state.get('cluster_reviews',{}).items():
        check_cancelled(cancel)
        view={'snapshot_id':snapshot.get('snapshot_id'),'current':False,
              'record_keys':{r['recording_id']:_review_record_key(r) for r in state['records']
                             if r.get('recording_id') in snapshot.get('recording_ids',[])}}
        try:
            study_cluster_review.require_current(snapshot,state['records'])
            view['summaries']={}
            for method in snapshot['methods']:
                check_cancelled(cancel)
                view['summaries'][method]=study_cluster_review.summaries(snapshot,state['records'],method)
            view['current']=True
        except (ValueError,KeyError,TypeError) as exc:
            view['error']=str(exc)
        output['clusters'][kind]=view
    check_cancelled(cancel)
    output['methods_text']=_methods_text(state,output)
    return output


def _methods_text(state,views):
    # Keep the on-screen audit bounded; the checkpoint retains every exact change.
    def preview(value):
        if isinstance(value,dict): return {k:preview(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return ([preview(v) for v in value] if len(value)<=50 else
                    [*[preview(v) for v in value[:50]],f'{len(value)-50} more entries saved in the workspace checkpoint'])
        return value
    details={'completed_recordings':[],'actions':state['history']}
    if state.get('compatibility_notice'): details['compatibility_warning']=state['compatibility_notice']
    for row in state['records']:
        info=dict(tag=row['tag'],source=str(row['source_path']))
        if 'prepared' in row: info['preparation']=row['prepared'].provenance
        if 'inference' in row: info.update(inference=row['inference'].provenance,coverage=row['inference'].coverage)
        if 'analysis' in row:
            catalog=row['analysis'].catalog
            info.update(candidates=len(catalog),automatic_accepted=int(catalog.passes_gate.sum()),
                        reviewed_accepted=int(catalog.accepted.sum()),measurement=row['analysis'].provenance)
        if row.get('event_input')=='main_workspace': info['event_input']='Saved main-workspace detection snapshot; acceptance retained'
        details['completed_recordings'].append(info)
    if state['comparison'] is not None: details['comparisons']=state['comparison'].metadata
    if state['clustering'] is not None: details['clustering']=state['clustering'][1]
    if state['mixture'] is not None: details['mixture']=state['mixture'][2]
    if state.get('cluster_reviews'):
        details['cluster_review_sources']={kind:dict(methods=snapshot['methods'],
            current=views['clusters'].get(kind,{}).get('current',False))
            for kind,snapshot in state['cluster_reviews'].items()}
    text=json.dumps(preview(details),indent=2,default=str)
    return text if len(text)<=100000 else text[:100000]+'\n… Display truncated; full audit evidence is saved in the workspace checkpoint.'


def _signal_source(row):
    return signal.SourceOptions(**{k:row.get(k,v) for k,v in asdict(signal.SourceOptions()).items()})


def _source_defaults(row):
    for k,v in asdict(signal.SourceOptions()).items(): row.setdefault(k,v)
    row.setdefault('comparison_only',False)
    return row


def _condition(record):
    return StudyRecording(record['tag'], record['source_path'], record['source_path'],
                          record['incubation_h'], record['pore_id'], record['pore_diameter_nm'],
                          record['concentration_pm'], record['voltage_mv'])


def _validate_record_identities(records):
    tags, identities = set(), set()
    for row in records:
        tag = row.get('tag')
        if not isinstance(tag, str) or not tag or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in tag):
            raise ValueError('Recording tags must contain only letters, digits, underscore or hyphen.')
        if tag in tags:
            raise ValueError(f'Duplicate recording tag {tag!r}. Each recording must have a unique tag; the workspace was not opened or changed.')
        tags.add(tag)
        identity = row.get('recording_id')
        if identity is not None:
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError('Recording identities must be nonempty and unique.')
            identities.add(identity)


def _validated_workspace_state(state):
    """Validate the GUI contract before replacing an active workspace.

    Codec envelopes also support arbitrary analysis fixtures, so decoding alone
    does not imply the saved value is a complete manual-window state. Older
    workspaces acquire stable recording IDs in memory without an implicit save.
    """
    required = _empty_state()
    if not isinstance(state, dict) or not required.keys() <= state.keys():
        raise ValueError('Invalid manual workspace state: required study fields are missing.')
    for key in ('records', 'figures', 'history'):
        if not isinstance(state[key], list):
            raise ValueError(f'Invalid manual workspace state: {key} must be a list.')
    if not isinstance(state['name'], str) or not isinstance(state['model_path'], str):
        raise ValueError('Invalid manual workspace name or model path.')
    for key in ('comparison_settings', 'cluster_settings', 'mixture_settings'):
        if not isinstance(state[key], dict):
            raise ValueError(f'Invalid manual workspace state: {key} must be a settings object.')
    compare.ComparisonSettings(**state['comparison_settings'])
    compare.ClusterSettings(**state['cluster_settings'])
    state=dict(state,mixture_settings=asdict(compare.MixtureSettings(**state['mixture_settings'])))
    for row in state['records']:
        _source_defaults(row); _signal_source(row)
        if row.get('schmitt_settings'): signal.SchmittSettings(**row['schmitt_settings'])
        source_keys = {'source_path', 'source_info', 'tag', 'incubation_h', 'pore_id',
                       'pore_diameter_nm', 'concentration_pm', 'voltage_mv', 'channel',
                       'event_polarity', 'reader_scaling', 'preparation_settings',
                       'inference_settings', 'decode_settings', 'measurement_settings'}
        if not isinstance(row, dict) or not source_keys <= row.keys():
            raise ValueError('Invalid manual workspace recording: required source fields are missing.')
        if not isinstance(row['source_path'], Path) or not isinstance(row['source_info'], dict):
            raise ValueError('Invalid manual workspace recording source.')
        if (not isinstance(row['channel'], int) or row['channel'] < 0 or
                row['event_polarity'] not in (-1, 1) or row['reader_scaling'] not in prep.READER_SCALINGS):
            raise ValueError('Invalid recording channel, polarity or reader convention.')
        if not isinstance(row['pore_id'], str) or not row['pore_id']:
            raise ValueError('A recording needs a physical pore identity.')
        values = [row[k] for k in ('incubation_h', 'pore_diameter_nm', 'concentration_pm', 'voltage_mv')]
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in values) or values[0] < 0 or values[1] <= 0 or values[2] < 0:
            raise ValueError('Invalid recording conditions.')
        MLPreprocessConfig(**row['preparation_settings'])
        detect.InferenceSettings(**row['inference_settings'])
        detect.DecodeSettings(**row['decode_settings'])
        detect.MeasurementSettings(**row['measurement_settings'])
        if 'prepared' in row and (not isinstance(row['prepared'], prep.RawPreparedRecording) or row['prepared'].tag != row['tag']):
            raise ValueError('Prepared data do not match their recording identity.')
        if 'inference' in row and not isinstance(row['inference'], detect.InferenceResult):
            raise ValueError('Invalid saved model inference.')
        if 'candidates' in row and not isinstance(row['candidates'], pd.DataFrame):
            raise ValueError('Invalid saved candidate table.')
        if row.get('event_input') == 'main_workspace':
            if (not isinstance(row.get('imported_events'), pd.DataFrame) or
                    not {'event_id', 'start_idx', 'end_idx', 'accepted'} <= set(row['imported_events']) or
                    not isinstance(row.get('imported_event_provenance'), dict)):
                raise ValueError('Invalid saved main-workspace detection snapshot.')
        if 'analysis' in row:
            analysis = row['analysis']
            if (not isinstance(analysis, RecordingAnalysis) or analysis.tag != row['tag'] or
                    not isinstance(analysis.catalog, pd.DataFrame) or
                    not {'event_id', 'start_idx', 'end_idx', 'passes_gate', 'accepted'} <= set(analysis.catalog)):
                raise ValueError('Invalid saved measurement table or recording identity.')
    _validate_record_identities(state['records'])
    if state['comparison'] is not None and not isinstance(state['comparison'], compare.ComparisonResult):
        raise ValueError('Invalid saved comparison result.')
    if state['features'] is not None and not isinstance(state['features'], pd.DataFrame):
        raise ValueError('Invalid saved waveform features.')
    for key, size in (('clustering', 2), ('mixture', 3)):
        result = state[key]
        if result is not None and (not isinstance(result, (tuple, list)) or len(result) != size or
                                   not isinstance(result[0], pd.DataFrame) or not isinstance(result[-1], dict)):
            raise ValueError(f'Invalid saved {key} result.')
    if state['mixture'] is not None and state.get('mixture_tag') not in {r['tag'] for r in state['records']}:
        raise ValueError('Saved mixture target does not match an included recording.')
    if any(not isinstance(path, Path) or not path.is_file() for path in state['figures']):
        raise ValueError('A saved workspace figure is missing or invalid.')
    state = _copy_state(state)
    if not isinstance(state.get('cluster_reviews', {}), dict):
        raise ValueError('Invalid saved cluster review snapshots.')
    for kind, snapshot in state.get('cluster_reviews', {}).items():
        if kind not in ('pooled', 'mixture') or not isinstance(snapshot, dict):
            raise ValueError('Invalid saved cluster review snapshot.')
        study_cluster_review.validate_snapshot(snapshot)
        if snapshot['kind'] != kind: raise ValueError('Cluster review source does not match its snapshot.')
    for row in state['records']:
        row.setdefault('recording_id', uuid.uuid4().hex)
    analyses = [r.get('analysis') for r in state['records']]
    cached = [state['features']] + [state[key][0] for key in ('clustering', 'mixture') if state[key] is not None]
    issue = next((message for frame in cached if frame is not None
                  if (message := compare.feature_orientation_issue(frame, analyses))), None)
    if issue:
        state['features'] = state['clustering'] = state['mixture'] = None
        state['figures'] = [p for p in state['figures'] if not p.stem.startswith(
            ('pooled_', 'quantitative_appendix_', 'population_examples_', 'population_onsets_'))]
        # cluster_reviews snapshots describe groupings WITHIN clustering/mixture --
        # clearing those two without also clearing this left a stale-but-"current"
        # review usable: require_current() only checks each source recording's own
        # basis fingerprint, which does not change just because clustering/mixture
        # were wiped here, so the review kept validating against a result that no
        # longer exists.
        state['cluster_reviews'] = {}
        state['compatibility_notice'] = issue + ' Cached population results were cleared from the active state; existing exported files remain on disk.'
    return state


class _OperationWorker(QtCore.QObject):
    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.cancel_event = threading.Event()

    def cancel(self): self.cancel_event.set()

    @QtCore.Slot()
    def run(self):
        try:
            with cancel_scope(self.cancel_event.is_set):
                result = self.operation(self.progress.emit, self.cancel_event.is_set)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
        else:
            self.finished.emit(result)


class _Form(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, fields):
        super().__init__()
        form = QtWidgets.QFormLayout(self)
        self.controls, self.kinds = {}, {}
        for key, label, kind, default in fields:
            if isinstance(kind, tuple):
                widget = QtWidgets.QComboBox()
                for title, value in kind: widget.addItem(title, value)
            elif kind is bool:
                widget = QtWidgets.QCheckBox()
            elif kind is int:
                widget = QtWidgets.QSpinBox(); widget.setRange(0, 1_000_000)
            elif kind is float:
                widget = QtWidgets.QDoubleSpinBox(); widget.setRange(-1e9, 1e9); widget.setDecimals(6)
            else:
                widget = QtWidgets.QLineEdit()
            widget.setAccessibleName(label)
            self.controls[key], self.kinds[key] = widget, kind
            form.addRow(label, widget)
            self.set_value(key, default)
            signal = (widget.currentIndexChanged if isinstance(kind, tuple) else
                      widget.toggled if kind is bool else widget.valueChanged if kind in (int, float) else widget.textChanged)
            signal.connect(self.changed)

    def set_value(self, key, value):
        widget, kind = self.controls[key], self.kinds[key]
        widget.blockSignals(True)
        if isinstance(kind, tuple): widget.setCurrentIndex(max(0, widget.findData(value)))
        elif kind is bool: widget.setChecked(bool(value))
        elif kind in (int, float): widget.setValue(value)
        else: widget.setText(', '.join(map(str, value)) if isinstance(value, (tuple, list)) else '' if value is None else str(value))
        widget.blockSignals(False)

    def set_values(self, values):
        for key, value in values.items():
            if key in self.controls: self.set_value(key, value)

    def values(self):
        result = {}
        for key, widget in self.controls.items():
            kind = self.kinds[key]
            result[key] = (widget.currentData() if isinstance(kind, tuple) else widget.isChecked() if kind is bool
                           else widget.value() if kind in (int, float) else widget.text().strip())
        return result


def _numbers(text, cast=float):
    return tuple(cast(v.strip()) for v in str(text).replace(';', ',').split(',') if v.strip())


def _prep_defaults():
    return MLPreprocessConfig(notch_enabled=False, notch_freqs=(), baseline_method='none',
                              baseline_decim=10, normalize=False).to_dict()


def _export_event_tables(catalog, tag, folder, *, cancel=None):
    """Publish both reviewed tables together; never leave half an export on failure."""
    folder=Path(folder)
    targets=[folder/f'{tag}_{suffix}.csv' for suffix in ('candidates','accepted')]
    if any(p.exists() for p in targets): raise FileExistsError('An event export already exists in this folder.')
    published=[]
    with tempfile.TemporaryDirectory(prefix='.nanoscope-export-', dir=folder) as staging:
        temporary=[Path(staging)/p.name for p in targets]
        for frame,path in zip((catalog,catalog[catalog.accepted]),temporary):
            check_cancelled(cancel); frame.to_csv(path,index=False)
        try:
            for source,target in zip(temporary,targets):
                check_cancelled(cancel); os.link(source,target); published.append(target)
        except BaseException:
            for path in published: path.unlink(missing_ok=True)
            raise
    return targets


class ManualStudyWindow(QtWidgets.QMainWindow):
    """No Run study action: each operation consumes the last committed prior stage."""
    operationFinished = QtCore.Signal(str)
    operationFailed = QtCore.Signal(str)
    operationCancelled = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle('Manual study workspace — NanoScope')
        self.resize(1420, 950)
        self.state, self.folder = _empty_state(), None
        self._revision = None
        self._review_views = {'records':{},'clusters':{}}
        self._table_view = None
        self._worker = self._thread = self._retiring = None
        self._install = None
        self._loading = False
        self._active_index = -1
        self._operation_label = ''
        self._curation_closed = False
        self._retire_timer = QtCore.QTimer(self)
        self._retire_timer.setInterval(100)
        self._retire_timer.timeout.connect(self._check_retired)
        self._build_ui()
        for form in (self._source_form, self._prep_form, self._inference_form,
                     self._decode_form, self._measure_form, self._comparison_form,
                     self._cluster_form, self._mixture_form, self._schmitt_form, self._null_form):
            form.changed.connect(self._settings_edited)
        self._model.textChanged.connect(self._settings_edited)
        self._mixture_target.currentIndexChanged.connect(self._settings_edited)
        self.refresh()

    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        self._heading = QtWidgets.QLabel('Incubation study')
        font = self._heading.font(); font.setPointSize(font.pointSize()+5); font.setBold(True); self._heading.setFont(font)
        layout.addWidget(self._heading)
        note = QtWidgets.QLabel('Import raw recordings, choose settings, and run each operation separately. Completed stages are saved automatically. Changing an upstream result clears dependent results; previous files remain in the workspace.')
        note.setWordWrap(True); layout.addWidget(note)
        self._toolbar = QtWidgets.QWidget(); bar = QtWidgets.QHBoxLayout(self._toolbar); bar.setContentsMargins(0,0,0,0)
        self._button('New workspace…', self._choose_new, bar)
        self._button('Open workspace…', self._choose_open, bar)
        self._button('Save workspace', self.save, bar)
        self._button('Add raw ABF…', self._choose_add, bar)
        self._selector = QtWidgets.QComboBox(); self._selector.setAccessibleName('Active study recording')
        self._selector.currentIndexChanged.connect(self._select_recording)
        bar.addWidget(self._selector, 1)
        layout.addWidget(self._toolbar)
        self._identity = QtWidgets.QLabel(); self._identity.setWordWrap(True); layout.addWidget(self._identity)
        self._record_summary = QtWidgets.QLabel(); self._record_summary.setWordWrap(True); layout.addWidget(self._record_summary)
        pending_row = QtWidgets.QHBoxLayout()
        self._pending = QtWidgets.QLabel(); self._pending.setWordWrap(True); pending_row.addWidget(self._pending, 1)
        self._restore = self._button('Restore completed settings', self.restore_settings, pending_row)
        layout.addLayout(pending_row)
        splitter = QtWidgets.QSplitter(); layout.addWidget(splitter, 1)
        self._stages = QtWidgets.QTabWidget(); self._stages.setTabPosition(QtWidgets.QTabWidget.West)
        splitter.addWidget(self._stages)
        self._build_sources(); self._build_preparation(); self._build_detection(); self._build_comparisons(); self._build_clustering(); self._build_figures()
        self._results = QtWidgets.QTabWidget(); splitter.addWidget(self._results); splitter.setSizes([530, 850])
        self._trace = pg.PlotWidget(background='w'); self._trace.setLabel('bottom','Retained acquisition time',units='s'); self._trace.setLabel('left','Current',units='pA')
        # The application's dark theme uses light global plot text. This preview
        # has a fixed white canvas, so keep its axes readable in either theme.
        for name in ('left', 'bottom'):
            self._trace.getAxis(name).setPen(pg.mkPen('#555555'))
            self._trace.getAxis(name).setTextPen(pg.mkPen('#333333'))
        self._trace.showGrid(x=True,y=True,alpha=.15)
        self._results.addTab(self._trace,'Trace / event')
        self._table = QtWidgets.QTableView(); self._table_model = _FrameModel(self); self._table.setModel(self._table_model)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows); self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.clicked.connect(self._focus_event); self._results.addTab(self._table,'Results table')
        self._figure = _FigureView(); self._results.addTab(self._figure,'Saved figure')
        self._details = QtWidgets.QTextBrowser(); self._results.addTab(self._details,'Methods / history')
        self._status = QtWidgets.QLabel('Create a workspace folder, then add the raw ABF recordings.')
        self._status.setWordWrap(True); layout.addWidget(self._status)
        row = QtWidgets.QHBoxLayout(); self._progress = QtWidgets.QProgressBar(); self._progress.setRange(0,100); row.addWidget(self._progress,1)
        self._cancel = self._button('Cancel operation', self.cancel, row); self._cancel.setEnabled(False); layout.addLayout(row)

    @staticmethod
    def _button(title, callback, layout):
        button=QtWidgets.QPushButton(title); button.clicked.connect(callback); layout.addWidget(button); return button

    def _page(self, title):
        page=QtWidgets.QWidget(); box=QtWidgets.QVBoxLayout(page)
        scroll=QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        self._stages.addTab(scroll,title)
        return box

    @staticmethod
    def _note(layout,text):
        label=QtWidgets.QLabel(text); label.setWordWrap(True); layout.addWidget(label)

    def _build_sources(self):
        box=self._page('1 Sources')
        self._study_details_form=_Form([('name','Study name',str,'Incubation study'),
            ('sample','Sample / material',str,''),('condition_label','Condition / treatment label',str,'')])
        self._study_details_form.changed.connect(self._settings_edited)
        box.addWidget(self._study_details_form)
        self._button('Apply study details',self.apply_study_details,box)
        self._note(box,'Set the recording identity and experimental conditions. Pore identity distinguishes independent physical pores, even when their diameters match.')
        self._source_form=_Form([
            ('tag','Recording tag',str,''),('incubation_h','Incubation (h)',float,0),('pore_id','Physical pore ID',str,''),
            ('pore_diameter_nm','Pore diameter (nm)',float,1),('concentration_pm','Concentration (pM)',float,0),('voltage_mv','Voltage (mV)',float,0),
            ('channel','Current channel (1-based)',int,1),('event_polarity','Event orientation',(('Downward in source',1),('Upward in source',-1)),1),
            ('reader_scaling','ABF numeric scaling',(('Notebook float32 convention','notebook_float32'),('Standard NanoScope reader','standard')),'notebook_float32'),
            ('add_channel','Add current channel (1-based; 0 disables)',int,0),
            ('add_channel_as_pa','Additional channel values are pA (override header)',bool,False),
            ('keep_start_s','Keep from source time (s)',float,0),('keep_stop_s','Keep until source time (s; 0 = end)',float,0),
            ('exclude_s','Exclude source intervals (start:stop; start:stop)',str,''),
            ('median_engine','Median window edges',(('Reflect samples at edges','reflect'),('Shrink window at edges','shrinking')),'reflect'),
            ('comparison_only','Supplementary recording (exclude from pooled series)',bool,False)])
        box.addWidget(self._source_form)
        self._button('Apply recording details',self.apply_recording_details,box)
        self._source_info=QtWidgets.QLabel(); self._source_info.setWordWrap(True); box.addWidget(self._source_info)
        self._button('Remove selected recording',self.remove_recording,box)
        box.addStretch()

    def _build_preparation(self):
        box=self._page('2 Prepare')
        self._note(box,'Prepare the retained source intervals before normalization. Source times remain available in exported event tables. Filters run in the order shown: sequential notch, optional float32 rounding, baseline subtraction, optional despiking. Inspect raw, filtered and baseline traces before detection.')
        self._prep_form=_Form([
            ('notch_enabled','Apply sequential notch',bool,False),('notch_freqs','Notch frequencies (Hz, comma separated)',str,''),('notch_q','Notch Q',float,30),
            ('notch_float32','Round notch output to float32',bool,False),
            ('baseline_method','Baseline method',(('None','none'),('Event-masked median','masked_median'),('Robust median','robust_median'),('Savitzky-Golay baseline','savgol')),'none'),
            ('masked_rough_s','Mask: rough window (s)',float,.5),('masked_base_s','Mask: baseline window (s)',float,.2),
            ('baseline_decim','Median-grid decimation (samples)',int,10),('masked_mask_sigma','Mask threshold (MAD sigma)',float,3),('masked_dilate_ms','Mask dilation (ms)',float,2),
            ('baseline_window_s','Robust median / SavGol window (s)',float,.5),('baseline_poly','SavGol polynomial order',int,3),('despike_size','Despike kernel (samples; 0 disables)',int,0)])
        box.addWidget(self._prep_form)
        self._button('Prepare selected recording',self.prepare_selected,box)
        self._preview_choice=QtWidgets.QComboBox(); self._preview_choice.addItems(['Prepared current','Raw current','Filtered current and baseline','Model probability'])
        self._preview_choice.currentIndexChanged.connect(self.preview_trace); box.addWidget(self._preview_choice)
        self._button('Show full recording',self.preview_trace,box)
        self._button('Export prepared CSV and metadata…',self.export_prepared,box)
        box.addStretch()

    def _build_detection(self):
        box=self._page('3 Detect / review')
        self._note(box,'Curate study events with the main app’s event table, Event Gallery and Inspector, including manual events, boundary edits, merging and undo. Save curation to study in the main workspace, then return here to confirm the selection for analysis.')
        self._main_curation_button=self._button('Curate in main workspace…',self.open_main_analysis,box)
        self._review_status=QtWidgets.QLabel(); self._review_status.setWordWrap(True); box.addWidget(self._review_status)
        self._button('Show all candidates',self.show_candidates,box)
        self._review_filter=QtWidgets.QComboBox(); self._review_filter.addItems(['All candidates','Accepted only','Rejected only'])
        self._review_filter.setAccessibleName('Study event review filter')
        self._review_filter.currentIndexChanged.connect(lambda: self.show_candidates(reset_filter=False)); box.addWidget(self._review_filter)
        row=QtWidgets.QHBoxLayout(); self._button('Accept selected',lambda:self.review_selected(True),row); self._button('Reject selected',lambda:self.review_selected(False),row); box.addLayout(row)
        self._button('Use existing main-workspace detections',self.use_main_detections,box)
        self._button('Use reviewed events for analysis',self.confirm_event_review,box)
        self._button('Reset review to automatic gate',self.reset_review,box)
        self._button('Export candidate and accepted tables…',self.export_events,box)
        self._note(box,'When review is complete, choose Use reviewed events for analysis. A later acceptance, boundary or measurement change requires another review confirmation. Use existing main-workspace detections can also capture a matching original recording’s event table.')
        self._event_input_label=QtWidgets.QLabel(); self._event_input_label.setWordWrap(True); box.addWidget(self._event_input_label)
        self._schmitt_form=_Form([
            ('smooth_samples','Signal mean window (samples)',int,5),('enter_sigma','Signal enter (MAD sigma)',float,4),
            ('enter_min_pa','Minimum enter threshold (pA)',float,8),('exit_sigma','Signal exit (MAD sigma)',float,1),
            ('merge_ms','Signal merge gap (ms)',float,5),('min_ms','Signal minimum duration (ms)',float,.2),
            ('plateau_ms','Plateau minimum duration (ms)',float,1),('plateau_sigma','Plateau sustained depth (MAD sigma)',float,3),
            ('i0_window_s','Raw I0 rolling window (s)',float,5),('i0_decim','Raw I0 grid (samples)',int,50)])
        box.addWidget(self._schmitt_form)
        self._button('Detect and measure signal events',self.run_signal_detection,box)
        self._null_form=_Form([('global_seeds','Global-phase null seeds',int,5),('block_seeds','Block-phase null seeds',int,3),('block_s','Null block duration (s)',float,10)])
        box.addWidget(self._null_form); self._button('Run signal null controls',self.run_signal_nulls,box)
        self._button('Show null-control counts',self.show_signal_nulls,box)
        self._note(box,'Signal events: short spikes pass; long regions require the sustained-depth threshold. Long shallow regions remain reviewable. Null thresholds are pinned to the real trace. The model route below is independent.')
        self._model=QtWidgets.QLineEdit(); self._model.setAccessibleName('Trained model path'); self._model.setPlaceholderText('Choose the supplied trained .keras model')
        box.addWidget(self._model); self._button('Choose trained model…',self._choose_model,box)
        self._inference_form=_Form([
            ('clip_sigma','Robust normalization: clip and divide (sigma)',float,30),('window_samples','Model window (samples)',int,2048),
            ('stride_samples','Window stride (samples)',int,1024),('batch_size','Inference batch size',int,64),
            ('tail_policy','Window coverage',(('Exclude exact last start (reference)','exclude_last_start'),('Include exact last start','include_last_start'),('Anchor at recording end','anchor_end')),'exclude_last_start')])
        box.addWidget(self._inference_form); self._button('Run model inference',self.run_inference,box)
        self._decode_form=_Form([('enter','Enter probability',float,.4),('exit','Exit probability',float,.15),('smooth','Probability median kernel (samples)',int,3),('merge_ms','Merge gap (ms)',float,.05),('min_ms','Minimum event duration (ms)',float,.1)])
        box.addWidget(self._decode_form); self._button('Find candidate events',self.find_candidates,box)
        self._measure_form=_Form([('min_depth_sigma','Sustained-depth acceptance (quiet MAD sigma)',float,4),('i0_block_s','Quiet raw I0 block (s)',float,.1),('i0_window_s','I0 median smoothing window (s)',float,5)])
        box.addWidget(self._measure_form); self._button('Measure and gate candidates',self.measure_candidates,box)
        self._note(box,'Every candidate is retained. The table keeps the automatic gate decision separate from your acceptance decision. Select a table row to inspect its full-resolution waveform. For reference reproduction, retain the automatic decisions.')

    def _build_comparisons(self):
        box=self._page('4 Compare')
        self._note(box,'Compare the committed, reviewed event tables from all included recordings. Each recording retains its full duration and physical pore identity. Peak/global, sustained/local and sustained/global amplitude definitions are exported separately.')
        self._comparison_form=_Form([('n_blocks','Time blocks',int,20),('n_boot','Bootstrap draws',int,4000),('seed','Random seed',int,0),('floor_start','Fractional floor: start',float,.02),('floor_stop','Fractional floor: stop',float,.2),('floor_step','Fractional floor: step',float,.01),('sensitivity_blocks','Sensitivity block counts',str,'10,20,40')])
        box.addWidget(self._comparison_form); self._button('Compute recording comparisons',self.run_comparisons,box)
        self._comparison_table=QtWidgets.QComboBox(); self._comparison_table.addItems(['Summary','Accepted events','gate_transfer_sweep','rate_block_sensitivity','gate_transfer_pairs','distribution_distances','signal_types','nominal_pairs','drift','monotonicity'])
        self._comparison_table.currentTextChanged.connect(self.show_comparison); box.addWidget(self._comparison_table)
        self._button('Show comparison table',self.show_comparison,box)
        box.addStretch()

    def _build_clustering(self):
        box=self._page('5 Features / populations')
        self._note(box,'Extract waveform features from reviewed events first. Pooled exploratory clustering and selected-recording mixture fitting are independent analyses. The feature list is explicit; component labels are descriptive, not molecular identities.')
        from .study_cluster_controls import StudyClusterControls
        self._cluster_review=StudyClusterControls(self)
        box.addWidget(self._cluster_review)
        self._button('Extract reviewed-event waveform features',self.extract_features,box)
        self._cluster_form=_Form([('feature_columns','Pooled clustering features (comma separated)',str,','.join(NOTEBOOK_FEATURES)),('seed','Clustering seed',int,0),('kmeans_k','K-means components',int,3),('pca_variance','PCA retained variance target',float,.95),('max_pcs','Maximum retained PCs',int,10),('hdbscan_min_samples','HDBSCAN minimum samples',int,10),('with_tsne','Compute descriptive t-SNE',bool,True)])
        box.addWidget(self._cluster_form); self._button('Run pooled exploratory clustering',self.run_clustering,box)
        self._button('Show saved clustering assignments',self.show_clustering,box)
        self._mixture_target=QtWidgets.QComboBox(); self._mixture_target.setAccessibleName('Mixture target recording'); box.addWidget(self._mixture_target)
        defaults=asdict(compare.MixtureSettings())
        self._mixture_form=_Form([('feature_columns','Mixture features (comma separated)',str,','.join(defaults['feature_columns'])),('p_grid','PCA dimension grid (blank = automatic)',str,''),('k_grid','Mixture K grid',str,','.join(map(str,defaults['k_grid']))),('cv_folds','Cross-validation folds',int,5),('n_boot','Mixture bootstrap replicates',int,200),('seed','Mixture selection / fit seed',int,0),('bootstrap_seed','Mixture bootstrap seed',int,0),('min_events','Minimum accepted events (0 disables)',int,0)])
        box.addWidget(self._mixture_form); self._button('Fit selected-recording mixture',self.run_mixture,box)
        self._mixture_table=QtWidgets.QComboBox(); self._mixture_table.setAccessibleName('Saved mixture table')
        self._mixture_table.addItems(['Assignments','Population quantification','Cross-validation grid','Mixture-weight bootstrap','BIC ladder','Exploratory crosstab'])
        self._mixture_table.currentTextChanged.connect(self.show_mixture); box.addWidget(self._mixture_table)
        self._button('Show saved mixture table',self.show_mixture,box)
        self._button('Show waveform feature table',self.show_features,box)

    def _build_figures(self):
        box=self._page('6 Figures')
        self._note(box,'Choose which completed analysis to render. Figure generation reads saved stage results; it does not run a detector or fit a model. Each group exports its plotting data and methods beside the images.')
        self._figure_group=QtWidgets.QComboBox(); self._figure_group.addItems(['Selected recording inspection (5 figures)','Recording comparisons','Pooled exploratory clustering','Selected-recording mixture'])
        box.addWidget(self._figure_group); self._button('Generate selected figure group',self.render_figures,box)
        self._figure_selector=QtWidgets.QComboBox(); self._figure_selector.setAccessibleName('Manual workspace saved figure'); self._figure_selector.currentIndexChanged.connect(self.show_figure); box.addWidget(self._figure_selector)
        self._button('Fit figure to view',lambda:self._figure.fit(),box)
        self._button('Style selected figure…',self.style_selected_figure,box)
        self._note(box,'Style selected figure opens NanoScope’s shared appearance controls. Edit fonts, panel labels, series colors, line widths and markers; save styling to this study or save a frozen figure to the main Figure Library for composition and export.')
        self._note(box,'Figures and numerical sidecars are saved in the workspace runs folder. Save the complete workspace to preserve the intermediate arrays, probabilities, review decisions, parameters and provenance. On reopening, completed stages are restored without recalculation.')
        box.addStretch()

    def _guard(self, action):
        try: return action()
        except Exception as exc:
            self._status.setText(str(exc)); self.operationFailed.emit(str(exc)); return False

    def _record(self):
        index=self._selector.currentIndex()
        if not 0 <= index < len(self.state['records']): raise ValueError('Add and select a recording first.')
        return index,self.state['records'][index]

    def _need_folder(self):
        if self.folder is None: raise ValueError('Create or open a manual workspace first.')
        return self.folder

    def _store_drafts(self):
        if self._loading: return
        if 0 <= self._active_index < len(self.state['records']):
            self.state['records'][self._active_index]['drafts'] = dict(
                source=self._source_form.values(), preparation=self._prep_form.values(), schmitt=self._schmitt_form.values(), null=self._null_form.values(),
                inference=self._inference_form.values(), decode=self._decode_form.values(),
                measurement=self._measure_form.values(), model_path=self._model.text().strip())
        self.state['drafts'] = dict(comparison=self._comparison_form.values(),
            cluster=self._cluster_form.values(), mixture=self._mixture_form.values(),
            mixture_tag=self._mixture_target.currentText(), model_path=self._model.text().strip(),
            study_details=self._study_details_form.values())

    def _settings_edited(self, *args):
        if self._loading: return
        self._store_drafts(); self._update_pending()

    @staticmethod
    def _record_pending(row):
        draft=row.get('drafts', {}); pending=[]
        source=dict(draft.get('source', {}))
        if source:
            source['channel']-=1
            if any(row.get(k)!=v for k,v in source.items()): pending.append('recording details')
        for name, cls, completed in [('preparation', MLPreprocessConfig, 'prepared'),
                                     ('inference', detect.InferenceSettings, 'inference'),
                                     ('decode', detect.DecodeSettings, 'candidates'),
                                     ('measurement', detect.MeasurementSettings, 'analysis')]:
            if (row.get('event_input') == 'main_workspace' and name in ('inference', 'decode')) or (row.get('measurement_method',row.get('event_input')) == 'schmitt' and name in ('inference','decode','measurement')):
                continue
            if name not in draft: continue
            try:
                values=dict(draft[name])
                if name=='preparation':
                    values['notch_freqs']=_numbers(values['notch_freqs'])
                    current=cls(**values, normalize=False).to_dict()
                else: current=asdict(cls(**values))
                if current!=row[name+'_settings']: pending.append(name)
            except (ValueError, TypeError): pending.append(name)
        if row.get('measurement_method',row.get('event_input')) == 'schmitt' and 'schmitt' in draft:
            try:
                if asdict(signal.SchmittSettings(**draft['schmitt'])) != row['schmitt_settings']: pending.append('signal detection')
            except (ValueError, TypeError): pending.append('signal detection')
        if row.get('event_input') not in ('main_workspace','schmitt') and 'inference' in row and draft.get('model_path',row.get('model_path'))!=row.get('model_path'): pending.append('model')
        return pending

    def _update_pending(self):
        pending=[f"{r['tag']}: {', '.join(self._record_pending(r))}" for r in self.state['records'] if self._record_pending(r)]
        if self._study_details_form.values()!=self._study_details(): pending.append('study details')
        for label,key,values in [('comparisons','comparison',self._comparison_values),('pooled clustering','clustering',self._cluster_values),('mixture','mixture',self._mixture_values)]:
            if self.state.get(key) is not None:
                try:
                    settings_key={'comparison':'comparison_settings','clustering':'cluster_settings','mixture':'mixture_settings'}[key]
                    changed=values()!=self.state[settings_key]
                    if key=='mixture': changed=changed or self._mixture_target.currentText()!=self.state.get('mixture_tag')
                    if changed: pending.append(label)
                except (ValueError,TypeError): pending.append(label)
        self._pending.setText('Parameter drafts — '+ '; '.join(pending)+'. Apply or rerun the named stage. Save workspace also preserves these drafts.' if pending else 'Displayed recording settings match the completed stages. Save workspace preserves parameter drafts.')
        if self.state.get('compatibility_notice'): self._pending.setText(self.state['compatibility_notice']+'\n'+self._pending.text())
        self._sync_main_curation()

    def restore_settings(self):
        if self.is_busy(): return
        try: _,row=self._record(); row.pop('drafts',None)
        except ValueError: pass
        self.state.pop('drafts',None)
        self.refresh()

    def _check_source(self, row):
        values=self._source_form.values(); values['channel']-=1
        if any(row.get(k)!=v for k,v in values.items()): raise ValueError('Recording details changed. Apply recording details before preparing or analysing this recording.')

    def _check_inference(self,row):
        if 'inference' not in row: raise ValueError('Run model inference first.')
        if asdict(detect.InferenceSettings(**self._inference_form.values()))!=row['inference_settings'] or self._model.text().strip()!=row.get('model_path'): raise ValueError('Inference settings or model changed. Run model inference first.')

    def _check_committed_records(self):
        self._store_drafts()
        pending=[f"{r['tag']}: {', '.join(self._record_pending(r))}" for r in self.state['records'] if self._record_pending(r)]
        if pending: raise ValueError('Apply or run changed recording stages before downstream analysis: '+'; '.join(pending))

    def _comparison_values(self):
        values=self._comparison_form.values(); values['sensitivity_blocks']=_numbers(values['sensitivity_blocks'],int)
        return asdict(compare.ComparisonSettings(**values))

    def _cluster_values(self):
        values=self._cluster_form.values(); values['feature_columns']=tuple(v.strip() for v in values['feature_columns'].split(',') if v.strip())
        return asdict(compare.ClusterSettings(**values))

    def _mixture_values(self):
        values=self._mixture_form.values(); values['p_grid']=_numbers(values['p_grid'],int) or None; values['k_grid']=_numbers(values['k_grid'],int); values['feature_columns']=tuple(v.strip() for v in values['feature_columns'].split(',') if v.strip())
        return asdict(compare.MixtureSettings(**values))

    def _stage_folder(self,label):
        folder=self._need_folder()/'runs'/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}-{label}-{uuid.uuid4().hex[:8]}'
        folder.mkdir(parents=True); return folder

    def _start(self,label,operation,install):
        if self.is_busy():
            self._status.setText('Wait for the current operation to stop.'); return False
        self._operation_label=label; self._install=install
        worker=_OperationWorker(operation); self._worker=worker
        worker.progress.connect(self._on_progress); worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed); worker.cancelled.connect(self._on_cancelled)
        self._set_busy(True); self._progress.setValue(0); self._status.setText(label+'…')
        self._thread=start_worker_thread(worker,domain=JOB_NONE,label='Manual study: '+label,owner=self)
        return True

    def _stage(self,label,compute,apply,details=None,*,output_folder=None,on_success=None,output_filter=None,preserve_cluster_review=False,require_reviewed=False,cleanup_output_on_failure=False):
        if self.is_busy(): return False
        self._store_drafts()
        folder=self._need_folder(); state=_copy_state(self.state); revision=self._revision
        def record_key(record):
            return ('id',record['recording_id']) if record.get('recording_id') else ('legacy_tag',record['tag'])
        previous_records={record_key(record):record for record in self.state['records']}
        _validate_record_identities(state['records'])
        def compute_stage(progress,cancel):
            checkpoint.assert_workspace_revision(folder,revision)
            if require_reviewed:
                study_review.require_reviewed([r for r in state['records'] if not r.get('comparison_only')])
            check_cancelled(cancel)
            result=compute(progress,cancel)
            check_cancelled(cancel)
            apply(state,result)
            for current in state['records']:
                previous=previous_records.get(record_key(current))
                if (previous is None or current.get('analysis') is not previous.get('analysis') or
                        current.get('prepared') is not previous.get('prepared')):
                    same_review=(preserve_cluster_review and previous is not None
                        and 'analysis' in previous and 'analysis' in current
                        and study_review.review_fingerprint(previous)==study_review.review_fingerprint(current))
                    if not same_review: current.pop(study_review.REVIEW_KEY,None)
                    if not preserve_cluster_review:
                        # A new measurement/import/detection retires old memberships,
                        # even when its numeric output happens to match. Acceptance
                        # edits explicitly preserve the frozen review maps.
                        identity=current.get('recording_id')
                        state['cluster_reviews']={kind:snapshot for kind,snapshot in state.get('cluster_reviews',{}).items()
                            if identity not in set(snapshot['recording_ids'])}
            evidence=details(result) if callable(details) else details
            state['history'].append(dict(operation=label,completed_utc=datetime.now(timezone.utc).isoformat(),settings=evidence or {}))
            views=_review_views(state,cancel)
            progress(.98,'Saving the completed stage…')
            owned_outputs=checkpoint.stage_output_files(output_folder) if output_folder is not None else ()
            if output_filter is not None: owned_outputs=tuple(p for p in owned_outputs if output_filter(p))
            saved_revision=checkpoint.save_workspace(state,folder,cancel=cancel,expected_revision=revision,owned_outputs=owned_outputs)
            return state,saved_revision,views
        def operation(progress,cancel):
            try: return compute_stage(progress,cancel)
            except BaseException:
                if cleanup_output_on_failure and output_folder is not None:
                    shutil.rmtree(output_folder,ignore_errors=True)
                raise
        def install(result):
            self._install_state(result)
            if on_success is not None: on_success()
        return self._start(label,operation,install)

    def _install_state(self,result):
        self.state,self._revision=result[:2]
        self._review_views=result[2] if len(result)>2 else {'records':{},'clusters':{}}
        self.refresh(preserve_table=True)

    def cluster_review_view(self,kind):
        snapshot=self.state.get('cluster_reviews',{}).get(kind)
        view=self._review_views['clusters'].get(kind,{})
        if snapshot is None or view.get('snapshot_id')!=snapshot.get('snapshot_id'):
            return {'current':False,'error':'Load saved clusters for review, or reopen the workspace to inspect its saved review.'}
        keys={r['recording_id']:_review_record_key(r) for r in self.state['records']
              if r.get('recording_id') in snapshot.get('recording_ids',[])}
        if keys!=view.get('record_keys'):
            return {'current':False,'error':'Cluster membership changed or is stale; reload the saved study before review.'}
        return view

    def _review_is_current(self,row):
        key,current=self._review_views['records'].get(row.get('recording_id'),(None,False))
        return key==_review_record_key(row) and current

    def new_workspace(self,folder,*,profile='general',profile_id=None,name='',sample='',condition_label=None):
        if self.is_busy(): raise ValueError('Wait for the current operation to stop.')
        folder=Path(folder).expanduser().resolve()
        if folder.exists() and any(not checkpoint._metadata_file(p.name) for p in folder.iterdir()): raise ValueError('A new workspace needs an empty folder.')
        from .incubation_profiles import initialize_study_metadata
        state=initialize_study_metadata(_empty_state(),profile=profile_id or profile,
            name=name or '',sample=sample,condition_label=condition_label)
        revision=checkpoint.save_workspace(state,folder,expected_revision=None)
        self.folder=folder; self.state=state; self._revision=revision; self._review_views={'records':{},'clusters':{}}; self._active_index=-1; self.refresh(); return True

    def _study_details(self):
        return dict(name=self.state.get('name','Incubation study'),sample=self.state.get('sample',''),
            condition_label=self.state.get('condition_label',''))

    def apply_study_details(self):
        def action():
            values={key:str(value).strip() for key,value in self._study_details_form.values().items()}
            if not values['name']: raise ValueError('Enter a study name.')
            def apply(state,result):
                state.update(values)
                drafts=dict(state.get('drafts',{}));drafts.pop('study_details',None);state['drafts']=drafts
            return self._stage('Apply study details',lambda p,c:None,apply,values)
        return self._guard(action)

    def open_workspace(self,folder):
        folder=Path(folder).expanduser().resolve()
        if not self.is_busy(): self._store_drafts()
        def operation(progress,cancel):
            state,revision=checkpoint.load_workspace(folder,cancel=cancel,with_revision=True)
            state=_validated_workspace_state(state)
            return state,revision,_review_views(state,cancel)
        def install(result):
            old=(self.state,self.folder,self._revision,self._selector.currentIndex(),self._table_view,self._results.currentWidget(),self._review_views)
            try:
                self.state,self._revision,self._review_views=result; self.folder=folder; self._active_index=-1
                self.refresh()
            except Exception:
                self.state,self.folder,self._revision,index,view,widget,self._review_views=old
                self._loading=False; self._active_index=-1
                self.refresh(); self._selector.setCurrentIndex(index)
                self._restore_table(view); self._results.setCurrentWidget(widget)
                raise
        return self._start('Open saved workspace',operation,install)

    def save(self):
        return self._guard(lambda:self._stage('Save workspace',lambda p,c:None,lambda s,r:None))

    def add_raw(self,paths):
        paths=[Path(p).expanduser().resolve() for p in paths]
        if len(set(paths))!=len(paths): raise ValueError('The same recording was selected twice.')
        if any(p==r['source_path'] for p in paths for r in self.state['records']): raise ValueError('A selected recording is already included.')
        def compute(progress,cancel):
            return [(path,prep.inspect_raw(path,cancel=cancel)) for path in paths]
        def apply(state,items):
            from .incubation_profiles import recording_defaults, saved_profile
            defaults=recording_defaults(saved_profile(state))
            tags={r['tag'] for r in state['records']}
            for path,info in items:
                number=1
                while f'recording_{number}' in tags: number+=1
                tag=f'recording_{number}'; tags.add(tag)
                state['records'].append(dict(recording_id=uuid.uuid4().hex,source_path=path,source_info=info,tag=tag,incubation_h=0.,pore_id=tag,pore_diameter_nm=1.,concentration_pm=0.,voltage_mv=0.,channel=0,event_polarity=1,reader_scaling='notebook_float32',preparation_settings=_prep_defaults(),inference_settings=asdict(detect.InferenceSettings()),decode_settings=asdict(detect.DecodeSettings()),measurement_settings=asdict(detect.MeasurementSettings()),**defaults))
            for row in state['records']: _source_defaults(row)
            _invalidate_global(state)
        return self._stage('Add raw recordings',compute,apply)

    def apply_recording_details(self):
        def action():
            index,record=self._record(); values=self._source_form.values(); values['channel']-=1
            signal.SourceOptions(**{k:values[k] for k in asdict(signal.SourceOptions())})
            tag=values['tag']
            if not tag or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in tag): raise ValueError('Use a nonempty tag containing only letters, digits, underscore or hyphen.')
            if any(r['tag']==tag for i,r in enumerate(self.state['records']) if i!=index): raise ValueError('Recording tags must be unique.')
            if values['channel']<0 or values['incubation_h']<0 or values['pore_diameter_nm']<=0 or values['concentration_pm']<0 or not values['pore_id']: raise ValueError('Check the channel, incubation, pore identity/diameter and concentration.')
            def apply(state,result):
                if all(record.get(k)==v for k,v in values.items()): return
                changed=any(record.get(k)!=values[k] for k in ('tag','channel','event_polarity','reader_scaling',*asdict(signal.SourceOptions())))
                row=state['records'][index]; row.update(values)
                if changed:
                    for key in ('prepared','inference','candidates','analysis','event_input','imported_events','imported_event_provenance','main_workspace_bridge','signal_nulls','measurement_method'): row.pop(key,None)
                _invalidate_global(state)
            return self._stage('Apply recording details',lambda p,c:None,apply,values)
        return self._guard(action)

    def remove_recording(self):
        def action():
            index,_=self._record()
            def apply(state,result): state['records'].pop(index); _invalidate_global(state)
            return self._stage('Remove recording',lambda p,c:None,apply)
        return self._guard(action)

    def preparation_config(self):
        values=self._prep_form.values(); values['notch_freqs']=_numbers(values['notch_freqs'])
        return MLPreprocessConfig(**values,normalize=False)

    def prepare_selected(self):
        def action():
            index,row=self._record(); self._check_source(row); config=self.preparation_config()
            def compute(progress,cancel): return prep.prepare_raw(row['source_path'],row['channel'],row['tag'],config,event_polarity=row['event_polarity'],reader_scaling=row['reader_scaling'],source_options=_signal_source(row),progress=progress,cancel=cancel)
            def apply(state,result):
                record=state['records'][index]; record['prepared']=result; record['preparation_settings']=config.to_dict()
                for key in ('inference','candidates','analysis','event_input','imported_events','imported_event_provenance','main_workspace_bridge','signal_nulls','measurement_method'): record.pop(key,None)
                _invalidate_global(state)
            return self._stage('Prepare '+row['tag'],compute,apply,config.to_dict())
        return self._guard(action)

    def _prepared(self):
        index,row=self._record()
        self._check_source(row)
        if 'prepared' not in row: raise ValueError('Prepare the selected recording first.')
        if self.preparation_config().to_dict()!=row['preparation_settings']: raise ValueError('Preparation settings have changed. Run Prepare selected recording first, or choose Restore completed settings.')
        return index,row

    def run_inference(self):
        def action():
            index,row=self._prepared(); settings=detect.InferenceSettings(**self._inference_form.values()); model=self._model.text().strip()
            if not model: raise ValueError('Choose the supplied trained model first.')
            def apply(state,result):
                record=state['records'][index]; record['inference']=result; record['inference_settings']=asdict(settings); record['model_path']=model; state['model_path']=model
                for key in ('candidates','analysis','event_input','imported_events','imported_event_provenance','main_workspace_bridge','signal_nulls','measurement_method'): record.pop(key,None)
                _invalidate_global(state)
            return self._stage('Infer '+row['tag'],lambda p,c:detect.run_inference(row['prepared'],model,settings,progress=p,cancel=c),apply,asdict(settings))
        return self._guard(action)

    def find_candidates(self):
        def action():
            index,row=self._prepared()
            if 'inference' not in row: raise ValueError('Run model inference first.')
            if asdict(detect.InferenceSettings(**self._inference_form.values()))!=row['inference_settings'] or self._model.text().strip()!=row.get('model_path'): raise ValueError('Inference settings or model changed. Run model inference first.')
            settings=detect.DecodeSettings(**self._decode_form.values())
            def apply(state,result):
                record=state['records'][index]; record['candidates']=result; record['decode_settings']=asdict(settings); record.pop('analysis',None); _invalidate_global(state)
                for key in ('event_input','imported_events','imported_event_provenance','main_workspace_bridge','signal_nulls','measurement_method'): record.pop(key,None)
            return self._stage('Find candidates in '+row['tag'],lambda p,c:detect.find_candidates(row['prepared'],row['inference'],settings,cancel=c),apply,asdict(settings))
        return self._guard(action)

    def measure_candidates(self):
        def action():
            index,row=self._prepared()
            if row.get('event_input') == 'main_workspace':
                return self._measure_main_snapshot(index,row,dict(events=row['imported_events'],provenance=row['imported_event_provenance']),preserve_review=True)
            if row.get('event_input') == 'schmitt':
                raise ValueError('Use Detect and measure signal events to apply signal settings; this button belongs to the model/import route.')
            self._check_inference(row)
            if 'candidates' not in row: raise ValueError('Find candidate events first.')
            if asdict(detect.DecodeSettings(**self._decode_form.values()))!=row['decode_settings']: raise ValueError('Candidate settings changed. Find candidate events first.')
            settings=detect.MeasurementSettings(**self._measure_form.values())
            def apply(state,result):
                state['records'][index]['analysis']=result; state['records'][index]['measurement_settings']=asdict(settings); _invalidate_global(state)
            return self._stage('Measure and gate '+row['tag'],lambda p,c:detect.measure_candidates(row['prepared'],row['candidates'],settings,progress=p,cancel=c),apply,asdict(settings))
        return self._guard(action)

    def run_signal_detection(self):
        def action():
            index,row=self._prepared(); settings=signal.SchmittSettings(**self._schmitt_form.values())
            def apply(state,result):
                record=state['records'][index]; record['analysis']=result
                record['candidates']=result.catalog[['start_idx','end_idx']].copy()
                record['event_input']='schmitt'; record['measurement_method']='schmitt'; record['schmitt_settings']=asdict(settings)
                for key in ('inference','imported_events','imported_event_provenance','main_workspace_bridge','signal_nulls'): record.pop(key,None)
                _invalidate_global(state)
            return self._stage('Signal detection: '+row['tag'],lambda p,c:signal.analyze(row['prepared'],settings,progress=p,cancel=c),apply,asdict(settings))
        return self._guard(action)

    def run_signal_nulls(self):
        def action():
            index,row=self._prepared(); self._check_committed_records()
            if 'analysis' not in row or 'schmitt_settings' not in row['analysis'].provenance: raise ValueError('Run signal detection first.')
            settings=signal.NullSettings(**self._null_form.values()); folder=self._stage_folder('signal-nulls')
            def compute(p,c):
                result=signal.null_controls(row['analysis'],settings,progress=p,cancel=c)
                result.to_csv(folder/(row['tag']+'_null_counts.csv'),index=False)
                return result
            def apply(state,result):
                state['records'][index]['signal_nulls']=result
                state['records'][index]['null_settings']=asdict(settings)
                _invalidate_global(state)
            return self._stage('Signal null controls: '+row['tag'],compute,apply,asdict(settings),output_folder=folder)
        return self._guard(action)

    def show_signal_nulls(self):
        def action():
            _,row=self._record()
            if 'signal_nulls' not in row: raise ValueError('Run signal null controls first.')
            self._display_table(row['signal_nulls'],('signal_nulls',row['tag']))
        return self._guard(action)

    def _main_workspace(self, method):
        owner=self.parent()
        callback=getattr(owner,method,None)
        if not callable(callback):
            raise ValueError('Open this study from NanoScope’s Analysis menu to connect it to the main analysis workspace.')
        return callback

    def open_main_analysis(self):
        """Open these study events in the application's shared curation tools."""
        def action():
            if self.is_busy(): return False
            self._store_drafts()
            index,row=self._prepared(); open_review=self._main_workspace('open_study_review')
            owner=self.parent(); launch_doc_ids=frozenset(doc.doc_id for doc in owner._session.documents())
            if self._record_pending(row):
                raise ValueError('Apply or restore changed recording settings before opening its detections in the main workspace.')
            from .study_event_bridge import prepare_review_export, catalog_signature, _assert_idle
            connected=callable(getattr(owner,'begin_study_curation',None))
            if connected:
                _assert_idle(owner)
                descriptor=row.get('main_workspace_bridge') or {}
                if descriptor.get('catalog_signature')==catalog_signature(row):
                    # One review copy may be open under another (case) spelling than the
                    # descriptor's; same_source asks the filesystem, resolve() strings do not.
                    for doc in owner._session.documents():
                        if descriptor.get('path') and same_source(doc.source_key,str(Path(descriptor['path']).expanduser().resolve())):
                            owner.begin_study_curation(self,row['tag'],doc.doc_id)
                            owner.show_study_curation(); self.hide(); return True
            session=owner._session; generation=getattr(owner,'_study_curation_project_generation',None)
            folder=self._stage_folder('main-workspace-review'); payload=[]
            def compute(progress,cancel):
                progress(.05,'Preparing a recording copy for the main analysis tools…')
                result=prepare_review_export(row,folder,cancel=cancel)
                payload.append(result)
                return result['descriptor']
            def apply(state,descriptor):
                next(r for r in state['records'] if r['recording_id']==row['recording_id'])['main_workspace_bridge']=descriptor
            def opened():
                if owner._session is not session or getattr(owner,'_study_curation_project_generation',None)!=generation:
                    raise ValueError('The main project changed while these events were opening. Open them again from the study.')
                doc_id=open_review(payload.pop(),launch_doc_ids)
                if connected:
                    owner.begin_study_curation(self,row['tag'],doc_id,fresh=True)
                    self.hide()
            return self._stage('Open '+row['tag']+' in main analysis workspace',compute,apply,
                dict(recording_id=row['recording_id']),output_folder=folder,
                on_success=opened,cleanup_output_on_failure=True,
                # The native calibration sidecar is an editable preference for
                # this review copy. Its immutable prepared samples are owned;
                # importing still verifies calibration, units and file identity.
                output_filter=lambda path:Path(path).name!='prepared-review.h5.npcal.json')
        return self._guard(action)

    def _curation_record(self,context,*,verify_current=False):
        if Path(context['folder'])!=self.folder:
            raise ValueError('The study workspace changed. Open this recording from its current study before saving curation.')
        row=next((r for r in self.state['records'] if r['tag']==context['tag']
            and r.get('recording_id')==context['recording_id']),None)
        if row is None:
            raise ValueError('The study recording changed. Open it from the study again before saving curation.')
        descriptor=row.get('main_workspace_bridge') or {}
        if descriptor.get('path')!=context['path']:
            raise ValueError('This review belongs to an earlier study state. Open the current study events before saving.')
        if verify_current:
            from .study_event_bridge import catalog_signature
            if descriptor.get('catalog_signature')!=catalog_signature(row) or self._record_pending(row):
                raise ValueError('The study changed since these events were opened. Return to study and open its current events.')
        return row

    def save_main_curation(self,context):
        """Save the captured recording with its committed measurement settings."""
        if self._curation_closed: raise ValueError('Return to the study to reopen it before saving curation.')
        if self.is_busy(): raise ValueError('Wait for the current study operation to finish.')
        self._store_drafts(); row=self._curation_record(context,verify_current=True)
        owner=self.parent()
        if (owner._session is not context['session']
                or owner._study_curation_project_generation!=context['project_generation']
                or not owner._session.has_document(context['doc_id'])):
            raise ValueError('The main workspace recording changed. Open the study events again before saving.')
        result=owner._analysis_panel.current_detection()
        if result is None or result.doc_id!=context['doc_id']:
            raise ValueError('Show this study recording’s events before saving curation.')
        snapshot=owner.snapshot_study_events(row)
        captured_signature=owner._study_curation_signature(result)
        context.pop('error',None)
        def saved():
            context['saved_signature']=captured_signature
            context['message']='Curation saved to study. Return to study and use reviewed events for analysis.'
            owner._sync_study_curation_bar()
            owner.statusBar().showMessage('Curation saved for '+row['tag']+'. Return to study and use reviewed events for analysis.',20000)
        index=next(i for i,r in enumerate(self.state['records']) if r.get('recording_id')==row['recording_id'])
        return self._measure_main_snapshot(index,row,snapshot,use_committed=True,on_success=saved,
            label='Save curation for '+row['tag'])

    def return_to_curation_record(self,context):
        row=self._curation_record(context)
        if not self.is_busy():
            self._store_drafts()
            index=next(i for i,r in enumerate(self.state['records']) if r.get('recording_id')==row['recording_id'])
            self._selector.setCurrentIndex(index); self._stages.setCurrentIndex(2)
            if context.get('pending'):
                self._status.setText('Main workspace curation has unsaved changes. Open Curate in main workspace and use Save curation to study to include them in analysis.')
        self.show(); self.raise_(); self.activateWindow()

    def use_main_detections(self):
        """Capture curation now; later main-window edits require an explicit update."""
        def action():
            if self.is_busy(): return False
            index,row=self._prepared()
            snapshot=self._main_workspace('snapshot_study_events')(row)
            return self._measure_main_snapshot(index,row,snapshot)
        return self._guard(action)

    def _measure_main_snapshot(self,index,row,snapshot,*,preserve_review=False,use_committed=False,on_success=None,label=None):
        settings=detect.MeasurementSettings(**(row['measurement_settings'] if use_committed else self._measure_form.values()))
        events=snapshot['events'].copy(deep=True); events.attrs=deepcopy(snapshot['events'].attrs)
        provenance=deepcopy(snapshot['provenance'])
        signal_settings=(signal.SchmittSettings(**(row.get('schmitt_settings',asdict(signal.SchmittSettings()))
            if use_committed else self._schmitt_form.values()))
            if row.get('measurement_method',row.get('event_input'))=='schmitt' else None)
        previous_analysis=row.get('analysis') if preserve_review else None
        def compute(progress,cancel):
            result=detect.measure_imported_events(row['prepared'],events,settings,
                source_provenance=provenance,progress=progress,cancel=cancel)
            if signal_settings is not None:
                result=signal.analyze(row['prepared'],signal_settings,imported=result,progress=progress,cancel=cancel)
            if previous_analysis is not None:
                result=study_review.retain_event_decisions(previous_analysis,result)
            return result
        def apply(state,result):
            record=next(r for r in state['records'] if r.get('recording_id')==row.get('recording_id') and r['tag']==row['tag'])
            record.update(event_input='main_workspace',imported_events=events,
                imported_event_provenance=provenance,analysis=result,
                candidates=result.catalog[['event_id','start_idx','end_idx']].copy(),
                measurement_settings=asdict(settings))
            if signal_settings is not None:
                record['measurement_method']='schmitt'; record['schmitt_settings']=asdict(signal_settings)
            descriptor=record.get('main_workspace_bridge')
            # The snapshot spells the review copy as the open document does, the descriptor
            # as the export did; same_source, not ==, so one file is one file.
            if isinstance(descriptor,dict) and same_source(provenance.get('source_path'),descriptor.get('path')):
                from .study_event_bridge import catalog_signature
                record['main_workspace_bridge']={**descriptor,'catalog_signature':catalog_signature(record)}
            _invalidate_global(state)
        def installed():
            if on_success is not None: on_success()
            else:
                owner=self.parent()
                for context in getattr(owner,'_study_curation_sessions',{}).values():
                    if (context['study'] is self and context['recording_id']==row.get('recording_id')
                            and same_source(provenance.get('source_path'),context['path'])):
                        from .study_event_bridge import catalog_signature
                        context['saved_signature']=(provenance.get('source_module_id'),catalog_signature({'candidates':events}))
                        context.pop('error',None)
                        context['message']='Curation saved to study. Return to study and use reviewed events for analysis.'
                self._sync_main_curation()
        return self._stage(label or 'Use curated detections for '+row['tag'],compute,apply,
            dict(measurement_settings=asdict(settings),source=provenance),on_success=installed)

    def _analyses(self):
        self._store_drafts()
        if not self.state['records'] or any('analysis' not in r for r in self.state['records']): raise ValueError('Measure and gate every included recording first.')
        self._check_committed_records()
        primary=[r for r in self.state['records'] if not r.get('comparison_only')]
        if not primary: raise ValueError('Include at least one primary recording in the series.')
        return [r['analysis'] for r in primary],[_condition(r) for r in primary]

    def _review(self,ids,value):
        index,row=self._record()
        if 'analysis' not in row: raise ValueError('Measure the candidates first.')
        analysis=row['analysis']
        def compute(progress,cancel):
            return study_review.apply_event_decision(analysis,ids,value)
        def apply(state,result):
            state['records'][index]['analysis']=result
            if not result.catalog.accepted.equals(analysis.catalog.accepted): _invalidate_global(state)
        return self._stage('Review '+row['tag'],compute,apply,dict(event_ids=list(ids),accepted=value),preserve_cluster_review=True)

    def review_selected(self,value):
        def action():
            frame=self._table_model.frame
            selected=self._table.selectionModel().selectedRows()
            if not selected or 'event_id' not in frame: raise ValueError('Show all candidates, then select event rows in the results table.')
            _,record=self._record()
            if (frame.attrs.get('manual_record_tag')!=record['tag'] or
                    frame.attrs.get('manual_record_id')!=record.get('recording_id')): raise ValueError('Show the selected recording\'s candidate table before reviewing it.')
            return self._review([frame.iloc[i.row()].event_id for i in selected],value)
        return self._guard(action)

    def reset_review(self): return self._guard(lambda:self._review([],None))

    def confirm_event_review(self):
        def action():
            index,row=self._record()
            self._check_committed_records()
            if 'analysis' not in row: raise ValueError('Measure the candidates first.')
            def apply(state,result): state['records'][index][study_review.REVIEW_KEY]=result
            return self._stage('Confirm event review: '+row['tag'],lambda p,c:study_review.confirm_review(row),apply,lambda result:result)
        return self._guard(action)

    def _require_reviewed(self,records=None):
        records=records if records is not None else [r for r in self.state['records'] if not r.get('comparison_only')]
        pending=[r['tag'] for r in records if not self._review_is_current(r)]
        if pending: raise ValueError('Confirm the reviewed events before running further analysis: '+', '.join(pending)+'. In Detect / review, choose Use reviewed events for analysis.')

    @staticmethod
    def _capture_cluster_review(state,kind):
        result=state['clustering' if kind=='pooled' else 'mixture']
        if result is None: raise ValueError('Run the selected clustering or mixture analysis first.')
        metadata=result[-1]
        if kind=='mixture': metadata={**metadata,'mixture_tag':state.get('mixture_tag')}
        return study_cluster_review.create_snapshot(state['records'],result[0],metadata,kind=kind,
            settings=state['cluster_settings' if kind=='pooled' else 'mixture_settings'])

    def load_cluster_review(self):
        def action():
            self._store_drafts();self._check_committed_records()
            kind=self._cluster_review.kind()
            source=self.state
            def apply(state,result):
                state['cluster_reviews']={**state.get('cluster_reviews',{}),kind:result}
            return self._stage('Load saved clusters for review: '+kind,lambda p,c:self._capture_cluster_review(source,kind),apply,
                dict(kind=kind),on_success=self._cluster_review.refresh)
        return self._guard(action)

    def _selected_cluster_review(self,*,selection=True):
        self._store_drafts();self._check_committed_records()
        snapshot=self._cluster_review.snapshot()
        if snapshot is None: raise ValueError('Load saved clusters for review first.')
        view=self.cluster_review_view(self._cluster_review.kind())
        if not view.get('current'): raise ValueError(view.get('error','Cluster review is stale; load saved clusters again.'))
        labels=self._cluster_review.selected_labels()
        if selection and not labels: raise ValueError('Select one or more rows in the cluster acceptance summary.')
        return snapshot,self._cluster_review.method.currentData(),labels

    def review_clusters(self,accepted,*,only=False):
        def action():
            snapshot,method,labels=self._selected_cluster_review()
            records=self.state['records']
            def compute(progress,cancel):
                return study_cluster_review.apply_decision(snapshot,records,method,labels,accepted,only=only)
            def apply(state,result):
                state['records'],details=result
                if details['changed_count']: _invalidate_global(state)
            return self._stage('Review clusters: '+('keep only selected' if only else 'accept' if accepted else 'reject'),
                compute,apply,lambda result:result[1],preserve_cluster_review=True)
        return self._guard(action)

    def show_cluster_members(self):
        def action():
            snapshot,method,labels=self._selected_cluster_review()
            # Display uses the worker-validated cache; mutations validate the full
            # scientific evidence independently inside their worker transaction.
            membership=snapshot['memberships']
            selected={identity:group.event_id.tolist() for identity,group in
                membership.loc[membership[method].isin(labels)].groupby('recording_id',sort=False)}
            frames=[]
            for record in self.state['records']:
                ids=selected.get(record['recording_id'],[])
                if not ids: continue
                frame=record['analysis'].catalog
                frame=frame.loc[frame.event_id.isin(ids)].copy(deep=True)
                frame['tag']=record['tag']
                membership=snapshot['memberships']
                grouped=membership.loc[membership.recording_id==record['recording_id']].set_index('event_id')
                frame['review_cluster']=frame.event_id.map(grouped[method])
                first=['tag','event_id','review_cluster','accepted','passes_gate','start_idx','end_idx']
                frame=frame[[c for c in first if c in frame]+[c for c in frame if c not in first]]
                frames.append(frame)
            frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
            self._display_table(frame,('cluster_members',(self._cluster_review.kind(),method,labels)))
        return self._guard(action)

    def confirm_cluster_review(self):
        def action():
            snapshot,method,_=self._selected_cluster_review(selection=False)
            identities=set(snapshot['recording_ids'])
            records=self.state['records']
            def compute(progress,cancel):
                study_cluster_review.require_current(snapshot,records)
                return {r['recording_id']:study_review.confirm_review(r) for r in records
                        if r['recording_id'] in identities}
            def apply(state,receipts):
                for row in state['records']:
                    if row['recording_id'] in receipts:row[study_review.REVIEW_KEY]=receipts[row['recording_id']]
            return self._stage('Confirm cluster-reviewed events for analysis',compute,apply,
                dict(kind=self._cluster_review.kind(),method=method,recordings=sorted(identities)),
                on_success=lambda:self._status.setText('Cluster-reviewed events confirmed. Recompute comparisons, or extract reviewed-event features and run the next population analysis.'))
        return self._guard(action)

    def run_comparisons(self):
        def action():
            self._require_reviewed()
            analyses,records=self._analyses(); values=self._comparison_form.values(); values['sensitivity_blocks']=_numbers(values['sensitivity_blocks'],int); settings=compare.ComparisonSettings(**values); folder=self._stage_folder('comparisons')
            def apply(state,result): state['comparison']=result; state['comparison_settings']=asdict(settings); state['comparison_folder']=folder; state['figures']=[]
            engine=signal_comparisons if any('schmitt_settings' in a.provenance for a in analyses) else compare
            return self._stage('Compute recording comparisons',lambda p,c:engine.compare_recordings(analyses,records,settings,folder=folder,progress=p,cancel=c),apply,asdict(settings),output_folder=folder,require_reviewed=True)
        return self._guard(action)

    def extract_features(self):
        def action():
            self._require_reviewed()
            analyses,records=self._analyses()
            def apply(state,result):
                state['features']=result; state['clustering']=state['mixture']=None; state['figures']=[]
                state.pop('compatibility_notice',None)
            return self._stage('Extract waveform features',lambda p,c:compare.extract_features(analyses,records,progress=p,cancel=c),apply,require_reviewed=True)
        return self._guard(action)

    def run_clustering(self):
        def action():
            self._require_reviewed()
            self._check_committed_records()
            if self.state['features'] is None: raise ValueError('Extract reviewed-event waveform features first.')
            values=self._cluster_form.values(); values['feature_columns']=tuple(v.strip() for v in values['feature_columns'].split(',') if v.strip()); settings=compare.ClusterSettings(**values); folder=self._stage_folder('clustering'); features=self.state['features']
            def apply(state,result):
                state['clustering']=result; state['cluster_settings']=asdict(settings); state['cluster_folder']=folder; state['figures']=[]
                reviews={**state.get('cluster_reviews',{})};reviews.pop('pooled',None)
                state['cluster_reviews']=reviews
                if 'hdbscan' in result[0] and result[1].get('kmeans_column') in result[0]:
                    reviews['pooled']=self._capture_cluster_review(state,'pooled')
            return self._stage('Run pooled exploratory clustering',lambda p,c:compare.cluster_features(features,settings,folder=folder,progress=p,cancel=c),apply,asdict(settings),output_folder=folder,require_reviewed=True)
        return self._guard(action)

    def run_mixture(self):
        def action():
            self._require_reviewed()
            self._check_committed_records()
            if self.state['features'] is None: raise ValueError('Extract reviewed-event waveform features first.')
            index=self._mixture_target.currentIndex()
            if index<0: raise ValueError('Select the mixture target recording.')
            record=self.state['records'][index]; values=self._mixture_form.values(); values['p_grid']=_numbers(values['p_grid'],int) or None; values['k_grid']=_numbers(values['k_grid'],int); values['feature_columns']=tuple(v.strip() for v in values['feature_columns'].split(',') if v.strip()); settings=compare.MixtureSettings(**values)
            folder=self._stage_folder('mixture'); features=self.state['features']
            def apply(state,result):
                state['mixture']=result; state['mixture_settings']=asdict(settings); state['mixture_tag']=record['tag']; state['mixture_folder']=folder; state['figures']=[]
                reviews={**state.get('cluster_reviews',{})};reviews.pop('mixture',None)
                state['cluster_reviews']=reviews
                if 'subpop' in result[0] and result[1] is not None:
                    reviews['mixture']=self._capture_cluster_review(state,'mixture')
            return self._stage('Fit mixture: '+record['tag'],lambda p,c:compare.fit_mixture(features,record['analysis'],settings,folder=folder,progress=p,cancel=c),apply,asdict(settings),output_folder=folder,require_reviewed=True)
        return self._guard(action)

    def render_figures(self):
        def action():
            self._store_drafts(); self._check_committed_records()
            group=self._figure_group.currentIndex(); folder=self._stage_folder('figures'); state=self.state
            if group==0:
                _,record=self._record()
                if 'analysis' not in record: raise ValueError('Measure and gate the selected recording first.')
                all_events,accepted,_=compare.reviewed_catalog(record['analysis'],_condition(record))
                rendered=replace(record['analysis'],catalog=all_events)
                compute=(lambda p,c:signal_study_figures.recording_figures(rendered,record.get('signal_nulls'),folder,cancel=c,study_metadata=state)) if 'schmitt_settings' in rendered.provenance else (lambda p,c:figures.recording_figures(rendered,accepted,folder,cancel=c))
            elif group==1:
                result=state['comparison']
                if result is None: raise ValueError('Compute recording comparisons first.')
                if self._comparison_values()!=state['comparison_settings']: raise ValueError('Comparison settings changed. Compute recording comparisons first, or restore completed settings.')
                compute=(lambda p,c:signal_study_figures.comparison_figures(result,state['records'],folder,cancel=c,study_metadata=state)) if result.metadata.get('signal_detector') else (lambda p,c:figures.comparison_figures(result.events,result.summary,result.tables,folder,cancel=c,settings=compare.ComparisonSettings(**state['comparison_settings'])))
            elif group==2:
                result=state['clustering']
                if result is None: raise ValueError('Run pooled exploratory clustering first.')
                if self._cluster_values()!=state['cluster_settings']: raise ValueError('Clustering settings changed. Run pooled exploratory clustering first, or restore completed settings.')
                compute=lambda p,c:figures.clustering_figures(result[0],folder,prefix='pooled',cancel=c,settings=compare.ClusterSettings(**state['cluster_settings']))
            else:
                result=state['mixture']
                if result is None: raise ValueError('Fit a selected-recording mixture first.')
                if self._mixture_values()!=state['mixture_settings'] or self._mixture_target.currentText()!=state['mixture_tag']: raise ValueError('Mixture settings or target changed. Fit the mixture first, or restore completed settings.')
                record=next(r for r in state['records'] if r['tag']==state['mixture_tag'])
                all_events,_,_=compare.reviewed_catalog(record['analysis'],_condition(record))
                rendered=replace(record['analysis'],catalog=all_events)
                compute=lambda p,c:figures.appendix_figures(result[0],result[1],rendered,state['mixture_folder'],folder,cancel=c)
            recipe=dict(group=group)
            if group in (0,3): recipe['tag']=record['tag']
            original_compute=compute
            def compute(progress,cancel):
                from ..plot.study_figure_style import styled_figures
                with styled_figures(state.get('figure_styles',{})):
                    return original_compute(progress,cancel)
            def apply(current,paths):
                names={Path(p).name for p in paths}; current['figures']=[p for p in current['figures'] if Path(p).name not in names]+[Path(p) for p in paths]
                current['figure_recipes']={**current.get('figure_recipes',{}),**{Path(p).stem:dict(recipe) for p in paths}}
            return self._stage('Generate '+self._figure_group.currentText(),compute,apply,output_folder=folder)
        return self._guard(action)

    def style_selected_figure(self):
        def action():
            if self.is_busy(): return False
            index=self._figure_selector.currentIndex()
            if not 0<=index<len(self.state['figures']):raise ValueError('Select a saved study figure first.')
            from .study_figure_recipe import reconstruct
            from .study_figure_window import StudyFigureWindow
            path=self.state['figures'][index]; state=self.state
            folder=self.folder; expected=[self._revision,path]
            def compute(progress,cancel):
                progress(.05,'Reconstructing the canvas from saved study results…')
                return reconstruct(state,path,cancel=cancel)
            def install(result):
                figure,data,resources,recipe=result
                provenance=dict(study=state['name'],workspace=str(folder),workspace_revision=expected[0],
                    figure=path.stem,recipe=recipe,source_records=[dict(tag=r['tag'],path=str(r['source_path']),
                        preparation=r.get('prepared').provenance if r.get('prepared') is not None else {},
                        review=deepcopy(r.get(study_review.REVIEW_KEY)),
                        measurement=deepcopy(r['analysis'].provenance) if 'analysis' in r else {}) for r in state['records']],
                    reconstruction='Original plot builders from committed results; plot-side densities and display envelopes reconstructed.')
                def save_style(style,snapshot,editor):
                    if self.is_busy():raise ValueError('Wait for the current study operation to finish.')
                    if self.folder!=folder or self._revision!=expected[0] or expected[1] not in self.state['figures']:
                        raise ValueError('The study changed while this figure was open. Reopen its figure editor before saving styling.')
                    destination=self._stage_folder('figure-style'); target=destination/path.name
                    def write(progress,cancel):
                        from .figure_bundle import write_figure_bundle
                        for name,value in resources.items():(destination/name).write_bytes(value)
                        target.write_bytes(snapshot.png_bytes)
                        target.with_suffix('.svg').write_bytes(snapshot.svg_bytes)
                        target.with_suffix('.style.json').write_text(json.dumps(style,indent=2))
                        write_figure_bundle(target.with_suffix('.figure.zip'),snapshot)
                    def apply(current,result):
                        current['figures']=[target if p==expected[1] else p for p in current['figures']]
                        current['figure_styles']={**current.get('figure_styles',{}),path.stem:deepcopy(style)}
                        current['figure_recipes']={**current.get('figure_recipes',{}),path.stem:dict(recipe)}
                    def done():
                        from shiboken6 import isValid
                        expected[:]=[self._revision,target]
                        if isValid(editor):editor.saved()
                        self._figure_selector.setCurrentIndex(self.state['figures'].index(target));self.show_figure(self._figure_selector.currentIndex())
                    return self._stage('Save figure styling: '+path.stem,write,apply,style,output_folder=destination,on_success=done)
                # Closing an editor only hides it; dispose of the closed ones on our own
                # schedule, here, where no save can still call back into one.
                for old in self.findChildren(StudyFigureWindow,options=QtCore.Qt.FindDirectChildrenOnly):
                    if old.isHidden():old.deleteLater()
                editor=StudyFigureWindow(figure,data,resources,name=path.stem,provenance=provenance,
                    style=state.get('figure_styles',{}).get(path.stem),save_style=save_style,
                    library_owner=self.parent(),parent=self)
                self._figure_editor=editor;editor.show()
            return self._start('Open figure styling',compute,install)
        return self._guard(action)

    def show_candidates(self,*,reset_filter=True):
        def action():
            if reset_filter:
                self._review_filter.blockSignals(True); self._review_filter.setCurrentIndex(0); self._review_filter.blockSignals(False)
            _,row=self._record()
            data=row['analysis'].catalog.copy(deep=False) if 'analysis' in row else row.get('candidates',pd.DataFrame())
            if 'accepted' in data and self._review_filter.currentIndex():
                data=data.loc[data.accepted==bool(self._review_filter.currentIndex()==1)]
            first=[key for key in ('event_id','accepted','passes_gate','kind','onset_s','dwell_ms',
                'depth_sust_pA','depth_peak_pA','start_idx','end_idx') if key in data]
            data=data[first+[key for key in data if key not in first]]
            data=data.copy(deep=False)
            data.attrs=dict(data.attrs,manual_record_tag=row['tag'],manual_record_id=row.get('recording_id'))
            self._display_table(data,('candidates',row.get('recording_id')))
        return self._guard(action)

    def show_comparison(self,*args):
        result=self.state['comparison']
        if result is None: return
        name=self._comparison_table.currentText(); frame=result.summary if name=='Summary' else result.events if name=='Accepted events' else result.tables.get(name,pd.DataFrame())
        self._display_table(frame,('comparison',name))

    def show_features(self):
        if self.state['features'] is not None: self._display_table(self.state['features'],('features',None))

    def show_clustering(self):
        if self.state['clustering'] is not None: self._display_table(self.state['clustering'][0],('clustering',None))

    def show_mixture(self,*args):
        def action():
            result=self.state['mixture']
            if result is None: return
            name=self._mixture_table.currentText(); assignments,quant,_=result
            if name=='Assignments': frame=assignments
            elif quant is None:
                self._display_table(pd.DataFrame(),('mixture',name))
                self._status.setText('The saved mixture did not produce a quantitative fit. See Methods / history for its recorded warning.'); return
            elif name=='Population quantification':
                record=next(r for r in self.state['records'] if r['tag']==self.state['mixture_tag'])
                frame=_appendix_population_table(quant,record['analysis'].duration_s)
            elif name=='Cross-validation grid':
                frame=pd.DataFrame(quant.selection.grid,index=pd.Index(quant.selection.p_grid,name='n_pcs'),columns=pd.Index(quant.selection.k_grid,name='k')).reset_index()
            elif name=='Mixture-weight bootstrap': frame=pd.DataFrame(quant.ci.samples,columns=[f'subpop_{k}' for k in range(quant.fit.k)])
            elif name=='BIC ladder': frame=pd.DataFrame(dict(k=list(quant.fit.bic or {}),bic=list((quant.fit.bic or {}).values())))
            else: frame=quant.crosstab.rename_axis('subpopulation').reset_index() if quant.crosstab is not None else pd.DataFrame()
            self._display_table(frame,('mixture',name))
        return self._guard(action)

    def _display_table(self,frame,view):
        self._table_view=view; self._table_model.set_frame(frame); self._results.setCurrentWidget(self._table)

    def _restore_table(self,view):
        if view is None: return
        kind,name=view
        if kind=='candidates':
            try: _,row=self._record()
            except ValueError: return
            if row.get('recording_id')==name: self.show_candidates()
        elif kind=='comparison':
            self._comparison_table.blockSignals(True); self._comparison_table.setCurrentText(name); self._comparison_table.blockSignals(False)
            self.show_comparison()
        elif kind=='features': self.show_features()
        elif kind=='clustering': self.show_clustering()
        elif kind=='mixture':
            self._mixture_table.blockSignals(True); self._mixture_table.setCurrentText(name); self._mixture_table.blockSignals(False)
            self.show_mixture()
        elif kind=='cluster_members':
            source,method,labels=name
            snapshot=self._cluster_review.snapshot()
            if (snapshot is not None and self.cluster_review_view(source).get('current')
                    and source==self._cluster_review.kind() and method==self._cluster_review.method.currentData()
                    and self._cluster_review.selected_labels()):
                self.show_cluster_members()

    def preview_trace(self,*args):
        if self._loading: return
        self._trace.clear(); self._trace.setTitle('')
        try: _,row=self._record()
        except ValueError: return
        if 'prepared' not in row: return
        prepared=row['prepared']; choice=self._preview_choice.currentIndex(); fs=prepared.fs_hz
        self._trace.setTitle(f"{row['tag']} | {self._preview_choice.currentText()} | full recording", color='#333333')
        self._trace.setLabel('left','Probability' if choice==3 else 'Current',units=None if choice==3 else 'pA')
        series=[(prepared.detrended,'#0072B2')] if choice==0 else [(prepared.raw,'#333333')] if choice==1 else [(prepared.filtered,'#777777'),(prepared.baseline,'#D55E00')] if choice==2 else [(row['inference'].probability,'#009E73')] if 'inference' in row else []
        for values,color in series:
            curve=self._trace.plot(np.arange(len(values))/fs,values,pen=pg.mkPen(color,width=1))
            curve.setDownsampling(auto=True,method='peak'); curve.setClipToView(True)
        self._trace.enableAutoRange(); self._results.setCurrentWidget(self._trace)

    def _focus_event(self,index):
        return self._guard(lambda:self._focus_event_checked(index))

    def _focus_event_checked(self,index):
        frame=self._table_model.frame
        if not {'start_idx','end_idx'}<=set(frame): return
        if not index.isValid() or not 0<=index.row()<len(frame): return
        event=frame.iloc[index.row()]
        tag=str(event.get('tag',frame.attrs.get('manual_record_tag','')))
        identity=frame.attrs.get('manual_record_id')
        try:
            _validate_record_identities(self.state['records'])
            matches=[r for r in self.state['records'] if r['recording_id']==identity] if identity else [r for r in self.state['records'] if r['tag']==tag]
            if len(matches)!=1 or matches[0]['tag']!=tag:
                raise ValueError('The event cannot be matched to one recording. Show a current saved event table before inspecting it.')
            record=matches[0]
            if 'prepared' not in record: raise ValueError('The event recording has no prepared trace.')
            if not (0<=event.start_idx<=event.end_idx<len(record['prepared'].detrended)) or int(event.start_idx)!=event.start_idx or int(event.end_idx)!=event.end_idx:
                raise ValueError('The event boundaries are outside the prepared recording.')
        except (KeyError,TypeError,ValueError):
            self._trace.clear(); self._trace.setTitle(''); raise
        prepared=record['prepared']; start,end=int(event.start_idx),int(event.end_idx); fs=prepared.fs_hz
        pad=max(round(.002*fs),end-start+1); lo=max(0,start-pad); hi=min(len(prepared.detrended),end+1+pad)
        self._trace.clear(); self._trace.setLabel('left','Prepared current',units='pA')
        self._trace.plot(np.arange(lo,hi)/fs,prepared.detrended[lo:hi],pen=pg.mkPen('#222222',width=1))
        region=pg.LinearRegionItem((start/fs,(end+1)/fs),movable=False,brush=pg.mkBrush(0,158,115,45)); self._trace.addItem(region)
        minimum=start+int(np.argmin(prepared.detrended[start:end+1]))
        self._trace.plot([minimum/fs],[float(prepared.detrended[minimum])],pen=None,symbol='o',symbolBrush='#D55E00',symbolSize=7)
        self._trace.setTitle(f"{event.get('event_id',tag)} | source onset {float(event.get('original_onset_s',event.onset_s)):.6f} s | inclusive samples {start}:{end} | {(end-start+1)/fs*1000:.3g} ms", color='#333333')
        self._trace.enableAutoRange(); self._results.setCurrentWidget(self._trace)

    def show_figure(self,index):
        if not self._loading and 0<=index<len(self.state['figures']):
            self._guard(lambda:self._figure.set_image(self.state['figures'][index])); self._results.setCurrentWidget(self._figure)

    def refresh(self,*,preserve_table=False):
        table_view=self._table_view if preserve_table else None
        result_widget=self._results.currentWidget()
        self._loading=True
        previous=max(0,self._selector.currentIndex()); target=self._mixture_target.currentText()
        drafts=self.state.get('drafts',{})
        from .incubation_profiles import get_profile, saved_profile
        profile=get_profile(saved_profile(self.state))
        self._heading.setText(self.state.get('name','Incubation study')+' · '+profile.label)
        self.setWindowTitle(self.state.get('name','Incubation study')+' — Incubation study — NanoScope')
        self._study_details_form.set_values(drafts.get('study_details',self._study_details()))
        self._selector.clear(); self._mixture_target.clear()
        for row in self.state['records']:
            status='measured' if 'analysis' in row else 'candidates' if 'candidates' in row else 'inferred' if 'inference' in row else 'prepared' if 'prepared' in row else 'raw'
            self._selector.addItem(f"{row['tag']} — {status}"); self._mixture_target.addItem(row['tag'])
        self._selector.setCurrentIndex(min(previous,len(self.state['records'])-1))
        self._mixture_target.setCurrentIndex(max(0,self._mixture_target.findText(drafts.get('mixture_tag',self.state.get('mixture_tag',target)))))
        self._model.setText(drafts.get('model_path',self.state['model_path']))
        self._comparison_form.set_values(drafts.get('comparison',self.state['comparison_settings'])); self._cluster_form.set_values(drafts.get('cluster',self.state['cluster_settings'])); self._mixture_form.set_values(drafts.get('mixture',self.state['mixture_settings']))
        self._identity.setText(f"Workspace: {self.folder or 'not yet created'}\n{len(self.state['records'])} recordings | {len(self.state['figures'])} saved figures | {len(self.state['history'])} completed actions")
        self._figure_selector.clear(); self._figure_selector.addItems([Path(p).stem.replace('_',' ') for p in self.state['figures']])
        self._figure.scene().clear()
        if self.state['figures']: self._figure.set_image(self.state['figures'][0])
        self._details.setPlainText(self._review_views.get('methods_text','Complete or save a stage to display its saved Methods / history.'))
        self._active_index=-1
        self._loading=False; self._select_recording(self._selector.currentIndex())
        self._cluster_review.refresh()
        self._table_model.set_frame(pd.DataFrame()); self._table_view=None
        self._restore_table(table_view)
        if preserve_table: self._results.setCurrentWidget(result_widget)
        self._update_pending()

    def _select_recording(self,index):
        if self._loading: return
        if not 0<=index<len(self.state['records']):
            self._active_index=-1; self._trace.clear(); self._trace.setTitle('')
            self._source_info.clear(); self._record_summary.clear(); self._event_input_label.clear(); self._source_form.set_values(dict(tag='',pore_id=''))
            self._review_status.setText('Detect or import events to begin review.'); return
        self._store_drafts(); self._active_index=index; self._loading=True
        row=self.state['records'][index]; _source_defaults(row); values=dict(row); values['channel']=row['channel']+1; self._source_form.set_values(values)
        self._event_input_label.setText(
            'Event input: plateau-aware Schmitt signal detector. Short spikes and sustained plateaus pass; shallow long regions remain available for review.' if row.get('event_input') == 'schmitt' else
            'Event input: saved main-workspace detections. Your acceptance decisions are used downstream. Measure and gate candidates recalculates measurements while retaining review. Use main workspace detections again to capture later changes.'
            if row.get('event_input') == 'main_workspace' else
            'Event input: model candidates and the study gate. You can use reviewed detections from the main workspace instead; no model is required for that route.')
        channels='; '.join(f"Channel {c['index']+1}: {c['name']} ({c['units']})" for c in row['source_info'].get('channels',[]))
        self._source_info.setText(str(row['source_path'])+'\n'+channels+f"\n{row['source_info'].get('fs_hz',0):g} Hz · {row['source_info'].get('n_samples',0):,} samples")
        summary=[]
        if 'prepared' in row:
            prepared=row['prepared']; summary.append(f"{prepared.duration_s:.5f} s prepared")
            noise=prepared.metadata.get('qc_mad_noise_pA')
            if noise is not None: summary.append(f"preparation noise {noise:.4g} pA (MAD)")
        if 'inference' in row:
            coverage=row['inference'].coverage
            summary.append(f"{coverage.get('covered_sample_count',0):,} samples covered by model; {coverage.get('uncovered_tail_samples',0):,} tail samples uncovered")
        if 'analysis' in row:
            catalog=row['analysis'].catalog
            summary.append(f"{len(catalog):,} candidates · {int(catalog.passes_gate.sum()):,} pass the automatic gate · {int(catalog.accepted.sum()):,} accepted after review")
            reviewed=self._review_is_current(row)
            self._review_status.setText(f"{int(catalog.accepted.sum()):,} accepted · {len(catalog)-int(catalog.accepted.sum()):,} rejected. "+
                ('Review confirmed for these events.' if reviewed else 'Review pending — inspect events, then Use reviewed events for analysis. Saved results remain available to inspect.'))
        elif 'candidates' in row: summary.append(f"{len(row['candidates']):,} candidates awaiting measurement")
        if 'analysis' not in row:self._review_status.setText('Detect or import measured events to begin review.')
        self._record_summary.setText(row['tag']+': '+' | '.join(summary) if summary else '')
        drafts=row.get('drafts',{})
        self._source_form.set_values(drafts.get('source',values))
        self._schmitt_form.set_values(drafts.get('schmitt',row.get('schmitt_settings',asdict(signal.SchmittSettings())))); self._null_form.set_values(drafts.get('null',row.get('null_settings',asdict(signal.NullSettings()))))
        self._prep_form.set_values(drafts.get('preparation',row['preparation_settings'])); self._inference_form.set_values(drafts.get('inference',row['inference_settings'])); self._decode_form.set_values(drafts.get('decode',row['decode_settings'])); self._measure_form.set_values(drafts.get('measurement',row['measurement_settings']))
        self._model.setText(drafts.get('model_path',row.get('model_path',self.state.get('model_path',''))))
        self._loading=False; self._update_pending()
        self.preview_trace()

    def export_prepared(self):
        def action():
            _,record=self._prepared(); folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Export prepared CSV and metadata into folder')
            if folder: return self._stage('Export prepared '+record['tag'],lambda p,c:prep.export_prepared(record['prepared'],folder),lambda s,r:None,dict(destination=folder))
        return self._guard(action)

    def export_events(self):
        def action():
            _,record=self._record()
            if 'analysis' not in record: raise ValueError('Measure candidates first.')
            folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Export candidate and accepted tables into folder')
            if not folder: return
            def compute(progress,cancel):
                return _export_event_tables(record['analysis'].catalog,record['tag'],folder,cancel=cancel)
            return self._stage('Export event tables',compute,lambda s,r:None,dict(destination=folder))
        return self._guard(action)

    def _choose_new(self):
        launch=getattr(self.parent(),'_open_incubation_study',None)
        if callable(launch): return launch()
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Choose an empty folder for the manual workspace')
        if folder: self._guard(lambda:self.new_workspace(folder))

    def _choose_open(self):
        launch=getattr(self.parent(),'_open_incubation_study',None)
        if callable(launch): return launch()
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'Open manual workspace folder')
        if folder: self._guard(lambda:self.open_workspace(folder))

    def _choose_add(self):
        paths,_=QtWidgets.QFileDialog.getOpenFileNames(self,'Add raw recordings','','Axon recordings (*.abf)')
        if paths: self._guard(lambda:self.add_raw(paths))

    def _choose_model(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Choose trained model','','Keras model (*.keras *.h5 *.hdf5)')
        if path: self._model.setText(path)

    def is_busy(self): return self._worker is not None or (self._retiring is not None and self._retiring.isRunning())

    def _set_busy(self,busy):
        self._toolbar.setEnabled(not busy); self._stages.setEnabled(not busy); self._restore.setEnabled(not busy); self._cancel.setEnabled(busy and self._worker is not None)
        self._sync_main_curation()

    def _sync_main_curation(self):
        callback=getattr(self.parent(),'_sync_study_curation_bar',None)
        if callable(callback): callback()

    @QtCore.Slot(float,str)
    def _on_progress(self,fraction,message):
        worker=self._worker
        if worker is None or self.sender() is not worker: return
        self._progress.setValue(max(0,min(100,round(fraction*100))))
        if self._worker is worker and not worker.cancel_event.is_set(): self._status.setText(message)

    def _retire(self):
        worker,thread=self._worker,self._thread; self._worker=self._thread=None
        stopped=retire_worker_thread(worker,thread)
        if not stopped: self._retiring=thread; self._retire_timer.start()
        self._set_busy(not stopped)

    @QtCore.Slot(object)
    def _on_finished(self,result):
        if self._worker is None or self.sender() is not self._worker: return
        install=self._install; label=self._operation_label; self._retire()
        try: install(result)
        except Exception as exc:
            self._loading=False
            message=(f'Open failed; the previous workspace and drafts remain active: {exc}' if label=='Open saved workspace' else f'Result saved, but display failed: {exc}. Reopen the workspace to restore the display.')
            self._status.setText(message); self.operationFailed.emit(message); return
        self._progress.setValue(100)
        message=('Workspace opened.' if label=='Open saved workspace' else 'Figure editor opened. Save styling to study to keep appearance changes.' if label=='Open figure styling' else label+' complete. Completed settings and results are saved.')
        if self.state.get('compatibility_notice'): message+=' '+self.state['compatibility_notice']
        self._status.setText(message); self.operationFinished.emit(label)
        if label.startswith(('Find candidates','Measure and gate','Review ','Use curated detections','Save curation for ')): self.show_candidates()
        elif label=='Compute recording comparisons': self.show_comparison()
        elif label=='Extract waveform features': self.show_features()
        elif label=='Run pooled exploratory clustering': self.show_clustering()
        elif label.startswith('Fit mixture:'): self._mixture_table.setCurrentText('Assignments'); self.show_mixture()
        elif label.startswith('Generate ') and self.state['figures']: self._figure_selector.setCurrentIndex(len(self.state['figures'])-1); self.show_figure(len(self.state['figures'])-1)

    @QtCore.Slot(str)
    def _on_failed(self,message):
        if self._worker is None or self.sender() is not self._worker: return
        if 'WorkspaceConflictError:' in message:
            message+=' Your current results and parameter drafts remain in this window. Reopen the workspace to load the newer saved revision; copy any draft settings you want to retain before reopening.'
        elif self._operation_label=='Open saved workspace':
            message+=' The previous workspace and parameter drafts remain active.'
        self._retire(); self._status.setText(message); self.operationFailed.emit(message)
        self._report_curation_issue('Could not save curation: '+message)

    @QtCore.Slot()
    def _on_cancelled(self):
        if self._worker is None or self.sender() is not self._worker: return
        self._retire(); self._status.setText('Operation cancelled. The previous completed checkpoint remains available.'); self.operationCancelled.emit()
        self._report_curation_issue('Saving curation was canceled. Your main workspace edits remain available.')

    def _report_curation_issue(self,message):
        if not self._operation_label.startswith('Save curation for '): return
        owner=self.parent(); tag=self._operation_label.removeprefix('Save curation for ')
        matching=[context for context in getattr(owner,'_study_curation_sessions',{}).values()
            if context['study'] is self and context['tag']==tag]
        if matching:
            for context in matching: context['error']=message
            owner._sync_study_curation_bar(); owner.statusBar().showMessage(message,20000)

    def cancel(self):
        if self._worker: self._worker.cancel(); self._cancel.setEnabled(False); self._status.setText('Cancellation requested; waiting for the current calculation to stop…')

    def abort(self):
        if self._worker: self._worker.cancel(); self._retire()

    def _check_retired(self):
        if self._retiring is None or not self._retiring.isRunning(): self._retiring=None; self._retire_timer.stop(); self._set_busy(False)

    def closeEvent(self,event):
        self._curation_closed=True
        self.abort(); super().closeEvent(event); self._sync_main_curation()

    def showEvent(self,event):
        self._curation_closed=False
        super().showEvent(event); self._sync_main_curation()
