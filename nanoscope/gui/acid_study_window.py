"""Independently editable acid-insulin stages using the calibrated acid engine.

Every completed operation is published through the normal study checkpoint codec.
Draft controls never silently replace the settings of a completed scientific stage.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..analysis import manual_workspace as checkpoint
from ..analysis.cancel import check_cancelled
from .manual_study_window import _OperationWorker, _Form
from .incubation_window import _FrameModel, _FigureView
from .worker_thread import JOB_NONE, retire_worker_thread, start_worker_thread


def _engine():
    # Lazy import keeps unrelated Zn/Cu workspaces independent of this engine.
    from ..analysis import acid_incubation
    return acid_incubation


PREPARATION_KEYS = (
    'LINE_NOISE_METHOD', 'MAINS_NOMINAL_HZ', 'MAINS_MIN_PROMINENCE', 'NOTCH_FREQS',
    'NOTCH_BW_HZ', 'LINE_FIT_WINDOW_S', 'LINE_FIT_MASK_SIGMA', 'BASELINE_WINDOW_S',
    'BASELINE_DECIM', 'MATCH_TAU_MS', 'MATCH_MIN_ROLLOFF_HZ', 'BESSEL_ORDER',
)
DETECTION_KEYS = (
    'DETECT_RESISTIVE', 'CUSUM_DUAL_SCALE', 'CUSUM_FA_BUDGET_PER_S',
    'CUSUM_RAW_FA_MAX_PER_S', 'CUSUM_CAL_TOTAL_S', 'CUSUM_PULSE_BLOCK_MS',
    'CUSUM_BLOCK_MS', 'CUSUM_PULSE_DELTA_GRID', 'CUSUM_PULSE_H_GRID',
    'CUSUM_LEVEL_DELTA_GRID', 'CUSUM_LEVEL_H_GRID', 'CUSUM_INJECT_DEPTHS_PA',
    'CUSUM_INJECT_DWELLS_MS', 'CUSUM_INJECT_PER_CELL', 'CUSUM_FALLBACK_TO_MATCHED',
    'PRIMARY_SET_NAME', 'GATE_STATISTIC', 'SUSTAINED_GATE_MODE',
    'SUSTAINED_GATE_SIGMA', 'GATE_BASELINE_MODE', 'AMPLITUDE_STATISTIC',
    'BASELINE_MODE', 'MIN_EVENT_MS', 'MERGE_GAP_MS', 'THRESHOLD_ENTER_SIGMA',
    'THRESHOLD_EXIT_SIGMA', 'LOCAL_CONTEXT_MS', 'MAX_LOCAL_CONTEXT_MS',
)
ANALYSIS_KEYS = (
    'FAMILY_REFERENCE_TAG', 'FAMILY_REFIT_REFERENCE', 'FAMILY_EXCLUDE_BURSTS',
    'BURST_MAX_GAP_MS', 'BURST_MIN_EVENTS', 'MIN_ANALYSIS_EVENTS', 'N_TIME_BLOCKS',
    'N_BLOCK_BOOTSTRAPS', 'GMM_COMPONENTS', 'GMM_MIN_WEIGHT', 'GMM_MIN_SEPARATION',
    'GMM_MIN_BLOCK_PRESENCE', 'GMM_MIN_DELTA_BIC', 'GMM_MIN_HELDOUT_LL_GAIN',
    'MIN_POSTERIOR_FOR_LABEL', 'RUN_MORPHOLOGY', 'MORPH_MIN_DWELL_MS',
    'MORPH_MIN_EVENTS', 'MORPH_POINTS', 'HDBSCAN_MIN_CLUSTER_FRACTIONS',
    'HDBSCAN_MIN_SAMPLES_GRID', 'HDBSCAN_MIN_CLUSTER_FLOOR', 'FIG_DPI',
)
LABELS = {
    'LINE_NOISE_METHOD': 'Mains removal method', 'MAINS_NOMINAL_HZ': 'Nominal mains frequency (Hz)',
    'MAINS_MIN_PROMINENCE': 'Minimum mains peak prominence', 'NOTCH_FREQS': 'Mains harmonics (Hz)',
    'NOTCH_BW_HZ': 'Notch bandwidth (Hz; notch mode)', 'LINE_FIT_WINDOW_S': 'Sinusoid fit window (s)',
    'LINE_FIT_MASK_SIGMA': 'Sinusoid excursion mask (sigma)', 'BASELINE_WINDOW_S': 'Median baseline window (s)',
    'BASELINE_DECIM': 'Baseline decimation', 'MATCH_TAU_MS': 'Target response time (ms; blank disables)',
    'MATCH_MIN_ROLLOFF_HZ': 'Match when roll-off exceeds (Hz)', 'BESSEL_ORDER': 'Causal matching filter order',
    'DETECT_RESISTIVE': 'Also detect upward / resistive events', 'CUSUM_DUAL_SCALE': 'Combine pulse and level scales',
    'CUSUM_FA_BUDGET_PER_S': 'Accepted false-alarm budget (/s)', 'CUSUM_RAW_FA_MAX_PER_S': 'Raw false-alarm cap (/s)',
    'CUSUM_CAL_TOTAL_S': 'Quiet calibration exposure (s)', 'CUSUM_PULSE_BLOCK_MS': 'Pulse baseline block (ms)',
    'CUSUM_BLOCK_MS': 'Level baseline block (ms)', 'CUSUM_PULSE_DELTA_GRID': 'Pulse drift grid',
    'CUSUM_PULSE_H_GRID': 'Pulse evidence grid', 'CUSUM_LEVEL_DELTA_GRID': 'Level drift grid',
    'CUSUM_LEVEL_H_GRID': 'Level evidence grid', 'CUSUM_INJECT_DEPTHS_PA': 'Injection depths (pA)',
    'CUSUM_INJECT_DWELLS_MS': 'Injection dwells (ms)', 'CUSUM_INJECT_PER_CELL': 'Injections per depth/dwell cell',
    'CUSUM_FALLBACK_TO_MATCHED': 'Use matched detections if calibration fails', 'PRIMARY_SET_NAME': 'Primary detection set',
    'GATE_STATISTIC': 'Acceptance depth statistic', 'SUSTAINED_GATE_MODE': 'Acceptance noise rule',
    'SUSTAINED_GATE_SIGMA': 'Acceptance significance (sigma)', 'GATE_BASELINE_MODE': 'Acceptance baseline',
    'AMPLITUDE_STATISTIC': 'Reported amplitude', 'BASELINE_MODE': 'Reported amplitude baseline',
    'MIN_EVENT_MS': 'Minimum event duration (ms)', 'MERGE_GAP_MS': 'Merge gap (ms)',
    'THRESHOLD_ENTER_SIGMA': 'Cross-check entry threshold (sigma)',
    'THRESHOLD_EXIT_SIGMA': 'Cross-check exit threshold (sigma)', 'LOCAL_CONTEXT_MS': 'Local context (ms)',
    'MAX_LOCAL_CONTEXT_MS': 'Maximum local context (ms)', 'FAMILY_REFERENCE_TAG': 'Family reference recording',
    'FAMILY_REFIT_REFERENCE': 'Refit the reference family model', 'FAMILY_EXCLUDE_BURSTS': 'Exclude flagged bursts from fits',
    'BURST_MAX_GAP_MS': 'Burst maximum onset gap (ms)', 'BURST_MIN_EVENTS': 'Minimum events in a burst',
    'MIN_ANALYSIS_EVENTS': 'Minimum accepted events for modeling', 'N_TIME_BLOCKS': 'Validation time blocks',
    'N_BLOCK_BOOTSTRAPS': 'Block bootstrap draws', 'GMM_COMPONENTS': 'Mixture component grid',
    'GMM_MIN_WEIGHT': 'Minimum mixture weight', 'GMM_MIN_SEPARATION': 'Minimum mixture separation',
    'GMM_MIN_BLOCK_PRESENCE': 'Required block presence fraction', 'GMM_MIN_DELTA_BIC': 'Required BIC improvement',
    'GMM_MIN_HELDOUT_LL_GAIN': 'Required held-out likelihood gain', 'MIN_POSTERIOR_FOR_LABEL': 'Minimum label probability',
    'RUN_MORPHOLOGY': 'Analyze long-event morphology', 'MORPH_MIN_DWELL_MS': 'Morphology minimum dwell (ms)',
    'MORPH_MIN_EVENTS': 'Morphology minimum events', 'MORPH_POINTS': 'Resampled waveform points',
    'HDBSCAN_MIN_CLUSTER_FRACTIONS': 'Density cluster-size fractions',
    'HDBSCAN_MIN_SAMPLES_GRID': 'Density minimum-sample grid', 'HDBSCAN_MIN_CLUSTER_FLOOR': 'Density cluster-size floor',
    'FIG_DPI': 'Figure resolution (dpi)',
}
CHOICES = {
    'LINE_NOISE_METHOD': ('sinusoid', 'notch', 'none'),
    'PRIMARY_SET_NAME': ('cusum', 'cusum_matched', 'threshold', 'cusum_resistive', 'cusum_resistive_matched'),
    'GATE_STATISTIC': ('peak', 'sustained'), 'SUSTAINED_GATE_MODE': ('timescale', 'sigma', 'absolute'),
    'AMPLITUDE_STATISTIC': ('peak', 'peak_corrected', 'sustained', 'q80', 'half_peak'),
    'BASELINE_MODE': ('global', 'block', 'local'), 'GATE_BASELINE_MODE': ('local', 'global', 'block'),
}


class _SettingsForm(_Form):
    """Typed settings, with comma-separated numerical grids and nullable tau."""
    def __init__(self, defaults, keys):
        self.defaults = {k: deepcopy(defaults[k]) for k in keys if k in defaults}
        fields = []
        for key, value in self.defaults.items():
            kind = (tuple((v.replace('_', ' '), v) for v in CHOICES[key]) if key in CHOICES
                    else str if isinstance(value, (list, tuple)) or value is None or key == 'MATCH_TAU_MS'
                    else type(value) if type(value) in (bool, int, float) else str)
            fields.append((key, LABELS.get(key, key.replace('_', ' ').capitalize()), kind, value))
        super().__init__(fields)
        for widget in self.controls.values():
            if isinstance(widget,QtWidgets.QDoubleSpinBox): widget.setDecimals(12)
        self.set_values(self.defaults)

    def values(self):
        values = super().values()
        for key, default in self.defaults.items():
            if isinstance(default, (tuple, list)):
                cast = str if default and isinstance(default[0], str) else int if default and type(default[0]) is int else float
                text = values[key].strip()
                result = [cast(v.strip()) for v in text.replace(';', ',').split(',') if v.strip()]
                if not result: raise ValueError(f'{LABELS.get(key, key)} must not be empty.')
                values[key] = tuple(result) if isinstance(default, tuple) else result
            elif key == 'MATCH_TAU_MS':
                values[key] = float(values[key]) if values[key] else None
                if values[key] is not None and (not np.isfinite(values[key]) or values[key] <= 0):
                    raise ValueError('Target response time must be positive, or blank to disable matching.')
            elif default is None:
                values[key] = json.loads(values[key]) if values[key] else None
        return values


class AcidStudyWindow(QtWidgets.QMainWindow):
    operationFinished = QtCore.Signal(str)
    operationFailed = QtCore.Signal(str)
    operationCancelled = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Incubation study — Acid — NanoScope')
        self.resize(1480, 960)
        self.folder = None
        self._revision = None
        self._worker = self._thread = self._retiring = None
        self._loading = False
        self._active_tag = None
        self._table_kind = None
        self._table_tag = None
        self._style_windows = []
        self._curation_closed = False
        self._retire_timer = QtCore.QTimer(self)
        self._retire_timer.setInterval(40)
        self._retire_timer.timeout.connect(self._check_retired)
        self.state = _engine().new_state('Acid-insulin study', [], _engine().default_settings())
        self.state['workspace_kind'] = 'acid_incubation_v1'
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self._toolbar = QtWidgets.QWidget(); tools = QtWidgets.QHBoxLayout(self._toolbar)
        for title, action in [('New workspace…', self._choose_new), ('Open workspace…', self._choose_open),
                              ('Save workspace', self.save), ('Add raw ABF…', self._choose_raw)]:
            tools.addWidget(self._button(title, action))
        self._selector = QtWidgets.QComboBox(); self._selector.setAccessibleName('Active acid study recording')
        self._selector.setMinimumWidth(180); tools.addWidget(self._selector, 1)
        self._selector.currentIndexChanged.connect(self._select_record)
        root.addWidget(self._toolbar)
        self._pending = QtWidgets.QLabel(); self._pending.setWordWrap(True); root.addWidget(self._pending)
        split = QtWidgets.QSplitter(); root.addWidget(split, 1)
        self._stages = QtWidgets.QTabWidget(); self._stages.setTabPosition(QtWidgets.QTabWidget.West)
        split.addWidget(self._stages)
        self._build_sources(); self._build_preparation(); self._build_detection(); self._build_review()
        self._build_analysis(); self._build_figures()
        self._results = QtWidgets.QTabWidget(); split.addWidget(self._results)
        self._trace = pg.PlotWidget(); self._trace.setLabel('bottom', 'Retained recording time', units='s')
        self._trace.setLabel('left', 'Current', units='pA'); self._results.addTab(self._trace, 'Trace / event')
        self._table = QtWidgets.QTableView(); self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.setSortingEnabled(False)
        self._table_model = _FrameModel(self); self._table.setModel(self._table_model)
        self._table.doubleClicked.connect(lambda index: self._guard(lambda: self.focus_event(index)))
        self._results.addTab(self._table, 'Results table')
        self._figure = _FigureView(); self._results.addTab(self._figure, 'Saved figure')
        self._details = QtWidgets.QPlainTextEdit(); self._details.setReadOnly(True)
        self._results.addTab(self._details, 'Methods / history')
        split.setSizes([590, 860])
        self._progress = QtWidgets.QProgressBar(); root.addWidget(self._progress)
        status = QtWidgets.QHBoxLayout(); self._status = QtWidgets.QLabel('Create or open an acid study workspace.')
        self._status.setWordWrap(True); status.addWidget(self._status, 1)
        self._cancel = self._button('Cancel current operation', self.cancel); self._cancel.setEnabled(False)
        status.addWidget(self._cancel); root.addLayout(status)
        self.refresh()

    def _button(self, text, action):
        button = QtWidgets.QPushButton(text)
        button.setAccessibleName(text.rstrip('…'))
        button.clicked.connect(lambda _=False: self._guard(action))
        return button

    def _page(self, title, explanation):
        page = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(page)
        label = QtWidgets.QLabel(explanation); label.setWordWrap(True); layout.addWidget(label)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page)
        self._stages.addTab(scroll, title)
        return layout

    def _settings(self, layout, keys, stage):
        form = _SettingsForm(self.state['settings'], keys)
        form.changed.connect(self._draft_changed); layout.addWidget(form)
        layout.addWidget(self._button('Apply '+stage+' settings', lambda: self.apply_settings(stage)))
        return form

    def _build_sources(self):
        layout = self._page('1 Sources', 'Add each raw ABF once. Whole 30 h and Part 3 are alternative recording scopes, not independent timepoints. Unknown pore diameter may be left blank.')
        self._source_form = _Form([
            ('tag', 'Recording tag', str, ''), ('incubation_h', 'Incubation time (h)', float, 0.),
            ('pore_id', 'Physical pore identity', str, ''), ('pore_diameter_nm', 'Nominal diameter (nm; blank if unresolved)', str, ''),
            ('concentration_pm', 'Insulin concentration (pM)', float, 100.), ('voltage_mv', 'Nominal voltage (mV)', float, -300.),
            ('channel', 'Current channel (1-based)', int, 1), ('comparison_only', 'Supplementary; exclude from primary series', bool, False),
        ])
        self._source_form.changed.connect(self._draft_changed); layout.addWidget(self._source_form)
        layout.addWidget(self._button('Apply recording details', self.apply_recording_details))
        layout.addWidget(self._button('Show source identity and channel metadata', self.show_source)); layout.addStretch()

    def _build_preparation(self):
        layout = self._page('2 Prepare', 'Prepare the selected recording independently. Matching changes the effective response; a target response is not a header-verified acquisition cutoff.')
        self._prep_form = self._settings(layout, PREPARATION_KEYS, 'preparation')
        layout.addWidget(self._button('Prepare selected recording', self.prepare_record))
        layout.addWidget(self._button('Preview prepared trace', self.preview_trace)); layout.addStretch()

    def _build_detection(self):
        layout = self._page('3 Detect', 'Calibrate pulse and level CUSUM against this recording’s quiet data and injected events, then detect and measure. Inspect calibration and cross-check results before accepting the selection.')
        self._detect_form = self._settings(layout, DETECTION_KEYS, 'detection')
        layout.addWidget(self._button('Calibrate and detect selected recording', self.detect_record))
        layout.addWidget(self._button('Show candidate events', self.show_events))
        layout.addWidget(self._button('Show detection and calibration methods', self.show_source)); layout.addStretch()

    def _build_review(self):
        layout = self._page('4 Review', 'Curate study events with the main app’s event table, Event Gallery and Inspector, including manual events, boundary edits, merging and undo. Save curation to study in the main workspace, then return here to confirm the selection for analysis.')
        self._main_curation_button = self._button('Curate in main workspace…', self.open_main_curation)
        layout.addWidget(self._main_curation_button)
        note = QtWidgets.QLabel('The controls below provide quick acceptance and family review. Confirmation approves the current selection; it does not claim that every event was personally inspected.')
        note.setWordWrap(True); layout.addWidget(note)
        self._review_filter = QtWidgets.QComboBox(); self._review_filter.addItems(['All candidates', 'Accepted only', 'Rejected only'])
        self._review_filter.setAccessibleName('Acid event acceptance filter'); self._review_filter.currentIndexChanged.connect(lambda *_: self._guard(self.show_events))
        layout.addWidget(self._review_filter)
        for title, action in [('Show candidate events', self.show_events), ('Inspect selected event', self.inspect_selected_event),
                ('Accept selected events', lambda: self.review_events(True)), ('Reject selected events', lambda: self.review_events(False)),
                ('Confirm event selection for analysis', self.confirm_selection)]: layout.addWidget(self._button(title, action))
        layout.addWidget(self._button('Build event families for review', self.build_families))
        note = QtWidgets.QLabel('Confirm each primary recording, then build families before running comparisons. Accept/reject whole families, confirm the new selections, and continue to Analyze. Membership stays tied to the frozen reference model.')
        note.setWordWrap(True); layout.addWidget(note)
        for title, action in [('Show event families', self.show_families), ('Show events in selected families', self.show_family_events),
                ('Accept selected families', lambda: self.review_families(True)), ('Reject selected families', lambda: self.review_families(False)),
                ('Keep only selected families', lambda: self.review_families(True, only=True))]: layout.addWidget(self._button(title, action))
        self._review_status = QtWidgets.QLabel(); self._review_status.setWordWrap(True); layout.addWidget(self._review_status); layout.addStretch()

    def _build_analysis(self):
        layout = self._page('5 Analyze', 'Confirm each primary recording’s current event selection, then compute comparisons, validation and event families. A reference model is transferred across recordings; family numbers are not molecular identities.')
        self._analysis_form = self._settings(layout, ANALYSIS_KEYS, 'analysis')
        layout.addWidget(self._button('Analyze confirmed event selections', self.analyze_study))
        self._analysis_table = QtWidgets.QComboBox(); self._analysis_table.setAccessibleName('Acid analysis result table')
        layout.addWidget(self._analysis_table); layout.addWidget(self._button('Show selected result table', self.show_analysis_table))
        advanced = QtWidgets.QGroupBox('Additional method settings'); extra = QtWidgets.QVBoxLayout(advanced)
        additional = ('SEED','CUSUM_PASS1_BLOCK_MS','CUSUM_END_TOL_SIGMA','CUSUM_PRE_BASELINE_MS',
            'CUSUM_LEVEL_EST_MS','CUSUM_MAX_EVENT_S','CUSUM_PULSE_MAX_DWELL_MS','CUSUM_LEVEL_MIN_DWELL_MS',
            'CUSUM_PULSE_OVERRIDE','CUSUM_LEVEL_OVERRIDE','CUSUM_RECALL_TOLERANCE','CUSUM_TARGET_MIN_DEPTH_PA',
            'CUSUM_TARGET_MIN_DWELL_MS','CUSUM_CAL_MARGIN_MS','CUSUM_CAL_MIN_SEGMENT_MS','CUSUM_CAL_SPACING_MS',
            'CUSUM_INJECT_EDGE_TAU_MS','MIN_QUIET_SAMPLES','LOCAL_RATE_WINDOW_MS','MATCH_TOLERANCE_MS',
            'SUSTAINED_GATE_NOISE_QUANTILE','FAMILY_PRECURSOR_SIGMA','FAMILY_FLAT_FRACTION',
            'FAMILY_SHARP_SLOPE_SIGMA','FAMILY_MIN_WEIGHT','FAMILY_DEEP_FRAC','PRECURSOR_MAX_GAP_MS',
            'EXT_CONTEXT_MS','EXT_MOMENT_MIN_SAMPLES','EXT_SPECTRAL_MIN_SAMPLES','EXT_MIN_SAMPLES_GRID')
        self._extra_keys = tuple(k for k in additional if k in self.state['settings'])
        self._advanced_form = _SettingsForm(self.state['settings'], self._extra_keys)
        self._advanced_form.changed.connect(self._draft_changed); extra.addWidget(self._advanced_form)
        extra.addWidget(self._button('Apply additional settings', lambda: self.apply_settings('all')))
        layout.addWidget(advanced); layout.addStretch()

    def _build_figures(self):
        layout = self._page('6 Figures', 'Generate figures from committed results. Styling edits change appearance only. Save the study after styling; save the main project separately for Figure Library entries.')
        layout.addWidget(self._button('Generate study figures', self.render_figures))
        self._figure_selector = QtWidgets.QComboBox(); self._figure_selector.setAccessibleName('Saved acid study figure')
        self._figure_selector.currentIndexChanged.connect(lambda *_: self._guard(self.show_figure)); layout.addWidget(self._figure_selector)
        layout.addWidget(self._button('Style selected figure…', self.style_selected_figure))
        layout.addWidget(self._button('Fit figure to view', lambda: self._figure.fit()))
        layout.addWidget(self._button('Show methods and provenance', self.show_source)); layout.addStretch()

    def _guard(self, operation):
        try: return operation()
        except Exception as exc:
            self._status.setText(str(exc)); self.operationFailed.emit(str(exc)); return False

    def _record(self):
        tag = self._selector.currentData()
        row = next((r for r in self.state['records'] if r['tag'] == tag), None)
        if row is None: raise ValueError('Add and select a recording first.')
        return row

    def _need_folder(self):
        if self.folder is None: raise ValueError('Create or open an acid study workspace first.')
        return self.folder

    def _store_drafts(self):
        if self._loading: return
        drafts = deepcopy(self.state.get('gui_drafts', {}))
        drafts['forms'] = {name: _Form.values(form) for name, form in self._forms().items()}
        if self._active_tag: drafts.setdefault('sources', {})[self._active_tag] = self._source_form.values()
        self.state['gui_drafts'] = drafts

    def _forms(self):
        return {'preparation': self._prep_form, 'detection': self._detect_form,
                'analysis': self._analysis_form, 'all': self._advanced_form}

    def _draft_changed(self):
        if self._loading: return
        self._store_drafts(); self._pending.setText('Settings draft. Apply recording details or the affected settings before running the next stage. Save workspace preserves drafts.')

    def _assert_no_drafts(self):
        self._store_drafts()
        for name, form in self._forms().items():
            if any(self.state['settings'].get(k) != v for k,v in form.values().items()):
                raise ValueError(f'Apply changed {name} settings before continuing.')
        for row in self.state['records']:
            draft = self.state.get('gui_drafts', {}).get('sources', {}).get(row['tag'])
            if draft and self._source_values(draft) != self._source_values(self._source_display(row)):
                raise ValueError(f"Apply changed recording details for {row['tag']} before continuing.")

    @staticmethod
    def _source_display(row):
        diameter = row.get('pore_diameter_nm', row.get('nominal_pore_nm'))
        if row.get('pore_uncertain', False): diameter = None
        return {**row, 'channel': int(row.get('channel', 0))+1,
                'pore_diameter_nm': '' if diameter is None else str(diameter)}

    @staticmethod
    def _source_values(values):
        keys = ('tag','incubation_h','pore_id','pore_diameter_nm','concentration_pm','voltage_mv','channel','comparison_only')
        out = {k: values.get(k, False if k=='comparison_only' else '') for k in keys}
        out['channel'] = int(out['channel'])-1
        value = out['pore_diameter_nm']; out['pore_diameter_nm'] = None if value in ('',None) else float(value)
        return out

    def refresh(self):
        self._loading = True
        selected = self._active_tag
        self._selector.blockSignals(True); self._selector.clear()
        for r in self.state['records']: self._selector.addItem(r['tag'], r['tag'])
        index = self._selector.findData(selected); self._selector.setCurrentIndex(max(0,index)); self._selector.blockSignals(False)
        drafts = self.state.get('gui_drafts', {}).get('forms', {})
        for name, form in self._forms().items():
            form.set_values(self.state['settings']); form.set_values(drafts.get(name, {}))
        self._active_tag = self._selector.currentData()
        if self._active_tag:
            row = self._record(); self._source_form.set_values(self._source_display(row))
            self._source_form.set_values(self.state.get('gui_drafts', {}).get('sources', {}).get(row['tag'], {}))
            cat = row.get('catalog')
            if not isinstance(cat,pd.DataFrame): cat = pd.DataFrame()
            count = int(cat.accepted.sum()) if 'accepted' in cat else 0
            self._review_status.setText(f'{count} accepted / {len(cat)} candidates. Automatic acceptance is retained until explicit event or family decisions. Confirm the current selection before analysis.')
        self._figure_selector.blockSignals(True); self._figure_selector.clear()
        figures=list(self.state.get('figures', []))
        for row in self.state['records']:
            for stage in ('prepared','detection'):
                figures.extend((row.get(stage) or {}).get('figures',[]))
            figures.extend(row.get('family_figures',[]))
        seen=set()
        for path in figures:
            key=_engine().figure_style_key(self.state,path)
            if key in seen:continue
            seen.add(key)
            self._figure_selector.addItem(key.replace('::', ' / '),str(path))
        self._figure_selector.blockSignals(False)
        self._analysis_table.clear(); self._result_tables = {}
        def collect(value, prefix=''):
            if isinstance(value, pd.DataFrame): self._result_tables[prefix or 'Summary'] = value
            elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                scalar_rows = [{k:val for k,val in v.items() if val is None or isinstance(val,(str,int,float,bool,np.generic))} for v in value]
                self._result_tables[prefix or 'Summary'] = pd.DataFrame(scalar_rows)
            elif isinstance(value, dict):
                for k,v in value.items(): collect(v, f'{prefix} / {k}' if prefix else str(k))
        collect(self.state.get('analysis', {}))
        for key in self._result_tables: self._analysis_table.addItem(key)
        self._loading = False
        self._pending.setText('Completed stages are saved. Each stage can be edited and rerun independently.')
        callback = getattr(self.parent(), '_sync_study_curation_bar', None)
        if callable(callback): callback()

    def _select_record(self, *_):
        if self._loading: return
        self._store_drafts(); self._active_tag = self._selector.currentData(); self.refresh()
        self._table_kind = self._table_tag = None; self._table_model.set_frame(pd.DataFrame()); self._trace.clear()

    def _start(self, label, operation, install):
        if self.is_busy(): raise ValueError('Wait for the current operation to finish or cancel it.')
        self._operation_label, self._install = label, install
        self._worker = worker = _OperationWorker(operation)
        worker.progress.connect(self._on_progress); worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed); worker.cancelled.connect(self._on_cancelled)
        self._set_busy(True); self._status.setText(label+'…'); self._progress.setValue(0)
        self._thread = start_worker_thread(worker, domain=JOB_NONE, label='Acid study: '+label, owner=self)
        return True

    def _stage(self, label, operation, *, fresh_output=False, allow_drafts=False, on_success=None, output_filter=None):
        if self.is_busy(): raise ValueError('Wait for the current operation to finish or cancel it.')
        if not allow_drafts: self._assert_no_drafts()
        self._store_drafts(); folder = self._need_folder(); revision = self._revision
        state = deepcopy(self.state)
        output = folder/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]) if fresh_output else None
        def compute(progress, cancel):
            checkpoint.assert_workspace_revision(folder, revision); check_cancelled(cancel)
            try:
                if output is not None: output.mkdir(parents=True)
                result = operation(state, output, progress, cancel)
                check_cancelled(cancel)
                if not isinstance(result, dict): raise TypeError('Acid stage did not return a workspace state.')
                result['workspace_kind'] = 'acid_incubation_v1'
                owned = checkpoint.stage_output_files(output) if output is not None else ()
                if output_filter is not None: owned = tuple(path for path in owned if output_filter(path))
                saved = checkpoint.save_workspace(result, folder, expected_revision=revision, owned_outputs=owned, cancel=cancel)
                return result, saved
            except BaseException:
                # This UUID directory belongs only to this uncommitted operation.
                # Failed/canceled stages must not accumulate full signal caches.
                if output is not None: shutil.rmtree(output,ignore_errors=True)
                raise
        def install(result):
            self.state, self._revision = result; self.refresh()
            self._table_kind = self._table_tag = None; self._table_model.set_frame(pd.DataFrame())
            if on_success is not None: on_success()
        return self._start(label, compute, install)

    def new_workspace(self, folder, *, profile='acid', name='', sample=''):
        if self.is_busy(): raise ValueError('Wait for the current operation to stop.')
        folder = Path(folder).expanduser().resolve()
        if folder.exists() and any(not checkpoint._metadata_file(p.name) for p in folder.iterdir()):
            raise ValueError('A new workspace needs an empty folder.')
        state = _engine().new_state('Acid-insulin study', [], _engine().default_settings())
        from .incubation_profiles import initialize_study_metadata, get_profile
        if get_profile(profile).engine != 'acid': raise ValueError('Choose the acid profile for this analysis method.')
        state = initialize_study_metadata(state, profile, name=name, sample=sample)
        revision = checkpoint.save_workspace(state, folder, expected_revision=None)
        self.folder, self.state, self._revision = folder, state, revision; self._active_tag = None; self.refresh(); return True

    def open_workspace(self, folder):
        folder = Path(folder).expanduser().resolve()
        def compute(progress, cancel):
            state, revision = checkpoint.load_workspace(folder, with_revision=True, cancel=cancel)
            if state.get('workspace_kind') != 'acid_incubation_v1' and state.get('schema') != 'nanoscope-acid-incubation-v1':
                raise ValueError('This study uses another method. Open it from Incubation study to restore the saved method.')
            if not isinstance(state.get('records'), list) or not isinstance(state.get('settings'), dict):
                raise ValueError('Invalid acid study checkpoint.')
            return state, revision
        def install(result):
            self.state, self._revision = result; self.folder = folder; self._active_tag = None; self.refresh()
        return self._start('Open acid workspace', compute, install)

    def save(self):
        return self._stage('Save acid workspace', lambda state, folder, progress, cancel: state, allow_drafts=True)

    def add_raw(self, paths):
        paths = [Path(p).expanduser().resolve(strict=True) for p in paths]
        existing = {Path(r['source_path']).resolve() for r in self.state['records']}
        if len(set(paths)) != len(paths) or any(p in existing for p in paths): raise ValueError('Each source recording can be added only once.')
        def operation(state, folder, progress, cancel):
            for i,path in enumerate(paths):
                check_cancelled(cancel); info = _engine().inspect_source(path)
                tag = 'acid_recording_'+uuid.uuid4().hex[:6]
                state['records'].append(dict(tag=tag, source_path=path, source_info=info,
                    recording_id=uuid.uuid4().hex,catalog=None,families=None,
                    incubation_h=0., pore_id=tag, pore_diameter_nm=None, concentration_pm=100.,
                    voltage_mv=-300., channel=0, comparison_only=False))
                progress((i+1)/len(paths), 'Inspected '+path.name)
            state['analysis'] = None; state['figures'] = []
            return state
        return self._stage('Add acid sources', operation, allow_drafts=True)

    def apply_recording_details(self):
        row = self._record(); old_tag = row['tag']; values = self._source_values(self._source_form.values())
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', values['tag']): raise ValueError('Use letters, digits, underscores or hyphens in the recording tag.')
        if any(r['tag'] == values['tag'] and r['tag'] != old_tag for r in self.state['records']): raise ValueError('Recording tags must be unique.')
        if not values['pore_id'] or values['incubation_h'] < 0 or values['concentration_pm'] <= 0 or values['voltage_mv'] == 0:
            raise ValueError('Check incubation, pore identity, concentration and voltage.')
        if values['pore_diameter_nm'] is not None and (not np.isfinite(values['pore_diameter_nm']) or values['pore_diameter_nm'] <= 0):
            raise ValueError('Diameter must be positive or blank when unresolved.')
        channels = row.get('source_info', {}).get('channels', [])
        if channels and not 0 <= values['channel'] < len(channels): raise ValueError('Select an existing source channel.')
        if values['channel'] < 0: raise ValueError('Current channel numbering starts at one.')
        def operation(state, folder, progress, cancel):
            current = next(r for r in state['records'] if r['tag'] == old_tag)
            changed = current.get('tag') != values['tag'] or current.get('channel',0) != values['channel']
            if changed:
                current = {k:v for k,v in current.items() if k in ('tag','recording_id','source_path','source_info','incubation_h','pore_id','pore_diameter_nm','concentration_pm','voltage_mv','channel','comparison_only')}
                index = next(i for i,r in enumerate(state['records']) if r['tag']==old_tag); state['records'][index] = current
            current.update(values)
            current['nominal_pore_nm'] = values['pore_diameter_nm']; current['pore_uncertain'] = values['pore_diameter_nm'] is None
            current.pop('selection_confirmation', None); current.pop('review_receipt', None)
            state['analysis'] = None; state['figures'] = []
            state.setdefault('gui_drafts', {}).setdefault('sources', {}).pop(old_tag, None)
            return state
        return self._stage('Apply acid recording details', operation, allow_drafts=True)

    def apply_settings(self, stage):
        settings = deepcopy(self.state['settings']); settings.update(self._forms()[stage].values())
        def operation(state, folder, progress, cancel):
            result = _engine().update_settings(state, settings, stage)
            result.setdefault('gui_drafts', {}).setdefault('forms', {}).pop(stage, None)
            return result
        return self._stage('Apply '+stage+' settings', operation, allow_drafts=True)

    def prepare_record(self):
        tag = self._record()['tag']
        return self._stage('Prepare '+tag, lambda state, folder, p,c: _engine().prepare_record(state,tag,folder,progress=p,cancel=c), fresh_output=True)

    def detect_record(self):
        tag = self._record()['tag']
        return self._stage('Calibrate and detect '+tag, lambda state,folder,p,c: _engine().detect_record(state,tag,folder,progress=p,cancel=c), fresh_output=True)

    def confirm_selection(self):
        tag = self._record()['tag']
        return self._stage('Confirm selection '+tag, lambda state,folder,p,c: _engine().confirm_selection(state,tag,method='user confirmed'))

    def _curation_owner(self):
        owner = self.parent()
        if not callable(getattr(owner, 'open_study_review', None)):
            raise ValueError('Open this study from NanoScope’s Analysis menu to use the main curation tools.')
        return owner

    def _curation_record(self, context, *, verify_current=False):
        if Path(context['folder']) != self.folder:
            raise ValueError('The study workspace changed. Open this recording from its current study before saving curation.')
        row = next((r for r in self.state['records'] if r['tag'] == context['tag']
                    and r.get('recording_id') == context['recording_id']), None)
        if row is None:
            raise ValueError('The study recording changed. Open it from the study again before saving curation.')
        descriptor = row.get('main_workspace_bridge') or {}
        if descriptor.get('path') != context['path']:
            raise ValueError('This review belongs to an earlier study state. Open the current study events before saving.')
        if verify_current:
            from .acid_event_bridge import review_signature
            if descriptor.get('acid_review_signature') != review_signature(self.state, context['tag']):
                raise ValueError('The study changed since these events were opened. Return to study and open its current events.')
        return row

    def open_main_curation(self):
        """Open the study candidates in the actual shared curation workspace."""
        if self.is_busy(): raise ValueError('Wait for the current study operation to finish.')
        self._assert_no_drafts()
        row = self._record(); tag = row['tag']; owner = self._curation_owner()
        from . import acid_event_bridge as bridge
        from .study_event_bridge import _assert_idle
        _assert_idle(owner)
        # Re-enter the live table rather than replacing the curator's unsaved work.
        descriptor = row.get('main_workspace_bridge') or {}
        current_review = descriptor.get('acid_review_signature') == bridge.review_signature(self.state, tag)
        for doc in owner._session.documents() if current_review else ():
            if descriptor.get('path') and doc.meta.path.resolve() == Path(descriptor['path']).resolve():
                owner.begin_study_curation(self, tag, doc.doc_id)
                owner.show_study_curation()
                self.hide()
                return True
        launch_doc_ids = frozenset(doc.doc_id for doc in owner._session.documents())
        session = owner._session; generation = owner._study_curation_project_generation; payload = []
        def operation(state, folder, progress, cancel):
            progress(.05, 'Opening study events in the main curation tools…')
            result = bridge.prepare_review_export(state, tag, folder, cancel=cancel)
            next(r for r in state['records'] if r['tag'] == tag)['main_workspace_bridge'] = result['descriptor']
            payload.append(result)
            return state
        def opened():
            if owner._session is not session or owner._study_curation_project_generation != generation:
                raise ValueError('The main project changed while the study events were opening. Open them again from the study.')
            doc_id = owner.open_study_review(payload.pop(), launch_doc_ids)
            owner.begin_study_curation(self, tag, doc_id, fresh=True)
            self.hide()
        return self._stage('Open '+tag+' for curation', operation, fresh_output=True, on_success=opened,
            # Native calibration is an editable preference; source samples remain
            # immutable and the bridge verifies the calibration before every save.
            output_filter=lambda path: Path(path).name != 'prepared-review.h5.npcal.json')

    def save_main_curation(self, context):
        """Freeze the matching native table now and remeasure it off the UI thread."""
        if self._curation_closed:
            raise ValueError('Return to the study to reopen it before saving curation.')
        if self.is_busy(): raise ValueError('Wait for the current study operation to finish.')
        self._curation_record(context, verify_current=True); self._assert_no_drafts()
        owner = self._curation_owner(); tag = context['tag']
        if (owner._session is not context['session']
                or context['project_generation'] != owner._study_curation_project_generation
                or not owner._session.has_document(context['doc_id'])):
            raise ValueError('The main workspace recording changed. Open the study events again before saving.')
        result = owner._analysis_panel.current_detection()
        if result is None or result.doc_id != context['doc_id']:
            raise ValueError('Show this study recording’s events before saving curation.')
        from . import acid_event_bridge as bridge
        snapshot = bridge.snapshot_events(owner, self.state, tag)
        context.pop('error', None)
        captured_signature = owner._study_curation_signature(result)
        def saved():
            context['saved_signature'] = captured_signature
            context['message'] = 'Curation saved to study. Return to study to confirm the selection for analysis.'
            owner._sync_study_curation_bar()
            owner.statusBar().showMessage('Curation saved for '+tag+'. Return to study to confirm the selection for analysis.', 20000)
        return self._stage('Save curation for '+tag,
            lambda state, folder, progress, cancel: bridge.apply_review(state, tag, snapshot, folder, progress=progress, cancel=cancel),
            fresh_output=True, on_success=saved)

    def return_to_curation_record(self, context):
        self._curation_record(context)
        if not self.is_busy():
            self._store_drafts(); self._active_tag = context['tag']; self.refresh()
            self._stages.setCurrentIndex(3)
            if context.get('pending'):
                self._status.setText('Main workspace curation has unsaved changes. Open Curate in main workspace and use Save curation to study to include them in analysis.')
        self.show(); self.raise_(); self.activateWindow()

    def build_families(self):
        return self._stage('Build event families for review', lambda state,folder,p,c: _engine().build_families(state,folder,progress=p,cancel=c), fresh_output=True)

    def analyze_study(self):
        return self._stage('Analyze acid study', lambda state,folder,p,c: _engine().analyze_study(state,folder,progress=p,cancel=c), fresh_output=True)

    def render_figures(self):
        return self._stage('Generate acid figures', lambda state,folder,p,c: _engine().render_figures(state,folder,progress=p,cancel=c), fresh_output=True)

    def _display_table(self, frame, kind, tag=None):
        self._table_kind, self._table_tag = kind, tag
        self._table_model.set_frame(frame); self._results.setCurrentWidget(self._table)
        self._table.resizeColumnsToContents()

    def show_events(self, *_):
        if self._loading or not self.state['records']: return
        row = self._record(); frame = row.get('catalog')
        if not isinstance(frame, pd.DataFrame): raise ValueError('Detect this recording first.')
        frame = frame.copy(deep=False); mode = self._review_filter.currentIndex()
        if mode: frame = frame.loc[frame.accepted.astype(bool) == (mode==1)]
        first = [c for c in ('event_id','accepted','polarity','start_idx','end_idx','onset_s','dwell_ms','peak_depth_pA','family') if c in frame]
        self._display_table(frame[first+[c for c in frame if c not in first]], 'events', row['tag'])

    def _selected(self, kind, column):
        row = self._record()
        if self._table_kind != kind or self._table_tag != row['tag']: raise ValueError('Show the current recording’s '+kind+' table first.')
        selected = self._table.selectionModel().selectedRows(); frame = self._table_model.frame
        if not selected or column not in frame: raise ValueError('Select one or more rows in the results table.')
        return [frame.iloc[i.row()][column] for i in selected]

    def review_events(self, accepted):
        tag = self._record()['tag']; ids = self._selected('events','event_id')
        return self._stage('Review acid events', lambda state,folder,p,c: _engine().set_event_acceptance(state,tag,ids,accepted))

    def _memberships(self):
        row = self._record(); families = row.get('families')
        if isinstance(families, pd.DataFrame) and {'event_id','family'} <= set(families): return families
        catalog = row.get('catalog')
        if isinstance(catalog,pd.DataFrame) and 'family' in catalog: return catalog[['event_id','family']]
        raise ValueError('Confirm event selections, then build event families for review.')

    def show_families(self):
        row = self._record(); members = self._memberships()
        cat = row['catalog'][['event_id','accepted']]
        frame = members[['event_id','family']].merge(cat,on='event_id',validate='one_to_one')
        summary = frame.groupby('family',dropna=False).agg(events=('event_id','size'),accepted=('accepted','sum')).reset_index()
        summary['rejected'] = summary.events-summary.accepted
        self._display_table(summary, 'families', row['tag'])

    def show_family_events(self):
        row = self._record(); labels = self._selected('families','family'); members = self._memberships()
        ids = members.loc[members.family.isin(labels),'event_id']
        self._display_table(row['catalog'].loc[row['catalog'].event_id.isin(ids)], 'events', row['tag'])

    def review_families(self, accepted, *, only=False):
        tag = self._record()['tag']; labels = self._selected('families','family')
        members = self._memberships(); others = [v for v in members.family.drop_duplicates() if v not in labels]
        unassigned = self._record()['catalog'].loc[~self._record()['catalog'].event_id.isin(members.event_id),'event_id'].tolist()
        def operation(state,folder,p,c):
            result = _engine().set_family_acceptance(state,tag,labels,accepted)
            if only and others: result = _engine().set_family_acceptance(result,tag,others,False)
            if only and unassigned: result = _engine().set_event_acceptance(result,tag,unassigned,False)
            return result
        return self._stage('Review acid families', operation)

    def preview_trace(self):
        row = self._record()
        x,y = _engine().read_trace(self.state,row['tag'],start_idx=0,stop_idx=None,max_points=10000)
        self._trace.clear(); self._trace.plot(x,y,pen=pg.mkPen('#267b91',width=1))
        self._trace.setTitle(f"{row['tag']} · Prepared recording")
        self._results.setCurrentWidget(self._trace)
        self._trace.autoRange()

    def inspect_selected_event(self):
        self._selected('events','event_id'); selected = self._table.selectionModel().selectedRows()
        return self.focus_event(selected[0])

    def focus_event(self, index):
        if self._table_kind != 'events' or self._table_tag != self._record()['tag']: return
        event = self._table_model.frame.iloc[index.row()]; row = self._record()
        start,end = int(event.start_idx),int(event.end_idx)
        if not row.get('prepared'): raise ValueError('Prepare this recording first.')
        fs = float(row['prepared']['metadata']['fs_hz'])
        if not np.isfinite(fs) or fs <= 0: raise ValueError('Prepared recording sample rate must be positive and finite.')
        pad = max(int(.002*fs), end-start+1)
        x,y = _engine().read_trace(self.state,row['tag'],start_idx=max(0,start-pad),stop_idx=end+1+pad,max_points=None)
        self._trace.clear(); self._trace.plot(x,y,pen=pg.mkPen('#267b91',width=1))
        region = pg.LinearRegionItem([start/fs,(end+1)/fs],movable=False,brush=(220,140,45,50))
        self._trace.addItem(region); self._trace.setTitle(f"{row['tag']} · {event.event_id} · "+('accepted' if event.accepted else 'rejected'))
        self._results.setCurrentWidget(self._trace)
        # Manual zoom can disable ongoing auto-range. Fit every newly selected
        # event to its own curve and inclusive event region.
        self._trace.autoRange()

    def show_analysis_table(self):
        key = self._analysis_table.currentText()
        if key not in self._result_tables: raise ValueError('Analyze the study first.')
        self._display_table(self._result_tables[key], 'analysis')

    def show_source(self):
        def compact(value):
            if isinstance(value,pd.DataFrame): return dict(rows=len(value),columns=list(value))
            if isinstance(value,np.ndarray): return dict(shape=list(value.shape),dtype=str(value.dtype))
            if isinstance(value,dict): return {k:compact(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)): return [compact(v) for v in value]
            return value
        row = self._record() if self.state['records'] else {}
        self._details.setPlainText(json.dumps(compact(dict(record=row,settings=self.state['settings'],history=self.state.get('history',[]))),indent=2,default=str))
        self._results.setCurrentWidget(self._details)

    def show_figure(self, *_):
        if self._loading: return
        path = self._figure_selector.currentData()
        if path: self._figure.set_image(path); self._results.setCurrentWidget(self._figure)

    def style_selected_figure(self):
        path = self._figure_selector.currentData()
        if not path: raise ValueError('Generate and select a figure first.')
        path = Path(path); expected = [self._revision,path]; state = deepcopy(self.state)
        style_key = _engine().figure_style_key(state, path)
        def compute(progress,cancel): return _engine().reconstruct_figure(state,path,progress=progress,cancel=cancel,apply_style=False)
        def install(result):
            from .study_figure_window import StudyFigureWindow
            figure,data,resources = result
            def save_style(style,snapshot,editor):
                if self._revision != expected[0] or self.is_busy(): raise ValueError('The study changed or is busy. Reopen the figure editor after it finishes.')
                saved = []
                def operation(current,folder,p,c):
                    target = folder/path.name; target.write_bytes(snapshot.png_bytes)
                    target.with_suffix('.svg').write_bytes(snapshot.svg_bytes)
                    from ..plot.acid_figure_recipe import capture_figure
                    from ..plot.study_figure_style import StudyFigureStyle
                    fresh,data,_ = _engine().reconstruct_figure(current,expected[1],cancel=c,apply_style=False)
                    StudyFigureStyle(fresh).apply(**style)
                    capture_figure(fresh,target,data); fresh.clear()
                    current.setdefault('figure_styles',{})[style_key] = deepcopy(style)
                    current.setdefault('figure_keys',{})[target] = style_key
                    current.setdefault('figure_sources',{})[style_key] = _engine().figure_source_path(current,expected[1])
                    figures=current.get('figures',[])
                    if any(Path(v)==expected[1] for v in figures):
                        current['figures'] = [target if Path(v)==expected[1] else v for v in figures]
                    else:current['figures']=[*figures,target]
                    saved.append(target)
                    return current
                def committed():
                    expected[:] = [self._revision,saved[0]]; editor.saved()
                self._stage('Save acid figure styling',operation,fresh_output=True,allow_drafts=True,on_success=committed)
            editor = StudyFigureWindow(figure,data,resources,name=path.stem,
                provenance=dict(study=self.state.get('name'),workspace=str(self.folder),revision=expected[0]),
                style=state.get('figure_styles',{}).get(style_key),save_style=save_style,library_owner=self.parent(),parent=self)
            self._style_windows.append(editor); editor.destroyed.connect(lambda *_: self._style_windows.remove(editor) if editor in self._style_windows else None)
            editor.show()
        return self._start('Open acid figure styling',compute,install)

    def _choose_new(self):
        launcher = getattr(self.parent(), '_open_incubation_study', None)
        if callable(launcher): return launcher()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,'Choose an empty acid workspace folder')
        if folder: self.new_workspace(folder)

    def _choose_open(self):
        launcher = getattr(self.parent(), '_open_incubation_study', None)
        if callable(launcher): return launcher()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,'Open acid workspace folder')
        if folder: self.open_workspace(folder)

    def _choose_raw(self):
        paths,_ = QtWidgets.QFileDialog.getOpenFileNames(self,'Add acid recordings','','Axon recordings (*.abf)')
        if paths: self.add_raw(paths)

    def is_busy(self): return self._worker is not None or (self._retiring is not None and self._retiring.isRunning())

    def _set_busy(self,busy):
        self._toolbar.setEnabled(not busy); self._stages.setEnabled(not busy)
        self._cancel.setEnabled(busy and self._worker is not None)
        callback = getattr(self.parent(), '_sync_study_curation_bar', None)
        if callable(callback): callback()

    @QtCore.Slot(float,str)
    def _on_progress(self,fraction,message):
        if self.sender() is not self._worker: return
        self._progress.setValue(max(0,min(100,round(fraction*100))))
        if not self._worker.cancel_event.is_set(): self._status.setText(message)

    def _retire(self):
        worker,thread = self._worker,self._thread; self._worker=self._thread=None
        stopped = retire_worker_thread(worker,thread)
        if not stopped: self._retiring=thread; self._retire_timer.start()
        self._set_busy(not stopped)

    @QtCore.Slot(object)
    def _on_finished(self,result):
        if self._worker is None or self.sender() is not self._worker: return
        install,label = self._install,self._operation_label; self._retire()
        try: install(result)
        except Exception as exc:
            self._status.setText('Completed result could not be displayed: '+str(exc)); self.operationFailed.emit(str(exc)); return
        self._progress.setValue(100); self._status.setText(label+' complete.'); self.operationFinished.emit(label)

    @QtCore.Slot(str)
    def _on_failed(self,message):
        if self._worker is None or self.sender() is not self._worker: return
        self._retire(); self._status.setText(message+' Previous completed stages remain available.'); self.operationFailed.emit(message)
        self._report_curation_issue('Could not save curation: '+message)

    @QtCore.Slot()
    def _on_cancelled(self):
        if self._worker is None or self.sender() is not self._worker: return
        self._retire(); self._status.setText('Operation canceled. The previous completed checkpoint remains available.'); self.operationCancelled.emit()
        self._report_curation_issue('Saving curation was canceled. Your main workspace edits remain available.')

    def _report_curation_issue(self, message):
        if not self._operation_label.startswith('Save curation for '): return
        owner = self.parent()
        tag = self._operation_label.removeprefix('Save curation for ')
        contexts = getattr(owner, '_study_curation_sessions', {}).values()
        matching = [context for context in contexts if context['study'] is self and context['tag'] == tag]
        if matching:
            for context in matching: context['error'] = message
            owner._sync_study_curation_bar()
            owner.statusBar().showMessage(message, 20000)

    def cancel(self):
        if self._worker:
            self._worker.cancel(); self._cancel.setEnabled(False); self._status.setText('Cancellation requested; waiting for the calculation to stop…')

    def abort(self):
        if self._worker: self._worker.cancel(); self._retire()

    def _check_retired(self):
        if self._retiring is None or not self._retiring.isRunning():
            self._retiring=None; self._retire_timer.stop(); self._set_busy(False)

    def closeEvent(self,event):
        self._curation_closed = True
        self.abort(); super().closeEvent(event)
        callback = getattr(self.parent(), '_sync_study_curation_bar', None)
        if callable(callback): callback()

    def showEvent(self,event):
        self._curation_closed = False
        super().showEvent(event)
        callback = getattr(self.parent(), '_sync_study_curation_bar', None)
        if callable(callback): callback()
