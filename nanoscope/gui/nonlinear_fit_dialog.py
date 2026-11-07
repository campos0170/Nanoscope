"""Nonlinear X-Y fitting, with asynchronous estimation and immutable saved diagnostics.

The dialog receives exact plotted samples in storage units. Display-axis transforms
never enter the estimator. Installing a result checks its source digest again.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import html
import io
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.atomic_write import staged_write


@dataclass
class CurveDataset:
    key: str
    label: str
    x: np.ndarray
    y: np.ndarray
    x_name: str = 'x'
    y_name: str = 'y'
    x_unit: str = ''
    y_unit: str = ''
    sigma_columns: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def source(self):
        digest = hashlib.sha256()
        for a in (self.x, self.y):
            a = np.ascontiguousarray(a, dtype='<f8')
            digest.update(str(a.shape).encode()); digest.update(a.tobytes())
        digest.update(json.dumps([self.key, self.x_name, self.y_name,
                                  self.x_unit, self.y_unit], ensure_ascii=False).encode())
        provenance = json.loads(json.dumps(self.provenance, sort_keys=True, default=lambda v: v.tolist() if hasattr(v,'tolist') else str(v)))
        digest.update(json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode())
        return dict(key=self.key, label=self.label, x_name=self.x_name, y_name=self.y_name,
                    x_unit=self.x_unit, y_unit=self.y_unit, input_digest=digest.hexdigest(),
                    provenance=provenance, n_input=len(self.x))


class _FitWorker(QtCore.QObject):
    computed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()
    def __init__(self, dataset, config, sigma):
        super().__init__()
        self.dataset, self.config, self.sigma = dataset, config, sigma
        self._cancel_event = threading.Event()
    def run(self):
        from ..analysis.curve_fitting import fit_curve, FitCancelled
        try:
            from ..analysis.cancel import cancel_scope
            with cancel_scope(self._cancel_event.is_set):
                result = fit_curve(self.dataset.x.copy(), self.dataset.y.copy(), self.config,
                                   y_sigma=self.sigma, cancel=self._cancel_event.is_set)
            if not self._cancel_event.is_set(): self.computed.emit(result.to_dict())
        except FitCancelled:
            self.failed.emit('Fit cancelled. The previous applied result is unchanged.')
        except Exception as exc:
            self.failed.emit(f'Fit unavailable: {exc}')
        finally:
            self.finished.emit()


def _retire(lifetime):
    """Bounded, cooperative cleanup even when the owning page is destroyed."""
    from .worker_thread import retire_worker_thread
    worker=lifetime.pop('worker',None);thread=lifetime.pop('thread',None)
    if worker is not None:worker._cancel_event.set()
    if thread is not None:retire_worker_thread(worker,thread,wait_ms=0)


def parameter_units(model, names, x_unit, y_unit):
    x=x_unit or 'unitless';y=y_unit or 'unitless'
    units={}
    for name in names:
        if model=='custom':unit='equation-defined'
        elif name in ('baseline','amplitude','amplitude1','amplitude2','bottom','top','intercept'):unit=y
        elif name in ('center','sigma','gamma','tau','ec50','midpoint_scale','v_half'):unit=x
        elif name in ('rate','rate1','rate2'):unit=f'1/{x}'
        elif name=='slope':unit=f'{y}/{x}' if model=='linear' else x
        else:unit='unitless'
        if model=='power_law' and name=='amplitude':unit=f'{y}/({x}^exponent)'
        units[name]=unit
    return units


def result_summary(state):
    result = state.get('result', {})
    config = result.get('config', {})
    source = state.get('source', {})
    metrics = result.get('metrics', {})
    params = ', '.join(f'{k}={float(v):.5g} {state.get("parameter_units",{}).get(k,"")}'.rstrip() for k, v in result.get('parameters', {}).items())
    ci = result.get('parameter_ci', {})
    lines = [f"Nonlinear fit: {config.get('model', 'model')} / {source.get('label', '')}",
             f"{source.get('y_name','y')} [{source.get('y_unit') or 'unitless'}] vs "
             f"{source.get('x_name','x')} [{source.get('x_unit') or 'unitless'}] — storage units",
             params, f"Raw-response residuals; weighting={config.get('weighting','uniform')}; "
             f"loss={config.get('loss','linear')}",
             'Metrics: ' + ', '.join(f'{k}={v:.5g}' if isinstance(v, (float,int)) else f'{k}={v}'
                                     for k,v in metrics.items())]
    if ci:
        confidence = float(config.get('confidence', .95)) * 100
        lines.append(f'{confidence:g}% parameter intervals: ' + '; '.join(f'{k}: {v}' for k,v in ci.items()))
    if result.get('message'):lines.append('Solver: '+str(result['message']))
    lines.extend(str(w) for w in result.get('warnings', ()))
    return '\n'.join(lines)


def result_rows(state):
    """Exact fitted observations and sampled uncertainty curves, in storage units."""
    r = state.get('result', {}); src = state.get('source', {})
    common = dict(source_key=src.get('key'), x_name=src.get('x_name'), y_name=src.get('y_name'),
                  x_unit=src.get('x_unit'), y_unit=src.get('y_unit'),
                  input_digest=src.get('input_digest'),
                  fit_config_json=json.dumps(r.get('config', {}), ensure_ascii=False),
                  fit_parameters_json=json.dumps(r.get('parameters', {}), ensure_ascii=False),
                  parameter_units_json=json.dumps(state.get('parameter_units',{}),ensure_ascii=False),
                  fit_metrics_json=json.dumps(r.get('metrics', {}), ensure_ascii=False))
    rows=[]
    for i, x in enumerate(r.get('x', [])):
        row=dict(common,part='nonlinear_observation',x=x)
        for key in ('y','y_fit','residuals','y_sigma'):
            values=r.get(key) or []; row[key]=values[i] if i < len(values) else None
        rows.append(row)
    for i,x in enumerate(r.get('x_curve', [])):
        row=dict(common,part='nonlinear_curve',x=x)
        for key in ('y_curve','confidence_lower','confidence_upper','prediction_lower','prediction_upper'):
            values=r.get(key) or []; row[key]=values[i] if i < len(values) else None
        rows.append(row)
    return rows


class NonlinearFitDialog(QtWidgets.QDialog):
    applied = QtCore.Signal(dict)
    cleared = QtCore.Signal()
    def __init__(self, datasets, parent=None, saved=None, *, display_log=(False, False)):
        super().__init__(parent)
        from ..analysis.curve_fitting import list_models
        self.setWindowTitle('Curve fit'); self.resize(1180, 880)
        self.datasets = datasets; self._worker = None; self._result = None; self._close_pending = False
        self._draft_dirty = False
        self._confidence = .95
        self._lifetime = {}
        lifetime = self._lifetime
        self.destroyed.connect(lambda *_: _retire(lifetime))
        root=QtWidgets.QVBoxLayout(self)
        split=QtWidgets.QSplitter(); root.addWidget(split,1)
        controls=QtWidgets.QWidget(); form=QtWidgets.QFormLayout(controls)
        scroll=QtWidgets.QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(controls);scroll.setMinimumWidth(430);split.addWidget(scroll)
        self.source_combo=QtWidgets.QComboBox()
        for data in datasets: self.source_combo.addItem(data.label, data.key)
        self.model_combo=QtWidgets.QComboBox()
        for model in list_models(): self.model_combo.addItem(model.label, model.id)
        form.addRow('Source',self.source_combo);form.addRow('Model',self.model_combo)
        self.formula=QtWidgets.QLabel();self.formula.setWordWrap(True);form.addRow(self.formula)
        self.units=QtWidgets.QLabel();self.units.setWordWrap(True);form.addRow(self.units)
        self.expression=QtWidgets.QLineEdit('baseline + amplitude * exp(-x / tau)')
        self.custom_names=QtWidgets.QLineEdit('baseline, amplitude, tau')
        self.custom_controls=QtWidgets.QWidget();custom_form=QtWidgets.QFormLayout(self.custom_controls)
        custom_form.setContentsMargins(0,0,0,0)
        custom_form.addRow('Custom equation',self.expression);custom_form.addRow('Parameters',self.custom_names)
        form.addRow(self.custom_controls)
        self.minimum=QtWidgets.QLineEdit();self.maximum=QtWidgets.QLineEdit()
        self.minimum.setPlaceholderText('all');self.maximum.setPlaceholderText('all')
        ranges=QtWidgets.QWidget();row=QtWidgets.QHBoxLayout(ranges);row.setContentsMargins(0,0,0,0)
        row.addWidget(self.minimum);row.addWidget(QtWidgets.QLabel('to'));row.addWidget(self.maximum)
        form.addRow('Fit x range',ranges)
        self.advanced_toggle=QtWidgets.QToolButton();self.advanced_toggle.setText('Advanced: starts, constraints and loss')
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(QtCore.Qt.RightArrow);self.advanced_toggle.setCheckable(True)
        form.addRow(self.advanced_toggle)
        self.constraint_summary=QtWidgets.QLabel();self.constraint_summary.setWordWrap(True)
        form.addRow(self.constraint_summary)
        self.advanced_controls=QtWidgets.QWidget();advanced=QtWidgets.QFormLayout(self.advanced_controls)
        self.advanced_controls.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Maximum)
        advanced.setContentsMargins(0,0,0,0);form.addRow(self.advanced_controls)
        self.advanced_controls.hide()
        self.advanced_toggle.toggled.connect(self._set_advanced_visible)
        self.guess_button=QtWidgets.QPushButton('Estimate starts')
        self.guess_button.setToolTip('Estimate free starting values within the current bounds. Fixed values and constraints are preserved.')
        self.reset_constraints_button=QtWidgets.QPushButton('Reset constraints')
        self.reset_constraints_button.setToolTip('Restore model bounds and release fixed parameters. Starting values are preserved.')
        start_buttons=QtWidgets.QWidget();start_row=QtWidgets.QHBoxLayout(start_buttons);start_row.setContentsMargins(0,0,0,0)
        start_row.addWidget(self.guess_button);start_row.addWidget(self.reset_constraints_button);advanced.addRow(start_buttons)
        self.parameters=QtWidgets.QTableWidget(0,6)
        self.parameters.setHorizontalHeaderLabels(['Parameter','Start','Lower','Upper','Fixed','Unit'])
        self.parameters.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.parameters.horizontalHeader().setSectionResizeMode(0,QtWidgets.QHeaderView.ResizeToContents)
        self.parameters.horizontalHeaderItem(1).setToolTip('Initial value, or the constant value when Fixed is checked.')
        self.parameters.setMinimumHeight(175);self.parameters.setMaximumHeight(240);advanced.addRow(self.parameters)
        self.weighting=QtWidgets.QComboBox();advanced.addRow('Error weights',self.weighting)
        self.loss=QtWidgets.QComboBox()
        for label,value in [('Least squares','linear'),('Robust soft L1','soft_l1'),('Robust Huber','huber')]:self.loss.addItem(label,value)
        advanced.addRow('Loss',self.loss)
        self.f_scale=QtWidgets.QLineEdit();self.f_scale.setPlaceholderText('automatic')
        self.f_scale.setToolTip('Robust transition scale in y units, or standard-deviation units when weighted. Positive values only.')
        self.robust_scale_label=QtWidgets.QLabel('Robust scale');advanced.addRow(self.robust_scale_label,self.f_scale)
        self.f_scale.textEdited.connect(self._edited)
        self.help=QtWidgets.QLabel('Weights use 1/σ² from a selected measurement-uncertainty column. Robust loss reduces outlier influence. Confidence bands describe the mean curve; prediction bands describe a new observation. All fits minimize raw-response residuals.')
        self.help.setWordWrap(True);advanced.addRow(self.help)
        right=QtWidgets.QWidget();right_layout=QtWidgets.QVBoxLayout(right);split.addWidget(right)
        self.result_identity=QtWidgets.QLabel('No completed fit. Review the observations, then Fit.')
        self.result_identity.setWordWrap(True);right_layout.addWidget(self.result_identity)
        self.view_tabs=QtWidgets.QTabWidget();right_layout.addWidget(self.view_tabs,1)
        draft_page=QtWidgets.QWidget();draft_layout=QtWidgets.QVBoxLayout(draft_page)
        self.draft_preview=pg.PlotWidget();draft_layout.addWidget(self.draft_preview,1)
        self.draft_summary=QtWidgets.QLabel();self.draft_summary.setWordWrap(True);draft_layout.addWidget(self.draft_summary)
        self.view_tabs.addTab(draft_page,'Draft data')
        completed_page=QtWidgets.QWidget();completed_layout=QtWidgets.QVBoxLayout(completed_page)
        completed_layout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        completed_layout.setAlignment(QtCore.Qt.AlignTop)
        self.completed_scroll=QtWidgets.QScrollArea();self.completed_scroll.setWidgetResizable(True)
        self.completed_scroll.setFrameShape(QtWidgets.QFrame.NoFrame);self.completed_scroll.setWidget(completed_page)
        self.view_tabs.addTab(self.completed_scroll,'Last completed fit');self.view_tabs.setTabEnabled(1,False)
        self.preview=pg.PlotWidget();self.preview.setMinimumHeight(180)
        self.residual=pg.PlotWidget();self.residual.setFixedHeight(190)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Ignored)
        self.residual.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Ignored)
        # Match the figure's presentation without transforming estimator inputs.
        # Residuals remain signed raw responses even on a logarithmic Y figure.
        self.preview.setLogMode(x=display_log[0],y=display_log[1])
        self.draft_preview.setLogMode(x=display_log[0],y=display_log[1])
        self.residual.setLogMode(x=display_log[0],y=False)
        self.residual.setXLink(self.preview)
        completed_layout.addWidget(self.preview,2);completed_layout.addWidget(self.residual,1)
        completed_layout.addSpacing(8)
        self.result_metrics=QtWidgets.QLabel();self.result_metrics.setWordWrap(True);completed_layout.addWidget(self.result_metrics)
        self.result_metrics.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Maximum)
        self.estimates=QtWidgets.QTableWidget(0,5)
        self.estimates.setHorizontalHeaderLabels(['Parameter','Estimate','Unit','95% interval','Status'])
        self.estimates.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.estimates.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.estimates.verticalHeader().hide();self.estimates.setMaximumHeight(190);completed_layout.addWidget(self.estimates)
        self.result_warnings=QtWidgets.QLabel();self.result_warnings.setWordWrap(True);completed_layout.addWidget(self.result_warnings)
        self.result_warnings.setSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Maximum)
        self.diagnostics_toggle=QtWidgets.QToolButton();self.diagnostics_toggle.setText('Detailed diagnostics')
        self.diagnostics_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.diagnostics_toggle.setArrowType(QtCore.Qt.RightArrow);self.diagnostics_toggle.setCheckable(True)
        completed_layout.addWidget(self.diagnostics_toggle)
        self.results=QtWidgets.QPlainTextEdit();self.results.setReadOnly(True);self.results.setFixedHeight(110)
        completed_layout.addWidget(self.results);self.results.hide()
        self.diagnostics_toggle.toggled.connect(self._set_diagnostics_visible)
        self.status=QtWidgets.QLabel('Choose a model and review its starting values, then Fit.');self.status.setWordWrap(True);root.addWidget(self.status)
        buttons=QtWidgets.QHBoxLayout();root.addLayout(buttons)
        self.fit_button=QtWidgets.QPushButton('Fit');self.cancel_button=QtWidgets.QPushButton('Cancel fit')
        self.apply_button=QtWidgets.QPushButton('Apply to figure');self.export_button=QtWidgets.QPushButton('Export last completed fit…')
        self.export_button.setToolTip('Export the completed fit identified above, including its original model, source and settings. Draft edits are not exported.')
        self.clear_button=QtWidgets.QPushButton('Remove applied fit');self.clear_button.setEnabled(bool(saved))
        self.clear_button.clicked.connect(self.clear_applied)
        self.close_button=QtWidgets.QPushButton('Close')
        for b in (self.fit_button,self.cancel_button,self.apply_button,self.export_button,self.clear_button,self.close_button):buttons.addWidget(b)
        self.cancel_button.setEnabled(False);self.apply_button.setEnabled(False);self.export_button.setEnabled(False)
        self.fit_button.clicked.connect(self.start_fit);self.cancel_button.clicked.connect(self.cancel_fit)
        self.apply_button.clicked.connect(self.apply_result);self.export_button.clicked.connect(self.export_result)
        self.close_button.clicked.connect(self.reject);self.guess_button.clicked.connect(self.estimate_starts)
        self.reset_constraints_button.clicked.connect(self.reset_constraints)
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.model_combo.currentIndexChanged.connect(self.reset_parameters)
        self.custom_names.editingFinished.connect(self._custom_parameters_changed)
        self.expression.textEdited.connect(self._edited);self.parameters.itemChanged.connect(self._edited)
        for w in (self.minimum,self.maximum):w.textChanged.connect(self._range_edited)
        self.weighting.currentIndexChanged.connect(self._edited);self.loss.currentIndexChanged.connect(self._edited)
        self.loss.currentIndexChanged.connect(self._sync_loss_controls)
        self._controls=controls
        self._source_changed()
        self._sync_loss_controls()
        if saved and saved.get('source',{}).get('key') in [d.key for d in datasets]:
            self.source_combo.setCurrentIndex([d.key for d in datasets].index(saved['source']['key']))
            config=saved.get('result',{}).get('config',{})
            self._confidence=float(config.get('confidence',.95))
            index=self.model_combo.findData(config.get('model'));self.model_combo.setCurrentIndex(max(0,index))
            if config.get('model')=='custom':
                self.expression.setText(config.get('custom_expression',''));self.custom_names.setText(', '.join(config.get('custom_parameters',())))
            self.reset_parameters(config)
            self.minimum.setText(str(config.get('x_min','')) if config.get('x_min') is not None else '')
            self.maximum.setText(str(config.get('x_max','')) if config.get('x_max') is not None else '')
            self.loss.setCurrentIndex(max(0,self.loss.findData(config.get('loss','linear'))))
            self.f_scale.setText(str(config['f_scale']) if config.get('f_scale') is not None else '')
            self.weighting.setCurrentIndex(max(0,self.weighting.findData(saved.get('source',{}).get('sigma_column'))))
            self._result=copy.deepcopy(saved);self._draft_dirty=False;self._show_result();self.apply_button.setEnabled(True);self.export_button.setEnabled(True)
    def dataset(self):return self.datasets[self.source_combo.currentIndex()]
    def _set_advanced_visible(self,visible):
        self.advanced_controls.setVisible(visible)
        self.advanced_toggle.setArrowType(QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow)
    def _set_diagnostics_visible(self,visible):
        self.results.setVisible(visible)
        self.preview.setMaximumHeight(180 if visible else 16777215)
        self.diagnostics_toggle.setArrowType(QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow)
    def _sync_loss_controls(self,*_):
        robust=self.loss.currentData()!='linear'
        self.f_scale.setVisible(robust);self.robust_scale_label.setVisible(robust)
        self.f_scale.setEnabled(robust)
        self._sync_constraint_summary()
    def _sync_constraint_summary(self):
        constraints=[]
        for row in range(self.parameters.rowCount()):
            items=[self.parameters.item(row,col) for col in range(5)]
            if any(item is None for item in items):continue
            name,start,lower,upper,fixed=items
            if fixed.checkState()==QtCore.Qt.Checked:
                constraints.append(f'{name.text()} fixed at {start.text()}')
            elif lower.text() not in ('-inf','-Infinity') or upper.text() not in ('inf','Infinity'):
                constraints.append(f'{name.text()}: {lower.text()} to {upper.text()}')
        weighting=self.weighting.currentText() or 'Uniform'
        scale=f'; scale {self.f_scale.text() or "automatic"}' if self.loss.currentData()!='linear' else ''
        self.constraint_summary.setText(f'{self.loss.currentText()} · {weighting} weights{scale}\n'
                                       + ('Constraints: '+ '; '.join(constraints) if constraints else 'No fixed parameters or finite bounds.'))
    def _sync_result_identity(self):
        if not self._result:
            self.result_identity.setText('No completed fit. Review the draft observations, then Fit.')
            return
        from ..analysis.curve_fitting import get_model
        model=get_model(self._result['result']['config']['model']).label
        source=self._result['source']['label']
        state=('Draft settings changed; Fit to update.' if self._draft_dirty
               else 'Completed result matches these settings.')
        if not self._result['result'].get('success',True):state+=' This result did not converge and cannot be applied.'
        if self._worker is not None:state='Fit running. The previous completed result remains available.'
        self.result_identity.setText(f'Last completed: {model} · {source}\n{state} Export writes this completed result.')
    def _edited(self,*_):
        self.apply_button.setEnabled(False)
        self._draft_dirty=bool(self._result)
        self._sync_constraint_summary();self._sync_result_identity()
        self.view_tabs.setCurrentIndex(0)
        if self._result:self.status.setText('Draft settings changed. Fit again to apply them. Export still writes the last completed fit.')
    def _range_edited(self,*_):
        self._edited();self._draw_draft_observations()
    def _draw_draft_observations(self):
        """Preview selection only; neither transform nor estimate the source data."""
        data=self.dataset();x=np.asarray(data.x,dtype=float);y=np.asarray(data.y,dtype=float)
        finite=np.isfinite(x)&np.isfinite(y);included=finite.copy();problem=''
        try:
            lower=float(self.minimum.text()) if self.minimum.text().strip() else -np.inf
            upper=float(self.maximum.text()) if self.maximum.text().strip() else np.inf
            if ((self.minimum.text().strip() and not np.isfinite(lower))
                    or (self.maximum.text().strip() and not np.isfinite(upper)) or lower>=upper):
                raise ValueError
            included&=(x>=lower)&(x<=upper)
        except ValueError:
            problem='Range is incomplete or invalid; showing all finite observations. '
        excluded=finite&~included
        self._draft_included_count=int(included.sum())
        self.draft_preview.clear()
        self.draft_preview.plot(x[included],y[included],pen=None,symbol='o',symbolSize=6,symbolBrush='#0072b2')
        if excluded.any():
            self.draft_preview.plot(x[excluded],y[excluded],pen=None,symbol='o',symbolSize=5,symbolBrush='#999999',symbolPen=None)
        self.draft_preview.setLabel('bottom',data.x_name,units=data.x_unit or None)
        self.draft_preview.setLabel('left',data.y_name,units=data.y_unit or None)
        self.draft_preview.setTitle(data.label)
        self.draft_summary.setText(problem+f'Blue: {included.sum()} observations in range. Gray: {excluded.sum()} outside range. '
                                   f'{(~finite).sum()} non-finite observations omitted. Model and uncertainty-column validity are checked when you Fit.')
    def _source_changed(self,*_):
        d=self.dataset();self.weighting.clear();self.weighting.addItem('Uniform',None)
        for name in d.sigma_columns:self.weighting.addItem(f'σ from {name}',name)
        self.units.setText(f'X: {d.x_name} [{d.x_unit or "unitless"}]; Y: {d.y_name} [{d.y_unit or "unitless"}]. Values and parameter units follow these storage units.')
        self.reset_parameters()
        self._draw_draft_observations()
    def reset_parameters(self,config=None):
        from ..analysis.curve_fitting import get_model
        config=config if isinstance(config,dict) else {}
        d=self.dataset();model=get_model(self.model_combo.currentData());custom=model.id=='custom'
        self.expression.setEnabled(custom);self.custom_names.setEnabled(custom);self.custom_controls.setVisible(custom)
        self.formula.setText(model.formula+'\n'+model.description)
        names=([n.strip() for n in self.custom_names.text().split(',') if n.strip()] if custom else model.parameter_names)
        try:guess=model.guess(d.x,d.y) if not custom else {n:1. for n in names};bounds=model.default_bounds(d.x,d.y) if not custom else {}
        except (ValueError,TypeError,FloatingPointError):guess={n:1. for n in names};bounds={}
        with QtCore.QSignalBlocker(self.parameters):
            self.parameters.setRowCount(len(names))
            for row,name in enumerate(names):
                lo,hi=config.get('bounds',{}).get(name,bounds.get(name,(-np.inf,np.inf)))
                lo = -np.inf if lo is None else lo
                hi = np.inf if hi is None else hi
                units=parameter_units(model.id,names,d.x_unit,d.y_unit)
                values=[name,config.get('fixed',{}).get(name,config.get('initial',{}).get(name,guess.get(name,1.))),lo,hi]
                for col,value in enumerate(values):
                    item=QtWidgets.QTableWidgetItem(repr(float(value)) if isinstance(value,(float,np.floating)) else str(value));self.parameters.setItem(row,col,item)
                    if col==0:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable);item.setToolTip('Unit: '+units[name])
                    else:item.setToolTip(item.text())
                item=QtWidgets.QTableWidgetItem();item.setFlags(QtCore.Qt.ItemIsUserCheckable|QtCore.Qt.ItemIsEnabled)
                item.setCheckState(QtCore.Qt.Checked if name in config.get('fixed',{}) else QtCore.Qt.Unchecked);self.parameters.setItem(row,4,item)
                item=QtWidgets.QTableWidgetItem(units[name]);item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.parameters.setItem(row,5,item)
        self._edited()
    def _custom_parameters_changed(self):
        names=[name.strip() for name in self.custom_names.text().split(',') if name.strip()]
        previous=[self.parameters.item(row,0).text() for row in range(self.parameters.rowCount())]
        if names==previous:return
        config=dict(initial={},bounds={},fixed={})
        for row,name in enumerate(previous):
            config['initial'][name]=self.parameters.item(row,1).text()
            config['bounds'][name]=[self.parameters.item(row,col).text() for col in (2,3)]
            if self.parameters.item(row,4).checkState()==QtCore.Qt.Checked:
                config['fixed'][name]=self.parameters.item(row,1).text()
        self.reset_parameters(config)
    def estimate_starts(self):
        """Refresh free starts without changing a user's bounds or fixed constants."""
        from ..analysis.curve_fitting import get_model
        model=get_model(self.model_combo.currentData());data=self.dataset()
        try:
            guess=model.guess(data.x,data.y) if model.id!='custom' else {}
            changes=[]
            for row in range(self.parameters.rowCount()):
                if self.parameters.item(row,4).checkState()==QtCore.Qt.Checked:continue
                name=self.parameters.item(row,0).text()
                if name not in guess:continue
                lo,hi=[float(self.parameters.item(row,c).text()) for c in (2,3)]
                if np.isnan(lo) or np.isnan(hi) or lo>=hi:raise ValueError(f'Invalid bounds for {name}.')
                changes.append((row,float(np.clip(guess[name],lo,hi))))
            with QtCore.QSignalBlocker(self.parameters):
                for row,value in changes:
                    item=self.parameters.item(row,1);item.setText(repr(value));item.setToolTip(item.text())
            self._edited()
            self.status.setText('Free starting values estimated; bounds and fixed values preserved.' if changes
                                else 'No automatic starts available. Enter custom starts in the table; constraints are unchanged.')
        except (ValueError,TypeError,FloatingPointError) as exc:self.status.setText(f'Starts unchanged: {exc}')
    def reset_constraints(self):
        from ..analysis.curve_fitting import get_model
        model=get_model(self.model_combo.currentData());data=self.dataset()
        try:bounds=model.default_bounds(data.x,data.y) if model.id!='custom' else {}
        except (ValueError,TypeError,FloatingPointError) as exc:
            self.status.setText(f'Constraints unchanged: {exc}');return
        with QtCore.QSignalBlocker(self.parameters):
            for row in range(self.parameters.rowCount()):
                name=self.parameters.item(row,0).text();lo,hi=bounds.get(name,(-np.inf,np.inf))
                self.parameters.item(row,2).setText(repr(float(lo)));self.parameters.item(row,3).setText(repr(float(hi)))
                self.parameters.item(row,4).setCheckState(QtCore.Qt.Unchecked)
        self._edited();self.status.setText('Model bounds restored and fixed parameters released. Starting values preserved; review them before fitting.')
    def fit_config(self):
        config=dict(model=self.model_combo.currentData(),initial={},bounds={},fixed={},
                    loss=self.loss.currentData(),weighting='sigma' if self.weighting.currentData() else 'uniform',confidence=self._confidence)
        for row in range(self.parameters.rowCount()):
            name=self.parameters.item(row,0).text();start,lo,hi=[float(self.parameters.item(row,col).text()) for col in (1,2,3)]
            if not np.isfinite(start) or np.isnan(lo) or np.isnan(hi) or lo>=hi:raise ValueError(f'Invalid start/bounds for {name}.')
            if not lo<=start<=hi:raise ValueError(f'{name}: starting value lies outside its bounds.')
            config['initial'][name]=start;config['bounds'][name]=[lo,hi]
            if self.parameters.item(row,4).checkState()==QtCore.Qt.Checked:config['fixed'][name]=start
        for key,widget in [('x_min',self.minimum),('x_max',self.maximum)]:
            if widget.text().strip():
                value=float(widget.text());
                if not np.isfinite(value):raise ValueError('Fit range must be finite.')
                config[key]=value
        if config.get('x_min',-np.inf)>=config.get('x_max',np.inf):raise ValueError('Fit minimum must be smaller than maximum.')
        if config['loss']!='linear' and self.f_scale.text().strip():
            config['f_scale']=float(self.f_scale.text())
            if not np.isfinite(config['f_scale']) or config['f_scale']<=0:raise ValueError('Robust scale must be a positive finite number.')
        if config['model']=='custom':config.update(custom_expression=self.expression.text().strip(),custom_parameters=[self.parameters.item(i,0).text() for i in range(self.parameters.rowCount())])
        return config
    def start_fit(self):
        if self._worker is not None:return
        try:config=self.fit_config()
        except (ValueError,TypeError) as exc:self.status.setText(str(exc));return
        data=self.dataset();sigma_name=self.weighting.currentData()
        sigma=None if sigma_name is None else np.asarray(data.sigma_columns[sigma_name],dtype=float).copy()
        self._fitting_source=data.source();self._fitting_source['sigma_column']=sigma_name
        if sigma is not None:self._fitting_source['sigma_digest']=hashlib.sha256(sigma.tobytes()).hexdigest()
        self._controls.setEnabled(False);self.fit_button.setEnabled(False);self.apply_button.setEnabled(False);self.cancel_button.setEnabled(True)
        self.status.setText('Fitting… You can cancel while the solver is running.')
        from .worker_thread import start_worker_thread, JOB_NONE
        self._worker=_FitWorker(data,config,sigma)
        self._worker.computed.connect(self._computed);self._worker.failed.connect(self.status.setText)
        self._worker.finished.connect(self._finished)
        # These isolated, immutable figure calculations never write the shared ResultStore.
        thread=start_worker_thread(self._worker,domain=JOB_NONE,label='Nonlinear curve fit',owner=self)
        self._lifetime.update(worker=self._worker,thread=thread)
        self._sync_result_identity()
    def cancel_fit(self):
        if self._worker:self._worker._cancel_event.set();self.status.setText('Cancelling…')
    def _computed(self,result):
        self._result=dict(schema_version=1,source=self._fitting_source,result=result,
            parameter_units=parameter_units(result['config']['model'],result['parameters'],
                                             self._fitting_source['x_unit'],self._fitting_source['y_unit']))
        self._draft_dirty=False
        self._show_result();self.apply_button.setEnabled(bool(result.get('success',True)));self.export_button.setEnabled(True)
        self.status.setText('Fit completed. Review residuals and warnings, then Apply to figure.' if result.get('success') else 'Fit did not converge. Review the reported warnings and revise the model or starting values.')
    def _finished(self):
        self._worker=None
        _retire(self._lifetime)
        self._controls.setEnabled(True);self.fit_button.setEnabled(True);self.cancel_button.setEnabled(False)
        self._sync_result_identity()
        if self._close_pending:super().reject()
    def _show_result(self):
        state=self._result;r=state['result'];src=state['source'];self.preview.clear();self.residual.clear()
        self.preview.plot(r.get('x',[]),r.get('y',[]),pen=None,symbol='o',symbolSize=5,symbolBrush='#888888')
        self.preview.plot(r.get('x_curve',[]),r.get('y_curve',[]),pen=pg.mkPen('#0072b2',width=2))
        for lo,hi,color in [('confidence_lower','confidence_upper',(0,114,178,60)),('prediction_lower','prediction_upper',(120,120,120,25))]:
            if r.get(lo) and r.get(hi):
                a=self.preview.plot(r['x_curve'],r[lo],pen=None);b=self.preview.plot(r['x_curve'],r[hi],pen=None)
                self.preview.addItem(pg.FillBetweenItem(a,b,brush=pg.mkBrush(color)))
        self.preview.setLabel('bottom',src['x_name'],units=src.get('x_unit') or None);self.preview.setLabel('left',src['y_name'],units=src.get('y_unit') or None)
        self.residual.setLabel('bottom',src['x_name'],units=src.get('x_unit') or None)
        self.residual.setLabel('left','Residual',units=src.get('y_unit') or None)
        self.residual.getAxis('left').setToolTip(f'{src["y_name"]}: observed minus fitted response, in {src.get("y_unit") or "unitless"} storage units.')
        self.residual.plot(r.get('x',[]),r.get('residuals',[]),pen=None,symbol='o',symbolSize=5,symbolBrush='#0072b2')
        self.residual.addItem(pg.InfiniteLine(pos=0,angle=0,pen=pg.mkPen('#888888')))
        self.results.setPlainText(result_summary(state))
        self._show_estimates()
        self.view_tabs.setTabEnabled(1,True);self.view_tabs.setCurrentIndex(1)
        self._sync_result_identity()
    def _show_estimates(self):
        state=self._result;result=state['result'];config=result.get('config',{});params=result.get('parameters',{})
        units=parameter_units(config['model'],params,state['source'].get('x_unit',''),state['source'].get('y_unit',''))
        units.update(state.get('parameter_units',{}));intervals=result.get('parameter_ci',{})
        confidence=100*float(config.get('confidence',.95))
        self.estimates.setHorizontalHeaderLabels(['Parameter','Estimate','Unit',f'{confidence:g}% interval','Status'])
        self.estimates.setRowCount(len(params))
        for row,(name,value) in enumerate(params.items()):
            interval=intervals.get(name);bounds=config.get('bounds',{}).get(name,(None,None))
            constrained=any(v is not None and np.isfinite(v) for v in bounds)
            status='Fixed' if name in config.get('fixed',{}) else ('Bounded' if constrained else 'Free')
            values=[name,f'{value:.6g}',units.get(name,''),
                    'Unavailable' if interval is None else f'{interval[0]:.6g} to {interval[1]:.6g}',status]
            for col,text in enumerate(values):self.estimates.setItem(row,col,QtWidgets.QTableWidgetItem(text))
        self.estimates.resizeRowsToContents()
        self.estimates.setMinimumHeight(min(190,self.estimates.horizontalHeader().height()+sum(self.estimates.rowHeight(r) for r in range(len(params)))+4))
        self.estimates.setMaximumHeight(self.estimates.minimumHeight())
        metrics=result.get('metrics',{});summary=[]
        for key,label in [('n','Observations'),('r_squared','R²'),('rmse','RMSE')]:
            value=metrics.get(key)
            if isinstance(value,(int,float)) and np.isfinite(value):summary.append(f'{label}: {value:.6g}')
        summary.append('Raw-response residuals')
        self.result_metrics.setText(' · '.join(summary))
        warnings=result.get('warnings',[])
        self.result_warnings.setText('\n'.join(warnings));self.result_warnings.setVisible(bool(warnings))
    def apply_result(self):
        if self._result and self.apply_button.isEnabled():
            self.applied.emit(copy.deepcopy(self._result))
    def clear_applied(self):
        self.cleared.emit();self.clear_button.setEnabled(False)
        self.status.setText('Applied nonlinear fit removed. You can fit these data again.')
    def export_result(self,path=None):
        if not self._result:return None
        # A native save dialog runs a nested event loop: a replacement fit can
        # finish there. Export the completed record identified when clicked.
        completed=copy.deepcopy(self._result)
        if not isinstance(path,(str,Path)):
            path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Export last completed fit','nonlinear-fit.json','Fit record (*.json);;Fit observations and curves (*.csv)')
        if not path:return None
        p=Path(path)
        try:
            if p.suffix.lower()=='.csv':
                rows=result_rows(completed);fields=list(dict.fromkeys(k for row in rows for k in row))
                buffer=io.StringIO(newline='')
                writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();writer.writerows(rows)
                text=buffer.getvalue()
            else:text=json.dumps(completed,indent=2,ensure_ascii=False)+'\n'
            with staged_write(p) as f:
                f.write(text.encode('utf-8'))
        except OSError as exc:self.status.setText(f'Export failed: {exc}');return None
        self.status.setText(f'Saved {p.name}: {completed["result"]["config"]["model"]} / {completed["source"]["label"]}');return p
    def reject(self):
        if self._worker:self._close_pending=True;self.cancel_fit();return
        super().reject()
    def closeEvent(self,event):
        if self._worker:self._close_pending=True;self.cancel_fit();event.ignore()
        else:super().closeEvent(event)


def validate_fit_state(state):
    """Validate persisted geometry before it reaches graphics; never re-fit it."""
    from ..analysis.curve_fitting import CurveFitResult, get_model, compile_custom_expression
    if (not isinstance(state,dict) or type(state.get('schema_version')) is not int
            or state['schema_version'] != 1):
        raise ValueError('Unsupported nonlinear fit record.')
    source=state.get('source',{})
    if not isinstance(source,dict) or any(not isinstance(source.get(key),str) for key in
            ('key','label','x_name','y_name','x_unit','y_unit','input_digest')):
        raise ValueError('Nonlinear fit source metadata is incomplete.')
    units=state.get('parameter_units',{})
    if (not isinstance(units,dict)
            or any(not isinstance(k,str) or not isinstance(v,str) for k,v in units.items())):
        raise ValueError('Nonlinear fit parameter units must map parameter names to text.')
    if 'surface' in state and not isinstance(state['surface'],str):
        raise ValueError('Nonlinear fit surface metadata is invalid.')
    if 'n_input' in source and (type(source['n_input']) is not int or source['n_input'] < 2):
        raise ValueError('Nonlinear fit input count is invalid.')
    if source.get('sigma_column') is not None and not isinstance(source['sigma_column'],str):
        raise ValueError('Nonlinear fit uncertainty column is invalid.')
    if 'sigma_digest' in source and not isinstance(source['sigma_digest'],str):
        raise ValueError('Nonlinear fit uncertainty digest is invalid.')
    provenance=source.get('provenance',{})
    if not isinstance(provenance,dict):
        raise ValueError('Nonlinear fit provenance must be a mapping.')
    try:
        json.dumps(provenance,allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ValueError('Nonlinear fit provenance must contain finite JSON values.') from exc
    try:
        result=CurveFitResult.from_dict(state['result'])
        get_model(result.config['model'])
        if result.config['model']=='custom':
            compile_custom_expression(result.config['custom_expression'],result.config['custom_parameters'])
        for keys in [('x','y','y_fit','residuals'),('x_curve','y_curve')]:
            arrays=[np.asarray(getattr(result,k),dtype=float) for k in keys]
            if any(a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() for a in arrays):
                raise ValueError('Nonlinear fit coordinates must be finite one-dimensional arrays.')
            if len({len(a) for a in arrays})!=1:raise ValueError('Nonlinear fit coordinate lengths differ.')
        for key in ('confidence_lower','confidence_upper','prediction_lower','prediction_upper'):
            a=getattr(result,key)
            if a is not None and (a.ndim!=1 or len(a)!=len(result.x_curve) or not np.isfinite(a).all()):
                raise ValueError('Nonlinear fit uncertainty coordinates are invalid.')
        if not result.success:raise ValueError('An unsuccessful nonlinear fit cannot be applied.')
        if any(not np.isfinite(v) for v in result.parameters.values()):raise ValueError('Nonlinear fit parameters are invalid.')
        if set(units)-set(result.parameters):raise ValueError('Nonlinear fit units name unknown parameters.')
        if source.get('n_input',len(result.x)) < len(result.x):raise ValueError('Nonlinear fit input count is smaller than the fitted sample.')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('Incomplete nonlinear fit result.') from exc
    normalized=copy.deepcopy(state);normalized['result']=result.to_dict()
    return normalized


class NonlinearFitController(QtCore.QObject):
    """Page adapter: source-safe curve/band drawing and exact JSON persistence."""
    def __init__(self,page,provider,plot,notify,surface):
        super().__init__(page)
        self.page=page;self.provider=provider;self.plot=plot;self.notify=notify;self.surface=surface
        self.state=None;self.items=[];self.label=None;self.dialog=None
    def open(self):
        datasets=self.provider()
        if not datasets:
            self.page.statusBar().showMessage('Plot numeric X and Y data before opening a nonlinear fit.',10000);return
        if self.dialog is not None:
            self.dialog.raise_();self.dialog.activateWindow();return
        display_log=(self.plot.getAxis('bottom').logMode,self.plot.getAxis('left').logMode)
        self.dialog=NonlinearFitDialog(datasets,self.page,saved=self.state,display_log=display_log)
        self.dialog.applied.connect(self.install);self.dialog.cleared.connect(self.clear)
        self.dialog.finished.connect(self._dialog_finished);self.dialog.show()
    def _dialog_finished(self,*_):
        dialog=self.dialog;self.dialog=None
        if dialog:dialog.deleteLater()
    def _matches(self,state):
        source=state.get('source',{})
        for data in self.provider():
            if data.source()['input_digest']!=source.get('input_digest'):continue
            sigma=source.get('sigma_column')
            if sigma:
                if sigma not in data.sigma_columns:return False
                digest=hashlib.sha256(np.asarray(data.sigma_columns[sigma],dtype=float).tobytes()).hexdigest()
                if digest!=source.get('sigma_digest'):return False
            return True
        return False
    def install(self,state):
        candidate=validate_fit_state(state)
        if not self._matches(candidate):
            if self.dialog:self.dialog.status.setText('Source or plotted coordinates changed. Close this dialog and fit the current figure.');self.dialog.apply_button.setEnabled(False)
            return False
        candidate['surface']=self.surface
        self._replace_state(candidate);self.notify()
        if self.dialog:
            self.dialog.clear_button.setEnabled(True)
            self.dialog.status.setText('Fit applied to figure. Settings and diagnostics will be saved with the project.')
        return True
    def clear(self):
        self.state=None;self.clear_graphics();self.sync_controls();self.notify()
    def snapshot(self):return copy.deepcopy(self.state)
    def restore(self,state):
        self._replace_state(validate_fit_state(state) if state is not None else None)
    def clear_graphics(self):
        for item in self.items:
            try:self.plot.removeItem(item)
            except RuntimeError:pass
        self.items=[]
        if self.label is not None:
            try:self.label.setParentItem(None)
            except RuntimeError:pass
            self.label=None
    def redraw(self,validate=True):
        state=self.state
        if state and validate and not self._matches(state):
            state=None
            self.page.statusBar().showMessage('Nonlinear fit cleared because the plotted source or coordinates changed.',10000)
        self._replace_state(state)
    def _replace_state(self,state):
        """Stage graphics before replacing the last valid record and its overlay.

        Even a renderer/Qt failure must leave the previously applied fit usable. The
        staging helper removes every partly attached item if creation fails.
        """
        items,label=self._create_graphics(state) if state else ([],None)
        self.clear_graphics()
        self.state,self.items,self.label=state,items,label
        self.sync_controls()
    def sync_controls(self):
        """Keep the advanced model identity separate from any quick-fit selector."""
        button=getattr(self.page,'_nonlinear_fit_btn',None)
        if button is not None:button.setText('Curve fit…')
        status=getattr(self.page,'_nonlinear_fit_status',None)
        if status is not None:
            if self.state:
                from ..analysis.curve_fitting import get_model
                model=get_model(self.state['result']['config']['model']).label
                status.setText(f'Applied curve fit: {model} · {self.state["source"]["label"]}')
            else:status.setText('No curve fit applied.')
    def _create_graphics(self,state):
        from .figure_series import tag_pg_series
        r=state['result'];src=state['source'];color='#d55e00'
        curve=pg.PlotDataItem(r['x_curve'],r['y_curve'],pen=pg.mkPen(color,width=2))
        items=[curve];label=None
        if r.get('confidence_lower') and r.get('confidence_upper'):
            lower=pg.PlotDataItem(r['x_curve'],r['confidence_lower'],pen=None)
            upper=pg.PlotDataItem(r['x_curve'],r['confidence_upper'],pen=None)
            items.extend([lower,upper,pg.FillBetweenItem(lower,upper,brush=pg.mkBrush(213,94,0,45))])
        try:
            for i,item in enumerate(items):
                item._presentation_role='fit'
                item._nonlinear_confidence_band = i > 0
                descriptor = tag_pg_series(item,view=self.surface,role='fit',identity=f"nonlinear:{src['key']}",identity_kind='source',label=f"{src['label']} nonlinear fit",color=color)
                item._figure_series_meta = descriptor.to_dict()
                self.plot.addItem(item)
                if i > 0 and hasattr(self.page,'_style'):
                    styles=getattr(self.page._style,'series_styles',{})
                    item.setVisible(bool(self.page._style.show_ci_band and styles.get(descriptor.series_id,{}).get('visible',True)))
            label=pg.LabelItem(justify='right');label.setParentItem(self.plot.vb)
            label.anchor(itemPos=(1,1),parentPos=(1,1),offset=(-8,-8))
            metrics=r.get('metrics',{});rsq=metrics.get('r_squared',metrics.get('r2'))
            text=f"{r.get('config',{}).get('model','Nonlinear fit')} · {src['label']} · n={len(r.get('x',[]))}"
            if rsq is not None:text+=f' · R²={float(rsq):.4g}'
            label.setText(html.escape(text),color=color,size='9pt')
            return items,label
        except Exception:
            for item in items:
                self.plot.removeItem(item)
            if label is not None:label.setParentItem(None)
            raise
    def set_band_visible(self,visible):
        for item in self.items:
            if getattr(item,'_nonlinear_confidence_band',False):
                series_id=getattr(getattr(item,'_nanoscope_series_descriptor',None),'series_id','')
                styles=getattr(getattr(self.page,'_style',None),'series_styles',{})
                item.setVisible(bool(visible and styles.get(series_id,{}).get('visible',True)))
    def export_rows(self):return result_rows(self.state) if self.state else []
