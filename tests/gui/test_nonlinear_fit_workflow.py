"""Real offscreen nonlinear workflows: worker, source binding, units and persistence."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
import time
import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore
from nanoscope.analysis.curve_fitting import fit_curve, FitCancelled
from nanoscope.core.session import Session
from nanoscope.gui.nonlinear_fit_dialog import CurveDataset, NonlinearFitDialog
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow

def dataset():
    x=np.linspace(.001,.009,100);y=2+7*np.exp(-.5*((x-.005)/.0008)**2)
    return CurveDataset('apo','Apo test',x,y,'dwell_s','depth_pA','s','pA',{'sigma':np.ones(100)*.1})

def fitted_state(data):
    return dict(schema_version=1,source=data.source(),result=fit_curve(data.x,data.y,{'model':'gaussian'}).to_dict())

def test_dialog_worker_applies_real_fit_and_exports(qapp,qtbot,tmp_path):
    d=NonlinearFitDialog([dataset()]);qtbot.addWidget(d)
    d.model_combo.setCurrentIndex(d.model_combo.findData('gaussian'))
    received=[];d.applied.connect(received.append)
    d.start_fit();qtbot.waitUntil(lambda:d._worker is None,timeout=15000)
    assert d.apply_button.isEnabled(),d.status.text()
    d.apply_result();assert len(received)==1
    r=received[0]['result'];assert abs(r['parameters']['center']-.005)<1e-8
    assert r['confidence_lower'];assert len(r['residuals'])==100
    assert received[0]['source']['x_unit']=='s'
    assert d.export_result(tmp_path/'fit.json').exists()
    assert d.export_result(tmp_path/'fit.csv').exists()
    assert 'nonlinear_curve' in (tmp_path/'fit.csv').read_text()
    assert json.loads((tmp_path/'fit.json').read_text())['result']['parameters']==r['parameters']

def test_worker_cancel_keeps_event_loop_alive(qapp,qtbot,monkeypatch):
    import nanoscope.analysis.curve_fitting as engine
    def slow(x,y,config,*,cancel,**kwargs):
        for _ in range(300):
            if cancel():raise FitCancelled()
            time.sleep(.005)
        raise AssertionError('Cancellation was not observed')
    monkeypatch.setattr(engine,'fit_curve',slow)
    d=NonlinearFitDialog([dataset()]);qtbot.addWidget(d)
    ticks=[];timer=QtCore.QTimer(d);timer.timeout.connect(lambda:ticks.append(1));timer.start(10)
    d.start_fit();qtbot.waitUntil(lambda:len(ticks)>4,timeout=2000)
    d.cancel_fit();qtbot.waitUntil(lambda:d._worker is None,timeout=2000)
    assert d._result is None and not d.apply_button.isEnabled()
    assert 'cancel' in d.status.text().lower()

def test_reopen_controls_preserve_fixed_weights_bounds(qapp,qtbot):
    data=dataset();state=fitted_state(data);state['source']['sigma_column']='sigma'
    state['result']['config'].update(fixed={'baseline':2.},weighting='sigma')
    d=NonlinearFitDialog([data],saved=state);qtbot.addWidget(d)
    cfg=d.fit_config();assert cfg['fixed']['baseline']==2.;assert cfg['weighting']=='sigma'
    assert np.isinf(cfg['bounds']['baseline'][1])
    d.parameters.item(0,1).setText('3');assert not d.apply_button.isEnabled()

def test_invalid_bounds_and_custom_equation_do_not_apply(qapp,qtbot):
    d=NonlinearFitDialog([dataset()]);qtbot.addWidget(d)
    d.parameters.item(0,2).setText('9');d.parameters.item(0,3).setText('1')
    d.start_fit();assert d._worker is None;assert 'Invalid' in d.status.text()
    d.model_combo.setCurrentIndex(d.model_combo.findData('custom'))
    d.expression.setText("__import__('os').getcwd()")
    d.start_fit();qtbot.waitUntil(lambda:d._worker is None,timeout=5000)
    assert not d.apply_button.isEnabled();assert d._result is None

@pytest.fixture
def scatter(qapp,qtbot):
    data=dataset();w=PlotsWindow(Session());qtbot.addWidget(w)
    w.set_data({'apo':pd.DataFrame({'dwell_s':data.x,'depth_pA':data.y,'sigma':data.sigma_columns['sigma']})})
    w.set_plot_type('Scatter');w.set_scatter_columns('dwell_s','depth_pA');return w

def test_scatter_real_source_overlay_export_and_restore(scatter,monkeypatch):
    import nanoscope.analysis.curve_fitting as engine
    data=scatter._nonlinear_drawn_datasets[0];np.testing.assert_array_equal(data.x,dataset().x)
    assert data.x_unit=='s' and data.y_unit=='pA'
    controller=scatter._nonlinear_controller();assert controller.install(fitted_state(data))
    state=scatter.nonlinear_fit_state();assert len(controller.items)==4
    exported=scatter._plot_data();assert 'nonlinear_curve' in set(exported.part)
    assert all(getattr(item,'_figure_series_meta',{}).get('series_id') for item in controller.items)
    monkeypatch.setattr(engine,'fit_curve',lambda *a,**k:pytest.fail('Restoration recomputed a fit'))
    scatter.restore_nonlinear_fit_state(json.loads(json.dumps(state)))
    assert scatter.nonlinear_fit_state()==state and len(controller.items)==4

def test_new_scatter_column_or_plot_type_invalidates_fit(scatter):
    c=scatter._nonlinear_controller();assert c.install(fitted_state(scatter._nonlinear_drawn_datasets[0]))
    scatter.set_scatter_columns('dwell_s','sigma');assert scatter.nonlinear_fit_state() is None and not c.items
    scatter.set_scatter_columns('dwell_s','depth_pA');assert c.install(fitted_state(scatter._nonlinear_drawn_datasets[0]))
    scatter.set_plot_type('Histogram');assert scatter.nonlinear_fit_state() is None and not c.items

def test_dialog_rejects_result_from_replaced_source(scatter):
    c=scatter._nonlinear_controller();state=fitted_state(scatter._nonlinear_drawn_datasets[0])
    scatter.set_data({'apo':pd.DataFrame({'dwell_s':dataset().x,'depth_pA':dataset().y*2,'sigma':np.ones(100)})})
    assert c.install(state) is False;assert c.state is None

def test_scatter_log_display_keeps_raw_fit_units(scatter):
    scatter._scatter_logx.setChecked(True);scatter._scatter_logy.setChecked(True)
    data=scatter._nonlinear_drawn_datasets[0]
    np.testing.assert_array_equal(data.x,dataset().x);np.testing.assert_array_equal(data.y,dataset().y)

def test_voltage_provider_uses_raw_seconds_and_volts(qapp,qtbot):
    import pyqtgraph as pg
    w=VDepWindow(Session());qtbot.addWidget(w);w._clear();w._drawn_metric='median_dwell_s';w._drawn_abs=True
    x=np.linspace(.1,.9,20);y=np.linspace(.001,.009,20)
    w._scatter_item=pg.PlotDataItem(x,y);w._plot.addItem(w._scatter_item);w._plot.setLogMode(y=True)
    data=w._nonlinear_datasets()[0];np.testing.assert_array_equal(data.x,x);np.testing.assert_array_equal(data.y,y)
    assert data.x_unit=='V' and data.y_unit=='s'

def test_saved_geometry_validation_rejects_mismatched_arrays(scatter):
    state=fitted_state(scatter._nonlinear_drawn_datasets[0])
    state['result']['y_curve']=state['result']['y_curve'][:-1]
    with pytest.raises(ValueError):scatter.restore_nonlinear_fit_state(state)
    assert scatter.nonlinear_fit_state() is None

def test_voltage_actual_points_and_same_values_new_recordings_invalidate(qapp,qtbot):
    from nanoscope.analysis.voltage_dep import VDepPoint
    w=VDepWindow(Session());qtbot.addWidget(w)
    x=np.linspace(-.9,-.1,12);y=2+3*np.exp(2*x)
    w._points=[VDepPoint(float(a),float(b),f'recording-{i}',doc_id=f'id-{i}') for i,(a,b) in enumerate(zip(x,y))]
    w._redraw()
    data=w._nonlinear_datasets()[0];assert data.provenance['source_doc_ids'][0]=='id-0'
    state=dict(schema_version=1,source=data.source(),result=fit_curve(data.x,data.y,{'model':'exp1'}).to_dict())
    assert w._nonlinear_controller().install(state)
    w._points=[p._replace(recording='replacement-'+p.recording,doc_id='new-'+p.doc_id) for p in w._points]
    w._redraw();assert w.nonlinear_fit_state() is None

def test_three_voltage_points_refuse_four_parameter_sigmoid(qapp,qtbot):
    data=CurveDataset('voltage','Three recordings',np.array([.4,.5,.6]),np.array([10.,20.,30.]),'voltage_V','capture_rate','V','Hz')
    d=NonlinearFitDialog([data]);qtbot.addWidget(d)
    d.model_combo.setCurrentIndex(d.model_combo.findData('logistic4'));d.start_fit()
    qtbot.waitUntil(lambda:d._worker is None,timeout=5000)
    assert not d.apply_button.isEnabled();assert 'parameter' in d.status.text().lower()

def test_invalid_histogram_range_preserves_previous_result(scatter):
    scatter.set_plot_type('Histogram');scatter.set_histogram_column('dwell_s');scatter.set_fit('exponential')
    before=scatter.fit_result()
    scatter._fit_lo.setText('9');scatter._fit_hi.setText('1');scatter._run_plot()
    assert scatter.fit_result() is before
    assert 'minimum must be smaller' in scatter.details_text()
    assert scatter._fit_lo.styleSheet()

def test_scatter_shared_diagnostic_toggles_control_saved_raw_residuals(scatter):
    c=scatter._nonlinear_controller();assert c.install(fitted_state(scatter._nonlinear_drawn_datasets[0]))
    scatter._ci_band_check.setChecked(True);scatter._residual_check.setChecked(True)
    assert all(item.isVisible() for item in c.items)
    assert not scatter._resid_widget.isHidden()
    assert scatter._presentation_residual[0]['mode']=='raw'
    np.testing.assert_allclose(scatter._presentation_residual[0]['residual'],c.state['result']['residuals'])
    scatter._ci_band_check.setChecked(False);scatter._residual_check.setChecked(False)
    assert c.items[0].isVisible() and not c.items[-1].isVisible()
    assert scatter._resid_widget.isHidden()

def test_gaussian_histogram_passes_range_to_conditional_likelihood(scatter,monkeypatch):
    from nanoscope.analysis import populations
    original=populations.fit_gaussian_mixture;seen=[]
    def fast(values,k=1,**kwargs):
        seen.append(dict(kwargs));kwargs['n_boot']=0
        return original(values,k=k,**kwargs)
    monkeypatch.setattr(populations,'fit_gaussian_mixture',fast)
    monkeypatch.setattr(scatter,'_deferred_note',lambda *a,**k:(False,None))
    scatter.set_plot_type('Histogram');scatter.set_histogram_column('depth_pA')
    scatter._fit_lo.setText('2.5');scatter._fit_hi.setText('6')
    scatter.set_fit('gaussian-mixture')
    assert scatter.fit_result() is not None,scatter.details_text()
    assert seen[-1]['lower']==2.5 and seen[-1]['upper']==6.
    assert scatter.fit_result().meta['truncated']

def test_exponential_gamma_point_worker_defers_200_bootstrap_replicates(monkeypatch):
    from nanoscope.analysis import stats
    sample=np.random.default_rng(1).gamma(3.,.004,80)+.001
    job=PlotsWindow._exp_gamma_job(sample,.001,None)
    result=job()
    assert isinstance(result,stats.FitResult)
    assert result.meta['n_boot_requested']==200 and result.meta['ci_pending']
    assert len(result._gui_deferred_ci_jobs)==1

def test_remove_applied_fit_clears_overlay_and_raw_diagnostics(scatter):
    c=scatter._nonlinear_controller();assert c.install(fitted_state(scatter._nonlinear_drawn_datasets[0]))
    scatter._residual_check.setChecked(True);c.clear()
    assert scatter.nonlinear_fit_state() is None and not c.items
    assert scatter._presentation_residual is None
    assert scatter._resid_widget.isHidden()
    assert 'Nonlinear fit:' not in scatter.details_text()
