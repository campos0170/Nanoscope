"""Offscreen user workflows: draft/result identity, constraints and discoverability."""
import copy
import csv
import json
import time

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.curve_fitting import FitCancelled, fit_curve
from nanoscope.core.session import Session
from nanoscope.gui.nonlinear_fit_dialog import CurveDataset, NonlinearFitDialog, parameter_units
from nanoscope.gui.plots_panel import PlotsWindow


@pytest.fixture
def data():
    x=np.linspace(.001,.009,80)
    return CurveDataset('apo','Gaussian source',x,2+7*np.exp(-.5*((x-.005)/.0008)**2),
                        'dwell_s','depth_pA','s','pA',{'sigma':np.full(len(x),.1)})


def state_for(data,**settings):
    result=fit_curve(data.x,data.y,dict(model='gaussian',**settings)).to_dict()
    return dict(schema_version=1,source=data.source(),result=result,
                parameter_units=parameter_units('gaussian',result['parameters'],'s','pA'))


def dialog_for(qtbot,data,saved=None,others=()):
    dialog=NonlinearFitDialog([data,*others],saved=saved)
    qtbot.addWidget(dialog);dialog.show()
    return dialog


def click(qtbot,button):
    qtbot.mouseClick(button,QtCore.Qt.LeftButton)


def test_first_open_has_data_and_simple_controls_without_running_solver(qapp,qtbot,monkeypatch,data):
    monkeypatch.setattr('nanoscope.analysis.curve_fitting.fit_curve',lambda *a,**kw:pytest.fail('Preview must not fit'))
    dialog=dialog_for(qtbot,data)
    assert dialog.view_tabs.currentIndex()==0
    assert not dialog.view_tabs.isTabEnabled(1)
    assert dialog.advanced_controls.isHidden() and dialog.custom_controls.isHidden()
    assert dialog.f_scale.isHidden() and not dialog.f_scale.isEnabled()
    observations=dialog.draft_preview.listDataItems()[0]
    np.testing.assert_array_equal(observations.xData,data.x)
    np.testing.assert_array_equal(observations.yData,data.y)
    assert not dialog.export_button.isEnabled() and not dialog.apply_button.isEnabled()


def test_range_edits_update_observations_without_fit_or_source_mutation(qapp,qtbot,monkeypatch,data):
    dialog=dialog_for(qtbot,data);original=data.x.copy()
    monkeypatch.setattr('nanoscope.analysis.curve_fitting.fit_curve',lambda *a,**kw:pytest.fail('Range preview must not fit'))
    dialog.minimum.setText('.003');dialog.maximum.setText('.006')
    included=(data.x>=.003)&(data.x<=.006)
    assert dialog._draft_included_count==included.sum()
    series=dialog.draft_preview.listDataItems();assert len(series)==2
    np.testing.assert_array_equal(series[0].xData,data.x[included])
    np.testing.assert_array_equal(series[1].xData,data.x[~included])
    dialog.minimum.setText('unfinished')
    assert 'invalid' in dialog.draft_summary.text()
    assert dialog._draft_included_count==len(data.x)
    np.testing.assert_array_equal(data.x,original)


def test_advanced_custom_and_robust_controls_follow_real_buttons(qapp,qtbot,data):
    dialog=dialog_for(qtbot,data)
    click(qtbot,dialog.advanced_toggle);assert dialog.advanced_controls.isVisible()
    dialog.loss.setCurrentIndex(dialog.loss.findData('huber'))
    assert dialog.f_scale.isVisible() and dialog.f_scale.isEnabled()
    dialog.f_scale.setText('-5')
    with pytest.raises(ValueError,match='Robust scale'):dialog.fit_config()
    dialog.loss.setCurrentIndex(dialog.loss.findData('linear'))
    assert dialog.f_scale.isHidden() and not dialog.f_scale.isEnabled()
    assert 'f_scale' not in dialog.fit_config()
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData('custom'))
    assert dialog.custom_controls.isVisible() and dialog.expression.isEnabled()
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData('gaussian'))
    assert dialog.custom_controls.isHidden()
    dialog.weighting.setCurrentIndex(1)
    click(qtbot,dialog.advanced_toggle)
    assert dialog.advanced_controls.isHidden()
    assert 'σ from sigma' in dialog.constraint_summary.text()
    assert 'sigma:' in dialog.constraint_summary.text()


def test_estimate_starts_preserves_bounds_and_fixed_values_reset_is_separate(qapp,qtbot,data):
    dialog=dialog_for(qtbot,data)
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData('gaussian'))
    click(qtbot,dialog.advanced_toggle)
    dialog.parameters.item(0,1).setText('2.125')
    dialog.parameters.item(0,4).setCheckState(QtCore.Qt.Checked)
    dialog.parameters.item(2,1).setText('.0065')
    dialog.parameters.item(2,2).setText('.006')
    dialog.parameters.item(2,3).setText('.007')
    before_bounds=[(dialog.parameters.item(r,2).text(),dialog.parameters.item(r,3).text()) for r in range(4)]
    click(qtbot,dialog.guess_button)
    assert dialog.parameters.item(0,1).text()=='2.125'
    assert dialog.parameters.item(0,4).checkState()==QtCore.Qt.Checked
    assert float(dialog.parameters.item(2,1).text())==.006
    assert [(dialog.parameters.item(r,2).text(),dialog.parameters.item(r,3).text()) for r in range(4)]==before_bounds
    assert 'baseline fixed at 2.125' in dialog.constraint_summary.text()
    starts=[dialog.parameters.item(r,1).text() for r in range(4)]
    click(qtbot,dialog.reset_constraints_button)
    assert [dialog.parameters.item(r,1).text() for r in range(4)]==starts
    assert all(dialog.parameters.item(r,4).checkState()==QtCore.Qt.Unchecked for r in range(4))
    assert float(dialog.parameters.item(3,2).text())>0
    assert np.isneginf(float(dialog.parameters.item(2,2).text()))


def test_edited_model_and_source_keep_completed_export_identity(qapp,qtbot,tmp_path,data):
    stored=state_for(data);original=copy.deepcopy(stored)
    other=CurveDataset('other','Other source',data.x,data.y*2,'time','amplitude','s','pA')
    dialog=dialog_for(qtbot,data,saved=stored,others=[other])
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData('linear'))
    dialog.source_combo.setCurrentIndex(1)
    assert not dialog.apply_button.isEnabled() and dialog.export_button.isEnabled()
    assert dialog.export_button.text()=='Export last completed fit…'
    assert 'Gaussian peak + baseline · Gaussian source' in dialog.result_identity.text()
    assert 'Draft settings changed' in dialog.result_identity.text()
    np.testing.assert_array_equal(dialog.draft_preview.listDataItems()[0].yData,other.y)
    assert dialog._result==original
    output=tmp_path/'completed.json';dialog.export_result(output)
    assert json.loads(output.read_text())==original
    dialog.view_tabs.setCurrentIndex(1)
    np.testing.assert_array_equal(dialog.preview.listDataItems()[0].yData,data.y)
    assert dialog.estimates.item(0,0).text()=='baseline'


def test_fit_button_populates_estimates_units_intervals_and_diagnostics(qapp,qtbot,data):
    data.y=data.y+np.random.default_rng(137).normal(0,.02,len(data.y))
    dialog=dialog_for(qtbot,data)
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData('gaussian'))
    click(qtbot,dialog.fit_button)
    qtbot.waitUntil(lambda:dialog._worker is None,timeout=15000)
    assert dialog.apply_button.isEnabled(),dialog.status.text()
    assert dialog.view_tabs.currentIndex()==1
    assert 'matches these settings' in dialog.result_identity.text()
    assert dialog.estimates.rowCount()==4 and dialog.results.isHidden()
    assert dialog.estimates.item(2,0).text()=='center'
    assert abs(float(dialog.estimates.item(2,1).text())-.005)<3e-5
    assert dialog.estimates.item(2,2).text()=='s'
    assert ' to ' in dialog.estimates.item(2,3).text()
    assert 'RMSE:' in dialog.result_metrics.text() and 'R²:' in dialog.result_metrics.text()
    click(qtbot,dialog.diagnostics_toggle);assert dialog.results.isVisible()
    received=[];dialog.applied.connect(received.append)
    click(qtbot,dialog.apply_button);assert len(received)==1
    assert received[0]['result']['parameters']==dialog._result['result']['parameters']


def test_saved_fixed_parameter_and_actual_confidence_are_visible(qapp,qtbot,data):
    dialog=dialog_for(qtbot,data,saved=state_for(data,confidence=.8,fixed={'baseline':2.}))
    assert dialog.estimates.horizontalHeaderItem(3).text()=='80% interval'
    assert dialog.estimates.item(0,4).text()=='Fixed'
    assert 'baseline fixed at 2' in dialog.constraint_summary.text()
    assert dialog.advanced_controls.isHidden()
    assert dialog.fit_config()['confidence']==.8


def test_estimated_start_remains_inside_high_precision_bounds(qapp,qtbot,data):
    dialog=dialog_for(qtbot,data)
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData('gaussian'))
    dialog.advanced_toggle.click()
    dialog.parameters.item(2,2).setText('.006000003')
    dialog.parameters.item(2,3).setText('.006000004')
    click(qtbot,dialog.guess_button)
    config=dialog.fit_config()
    assert config['initial']['center']==.006000003
    assert config['bounds']['center']==[.006000003,.006000004]


def test_saved_values_restore_without_rounding_bounds_fixed_or_starts(qapp,qtbot,data):
    saved=state_for(data)
    config=saved['result']['config']
    config['initial']['center']=.005000000031
    config['bounds']['center']=[.00500000003,.005000000032]
    config['fixed']['baseline']=2.000000031
    dialog=dialog_for(qtbot,data,saved=saved)
    actual=dialog.fit_config()
    assert actual['initial']['center']==config['initial']['center']
    assert actual['bounds']['center']==config['bounds']['center']
    assert actual['fixed']['baseline']==config['fixed']['baseline']


def test_custom_name_focus_and_extension_preserve_existing_constraints(qapp,qtbot,data):
    dialog=dialog_for(qtbot,data)
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData('custom'))
    dialog.parameters.item(0,1).setText('2.031')
    dialog.parameters.item(0,4).setCheckState(QtCore.Qt.Checked)
    dialog.parameters.item(1,2).setText('0.125')
    dialog.custom_names.editingFinished.emit()
    assert dialog.parameters.item(0,4).checkState()==QtCore.Qt.Checked
    dialog.custom_names.setText('baseline, amplitude, tau, offset')
    dialog.custom_names.editingFinished.emit()
    assert dialog.parameters.rowCount()==4
    assert dialog.parameters.item(0,1).text()=='2.031'
    assert dialog.parameters.item(0,4).checkState()==QtCore.Qt.Checked
    assert dialog.parameters.item(1,2).text()=='0.125'


@pytest.mark.parametrize('size',[(1000,760),(1240,900)])
def test_expanded_diagnostics_scroll_instead_of_squeezing_residual_labels(qapp,qtbot,data,size):
    dialog=dialog_for(qtbot,data,saved=state_for(data))
    dialog.resize(*size)
    click(qtbot,dialog.diagnostics_toggle);qapp.processEvents()
    assert dialog.residual.height()>=175
    assert dialog.preview.height()>=180
    if size[1]<800:assert dialog.completed_scroll.verticalScrollBar().maximum()>0
    assert dialog.results.geometry().bottom()<=dialog.completed_scroll.widget().height()
    assert dialog.apply_button.isVisible()
    assert dialog.export_button.isVisible()


def test_export_during_replacement_and_cancel_still_names_previous_result(qapp,qtbot,monkeypatch,tmp_path,data):
    stored=state_for(data)
    def slow(x,y,config,*,cancel,**kwargs):
        for _ in range(500):
            if cancel():raise FitCancelled()
            time.sleep(.002)
        pytest.fail('Cancel button did not reach worker')
    monkeypatch.setattr('nanoscope.analysis.curve_fitting.fit_curve',slow)
    dialog=dialog_for(qtbot,data,saved=stored)
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData('linear'))
    click(qtbot,dialog.fit_button)
    assert 'Fit running' in dialog.result_identity.text()
    assert dialog.export_button.isEnabled() and not dialog.apply_button.isEnabled()
    path=tmp_path/'previous.json';dialog.export_result(path)
    assert json.loads(path.read_text())==stored
    click(qtbot,dialog.cancel_button)
    qtbot.waitUntil(lambda:dialog._worker is None,timeout=5000)
    assert dialog._result==stored and 'Draft settings changed' in dialog.result_identity.text()
    assert not dialog.apply_button.isEnabled()


@pytest.mark.parametrize('suffix',['json','csv'])
def test_save_dialog_keeps_result_identified_when_export_was_clicked(qapp,qtbot,monkeypatch,tmp_path,data,suffix):
    original=state_for(data);dialog=dialog_for(qtbot,data,saved=original)
    replacement=copy.deepcopy(original);replacement['source']['key']='replacement'
    replacement['source']['label']='Replacement source'
    path=tmp_path/f'clicked-result.{suffix}'
    def choose_destination(*args):
        # Simulate a worker completion during QFileDialog's nested event loop.
        dialog._result=replacement;dialog._show_result()
        return str(path),''
    monkeypatch.setattr(QtWidgets.QFileDialog,'getSaveFileName',choose_destination)
    click(qtbot,dialog.export_button)
    if suffix=='json':
        assert json.loads(path.read_text())==original
    else:
        with path.open() as handle:rows=list(csv.DictReader(handle))
        assert rows and all(row['source_key']==original['source']['key'] for row in rows)
    assert 'Gaussian source' in dialog.status.text()
    assert dialog._result==replacement


@pytest.fixture
def scatter(qapp,qtbot,data):
    page=PlotsWindow(Session());qtbot.addWidget(page)
    page.set_data({'apo':pd.DataFrame({'dwell_s':data.x,'depth_pA':data.y})})
    page.set_plot_type('Scatter');page.set_scatter_columns('dwell_s','depth_pA')
    return page


def test_applied_curve_status_is_separate_from_quick_fit_on_restore_clear_and_replacement(scatter):
    controller=scatter._nonlinear_controller();data=scatter._nonlinear_drawn_datasets[0]
    assert controller.install(state_for(data))
    assert 'Applied curve fit: Gaussian peak + baseline' in scatter._nonlinear_fit_status.text()
    assert scatter._nonlinear_fit_btn.text()=='Curve fit…'
    assert 'Quick fit applied:' in scatter._scatter_fit_status.text()
    state=controller.snapshot();controller.clear()
    assert scatter._nonlinear_fit_status.text()=='No curve fit applied.'
    scatter.restore_nonlinear_fit_state(state)
    assert 'Gaussian peak + baseline' in scatter._nonlinear_fit_status.text()
    scatter.set_scatter_columns('depth_pA','dwell_s')
    assert scatter.nonlinear_fit_state() is None
    assert scatter._nonlinear_fit_status.text()=='No curve fit applied.'
    assert 'log-log space' in scatter._scatter_fit.toolTip()
    assert 'raw-response' in scatter._scatter_fit.toolTip()


def test_histogram_bins_describe_and_keep_immediate_preview_connection(scatter,monkeypatch):
    scatter.set_plot_type('Histogram')
    assert 'Live preview' in scatter._bins.toolTip()
    assert any(label.text()=='Bins (live preview)' for label in scatter.findChildren(QtWidgets.QLabel))
    redraw=[];monkeypatch.setattr(scatter,'_redraw',lambda:redraw.append(scatter._bins.value()))
    scatter._bins.setValue(scatter._bins.value()+1)
    assert redraw==[scatter._bins.value()]
