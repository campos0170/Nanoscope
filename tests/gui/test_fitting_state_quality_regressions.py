"""Real offscreen regressions for fit persistence and failure transactions."""
import copy
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.curve_fitting import fit_curve
from nanoscope.core.session import Session
from nanoscope.gui.nonlinear_fit_dialog import (
    CurveDataset, NonlinearFitDialog, parameter_units, result_summary, validate_fit_state,
)
from nanoscope.gui.plots_panel import PlotsWindow


@pytest.fixture
def data():
    x=np.linspace(.001,.009,80)
    return CurveDataset('apo','Apo',x,2+7*np.exp(-.5*((x-.005)/.0008)**2),
                        'dwell_s','depth_pA','s','pA')


def fitted(data,confidence=.95):
    result=fit_curve(data.x,data.y,{'model':'gaussian','confidence':confidence}).to_dict()
    return dict(schema_version=1,source=data.source(),result=result,
                parameter_units=parameter_units('gaussian',result['parameters'],'s','pA'))


@pytest.mark.parametrize('suffix',['json','csv'])
def test_partial_export_failure_preserves_previous_file(qapp,qtbot,tmp_path,monkeypatch,data,suffix):
    from nanoscope.gui import nonlinear_fit_dialog as module
    dialog=NonlinearFitDialog([data]);qtbot.addWidget(dialog);dialog._result=fitted(data)
    target=tmp_path/f'fit.{suffix}'
    assert dialog.export_result(target)==target
    original=target.read_bytes()
    stage=module.staged_write
    @contextmanager
    def fail_after_partial_write(path):
        with stage(path) as handle:
            class PartialWriter:
                def write(self,value):
                    handle.write(value[:23])
                    raise OSError(28,'No space left on device')
            yield PartialWriter()
    monkeypatch.setattr(module,'staged_write',fail_after_partial_write)
    assert dialog.export_result(target) is None
    assert target.read_bytes()==original
    assert list(tmp_path.iterdir())==[target]
    assert 'Export failed' in dialog.status.text()


def test_summary_labels_actual_confidence(data):
    state=fitted(data,.8)
    assert '80% parameter intervals' in result_summary(state)
    assert '95% parameter intervals' not in result_summary(state)


def test_residual_axes_preserve_quantity_units_and_linked_tick_scale(qapp,qtbot,data):
    dialog=NonlinearFitDialog([data],saved=fitted(data));qtbot.addWidget(dialog)
    dialog.show();qtbot.wait(30)
    dialog.preview.setXRange(.001,.009,padding=0);qapp.processEvents()
    main=dialog.preview.getAxis('bottom');residual=dialog.residual.getAxis('bottom')
    assert main.labelText==residual.labelText=='dwell_s'
    assert main.labelUnits==residual.labelUnits=='s'
    assert main.autoSIPrefixScale==residual.autoSIPrefixScale==1000
    assert dialog.residual.getAxis('left').labelUnits=='pA'
    assert dialog.residual.getAxis('left').labelText=='Residual'
    assert 'observed minus fitted response' in dialog.residual.getAxis('left').toolTip()
    # Pyqtgraph links screen coordinates when the two Y-axis labels need
    # different widths; equal numerical view limits are not required.
    from PySide6 import QtCore
    def displayed_x(widget,value):
        point=widget.getViewBox().mapViewToScene(QtCore.QPointF(value,0))
        return widget.mapToGlobal(widget.mapFromScene(point)).x()
    assert abs(displayed_x(dialog.preview,.005)-displayed_x(dialog.residual,.005))<=1


@pytest.mark.parametrize('log_x,log_y',[(True,False),(True,True),(False,True)])
def test_controller_preview_keeps_figure_log_display_and_signed_raw_residuals(qapp,qtbot,data,log_x,log_y):
    page=PlotsWindow(Session());qtbot.addWidget(page)
    page.set_data({'apo':pd.DataFrame({'dwell_s':data.x,'depth_pA':data.y})})
    page.set_plot_type('Scatter');page.set_scatter_columns('dwell_s','depth_pA')
    page._scatter_logx.setChecked(log_x);page._scatter_logy.setChecked(log_y)
    controller=page._nonlinear_controller()
    state=fitted(page._nonlinear_drawn_datasets[0]);assert controller.install(state)
    controller.open();dialog=controller.dialog
    assert dialog.preview.getAxis('bottom').logMode==log_x
    assert dialog.preview.getAxis('left').logMode==log_y
    assert dialog.residual.getAxis('bottom').logMode==log_x
    assert not dialog.residual.getAxis('left').logMode
    assert dialog._result==state | {'surface':'scatter'}
    observations=dialog.preview.listDataItems()[0];residuals=dialog.residual.listDataItems()[0]
    np.testing.assert_array_equal(observations.xData,state['result']['x'])
    np.testing.assert_array_equal(observations.yData,state['result']['y'])
    np.testing.assert_array_equal(residuals.yData,state['result']['residuals'])
    if log_x:
        np.testing.assert_allclose(observations.getData()[0],np.log10(state['result']['x']))
    dialog.reject()


@pytest.mark.parametrize('mutate',[
    lambda s:s.update(parameter_units=[]),
    lambda s:s.update(parameter_units={'sigma':[]}),
    lambda s:s.update(parameter_units={'unknown':'s'}),
    lambda s:s.update(schema_version=True),
    lambda s:s.update(surface=[]),
    lambda s:s['source'].update(n_input=True),
    lambda s:s['source'].update(provenance=[]),
    lambda s:s['source'].update(provenance={'scale':float('nan')}),
    lambda s:s['source'].update(sigma_column=[]),
])
def test_saved_ancillary_metadata_is_validated(data,mutate):
    state=fitted(data);mutate(state)
    with pytest.raises(ValueError):validate_fit_state(state)


@pytest.mark.parametrize('operation',['install','restore'])
def test_overlay_creation_failure_preserves_last_valid_state(qapp,qtbot,monkeypatch,data,operation):
    page=PlotsWindow(Session());qtbot.addWidget(page)
    page.set_data({'apo':pd.DataFrame({'dwell_s':data.x,'depth_pA':data.y})})
    page.set_plot_type('Scatter');page.set_scatter_columns('dwell_s','depth_pA')
    controller=page._nonlinear_controller()
    state=fitted(page._nonlinear_drawn_datasets[0]);assert controller.install(state)
    previous=controller.snapshot();items=list(controller.items);label=controller.label
    before_scene=set(page._plot.items)
    original=page._plot.addItem;calls=[]
    def fail_midway(item,*args,**kwargs):
        calls.append(item)
        original(item,*args,**kwargs)
        raise RuntimeError('Injected graphics failure')
    candidate=copy.deepcopy(state);candidate['source']['label']='Renamed'
    with monkeypatch.context() as patch:
        patch.setattr(page._plot,'addItem',fail_midway)
        with pytest.raises(RuntimeError,match='Injected graphics failure'):
            getattr(controller,operation)(candidate)
    assert controller.snapshot()==previous
    assert controller.items==items and controller.label is label
    assert set(page._plot.items)==before_scene
    assert all(item.scene() is not None for item in items)


def test_source_legend_rename_preserves_builtin_template(qapp,qtbot,data):
    page=PlotsWindow(Session());qtbot.addWidget(page)
    page.set_data({'apo':pd.DataFrame({'dwell_s':data.x,'depth_pA':data.y})})
    name='Dwell-time histogram (log-x, exponential)'
    page._on_template_selected(name)
    before=page.config().as_template()
    assert page._template_would_give(page._BUILTIN_TEMPLATES[name]).as_template()==before
    page._source_list.item(0).setText('Renamed apo')
    assert page.config().as_template()==before
    assert page._template_combo.currentText()==name


@pytest.mark.parametrize('background,tick_pt',[('light',7),('dark',11)])
def test_native_saved_fit_retains_residual_axis_appearance_without_refitting(
        qapp,qtbot,monkeypatch,data,background,tick_pt):
    from nanoscope.analysis import curve_fitting
    from nanoscope.gui.figure_adapter_pages import adapter_for
    page=PlotsWindow(Session());qtbot.addWidget(page)
    page.set_data({'apo':pd.DataFrame({'dwell_s':data.x,'depth_pA':data.y})})
    page.set_plot_type('Scatter');page.set_scatter_columns('dwell_s','depth_pA')
    page._scatter_logx.setChecked(True);page._run_plot()
    state=fitted(page._nonlinear_drawn_datasets[0])
    assert page._nonlinear_controller().install(state)
    page._residual_check.setChecked(True);page._ci_band_check.setChecked(True)
    page.set_background(background);page._tick_size.setValue(tick_pt)
    qapp.processEvents()
    def appearance(plot):
        return {side:(axis.pen().color().name(),axis.textPen().color().name(),
                      axis.style['tickFont'].toString() if axis.style['tickFont'] else None,
                      axis.logMode,axis.labelText,dict(axis.labelStyle))
                for side in ('left','bottom') for axis in [plot.getAxis(side)]}
    expected=appearance(page._resid)
    assert expected['left'][0]==page._fg_hex()
    assert page._resid.getAxis('left').style['tickFont'].pointSizeF()==tick_pt
    native=adapter_for(page,'distributions').capture_native()
    clone=PlotsWindow(Session());qtbot.addWidget(clone)
    def forbid_refit(*args,**kwargs):
        pytest.fail('Restoring residual appearance must not recompute the fit')
    monkeypatch.setattr(curve_fitting,'fit_curve',forbid_refit)
    report=adapter_for(clone,'distributions').restore_native(native)
    assert not report.skipped,report
    assert appearance(clone._resid)==expected
    assert clone._resid.getAxis('bottom').logMode
    assert not clone._resid.getAxis('left').logMode
    assert clone.nonlinear_fit_state()==page.nonlinear_fit_state()
    assert clone._presentation_residual==page._presentation_residual
    # Includes the saved fitted curve and both confidence-band boundaries.
    assert len(clone._nonlinear_fitter.items)==len(page._nonlinear_fitter.items)
    for original,restored in zip(page._nonlinear_fitter.items,clone._nonlinear_fitter.items):
        if hasattr(original,'xData'):
            np.testing.assert_array_equal(restored.xData,original.xData)
            np.testing.assert_array_equal(restored.yData,original.yData)
    main_range=copy.deepcopy(clone._plot.getViewBox().viewRange())
    residual_range=copy.deepcopy(clone._resid.getViewBox().viewRange())
    clone._style_residual_axes()
    assert clone._plot.getViewBox().viewRange()==main_range
    assert clone._resid.getViewBox().viewRange()==residual_range
