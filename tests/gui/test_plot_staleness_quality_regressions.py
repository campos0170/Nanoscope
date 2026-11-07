"""Baseline judgement reaches retained figures without silently regenerating them."""
import numpy as np
import pytest
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectSchmitt
from nanoscope.gui.main_window import MainWindow
from .test_p3_regressions import _doc, _install, _trace


@pytest.fixture
def plotted(qtbot,tmp_path):
    window=MainWindow(recovery_enabled=False);qtbot.addWidget(window)
    window._confirm_close_choice=lambda:'discard'
    key,channel=_install(window,_doc(tmp_path/'apo-synthetic.dat',_trace()))
    session=window._session
    result=DetectSchmitt().run(SimpleAnalysisContext(session.calibrated(key,channel),
        session.document(key).meta,window._analysis_panel._form_params()))
    session.add_result(result)
    window._open_plots();page=window._plots_window
    page.set_sources([source.label for source in page.sources()])
    page.set_plot_type('Histogram');page.set_histogram_column('depth_pA')
    window._analysis_explorer.refresh_current()
    yield window,page
    window._dirty=False


def test_baseline_edit_and_revert_update_current_and_dormant_badges(plotted,qtbot):
    window,page=plotted;owner=window._figure_workspace;key=owner.current_id()
    assert owner.get(key).spec.result.status=='current'
    saved=page._plot_data().copy(deep=True)
    duplicate=owner.duplicate(key)
    assert owner.activate(key)
    assert owner.suspend(duplicate)
    assert owner.page(duplicate) is None
    page._plot.getViewBox().setRange(xRange=(10,80),yRange=(0,12),padding=0)
    ranges=page._plot.getViewBox().viewRange()
    knob=window._analysis_panel.baseline_panel.rough_s;original=knob.value()
    knob.setValue(original+1.5)
    qtbot.waitUntil(lambda:'STALE' in page.details_text(),timeout=3000)
    assert owner.get(key).spec.result.status=='stale'
    assert owner.get(duplicate).spec.result.status=='stale'
    np.testing.assert_allclose(page._plot.getViewBox().viewRange(),ranges)
    np.testing.assert_array_equal(page._plot_data().to_numpy(),saved.to_numpy())
    assert owner.page(duplicate) is None
    knob.setValue(original)
    qtbot.waitUntil(lambda:'STALE' not in page.details_text(),timeout=3000)
    assert owner.get(key).spec.result.status=='current'
    assert owner.get(duplicate).spec.result.status=='current'


def test_baseline_edit_does_not_stale_dwell_only_plot(plotted,qtbot):
    window,page=plotted;owner=window._figure_workspace;key=owner.current_id()
    page.set_histogram_column('dwell_s');window._analysis_explorer.refresh_current()
    window._analysis_panel.baseline_panel.rough_s.setValue(2.)
    qtbot.wait(400)
    assert owner.get(key).spec.result.status=='current'
    assert 'STALE' not in page.details_text()


def test_reverting_baseline_does_not_clear_independent_source_staleness(plotted):
    window,page=plotted;owner=window._figure_workspace;key=owner.current_id()
    knob=window._analysis_panel.baseline_panel.rough_s;original=knob.value()
    knob.setValue(original+1.5)
    assert owner.get(key).spec.result.status=='stale'
    owner.sources_changed()
    knob.setValue(original)
    assert owner.get(key).spec.result.status=='stale'


def test_reverting_baseline_keeps_pending_figure_settings_stale(plotted):
    window,page=plotted;owner=window._figure_workspace;key=owner.current_id()
    knob=window._analysis_panel.baseline_panel.rough_s;original=knob.value()
    knob.setValue(original+1.5)
    page._hist_col.showPopup()
    page._hist_col.setCurrentText('dwell_s')
    page._hist_col.hidePopup()
    assert page.inspector_adapter().pending().count
    owner.checkpoint(key)
    knob.setValue(original)
    assert owner.get(key).spec.result.status=='stale'
    assert page.inspector_adapter().pending().count
