"""Follow selection stages Noise/Event sources; only Update commits their results."""
import numpy as np
import pandas as pd
import pytest

from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.figure_codecs import encode_legacy
from nanoscope.core.project import ProjectState, save_project, load_project
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui import psd_panel, event_plot_panel
from .test_event_plot import _doc_and_events
from .test_s3_units_psd_window import _doc
from .test_managed_pending_analysis import _count_kernel


def _window(qtbot, tmp_path, kind, monkeypatch):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    ids, results = [], []
    for name, sigma, count in (('A.dat', 2., 24), ('B.dat', 8., 12)):
        if kind == 'noise':
            document = _doc(tmp_path, name, units='pA', sigma=sigma)
        else:
            document, events = _doc_and_events(tmp_path, name=name, n_events=count)
        ids.append(window._session.add_document(document))
        if kind == 'event_waveforms':
            result = AnalysisResult(module_id='detect.schmitt', module_version='1.0',
                doc_id=ids[-1], channel=0, params={}, calibration_fingerprint=(1., 0.),
                payload=events)
            window._session.add_result(result)
            results.append(result)
    selected = [0]
    if results:
        monkeypatch.setattr(window._analysis_panel, 'current_detection',
                            lambda: results[selected[0]])
    window._current_selection = (ids[0], 0)
    return window, ids, results, selected


def _result(page, kind):
    return page.psd_result()[1] if kind == 'noise' else page.density_counts()


@pytest.mark.parametrize('kind', ['noise', 'event_waveforms'])
def test_follow_selection_preserves_applied_result_and_draft_through_project_roundtrip(
        qapp, qtbot, tmp_path, monkeypatch, kind):
    window, ids, results, selected = _window(qtbot, tmp_path, kind, monkeypatch)
    before_events = [result.payload.df.copy(deep=True) for result in results]
    calls = []
    if kind == 'noise':
        _count_kernel(monkeypatch, psd_panel.psd_mod, 'welch_psd', calls)
    else:
        for name in ('density_2d', 'event_ensemble'):
            _count_kernel(monkeypatch, event_plot_panel.event_shape, name, calls)
    owner = window._ensure_figure_workspace()
    key = owner.create(kind, binding={**window._figure_current_binding(kind), 'follow': True})
    page = owner.page(key)
    owner.sources_changed(selection_only=True)
    assert calls == [] and not page.has_data()
    assert owner.get(key).spec.result.status == 'empty'
    page.inspector_adapter().update()
    applied = _result(page, kind).copy()
    calls.clear()
    if kind == 'noise':
        page._nperseg_combo.setCurrentText('1024')
        page._start_spin.setValue(.2)
        page._end_spin.setValue(1.2)
    else:
        page._bins.setValue(20)
    selected[0] = 1
    window._current_selection = (ids[1], 0)
    owner.sources_changed(selection_only=True)
    assert calls == []
    np.testing.assert_array_equal(_result(page, kind), applied)
    assert 'A.dat' in window._analysis_explorer_context()
    assert 'B.dat' not in window._analysis_explorer_context()
    assert owner.get(key).spec.result.status == 'stale'
    assert page.inspector_adapter().pending().count >= 2
    assert owner.binding(key)['source']['source_key'].endswith('B.dat')
    owner.set_follow_selection(key, False)
    assert owner.binding(key)['source']['source_key'].endswith('B.dat')
    assert calls == []
    owner.set_follow_selection(key, True)
    assert calls == []
    assert owner.checkpoint(key)
    native = encode_legacy(owner.get(key).spec, owner.get(key).resources)
    if kind == 'noise':
        assert native.payload['config']['target_key'].endswith('A.dat#0')
        assert native.payload['pending_analysis']['inputs']['target_key'].endswith('B.dat#0')
    else:
        assert native.payload['source']['source_key'].endswith('A.dat')
        assert native.payload['pending_analysis']['inputs']['source']['source_key'].endswith('B.dat')
    path = tmp_path / f'{kind}.nanoscope'
    save_project(path, ProjectState(workspace=owner.capture_workspace()))
    reopened, new_ids, _results, new_selected = _window(qtbot, tmp_path, kind, monkeypatch)
    new_selected[0] = 1
    reopened._current_selection = (new_ids[1], 0)
    new_owner = reopened._ensure_figure_workspace()
    new_owner.install_workspace(load_project(path).workspace)
    assert new_owner.activate(key)
    restored = new_owner.page(key)
    assert calls == [], 'restore and follow staging must not run analytical kernels'
    np.testing.assert_array_equal(_result(restored, kind), applied)
    assert 'A.dat' in reopened._analysis_explorer_context()
    assert restored.inspector_adapter().pending().count >= 2
    restored.inspector_adapter().update()
    assert calls == (['welch_psd'] if kind == 'noise' else ['density_2d', 'event_ensemble'])
    assert restored.inspector_adapter().pending().count == 0
    assert new_owner.get(key).spec.result.status == 'current'
    assert 'B.dat' in reopened._analysis_explorer_context()
    if kind == 'noise':
        assert restored.curves()[0].doc_id == new_ids[1]
        assert restored._nperseg_used == 1024
        assert restored._start_spin.value() == pytest.approx(.2)
        assert restored._end_spin.value() == pytest.approx(1.2)
    else:
        assert restored._doc_id == new_ids[1]
        assert restored.density_counts().shape == (20, 20)
        assert restored.density_counts().sum() == 12
    for result, before in zip(results, before_events):
        pd.testing.assert_frame_equal(result.payload.df, before)
