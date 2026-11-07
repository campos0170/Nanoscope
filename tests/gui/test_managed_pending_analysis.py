"""Managed analytical inputs stay drafts beside the saved scientific result."""
import numpy as np
import pandas as pd
import pytest
from uuid import uuid4
from dataclasses import replace
from copy import deepcopy
from PySide6 import QtCore

from nanoscope.core.analysis_workspace import (
    capture_psd, restore_psd, capture_event_plots, restore_event_plots,
)
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, save_project, load_project
from nanoscope.core.session import Session
from nanoscope.gui import psd_panel, event_plot_panel
from nanoscope.gui.figure_bundle import capture_figure_snapshot
from nanoscope.gui.figure_adapter_pages import capture_applied_record
from nanoscope.core.figure_codecs import decode_legacy, encode_legacy
from nanoscope.core.figure_library import source_recipes

from .test_s3_units_psd_window import _doc
from .test_event_plot import _fed_window


def _noise(qtbot, session, doc_id=None):
    page = psd_panel.PSDWindow(session, defer_initial_refresh=True)
    qtbot.addWidget(page)
    page._nanoscope_figure_managed = True
    if doc_id is not None:
        page.set_target(doc_id, 0)
    return page


def _roundtrip(tmp_path, artifact):
    path = tmp_path / 'pending.nanoscope'
    save_project(path, ProjectState(workspace=AnalyticalWorkspace(artifacts=[artifact])))
    return load_project(path).workspace.artifacts[0]


def _count_kernel(monkeypatch, module, name, calls):
    original = getattr(module, name)
    def counted(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, counted)


def test_noise_inputs_wait_for_update_and_appearance_exports_keep_applied_values(
        qapp, qtbot, tmp_path, monkeypatch):
    session = Session()
    first = session.add_document(_doc(tmp_path, 'A.dat', units='pA'))
    second = session.add_document(_doc(tmp_path, 'B.dat', units='pA', sigma=9.0))
    page = _noise(qtbot, session, first)
    before = page._plot_data()
    original_psd = page.psd_result()[1]
    chosen, calls = [], []
    page.sourceSelected.connect(lambda doc_id, channel: chosen.append((doc_id, channel)))
    _count_kernel(monkeypatch, psd_panel.psd_mod, 'welch_psd', calls)
    _count_kernel(monkeypatch, psd_panel.psd_mod, 'fft_magnitude', calls)
    page._rec_combo.setCurrentIndex(next(i for i in range(page._rec_combo.count())
                                       if page._rec_combo.itemData(i) == (second, 0)))
    page._nperseg_combo.setCurrentText('1024')
    page._rms_lo_spin.setValue(200.0)
    assert chosen == [(second, 0)]
    assert calls == [] and page.inspector_adapter().pending().count >= 3
    np.testing.assert_array_equal(page.psd_result()[1], original_psd)
    pd.testing.assert_frame_equal(page._plot_data(), before)
    assert set(page._plot_data()['source_label']) == {'A.dat'}
    assert all(str(key).endswith('A.dat#0') for key in page._plot_data()['source_key'])
    assert page._plot.getAxis('left').labelText == 'PSD (pA²/Hz)'
    page.set_appearance_state({'font_family': 'Arial', 'title_size': 18})
    assert calls == []
    artifact = capture_psd(page)
    assert artifact.payload['config']['target_key'].endswith('A.dat#0')
    assert artifact.payload['config']['nperseg'] == 'Auto'
    assert artifact.payload['pending_analysis']['inputs']['target_key'].endswith('B.dat#0')
    assert '#load' not in artifact.payload['pending_analysis']['inputs']['target_key']
    assert page._current_target() == (second, 0)
    assert page._nperseg_combo.currentText() == '1024'
    assert chosen == [(second, 0)], 'capture/appearance must not look like source selection'
    applied = capture_applied_record(decode_legacy(
        artifact, figure_id=str(uuid4()), name='Shown A').record)
    applied_native = encode_legacy(applied.spec, applied.resources)
    assert 'pending_analysis' not in applied_native.payload
    snapshot = capture_figure_snapshot(
        page.active_plot_item(), data=page._plot_data(), config=artifact.payload['config'],
        provenance={'details': page.details_text(), 'source_recipes': source_recipes(applied_native)},
        applied_spec=applied.spec)
    pd.testing.assert_frame_equal(snapshot.dataframe(), before)
    assert snapshot.config()['target_key'].endswith('A.dat#0')
    assert snapshot.config()['rms_lo_hz'] == 0.0
    assert snapshot.provenance()['source_recipes']
    assert all(recipe['source_key'].endswith('A.dat')
               for recipe in snapshot.provenance()['source_recipes'])
    assert calls == []
    page.inspector_adapter().update()
    assert calls == ['welch_psd']
    assert page.curves()[0].doc_id == second
    assert page.inspector_adapter().pending().count == 0
    assert capture_psd(page).payload['pending_analysis'] is None
    page.set_view(psd_panel.VIEW_FFT)
    assert page.current_view() == psd_panel.VIEW_PSD
    assert calls == ['welch_psd']
    page.inspector_adapter().update()
    assert page.current_view() == psd_panel.VIEW_FFT
    assert 'fft_magnitude' in calls


def test_noise_colour_limits_keep_applied_estimator_in_spectrogram_caption(
        qapp, qtbot, tmp_path, monkeypatch):
    session = Session()
    source = session.add_document(_doc(tmp_path, 'A.dat', units='pA'))
    page = _noise(qtbot, session, source)
    page._overlap_spin.setValue(33)
    page.set_view(psd_panel.VIEW_SPECTROGRAM)
    page.inspector_adapter().update()
    assert 'hann' in page.caption_text()
    calls = []
    for name in ('welch_psd', 'spectrogram_psd'):
        _count_kernel(monkeypatch, psd_panel.psd_mod, name, calls)
    page._window_combo.setCurrentText('hamming')
    page.set_spectrogram_levels((-5.0, -1.0))
    assert calls == []
    assert 'hann' in page.caption_text() and 'hamming' not in page.caption_text()
    assert page._window_combo.currentText() == 'hamming'
    assert page.inspector_adapter().pending().count == 1


def test_canonical_noise_roundtrip_restores_source_before_its_segment_and_comparisons(
        qapp, qtbot, tmp_path, monkeypatch):
    session = Session()
    first = session.add_document(_doc(tmp_path, 'A.dat', units='pA'))
    longer = _doc(tmp_path, 'B.dat', units='pA')
    # The same sample count spans four seconds, versus A's two seconds.
    longer = replace(longer, meta=replace(longer.meta, sample_rate_hz=25_000.))
    second = session.add_document(longer)
    session.add_document(_doc(tmp_path, 'C.dat', units='pA'))
    page = _noise(qtbot, session, first)
    applied = page.psd_result()[1]
    page._select_combo_target((second, 0))
    page._start_spin.setValue(2.5)
    page._end_spin.setValue(3.5)
    item = next(page._compare_list.item(i) for i in range(page._compare_list.count())
                if page._compare_list.item(i).data(QtCore.Qt.UserRole).endswith('C.dat#0'))
    item.setCheckState(QtCore.Qt.Checked)
    native = capture_psd(page)
    record = decode_legacy(native, figure_id=str(uuid4()), name='Pending B + C').record
    # FigureSpec canonicalization sorts mappings; restore must still respect the
    # dependency between the primary source, its bounds and comparison choices.
    artifact = _roundtrip(tmp_path, encode_legacy(record.spec, record.resources))
    restored = _noise(qtbot, session)
    calls = []
    _count_kernel(monkeypatch, psd_panel.psd_mod, 'welch_psd', calls)
    assert not restore_psd(restored, artifact).skipped
    assert calls == []
    assert restored._current_target() == (second, 0)
    assert restored._start_spin.value() == pytest.approx(2.5)
    assert restored._end_spin.value() == pytest.approx(3.5)
    assert len(restored.compare_keys()) == 1 and restored.compare_keys()[0].endswith('C.dat#0')
    np.testing.assert_array_equal(restored.psd_result()[1], applied)
    assert capture_psd(restored).payload['pending_analysis'] == native.payload['pending_analysis']


def test_noise_frozen_calibration_provenance_changes_only_with_explicit_update(
        qapp, qtbot, tmp_path, monkeypatch):
    session = Session()
    source = session.add_document(_doc(tmp_path, 'A.dat', units='pA'))
    page = _noise(qtbot, session, source)
    before = page.psd_result()[1]
    original = session.calibration(source, 0)
    old_fingerprint = original.fingerprint()
    assert page.curves()[0].calibration_fingerprint == old_fingerprint
    calls = []
    _count_kernel(monkeypatch, psd_panel.psd_mod, 'welch_psd', calls)
    session.set_calibration(source, 0, replace(original, gain=original.gain * 2.0))
    assert calls == []
    native = capture_psd(page)
    assert native.payload['curves'][0]['calibration_fingerprint'] == list(old_fingerprint)
    record = decode_legacy(native, figure_id=str(uuid4()), name='Old calibration').record
    artifact = _roundtrip(tmp_path, encode_legacy(record.spec, record.resources))
    restored = _noise(qtbot, session)
    report = restore_psd(restored, artifact)
    assert not report.skipped and calls == []
    assert any('different calibration' in warning for warning in report.warnings)
    np.testing.assert_array_equal(restored.psd_result()[1], before)
    assert capture_psd(restored).payload['curves'][0]['calibration_fingerprint'] == list(old_fingerprint)
    # Older snapshots with no captured calibration cannot acquire today's one
    # merely because that recording is open while they are resaved.
    legacy_payload = deepcopy(artifact.payload)
    legacy_payload['curves'][0].pop('calibration_fingerprint')
    legacy = _noise(qtbot, session)
    assert not restore_psd(legacy, replace(artifact, payload=legacy_payload)).skipped
    assert legacy.curves()[0].calibration_fingerprint is None
    assert capture_psd(legacy).payload['curves'][0]['calibration_fingerprint'] is None
    assert calls == []
    restored.inspector_adapter().update()
    assert calls == ['welch_psd']
    assert restored.curves()[0].calibration_fingerprint == session.calibration(source, 0).fingerprint()
    assert capture_psd(restored).payload['curves'][0]['calibration_fingerprint'] == list(
        session.calibration(source, 0).fingerprint())
    np.testing.assert_allclose(restored.psd_result()[1], before * 4., rtol=2e-5)


@pytest.mark.parametrize('missing', [False, True])
def test_noise_draft_source_roundtrip_uses_persistent_identity_and_retains_frozen_arrays(
        qapp, qtbot, tmp_path, monkeypatch, missing):
    session = Session()
    first = session.add_document(_doc(tmp_path, 'A.dat', units='pA'))
    second = session.add_document(_doc(tmp_path, 'B.dat', units='pA'))
    page = _noise(qtbot, session, first)
    page._select_combo_target((second, 0))
    page._start_spin.setValue(.2)
    page._end_spin.setValue(1.2)
    artifact = _roundtrip(tmp_path, capture_psd(page))
    saved = page.psd_result()
    reopened = Session()
    reopened.add_document(_doc(tmp_path, 'A.dat', units='pA'))
    new_second = None if missing else reopened.add_document(_doc(tmp_path, 'B.dat', units='pA'))
    assert new_second != second
    restored = _noise(qtbot, reopened)
    calls = []
    _count_kernel(monkeypatch, psd_panel.psd_mod, 'welch_psd', calls)
    report = restore_psd(restored, artifact)
    assert not report.skipped and calls == []
    assert set(restored._plot_data()['source_label']) == {'A.dat'}
    assert capture_psd(restored).payload['config']['target_key'].endswith('A.dat#0')
    for actual, expected in zip(restored.psd_result(), saved):
        np.testing.assert_array_equal(actual, expected)
    assert restored._start_spin.value() == pytest.approx(.2)
    assert restored._end_spin.value() == pytest.approx(1.2)
    assert restored._pending_target_key().endswith('B.dat#0')
    assert restored.inspector_adapter().pending().count > 0
    assert capture_psd(restored).payload['pending_analysis'] == artifact.payload['pending_analysis']
    if missing:
        assert 'Unavailable recording' in restored._rec_combo.currentText()
        assert 'selected recording unavailable' in restored.inspector_adapter().pending().reasons
        assert restored.inspector_adapter().update() is False
        assert calls == []
        np.testing.assert_array_equal(restored.psd_result()[1], saved[1])
    else:
        assert restored._current_target() == (new_second, 0)
        restored.inspector_adapter().update()
        assert calls == ['welch_psd']
        assert restored._start_spin.value() == pytest.approx(.2)
        assert restored._end_spin.value() == pytest.approx(1.2)
        assert restored.inspector_adapter().pending().count == 0


@pytest.mark.parametrize('applied', [False, True])
def test_missing_noise_comparison_stays_checked_and_blocks_update_until_removed(
        qapp, qtbot, tmp_path, monkeypatch, applied):
    session = Session()
    first = session.add_document(_doc(tmp_path, 'A.dat', units='pA'))
    session.add_document(_doc(tmp_path, 'B.dat', units='pA'))
    page = _noise(qtbot, session, first)
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    if applied:
        page.inspector_adapter().update()
    artifact = _roundtrip(tmp_path, capture_psd(page))
    reopened = Session()
    reopened.add_document(_doc(tmp_path, 'A.dat', units='pA'))
    restored = _noise(qtbot, reopened)
    calls = []
    _count_kernel(monkeypatch, psd_panel.psd_mod, 'welch_psd', calls)
    assert not restore_psd(restored, artifact).skipped
    assert restored.compare_keys()[0].endswith('B.dat#0')
    assert 'Unavailable recording' in restored._compare_list.item(0).text()
    assert restored.inspector_adapter().update() is False and calls == []
    assert capture_psd(restored).payload['pending_analysis'] == artifact.payload['pending_analysis']
    restored._compare_list.item(0).setCheckState(QtCore.Qt.Unchecked)
    restored.inspector_adapter().update()
    assert calls == ['welch_psd'] and restored.inspector_adapter().pending().count == 0
    assert capture_psd(restored).payload['config']['compare_sources'] == []


def test_event_inputs_and_templates_wait_for_update_while_style_keeps_frozen_results(
        qapp, qtbot, tmp_path, monkeypatch):
    page = _fed_window(qtbot, tmp_path)
    page._nanoscope_figure_managed = True
    counts = page.density_counts()
    ensemble = page.ensemble_traces()
    data = page._plot_data()
    calls = []
    for name in ('density_2d', 'event_ensemble'):
        _count_kernel(monkeypatch, event_plot_panel.event_shape, name, calls)
    page._bins.setValue(20)
    page._normalize.setChecked(False)
    page._npoints.setValue(200)
    assert calls == [] and page.inspector_adapter().pending().count == 3
    np.testing.assert_array_equal(page.density_counts(), counts)
    for actual, expected in zip(page.ensemble_traces(), ensemble):
        np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(page._plot_data(), data)
    page.set_appearance_state({'font_family': 'Arial', 'title_size': 18})
    page.set_colormap('plasma')
    assert calls == []
    artifact = _roundtrip(tmp_path, capture_event_plots(page))
    assert artifact.payload['config']['bins'] == 40
    assert artifact.payload['config']['normalize_time'] is True
    assert artifact.payload['pending_analysis']['inputs']['bins'] == 20
    restored = event_plot_panel.EventPlotWindow(page._session)
    qtbot.addWidget(restored)
    restored._nanoscope_figure_managed = True
    restored.set_source(page._doc_id, page._channel, page._eventset, defer_analysis=True)
    assert not restore_event_plots(restored, artifact).skipped
    assert calls == [] and restored.inspector_adapter().pending().count == 3
    assert restored._align_combo.isEnabled()
    np.testing.assert_array_equal(restored.density_counts(), counts)
    assert capture_event_plots(restored).payload['pending_analysis'] == artifact.payload['pending_analysis']
    restored.refreshRequested.connect(restored._redraw)
    restored.inspector_adapter().update()
    assert calls == ['density_2d', 'event_ensemble']
    assert restored.density_counts().shape == (20, 20)
    assert restored.inspector_adapter().pending().count == 0
    assert restored._normalize_shown is False
    settings, appearance = restored.template_settings()
    restored.apply_template_settings({**settings, 'bins': 12}, appearance)
    assert calls == ['density_2d', 'event_ensemble']
    assert restored.inspector_adapter().pending().count == 1
