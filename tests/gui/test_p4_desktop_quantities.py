"""Desktop display boundaries preserve raw inputs, tables and invalid legacy drafts."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.gui.baseline_panel import BaselinePanel, _OpenPoreReferenceEditor
from nanoscope.gui.incubation_window import _FrameModel
from nanoscope.gui.manual_study_window import _Form
from nanoscope.gui.measure_panel import _MeasurementsModel, _SORT_ROLE as MEASURE_SORT
from nanoscope.gui.panels import DetectionPanel, _EventTableModel, _SORT_ROLE as EVENT_SORT
from nanoscope.gui.time_controls import SecondsSpinBox, HoursSpinBox


def test_detection_windows_show_ms_but_params_and_programmatic_values_stay_seconds(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    for name in ('median_window_s', 'detrend_window_s', 'detrend_rough_s', 'detrend_base_s',
                 'rmean_window_s', 'savgol_window_s', 'min_dwell_s', 'max_dwell_s'):
        widget = getattr(panel, name)
        assert isinstance(widget, SecondsSpinBox)
        widget.setValue(.05)
        assert widget.value() == .05 and widget.text().startswith('50')
    assert panel.params()['min_dwell_s'] == .05
    panel.min_dwell_s.lineEdit().setText('25 ms')
    panel.min_dwell_s.interpretText()
    assert panel.params()['min_dwell_s'] == .025
    # An existing ms-valued control keeps its original numeric API.
    panel.merge_gap_ms.setValue(2)
    assert not isinstance(panel.merge_gap_ms, SecondsSpinBox)
    assert panel.params()['merge_gap_s'] == .002


def test_reference_intervals_inverse_and_history_preserve_invalid_legacy_drafts(qtbot):
    editor = _OpenPoreReferenceEditor()
    qtbot.addWidget(editor)
    editor.set_recording(10000, 5.)
    assert editor.add_interval(.05, .1)
    assert editor.table.item(0, 0).text() == '50.0'
    assert editor.params()['baseline_open_pore_reference']['intervals_s'] == [[.05, .1]]
    saved = editor.form_state()
    assert saved['interval_display'] == {'version': 1, 'units': [['ms', 'ms']]}
    editor.table.item(0, 0).setText('75')
    assert editor.params()['baseline_open_pore_reference']['intervals_s'] == [[.075, .1]]
    editor.restore_form_state(saved)
    assert editor.form_state() == saved
    legacy = deepcopy(saved)
    legacy.pop('interval_display')
    legacy['intervals'] = [['0.05', 'unfinished e']]
    editor.restore_form_state(legacy)
    assert editor.table.item(0, 0).text() == '50.0'
    assert editor.table.item(0, 1).text() == 'unfinished e'
    assert editor.form_state()['interval_display']['units'] == [['ms', 's']]
    assert editor.error()
    # Editing the retained invalid token explicitly adopts the current ms unit.
    editor.table.item(0, 1).setText('125')
    assert editor.params()['baseline_open_pore_reference']['intervals_s'] == [[.05, .125]]
    before = editor.form_state()
    with pytest.raises(ValueError):
        editor.restore_form_state({**before, 'interval_display': {'version': 999}})
    assert editor.form_state() == before


def test_baseline_estimator_window_report_matches_displayed_units(qtbot):
    panel = BaselinePanel()
    qtbot.addWidget(panel)
    assert isinstance(panel.rough_s, SecondsSpinBox)
    assert isinstance(panel.base_s, SecondsSpinBox)
    assert panel._windows_text(.08, .016) == '80 ms / 16 ms'


def test_event_table_has_friendly_ms_headers_but_raw_sort_edit_and_frame(qtbot):
    frame = pd.DataFrame({'t_start_s': [.002, .0005], 'dwell_s': [.00025, .002],
                          'ecd_pA_s': [.01, .2], 'start_idx': [200, 50],
                          'plugin_surprise_s': [3., 4.]})
    original = frame.copy(deep=True)
    model = _EventTableModel(frame)
    for key, displayed in (('t_start_s', '2'), ('dwell_s', '0.25'), ('ecd_pA_s', '0.01')):
        col = model._columns.index(key)
        index = model.index(0, col)
        assert model.data(index) == displayed
        assert model.data(index, EVENT_SORT) == frame.loc[0, key]
        assert float(model.data(index, QtCore.Qt.EditRole)) == frame.loc[0, key]
        assert key in model.headerData(col, QtCore.Qt.Horizontal, QtCore.Qt.ToolTipRole)
    assert model.headerData(1, QtCore.Qt.Horizontal) == 'Dwell time (ms)'
    assert 'units unknown' in model.headerData(4, QtCore.Qt.Horizontal)
    pd.testing.assert_frame_equal(frame, original)


def test_measurement_table_keeps_source_clock_stamps_and_label_editing(qtbot):
    model = _MeasurementsModel()
    model.set_df(pd.DataFrame({'measurement_id': [0], 't_start_s': [.002],
                              'dwell_s': [.00025], 't0_s': [61.3034], 'label': ['literal']}))
    raw = model.df.copy(deep=True)
    for key, displayed in (('t_start_s', '2'), ('dwell_s', '0.25'), ('t0_s', '61303.4')):
        col = model.column_index(key)
        assert model.data(model.index(0, col)) == displayed
        assert model.data(model.index(0, col), MEASURE_SORT) == raw.loc[0, key]
    label = model.index(0, model.column_index('label'))
    assert model.setData(label, 'literal seconds remain words')
    assert model.df.loc[0, 'label'] == 'literal seconds remain words'
    pd.testing.assert_frame_equal(model.df.drop(columns=['label']), raw.drop(columns=['label']))


def test_study_frame_metadata_uses_actual_ms_us_hours_and_unknown_basis(qtbot):
    frame = pd.DataFrame({'dwell_ms': [2.], 'median_dwell_us': [2000.],
                          'incubation_h': [1.], 'onset_s': [.05], 'mystery_s': [9.]})
    model = _FrameModel()
    model.set_frame(frame)
    assert [model.data(model.index(0, i)) for i in range(5)] == ['2', '2', '3600000', '50', '9']
    assert model.data(model.index(0, 2), QtCore.Qt.UserRole) == 1.
    pd.testing.assert_frame_equal(model.frame, frame)


def test_study_numeric_and_interval_forms_keep_seconds_and_hours_recipes(qtbot):
    form = _Form([('keep_start_s', 'Start (s)', float, .05),
                  ('incubation_h', 'Incubation (h)', float, 1.),
                  ('exclude_s', 'Excluded intervals', str, '.05:.1; invalid:.2'),
                  ('CUSUM_MAX_EVENT_S', 'Limit (s)', str, ''),
                  ('merge_ms', 'Merge (ms)', float, 3.)])
    qtbot.addWidget(form)
    assert isinstance(form.controls['keep_start_s'], SecondsSpinBox)
    assert isinstance(form.controls['incubation_h'], HoursSpinBox)
    assert form.controls['incubation_h'].text().endswith(' ms')
    assert form.controls['incubation_h'].valueFromText(form.controls['incubation_h'].text()) == 1.
    assert form.values()['keep_start_s'] == .05
    assert form.values()['incubation_h'] == 1.
    assert form.values()['CUSUM_MAX_EVENT_S'] == ''
    assert form.values()['merge_ms'] == 3.
    form.controls['exclude_s'].setText('50:100; invalid:200')
    values = form.values()
    first, second = values['exclude_s'].split(';')
    assert [float(v) for v in first.split(':')] == [.05, .1]
    assert second.split(':')[0].strip() == 'invalid' and float(second.split(':')[1]) == .2
    form.set_values(values)
    assert form.controls['exclude_s'].text() == '50:100; invalid:200'
    assert np.isclose(form.controls['incubation_h'].valueFromText('1800000 ms'), .5)


def test_trace_tick_unit_changes_without_moving_geometry_or_sample_hit(qtbot, tmp_path):
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.session import Session
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    from nanoscope.gui.trace_view import TraceView

    samples = np.arange(32, dtype=np.float32)
    info = ChannelInfo(index=0, name='current', raw_units='pA', kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / 'declared.dat', format_name='synthetic', n_samples=32,
                         sample_rate_hz=1000., rate_provenance=Provenance.FILE_META,
                         channels=(info,), t0_s=61.3034)
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(samples,)))
    layer = session.add_layer(doc_id, 0)
    before = session.x_axis(doc_id).values.copy()
    view = TraceView(session)
    qtbot.addWidget(view)
    axis = view._panes[0].plot.getAxis('bottom')
    assert axis.scale == 1000.
    assert 'ms' in axis.labelText and '61303.4' in axis.toolTip()
    assert session.time_to_index(doc_id, 0, before[2]) == 2
    assert view.crosshair._x_text(layer, session.x_axis(doc_id), 2) == '2.0 ms in  ·  61305.4 ms instrument'
    np.testing.assert_array_equal(session.x_axis(doc_id).values, before)
    np.testing.assert_array_equal(session.document(doc_id).channels[0], samples)
    from nanoscope.gui.navigator import Navigator
    navigator = Navigator()
    qtbot.addWidget(navigator)
    navigator.bind(session, view)
    assert navigator.getPlotItem().getAxis('bottom').scale == 1000.
    np.testing.assert_array_equal(session.x_axis(doc_id).values, before)



def test_summary_microsecond_metric_is_displayed_in_ms_without_mutating_summary(qtbot):
    from nanoscope.gui.panels import StatsPanel
    panel = StatsPanel()
    qtbot.addWidget(panel)
    stats = {'median_dwell_us': 250., 'rate_accepted_hz': 7.}
    original = deepcopy(stats)
    panel.set_summary(stats)
    assert 'ms' in panel.summary_label('median_dwell_us')
    assert float(panel.summary_text('median_dwell_us')) == .25
    assert float(panel.summary_text('rate_accepted_hz')) == 7.
    assert stats == original
