"""Display Jacobians and mixed config migration keep scientific values authoritative."""
from dataclasses import replace

import nanoscope  # noqa: F401
import numpy as np
import pytest

from nanoscope.gui.plots_panel import (
    PlotConfig, AxesConfig, HistogramConfig, FitConfig, StyleSettings, FrozenMap,
    migrate_plot_quantity, plot_config_to_json, plot_config_from_json, PlotsWindow)
from nanoscope.core.quantity_display import QuantityError


def test_old_density_display_limits_convert_but_raw_pins_and_words_do_not():
    old = PlotConfig(histogram=HistogramConfig(column='dwell_s', normalization='density'),
        axes=AxesConfig(time_unit='s', x_min='.001', x_max='.02', y_min='0', y_max='200',
                        manual_x=(.001, .02), manual_y=(0., 200.)),
        fit=FitConfig(range_lo='.002', range_hi='.01', range_unit='s'),
        style=StyleSettings(title='Literal seconds (s)', x_label='My time (s)'))
    current = migrate_plot_quantity(old)
    assert float(current.axes.x_min) == 1.
    assert float(current.axes.y_max) == .2
    assert current.axes.manual_x == old.axes.manual_x and current.axes.manual_y == old.axes.manual_y
    assert float(current.fit.range_lo) == 2.
    assert current.style == old.style
    assert current.identity.axis_factors == (1000., .001)
    assert migrate_plot_quantity(current) == current
    assert plot_config_from_json(plot_config_to_json(current)) == current


def test_future_basis_and_incomplete_legacy_numbers_refuse_without_relabeling():
    old = PlotConfig(histogram=HistogramConfig(column='dwell_s'),
                     axes=AxesConfig(time_unit='s', x_min='1e'))
    with pytest.raises(QuantityError):
        migrate_plot_quantity(old)
    assert old.axes.time_unit == 's' and old.axes.x_min == '1e'
    future = replace(old, axes=replace(old.axes, quantity_display=FrozenMap({'version': 9, 'axes': {}})))
    with pytest.raises(QuantityError):
        migrate_plot_quantity(future)


def test_density_axis_and_range_inverse_agree_without_touching_histogram(qtbot):
    from .test_p1_summaries_hosted import _source
    from nanoscope.core.session import Session
    page = PlotsWindow(Session()); qtbot.addWidget(page)
    page.set_data([_source()])
    page.set_histogram_column('dwell_s')
    page.set_normalization('density')
    page._run_plot()
    axis = page._plot.getAxis('left')
    assert axis.scale == .001
    assert 'ms' in axis.labelText
    assert page._axis_unit_factors() == (1000., .001)
    raw = page._plot_data().copy(deep=True)
    assert set(raw['dwell_s_display_unit']) == {'ms'}
    assert set(raw['density_display_unit']) == {'1/ms'}
    np.testing.assert_allclose(raw['dwell_s_display'], raw['dwell_s'] * 1000., equal_nan=True)
    np.testing.assert_allclose(raw['density_display'], raw['density'] * .001, equal_nan=True)
    page._presentation_changed()
    assert page._plot_data().equals(raw)
    assert page._figure_axis_state('left').shown_range == pytest.approx(
        tuple(value * .001 for value in page._plot.getViewBox().viewRange()[1]))


def test_unknown_and_sample_histogram_units_are_not_assigned_time():
    for column in ('undeclared_measurement', 'n_samples', 'n_levels'):
        config = migrate_plot_quantity(PlotConfig(histogram=HistogramConfig(column=column, normalization='density')))
        axes = config.axes.quantity_display['axes']
        assert axes['bottom']['quantity']['factor'] == 1.
        assert axes['left']['quantity']['factor'] == 1.


def test_column_picker_displays_names_but_returns_raw_recipe_keys(qtbot):
    from nanoscope.gui.quantity_selectors import ScientificColumnCombo
    from PySide6 import QtCore
    picker = ScientificColumnCombo(); qtbot.addWidget(picker)
    picker.addItems(['dwell_s', 'ecd_pA_s', 'frac_depth'])
    picker.setCurrentText('dwell_s')
    assert picker.currentData() == picker.currentText() == 'dwell_s'
    assert picker.itemText(picker.currentIndex()) == 'Dwell time (ms)'
    assert 'stored unit: s' in picker.itemData(picker.currentIndex(), QtCore.Qt.ToolTipRole)
    picker.setCurrentIndex(1)
    assert picker.currentText() == 'ecd_pA_s'
    assert 'ms' not in picker.itemText(picker.currentIndex())


@pytest.mark.parametrize('setter,attrs', [
    ('set_histogram_column', ('_hist_col',)),
    ('set_scatter_columns', ('_scatter_x', '_scatter_y')),
    ('set_kde_columns', ('_kde_x', '_kde_y')),
    ('set_heatmap_columns', ('_heatmap_x', '_heatmap_y')),
])
def test_every_column_setter_accepts_raw_keys_and_friendly_labels(qtbot, setter, attrs):
    from .test_p1_summaries_hosted import _source
    from nanoscope.core.session import Session
    page = PlotsWindow(Session()); qtbot.addWidget(page)
    page.set_data([_source()])
    keys = ('dwell_s', 'depth_pA')[:len(attrs)]
    getattr(page, setter)(*keys)
    assert tuple(getattr(page, attr).currentText() for attr in attrs) == keys
    labels = tuple(getattr(page, attr).itemText(getattr(page, attr).currentIndex()) for attr in attrs)
    getattr(page, setter)(*labels)
    assert tuple(getattr(page, attr).currentData() for attr in attrs) == keys
    # Typed, case-insensitive prefixes preserve the historical public API too.
    getattr(page, setter)(*(key.upper() for key in keys))
    assert tuple(getattr(page, attr).currentText() for attr in attrs) == keys


def test_column_setter_rejects_ambiguous_labels_without_changing_raw_selection(qtbot):
    from nanoscope.gui.quantity_selectors import ScientificColumnCombo
    from nanoscope.gui.combos import choose_combo_item
    picker = ScientificColumnCombo(); qtbot.addWidget(picker)
    picker.addItems(['dwell_s', 'dwell_ms', 'frac_depth'])
    choose_combo_item(picker, 'frac_depth', 'column')
    for ambiguous in ('dwell', 'Dwell time (ms)'):
        with pytest.raises(ValueError, match='ambiguous'):
            choose_combo_item(picker, ambiguous, 'column')
        assert picker.currentText() == picker.currentData() == 'frac_depth'
    choose_combo_item(picker, 'dwell_s', 'column')
    assert picker.currentText() == picker.currentData() == 'dwell_s'
    with pytest.raises(ValueError, match='unknown'):
        choose_combo_item(picker, 'absent', 'column')
    assert picker.currentText() == 'dwell_s'



def test_legacy_committed_picture_migrates_separately_from_another_pending_quantity():
    from nanoscope.gui.figure_adapter_plots import _config_for_applied
    applied = PlotConfig(histogram=HistogramConfig(column='dwell_s', normalization='density'),
                         axes=AxesConfig(time_unit='s', x_min='.002', x_max='.008', y_max='100'))
    applied_json = plot_config_to_json(applied)
    applied_json['axes'] = {'time_unit': 's'}  # authentic Phase 2 render envelope
    pending = replace(applied, histogram=HistogramConfig(column='frac_depth', normalization='density'))
    # Restoration uses the raw draft to retain the pre-migration units of shared boxes.
    restored = _config_for_applied(pending, applied_json)
    assert restored.histogram.column == 'dwell_s'
    assert float(restored.axes.x_min) == 2.
    assert float(restored.axes.y_max) == .1
    assert restored.axes.quantity_display['axes']['bottom']['quantity']['key'] == 'dwell_s'
    draft = migrate_plot_quantity(pending)
    assert draft.histogram.column == 'frac_depth'
    assert float(draft.axes.x_min) == .002
    assert float(draft.axes.y_max) == 100.
    # An explicitly witnessed applied capture always wins over the draft witness.
    current = plot_config_to_json(restored)
    current['axes'] = {'time_unit': 'ms', 'quantity_display': current['axes']['quantity_display']}
    again = _config_for_applied(replace(draft, axes=replace(draft.axes, x_min='2', y_max='.1')), current)
    assert again.axes.quantity_display['axes']['bottom']['quantity']['key'] == 'dwell_s'
    assert float(again.axes.x_min) == 2.
    assert float(again.axes.y_max) == .1



def test_interactive_time_display_has_one_ms_policy_but_legacy_configs_migrate(qtbot):
    from nanoscope.core.session import Session
    page = PlotsWindow(Session()); qtbot.addWidget(page)
    assert [page._time_unit.itemText(i) for i in range(page._time_unit.count())] == ['ms']
    assert not page._time_unit.isEnabled()
    for unit, low in (('s', '.002'), ('ms', '2'), ('µs', '2000')):
        original = PlotConfig(histogram=HistogramConfig(column='dwell_s'),
                              axes=AxesConfig(time_unit=unit, x_min=low, manual_x=(.002, .01)))
        migrated = migrate_plot_quantity(original)
        assert migrated.axes.time_unit == 'ms'
        assert float(migrated.axes.x_min) == 2.
        assert migrated.axes.manual_x == (.002, .01)



@pytest.mark.parametrize('family', ['linear', 'power law'])
def test_frozen_scatter_readouts_convert_from_saved_geometry_without_refit(family, monkeypatch):
    import copy
    import pandas as pd
    from nanoscope.gui.plots_panel import ScatterConfig, SCATTER, populations
    from nanoscope.gui.figure_adapter_plots import _scatter_readouts_in_display_units
    x = np.array([.001, .002, .004])
    y = 2000. * x + 5. if family == 'linear' else 500. * x ** 2
    frame = pd.DataFrame({'x': x, 'y': y})
    before = frame.copy(deep=True)
    old = 'old generated equation (x in µs, y in pA)'
    render = {'nodes': [{'role': 'scatter_fit', 'table': 'fit', 'source': 'a',
                        'attributes': {'_presentation_fit_family': family}}],
              'readouts': [['a', {'parts': {'equation': old, 'r2': 'R² = 0.987', 'n': 'n = 9'},
                                  'refusals': [], 'source': 'My literal name (s)', 'colour': '#123456'}]],
              'details': 'Fit: ' + old}
    original = copy.deepcopy(render)
    config = migrate_plot_quantity(PlotConfig(plot_type=SCATTER,
        scatter=ScatterConfig(x='dwell_s', y='depth_pA'), axes=AxesConfig(time_unit='µs')))
    monkeypatch.setattr(populations, 'linear_fit', lambda *_a, **_k: pytest.fail('refit'))
    changed = _scatter_readouts_in_display_units(render, {'fit': frame}, config)
    equation = changed['readouts'][0][1]['parts']['equation']
    assert equation == ('y = 2·x + 5   (x in ms, y in pA)' if family == 'linear'
                        else 'y = 0.0005·x^2   (x in ms, y in pA)')
    assert changed['details'] == 'Fit: ' + equation
    assert changed['readouts'][0][1]['parts']['r2'] == 'R² = 0.987'
    assert changed['readouts'][0][1]['source'] == 'My literal name (s)'
    assert render == original
    pd.testing.assert_frame_equal(frame, before)
