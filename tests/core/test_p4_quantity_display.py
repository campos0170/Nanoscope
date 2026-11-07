"""Presentation conversions never change scientific storage or guess acquisition units."""
from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from nanoscope.core.clock import ClockOffset
from nanoscope.core.figure_spec import FigureValidationError
from nanoscope.core.quantity_display import (
    AxisDisplayBasis, QuantityError, QuantityPresentation, clock_quantity,
    column_entry, convert_pin, density_quantity, display_to_view, format_quantity,
    from_display, parse_display_number, quantity_for, sample_time_quantity,
    to_display, view_to_display,
)


@pytest.mark.parametrize('unit,value,shown', [
    ('s', .00125, 1.25), ('ms', 1.25, 1.25), ('us', 1250., 1.25),
    ('μs', 1250., 1.25), ('µs', 1250., 1.25), ('min', .25, 15000.),
    ('h', .5, 1800000.),
])
def test_declared_time_units_have_an_exact_inverse(unit, value, shown):
    descriptor = quantity_for('declared_time', declared_unit=unit)
    assert descriptor.display_unit == 'ms'
    assert to_display(value, descriptor) == pytest.approx(shown)
    assert from_display(shown, descriptor) == pytest.approx(value)


def test_actual_event_and_measurement_schemas_keep_clock_duration_charge_and_sample_roles():
    from nanoscope.analysis.events import SCHEMA
    from nanoscope.analysis.measure import MEASUREMENT_SCHEMA

    for schema in (SCHEMA, MEASUREMENT_SCHEMA):
        start = quantity_for('t_start_s', schema=schema)
        assert (start.quantity, start.clock_basis, start.factor) == ('clock', 'recording', 1000.)
        assert quantity_for('dwell_s', schema=schema).quantity == 'duration'
        assert quantity_for('start_idx', schema=schema).quantity == 'sample'
    assert quantity_for('event_id', schema=SCHEMA).quantity == 'count'
    assert quantity_for('accepted', schema=SCHEMA).quantity == 'categorical'
    assert quantity_for('ecd_pA_s', schema=SCHEMA).factor == 1
    assert quantity_for('area_ecd', schema=MEASUREMENT_SCHEMA).factor == 1
    assert quantity_for('t0_s', schema=MEASUREMENT_SCHEMA).clock_basis == 'instrument'


def test_mapping_schema_and_explicit_source_declaration_override_only_declared_meaning():
    schema = {'t_start_s': {'unit': 's', 'description': 'recording instant'},
              'custom': {'unit': 's', 'quantity': 'clock', 'clock_basis': 'relative',
                         'label': 'Aligned time'}}
    assert quantity_for('t_start_s', schema=schema).clock_basis == 'recording'
    custom = quantity_for('custom', schema=schema)
    assert (custom.label, custom.clock_basis) == ('Aligned time', 'relative')
    # A historical _pA spelling is not evidence for a source which declares none.
    unknown = quantity_for('depth_pA', declared_unit='')
    assert (unknown.known, unknown.quantity, unknown.factor) == (False, 'unknown', 1)
    volts = quantity_for('depth_pA', declared_unit='mV')
    assert (volts.quantity, volts.stored_unit, volts.display_unit, volts.factor) == ('voltage', 'mV', 'mV', 1)


def test_unknown_suffix_never_becomes_seconds_and_raw_key_is_visible():
    descriptor = quantity_for('plugin_surprise_s')
    assert not descriptor.known
    assert descriptor.factor == 1
    key, label, tip = column_entry('plugin_surprise_s', descriptor)
    assert key == 'plugin_surprise_s'
    assert 'units unknown' in label and key in tip and 'undeclared' in tip
    assert to_display(2.5, descriptor) == 2.5


@pytest.mark.parametrize('key', ['frac_depth', 'capture_rate_hz', 'ecd_pA_s', 'start_idx', 'n_events'])
def test_rates_charge_dimensionless_counts_and_samples_are_not_time(key):
    descriptor = quantity_for(key)
    original = np.array([1., 2., 3.])
    result = to_display(original, descriptor)
    np.testing.assert_array_equal(result, original)
    assert not np.shares_memory(result, original)


def test_conversion_detaches_inputs_and_keeps_unavailable_measurements():
    descriptor = quantity_for('dwell_s')
    original = np.array([.001, np.nan, np.inf, -np.inf])
    before = original.copy()
    shown = to_display(original, descriptor)
    assert shown[0] == 1 and math.isnan(shown[1])
    assert shown[2] == math.inf and shown[3] == -math.inf
    shown[0] = 9
    np.testing.assert_array_equal(original, before)
    nested = [[.001, None], (.002,)]
    assert to_display(nested, descriptor) == [[1., None], (2.,)]
    assert nested == [[.001, None], (.002,)]
    with pytest.raises(QuantityError):
        to_display(True, descriptor)
    with pytest.raises(QuantityError):
        to_display(np.array([True]), descriptor)


def test_descriptor_and_axis_basis_are_frozen_lossless_json_values():
    raw = quantity_for('dwell_s').to_dict()
    raw['future'] = {'values': [1, 2]}
    descriptor = QuantityPresentation.from_dict(raw)
    raw['future']['values'].append(3)
    assert descriptor.to_dict()['future'] == {'values': [1, 2]}
    with pytest.raises(FrozenInstanceError):
        descriptor.factor = 2
    basis = AxisDisplayBasis(descriptor, 'log10', extra={'native_frame': 'log10(seconds)'})
    assert AxisDisplayBasis.from_dict(basis.to_dict()) == basis
    with pytest.raises(FigureValidationError):
        QuantityPresentation.from_dict({**descriptor.to_dict(), 'factor': float('nan')})


@pytest.mark.parametrize('rate', [None, 0, -1, float('nan'), float('inf'), True])
def test_samples_need_a_known_positive_rate(rate):
    with pytest.raises(QuantityError):
        sample_time_quantity('start_idx', sample_rate_hz=rate)


def test_sample_clock_and_duration_keep_sample_storage_and_round_trip():
    clock = sample_time_quantity('start_idx', sample_rate_hz=1000,
                                 origin_sample=100, origin_s=10)
    assert clock.stored_unit == 'sample'
    assert to_display(101, clock) == pytest.approx(10001)
    assert from_display(10001, clock) == pytest.approx(101)
    duration = sample_time_quantity('n_samples', sample_rate_hz=20000, duration=True)
    assert to_display(40, duration) == 2
    assert duration.clock_basis == ''
    with pytest.raises(QuantityError):
        sample_time_quantity('n_samples', sample_rate_hz=1000, origin_s=10, duration=True)


def test_instrument_clock_includes_rebased_origin_exactly_once():
    offset = ClockOffset(.25, rebased_from_s=1158.9909)
    descriptor = clock_quantity('t_start_s', offset_s=offset)
    instrument_ms = (1158.9909 + .25 + .002) * 1000
    assert to_display(.002, descriptor) == pytest.approx(instrument_ms)
    assert from_display(instrument_ms, descriptor) == pytest.approx(.002)
    assert descriptor.extra['rebased_origin_s'] == 1158.9909
    inverse = clock_quantity('instrument_time', offset_s=offset, stored_basis='instrument',
                             display_basis='recording')
    assert to_display(instrument_ms / 1000, inverse) == pytest.approx(2)
    with pytest.raises(QuantityError):
        clock_quantity('t_start_s', offset_s=None)
    unchanged = clock_quantity('t_start_s', offset_s=None, display_basis='recording')
    assert to_display(.002, unchanged) == 2


def test_friendly_columns_keep_stable_item_keys_and_known_units():
    descriptor = quantity_for('dwell_s')
    key, label, tip = column_entry('dwell_s', descriptor)
    assert (key, label) == ('dwell_s', 'Dwell time (ms)')
    assert 'dwell_s' in tip and 'stored unit: s' in tip and 'displayed unit: ms' in tip
    with pytest.raises(QuantityError):
        column_entry('rise_s', descriptor)
    assert format_quantity(.00125, descriptor) == '1.25 ms'
    assert format_quantity(float('nan'), descriptor) == '—'
    assert format_quantity(1e308, descriptor) == '—'
    assert format_quantity('reviewed', quantity_for('flag')) == 'reviewed'


def test_edited_display_values_inverse_without_evaluating_expressions_or_units():
    descriptor = quantity_for('dwell_s')
    assert parse_display_number(' 1.25 ', descriptor) == .00125
    assert parse_display_number('−1', descriptor) == -.001
    assert parse_display_number('', descriptor, allow_blank=True) is None
    for raw in ('', '1 ms', '1/2', 'nan', 'inf', '__import__("os")'):
        with pytest.raises(QuantityError):
            parse_display_number(raw, descriptor)


def test_density_jacobian_preserves_probability_and_never_changes_raw_data():
    axis = quantity_for('dwell_s')
    edges_s = np.array([0., .001, .003])
    per_second = np.array([500., 250.])
    descriptor = density_quantity([axis])
    assert (descriptor.stored_unit, descriptor.display_unit) == ('1/s', '1/ms')
    edges_ms = to_display(edges_s, axis)
    per_ms = to_display(per_second, descriptor)
    assert np.sum(per_second * np.diff(edges_s)) == pytest.approx(1)
    assert np.sum(per_ms * np.diff(edges_ms)) == pytest.approx(1)
    np.testing.assert_array_equal(per_second, [500., 250.])
    np.testing.assert_array_equal(from_display(per_ms, descriptor), per_second)
    two_times = density_quantity([axis, quantity_for('rise_s')])
    assert two_times.factor == pytest.approx(1e-6)
    assert two_times.display_unit == '1/(ms·ms)'
    for normalization in ('count', 'probability'):
        assert density_quantity([axis], normalization=normalization).factor == 1


def test_per_decade_density_is_distinct_from_logarithmic_axis_coordinates():
    axis = quantity_for('dwell_s')
    assert density_quantity([axis], measures='log10').factor == 1
    assert density_quantity([axis], measures='log10').display_unit == '1/decade'
    assert density_quantity([axis], measures='linear').factor == .001
    mixed = density_quantity([axis, quantity_for('rise_s')], measures=['log10', 'linear'])
    assert mixed.factor == .001 and mixed.display_unit == '1/(decade·ms)'
    with pytest.raises(QuantityError):
        density_quantity([quantity_for('unknown_s')])
    with pytest.raises(QuantityError):
        density_quantity([quantity_for('event_id')])
    with pytest.raises(QuantityError):
        density_quantity([clock_quantity('t_start_s', offset_s=10)], measures='log10')


def test_log_axis_round_trip_has_multiplicative_units_without_double_log():
    basis = AxisDisplayBasis(quantity_for('dwell_s'), 'log10')
    np.testing.assert_allclose(view_to_display(np.array([-3., -2.]), basis), [1., 10.])
    np.testing.assert_allclose(display_to_view(np.array([1., 10.]), basis), [-3., -2.])
    assert math.isnan(display_to_view(float('nan'), basis))
    for invalid in (0., -1., -math.inf):
        with pytest.raises(QuantityError):
            display_to_view(invalid, basis)
    with pytest.raises(QuantityError):
        AxisDisplayBasis(clock_quantity('t_start_s', offset_s=10), 'log10')


def test_display_pins_convert_endpoints_and_tick_intervals_with_separate_clock_origin():
    old = quantity_for('t_start_s', display_unit='s')
    new = clock_quantity('t_start_s', offset_s=10, display_unit='ms')
    raw = {'min': '1', 'max': 2., 'tick_major': .25, 'future': {'unchanged': [3]}}
    shown = convert_pin(raw, old, new)
    assert shown == {'min': '11000', 'max': 12000., 'tick_major': 250., 'future': {'unchanged': [3]}}
    assert convert_pin(shown, new, old) == raw
    shown['future']['unchanged'].append(4)
    assert raw['future'] == {'unchanged': [3]}


def test_mixed_distributions_pin_text_and_data_ranges_have_one_declared_basis_each():
    old = quantity_for('dwell_s', display_unit='s')
    new = quantity_for('dwell_s')
    raw = {'x_min': '.001', 'x_max': '.003', 'manual_x': [.001, .003]}
    basis = {'x_min': 'display', 'x_max': 'display', 'manual_x': 'data'}
    shown = convert_pin(raw, old, new, value_unit_basis=basis)
    assert shown == {'x_min': '1', 'x_max': '3', 'manual_x': [.001, .003]}
    assert raw['x_min'] == '.001'
    shown['manual_x'][0] = 99
    assert raw['manual_x'] == [.001, .003]
    with pytest.raises(QuantityError):
        convert_pin(raw, old, new, value_unit_basis={'x_min': 'display'})
    with pytest.raises(QuantityError):
        convert_pin({'min': 'unfinished e'}, old, new)
    assert convert_pin({'min': '', 'max': None}, old, new) == {'min': '', 'max': None}


def test_pin_conversion_refuses_quantity_sample_rate_and_stored_clock_changes():
    old = quantity_for('dwell_s', display_unit='s')
    with pytest.raises(QuantityError):
        convert_pin({'min': 1}, old, quantity_for('rise_s'))
    a = sample_time_quantity('start_idx', sample_rate_hz=1000)
    b = sample_time_quantity('start_idx', sample_rate_hz=2000)
    with pytest.raises(QuantityError):
        convert_pin({'min': 1}, a, b)
    recording = clock_quantity('t_start_s', offset_s=10, stored_basis='recording')
    instrument = clock_quantity('t_start_s', offset_s=10, stored_basis='instrument')
    with pytest.raises(QuantityError):
        convert_pin({'min': 1}, recording, instrument)


def test_density_of_compound_charge_uses_unambiguous_inverse_unit():
    charge = quantity_for('ecd_pA_s')
    descriptor = density_quantity([charge])
    assert descriptor.display_unit == '1/(pA·s)'
    assert descriptor.stored_unit == '1/(pA·s)'
    assert descriptor.factor == 1
    assert charge.key == 'ecd_pA_s' and charge.stored_unit == 'pA*s'
    assert density_quantity([quantity_for('dwell_s')]).display_unit == '1/ms'
