"""Declared pin/history units change once without rewriting science or literal words."""
from copy import deepcopy

import pytest

from nanoscope.core.figure_history_state import HistoryCapsule
from nanoscope.core.figure_quantity_basis import (
    migrate_appearance_basis, migrate_appearance_history, read_quantity_bases)
from nanoscope.core.quantity_display import AxisDisplayBasis, QuantityError, quantity_for


def bases(unit, *, log=False):
    return {'ensemble.bottom': AxisDisplayBasis(quantity_for('time_s', display_unit=unit),
                                                'log10' if log else 'linear')}


def test_seconds_pin_migrates_once_and_other_views_and_words_stay_literal():
    original = {'title': 'Apo at 1 s', 'x_label': 'my time (s)', 'axes': {
        'ensemble.bottom': {'min': -.002, 'max': .003, 'tick_major': .001, 'future': [1, None]},
        'map.left': {'min': .1, 'max': .7}}, 'text': {'title': {'bold': True}},
        'manual_view': [[-.003, .004], [10, 20]], 'unknown': {'raw': True}}
    expected = deepcopy(original)
    result = migrate_appearance_basis(original, bases('ms'), legacy_axes=bases('s'))
    assert result['axes']['ensemble.bottom'] == {'min': -2., 'max': 3., 'tick_major': 1., 'future': [1, None]}
    for key in ('title', 'x_label', 'text', 'manual_view', 'unknown'):
        assert result[key] == expected[key]
    assert result['axes']['map.left'] == original['axes']['map.left']
    assert migrate_appearance_basis(result, bases('ms'), legacy_axes=bases('s')) == result
    assert original == expected


def test_log_range_scales_linear_bounds_but_major_spacing_remains_decades():
    value = {'axes': {'ensemble.bottom': {'min': .001, 'max': 1., 'tick_major': .5, 'log': True}}}
    result = migrate_appearance_basis(value, bases('ms', log=True), legacy_axes=bases('s', log=True))
    assert result['axes']['ensemble.bottom'] == {'min': 1., 'max': 1000., 'tick_major': .5, 'log': True}
    basis = read_quantity_bases(result['quantity_display'])['ensemble.bottom']
    assert basis.coordinate_frame == 'log10' and basis.quantity.display_unit == 'ms'


def test_all_history_values_convert_without_changing_order_or_memos():
    values = [{'axes': {'ensemble.bottom': {'min': x, 'max': x + .01}}, 'title': str(x)}
              for x in (-.02, 0., .02)]
    capsule = HistoryCapsule('nanoscope.appearance-history/1', values[1],
        past=(values[0],), future=(values[2],), dropped=4, extra={'memos': {'raw_s': .002}})
    migrated = migrate_appearance_history(capsule, bases('ms'), legacy_axes=bases('s'))
    assert migrated.base['axes']['ensemble.bottom']['max'] == 10.
    assert migrated.past[0]['axes']['ensemble.bottom']['min'] == -20.
    assert migrated.future[-1]['axes']['ensemble.bottom']['min'] == 20.
    assert migrated.dropped == 4 and migrated.extra == capsule.extra
    assert migrate_appearance_history(migrated, bases('ms'), legacy_axes=bases('s')) == migrated


def test_missing_or_future_basis_refuses_instead_of_guessing_and_mutating():
    unknown = {'axes': {'ensemble.bottom': {'min': .001, 'max': .1}}}
    original = deepcopy(unknown)
    with pytest.raises(QuantityError, match='No declared previous'):
        migrate_appearance_basis(unknown, bases('ms'))
    unknown['quantity_display'] = {'version': 2, 'future': [None, 'keep']}
    future = deepcopy(unknown)
    with pytest.raises(QuantityError, match='Unsupported'):
        migrate_appearance_basis(unknown, bases('ms'), legacy_axes=bases('s'))
    assert unknown == future and original['axes'] == unknown['axes']


def test_unrelated_or_dimensionless_pin_is_never_reinterpreted_as_time():
    old = {'ensemble.bottom': AxisDisplayBasis(quantity_for('phase', declared_unit='-'))}
    value = {'axes': {'ensemble.bottom': {'min': 0., 'max': 1.}}}
    with pytest.raises(QuantityError, match='different stored quantity'):
        migrate_appearance_basis(value, bases('ms'), legacy_axes=old)


def test_native_codec_owns_axis_witness_once_and_round_trips_unknowns():
    from nanoscope.core.figure_codecs import NATIVE_KINDS, decode_legacy, encode_legacy
    from nanoscope.core.project import WorkspaceArtifact
    native_id, (kind, _, _) = next((key, value) for key, value in NATIVE_KINDS.items() if value[1] == 'iv')
    witness = {'version': 1, 'future': {'kept': [None, 2]}, 'axes': {
        'bottom': AxisDisplayBasis(quantity_for('dwell_s')).to_dict()}}
    payload = {'appearance': {'title': 'Literal seconds', 'quantity_display': witness,
                              'axes': {'bottom': {'min': 1., 'max': 5.}}}}
    artifact = WorkspaceArtifact(native_id, kind, payload=payload)
    decoded = decode_legacy(artifact, figure_id='1b79df34-6f0f-4b70-95b0-8575be467d35', name='Witness')
    axis = next(axis for axis in decoded.spec.axes if axis.axis_id == 'bottom')
    assert axis.extra['display_basis']['quantity']['display_unit'] == 'ms'
    assert axis.pin.units == 'ms'
    assert encode_legacy(decoded.spec, decoded.resources).payload == payload
    remainder = decoded.spec.result.payload.get('appearance', {})
    assert not remainder.get('quantity_display', {}).get('axes', {}).get('bottom')


def test_native_codec_carries_future_witness_without_reinterpretation():
    from nanoscope.core.figure_codecs import NATIVE_KINDS, decode_legacy, encode_legacy
    from nanoscope.core.project import WorkspaceArtifact
    native_id, (kind, _, _) = next((key, value) for key, value in NATIVE_KINDS.items() if value[1] == 'iv')
    payload = {'appearance': {'quantity_display': {'version': 9, 'future': ['raw', None]}}}
    artifact = WorkspaceArtifact(native_id, kind, payload=payload)
    decoded = decode_legacy(artifact, figure_id='b461cd36-976b-4786-89d9-dddc27f61693', name='Future')
    assert encode_legacy(decoded.spec, decoded.resources).payload == payload
    assert decoded.spec.result.payload['appearance']['quantity_display']['version'] == 9


def test_legacy_event_map_log_coordinates_become_physical_ms_once():
    value = {'axes': {'bottom': {'min': -3., 'max': -1., 'tick_major': 1.}}}
    old = {'bottom': AxisDisplayBasis(quantity_for('dwell_s', display_unit='s'), 'log10')}
    new = {'bottom': AxisDisplayBasis(quantity_for('dwell_s'), 'log10')}
    result = migrate_appearance_basis(value, new, legacy_axes=old, legacy_view_pins=('bottom',))
    assert result['axes']['bottom'] == {'min': 1., 'max': 100., 'tick_major': 1.}
    assert migrate_appearance_basis(result, new, legacy_axes=old, legacy_view_pins=('bottom',)) == result


def test_native_dormant_contract_uses_applied_summary_not_pending_metric():
    from nanoscope.gui.figure_quantity_native import migrate_native_appearance
    payload = {'applied': {'metric': 'dwell_s'}, 'draft': {'metric': 'depth_pA'},
               'snapshot': {'categorical': False}}
    current = migrate_native_appearance({'axes': {'bottom': {'min': .001, 'max': .01}}},
                                        payload, 'distribution_summaries')
    assert current['axes']['bottom']['min'] == 1.
    assert current['quantity_display']['axes']['bottom']['quantity']['key'] == 'dwell_s'


def test_a_change_of_log_frame_carries_the_spacing_as_the_number_it_is_and_never_raises():
    """2026-11-11: this raised ``QuantityError`` ('its owner must handle tick spacing') inside the
    slot of every redraw that came before the owner -- the Event-count map's Log boxes -- and left
    the page half drawn. The spacing now crosses as the number it is (no unit factor turns decades
    into a unit or back), and its owner, the Log box or Axis properties, decides what it means.

    The unit is the same on both sides of every crossing here: no caller changes the unit and the
    frame together (the legacy basis of a page is always built with the frame of the new one), and
    a test that did would pin a spacing of 20 s coming out as 20 ms as the contract."""
    def bases(frame, unit):
        return {'map.left': AxisDisplayBasis(quantity_for('dwell_s', display_unit=unit), frame)}

    seconds = migrate_appearance_basis({'axes': {'map.left': {'min': 1., 'max': 100., 'tick_major': 20.}}},
                                       bases('linear', 's'), legacy_axes=bases('linear', 's'))
    # Linear seconds -> linear milliseconds: on a linear axis a spacing is a length in the unit,
    # so it takes the unit's factor with the range (the spacing now typed is 20000 ms).
    typed = migrate_appearance_basis(seconds, bases('linear', 'ms'))
    assert typed['axes']['map.left'] == {'min': 1000., 'max': 100000., 'tick_major': 20000.}
    # The same axis on a log scale, in the same unit: the range is a display value and stays, the
    # spacing stays the number it is.
    crossed = migrate_appearance_basis(typed, bases('log10', 'ms'))
    assert crossed['axes']['map.left'] == typed['axes']['map.left']
    assert read_quantity_bases(crossed['quantity_display'])['map.left'].coordinate_frame == 'log10'
    back = migrate_appearance_basis(crossed, bases('linear', 'ms'))
    assert back['axes']['map.left'] == typed['axes']['map.left'], "nothing is lost on the way back"
    # An Undo history that holds the spacing crosses the same way: the states kept on the other
    # frame are carried to the log one with their spacing as it is.
    capsule = HistoryCapsule('nanoscope.appearance-history/1', crossed, past=(typed,), future=(back,))
    moved = migrate_appearance_history(capsule, bases('log10', 'ms'))
    for state in (moved.past[0], moved.base, moved.future[0]):
        assert state['axes']['map.left'] == {'min': 1000., 'max': 100000., 'tick_major': 20000.}
