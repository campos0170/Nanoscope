"""Series appearance has stable identities and one canonical persisted owner."""
from dataclasses import replace
import pytest
from nanoscope.core.figure_series import SeriesDescriptor, stable_series_id, validate_series_styles
from nanoscope.core.figure_spec import FigureValidationError
from nanoscope.core.figure_codecs import decode_legacy, encode_legacy, NATIVE_KINDS
from test_p2_figure_codecs import native, ID


def test_identity_is_independent_of_display_label():
    key = stable_series_id('map', 'data', 'recording#channel#detector')
    first = SeriesDescriptor(key, 'Pore A', identity_kind='source', identity_key='recording#channel#detector')
    assert replace(first, label='Edited words').series_id == key
    assert stable_series_id('other', 'data', first.identity_key) != key
    assert SeriesDescriptor.from_dict(first.to_dict()) == first


@pytest.mark.parametrize('value', [{'color':'red'}, {'line_width_pt':0}, {'marker_size_pt':float('inf')}, {'visible':1}])
def test_invalid_series_values_are_rejected_before_install(value):
    with pytest.raises(FigureValidationError):validate_series_styles({'stable':value})


@pytest.mark.parametrize('identity', tuple(NATIVE_KINDS))
def test_native_series_overrides_have_one_canonical_owner_and_exact_inverse(identity):
    original = native(identity)
    if identity == 'builtin.plots':
        appearance = original.payload['config']['fields']['style']['fields']
    elif identity in ('builtin.event-plots','builtin.psd','builtin.voltage-dependence'):
        appearance = original.payload['config']['appearance']
    else:
        appearance = original.payload['appearance']
    appearance['series_styles'] = {'stable-source-series': {'color':'#123456', 'line_width_pt':.625, 'visible':False}}
    decoded = decode_legacy(original, figure_id=ID, name='Series')
    target = next(s for s in decoded.spec.series if s.primitive == 'style-target')
    assert target.overrides['line_width_pt'] == .625
    restored = encode_legacy(decoded.spec, decoded.resources)
    assert restored.payload == original.payload
    changed = replace(target, overrides={**dict(target.overrides),'color':'#abcdef'})
    spec = replace(decoded.spec, series=tuple(changed if s == target else s for s in decoded.spec.series))
    saved = encode_legacy(spec, decoded.resources)
    assert '#abcdef' in str(saved.payload) and saved.tables.keys() == original.tables.keys()



def _appearance(artifact, identity):
    if identity == 'builtin.plots':
        return artifact.payload['config']['fields']['style']['fields']
    if identity in ('builtin.event-plots', 'builtin.psd', 'builtin.voltage-dependence'):
        return artifact.payload['config']['appearance']
    return artifact.payload['appearance']


@pytest.mark.parametrize('identity', tuple(NATIVE_KINDS))
@pytest.mark.parametrize('empty_map', [False, True])
def test_native_series_and_frame_reset_do_not_resurrect_previous_overrides(identity, empty_map):
    from copy import deepcopy
    from nanoscope.core.figure_spec import FigureRecord
    from nanoscope.core.figure_codecs import merge_native_capture
    original = native(identity)
    appearance = _appearance(original, identity)
    appearance['series_styles'] = {'stable-a': {'color': '#123456'}, 'stable-b': {'visible': False}}
    appearance['frame'] = False
    appearance['future_appearance'] = {'opaque': [1, 2]}
    decoded = decode_legacy(original, figure_id=ID, name='Reset series')
    previous = FigureRecord(decoded.spec, decoded.resources)
    captured = replace(original, payload=deepcopy(original.payload))
    now = _appearance(captured, identity)
    now.pop('frame')
    now.pop('future_appearance')  # unrelated unknown forward content still carries
    now.pop('series_styles')
    if empty_map:
        now['series_styles'] = {}
    merged = merge_native_capture(captured, previous)
    restored = encode_legacy(merged.spec, merged.resources)
    values = _appearance(restored, identity)
    assert not values.get('series_styles')
    assert 'frame' not in values
    assert values['future_appearance'] == {'opaque': [1, 2]}
    assert not any(series.primitive == 'style-target' for series in merged.spec.series)
    assert _appearance(original, identity)['frame'] is False


def test_removing_one_series_keeps_only_the_other_owned_override():
    from copy import deepcopy
    from nanoscope.core.figure_spec import FigureRecord
    from nanoscope.core.figure_codecs import merge_native_capture
    original = native('builtin.plots')
    values = _appearance(original, 'builtin.plots')
    values['series_styles'] = {'first': {'color': '#123456'}, 'second': {'visible': False}}
    decoded = decode_legacy(original, figure_id=ID, name='One series reset')
    captured = replace(original, payload=deepcopy(original.payload))
    _appearance(captured, 'builtin.plots')['series_styles'].pop('first')
    merged = merge_native_capture(captured, FigureRecord(decoded.spec, decoded.resources))
    restored = encode_legacy(merged.spec, merged.resources)
    assert _appearance(restored, 'builtin.plots')['series_styles'] == {'second': {'visible': False}}
