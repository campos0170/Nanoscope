"""Reusable palettes are defaults; identity and deliberate series ink stay authoritative."""
import pytest

from nanoscope.core.figure_spec import FigureValidationError
from nanoscope.core.figure_style import FigureStyle, style_binding, resolve_series_color
from nanoscope.core.figure_style_mapping import project_legacy_style, import_legacy_look


def binding(**overrides):
    return style_binding(FigureStyle('palette.lab', 1, 'Lab', look={
        'palette': ['#112233', '#445566'], 'frame': False,
        'data_color': '#778899', 'fit_color': '#aa4422'}), overrides)


def test_palette_defaults_only_unclassified_data_and_keeps_identity_and_explicit_color():
    value = binding()
    assert resolve_series_color(value, index=0) == '#112233'
    assert resolve_series_color(value, index=3) == '#445566'
    assert resolve_series_color(value, role='fit') == '#aa4422'
    assert resolve_series_color(value, identity_color='#00ff00') == '#00ff00'
    assert resolve_series_color(value, identity_color='#00ff00', explicit_color='#cc00cc') == '#cc00cc'
    assert resolve_series_color(binding(data_color='#999999')) == '#999999'
    assert resolve_series_color(binding(data_color='#999999'), identity_color='#00ff00') == '#00ff00'
    assert resolve_series_color(None, native_color='original') == 'original'
    assert value['snapshot']['look']['palette'] == ['#112233', '#445566']


@pytest.mark.parametrize('bad', [[], ['red'], ['#123'], ['#12zz00'], 'viridis', [None], ['#000000'] * 257])
def test_invalid_palette_is_refused_before_registry_or_artist_mutation(bad):
    with pytest.raises(FigureValidationError):
        FigureStyle('bad', 1, 'Bad', look={'palette': bad})


@pytest.mark.parametrize('vocabulary', ['plot_appearance', 'matplotlib_appearance', 'style_settings'])
def test_frame_projects_as_explicit_false_without_touching_words_or_pins(vocabulary):
    source = {'title': 'Literal', 'axes': {'left': {'min': 2}}, 'unknown': [None]}
    rendered = project_legacy_style(binding(), vocabulary, source)
    assert rendered['frame'] is False
    assert rendered['title'] == source['title'] and rendered['axes'] == source['axes']
    assert rendered['unknown'] == [None]
    assert rendered['style_binding']['snapshot']['look']['palette'] == ['#112233', '#445566']
    assert import_legacy_look({'frame': False}, vocabulary).look['frame'] is False


def test_native_frame_absence_is_not_replaced_by_a_new_default():
    style = FigureStyle('legacy', 1, 'Legacy', look={'grid': False}, provenance={'inherit_native_defaults': True})
    rendered = project_legacy_style(style_binding(style), 'plot_appearance', {'title': 'Keep'})
    assert 'frame' not in rendered
    with pytest.raises(FigureValidationError):
        FigureStyle('bad', 1, 'Bad', look={'frame': 'box'})
