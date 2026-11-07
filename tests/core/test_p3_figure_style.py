"""Pure shared-look semantics; no renderer imports."""
from dataclasses import replace
import pytest

from nanoscope.core.figure_spec import FigureValidationError
from nanoscope.core.figure_style import (FigureStyle, StyleRef, resolve_look, builtin_styles,
    style_binding, normalize_style_binding, points_to_logical_pixels, logical_pixels_to_points)
from nanoscope.core.figure_style_mapping import (import_legacy_look, export_legacy_look,
                                                project_legacy_style)


def test_fractional_metrics_and_each_origin_are_kept():
    style = FigureStyle('style.a', 1, 'A', {'line_width_pt': .75, 'body_size_pt': 7.,
        'text_roles': {'title': {'bold': False, 'color': 'blue'}}})
    resolved = resolve_look(style, {'body_size_pt': 7., 'text_roles': {'title': {'bold': True}}},
                            {'text_roles': {'title': {'color': 'red'}}})
    assert resolved.values['line_width_pt'] == .75
    assert resolved.origins['line_width_pt'] == 'style'
    assert resolved.origins['body_size_pt'] == 'figure'  # equal is still explicit
    assert resolved.origins['font_family'] == 'default'
    assert resolved.origins['text_roles']['title'] == {'bold': 'figure', 'color': 'element'}
    assert points_to_logical_pixels(.75) == 1.
    assert logical_pixels_to_points(1.) == .75
    with pytest.raises(AttributeError): style.look._items = ()


@pytest.mark.parametrize('bad', [True, 0, -1, float('nan'), float('inf'), '7'])
def test_invalid_physical_metrics_fail_before_rendering(bad):
    with pytest.raises(FigureValidationError):
        FigureStyle('a', 1, 'A', {'line_width_pt': bad})


def test_unknown_style_nodes_and_bindings_survive():
    raw = builtin_styles()[0].to_dict()
    raw['future'] = {'a': [None, 2]}; raw['extra']['_unknown_fields'] = 'user value'
    style = FigureStyle.from_dict(raw)
    assert style.to_dict() == raw
    binding = style_binding(style, {'grid': False})
    binding['future'] = ['kept']
    assert normalize_style_binding(binding)['future'] == ['kept']
    binding['ref']['revision'] = 2
    with pytest.raises(FigureValidationError, match='disagrees'): normalize_style_binding(binding)


@pytest.mark.parametrize('vocabulary,native', [
    ('plot_appearance', {'title': 'Typed $x$', 'line_width': .75, 'marker_size': 3.,
        'title_size': 7, 'axes': {'left': {'min': 2}}, 'future': 4,
        'text': {'legend.entry.a': {'text': 'source', 'bold': True}}}),
    ('matplotlib_appearance', {'title': 'MPL', 'line_width': .75, 'marker_size': 3.}),
    ('style_settings', {'font_size_pt': 10, 'title_font_size_pt': 0,
        'legend_font_size_pt': 0, 'line_width': 1, 'point_size': 7,
        'legend_counts': False, 'heatmap_colormap': 'magma', 'show_residuals': True,
        'export_w': 123, 'fit_label': {'template': '{tau}', 'fields': ['n']}}),
    ('study_figure_style', {'appearance': {'title': 'Study', 'line_width': .75},
        'panels': {'0': {'title': 'Preserve'}}, 'colors': {'#123456': '#abcdef'},
        'event_titles': {'identity': 'Event'}, 'future': {'shape': 1}}),
])
def test_exact_no_edit_legacy_roundtrip(vocabulary, native):
    conversion = import_legacy_look(native, vocabulary)
    assert export_legacy_look(conversion.look, vocabulary, native) == native
    assert not conversion.report['unreadable']


def test_pg_and_matplotlib_project_same_physical_width_with_distinct_native_values():
    binding = style_binding(builtin_styles()[0])
    native = {'line_width': 99, 'title': 'Keep', 'axes': {'left': {'min': 4}},
              'text': {'title': {'bold': True}}, 'fit_label': {'fields': ['n']}}
    pg = project_legacy_style(binding, 'plot_appearance', native)
    mpl = project_legacy_style(binding, 'matplotlib_appearance', native)
    assert pg['line_width'] == 1. and mpl['line_width'] == .75
    for key in ('title', 'axes', 'text', 'fit_label'):
        assert pg[key] == mpl[key] == native[key]
    assert native['line_width'] == 99


def test_invalid_legacy_look_is_reported_and_carried_not_defaulted():
    native = {'line_width': 'bad', 'font_family': 'A'}
    conversion = import_legacy_look(native, 'plot_appearance')
    assert conversion.report['unreadable'] == ('line_width',)
    assert conversion.remainder['line_width'] == 'bad'
    assert export_legacy_look(conversion.look, 'plot_appearance', native) == native


def test_bound_text_roles_have_shared_origins_without_copying_typed_words():
    from nanoscope.core.figure_style import resolve_bound_text
    style = FigureStyle('custom', 1, 'Custom', {'text_roles': {
        'title': {'bold': True}, 'axis': {'color': 'blue'},
        'axis.bottom.label': {'italic': True}, 'fit_label': {'size_pt': 8.5}}})
    binding = style_binding(style, {'text_roles': {'axis': {'color': 'green'}}})
    value, origin = resolve_bound_text(binding, 'axis.bottom.label',
                                     {'color': 'red', 'text': 'Never copied'})
    assert value['italic'] is True and origin['italic'] == 'style'
    assert value['color'] == 'red' and origin['color'] == 'element'
    assert 'text' not in value
    assert resolve_bound_text(binding, 'fit.label')[0]['size_pt'] == 8.5
    assert resolve_bound_text(binding, 'annotation.fit')[0]['size_pt'] == 8.5


def test_builtin_reference_columns_make_no_journal_compliance_claim_and_slides_keep_16pt():
    styles = builtin_styles()
    for style in styles:
        assert style.columns['basis'] == 'generic-reference'
        assert 'not verified journal requirements' in style.provenance['column_note']
    slides = next(s for s in styles if s.style_id == 'builtin.slides.arial18')
    assert slides.look['tick_size_pt'] == slides.look['legend_size_pt'] == 16.


def test_sparse_imported_look_keeps_native_fallbacks_absent_across_projection_and_text():
    from nanoscope.core.figure_style import resolve_bound_text
    legacy = FigureStyle('legacy.a', 1, 'Old partial', {'title_size_pt': 19},
                         provenance={'legacy_identity': 'legacy.a'})
    bound = style_binding(legacy)
    assert dict(resolve_look(legacy).values) == {'title_size_pt': 19}
    projected = project_legacy_style(bound, 'plot_appearance', {'title': 'Typed'})
    assert projected['title_size'] == 19 and projected['title'] == 'Typed'
    assert not {'axis_size', 'tick_size', 'line_width', 'marker_size', 'data_color'} & projected.keys()
    assert resolve_bound_text(bound, 'axis.bottom.label') == ({}, {})
    assert resolve_bound_text(bound, 'title') == ({'size_pt': 19}, {'size_pt': 'style'})
    # Explicit native values promoted by the bridge remain canonical overrides.
    promoted = style_binding(legacy, {'line_width_pt': 2.25, 'axis_size_pt': 12})
    projected = project_legacy_style(promoted, 'plot_appearance', {})
    assert projected['line_width'] == 3 and projected['axis_size'] == 12
    assert resolve_look(legacy, defaults={'body_size_pt': 10}).values['body_size_pt'] == 10
    # Ordinary shared definitions still inherit the declared renderer-independent base.
    assert resolve_look(FigureStyle('new', 1, 'New', {'title_size_pt': 19})).values['axis_size_pt'] == 7


def test_saved_sparse_native_look_keeps_its_absence_without_faking_a_legacy_identity():
    from nanoscope.core.figure_style import inherits_native_defaults, resolve_bound_text
    copied = FigureStyle('project.saved', 1, 'Saved from native', {'title_size_pt': 12},
                         provenance={'project_style': True, 'inherit_native_defaults': True})
    assert inherits_native_defaults(copied)
    assert inherits_native_defaults(copied.to_dict())
    assert dict(resolve_look(copied).values) == {'title_size_pt': 12}
    assert resolve_bound_text(style_binding(copied), 'axis.left.label') == ({}, {})
    edited = replace(copied, revision=2, look={'title_size_pt': 14})
    assert dict(resolve_look(edited).values) == {'title_size_pt': 14}
