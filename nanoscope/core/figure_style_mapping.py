"""Plain-map conversion at native boundaries; no renderer, Qt or settings imports."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from copy import deepcopy

from .figure_spec import FrozenMap, FigureValidationError, thaw_json
from .figure_style import (FigureStyle, validate_look, resolve_look, normalize_style_binding,
                           points_to_logical_pixels, logical_pixels_to_points)

_APPEARANCE = dict(font_family='font_family', title_size='title_size_pt', axis_size='axis_size_pt',
    tick_size='tick_size_pt', legend_size='legend_size_pt', background='background',
    data_color='data_color', fit_color='fit_color', grid='grid', frame='frame', line_width='line_width_pt',
    marker_size='marker_size_pt')
_SETTINGS = dict(font_family='font_family', font_size_pt='body_size_pt',
    title_font_size_pt='title_size_pt', axis_label_font_size_pt='axis_size_pt',
    tick_font_size_pt='tick_size_pt', legend_font_size_pt='legend_size_pt',
    data_color='data_color', fit_color='fit_color', background='background', show_grid='grid', frame='frame',
    line_width='line_width_pt', point_size='marker_size_pt')
# These are explicitly figure-owned. Unknown fields follow the same lossless rule.
STYLE_SETTINGS_OWNERS = FrozenMap({
    **{key: 'shared-look' for key in _SETTINGS},
    'title': 'words', 'x_label': 'words', 'y_label': 'words',
    'show_legend': 'scientific-visibility', 'export_w': 'export', 'export_h': 'export',
    'label_overrides': 'source-words', 'legend_pos': 'annotation-position',
    'legend_counts': 'scientific-annotation', 'legend_include': 'scientific-visibility',
    'show_ci_band': 'scientific-visibility', 'show_provenance': 'scientific-visibility',
    'show_residuals': 'scientific-visibility', 'heatmap_colormap': 'figure-colormap',
    'text_styles': 'element-overrides-and-words', 'fit_label': 'fit-annotation',
    'series_styles': 'figure-owned-series-overrides',
    'style_binding': 'shared-style-binding'})


@dataclass(frozen=True)
class LegacyLookConversion:
    look: FrozenMap
    remainder: FrozenMap
    report: FrozenMap


def _shape(mapping, vocabulary):
    if not isinstance(mapping, Mapping):
        raise FigureValidationError('A legacy style must be an object')
    if vocabulary == 'style_settings':
        return dict(mapping), _SETTINGS, True, ()
    if vocabulary in ('plot_appearance', 'matplotlib_appearance'):
        return dict(mapping), _APPEARANCE, vocabulary == 'plot_appearance', ()
    if vocabulary == 'study_figure_style':
        appearance = mapping.get('appearance', {})
        if not isinstance(appearance, Mapping):
            raise FigureValidationError('Study appearance must be an object')
        # Study colour groups and their alpha/identity are owned by the figure.
        fields = {k: v for k, v in _APPEARANCE.items() if k not in ('data_color', 'fit_color')}
        return dict(appearance), fields, False, ('appearance',)
    raise FigureValidationError(f'Unknown legacy style vocabulary: {vocabulary}')


def import_legacy_look(mapping, vocabulary, *, logical_dpi=96.):
    """Claim only global look; wording, per-element overrides and unknown values survive."""
    native, fields, pixels, prefix = _shape(mapping, vocabulary)
    remainder = thaw_json(FrozenMap(mapping))
    remaining = remainder['appearance'] if prefix and 'appearance' in remainder else remainder
    look, claimed, carried = {}, [], []
    for old, new in fields.items():
        if old not in native or native[old] is None:
            continue
        value = native[old]
        if vocabulary == 'style_settings' and old in ('title_font_size_pt', 'axis_label_font_size_pt',
                                                      'tick_font_size_pt', 'legend_font_size_pt') and value == 0:
            body = native.get('font_size_pt', 10)
            value = max(6, body - 3) if old == 'legend_font_size_pt' else body
        try:
            if new in ('line_width_pt', 'marker_size_pt') and pixels:
                value = logical_pixels_to_points(value, logical_dpi)
            validate_look({new: value})
        except FigureValidationError:
            # Never substitute a journal default for an invalid legacy override.
            carried.append('.'.join((*prefix, old)))
            continue
        look[new] = value
        remaining.pop(old, None)
        claimed.append('.'.join((*prefix, old)))
    if prefix and not remaining and 'appearance' not in mapping:
        remainder.pop('appearance', None)
    return LegacyLookConversion(FrozenMap(look), FrozenMap(remainder),
        FrozenMap({'vocabulary': vocabulary, 'claimed': claimed, 'unreadable': carried,
                   'logical_dpi': logical_dpi, 'metric_units': 'points'}))


def export_legacy_look(look, vocabulary, original, *, logical_dpi=96., _authoritative=False):
    """Project known look properties and retain every unclaimed native value.

    A no-edit round trip retains legacy zero/inherit sentinels and omitted values.
    Binding projection instead clears all old global look before writing its authority.
    """
    look = validate_look(look)
    native, fields, pixels, prefix = _shape(original, vocabulary)
    out = thaw_json(FrozenMap(original))
    current = import_legacy_look(original, vocabulary, logical_dpi=logical_dpi).look
    projected = deepcopy(native)
    if _authoritative:
        for key in fields:
            projected.pop(key, None)
    for old, new in fields.items():
        if new not in look:
            continue
        if not _authoritative and new in current and look[new] == current[new] and old in native:
            continue
        value = thaw_json(look[new])
        if new in ('line_width_pt', 'marker_size_pt') and pixels:
            value = points_to_logical_pixels(value, logical_dpi)
        projected[old] = value
    if prefix:
        if projected or 'appearance' in out:
            out['appearance'] = projected
    else:
        out = projected
    return out


def project_legacy_style(binding, vocabulary, original, *, logical_dpi=96.):
    binding = normalize_style_binding(binding)
    style = FigureStyle.from_dict(binding['snapshot'])
    values = resolve_look(style, binding['overrides']).values
    out = export_legacy_look(values, vocabulary, original, logical_dpi=logical_dpi,
                             _authoritative=True)
    out['style_binding'] = deepcopy(binding)
    return out
