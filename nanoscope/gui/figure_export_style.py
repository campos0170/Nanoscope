"""Detached style metadata for export sizing; never apply a look to captured artwork."""
from collections.abc import Mapping
from dataclasses import replace
import math

from ..core.figure_style import FigureStyle, builtin_styles, normalize_style_binding, resolve_look


def style_from_binding(binding):
    if not binding:
        return None
    binding = normalize_style_binding(binding)
    style = FigureStyle.from_dict(binding['snapshot'])
    return (replace(style, look=resolve_look(style, binding['overrides']).values)
            if binding.get('overrides') else style)


def export_style_for(owner):
    """Use this figure's pinned definition, never the project's current default."""
    getter = getattr(owner, 'appearance_state', None)
    if callable(getter):
        appearance = getter()
        if isinstance(appearance, Mapping):
            return style_from_binding(appearance.get('style_binding'))
    getter = getattr(owner, 'style', None)
    if callable(getter):
        state = getter()
        if isinstance(state, Mapping) and state.get('style_binding'):
            return style_from_binding(state['style_binding'])
    widget = getattr(owner, '_appearance_widget', None)
    state = widget.state() if widget is not None else None
    return style_from_binding(getattr(state, 'style_binding', None))


def snapshot_export_style(snapshot):
    spec = snapshot.figure_spec()
    if isinstance(spec, Mapping):
        presentation = spec.get('presentation', {})
        raw = presentation.get('style_snapshot', {}) if isinstance(presentation, Mapping) else {}
        if isinstance(raw, Mapping) and raw.get('style_id'):
            return style_from_binding({'version': 1, 'ref': presentation.get('style_ref') or {
                'style_id': raw['style_id'], 'revision': raw['revision']}, 'snapshot': raw,
                                       'overrides': presentation.get('overrides', {})})
    config = snapshot.config()
    if isinstance(config, Mapping) and config.get('kind') == 'figure-composition':
        # Compositions pin their own lettering style beside the layout. Its canvas
        # and column dimensions still describe the exported composition; the
        # original panel artwork remains frozen and is never restyled here.
        return style_from_binding(config.get('style_binding'))
    style = config.get('style', {}) if isinstance(config, Mapping) else {}
    return style_from_binding(style.get('style_binding')) if isinstance(style, Mapping) else None


def column_presets(style=None):
    """Return named physical widths and honest reference provenance for the dialog."""
    if style is None:
        columns = builtin_styles()[0].columns
        name = 'Reference columns'
    else:
        if not isinstance(style, FigureStyle):
            style = FigureStyle.from_dict(style)
        columns, name = style.columns, style.name
    note = str(columns.get('label') or 'Style columns')
    if columns.get('basis') == 'generic-reference' or style is None:
        note += ' · generic reference widths; not verified publication requirements'
    result = []
    for key, label in (('single_mm', 'Single column'), ('double_mm', 'Double column')):
        if key in columns:
            value = float(columns[key])
            result.append((f'{label} · {value:g} mm · {name}', value))
    return tuple(result), note


def canvas_dimensions(style, native_size):
    """Resolve an optional physical canvas default without altering captured artwork."""
    if style is None:
        return None
    if not isinstance(style, FigureStyle):
        style = FigureStyle.from_dict(style)
    width, height = style.look.get('canvas_width_mm'), style.look.get('canvas_height_mm')
    if width is None and height is None:
        return None
    native_width, native_height = map(float, native_size)
    if not all(math.isfinite(value) and value > 0 for value in (native_width, native_height)):
        raise ValueError("Captured canvas dimensions must be finite and positive.")
    if width is None:
        width = float(height) * native_width / native_height
    if height is None:
        height = float(width) * native_height / native_width
    return float(width), float(height)
