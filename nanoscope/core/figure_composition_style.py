"""Panel lettering shares the physical style model; source panels stay frozen."""
from __future__ import annotations

from .figure_spec import FrozenMap, thaw_json
from .figure_style import (logical_pixels_to_points, points_to_logical_pixels,
                           normalize_style_binding, resolve_bound_text, validate_look)

DEFAULT_LABEL_STYLE = {"family": "Arial", "size_px": 18, "bold": True, "color": "#111111"}
PANEL_PROPERTIES = frozenset({'family', 'size_pt', 'bold', 'italic', 'color'})


def panel_overrides_from_legacy(style):
    """Only explicitly changed old lettering becomes a manual override on Apply."""
    result = {}
    for key, value in dict(style).items():
        if key == 'size_px':
            if value != DEFAULT_LABEL_STYLE[key]:
                result['size_pt'] = logical_pixels_to_points(value)
        elif key in PANEL_PROPERTIES and value != DEFAULT_LABEL_STYLE.get(key, False):
            result[key] = value
    return result


def panel_style(binding, overrides=None, *, legacy=None):
    """Project inherited points to explicit 96-dpi composition coordinates.

    Legacy compositions have no binding and keep their pixel values exactly. For
    bound compositions, label_style is a rendering projection, never another
    writable style authority. Words and panel positions are outside this function.
    """
    baseline = dict(DEFAULT_LABEL_STYLE if legacy is None else legacy)
    if not binding:
        return baseline
    binding = normalize_style_binding(binding)
    explicit = thaw_json(FrozenMap(overrides or {}))
    if set(explicit) - PANEL_PROPERTIES:
        raise ValueError('Unsupported manual panel-letter property')
    validate_look({'text_roles': {'panel': explicit}})
    values, _origins = resolve_bound_text(binding, 'annotation.panel', explicit)
    for key in PANEL_PROPERTIES - {'size_pt'}:
        if key in values:
            baseline[key] = values[key]
    if 'size_pt' in values:
        baseline['size_px'] = points_to_logical_pixels(values['size_pt'])
    baseline.setdefault('italic', False)
    return baseline
