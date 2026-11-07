"""Renderer-independent immutable looks: text/strokes in points, canvas in mm."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import re

from .figure_spec import _Node, FrozenMap, FigureValidationError, thaw_json

STYLE_VERSION = 1
SIZE_KEYS = frozenset({'body_size_pt', 'title_size_pt', 'axis_size_pt', 'tick_size_pt',
    'legend_size_pt', 'caption_size_pt', 'fit_label_size_pt', 'panel_size_pt', 'line_width_pt', 'marker_size_pt'})
CANVAS_KEYS = frozenset({'canvas_width_mm', 'canvas_height_mm'})
CANVAS_MIN_MM, CANVAS_MAX_MM = 0.1, 2000.
TEXT_KEYS = frozenset({'font_family', 'background', 'data_color', 'fit_color'})
ROLE_PROPERTIES = frozenset({'family', 'size_pt', 'bold', 'italic', 'color', 'align'})
LOOK_KEYS = SIZE_KEYS | CANVAS_KEYS | TEXT_KEYS | {'grid', 'frame', 'palette', 'text_roles'}
DEFAULT_LOOK = FrozenMap(dict(font_family='Arial', body_size_pt=7., title_size_pt=7.,
    axis_size_pt=7., tick_size_pt=6., legend_size_pt=6., caption_size_pt=7.,
    fit_label_size_pt=7., panel_size_pt=8., line_width_pt=.75, marker_size_pt=3., background='light',
    data_color='#0072b2', fit_color='#d55e00', grid=False, text_roles={'panel': {'bold': True}}))


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise FigureValidationError(f'{name} must be a finite positive number')


def validate_look(value):
    """Validate known values without discarding a future look property."""
    out = FrozenMap(value)
    for key, item in out.items():
        if key in SIZE_KEYS:
            _positive(item, key)
        elif key in CANVAS_KEYS:
            _positive(item, key)
            if not CANVAS_MIN_MM <= item <= CANVAS_MAX_MM:
                raise FigureValidationError(f'{key} must be between {CANVAS_MIN_MM:g} and {CANVAS_MAX_MM:g} mm')
        elif key in TEXT_KEYS and type(item) is not str:
            raise FigureValidationError(f'{key} must be text')
        elif key in ('grid', 'frame') and type(item) is not bool:
            raise FigureValidationError(f'{key} must be boolean')
        elif key == 'palette':
            if (not isinstance(item, (tuple, list)) or not 1 <= len(item) <= 256
                    or any(type(color) is not str or re.fullmatch(r'#[0-9a-fA-F]{6}', color) is None
                           for color in item)):
                raise FigureValidationError('palette requires 1–256 hexadecimal #RRGGBB colors')
        elif key == 'text_roles':
            if not isinstance(item, Mapping):
                raise FigureValidationError('text_roles must be an object')
            for role, properties in item.items():
                if not isinstance(properties, Mapping):
                    raise FigureValidationError(f'Text role {role} must be an object')
                for prop, setting in properties.items():
                    if prop == 'size_pt':
                        _positive(setting, f'{role}.size_pt')
                    elif prop in ('bold', 'italic') and type(setting) is not bool:
                        raise FigureValidationError(f'{role}.{prop} must be boolean')
                    elif prop in ('family', 'color', 'align') and type(setting) is not str:
                        raise FigureValidationError(f'{role}.{prop} must be text')
    return out


@dataclass(frozen=True)
class StyleRef(_Node):
    style_id: str
    revision: int
    _strings = ('style_id',)

    def __post_init__(self):
        super().__post_init__()
        if not self.style_id or type(self.revision) is not int or self.revision < 1:
            raise FigureValidationError('A style reference needs an identity and positive revision')


@dataclass(frozen=True)
class FigureStyle(_Node):
    style_id: str
    revision: int
    name: str
    look: FrozenMap = field(default_factory=FrozenMap)
    columns: FrozenMap = field(default_factory=FrozenMap)
    provenance: FrozenMap = field(default_factory=FrozenMap)
    version: int = STYLE_VERSION
    _strings = ('style_id', 'name')
    _maps = ('look', 'columns', 'provenance')

    def __post_init__(self):
        super().__post_init__()
        StyleRef(self.style_id, self.revision)
        if type(self.version) is not int or self.version != STYLE_VERSION:
            raise FigureValidationError('Unsupported figure style version')
        if not self.name.strip():
            raise FigureValidationError('A figure style needs a name')
        validate_look(self.look)
        for key in ('single_mm', 'double_mm'):
            if key in self.columns:
                _positive(self.columns[key], key)

    @property
    def ref(self):
        return StyleRef(self.style_id, self.revision)


@dataclass(frozen=True)
class ResolvedLook:
    values: FrozenMap
    origins: FrozenMap


def inherits_native_defaults(style):
    """Whether omitted fields retain renderer defaults, as a sparse native look did."""
    provenance = (style.provenance if isinstance(style, FigureStyle) else
                  style.get('provenance', {}) if isinstance(style, Mapping) else {})
    return bool(provenance.get('legacy_identity') or
                provenance.get('inherit_native_defaults') is True)


def resolve_look(style_or_look, figure_overrides=None, element_overrides=None, defaults=None):
    """Resolve each property without turning inherited values into explicit overrides.

    Nested role origins have the same shape as values. None means absent; an explicit
    false, empty family or value equal to a default remains explicit.
    """
    resolved, origins = {}, {}

    def merge(dst, origin, source, label):
        for key, value in source.items():
            if isinstance(value, Mapping):
                if not isinstance(dst.get(key), dict):
                    dst[key], origin[key] = {}, {}
                merge(dst[key], origin[key], value, label)
            else:
                dst[key], origin[key] = thaw_json(value), label

    look = style_or_look.look if isinstance(style_or_look, FigureStyle) else style_or_look
    if defaults is None:
        # A legacy preset describes only its explicit overrides. Its omitted values
        # historically belonged to each native renderer/series, not a journal default.
        # The native bridge promotes any explicit prior values into figure overrides.
        defaults = ({} if isinstance(style_or_look, FigureStyle) and
                    inherits_native_defaults(style_or_look) else DEFAULT_LOOK)
    for value, label in ((defaults, 'default'),
                         (look or {}, 'style'), (figure_overrides or {}, 'figure'),
                         (element_overrides or {}, 'element')):
        merge(resolved, origins, validate_look(value), label)
    return ResolvedLook(FrozenMap(resolved), FrozenMap(origins))


def points_to_logical_pixels(points, logical_dpi=96.):
    _positive(points, 'points'); _positive(logical_dpi, 'logical_dpi')
    return float(points) * float(logical_dpi) / 72.


def logical_pixels_to_points(pixels, logical_dpi=96.):
    _positive(pixels, 'pixels'); _positive(logical_dpi, 'logical_dpi')
    return float(pixels) * 72. / float(logical_dpi)


def normalize_style_binding(value):
    """Validate a detached history payload, preserving unknown extension keys."""
    out = thaw_json(FrozenMap(value))
    if type(out.get('version')) is not int or out['version'] != 1:
        raise FigureValidationError('Unsupported style binding version')
    snapshot = FigureStyle.from_dict(out.get('snapshot'))
    ref = None if out.get('ref') is None else StyleRef.from_dict(out['ref'])
    if ref is not None and (ref.style_id, ref.revision) != (snapshot.style_id, snapshot.revision):
        raise FigureValidationError('Style binding reference disagrees with its snapshot')
    out['snapshot'] = snapshot.to_dict()
    out['ref'] = None if ref is None else ref.to_dict()
    out['overrides'] = thaw_json(validate_look(out.get('overrides', {})))
    return out


def style_binding(style, overrides=None):
    return normalize_style_binding(dict(version=1, ref=style.ref.to_dict(),
                                       snapshot=style.to_dict(), overrides=overrides or {}))


def builtin_styles():
    """Immutable journal definitions, independent of renderer or application settings."""
    result = []
    for ident, name, size, family, width, marker in (
        ('builtin.journal.arial7', '7 pt Arial · Nature, Elsevier', 7., 'Arial', .75, 3.),
        ('builtin.journal.arial8', '8 pt Arial · PLOS, Springer, AIP', 8., 'Arial', 1., 4.),
        ('builtin.journal.times9', '9 pt Times New Roman · IEEE', 9., 'Times New Roman', 1., 4.),
        ('builtin.slides.arial18', 'Slides · 18 pt Arial', 18., 'Arial', 3., 12.)):
        look = thaw_json(DEFAULT_LOOK)
        look.update(font_family=family, body_size_pt=size, title_size_pt=size,
                    axis_size_pt=size, tick_size_pt=16. if size == 18. else size-1,
                    legend_size_pt=16. if size == 18. else size-1,
                    caption_size_pt=size, fit_label_size_pt=size, panel_size_pt=size + 1.,
                    line_width_pt=width, marker_size_pt=marker)
        result.append(FigureStyle(ident, 1, name, look,
            columns={'single_mm': 89., 'double_mm': 183.,
                     'basis': 'generic-reference', 'label': 'Reference columns'},
            provenance={'builtin': True, 'column_note':
                'Generic reference widths; not verified journal requirements or compliance.'}))
    return tuple(result)


def resolve_bound_text(binding, key, explicit=None):
    """Effective TextStyle properties and origins; inherited roles stay out of storage."""
    binding = normalize_style_binding(binding)
    resolved = resolve_look(FigureStyle.from_dict(binding["snapshot"]), binding["overrides"])
    look, sources = thaw_json(resolved.values), thaw_json(resolved.origins)
    if key == "title":
        role, size = "title", "title_size_pt"
    elif key.startswith("axis.") and key.endswith(".label"):
        role, size = "axis", "axis_size_pt"
    elif key.startswith("axis.") and key.endswith(".ticks"):
        role, size = "ticks", "tick_size_pt"
    elif key.startswith("legend.entry."):
        role, size = "legend", "legend_size_pt"
    elif key in ("annotation.fit", "fit.label"):
        role, size = "fit_label", "fit_label_size_pt"
    elif key in ("annotation.provenance", "annotation.caption", "caption"):
        role, size = "caption", "caption_size_pt"
    elif key == "annotation.panel" or key.startswith("annotation.panel."):
        role, size = "panel", "panel_size_pt"
    else:
        role, size = "body", "body_size_pt"
    values = {"family": look.get("font_family"), "size_pt": look.get(size)}
    origins = {"family": sources.get("font_family", "default"),
               "size_pt": sources.get(size, "default")}
    for name in dict.fromkeys((role, key)):
        for prop, value in look.get("text_roles", {}).get(name, {}).items():
            values[prop] = value
            origins[prop] = sources.get("text_roles", {}).get(name, {}).get(prop, "style")
    for prop, value in (explicit or {}).items():
        if prop != "text" and value is not None:
            values[prop], origins[prop] = value, "element"
    # Missing legacy properties must reach the renderer as absence, never a None
    # override that would reset its own font, category colour or native size.
    values = {prop: value for prop, value in values.items() if value is not None}
    return values, {prop: origins[prop] for prop in values}


def resolve_series_color(binding, *, role="data", index=0, native_color=None,
                         identity_color=None, explicit_color=None):
    """Resolve ink without changing source/class identities or recorded artwork.

    Applying a palette explicitly is an owner action that writes per-series
    ``explicit_color``. A style application alone never rewrites identity colors.
    ``index`` is the stable descriptor order, never the order of visible artists.
    """
    if explicit_color is not None:
        return explicit_color
    if identity_color is not None:
        return identity_color
    if not binding:
        return native_color
    binding = normalize_style_binding(binding)
    style = FigureStyle.from_dict(binding['snapshot'])
    look = resolve_look(style, binding['overrides']).values
    key = 'fit_color' if role == 'fit' else 'data_color'
    if key in binding['overrides']:
        return binding['overrides'][key]
    palette = look.get('palette', ())
    if role == 'data' and palette:
        if type(index) is not int or index < 0:
            raise FigureValidationError('A palette index must be a nonnegative integer')
        return palette[index % len(palette)]
    return look.get(key, native_color)
