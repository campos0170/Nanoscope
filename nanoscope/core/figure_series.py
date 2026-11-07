"""Stable figure-local series targets and presentation-only overrides."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from .figure_spec import _Node, FrozenMap, FigureValidationError, thaw_json

SERIES_FIELDS = frozenset({'color', 'line_width_pt', 'marker_size_pt', 'visible'})


def stable_series_id(view, role, identity):
    """Identity is a native source/category/semantic key, never a displayed label."""
    parts = (str(view), str(role), str(identity))
    return 'series.' + hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:32]


def validate_series_styles(value):
    if not isinstance(value, Mapping) or len(value) > 2048:
        raise FigureValidationError('Series styles must be a bounded mapping')
    out = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key or len(key) > 4096 or not isinstance(raw, Mapping) or len(raw) > 32:
            raise FigureValidationError('Invalid series style identity')
        style = thaw_json(FrozenMap(raw))
        for field, item in style.items():
            if field == 'color' and (not isinstance(item, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', item)):
                raise FigureValidationError('A series color must be #RRGGBB')
            if field in ('line_width_pt', 'marker_size_pt') and (
                    type(item) not in (int, float) or not math.isfinite(item) or not 0 < item <= 1000):
                raise FigureValidationError('Series sizes must be finite positive points')
            if field == 'visible' and type(item) is not bool:
                raise FigureValidationError('Series visibility must be boolean')
        if style:
            out[key] = style
    return FrozenMap(out)


@dataclass(frozen=True)
class SeriesDescriptor(_Node):
    series_id: str
    label: str
    role: str = 'data'
    identity_kind: str = 'none'
    identity_key: str = ''
    color: str | None = None
    capabilities: tuple[str, ...] = ('color', 'line_width_pt', 'marker_size_pt', 'visible')
    _strings = ('series_id', 'label', 'role', 'identity_kind', 'identity_key')
    _sequences = {'capabilities': str}

    def __post_init__(self):
        super().__post_init__()
        if not self.series_id or self.identity_kind not in ('none', 'source', 'category', 'artist'):
            raise FigureValidationError('A series needs a stable identity')
        if self.identity_kind != 'none' and not self.identity_key:
            raise FigureValidationError('An identified series needs its original key')
        if not set(self.capabilities) <= SERIES_FIELDS:
            raise FigureValidationError('Unsupported series control')
        if self.color is not None:
            validate_series_styles({self.series_id: {'color': self.color}})
