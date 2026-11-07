"""Lossless native appearance/history witnesses for displayed scientific quantities.

No label text or data magnitude is evidence of units. Callers supply old page-contract
bases for unmarked native values, and the applied (not pending) bases for the new view.
This module never reads a source, creates a renderer, or changes numerical resources.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from .figure_spec import FrozenMap, thaw_json
from .figure_history_state import HistoryCapsule
from .quantity_display import AxisDisplayBasis, QuantityError, convert_pin, view_to_display

WITNESS_KEY = 'quantity_display'
WITNESS_VERSION = 1


def _basis(value):
    return value if isinstance(value, AxisDisplayBasis) else AxisDisplayBasis.from_dict(value)


def quantity_witness(axes, *, original=None):
    """Replace declared axes while carrying every unrecognized witness field."""
    raw = thaw_json(FrozenMap(original or {}))
    if raw and (type(raw.get('version')) is not int or raw['version'] != WITNESS_VERSION):
        raise QuantityError('Unsupported display-quantity witness; original values must be preserved')
    previous = raw.get('axes', {})
    if not isinstance(previous, Mapping):
        raise QuantityError('Display-quantity axes must be an object')
    converted = {str(key): _basis(value).to_dict() for key, value in axes.items()}
    return {**raw, 'version': WITNESS_VERSION, 'axes': {**previous, **converted}}


def read_quantity_bases(witness):
    if not witness:
        return {}
    if not isinstance(witness, Mapping) or type(witness.get('version')) is not int or witness['version'] != WITNESS_VERSION:
        raise QuantityError('Unsupported display-quantity witness; original values must be preserved')
    axes = witness.get('axes', {})
    if not isinstance(axes, Mapping):
        raise QuantityError('Display-quantity axes must be an object')
    return {str(key): _basis(value) for key, value in axes.items()}


def migrate_appearance_basis(appearance, axes, *, legacy_axes=None, legacy_view_pins=()):
    """Re-express DISPLAY pins once and preserve words, style and raw view ranges.

    ``axes`` and ``legacy_axes`` use the actual native pin keys: primary bare sides or
    namespaced view.side. Every unmarked pin to be migrated requires explicit legacy
    evidence. Other views/unknown fields are carried unchanged. Log tick spacing is in
    decades and therefore never receives a linear unit factor.
    """
    raw = thaw_json(FrozenMap(appearance or {}))
    desired = {str(key): _basis(value) for key, value in axes.items()}
    legacy = {str(key): _basis(value) for key, value in (legacy_axes or {}).items()}
    previous = read_quantity_bases(raw.get(WITNESS_KEY))
    pins = deepcopy(raw.get('axes', {}))
    if not isinstance(pins, Mapping):
        raise QuantityError('Appearance axes must be an object')
    for key, new in desired.items():
        pin = pins.get(key)
        old = previous.get(key, legacy.get(key))
        if pin and old is None:
            raise QuantityError(f'No declared previous display basis for {key}')
        if pin:
            if not isinstance(pin, Mapping):
                raise QuantityError(f'Axis pin {key} must be an object')
            if old.coordinate_frame != new.coordinate_frame and pin.get('tick_major') is not None:
                raise QuantityError(f'Axis {key} changed log frame; its owner must handle tick spacing')
            if key not in previous and key in legacy_view_pins:
                pin = dict(pin)
                for field in ('min', 'max'):
                    if pin.get(field) is not None:
                        pin[field] = view_to_display(pin[field], old)
            field_bases = {field: ('display' if field in ('min', 'max') or
                (field == 'tick_major' and old.coordinate_frame != 'log10') else 'data')
                for field in pin}
            pins[key] = convert_pin(pin, old.quantity, new.quantity, value_unit_basis=field_bases)
    if 'axes' in raw or pins:
        raw['axes'] = pins
    raw[WITNESS_KEY] = quantity_witness(desired, original=raw.get(WITNESS_KEY))
    return raw


def migrate_appearance_history(capsule, axes, *, legacy_axes=None, legacy_view_pins=()):
    """Transform every appearance Undo/Redo value, retaining stack order and memos."""
    if not isinstance(capsule, HistoryCapsule):
        capsule = HistoryCapsule.from_dict(capsule)
    if capsule.codec != 'nanoscope.appearance-history/1':
        raise QuantityError('Appearance basis migration requires an appearance history capsule')
    def migrate(value):
        existing = read_quantity_bases(value.get(WITNESS_KEY))
        # Undo may retain a different measurement's pins. Its already-declared basis
        # remains authoritative; a new visible measurement cannot reinterpret it.
        applicable = {key: basis for key, basis in axes.items()
                      if key not in existing or existing[key].quantity.key == _basis(basis).quantity.key}
        return migrate_appearance_basis(value, applicable, legacy_axes=legacy_axes, legacy_view_pins=legacy_view_pins)
    return replace(capsule, base=migrate(capsule.base),
                   past=tuple(migrate(value) for value in capsule.past),
                   future=tuple(migrate(value) for value in capsule.future))
