"""Declared scientific quantities at presentation boundaries.

This module changes no analytical values, schema, acquisition declarations or stored
recipes. It imports no GUI or analysis modules. A missing declaration never acquires
seconds from a column suffix. Callers pass the existing schema and source units; the
small fallback catalog uses only established NanoScope keys.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Real
from types import MappingProxyType

from .figure_spec import _Node, FrozenMap, FigureValidationError, thaw_json


class QuantityError(FigureValidationError):
    """The declared presentation cannot be converted without guessing its meaning."""


_TIME_SECONDS = MappingProxyType({'s': 1., 'ms': .001, 'µs': .000001, 'min': 60., 'h': 3600.})
_UNIT_ALIASES = MappingProxyType({
    'second': 's', 'seconds': 's', 'sec': 's', 'millisecond': 'ms', 'milliseconds': 'ms',
    'us': 'µs', 'μs': 'µs', 'microsecond': 'µs', 'microseconds': 'µs',
    'minute': 'min', 'minutes': 'min', 'hour': 'h', 'hours': 'h',
    'samples': 'sample', 'dimensionless': '-', '1': '-',
})
_CLOCK_BASES = frozenset({'', 'recording', 'instrument', 'relative', 'view'})
_QUANTITIES = frozenset({'duration', 'clock', 'sample', 'rate', 'dimensionless', 'current',
    'signal', 'voltage', 'charge', 'density', 'count', 'categorical', 'unknown'})
# label, stored unit, semantic quantity, clock basis. Source declarations override units.
_FIELDS = MappingProxyType({
    'event_id': ('Event ID', '-', 'count', ''),
    'recording': ('Recording', '-', 'categorical', ''),
    'pore_id': ('Pore ID', '-', 'categorical', ''),
    'start_idx': ('Start sample', 'sample', 'sample', ''),
    'end_idx': ('End sample', 'sample', 'sample', ''),
    'n_samples': ('Samples', 'sample', 'sample', ''),
    't_start_s': ('Start time', 's', 'clock', 'recording'),
    't_end_s': ('End time', 's', 'clock', 'recording'),
    't0_s': ('Clock origin', 's', 'clock', 'instrument'),
    'time_s': ('Time', 's', 'clock', 'relative'),
    'onset_s': ('Onset time', 's', 'clock', 'recording'),
    'original_onset_s': ('Original source onset', 's', 'clock', 'recording'),
    'dwell_s': ('Dwell time', 's', 'duration', ''),
    'rise_s': ('Rise time', 's', 'duration', ''),
    'fall_s': ('Fall time', 's', 'duration', ''),
    'duration_s': ('Duration', 's', 'duration', ''),
    'analyzed_duration_s': ('Analyzed duration', 's', 'duration', ''),
    'median_dwell_s': ('Median dwell', 's', 'duration', ''),
    'median_dwell_us': ('Median dwell', 'µs', 'duration', ''),
    'dwell_ms': ('Dwell time', 'ms', 'duration', ''),
    'incubation_h': ('Incubation time', 'h', 'duration', ''),
    'baseline_pA': ('Baseline current', 'pA', 'current', ''),
    'depth_pA': ('Median depth', 'pA', 'current', ''),
    'max_dev_pA': ('Peak excursion', 'pA', 'current', ''),
    'mean_dev_pA': ('Mean excursion', 'pA', 'current', ''),
    'amplitude_pA': ('Selected amplitude', 'pA', 'current', ''),
    'frac_depth': ('Fractional depth ΔI/I₀', '-', 'dimensionless', ''),
    'frac_amplitude': ('Fractional amplitude ΔI/I₀', '-', 'dimensionless', ''),
    'ecd_pA_s': ('Integrated charge deficit', 'pA*s', 'charge', ''),
    'area_ecd': ('Integrated charge deficit', 'signal*s', 'charge', ''),
    'snr': ('Signal-to-noise ratio', '-', 'dimensionless', ''),
    'n_levels': ('Resolved levels', '-', 'count', ''),
    'n_events': ('Event count', '-', 'count', ''),
    'capture_rate_hz': ('Capture rate', 'Hz', 'rate', ''),
    'accepted': ('Accepted', '-', 'categorical', ''),
    'filter_limited': ('Filter limited', '-', 'categorical', ''),
    'flag': ('Flags', '-', 'categorical', ''),
})
# Durations are shown in ms unless the quantity has a unit of its own here. Incubation is hours
# long (2 h would read 7,200,000 ms): it is typed, measured and written in hours.
_DISPLAY_UNITS = MappingProxyType({'incubation_h': 'h'})


def _unit(value):
    if type(value) is not str:
        raise QuantityError('A unit declaration must be text')
    return _UNIT_ALIASES.get(value, value)


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise QuantityError(f'{name} must be a finite real number')
    if positive and value <= 0:
        raise QuantityError(f'{name} must be positive')
    return float(value)


@dataclass(frozen=True)
class QuantityPresentation(_Node):
    key: str
    label: str
    quantity: str
    stored_unit: str
    display_unit: str
    factor: float = 1.
    offset: float = 0.
    clock_basis: str = ''
    known: bool = True
    source: str = 'declared'
    _strings = ('key', 'label', 'quantity', 'stored_unit', 'display_unit', 'clock_basis', 'source')

    def __post_init__(self):
        super().__post_init__()
        if not self.key or self.quantity not in _QUANTITIES:
            raise QuantityError('A quantity needs a stable key and supported semantic kind')
        if self.clock_basis not in _CLOCK_BASES:
            raise QuantityError('Unknown clock basis')
        if type(self.known) is not bool:
            raise QuantityError('known must be boolean')
        object.__setattr__(self, 'factor', _number(self.factor, 'display factor', positive=True))
        object.__setattr__(self, 'offset', _number(self.offset, 'display offset'))
        if not self.known and (self.factor != 1 or self.offset != 0):
            raise QuantityError('An unknown quantity cannot be silently rescaled')
        if self.offset and self.quantity != 'clock':
            raise QuantityError('Only clock instants may have an origin offset')
        if self.quantity != 'clock' and self.clock_basis:
            raise QuantityError('Only clock instants may name a clock basis')


@dataclass(frozen=True)
class AxisDisplayBasis(_Node):
    quantity: QuantityPresentation
    coordinate_frame: str = 'linear'
    _nodes = {'quantity': QuantityPresentation}
    _strings = ('coordinate_frame',)

    def __post_init__(self):
        super().__post_init__()
        if self.coordinate_frame not in ('linear', 'log10'):
            raise QuantityError('An axis coordinate frame is linear or log10')
        if self.coordinate_frame == 'log10' and self.quantity.offset:
            raise QuantityError('A shifted clock cannot be relabeled on a log axis')


def _classify(unit):
    if unit in _TIME_SECONDS: return 'duration'
    if unit == 'sample': return 'sample'
    if unit == '-': return 'dimensionless'
    if unit in ('Hz', 'kHz', '1/s', '/s', 's^-1', 's⁻¹'): return 'rate'
    if unit in ('A', 'mA', 'µA', 'uA', 'nA', 'pA'): return 'current'
    if unit in ('V', 'mV', 'µV', 'uV'): return 'voltage'
    if unit in ('pA*s', 'pA·s', 'signal*s', 'signal·s', 'C', 'pC'): return 'charge'
    if unit == 'signal': return 'signal'
    return 'unknown'


def _schema_kind(unit, fallback):
    classified = _classify(_unit(unit))
    if fallback == 'clock' and classified == 'duration': return 'clock'
    if fallback in ('count', 'categorical') and classified == 'dimensionless': return fallback
    return classified


def quantity_for(key, *, schema=None, declared_unit=None, quantity=None, clock_basis=None,
                 display_unit=None, label=None):
    """Resolve a known key or existing schema declaration; never infer units by suffix.

    ``schema[key]`` accepts existing (dtype, unit, description) tuples or explicit
    metadata with unit/quantity/label/clock_basis. ``declared_unit=''`` is an explicit
    unknown source unit and overrides a historical column spelling such as depth_pA.
    A duration or clock is shown in ``display_unit``; left out, that is ms, except for
    the one quantity that has a unit of its own (incubation time, in hours).
    """
    if type(key) is not str or not key:
        raise QuantityError('A quantity key must be nonempty text')
    fallback = _FIELDS.get(key, (key.replace('_', ' ').strip().capitalize(), '', 'unknown', ''))
    title, unit, kind, basis = fallback
    source, description, dtype = ('catalog' if key in _FIELDS else 'unknown'), '', ''
    item = schema.get(key) if isinstance(schema, Mapping) else None
    if item is not None:
        source = 'schema'
        if isinstance(item, Mapping):
            unit = item.get('unit', '')
            kind = item.get('quantity', _schema_kind(unit, fallback[2]))
            title = item.get('label', title)
            basis = item.get('clock_basis', basis)
            description, dtype = item.get('description', ''), item.get('dtype', '')
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            dtype, unit, description = item[:3]
            # Existing clock keys keep their declared clock semantics when the schema
            # confirms a time unit; a conflicting declaration wins over the catalog.
            kind = _schema_kind(unit, fallback[2])
        else:
            raise QuantityError('A schema field needs unit metadata or a dtype/unit/description tuple')
        if type(dtype) is str and (dtype.startswith('string') or dtype in ('bool', 'boolean')):
            kind = 'categorical'
    if declared_unit is not None:
        unit = _unit(declared_unit)
        classified = _classify(unit)
        kind = 'clock' if kind == 'clock' and classified == 'duration' else classified
        source = 'source-unit'
    unit = _unit(unit)
    if quantity is not None: kind = quantity
    if clock_basis is not None:
        basis = clock_basis
        if quantity is None and kind == 'duration': kind = 'clock'
    if kind != 'clock': basis = ''
    if label is not None: title = label
    known = kind != 'unknown' and bool(unit)
    shown, factor = ('' if unit == '-' else unit), 1.
    if kind in ('duration', 'clock'):
        target = _unit(_DISPLAY_UNITS.get(key, 'ms') if display_unit is None else display_unit)
        if unit not in _TIME_SECONDS or target not in _TIME_SECONDS:
            raise QuantityError('Time conversion requires declared supported time units')
        shown, factor = target, _TIME_SECONDS[unit] / _TIME_SECONDS[target]
        known = True
    return QuantityPresentation(key, title, kind, unit, shown, factor,
        clock_basis=basis, known=known, source=source,
        extra={'description': description, 'dtype': dtype})


def sample_time_quantity(key, *, sample_rate_hz, origin_sample=0., origin_s=0.,
                         duration=False, clock_basis='recording', display_unit='ms', label=None):
    """Explicit known-rate conversion; ordinary sample columns remain sample columns."""
    rate = _number(sample_rate_hz, 'sample rate', positive=True)
    start = _number(origin_sample, 'sample origin')
    origin = _number(origin_s, 'time origin')
    target = _unit(display_unit)
    if target not in _TIME_SECONDS:
        raise QuantityError('Sample time requires a declared supported display unit')
    if duration and (start or origin):
        raise QuantityError('A sample duration has no clock origin')
    scale = 1. / _TIME_SECONDS[target]
    return QuantityPresentation(key, label or ('Duration' if duration else 'Time'),
        'duration' if duration else 'clock', 'sample', target, scale / rate,
        0. if duration else (origin - start / rate) * scale,
        clock_basis='' if duration else clock_basis, source='known-sample-rate',
        extra={'sample_rate_hz': rate, 'origin_sample': start, 'origin_s': origin})


def clock_quantity(key, *, offset_s, stored_basis='recording', display_basis='instrument',
                   stored_unit='s', display_unit='ms', label=None):
    """Name-changing clock conversion delegates to the existing rebase-aware clock law."""
    if stored_basis not in ('recording', 'instrument') or display_basis not in ('recording', 'instrument'):
        raise QuantityError('Clock conversion must name recording or instrument basis')
    base = quantity_for(key, declared_unit=stored_unit, quantity='clock',
                        clock_basis=stored_basis, display_unit=display_unit, label=label)
    shift = 0.
    if stored_basis != display_basis:
        from .clock import to_instrument_s, to_recording_s
        shift = (to_instrument_s(offset_s, 0.) if display_basis == 'instrument'
                 else to_recording_s(offset_s, 0.))
        if shift is None:
            raise QuantityError('No known clock origin; cannot name an instrument/recording conversion')
    from .clock import rebased_origin_s
    return replace(base, offset=_number(shift, 'clock origin') / _TIME_SECONDS[base.display_unit],
        clock_basis=display_basis, source='clock', extra={**thaw_json(base.extra),
            'stored_clock_basis': stored_basis, 'display_clock_basis': display_basis,
            'rebased_origin_s': rebased_origin_s(offset_s)})


def _transform(value, operation, *, identity=False):
    if value is None: return None
    if isinstance(value, bool): raise QuantityError('Boolean data are not numeric quantities')
    if isinstance(value, Real): return value if identity else operation(value)
    if isinstance(value, list): return [_transform(v, operation, identity=identity) for v in value]
    if isinstance(value, tuple): return tuple(_transform(v, operation, identity=identity) for v in value)
    if hasattr(value, '__array__'):
        # Optional array support is detached and lazy; importing this module remains
        # independent of a numeric renderer. Scientific input arrays are never mutated.
        import numpy as np
        array = np.asarray(value)
        if array.dtype.kind not in 'iuf':
            raise QuantityError('A numeric quantity array must contain real numbers')
        return array.copy() if identity else operation(array.astype(float, copy=True))
    raise QuantityError('A numeric quantity requires a real scalar, sequence or array')


def to_display(value, descriptor):
    """Convert detached values; None/nonfinite values remain unavailable values."""
    if not isinstance(descriptor, QuantityPresentation): raise QuantityError('Expected QuantityPresentation')
    return _transform(value, lambda x: x * descriptor.factor + descriptor.offset,
                      identity=descriptor.factor == 1 and descriptor.offset == 0)


def from_display(value, descriptor):
    if not isinstance(descriptor, QuantityPresentation): raise QuantityError('Expected QuantityPresentation')
    return _transform(value, lambda x: (x - descriptor.offset) / descriptor.factor,
                      identity=descriptor.factor == 1 and descriptor.offset == 0)


def parse_display_number(text, descriptor, *, allow_blank=False):
    """An explicit finite edit; arbitrary units/expressions are never evaluated."""
    if type(text) is not str: raise QuantityError('An edited number must be text')
    raw = text.strip().replace('−', '-')
    if not raw and allow_blank: return None
    try: value = float(raw)
    except (ValueError, OverflowError) as exc: raise QuantityError('Enter a finite number') from exc
    _number(value, 'edited value')
    return from_display(value, descriptor)


def column_entry(key, descriptor):
    """A ColumnChecklist/combo entry; the itemData key is never replaced by its label."""
    if key != descriptor.key: raise QuantityError('Column key disagrees with its quantity descriptor')
    unit = descriptor.display_unit
    shown = f'{descriptor.label} ({unit})' if unit else descriptor.label
    if not descriptor.known and not unit: shown += ' (units unknown)'
    tip = f'{key} — stored unit: {descriptor.stored_unit or "undeclared"}'
    if unit: tip += f'; displayed unit: {unit}'
    if descriptor.clock_basis: tip += f'; {descriptor.clock_basis} clock'
    if descriptor.extra.get('description'): tip += '\n' + descriptor.extra['description']
    return key, shown, tip


def format_quantity(value, descriptor, *, precision=6, include_unit=True, missing='—'):
    if type(precision) is not int or not 1 <= precision <= 17: raise QuantityError('Precision must be 1–17')
    if value is None: return missing
    if descriptor.quantity == 'categorical': return str(value)
    if isinstance(value, bool) or not isinstance(value, Real): raise QuantityError('Expected numeric display value')
    if not math.isfinite(value): return missing
    displayed = to_display(value, descriptor)
    if not math.isfinite(displayed): return missing
    text = f'{displayed:.{precision}g}'
    return f'{text} {descriptor.display_unit}' if include_unit and descriptor.display_unit else text


def density_quantity(axes, *, measures=None, normalization='density', key='density', label='Density'):
    """Linear densities use inverse Jacobians; counts/probabilities never do.

    An axis drawn logarithmically can still hold density per *linear stored unit*.
    Use measures='log10' only for a kernel explicitly declared per decade.
    """
    axes = tuple(axes)
    if not axes or any(not isinstance(v, QuantityPresentation) for v in axes):
        raise QuantityError('Density requires declared axis quantities')
    if normalization == 'count': return QuantityPresentation(key, 'Count', 'count', 'count', 'count')
    if normalization == 'probability': return QuantityPresentation(key, 'Probability per bin', 'dimensionless', '-', '')
    if normalization != 'density': raise QuantityError('Unknown density normalization')
    measures = (('linear',) * len(axes) if measures is None else
                (measures,) * len(axes) if isinstance(measures, str) else tuple(measures))
    if len(measures) != len(axes) or any(m not in ('linear', 'log10') for m in measures):
        raise QuantityError('Every density axis must declare linear or log10 measure')
    if any(not q.known or q.quantity in ('categorical', 'count', 'sample') for q in axes):
        raise QuantityError('Density units require known continuous axis quantities')
    before, after, factor = [], [], 1.
    for axis, measure in zip(axes, measures):
        if measure == 'log10':
            if axis.offset: raise QuantityError('A shifted clock has no multiplicative per-decade conversion')
            before.append('decade'); after.append('decade')
        else:
            factor /= axis.factor
            if axis.stored_unit not in ('', '-'): before.append(axis.stored_unit)
            if axis.display_unit: after.append(axis.display_unit)
    def inverse(units):
        if not units:
            return ''
        denominator = '·'.join(unit.replace('*', '·') for unit in units)
        compound = len(units) > 1 or any(sign in denominator for sign in ('·', '/', ' '))
        return '1/' + ('(' + denominator + ')' if compound else denominator)
    return QuantityPresentation(key, label, 'density', inverse(before), inverse(after), factor,
        source='jacobian', extra={'axes': [a.to_dict() for a in axes], 'measures': list(measures)})


def view_to_display(value, basis):
    if not isinstance(basis, AxisDisplayBasis): raise QuantityError('Expected AxisDisplayBasis')
    if basis.coordinate_frame == 'linear': return to_display(value, basis.quantity)
    def power(x):
        try: return 10. ** x
        except OverflowError: return math.inf
    return to_display(_transform(value, power), basis.quantity)


def display_to_view(value, basis):
    if not isinstance(basis, AxisDisplayBasis): raise QuantityError('Expected AxisDisplayBasis')
    stored = from_display(value, basis.quantity)
    if basis.coordinate_frame == 'linear': return stored
    def logarithm(x):
        if hasattr(x, '__array__'):
            import numpy as np
            if np.any(x <= 0): raise QuantityError('Log-axis values must be positive')
            return np.log10(x)
        if x <= 0: raise QuantityError('Log-axis values must be positive')
        return math.log10(x)
    return _transform(stored, logarithm)


def convert_pin(values, old, new, *, value_unit_basis=None):
    """Re-express declared display pins without touching data pins or unknown fields.

    The default native AxisPin vocabulary is min/max/tick_major. Mixed Distributions
    maps must supply their existing value_unit_basis map; manual_x/y stay data values.
    Invalid typed drafts are refused here so their owner can retain text with old basis.
    """
    if not isinstance(values, Mapping): raise QuantityError('Pin values must be an object')
    if not isinstance(old, QuantityPresentation) or not isinstance(new, QuantityPresentation):
        raise QuantityError('Pin conversion requires old and new quantity descriptors')
    if (old.key, old.quantity, old.stored_unit) != (new.key, new.quantity, new.stored_unit):
        raise QuantityError('A unit change cannot transfer a pin to a different stored quantity')
    if old.extra.get('sample_rate_hz') != new.extra.get('sample_rate_hz'):
        raise QuantityError('A unit change cannot substitute a different sample-rate declaration')
    old_clock = old.extra.get('stored_clock_basis', old.clock_basis)
    new_clock = new.extra.get('stored_clock_basis', new.clock_basis)
    if old_clock != new_clock: raise QuantityError('Pin values have different stored clock bases')
    out = thaw_json(FrozenMap(values))
    if value_unit_basis is None:
        bases = {key: 'display' for key in ('min', 'max', 'tick_major') if key in values}
    else:
        if not isinstance(value_unit_basis, Mapping) or set(value_unit_basis) != set(values):
            raise QuantityError('Mixed pins require one unit basis for every stored field')
        bases = dict(value_unit_basis)
        if any(v not in ('data', 'display') for v in bases.values()): raise QuantityError('Unknown pin value basis')
    for key, basis in bases.items():
        value = values[key]
        if basis == 'data' or value is None or value == '': continue
        if (old.factor, old.offset) == (new.factor, new.offset): continue
        text = type(value) is str
        if text:
            try: value = float(value.replace('−', '-'))
            except ValueError as exc: raise QuantityError(f'Cannot convert invalid pin draft {key}') from exc
        _number(value, f'pin {key}')
        converted = (value * new.factor / old.factor if key == 'tick_major' else
                     to_display(from_display(value, old), new))
        out[key] = format(converted, '.17g') if text else converted
    return out
