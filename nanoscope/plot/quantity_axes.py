"""Declared Matplotlib axis presentation over retained scientific artist coordinates."""
from __future__ import annotations

import math

from matplotlib.ticker import Formatter, FuncFormatter, NullFormatter

from ..core.quantity_display import (
    AxisDisplayBasis, QuantityPresentation, quantity_for, view_to_display,
)


class _QuantityMinorFormatter(Formatter):
    """Keep a log formatter's minor-label suppression, converting visible numbers."""
    def __init__(self, original, convert):
        self._original = original
        self._convert = convert

    def set_axis(self, axis):
        super().set_axis(axis)
        self._original.set_axis(axis)

    def set_locs(self, locs):
        super().set_locs(locs)
        self._original.set_locs(locs)

    def __call__(self, value, position=None):
        return self._convert(value, position) if self._original(value, position) else ''


def set_quantity_axis(axis, side, quantity, *, coordinate_frame='linear', label=None):
    """Install a display formatter and a reversible, explicit coordinate witness.

    Matplotlib log axes still store LINEAR artist coordinates; pass 'log10' only
    when the builder itself stored log10(values), e.g. a log-dwell KDE. No artist,
    locator, limits, source data, fitted values or styles are changed here. Labels
    are changed only when the builder supplies an automatic label explicitly.
    """
    if side not in ('bottom', 'left'):
        raise ValueError('A quantity axis is bottom or left')
    if not isinstance(quantity, QuantityPresentation):
        raise TypeError('An axis needs a declared QuantityPresentation')
    basis = AxisDisplayBasis(quantity, coordinate_frame)
    values = dict(getattr(axis, '_nanoscope_quantity_axes', {}))
    values[side] = basis
    axis._nanoscope_quantity_axes = values
    mpl_axis = axis.xaxis if side == 'bottom' else axis.yaxis

    def format_tick(value, _position=None):
        shown = view_to_display(value, basis)
        return '' if not math.isfinite(shown) else f'{shown:g}'

    mpl_axis.set_major_formatter(FuncFormatter(format_tick))
    minor = mpl_axis.get_minor_formatter()
    if not isinstance(minor, NullFormatter):
        mpl_axis.set_minor_formatter(_QuantityMinorFormatter(minor, format_tick))
    if label is not None:
        (axis.set_xlabel if side == 'bottom' else axis.set_ylabel)(str(label))
    return basis


def time_axis(axis, side, key, *, stored_unit='s', clock_basis='', coordinate_frame='linear', label=None):
    """Use a builder's declared artist-coordinate units, never its label text."""
    quantity = quantity_for(key, declared_unit=stored_unit,
                            quantity='clock' if clock_basis else 'duration',
                            clock_basis=clock_basis, display_unit='ms')
    return set_quantity_axis(axis, side, quantity, coordinate_frame=coordinate_frame, label=label)


def figure_quantity_metadata(figure):
    """Detached JSON witness for exported Details and retained figure resources."""
    return {'version': 1, 'axes': [
        {'panel': index, 'side': side, 'basis': basis.to_dict()}
        for index, axis in enumerate(figure.axes)
        for side, basis in getattr(axis, '_nanoscope_quantity_axes', {}).items()
    ]}


def log_time_axis(axis, side, key='dwell_ms', *, stored_unit='ms'):
    """Keep an explicitly labelled log10(time-unit) plot in that existing measure."""
    label = f'log10({stored_unit})'
    quantity = QuantityPresentation(f'log10.{key}', label, 'dimensionless', label, label,
        extra={'underlying_quantity': key, 'underlying_unit': stored_unit, 'transform': 'log10'})
    return set_quantity_axis(axis, side, quantity)
