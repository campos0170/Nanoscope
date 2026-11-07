"""Attach explicit unit presentation to existing axes without changing their data."""
from ..core.quantity_display import AxisDisplayBasis, QuantityError


def configure_quantity_axis(axis, basis):
    """Use existing linear/log tick multipliers; pointer and editor share the witness.

    The axis is an existing pyqtgraph AxisItem. It stays in raw data/log10 coordinates,
    so geometry, data ranges, source hit testing and zoom history need no rescaling.
    """
    if not isinstance(basis, AxisDisplayBasis):
        basis = AxisDisplayBasis.from_dict(basis)
    if basis.quantity.offset:
        raise QuantityError('An axis origin must be applied in its declared clock coordinates before unit presentation')
    axis.enableAutoSIPrefix(False)
    factor = basis.quantity.factor
    if basis.coordinate_frame == 'log10':
        setter = getattr(axis, 'set_display_factor', None)
        if not callable(setter):
            raise QuantityError('Log quantity presentation requires a readable log axis')
        axis.setScale(1.)
        setter(factor)
    else:
        setter = getattr(axis, 'set_display_factor', None)
        if callable(setter):
            setter(1.)
        axis.setScale(factor)
    axis._nanoscope_quantity_basis = basis
    return basis


def present_page_axes(page, plot, quantities, *, view='', primary='', frames=None, legacy=None, legacy_view_pins=()):
    """Present known applied quantities, migrating native pins/history once.

    Call before applying appearance pins. Raw view ranges and all plot items stay in
    their existing coordinates; only tick labels and the common editor inverse change.
    """
    from ..core.figure_quantity_basis import read_quantity_bases
    from ..core.quantity_display import quantity_for
    frames = frames or {}
    desired, previous = {}, {}
    existing = read_quantity_bases(page._appearance.quantity_display)
    for side, quantity in quantities.items():
        axis = plot.getAxis(side)
        frame = frames.get(side, 'log10' if axis.logMode else 'linear')
        basis = AxisDisplayBasis(quantity, frame)
        keys = [view + '.' + side] if view else [side]
        if view and view == primary:
            keys.append(side)
        for key in keys:
            desired[key] = basis
            old_quantity = (legacy or {}).get(side)
            if old_quantity is None:
                old_quantity = quantity_for(quantity.key, declared_unit=quantity.stored_unit,
                    quantity=quantity.quantity, clock_basis=quantity.clock_basis,
                    display_unit=quantity.stored_unit, label=quantity.label)
            previous[key] = AxisDisplayBasis(old_quantity, frame)
        configure_quantity_axis(axis, basis)
    # Quantity changes already have their own native reset rule. An old mark awaiting
    # that reset must not be converted using the next measurement's identity.
    migration = {key: basis for key, basis in desired.items()
                 if key not in existing or existing[key].quantity.key == basis.quantity.key}
    if migration:
        page._appearance = page._appearance_widget.migrate_quantity_bases(
            migration, legacy_axes=previous, legacy_view_pins=legacy_view_pins)
    different = {key: basis for key, basis in desired.items() if key not in migration}
    if different:
        page._appearance = page._appearance_widget.adopt_quantity_bases(different)


def export_display_columns(frame, columns):
    """Add explicit display values/units without changing any scientific CSV column.

    ``columns`` maps an existing raw column to its declared quantity (not view/log
    coordinates). Unknown quantities stay unscaled and have an empty unit witness.
    """
    if frame.empty:
        return frame
    from ..core.quantity_display import to_display
    result = frame.copy(deep=True)
    for name, quantity in columns.items():
        if name not in result:
            continue
        shown = name + '_display'
        unit = shown + '_unit'
        if shown in result or unit in result:
            raise QuantityError('A source column conflicts with a display export column')
        result[shown] = to_display(result[name].to_numpy(copy=True), quantity)
        result[unit] = quantity.display_unit
    return result
