"""Pure native-payload display migration shared by resident and dormant figures."""
from dataclasses import replace

from ..core.figure_quantity_basis import migrate_appearance_basis, migrate_appearance_history
from ..core.quantity_display import AxisDisplayBasis, quantity_for


def native_quantity_contract(payload, kind):
    """Use persisted page contracts, never labels, pending drafts or table magnitudes."""
    axes, legacy, view_pins = {}, {}, []
    def add(side, quantity, *, view='', primary='', log=False, legacy_log_values=False):
        basis = AxisDisplayBasis(quantity, 'log10' if log else 'linear')
        old = AxisDisplayBasis(replace(quantity, display_unit=quantity.stored_unit, factor=1., offset=0.), basis.coordinate_frame)
        keys = [view + '.' + side] if view else [side]
        if view and view == primary:
            keys.append(side)
        for key in keys:
            axes[key], legacy[key] = basis, old
            if legacy_log_values and log:
                view_pins.append(key)
    config = payload.get('config', {})
    if kind == 'distribution_summaries':
        applied, snapshot = payload.get('applied', {}), payload.get('snapshot', {})
        if applied:
            categorical = bool(snapshot.get('categorical'))
            view = 'categories' if categorical else 'curves'
            side = 'left' if categorical else 'bottom'
            add(side, quantity_for(applied.get('metric') or 'unselected'), view=view, primary='curves')
            add('bottom' if categorical else 'left', quantity_for('source' if categorical else 'probability',
                declared_unit='-', quantity='categorical' if categorical else 'dimensionless'), view=view, primary='curves')
    elif kind == 'noise':
        add('bottom', quantity_for('time_s', clock_basis='recording'), view='spectrogram', primary='psd')
    elif kind == 'voltage_series':
        add('left', quantity_for(config.get('metric') or 'capture_rate_hz'), log=bool(config.get('log_y', False)))
    elif kind == 'event_waveforms':
        if payload.get('density_counts') is not None:
            for side, field, log in (('bottom', 'x_metric', 'log_x'), ('left', 'y_metric', 'log_y')):
                add(side, quantity_for(config.get(field) or ('dwell_s' if side == 'bottom' else 'depth_pA')),
                    view='map', primary='map', log=bool(config.get(log, side == 'bottom')), legacy_log_values=True)
        if payload.get('ensemble') is not None:
            quantity = (quantity_for('normalized_time', declared_unit='-', quantity='dimensionless')
                        if config.get('normalize_time', True) else quantity_for('time_s', clock_basis='relative'))
            add('bottom', quantity, view='ensemble', primary='map')
    # Comparison's Matplotlib host keeps every old unmarked pin in its known artist
    # units until an actual axis edit; the renderer's descriptor supplies the inverse.
    return axes, legacy, tuple(view_pins)


def migrate_native_appearance(value, payload, kind):
    axes, legacy, view_pins = native_quantity_contract(payload, kind)
    if not axes:
        return value
    return migrate_appearance_basis(value, axes, legacy_axes=legacy, legacy_view_pins=view_pins)


def migrate_native_history(capsule, payload, kind):
    axes, legacy, view_pins = native_quantity_contract(payload, kind)
    return migrate_appearance_history(capsule, axes, legacy_axes=legacy, legacy_view_pins=view_pins) if axes else capsule
