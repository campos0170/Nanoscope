"""Portable shared styles saved explicitly on this computer, separate from a project."""
from dataclasses import replace
import hashlib
import json
from uuid import uuid4

from PySide6 import QtCore

from ..core.figure_spec import thaw_json, canonical_json
from ..core.figure_style import FigureStyle
from ..core.figure_style_mapping import import_legacy_look
from .plot_appearance import PlotAppearanceWidget

KEY = 'figure_styles/computer_v1'


def _read():
    raw = QtCore.QSettings().value(KEY, '')
    if not raw:
        return {'version': 1, 'styles': {}}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError('The computer style catalog is unreadable and has been left unchanged.') from exc
    if not isinstance(value, dict) or value.get('version') != 1 or not isinstance(value.get('styles'), dict):
        raise ValueError('The computer style catalog belongs to another format and has been left unchanged.')
    return value


def styles():
    """Shared entries plus legacy looks; neither read changes either catalog."""
    result = {}
    for name, native in PlotAppearanceWidget.user_preset_states().items():
        converted = import_legacy_look(native, 'plot_appearance')
        if converted.report['unreadable']:
            continue
        identity = hashlib.sha256(canonical_json({'name': name, 'look': native}).encode()).hexdigest()[:24]
        result[name] = FigureStyle('computer.legacy.' + identity, 1, name, converted.look,
                                  provenance={'inherit_native_defaults': True, 'computer_style': True,
                                              'legacy_computer_preset': name})
    for name, raw in _read()['styles'].items():
        if not isinstance(name, str):
            continue
        try:
            style = FigureStyle.from_dict(raw)
        except (TypeError, ValueError):
            continue  # Preserve unrecognized entries; writes never rebuild them from this filtered list.
        if style.name == name:
            result[name] = style
    return tuple(result[name] for name in sorted(result, key=str.casefold))


def _write(value):
    settings = QtCore.QSettings()
    settings.setValue(KEY, json.dumps(value, sort_keys=True, allow_nan=False))
    settings.sync()
    if settings.status() != QtCore.QSettings.Status.NoError:
        raise ValueError('The computer style catalog could not be saved.')


def save(style, name, *, overwrite=False):
    name = str(name).strip()
    if not name or len(name) > 100:
        raise ValueError('A style name must contain 1–100 characters.')
    value = _read()
    existing = next((entry for entry in styles() if entry.name == name), None)
    if existing is not None and not overwrite:
        raise ValueError('A style with this name already exists on this computer.')
    known = value['styles'].get(name)
    # A legacy preset of the same name must not hide an unreadable future entry.
    if known is not None:
        try:
            checked = FigureStyle.from_dict(known)
            if checked.name != name:
                raise ValueError('The saved style name disagrees with its catalog key.')
        except (TypeError, ValueError) as exc:
            raise ValueError('This saved style cannot be replaced by this build.') from exc
        existing = checked
    provenance = {**thaw_json(style.provenance), 'computer_style': True}
    provenance.pop('builtin', None)
    provenance.pop('project_style', None)
    copied = replace(style, style_id=(existing.style_id if known is not None else 'computer.' + str(uuid4())),
                     revision=(existing.revision + 1 if known is not None else 1), name=name,
                     provenance=provenance)
    value['styles'][name] = copied.to_dict()
    _write(value)
    return copied


def delete(name):
    value = _read()
    native = PlotAppearanceWidget.user_preset_states()
    changed = name in value['styles'] or name in native
    if name in value['styles']:
        try:
            checked = FigureStyle.from_dict(value['styles'][name])
            if checked.name != name:
                raise ValueError('The saved style name disagrees with its catalog key.')
        except (TypeError, ValueError) as exc:
            raise ValueError('This saved style cannot be deleted by this build.') from exc
        del value['styles'][name]
        _write(value)
    if name in native:
        # The old wire remains readable by older builds. Delete only the requested
        # entry, keeping the raw payload of every other entry untouched.
        settings = QtCore.QSettings()
        raw = json.loads(settings.value(PlotAppearanceWidget._PRESETS_KEY, '{}'))
        raw.pop(name, None)
        settings.setValue(PlotAppearanceWidget._PRESETS_KEY, json.dumps(raw, sort_keys=True, allow_nan=False))
        settings.sync()
        if settings.status() != QtCore.QSettings.Status.NoError:
            raise ValueError('The old computer preset could not be removed.')
    return changed
