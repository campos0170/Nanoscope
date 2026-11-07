"""Shared-style projection at the legacy native-value boundary.

The binding is part of each native history value. Compatibility look fields are
derived from it on restore; deliberate edits update its canonical overrides first.
This module imports no Qt and does not construct a renderer.
"""
from __future__ import annotations

from copy import deepcopy

from ..core.figure_spec import thaw_json
from ..core.figure_style import resolve_bound_text


NATIVE_LOOK_KEYS = {
    "font_family": "font_family", "title_size": "title_size_pt",
    "axis_size": "axis_size_pt", "tick_size": "tick_size_pt",
    "legend_size": "legend_size_pt", "line_width": "line_width_pt",
    "marker_size": "marker_size_pt", "background": "background",
    "data_color": "data_color", "fit_color": "fit_color", "grid": "grid", "frame": "frame",
}


def resolved_binding(binding):
    from ..core.figure_style import FigureStyle, normalize_style_binding, resolve_look
    binding = normalize_style_binding(binding)
    return resolve_look(FigureStyle.from_dict(binding["snapshot"]), binding["overrides"])


def project_native(mapping, vocabulary="plot_appearance", *, logical_dpi=96):
    """Resolve a binding authoritatively, keeping non-look native values intact."""
    from ..plot.figure_style_mapping import project_legacy_style

    raw = deepcopy(thaw_json(mapping))
    binding = raw.get("style_binding")
    if not binding:
        return raw
    return project_legacy_style(binding, vocabulary, raw, logical_dpi=logical_dpi)


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, path + (key,))
    else:
        yield path, value


def _put(root, path, value):
    for key in path[:-1]:
        root = root.setdefault(key, {})
    root[path[-1]] = deepcopy(value)


def _remove(root, path):
    parents = []
    for key in path[:-1]:
        if key not in root:
            return
        parents.append((root, key))
        root = root[key]
    root.pop(path[-1], None)
    for parent, key in reversed(parents):
        if not parent[key]:
            parent.pop(key)


def reconcile_native_edit(before, after, vocabulary="plot_appearance", *, logical_dpi=96):
    """Make a legacy control gesture an explicit canonical look override.

    Unchanged properties keep their existing explicit/inherited status, even when
    an explicit value equals the style. A removed property clears that override.
    A replaced binding is a style action, so its projection wins over old fields.
    """
    from ..core.figure_style import normalize_style_binding
    from ..plot.figure_style_mapping import import_legacy_look

    before, after = thaw_json(before), deepcopy(thaw_json(after))
    binding = after.get("style_binding")
    if not binding:
        return after
    if binding != before.get("style_binding"):
        return project_native(after, vocabulary, logical_dpi=logical_dpi)
    binding = deepcopy(thaw_json(normalize_style_binding(binding)))
    raw_before = {key: value for key, value in before.items() if key != "style_binding"}
    raw_after = {key: value for key, value in after.items() if key != "style_binding"}
    was = dict(_leaves(thaw_json(import_legacy_look(raw_before, vocabulary,
                                                  logical_dpi=logical_dpi).look)))
    now = dict(_leaves(thaw_json(import_legacy_look(raw_after, vocabulary,
                                                  logical_dpi=logical_dpi).look)))
    overrides = deepcopy(binding.get("overrides", {}))
    for path in was.keys() | now.keys():
        if path in now and path in was and now[path] == was[path]:
            continue
        if path in now:
            _put(overrides, path, now[path])
        else:
            _remove(overrides, path)
    binding["overrides"] = overrides
    after["style_binding"] = binding
    return project_native(after, vocabulary, logical_dpi=logical_dpi)


def bind_native_style(mapping, style, vocabulary="plot_appearance", *, logical_dpi=96):
    """Install a pinned style as one native value, without changing scientific values."""
    from ..core.figure_style import FigureStyle

    if not isinstance(style, FigureStyle):
        style = FigureStyle.from_dict(style)
    raw = deepcopy(thaw_json(mapping))
    raw["style_binding"] = {
        "version": 1,
        "ref": {"style_id": style.style_id, "revision": style.revision},
        "snapshot": style.to_dict(),
        "overrides": {},
    }
    return project_native(raw, vocabulary, logical_dpi=logical_dpi)


def preserve_legacy_baseline(mapping, baseline, vocabulary="plot_appearance"):
    """Sparse imported presets keep unspecified native look values as figure overrides.

    The imported definition is exact and remains immutable. Only these compatibility
    baselines become figure-owned values; a full shared style needs no such fallback.
    """
    from ..core.figure_style import normalize_style_binding
    from ..core.figure_style_mapping import import_legacy_look
    raw = deepcopy(thaw_json(mapping))
    if not raw.get("style_binding"):
        return raw
    binding = normalize_style_binding(raw["style_binding"])
    snapshot = binding["snapshot"]
    from ..core.figure_style import inherits_native_defaults
    if not inherits_native_defaults(snapshot):
        return raw
    previous = import_legacy_look(baseline, vocabulary).look
    overridden = deepcopy(binding["overrides"])
    for key, value in previous.items():
        if key not in snapshot["look"] and key not in overridden:
            overridden[key] = thaw_json(value)
    binding["overrides"] = overridden
    raw["style_binding"] = binding
    return raw
