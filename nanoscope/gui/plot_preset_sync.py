"""The project's copy of every plot preset and template (Plot Studio step 6, F1).

Presets and templates are QSettings preferences, so they live on one computer.
:func:`local_plot_presets` is what Save Project copies into the project
(:mod:`nanoscope.core.plot_preset_state`), and :func:`install_project_presets` is what opening one
adds back -- only what this computer lacks, never over a local entry, and saying what it added.
Each carried entry is read with this build's own decoder first, so a template this build cannot
read is named and left in the project rather than planted in the preferences.

Its own module, called only by ``MainWindow``: it has to read the Distributions templates, which
live on ``PlotsWindow``, and a helper the other analytical windows embed may not reach a panel
(``tests/gui/test_measure_navigation.py``). MainWindow is the one place allowed to know them all.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from PySide6 import QtCore

from ..core.plot_preset_state import (
    APPEARANCE_PRESETS, DISTRIBUTIONS_TEMPLATES, WINDOW_TEMPLATES, WORKSPACE_KEY,
    merge_carried, prepare_plot_preset_workspace,
)
from .plot_appearance import PlotAppearance, PlotAppearanceWidget, preset_store
from .plot_templates import (
    WINDOW_KINDS, decode_window_template, load_window_templates, store_window_templates,
)
from .plots_panel import PlotsWindow, plot_config_from_json, plot_config_to_json

__all__ = ["install_project_presets", "local_plot_presets", "unreadable_preset"]


def local_plot_presets() -> dict[str, Any]:
    """This computer's presets and templates, as ``capture_plot_presets`` takes them."""
    return {
        APPEARANCE_PRESETS: PlotAppearanceWidget.user_preset_states(),
        DISTRIBUTIONS_TEMPLATES: {name: value for name, value in
                                  PlotsWindow._load_templates().items()
                                  if isinstance(name, str) and isinstance(value, dict)},
        WINDOW_TEMPLATES: {kind: load_window_templates(kind) for kind in WINDOW_KINDS},
    }


def unreadable_preset(family: str, kind: str | None, payload: Any) -> bool:
    """Whether this build cannot read a carried entry -- so a save must keep it verbatim.

    Passed to ``capture_plot_presets`` as its ``keep`` rule: an entry this build cannot read
    was never installed here, so it is not the computer's to drop.

    ANY failure of a decoder means "unreadable", not only the error it documents: the
    payload comes from a file somebody else wrote, and a reader that raises something
    unforeseen (``int(1e400)`` is an OverflowError) must not stop a project from opening.
    A payload that is not strict JSON (NaN, Infinity) is unreadable too, because every store
    here writes with ``allow_nan=False`` and would refuse it half-way through the open.
    """
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return True
    if family == APPEARANCE_PRESETS:
        return not isinstance(payload, Mapping)
    if family == DISTRIBUTIONS_TEMPLATES:
        try:
            plot_config_from_json(payload)
        except Exception:   # noqa: BLE001 -- see above: any reader failure is "unreadable"
            return True
        return False
    if kind not in WINDOW_KINDS:
        return True
    try:
        decode_window_template(kind, payload)
    except Exception:       # noqa: BLE001 -- as above
        return True
    return False


def _readable(family: str, kind: str | None, carried: Any, unread: list[str]
              ) -> dict[str, Any]:
    if not isinstance(carried, Mapping):
        return {}
    label = {APPEARANCE_PRESETS: "appearance preset",
             DISTRIBUTIONS_TEMPLATES: "Distributions template"}.get(
        family, f"{WINDOW_KINDS.get(str(kind), str(kind))} template")
    readable: dict[str, Any] = {}
    for name, payload in carried.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if unreadable_preset(family, kind, payload):
            unread.append(f"{label} “{name}”")
        elif family == APPEARANCE_PRESETS:
            # Style only, as user_preset_states() reads this computer's copy: an older
            # build's preset that still carries wording must compare equal to it.
            readable[name] = PlotAppearance.from_mapping(payload).style_only().to_dict()
        else:
            readable[name] = payload
    return readable


def install_project_presets(extra: dict) -> tuple[str, ...]:
    """Add the project's carried presets and templates this computer lacks. Banner sentences.

    Never overwrites a local entry (``merge_carried``), never imports one this build cannot
    read, and never marks the project changed: the project already holds what was added.
    """
    notes = list(prepare_plot_preset_workspace(extra))
    section = extra.get(WORKSPACE_KEY)
    if not isinstance(section, dict) or not section:
        return tuple(notes)
    added: list[str] = []
    renamed: list[str] = []
    unread: list[str] = []

    def record(label: str, pairs: list[tuple[str, str]]) -> None:
        for carried_name, stored in pairs:
            added.append(f"{label} “{stored}”")
            if stored != carried_name:
                renamed.append(f"“{carried_name}” → “{stored}”")

    carried = _readable(APPEARANCE_PRESETS, None, section.get(APPEARANCE_PRESETS), unread)
    if carried:
        merged, pairs = merge_carried(PlotAppearanceWidget.user_preset_states(), carried)
        if pairs:
            settings = QtCore.QSettings()
            settings.setValue(PlotAppearanceWidget._PRESETS_KEY,
                              json.dumps(merged, sort_keys=True, allow_nan=False))
            settings.sync()
            record("appearance preset", pairs)
    carried = _readable(DISTRIBUTIONS_TEMPLATES, None, section.get(DISTRIBUTIONS_TEMPLATES),
                        unread)
    if carried:
        # The shipped templates and the "— manual —" sentinel are names the Distributions
        # combo shows ahead of any stored one, so a carried entry under one would be stored
        # and never be pickable: they count as taken.
        shipped = {name: plot_config_to_json(config)
                   for name, config in PlotsWindow._BUILTIN_TEMPLATES.items()}
        shipped[PlotsWindow._TPL_MANUAL] = None
        merged, pairs = merge_carried(PlotsWindow._load_templates(), carried, reserved=shipped)
        if pairs:
            PlotsWindow._store_templates(merged)
            record("Distributions template", pairs)
    windows = section.get(WINDOW_TEMPLATES)
    if isinstance(windows, Mapping):
        for kind, entries in windows.items():
            if kind not in WINDOW_KINDS:
                if entries:
                    unread.append(f"templates for a window this version does not have ({kind})")
                continue
            carried = _readable(WINDOW_TEMPLATES, kind, entries, unread)
            if not carried:
                continue
            merged, pairs = merge_carried(load_window_templates(kind), carried)
            if pairs:
                store_window_templates(kind, merged)
                record(f"{WINDOW_KINDS[kind]} template", pairs)
    if added:
        notes.append(
            f"This project carried {len(added)} plot preset(s) or template(s) this computer did "
            f"not have, and they were added to it: {', '.join(added)}.")
    if renamed:
        notes.append(
            "A preset or template of the same name but different settings already exists on this "
            "computer, so the project's copy came in under a new name and yours is unchanged: "
            + ", ".join(renamed) + ".")
    if unread:
        notes.append(
            f"{len(unread)} of the project's plot presets or templates could not be read by this "
            f"version and were not added; they stay in the project: {', '.join(unread)}.")
    if added:
        # Every open list of presets or templates rebuilds itself; no plot changes.
        preset_store().changed.emit()
    return tuple(notes)
