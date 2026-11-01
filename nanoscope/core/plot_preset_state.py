"""Copies of this computer's plot presets and templates, carried in the v11 workspace.

Appearance presets, Distributions templates and the other windows' templates are QSettings
preferences: they live on the computer that saved them. A project opened on another computer, or
after a preset was deleted, then offered none of the looks its figures were made with. So Save
Project writes a COPY of them here and opening the project adds back the ones the computer lacks.

The sibling of :mod:`nanoscope.core.table_choice_state`, with the same container contract: one
versioned key in ``workspace.extra`` (no project VERSION bump, because a v11 build preserves
unknown extras byte-for-byte and a v10 build already refuses a v11 project), and an unreadable
section quarantined rather than overwritten. Two rules are its own:

* THE COPY NEVER WINS OVER THE COMPUTER. A carried entry is added only under a name the computer
  does not use; one whose name is taken by different content comes in as ``Name (from project)``
  (then ``(from project 2)`` …), so opening somebody's project can never restyle the presets the
  user has already made. Identical content under the same BASE name is not added again, or two
  computers passing one project back and forth would grow a ``(from project)`` copy each trip.
* A NEWER BUILD'S SUB-KEYS ARE KEPT. The section is an object of families; a family this build
  does not know (a future "figure templates") is carried through a save untouched, the same
  promise ``WorkspaceArtifact.extra`` makes one level up.

This module is Qt-free: it moves plain JSON in and out of a project and decides names. Whether a
carried entry is READABLE is the GUI's question (``gui.plot_preset_sync``), because only the windows'
own decoders can answer it.
"""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

WORKSPACE_KEY = "plot_presets_v1"
UNREADABLE_KEY = WORKSPACE_KEY + "_unreadable"

#: The families this build writes. ``window_templates`` is one level deeper: window kind → name.
APPEARANCE_PRESETS = "appearance_presets"
DISTRIBUTIONS_TEMPLATES = "distributions_templates"
WINDOW_TEMPLATES = "window_templates"
FAMILIES = (APPEARANCE_PRESETS, DISTRIBUTIONS_TEMPLATES, WINDOW_TEMPLATES)

#: The longest name either store accepts (``PlotAppearanceWidget.save_user_preset``).
NAME_LIMIT = 100
_FROM_PROJECT = re.compile(r"^(?P<base>.*) \(from project(?: (?P<n>\d+))?\)$")


def prepare_plot_preset_workspace(extra: dict) -> tuple[str, ...]:
    """Quarantine an unreadable section without losing it or blocking future saves."""
    if WORKSPACE_KEY not in extra or isinstance(extra[WORKSPACE_KEY], dict):
        return ()
    key = UNREADABLE_KEY
    suffix = 1
    while key in extra:
        key = f"{UNREADABLE_KEY}_{suffix}"
        suffix += 1
    extra[key] = extra[WORKSPACE_KEY]
    extra[WORKSPACE_KEY] = {}
    return ("The plot presets and templates saved in this project could not be read. Their "
            "original contents are preserved in the project; nothing was added to this "
            "computer.",)


def has_any(local: Mapping[str, Any]) -> bool:
    """Whether ``local`` (as :func:`capture_plot_presets` takes it) holds any entry at all."""
    if local.get(APPEARANCE_PRESETS) or local.get(DISTRIBUTIONS_TEMPLATES):
        return True
    windows = local.get(WINDOW_TEMPLATES)
    return isinstance(windows, Mapping) and any(windows.values())


def capture_plot_presets(local: Mapping[str, Any], previous: Any = None, *,
                         keep: Callable[[str, str | None, Any], bool] | None = None
                         ) -> dict[str, Any]:
    """The section to save: this computer's three families over ``previous``.

    The known families are REPLACED, not merged: the project carries what the saving computer
    has now, so a preset deleted here is not written back by the next save. Two things survive
    from ``previous`` all the same: a family this build does not know, verbatim, and any entry
    ``keep(family, window_kind, payload)`` says this build could not read -- it was never
    installed here, so it is not this computer's to drop. ``local`` is
    ``{family: {name: payload}}`` (``window_templates``: ``{kind: {name: payload}}``).
    """
    if previous is not None and not isinstance(previous, dict):
        raise ValueError("Cannot overwrite an unreadable plot presets section.")
    section = copy.deepcopy(previous) if previous else {}

    def carry(family: str, kind: str | None, old: Any, fresh: dict[str, Any]) -> None:
        if keep is None or not isinstance(old, Mapping):
            return
        for name, payload in old.items():
            if name not in fresh and keep(family, kind, payload):
                fresh[name] = copy.deepcopy(payload)

    for family in (APPEARANCE_PRESETS, DISTRIBUTIONS_TEMPLATES):
        fresh = copy.deepcopy(dict(local.get(family) or {}))
        carry(family, None, section.get(family), fresh)
        section[family] = fresh
    windows = {str(kind): copy.deepcopy(dict(entries))
               for kind, entries in (local.get(WINDOW_TEMPLATES) or {}).items()}
    old_windows = section.get(WINDOW_TEMPLATES)
    if isinstance(old_windows, Mapping):
        for kind, entries in old_windows.items():
            carry(WINDOW_TEMPLATES, str(kind), entries, windows.setdefault(str(kind), {}))
    section[WINDOW_TEMPLATES] = {kind: entries for kind, entries in windows.items() if entries}
    return section


def _canonical(payload: Any) -> str:
    """Content identity: the stored JSON, ignoring which build wrote it.

    ``app_version`` is provenance, not content -- the same template saved by two builds is the
    same template, and comparing it would turn every app update into a "(from project)" copy.
    """
    if isinstance(payload, Mapping):
        payload = {k: v for k, v in payload.items() if k != "app_version"}
    return json.dumps(payload, sort_keys=True, default=str)


def base_name(name: str) -> str:
    """``name`` without a ``(from project)`` / ``(from project N)`` suffix."""
    match = _FROM_PROJECT.match(name)
    return match.group("base") if match else name


def merge_carried(local: Mapping[str, Any], carried: Mapping[str, Any], *,
                  reserved: Mapping[str, Any] | None = None
                  ) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """``carried`` added to ``local`` without replacing anything. Returns (merged, added).

    ``added`` lists ``(carried name, name it was stored under)`` in the carried order. Both
    mappings are ``{name: payload}`` for ONE family (one window kind). Payloads are assumed
    readable already; this decides names only.

    ``reserved`` names entries the computer SHOWS but does not store -- the Distributions
    window's shipped templates, which its combo lists in place of a stored one of the same
    name. They count as taken (a carried entry under one is added as ``(from project)``, or
    it would be stored and never be pickable) and as present (identical content is not
    added at all). ``None`` as a payload reserves the name only.
    """
    merged = dict(local)
    taken = dict(reserved or {})
    added: list[tuple[str, str]] = []
    for raw_name, payload in carried.items():
        name = str(raw_name).strip()[:NAME_LIMIT]
        if not name:
            continue
        content = _canonical(payload)
        family_base = base_name(name)
        if any(base_name(existing) == family_base and value is not None
               and _canonical(value) == content
               for existing, value in (*taken.items(), *merged.items())):
            continue            # the computer already has this look under this name
        stored = name
        number = 1
        while stored in merged or stored in taken:
            suffix = " (from project)" if number == 1 else f" (from project {number})"
            stored = family_base[:NAME_LIMIT - len(suffix)] + suffix
            number += 1
        merged[stored] = copy.deepcopy(payload)
        added.append((name, stored))
    return merged, added
