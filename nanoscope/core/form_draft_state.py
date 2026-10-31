"""The analysis forms' un-run edits, one per event table, kept in the v11 workspace (design §5.2
and §8, the owner's decision 4).

The sibling of :mod:`nanoscope.core.table_choice_state`, with the same contract: one versioned key
in ``workspace.extra`` (no project VERSION bump, because a v11 build preserves unknown extras
byte-for-byte), records keyed by the document's PERSISTENT key and matched with
:func:`~nanoscope.core.identity.same_source` (never a ``doc_id``, which carries a per-load
token), unknown or malformed records preserved verbatim and never applied, and an unreadable
section quarantined rather than overwritten.

A record is ``{"doc_id": <persistent key>, "channel": <index>, "channel_name": <name or None>,
"module_id": <the table's module id, or None for a channel with no table yet>, "state": <the
form, as AnalysisPanel.form_state() gives it>}``. The panel owns the live drafts; this module only
moves them in and out of a project.

ONE RULE DIFFERS FROM THE TABLE CHOICES, and it is why this is not a copy of that module. A
channel's table choice outlives the session's interest in it, so a saved choice is replaced only
by a live one. A draft does not: it is DROPPED when its form produces a run, or when the user
puts the form back, and that absence is information. Merging the live drafts over the previous
save the way the choices are merged kept every draft the session had since run away, and the
next reopen put a form back that its own run had already settled. So the live session is the
authority for exactly the channels it holds (:func:`capture_form_drafts`' ``held``): every
readable record about one of them is replaced by the live drafts, present or not, and a record
about a recording that is not open, or a channel that did not come back as itself, is kept.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any

from .identity import same_source

WORKSPACE_KEY = "analysis_form_drafts_v1"
UNREADABLE_KEY = WORKSPACE_KEY + "_unreadable"
_FIELDS = {"doc_id", "channel", "channel_name", "module_id", "state"}
#: The parts of a form state this build can put back (``AnalysisPanel.form_state``). A state
#: naming any other part was written by a build that knows more than this one, so its draft
#: is kept for that build and never half-applied by this one.
_PARTS = frozenset({"detection", "intents", "ml", "baseline"})


def prepare_form_draft_workspace(extra: dict) -> tuple[str, ...]:
    """Quarantine an unreadable section without losing it or blocking future saves."""
    if WORKSPACE_KEY not in extra or isinstance(extra[WORKSPACE_KEY], list):
        return ()
    key = UNREADABLE_KEY
    suffix = 1
    while key in extra:
        key = f"{UNREADABLE_KEY}_{suffix}"
        suffix += 1
    extra[key] = extra[WORKSPACE_KEY]
    extra[WORKSPACE_KEY] = []
    return ("The unsaved analysis-form edits could not be read. Their original contents are "
            "preserved in the project; new edits and everything else can still be saved.",)


def _readable(rec: Any) -> tuple[str, int, str | None, str | None, dict] | None:
    """``(doc_id, channel, channel_name, module_id, state)`` for a record this build can apply,
    or ``None`` for one it cannot: not a dict, a field more or less than :data:`_FIELDS`, a
    value of the wrong type, or a state with a part this build does not know."""
    if not isinstance(rec, dict) or set(rec) != _FIELDS:
        return None
    doc_id, channel = rec["doc_id"], rec["channel"]
    name, module_id, state = rec["channel_name"], rec["module_id"], rec["state"]
    if not isinstance(doc_id, str) or not doc_id:
        return None
    if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
        return None
    if name is not None and not isinstance(name, str):
        return None
    if module_id is not None and (not isinstance(module_id, str) or not module_id):
        return None
    if (not isinstance(state, dict) or not state or not set(state) <= _PARTS
            or not all(isinstance(part, dict) for part in state.values())):
        return None
    return doc_id, channel, name, module_id, state


def _applies(rec: tuple, persistent_key: str, channels: Mapping[int, str | None]) -> bool:
    """Would the restore put this readable record on that document? Its recording, a channel
    this load has, and that channel under the name the draft was made for (a record that
    names none predates the check and goes by its index)."""
    doc_id, channel, saved_name, _module_id, _state = rec
    if not same_source(doc_id, persistent_key) or channel not in channels:
        return False
    return saved_name is None or channels[channel] in (None, saved_name)


def capture_form_drafts(live: list[dict[str, Any]], previous: Any = None,
                        held: Iterable[tuple[str, Mapping[int, str | None]]] = ()
                        ) -> list[dict[str, Any]]:
    """The records to save: every live draft, and every previous record the live session cannot
    speak for.

    ``live`` is a list of record dicts (``doc_id`` already the persistent key). ``held`` is
    ``(persistent key, {channel index: channel name})`` for every open document: a previous
    record that the restore would apply to one of those channels is the live session's to
    replace, so it goes, whether or not a live draft takes its place (see the module
    docstring). An unknown or damaged record is kept verbatim, as is a record about a
    recording that is not open."""
    if previous is not None and not isinstance(previous, list):
        raise ValueError("Cannot overwrite an unreadable analysis-form drafts section.")
    held = [(str(key), dict(channels)) for key, channels in held]

    def superseded(rec: Any) -> bool:
        parsed = _readable(rec)
        return parsed is not None and any(_applies(parsed, key, channels)
                                          for key, channels in held)

    kept = [copy.deepcopy(r) for r in (previous or []) if not superseded(r)]
    return kept + [copy.deepcopy(dict(r)) for r in live]


def restore_form_drafts(records: Any, persistent_key: str,
                        channels: Mapping[int, str | None]
                        ) -> tuple[dict[tuple[int, str | None], dict], tuple[str, ...]]:
    """The saved drafts for ONE just-landed document, and what could not be applied; never raises.

    Returns ``({(channel, module_id): state}, messages)``. ``channels`` maps each channel index
    of this load to its name, so a record whose channel was renamed or re-ordered is not put on
    the wrong trace, and one naming a channel this load lacks is not put anywhere. Neither is
    an unknown or damaged record: each is said once in ``messages`` and left in the project."""
    if not isinstance(records, list):
        return {}, ("The unsaved analysis-form edits could not be restored: expected a list.",)
    drafts: dict[tuple[int, str | None], dict] = {}
    messages: list[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            messages.append("A malformed analysis-form draft was not applied.")
            continue
        if not same_source(str(rec.get("doc_id", "")), persistent_key):
            continue
        parsed = _readable(rec)
        if parsed is None:
            messages.append("An analysis-form draft was not applied (its fields are not ones "
                            "this version can read).")
            continue
        if not _applies(parsed, persistent_key, channels):
            messages.append("An analysis-form draft was not applied (the channel is not the "
                            "one it was made for).")
            continue
        _doc_id, channel, _name, module_id, state = parsed
        if (channel, module_id) in drafts:
            # The first one wins, as a duplicate sample-rate declaration does: a later
            # record silently overwriting it would make the file's order decide the form.
            messages.append("A duplicate analysis-form draft for one table was not applied.")
            continue
        drafts[(channel, module_id)] = copy.deepcopy(state)
    return drafts, tuple(messages)
