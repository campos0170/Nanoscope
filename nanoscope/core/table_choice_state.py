"""Which event table each channel files hand-made events into, kept in the v11 workspace.

The sibling of :mod:`nanoscope.core.sample_rate_state`, with the same contract: one versioned key
in ``workspace.extra`` (no project VERSION bump, because a v11 build preserves unknown extras
byte-for-byte), records keyed by the document's PERSISTENT key and matched with
:func:`~nanoscope.core.identity.same_source` (never a ``doc_id``, which carries a per-load
token), unknown or malformed records preserved verbatim and never applied, and an unreadable
section quarantined rather than overwritten.

A record is ``{"doc_id": <persistent key>, "channel": <index>, "channel_name": <name or None>,
"module_id": "detect.manual"}``. The panel owns the live choice; this module only moves it in and
out of a project.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from .identity import same_source

WORKSPACE_KEY = "event_table_choices_v1"
UNREADABLE_KEY = WORKSPACE_KEY + "_unreadable"
_FIELDS = {"doc_id", "channel", "channel_name", "module_id"}


def prepare_table_choice_workspace(extra: dict) -> tuple[str, ...]:
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
    return ("The event-table choices could not be read. Their original contents are preserved "
            "in the project; new choices and other edits can still be saved.",)


def capture_table_choices(live: list[dict[str, Any]], previous: Any = None) -> list[dict[str, Any]]:
    """The records to save: every live choice, merged over ``previous`` so a record for a
    recording that is not open (or one this build cannot read) is kept, not erased.

    ``live`` is a list of record dicts (``doc_id`` already the persistent key)."""
    if previous is not None and not isinstance(previous, list):
        raise ValueError("Cannot overwrite an unreadable event-table choices section.")
    records = copy.deepcopy(previous) if previous else []

    def superseded(rec: Any) -> bool:
        if not isinstance(rec, dict) or set(rec) != _FIELDS:
            return False                    # unknown or damaged: kept verbatim
        return any(same_source(str(rec["doc_id"]), str(new["doc_id"]))
                   and rec.get("channel") == new.get("channel") for new in live)

    return [r for r in records if not superseded(r)] + [dict(r) for r in live]


def restore_table_choices(
    records: Any,
    persistent_key: str,
    apply: Callable[[int, str], None],
    channel_name_of: Callable[[int], str | None],
) -> tuple[str, ...]:
    """Apply the saved choices for ONE just-landed document; messages, never raises.

    ``apply(channel, module_id)`` records a choice (the panel's); ``channel_name_of(channel)``
    names the channel in this load so a record whose channel was renamed or re-ordered is not
    applied to the wrong trace."""
    if not isinstance(records, list):
        return ("Event-table choices could not be restored: expected a list.",)
    messages: list[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            messages.append("Malformed event-table choice entry was not applied.")
            continue
        if not same_source(str(rec.get("doc_id", "")), persistent_key):
            continue
        try:
            if set(rec) - _FIELDS:
                raise ValueError("unsupported fields")
            channel = int(rec["channel"])
            module_id = str(rec["module_id"])
            saved_name = rec.get("channel_name")
            if saved_name is not None and channel_name_of(channel) not in (None, saved_name):
                raise ValueError("the channel is not the one the choice was made for")
            apply(channel, module_id)
        except (KeyError, TypeError, ValueError) as exc:
            messages.append(f"An event-table choice was not applied ({exc}).")
    return tuple(messages)
