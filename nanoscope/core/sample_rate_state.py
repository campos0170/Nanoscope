"""Project-lifetime sample-rate declarations, kept in the v11 extensible workspace.

The sibling of :mod:`nanoscope.core.acquisition_state`, with the same contract and for the
same reasons: one versioned key in ``workspace.extra`` (no project VERSION bump — a v11
build preserves unknown extras byte-for-byte), records keyed by the document's PERSISTENT key
and matched with :func:`~nanoscope.core.identity.same_source` (never a ``doc_id``, which
carries a per-load token), unknown or malformed records preserved verbatim and never applied,
and an unreadable section quarantined rather than overwritten. One record per document, not
per channel: a recording has one time axis.
"""
from __future__ import annotations

import copy

from .identity import same_source
from .sample_rate import SampleRateDeclaration
from .segment import SegmentDocument, persistent_key

WORKSPACE_KEY = "sample_rate_declarations_v1"
UNREADABLE_KEY = WORKSPACE_KEY + "_unreadable"
_MISSING = object()
_FIELDS = {"doc_id", "declaration"}


def prepare_sample_rate_workspace(extra: dict) -> tuple[str, ...]:
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
    return ("Sample-rate declarations could not be read. Their original contents are "
            "preserved in the project; new declarations and other edits can still be saved.",)


def capture_sample_rate_declarations(session, previous=_MISSING) -> list:
    """The records to save: every live declaration, merged over ``previous`` so a record for
    a recording that is not open (or one this build cannot read) is kept, not erased."""
    if previous is not _MISSING and not isinstance(previous, list):
        raise ValueError("Cannot overwrite an unreadable sample-rate declarations section.")
    records = copy.deepcopy(previous) if previous is not _MISSING else []

    def replaceable(rec, key, read, superseded):
        if not isinstance(rec, dict) or set(rec) != _FIELDS:
            return False
        try:
            value = SampleRateDeclaration.from_dict(rec["declaration"])
        except (KeyError, TypeError, ValueError):
            return False   # preserve damaged/unknown future evidence verbatim
        if not same_source(str(rec.get("doc_id", "")), key):
            return False
        if superseded:
            return True    # the user's current statement replaces every earlier one
        try:
            value.resolve(read)
        except ValueError:
            # One THIS session could not have applied (the recording now has no time base
            # to re-read) and has not replaced: kept verbatim, as the acquisition precedent
            # keeps a record whose channel is gone.
            return False
        return True

    for doc in session.documents():
        if isinstance(doc, SegmentDocument):
            continue            # a segment follows its parent; it never holds a record
        key = persistent_key(doc)
        read = session.time_base_as_read(doc.doc_id)
        declaration = session.sample_rate_declaration(doc.doc_id)
        records = [r for r in records
                   if not replaceable(r, key, read, superseded=declaration is not None)]
        if declaration is not None:
            records.append({"doc_id": key, "declaration": declaration.to_dict()})
    return records


def restore_sample_rate_declarations(session, doc_id: str, records, *, match_key=None,
                                     clear_existing=False) -> tuple[str, ...]:
    """Apply the saved declaration for ONE just-landed document; messages, never raises.

    ``clear_existing`` is passed when the project HAS this section: it is then authoritative,
    so a document the section records no declaration for returns to the file's own time
    base (a re-derived segment may have inherited its parent's live declaration)."""
    doc = session.document(doc_id)
    if isinstance(doc, SegmentDocument):
        return ()               # re-derived from its parent, whose declaration is already back
    key = match_key if match_key is not None else persistent_key(doc)
    if not isinstance(records, list):
        return ("Sample-rate declarations could not be restored: expected a list.",)
    if clear_existing:
        session.set_sample_rate_declaration(doc_id, None)
    messages = []
    applied = False
    for rec in records:
        if not isinstance(rec, dict):
            messages.append("Malformed sample-rate declaration entry was not applied.")
            continue
        if not same_source(str(rec.get("doc_id", "")), key):
            continue
        try:
            if set(rec) - _FIELDS:
                raise ValueError("unsupported sample-rate declaration fields")
            if applied:
                raise ValueError("duplicate sample-rate declaration for this recording")
            value = SampleRateDeclaration.from_dict(rec["declaration"])
            session.set_sample_rate_declaration(doc_id, value)
            applied = True
        except (KeyError, TypeError, ValueError) as exc:
            messages.append(f"{doc.display_name}: sample-rate declaration not applied ({exc}).")
    return tuple(messages)
