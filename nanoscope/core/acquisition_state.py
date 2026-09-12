"""Project-lifetime acquisition overrides, kept in the v11 extensible workspace.

Older v11 applications preserve unknown workspace extras. Match source and channel
identity, not positions; keep unresolved entries so reopening without data is lossless.
"""
from __future__ import annotations

import copy

from .acquisition import AcquisitionBandwidth
from .calibration import channel_identity, resolve_channel
from .identity import same_source
from .segment import persistent_key

WORKSPACE_KEY = "acquisition_bandwidth_overrides_v1"
_MISSING = object()


def capture_acquisition_overrides(session, previous=_MISSING) -> list:
    if previous is not _MISSING and not isinstance(previous, list):
        raise ValueError("Cannot overwrite an unreadable acquisition metadata section.")
    records = copy.deepcopy(previous) if previous is not _MISSING else []

    def replaceable(rec, key, name, kind):
        if not isinstance(rec, dict) or set(rec) != {
                "doc_id", "channel", "channel_name", "channel_kind", "bandwidth"}:
            return False
        try:
            value = AcquisitionBandwidth.from_dict(rec["bandwidth"])
        except (KeyError, TypeError, ValueError):
            return False  # preserve damaged/unknown future evidence verbatim
        return (value.source == "user" and same_source(str(rec.get("doc_id", "")), key)
                and rec.get("channel_name") == name and rec.get("channel_kind") == kind)

    for doc in session.documents():
        key = persistent_key(doc)
        for info in doc.meta.channels:
            name, kind = channel_identity(info)
            records = [r for r in records if not replaceable(r, key, name, kind)]
            override = session.acquisition_bandwidth_override(doc.doc_id, info.index)
            if override is not None:
                records.append({"doc_id": key, "channel": info.index,
                                "channel_name": name, "channel_kind": kind,
                                "bandwidth": override.to_dict()})
    return records


def restore_acquisition_overrides(session, doc_id: str, records, *, match_key=None,
                                  clear_existing=False) -> tuple[str, ...]:
    doc = session.document(doc_id)
    key = match_key if match_key is not None else persistent_key(doc)
    if not isinstance(records, list):
        return ("Acquisition metadata could not be restored: expected a list.",)
    if clear_existing:
        # A segment inherits its parent's live override when re-derived. A project
        # with this section is authoritative even when it records no child override:
        # the child may have explicitly returned to its original file declaration.
        for info in doc.meta.channels:
            session.set_acquisition_bandwidth(doc_id, info.index, None)
    messages = []
    applied = set()
    for rec in records:
        if not isinstance(rec, dict):
            messages.append("Malformed acquisition metadata entry was not applied.")
            continue
        if not same_source(str(rec.get("doc_id", "")), key):
            continue
        try:
            if set(rec) - {"doc_id", "channel", "channel_name", "channel_kind", "bandwidth"}:
                raise ValueError("unsupported acquisition metadata fields")
            if not rec.get("channel_name") or not rec.get("channel_kind"):
                raise ValueError("missing channel identity")
            ch, reason = resolve_channel(doc.meta.channels, rec["channel"],
                                         rec["channel_name"], rec["channel_kind"])
            if ch is None:
                raise ValueError(reason or "channel unavailable")
            if ch in applied:
                raise ValueError("duplicate acquisition override for channel")
            value = AcquisitionBandwidth.from_dict(rec["bandwidth"])
            session.set_acquisition_bandwidth(doc_id, ch, value)
            applied.add(ch)
            if reason:
                messages.append(f"{doc.display_name}: acquisition metadata {reason}")
        except (KeyError, TypeError, ValueError) as exc:
            messages.append(f"{doc.display_name}: acquisition metadata not applied ({exc}).")
    return tuple(messages)
