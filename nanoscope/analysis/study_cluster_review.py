"""Frozen cluster memberships used to curate the manual study event selection.

Membership remains usable after acceptance changes, including after reopening.
Source, preparation, geometry and measurement changes retire it. The caller also
retires snapshots on a newly completed detection/measurement run, even if that
run happens to reproduce identical numerical output. Long traces are identified
by the existing measurement hashes, never rehashed by these UI-facing helpers.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import study_review

VERSION = "manual-study-cluster-review-v1"
UNASSIGNED = "__unassigned__"
_CURATION_PROVENANCE = {"accepted_count", "acceptance_source", "cluster_review"}


def _plain(value):
    if value is pd.NA or value is pd.NaT:
        return None
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.generic):
        return _plain(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if all(isinstance(k, str) for k in value):
            return {k: _plain(v) for k, v in value.items()}
        pairs = [[_plain(k), _plain(v)] for k, v in value.items()]
        return {"mapping": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    raise ValueError(f"Unsupported cluster evidence: {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _event_key(value):
    if isinstance(value, str) and value.strip():
        return "string", value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0:
        return "integer", int(value)
    raise ValueError("Cluster event IDs must be nonempty strings or nonnegative integers")


def _record_map(records):
    by_id, by_tag = {}, {}
    for record in records:
        identity, tag = record.get("recording_id"), record.get("tag")
        if (not isinstance(identity, str) or not identity or identity in by_id
                or not isinstance(tag, str) or not tag or tag in by_tag):
            raise ValueError("Cluster review needs unique recording IDs and tags")
        by_id[identity], by_tag[tag] = record, record
    return by_id, by_tag


def _basis(record):
    analysis, frame, _ = study_review._catalog(record)
    order = sorted(range(len(frame)), key=lambda i: _event_key(frame.event_id.iloc[i]))
    frame = frame.iloc[order].reset_index(drop=True).drop(columns=list(study_review.EVENT_REVIEW_COLUMNS), errors="ignore").copy(deep=True)
    frame["accepted"] = False
    provenance = {k: v for k, v in analysis.provenance.items() if k not in _CURATION_PROVENANCE}
    return study_review.review_fingerprint({**record,
        "analysis": replace(analysis, catalog=frame, provenance=provenance)})


def _stored_label(value, method):
    if value is pd.NA or value is None:
        return UNASSIGNED
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Cluster labels cannot be booleans")
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return UNASSIGNED
        # Mixture assignment columns contain integer-valued floats because rows
        # from other recordings are NaN. Validate this representation explicitly.
        if not np.isfinite(value) or value != np.floor(value):
            raise ValueError("Stored cluster labels must be exact finite integers or unassigned")
        value = int(value)
    if not isinstance(value, (int, np.integer)):
        raise ValueError("Stored cluster labels must be integers or unassigned")
    value = int(value)
    if value < (-1 if method == "hdbscan" else 0):
        raise ValueError("Only HDBSCAN may use the noise label -1")
    return value


def _snapshot_digest(snapshot):
    evidence = {k: v for k, v in snapshot.items() if k != "snapshot_id"}
    evidence["memberships"] = snapshot["memberships"].to_dict(orient="records")
    return _digest(evidence)


def create_snapshot(records, assignments, metadata, *, kind, settings) -> dict:
    """Freeze pooled K-means/HDBSCAN or target-only hard mixture membership.

    The domain also includes candidates absent from the feature/assignment table;
    each method puts those events in the explicit ``UNASSIGNED`` bucket. No event
    decisions are changed. Snapshot dictionaries/DataFrames use the existing
    workspace codec and contain no signal arrays or executable estimators.
    """
    records = list(records)
    by_id, by_tag = _record_map(records)
    if not isinstance(assignments, pd.DataFrame) or not assignments.columns.is_unique:
        raise ValueError("Cluster assignments must be a table with unique columns")
    if not isinstance(metadata, Mapping) or kind not in ("pooled", "mixture"):
        raise ValueError("Choose pooled clustering or a selected-recording mixture")
    required = {"tag", "event_id", "start_idx", "end_idx"}
    if not required.issubset(assignments.columns):
        raise ValueError("Cluster assignments require source tags, event IDs and sample bounds")
    if kind == "mixture":
        target = metadata.get("mixture_tag")
        if target not in by_tag or "subpop" not in assignments:
            raise ValueError("Mixture review requires hard assignments and the selected recording")
        domain, methods = [by_tag[target]], ["subpop"]
    else:
        domain = [r for r in records if not r.get("comparison_only")]
        methods = []
        kmeans = metadata.get("kmeans_column")
        if kmeans is None:
            candidates = [c for c in assignments if re.fullmatch(r"kmeans[1-9][0-9]*", str(c))]
            if len(candidates) == 1:
                kmeans = candidates[0]
        if kmeans is not None:
            if not isinstance(kmeans, str) or not re.fullmatch(r"kmeans[1-9][0-9]*", kmeans) or kmeans not in assignments:
                raise ValueError("The saved K-means method does not match its assignments")
            methods.append(kmeans)
        if "hdbscan" in assignments:
            methods.append("hdbscan")
        if not methods:
            raise ValueError("This result has no fitted pooled cluster labels")
    if not domain:
        raise ValueError("Select at least one recording for cluster review")
    domain_ids = {r["recording_id"] for r in domain}
    catalogs, source_records = {}, []
    for record in records:
        if "analysis" not in record:
            if record["recording_id"] in domain_ids:
                raise ValueError("Measure each cluster-review recording first")
            continue
        _, frame, _ = study_review._catalog(record)
        catalogs[record["tag"]] = {_event_key(v): i for i, v in enumerate(frame.event_id)}
    for record in domain:
        source_records.append({"recording_id": record["recording_id"], "tag": record["tag"],
                               "basis_fingerprint": _basis(record)})
    seen, assigned = set(), {}
    for values in assignments.to_dict(orient="records"):
        tag = values["tag"]
        if not isinstance(tag, str) or tag not in catalogs:
            raise ValueError("Cluster assignments refer to a foreign recording")
        key = (by_tag[tag]["recording_id"], _event_key(values["event_id"]))
        if key in seen:
            raise ValueError("Cluster assignments contain duplicate event identities")
        seen.add(key)
        index = catalogs[tag].get(key[1])
        if index is None:
            raise ValueError("Cluster assignments contain a foreign event ID")
        event = by_tag[tag]["analysis"].catalog.iloc[index]
        for column in ("start_idx", "end_idx"):
            value = values[column]
            if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                    or value != event[column]):
                raise ValueError("Cluster assignment bounds must match exact integer event bounds")
        labels = {method: _stored_label(values[method], method) for method in methods}
        if key[0] not in domain_ids:
            if kind != "mixture" or any(v != UNASSIGNED for v in labels.values()):
                raise ValueError("Cluster labels cannot assign events outside the review domain")
            continue
        assigned[key] = labels
    memberships = []
    for record in sorted(domain, key=lambda r: r["recording_id"]):
        frame = record["analysis"].catalog
        for index in sorted(range(len(frame)), key=lambda i: _event_key(frame.event_id.iloc[i])):
            event = frame.iloc[index]
            key = (record["recording_id"], _event_key(event.event_id))
            memberships.append({"recording_id": record["recording_id"], "tag": record["tag"],
                "event_id": event.event_id, "start_idx": int(event.start_idx), "end_idx": int(event.end_idx),
                **assigned.get(key, {method: UNASSIGNED for method in methods})})
    columns = ["recording_id", "tag", "event_id", "start_idx", "end_idx", *methods]
    snapshot = {"version": VERSION, "kind": kind, "methods": methods,
        "created_at": datetime.now(timezone.utc).isoformat(), "settings": _plain(settings),
        "metadata": _plain(metadata), "records": source_records,
        "recording_ids": [r["recording_id"] for r in source_records],
        "memberships": pd.DataFrame(memberships, columns=columns)}
    snapshot["snapshot_id"] = _snapshot_digest(snapshot)
    return snapshot


def validate_snapshot(snapshot) -> None:
    """Validate saved structure/integrity without requiring current live sources."""
    if not isinstance(snapshot, dict) or snapshot.get("version") != VERSION:
        raise ValueError("Choose a saved cluster review snapshot")
    try:
        methods, sources, table = snapshot["methods"], snapshot["records"], snapshot["memberships"]
        if (snapshot["kind"] not in ("pooled", "mixture") or not isinstance(methods, list)
                or not methods or len(set(methods)) != len(methods)
                or any(not isinstance(m, str) or not (m in ("hdbscan", "subpop") or re.fullmatch(r"kmeans[1-9][0-9]*", m)) for m in methods)
                or (snapshot["kind"] == "mixture" and methods != ["subpop"])
                or (snapshot["kind"] == "pooled" and "subpop" in methods)):
            raise ValueError("Invalid saved cluster methods")
        if not isinstance(sources, list) or not sources:
            raise ValueError("Cluster review requires its source domain")
        domain = {}
        for source in sources:
            identity, tag, fingerprint = source["recording_id"], source["tag"], source["basis_fingerprint"]
            if (not isinstance(identity, str) or not identity or identity in domain
                    or not isinstance(tag, str) or not tag or tag in domain.values()
                    or not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)):
                raise ValueError("Invalid saved cluster source identities")
            domain[identity] = tag
        if snapshot["recording_ids"] != list(domain):
            raise ValueError("Invalid saved cluster domain")
        required = {"recording_id", "tag", "event_id", "start_idx", "end_idx", *methods}
        if not isinstance(table, pd.DataFrame) or not table.columns.is_unique or not required.issubset(table.columns):
            raise ValueError("Invalid frozen membership table")
        seen = set()
        for values in table.to_dict(orient="records"):
            identity = values["recording_id"]
            key = identity, _event_key(values["event_id"])
            if identity not in domain or domain[identity] != values["tag"] or key in seen:
                raise ValueError("Invalid or duplicate frozen member identity")
            seen.add(key)
            for name in ("start_idx", "end_idx"):
                value = values[name]
                if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
                    raise ValueError("Invalid frozen event bounds")
            if values["end_idx"] < values["start_idx"]:
                raise ValueError("Invalid frozen event bounds")
            for method in methods:
                value = values[method]
                if isinstance(value, str) and value == UNASSIGNED:
                    continue
                if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < (-1 if method == "hdbscan" else 0):
                    raise ValueError("Invalid frozen cluster label")
        if snapshot.get("snapshot_id") != _snapshot_digest(snapshot):
            raise ValueError("Frozen cluster membership was changed")
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Invalid cluster review snapshot") from exc


def require_current(snapshot, records) -> None:
    """Refuse an entire operation when any participating membership is stale."""
    validate_snapshot(snapshot)
    try:
        by_id, _ = _record_map(records)
        for source in snapshot["records"]:
            record = by_id.get(source["recording_id"])
            if record is None or source["basis_fingerprint"] != _basis(record):
                raise ValueError("Cluster membership is stale for " + source["tag"] +
                    "; rerun clustering or the mixture after the source/event changes")
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Invalid or stale cluster review snapshot") from exc


def current(snapshot, records) -> bool:
    try:
        require_current(snapshot, records)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _labels(snapshot, method, labels):
    if method not in snapshot["methods"]:
        raise ValueError("Choose a method from this saved clustering")
    result = []
    for label in labels:
        if label == UNASSIGNED and isinstance(label, str):
            value = UNASSIGNED
        elif isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_)):
            value = int(label)
        else:
            raise ValueError("Selected cluster labels must be integers or the explicit unassigned bucket")
        if value in result:
            raise ValueError("Select each cluster only once")
        result.append(value)
    available = set(snapshot["memberships"][method]) | {UNASSIGNED}
    if not result or any(label not in available for label in result):
        raise ValueError("Select an existing cluster or the unassigned bucket")
    return result


def members(snapshot, records, method, labels) -> dict:
    """Return every matching event, independent of display filters or sampling."""
    records = list(records)
    require_current(snapshot, records)
    labels = _labels(snapshot, method, labels)
    table = snapshot["memberships"]
    selected = table.loc[table[method].isin(labels)]
    return {str(identity): group.event_id.tolist() for identity, group in selected.groupby("recording_id", sort=False)}


def summaries(snapshot, records, method) -> pd.DataFrame:
    records = list(records)
    require_current(snapshot, records)
    if method not in snapshot["methods"]:
        raise ValueError("Choose a method from this saved clustering")
    by_id, _ = _record_map(records)
    decisions = {(identity, _event_key(row.event_id)): bool(row.accepted)
        for identity, record in by_id.items() if identity in snapshot["recording_ids"]
        for row in record["analysis"].catalog.itertuples()}
    table, output = snapshot["memberships"], []
    labels = sorted(v for v in set(table[method]) if v != UNASSIGNED) + [UNASSIGNED]
    for label in labels:
        group = table.loc[table[method] == label]
        accepted = sum(decisions[(r.recording_id, _event_key(r.event_id))] for r in group.itertuples())
        n = len(group)
        name = "Unassigned / no waveform features" if label == UNASSIGNED else "Noise (-1)" if label == -1 else f"Cluster {label}"
        output.append({"label": label, "name": name, "n_total": n, "n_accepted": accepted,
            "n_rejected": n-accepted, "status": "empty" if not n else "accepted" if accepted == n else "rejected" if not accepted else "mixed"})
    return pd.DataFrame(output)


def apply_decision(snapshot, records, method, labels, accepted, only=False):
    """Return copied rows plus an audit action; never approve events implicitly.

    Ordinary actions affect selected members only. ``only=True`` accepts selected
    members and rejects every other candidate in the frozen domain, including
    unassigned candidates unless that bucket is selected. The latest explicit
    individual or cluster gesture wins. Changed rows include before/after choices
    in the returned action, allowing the caller to persist an exact undo record.
    """
    if not isinstance(accepted, (bool, np.bool_)) or not isinstance(only, bool):
        raise ValueError("Cluster acceptance and keep-only flags must be booleans")
    if only and not accepted:
        raise ValueError("Keep only selected clusters requires an accept decision")
    records = list(records)
    labels = list(labels)
    selection = members(snapshot, records, method, labels)
    selected = {identity: {_event_key(v) for v in ids} for identity, ids in selection.items()}
    domain = {r["recording_id"] for r in snapshot["records"]}
    action = {"version": VERSION, "snapshot_id": snapshot["snapshot_id"], "kind": snapshot["kind"],
        "method": method, "labels": _labels(snapshot, method, labels), "accepted": bool(accepted),
        "only": only, "completed_at": datetime.now(timezone.utc).isoformat(), "changes": [],
        "target_count": len(snapshot["memberships"]) if only else sum(map(len, selection.values()))}
    output = [dict(record) for record in records]
    for record in output:
        identity = record["recording_id"]
        if identity not in domain:
            continue
        analysis = record["analysis"]
        frame = analysis.catalog.copy(deep=True)
        mask = np.array([_event_key(v) in selected.get(identity, set()) for v in frame.event_id], dtype=bool)
        before = frame.accepted.to_numpy(dtype=bool)
        after = mask if only else np.where(mask, bool(accepted), before)
        changed = np.flatnonzero(before != after)
        reviewed = np.ones(len(frame), dtype=bool) if only else mask
        if not reviewed.any():
            continue
        for index in changed:
            action["changes"].append({"recording_id": identity, "event_id": frame.event_id.iloc[index],
                "before": bool(before[index]), "after": bool(after[index])})
        frame["accepted"] = after
        evidence = study_review.event_reviews(analysis)
        evidence.iloc[np.flatnonzero(reviewed)] = study_review.REVIEW_DONE
        frame[study_review.EVENT_REVIEW_COL] = evidence
        conflicts = study_review.event_review_conflicts(analysis)
        if conflicts is not None:
            conflicts.iloc[np.flatnonzero(reviewed)] = ""
            frame[study_review.EVENT_REVIEW_CONFLICT_COL] = conflicts
        provenance = analysis.provenance
        if len(changed):
            provenance = {**provenance, "accepted_count": int(after.sum()), "acceptance_source": "Explicit cluster review",
                "cluster_review": {key: deepcopy(value) for key, value in action.items() if key != "changes"}}
            record.pop(study_review.REVIEW_KEY, None)
        record["analysis"] = replace(analysis, catalog=frame, provenance=provenance)
    action["changed_count"] = len(action["changes"])
    return output, _plain(action)
