"""Explicit approval of a manual study's measured, curated event snapshot.

The receipt is detached JSON data, suitable for the normal workspace codec.
It certifies the saved event state, not that every row was personally inspected.
Prepared samples are identified by the hashes already computed by measurement
and by preparation provenance. UI status checks deliberately do not rehash long
traces. The caller must invalidate approval when it replaces a scientific stage;
unsupported in-place mutation of sample arrays requires fresh measurement.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .incubation_detection import RecordingAnalysis
from .events import (REVIEW_DONE, REVIEW_NONE, REVIEW_VALUES, event_status, STATUS_REJECTED,
                     STUDY_REVIEW_POLICY_COL)

REVIEW_KEY = "review_receipt"
REVIEW_VERSION = "manual-study-review-v1"
EVENT_REVIEW_COL = "event_review"
EVENT_REVIEW_CONFLICT_COL = "event_review_conflict"
EVENT_REVIEW_COLUMNS = (EVENT_REVIEW_COL, EVENT_REVIEW_CONFLICT_COL)


def event_reviews(analysis) -> pd.Series:
    """Current per-event reviewedness, independent of whole-study approval.

    Explicit study annotations take precedence over archived native review. Older
    snapshots retain recorded native attention; only a changed decision proves a
    later study gesture. A different detector's automatic gate is not proof that
    its imported events were reviewed by a person.
    """
    frame = analysis.catalog
    if EVENT_REVIEW_COL in frame:
        values = frame[EVENT_REVIEW_COL].astype("string")
        if not values.dropna().isin(REVIEW_VALUES).all():
            raise ValueError("Study event review must be reviewed, unreviewed, or unknown")
        return values.copy(deep=True)
    if analysis.provenance.get("acceptance_source") == "Automatic study gate restored":
        return pd.Series(REVIEW_NONE, index=frame.index, dtype="string")
    imported = "source_event_id" in frame
    if "source_review" in frame:
        values = frame.source_review.astype("string").str.strip()
        values = values.where(values.isin(REVIEW_VALUES), pd.NA)
    else:
        values = pd.Series(pd.NA if imported else REVIEW_NONE, index=frame.index, dtype="string")
    if imported and "source_accepted" in frame:
        # Respect the native legacy rule for bare human rejections, while the
        # scoped policy distinguishes untouched study-gate negatives on return.
        source = pd.DataFrame({"accepted": frame.source_accepted, "review": values})
        if "source_flag" in frame:
            source["flag"] = frame.source_flag
        if "source_" + STUDY_REVIEW_POLICY_COL in frame:
            source[STUDY_REVIEW_POLICY_COL] = frame["source_" + STUDY_REVIEW_POLICY_COL]
        known = source.accepted.notna()
        positions = np.flatnonzero(known.to_numpy())
        inferred = (event_status(source.iloc[positions]) == STATUS_REJECTED).to_numpy()
        values.iloc[positions[inferred]] = REVIEW_DONE
    original = "source_accepted" if imported else "passes_gate"
    if "accepted" in frame and original in frame:
        # Missing archived decisions cannot establish a change.
        changed = frame[original].notna() & frame.accepted.ne(frame[original]).fillna(False)
        values.loc[changed] = REVIEW_DONE
    return values


def event_review_conflicts(analysis):
    """Current unresolved native label conflicts, or None when never recorded."""
    frame = analysis.catalog
    for name in (EVENT_REVIEW_CONFLICT_COL, "source_consensus_curation_conflict"):
        if name in frame:
            return frame[name].astype("string").fillna("").copy(deep=True)
    return None


def _event_identity(value):
    if isinstance(value, str) and value.strip():
        return "string", value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0:
        return "integer", int(value)
    raise ValueError("Event identities must be nonempty strings or nonnegative integers")


def apply_event_decision(analysis, ids, value):
    """Copy an explicit individual decision, including an acceptance no-op.

    ``None`` restores automatic acceptance and clears current review marks. Prior
    native annotations remain archived in source_* columns. Attention alone does
    not change numerical provenance or invalidate completed scientific results.
    """
    if value is not None and not isinstance(value, (bool, np.bool_)):
        raise ValueError("Event acceptance must be boolean or an automatic reset")
    _, original, _ = _catalog({"tag": analysis.tag, "analysis": analysis})
    frame = original.copy(deep=True)
    reviews = event_reviews(analysis)
    conflicts = event_review_conflicts(analysis)
    if value is None:
        if "passes_gate" not in frame or any(not isinstance(v, (bool, np.bool_)) for v in frame.passes_gate):
            raise ValueError("Automatic reset requires explicit gate decisions")
        after = frame.passes_gate.to_numpy(dtype=bool)
        reviews[:] = REVIEW_NONE
        if "source_consensus_curation_conflict" in frame:
            conflicts = frame.source_consensus_curation_conflict.astype("string").fillna("").copy(deep=True)
    else:
        wanted = {_event_identity(v) for v in ids}
        existing = [_event_identity(v) for v in frame.event_id]
        if not wanted or not wanted.issubset(set(existing)):
            raise ValueError("Select existing event identities for review")
        selected = np.array([key in wanted for key in existing], dtype=bool)
        after = frame.accepted.to_numpy(dtype=bool, copy=True)
        after[selected] = bool(value)
        reviews.iloc[np.flatnonzero(selected)] = REVIEW_DONE
        if conflicts is not None:
            conflicts.iloc[np.flatnonzero(selected)] = ""
    changed = not np.array_equal(after, frame.accepted.to_numpy(dtype=bool))
    frame["accepted"] = after
    frame[EVENT_REVIEW_COL] = reviews
    if conflicts is not None:
        frame[EVENT_REVIEW_CONFLICT_COL] = conflicts
    provenance = analysis.provenance
    if changed:
        provenance = {**provenance, "accepted_count": int(after.sum()),
            "acceptance_source": "Automatic study gate restored" if value is None else "Manual review in study workspace"}
    return replace(analysis, catalog=frame, provenance=provenance)


def retain_event_decisions(previous, measured):
    """Retain current decisions and attention during imported-event remeasurement.

    Match exact source IDs and inclusive bounds, never row positions. A new
    native import deliberately bypasses this helper because its latest gestures
    and edited geometry are authoritative.
    """
    if (not isinstance(previous.provenance.get("recording_identity"), dict)
            or previous.tag != measured.tag or previous.fs_hz != measured.fs_hz
            or previous.duration_s != measured.duration_s
            or previous.provenance.get("recording_identity") != measured.provenance.get("recording_identity")):
        raise ValueError("Cannot retain event decisions on a different recording")
    _catalog({"tag": previous.tag, "analysis": previous})
    _catalog({"tag": measured.tag, "analysis": measured})
    before, after = previous.catalog, measured.catalog.copy(deep=True)
    if "source_event_id" not in before or "source_event_id" not in after:
        raise ValueError("Remeasurement review requires native source event identities")
    old_keys = [_event_identity(v) for v in before.source_event_id]
    new_keys = [_event_identity(v) for v in after.source_event_id]
    if len(set(old_keys)) != len(old_keys) or len(set(new_keys)) != len(new_keys) or set(old_keys) != set(new_keys):
        raise ValueError("Remeasurement must retain the same source event identities")
    locations = {key: i for i, key in enumerate(old_keys)}
    order = [locations[key] for key in new_keys]
    prior = before.iloc[order]
    if not np.array_equal(prior[["start_idx", "end_idx"]].to_numpy(), after[["start_idx", "end_idx"]].to_numpy()):
        raise ValueError("Cannot retain event review after event bounds changed")
    after["accepted"] = prior.accepted.to_numpy(dtype=bool)
    after[EVENT_REVIEW_COL] = event_reviews(previous).iloc[order].to_numpy()
    after[EVENT_REVIEW_COL] = after[EVENT_REVIEW_COL].astype("string")
    conflicts = event_review_conflicts(previous)
    if conflicts is not None:
        after[EVENT_REVIEW_CONFLICT_COL] = pd.array(conflicts.iloc[order].to_numpy(), dtype="string")
    return replace(measured, catalog=after, provenance={**measured.provenance,
        "accepted_count": int(after.accepted.sum()),
        "acceptance_source": "Saved imported detections with subsequent study review retained during remeasurement"})

# Applied scientific settings only: draft controls, figures and display styling
# do not change the reviewed measurements. The GUI separately blocks unapplied
# scientific drafts before confirming or starting a dependent stage.
_SCIENTIFIC_KEYS = (
    "source_path", "source_info", "channel", "event_polarity", "reader_scaling",
    "incubation_h", "pore_id", "pore_diameter_nm", "concentration_pm", "voltage_mv",
    "add_channel", "add_channel_as_pa", "keep_start_s", "keep_stop_s", "exclude_s",
    "median_engine", "preparation_settings", "model_path", "inference_settings",
    "decode_settings", "measurement_settings", "schmitt_settings", "event_input",
    "measurement_method", "imported_event_provenance",
)


def _canonical(value):
    """Stable scalar/container representation across a workspace round trip."""
    if value is pd.NA or value is pd.NaT:
        return {"missing": True}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": str(value)}
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Review provenance must use string dictionary keys")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"array": _canonical(value.tolist()), "dtype": value.dtype.str,
                "shape": list(value.shape)}
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return {type(value).__name__: str(value)}
    raise ValueError(f"Unsupported value in review evidence: {type(value).__name__}")


def _descriptor(recording):
    raw, signal = np.asarray(recording.raw), np.asarray(recording.detrended)
    if (raw.ndim != 1 or signal.ndim != 1 or not len(raw)
            or len(raw) != len(signal)):
        raise ValueError("Review requires equally sized, nonempty prepared current arrays")
    if isinstance(recording.fs_hz, (bool, np.bool_)):
        raise ValueError("Review requires a positive recording sample rate")
    fs, duration = float(recording.fs_hz), float(recording.duration_s)
    if (not np.isfinite(fs) or fs <= 0 or not np.isfinite(duration)
            or not np.isclose(duration, len(raw) / fs, rtol=1e-12, atol=1e-12)):
        raise ValueError("Review requires a valid sample rate and complete recording duration")
    return {"tag": recording.tag, "n_samples": len(raw), "fs_hz": fs,
            "duration_s": duration}


def _catalog_hash(frame):
    """Hash small measurement columns efficiently, including exact float values."""
    digest = hashlib.sha256()
    for name in sorted(frame.columns):
        values = frame[name].to_numpy()
        header = json.dumps([name, str(frame[name].dtype), len(values)]).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        if values.dtype.hasobject:
            data = json.dumps(_canonical(values.tolist()), sort_keys=True,
                              separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
        else:
            data = np.ascontiguousarray(values).tobytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _catalog(row):
    if not isinstance(row, Mapping):
        raise ValueError("Review requires a study recording")
    tag, analysis = row.get("tag"), row.get("analysis")
    if (not isinstance(tag, str) or not tag.strip()
            or not isinstance(analysis, RecordingAnalysis) or analysis.tag != tag):
        raise ValueError("Measure this recording's candidate events before confirming review")
    frame = analysis.catalog
    required = {"event_id", "start_idx", "end_idx", "accepted"}
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or any(not isinstance(name, str) for name in frame.columns)
            or not required.issubset(frame.columns)):
        raise ValueError("Review requires event identities, sample bounds and acceptance decisions")
    ids = frame.event_id
    valid_id = lambda value: ((isinstance(value, str) and bool(value.strip())) or
                             (isinstance(value, (int, np.integer)) and
                              not isinstance(value, (bool, np.bool_)) and value >= 0))
    if not all(valid_id(value) for value in ids) or ids.duplicated().any():
        raise ValueError("Reviewed event identities must be nonempty and unique")
    if any(not isinstance(value, (bool, np.bool_)) for value in frame.accepted):
        raise ValueError("Reviewed acceptance decisions must be explicit booleans")
    bounds = []
    for name in ("start_idx", "end_idx"):
        values = frame[name]
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
               or value < 0 or value > np.iinfo(np.int64).max for value in values):
            raise ValueError("Reviewed event bounds must be nonnegative integer sample indices")
        bounds.append(values.to_numpy(dtype=np.int64))
    starts, ends = bounds
    geometry = _descriptor(analysis)
    order = np.argsort(starts, kind="stable")
    if (np.any(ends < starts) or np.any(ends >= geometry["n_samples"])
            or (len(order) > 1 and np.any(starts[order][1:] <= ends[order][:-1]))):
        raise ValueError("Reviewed events must be non-overlapping inclusive intervals within the recording")
    if not isinstance(analysis.provenance, dict):
        raise ValueError("Review requires measurement provenance")
    identity = analysis.provenance.get("recording_identity")
    if isinstance(identity, dict):
        for key, value in geometry.items():
            if key in identity and identity[key] != value:
                raise ValueError("Measured event identity does not match the recording")
    if row.get("prepared") is not None and _descriptor(row["prepared"]) != geometry:
        raise ValueError("Prepared recording changed; measure its events again before review")
    return analysis, frame, geometry


def review_fingerprint(row) -> str:
    """Hash event decisions, measurements and their applied scientific context.

    All candidate rows, including rejected events, participate. Display indexes
    are omitted; event IDs and actual catalog row order retain their identities.
    The existing measurement provenance carries full prepared-sample hashes.
    """
    analysis, frame, geometry = _catalog(row)
    evidence = {
        "version": REVIEW_VERSION, "recording_id": row.get("recording_id"),
        "recording": geometry, "measurement_provenance": analysis.provenance,
        "settings": {key: row[key] for key in _SCIENTIFIC_KEYS if key in row},
        # Approval is of the scientific selection. Recording an explicit no-op
        # review gesture does not invalidate an otherwise identical selection.
        "catalog_sha256": _catalog_hash(frame.drop(columns=list(EVENT_REVIEW_COLUMNS), errors="ignore")),
        "catalog_provenance": frame.attrs,
    }
    prepared = row.get("prepared")
    if prepared is not None:
        evidence["preparation"] = {"metadata": prepared.metadata,
                                   "provenance": prepared.provenance}
    encoded = json.dumps(_canonical(evidence), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def confirm_review(row) -> dict:
    """Return a detached approval receipt; the caller explicitly saves it."""
    fingerprint = review_fingerprint(row)
    frame = row["analysis"].catalog
    return {"version": REVIEW_VERSION, "fingerprint": fingerprint, "tag": row["tag"],
            "recording_id": row.get("recording_id"), "event_count": len(frame),
            "accepted_count": int(frame.accepted.sum()),
            "confirmed_at": datetime.now(timezone.utc).isoformat()}


def review_is_current(row) -> bool:
    """Missing legacy receipts or changed/invalid evidence always need review."""
    if not isinstance(row, Mapping):
        return False
    receipt = row.get(REVIEW_KEY)
    if not isinstance(receipt, dict) or receipt.get("version") != REVIEW_VERSION:
        return False
    try:
        frame = row["analysis"].catalog
        return (receipt.get("tag") == row.get("tag")
                and receipt.get("recording_id") == row.get("recording_id")
                and type(receipt.get("event_count")) is int
                and receipt["event_count"] == len(frame)
                and type(receipt.get("accepted_count")) is int
                and receipt["accepted_count"] == int(frame.accepted.sum())
                and isinstance(receipt.get("confirmed_at"), str)
                and receipt.get("fingerprint") == review_fingerprint(row))
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError):
        return False


def require_reviewed(records) -> None:
    """Require explicit current approval for exactly the supplied recordings."""
    pending = [str(row.get("tag", "unnamed recording")) if isinstance(row, Mapping)
               else "unnamed recording" for row in records if not review_is_current(row)]
    if pending:
        raise ValueError("Confirm the reviewed events before running further analysis: "
                         + ", ".join(pending)
                         + ". In Detect / review, choose Use reviewed events for analysis.")
