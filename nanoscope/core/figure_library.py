"""Project-owned, immutable numerical/picture snapshots, separate from event galleries.

Each figure is an independent workspace artifact. Unknown future versions are carried
by the project container unchanged; this module only edits versions it understands.
Nothing in the library invokes a detector, fit, histogram or waveform calculation.
"""
from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping
from uuid import uuid4
import weakref

import numpy as np

from .analysis_workspace import (
    _digestable, _resolve_source, fingerprint_frame, result_recipe,
)
from .project import AnalyticalWorkspace, WorkspaceArtifact
from .identity import fingerprint_source
from .document import TraceDocument

FIGURE_KIND = "nanoscope.saved-figure"
FIGURE_SCHEMA = 1
FIGURE_PREFIX = "figure."
_LOADED_STAMP_CACHE_MAX = 128
_LOADED_STAMP_CACHE: OrderedDict[int, tuple] = OrderedDict()


def _hash_loaded_channel(array: np.ndarray) -> str:
    """Hash a contiguous immutable recording buffer without materializing a copy."""
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def loaded_source_content(document: Any) -> list[dict] | None:
    """Content identity of the actual loaded samples, independent of today's file.

    TraceDocument owns immutable contiguous channel buffers by contract. Cache one
    complete hash per document/buffer/metadata identity; styles and calibrations do
    not rewrite those buffers. The weak, bounded cache never keeps a recording (or
    its potentially very large arrays) alive. A replacement load is hashed afresh,
    even if its pathname, sample count and on-disk fingerprint are unchanged.
    Non-trace or unavailable buffers are unverified, never assumed identical.
    """
    if not isinstance(document, TraceDocument):
        return None
    arrays = document.channels
    if (not arrays or len(arrays) != len(document.meta.channels)
            or any(not isinstance(array, np.ndarray) or array.dtype != np.float32
                   or array.ndim != 1 or not array.flags.c_contiguous
                   or len(array) != document.meta.n_samples for array in arrays)):
        return None
    signature = (id(document.meta), str(document.meta.load_uid),
                 tuple((id(array), array.shape, array.strides, array.dtype.str)
                       for array in arrays))
    key = id(document)
    cached = _LOADED_STAMP_CACHE.get(key)
    if cached is not None and cached[0]() is document and cached[1] == signature:
        _LOADED_STAMP_CACHE.move_to_end(key)
        return deepcopy(cached[2])
    content = [{"channel": int(info.index), "dtype": array.dtype.str,
                "shape": list(array.shape), "sha256": _hash_loaded_channel(array)}
               for info, array in zip(document.meta.channels, arrays)]

    def discard(reference, identity=key):
        previous = _LOADED_STAMP_CACHE.get(identity)
        if previous is not None and previous[0] is reference:
            _LOADED_STAMP_CACHE.pop(identity, None)

    _LOADED_STAMP_CACHE[key] = (weakref.ref(document, discard), signature, content)
    _LOADED_STAMP_CACHE.move_to_end(key)
    while len(_LOADED_STAMP_CACHE) > _LOADED_STAMP_CACHE_MAX:
        _LOADED_STAMP_CACHE.popitem(last=False)
    return deepcopy(content)


def figures(workspace: AnalyticalWorkspace) -> tuple[WorkspaceArtifact, ...]:
    """Readable figures in saved order (never discard unreadable/unknown artifacts)."""
    return tuple(a for a in workspace.artifacts
                 if a.kind == FIGURE_KIND and a.schema_version == FIGURE_SCHEMA
                 and isinstance(a.payload, dict))


def source_recipes(artifact: WorkspaceArtifact) -> list[dict[str, Any]]:
    """Collect all durable source bindings, including multi-recording plots."""
    found: dict[str, dict] = {}

    def visit(value):
        if isinstance(value, Mapping):
            if value.get("source_key"):
                recipe = {key: deepcopy(value.get(key)) for key in (
                    "source_key", "channel", "channel_name", "channel_kind",
                    "kind", "module_id")}
                recipe.update({key: deepcopy(value[key]) for key in (
                    "event_frame_fingerprint", "result_fingerprint", "module_version")
                    if key in value})
                found[json.dumps(recipe, sort_keys=True)] = recipe
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    # A figure can be captured while different sources are selected but not yet
    # applied. The committed picture's bindings win; drafts stay in its config.
    payload = artifact.payload
    if isinstance(payload, Mapping) and "figure_sources" in payload:
        visit(payload["figure_sources"])
    else:
        visit(payload)
    return [found[key] for key in sorted(found)]


def source_state(owner: Any, recipes: list[dict]) -> dict | None:
    """Read a scoped source/calibration/result stamp; None means not fully verifiable.

    Loaded channel bytes are hashed once per immutable load, without copying them
    or running analysis. The separate bounded disk fingerprint describes the file
    at capture/check time, not necessarily the bytes originally loaded. Together
    with calibration and result tables these distinguish a changed reload from an
    unchanged picture. Display styles are not scientific mutations. Missing or
    unverifiable sources remain snapshots, never falsely labelled current.
    """
    if not recipes:
        return None
    session = getattr(owner, "_session", None)
    states = []
    for recipe in recipes:
        if recipe.get("kind") == "manual":
            panel = getattr(getattr(owner, "_analysis_panel", None), "measure_panel", None)
            frame = getattr(panel, "measurements_df", None)
            if callable(frame):
                frame = frame()
            if frame is None:
                return None
            states.append({"source": recipe, "table": fingerprint_frame(frame)})
            continue
        doc_id, channel, _note = _resolve_source(owner, recipe)
        if doc_id is None or session is None:
            return None
        document = session.document(doc_id)
        meta = document.meta
        loaded = loaded_source_content(document)
        if loaded is None:
            return None
        # The loaded content and today's file are distinct evidence. Do not stamp
        # old in-memory samples as a new recording after its path is overwritten.
        # Exclude launch-local ids from the persisted, content-based comparison.
        disk = fingerprint_source(meta.path, meta.n_samples)
        if disk is None:
            return None
        record = {
            "source": recipe,
            "samples": getattr(meta, "n_samples", None),
            "sample_rate": meta.sample_rate_hz,
            "rate_provenance": str(meta.rate_provenance.value),
            "t0_s": meta.t0_s,
            "loaded_range": (None if meta.loaded_range is None else
                             [meta.loaded_range.start, meta.loaded_range.stop]),
            "nominal_voltage_mV": meta.nominal_voltage_mV,
            "voltage_provenance": str(meta.voltage_provenance.value),
            "file_content": {"size": disk.size, "digest": disk.digest},
            "loaded_content": loaded,
            "calibration": (None if channel is None else
                            list(session.calibration(doc_id, channel).fingerprint())),
            "results": [],
        }
        for result in session.results.for_doc(doc_id):
            if channel is not None and int(result.channel) != channel:
                continue
            module = recipe.get("module_id")
            if module and result.module_id != module:
                continue
            record["results"].append(result_recipe(owner, result))
        record["results"].sort(key=lambda value: str(value.get("module_id")))
        states.append(record)
    return _digestable({"sources": states})


def _plain_stamp(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                    separators=(",", ":")).encode()).hexdigest()


def figure_status(artifact: WorkspaceArtifact, live_state: dict | None) -> tuple[str, str]:
    """Freshness is about source lineage; the image always remains a saved snapshot."""
    saved = artifact.payload.get("source_state")
    captured = artifact.payload.get("capture_status", "snapshot")
    if captured == "stale":
        return "Outdated", "The visible plot was already stale when this figure was saved."
    if saved is None or live_state is None:
        return "Saved snapshot", "Source lineage is unavailable or was not verified at capture; no analysis was rerun."
    if _plain_stamp(saved) != _plain_stamp(live_state):
        return "Outdated", "Source data, curation, parameters or calibration changed. The saved numbers and picture are unchanged."
    return "Current", "Source lineage matches. The figure retains its saved analysis settings and appearance."


def make_figure(*, name: str, page_id: str, source_label: str,
                analysis: WorkspaceArtifact, snapshot_blobs: Mapping[str, bytes],
                snapshot_metadata: Mapping[str, Any], live_state: dict | None,
                capture_status: str = "current") -> WorkspaceArtifact:
    """Detach all resources before the caller can mutate a live plot or source."""
    name = str(name).strip()
    if not name or len(name) > 200:
        raise ValueError("Figure name must contain 1–200 characters.")
    return WorkspaceArtifact(
        artifact_id=FIGURE_PREFIX + str(uuid4()), kind=FIGURE_KIND,
        schema_version=FIGURE_SCHEMA,
        payload={
            "name": name, "page_id": str(page_id),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_label": str(source_label), "source_recipes": source_recipes(analysis),
            "source_state": deepcopy(live_state), "capture_status": str(capture_status),
            "snapshot": deepcopy(dict(snapshot_metadata)), "export_path": None,
            "analysis": {"artifact_id": analysis.artifact_id, "kind": analysis.kind,
                         "schema_version": analysis.schema_version,
                         "payload": deepcopy(analysis.payload), "extra": deepcopy(analysis.extra)},
        },
        tables={name: frame.copy(deep=True) for name, frame in analysis.tables.items()},
        blobs={**{"analysis:" + key: bytes(value) for key, value in analysis.blobs.items()},
               **{str(key): bytes(value) for key, value in snapshot_blobs.items()}},
    )


def rename_figure(artifact: WorkspaceArtifact, name: str) -> None:
    """Rename metadata only; the numerical and picture resources are not touched."""
    name = str(name).strip()
    if not name or len(name) > 200:
        raise ValueError("Figure name must contain 1–200 characters.")
    artifact.payload["name"] = name


def duplicate_figure(artifact: WorkspaceArtifact) -> WorkspaceArtifact:
    duplicate = deepcopy(artifact)
    duplicate.artifact_id = FIGURE_PREFIX + str(uuid4())
    duplicate.payload["name"] = str(artifact.payload.get("name", "Figure"))[:193] + " (copy)"
    duplicate.payload["created_at"] = datetime.now(timezone.utc).isoformat()
    duplicate.payload["export_path"] = None
    return duplicate
