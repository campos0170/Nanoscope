"""Qt-free adapters for the durable analytical workspace.

The project container knows how to persist :class:`WorkspaceArtifact` values, but it
must not know which widgets own a PCA, a PSD, or an analysis form.  This module is the
small bridge between those layers.  It deliberately imports no GUI module and talks to
windows by their public API (with a few named compatibility fallbacks for state that the
current widgets do not expose publicly).

Capture must run on the GUI thread.  Every DataFrame/array is copied into an artifact
before the result is returned, so the resulting :class:`AnalyticalWorkspace` is safe to
hand to a worker which serializes the project.  Restore must likewise run on the GUI
thread after MainWindow has rebound each window to its live Session/source data.

Large numerical values never enter JSON metadata.  They are extracted into the
artifact's ``tables`` mapping, which the project format stores as Parquet.  The tagged
metadata codec is intentionally small and reconstructs only NanoScope analysis/core
value types; it never imports arbitrary classes named by a project file.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import date, datetime, time
from enum import Enum
from collections import OrderedDict
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from .project import AnalyticalWorkspace, WorkspaceArtifact
from .calibration import channel_identity, resolve_channel
from .identity import same_source
from .segment import persistent_key


__all__ = [
    "ANALYSIS_PANEL_ARTIFACT_ID",
    "EVENT_PLOTS_ARTIFACT_ID",
    "DISTRIBUTION_SUMMARIES_ARTIFACT_ID",
    "IV_ARTIFACT_ID",
    "OWNED_ARTIFACT_IDS",
    "PLOTS_ARTIFACT_ID",
    "PSD_ARTIFACT_ID",
    "SUBPOPULATION_ARTIFACT_ID",
    "VOLTAGE_DEPENDENCE_ARTIFACT_ID",
    "FileFingerprint",
    "RestoreReport",
    "capture_analysis_panel",
    "capture_analysis_workspace",
    "capture_event_plots",
    "capture_distribution_summaries",
    "capture_iv",
    "capture_model_reference",
    "capture_plots",
    "capture_psd",
    "capture_subpopulation",
    "capture_voltage_dependence",
    "artifact_source_recipe",
    "clear_unresolved_model_reference",
    "file_fingerprint",
    "fingerprint_frame",
    "result_recipe",
    "resolve_result_recipe",
    "restore_analysis_panel",
    "restore_analysis_workspace",
    "restore_event_plots",
    "restore_distribution_summaries",
    "restore_iv",
    "restore_plots",
    "restore_psd",
    "restore_subpopulation",
    "restore_voltage_dependence",
]


# Stable built-in identities.  A producer replaces only its own id; plugin/unknown
# artifacts retain their position and bytes when the workspace is captured again.
PLOTS_ARTIFACT_ID = "builtin.plots"
DISTRIBUTION_SUMMARIES_ARTIFACT_ID = "builtin.distribution-summaries"
SUBPOPULATION_ARTIFACT_ID = "builtin.subpopulation"
PSD_ARTIFACT_ID = "builtin.psd"
EVENT_PLOTS_ARTIFACT_ID = "builtin.event-plots"
IV_ARTIFACT_ID = "builtin.iv"
VOLTAGE_DEPENDENCE_ARTIFACT_ID = "builtin.voltage-dependence"
ANALYSIS_PANEL_ARTIFACT_ID = "builtin.analysis-panel"

_KINDS = {
    PLOTS_ARTIFACT_ID: "nanoscope.plots",
    DISTRIBUTION_SUMMARIES_ARTIFACT_ID: "nanoscope.distribution-summaries",
    SUBPOPULATION_ARTIFACT_ID: "nanoscope.subpopulation-report",
    PSD_ARTIFACT_ID: "nanoscope.psd",
    EVENT_PLOTS_ARTIFACT_ID: "nanoscope.event-plots",
    IV_ARTIFACT_ID: "nanoscope.iv",
    VOLTAGE_DEPENDENCE_ARTIFACT_ID: "nanoscope.voltage-dependence",
    ANALYSIS_PANEL_ARTIFACT_ID: "nanoscope.analysis-panel",
}
_ORDER = tuple(_KINDS)
OWNED_ARTIFACT_IDS = frozenset(_ORDER)
_SCHEMA_VERSION = 1

_TAG = "__nanoscope_workspace_value__"
_SAFE_TYPE_MODULES = (
    "nanoscope.analysis.",
    "nanoscope.core.analysis_workspace",
)
_TABLE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MODEL_FINGERPRINT_CACHE: "OrderedDict[tuple[Any, ...], FileFingerprint]" = OrderedDict()
_MODEL_FINGERPRINT_CACHE_MAX = 16


@dataclass(frozen=True)
class FileFingerprint:
    """Identity of model content at capture time, independent of its pathname."""

    size: int
    mtime_ns: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": int(self.size),
            "mtime_ns": int(self.mtime_ns),
            "sha256": str(self.sha256),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FileFingerprint":
        return cls(
            size=int(value["size"]),
            mtime_ns=int(value["mtime_ns"]),
            sha256=str(value["sha256"]),
        )


@dataclass(frozen=True)
class RestoreReport:
    """What a restore applied, skipped, and could not reproduce exactly."""

    restored: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def merge(cls, reports: Iterable["RestoreReport"]) -> "RestoreReport":
        restored: list[str] = []
        skipped: list[str] = []
        warnings: list[str] = []
        for report in reports:
            restored.extend(report.restored)
            skipped.extend(report.skipped)
            warnings.extend(report.warnings)
        return cls(tuple(restored), tuple(skipped), tuple(warnings))


class _Resources:
    """Allocate deterministic logical names while copying every bulk resource."""

    def __init__(self) -> None:
        self.tables: dict[str, pd.DataFrame] = {}
        self.blobs: dict[str, bytes] = {}
        self._ordinal = 0

    def table(self, value: pd.DataFrame, hint: str) -> str:
        stem = _TABLE_NAME_RE.sub("_", hint).strip("._") or "table"
        name = f"t{self._ordinal:03d}.{stem}"[:180]
        self._ordinal += 1
        while name in self.tables or name in self.blobs:
            name = f"t{self._ordinal:03d}.{stem}"[:180]
            self._ordinal += 1
        # deep=True owns every ndarray block.  Object cells are required by the workspace
        # contract to be serialization-friendly; copying their containers is handled by
        # ProjectState's outer snapshot if a plugin uses them.
        self.tables[name] = value.copy(deep=True)
        return name

    def blob(self, value: bytes, hint: str) -> str:
        stem = _TABLE_NAME_RE.sub("_", hint).strip("._") or "blob"
        name = f"b{self._ordinal:03d}.{stem}"[:180]
        self._ordinal += 1
        while name in self.tables or name in self.blobs:
            name = f"b{self._ordinal:03d}.{stem}"[:180]
            self._ordinal += 1
        self.blobs[name] = bytes(value)
        return name


def file_fingerprint(path: str | Path, *, chunk_size: int = 1024 * 1024) -> FileFingerprint:
    """Hash a regular model file and reject a file which changes during the read.

    ``.keras`` models are regular ZIP files, so hashing their bytes fingerprints both
    architecture and weights without importing TensorFlow.  Missing paths and directories
    raise their ordinary filesystem exceptions; :func:`capture_model_reference` is the
    non-throwing UI-facing wrapper.
    """

    source = Path(path).expanduser()
    before = source.stat()
    if not source.is_file():
        raise IsADirectoryError(str(source))
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    after = source.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns)):
        raise OSError(f"{source}: file changed while its fingerprint was being computed")
    return FileFingerprint(int(after.st_size), int(after.st_mtime_ns), digest.hexdigest())


def capture_model_reference(path: str | Path | None) -> dict[str, Any] | None:
    """Return a path plus content fingerprint; retain an unavailable path with a reason."""

    if not path:
        return None
    text = str(path)
    try:
        source = Path(text).expanduser()
        stat = source.stat()
        resolved = source.resolve(strict=True)
        # ctime closes the same-size rewrite + restored-mtime hole.  A model manager may
        # preserve mtime intentionally, but replacing bytes still advances the inode change
        # token on the filesystems NanoScope supports.  The persisted identity remains the
        # full SHA; this key only avoids repeating that SHA while a file is unchanged.
        key = (str(resolved), int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
               int(stat.st_mtime_ns), int(stat.st_ctime_ns))
        fingerprint = _MODEL_FINGERPRINT_CACHE.get(key)
        if fingerprint is None:
            fingerprint = file_fingerprint(source)
            _MODEL_FINGERPRINT_CACHE[key] = fingerprint
            _MODEL_FINGERPRINT_CACHE.move_to_end(key)
            while len(_MODEL_FINGERPRINT_CACHE) > _MODEL_FINGERPRINT_CACHE_MAX:
                _MODEL_FINGERPRINT_CACHE.popitem(last=False)
        else:
            _MODEL_FINGERPRINT_CACHE.move_to_end(key)
    except FileNotFoundError:
        return {"path": text, "fingerprint": None, "status": "missing"}
    except IsADirectoryError:
        return {"path": text, "fingerprint": None, "status": "not-a-file"}
    except OSError:
        return {"path": text, "fingerprint": None, "status": "unreadable"}
    return {"path": text, "fingerprint": fingerprint.to_dict(), "status": "ok"}


def clear_unresolved_model_reference(panel: Any) -> None:
    """Forget a carried missing/changed model after an explicit user clear action.

    Restore deliberately keeps an unavailable reference so editing another form does
    not erase it on the next save.  A UI action whose meaning is *remove this model*
    should call this helper after ``DetectionPanel.set_model_path(None)``.
    """

    detection = getattr(panel, "detection_panel", panel)
    if hasattr(detection, "_nanoscope_unresolved_model_reference"):
        delattr(detection, "_nanoscope_unresolved_model_reference")


def _model_reference_matches(reference: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not reference:
        return True, ""
    path = str(reference.get("path") or "")
    raw = reference.get("fingerprint")
    if not path or not isinstance(raw, Mapping):
        return False, "the saved ML model has no verifiable content fingerprint"
    try:
        wanted = FileFingerprint.from_dict(raw)
        current = file_fingerprint(path)
    except (OSError, KeyError, TypeError, ValueError):
        return False, f"the saved ML model is unavailable at {path!r}"
    if current.sha256 != wanted.sha256 or current.size != wanted.size:
        return False, f"the ML model at {path!r} no longer matches the saved content fingerprint"
    return True, ""


def fingerprint_frame(frame: pd.DataFrame | None) -> str | None:
    """Deterministic content/schema stamp used to guard restored derived results."""

    if frame is None:
        return None
    digest = hashlib.sha256()
    schema = {
        "columns": [repr(v) for v in frame.columns.tolist()],
        "column_names": [repr(v) for v in frame.columns.names],
        "index_names": [repr(v) for v in frame.index.names],
        "dtypes": [str(v) for v in frame.dtypes.tolist()],
        "shape": list(frame.shape),
    }
    digest.update(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    try:
        hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    except (TypeError, ValueError):
        # Nested object cells are unusual in analysis tables.  repr is the honest fallback:
        # it still makes a mutation visible and does not mutate/coerce the caller's frame.
        digest.update(repr(frame.to_dict(orient="split")).encode("utf-8", "backslashreplace"))
    else:
        digest.update(np.asarray(hashed, dtype=np.uint64).tobytes(order="C"))
    return digest.hexdigest()


def _channel_info(document: Any, channel: int | None) -> Any | None:
    if channel is None:
        return None
    try:
        wanted = int(channel)
        channels = tuple(document.meta.channels)
    except (AttributeError, TypeError, ValueError):
        return None
    return next((info for info in channels if int(info.index) == wanted), None)


def _owner_session(owner: Any) -> Any | None:
    """Find a Session on a window or its immediate owner without importing Qt.

    Most analytical windows own ``_session`` directly.  The subpopulation report is a
    presentation-only child of MainWindow, so its durable ``AnalysisResult`` lineage is
    resolved through the parent's Session instead.
    """

    session = getattr(owner, "_session", None)
    if session is not None:
        return session
    for accessor_name in ("parent", "parentWidget"):
        accessor = getattr(owner, accessor_name, None)
        if not callable(accessor):
            continue
        try:
            parent = accessor()
        except (RuntimeError, TypeError):
            continue
        session = getattr(parent, "_session", None)
        if session is not None:
            return session
    return None


def _stable_source(window: Any, doc_id: str, channel: int | None = None) -> dict[str, Any]:
    """Translate a launch-local target into durable recording *and channel* identity."""

    source_key = ""
    document = None
    session = _owner_session(window)
    try:
        document = session.document(doc_id)
        source_key = str(persistent_key(document))
    except (AttributeError, LookupError, TypeError, ValueError):
        pass
    if channel is None:
        try:
            channel = session.resolve_current_channel(doc_id, None)
        except (AttributeError, KeyError, TypeError, ValueError):
            channel = None
    name = kind = None
    info = _channel_info(document, channel)
    if info is not None:
        try:
            name, kind = channel_identity(info)
        except (AttributeError, TypeError, ValueError):
            pass
    return {
        "source_key": source_key,
        "channel": None if channel is None else int(channel),
        "channel_name": name,
        "channel_kind": kind,
    }


def _same_persistent_source(left: str, right: str) -> bool:
    """Compare file keys through the filesystem, while segment UIDs remain exact."""

    return bool(left and right and (left == right or same_source(left, right)))


def _resolve_source(window: Any, recipe: Mapping[str, Any]) -> tuple[str | None, int | None, str | None]:
    """Resolve a persisted source recipe to ``(live doc_id, live channel, note)``.

    ``note`` is non-empty when the path spelling or channel position moved.  Callers
    decide whether that belongs in a restore report; failure is represented by two
    ``None`` values and an actionable reason.
    """

    source_key = str(recipe.get("source_key") or "")
    if not source_key:
        return None, None, "the saved source has no persistent recording identity"
    try:
        documents = tuple(_owner_session(window).documents())
    except AttributeError:
        return None, None, f"source {source_key!r} cannot be resolved without a session"
    # Prefer exact key equality so duplicate file handles cannot make a samefile match
    # ambiguous.  `same_source` is the established fallback for symlink/case relocation.
    exact = [doc for doc in documents if str(persistent_key(doc)) == source_key]
    matching = exact or [doc for doc in documents
                         if _same_persistent_source(str(persistent_key(doc)), source_key)]
    if len(matching) != 1:
        if not matching:
            return None, None, f"source {source_key!r} is not open"
        return None, None, f"source {source_key!r} resolves to more than one open recording"
    document = matching[0]
    path_note = None if exact else f"source {source_key!r} is open under another path spelling"

    raw_channel = recipe.get("channel")
    if raw_channel is None:
        return str(document.doc_id), None, path_note
    try:
        resolved, moved = resolve_channel(
            tuple(document.meta.channels), int(raw_channel),
            None if recipe.get("channel_name") is None else str(recipe.get("channel_name")),
            None if recipe.get("channel_kind") is None else str(recipe.get("channel_kind")),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return None, None, f"source {source_key!r} channel identity is invalid: {exc}"
    if resolved is None:
        return None, None, f"source {source_key!r}: {moved or 'saved channel is unavailable'}"
    notes = [value for value in (path_note, moved) if value]
    return str(document.doc_id), int(resolved), "; ".join(notes) or None


def _live_doc_id(window: Any, source_key: str) -> str | None:
    """Resolve durable recording identity, including symlink/case-equivalent paths."""

    if not source_key:
        return None
    try:
        documents = tuple(_owner_session(window).documents())
    except AttributeError:
        return None
    exact = [doc for doc in documents if str(persistent_key(doc)) == source_key]
    matching = exact or [doc for doc in documents
                         if _same_persistent_source(str(persistent_key(doc)), source_key)]
    return str(matching[0].doc_id) if len(matching) == 1 else None


def _digestable(value: Any) -> Any:
    """Small deterministic projection used only for result identity stamps."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.generic):
        return _digestable(value.item())
    if isinstance(value, (Path, Enum)):
        return str(getattr(value, "value", value))
    if isinstance(value, pd.DataFrame):
        return {"dataframe": fingerprint_frame(value)}
    if isinstance(value, Mapping):
        return {str(key): _digestable(value[key]) for key in sorted(value, key=lambda k: str(k))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items.sort(key=repr)
        return [_digestable(item) for item in items]
    return {"type": _type_name(type(value)), "repr": repr(value)}


def _result_recipe(window: Any, result: Any) -> dict[str, Any]:
    payload = getattr(result, "payload", None)
    frame = getattr(payload, "df", None)
    frame_stamp = fingerprint_frame(frame if isinstance(frame, pd.DataFrame) else None)
    body = {
        "module_id": str(getattr(result, "module_id", "") or ""),
        "module_version": str(getattr(result, "module_version", "") or ""),
        "event_frame_fingerprint": frame_stamp,
        "params": _digestable(getattr(result, "params", {}) or {}),
        "provenance": _digestable(getattr(payload, "provenance", {}) or {}),
        "calibration_fingerprint": _digestable(
            getattr(result, "calibration_fingerprint", None)),
        "baseline_fingerprint": getattr(result, "baseline_fingerprint", None),
    }
    stamp = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"))
                           .encode("utf-8")).hexdigest()
    return {
        **_stable_source(window, str(getattr(result, "doc_id", "")),
                         int(getattr(result, "channel", 0))),
        "module_id": body["module_id"],
        "module_version": body["module_version"],
        "event_frame_fingerprint": frame_stamp,
        "result_fingerprint": stamp,
    }


def result_recipe(owner: Any, result: Any) -> dict[str, Any]:
    """Capture an exact, durable binding recipe for one live analysis result.

    This is the launch-time seam for asynchronous report jobs: callers can retain
    the returned plain-value copy even if the source document closes before the job
    produces its window.  A launch-local ``doc_id`` is never serialized.  The source
    must still be resolvable at capture time so a blank or ephemeral identity cannot
    silently enter a project.

    Raises:
        ValueError: if ``result`` cannot be tied to exactly one currently open,
            persistently identified recording/channel.
    """

    recipe = _result_recipe(owner, result)
    if not recipe.get("source_key"):
        raise ValueError(
            "the analysis result source is not open; capture its durable recipe "
            "before starting asynchronous work")
    if recipe.get("channel") is None:
        raise ValueError("the analysis result has no durable channel identity")
    return _clone_plain(recipe)


def _event_window_source_recipe(window: Any) -> dict[str, Any] | None:
    doc_id = getattr(window, "_doc_id", None)
    channel = getattr(window, "_channel", None)
    eventset = getattr(window, "_eventset", None)
    if doc_id is None or channel is None or eventset is None:
        return None
    frame = getattr(eventset, "df", None)
    frame_stamp = fingerprint_frame(frame if isinstance(frame, pd.DataFrame) else None)
    try:
        results = tuple(window._session.results.all())
    except AttributeError:
        results = ()
    exact = [res for res in results
             if str(getattr(res, "doc_id", "")) == str(doc_id)
             and int(getattr(res, "channel", -1)) == int(channel)
             and getattr(res, "payload", None) is eventset]
    candidates = exact or [res for res in results
                           if str(getattr(res, "doc_id", "")) == str(doc_id)
                           and int(getattr(res, "channel", -1)) == int(channel)
                           and fingerprint_frame(getattr(getattr(res, "payload", None), "df", None))
                           == frame_stamp]
    if len(candidates) == 1:
        return _result_recipe(window, candidates[0])
    return {
        **_stable_source(window, str(doc_id), int(channel)),
        "module_id": None,
        "module_version": None,
        "event_frame_fingerprint": frame_stamp,
        "result_fingerprint": None,
    }


def artifact_source_recipe(artifact: WorkspaceArtifact) -> dict[str, Any] | None:
    """Return a copy of the artifact's primary durable binding recipe, if any.

    MainWindow can inspect this before constructing/rebinding a lazy window.  The
    recipe never contains a launch-local ``doc_id``.
    """

    if not isinstance(artifact.payload, Mapping):
        return None
    raw = artifact.payload.get("source")
    if not isinstance(raw, Mapping):
        config = artifact.payload.get("config")
        if isinstance(config, Mapping):
            raw = config.get("target_source")
    if not isinstance(raw, Mapping):
        selected = artifact.payload.get("selected_sources")
        if isinstance(selected, (list, tuple)) and selected:
            raw = selected[0]
    return _clone_plain(dict(raw)) if isinstance(raw, Mapping) else None


def resolve_result_recipe(owner: Any, recipe: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    """Resolve a durable recipe only when the stored result has the exact saved lineage.

    The returned result must match recording identity, channel identity, module id and
    version, event-frame content, and the full parameters/provenance/calibration result
    fingerprint.  A relocated path spelling or reordered channel may still resolve through
    the established identity rules; that successful result is returned with the relocation
    note as its warning.  Missing or stale lineage returns ``(None, actionable_reason)``.
    """

    if not isinstance(recipe, Mapping):
        return None, "the saved result recipe is not an object"
    doc_id, channel, source_note = _resolve_source(owner, recipe)
    if doc_id is None:
        return None, source_note or "the saved result source is unavailable"
    if channel is None:
        return None, "the saved result recipe does not identify a channel"
    module_id = str(recipe.get("module_id") or "")
    if not module_id:
        return None, "the saved result recipe has no module identity"
    if "module_version" not in recipe or recipe.get("module_version") is None:
        return None, f"saved result {module_id!r} has no module-version identity"
    saved_rows = recipe.get("event_frame_fingerprint")
    if not saved_rows:
        return None, f"saved result {module_id!r} has no event-table fingerprint"
    saved_result = recipe.get("result_fingerprint")
    if not saved_result:
        return None, f"saved result {module_id!r} has no full lineage fingerprint"

    session = _owner_session(owner)
    try:
        result = session.results.get(doc_id, int(channel), module_id)
    except (AttributeError, KeyError, TypeError, ValueError):
        result = None
    if result is None:
        return None, (
            f"saved result {module_id!r} is not available for source "
            f"{recipe.get('source_key')!r}, channel {channel}")
    saved_version = str(recipe.get("module_version"))
    live_version = str(getattr(result, "module_version", ""))
    if live_version != saved_version:
        return None, (
            f"saved result {module_id!r} used module version {saved_version!r}; "
            f"the available result uses {live_version!r}")
    live_rows = fingerprint_frame(getattr(getattr(result, "payload", None), "df", None))
    if live_rows != str(saved_rows):
        return None, f"saved result {module_id!r} no longer has the same event-table content"
    live_result = _result_recipe(owner, result).get("result_fingerprint")
    if live_result != str(saved_result):
        return None, (
            f"saved result {module_id!r} has different parameters, provenance, "
            "calibration, or baseline lineage")
    return result, source_note


def _type_name(value: type) -> str:
    return f"{value.__module__}:{value.__qualname__}"


def _encode(value: Any, resources: _Resources, hint: str) -> Any:
    """Encode a value tree into JSON metadata plus copied table/blob resources."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {_TAG: "float", "value": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if value is pd.NA:
        return {_TAG: "pd.NA"}
    if value is pd.NaT:
        return {_TAG: "pd.NaT"}
    if isinstance(value, np.generic):
        return _encode(value.item(), resources, hint)
    if isinstance(value, complex):
        return {_TAG: "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, pd.Timestamp):
        return {_TAG: "pd.Timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {_TAG: "pd.Timedelta", "value_ns": int(value.value)}
    if isinstance(value, (datetime, date, time)):
        return {_TAG: type(value).__name__, "value": value.isoformat()}
    if isinstance(value, Path):
        return {_TAG: "path", "value": str(value)}
    if isinstance(value, Enum):
        return {_TAG: "enum", "type": _type_name(type(value)),
                "value": _encode(value.value, resources, hint + ".value")}
    if isinstance(value, pd.DataFrame):
        name = resources.table(value, hint)
        return {_TAG: "dataframe", "table": name}
    if isinstance(value, pd.Series):
        name = resources.table(value.to_frame("__value__"), hint)
        return {_TAG: "series", "table": name,
                "name": _encode(value.name, resources, hint + ".name")}
    if isinstance(value, pd.RangeIndex):
        return {_TAG: "range-index", "start": value.start, "stop": value.stop,
                "step": value.step, "name": _encode(value.name, resources, hint + ".name")}
    if isinstance(value, pd.MultiIndex):
        name = resources.table(value.to_frame(index=False, allow_duplicates=True), hint)
        return {_TAG: "multi-index", "table": name,
                "names": _encode(tuple(value.names), resources, hint + ".names")}
    if isinstance(value, pd.Index):
        name = resources.table(pd.DataFrame({"__value__": pd.Series(value.array)}), hint)
        return {_TAG: "index", "table": name,
                "name": _encode(value.name, resources, hint + ".name")}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return {_TAG: "ndarray-scalar", "dtype": value.dtype.str,
                    "value": _encode(value.item(), resources, hint + ".value")}
        flat = np.ascontiguousarray(value).reshape(-1)
        name = resources.table(pd.DataFrame({"__value__": flat}), hint)
        return {_TAG: "ndarray", "table": name, "shape": list(value.shape),
                "dtype": value.dtype.str}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {_TAG: "bytes", "blob": resources.blob(bytes(value), hint)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            _TAG: "dataclass",
            "type": _type_name(type(value)),
            "fields": {f.name: _encode(getattr(value, f.name), resources, f"{hint}.{f.name}")
                       for f in fields(value)},
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            _TAG: "namedtuple",
            "type": _type_name(type(value)),
            "fields": {name: _encode(getattr(value, name), resources, f"{hint}.{name}")
                       for name in value._fields},
        }
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [_encode(v, resources, f"{hint}.{i}")
                                            for i, v in enumerate(value)]}
    if isinstance(value, list):
        return [_encode(v, resources, f"{hint}.{i}") for i, v in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        ordered = sorted(value, key=repr)
        return {_TAG: "frozenset" if isinstance(value, frozenset) else "set",
                "items": [_encode(v, resources, f"{hint}.{i}")
                          for i, v in enumerate(ordered)]}
    if isinstance(value, Mapping):
        if all(isinstance(k, str) and k != _TAG for k in value):
            return {k: _encode(v, resources, f"{hint}.{k}") for k, v in value.items()}
        return {_TAG: "mapping", "items": [
            [_encode(k, resources, f"{hint}.key{i}"),
             _encode(v, resources, f"{hint}.value{i}")]
            for i, (k, v) in enumerate(value.items())
        ]}
    raise TypeError(f"workspace value at {hint!r} has unsupported type {type(value).__name__}")


def _resolve_type(name: str) -> type:
    module_name, sep, qualname = str(name).partition(":")
    if not sep or not any(module_name == p.rstrip(".") or module_name.startswith(p)
                          for p in _SAFE_TYPE_MODULES):
        raise ValueError(f"workspace value names unsafe type {name!r}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        if part == "<locals>":
            raise ValueError(f"workspace value names non-importable type {name!r}")
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"workspace value {name!r} does not name a type")
    return obj


def _decode(value: Any, tables: Mapping[str, pd.DataFrame], blobs: Mapping[str, bytes]) -> Any:
    if isinstance(value, list):
        return [_decode(v, tables, blobs) for v in value]
    if not isinstance(value, Mapping):
        return value
    kind = value.get(_TAG)
    if kind is None:
        return {str(k): _decode(v, tables, blobs) for k, v in value.items()}
    if kind == "float":
        return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}[str(value["value"])]
    if kind == "pd.NA":
        return pd.NA
    if kind == "pd.NaT":
        return pd.NaT
    if kind == "complex":
        return complex(float(value["real"]), float(value["imag"]))
    if kind == "pd.Timestamp":
        return pd.Timestamp(value["value"])
    if kind == "pd.Timedelta":
        return pd.Timedelta(int(value["value_ns"]), unit="ns")
    if kind == "datetime":
        return datetime.fromisoformat(str(value["value"]))
    if kind == "date":
        return date.fromisoformat(str(value["value"]))
    if kind == "time":
        return time.fromisoformat(str(value["value"]))
    if kind == "path":
        return Path(str(value["value"]))
    if kind == "enum":
        return _resolve_type(str(value["type"]))(_decode(value["value"], tables, blobs))
    if kind == "dataframe":
        return tables[str(value["table"])].copy(deep=True)
    if kind == "series":
        frame = tables[str(value["table"])].copy(deep=True)
        series = frame.iloc[:, 0].copy()
        series.name = _decode(value.get("name"), tables, blobs)
        return series
    if kind == "range-index":
        return pd.RangeIndex(int(value["start"]), int(value["stop"]), int(value["step"]),
                             name=_decode(value.get("name"), tables, blobs))
    if kind == "multi-index":
        frame = tables[str(value["table"])].copy(deep=True)
        out = pd.MultiIndex.from_frame(frame)
        out.names = list(_decode(value.get("names"), tables, blobs))
        return out
    if kind == "index":
        frame = tables[str(value["table"])].copy(deep=True)
        return pd.Index(frame.iloc[:, 0].array,
                        name=_decode(value.get("name"), tables, blobs))
    if kind == "ndarray-scalar":
        return np.asarray(_decode(value["value"], tables, blobs),
                          dtype=np.dtype(str(value["dtype"]))).reshape(())
    if kind == "ndarray":
        frame = tables[str(value["table"])].copy(deep=True)
        array = frame.iloc[:, 0].to_numpy(copy=True).astype(np.dtype(str(value["dtype"])), copy=False)
        return array.reshape(tuple(int(v) for v in value["shape"]))
    if kind == "bytes":
        return bytes(blobs[str(value["blob"])])
    if kind in ("dataclass", "namedtuple"):
        cls = _resolve_type(str(value["type"]))
        kwargs = {str(k): _decode(v, tables, blobs)
                  for k, v in dict(value["fields"]).items()}
        return cls(**kwargs)
    if kind == "tuple":
        return tuple(_decode(v, tables, blobs) for v in value["items"])
    if kind == "set":
        return set(_decode(v, tables, blobs) for v in value["items"])
    if kind == "frozenset":
        return frozenset(_decode(v, tables, blobs) for v in value["items"])
    if kind == "mapping":
        return {_decode(k, tables, blobs): _decode(v, tables, blobs)
                for k, v in value["items"]}
    raise ValueError(f"unknown workspace value tag {kind!r}")


def _decode_like(template: Any, value: Any, artifact: WorkspaceArtifact) -> Any:
    """Rebuild GUI dataclass configuration from a live instance, importing no GUI type."""

    if is_dataclass(template) and not isinstance(template, type):
        if not isinstance(value, Mapping) or value.get(_TAG) != "dataclass":
            raise TypeError("saved configuration does not match the live dataclass")
        raw_fields = dict(value.get("fields", {}))
        changes = {
            f.name: _decode_like(getattr(template, f.name), raw_fields[f.name], artifact)
            for f in fields(template) if f.name in raw_fields
        }
        return replace(template, **changes)
    if isinstance(template, tuple) and isinstance(value, Mapping) and value.get(_TAG) == "tuple":
        return tuple(_decode(v, artifact.tables, artifact.blobs) for v in value["items"])
    if isinstance(template, frozenset) and isinstance(value, Mapping) and value.get(_TAG) == "frozenset":
        return frozenset(_decode(v, artifact.tables, artifact.blobs) for v in value["items"])
    return _decode(value, artifact.tables, artifact.blobs)


def _artifact(artifact_id: str, payload: Any, resources: _Resources) -> WorkspaceArtifact:
    return WorkspaceArtifact(
        artifact_id=artifact_id,
        kind=_KINDS[artifact_id],
        schema_version=_SCHEMA_VERSION,
        payload=payload,
        tables=resources.tables,
        blobs=resources.blobs,
    )


def _artifact_ok(artifact: WorkspaceArtifact, artifact_id: str) -> tuple[bool, str]:
    if artifact.artifact_id != artifact_id:
        return False, f"expected artifact {artifact_id!r}, got {artifact.artifact_id!r}"
    if artifact.kind != _KINDS[artifact_id]:
        return False, f"artifact {artifact_id!r} has unsupported kind {artifact.kind!r}"
    if artifact.schema_version != _SCHEMA_VERSION:
        return False, (f"artifact {artifact_id!r} uses schema {artifact.schema_version}; "
                       f"this build supports {_SCHEMA_VERSION}")
    if not isinstance(artifact.payload, Mapping):
        return False, f"artifact {artifact_id!r} payload is not an object"
    return True, ""


def _widget_value(owner: Any, name: str, method: str, default: Any = None) -> Any:
    widget = getattr(owner, name, None)
    fn = getattr(widget, method, None)
    return fn() if callable(fn) else default


def _copy_frame_from_plot(window: Any) -> pd.DataFrame | None:
    getter = getattr(window, "_current_frame", None)
    if not callable(getter):
        return None
    frame = getter()
    return frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else None


def _preprocessing(frame: pd.DataFrame | None, result: Any) -> dict[str, Any]:
    columns = list(getattr(result, "columns", ()) or ())
    meta = dict(getattr(result, "meta", {}) or {})
    out: dict[str, Any] = {
        "standardize": bool(meta.get("standardize", True)),
        "columns": columns,
        "random_state": meta.get("random_state"),
    }
    if frame is None or not columns or not all(c in frame.columns for c in columns):
        out["available"] = False
        return out
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.shape[0] != int(getattr(result, "n_kept", values.shape[0])):
        out["available"] = False
        return out
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    out.update({"available": True, "mean": mean.tolist(), "scale": scale.tolist(),
                "n_rows": int(values.shape[0])})
    return out


def _plot_source_recipe(window: Any, source: Any) -> dict[str, Any]:
    """Durable identity of one PlotSource, never its launch-local key alone."""

    kind = str(getattr(source, "kind", "detection") or "detection")
    if kind == "manual":
        return {
            "kind": "manual",
            "source_key": str(getattr(source, "doc_key", "manual-measurements")),
            "channel": None,
            "channel_name": None,
            "channel_kind": None,
            "module_id": None,
            "event_frame_fingerprint": fingerprint_frame(
                getattr(source, "df", None) if isinstance(getattr(source, "df", None), pd.DataFrame)
                else None),
        }
    recipe = _stable_source(
        window, str(getattr(source, "doc_id", "") or ""),
        None if getattr(source, "channel", None) is None else int(source.channel))
    # A carried source may not currently be bound.  Its PlotSource still owns the
    # durable doc key, so keep that instead of replacing it with an empty lookup.
    if not recipe["source_key"]:
        recipe["source_key"] = str(getattr(source, "doc_key", "") or "")
    return {
        "kind": kind,
        **recipe,
        "module_id": str(getattr(source, "module_id", "") or "") or None,
        "event_frame_fingerprint": fingerprint_frame(
            getattr(source, "df", None) if isinstance(getattr(source, "df", None), pd.DataFrame)
            else None),
    }


def _resolve_plot_recipe(window: Any, recipe: Mapping[str, Any], *,
                         require_same_rows: bool = False) -> tuple[str | None, str | None]:
    """Map one saved PlotSource recipe onto the source key offered in this session."""

    candidates: list[tuple[str, Any, str | None]] = []
    for key, source in dict(getattr(window, "_sources", {}) or {}).items():
        if str(recipe.get("kind") or "detection") != str(
                getattr(source, "kind", "detection") or "detection"):
            continue
        if str(recipe.get("kind") or "") == "manual":
            if str(getattr(source, "doc_key", "")) == str(recipe.get("source_key") or ""):
                candidates.append((str(key), source, None))
            continue
        if str(getattr(source, "module_id", "") or "") != str(recipe.get("module_id") or ""):
            continue
        if not _same_persistent_source(str(getattr(source, "doc_key", "") or ""),
                                       str(recipe.get("source_key") or "")):
            continue
        # Resolve the stored channel against the live document, then require this exact
        # PlotSource to be on it.  This migrates colour/source keys when columns reorder.
        doc_id, channel, note = _resolve_source(window, recipe)
        if doc_id is None or channel is None:
            continue
        if (str(getattr(source, "doc_id", "") or "") == doc_id
                and int(getattr(source, "channel", -1)) == channel):
            candidates.append((str(key), source, note))
    if len(candidates) != 1:
        return None, (f"plot source {recipe.get('source_key')!r} is not uniquely available"
                      if not candidates else
                      f"plot source {recipe.get('source_key')!r} is ambiguous")
    key, source, relocation_note = candidates[0]
    saved_stamp = recipe.get("event_frame_fingerprint")
    live_stamp = fingerprint_frame(
        source.df if isinstance(getattr(source, "df", None), pd.DataFrame) else None)
    if saved_stamp and live_stamp != saved_stamp:
        warning = f"plot source {recipe.get('source_key')!r} contains changed event rows"
        if relocation_note:
            warning = f"{relocation_note}; {warning}"
        return (None, warning) if require_same_rows else (key, warning)
    return key, relocation_note


def _copy_resource_tree(value: Any, artifact: WorkspaceArtifact,
                        resources: _Resources, hint: str) -> Any:
    """Copy a carried artifact subtree while rebasing all resource references."""

    table_names = {name: resources.table(frame, f"{hint}.{name}")
                   for name, frame in artifact.tables.items()}
    blob_names = {name: resources.blob(data, f"{hint}.{name}")
                  for name, data in artifact.blobs.items()}
    table_fields = {"table", "labels_table", "means_table", "scores_table", "loadings_table"}

    def rewrite(item: Any) -> Any:
        if isinstance(item, list):
            return [rewrite(part) for part in item]
        if isinstance(item, tuple):
            return tuple(rewrite(part) for part in item)
        if not isinstance(item, Mapping):
            return _clone_plain(item)
        out: dict[str, Any] = {}
        for key, part in item.items():
            if key in table_fields and isinstance(part, str) and part in table_names:
                out[str(key)] = table_names[part]
            elif key == "blob" and isinstance(part, str) and part in blob_names:
                out[str(key)] = blob_names[part]
            else:
                out[str(key)] = rewrite(part)
        return out

    return rewrite(value)


def _encode_plot_fit(value: Any, resources: _Resources, hint: str) -> Any:
    """Use the analysis module's closure-free format for distribution fits."""

    from nanoscope.analysis.stats import FitResult, fit_to_json

    if isinstance(value, FitResult):
        return {"format": "stats-fit-json", "value": fit_to_json(value)}
    return {"format": "workspace-value", "value": _encode(value, resources, hint)}


def _decode_plot_fit(value: Any, artifact: WorkspaceArtifact) -> Any:
    if isinstance(value, Mapping) and value.get("format") == "stats-fit-json":
        from nanoscope.analysis.stats import fit_from_json

        return fit_from_json(dict(value.get("value") or {}))
    if isinstance(value, Mapping) and value.get("format") == "workspace-value":
        value = value.get("value")
    return _decode(value, artifact.tables, artifact.blobs)


# -------------------------------------------------------------------------------------------------
# Plots: whole PlotConfig + current PCA and current GMM/k-means result.
# -------------------------------------------------------------------------------------------------


def capture_plots(window: Any) -> WorkspaceArtifact:
    resources = _Resources()
    config = window.config()
    frame = _copy_frame_from_plot(window)
    source_map = dict(getattr(window, "_sources", {}) or {})
    selected_recipes = [
        _plot_source_recipe(window, source_map[key])
        for key in window.selected_source_keys() if key in source_map
    ]
    carried = getattr(window, "_nanoscope_unresolved_plot_sources", ())
    known = {(r.get("source_key"), r.get("channel_name"), r.get("channel_kind"),
              r.get("module_id"), r.get("kind")) for r in selected_recipes}
    for raw in carried if isinstance(carried, (list, tuple)) else ():
        if not isinstance(raw, Mapping):
            continue
        ident = (raw.get("source_key"), raw.get("channel_name"), raw.get("channel_kind"),
                 raw.get("module_id"), raw.get("kind"))
        if ident not in known:
            selected_recipes.append(_clone_plain(dict(raw)))
            known.add(ident)
    payload: dict[str, Any] = {
        "config": _encode(config, resources, "config"),
        "selected_sources": selected_recipes,
        "source_frame_fingerprint": fingerprint_frame(frame),
        "cluster": None,
        "pca": None,
        "fits": [],
    }

    for key, fit in dict(getattr(window, "_fits", {}) or {}).items():
        source = source_map.get(key)
        if source is None:
            continue
        payload["fits"].append({
            "source": _plot_source_recipe(window, source),
            "column": str(dict(getattr(window, "_fit_columns", {}) or {}).get(key, "")),
            "fit": _encode_plot_fit(fit, resources, f"fits.{len(payload['fits'])}"),
        })

    cluster = window.cluster_result()
    if cluster is not None:
        labels_name = resources.table(cluster.labels.to_frame("assignment"), "cluster.assignments")
        means_name = resources.table(cluster.means, "cluster.means")
        payload["cluster"] = {
            "labels_table": labels_name,
            "means_table": means_name,
            "k": int(cluster.k),
            "method": str(cluster.method),
            "columns": list(cluster.columns),
            "sizes": _encode(cluster.sizes, resources, "cluster.sizes"),
            "bic": _encode(cluster.bic, resources, "cluster.bic"),
            "chosen_k": int(cluster.chosen_k),
            "n_kept": int(cluster.n_kept),
            "warnings": list(cluster.warnings),
            "meta": _encode(cluster.meta, resources, "cluster.meta"),
            "random_state": (cluster.meta or {}).get("random_state"),
            "preprocessing": _preprocessing(frame, cluster),
        }

    pca = window.pca_result()
    if pca is not None:
        n_components = int(np.asarray(pca.scores).shape[1])
        pc_names = [f"PC{i + 1}" for i in range(n_components)]
        scores = pd.DataFrame(np.asarray(pca.scores), index=pca.kept_index, columns=pc_names)
        loadings = pd.DataFrame(np.asarray(pca.components).T,
                                index=pd.Index(list(pca.columns), name="feature"),
                                columns=pc_names)
        payload["pca"] = {
            "scores_table": resources.table(scores, "pca.scores"),
            "loadings_table": resources.table(loadings, "pca.loadings"),
            "explained_variance_ratio": np.asarray(pca.explained_variance_ratio,
                                                    dtype=np.float64).tolist(),
            "columns": list(pca.columns),
            "n_kept": int(pca.n_kept),
            "meta": _encode(pca.meta, resources, "pca.meta"),
            "random_state": (pca.meta or {}).get("random_state"),
            "preprocessing": _preprocessing(frame, pca),
        }
    carried = getattr(window, "_nanoscope_unresolved_plot_results", None)
    if isinstance(carried, Mapping) and isinstance(carried.get("artifact"), WorkspaceArtifact):
        wanted: dict[str, Any] = {}
        if payload["cluster"] is None and carried.get("cluster"):
            wanted["cluster"] = carried["cluster"]
        if payload["pca"] is None and carried.get("pca"):
            wanted["pca"] = carried["pca"]
        if carried.get("fits"):
            wanted["fits"] = list(carried["fits"])
        if wanted:
            copied = _copy_resource_tree(wanted, carried["artifact"], resources, "carried")
            for name in ("cluster", "pca"):
                if name in copied:
                    payload[name] = copied[name]
            payload["fits"].extend(copied.get("fits", ()))
    return _artifact(PLOTS_ARTIFACT_ID, payload, resources)


def restore_plots(window: Any, artifact: WorkspaceArtifact) -> RestoreReport:
    ok, reason = _artifact_ok(artifact, PLOTS_ARTIFACT_ID)
    if not ok:
        return RestoreReport(skipped=(PLOTS_ARTIFACT_ID,), warnings=(reason,))
    payload = artifact.payload
    warnings: list[str] = []
    try:
        current = window.config()
        config = _decode_like(current, payload["config"], artifact)
        recipes = payload.get("selected_sources")
        unresolved: list[dict[str, Any]] = []
        if isinstance(recipes, (list, tuple)):
            live_keys: list[str] = []
            for raw in recipes:
                if not isinstance(raw, Mapping):
                    unresolved.append({})
                    warnings.append("one saved plot source has a malformed identity recipe")
                    continue
                key, source_warning = _resolve_plot_recipe(window, raw)
                if key is None:
                    unresolved.append(_clone_plain(dict(raw)))
                else:
                    live_keys.append(key)
                if source_warning:
                    warnings.append(source_warning)
            # The dataclass type comes from the live window; no GUI import is needed.
            config = replace(config, selection=replace(
                config.selection, source_keys=tuple(live_keys)))
            setattr(window, "_nanoscope_unresolved_plot_sources", unresolved)
        install_config = getattr(window, "_set_config", None)
        if callable(install_config):
            # Restore is not a fresh Run gesture.  The window's one writer supports a
            # draw-free install; frozen outputs are attached below before a draw-only
            # workspace renderer is invited to paint them.
            try:
                install_config(config, draw=False, own=True)
            except TypeError:
                install_config(config, draw=False)
        else:
            window.apply_config(config)
    except Exception as exc:  # a bad artifact must not prevent the project opening
        return RestoreReport(skipped=(PLOTS_ARTIFACT_ID,),
                             warnings=(f"could not restore plot configuration: {exc}",))

    saved_fingerprint = payload.get("source_frame_fingerprint")
    current_fingerprint = fingerprint_frame(_copy_frame_from_plot(window))
    install_results = bool(saved_fingerprint and saved_fingerprint == current_fingerprint)
    carry_cluster = carry_pca = None
    if (payload.get("cluster") or payload.get("pca")) and not install_results:
        carry_cluster = payload.get("cluster")
        carry_pca = payload.get("pca")
        warnings.append(
            "plot PCA/clustering outputs were not restored because their selected source data "
            "are unavailable or changed; the plot configuration was restored"
        )

    if install_results:
        try:
            # Imported only on an actual GUI-thread restore; importing this module itself
            # remains Qt-free.
            from nanoscope.analysis.populations import ClusterResult, PCAResult

            raw_cluster = payload.get("cluster")
            if isinstance(raw_cluster, Mapping):
                labels = artifact.tables[str(raw_cluster["labels_table"])].iloc[:, 0].copy()
                labels.name = None
                window._cluster_result = ClusterResult(
                    labels=labels,
                    k=int(raw_cluster["k"]),
                    method=str(raw_cluster["method"]),
                    columns=[str(v) for v in raw_cluster["columns"]],
                    sizes=_decode(raw_cluster["sizes"], artifact.tables, artifact.blobs),
                    means=artifact.tables[str(raw_cluster["means_table"])].copy(deep=True),
                    bic=_decode(raw_cluster.get("bic"), artifact.tables, artifact.blobs),
                    chosen_k=int(raw_cluster["chosen_k"]),
                    n_kept=int(raw_cluster["n_kept"]),
                    warnings=tuple(str(v) for v in raw_cluster.get("warnings", ())),
                    meta=_decode(raw_cluster.get("meta", {}), artifact.tables, artifact.blobs),
                )
            raw_pca = payload.get("pca")
            if isinstance(raw_pca, Mapping):
                scores = artifact.tables[str(raw_pca["scores_table"])].copy(deep=True)
                loadings = artifact.tables[str(raw_pca["loadings_table"])].copy(deep=True)
                window._pca_result = PCAResult(
                    scores=scores.to_numpy(copy=True),
                    kept_index=scores.index.copy(),
                    explained_variance_ratio=np.asarray(raw_pca["explained_variance_ratio"],
                                                        dtype=np.float64),
                    components=loadings.to_numpy(copy=True).T,
                    columns=[str(v) for v in raw_pca["columns"]],
                    n_kept=int(raw_pca["n_kept"]),
                    meta=_decode(raw_pca.get("meta", {}), artifact.tables, artifact.blobs),
                )
        except Exception as exc:
            # A missing/malformed Parquet resource is local to derived output; the
            # independently valid controls and source selection remain restored.
            window._cluster_result = None
            window._pca_result = None
            warnings.append(f"plot configuration restored but PCA/clustering output was invalid: {exc}")

    # Distribution-fit values are installed *after* any compatibility redraw above so a
    # redraw cannot overwrite the frozen values.  A GUI draw-only seam, when present,
    # renders these without invoking a solver.
    restored_fits: dict[str, Any] = {}
    restored_columns: dict[str, str] = {}
    carried_fits: list[Any] = []
    for raw in payload.get("fits", ()) if isinstance(payload.get("fits", ()), (list, tuple)) else ():
        try:
            recipe = raw["source"]
            key, source_warning = _resolve_plot_recipe(window, recipe, require_same_rows=True)
            if key is None:
                carried_fits.append(_clone_plain(raw))
                if source_warning:
                    warnings.append(source_warning)
                continue
            restored_fits[key] = _decode_plot_fit(raw["fit"], artifact)
            restored_columns[key] = str(raw.get("column", ""))
        except Exception as exc:
            warnings.append(f"one saved plot fit was invalid and was skipped: {exc}")
    if restored_fits:
        window._fits = restored_fits
        window._fit_columns = restored_columns
        primary = next((key for key in window.selected_source_keys() if key in restored_fits), None)
        window._last_fit = restored_fits.get(primary)
    if install_results or restored_fits:
        render = getattr(window, "_render_workspace_snapshot", None)
        if callable(render):
            render()
    if carry_cluster is not None or carry_pca is not None or carried_fits:
        setattr(window, "_nanoscope_unresolved_plot_results", {
            "artifact": _clone_artifact(artifact),
            "cluster": _clone_plain(carry_cluster),
            "pca": _clone_plain(carry_pca),
            "fits": _clone_plain(carried_fits),
        })
    elif hasattr(window, "_nanoscope_unresolved_plot_results"):
        delattr(window, "_nanoscope_unresolved_plot_results")
    return RestoreReport(restored=(PLOTS_ARTIFACT_ID,), warnings=tuple(warnings))


# -------------------------------------------------------------------------------------------------
# Full subpopulation report and its report-window view state.
# -------------------------------------------------------------------------------------------------


def capture_subpopulation(window: Any) -> WorkspaceArtifact:
    resources = _Resources()
    report = window.report()
    view = window.embedding_view()
    assignment = window.assignment_state()
    header = window.header() if callable(getattr(window, "header", None)) else ""
    subtitle = header.split("\n", 1)[0] if "\n" in header else ""
    source_result = getattr(window, "_nanoscope_workspace_source_result", None)
    carried = getattr(window, "_nanoscope_workspace_source", None)
    if isinstance(carried, Mapping) and carried.get("source_key"):
        # This is the exact recipe MainWindow captured while launching the report
        # job (or restored with the report).  It remains authoritative if the live
        # result disappears or is replaced before the asynchronous job lands.
        source = _clone_plain(dict(carried))
    elif source_result is not None:
        live_source = _result_recipe(window, source_result)
        # A report worker can finish after its source/result was closed or replaced.
        # MainWindow captures the public ``result_recipe`` at job launch and installs
        # it as this sidecar; prefer it when a late live lookup can no longer produce
        # persistent identity.  This keeps the completed, self-contained report tied
        # to its real lineage without ever falling back to a launch-local doc id.
        source = live_source
    else:
        source = _clone_plain(dict(carried)) if isinstance(carried, Mapping) else None
    accepted_ids = getattr(window, "_nanoscope_workspace_accepted_event_ids", None)
    if accepted_ids is not None:
        accepted_ids = sorted({int(value) for value in accepted_ids})
    payload = {
        "source": source,
        "accepted_event_ids": accepted_ids,
        "report": _encode(report, resources, "report"),
        "report_features_fingerprint": fingerprint_frame(
            getattr(report, "features", None) if report is not None else None),
        "subtitle": subtitle,
        "view": {
            "embedding": view.embedding() if callable(getattr(view, "embedding", None)) else "",
            "colour_mode": view.colour_mode() if callable(getattr(view, "colour_mode", None)) else "",
            "splitter_sizes": list(window.splitter().sizes())
            if callable(getattr(window, "splitter", None)) else [],
            "figure_splitter_sizes": list(window.figures_splitter().sizes())
            if callable(getattr(window, "figures_splitter", None)) else [],
        },
        "assignment_state": str(assignment[0]),
        "assignment_reason": str(assignment[1]),
        "curation_note": str(window.curation_note()),
        "appearance": _capture_appearance(window),
    }
    return _artifact(SUBPOPULATION_ARTIFACT_ID, payload, resources)


def restore_subpopulation(window: Any, artifact: WorkspaceArtifact) -> RestoreReport:
    ok, reason = _artifact_ok(artifact, SUBPOPULATION_ARTIFACT_ID)
    if not ok:
        return RestoreReport(skipped=(SUBPOPULATION_ARTIFACT_ID,), warnings=(reason,))
    try:
        payload = artifact.payload
        source = payload.get("source")
        if isinstance(source, Mapping):
            setattr(window, "_nanoscope_workspace_source", _clone_plain(dict(source)))
        accepted_ids = payload.get("accepted_event_ids")
        setattr(window, "_nanoscope_workspace_accepted_event_ids",
                None if accepted_ids is None else tuple(sorted({int(v) for v in accepted_ids})))
        report = _decode(payload.get("report"), artifact.tables, artifact.blobs)
        window.set_report(report, subtitle=str(payload.get("subtitle") or ""))
        view_state = payload.get("view", {})
        view = window.embedding_view()
        embedding = str(view_state.get("embedding") or "")
        colour = str(view_state.get("colour_mode") or "")
        if embedding:
            view.set_embedding(embedding)
        if colour:
            view.set_colour_mode(colour)
        if view_state.get("splitter_sizes") and callable(getattr(window, "splitter", None)):
            window.splitter().setSizes([int(v) for v in view_state["splitter_sizes"]])
        if (view_state.get("figure_splitter_sizes")
                and callable(getattr(window, "figures_splitter", None))):
            window.figures_splitter().setSizes(
                [int(v) for v in view_state["figure_splitter_sizes"]])
        window.set_assignment_state(str(payload.get("assignment_state") or "none"),
                                    str(payload.get("assignment_reason") or ""))
        window.set_curation_note(str(payload.get("curation_note") or ""))
        _restore_appearance(window, payload)
    except Exception as exc:
        return RestoreReport(skipped=(SUBPOPULATION_ARTIFACT_ID,),
                             warnings=(f"could not restore subpopulation report: {exc}",))
    return RestoreReport(restored=(SUBPOPULATION_ARTIFACT_ID,))


# -------------------------------------------------------------------------------------------------
# PSD / FFT / spectrogram / all-points state.
# -------------------------------------------------------------------------------------------------


def _psd_target_key(window: Any) -> str | None:
    getter = getattr(window, "_current_target", None)
    target = getter() if callable(getter) else None
    for record in list(getattr(window, "_recordings", ())):
        if target is not None and (record.get("doc_id"), record.get("channel")) == tuple(target):
            return str(record.get("key") or "") or None
    return None


def _psd_record_recipe(window: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    return _stable_source(window, str(record.get("doc_id") or ""),
                          None if record.get("channel") is None else int(record["channel"]))


def _psd_record_for_recipe(window: Any, recipe: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    doc_id, channel, note = _resolve_source(window, recipe)
    if doc_id is None or channel is None:
        return None, note
    record = next((item for item in list(getattr(window, "_recordings", ()))
                   if str(item.get("doc_id")) == doc_id and int(item.get("channel", -1)) == channel),
                  None)
    return record, note if record is not None else f"resolved PSD source {recipe.get('source_key')!r} is not offered"


def _capture_appearance(window: Any) -> dict:
    """Optional presentation overrides; never part of a numerical recipe."""
    state = getattr(window, "_appearance", None)
    encode = getattr(state, "to_dict", None)
    return encode() if callable(encode) else {}


def _restore_appearance(window: Any, payload: Mapping) -> None:
    restore = getattr(window, "set_appearance_state", None)
    if callable(restore):
        value = payload.get("appearance", {})
        restore(value if isinstance(value, Mapping) else {})


def capture_psd(window: Any) -> WorkspaceArtifact:
    resources = _Resources()
    style = window.style()
    remembered_curve_list = [item for item in
                             (getattr(window, "_nanoscope_workspace_psd_curves", ()) or ())
                             if isinstance(item, Mapping)]
    remembered_curves = {
        str(item.get("key") or ""): item
        for item in remembered_curve_list
    }
    curves: list[dict[str, Any]] = []
    for index, curve in enumerate(window.curves()):
        table = pd.DataFrame({"frequency_hz": np.asarray(curve.freqs),
                              "psd": np.asarray(curve.psd)})
        calibration_fingerprint = None
        try:
            calibration_fingerprint = list(
                window._session.calibration(curve.doc_id, int(curve.channel)).fingerprint())
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        identity = _stable_source(window, str(curve.doc_id), int(curve.channel))
        if not identity["source_key"]:
            carried = remembered_curves.get(str(curve.key))
            if carried is None and index < len(remembered_curve_list):
                carried = remembered_curve_list[index]
            if isinstance(carried, Mapping):
                identity = {name: carried.get(name) for name in
                            ("source_key", "channel", "channel_name", "channel_kind")}
        curves.append({
            "table": resources.table(table, f"curves.{index}"),
            "key": str(curve.key),
            **identity,
            "channel": int(curve.channel), "label": str(curve.label),
            "colour": str(curve.colour), "summary": _encode(curve.summary, resources, f"curves.{index}.summary"),
            "units": str(curve.units), "fs": float(curve.fs), "nperseg": int(curve.nperseg),
            "noverlap": int(curve.noverlap), "nseg": int(curve.nseg),
            "n_samples": int(curve.n_samples), "i0": int(curve.i0), "note": str(curve.note),
            "channel_issues": list(curve.channel_issues), "constant": bool(curve.constant),
            "underflow": bool(curve.underflow), "time_rms": float(curve.time_rms),
            "calibration_fingerprint": calibration_fingerprint,
        })
    target = getattr(window, "_current_target", lambda: None)()
    target_record = next((record for record in list(getattr(window, "_recordings", ()))
                          if target is not None and
                          (record.get("doc_id"), record.get("channel")) == tuple(target)), None)
    records_by_key = {str(record.get("key")): record
                      for record in list(getattr(window, "_recordings", ())) }
    target_source = (_psd_record_recipe(window, target_record)
                     if target_record is not None else None)
    if target_source is None:
        carried_target = getattr(window, "_nanoscope_workspace_psd_target", None)
        if isinstance(carried_target, Mapping):
            target_source = _clone_plain(dict(carried_target))
    compare_sources = [
        _psd_record_recipe(window, records_by_key[key])
        for key in window.compare_keys() if key in records_by_key
    ]
    for carried in getattr(window, "_nanoscope_unresolved_psd_compare", ()) or ():
        if isinstance(carried, Mapping):
            compare_sources.append(_clone_plain(dict(carried)))
    payload = {
        "config": {
            "target_key": _psd_target_key(window),
            "compare_keys": list(window.compare_keys()),
            "target_source": target_source,
            "compare_sources": compare_sources,
            "start_s": float(_widget_value(window, "_start_spin", "value", 0.0)),
            "end_s": float(_widget_value(window, "_end_spin", "value", 0.0)),
            "nperseg": str(_widget_value(window, "_nperseg_combo", "currentText", "Auto")),
            "window": str(_widget_value(window, "_window_combo", "currentText", "hann")),
            "overlap_percent": int(_widget_value(window, "_overlap_spin", "value", 50)),
            "log_log": bool(_widget_value(window, "_loglog_check", "isChecked", True)),
            "asd": bool(_widget_value(window, "_asd_check", "isChecked", False)),
            "rms_lo_hz": float(_widget_value(window, "_rms_lo_spin", "value", 0.0)),
            "rms_hi_hz": float(_widget_value(window, "_rms_hi_spin", "value", 0.0)),
            "view": str(window.current_view()),
            "spectrogram_nperseg": str(_widget_value(window, "_spec_nperseg_combo", "currentText", "Auto")),
            "spectrogram_colormap": str(_widget_value(window, "_spec_cmap_combo", "currentText", "viridis")),
            "histogram_bins": str(_widget_value(window, "_hist_bins_combo", "currentText", "256")),
            "histogram_clip": bool(_widget_value(window, "_hist_clip_check", "isChecked", False)),
            "histogram_levels": bool(_widget_value(window, "_hist_levels_check", "isChecked", True)),
            "appearance": _capture_appearance(window),
            "style": {
                "title": str(style.title), "data_color": str(style.data_color),
                "fit_color": str(style.fit_color), "background": str(style.background),
                "export_w": int(style.export_w), "export_h": int(style.export_h),
            },
        },
        "curves": curves,
        "fft": _encode(window.fft_spectrum(), resources, "fft"),
        "spectrogram": _encode(window.spectrogram(), resources, "spectrogram"),
        "all_points": _encode(window.all_points(), resources, "all_points"),
        "histogram_levels": _encode(window.hist_levels(), resources, "histogram_levels"),
        "compute_notes": [str(v) for v in getattr(window, "_compute_notes", ())],
    }
    return _artifact(PSD_ARTIFACT_ID, payload, resources)


def _set_combo_text(owner: Any, attr: str, text: str) -> None:
    widget = getattr(owner, attr, None)
    setter = getattr(widget, "setCurrentText", None)
    if callable(setter):
        setter(str(text))


def _set_checked(owner: Any, attr: str, value: Any) -> None:
    widget = getattr(owner, attr, None)
    setter = getattr(widget, "setChecked", None)
    if callable(setter):
        setter(bool(value))


def _blocked(widget: Any, action: Any) -> None:
    """Run one widget mutation without emitting a compute-triggering signal."""

    blocker = getattr(widget, "blockSignals", None)
    if not callable(blocker):
        action()
        return
    previous = blocker(True)
    try:
        action()
    finally:
        blocker(previous)


def restore_psd(window: Any, artifact: WorkspaceArtifact) -> RestoreReport:
    ok, reason = _artifact_ok(artifact, PSD_ARTIFACT_ID)
    if not ok:
        return RestoreReport(skipped=(PSD_ARTIFACT_ID,), warnings=(reason,))
    payload = artifact.payload
    config = payload.get("config", {})
    warnings: list[str] = []
    try:
        # All controls are restored with signals suppressed.  Calling the public setters
        # here would launch Welch/spectrogram/histogram kernels repeatedly and replace the
        # durable snapshot before the user asked for a new run.
        previous_updating = bool(getattr(window, "_updating", False))
        window._updating = True
        target_key = config.get("target_key")
        target_record = None
        target_recipe = config.get("target_source")
        if isinstance(target_recipe, Mapping):
            setattr(window, "_nanoscope_workspace_psd_target", _clone_plain(dict(target_recipe)))
            target_record, target_note = _psd_record_for_recipe(window, target_recipe)
            if target_note:
                warnings.append(target_note)
            if target_record is not None:
                select = getattr(window, "_select_combo_target", None)
                if callable(select):
                    select((target_record["doc_id"], int(target_record["channel"])))
        elif target_key:
            target_record = next((r for r in list(getattr(window, "_recordings", ()))
                                  if r.get("key") == target_key), None)
            if target_record is not None:
                select = getattr(window, "_select_combo_target", None)
                if callable(select):
                    select((target_record["doc_id"], int(target_record["channel"])))
            else:
                warnings.append(f"PSD target {target_key!r} is not open")

        getattr(window, "_start_spin").setValue(float(config.get("start_s", 0.0)))
        getattr(window, "_end_spin").setValue(float(config.get("end_s", 0.0)))
        nperseg = str(config.get("nperseg", "Auto"))
        nperseg_combo = getattr(window, "_nperseg_combo")
        if nperseg_combo.findText(nperseg) < 0:
            nperseg_combo.addItem(nperseg)
        nperseg_combo.setCurrentText(nperseg)
        getattr(window, "_window_combo").setCurrentText(str(config.get("window", "hann")))
        getattr(window, "_overlap_spin").setValue(int(config.get("overlap_percent", 50)))
        getattr(window, "_loglog_check").setChecked(bool(config.get("log_log", True)))
        getattr(window, "_asd_check").setChecked(bool(config.get("asd", False)))
        getattr(window, "_rms_lo_spin").setValue(float(config.get("rms_lo_hz", 0.0)))
        getattr(window, "_rms_hi_spin").setValue(float(config.get("rms_hi_hz", 0.0)))
        style = config.get("style", {})
        live_style = window.style()
        live_style.title = str(style.get("title", ""))
        live_style.data_color = str(style.get("data_color", live_style.data_color))
        live_style.fit_color = str(style.get("fit_color", live_style.fit_color))
        live_style.background = str(style.get("background", "dark"))
        live_style.export_w = int(style.get("export_w", 0))
        live_style.export_h = int(style.get("export_h", 0))
        getattr(window, "_title_edit").setText(live_style.title)
        getattr(window, "_bg_combo").setCurrentText(
            "Light" if live_style.background.lower() == "light" else "Dark")
        getattr(window, "_export_w").setValue(live_style.export_w)
        getattr(window, "_export_h").setValue(live_style.export_h)
        paint = getattr(window, "_paint_swatch", None)
        if callable(paint):
            paint(getattr(window, "_data_swatch"), live_style.data_color)
            paint(getattr(window, "_fit_swatch"), live_style.fit_color)
        _set_combo_text(window, "_spec_nperseg_combo", config.get("spectrogram_nperseg", "Auto"))
        _set_combo_text(window, "_spec_cmap_combo", config.get("spectrogram_colormap", "viridis"))
        _set_combo_text(window, "_hist_bins_combo", config.get("histogram_bins", "256"))
        _set_checked(window, "_hist_clip_check", config.get("histogram_clip", False))
        _set_checked(window, "_hist_levels_check", config.get("histogram_levels", True))
        _restore_appearance(window, config)

        wanted_compare: set[str] = set()
        unresolved_compare: list[dict[str, Any]] = []
        compare_recipes = config.get("compare_sources", ())
        if isinstance(compare_recipes, (list, tuple)):
            for raw in compare_recipes:
                if not isinstance(raw, Mapping):
                    warnings.append("one PSD comparison source has a malformed identity recipe")
                    continue
                record, compare_note = _psd_record_for_recipe(window, raw)
                if record is None:
                    unresolved_compare.append(_clone_plain(dict(raw)))
                else:
                    wanted_compare.add(str(record.get("key")))
                if compare_note:
                    warnings.append(compare_note)
            setattr(window, "_nanoscope_unresolved_psd_compare", unresolved_compare)
        else:
            wanted_compare = {str(v) for v in config.get("compare_keys", ())}
        compare_list = getattr(window, "_compare_list", None)
        if compare_list is not None:
            for index in range(compare_list.count()):
                item = compare_list.item(index)
                state_type = type(item.checkState())
                checked = getattr(state_type, "Checked", 2)
                unchecked = getattr(state_type, "Unchecked", 0)
                item.setCheckState(checked if str(item.data(256)) in wanted_compare else unchecked)

        view_name = str(config.get("view", "psd"))
        view_combo = getattr(window, "_view_combo")
        for index in range(view_combo.count()):
            if view_combo.itemData(index) == view_name:
                view_combo.setCurrentIndex(index)
                break
        window._updating = previous_updating

        # Rehydrate the exact numerical snapshot.  This local GUI import is intentional:
        # module import and worker-side project serialization remain Qt-free, while restore
        # runs on the GUI thread and needs the window's own curve value class.
        from nanoscope.gui.psd_panel import PSDCurve, VIEW_ALLPOINTS, VIEW_FFT, VIEW_SPECTROGRAM

        curves = []
        setattr(window, "_nanoscope_workspace_psd_curves",
                [_clone_plain(dict(raw)) for raw in payload.get("curves", ())
                 if isinstance(raw, Mapping)])
        for raw_curve in payload.get("curves", ()):
            table = artifact.tables[str(raw_curve["table"])]
            source_key = str(raw_curve.get("source_key") or "")
            live_doc_id, live_channel, source_note = _resolve_source(window, raw_curve)
            if live_doc_id is None or live_channel is None:
                warnings.append(
                    f"PSD source {source_key or raw_curve.get('label', '')!r} is not open; "
                    "the saved numerical snapshot remains available")
                live_doc_id = ""
                live_channel = int(raw_curve["channel"])
                live_key = str(raw_curve["key"])
            else:
                live_record = next((item for item in list(getattr(window, "_recordings", ()))
                                    if str(item.get("doc_id")) == live_doc_id
                                    and int(item.get("channel", -1)) == live_channel), None)
                live_key = str(live_record.get("key")) if live_record is not None else str(raw_curve["key"])
                if source_note:
                    warnings.append(source_note)
            curve = PSDCurve(
                key=live_key, doc_id=live_doc_id,
                channel=live_channel, label=str(raw_curve["label"]),
                colour=str(raw_curve["colour"]),
                freqs=table["frequency_hz"].to_numpy(copy=True),
                psd=table["psd"].to_numpy(copy=True),
                summary=_decode(raw_curve.get("summary", {}), artifact.tables, artifact.blobs),
                units=str(raw_curve.get("units", "")), fs=float(raw_curve.get("fs", 0.0)),
                nperseg=int(raw_curve.get("nperseg", 0)),
                noverlap=int(raw_curve.get("noverlap", 0)), nseg=int(raw_curve.get("nseg", 0)),
                n_samples=int(raw_curve.get("n_samples", 0)), i0=int(raw_curve.get("i0", 0)),
                note=str(raw_curve.get("note", "")),
                channel_issues=tuple(str(v) for v in raw_curve.get("channel_issues", ())),
                constant=bool(raw_curve.get("constant", False)),
                underflow=bool(raw_curve.get("underflow", False)),
                time_rms=float(raw_curve.get("time_rms", 0.0)),
            )
            curves.append(curve)
            saved_cal = raw_curve.get("calibration_fingerprint")
            if saved_cal is not None:
                try:
                    current_cal = list(window._session.calibration(
                        curve.doc_id, curve.channel).fingerprint())
                except (AttributeError, KeyError, TypeError, ValueError):
                    current_cal = None
                if current_cal != list(saved_cal):
                    warnings.append(
                        f"PSD snapshot for {curve.label!r} was computed under a different "
                        "calibration; saved numbers were restored without recomputing")

        window._curves = curves
        window._compute_notes = [str(v) for v in payload.get("compute_notes", ())]
        if curves:
            head = curves[0]
            window._freqs, window._psd, window._summary = head.freqs, head.psd, head.summary
            window._units, window._fs = head.units, head.fs
            window._nperseg_used, window._noverlap_used = head.nperseg, head.noverlap
            window._nseg_used = head.nseg
            update_segment = getattr(window, "_update_seg_label", None)
            if callable(update_segment):
                update_segment(head.i0, head.i0 + head.n_samples, head.n_samples)
        else:
            window._freqs = window._psd = window._summary = None
            window._nperseg_used = window._noverlap_used = window._nseg_used = 0

        window._fft = _decode(payload.get("fft"), artifact.tables, artifact.blobs)
        window._spec = _decode(payload.get("spectrogram"), artifact.tables, artifact.blobs)
        window._hist = _decode(payload.get("all_points"), artifact.tables, artifact.blobs)
        saved_levels = _decode(payload.get("histogram_levels", []), artifact.tables, artifact.blobs)
        window._hist_levels = list(saved_levels)

        # Seed each view's existing cache signature with the saved result.  `_refresh_view`
        # then performs drawing only; none of fft_magnitude/spectrogram_psd/
        # all_points_histogram is called.
        window._view_cache = {}
        record = target_record
        span = window._span_for(record) if record is not None else None
        if span is not None and not isinstance(span, str):
            for name, result in ((VIEW_FFT, window._fft),
                                 (VIEW_SPECTROGRAM, window._spec),
                                 (VIEW_ALLPOINTS, window._hist)):
                if result is not None:
                    window._view_cache[name] = (window._view_signature(name, record, span), result)
        render = getattr(window, "_render_workspace_snapshot", None)
        if callable(render):
            render()
        else:
            # Compatibility with a window implementation predating the explicit snapshot
            # seam.  This fallback is safe only for the PSD page or a cache-backed live
            # span; modern windows take the branch above and can also paint missing-source
            # FFT/spectrogram/histogram snapshots without any kernel.
            redraw = getattr(window, "_redraw", None)
            if callable(redraw):
                redraw()
            refresh_view = getattr(window, "_refresh_view", None)
            if callable(refresh_view) and (view_name == "psd" or span is not None):
                refresh_view()
                window._hist_levels = list(saved_levels)
    except Exception as exc:
        if hasattr(window, "_updating"):
            window._updating = False
        return RestoreReport(skipped=(PSD_ARTIFACT_ID,),
                             warnings=(f"could not restore PSD configuration: {exc}",))
    return RestoreReport(restored=(PSD_ARTIFACT_ID,), warnings=tuple(warnings))


# -------------------------------------------------------------------------------------------------
# Density / ensemble event plots.
# -------------------------------------------------------------------------------------------------


def capture_event_plots(window: Any) -> WorkspaceArtifact:
    resources = _Resources()
    source = _event_window_source_recipe(window)
    amplitude_snapshot = bool(getattr(window, "_amplitude_frozen_snapshot", False))
    if source is None or amplitude_snapshot:
        carried = getattr(window, "_nanoscope_workspace_event_source", None)
        if isinstance(carried, Mapping):
            source = _clone_plain(dict(carried))
    edge_getter = getattr(window, "density_edges", None)
    density_edges = edge_getter() if callable(edge_getter) else None
    accepted_count = getattr(window, "_nanoscope_workspace_accepted_event_count", None)
    eventset = getattr(window, "_eventset", None)
    if eventset is not None and not amplitude_snapshot:
        try:
            accepted_count = int(len(eventset.accepted.df))
        except (AttributeError, TypeError, ValueError):
            pass
    payload = {
        "source": source,
        "units": str(getattr(window, "_units", "") or ""),
        "accepted_event_count": (None if accepted_count is None else int(accepted_count)),
        "config": {
            "view": str(window.current_view()),
            "x_metric": str(_widget_value(window, "_x_combo", "currentData", "dwell_s")),
            "y_metric": str(_widget_value(window, "_y_combo", "currentData", "depth_pA")),
            "log_x": bool(_widget_value(window, "_logx", "isChecked", True)),
            "log_y": bool(_widget_value(window, "_logy", "isChecked", False)),
            "bins": int(_widget_value(window, "_bins", "value", 40)),
            "colormap": str(_widget_value(window, "_cmap_combo", "currentText", "viridis")),
            "density_title": str(_widget_value(window, "_d_title", "text", "")),
            "density_x_label": str(_widget_value(window, "_d_xlabel", "text", "")),
            "density_y_label": str(_widget_value(window, "_d_ylabel", "text", "")),
            "normalize_time": bool(_widget_value(window, "_normalize", "isChecked", True)),
            "align": str(_widget_value(window, "_align_combo", "currentText", "start")),
            "n_points": int(_widget_value(window, "_npoints", "value", 120)),
            "pad_percent": int(_widget_value(window, "_pad", "value", 20)),
            "ensemble_title": str(_widget_value(window, "_e_title", "text", "")),
            "appearance": _capture_appearance(window),
        },
        "density_counts": _encode(window.density_counts(), resources, "density.counts"),
        "amplitude_warnings": list(getattr(window, "_amplitude_warnings", ())),
        "amplitude_snapshot_units": str(getattr(window, "_amplitude_snapshot_units", "")),
        "density_edges": _encode(density_edges, resources, "density.edges"),
        "ensemble": _encode(window.ensemble_traces(), resources, "ensemble"),
    }
    return _artifact(EVENT_PLOTS_ARTIFACT_ID, payload, resources)


def _set_text(owner: Any, attr: str, value: Any) -> None:
    widget = getattr(owner, attr, None)
    setter = getattr(widget, "setText", None)
    if callable(setter):
        setter(str(value or ""))


def restore_event_plots(window: Any, artifact: WorkspaceArtifact) -> RestoreReport:
    ok, reason = _artifact_ok(artifact, EVENT_PLOTS_ARTIFACT_ID)
    if not ok:
        return RestoreReport(skipped=(EVENT_PLOTS_ARTIFACT_ID,), warnings=(reason,))
    config = artifact.payload.get("config", {})
    warnings: list[str] = []
    try:
        previous_updating = bool(getattr(window, "_updating", False))
        window._updating = True
        _blocked(getattr(window, "_x_combo"),
                 lambda: window.set_x_metric(str(config.get("x_metric", "dwell_s"))))
        _blocked(getattr(window, "_y_combo"),
                 lambda: window.set_y_metric(str(config.get("y_metric", "depth_pA"))))
        _blocked(getattr(window, "_logx"),
                 lambda: getattr(window, "_logx").setChecked(bool(config.get("log_x", True))))
        _blocked(getattr(window, "_logy"),
                 lambda: getattr(window, "_logy").setChecked(bool(config.get("log_y", False))))
        _blocked(getattr(window, "_bins"),
                 lambda: getattr(window, "_bins").setValue(int(config.get("bins", 40))))
        _blocked(getattr(window, "_cmap_combo"),
                 lambda: getattr(window, "_cmap_combo").setCurrentText(
                     str(config.get("colormap", "viridis"))))
        _blocked(getattr(window, "_normalize"),
                 lambda: getattr(window, "_normalize").setChecked(
                     bool(config.get("normalize_time", True))))
        getattr(window, "_align_combo").setEnabled(
            not bool(config.get("normalize_time", True)))
        _blocked(getattr(window, "_align_combo"),
                 lambda: getattr(window, "_align_combo").setCurrentText(
                     str(config.get("align", "start"))))
        _blocked(getattr(window, "_npoints"),
                 lambda: getattr(window, "_npoints").setValue(int(config.get("n_points", 120))))
        _blocked(getattr(window, "_pad"),
                 lambda: getattr(window, "_pad").setValue(int(config.get("pad_percent", 20))))
        _set_text(window, "_d_title", config.get("density_title", ""))
        _set_text(window, "_d_xlabel", config.get("density_x_label", ""))
        _set_text(window, "_d_ylabel", config.get("density_y_label", ""))
        _set_text(window, "_e_title", config.get("ensemble_title", ""))
        tabs = getattr(window, "_tabs", None)
        set_view = getattr(window, "set_view", None)
        if callable(set_view):
            _blocked(tabs, lambda: set_view(str(config.get("view", "Density"))))
        else:
            _select_tab_by_text(tabs, str(config.get("view", "Density")))
        # Numerical state is installed verbatim.  Rebinding a source or pressing Refresh is
        # the explicit operation which is allowed to run histogram/ensemble kernels again.
        source = artifact.payload.get("source")
        if isinstance(source, Mapping):
            setattr(window, "_nanoscope_workspace_event_source", _clone_plain(dict(source)))
            live = _event_window_source_recipe(window)
            if live is None:
                warnings.append(
                    "event-plot snapshot was restored without its recording; its durable "
                    "source recipe was retained for a later rebind")
            else:
                same_recording = _same_persistent_source(
                    str(source.get("source_key") or ""), str(live.get("source_key") or ""))
                same_channel = (source.get("channel_name"), source.get("channel_kind")) == (
                    live.get("channel_name"), live.get("channel_kind"))
                same_module = (not source.get("module_id") or
                               source.get("module_id") == live.get("module_id"))
                same_rows = (not source.get("event_frame_fingerprint") or
                             source.get("event_frame_fingerprint") ==
                             live.get("event_frame_fingerprint"))
                if not (same_recording and same_channel and same_module and same_rows):
                    warnings.append(
                        "the live event source does not match the saved recording/module/result; "
                        "the frozen plot remains labelled by its saved source recipe")
        window._density_counts = _decode(artifact.payload.get("density_counts"),
                                         artifact.tables, artifact.blobs)
        amplitude_warnings = artifact.payload.get("amplitude_warnings", [])
        window._amplitude_warnings = tuple(str(w) for w in amplitude_warnings) if isinstance(
            amplitude_warnings, (tuple, list)) else ()
        window._amplitude_snapshot_units = str(artifact.payload.get("amplitude_snapshot_units", ""))
        window._units = str(artifact.payload.get("units", getattr(window, "_units", "")))
        accepted_count = artifact.payload.get("accepted_event_count")
        if accepted_count is not None:
            window._nanoscope_workspace_accepted_event_count = int(accepted_count)
        density_edges = _decode(artifact.payload.get("density_edges"),
                                artifact.tables, artifact.blobs)
        if density_edges is None:
            window._density_x_edges = window._density_y_edges = None
        else:
            window._density_x_edges, window._density_y_edges = (
                np.asarray(density_edges[0]).copy(), np.asarray(density_edges[1]).copy())
        ensemble = _decode(artifact.payload.get("ensemble"), artifact.tables, artifact.blobs)
        window._ensemble = None if ensemble is None else tuple(ensemble)
        _restore_appearance(window, config)
        render = getattr(window, "_render_workspace_snapshot", None)
        if callable(render):
            render()
        window._updating = previous_updating
    except Exception as exc:
        if hasattr(window, "_updating"):
            window._updating = False
        return RestoreReport(skipped=(EVENT_PLOTS_ARTIFACT_ID,),
                             warnings=(f"could not restore event-plot configuration: {exc}",))
    if not window.has_data():
        warnings.append("event-plot controls were restored, but MainWindow has not rebound an event source")
    return RestoreReport(restored=(EVENT_PLOTS_ARTIFACT_ID,), warnings=tuple(warnings))


# -------------------------------------------------------------------------------------------------
# I-V and voltage-dependence values.  These install exact saved analysis outputs; the windows'
# ordinary mutation/Refresh wiring replaces them when live session data changes.
# -------------------------------------------------------------------------------------------------


def capture_iv(window: Any) -> WorkspaceArtifact:
    resources = _Resources()
    points = []
    remembered_points = [v for v in (getattr(window, "_nanoscope_workspace_iv_points", ()) or ())
                         if isinstance(v, Mapping)]
    for index, point in enumerate(window.points()):
        identity = _stable_source(window, str(point.doc_id))
        if not identity["source_key"] and index < len(remembered_points):
            identity = {name: remembered_points[index].get(name) for name in
                        ("source_key", "channel", "channel_name", "channel_kind")}
        points.append({
            **identity,
            "v_volts": float(point.v_volts), "i_open": float(point.i_open),
            "current_units": str(point.current_units),
            "voltage_source": str(point.voltage_source), "label": str(point.label),
        })
    exclusions = []
    remembered_exclusions = [v for v in
                             (getattr(window, "_nanoscope_workspace_iv_exclusions", ()) or ())
                             if isinstance(v, Mapping)]
    for index, excluded in enumerate(window.exclusions()):
        identity = _stable_source(window, str(excluded.doc_id))
        if not identity["source_key"] and index < len(remembered_exclusions):
            identity = {name: remembered_exclusions[index].get(name) for name in
                        ("source_key", "channel", "channel_name", "channel_kind")}
        exclusions.append({
            **identity,
            "label": str(excluded.label), "v_value": float(excluded.v_value),
            "v_units": str(excluded.v_units), "i_open": float(excluded.i_open),
            "current_units": str(excluded.current_units), "reason": str(excluded.reason),
        })
    payload = {
        "points": points,
        "exclusions": exclusions,
        "appearance": _capture_appearance(window),
        "fit": _encode(window.fit(), resources, "fit"),
        "details_text": str(window.details_text()),
        "headline": str(_widget_value(window, "_headline", "text", "")),
    }
    return _artifact(IV_ARTIFACT_ID, payload, resources)


def restore_iv(window: Any, artifact: WorkspaceArtifact) -> RestoreReport:
    ok, reason = _artifact_ok(artifact, IV_ARTIFACT_ID)
    if not ok:
        return RestoreReport(skipped=(IV_ARTIFACT_ID,), warnings=(reason,))
    try:
        from nanoscope.analysis.iv import IVExclusion, IVPoint

        warnings: list[str] = []
        points = []
        setattr(window, "_nanoscope_workspace_iv_points",
                [_clone_plain(dict(v)) for v in artifact.payload.get("points", ())
                 if isinstance(v, Mapping)])
        setattr(window, "_nanoscope_workspace_iv_exclusions",
                [_clone_plain(dict(v)) for v in artifact.payload.get("exclusions", ())
                 if isinstance(v, Mapping)])
        for raw in artifact.payload.get("points", ()):
            doc_id, _channel, source_note = _resolve_source(window, raw)
            if doc_id is None:
                warnings.append(f"I-V source {raw.get('source_key')!r} is not open")
                doc_id = ""
            elif source_note:
                warnings.append(source_note)
            points.append(IVPoint(
                float(raw["v_volts"]), float(raw["i_open"]), str(raw["current_units"]),
                str(raw["voltage_source"]), doc_id, str(raw["label"])))
        exclusions = []
        for raw in artifact.payload.get("exclusions", ()):
            doc_id, _channel, source_note = _resolve_source(window, raw)
            if doc_id is None:
                warnings.append(f"I-V excluded source {raw.get('source_key')!r} is not open")
                doc_id = ""
            elif source_note:
                warnings.append(source_note)
            exclusions.append(IVExclusion(
                doc_id, str(raw["label"]), float(raw["v_value"]), str(raw["v_units"]),
                float(raw["i_open"]), str(raw["current_units"]), str(raw["reason"])))
        window._points = points
        window._exclusions = exclusions
        window._fit = _decode(artifact.payload.get("fit"), artifact.tables, artifact.blobs)
        _restore_appearance(window, artifact.payload)
        details = getattr(window, "_details", None)
        if details is not None:
            details.setPlainText(str(artifact.payload.get("details_text", "")))
        headline = getattr(window, "_headline", None)
        if headline is not None:
            headline.setText(str(artifact.payload.get("headline", "")))
        render = getattr(window, "_render_workspace_snapshot", None)
        if callable(render):
            render()
    except Exception as exc:
        return RestoreReport(skipped=(IV_ARTIFACT_ID,),
                             warnings=(f"could not restore I-V state: {exc}",))
    return RestoreReport(restored=(IV_ARTIFACT_ID,), warnings=tuple(warnings))


def capture_voltage_dependence(window: Any) -> WorkspaceArtifact:
    resources = _Resources()
    series = window.series()
    series_fields = {
        name: _encode(getattr(series, name), resources, f"series.{name}")
        for name in series._fields if name != "excluded"
    }
    remembered_excluded = [v for v in
                           (getattr(window, "_nanoscope_workspace_vdep_excluded", ()) or ())
                           if isinstance(v, Mapping)]
    excluded = []
    for index, item in enumerate(series.excluded):
        identity = _stable_source(window, str(item.doc_id))
        if not identity["source_key"] and index < len(remembered_excluded):
            identity = {name: remembered_excluded[index].get(name) for name in
                        ("source_key", "channel", "channel_name", "channel_kind")}
        excluded.append({
            **identity,
            "recording": str(item.recording), "reason": str(item.reason),
            "detail": str(item.detail),
        })

    # VDepPoint predates durable source identity and carries only a display label.  Build
    # the parallel identity vector by replaying its documented Session.documents ordering,
    # removing explicitly excluded documents and matching voltage as well as label.  This
    # remains deterministic when two folders contain files with the same basename.
    documents = []
    try:
        documents = list(window._session.documents())
    except AttributeError:
        pass
    remembered_points = [v for v in
                         (getattr(window, "_nanoscope_workspace_vdep_points", ()) or ())
                         if isinstance(v, Mapping)]
    same_restored_series = series is getattr(window, "_nanoscope_workspace_vdep_series", None)
    point_sources = [_clone_plain(dict(v)) for v in remembered_points] if same_restored_series else []
    if not same_restored_series:
        from nanoscope.analysis.iv import iv_point

        excluded_ids = {str(getattr(item, "doc_id", "")) for item in series.excluded}
        cursor = 0
        for point in series.points:
            matched = None
            while cursor < len(documents):
                document = documents[cursor]
                cursor += 1
                if str(getattr(document, "doc_id", "")) in excluded_ids:
                    continue
                if str(getattr(document, "display_name", "")) != str(point.recording):
                    continue
                try:
                    voltage = iv_point(window._session, document.doc_id)
                except (AttributeError, KeyError, TypeError, ValueError):
                    voltage = None
                if voltage is None or not np.isclose(float(voltage.v_volts),
                                                      float(point.v_volts), equal_nan=True):
                    continue
                matched = document
                break
            if matched is None:
                point_sources.append({"source_key": "", "channel": None,
                                      "channel_name": None, "channel_kind": None,
                                      "module_id": "detect.schmitt",
                                      "result_fingerprint": None})
                continue
            result = next((res for res in window._session.results.for_doc(matched.doc_id)
                           if str(getattr(res, "module_id", "")) == "detect.schmitt"
                           and hasattr(getattr(res, "payload", None), "df")), None)
            if result is not None:
                point_sources.append(_result_recipe(window, result))
            else:
                point_sources.append({**_stable_source(window, str(matched.doc_id)),
                                      "module_id": "detect.schmitt",
                                      "result_fingerprint": None})
    payload = {
        "config": {
            "metric": str(window._metric()),
            "selected_fit": str(_widget_value(window, "_fit_combo", "currentText", "none")),
            "applied_fit": str(getattr(window, "_applied_fit", "none")),
            "absolute_voltage": bool(_widget_value(window, "_abs_v", "isChecked", False)),
            "log_y": bool(_widget_value(window, "_log_y", "isChecked", False)),
            "accepted_only": bool(_widget_value(window, "_accepted_only", "isChecked", True)),
            "appearance": _capture_appearance(window),
        },
        "series": series_fields,
        "series_excluded": excluded,
        "point_sources": point_sources,
        "fit": _encode(window.fit(), resources, "fit"),
        "not_in_fit": _encode(getattr(window, "_not_in_fit", {}), resources, "not_in_fit"),
        "details_text": str(window.details_text()),
        "warning_text": str(window.warning_text()),
        "headline": str(_widget_value(window, "_headline", "text", "")),
    }
    return _artifact(VOLTAGE_DEPENDENCE_ARTIFACT_ID, payload, resources)


def restore_voltage_dependence(window: Any, artifact: WorkspaceArtifact) -> RestoreReport:
    ok, reason = _artifact_ok(artifact, VOLTAGE_DEPENDENCE_ARTIFACT_ID)
    if not ok:
        return RestoreReport(skipped=(VOLTAGE_DEPENDENCE_ARTIFACT_ID,), warnings=(reason,))
    config = artifact.payload.get("config", {})
    try:
        # Set controls with their compute-triggering signals blocked.  Restore is a snapshot
        # install, not an implicit second statistical run.
        _blocked(getattr(window, "_metric_combo"),
                 lambda: window.set_metric(str(config.get("metric", "capture_rate_hz"))))
        _blocked(getattr(window, "_abs_v"),
                 lambda: getattr(window, "_abs_v").setChecked(
                     bool(config.get("absolute_voltage", False))))
        _blocked(getattr(window, "_log_y"),
                 lambda: getattr(window, "_log_y").setChecked(bool(config.get("log_y", False))))
        _blocked(getattr(window, "_accepted_only"),
                 lambda: getattr(window, "_accepted_only").setChecked(
                     bool(config.get("accepted_only", True))))
        applied = str(config.get("applied_fit", "none"))
        combo = getattr(window, "_fit_combo")
        _blocked(combo, lambda: combo.setCurrentText(
            str(config.get("selected_fit", applied))))
        window._applied_fit = applied
        selected = str(config.get("selected_fit", config.get("applied_fit", "none")))
        if selected != applied:
            sync = getattr(window, "_sync_fit_apply_btn", None)
            if callable(sync):
                sync()

        from nanoscope.analysis.voltage_dep import VDepExclusion, VDepSeries

        raw_series = artifact.payload.get("series", {})
        kwargs = {str(name): _decode(value, artifact.tables, artifact.blobs)
                  for name, value in raw_series.items()}
        restored_excluded = []
        warnings: list[str] = []
        setattr(window, "_nanoscope_workspace_vdep_excluded",
                [_clone_plain(dict(v)) for v in artifact.payload.get("series_excluded", ())
                 if isinstance(v, Mapping)])
        setattr(window, "_nanoscope_workspace_vdep_points",
                [_clone_plain(dict(v)) for v in artifact.payload.get("point_sources", ())
                 if isinstance(v, Mapping)])
        for raw in artifact.payload.get("series_excluded", ()):
            doc_id, _channel, source_note = _resolve_source(window, raw)
            if doc_id is None:
                warnings.append(
                    f"voltage-dependence excluded source {raw.get('source_key')!r} is not open")
                doc_id = ""
            elif source_note:
                warnings.append(source_note)
            restored_excluded.append(VDepExclusion(
                doc_id, str(raw["recording"]), str(raw["reason"]), str(raw.get("detail", ""))))
        kwargs["excluded"] = tuple(restored_excluded)
        series = VDepSeries(**kwargs)
        window._series = series
        window._nanoscope_workspace_vdep_series = series
        window._points = list(series.points)
        for index, recipe in enumerate(artifact.payload.get("point_sources", ())):
            if not isinstance(recipe, Mapping):
                warnings.append(f"voltage-dependence point {index + 1} has malformed source identity")
                continue
            doc_id, channel, source_note = _resolve_source(window, recipe)
            if doc_id is None:
                warnings.append(
                    f"voltage-dependence point {index + 1} source "
                    f"{recipe.get('source_key')!r} is not open")
                continue
            if source_note:
                warnings.append(source_note)
            module_id = str(recipe.get("module_id") or "")
            saved_result = recipe.get("result_fingerprint")
            if module_id and saved_result and channel is not None:
                result = window._session.results.get(doc_id, channel, module_id)
                if result is None or _result_recipe(window, result).get("result_fingerprint") != saved_result:
                    warnings.append(
                        f"voltage-dependence point {index + 1} was derived from a changed "
                        f"{module_id!r} result; frozen values were retained")
        window._fit = _decode(artifact.payload.get("fit"), artifact.tables, artifact.blobs)
        window._not_in_fit = _decode(artifact.payload.get("not_in_fit", {}),
                                     artifact.tables, artifact.blobs)
        _restore_appearance(window, config)
        details = getattr(window, "_details", None)
        if details is not None:
            details.setPlainText(str(artifact.payload.get("details_text", "")))
        set_warnings = getattr(window, "_set_warnings", None)
        warning_text = str(artifact.payload.get("warning_text", ""))
        if callable(set_warnings):
            set_warnings((warning_text,) if warning_text else ())
        headline = getattr(window, "_headline", None)
        if headline is not None:
            headline.setText(str(artifact.payload.get("headline", "")))
        render = getattr(window, "_render_workspace_snapshot", None)
        if callable(render):
            render()
    except Exception as exc:
        return RestoreReport(skipped=(VOLTAGE_DEPENDENCE_ARTIFACT_ID,),
                             warnings=(f"could not restore voltage-dependence state: {exc}",))
    return RestoreReport(restored=(VOLTAGE_DEPENDENCE_ARTIFACT_ID,), warnings=tuple(warnings))


# -------------------------------------------------------------------------------------------------
# Analysis inspector configuration, including ML model identity.
# -------------------------------------------------------------------------------------------------


def capture_analysis_panel(panel: Any) -> WorkspaceArtifact:
    resources = _Resources()
    detection = panel.detection_panel
    baseline = panel.baseline_panel
    measure = panel.measure_panel
    stats = getattr(panel, "_stats", None)
    tabs = getattr(detection, "mode_tabs", None)
    tab_name = ""
    if tabs is not None:
        tab_name = str(tabs.tabText(tabs.currentIndex()))
    target = panel.current_target()
    result = panel.current_detection()
    if result is not None:
        source = _result_recipe(panel, result)
    elif target is not None:
        fresh_source = {
            **_stable_source(panel, str(target[0]), int(target[1])),
            "module_id": str(getattr(panel, "_active_detection", "") or "") or None,
            "module_version": None,
            "event_frame_fingerprint": None,
            "result_fingerprint": None,
        }
        carried_source = getattr(panel, "_nanoscope_workspace_analysis_source", None)
        if (isinstance(carried_source, Mapping)
                and _same_persistent_source(
                    str(carried_source.get("source_key") or ""),
                    str(fresh_source.get("source_key") or ""))
                and (carried_source.get("channel_name"), carried_source.get("channel_kind"))
                == (fresh_source.get("channel_name"), fresh_source.get("channel_kind"))
                and str(carried_source.get("module_id") or "")
                == str(fresh_source.get("module_id") or "")):
            # The source is open but the saved derived result is temporarily unavailable.
            # Keep its exact lineage until a new result replaces it.
            source = _clone_plain(dict(carried_source))
        else:
            source = fresh_source
    else:
        carried_source = getattr(panel, "_nanoscope_workspace_analysis_source", None)
        source = (_clone_plain(dict(carried_source))
                  if isinstance(carried_source, Mapping) else None)

    model_path = detection.model_path()
    model_reference = capture_model_reference(model_path)
    if model_reference is None:
        unresolved = getattr(detection, "_nanoscope_unresolved_model_reference", None)
        if isinstance(unresolved, Mapping):
            model_reference = _clone_plain(dict(unresolved))
    elif hasattr(detection, "_nanoscope_unresolved_model_reference"):
        delattr(detection, "_nanoscope_unresolved_model_reference")
    payload = {
        "source": source,
        "detection_params": _encode(detection.params(), resources, "detection_params"),
        "ml_params": _encode(detection.ml_params(), resources, "ml_params"),
        "ml_model": model_reference,
        "baseline_params": _encode(baseline.params(), resources, "baseline_params"),
        "baseline_view": {
            "show_track": bool(_widget_value(baseline, "show_track", "isChecked", False)),
            "subtract_view": bool(_widget_value(baseline, "subtract_view", "isChecked", False)),
        },
        "active_detection": str(getattr(panel, "_active_detection", "")),
        "mode_tab": tab_name,
        "filter_preview": bool(_widget_value(detection, "filter_preview", "isChecked", False)),
        "measure": {
            "method": str(_widget_value(measure, "_method_combo", "currentText", "median")),
            "basis": str(measure.measurement_basis()),
            "active": bool(measure.is_active()),
            "this_recording_only": bool(_widget_value(measure, "_scope_check", "isChecked", False)),
            "markers_visible": bool(measure.markers_visible()),
        },
        "merge_window_s": float(panel.merge_controls.window_s()),
        "fit_controls": {
            "n_components_dwell": int(_widget_value(stats, "n_components_dwell", "value", 1)),
            "n_components_depth": int(_widget_value(stats, "n_components_depth", "value", 1)),
            "use_efficiency": bool(_widget_value(stats, "use_efficiency", "isChecked", False)),
            "use_filter_limited": bool(_widget_value(stats, "use_filter_limited", "isChecked", False)),
        },
    }
    return _artifact(ANALYSIS_PANEL_ARTIFACT_ID, payload, resources)


def _select_tab_by_text(tabs: Any, text: str) -> bool:
    if tabs is None:
        return False
    for index in range(tabs.count()):
        if tabs.tabText(index) == text:
            tabs.setCurrentIndex(index)
            return True
    return False


def restore_analysis_panel(panel: Any, artifact: WorkspaceArtifact) -> RestoreReport:
    ok, reason = _artifact_ok(artifact, ANALYSIS_PANEL_ARTIFACT_ID)
    if not ok:
        return RestoreReport(skipped=(ANALYSIS_PANEL_ARTIFACT_ID,), warnings=(reason,))
    payload = artifact.payload
    warnings: list[str] = []
    try:
        detection = panel.detection_panel
        baseline = panel.baseline_panel
        measure = panel.measure_panel

        source = payload.get("source")
        if isinstance(source, Mapping):
            setattr(panel, "_nanoscope_workspace_analysis_source", _clone_plain(dict(source)))
            doc_id, channel, source_note = _resolve_source(panel, source)
            if doc_id is not None and channel is not None:
                panel.set_selection(doc_id, channel)
                if source_note:
                    warnings.append(source_note)
                live_result = panel.current_detection()
                saved_module = str(source.get("module_id") or "")
                if saved_module:
                    wanted = panel._session.results.get(doc_id, channel, saved_module)
                    if wanted is not None:
                        set_active = getattr(panel, "_set_active_detection", None)
                        if callable(set_active):
                            set_active(saved_module)
                        live_result = wanted
                    else:
                        warnings.append(
                            f"saved analysis result {saved_module!r} is not available on the restored source")
                saved_rows = source.get("event_frame_fingerprint")
                live_rows = fingerprint_frame(getattr(getattr(live_result, "payload", None),
                                                      "df", None))
                if saved_rows and saved_rows != live_rows:
                    warnings.append("the restored analysis result's event rows differ from the saved snapshot")
                saved_result = source.get("result_fingerprint")
                if live_result is not None and saved_result:
                    live_result_stamp = _result_recipe(panel, live_result).get("result_fingerprint")
                    if live_result_stamp != saved_result:
                        warnings.append(
                            "the restored analysis result provenance/parameters differ from the saved snapshot")
            else:
                warnings.append(source_note or "the saved analysis-panel source is unavailable")

        active = str(payload.get("active_detection") or "")
        set_active = getattr(panel, "_set_active_detection", None)
        if active and callable(set_active):
            set_active(active)
            sync = getattr(panel, "_sync_result_selector", None)
            if callable(sync):
                sync()

        detection.set_params(_decode(payload.get("detection_params", {}), artifact.tables,
                                     artifact.blobs))
        baseline.set_params(_decode(payload.get("baseline_params", {}), artifact.tables,
                                    artifact.blobs))
        model_ok, model_reason = _model_reference_matches(payload.get("ml_model"))
        ml_params = _decode(payload.get("ml_params", {}), artifact.tables, artifact.blobs)
        if not model_ok:
            ml_params = dict(ml_params)
            ml_params.pop("model_path", None)
            if isinstance(payload.get("ml_model"), Mapping):
                setattr(detection, "_nanoscope_unresolved_model_reference",
                        _clone_plain(dict(payload["ml_model"])))
            # Keep the carried reference present while set_model_path emits its public
            # change signal. MainWindow may snapshot synchronously from that signal and
            # must not observe a transient state which drops the unavailable model.
            detection._nanoscope_preserve_unresolved_model_reference = True
            try:
                detection.set_model_path(None)
            finally:
                if hasattr(detection, "_nanoscope_preserve_unresolved_model_reference"):
                    delattr(detection, "_nanoscope_preserve_unresolved_model_reference")
            warnings.append(model_reason + "; ML settings were restored but the model was not loaded")
        else:
            model_reference = payload.get("ml_model")
            model_path = (str(model_reference.get("path"))
                          if isinstance(model_reference, Mapping) and model_reference.get("path")
                          else None)
            detection.set_model_path(model_path)
            if hasattr(detection, "_nanoscope_unresolved_model_reference"):
                delattr(detection, "_nanoscope_unresolved_model_reference")
        detection.set_ml_params(ml_params)

        view = payload.get("baseline_view", {})
        _set_checked(baseline, "show_track", view.get("show_track", False))
        _set_checked(baseline, "subtract_view", view.get("subtract_view", False))
        _set_checked(detection, "filter_preview", payload.get("filter_preview", False))

        measure_state = payload.get("measure", {})
        measure.set_method(str(measure_state.get("method", "median")))
        measure.set_measurement_basis(str(measure_state.get("basis", "raw")))
        scoped = getattr(measure, "set_scoped", None)
        if callable(scoped):
            scoped(bool(measure_state.get("this_recording_only", False)))
        else:
            _set_checked(measure, "_scope_check", measure_state.get("this_recording_only", False))
        _set_checked(measure, "_markers_check", measure_state.get("markers_visible", True))

        panel.merge_controls.set_window_s(float(payload.get("merge_window_s", 0.01)))
        fit = payload.get("fit_controls", {})
        stats = getattr(panel, "_stats", None)
        for attr, key in (("n_components_dwell", "n_components_dwell"),
                          ("n_components_depth", "n_components_depth")):
            widget = getattr(stats, attr, None)
            if widget is not None:
                widget.setValue(int(fit.get(key, 1)))
        _set_checked(stats, "use_efficiency", fit.get("use_efficiency", False))
        _set_checked(stats, "use_filter_limited", fit.get("use_filter_limited", False))

        tab = str(payload.get("mode_tab") or "")
        if tab and not _select_tab_by_text(getattr(detection, "mode_tabs", None), tab):
            warnings.append(f"analysis tab {tab!r} is not available in this build")
        if bool(measure_state.get("active", False)):
            measure.set_active(True)
            if not measure.is_active():
                warnings.append("manual-measurement cursors could not be restored before a trace was bound")
        else:
            measure.set_active(False)
    except Exception as exc:
        return RestoreReport(skipped=(ANALYSIS_PANEL_ARTIFACT_ID,),
                             warnings=(f"could not restore analysis-panel configuration: {exc}",))
    return RestoreReport(restored=(ANALYSIS_PANEL_ARTIFACT_ID,), warnings=tuple(warnings))


# -------------------------------------------------------------------------------------------------
# Workspace orchestration and unknown-artifact preservation.
# -------------------------------------------------------------------------------------------------


def _clone_artifact(artifact: WorkspaceArtifact) -> WorkspaceArtifact:
    return WorkspaceArtifact(
        artifact_id=str(artifact.artifact_id),
        kind=str(artifact.kind),
        schema_version=int(artifact.schema_version),
        payload=_clone_plain(artifact.payload),
        tables={str(k): v.copy(deep=True) for k, v in artifact.tables.items()},
        blobs={str(k): bytes(v) for k, v in artifact.blobs.items()},
        extra=_clone_plain(artifact.extra),
    )


def _clone_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_plain(v) for v in value)
    if isinstance(value, set):
        return {_clone_plain(v) for v in value}
    if isinstance(value, frozenset):
        return frozenset(_clone_plain(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    return value


def capture_distribution_summaries(window: Any) -> WorkspaceArtifact:
    """Persist draft controls and committed summary coordinates, never rerun kernels."""
    from .figure_tables import encode_figure_table

    payload, tables = window.snapshot_state()
    if not isinstance(payload, Mapping) or not isinstance(tables, Mapping):
        raise ValueError("Distribution summary state must contain a payload and tables.")
    if any(not isinstance(name, str) or not isinstance(frame, pd.DataFrame)
           for name, frame in tables.items()):
        raise ValueError("Distribution summary resources must be named DataFrames.")
    payload = _clone_plain(dict(payload))
    payload["table_resources"] = {name: f"table_{index}.parquet" for index, name in enumerate(tables)}
    # Opaque Parquet resources preserve mixed scalar values and missing sentinels
    # in observations excluded from the summary. The project container's ordinary
    # table codec and every existing analytical artifact remain unchanged.
    return WorkspaceArtifact(
        artifact_id=DISTRIBUTION_SUMMARIES_ARTIFACT_ID,
        kind=_KINDS[DISTRIBUTION_SUMMARIES_ARTIFACT_ID], schema_version=_SCHEMA_VERSION,
        payload=payload,
        blobs={payload["table_resources"][name]: encode_figure_table(frame)
               for name, frame in tables.items()})


def restore_distribution_summaries(window: Any, artifact: WorkspaceArtifact) -> RestoreReport:
    from .figure_tables import decode_figure_table

    ok, reason = _artifact_ok(artifact, DISTRIBUTION_SUMMARIES_ARTIFACT_ID)
    if not ok:
        return RestoreReport(skipped=(DISTRIBUTION_SUMMARIES_ARTIFACT_ID,), warnings=(reason,))
    try:
        payload = _clone_plain(artifact.payload)
        resources = payload.pop("table_resources", {})
        if (not isinstance(resources, Mapping) or any(
                not isinstance(name, str) or not isinstance(blob, str) or blob not in artifact.blobs
                for name, blob in resources.items())):
            raise ValueError("Distribution summary table resources are missing or malformed.")
        tables = {name: frame.copy(deep=True) for name, frame in artifact.tables.items()}
        if set(resources).intersection(tables):
            raise ValueError("Distribution summary tables have duplicate resource names.")
        tables.update({name: decode_figure_table(artifact.blobs[blob]) for name, blob in resources.items()})
        window.restore_state(payload, tables)
    except (KeyError, TypeError, ValueError) as exc:
        return RestoreReport(skipped=(DISTRIBUTION_SUMMARIES_ARTIFACT_ID,),
                             warnings=(f"Distribution summary snapshot could not be restored: {exc}",))
    return RestoreReport(restored=(DISTRIBUTION_SUMMARIES_ARTIFACT_ID,))


def capture_analysis_workspace(
    existing: AnalyticalWorkspace | None = None,
    *,
    plots: Any = None,
    distribution_summaries: Any = None,
    subpopulation: Any = None,
    psd: Any = None,
    event_plots: Any = None,
    iv: Any = None,
    voltage_dependence: Any = None,
    analysis_panel: Any = None,
) -> AnalyticalWorkspace:
    """Capture supplied live surfaces and merge them into ``existing``.

    ``None`` means "surface not instantiated" and therefore preserves an existing artifact;
    it does not delete it.  Only stable ids owned by this module are replaced.  Unknown/plugin
    artifacts, workspace ``extra`` fields, and their ordering survive unchanged.
    """

    captures = (
        (PLOTS_ARTIFACT_ID, plots, capture_plots),
        (DISTRIBUTION_SUMMARIES_ARTIFACT_ID, distribution_summaries, capture_distribution_summaries),
        (SUBPOPULATION_ARTIFACT_ID, subpopulation, capture_subpopulation),
        (PSD_ARTIFACT_ID, psd, capture_psd),
        (EVENT_PLOTS_ARTIFACT_ID, event_plots, capture_event_plots),
        (IV_ARTIFACT_ID, iv, capture_iv),
        (VOLTAGE_DEPENDENCE_ARTIFACT_ID, voltage_dependence, capture_voltage_dependence),
        (ANALYSIS_PANEL_ARTIFACT_ID, analysis_panel, capture_analysis_panel),
    )
    replacements: dict[str, WorkspaceArtifact] = {}
    for artifact_id, surface, capture in captures:
        if surface is not None:
            replacements[artifact_id] = capture(surface)

    prior = existing or AnalyticalWorkspace()
    artifacts: list[WorkspaceArtifact] = []
    for old in prior.artifacts:
        replacement = replacements.get(old.artifact_id)
        # A newer build may already own this stable id under a schema/kind we cannot
        # understand.  Never downgrade it merely because the corresponding window exists:
        # preserve it byte-for-byte and suppress our v1 replacement (duplicate ids are
        # forbidden by the project container).
        compatible = (old.artifact_id in _KINDS
                      and old.kind == _KINDS[old.artifact_id]
                      and old.schema_version == _SCHEMA_VERSION)
        if replacement is not None and compatible:
            artifacts.append(replacements.pop(old.artifact_id))
        else:
            artifacts.append(_clone_artifact(old))
            if replacement is not None:
                replacements.pop(old.artifact_id)
    for artifact_id in _ORDER:
        replacement = replacements.pop(artifact_id, None)
        if replacement is not None:
            artifacts.append(replacement)
    # Defensive: should be empty, but retaining future captures is safer than dropping them.
    artifacts.extend(replacements.values())
    return AnalyticalWorkspace(version=int(prior.version), artifacts=artifacts,
                               extra=_clone_plain(prior.extra))


def restore_analysis_workspace(
    workspace: AnalyticalWorkspace,
    *,
    plots: Any = None,
    distribution_summaries: Any = None,
    subpopulation: Any = None,
    psd: Any = None,
    event_plots: Any = None,
    iv: Any = None,
    voltage_dependence: Any = None,
    analysis_panel: Any = None,
) -> RestoreReport:
    """Restore each supplied live surface from its stable artifact, if present."""

    surfaces = {
        PLOTS_ARTIFACT_ID: (plots, restore_plots),
        DISTRIBUTION_SUMMARIES_ARTIFACT_ID: (distribution_summaries, restore_distribution_summaries),
        SUBPOPULATION_ARTIFACT_ID: (subpopulation, restore_subpopulation),
        PSD_ARTIFACT_ID: (psd, restore_psd),
        EVENT_PLOTS_ARTIFACT_ID: (event_plots, restore_event_plots),
        IV_ARTIFACT_ID: (iv, restore_iv),
        VOLTAGE_DEPENDENCE_ARTIFACT_ID: (voltage_dependence, restore_voltage_dependence),
        ANALYSIS_PANEL_ARTIFACT_ID: (analysis_panel, restore_analysis_panel),
    }
    by_id = {artifact.artifact_id: artifact for artifact in workspace.artifacts}
    reports: list[RestoreReport] = []
    for artifact_id in _ORDER:
        surface, restore = surfaces[artifact_id]
        if surface is not None and artifact_id in by_id:
            reports.append(restore(surface, by_id[artifact_id]))
    return RestoreReport.merge(reports)
