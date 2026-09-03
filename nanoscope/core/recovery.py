"""Crash-recovery snapshots for a NanoScope project session.

Recovery is deliberately a layer *around* :mod:`nanoscope.core.project`, not a
second project serializer.  Each generation is a small ZIP envelope containing
``envelope.json`` plus an ordinary, byte-for-byte ``project.nanoscope`` archive.
The envelope has its own format/version, while the inner project continues to be
written and read by :func:`save_project` / :func:`load_project` (currently project
format v10).

One UUID directory belongs to one application session and contains at most two
committed generations::

    <root>/<session UUID>/current.nanoscope-recovery
                          previous.nanoscope-recovery

The new archive is completely written, flushed and validated before ``current``
is moved to ``previous``.  A failed serialization therefore changes neither
generation.  A failure in the tiny rename window can leave only ``previous``;
discovery intentionally falls back to it.  A corrupt ``current`` is never rotated
over a good ``previous``.

There is no Qt import here.  In particular, :func:`clone_project_state` is the
GUI-thread snapshot seam: it detaches every mutable frame/container before a
worker is allowed to serialize it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from . import atomic_write
from .project import (
    FORMAT as PROJECT_FORMAT,
    VERSION as PROJECT_VERSION,
    ProjectError,
    ProjectState,
    load_project,
    save_project,
)

__all__ = [
    "RECOVERY_FORMAT",
    "RECOVERY_VERSION",
    "ENVELOPE_NAME",
    "PAYLOAD_NAME",
    "CURRENT_NAME",
    "PREVIOUS_NAME",
    "RETIREMENT_NAME",
    "RecoveryError",
    "RecoveryVersionError",
    "RecoveryChangedError",
    "StaleRecoveryWrite",
    "RecoveryMetadata",
    "RecoveryCandidate",
    "RecoveredProject",
    "clone_project_state",
    "default_recovery_root",
    "RecoveryStore",
]


RECOVERY_FORMAT = "nanoscope-recovery"
RECOVERY_VERSION = 1
ENVELOPE_NAME = "envelope.json"
PAYLOAD_NAME = "project.nanoscope"
CURRENT_NAME = "current.nanoscope-recovery"
PREVIOUS_NAME = "previous.nanoscope-recovery"
RETIREMENT_NAME = "retirement.json"

_RETIREMENT_FORMAT = "nanoscope-recovery-retirement"
_RETIREMENT_VERSION = 1

DEFAULT_MAX_SESSIONS = 20
DEFAULT_MAX_AGE_SECONDS = 30 * 24 * 60 * 60

# Two RecoveryStore objects may address the same injected root/session in tests or
# support tooling. They must still share the retirement/write critical section.
_SESSION_LOCKS_GUARD = threading.Lock()
_SESSION_LOCKS: dict[str, threading.RLock] = {}


class RecoveryError(ValueError):
    """A recovery envelope is missing, damaged, or malformed."""


class RecoveryVersionError(RecoveryError):
    """A recovery envelope needs a different NanoScope recovery reader."""


class RecoveryChangedError(RecoveryError):
    """A candidate changed after it was presented to the caller."""


class StaleRecoveryWrite(RecoveryError):
    """An older asynchronous snapshot tried to replace a newer revision."""


@dataclass(frozen=True)
class RecoveryMetadata:
    """Metadata stored outside the ordinary project payload."""

    session_id: str
    revision: int
    timestamp: float
    source_project_path: str | None
    covered_explicit_save_revision: int


@dataclass(frozen=True)
class RecoveryCandidate:
    """One validated recovery generation suitable for presenting to a user."""

    session_id: str
    revision: int
    timestamp: float
    source_project_path: str | None
    covered_explicit_save_revision: int
    envelope_path: Path
    generation: Literal["current", "previous"]
    payload_member: str = PAYLOAD_NAME

    @property
    def path(self) -> Path:
        """Compatibility/convenience alias for :attr:`envelope_path`."""

        return self.envelope_path

    @property
    def metadata(self) -> RecoveryMetadata:
        return RecoveryMetadata(
            session_id=self.session_id,
            revision=self.revision,
            timestamp=self.timestamp,
            source_project_path=self.source_project_path,
            covered_explicit_save_revision=self.covered_explicit_save_revision,
        )


@dataclass(frozen=True)
class RecoveredProject:
    """A validated candidate together with its ordinary :class:`ProjectState`."""

    candidate: RecoveryCandidate
    state: ProjectState

    @property
    def metadata(self) -> RecoveryMetadata:
        return self.candidate.metadata


def default_recovery_root() -> Path:
    """Return the persistent, per-user recovery root without creating it.

    ``NANOSCOPE_RECOVERY_ROOT`` is intentionally first so tests, managed
    deployments, and support sessions can put recovery on a known volume without
    importing Qt merely to ask ``QStandardPaths``.
    """

    configured = os.environ.get("NANOSCOPE_RECOVERY_ROOT")
    if configured:
        return Path(configured).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "NanoScope" / "Recovery"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return ((Path(base).expanduser() if base else home / "AppData" / "Local")
                / "NanoScope" / "Recovery")
    base = os.environ.get("XDG_STATE_HOME")
    return ((Path(base).expanduser() if base else home / ".local" / "state")
            / "NanoScope" / "recovery")


def _clone_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Deep-copy a frame, including Python objects held in ``object`` cells.

    Pandas' ``deep=True`` detaches its arrays but intentionally does not recursively
    copy Python objects inside object-dtype arrays.  Provenance occasionally lives in
    exactly those cells, so the extra per-cell copy is part of the worker-snapshot
    contract rather than belt-and-braces copying.
    """

    out = frame.copy(deep=True)
    for position in range(out.shape[1]):
        # Positional selection deliberately supports frames with duplicate column
        # labels; ``out[name]`` would return a DataFrame rather than a Series there.
        series = out.iloc[:, position]
        if series.dtype == object:
            values = [copy.deepcopy(value) for value in series.array]
            out.isetitem(position, pd.Series(
                values, index=out.index, dtype=object, name=series.name))
    out.attrs = copy.deepcopy(frame.attrs)
    return out


def _deep_clone(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _clone_frame(value)
    if isinstance(value, pd.Series):
        out = value.copy(deep=True)
        if out.dtype == object:
            out = pd.Series([copy.deepcopy(v) for v in out.array], index=out.index,
                            dtype=object, name=out.name)
        out.attrs = copy.deepcopy(value.attrs)
        return out
    if isinstance(value, np.ndarray):
        # copy.deepcopy is needed for object arrays; ndarray.copy only detaches the
        # pointer array and would leave dictionaries/lists shared.
        return copy.deepcopy(value) if value.dtype == object else value.copy()
    if is_dataclass(value) and not isinstance(value, type):
        values = {f.name: _deep_clone(getattr(value, f.name)) for f in fields(value)}
        return type(value)(**values)
    if isinstance(value, dict):
        return {_deep_clone(k): _deep_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_deep_clone(v) for v in value)
    if isinstance(value, set):
        return {_deep_clone(v) for v in value}
    return copy.deepcopy(value)


def clone_project_state(state: ProjectState) -> ProjectState:
    """Return a fully detached, type-preserving snapshot of ``state``.

    Call this on the GUI thread.  The returned object may then be handed to a
    background recovery writer while the live session continues to mutate.
    """

    if not isinstance(state, ProjectState):
        raise TypeError("state must be a ProjectState")
    cloned = _deep_clone(state)
    assert isinstance(cloned, ProjectState)
    return cloned


def _canonical_session_id(value: str | uuid.UUID | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    try:
        parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"session_id must be a UUID, got {value!r}") from exc
    return str(parsed)


def _is_canonical_session_dir(path: Path) -> bool:
    try:
        return str(uuid.UUID(path.name)) == path.name
    except (ValueError, TypeError):
        return False


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalise_source_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    return os.path.abspath(os.path.expanduser(os.fspath(value)))


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory durability after a generation rename."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _session_lock(directory: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(directory))
    with _SESSION_LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(key, threading.RLock())


class RecoveryStore:
    """Persistent two-generation recovery storage.

    ``root`` and ``session_id`` are injectable for tests.  In production each app
    launch omits ``session_id`` and consequently receives a fresh UUID lineage.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        session_id: str | uuid.UUID | None = None,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_age_seconds: float | None = DEFAULT_MAX_AGE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(max_sessions, int) or isinstance(max_sessions, bool) \
                or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        if max_age_seconds is not None:
            max_age_seconds = float(max_age_seconds)
            if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
                raise ValueError("max_age_seconds must be finite and non-negative")
        self.root = Path(root).expanduser() if root is not None else default_recovery_root()
        self.session_id = _canonical_session_id(session_id)
        self.max_sessions = max_sessions
        self.max_age_seconds = max_age_seconds
        self._clock = clock
        self._mutex = _session_lock(self.session_directory)

    @property
    def session_directory(self) -> Path:
        return self.root / self.session_id

    @property
    def current_path(self) -> Path:
        return self.session_directory / CURRENT_NAME

    @property
    def previous_path(self) -> Path:
        return self.session_directory / PREVIOUS_NAME

    @property
    def retirement_path(self) -> Path:
        return self.session_directory / RETIREMENT_NAME

    def _ensure_session_directory(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.session_directory.is_symlink():
            raise RecoveryError(
                f"recovery session location may not be a symlink: {self.session_directory}")
        self.session_directory.mkdir(mode=0o700, exist_ok=True)
        # Best effort on platforms which implement Unix permission bits.  Do not make
        # a successful recovery write fail merely because a filesystem ignores chmod.
        for directory in (self.root, self.session_directory):
            try:
                mode = directory.stat(follow_symlinks=False).st_mode
                if not stat.S_ISDIR(mode):
                    raise OSError(f"recovery location is not a directory: {directory}")
                if mode & 0o077:
                    os.chmod(directory, 0o700)
            except NotImplementedError:  # pragma: no cover - Windows/filesystem dependent
                pass

    @staticmethod
    def _envelope(metadata: RecoveryMetadata, payload: bytes) -> dict[str, Any]:
        return {
            "format": RECOVERY_FORMAT,
            "version": RECOVERY_VERSION,
            "metadata": {
                "session_uuid": metadata.session_id,
                "revision": metadata.revision,
                "timestamp": metadata.timestamp,
                "source_project_path": metadata.source_project_path,
                "covered_explicit_save_revision":
                    metadata.covered_explicit_save_revision,
            },
            "payload": {
                "member": PAYLOAD_NAME,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "format": PROJECT_FORMAT,
                "version": PROJECT_VERSION,
            },
        }

    @staticmethod
    def _parse_envelope(raw: bytes, name: str) -> tuple[RecoveryMetadata, dict[str, Any]]:
        try:
            envelope = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise RecoveryError(f"{name}: {ENVELOPE_NAME} is not valid JSON") from exc
        if not isinstance(envelope, dict):
            raise RecoveryError(f"{name}: {ENVELOPE_NAME} is not a JSON object")
        if envelope.get("format") != RECOVERY_FORMAT:
            raise RecoveryError(f"{name} is not a NanoScope recovery snapshot")
        version = envelope.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise RecoveryVersionError(f"{name} has no valid recovery format version")
        if version != RECOVERY_VERSION:
            raise RecoveryVersionError(
                f"{name} uses recovery format {version}; this build reads "
                f"version {RECOVERY_VERSION}")
        meta = envelope.get("metadata")
        payload = envelope.get("payload")
        if not isinstance(meta, dict) or not isinstance(payload, dict):
            raise RecoveryError(f"{name}: recovery metadata or payload descriptor is missing")
        try:
            session_id = _canonical_session_id(meta["session_uuid"])
            revision = _nonnegative_int(meta["revision"], "revision")
            covered = _nonnegative_int(
                meta["covered_explicit_save_revision"],
                "covered_explicit_save_revision",
            )
            timestamp = float(meta["timestamp"])
            source = meta.get("source_project_path")
        except (KeyError, TypeError, ValueError) as exc:
            raise RecoveryError(f"{name}: recovery metadata is malformed ({exc})") from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise RecoveryError(f"{name}: recovery timestamp is invalid")
        if source is not None and not isinstance(source, str):
            raise RecoveryError(f"{name}: source project path is invalid")
        return RecoveryMetadata(session_id, revision, timestamp, source, covered), payload

    @staticmethod
    def _read_archive(path: Path) -> tuple[RecoveryMetadata, bytes]:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            raise RecoveryError(f"{path.name}: no such recovery snapshot")
        if not stat.S_ISREG(mode):
            raise RecoveryError(f"{path.name} is not a regular recovery snapshot")
        if not zipfile.is_zipfile(path):
            raise RecoveryError(f"{path.name} is not a complete recovery archive")
        try:
            with zipfile.ZipFile(path, "r") as zf:
                try:
                    envelope_raw = zf.read(ENVELOPE_NAME)
                except KeyError:
                    raise RecoveryError(
                        f"{path.name} is missing {ENVELOPE_NAME}") from None
                metadata, descriptor = RecoveryStore._parse_envelope(
                    envelope_raw, path.name)
                member = descriptor.get("member")
                if member != PAYLOAD_NAME:
                    raise RecoveryError(f"{path.name}: recovery payload member is invalid")
                try:
                    payload = zf.read(PAYLOAD_NAME)
                except KeyError:
                    raise RecoveryError(
                        f"{path.name} is missing {PAYLOAD_NAME}") from None
        except (zipfile.BadZipFile, OSError) as exc:
            raise RecoveryError(f"{path.name} is damaged ({exc})") from exc

        size = descriptor.get("size")
        digest = descriptor.get("sha256")
        project_format = descriptor.get("format")
        project_version = descriptor.get("version")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RecoveryError(f"{path.name}: recovery payload size is invalid")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RecoveryError(f"{path.name}: recovery payload digest is invalid")
        if project_format != PROJECT_FORMAT:
            raise RecoveryError(f"{path.name}: embedded payload is not a NanoScope project")
        if not isinstance(project_version, int) or isinstance(project_version, bool):
            raise RecoveryError(f"{path.name}: embedded project version is invalid")
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise RecoveryError(f"{path.name}: embedded project payload failed validation")
        return metadata, payload

    @staticmethod
    def _load_payload(payload: bytes) -> ProjectState:
        # load_project intentionally takes a path.  Extract to the OS temporary area,
        # not beside the recovery: discovery/load must continue to work when the
        # recovery root has become read-only.
        fd, named = tempfile.mkstemp(prefix="nanoscope-recovery-read-", suffix=".nanoscope")
        path = Path(named)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            try:
                return load_project(path)
            except ProjectError as exc:
                raise RecoveryError(f"embedded project is not readable ({exc})") from exc
        finally:
            _safe_unlink(path)

    @staticmethod
    def _generation_for_path(path: Path) -> Literal["current", "previous"]:
        return "previous" if path.name == PREVIOUS_NAME else "current"

    def _decode(self, path: Path, *, expected_session: str | None = None,
                load_state: bool = True) -> tuple[RecoveryCandidate, ProjectState | None]:
        metadata, payload = self._read_archive(path)
        if expected_session is not None and metadata.session_id != expected_session:
            raise RecoveryError(
                f"{path.name}: session metadata does not match its recovery directory")
        state = self._load_payload(payload) if load_state else None
        candidate = RecoveryCandidate(
            session_id=metadata.session_id,
            revision=metadata.revision,
            timestamp=metadata.timestamp,
            source_project_path=metadata.source_project_path,
            covered_explicit_save_revision=metadata.covered_explicit_save_revision,
            envelope_path=path,
            generation=self._generation_for_path(path),
        )
        return candidate, state

    def validate(self, path: str | Path) -> RecoveryCandidate:
        """Fully validate an envelope *and* its embedded project payload."""

        candidate, _state = self._decode(Path(path), load_state=True)
        return candidate

    def _serialize_payload(self, state: ProjectState) -> bytes:
        token = uuid.uuid4().hex
        path = self.session_directory / f".payload-{token}.nanoscope"
        try:
            save_project(path, state)
            return path.read_bytes()
        finally:
            _safe_unlink(path)

    @staticmethod
    def _write_pending(path: Path, envelope: dict[str, Any], payload: bytes) -> None:
        with open(path, "xb") as fh:
            with zipfile.ZipFile(fh, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(ENVELOPE_NAME, json.dumps(envelope, indent=2, sort_keys=True))
                # The inner project is already compressed. Storing avoids spending CPU
                # to recompress parquet/ZIP bytes for no useful size reduction.
                zf.writestr(PAYLOAD_NAME, payload, compress_type=zipfile.ZIP_STORED)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _valid_at(self, path: Path, session_id: str) -> RecoveryCandidate | None:
        try:
            candidate, _state = self._decode(
                path, expected_session=session_id, load_state=True)
            return candidate
        except (RecoveryError, OSError):
            return None

    @staticmethod
    def _read_retirement(directory: Path) -> tuple[int, bool]:
        """Return ``(through_revision, all_revisions)`` for one lineage.

        A malformed tombstone errs on the side of retention (fence everything).  It
        is internal state written only by this module; treating damage as "no fence"
        could resurrect the exact recovery a user intentionally discarded.
        """

        path = directory / RETIREMENT_NAME
        if path.is_symlink():
            return -1, True
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return -1, False
        except (OSError, UnicodeDecodeError, ValueError):
            return -1, True
        if not isinstance(raw, dict) or raw.get("format") != _RETIREMENT_FORMAT \
                or raw.get("version") != _RETIREMENT_VERSION:
            return -1, True
        if raw.get("all_revisions") is True:
            return -1, True
        through = raw.get("through_revision")
        if not isinstance(through, int) or isinstance(through, bool) or through < 0:
            return -1, True
        return through, False

    @staticmethod
    def _write_retirement(directory: Path, through: int, all_revisions: bool) -> None:
        """Atomically advance (never lower) a lineage's retirement fence."""

        old_through, old_all = RecoveryStore._read_retirement(directory)
        all_revisions = bool(all_revisions or old_all)
        through = max(int(through), old_through)
        body = {
            "format": _RETIREMENT_FORMAT,
            "version": _RETIREMENT_VERSION,
            "through_revision": max(0, through),
            "all_revisions": all_revisions,
        }
        if (directory / RETIREMENT_NAME).is_symlink():
            raise RecoveryError("recovery retirement marker may not be a symlink")
        with atomic_write.staged_write(directory / RETIREMENT_NAME) as fh:
            fh.write(json.dumps(body, indent=2, sort_keys=True).encode("utf-8"))

    @staticmethod
    def _is_retired(candidate: RecoveryCandidate, directory: Path) -> bool:
        through, all_revisions = RecoveryStore._read_retirement(directory)
        return all_revisions or candidate.revision <= through

    def write(
        self,
        state: ProjectState,
        revision: int,
        *,
        source_project_path: str | os.PathLike[str] | None = None,
        covered_explicit_save_revision: int = 0,
        timestamp: float | None = None,
    ) -> RecoveryCandidate | None:
        """Commit a recovery snapshot and rotate the former current generation.

        ``state`` should normally be the result of :func:`clone_project_state`, made
        on the GUI thread.  This method performs no mutation of it.
        """

        if not isinstance(state, ProjectState):
            raise TypeError("state must be a ProjectState")
        revision = _nonnegative_int(revision, "revision")
        covered = _nonnegative_int(
            covered_explicit_save_revision, "covered_explicit_save_revision")
        stamp = float(self._clock() if timestamp is None else timestamp)
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        metadata = RecoveryMetadata(
            self.session_id,
            revision,
            stamp,
            _normalise_source_path(source_project_path),
            covered,
        )

        with self._mutex:
            self._ensure_session_directory()
            retired_through, retired_all = self._read_retirement(self.session_directory)
            if retired_all or revision <= retired_through:
                # A delayed autosave which started before an explicit save/Discard
                # must not recreate the recovery that action just retired.
                return None
            current = self._valid_at(self.current_path, self.session_id)
            previous = self._valid_at(self.previous_path, self.session_id)
            newest = max(
                (item for item in (current, previous) if item is not None),
                key=lambda item: (item.revision, item.timestamp),
                default=None,
            )
            if newest is not None and revision < newest.revision:
                raise StaleRecoveryWrite(
                    f"recovery revision {revision} is older than committed revision "
                    f"{newest.revision}")
            if newest is not None and revision == newest.revision:
                # Timers may request the same immutable revision twice. Rotating it
                # would replace a useful previous generation with a duplicate.
                return newest

            payload = self._serialize_payload(state)
            envelope = self._envelope(metadata, payload)
            pending = atomic_write.temp_sibling(
                self.current_path, tag="saving", suffix=self.current_path.suffix)
            try:
                self._write_pending(pending, envelope, payload)
                # Validate the exact bytes before either good generation can move.
                candidate, _state = self._decode(
                    pending, expected_session=self.session_id, load_state=True)

                # A second check protects callers which coordinate retirement from
                # another RecoveryStore instance. The in-process GUI uses this same
                # object's lock, so its retire cannot interleave with this block.
                retired_through, retired_all = self._read_retirement(self.session_directory)
                if retired_all or revision <= retired_through:
                    _safe_unlink(pending)
                    return None

                # Never rotate a corrupt current over a known-good previous.  A crash
                # between these two atomic renames leaves the former current at
                # ``previous``, which discovery is explicitly designed to accept.
                if current is not None \
                        and (previous is None or current.revision >= previous.revision):
                    os.replace(self.current_path, self.previous_path)
                    _fsync_directory(self.session_directory)
                os.replace(pending, self.current_path)
                _fsync_directory(self.session_directory)
            except BaseException:
                _safe_unlink(pending)
                raise

            committed = RecoveryCandidate(
                session_id=candidate.session_id,
                revision=candidate.revision,
                timestamp=candidate.timestamp,
                source_project_path=candidate.source_project_path,
                covered_explicit_save_revision=candidate.covered_explicit_save_revision,
                envelope_path=self.current_path,
                generation="current",
            )
            # Cleanup is housekeeping after a durable commit. A failure to clean an
            # old lineage must not turn a successful recovery write into a reported
            # failure (which could prompt a duplicate retry/rotation).
            try:
                self.cleanup()
            except OSError:
                pass
            return committed

    def list_candidates(self) -> list[RecoveryCandidate]:
        """Return one newest valid candidate per session, newest first.

        ``current`` is authoritative when valid.  If it is incomplete, corrupt, or
        has mismatched session metadata, ``previous`` is tried.  Invalid material is
        never surfaced as a recoverable session.
        """

        if not self.root.exists():
            return []
        found: list[RecoveryCandidate] = []
        try:
            directories = list(self.root.iterdir())
        except OSError:
            return []
        for directory in directories:
            try:
                if directory.is_symlink() or not directory.is_dir() \
                        or not _is_canonical_session_dir(directory):
                    continue
            except OSError:
                continue
            candidate = self._valid_at(directory / CURRENT_NAME, directory.name)
            if candidate is None:
                candidate = self._valid_at(directory / PREVIOUS_NAME, directory.name)
            if candidate is not None and not self._is_retired(candidate, directory):
                found.append(candidate)
        found.sort(key=lambda c: (c.timestamp, c.revision, c.session_id), reverse=True)
        return found

    def load(self, candidate: RecoveryCandidate) -> RecoveredProject:
        """Load the exact candidate previously returned by discovery/``write``."""

        if not isinstance(candidate, RecoveryCandidate):
            raise TypeError("candidate must be a RecoveryCandidate")
        if self._is_retired(candidate, self.root / candidate.session_id):
            raise RecoveryChangedError("the selected recovery snapshot has been retired")
        decoded, state = self._decode(
            candidate.envelope_path, expected_session=candidate.session_id,
            load_state=True)
        expected = (
            candidate.session_id,
            candidate.revision,
            candidate.timestamp,
            candidate.source_project_path,
            candidate.covered_explicit_save_revision,
        )
        actual = (
            decoded.session_id,
            decoded.revision,
            decoded.timestamp,
            decoded.source_project_path,
            decoded.covered_explicit_save_revision,
        )
        if actual != expected:
            raise RecoveryChangedError(
                "the recovery snapshot changed after it was selected; rescan recoveries")
        assert state is not None
        return RecoveredProject(decoded, state)

    @staticmethod
    def _remove_session_directory(directory: Path) -> int:
        """Remove one UUID lineage without following nested directories/symlinks."""

        removed = 0
        if directory.is_symlink():
            return 0
        try:
            entries = list(directory.iterdir())
        except FileNotFoundError:
            return 0
        for entry in entries:
            try:
                # This module never creates a nested directory. Refusing to recurse
                # keeps cleanup tightly scoped if the on-disk tree is tampered with.
                if entry.is_dir() and not entry.is_symlink():
                    continue
                entry.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError:
                continue
        try:
            directory.rmdir()
        except OSError:
            pass
        return removed

    def retire_through(self, revision: int, *, session_id: str | uuid.UUID | None = None) -> int:
        """Retire generations in one lineage whose revision is ``<= revision``.

        This is the explicit-save fence.  A save of revision N must not delete a
        recovery of N+1 that landed while it was running.  Pass a recovered
        candidate's ``session_id`` and revision to retire that old lineage through
        exactly what was restored.
        """

        revision = _nonnegative_int(revision, "revision")
        sid = self.session_id if session_id is None else _canonical_session_id(session_id)
        directory = self.root / sid
        removed = 0
        with self._mutex:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if directory.is_symlink():
                raise RecoveryError(f"recovery session location may not be a symlink: {directory}")
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Fence FIRST. If unlinking fails or a delayed worker wakes afterwards,
            # discovery/write still cannot resurrect a covered revision.
            self._write_retirement(directory, revision, False)
            for path in (directory / CURRENT_NAME, directory / PREVIOUS_NAME):
                candidate = self._valid_at(path, sid)
                if candidate is not None and candidate.revision <= revision:
                    try:
                        path.unlink()
                        removed += 1
                    except FileNotFoundError:
                        pass
        return removed

    def retire_after_save(
        self,
        committed_revision: int,
        *,
        recovered_candidate: RecoveryCandidate | None = None,
    ) -> int:
        """Apply the current-session and optional recovered-lineage save fences."""

        removed = self.retire_through(committed_revision)
        if recovered_candidate is not None and recovered_candidate.session_id != self.session_id:
            removed += self.retire_through(
                recovered_candidate.revision,
                session_id=recovered_candidate.session_id,
            )
        return removed

    def discard(self, candidate_or_session: RecoveryCandidate | str | uuid.UUID | None = None) -> int:
        """Intentionally discard an entire lineage, including both generations.

        Deleting only the surfaced ``current`` would make ``previous`` appear on the
        next launch, contradicting the user's Discard choice.
        """

        if isinstance(candidate_or_session, RecoveryCandidate):
            sid = candidate_or_session.session_id
        elif candidate_or_session is None:
            sid = self.session_id
        else:
            sid = _canonical_session_id(candidate_or_session)
        with self._mutex:
            directory = self.root / sid
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if directory.is_symlink():
                raise RecoveryError(f"recovery session location may not be a symlink: {directory}")
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Persistent all-revisions tombstone before deletion: a queued writer
            # which has not entered write() yet becomes a harmless no-op.
            self._write_retirement(directory, 0, True)
            removed = 0
            for entry in list(directory.iterdir()):
                if entry.name == RETIREMENT_NAME:
                    continue
                try:
                    if entry.is_dir() and not entry.is_symlink():
                        continue
                    entry.unlink()
                    removed += 1
                except (FileNotFoundError, OSError):
                    pass
            return removed

    def cleanup(self, *, now: float | None = None) -> int:
        """Bound old UUID lineages by age and count; never remove this live session.

        The return value counts session directories removed (not generation files).
        Inaccessible/tampered directories are skipped rather than making recovery
        discovery or a completed write fail.
        """

        if not self.root.exists():
            return 0
        moment = float(self._clock() if now is None else now)
        if not math.isfinite(moment):
            raise ValueError("now must be finite")
        try:
            directories = [
                p for p in self.root.iterdir()
                if p.name != self.session_id and _is_canonical_session_dir(p)
                and not p.is_symlink() and p.is_dir()
            ]
        except OSError:
            return 0

        def touched(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        removed = 0
        survivors: list[Path] = []
        for directory in directories:
            if self.max_age_seconds is not None \
                    and moment - touched(directory) > self.max_age_seconds:
                self._remove_session_directory(directory)
                if not directory.exists():
                    removed += 1
                    continue
            survivors.append(directory)

        # Include this live session in the count when it exists. It is protected,
        # so all excess is paid by the oldest other lineages.
        live_count = 1 if self.session_directory.exists() else 0
        excess = max(0, len(survivors) + live_count - self.max_sessions)
        for directory in sorted(survivors, key=touched)[:excess]:
            self._remove_session_directory(directory)
            if not directory.exists():
                removed += 1
        return removed
