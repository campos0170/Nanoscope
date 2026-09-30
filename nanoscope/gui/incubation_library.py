"""A persistent, manifest-only index of incubation workspaces.

The library never edits a study or loads scientific payloads. The launcher calls
``remember`` only after its normal engine loader has verified the full study.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time

from PySide6 import QtCore

from .incubation_profiles import peek_workspace


class StudyLibraryError(ValueError):
    """An actionable library error; the study itself may still be usable."""


@dataclass(frozen=True)
class LibraryEntry:
    path: Path
    label: str
    profile_id: str
    available: bool
    error: str


def _canonical(path):
    return Path(path).expanduser().resolve()


def default_library_root():
    """Return the configured default without touching the filesystem."""
    override = os.environ.get('NANOSCOPE_INCUBATION_LIBRARY_ROOT')
    if override:
        return Path(override).expanduser().absolute()
    documents = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation)
    return (Path(documents) if documents else Path.home()/'Documents')/'NanoScope'/'Incubation Studies'


def _unique_object(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f'Duplicate registry field: {key}.')
        values[key] = value
    return values


class StudyLibrary:
    """Saved workspaces plus immediate child workspaces of ``root``.

    An explicit root takes precedence over NANOSCOPE_INCUBATION_LIBRARY_ROOT.
    The default lives in the user's Documents/NanoScope/Incubation Studies.
    Registry mutations are serialized with QLockFile and atomic replacement;
    each writer rereads the latest registry while holding the lock.
    """

    def __init__(self, root=None):
        if root is None:
            root = default_library_root()
        try:
            self.root = _canonical(root)
            self.root.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise StudyLibraryError(f'Cannot access the incubation study library at {root}: {exc}') from exc
        self._registry = self.root/'studies.json'

    def _registry_error(self, detail):
        return StudyLibraryError(
            f'Cannot read the study library registry at {self._registry}: {detail} '
            'It has not been changed. Restore this file from a backup, or move it aside '
            'after keeping a copy and reopen the library. Study folders are unchanged.')

    def _read_registry(self):
        try:
            text = self._registry.read_text(encoding='utf-8')
        except FileNotFoundError:
            # A broken registry symlink is not an empty registry.
            if self._registry.is_symlink():
                raise self._registry_error('The registry link has a missing target.')
            return {'version': 1, 'entries': []}
        except (OSError, UnicodeError) as exc:
            raise self._registry_error(str(exc)) from exc
        try:
            data = json.loads(text, object_pairs_hook=_unique_object)
            if (not isinstance(data, dict) or type(data.get('version')) is not int
                    or data['version'] != 1 or not isinstance(data.get('entries'), list)):
                raise ValueError('Unsupported or damaged registry format; expected version 1.')
            for record in data['entries']:
                if not isinstance(record, dict):
                    raise ValueError('A registry entry is not an object.')
                if not isinstance(record.get('path'), str) or not record['path'].strip() or '\x00' in record['path']:
                    raise ValueError('A registry entry has an invalid path.')
                if not Path(record['path']).expanduser().is_absolute():
                    raise ValueError('Registry workspace paths must be absolute.')
                if not isinstance(record.get('label'), str) or not record['label'].strip():
                    raise ValueError('A registry entry has an invalid label.')
                if not isinstance(record.get('profile_id', ''), str):
                    raise ValueError('A registry entry has an invalid saved profile.')
                opened = record.get('last_opened', 0)
                if type(opened) not in (int, float) or not math.isfinite(opened) or opened < 0:
                    raise ValueError('A registry entry has an invalid last-opened time.')
            return data
        except (ValueError, TypeError) as exc:
            raise self._registry_error(str(exc)) from exc

    @staticmethod
    def _entry(path, record=None):
        label = record['label'] if record else path.name
        profile_id = record.get('profile_id', '') if record else ''
        try:
            if not path.exists():
                return LibraryEntry(path, label, profile_id, False,
                    'Study folder is missing or its linked location is unavailable. Browse to its current location to open it.')
            route = peek_workspace(path)
            return LibraryEntry(route.path, label if record else route.name, route.profile_id, True, '')
        except (OSError, ValueError, RuntimeError) as exc:
            return LibraryEntry(path, label, profile_id, False, str(exc))

    def entries(self):
        """Read index and immediate manifests; unavailable studies stay visible.

        ``available`` means routing metadata is readable, not that the complete
        checkpoint has passed the engine's integrity checks.
        """
        data = self._read_registry()
        result, seen = [], set()
        # Stable ordering preserves the most recently remembered entry on ties.
        for record in sorted(data['entries'], key=lambda x: x.get('last_opened', 0), reverse=True):
            try:
                path = _canonical(record['path'])
            except (OSError, RuntimeError, ValueError) as exc:
                result.append(LibraryEntry(Path(record['path']), record['label'], record.get('profile_id', ''), False, str(exc)))
                continue
            if path not in seen:
                seen.add(path)
                result.append(self._entry(path, record))
        try:
            children = sorted(self.root.iterdir(), key=lambda p: p.name.casefold())
        except OSError as exc:
            raise StudyLibraryError(f'Cannot list the incubation study library at {self.root}: {exc}') from exc
        for child in children:
            try:
                # Only immediate workspace manifests qualify. No recursive scan,
                # and no access to arrays or paths referenced by the manifest.
                manifest = child/'workspace.json'
                if not child.is_dir() or not (manifest.exists() or manifest.is_symlink()):
                    continue
                path = _canonical(child)
                if path not in seen:
                    seen.add(path)
                    result.append(self._entry(path))
            except (OSError, RuntimeError, ValueError):
                # Unreadable registered folders were already retained above.
                # Arbitrary child folders are not enough to identify a study.
                continue
        return result

    def _write_registry(self, data):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root,
                    prefix='.studies-', suffix='.json.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._registry)
            temporary = None
        except (OSError, ValueError) as exc:
            raise StudyLibraryError(
                f'Cannot save the incubation study library at {self._registry}: {exc}. '
                'The study folder is unchanged; check library permissions and available disk space.') from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def remember(self, folder, label=None):
        """Index a successfully opened/created study without changing its name.

        A supplied label is a library alias. An omitted label preserves an
        existing alias; otherwise the saved manifest name is used.
        """
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise StudyLibraryError('A library study label must contain text.')
        try:
            route = peek_workspace(folder)
        except (OSError, ValueError, RuntimeError) as exc:
            raise StudyLibraryError(f'This study cannot be added to the library: {exc}') from exc
        lock = QtCore.QLockFile(str(self.root/'studies.json.lock'))
        if not lock.tryLock(2000):
            raise StudyLibraryError(
                f'The study library at {self.root} is busy or cannot be locked. '
                'Close another NanoScope library update, check folder permissions, and retry.')
        try:
            data = self._read_registry()
            same, other = [], []
            for record in data['entries']:
                try:
                    matching = _canonical(record['path']) == route.path
                except (OSError, RuntimeError, ValueError):
                    matching = False
                (same if matching else other).append(record)
            same.sort(key=lambda x: x.get('last_opened', 0), reverse=True)
            alias = label.strip() if label is not None else same[0]['label'] if same else route.name
            # Keep unknown per-entry fields rather than silently losing them.
            record = dict(same[0]) if same else {}
            record.update(path=str(route.path), label=alias or route.path.name,
                          profile_id=route.profile_id, last_opened=int(time.time()))
            updated = dict(data, entries=[record, *other])
            if updated != data:
                self._write_registry(updated)
        finally:
            lock.unlock()
        return LibraryEntry(route.path, record['label'], route.profile_id, True, '')

    def picker_directory(self):
        """Start at the last opened available study, otherwise the library root."""
        data = self._read_registry()
        for record in sorted(data['entries'], key=lambda x: x.get('last_opened', 0), reverse=True):
            try:
                entry = self._entry(_canonical(record['path']), record)
            except (OSError, RuntimeError, ValueError):
                continue
            if entry.available:
                return str(entry.path)
        return str(self.root)
