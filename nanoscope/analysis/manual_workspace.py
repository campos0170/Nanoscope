"""Portable, non-pickle checkpoints for independently run study operations."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from .cancel import check_cancelled

FORMAT = 'nanoscope-manual-workspace'
VERSION = 1
_UNSPECIFIED_REVISION = object()
_LOCK_NAME = '.nanoscope-workspace.lock'


class WorkspaceConflictError(ValueError):
    """A checkpoint changed after this session read it."""


def _metadata_file(relative):
    """OS bookkeeping is never scientific output, including legacy inventories."""
    names = {'.DS_Store', '.Spotlight-V100', '.Trashes', '.fseventsd', '__MACOSX',
             'Thumbs.db', 'desktop.ini', _LOCK_NAME}
    return any(part in names or part.startswith('._') for part in Path(relative).parts)


def workspace_revision(folder):
    """Return the exact manifest revision, or None before its first publication."""
    try:
        data = (Path(folder).expanduser().resolve() / 'workspace.json').read_bytes()
    except FileNotFoundError:
        return None
    return hashlib.sha256(data).hexdigest()


def assert_workspace_revision(folder, expected_revision):
    """Cheap preflight; save repeats this check while holding the writer lock."""
    if workspace_revision(folder) != expected_revision:
        raise WorkspaceConflictError(
            'This workspace changed in another session. Your current settings and '
            'results have not been saved. Open the latest workspace before continuing.')


@contextmanager
def _writer_lock(folder, cancel):
    """OS-owned lock: a process exit releases it, even after an unhandled crash.

    Keep the inert lock file in place. Unlinking a lock file allows a new writer
    to lock a different inode while a queued writer still owns the original.
    """
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = folder / _LOCK_NAME
    if lock_path.is_symlink():
        raise ValueError('Workspace lock must not be a symbolic link')
    flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(lock_path, flags, 0o600)
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b'\0')
            def acquire():
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            def release():
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def acquire():
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def release():
                fcntl.flock(fd, fcntl.LOCK_UN)
        while not acquired:
            check_cancelled(cancel)
            try:
                acquire()
                acquired = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                time.sleep(.025)
        yield
    finally:
        if acquired:
            release()
        os.close(fd)


def stage_output_files(folder):
    """Snapshot files from one newly completed stage, excluding OS metadata.

    Call only for that stage's fresh output directory. Subsequent saves retain
    this explicit inventory and never adopt unrelated files added later.
    """
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Stage output folder must be a regular directory')
    result = []
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(folder)
        if _metadata_file(relative):
            continue
        if path.is_symlink():
            raise ValueError(f'Owned workspace outputs must not be symbolic links: {relative}')
        if path.is_file():
            result.append(path)
    return tuple(result)


def _sha(path, cancel=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            check_cancelled(cancel)
            digest.update(block)
    return digest.hexdigest()


def _registry():
    from .incubation_detection import PreparedRecording, RecordingAnalysis
    from .incubation_study import StudyRecording
    from .manual_preparation import RawPreparedRecording
    from .manual_detection import InferenceResult, InferenceSettings, DecodeSettings, MeasurementSettings
    from .manual_comparisons import ComparisonResult, ComparisonSettings, ClusterSettings, MixtureSettings
    from . import subpop_gmm as sg
    classes = [PreparedRecording, RecordingAnalysis, RawPreparedRecording, StudyRecording,
               InferenceResult, InferenceSettings, DecodeSettings, MeasurementSettings,
               ComparisonResult, ComparisonSettings, ClusterSettings, MixtureSettings,
               sg.PCSelection, sg.SubpopFit, sg.Separation, sg.FractionCI, sg.SubpopQuantification]
    return {c.__name__: c for c in classes}


def _inside(folder, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Workspace paths must be relative')
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder):
        raise ValueError('Workspace path escapes its folder')
    return path


def _verify_inventory(folder, inventory, cancel=None):
    if not isinstance(inventory, dict):
        raise ValueError('Invalid workspace file inventory')
    for relative, digest in inventory.items():
        check_cancelled(cancel)
        path = _inside(folder, relative)
        if not path.is_file() or _sha(path, cancel) != digest:
            raise ValueError(f'Missing or changed workspace data: {relative}')


def _owned_path(folder, relative):
    path = _inside(folder, relative)
    unresolved = folder / relative
    for parent in (unresolved, *unresolved.parents):
        if parent == folder:
            break
        if parent.is_symlink():
            raise ValueError(f'Owned workspace outputs must not be symbolic links: {relative}')
    return path


def _owned_inventory(folder, meta):
    """Migrate only known bookkeeping entries; retain all real legacy outputs."""
    inventory = meta.get('owned_file_sha256', {})
    if not isinstance(inventory, dict):
        raise ValueError('Invalid workspace file inventory')
    owned = {}
    for relative, digest in inventory.items():
        _inside(folder, relative)
        if not _metadata_file(relative):
            _owned_path(folder, relative)
            owned[relative] = digest
    return owned


def _npy_digest(array):
    """Hash canonical NPY bytes without allocating a second whole-array buffer."""
    digest = hashlib.sha256()
    class Sink:
        def write(self, data):
            digest.update(data)
            return len(data)
        def flush(self):
            pass
    np.save(Sink(), array, allow_pickle=False)
    return digest.hexdigest()


def save_workspace(state, folder, *, cancel=None,
                   expected_revision=_UNSPECIFIED_REVISION, owned_outputs=()):
    """Atomically publish a checkpoint, validating immutable data before reuse.

    Existing arrays are reused only when their canonical bytes match the current
    in-memory value. Corrupt blobs are repaired atomically from that value. Saved
    stage files are explicitly registered through owned_outputs and cannot be
    silently blessed after alteration. Local file Paths in state are also owned.
    Original source/model references remain external and are not read on reopen.

    Return the published manifest's SHA-256 revision. Pass the revision returned
    by load_workspace(..., with_revision=True) on every session save, or None
    when creating a new workspace. Omitting it preserves one-shot codec callers,
    with a checked revision captured at function entry (not stale-session safety).
    """
    folder = Path(folder).expanduser().resolve()
    check_cancelled(cancel)
    if expected_revision is _UNSPECIFIED_REVISION:
        expected_revision = workspace_revision(folder)
    elif expected_revision is not None and (not isinstance(expected_revision, str) or
            len(expected_revision) != 64 or any(c not in '0123456789abcdef' for c in expected_revision)):
        raise ValueError('Invalid workspace revision')
    with _writer_lock(folder, cancel):
        assert_workspace_revision(folder, expected_revision)
        return _save_locked(state, folder, cancel=cancel, owned_outputs=owned_outputs)


def _save_locked(state, folder, *, cancel, owned_outputs):
    manifest_path = folder / 'workspace.json'
    if any(not _metadata_file(p.name) for p in folder.iterdir()) and not manifest_path.exists():
        raise ValueError('Choose a new/empty folder or an existing manual study workspace.')
    owned = {}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get('format') != FORMAT or prior.get('version') != VERSION:
            raise ValueError('Not a supported manual study workspace.')
        # An external edit to a saved figure/table must not become legitimate just
        # because an unrelated setting was saved. Array blobs can be repaired below.
        owned = _owned_inventory(folder, prior)
        _verify_inventory(folder, owned, cancel)
    arrays = folder / 'arrays'
    arrays.mkdir(exist_ok=True)
    if arrays.resolve() != arrays:
        raise ValueError('Workspace arrays directory must not be a symbolic link')
    inventory, memo, local_files = {}, {}, set()
    registry = _registry()

    def pack(value):
        check_cancelled(cancel)
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if value is pd.NA:
            return {'type': 'missing'}
        if value is pd.NaT:
            return {'type': 'nat'}
        if isinstance(value, pd.Timestamp):
            return {'type': 'timestamp', 'value': value.isoformat()}
        if isinstance(value, pd.Timedelta):
            return {'type': 'timedelta', 'value': int(value.value)}
        if isinstance(value, np.generic):
            return {'type': 'numpy_scalar', 'array': pack(np.asarray(value))}
        if isinstance(value, float):
            return value if math.isfinite(value) else {'type': 'float', 'value': str(value)}
        if isinstance(value, Path):
            path = value.resolve()
            if path.is_relative_to(folder) and path.is_file():
                # Exact file references are protected; a directory reference is
                # not permission to adopt every user file inside that directory.
                relative = value.absolute().relative_to(folder).as_posix()
                _owned_path(folder, relative)
                if not _metadata_file(relative):
                    local_files.add(relative)
            return {'type': 'workspace_path', 'value': path.relative_to(folder).as_posix()} if path.is_relative_to(folder) else {'type': 'path', 'value': str(value)}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                if value.dtype.fields:
                    raise ValueError('Structured arrays with object fields are not supported')
                return {'type': 'object_array_flat', 'shape': list(value.shape),
                        'values': [pack(v) for v in value.flat]}
            if id(value) in memo:
                return memo[id(value)][1]
            # ascontiguousarray promotes a scalar to shape (1,); asarray preserves ().
            contiguous = np.asarray(value, order='C')
            digest = _npy_digest(contiguous)
            relative = f'arrays/{digest}.npy'
            path = folder / relative
            if not path.is_file() or _sha(path, cancel) != digest:
                fd, temporary = tempfile.mkstemp(prefix='.array-', suffix='.npy', dir=arrays)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        np.save(stream, contiguous, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    check_cancelled(cancel)
                    if _sha(temporary, cancel) != digest:
                        raise ValueError('Array changed while checkpointing; retry the operation')
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary): os.unlink(temporary)
            inventory[relative] = digest
            packed = {'type': 'array', 'path': relative}
            # Hold the object as well as its ID. Temporary Series.to_numpy arrays
            # may otherwise be freed and reuse an earlier ID in this same save.
            memo[id(value)] = (value, packed)
            return packed
        if isinstance(value, pd.Index):
            if isinstance(value, pd.RangeIndex):
                return {'type': 'range_index', 'start': value.start, 'stop': value.stop,
                        'step': value.step, 'name': pack(value.name)}
            if isinstance(value, pd.MultiIndex):
                return {'type': 'multi_index', 'levels': [pack(v) for v in value.levels],
                        'codes': [pack(np.asarray(v)) for v in value.codes], 'names': pack(value.names)}
            if isinstance(value, pd.CategoricalIndex):
                return {'type': 'categorical_index', 'data': pack(value.array), 'name': pack(value.name)}
            return {'type': 'index', 'values': pack(value.to_numpy()), 'dtype': str(value.dtype),
                    'name': pack(value.name), 'freq': getattr(value, 'freqstr', None)}
        if isinstance(value, pd.Categorical):
            return {'type': 'categorical', 'codes': pack(value.codes),
                    'categories': pack(value.categories), 'ordered': value.ordered}
        if isinstance(value, pd.DataFrame):
            dtypes = [{'dtype': str(v), 'storage': v.storage} if isinstance(v, pd.StringDtype)
                      else {'dtype': str(v)} for v in value.dtypes]
            return {'type': 'frame_v2', 'columns': pack(value.columns), 'index': pack(value.index),
                    'data': [pack(value.iloc[:, i].array if isinstance(value.dtypes.iloc[i], pd.CategoricalDtype)
                                  else value.iloc[:, i].to_numpy()) for i in range(len(value.columns))],
                    'dtypes': dtypes, 'attrs': pack(value.attrs)}
        if isinstance(value, pd.Series):
            return {'type': 'series', 'frame': pack(value.to_frame(name='value')), 'name': pack(value.name)}
        if is_dataclass(value):
            name = type(value).__name__
            if registry.get(name) is not type(value):
                raise ValueError(f'Unsupported workspace result type: {name}')
            return {'type': 'result', 'class': name, 'fields': pack({f.name: getattr(value, f.name) for f in fields(value)})}
        if isinstance(value, dict):
            return {'type': 'mapping', 'items': [[pack(k), pack(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {'type': 'tuple' if isinstance(value, tuple) else 'list', 'values': [pack(v) for v in value]}
        raise ValueError(f'Unsupported checkpoint value: {type(value).__name__}')

    payload = pack(state)
    for value in owned_outputs:
        check_cancelled(cancel)
        path = Path(value)
        relative = path.absolute().relative_to(folder).as_posix() if path.is_absolute() else path.as_posix()
        _owned_path(folder, relative)
        if not _metadata_file(relative):
            local_files.add(relative)
    for relative in sorted(local_files):
        check_cancelled(cancel)
        if relative in owned:
            continue
        path = _owned_path(folder, relative)
        if Path(relative).parts[0] == 'arrays' or path == manifest_path or path.name.startswith(('.workspace-', '.array-')):
            raise ValueError(f'Not a stage output: {relative}')
        if not path.is_file():
            raise ValueError(f'Missing workspace output: {relative}')
        owned[relative] = _sha(path, cancel)
    manifest = dict(format=FORMAT, version=VERSION, saved_utc=datetime.now(timezone.utc).isoformat(),
                    array_sha256=inventory, owned_file_sha256=owned, state=payload)
    data = json.dumps(manifest, allow_nan=False, separators=(',', ':')).encode()
    fd, temporary = tempfile.mkstemp(prefix='.workspace-', suffix='.json', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        check_cancelled(cancel)
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def load_workspace(folder, *, cancel=None, with_revision=False):
    """Restore saved stages without inference or original source access."""
    folder = Path(folder).expanduser().resolve()
    data = (folder / 'workspace.json').read_bytes()
    revision = hashlib.sha256(data).hexdigest()
    meta = json.loads(data)
    if meta.get('format') != FORMAT or meta.get('version') != VERSION:
        raise ValueError('Not a supported manual study workspace.')
    registry, cache = _registry(), {}
    inventory = meta.get('array_sha256', {})
    _verify_inventory(folder, inventory, cancel)
    _verify_inventory(folder, _owned_inventory(folder, meta), cancel)

    def unpack(value):
        check_cancelled(cancel)
        if not isinstance(value, dict): return value
        kind = value.get('type')
        if kind == 'missing': return pd.NA
        if kind == 'nat': return pd.NaT
        if kind == 'timestamp': return pd.Timestamp(value['value'])
        if kind == 'timedelta': return pd.Timedelta(value['value'], unit='ns')
        if kind == 'numpy_scalar': return unpack(value['array'])[()]
        if kind == 'float':
            if value['value'] not in ('nan', 'inf', '-inf'): raise ValueError('Invalid non-finite value')
            return float(value['value'])
        if kind == 'path': return Path(value['value'])
        if kind == 'workspace_path': return _inside(folder, value['value'])
        if kind == 'array':
            relative = value['path']
            if relative not in inventory: raise ValueError('Unverified array reference')
            if relative not in cache:
                path = _inside(folder, relative)
                # Verify bytes and load through the same opened descriptor. Atomic
                # replacement of its directory entry cannot swap in unverified data.
                with path.open('rb') as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024*1024), b''):
                        check_cancelled(cancel); digest.update(block)
                    if digest.hexdigest() != inventory[relative]:
                        raise ValueError(f'Missing or changed workspace data: {relative}')
                    stream.seek(0)
                    cache[relative] = np.load(stream, allow_pickle=False)
            return cache[relative]
        if kind == 'object_array_flat':
            shape = tuple(value['shape'])
            array = np.empty(shape, dtype=object)
            if array.size != len(value['values']): raise ValueError('Invalid object array shape')
            for i, item in enumerate(value['values']): array.flat[i] = unpack(item)
            return array
        if kind == 'object_array': return np.asarray(unpack(value['values']), dtype=object).reshape(value['shape'])
        if kind in ('list', 'tuple'):
            items = [unpack(v) for v in value['values']]
            return tuple(items) if kind == 'tuple' else items
        if kind == 'mapping': return {unpack(k): unpack(v) for k, v in value['items']}
        if kind == 'range_index': return pd.RangeIndex(value['start'], value['stop'], value['step'], name=unpack(value['name']))
        if kind == 'multi_index':
            return pd.MultiIndex(levels=[unpack(v) for v in value['levels']], codes=[unpack(v) for v in value['codes']], names=unpack(value['names']))
        if kind == 'categorical':
            return pd.Categorical.from_codes(unpack(value['codes']), categories=unpack(value['categories']), ordered=value['ordered'])
        if kind == 'categorical_index': return pd.CategoricalIndex(unpack(value['data']), name=unpack(value['name']))
        if kind == 'index':
            result = pd.Index(unpack(value['values']), dtype=value['dtype'], name=unpack(value['name']), tupleize_cols=False)
            if value.get('freq') is not None and isinstance(result, (pd.DatetimeIndex, pd.TimedeltaIndex)):
                result.freq = value['freq']
            return result
        if kind in ('frame', 'frame_v2'):
            columns, index = unpack(value['columns']), unpack(value['index'])
            if len(value['data']) != len(columns) or len(value['dtypes']) != len(columns):
                raise ValueError('Invalid DataFrame column metadata')
            series=[]
            for encoded, spec in zip(value['data'], value['dtypes']):
                spec = {'dtype': spec} if isinstance(spec, str) else spec
                array = unpack(encoded)
                dtype = pd.StringDtype(storage=spec['storage']) if 'storage' in spec else spec['dtype']
                # Retain categorical codes, unused levels and order from the array.
                if isinstance(array, pd.Categorical): dtype = array.dtype
                series.append(pd.Series(array, index=index, dtype=dtype))
            frame = pd.concat(series, axis=1) if series else pd.DataFrame(index=index)
            frame.columns = columns if kind == 'frame_v2' else pd.Index(columns, name=unpack(value['column_name']))
            if kind == 'frame': frame.index.name = unpack(value['index_name'])
            frame.attrs = unpack(value['attrs'])
            return frame
        if kind == 'series':
            result = unpack(value['frame']).iloc[:, 0]; result.name = unpack(value['name']); return result
        if kind == 'result':
            cls = registry.get(value['class'])
            if cls is None: raise ValueError('Unsupported workspace result class')
            return cls(**unpack(value['fields']))
        raise ValueError(f'Unsupported workspace encoding: {kind}')
    state = unpack(meta['state'])
    return (state, revision) if with_revision else state
