"""Content-aware, inter-process guard for project-file replacement.

Project archives are replaced atomically by :mod:`nanoscope.core.atomic_write`, which
prevents a torn file but does not by itself prevent two NanoScope processes from both
checking an old file and then replacing one another.  This module supplies the missing
compare-and-commit boundary:

* :func:`project_disk_stamp` fingerprints the bytes, not only metadata.  A writer that
  deliberately preserves ``st_size`` and ``st_mtime_ns`` is therefore still visible.
* :func:`guarded_project_write` holds one advisory inter-process lock across the final
  stamp comparison and the caller's atomic replacement.

The lock lives in the per-user temporary directory, never beside the researcher's
project.  Lock files are intentionally not unlinked: removing one while another process
is waiting on its inode can split later contenders across two different locks.  The OS
may sweep the per-user temporary directory between sessions; an open descriptor remains
the authority during a session and is released automatically if a process dies.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, TypeVar

__all__ = [
    "ProjectDiskStamp",
    "ProjectSaveConflict",
    "guarded_project_write",
    "locked_project_read",
    "project_changed",
    "project_disk_stamp",
    "project_save_lock",
    "project_save_lock_path",
]


class ProjectDiskStamp(NamedTuple):
    """The exact project revision observed on disk.

    The first three positions deliberately retain the historical
    ``(real_path, mtime_ns, size)`` layout used by :class:`MainWindow`; ``sha256`` adds
    content identity without breaking diagnostics/tests that index those positions.
    """

    real_path: str
    mtime_ns: int
    size: int
    sha256: str


class ProjectSaveConflict(RuntimeError):
    """The target no longer matches the revision the caller has seen.

    ``current_stamp`` is the precise revision for which an overwrite confirmation may
    be requested.  Passing that stamp back as ``approved_stamp`` authorises exactly that
    revision, not an intervening third writer.
    """

    def __init__(self, current_stamp: ProjectDiskStamp):
        super().__init__("project changed on disk")
        self.current_stamp = current_stamp


def _same_open_file(fd_stat: os.stat_result, path_stat: os.stat_result) -> bool:
    """Whether ``path_stat`` still names the file descriptor that was fingerprinted."""

    return (
        fd_stat.st_dev,
        fd_stat.st_ino,
        fd_stat.st_size,
        fd_stat.st_mtime_ns,
        fd_stat.st_ctime_ns,
    ) == (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_size,
        path_stat.st_mtime_ns,
        path_stat.st_ctime_ns,
    )


def _unchanged_while_hashed(before: os.stat_result, after: os.stat_result) -> bool:
    """Whether an in-place writer stayed out for the duration of fingerprinting."""

    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def project_disk_stamp(path: str | Path) -> ProjectDiskStamp | None:
    """Return a stable, content-aware stamp for ``path``, or ``None`` if it is absent.

    Opening before hashing means an atomic replacement cannot splice two archives into
    one digest: the descriptor continues to name either the old inode or the new one.
    The post-read path check detects that replacement and retries so the returned stamp
    describes what the path names at return time.  Under :func:`project_save_lock`, all
    cooperating NanoScope writers are excluded for the whole operation.
    """

    named = os.fspath(path)
    while True:
        real = os.path.realpath(named)
        try:
            with open(real, "rb") as fh:
                before_stat = os.fstat(fh.fileno())
                digest = hashlib.sha256()
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
                fd_stat = os.fstat(fh.fileno())
        except FileNotFoundError:
            return None
        except OSError:
            # Permission errors and I/O failures are material to saving and must reach
            # MainWindow's ordinary save-failure reporting path.  Only disappearance is
            # the established non-conflict case.
            raise

        try:
            path_stat = os.stat(real)
        except FileNotFoundError:
            return None
        # ``real`` must still be what the USER-NAMED path resolves to as well. Merely
        # checking the old destination would let a concurrently retargeted symlink return
        # a perfectly self-consistent stamp for a file the name no longer addresses.
        if (os.path.realpath(named) == real
                and _unchanged_while_hashed(before_stat, fd_stat)
                and _same_open_file(fd_stat, path_stat)):
            return ProjectDiskStamp(
                real, fd_stat.st_mtime_ns, fd_stat.st_size, digest.hexdigest())
        # The name was atomically replaced while it was being read.  Retry against the
        # inode now installed at the path; a cooperating writer cannot cause this while
        # our project lock is held, but this also makes standalone inspection honest.


def project_changed(expected: ProjectDiskStamp | None, path: str | Path,
                    current: ProjectDiskStamp | None = None, *,
                    expected_absent: bool = False) -> bool:
    """Whether ``path`` contains a newer/different revision than ``expected``.

    This preserves NanoScope's existing policy for a remembered project: a deleted target
    is not a conflict and may be recreated. For a new Save-As target, ``expected_absent``
    distinguishes "the chooser saw no file" from "do not compare" so a file another
    process creates before commit cannot be overwritten silently.
    """

    if expected is None:
        if not expected_absent:
            return False
        if current is None:
            current = project_disk_stamp(path)
        return current is not None
    if os.path.realpath(os.fspath(path)) != expected.real_path:
        # Callers use ``expected=None`` for a genuinely different Save-As target. With an
        # expected stamp, a different resolution means the named project symlink moved
        # under this window and must not silently inherit the old approval.
        if current is None:
            current = project_disk_stamp(path)
        return current is not None
    if current is None:
        current = project_disk_stamp(path)
    if current is None:
        return False
    return current != expected


def _lock_root() -> Path:
    """Return a private per-user lock directory, creating and validating it."""

    uid = getattr(os, "getuid", lambda: 0)()
    root = Path(tempfile.gettempdir()) / f"nanoscope-project-save-locks-{uid}"
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    st = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"project-save lock location is not a directory: {root}")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise PermissionError(f"project-save lock location is not owned by this user: {root}")
    # A prior umask or old build may have made this wider.  It contains no project data,
    # only hashed names, but keeping it private also prevents another user substituting a
    # lock file and splitting the advisory-lock domain.
    if st.st_mode & 0o077:
        os.chmod(root, 0o700)
    return root


def project_save_lock_path(path: str | Path) -> Path:
    """Stable lock path for the resolved project target.

    Hashing the resolved path makes a symlink spelling and its destination contend on
    the same lock, while keeping project names and locations out of the lock directory.
    """

    real = os.path.realpath(os.fspath(path))
    key = hashlib.sha256(os.fsencode(real)).hexdigest()
    return _lock_root() / f"{key}.lock"


def _open_lock_file(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(f"project-save lock is not a regular file: {path}")
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise PermissionError(f"project-save lock is not owned by this user: {path}")
        # Windows byte-range locks need an existing byte.  POSIX flock ignores this,
        # and writing once is safe because the lock file carries no state.
        if st.st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _acquire_lock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                retryable = {
                    errno.EACCES,
                    errno.EAGAIN,
                    getattr(errno, "EDEADLK", errno.EACCES),
                }
                if exc.errno not in retryable:
                    raise
                time.sleep(0.05)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


def _release_lock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def project_save_lock(path: str | Path) -> Iterator[None]:
    """Hold the exclusive inter-process lock for the resolved project target."""

    fd = _open_lock_file(project_save_lock_path(path))
    acquired = False
    try:
        _acquire_lock(fd)
        acquired = True
        yield
    finally:
        try:
            if acquired:
                _release_lock(fd)
        finally:
            os.close(fd)


_T = TypeVar("_T")


def guarded_project_write(
    path: str | Path,
    expected_stamp: ProjectDiskStamp | None,
    approved_stamp: ProjectDiskStamp | None,
    write: Callable[[], _T],
    *,
    expected_absent: bool = False,
) -> tuple[_T, ProjectDiskStamp | None]:
    """Compare and call ``write`` while holding the target's project-save lock.

    A changed revision is accepted only when it is byte-for-byte the same revision the
    caller explicitly approved.  If another process commits between that approval and
    this call, its different stamp raises a fresh :class:`ProjectSaveConflict` instead
    of becoming a lost update.  The caller's atomic writer runs before the lock is
    released, closing the check/write TOCTOU window among cooperating processes.
    """

    with project_save_lock(path):
        current = project_disk_stamp(path)
        if project_changed(expected_stamp, path, current,
                           expected_absent=expected_absent) \
                and current != approved_stamp:
            # project_changed can only be True for a present file.
            assert current is not None
            raise ProjectSaveConflict(current)
        result = write()
        return result, project_disk_stamp(path)


def locked_project_read(path: str | Path,
                        read: Callable[[], _T]) -> tuple[_T, ProjectDiskStamp | None]:
    """Read a project and stamp the same revision under one inter-process lock.

    Without this boundary a writer can commit after ``read`` returns but before the
    caller fingerprints the file.  The caller would then hold the old in-memory state
    while believing it had loaded the new revision, allowing its next save to overwrite
    that new work without a conflict.
    """

    with project_save_lock(path):
        result = read()
        return result, project_disk_stamp(path)
