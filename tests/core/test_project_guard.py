"""The project revision guard closes metadata, TOCTOU, and first-save races."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from nanoscope.core.atomic_write import staged_write
from nanoscope.core.project_guard import (
    ProjectDiskStamp,
    ProjectSaveConflict,
    guarded_project_write,
    project_changed,
    project_disk_stamp,
    project_save_lock_path,
)


def _replace(path: str | Path, payload: bytes) -> None:
    with staged_write(path) as fh:
        fh.write(payload)


def _writer_process(path: str, expected: ProjectDiskStamp, payload: bytes, label: str,
                    started, entered, release, results) -> None:
    """Spawn-safe worker used by the deterministic two-process lock test."""

    started.set()

    def _write() -> None:
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test writer was not released")
        _replace(path, payload)

    try:
        _, stamp = guarded_project_write(path, expected, None, _write)
        results.put((label, "saved", None if stamp is None else stamp.sha256))
    except ProjectSaveConflict as exc:
        results.put((label, "conflict", exc.current_stamp.sha256))
    except BaseException as exc:  # pragma: no cover - reported verbatim on test failure
        results.put((label, "error", repr(exc)))


def test_digest_detects_same_size_same_mtime_rewrite(tmp_path):
    path = tmp_path / "study.nanoscope"
    path.write_bytes(b"curation-A")
    before = project_disk_stamp(path)
    assert before is not None
    old_stat = path.stat()

    path.write_bytes(b"curation-B")  # exactly the same byte count
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    after = project_disk_stamp(path)

    assert after is not None
    assert (after.mtime_ns, after.size) == (before.mtime_ns, before.size)
    assert after.sha256 != before.sha256
    assert project_changed(before, path, after) is True


def test_symlink_and_destination_use_one_private_lock(tmp_path):
    target = tmp_path / "actual.nanoscope"
    target.write_bytes(b"project")
    link = tmp_path / "current.nanoscope"
    try:
        link.symlink_to(target.name)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    target_lock = project_save_lock_path(target)
    link_lock = project_save_lock_path(link)

    assert link_lock == target_lock
    assert target_lock.parent != tmp_path
    assert not list(tmp_path.glob("*.lock")), "no lock artifact belongs beside user data"


def test_expected_absence_conflicts_with_another_first_save(tmp_path):
    """Two Save As operations that both chose an absent name cannot silently overwrite."""

    path = tmp_path / "new-project.nanoscope"
    guarded_project_write(
        path, None, None, lambda: _replace(path, b"first window"), expected_absent=True)

    called = False

    def _second_write() -> None:
        nonlocal called
        called = True
        _replace(path, b"second window")

    with pytest.raises(ProjectSaveConflict):
        guarded_project_write(
            path, None, None, _second_write, expected_absent=True)

    assert called is False
    assert path.read_bytes() == b"first window"


def test_approval_is_for_one_exact_revision(tmp_path):
    path = tmp_path / "study.nanoscope"
    path.write_bytes(b"window A")
    expected = project_disk_stamp(path)
    assert expected is not None

    _replace(path, b"window B")
    with pytest.raises(ProjectSaveConflict) as first:
        guarded_project_write(path, expected, None, lambda: _replace(path, b"mine"))
    approved_b = first.value.current_stamp

    _replace(path, b"window C")
    called = False

    def _mine() -> None:
        nonlocal called
        called = True
        _replace(path, b"mine")

    with pytest.raises(ProjectSaveConflict) as third:
        guarded_project_write(path, expected, approved_b, _mine)

    assert called is False
    assert third.value.current_stamp != approved_b
    assert path.read_bytes() == b"window C"


@pytest.mark.skipif(os.name == "nt", reason="spawned Windows lock timing is tested in CI")
def test_two_processes_cannot_both_pass_check_then_commit(tmp_path):
    """Writer B starts while A is paused inside its commit; B must conflict, not win."""

    path = tmp_path / "shared.nanoscope"
    path.write_bytes(b"common baseline")
    expected = project_disk_stamp(path)
    assert expected is not None

    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    a_started, a_entered, release_a = ctx.Event(), ctx.Event(), ctx.Event()
    b_started, b_entered, release_b = ctx.Event(), ctx.Event(), ctx.Event()
    release_b.set()
    writer_a = ctx.Process(
        target=_writer_process,
        args=(str(path), expected, b"writer A", "A",
              a_started, a_entered, release_a, results),
    )
    writer_b = ctx.Process(
        target=_writer_process,
        args=(str(path), expected, b"writer B", "B",
              b_started, b_entered, release_b, results),
    )

    writer_a.start()
    assert a_started.wait(10) and a_entered.wait(10), "writer A never acquired the lock"
    writer_b.start()
    assert b_started.wait(10), "writer B never reached the guarded write"
    release_a.set()
    writer_a.join(15)
    writer_b.join(15)
    if writer_a.is_alive():  # pragma: no cover - failure cleanup
        writer_a.terminate()
        writer_a.join()
    if writer_b.is_alive():  # pragma: no cover - failure cleanup
        writer_b.terminate()
        writer_b.join()

    assert writer_a.exitcode == 0
    assert writer_b.exitcode == 0
    outcomes = {label: outcome for label, outcome, _ in (results.get(), results.get())}
    assert outcomes == {"A": "saved", "B": "conflict"}
    assert b_entered.is_set() is False, "B's writer ran despite the changed revision"
    assert path.read_bytes() == b"writer A"
