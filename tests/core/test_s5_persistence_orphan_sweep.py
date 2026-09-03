"""CR5-01 / EX5-01 — staging debris is removed AND disclosed; a failed commit frees its
own half-copied backup.

Two sweep-5 defects with one shape: the atomicity guarantee's RESIDUE outliving the write
in the user's own folders, silently.

* **CR5-01.** A SIGKILL / power loss mid-save leaves the hidden ``.<name>.saving-*`` temp
  behind (measured: one 35 MB partial; 464 MB after ten killed exports). No code path
  anywhere in ``nanoscope/`` ever removed or mentioned them — the next write in the same
  folder succeeded with an empty disclosure ledger. ``temp_sibling``'s own docstring
  concedes the identifiability ("that is what identifies the temp if one is ever left
  behind"); now :func:`atomic_write._sweep_stale_temps` acts on it at the seam every
  writer passes through, and the removal is SAID through the identity-notes channel.
* **EX5-01.** ``StagedSet.commit`` copies an existing target to a ``.replacing-*`` backup
  before replacing it; a failure MID-COPY (the measured case: ENOSPC with an 8 MB target
  on a nearly-full disk) left the partial backup on disk — it was never appended to
  ``backups``, and both the rollback and ``abandon`` only look at their own lists. Every
  retry leaked another. The failure path now discards the in-progress backup FIRST,
  freeing the space the user needs back.

The controls matter as much as the removals: a temp whose embedded pid is ALIVE — another
NanoScope's write in flight, or this process's own — is never touched.
"""

from __future__ import annotations

import errno
import os
import subprocess
from pathlib import Path

import pytest

from nanoscope.core import atomic_write
from nanoscope.core.atomic_write import (
    StagedSet,
    _sweep_stale_temps,
    staged_write,
    staged_write_set,
    take_identity_notes,
)


def _dead_pid() -> int:
    """A pid guaranteed to belong to no running process: a child we already reaped."""
    p = subprocess.Popen(["/usr/bin/true"])
    p.wait()
    # The pid is free the moment the child is reaped; recycling within this test's
    # lifetime would require the OS to wrap the whole pid space in milliseconds.
    return p.pid


def _orphan(directory: Path, stem: str = "data.csv", *, pid: int, tag: str = "saving",
            suffix: str = ".csv", payload: bytes = b"partial" * 100) -> Path:
    """A file named exactly the way temp_sibling names its temps."""
    p = directory / f".{stem}.{tag}-{pid}-abcdef12{suffix}"
    p.write_bytes(payload)
    return p


@pytest.fixture(autouse=True)
def _drained_ledger():
    """The identity ledger is process-global; start and end each test with it empty so a
    sentence queued here can neither leak out nor leak in."""
    take_identity_notes()
    yield
    take_identity_notes()


# ================================================================================================
# CR5-01 — the sweep, at each of the three seams
# ================================================================================================

def test_staged_write_removes_a_dead_pid_orphan_and_says_so(tmp_path):
    orphan = _orphan(tmp_path, pid=_dead_pid())
    target = tmp_path / "data.csv"
    with staged_write(target) as fh:
        fh.write(b"fresh")
    assert not orphan.exists(), "the crash orphan must be gone after the next write"
    assert target.read_bytes() == b"fresh"
    notes = take_identity_notes(target)
    assert any("leftover staging file" in n and "removed" in n for n in notes), notes


def test_staged_path_and_replacing_orphans_are_swept_too(tmp_path):
    saving = _orphan(tmp_path, pid=_dead_pid(), tag="saving", suffix=".tmp")
    replacing = _orphan(tmp_path, pid=_dead_pid(), tag="replacing", suffix=".tmp")
    target = tmp_path / "fig.pdf"
    with atomic_write.staged_path(target) as tmp:
        tmp.write_bytes(b"%PDF")
    assert not saving.exists() and not replacing.exists()
    notes = take_identity_notes(target)
    assert any("2 leftover staging file(s)" in n for n in notes), notes


def test_staged_set_sweeps_once_and_delivers_through_its_notes(tmp_path):
    orphan = _orphan(tmp_path, pid=_dead_pid())
    notes: list[str] = []
    with staged_write_set(notes=notes) as ss:
        for name in ("a.csv", "b.csv"):
            ss.stage(tmp_path / name).write_bytes(b"x")
    assert not orphan.exists()
    assert sum("leftover staging file" in n for n in notes) == 1, (
        f"one folder, one sentence — not one per member: {notes}")


def test_a_live_pid_temp_is_never_touched(tmp_path):
    """Another process's write in flight — or our own — is that write's own business."""
    own = _orphan(tmp_path, pid=os.getpid())
    # pid 1 (launchd) exists on every macOS/Linux box and is not ours to signal.
    other_alive = _orphan(tmp_path, stem="other.csv", pid=1)
    target = tmp_path / "data.csv"
    with staged_write(target) as fh:
        fh.write(b"fresh")
    assert own.exists(), "this process's own in-flight temp must survive the sweep"
    assert other_alive.exists(), "a live foreign pid's temp must survive the sweep"
    assert take_identity_notes(target) == []


def test_an_unrelated_dotfile_is_not_mistaken_for_debris(tmp_path):
    innocent = tmp_path / ".DS_Store"
    innocent.write_bytes(b"finder")
    lookalike = tmp_path / ".notes-about-saving-money.txt"
    lookalike.write_bytes(b"prose")
    with staged_write(tmp_path / "data.csv") as fh:
        fh.write(b"fresh")
    assert innocent.exists() and lookalike.exists()


def test_a_failed_write_still_discloses_the_sweep(tmp_path):
    """The removal happened whether or not this write landed; the sentence must not die
    with the failed write — it goes to the ledger."""
    orphan = _orphan(tmp_path, pid=_dead_pid())
    target = tmp_path / "data.csv"
    with pytest.raises(RuntimeError):
        with staged_write(target) as fh:
            fh.write(b"half")
            raise RuntimeError("the caller's own failure")
    assert not orphan.exists()
    assert not target.exists(), "the failed write must not land"
    notes = take_identity_notes(target)
    assert any("leftover staging file" in n for n in notes), notes


def test_an_abandoned_set_still_discloses_the_sweep(tmp_path):
    orphan = _orphan(tmp_path, pid=_dead_pid())
    ss = StagedSet()
    ss.stage(tmp_path / "a.csv").write_bytes(b"x")
    ss.abandon()
    assert not orphan.exists()
    notes = take_identity_notes(tmp_path / "a.csv")
    assert any("leftover staging file" in n for n in notes), notes


def test_a_stuck_removal_is_reported_as_still_there(tmp_path, monkeypatch):
    """A claim of success only on success: when unlink fails, the sentence says the file
    remains and that it is safe to delete by hand."""
    orphan = _orphan(tmp_path, pid=_dead_pid())
    real_unlink = os.unlink

    def deny(path, *a, **k):
        if Path(path).name == orphan.name:
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        return real_unlink(path, *a, **k)

    monkeypatch.setattr(os, "unlink", deny)
    sentences = _sweep_stale_temps(tmp_path)
    assert orphan.exists()
    assert any("could not be removed" in s for s in sentences), sentences
    assert not any("were found in" in s and "and removed" in s for s in sentences)


# ================================================================================================
# EX5-01 — the half-copied backup is freed on the failure path
# ================================================================================================

def _fail_copy_partway(monkeypatch, *, after_bytes: int = 1024):
    """Make _copy_file behave like ENOSPC mid-copy: dst is created, partially written,
    then the write raises — exactly what a full disk does."""
    def broken(src: Path, dst: Path) -> None:
        with open(src, "rb") as rfh, open(dst, "wb") as wfh:
            wfh.write(rfh.read(after_bytes))
        raise OSError(errno.ENOSPC, "No space left on device", str(dst))

    monkeypatch.setattr(atomic_write, "_copy_file", broken)


def test_commit_failure_mid_backup_copy_leaves_no_replacing_file(tmp_path, monkeypatch):
    target = tmp_path / "bigreport.csv"
    old = b"OLDBYTES" * 16_384
    target.write_bytes(old)

    _fail_copy_partway(monkeypatch)
    with pytest.raises(OSError) as err:
        with staged_write_set() as ss:
            ss.stage(target).write_bytes(b"NEW")
    # `restate_staging_failure` re-addresses the sentence to the user's file (type kept,
    # message rewritten), so the reason survives as text rather than as errno.
    assert "No space left on device" in str(err.value)

    assert target.read_bytes() == old, "the existing target is byte-for-byte untouched"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != target.name]
    assert leftovers == [], (
        f"a failed commit must leave nothing beside the target — found {leftovers}")


def test_commit_failure_on_a_later_member_still_frees_its_partial_backup(tmp_path,
                                                                         monkeypatch):
    """Member 1 has no existing target (no backup); member 2 does and its backup copy
    fails half-way. The whole set rolls back, and the partial backup goes with it."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    b_old = b"B-OLD" * 1000
    b.write_bytes(b_old)

    _fail_copy_partway(monkeypatch, after_bytes=16)
    with pytest.raises(OSError):
        with staged_write_set() as ss:
            ss.stage(a).write_bytes(b"A-NEW")
            ss.stage(b).write_bytes(b"B-NEW")

    assert not a.exists(), "the set must not split: member 1 is rolled back too"
    assert b.read_bytes() == b_old
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != b.name]
    assert leftovers == [], leftovers


def test_a_healthy_commit_still_replaces_and_cleans_up(tmp_path):
    """CONTROL: the failure-path work must cost the ordinary commit nothing."""
    target = tmp_path / "report.csv"
    target.write_bytes(b"old")
    with staged_write_set() as ss:
        ss.stage(target).write_bytes(b"new")
    assert target.read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir()] == [target.name]
