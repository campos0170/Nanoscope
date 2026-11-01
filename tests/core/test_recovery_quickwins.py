"""Crash-recovery quick wins in the store (recoverability plan Q1, Q4, Q6, Q10).

Each test pins one failure the recoverability audit measured with synthetic data:

* s7: a second instance offered, and could Discard, a LIVE session's recovery;
* s11a/b: a deferred crash copy was evicted by 20 later sessions or by 30 days;
* s11c/d: a truncated or newer-build copy was skipped with no word;
* a malformed ``retirement.json`` hid a valid copy;
* Discard deleted at once, and the prompt could not say what a copy held;
* nothing in a copy said which build or process wrote it.

All roots are pytest temporary directories.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nanoscope.core.project import DocumentRef, ProjectState, ResultRec
from nanoscope.core.recovery import (
    CURRENT_NAME,
    DISCARDED_DIRNAME,
    LIVE_LOCK_NAME,
    PREVIOUS_NAME,
    RETIREMENT_NAME,
    RecoveryChangedError,
    RecoveryInUseError,
    RecoveryStore,
)

SID_A = "00000000-0000-0000-0000-00000000000a"
SID_B = "00000000-0000-0000-0000-00000000000b"
SID_C = "00000000-0000-0000-0000-00000000000c"
DAY = 24 * 60 * 60


def _state(tag: str = "one") -> ProjectState:
    events = pd.DataFrame({
        "start_idx": pd.Series([1, 8, 20], dtype="int64"),
        "end_idx": pd.Series([4, 12, 25], dtype="int64"),
        "accepted": pd.Series([True, False, True], dtype="bool"),
    })
    return ProjectState(
        documents=[DocumentRef(path=f"/data/{tag}.abf", loaded_range=None)],
        results=[ResultRec(doc_id=f"/data/{tag}.abf", channel=0,
                           module_id="detect.schmitt", events=events)],
        label_vocabulary=[{"key": "1", "value": tag}],
    )


def _ids(store: RecoveryStore) -> set[str]:
    return {c.session_id for c in store.list_candidates()}


# -- Q1: liveness ---------------------------------------------------------------------------------
def test_a_live_lineage_is_neither_offered_discarded_nor_cleaned_by_another_store(tmp_path):
    """s7 in the store: B could list A's running session and tombstone it."""
    root = tmp_path / "R"
    live = RecoveryStore(root, SID_A, hold_liveness=True)
    live.write(_state("live"), 3)
    assert live.holds_liveness
    assert (root / SID_A / LIVE_LOCK_NAME).exists()
    assert SID_A in _ids(live), "a session still sees its own copy"

    other = RecoveryStore(root, SID_B, max_age_seconds=0, max_work_age_seconds=0)
    assert SID_A not in _ids(other)
    with pytest.raises(RecoveryInUseError):
        other.discard(SID_A)
    assert not (root / SID_A / RETIREMENT_NAME).exists()
    other.write(_state("other"), 1)          # its write runs cleanup with age 0
    assert (root / SID_A / CURRENT_NAME).exists()

    # A keeps working, and its next copy still lands.
    assert live.write(_state("more"), 4) is not None

    live.release_liveness()                   # what the kernel does when A dies
    assert SID_A in _ids(other)
    other.discard(SID_A)
    assert SID_A not in _ids(other)


_HOLDER = r"""
import sys, time
from nanoscope.core.project import ProjectState
from nanoscope.core.recovery import RecoveryStore
store = RecoveryStore(sys.argv[1], sys.argv[2], hold_liveness=True)
store.write(ProjectState(label_vocabulary=[{"key": "1", "value": "x"}]), 2)
print("ready", flush=True)
time.sleep(60)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_the_kernel_releases_liveness_when_the_owning_process_dies(tmp_path):
    root = tmp_path / "R"
    tree = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(tree))
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(root), SID_A],
                            cwd=str(tree), env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        reader = RecoveryStore(root, SID_B)
        assert SID_A not in _ids(reader)
        proc.send_signal(signal.SIGKILL)
        proc.wait(30)
        assert SID_A in _ids(reader)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(30)


# -- Q10: who wrote it -----------------------------------------------------------------------------
def test_the_envelope_records_build_frozen_executable_and_pid(tmp_path):
    store = RecoveryStore(tmp_path / "R", SID_A)
    candidate = store.write(_state(), 1)
    with zipfile.ZipFile(candidate.envelope_path) as zf:
        writer = json.loads(zf.read("envelope.json"))["writer"]
    assert writer["pid"] == os.getpid()
    assert writer["frozen"] is False
    assert writer["executable"] == sys.executable
    assert writer["build_id"] and writer["version"]
    listed = store.list_candidates()[0]
    assert listed.writer["pid"] == os.getpid()


# -- Q4: never evict or hide silently -------------------------------------------------------------
def test_a_deferred_copy_survives_many_later_sessions_and_a_month(tmp_path):
    """s11a/s11b: 20 later saved sessions, or 30 days, used to take it."""
    root = tmp_path / "R"
    now = [1_000_000.0]
    crashed = RecoveryStore(root, SID_A, clock=lambda: now[0])
    crashed.write(_state("crashed"), 3)
    for _ in range(25):
        now[0] += 60
        later = RecoveryStore(root, clock=lambda: now[0])
        later.write(_state("later"), 1)
        later.retire_through(1)               # an ordinary save: a fence-only directory
    assert SID_A in _ids(RecoveryStore(root))

    os.utime(root / SID_A, (now[0] - 31 * DAY,) * 2)
    RecoveryStore(root, SID_B, clock=lambda: now[0]).write(_state("b"), 1)
    assert SID_A in _ids(RecoveryStore(root)), "a valid copy outlives the 30-day bound"

    os.utime(root / SID_A, (now[0] - 91 * DAY,) * 2)
    RecoveryStore(root, SID_C, clock=lambda: now[0]).write(_state("c"), 1)
    assert not (root / SID_A).exists(), "work ages out only after about 90 days"


def test_fence_only_directories_do_not_count_toward_the_cap(tmp_path):
    root = tmp_path / "R"
    for n in range(4):
        spent = RecoveryStore(root, max_sessions=2)
        spent.write(_state(str(n)), 1)
        spent.retire_through(1)
    live = RecoveryStore(root, SID_A, max_sessions=2)
    live.write(_state(), 1)
    assert live.cleanup() == 0
    assert len([p for p in root.iterdir() if p.is_dir()]) == 5


def _newer_build(path: Path) -> None:
    """Rewrite a generation so its inner project claims a version this build cannot read."""
    with zipfile.ZipFile(path) as outer:
        envelope = json.loads(outer.read("envelope.json"))
        payload = outer.read("project.nanoscope")
    inner = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as zin, zipfile.ZipFile(inner, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["version"] = 99
                data = json.dumps(manifest).encode()
            zout.writestr(item, data)
    payload = inner.getvalue()
    envelope["payload"]["size"] = len(payload)
    envelope["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
    envelope["writer"]["build_id"] = "a-newer-build"
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("envelope.json", json.dumps(envelope))
        outer.writestr("project.nanoscope", payload)


def test_unreadable_copies_are_reported_not_skipped(tmp_path):
    """s11c (both generations truncated) and s11d (a newer build's copy)."""
    root = tmp_path / "R"
    truncated = RecoveryStore(root, SID_A)
    truncated.write(_state("a"), 1)
    truncated.write(_state("b"), 2)
    for path in (truncated.current_path, truncated.previous_path):
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])
    newer = RecoveryStore(root, SID_B)
    newer.write(_state("c"), 5)
    _newer_build(newer.current_path)

    scan = RecoveryStore(root).scan()
    assert scan.candidates == []
    by_id = {item.session_id: item for item in scan.unreadable}
    assert set(by_id) == {SID_A, SID_B}
    assert by_id[SID_B].revision == 5
    assert by_id[SID_B].writer["build_id"] == "a-newer-build"
    assert "newer NanoScope" in by_id[SID_B].reason
    assert "nanoscope-recovery-read-" not in by_id[SID_B].reason

    # ...and cleanup keeps them by count as well as by the spent-lineage age.
    keeper = RecoveryStore(root, SID_C, max_sessions=1, max_age_seconds=0)
    keeper.write(_state(), 1)
    assert (root / SID_A).exists() and (root / SID_B).exists()


def test_a_malformed_retirement_marker_does_not_hide_a_valid_copy(tmp_path):
    root = tmp_path / "R"
    store = RecoveryStore(root, SID_A)
    store.write(_state(), 4)
    (root / SID_A / RETIREMENT_NAME).write_text("{not json", encoding="utf-8")
    reader = RecoveryStore(root, SID_B)
    assert SID_A in _ids(reader)
    candidate = next(c for c in reader.list_candidates() if c.session_id == SID_A)
    assert reader.load(candidate).state.label_vocabulary[0]["value"] == "one"
    # Writing stays conservative: a damaged fence still stops this lineage's own writer.
    assert store.write(_state("late"), 9) is None


# -- Q6: name the work; Discard is reversible for a week ------------------------------------------
def test_candidates_name_their_recordings_and_tables(tmp_path):
    store = RecoveryStore(tmp_path / "R", SID_A)
    store.write(_state("trace"), 1)
    summary = store.list_candidates()[0].summary
    assert summary.recordings == ("trace.abf",)
    assert summary.tables == (("detect.schmitt", "trace.abf", 3, 2),)


def test_discard_moves_the_lineage_to_discarded_and_it_can_be_restored(tmp_path):
    root = tmp_path / "R"
    now = [5_000_000.0]
    crashed = RecoveryStore(root, SID_A, clock=lambda: now[0])
    crashed.write(_state("one"), 1)
    crashed.write(_state("two"), 2)
    chooser = RecoveryStore(root, SID_B, clock=lambda: now[0])
    candidate = next(c for c in chooser.list_candidates() if c.session_id == SID_A)

    assert chooser.discard(candidate) == 2
    assert SID_A not in _ids(chooser)
    kept = root / DISCARDED_DIRNAME / SID_A
    assert sorted(p.name for p in kept.iterdir()) == sorted(
        [CURRENT_NAME, PREVIOUS_NAME, "discarded.json"])
    listed = chooser.list_discarded()
    assert [item.session_id for item in listed] == [SID_A]
    assert listed[0].candidate.revision == 2
    # The tombstone still stops a writer of that lineage while it is discarded.
    assert crashed.write(_state("late"), 3) is None

    restored = chooser.restore_discarded(listed[0])
    assert restored is not None and restored.revision == 2
    assert SID_A in _ids(chooser)
    assert chooser.load(restored).state.label_vocabulary[0]["value"] == "two"
    assert not kept.exists()


def test_discarded_lineages_are_kept_for_seven_days_then_purged(tmp_path):
    root = tmp_path / "R"
    now = [5_000_000.0]
    crashed = RecoveryStore(root, SID_A, clock=lambda: now[0])
    crashed.write(_state(), 1)
    chooser = RecoveryStore(root, SID_B, clock=lambda: now[0])
    chooser.discard(SID_A)
    now[0] += 6 * DAY
    chooser.cleanup()
    assert [item.session_id for item in chooser.list_discarded()] == [SID_A]
    now[0] += 2 * DAY
    chooser.cleanup()
    assert chooser.list_discarded() == []
    assert not (root / DISCARDED_DIRNAME / SID_A).exists()


def test_a_sessions_own_discard_moves_its_covered_copies_too(tmp_path):
    """The close guard's Discard of this session's work (``discard_through``)."""
    root = tmp_path / "R"
    store = RecoveryStore(root, SID_A, hold_liveness=True)
    store.write(_state("one"), 1)
    store.write(_state("two"), 2)
    assert store.discard_through(2) == 2
    assert SID_A not in _ids(store)
    assert (root / SID_A / LIVE_LOCK_NAME).exists(), "the live claim is never moved"
    assert [item.candidate.revision for item in store.list_discarded()] == [2]
    assert store.write(_state("three"), 3) is not None, "later work is still protected"
    assert store.discard_through(3) == 1
    assert len(store.list_discarded()) == 2, "a second discard does not replace the first"


def test_restoring_a_discard_never_overwrites_mixes_or_deletes_copies(tmp_path):
    """A restore into a lineage that holds copies again deleted the discarded ones."""
    root = tmp_path / "R"
    # (a) Discard at the Open guard, keep working, crash: the folder holds rev 6.
    store = RecoveryStore(root, SID_A, hold_liveness=True)
    store.write(_state("one"), 1)
    store.write(_state("two"), 2)
    store.discard_through(2)
    store.write(_state("six"), 6)
    store.release_liveness()                 # the crash
    chooser = RecoveryStore(root, SID_B)
    [entry] = chooser.list_discarded()
    with pytest.raises(RecoveryChangedError):
        chooser.restore_discarded(entry)
    assert sorted(p.name for p in entry.directory.iterdir()) == sorted(
        [CURRENT_NAME, PREVIOUS_NAME, "discarded.json"])
    [offered] = chooser.list_candidates()
    assert offered.revision == 6
    assert chooser.load(offered).state.label_vocabulary[0]["value"] == "six"

    # (b) Two guard Discards in one session, restored one after the other.
    store = RecoveryStore(root, SID_C, hold_liveness=True)
    for revision, tag in ((1, "one"), (2, "two")):
        store.write(_state(tag), revision)
    store.discard_through(2)
    for revision, tag in ((4, "four"), (5, "five")):
        store.write(_state(tag), revision)
    store.discard_through(5)
    store.release_liveness()
    entries = sorted((e for e in chooser.list_discarded() if e.session_id == SID_C),
                     key=lambda e: e.candidate.revision)
    assert [e.candidate.revision for e in entries] == [2, 5]
    assert chooser.restore_discarded(entries[0]).revision == 2
    with pytest.raises(RecoveryChangedError):
        chooser.restore_discarded(entries[1])
    assert entries[1].directory.is_dir(), "revisions 5 and 4 are still kept"
    chooser.discard(SID_C)                   # put revision 2 aside again, then 5 comes back
    assert chooser.restore_discarded(entries[1]).revision == 5
    revisions = sorted(e.candidate.revision for e in chooser.list_discarded()
                       if e.session_id == SID_C)
    assert revisions == [2]


def test_a_live_lineage_cannot_be_restored_into(tmp_path):
    root = tmp_path / "R"
    store = RecoveryStore(root, SID_A, hold_liveness=True)
    store.write(_state("one"), 1)
    store.discard_through(1)                 # the Open guard's Discard; the window lives on
    chooser = RecoveryStore(root, SID_B)
    [entry] = chooser.list_discarded()
    with pytest.raises(RecoveryInUseError):
        chooser.restore_discarded(entry)
    assert entry.directory.is_dir()
    store.release_liveness()
    assert chooser.restore_discarded(entry).revision == 1
