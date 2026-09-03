"""Stage 2 crash-recovery store: durable generations, validation and retirement."""

from __future__ import annotations

import errno
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nanoscope.core.recovery as recovery
from nanoscope.core.project import (
    VERSION as PROJECT_VERSION,
    DocumentRef,
    FitRec,
    ProjectState,
    ResultRec,
)
from nanoscope.core.recovery import (
    CURRENT_NAME,
    ENVELOPE_NAME,
    PAYLOAD_NAME,
    PREVIOUS_NAME,
    RETIREMENT_NAME,
    RECOVERY_FORMAT,
    RECOVERY_VERSION,
    RecoveryChangedError,
    RecoveryError,
    RecoveryStore,
    StaleRecoveryWrite,
    clone_project_state,
    default_recovery_root,
)


SID1 = "00000000-0000-0000-0000-000000000001"
SID2 = "00000000-0000-0000-0000-000000000002"
SID3 = "00000000-0000-0000-0000-000000000003"


def _state(tag: str = "one") -> ProjectState:
    events = pd.DataFrame({
        "start_idx": pd.Series([1, 8], dtype="int64"),
        "end_idx": pd.Series([4, 12], dtype="int64"),
        "accepted": pd.Series([True, False], dtype="bool"),
        "source": pd.Series(["manual", "schmitt"], dtype="string"),
    })
    return ProjectState(
        documents=[DocumentRef(path=f"/data/{tag}.abf", loaded_range=(0, 100))],
        results=[ResultRec(
            doc_id=f"/data/{tag}.abf",
            channel=0,
            module_id=f"detect.{tag}",
            module_version="2.3",
            events=events,
            provenance={"tag": tag, "nested": {"thresholds": [1.0, 2.0]}},
            params={"sigma": 4.5},
        )],
        measurements=pd.DataFrame({"label": pd.Series([tag], dtype="string"),
                                   "value": [1.25]}),
        fits=[FitRec(doc_id=f"/data/{tag}.abf", channel=0, module_id="fit.depth",
                    curve={"params": {"mu": 1.2}, "model": "normal"})],
        label_vocabulary=[{"key": "1", "value": tag}],
        colours=[{"key": f"/data/{tag}.abf::0", "slot": 2, "ink": "#123456"}],
    )


def _store(tmp_path: Path, sid: str = SID1, **kwargs) -> RecoveryStore:
    return RecoveryStore(tmp_path / "recoveries", session_id=sid, **kwargs)


def _envelope(path: Path) -> tuple[dict, bytes]:
    with zipfile.ZipFile(path) as outer:
        return json.loads(outer.read(ENVELOPE_NAME)), outer.read(PAYLOAD_NAME)


def test_default_root_honours_the_environment_without_touching_cwd(monkeypatch, tmp_path):
    chosen = tmp_path / "managed" / "recovery"
    monkeypatch.setenv("NANOSCOPE_RECOVERY_ROOT", str(chosen))
    assert default_recovery_root() == chosen
    store = RecoveryStore(session_id=SID1)
    assert store.root == chosen
    assert not chosen.exists()  # construction itself writes nothing


def test_each_application_session_gets_a_unique_uuid_directory(tmp_path):
    one = RecoveryStore(tmp_path)
    two = RecoveryStore(tmp_path)
    assert one.session_id != two.session_id
    assert str(uuid.UUID(one.session_id)) == one.session_id
    assert one.session_directory != two.session_directory


def test_recovery_envelope_and_project_payload_are_independently_versioned(tmp_path):
    store = _store(tmp_path)
    candidate = store.write(_state(), 7, source_project_path="study.nanoscope",
                            covered_explicit_save_revision=4, timestamp=1234.5)
    assert candidate is not None
    envelope, payload = _envelope(candidate.path)
    assert envelope["format"] == RECOVERY_FORMAT
    assert envelope["version"] == RECOVERY_VERSION
    assert envelope["payload"]["version"] == PROJECT_VERSION == 10
    payload_path = tmp_path / "inner.nanoscope"
    payload_path.write_bytes(payload)
    with zipfile.ZipFile(payload_path) as inner:
        manifest = json.loads(inner.read("manifest.json"))
    assert manifest["version"] == 10
    assert envelope["metadata"]["session_uuid"] == SID1
    assert envelope["metadata"]["revision"] == 7
    assert envelope["metadata"]["covered_explicit_save_revision"] == 4


def test_round_trip_returns_normal_project_state_and_all_candidate_metadata(tmp_path):
    store = _store(tmp_path)
    source = tmp_path / "saved" / "study.nanoscope"
    candidate = store.write(_state("roundtrip"), 9,
                            source_project_path=source,
                            covered_explicit_save_revision=6,
                            timestamp=2345.0)
    assert candidate is not None
    assert candidate.session_id == SID1
    assert candidate.revision == 9
    assert candidate.timestamp == 2345.0
    assert candidate.source_project_path == str(source.absolute())
    assert candidate.covered_explicit_save_revision == 6
    assert candidate.envelope_path == candidate.path
    assert candidate.payload_member == PAYLOAD_NAME
    restored = store.load(candidate)
    assert isinstance(restored.state, ProjectState)
    assert restored.state.documents[0].path == "/data/roundtrip.abf"
    assert restored.state.results[0].module_id == "detect.roundtrip"
    pd.testing.assert_frame_equal(restored.state.results[0].events,
                                  _state("roundtrip").results[0].events)


def test_only_two_atomic_generations_are_kept(tmp_path):
    store = _store(tmp_path)
    for revision in (1, 2, 3):
        assert store.write(_state(str(revision)), revision, timestamp=float(revision))
    names = {p.name for p in store.session_directory.iterdir()}
    assert names == {CURRENT_NAME, PREVIOUS_NAME}
    current = store.validate(store.current_path)
    previous = store.validate(store.previous_path)
    assert (current.revision, previous.revision) == (3, 2)


def test_a_corrupt_newest_generation_falls_back_to_previous(tmp_path):
    store = _store(tmp_path)
    store.write(_state("old"), 1, timestamp=1)
    store.write(_state("new"), 2, timestamp=2)
    store.current_path.write_bytes(b"PK\x03\x04 interrupted")
    candidates = store.list_candidates()
    assert len(candidates) == 1
    assert candidates[0].generation == "previous"
    assert candidates[0].revision == 1
    assert store.load(candidates[0]).state.documents[0].path == "/data/old.abf"


def test_corrupt_or_incomplete_sessions_are_not_listed(tmp_path):
    store = _store(tmp_path)
    store.write(_state(), 1)
    store.current_path.write_bytes(b"not a zip")
    store.previous_path.write_bytes(b"also broken")
    (store.session_directory / ".current.saving-99999999-deadbeef.recovery").write_bytes(
        b"partial")
    assert store.list_candidates() == []
    with pytest.raises(RecoveryError):
        store.validate(store.current_path)


def test_session_metadata_must_match_its_uuid_directory(tmp_path):
    store = _store(tmp_path, SID1)
    candidate = store.write(_state(), 1)
    assert candidate is not None
    other_dir = store.root / SID2
    other_dir.mkdir()
    (other_dir / CURRENT_NAME).write_bytes(candidate.path.read_bytes())
    listed = store.list_candidates()
    assert [item.session_id for item in listed] == [SID1]


def test_candidates_are_sorted_newest_first_across_sessions(tmp_path):
    first = _store(tmp_path, SID1)
    second = _store(tmp_path, SID2)
    third = _store(tmp_path, SID3)
    first.write(_state("a"), 3, timestamp=30)
    second.write(_state("b"), 1, timestamp=10)
    third.write(_state("c"), 2, timestamp=20)
    assert [c.session_id for c in first.list_candidates()] == [SID1, SID3, SID2]


def test_disk_full_while_serializing_costs_no_committed_generation(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.write(_state("one"), 1)
    before = store.current_path.read_bytes()

    def full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(recovery, "save_project", full)
    with pytest.raises(OSError) as caught:
        store.write(_state("two"), 2)
    assert caught.value.errno == errno.ENOSPC
    assert store.current_path.read_bytes() == before
    assert not store.previous_path.exists()
    assert not list(store.session_directory.glob(".payload-*"))


def test_mid_envelope_write_costs_no_committed_generation_or_temp(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.write(_state("one"), 1)
    before = store.current_path.read_bytes()
    real = zipfile.ZipFile.writestr

    def fail_payload(self, name, data, *args, **kwargs):
        if str(getattr(name, "filename", name)) == PAYLOAD_NAME:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, name, data, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail_payload)
    with pytest.raises(OSError):
        store.write(_state("two"), 2)
    assert store.current_path.read_bytes() == before
    assert not store.previous_path.exists()
    assert {p.name for p in store.session_directory.iterdir()} == {CURRENT_NAME}


def test_failure_after_rotation_still_leaves_previous_recoverable(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.write(_state("old"), 1)
    real_replace = os.replace

    def fail_final(source, destination):
        if Path(destination) == store.current_path:
            raise OSError(errno.EIO, "injected final rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(recovery.os, "replace", fail_final)
    with pytest.raises(OSError):
        store.write(_state("new"), 2)
    candidates = store.list_candidates()
    assert len(candidates) == 1
    assert candidates[0].generation == "previous"
    assert candidates[0].revision == 1


def test_write_denial_leaves_the_old_generation_readable(tmp_path, monkeypatch):
    store = _store(tmp_path)
    old = store.write(_state("old"), 1)
    assert old is not None
    before = old.path.read_bytes()

    def denied(*_args, **_kwargs):
        raise PermissionError(errno.EACCES, "read-only recovery location")

    monkeypatch.setattr(RecoveryStore, "_write_pending", staticmethod(denied))
    with pytest.raises(PermissionError):
        store.write(_state("new"), 2)
    assert store.current_path.read_bytes() == before
    assert store.validate(store.current_path).revision == 1


def test_list_and_load_need_no_write_access_to_recovery_root(tmp_path):
    store = _store(tmp_path)
    candidate = store.write(_state(), 1)
    assert candidate is not None
    os.chmod(store.session_directory, 0o500)
    os.chmod(store.root, 0o500)
    try:
        listed = store.list_candidates()
        assert [c.revision for c in listed] == [1]
        assert store.load(listed[0]).state.results[0].module_id == "detect.one"
    finally:
        os.chmod(store.root, 0o700)
        os.chmod(store.session_directory, 0o700)


def test_exact_save_fence_retires_only_covered_revisions(tmp_path):
    store = _store(tmp_path)
    store.write(_state("one"), 1)
    store.write(_state("two"), 2)
    assert store.retire_through(1) == 1
    assert store.current_path.exists()
    assert not store.previous_path.exists()
    assert [c.revision for c in store.list_candidates()] == [2]
    # Delayed work at/below the fence is a no-op; later edits remain recoverable.
    assert store.write(_state("late-one"), 1) is None
    newer = store.write(_state("three"), 3)
    assert newer is not None and newer.revision == 3


def test_explicit_save_of_current_revision_retires_both_generations(tmp_path):
    store = _store(tmp_path)
    store.write(_state("one"), 1)
    store.write(_state("two"), 2)
    assert store.retire_through(2) == 2
    assert store.list_candidates() == []
    assert (store.session_directory / RETIREMENT_NAME).exists()
    assert store.write(_state("delayed"), 2) is None


def test_discard_retires_the_whole_lineage_not_just_the_surface(tmp_path):
    store = _store(tmp_path)
    store.write(_state("one"), 1)
    candidate = store.write(_state("two"), 2)
    assert candidate is not None
    assert store.discard(candidate) == 2
    assert not store.current_path.exists()
    assert not store.previous_path.exists()
    assert store.list_candidates() == []
    assert store.write(_state("resurrect"), 999) is None


def test_a_delayed_writer_cannot_resurrect_an_explicitly_saved_revision(tmp_path):
    store = _store(tmp_path)
    store._mutex.acquire()
    entered = threading.Event()
    finished = threading.Event()
    result: list[object] = []

    def delayed_write():
        entered.set()
        result.append(store.write(_state("delayed"), 4))
        finished.set()

    thread = threading.Thread(target=delayed_write)
    thread.start()
    assert entered.wait(2)
    # RLock lets this owner install the fence while the worker is blocked at write().
    store.retire_through(4)
    store._mutex.release()
    assert finished.wait(10)
    thread.join()
    assert result == [None]
    assert store.list_candidates() == []


def test_retirement_waiting_on_an_inflight_write_wins_after_it_commits(tmp_path, monkeypatch):
    store = _store(tmp_path)
    serializing = threading.Event()
    release = threading.Event()
    real = store._serialize_payload

    def paused(state):
        serializing.set()
        assert release.wait(10)
        return real(state)

    monkeypatch.setattr(store, "_serialize_payload", paused)
    writer = threading.Thread(target=lambda: store.write(_state(), 5))
    writer.start()
    assert serializing.wait(2)
    retired: list[int] = []
    retirement = threading.Thread(target=lambda: retired.append(store.retire_through(5)))
    retirement.start()
    release.set()
    writer.join(15)
    retirement.join(15)
    assert retired == [1]
    assert store.list_candidates() == []


def test_save_after_restoring_retires_both_current_and_recovered_lineages(tmp_path):
    old = _store(tmp_path, SID1)
    recovered = old.write(_state("crashed"), 8)
    assert recovered is not None
    live = _store(tmp_path, SID2)
    live.write(_state("continued"), 3)
    assert live.retire_after_save(3, recovered_candidate=recovered) == 2
    assert live.list_candidates() == []
    assert old.write(_state("old-delayed"), 8) is None
    assert live.write(_state("live-delayed"), 3) is None


def test_an_out_of_order_worker_cannot_replace_a_newer_snapshot(tmp_path):
    store = _store(tmp_path)
    store.write(_state("new"), 5)
    before = store.current_path.read_bytes()
    with pytest.raises(StaleRecoveryWrite):
        store.write(_state("old"), 4)
    assert store.current_path.read_bytes() == before


def test_loading_a_path_reused_for_a_new_generation_detects_the_change(tmp_path):
    store = _store(tmp_path)
    selected = store.write(_state("one"), 1)
    assert selected is not None
    store.write(_state("two"), 2)
    with pytest.raises(RecoveryChangedError):
        store.load(selected)


def test_cleanup_bounds_old_sessions_but_never_the_live_one(tmp_path):
    root = tmp_path / "recoveries"
    stores = [RecoveryStore(root, session_id=sid, max_sessions=2,
                            max_age_seconds=None)
              for sid in (SID1, SID2, SID3)]
    for n, store in enumerate(stores, 1):
        store.write(_state(str(n)), n)
        os.utime(store.session_directory, (float(n), float(n)))
    # The third store's write cleanup already applies the bound. Running it again is
    # idempotent and the live SID3 directory is protected.
    stores[2].cleanup(now=100)
    remaining = {p.name for p in root.iterdir() if p.is_dir()}
    assert SID3 in remaining
    assert len(remaining) <= 2
    assert SID1 not in remaining


def test_cleanup_expires_old_lineages_by_age(tmp_path):
    root = tmp_path / "recoveries"
    old = RecoveryStore(root, SID1, max_sessions=20, max_age_seconds=None)
    old.write(_state(), 1)
    os.utime(old.session_directory, (10, 10))
    live = RecoveryStore(root, SID2, max_sessions=20, max_age_seconds=50)
    assert live.cleanup(now=100) == 1
    assert not old.session_directory.exists()


def test_clone_project_state_detaches_every_mutable_container_and_frame_cell():
    original = _state("clone")
    original.results[0].events["object"] = pd.Series(
        [{"labels": ["a"]}, {"labels": ["b"]}], dtype=object)
    original.measurements.attrs["nested"] = {"items": [1]}
    cloned = clone_project_state(original)

    assert isinstance(cloned, ProjectState)
    assert isinstance(cloned.results[0], ResultRec)
    assert isinstance(cloned.fits[0], FitRec)
    assert cloned.results[0].events is not original.results[0].events
    assert cloned.measurements is not original.measurements

    original.results[0].provenance["nested"]["thresholds"].append(3.0)
    original.results[0].events.at[0, "object"]["labels"].append("changed")
    original.measurements.attrs["nested"]["items"].append(2)
    original.label_vocabulary[0]["value"] = "changed"
    original.colours[0]["ink"] = "#ffffff"

    assert cloned.results[0].provenance["nested"]["thresholds"] == [1.0, 2.0]
    assert cloned.results[0].events.at[0, "object"] == {"labels": ["a"]}
    assert cloned.measurements.attrs["nested"] == {"items": [1]}
    assert cloned.label_vocabulary[0]["value"] == "clone"
    assert cloned.colours[0]["ink"] == "#123456"


def test_clone_rejects_non_project_values():
    with pytest.raises(TypeError):
        clone_project_state({})  # type: ignore[arg-type]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    repo = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo if not existing else os.pathsep.join((repo, existing))
    return env


def test_completed_snapshot_survives_a_real_os_exit(tmp_path):
    """No Qt teardown, ``finally`` block, or interpreter cleanup is helping here."""

    root = tmp_path / "recoveries"
    code = f"""
import os
from nanoscope.core.project import DocumentRef, ProjectState
from nanoscope.core.recovery import RecoveryStore
store = RecoveryStore({str(root)!r}, session_id={SID1!r})
store.write(ProjectState(documents=[DocumentRef('/data/before-exit.abf')]), 11,
            timestamp=123.0)
os._exit(23)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=_subprocess_env(),
        cwd=Path(__file__).resolve().parents[2], timeout=20,
    )
    assert result.returncode == 23
    reader = RecoveryStore(root, session_id=SID2)
    candidates = reader.list_candidates()
    assert [(c.session_id, c.revision) for c in candidates] == [(SID1, 11)]
    assert reader.load(candidates[0]).state.documents[0].path == "/data/before-exit.abf"


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="requires POSIX SIGKILL")
def test_sigkill_in_the_rename_window_falls_back_to_previous(tmp_path):
    """Kill after current→previous but before pending→current, using real OS state."""

    root = tmp_path / "recoveries"
    code = f"""
import os
import time
from pathlib import Path
import nanoscope.core.recovery as recovery
from nanoscope.core.project import DocumentRef, ProjectState
store = recovery.RecoveryStore({str(root)!r}, session_id={SID1!r})
store.write(ProjectState(documents=[DocumentRef('/data/good.abf')]), 1,
            timestamp=1.0)
real_replace = os.replace
def pause_at_final_replace(source, destination):
    if Path(destination) == store.current_path:
        print('ROTATED', flush=True)
        time.sleep(60)
    return real_replace(source, destination)
recovery.os.replace = pause_at_final_replace
store.write(ProjectState(documents=[DocumentRef('/data/new.abf')]), 2,
            timestamp=2.0)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code], env=_subprocess_env(),
        cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=15), "child never reached the rename window"
        assert process.stdout.readline().strip() == "ROTATED"
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == -signal.SIGKILL
    reader = RecoveryStore(root, session_id=SID2)
    candidates = reader.list_candidates()
    assert len(candidates) == 1
    assert candidates[0].session_id == SID1
    assert candidates[0].generation == "previous"
    assert candidates[0].revision == 1
    assert reader.load(candidates[0]).state.documents[0].path == "/data/good.abf"
