"""Deterministic contract tests for the Stage 2 recovery scheduler."""

from __future__ import annotations

# Qt bootstrap must precede the PySide import (the project-wide test invariant).
import os

import nanoscope  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
from dataclasses import dataclass

import pytest
from PySide6 import QtCore

from nanoscope.gui.recovery_controller import (
    DEFAULT_IDLE_MS,
    DEFAULT_MAX_MS,
    RecoveryController,
)
from nanoscope.gui.worker_thread import detached_count, drain_detached


@dataclass(frozen=True)
class _Snapshot:
    revision: int
    value: str


@pytest.fixture
def controller_factory(qapp, qtbot):
    made: list[RecoveryController] = []

    def build(snapshot_factory, persist, **kwargs) -> RecoveryController:
        controller = RecoveryController(snapshot_factory, persist, **kwargs)
        made.append(controller)
        return controller

    yield build

    for controller in made:
        controller.abort(1000)
    qtbot.waitUntil(lambda: drain_detached(100) == 0, timeout=5000)


def test_defaults_and_argument_validation(controller_factory):
    controller = controller_factory(lambda revision: None, lambda snapshot, revision: None)
    assert controller.idle_interval_ms == DEFAULT_IDLE_MS == 15_000
    assert controller.max_interval_ms == DEFAULT_MAX_MS == 60_000
    assert controller._idle_timer.isSingleShot()
    assert controller._max_timer.isSingleShot()

    with pytest.raises(ValueError, match="positive"):
        RecoveryController(lambda revision: None, lambda snapshot, revision: None,
                           idle_ms=0)
    with pytest.raises(TypeError, match="snapshot_factory"):
        RecoveryController(None, lambda snapshot, revision: None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="revision"):
        controller.note_dirty(True)
    with pytest.raises(ValueError, match="non-negative"):
        controller.note_dirty(-1)


def test_note_dirty_only_arms_timers_and_idle_captures_on_gui_thread(
    controller_factory, qapp, qtbot,
):
    captures: list[tuple[int, object]] = []
    writes: list[tuple[object, int, object]] = []

    def snapshot_factory(revision):
        captures.append((revision, QtCore.QThread.currentThread()))
        return _Snapshot(revision, f"state-{revision}")

    def persist(snapshot, revision):
        writes.append((snapshot, revision, QtCore.QThread.currentThread()))
        return f"candidate-{revision}"

    controller = controller_factory(
        snapshot_factory, persist, idle_ms=25, max_ms=500)
    saved: list[tuple[int, object]] = []
    controller.saved.connect(lambda revision, result: saved.append((revision, result)))

    assert controller.note_dirty(1)
    assert captures == []
    assert writes == []
    assert controller.note_dirty(2)
    assert captures == []

    qtbot.waitUntil(lambda: saved == [(2, "candidate-2")], timeout=2000)
    assert [item[0] for item in captures] == [2]
    assert captures[0][1] is qapp.thread()
    assert writes[0][0] == _Snapshot(2, "state-2")
    assert writes[0][1] == 2
    assert writes[0][2] is not qapp.thread()
    assert not controller.is_busy()
    assert not controller.has_pending()


def test_idle_restarts_but_hard_deadline_does_not(controller_factory, qtbot):
    requested: list[int] = []
    controller = controller_factory(
        lambda revision: requested.append(revision) or _Snapshot(revision, "x"),
        lambda snapshot, revision: revision,
        idle_ms=250,
        max_ms=1000,
    )

    controller.note_dirty(1)
    max_before = controller._max_timer.remainingTime()
    qtbot.wait(35)
    idle_before_edit = controller._idle_timer.remainingTime()
    controller.note_dirty(2)
    max_after = controller._max_timer.remainingTime()
    idle_after = controller._idle_timer.remainingTime()

    assert 0 < max_after < max_before
    assert idle_after > idle_before_edit
    assert requested == []
    controller._capture_due()  # deterministic stand-in for the still-running max timer
    qtbot.waitUntil(lambda: not controller.is_busy(), timeout=2000)
    assert requested == [2]


def test_continuous_edits_are_captured_by_the_real_hard_timer(
    controller_factory, qtbot,
):
    captured: list[int] = []
    controller = controller_factory(
        lambda revision: captured.append(revision) or _Snapshot(revision, "continuous"),
        lambda snapshot, revision: revision,
        idle_ms=200,
        max_ms=80,
    )
    saved: list[int] = []
    controller.saved.connect(lambda revision, result: saved.append(revision))

    controller.note_dirty(1)
    for revision in (2, 3, 4):
        qtbot.wait(20)
        controller.note_dirty(revision)
    # The last idle deadline is about 260 ms after the first edit.  The hard
    # timer is due at 80 ms and therefore wins even while edits continue.
    qtbot.waitUntil(lambda: bool(saved), timeout=180)
    assert captured == [4]
    assert saved == [4]


def test_none_snapshot_settles_without_start_or_failure(controller_factory):
    writes: list[int] = []
    failures: list[tuple[int, str]] = []
    controller = controller_factory(
        lambda revision: None,
        lambda snapshot, revision: writes.append(revision),
    )
    controller.failed.connect(lambda revision, message: failures.append((revision, message)))

    controller.note_dirty(7)
    assert controller.flush() is False
    assert writes == []
    assert failures == []
    assert not controller.has_pending()
    assert not controller.is_busy()
    assert controller.note_dirty(7) is False


def test_capture_uses_newer_actual_revision_and_settles_reentrant_dirty(
    controller_factory, qtbot,
):
    writes: list[tuple[object, int]] = []
    controller: RecoveryController

    def capture(requested):
        # MainWindow's checkpoint can expose a silent edit while gathering.
        controller.note_dirty(requested + 1)
        return _Snapshot(requested + 1, "checkpointed")

    controller = controller_factory(
        capture,
        lambda snapshot, revision: writes.append((snapshot, revision)) or "written",
    )
    saved: list[tuple[int, object]] = []
    controller.saved.connect(lambda revision, result: saved.append((revision, result)))

    controller.note_dirty(10)
    assert controller.flush()
    qtbot.waitUntil(lambda: saved == [(11, "written")], timeout=2000)
    assert writes == [(_Snapshot(11, "checkpointed"), 11)]
    assert controller.scheduled_revision is None
    assert not controller.has_pending()


def test_tuple_snapshot_contract_passes_payload_and_actual_revision(
    controller_factory, qtbot,
):
    writes: list[tuple[object, int]] = []
    controller = controller_factory(
        lambda requested: (requested + 3, {"frozen": requested}),
        lambda snapshot, revision: writes.append((snapshot, revision)) or "candidate",
    )
    saved: list[int] = []
    controller.saved.connect(lambda revision, result: saved.append(revision))
    controller.note_dirty(2)
    controller.flush()
    qtbot.waitUntil(lambda: saved == [5], timeout=2000)
    assert writes == [({"frozen": 2}, 5)]


def test_only_latest_pending_snapshot_follows_one_inflight_writer(
    controller_factory, qtbot,
):
    first_entered = threading.Event()
    release_first = threading.Event()
    calls: list[int] = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def persist(snapshot, revision):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            calls.append(revision)
            if revision == 1:
                first_entered.set()
                assert release_first.wait(3), "test did not release first recovery write"
            return f"candidate-{revision}"
        finally:
            with lock:
                active -= 1

    controller = controller_factory(
        lambda revision: _Snapshot(revision, f"state-{revision}"), persist)
    saved: list[int] = []
    busy: list[bool] = []
    controller.saved.connect(lambda revision, result: saved.append(revision))
    controller.busyChanged.connect(busy.append)

    controller.note_dirty(1)
    controller.flush()
    assert first_entered.wait(2)
    controller.note_dirty(2)
    controller.flush()
    assert controller.pending_revision == 2
    controller.note_dirty(3)  # invalidates the captured revision-2 successor
    assert controller.pending_revision is None
    controller.flush()
    assert controller.pending_revision == 3

    release_first.set()
    qtbot.waitUntil(lambda: saved == [1, 3], timeout=3000)
    assert calls == [1, 3]
    assert max_active == 1
    assert busy == [True, False]
    assert not controller.is_busy()


def test_persist_failure_is_reported_and_candidate_can_be_retried(
    controller_factory, qtbot,
):
    attempts = 0

    def persist(snapshot, revision):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("disk full")
        return "recovered-after-retry"

    controller = controller_factory(
        lambda revision: _Snapshot(revision, "important"), persist)
    failures: list[tuple[int, str]] = []
    saved: list[tuple[int, object]] = []
    controller.failed.connect(lambda revision, message: failures.append((revision, message)))
    controller.saved.connect(lambda revision, result: saved.append((revision, result)))

    controller.note_dirty(4)
    controller.flush()
    qtbot.waitUntil(lambda: len(failures) == 1, timeout=2000)
    assert failures[0][0] == 4
    assert "OSError: disk full" in failures[0][1]
    assert controller.pending_revision == 4
    assert not controller.is_busy()

    assert controller.flush()
    qtbot.waitUntil(lambda: saved == [(4, "recovered-after-retry")], timeout=2000)
    assert attempts == 2
    assert controller.pending_revision is None


def test_snapshot_failure_is_reported_without_retry_spin(controller_factory):
    attempts = 0

    def capture(revision):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("checkpoint failed")

    controller = controller_factory(capture, lambda snapshot, revision: None)
    failures: list[tuple[int, str]] = []
    controller.failed.connect(lambda revision, message: failures.append((revision, message)))
    controller.note_dirty(8)
    assert controller.flush() is False
    assert attempts == 1
    assert failures == [(8, "RuntimeError: checkpoint failed")]
    assert controller.scheduled_revision == 8
    assert not controller._idle_timer.isActive()
    assert not controller._max_timer.isActive()


def test_older_than_requested_snapshot_is_refused(controller_factory):
    writes: list[int] = []
    controller = controller_factory(
        lambda revision: _Snapshot(revision - 1, "stale"),
        lambda snapshot, revision: writes.append(revision),
    )
    failures: list[tuple[int, str]] = []
    controller.failed.connect(lambda revision, message: failures.append((revision, message)))
    controller.note_dirty(5)
    assert controller.flush() is False
    assert writes == []
    assert failures and failures[0][0] == 5
    assert "older than requested" in failures[0][1]


def test_explicit_save_fence_drops_scheduled_and_queued_but_reports_inflight_finish(
    controller_factory, qtbot,
):
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def persist(snapshot, revision):
        calls.append(revision)
        if revision == 1:
            entered.set()
            assert release.wait(3)
        return f"candidate-{revision}"

    controller = controller_factory(
        lambda revision: _Snapshot(revision, str(revision)), persist)
    saved: list[int] = []
    controller.saved.connect(lambda revision, result: saved.append(revision))
    controller.note_dirty(1)
    controller.flush()
    assert entered.wait(2)
    controller.note_dirty(2)
    controller.flush()
    assert controller.pending_revision == 2

    controller.retire_through(2)
    assert controller.pending_revision is None
    assert controller.note_dirty(2) is False
    release.set()
    qtbot.waitUntil(lambda: saved == [1], timeout=3000)
    assert calls == [1]

    assert controller.note_dirty(3)
    controller.flush()
    qtbot.waitUntil(lambda: saved == [1, 3], timeout=2000)
    assert calls == [1, 3]


def test_reset_session_exposes_late_old_write_and_serializes_new_generation(
    controller_factory, qtbot,
):
    old_entered = threading.Event()
    release_old = threading.Event()
    session = ["A"]
    calls: list[tuple[str, str]] = []

    def snapshot_factory(revision):
        return _Snapshot(revision, session[0])

    def old_persist(snapshot, revision):
        calls.append(("old-store", snapshot.value))
        old_entered.set()
        assert release_old.wait(3)
        return f"{snapshot.value}-candidate"

    def new_persist(snapshot, revision):
        calls.append(("new-store", snapshot.value))
        return f"{snapshot.value}-candidate"

    controller = controller_factory(snapshot_factory, old_persist)
    normal: list[tuple[int, object]] = []
    stale: list[tuple[int, int, object]] = []
    controller.saved.connect(lambda revision, result: normal.append((revision, result)))
    controller.staleSaved.connect(
        lambda generation, revision, result:
        stale.append((generation, revision, result)))

    assert controller.generation == 0
    controller.note_dirty(1)
    controller.flush()
    assert old_entered.wait(2)

    controller.reset_session(persist=new_persist)
    assert controller.generation == 1
    assert controller.is_busy()  # the atomic old write is not killed or hidden
    session[0] = "B"
    assert controller.note_dirty(1)  # revisions may restart in a new generation
    controller.flush()
    assert controller.pending_revision == 1

    release_old.set()
    qtbot.waitUntil(lambda: stale == [(0, 1, "A-candidate")], timeout=3000)
    qtbot.waitUntil(lambda: normal == [(1, "B-candidate")], timeout=3000)
    assert calls == [("old-store", "A"), ("new-store", "B")]
    assert not controller.is_busy()


def test_abort_cancels_timers_and_detaches_a_slow_writer_without_qthread_leak(
    controller_factory, qtbot,
):
    captures: list[int] = []
    controller = controller_factory(
        lambda revision: captures.append(revision) or _Snapshot(revision, "timer"),
        lambda snapshot, revision: revision,
        idle_ms=40,
        max_ms=100,
    )
    controller.note_dirty(1)
    assert controller._idle_timer.isActive()
    assert controller._max_timer.isActive()
    assert controller.abort_stopped(0)
    qtbot.wait(130)
    assert captures == []
    assert controller.note_dirty(2) is False

    entered = threading.Event()
    release = threading.Event()

    def slow_persist(snapshot, revision):
        entered.set()
        assert release.wait(3)
        return revision

    slow = controller_factory(lambda revision: _Snapshot(revision, "slow"), slow_persist)
    saved: list[int] = []
    slow.saved.connect(lambda revision, result: saved.append(revision))
    slow.note_dirty(1)
    slow.flush()
    assert entered.wait(2)
    assert slow.abort_stopped(0) is False
    assert not slow.is_busy()
    assert detached_count() >= 1
    release.set()
    qtbot.waitUntil(lambda: drain_detached(50) == 0, timeout=3000)
    assert saved == []
