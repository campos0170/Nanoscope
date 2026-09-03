"""Bounded, coalescing crash-recovery writes for the Qt application.

The controller deliberately separates the two halves of a recovery save:

* ``snapshot_factory(revision)`` runs on the GUI thread, after the user has
  paused for :data:`DEFAULT_IDLE_MS` or no later than
  :data:`DEFAULT_MAX_MS` after the first unsnapshotted edit.  It is therefore
  allowed to checkpoint widgets and gather the project value object.
* ``persist(snapshot, revision)`` runs on one worker thread.  It may do file
  I/O, compression and hashing, but must not touch Qt or the live session.

Only the newest due snapshot is retained behind an in-flight write.  A newer
edit invalidates an older queued snapshot and starts a fresh idle/deadline
window; continuous edits cannot postpone capture past the hard deadline.

Session changes need one extra rule.  A write already inside ``persist`` cannot
be killed safely and may finish after the outgoing recovery has been retired.
``reset_session()`` consequently invalidates its generation but keeps it
connected.  Its terminal value is reported through :attr:`staleSaved`, so the
window can retire that late candidate again.  Teardown uses ``abort()`` and the
shared worker-lifetime seam, which never terminates a QThread and never waits
without a bound.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from PySide6 import QtCore

from .worker_thread import (
    DEFAULT_JOIN_MS,
    JOB_NONE,
    retire_worker_thread,
    start_worker_thread,
)

__all__ = [
    "DEFAULT_IDLE_MS",
    "DEFAULT_MAX_MS",
    "RecoveryController",
]


# Fifteen seconds makes an ordinary annotation burst one recovery transaction,
# while the sixty-second ceiling bounds loss during uninterrupted curation.
DEFAULT_IDLE_MS = 15_000
DEFAULT_MAX_MS = 60_000


@dataclass(frozen=True)
class _WriteCandidate:
    """One immutable value already detached from the live GUI state."""

    generation: int
    revision: int
    snapshot: object


class _RecoveryWriteWorker(QtCore.QObject):
    """Run exactly one injected persistence call outside the GUI thread."""

    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(
        self,
        persist: Callable[[object, int], object],
        candidate: _WriteCandidate,
    ) -> None:
        super().__init__()
        self._persist = persist
        self.candidate = candidate
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Request result suppression without interrupting an atomic file write.

        This is intentionally a plain method.  A queued Qt slot would run only
        after ``run()`` returned because the worker thread's event loop is busy
        inside it.  The persistence callable itself is allowed to finish: killing
        a thread halfway through a replace is precisely what recovery must avoid.
        """
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        if self._cancel.is_set():
            self.cancelled.emit()
            return
        try:
            result = self._persist(
                self.candidate.snapshot,
                self.candidate.revision,
            )
        except Exception as exc:  # noqa: BLE001 - failures are user-visible, never fatal
            if self._cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancel.is_set():
            self.cancelled.emit()
        else:
            self.finished.emit(result)


class RecoveryController(QtCore.QObject):
    """Schedule crash-recovery snapshots with one coalescing background writer.

    ``revision`` values are monotonic within one session.  ``reset_session()``
    starts a new generation, in which the counter may start over.  The factory
    may return:

    * an immutable snapshot object, optionally exposing an integer ``revision``
      attribute newer than the requested revision;
    * ``(actual_revision, snapshot)``;
    * ``None`` when the live session contains nothing recoverable.

    The actual captured revision is passed to ``persist`` and emitted by
    :attr:`saved` / :attr:`failed`.
    """

    saveStarted = QtCore.Signal(int)       # captured revision
    saved = QtCore.Signal(int, object)     # captured revision, persist return value
    failed = QtCore.Signal(int, str)       # captured/requested revision, readable error
    staleSaved = QtCore.Signal(int, int, object)  # old generation, revision, persist result
    busyChanged = QtCore.Signal(bool)

    def __init__(
        self,
        snapshot_factory: Callable[[int], object | None],
        persist: Callable[[object, int], object],
        parent: QtCore.QObject | None = None,
        *,
        idle_ms: int = DEFAULT_IDLE_MS,
        max_ms: int = DEFAULT_MAX_MS,
    ) -> None:
        super().__init__(parent)
        if not callable(snapshot_factory):
            raise TypeError("snapshot_factory must be callable")
        if not callable(persist):
            raise TypeError("persist must be callable")
        if int(idle_ms) <= 0 or int(max_ms) <= 0:
            raise ValueError("recovery timer intervals must be positive")

        self._snapshot_factory = snapshot_factory
        self._persist = persist
        self._idle_ms = int(idle_ms)
        self._max_ms = int(max_ms)

        self._idle_timer = QtCore.QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._idle_timer.timeout.connect(self._capture_due)
        self._max_timer = QtCore.QTimer(self)
        self._max_timer.setSingleShot(True)
        self._max_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._max_timer.timeout.connect(self._capture_due)

        self._generation = 0
        self._retired = False
        self._retired_through = -1
        self._settled_revision = -1
        self._scheduled_revision: int | None = None
        self._pending: _WriteCandidate | None = None
        self._worker: _RecoveryWriteWorker | None = None
        self._thread: QtCore.QThread | None = None
        self._busy_signal_state = False

    # -- public state -------------------------------------------------------------------------
    @property
    def generation(self) -> int:
        return self._generation

    @property
    def scheduled_revision(self) -> int | None:
        return self._scheduled_revision

    @property
    def pending_revision(self) -> int | None:
        return None if self._pending is None else self._pending.revision

    @property
    def inflight_revision(self) -> int | None:
        worker = self._worker
        return None if worker is None else worker.candidate.revision

    @property
    def idle_interval_ms(self) -> int:
        return self._idle_ms

    @property
    def max_interval_ms(self) -> int:
        return self._max_ms

    def is_busy(self) -> bool:
        return self._thread is not None

    def has_pending(self) -> bool:
        return self._scheduled_revision is not None or self._pending is not None

    # -- scheduling ---------------------------------------------------------------------------
    @QtCore.Slot(int)
    def note_dirty(self, revision: int) -> bool:
        """Record a mutation and (re)arm the idle timer.

        This method does *not* gather project state.  Duplicate/older revision
        announcements are ignored, so an incidental duplicate signal cannot keep
        moving the idle timer.  The hard timer is started for the first uncovered
        revision and is never restarted by later edits in that burst.
        """
        if self._retired:
            return False
        revision = self._coerce_revision(revision)
        if revision <= self._covered_revision():
            return False
        if self._scheduled_revision is not None and revision <= self._scheduled_revision:
            return False

        # A captured pending value ceased to be the latest value as soon as a newer
        # live edit landed.  Drop it rather than performing a knowingly stale write.
        if self._pending is not None and revision > self._pending.revision:
            self._pending = None

        first_in_burst = self._scheduled_revision is None
        self._scheduled_revision = revision
        self._idle_timer.start(self._idle_ms)
        if first_in_burst or not self._max_timer.isActive():
            self._max_timer.start(self._max_ms)
        return True

    @QtCore.Slot()
    def flush(self) -> bool:
        """Capture/write the latest outstanding revision immediately.

        If a writer is active, capture still happens synchronously now and the
        immutable value becomes its single coalesced successor.  Returns ``True``
        when work was captured, queued or started; ``False`` when there was no
        recoverable work (or the controller has been retired).
        """
        if self._retired:
            return False
        if self._scheduled_revision is not None:
            return self._capture_due()
        if self._pending is None:
            return False
        if self.is_busy():
            return True
        candidate, self._pending = self._pending, None
        self._launch(candidate)
        return True

    def retire_through(self, revision: int) -> None:
        """Fence recovery work already covered by an explicit project save.

        Uncaptured and queued values at or below the fence are discarded.  An
        in-flight write is allowed to complete safely and still emits ``saved``;
        the window can then apply the same fence to the store once more, closing
        the small save/write race without ever killing a filesystem operation.
        """
        revision = self._coerce_revision(revision)
        self._retired_through = max(self._retired_through, revision)
        self._settled_revision = max(self._settled_revision, revision)
        if (self._scheduled_revision is not None
                and self._scheduled_revision <= self._retired_through):
            self._scheduled_revision = None
            self._stop_timers()
        if self._pending is not None and self._pending.revision <= self._retired_through:
            self._pending = None

    # Alias reads naturally at the explicit-save call site and keeps the semantic
    # operation discoverable without creating two implementations.
    mark_explicitly_saved = retire_through

    def reset_session(
        self,
        *,
        persist: Callable[[object, int], object] | None = None,
    ) -> None:
        """Invalidate queued state and begin a fresh revision generation.

        A live writer remains connected until it finishes.  If it did write, its
        result is emitted as ``staleSaved`` rather than silently disappearing, so
        the owner can retire the late on-disk candidate.  New-generation work can
        be scheduled immediately and will wait as the sole coalesced successor.

        ``persist`` may replace the persistence destination for the new generation.
        The active worker already owns the old callable, so rotating a recovery-store
        session here cannot redirect an outgoing write into the incoming lineage.
        """
        if persist is not None:
            if not callable(persist):
                raise TypeError("persist must be callable")
            self._persist = persist
        self._generation += 1
        self._retired = False
        self._retired_through = -1
        self._settled_revision = -1
        self._scheduled_revision = None
        self._pending = None
        self._stop_timers()

    # A compact name for callers that do not need to spell the reason for the reset.
    reset = reset_session

    def abort(self, wait_ms: int = DEFAULT_JOIN_MS) -> None:
        """Retire the controller for teardown, emitting no later outcomes."""
        self.abort_stopped(wait_ms)

    def abort_stopped(self, wait_ms: int = DEFAULT_JOIN_MS) -> bool:
        """Retire for teardown and report whether the writer stopped in time."""
        self._generation += 1
        self._retired = True
        self._scheduled_revision = None
        self._pending = None
        self._stop_timers()
        worker, thread = self._worker, self._thread
        was_busy = thread is not None
        self._worker, self._thread = None, None
        if worker is not None:
            worker.cancel()
        stopped = retire_worker_thread(worker, thread, wait_ms=max(0, int(wait_ms)))
        # Also clear the logical busy transition when abort() is called re-entrantly
        # from a terminal signal: at that instant the old thread reference is already
        # retired but a coalesced successor has not yet been launched.
        if was_busy or self._busy_signal_state:
            self._set_busy(False)
        return stopped

    # -- capture (GUI thread) -----------------------------------------------------------------
    @QtCore.Slot()
    def _capture_due(self) -> bool:
        requested = self._scheduled_revision
        if self._retired or requested is None:
            self._stop_timers()
            return False

        generation = self._generation
        self._scheduled_revision = None
        self._stop_timers()
        try:
            raw = self._snapshot_factory(requested)
        except Exception as exc:  # noqa: BLE001 - gather failures are reportable too
            if generation == self._generation and not self._retired:
                # Retain it for an explicit flush or the next edit, but do not
                # automatically spin on a deterministic gather failure.
                current = self._scheduled_revision
                self._scheduled_revision = (
                    requested if current is None else max(requested, current))
                self.failed.emit(requested, f"{type(exc).__name__}: {exc}")
            return False

        if generation != self._generation or self._retired:
            return False
        if raw is None:
            self._settled_revision = max(self._settled_revision, requested)
            self._discard_scheduled_through(requested)
            return False

        try:
            actual, snapshot = self._resolve_snapshot(raw, requested)
        except Exception as exc:  # noqa: BLE001 - malformed factories fail as data, not Qt
            self._scheduled_revision = requested
            self.failed.emit(requested, f"{type(exc).__name__}: {exc}")
            return False
        if actual < requested:
            self._scheduled_revision = requested
            self.failed.emit(
                requested,
                f"Recovery snapshot revision {actual} is older than requested "
                f"revision {requested}.",
            )
            return False

        self._discard_scheduled_through(actual)
        if actual <= self._retired_through:
            self._settled_revision = max(self._settled_revision, actual)
            return False
        candidate = _WriteCandidate(generation, actual, snapshot)
        if self.is_busy():
            if self._pending is None or candidate.revision >= self._pending.revision:
                self._pending = candidate
        else:
            self._launch(candidate)
        return True

    @staticmethod
    def _resolve_snapshot(raw: object, requested: int) -> tuple[int, object]:
        if (isinstance(raw, tuple) and len(raw) == 2
                and isinstance(raw[0], int) and not isinstance(raw[0], bool)):
            return int(raw[0]), raw[1]
        actual = getattr(raw, "revision", requested)
        return RecoveryController._coerce_revision(actual), raw

    def _discard_scheduled_through(self, revision: int) -> None:
        if (self._scheduled_revision is not None
                and self._scheduled_revision <= revision):
            self._scheduled_revision = None
            self._stop_timers()

    def _covered_revision(self) -> int:
        covered = max(self._settled_revision, self._retired_through)
        worker = self._worker
        if worker is not None and worker.candidate.generation == self._generation:
            covered = max(covered, worker.candidate.revision)
        if self._pending is not None and self._pending.generation == self._generation:
            covered = max(covered, self._pending.revision)
        return covered

    @staticmethod
    def _coerce_revision(revision: Any) -> int:
        if isinstance(revision, bool):
            raise TypeError("revision must be an integer")
        value = int(revision)
        if value < 0:
            raise ValueError("revision must be non-negative")
        return value

    def _stop_timers(self) -> None:
        self._idle_timer.stop()
        self._max_timer.stop()

    # -- one writer ---------------------------------------------------------------------------
    def _launch(self, candidate: _WriteCandidate) -> bool:
        if self._retired or candidate.generation != self._generation:
            return False
        if candidate.revision <= self._retired_through:
            self._settled_revision = max(self._settled_revision, candidate.revision)
            return False
        if self.is_busy():
            self._pending = candidate
            return True
        worker = _RecoveryWriteWorker(self._persist, candidate)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self._thread = start_worker_thread(
            worker,
            domain=JOB_NONE,
            label="recovery.write",
            owner=self,
        )
        self._set_busy(True)
        self.saveStarted.emit(candidate.revision)
        return True

    def _set_busy(self, busy: bool) -> None:
        busy = bool(busy)
        if busy == self._busy_signal_state:
            return
        self._busy_signal_state = busy
        self.busyChanged.emit(busy)

    def _is_active(self) -> bool:
        return self._worker is not None and self.sender() is self._worker

    def _retire_active(self, *, final: bool) -> _RecoveryWriteWorker | None:
        worker, thread = self._worker, self._thread
        self._worker, self._thread = None, None
        retire_worker_thread(worker, thread)
        if final:
            self._set_busy(False)
        return worker

    def _take_current_pending(self) -> _WriteCandidate | None:
        candidate, self._pending = self._pending, None
        if candidate is None:
            return None
        if candidate.generation != self._generation:
            return None
        if candidate.revision <= self._retired_through:
            self._settled_revision = max(self._settled_revision, candidate.revision)
            return None
        return candidate

    @QtCore.Slot(object)
    def _on_finished(self, result: object) -> None:
        if not self._is_active():
            return
        worker = self._worker
        assert worker is not None
        candidate = worker.candidate
        successor = self._take_current_pending()
        self._retire_active(final=successor is None)

        if candidate.generation == self._generation and not self._retired:
            self._settled_revision = max(self._settled_revision, candidate.revision)
            self.saved.emit(candidate.revision, result)
        else:
            self.staleSaved.emit(candidate.generation, candidate.revision, result)

        if (successor is not None and not self._retired
                and not self._launch(successor)):
            self._set_busy(False)

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        if not self._is_active():
            return
        worker = self._worker
        assert worker is not None
        failed_candidate = worker.candidate
        successor = self._take_current_pending()

        # A current, uncovered failure stays available for explicit retry, unless
        # newer work already superseded it.  Stale/covered failures need no warning:
        # their session or explicit project save is already authoritative.
        report = (failed_candidate.generation == self._generation
                  and not self._retired
                  and failed_candidate.revision > self._retired_through)
        if successor is None and report:
            self._pending = failed_candidate
        self._retire_active(final=successor is None)
        if report:
            self.failed.emit(failed_candidate.revision, message)
        if (successor is not None and not self._retired
                and not self._launch(successor)):
            self._set_busy(False)

    @QtCore.Slot()
    def _on_cancelled(self) -> None:
        if not self._is_active():
            return
        successor = self._take_current_pending()
        self._retire_active(final=successor is None)
        if (successor is not None and not self._retired
                and not self._launch(successor)):
            self._set_busy(False)
