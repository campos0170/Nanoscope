"""Run an :class:`AnalysisModule` off the Qt main thread (PORT_PLAN §6, gap #2).

This mirrors :mod:`nanoscope.gui.workers` exactly -- ONE worker, moved to a
``QThread``, reporting progress and polling a ``threading.Event`` for cancel --
because an analysis run is the same shape as a load: one long-lived task needing
progress, cancel and deterministic teardown, which is what ``QThread`` +
``moveToThread`` is for (NOT ``QRunnable``/``QThreadPool``).

The seam the worker sits on is deliberately narrow. A module receives a
:class:`~nanoscope.analysis.context.SimpleAnalysisContext` and nothing else: it
never sees the :class:`~nanoscope.core.session.Session`, never imports Qt, never
draws. So the controller -- which DOES hold the Session -- resolves the selected
``(doc_id, channel)`` into a :class:`~nanoscope.core.document.CalibratedChannel`
plus :class:`~nanoscope.core.types.RecordingMeta`, and injects the two things the
module cannot compute for itself: ``_calibration_fingerprint`` (the staleness
stamp, ``session.calibration(doc_id, ch).fingerprint()``) and ``pore_id`` (PORT_PLAN
§4, R2). The module strips the ``_``-prefixed key before it reaches the detector.

On ``finished(AnalysisResult)`` the MAIN THREAD installs the result via
``Session.add_result`` -- the controller emits, the window (or a test) stores.
Detection is the only channel-driven module; the null/sweep/recovery jobs return
plain dicts/DataFrames, so :meth:`AnalysisController.run_callable` gives them a
generic ``finished(object)`` path that reuses the same thread lifetime.
"""

from __future__ import annotations

import math
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from PySide6 import QtCore
import numpy as np

from ..analysis.baseline import BATCH_SCOPE_KEY
from ..analysis.cancel import OperationCancelled, cancel_scope
from ..analysis.context import SimpleAnalysisContext
from ..core.analysis import ANALYSIS_REGISTRY, AnalysisModule, AnalysisResult
from ..core.document import CalibratedChannel
from ..core.calibration import ChannelCalibration
from ..core.types import RecordingMeta, nonfinite_reason, scan_nonfinite
from .worker_thread import (
    DEFAULT_JOIN_MS,
    JOB_ANALYSIS,
    blocking_job,
    refusal_text,
    retire_worker_thread,
    start_worker_thread,
)

__all__ = [
    "AnalysisCancelled",
    "AnalysisWorker",
    "CallableWorker",
    "AnalysisController",
    "BatchJob",
    "BatchOutcome",
    "BatchDetectWorker",
    "BatchDetectController",
    "FrozenChannelSource",
]

#: A module (or any kernel underneath it) observed the cancel token and stopped
#: cooperatively. The worker turns this into ``cancelled()``, never ``failed()`` --
#: the mirror of ``LoadCancelled`` in :mod:`nanoscope.io.errors`.
#:
#: This is an ALIAS of :class:`nanoscope.analysis.cancel.OperationCancelled`, not a
#: second exception type, because there must be exactly one "the user asked to stop"
#: signal in the codebase: the pure analysis layer must be able to raise it without
#: importing anything from :mod:`nanoscope.gui`, and the worker must catch what the
#: pure layer raises. Two types would mean a kernel that stops cooperatively is
#: reported to the user as a crash.
AnalysisCancelled = OperationCancelled


class AnalysisWorker(QtCore.QObject):
    """``module.run(ctx)`` off the main thread, reporting progress and polling cancel.

    Built with the already-resolved calibrated ``channel`` / ``meta`` / ``params``
    (the controller did the Session lookups); ``run()`` wraps them in a
    :class:`SimpleAnalysisContext` wired to this worker's progress signal and cancel
    Event, then calls the module. It never lets an exception escape: ``cancelled()``
    on a cooperative stop, ``failed(str)`` on anything else.
    """

    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)   # AnalysisResult — a frozen value object, crosses threads fine
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(
        self,
        module: AnalysisModule,
        channel: CalibratedChannel,
        meta: RecordingMeta,
        params: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self._module = module
        self._channel = channel
        self._meta = meta
        self._params = dict(params)
        self._cancel = threading.Event()   # NOT a slot. See cancel().

    def cancel(self) -> None:
        """*** A PLAIN METHOD, CALLED DIRECTLY FROM THE GUI THREAD. ***
        Deliberately NOT a @Slot, and NEVER invoked via signal/slot or QMetaObject.invokeMethod.

        A @Slot cancel() on a worker moved to a QThread is delivered through THAT
        THREAD'S event loop, which is blocked inside run() — so the call queues
        behind run() and arrives only after the analysis has already finished. A
        threading.Event set from the GUI thread is seen by the worker (via
        ctx.cancelled()) with no Qt in the path. Same reasoning as LoadWorker.cancel().
        """
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            ctx = SimpleAnalysisContext(
                channel=self._channel,
                meta=self._meta,
                params=self._params,
                progress_cb=self.progress.emit,
                cancel_flag=self._cancel.is_set,
            )
            # The token is also installed AMBIENTLY for this thread, so every kernel
            # underneath the module -- detect_events, characterize, merge, the truth
            # studies -- polls it without the module having to forward ``ctx.cancelled``
            # down four layers of call site. See analysis.cancel.
            with cancel_scope(self._cancel.is_set):
                result = self._module.run(ctx)
            # A module that ignored the flag still gets its result dropped: the user
            # asked to stop, so honour it rather than emit a result they cancelled.
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            self.finished.emit(result)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:  # never let a detector/numpy traceback escape the worker
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CallableWorker(QtCore.QObject):
    """Generic job path for the non-module analyses (null/sweep/recovery).

    They consume an event table, not a raw channel, and return plain
    dicts/DataFrames rather than an :class:`AnalysisResult`, so they do not fit the
    ``AnalysisModule`` contract (PORT_PLAN D3). This runs an arbitrary zero-arg
    callable on the same thread lifetime and emits ``finished(object)`` with
    whatever it returned.

    THE CALLABLE IS ABANDONED IN PROGRESS, not merely disowned afterwards. This
    used to call ``self._fn()`` to completion and only then look at the cancel
    Event, which made Cancel a lie for exactly the jobs that need it: the surrogate
    studies dispatched here are the slowest work in the app (``sweep_threshold`` is
    18.9 s on the owner's SMALLEST apo recording). Running the callable inside a
    :func:`~nanoscope.analysis.cancel.cancel_scope` lets the kernels underneath it
    raise out of their loops the moment the Event is set.
    """

    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, fn: Callable[..., Any], *, wants_progress: bool = False) -> None:
        super().__init__()
        self._fn = fn
        self._wants_progress = bool(wants_progress)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Plain method, set from the GUI thread — see AnalysisWorker.cancel()."""
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            with cancel_scope(self._cancel.is_set):
                # ONE ARGUMENT OR NONE, decided at construction, never sniffed from
                # the callable. Inspecting the signature would make "does this job
                # report progress?" a property of how `fn` happens to be spelled — a
                # `functools.partial`, a bound method or a lambda with a default
                # argument each answer differently — so the caller says which
                # contract it is handing over and the worker holds it to that.
                result = self._fn(self.progress.emit) if self._wants_progress else self._fn()
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            self.finished.emit(result)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class AnalysisController(QtCore.QObject):
    """Owns the analysis thread lifetime; mirrors :class:`LoadController`.

    ONE worker at a time: :meth:`start` aborts any in-flight run first, so several
    runs through one controller are SEQUENTIAL. Holds the Session (which the module
    is forbidden to touch) to resolve the selected ``(doc_id, channel)`` and to
    inject the calibration fingerprint + pore_id. Emits
    ``started``/``progressed``/``finished(AnalysisResult)``/``failed``/``cancelled``;
    the caller stores the result on the main thread via ``Session.add_result``. A run
    aborted that way reports none of the terminal signals, so ``replaced(label)`` says
    that it happened -- once, from the one place a job is replaced.

    ONE ANALYSIS IN THE WHOLE APP, not merely one per controller. Both entry points
    consult the run gate (:mod:`nanoscope.gui.worker_thread`) before they spin a
    thread, so a batch pass and a single Detect can no longer both be in flight
    writing the same ``ResultStore`` key. A refused start emits ``refused(str)`` and
    returns ``False``; it starts nothing and changes no state. The return value is
    what lets a caller keep its own bookkeeping honest, but a caller that ignores it
    still cannot start the job -- a forgotten check costs a word, never a row.
    """

    started = QtCore.Signal(str)            # module_id (or "callable")
    progressed = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)        # AnalysisResult | dict | DataFrame
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()
    refused = QtCore.Signal(str)            # nothing started; the sentence says why
    #: A live job was dropped to make room for the one named here (its label). The dropped job
    #: reports nothing itself -- ``abort()`` emits no terminal signal -- so this is the only way
    #: its owner learns that whatever it recorded about it is void. See :meth:`_supersede`.
    replaced = QtCore.Signal(str)

    def __init__(self, session: Any, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._thread: QtCore.QThread | None = None
        self._worker: QtCore.QObject | None = None
        #: Workers :meth:`_stop` has retired whose queued signals may not have been delivered
        #: yet, oldest first. See :meth:`_stop` for why they are kept.
        self._retired: list[QtCore.QObject] = []

    # -- public API ------------------------------------------------------------------------------
    def start(
        self,
        module: AnalysisModule | str,
        doc_id: str,
        channel: int,
        params: Mapping[str, Any] | None = None,
        *,
        pore_id: str = "",
    ) -> bool:
        """Run ``module`` on ``(doc_id, channel)``; True when a job actually started.

        ``module`` may be an id string (resolved through ANALYSIS_REGISTRY) or an
        AnalysisModule instance.

        The Session lookups happen HERE, on the main thread, so the worker only ever
        holds plain value objects: the calibrated channel, its meta, and a params
        dict carrying the injected ``_calibration_fingerprint`` and ``pore_id``.

        Refused (``False``, ``refused`` emitted, nothing aborted and nothing started)
        when another owner already holds the analysis domain -- a batch pass, in
        practice. The refusal is checked BEFORE ``abort()`` on purpose: a start that
        is not going to happen must not stop the run that is already happening. A run that IS
        stopped to make room is announced on :attr:`replaced`.
        """
        label = module if isinstance(module, str) else module.id
        if not self._admit(label):
            return False
        self._supersede(label)

        if isinstance(module, str):
            module = ANALYSIS_REGISTRY[module]

        ch = self._session.calibrated(doc_id, channel)
        meta = self._session.document(doc_id).meta
        fingerprint = self._session.calibration(doc_id, channel).fingerprint()

        merged = dict(params or {})
        merged["_calibration_fingerprint"] = tuple(fingerprint)
        merged.setdefault("pore_id", pore_id)

        worker = AnalysisWorker(module, ch, meta, merged)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self._spin_up(worker, module.id)
        self.started.emit(module.id)
        return True

    def run_callable(
        self,
        fn: Callable[..., Any],
        label: str = "callable",
        *,
        wants_progress: bool = False,
    ) -> bool:
        """Run a callable off the main thread and emit ``finished`` with its return
        value. For the non-module analyses (null/sweep/recovery, and the
        subpopulation report).

        ``fn`` takes no arguments by default. Pass ``wants_progress=True`` and it is
        called as ``fn(report)`` instead, where ``report(fraction, message)`` reaches
        the controller's ``progressed`` signal — the same one a module's
        :class:`AnalysisWorker` uses, so the status bar cannot tell the two kinds of
        job apart. Without it a minutes-long callable can only show a bar that never
        moves, which is indistinguishable from a hang; the surrogate studies lived
        with that because they take seconds, and the subpopulation pipeline (480
        cross-validated mixture fits plus a 200-resample bootstrap) does not.

        True when a job actually started; see :meth:`start` for the refusal."""
        if not self._admit(label):
            return False
        self._supersede(label)
        worker = CallableWorker(fn, wants_progress=wants_progress)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self._spin_up(worker, label)
        self.started.emit(label)
        return True

    def _admit(self, label: str) -> bool:
        """The run gate, asked once for both entry points. Emits ``refused`` and returns
        False when another owner holds the analysis domain."""
        blocker = blocking_job(JOB_ANALYSIS, self)
        if blocker is None:
            return True
        self.refused.emit(refusal_text(blocker, label))
        return False

    def _supersede(self, label: str) -> None:
        """Stop the run in flight, if any, to make room for the job named ``label`` -- and SAY SO.

        ``abort()`` emits nothing, by design (a Cancel or a project reset reports itself), so a
        job dropped here reaches nobody's terminal slot, and whatever its owner recorded about
        it -- the Merge-all check-list, the re-measure target, a button reading "Measuring…" --
        would outlive it for the rest of the session. Both entry points come through here, so
        this is the ONE place a job is replaced and the one place to announce it. ``replaced`` goes
        out synchronously, after the old job is detached (``is_busy()`` is False) and before the
        new one exists, so a listener forgets the old job's records before the caller records
        the new job's (it does that when ``start()`` returns). Nothing is announced when
        nothing was running."""
        was_running = self.is_busy()
        self.abort()
        if was_running:
            self.replaced.emit(label)

    def abort(self, wait_ms: int = DEFAULT_JOIN_MS) -> None:
        """Ask the in-flight run to stop, detach it, and return. EMITS NOTHING.

        Called from start()/run_callable() before a new run, from the Cancel button,
        from ``_clear_session`` on File > Open Project, and from
        ``MainWindow.closeEvent``. Sets the cancel Event (which the kernels poll --
        see :mod:`nanoscope.analysis.cancel`), severs the worker's signals, and joins
        the thread for at most ``wait_ms``.

        It does NOT guarantee the thread has stopped when it returns -- see
        :meth:`abort_stopped` when the caller needs to know, because the honest answer
        is what the status bar must show. What it DOES guarantee is that nothing the
        detached worker produces can ever reach this controller or the session."""
        self.abort_stopped(wait_ms)

    def abort_stopped(self, wait_ms: int = DEFAULT_JOIN_MS) -> bool:
        """:meth:`abort`, reporting whether the worker thread had actually stopped.

        ``False`` means the run was detached and is still winding down in the
        background; its result is discarded either way. Never terminates a thread:
        ``QThread.terminate()`` on a worker holding the GIL deadlocks the whole
        application (C19), which is what this replaced."""
        worker = self._worker
        if worker is not None:
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()
        return self._stop(wait_ms)

    def is_busy(self) -> bool:
        return self._thread is not None

    def _stop(self, wait_ms: int = DEFAULT_JOIN_MS) -> bool:
        """Detach the current worker and stop its thread; True if the thread stopped.

        Shared by abort() (which first sets the cancel Event) and the terminal slots
        (which run once ``run()`` has already returned, so the join is immediate).
        Emits nothing.

        THE RETIRED WORKER IS KEPT ALIVE until the signals it already queued have been
        delivered. A worker that finished just as it was replaced has its answer sitting in the
        main thread's queue, and :meth:`_is_active` tells that stale answer from the new job's by
        ``sender()``. Freed at once, the old worker's address can be handed to the NEW worker,
        and the stale answer then names the new worker as its sender: it passes the guard, ends
        the new job and detaches it, so the new job's own result is lost. Measured, a callable
        replacing a callable whose answer was queued, 300 times over: 106, 98 and 109 stale
        answers got through. The release is queued behind the signals the worker posted before it
        was retired, so by the time it runs they have been delivered -- and dropped."""
        worker, thread = self._worker, self._thread
        self._worker, self._thread = None, None
        stopped = retire_worker_thread(worker, thread, wait_ms=wait_ms)
        if worker is not None:
            self._retired.append(worker)
            QtCore.QMetaObject.invokeMethod(
                self, "_release_retired", QtCore.Qt.ConnectionType.QueuedConnection)
        return stopped

    @QtCore.Slot()
    def _release_retired(self) -> None:
        """Let go of the oldest retired worker: one release is queued per retirement, in order,
        so this one runs after the signals of exactly the worker it frees."""
        if self._retired:
            del self._retired[0]

    # -- internals -------------------------------------------------------------------------------
    def _spin_up(self, worker: QtCore.QObject, label: str = "callable") -> None:
        """Register + start. The label defaults only so a lifetime test can drive this
        helper directly; both real entry points always pass the job's own token."""
        self._worker = worker
        self._thread = start_worker_thread(
            worker, domain=JOB_ANALYSIS, label=label, owner=self)

    def _is_active(self) -> bool:
        """A terminal signal is honoured only if it comes from the CURRENT worker.
        Guards against a stale signal queued by a worker that abort() already detached.

        ``self._worker is not None`` is load-bearing, not defensive noise: after
        abort() BOTH sides are None whenever the detached worker's C++ object has
        already been collected (``sender()`` then returns None too), and ``None is
        None`` let a cancelled run through -- which is how abort(), documented to emit
        nothing, ended up emitting ``cancelled`` and hiding the progress bar in the
        middle of a project restore (C40). The check compares OBJECTS, so it needs a retired
        worker's object to still exist when its stale signal arrives: see :meth:`_stop`."""
        return self._worker is not None and self.sender() is self._worker

    @QtCore.Slot(float, str)
    def _on_progress(self, frac: float, msg: str) -> None:
        if self._is_active():
            self.progressed.emit(frac, msg)

    @QtCore.Slot(object)
    def _on_finished(self, result: object) -> None:
        if not self._is_active():
            return
        self._stop()               # join the finished thread before announcing (is_busy -> False)
        self.finished.emit(result)

    @QtCore.Slot(str)
    def _on_failed(self, msg: str) -> None:
        if not self._is_active():
            return
        self._stop()
        self.failed.emit(msg)

    @QtCore.Slot()
    def _on_cancelled(self) -> None:
        if not self._is_active():
            return
        self._stop()
        self.cancelled.emit()


# ================================================================================================
# Batch detection: run ONE detector over MANY recordings on a single background pass.
# ================================================================================================


@dataclass(frozen=True)
class FrozenChannelSource:
    """Immutable load/calibration descriptor; no Session or eager calibrated array.

    The read-only view shares the application's immutable loaded samples. Metadata
    and calibration are frozen at dispatch, so UI edits cannot change queued jobs.
    Materialization is deliberately a worker operation, one recording at a time.
    """

    raw: np.ndarray
    meta: RecordingMeta
    channel: int
    calibration: ChannelCalibration

    @classmethod
    def capture(cls, document, channel: int, calibration: ChannelCalibration):
        raw = np.asarray(document.raw(int(channel))).view()
        raw.setflags(write=False)
        return cls(raw, deepcopy(document.meta), int(channel), deepcopy(calibration))

    def materialize(self) -> CalibratedChannel:
        """Use the existing affine kernel, preserving Session's damage disclosure."""
        from ..analysis.cancel import check_cancelled
        check_cancelled()
        cal = self.calibration
        data = cal.apply(self.raw)
        info = self.meta.channels[self.channel]
        issues: tuple[str, ...] = ()
        if cal.fingerprint() != (1.0, 0.0):
            # This is the same reporting predicate as Session.calibration_issues,
            # evaluated here without materializing every queued recording on Qt.
            raw_bad, _ = scan_nonfinite(self.raw)
            count, first = scan_nonfinite(data)
            if count > raw_bad:
                issues = (
                    f"channel {self.channel} ({info.name or 'unnamed'}) AFTER calibration "
                    f"(gain {cal.gain:g}, offset {cal.offset:g}): "
                    f"{nonfinite_reason(count, int(data.size), first)} The recording as loaded "
                    f"carried {raw_bad:,} of them; this calibration made the rest, by overflowing "
                    "float32. Any statistic computed over them — RMS noise, baseline, event depth "
                    "— is undefined. Lower the gain, or work in the raw units.",
                )
        check_cancelled()
        return CalibratedChannel(
            info=info, data=data, units=cal.target_units or info.raw_units,
            sample_rate_hz=self.meta.sample_rate_hz, t0_s=self.meta.t0_s, issues=issues)


@dataclass(frozen=True)
class BatchJob:
    """One pre-resolved detection job: the controller did the Session lookups.

    Mirrors what :class:`AnalysisWorker` is handed for a single run -- a calibrated
    ``channel`` + its ``meta`` + the merged ``params`` (carrying the injected
    ``_calibration_fingerprint`` and ``pore_id``) -- plus a human ``label`` the
    worker puts in its "recording i of N: name" progress. Holding these plain
    value objects is what keeps the worker off the Session (THREADING, above).
    """

    doc_id: str
    channel: int
    label: str
    channel_data: CalibratedChannel | None
    meta: RecordingMeta
    params: Mapping[str, Any]
    source: FrozenChannelSource | None = None


@dataclass(frozen=True)
class BatchOutcome:
    """What a batch pass produced: the results to install and any per-doc failures.

    ``results`` are :class:`AnalysisResult`s to store on the MAIN THREAD (via
    ``Session.add_result``), one per recording that detected; ``errors`` pairs a
    recording ``label`` with the failure message for the ones that raised, so a
    single bad recording is reported without sinking the whole batch.
    """

    results: tuple[AnalysisResult, ...] = ()
    errors: tuple[tuple[str, str], ...] = ()


class BatchDetectWorker(QtCore.QObject):
    """Run ``module.run(ctx)`` over MANY pre-resolved channels, off the main thread.

    The single-run :class:`AnalysisWorker` shape, looped: it detects on each
    :class:`BatchJob` in turn, reporting ``progress`` as "recording i of N: name"
    and polling the cancel Event BETWEEN recordings (so Cancel stops promptly at
    the next boundary). A recording that raises is caught and recorded in
    ``BatchOutcome.errors`` rather than aborting the pass. It emits exactly one
    terminal signal: ``finished(BatchOutcome)`` normally, ``cancelled()`` if the
    Event was set, ``failed(str)`` only on an unexpected framework error.
    """

    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)   # BatchOutcome — a frozen value object, crosses threads fine
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, module: AnalysisModule, jobs: Sequence[BatchJob]) -> None:
        super().__init__()
        self._module = module
        self._jobs = list(jobs)
        self._cancel = threading.Event()   # NOT a slot. See AnalysisWorker.cancel().

    def cancel(self) -> None:
        """Plain method, set from the GUI thread — see AnalysisWorker.cancel()."""
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            results: list[AnalysisResult] = []
            errors: list[tuple[str, str]] = []
            n = len(self._jobs)
            # Ambient for the whole pass: polling BETWEEN recordings alone means a
            # Cancel waits out the current one, and one recording is the whole 18 s
            # problem on the owner's data. Inside the scope the detector stops mid-run.
            with cancel_scope(self._cancel.is_set):
                for i, job in enumerate(self._jobs):
                    if self._cancel.is_set():
                        self.cancelled.emit()
                        return
                    self.progress.emit(i / n if n else 1.0,
                                       f"recording {i + 1} of {n}: {job.label}")
                    ctx = channel_data = None
                    try:
                        channel_data = (job.source.materialize() if job.source is not None
                                        else job.channel_data)
                        if channel_data is None:
                            raise ValueError("batch job has no frozen channel source")
                        ctx = SimpleAnalysisContext(
                            channel=channel_data, meta=job.meta,
                            params=deepcopy(dict(job.params)),
                            progress_cb=_swallow_progress,
                            cancel_flag=self._cancel.is_set,
                        )
                        results.append(self._module.run(ctx))
                    except AnalysisCancelled:
                        self.cancelled.emit()
                        return
                    except Exception as exc:  # one bad recording must not sink the batch
                        errors.append((job.label, f"{type(exc).__name__}: {exc}"))
                    finally:
                        # Release derived samples before preparing the next recording.
                        # Results carry event tables, never ownership of this context.
                        ctx = channel_data = None
            if self._cancel.is_set():
                self.cancelled.emit()
                return
            self.progress.emit(1.0, f"{len(results)} recording(s) detected")
            self.finished.emit(BatchOutcome(results=tuple(results), errors=tuple(errors)))
        except Exception as exc:  # never let a traceback escape the worker
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def _swallow_progress(frac: float, msg: str = "") -> None:
    """A module's per-sample progress is dropped during a batch (we report per-recording)."""
    return None


#: The two ``baseline.PARAM_SOURCES`` values that mean A HUMAN CHOSE THESE WINDOWS
#: FOR THIS RECORDING: ``'proposed'`` (measured here, then confirmed with "Use these
#: values") and ``'user'`` (typed into this recording's own form). Everything else is
#: a claim about somewhere else or about nobody -- ``'batch'`` says outright that the
#: windows belong to a DIFFERENT recording, ``'restored'`` came back from a project
#: file without a confirmation, ``'fallback'`` is a legacy clamp and ``'default'`` is
#: the app's own guess -- so none of them survives a re-batch. Keeping ``'batch'``
#: would be the worst of them: one batch's copy would immunise a recording against
#: every later batch, freezing the very stamp that means "never measured here".
_CONFIRMED_STAMPS = frozenset({"proposed", "user"})


def _stored_stamp_to_keep(
    session: Any, doc_id: str, channel: int, module_id: str, base: Mapping[str, Any]
) -> str | None:
    """The stored result's confirmed baseline stamp, when ``base`` holds the windows it
    was confirmed AT -- else ``None``.

    THE GATE IS THE WINDOWS, NOT THE RECORDING THE BATCH WAS LAUNCHED FROM. A stamp is
    a claim about a pair of open-pore widths, so it survives only while the batch is
    about to re-run that recording at exactly those widths; move either knob and the
    stored ``'proposed'`` is no longer true of what is running, and the ordinary
    S7-DET1 rule (`baseline.resolve_param_source` -> ``'batch'``) is the honest answer
    again. Measured through the real widgets on the owner's -600 mV / -1000 mV files:
    a re-batch at the confirmed 2.0 s / 0.5 s leaves -600 ``'proposed'`` and -1000
    ``'batch'``; a re-batch after typing 1.5 s / 0.5 s into the -600 form stamps
    ``'user'`` on -600 and ``'batch'`` on -1000 (fix-batch E1). Row counts 276 / 98
    either way -- this changes a stamp, never a row.

    Why it lives here and not in
    :meth:`~nanoscope.gui.baseline_panel.BaselinePanel.batch_params`: the panel holds
    ONE form and its OWN target, and cannot see another recording's stored result.
    This loop is the one place that has the Session and every target at once, which is
    why the keep is per target -- the scope and the source go on that job's params
    alone, so the resolver returns the kept stamp for it and ``'batch'`` for the rest.

    IT NEVER SPEAKS FOR THE RECORDING THE FORM IS ABOUT. When ``base`` already carries
    a `BATCH_SCOPE_KEY` naming this target, the form on screen is that recording's own
    form and its source is the truth about what is running; a stored stamp read back
    over it would report the last confirmation instead of the gesture just made.
    Measured (``probe_home_stamp.py``, fix-batch close): confirm 0.2 s / 0.05 s on A,
    batch, then TYPE the same two numbers into A's form ("set by hand", ``'user'``) and
    re-batch -- without this guard A came back ``'proposed'``, i.e. the panel and the
    export claimed a confirmation the user had not just made, which is the E4 defect
    wearing the fix's clothes. Everything the keep exists for is the OTHER targets.

    Float windows are compared with :func:`math.isclose` because they make the round
    trip through a result's provenance and (for a restored session) a JSON project
    file; a malformed or missing pair is not a match, never an exception.
    """
    scope = base.get(BATCH_SCOPE_KEY)
    if scope is not None:
        try:
            at_home = tuple(scope) == (doc_id, int(channel))
        except TypeError:
            at_home = scope == (doc_id, int(channel))
        if at_home:
            return None
    stored = session.results.get(doc_id, channel, module_id)
    if stored is None:
        return None
    prov = getattr(stored.payload, "provenance", None) or {}
    src = prov.get("baseline_param_source")
    if src not in _CONFIRMED_STAMPS:
        return None
    try:
        same = (
            math.isclose(float(prov["baseline_rough_s"]), float(base["baseline_rough_s"]),
                         rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(float(prov["baseline_base_s"]), float(base["baseline_base_s"]),
                             rel_tol=1e-9, abs_tol=1e-12))
    except (KeyError, TypeError, ValueError):
        return None
    return src if same else None


class BatchDetectController(QtCore.QObject):
    """Owns the batch-detection thread lifetime; mirrors :class:`AnalysisController`.

    ONE batch at a time. Holds the Session (which a module never touches) to
    resolve every ``(doc_id, channel)`` target into a calibrated channel + meta +
    the injected calibration fingerprint / pore_id -- the same lookups
    ``AnalysisController.start`` does for one run, done here up front for all of
    them so the worker only ever holds plain :class:`BatchJob` value objects. The
    caller installs the returned results on the main thread via
    ``Session.add_result``.
    """

    started = QtCore.Signal(int)            # number of recordings queued
    progressed = QtCore.Signal(float, str)
    finished = QtCore.Signal(object)        # BatchOutcome
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()
    refused = QtCore.Signal(str)            # nothing started; the sentence says why

    #: The token this controller registers with the run gate, and the one a refused
    #: single run will name back at the user.
    JOB_LABEL = "batch.detect"

    def __init__(self, session: Any, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self._thread: QtCore.QThread | None = None
        self._worker: BatchDetectWorker | None = None

    # -- public API ------------------------------------------------------------------------------
    def start(
        self,
        targets: Sequence[tuple[str, int]],
        params: Mapping[str, Any] | None = None,
        *,
        module: AnalysisModule | str = "detect.schmitt",
    ) -> bool:
        """Detect on every ``(doc_id, channel)`` in ``targets`` on one background pass.
        True when a pass actually started.

        The Session lookups happen HERE, on the main thread, so the worker only
        holds value objects. Each target's params carry the same injected
        ``_calibration_fingerprint`` (staleness stamp) and ``pore_id`` a single run
        would get; ``pore_id`` defaults to the document's ``stem_key`` so batch
        results identify their pore exactly as an interactive detection does.

        Each target's params also carry the ONE thing the batch form cannot know: when
        THIS recording's stored ``detect.schmitt`` result was confirmed by a human at
        the very windows the form now holds, that stamp is kept
        (`_stored_stamp_to_keep` -> a per-target `BATCH_SCOPE_KEY` and
        ``baseline_param_source``, which is all `baseline.resolve_param_source` needs
        to answer for this target alone). Without it a SECOND Batch detect, launched
        while the never-measured recording was selected, re-stamped the recording the
        user HAD measured ``'batch'`` and told him it was never measured -- provenance,
        the panel badge, the Stats advisory and the exported ``.meta.json``, with no
        undo (fix-batch E1, measured on the owner's -600 mV / -1000 mV files).

        Refused (``False``, ``refused`` emitted) while a single run holds the analysis
        domain -- the mirror of :meth:`AnalysisController.start`'s refusal, from the
        same one gate. ``MainWindow._batch_detect`` used to answer this case with a
        bare ``return``, i.e. a menu item that did nothing and said nothing.
        """
        blocker = blocking_job(JOB_ANALYSIS, self)
        if blocker is not None:
            self.refused.emit(refusal_text(blocker, self.JOB_LABEL))
            return False
        self.abort()

        if isinstance(module, str):
            module = ANALYSIS_REGISTRY[module]

        base = deepcopy(dict(params or {}))
        jobs: list[BatchJob] = []
        for doc_id, channel in targets:
            doc = self._session.document(doc_id)
            calibration = self._session.calibration(doc_id, channel)
            source = FrozenChannelSource.capture(doc, channel, calibration)
            fingerprint = source.calibration.fingerprint()
            merged = deepcopy(base)
            merged["_calibration_fingerprint"] = tuple(fingerprint)
            merged.setdefault("pore_id", doc.meta.stem_key)
            # A re-batch must not DOWNGRADE a stamp a human chose. See
            # `_stored_stamp_to_keep`: per target, and gated on the windows.
            keep = _stored_stamp_to_keep(self._session, doc_id, channel, module.id, base)
            if keep is not None:
                merged[BATCH_SCOPE_KEY] = (doc_id, int(channel))
                merged["baseline_param_source"] = keep
            jobs.append(BatchJob(
                doc_id=doc_id, channel=int(channel), label=doc.display_name,
                channel_data=None, meta=source.meta, params=merged, source=source))

        worker = BatchDetectWorker(module, jobs)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self._spin_up(worker)
        self.started.emit(len(jobs))
        return True

    def abort(self, wait_ms: int = DEFAULT_JOIN_MS) -> None:
        """Ask the batch to stop and detach it. EMITS NOTHING — mirrors
        :meth:`AnalysisController.abort`, including never terminating a thread."""
        self.abort_stopped(wait_ms)

    def abort_stopped(self, wait_ms: int = DEFAULT_JOIN_MS) -> bool:
        """:meth:`abort`, reporting whether the batch thread had actually stopped."""
        worker = self._worker
        if worker is not None:
            worker.cancel()
        return self._stop(wait_ms)

    def is_busy(self) -> bool:
        return self._thread is not None

    def _stop(self, wait_ms: int = DEFAULT_JOIN_MS) -> bool:
        worker, thread = self._worker, self._thread
        self._worker, self._thread = None, None
        return retire_worker_thread(worker, thread, wait_ms=wait_ms)

    # -- internals -------------------------------------------------------------------------------
    def _spin_up(self, worker: BatchDetectWorker) -> None:
        self._worker = worker
        self._thread = start_worker_thread(
            worker, domain=JOB_ANALYSIS, label=self.JOB_LABEL, owner=self)

    def _is_active(self) -> bool:
        """See :meth:`AnalysisController._is_active` — the ``_worker is not None``
        half is what stops a detached batch's terminal signal being honoured."""
        return self._worker is not None and self.sender() is self._worker

    @QtCore.Slot(float, str)
    def _on_progress(self, frac: float, msg: str) -> None:
        if self._is_active():
            self.progressed.emit(frac, msg)

    @QtCore.Slot(object)
    def _on_finished(self, outcome: object) -> None:
        if not self._is_active():
            return
        self._stop()
        self.finished.emit(outcome)

    @QtCore.Slot(str)
    def _on_failed(self, msg: str) -> None:
        if not self._is_active():
            return
        self._stop()
        self.failed.emit(msg)

    @QtCore.Slot()
    def _on_cancelled(self) -> None:
        if not self._is_active():
            return
        self._stop()
        self.cancelled.emit()
