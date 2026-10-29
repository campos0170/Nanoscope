"""THE worker-thread lifetime. Every controller spins up and retires through here.

Why this module exists
----------------------
Three controllers -- :class:`~nanoscope.gui.workers.LoadController`,
:class:`~nanoscope.gui.analysis_worker.AnalysisController` and
:class:`~nanoscope.gui.analysis_worker.BatchDetectController` -- each carried
their own copy of the same eleven lines, and every copy carried the same two
defects:

.. code-block:: python

    try:
        worker.disconnect()                 # C40: raises TypeError, ALWAYS
    except (RuntimeError, TypeError):
        pass                                #      ...and the slot still fires
    thread.quit()
    if not thread.wait(wait_ms):
        thread.terminate()                  # C19: kills a thread holding the GIL
        thread.wait()                       #      ...then waits for it FOREVER

``QObject.disconnect()`` with no arguments is a Qt C++ overload PySide6 does not
expose: it raises ``TypeError: not enough arguments``, which the ``except`` then
swallowed, so the "detached worker" protection HAD NEVER ONCE EXECUTED in the
lifetime of this codebase. And ``terminate()`` stops a thread at an arbitrary
machine instruction -- including one holding the CPython GIL, which is where a
numpy-heavy analysis spends nearly all of its time. The GIL is then never
released, the unbounded ``wait()`` that follows can never be satisfied, and the
process is dead: four measured Cancel-at-0.5 s runs of ``sweep_threshold`` on the
owner's 2.99 M-sample apo recording hung four times out of four, 0% CPU, stuck in
``take_gil`` / ``__psynch_mutexwait``. Force Quit was the only exit, and every
unsaved detection, measurement and calibration went with it.

The rules this module enforces, once, for all three controllers
---------------------------------------------------------------
1. **A worker thread is never terminated.** A thread that will not stop is
   *detached* -- severed from its controller, parked so Qt and Python keep it
   alive, and left to wind down harmlessly. Its results go nowhere because
   nothing is listening any more.
2. **No wait is ever unbounded.** :func:`retire_worker_thread` joins for at most
   ``wait_ms`` and then detaches. It returns ``False`` in that case so the caller
   can tell the user the truth instead of pretending the work stopped.
3. **Detaching actually detaches**, via the disconnect form PySide6 does
   implement.
4. **Two jobs that would answer the same question never run at once** -- see
   THE RUN GATE below. This is rule 1's sibling: rule 1 makes sure a stopped
   job's answer goes nowhere, rule 4 makes sure two live jobs never race to be
   the answer.

The join budget is deliberately short (:data:`DEFAULT_JOIN_MS`). Joining is an
optimisation for the common case where the worker has already stopped; it is not
how correctness is obtained. Correctness comes from the cooperative cancel token
(:mod:`nanoscope.analysis.cancel`) plus the sender guard in each controller.

THE RUN GATE -- one place decides whether a background job may start
--------------------------------------------------------------------
:class:`~nanoscope.gui.analysis_worker.AnalysisController` and
:class:`~nanoscope.gui.analysis_worker.BatchDetectController` write into the SAME
``ResultStore``, keyed ``(doc_id, channel, module_id)``. Nothing stopped them
running together: ``MainWindow._batch_detect`` guarded ``batch_controller.is_busy()``
but ``AnalysisPanel._run_detection`` had no busy check at all, and the Detect button
stayed live during a batch (``can_run()`` -> True). Measured through a real
``MainWindow`` on four open recordings: pressing Detect during a batch put BOTH
controllers in flight and produced two writes to one key --
``(20 events, k_in=12.0)`` from the click, then ``(20 events, k_in=5.0)`` from the
batch, which is the one that survived. The user asked for one detection and silently
got the other.

Fixing that at either call site would have been the third repeat of this project's
oldest mistake, so the decision lives HERE, at the seam every background job already
passes through:

* every job registers its ``domain`` and a human ``label`` as a side effect of
  :func:`start_worker_thread`, and de-registers in :func:`retire_worker_thread`;
* ``domain`` and ``label`` are REQUIRED keyword arguments, so a controller added
  later cannot quietly opt out of the gate -- it can only opt out on purpose, by
  passing :data:`JOB_NONE`;
* :func:`blocking_job` answers "may an owner start a job in this domain?", and
  :func:`refusal_text` turns a blocker into the sentence the user reads.

Two jobs of the same domain with the SAME owner are allowed: a controller's own
``start()`` aborts its predecessor first, which detaches it and drops its answer, so
there is no race to lose. The gate is about two DIFFERENT owners.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

from PySide6 import QtCore

__all__ = [
    "WORKER_STACK_BYTES",
    "DEFAULT_JOIN_MS",
    "JOB_NONE",
    "JOB_LOAD",
    "JOB_ANALYSIS",
    "LiveJob",
    "start_worker_thread",
    "detach_worker",
    "retire_worker_thread",
    "detached_count",
    "drain_detached",
    "unstopped_worker_threads",
    "drain_worker_threads",
    "live_jobs",
    "blocking_job",
    "describe_job",
    "refusal_text",
]

# QThread's default secondary-thread stack (512 KB on macOS pthreads) is too small for
# the LAPACK workspace Accelerate/vecLib allocates on the stack. Zero-phase IIR filtering
# -- scipy ``sosfiltfilt`` -> ``sosfilt_zi`` -> ``numpy.linalg.solve`` -- overflows it and
# the whole process dies with SIGBUS the instant a low-pass / high-pass / notch runs on the
# worker. The Qt MAIN thread has an 8 MB stack and never trips this; a plain
# ``threading.Thread`` also gets a large stack, which is why ONLY the QThread path crashed
# and no prior test (none ran an IIR filter off-thread) ever surfaced it. Give every worker
# the same headroom the main thread has, once, at the seam every background job passes
# through -- the load path reads and converts arrays too, and a reserved stack costs
# address space, not memory.
WORKER_STACK_BYTES = 16 * 1024 * 1024

#: How long a retiring controller will block the GUI thread waiting for its worker.
#: Long enough to join a worker that has already stopped (the normal case: the terminal
#: slots retire a thread whose ``run()`` has returned); short enough that Cancel never
#: costs a visible freeze. Anything slower is detached, not killed.
DEFAULT_JOIN_MS = 100

# Threads that outlived their controller. Parked -- NOT killed -- so that neither Qt
# (which aborts the process on a QThread destroyed while running) nor Python (which
# would collect the worker out from under it) can pull the rug. Entries are pruned
# once the thread reports finished.
_DETACHED: list[tuple[QtCore.QThread, QtCore.QObject]] = []

# EVERY thread this process has started, whether or not anybody still owns it.
#
# WHY THIS EXISTS AND WHY IT IS NOT ONE OF THE OTHER TWO REGISTRIES. There are three
# questions about a worker thread and they have three different answers:
#
#   * ``_LIVE``     -- "is a controller waiting on an answer?"  (the run gate). It is
#     keyed on ``domain``, so a job started with :data:`JOB_NONE` -- the Plots window's
#     fit-note batch is one -- is DELIBERATELY absent. Counting it would refuse the
#     user's next click for a job that excludes nothing.
#   * ``_DETACHED`` -- "which threads did somebody fail to stop?" Only ever appended to
#     on that failure, so a thread NOBODY EVER ASKED TO STOP is absent from it too. That
#     absence is exactly how ``MainWindow._teardown`` came to report a clean quit over a
#     running worker: it read ``drain_detached()``, which can only see threads that
#     reached :func:`retire_worker_thread`.
#   * ``_STARTED``  -- "is an OS thread still running?" The only one of the three that is
#     a property of the THREAD rather than of anybody's intent, and therefore the only
#     honest source for a teardown's return value.
#
# Held WEAKLY, deliberately. A strong reference here would keep a finished thread -- and
# through ``thread.started -> worker.run``, the worker and everything the worker's
# connections reach -- alive until the next prune, which is the leak ``_park_orphan``
# below documents from the other direction. A thread that is still RUNNING is never
# reachable only from here: its controller holds it, or ``_DETACHED`` does, or
# :func:`drain_worker_threads` parks it before letting go.
_STARTED: list[weakref.ReferenceType] = []


# ================================================================================================
# The run gate (see THE RUN GATE in the module docstring).
# ================================================================================================

#: Deliberately ungated. A job in this domain excludes nothing and is excluded by
#: nothing. It is spelled out rather than defaulted so that opting out is a choice
#: somebody made, not a keyword somebody forgot.
JOB_NONE = ""

#: Reading a recording off disk. Not exclusive with an analysis: opening a second file
#: while a detection runs is a normal thing to do, and the two touch different data.
JOB_LOAD = "load"

#: Computing an answer that will be written into the ``ResultStore``. Single-run
#: detection, ML detection, the surrogate studies, the open-pore probe, a re-measure and
#: a batch pass are all in here, because any two of them can name the same
#: ``(doc_id, channel, module_id)`` key -- and because they would otherwise be competing
#: for the same cores on the same recording.
JOB_ANALYSIS = "analysis"


@dataclass(frozen=True)
class LiveJob:
    """One background job that is currently owned by a controller.

    ``label`` is the token the controller already had for the job (a module id, or the
    label passed to ``run_callable``); :func:`describe_job` turns it into prose. The
    owner is held WEAKLY: the registry must never be the reason a controller outlives
    its window.
    """

    domain: str
    label: str
    thread: QtCore.QThread
    _owner: weakref.ReferenceType | None = None

    @property
    def owner(self) -> object | None:
        return None if self._owner is None else self._owner()


_LIVE: list[LiveJob] = []


def _prune_live() -> None:
    """Drop entries that can no longer be about anything.

    Entries are normally removed by :func:`retire_worker_thread`, which every controller
    reaches from both its terminal slots and its ``abort()``. This catches the one case
    that never gets there: a controller (a whole MainWindow, in a test) collected while
    its worker was still running. Such an entry can never receive an answer, so it must
    not go on refusing everybody else's -- and holding it also pins the QThread it names.

    DELIBERATELY NOT PRUNED ON ``thread.isFinished()``. A worker's ``finished`` signal
    ends the thread's event loop before its controller's slot has run, so a
    finished-thread rule would open the gate while a result was still queued for
    installation -- which is the two-answers-for-one-key defect, in a smaller window.
    The gate stays closed until the controller has actually retired the job."""
    if not _LIVE:
        return
    kept: list[LiveJob] = []
    orphans: list[QtCore.QThread] = []
    for job in _LIVE:
        try:
            job.thread.isRunning()
        except RuntimeError:      # the C++ QThread is gone
            continue
        if job._owner is not None and job._owner() is None:
            orphans.append(job.thread)   # the controller is gone; nothing can arrive
            continue
        kept.append(job)
    _LIVE[:] = kept
    for thread in orphans:
        _park_orphan(thread)


def _park_orphan(thread: QtCore.QThread) -> None:
    """Move an orphaned job's thread into :data:`_DETACHED` instead of letting it go.

    Written the obvious way first -- just drop the registry entry -- and it took the
    process down with ``qFatal`` on the first run, because at that moment the registry
    held the LAST reference to a RUNNING ``QThread``. Which is rule 1 of this module,
    arriving from a direction nothing had come from before. An orphan is retired exactly
    as :func:`retire_worker_thread` retires anything else: asked to quit, given no time,
    and parked so neither Qt nor Python can pull the rug.

    The worker is not parked with it: the worker's only remaining reference is the
    ``thread.started -> worker.run`` connection, which the thread itself owns, and holding
    a strong reference here would keep the worker's own signal connections -- and through
    them the very controller whose disappearance defines this case -- alive forever."""
    _prune()
    thread.quit()
    if thread.wait(0):
        return
    if all(t is not thread for t, _w in _DETACHED):
        _DETACHED.append((thread, None))   # type: ignore[arg-type]


def live_jobs() -> tuple[LiveJob, ...]:
    """Every background job a controller currently owns. Empty in an idle app."""
    _prune_live()
    return tuple(_LIVE)


def blocking_job(domain: str, owner: object | None = None) -> LiveJob | None:
    """The live job that would collide with a new ``domain`` job started by ``owner``.

    ``None`` means the start may proceed. A job owned by ``owner`` itself never blocks:
    a controller's ``start()`` aborts its own predecessor first, which detaches it and
    drops its answer, so there is nothing to race with. :data:`JOB_NONE` never blocks
    and is never blocked.
    """
    if not domain:
        return None
    _prune_live()
    for job in _LIVE:
        if job.domain != domain:
            continue
        if owner is not None and job.owner is owner:
            continue
        return job
    return None


#: Prose for the job tokens the controllers already use. A token with no entry here
#: falls back to itself, so a job added later reads awkwardly rather than anonymously.
_JOB_PROSE: dict[str, str] = {
    "detect.schmitt": "A detection",
    "detect.cusum": "A CUSUM detection",
    "detect.keras": "An ML detection",
    "sweep_threshold": "A threshold sweep",
    "null_control": "A null control",
    "recovery_map": "A recovery map",
    "baseline_probe": "An open-pore probe",
    "baseline_remeasure": "A re-measure of the event depths",
    "baseline_compare": "An estimator comparison",
    "manual_rebuild": "A rebuild of the Manual table",
    "batch.detect": "A batch detection",
    "callable": "A background analysis",
    "load": "A file load",
    "probe": "A file probe",
}


def describe_job(label: str) -> str:
    """A capitalised, human noun phrase for a job token ("A detection")."""
    known = _JOB_PROSE.get(label)
    if known is not None:
        return known
    return label or "A background analysis"


def refusal_text(blocker: LiveJob, attempted: str) -> str:
    """The sentence shown when :func:`blocking_job` refused a start.

    It names the job that is ALREADY running (the first real blocker, not a
    downstream symptom), states the process-wide admission policy, and gives the
    remedy in the window that actually owns the running job."""
    return (
        f"{describe_job(blocker.label)} is already running. This app runs one "
        f"analysis job at a time, so the request for {describe_job(attempted).lower()} "
        "was refused; nothing was started. Use Cancel in the window running that "
        "job, or wait for it to finish."
    )


def start_worker_thread(
    worker: QtCore.QObject,
    *,
    domain: str,
    label: str,
    owner: object | None = None,
    stack_bytes: int = WORKER_STACK_BYTES,
) -> QtCore.QThread:
    """Move ``worker`` onto a fresh :class:`QThread`, wire it, start it, and REGISTER it.

    ``thread.started -> worker.run`` drives the job; every terminal signal the
    worker declares (``finished`` / ``failed`` / ``cancelled``) also stops the
    thread's event loop, so a completed worker's thread ends on its own.

    ``domain`` and ``label`` are required, not defaulted, and that is the whole point of
    the run gate: a controller written later cannot fall outside it by omission. Pass
    :data:`JOB_NONE` to opt out deliberately.
    """
    thread = QtCore.QThread()
    thread.setStackSize(int(stack_bytes))   # must precede start(); see WORKER_STACK_BYTES
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for name in ("finished", "failed", "cancelled"):
        signal = getattr(worker, name, None)
        if signal is not None:
            signal.connect(thread.quit)
    if domain:
        _prune_live()
        _LIVE.append(LiveJob(
            domain=str(domain), label=str(label), thread=thread,
            _owner=None if owner is None else weakref.ref(owner)))
    # Registered BEFORE ``start()`` and unconditionally -- not under ``if domain`` --
    # so there is no instant in which a running thread is invisible, and so opting out
    # of the run gate (:data:`JOB_NONE`) never means opting out of being counted.
    _prune_started()
    _STARTED.append(weakref.ref(thread))
    thread.start()
    return thread


def detach_worker(worker: QtCore.QObject | None) -> None:
    """Sever every connection whose SENDER is ``worker``.

    ``worker.disconnect()`` -- the obvious spelling, and the one this codebase used
    everywhere -- raises ``TypeError: not enough arguments`` under PySide6 and
    disconnects nothing (C40). The four-argument static form is the one PySide6
    binds, and it is what actually makes a detached worker silent.
    """
    if worker is None:
        return
    try:
        QtCore.QObject.disconnect(worker, None, None, None)
    except (RuntimeError, TypeError):
        # RuntimeError: the C++ object is already gone -- nothing left to sever.
        # TypeError is kept only so a future PySide6 signature change degrades to
        # the old behaviour rather than taking the app down; if it ever fires again,
        # the sender guard in each controller is the second line of defence.
        pass


def retire_worker_thread(
    worker: QtCore.QObject | None,
    thread: QtCore.QThread | None,
    *,
    wait_ms: int = DEFAULT_JOIN_MS,
) -> bool:
    """Detach ``worker`` and stop ``thread``. NEVER terminates. NEVER waits unbounded.

    Returns ``True`` if the thread had stopped by the time this returned, ``False``
    if it was still running and has been detached -- in which case it is parked here
    until it winds down, its signals go nowhere, and the CALLER MUST NOT CLAIM THE
    WORK WAS STOPPED. Telling the user "cancelled" when it was not is the lie this
    return value exists to prevent.

    Also de-registers the job from the run gate. A DETACHED job is de-registered too,
    and deliberately: the gate asks "is a controller waiting on an answer", and a
    detached run's answer is already discarded by the sender guard, so it must not go
    on refusing the user's next click.
    """
    _prune()
    _unregister(thread)
    detach_worker(worker)
    if thread is None:
        return True
    # quit() before the join: a pending quit is honoured even if the thread has not
    # reached exec() yet, so the event loop ends as soon as run() returns.
    thread.quit()
    if thread.wait(max(0, int(wait_ms))):
        return True
    _DETACHED.append((thread, worker))   # type: ignore[arg-type]
    return False


def detached_count() -> int:
    """How many detached worker threads are still running. 0 in a healthy app."""
    _prune()
    return len(_DETACHED)


def drain_detached(wait_ms: int = 2000) -> int:
    """Give every detached thread a bounded chance to finish; return how many did not.

    Called on application teardown, where it is worth spending a moment so the
    process exits with no live QThread. Still bounded: a hung worker delays the quit
    by ``wait_ms``, it does not prevent it.

    THE RETURN VALUE IS NOT ADVISORY. ``MainWindow._teardown`` used to throw it away,
    and that discard is the whole of the SIGABRT on quit: a worker still running when
    the drain gave up stays in ``_DETACHED``; at interpreter shutdown that global is
    cleared, shiboken deletes the C++ ``QThread`` while its OS thread runs, and Qt calls
    ``qFatal``. Measured 4/4 runs, exit code 134, with ``QThread: Destroyed while thread
    '' is still running`` on stderr. See :func:`exit_without_destroying_live_threads`.
    """
    _prune()
    if not _DETACHED:
        return 0
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    for thread, _worker in list(_DETACHED):
        remaining = max(0, int(wait_ms) - int(deadline.elapsed()))
        thread.wait(remaining)
    _prune()
    return len(_DETACHED)


def _thread_finished(thread: QtCore.QThread | None) -> bool:
    """``True`` if ``thread`` has stopped -- or no longer exists to be running."""
    if thread is None:
        return True
    try:
        return bool(thread.isFinished())
    except RuntimeError:      # the C++ QThread is gone; nothing is running
        return True


def _prune_started() -> None:
    """Drop registrations for threads that have finished or been collected."""
    if not _STARTED:
        return
    _STARTED[:] = [ref for ref in _STARTED
                   if ref() is not None and not _thread_finished(ref())]


def unstopped_worker_threads() -> tuple[QtCore.QThread, ...]:
    """Every thread started through :func:`start_worker_thread` that has NOT finished.

    "Not finished" rather than "is running" on purpose: a thread between ``start()``
    and its first instruction reports ``isRunning() == False`` and is nevertheless
    about to run, and a teardown that called that one stopped would be telling the same
    lie in a smaller window.

    Includes detached threads (they were started here too), includes
    :data:`JOB_NONE` jobs, and includes a thread whose owner never asked it to stop --
    which is the whole reason this is not spelled ``len(_DETACHED)`` or
    ``len(live_jobs())``. Empty in an idle app.
    """
    _prune_started()
    return tuple(t for t in (ref() for ref in _STARTED) if t is not None)


def drain_worker_threads(wait_ms: int = 2000) -> int:
    """Give EVERY unstopped worker thread a bounded chance to finish; return how many
    did not, and PARK the ones that did not so nothing can delete them while they run.

    The superset of :func:`drain_detached`, and what ``MainWindow._teardown`` returns.
    ``drain_detached`` can only wait on threads that reached
    :func:`retire_worker_thread`; a thread whose owner was never asked to stop is not in
    ``_DETACHED``, so that count came back 0 with an OS thread still running -- measured
    through a real ``MainWindow``: ``teardown() returned: 0    detached_count(): 0``
    while ``thread.isRunning()`` was ``True``, followed by ``QThread: Destroyed while
    thread '' is still running``. The union below is what makes the answer honest.

    STILL BOUNDED, STILL NEVER TERMINATES. One shared ``wait_ms`` budget across all of
    them, so a second holder cannot multiply the quit's delay, and a thread that will
    not stop delays the quit -- it never prevents it, and it is never killed.

    PARKING THE SURVIVORS IS NOT BOOKKEEPING. ``gui.app.main`` reads ``drain_detached()``
    after ``app.exec()`` returns and takes :func:`exit_without_destroying_live_threads`
    when it is non-zero; a survivor that never entered ``_DETACHED`` would leave that
    last guard with nothing to see, and interpreter shutdown would delete a running
    ``QThread`` -- the ``qFatal`` this module exists to prevent, reached from the one
    direction ``retire_worker_thread`` does not cover.
    """
    _prune()
    threads: list[QtCore.QThread] = []
    seen: set[int] = set()
    for thread, _worker in list(_DETACHED):
        if id(thread) not in seen:
            seen.add(id(thread))
            threads.append(thread)
    for thread in unstopped_worker_threads():
        if id(thread) not in seen:
            seen.add(id(thread))
            threads.append(thread)
    if not threads:
        return 0
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    for thread in threads:
        remaining = max(0, int(wait_ms) - int(deadline.elapsed()))
        try:
            # quit() before the join, for the same reason retire_worker_thread does it:
            # a pending quit is honoured even if the thread has not reached exec() yet.
            thread.quit()
            thread.wait(remaining)
        except RuntimeError:      # the C++ QThread went away mid-drain
            continue
    survivors = [t for t in threads if not _thread_finished(t)]
    for thread in survivors:
        try:
            _park_orphan(thread)
        except RuntimeError:
            # The C++ QThread went away between the check above and the park. Nothing to
            # park, and NOTHING HERE MAY RAISE: this runs inside ``MainWindow._teardown``,
            # inside ``closeEvent``, and an exception escaping it would keep the window
            # open — a quit is never blocked by the machinery that reports on it.
            continue
    _prune()
    _prune_started()
    return len(survivors)


def exit_without_destroying_live_threads(code: int) -> None:
    """Leave the process at ``code`` without ever letting Qt destroy a running QThread.

    THE LAST RESORT, and only from the application's own entry point -- after
    ``app.exec()`` has returned, i.e. after the close guard has offered to save, after
    ``_save_settings`` has synced, and after :func:`drain_detached` has been given a
    generous second chance. At that point the process has nothing left to do but end,
    and the only thing an orderly Python shutdown can still add is the ``qFatal`` above.

    Why not one of the alternatives. *Holding the quit* until the worker stops is the
    behaviour a user reads as a hang, and there is nothing to wait FOR: the run was
    cancelled, its answer is already discarded, and all that remains is an OS thread
    winding down. *Releasing the wrapper's ownership* is not available -- PySide6 6.11.1
    exposes no ``releaseOwnership``, and ``shiboken6.invalidate`` deletes the C++ object,
    which is the very thing that aborts (measured: still 134, 3/3). *Telling the user* is
    addressed where a user is still present to be told (Cancel, and File > Open Project);
    at ``⌘Q`` the window is already gone.

    ``os._exit`` skips atexit handlers, garbage collection and C++ destructors. Every one
    of those is either already done (settings are synced above) or is exactly what we are
    trying to avoid. Measured: exit 0, 3/3, where the orderly path gave 134, 4/4.
    """
    import os
    import sys

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:      # a closed stream must not become the reason we do not exit
        pass
    os._exit(int(code))


def _prune() -> None:
    """Drop parked entries whose thread has finished, releasing worker + thread."""
    if _DETACHED:
        _DETACHED[:] = [(t, w) for t, w in _DETACHED if not t.isFinished()]


def _unregister(thread: QtCore.QThread | None) -> None:
    """Drop ``thread``'s run-gate registration, if it had one."""
    if thread is None or not _LIVE:
        return
    _LIVE[:] = [job for job in _LIVE if job.thread is not thread]
