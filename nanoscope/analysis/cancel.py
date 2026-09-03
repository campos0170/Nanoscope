"""THE cancellation invariant. One token, ambient, polled inside the loops.

Why this module exists rather than a ``cancel=`` keyword threaded by hand
------------------------------------------------------------------------
Cancellation used to be enforced *after* the work: :class:`CallableWorker` called
``self._fn()`` to completion and only then looked at its ``threading.Event``, and
nothing in :mod:`nanoscope.analysis.detect`, :mod:`~nanoscope.analysis.truth` or
:mod:`~nanoscope.analysis.characterize` polled anything at all. Measured on the
owner's smallest apo recording (2.99 M samples, -600 mV), ``sweep_threshold``
takes 18.9 s; pressing Cancel at 0.5 s therefore froze the GUI for the remaining
18 s and then -- because the controller answered a stuck thread with
``QThread.terminate()`` followed by an UNBOUNDED ``wait()`` -- deadlocked the
whole process on the GIL the terminated worker was still holding. Four runs, four
hangs. Force Quit was the only way out.

The lesson from the round before this one was that a fix applied at ONE CALL SITE
is not a fix. So cancellation is not a parameter every caller must remember to
pass down: it is an **ambient token**, installed once by whoever is running the
work (the Qt worker) and read by every kernel that loops. A new long-running
kernel becomes cancellable by calling :func:`check_cancelled` (or ticking a
:class:`CancelPoller`) in its loop -- it does not have to negotiate a keyword
through four layers of call site, and no call site can forget to forward it.

    worker thread                     pure analysis code
    -------------                     ------------------
    with cancel_scope(event.is_set):  detect_events(...)      <- polls, no kwarg
        fn()                          -> characterize(...)    <- polls, no kwarg
                                      sweep_threshold(...)    <- polls, no kwarg

The explicit ``cancel=`` argument the kernels also accept is for direct
(non-worker) use and for tests; when it is ``None`` the ambient token is used.

Thread safety: the token lives in a :class:`~contextvars.ContextVar`, and a
context is per-thread -- an OS thread that acquires the GIL gets its own Python
thread state and therefore its own context. Installing the token inside the
worker thread's ``run()`` is scoped to that thread by construction; the GUI
thread never sees it. The token itself is a plain ``threading.Event.is_set``
bound method: reading it takes no lock and involves no Qt, which is exactly why
cancel is a ``threading.Event`` and never a ``@Slot`` (see
``AnalysisWorker.cancel``).

Polling cost, measured: ``CancelPoller`` with the default stride costs under
0.2% of ``characterize`` on the 1738-event -600 mV table, and the resulting
worst-case cancel latency inside a per-event loop is a few milliseconds. See
``tests/analysis/test_cancel.py::test_polling_overhead_is_negligible``.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Union

__all__ = [
    "OperationCancelled",
    "CancelFlag",
    "CancelPoller",
    "cancel_scope",
    "check_cancelled",
    "current_cancel",
    "is_cancelled",
    "resolve_cancel",
    "EVENT_POLL_STRIDE",
]

#: A cancel token answers "has the user asked to stop?". Either a zero-arg predicate
#: or the ``threading.Event`` the workers already hold -- passing the Event where the
#: predicate belongs is the obvious slip, and it would otherwise fail deep inside a
#: loop with ``'Event' object is not callable`` instead of stopping the work.
CancelFlag = Union[Callable[[], bool], threading.Event]


class OperationCancelled(RuntimeError):
    """Raised from inside a long-running kernel that observed the cancel token.

    The Qt worker turns this into ``cancelled()``, never ``failed(str)`` -- the
    user asked to stop, which is not an error. Because it unwinds the stack, any
    partial result is discarded rather than installed: a cancelled run must leave
    NOTHING behind (that is half of what "cancel" means; the other half is that
    it actually stops).
    """


_CANCEL: contextvars.ContextVar[CancelFlag | None] = contextvars.ContextVar(
    "nanoscope_cancel", default=None
)

#: Poll every N iterations of a per-event loop. Chosen from the measured cost of a
#: bound-method call (~100 ns) against the ~50 us an event costs in
#: ``characterize``: at 64 the polling overhead is ~0.03% and the worst-case
#: latency ~3 ms, i.e. imperceptible in both directions.
EVENT_POLL_STRIDE = 64


@contextmanager
def cancel_scope(flag: CancelFlag | None) -> Iterator[None]:
    """Install ``flag`` as the ambient cancel token for this thread's context.

    The Qt workers wrap the whole of the work they run in this, so every kernel
    underneath is cancellable without a single call site being changed. Nesting
    restores the previous token on exit.
    """
    token = _CANCEL.set(resolve_cancel(flag) if flag is not None else None)
    try:
        yield
    finally:
        _CANCEL.reset(token)


def current_cancel() -> Callable[[], bool] | None:
    """The ambient cancel token for this thread, or ``None`` outside a scope."""
    return _CANCEL.get()


def resolve_cancel(cancel: CancelFlag | None = None) -> Callable[[], bool] | None:
    """An explicit token wins; otherwise fall back to the ambient one.

    Normalises a ``threading.Event`` to its ``is_set`` so callers may pass either."""
    flag = cancel if cancel is not None else _CANCEL.get()
    if isinstance(flag, threading.Event):
        return flag.is_set
    return flag


def is_cancelled(cancel: CancelFlag | None = None) -> bool:
    """Has cancellation been requested? Never raises, never propagates a token bug."""
    flag = resolve_cancel(cancel)
    if flag is None:
        return False
    try:
        return bool(flag())
    except Exception:  # a broken token must not masquerade as a cancelled run
        return False


def check_cancelled(cancel: CancelFlag | None = None) -> None:
    """Raise :class:`OperationCancelled` if cancellation has been requested.

    THE call to put in a loop. Cheap enough to run every iteration of anything
    that costs more than a few microseconds per pass; use :class:`CancelPoller`
    for the tightest per-event loops where even that shows up.
    """
    if is_cancelled(cancel):
        raise OperationCancelled("cancelled by request")


class CancelPoller:
    """Amortised :func:`check_cancelled` for a hot loop.

    ``poller()`` is called once per iteration and only actually reads the token
    every ``stride``-th time, so a loop over 100 k events pays ~1500 predicate
    calls instead of 100 k while still stopping within a few milliseconds.

    Construct one OUTSIDE the loop: resolving the ambient token is a ContextVar
    read, and doing that per iteration is the cost we are avoiding.
    """

    __slots__ = ("_flag", "_stride", "_n")

    def __init__(self, cancel: CancelFlag | None = None, *, stride: int = EVENT_POLL_STRIDE):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self._flag = resolve_cancel(cancel)
        self._stride = int(stride)
        self._n = 0

    @property
    def active(self) -> bool:
        """False when there is no token at all -- the loop can skip ticking entirely."""
        return self._flag is not None

    def __call__(self) -> None:
        """Tick once; raise :class:`OperationCancelled` when a due poll finds the flag set."""
        if self._flag is None:
            return
        self._n += 1
        if self._n >= self._stride:
            self._n = 0
            if self._flag():
                raise OperationCancelled("cancelled by request")

    def check(self) -> None:
        """Poll now, regardless of the stride."""
        if self._flag is not None and self._flag():
            raise OperationCancelled("cancelled by request")
