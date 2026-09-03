"""C19 regression: the analysis kernels must be cancellable FROM INSIDE THEIR LOOPS.

The original failure, measured on the owner's smallest apo recording (2.99 M
samples, -600 mV): ``sweep_threshold`` takes 18.9 s and NOTHING in the path polled
anything. ``CallableWorker.run`` called ``self._fn()`` to completion and only then
looked at its cancel Event; ``detect.py`` and ``truth.py`` had no cancel poll at
all; and ``characterize``'s two per-event Python loops -- the 1738-event
``rise_fall_over_set`` and ``_count_levels`` -- were where a Cancel actually landed, with
nothing to land on. Pressing Cancel therefore could not stop the work; the
controller answered by terminating the thread, and the app deadlocked.

Every test here fails on the pre-fix code: the kernels took no cancel argument and
polled no ambient token, so a cancel could only ever be observed after the work had
already finished.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from nanoscope.analysis.cancel import (
    CancelPoller,
    OperationCancelled,
    cancel_scope,
    check_cancelled,
    current_cancel,
    is_cancelled,
)
from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.truth import SynthRecording, aaft_surrogate, sweep_threshold

FS = 100_000.0


def _event_trace(n: int = 300_000, period: int = 4_000, width: int = 250, seed: int = 0):
    """A flat noisy trace with a regular train of square blockades."""
    rng = np.random.default_rng(seed)
    x = np.full(n, -100.0) + rng.normal(0.0, 2.0, n)
    for s in range(5_000, n - 5_000, period):
        x[s:s + width] += 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


class _CountingFlag:
    """A cancel token that flips True after it has been read ``after`` times."""

    def __init__(self, after: int) -> None:
        self.after = after
        self.reads = 0

    def __call__(self) -> bool:
        self.reads += 1
        return self.reads > self.after


# ================================================================================================
# the token itself
# ================================================================================================

def test_ambient_scope_is_visible_without_any_argument_being_passed():
    """THE invariant: a kernel reads the token nobody handed it.

    This is what makes the fix structural rather than another one-call-site patch --
    ``analysis_panel`` builds ``lambda: sweep_threshold(rec, detect_kwargs=kw)`` and
    forwards no cancel anything, yet the work underneath must still stop."""
    assert current_cancel() is None
    assert not is_cancelled()

    flag = threading.Event()
    with cancel_scope(flag):          # an Event is accepted where a predicate is
        assert current_cancel() == flag.is_set
        assert not is_cancelled()
        check_cancelled()          # not set yet: must not raise
        flag.set()
        assert is_cancelled()
        with pytest.raises(OperationCancelled):
            check_cancelled()

    assert current_cancel() is None   # restored on exit
    assert not is_cancelled()


def test_explicit_token_beats_the_ambient_one():
    with cancel_scope(lambda: True):
        check_cancelled(lambda: False)      # explicit wins -> no raise
    with cancel_scope(lambda: False):
        with pytest.raises(OperationCancelled):
            check_cancelled(lambda: True)


def test_a_broken_token_is_not_read_as_a_cancel():
    """A token that raises must not be mistaken for "the user pressed Cancel"."""
    def boom() -> bool:
        raise RuntimeError("token is broken")

    assert is_cancelled(boom) is False
    check_cancelled(boom)   # must not raise


def test_poller_amortises_but_still_stops():
    flag = _CountingFlag(after=0)          # True from its very first read
    poller = CancelPoller(flag, stride=8)
    for i in range(7):
        poller()                            # not due yet
    assert flag.reads == 0
    with pytest.raises(OperationCancelled):
        poller()
    assert flag.reads == 1


def test_poller_with_no_token_is_inert():
    poller = CancelPoller(None)
    assert not poller.active
    for _ in range(1000):
        poller()


def test_there_is_exactly_one_cancel_signal_in_the_codebase():
    """Every "the user asked to stop" exception must be catchable as one type.

    Two independent cancellation exceptions is the same shape of mistake as fixing an
    invariant at one call site: the Qt worker catches one of them and reports the other
    to the user as a crash. ``AnalysisCancelled`` is the pure type (the analysis layer
    must be able to raise it without importing gui), and the ML detector's narrower
    ``MLCancelled`` is a subclass of it."""
    from nanoscope.analysis.ml_detect import MLCancelled
    from nanoscope.gui.analysis_worker import AnalysisCancelled

    assert AnalysisCancelled is OperationCancelled
    assert issubclass(MLCancelled, OperationCancelled)


# ================================================================================================
# characterize: the two per-event loops the brief points at
# ================================================================================================

def _characterize_inputs(n_events: int = 400):
    n = n_events * 500 + 1000
    rng = np.random.default_rng(1)
    cur = np.full(n, -100.0) + rng.normal(0.0, 2.0, n)
    starts = np.arange(n_events, dtype=np.int64) * 500 + 500
    ends = starts + 200
    for s, e in zip(starts, ends):
        cur[s:e] += 30.0
    base = np.full(n, -100.0)
    return cur, base, starts, ends


def test_characterize_stops_inside_its_per_event_loop():
    """The loops at characterize.py :348 / :399 must poll; before the fix neither did,
    and ``characterize`` took no cancel argument at all."""
    cur, base, starts, ends = _characterize_inputs()
    flag = _CountingFlag(after=1)   # let one poll through, then cancel

    with pytest.raises(OperationCancelled):
        characterize(cur, base, starts, ends, FS, 1, noise_rms=2.0, cancel=flag)

    assert flag.reads >= 2, "the token was never polled inside the loop"


def test_characterize_returns_nothing_when_cancelled():
    """A cancelled measurement must leave NO partial table -- half a table is not a
    smaller answer, it is a wrong one."""
    cur, base, starts, ends = _characterize_inputs()
    with pytest.raises(OperationCancelled):
        characterize(cur, base, starts, ends, FS, 1, noise_rms=2.0, cancel=lambda: True)


def test_characterize_is_unchanged_when_nothing_cancels():
    cur, base, starts, ends = _characterize_inputs(n_events=50)
    a = characterize(cur, base, starts, ends, FS, 1, noise_rms=2.0)
    b = characterize(cur, base, starts, ends, FS, 1, noise_rms=2.0, cancel=lambda: False)
    assert a.equals(b)


# ================================================================================================
# detect_events: cancellable through the whole pipeline, including via the ambient token
# ================================================================================================

def test_detect_events_stops_on_an_explicit_token():
    x = _event_trace()
    with pytest.raises(OperationCancelled):
        detect_events(x, FS, polarity=1, cancel=lambda: True)


def test_detect_events_stops_on_the_ambient_token_with_no_argument():
    """``modules.DetectSchmitt`` does not forward a cancel; the scope is what reaches it."""
    x = _event_trace()
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            detect_events(x, FS, polarity=1)


def test_detect_events_result_is_bit_identical_with_a_live_token():
    """Polling must not perturb the science. Same events, same provenance numbers."""
    x = _event_trace(n=120_000)
    df_a, prov_a = detect_events(x, FS, polarity=1)
    df_b, prov_b = detect_events(x, FS, polarity=1, cancel=lambda: False)
    assert df_a.equals(df_b)
    for key in ("n_events", "noise_rms_pA", "threshold_in_pA", "n_candidate_runs"):
        assert prov_a[key] == prov_b[key]


def test_aaft_surrogate_is_cancellable_and_otherwise_identical():
    """1.68 s of uninterruptible numpy on the owner's recording, run before the first
    detection of every surrogate study."""
    x = np.random.default_rng(5).normal(size=20_000)
    with pytest.raises(OperationCancelled):
        aaft_surrogate(x, rng=0, cancel=lambda: True)
    assert np.array_equal(aaft_surrogate(x, rng=0), aaft_surrogate(x, rng=0, cancel=lambda: False))


# ================================================================================================
# the headline: the operation the user actually cancels
# ================================================================================================

def test_sweep_threshold_stops_promptly_instead_of_running_to_completion():
    """THE original failure. ``sweep_threshold`` is the slowest thing in the app; before
    the fix a Cancel could not touch it, and the user waited out the whole run (18.29 s
    on their own smallest apo recording) before anything happened -- if the app survived."""
    rec = SynthRecording(data=_event_trace(600_000), sample_rate_hz=FS, polarity=1, name="synth")

    t0 = time.perf_counter()
    uncancelled = sweep_threshold(rec)
    full_s = time.perf_counter() - t0
    assert len(uncancelled) == 16
    assert full_s > 0.5, "workload too small to be a meaningful cancel test"

    flag = threading.Event()
    timer = threading.Timer(0.15, flag.set)
    timer.start()
    try:
        t0 = time.perf_counter()
        with pytest.raises(OperationCancelled):
            sweep_threshold(rec, cancel=flag.is_set)
        cancel_s = time.perf_counter() - t0
    finally:
        timer.cancel()

    assert cancel_s < 0.5 * full_s, (
        f"cancel took {cancel_s:.2f}s of a {full_s:.2f}s run: the poll is too coarse")


def test_sweep_threshold_uses_the_ambient_token_for_its_nested_detections():
    """The GUI's job closure passes no cancel; the scope must reach all the way into
    ``detect_events`` underneath, or a Cancel waits out the current detection."""
    rec = SynthRecording(data=_event_trace(200_000), sample_rate_hz=FS, polarity=1, name="synth")
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            sweep_threshold(rec)


def test_polling_overhead_is_negligible():
    """The cost side of "polled often enough that cancel feels immediate but not so often
    it costs measurable throughput". Interleaved A/B, min-of-N, so machine drift cancels.

    Measured on the owner's -600 mV apo recording (2.99 M samples, 1738 events):
    characterize 87.5 ms -> 86.8 ms, detect_events 284.4 ms -> 286.1 ms; i.e. inside the
    noise floor in both directions."""
    cur, base, starts, ends = _characterize_inputs(n_events=800)
    live = lambda: False   # noqa: E731  a real token that never fires: worst-case cost

    bare, polled = [], []
    for _ in range(7):
        t0 = time.perf_counter()
        characterize(cur, base, starts, ends, FS, 1, noise_rms=2.0)
        bare.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        characterize(cur, base, starts, ends, FS, 1, noise_rms=2.0, cancel=live)
        polled.append(time.perf_counter() - t0)

    ratio = min(polled) / min(bare)
    assert ratio < 1.25, f"polling costs {(ratio - 1) * 100:.1f}% of characterize"
