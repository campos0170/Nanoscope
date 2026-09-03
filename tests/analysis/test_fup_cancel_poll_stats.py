"""H3b — the two fit families that never asked whether anyone still wanted them.

THE DEFECT, AND WHY IT IS AN INVARIANT AND NOT A SLOW FUNCTION.

H3 moved the Plots window's advisory work (the cross-family BIC table and the
"another component count fits better" ladder) onto a worker thread, and made the
panel abort that worker on every curation click with a **100 ms bounded join** on the
GUI thread. Aborting is cooperative: ``analysis.cancel`` installs an ambient token and
the work is supposed to poll it.

Only one thing in ``stats`` polled anything — the bootstrap loops. And
:func:`~nanoscope.analysis.stats.compare_dwell_models` wraps itself in
``point_fits_only()``, which is precisely the switch that turns those loops OFF. So the
two kernels the worker actually runs polled **nothing**, every abort burned its whole
100 ms budget, and then detached a still-running, GIL-holding worker into the next
click. Measured through a real ``MainWindow`` on the owner's -700 mV apo run (n = 984)
at a realistic 600 ms curation cadence, median of 8 real checkbox clicks:

    exponential          139 ms   ->  73 ms      (control, Fit = none: 48 ms)
    gaussian-mixture     189      ->  76
    exponential + gamma  297      -> 253

The natural experiment that fingered it: 7 of 8 clicks aborted a batch in *every*
family, and only ``gamma``/``inverse-gaussian`` — whose deferred job IS a bootstrap
loop, the one thing that polled — were already at the control's speed.

So the rule these tests hold the module to is not "be fast". It is:

    **Every kernel this module offers to a worker polls the ambient cancel token, at a
    granularity inside the caller's join budget — and no blanket ``except`` may file
    the token as a failed fit.**

That second clause is half the work. ``OperationCancelled`` is a ``RuntimeError``, and
this module catches ``RuntimeError`` in four places to mean "this family/replicate/start
did not work". Adding a poll without fixing those turns a cancellation into a *wrong
answer*: a MODEL COMPARISON row reading ``refused: cancelled by request``, or a CI
quietly computed from fewer replicates than it claims.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import nanoscope
from nanoscope.analysis import stats
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope, is_cancelled

#: The panel's own bounded join (``worker_thread.DEFAULT_JOIN_MS``). A kernel that
#: cannot stop inside this makes the join a fixed tax AND detaches a live worker.
JOIN_MS = 100.0


# ================================================================================================
# fixtures — a sample shaped like the owner's, because the cost is shape-dependent
# ================================================================================================

def _apo_dwells(n: int = 900, *, seed: int = 7) -> np.ndarray:
    """A narrow ~19 ms mode plus a sub-millisecond spike: the owner's real dwell shape.

    It matters that it is this shape and not an exponential sample. ``exponential +
    gamma`` is the expensive family (a 12-start Nelder-Mead), and it is expensive
    *because* the likelihood has a shallow ridge on bimodal data — on a clean
    exponential sample it converges from every start and the timing this file asserts
    would not reproduce.
    """
    rng = np.random.default_rng(seed)
    spike = np.abs(rng.normal(4.0e-4, 8e-5, n // 10))
    mode = np.abs(rng.normal(1.9e-2, 2.0e-3, n - n // 10))
    return np.ascontiguousarray(np.r_[spike, mode], dtype=np.float64)


def _depths(n: int = 600, *, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.abs(np.r_[rng.normal(1.6, 0.2, n // 2), rng.normal(2.4, 0.25, n - n // 2)])


class _Latch:
    """A cancel token that flips ``after_s`` seconds after the work starts."""

    def __init__(self, after_s: float) -> None:
        self._t0 = time.perf_counter()
        self._after = after_s

    def __call__(self) -> bool:
        return (time.perf_counter() - self._t0) >= self._after


def _stops_within(fn, *, arm_after_s: float = 0.005, tries: int = 3) -> float:
    """Run ``fn`` under a token that trips almost at once; BEST-OF-``tries`` ms to give up.

    Fails loudly if it returns NORMALLY: a kernel that finished is a kernel that did
    not stop, and reporting its (short) wall time as a cancel latency is exactly how a
    test passes against this defect.

    BEST OF THREE, not one shot, and the reason is measured rather than assumed. These
    gates compare a cancel latency against ``JOIN_MS`` = 100 ms, and the defect they
    catch is a kernel that polls NOTHING — which does not come in 12 ms over budget, it
    comes in at the full fit (189 ms per click for the mixture ladder, and far worse for
    the bootstrap families). A single wall-clock sample of a ~10 ms operation on a loaded
    machine is not a measurement of the code: on 2026-08-24 this file's mixture gate came
    in at 112.8 ms inside a full-suite run with three pytest processes sharing the box,
    and passed alone at 1.2 s total, three times running. The minimum of a few runs is
    the standard estimator of an unloaded latency, and a kernel that genuinely stopped
    polling is slow on EVERY run — so the minimum moves with the regression and the gate
    still fires. Neither the 100 ms budget nor what it defends changes.
    """
    best = float("inf")
    for _ in range(tries):
        latch = _Latch(arm_after_s)
        t0 = time.perf_counter()
        with cancel_scope(latch):
            try:
                fn()
            except OperationCancelled:
                best = min(best, (time.perf_counter() - t0) * 1000.0)
                continue
        raise AssertionError(
            "the kernel ran to completion under a cancel token that was set almost "
            "immediately: it polls nothing, so the panel's bounded join can only time out "
            "and detach it")
    return best


# ================================================================================================
# where is this tree?
# ================================================================================================

def test_the_tree_under_test_is_named_out_loud():
    """A revert stub with the wrong signature once manufactured four fake failures on
    this project and was believed. Resolve the package, say where it came from, and let
    the runner declare which root it MEANT to test."""
    import inspect

    root = Path(nanoscope.__file__).resolve().parent.parent
    expected = Path(os.environ.get(
        "NANOSCOPE_EXPECT_ROOT", "/Users/abrahamcampos/Desktop/nanoscope")).resolve()
    assert root == expected, f"`nanoscope` resolved from {root}, not {expected}"
    assert Path(inspect.getsourcefile(stats)).resolve().is_relative_to(root)


# ================================================================================================
# THE REGRESSION — the two things the advisory worker runs now stop
# ================================================================================================

def test_the_cross_family_comparison_stops_inside_the_join():
    """``compare_dwell_models`` is one of the two jobs on the worker, and it polled
    nothing: ``point_fits_only()`` switches off the bootstrap loops, which were the
    module's ONLY cancel poll."""
    t = _apo_dwells()
    uncancelled = time.perf_counter()
    stats.compare_dwell_models(t, t_min=2e-5)
    full_ms = (time.perf_counter() - uncancelled) * 1000.0

    ms = _stops_within(lambda: stats.compare_dwell_models(t, t_min=2e-5))
    assert ms < JOIN_MS, (
        f"the comparison took {ms:.0f} ms to notice a cancel (the whole scan is "
        f"{full_ms:.0f} ms); the panel joins its worker for {JOIN_MS:.0f} ms, so this "
        f"burns the join and detaches a GIL-holding worker into the next click")
    assert full_ms > 4 * ms, (
        f"the uncancelled scan is only {full_ms:.0f} ms, so {ms:.0f} ms proves nothing "
        f"about polling; this fixture is too small to test the invariant")


def test_a_cancelled_comparison_is_not_reported_as_a_family_that_refused():
    """THE TRAP THE POLL OPENS. ``OperationCancelled`` is a ``RuntimeError``, and this
    function catches ``RuntimeError`` per family to mean "this family cannot fit these
    data" -- putting the message in ``row['error']``, which the panel prints verbatim
    as ``exponential + gamma (bimodal)   refused: ...``. A poll without a re-raise
    replaces a freeze with a false statement about the data."""
    t = _apo_dwells()
    with pytest.raises(OperationCancelled):
        with cancel_scope(_Latch(0.005)):
            rows = stats.compare_dwell_models(t, t_min=2e-5)
            raise AssertionError(
                "the cancel was swallowed and filed as a per-family refusal: "
                + repr([(r["label"], r["error"]) for r in rows if r["error"]]))


def test_the_multi_start_exp_gamma_stops_between_its_starts():
    """The single most expensive kernel on the worker: twelve Nelder-Mead starts,
    ~450 ms on the owner's -700 mV sample. A per-CALL poll would still overshoot the
    100 ms join by 4x, so the poll is per START."""
    t = _apo_dwells()
    t0 = time.perf_counter()
    stats.fit_dwell_exp_gamma(t, t_min=2e-5)
    full_ms = (time.perf_counter() - t0) * 1000.0
    ms = _stops_within(lambda: stats.fit_dwell_exp_gamma(t, t_min=2e-5))
    assert ms < max(JOIN_MS, full_ms / 4), (
        f"exp+gamma took {ms:.0f} ms of its {full_ms:.0f} ms to stop -- that is not a "
        f"poll between starts")


def test_the_exponential_mixture_stops_between_its_starts():
    t = _apo_dwells()
    ms = _stops_within(
        lambda: stats.fit_dwell_exponential(t, t_min=2e-5, n_components=3, n_boot=0))
    assert ms < JOIN_MS


def test_the_peaked_point_fit_polls_before_it_maximises():
    """Cheap on its own (~4-15 ms) — but ``compare_dwell_models`` and the better-k
    ladder run it in a LOOP, and a loop of unpollable steps is an unpollable loop.

    Armed from t = 0, deliberately: one maximisation is a single indivisible step, so
    what has to be true of it is that it does not START once the token is set. Arming
    it mid-flight would only assert that a 5 ms function takes 5 ms.
    """
    t = _apo_dwells()
    for model in ("gamma", "inverse-gaussian"):
        ms = _stops_within(
            lambda m=model: stats.fit_dwell_peaked(t, model=m, t_min=2e-5, n_boot=0),
            arm_after_s=0.0)
        assert ms < JOIN_MS, f"{model} did not stop ({ms:.0f} ms)"


def test_the_depth_mixture_stops_between_its_restarts():
    """The gaussian-mixture family's own ladder. Not in the brief's table, and measured
    at 189 ms per click before this change against a 48 ms control — the same
    mechanism, found by re-measuring all six families instead of the four named."""
    x = _depths()
    ms = _stops_within(lambda: stats.fit_depth_gaussians(x, n_components=4, n_boot=0))
    assert ms < JOIN_MS


# ================================================================================================
# ...and a cancelled replicate is never read as a failed one
# ================================================================================================

def test_a_cancel_inside_a_dwell_bootstrap_replicate_stops_instead_of_dropping_it():
    """``_bootstrap_dwell`` skips a replicate whose ``_fit_mix`` raised. ``_fit_mix``
    now polls, so without a re-raise every remaining replicate would be silently
    DROPPED and the CI reported from whatever handful had already landed -- a narrower
    interval than the data supports, with no sign anything went wrong."""
    t = _apo_dwells(400)
    with pytest.raises(OperationCancelled):
        with cancel_scope(_Latch(0.02)):
            stats.fit_dwell_exponential(t, t_min=2e-5, n_components=2, n_boot=200)


def test_a_cancel_inside_a_depth_bootstrap_replicate_stops_instead_of_dropping_it():
    x = _depths(400)
    with pytest.raises(OperationCancelled):
        with cancel_scope(_Latch(0.02)):
            stats.fit_depth_gaussians(x, n_components=3, n_boot=400)


def test_nothing_polls_when_there_is_no_token():
    """The ambient token is absent on the GUI thread, and every poll above must be a
    no-op there -- this is the path every synchronous fit in the app takes."""
    t = _apo_dwells(300)
    assert not is_cancelled()
    rows = stats.compare_dwell_models(t, t_min=2e-5)
    assert [r["label"] for r in rows] and all(
        r["error"] is None or "cancel" not in r["error"] for r in rows)


# ================================================================================================
# the numbers may not move
# ================================================================================================

def test_polling_changes_no_fitted_value():
    """A poll is a predicate call between two pieces of arithmetic. Every reported
    number must be bit-identical to the same fit run with a token installed that is
    never set -- i.e. the poll path and the no-poll path agree exactly."""
    t = _apo_dwells(500)

    def numbers():
        out = {}
        for label, row in ((r["label"], r) for r in stats.compare_dwell_models(t, t_min=2e-5)):
            out[label] = (repr(row["bic"]), repr(row["aic"]), row["n"], row["error"])
        f = stats.fit_dwell_exp_gamma(t, t_min=2e-5)
        out["expgamma_params"] = {k: repr(v) for k, v in f.params.items()}
        out["expgamma_ll"] = repr(float(f.loglik))
        g = stats.fit_dwell_peaked(t, model="gamma", t_min=2e-5, n_boot=50, seed=3)
        out["gamma_params"] = {k: repr(v) for k, v in g.params.items()}
        out["gamma_ci"] = {k: (repr(a), repr(b)) for k, (a, b) in g.ci.items()}
        return out

    bare = numbers()
    with cancel_scope(lambda: False):          # a token that never trips
        polled = numbers()
    assert bare == polled


# ================================================================================================
# ci_failed — the promise that had no way to be broken
# ================================================================================================

def test_a_failed_bootstrap_can_finally_discharge_its_pending_flag():
    """``meta['ci_pending'] = True`` means "an interval is being computed and will
    replace the empty ci". Only ``apply_ci`` ever cleared it, so a bootstrap that
    RAISED left that promise standing for the life of the FitResult -- and
    ``fit_to_json`` serialises ``meta``, so it went into saved projects too."""
    t = _apo_dwells(300)
    with stats.deferred_ci() as jobs:
        fit = stats.fit_dwell_peaked(t, model="gamma", t_min=2e-5)
    assert jobs, "this family should have deferred a bootstrap"
    assert fit.meta.get("ci_pending") is True and fit.ci == {}

    stats.ci_failed(fit, "the deferred bootstrap raised")
    assert "ci_pending" not in fit.meta, (
        "a bootstrap that raised still claims an interval is on its way")
    assert fit.meta["ci_failed"] == "the deferred bootstrap raised"
    assert fit.ci == {}

    blob = stats.fit_to_json(fit)
    assert "ci_pending" not in (blob.get("meta") or {})
    back = stats.fit_from_json(blob)
    assert "ci_pending" not in back.meta and back.meta.get("ci_failed")


def test_ci_failed_is_idempotent_and_distinguishable_from_never_bootstrapped():
    """Three states, not two: an interval, a failure, and a family that offers none.
    ``exponential + gamma`` is the third and must not acquire a failure stamp."""
    t = _apo_dwells(300)
    fit = stats.fit_dwell_exp_gamma(t, t_min=2e-5)
    assert "ci_pending" not in fit.meta and "ci_failed" not in fit.meta

    with stats.deferred_ci():
        g = stats.fit_dwell_peaked(t, model="gamma", t_min=2e-5)
    stats.ci_failed(g, "boom")
    stats.ci_failed(g, "boom")
    assert g.meta["ci_failed"] == "boom" and "ci_pending" not in g.meta


def test_apply_ci_still_discharges_the_pending_flag_the_other_way():
    """The success path is unchanged -- this is the control for the test above."""
    t = _apo_dwells(300)
    with stats.deferred_ci() as jobs:
        fit = stats.fit_dwell_peaked(t, model="gamma", t_min=2e-5, n_boot=30, seed=1)
    stats.apply_ci(fit, jobs[0]())
    assert "ci_pending" not in fit.meta and "ci_failed" not in fit.meta and fit.ci


# ================================================================================================
# the process-global thing a worker must not swap out from under the GUI thread
# ================================================================================================

def test_the_warning_filter_swap_has_exactly_one_writer_and_never_blocks():
    """``warnings.filters`` is process-global and ``warnings.catch_warnings()`` is a
    save/mutate/restore with no lock. Since the advisory notes moved onto a worker,
    the GUI thread's ``_overlay_fit`` and the worker's rival ladder enter and leave it
    concurrently and restore each other's captured lists (H3 disclosed this and did not
    fix it).

    The fix may not be "serialise them": that would put the GUI thread behind a 450 ms
    optimisation, which is the freeze the whole deferral exists to remove. So a thread
    that cannot have the filters does not wait for them -- it runs unsuppressed and
    says so."""
    import warnings

    held = threading.Event()
    release = threading.Event()
    saw: dict = {}

    def holder():
        with stats.quiet_runtime_warnings() as got:
            saw["holder_got"] = got
            held.set()
            release.wait(10.0)

    th = threading.Thread(target=holder, daemon=True)
    th.start()
    assert held.wait(10.0)

    t0 = time.perf_counter()
    with stats.quiet_runtime_warnings() as got:
        waited_ms = (time.perf_counter() - t0) * 1000.0
        assert got is False, "the second thread swapped the filters while the first held them"
    assert waited_ms < 20.0, f"the second thread WAITED {waited_ms:.0f} ms for the filters"
    release.set()
    th.join(10.0)
    assert saw["holder_got"] is True

    # ...and when it does get them it really suppresses, and really puts them back.
    before = list(warnings.filters)
    with stats.quiet_runtime_warnings() as got:
        assert got is True
        with warnings.catch_warnings(record=True) as rec:
            warnings.warn("quiet please", RuntimeWarning)
        assert not rec
    assert list(warnings.filters) == before


def test_quiet_runtime_warnings_is_reentrant_within_one_thread():
    """The ladder holds it for the whole scan while a fit underneath may want it too."""
    with stats.quiet_runtime_warnings() as outer:
        with stats.quiet_runtime_warnings() as inner:
            assert outer is True and inner is True
