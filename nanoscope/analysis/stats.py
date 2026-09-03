"""Distribution fits for translocation statistics: dwell times and blockade depths.

Two models, both fit by maximum likelihood, both reporting AIC/BIC so the number
of components is *chosen* rather than assumed.

The thing this module exists to get right
-----------------------------------------
Events shorter than your detection limit are **not in your data**. They were
never counted. Fitting an exponential to the survivors as if the sample were
complete estimates ``E[t] = tau + t_min`` instead of ``tau`` -- the fit is
biased long by exactly ``t_min``, i.e. by ``t_min/tau`` in relative terms, which
for a typical ``t_min`` of 50 us against a 200 us tau is a 25% error. The fitted
curve lies beautifully on the histogram either way, because the histogram is
also missing those events. It is not detectable by eye, only by conditioning the
likelihood on survival:

    L = prod_i  f(t_i) / (1 - F(t_min))

For a single exponential this collapses to ``tau_hat = mean(t) - t_min``.

The same argument runs the other way. If the detector was given a
``max_dwell_s`` -- or the user curated out the long events -- the sample is
truncated on the RIGHT too, and a left-only likelihood then biases tau *short*
(measured -3.7% at ``t_max = 5 tau``, -24% at ``t_max = 2.5 tau``). Pass
``t_max`` and the likelihood is conditioned on ``t_min <= t <= t_max``; leave it
``None`` and everything below is the single-sided model, unchanged.

Depths have a floor for exactly the same reason: the Schmitt trigger defines an
event as a run of ``dev > k_out*noise``, so a blockade population sitting near
that threshold is cut off from below. :func:`fit_depth_gaussians` does not
(yet) model that truncation, but given ``depth_floor`` it will refuse to let you
read a clean mu/sigma off a cut-off population without being told.

Units: dwell times in seconds, depths as dimensionless dI/I0 (``frac_depth``).
"""

from __future__ import annotations

import math
import threading
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nanoscope.analysis._compat import trapezoid as _trapz
from nanoscope.analysis.cancel import OperationCancelled, check_cancelled, resolve_cancel
from nanoscope.analysis.events import FLAG_NON_FINITE, FLAG_ZERO_BASELINE

__all__ = [
    "FitResult",
    "fit_dwell_exponential",
    "fit_depth_gaussians",
    "DEPTH_FLOOR_WARNING",
    # v6 persistence: the one implementation of every fitted density, plus the
    # JSON round trip that project storage is built on. See density_from().
    "FIT_FAMILIES",
    "density_from",
    "fit_to_json",
    "fit_from_json",
    "fit_dwell_exp_gamma",
    "EXP_GAMMA",
    # A fit that is WRONG must be able to say so. See goodness_of_fit / compare_dwell_models.
    "goodness_of_fit",
    # ...and WHERE it is wrong, bin by bin, with the only chi-square this module prints.
    "binned_residuals",
    "MIN_EXPECTED_PER_BIN",
    # A band, not the corners of a parameter box. See ci_band().
    "ci_band",
    "BOOT_KEY",
    "BOOT_PARAMS_KEY",
    "compare_dwell_models",
    "DWELL_MODEL_LABELS",
    "DEFAULT_DEPTH_QUANTITY",
    # A fit whose CIs nobody reads must not pay for them. See point_fits_only().
    "point_fits_only",
    "ci_wanted",
    # ...and a fit whose CIs ARE read must not pay for them ON THE CLICK.
    # See deferred_ci().
    "deferred_ci",
    "ci_deferred",
    "apply_ci",
    "ci_failed",
    # ...and a fit nobody is looking at any more must STOP -- INSIDE its own
    # minimisation, not merely between starts. See cancel_callback().
    "cancel_callback",
    # See quiet_runtime_warnings() for the other half of "this runs on a worker
    # beside a live GUI thread".
    "quiet_runtime_warnings",
    # An R2 with no residual degrees of freedom is arithmetic, not evidence.
    # See r2_dof_tag / r2_dof_caveat -- the ONE writer both fitting surfaces use.
    "r2_dof_tag",
    "r2_dof_caveat",
    # A truncation bound is a number; WHO imposed it is a second fact, and the note
    # that prints it must not assert one. See truncation_source_clause -- the ONE
    # writer of those words -- and fit_dwell_exponential's `t_max_source`.
    "truncation_source_clause",
    "T_MAX_FROM_DETECTOR",
    "T_MAX_FROM_FIT_RANGE",
    "T_MAX_FROM_CURATION",
    # ... and WHICH `quantity=` names a dwell, so "is this a dwell fit?" has one
    # answer. See DWELL_QUANTITY_NAMES.
    "DWELL_QUANTITY_NAMES",
]

# ===========================================================================
# THE BOOTSTRAP IS THE COST OF A FIT, AND NOT EVERY CALLER READS ITS ANSWER.
# ===========================================================================
# Every fit here reports its point estimate, its AIC/BIC and a percentile
# bootstrap CI. The point estimate and the information criteria come out of ONE
# likelihood maximisation; the CI costs `n_boot` more of them. Measured at the
# owner's scale (317 dwell times): k=2 point fit 34.7 ms, with its 1000-replicate
# CI 2510 ms (72x); k=3 57.2 ms against 4367 ms (76x). The k=1 exponential has a
# closed form and a vectorised bootstrap, so it pays 10.7 against 8.8 ms.
#
# THAT IS FINE FOR A FIT SOMEBODY READS, AND WRONG FOR A FIT NOBODY DOES.
# `PlotsWindow._better_component_count` re-fits k = 1, 2, 3 purely to compare
# `FitResult.bic` -- it discards params, ci and meta -- and it runs on every
# redraw. Once the standalone windows were correctly wired to curation (C-1),
# that turned ONE accept/reject click on the owner's `_0006_Process_-600 mV.abf`
# into 3.40 s, of which 3.38 s was two rivals' discarded CIs. Measured through a
# real MainWindow: 3400 ms before, 74 ms after, 20 ms with the fit switched off.
#
# So the decision "are the CIs going to be read" is made ONCE, by the caller
# that knows, and every bootstrap in this module reads the answer from here --
# rather than each rival call site being handed its own `n_boot=0`, which is
# three places to forget and a fourth the day a fourth family is added.
#
# A ContextVar, not a module global: fits run on worker threads as well as the
# GUI thread (`worker_thread`), and a flag that leaked across them would silently
# strip the CIs off a fit somebody IS reading. The default is True, so nothing
# that does not opt in can lose a CI.
_CI_WANTED: ContextVar[bool] = ContextVar("nanoscope_fit_ci_wanted", default=True)


def ci_wanted() -> bool:
    """Whether the caller in this context will read the CIs of the fits it asks for.

    True everywhere except inside :func:`point_fits_only`. Read by every bootstrap
    in this module; there is no other switch.
    """
    return bool(_CI_WANTED.get())


@contextmanager
def point_fits_only():
    """Ask for fits WITHOUT confidence intervals, for the duration of this block.

    For a caller that reads only ``loglik``/``aic``/``bic``/``params`` -- those come
    from the maximisation, before any resampling, so they are **bit-identical** to
    what the same call returns outside this block. What changes is that ``ci`` comes
    back empty and ``meta['n_boot_ok']`` is 0.

    Use it when the fit is a means to a comparison rather than a reported number.
    Do NOT use it around a fit whose numbers reach the user: a tau with no CI beside
    it is the failure this module is otherwise built to avoid.
    """
    token = _CI_WANTED.set(False)
    try:
        yield
    finally:
        _CI_WANTED.reset(token)


# ===========================================================================
# ...AND A CI SOMEBODY *DOES* READ MUST STILL NOT BE PAID FOR ON THE CLICK.
# ===========================================================================
# `point_fits_only` answers "will anyone read these CIs?". It cannot answer the
# other question, which is the one that re-created report finding #3:
#
#   Finding #10 asked for CIs on `gamma` and `inverse-gaussian` (they used to
#   return `ci = {}`, so the only fits carrying error bars were the ones the data
#   reject). They were added. A CI that is REPORTED cannot hide inside
#   `point_fits_only`. So 200 bootstrap replicates landed on the curation click
#   path for exactly the two families that gained them: measured through a real
#   MainWindow on the owner's -700 mV apo run (n = 986), ONE accept/reject
#   checkbox click cost 507 ms with `gamma (peaked)` on screen and 1101 ms with
#   `inverse-gaussian (first passage)`, against 44 ms with the fit box set to
#   none. The budget is 150 ms.
#
# Moving the cost AGAIN -- to a longer gate, a coarser cache, a 400 ms timer --
# is the mistake this replaces. It is the same cost in a different place, and the
# timer version did not help: the freeze merely moved to 400 ms after the click
# (measured 694 ms of blocked GUI thread on `gamma`, 1266 ms on `inverse-gaussian`,
# against a 35 ms control), so the user paid it while sitting still.
#
# THE INVARIANT: a fit has TWO parts with TWO latency budgets. The point estimate
# (and therefore `params`, `loglik`, `aic`, `bic` and the drawn curve) comes out of
# ONE likelihood maximisation and is cheap -- measured on that same sample, 3.9 ms
# for `gamma` and 14.7 ms for `inverse-gaussian`. The interval costs `n_boot` more
# maximisations, 975 ms and 1634 ms respectively. The first belongs on the
# interaction path. The second belongs on a worker, and the surface that drew the
# curve is responsible for saying, out loud, that the interval is not there yet.
#
# So this does NOT strip the CI and it does NOT make it optional. Inside this block
# a fitter returns its point estimate immediately, stamps `meta['ci_pending'] = True`
# so no surface can mistake "not computed yet" for "this family has no CI", and
# appends its bootstrap -- a zero-argument callable closing over the EXACT sample,
# starting simplex, seed and replicate count the synchronous call would have used --
# to the list this yields. Run that callable anywhere (it polls the ambient cancel
# token, so a superseded one stops inside its own loop) and hand the answer to
# :func:`apply_ci`. The resulting CI is bit-identical to the synchronous one,
# because it IS the same arithmetic in the same order from the same seed.
#
# `meta` stays JSON-safe: the callable is handed back through the context manager,
# never stored on the FitResult, because `fit_to_json` has to be able to serialise
# every fit this module returns.
_CI_DEFER: ContextVar[list | None] = ContextVar("nanoscope_fit_ci_deferred", default=None)


def ci_deferred() -> bool:
    """Are fits in this context handing their bootstrap back instead of running it?"""
    return _CI_DEFER.get() is not None


@contextmanager
def deferred_ci():
    """Point estimates now; bootstraps handed back as callables, to run elsewhere.

    Yields the list the fits append to::

        with stats.deferred_ci() as jobs:
            fit = stats.fit_dwell_peaked(dwell, model="gamma", t_min=t0)
        # fit.params / .bic / .predict() are final and identical to a plain call;
        # fit.ci is {} and fit.meta['ci_pending'] is True
        payload = jobs[0]()                 # <- on a worker thread
        stats.apply_ci(fit, payload)        # <- back on the owning thread

    Nests inside :func:`point_fits_only` the sensible way round: a caller that does
    not want CIs at all appends nothing, because there is no interval to defer.

    A fitter that has no bootstrap to offer (``fit_dwell_exp_gamma``, whose every
    replicate is a 12-start optimisation, so it reports no CI by design) appends
    nothing and is unchanged -- ``jobs`` may legitimately come back empty, and a
    caller must not assume one job per fit.
    """
    jobs: list[Callable[[], dict[str, Any]]] = []
    token = _CI_DEFER.set(jobs)
    try:
        yield jobs
    finally:
        _CI_DEFER.reset(token)


def apply_ci(fit: "FitResult", payload: Mapping[str, Any]) -> "FitResult":
    """Install a deferred bootstrap's answer onto the fit it came from.

    Mutates and returns ``fit``: ``ci``, ``meta['n_boot_ok']``, the "bootstrap
    failed" note when there were too few usable replicates, and the removal of
    ``meta['ci_pending']`` -- which is what turns "not computed yet" back into a
    finished fit. The result is what the synchronous call would have returned.

    Idempotent, because the caller is a redraw: applying the same payload to a
    freshly refit result twice must not append the note twice, so it is only added
    when it is not already there.
    """
    fit.ci = {k: (float(v[0]), float(v[1])) for k, v in (payload.get("ci") or {}).items()}
    fit.meta["n_boot_ok"] = int(payload.get("n_boot_ok") or 0)
    # The replicates cross the same seam as the intervals, or a deferred CI would be the
    # one path that can never draw a band (see ci_band()). Absent stays absent rather than
    # overwriting with None: a payload from an older caller must not erase what is there.
    if payload.get("boot") is not None:
        fit.meta[BOOT_KEY] = np.asarray(payload["boot"], dtype=np.float64)
    fit.meta.pop("ci_pending", None)
    note = payload.get("note")
    if note:
        warns = fit.meta.setdefault("warnings", [])
        if str(note) not in warns:
            warns.append(str(note))
    return fit


def ci_failed(fit: "FitResult", reason: str = "") -> "FitResult":
    """The OTHER way a deferred bootstrap ends: it raised, and there is no payload.

    ``meta['ci_pending'] = True`` is a PROMISE -- "an interval is being computed for
    this fit and will replace the empty ``ci``". :func:`apply_ci` is the only thing
    that ever discharged it, so a bootstrap that raised left the promise standing for
    the lifetime of the FitResult: ``ci = {}`` beside ``ci_pending = True`` for ever,
    which is the state that means "still coming" and is now permanently a lie. It
    travels: ``fit_to_json`` serialises ``meta``, so a saved project carries the
    forever-pending flag, and any surface that reads it (the Details pane's
    placeholder, a future consumer of ``fit_result()``) reads "wait" instead of "this
    family has no interval on this sample".

    So a failure has to be settled explicitly, exactly like a success. Mutates and
    returns ``fit``: drops ``ci_pending``, records ``ci_failed`` (a fit that never had
    a bootstrap at all is a THIRD state and must stay distinguishable from one whose
    bootstrap blew up), and leaves ``ci`` empty.

    Idempotent for the same reason :func:`apply_ci` is: the caller is a redraw.
    """
    fit.meta.pop("ci_pending", None)
    fit.meta["ci_failed"] = str(reason) if reason else True
    return fit


# ---------------------------------------------------------------------------
# ...and the process-global thing a worker must not swap out from under the GUI
# ---------------------------------------------------------------------------
#: ``warnings.filters`` is PROCESS-global and ``warnings.catch_warnings()`` is a
#: save/mutate/restore on it with no lock of its own. Since the advisory notes moved
#: onto a worker thread, two threads enter and leave that save/restore concurrently --
#: the panel's own ``_overlay_fit`` on the GUI thread against the rival ladder and
#: :func:`compare_dwell_models` on the worker -- and the interleaving
#: ``save(A) save(B) restore(B) restore(A)`` puts back a filter list that was captured
#: while the OTHER thread had already installed its "ignore". Nothing numeric passes
#: through it; what it decides is whether a RuntimeWarning the user should see is
#: printed, or one they should not is.
#:
#: THE LOCK IS NEVER WAITED ON. Serialising the two threads would put the GUI thread
#: behind a 450 ms optimisation on the worker, which is the freeze this whole deferral
#: exists to remove -- trading a latency defect for a correctness one is how this
#: project got here. So: exactly one thread at a time may swap the filters, and a
#: thread that cannot have it right now simply does not swap them and lets the warning
#: through. A printed RuntimeWarning is the cheapest possible failure mode, and every
#: message suppressed here is also stamped into ``FitResult.meta['warnings']``, so
#: nothing the user needs is lost either way.
_WARN_FILTER_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# THE POLL THAT REACHES INSIDE A MINIMISATION
# ---------------------------------------------------------------------------
def cancel_callback(cancel: Any = None):
    """A ``scipy.optimize.minimize`` ``callback=`` that polls the cancel token, or ``None``.

    THE INVARIANT THIS EXISTS FOR: **a bounded join is a promise about the worst case,
    and a poll that can only be reached BETWEEN units of work cannot keep it on a
    sample where one unit exceeds the bound.**

    Every multi-start fitter in this module polled once per start, which is a poll
    between units. That is a bound only while a start is short. Measured on the
    owner's -600 mV apo recording (n = 240 after merge, the sample the Plots window
    actually fits), one ``fit_dwell_exp_gamma`` is **9.5 s** -- ~790 ms per start --
    against 207 ms for the whole twelve on -700 mV (n = 986). The panel's abort joins
    its advisory worker for :data:`~nanoscope.gui.worker_thread.DEFAULT_JOIN_MS` = 100
    ms, so on that recording a start is EIGHT TIMES the entire join and the poll could
    never be reached inside one: two aborts in three blew the join and detached a
    still-running, GIL-holding worker into the next curation click.

    ``scipy``'s Nelder-Mead calls ``callback`` once per iteration, in the ``finally``
    of its own loop body -- so the poll lands inside the minimisation, and the
    worst-case cancel latency becomes ONE ITERATION (~25 us on that sample) instead of
    one start. Only ``StopIteration`` is caught by ``_call_callback_maybe_halt``;
    anything else propagates, which is exactly what
    :class:`~nanoscope.analysis.cancel.OperationCancelled` wants to do.

    ``None`` WHEN THERE IS NO TOKEN, deliberately. Off a worker -- a test, a notebook,
    the Fits panel's own synchronous call -- there is nothing to poll, and returning
    ``None`` means ``minimize`` is handed exactly the arguments it was handed before
    this function existed. The cost of cancellability is paid only where cancellation
    is possible.

    The parameter is named ``intermediate_result`` because ``scipy._wrap_callback``
    reads the signature: that name gets the ``OptimizeResult`` passed by keyword with
    no ``np.copy`` of the iterate, and older scipy, which passes the vector
    positionally, still binds it. Either way the value is ignored.

    Args:
        cancel: An explicit token; ``None`` (the default) uses the ambient one.

    Returns:
        A one-argument callable, or ``None`` when no token is installed.
    """
    flag = resolve_cancel(cancel)
    if flag is None:
        return None

    def _poll(intermediate_result=None):
        if flag():
            raise OperationCancelled("cancelled by request")

    return _poll


#: The accuracy budget of a peaked-family maximisation: how many Nelder-Mead
#: iterations it may spend before it is cut off. Unchanged -- a healthy fit on any of
#: the owner's recordings finishes in 68-83.
_NM_MAXITER = 4000

#: The LATENCY budget of the same maximisation, in iterations that changed nothing.
#:
#: WHY A SECOND BUDGET EXISTS. ``maxiter`` bounds a search that is still moving. It is
#: not a bound on a search that has ALREADY STOPPED and cannot say so: scipy's
#: Nelder-Mead terminates when the simplex spread is within ``xatol`` *and* the spread
#: of function values within ``fatol``, both tested on the LOG parameters this module
#: optimises in -- so an optimum that lies at the edge of the family (a gamma whose
#: shape ``k -> 0`` is ``log k -> -inf``) can never satisfy the first, whatever the
#: second does. The cap then becomes the only stopping rule that ever fires.
#:
#: MEASURED, on the owner's -600 mV merged table (n = 238, ``t_min`` guessed from the
#: data because the pushed table carried no detector provenance): the best vertex is
#: BIT-IDENTICAL from iteration 82 to iteration 4000, and the parameters returned at
#: ``maxiter=200`` and at ``maxiter=200000`` agree to every digit (k = 2.69778e-11,
#: theta = 0.0153339, nll = -886.6304335576). The 3,916 iterations in between cost
#: 218 ms of a 220 ms fit -- on the curation CLICK path, against a 150 ms budget --
#: and bought nothing. Six of twenty flips of one event did this on gamma, four of
#: twenty on inverse-gaussian: the bimodal click the last report could not explain.
#:
#: WHY 200. Over 160 healthy fits (both peaked models, both merged tables, twenty
#: single-event curation flips each, with and without the detector's real ``t_min``)
#: the LONGEST run of a frozen best vertex before a successful convergence was 20
#: iterations. 200 is ten times that margin, and it is measured rather than guessed
#: because a guard that fires on a fit that was about to improve would move a number.
_NM_STALL_ITERS = 200


class _StallStop:
    """A ``minimize`` callback that stops a search which has stopped moving.

    Wraps the ambient cancel poll (:func:`cancel_callback`, which may be ``None``) so
    one callback slot serves both: cancellation still raises
    :class:`~nanoscope.analysis.cancel.OperationCancelled` straight out of ``minimize``,
    and the stall raises ``StopIteration``, which is the one exception
    ``scipy._lib._util._call_callback_maybe_halt`` catches and turns into a clean halt.

    IT COMPARES THE BEST VERTEX, NOT THE OBJECTIVE, and that is deliberate: the
    objective at the best vertex would cost one extra evaluation of the likelihood per
    iteration -- ~30% on top of the fit this is meant to make cheaper -- while the
    vertex itself is already in hand. Equality is EXACT (``array_equal``): a vertex
    that moved by one ulp is a search that is still moving, and this guard exists for
    the case where it has not moved at all.

    ``stopped`` is the flag the caller reports from, and it is the caller's OWN record
    that the search was cut off. scipy 1.11+ also marks a callback halt
    (``success=False``, ``status=99``, ``message="`callback` raised `StopIteration`."``)
    -- but that is post-processing in ``_minimize``, not in the Nelder-Mead kernel,
    which leaves ``warnflag`` at 0 for a break that is neither maxiter nor maxfev. A
    fact this module reports to the user does not rest on which of the two ran.
    """

    __slots__ = ("_inner", "_limit", "_last", "_same", "stopped")

    def __init__(self, inner=None, limit: int | None = None) -> None:
        self._inner = inner
        # Read at CONSTRUCTION, not bound as a default at class-definition time: a
        # default argument would freeze the module constant into the signature, and a
        # test that needs to see what the guard is protecting against (the full
        # 4000-iteration run, to prove the two return the same numbers) could then
        # only get there by editing this file.
        self._limit = int(_NM_STALL_ITERS if limit is None else limit)
        self._last = None
        self._same = 0
        self.stopped = False

    def __call__(self, intermediate_result=None):
        # The cancel poll goes FIRST and is never swallowed: a cancelled fit must stop
        # for the token even on the iteration the stall guard would also have fired.
        if self._inner is not None:
            self._inner(intermediate_result)
        x = getattr(intermediate_result, "x", intermediate_result)
        if x is None:                       # pragma: no cover - no iterate to compare
            return
        x = np.asarray(x, dtype=np.float64)
        if self._last is not None and np.array_equal(x, self._last):
            self._same += 1
            if self._same >= self._limit:
                self.stopped = True
                raise StopIteration
        else:
            self._same = 0
            self._last = x.copy()


@contextmanager
def quiet_runtime_warnings():
    """Suppress ``RuntimeWarning`` here, without racing another thread's filter swap.

    Yields True when the filters were actually swapped, False when another thread held
    them and this block ran unsuppressed. Re-entrant within one thread (an outer
    ladder-wide block may contain an inner per-fit one).
    """
    if not _WARN_FILTER_LOCK.acquire(blocking=False):
        yield False
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            yield True
    finally:
        _WARN_FILTER_LOCK.release()


_MIN_EVENTS = 20
_ETA_FLOOR = 1e-12

#: What the user is told when the depth mixture is fit to a floor-truncated
#: sample (B4b). One string, formatted at the single site that detects the
#: condition, so every window quotes the same sentence.
#: ``{unit}`` is a PRE-SPACED suffix ("" for a dimensionless fit, " pA", " ms") and is
#: required: this warning states four magnitudes, and four magnitudes with no unit is
#: the hole the identifiability notes had on the same surface. Formatted only by
#: :func:`fit_depth_gaussians`, which decides the reading unit once for every note it
#: writes.
DEPTH_FLOOR_WARNING = (
    "DEPTH FLOOR: the detector only reports excursions above threshold_out, i.e. depths "
    "above ~{floor:.4g}{unit}, but the fitted shallowest component (mu_1 = {mu:.4g}{unit}, "
    "sigma_1 = {sigma:.4g}{unit}, mu_1 - 2*sigma_1 = {edge:.4g}{unit}) puts {mass:.1%} of its "
    "mass below "
    "that floor. This mixture is UNtruncated, so that component is fit to a population "
    "missing its own left side: mu_1 is biased HIGH and sigma_1 biased LOW (measured +7% / "
    "-22% through the real detector). Treat mu_1/sigma_1 as floor-biased, not as a "
    "measurement."
)

#: Mass the shallowest component may place below the detector's floor before the
#: fit is called floor-biased. The audit's own rule of thumb is
#: ``mu_1 - 2*sigma_1 < floor``, i.e. 2.3% -- but truncation shrinks sigma, which
#: pushes ``mu_1 - 2*sigma_1`` back UP and lets a genuinely cut-off population
#: slip under that exact test (measured: a sample truncated hard enough to bias
#: sigma_1 by -23% still came back at 2.1%). 1% keeps the audit's condition and
#: closes that gap.
_DEPTH_FLOOR_MASS = 0.01

#: What :func:`fit_depth_gaussians` calls the fitted quantity when the caller does
#: not say. The dimensionless ``frac_depth`` is the column the function was written
#: for, so it stays the default -- but it must be a DEFAULT and not a hardcoded fact,
#: because the same EM is used on ``depth_pA``, ``max_dev_pA`` and ``mean_dev_pA``,
#: and labelling picoamps "dI/I0" is the one confusion this dataset cannot afford
#: (report High-4: apo-ferritin has frac_depth 1.81-2.57, so a dI/I0 caption over a
#: 85-106 pA mu list reads as a 100x blockade fraction).
DEFAULT_DEPTH_QUANTITY = "dI/I0"

# ===========================================================================
# DEGENERACY AND IDENTIFIABILITY: A MIXTURE THAT FOUND NOTHING MUST SAY SO.
# ===========================================================================
# A k-component fit ALWAYS returns k components. Nothing in the arithmetic
# refuses when the data support fewer, so the panel prints k means and the reader
# reports k populations. Measured on the owner's apo-ferritin run:
#
#   -700 mV dwell, exponential k=2:  tau_1 = tau_2 = 19.809411 ms to 8 s.f.
#                                    (weights 0.88/0.12) -- one component wearing
#                                    two sets of parameters (report Low-20).
#   -700 mV depth_pA, k=3:           mu_1 CI [17.6, 87.6] pA -- five times the
#                                    component's own sigma; component 2 carries
#                                    2.7% weight with a weight CI of [0.014, 0.671]
#                                    (report Medium-11).
#
# Every threshold below is scale-free and was chosen against those measurements,
# so each one states a condition a reader can check by hand rather than a taste.

#: Two exponential taus (or two mixture means) closer than this, RELATIVE to their
#: own size, are the same component twice. The -700 mV pair above differ by 3.6e-8
#: relative; 1e-3 is three orders of magnitude looser than that and still far
#: tighter than any separation a bootstrap could resolve.
_DEGENERATE_REL_GAP = 1e-3

#: A component carrying fewer than this many events, by weight, is decoration:
#: :func:`_clean_1d` refuses to fit a whole SAMPLE smaller than ``_MIN_EVENTS``, so
#: a component supported by less than that is being estimated from a sample the
#: module would not accept on its own.
_MIN_COMPONENT_EVENTS = _MIN_EVENTS


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """A fitted model: parameters, uncertainty, and the curve to plot.

    Attributes:
        params: Point estimates, e.g. ``{'tau_1': 2.4e-4, 'w_1': 0.6, ...}``.
            Exponential taus are the **untruncated** physical time constants,
            not the mean of the observed (truncated) sample.
        ci: ``{param: (lo, hi)}`` percentile bootstrap intervals.
        loglik: Log-likelihood of the observed data under the fitted model,
            including the truncation / efficiency normalisation.
        aic, bic: Information criteria; lower is better. Comparable only
            between fits to the *same* data under the *same* censoring model.
        n: Number of observations fit.
        model: Human-readable model name.
        meta: Diagnostics (bootstrap count, convergence, censoring model, ...),
            including ``param_units`` -- see :meth:`unit_of`.
    """

    params: dict[str, float]
    ci: dict[str, tuple[float, float]]
    loglik: float
    aic: float
    bic: float
    n: int
    model: str
    meta: dict[str, Any] = field(default_factory=dict)
    _pdf: Callable[[np.ndarray], np.ndarray] | None = None

    def predict(self, x) -> np.ndarray:
        """Density of the **observed** data at ``x`` (unit-area against a density histogram).

        This is the censored density -- ``eta(t) f(t) / Z`` -- not the underlying
        ``f(t)``. That is deliberate: it is what a histogram of your detected
        events actually follows, so it overlays correctly. The uncensored physics
        lives in ``.params``.
        """
        if self._pdf is None:
            raise RuntimeError("this FitResult carries no density")
        x = np.asarray(x, dtype=np.float64)
        return self._pdf(np.atleast_1d(x)).reshape(x.shape) if x.ndim else self._pdf(
            np.atleast_1d(x)
        )[0]

    def unit_of(self, param: str) -> str:
        """The unit of ONE fitted parameter, or ``""`` when it is dimensionless.

        THE UNIT TRAVELS WITH THE FIT, PER PARAMETER, BECAUSE A FIT IS NOT ALL ONE
        UNIT. ``gamma`` returns a dimensionless shape ``k`` beside a ``theta`` in
        seconds; the depth mixture returns ``mu``/``sigma`` in whatever the fitted
        column was and a dimensionless ``w``. A window that captions the whole
        FitResult with one unit is therefore guaranteed to mislabel something, which
        is how the Details pane came to print ``mu = 85.46, 99.69, 105.8`` -- picoamps
        -- under a model line that said ``dI/I0`` (report High-4). Every fitter in this
        module stamps ``meta['param_units']``; every window reads it from here.

        Unknown parameters answer ``""`` rather than raising: a caption is not worth
        an exception, and a missing unit is the pre-existing behaviour.
        """
        units = self.meta.get("param_units") or {}
        return str(units.get(param, ""))

    def summary(self) -> str:
        """A readable table of estimates and 95% bootstrap CIs."""
        lines = [
            f"{self.model}",
            f"  n = {self.n}   loglik = {self.loglik:.4g}   AIC = {self.aic:.4g}   BIC = {self.bic:.4g}",
        ]
        cens = self.meta.get("censoring")
        if cens:
            lines.append(f"  censoring: {cens}")
        # DOES THIS MODEL DESCRIBE THESE DATA? Printed on every fit, pass or fail,
        # because a parameter table with no verdict beside it reads as a measurement
        # whatever the curve is doing (report High-3). Stamped by `_stamp_gof`, so this
        # surface and the Plots window quote one number, not two computations of it.
        gof = self.meta.get("gof")
        if isinstance(gof, Mapping) and "ks" in gof:
            lines.append(f"  goodness of fit: {gof.get('note', '')}")
        w = max((len(k) for k in self.params), default=1)
        lines.append(f"  {'param':<{w}}  {'estimate':>12}  {'95% CI':>26}")
        for k, v in self.params.items():
            lo, hi = self.ci.get(k, (np.nan, np.nan))
            ci = f"[{lo:>10.4g}, {hi:>10.4g}]" if np.isfinite(lo) else f"{'--':>24}"
            unit = self.unit_of(k)
            lines.append(f"  {k:<{w}}  {v:>12.6g}  {ci:>26}" + (f"  {unit}" if unit else ""))
        for note in self.meta.get("warnings", []):
            lines.append(f"  ! {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<FitResult {self.model} n={self.n} BIC={self.bic:.4g}>"


# ---------------------------------------------------------------------------
# input validation -- fail loudly, a silent wrong tau is the worst outcome
# ---------------------------------------------------------------------------


#: How many offending row labels a refusal quotes before it says "and N more". Long
#: enough to be actionable in the Event table's search box, short enough to read.
_MAX_NAMED_ROWS = 8


def _row_labels(index: Any, mask: np.ndarray) -> str:
    """``"rows 3, 17, 42"`` for the offending entries, in the caller's OWN labels.

    THE POINT OF THIS FUNCTION IS THE WORD "rows". A refusal that says only "contains
    3 non-finite value(s). Fix them upstream" tells a user who has just unticked "Use
    accepted events only" nothing they can act on: the Plots panel handed a whole
    column over, the offending rows are automatically-rejected ``non_finite`` or
    ``zero_baseline`` ones, and there is nothing on screen that says which. Naming them
    turns the message into an instruction -- go and look at these rows, and at their
    ``flag`` column, which already says why.

    A `pandas.Series` carries the event table's own index, so the labels quoted here
    are the ones the table shows; a bare array can only be quoted by position, and the
    message says which of the two it is.
    """
    where = np.flatnonzero(mask)
    if index is None:
        labels = [str(int(i)) for i in where[:_MAX_NAMED_ROWS]]
        lead = "at positions"
    else:
        labels = [str(v) for v in np.asarray(index)[where[:_MAX_NAMED_ROWS]]]
        lead = "at rows"
    more = "" if where.size <= _MAX_NAMED_ROWS else f" and {where.size - _MAX_NAMED_ROWS} more"
    return f"{lead} {', '.join(labels)}{more}"


def _clean_1d(x, name: str, *, positive: bool) -> np.ndarray:
    index = None
    if isinstance(x, pd.Series):
        index = x.index
        x = x.to_numpy()
    x = np.asarray(x, dtype=np.float64).ravel()
    if index is not None and len(index) != x.size:
        index = None                     # a ravel changed the shape; positions it is
    if x.size == 0:
        raise ValueError(f"{name} is empty; nothing to fit")
    if not np.all(np.isfinite(x)):
        bad = ~np.isfinite(x)
        n_bad = int(bad.sum())
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s), {_row_labels(index, bad)}. "
            f"Fix or drop them upstream -- silently dropping them here would change the "
            f"sample size behind your back. Those rows carry a 'flag' saying why they "
            f"were not measured ({FLAG_NON_FINITE}, {FLAG_ZERO_BASELINE}); ticking "
            f"'Use accepted events only' excludes them, which is what every export "
            f"already does."
        )
    if positive and np.any(x <= 0):
        bad = x <= 0
        n_bad = int(bad.sum())
        raise ValueError(f"{name} contains {n_bad} non-positive value(s), "
                         f"{_row_labels(index, bad)}; dwell times must be > 0")
    if x.size < _MIN_EVENTS:
        raise ValueError(
            f"{name} has only {x.size} events; refusing to fit fewer than {_MIN_EVENTS}. "
            f"A distribution fit to a handful of events is decoration, not a measurement."
        )
    if np.ptp(x) == 0:
        raise ValueError(
            f"all {name} values are identical ({x[0]:g}); there is no distribution to fit. "
            f"This usually means a quantisation or detector bug upstream."
        )
    return x


# ---------------------------------------------------------------------------
# detection efficiency eta(t)
# ---------------------------------------------------------------------------

_DWELL_KEYS = ("dwell_s", "dwell", "true_dwell_s", "dwell_true_s", "t_s", "t", "dwell_time_s")
_ETA_KEYS = (
    "efficiency", "eta", "recovery", "recall", "detected_frac", "recovered_frac",
    "frac_detected", "p_detect", "tpr", "detection_efficiency", "hit_rate",
)
_NUM_KEYS = ("n_detected", "n_recovered", "n_found", "detected")
_DEN_KEYS = ("n_injected", "n_true", "n_total", "injected")


def _eta_from_frame(df: pd.DataFrame, notes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cols = {c.lower(): c for c in df.columns}
    tcol = next((cols[k] for k in _DWELL_KEYS if k in cols), None)
    ecol = next((cols[k] for k in _ETA_KEYS if k in cols), None)
    if tcol is None:
        raise ValueError(
            f"efficiency table has no recognisable dwell column (looked for {_DWELL_KEYS}); "
            f"columns are {list(df.columns)}"
        )
    t = np.asarray(df[tcol], dtype=np.float64)
    if ecol is not None:
        e = np.asarray(df[ecol], dtype=np.float64)
    else:
        ncol = next((cols[k] for k in _NUM_KEYS if k in cols), None)
        dcol = next((cols[k] for k in _DEN_KEYS if k in cols), None)
        if ncol is None or dcol is None:
            raise ValueError(
                f"efficiency table has no recognisable efficiency column (looked for {_ETA_KEYS}, "
                f"or a detected/injected count pair); columns are {list(df.columns)}"
            )
        den = np.asarray(df[dcol], dtype=np.float64)
        if np.any(den <= 0):
            raise ValueError("efficiency table has a non-positive injected count")
        e = np.asarray(df[ncol], dtype=np.float64) / den

    # recovery_map may be a 2-D map (dwell x depth). Collapsing it to eta(t)
    # marginalises over the injected depth grid, which is only right if that grid
    # resembles the depth distribution of the sample being fit.
    g = pd.DataFrame({"t": t, "e": e})
    if g["t"].duplicated().any():
        notes.append(
            "efficiency table had multiple rows per dwell (a dwell x depth map); "
            "averaged over the depth axis, which assumes the injected depth grid "
            "matches your sample's depth distribution"
        )
        g = g.groupby("t", as_index=False)["e"].mean()
    g = g.sort_values("t")
    return g["t"].to_numpy(), g["e"].to_numpy()


def _resolve_efficiency(efficiency, t_min: float, t_max: float,
                        notes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a recovery_map result (or friends) into a sorted (dwell_grid, eta) pair."""
    if isinstance(efficiency, pd.DataFrame):
        t, e = _eta_from_frame(efficiency, notes)
    elif isinstance(efficiency, dict):
        t, e = _eta_from_frame(pd.DataFrame(efficiency), notes)
    elif isinstance(efficiency, (str, bytes)):
        raise TypeError(
            "efficiency must be the DataFrame from truth.recovery_map(), a callable eta(t), "
            f"or a (dwell_grid, eta) pair; got {type(efficiency).__name__}"
        )
    elif callable(efficiency):
        # grid spans the data, not just t_min: eta must be sampled where the
        # events actually are or the flat-extrapolation region swallows the fit
        t = np.geomspace(min(t_min, t_max) * 0.5, t_max * 10.0, 1024)
        e = np.asarray(efficiency(t), dtype=np.float64)
    elif isinstance(efficiency, (tuple, list)) and len(efficiency) == 2:
        t = np.asarray(efficiency[0], dtype=np.float64).ravel()
        e = np.asarray(efficiency[1], dtype=np.float64).ravel()
    else:
        try:
            arr = np.asarray(efficiency, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError(
                "efficiency must be the DataFrame from truth.recovery_map(), a callable eta(t), "
                f"or a (dwell_grid, eta) pair; got {type(efficiency).__name__}"
            ) from None
        if arr.ndim != 2 or 2 not in arr.shape:
            raise TypeError(
                "efficiency must be the DataFrame from truth.recovery_map(), a callable eta(t), "
                f"or a (dwell_grid, eta) pair; got {type(efficiency).__name__}"
            )
        if arr.shape[1] != 2:
            arr = arr.T
        t, e = arr[:, 0], arr[:, 1]

    if t.shape != e.shape or t.size < 2:
        raise ValueError(f"efficiency grid and values must be matching 1-D arrays of length >= 2, got {t.shape} and {e.shape}")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(e))):
        raise ValueError("efficiency grid contains non-finite values")
    if np.any(t <= 0):
        raise ValueError("efficiency dwell grid must be positive")
    if np.any(e < 0) or np.any(e > 1):
        raise ValueError(f"efficiency must lie in [0, 1]; got range [{e.min():g}, {e.max():g}]")
    order = np.argsort(t)
    t, e = t[order], e[order]
    if np.all(e <= 0):
        raise ValueError("measured efficiency is zero everywhere; nothing could have been detected")
    return t, e


def _eta_at(t: np.ndarray, grid: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Piecewise-linear eta, held flat outside the measured grid."""
    return np.interp(t, grid, vals, left=vals[0], right=vals[-1])


# ---------------------------------------------------------------------------
# exponential mixture, left-truncated and/or efficiency-weighted
# ---------------------------------------------------------------------------


def _mix_pdf(t: np.ndarray, taus: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sum(w[:, None] / taus[:, None] * np.exp(-t[None, :] / taus[:, None]), axis=0)


def _mix_surv(t: float | np.ndarray, taus: np.ndarray, w: np.ndarray) -> np.ndarray:
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    return np.sum(w[:, None] * np.exp(-t[None, :] / taus[:, None]), axis=0)


class _Normalizer:
    """Z(theta) = integral_{t_min}^{t_max} eta(t) f(t) dt (``t_max = inf`` by default).

    Split at the top of the measured efficiency grid: below it eta is
    piecewise-linear and the integral is done numerically on a fixed dense grid;
    above it eta is flat and the tail is the exact analytic survival, so no
    quadrature error creeps into the long-tau limit.

    ``t_max`` is the OPTIONAL upper truncation point -- the longest dwell that
    could have been reported (``detect_events(max_dwell_s=...)`` de-accepts
    everything above it). With ``t_max is None`` every expression below reduces
    to the single-sided form this class has always computed, term for term, so
    the untruncated fit is numerically unchanged.
    """

    def __init__(self, t_min: float, grid: np.ndarray | None, vals: np.ndarray | None,
                 t_max: float | None = None):
        self.t_min = float(t_min)
        self.t_max = None if t_max is None else float(t_max)
        self.grid, self.vals = grid, vals
        if grid is None:
            self.T = t_min
            self.q_t = None
            self.eta_tail = 1.0
        else:
            self.T = float(max(grid[-1], t_min))
            if self.t_max is not None:
                # eta is only ever needed inside the observable window
                self.T = float(min(self.T, self.t_max))
            self.eta_tail = float(vals[-1])
            if self.T > t_min:
                dense = np.geomspace(t_min, self.T, 2048)
                nodes = grid[(grid > t_min) & (grid < self.T)]
                self.q_t = np.unique(np.concatenate([dense, nodes, [t_min, self.T]]))
                self.q_eta = _eta_at(self.q_t, grid, vals)
            else:
                self.q_t = None

    def __call__(self, taus: np.ndarray, w: np.ndarray) -> float:
        if self.t_max is None:
            tail = self.eta_tail * float(_mix_surv(self.T, taus, w)[0])
        else:
            # Above T eta is flat, so the band [T, t_max] contributes
            # eta_tail * (S(T) - S(t_max)); it is empty when T == t_max.
            s = _mix_surv(np.array([self.T, self.t_max], dtype=np.float64), taus, w)
            tail = self.eta_tail * float(s[0] - s[1])
        if self.q_t is None:
            return tail
        f = _mix_pdf(self.q_t, taus, w)
        return float(_trapz(self.q_eta * f, self.q_t)) + tail


def _unpack(theta: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    taus = np.exp(theta[:k])
    a = np.concatenate([[0.0], theta[k:]])
    a = a - a.max()
    w = np.exp(a)
    return taus, w / w.sum()


def _nll_factory(t: np.ndarray, k: int, norm: _Normalizer):
    n = t.size

    def nll(theta: np.ndarray) -> float:
        taus, w = _unpack(theta, k)
        f = _mix_pdf(t, taus, w)
        if not np.all(f > 0):
            return 1e12
        z = norm(taus, w)
        if not (z > 0) or not np.isfinite(z):
            return 1e12
        v = -float(np.sum(np.log(f))) + n * float(np.log(z))
        return v if np.isfinite(v) else 1e12

    return nll


def _init_thetas(t: np.ndarray, k: int, t_min: float, rng: np.random.Generator, n_restarts: int):
    span = max(float(np.mean(t)) - t_min, float(np.mean(t)) * 1e-3)
    thetas = []
    # quantile-split init: each component owns a slice of the observed range
    qs = np.quantile(t, np.linspace(0, 1, k + 1))
    taus0 = []
    for j in range(k):
        sel = t[(t >= qs[j]) & (t <= qs[j + 1])]
        taus0.append(max(float(np.mean(sel)) - t_min, span * 1e-2))
    thetas.append(np.concatenate([np.log(taus0), np.zeros(k - 1)]))
    # geometric spread init
    thetas.append(np.concatenate([np.log(span * np.geomspace(0.15, 6.0, k)), np.zeros(k - 1)]))
    for _ in range(n_restarts):
        lt = np.log(span) + rng.uniform(-2.5, 2.5, size=k)
        thetas.append(np.concatenate([lt, rng.normal(0, 1.0, size=k - 1)]))
    return thetas


def _fit_mix(t: np.ndarray, k: int, t_min: float, norm: _Normalizer, rng, n_restarts: int = 8,
             warm: np.ndarray | None = None):
    nll = _nll_factory(t, k, norm)
    # tau can be far below t.min() when truncation is aggressive (tau ~ mean(t)-t_min),
    # so the lower bound tracks that scale, not the smallest observed dwell
    scale = max(float(np.mean(t)) - t_min, float(np.ptp(t)) * 1e-3, 1e-15)
    lo, hi = np.log(scale * 1e-8), np.log(t.max() * 1e4)
    bounds = [(lo, hi)] * k + [(-25.0, 25.0)] * (k - 1)
    starts = [warm] if warm is not None else _init_thetas(t, k, t_min, rng, n_restarts)
    best, best_v = None, np.inf
    for th0 in starts:
        # A MULTI-START LOOP IS A LOOP, AND EVERY LOOP HERE POLLS. Deliberately
        # OUTSIDE the `except Exception` below: OperationCancelled is a RuntimeError,
        # so raising it inside that try would be swallowed as "this start failed" and
        # the cancel would be lost. See compare_dwell_models for the same trap.
        check_cancelled()
        try:
            r = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
        except Exception:
            continue
        if r.fun < best_v and np.all(np.isfinite(r.x)):
            best, best_v = r.x, float(r.fun)
    if best is None:
        raise RuntimeError("exponential mixture MLE failed to converge from any start")
    taus, w = _unpack(best, k)
    order = np.argsort(taus)  # identifiability: components ordered by tau
    return taus[order], w[order], -best_v, best


def _degenerate_pairs(values, rel_gap: float) -> list[tuple[int, int]]:
    """Indices of ADJACENT components whose values coincide to within ``rel_gap``.

    Adjacent in the SORTED order the fitters already impose (taus ascending, mixture
    means ascending), so a run of three identical components reports as two pairs and
    the count of collapsed components is ``len(pairs)`` -- which is what the message
    subtracts from k.

    The comparison is relative because these are scale-free quantities: a 20 ms and a
    20.0000001 ms dwell constant are the same component, and so are a 200 ns pair
    differing in their last bit. An absolute epsilon would be right for exactly one
    recording's units.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    out: list[tuple[int, int]] = []
    for i in range(v.size - 1):
        scale = max(abs(float(v[i])), abs(float(v[i + 1])))
        if scale <= 0:
            continue
        if abs(float(v[i + 1]) - float(v[i])) <= rel_gap * scale:
            out.append((i, i + 1))
    return out


def _pack_sorted(taus, w) -> np.ndarray:
    k = taus.size
    a = np.log(np.maximum(w, 1e-12))
    return np.concatenate([np.log(taus), (a[1:] - a[0])])


#: WHERE A RIGHT TRUNCATION CAME FROM, spelled the way ``DwellFitInputs.t_min_source``
#: spells the FLOOR's provenance, so one pane naming both bounds names them alike.
#:
#: ``t_max`` is a number; who imposed it is a separate fact, and the note that prints it
#: used to ASSERT one answer ("detector max_dwell_s") for every dwell fit. That is right
#: only when the detector's own ceiling is the bound the likelihood got. The Plots window
#: intersects it with a typed Fit range, and on the owner's runs the typed bound wins
#: routinely -- so the pane blamed a detector setting of 0.005 s for a truncation that
#: actually happened at the 0.004 s the user typed, with the real ``max_dwell_s``
#: appearing nowhere. Callers that KNOW pass one of these tokens; callers that do not
#: pass nothing and get the historical hedge, which claims nothing it cannot support.
T_MAX_FROM_DETECTOR = "max_dwell_s"
T_MAX_FROM_FIT_RANGE = "explicit"
T_MAX_FROM_CURATION = "curation"

#: The parenthetical each token licenses. The dwell wording ("detector max_dwell_s") is
#: kept for the detector token so a fit that really is bounded by the detector reads
#: exactly as it always has.
_T_MAX_SOURCE_CLAUSES: dict[str, str] = {
    T_MAX_FROM_DETECTOR: "detector max_dwell_s",
    T_MAX_FROM_FIT_RANGE: "the fit range that was typed, NOT a detector limit",
    T_MAX_FROM_CURATION: "curation removed everything above it, NOT a detector limit",
}


#: THE COLUMN NAMES THAT MEAN "THIS FIT IS A DWELL FIT".
#:
#: ``quantity=None`` is the historical dwell default of :func:`fit_dwell_exponential`
#: and always means a dwell. But a caller may also name the column explicitly, and a
#: duration column is not always spelled ``dwell_s``: the Plots window accepts
#: ``dwell``/``dwell_time_s`` as well (``gui.plots_panel._DWELL_COLUMNS``, "columns the
#: detector's own dwell provenance actually describes"), and ``_column_quantity``
#: answers unit ``""`` for the bare ``dwell`` spelling -- so that column reaches this
#: module as ``quantity="dwell"``, a dwell under a name ``quantity is None`` cannot see.
#:
#: Testing ``quantity is None`` for "is this a dwell" therefore mis-answers exactly the
#: aliases the window declares it supports: it refused the detector token outright for a
#: frame whose duration column is spelled ``dwell`` ("Fit unavailable" in the Details
#: pane, for a fit the detector really did bound), and gave the non-dwell hedge to the
#: same column when no token was passed.
#:
#: SAME NAMES AS THE WINDOW, AND ONLY ONE LIST SHOULD EXIST. The analysis layer is the
#: right owner (``gui`` already reads ``analysis`` for such lists -- see
#: ``plots_panel._ORIGIN_COLUMNS``, taken from ``scoring.RECORDING_CLOCK_COLUMNS`` with
#: the comment "a private copy is how two surfaces end up disagreeing"), and the window's
#: tuple should become ``tuple(stats.DWELL_QUANTITY_NAMES)``. Until it does, the two are
#: pinned equal by ``tests/gui/test_s3b_expl_dwell_alias_truncation.py`` so a name added
#: on either side cannot drift away in silence.
DWELL_QUANTITY_NAMES: tuple[str, ...] = ("dwell_s", "dwell", "dwell_time_s")


def _is_dwell_quantity(quantity: str | None) -> bool:
    """Whether ``quantity`` names a dwell -- the one place that decides it.

    Case-sensitive, and by exact name, because that is what the window does: a column
    it does not match against ``_DWELL_COLUMNS`` gets no dwell provenance at all, so a
    laxer test here would credit the detector for a bound it was never given.
    """
    return quantity is None or str(quantity) in DWELL_QUANTITY_NAMES


def truncation_source_clause(t_max_source: str | None, *, quantity: str | None = None) -> str:
    """The parenthetical naming who imposed ``t_max``.

    One writer, because the sentence has to mean the same thing wherever it is printed.
    ``None`` is the honest unknown and keeps the two historical hedges bit-for-bit: a
    dwell fit could be bounded by the detector or by curation, and a fit to any other
    column has no detector ceiling to be bounded by at all.

    "Is this a dwell?" is asked through :func:`_is_dwell_quantity`, not through
    ``quantity is None``: a caller that names the column it fitted must still get the
    dwell wording when that column IS a dwell (``dwell``/``dwell_time_s``, the aliases
    the Plots window accepts), and the non-dwell hedge only when it is not.

    Raises:
        ValueError: on a token this module does not define, and on
            :data:`T_MAX_FROM_DETECTOR` for a column that is not a dwell --
            ``max_dwell_s`` is a dwell setting, so "detector max_dwell_s" over a
            picoamp ceiling names a mechanism that cannot have produced it. That is
            the same wrong-column fault EXPL3-04 removed from the default wording,
            and an explicit token must not be able to put it back.
    """
    if t_max_source is None:
        return ("(detector max_dwell_s, or curation)" if _is_dwell_quantity(quantity)
                else "(a fit range, or curation)")
    key = str(t_max_source)
    if key not in _T_MAX_SOURCE_CLAUSES:
        raise ValueError(
            f"unknown t_max_source {t_max_source!r}; known: "
            f"{sorted(_T_MAX_SOURCE_CLAUSES)} (or None for 'not known')")
    if key == T_MAX_FROM_DETECTOR and not _is_dwell_quantity(quantity):
        raise ValueError(
            f"t_max_source={T_MAX_FROM_DETECTOR!r} claims the detector's dwell ceiling "
            f"bounded a fit to {quantity!r}, which is not a dwell column; max_dwell_s "
            f"cannot have imposed it")
    return f"({_T_MAX_SOURCE_CLAUSES[key]})"


def fit_dwell_exponential(
    dwell_s,
    *,
    t_min: float,
    t_max: float | None = None,
    t_max_source: str | None = None,
    n_components: int = 1,
    efficiency=None,
    n_boot: int = 1000,
    seed: int | None = 0,
    quantity: str | None = None,
    unit: str | None = None,
    display: tuple[float, str] | None = None,
) -> FitResult:
    """Fit dwell times with a truncated exponential (mixture) by censored MLE.

    Args:
        dwell_s: Observed event dwell times in **seconds** (already curated).
        t_min: Detection limit in seconds -- the shortest dwell your detector
            could have reported. Events below it are missing from ``dwell_s``,
            and the likelihood is conditioned on that. Getting this wrong
            propagates straight into tau.
        t_max: Optional UPPER limit in seconds -- the longest dwell that could
            have been reported. ``detect_events(max_dwell_s=...)`` de-accepts
            everything above it, which truncates the sample on the right exactly
            as ``t_min`` truncates it on the left, and biases tau **short**
            (measured -24% at ``t_max = 2.5 tau``). Passing it conditions the
            likelihood on ``t_min <= t <= t_max`` instead. ``None`` (the
            default) is the untruncated-right model and is numerically identical
            to this function's behaviour before ``t_max`` existed -- including
            the k=1 closed form and its vectorised bootstrap, which are used
            only on that path.
        t_max_source: WHO imposed ``t_max`` -- :data:`T_MAX_FROM_DETECTOR`,
            :data:`T_MAX_FROM_FIT_RANGE` or :data:`T_MAX_FROM_CURATION`. The
            MODELLED-truncation note names it. ``None`` (the default) means the
            caller does not know, and prints the hedge this function always
            printed ("detector max_dwell_s, or curation"); pass a token whenever
            you do know, because the hedge read as an assertion of the detector
            story on a bound the user had typed themselves. Only meaningful with
            ``t_max``; passing it without one raises.
        n_components: Number of exponential components. Compare BIC across
            values rather than assuming.
        efficiency: Optional measured detection efficiency eta(t) -- the
            DataFrame from ``nanoscope.core.truth.recovery_map``, a callable
            ``eta(t)``, or a ``(dwell_grid, eta)`` pair. When given, the hard
            ``eta == 1 above t_min`` step is replaced by the measured curve:
            ``L = prod eta(t_i) f(t_i) / integral eta(t) f(t) dt``.
        n_boot: Bootstrap replicates for the CIs.
        seed: RNG seed for the bootstrap (None for entropy).
        quantity: What the fitted column MEASURES, when it is not a dwell time --
            the same contract :func:`fit_depth_gaussians` has (report High-4).
            The GUI lets this family loose on ANY positive column, and with the
            default labelling a fit to ``depth_pA`` came back with ``tau`` stamped
            seconds: the pane printed "tau = 81.54 s" for a number that is
            picoamps, two lines under a note correctly saying no detection limit
            is known for that column. Pass it (with ``unit``) whenever the data
            are not seconds; it also names the model ("exponential (left-truncated
            MLE) on depth_pA") so a stored fit says what it was fitted to.
            ``None`` (the default) keeps the dwell reading exactly as before.
        unit: The unit ``tau_j`` (and the sample) are IN -- ``"pA"`` for
            ``depth_pA``, ``""`` for a ratio. Defaults to ``"s"`` when
            ``quantity`` is ``None`` (the dwell contract) and to ``quantity``
            otherwise. This is the STORAGE unit: ``params``, ``ci`` and
            ``meta['param_units']`` stay in it; readers convert at display time
            via :meth:`FitResult.unit_of`. Weights are shares and never take it.
        display: ``(factor, name)`` -- the unit the WORDS in ``meta['warnings']``
            are read in, and the multiplier from ``unit`` to it; ``None`` means
            "same as ``unit``". Same reason :func:`fit_depth_gaussians` grew it:
            the estimate line and the degeneracy note are written by different
            code, and the scale has to travel WITH the value or one pane prints
            the same tau in two units 1000x apart. Nothing numeric depends on it.

    Returns:
        FitResult with ``tau_j`` in seconds and ``w_j`` the **untruncated**
        mixing weights (not the observed fractions), plus AIC/BIC and
        percentile bootstrap CIs. ``.predict`` gives the observed (censored)
        density, which is what overlays a histogram of the data.

    Raises:
        ValueError: on fewer than 20 events, identical values, ``t_min <= 0``,
            ``t_max <= t_min``, any dwell shorter than ``t_min``, or any dwell
            longer than ``t_max`` (which means the stated limit is wrong or the
            detector is reporting events outside its own window -- either way
            the censoring model is a lie and the fit would be meaningless).

    The gotcha:
        Do **not** substitute a log-log histogram slope or ``scipy.stats.expon.fit``
        for this. Both ignore truncation and bias tau long by ~t_min, which is a
        tens-of-percent error at typical nanopore bandwidths and is invisible on
        the plot.
    """
    # THE STORAGE UNIT AND THE READING UNIT, decided once, before anything is worded --
    # the same rule (and the same variable names) as fit_depth_gaussians. The default is
    # the dwell contract: quantity None, unit "s", notes worded in the storage unit.
    # A refusal names what the caller actually fitted, not "dwell_s" regardless.
    quantity = None if quantity is None else str(quantity)
    unit = ("s" if quantity is None else quantity) if unit is None else str(unit)
    note_factor, note_unit = (1.0, unit) if display is None else (
        float(display[0]), str(display[1]))
    note_suffix = f" {note_unit}" if note_unit else ""
    # tau over seconds is a time constant; tau over anything else is a decay constant
    # of that axis, and calling it a time constant would be the exact lie the unit fix
    # exists to stop.
    tau_noun = "time constant" if unit == "s" else "decay constant"
    # THE TRUNCATION/CENSORING SENTENCES OBEY THE SAME CONTRACT AS THE PARAMETERS.
    # BL-3 stamped tau in the fitted column's unit, but the refusals and the
    # truncation notes below still hardcoded " s" and dwell nouns ("events longer
    # than it", "detector max_dwell_s") -- so a depth_pA fit with a fit-range max
    # printed "t_max=300 s" two lines above a FIT RANGE line correctly saying pA.
    # `unit_sfx` is the STORAGE-unit suffix (refusals and meta['censoring'] speak
    # raw stored values); the MODELLED note goes through (note_factor, note_suffix)
    # like the degeneracy note, so the sentences this function writes cannot
    # disagree with the estimate line. `quantity is None` keeps every dwell
    # sentence bit-for-bit.
    unit_sfx = f" {unit}" if unit else ""
    val_noun = "event" if quantity is None else "value"
    t = _clean_1d(dwell_s, quantity if quantity is not None else "dwell_s", positive=True)
    t_min = float(t_min)
    if not np.isfinite(t_min) or t_min <= 0:
        raise ValueError(
            f"t_min must be a positive duration in seconds, got {t_min!r}"
            if quantity is None else
            f"t_min must be a positive {quantity} bound"
            + (f" in {unit}" if unit else "") + f", got {t_min!r}")
    if t_max is not None:
        t_max = float(t_max)
        if not np.isfinite(t_max) or t_max <= t_min:
            raise ValueError(
                f"t_max must be a finite duration greater than t_min={t_min:g} s, got {t_max!r}"
                if quantity is None else
                f"t_max must be a finite {quantity} bound greater than "
                f"t_min={t_min:g}{unit_sfx}, got {t_max!r}"
            )
        if t.max() > t_max * (1 + 1e-9):
            n_over = int((t > t_max * (1 + 1e-9)).sum())
            raise ValueError(
                (f"{n_over} event(s) are longer than the stated "
                 f"upper limit t_max={t_max:g} s (max dwell = {t.max():g} s). Either t_max is "
                 f"wrong or those events were never truncated away. "
                 if quantity is None else
                 f"{n_over} {quantity} value(s) are above the stated "
                 f"upper limit t_max={t_max:g}{unit_sfx} (max {quantity} = "
                 f"{t.max():g}{unit_sfx}). Either t_max is wrong or those values were "
                 f"never truncated away. ")
                + "Refusing to fit: the right-truncation correction would be nonsense."
            )
    # THE LABEL IS CHECKED WHERE THE BOUND IS, not where the sentence is written: a
    # misspelt provenance must fail on every call, not only on the branch that happens
    # to print it, and a provenance for a bound that was never given describes nothing.
    if t_max_source is not None:
        if t_max is None:
            raise ValueError(
                f"t_max_source={t_max_source!r} says where an upper truncation came from, "
                f"but no t_max was given; there is no bound for it to describe")
        truncation_source_clause(t_max_source, quantity=quantity)   # validates the token
    if int(n_components) < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    k = int(n_components)
    # tolerance is for float round-trips of n_samples/fs, not a real allowance
    if t.min() < t_min * (1 - 1e-9):
        n_under = int((t < t_min * (1 - 1e-9)).sum())
        raise ValueError(
            (f"{n_under} event(s) are shorter than the stated "
             f"detection limit t_min={t_min:g} s (min dwell = {t.min():g} s). Either t_min "
             f"is wrong or the detector is emitting events below its own resolution. "
             if quantity is None else
             f"{n_under} {quantity} value(s) are below the stated "
             f"lower limit t_min={t_min:g}{unit_sfx} (min {quantity} = "
             f"{t.min():g}{unit_sfx}). Either t_min is wrong or values below the stated "
             f"floor were reported anyway. ")
            + "Refusing to fit: the truncation correction would be nonsense."
        )
    if t.size < _MIN_EVENTS * k:
        # NAMES WHAT WAS FITTED, like every other refusal in this function: the
        # window offers this family on any positive column, and "20 events is too
        # few" under a histogram of rise_s says nothing about which 20.
        raise ValueError(
            f"{t.size} {val_noun}s of "
            f"{'dwell_s' if quantity is None else quantity} is too few for a "
            f"{k}-component mixture (need >= {_MIN_EVENTS * k}); the extra components "
            f"would fit noise"
        )

    notes: list[str] = []
    rng = np.random.default_rng(seed)

    grid = vals = None
    if efficiency is not None:
        grid, vals = _resolve_efficiency(efficiency, t_min, float(t.max()), notes)
        if grid[0] > t_min * (1 + 1e-9):
            notes.append(
                f"efficiency grid starts at {grid[0]:g} s, above t_min={t_min:g} s; "
                f"eta is held flat at {vals[0]:.3g} below it (extrapolation)"
            )
        if vals[0] > 0.05 and grid[0] <= t_min * (1 + 1e-9):
            notes.append(
                f"efficiency is already {vals[0]:.3g} at the bottom of the measured grid; "
                f"the recovery map does not reach the detection floor, so short-dwell "
                f"efficiency is extrapolated, not measured"
            )
        eta_i = _eta_at(t, grid, vals)
        if np.any(eta_i <= 0):
            notes.append(
                f"{int((eta_i <= 0).sum())} observed event(s) fall where the measured "
                f"efficiency is 0 -- the recovery map and the detector disagree"
            )
        const = float(np.sum(np.log(np.maximum(eta_i, _ETA_FLOOR))))
        censoring = f"measured eta(t) on [{grid[0]:g}, {grid[-1]:g}] s, truncated at t_min={t_min:g} s"
        if t_max is not None:
            censoring += f" and at t_max={t_max:g} s (right-truncated)"
    else:
        const = 0.0
        # meta['censoring'] speaks the STORAGE unit (`unit_sfx`), like the stored
        # t_min/t_max beside it; for the dwell default that is " s", bit-for-bit.
        censoring = (
            f"hard left-truncation at t_min={t_min:g}{unit_sfx} (eta == 1 above it)"
            if t_max is None else
            f"DOUBLE truncation, t_min={t_min:g}{unit_sfx} <= t <= t_max={t_max:g}{unit_sfx} "
            f"(eta == 1 between them); NOT left-truncation only"
        )

    if t_max is not None:
        # Worded through (note_factor, note_suffix) like the degeneracy note below:
        # the bound must print in the same unit as the FIT RANGE line and the
        # estimate line on the pane that shows it. The provenance clause is the
        # dwell story ("detector max_dwell_s") only when this IS a dwell fit; a
        # generic column's ceiling can only come from a fit range or curation, and
        # naming a detector setting that never existed would be a false provenance.
        #
        # ...AND EVEN ON A DWELL FIT THE HEDGE WAS READ AS AN ASSERTION. `t_max_source`
        # is the caller's answer to "who imposed this bound", worded once in
        # :func:`truncation_source_clause`: with a typed Fit range max the pane no
        # longer blames a `max_dwell_s` that is a different number and is printed
        # nowhere. `None` keeps both historical hedges bit-for-bit.
        notes.append(
            f"right truncation at t_max={t_max * note_factor:g}{note_suffix} is MODELLED: "
            + ("events longer than it were never reported "
               if quantity is None else
               f"{quantity} values above it were never reported ")
            + truncation_source_clause(t_max_source, quantity=quantity)
            + f", so the likelihood is conditioned on t_min <= t <= t_max. Fitting these "
            f"{val_noun}s as if only the left tail were missing biases tau SHORT (measured "
            f"-3.7% at t_max = 5 tau, -24% at t_max = 2.5 tau)."
        )

    norm = _Normalizer(t_min, grid, vals, t_max)

    # The k=1 closed form mean(t)-t_min is the exact MLE only for the
    # left-truncated-only likelihood; a right limit makes it wrong, so that path
    # (and its vectorised bootstrap) is reserved for the untruncated case.
    use_closed_form = k == 1 and efficiency is None and t_max is None

    if use_closed_form:
        tau_cf = float(np.mean(t)) - t_min
        if tau_cf <= 0:
            # The refusal names what was actually averaged: "mean dwell ... s" for a
            # picoamp sample is the same seconds-for-picoamps lie the notes above fix.
            raise ValueError(
                (f"mean dwell ({np.mean(t):g} s) does not exceed t_min ({t_min:g} s); "
                 if quantity is None else
                 f"mean {quantity} ({np.mean(t):g}{unit_sfx}) does not exceed t_min "
                 f"({t_min:g}{unit_sfx}); ")
                + "the truncated-exponential MLE is degenerate. t_min is probably wrong."
            )
        taus_n, _, _, _ = _fit_mix(t, 1, t_min, norm, rng)
        # the closed form is the exact MLE here; the optimiser must agree or the
        # numerical machinery used for k>1 is not trustworthy either
        if abs(taus_n[0] - tau_cf) > 1e-3 * tau_cf:
            raise RuntimeError(
                f"numerical MLE ({taus_n[0]:.6g}) disagrees with the closed-form truncated "
                f"MLE mean(t)-t_min ({tau_cf:.6g}); the likelihood is misspecified"
            )
        taus, w = np.array([tau_cf]), np.array([1.0])
        warm = _pack_sorted(taus, w)
    else:
        taus, w, _, warm = _fit_mix(t, k, t_min, norm, rng)

    ll = float(np.sum(np.log(_mix_pdf(t, taus, w)))) - t.size * float(np.log(norm(taus, w))) + const
    n_par = 2 * k - 1
    aic = 2 * n_par - 2 * ll
    bic = n_par * np.log(t.size) - 2 * ll

    params: dict[str, float] = {}
    for j in range(k):
        params[f"tau_{j + 1}"] = float(taus[j])
    if k > 1:
        for j in range(k):
            params[f"w_{j + 1}"] = float(w[j])

    # -- degeneracy (Low-20) -----------------------------------------------
    # A k-component mixture whose taus coincide is a (k-1)-component solution with
    # k sets of parameters, and it prints as k time constants. Said here, in the
    # fitter, because BOTH panels print these params and neither can tell.
    degenerate = _degenerate_pairs(taus, _DEGENERATE_REL_GAP)
    light = [j for j in range(k) if k > 1 and w[j] * t.size < _MIN_COMPONENT_EVENTS]
    if degenerate:
        # Worded through (note_factor, note_suffix) like every number the depth mixture
        # prints: the estimate line converts tau into the axis's unit, and a note quoting
        # the same tau in raw storage units two lines below is the one-pane-two-units
        # defect all over again.
        pairs = "; ".join(
            f"tau_{i + 1} = {taus[i] * note_factor:.6g}{note_suffix} and "
            f"tau_{j + 1} = {taus[j] * note_factor:.6g}{note_suffix}"
            for i, j in degenerate)
        notes.insert(0, (
            f"DEGENERATE MIXTURE: {pairs} — these components have the same {tau_noun} to "
            f"within {_DEGENERATE_REL_GAP * 100:g}%, so this k={k} solution is really a "
            f"k={k - len(degenerate)} solution carrying {2 * k - 1} parameters. Do NOT report "
            f"them as separate populations; refit with fewer components (BIC will agree)."
        ))
    if light:
        notes.append(
            "component(s) " + ", ".join(f"{j + 1} (w = {w[j]:.3g}, ~{w[j] * t.size:.0f} events)"
                                        for j in light)
            + f" are supported by fewer than {_MIN_COMPONENT_EVENTS} events — the minimum this "
              f"module will fit a whole sample with. Treat them as unresolved, not as a "
              f"measured population.")

    # -- bootstrap ---------------------------------------------------------
    boot = _bootstrap_dwell(t, k, t_min, norm, use_closed_form, n_boot, rng, warm)
    ci: dict[str, tuple[float, float]] = {}
    if boot is not None and boot.shape[0] >= 20:
        qs = np.percentile(boot, [2.5, 97.5], axis=0)
        for i, key in enumerate(params):
            ci[key] = (float(qs[0, i]), float(qs[1, i]))
    elif ci_wanted():
        # Declined is not failed: inside `point_fits_only` the caller asked for a
        # point fit, and saying the bootstrap "failed" would be a false report.
        notes.append("bootstrap failed to produce enough replicates; CIs unavailable")

    if k > 1:
        name = f"exponential mixture (k={k}, censored MLE)"
    else:
        name = ("exponential (left-truncated MLE)" if t_max is None
                else "exponential (doubly-truncated MLE)")
    if quantity is not None:
        # What was fitted travels with the fit, exactly as the depth mixture's model
        # line does ("Gaussian mixture on depth_pA"): a stored exponential over
        # picoamps must not read back as an anonymous dwell fit.
        name += f" on {quantity}"
    meta = {
        # `family` and the two `eta_*` keys are v6 PERSISTENCE state, not diagnostics:
        # together with `params` they are exactly what density_from() needs to rebuild the
        # curve. eta used to live only in this function's closure, which is precisely why a
        # dwell fit could be stored as numbers but never redrawn. See density_from().
        "family": "exp-mixture",
        "eta_grid": None if grid is None else [float(g) for g in grid],
        "eta_vals": None if vals is None else [float(v) for v in vals],
        "censoring": censoring, "t_min": t_min, "t_max": t_max, "n_components": k,
        # WHO IMPOSED THE CEILING, stored beside it exactly as `t_min_source` is stored
        # beside the floor (populations.fit_exponential / modules.stamp_fit_provenance).
        # A fit read back out of a project has to be able to say which of its two bounds
        # was the detector's and which was the analyst's; None means "not recorded".
        "t_max_source": None if t_max_source is None else str(t_max_source),
        "n_boot_ok": 0 if boot is None else int(boot.shape[0]),
        # THE REPLICATES THEMSELVES, kept rather than reduced to two percentiles and
        # dropped: a confidence BAND is the pointwise envelope of the replicate curves
        # and cannot be rebuilt from the per-parameter intervals below (see ci_band()).
        # Columns are in `params` order, which is the order the intervals were read off
        # `boot` in, so the two can never disagree about which column is which.
        # `fit_to_json` drops both keys -- the memory is noise, the persistence is not.
        "boot": None if boot is None else np.asarray(boot, dtype=np.float64),
        "boot_params": list(params),
        "closed_form": bool(use_closed_form),
        "right_truncated": bool(t_max is not None),
        "mean_dwell_observed_s": float(np.mean(t)),
        # What was fitted, when the caller said (None = the dwell default). See
        # fit_depth_gaussians, which stamps the same key for the same reason.
        "quantity": quantity,
        # Per-parameter units, so a window can caption tau without guessing. See
        # FitResult.unit_of(). `unit` is the STORAGE unit of the sample -- "s" for
        # dwell times, the column's own unit otherwise; a weight is a share and
        # never takes one.
        "param_units": {p: (unit if p.startswith("tau") else "") for p in params},
        "degenerate_pairs": [[int(i) + 1, int(j) + 1] for i, j in degenerate],
        "degenerate": bool(degenerate),
        "warnings": notes,
    }
    return _stamp_gof(FitResult(
        params=params, ci=ci, loglik=ll, aic=float(aic), bic=float(bic), n=int(t.size),
        model=name, meta=meta,
        _pdf=density_from("exp-mixture", params, meta),
    ), t)


def _bootstrap_dwell(t, k, t_min, norm, closed_form, n_boot, rng, warm):
    n = t.size
    if n_boot is None or n_boot < 1 or not ci_wanted():
        return None
    if closed_form:
        # closed form -> the whole bootstrap is a resampled mean; chunk the index
        # draw so a large n * n_boot never materialises as one array
        out = np.empty(n_boot)
        step = max(1, int(2e7 // max(n, 1)))
        for i in range(0, n_boot, step):
            # EVERY loop in this module polls, including the one no worker runs today:
            # the k=1 closed form is reached only from the GUI thread (where the token
            # is absent and this is a no-op) because the rival scans set n_boot = 0 --
            # but "which loops can a worker reach" is the kind of fact that changes
            # under a later edit, and a poll costs one predicate call per 20 M-element
            # chunk. The rule is the invariant; the current call graph is not.
            check_cancelled()
            m = min(step, n_boot - i)
            idx = rng.integers(0, n, size=(m, n))
            out[i:i + m] = t[idx].mean(axis=1) - t_min
        return out[:, None]
    rows = []
    n_par = k + (k if k > 1 else 0)
    for _ in range(n_boot):
        # Polled per replicate, and the `except Exception` below re-raises the token:
        # `_fit_mix` polls too, and a cancel arriving inside it would otherwise be
        # read as "this replicate failed to converge" -- silently DROPPING replicates
        # from a CI that then gets reported as if it had all of them.
        check_cancelled()
        s = t[rng.integers(0, n, size=n)]
        try:
            tb, wb, _, _ = _fit_mix(s, k, t_min, norm, rng, warm=warm)
        except OperationCancelled:
            raise
        except Exception:
            continue
        rows.append(np.concatenate([tb, wb]) if k > 1 else tb)
    if not rows:
        return None
    return np.asarray(rows)[:, :n_par]


# ---------------------------------------------------------------------------
# Gaussian mixture on blockade depth
# ---------------------------------------------------------------------------


def _norm_pdf(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = (x[None, :] - mu[:, None]) / sd[:, None]
    return np.exp(-0.5 * z * z) / (sd[:, None] * np.sqrt(2 * np.pi))


def _normal_cdf(z: float) -> float:
    """Phi(z), via erfc -- no scipy.stats import for one number."""
    return float(0.5 * math.erfc(-float(z) / math.sqrt(2.0)))


def _em_gauss(x: np.ndarray, k: int, rng, *, n_iter: int = 500, tol: float = 1e-10,
              init: tuple | None = None):
    n = x.size
    sd_floor = max(float(np.std(x)) * 1e-3, 1e-12)
    if init is None:
        mu = np.quantile(x, (np.arange(k) + 0.5) / k) + rng.normal(0, np.std(x) * 0.05, k)
        sd = np.full(k, max(float(np.std(x)) / k, sd_floor))
        w = np.full(k, 1.0 / k)
    else:
        mu, sd, w = (np.array(v, dtype=float) for v in init)
    ll_old = -np.inf
    for _ in range(n_iter):
        comp = w[:, None] * _norm_pdf(x, mu, sd)
        tot = comp.sum(axis=0)
        tot = np.maximum(tot, 1e-300)
        r = comp / tot
        nk = r.sum(axis=1)
        if np.any(nk < 1e-8):  # a component died; restart it on a random point
            dead = nk < 1e-8
            mu[dead] = rng.choice(x, size=int(dead.sum()))
            sd[dead] = max(float(np.std(x)) / k, sd_floor)
            w[dead] = 1.0 / k
            continue
        w = nk / n
        mu = (r @ x) / nk
        sd = np.sqrt(np.maximum((r * (x[None, :] - mu[:, None]) ** 2).sum(axis=1) / nk, sd_floor**2))
        ll = float(np.sum(np.log(tot)))
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            ll_old = ll
            break
        ll_old = ll
    comp = w[:, None] * _norm_pdf(x, mu, sd)
    ll_old = float(np.sum(np.log(np.maximum(comp.sum(axis=0), 1e-300))))
    o = np.argsort(mu)  # identifiability: components ordered by mean
    return mu[o], sd[o], w[o], ll_old


def _split_ladder_init(x: np.ndarray, k: int, rng) -> tuple | None:
    """A DETERMINISTIC start for k-component EM: fit 1, split, refit, ... up to k.

    WHY THIS EXISTS (report Medium-12). The random-quantile starts below land in a
    local optimum often enough to make BIC non-monotonic in k, which reads to a user
    as "two Gaussians describe this worse than one" -- an impossible statement about
    nested models, and one the panel then repeats. Measured on the owner's -500 mV
    apo-ferritin ``depth_pA`` (n = 863): ten random starts at k=2 returned
    loglik -3668.45 (mu 65.2 / 81.6 pA, BIC 7370.71) against 7365.41 at k=1. Splitting
    the k=1 solution and re-running EM reaches loglik **-3615.90** (mu 55.1 / 70.3 pA,
    BIC 7265.59) -- a likelihood 52.5 higher, so the old answer was provably not the
    MLE -- and the BIC ladder becomes monotone: 7365.41, 7265.59, 7198.43, 7170.43.

    The component split is the widest-by-mass one (largest ``w*sigma``), halved in
    place at ``mu +/- sigma/2``: the standard split-and-merge move, and the one that
    cannot miss a mode the previous level was averaging over.

    ``rng`` is a PRIVATE generator (see the call site). This must not consume the
    stream the random restarts draw from, or every existing fit's numbers would move.
    """
    if k < 2 or x.size < _MIN_EVENTS * k:
        return None
    mu = np.array([float(np.mean(x))])
    sd = np.array([max(float(np.std(x, ddof=0)), 1e-12)])
    w = np.array([1.0])
    for kk in range(2, k + 1):
        j = int(np.argmax(w * sd))
        m_j, s_j, w_j = float(mu[j]), float(sd[j]), float(w[j])
        # Read the three scalars BEFORE growing the arrays: after the insert, index
        # j+1 is the original component and j is its new twin.
        mu = np.insert(mu, j, m_j - s_j / 2.0)
        mu[j + 1] = m_j + s_j / 2.0
        sd = np.insert(sd, j, s_j / 2.0)
        sd[j + 1] = s_j / 2.0
        w = np.insert(w, j, w_j / 2.0)
        w[j + 1] = w_j / 2.0
        mu, sd, w, _ = _em_gauss(x, kk, rng, init=(mu, sd, w))
    return (mu, sd, w)


def _identifiability_notes(mu, sd, w, ci: dict, n: int, k: int, unit: str,
                           factor: float = 1.0) -> list[str]:
    """The components this fit did NOT resolve, in words (report Medium-11).

    Three independent failures, each stated as a condition the reader can check:

    1. **The mean is not pinned** -- its 95% bootstrap CI is wider than 2*sigma, i.e.
       the fit is less sure where the component sits than the component is wide.
       (-700 mV depth_pA k=3: mu_1 = 85.5 pA, CI [17.6, 87.6], sigma_1 = 7.8 pA.)
    2. **The weight is not pinned** -- its CI is wider than the weight itself, so the
       component's share of the sample is unresolved. (Same fit: w_2 = 0.027 with a
       CI of [0.014, 0.671].)
    3. **Two components are not separated** -- their means differ by less than the
       NARROWER one's own sigma, so they are one population split in two. (-500 mV
       k=3: mu = 55.70 and 56.92 pA with sigma_1 = 5.30.)

    A component too light to be a sample in its own right (< ``_MIN_COMPONENT_EVENTS``
    by weight) is reported by the caller, which knows n.

    THE SCALE TRAVELS WITH THE VALUE. ``unit``/``factor`` are the unit these numbers
    are to be READ in and the multiplier that gets them there, and every quantity in
    that unit -- mu, 2*sigma, and the CI BOUNDS -- goes through both. They are one
    argument pair rather than two decisions because the alternative is what shipped:
    the caller scaled the estimate line into milliseconds and this function composed
    the note from the raw seconds it was handed, so one pane printed

        mu = 0.1923, 1.923, 2.314 ms
        mu_3 = 0.002314 s has a 95% CI of [0.002127, 0.00256]

    two lines apart -- the same parameter 1000x apart, and the interval inside the
    note wearing no unit at all while the value beside it said "s". A formatter that
    receives scaled values and a formatter that receives raw ones cannot both be
    right; the only fix that stays fixed is for the scale to arrive WITH the numbers.

    ``factor`` defaults to 1.0 and ``unit`` keeps its old meaning, so a caller that
    knows nothing about display units (``modules.fit_depth``, the Fits panel, a
    notebook) gets exactly the strings it got before.
    """
    notes: list[str] = []
    suffix = f" {unit}" if unit else ""
    f = float(factor)
    unpinned: list[str] = []
    for j in range(k):
        lo, hi = ci.get(f"mu_{j + 1}", (np.nan, np.nan))
        if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > 2.0 * float(sd[j]):
            unpinned.append(
                f"mu_{j + 1} = {mu[j] * f:.4g}{suffix} has a 95% CI of "
                f"[{lo * f:.4g}, {hi * f:.4g}]{suffix} — "
                f"wider than 2*sigma_{j + 1} ({2 * float(sd[j]) * f:.4g}{suffix})")
    if unpinned:
        notes.append("UNIDENTIFIED COMPONENT(S): " + "; ".join(unpinned)
                     + ". The fit does not know where these components are.")
    soft: list[str] = []
    for j in range(k):
        lo, hi = ci.get(f"w_{j + 1}", (np.nan, np.nan))
        if np.isfinite(lo) and np.isfinite(hi) and (hi - lo) > float(w[j]):
            # A weight is a SHARE. It has no unit and must never take one, whatever
            # the fitted column is in -- see `param_units`, which says the same thing
            # on the other side of the fit.
            soft.append(f"w_{j + 1} = {w[j]:.3g} with a 95% CI of [{lo:.3g}, {hi:.3g}]")
    if soft:
        notes.append("UNRESOLVED WEIGHT(S): " + "; ".join(soft)
                     + ". Each interval is wider than the estimate itself, so these "
                       "components' shares of the sample are not measured.")
    close = [
        (j, j + 1) for j in range(k - 1)
        if abs(float(mu[j + 1]) - float(mu[j])) < min(float(sd[j]), float(sd[j + 1]))
    ]
    if close:
        notes.append(
            "COMPONENTS NOT SEPARATED: " + "; ".join(
                f"mu_{i + 1} = {mu[i] * f:.4g}{suffix} and mu_{j + 1} = {mu[j] * f:.4g}"
                f"{suffix} differ by less than the narrower component's own sigma "
                f"({min(float(sd[i]), float(sd[j])) * f:.4g}{suffix})" for i, j in close)
            + ". These are one population reported twice, not two populations.")
    return notes


def fit_depth_gaussians(frac_depth, *, n_components: int = 1, depth_floor: float | None = None,
                        n_boot: int = 400, seed: int | None = 0,
                        quantity: str | None = None, unit: str | None = None,
                        display: tuple[float, str] | None = None) -> FitResult:
    """Fit a Gaussian mixture to blockade depths by EM, to resolve populations.

    Args:
        frac_depth: Blockade depths, always positive magnitudes -- see the sign
            convention in ``core.events``. Dimensionless ``frac_depth`` is the
            default reading; pass ``quantity`` for anything else.
        n_components: Number of populations. Use BIC across values to choose;
            more components always fit better in-sample.
        depth_floor: The detector's effective depth floor in the SAME units as
            ``frac_depth``. The trigger defines an event as a run of
            ``dev > k_out*noise`` and ``depth_pA`` is the MEDIAN in-event
            deviation, so the sample is truncated at roughly ``thr_out``, NOT at
            ``thr_in``. In fractional units that has to be divided by the same
            baseline ``frac_depth`` was divided by -- each event's OWN
            ``baseline_pA``, never the recording-wide ``baseline_median_pA``; let
            :func:`nanoscope.analysis.modules.depth_floor_from_provenance` work it
            out. When the shallowest fitted component
            puts more than 1% of its mass below that floor -- the audit's
            ``mu_1 - 2*sigma_1 < floor`` rule, widened because truncation shrinks
            sigma and would otherwise hide itself -- a loud entry is added to
            ``meta['warnings']``: the mixture here is UNtruncated, so a
            population sitting on the floor comes back with its mean biased high
            and its sigma biased low. ``None`` (the default) skips the check and
            leaves the fit numerically unchanged.
        n_boot: Bootstrap replicates for the CIs.
        seed: RNG seed for EM restarts and the bootstrap.
        quantity: What the fitted column MEASURES -- ``"depth_pA"``, ``"max_dev_pA"``,
            ``"dI/I0"``. It names the model (``"Gaussian mixture on depth_pA (k=3,
            EM)"``) and supplies the unit every ``mu_j``/``sigma_j`` is reported in,
            via :meth:`FitResult.unit_of`. Every refusal this function raises
            (empty, non-finite, too few events, all-identical, negative depths)
            quotes the same name, so a refusal cannot blame a column the caller
            never fitted; the dimensionless default reads back as ``frac_depth``,
            the column it stands for. Defaults to
            :data:`DEFAULT_DEPTH_QUANTITY`. **Pass it whenever the column is not
            ``frac_depth``.** The label used to be hardcoded to dI/I0, so fitting
            ``depth_pA`` returned "mu = 85.46, 99.69, 105.8" -- picoamps -- under a
            dimensionless caption, on a dataset whose real dI/I0 is 1.81-2.57
            (report High-4).
        unit: The unit ``mu_j``/``sigma_j`` are IN -- ``"pA"`` for ``depth_pA``, ``""``
            for a ratio. Separate from ``quantity`` because the two answer different
            questions ("what was fitted" names the model; "in what unit" captions the
            numbers) and a column name is not a unit. Defaults to ``""`` for the
            dimensionless default quantity and to ``quantity`` otherwise.

            THIS IS THE STORAGE UNIT, and it stays that. ``params``, ``ci`` and
            ``meta['param_units']`` are all in it, which is what lets a reader
            (``FitResult.unit_of``, the Plots pane, the Inspector) convert once, at the
            point of display, instead of guessing.
        display: ``(factor, name)`` -- the unit the WORDS in ``meta['warnings']`` should
            be read in, and the multiplier from ``unit`` to it. ``None`` (the default)
            means "same as ``unit``", so every existing caller composes exactly the
            strings it composed before.

            It exists because the estimate line and the identifiability notes are
            written by two different pieces of code: a window that shows seconds as
            milliseconds scaled the estimate and left the note in seconds, and the two
            then disagreed by a factor of 1000 two lines apart in ONE pane. The scale
            travels with the value so they cannot. Nothing NUMERIC depends on it: the
            fit, the CIs and ``param_units`` are identical whatever is passed here.

    Returns:
        FitResult with ``mu_j``, ``sigma_j``, ``w_j`` per component, plus AIC/BIC
        and percentile bootstrap CIs. ``.predict`` gives the mixture density.
        ``meta['warnings']`` names every component the fit did not resolve, and
        ``meta['local_optimum']`` says whether the EM solution was beaten by the
        one-fewer-component fit (which is impossible for a true MLE of a nested
        model, and therefore proof the optimiser stopped early).

    Raises:
        ValueError: on fewer than 20 events, all-identical values, or negative
            depths (which mean the sign convention was violated upstream).

    The gotcha:
        Depth is truncated at the detector's own threshold just as dwell is
        truncated at t_min -- a mode sitting near that threshold is partly cut
        off, so its weight is underestimated and its mean pushed high. Measured
        through the real detector (fs 250 kHz, planted mu=50 sigma=20 pA,
        k_in=5, k_out=2): mu_1 came back +7% and sigma_1 -22%. Pass
        ``depth_floor`` and the fit will say so instead of leaving you to
        check ``mu_1 - 2*sigma_1`` by hand.
    """
    quantity = DEFAULT_DEPTH_QUANTITY if quantity is None else str(quantity)
    # THE REFUSAL NAMES WHAT THE CALLER FITTED. `quantity` is the same name the
    # mu/sigma captions carry (report High-4), so a refusal and a fit can never
    # disagree about which column was on the axis. The name used to be hardcoded
    # to "frac_depth", so refusing a 3-event depth_pA histogram told the user
    # "frac_depth has only 3 events" -- a column they never selected and that may
    # not even be in their frame (beta EXPL-2). The dimensionless default names
    # the READING of the ``frac_depth`` column, so it maps back to that column's
    # name; every other ``quantity`` already IS the fitted column's name.
    fitted_name = "frac_depth" if quantity == DEFAULT_DEPTH_QUANTITY else quantity
    x = _clean_1d(frac_depth, fitted_name, positive=False)
    if np.any(x < 0):
        raise ValueError(
            f"{int((x < 0).sum())} negative {fitted_name} value(s). Depths are positive "
            f"magnitudes by the sign convention (dev = polarity * (current - baseline)); a "
            f"negative depth means polarity was applied wrong upstream."
        )
    if int(n_components) < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    k = int(n_components)
    if x.size < _MIN_EVENTS * k:
        # `fitted_name` for the same reason the too-few-events refusal above uses it:
        # a refusal names the column the caller actually handed in.
        raise ValueError(
            f"{x.size} values of {fitted_name} is too few for a {k}-component mixture "
            f"(need >= {_MIN_EVENTS * k})"
        )

    rng = np.random.default_rng(seed)
    notes: list[str] = []
    # (`quantity` was normalised above, before the refusals that quote it.)
    # THE STORAGE UNIT AND THE READING UNIT, decided ONCE, here, before anything is
    # worded. Every note this function writes is composed through this pair, so a
    # pane cannot show one line in seconds and the next in milliseconds.
    unit = ("" if quantity == DEFAULT_DEPTH_QUANTITY else quantity) if unit is None else str(unit)
    note_factor, note_unit = (1.0, unit) if display is None else (
        float(display[0]), str(display[1]))
    note_suffix = f" {note_unit}" if note_unit else ""

    ll_one = float(np.sum(np.log(np.maximum(
        _norm_pdf(x, np.array([float(np.mean(x))]),
                  np.array([max(float(np.std(x, ddof=0)), 1e-300)]))[0], 1e-300))))
    local_optimum = False
    ladder_won = False
    if k == 1:
        mu = np.array([float(np.mean(x))])
        sd = np.array([float(np.std(x, ddof=0))])
        w = np.array([1.0])
        ll = ll_one
    else:
        # THE TEN RANDOM RESTARTS RUN FIRST, ON THE UNTOUCHED STREAM. Everything
        # below only ever ADDS a candidate and keeps the better likelihood, so no
        # existing fit can come back worse -- and any fit the random starts already
        # won stays bit-identical.
        best = None
        for _ in range(10):
            check_cancelled()       # a multi-start loop is a loop; see _fit_mix
            r = _em_gauss(x, k, rng)
            if best is None or r[3] > best[3]:
                best = r
        # ... then one deterministic split-ladder start, on a PRIVATE generator, to
        # escape the local optima that made BIC non-monotonic in k (Medium-12).
        # `ladder_used` is recorded in meta rather than left to inference: a start
        # that silently stopped working would put the old local optima back with
        # nothing on screen to say so.
        init = _split_ladder_init(x, k, np.random.default_rng(seed))
        if init is not None:
            r = _em_gauss(x, k, np.random.default_rng(seed), init=init)
            if np.isfinite(r[3]) and r[3] > best[3]:
                best = r
                ladder_won = True
        mu, sd, w, ll = best
        # A k-component mixture CONTAINS every (k-1)-component one, so its maximised
        # likelihood cannot be lower. If it is, EM stopped in a local optimum and the
        # BIC printed beside it is not this model's BIC. Say so rather than let the
        # panel report "3 components fit worse than 1".
        if ll < ll_one - 1e-9:
            local_optimum = True
            notes.insert(0, (
                f"LOCAL OPTIMUM: this k={k} solution has loglik {ll:.6f}, BELOW the k=1 fit's "
                f"{ll_one:.6f}. A k-component mixture contains every 1-component one, so a true "
                f"maximum cannot be worse — EM stopped early. The parameters and the BIC below "
                f"describe where it stopped, not the best k={k} model. Re-run with a different "
                f"seed, or read this k as unsupported."))

    n_par = 3 * k - 1
    aic = 2 * n_par - 2 * ll
    bic = n_par * np.log(x.size) - 2 * ll

    params: dict[str, float] = {}
    for j in range(k):
        params[f"mu_{j + 1}"] = float(mu[j])
        params[f"sigma_{j + 1}"] = float(sd[j])
        if k > 1:
            params[f"w_{j + 1}"] = float(w[j])

    # -- the floor check (B4b) --------------------------------------------
    # The mixture above is UNtruncated. The detector's own definition of an
    # event (a run of dev > k_out*noise) truncates the depth sample from below,
    # so a component whose lower tail crosses that floor is estimated from a
    # sample that is missing its own left side. Say so; do not let the user read
    # a clean mu/sigma off a cut-off population.
    floor_edge = float(mu[0] - 2.0 * sd[0])
    floor = mass_below = None
    if depth_floor is not None and np.isfinite(float(depth_floor)) and float(depth_floor) > 0:
        floor = float(depth_floor)
        mass_below = _normal_cdf((floor - float(mu[0])) / max(float(sd[0]), 1e-300))
        if mass_below > _DEPTH_FLOOR_MASS:
            # ...in the unit the rest of the pane is reading, like every other number
            # this function words. Four magnitudes with no unit at all was the same
            # hole the identifiability notes had, on the same surface.
            notes.insert(0, DEPTH_FLOOR_WARNING.format(
                floor=floor * note_factor, mu=float(mu[0]) * note_factor,
                sigma=float(sd[0]) * note_factor, edge=floor_edge * note_factor,
                mass=mass_below, unit=note_suffix,
            ))

    rows = []
    init = (mu, sd, w)
    for _ in range(int(n_boot) if ci_wanted() else 0):
        # Same rule as every other resampling loop in this module: poll per replicate,
        # OUTSIDE the try, and never let the blanket `except` read the cancel token as
        # a failed replicate (that would silently shrink the sample the CI is taken
        # from instead of stopping).
        check_cancelled()
        s = x[rng.integers(0, x.size, size=x.size)]
        try:
            if k == 1:
                rows.append([float(np.mean(s)), float(np.std(s))])
            else:
                mb, sb, wb, _ = _em_gauss(s, k, rng, init=init, n_iter=200)
                rows.append(np.concatenate([[mb[j], sb[j], wb[j]] for j in range(k)]))
        except OperationCancelled:
            raise
        except Exception:
            continue
    ci: dict[str, tuple[float, float]] = {}
    if len(rows) >= 20:
        b = np.asarray(rows)
        qs = np.percentile(b, [2.5, 97.5], axis=0)
        for i, key in enumerate(params):
            ci[key] = (float(qs[0, i]), float(qs[1, i]))
    elif ci_wanted():
        notes.append("bootstrap failed to produce enough replicates; CIs unavailable")

    # -- identifiability (Medium-11) --------------------------------------
    # A k-component fit always RETURNS k components; nothing above refuses when the
    # data support fewer. These notes are the refusal, and they run after the
    # bootstrap because two of the three tests are statements about the CIs.
    if k > 1:
        notes.extend(_identifiability_notes(mu, sd, w, ci, int(x.size), k,
                                            note_unit, note_factor))
        light = [j for j in range(k) if float(w[j]) * x.size < _MIN_COMPONENT_EVENTS]
        if light:
            notes.append(
                "component(s) " + ", ".join(
                    f"{j + 1} (w = {w[j]:.3g}, ~{w[j] * x.size:.0f} events)" for j in light)
                + f" are supported by fewer than {_MIN_COMPONENT_EVENTS} events — the minimum "
                  f"this module will fit a whole sample with. Treat them as unresolved.")

    meta = {
        "family": "gauss-mixture",     # v6 persistence discriminator; see density_from()
        "n_components": k, "n_boot_ok": len(rows),
        # See fit_dwell_exponential: the replicates a band needs, in `params` order.
        "boot": (np.asarray(rows, dtype=np.float64) if rows else None),
        "boot_params": list(params),
        "depth_floor": floor, "mu_minus_2sigma": floor_edge,
        "mass_below_floor": mass_below,
        "floor_biased": bool(mass_below is not None and mass_below > _DEPTH_FLOOR_MASS),
        # What was fitted, and therefore what mu/sigma are IN. See FitResult.unit_of().
        "quantity": quantity,
        "param_units": {p: ("" if p.startswith("w") else unit) for p in params},
        "loglik_k1": float(ll_one),
        "local_optimum": bool(local_optimum),
        "split_ladder_won": bool(ladder_won),
        "warnings": notes,
    }
    return _stamp_gof(FitResult(
        params=params, ci=ci, loglik=float(ll), aic=float(aic), bic=float(bic), n=int(x.size),
        model=f"Gaussian mixture on {quantity} (k={k}, EM)", meta=meta,
        _pdf=density_from("gauss-mixture", params, meta),
    ), x)


# ================================================================================================
# Peaked dwell models. An exponential mixture CANNOT have an interior mode.
# ================================================================================================
#: Families offered for a dwell distribution whose histogram has a PEAK.
#:
#: A mixture of exponentials is a sum of decaying terms, so it is monotonically decreasing
#: on t > 0 at every component count -- it can never place a maximum away from the origin.
#: On the owner's -700 mV apo-ferritin run the dwell histogram has a clear mode at ~18 ms:
#: k=1 gave tau = 16.8 ms through the middle of the data, k=2 recovered a 0.40 ms spike
#: (9%) and improved BIC by 78 while STILL decaying straight past the 18 ms peak. No number
#: of components fixes that; the model FAMILY is wrong.
#:
#: Both families here have a mode, and both are truncated exactly as the exponential fit is
#: -- normalised by the probability mass the detector could actually observe -- so their
#: AIC/BIC are directly comparable with it.
DWELL_PEAKED_MODELS = ("gamma", "inverse-gaussian")


def _truncated_nll(logpdf, cdf, t, t_min, t_max):
    """-log L for a left/right truncated sample. Shared by both peaked families.

    THE ARGUMENT IS A **LOG** DENSITY, AND THAT IS THE WHOLE POINT.
    ------------------------------------------------------------------
    This used to take ``pdf`` and refuse (``inf``) whenever any density underflowed
    to zero. For a peaked family that is not a guard, it is a landmine: the
    inverse-Gaussian density carries ``exp(-lam*(t-mu)^2 / (2*mu^2*t))``, and one
    short event far below the mode drives that exponent past the double-precision
    floor while the LIKELIHOOD is merely very small, not undefined. Measured on the
    owner's -700 mV apo-ferritin run (n = 986, mu = 19.83 ms, lam = 0.636): the
    shortest merged event is 0.10 ms, giving an exponent of about -7.6e3, so
    ``exp()`` returned exactly 0 and the fit came back with **BIC = +inf** -- still
    offered in the menu, still drawn, silently unusable (report Low-19).

    Summing log-densities instead makes that event contribute its true -7.6e3 to the
    log-likelihood: a finite, enormous penalty, which is exactly the right verdict.
    The arithmetic is otherwise identical, so every fit that did not underflow is
    unchanged to the last bit.
    """
    ld = logpdf(t)
    if not np.all(np.isfinite(ld)):
        return np.inf
    lo = cdf(t_min) if t_min and t_min > 0 else 0.0
    hi = cdf(t_max) if (t_max is not None and np.isfinite(t_max)) else 1.0
    mass = float(hi - lo)
    if not (mass > 0):
        return np.inf
    return float(-np.sum(ld) + t.size * np.log(mass))


def fit_dwell_peaked(
    dwell,
    *,
    model: str = "gamma",
    t_min: float | None = None,
    t_max: float | None = None,
    n_boot: int = 200,
    seed: int | None = 0,
    quantity: str | None = None,
    unit: str | None = None,
) -> FitResult:
    """Fit a dwell distribution that HAS A MODE -- gamma or inverse-Gaussian.

    ``gamma``: shape ``k`` and scale ``theta``; the mode sits at ``(k-1)*theta`` for
    ``k > 1``, and ``k = 1`` degenerates to the exponential, so this family CONTAINS the
    old model rather than competing with it.

    ``inverse-gaussian``: the first-passage-time density of drifting Brownian motion --
    i.e. the physical model of a molecule driven through a pore. Parameters are the mean
    passage time ``mu`` and the shape ``lambda``; the sharper the drift relative to
    diffusion, the more peaked it is.

    Truncation is handled exactly as :func:`fit_dwell_exponential` does: the likelihood is
    divided by the mass between ``t_min`` (the detector's floor -- it cannot report a dwell
    below its own resolution) and ``t_max``. Without that the estimates are biased by the
    events the detector could never have shown you, and the AIC/BIC would not be comparable
    with the exponential's.

    Args:
        dwell: Observed dwell times in seconds.
        model: One of :data:`DWELL_PEAKED_MODELS`.
        t_min: Detection limit; the likelihood is conditioned on ``t >= t_min``.
        t_max: Optional upper truncation.
        n_boot: Percentile-bootstrap replicates for the CIs, honouring
            :func:`point_fits_only` like every other bootstrap here. **These families
            used to return ``ci = {}``** -- and they are the two that can have a mode,
            i.e. the two that actually describe the owner's dwell distributions, so
            the only fits carrying error bars were the ones the data reject (report
            Medium-18). Each replicate is a 2-parameter Nelder-Mead warm-started from
            the point estimate; measured on the -700 mV apo run (n = 986), 200
            replicates cost ~0.6 s against 8 ms for the point fit, which is why the
            rival scans wrap themselves in ``point_fits_only()``.

            A caller that DOES read the CI -- so ``point_fits_only`` is not open to
            it -- but is on an interaction path wraps itself in :func:`deferred_ci`
            instead: the point estimate comes back immediately, ``meta['ci_pending']``
            says the interval is still coming, and the bootstrap is handed over as a
            callable to run off that thread. Re-measured on the same run: 3.9 ms for
            the point fit against 975 ms with the interval (``gamma``), 14.7 ms
            against 1634 ms (``inverse-gaussian``).
        seed: RNG seed for the bootstrap.
        quantity: What the fitted column MEASURES, when it is not a dwell time --
            the :func:`fit_depth_gaussians` contract (report High-4). The GUI
            offers these families on ANY positive column, and the default
            labelling stamped ``theta``/``mu``/``lambda`` as seconds whatever was
            fitted, so an inverse-Gaussian over ``depth_pA`` captioned
            "μ = 79.72 s" for a number that is picoamps. It also names the model
            ("gamma (truncated MLE) on depth_pA") and the refusals, so an error
            cannot blame dwell times the caller never fitted. ``None`` (the
            default) keeps the dwell reading exactly as before.
        unit: The STORAGE unit the scale parameters are in -- ``"pA"`` for
            ``depth_pA``, ``""`` for a ratio. Defaults to ``"s"`` when
            ``quantity`` is ``None`` and to ``quantity`` otherwise. The gamma
            shape ``k`` is dimensionless and never takes it; the
            inverse-Gaussian ``mu`` and ``lambda`` are both in it (lambda scales
            with the variate).

    Raises:
        ValueError: on a non-positive / too-small sample, or an unknown ``model``.
        RuntimeError: when the maximised likelihood is not finite -- an unusable fit
            that used to be returned with ``BIC = inf`` and drawn anyway (Low-19).
    """
    from scipy import optimize, special

    if model not in DWELL_PEAKED_MODELS:
        raise ValueError(f"model must be one of {DWELL_PEAKED_MODELS}, got {model!r}")
    # The storage unit, decided before anything is worded or stamped: the dwell
    # contract ("s") unless the caller says what was really fitted -- the same
    # normalisation fit_depth_gaussians and fit_dwell_exponential apply.
    quantity = None if quantity is None else str(quantity)
    unit = ("s" if quantity is None else quantity) if unit is None else str(unit)
    fitted_name = quantity if quantity is not None else "dwell"
    t = _clean_1d(dwell, fitted_name, positive=True)
    if t.size < 20:
        raise ValueError(
            f"need >= 20 positive {fitted_name} values to fit, got {t.size}"
            if quantity is not None else
            f"need >= 20 positive dwell times to fit, got {t.size}")
    lo = float(t_min) if t_min else 0.0
    hi = float(t_max) if (t_max is not None and np.isfinite(t_max)) else None
    mean, var = float(t.mean()), float(t.var(ddof=1)) or float(t.mean()) ** 2

    # Both families are returned as LOG densities: see _truncated_nll on why summing
    # log f beats logging a product that underflows.
    if model == "gamma":
        def _pdf_for(p):
            k, th = np.exp(p)
            return (lambda x: ((k - 1) * np.log(np.maximum(x, 1e-300)) - x / th
                               - special.gammaln(k) - k * np.log(th)),
                    lambda x: special.gammainc(k, np.asarray(x, float) / th))
        p0 = np.log([max(mean * mean / var, 1e-3), max(var / mean, 1e-12)])
        names = ("k", "theta")
        # theta is a scale ON THE FITTED AXIS: seconds for dwells, the column's own
        # unit otherwise. Stamping "s" unconditionally is how a gamma over depth_pA
        # reported its picoamp scale as a time (the report's unit-hole family).
        units = {"k": "", "theta": unit}
    else:
        def _pdf_for(p):
            mu, lam = np.exp(p)
            return (lambda x: 0.5 * np.log(lam / (2 * np.pi * np.maximum(x, 1e-300) ** 3))
                    - lam * (x - mu) ** 2 / (2 * mu * mu * np.maximum(x, 1e-300)),
                    lambda x: _invgauss_cdf(np.asarray(x, float), mu, lam))
        p0 = np.log([max(mean, 1e-12), max(mean ** 3 / var, 1e-12)])
        names = ("mu", "lambda")
        # Both parameters carry the variate's own unit (lambda scales with it); the
        # variate is a dwell only when the caller did not say otherwise.
        units = {"mu": unit, "lambda": unit}

    def nll(p):
        try:
            logpdf, cdf = _pdf_for(p)
            return _truncated_nll(logpdf, cdf, t, lo, hi)
        except (FloatingPointError, ValueError):
            return np.inf

    # The last poll before a single ~4-15 ms maximisation. Cheap, and it is what makes
    # a LADDER of these (`_compute_better_k`, `compare_dwell_models`) stop between
    # rivals rather than only between families.
    check_cancelled()
    stall = _StallStop(cancel_callback())
    with np.errstate(all="ignore"):
        res = optimize.minimize(nll, p0, method="Nelder-Mead",
                                callback=stall,
                                options={"maxiter": _NM_MAXITER, "xatol": 1e-10,
                                         "fatol": 1e-10})
    p = res.x
    ll = -float(nll(p))
    if not np.isfinite(ll):
        # An unusable fit must not be handed back wearing a number. It used to come
        # out as BIC = +inf, sort last in no comparison anyone ran, and still be drawn.
        raise RuntimeError(
            f"the {model} likelihood is not finite at its own optimum on this sample "
            f"(n = {t.size}, t_min = {lo:g}{' s' if unit == 's' else ''}): the family "
            f"cannot describe these "
            f"{'dwell times' if quantity is None else quantity + ' values'}. Choose "
            f"another model rather than reading an infinite BIC.")
    n_par = 2
    aic = 2 * n_par - 2 * ll
    bic = n_par * np.log(t.size) - 2 * ll
    a, b = np.exp(p)
    fit_params = {names[0]: float(a), names[1]: float(b)}
    notes: list[str] = []
    # A SEARCH STOPPED BY THE STALL GUARD HAS NOT MET ITS TOLERANCE. scipy 1.11+ says
    # so itself (`_minimize` turns a callback's StopIteration into success=False,
    # status=99), but this does not READ that: `stall.stopped` is the panel-visible
    # fact and it must not depend on a post-processing step in another library, whose
    # older versions left `warnflag` -- and therefore `success` -- untouched by a
    # callback halt. The two are ANDed; either one alone is enough to refuse the word
    # "converged".
    converged = bool(res.success) and not stall.stopped
    if stall.stopped:
        # ONE sentence, not two: `res.message` here is "`callback` raised
        # `StopIteration`." -- true, and it explains nothing to the person reading a
        # dwell histogram. This says what stopped and why it could never have stopped
        # any other way.
        notes.append(
            f"the {model} search STOPPED MOVING: its best point was unchanged for "
            f"{_NM_STALL_ITERS} consecutive iterations, so the maximisation was ended "
            f"there ({int(getattr(res, 'nit', -1))} iterations) instead of spending the "
            f"rest of its {_NM_MAXITER}-iteration budget re-testing the same point. It "
            f"did NOT meet its convergence tolerance: that tolerance is applied to the "
            f"LOG of each parameter, so an optimum sitting at the EDGE of the family — "
            f"for gamma, shape k → 0, which is a density with no mode at all — can "
            f"never satisfy it, and the iteration cap becomes the only stopping rule "
            f"that ever fires. The estimates below are that point. Read the shape "
            f"parameter before the BIC: a “peaked” family that has collapsed onto its "
            f"own boundary is not describing a peak.")
    elif not converged:
        notes.append(
            f"the {model} optimiser reported non-convergence ({res.message}); the estimates "
            f"below are where it stopped.")

    # -- bootstrap (Medium-18) --------------------------------------------
    # ONE CALLABLE, so the same arithmetic can run here or on a worker thread.
    # It closes over the fitted simplex ``p``, the sample, the truncation and the
    # seed, which is what makes a deferred CI bit-identical to a synchronous one
    # rather than merely similar: nothing is re-derived from the rounded params.
    # See deferred_ci() for why this has to be movable at all.
    def _bootstrap() -> dict[str, Any]:
        rows = []
        # THE SWITCH LIVES WITH THE LOOP IT SWITCHES OFF, not only at the call site.
        # This callable can be invoked from anywhere -- including a worker thread,
        # whose fresh context restores the default -- so the resampling loop asks
        # `ci_wanted()` itself rather than trusting whoever kept hold of it. Same rule
        # every other bootstrap in this module follows; there is no second switch.
        if not (ci_wanted() and n_boot and int(n_boot) > 0):
            return {"ci": {}, "n_boot_ok": 0, "note": None, "boot": None}
        rng = np.random.default_rng(seed)
        # One replicate is 3-8 ms on -700 mV but 25 ms (gamma) to 46 ms
        # (inverse-gaussian) on the owner's -600 mV sample -- half the 100 ms join, on
        # the recording that already breaks the between-units assumption elsewhere in
        # this module. The poll goes inside the replicate's own minimisation as well as
        # between replicates, so the margin does not depend on the sample.
        _poll_inside = cancel_callback()
        with np.errstate(all="ignore"):
            for _ in range(int(n_boot)):
                # Cooperative cancellation, polled every replicate: a replicate is
                # ~3-8 ms, so the token costs nothing here and a superseded CI stops
                # inside its own loop instead of running to completion and then being
                # thrown away. Never a kwarg -- see analysis.cancel.
                check_cancelled()
                s = t[rng.integers(0, t.size, size=t.size)]

                def _nll_b(q, _s=s):
                    try:
                        lp, cd = _pdf_for(q)
                        return _truncated_nll(lp, cd, _s, lo, hi)
                    except (FloatingPointError, ValueError):
                        return np.inf
                rb = optimize.minimize(_nll_b, p, method="Nelder-Mead",
                                       callback=_poll_inside,
                                       options={"maxiter": 2000, "xatol": 1e-8,
                                                "fatol": 1e-8})
                if np.isfinite(rb.fun):
                    rows.append(np.exp(rb.x))
        # The replicates travel WITH the percentiles, in `fit_params` order, so a
        # deferred bootstrap can hand a band back over the same seam a deferred interval
        # already crosses (apply_ci). See ci_band() for why a band needs them.
        out: dict[str, Any] = {"ci": {}, "n_boot_ok": len(rows), "note": None,
                               "boot": (np.asarray(rows, dtype=np.float64) if rows else None)}
        if len(rows) >= 20:
            qs = np.percentile(np.asarray(rows), [2.5, 97.5], axis=0)
            out["ci"] = {key: (float(qs[0, i]), float(qs[1, i]))
                         for i, key in enumerate(fit_params)}
        else:
            out["note"] = "bootstrap failed to produce enough replicates; CIs unavailable"
        return out

    ci: dict[str, tuple[float, float]] = {}
    n_boot_ok = 0
    boot_rows = None
    ci_pending = False
    jobs = _CI_DEFER.get()
    if ci_wanted() and n_boot and int(n_boot) > 0:
        if jobs is not None:
            # The caller has said it will run this somewhere that is not an
            # interaction path. Hand it over; say so in meta so no surface reads an
            # empty `ci` as "this family does not offer one".
            jobs.append(_bootstrap)
            ci_pending = True
        else:
            payload = _bootstrap()
            ci = payload["ci"]
            n_boot_ok = int(payload["n_boot_ok"])
            boot_rows = payload.get("boot")
            if payload["note"]:
                notes.append(payload["note"])

    meta = {"family": model,   # v6 persistence discriminator; see density_from()
            "censoring": f"left {lo:g}" + (f", right {hi:g}" if hi else ""),
            "converged": converged, "t_min": lo, "t_max": hi,
            # How many iterations the search spent, and whether the stall guard is why
            # it stopped. Recorded rather than inferred: "nit < maxiter" used to mean
            # "converged" and no longer does.
            "n_iter": int(getattr(res, "nit", -1)), "stalled": bool(stall.stopped),
            "n_boot_ok": n_boot_ok,
            "boot": boot_rows,
            "boot_params": list(fit_params),
            **({"ci_pending": True} if ci_pending else {}),
            # What was fitted (None = the dwell default), same key as the depth mixture.
            "quantity": quantity,
            "param_units": units,
            "mode": float((a - 1) * b) if model == "gamma" and a > 1 else None,
            "warnings": notes}
    return _stamp_gof(FitResult(
        params=fit_params,
        ci=ci, loglik=ll, aic=float(aic), bic=float(bic), n=int(t.size),
        # What was fitted travels with the fit (cf. "Gaussian mixture on depth_pA"):
        # the dwell default keeps the exact historical name.
        model=f"{model} (truncated MLE)" + (f" on {quantity}" if quantity else ""),
        meta=meta,
        _pdf=density_from(model, fit_params, meta),
    ), t)


def _invgauss_cdf(x, mu, lam):
    """Inverse-Gaussian CDF via the standard normal, guarding x <= 0."""
    from scipy import special
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    ok = x > 0
    if not np.any(ok):
        return out
    xa = x[ok]
    r = np.sqrt(lam / xa)
    phi = lambda z: 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))   # noqa: E731
    out[ok] = phi(r * (xa / mu - 1.0)) + np.exp(2 * lam / mu) * phi(-r * (xa / mu + 1.0))
    return out


# ================================================================================================
# Rebuilding a fitted density from stored state (v6 project persistence)
# ================================================================================================
#: Stable discriminators for the fitted families, written into ``FitResult.meta['family']``.
#:
#: These exist because ``FitResult.model`` is a HUMAN-READABLE string ("exponential mixture
#: (k=3, censored MLE)") that is safe to reword; dispatching persistence on it would make a
#: cosmetic edit to a label silently unpick every stored fit.
FIT_FAMILIES = ("exp-mixture", "gauss-mixture", "gamma", "inverse-gaussian",
                "exp-gamma-mixture")


def _mix_params(params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """``{tau_1, ..., w_1, ...}`` -> sorted ``(taus, weights)``; k=1 implies w=[1]."""
    taus = [float(params[k]) for k in sorted(
        (p for p in params if p.startswith("tau_")), key=lambda s: int(s.split("_")[1]))]
    ws = [float(params[k]) for k in sorted(
        (p for p in params if p.startswith("w_")), key=lambda s: int(s.split("_")[1]))]
    if not ws:
        ws = [1.0] * len(taus)
    return np.asarray(taus, float), np.asarray(ws, float)


def density_from(family: str, params: dict[str, float],
                 meta: dict[str, Any] | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Rebuild a fit's density function from JSON-safe state alone.

    This is THE implementation of every fitted density in this module: each fitter calls it
    to build its own ``FitResult._pdf``, and :mod:`nanoscope.core.project` calls it to
    restore one from a saved project. That sharing is the whole point. A separate
    "deserialisation" copy of these formulae could drift from the ones that produced the
    numbers, and the failure would be silent and awful -- a reopened project drawing a curve
    that no longer corresponds to its own tau/AIC. There is one formula per family, used on
    both paths, so a round-trip is an identity by construction and the test that asserts
    ``restored.predict(x) == original.predict(x)`` is checking wiring rather than algebra.

    ``meta`` must carry whatever the family needs beyond ``params``:

    * ``exp-mixture`` -- ``t_min``, ``t_max``, and the measured efficiency curve as
      ``eta_grid``/``eta_vals``. eta is the one piece of fit state that is neither a
      parameter nor a scalar; before v6 it lived only in a closure, which is precisely why
      this family could not be persisted.
    * ``gamma`` / ``inverse-gaussian`` -- ``t_min``, ``t_max`` for the truncation mass.
    * ``gauss-mixture`` -- nothing; the fit is untruncated.
    """
    from scipy import special

    meta = dict(meta or {})
    if family == "gauss-mixture":
        k = sum(1 for p in params if p.startswith("mu_"))
        mu = np.array([float(params[f"mu_{j + 1}"]) for j in range(k)])
        sd = np.array([float(params[f"sigma_{j + 1}"]) for j in range(k)])
        w = (np.array([float(params[f"w_{j + 1}"]) for j in range(k)]) if k > 1
             else np.array([1.0]))

        def pdf_gauss(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.float64)
            return np.sum(w[:, None] * _norm_pdf(v, mu, sd), axis=0)

        return pdf_gauss

    t_min = float(meta.get("t_min") or 0.0)
    _tmax = meta.get("t_max")
    t_max = None if _tmax is None or not np.isfinite(float(_tmax)) else float(_tmax)

    if family == "exp-mixture":
        taus, w = _mix_params(params)
        g, v = meta.get("eta_grid"), meta.get("eta_vals")
        grid = None if g is None else np.asarray(g, dtype=np.float64)
        vals = None if v is None else np.asarray(v, dtype=np.float64)
        norm = _Normalizer(t_min, grid, vals, t_max)
        z = float(norm(taus, w))

        def pdf_exp(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            out = np.zeros_like(x)
            m = x >= t_min
            if t_max is not None:
                m = m & (x <= t_max)   # no density outside the observable window
            if np.any(m):
                f = _mix_pdf(x[m], taus, w)
                if grid is not None:
                    f = f * _eta_at(x[m], grid, vals)
                out[m] = f / z
            return out

        return pdf_exp

    if family == "exp-gamma-mixture":
        w = float(params["w_exp"]); tau = float(params["tau"])
        k = float(params["k"]); th = float(params["theta"])

        def _f(x):
            x = np.maximum(np.asarray(x, float), 1e-300)
            return (w * np.exp(-x / tau) / tau
                    + (1 - w) * np.exp((k - 1) * np.log(x) - x / th
                                       - special.gammaln(k) - k * np.log(th)))

        z_e = np.exp(-t_min / tau) - (np.exp(-t_max / tau) if t_max else 0.0)
        z_g = ((special.gammainc(k, t_max / th) if t_max else 1.0)
               - (special.gammainc(k, t_min / th) if t_min > 0 else 0.0))
        z = float(w * z_e + (1 - w) * z_g) or 1.0

        def pdf_eg(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            out = np.zeros_like(x)
            m = x >= t_min
            if t_max is not None:
                m = m & (x <= t_max)
            if np.any(m):
                out[m] = _f(x[m]) / z
            return out

        return pdf_eg

    if family in ("gamma", "inverse-gaussian"):
        if family == "gamma":
            a, b = float(params["k"]), float(params["theta"])
            def _f(x):
                return np.exp((a - 1) * np.log(np.maximum(x, 1e-300)) - x / b
                              - special.gammaln(a) - a * np.log(b))
            def _c(x):
                return special.gammainc(a, np.asarray(x, float) / b)
        else:
            a, b = float(params["mu"]), float(params["lambda"])
            def _f(x):
                return (np.sqrt(b / (2 * np.pi * np.maximum(x, 1e-300) ** 3))
                        * np.exp(-b * (x - a) ** 2 / (2 * a * a * np.maximum(x, 1e-300))))
            def _c(x):
                return _invgauss_cdf(np.asarray(x, float), a, b)
        mass = float((_c(t_max) if t_max else 1.0) - (_c(t_min) if t_min > 0 else 0.0)) or 1.0
        return lambda x: np.asarray(_f(np.asarray(x, float)), float) / mass

    raise ValueError(f"unknown fit family {family!r}; expected one of {FIT_FAMILIES}")


def fit_to_json(fit: FitResult) -> dict[str, Any]:
    """Flatten a :class:`FitResult` to JSON-safe builtins for project storage.

    The density is deliberately NOT included: it is a closure, and the point of
    :func:`density_from` is that it never has to be. Everything here is a number, a
    string, or a list of them.
    """
    return {
        "params": {str(k): float(v) for k, v in fit.params.items()},
        "ci": {str(k): [float(lo), float(hi)] for k, (lo, hi) in fit.ci.items()},
        "loglik": float(fit.loglik), "aic": float(fit.aic), "bic": float(fit.bic),
        "n": int(fit.n), "model": str(fit.model),
        # THE REPLICATES ARE DROPPED HERE, DELIBERATELY. `meta` is serialised wholesale,
        # and the retained bootstrap is up to 8000 float64 per fit -- megabytes of
        # resampled parameter noise per project that nothing reads back, to save a band
        # a refit rebuilds for free. Intervals persist; the band does not. See ci_band().
        "meta": _json_scalars({k: v for k, v in fit.meta.items()
                               if k not in (BOOT_KEY, BOOT_PARAMS_KEY)}),
    }


def _json_scalars(obj: Any) -> Any:
    """numpy -> builtins, recursively; anything exotic degrades to ``str`` rather than
    aborting a save."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_json_scalars(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_scalars(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_scalars(v) for v in obj]
    return str(obj)


def fit_from_json(blob: dict[str, Any]) -> FitResult:
    """Rebuild a :class:`FitResult` from :func:`fit_to_json`, density included.

    The density comes from :func:`density_from` — the same function the fitter used — so
    ``fit_from_json(fit_to_json(f)).predict(x)`` equals ``f.predict(x)`` exactly, not
    approximately, for every family that names itself in ``meta['family']``.

    A fit written before families were stamped (or by a family this build does not know)
    restores with its numbers and **no** density: ``predict()`` then raises its usual
    "carries no density" error instead of silently drawing a curve from a guess.
    """
    meta = dict(blob.get("meta") or {})
    params = {str(k): float(v) for k, v in (blob.get("params") or {}).items()}
    family = meta.get("family")
    pdf = None
    if family in FIT_FAMILIES:
        try:
            pdf = density_from(str(family), params, meta)
        except (KeyError, ValueError, TypeError):
            pdf = None      # numbers survive; a broken curve must not break the reopen
    return FitResult(
        params=params,
        ci={str(k): (float(v[0]), float(v[1])) for k, v in (blob.get("ci") or {}).items()},
        loglik=float(blob.get("loglik", float("nan"))),
        aic=float(blob.get("aic", float("nan"))),
        bic=float(blob.get("bic", float("nan"))),
        n=int(blob.get("n", 0)),
        model=str(blob.get("model", "")),
        meta=meta,
        _pdf=pdf,
    )


# ================================================================================================
# Exponential + gamma mixture: a fast escape AND a slow multi-step passage
# ================================================================================================
#: The family name, for ``meta['family']`` and :func:`density_from`.
EXP_GAMMA = "exp-gamma-mixture"


def fit_dwell_exp_gamma(
    dwell,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    quantity: str | None = None,
    unit: str | None = None,
) -> FitResult:
    """Fit ``w·Exp(tau) + (1-w)·Gamma(k, theta)`` to dwell times, truncated like the rest.

    WHY THIS FAMILY EXISTS. Neither model already here can describe a dwell distribution
    that is bimodal with a PEAKED slow component. An exponential mixture is a sum of
    decaying terms, so in linear density it is monotone and its log-space mode can only sit
    at a tau -- it can lean toward a peak but never make a sharp one. A single gamma has one
    mode, so forced to explain a fast population as well it collapses to k ~ 1, which IS an
    exponential, and gains nothing.

    Measured on the owner's apo-ferritin -600 mV recording (n = 316, both other models fit
    through the app's own Fits panel):

        exponential mixture k=2   BIC -2062.4
        gamma (single, truncated) BIC -2040.4   (k = 1.041 -- collapsed to an exponential)
        exponential + gamma       BIC -2681.9   <- this family

    dBIC ~ 620. The fitted split was 25% fast (tau = 1.97 ms) and 75% gamma with k = 70,
    mean 18.4 ms -- a near-deterministic duration (relative sd 1/sqrt(k) = 12%), which is
    what a multi-step passage looks like and is consistent with the 3.1 CUSUM sub-state
    levels per event measured on the same file.

    Args:
        dwell: Observed dwell times in seconds.
        t_min: Detection limit. The likelihood is conditioned on ``t >= t_min``, exactly as
            :func:`fit_dwell_exponential` does -- without it the fast component absorbs the
            missing short events and both weights move.
        t_max: Optional upper truncation (``detect_events(max_dwell_s=...)`` or curation).
        quantity: What the fitted column MEASURES, when it is not a dwell time --
            the same contract as the other dwell fitters and
            :func:`fit_depth_gaussians`. Names the model and the refusals, and
            fixes the unit ``tau``/``theta`` are stamped with, so a fit the GUI
            let loose on a picoamp column does not persist its scale parameters
            as seconds. ``None`` (the default) keeps the dwell reading.
        unit: The STORAGE unit of the sample (and so of ``tau`` and ``theta``);
            defaults to ``"s"`` when ``quantity`` is ``None`` and to ``quantity``
            otherwise. ``w_exp`` and ``k`` are dimensionless and never take it.

    Returns:
        A :class:`FitResult` whose ``params`` are ``w_exp``, ``tau``, ``k``, ``theta``.
    """
    from scipy import optimize, special

    quantity = None if quantity is None else str(quantity)
    unit = ("s" if quantity is None else quantity) if unit is None else str(unit)
    # The refusal and the truncation notes speak the sample's own unit, exactly as
    # `param_units` below stamps tau/theta (BL-3): a depth_pA fit with a fit range
    # used to print "DOUBLE truncation, 10 s <= t <= 300 s" for picoamp bounds.
    # For the dwell default (`quantity is None`) `unit` is "s" and every sentence
    # is bit-for-bit what it always was.
    unit_sfx = f" {unit}" if unit else ""
    t = _clean_1d(dwell, quantity if quantity is not None else "dwell", positive=True)
    if t.size < _MIN_EVENTS:
        raise ValueError(
            f"need >= {_MIN_EVENTS} positive dwell times to fit, got {t.size}"
            if quantity is None else
            f"need >= {_MIN_EVENTS} positive {quantity} values to fit, got {t.size}")
    lo = float(t_min) if t_min else 0.0
    hi = float(t_max) if (t_max is not None and np.isfinite(t_max)) else None
    if lo > 0 and float(np.min(t)) < lo * (1 - 1e-9):
        raise ValueError(
            (f"{int(np.sum(t < lo))} event(s) are shorter than the stated detection limit "
             f"t_min={lo:g} s (min dwell = {np.min(t):g} s). Either t_min is wrong or the "
             f"detector is emitting events below its own resolution. "
             if quantity is None else
             f"{int(np.sum(t < lo))} {quantity} value(s) are below the stated lower limit "
             f"t_min={lo:g}{unit_sfx} (min {quantity} = {np.min(t):g}{unit_sfx}). Either "
             f"t_min is wrong or values below the stated floor were reported anyway. ")
            + "Refusing to fit."
        )
    notes: list[str] = []
    if hi is not None:
        notes.append(f"DOUBLE truncation, {lo:g}{unit_sfx} <= t <= {hi:g}{unit_sfx}")
    elif lo > 0:
        notes.append(f"hard left-truncation at t_min={lo:g}{unit_sfx}")

    mean, var = float(t.mean()), float(t.var(ddof=1)) or float(t.mean()) ** 2

    def unpack(p):
        w = 1.0 / (1.0 + np.exp(-p[0]))
        return w, np.exp(p[1]), np.exp(p[2]), np.exp(p[3])

    def _nll(p):
        w, tau, k, th = unpack(p)
        if not all(np.isfinite([w, tau, k, th])) or min(tau, k, th) <= 0:
            return np.inf
        f = (w * np.exp(-t / tau) / tau
             + (1 - w) * np.exp((k - 1) * np.log(t) - t / th
                                - special.gammaln(k) - k * np.log(th)))
        # Truncation mass, per component, so each is conditioned on the observable window.
        z_e = np.exp(-lo / tau) - (np.exp(-hi / tau) if hi else 0.0)
        z_g = ((special.gammainc(k, hi / th) if hi else 1.0)
               - (special.gammainc(k, lo / th) if lo > 0 else 0.0))
        z = w * z_e + (1 - w) * z_g
        if not np.isfinite(z) or z <= 0:
            return np.inf
        return -float(np.sum(np.log(np.maximum(f, 1e-300)))) + t.size * float(np.log(z))

    # Multi-start: the likelihood has a shallow ridge between "one broad gamma" and "a fast
    # exponential plus a sharp gamma", and a single start lands on whichever the initial k
    # is nearer. Starting from several k spans both.
    best = None
    # THE POLL THAT MAKES THIS FAMILY CANCELLABLE AT ALL, and it is INSIDE the
    # minimisation, not between the starts.
    #
    # It used to sit once per start, on the reasoning that a start was ~37 ms against a
    # 100 ms join. That reasoning was measured on ONE sample. On the owner's -600 mV apo
    # recording (n = 240 after merge) a single start is ~790 ms and the twelve are 9.5 s
    # -- against 207 ms for all twelve on -700 mV (n = 986), on a QUARTER of the events
    # -- so the poll sat eight times further apart than the join it was supposed to keep,
    # and two aborts in three detached a still-running worker. A poll reachable only
    # between units of work is not a bound; see :func:`cancel_callback`.
    _poll_inside = cancel_callback()
    with np.errstate(all="ignore"):
        for k0 in (2.0, 5.0, 20.0, 60.0):
            for w0 in (-1.0, 0.0, 1.0):
                check_cancelled()
                p0 = [w0, np.log(max(mean * 0.1, 1e-9)), np.log(k0), np.log(max(mean / k0, 1e-12))]
                r = optimize.minimize(_nll, p0, method="Nelder-Mead",
                                      callback=_poll_inside,
                                      options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
                if best is None or r.fun < best.fun:
                    best = r
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("the exponential+gamma likelihood did not converge from any start")

    w, tau, k, th = unpack(best.x)
    ll = -float(best.fun)
    n_par = 4
    params = {"w_exp": float(w), "tau": float(tau), "k": float(k), "theta": float(th)}

    # No bootstrap: a replicate here costs a full 12-start Nelder-Mead, so CIs would take
    # minutes. The absence is SAID rather than left as a silently empty `ci` dict, because
    # the other fitters in this module do produce CIs and a blank column reads as "the
    # bootstrap failed" instead of "this family does not offer one".
    ci: dict[str, tuple[float, float]] = {}
    notes.append(
        "no bootstrap CIs for this family: each replicate is a 12-start optimisation. "
        "The point estimates stand on the likelihood; compare families by BIC.")

    if k <= 1.05:
        notes.append(
            f"the gamma component came back k = {k:.3g}, i.e. essentially an exponential — "
            f"this data may not need a peaked component; compare BIC against the plain "
            f"exponential mixture before reporting a two-mechanism interpretation")

    meta = {
        "family": EXP_GAMMA,
        "t_min": lo, "t_max": hi,
        # What was fitted (None = the dwell default), same key as the other fitters.
        "quantity": quantity,
        # w_exp is a probability and k a gamma shape; only tau and theta carry the
        # sample's unit -- seconds for dwells, the fitted column's own otherwise.
        "param_units": {"w_exp": "", "tau": unit, "k": "", "theta": unit},
        "censoring": f"left {lo:g}" + (f", right {hi:g}" if hi else ""),
        "converged": bool(best.success),
        "gamma_mean_s": float(k * th),
        "gamma_mode_s": float((k - 1) * th) if k > 1 else None,
        "gamma_rel_sd": float(1.0 / np.sqrt(k)) if k > 0 else None,
        "warnings": notes,
    }
    return _stamp_gof(FitResult(
        params=params, ci=ci, loglik=ll,
        aic=float(2 * n_par - 2 * ll), bic=float(n_par * np.log(t.size) - 2 * ll),
        n=int(t.size),
        # What was fitted travels with the fit; the dwell default keeps the old name.
        model="exponential + gamma mixture (censored MLE)"
              + (f" on {quantity}" if quantity else ""),
        meta=meta,
        _pdf=density_from(EXP_GAMMA, params, meta),
    ), t)


# ================================================================================================
# IS THIS MODEL EVEN RIGHT? -- goodness of fit, and the rivals it is being preferred over
# ================================================================================================
# THE DEFECT THIS SECTION EXISTS FOR (report High-3).
#
# The Plots panel printed
#
#     tau = 0.01981   (n = 986)          CI [0.019589, 0.020032]
#
# for the owner's -700 mV apo-ferritin dwell times, with NO goodness-of-fit statistic and
# NO warning, while the drawn curve visibly missed the histogram. The exponential is the
# WORST model in the app's own menu on all three of those recordings -- gamma beats it by
# 90 to 2111 BIC and exponential+gamma by 856 to 2729 -- and that tau is arithmetically
# just ``mean(t) - t_min`` (observed mean 19.829 ms, tau 19.809 ms), i.e. a restatement of
# the sample mean wearing the authority of a fitted time constant.
#
# Two things were missing, and neither of them is a different default model:
#
#   * a statement of whether the fitted curve DESCRIBES the data at all, and
#   * the cross-family BIC comparison the app was one call away from being able to make,
#     since it can already fit every family in its own menu.
#
# Both live here rather than in a panel, because both panels print fits and the answer
# must not depend on which window you asked in.

#: KS critical value coefficient at alpha = 0.01 for the one-sample two-sided test:
#: ``D_crit = _KS_CRIT_001 / sqrt(n)``. Deliberately the TABULATED value, which assumes
#: the model's parameters were known in advance rather than estimated from this very
#: sample. Estimating them makes the true critical value SMALLER, so this threshold errs
#: toward staying quiet -- a fit has to be clearly wrong before the panel says so.
_KS_CRIT_001 = 1.62762


# ===========================================================================
# AN R2 IS ONLY AS GOOD AS THE POINTS LEFT OVER AFTER THE PARAMETERS.
# ===========================================================================
# A straight line has two free parameters. Through two points it passes exactly,
# and R2 = 1.000 -- for ANY two points, including two points of pure noise. The
# same is true of the log-log line (two parameters again, coefficient and
# exponent) and of the voltage-dependence trend (slope/intercept, or A/V0).
# Nothing is computed wrongly; the number simply carries no information, and
# printed bare beside "R2 = 0.997 (n = 986)" it reads as the stronger of the two.
#
# The app already says this in one place -- ``gui.psd_panel._flicker_lines``
# ("A log-log line has two free parameters, so R2 = ... over N points is nearly
# free and is not evidence the power law holds") -- and said it nowhere else, so
# the Plots scatter printed "R2 = 1.000 (n = 2)" in its title, its legend and its
# Details in the same window whose histogram refuses those two events outright
# ("a distribution fit to a handful of events is decoration, not a measurement"),
# and the voltage-dependence panel printed "Goodness of fit: R2 = 1.00000 . n = 2
# recording(s)" for a line through two voltages.
#
# One writer, here, because both of those surfaces are fitting surfaces and the
# answer must not depend on which window asked. The short form goes where the
# number goes (a title has a wrap budget); the sentence goes in the details pane.

def r2_dof_tag(n: int, n_params: int) -> str:
    """The short qualifier that must ride ALONG WITH a printed R2, or ``""``.

    Empty as soon as one residual degree of freedom exists: this is a guard on a
    degenerate case, not a running commentary on small samples.
    """
    dof = int(n) - int(n_params)
    if dof > 0:
        return ""
    if dof == 0:
        return "exact by construction"
    return "under-determined"


def r2_dof_caveat(n: int, n_params: int, *, model: str = "line",
                  noun: str = "point") -> str:
    """The full sentence for a printed R2 with no residual degrees of freedom, or ``""``.

    Args:
        n: How many observations were fitted.
        n_params: How many free parameters the model has (2 for every line this
            app fits, straight or log-log; 2 for both voltage-dependence trends).
        model: What to call the model in the sentence ("line", "trend", ...).
        noun: What to call one observation ("point", "recording", ...).
    """
    n, n_params = int(n), int(n_params)
    dof = n - n_params
    if dof > 0:
        return ""
    if dof == 0:
        return (f"R² = 1 here is EXACT BY CONSTRUCTION, not evidence: a {model} has "
                f"{n_params} free parameters and {n} {noun}s were fitted, so no residual "
                f"degrees of freedom remain and ANY {n} {noun}s whatsoever -- noise "
                f"included -- would score exactly the same. There is no goodness of fit "
                f"to report here.")
    return (f"the {model} has {n_params} free parameters and only {n} {noun}"
            f"{'' if n == 1 else 's'} {'was' if n == 1 else 'were'} fitted, so the fit is "
            f"under-determined: R² is arithmetic, not evidence, and the parameters are "
            f"not identified.")


def goodness_of_fit(fit: FitResult, sample) -> dict[str, Any]:
    """How well ``fit`` describes ``sample``: a Kolmogorov-Smirnov statistic and a verdict.

    The reference CDF is obtained by integrating :meth:`FitResult.predict` -- the
    **observed** (censored) density, i.e. the very curve the panel draws -- over the
    window the sample occupies. Two consequences worth being explicit about:

    * It tests the drawn curve. A fit whose overlay misses the histogram cannot pass
      here, whatever its parameters say, which is exactly the failure being caught.
    * It is family-agnostic. Every family in this module publishes a density, so one
      implementation covers exponential mixtures, gamma, inverse-Gaussian and the
      exponential+gamma mixture -- and the numbers are comparable between them.

    THE STATISTIC IS CONDITIONED ON THE OBSERVED RANGE. The reference CDF is
    renormalised over ``[min(sample), max(sample)]`` (or ``[t_min, max]`` where the fit
    knows a detection floor), so this asks "does this shape describe this histogram",
    not "does this model also predict the right amount of mass beyond the data". On a
    correctly specified fit the two agree: measured against ``scipy.stats.kstest`` with
    the analytic truncated-exponential CDF on 4000 draws from a truncated exponential,
    D = 0.015456 here against 0.015518 there. ``grid_mass`` reports how much of the
    model's own probability landed inside the window, so a family leaking most of its
    mass past the data is visible rather than flattered.

    Args:
        fit: Any :class:`FitResult` carrying a density.
        sample: The observations it was fit to (seconds for dwell, data units for depth).

    Returns:
        ``{'ks', 'ks_critical_001', 'describes_data', 'n', 'grid_mass', 'note'}``.
        ``describes_data`` is False when ``ks`` exceeds the 1% critical value.

    The gotcha:
        The p-value of a KS test whose parameters were estimated from the same sample is
        not the tabulated one, so none is reported. ``ks`` against ``ks_critical_001`` is
        a decision, not a probability, and it is stated that way in the note.
    """
    x = np.asarray(sample, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        raise ValueError(f"need >= 2 finite observations to test a fit, got {n}")
    lo = float(np.min(x))
    hi = float(np.max(x))
    if not (hi > lo):
        raise ValueError("all observations are identical; there is no distribution to test")
    t_min = fit.meta.get("t_min")
    if t_min is not None and np.isfinite(float(t_min)) and 0 < float(t_min) <= lo:
        lo = float(t_min)
    # A log grid where the data are positive and span decades (dwell times), linear
    # otherwise (depths, which can sit anywhere): both give the trapezoid a comparable
    # number of points per unit of probability.
    span = hi / lo if lo > 0 else 0.0
    grid = (np.geomspace(lo, hi, 4001) if span >= 10.0 else np.linspace(lo, hi, 4001))
    dens = np.asarray(fit.predict(grid), dtype=np.float64)
    dens = np.where(np.isfinite(dens), dens, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    total = float(cum[-1])
    if not (total > 0):
        raise ValueError(
            f"{fit.model} places no probability mass on the observed range "
            f"[{lo:g}, {hi:g}]; it cannot be compared with this sample")
    cdf = cum / total
    xs = np.sort(x)
    f = np.interp(xs, grid, cdf)
    i = np.arange(1, n + 1, dtype=np.float64)
    ks = float(max(np.max(i / n - f), np.max(f - (i - 1) / n)))
    crit = _KS_CRIT_001 / math.sqrt(n)
    ok = bool(ks <= crit)
    note = (
        f"KS D = {ks:.4f} against a 1% critical value of {crit:.4f} (n = {n}). "
        + ("The fitted curve is consistent with these data."
           if ok else
           "THE FITTED CURVE DOES NOT DESCRIBE THESE DATA — the largest gap between the "
           "model and the empirical distribution is far beyond chance. Its parameters "
           "should not be reported as measurements.")
    )
    return {"ks": ks, "ks_critical_001": float(crit), "describes_data": ok,
            "n": n, "grid_mass": total, "note": note}


def _stamp_gof(fit: FitResult, sample) -> FitResult:
    """Run :func:`goodness_of_fit` on a finished fit and record the verdict IN IT.

    THE VERDICT BELONGS TO THE FIT, NOT TO A PANEL. :func:`goodness_of_fit` was
    written family-agnostic and correct, and then called from exactly one place --
    ``gui/plots_panel.py``. The other fit surface, the Inspector's Fits box
    (``AnalysisPanel._on_fit_requested`` -> ``modules.fit_dwell`` ->
    ``StatsPanel.set_fit_text(FitResult.summary())``), went on printing the report's
    own quoted read-out on the same -700 mV data --

        tau_1  0.0198094  [0.01959, 0.02003]  s

    -- with no statistic, no rival, no warning, while the drawn curve in the other
    window was being told it does not describe the data. That is High-3 surviving in
    the surface it was not tested on.

    Medium-11 and Low-20 got this right by putting their notes in the FITTER, where
    ``meta['warnings']`` reaches both surfaces. This does the same for the verdict:
    ``meta['gof']`` is stamped on every fit this module returns, :meth:`FitResult.summary`
    prints it, and the Plots panel reads the stamped one instead of recomputing.

    Never raises into a fit. A family whose density cannot be integrated over the
    sample still gets returned -- it just does not get a verdict, exactly as before.
    Measured cost: ~1 ms on a 1000-event sample, against the 8-200 ms the fit itself
    takes.

    Deliberately NOT appended to ``meta['warnings']``: the Plots panel already renders
    the verdict in full, and both surfaces reading one key means neither prints it
    twice. ``meta`` is the seam; ``warnings`` is one renderer's list.
    """
    try:
        gof = goodness_of_fit(fit, sample)
    except (ValueError, RuntimeError, TypeError, ZeroDivisionError, FloatingPointError):
        return fit
    fit.meta["gof"] = gof
    return fit


# ================================================================================================
# WHERE THE FIT MISSES, BIN BY BIN -- and the ONLY chi-square this module will print
# ================================================================================================
# :func:`goodness_of_fit` answers "does this curve describe these data" with ONE number.
# It cannot answer "and WHERE does it miss", which is the question a peaked family fit to
# a monotonic sample fails: KS D is dominated by the single largest CDF gap and says
# nothing about the near-zero bins that produced it. The residual strip the Plots panel
# draws under its histogram is that second question, and this is its arithmetic -- here
# rather than in the panel, for the reason goodness_of_fit is here: both fit surfaces
# print fits, and the answer must not depend on which window asked.
#
# THE CHI-SQUARE, AND WHY IT IS DELIBERATELY NOT A HEADLINE STATISTIC.
# The owner asked for one. What is offered is a REDUCED chi-square over the histogram's
# own bins, reported BESIDE the residuals it is the sum of squares of, and never as a
# rival to the KS verdict, because a Pearson chi-square on a dwell histogram is worse
# than KS in three ways this module can state exactly:
#
#   * IT IS BINNING-DEPENDENT. Move the Bins spin from 30 to 60 and chi-square changes
#     while the data, the fit and KS D do not. It is a statistic about a PICTURE.
#   * ITS LOW-COUNT BINS ARE NOT chi-squared AT ALL. The Pearson approximation needs an
#     expected count of roughly 5 per bin; a log-spaced dwell histogram routinely ends in
#     a decade of bins expecting 0.3 events each, where (obs-exp)^2/exp is a lottery.
#     Hence the pooling below -- which changes the bins, and therefore the statistic,
#     again.
#   * ITS DOF IS NOT THE TEXTBOOK VALUE. The textbook (bins - 1 - p) holds when the
#     parameters were estimated by MINIMISING THIS chi-square on THESE bins. Every fit in
#     this module comes from an unbinned, truncated maximum-likelihood fit instead, and
#     Chernoff-Lehmann says the true null distribution then sits BETWEEN chi2(bins-1-p)
#     and chi2(bins-1) -- so a p-value from either is wrong in a known direction and none
#     is offered. This is the same objection :func:`goodness_of_fit` already states about
#     its own KS p-value ("a decision, not a probability"), applied to the other test.
#
# What survives all three is what is actually useful: the standardised residuals
# themselves (a shape, read by eye, where a systematic run of same-sign bars is the
# finding) and their mean square as a single "how far off, in units of counting noise"
# scale factor. Reduced chi-square ~ 1 is consistent; 5 is not; 0.2 means the bins are
# pooled so hard the test has no resolution left. That is all it is claimed to mean.

#: The expected count a bin needs before ``(obs - exp)^2 / exp`` is even approximately
#: chi-squared. The conventional Cochran floor; bins below it are pooled with their
#: neighbours rather than dropped, because dropping them would discard real events and
#: quietly flatter the fit in exactly the tail where families differ most.
MIN_EXPECTED_PER_BIN = 5.0


def _n_free_params(fit: FitResult) -> int:
    """How many FREE parameters ``fit`` spent, from its own parameter names.

    ``len(fit.params)`` overcounts every mixture: a k-component fit stores k weights but
    only k-1 of them are free (they sum to one). Counted from the names rather than from
    a per-family table so a family added later is right by default, and stated here
    because the residual dof is the only place in this module where the difference
    changes a printed number.
    """
    keys = list(fit.params)
    weights = sum(1 for p in keys if p.startswith("w_") or p == "w_exp")
    return max(1, len(keys) - (1 if weights >= 2 else 0))


def _pool_to_floor(expected: np.ndarray, min_expected: float) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` groups of raw bins, each expecting >= ``min_expected``.

    Left to right, folding any leftover tail into the LAST group -- so the final group can
    exceed the floor but none falls below it, which is the direction that matters (a
    below-floor bin is the one whose chi-square term is meaningless).
    """
    if min_expected <= 0:
        return [(i, i + 1) for i in range(expected.size)]
    groups: list[tuple[int, int]] = []
    start, acc = 0, 0.0
    for i, e in enumerate(expected):
        acc += float(e)
        if acc >= min_expected:
            groups.append((start, i + 1))
            start, acc = i + 1, 0.0
    if start < expected.size:
        if groups:
            groups[-1] = (groups[-1][0], expected.size)
        else:
            groups = [(0, expected.size)]
    return groups


def binned_residuals(fit: FitResult, sample, edges,
                     *, min_expected: float = MIN_EXPECTED_PER_BIN) -> dict[str, Any]:
    """Standardised residuals of ``fit`` against ``sample`` on ``edges``, plus their chi².

    THE STRIP AND THE NUMBER ARE ONE OBJECT. ``residual`` is what a residual panel draws;
    ``chi2`` is the sum of its squares and ``reduced_chi2`` that over the residual degrees
    of freedom. Returning both from one call is what keeps the printed number and the
    drawn bars describing the same bins -- a chi-square computed over a different pooling
    than the picture shows is a number nobody can check by eye.

    The expected counts are CONDITIONED ON THE BINNED RANGE, the same convention
    :func:`goodness_of_fit` uses: the model's density is renormalised over
    ``[edges[0], edges[-1]]`` and multiplied by the number of observations that landed
    there. So this asks "does this shape describe this histogram", not "does this model
    also predict the right amount of mass off the edge of the picture" -- ``grid_mass``
    reports the latter for anyone who wants it.

    Args:
        fit: Any :class:`FitResult` carrying a density.
        sample: The observations, in the fit's own units (seconds for dwell).
        edges: The histogram's bin edges, ascending, length >= 3. These are the bins the
            picture shows; pooling below only merges them, never re-cuts them.
        min_expected: The Cochran floor bins are pooled up to; ``0`` disables pooling and
            gives one group per drawn bin (useful for a strip that must line up with the
            bars exactly, at the cost of a meaningless chi-square).

    Returns:
        ``{'edges', 'centers', 'observed', 'expected', 'residual', 'chi2', 'dof',
        'reduced_chi2', 'n_params', 'n_bins', 'n_bins_raw', 'pooled', 'n', 'grid_mass',
        'min_expected', 'note'}``. ``edges``/``centers``/``observed``/``expected``/
        ``residual`` describe the POOLED bins, so ``chi2 == sum(residual**2)`` exactly.
        ``dof`` is ``None`` when pooling left too few groups to spend the parameters on,
        and ``reduced_chi2`` is then ``None`` too -- an under-determined reduced
        chi-square is arithmetic, not evidence, exactly as :func:`r2_dof_caveat` says of
        an R² with no residual dof.

    Raises:
        ValueError: fewer than two finite observations, fewer than two bins, non-ascending
            edges, or a model placing no probability mass on the binned range.
    """
    x = np.asarray(sample, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError(f"need >= 2 finite observations to residual a fit, got {x.size}")
    e = np.asarray(edges, dtype=np.float64).ravel()
    if e.size < 3 or not np.all(np.diff(e) > 0):
        raise ValueError("edges must be >= 3 strictly ascending bin boundaries")

    observed = np.histogram(x, bins=e)[0].astype(np.float64)
    n_in = float(observed.sum())
    if n_in < 2:
        raise ValueError("fewer than 2 observations fall inside the given bin edges")

    # The same integration grid goodness_of_fit uses, for the same reason: log where the
    # data span decades (dwell), linear otherwise (depths), so the trapezoid gets a
    # comparable number of points per unit of probability either way. Sampled densely and
    # interpolated at the edges rather than integrated bin by bin, so a narrow first bin
    # of a log histogram is not represented by two points.
    lo, hi = float(e[0]), float(e[-1])
    span = hi / lo if lo > 0 else 0.0
    grid = (np.geomspace(lo, hi, 4001) if span >= 10.0 else np.linspace(lo, hi, 4001))
    dens = np.asarray(fit.predict(grid), dtype=np.float64)
    dens = np.where(np.isfinite(dens), dens, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    total = float(cum[-1])
    if not (total > 0):
        raise ValueError(
            f"{fit.model} places no probability mass on [{lo:g}, {hi:g}]; "
            f"its residuals against this histogram are undefined")
    at_edges = np.interp(e, grid, cum)
    expected_raw = np.diff(at_edges) / total * n_in

    groups = _pool_to_floor(expected_raw, float(min_expected))
    pooled_obs = np.array([observed[a:b].sum() for a, b in groups])
    pooled_exp = np.array([expected_raw[a:b].sum() for a, b in groups])
    pooled_edges = np.array([e[a] for a, _ in groups] + [e[-1]])
    # Guard the degenerate group a pathological density can still produce (every bin of
    # a group expecting ~0). Its residual is not defined; report it as 0 rather than as
    # an inf that would dominate the strip's scale and hide every real deviation.
    safe = np.where(pooled_exp > 0, pooled_exp, np.nan)
    residual = np.where(pooled_exp > 0, (pooled_obs - pooled_exp) / np.sqrt(safe), 0.0)
    chi2 = float(np.sum(residual ** 2))

    n_params = _n_free_params(fit)
    # bins - 1 - p: one constraint for having normalised the model to the OBSERVED count
    # (which is what conditioning on the binned range did), p for the fitted parameters.
    # See the section header for why this is called approximate rather than exact.
    dof_raw = len(groups) - 1 - n_params
    dof = dof_raw if dof_raw >= 1 else None
    reduced = (chi2 / dof) if dof else None

    centers = 0.5 * (pooled_edges[:-1] + pooled_edges[1:])
    pooled_any = len(groups) != expected_raw.size
    if dof is None:
        note = (
            f"Reduced χ² is unavailable here: pooling to an expected count of "
            f"{float(min_expected):g} left {len(groups)} bin(s) and the model spends "
            f"{n_params}, so no residual degrees of freedom remain. The residuals are "
            f"still drawn; the summary number would be arithmetic, not evidence.")
    else:
        note = (
            f"Reduced χ² = {reduced:.2f} ({chi2:.1f} over {dof} dof) across {len(groups)} "
            f"bin(s)"
            + (f", pooled from {expected_raw.size} drawn bins to keep every expected count "
               f"at or above {float(min_expected):g}" if pooled_any else "")
            + ". THE DOF IS APPROXIMATE: these parameters came from an unbinned truncated "
              "MLE, not from minimising this χ², so the null distribution lies between "
              f"χ²({len(groups) - 1 - n_params}) and χ²({len(groups) - 1}) and no p-value "
              "is reported. Read it as a scale — about 1 is consistent with counting "
              "noise, and the shape of the residuals above says more than the number.")
    return {"edges": pooled_edges, "centers": centers,
            "observed": pooled_obs, "expected": pooled_exp, "residual": residual,
            "chi2": chi2, "dof": dof, "reduced_chi2": reduced,
            "n_params": n_params, "n_bins": len(groups), "n_bins_raw": int(expected_raw.size),
            "pooled": bool(pooled_any), "n": int(n_in), "grid_mass": total,
            "min_expected": float(min_expected), "note": note}


# ------------------------------------------------------------------------------------------------
# CONFIDENCE BANDS -- and what it cost to be able to draw one
# ------------------------------------------------------------------------------------------------
# PER-PARAMETER INTERVALS WERE ALREADY HERE; A BAND WAS NOT, and the band cannot be
# reconstructed from the intervals. Drawing the density at (tau_lo) and at (tau_hi) is not
# a 95% band: for a two-parameter family it is a corner of the parameter box, the
# parameters are correlated (measured on the owner's -700 mV sample: the gamma fit's k and
# theta correlate at r = -0.97 across replicates, so the box's corners are exactly the
# combinations the data rule OUT), and for a mixture the corner need not even be a valid
# density. The honest band is pointwise percentiles of the REPLICATE CURVES, which needs
# the replicates.
#
# WHAT RETAINING THEM COSTS, since they were previously reduced to two percentiles and
# thrown away:
#   * MEMORY, per fit: n_boot x n_params float64. The defaults are 1000 replicates for the
#     exponential mixture (8 columns at k=4 -> 64 kB), 400 for the depth mixture (12
#     columns at k=4 -> 38 kB), 200 for the peaked families (2 columns -> 3.2 kB). Against
#     the FitResult already carrying an eta grid of up to a few thousand floats, and
#     against the event frames in the same window, this is noise.
#   * PERSISTENCE, which is the real cost and is why it is NOT paid: ``fit_to_json``
#     serialises ``meta`` wholesale, so retaining naively would write 8000 JSON numbers per
#     stored fit into every saved project -- a project with a dozen fits growing by
#     megabytes of resampled parameter noise that nothing reads back. ``fit_to_json`` drops
#     the key instead. A reopened fit therefore has intervals and no band, which is the
#     right trade: the band is a redrawing convenience, and a refit rebuilds it in the time
#     the fit already takes.
#   * TIME: none. The replicates are already computed; this is the difference between
#     keeping an array that exists and letting it fall out of scope.

#: ``meta`` key holding the raw bootstrap replicates as an ``(n_ok, n_params)`` array, and
#: the parameter names its columns are in. Dropped by :func:`fit_to_json` -- see above.
BOOT_KEY = "boot"
BOOT_PARAMS_KEY = "boot_params"


def ci_band(fit: FitResult, x, *, level: float = 0.95) -> dict[str, Any] | None:
    """A pointwise confidence band for ``fit``'s density at ``x``, or ``None``.

    ``None`` -- never an exception, and never a fabricated band -- whenever the replicates
    are not there to build one from: a fit restored from a project (see :func:`fit_to_json`),
    a fit made under :func:`point_fits_only`, a family that offers no bootstrap
    (:func:`fit_dwell_exp_gamma`, whose replicate is a 12-start optimisation), or one whose
    bootstrap produced too few usable rows. The caller draws a band or does not; it must
    never draw a band that is really the corners of a parameter box.

    Each retained replicate's parameter vector is turned back into a density by
    :func:`density_from` -- THE one implementation, the same call the fitter itself used --
    under the fit's own censoring, and the band is the pointwise percentile envelope of
    those curves. So the band is in the same units, and normalised the same way, as the
    curve the panel already draws.

    Args:
        fit: A fit whose ``meta`` still carries its replicates.
        x: Where to evaluate, in the fit's own units.
        level: Central mass of the band (0.95 -> the 2.5/97.5 percentiles), matching the
            per-parameter intervals in :attr:`FitResult.ci`.

    Returns:
        ``{'x', 'lo', 'hi', 'level', 'n_replicates', 'note'}`` or ``None``.
    """
    rows = fit.meta.get(BOOT_KEY)
    names = fit.meta.get(BOOT_PARAMS_KEY)
    family = fit.meta.get("family")
    if rows is None or not names or family not in FIT_FAMILIES:
        return None
    boot = np.asarray(rows, dtype=np.float64)
    if boot.ndim != 2 or boot.shape[0] < 20 or boot.shape[1] != len(names):
        # The same >= 20 floor the per-parameter intervals use: below it the percentiles
        # are the extremes of a handful of curves, which is a picture of the resampler
        # rather than of the uncertainty.
        return None
    grid = np.asarray(x, dtype=np.float64).ravel()
    if grid.size == 0:
        return None
    if not (0 < float(level) < 1):
        raise ValueError(f"level must lie in (0, 1), got {level!r}")

    curves = []
    for row in boot:
        params = {str(k): float(v) for k, v in zip(names, row)}
        try:
            curves.append(np.asarray(density_from(str(family), params, fit.meta)(grid),
                                     dtype=np.float64))
        except (KeyError, ValueError, TypeError, FloatingPointError, ZeroDivisionError):
            continue     # a replicate that will not rebuild is dropped, never guessed at
    if len(curves) < 20:
        return None
    stack = np.vstack(curves)
    stack = np.where(np.isfinite(stack), stack, np.nan)
    tail = 100.0 * (1.0 - float(level)) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # an all-NaN column is answered below
        lo = np.nanpercentile(stack, tail, axis=0)
        hi = np.nanpercentile(stack, 100.0 - tail, axis=0)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi), hi, 0.0)
    return {"x": grid, "lo": lo, "hi": hi, "level": float(level),
            "n_replicates": int(stack.shape[0]),
            "note": (f"{float(level) * 100:g}% pointwise band from {int(stack.shape[0])} "
                     f"bootstrap replicates of the whole parameter vector — POINTWISE, so "
                     f"it is the uncertainty of the curve AT each x, not a simultaneous "
                     f"band the whole curve lies inside with that probability.")}


#: The dwell families this module can fit, as ``label -> (kind, kwargs)``, in the order a
#: comparison should list them. The labels are the ones the Plots panel's own Fit combo
#: shows, so a recommendation names something the user can actually select -- a note that
#: recommends a model the menu does not contain is worse than no note.
DWELL_MODEL_LABELS: dict[str, tuple[str, dict[str, Any]]] = {
    "exponential": ("exp", {}),
    "gamma (peaked)": ("peaked", {"model": "gamma"}),
    "inverse-gaussian (first passage)": ("peaked", {"model": "inverse-gaussian"}),
    "exponential + gamma (bimodal)": ("exp-gamma", {}),
}


def compare_dwell_models(
    dwell,
    *,
    t_min: float,
    t_max: float | None = None,
    n_components: int = 1,
    families: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Fit every dwell family to the SAME sample under the SAME censoring; rank by BIC.

    This is the comparison the app could always have made and never did. Its whole value
    is in the two words "same" above: BIC is comparable only between fits to identical
    data under an identical censoring model, so every family here is handed the same
    array, the same ``t_min`` and the same ``t_max``, and none of them may resolve its own.

    Args:
        dwell: Observed dwell times in seconds.
        t_min: The detector's floor, in seconds.
        t_max: Optional right truncation.
        n_components: Component count for the exponential mixture (the only family here
            that takes one). Others ignore it.
        families: Restrict to these labels (keys of :data:`DWELL_MODEL_LABELS`).

    Returns:
        One dict per family -- ``{'label', 'bic', 'aic', 'n', 'fit', 'error'}`` -- sorted
        with the lowest (best) BIC first, and every family that REFUSED kept in the list
        with ``bic = None`` and its message in ``error``. A family that cannot fit these
        data is a result, not an omission: silently dropping it is how the
        inverse-Gaussian's infinite BIC went unnoticed for a whole run.

    Cost:
        Point fits only -- this wraps itself in :func:`point_fits_only`, because a
        comparison reads ``bic`` and nothing else. Measured on the owner's -700 mV run
        (n = 986): 8 ms gamma, 6 ms inverse-Gaussian, 200 ms exponential+gamma, against
        several seconds if each had bootstrapped CIs nobody would look at.

    Cancellation:
        Polled between families AND inside every fitter it calls, because this is one
        of the two things the Plots window's advisory worker runs and the panel aborts
        that worker on every curation click with a 100 ms bounded join. It used to poll
        NOTHING: `point_fits_only` switches off the bootstrap loops, which were the only
        place in this module that read the token, so a comparison ran to completion --
        the join burned its whole budget and detached a still-running, GIL-holding
        worker into the next click. Measured on the owner's -700 mV run at a realistic
        600 ms curation cadence, `exponential` on screen: 177 ms per click before,
        against an 85 ms no-fit control.
    """
    labels = list(DWELL_MODEL_LABELS) if families is None else [str(f) for f in families]
    unknown = [f for f in labels if f not in DWELL_MODEL_LABELS]
    if unknown:
        raise ValueError(
            f"unknown dwell family/families {unknown}; known: {list(DWELL_MODEL_LABELS)}")
    out: list[dict[str, Any]] = []
    with point_fits_only():
        for label in labels:
            check_cancelled()
            kind, kwargs = DWELL_MODEL_LABELS[label]
            row: dict[str, Any] = {"label": label, "bic": None, "aic": None,
                                   "n": None, "fit": None, "error": None}
            try:
                with quiet_runtime_warnings():
                    if kind == "exp":
                        fit = fit_dwell_exponential(
                            dwell, t_min=t_min, t_max=t_max,
                            n_components=int(n_components))
                    elif kind == "peaked":
                        fit = fit_dwell_peaked(dwell, t_min=t_min, t_max=t_max, **kwargs)
                    else:
                        fit = fit_dwell_exp_gamma(dwell, t_min=t_min, t_max=t_max)
                if not np.isfinite(fit.bic):
                    raise RuntimeError(f"{label} returned a non-finite BIC ({fit.bic})")
            # THE TRAP THIS EXISTS FOR: OperationCancelled IS a RuntimeError, so the
            # blanket clause below would file a cancelled run as "this family refused"
            # -- printing `refused: cancelled by request` in the user's MODEL COMPARISON
            # table, and carrying on to the next family instead of stopping.
            except OperationCancelled:
                raise
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                row["error"] = str(exc)
            else:
                row.update(bic=float(fit.bic), aic=float(fit.aic), n=int(fit.n), fit=fit)
            out.append(row)
    out.sort(key=lambda r: (r["bic"] is None, r["bic"] if r["bic"] is not None else 0.0))
    return out
