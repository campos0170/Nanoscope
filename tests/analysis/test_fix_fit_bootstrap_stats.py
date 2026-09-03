"""H3 -- the bootstrap CI of a DRAWN fit must be movable off the interaction path.

WHAT THIS PINS, AND WHY IT IS A STATS TEST AND NOT ONLY A GUI ONE.

Report finding #10 asked for confidence intervals on ``gamma`` and
``inverse-gaussian``: they used to return ``ci = {}``, so on the owner's apo-ferritin
data the only fits carrying error bars were the ones the KS test rejects. They were
added -- and that re-created finding #3, because a CI that is REPORTED cannot hide
inside ``stats.point_fits_only()`` the way a rival's discarded one can. 200 Nelder-Mead
replicates therefore landed on the curation click path for exactly the two families that
had just gained them. Measured through a real MainWindow on the owner's -700 mV run
(n = 984), median of 12 real checkbox clicks: **507 ms** with ``gamma (peaked)`` on
screen and **1101 ms** with ``inverse-gaussian (first passage)``, against 44 ms with the
fit box set to none. The budget is 150 ms.

The invariant is NOT "make the bootstrap cheaper" and NOT "hide it behind a gate" -- both
have been tried, and moving the same cost to a longer gate or a 400 ms timer is what
finding #3 was the first time. It is:

    **A fit has two parts with two latency budgets. The point estimate belongs on the
    interaction path; the interval belongs on a worker; and moving WHEN the interval is
    computed must not move WHAT it is.**

So these tests assert both halves. The deferral must actually defer (the point fit alone
is measurably a fraction of the cost), and the interval it hands back must be
BIT-IDENTICAL to the one the synchronous call produces -- same seed, same starting
simplex, same sample, same order -- on the owner's real -700 mV merged dwell times.

Reverted by: restoring ``nanoscope/analysis/stats.py`` and
``nanoscope/gui/plots_panel.py`` to their pre-fix state.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nanoscope
from nanoscope.analysis import stats
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope

#: The owner's own merged event tables, as the 2026-08-22 battery left them.
APO = Path("/Users/abrahamcampos/Desktop/nanoscope-reports/apo-run-2026-08-22")
#: The detector's real resolution on these runs (50 kHz, min_dwell_s = 20 us).
APO_TMIN = 20e-6


def _apo_dwells(run: str = "apo_-700mV") -> np.ndarray:
    path = APO / run / "events_after_merge.csv"
    if not path.exists():
        pytest.skip(f"apo-ferritin merged event set not available: {path}")
    df = pd.read_csv(path)
    d = df.loc[df["accepted"].astype(bool), "dwell_s"].to_numpy(dtype=np.float64)
    return d[np.isfinite(d) & (d > 0)]


@pytest.fixture(scope="module")
def apo700():
    return _apo_dwells()


# ================================================================================================
# WHICH TREE IS THIS? Ask before believing anything else in the file.
# ================================================================================================

def test_the_tree_under_test_is_named_out_loud():
    """A revert stub with the wrong signature once manufactured four fake failures on
    this project and was believed. So: resolve the package, say where it came from, and
    let the runner declare which root it MEANT to test.

    ``NANOSCOPE_EXPECT_ROOT`` is what a revert run sets to its own out-of-tree copy; with
    it unset this pins the working tree, exactly as
    ``test_plots_fit_honesty.test_the_tree_under_test_is_the_real_one`` does.
    """
    root = Path(nanoscope.__file__).resolve().parent.parent
    expected = Path(os.environ.get(
        "NANOSCOPE_EXPECT_ROOT", "/Users/abrahamcampos/Desktop/nanoscope")).resolve()
    assert root == expected, (
        f"the `nanoscope` under test resolved from {root}, not {expected}. Every "
        f"assertion below is about a tree, and this one says which.")
    # ...and the two files this fix lives in come from that same root, not from a
    # half-installed egg-link or a stale sys.path entry.
    import inspect
    from nanoscope.gui import plots_panel
    for mod in (stats, plots_panel):
        assert Path(inspect.getsourcefile(mod)).resolve().is_relative_to(root), mod


# ================================================================================================
# HUNK 1 -- stats grew a way to take the point fit now and the interval later
# ================================================================================================

def test_stats_offers_a_deferral_and_it_is_off_by_default():
    for name in ("deferred_ci", "ci_deferred", "apply_ci"):
        assert hasattr(stats, name), (
            f"stats.{name} is missing: without it the CI of a DRAWN fit has nowhere to "
            f"go but the click path, which is report finding #3")
        assert name in stats.__all__, f"stats.{name} is not exported"
    assert stats.ci_deferred() is False, "deferral must never be the ambient default"
    with stats.deferred_ci():
        assert stats.ci_deferred() is True
    assert stats.ci_deferred() is False, "the context manager did not restore"


def test_a_peaked_fit_inside_the_deferral_returns_its_estimate_and_no_interval(apo700):
    """The point half is complete and final; the interval half is explicitly PENDING.

    ``ci_pending`` is the load-bearing part. An empty ``ci`` with nothing beside it is
    what ``gamma``/``inverse-gaussian`` returned BEFORE finding #10, and it read as "this
    family reports no uncertainty" -- so a deferral that merely blanked the interval
    would re-open the defect it is standing next to.
    """
    for model in ("gamma", "inverse-gaussian"):
        with stats.deferred_ci() as jobs:
            fit = stats.fit_dwell_peaked(apo700, model=model, t_min=APO_TMIN)
        assert len(jobs) == 1, (model, "the bootstrap was not handed back")
        assert callable(jobs[0])
        assert fit.ci == {}, model
        assert fit.meta["ci_pending"] is True, (
            f"{model}: an interval that has not been computed must SAY so, or an empty "
            f"ci is indistinguishable from a family that offers none")
        assert fit.meta["n_boot_ok"] == 0, model
        assert np.isfinite(fit.bic) and np.isfinite(fit.loglik), model


def test_the_deferred_interval_is_bit_identical_to_the_synchronous_one(apo700):
    """THE NUMBER MUST NOT MOVE. Only when it is computed may move.

    ``repr`` on the floats, not ``approx``: this fix is allowed to change latency and
    nothing else, and "close enough" is how a fit drifts one bit per round.
    """
    for model in ("gamma", "inverse-gaussian"):
        full = stats.fit_dwell_peaked(apo700, model=model, t_min=APO_TMIN)
        with stats.deferred_ci() as jobs:
            part = stats.fit_dwell_peaked(apo700, model=model, t_min=APO_TMIN)
        stats.apply_ci(part, jobs[0]())

        assert {k: repr(v) for k, v in part.params.items()} == \
               {k: repr(v) for k, v in full.params.items()}, model
        assert repr(part.bic) == repr(full.bic), model
        assert repr(part.aic) == repr(full.aic), model
        assert repr(part.loglik) == repr(full.loglik), model
        assert {k: (repr(a), repr(b)) for k, (a, b) in part.ci.items()} == \
               {k: (repr(a), repr(b)) for k, (a, b) in full.ci.items()}, (model, part.ci)
        assert part.ci, model
        assert part.meta["n_boot_ok"] == full.meta["n_boot_ok"] == 200, model
        assert "ci_pending" not in part.meta, model
        assert part.meta["warnings"] == full.meta["warnings"], model
        # ...and the goodness-of-fit verdict the panel prints beside it, too.
        assert repr(part.meta["gof"]["ks"]) == repr(full.meta["gof"]["ks"]), model


def test_the_point_half_really_is_the_cheap_half(apo700):
    """The deferral must DEFER, not merely relabel.

    Measured on this sample: 3.9 ms for the gamma point fit against 975 ms with the
    interval, 14.7 ms against 1634 ms for inverse-gaussian. The assertion is 10x, i.e.
    two orders of magnitude of headroom, so a loaded machine cannot make it lie -- but a
    revert, where the bootstrap still runs inside the call, cannot pass it.
    """
    for model in ("gamma", "inverse-gaussian"):
        t0 = time.perf_counter()
        with stats.deferred_ci() as jobs:
            stats.fit_dwell_peaked(apo700, model=model, t_min=APO_TMIN)
        point_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        stats.fit_dwell_peaked(apo700, model=model, t_min=APO_TMIN)
        full_ms = (time.perf_counter() - t0) * 1000.0
        assert len(jobs) == 1
        assert point_ms * 10 < full_ms, (
            f"{model}: the deferred call cost {point_ms:.1f} ms against {full_ms:.1f} ms "
            f"for the full one -- the bootstrap is still running on the caller's thread")


def test_point_fits_only_still_wins_inside_the_deferral(apo700):
    """A caller that will not read the CI at all must not be handed a job to run.

    The rival scans are inside ``point_fits_only``; if the deferral appended a job for
    them too, the panel would faithfully compute -- on a worker, politely -- 200
    replicates whose answer is discarded on the next line. That is finding #3 wearing a
    thread."""
    with stats.point_fits_only(), stats.deferred_ci() as jobs:
        fit = stats.fit_dwell_peaked(apo700, model="gamma", t_min=APO_TMIN)
    assert jobs == []
    assert fit.ci == {}
    assert fit.meta["n_boot_ok"] == 0
    assert "ci_pending" not in fit.meta


def test_the_resampling_loop_asks_the_switch_itself(apo700):
    """The handed-back callable can be invoked anywhere -- including a worker thread,
    whose fresh context restores every ContextVar default. So the gate lives WITH the
    loop, not only at the site that built it."""
    with stats.deferred_ci() as jobs:
        stats.fit_dwell_peaked(apo700, model="gamma", t_min=APO_TMIN)
    with stats.point_fits_only():
        payload = jobs[0]()
    assert payload["ci"] == {} and payload["n_boot_ok"] == 0


# ================================================================================================
# HUNK 2 -- a superseded interval stops INSIDE its own loop
# ================================================================================================

def test_the_deferred_bootstrap_polls_the_ambient_cancel_token(apo700):
    """Cancellation is ambient, never a kwarg -- and it must reach the replicate loop,
    or a superseded CI runs the full 200 minimisations before anyone can drop it."""
    with stats.deferred_ci() as jobs:
        stats.fit_dwell_peaked(apo700, model="inverse-gaussian", t_min=APO_TMIN)
    t0 = time.perf_counter()
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            jobs[0]()
    dt = (time.perf_counter() - t0) * 1000.0
    assert dt < 200.0, (
        f"the cancelled bootstrap took {dt:.0f} ms to notice; it is not polling the "
        f"token inside its loop (the whole run is ~1.6 s on this sample)")


def test_the_bootstrap_runs_where_it_is_called_not_where_it_was_built(apo700):
    """The deferral is only worth anything if the callable is portable."""
    with stats.deferred_ci() as jobs:
        fit = stats.fit_dwell_peaked(apo700, model="gamma", t_min=APO_TMIN)
    seen: dict = {}

    def run():
        seen["thread"] = threading.get_ident()
        seen["payload"] = jobs[0]()

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=120)
    assert not t.is_alive()
    assert seen["thread"] != threading.get_ident()
    stats.apply_ci(fit, seen["payload"])
    assert set(fit.ci) == {"k", "theta"}
    assert fit.meta["n_boot_ok"] == 200


# ================================================================================================
# HUNK 3 -- what a pending fit may and may not do to the rest of the app
# ================================================================================================

def test_a_pending_fit_is_still_json_safe(apo700):
    """``meta`` is persisted. The bootstrap callable is handed back through the context
    manager and NEVER stored on the FitResult, because ``fit_to_json`` has to be able to
    serialise every fit this module returns -- including one drawn while its interval is
    still in flight."""
    import json
    with stats.deferred_ci():
        fit = stats.fit_dwell_peaked(apo700, model="gamma", t_min=APO_TMIN)
    blob = json.dumps(stats.fit_to_json(fit))
    assert "ci_pending" in blob
    back = stats.fit_from_json(json.loads(blob))
    assert repr(back.bic) == repr(fit.bic)
    assert back.meta["ci_pending"] is True


def test_apply_ci_records_a_failed_bootstrap_once(apo700):
    """A bootstrap that could not produce enough replicates must say so -- and a redraw
    that re-applies the same payload must not say it twice."""
    with stats.deferred_ci():
        fit = stats.fit_dwell_peaked(apo700, model="gamma", t_min=APO_TMIN)
    payload = {"ci": {}, "n_boot_ok": 3,
               "note": "bootstrap failed to produce enough replicates; CIs unavailable"}
    stats.apply_ci(fit, payload)
    stats.apply_ci(fit, payload)
    assert fit.ci == {}
    assert fit.meta["n_boot_ok"] == 3
    assert "ci_pending" not in fit.meta
    assert fit.meta["warnings"].count(payload["note"]) == 1


def test_nothing_changed_for_a_caller_that_does_not_opt_in(apo700):
    """The default path is untouched: same interval, same count, no pending flag."""
    fit = stats.fit_dwell_peaked(apo700, model="gamma", t_min=APO_TMIN, n_boot=40, seed=0)
    assert set(fit.ci) == {"k", "theta"}
    assert fit.meta["n_boot_ok"] == 40
    assert "ci_pending" not in fit.meta


def test_the_family_with_no_bootstrap_hands_nothing_back(apo700):
    """``exponential + gamma`` reports no CI by design (every replicate is a 12-start
    optimisation). A caller must not assume one job per fit."""
    with stats.deferred_ci() as jobs:
        fit = stats.fit_dwell_exp_gamma(apo700, t_min=APO_TMIN)
    assert jobs == []
    assert fit.ci == {}
    assert "ci_pending" not in fit.meta
