"""A FIT NOBODY READS MUST NOT PAY FOR CONFIDENCE INTERVALS.

Measured on the owner's ``500 pM Apo ferritin…_0006_Process_-600 mV.abf`` (ch 2, 317
events) through a real ``MainWindow`` with Analysis ▸ Plots open on a ``dwell_s``
histogram and an exponential fit — the configuration C-1 exists to keep correct:

=========================================  =========
one accept/reject click                    latency
=========================================  =========
Populations open, fit = none                  20 ms
Populations open, fit = exponential          3400 ms
Populations open, fit = exponential (fix)      74 ms
=========================================  =========

``cProfile`` of one click: 5.599 s of 5.636 s inside
``PlotsWindow._better_component_count``, 2003 ``stats._fit_mix`` calls driving 2030
``scipy.optimize.minimize`` runs. That method re-fits k = 1, 2, 3 solely to compare
``FitResult.bic``; it discards params, ci and meta. ``bic`` comes out of the likelihood
maximisation, BEFORE any resampling — so every one of those 2000 minimisations was a
bootstrap for confidence intervals thrown away on the next line.

WHY IT ONLY SURFACED NOW. While ``_refresh_surface_windows`` was wired to four signals,
the cost was paid on the user's own Refresh click. C-1 correctly wired it to the ten
signals that change what those windows draw, which moved it onto every curation click —
the click a curator makes hundreds of times in a row. The C-1 fix is right; the hint was
always this expensive and only now lands somewhere it is felt. Narrowing the signal set
back would restore a wrong number to fix a slow one.

THE SEAM. ``stats.point_fits_only()`` — one decision, made by the caller that knows
whether the CIs will be read, honoured by every bootstrap in ``stats``. Not an
``n_boot=0`` at each rival call site: there are three fit families reached through three
lambdas in ``_overlay_fit``, which is three places to forget and a fourth the day a
fourth family is added.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import stats


def _api(name: str):
    """Look the new API up LATE and say so by name when it is absent, so a run against a
    pre-fix package fails on this assertion rather than on a collection error."""
    obj = getattr(stats, name, None)
    assert obj is not None, f"nanoscope.analysis.stats.{name} does not exist"
    return obj


@pytest.fixture
def dwell() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.exponential(0.012, 320) + 5e-5


@pytest.fixture
def depth() -> np.ndarray:
    rng = np.random.default_rng(11)
    return np.r_[rng.normal(0.22, 0.03, 160), rng.normal(0.48, 0.04, 160)]


# ================================================================================================
# the number the caller reads is bit-identical
# ================================================================================================

def test_a_point_fit_reports_the_same_bic_as_a_bootstrapped_one(dwell):
    """PASSES BOTH — a CONTROL, and the whole justification. If BIC moved, the hint
    would be answering a different question cheaply, which is worse than answering the
    right one slowly."""
    point_fits_only = _api("point_fits_only")
    for k in (1, 2, 3):
        full = stats.fit_dwell_exponential(dwell, t_min=5e-5, n_components=k)
        with point_fits_only():
            cheap = stats.fit_dwell_exponential(dwell, t_min=5e-5, n_components=k)
        assert cheap.bic == full.bic, \
            f"k={k}: BIC moved {full.bic!r} -> {cheap.bic!r}"
        assert cheap.aic == full.aic
        assert cheap.loglik == full.loglik
        assert cheap.params == full.params
        assert cheap.n == full.n


def test_the_gaussian_family_is_covered_by_the_same_switch(depth):
    """`_better_component_count` runs over depth fits too — a seam that covered only the
    exponential would leave the amplitude histogram paying full price."""
    point_fits_only = _api("point_fits_only")
    full = stats.fit_depth_gaussians(depth, n_components=2)
    with point_fits_only():
        cheap = stats.fit_depth_gaussians(depth, n_components=2)
    assert cheap.bic == full.bic and cheap.params == full.params
    assert cheap.ci == {}
    assert full.ci, "the default must still produce CIs, or this test proves nothing"


def test_what_is_dropped_is_exactly_the_confidence_intervals(dwell):
    point_fits_only = _api("point_fits_only")
    with point_fits_only():
        cheap = stats.fit_dwell_exponential(dwell, t_min=5e-5, n_components=2)
    assert cheap.ci == {}
    assert cheap.meta["n_boot_ok"] == 0


def test_a_declined_bootstrap_is_not_reported_as_a_failed_one(dwell):
    """PASSES BOTH (the pre-fix code ran the bootstrap, so it did not warn either).
    `meta['warnings']` is shown to the user. "bootstrap failed to produce enough
    replicates" would be a false statement about a bootstrap that was never asked for."""
    point_fits_only = _api("point_fits_only")
    with point_fits_only():
        cheap = stats.fit_dwell_exponential(dwell, t_min=5e-5, n_components=2)
        gauss = stats.fit_depth_gaussians(np.abs(dwell) * 10, n_components=1)
    for fit in (cheap, gauss):
        assert not any("bootstrap failed" in w for w in fit.meta["warnings"]), \
            fit.meta["warnings"]


# ================================================================================================
# the cost actually goes away
# ================================================================================================

def test_a_point_fit_runs_no_bootstrap_at_all(dwell, monkeypatch):
    """THE REPRO, counted rather than timed — a wall-clock assertion on a shared machine
    is a flake. Before the fix a k=2 rival ran 1000 extra ``_fit_mix`` calls."""
    point_fits_only = _api("point_fits_only")
    calls = {"n": 0}
    real = stats._fit_mix

    def counted(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(stats, "_fit_mix", counted)

    calls["n"] = 0
    stats.fit_dwell_exponential(dwell, t_min=5e-5, n_components=2)
    full = calls["n"]

    calls["n"] = 0
    with point_fits_only():
        stats.fit_dwell_exponential(dwell, t_min=5e-5, n_components=2)
    cheap = calls["n"]

    assert full > 500, f"the default must still bootstrap ({full} fits)"
    assert cheap == 1, f"a point fit must be ONE maximisation, not {cheap}"


# ================================================================================================
# it cannot strip a CI off a fit somebody IS reading
# ================================================================================================

def test_the_default_is_confidence_intervals(dwell):
    """PASSES BOTH — the non-regression the switch could plausibly have broken."""
    ci_wanted = _api("ci_wanted")
    assert ci_wanted() is True
    assert stats.fit_dwell_exponential(dwell, t_min=5e-5).ci


def test_the_block_restores_the_previous_answer_even_when_it_raises(dwell):
    point_fits_only, ci_wanted = _api("point_fits_only"), _api("ci_wanted")
    with pytest.raises(RuntimeError):
        with point_fits_only():
            assert ci_wanted() is False
            raise RuntimeError("boom")
    assert ci_wanted() is True


def test_nesting_does_not_leak(dwell):
    point_fits_only, ci_wanted = _api("point_fits_only"), _api("ci_wanted")
    with point_fits_only():
        with point_fits_only():
            assert ci_wanted() is False
        assert ci_wanted() is False
    assert ci_wanted() is True


def test_it_does_not_reach_another_thread(dwell):
    """PASSES BOTH against a revert that removes the switch entirely; it discriminates a
    switch implemented as a module global, which is the shape this had to avoid.

    A ContextVar and not a module global, because fits run on worker threads too
    (``gui.worker_thread``). A flag that leaked across them would silently strip the CIs
    off a fit the user is reading, which is the one thing this must never do."""
    import threading

    point_fits_only, ci_wanted = _api("point_fits_only"), _api("ci_wanted")
    seen: list[bool] = []

    with point_fits_only():
        t = threading.Thread(target=lambda: seen.append(ci_wanted()))
        t.start()
        t.join()
    assert seen == [True], "the worker thread inherited a switch it never set"


# ================================================================================================
# the seam covers EVERY bootstrap in the module, not the two that were there today
# ================================================================================================

def test_every_resampling_loop_in_stats_reads_the_switch():
    """A third fit family added later must not quietly reopen this.

    ``stats`` resamples in exactly two places (``_bootstrap_dwell`` and the Gaussian
    mixture's own loop); ``fit_dwell_peaked`` and ``fit_dwell_exp_gamma`` have no
    bootstrap at all. This finds every ``rng.integers`` in the module and demands the
    function containing it consult :func:`ci_wanted`.
    """
    import ast
    import inspect
    from pathlib import Path

    src = Path(inspect.getsourcefile(stats)).read_text()
    tree = ast.parse(src)
    offenders = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        body = ast.get_source_segment(src, fn) or ""
        if "rng.integers(" in body and "ci_wanted()" not in body:
            offenders.append(fn.name)
    assert not offenders, (
        f"these functions resample without asking whether the CIs will be read: "
        f"{offenders}. Route them through ci_wanted() rather than adding a second "
        "switch")
