"""H3c — the poll that reaches INSIDE a minimisation, on the sample that needs it.

THE DEFECT, AND WHY THE PREVIOUS FIX WAS NOT ONE.

The Plots window aborts its advisory worker on every curation click and joins it for
``worker_thread.DEFAULT_JOIN_MS`` = 100 ms. Aborting is cooperative, so every kernel the
worker runs polls the ambient cancel token. H3b put that poll in the multi-start fitters
— **once per start** — and costed it on one sample: ~37 ms per start against a 100 ms
join, an order of magnitude inside it.

A poll between units of work is a bound only while a unit is short. On the owner's
-600 mV apo-ferritin recording (n = 240 after the 10 ms merge — the sample embedded
below, byte for byte out of his own merged table) **one Nelder-Mead start is ~790 ms**
and the twelve are 9.5 s, against 207 ms for all twelve on -700 mV (n = 986) — 43x the
cost on a quarter of the events, because the exponential+gamma likelihood has a shallow
ridge on this sample and the simplex grinds toward its 20000-iteration cap. The poll
therefore sat EIGHT TIMES further apart than the join it was meant to keep, and the
follow-up pass measured the consequence through a real window: two aborts in three blew
the 100 ms join and detached a still-running, GIL-holding worker into the next click.

    -600 mV, n = 240:  one advisory batch = 48703 ms   (join budget 100 ms)
      trial 1: abort blocked the GUI for 311.8 ms   detached 0 -> 1
      trial 2: abort blocked the GUI for 255.0 ms   detached 0 -> 1
      trial 3: abort blocked the GUI for   0.3 ms   detached 0 -> 0

THE INVARIANT THESE TESTS HOLD THE MODULE TO:

    **A bounded join is a promise about the worst case. The cancel poll must be
    reachable INSIDE one minimisation, not merely between them — because the length of
    one minimisation is a property of the DATA, and no sample gets to be a special
    case.**

Two of the assertions below are counting arguments rather than timings, on purpose: a
poll per start reads the token exactly twelve times per fit no matter how fast the
machine is, and a poll per iteration reads it thousands of times. That distinction is
what is being tested, and it cannot be flaked away by a fast or a loaded CPU.

Reverted by: removing ``stats.cancel_callback`` and the ``callback=`` arguments in
``fit_dwell_exp_gamma`` and ``fit_dwell_peaked``.
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
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope

#: The panel's own bounded join (``worker_thread.DEFAULT_JOIN_MS``).
JOIN_MS = 100.0

#: How many Nelder-Mead starts ``fit_dwell_exp_gamma`` runs: 4 shapes x 3 weights.
N_STARTS = 12

#: The owner's -600 mV apo-ferritin dwell times, seconds, n = 240 — the merged table
#: this project froze as ``reference/apo_-600mV__click1/events_after_merge.csv``, at
#: full precision. EMBEDDED rather than read from his report tree, because a test that
#: skips when a directory is missing is a test that reports "the fix is unreachable"
#: on every other machine — and this is the ONE sample on which the defect reproduces.
APO_600_DWELL_S = np.array([
    0.071119998304367, 0.0253599993953704, 0.0217599994812011, 0.0190199995465278, 
    0.0262399993743896, 0.0245399994149208, 0.0250199994034767, 0.0245599994144439, 
    0.0246399994125366, 0.0236999994349479, 0.0248999994063377, 0.0233799994425773, 
    0.0229999994516372, 0.0209199995012283, 0.0196199995322227, 0.0204799995117187, 
    0.0209599995002746, 0.0210599994978904, 0.0208399995031356, 0.0204599995121955, 
    0.0180999995684623, 0.0204999995112419, 0.0194599995360374, 0.01915999954319, 
    0.0173599995861053, 0.0182199995656013, 0.0184599995598793, 0.0196399995317459, 
    0.0179999995708465, 0.0171799995903968, 0.0177599995765686, 0.0195199995346069, 
    0.0197199995298385, 0.0187799995522499, 0.0198199995274543, 0.0198199995274543, 
    0.0182599995646476, 0.018179999566555, 0.0195999995326995, 0.0235799994378089, 
    0.0200399995222091, 0.01915999954319, 0.0184399995603561, 0.0214999994874, 
    0.0190599995455741, 0.0185999995565414, 0.0214399994888305, 0.0217999994802475, 
    0.0189599995479583, 0.0175799995808601, 0.0151599996385574, 0.0182599995646476, 
    0.0150999996399879, 0.0183399995627403, 0.0191999995422363, 0.0215999994850158, 
    0.0192799995403289, 0.0188999995493888, 0.017939999572277, 0.0179199995727539, 
    0.0146999996495246, 0.0175799995808601, 0.0188999995493888, 0.0150999996399879, 
    0.0151399996390342, 0.0184399995603561, 0.0178599995741844, 0.0179999995708465, 
    0.0172399995889663, 0.0155799996285438, 0.0194799995355606, 0.0172399995889663, 
    0.0188999995493888, 0.0185399995579719, 0.0184599995598793, 0.0155399996294975, 
    0.0173799995856285, 0.0175199995822906, 0.0137799996714591, 0.0179999995708465, 
    0.0195199995346069, 0.0167199996013641, 0.0179199995727539, 0.0187199995536804, 
    0.0176399995794296, 0.0185599995574951, 0.0196799995307922, 0.020759999505043, 
    0.0167599996004104, 0.0193999995374679, 0.0167999995994567, 0.0187399995532035, 
    0.0181399995675086, 0.018059999569416, 0.0160599996170997, 0.0179599995718002, 
    0.0174799995832443, 0.0151199996395111, 0.0158599996218681, 0.017319999587059, 
    0.0170999995923042, 0.0180799995689392, 0.0146999996495246, 0.0141399996628761, 
    0.0158599996218681, 0.0163799996094703, 0.0143999996566772, 0.0166199996037483, 
    0.0166799996023178, 0.0155399996294975, 0.018919999548912, 0.0173999995851516, 
    0.0149399996438026, 0.0189799995474815, 0.0192999995398521, 0.0193999995374679, 
    0.0185999995565414, 0.0195799995331764, 0.0188199995512962, 0.0191399995436668, 
    0.0169199995965957, 0.0180799995689392, 0.0175799995808601, 0.0174599995837211, 
    0.0162399996128082, 0.0175199995822906, 0.0172599995884895, 0.017939999572277, 
    0.0177599995765686, 0.0166399996032714, 0.0158799996213912, 0.0172399995889663, 
    0.0171799995903968, 0.0177399995770454, 0.016459999607563, 0.0185199995584487, 
    0.0163799996094703, 0.0168799995975494, 0.0170199995942115, 0.0163199996109008, 
    0.0165599996051788, 0.016219999613285, 0.0174599995837211, 0.0185399995579719, 
    0.0175599995813369, 0.0186999995541572, 0.0177399995770454, 0.0171399995913505, 
    0.0149399996438026, 0.0157399996247291, 0.0148799996452331, 0.0144199996562004, 
    0.0149599996433258, 0.0151599996385574, 0.0132399996843338, 0.0180199995703697, 
    0.0149599996433258, 0.0160599996170997, 0.0165199996061325, 0.0175799995808601, 
    0.0169399995961189, 0.0161799996142387, 0.0152599996361732, 0.5194999876141548, 
    0.0193799995379447, 0.4711999887657165, 0.0223399994673728, 0.0165999996042251, 
    0.0172199995894432, 0.0177599995765686, 0.0177799995760917, 0.0186999995541572, 
    0.0195799995331764, 0.01953999953413, 0.0199799995236396, 0.0211199994964599, 
    0.0188399995508193, 0.0211799994950294, 0.0185799995570182, 0.0212799994926452, 
    0.0186399995555877, 0.0192599995408058, 0.0191799995427131, 0.0192599995408058, 
    0.0209999994993209, 0.0226599994597435, 0.0184399995603561, 0.0195599995336532, 
    0.0206599995074272, 0.0191799995427131, 0.0188399995508193, 0.0191799995427131, 
    0.0204199995131492, 0.0191399995436668, 0.0188599995503425, 0.0202399995174407, 
    0.0184399995603561, 0.0180799995689392, 0.0194399995365142, 0.0189799995474815, 
    0.0204999995112419, 0.0174799995832443, 0.0149999996423721, 0.0168799995975494, 
    0.017079999592781, 0.0161999996137619, 0.0179599995718002, 0.016959999595642, 
    0.0185999995565414, 0.0184199995608329, 0.0199399995245933, 0.0188999995493888, 
    0.0187399995532035, 0.0171399995913505, 0.0234799994401931, 0.0190599995455741, 
    0.0194799995355606, 0.0198599995265007, 0.0188199995512962, 0.0176599995789527, 
    0.0182599995646476, 0.0194599995360374, 0.0150999996399879, 0.0180399995698928, 
    0.0196999995303154, 0.0201999995183944, 0.0184599995598793, 0.0194799995355606, 
    0.018299999563694, 0.0186399995555877, 0.0174599995837211, 0.0185999995565414, 
    0.0193799995379447, 0.0215599994859695, 0.0182199995656013, 0.0151999996376037, 
    0.0203599995145797, 0.0156199996275901, 0.0194599995360374, 0.2337199944276809,
], dtype=np.float64)

#: The detector's floor on that recording, as the window resolves it (observed_min).
APO_600_TMIN = float(APO_600_DWELL_S.min())


class _CountingToken:
    """A cancel token that COUNTS its reads and can be armed to trip on the Nth.

    The whole argument in one object. ``fit_dwell_exp_gamma`` polls once per start
    (12 reads, forever) or once per Nelder-Mead ITERATION (thousands). Counting reads
    tells the two apart with no clock in the assertion.
    """

    def __init__(self, trip_after: int | None = None) -> None:
        self.reads = 0
        self._trip = trip_after

    def __call__(self) -> bool:
        self.reads += 1
        return self._trip is not None and self.reads > self._trip


# ================================================================================================
# where is this tree?
# ================================================================================================

def test_the_tree_under_test_is_named_out_loud():
    """A revert run from inside the repo silently tests the LIVE tree: ``PYTHONPATH``
    does not beat cwd when cwd contains the package. So the package says where it came
    from and the runner declares which root it MEANT to exercise."""
    import inspect

    root = Path(nanoscope.__file__).resolve().parent.parent
    expected = Path(os.environ.get(
        "NANOSCOPE_EXPECT_ROOT", "/Users/abrahamcampos/Desktop/nanoscope")).resolve()
    assert root == expected, f"`nanoscope` resolved from {root}, not {expected}"
    assert Path(inspect.getsourcefile(stats)).resolve().is_relative_to(root)


def test_the_embedded_sample_is_the_one_that_breaks_the_join():
    """The fixture is load-bearing, so it is checked rather than trusted: 240 events,
    a 13.2 ms floor, an 18.4 ms median and a 0.52 s tail. A sample that quietly stopped
    being the owner's would make every timing below meaningless AND still green."""
    t = APO_600_DWELL_S
    assert t.size == 240
    assert t.min() == pytest.approx(0.0132399996843338, rel=0, abs=1e-15)
    assert float(np.median(t)) == pytest.approx(0.0184399995603561, rel=0, abs=1e-15)
    assert t.max() == pytest.approx(0.5194999876141548, rel=0, abs=1e-15)


# ================================================================================================
# 1 — the counting argument: the poll is INSIDE the minimisation
# ================================================================================================

def test_the_exp_gamma_poll_is_read_far_more_often_than_once_per_start():
    """THE MECHANISM, counted. A poll between starts reads the token exactly
    :data:`N_STARTS` times however long a start takes; a poll inside the minimisation
    reads it once per Nelder-Mead iteration.

    Uses a small well-conditioned sample so the test costs milliseconds: the COUNT is
    the claim, and it does not depend on the sample being pathological.
    """
    rng = np.random.default_rng(4)
    t = np.r_[np.abs(rng.normal(4e-4, 8e-5, 40)), np.abs(rng.normal(1.9e-2, 2e-3, 260))]
    token = _CountingToken()
    with cancel_scope(token), stats.quiet_runtime_warnings():
        stats.fit_dwell_exp_gamma(t, t_min=float(t.min()))
    assert token.reads > 20 * N_STARTS, (
        f"the cancel token was read {token.reads} times across {N_STARTS} starts "
        f"({token.reads / N_STARTS:.1f} per start). A poll that is only reachable "
        f"between starts cannot keep a {JOIN_MS:.0f} ms join on a sample where one "
        f"start is longer than that.")


def test_the_peaked_families_poll_inside_their_minimisations_too():
    """``gamma`` and ``inverse-gaussian`` have the same structure one level down: a
    point fit, then 200 bootstrap replicates, each its own Nelder-Mead. One replicate
    is 3-8 ms on -700 mV but 25-46 ms on the owner's -600 mV sample — half the join,
    on the recording that already broke the between-units assumption."""
    for model in ("gamma", "inverse-gaussian"):
        token = _CountingToken()
        with cancel_scope(token), stats.quiet_runtime_warnings():
            stats.fit_dwell_peaked(APO_600_DWELL_S, model=model, t_min=APO_600_TMIN,
                                   n_boot=12)
        assert token.reads > 20 * 12, (
            f"{model}: token read {token.reads} times over 12 replicates — the poll is "
            f"still only between replicates")


def test_an_armed_token_stops_the_first_start_rather_than_the_twelfth():
    """The same counting argument, made to bite. A token armed to trip on read 40 can
    only be seen inside a minimisation: a per-start poll never reaches read 40, so the
    fit would run to completion and return a FitResult."""
    token = _CountingToken(trip_after=40)
    with cancel_scope(token), stats.quiet_runtime_warnings():
        with pytest.raises(OperationCancelled):
            stats.fit_dwell_exp_gamma(APO_600_DWELL_S, t_min=APO_600_TMIN)
    assert token.reads == 41, (
        f"stopped on read {token.reads}; expected the very next poll after the token "
        f"was armed")


# ================================================================================================
# 2 — the timing, on the owner's own -600 mV sample
# ================================================================================================

@pytest.fixture(scope="module")
def apo600_uncancelled():
    """One honest, uncancelled ``exponential + gamma`` fit on the owner's -600 mV
    dwells: its wall time, and the FitResult every other test compares against.

    Module-scoped because it is the expensive thing in this file (seconds), and every
    assertion that needs it needs the SAME one.
    """
    t0 = time.perf_counter()
    with stats.quiet_runtime_warnings():
        fit = stats.fit_dwell_exp_gamma(APO_600_DWELL_S, t_min=APO_600_TMIN)
    return (time.perf_counter() - t0) * 1000.0, fit


def test_one_start_on_this_sample_is_longer_than_the_whole_join(apo600_uncancelled):
    """THE PREMISE OF THE WHOLE FIX, re-measured rather than quoted. If this ever stops
    being true the sample has stopped exercising the defect, and the tests below would
    pass for the wrong reason — so it fails loudly instead of skipping."""
    ms, _fit = apo600_uncancelled
    per_start = ms / N_STARTS
    assert per_start > JOIN_MS, (
        f"one Nelder-Mead start on the owner's -600 mV dwells took {per_start:.0f} ms "
        f"(whole fit {ms:.0f} ms), which is INSIDE the {JOIN_MS:.0f} ms join. This "
        f"sample no longer reproduces the defect and these tests no longer prove "
        f"anything about it.")


def test_the_abort_is_observed_inside_the_join_on_the_600_mv_sample():
    """What the follow-up measured through the real window — 311.8 ms and 255.0 ms of
    blocked GUI, two detached workers in three trials — reduced to its kernel."""
    # HOW LONG THE UNCANCELLED FIT ACTUALLY TAKES ON THIS MACHINE, measured now rather
    # than assumed. This used to sleep a FIXED 50 ms before setting the token, on the
    # strength of the 311.8 ms and 255.0 ms in the docstring above. That premise expired:
    # after 8 GB of stale processes and two runaway Julia jobs were cleared off this Mac
    # (2026-08-31) the same fit finished in UNDER 50 ms, so the token was set after the
    # fit had already returned and the trial failed with 'RAN TO COMPLETION' -- reporting
    # a machine that got faster as a kernel that stopped polling. A fixed wall-clock
    # offset into a workload whose duration is not fixed is the bug; measure and scale.
    with stats.quiet_runtime_warnings():
        _t0 = time.perf_counter()
        stats.fit_dwell_exp_gamma(APO_600_DWELL_S, t_min=APO_600_TMIN)
        solo_ms = (time.perf_counter() - _t0) * 1000.0
    # 30% in: comfortably past the first start on any machine, and comfortably before the
    # end on a fast one. Floored so a sub-millisecond fit still gets a real offset.
    cut_s = max(solo_ms * 0.30, 1.0) / 1000.0

    dts: list[float] = []
    for trial in range(3):
        flag = threading.Event()
        started = threading.Event()
        out: dict[str, object] = {}

        def run() -> None:
            started.set()
            try:
                with cancel_scope(flag.is_set), stats.quiet_runtime_warnings():
                    stats.fit_dwell_exp_gamma(APO_600_DWELL_S, t_min=APO_600_TMIN)
                out["v"] = "RAN TO COMPLETION"
            except OperationCancelled:
                out["v"] = "stopped"

        th = threading.Thread(target=run)
        th.start()
        assert started.wait(30.0)
        time.sleep(cut_s)         # ~30% into the fit, wherever that falls today
        t0 = time.perf_counter()
        flag.set()
        th.join(120.0)
        dt = (time.perf_counter() - t0) * 1000.0
        assert out.get("v") == "stopped", (
            f"trial {trial + 1}: the fit {out.get('v')!r} under a token set "
            f"{cut_s * 1000:.1f} ms in (the uncancelled fit measures {solo_ms:.1f} ms "
            f"on this machine) -- a kernel that polls its token would have stopped")
        dts.append(dt)
    # >= 2 of 3 trials inside the join, not every trial: the defect this gate exists to
    # catch (a kernel that does not poll its token) blew the join on TWO of three trials
    # -- 311.8 ms and 255.0 ms, the numbers in this test's own docstring -- so requiring
    # two clean trials still fails it outright. A single trial pushed past 100 ms by an
    # OS scheduler hiccup, though, is noise about the machine, not evidence about the
    # kernel: this gate failed exactly that way once in a full-suite chunk (passed alone
    # and on the chunk rerun, 2026-08-23). One bad trial passes; two cannot be a hiccup.
    inside = sum(dt < JOIN_MS for dt in dts)
    assert inside >= 2, (
        f"the cancel was observed outside the {JOIN_MS:.0f} ms join the panel promises "
        f"on {3 - inside} of 3 trials ({', '.join(f'{d:.1f} ms' for d in dts)}) -- "
        f"a population, not a scheduler hiccup: the kernel is not polling its token")


# ================================================================================================
# 3 — and NOTHING it computes moved
# ================================================================================================

def test_the_poll_changes_when_a_fit_stops_and_never_what_it_returns(apo600_uncancelled):
    """``callback=`` is observational: scipy's Nelder-Mead calls it in the ``finally``
    of its own loop and only ``StopIteration`` halts anything. So the fit under an armed
    (never-tripping) token must be BIT-IDENTICAL to the fit with no token at all —
    params, log-likelihood, AIC, BIC and the stamped KS verdict."""
    _ms, plain = apo600_uncancelled
    token = _CountingToken()
    with cancel_scope(token), stats.quiet_runtime_warnings():
        polled = stats.fit_dwell_exp_gamma(APO_600_DWELL_S, t_min=APO_600_TMIN)
    assert token.reads > N_STARTS, "the token was never polled inside a start"
    assert polled.params == plain.params
    assert (polled.loglik, polled.aic, polled.bic, polled.n) == (
        plain.loglik, plain.aic, plain.bic, plain.n)
    assert polled.meta.get("gof") == plain.meta.get("gof")


def test_the_peaked_families_keep_their_intervals_bit_for_bit():
    """The bootstrap is the half a poll could plausibly disturb — 200 resamples off one
    seeded generator. Same CI, same ``n_boot_ok``, token or no token."""
    for model in ("gamma", "inverse-gaussian"):
        with stats.quiet_runtime_warnings():
            plain = stats.fit_dwell_peaked(APO_600_DWELL_S, model=model,
                                           t_min=APO_600_TMIN, n_boot=40)
        token = _CountingToken()
        with cancel_scope(token), stats.quiet_runtime_warnings():
            polled = stats.fit_dwell_peaked(APO_600_DWELL_S, model=model,
                                            t_min=APO_600_TMIN, n_boot=40)
        assert polled.params == plain.params, model
        assert polled.ci == plain.ci, model
        assert polled.meta.get("n_boot_ok") == plain.meta.get("n_boot_ok"), model
        assert (polled.bic, polled.aic, polled.loglik) == (
            plain.bic, plain.aic, plain.loglik), model


def test_no_token_means_no_callback_and_therefore_no_cost():
    """Off a worker there is nothing to poll, and the fitter must be handed exactly the
    arguments it was handed before this fix existed. ``cancel_callback()`` answers
    ``None`` outside a scope, which is what ``minimize(callback=None)`` means."""
    assert stats.cancel_callback() is None
    with cancel_scope(_CountingToken()):
        assert callable(stats.cancel_callback())
    assert stats.cancel_callback() is None


def test_a_cancelled_fit_leaves_nothing_behind():
    """Half of what "cancel" means. ``OperationCancelled`` unwinds out of
    ``minimize`` — scipy catches only ``StopIteration`` — so no partial ``best`` is
    ever packed into a FitResult and handed back as an answer."""
    token = _CountingToken(trip_after=25)
    with cancel_scope(token), stats.quiet_runtime_warnings():
        with pytest.raises(OperationCancelled):
            stats.fit_dwell_exp_gamma(APO_600_DWELL_S, t_min=APO_600_TMIN)
