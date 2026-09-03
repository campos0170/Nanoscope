"""REP/UNITS — one pane, one unit: the stats side.

Two defects, both of them the module saying something untrue about numbers it computed
correctly, and both invisible to the tests that shipped with the code they are in.

1. **THE SAME PARAMETER, 1000x APART, TWO LINES APART.** ``fit_depth_gaussians`` words
   its identifiability notes from the RAW parameters while the window that shows them
   converts to the axis's unit, so one Details pane read

       mu = 0.1923, 1.923, 2.314 ms
       mu_3 = 0.002314 s has a 95% CI of [0.002127, 0.00256]

   -- and the interval INSIDE the note wore no unit at all while the value beside it
   said "s". The scale now travels with the value (``display=``).

   **THE SHIPPED UNIT TEST ASSERTS AT k = 1**, which is the one component count at
   which ``_identifiability_notes`` is never called: the pane has no second line to
   disagree with. Every unit test here is at k >= 2.

2. **A MAXIMISATION THAT HAD STOPPED MOVING WAS STILL SPENDING ITS BUDGET.** On the
   owner's -600 mV merged dwells, fit with the ``t_min`` the window GUESSES when a
   pushed table carries no detector provenance, the peaked families' Nelder-Mead reaches
   its answer at ~iteration 82 and then re-tests the same point 3,916 more times,
   because its step tolerance is applied to the LOG of each parameter and the optimum
   sits at the edge of the family (gamma shape k -> 0). 218 ms of a 220 ms fit, on the
   curation click path, for parameters that do not change.

Reverted by: ``display-factor-travels-with-the-value`` (stats.py) and
``nm-stall-guard`` (stats.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import stats

#: The owner's -600 mV apo-ferritin merged dwell times, seconds, n = 240 -- his own
#: one-click reference table
#: (``rank-2026-08-22/reference/apo_-600mV_oneclick_events_after_merge.csv``) at full
#: precision. EMBEDDED, because this is the sample on which the peaked maximisation
#: grinds, and a test that skips when a report directory is missing is a test that says
#: "the fix is unreachable" everywhere else.
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
    0.0203599995145797, 0.0156199996275901, 0.0194599995360374, 0.2337199944276809,])

#: The detector's own floor for that run, out of its provenance. With this the peaked
#: fits are well posed and converge in 68-83 iterations; it is the CONTROL for the
#: stall guard.
APO_600_TMIN = 2e-05


# ================================================================================================
# where is this tree?
# ================================================================================================

def test_the_tree_under_test_is_named_out_loud():
    """A revert run from inside the repo silently tests the LIVE tree: ``PYTHONPATH``
    does not beat cwd when cwd contains the package. Both files this pass changed are
    named, so a snapshot that forgot one cannot pass quietly."""
    import inspect
    import os
    from pathlib import Path

    import nanoscope
    from nanoscope.gui import plots_panel

    root = Path(nanoscope.__file__).resolve().parent.parent
    expected = Path(os.environ.get(
        "NANOSCOPE_EXPECT_ROOT", "/Users/abrahamcampos/Desktop/nanoscope")).resolve()
    assert root == expected, f"`nanoscope` resolved from {root}, not {expected}"
    assert Path(inspect.getsourcefile(stats)).resolve().is_relative_to(root)
    assert Path(inspect.getsourcefile(plots_panel)).resolve().is_relative_to(root)


# ================================================================================================
# 1 — one pane, one unit
# ================================================================================================

def _time_mixture(k: int, **kw):
    """A mixture over a TIME column, on the owner's OWN -600 mV dwells -- the sample
    the window was fitting when it printed the two contradicting lines. It leaves
    components unidentified at k = 2 and at k = 3, so there are notes to get wrong.

    Not synthetic: a well-separated invented mixture resolves cleanly, writes no
    identifiability note, and would leave this whole defect untested."""
    return stats.fit_depth_gaussians(APO_600_DWELL_S, n_components=k, quantity="dwell_s",
                                     unit="s", n_boot=120, seed=0, **kw)


def _mu_notes(fit) -> str:
    return " ".join(n for n in fit.meta["warnings"] if n.startswith(("UNIDENTIFIED",
                                                                    "COMPONENTS NOT")))


@pytest.mark.parametrize("k", [2, 3])
def test_the_notes_are_written_in_the_unit_the_window_will_show(k):
    """THE DEFECT, at the two strings that disagreed — at k = 2 AND k = 3, because the
    shipped test asserts at k = 1 where no note is written at all.

    Every ``mu_j`` inside a note must be the SAME NUMBER the estimate line shows, in
    the SAME unit, and the interval quoted beside it must be labelled too."""
    fit = _time_mixture(k, display=(1e3, "ms"))
    notes = _mu_notes(fit)
    assert notes, f"k={k} produced no identifiability note to check"
    for j in range(1, k + 1):
        mu = fit.params[f"mu_{j}"]
        if f"mu_{j} = " not in notes:
            continue
        assert f"mu_{j} = {mu * 1e3:.4g} ms" in notes, (
            f"the note states mu_{j} in the STORAGE unit while the window shows "
            f"{mu * 1e3:.4g} ms:\n  {notes}")
        assert f"mu_{j} = {mu:.4g} s" not in notes, (
            f"the raw-seconds spelling is still there:\n  {notes}")
        lo, hi = fit.ci[f"mu_{j}"]
        if f"mu_{j} = {mu * 1e3:.4g} ms has a 95% CI" in notes:
            assert f"[{lo * 1e3:.4g}, {hi * 1e3:.4g}] ms" in notes, (
                "the interval INSIDE the note is unlabelled, or in the other unit, "
                f"beside a value that is labelled:\n  {notes}")


def test_a_k1_fit_writes_no_note_at_all():
    """WHY THE SHIPPED TEST COULD NOT SEE THIS — pinned, so nobody re-derives the same
    blind spot. ``_identifiability_notes`` is called only for k > 1: at k = 1 there is
    no second line in the pane, so there is nothing for the estimate line to disagree
    with, and a unit test that samples only k = 1 is measuring the safe case.

    PASSES BOTH — it is a statement about the blind spot, not about the fix, so it is
    written without ``display`` and holds on either tree."""
    fit = _time_mixture(1)
    assert not _mu_notes(fit)
    assert not any(n.startswith("UNIDENTIFIED") for n in fit.meta["warnings"])


def test_the_weights_never_take_the_unit_the_means_took():
    """A weight is a SHARE of the sample. Scaling it by the display factor, or
    captioning it "ms", would be this defect pointing the other way — and it is the
    easy mistake to make when a factor is threaded through a formatter."""
    fit = _time_mixture(3, display=(1e3, "ms"))
    soft = [n for n in fit.meta["warnings"] if n.startswith("UNRESOLVED WEIGHT")]
    for note in soft:
        for j in range(1, 4):
            key = f"w_{j}"
            if key in fit.ci and f"{key} = " in note:
                assert f"{key} = {fit.params[key]:.3g} with" in note, note
        assert " ms" not in note, f"a weight was given a unit:\n  {note}"


def test_the_default_caller_still_gets_exactly_the_strings_it_got_before():
    """PASSES BOTH — a CONTROL, and the backward-compatibility guarantee. Every caller
    that knows nothing about display units (``modules.fit_depth``, the Fits panel, a
    notebook) passes no ``display`` and must see its own storage unit."""
    fit = _time_mixture(3)
    notes = _mu_notes(fit)
    for j in range(1, 4):
        if f"mu_{j} = " in notes:
            assert f"mu_{j} = {fit.params[f'mu_{j}']:.4g} s" in notes, notes


def test_no_number_moves_when_the_display_unit_does():
    """The display unit is a caption, not a conversion of the DATA. Params, intervals,
    likelihood, AIC/BIC and ``param_units`` are bit-identical; only the words move."""
    a = _time_mixture(3)
    b = _time_mixture(3, display=(1e3, "ms"))
    assert a.params == b.params
    assert a.ci == b.ci
    assert (a.loglik, a.aic, a.bic, a.n) == (b.loglik, b.aic, b.bic, b.n)
    assert a.meta["param_units"] == b.meta["param_units"] == {
        "mu_1": "s", "sigma_1": "s", "w_1": "",
        "mu_2": "s", "sigma_2": "s", "w_2": "",
        "mu_3": "s", "sigma_3": "s", "w_3": ""}, (
        "the STORAGE unit followed the display unit — every reader that converts "
        "(FitResult.unit_of, the Plots pane, the Inspector) would then convert twice")


def test_the_depth_floor_warning_says_what_its_four_magnitudes_are_in():
    """The same hole, in the other note this function writes: ``DEPTH FLOOR`` states a
    floor, a mu, a sigma and an edge, and stated all four with no unit — in a pane whose
    next line reads "μ = 16.88, 88.55, 103.1 pA"."""
    rng = np.random.default_rng(3)
    x = np.abs(rng.normal(16.0, 6.0, 300))
    fit = stats.fit_depth_gaussians(x, n_components=1, depth_floor=9.049,
                                    quantity="depth_pA", unit="pA", n_boot=0)
    floor = [n for n in fit.meta["warnings"] if n.startswith("DEPTH FLOOR")]
    assert floor, fit.meta["warnings"]
    note = floor[0]
    assert "above ~9.049 pA" in note, note
    for name, key in (("mu_1", "mu_1"), ("sigma_1", "sigma_1")):
        assert f"{name} = {fit.params[key]:.4g} pA" in note, note
    edge = fit.params["mu_1"] - 2 * fit.params["sigma_1"]
    assert f"mu_1 - 2*sigma_1 = {edge:.4g} pA" in note, note
    assert "% of its mass" in note, note


# ================================================================================================
# 2 — the maximisation that had stopped moving
# ================================================================================================

def _sample(flip: int) -> np.ndarray:
    """The -600 mV dwells with ONE event curated in or out: exactly what one curation
    click changes."""
    keep = np.ones(APO_600_DWELL_S.size, dtype=bool)
    keep[flip] = False
    return APO_600_DWELL_S[keep]


@pytest.mark.parametrize("model,flip", [("gamma", 0), ("gamma", 5),
                                        ("inverse-gaussian", 2)])
def test_the_search_that_cannot_converge_is_stopped_when_it_stops_moving(model, flip):
    """THE DEFECT, on the owner's own -600 mV sample, through the code path a table
    with no detector provenance takes (``t_min`` guessed as the shortest observed
    dwell) — which is the path every shipped budget test drives and the one the last
    report measured.

    Not "it is faster": the numbers must be THE SAME. The guarded run and the full
    4000-iteration run are compared bit for bit."""
    t = _sample(flip)
    guessed = float(t.min())
    with stats.point_fits_only():
        guarded = stats.fit_dwell_peaked(t, model=model, t_min=guessed)
    assert guarded.meta["stalled"] is True, (
        f"{model}/flip {flip} did not stall; this sample is the reason this guard "
        f"exists (nit = {guarded.meta['n_iter']})")
    assert guarded.meta["n_iter"] < stats._NM_MAXITER, guarded.meta["n_iter"]
    assert guarded.meta["n_iter"] <= stats._NM_STALL_ITERS + 200, (
        f"the guard let the search run {guarded.meta['n_iter']} iterations past a "
        f"point it had already stopped moving from")


@pytest.mark.parametrize("model,flip", [("gamma", 0), ("gamma", 5),
                                        ("inverse-gaussian", 2)])
def test_stopping_the_stalled_search_changes_no_number_it_reports(model, flip, monkeypatch):
    """The other half, and the one that makes the guard safe: the SAME fit with the
    guard effectively switched off (a stall limit no search can reach) returns the same
    parameters, the same likelihood and the same BIC — it just spends thousands of
    iterations re-testing the point it already had."""
    t = _sample(flip)
    guessed = float(t.min())
    with stats.point_fits_only():
        guarded = stats.fit_dwell_peaked(t, model=model, t_min=guessed)
        monkeypatch.setattr(stats, "_NM_STALL_ITERS", 10 ** 9)
        full = stats.fit_dwell_peaked(t, model=model, t_min=guessed)
    assert full.meta["stalled"] is False and full.meta["n_iter"] == stats._NM_MAXITER, (
        "the unguarded run did not reach the iteration cap, so this sample no longer "
        "demonstrates what the guard is for")
    assert guarded.params == full.params, (guarded.params, full.params)
    assert (guarded.loglik, guarded.aic, guarded.bic) == (full.loglik, full.aic, full.bic)
    assert guarded.meta["n_iter"] * 4 < full.meta["n_iter"], (
        f"guarded {guarded.meta['n_iter']} vs full {full.meta['n_iter']} iterations — "
        f"the guard is not saving the work it exists to save")


def test_a_stalled_search_is_never_reported_as_converged():
    """A search stopped because it was going nowhere has NOT met its tolerance. scipy
    reports a callback halt as ``success=False`` today; this asserts the fit's own
    answer, which is what every window reads."""
    t = _sample(0)
    with stats.point_fits_only():
        fit = stats.fit_dwell_peaked(t, model="gamma", t_min=float(t.min()))
    assert fit.meta["stalled"] is True
    assert fit.meta["converged"] is False
    note = [n for n in fit.meta["warnings"] if "STOPPED MOVING" in n]
    assert note, fit.meta["warnings"]
    assert "did NOT meet its convergence tolerance" in note[0]
    assert "no mode at all" in note[0], (
        "the note has to say what a shape at the edge of the family MEANS; "
        f"k = {fit.params['k']:.4g} is not a peak:\n  {note[0]}")


@pytest.mark.parametrize("model", ["gamma", "inverse-gaussian"])
def test_the_guard_never_fires_on_a_well_posed_fit(model):
    """PASSES BOTH — the CONTROL that makes the guard safe to have. With the detector's
    real ``t_min`` the same sample, curated twelve different ways, converges in under a
    hundred iterations every time: 200 unmoved iterations is ten times the widest
    stall a healthy search on these recordings has ever shown (measured: 20)."""
    with stats.point_fits_only():
        for flip in range(12):
            fit = stats.fit_dwell_peaked(_sample(flip), model=model, t_min=APO_600_TMIN)
            assert fit.meta.get("stalled", False) is False, (
                f"the stall guard fired on a HEALTHY {model} fit (flip {flip}, "
                f"nit = {fit.meta.get('n_iter')}) — it would be moving numbers")
            assert fit.meta.get("converged", True) is True
            assert fit.meta.get("n_iter", 0) < 150


def test_the_stall_guard_still_lets_cancellation_through(monkeypatch):
    """The guard shares ``minimize``'s one callback slot with the ambient cancel poll,
    so it is the thing that could swallow a cancel. It must not: an
    ``OperationCancelled`` raised by the poll leaves ``minimize`` unchanged."""
    from nanoscope.analysis.cancel import OperationCancelled, cancel_scope

    t = _sample(0)
    with cancel_scope(lambda: True):
        with pytest.raises(OperationCancelled):
            stats.fit_dwell_peaked(t, model="gamma", t_min=APO_600_TMIN, n_boot=0)


def test_the_stall_guard_counts_only_a_vertex_that_did_not_move_at_all():
    """The guard's own unit test, on its own terms: a moving iterate resets the count,
    and only ``limit`` consecutive IDENTICAL iterates stop the search."""
    stop = stats._StallStop(None, limit=3)
    x = np.array([1.0, 2.0])
    for _ in range(2):
        stop(type("R", (), {"x": x})())
    stop(type("R", (), {"x": np.array([1.0, 2.0 + 1e-15])})())    # one ulp = movement
    assert stop.stopped is False
    for _ in range(3):
        last = type("R", (), {"x": np.array([1.0, 2.0 + 1e-15])})()
        if stop.stopped:
            break
        try:
            stop(last)
        except StopIteration:
            break
    assert stop.stopped is True, "three identical iterates did not stop the search"
