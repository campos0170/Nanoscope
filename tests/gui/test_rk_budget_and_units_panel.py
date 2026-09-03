"""H3c — the Plots window, through a real MainWindow: the click budget and the units.

Every string asserted here is read out of ``details_text()`` on a :class:`PlotsWindow`
that a real :class:`~nanoscope.gui.main_window.MainWindow` opened through its real
``Analysis > Plots...`` action, over events a real detection produced, with the event
loop drained. The last batch's four message defects were all invisible to a test that
builds the widget in isolation.

TWO DEFECTS ARE PINNED HERE.

1. **``exponential + gamma`` was over the click budget, and could not be brought under
   it where it was.** Its point estimate is a twelve-start Nelder-Mead — the whole cost,
   with no bootstrap to defer — and it ran ON THE GUI THREAD between a curation click
   and the repaint. Measured through a real window at a realistic 600 ms cadence,
   median of 8 real curation clicks against a 150 ms budget:

       -700 mV (n = 984)   239.3 ms   ->  5.9 ms
       -600 mV (n = 240)  4351.5 ms   ->  5.5 ms      (worst click 12.6 SECONDS)

   The -600 mV number is the point: the cost is a property of the DATA, not of the
   event count, so no amount of tuning makes a synchronous twelve-start fit fit inside
   a click. The maximisation goes onto the panel's own advisory worker, exactly as the
   peaked families' bootstraps already do, and the pane SAYS it is waiting.

   The mechanism is asserted, not just the milliseconds: a curation click must run NO
   ``fit_dwell_exp_gamma`` on the GUI thread. A test that only timed the click would go
   green on a fast machine with the defect still in place.

2. **A quantity printed in two units 1000x apart, with nothing on the page to say so.**
   ``_column_quantity`` answered ``""`` — "dimensionless" — for every column it did not
   recognise, which includes every TIME column in the schema. So a ``gaussian-mixture``
   fit to ``dwell_s`` printed ``mu_1 [0.01987, 0.02027]`` in bare, unlabelled seconds on
   the same window, the same axis and the same events where the ``exponential`` printed
   ``tau_1 [19.85, 20.27] ms``. Measured on the owner's -600 mV table, before and after:

       BEFORE   mu = 0.01844, 0.4078      mu_1 [0.01811, 0.01874]
       AFTER    mu = 18.44, 407.8 ms      mu_1 [18.11, 18.74] ms

Reverted by: ``expgamma-sync-fit`` (the deferral), ``betterk-all-families`` (the ladder
gate) and ``column-quantity-no-seconds`` (the unit).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import stats
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plots_panel import PlotsWindow, _column_quantity

FS = 100_000.0
_NONE = "none"
_EXP = "exponential"
_GAMMA = "gamma (peaked)"
_INVGAUSS = "inverse-gaussian (first passage)"
_EXP_GAMMA = "exponential + gamma (bimodal)"
_GMM = "gaussian-mixture"

#: One curation click, in milliseconds. The budget this window has been held to since
#: the deferral work began.
CLICK_BUDGET_MS = 150.0

#: The owner's -600 mV apo-ferritin dwell times, seconds, n = 240 — his own merged
#: table (``reference/apo_-600mV__click1/events_after_merge.csv``) at full precision.
#: EMBEDDED because this is the sample on which the defect is 63x the budget rather
#: than 1.6x, and a test that skips when a report directory is missing is a test that
#: says "the fix is unreachable" everywhere else.
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


# ================================================================================================
# a real recording, a real detection, the real Plots action
# ================================================================================================

def _plan(n_events: int, seed: int):
    """A trace with ``n_events`` blockades: a narrow ~2 ms mode plus a short spike.

    The SHAPE is load-bearing: ``exponential + gamma`` is only expensive on data with
    two lobes, and the rival scan is gated on the drawn fit being REJECTED by its KS
    test, so a well-specified fixture would leave both untested.
    """
    rng = np.random.default_rng(seed)
    spike = np.maximum(rng.normal(2.0e-4, 4e-5, n_events // 8), 8e-5)
    mode = np.maximum(rng.normal(2.0e-3, 2.5e-4, n_events - n_events // 8), 5e-4)
    dwells = np.r_[spike, mode]
    rng.shuffle(dwells)
    gap = 3_500
    n = gap * (n_events + 2)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.2, n)
    for i, d in enumerate(dwells):
        s = gap * (i + 1)
        x[s:s + max(int(round(d * FS)), 4)] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, n_events: int, seed: int, name: str) -> TraceDocument:
    arr = _plan(n_events, seed)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    if w._plots_window is not None:
        w._plots_window.close()
    w._analysis_controller.abort()


@pytest.fixture
def plots(window, qtbot, tmp_path):
    """MainWindow -> open a recording -> real detection -> real ``Analysis > Plots...``."""
    session = window._session
    doc = _doc(tmp_path, 240, 5, "apo_synth.dat")
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-5)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=120_000):
        panel.trigger_detection()
    qtbot.wait(30)

    window._sync_plots_action()
    assert window._act_plots.isEnabled(), "the real Plots action never lit up"
    window._act_plots.trigger()
    qtbot.wait(60)
    pp = window._plots_window
    assert isinstance(pp, PlotsWindow)
    pp.set_plot_type("Histogram")
    pp.set_histogram_column("dwell_s")
    qtbot.wait(30)
    return pp


def _drain(pp: PlotsWindow, qtbot, *, timeout_s: float = 240.0) -> None:
    """Settle every deferred note. A silent timeout is indistinguishable from a product
    regression, so it is never silent."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        qtbot.wait(10)
        if (not pp._pending_notes and not pp._note_timer.isActive()
                and not pp._inflight_notes):
            qtbot.wait(10)
            return
    raise AssertionError(
        f"the panel never settled: pending={sorted(k[0] for k in pp._pending_notes)} "
        f"inflight={sorted(k[0] for k in pp._inflight_notes)}")


def _apo600_frame() -> pd.DataFrame:
    """The owner's -600 mV merged dwell table, in the shape curation pushes."""
    n = APO_600_DWELL_S.size
    return pd.DataFrame({
        "dwell_s": APO_600_DWELL_S,
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _curate(pp: PlotsWindow, frame: pd.DataFrame, row: int) -> float:
    """One curation click — flip one event's ``accepted`` and push the table. ms."""
    d = frame.copy()
    j = d.index[row % len(d)]
    d.loc[j, "accepted"] = not bool(d.loc[j, "accepted"])
    t0 = time.perf_counter()
    pp.set_data({"apo": d})
    return (time.perf_counter() - t0) * 1000.0


# ================================================================================================
# where is this tree?
# ================================================================================================

def test_the_tree_under_test_is_named_out_loud():
    """A revert run from inside the repo silently tests the LIVE tree: ``PYTHONPATH``
    does not beat cwd when cwd contains the package."""
    import inspect

    from nanoscope.gui import plots_panel

    root = Path(nanoscope.__file__).resolve().parent.parent
    expected = Path(os.environ.get(
        "NANOSCOPE_EXPECT_ROOT", "/Users/abrahamcampos/Desktop/nanoscope")).resolve()
    assert root == expected, f"`nanoscope` resolved from {root}, not {expected}"
    assert Path(inspect.getsourcefile(plots_panel)).resolve().is_relative_to(root)
    assert Path(inspect.getsourcefile(stats)).resolve().is_relative_to(root)


# ================================================================================================
# 1 — no twelve-start optimisation on the click path
# ================================================================================================

def test_a_curation_click_runs_no_exp_gamma_maximisation_on_the_gui_thread(
        plots, qtbot, monkeypatch):
    """THE MECHANISM, not the milliseconds. Timing alone would go green on a fast
    machine with the defect intact; what must be true is that the GUI thread does not
    run this family's maximisation AT ALL between a click and the repaint."""
    pp = plots
    gui_thread = threading.current_thread()
    seen: list[str] = []
    real = stats.fit_dwell_exp_gamma

    def watched(*a, **k):
        seen.append("GUI" if threading.current_thread() is gui_thread else "worker")
        return real(*a, **k)

    monkeypatch.setattr("nanoscope.analysis.stats.fit_dwell_exp_gamma", watched)

    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    pp.set_fit(_EXP_GAMMA)
    qtbot.wait(5)
    seen.clear()

    _curate(pp, frame, 3)
    qtbot.wait(5)
    assert "GUI" not in seen, (
        f"a curation click ran {seen.count('GUI')} exponential+gamma maximisation(s) on "
        f"the GUI thread. On the owner's -600 mV sample one of those is 9.5 SECONDS.")

    _drain(pp, qtbot)
    assert "worker" in seen, (
        "the fit never ran anywhere: the deferral must MOVE the maximisation, not skip "
        "it — the user is owed the curve")
    assert "GUI" not in seen, f"the maximisation reached the GUI thread later: {seen}"


def test_the_click_budget_holds_on_the_sample_that_broke_it(plots, qtbot):
    """The measurement itself, on the owner's own -600 mV dwells: 4351.5 ms median per
    click before, 5.5 ms after, against a 150 ms budget."""
    pp = plots
    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    pp.set_fit(_EXP_GAMMA)
    _drain(pp, qtbot)

    clicks = [_curate(pp, frame, i) for i in range(1, 6)]
    median = statistics.median(clicks)
    assert median < CLICK_BUDGET_MS, (
        f"a curation click with '{_EXP_GAMMA}' on screen cost {median:.0f} ms "
        f"(all: {[round(c) for c in clicks]}) against a {CLICK_BUDGET_MS:.0f} ms budget")
    _drain(pp, qtbot)


def test_every_family_is_inside_the_budget_on_that_sample(plots, qtbot):
    """All six, because a budget that holds for one family is not a budget. The
    peaked families defer their bootstrap and the mixture is cheap; this is the guard
    that a later change cannot put any of them back on the click path."""
    pp = plots
    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    worst: dict[str, float] = {}
    for fam in (_NONE, _EXP, _GAMMA, _INVGAUSS, _EXP_GAMMA, _GMM):
        pp.set_fit(fam)
        _drain(pp, qtbot)
        clicks = [_curate(pp, frame, i) for i in range(10, 15)]
        worst[fam] = statistics.median(clicks)
        _drain(pp, qtbot)
    over = {k: round(v) for k, v in worst.items() if v >= CLICK_BUDGET_MS}
    assert not over, f"over the {CLICK_BUDGET_MS:.0f} ms click budget: {over}"


def test_the_pane_says_the_fit_is_coming_and_then_shows_it(plots, qtbot):
    """A curve that simply is not there reads as "this family has nothing to say about
    these events", which is the opposite of the truth. Both states, out of the real
    Details pane, with the loop drained between."""
    pp = plots
    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    pp.set_fit(_EXP_GAMMA)
    qtbot.wait(5)

    pending = pp.details_text()
    assert PlotsWindow._FIT_PENDING_TEXT in pending, (
        "the fit went off the click path without saying so; a silent empty plot is the "
        "failure this deferral would otherwise introduce:\n" + pending)
    assert PlotsWindow._FIT_PENDING_SUMMARY in pending, (
        "the summary line vanished instead of saying what it was waiting for:\n" + pending)
    assert "Fit unavailable" not in pending, (
        "'waiting' was printed as 'unavailable' — that is the sentence this window uses "
        "when a family REFUSES a sample, and the two must never be confused")

    _drain(pp, qtbot)
    settled = pp.details_text()
    fit = pp.fit_result()
    assert fit is not None, "the deferred maximisation never landed on the window"
    assert PlotsWindow._FIT_PENDING_TEXT not in settled, "still promising after it landed"
    assert PlotsWindow._FIT_PENDING_SUMMARY not in settled
    assert "BIC = {:.6g}".format(fit.bic) in settled, (
        "the promise resolved into nothing: the fit is on the window and printed "
        "nowhere:\n" + settled)
    assert "k = {:.4g}".format(fit.params["k"]) in settled, settled
    assert "no bootstrap CIs for this family" in settled, (
        "the family's own explanation of why it has no interval went missing on the way "
        "through the worker:\n" + settled)


def test_the_cached_fit_is_never_the_one_the_draw_mutates(plots, qtbot):
    """FOUND IN THIS FIX'S OWN VERIFY PASS. The deferred FitResult is CACHED, and this
    window MUTATES what it is handed: ``modules.stamp_fit_provenance`` PREPENDS the
    resolution's notes to ``meta['warnings']`` and is not idempotent, and ``apply_ci`` /
    ``ci_failed`` rewrite ``meta`` too. Without a copy on the way out of the cache, the
    second redraw over the same sample — a bin change, a time-unit toggle, a resize —
    prints every provenance warning twice, then three times, and the cache degrades a
    little more on every draw for the life of the window.

    The synchronous call this replaced could not have this defect: it built a fresh
    FitResult every draw. Deferring it is what created the possibility, so the guard
    belongs to the deferral.

    Driven over the fixture's OWN detected events, which carry real detector
    provenance — a bare pushed frame does not, and ``stamp_fit_provenance`` is skipped
    for it, so that version of this test would pass without proving anything.
    """
    pp = plots
    pp.set_histogram_column("dwell_s")
    pp.set_fit(_EXP_GAMMA)
    _drain(pp, qtbot)

    keys = [k for k in pp._note_cache if k[0] == "expgammafit"]
    assert keys, f"the maximisation was not cached at all: {sorted(pp._note_cache)}"
    cached = pp._note_cache[keys[0]]
    drawn = pp.fit_result()
    assert drawn is not None and drawn is not cached, (
        "the window is drawing the cached FitResult itself; every mutation the draw "
        "makes now outlives the draw")
    before = list(cached.meta.get("warnings", []))

    for bins in (34, 38, 42):
        pp.set_bins(bins)
        _drain(pp, qtbot)
    assert list(cached.meta.get("warnings", [])) == before, (
        "three redraws over the same sample grew the CACHED fit's warning list from "
        f"{len(before)} to {len(cached.meta.get('warnings', []))} entries")

    lines = [ln for ln in pp.details_text().splitlines() if ln.strip()]
    dupes = sorted({ln for ln in lines if lines.count(ln) > 1})
    assert not dupes, f"lines printed more than once after four redraws: {dupes}"


def test_a_family_with_no_component_count_runs_no_component_count_ladder(
        plots, qtbot, monkeypatch):
    """``exponential + gamma`` has a fixed parameterisation: its ``_refit`` takes ``kk``
    and ignores it, so every rung of the ladder was the same deterministic maximisation
    compared with itself — four more 9.5 s fits per draw on the owner's -600 mV sample,
    ~38 s of a 48.7 s advisory batch, and the ladder could not print anything either way.

    Counted, because the note it produces is empty in both worlds."""
    pp = plots
    calls = {"n": 0}
    real = stats.fit_dwell_exp_gamma

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr("nanoscope.analysis.stats.fit_dwell_exp_gamma", counted)

    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    pp.set_fit(_EXP_GAMMA)
    _drain(pp, qtbot)
    assert calls["n"] <= 2, (
        f"one settled draw ran {calls['n']} exponential+gamma maximisations. It needs "
        f"two: the drawn fit, and the one inside the cross-family BIC table. The rest "
        f"are a component-count ladder over a family that has no component count.")
    assert "COMPONENT COUNT" not in pp.details_text()


def test_the_ladder_still_runs_where_a_component_count_means_something(plots, qtbot):
    """PASSES BOTH — a CONTROL. The gate must not switch the recommender off for the
    families that DO have a component count, or the cheap fix loses the feature."""
    pp = plots
    rng = np.random.default_rng(3)
    dwell = np.r_[rng.exponential(3e-4, 300), rng.exponential(1.4e-2, 300)] + 5e-5
    pp.set_data({"apo": pd.DataFrame({
        "dwell_s": dwell,
        "accepted": np.ones(dwell.size, dtype=bool),
        "filter_limited": np.zeros(dwell.size, dtype=bool),
    })})
    pp.set_histogram_column("dwell_s")
    pp.set_fit(_EXP)
    _drain(pp, qtbot)
    assert "components fit markedly better" in pp.details_text(), pp.details_text()


# ================================================================================================
# 2 — the unit hole
# ================================================================================================

@pytest.mark.parametrize("col,unit", [
    ("dwell_s", "s"), ("t_start_s", "s"), ("rise_s", "s"), ("fall_s", "s"),
    ("ecd_pA_s", "pA·s"), ("depth_pA", "pA"), ("baseline_pA", "pA"),
    ("max_dev_pA", "pA"), ("mean_dev_pA", "pA"),
    ("frac_depth", ""), ("snr", ""), ("n_levels", ""),
])
def test_every_column_in_the_schema_reports_the_unit_it_is_actually_in(col, unit):
    """The whole column list, not ``dwell_s`` special-cased. Everything unrecognised
    used to answer ``""`` — "dimensionless" — and four of the schema's columns are
    seconds. ``ecd_pA_s`` is the trap: it ends in ``_s`` and is a charge."""
    assert _column_quantity(col)[1] == unit


def test_a_mixture_over_dwell_prints_the_same_unit_the_exponential_does(plots, qtbot):
    """THE DEFECT, at the two strings that disagreed. Same window, same column, same
    events, same time unit: ``exponential`` printed ``tau_1 [19.85, 20.27] ms`` and
    ``gaussian-mixture`` printed ``mu_1 [0.01987, 0.02027]`` — the same quantity, a
    thousandfold apart, with nothing on the page to say which was which."""
    pp = plots
    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    pp.set_time_unit("ms")

    pp.set_fit(_EXP)
    pp.set_fit_components(1)
    _drain(pp, qtbot)
    tau = float(pp.fit_result().params["tau_1"])
    tau_line = next(ln for ln in pp.details_text().splitlines() if ln.startswith("τ ="))
    assert "ms" in tau_line, tau_line

    pp.set_fit(_GMM)
    pp.set_fit_components(1)
    _drain(pp, qtbot)
    gmm = pp.fit_result()
    text = pp.details_text()
    mu_line = next(ln for ln in text.splitlines() if ln.startswith("μ ="))
    ci_line = next(ln for ln in text.splitlines() if ln.startswith("95% CI"))

    mu = float(gmm.params["mu_1"])
    assert "{:.4g} ms".format(mu * 1e3) in mu_line, (
        "the mixture printed its mean over a TIME column in raw, unlabelled seconds "
        "under an axis reading ms:\n  {}\nbeside the exponential's\n  {}".format(
            mu_line, tau_line))
    lo, hi = gmm.ci["mu_1"]
    assert "mu_1 [{:.4g}, {:.4g}] ms".format(lo * 1e3, hi * 1e3) in ci_line, (
        "the interval is not in the unit its own parameter is printed in:\n  " + ci_line)
    # ...and the two lines are now about numbers of the same size.
    assert 0.1 < (mu / tau) < 10.0, (
        "mu = {!r} and tau = {!r} are not the same quantity any more".format(mu, tau))


def test_a_dimensionless_column_is_still_not_given_a_unit(plots, qtbot):
    """PASSES BOTH — a CONTROL. ``frac_depth`` is a ratio (1.81-2.57 on these
    recordings, i.e. an ENHANCEMENT) and must never acquire a caption on the way
    through the same code path that just learned about seconds."""
    pp = plots
    n = 400
    rng = np.random.default_rng(9)
    pp.set_data({"apo": pd.DataFrame({
        "dwell_s": np.abs(rng.normal(2e-3, 3e-4, n)),
        "frac_depth": np.abs(rng.normal(2.1, 0.2, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })})
    pp.set_histogram_column("frac_depth")
    pp.set_fit(_GMM)
    _drain(pp, qtbot)
    text = pp.details_text()
    assert "Gaussian mixture on dI/I0" in text, text
    mu_line = next(ln for ln in text.splitlines() if ln.startswith("μ ="))
    assert " s" not in mu_line and "pA" not in mu_line, mu_line
