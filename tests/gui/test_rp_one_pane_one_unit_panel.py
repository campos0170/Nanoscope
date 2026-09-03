"""REP/UNITS — one pane, one unit, through a real MainWindow.

Every string here is read out of ``details_text()`` on a :class:`PlotsWindow` that a
real :class:`~nanoscope.gui.main_window.MainWindow` opened through its real
``Analysis > Plots...`` action, over events a real detection produced.

WHAT THE SHIPPED WIRED TESTS COULD NOT SEE, and this file therefore does:

* **k = 3, not k = 1.** ``tests/gui/test_rk_budget_and_units_panel.py`` drives the
  mixture at ``set_fit_components(1)`` -- the one component count at which
  ``stats._identifiability_notes`` is never called, so the pane has no second line to
  contradict the first. At k = 3 the same pane read

      μ = 18.16, 24.38, 110.7 ms
      UNIDENTIFIED COMPONENT(S): mu_1 = 0.01816 s has a 95% CI of [0.01373, 0.0184] …
      95% CI …: mu_1 [13.73, 18.4] ms …

  measured through this window on the owner's -600 mV run: the same parameter in two
  units 1000x apart on three consecutive lines, and the interval INSIDE the note with
  no unit at all.

* **p90 and max, not the median.** The shipped budget tests assert
  ``statistics.median`` of 5-8 clicks against 150 ms. On the -600 mV table the
  ``gamma`` and ``inverse-gaussian`` clicks are BIMODAL -- 12 ms or 220 ms, 3 of 12 and
  1 of 12 over budget with maxima of 223.8 ms and 507.0 ms -- and a median cannot see
  a mode that holds a minority of the sample. Quantiles a bimodal distribution cannot
  hide behind, plus the mechanism underneath them, which is not a timing at all.

* **the column list, not ``dwell_s``.** ``start_idx``/``end_idx`` are declared
  ``"sample"`` by ``events.SCHEMA`` and answered "dimensionless" here.

Reverted by: ``display-factor-travels-with-the-value`` and ``schema-units`` and
``nm-stall-guard``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import events as events_schema
from nanoscope.analysis import stats
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plots_panel import PlotsWindow, _column_quantity

FS = 100_000.0
_EXP = "exponential"
_GAMMA = "gamma (peaked)"
_INVGAUSS = "inverse-gaussian (first passage)"
_EXP_GAMMA = "exponential + gamma (bimodal)"
_GMM = "gaussian-mixture"
_NONE = "none"
ALL_FAMILIES = (_NONE, _EXP, _GAMMA, _INVGAUSS, _EXP_GAMMA, _GMM)

#: One curation click, in milliseconds. The budget this window has been held to since
#: the deferral work began.
CLICK_BUDGET_MS = 150.0

#: The owner's -600 mV apo-ferritin merged dwell times, seconds, n = 240 -- his own
#: reference table at full precision, the sample on which the peaked maximisation
#: grinds. EMBEDDED for the reason the sibling file embeds it: a test that skips when a
#: report directory is missing says "the fix is unreachable" everywhere else.
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
    0.0203599995145797, 0.0156199996275901, 0.0194599995360374, 0.2337199944276809,], dtype=np.float64)


# ================================================================================================
# a real recording, a real detection, the real Plots action
# ================================================================================================

def _plan(n_events: int, seed: int):
    """A trace with ``n_events`` blockades: a narrow ~2 ms mode plus a short spike."""
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


def _quantile(values, p: float) -> float:
    v = sorted(values)
    i = (len(v) - 1) * p
    a, b = int(i), min(int(i) + 1, len(v) - 1)
    return v[a] + (v[b] - v[a]) * (i - a)


# ================================================================================================
# where is this tree?
# ================================================================================================

def test_the_tree_under_test_is_named_out_loud():
    """A revert run from inside the repo silently tests the LIVE tree."""
    import inspect

    from nanoscope.gui import plots_panel

    root = Path(nanoscope.__file__).resolve().parent.parent
    expected = Path(os.environ.get(
        "NANOSCOPE_EXPECT_ROOT", "/Users/abrahamcampos/Desktop/nanoscope")).resolve()
    assert root == expected, f"`nanoscope` resolved from {root}, not {expected}"
    assert Path(inspect.getsourcefile(plots_panel)).resolve().is_relative_to(root)
    assert Path(inspect.getsourcefile(stats)).resolve().is_relative_to(root)


# ================================================================================================
# 1 — one pane, one unit, at the component count that has something to say
# ================================================================================================

def _mu_values(text: str) -> list[float]:
    """The numbers off the pane's own estimate line (``μ = 18.16, 24.38, 110.7 ms``)."""
    line = next(ln for ln in text.splitlines() if ln.startswith("μ ="))
    body = line.split("=", 1)[1].split("(")[0]
    return [float(tok) for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?",
                                             body.replace("ms", " ").replace("µs", " "))]


def test_a_mixture_at_k3_states_one_unit_for_one_parameter(plots, qtbot):
    """THE DEFECT. Same window, same events, same axis: the estimate line, the
    identifiability note and the interval line must all say the same number in the same
    unit for the same ``mu_j``.

    ONE STEP PAST THE SHIPPED TEST, which drives this at k = 1 — the component count at
    which no note is written and the contradiction cannot appear."""
    pp = plots
    pp.set_data({"apo": _apo600_frame()})
    pp.set_histogram_column("dwell_s")
    pp.set_time_unit("ms")
    pp.set_fit(_GMM)
    pp.set_fit_components(3)
    _drain(pp, qtbot)

    fit = pp.fit_result()
    text = pp.details_text()
    notes = " ".join(ln for ln in text.splitlines()
                     if ln.startswith(("UNIDENTIFIED", "COMPONENTS NOT")))
    assert notes, "k=3 on this sample wrote no identifiability note:\n" + text
    assert " s has a 95% CI" not in notes, (
        "the note is in raw seconds under an axis reading ms — the exact string the "
        f"pane contradicted itself with:\n  {notes}")

    for j in range(1, 4):
        mu = float(fit.params[f"mu_{j}"])
        if f"mu_{j} = " not in notes:
            continue
        assert f"mu_{j} = {mu * 1e3:.4g} ms" in notes, (
            f"mu_{j} is {mu * 1e3:.4g} ms on the estimate line and something else in "
            f"the note:\n  {notes}")
        lo, hi = fit.ci[f"mu_{j}"]
        if "has a 95% CI" in notes:
            assert f"[{lo * 1e3:.4g}, {hi * 1e3:.4g}] ms" in notes, (
                "the interval inside the note is unlabelled or in the other unit:\n  "
                + notes)
    # ...and the note's numbers are the ones on the estimate line, not a thousandth of
    # them: every mu the pane prints is inside the histogram's own decade.
    shown = _mu_values(text)
    assert shown and min(shown) > 1.0, (
        f"the estimate line is not in ms at all: {shown}")


def test_the_same_pane_in_seconds_says_seconds_on_every_line(plots, qtbot):
    """PASSES BOTH — a CONTROL, and the one that stops "fix it by hardwiring ms". With
    the axis in seconds the note and the estimate line must BOTH read seconds."""
    pp = plots
    pp.set_data({"apo": _apo600_frame()})
    pp.set_histogram_column("dwell_s")
    pp.set_time_unit("s")
    pp.set_fit(_GMM)
    pp.set_fit_components(3)
    _drain(pp, qtbot)

    fit = pp.fit_result()
    text = pp.details_text()
    notes = " ".join(ln for ln in text.splitlines() if ln.startswith("UNIDENTIFIED"))
    for j in range(1, 4):
        if f"mu_{j} = " in notes:
            assert f"mu_{j} = {float(fit.params[f'mu_{j}']):.4g} s" in notes, notes
            assert " ms" not in notes, notes


def test_the_unit_follows_the_axis_when_the_user_changes_it(plots, qtbot):
    """The display unit is a live control, not a value baked in at fit time: toggling
    it must move BOTH lines together."""
    pp = plots
    pp.set_data({"apo": _apo600_frame()})
    pp.set_histogram_column("dwell_s")
    pp.set_fit(_GMM)
    pp.set_fit_components(3)
    seen = {}
    for unit, factor in (("s", 1.0), ("ms", 1e3), ("µs", 1e6)):
        pp.set_time_unit(unit)
        _drain(pp, qtbot)
        fit = pp.fit_result()
        notes = " ".join(ln for ln in pp.details_text().splitlines()
                         if ln.startswith("UNIDENTIFIED"))
        seen[unit] = notes
        for j in range(1, 4):
            if f"mu_{j} = " in notes:
                assert f"mu_{j} = {float(fit.params[f'mu_{j}']) * factor:.4g} {unit}" \
                    in notes, f"axis says {unit}; the note says:\n  {notes}"
    written = [v for v in seen.values() if v]
    assert len(written) == 3, (
        f"one of the three axis units produced no identifiability note at all, so this "
        f"test proved nothing about the other two: {seen}")
    assert len(set(written)) == 3, (
        "the note did not change when the axis unit did: it was composed once and "
        f"cached: {seen}")


def test_a_depth_column_keeps_its_own_unit_through_the_same_path(plots, qtbot):
    """PICOAMPS ARE NOT SECONDS. The same fit over ``depth_pA`` must not acquire the
    time axis's factor — the display rule is the PARAMETER's unit, not the window's."""
    pp = plots
    n = 400
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "dwell_s": np.abs(rng.normal(2e-3, 3e-4, n)),
        "depth_pA": np.abs(np.r_[rng.normal(16.0, 6.0, 40), rng.normal(88.0, 7.0, 300),
                                 rng.normal(120.0, 25.0, 60)]),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })
    pp.set_data({"apo": frame})
    pp.set_histogram_column("depth_pA")
    pp.set_time_unit("ms")
    pp.set_fit(_GMM)
    pp.set_fit_components(3)
    _drain(pp, qtbot)

    fit = pp.fit_result()
    text = pp.details_text()
    notes = " ".join(ln for ln in text.splitlines()
                     if ln.startswith(("UNIDENTIFIED", "COMPONENTS NOT", "DEPTH FLOOR")))
    for j in range(1, 4):
        if f"mu_{j} = " in notes:
            assert f"mu_{j} = {float(fit.params[f'mu_{j}']):.4g} pA" in notes, notes
    assert " ms" not in notes, f"a picoamp was captioned in milliseconds:\n  {notes}"


@pytest.mark.parametrize("col,unit", [
    ("dwell_s", "s"), ("t_start_s", "s"), ("rise_s", "s"), ("fall_s", "s"),
    ("ecd_pA_s", "pA·s"), ("depth_pA", "pA"), ("baseline_pA", "pA"),
    ("max_dev_pA", "pA"), ("mean_dev_pA", "pA"),
    ("start_idx", "sample"), ("end_idx", "sample"),
    ("frac_depth", ""), ("snr", ""), ("n_levels", ""), ("event_id", ""),
])
def test_every_column_reports_the_unit_the_schema_declares(col, unit):
    """THE WHOLE COLUMN LIST, asked of the schema that declares it. ``start_idx`` and
    ``end_idx`` are ``"sample"`` in ``events.SCHEMA`` and answered "dimensionless"
    here, so a mixture over one printed a bare ``mu = 1.196e+06``."""
    assert _column_quantity(col)[1] == unit


def test_the_schema_is_the_source_and_not_a_copy_of_it():
    """A second list of units is a second thing to forget to update. Every column in
    the schema must come back with the unit the schema declares (modulo the two
    spellings this window translates), so a column ADDED to the schema cannot arrive
    here unlabelled."""
    for col, (_dtype, unit, _desc) in events_schema.SCHEMA.items():
        if col == "frac_depth":          # named, not captioned: it IS the ratio dI/I0
            continue
        expected = "" if unit == "-" else "pA·s" if unit == "pA*s" else unit
        assert _column_quantity(col)[1] == expected, (
            f"{col}: schema says {unit!r}, the window says "
            f"{_column_quantity(col)[1]!r}")


# ================================================================================================
# 2 — the click budget, at a quantile a bimodal distribution cannot hide behind
# ================================================================================================

def test_the_click_budget_holds_at_p90_and_max_on_the_sample_that_breaks_it(plots, qtbot):
    """THE CLAIM, measured the way a bimodal cost has to be measured.

    On the owner's -600 mV table the ``gamma`` and ``inverse-gaussian`` clicks are two
    populations, not one with a tail: 12 ms or 220 ms. The shipped tests assert the
    MEDIAN of 5-8 clicks, which on 3-of-12 over budget is inside the budget by
    construction. Twelve clicks per family, p90 AND max."""
    pp = plots
    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    over = {}
    for fam in ALL_FAMILIES:
        pp.set_fit(fam)
        _drain(pp, qtbot)
        clicks = [_curate(pp, frame, i) for i in range(12)]
        _drain(pp, qtbot)
        p90, worst = _quantile(clicks, 0.90), max(clicks)
        # >= 2 over-budget clicks, not max(): the defect this gate exists to catch is a
        # SECOND POPULATION (3-of-12 clicks at 220 ms), and a population is at least two
        # members. A max-of-one-click bound with zero margin also fails on a single OS
        # scheduler hiccup -- measured once at 1593 ms among eleven clicks of 15-20 ms,
        # ~once per full suite run on the owner's machine -- which is noise about the
        # machine, not evidence about the code. Two clicks over budget in twelve cannot
        # be a hiccup; one can. p90 keeps its zero-tolerance bound unchanged.
        n_over = sum(1 for c in clicks if c >= CLICK_BUDGET_MS)
        if p90 >= CLICK_BUDGET_MS or n_over >= 2:
            over[fam] = {"median": round(statistics.median(clicks)),
                         "p90": round(p90), "max": round(worst), "n_over": n_over,
                         "clicks": [round(c) for c in clicks]}
    assert not over, (
        f"over the {CLICK_BUDGET_MS:.0f} ms click budget at p90 or on two or more "
        f"clicks (the median hides it): {over}")


def test_no_curation_click_runs_a_maximisation_that_had_stopped_moving(plots, qtbot):
    """THE MECHANISM, so this does not rest on a stopwatch. A timing test goes green on
    a fast machine with the defect intact; what must be true is that no click spends
    thousands of iterations re-testing a point the search already reached.

    Counted on the GUI thread only: the advisory worker may do what it likes."""
    from scipy import optimize

    pp = plots
    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    gui = threading.current_thread()
    seen: list[int] = []
    real = optimize.minimize

    def counted(*a, **k):
        r = real(*a, **k)
        if threading.current_thread() is gui:
            seen.append(int(getattr(r, "nit", 0)))
        return r

    for fam in (_GAMMA, _INVGAUSS):
        pp.set_fit(fam)
        _drain(pp, qtbot)
        seen.clear()
        try:
            optimize.minimize = counted
            for i in range(12):
                _curate(pp, frame, i)
        finally:
            optimize.minimize = real
        _drain(pp, qtbot)
        assert seen, f"{fam}: no maximisation ran on the click path at all"
        assert max(seen) < stats._NM_MAXITER, (
            f"{fam}: a curation click ran a {max(seen)}-iteration Nelder-Mead. On this "
            f"sample the answer is reached at ~iteration 82 and never changes again; "
            f"the rest is 218 ms of a 220 ms click. iterations seen: {seen}")
        assert max(seen) <= stats._NM_STALL_ITERS + 250, (
            f"{fam}: {max(seen)} iterations is past the stall budget: {seen}")


def test_the_pane_says_why_the_peaked_fit_stopped(plots, qtbot):
    """A fit that was cut off must SAY it was cut off, and say what it means. This is
    the family's own boundary — gamma with k → 0 has no mode at all — printed beside a
    BIC that a cross-family comparison then ranks."""
    pp = plots
    frame = _apo600_frame()
    pp.set_data({"apo": frame})
    pp.set_histogram_column("dwell_s")
    pp.set_fit(_GAMMA)
    _drain(pp, qtbot)
    stalled = None
    for i in range(12):
        _curate(pp, frame, i)
        qtbot.wait(5)
        fit = pp.fit_result()
        if fit is not None and fit.meta.get("stalled"):
            stalled = pp.details_text()
            break
    _drain(pp, qtbot)
    assert stalled is not None, (
        "no curation click on this sample produced a stalled maximisation; the sample "
        "no longer demonstrates the defect")
    assert "STOPPED MOVING" in stalled, stalled
    assert "no mode at all" in stalled, stalled
    assert "Maximum number of iterations has been exceeded" not in stalled, (
        "the pane is still reporting scipy's own sentence, which says nothing about "
        "what happened:\n" + stalled)


def test_a_well_posed_peaked_fit_is_never_cut_off(plots, qtbot):
    """PASSES BOTH — a CONTROL. The fixture's OWN detected events carry real detector
    provenance, so the fit is well posed; the guard must be invisible there, or it is
    moving numbers on the path every real curation click takes."""
    pp = plots
    pp.set_histogram_column("dwell_s")
    for fam in (_GAMMA, _INVGAUSS):
        pp.set_fit(fam)
        _drain(pp, qtbot)
        fit = pp.fit_result()
        assert fit is not None
        assert fit.meta.get("stalled", False) is False, (
            f"{fam}: the stall guard fired on a fit with real detection provenance")
        assert fit.meta.get("converged", True) is True
