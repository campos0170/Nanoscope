"""BL-3 — the dwell-family fit captions speak the fitted column's unit, not seconds.

The Plots window offers its four dwell families (exponential, gamma, inverse-gaussian,
exponential + gamma) on ANY positive column, and every one of them stamped its scale
parameters ``"s"`` whatever it was handed. Measured through a real PlotsWindow before
the fix, on an exponential-shaped ``depth_pA`` column (n = 400):

    τ = 77.58 s   (n = 400)                                   <- 77.58 is PICOAMPS
    ...
    no detection limit is known for “depth_pA”: ...           <- same pane, correct
    provenance has no 'min_dwell_s'; using the shortest observed dwell (2.13974 s) ...
    95% CI ...: tau_1 [70.19, 84.55] s

and the same hole for ``inverse-gaussian`` (``μ = 79.72 s``) and for ``gamma``'s
deferred interval (``theta [52.43, 69.48] s``) — three families whose "time constants"
were picoamps, on the pane whose whole job is captioning numbers honestly.

The fix is the ``fit_depth_gaussians`` contract (report High-4) extended to the three
dwell fitters: the window passes ``quantity=``/``unit=`` from ``_column_quantity``
whenever the column's own unit is not ``"s"``, the fitter stamps ``param_units`` (and
its model name) with the truth, and the existing shared display rule
(:meth:`PlotsWindow._display_unit`) does the rest — estimate line, ``_ci_note`` and the
fitter's own notes all read from the one stamp. Dwell fits keep their historical
labels bit-for-bit; the exponential's degeneracy note additionally converts its taus
through the same ``display=`` pair the depth mixture uses, and calls tau a "decay
constant" when the axis is not a time.

Every assertion here is read out of ``details_text()`` on a real
:class:`~nanoscope.gui.plots_panel.PlotsWindow` driven by its public setters, in the
style of ``tests/gui/test_plots_fit_honesty.py`` (whose contract this file extends and
must not regress).
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

#: The detector settings used for the dwell CONTROLS, so the provenance path (the one
#: every historical test exercises) is represented here too and provably unchanged.
PROV = {"min_dwell_s": 2e-4, "fs": 100_000.0}


def test_the_tree_under_test_is_the_real_one():
    """Same guard as the sibling honesty file: believe nothing before knowing where
    the package came from."""
    here = Path(nanoscope.__file__).resolve()
    assert here.parent.parent == Path("/Users/abrahamcampos/Desktop/nanoscope").resolve(), here


def _frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Exponential-shaped columns in three unit kinds: seconds, picoamps, none.

    ``depth_pA`` is exponential-shaped ON PURPOSE — the dwell families fit it happily
    (the flagship KS verdict before the fix was "consistent with these data"), which is
    exactly why a wrong unit there is dangerous: nothing else on the pane looks off.
    Seed 7 also makes the k=2 exponential on it DEGENERATE, which this file uses.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-4 + rng.exponential(2e-3, n),
        "depth_pA": 2.0 + rng.exponential(80.0, n),
        "snr": 0.5 + rng.exponential(4.0, n),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _window(qtbot, col: str, *, provenance: bool = False) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    if provenance:
        w.set_data({"probe": _frame()}, {"probe": dict(PROV)})
    else:
        w.set_data({"probe": _frame()})
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    w.set_time_unit("ms")
    return w


def _drain(pp: PlotsWindow, qtbot, *, timeout_s: float = 240.0) -> None:
    """Wait for the deferred notes (peaked CIs, the exp+gamma fit itself, the advisory
    scans) to land, loudly if they never do — same rule as the sibling files: a silent
    timeout is indistinguishable from a product regression."""
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


def _line(text: str, prefix: str) -> str:
    hits = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    assert hits, f"no line starting {prefix!r} in:\n{text}"
    return hits[0]


# ================================================================================================
# THE FLAGSHIP — exponential over picoamps called its decay constant a time in seconds
# ================================================================================================

def test_exponential_on_a_picoamp_column_prints_picoamps(qapp, qtbot):
    """MEASURED before the fix: ``τ = 77.58 s`` for a number that is picoamps, two
    lines below the pane correctly saying no detection limit is known for depth_pA —
    with the interval (``tau_1 [70.19, 84.55] s``) and a stamped provenance note
    ("shortest observed dwell (2.13974 s)") wearing the same wrong unit and calling
    the column a dwell."""
    pp = _window(qtbot, "depth_pA")
    pp.set_fit("exponential")
    # The bootstrap CI this test reads below is DEFERRED now (it is what used to sit
    # between a curation click and the repaint), so it has to be waited for like every
    # other deferred note in this file. Nothing about the assertions changes: the unit
    # stamp and the interval wording are exactly what they were.
    _drain(pp, qtbot)
    fit = pp.fit_result()
    text = pp.details_text()

    tau = float(fit.params["tau_1"])
    cap = _line(text, "τ =")
    assert f"τ = {tau:.4g} pA" in cap, cap
    assert " s " not in cap and not cap.split("(")[0].rstrip().endswith(" s"), cap

    # ...and the stamp is ON THE FIT, where every other reader (the Inspector, a saved
    # project) takes it from, not just in this pane's wording.
    assert fit.unit_of("tau_1") == "pA"

    # The interval line follows the same stamp.
    lo, hi = fit.ci["tau_1"]
    assert f"tau_1 [{lo:.4g}, {hi:.4g}] pA" in text, text

    # The model line says what was fitted, like the depth mixture's always has.
    assert "exponential (left-truncated MLE) on depth_pA" in text, text

    # The pane still says — once, correctly — that the floor was guessed...
    assert "no detection limit is known for “depth_pA”" in text, text
    # ...and no longer ALSO prints the dwell-worded copy that labelled picoamps " s".
    assert "provenance has no 'min_dwell_s'" not in text, text
    assert " s)" not in text, f"a seconds label survived on this picoamp pane:\n{text}"


def test_the_inverse_gaussian_scale_parameters_are_picoamps(qapp, qtbot):
    """MEASURED before the fix: ``μ = 79.72 s`` on the estimate line, and the deferred
    interval would have landed as ``mu [...] s   lambda [...] s``."""
    pp = _window(qtbot, "depth_pA")
    pp.set_fit("inverse-gaussian (first passage)")
    _drain(pp, qtbot)
    fit = pp.fit_result()
    text = pp.details_text()

    mu = float(fit.params["mu"])
    cap = _line(text, "μ =")
    assert f"μ = {mu:.4g} pA" in cap, cap
    assert fit.unit_of("mu") == "pA" and fit.unit_of("lambda") == "pA"
    assert "inverse-gaussian (truncated MLE) on depth_pA" in text, text

    ci_line = _line(text, "95% CI")
    assert "mu [" in ci_line and "lambda [" in ci_line, ci_line
    assert "] s" not in ci_line, ci_line
    assert ci_line.count("] pA") == 2, ci_line


def test_the_gamma_theta_interval_is_picoamps_and_k_stays_bare(qapp, qtbot):
    """The gamma caption prints the dimensionless shape ``k`` (correct before and
    after); the hole was the deferred interval, which captioned the picoamp scale
    ``theta [...] s``. A fit is not all one unit — the same rule in the other
    direction as ``test_a_dimensionless_shape_parameter_gets_no_time_unit``."""
    pp = _window(qtbot, "depth_pA")
    pp.set_fit("gamma (peaked)")
    _drain(pp, qtbot)
    fit = pp.fit_result()
    text = pp.details_text()

    cap = _line(text, "k =")
    assert "pA" not in cap and " s" not in cap.split("(")[0], cap
    assert fit.unit_of("k") == "" and fit.unit_of("theta") == "pA"
    assert "gamma (truncated MLE) on depth_pA" in text, text

    ci_line = _line(text, "95% CI")
    assert "theta [" in ci_line and "] pA" in ci_line, ci_line
    assert "] s" not in ci_line, ci_line


def test_the_exp_gamma_fit_stamps_the_fitted_unit(qapp, qtbot):
    """This family prints only the dimensionless ``k`` on this pane, but its ``tau``
    and ``theta`` persist through ``unit_of`` into every other surface — stamping them
    seconds for a picoamp fit is the same lie one save away."""
    pp = _window(qtbot, "depth_pA")
    pp.set_fit("exponential + gamma (bimodal)")
    _drain(pp, qtbot)
    fit = pp.fit_result()
    text = pp.details_text()

    assert fit is not None, text
    assert fit.unit_of("tau") == "pA" and fit.unit_of("theta") == "pA"
    assert fit.unit_of("w_exp") == "" and fit.unit_of("k") == ""
    assert "exponential + gamma mixture (censored MLE) on depth_pA" in text, text
    cap = _line(text, "k =")
    assert "pA" not in cap and " s" not in cap.split("(")[0], cap


# ================================================================================================
# CONTROLS — the fix is "the column's unit", not "always picoamps" or "never seconds"
# ================================================================================================

def test_a_dimensionless_column_takes_no_unit_at_all(qapp, qtbot):
    """``snr`` is a ratio: before the fix its decay constant read ``τ = 3.904 s``;
    the honest caption is the bare number, not some other unit."""
    pp = _window(qtbot, "snr")
    pp.set_fit("exponential")
    fit = pp.fit_result()
    cap = _line(pp.details_text(), "τ =")
    assert f"τ = {float(fit.params['tau_1']):.4g}   (" in cap, cap
    assert fit.unit_of("tau_1") == ""
    assert "exponential (left-truncated MLE) on snr" in pp.details_text()


def test_a_dwell_fit_keeps_its_historical_caption(qapp, qtbot):
    """PASSES BOTH WAYS — the control that pins the provenance path: a real dwell fit
    still captions in the axis's unit and keeps the exact historical model name (no
    " on dwell_s" suffix), so nothing downstream of that string moves."""
    pp = _window(qtbot, "dwell_s", provenance=True)
    pp.set_fit("exponential")
    fit = pp.fit_result()
    text = pp.details_text()
    cap = _line(text, "τ =")
    assert f"τ = {float(fit.params['tau_1']) * 1e3:.4g} ms" in cap, cap
    assert fit.unit_of("tau_1") == "s"
    assert "exponential (left-truncated MLE)\n" in text + "\n", text
    assert " on dwell_s" not in text, text


# ================================================================================================
# THE FITTER'S OWN NOTE — the degeneracy sentence follows the axis, and names the
# parameter honestly for it
# ================================================================================================

def test_the_degenerate_note_speaks_the_axis_unit_for_dwells(qapp, qtbot):
    """The exponential's degeneracy note used to quote its taus in raw seconds under
    an axis reading ms — the exact one-pane-two-units defect the depth mixture's
    ``display=`` pair was built to end (see test_rp_one_pane_one_unit_panel). Seed 2
    makes a pure-exponential dwell sample whose k=2 fit collapses."""
    rng = np.random.default_rng(2)
    n = 986
    frame = pd.DataFrame({
        "dwell_s": 2e-4 + rng.exponential(0.02, n),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })
    pp = PlotsWindow(Session())
    qtbot.addWidget(pp)
    pp.set_data({"probe": frame}, {"probe": dict(PROV)})
    pp.set_plot_type("Histogram")
    pp.set_histogram_column("dwell_s")
    pp.set_time_unit("ms")
    pp.set_fit("exponential")
    pp.set_fit_components(2)
    fit = pp.fit_result()
    assert fit.meta.get("degenerate") is True, (
        "this sample's k=2 fit no longer collapses; the note under test never fired")
    text = pp.details_text()
    note = _line(text, "DEGENERATE MIXTURE")
    assert f"tau_1 = {float(fit.params['tau_1']) * 1e3:.6g} ms" in note, note
    assert "time constant" in note, note
    assert " s and " not in note, f"raw seconds under an axis reading ms:\n{note}"


def test_the_degenerate_note_on_picoamps_names_a_decay_constant(qapp, qtbot):
    """A decay constant over pA is not a time constant, and the note must not call it
    one — the naming half of the flagship fix, on the same wiring."""
    pp = _window(qtbot, "depth_pA")
    pp.set_fit("exponential")
    pp.set_fit_components(2)
    fit = pp.fit_result()
    assert fit.meta.get("degenerate") is True, (
        "this sample's k=2 fit no longer collapses; the note under test never fired")
    note = _line(pp.details_text(), "DEGENERATE MIXTURE")
    assert f"tau_1 = {float(fit.params['tau_1']):.6g} pA" in note, note
    assert "decay constant" in note, note
    assert "time constant" not in note, note


# ================================================================================================
# THE FIT RANGE — the truncation window prints in the same unit as the parameters
# ================================================================================================

def test_the_fit_range_note_names_the_columns_unit(qapp, qtbot):
    """The range note answered "" for every non-time column, so a picoamp truncation
    window printed bare numbers on the pane whose job is saying what its numbers are
    in. It now goes through the same shared display rule as everything above it."""
    pp = _window(qtbot, "depth_pA")
    pp.set_fit("exponential")
    pp._fit_lo.setText("10")
    pp._fit_hi.setText("300")
    pp._redraw()
    assert "FIT RANGE [10, 300] pA" in pp.details_text(), pp.details_text()

    # ...and a dwell range still reads in the axis's unit, exactly as before.
    pd_dwell = _window(qtbot, "dwell_s", provenance=True)
    pd_dwell.set_fit("exponential")
    pd_dwell._fit_lo.setText("0.5")
    pd_dwell._fit_hi.setText("20")
    pd_dwell._redraw()
    assert "FIT RANGE [0.5, 20] ms" in pd_dwell.details_text(), pd_dwell.details_text()
