"""FIT HONESTY, in the window the owner actually reads: units, verdicts, and rivals.

The GUI half of Area B of the apo-ferritin end-to-end report
(``nanoscope-reports/apo-run-2026-08-19/SUMMARY.md`` §7). Every screenshot the report
names was taken from this window, so every number here is asserted through a real
:class:`~nanoscope.gui.plots_panel.PlotsWindow` driven by its own public setters, fed the
REAL merged event sets from that run.

    High-3   "τ = 0.01981 (n = 986)" with a tight CI, no goodness-of-fit statistic and no
             warning, while the drawn curve missed the histogram.
    High-4   the Details pane printed "μ = 85.46, 99.69, 105.8" with no units, under a
             model line that said dI/I0, for a fit to depth_pA.
    High-7   with Time unit = ms the dwell axis read ``dwell (s, log)`` AND carried a
             pyqtgraph ``(×0.001)`` multiplier, so an 18 ms event landed at x = 10¹.
    Medium-10 no Auto-BIC for histogram fits, and the recommender refit only k ∈ {1,2,3}
             while the spin accepted 1-4.
    Low-19   a fit with BIC = +inf was still offered and still drawn.
    Low-21   ``set_fit()`` failed silently on an unmatched name, and the panel went on
             reporting the PREVIOUS model's numbers.
    Low-28   τ was printed without units while the axis was in ms.

Reverted, hunk by hunk, by the out-of-tree ``revert_areaB`` plugin; run this file with
``-p revert_areaB`` and 16 of the 21 tests must fail. The five that pass BOTH ways are
deliberately not pins and say so in their own docstrings:

* ``test_the_tree_under_test_is_the_real_one`` — the guard on the revert itself.
* ``test_a_dimensionless_column_is_not_given_a_unit`` and
  ``test_a_dimensionless_shape_parameter_gets_no_time_unit`` — SILENCE controls. A flag
  that can only fire says nothing; these check it stays quiet where it should.
* ``test_a_fit_with_a_non_finite_bic_is_refused_rather_than_drawn`` — the general
  invariant behind Low-19, tested by injection. Low-19's own case is pinned by
  ``test_the_inverse_gaussian_draws_a_finite_fit_on_700mV``, which does fail reverted.
* ``test_auto_bic_is_offered_only_where_a_component_count_means_something`` — a
  control on the new widget's enablement.

STANDING CAVEAT, from the report's front matter: the gain on these recordings is
uncalibrated (open pore -27 to -52 pA where a 22 nm pore should be ~1000x larger), and
``frac_depth`` is 1.81-2.57 because the current MAGNITUDE grows during an event. No
absolute picoamp below is a physical measurement; they are fixtures for a labelling
contract, which is precisely the contract that matters on a dataset where dI/I0 > 1.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

APO = Path("/Users/abrahamcampos/Desktop/nanoscope-reports/apo-run-2026-08-19/verification")

#: The detector settings the run used on all three recordings (report §1).
APO_PROV = {"min_dwell_s": 20e-6, "fs": 50000.0}



def _settle(win, *_ignored):
    """Let the panel's deferred advisory notes arrive.

    The cross-family BIC table and the component-count hint are computed OFF the redraw
    path (see ``PlotsWindow._deferred_note``): they used to run inside the redraw, which
    put 0.3-1.0 s between a curation click and the repaint. These tests assert what the
    notes SAY, which is unchanged -- only WHEN they say it moved.

    Deliberately does not take ``qtbot``: several of the tests that read a note do not
    request that fixture, and a settle helper that only works in some of them is a trap.

    THE BUDGET IS A DRAIN, NOT A GUESS, AND IT IS SPENT ONLY WHEN NEEDED (the loop
    breaks the moment the window is quiescent). It was ``_NOTE_SETTLE_MS * 6`` = 2.4 s,
    and one test in this file needs 7.5 s: ``test_the_panel_reports_bic_and_names_a_
    better_component_count`` fits two well-separated exponential populations, n = 1000,
    and its rival-family scan does not finish inside 2.4 s. That test PASSED anyway --
    it only reads the better-k note, which lands in the first round -- and then RETURNED
    WITH ITS TIMER STILL ARMED. ``qtbot`` closes a widget at teardown but the delete is
    deferred, so the leftover ``_settle_notes`` fired inside the NEXT test's
    ``processEvents`` and ate that test's whole 2.4 s: ``test_the_rival_hint_still_names_
    the_better_component_count`` failed with "still computing -- it appears when you stop
    clicking", deterministically, but ONLY when the file was run whole. Alone it passes
    in 0.83 s. One test's unfinished homework became the next test's failure.

    So: drain until this window has nothing pending, and say so loudly if it never does.
    A silent timeout here is indistinguishable from a real product regression, and this
    one cost a suite run to tell apart.
    """
    import time
    from PySide6 import QtWidgets
    from nanoscope.gui.plots_panel import PlotsWindow
    app = QtWidgets.QApplication.instance()
    deadline = time.time() + (PlotsWindow._NOTE_SETTLE_MS * 30) / 1000.0
    while time.time() < deadline:
        app.processEvents()
        if not win._pending_notes and not win._note_timer.isActive():
            app.processEvents()
            break
    else:
        raise AssertionError(
            "the panel's deferred notes never settled: still pending "
            f"{sorted(k[0] for k in win._pending_notes)} after "
            f"{PlotsWindow._NOTE_SETTLE_MS * 30 / 1000:.0f} s. A test that returns here "
            "leaves its timer armed and the NEXT test pays for it.")

def test_the_tree_under_test_is_the_real_one():
    """pytest's rootdir insertion has twice swapped the tree under a revert run on this
    project, silently turning a "before" run into an "after" run. Assert where the
    package came from before believing anything else in this file."""
    here = Path(nanoscope.__file__).resolve()
    assert here.parent.parent == Path("/Users/abrahamcampos/Desktop/nanoscope").resolve(), here


def _apo(run: str) -> pd.DataFrame:
    path = APO / run / "events_merge.csv"
    if not path.exists():
        pytest.skip(f"apo-ferritin merged event set not available: {path}")
    return pd.read_csv(path)


def _window(qtbot, run: str) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({run: _apo(run)}, {run: dict(APO_PROV)})
    w.set_plot_type("Histogram")
    return w


@pytest.fixture
def apo700(qapp, qtbot):
    return _window(qtbot, "apo_-700mV")


@pytest.fixture
def apo500(qapp, qtbot):
    return _window(qtbot, "apo_-500mV")


def _axis(win, side="bottom"):
    return win.plot_item().getAxis(side)


def _drawn_labels(ax) -> list[str]:
    """The tick labels this axis actually PAINTS, in axis order.

    ``tickStrings`` is what the axis offers; ``generateDrawSpecs`` is what survives the
    collision filter and reaches the pixel. Every assertion about readability has to be
    made on this side of it.
    """
    from PySide6 import QtGui

    img = QtGui.QImage(1400, 900, QtGui.QImage.Format_ARGB32)
    painter = QtGui.QPainter(img)
    try:
        specs = ax.generateDrawSpecs(painter)
    finally:
        painter.end()
    if specs is None:
        return []
    horizontal = ax.orientation in ("bottom", "top")
    texts = [(r.center().x() if horizontal else -r.center().y(), t)
             for r, _flags, t in specs[2] if str(t).strip()]
    return [t for _pos, t in sorted(texts)]


# ================================================================================================
# High-7 (dwell half) -- ONE UNIT STATEMENT PER AXIS
# ================================================================================================

def test_the_log_dwell_axis_shows_milliseconds_and_says_milliseconds(apo700):
    """MEASURED (report High-7): the axis read ``dwell (s, log)`` and carried a pyqtgraph
    ``(×0.001)`` multiplier at the same time, and the tick labels mixed decade
    (``0.01, 0.1, 1``) with exponent (``10¹, 10²``) formats. A 19 ms event landed at
    x = 10¹ under a label claiming seconds — an invitation to a 1000x error.

    Reverted by: ``axis-label-seconds``.
    """
    apo700.set_histogram_column("dwell_s")
    apo700.set_time_unit("ms")
    apo700.set_log_x(True)
    ax = _axis(apo700)

    assert ax.labelText == "dwell (ms, log)"
    assert "(x" not in ax.labelString(), ax.labelString()
    assert ax.autoSIPrefix is False
    assert ax.display_factor == pytest.approx(1e3)

    # The labels that are actually PAINTED, through pyqtgraph's own pipeline and this
    # axis's collision filter. Asserted on the drawn set rather than on every emitted
    # tick string: minor ticks are now labelled and thinned by collision (they used to
    # be blanked, which emptied an axis zoomed inside one decade and painted every
    # decade twice — see tests/gui/test_plots_panel.py).
    labels = _drawn_labels(ax)
    assert labels, "the axis drew no labels at all"

    # ONE format across the whole axis. The defect read "0.01, 0.1, 1, 10¹, 10²" — two
    # notations, so `1` and `10¹` sat two ticks apart looking like different KINDS of
    # number. Superscripts are allowed, but not mixed with decimals.
    superscript = [s for s in labels if any(c in s for c in "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")]
    assert not superscript or len(superscript) == len(labels), labels

    # ...and the decades are MILLISECONDS. This recording's dwells run 0.10 to 31.12 ms,
    # so the top labelled decade is 10 (ms). In seconds it would have been 0.01.
    # Every decade in range survives the thinning; that is the property, not the exact
    # set of minor labels that happened to fit at this pixel width.
    assert {"0.1", "1", "10"} <= set(labels), labels
    assert "0.0001" not in labels, f"these are seconds, not milliseconds: {labels}"
    # No label is drawn twice.
    assert len(labels) == len(set(labels)), labels


def test_the_seconds_unit_still_reads_in_seconds(apo700):
    """The fix is not "always ms": the axis names whatever unit is selected."""
    apo700.set_histogram_column("dwell_s")
    apo700.set_time_unit("s")
    apo700.set_log_x(True)
    ax = _axis(apo700)
    assert ax.labelText == "dwell (s, log)"
    assert ax.display_factor == pytest.approx(1.0)
    assert "(x" not in ax.labelString()


def test_a_non_time_log_axis_resets_the_display_factor(apo700):
    """A stale x1000 left on a pA axis would be a silent factor-of-1000 lie — the same
    rule the linear branch has always had."""
    apo700.set_time_unit("ms")
    apo700.set_histogram_column("dwell_s")
    apo700.set_log_x(True)
    assert _axis(apo700).display_factor == pytest.approx(1e3)
    apo700.set_histogram_column("depth_pA")
    assert _axis(apo700).display_factor == pytest.approx(1.0)
    assert _axis(apo700).labelText == "depth_pA (log)"


# ================================================================================================
# High-4 + Low-28 -- THE DETAILS PANE PRINTS UNITS
# ================================================================================================

def test_tau_is_printed_in_the_unit_the_axis_is_showing(apo700):
    """MEASURED (report Low-28): the pane printed "τ = 0.02331" — seconds, unlabelled —
    beside a histogram axis in ms.

    Reverted by: ``fitresult-unit-of``.
    """
    apo700.set_histogram_column("dwell_s")
    apo700.set_time_unit("ms")
    apo700.set_fit("exponential")
    line = [ln for ln in apo700.details_text().splitlines() if "τ" in ln][0]
    assert "τ = 19.81 ms" in line, line

    apo700.set_time_unit("s")
    line = [ln for ln in apo700.details_text().splitlines() if "τ" in ln][0]
    assert "τ = 0.01981 s" in line, line


def test_the_depth_mixture_names_its_column_and_its_unit(apo700):
    """MEASURED (report High-4): fitting ``depth_pA`` printed "μ = 85.46, 99.69, 105.8"
    with no units under a model labelled ``Gaussian mixture on dI/I0``.

    Reverted by: ``depth-quantity-label`` + ``fitresult-unit-of``.
    """
    apo700.set_histogram_column("depth_pA")
    apo700.set_fit("gaussian-mixture")
    apo700.set_fit_components(3)
    text = apo700.details_text()

    assert "Gaussian mixture on depth_pA (k=3, EM)" in text, text
    assert "dI/I0" not in text, text
    line = [ln for ln in text.splitlines() if ln.startswith("μ =")][0]
    assert line.startswith("μ = 85.46, 99.69, 105.8 pA"), line


def test_a_dimensionless_column_is_not_given_a_unit(apo700):
    """``frac_depth`` is a ratio — 1.81-2.57 on these recordings, i.e. an ENHANCEMENT —
    and must never acquire a picoamp caption on the way through."""
    apo700.set_histogram_column("frac_depth")
    apo700.set_fit("gaussian-mixture")
    text = apo700.details_text()
    assert "Gaussian mixture on dI/I0" in text, text
    line = [ln for ln in text.splitlines() if ln.startswith("μ =")][0]
    assert "pA" not in line, line


def test_a_dimensionless_shape_parameter_gets_no_time_unit(apo700):
    """The gamma shape ``k`` is dimensionless. A fit is not all one unit."""
    apo700.set_histogram_column("dwell_s")
    apo700.set_time_unit("ms")
    apo700.set_fit("gamma (peaked)")
    line = [ln for ln in apo700.details_text().splitlines() if ln.startswith("k =")][0]
    assert "ms" not in line and "s" not in line.split("(")[0], line


# ================================================================================================
# High-3 -- THE PANEL SAYS WHEN THE MODEL IS WRONG, AND WHAT WOULD BE RIGHT
# ================================================================================================

def test_the_details_pane_carries_a_goodness_of_fit_verdict(apo700):
    """MEASURED (report High-3): "τ = 0.01981 (n = 986)" with a bootstrap CI of
    [0.019589, 0.020032] and NO goodness-of-fit statistic and NO warning, while the drawn
    curve visibly missed the histogram.

    Reverted by: ``panel-no-verdict``.
    """
    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("exponential")
    text = apo700.details_text()
    assert "KS D = 0.6395" in text, text
    assert "DOES NOT DESCRIBE THESE DATA" in text, text


def test_the_details_pane_ranks_the_rival_families(apo700):
    """The comparison the app was one call away from making: it could already fit all
    four families in its own menu and never compared them. On -700 mV the exponential is
    the WORST of them, by 2729 BIC.

    Reverted by: ``panel-no-verdict``.
    """
    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("exponential")
    _settle(apo700)   # the advisory notes are computed off the redraw path
    text = apo700.details_text()

    assert "MODEL COMPARISON" in text, text
    assert "exponential + gamma (bimodal)" in text
    assert "<- best" in text and "<- shown" in text
    assert "fits these data better by 2728" in text, text
    # The recommendation must name something the Fit box actually offers.
    assert "exponential + gamma (bimodal)" in [
        apo700._fit_combo.itemText(i) for i in range(apo700._fit_combo.count())]


def test_a_model_that_passes_its_verdict_does_not_drag_in_the_rival_scan(qapp, qtbot):
    """THE GATE IS THE POINT. Refitting every family costs ~215 ms; the scan runs only
    once the KS test has already said the drawn curve is wrong. A well-specified fit pays
    nothing, which is what keeps a curation click fast."""
    rng = np.random.default_rng(3)
    n = 800
    df = pd.DataFrame({"dwell_s": 2e-5 + rng.exponential(2e-3, n),
                       "accepted": np.ones(n, bool),
                       "filter_limited": np.zeros(n, bool)})
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"clean": df}, {"clean": dict(APO_PROV)})
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    text = w.details_text()
    assert "The fitted curve is consistent with these data." in text, text
    assert "MODEL COMPARISON" not in text, text


# ================================================================================================
# Low-19 -- A NON-FINITE FIT IS NOT DRAWN
# ================================================================================================

def test_the_inverse_gaussian_draws_a_finite_fit_on_700mV(apo700):
    """MEASURED (report Low-19): ``inverse-gaussian (first passage)`` returned BIC = +inf
    on this recording and was STILL offered and STILL drawn, with its mu and lambda
    printed. The likelihood now works in log space, so the fit is finite and usable —
    BIC -5950.75, independently confirmed against ``scipy.stats.invgauss`` in
    ``tests/analysis/test_fit_honesty.py``.

    Reverted by: ``truncated-nll-density-space``.
    """
    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("inverse-gaussian (first passage)")
    text = apo700.details_text()
    assert "Fit unavailable" not in text, text
    assert "BIC = -5950.7" in text, text
    assert apo700.fit_result() is not None
    assert np.isfinite(apo700.fit_result().bic)


def test_a_fit_with_a_non_finite_bic_is_refused_rather_than_drawn(apo700, monkeypatch):
    """The invariant behind Low-19, independent of which family trips it: a fit with no
    finite likelihood is not a fit, is not drawn, and says so in Details."""
    from nanoscope.analysis import stats as st

    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("exponential")
    assert "Fit unavailable" not in apo700.details_text()
    n_before = len(apo700.plot_item().listDataItems())

    real = st.fit_dwell_exponential

    def _infinite(*a, **kw):
        fit = real(*a, **kw)
        fit.bic = float("inf")
        return fit
    monkeypatch.setattr("nanoscope.analysis.populations.fit_dwell_exponential", _infinite)
    apo700._redraw()

    text = apo700.details_text()
    assert "Fit unavailable" in text, text
    assert "non-finite likelihood" in text, text
    assert len(apo700.plot_item().listDataItems()) < n_before, "the bad fit was drawn anyway"


# ================================================================================================
# Low-21 -- A SETTER THAT CANNOT HONOUR ITS ARGUMENT MUST NOT RETURN QUIETLY
# ================================================================================================

def test_set_fit_never_leaves_the_previous_model_reporting(apo700):
    """MEASURED (report Low-21): ``set_fit("gamma")`` — one word short of
    ``"gamma (peaked)"`` — is a no-op on a non-editable combo. The panel kept the previous
    model selected and went on printing THAT model's numbers, with nothing raised. It cost
    a full run of wrong numbers before a read-back check caught it.

    Reverted by: ``set-fit-silent``.
    """
    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("exponential")
    assert "exponential (left-truncated MLE)" in apo700.details_text()

    # An unambiguous short form RESOLVES rather than silently doing nothing...
    apo700.set_fit("gamma")
    assert apo700._fit_combo.currentText() == "gamma (peaked)"
    assert "gamma (truncated MLE)" in apo700.details_text()

    # ...and a name this window does not offer RAISES, naming the options.
    with pytest.raises(ValueError) as err:
        apo700.set_fit("weibull")
    assert "unknown fit" in str(err.value)
    assert "gamma (peaked)" in str(err.value)


def test_an_ambiguous_short_form_is_an_error_not_a_coin_toss(apo700):
    """``"exponential"`` is both an exact item and a prefix of
    ``"exponential + gamma (bimodal)"``; the exact match must win. A prefix that matches
    two items and neither exactly must raise."""
    apo700.set_fit("exponential")
    assert apo700._fit_combo.currentText() == "exponential"
    apo700.set_fit("exponential + gamma (bimodal)")
    assert apo700._fit_combo.currentText() == "exponential + gamma (bimodal)"
    with pytest.raises(ValueError, match="ambiguous"):
        apo700.set_fit("e")


def test_set_scatter_fit_is_held_to_the_same_rule(apo700):
    apo700.set_plot_type("Scatter")
    apo700.set_scatter_columns("dwell_s", "depth_pA")
    apo700.set_scatter_fit("power law")          # the documented short form still works
    assert apo700._scatter_fit.currentText() == "power law (log-log)"
    with pytest.raises(ValueError, match="unknown scatter fit"):
        apo700.set_scatter_fit("quadratic")


# ================================================================================================
# Medium-10 -- THE RECOMMENDER SEARCHES THE SET THE USER CAN CHOOSE FROM
# ================================================================================================

def test_the_recommender_reaches_the_top_of_the_spin(apo500):
    """MEASURED (report Medium-10): on -500 mV ``depth_pA`` the panel advised
    "3 components fit markedly better" and never mentioned that 4 was better still —
    BIC 7170.43 against 7198.43 — because it refit only k ∈ {1, 2, 3} while the spin
    accepted 1-4.

    Reverted by: ``component-range-123``.
    """
    apo500.set_histogram_column("depth_pA")
    apo500.set_fit("gaussian-mixture")
    apo500.set_fit_components(1)
    _settle(apo500)   # the advisory notes are computed off the redraw path
    text = apo500.details_text()
    assert "4 components fit markedly better" in text, text
    assert "7170.43" in text, text


def test_auto_bic_picks_the_component_count_and_writes_it_back(apo500):
    """The parity with the clustering group's Auto (BIC) that the fit group never had.
    The chosen k goes back into the spin, so the number on screen is the number fitted,
    and every candidate's score is printed so the choice is auditable."""
    apo500.set_histogram_column("depth_pA")
    apo500.set_fit("gaussian-mixture")
    apo500.set_fit_components(1)
    assert apo500._fit_k_auto.isEnabled()
    apo500._fit_k_auto.setChecked(True)

    assert apo500._fit_k.value() == 4
    assert apo500._fit_k.isEnabled() is False, "Auto owns the spin while it is on"
    text = apo500.details_text()
    assert "AUTO (BIC) chose k = 4" in text, text
    for k, bic in ((1, "7365.41"), (2, "7265.59"), (3, "7198.43"), (4, "7170.43")):
        assert f"k={k}: {bic}" in text, text
    # The DRAWN fit is the chosen one, refit properly — with its CIs, and therefore with
    # the identifiability warnings that are statements about those CIs. The CIs arrive
    # deferred (the bootstrap is off the click path; see the note in test_plots_panel's
    # confidence-interval test), so the guarantee is that they ARRIVE, not that they are
    # already there the instant Auto flips the spin.
    assert apo500.fit_result().meta["n_components"] == 4
    _settle(apo500)
    fit = apo500.fit_result()
    assert fit.meta["n_components"] == 4, "settling changed which fit is drawn"
    assert fit.ci, "Auto mode must not hand back a point fit stripped of its error bars"


def test_auto_bic_is_offered_only_where_a_component_count_means_something(apo700):
    """``gamma``, ``inverse-gaussian`` and ``exponential + gamma`` have a fixed
    parameterisation; an Auto-k over them would be theatre."""
    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("exponential")
    assert apo700._fit_k_auto.isEnabled()
    apo700.set_fit("gamma (peaked)")
    assert apo700._fit_k_auto.isEnabled() is False
    apo700.set_fit("none")
    assert apo700._fit_k_auto.isEnabled() is False


# ================================================================================================
# Medium-11 / Low-20 -- THE WARNINGS REACH THE PANE, NOT JUST meta
# ================================================================================================

def test_the_identifiability_warnings_reach_the_details_pane(apo700):
    """The -700 mV k=3 depth fit is the report's Medium-11 example. Its warnings are
    useless in ``meta`` alone: the pane is where the mu list is read."""
    apo700.set_histogram_column("depth_pA")
    apo700.set_fit("gaussian-mixture")
    apo700.set_fit_components(3)
    # Two of these three warnings are statements ABOUT THE INTERVALS ("wider than
    # 2*sigma", "the weight's CI spans..."), so they cannot exist until the bootstrap
    # that produces the intervals has. That bootstrap is deferred now — it is what froze
    # the window for seconds on a click — and while it is pending the pane says so
    # ("95% CONFIDENCE INTERVALS: still computing"). COMPONENTS NOT SEPARATED needs no
    # interval and is there immediately, which is what the un-settled read below pins.
    early = apo700.details_text()
    assert "COMPONENTS NOT SEPARATED" in early, early
    _settle(apo700)
    text = apo700.details_text()
    assert "UNIDENTIFIED COMPONENT(S)" in text, text
    assert "UNRESOLVED WEIGHT(S)" in text, text
    assert "COMPONENTS NOT SEPARATED" in text, text


def test_the_degenerate_mixture_warning_reaches_the_details_pane(apo700):
    """-700 mV k=2 returns tau_1 = tau_2 = 19.809 ms and the pane prints
    "τ = 19.81, 19.81 ms" — two numbers that are one number (report Low-20)."""
    apo700.set_histogram_column("dwell_s")
    apo700.set_time_unit("ms")
    apo700.set_fit("exponential")
    apo700.set_fit_components(2)
    text = apo700.details_text()
    assert "τ = 19.81, 19.81 ms" in text, text
    assert "DEGENERATE MIXTURE" in text, text


# ================================================================================================
# THE DIAGNOSTICS MAY NOT COST MORE THAN THE THING THEY DIAGNOSE
# ================================================================================================
# The cross-family scan was gated on the goodness-of-fit verdict being REJECT and costed as
# an exceptional path — "0 extra on a well-specified fit". True, and beside the point on
# this dataset: the exponential is rejected on ALL THREE of the owner's recordings (report
# §3), so the gate is open on every redraw and he always pays. Measured, median of 3, on a
# real window: −600 mV (n = 239) 2.5 ms with no fit, 331 ms exponential, 656 ms
# exponential+gamma; −700 mV (n = 986) 371 ms / 1066 ms. Every bin change, every unit
# change and every log-x toggle paid it again, for numbers that cannot have moved.
#
# The rows are a function of the sample and the censoring, so they are remembered per
# (sample digest, t_min, t_max, k). After: 14.7 ms and 171 ms on −600 mV, 18.4 ms and
# 213 ms on −700 mV — what is left is the FIT itself, not the diagnosis of it.

def test_the_expensive_notes_are_not_recomputed_on_a_redraw_that_changed_nothing(apo700):
    """Same numbers, much faster — and the SAME TEXT, which is the part that matters."""
    import time

    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("exponential")
    apo700._note_cache.clear()

    t0 = time.perf_counter()
    apo700._redraw()
    cold = time.perf_counter() - t0
    _settle(apo700)   # the advisory notes are computed off the redraw path
    first = apo700.details_text()

    t0 = time.perf_counter()
    apo700._redraw()
    warm = time.perf_counter() - t0

    assert apo700.details_text() == first, "the cache changed what the panel says"
    assert "MODEL COMPARISON" in first, first        # the gate really is open here

    # NOT a cold/warm ratio any more, and the reason is the point of the change. The
    # scans no longer run inside a redraw at all, so the COLD redraw is already cheap
    # (measured 25 ms here, against ~300 ms when it computed inline) and a 3x ratio
    # cannot be met by a correct implementation. What actually has to hold is that a
    # redraw which changed nothing schedules no work and refits nothing:
    assert not apo700._pending_notes, "an unchanged redraw queued the scans again"
    assert not apo700._note_timer.isActive(), "an unchanged redraw re-armed the settle timer"
    assert warm < 0.150, f"a warm redraw took {warm * 1e3:.1f} ms"


def test_the_cache_is_keyed_on_the_numbers_not_on_the_draw(apo700, apo500):
    """A different sample must not be answered out of another sample's cache.

    Keyed on the sample's BYTES: `_redraw` rebuilds the array from the frame every time,
    so anything keyed on identity would miss on every call (no cache at all) and anything
    keyed on length alone would collide across recordings.
    """
    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("exponential")
    apo700._redraw()
    keys_700 = set(apo700._note_cache)

    apo500.set_histogram_column("dwell_s")
    apo500.set_fit("exponential")
    apo500._redraw()

    digests_700 = {k[1] for k in keys_700}
    digests_500 = {k[1] for k in apo500._note_cache}
    assert digests_700.isdisjoint(digests_500), (digests_700, digests_500)


def test_changing_the_component_count_recomputes(apo700):
    """The key carries k, so the table a k=2 fit is compared against is a k=2 table."""
    apo700.set_histogram_column("dwell_s")
    apo700.set_fit("exponential")
    apo700.set_fit_components(1)
    apo700._redraw()
    one = apo700.details_text()
    apo700.set_fit_components(2)
    apo700._redraw()
    assert apo700.details_text() != one
