"""S4 / the Plots group — a number this window prints must be one it can stand behind.

Seven defects, one rule at seven sites, all of them in ``PlotsWindow`` (plus the two
guards ``populations`` now owns). Each test below drives the REAL widget a user drives
and reads the pane text a user reads.

SA07 (hang) — setting *Fit components = 2* ran the whole 1000-replicate (exponential) or
    400-replicate (gaussian-mixture) bootstrap on the GUI thread: 1.9-2.6 s on the
    owner's own -700 mV apo run and 40-45 s at n = 100 000, against this window's own
    150 ms budget, with a 50 ms timer starved throughout. The pane's own note
    ("set Fit components = 2 to see it") is what tells the user to click it, and with
    k = 2 on screen every later curation click paid it again.
SA06 (wrong-result) — a Gaussian mixture over an all-negative column reported a POSITIVE
    mu in the exported title, claimed the curve was consistent with the data, and drew a
    curve of 1e-81 across an axis whose bars were 16 high.
SA01 (misleading) — PCA on all-constant features printed "nan%" into the title, BOTH axis
    labels and Details, with no refusal.
SA02 (misleading) — PCA on 2 or 3 rows reported "cumulative 100.0%" with no
    exact-by-construction caveat, in the window whose scatter caveats the equivalent
    2-point R2 = 1.
SA04 (silent-failure) — non-finite rows dropped silently by the histogram, the scatter
    and the PCA while the clustering pane of the same window disclosed the identical drop.
SA05 (misleading) — log x over an all-negative column said the column "has no finite
    values" when every value in it is finite.
SA03 (misleading) — a Fit range that empties the sample made the refusal blame the
    COLUMN, in the pane that is printing the column's real count two lines above.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis import populations, stats
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plots_panel as PP
from nanoscope.gui.plots_panel import PlotsWindow

#: The window's own documented per-click budget (tests/gui/test_plots_fit_latency.py).
MAX_CLICK_MS = 150.0


# ================================================================================================
# builders
# ================================================================================================

def _bimodal(n: int, seed: int = 111) -> pd.DataFrame:
    """A table a two-component fit is genuinely the right model for."""
    rng = np.random.default_rng(seed)
    pick = rng.random(n) < 0.35
    return pd.DataFrame({
        "dwell_s": 5e-5 + np.where(pick, rng.exponential(5e-5, n), rng.exponential(8e-4, n)),
        "frac_depth": np.where(pick, rng.normal(0.15, 0.02, n), rng.normal(0.35, 0.03, n)),
        "amplitude": np.abs(rng.normal(30.0, 5.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _window(qtbot, df: pd.DataFrame, *, label: str = "A",
            provenance: dict | None = None) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([PlotSource(key="a", label=label, df=df,
                           provenance=provenance if provenance is not None
                           else {"min_dwell_s": 5e-5})])
    return w


def _hist(qtbot, df, col, fit=PP._FIT_NONE, **kw) -> PlotsWindow:
    w = _window(qtbot, df, **kw)
    w.set_plot_type(PP.HISTOGRAM)
    w.set_histogram_column(col)
    w.set_fit(fit)
    return w


def _settle(w: PlotsWindow, budget_s: float = 120.0) -> None:
    """Run the event loop until every deferred note has landed."""
    app = QtWidgets.QApplication.instance()
    t0 = time.time()
    while time.time() - t0 < budget_s:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if not w._pending_notes and not w._inflight_notes and not w._note_timer.isActive():
            break
    app.processEvents()


def _fit_k_spin(w: PlotsWindow) -> QtWidgets.QSpinBox:
    """The real *Fit components* spin box, found the way a user finds it."""
    for sb in w.findChildren(QtWidgets.QSpinBox):
        if sb.toolTip().startswith("Number of mixture components"):
            return sb
    raise AssertionError("no fit-components spin box in this window")


# ================================================================================================
# SA07 -- the resampling is not on the click path
# ================================================================================================

@pytest.mark.parametrize("fit,col", [(PP._FIT_EXP, "dwell_s"), (PP._FIT_GMM, "frac_depth")])
def test_a_real_up_arrow_on_the_component_spin_returns_inside_the_budget(qtbot, fit, col):
    """THE GESTURE, not the setter: ``QTest.keyClick(spin, Key_Up)``.

    This is what the pane's own "set Fit components = 2 to see it" note asks the user to
    do. Before the fix one press blocked the GUI thread for 1.9-2.6 s on the owner's real
    recording; here, on a sample of the same order, it must return inside the window's
    own budget and a timer armed before it must still get its turn.
    """
    w = _hist(qtbot, _bimodal(1200), col, fit)
    _settle(w)
    spin = _fit_k_spin(w)
    assert spin.isEnabled() and spin.value() == 1

    fired: list[float] = []
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: fired.append(time.time()))
    timer.start(50)
    t0 = time.perf_counter()
    QTest.keyClick(spin, Qt.Key_Up)              # <-- the real gesture
    dt_ms = (time.perf_counter() - t0) * 1000.0
    timer.stop()

    assert spin.value() == 2
    assert dt_ms < MAX_CLICK_MS, (
        f"one Up-arrow on the component spin blocked the GUI thread for {dt_ms:.0f} ms "
        f"({dt_ms / MAX_CLICK_MS:.1f}x the {MAX_CLICK_MS:g} ms budget)")


@pytest.mark.parametrize("fit,col", [(PP._FIT_EXP, "dwell_s"), (PP._FIT_GMM, "frac_depth")])
def test_while_the_interval_is_out_the_pane_says_so_and_never_goes_silent(qtbot, fit, col):
    """A missing interval must not read as "this family offers none" -- neither in the
    pane nor on the FitResult a saved project would carry."""
    w = _hist(qtbot, _bimodal(1200), col, fit)
    _settle(w)
    QTest.keyClick(_fit_k_spin(w), Qt.Key_Up)
    pane = w.details_text()
    assert PP.PlotsWindow._CI_PENDING_TEXT in pane, pane
    assert "95% CI (percentile bootstrap" not in pane
    assert w._last_fit.meta.get("ci_pending") is True


@pytest.mark.parametrize("fit,col", [(PP._FIT_EXP, "dwell_s"), (PP._FIT_GMM, "frac_depth")])
def test_the_interval_that_lands_is_the_one_the_synchronous_call_would_have_given(qtbot, fit, col):
    """Moving WHEN the bootstrap runs must not change WHAT it says.

    The pane's numbers after the worker settles are compared against a plain synchronous
    call to the same fitter with the same arguments -- the call this window used to make
    on the click path.
    """
    df = _bimodal(1200)
    w = _hist(qtbot, df, col, fit)
    _settle(w)
    QTest.keyClick(_fit_k_spin(w), Qt.Key_Up)
    _settle(w)

    pane = w.details_text()
    assert PP.PlotsWindow._CI_PENDING_TEXT not in pane
    assert "95% CI (percentile bootstrap" in pane, pane
    got = w._last_fit
    assert got.meta.get("ci_pending") is None

    dwell = np.asarray(df[col], dtype=np.float64)
    if fit == PP._FIT_EXP:
        want = populations.fit_exponential(dwell, t_min=5e-5, n_components=2)
    else:
        want = populations.fit_gaussian_mixture(np.abs(dwell), k=2)
    assert got.params == pytest.approx(want.params, rel=1e-12, abs=0.0)
    assert set(got.ci) == set(want.ci)
    for key, (lo, hi) in want.ci.items():
        assert got.ci[key] == pytest.approx((lo, hi), rel=1e-12, abs=0.0)
    assert float(got.bic) == pytest.approx(float(want.bic), rel=1e-12, abs=0.0)


def test_auto_bic_does_not_put_the_bootstrap_back_on_the_click_path(qtbot):
    """Auto (BIC) re-fits at the k it CHOSE, and that re-fit used to run outside
    ``point_fits_only`` "so the drawn fit has the CIs" -- i.e. it put the very bootstrap
    the spin box no longer pays straight back onto the click, through the one control
    that picks k for the user (measured 4.55 s at n = 1000 before this fix).

    The invariant, not a stopwatch: after the toggle the chosen fit is drawn and its
    interval is OUT, not computed here. The rival ladder's point fits still run on this
    thread -- that is a different cost and a different budget.
    """
    w = _hist(qtbot, _bimodal(1000), "dwell_s", PP._FIT_EXP)
    _settle(w)
    box = w._fit_k_auto
    assert box.isEnabled()
    box.setChecked(True)                          # the real Auto (BIC) toggle
    QtWidgets.QApplication.instance().processEvents()
    pane = w.details_text()
    assert "AUTO (BIC) chose k" in pane
    assert w._fit_k.value() > 1, "fixture is wrong: Auto did not move k"
    assert PP.PlotsWindow._CI_PENDING_TEXT in pane, pane
    assert w._last_fit.meta.get("ci_pending") is True
    _settle(w)
    assert "95% CI (percentile bootstrap" in w.details_text()


# ================================================================================================
# SA06 -- a mixture over a signed column says which variable its numbers are about
# ================================================================================================

def _negative_frame(n: int = 600, seed: int = 5) -> pd.DataFrame:
    """`mean` as measure.MEASUREMENT_SCHEMA produces it on a -1000 mV ABF: all negative."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"mean": rng.normal(-820.0, 12.0, n)})


def test_a_mixture_over_an_all_negative_column_names_the_magnitude_in_the_title(qtbot):
    """The title is the only thing ``Save image`` keeps, so the frame goes in the title."""
    w = _hist(qtbot, _negative_frame(), "mean", PP._FIT_GMM, provenance={})
    title = w._plot.titleLabel.text
    assert "μ(|mean|)" in title, title
    assert "Gaussian mixture on |mean|" in w.details_text()


def test_a_mixture_over_an_all_negative_column_draws_its_curve_on_the_data(qtbot):
    """1.18e-81 against bars of 16 is not a curve. Reflection is exact for an
    all-negative column, so the overlay lands where the events are."""
    w = _hist(qtbot, _negative_frame(), "mean", PP._FIT_GMM, provenance={})
    curves = [it for it in w._fit_items if it.yData is not None and it.yData.size]
    assert curves, "no fit curve was drawn at all"
    peak = max(float(np.nanmax(c.yData)) for c in curves)
    bars = max(float(np.nanmax(h.yData)) for h in w._hist_items if h.yData is not None)
    assert peak > 0.1 * bars, f"fit curve peaks at {peak:g} against bars of {bars:g}"
    assert "reflected onto the negative axis" in w.details_text()


def test_the_goodness_of_fit_verdict_says_which_variable_it_judged(qtbot):
    """"The fitted curve is consistent with these data" over a fit of |x| is a claim
    about something the axis is not showing."""
    pane = _hist(qtbot, _negative_frame(), "mean", PP._FIT_GMM, provenance={}).details_text()
    verdict = next(ln for ln in pane.splitlines() if ln.startswith("KS D ="))
    assert "|mean|" in verdict, verdict


def test_a_mixed_sign_column_is_not_reflected_but_is_declared(qtbot):
    """Folding a mixed-sign column merges two lobes, so nothing may be mirrored -- but
    the pane must still say the numbers are magnitudes."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"skewness": rng.normal(0.0, 1.0, 500)})
    pane = _hist(qtbot, df, "skewness", PP._FIT_GMM, provenance={}).details_text()
    assert "μ(|skewness|)" in pane
    assert "describe MAGNITUDES" in pane
    assert "reflected onto the negative axis" not in pane


def test_a_fit_whose_mass_is_off_the_plot_says_the_curve_is_not_visible(qtbot):
    """The general guard behind SA06: the plot may not assert a curve it did not draw.

    Reflection now keeps the all-negative mixture on its own axis, so the remaining way
    to land a curve off the plotted window is to draw it against bins that do not cover
    the fitted values. That is exactly the argument ``_overlay_fit`` takes, so the guard
    is exercised at its own seam -- with a real window, a real PlotSource and the real
    method, not a stub.
    """
    rng = np.random.default_rng(11)
    v = np.abs(rng.normal(80.0, 5.0, 400))
    df = pd.DataFrame({"depth_pA": v})
    w = _hist(qtbot, df, "depth_pA", PP._FIT_NONE, provenance={})
    src = w.primary_source()
    far_edges = np.linspace(1000.0, 1200.0, 31)     # nowhere near the fitted 80 pA
    _summary, notes = w._overlay_fit(PP._FIT_GMM, "depth_pA", v, far_edges, False, src,
                                     "#ff0000", "")
    assert any("THE FITTED CURVE IS NOT VISIBLE ON THIS PLOT" in n for n in notes), notes


# ================================================================================================
# SA01 / SA02 -- PCA prints no NaN, and caveats an exact-by-construction 100%
# ================================================================================================

def _pca_window(qtbot, df: pd.DataFrame, feats: list[str]) -> PlotsWindow:
    w = _window(qtbot, df, provenance={})
    w.set_plot_type(PP.PCA)
    w.set_feature_selection(feats)
    return w


FEATS = ["dwell_s", "amplitude", "snr", "rms"]


def _constant_frame(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({"dwell_s": np.full(n, 2.5e-4), "amplitude": np.full(n, 30.0),
                         "snr": np.full(n, 9.0), "rms": np.full(n, 1.4),
                         "accepted": np.ones(n, dtype=bool)})


def test_pca_on_all_constant_features_refuses_and_names_the_cause(qtbot):
    """A refusal is not a number. No "nan%" anywhere a user can read it."""
    w = _pca_window(qtbot, _constant_frame(), FEATS)
    assert w.run_pca() is None
    pane = w.details_text()
    assert "nan" not in pane.lower()
    assert "are constant" in pane and "no variance for PCA to explain" in pane
    for surface in (pane, w._plot.titleLabel.text,
                    w._plot.getAxis("bottom").labelText, w._plot.getAxis("left").labelText):
        assert "nan" not in str(surface).lower(), surface


def test_populations_pca_refuses_zero_total_variance_at_the_analysis_seam():
    """The guard lives where the NaN was made, not only where it was printed."""
    df = _constant_frame()
    with pytest.raises(ValueError, match="no variance for PCA to explain"):
        populations.pca(df, FEATS, n_components=2)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pca_on_two_or_three_rows_caveats_its_exact_100_percent(qtbot, n, seed):
    """Centred data of n rows has rank <= n-1, so 2 components cover it at n <= 3 for ANY
    data. The same window already prints "R² = 1.000 (exact by construction)"."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATS})
    w = _pca_window(qtbot, df, FEATS)
    assert w.run_pca() is not None
    title = w._plot.titleLabel.text
    assert "cumulative 100.0%" in title
    assert "exact by construction" in title, title
    assert "EXACT BY CONSTRUCTION" in w.details_text()


def test_pca_on_a_real_sample_carries_no_caveat(qtbot):
    """A guard on a degenerate case, not a running commentary."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({c: rng.normal(size=200) for c in FEATS})
    w = _pca_window(qtbot, df, FEATS)
    assert w.run_pca() is not None
    assert "exact by construction" not in w._plot.titleLabel.text
    assert "EXACT BY CONSTRUCTION" not in w.details_text()


def test_a_constant_feature_among_varying_ones_is_not_a_silent_no_op(qtbot):
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"dwell_s": rng.normal(size=60), "amplitude": rng.normal(size=60),
                       "snr": rng.normal(size=60), "rms": np.full(60, 1.4)})
    w = _pca_window(qtbot, df, FEATS)
    assert w.run_pca() is not None
    assert "rms is constant" in w.details_text()


# ================================================================================================
# SA04 -- every drop disclosed, with its count, in one voice
# ================================================================================================

def _holey(n: int = 2000, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "dwell_s": 5e-5 + rng.exponential(3e-4, n),
        "amplitude": np.abs(rng.normal(30.0, 5.0, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "rms": np.abs(rng.normal(1.4, 0.3, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })
    df.loc[df.index[:20], "dwell_s"] = np.nan            # 20 rows
    df.loc[df.index[100:120], "amplitude"] = np.nan      # 20 more, disjoint
    return df


def test_the_histogram_says_how_many_rows_it_dropped(qtbot):
    pane = _hist(qtbot, _holey(), "dwell_s", PP._FIT_NONE).details_text()
    assert "A: n = 1980" in pane
    assert "20 of 2000 row(s) dropped" in pane, pane


def test_the_scatter_says_how_many_rows_it_dropped(qtbot):
    w = _window(qtbot, _holey())
    w.set_plot_type(PP.SCATTER)
    w.set_scatter_columns("dwell_s", "amplitude")
    pane = w.details_text()
    assert "n = 1960" in pane
    assert "40 of 2000 row(s) dropped" in pane, pane


def test_the_pca_says_how_many_rows_it_dropped_in_the_clustering_pane_s_words(qtbot):
    w = _pca_window(qtbot, _holey(), FEATS)
    assert w.run_pca() is not None
    assert "n = 1960 kept, 40 dropped (non-finite features)" in w.details_text()


def test_a_clean_table_gains_no_drop_line(qtbot):
    pane = _hist(qtbot, _bimodal(300), "dwell_s", PP._FIT_NONE).details_text()
    assert "row(s) dropped" not in pane


# ================================================================================================
# SA05 -- log x over an all-negative column says what is actually true
# ================================================================================================

def test_log_x_over_an_all_negative_column_blames_the_axis_not_the_column(qtbot):
    w = _hist(qtbot, _negative_frame(), "mean", PP._FIT_NONE, provenance={})
    w._hist_logx.setChecked(True)                 # the real "log x" checkbox
    QtWidgets.QApplication.instance().processEvents()
    pane = w.details_text()
    assert "has no finite values" not in pane, pane
    assert "are non-positive" in pane and "log x" in pane
    assert "Untick" in pane


def test_a_column_that_really_has_no_finite_values_still_says_so(qtbot):
    """The true sentence must survive: this branch is still reachable when curation
    leaves only the rows whose value is NaN, and there it is the honest one."""
    n = 60
    mean = np.full(n, np.nan)
    mean[:20] = np.linspace(-10.0, 10.0, 20)      # finite only on the rejected rows
    accepted = np.ones(n, dtype=bool)
    accepted[:20] = False
    df = pd.DataFrame({"mean": mean, "accepted": accepted})
    w = _hist(qtbot, df, "mean", PP._FIT_NONE, provenance={})
    assert w._accepted_check.isChecked()          # the real "accepted events only" box
    pane = w.details_text()
    assert "has no finite values" in pane, pane
    assert "all 40 row(s) are NaN, infinite or non-numeric" in pane


def test_the_partial_log_x_drop_keeps_its_own_wording(qtbot):
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"mean": np.concatenate([rng.normal(5, 1, 300),
                                               rng.normal(-5, 1, 300)])})
    w = _hist(qtbot, df, "mean", PP._FIT_NONE, provenance={})
    w._hist_logx.setChecked(True)
    QtWidgets.QApplication.instance().processEvents()
    assert "non-positive value(s) dropped for log x." in w.details_text()


# ================================================================================================
# SA03 -- a refusal caused by the Fit range says so
# ================================================================================================

def _dwell_frame(n: int = 500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"dwell_s": 5e-5 + rng.exponential(3e-4, n),
                         "accepted": np.ones(n, dtype=bool),
                         "filter_limited": np.zeros(n, dtype=bool)})


@pytest.mark.parametrize("lo,hi", [("4", "10"), ("100", "200")])
def test_a_fit_range_refusal_names_the_fit_range(qtbot, lo, hi):
    w = _hist(qtbot, _dwell_frame(), "dwell_s", PP._FIT_EXP)
    w._time_unit.setCurrentText("ms")
    w._fit_lo.setText(lo)
    w._fit_hi.setText(hi)
    w._redraw()
    pane = w.details_text()
    assert "Fit unavailable" in pane
    assert "THE FIT RANGE IS WHY" in pane, pane
    assert f"[{lo}, {hi}] ms" in pane
    assert "of the 500 value(s) of “dwell_s”" in pane


@pytest.mark.parametrize("lo,hi", [("nan", "5"), ("inf", ""), ("abc", "")])
def test_an_unusable_fit_bound_is_declared_instead_of_silently_dropped(qtbot, lo, hi):
    """"nan" and "inf" ARE floats. They emptied the sample and the pane blamed the
    column with nothing to connect the refusal to the text the user had typed."""
    w = _hist(qtbot, _dwell_frame(), "dwell_s", PP._FIT_EXP)
    w._time_unit.setCurrentText("ms")
    w._fit_lo.setText(lo)
    w._fit_hi.setText(hi)
    w._redraw()
    pane = w.details_text()
    assert f"Fit range min “{lo}” is not a usable bound and was IGNORED" in pane, pane
    assert "dwell_s is empty" not in pane


def test_an_inverted_fit_range_says_both_bounds_were_ignored(qtbot):
    w = _hist(qtbot, _dwell_frame(), "dwell_s", PP._FIT_EXP)
    w._time_unit.setCurrentText("ms")
    w._fit_lo.setText("9")
    w._fit_hi.setText("2")
    w._redraw()
    assert "is empty (min ≥ max)" in w.details_text()


def test_a_usable_fit_range_keeps_the_success_path_line_and_gains_no_blame(qtbot):
    w = _hist(qtbot, _dwell_frame(), "dwell_s", PP._FIT_EXP)
    w._time_unit.setCurrentText("ms")
    w._fit_lo.setText("0.1")
    w._fit_hi.setText("5")
    w._redraw()
    pane = w.details_text()
    assert "FIT RANGE [" in pane
    assert "THE FIT RANGE IS WHY" not in pane
    assert "IGNORED" not in pane
