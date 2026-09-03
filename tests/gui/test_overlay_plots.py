"""Overlaying two of anything — the capability the Plots window did not have.

The window used to render ONE source: whichever detection was current, or the
manual-measurement table. A two-pore or two-segment comparison — the comparison a
nanopore paper is built out of — could not be expressed at all, and after
``run_clustering`` found two populations the histogram had no way to draw their
distributions apart. These tests pin the fix, and they check the things that make
an overlay *mean* something rather than merely look busy:

* two sources draw two histograms / two scatters, in DISTINCT colours, with a
  legend entry naming each;
* the bin EDGES are shared, and density normalization makes a 900-event and a
  90-event recording comparable — asserted as areas, not as a screenshot;
* **each source is fit with ITS OWN provenance-resolved ``t_min``** (the
  regression guard for the fit-provenance work: two sources detected at different
  ``min_dwell_s`` must produce the two taus an independent per-source fit gives,
  never one floor applied to both, and never a single pooled fit);
* the histogram splits by cluster label, and colour has exactly one owner —
  source while an overlay is up, cluster only when a single source is drawn;
* comparing an absolute-pA column across uncalibrated recordings says so;
* the legend is readable on the light background as well as the dark one.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nanoscope.analysis import modules, populations
from nanoscope.analysis.events import SCHEMA
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import PlotsWindow

FS = 250_000.0

# The default fixture is TWO PORES DETECTED UNDER ONE BASELINE WINDOW: they share a
# baseline_fingerprint, so the baseline guard has nothing to say and these tests keep
# exercising the GAIN guard in isolation. Tests about the baseline guard hand
# ``_source(..., baseline=...)`` two different fingerprints (or None) on purpose.
_SHARED_BASELINE = "sha256:0000aaaa1111"


# ================================================================================================
# builders
# ================================================================================================

def _frame(n: int, *, tau: float, t_min: float, seed: int = 0,
           depth: float = 30.0) -> pd.DataFrame:
    """A full 20-column event frame whose dwell times are exponential ABOVE ``t_min``.

    Left-truncated on purpose: that is what a detector produces, and it is what
    makes the fitted tau depend on the ``t_min`` the fit is told about.
    """
    rng = np.random.default_rng(seed)
    dwell = t_min + rng.exponential(tau, n)
    cols = {}
    for name, (dtype, _u, _d) in SCHEMA.items():
        if dtype == "int64":
            cols[name] = np.arange(n, dtype=np.int64)
        elif dtype == "bool":
            cols[name] = np.zeros(n, dtype=bool)
        elif dtype == "string":
            cols[name] = pd.Series([""] * n, dtype="string")
        else:
            cols[name] = np.zeros(n, dtype=np.float64)
    cols["dwell_s"] = dwell
    cols["baseline_pA"] = np.full(n, 120.0)
    cols["depth_pA"] = np.abs(rng.normal(depth, depth / 10, n))
    cols["frac_depth"] = cols["depth_pA"] / 120.0
    cols["snr"] = np.abs(rng.normal(9.0, 1.5, n))
    cols["accepted"] = np.ones(n, dtype=bool)
    return pd.DataFrame(cols)


def _source(key: str, df: pd.DataFrame, *, colour: str, min_dwell_s: float | None = None,
            fingerprint=(1.0, 0.0), label: str | None = None,
            baseline: str | None = _SHARED_BASELINE,
            warnings: tuple[str, ...] = ()) -> PlotSource:
    prov = {"fs": FS}
    if min_dwell_s is not None:
        prov["min_dwell_s"] = min_dwell_s
    if baseline is not None:
        prov["baseline_fingerprint"] = baseline
    return PlotSource(key=key, doc_key=key, label=label or key, df=df, provenance=prov,
                      colour=colour, doc_id=key, channel=0, module_id="detect.schmitt",
                      display_name=label or key, calibration_fingerprint=fingerprint,
                      warnings=warnings)


@pytest.fixture
def window(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    return w


def _two_pores(window, *, n_a=900, n_b=90, t_min_a=5e-5, t_min_b=5e-5,
               tau_a=3e-4, tau_b=3e-4,
               baseline_a=_SHARED_BASELINE, baseline_b=_SHARED_BASELINE):
    a = _source("pore-a", _frame(n_a, tau=tau_a, t_min=t_min_a, seed=1),
                colour="#0072B2", min_dwell_s=t_min_a, label="pore A",
                baseline=baseline_a)
    b = _source("pore-b", _frame(n_b, tau=tau_b, t_min=t_min_b, seed=2),
                colour="#D55E00", min_dwell_s=t_min_b, label="pore B",
                baseline=baseline_b)
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    return a, b


def _pen_colour(item) -> str:
    return item.opts["pen"].color().name()


def _areas(item) -> float:
    """∫ y dx over a step histogram item (x are the shared edges, y the heights)."""
    x = np.asarray(item.xData, dtype=np.float64)
    y = np.asarray(item.yData, dtype=np.float64)
    return float(np.sum(y * np.diff(x)))


# ================================================================================================
# 1. the source model reaches the window
# ================================================================================================

def test_the_window_offers_every_source_and_preselects_the_current_one(window):
    a = _source("pore-a", _frame(50, tau=3e-4, t_min=5e-5), colour="#0072B2", label="pore A")
    b = _source("pore-b", _frame(50, tau=3e-4, t_min=5e-5, seed=3), colour="#D55E00",
                label="pore B")
    b = PlotSource(**{**b.__dict__, "current": True})
    window.set_data([a, b])

    assert window.source_labels() == ["pore A", "pore B"]
    # Only the app's active selection is ticked on the first push — opening Plots
    # shows what the user was already looking at, not everything at once.
    assert window.selected_source_keys() == ["pore-b"]
    assert window.current_source_name() == "pore B"


def test_a_refresh_keeps_the_ticked_set(window):
    a, b = _two_pores(window)
    assert window.selected_source_keys() == ["pore-a", "pore-b"]
    window.set_data([a, b])            # what MainWindow's Refresh does
    assert window.selected_source_keys() == ["pore-a", "pore-b"]


def test_the_legacy_mapping_form_still_works(window):
    """A bare {label: frame} snapshot is still accepted (tests, notebook sessions)."""
    window.set_data({"one": _frame(40, tau=3e-4, t_min=5e-5)},
                    {"one": {"min_dwell_s": 5e-5, "fs": FS}})
    assert window.source_labels() == ["one"]
    assert window.primary_source().provenance["min_dwell_s"] == 5e-5


# ================================================================================================
# 2. two histograms, two colours, two legend entries
# ================================================================================================

def test_two_sources_draw_two_histograms_in_distinct_colours_with_a_legend(window):
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("none")

    assert len(window._hist_items) == 2, "an overlay must draw one histogram per source"
    colours = [_pen_colour(i) for i in window._hist_items]
    assert colours[0] != colours[1], "the two sources must be told apart by colour"
    assert colours == ["#0072b2", "#d55e00"]        # each source's own Okabe-Ito ink

    labels = window.legend_labels()
    assert len(labels) == 2
    assert any("pore A" in t for t in labels)
    assert any("pore B" in t for t in labels)
    # Both are in the scene, not just in our bookkeeping.
    drawn = list(window.plot_item().listDataItems())
    assert all(item in drawn for item in window._hist_items)


def test_a_single_source_keeps_the_style_data_colour(window):
    """No regression for the one-pore case: colour still follows the Style group."""
    _two_pores(window)
    window.set_sources(["pore-a"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_data_color("#123456")
    assert len(window._hist_items) == 1
    assert _pen_colour(window._hist_items[0]) == "#123456"


def test_two_sources_draw_two_scatters_with_a_legend(window):
    _two_pores(window)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "frac_depth")

    assert len(window._scatter_items) == 2
    labels = window.legend_labels()
    assert any("pore A" in t for t in labels) and any("pore B" in t for t in labels)
    # Each cloud holds its own source's rows.
    assert len(window._scatter_items[0].getData()[0]) == 900
    assert len(window._scatter_items[1].getData()[0]) == 90


def test_unticking_back_to_one_source_removes_the_second_curve_and_the_legend(window):
    _two_pores(window)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "frac_depth")
    assert len(window._scatter_items) == 2

    window.set_sources(["pore-a"])
    assert len(window._scatter_items) == 1
    assert window.legend_labels() == [], "stale legend rows must not survive a redraw"


# ================================================================================================
# 3. shared bins + normalization — the thing that makes an overlay comparable
# ================================================================================================

def test_overlaid_sources_share_one_bin_edge_set(window):
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_bins(24)

    a_edges = np.asarray(window._hist_items[0].xData, dtype=np.float64)
    b_edges = np.asarray(window._hist_items[1].xData, dtype=np.float64)
    assert a_edges.size == 25
    np.testing.assert_allclose(a_edges, b_edges)
    # The shared range spans BOTH samples, not just the first one.
    lo = min(float(s.df["dwell_s"].min()) for s in window.selected_sources())
    hi = max(float(s.df["dwell_s"].max()) for s in window.selected_sources())
    assert a_edges[0] == pytest.approx(lo)
    assert a_edges[-1] == pytest.approx(hi)


def test_density_defaults_on_and_makes_unequal_n_sources_comparable(window):
    """A 900-event and a 90-event recording as raw counts is a decade apart and
    tells you nothing. As densities both integrate to 1 — the SHAPES are what the
    comparison is about."""
    _two_pores(window, n_a=900, n_b=90)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")

    assert window.normalization() == "density", "density is the multi-source default"
    areas = [_areas(i) for i in window._hist_items]
    assert areas[0] == pytest.approx(1.0, rel=1e-9)
    assert areas[1] == pytest.approx(1.0, rel=1e-9)
    # ...and the peak heights are now the same order of magnitude, which is exactly
    # what raw counts destroyed.
    peaks = [float(np.max(i.yData)) for i in window._hist_items]
    assert 0.2 < peaks[0] / peaks[1] < 5.0


def test_counts_are_available_and_are_raw_counts(window):
    _two_pores(window, n_a=900, n_b=90)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_normalization("count")

    sums = [float(np.sum(i.yData)) for i in window._hist_items]
    assert sums == [900.0, 90.0]


def test_probability_per_bin_sums_to_one_per_source(window):
    _two_pores(window, n_a=900, n_b=90)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_normalization("probability")

    for item in window._hist_items:
        assert float(np.sum(item.yData)) == pytest.approx(1.0, rel=1e-9)


def test_a_single_source_defaults_back_to_count(window):
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    assert window.normalization() == "density"
    window.set_sources(["pore-a"])
    assert window.normalization() == "count"


def test_an_explicit_normalization_choice_is_not_overridden(window):
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_normalization("probability")
    window.set_sources(["pore-a"])           # back to one source
    assert window.normalization() == "probability per bin"


def test_density_normalization_is_correct_for_log_bins(window):
    """Log bins have unequal widths; a density that divided by a single width would
    be wrong by a factor spanning decades."""
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_log_x(True)
    for item in window._hist_items:
        assert _areas(item) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("seed_a,seed_b,n_a,n_b", [
    (1, 2, 900, 90), (21, 22, 5000, 200), (5, 6, 250, 37), (11, 12, 640, 640),
])
def test_log_bins_never_drop_an_event_off_either_end(window, seed_a, seed_b, n_a, n_b):
    """Every event must land in a bin — on the LOG axis as well as the linear one.

    ``np.logspace(log10(lo), log10(hi), k)`` round-trips through log10 and back, so
    its outermost edges can come back an ulp *inside* ``[lo, hi]``; ``np.histogram``
    then silently discards the smallest and/or largest event. Whether it bites is
    pure floating-point luck of the data, which is why a single fixed pair of
    samples is not a guard — this sweeps several.

    The damage is exactly the comparability the overlay is for: the loss falls on
    whichever source owns each extreme, so under *density* the two
    series integrate to DIFFERENT values (measured 0.9998 vs 0.9950 for a 5000- and
    a 200-event source: the small-n source is penalised ~25x harder), and the drawn
    bars stop agreeing with the ``n = ...`` the legend prints.
    """
    a = _source("pore-a", _frame(n_a, tau=3e-4, t_min=5e-5, seed=seed_a),
                colour="#0072B2", min_dwell_s=5e-5, label="pore A")
    b = _source("pore-b", _frame(n_b, tau=9e-4, t_min=5e-5, seed=seed_b),
                colour="#D55E00", min_dwell_s=5e-5, label="pore B")
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_log_x(True)

    window.set_normalization("count")
    assert [float(np.sum(i.yData)) for i in window._hist_items] == [float(n_a), float(n_b)]

    window.set_normalization("density")
    for item in window._hist_items:
        assert _areas(item) == pytest.approx(1.0, abs=1e-12)

    window.set_normalization("probability")
    for item in window._hist_items:
        assert float(np.sum(item.yData)) == pytest.approx(1.0, abs=1e-12)


def test_the_shared_edges_span_the_data_exactly(window):
    """The end edges ARE the min and max of the drawn samples, not a rounded copy."""
    a, b = _two_pores(window, n_a=900, n_b=90)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    lo = min(float(a.df.dwell_s.min()), float(b.df.dwell_s.min()))
    hi = max(float(a.df.dwell_s.max()), float(b.df.dwell_s.max()))
    for log_x in (False, True):
        window.set_log_x(log_x)
        edges = np.asarray(window._hist_items[0].xData, dtype=np.float64)
        assert edges[0] == lo, (log_x, edges[0], lo)
        assert edges[-1] == hi, (log_x, edges[-1], hi)


# ================================================================================================
# 4. THE REGRESSION GUARD: per-source fits use per-source t_min
# ================================================================================================

def _independent_tau(df: pd.DataFrame, prov: dict) -> float:
    """The tau an independent fit of THIS source, with ITS OWN provenance, returns.

    Goes through the same public resolution the Fits panel uses, so the comparison
    is against the app's one answer rather than a reimplementation of it."""
    inp = modules.dwell_fit_inputs((df, prov), accepted_only=True, use_filter_limited=False)
    fit = populations.fit_exponential(inp.dwell, t_min=inp.t_min, t_max=inp.t_max,
                                      n_components=1)
    return float(fit.params["tau_1"])


#: Two DETECTORS, one dwell distribution. Both recordings happen to contain the same
#: dwell times (nothing shorter than 400 us was seen in either), but one detector was
#: gated at 20 us and the other at 400 us. The censored MLE is tau = mean(t) - t_min,
#: so the ONLY thing that can make the two fitted taus differ is the per-source floor
#: — which makes this the sharpest possible probe of "did each source use its own?".
_T_MIN_LOOSE, _T_MIN_TIGHT = 2e-5, 4e-4


def _same_data_different_floors(window, n: int = 600, tau: float = 8e-4):
    a = _source("pore-a", _frame(n, tau=tau, t_min=_T_MIN_TIGHT, seed=11),
                colour="#0072B2", min_dwell_s=_T_MIN_LOOSE, label="pore A")
    b = _source("pore-b", _frame(n, tau=tau, t_min=_T_MIN_TIGHT, seed=11),
                colour="#D55E00", min_dwell_s=_T_MIN_TIGHT, label="pore B")
    np.testing.assert_allclose(a.df["dwell_s"].to_numpy(), b.df["dwell_s"].to_numpy())
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")
    return a, b


def test_each_overlaid_source_is_fit_with_its_own_t_min(window):
    """Two recordings detected at DIFFERENT resolutions must yield the two taus their
    own detection floors imply. One source's t_min applied to the other's dwell times
    is exactly the bias the fit-provenance work removed; an overlay must not smuggle
    it back in, and pooling both samples into one fit would hide it entirely.

    The two sources here hold IDENTICAL dwell times, so identical taus would prove the
    fit had ignored one source's provenance — there is no other way for them to agree.
    """
    a, b = _same_data_different_floors(window)

    fits = window.fit_results()
    assert set(fits) == {"pore-a", "pore-b"}, "one fit per source — never a pooled one"

    # Each fit reproduces an independent fit of that source with that source's floor.
    assert fits["pore-a"].params["tau_1"] == pytest.approx(
        _independent_tau(a.df, dict(a.provenance)), rel=1e-9)
    assert fits["pore-b"].params["tau_1"] == pytest.approx(
        _independent_tau(b.df, dict(b.provenance)), rel=1e-9)

    # Each fit RECORDS the floor it used, and they are the two different real ones.
    assert fits["pore-a"].meta["t_min"] == pytest.approx(_T_MIN_LOOSE)
    assert fits["pore-b"].meta["t_min"] == pytest.approx(_T_MIN_TIGHT)
    assert fits["pore-a"].meta["t_min_source"] == "min_dwell_s"
    assert fits["pore-b"].meta["t_min_source"] == "min_dwell_s"

    # Same data, different detectors -> different taus, by exactly the floor gap.
    gap = fits["pore-a"].params["tau_1"] - fits["pore-b"].params["tau_1"]
    assert gap == pytest.approx(_T_MIN_TIGHT - _T_MIN_LOOSE, rel=1e-6)


def test_borrowing_the_other_sources_t_min_would_have_been_detected(window):
    """The guard above is only a guard if the wrong answer differs from the right one.

    tau_hat = mean(t) - t_min, so a shared floor would move one source's tau by the
    380 us gap between the two floors — a ~48% error against an 800 us tau. This
    computes that wrong answer explicitly and asserts the window did not produce it,
    so the assertions above cannot pass by accident."""
    a, _b = _same_data_different_floors(window)
    fits = window.fit_results()

    right = _independent_tau(a.df, {"min_dwell_s": _T_MIN_LOOSE, "fs": FS})
    wrong = _independent_tau(a.df, {"min_dwell_s": _T_MIN_TIGHT, "fs": FS})
    assert abs(right - wrong) == pytest.approx(_T_MIN_TIGHT - _T_MIN_LOOSE, rel=1e-6)
    assert abs(right - wrong) / right > 0.3, "the two answers must be far apart"

    got = fits["pore-a"].params["tau_1"]
    assert got == pytest.approx(right, rel=1e-9)
    assert abs(got - wrong) / right > 0.3, "pore A was fit with pore B's detection floor"


def test_two_sources_are_never_pooled_into_one_fit(window):
    """Two populations with genuinely different taus must come back as two numbers.

    Pooling them would return a single tau somewhere in between — a value that
    describes neither pore, reported as if it described both."""
    floor = 5e-5
    a = _source("pore-a", _frame(600, tau=3e-4, t_min=floor, seed=21),
                colour="#0072B2", min_dwell_s=floor, label="pore A")
    b = _source("pore-b", _frame(600, tau=1.2e-3, t_min=floor, seed=22),
                colour="#D55E00", min_dwell_s=floor, label="pore B")
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")

    fits = window.fit_results()
    tau_a = fits["pore-a"].params["tau_1"]
    tau_b = fits["pore-b"].params["tau_1"]
    assert tau_a == pytest.approx(_independent_tau(a.df, dict(a.provenance)), rel=1e-9)
    assert tau_b == pytest.approx(_independent_tau(b.df, dict(b.provenance)), rel=1e-9)

    pooled_tau = _independent_tau(pd.concat([a.df, b.df], ignore_index=True),
                                  dict(a.provenance))
    assert tau_a < pooled_tau < tau_b, "the pooled answer sits between the two real ones"
    assert abs(tau_a - pooled_tau) / pooled_tau > 0.2
    assert abs(tau_b - pooled_tau) / pooled_tau > 0.2
    # ...and each fit counted one source's events, not both.
    assert fits["pore-a"].n == len(a.df) and fits["pore-b"].n == len(b.df)


def test_the_details_pane_reports_every_source_fit_by_name(window):
    t_min_a, t_min_b = 2e-5, 4e-4
    a = _source("pore-a", _frame(400, tau=8e-4, t_min=t_min_a, seed=11),
                colour="#0072B2", min_dwell_s=t_min_a, label="pore A")
    b = _source("pore-b", _frame(400, tau=8e-4, t_min=t_min_b, seed=12),
                colour="#D55E00", min_dwell_s=t_min_b, label="pore B")
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")

    text = window.details_text()
    assert "pore A" in text and "pore B" in text
    assert text.count("τ =") >= 2, "each source's tau must be reported, not just one"
    # Both detection floors are named, and the different-resolution warning is raised.
    assert f"{t_min_a:g}" in text and f"{t_min_b:g}" in text
    assert "DIFFERENT resolutions" in text
    # The legend names each fit too.
    labels = window.legend_labels()
    assert sum("fit" in t for t in labels) == 2


def test_the_single_source_fit_is_unchanged_by_the_overlay_work(window):
    """The one-pore path still produces the exact fit the Fits panel would."""
    a = _source("pore-a", _frame(500, tau=6e-4, t_min=5e-5, seed=7),
                colour="#0072B2", min_dwell_s=5e-5, label="pore A")
    window.set_data([a])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")

    assert window.fit_result() is not None
    assert window.fit_result().params["tau_1"] == pytest.approx(
        _independent_tau(a.df, dict(a.provenance)), rel=1e-9)
    assert window.fit_result().meta["t_min_source"] == "min_dwell_s"


def test_the_fit_curve_follows_the_normalization(window):
    """A count-scaled fit curve laid over a density histogram would sit n times too
    high. The curve must be in the same y quantity as the bars it explains."""
    a = _source("pore-a", _frame(800, tau=6e-4, t_min=5e-5, seed=7),
                colour="#0072B2", min_dwell_s=5e-5, label="pore A")
    b = _source("pore-b", _frame(120, tau=6e-4, t_min=5e-5, seed=8),
                colour="#D55E00", min_dwell_s=5e-5, label="pore B")
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")

    assert window.normalization() == "density"
    for hist, fit in zip(window._hist_items, window._fit_items):
        peak_bars = float(np.max(hist.yData))
        peak_fit = float(np.max(fit.yData))
        assert 0.3 < peak_fit / peak_bars < 3.0, "fit curve is not in the bars' units"


def test_a_source_without_provenance_still_fits_and_says_what_it_guessed(window):
    manual = PlotSource(key="manual", doc_key="manual", label="Manual measurements",
                        df=_frame(200, tau=5e-4, t_min=1e-4, seed=5), kind="manual")
    window.set_data([manual])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")

    assert window.fit_result() is not None
    assert "t_min" in window.details_text()


# ================================================================================================
# 5. cluster split on the histogram — the missing half of the clustering workflow
# ================================================================================================

def _clustered(window):
    a = _source("pore-a", _two_blob_frame(), colour="#0072B2", min_dwell_s=1e-5,
                label="pore A")
    window.set_data([a])
    window.set_feature_selection(["dwell_s", "depth_pA"])
    res = window.run_clustering()
    return a, res


def _two_blob_frame(n: int = 300, seed: int = 4) -> pd.DataFrame:
    """Two well-separated populations in (dwell_s, depth_pA) — something real to find."""
    rng = np.random.default_rng(seed)
    half = n // 2
    df = _frame(n, tau=1e-4, t_min=1e-5, seed=seed)
    df["dwell_s"] = np.r_[np.abs(rng.normal(1e-4, 8e-6, half)),
                          np.abs(rng.normal(9e-4, 6e-5, n - half))]
    df["depth_pA"] = np.r_[np.abs(rng.normal(20.0, 1.5, half)),
                           np.abs(rng.normal(55.0, 2.5, n - half))]
    df["frac_depth"] = df["depth_pA"] / 120.0
    return df


def test_clustering_splits_the_histogram_into_one_curve_per_cluster(window):
    _a, res = _clustered(window)
    assert res is not None and res.chosen_k >= 2

    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")

    # run_clustering turns the split on: the histogram is the other half of the
    # workflow the scatter/PCA already had.
    assert window.histogram_colour_by() == "Cluster"
    assert len(window._hist_items) == res.chosen_k
    colours = [_pen_colour(i) for i in window._hist_items]
    assert len(set(colours)) == len(colours), "each cluster needs its own colour"

    labels = window.legend_labels()
    assert sum("cluster" in t for t in labels) == res.chosen_k


def test_the_cluster_split_can_be_turned_off_again(window):
    _clustered(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_histogram_colour_by("None")
    assert len(window._hist_items) == 1


def test_cluster_histograms_are_parts_of_one_whole(window):
    """Clusters are a decomposition, not independent recordings: their densities must
    add up to the full distribution's, or "one population is rare" — the finding the
    clustering just made — would be normalized away."""
    _clustered(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_normalization("density")

    total = sum(_areas(i) for i in window._hist_items)
    assert total == pytest.approx(1.0, rel=1e-9)


# ================================================================================================
# 6. one owner for the colour channel
# ================================================================================================

def test_multi_source_overrides_the_cluster_split_and_says_so(window):
    """Two features must not fight over the same ink. With an overlay up colour means
    SOURCE, and the cluster control is disabled rather than silently losing."""
    a = _source("pore-a", _two_blob_frame(), colour="#0072B2", min_dwell_s=1e-5,
                label="pore A")
    b = _source("pore-b", _two_blob_frame(seed=9), colour="#D55E00", min_dwell_s=1e-5,
                label="pore B")
    window.set_data([a])
    window.set_feature_selection(["dwell_s", "depth_pA"])
    window.run_clustering()
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    assert window.histogram_colour_by() == "Cluster"

    # Tick a second source: the split turns off, the control goes dead, and the two
    # curves on screen are the two SOURCES.
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    assert window.histogram_colour_by() == "None"
    assert not window._hist_colour_by.isEnabled()
    assert "already means SOURCE" in window._hist_colour_by.toolTip()
    assert len(window._hist_items) == 2
    assert "colour = source" in window.details_text()

    # ...and asking for the split while an overlay is up is refused, not half-honoured.
    window.set_histogram_colour_by("Cluster")
    assert window.histogram_colour_by() == "None"


def test_the_scatter_says_when_cluster_colouring_is_being_overridden(window):
    a = _source("pore-a", _two_blob_frame(), colour="#0072B2", min_dwell_s=1e-5,
                label="pore A")
    b = _source("pore-b", _two_blob_frame(seed=9), colour="#D55E00", min_dwell_s=1e-5,
                label="pore B")
    window.set_data([a])
    window.set_feature_selection(["dwell_s", "depth_pA"])
    window.run_clustering()

    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "depth_pA")
    window.set_colour_by("Cluster")

    assert "colour means SOURCE here" in window.details_text()
    assert len(window._scatter_items) == 2


def test_a_single_source_still_colours_the_scatter_by_cluster_and_names_them(window):
    _a, res = _clustered(window)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "depth_pA")
    window.set_colour_by("Cluster")

    assert len(window._scatter_items) == 1
    colours = {p.brush().color().name() for p in window._scatter_items[0].points()}
    assert len(colours) == res.chosen_k
    assert sum("cluster" in t for t in window.legend_labels()) == res.chosen_k


# ================================================================================================
# 7. the uncalibrated-gain caution
# ================================================================================================

def test_comparing_absolute_pA_across_uncalibrated_recordings_warns(window):
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")

    text = window.details_text()
    assert "CAUTION" in text and "depth_pA" in text
    assert "UNCALIBRATED" in text
    assert "frac_depth" in text, "the warning must say what IS comparable"


def test_dwell_and_frac_depth_are_not_warned_about(window):
    """A time and a ratio survive an uncalibrated gain; warning about them would be
    noise that teaches the user to ignore the warning that matters."""
    _two_pores(window)
    window.set_plot_type("Histogram")
    for col in ("dwell_s", "frac_depth"):
        window.set_histogram_column(col)
        assert "CAUTION" not in window.details_text(), col


def test_this_files_two_pores_share_a_baseline_window_on_purpose(window):
    """The isolation these 42 tests rely on, ASSERTED rather than assumed.

    ``_source`` defaults to one shared ``baseline_fingerprint`` so this file exercises
    the GAIN guard alone. That default is also a blindfold: if the baseline verdict
    ever stopped appearing, nothing in here would notice. So the two halves are
    pinned together — the default is silent, and one changed fingerprint is not."""
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    assert "NOT COMPARABLE" not in window.details_text()

    _two_pores(window, baseline_b="sha256:a different window")
    window.set_histogram_column("frac_depth")
    text = window.details_text()
    assert "NOT COMPARABLE" in text, "the baseline guard must still be reachable here"
    assert _SHARED_BASELINE in text and "sha256:a different window" in text


def test_the_gain_caution_drops_frac_depth_in_this_files_overlay_too(window):
    """The reconciliation, seen from the file that owns the gain guard's tests: with
    the baselines disagreeing, ``depth_pA``'s caution must not send the user to
    ``frac_depth`` — which is what the sentence above (``"frac_depth" in text``)
    asserts it DOES say when they agree."""
    _two_pores(window, baseline_b="sha256:a different window")
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")

    caution = [line for line in window.details_text().splitlines() if "CAUTION" in line]
    assert caution and "frac_depth" not in caution[0]
    assert "dwell_s IS comparable" in caution[0]


def test_no_caution_for_a_single_source(window):
    _two_pores(window)
    window.set_sources(["pore-a"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    assert "CAUTION" not in window.details_text()


def test_a_shared_real_calibration_removes_the_caution(window):
    a = _source("pore-a", _frame(200, tau=3e-4, t_min=5e-5, seed=1), colour="#0072B2",
                min_dwell_s=5e-5, fingerprint=(2.5, 0.0), label="pore A")
    b = _source("pore-b", _frame(200, tau=3e-4, t_min=5e-5, seed=2), colour="#D55E00",
                min_dwell_s=5e-5, fingerprint=(2.5, 0.0), label="pore B")
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    assert "CAUTION" not in window.details_text()


# ================================================================================================
# 8. legends
# ================================================================================================

def test_the_legend_is_readable_on_both_backgrounds(window):
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")

    for mode, expect in (("dark", "#b9bfc7"), ("light", "#1a1a1a")):
        window.set_background(mode)
        legend = window.plot_item().legend
        assert legend is not None and legend.isVisible()
        assert legend.opts["labelTextColor"].name() == expect
        # ...over a backing painted in the CURRENT canvas colour, not last draw's.
        canvas = "#111316" if mode == "dark" else "#ffffff"
        assert legend.opts["brush"].color().name() == canvas
        assert len(window.legend_labels()) == 2


def test_no_empty_legend_box_on_a_plain_single_source_scatter(window):
    _two_pores(window)
    window.set_sources(["pore-a"])
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "frac_depth")
    window.set_colour_by("None")

    legend = window.plot_item().legend
    assert legend is None or not legend.isVisible()


def test_legend_entries_do_not_accumulate_across_redraws(window):
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    assert len(window.legend_labels()) == 2
    for _ in range(4):
        window.set_bins(int(window._bins.value()) + 1)
    assert len(window.legend_labels()) == 2


# ================================================================================================
# 9. columns an overlay can actually draw
# ================================================================================================

def test_only_columns_every_drawn_source_carries_are_offered(window):
    """Offering a column one source lacks would draw a single curve while the legend
    promised two."""
    a = _source("pore-a", _frame(50, tau=3e-4, t_min=5e-5, seed=1), colour="#0072B2",
                min_dwell_s=5e-5, label="pore A")
    extra = _frame(50, tau=3e-4, t_min=5e-5, seed=2)
    extra["custom_metric"] = np.arange(50, dtype=float)
    b = _source("pore-b", extra, colour="#D55E00", min_dwell_s=5e-5, label="pore B")
    window.set_data([a, b])

    window.set_sources(["pore-b"])
    cols = [window._hist_col.itemText(i) for i in range(window._hist_col.count())]
    assert "custom_metric" in cols

    window.set_sources(["pore-a", "pore-b"])
    cols = [window._hist_col.itemText(i) for i in range(window._hist_col.count())]
    assert "custom_metric" not in cols
    assert "dwell_s" in cols


# ================================================================================================
# 10. PCA keeps its cluster colours — and now names them
# ================================================================================================

def test_pca_gets_a_cluster_legend(window):
    _a, res = _clustered(window)
    assert window.run_pca() is not None
    assert sum("cluster" in t for t in window.legend_labels()) == res.chosen_k


def test_pca_without_a_clustering_has_no_legend(window):
    a = _source("pore-a", _two_blob_frame(), colour="#0072B2", min_dwell_s=1e-5,
                label="pore A")
    window.set_data([a])
    window.set_feature_selection(["dwell_s", "depth_pA"])
    assert window.run_pca() is not None
    legend = window.plot_item().legend
    assert legend is None or not legend.isVisible()


# ================================================================================================
# 11. the empty / degenerate states still behave
# ================================================================================================

def test_unticking_everything_says_so_instead_of_crashing(window):
    _two_pores(window)
    window._updating = True
    for i in range(window._source_list.count()):
        window._source_list.item(i).setCheckState(Qt.Unchecked)
    window._updating = False
    window._on_sources_changed()

    assert window.selected_source_keys() == []
    assert window._message_item is not None
    assert "Tick a source" in window.details_text()


def test_a_source_with_nothing_to_draw_is_named_not_silently_dropped(window):
    """One curve short of the legend's promise is a lie; say which source dropped out."""
    a = _source("pore-a", _frame(60, tau=3e-4, t_min=5e-5, seed=1), colour="#0072B2",
                min_dwell_s=5e-5, label="pore A")
    b_df = _frame(60, tau=3e-4, t_min=5e-5, seed=2)
    b_df["dwell_s"] = -b_df["dwell_s"]             # nothing positive to put on a log axis
    b = _source("pore-b", b_df, colour="#D55E00", min_dwell_s=5e-5, label="pore B")
    window.set_data([a, b])
    window.set_sources(["pore-a", "pore-b"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_log_x(True)

    assert len(window._hist_items) == 1
    assert "pore B: no plottable values" in window.details_text()


# ================================================================================================
# 12. the title stays legible
# ================================================================================================

def test_a_multi_source_fit_does_not_run_a_two_tau_title_off_the_plot(window):
    """Two fit summaries joined into one title overflow the plot and get clipped, which
    is worse than no title: the numbers are half-visible rather than absent. With an
    overlay the legend already names every fit, so the title stays clear — and Details
    still carries the full list."""
    _two_pores(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")

    assert window.plot_item().titleLabel.text in ("", None)
    # ...but nothing was lost: both taus are named in the legend and in Details.
    assert sum("fit" in t for t in window.legend_labels()) == 2
    assert window.details_text().count("τ =") >= 2

    # A single source keeps its tau in the title, exactly as before.
    window.set_sources(["pore-a"])
    assert "τ" in window.plot_item().titleLabel.text

    # ...and a user's own title always wins.
    window.set_sources(["pore-a", "pore-b"])
    window.set_title("Two pores, dwell distribution")
    assert "Two pores" in window.plot_item().titleLabel.text
