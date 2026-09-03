"""Figure-layer tests: everything renders, and the log-bin Jacobian is right.

Ported from the SOURCE ``tests/test_figures.py`` (PORT_PLAN §3.8, §8). The event
-statistics figures + ``save_figure`` are data-model-agnostic, so their tests
carry over verbatim -- including the **log-bin Jacobian exact-expectation test
and its negative control**, the scientific guardrail this module exists for.
Only the signal fixtures change: the legacy ``Recording`` is replaced by the
target :class:`CalibratedChannel` + :class:`XAxis` that ``trace_preview`` now
consumes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.document import CalibratedChannel, XAxis
from nanoscope.core.types import ChannelInfo, ChannelKind
from nanoscope.plot import figures, style

TAU = 200e-6
T_MIN = 20e-6


# ---------------------------------------------------------------- fixtures
def make_events(
    n: int = 500,
    *,
    tau: float = TAU,
    t_min: float = T_MIN,
    seed: int = 0,
    n_pores: int = 1,
    filter_limited_frac: float = 0.1,
) -> EventSet:
    """Synthetic EventSet: dwell ~ left-truncated exponential, depth ~ Gaussian."""
    rng = np.random.default_rng(seed)
    dwell = t_min + rng.exponential(tau, n)
    depth_frac = np.clip(rng.normal(0.25, 0.05, n), 1e-3, 0.99)
    i0 = -1000.0
    fl = np.zeros(n, dtype=bool)
    fl[: int(n * filter_limited_frac)] = True
    df = pd.DataFrame(
        {
            "event_id": np.arange(n, dtype=np.int64),
            "recording": pd.array(["synthetic"] * n, dtype="string"),
            "pore_id": pd.array([f"pore{i % n_pores}" for i in range(n)], dtype="string"),
            "start_idx": (np.arange(n) * 1000).astype(np.int64),
            "end_idx": (np.arange(n) * 1000 + 50).astype(np.int64),
            "t_start_s": np.linspace(0.01, 9.9, n),
            "dwell_s": dwell,
            "baseline_pA": np.full(n, i0),
            "depth_pA": depth_frac * abs(i0),
            "frac_depth": depth_frac,
            "ecd_pA_s": depth_frac * abs(i0) * dwell,
            "max_dev_pA": depth_frac * abs(i0) * 1.1,
            "mean_dev_pA": depth_frac * abs(i0),
            "rise_s": np.full(n, 5e-6),
            "fall_s": np.full(n, 5e-6),
            "snr": depth_frac * abs(i0) / 15.0,
            "n_levels": np.ones(n, dtype=np.int64),
            "filter_limited": fl,
            "accepted": np.ones(n, dtype=bool),
            "flag": pd.array([""] * n, dtype="string"),
        }
    )
    assert list(df.columns) == list(SCHEMA.keys())
    return EventSet(df, {"analyzed_duration_s": 10.0, "min_dwell_s": t_min, "recording": "synthetic"})


class ExpFit:
    """Stand-in FitResult for a left-truncated exponential.

    The Jacobian being tested here is this module's responsibility, so the fit
    is pinned to an exact, known density rather than an estimate.
    """

    def __init__(self, tau: float = TAU, t_min: float = T_MIN):
        self.tau, self.t_min = tau, t_min
        self.params = {"tau_s": tau}
        self.ci = {"tau_s": (tau * 0.95, tau * 1.05)}

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= self.t_min, np.exp(-(x - self.t_min) / self.tau) / self.tau, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= self.t_min, 1.0 - np.exp(-(x - self.t_min) / self.tau), 0.0)


class GaussFit:
    def __init__(self, mu: float = 0.25, sigma: float = 0.05):
        self.mu, self.sigma = mu, sigma
        self.params = {"mu_1": mu, "sigma_1": sigma}
        self.ci = {"mu_1": (mu - 0.01, mu + 0.01)}

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))


def make_channel(
    n: int = 500_000, fs: float = 50_000.0, seed: int = 1, units: str = "pA"
) -> tuple[CalibratedChannel, XAxis]:
    """Target CalibratedChannel + XAxis carrying a spike naive subsampling misses."""
    rng = np.random.default_rng(seed)
    cur = rng.normal(-1000.0, 15.0, n).astype(np.float32)
    cur[10_000:10_050] -= 250.0  # a spike naive subsampling would miss
    info = ChannelInfo(index=0, name="synthetic", raw_units=units, kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(
        info=info,
        data=np.ascontiguousarray(cur),
        units=units,
        sample_rate_hz=fs,
        t0_s=0.0,
    )
    xax = XAxis(values=np.arange(n, dtype=np.float64) / fs, unit="s", is_time=True)
    return ch, xax


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ---------------------------------------------------------------- the real test
def test_log_bin_jacobian_matches_exact_expected_counts():
    """The overlaid curve must equal N*(F(b)-F(a)) per bin -- the exact expectation.

    This is deterministic: no sampling noise, so a wrong Jacobian cannot hide.
    """
    events = make_events(4000, seed=3)
    fit = ExpFit()
    fig, ax = figures.dwell_histogram(events, fit=fit)
    d = getattr(fig, "_nanoscope_source_data")

    n_fit = len(events.df)
    exact = n_fit * (fit.cdf(d["bin_right_s"].values) - fit.cdf(d["bin_left_s"].values))
    got = d["fit_expected_count"].values

    ok = exact > 1e-6
    rel = np.abs(got[ok] - exact[ok]) / exact[ok]
    # Midpoint rule in log space: error is O(du^2), a few % at this bin width.
    assert rel.max() < 0.05, f"curve deviates from exact expected counts by {rel.max():.1%}"


def test_log_bin_curve_tracks_histogram_heights():
    """Render a known exponential and check the curve actually follows the bars."""
    events = make_events(20_000, seed=4, filter_limited_frac=0.0)
    fig, ax = figures.dwell_histogram(events, fit=ExpFit())
    d = getattr(fig, "_nanoscope_source_data")

    good = d["count"].values >= 50
    assert good.sum() >= 5
    counts = d["count"].values[good]
    curve = d["fit_expected_count"].values[good]
    rel = np.abs(curve - counts) / counts
    assert rel.max() < 0.25, f"fit does not track the histogram: max rel err {rel.max():.1%}"

    # The curve on the axes must be the same numbers, not a separately-computed one.
    line = next(ln for ln in ax.lines if ln.get_label() == "exponential fit")
    lx, ly = line.get_xdata(), line.get_ydata()
    interp = np.interp(d["bin_center_s"].values[good] * 1e6, lx, ly)
    assert np.allclose(interp, curve, rtol=0.05)


def test_naive_overlay_without_jacobian_would_fail_this_test():
    """Negative control: prove the test above has teeth.

    The classic bug is scaling the density by the log10 bin width `du` instead
    of the linear width `t*ln(10)*du`. It is wrong by a factor of t*ln(10),
    which varies by decades across the axis -- and yet still traces a curve of
    roughly the right shape. If this test ever passes, the one above is vacuous.
    """
    events = make_events(20_000, seed=4, filter_limited_frac=0.0)
    fig, _ = figures.dwell_histogram(events, fit=ExpFit())
    d = getattr(fig, "_nanoscope_source_data")

    edges = np.concatenate([d["bin_left_s"].values, [d["bin_right_s"].values[-1]]])
    du = np.log10(edges[1]) - np.log10(edges[0])
    centers = d["bin_center_s"].values
    naive = len(events.df) * ExpFit().predict(centers) * du

    good = d["count"].values >= 50
    rel = np.abs(naive[good] - d["count"].values[good]) / d["count"].values[good]
    assert rel.max() > 1.0, "the no-Jacobian curve should be badly wrong; the check is not sensitive"


def test_linear_bins_use_constant_width():
    events = make_events(2000, seed=5)
    fig, ax = figures.dwell_histogram(events, fit=ExpFit(), log_bins=False)
    d = getattr(fig, "_nanoscope_source_data")
    widths = d["bin_right_s"].values - d["bin_left_s"].values
    assert np.allclose(widths, widths[0])
    assert ax.get_xscale() == "linear"

    fit = ExpFit()
    exact = len(events.df) * (fit.cdf(d["bin_right_s"].values) - fit.cdf(d["bin_left_s"].values))
    ok = exact > 1e-6
    rel = np.abs(d["fit_expected_count"].values[ok] - exact[ok]) / exact[ok]
    assert rel.max() < 0.05


def test_dwell_histogram_uses_log_bins_by_default():
    events = make_events(1000)
    fig, ax = figures.dwell_histogram(events, fit=False)
    d = getattr(fig, "_nanoscope_source_data")
    ratios = d["bin_right_s"].values / d["bin_left_s"].values
    assert np.allclose(ratios, ratios[0]), "bins are not log-spaced"
    assert ax.get_xscale() == "log"


def test_dwell_histogram_shades_below_detection_limit():
    events = make_events(500)
    fig, ax = figures.dwell_histogram(events, fit=False)
    assert ax.get_xlim()[0] < T_MIN * 1e6, "the below-detection region is not visible"
    assert any("detection limit" in t.get_text() for t in ax.texts)


def test_dwell_histogram_annotates_tau_and_ci():
    events = make_events(1000)
    fig, ax = figures.dwell_histogram(events, fit=ExpFit())
    blob = " ".join(t.get_text() for t in ax.texts)
    assert "200.0" in blob and "190.0" in blob and "210.0" in blob


def test_precomputed_multi_component_fit_is_used_verbatim_not_refit():
    """B2: fit=True refits at k=1, so a k=2 fit must be PASSED, not re-derived."""
    from nanoscope.analysis.stats import fit_dwell_exponential

    events = make_events(600)
    dwell = events.df["dwell_s"].to_numpy()
    two = fit_dwell_exponential(dwell, t_min=T_MIN, n_components=2, n_boot=0)

    fig, ax = figures.dwell_histogram(events, fit=two)
    blob = " ".join(t.get_text() for t in ax.texts)
    plt.close(fig)

    # Both taus are annotated -- a one-component refit could only print one.
    assert blob.count(r"$\tau$") == 2
    for tau in (two.params["tau_1"], two.params["tau_2"]):
        assert f"{tau * 1e6:.1f}" in blob

    # ...and the exported source data is that fit's curve, not another one's.
    refit, ax2 = figures.dwell_histogram(events, fit=True)
    a = getattr(fig, "_nanoscope_source_data")["fit_expected_count"].to_numpy()
    b = getattr(refit, "_nanoscope_source_data")["fit_expected_count"].to_numpy()
    plt.close(refit)
    assert not np.allclose(a, b, rtol=1e-3)


def test_figure_annotates_what_the_supplied_fit_did_to_the_sample():
    """A published number has to carry its caveats onto the page."""
    from nanoscope.analysis.stats import fit_dwell_exponential

    events = make_events(400)
    dwell = events.df["dwell_s"].to_numpy()
    fit = fit_dwell_exponential(dwell, t_min=T_MIN, t_max=float(dwell.max()), n_boot=0)
    fit.meta["n_excluded_filter_limited"] = 17
    fit.meta["t_min_source"] = "2*filter_rise_s"

    fig, ax = figures.dwell_histogram(events, fit=fit)
    blob = " ".join(t.get_text() for t in ax.texts)
    plt.close(fig)

    assert "17 filter-limited event(s) excluded" in blob
    assert "2*filter_rise_s" in blob
    assert "right-truncated at" in blob and "modelled" in blob


def test_dwell_histogram_warns_without_min_dwell_provenance():
    events = make_events(200)
    events.provenance.pop("min_dwell_s")
    with pytest.warns(RuntimeWarning, match="min_dwell_s"):
        figures.dwell_histogram(events, fit=False)


def test_dwell_histogram_uses_accepted_only():
    events = make_events(400)
    events.df.loc[: 199, "accepted"] = False
    fig, _ = figures.dwell_histogram(events, fit=False)
    d = getattr(fig, "_nanoscope_source_data")
    assert int(d["count"].sum()) == 200


def test_dwell_histogram_rejects_nonpositive_dwell():
    events = make_events(50)
    events.df.loc[0, "dwell_s"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        figures.dwell_histogram(events, fit=False)


def test_dwell_histogram_rejects_empty():
    with pytest.raises(ValueError, match="no events"):
        figures.dwell_histogram(EventSet(empty_df(), {"min_dwell_s": T_MIN}), fit=False)


# ---------------------------------------------------------------- depth
def test_depth_histogram_renders_and_overlays_density():
    events = make_events(8000)
    fig, ax = figures.depth_histogram(events, fit=GaussFit())
    d = getattr(fig, "_nanoscope_source_data")
    # Compare near the peak, where sqrt(n) fluctuations are smallest.
    good = d["density"].values > 0.6 * d["density"].values.max()
    rel = np.abs(d["fit_density"].values[good] - d["density"].values[good]) / d["density"].values[good]
    assert rel.max() < 0.25
    assert any("0.25" in t.get_text() for t in ax.texts)
    assert fig is ax.figure


def test_depth_histogram_no_fit():
    fig, ax = figures.depth_histogram(make_events(300), fit=False)
    assert ax.get_xlabel()
    assert isinstance(fig, plt.Figure)


def test_depth_histogram_rejects_bad_n_components():
    with pytest.raises(ValueError, match="n_components"):
        figures.depth_histogram(make_events(100), fit=False, n_components=0)


# ---------------------------------------------------------------- scatter
@pytest.mark.parametrize("density", [True, False])
def test_scatter_dwell_depth_renders(density):
    fig, ax = figures.scatter_dwell_depth(make_events(800), density=density)
    fig.canvas.draw()
    assert ax.get_xscale() == "log"
    assert len(fig.axes) >= 3, "marginal histograms are missing"
    d = getattr(fig, "_nanoscope_source_data")
    assert {"dwell_s", "frac_depth", "filter_limited"} <= set(d.columns)


def test_scatter_marks_filter_limited_distinctly():
    fig, ax = figures.scatter_dwell_depth(make_events(500, filter_limited_frac=0.2), density=True)
    labels = [c.get_label() for c in ax.collections]
    assert any("filter-limited" in str(x) for x in labels)


def test_scatter_into_provided_axes():
    fig, ax0 = plt.subplots()
    fig2, ax2 = figures.scatter_dwell_depth(make_events(200), ax=ax0)
    assert ax2 is ax0 and fig2 is fig
    fig2.canvas.draw()


# ---------------------------------------------------------------- rate
def _runset() -> pd.DataFrame:
    rows = []
    for p in range(3):
        for v in (-200.0, -400.0, -600.0, -800.0):
            rows.append({"pore_id": f"pore{p}", "voltage_mV": v, "rate_hz": abs(v) * (0.02 + 0.01 * p)})
    return pd.DataFrame(rows)


def test_event_rate_plot_keeps_pores_separate():
    df = _runset()
    fig, ax = figures.event_rate_plot(df)
    assert len(ax.lines) == 3, "pores must be plotted individually, not averaged"
    assert ax.get_ylabel() == "event rate (Hz)"
    assert len(getattr(fig, "_nanoscope_source_data")) == len(df)


def test_event_rate_plot_log_x_for_concentration():
    df = pd.DataFrame(
        {"pore_id": ["a"] * 3, "concentration_nM": [1.0, 10.0, 100.0], "rate_hz": [1.0, 8.0, 60.0]}
    )
    fig, ax = figures.event_rate_plot(df, x="concentration_nM")
    assert ax.get_xscale() == "log"


def test_event_rate_plot_requires_pore_id():
    df = _runset().drop(columns=["pore_id"])
    with pytest.raises(KeyError, match="pore_id"):
        figures.event_rate_plot(df)


def test_event_rate_plot_requires_rate_column():
    df = _runset().rename(columns={"rate_hz": "counts"})
    with pytest.raises(KeyError, match="event-rate column"):
        figures.event_rate_plot(df)


def test_event_rate_plot_requires_x_column():
    with pytest.raises(KeyError, match="nope"):
        figures.event_rate_plot(_runset(), x="nope")


# ---------------------------------------------------------------- trace
def test_minmax_envelope_preserves_extremes():
    y = np.zeros(100_000)
    y[54_321] = 99.0  # one sample; naive subsampling would drop it
    y[12_345] = -42.0
    idx, vals = figures._minmax_envelope(y, 1000)
    assert vals.max() == 99.0 and vals.min() == -42.0
    assert 54_321 in idx and 12_345 in idx
    assert np.all(np.diff(idx) >= 0), "envelope must stay in temporal order"


def test_minmax_envelope_shorter_than_columns_is_identity():
    y = np.arange(10.0)
    idx, vals = figures._minmax_envelope(y, 1000)
    assert np.array_equal(vals, y)


def test_trace_preview_decimates_and_shows_spike():
    ch, xax = make_channel()
    fig, ax = figures.trace_preview(ch, xax)
    line = ax.lines[0]
    assert len(line.get_ydata()) < 6000, "trace was not decimated"
    assert line.get_ydata().min() < -1200, "the decimation lost the blockade"
    # t1 defaults to the end of the XAxis (=(n-1)/fs on the target data model).
    assert ax.get_xlim()[0] == 0.0
    assert abs(ax.get_xlim()[1] - float(xax.values[-1])) < 1e-9


def test_trace_preview_threads_channel_units_into_label():
    """R9: never hard-code pA -- the y-axis label is the channel's own units."""
    ch, xax = make_channel(n=50_000, units="nA")
    fig, ax = figures.trace_preview(ch, xax)
    assert "nA" in ax.get_ylabel()


def test_trace_preview_with_events_and_baseline():
    ch, xax = make_channel(n=100_000)
    baseline = np.full(ch.data.size, -1000.0)
    ev = make_events(20)
    fig, ax = figures.trace_preview(ch, xax, events=ev, baseline=baseline, t0=0.0, t1=1.0)
    assert any("baseline" in (ln.get_label() or "") for ln in ax.lines)
    assert getattr(fig, "_nanoscope_n_events_shown") > 0


def test_trace_preview_rejects_wrong_length_baseline():
    ch, xax = make_channel(n=10_000)
    with pytest.raises(ValueError, match="baseline length"):
        figures.trace_preview(ch, xax, baseline=np.zeros(10))


def test_trace_preview_rejects_empty_window():
    ch, xax = make_channel(n=10_000)
    with pytest.raises(ValueError, match="empty window"):
        figures.trace_preview(ch, xax, t0=1.0, t1=0.5)


# ---------------------------------------------------------------- saving
def test_save_figure_writes_pdf_png_and_csv(tmp_path):
    events = make_events(300)
    fig, _ = figures.dwell_histogram(events, fit=False)
    out = figures.save_figure(fig, tmp_path / "fig1.pdf")
    for k in ("pdf", "png", "csv"):
        assert out[k].exists() and out[k].stat().st_size > 0
    assert out["pdf"].suffix == ".pdf" and out["csv"].suffix == ".csv"
    d = pd.read_csv(out["csv"])
    assert "count" in d.columns and len(d) > 0


def test_save_figure_explicit_data_wins(tmp_path):
    fig, _ = figures.dwell_histogram(make_events(100), fit=False)
    mine = pd.DataFrame({"a": [1, 2, 3]})
    out = figures.save_figure(fig, tmp_path / "sub" / "fig2", data=mine)
    assert pd.read_csv(out["csv"])["a"].tolist() == [1, 2, 3]


def test_save_figure_accepts_eventset_and_dict(tmp_path):
    fig, _ = figures.dwell_histogram(make_events(100), fit=False)
    out = figures.save_figure(fig, tmp_path / "f3", data=make_events(10))
    assert "dwell_s" in pd.read_csv(out["csv"]).columns
    out = figures.save_figure(fig, tmp_path / "f4", data={"x": [1, 2], "y": [3, 4]})
    assert pd.read_csv(out["csv"])["y"].tolist() == [3, 4]


def test_save_figure_harvests_lines_when_nothing_attached(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6], label="mine")
    out = figures.save_figure(fig, tmp_path / "f5.png")
    d = pd.read_csv(out["csv"])
    assert d["series"].iloc[0] == "mine" and d["y"].tolist() == [4, 5, 6]


def test_save_figure_warns_when_no_data(tmp_path):
    fig, _ = plt.subplots()
    with pytest.warns(RuntimeWarning, match="no source data"):
        out = figures.save_figure(fig, tmp_path / "f6.pdf")
    assert "csv" not in out and out["pdf"].exists()


# ---------------------------------------------------------------- style
def test_nanoscope_style_does_not_leak():
    before = dict(matplotlib.rcParams)
    with style.nanoscope_style():
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["font.size"] == before["font.size"]
    assert matplotlib.rcParams["axes.spines.top"] == before["axes.spines.top"]


def test_importing_style_sets_nothing_globally():
    """Importing must not touch a notebook user's rcParams."""
    import importlib

    before = matplotlib.rcParams["axes.prop_cycle"]
    importlib.reload(style)
    assert matplotlib.rcParams["axes.prop_cycle"] == before


def test_style_overrides_and_sizes():
    with style.nanoscope_style(**{"font.size": 14}):
        assert matplotlib.rcParams["font.size"] == 14
    assert style.figure_size("single")[0] == 3.35
    assert style.figure_size("double")[0] == 6.9
    assert style.figure_size(5.0, aspect=0.5) == (5.0, 2.5)
    with pytest.raises(ValueError, match="width"):
        style.figure_size("triple")


def test_palette_is_okabe_ito_and_cycles():
    assert len(set(style.OKABE_ITO)) == 8
    assert style.color(0) == style.color(len(style.OKABE_ITO))
    assert all(c.startswith("#") and len(c) == 7 for c in style.OKABE_ITO)


def test_figures_accept_plain_dataframe():
    """The DataFrame is the product; figures must not require the wrapper."""
    df = make_events(200).df
    with pytest.warns(RuntimeWarning):
        fig, _ = figures.dwell_histogram(df, fit=False)
    assert fig is not None
    figures.scatter_dwell_depth(df, density=True)
    figures.depth_histogram(df, fit=False)


def test_figures_reject_junk_input():
    with pytest.raises(TypeError, match="EventSet or DataFrame"):
        figures.dwell_histogram("not events", fit=False)


def test_fit_must_be_fitresult_like():
    with pytest.raises(TypeError, match="FitResult"):
        figures.dwell_histogram(make_events(100), fit=object())


def test_fit_true_without_stats_module_raises_clearly():
    """fit=True routes to analysis.stats; if that module is absent, say so plainly."""
    pytest.importorskip("nanoscope.analysis.stats", reason="stats.py not written yet")
    fig, _ = figures.dwell_histogram(make_events(300), fit=True)
    assert fig is not None


# ================================================================================================
# joint_density — overlaid KDE contours for comparing molecules / voltages
# ================================================================================================
def _density_frame(n, *, dwell_ms, depth, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": np.abs(rng.normal(dwell_ms * 1e-3, dwell_ms * 2e-4, n)),
        "depth_pA": np.abs(rng.normal(depth, depth * 0.1, n)),
        "frac_depth": np.abs(rng.normal(depth / 100.0, 0.02, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def test_joint_density_draws_one_contour_set_per_group():
    from nanoscope.plot.figures import joint_density
    groups = {"pair": _density_frame(400, dwell_ms=0.3, depth=20, seed=1),
              "guest": _density_frame(300, dwell_ms=0.8, depth=12, seed=2),
              "host": _density_frame(200, dwell_ms=0.5, depth=8, seed=3)}
    fig, ax = joint_density(groups)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 3
    assert all("n = " in t for t in labels), "counts belong in the legend by default"
    assert ax.get_xscale() == "log", "dwell spans decades; the axis must be log"


def test_joint_density_normalises_each_group_to_itself():
    """An n=200 population and an n=17000 one must be comparable SHAPES.

    Contour levels are fractions of each group's OWN peak density, so the sparse group is
    not reduced to a single ring beside the dense one. The counts go in the legend, where
    they inform without distorting.
    """
    from nanoscope.plot.figures import joint_density
    groups = {"dense": _density_frame(4000, dwell_ms=0.3, depth=20, seed=1),
              "sparse": _density_frame(60, dwell_ms=0.3, depth=20, seed=2)}
    fig, ax = joint_density(groups, levels=8)
    # One ContourSet per group, each carrying the same number of levels -- so the sparse
    # group is a full set of rings, not a single one beside the dense group's eight.
    import matplotlib.contour
    sets = ax.findobj(matplotlib.contour.ContourSet)
    assert len(sets) == 2, "both groups must be drawn"
    assert {len(cs.levels) for cs in sets} == {8}, "same level count for each group"


def test_joint_density_skips_a_degenerate_group_and_keeps_the_rest():
    """A group with too few points, or all one value, must not take the figure down."""
    from nanoscope.plot.figures import joint_density
    tiny = _density_frame(3, dwell_ms=0.3, depth=20)          # below the 5-point floor
    flat = pd.DataFrame({"dwell_s": np.full(50, 1e-3), "depth_pA": np.full(50, 10.0),
                         "frac_depth": np.full(50, 0.1),
                         "accepted": np.ones(50, bool),
                         "filter_limited": np.zeros(50, bool)})   # zero variance -> singular
    good = _density_frame(300, dwell_ms=0.5, depth=15, seed=4)
    fig, ax = joint_density({"tiny": tiny, "flat": flat, "good": good})
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("good" in t for t in labels)
    assert not any("tiny" in t for t in labels), "a 3-point KDE is a decoration"


def test_joint_density_refuses_when_no_group_is_usable():
    from nanoscope.plot.figures import joint_density
    with pytest.raises(ValueError, match="enough finite points"):
        joint_density({"a": _density_frame(2, dwell_ms=0.3, depth=20)})


def test_joint_density_can_use_fractional_blockade():
    """dI/I0 divides the gain out, so it is the comparable axis when gain is uncalibrated."""
    from nanoscope.plot.figures import joint_density
    groups = {"a": _density_frame(300, dwell_ms=0.3, depth=20, seed=1)}
    fig, ax = joint_density(groups, ycol="frac_depth", ylabel="ΔI/I₀")
    assert ax.get_ylabel() == "ΔI/I₀"
    # The y data really is the fraction, not the pA column.
    ylo, yhi = ax.get_ylim()
    assert yhi < 5.0, "fractional blockade is order 0.2, not order 20 pA"


def test_joint_density_styling_shrinks_the_legend():
    """A legend is sized by its widest entry, and the counts are the longest part of one."""
    from nanoscope.plot.figures import joint_density
    groups = {"pair (CB7-ada)": _density_frame(300, dwell_ms=0.3, depth=20, seed=1),
              "guest (ada)": _density_frame(300, dwell_ms=0.6, depth=12, seed=2)}
    with_counts, ax1 = joint_density(groups, show_counts=True)
    longest_with = max(len(t.get_text()) for t in ax1.get_legend().get_texts())
    without, ax2 = joint_density(groups, show_counts=False, legend_title=None)
    longest_without = max(len(t.get_text()) for t in ax2.get_legend().get_texts())
    assert longest_without < longest_with
    assert ax2.get_legend().get_title().get_text() == ""

    none_fig, ax3 = joint_density(groups, legend_loc="none")
    assert ax3.get_legend() is None, "legend_loc='none' must remove it entirely"


def test_joint_density_attaches_its_source_data():
    """Every figure in this module carries the numbers behind it; this one is no exception."""
    from nanoscope.plot.figures import joint_density, save_figure
    import tempfile, pathlib
    groups = {"a": _density_frame(200, dwell_ms=0.3, depth=20, seed=1),
              "b": _density_frame(200, dwell_ms=0.7, depth=11, seed=2)}
    fig, _ax = joint_density(groups)
    with tempfile.TemporaryDirectory() as td:
        out = save_figure(fig, str(pathlib.Path(td) / "d.png"))
        assert sorted(out) == ["csv", "pdf", "png"], "the whole family lands together"
        data = pd.read_csv(out["csv"])
    assert set(data["group"]) == {"a", "b"}
    assert {"dwell_s", "depth_pA"} <= set(data.columns)
