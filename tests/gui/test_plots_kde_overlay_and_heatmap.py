"""Distributions: a 1-D KDE overlay on histograms, and an Event-count heatmap plot type.

Plot Studio step 4 (Distributions items). Both are ANALYSIS settings of the Plots window,
so the assertions here are about the numbers first and the plumbing second:

* the KDE curve sits on the bars it smooths in every y quantity -- count, density,
  probability per bin -- on a linear AND a log x axis (on log x it is estimated on log10
  of the column, uniform in the log bins), per source and per cluster;
* a hand tick of the overlay waits for Plot / Apply -- also past a live bin, log or
  normalization redraw; the setter commits at once;
* the setting rides undo, templates, the JSON sidecar (version 3), the project's
  workspace snapshot (and project open does NOT estimate it) and the Figure Library's
  applied config, which names the overlay only once it is drawn; Export plot data
  carries the curves as ``kde`` rows;
* the heatmap's counts are Event waveforms' kernel, bin for bin, for the first ticked
  source only; its Details carries the scatter's comparability verdict; its colour bar
  is inside the PlotItem (so it exports), leaves with it and has no colormap menu of its
  own; a colour-map change re-bins nothing and keeps the zoom; an imported map this
  build lacks is replaced and disclosed; ``cell`` rows export it.

Synthetic frames only; offscreen.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt
from scipy.stats import gaussian_kde

import nanoscope.gui.plots_panel as pp
from nanoscope.analysis import event_shape
from nanoscope.core import analysis_workspace as workspace
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import (
    HEATMAP,
    HeatmapConfig,
    PlotConfig,
    PlotsWindow,
    plot_config_from_json,
    plot_config_to_json,
)


# ================================================================================================
# builders
# ================================================================================================

def _events(n: int = 4000, *, seed: int = 1, centre: float = -3.0) -> pd.DataFrame:
    """Log-normal dwells over two decades and a depth that steps up for the long events,
    so a heatmap has structure and a log-x histogram is a real log axis."""
    rng = np.random.default_rng(seed)
    dwell = 10.0 ** rng.normal(centre, 0.3, n)
    depth = np.abs(rng.normal(80.0, 10.0, n)) + 30.0 * (dwell > 10.0 ** centre)
    return pd.DataFrame({
        "dwell_s": dwell,
        "depth_pA": depth,
        "frac_depth": depth / 300.0,
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "accepted": np.ones(n, dtype=bool),
    })


def _source(key: str = "doc0:0", label: str = "pore A", *, seed: int = 1,
            n: int = 4000, centre: float = -3.0) -> PlotSource:
    return PlotSource(key=key, label=label, df=_events(n, seed=seed, centre=centre),
                      provenance={"fs": 250_000.0, "min_dwell_s": 1e-6},
                      colour="#0072B2", doc_id=key.split(":")[0], channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source()])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    return w


@pytest.fixture
def two(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source(), _source("doc1:0", "pore B", seed=2, n=900, centre=-2.6)])
    w.set_sources(["pore A", "pore B"])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    return w


def _kde_items(w) -> list:
    return [i for i in w.plot_item().items if getattr(i, "_plot_data_part", None) == "kde"]


def _bars(w) -> list:
    return list(w._hist_items)


def _curve_at_bar_centres(item, edges: np.ndarray, log_x: bool) -> np.ndarray:
    """The KDE curve read at each bar's centre (geometric centre on log x)."""
    x, y = np.asarray(item.xData, float), np.asarray(item.yData, float)
    if log_x:
        centres = np.sqrt(edges[:-1] * edges[1:])
        return np.interp(np.log10(centres), np.log10(x), y, left=np.nan, right=np.nan)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(centres, x, y, left=np.nan, right=np.nan)


# ================================================================================================
# D1 -- the KDE overlay's numbers
# ================================================================================================

@pytest.mark.parametrize("log_x", [False, True])
@pytest.mark.parametrize("norm", ["count", "density", "probability per bin"])
def test_the_kde_sits_on_the_bars_it_smooths_in_every_y_quantity(win, norm, log_x):
    """THE SCALING, CHECKED AGAINST THE BARS rather than against its own formula.

    A KDE scaled to the wrong quantity is off by n, by a bin width or by x·ln10 -- factors
    of 4000, ~1e-4 or ~2 here -- so agreeing with the bars to within a few percent where
    they are well populated can only happen with the right scaling. On log x this also
    pins that the curve was estimated in log10 space: a linear-space kernel's bandwidth
    would be set by the long tail and smear the short events flat.
    """
    win.set_log_x(log_x)
    win.set_normalization(norm)
    win.set_kde_overlay(True)
    (bar,) = _bars(win)
    (kde,) = _kde_items(win)
    edges = np.asarray(bar.xData, float)
    heights = np.asarray(bar.yData, float)
    curve = _curve_at_bar_centres(kde, edges, log_x)
    full = heights >= 0.5 * heights.max()           # the well-populated bins
    ratio = curve[full] / heights[full]
    assert np.nanmedian(ratio) == pytest.approx(1.0, abs=0.06), (norm, log_x, ratio)
    assert np.all(np.isfinite(ratio))


def test_a_density_kde_integrates_to_one_in_the_stored_unit(win):
    """Density means per the column's STORED unit (per second for a dwell), as the bars
    are -- whatever the ms ticks say. Over the observed range the area is just under 1
    (a Gaussian kernel leaks a little past the extremes), on a linear and a log axis."""
    win.set_normalization("density")
    win.set_kde_overlay(True)
    for log_x in (False, True):
        win.set_log_x(log_x)
        (kde,) = _kde_items(win)
        x, y = np.asarray(kde.xData, float), np.asarray(kde.yData, float)
        area = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
        assert 0.95 < area <= 1.0, (log_x, area)


def test_each_source_gets_its_own_kde_and_details_names_its_bandwidth(two):
    """One curve per drawn source, each normalised to itself as its bars are, and the
    bandwidth said per source in the axis's own time unit (ms by default) or, on log x,
    in decades -- the number a reader needs to judge the smoothing."""
    two.set_normalization("density")
    two.set_kde_overlay(True)
    items = _kde_items(two)
    assert len(items) == 2
    for item in items:
        x, y = np.asarray(item.xData, float), np.asarray(item.yData, float)
        assert 0.95 < float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) <= 1.0
    details = two.details_text()
    for label, seed, n, centre in (("pore A", 1, 4000, -3.0), ("pore B", 2, 900, -2.6)):
        dwell = _events(n, seed=seed, centre=centre)["dwell_s"].to_numpy()
        h_ms = float(np.sqrt(gaussian_kde(dwell, bw_method="scott").covariance[0, 0])) * 1e3
        line = [ln for ln in details.splitlines() if ln.strip().startswith(f"{label}: n = {n}, bandwidth")]
        assert line, details
        assert f"{h_ms:.4g} ms" in line[0], (line[0], h_ms)
    two.set_log_x(True)
    assert "decades of dwell_s" in two.details_text()
    assert "KDE (Gaussian, Scott)" in two.legend_labels()


def test_cluster_kdes_add_up_to_the_whole_source(win):
    """With the histogram split by cluster, every cluster normalises against the whole
    source (its bars do), so the clusters' density curves share the source's area."""
    win.set_feature_selection(["dwell_s", "depth_pA"])
    win.set_cluster_method("GMM")
    win.set_cluster_k(2)
    assert win.run_clustering() is not None
    win.set_plot_type("Histogram")
    win.set_histogram_colour_by("Cluster")
    win.set_normalization("density")
    win.set_kde_overlay(True)
    items = _kde_items(win)
    assert len(items) >= 2
    total = 0.0
    for item in items:
        x, y = np.asarray(item.xData, float), np.asarray(item.yData, float)
        total += float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
    assert 0.9 < total <= 1.0, total


def test_too_few_values_skip_the_curve_and_say_so(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source(n=2)])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_kde_overlay(True)
    assert _kde_items(w) == []
    assert "KDE overlay skipped" in w.details_text()


# ================================================================================================
# D1 -- the gate, and every place the setting has to live
# ================================================================================================

def test_a_hand_tick_waits_for_plot_and_the_setter_commits_at_once(win, monkeypatch):
    """An estimate must not appear on a click (owner note #12's gate): the tick badges
    Plot ● and draws nothing until Plot / Apply. The programmatic setter IS the run."""
    assert win.config().histogram.kde_overlay is False
    monkeypatch.setattr(win, "_from_the_user", lambda *_a: True)
    win._hist_kde.setChecked(True)
    assert _kde_items(win) == []
    assert win._stale and "KDE overlay" in win._stale_reason
    assert "●" in win._plot_btn.text()
    applied = win.figure_applied_config()
    assert applied["histogram"]["kde_overlay"] is False, "the picture has no KDE yet"
    win._run_plot()
    assert len(_kde_items(win)) == 1 and not win._stale
    assert win.figure_applied_config()["histogram"]["kde_overlay"] is True

    monkeypatch.setattr(win, "_from_the_user", lambda *_a: False)
    win.set_kde_overlay(False)
    assert _kde_items(win) == [] and not win._stale


def test_a_live_redraw_after_a_hand_tick_does_not_draw_the_pending_kde(win, monkeypatch):
    """The gate holds past the NEXT redraw, as the fit's does: a bin count, a log toggle
    and a y-quantity switch all redraw live, and none of them is the Plot press the
    tick is waiting for. Before, the draw read the box, so the curve appeared under a
    banner still saying the plot was the PREVIOUS selection, and the Figure Library's
    applied config claimed a KDE whichever way the picture went."""
    monkeypatch.setattr(win, "_from_the_user", lambda *_a: True)
    win._hist_kde.setChecked(True)
    monkeypatch.setattr(win, "_from_the_user", lambda *_a: False)
    win._bins.setValue(win._bins.value() + 5)
    win._hist_logx.setChecked(True)
    win._norm_combo.setCurrentText("density")
    assert _kde_items(win) == []
    assert win._stale and "KDE overlay" in win._stale_reason
    assert win.figure_applied_config()["histogram"]["kde_overlay"] is False
    assert win.config().histogram.kde_overlay is True       # the proposal stands
    win._run_plot()
    assert len(_kde_items(win)) == 1 and not win._stale
    assert win.figure_applied_config()["histogram"]["kde_overlay"] is True


def test_undo_takes_the_overlay_back(win):
    win.set_kde_overlay(True)
    assert len(_kde_items(win)) == 1
    assert win.undo_config()
    assert win.config().histogram.kde_overlay is False
    assert _kde_items(win) == []
    assert win.redo_config()
    assert win.config().histogram.kde_overlay is True
    assert len(_kde_items(win)) == 1


def test_a_template_carries_the_overlay_and_the_heatmap_recipe(win):
    """Both are recipes, not references to rows, so a template keeps them; a blank
    heatmap column means "keep what the window has", as for every other page."""
    win.set_kde_overlay(True)
    cfg = replace(win.config(), heatmap=HeatmapConfig(x="", y="", log_x=False, bins=25))
    tpl = cfg.as_template()
    assert tpl.histogram.kde_overlay is True
    landed = PlotConfig().with_template(tpl)
    assert landed.histogram.kde_overlay is True
    assert landed.heatmap.bins == 25 and landed.heatmap.log_x is False
    mine = replace(PlotConfig(), heatmap=HeatmapConfig(x="dwell_s", y="depth_pA"))
    assert mine.with_template(tpl).heatmap.x == "dwell_s"


def test_the_sidecar_is_version_3_and_round_trips_both_settings():
    cfg = replace(PlotConfig(plot_type=HEATMAP),
                  histogram=replace(PlotConfig().histogram, kde_overlay=True),
                  heatmap=HeatmapConfig(x="dwell_s", y="depth_pA", log_x=False,
                                        log_y=True, bins=17),
                  style=replace(PlotConfig().style, heatmap_colormap="magma"))
    payload = plot_config_to_json(cfg)
    assert payload["version"] == pp.PLOT_CONFIG_VERSION == 4
    assert payload["histogram"]["kde_overlay"] is True
    assert payload["heatmap"] == {"x": "dwell_s", "y": "depth_pA", "log_x": False,
                                  "log_y": True, "bins": 17}
    assert plot_config_from_json(payload) == cfg

    # A v2 file has none of the new keys; it still opens, with the defaults.
    old = plot_config_to_json(PlotConfig())
    old["version"] = 2
    del old["heatmap"]
    del old["histogram"]["kde_overlay"]
    del old["style"]["heatmap_colormap"]
    back = plot_config_from_json(old)
    assert back.histogram.kde_overlay is False
    assert back.heatmap == HeatmapConfig()
    assert back.style.heatmap_colormap == "viridis"


def test_export_plot_data_carries_the_kde_as_its_own_rows(two):
    """``kde`` rows, named for the series whose bars they smooth, in the bars' y
    quantity -- and never taking the place (or the name) of a fitted curve."""
    two.set_normalization("density")
    two.set_kde_overlay(True)
    two.set_fit("exponential")
    data = two._plot_data()
    kde = data[data["part"] == "kde"]
    assert set(kde["series"].map(lambda s: s.split("  (")[0])) == {"pore A", "pore B"}
    assert len(kde) == 2 * 256
    assert "density" in kde.columns and kde["density"].notna().all()
    fits = data[data["part"] == "fit"]
    assert all("fit" in str(s) for s in fits["series"].unique()), fits["series"].unique()


def _manual(frame: pd.DataFrame) -> PlotSource:
    return PlotSource(key="manual-measurements", doc_key="manual-measurements",
                      label="Manual", df=frame, provenance={}, colour="#0072B2",
                      kind="manual")


def test_the_project_snapshot_keeps_the_overlay_and_project_open_does_not_estimate_it(
        qapp, qtbot, monkeypatch):
    """The setting round-trips through the analytical workspace, and reopening it is a
    reconstruction, not an analysis run -- the 2-D KDE page's rule -- so the curve is
    left for Plot (●) and Details says why."""
    frame = _events(600)
    before = PlotsWindow(Session())
    qtbot.addWidget(before)
    before.set_data([_manual(frame)])
    before.set_plot_type("Histogram")
    before.set_histogram_column("dwell_s")
    before.set_kde_overlay(True)
    assert len(_kde_items(before)) == 1
    artifact = workspace.capture_plots(before)

    after = PlotsWindow(Session())
    qtbot.addWidget(after)
    after.set_data([_manual(frame)])
    calls = []
    monkeypatch.setattr(pp, "kde_curve", lambda *a, **k: calls.append(1) or 1 / 0)
    report = workspace.restore_plots(after, artifact)
    assert report.skipped == (), report.warnings
    assert after.config().histogram.kde_overlay is True
    assert calls == [] and _kde_items(after) == []
    assert after._stale and "●" in after._plot_btn.text()
    assert "NOT recomputed on project open" in after.details_text()
    # ...and what the Figure Library would capture now describes the picture, which
    # has no curve -- not the restored box. A bin-count change does not estimate it.
    assert after.figure_applied_config()["histogram"]["kde_overlay"] is False
    assert "kde" not in set(after._plot_data()["part"])
    after._bins.setValue(after._bins.value() + 3)
    assert calls == [] and _kde_items(after) == []
    assert after.figure_applied_config()["histogram"]["kde_overlay"] is False


# ================================================================================================
# D2 -- the Event-count heatmap
# ================================================================================================

def _heat(w, *, log_x=True, log_y=False, bins=40):
    w.set_plot_type(HEATMAP)
    w.set_heatmap_columns("dwell_s", "depth_pA")
    w.set_heatmap_log(log_x, log_y)
    w.set_heatmap_bins(bins)
    return w._heatmap_image


@pytest.mark.parametrize("log_x,log_y", [(True, False), (False, False), (True, True)])
def test_the_heatmap_is_event_waveforms_binning_bin_for_bin(win, log_x, log_y):
    image = _heat(win, log_x=log_x, log_y=log_y, bins=33)
    df = _events()
    want, xe, ye = event_shape.density_2d(df["dwell_s"], df["depth_pA"], xbins=33,
                                          ybins=33, xlog=log_x, ylog=log_y)
    assert np.array_equal(np.asarray(image.image), want)
    got_xe, got_ye = image._heatmap_edges
    assert np.array_equal(got_xe, xe) and np.array_equal(got_ye, ye)
    # Placed in log10 view coordinates on a logged axis, so the decade ticks are right.
    rect = image.mapRectToParent(image.boundingRect())
    x0 = np.log10(xe[0]) if log_x else xe[0]
    y0 = np.log10(ye[0]) if log_y else ye[0]
    assert rect.left() == pytest.approx(x0) and rect.top() == pytest.approx(y0)
    assert win.plot_item().getAxis("bottom").logMode is log_x


def test_the_heatmap_shows_the_first_ticked_source_and_says_so(two):
    image = _heat(two)
    assert int(np.asarray(image.image).sum()) == 4000      # pore A only, not A + B
    details = two.details_text()
    assert "first ticked source only" in details and "1 other ticked source" in details
    assert len(two.figure_source_recipes()) == 1
    assert two._presentation_sources[0][0] == "doc0:0"
    assert len(two._presentation_sources) == 1
    assert "counts, not a KDE density" in details


def test_the_colour_bar_is_inside_the_plot_and_leaves_with_the_heatmap(win):
    plot = win.plot_item()
    _heat(win)
    bar = win._heatmap_bar
    assert bar is not None and bar.parentItem() is plot
    assert bar.getAxis("left").labelText == "Event count"
    win.set_plot_type("Scatter")
    assert win._heatmap_bar is None and bar.parentItem() is None
    assert plot.layout.columnCount() == 3


def test_a_colormap_change_recolours_without_rebinning_or_moving_the_zoom(win, monkeypatch):
    image = _heat(win)
    win.plot_item().vb.setRange(xRange=(-3.5, -2.5), yRange=(70.0, 100.0), padding=0)
    before = [list(r) for r in win.plot_item().vb.viewRange()]
    calls = []
    real = event_shape.density_2d
    monkeypatch.setattr(event_shape, "density_2d",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    win.set_heatmap_colormap("magma")
    assert calls == [], "a colour-map change must not re-bin"
    assert win._heatmap_image is image
    assert win.config().style.heatmap_colormap == "magma"
    assert win._heatmap_bar._colorMap.name == "magma"
    after = [list(r) for r in win.plot_item().vb.viewRange()]
    assert np.allclose(before, after)
    assert win.undo_config()
    assert win.config().style.heatmap_colormap == "viridis"


def test_the_colour_bar_has_no_colormap_menu_of_its_own(win, monkeypatch):
    """pyqtgraph's right-click colormap menu would recolour the picture behind the
    style's back (the combo, undo, templates and the project would all disagree with
    it); the Style group's combo is the one door."""
    _heat(win)
    popped, accepted = [], []
    monkeypatch.setattr(pg.ColorMapMenu, "popup", lambda *_a, **_k: popped.append(1))

    class _RightClick:
        def button(self):
            return Qt.MouseButton.RightButton

        def screenPos(self):
            return QPointF(0.0, 0.0)

        def accept(self):
            accepted.append(1)

    win._heatmap_bar.mouseClickEvent(_RightClick())
    assert popped == [] and accepted == []
    assert win._heatmap_bar._colorMap.name == win.config().style.heatmap_colormap


def test_the_heatmap_carries_the_amplitude_verdict_the_scatter_gives(qapp, qtbot):
    """Scientific warnings stay in Details whatever the page: a count image of an
    un-audited ``amplitude_pA`` with missing selections carries the SAME pooling /
    amplitude verdict the scatter and KDE pages give for those columns, judged over the
    one drawn source -- and keeps its headline beside the title, not in it."""
    frame = _events(800)
    amp = frame["depth_pA"].to_numpy().copy()
    amp[:100] = np.nan
    frame["amplitude_pA"] = amp
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([PlotSource(key="doc0:0", label="pore A", df=frame, colour="#0072B2",
                           provenance={"fs": 250_000.0}, doc_id="doc0", channel=0,
                           module_id="detect.threshold")])
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "amplitude_pA")
    src = w.selected_sources()[0]
    verdict = pp.source_comparability([src], ("dwell_s", "amplitude_pA"),
                                      frames=[w._frame_of(src)])
    assert verdict.warnings, "the fixture must earn a warning for the test to mean anything"
    w.set_plot_type(HEATMAP)
    w.set_heatmap_columns("dwell_s", "amplitude_pA")
    details = w.details_text()
    for line in verdict.warnings:
        assert line in details, (line, details)
    assert verdict.headline and verdict.headline in (w._title_presentation[2] or "")
    assert verdict.headline not in w.plot_item().titleLabel.text


def test_an_imported_colormap_this_build_lacks_is_replaced_and_disclosed(win, tmp_path):
    """A sidecar naming ``jet`` is drawn in viridis -- so the window SAYS viridis
    (config, combo, re-export) and names the refused value, as it does for an
    out-of-range bin count, instead of carrying a map nobody can see."""
    import json
    _heat(win)
    win.set_heatmap_colormap("magma")
    payload = plot_config_to_json(win.config())
    payload["style"]["heatmap_colormap"] = "jet"
    path = tmp_path / "jet.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert win.import_config_json(str(path))
    assert win.config().style.heatmap_colormap == "viridis"
    assert win._heatmap_cmap.currentText() == "viridis"
    assert win._heatmap_bar._colorMap.name == "viridis"
    assert "style.heatmap_colormap “jet”" in win._config_losses.note()
    assert plot_config_to_json(win.config())["style"]["heatmap_colormap"] == "viridis"


def test_heatmap_cells_export_with_their_edges_and_counts(win):
    _heat(win, bins=12)
    data = win._plot_data()
    cells = data[data["part"] == "cell"]
    assert len(cells) == 12 * 12
    assert int(cells["count"].sum()) == 4000
    assert {"bin_left", "bin_right", "y_bin_low", "y_bin_high", "dwell_s",
            "depth_pA"} <= set(cells.columns)
    assert (cells["bin_right"] > cells["bin_left"]).all()
    assert (cells["y_bin_high"] > cells["y_bin_low"]).all()


def test_the_heatmap_has_no_lasso_and_its_config_is_its_own(win):
    _heat(win, log_x=False, bins=21)
    assert not win._lasso_action.isEnabled()
    cfg = win.config()
    assert cfg.plot_type == HEATMAP
    assert cfg.heatmap == HeatmapConfig(x="dwell_s", y="depth_pA", log_x=False,
                                        log_y=False, bins=21)
    assert cfg.identity.x_column == "dwell_s" and cfg.identity.y_column == "depth_pA"
    assert cfg.log_state == (False, False)
    # The one writer and the reader round-trip.
    win._set_config(cfg, draw=False)
    assert win.config() == cfg


def test_the_exported_heatmap_figure_carries_its_colour_bar(win, tmp_path):
    _heat(win)
    out = win.save_image(str(tmp_path / "heat.svg"))
    assert out is not None
    assert "Event count" in out.read_text(encoding="utf-8")


def test_the_hot_cell_is_drawn_where_its_events_are(qapp, qtbot):
    """ORIENTATION -- the Event waveforms caveat in the guide's §14: counts[x, y] must be
    drawn at (x, y), not transposed. 90% of the events sit at short dwell and DEEP depth,
    so the brightest cell has to be drawn there."""
    rng = np.random.default_rng(9)
    n_hot, n_rest = 900, 100
    frame = pd.DataFrame({
        "dwell_s": np.r_[10.0 ** rng.normal(-4.0, 0.02, n_hot),
                         10.0 ** rng.uniform(-4.0, -2.0, n_rest)],
        "depth_pA": np.r_[rng.normal(150.0, 1.0, n_hot), rng.uniform(20.0, 150.0, n_rest)],
        "accepted": np.ones(n_hot + n_rest, dtype=bool),
    })
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([PlotSource(key="doc0:0", label="corner", df=frame, colour="#0072B2",
                           doc_id="doc0", channel=0, module_id="detect.threshold")])
    image = _heat(w, log_x=True, bins=20)
    counts = np.asarray(image.image)
    i, j = np.unravel_index(int(np.argmax(counts)), counts.shape)
    centre = image.mapToParent(QPointF(i + 0.5, j + 0.5))
    assert centre.x() == pytest.approx(-4.0, abs=0.2)      # log10 of 1e-4 s
    assert centre.y() == pytest.approx(150.0, abs=10.0)


def test_the_heatmap_is_restored_from_a_project_snapshot(qapp, qtbot):
    """Binning is drawing, not estimating (the histogram's bars are re-binned on open the
    same way), so a restored heatmap is painted at once from the saved recipe."""
    frame = _events(700)
    before = PlotsWindow(Session())
    qtbot.addWidget(before)
    before.set_data([_manual(frame)])
    _heat(before, log_x=False, bins=15)
    before.set_heatmap_colormap("cividis")
    artifact = workspace.capture_plots(before)

    after = PlotsWindow(Session())
    qtbot.addWidget(after)
    after.set_data([_manual(frame)])
    report = workspace.restore_plots(after, artifact)
    assert report.skipped == (), report.warnings
    assert after.config().heatmap == before.config().heatmap
    assert after.config().style.heatmap_colormap == "cividis"
    assert after._heatmap_image is not None
    assert np.array_equal(np.asarray(after._heatmap_image.image),
                          np.asarray(before._heatmap_image.image))


# ================================================================================================
# the heatmap under plot_interaction (Reset view, zoom history, pointer readout)
# ================================================================================================

def _readout_at(w, vx: float, vy: float) -> str:
    from nanoscope.gui import plot_interaction as pi
    hover = w.plot_item().scene()._nanoscope_hover
    options = next(o for plot, _ro, o in hover._entries if plot is w.plot_item())
    return pi.readout_text(w.plot_item(), QPointF(vx, vy), **options)


def test_the_heatmap_readout_reads_real_values_and_the_count_under_the_pointer(win):
    """Log x is in log10 view coordinates, so the readout gives the real dwell, in the Time
    unit the ticks show, and adds the count of the cell under the pointer."""
    image = _heat(win, log_x=True, bins=20)
    xe, ye = image._heatmap_edges
    i, j = 3, 5
    x = float(np.sqrt(xe[i] * xe[i + 1]))
    vy = 0.5 * (ye[j] + ye[j + 1])
    text = _readout_at(win, float(np.log10(x)), vy)
    factor, unit = pp._TIME_UNITS[win._time_unit.currentText()]
    assert f"dwell ({unit}" in text and f"{x * factor:.5g}" in text
    assert f"Event count {np.asarray(image.image)[i, j]:.0f}" in text
    # Off the image there is no cell to count.
    assert "Event count" not in _readout_at(win, float(np.log10(xe[0])) - 1.0, vy)
    # Another plot type drops the image, and its count with it.
    win.set_plot_type("Scatter")
    assert "Event count" not in _readout_at(win, float(np.log10(x)), vy)


def test_reset_view_frames_the_heatmap_as_drawn_and_keeps_history_on_restyle(win, qtbot):
    from nanoscope.gui import plot_interaction as pi
    _heat(win, log_x=True, bins=20)
    vb = win.plot_item().vb
    drawn = np.asarray(vb.viewRange())
    nav = pi.navigation_for(win.plot_item())
    vb.setRange(xRange=(drawn[0][0] + 0.3, drawn[0][1] - 0.3), padding=0)
    vb.sigRangeChangedManually.emit([True, True])
    qtbot.wait(pi.SETTLE_MS + 100)
    assert nav.can_go_back()
    nav.reset_view()
    np.testing.assert_allclose(vb.viewRange(), drawn)      # the drawn frame, not a padded fit
    kept = nav.history()
    win.set_heatmap_colormap("magma")                      # appearance only
    qtbot.wait(pi.SETTLE_MS + 100)
    assert nav.history() == kept
    win.set_heatmap_bins(25)                               # new geometry
    assert len(nav.history()) == 1
