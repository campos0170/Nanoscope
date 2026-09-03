"""Tests for the interactive population-plots window (Phase 2, offscreen).

The :class:`~nanoscope.gui.plots_panel.PlotsWindow` is standalone: it is fed
per-event tables via :meth:`set_data` and renders them with pyqtgraph, driving
:mod:`nanoscope.analysis.populations` (clustering/PCA) and
:mod:`nanoscope.analysis.stats` (fits) underneath. These tests exercise that
wiring the way the owner + a user would:

  - set_data populates the numeric-column pickers (and hides boolean flags);
  - Histogram draws bars and, with a fit chosen, overlays the fitted density
    with the parameters printed on the plot;
  - Scatter draws one point per finite (X, Y) row;
  - GMM clustering assigns labels and recolours the scatter into k groups;
  - PCA projects the feature set to a 2-column scatter with explained variance;
  - Save image writes a PNG;
  - empty / too-few-rows / too-few-features states show a friendly message
    instead of crashing.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow


# ================================================================================================
# builders
# ================================================================================================

def _two_blob_events(n: int = 80, *, seed: int = 0, n_rejected: int = 6) -> pd.DataFrame:
    """A per-event frame with two separable blobs in (dwell_s, amplitude).

    Mirrors an ``EventSet.df``: positive dwell/amplitude, an ``accepted`` flag,
    and a couple of extra numeric columns for the feature set. Two blobs so a
    k=2 clustering has something real to find.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    dwell = np.abs(np.r_[rng.normal(1e-4, 1.5e-5, half), rng.normal(6e-4, 6e-5, n - half)])
    amp = np.abs(np.r_[rng.normal(20.0, 2.0, half), rng.normal(48.0, 3.0, n - half)])
    accepted = np.ones(n, dtype=bool)
    accepted[-n_rejected:] = False
    return pd.DataFrame({
        "dwell_s": dwell,
        "amplitude": amp,
        "frac_depth": np.abs(rng.normal(0.25, 0.05, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "rms": np.abs(rng.normal(1.4, 0.3, n)),
        "accepted": accepted,
        "filter_limited": np.zeros(n, dtype=bool),
    })


@pytest.fixture
def window(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    return w


@pytest.fixture
def loaded(window):
    window.set_data({"Detected events": _two_blob_events()})
    return window


def _data_items(win: PlotsWindow) -> list:
    return list(win.plot_item().listDataItems())


# ================================================================================================
# set_data / column plumbing
# ================================================================================================


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
    # * 90, not * 30: the BIC test's rival-family scan was measured at ~12 s cold on the
    # owner's machine (2026-08-23, six interleaved runs, 11.9-12.1 s) against the * 30
    # budget of exactly 12 s -- a zero-margin wall-clock gate, the same disease the
    # latency files were cured of the same day. The loop breaks the moment the window is
    # quiescent, so the extra budget is spent ONLY on a machine that needs it; the gate
    # still fails loudly if the notes genuinely never settle.
    deadline = time.time() + (PlotsWindow._NOTE_SETTLE_MS * 90) / 1000.0
    while time.time() < deadline:
        app.processEvents()
        if not win._pending_notes and not win._note_timer.isActive():
            app.processEvents()
            break
    else:
        raise AssertionError(
            "the panel's deferred notes never settled: still pending "
            f"{sorted(k[0] for k in win._pending_notes)} after "
            f"{PlotsWindow._NOTE_SETTLE_MS * 90 / 1000:.0f} s. A test that returns here "
            "leaves its timer armed and the NEXT test pays for it.")

def test_set_data_populates_numeric_pickers_and_hides_flags(loaded):
    assert loaded.current_source_name() == "Detected events"
    # Numeric columns are offered; boolean flags (accepted, filter_limited) are not.
    hist_cols = [loaded._hist_col.itemText(i) for i in range(loaded._hist_col.count())]
    assert "dwell_s" in hist_cols and "amplitude" in hist_cols
    assert "accepted" not in hist_cols and "filter_limited" not in hist_cols
    # The events source enables the accepted-only filter.
    assert loaded._accepted_check.isEnabled()
    feats = [loaded._features.item(i).text() for i in range(loaded._features.count())]
    assert "accepted" not in feats


# ================================================================================================
# histogram + fit overlay
# ================================================================================================

def test_histogram_draws_bars_and_exponential_fit_overlay(loaded):
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_bins(20)
    loaded.set_fit("none")

    assert loaded._hist_item is not None
    assert isinstance(loaded._hist_item, pg.PlotDataItem)
    # Bars carry all the events (counts sum to n).
    assert float(np.nansum(loaded._hist_item.yData)) > 0
    assert loaded._fit_item is None  # no fit yet

    # Turning the fit on overlays a second curve and prints tau on the plot.
    loaded.set_fit("exponential")
    assert loaded._fit_item is not None
    assert loaded._fit_item in _data_items(loaded)
    assert loaded._hist_item in _data_items(loaded)
    assert "τ" in loaded.plot_item().titleLabel.text
    assert "τ" in loaded.details_text()


def test_histogram_gaussian_mixture_fit_on_amplitude(loaded):
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("amplitude")
    loaded.set_fit("gaussian-mixture")
    loaded.set_fit_components(2)

    assert loaded._fit_item is not None
    assert "μ" in loaded.plot_item().titleLabel.text


def test_histogram_log_x_still_draws(loaded):
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_log_x(True)
    assert loaded._hist_item is not None
    assert loaded.plot_item().getAxis("bottom").logMode is True
    # And back to linear.
    loaded.set_log_x(False)
    assert loaded.plot_item().getAxis("bottom").logMode is False


def test_histogram_log_y_count_axis(loaded):
    """Log-count axis (the missing scale toggle) draws and sets the LEFT axis to log."""
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_log_y(True)
    assert loaded._hist_item is not None
    assert loaded.plot_item().getAxis("left").logMode is True
    loaded.set_log_y(False)
    assert loaded.plot_item().getAxis("left").logMode is False


def test_manual_axis_range_pins_and_survives_redraw(loaded):
    """Typed limits pin the axis (in data units, log-converted internally) and persist across a
    redraw; switching plot type or hitting Auto releases them."""
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    loaded.set_axis_range(ymin=0.0, ymax=100.0)      # linear y
    vb = loaded.plot_item().getViewBox()
    ylo, yhi = vb.viewRange()[1]
    assert ylo == pytest.approx(0.0, abs=1e-6) and yhi == pytest.approx(100.0, abs=1e-6)
    # A redraw (bin/column change is a redraw) must keep the pin.
    loaded.set_scatter_columns("dwell_s", "amplitude")
    ylo2, yhi2 = vb.viewRange()[1]
    assert (ylo2, yhi2) == pytest.approx((0.0, 100.0), abs=1e-6)
    # Switching plot type clears it.
    loaded.set_plot_type("Histogram")
    assert loaded._manual_x is None and loaded._manual_y is None


def test_manual_range_converts_to_log_on_log_axis(loaded):
    """On a log axis, a data-unit range is converted to log10 for the viewbox."""
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_log_x(True)
    loaded.set_axis_range(xmin=1e-4, xmax=1e-2)
    xlo, xhi = loaded.plot_item().getViewBox().viewRange()[0]   # log10 coords
    assert xlo == pytest.approx(-4.0, abs=1e-6) and xhi == pytest.approx(-2.0, abs=1e-6)


# ================================================================================================
# scatter
# ================================================================================================

def test_scatter_draws_one_point_per_row(loaded):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    assert loaded._scatter_item is not None
    n_accepted = int(_two_blob_events()["accepted"].sum())
    assert len(loaded._scatter_item.points()) == n_accepted


# ================================================================================================
# scatter best-fit line (linear + power law)
# ================================================================================================

def _linear_frame(n: int = 40) -> pd.DataFrame:
    """A clean y = 2x + 1 frame (plus a power-law column y = 3x^2)."""
    x = np.linspace(1.0, 10.0, n)
    return pd.DataFrame({"x": x, "lin": 2.0 * x + 1.0, "pow": 3.0 * x ** 2})


def test_scatter_linear_fit_overlays_line_and_reports_params(window):
    window.set_data({"Manual measurements": _linear_frame()})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("x", "lin")
    # No fit yet: no fit line item.
    assert window._scatter_fit_item is None

    window.set_scatter_fit("linear")
    assert window._scatter_fit_item is not None
    assert window._scatter_fit_item in _data_items(window)
    assert window._scatter_item in _data_items(window)   # the points still draw
    # slope 2, intercept 1, perfect fit -- shown in title AND details.
    for text in (window.plot_item().titleLabel.text, window.details_text()):
        assert "y = 2·x + 1" in text
        assert "R² = 1.000" in text


def test_scatter_power_law_fit_recovers_exponent(window):
    window.set_data({"Manual measurements": _linear_frame()})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("x", "pow")
    window.set_scatter_fit("power law")
    assert window._scatter_fit_item is not None
    # y = 3·x^2 recovered: exponent 2, coefficient 3.
    detail = window.details_text()
    assert "x^2" in detail
    assert "y = 3·x^2" in detail
    assert "R² = 1.000" in detail


def test_scatter_fit_none_removes_the_line(window):
    window.set_data({"Manual measurements": _linear_frame()})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("x", "lin")
    window.set_scatter_fit("linear")
    assert window._scatter_fit_item is not None
    window.set_scatter_fit("none")
    assert window._scatter_fit_item is None
    assert window._scatter_item is not None               # points remain


def test_scatter_fit_respects_title_override(window):
    window.set_data({"Manual measurements": _linear_frame()})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("x", "lin")
    window.set_title("My scatter")
    window.set_scatter_fit("linear")
    # The user's title wins on the plot; the fit summary still lands in details.
    assert "My scatter" in window.plot_item().titleLabel.text
    assert "y = 2·x + 1" in window.details_text()


def test_scatter_fit_degenerate_x_shows_message_and_still_draws(window):
    # A constant-x cloud: the fit is undefined, but the points must still draw.
    df = pd.DataFrame({"x": np.full(20, 5.0), "y": np.linspace(1.0, 20.0, 20)})
    window.set_data({"Manual measurements": df})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("x", "y")
    window.set_scatter_fit("linear")
    assert window._scatter_item is not None               # scatter survives
    assert window._scatter_fit_item is None               # no fit line
    assert "Fit unavailable" in window.details_text()


# ================================================================================================
# clustering recolours the scatter
# ================================================================================================

def test_gmm_clustering_assigns_labels_and_recolours_into_k_groups(loaded):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")

    loaded.set_feature_selection(["dwell_s", "amplitude"])
    loaded.set_cluster_method("GMM")
    loaded.set_cluster_k(2)
    res = loaded.run_clustering()

    assert res is not None
    assert res.k == 2 and res.chosen_k == 2
    # Every finite row got a real label (0/1), none dropped.
    assert set(int(v) for v in res.labels.unique()) == {0, 1}
    assert res.n_kept == int(_two_blob_events()["accepted"].sum())

    # The scatter is now coloured by cluster into exactly two distinct brushes.
    assert loaded._colour_by.currentText() == "Cluster"
    colours = {p.brush().color().name() for p in loaded._scatter_item.points()}
    assert len(colours) == 2


def test_kmeans_clustering_also_works(loaded):
    loaded.set_feature_selection(["dwell_s", "amplitude", "snr"])
    loaded.set_cluster_method("k-means")
    loaded.set_cluster_k(3)
    res = loaded.run_clustering()
    assert res is not None
    assert res.method == "kmeans"
    assert res.k == 3


# ================================================================================================
# PCA
# ================================================================================================

def test_pca_projects_to_two_columns_and_shows_variance(loaded):
    loaded.set_feature_selection(["dwell_s", "amplitude", "snr", "rms"])
    res = loaded.run_pca()

    assert res is not None
    assert res.scores.shape[1] == 2
    assert res.scores.shape[0] == int(_two_blob_events()["accepted"].sum())
    # The view switched to PCA and drew a 2-D scatter.
    assert loaded._plot_type() == "PCA"
    assert loaded._scatter_item is not None
    assert len(loaded._scatter_item.points()) == res.scores.shape[0]
    assert "Explained variance" in loaded.details_text()


def test_pca_then_cluster_colours_the_projection(loaded):
    loaded.set_feature_selection(["dwell_s", "amplitude"])
    loaded.set_cluster_method("GMM")
    loaded.set_cluster_k(2)
    loaded.run_clustering()
    loaded.set_feature_selection(["dwell_s", "amplitude", "snr", "rms"])
    loaded.run_pca()

    colours = {p.brush().color().name() for p in loaded._scatter_item.points()}
    assert len(colours) == 2


# ================================================================================================
# export
# ================================================================================================

def test_save_image_writes_png(loaded, tmp_path):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    out = loaded.save_image(str(tmp_path / "plot.png"))
    assert out is not None and out.exists()
    assert out.stat().st_size > 0


# ================================================================================================
# style / plot-design controls
# ================================================================================================

def _hist_bar(win: PlotsWindow):
    win.set_plot_type("Histogram")
    win.set_histogram_column("dwell_s")
    return win._hist_item


def test_custom_title_shows_on_the_plot(loaded):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    title = loaded.plot_item().titleLabel
    # Scatter has no automatic title; a blank override leaves the title hidden.
    assert not title.isVisible()
    loaded.set_title("My figure")
    assert loaded.style().title == "My figure"
    assert title.isVisible() and "My figure" in title.text
    # Clearing the override hides the title again.
    loaded.set_title("")
    assert not title.isVisible()


def test_data_colour_changes_histogram_and_scatter_brushes(loaded):
    green = "#009e73"
    # Histogram fill picks up the data colour (with the fill alpha preserved).
    bar = _hist_bar(loaded)
    loaded.set_data_color(green)
    fill = loaded._hist_item.opts["fillBrush"].color()
    assert fill.name() == green
    assert fill.alpha() == 110
    assert loaded._hist_item.opts["pen"].color().name() == green
    # Plain scatter points pick it up too.
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    pt = loaded._scatter_item.points()[0]
    assert pt.brush().color().name() == green


def test_point_size_changes_scatter_marker_size(loaded):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    loaded.set_point_size(15)
    assert loaded.style().point_size == 15
    assert loaded._scatter_item.points()[0].size() == 15


def test_line_width_changes_fit_and_hist_pens(loaded):
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_fit("exponential")
    loaded.set_line_width(4)
    assert loaded._hist_item.opts["pen"].width() == 4
    # The fit curve draws one px heavier than the bar outline.
    assert loaded._fit_item.opts["pen"].width() == 5


def test_font_size_applies_to_axes(loaded):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    loaded.set_font_size(20)
    assert loaded.style().font_size_pt == 20
    tick_font = loaded.plot_item().getAxis("bottom").style["tickFont"]
    assert tick_font is not None and tick_font.pointSize() == 20


def test_light_background_sets_white_canvas_and_dark_foreground(loaded):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    loaded.set_background("light")
    assert loaded.style().background == "light"
    bg = loaded._plot_widget.backgroundBrush().color()
    assert bg.red() == 255 and bg.green() == 255 and bg.blue() == 255
    # Axis ink flipped to a dark foreground so it reads on white.
    fg = loaded.plot_item().getAxis("bottom").pen().color()
    assert fg.red() < 80 and fg.green() < 80 and fg.blue() < 80


def test_export_size_writes_png_of_that_width(loaded, tmp_path):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    loaded.set_export_size(640, 480)
    out = loaded.save_image(str(tmp_path / "sized.png"))
    assert out is not None and out.exists()
    img = pg.QtGui.QImage(str(out))
    assert img.width() == 640


def test_export_size_both_axes_forces_exact_pixels(loaded, tmp_path):
    """Width AND height pinned must produce EXACTLY that size. pyqtgraph's ImageExporter
    locks the aspect ratio (setting one axis rewrites the other), so a naive assignment
    lets the second value clobber the first -- 800×500 came out 666×500. Both must stick."""
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    loaded.set_export_size(800, 500)
    out = loaded.save_image(str(tmp_path / "both.png"))
    assert out is not None and out.exists()
    img = pg.QtGui.QImage(str(out))
    assert (img.width(), img.height()) == (800, 500)


# ================================================================================================
# empty / invalid states are handled, not crashed
# ================================================================================================

def test_no_data_shows_message(window):
    assert "No data" in window.details_text()


def test_clustering_needs_two_features(loaded):
    loaded.set_feature_selection(["dwell_s"])
    assert loaded.run_clustering() is None
    assert "at least 2 feature" in loaded.details_text()


def test_pca_needs_two_features(loaded):
    loaded.set_feature_selection(["dwell_s"])
    assert loaded.run_pca() is None
    assert "at least 2 feature" in loaded.details_text()


def test_fit_guardrail_surfaced_not_swallowed(window):
    # Fewer than the stats module's 20-event floor: the histogram still draws,
    # but the fit refuses and says why.
    small = pd.DataFrame({"dwell_s": np.linspace(1e-4, 5e-4, 8),
                          "amplitude": np.linspace(10, 40, 8)})
    window.set_data({"Manual measurements": small})
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")
    assert window._hist_item is not None            # histogram survives
    assert window._fit_item is None                 # fit refused
    assert "Fit unavailable" in window.details_text()


def test_accepted_only_filter_changes_row_count(loaded):
    df = _two_blob_events()
    n_all, n_acc = len(df), int(df["accepted"].sum())
    assert n_acc < n_all

    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded._accepted_check.setChecked(True)
    assert f"n = {n_acc}" in loaded.details_text()

    loaded._accepted_check.setChecked(False)
    assert f"n = {n_all}" in loaded.details_text()


# ================================================================================================
# time display unit — an AXIS scale, never a data change
# ================================================================================================

def _axis(win: PlotsWindow, side: str):
    return win.plot_item().getAxis(side)


def test_dwell_axis_defaults_to_milliseconds(loaded):
    """Seconds is the storage unit; ms is the unit a dwell is read in."""
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    assert loaded.time_unit() == "ms"
    assert _axis(loaded, "bottom").scale == pytest.approx(1e3)
    assert _axis(loaded, "bottom").labelText == "dwell (ms)"


@pytest.mark.parametrize("unit,factor", [("s", 1.0), ("ms", 1e3), ("µs", 1e6)])
def test_every_time_unit_sets_its_axis_scale(loaded, unit, factor):
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_time_unit(unit)
    assert _axis(loaded, "bottom").scale == pytest.approx(factor)
    assert _axis(loaded, "bottom").labelText == f"dwell ({unit})"


def test_changing_the_time_unit_does_not_move_the_fit(loaded):
    """THE reason this is an axis scale and not a data transform.

    The exponential dwell fit takes its detection floor from the detector's own
    ``min_dwell_s`` in SECONDS (see this module's header). Rescaling the sample without
    rescaling that floor would bias tau silently — the exact failure the header warns
    about. An axis scale cannot, because the fit never sees it: the FITTED tau, in
    seconds, must be bit-identical under every unit.

    What the unit is allowed to change is how that one number is PRINTED. It used to be
    printed as a bare "τ = 0.0002716" beside an axis reading ms — a units mismatch on
    the two numbers a reader compares first (report Low-28). So the printed value now
    follows the axis, and this test pins both halves: the fit is invariant, the caption
    is not, and the caption always describes the same physical duration.
    """
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_fit("exponential")

    taus, printed = {}, {}
    for unit in ("s", "ms", "µs"):
        loaded.set_time_unit(unit)
        text = loaded.details_text()
        line = [ln for ln in text.splitlines() if "τ" in ln]
        # Guard against passing vacuously: there must BE a fit for invariance to mean
        # anything. Three missing lines are also "identical".
        assert line, f"no fit reported at {unit}"
        taus[unit] = float(loaded.fit_result().params["tau_1"])
        printed[unit] = line[0]

    assert taus["s"] == taus["ms"] == taus["µs"], (
        f"the FIT moved with a DISPLAY unit: {taus}")

    # The caption follows the axis, and names the unit it is in.
    assert "τ = 0.0002716 s" in printed["s"], printed["s"]
    assert "τ = 0.2716 ms" in printed["ms"], printed["ms"]
    assert "τ = 271.6 µs" in printed["µs"], printed["µs"]


def test_a_non_time_column_resets_the_scale(loaded):
    """A stale x1000 left on a pA axis would be a silent factor-of-1000 lie."""
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    assert _axis(loaded, "bottom").scale == pytest.approx(1e3)
    loaded.set_histogram_column("amplitude")
    assert _axis(loaded, "bottom").scale == pytest.approx(1.0)
    assert _axis(loaded, "bottom").labelText == "amplitude"


@pytest.mark.parametrize("unit,factor", [("s", 1.0), ("ms", 1e3), ("µs", 1e6)])
def test_log_dwell_names_the_unit_its_ticks_are_actually_in(loaded, unit, factor):
    """MEASURED (report High-7): with Time unit = ms the x axis read ``dwell (s, log)``
    AND carried a pyqtgraph ``(x0.001)`` multiplier, so an 18 ms event was drawn at
    x = 10¹ under a label claiming seconds.

    ``AxisItem.setScale`` genuinely cannot be used here — in log mode pyqtgraph
    multiplies the log10 RANGE by it — so the axis stays at scale 1.0. But leaving the
    ticks alone did not leave them in seconds: pyqtgraph's own automatic SI prefix
    supplied a factor of 1000 and appended its own contradicting unit statement. The
    panel now owns the factor (``display_factor``) and the label together.
    """
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_time_unit(unit)
    loaded.set_log_x(True)
    ax = _axis(loaded, "bottom")

    assert ax.scale == pytest.approx(1.0), "setScale on a log axis scales an exponent"
    assert ax.labelText == f"dwell ({unit}, log)"
    assert ax.display_factor == pytest.approx(factor)
    # No SECOND unit statement from pyqtgraph on top of ours.
    assert ax.autoSIPrefix is False
    assert "x0.001" not in ax.labelString() and "(x" not in ax.labelString()


def test_scatter_scales_only_the_time_axis(loaded):
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "frac_depth")
    loaded.set_scatter_log(False, False)
    assert _axis(loaded, "bottom").scale == pytest.approx(1e3)
    assert _axis(loaded, "bottom").labelText == "dwell (ms)"
    assert _axis(loaded, "left").scale == pytest.approx(1.0)
    assert _axis(loaded, "left").labelText == "frac_depth"


# ================================================================================================
# legend width — a legend is an index, not content
# ================================================================================================

def test_a_long_source_label_is_middle_elided(loaded):
    """A pyqtgraph legend is sized by its widest label, and a source label is a RECORDING
    FILENAME — the owner's run past 60 characters, which spanned most of the plot width
    however small the font was. Shrinking text alone could never fix that."""
    long_name = "500 pM Apo ferritin_1 x PBS_22 nm_0008_process data_-500 mV.csv (n = 850)"
    elided = PlotsWindow._elide_legend(long_name)
    assert len(elided) <= PlotsWindow._LEGEND_LABEL_CHARS
    assert "…" in elided
    # The ENDS carry the identity: leading sample, trailing voltage/n.
    assert elided.startswith("500 pM Apo")
    assert elided.endswith("850)")


def test_a_short_legend_label_is_left_alone(loaded):
    assert PlotsWindow._elide_legend("Detected events") == "Detected events"


def test_the_drawn_legend_entry_is_bounded(loaded):
    loaded.set_data({"500 pM Apo ferritin_1 x PBS_22 nm_0008_process data_-500 mV.csv":
                     _two_blob_events()})
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_fit("exponential")
    for label in loaded.legend_labels():
        assert len(label) <= PlotsWindow._LEGEND_LABEL_CHARS + 24, label


def test_the_panel_reports_bic_and_names_a_better_component_count(loaded):
    """The app COMPUTED aic/bic for every fit and never showed them.

    A single exponential is the default. On the owner's -700 mV apo-ferritin run it
    returned tau = 16.8 ms for a distribution with a 0.4 ms spike and an 18 ms mode --
    one authoritative number, no goodness-of-fit beside it, and a curve that visibly
    missed the data. k=2 improved BIC by 78. Nothing said so.

    The fit is NOT changed: the user's choice stands. The panel just stops keeping the
    evidence against that choice to itself.
    """
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    # Two clearly separated exponential populations -> k=2 must win decisively.
    fast = rng.exponential(2e-4, 400)
    slow = rng.exponential(2e-2, 600)
    loaded.set_data({"two-pop": pd.DataFrame({
        "dwell_s": np.r_[fast, slow],
        "amplitude": np.abs(rng.normal(30.0, 3.0, 1000)),
        "accepted": np.ones(1000, dtype=bool),
    })})
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_fit("exponential")

    loaded.set_fit_components(1)
    _settle(loaded)   # the advisory notes are computed off the redraw path
    text = loaded.details_text()
    assert "BIC =" in text and "AIC =" in text, "the fit's own criteria must be shown"
    assert "components = 1" in text
    assert "2 components fit markedly better" in text, text

    # ...and once the user takes the hint, the panel stops giving it.
    loaded.set_fit_components(2)
    text2 = loaded.details_text()
    assert "components = 2" in text2
    assert "markedly better" not in text2, text2


def test_log_scatter_points_are_transformed_like_the_fit_line(loaded):
    """PlotItem.setLogMode transforms PlotDataItem children; a ScatterPlotItem added
    straight to the scene is NOT touched -- the same trap _shade_below_limit documents.

    Untransformed, a dwell of 0.018 s drew at position 0.018 on an axis whose ticks read
    10**x, so every point piled against the right edge while the fitted line (a
    PlotDataItem, therefore transformed) spanned the axis correctly. The picture said the
    fit had missed the data. The fit was right; the points were in the wrong place.
    """
    import numpy as np
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")

    loaded.set_scatter_log(False, False)
    lin = np.concatenate([i.getData()[0] for i in loaded._scatter_items])
    loaded.set_scatter_log(True, False)
    log = np.concatenate([i.getData()[0] for i in loaded._scatter_items])

    assert lin.min() > 0, "linear mode plots the raw values"
    # Log mode must plot log10(value) -- negative here, since the dwells are sub-second.
    np.testing.assert_allclose(np.sort(log), np.sort(np.log10(lin)), rtol=1e-9)
    assert log.max() < 0

    # ...and the y axis, left linear, must NOT be transformed.
    ys = np.concatenate([i.getData()[1] for i in loaded._scatter_items])
    assert ys.min() > 1.0, "a linear y axis must keep raw values"


def test_log_y_scatter_transforms_only_y(loaded):
    import numpy as np
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("dwell_s", "amplitude")
    loaded.set_scatter_log(False, True)
    xs = np.concatenate([i.getData()[0] for i in loaded._scatter_items])
    ys = np.concatenate([i.getData()[1] for i in loaded._scatter_items])
    assert xs.min() > 0 and xs.max() < 1e-2, "x stays raw"
    assert ys.max() < 3.0, "y is log10 of an amplitude ~20-50"


# ================================================================================================
# log axes label DECADES only
# ================================================================================================

# ``qapp`` is REQUIRED, not decoration: these two build a pyqtgraph ``AxisItem``
# directly, and constructing a QGraphicsItem with no QApplication does not raise --
# it takes the interpreter down with "Fatal Python error: Aborted" (seen at
# ``axes.py:90`` -> ``AxisItem.__init__``). They passed only because some earlier
# test in the file had already made the app; run either one FIRST and the process
# died before pytest could report anything.
def test_a_log_axis_labels_every_tick_in_one_notation_and_thins_by_collision(qapp):
    """MEASURED: the dwell scatter's x axis rendered as "0.40.50.60.70.8.91" and
    "3.10¹4.10¹5.10…" — an unreadable smear.

    pyqtgraph's AxisItem.logTickValues appends an extra MINOR tick level whenever a log
    axis spans fewer than ~3 decades, and then labels every one of them; at 2–9 per
    decade those labels overlap.

    THE FIRST FIX BLANKED THEM, AND THIS TEST PINNED THE BLANKING. That removed the
    smear and introduced two defects of its own, both measured on the −600 mV merged
    dwells through the real window: each decade label was painted TWICE at identical
    coordinates (10 text specs at 5 positions — blanking minors does nothing about a
    decade emitted on two tick levels), and an axis zoomed to 15–25 ms, inside one
    decade and astride the median dwell of all three recordings, drew ZERO labels.

    So every tick is labelled and only COLLIDING labels are dropped, decades first —
    which is what the PSD window was already doing, and is now shared code
    (:class:`nanoscope.gui.axes.ReadableLogAxis`). This test therefore pins the two
    properties that survive: one notation, and a minor tick that CAN be labelled.
    """
    from nanoscope.gui.plots_panel import _DecadeLabelAxis

    ax = _DecadeLabelAxis(orientation="bottom")
    values = [-1.0, -0.699, -0.523, -0.398, 0.0, 0.301, 0.477, 1.0]   # decades + 2,3,4,5 within
    out = ax.logTickStrings(values, 1.0, None)

    assert all(s for s in out), f"a tick went unlabelled: {list(zip(values, out))}"
    # The decades, in this axis's decimal notation...
    assert out[0] == "0.1" and out[4] == "1" and out[7] == "10", out
    # ...and the minors as real numbers, not as blanks.
    assert out[1] == "0.2" and out[5] == "2", out
    # ONE notation: no superscript beside a decimal.
    assert not any(c in "⁻⁰¹²³⁴⁵⁶⁷⁸⁹" for s in out for c in s), out


def test_a_log_axis_zoomed_inside_one_decade_still_has_numbers_on_it(qapp):
    """The defect the blanking version could not avoid, at the axis level.

    15–25 ms contains no decade. Under "decades only" every label was empty and the
    drawn axis carried no numbers at all.
    """
    import numpy as np

    from nanoscope.gui.plots_panel import _DecadeLabelAxis

    ax = _DecadeLabelAxis(orientation="bottom")
    values = list(np.log10([0.015, 0.018, 0.020, 0.022, 0.025]))
    out = ax.logTickStrings(values, 1.0, None)
    assert [s for s in out if s], "an axis inside one decade drew no labels"


def test_the_plots_window_uses_that_axis_on_both_log_capable_axes(window):
    """The fix has to be installed, not merely available — the defect was visible in the
    app, so this asserts the app's own plot carries it."""
    from nanoscope.gui.plots_panel import _DecadeLabelAxis

    for side in ("bottom", "left"):
        ax = window._plot.getAxis(side)
        assert isinstance(ax, _DecadeLabelAxis), f"{side} axis is a plain AxisItem"


# ================================================================================================
# THE HINT MUST NOT COST MORE THAN THE FIT IT IS HINTING ABOUT
# ================================================================================================
# `_better_component_count` re-fits k = 1, 2, 3 to compare BIC and discards everything
# else. Left at the default each rival also ran a 1000-replicate bootstrap for confidence
# intervals nobody reads. Measured on the owner's `_0006_Process_-600 mV.abf` (317 events,
# exponential fit) through a real MainWindow: ONE reject click took 3.40 s, of which 3.38 s
# was those rivals; 74 ms after the fix, with every drawn and printed number unchanged.
# It became a live problem when C-1 correctly wired the standalone windows to curation, so
# the cost moved off the user's Refresh click and onto every click he makes.

def _fit_mix_counter(monkeypatch) -> dict:
    from nanoscope.analysis import stats

    calls = {"n": 0}
    real = stats._fit_mix

    def counted(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(stats, "_fit_mix", counted)
    return calls


def test_a_redraw_with_an_exponential_fit_runs_no_rival_bootstrap(loaded, monkeypatch):
    """THE REPRO, counted rather than timed. Before the fix one redraw of a dwell
    histogram with an exponential fit ran 2003 ``_fit_mix`` calls; the point fits it
    actually needs are three."""
    calls = _fit_mix_counter(monkeypatch)
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_fit("exponential")
    calls["n"] = 0
    loaded._redraw()
    assert calls["n"] < 40, (
        f"one redraw ran {calls['n']} likelihood maximisations; the k=1/2/3 point fits "
        "need a handful. The rivals are bootstrapping confidence intervals that "
        "_better_component_count discards")


def test_the_rival_hint_still_names_the_better_component_count(window, monkeypatch):
    """PASSES BOTH — a CONTROL. The hint has to keep WORKING: a cheap hint that stopped
    firing would pass the test above and lose the feature. Two well-separated dwell
    modes, k=1 selected: the panel must still say k=2 fits markedly better."""
    rng = np.random.default_rng(3)
    dwell = np.r_[rng.exponential(3e-4, 300), rng.exponential(1.4e-2, 300)] + 5e-5
    window.set_data({"Detected events": pd.DataFrame({
        "dwell_s": dwell,
        "accepted": np.ones(dwell.size, dtype=bool),
        "filter_limited": np.zeros(dwell.size, dtype=bool),
    })})
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")
    window._redraw()
    _settle(window)   # the advisory notes are computed off the redraw path
    assert "components fit markedly better" in window.details_text(), window.details_text()


def test_the_fit_the_user_reads_still_carries_its_confidence_interval(loaded,
                                                                      monkeypatch):
    """The switch must reach the RIVALS and not the primary fit. A tau with no CI beside
    it is the failure `stats` is otherwise built to avoid, so this pins WHICH fits were
    asked for point estimates: the primary one is not among them."""
    from nanoscope.analysis import stats

    seen: list[bool] = []
    real = stats.ci_wanted
    monkeypatch.setattr(stats, "ci_wanted", lambda: seen.append(real()) or seen[-1])

    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_fit("exponential")
    loaded._redraw()
    # The rival fits are DEFERRED off the redraw path, so they are not among `seen` until
    # the panel settles. Without this the test observes the primary fit alone and its
    # "the rivals still bootstrapped" assertion fails against correct behaviour.
    _settle(loaded)

    assert seen, "no fit asked whether its CIs would be read"
    assert False in seen, "the rivals still bootstrapped"
    assert real() is True, "the switch leaked out of the hint"

    # THE GUARANTEE IS THE CI THE USER ENDS UP READING, NOT WHICH CALL COMPUTED IT.
    # This used to assert `seen[0] is True` — that the PRIMARY fit was never asked for a
    # point estimate — because at the time the only way to keep a tau's interval was to
    # bootstrap it on the click path. That is no longer true, and pinning it there is now
    # actively wrong: the primary bootstrap is what froze the whole window for 2.3 s on
    # the owner's own recording and 45 s at 100k events, so it is deferred like the
    # rivals. The failure this test exists to prevent — "a tau with no CI beside it" —
    # is unchanged and is what is checked here instead, at the surface a person reads:
    # while the interval is pending the pane SAYS so in words, and once it settles the
    # interval is really there. A deferred CI is only honest if both halves hold.
    pending = loaded.details_text()
    assert ("95% CONFIDENCE INTERVALS" in pending or "tau_1 [" in pending), (
        "the pane said nothing about the interval at all, neither a value nor that one "
        f"is still coming:\n{pending}")

    _settle(loaded)
    fit = loaded.fit_result()
    assert getattr(fit, "ci", None) and "tau_1" in fit.ci, (
        "the tau the user reads never got its confidence interval, deferred or "
        f"otherwise: ci={getattr(fit, 'ci', None)!r}")
    assert "tau_1 [" in loaded.details_text(), (
        f"the settled pane still shows no interval line:\n{loaded.details_text()}")

# ================================================================================================
# owner note #14: the fitted parameters as a text box ON the figure
# ================================================================================================

def test_fit_values_box_lands_on_the_plot_when_asked(loaded):
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_bins(20)
    loaded.set_fit("exponential")
    # Off by default — a box appearing unasked is clutter, not presentation.
    assert loaded._fit_values_item is None

    loaded._fit_values_check.setChecked(True)
    assert loaded._fit_values_item is not None, "the checkbox must draw the box"
    assert "τ" in loaded._fit_values_item.text, "the box carries the fitted parameters"
    # Anchored to the ViewBox — it lands in a saved image, unlike Details.
    assert loaded._fit_values_item.parentItem() is loaded.plot_item().vb

    # No fit → no box, even with the option on.
    loaded.set_fit("none")
    assert loaded._fit_values_item is None

    # ...and turning the option back off removes it with the next draw.
    loaded.set_fit("exponential")
    assert loaded._fit_values_item is not None
    loaded._fit_values_check.setChecked(False)
    assert loaded._fit_values_item is None

# ================================================================================================
# owner note #19: KDE is a plot type beside the others, not a separate window
# ================================================================================================

def test_kde_draws_contours_in_data_coordinates(loaded):
    loaded.set_plot_type("KDE (density)")
    loaded.set_kde_columns("dwell_s", "amplitude")
    loaded.set_kde_log(False, False)

    curves = [it for it in loaded.plot_item().items
              if isinstance(it, pg.IsocurveItem)]
    assert len(curves) == loaded._kde_levels.value(), (
        "one source → one contour per level")
    # The transform must land the contours on the DATA ranges, not grid indices —
    # an index-coordinate isocurve over a 128² grid would span 0..127 on both axes.
    df = loaded._sources["events"].df if "events" in loaded._sources else None
    xs = [c.mapRectToParent(c.boundingRect()) for c in curves]
    span_x = max(r.right() for r in xs) - min(r.left() for r in xs)
    assert span_x < 10.0, (
        f"contours span {span_x:.1f} on x — dwell_s is milliseconds-scale, so a span "
        "this wide means the isocurves were left in grid-index coordinates")
    assert "KDE" in loaded.details_text()
    assert "bandwidth factor" in loaded.details_text()

    # Log-x: the same draw in log10 coordinates, still bounded sanely.
    loaded.set_kde_log(True, False)
    curves = [it for it in loaded.plot_item().items if isinstance(it, pg.IsocurveItem)]
    assert curves, "log-x KDE must still draw"


def test_kde_defaults_prefer_the_signature_axes(loaded):
    """dwell on X; frac_depth on Y when the table carries it — the signature
    translocation axes (fractional blockade survives a gain revision)."""
    loaded.set_plot_type("KDE (density)")
    assert loaded._kde_x.currentText() == "dwell_s"
    assert loaded._kde_y.currentText() == "frac_depth"


def test_kde_refuses_honestly_below_five_points(qapp, qtbot):
    from nanoscope.gui.plots_panel import PlotsWindow
    win = PlotsWindow(Session())
    qtbot.addWidget(win)
    df = pd.DataFrame({"dwell_s": [1e-3, 2e-3, 3e-3], "amplitude": [30.0, 31.0, 29.0],
                       "accepted": [True] * 3})
    win.set_data({"tiny": df})
    win.set_plot_type("KDE (density)")
    assert "5 needed" in win.details_text() or "KDE needs at least 5" in win.details_text() \
        or win._message_item is not None

# ================================================================================================
# owner round-2 notes: optional legend, editable axis labels, range boxes in the
# displayed unit
# ================================================================================================

def test_show_legend_unticked_draws_no_legend_entries(loaded):
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_fit("exponential")
    legend = loaded.plot_item().legend
    assert legend is not None and len(legend.items) > 0, "legend on by default"

    loaded._legend_check.setChecked(False)
    legend = loaded.plot_item().legend
    assert legend is None or len(legend.items) == 0, (
        "with Show legend off the redraw must add no entries")

    loaded._legend_check.setChecked(True)
    assert len(loaded.plot_item().legend.items) > 0


def test_axis_label_overrides_replace_text_but_keep_tick_scaling(loaded):
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_time_unit("ms")
    axis = loaded.plot_item().getAxis("bottom")
    assert "ms" in axis.labelText and axis.scale == pytest.approx(1e3)

    loaded._x_label_edit.setText("Dwell time (ms)")
    loaded._y_label_edit.setText("Events per bin")
    assert loaded.plot_item().getAxis("bottom").labelText == "Dwell time (ms)"
    assert loaded.plot_item().getAxis("left").labelText == "Events per bin"
    # TEXT ONLY: the ms tick multiplier still applies under the custom label.
    assert loaded.plot_item().getAxis("bottom").scale == pytest.approx(1e3)

    loaded._x_label_edit.clear()
    assert "dwell" in loaded.plot_item().getAxis("bottom").labelText


def test_axis_range_is_typed_in_the_displayed_unit(loaded):
    """The axis shows ms (the Time-unit default); the range boxes must accept the
    number read off the ticks — not silently interpret it as seconds."""
    loaded.set_plot_type("Histogram")
    loaded.set_histogram_column("dwell_s")
    loaded.set_time_unit("ms")
    loaded._range_edits["xmin"].setText("1")     # 1 ms, as the axis shows
    loaded._range_edits["xmax"].setText("30")    # 30 ms
    loaded._apply_axis_range()
    assert loaded._manual_x == pytest.approx((1e-3, 30e-3)), (
        "typed ms must be stored as seconds (data units)")

    # A non-time axis stays in data units.
    loaded.set_plot_type("Scatter")
    loaded.set_scatter_columns("amplitude", "amplitude")
    loaded._range_edits["ymin"].setText("10")
    loaded._range_edits["ymax"].setText("50")
    loaded._apply_axis_range()
    assert loaded._manual_y == pytest.approx((10.0, 50.0))
