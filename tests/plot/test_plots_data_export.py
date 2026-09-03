"""B2 — *Export plot data…*: the numbers behind the interactive plot, as CSV.

``save_figure`` has written a source-data CSV beside every publication figure since it
was written ("Researchers always need to restyle a figure later, usually in another
tool, usually the week of submission"). No interactive panel wrote one — and the
interactive panel is where the user decides what the figure IS. A PNG of a histogram
whose bin edges exist only inside a running Qt process is a picture nobody can re-plot,
re-bin, or check.

THE STANDARD THESE TESTS HOLD THE FILE TO is the one the assignment sets: *a CSV that
cannot reproduce the figure is not worth writing*. So each plot type is checked against
the source events, not merely against itself:

* histogram — the exported bins re-derive from ``np.histogram`` on the accepted rows at
  the exported edges, and the counts add up to the drawn n;
* scatter — the exported points ARE the plotted rows, in stored units, with the log the
  panel applied by hand undone (the trap: a ScatterPlotItem holds display coordinates,
  because ``setLogMode`` cannot reach one);
* KDE — the density grid the contours were cut from, in data coordinates, plus the
  levels, so it re-contours anywhere;
* PCA — the scores AND the loadings, without which the axes name nothing.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui import plots_panel
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.plot import figures


# ================================================================================================
# builders
# ================================================================================================
def _events(n: int = 240, *, seed: int = 0, n_rejected: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n // 2
    accepted = np.ones(n, dtype=bool)
    accepted[-n_rejected:] = False
    return pd.DataFrame({
        "dwell_s": np.abs(np.r_[rng.normal(1e-4, 1.5e-5, half),
                                rng.normal(6e-4, 6e-5, n - half)]),
        "amplitude": np.abs(np.r_[rng.normal(20.0, 2.0, half),
                                  rng.normal(48.0, 3.0, n - half)]),
        "frac_depth": np.abs(rng.normal(0.25, 0.05, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "accepted": accepted,
        "filter_limited": np.zeros(n, dtype=bool),
    })


@pytest.fixture
def window(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"apo −700 mV": _events()})
    return w


def _part(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["part"] == name].reset_index(drop=True)


# ================================================================================================
# histogram
# ================================================================================================
def test_the_histogram_csv_reproduces_the_bars_from_the_source_events(window):
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_bins(15)
    window.set_fit("none")

    df = window._plot_data()
    bins = _part(df, "bins")
    assert len(bins) == 15
    assert list(df.columns) == ["series", "part", "bin_left", "bin_right",
                                "dwell_s", "count"]

    # The exported edges, re-binned over the ACCEPTED source rows, must give the
    # exported heights back. This is the whole claim of the file in one assertion:
    # the CSV is enough to redraw the figure.
    src = _events()
    dwell = src.loc[src["accepted"], "dwell_s"].to_numpy()
    edges = np.r_[bins["bin_left"].to_numpy(), bins["bin_right"].to_numpy()[-1]]
    expected, _ = np.histogram(dwell, bins=edges)
    assert np.array_equal(bins["count"].to_numpy().astype(int), expected)
    assert int(bins["count"].sum()) == int(len(dwell))

    # ...and the centre column is the centre, not a third independent number.
    assert bins["dwell_s"].to_numpy() == pytest.approx(
        0.5 * (bins["bin_left"] + bins["bin_right"]).to_numpy())


def test_the_height_column_is_named_for_the_quantity_it_holds(window):
    """A count and a density are different numbers off the same bars, so the column says which."""
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_normalization("count")
    assert "count" in window._plot_data().columns

    window.set_normalization("density")
    df = window._plot_data()
    assert "density" in df.columns and "count" not in df.columns
    # ...and exactly ONE column of that name: ``density`` is also the KDE's third
    # column, and two columns of one name in a CSV is silent data loss.
    assert list(df.columns).count("density") == 1
    # The drawn density integrates to 1 over the bars.
    bins = _part(df, "bins")
    widths = (bins["bin_right"] - bins["bin_left"]).to_numpy()
    assert float((bins["density"].to_numpy() * widths).sum()) == pytest.approx(1.0, rel=1e-6)

    # THE SPLIT S6-03 INTRODUCED, PINNED HERE (this is the test that fails if somebody
    # later "simplifies" ``_y_axis_label`` back into ``_y_label``). The PICTURE's y axis now
    # reads ``density (1/s)``, because its x TICKS are re-scaled for display and the heights
    # are not — but this CSV must NOT be renamed to match. A table has no display scale, and
    # its x column is the schema's own unit-bearing name, so a column called ``density``
    # beside ``dwell_s`` is per-second by construction. Renaming it would break a
    # user-facing column for no gain and disturb ``plot_source_data``'s
    # ``density_name in (x_name, y_name)`` guard, whose own comment names this case.
    for unit, expected_axis in (("s", "density (1/s)"), ("ms", "density (1/s)"),
                                ("µs", "density (1/s)")):
        window.set_time_unit(unit)
        assert window._plot.getAxis("left").labelText == expected_axis
        df = window._plot_data()
        assert list(df.columns).count("density") == 1
        assert not any(c.startswith("density (") for c in df.columns)
        bins = _part(df, "bins")
        widths = (bins["bin_right"] - bins["bin_left"]).to_numpy()
        # ...and the numbers themselves are the SAME numbers in every unit: the export is
        # in stored units, always, because nothing about the display reached the data.
        assert float((bins["density"].to_numpy() * widths).sum()) == pytest.approx(
            1.0, rel=1e-6)


def test_a_fitted_curve_travels_with_the_bars_it_was_drawn_over(window):
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("exponential")

    df = window._plot_data()
    fit = _part(df, "fit")
    assert len(fit) > 100                      # the drawn grid, not just the bin centres
    assert np.isfinite(fit["count"]).all()
    # The fit rows carry no bin edges — they are a curve, not bars.
    assert fit["bin_left"].isna().all()
    # The series name is the fit's own legend entry, so a reader can tell WHICH fit.
    assert "fit" in fit["series"].iloc[0]


def test_two_overlaid_sources_are_two_named_series_on_shared_edges(window):
    window.set_data({"A": _events(seed=1), "B": _events(120, seed=2)})
    window.set_sources(["A", "B"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")

    bins = _part(window._plot_data(), "bins")
    names = list(dict.fromkeys(bins["series"]))
    assert len(names) == 2
    assert names[0].startswith("A") and names[1].startswith("B")
    # Shared bin edges are what makes an overlay a comparison; the CSV must show that.
    a = bins[bins["series"] == names[0]]["bin_left"].to_numpy()
    b = bins[bins["series"] == names[1]]["bin_left"].to_numpy()
    assert np.array_equal(a, b)


# ================================================================================================
# scatter
# ================================================================================================
def test_the_scatter_csv_is_the_plotted_rows_in_stored_units(window):
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")

    df = window._plot_data()
    pts = _part(df, "points")
    assert list(df.columns) == ["series", "part", "dwell_s", "amplitude"]

    src = _events()
    kept = src[src["accepted"]]
    assert len(pts) == len(kept)
    assert sorted(pts["dwell_s"]) == pytest.approx(sorted(kept["dwell_s"]))
    assert sorted(pts["amplitude"]) == pytest.approx(sorted(kept["amplitude"]))


def test_a_log_scatter_exports_seconds_and_not_log_seconds(window):
    """The trap this file exists to pin.

    ``_draw_scatter`` log10s its coordinates BY HAND (``setLogMode`` transforms
    PlotDataItem children and leaves a ScatterPlotItem alone), so the item literally
    holds ``-3.7`` for a 0.2 ms event. Exporting that under a column called ``dwell_s``
    would be a file that reads as a measurement and is not one.
    """
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    window.set_scatter_log(True, True)

    pts = _part(window._plot_data(), "points")
    src = _events()
    kept = src[src["accepted"]]
    assert sorted(pts["dwell_s"]) == pytest.approx(sorted(kept["dwell_s"]))
    assert float(pts["dwell_s"].max()) < 1e-2     # seconds, not log10 seconds


def test_the_scatter_fit_line_is_exported_too(window):
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    window.set_scatter_fit("linear")

    fit = _part(window._plot_data(), "fit")
    assert len(fit) > 2
    assert np.isfinite(fit["amplitude"]).all()


# ================================================================================================
# KDE
# ================================================================================================
def test_the_kde_csv_carries_the_grid_and_the_levels(window):
    window.set_plot_type("KDE (density)")
    window.set_kde_columns("dwell_s", "amplitude")
    window.set_kde_log(True, False)

    df = window._plot_data()
    grid, levels = _part(df, "grid"), _part(df, "level")
    n = PlotsWindow._KDE_GRID
    assert len(grid) == n * n
    assert len(levels) == window._kde_levels.value()
    assert "density" in df.columns
    assert (grid["density"] > 0).all()

    # Data coordinates, not log10 and not grid indices: the grid must SPAN the events
    # it was estimated from, in seconds.
    src = _events()
    dwell = src.loc[src["accepted"], "dwell_s"]
    assert grid["dwell_s"].min() < dwell.min()
    assert grid["dwell_s"].max() > dwell.max()
    # ...and every contour level is a density inside the grid's own range, which is
    # what makes "re-contour this anywhere" true.
    assert levels["density"].min() > 0
    assert levels["density"].max() <= grid["density"].max()
    # A level row is a level, not a point: it names no coordinate.
    assert levels["dwell_s"].isna().all()

    # S6-03: THE LEVELS' OWN DENOMINATOR IS NOW STATED, IN PROSE ONLY. A contour level IS
    # a density, and a density with no unit is a number nobody can check — and this page
    # has the sharper version of the histogram's problem: with the log toggle on,
    # ``gaussian_kde`` is fitted to log10(dwell_s), so the denominator is a DECADE of it,
    # while ``plot_source_data`` un-logs the exported grid's x back to linear. The sentence
    # says so. NOTHING NUMERIC MOVED — measured, the exported CSV is byte-identical before
    # and after, which is why the whole assertion block above is re-run rather than relaxed.
    detail = window.details_text()
    assert "decade of dwell_s" in detail, "the log toggle changes the denominator"
    assert "amplitude" in detail and "decade of amplitude" not in detail, (
        "y is linear here, so only x is per-decade")
    assert "density" in df.columns and list(df.columns).count("density") == 1


def test_two_kde_sources_keep_their_own_grids(window):
    window.set_data({"A": _events(seed=1), "B": _events(seed=2)})
    window.set_sources(["A", "B"])
    window.set_plot_type("KDE (density)")
    window.set_kde_columns("dwell_s", "amplitude")

    grid = _part(window._plot_data(), "grid")
    counts = grid.groupby("series").size()
    n = PlotsWindow._KDE_GRID
    assert sorted(counts.tolist()) == [n * n, n * n]
    assert set(counts.index) == {"A", "B"}


# ================================================================================================
# PCA
# ================================================================================================
def test_the_pca_csv_carries_the_scores_and_the_loadings(window):
    window.set_feature_selection(["dwell_s", "amplitude", "frac_depth", "snr"])
    res = window.run_pca()
    assert res is not None

    df = window._plot_data()
    assert list(df.columns) == ["series", "part", "PC1", "PC2"]
    assert len(_part(df, "points")) == res.n_kept
    loadings = _part(df, "loading")
    # Without these the axes are unnamed combinations of columns and the file cannot
    # be interpreted at all.
    assert list(loadings["series"]) == list(res.columns)
    assert loadings["PC1"].to_numpy() == pytest.approx(res.components[0])
    assert loadings["PC2"].to_numpy() == pytest.approx(res.components[1])


# ================================================================================================
# the series names
# ================================================================================================
def test_the_series_column_carries_the_full_name_not_the_elided_legend_text(window):
    """A legend entry is elided at 34 characters; a data file must not be.

    ``_legend_offered`` records the offer BEFORE the eliding and before the *Show
    legend* gate, which is why the export reads from there and not from the legend.
    """
    long_name = "500 pM Apo ferritin_1 x PBS_22 nm_0008_process data_-500 mV.csv"
    window.set_data({long_name: _events(seed=7), "B": _events(80, seed=8)})
    window.set_sources([long_name, "B"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")

    series = list(_part(window._plot_data(), "bins")["series"])
    assert any(s.startswith(long_name) for s in series)
    assert not any("…" in s for s in series)


def test_switching_the_legend_off_does_not_anonymise_the_data(window):
    window.set_data({"A": _events(seed=1), "B": _events(seed=2)})
    window.set_sources(["A", "B"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window._on_style_show_legend(False)      # the *Show legend* tickbox's own slot

    series = set(_part(window._plot_data(), "bins")["series"])
    assert {s.split(" ")[0] for s in series} == {"A", "B"}


# ================================================================================================
# the writer
# ================================================================================================
def test_plot_data_paths_applies_the_csv_suffix():
    assert figures.plot_data_paths("/tmp/thing") == (pathlib.Path("/tmp/thing.csv"),)
    assert figures.plot_data_paths("/tmp/thing.txt")[0].suffix == ".csv"


def test_export_data_writes_a_csv_that_reads_back_identically(window, tmp_path):
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    expected = window._plot_data()

    out = window.export_data(str(tmp_path / "points.csv"))
    assert out == tmp_path / "points.csv"
    # A plain read_csv, with no ``comment=`` and no ``skiprows``: the provenance is in
    # the column NAMES and in the ``part`` column, never in a header the reader has to
    # be told about.
    back = pd.read_csv(out)
    assert list(back.columns) == list(expected.columns)
    assert len(back) == len(expected)
    assert back["dwell_s"].to_numpy() == pytest.approx(expected["dwell_s"].to_numpy())


def test_a_name_without_an_extension_still_lands_where_the_prompt_said(window, tmp_path):
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    out = window.export_data(str(tmp_path / "bare"))
    assert out == tmp_path / "bare.csv" and out.is_file()


def test_an_empty_plot_writes_nothing_and_says_so(window, tmp_path, monkeypatch):
    told: list[str] = []
    monkeypatch.setattr(plots_panel.QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a: told.append(a[-1])))
    window.set_data({})                       # nothing drawn at all
    target = tmp_path / "empty.csv"
    assert window.export_data(str(target)) is None
    assert not target.exists()
    assert told and "nothing plotted" in told[0]


def test_save_plot_data_refuses_an_empty_frame(tmp_path):
    """A header with no numbers under it is indistinguishable, later, from a real result."""
    with pytest.raises(ValueError):
        figures.save_plot_data(tmp_path / "x.csv", pd.DataFrame())
    assert not (tmp_path / "x.csv").exists()


def test_a_failed_csv_write_leaves_the_previous_export_whole(window, tmp_path,
                                                             monkeypatch):
    """USER_GUIDE §12 again: ``to_csv`` opens and truncates the name it is handed."""
    target = tmp_path / "prev.csv"
    target.write_text("yesterday,export\n1,2\n")
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")

    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    shown: list[str] = []
    monkeypatch.setattr(plots_panel.QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a: shown.append(a[-1])))

    assert window.export_data(str(target)) is None
    assert target.read_text() == "yesterday,export\n1,2\n"
    assert shown and "disk full" in shown[0]


def test_the_export_records_itself_in_details(window, tmp_path):
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    out = window.export_data(str(tmp_path / "d.csv"))
    assert f"Saved plot data → {out.name}" in window.details_text()


# ================================================================================================
# the colour channel is data too
# ================================================================================================
def _clustered(window):
    window.set_feature_selection(["dwell_s", "amplitude"])
    window.set_cluster_k(2)
    res = window.run_clustering()
    assert res is not None
    return res


def test_a_cluster_coloured_scatter_names_the_source_and_carries_the_cluster(window):
    """MEASURED BEFORE: 200 points of a two-cluster scatter all labelled "cluster 0 (n = 100)".

    A cluster entry in ``_legend_offered`` is a drawn SERIES on the split histogram and
    a legend SWATCH on the scatter and PCA pages, where one ScatterPlotItem carries a
    brush per point. Reading the swatches as series names both mislabelled every row
    and lost the partition the picture is about.
    """
    res = _clustered(window)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    window.set_colour_by("Cluster")

    pts = _part(window._plot_data(), "points")
    assert set(pts["series"]) == {"apo −700 mV"}          # the SOURCE, not a cluster
    assert "cluster" in pts.columns
    assert len(pts) == len(res.labels)
    # The colour channel, recovered from the brushes the draw set, agrees row for row
    # with the clustering it came from.
    assert np.array_equal(pts["cluster"].to_numpy(dtype=int),
                          res.labels.to_numpy(dtype=int))


def test_a_cluster_coloured_pca_carries_the_cluster_and_leaves_the_loadings_blank(window):
    _clustered(window)
    window.set_feature_selection(["dwell_s", "amplitude", "frac_depth", "snr"])
    res = window.run_pca()

    df = window._plot_data()
    pts, loadings = _part(df, "points"), _part(df, "loading")
    assert len(pts) == res.n_kept
    assert pts["cluster"].notna().all()
    # A loading is a property of a FEATURE, not of a point, so it belongs to no
    # cluster — <NA>, never a 0 that would read as "cluster 0".
    assert loadings["cluster"].isna().all()


def test_a_split_histogram_still_names_its_clusters_as_series(window):
    """The other side of the same rule: there the clusters really ARE the drawn series."""
    _clustered(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_histogram_colour_by("Cluster")

    bins = _part(window._plot_data(), "bins")
    names = {s.split(" ")[0] + " " + s.split(" ")[1] for s in bins["series"]}
    assert names == {"cluster 0", "cluster 1"}
    # ...and no per-point cluster column: a bar belongs to one cluster by being ITS bar.
    assert "cluster" not in bins.columns


def test_a_stale_clustering_does_not_stamp_an_uncoloured_scatter(window):
    """The colour palettes overlap, so the gate may not be "a clustering exists".

    A clustering stays on the window after *Colour by* goes back to None or a second
    source is ticked, and both repaint the markers from the SOURCE palette — which is
    Okabe-Ito, the same one ``category_colour`` draws from. A colour lookup would match
    and stamp every row with a cluster the picture is not showing.
    """
    _clustered(window)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    window.set_colour_by("Cluster")
    assert "cluster" in window._plot_data().columns          # it really is on

    window.set_colour_by("None")
    assert "cluster" not in window._plot_data().columns

    window.set_data({"A": _events(seed=1), "B": _events(seed=2)})
    window.set_sources(["A", "B"])
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    assert "cluster" not in window._plot_data().columns
